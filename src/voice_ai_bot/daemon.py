from __future__ import annotations

import logging
import signal
import sys
import time
from datetime import datetime
from types import FrameType
from zoneinfo import ZoneInfo

from .audio_io import Recorder
from .config import Config
from .conversation import ConversationStore
from .hardware import HatHardware
from .memory import MemoryStore
from .memory_consolidation import MemoryConsolidator
from .openai_voice import OpenAIVoiceClient
from .realtime_voice import RealtimeConversationSession
from .scheduled_tasks import ScheduledTask, ScheduledTaskStore

LOGGER = logging.getLogger(__name__)


class VoiceDaemon:
    def __init__(self, config: Config):
        self.config = config
        self.hardware = HatHardware(config.button_gpio, config.led_gpio, config.button_pull_up)
        self.recorder = Recorder(config)
        self.conversation = ConversationStore(config.conversation_file)
        self.memory = MemoryStore(config)
        self.memory.ensure_workspace()
        self.memory_consolidator = MemoryConsolidator(config, self.memory)
        self.scheduled_tasks = ScheduledTaskStore(config)
        self.openai = OpenAIVoiceClient(config) if config.voice_bot_backend == "responses" else None
        self.realtime = (
            RealtimeConversationSession(config, self.scheduled_tasks, self.memory)
            if config.voice_bot_backend == "realtime"
            else None
        )
        self.running = True
        self._led_mode = "off"
        if config.voice_bot_backend == "realtime" and config.record_rate != config.realtime_input_rate:
            LOGGER.warning(
                "realtime backend expects RECORD_RATE=%s; current RECORD_RATE=%s",
                config.realtime_input_rate,
                config.record_rate,
            )

    def stop(self) -> None:
        self.running = False
        if self.realtime is not None:
            self.realtime.music.stop()
            self.realtime.close()
        self.memory_consolidator.flush()
        self.hardware.off()

    def run(self) -> None:
        if self.config.voice_bot_backend == "realtime":
            self._run_realtime()
            return
        LOGGER.info("voice daemon ready with %s backend", self.config.voice_bot_backend)
        self.hardware.off()
        while self.running:
            if not self.hardware.wait_for_press(timeout=0.5):
                continue
            try:
                self._handle_press()
            except Exception:
                LOGGER.exception("button turn failed")
                self.hardware.signal_error()
                self.hardware.off()

    def _run_realtime(self) -> None:
        LOGGER.info("voice daemon ready with realtime streaming backend")
        self._set_led_mode("off")
        while self.running:
            try:
                self._drain_realtime_turns()
                assert self.realtime is not None
                if not self.realtime.is_voice_busy:
                    self.realtime.apply_deferred_music_after_voice()
                self.realtime.check_health()
                self.realtime.close_if_too_old()
                self.realtime.close_if_idle()
                self.realtime.cool_down_if_silent()
                self._run_due_scheduled_tasks()
                self._run_due_background_wakeups()
                self._sync_realtime_led()
                if not self.hardware.wait_for_press(timeout=0.05):
                    continue
                self._handle_realtime_press()
            except Exception:
                LOGGER.exception("realtime loop failed")
                if self.realtime is not None:
                    self.realtime.close()
                self.hardware.signal_error()
                try:
                    self.hardware.wait_for_release(timeout=2)
                except Exception:
                    LOGGER.exception("failed while waiting for button release after realtime error")
                self._led_mode = "off"

    def _handle_realtime_press(self) -> None:
        assert self.realtime is not None
        LOGGER.info("realtime button pressed")
        self._set_led_mode("on")
        self.realtime.pause_music_for_voice()
        self.realtime.begin_turn(self.conversation.load())
        self.hardware.wait_for_release()
        duration = self.realtime.stop_recording()
        LOGGER.info("realtime button released after %.3fs", duration)

        if duration < self.config.min_record_seconds:
            if duration <= self.config.short_click_seconds and self._consume_second_click():
                LOGGER.info("double click detected; clearing conversation and closing realtime session")
                self.realtime.clear_pending_input()
                self.realtime.close()
                self.memory.flush_conversation(self.conversation.load(), "double-click clear")
                self.memory_consolidator.request("double-click clear")
                self.conversation.clear()
                self.hardware.confirm_clear()
                self._led_mode = "off"
                self.realtime.apply_deferred_music_after_voice()
                return
            LOGGER.info("ignoring short realtime recording: %.3fs", duration)
            self.realtime.clear_pending_input()
            self._set_led_mode("off")
            self.realtime.apply_deferred_music_after_voice()
            return

        self.realtime.commit_recording()
        self._set_led_mode("blink")

    def _drain_realtime_turns(self) -> None:
        if self.realtime is None:
            return
        for result in self.realtime.pop_completed_turns():
            if result.assistant_text:
                self.conversation.append_pair(result.user_text or "[voice input]", result.assistant_text)
                self.memory.append_turn(result.user_text or "[voice input]", result.assistant_text)
                self.memory_consolidator.request("realtime turn")
                LOGGER.info("realtime turn persisted")
            elif result.requested_close and result.user_text:
                self.conversation.append_pair(result.user_text, "[realtime session closed]")
                self.memory.append_turn(result.user_text, "[realtime session closed]")
                self.memory_consolidator.request("realtime close")
                LOGGER.info("realtime close turn persisted")

    def _sync_realtime_led(self) -> None:
        assert self.realtime is not None
        if self.realtime.is_recording:
            self._set_led_mode("on")
        elif self.realtime.is_responding:
            self._set_led_mode("blink")
        else:
            self._set_led_mode("off")

    def _run_due_scheduled_tasks(self) -> None:
        if self.realtime is None or self.realtime.is_voice_busy:
            return
        for task in self.scheduled_tasks.due(limit=1):
            if self._start_scheduled_task(task):
                return

    def _start_scheduled_task(self, task: ScheduledTask) -> bool:
        assert self.realtime is not None
        if task.action == "speak":
            paused_music = self.realtime.pause_music_for_voice()
            started = self.realtime.trigger_scheduled_speech(
                self.conversation.load(),
                title=task.title,
                prompt=task.prompt,
            )
            if not started:
                if paused_music:
                    self.realtime.apply_deferred_music_after_voice()
                return False
            self.scheduled_tasks.mark_started(task.id)
            self._set_led_mode("blink")
            LOGGER.info("started scheduled speech task %s: %s", task.id, task.title)
            return True
        if task.action == "background_task":
            result = self.realtime.tasks.start(
                task.prompt,
                history=self.conversation.load(),
                title=task.title,
                source="scheduled",
            )
            if not result.get("ok"):
                LOGGER.warning("failed to start scheduled background task %s: %s", task.id, result)
                return False
            self.scheduled_tasks.mark_started(task.id)
            LOGGER.info("started scheduled background task %s: %s", task.id, task.title)
            return True
        LOGGER.warning("unknown scheduled task action %s for %s", task.action, task.id)
        return False

    def _run_due_background_wakeups(self) -> None:
        if self.realtime is None or self.realtime.is_voice_busy:
            return
        if self._quiet_for_unsolicited_speech():
            return
        for task in self.realtime.pending_background_wakeups(limit=1):
            paused_music = self.realtime.pause_music_for_voice()
            if self.realtime.trigger_background_task_wakeup(self.conversation.load(), task):
                task_id = str(task.get("id") or "")
                wakeup = task.get("wakeup") if isinstance(task.get("wakeup"), dict) else {}
                self.realtime.mark_background_wakeup_reported(task_id, str(wakeup.get("message_id") or ""))
                self._set_led_mode("blink")
                LOGGER.info("started background wakeup for task %s", task_id)
                return
            if paused_music:
                self.realtime.apply_deferred_music_after_voice()

    def _quiet_for_unsolicited_speech(self) -> bool:
        try:
            now = datetime.now(ZoneInfo(self.config.user_timezone))
        except Exception:
            now = datetime.now().astimezone()
        return self.scheduled_tasks.is_quiet_time(now)

    def _set_led_mode(self, mode: str) -> None:
        if self._led_mode == mode:
            return
        self._led_mode = mode
        if mode == "on":
            self.hardware.on()
        elif mode == "blink":
            self.hardware.blink()
        else:
            self.hardware.off()

    def _handle_press(self) -> None:
        LOGGER.info("button pressed")
        self.hardware.on()
        recording_path = self.recorder.start()
        self.hardware.wait_for_release()
        recording_path, duration = self.recorder.stop()
        LOGGER.info("button released after %.3fs", duration)
        self.hardware.off()

        if duration <= self.config.short_click_seconds and self._consume_second_click():
            LOGGER.info("double click detected; clearing conversation")
            self.memory.flush_conversation(self.conversation.load(), "double-click clear")
            self.memory_consolidator.request("double-click clear")
            self.conversation.clear()
            self.hardware.confirm_clear()
            return

        if not self.recorder.is_usable(recording_path, duration):
            LOGGER.info("ignoring short or empty recording: %.3fs %s", duration, recording_path)
            return

        with self.hardware.blinking():
            history = self.conversation.load()
            if self.config.voice_bot_backend == "realtime":
                assert self.realtime is not None
                self.realtime.wait_for_connectivity()
                result = self.realtime.respond_to_audio(history, recording_path)
                if result.assistant_text:
                    self.conversation.append_pair(result.user_text or "[voice input]", result.assistant_text)
                    self.memory.append_turn(result.user_text or "[voice input]", result.assistant_text)
                    self.memory_consolidator.request("voice turn")
                elif result.requested_close and result.user_text:
                    self.conversation.append_pair(result.user_text, "[realtime session closed]")
                    self.memory.append_turn(result.user_text, "[realtime session closed]")
                    self.memory_consolidator.request("realtime close")
                LOGGER.info("realtime turn complete")
            else:
                assert self.openai is not None
                self.openai.wait_for_connectivity()
                transcript = self.openai.transcribe(recording_path)
                answer = self.openai.respond_and_speak(history, transcript)
                self.conversation.append_pair(transcript, answer)
                self.memory.append_turn(transcript, answer)
                self.memory_consolidator.request("responses turn")
                LOGGER.info("turn complete")

    def _consume_second_click(self) -> bool:
        if not self.hardware.wait_for_press(timeout=self.config.double_click_window_seconds):
            return False
        self.hardware.on()
        self.hardware.wait_for_release(timeout=2)
        self.hardware.off()
        return True


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> None:
    try:
        config = Config.from_env()
    except Exception as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    configure_logging(config.log_level)
    daemon = VoiceDaemon(config)

    def handle_signal(signum: int, _frame: FrameType | None) -> None:
        LOGGER.info("received signal %s", signum)
        daemon.stop()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    while daemon.running:
        try:
            daemon.run()
        except KeyboardInterrupt:
            daemon.stop()
        except Exception:
            LOGGER.exception("daemon crashed; retrying")
            time.sleep(2)
