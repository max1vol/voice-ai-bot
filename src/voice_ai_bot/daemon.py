from __future__ import annotations

import logging
import signal
import sys
import time
from types import FrameType

from .audio_io import Recorder
from .config import Config
from .conversation import ConversationStore
from .hardware import HatHardware
from .openai_voice import OpenAIVoiceClient

LOGGER = logging.getLogger(__name__)


class VoiceDaemon:
    def __init__(self, config: Config):
        self.config = config
        self.hardware = HatHardware(config.button_gpio, config.led_gpio, config.button_pull_up)
        self.recorder = Recorder(config)
        self.conversation = ConversationStore(config.conversation_file)
        self.openai = OpenAIVoiceClient(config)
        self.running = True

    def stop(self) -> None:
        self.running = False
        self.hardware.off()

    def run(self) -> None:
        LOGGER.info("voice daemon ready")
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
            self.conversation.clear()
            self.hardware.confirm_clear()
            return

        if not self.recorder.is_usable(recording_path, duration):
            LOGGER.info("ignoring short or empty recording: %.3fs %s", duration, recording_path)
            return

        with self.hardware.blinking():
            self.openai.wait_for_connectivity()
            transcript = self.openai.transcribe(recording_path)
            history = self.conversation.load()
            answer = self.openai.respond_and_speak(history, transcript)
            self.conversation.append_pair(transcript, answer)
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
