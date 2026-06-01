# Voice AI Bot

Push-to-talk daemon for a Raspberry Pi with a Google AIY Voice HAT.

Behavior:

- Press and hold the HAT button to record from the HAT microphones.
- The button LED is solid while recording.
- Release the button to send the turn, stream the response, and play it through the HAT speaker.
- The LED blinks while the API call and speech playback are running.
- In realtime mode, pressing the button while the speaker is talking cancels playback and starts the next turn.
- Double-click the button to clear the saved conversation.
- Conversation history is persisted as JSON on the Pi.
- Max Code has a file-backed memory workspace under `/var/lib/voice-ai-bot/agent`, with durable memory tools for searching, adding, correcting, and forgetting saved facts.
- Realtime turns include the current local time and user context for Cambridge, UK.
- The realtime model has background task tools backed by GPT-5.5 with hosted web search, reasoning summaries, and hosted code interpreter for current facts, calculations, code generation, and code checks.
- Background tasks can wake the realtime model when they finish so the device can speak the result, but unsolicited wakeups obey the same quiet-hours limit.
- The realtime model can create, list, and remove scheduled reminders, alarms, and timed tasks. Scheduled speech is persisted as JSON and will not start during quiet hours: 21:00-07:30 local time.
- Before calling OpenAI, the daemon waits for DNS and TCP connectivity to `api.openai.com`; this avoids losing the first turn while Wi-Fi is still settling after boot.

Two backends are available:

- `VOICE_BOT_BACKEND=responses`: the original flow, using speech-to-text, GPT-5.5, and TTS.
- `VOICE_BOT_BACKEND=realtime`: a persistent push-to-talk WebSocket session using `gpt-realtime-2`. It disables VAD, streams PCM from the mic while the button is held, streams PCM audio back to the speaker, supports button barge-in, can start/list/inspect/cancel background GPT-5.5 tasks, can manage memory and scheduled reminders/alarms, and closes on idle timeout, hard session timeout, double-click, or when the model calls `close_realtime_session`.

## Hardware Defaults

The Google AIY Voice HAT uses GPIO 23 for the button and GPIO 25 for the LED. The daemon records and plays through ALSA device `plughw:1,0`, which is the HAT after enabling `dtoverlay=googlevoicehat-soundcard`.

## Pi Setup

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. Then run:

```bash
scripts/install_pi.sh pi@192.168.1.90
```

The install script copies this repo to `/opt/voice-ai-bot`, creates a venv with system GPIO packages visible, installs Python dependencies, installs the systemd unit, enables the Voice HAT overlay, and starts the service.

## Useful Commands

```bash
sudo systemctl status voice-ai-bot
sudo journalctl -u voice-ai-bot -f
arecord -D plughw:1,0 -f S16_LE -r 16000 -c 1 -d 2 /tmp/test.wav
aplay -D plughw:1,0 /tmp/test.wav
```

For the realtime backend, record at 24 kHz mono PCM:

```bash
VOICE_BOT_BACKEND=realtime
RECORD_RATE=24000
REALTIME_INPUT_RATE=24000
REALTIME_MODEL=gpt-realtime-2
REALTIME_REASONING_EFFORT=medium
REALTIME_IDLE_TIMEOUT_SECONDS=45
REALTIME_MAX_SESSION_SECONDS=300
REALTIME_SILENT_COOLDOWN_SECONDS=5
USER_CITY=Cambridge
USER_REGION=Cambridgeshire
USER_COUNTRY=GB
USER_TIMEZONE=Europe/London
WEB_SEARCH_MODEL=gpt-5.5
WEB_SEARCH_REASONING_EFFORT=high
TASK_MODEL=gpt-5.5
TASK_REASONING_EFFORT=high
TASK_REASONING_SUMMARY=auto
TASK_CODE_EXECUTION=true
SCHEDULED_TASKS_FILE=/var/lib/voice-ai-bot/scheduled_tasks.json
SCHEDULE_QUIET_START=21:00
SCHEDULE_QUIET_END=07:30
MEMORY_DIR=/var/lib/voice-ai-bot/agent
MEMORY_BOOTSTRAP_CHARS=12000
MEMORY_ACTIVE_CONTEXT_CHARS=1800
```

The service intentionally keeps secrets out of git. `.env` is copied to the Pi during install but is ignored locally and remotely.

If a turn fails, the LED flashes rapidly several times and the journal includes the root error:

```bash
sudo journalctl -u voice-ai-bot -n 120 --no-pager
```
