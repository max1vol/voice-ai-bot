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
- SipQuest has a file-backed memory workspace under `/var/lib/voice-ai-bot/agent`. Raw turn notes are written immediately as an audit log, and GPT-5.5 asynchronously consolidates them into compact durable memory entries.
- Realtime turns include the current local time and user context for Cambridge, UK.
- The local OpenWeather tool supports current conditions plus up to 5 forecast days, defaults to Cambridge, UK, and can also answer for any named location.
- The realtime model has background task tools backed by GPT-5.5 with hosted web search, explicit user-facing status updates, and hosted code interpreter for current facts, calculations, code generation, and code checks.
- GPT-5.5 background tasks can call the same local weather tool directly instead of using web search for weather.
- Background tasks can be steered by later push-to-talk turns, and can wake the realtime model for important progress updates or final results, but unsolicited wakeups obey the same quiet-hours limit.
- SipQuest can list and play local songs from `/var/lib/voice-ai-bot/music`. Music listing includes durations, and music can be paused, resumed, stopped, or volume-adjusted by voice while a song is playing.
- Voice and music volume default to `5` and `4`. Runtime volume changes are persisted in `/var/lib/voice-ai-bot/settings.json` unless `SETTINGS_FILE` points elsewhere.
- SipQuest can take webcam pictures only when the realtime model calls a camera tool. Button presses do not automatically capture images. One-shot pictures wait for the USB webcam to settle, play a shutter cue, and then attach the image to the live conversation.
- For interactive visual tasks, the realtime model can start continuous webcam capture. Continuous capture keeps the USB camera open and attaches the latest frame every 1-5 seconds, defaulting to every 4 seconds, until the model stops it.
- The Pi debug web UI runs separately from the voice daemon and exposes a camera view over HTTP on port `400` by default. Its Capture button uses the same one-shot camera path as the bot; Start Live repeatedly refreshes that same snapshot path without holding the camera open between frames.
- The realtime model can create, list, and remove scheduled reminders, alarms, and timed tasks. Scheduled speech is persisted as JSON and will not start during quiet hours: 21:00-07:30 local time.
- Before calling OpenAI, the daemon waits for DNS and TCP connectivity to `api.openai.com`; this avoids losing the first turn while Wi-Fi is still settling after boot.

Two backends are available:

- `VOICE_BOT_BACKEND=responses`: the original flow, using speech-to-text, GPT-5.5, and TTS.
- `VOICE_BOT_BACKEND=realtime`: a persistent push-to-talk WebSocket session using `gpt-realtime-2`. It disables VAD, streams PCM from the mic while the button is held, streams PCM audio back to the speaker, supports button barge-in, can start/list/inspect/cancel background GPT-5.5 tasks, can manage memory and scheduled reminders/alarms, and closes on idle timeout, hard session timeout, double-click, or when the model calls `close_realtime_session`.

Memory works in two layers:

- `memory/YYYY-MM-DD.md` receives raw user/assistant turn notes immediately. This does not depend on model judgment, so it is crash-safe.
- `MEMORY.md` contains compact durable entries. The realtime model can still use explicit memory tools, but normal curation is handled by an asynchronous GPT-5.5 consolidation worker with high reasoning. The worker reads queued raw notes, proposes JSON add/update/forget/ignore operations, and application code applies those operations.

## Hardware Defaults

The Google AIY Voice HAT uses GPIO 23 for the button and GPIO 25 for the LED. The daemon records and plays through ALSA device `plughw:1,0`, which is the HAT after enabling `dtoverlay=googlevoicehat-soundcard`.

## Max AI Watch

The companion LILYGO T-Watch 2020 V1 firmware lives in `watch/max-ai-watch`. It is a PlatformIO project for the "Max AI Watch" watchface, Wi-Fi/NTP time sync, Cambridge weather display, drawer controls, and on-watch OpenAI TTS time announcements.

The real watch secrets file is not committed. See `watch/max-ai-watch/README.md` for dependency setup, `src/secrets.h` generation, build, flash, and serial monitor commands.

## Music

Put audio files in `/var/lib/voice-ai-bot/music`. The preferred format is high-quality mono Opus (`.opus`); the player decodes Opus to the same 24 kHz mono PCM stream used by the speaker output. WAV still works as a compatibility fallback, but Opus is the intended storage format to save SD-card space. The bot exposes voice tools to list songs, play by title, pause, resume, stop, and set song volume. When the button is pressed during music playback, music pauses immediately; it resumes after the voice turn unless the user asked to pause, stop, or play something else.

## Pi Setup

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. Then run:

```bash
scripts/install_pi.sh pi@192.168.1.90
```

The install script first builds a minimal staged deploy bundle, syncs only that bundle to `/opt/voice-ai-bot`, creates a venv with system GPIO packages visible, installs Python dependencies, installs the systemd units, enables the Voice HAT overlay, and starts the services. It intentionally excludes `watch/`, `docs/`, `tests/`, local caches, and other non-runtime files from the Pi.

The install also starts `voice-ai-bot-debug.service`. If the laptop can reach Pi services over Tailscale/MagicDNS, open:

```bash
open http://pi3:400/
```

If direct access is blocked and you want a local forwarded port instead, run this on the laptop and then open `http://127.0.0.1:400/`:

```bash
ssh -N -L 400:127.0.0.1:400 pi@pi3
```

Use your normal Tailscale shell for command-line access:

```bash
tailscale ssh pi@pi3
```

## Useful Commands

```bash
sudo systemctl status voice-ai-bot
sudo systemctl status voice-ai-bot-debug
sudo journalctl -u voice-ai-bot -f
sudo journalctl -u voice-ai-bot-debug -f
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
REALTIME_SILENT_COOLDOWN_SECONDS=15
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
SETTINGS_FILE=/var/lib/voice-ai-bot/settings.json
VOICE_VOLUME=5
MUSIC_DIR=/var/lib/voice-ai-bot/music
MUSIC_VOLUME=4
CAMERA_CAPTURE_ON_BUTTON_PRESS=false
CAMERA_SNAPSHOT_SETTLE_SECONDS=3
CAMERA_SHUTTER_SOUND_ENABLED=true
CAMERA_CONTINUOUS_INTERVAL_SECONDS=4
CAMERA_CONTINUOUS_MIN_INTERVAL_SECONDS=1
CAMERA_CONTINUOUS_MAX_INTERVAL_SECONDS=5
DEBUG_WEB_HOST=0.0.0.0
DEBUG_WEB_PORT=400
SCHEDULED_TASKS_FILE=/var/lib/voice-ai-bot/scheduled_tasks.json
SCHEDULE_QUIET_START=21:00
SCHEDULE_QUIET_END=07:30
MEMORY_DIR=/var/lib/voice-ai-bot/agent
MEMORY_BOOTSTRAP_CHARS=12000
MEMORY_ACTIVE_CONTEXT_CHARS=1800
MEMORY_CONSOLIDATION_ENABLED=true
MEMORY_CONSOLIDATION_MODEL=gpt-5.5
MEMORY_CONSOLIDATION_REASONING_EFFORT=high
MEMORY_CONSOLIDATION_DEBOUNCE_SECONDS=5
MEMORY_CONSOLIDATION_SHUTDOWN_TIMEOUT_SECONDS=30
MEMORY_CONSOLIDATION_MAX_NOTES=12
MEMORY_CONSOLIDATION_MAX_CHARS=16000
```

The service intentionally keeps secrets out of git. `.env` is copied to the Pi during install but is ignored locally and remotely.

If a turn fails, the LED flashes rapidly several times and the journal includes the root error:

```bash
sudo journalctl -u voice-ai-bot -n 120 --no-pager
```
