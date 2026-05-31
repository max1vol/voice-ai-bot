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
- Realtime turns include the current local time and user context for Cambridge, UK.
- The realtime model has a `web_search` function backed by GPT-5.5 with hosted web search for current or local facts.
- Before calling OpenAI, the daemon waits for DNS and TCP connectivity to `api.openai.com`; this avoids losing the first turn while Wi-Fi is still settling after boot.

Two backends are available:

- `VOICE_BOT_BACKEND=responses`: the original flow, using speech-to-text, GPT-5.5, and TTS.
- `VOICE_BOT_BACKEND=realtime`: a persistent push-to-talk WebSocket session using `gpt-realtime-2`. It disables VAD, streams PCM from the mic while the button is held, streams PCM audio back to the speaker, supports button barge-in, and closes on idle timeout, hard session timeout, double-click, or when the model calls `close_realtime_session`.

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
REALTIME_REASONING_EFFORT=low
REALTIME_IDLE_TIMEOUT_SECONDS=45
REALTIME_MAX_SESSION_SECONDS=300
USER_CITY=Cambridge
USER_REGION=Cambridgeshire
USER_COUNTRY=GB
USER_TIMEZONE=Europe/London
WEB_SEARCH_MODEL=gpt-5.5
WEB_SEARCH_REASONING_EFFORT=medium
```

The service intentionally keeps secrets out of git. `.env` is copied to the Pi during install but is ignored locally and remotely.

If a turn fails, the LED flashes rapidly several times and the journal includes the root error:

```bash
sudo journalctl -u voice-ai-bot -n 120 --no-pager
```
