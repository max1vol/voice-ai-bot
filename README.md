# Voice AI Bot

Push-to-talk daemon for a Raspberry Pi with a Google AIY Voice HAT.

Behavior:

- Press and hold the HAT button to record from the HAT microphones.
- The button LED is solid while recording.
- Release the button to transcribe, stream a GPT-5.5 response, synthesize speech, and play it through the HAT speaker.
- The LED blinks while the API call and speech playback are running.
- Double-click the button to clear the saved conversation.
- Conversation history is persisted as JSON on the Pi.

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

The service intentionally keeps secrets out of git. `.env` is copied to the Pi during install but is ignored locally and remotely.
