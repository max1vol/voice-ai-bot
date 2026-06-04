# Max AI Watch

PlatformIO firmware for the LILYGO T-Watch 2020 V1 companion watch.

The watchface shows:

- Local Europe/London time from Wi-Fi + NTP.
- Wi-Fi and battery status icons.
- Cambridge weather from OpenWeather, cached on-device.
- A top-region double-tap TTS route that speaks the current time and weather through the watch speaker.
- A lower-region triple-tap drawer with brightness, volume, and Voice toggle controls.

## Controls

- On battery, the watch deep-sleeps when the screen times out. Touch the screen once to wake it, then use the gestures below.
- Double-tap the top third of the touchscreen, full width around the title/header, to speak the current time and weather.
- Triple-tap the lower two thirds of the touchscreen, full width, to open the drawer.
- Tap the top-left `X` button to close the drawer.
- Use the drawer Voice toggle to disable or re-enable TTS.

During a TTS request, the header replaces `Max AI Watch` with `connecting...`, `weather...`, `calling tts...`, `waiting tts...`, and `speaking...`. Network, weather, and TTS errors are shown in red for 10 seconds or until the next status.

## Battery Policy

- Wi-Fi is off by default and is only enabled for short NTP, weather, or TTS bursts.
- On battery, screen timeout enters ESP32 deep sleep and wakes on touchscreen interrupt or a 6-hour maintenance timer.
- The maintenance timer is used for rare NTP sync; weather is refreshed while the screen is on or before TTS, with the last result retained across deep sleep.
- On USB/external power, the firmware turns off the display, radio, and peripherals at timeout but avoids deep sleep so serial debugging and flashing remain practical.

## Hardware

Target hardware:

- LILYGO T-Watch 2020 V1
- ESP32 main chip
- ST7789 240x240 LCD
- FT6236 touch controller
- AXP202 power management
- PCF8563 RTC

## Dependencies

The firmware uses the LILYGO watch library as a sibling directory to this PlatformIO project. From the repository root:

```bash
git clone --depth 1 https://github.com/Xinyuan-LilyGO/TTGO_TWatch_Library.git watch/TTGO_TWatch_Library
```

PlatformIO installs the remaining Arduino libraries from `platformio.ini`.

## Secrets

`src/secrets.h` is intentionally ignored by git. Create it either by copying the template:

```bash
cp watch/max-ai-watch/src/secrets.example.h watch/max-ai-watch/src/secrets.h
```

or by generating it from local secret files:

```bash
python3 watch/max-ai-watch/scripts/generate_secrets.py
```

The generator reads `watch/max-ai-watch/.env`, `~/.wifi`, and `~/.zshrc-secrets`. For multiple Wi-Fi networks, the easiest format is:

```bash
MAX_AI_WATCH_WIFI_NETWORKS="ssid1=password1;ssid2=password2;ssid3=password3"
OPENAI_API_KEY="sk-..."
OPENWEATHER_API_KEY="..."
```

It also accepts `AI_GATEWAY_API_KEY` as the OpenAI key source.

## Build And Flash

Connect the watch over USB, then run from the repository root:

```bash
pio run -d watch/max-ai-watch -e t-watch-2020-v1
pio run -d watch/max-ai-watch -e t-watch-2020-v1 --target upload
```

The default upload port is `/dev/cu.usbserial-022157BE`. If your adapter appears as a different port, edit `watch/max-ai-watch/platformio.ini` or pass a PlatformIO upload port override.

Serial monitor:

```bash
pio device monitor -p /dev/cu.usbserial-022157BE -b 115200
```
