#include "config.h"
#include "max_ai_face.h"
#include "secrets.h"

#include <ArduinoJson.h>
#include <AudioFileSourceID3.h>
#include <AudioFileSourceSPIFFS.h>
#include <AudioGeneratorMP3.h>
#include <AudioOutputI2S.h>
#include <HTTPClient.h>
#include <SPIFFS.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <math.h>
#include <time.h>

namespace {
constexpr char kTimezoneLondon[] = "GMT0BST,M3.5.0/1,M10.5.0";
constexpr char kNtpServer1[] = "pool.ntp.org";
constexpr char kNtpServer2[] = "time.google.com";
constexpr char kNtpServer3[] = "time.cloudflare.com";
constexpr uint8_t kActiveBrightness = 180;
constexpr uint32_t kBacklightTimeoutMs = 12000;
constexpr uint32_t kWifiIconPollMs = 5000;
constexpr uint32_t kBatteryPollMs = 15000;
constexpr uint32_t kWeatherCacheMs = 10UL * 60UL * 1000UL;
constexpr uint32_t kWeatherRetryMs = 60UL * 1000UL;
constexpr uint32_t kDrawerAutoCloseMs = 10000;
constexpr uint32_t kTouchTapSuppressMs = 600;
constexpr uint32_t kTripleTapWindowMs = 900;
constexpr uint32_t kTapMaxDurationMs = 700;
constexpr uint8_t kMinBrightness = 25;
constexpr int kScreenH = 240;
constexpr int kDrawerX = 8;
constexpr int kDrawerW = 224;
constexpr int kDrawerH = 204;
constexpr int kDrawerY = (kScreenH - kDrawerH) / 2;
constexpr int kDrawerCloseButtonR = 15;
constexpr int kDrawerCloseButtonX = kDrawerCloseButtonR;
constexpr int kDrawerCloseButtonY = kDrawerCloseButtonR;
constexpr int kDrawerCloseHitR = 30;
constexpr int kSliderX = 38;
constexpr int kSliderW = 164;
constexpr int kSliderTrackH = 10;
constexpr int kSliderKnobR = 12;
constexpr int kBrightnessSliderY = kDrawerY + 54;
constexpr int kVolumeSliderY = kDrawerY + 114;
constexpr int kVoiceToggleY = kDrawerY + 172;
constexpr int kTapMaxTravel = 48;
constexpr char kSpeechPath[] = "/speech.mp3";

TTGOClass *watch = nullptr;
TFT_eSPI *display = nullptr;

bool timeSynced = false;
bool usedRtcFallback = false;
int lastMinuteOfDay = -1;
int lastYearDay = -1;
bool backlightOn = true;
bool touchWasDown = false;
bool touchConsumed = false;
bool ttsRequested = false;
bool ttsBusy = false;
bool pendingTouchTts = false;
uint32_t lastWifiAttemptMs = 0;
uint32_t lastNtpAttemptMs = 0;
uint32_t lastSerialStatusMs = 0;
uint32_t lastBacklightWakeMs = 0;
uint32_t lastPmuPollMs = 0;
uint32_t lastWeatherAttemptMs = 0;
uint32_t suppressTouchTapUntilMs = 0;
uint32_t suppressVoiceToggleUntilMs = 0;
uint32_t touchStartMs = 0;
uint32_t lastTapMs = 0;
uint32_t pendingTouchTtsAtMs = 0;
int lastPmuIntPinLevel = -1;
char lastStatusText[48] = "";
volatile bool powerButtonIrq = false;
int nextWifiNetworkIndex = 0;
int activeWifiNetworkIndex = -1;
int16_t touchStartX = 0;
int16_t touchStartY = 0;
int16_t touchLastX = 0;
int16_t touchLastY = 0;
int16_t touchMinX = 0;
int16_t touchMaxX = 0;
int16_t touchMinY = 0;
int16_t touchMaxY = 0;
uint8_t currentBrightness = kActiveBrightness;
float speechVolume = 0.75f;
bool drawerOpen = false;
bool drawerNeedsRedraw = false;
bool voiceEnabled = true;
bool voiceToggleTouchConsumed = false;
uint8_t tapCount = 0;
uint32_t drawerLastInteractionMs = 0;

struct WeatherStatus {
    bool valid;
    int temperatureC;
    bool rainToday;
    char condition[20];
    uint32_t fetchedMs;
};

WeatherStatus cachedWeather {false, 0, false, "", 0};

struct BatteryStatus {
    int percent;
    bool valid;
    bool charging;
    bool externalPower;
};

int cachedWifiBars = -2;
int lastDrawnWifiBars = -99;
uint32_t lastWifiIconPollMs = 0;
bool wifiIconCacheReady = false;

BatteryStatus cachedBatteryStatus {-1, false, false, false};
int lastDrawnBatteryPercent = -99;
bool lastDrawnBatteryValid = false;
bool lastDrawnBatteryCharging = false;
bool lastDrawnExternalPower = false;
uint32_t lastBatteryPollMs = 0;
bool batteryCacheReady = false;

const char *const kSmallNumbers[] = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen"
};
const char *const kTensNumbers[] = {
    "", "", "twenty", "thirty", "forty", "fifty"
};

void IRAM_ATTR onPowerButton()
{
    powerButtonIrq = true;
}

uint16_t dimColor(uint8_t r, uint8_t g, uint8_t b)
{
    return display->color565(r, g, b);
}

void drawPanel(int16_t x, int16_t y, int16_t w, int16_t h, uint16_t border)
{
    display->fillRoundRect(x, y, w, h, 8, TFT_BLACK);
    display->drawRoundRect(x, y, w, h, 8, border);
}

void drawStaticFace()
{
    display->pushImage(0, 0, MAX_AI_FACE_WIDTH, MAX_AI_FACE_HEIGHT, maxAiFace);
    drawPanel(16, 14, 208, 34, dimColor(24, 90, 120));
    drawPanel(15, 68, 210, 76, dimColor(35, 130, 170));
    drawPanel(28, 168, 184, 44, dimColor(95, 72, 36));

    display->setTextDatum(MC_DATUM);
    display->setTextColor(dimColor(198, 245, 255), TFT_BLACK);
    display->drawString("Max AI Watch", 120, 31, 2);
}

int wifiSignalBars()
{
    if (WiFi.status() != WL_CONNECTED) {
        return -1;
    }

    const int rssi = WiFi.RSSI();
    if (rssi >= -55) {
        return 4;
    }
    if (rssi >= -67) {
        return 3;
    }
    if (rssi >= -75) {
        return 2;
    }
    return 1;
}

int currentWifiBars()
{
    const uint32_t nowMs = millis();
    if (!wifiIconCacheReady || nowMs - lastWifiIconPollMs > kWifiIconPollMs) {
        cachedWifiBars = wifiSignalBars();
        wifiIconCacheReady = true;
        lastWifiIconPollMs = nowMs;
    }
    return cachedWifiBars;
}

BatteryStatus readBatteryStatus()
{
    BatteryStatus status {-1, false, false, false};
    if (watch == nullptr || watch->power == nullptr) {
        return status;
    }

    status.charging = watch->power->isChargeing();
    status.externalPower = watch->power->isVBUSPlug();
    if (!watch->power->isBatteryConnect()) {
        return status;
    }

    int percent = watch->power->getBattPercentage();
    if (percent < 0) {
        percent = 0;
    }
    if (percent > 100) {
        percent = 100;
    }
    status.percent = percent;
    status.valid = true;
    return status;
}

BatteryStatus currentBatteryStatus()
{
    const uint32_t nowMs = millis();
    if (!batteryCacheReady || nowMs - lastBatteryPollMs > kBatteryPollMs) {
        cachedBatteryStatus = readBatteryStatus();
        batteryCacheReady = true;
        lastBatteryPollMs = nowMs;
    }
    return cachedBatteryStatus;
}

void drawWifiIcon(int bars)
{
    const int x = 27;
    const int baseY = 39;
    const int heights[] = {4, 7, 10, 13};
    const uint16_t active = dimColor(95, 228, 255);
    const uint16_t inactive = dimColor(36, 64, 72);

    display->fillRect(22, 20, 34, 22, TFT_BLACK);
    for (int i = 0; i < 4; ++i) {
        const int barX = x + (i * 6);
        const int barH = heights[i];
        const uint16_t color = bars >= i + 1 ? active : inactive;
        display->fillRoundRect(barX, baseY - barH, 4, barH, 1, color);
    }

    if (bars < 0) {
        const uint16_t offline = dimColor(230, 72, 72);
        display->drawLine(25, 24, 51, 40, offline);
        display->drawLine(25, 25, 51, 41, offline);
    }
}

void drawBatteryIcon(const BatteryStatus &status)
{
    const int x = 185;
    const int y = 23;
    const int w = 25;
    const int h = 12;

    display->fillRect(181, 20, 38, 20, TFT_BLACK);

    uint16_t outline = dimColor(90, 120, 125);
    if (status.valid) {
        if (status.percent <= 15) {
            outline = dimColor(235, 72, 72);
        } else if (status.percent <= 35) {
            outline = dimColor(246, 191, 76);
        } else {
            outline = dimColor(95, 224, 164);
        }
    }
    if (status.charging || status.externalPower) {
        outline = dimColor(116, 238, 126);
    }

    display->drawRect(x, y, w, h, outline);
    display->fillRect(x + w, y + 4, 2, 4, outline);

    if (status.valid) {
        const int fillMax = w - 4;
        int fillW = (status.percent * fillMax + 99) / 100;
        if (fillW < 1 && status.percent > 0) {
            fillW = 1;
        }
        if (fillW > fillMax) {
            fillW = fillMax;
        }
        if (fillW > 0) {
            display->fillRect(x + 2, y + 2, fillW, h - 4, outline);
        }
    }

    if (status.charging || status.externalPower) {
        display->fillTriangle(x + 12, y + 2, x + 8, y + 8, x + 13, y + 8, TFT_WHITE);
        display->fillTriangle(x + 11, y + 6, x + 16, y + 6, x + 10, y + 11, TFT_WHITE);
    }
}

void drawStatusIcons()
{
    const int wifiBars = currentWifiBars();
    if (wifiBars != lastDrawnWifiBars) {
        drawWifiIcon(wifiBars);
        lastDrawnWifiBars = wifiBars;
    }

    const BatteryStatus battery = currentBatteryStatus();
    if (
        battery.percent != lastDrawnBatteryPercent ||
        battery.valid != lastDrawnBatteryValid ||
        battery.charging != lastDrawnBatteryCharging ||
        battery.externalPower != lastDrawnExternalPower
    ) {
        drawBatteryIcon(battery);
        lastDrawnBatteryPercent = battery.percent;
        lastDrawnBatteryValid = battery.valid;
        lastDrawnBatteryCharging = battery.charging;
        lastDrawnExternalPower = battery.externalPower;
    }
}

bool getLocalTimeInfo(tm &info, uint32_t timeoutMs = 10)
{
    return getLocalTime(&info, timeoutMs);
}

void syncRtcFromSystem()
{
    if (watch->rtc == nullptr) {
        return;
    }
    watch->rtc->syncToRtc();
    Serial.println("[time] RTC updated from NTP time");
}

bool syncTimeFromNtp()
{
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[time] NTP skipped: WiFi is not connected");
        return false;
    }

    Serial.println("[time] Starting NTP sync for Europe/London");
    setenv("TZ", kTimezoneLondon, 1);
    tzset();
    configTzTime(kTimezoneLondon, kNtpServer1, kNtpServer2, kNtpServer3);

    tm info {};
    for (int i = 0; i < 20; ++i) {
        if (getLocalTimeInfo(info, 500)) {
            syncRtcFromSystem();
            timeSynced = true;
            usedRtcFallback = false;
            char stamp[32];
            strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S %Z", &info);
            Serial.print("[time] NTP synced: ");
            Serial.println(stamp);
            return true;
        }
        delay(100);
    }
    Serial.println("[time] NTP sync failed; will retry");
    return false;
}

void syncSystemFromRtc()
{
    if (watch->rtc == nullptr || !watch->rtc->isValid()) {
        return;
    }

    setenv("TZ", kTimezoneLondon, 1);
    tzset();
    watch->rtc->syncToSystem();
    usedRtcFallback = true;
    Serial.println("[time] System time loaded from RTC fallback");
}

void connectWifi()
{
    if (WiFi.status() == WL_CONNECTED) {
        return;
    }
    if (WIFI_NETWORK_COUNT <= 0) {
        Serial.println("[wifi] No configured networks");
        return;
    }

    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);
    WiFi.disconnect();

    activeWifiNetworkIndex = nextWifiNetworkIndex;
    nextWifiNetworkIndex = (nextWifiNetworkIndex + 1) % WIFI_NETWORK_COUNT;

    Serial.print("[wifi] Connecting to ");
    Serial.println(WIFI_SSIDS[activeWifiNetworkIndex]);
    WiFi.begin(WIFI_SSIDS[activeWifiNetworkIndex], WIFI_PASSWORDS[activeWifiNetworkIndex]);
    lastWifiAttemptMs = millis();
    wifiIconCacheReady = false;
}

bool waitForWifi(uint32_t timeoutMs)
{
    const uint32_t started = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - started < timeoutMs) {
        delay(250);
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.print("[wifi] Connected to ");
        Serial.print(WiFi.SSID());
        Serial.print(", IP=");
        Serial.println(WiFi.localIP());
        return true;
    }

    if (activeWifiNetworkIndex >= 0 && activeWifiNetworkIndex < WIFI_NETWORK_COUNT) {
        Serial.print("[wifi] Timed out on ");
        Serial.println(WIFI_SSIDS[activeWifiNetworkIndex]);
    } else {
        Serial.println("[wifi] Connection timed out");
    }
    return false;
}

bool connectWifiOrdered(uint32_t perNetworkTimeoutMs)
{
    for (int i = 0; i < WIFI_NETWORK_COUNT; ++i) {
        connectWifi();
        if (waitForWifi(perNetworkTimeoutMs)) {
            return true;
        }
    }

    Serial.println("[wifi] All configured networks failed; will retry in background");
    return false;
}

const char *syncLabel();

void maintainWifiAndTime()
{
    const uint32_t nowMs = millis();

    if (WiFi.status() != WL_CONNECTED && nowMs - lastWifiAttemptMs > 15000) {
        WiFi.disconnect();
        connectWifi();
    }

    if (!timeSynced && WiFi.status() == WL_CONNECTED && nowMs - lastNtpAttemptMs > 10000) {
        lastNtpAttemptMs = nowMs;
        syncTimeFromNtp();
    }

    if (nowMs - lastSerialStatusMs > 60000) {
        lastSerialStatusMs = nowMs;
        Serial.print("[status] WiFi=");
        Serial.print(WiFi.status() == WL_CONNECTED ? "connected" : "disconnected");
        Serial.print(" sync=");
        Serial.println(syncLabel());
    }
}

const char *syncLabel()
{
    if (timeSynced) {
        return "NTP";
    }
    if (usedRtcFallback) {
        return "RTC";
    }
    return "SYNC";
}

const char *temperatureWord(int temperatureC)
{
    if (temperatureC >= 25) {
        return "hot";
    }
    if (temperatureC >= 18) {
        return "warm";
    }
    if (temperatureC >= 10) {
        return "cool";
    }
    return "cold";
}

bool isWetCondition(const char *condition)
{
    return strcmp(condition, "Rain") == 0 ||
           strcmp(condition, "Drizzle") == 0 ||
           strcmp(condition, "Thunderstorm") == 0 ||
           strcmp(condition, "Snow") == 0;
}

bool isCloudyCondition(const char *condition)
{
    return strcmp(condition, "Clouds") == 0 ||
           strcmp(condition, "Mist") == 0 ||
           strcmp(condition, "Fog") == 0 ||
           strcmp(condition, "Haze") == 0;
}

bool fetchWeatherForecast()
{
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[weather] skipped: WiFi is not connected");
        return cachedWeather.valid;
    }

    WiFiClientSecure client;
    client.setInsecure();

    HTTPClient http;
    http.setTimeout(15000);
    http.setReuse(false);

    String url = "https://api.openweathermap.org/data/2.5/forecast?q=Cambridge,GB&units=metric&appid=";
    url += OPENWEATHER_API_KEY;
    if (!http.begin(client, url)) {
        Serial.println("[weather] HTTP begin failed");
        return cachedWeather.valid;
    }

    Serial.println("[weather] fetching Cambridge forecast");
    const int code = http.GET();
    if (code != HTTP_CODE_OK) {
        Serial.print("[weather] HTTP error: ");
        Serial.println(code);
        http.end();
        return cachedWeather.valid;
    }

    DynamicJsonDocument doc(24576);
    DeserializationError error = deserializeJson(doc, http.getStream());
    http.end();
    if (error) {
        Serial.print("[weather] JSON parse failed: ");
        Serial.println(error.c_str());
        return cachedWeather.valid;
    }

    JsonArray list = doc["list"].as<JsonArray>();
    if (list.isNull() || list.size() == 0) {
        Serial.println("[weather] empty forecast");
        return cachedWeather.valid;
    }

    JsonObject first = list[0];
    const float temp = first["main"]["temp"] | 0.0f;
    const char *firstCondition = first["weather"][0]["main"] | "";

    tm nowInfo {};
    const bool haveLocalDay = getLocalTimeInfo(nowInfo, 50);
    bool rainToday = false;
    for (JsonObject item : list) {
        if (haveLocalDay) {
            const long dt = item["dt"] | 0;
            time_t forecastTime = static_cast<time_t>(dt);
            tm forecastLocal {};
            localtime_r(&forecastTime, &forecastLocal);
            if (forecastLocal.tm_year != nowInfo.tm_year || forecastLocal.tm_yday != nowInfo.tm_yday) {
                continue;
            }
        }

        const char *condition = item["weather"][0]["main"] | "";
        const float pop = item["pop"] | 0.0f;
        if (isWetCondition(condition) || pop >= 0.30f || !item["rain"].isNull()) {
            rainToday = true;
            break;
        }
    }

    cachedWeather.valid = true;
    cachedWeather.temperatureC = static_cast<int>(roundf(temp));
    cachedWeather.rainToday = rainToday;
    strlcpy(cachedWeather.condition, firstCondition, sizeof(cachedWeather.condition));
    cachedWeather.fetchedMs = millis();

    Serial.printf(
        "[weather] cached: %dC condition=%s precipitation=%s\n",
        cachedWeather.temperatureC,
        cachedWeather.condition,
        cachedWeather.rainToday ? "yes" : "no"
    );
    return true;
}

bool ensureWeatherForecast()
{
    if (cachedWeather.valid && millis() - cachedWeather.fetchedMs < kWeatherCacheMs) {
        return true;
    }
    return fetchWeatherForecast();
}

String weatherSpeechClause()
{
    if (!ensureWeatherForecast() || !cachedWeather.valid) {
        return "";
    }

    String clause = ", it will be ";
    clause += temperatureWord(cachedWeather.temperatureC);
    clause += " ";
    clause += String(cachedWeather.temperatureC);
    clause += " degrees, ";
    if (cachedWeather.rainToday) {
        clause += "rain likely today";
    } else if (isCloudyCondition(cachedWeather.condition)) {
        clause += "cloudy, no rain today";
    } else {
        clause += "no rain today";
    }
    return clause;
}

void maintainWeather()
{
    if (WiFi.status() != WL_CONNECTED || ttsBusy) {
        return;
    }
    if (cachedWeather.valid && millis() - cachedWeather.fetchedMs < kWeatherCacheMs) {
        return;
    }
    if (millis() - lastWeatherAttemptMs < kWeatherRetryMs) {
        return;
    }

    lastWeatherAttemptMs = millis();
    if (fetchWeatherForecast()) {
        lastStatusText[0] = '\0';
    }
}

void weatherDisplayText(char *buffer, size_t size)
{
    if (cachedWeather.valid) {
        const char *summary = "no rain";
        if (cachedWeather.rainToday) {
            summary = "rain";
        } else if (isCloudyCondition(cachedWeather.condition)) {
            summary = "cloudy";
        }
        snprintf(buffer, size, "%dC  %s", cachedWeather.temperatureC, summary);
        return;
    }
    strlcpy(buffer, "Weather syncing", size);
}

String minuteWords(int minute)
{
    if (minute == 0) {
        return "";
    }
    if (minute < 10) {
        return String(" oh ") + kSmallNumbers[minute];
    }
    if (minute < 20) {
        return String(" ") + kSmallNumbers[minute];
    }

    const int tens = minute / 10;
    const int ones = minute % 10;
    String value = String(" ") + kTensNumbers[tens];
    if (ones != 0) {
        value += String(" ") + kSmallNumbers[ones];
    }
    return value;
}

String spokenTimePhrase(const tm &info)
{
    int hour = info.tm_hour;
    const bool pm = hour >= 12;
    hour %= 12;
    if (hour == 0) {
        hour = 12;
    }

    String phrase = "Hi Max, it is ";
    phrase += kSmallNumbers[hour];
    phrase += minuteWords(info.tm_min);
    phrase += pm ? " PM" : " AM";
    phrase += " in Cambridge";
    phrase += weatherSpeechClause();
    phrase += ".";
    return phrase;
}

String jsonEscape(const String &input)
{
    String escaped;
    escaped.reserve(input.length() + 8);
    for (size_t i = 0; i < input.length(); ++i) {
        const char c = input[i];
        switch (c) {
        case '\\':
            escaped += "\\\\";
            break;
        case '"':
            escaped += "\\\"";
            break;
        case '\n':
            escaped += "\\n";
            break;
        case '\r':
            escaped += "\\r";
            break;
        case '\t':
            escaped += "\\t";
            break;
        default:
            escaped += c;
            break;
        }
    }
    return escaped;
}

void redrawBeforeWake();
void wakeBacklight(const char *reason);

void resetDrawState()
{
    lastMinuteOfDay = -1;
    lastYearDay = -1;
    lastStatusText[0] = '\0';
    lastDrawnWifiBars = -99;
    lastDrawnBatteryPercent = -99;
    lastDrawnBatteryValid = false;
    lastDrawnBatteryCharging = false;
    lastDrawnExternalPower = false;
    wifiIconCacheReady = false;
    batteryCacheReady = false;
}

void drawWatchFace(const tm &info)
{
    char timeText[8];
    char dateText[24];
    strftime(timeText, sizeof(timeText), "%H:%M", &info);
    strftime(dateText, sizeof(dateText), "%a %d %b", &info);

    display->setTextDatum(MC_DATUM);
    drawStatusIcons();

    const int minuteOfDay = info.tm_hour * 60 + info.tm_min;
    if (minuteOfDay != lastMinuteOfDay) {
        lastMinuteOfDay = minuteOfDay;
        display->fillRect(35, 79, 170, 54, TFT_BLACK);
        display->setTextColor(TFT_WHITE, TFT_BLACK);
        display->drawString(timeText, 120, 106, 7);
    }

    if (info.tm_yday != lastYearDay) {
        lastYearDay = info.tm_yday;
        display->fillRect(54, 173, 132, 18, TFT_BLACK);
        display->setTextColor(dimColor(255, 202, 112), TFT_BLACK);
        display->drawString(dateText, 120, 181, 2);
    }

    char statusText[48];
    weatherDisplayText(statusText, sizeof(statusText));
    if (strcmp(statusText, lastStatusText) != 0) {
        strlcpy(lastStatusText, statusText, sizeof(lastStatusText));
        display->fillRect(43, 195, 154, 13, TFT_BLACK);
        display->setTextColor(dimColor(132, 220, 232), TFT_BLACK);
        display->drawString(statusText, 120, 202, 1);
    }
}

int clampInt(int value, int low, int high)
{
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

int percentFromSliderX(int16_t x)
{
    return clampInt(((x - kSliderX) * 100) / kSliderW, 0, 100);
}

int currentBrightnessPercent()
{
    return ((currentBrightness - kMinBrightness) * 100) / (255 - kMinBrightness);
}

int currentVolumePercent()
{
    return clampInt(static_cast<int>(speechVolume * 100.0f + 0.5f), 0, 100);
}

void setBrightnessPercent(int percent)
{
    percent = clampInt(percent, 0, 100);
    if (percent == currentBrightnessPercent()) {
        return;
    }
    currentBrightness = kMinBrightness + ((255 - kMinBrightness) * percent) / 100;
    if (backlightOn) {
        watch->setBrightness(currentBrightness);
    }
}

void setVolumePercent(int percent)
{
    percent = clampInt(percent, 0, 100);
    if (percent == currentVolumePercent()) {
        return;
    }
    speechVolume = percent / 100.0f;
}

void suppressTouchTap(uint32_t durationMs = kTouchTapSuppressMs)
{
    suppressTouchTapUntilMs = millis() + durationMs;
}

bool pointInCircle(int16_t x, int16_t y, int16_t cx, int16_t cy, int16_t radius)
{
    const int32_t dx = static_cast<int32_t>(x) - cx;
    const int32_t dy = static_cast<int32_t>(y) - cy;
    return dx * dx + dy * dy <= static_cast<int32_t>(radius) * radius;
}

void drawSlider(const char *label, int trackY, int percent, uint16_t color, bool clearRow)
{
    const int rowY = trackY - 38;
    if (clearRow) {
        display->fillRect(kDrawerX + 12, rowY, kDrawerW - 24, 62, TFT_BLACK);
    }

    percent = clampInt(percent, 0, 100);
    const int fillW = (kSliderW * percent) / 100;

    display->setTextDatum(TL_DATUM);
    display->setTextColor(dimColor(210, 232, 236), TFT_BLACK);
    display->drawString(label, kDrawerX + 18, trackY - 34, 2);

    char valueText[8];
    snprintf(valueText, sizeof(valueText), "%d%%", percent);
    display->setTextDatum(TR_DATUM);
    display->drawString(valueText, kDrawerX + kDrawerW - 18, trackY - 34, 2);

    display->fillRoundRect(kSliderX, trackY, kSliderW, kSliderTrackH, 5, dimColor(32, 48, 54));
    if (fillW > 0) {
        display->fillRoundRect(kSliderX, trackY, fillW, kSliderTrackH, 5, color);
    }

    const int knobX = kSliderX + fillW;
    const int knobY = trackY + (kSliderTrackH / 2);
    display->fillCircle(knobX, knobY, kSliderKnobR, TFT_BLACK);
    display->drawCircle(knobX, knobY, kSliderKnobR, color);
    display->drawCircle(knobX, knobY, kSliderKnobR - 1, color);
    display->fillCircle(knobX, knobY, kSliderKnobR - 5, color);
}

void drawVoiceToggle(bool clearRow)
{
    const int rowY = kVoiceToggleY - 22;
    if (clearRow) {
        display->fillRect(kDrawerX + 12, rowY, kDrawerW - 24, 48, TFT_BLACK);
    }

    display->setTextDatum(TL_DATUM);
    display->setTextColor(dimColor(210, 232, 236), TFT_BLACK);
    display->drawString("Voice", kDrawerX + 18, kVoiceToggleY - 10, 2);

    const int switchX = kDrawerX + kDrawerW - 84;
    const int switchY = kVoiceToggleY - 15;
    const int switchW = 58;
    const int switchH = 30;
    const uint16_t onColor = dimColor(95, 228, 164);
    const uint16_t offColor = dimColor(82, 92, 98);
    const uint16_t color = voiceEnabled ? onColor : offColor;

    display->fillRoundRect(switchX, switchY, switchW, switchH, 15, dimColor(22, 32, 36));
    display->drawRoundRect(switchX, switchY, switchW, switchH, 15, color);
    if (voiceEnabled) {
        display->fillRoundRect(switchX + 28, switchY + 4, 24, 22, 11, color);
    } else {
        display->fillRoundRect(switchX + 6, switchY + 4, 24, 22, 11, color);
    }

    display->setTextDatum(MC_DATUM);
    display->setTextColor(dimColor(226, 242, 244), dimColor(22, 32, 36));
    display->drawString(voiceEnabled ? "ON" : "OFF", switchX + (voiceEnabled ? 16 : 42), switchY + 15, 1);
}

void drawCloseButton()
{
    const uint16_t fill = TFT_BLACK;
    const uint16_t outline = dimColor(128, 92, 255);
    const uint16_t mark = TFT_WHITE;
    display->fillCircle(kDrawerCloseButtonX, kDrawerCloseButtonY, kDrawerCloseButtonR, fill);
    display->drawCircle(kDrawerCloseButtonX, kDrawerCloseButtonY, kDrawerCloseButtonR, outline);
    display->drawCircle(kDrawerCloseButtonX, kDrawerCloseButtonY, kDrawerCloseButtonR - 1, outline);
    display->drawLine(kDrawerCloseButtonX - 6, kDrawerCloseButtonY - 6, kDrawerCloseButtonX + 6, kDrawerCloseButtonY + 6, mark);
    display->drawLine(kDrawerCloseButtonX + 6, kDrawerCloseButtonY - 6, kDrawerCloseButtonX - 6, kDrawerCloseButtonY + 6, mark);
    display->drawLine(kDrawerCloseButtonX - 5, kDrawerCloseButtonY - 6, kDrawerCloseButtonX + 7, kDrawerCloseButtonY + 6, mark);
    display->drawLine(kDrawerCloseButtonX + 7, kDrawerCloseButtonY - 6, kDrawerCloseButtonX - 5, kDrawerCloseButtonY + 6, mark);
}

void drawDrawer()
{
    if (!drawerOpen || !drawerNeedsRedraw) {
        return;
    }

    display->fillRoundRect(kDrawerX, kDrawerY, kDrawerW, kDrawerH, 8, TFT_BLACK);
    display->drawRoundRect(kDrawerX, kDrawerY, kDrawerW, kDrawerH, 8, dimColor(84, 154, 172));
    drawSlider("Brightness", kBrightnessSliderY, currentBrightnessPercent(), dimColor(255, 202, 112), false);
    drawSlider("Volume", kVolumeSliderY, currentVolumePercent(), dimColor(95, 228, 255), false);
    drawVoiceToggle(false);
    drawCloseButton();

    drawerNeedsRedraw = false;
}

void openDrawer()
{
    if (!backlightOn) {
        wakeBacklight("drawer");
    }
    pendingTouchTts = false;
    tapCount = 0;
    drawerOpen = true;
    drawerNeedsRedraw = true;
    drawerLastInteractionMs = millis();
    suppressTouchTap();
    Serial.println("[ui] drawer open");
    drawDrawer();
}

void closeDrawer()
{
    if (!drawerOpen) {
        return;
    }
    drawerOpen = false;
    drawerNeedsRedraw = false;
    pendingTouchTts = false;
    tapCount = 0;
    suppressTouchTap();
    Serial.println("[ui] drawer close");
    redrawBeforeWake();
}

bool updateDrawerFromTouch(int16_t x, int16_t y)
{
    if (!drawerOpen) {
        return false;
    }

    drawerLastInteractionMs = millis();
    if (pointInCircle(x, y, kDrawerCloseButtonX, kDrawerCloseButtonY, kDrawerCloseHitR)) {
        closeDrawer();
        return true;
    }

    if (y >= kBrightnessSliderY - 30 && y <= kBrightnessSliderY + 30) {
        const int percent = percentFromSliderX(x);
        if (percent != currentBrightnessPercent()) {
            setBrightnessPercent(percent);
            drawSlider("Brightness", kBrightnessSliderY, currentBrightnessPercent(), dimColor(255, 202, 112), true);
            drawCloseButton();
        }
        return true;
    }
    if (y >= kVolumeSliderY - 30 && y <= kVolumeSliderY + 30) {
        const int percent = percentFromSliderX(x);
        if (percent != currentVolumePercent()) {
            setVolumePercent(percent);
            drawSlider("Volume", kVolumeSliderY, currentVolumePercent(), dimColor(95, 228, 255), true);
            drawCloseButton();
        }
        return true;
    }
    if (y >= kVoiceToggleY - 28 && y <= kVoiceToggleY + 28) {
        if (!voiceToggleTouchConsumed && millis() >= suppressVoiceToggleUntilMs) {
            voiceEnabled = !voiceEnabled;
            voiceToggleTouchConsumed = true;
            suppressVoiceToggleUntilMs = millis() + 1200;
            suppressTouchTap();
            Serial.println(voiceEnabled ? "[ui] voice on" : "[ui] voice off");
            drawVoiceToggle(true);
        }
        return true;
    }
    if (y > kDrawerY + kDrawerH || x < kDrawerX || x > kDrawerX + kDrawerW) {
        closeDrawer();
        return true;
    }
    return true;
}

void resetTouchState()
{
    touchWasDown = false;
    touchConsumed = false;
    pendingTouchTts = false;
    tapCount = 0;
    voiceToggleTouchConsumed = false;
}

void requestTts(const char *source)
{
    if (!voiceEnabled) {
        Serial.print("[tts] skipped: voice disabled from ");
        Serial.println(source);
        return;
    }
    if (!ttsBusy) {
        ttsRequested = true;
    }
}

void redrawBeforeWake()
{
    drawStaticFace();
    resetDrawState();

    tm info {};
    if (getLocalTimeInfo(info)) {
        drawWatchFace(info);
    }
}

void wakeBacklight(const char *reason)
{
    lastBacklightWakeMs = millis();
    if (backlightOn) {
        return;
    }

    redrawBeforeWake();
    watch->openBL();
    watch->setBrightness(currentBrightness);
    backlightOn = true;
    Serial.print("[display] Backlight on: ");
    Serial.println(reason);
}

void sleepBacklight()
{
    if (!backlightOn) {
        return;
    }

    watch->closeBL();
    backlightOn = false;
    Serial.println("[display] Backlight off");
}

void recordTouchTap()
{
    const uint32_t nowMs = millis();
    if (nowMs < suppressTouchTapUntilMs) {
        pendingTouchTts = false;
        tapCount = 0;
        return;
    }

    if (nowMs - lastTapMs > kTripleTapWindowMs) {
        tapCount = 0;
    }

    lastTapMs = nowMs;
    ++tapCount;
    Serial.printf("[ui] tap count=%u\n", tapCount);

    if (tapCount >= 3) {
        Serial.println("[ui] drawer triple tap");
        pendingTouchTts = false;
        tapCount = 0;
        openDrawer();
        return;
    }

    pendingTouchTts = tapCount == 1;
    pendingTouchTtsAtMs = nowMs + kTripleTapWindowMs;
}

void maintainPendingTouchTts()
{
    if (!pendingTouchTts) {
        return;
    }
    if (drawerOpen || ttsBusy) {
        pendingTouchTts = false;
        tapCount = 0;
        return;
    }
    if (millis() < pendingTouchTtsAtMs) {
        return;
    }

    pendingTouchTts = false;
    tapCount = 0;
    Serial.println("[button] touch tap");
    wakeBacklight("touch");
    requestTts("touch");
}

void maintainTouchWake()
{
    int16_t x = 0;
    int16_t y = 0;
    const bool touched = watch->getTouch(x, y);

    if (touched) {
        lastBacklightWakeMs = millis();
        if (!touchWasDown) {
            touchStartX = x;
            touchStartY = y;
            touchLastX = x;
            touchLastY = y;
            touchMinX = x;
            touchMaxX = x;
            touchMinY = y;
            touchMaxY = y;
            touchStartMs = millis();
            touchConsumed = false;
            voiceToggleTouchConsumed = false;
            if (!backlightOn) {
                wakeBacklight("touch");
            }
        }

        touchLastX = x;
        touchLastY = y;
        if (x < touchMinX) {
            touchMinX = x;
        }
        if (x > touchMaxX) {
            touchMaxX = x;
        }
        if (y < touchMinY) {
            touchMinY = y;
        }
        if (y > touchMaxY) {
            touchMaxY = y;
        }
        if (drawerOpen) {
            touchConsumed = updateDrawerFromTouch(x, y) || touchConsumed;
        }
    } else if (touchWasDown) {
        const int dx = abs(touchMaxX - touchMinX);
        const int dy = abs(touchMaxY - touchMinY);
        const uint32_t touchDurationMs = millis() - touchStartMs;
        if (
            !drawerOpen &&
            !touchConsumed &&
            dx <= kTapMaxTravel &&
            dy <= kTapMaxTravel &&
            touchDurationMs <= kTapMaxDurationMs
        ) {
            recordTouchTap();
        }
        touchConsumed = false;
        voiceToggleTouchConsumed = false;
    }

    if (drawerOpen && !touched && millis() - drawerLastInteractionMs > kDrawerAutoCloseMs) {
        closeDrawer();
    }

    touchWasDown = touched;
}

void maintainPowerButtonWake()
{
    if (watch->power == nullptr) {
        return;
    }

    const uint32_t nowMs = millis();
    const bool shouldPoll = nowMs - lastPmuPollMs > 250;
    if (!powerButtonIrq && !shouldPoll) {
        return;
    }

    lastPmuPollMs = nowMs;
    powerButtonIrq = false;

    uint8_t rawIrq[5] = {};
    for (int i = 0; i < 5; ++i) {
        rawIrq[i] = watch->power->readRegister(AXP202_INTSTS1 + i);
    }
    const int irqPinLevel = digitalRead(AXP202_INT);
    const bool rawPekEdge = (rawIrq[4] & ((1 << 5) | (1 << 6))) != 0;
    const bool anyRawIrq = rawIrq[0] || rawIrq[1] || rawIrq[2] || rawIrq[3] || rawIrq[4];

    if (anyRawIrq || irqPinLevel != lastPmuIntPinLevel) {
        Serial.printf(
            "[pmu] int=%d irq=%02X %02X %02X %02X %02X\n",
            irqPinLevel,
            rawIrq[0],
            rawIrq[1],
            rawIrq[2],
            rawIrq[3],
            rawIrq[4]
        );
        lastPmuIntPinLevel = irqPinLevel;
    }

    watch->power->readIRQ();
    const bool shortPress = watch->power->isPEKShortPressIRQ();
    const bool longPress = watch->power->isPEKLongPressIRQ();
    if (shortPress || longPress || rawPekEdge) {
        if (shortPress) {
            Serial.println("[button] PMU short press");
        } else if (longPress) {
            Serial.println("[button] PMU long press");
        } else {
            Serial.println("[button] PMU edge press");
        }
        wakeBacklight("button");
        requestTts("button");
    }
    watch->power->clearIRQ();
}

void maintainBacklight()
{
    maintainTouchWake();
    maintainPowerButtonWake();
    maintainPendingTouchTts();

    if (backlightOn && millis() - lastBacklightWakeMs > kBacklightTimeoutMs) {
        sleepBacklight();
    }
}

bool downloadSpeechMp3(const String &input)
{
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[tts] skipped: WiFi is not connected");
        return false;
    }

    SPIFFS.remove(kSpeechPath);
    fs::File speech = SPIFFS.open(kSpeechPath, FILE_WRITE);
    if (!speech) {
        Serial.println("[tts] failed to open speech file for writing");
        return false;
    }

    WiFiClientSecure client;
    client.setInsecure();

    HTTPClient http;
    http.setTimeout(20000);
    http.setReuse(false);
    if (!http.begin(client, "https://api.openai.com/v1/audio/speech")) {
        Serial.println("[tts] HTTP begin failed");
        speech.close();
        return false;
    }

    String auth = "Bearer ";
    auth += OPENAI_API_KEY;
    http.addHeader("Authorization", auth);
    http.addHeader("Content-Type", "application/json");

    String body = "{";
    body += "\"model\":\"tts-1\",";
    body += "\"voice\":\"alloy\",";
    body += "\"response_format\":\"mp3\",";
    body += "\"input\":\"";
    body += jsonEscape(input);
    body += "\"}";

    Serial.print("[tts] requesting: ");
    Serial.println(input);
    const int code = http.POST(body);
    if (code != HTTP_CODE_OK) {
        Serial.print("[tts] OpenAI HTTP error: ");
        Serial.println(code);
        String error = http.getString();
        if (error.length() > 0) {
            Serial.println(error.substring(0, 180));
        }
        http.end();
        speech.close();
        SPIFFS.remove(kSpeechPath);
        return false;
    }

    const int written = http.writeToStream(&speech);
    http.end();
    speech.close();

    if (written <= 0) {
        Serial.println("[tts] no MP3 bytes written");
        SPIFFS.remove(kSpeechPath);
        return false;
    }

    Serial.print("[tts] MP3 bytes saved: ");
    Serial.println(written);
    return true;
}

bool playSpeechMp3()
{
    if (!SPIFFS.exists(kSpeechPath)) {
        Serial.println("[audio] speech file missing");
        return false;
    }

    Serial.println("[audio] playing speech");
    watch->enableLDO3(true);
    delay(50);

    AudioFileSourceSPIFFS *file = new AudioFileSourceSPIFFS(kSpeechPath);
    AudioFileSourceID3 *id3 = new AudioFileSourceID3(file);
    AudioOutputI2S *out = new AudioOutputI2S();
    out->SetPinout(TWATCH_DAC_IIS_BCK, TWATCH_DAC_IIS_WS, TWATCH_DAC_IIS_DOUT);
    out->SetGain(speechVolume);

    AudioGeneratorMP3 *mp3 = new AudioGeneratorMP3();
    bool ok = mp3->begin(id3, out);
    while (ok && mp3->isRunning()) {
        if (!mp3->loop()) {
            mp3->stop();
        }
        delay(1);
    }

    delete mp3;
    delete out;
    delete id3;
    delete file;
    watch->enableLDO3(false);
    Serial.println("[audio] playback done");
    return ok;
}

void handleTtsRequest()
{
    if (!ttsRequested || ttsBusy) {
        return;
    }

    ttsRequested = false;
    if (!voiceEnabled) {
        Serial.println("[tts] skipped: voice disabled");
        return;
    }
    ttsBusy = true;
    resetTouchState();
    suppressTouchTap(1500);
    wakeBacklight("tts");

    tm info {};
    if (!getLocalTimeInfo(info, 250)) {
        Serial.println("[tts] no valid local time");
        ttsBusy = false;
        resetTouchState();
        return;
    }

    const String phrase = spokenTimePhrase(info);
    if (downloadSpeechMp3(phrase)) {
        playSpeechMp3();
    }

    lastBacklightWakeMs = millis();
    ttsBusy = false;
    resetTouchState();
    suppressTouchTap(1200);
}
} // namespace

void setup()
{
    Serial.begin(115200);
    delay(200);
    Serial.println();
    Serial.println("[boot] Max AI Watch");

    watch = TTGOClass::getWatch();
    watch->begin();
    watch->openBL();
    watch->setBrightness(currentBrightness);
    lastBacklightWakeMs = millis();
    if (!SPIFFS.begin(true)) {
        Serial.println("[boot] SPIFFS mount failed");
    }

    pinMode(TOUCH_INT, INPUT);
    if (watch->power != nullptr) {
        watch->powerAttachInterrupt(onPowerButton);
        watch->power->enableIRQ(
            AXP202_PEK_SHORTPRESS_IRQ |
            AXP202_PEK_LONGPRESS_IRQ |
            AXP202_PEK_FALLING_EDGE_IRQ |
            AXP202_PEK_RISING_EDGE_IRQ,
            true
        );
        watch->power->clearIRQ();
        Serial.printf(
            "[button] AXP PEK IRQ enabled: INTEN3=%02X INTEN5=%02X\n",
            watch->power->readRegister(AXP202_INTEN3),
            watch->power->readRegister(AXP202_INTEN5)
        );
    }

    display = watch->tft;
    display->setRotation(0);
    display->setSwapBytes(true);
    display->fillScreen(TFT_BLACK);
    drawStaticFace();
    Serial.println("[boot] Display initialized");

    setenv("TZ", kTimezoneLondon, 1);
    tzset();
    syncSystemFromRtc();
    if (connectWifiOrdered(7000)) {
        syncTimeFromNtp();
    }
}

void loop()
{
    maintainWifiAndTime();
    maintainWeather();
    maintainBacklight();
    handleTtsRequest();

    if (backlightOn && drawerOpen) {
        drawDrawer();
    }

    tm info {};
    if (!getLocalTimeInfo(info)) {
        delay(250);
        return;
    }

    if (backlightOn && !drawerOpen) {
        drawStatusIcons();
        drawWatchFace(info);
    }

    delay(35);
}
