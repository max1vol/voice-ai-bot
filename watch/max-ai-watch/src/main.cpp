#include "config.h"
#include "max_ai_face.h"
#include "secrets.h"

#include <ArduinoJson.h>
#include <AudioFileSourceID3.h>
#include <AudioFileSourceSPIFFS.h>
#include <AudioGeneratorMP3.h>
#include <AudioOutputI2S.h>
#include <HTTPClient.h>
#include <Preferences.h>
#include <SPIFFS.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <esp32-hal-bt.h>
#include <esp_sleep.h>
#include <esp_sntp.h>
#include <math.h>
#include <sys/time.h>
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
constexpr uint32_t kActiveWeatherRefreshMs = 30UL * 60UL * 1000UL;
constexpr uint32_t kWeatherRetainMs = 2UL * 60UL * 60UL * 1000UL;
constexpr uint32_t kWeatherRetryMs = 60UL * 1000UL;
constexpr uint64_t kIdleMaintenanceWakeSeconds = 6ULL * 60ULL * 60ULL;
constexpr uint64_t kMicrosPerSecond = 1000000ULL;
constexpr uint32_t kDrawerAutoCloseMs = 10000;
constexpr uint32_t kTouchTapSuppressMs = 600;
constexpr uint32_t kDoubleTapWindowMs = 700;
constexpr uint32_t kTripleTapWindowMs = 900;
constexpr uint32_t kTapMaxDurationMs = 700;
constexpr uint32_t kHeaderErrorMs = 10000;
constexpr uint8_t kMinBrightness = 25;
constexpr int kScreenH = 240;
constexpr int kTtsTapRegionH = kScreenH / 3;
constexpr int kDrawerX = 8;
constexpr int kDrawerW = 224;
constexpr int kDrawerH = kScreenH - (kDrawerX * 2);
constexpr int kDrawerY = (kScreenH - kDrawerH) / 2;
constexpr int kDrawerCloseButtonR = 15;
constexpr int kDrawerCloseButtonX = kDrawerCloseButtonR;
constexpr int kDrawerCloseButtonY = kDrawerCloseButtonR;
constexpr int kDrawerCloseHitR = 30;
constexpr int kSliderX = 38;
constexpr int kSliderW = 164;
constexpr int kSliderTrackH = 10;
constexpr int kSliderKnobR = 12;
constexpr int kBrightnessSliderY = kDrawerY + 62;
constexpr int kVolumeSliderY = kDrawerY + 132;
constexpr int kVoiceToggleY = kDrawerY + 196;
constexpr int kTapMaxTravel = 48;
constexpr char kSpeechPath[] = "/speech.mp3";
constexpr char kHeaderDefaultText[] = "Max AI Watch";
constexpr char kOpenAiSpeechHost[] = "api.openai.com";
constexpr char kOpenAiSpeechPath[] = "/v1/audio/speech";
constexpr uint32_t kRtcUtcFormatVersion = 20260630;
constexpr char kTimePreferencesNamespace[] = "max-ai-time";
constexpr char kRtcFormatPreferenceKey[] = "rtc-utc-ver";

TTGOClass *watch = nullptr;
TFT_eSPI *display = nullptr;

bool timeSynced = false;
bool usedRtcFallback = false;
volatile uint32_t ntpSyncSequence = 0;
uint32_t appliedNtpSyncSequence = 0;
int lastMinuteOfDay = -1;
int lastYearDay = -1;
bool backlightOn = true;
bool touchWasDown = false;
bool touchConsumed = false;
bool ttsRequested = false;
bool ttsBusy = false;
uint32_t lastWifiAttemptMs = 0;
uint32_t lastSerialStatusMs = 0;
uint32_t lastBacklightWakeMs = 0;
uint32_t lastWeatherAttemptMs = 0;
uint32_t suppressTouchTapUntilMs = 0;
uint32_t suppressVoiceToggleUntilMs = 0;
uint32_t touchStartMs = 0;
uint32_t lastTapMs = 0;
uint32_t headerStatusUntilMs = 0;
char lastStatusText[48] = "";
char headerStatusText[24] = "Max AI Watch";
char lastHeaderStatusText[24] = "";
RTC_DATA_ATTR int nextWifiNetworkIndex = 0;
int activeWifiNetworkIndex = -1;
volatile uint8_t lastWifiDisconnectReason = 0;
volatile bool wifiDisconnectReasonAvailable = false;
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
bool headerStatusError = false;
bool lastHeaderStatusError = false;
bool lastWeatherFetchNetworkError = false;
bool wifiConnectionFailed = false;
uint8_t tapCount = 0;
uint32_t drawerLastInteractionMs = 0;
esp_sleep_wakeup_cause_t bootWakeCause = ESP_SLEEP_WAKEUP_UNDEFINED;

enum class TapRegion : uint8_t {
    None,
    Header,
    Drawer
};

TapRegion activeTapRegion = TapRegion::None;

struct WeatherStatus {
    bool valid;
    int temperatureC;
    bool rainToday;
    char condition[20];
    uint32_t fetchedMs;
    time_t fetchedEpoch;
};

WeatherStatus cachedWeather {false, 0, false, "", 0, 0};
RTC_DATA_ATTR WeatherStatus rtcCachedWeather {false, 0, false, "", 0, 0};
RTC_DATA_ATTR time_t rtcLastNtpSyncEpoch = 0;
RTC_DATA_ATTR uint32_t rtcUtcFormatVersion = 0;

void refreshStatusIcons(bool force = false);
void serviceUiDuringBlocking();
void delayWithUi(uint32_t durationMs);

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

uint16_t dimColor(uint8_t r, uint8_t g, uint8_t b)
{
    return display->color565(r, g, b);
}

void drawPanel(int16_t x, int16_t y, int16_t w, int16_t h, uint16_t border)
{
    display->fillRoundRect(x, y, w, h, 8, TFT_BLACK);
    display->drawRoundRect(x, y, w, h, 8, border);
}

void drawHeaderStatus(bool force = false)
{
    if (display == nullptr || drawerOpen) {
        return;
    }
    if (
        !force &&
        strcmp(headerStatusText, lastHeaderStatusText) == 0 &&
        headerStatusError == lastHeaderStatusError
    ) {
        return;
    }

    strlcpy(lastHeaderStatusText, headerStatusText, sizeof(lastHeaderStatusText));
    lastHeaderStatusError = headerStatusError;

    display->fillRect(58, 22, 124, 18, TFT_BLACK);
    display->setTextDatum(MC_DATUM);
    display->setTextColor(
        headerStatusError ? dimColor(255, 68, 82) : dimColor(198, 245, 255),
        TFT_BLACK
    );
    display->drawString(headerStatusText, 120, 31, 2);
}

void setHeaderStatus(const char *text, bool isError = false, uint32_t durationMs = 0)
{
    strlcpy(headerStatusText, text, sizeof(headerStatusText));
    headerStatusError = isError;
    headerStatusUntilMs = isError && durationMs > 0 ? millis() + durationMs : 0;
    if (backlightOn) {
        drawHeaderStatus();
    }
}

void clearHeaderStatus()
{
    setHeaderStatus(kHeaderDefaultText);
}

void setHeaderError(const char *text)
{
    setHeaderStatus(text, true, kHeaderErrorMs);
}

void maintainHeaderStatus()
{
    if (headerStatusError && headerStatusUntilMs > 0 && millis() >= headerStatusUntilMs) {
        clearHeaderStatus();
    } else if (backlightOn) {
        drawHeaderStatus();
    }
}

void drawStaticFace()
{
    display->pushImage(0, 0, MAX_AI_FACE_WIDTH, MAX_AI_FACE_HEIGHT, maxAiFace);
    drawPanel(16, 14, 208, 34, dimColor(24, 90, 120));
    drawPanel(15, 68, 210, 76, dimColor(35, 130, 170));
    drawPanel(28, 168, 184, 44, dimColor(95, 72, 36));
    drawHeaderStatus(true);
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

time_t currentEpoch()
{
    const time_t now = time(nullptr);
    return now > 1700000000 ? now : 0;
}

bool weatherCacheFresh(uint32_t maxAgeMs)
{
    if (!cachedWeather.valid) {
        return false;
    }

    const time_t now = currentEpoch();
    const uint32_t maxAgeSeconds = maxAgeMs / 1000UL;
    if (now > 0 && cachedWeather.fetchedEpoch > 0) {
        return now >= cachedWeather.fetchedEpoch &&
               static_cast<uint32_t>(now - cachedWeather.fetchedEpoch) < maxAgeSeconds;
    }

    return cachedWeather.fetchedMs > 0 && millis() - cachedWeather.fetchedMs < maxAgeMs;
}

void persistWeatherCache()
{
    rtcCachedWeather = cachedWeather;
}

void restoreWeatherCache()
{
    if (!rtcCachedWeather.valid || rtcCachedWeather.fetchedEpoch == 0) {
        return;
    }

    cachedWeather = rtcCachedWeather;
    cachedWeather.fetchedMs = millis();
    if (!weatherCacheFresh(kWeatherRetainMs)) {
        cachedWeather.valid = false;
        Serial.println("[weather] RTC cache expired");
        return;
    }

    Serial.printf(
        "[weather] restored RTC cache: %dC condition=%s precipitation=%s\n",
        cachedWeather.temperatureC,
        cachedWeather.condition,
        cachedWeather.rainToday ? "yes" : "no"
    );
}

bool ntpSyncDue()
{
    const time_t now = currentEpoch();
    if (now == 0 || rtcLastNtpSyncEpoch == 0) {
        return true;
    }
    return static_cast<uint64_t>(now - rtcLastNtpSyncEpoch) >= kIdleMaintenanceWakeSeconds;
}

bool externalPowerPresent()
{
    if (watch == nullptr || watch->power == nullptr) {
        return false;
    }
    return watch->power->isVBUSPlug();
}

void shutdownRadio(const char *reason)
{
    if (WiFi.status() == WL_CONNECTED || WiFi.getMode() != WIFI_OFF) {
        Serial.print("[power] radio off: ");
        Serial.println(reason);
        WiFi.disconnect(true, false);
        WiFi.mode(WIFI_OFF);
    }
    btStop();
    cachedWifiBars = -1;
    wifiIconCacheReady = false;
    lastDrawnWifiBars = -99;
    activeWifiNetworkIndex = -1;
    refreshStatusIcons(true);
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

    if (bars < 0 && wifiConnectionFailed) {
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

void refreshStatusIcons(bool force)
{
    if (display == nullptr || !backlightOn || drawerOpen) {
        return;
    }

    if (force) {
        wifiIconCacheReady = false;
        batteryCacheReady = false;
        lastDrawnWifiBars = -99;
        lastDrawnBatteryPercent = -99;
        lastDrawnBatteryValid = false;
        lastDrawnBatteryCharging = false;
        lastDrawnExternalPower = false;
    }
    drawStatusIcons();
}

bool getLocalTimeInfo(tm &info, uint32_t timeoutMs = 10)
{
    return getLocalTime(&info, timeoutMs);
}

void onNtpTimeAvailable(timeval *tv)
{
    (void)tv;
    ntpSyncSequence++;
}

uint32_t persistedRtcUtcFormatVersion()
{
    Preferences preferences;
    if (!preferences.begin(kTimePreferencesNamespace, true)) {
        Serial.println("[time] RTC format preferences unavailable");
        return 0;
    }

    const uint32_t version = preferences.getUInt(kRtcFormatPreferenceKey, 0);
    preferences.end();
    return version;
}

void markRtcAsUtc()
{
    rtcUtcFormatVersion = kRtcUtcFormatVersion;

    Preferences preferences;
    if (!preferences.begin(kTimePreferencesNamespace, false)) {
        Serial.println("[time] RTC UTC marker could not be persisted");
        return;
    }

    if (preferences.getUInt(kRtcFormatPreferenceKey, 0) != kRtcUtcFormatVersion) {
        preferences.putUInt(kRtcFormatPreferenceKey, kRtcUtcFormatVersion);
    }
    preferences.end();
}

void syncRtcFromSystem()
{
    if (watch->rtc == nullptr) {
        return;
    }

    time_t now = 0;
    time(&now);
    if (now < 1600000000) {
        Serial.println("[time] RTC update skipped: system time is invalid");
        return;
    }

    tm utcInfo {};
    gmtime_r(&now, &utcInfo);
    watch->rtc->setDateTime(
        utcInfo.tm_year + 1900,
        utcInfo.tm_mon + 1,
        utcInfo.tm_mday,
        utcInfo.tm_hour,
        utcInfo.tm_min,
        utcInfo.tm_sec
    );
    markRtcAsUtc();
    Serial.println("[time] RTC updated from NTP time in UTC");
}

bool recordObservedNtpSync(const char *source)
{
    const uint32_t sequence = ntpSyncSequence;
    if (sequence == 0 || sequence == appliedNtpSyncSequence) {
        return false;
    }

    tm info {};
    if (!getLocalTimeInfo(info, 100)) {
        return false;
    }

    appliedNtpSyncSequence = sequence;
    syncRtcFromSystem();
    timeSynced = true;
    usedRtcFallback = false;
    rtcLastNtpSyncEpoch = currentEpoch();
    lastMinuteOfDay = -1;
    lastYearDay = -1;
    lastStatusText[0] = '\0';

    char stamp[32];
    strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S %Z", &info);
    Serial.print("[time] NTP synced via ");
    Serial.print(source);
    Serial.print(": ");
    Serial.println(stamp);
    return true;
}

bool syncTimeFromNtp()
{
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[time] NTP skipped: WiFi is not connected");
        return false;
    }

    if (recordObservedNtpSync("pending")) {
        return true;
    }

    Serial.println("[time] Starting NTP sync for Europe/London");
    const uint32_t startedSequence = ntpSyncSequence;
    setenv("TZ", kTimezoneLondon, 1);
    tzset();
    sntp_set_time_sync_notification_cb(onNtpTimeAvailable);
    configTzTime(kTimezoneLondon, kNtpServer1, kNtpServer2, kNtpServer3);
    sntp_set_time_sync_notification_cb(onNtpTimeAvailable);

    for (int i = 0; i < 40; ++i) {
        if (ntpSyncSequence != startedSequence) {
            if (recordObservedNtpSync("network")) {
                return true;
            }
            if (appliedNtpSyncSequence == ntpSyncSequence && timeSynced) {
                return true;
            }
        }
        delayWithUi(250);
    }
    Serial.println("[time] NTP sync failed: no SNTP response; will retry");
    return false;
}

bool syncSystemFromRtc()
{
    if (watch->rtc == nullptr || !watch->rtc->isValid()) {
        Serial.println("[time] RTC fallback unavailable: RTC is missing or invalid");
        return false;
    }

    const bool rtcKnownUtc =
        rtcUtcFormatVersion == kRtcUtcFormatVersion ||
        persistedRtcUtcFormatVersion() == kRtcUtcFormatVersion;
    if (rtcKnownUtc) {
        rtcUtcFormatVersion = kRtcUtcFormatVersion;
    } else {
        Serial.println(
            "[time] RTC UTC marker missing; assuming valid RTC is UTC until NTP corrects it"
        );
    }

    RTC_Date dt = watch->rtc->getDateTime();
    tm utcInfo {};
    utcInfo.tm_year = dt.year - 1900;
    utcInfo.tm_mon = dt.month - 1;
    utcInfo.tm_mday = dt.day;
    utcInfo.tm_hour = dt.hour;
    utcInfo.tm_min = dt.minute;
    utcInfo.tm_sec = dt.second;
    utcInfo.tm_isdst = 0;

    setenv("TZ", "UTC0", 1);
    tzset();
    time_t epoch = mktime(&utcInfo);
    setenv("TZ", kTimezoneLondon, 1);
    tzset();
    if (epoch < 1600000000) {
        Serial.println("[time] RTC fallback rejected: invalid UTC time");
        return false;
    }

    timeval tv {};
    tv.tv_sec = epoch;
    tv.tv_usec = 0;
    settimeofday(&tv, nullptr);
    usedRtcFallback = true;
    Serial.println("[time] System time loaded from RTC fallback in UTC");
    return true;
}

void maintainNtpSync()
{
    if (ntpSyncSequence != appliedNtpSyncSequence) {
        recordObservedNtpSync("async");
    }
}

const char *wifiDisconnectReasonName(uint8_t reason)
{
    switch (reason) {
    case WIFI_REASON_NO_AP_FOUND:
        return "no AP found";
    case WIFI_REASON_AUTH_FAIL:
        return "authentication failed";
    case WIFI_REASON_ASSOC_FAIL:
        return "association failed";
    case WIFI_REASON_HANDSHAKE_TIMEOUT:
        return "handshake timed out";
    case WIFI_REASON_CONNECTION_FAIL:
        return "connection failed";
    case WIFI_REASON_BEACON_TIMEOUT:
        return "beacon timed out";
    case WIFI_REASON_MISSING_ACKS:
        return "missing ACKs";
    case WIFI_REASON_TIMEOUT:
        return "timed out";
    default:
        return "other";
    }
}

void onWifiEvent(WiFiEvent_t event, WiFiEventInfo_t info)
{
    if (event != ARDUINO_EVENT_WIFI_STA_DISCONNECTED) {
        return;
    }

    lastWifiDisconnectReason = info.wifi_sta_disconnected.reason;
    wifiDisconnectReasonAvailable = true;
}

void connectWifiNetwork(int networkIndex)
{
    if (WiFi.status() == WL_CONNECTED) {
        return;
    }
    if (WIFI_NETWORK_COUNT <= 0) {
        Serial.println("[wifi] No configured networks");
        return;
    }

    if (networkIndex < 0 || networkIndex >= WIFI_NETWORK_COUNT) {
        Serial.println("[wifi] Invalid network index");
        return;
    }

    WiFi.mode(WIFI_STA);
    // Network use is brief and the radio is shut down after each task. Keeping
    // modem sleep disabled here avoids missed association acknowledgements.
    WiFi.setSleep(false);
    WiFi.disconnect(false, false);
    delayWithUi(150);

    activeWifiNetworkIndex = networkIndex;
    lastWifiDisconnectReason = 0;
    wifiDisconnectReasonAvailable = false;
    wifiConnectionFailed = false;

    Serial.print("[wifi] Connecting to ");
    Serial.println(WIFI_SSIDS[activeWifiNetworkIndex]);
    WiFi.begin(WIFI_SSIDS[activeWifiNetworkIndex], WIFI_PASSWORDS[activeWifiNetworkIndex]);
    lastWifiAttemptMs = millis();
    wifiIconCacheReady = false;
    refreshStatusIcons(true);
}

bool waitForWifi(uint32_t timeoutMs)
{
    const uint32_t started = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - started < timeoutMs) {
        delayWithUi(250);
    }

    if (WiFi.status() == WL_CONNECTED) {
        nextWifiNetworkIndex = activeWifiNetworkIndex;
        wifiConnectionFailed = false;
        Serial.print("[wifi] Connected to ");
        Serial.print(WiFi.SSID());
        Serial.print(", IP=");
        Serial.println(WiFi.localIP());
        refreshStatusIcons(true);
        return true;
    }

    if (activeWifiNetworkIndex >= 0 && activeWifiNetworkIndex < WIFI_NETWORK_COUNT) {
        Serial.print("[wifi] Timed out on ");
        Serial.println(WIFI_SSIDS[activeWifiNetworkIndex]);
    } else {
        Serial.println("[wifi] Connection timed out");
    }
    Serial.print("[wifi] Final status=");
    Serial.print(static_cast<int>(WiFi.status()));
    if (wifiDisconnectReasonAvailable) {
        const uint8_t reason = lastWifiDisconnectReason;
        Serial.print(" disconnect reason=");
        Serial.print(reason);
        Serial.print(" (");
        Serial.print(wifiDisconnectReasonName(reason));
        Serial.println(")");
    } else {
        Serial.println(" without a disconnect reason");
    }
    return false;
}

bool connectWifiOrdered(uint32_t perNetworkTimeoutMs)
{
    if (WIFI_NETWORK_COUNT <= 0) {
        Serial.println("[wifi] No configured networks");
        wifiConnectionFailed = true;
        shutdownRadio("no configured networks");
        return false;
    }

    if (nextWifiNetworkIndex < 0 || nextWifiNetworkIndex >= WIFI_NETWORK_COUNT) {
        nextWifiNetworkIndex = 0;
    }
    const int preferredNetworkIndex = nextWifiNetworkIndex;

    // The access point can occasionally miss the ESP32's first association.
    // Retry the last successful network before falling back to other SSIDs.
    for (int attempt = 0; attempt < 2; ++attempt) {
        connectWifiNetwork(preferredNetworkIndex);
        if (waitForWifi(perNetworkTimeoutMs)) {
            return true;
        }
        if (attempt == 0) {
            Serial.println("[wifi] Retrying preferred network");
            delayWithUi(500);
        }
    }

    for (int offset = 1; offset < WIFI_NETWORK_COUNT; ++offset) {
        const int networkIndex = (preferredNetworkIndex + offset) % WIFI_NETWORK_COUNT;
        connectWifiNetwork(networkIndex);
        if (waitForWifi(perNetworkTimeoutMs)) {
            return true;
        }
    }

    Serial.println("[wifi] All configured networks failed");
    wifiConnectionFailed = true;
    shutdownRadio("connect failed");
    return false;
}

const char *syncLabel();

void maintainWifiAndTime()
{
    maintainNtpSync();
    const uint32_t nowMs = millis();

    if (WiFi.status() == WL_CONNECTED && !ttsBusy) {
        shutdownRadio("idle");
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

bool isMistCondition(const char *condition)
{
    return strcmp(condition, "Mist") == 0 ||
           strcmp(condition, "Fog") == 0 ||
           strcmp(condition, "Haze") == 0;
}

bool fetchWeatherForecast()
{
    lastWeatherFetchNetworkError = false;
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[weather] skipped: WiFi is not connected");
        lastWeatherFetchNetworkError = true;
        return false;
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
        lastWeatherFetchNetworkError = true;
        return false;
    }

    Serial.println("[weather] fetching Cambridge forecast");
    const int code = http.GET();
    if (code != HTTP_CODE_OK) {
        Serial.print("[weather] HTTP error: ");
        Serial.println(code);
        http.end();
        if (code <= 0) {
            lastWeatherFetchNetworkError = true;
        }
        return false;
    }

    DynamicJsonDocument doc(24576);
    DeserializationError error = deserializeJson(doc, http.getStream());
    http.end();
    if (error) {
        Serial.print("[weather] JSON parse failed: ");
        Serial.println(error.c_str());
        return false;
    }

    JsonArray list = doc["list"].as<JsonArray>();
    if (list.isNull() || list.size() == 0) {
        Serial.println("[weather] empty forecast");
        return false;
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
    cachedWeather.fetchedEpoch = currentEpoch();
    persistWeatherCache();

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
    if (weatherCacheFresh(kWeatherCacheMs)) {
        return true;
    }
    return fetchWeatherForecast();
}

String weatherSpeechClause()
{
    if (!cachedWeather.valid) {
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
    if (!backlightOn || drawerOpen || ttsBusy) {
        return;
    }
    if (weatherCacheFresh(kActiveWeatherRefreshMs)) {
        return;
    }
    if (lastWeatherAttemptMs > 0 && millis() - lastWeatherAttemptMs < kWeatherRetryMs) {
        return;
    }

    lastWeatherAttemptMs = millis();
    if (!connectWifiOrdered(5000)) {
        return;
    }
    if (fetchWeatherForecast()) {
        lastStatusText[0] = '\0';
    }
    shutdownRadio("weather");
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

enum class WeatherIconKind : uint8_t {
    Unknown,
    Sun,
    Cloud,
    Rain,
    Snow,
    Thunder,
    Mist
};

WeatherIconKind currentWeatherIconKind()
{
    if (!cachedWeather.valid) {
        return WeatherIconKind::Unknown;
    }
    if (strcmp(cachedWeather.condition, "Thunderstorm") == 0) {
        return WeatherIconKind::Thunder;
    }
    if (strcmp(cachedWeather.condition, "Snow") == 0) {
        return WeatherIconKind::Snow;
    }
    if (cachedWeather.rainToday || isWetCondition(cachedWeather.condition)) {
        return WeatherIconKind::Rain;
    }
    if (isMistCondition(cachedWeather.condition)) {
        return WeatherIconKind::Mist;
    }
    if (strcmp(cachedWeather.condition, "Clear") == 0) {
        return WeatherIconKind::Sun;
    }
    if (isCloudyCondition(cachedWeather.condition)) {
        return WeatherIconKind::Cloud;
    }
    return WeatherIconKind::Cloud;
}

int degreeOffsetY(int font)
{
    if (font >= 4) {
        return 12;
    }
    if (font >= 2) {
        return 8;
    }
    return 6;
}

int temperatureLabelWidth(const char *digits, int font)
{
    return display->textWidth(digits, font) + display->textWidth("C", font) + 7;
}

int statusRowFont(const char *dateText, const char *temperatureDigits)
{
    constexpr int kStatusRowW = 156;
    constexpr int kWeatherIconW = 20;
    constexpr int kWeatherGap = 8;
    constexpr int kMinMiddleGap = 12;
    const int fonts[] = {4, 2, 1};

    for (int font : fonts) {
        const int dateW = display->textWidth(dateText, font);
        const int weatherW = temperatureLabelWidth(temperatureDigits, font) + kWeatherGap + kWeatherIconW;
        if (dateW + weatherW + kMinMiddleGap <= kStatusRowW) {
            return font;
        }
    }
    return 1;
}

void drawCloudShape(int16_t cx, int16_t cy, uint16_t color)
{
    display->fillCircle(cx - 6, cy + 2, 5, color);
    display->fillCircle(cx, cy - 2, 6, color);
    display->fillCircle(cx + 7, cy + 3, 4, color);
    display->fillRoundRect(cx - 11, cy + 2, 22, 8, 3, color);
}

void drawWeatherIcon(int16_t cx, int16_t cy)
{
    const uint16_t sun = dimColor(255, 202, 72);
    const uint16_t cloud = dimColor(150, 170, 180);
    const uint16_t rain = dimColor(76, 188, 255);
    const uint16_t snow = dimColor(220, 250, 255);
    const uint16_t mist = dimColor(120, 205, 220);

    switch (currentWeatherIconKind()) {
    case WeatherIconKind::Sun:
        display->fillCircle(cx, cy, 5, sun);
        display->drawLine(cx, cy - 10, cx, cy - 8, sun);
        display->drawLine(cx, cy + 8, cx, cy + 10, sun);
        display->drawLine(cx - 10, cy, cx - 8, cy, sun);
        display->drawLine(cx + 8, cy, cx + 10, cy, sun);
        display->drawLine(cx - 7, cy - 7, cx - 5, cy - 5, sun);
        display->drawLine(cx + 5, cy + 5, cx + 7, cy + 7, sun);
        display->drawLine(cx + 7, cy - 7, cx + 5, cy - 5, sun);
        display->drawLine(cx - 5, cy + 5, cx - 7, cy + 7, sun);
        break;
    case WeatherIconKind::Rain:
        drawCloudShape(cx, cy - 3, cloud);
        display->drawLine(cx - 7, cy + 8, cx - 9, cy + 12, rain);
        display->drawLine(cx, cy + 8, cx - 2, cy + 13, rain);
        display->drawLine(cx + 7, cy + 8, cx + 5, cy + 12, rain);
        break;
    case WeatherIconKind::Snow:
        drawCloudShape(cx, cy - 3, cloud);
        display->drawLine(cx - 6, cy + 11, cx + 6, cy + 11, snow);
        display->drawLine(cx, cy + 5, cx, cy + 17, snow);
        display->drawLine(cx - 5, cy + 6, cx + 5, cy + 16, snow);
        display->drawLine(cx + 5, cy + 6, cx - 5, cy + 16, snow);
        break;
    case WeatherIconKind::Thunder:
        drawCloudShape(cx, cy - 3, cloud);
        display->fillTriangle(cx + 1, cy + 5, cx - 4, cy + 15, cx + 2, cy + 12, sun);
        display->fillTriangle(cx + 2, cy + 9, cx + 8, cy + 9, cx, cy + 18, sun);
        break;
    case WeatherIconKind::Mist:
        display->drawFastHLine(cx - 10, cy - 6, 20, mist);
        display->drawFastHLine(cx - 7, cy, 17, cloud);
        display->drawFastHLine(cx - 10, cy + 6, 20, mist);
        break;
    case WeatherIconKind::Cloud:
        drawCloudShape(cx, cy - 1, cloud);
        break;
    case WeatherIconKind::Unknown:
        display->drawCircle(cx, cy, 7, dimColor(110, 125, 130));
        display->drawFastHLine(cx - 4, cy, 8, dimColor(110, 125, 130));
        break;
    }
}

void drawTemperatureLabel(int16_t rightX, int16_t centerY, const char *digits, int font, uint16_t color)
{
    const int digitW = display->textWidth(digits, font);
    const int labelW = temperatureLabelWidth(digits, font);
    const int leftX = rightX - labelW;
    const int degreeX = leftX + digitW + 3;

    display->setTextDatum(ML_DATUM);
    display->setTextColor(color, TFT_BLACK);
    display->drawString(digits, leftX, centerY, font);
    display->drawCircle(degreeX, centerY - degreeOffsetY(font), 2, color);
    display->drawString("C", degreeX + 5, centerY, font);
}

void redrawBeforeWake();
void wakeBacklight(const char *reason);

void resetDrawState()
{
    lastMinuteOfDay = -1;
    lastYearDay = -1;
    lastStatusText[0] = '\0';
    lastHeaderStatusText[0] = '\0';
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
    drawHeaderStatus();

    const int minuteOfDay = info.tm_hour * 60 + info.tm_min;
    if (minuteOfDay != lastMinuteOfDay) {
        lastMinuteOfDay = minuteOfDay;
        display->fillRect(35, 79, 170, 54, TFT_BLACK);
        display->setTextColor(TFT_WHITE, TFT_BLACK);
        display->drawString(timeText, 120, 106, 7);
    }

    char temperatureDigits[8];
    if (cachedWeather.valid) {
        snprintf(temperatureDigits, sizeof(temperatureDigits), "%d", cachedWeather.temperatureC);
    } else {
        strlcpy(temperatureDigits, "--", sizeof(temperatureDigits));
    }

    char statusText[48];
    snprintf(
        statusText,
        sizeof(statusText),
        "%s|%s|%s|%d",
        dateText,
        temperatureDigits,
        cachedWeather.valid ? cachedWeather.condition : "",
        cachedWeather.rainToday ? 1 : 0
    );
    if (info.tm_yday != lastYearDay || strcmp(statusText, lastStatusText) != 0) {
        lastYearDay = info.tm_yday;
        strlcpy(lastStatusText, statusText, sizeof(lastStatusText));

        constexpr int kRowX = 42;
        constexpr int kRowY = 190;
        constexpr int kWeatherIconX = 191;
        constexpr int kTemperatureRight = 175;
        const int rowFont = statusRowFont(dateText, temperatureDigits);

        display->fillRect(39, 176, 163, 29, TFT_BLACK);
        display->setTextDatum(ML_DATUM);
        display->setTextColor(dimColor(255, 202, 112), TFT_BLACK);
        display->drawString(dateText, kRowX, kRowY, rowFont);
        drawTemperatureLabel(kTemperatureRight, kRowY, temperatureDigits, rowFont, dimColor(132, 220, 232));
        drawWeatherIcon(kWeatherIconX, kRowY);
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
    activeTapRegion = TapRegion::None;
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
    activeTapRegion = TapRegion::None;
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
    activeTapRegion = TapRegion::None;
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

void enterBatteryDeepSleep(const char *reason)
{
    Serial.print("[power] deep sleep: ");
    Serial.println(reason);
    shutdownRadio(reason);

    if (watch != nullptr) {
        watch->closeBL();
        watch->displaySleep();
        watch->powerOff();
    }

    pinMode(TOUCH_INT, INPUT);
    esp_sleep_enable_ext1_wakeup(GPIO_SEL_38, ESP_EXT1_WAKEUP_ALL_LOW);
    esp_sleep_enable_timer_wakeup(kIdleMaintenanceWakeSeconds * kMicrosPerSecond);
    Serial.flush();
    esp_deep_sleep_start();
}

void wakeBacklight(const char *reason)
{
    lastBacklightWakeMs = millis();
    if (backlightOn) {
        return;
    }

    watch->displayWakeup();
    watch->openBL();
    watch->setBrightness(currentBrightness);
    backlightOn = true;
    redrawBeforeWake();
    Serial.print("[display] Backlight on: ");
    Serial.println(reason);
}

void sleepBacklight()
{
    if (!backlightOn) {
        return;
    }

    shutdownRadio("display idle");
    watch->closeBL();
    watch->displaySleep();
    watch->powerOff();
    backlightOn = false;
    Serial.println("[display] Backlight off");
    if (!externalPowerPresent()) {
        enterBatteryDeepSleep("display idle");
    }
}

void resetTapSequence()
{
    activeTapRegion = TapRegion::None;
    tapCount = 0;
}

TapRegion tapRegionForY(int16_t y)
{
    return y < kTtsTapRegionH ? TapRegion::Header : TapRegion::Drawer;
}

uint32_t tapWindowForRegion(TapRegion region)
{
    return region == TapRegion::Header ? kDoubleTapWindowMs : kTripleTapWindowMs;
}

void recordTouchTap(int16_t y)
{
    const uint32_t nowMs = millis();
    if (nowMs < suppressTouchTapUntilMs) {
        resetTapSequence();
        return;
    }

    const TapRegion region = tapRegionForY(y);
    if (region != activeTapRegion || nowMs - lastTapMs > tapWindowForRegion(region)) {
        tapCount = 0;
        activeTapRegion = region;
    }

    lastTapMs = nowMs;
    ++tapCount;
    Serial.printf(
        "[ui] %s tap count=%u\n",
        region == TapRegion::Header ? "header" : "drawer",
        tapCount
    );

    if (region == TapRegion::Header && tapCount >= 2) {
        Serial.println("[tts] touch header double tap");
        resetTapSequence();
        wakeBacklight("touch tts");
        requestTts("touch double tap");
        suppressTouchTap(900);
        return;
    }

    if (region == TapRegion::Drawer && tapCount >= 3) {
        Serial.println("[ui] drawer lower triple tap");
        resetTapSequence();
        openDrawer();
        return;
    }
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
            recordTouchTap((touchStartY + touchLastY) / 2);
        }
        touchConsumed = false;
        voiceToggleTouchConsumed = false;
    }

    if (drawerOpen && !touched && millis() - drawerLastInteractionMs > kDrawerAutoCloseMs) {
        closeDrawer();
    }

    touchWasDown = touched;
}

void maintainBacklight()
{
    maintainTouchWake();

    if (!ttsBusy && backlightOn && millis() - lastBacklightWakeMs > kBacklightTimeoutMs) {
        sleepBacklight();
    }
}

void serviceUiDuringBlocking()
{
    if (display == nullptr || watch == nullptr) {
        return;
    }

    maintainNtpSync();
    maintainTouchWake();
    maintainHeaderStatus();
    refreshStatusIcons();
    if (backlightOn && drawerOpen) {
        drawDrawer();
    }
}

void delayWithUi(uint32_t durationMs)
{
    const uint32_t started = millis();
    do {
        serviceUiDuringBlocking();
        const uint32_t elapsed = millis() - started;
        if (elapsed >= durationMs) {
            break;
        }
        const uint32_t remaining = durationMs - elapsed;
        delay(remaining > 35 ? 35 : remaining);
    } while (true);
}

enum class SpeechDownloadResult : uint8_t {
    Ok,
    NetworkError,
    TtsError
};

SpeechDownloadResult downloadSpeechMp3(const String &input)
{
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[tts] skipped: WiFi is not connected");
        return SpeechDownloadResult::NetworkError;
    }

    SPIFFS.remove(kSpeechPath);
    fs::File speech = SPIFFS.open(kSpeechPath, FILE_WRITE);
    if (!speech) {
        Serial.println("[tts] failed to open speech file for writing");
        return SpeechDownloadResult::TtsError;
    }

    WiFiClientSecure client;
    client.setInsecure();
    client.setTimeout(20);

    setHeaderStatus("calling tts...");
    Serial.println("[tts] connecting to OpenAI");
    if (!client.connect(kOpenAiSpeechHost, 443, 15000)) {
        Serial.println("[tts] OpenAI connection failed");
        speech.close();
        SPIFFS.remove(kSpeechPath);
        return SpeechDownloadResult::NetworkError;
    }
    Serial.println("[tts] OpenAI connection established");
    setHeaderStatus("waiting tts...");
    serviceUiDuringBlocking();

    HTTPClient http;
    http.setTimeout(20000);
    http.setReuse(false);
    http.useHTTP10(true);
    if (!http.begin(client, kOpenAiSpeechHost, 443, kOpenAiSpeechPath, true)) {
        Serial.println("[tts] HTTP begin failed");
        speech.close();
        SPIFFS.remove(kSpeechPath);
        return SpeechDownloadResult::NetworkError;
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
    serviceUiDuringBlocking();
    const int code = http.POST(body);
    serviceUiDuringBlocking();
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
        return code <= 0 ? SpeechDownloadResult::NetworkError : SpeechDownloadResult::TtsError;
    }

    Serial.print("[tts] HTTP OK response size: ");
    Serial.println(http.getSize());
    const int written = http.writeToStream(&speech);
    serviceUiDuringBlocking();
    http.end();
    speech.close();

    if (written <= 0) {
        Serial.print("[tts] MP3 write failed, result=");
        Serial.println(written);
        SPIFFS.remove(kSpeechPath);
        return SpeechDownloadResult::TtsError;
    }

    Serial.print("[tts] MP3 bytes saved: ");
    Serial.println(written);
    return SpeechDownloadResult::Ok;
}

bool playSpeechMp3()
{
    if (!SPIFFS.exists(kSpeechPath)) {
        Serial.println("[audio] speech file missing");
        return false;
    }

    Serial.println("[audio] playing speech");
    watch->enableLDO3(true);
    delayWithUi(50);

    AudioFileSourceSPIFFS *file = new AudioFileSourceSPIFFS(kSpeechPath);
    AudioFileSourceID3 *id3 = new AudioFileSourceID3(file);
    AudioOutputI2S *out = new AudioOutputI2S();
    out->SetPinout(TWATCH_DAC_IIS_BCK, TWATCH_DAC_IIS_WS, TWATCH_DAC_IIS_DOUT);
    out->SetGain(speechVolume);

    AudioGeneratorMP3 *mp3 = new AudioGeneratorMP3();
    bool ok = mp3->begin(id3, out);
    float appliedVolume = speechVolume;
    while (ok && mp3->isRunning()) {
        serviceUiDuringBlocking();
        if (fabsf(speechVolume - appliedVolume) > 0.01f) {
            appliedVolume = speechVolume;
            out->SetGain(appliedVolume);
        }
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

    setHeaderStatus("connecting...");
    if (WiFi.status() != WL_CONNECTED && !connectWifiOrdered(10000)) {
        setHeaderError("net error");
        shutdownRadio("tts wifi error");
        ttsBusy = false;
        resetTouchState();
        suppressTouchTap(1200);
        return;
    }

    if (!timeSynced || usedRtcFallback || ntpSyncDue()) {
        setHeaderStatus("time...");
        if (!syncTimeFromNtp() && !timeSynced) {
            setHeaderError("time error");
            shutdownRadio("tts time error");
            ttsBusy = false;
            resetTouchState();
            suppressTouchTap(1200);
            return;
        }
    }

    setHeaderStatus("weather...");
    if (!ensureWeatherForecast()) {
        setHeaderError(lastWeatherFetchNetworkError ? "net error" : "weather error");
        shutdownRadio("tts weather error");
        ttsBusy = false;
        resetTouchState();
        suppressTouchTap(1200);
        return;
    }

    tm info {};
    if (!getLocalTimeInfo(info, 250)) {
        Serial.println("[tts] no valid local time");
        setHeaderError("tts error");
        shutdownRadio("tts time error");
        ttsBusy = false;
        resetTouchState();
        suppressTouchTap(1200);
        return;
    }

    const String phrase = spokenTimePhrase(info);
    const SpeechDownloadResult downloadResult = downloadSpeechMp3(phrase);
    if (downloadResult == SpeechDownloadResult::Ok) {
        shutdownRadio("tts downloaded");
        setHeaderStatus("speaking...");
        if (playSpeechMp3()) {
            clearHeaderStatus();
        } else {
            setHeaderError("tts error");
        }
    } else if (downloadResult == SpeechDownloadResult::NetworkError) {
        shutdownRadio("tts network error");
        setHeaderError("net error");
    } else {
        shutdownRadio("tts error");
        setHeaderError("tts error");
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
    bootWakeCause = esp_sleep_get_wakeup_cause();
    Serial.print("[boot] Wake cause=");
    Serial.println(static_cast<int>(bootWakeCause));

    watch = TTGOClass::getWatch();
    watch->begin();
    WiFi.onEvent(onWifiEvent, ARDUINO_EVENT_WIFI_STA_DISCONNECTED);
    const bool timerMaintenanceWake =
        bootWakeCause == ESP_SLEEP_WAKEUP_TIMER && !externalPowerPresent();
    if (timerMaintenanceWake) {
        backlightOn = false;
        watch->closeBL();
        watch->displaySleep();
    } else {
        watch->displayWakeup();
        watch->openBL();
        watch->setBrightness(currentBrightness);
        lastBacklightWakeMs = millis();
    }

    if (!SPIFFS.begin(true)) {
        Serial.println("[boot] SPIFFS mount failed");
    }

    pinMode(TOUCH_INT, INPUT);

    display = watch->tft;
    display->setRotation(0);
    display->setSwapBytes(true);
    if (!timerMaintenanceWake) {
        display->fillScreen(TFT_BLACK);
        drawStaticFace();
        Serial.println("[boot] Display initialized");
    } else {
        Serial.println("[boot] Display kept asleep for timer maintenance");
    }

    setenv("TZ", kTimezoneLondon, 1);
    tzset();
    const bool rtcLoaded = syncSystemFromRtc();
    restoreWeatherCache();

    if (timerMaintenanceWake) {
        if (ntpSyncDue() && connectWifiOrdered(7000)) {
            syncTimeFromNtp();
            shutdownRadio("timer ntp");
        }
        enterBatteryDeepSleep("timer maintenance complete");
    }

    if (!rtcLoaded || (bootWakeCause == ESP_SLEEP_WAKEUP_UNDEFINED && ntpSyncDue())) {
        if (connectWifiOrdered(7000)) {
            syncTimeFromNtp();
            shutdownRadio("boot ntp");
        }
    }
}

void loop()
{
    maintainWifiAndTime();
    maintainWeather();
    maintainBacklight();
    handleTtsRequest();
    maintainHeaderStatus();

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
