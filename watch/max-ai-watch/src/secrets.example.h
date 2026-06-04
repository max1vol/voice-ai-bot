#pragma once

// Copy this file to src/secrets.h and fill in local values.
// src/secrets.h is ignored by git.

constexpr const char *WIFI_SSIDS[] = {
    "first-network",
    "second-network",
};

constexpr const char *WIFI_PASSWORDS[] = {
    "first-password",
    "second-password",
};

constexpr int WIFI_NETWORK_COUNT = sizeof(WIFI_SSIDS) / sizeof(WIFI_SSIDS[0]);
constexpr char OPENAI_API_KEY[] = "sk-...";
constexpr char OPENWEATHER_API_KEY[] = "...";
