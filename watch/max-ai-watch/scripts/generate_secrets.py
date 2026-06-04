from __future__ import annotations

import ast
from pathlib import Path


def read_key_value_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if "#" in value and not value.startswith(("'", '"')):
            value = value.split("#", 1)[0].strip()

        if value.startswith(("'", '"')):
            try:
                value = ast.literal_eval(value)
            except Exception:
                value = value.strip("'\"")

        values[key] = value

    return values


def c_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def read_numbered_wifi(values: dict[str, str]) -> list[tuple[str, str]]:
    networks: list[tuple[str, str]] = []
    for index in range(1, 10):
        ssid = values.get(f"WIFI_SSID_{index}") or values.get(f"MAX_AI_WATCH_WIFI_SSID_{index}")
        password = values.get(f"WIFI_PASSWORD_{index}") or values.get(
            f"MAX_AI_WATCH_WIFI_PASSWORD_{index}"
        )
        if ssid and password is not None:
            networks.append((ssid, password))
    return networks


def read_pair_list(value: str | None) -> list[tuple[str, str]]:
    if not value:
        return []

    networks: list[tuple[str, str]] = []
    for part in value.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(
                "MAX_AI_WATCH_WIFI_NETWORKS entries must use SSID=password format"
            )
        ssid, password = part.split("=", 1)
        networks.append((ssid.strip(), password.strip()))
    return networks


project_root = Path(__file__).resolve().parents[1]
values: dict[str, str] = {}
for path in (
    project_root / ".env",
    Path.home() / ".wifi",
    Path.home() / ".zshrc-secrets",
):
    values.update(read_key_value_file(path))

wifi_networks = read_pair_list(
    values.get("MAX_AI_WATCH_WIFI_NETWORKS") or values.get("WATCH_WIFI_NETWORKS")
)
if not wifi_networks:
    wifi_networks = read_numbered_wifi(values)

if not wifi_networks:
    ssids = split_csv(values.get("WIFI_SSIDS") or values.get("MAX_AI_WATCH_WIFI_SSIDS"))
    passwords = split_csv(
        values.get("WIFI_PASSWORDS") or values.get("MAX_AI_WATCH_WIFI_PASSWORDS")
    )
    if ssids and len(ssids) == len(passwords):
        wifi_networks = list(zip(ssids, passwords))

if not wifi_networks:
    ssid = values.get("WIFI") or values.get("SSID") or values.get("WIFI_SSID")
    password = (
        values.get("PASSWORD")
        or values.get("PASS")
        or values.get("WIFI_PASSWORD")
    )
    if ssid and password is not None:
        wifi_networks = [(ssid, password)]

openai_key = values.get("OPENAI_API_KEY") or values.get("AI_GATEWAY_API_KEY")
openweather_key = values.get("OPENWEATHER_API_KEY")

missing = []
if not wifi_networks:
    missing.append("Wi-Fi network credentials")
if not openai_key:
    missing.append("OPENAI_API_KEY or AI_GATEWAY_API_KEY")
if not openweather_key:
    missing.append("OPENWEATHER_API_KEY")
if missing:
    raise SystemExit("Missing secret values: " + ", ".join(missing))

ssid_lines = "\n".join(f'    "{c_string(ssid)}",' for ssid, _ in wifi_networks)
password_lines = "\n".join(f'    "{c_string(password)}",' for _, password in wifi_networks)

out = project_root / "src" / "secrets.h"
out.write_text(
    "#pragma once\n\n"
    "// Generated from local secret files. Do not commit this file.\n"
    "constexpr const char *WIFI_SSIDS[] = {\n"
    f"{ssid_lines}\n"
    "};\n\n"
    "constexpr const char *WIFI_PASSWORDS[] = {\n"
    f"{password_lines}\n"
    "};\n\n"
    "constexpr int WIFI_NETWORK_COUNT = "
    "sizeof(WIFI_SSIDS) / sizeof(WIFI_SSIDS[0]);\n"
    f'constexpr char OPENAI_API_KEY[] = "{c_string(openai_key)}";\n'
    f'constexpr char OPENWEATHER_API_KEY[] = "{c_string(openweather_key)}";\n',
    encoding="utf-8",
)

print("generated src/secrets.h")
