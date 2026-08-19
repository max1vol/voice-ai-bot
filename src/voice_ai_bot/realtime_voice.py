from __future__ import annotations

import base64
import json
import logging
import math
import queue
import socket
import threading
import time
import wave
import uuid
from array import array
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import certifi
import websocket
from openai import OpenAI

from .audio_io import PcmOutputStream, PcmPlayer, RawPcmRecorder
from .camera import CameraCapture, CameraError, CameraSnapshot, ContinuousCameraCapture
from .config import Config, REALTIME_SYSTEM_PROMPT
from .conversation import Message
from .memory import MemoryStore
from .music import MusicPlayer
from .scheduled_tasks import ScheduledTaskStore
from .settings import RuntimeSettings
from .weather import OpenWeatherService

LOGGER = logging.getLogger(__name__)


CLOSE_TOOL_NAME = "close_realtime_session"
WEB_SEARCH_TOOL_NAME = "web_search"
START_TASK_TOOL_NAME = "start_background_task"
STEER_TASK_TOOL_NAME = "steer_background_task"
LIST_TASKS_TOOL_NAME = "list_background_tasks"
GET_TASK_TOOL_NAME = "get_background_task"
CANCEL_TASK_TOOL_NAME = "cancel_background_task"
TASK_STATUS_TOOL_NAME = "emit_status_update"
ADD_SCHEDULED_TASK_TOOL_NAME = "add_scheduled_task"
LIST_SCHEDULED_TASKS_TOOL_NAME = "list_scheduled_tasks"
DELETE_SCHEDULED_TASK_TOOL_NAME = "delete_scheduled_task"
GET_WEATHER_TOOL_NAME = "get_weather"
SET_VOICE_VOLUME_TOOL_NAME = "set_voice_volume"
GET_VOICE_VOLUME_TOOL_NAME = "get_voice_volume"
GET_MUSIC_VOLUME_TOOL_NAME = "get_music_volume"
LIST_MUSIC_TOOL_NAME = "list_music"
GET_MUSIC_STATUS_TOOL_NAME = "get_music_status"
PLAY_MUSIC_TOOL_NAME = "play_music"
PAUSE_MUSIC_TOOL_NAME = "pause_music"
RESUME_MUSIC_TOOL_NAME = "resume_music"
STOP_MUSIC_TOOL_NAME = "stop_music"
SET_MUSIC_VOLUME_TOOL_NAME = "set_music_volume"
TAKE_PICTURE_TOOL_NAME = "take_picture"
START_CONTINUOUS_CAMERA_TOOL_NAME = "start_continuous_camera"
STOP_CONTINUOUS_CAMERA_TOOL_NAME = "stop_continuous_camera"
MEMORY_SEARCH_TOOL_NAME = "memory_search"
MEMORY_LIST_TOOL_NAME = "memory_list"
MEMORY_ADD_TOOL_NAME = "memory_add"
MEMORY_UPDATE_TOOL_NAME = "memory_update"
MEMORY_FORGET_TOOL_NAME = "memory_forget"
MEMORY_GET_SOURCE_TOOL_NAME = "memory_get_source"
ASYNC_TASK_TOOL_NAMES = {
    START_TASK_TOOL_NAME,
    STEER_TASK_TOOL_NAME,
    LIST_TASKS_TOOL_NAME,
    GET_TASK_TOOL_NAME,
    CANCEL_TASK_TOOL_NAME,
    WEB_SEARCH_TOOL_NAME,
    ADD_SCHEDULED_TASK_TOOL_NAME,
    LIST_SCHEDULED_TASKS_TOOL_NAME,
    DELETE_SCHEDULED_TASK_TOOL_NAME,
    GET_WEATHER_TOOL_NAME,
    SET_VOICE_VOLUME_TOOL_NAME,
    GET_VOICE_VOLUME_TOOL_NAME,
    GET_MUSIC_VOLUME_TOOL_NAME,
    LIST_MUSIC_TOOL_NAME,
    GET_MUSIC_STATUS_TOOL_NAME,
    PLAY_MUSIC_TOOL_NAME,
    PAUSE_MUSIC_TOOL_NAME,
    RESUME_MUSIC_TOOL_NAME,
    STOP_MUSIC_TOOL_NAME,
    SET_MUSIC_VOLUME_TOOL_NAME,
    TAKE_PICTURE_TOOL_NAME,
    START_CONTINUOUS_CAMERA_TOOL_NAME,
    STOP_CONTINUOUS_CAMERA_TOOL_NAME,
    MEMORY_SEARCH_TOOL_NAME,
    MEMORY_LIST_TOOL_NAME,
    MEMORY_ADD_TOOL_NAME,
    MEMORY_UPDATE_TOOL_NAME,
    MEMORY_FORGET_TOOL_NAME,
    MEMORY_GET_SOURCE_TOOL_NAME,
}


@dataclass(frozen=True)
class RealtimeTurnResult:
    user_text: str
    assistant_text: str
    requested_close: bool


@dataclass
class BackgroundTask:
    id: str
    request_text: str
    history: list[Message] = field(default_factory=list)
    memory_context: str = ""
    title: str = ""
    source: str = "realtime"
    wakeup_on_complete: bool = False
    wakeup_on_progress: bool = False
    wakeup_reported: bool = False
    progress_wakeup_index: int = 0
    status: str = "queued"
    progress: str = "queued"
    response_id: str = ""
    result: str = ""
    reasoning_summary: str = ""
    error: str = ""
    events: list[str] = field(default_factory=list)
    status_updates: list[dict[str, Any]] = field(default_factory=list)
    steering_messages: list[dict[str, Any]] = field(default_factory=list)
    revision: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    active_stream: Any = field(default=None, repr=False)


def local_context(config: Config) -> str:
    try:
        now = datetime.now(ZoneInfo(config.user_timezone))
    except Exception:
        now = datetime.now().astimezone()
    utc_offset = now.strftime("%z")
    formatted_offset = f"{utc_offset[:3]}:{utc_offset[3:]}" if utc_offset else "unknown UTC offset"
    return (
        f"Current local time: {now:%Y-%m-%d %H:%M:%S} {now.tzname()} (UTC{formatted_offset}). "
        f"User location: {config.user_city}, {config.user_region}, {config.user_country}. "
        f"User timezone: {config.user_timezone}."
    )


def realtime_session_instructions(config: Config, memory_context: str = "") -> str:
    parts = [
        REALTIME_SYSTEM_PROMPT,
        (
            "Local context:\n"
            f"- User location: {config.user_city}, {config.user_region}, {config.user_country}.\n"
            "- The application provides current local time and location context on each turn.\n"
            "- Use the user's local time, date, timezone, and location for words like today, tomorrow, local, "
            "nearby, morning, evening, and current."
        ),
        (
            "Response rules:\n"
            "- Identity answers should be brief. The first sentence should say Max Code.\n"
            "- For quick device-control confirmations, use one short sentence, usually under 10 words.\n"
            "- Do not preamble before quick local tool calls.\n"
            "- Only give extra explanation when the user asks for it."
        ),
        (
            "Weather:\n"
            "- Use get_weather for weather questions.\n"
            "- get_weather supports current conditions and forecasts up to 5 days.\n"
            "- If the user wants a forecast, set forecast_days.\n"
            "- If the user names another place, pass location.\n"
            "- If location is omitted, Cambridge, GB is the default.\n"
            "- Set no_cache true only when the user explicitly asks to refresh or recheck.\n"
            "- After a weather result, answer directly instead of saying that you are checking."
        ),
        (
            "Background tasks:\n"
            "- Use start_background_task for current web research, news, opening hours, prices, schedules, "
            "or other time-sensitive multi-step work.\n"
            "- Prefer background tasks for web, code, calculation, or multi-step research so the voice session "
            "stays interruptible.\n"
            "- Do not rewrite the user's request into a task query; start_background_task uses the current user "
            "transcript captured by the application.\n"
            "- If the user clarifies or redirects an existing task, call steer_background_task.\n"
            "- Do not summarize or rewrite steering messages; the application passes the user's latest transcript "
            "verbatim.\n"
            "- Background GPT-5.5 tasks receive the raw user transcript, recent conversation, current local time, "
            "timezone, and location, and can use hosted web search plus code execution.\n"
            "- When a background task starts, briefly tell the user it started and do not wait in the same reply.\n"
            "- Set wakeup_on_complete true only when the user expects the result to be spoken automatically.\n"
            "- Use list_background_tasks or get_background_task when the user asks what is running, asks for "
            "progress, or asks for results."
        ),
        (
            "Scheduled tasks:\n"
            "- The user can ask you to create, list, or remove reminders, alarms, and timed tasks.\n"
            "- Use the scheduled-task tools for that.\n"
            "- Spoken scheduled tasks must not start during quiet hours "
            f"{config.schedule_quiet_start}-{config.schedule_quiet_end} local time.\n"
            "- If the requested time falls in quiet hours, explain that the device will stay quiet then and ask "
            "for another time."
        ),
        (
            "Memory:\n"
            "- Use memory_search or memory_list before answering questions about what you remember, prior "
            "conversations, user preferences, personal facts, or saved decisions.\n"
            "- If the user asks you to remember something, call memory_add.\n"
            "- If the user says memory is wrong, call memory_update.\n"
            "- If the user asks to forget or delete memory, call memory_forget.\n"
            "- Confirm memory changes briefly.\n"
            "- Do not store secrets or credentials.\n"
            "- Do not store runtime device settings such as voice volume, music volume, playback state, selected "
            "song, or camera state in memory; those are handled by device settings or transient runtime state."
        ),
        (
            "Music:\n"
            "- Use list_music to see available songs and their durations.\n"
            "- Use play_music, pause_music, resume_music, stop_music, get_music_status, and set_music_volume for "
            "song playback.\n"
            "- Use set_voice_volume only for your spoken voice volume.\n"
            "- Use get_voice_volume and get_music_volume when the user asks how loud voice or music is set.\n"
            "- Voice and music volume changes persist to the device settings file and survive service restarts.\n"
            "- If the user asks what songs are available, include durations when helpful.\n"
            "- If play_music or resume_music returns deferred true, briefly acknowledge it; the application starts "
            "or resumes music after you finish speaking so music and voice do not overlap."
        ),
        (
            "Vision:\n"
            "- The device may have a webcam looking down at paper or objects near the voice box.\n"
            "- Camera images are attached only when you call a camera tool; button presses do not automatically "
            "take pictures.\n"
            "- Use take_picture when the user asks you to look, inspect, read, see, check the paper, or when a "
            "visual board state matters.\n"
            "- Before calling take_picture, briefly tell the user that the camera will take a picture in a few "
            "seconds. The application waits for the USB camera to settle and plays a shutter cue at capture time.\n"
            "- Use start_continuous_camera when you need ongoing visual context, such as an interactive board game. "
            "It keeps the USB camera open and uploads the latest frame repeatedly until stopped.\n"
            "- Continuous camera capture defaults to every 4 seconds. You can request any interval from 1 to 5 "
            "seconds, and calling start_continuous_camera again changes the rate.\n"
            "- Stop continuous capture when the game, inspection, or visual task is over.\n"
            "- For noughts and crosses / tic-tac-toe, inspect the latest image, decide the best move, and say the "
            "square or position where you will go."
        ),
        (
            "Interruptions:\n"
            "- If interrupted by a new button press, stop the previous reply and treat the new speech as the latest "
            "user turn."
        ),
    ]
    if memory_context.strip():
        parts.append("\n\nPersistent workspace context:\n" + memory_context.strip())
    return "\n\n".join(parts)


def realtime_turn_instructions(config: Config, memory_context: str = "") -> str:
    parts = [
        (
            f"{local_context(config)}\n"
            "Critical rules for this turn:\n"
            "- You are Max Code, a general-purpose voice assistant running on a local Raspberry Pi speaker.\n"
            "- Do not present yourself as ChatGPT.\n"
            "- If the user asks who you are or asks about yourself, answer exactly: "
            "\"I'm Max Code, your general-purpose voice assistant.\"\n"
            "- Answer the user's actual request directly. Do not steer the conversation into shopkeeping, vending, "
            "drinks, sales, or any other special scenario unless the user asks for it.\n"
            "Turn rules:\n"
            "- Answer only the latest user turn.\n"
            "- Keep spoken replies concise.\n"
            "- If a fast local tool result is available, answer directly with no preamble.\n"
            "- For quick device-control confirmations, use one short sentence, usually under 10 words.\n"
            "- Use background and scheduled-task tools when they fit the user's request."
        )
    ]
    if memory_context.strip():
        parts.append(memory_context.strip())
    return "\n\n".join(parts)


def realtime_turn_detection(config: Config) -> dict[str, Any] | None:
    mode = config.realtime_turn_detection.strip().lower()
    if mode in {"", "none", "off", "disabled", "manual"}:
        return None
    if mode == "semantic_vad":
        return {
            "type": "semantic_vad",
            "eagerness": config.realtime_semantic_vad_eagerness,
            "create_response": False,
            "interrupt_response": False,
        }
    return {
        "type": "server_vad",
        "threshold": config.realtime_vad_threshold,
        "prefix_padding_ms": config.realtime_vad_prefix_padding_ms,
        "silence_duration_ms": config.realtime_vad_silence_duration_ms,
        "create_response": False,
        "interrupt_response": False,
    }


def realtime_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": CLOSE_TOOL_NAME,
            "description": (
                "Close the current realtime session when the user asks to stop, disconnect, sleep, "
                "end the conversation for now, or otherwise stop spending on this live connection."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief reason the session should be closed.",
                    }
                },
                "required": ["reason"],
            },
        },
        {
            "type": "function",
            "name": TAKE_PICTURE_TOOL_NAME,
            "description": (
                "Capture one webcam snapshot and attach it to the live conversation. Use this when the user asks "
                "you to look, inspect, read, see, check the paper, or when visual state is needed, such as deciding "
                "a noughts-and-crosses move. Before calling this tool, say a short spoken heads-up that the camera "
                "will take a picture in a few seconds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief reason for taking the picture.",
                    }
                },
            },
        },
        {
            "type": "function",
            "name": START_CONTINUOUS_CAMERA_TOOL_NAME,
            "description": (
                "Start continuous webcam capture, or change the capture rate if it is already running. The app keeps "
                "the USB camera open, continuously reads frames, and attaches the latest image to the live "
                "conversation at the requested interval. Use this for interactive games or tasks where visual state "
                "changes while the user is not pressing the button."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief reason for continuous capture, such as tic-tac-toe or reading a board.",
                    },
                    "interval_seconds": {
                        "type": "number",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "Seconds between uploaded frames. Use 4 by default; allowed range is 1 to 5.",
                    },
                },
            },
        },
        {
            "type": "function",
            "name": STOP_CONTINUOUS_CAMERA_TOOL_NAME,
            "description": "Stop continuous webcam capture when ongoing visual context is no longer needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief reason for stopping continuous capture.",
                    }
                },
            },
        },
        {
            "type": "function",
            "name": START_TASK_TOOL_NAME,
            "description": (
                "Start an asynchronous GPT-5.5 task for current web research, code generation, code execution, "
                "calculation, or multi-step analysis. The task runs in the app while the realtime voice session "
                "continues. The application passes the latest user transcript and recent conversation to GPT-5.5; "
                "do not provide or rewrite the task request yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Optional short human-readable title for the task.",
                    },
                    "wakeup_on_progress": {
                        "type": "boolean",
                        "description": (
                            "Set true only when the user wants important progress updates spoken automatically."
                        ),
                    },
                    "wakeup_on_complete": {
                        "type": "boolean",
                        "description": (
                            "Set true when the device should speak the final result automatically after the task "
                            "finishes. Set false for silent tasks the user will ask about later."
                        ),
                    },
                },
            },
        },
        {
            "type": "function",
            "name": STEER_TASK_TOOL_NAME,
            "description": (
                "Send the latest user transcript as a steering message to an existing background GPT-5.5 task. "
                "Use this when the user clarifies, redirects, narrows, expands, or adds instructions to a task "
                "that is already running or retained. The application passes the transcript verbatim."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Optional task id. If omitted, the newest active task is steered.",
                    },
                    "wakeup_on_complete": {
                        "type": "boolean",
                        "description": "Optionally update whether the final result should be spoken automatically.",
                    },
                },
            },
        },
        {
            "type": "function",
            "name": LIST_TASKS_TOOL_NAME,
            "description": "List background GPT-5.5 tasks, including running status and user-facing progress updates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_completed": {
                        "type": "boolean",
                        "description": "Include completed, failed, and cancelled tasks as well as running tasks.",
                    }
                },
            },
        },
        {
            "type": "function",
            "name": GET_TASK_TOOL_NAME,
            "description": "Get progress, user-facing status updates, steering messages, and final result for a background task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Task id returned by start_background_task. If omitted, returns the newest task.",
                    },
                    "include_result": {
                        "type": "boolean",
                        "description": "Include the task result text. Use true when reporting a completed task.",
                    },
                },
            },
        },
        {
            "type": "function",
            "name": CANCEL_TASK_TOOL_NAME,
            "description": "Cancel a running background GPT-5.5 task when the user no longer wants it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Task id returned by start_background_task.",
                    }
                },
                "required": ["task_id"],
            },
        },
        {
            "type": "function",
            "name": ADD_SCHEDULED_TASK_TOOL_NAME,
            "description": (
                "Create a reminder, alarm, or other timed task. Use this when the user wants the device to do "
                "something at a specific future time. Spoken tasks will be rejected during quiet hours."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short label for the scheduled task, such as 'Wake up' or 'Medication reminder'.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": (
                            "What the assistant should say or do when the time arrives. For a spoken reminder, "
                            "write the message to speak."
                        ),
                    },
                    "run_at": {
                        "type": "string",
                        "description": (
                            "ISO local datetime for when it should run, e.g. 2026-06-02T07:30:00. "
                            "Use the current local time from the turn instructions to resolve relative requests."
                        ),
                    },
                    "action": {
                        "type": "string",
                        "enum": ["speak", "background_task"],
                        "description": "Use speak for alarms/reminders; use background_task for silent research/work.",
                    },
                    "repeat": {
                        "type": "string",
                        "enum": ["once", "daily"],
                        "description": "Use daily for repeating alarms or reminders; otherwise use once.",
                    },
                },
                "required": ["title", "prompt", "run_at"],
            },
        },
        {
            "type": "function",
            "name": LIST_SCHEDULED_TASKS_TOOL_NAME,
            "description": "List reminders, alarms, and other timed tasks so the user can review or discuss them.",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_inactive": {
                        "type": "boolean",
                        "description": "Include completed and skipped items as well as active items.",
                    }
                },
            },
        },
        {
            "type": "function",
            "name": DELETE_SCHEDULED_TASK_TOOL_NAME,
            "description": "Remove a reminder, alarm, or other timed task by id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Scheduled task id returned by add_scheduled_task or list_scheduled_tasks.",
                    }
                },
                "required": ["task_id"],
            },
        },
        {
            "type": "function",
            "name": GET_WEATHER_TOOL_NAME,
            "description": (
                "Get current weather or a forecast from OpenWeather. Use this for weather questions instead of web "
                "search. It defaults to Cambridge, GB when location is omitted, supports any named location, and "
                "caches results for 10 minutes unless no_cache is true."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Location name such as Cambridge, GB. Omit to use the user's configured city.",
                    },
                    "units": {
                        "type": "string",
                        "enum": ["metric", "imperial", "standard"],
                        "description": "Unit system. Use metric by default for Cambridge, UK.",
                    },
                    "forecast_days": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 5,
                        "description": "How many forecast days to return. Omit or use 0 for current-only.",
                    },
                    "no_cache": {
                        "type": "boolean",
                        "description": "Set true only when the user explicitly asks to refresh or recheck.",
                    },
                },
            },
        },
        {
            "type": "function",
            "name": SET_VOICE_VOLUME_TOOL_NAME,
            "description": (
                "Set the assistant's spoken output volume from 1 to 10. This adjusts software PCM volume before "
                "audio is sent to the speaker and persists the setting for future service restarts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Voice volume level, where 1 is quietest and 10 is loudest.",
                    }
                },
                "required": ["level"],
            },
        },
        {
            "type": "function",
            "name": GET_VOICE_VOLUME_TOOL_NAME,
            "description": "Get the assistant's current spoken output volume from 1 to 10.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "type": "function",
            "name": GET_MUSIC_VOLUME_TOOL_NAME,
            "description": "Get the current music playback volume from 1 to 10.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "type": "function",
            "name": LIST_MUSIC_TOOL_NAME,
            "description": (
                "List songs available on the Raspberry Pi. The response includes each song's title, id, and "
                "duration so you can tell the user what is available and how long it is."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "type": "function",
            "name": GET_MUSIC_STATUS_TOOL_NAME,
            "description": "Get the current music playback state, selected song, position, pending action, and volume.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "type": "function",
            "name": PLAY_MUSIC_TOOL_NAME,
            "description": (
                "Play a song from the Raspberry Pi music library. Use the user's requested title or words, "
                "such as baby shark or chopin spring. Playback starts after your spoken acknowledgement finishes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Song title, id, or user phrase to match against the local library.",
                    }
                },
                "required": ["query"],
            },
        },
        {
            "type": "function",
            "name": PAUSE_MUSIC_TOOL_NAME,
            "description": "Pause the currently playing song and keep its position for later resume.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "type": "function",
            "name": RESUME_MUSIC_TOOL_NAME,
            "description": (
                "Resume the selected paused song. Playback starts after your spoken acknowledgement finishes."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "type": "function",
            "name": STOP_MUSIC_TOOL_NAME,
            "description": "Stop music playback and clear the selected song.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "type": "function",
            "name": SET_MUSIC_VOLUME_TOOL_NAME,
            "description": (
                "Set song playback volume from 1 to 10 and persist it for future service restarts. Use this for "
                "music volume, not the assistant's spoken voice."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Music volume level, where 1 is quietest and 10 is loudest.",
                    }
                },
                "required": ["level"],
            },
        },
        {
            "type": "function",
            "name": MEMORY_SEARCH_TOOL_NAME,
            "description": (
                "Search the assistant's persisted memory before answering questions about previous conversations, "
                "stored user facts, preferences, decisions, or what the assistant remembers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query in the user's words.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of memory snippets to return.",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "type": "function",
            "name": MEMORY_LIST_TOOL_NAME,
            "description": "List durable long-term memory entries so the user can review them.",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_forgotten": {
                        "type": "boolean",
                        "description": "Include entries that were explicitly forgotten.",
                    }
                },
            },
        },
        {
            "type": "function",
            "name": MEMORY_ADD_TOOL_NAME,
            "description": (
                "Add a durable memory entry when the user explicitly asks to remember something, or when they "
                "clearly approve saving a stable preference or fact. Do not use this for runtime device settings "
                "such as voice volume or music volume, because those are stored in device settings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "The durable fact, preference, or instruction to remember. Do not store runtime "
                            "device settings such as voice volume or music volume here."
                        ),
                    },
                    "kind": {
                        "type": "string",
                        "description": "Short category such as preference, fact, instruction, project, or note.",
                    },
                },
                "required": ["text"],
            },
        },
        {
            "type": "function",
            "name": MEMORY_UPDATE_TOOL_NAME,
            "description": "Correct an existing durable memory entry when the user says it is wrong or outdated.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entry_id": {
                        "type": "string",
                        "description": "Memory entry id returned by memory_list or memory_search.",
                    },
                    "text": {
                        "type": "string",
                        "description": "Replacement memory text.",
                    },
                },
                "required": ["entry_id", "text"],
            },
        },
        {
            "type": "function",
            "name": MEMORY_FORGET_TOOL_NAME,
            "description": (
                "Forget a durable memory entry when the user asks to delete, remove, or forget it. Use entry_id "
                "when available; otherwise use query to match active entries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entry_id": {
                        "type": "string",
                        "description": "Memory entry id to forget.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Fallback text to match against active memory entries.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief user-facing reason, if any.",
                    },
                },
            },
        },
        {
            "type": "function",
            "name": MEMORY_GET_SOURCE_TOOL_NAME,
            "description": "Read a bounded excerpt from a memory source path returned by memory_search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Memory path returned by memory_search, such as MEMORY.md or memory/YYYY-MM-DD.md.",
                    },
                    "from_line": {
                        "type": "integer",
                        "description": "1-based line number to start reading.",
                    },
                    "lines": {
                        "type": "integer",
                        "description": "Number of lines to read.",
                    },
                },
                "required": ["path"],
            },
        },
    ]


class BackgroundTaskManager:
    def __init__(self, config: Config, memory: MemoryStore | None = None):
        self.config = config
        self.memory = memory
        self.weather = OpenWeatherService(config)
        self.client = OpenAI(api_key=config.openai_api_key, timeout=config.task_timeout_seconds)
        self.tasks_dir = config.memory_dir / "tasks"
        self._tasks: dict[str, BackgroundTask] = {}
        self._lock = threading.RLock()
        self._load_persisted_tasks()

    def start(
        self,
        request_text: str,
        history: Iterable[Message] = (),
        title: str = "",
        wakeup_on_complete: bool = False,
        wakeup_on_progress: bool = False,
        source: str = "realtime",
    ) -> dict[str, Any]:
        request_text = request_text.strip()
        if not request_text:
            return {"ok": False, "error": "current user transcript is required"}
        task_id = f"task_{uuid.uuid4().hex[:10]}"
        clean_title = title.strip()
        memory_context = ""
        if self.memory is not None:
            try:
                memory_context = self.memory.active_context(request_text)
            except Exception:
                LOGGER.exception("failed to collect active memory for background task")
        task = BackgroundTask(
            id=task_id,
            request_text=request_text,
            history=list(history),
            memory_context=memory_context,
            title=clean_title,
            source=source,
            wakeup_on_complete=wakeup_on_complete,
            wakeup_on_progress=wakeup_on_progress,
        )
        if clean_title:
            task.progress = f"queued: {clean_title}"
        with self._lock:
            self._prune_locked()
            if len(self._tasks) >= self.config.max_background_tasks:
                return {"ok": False, "error": "too many retained background tasks; cancel or wait for older tasks"}
            self._tasks[task_id] = task
            self._persist_task_locked(task)
        thread = threading.Thread(target=self._run_task, args=(task_id,), name=f"task-{task_id}", daemon=True)
        thread.start()
        LOGGER.info("started background task %s wakeup=%s: %s", task_id, wakeup_on_complete, request_text)
        return {"ok": True, "task": self._snapshot(task, include_result=False)}

    def steer(
        self,
        message: str,
        task_id: str = "",
        wakeup_on_complete: bool | None = None,
    ) -> dict[str, Any]:
        clean = message.strip()
        if not clean:
            return {"ok": False, "error": "current user transcript is required for steering"}
        should_restart = False
        stream_to_close = None
        with self._lock:
            task = self._get_locked(task_id, prefer_active=True)
            if task is None:
                return {"ok": False, "error": f"unknown task id: {task_id or '<latest active>'}"}
            task.steering_messages.append(
                {
                    "id": f"steer_{uuid.uuid4().hex[:10]}",
                    "text": clean,
                    "created_at": _iso_time(time.time()),
                }
            )
            task.revision += 1
            if wakeup_on_complete is not None:
                task.wakeup_on_complete = wakeup_on_complete
                task.wakeup_reported = False
            if task.status not in {"queued", "running", "cancelling"}:
                task.status = "queued"
                task.completed_at = None
                task.error = ""
                task.result = ""
                should_restart = True
            elif task.status == "running":
                stream_to_close = task.active_stream
            self._update_locked(task, progress="steering update queued")
            self._append_event_locked(task, "steering update queued")
            self._persist_task_locked(task)
            snapshot = self._snapshot(task, include_result=False)
        if stream_to_close is not None:
            close = getattr(stream_to_close, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    LOGGER.exception("failed to close active background stream for steering")
        if should_restart:
            thread = threading.Thread(target=self._run_task, args=(task.id,), name=f"task-{task.id}", daemon=True)
            thread.start()
        LOGGER.info("steered background task %s with raw user transcript", task.id)
        return {"ok": True, "task": snapshot}

    def list(self, include_completed: bool = False) -> dict[str, Any]:
        with self._lock:
            tasks = list(self._tasks.values())
            if not include_completed:
                tasks = [task for task in tasks if task.status in {"queued", "running", "cancelling"}]
            tasks.sort(key=lambda task: task.created_at, reverse=True)
            return {
                "ok": True,
                "tasks": [self._snapshot(task, include_result=False) for task in tasks],
                "running_count": sum(
                    1 for task in self._tasks.values() if task.status in {"queued", "running", "cancelling"}
                ),
            }

    def get(self, task_id: str = "", include_result: bool = True) -> dict[str, Any]:
        with self._lock:
            task = self._get_locked(task_id)
            if task is None:
                return {"ok": False, "error": f"unknown task id: {task_id or '<latest>'}"}
            return {"ok": True, "task": self._snapshot(task, include_result=include_result)}

    def cancel(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return {"ok": False, "error": f"unknown task id: {task_id}"}
            if task.status not in {"queued", "running", "cancelling"}:
                return {"ok": True, "task": self._snapshot(task, include_result=False)}
            task.cancel_event.set()
            self._update_locked(task, status="cancelling", progress="cancellation requested")
            LOGGER.info("cancellation requested for background task %s", task_id)
            return {"ok": True, "task": self._snapshot(task, include_result=False)}

    def has_running(self) -> bool:
        with self._lock:
            return any(task.status in {"queued", "running", "cancelling"} for task in self._tasks.values())

    def pending_wakeups(self, limit: int = 1) -> list[dict[str, Any]]:
        with self._lock:
            wakeups: list[dict[str, Any]] = []
            for task in self._tasks.values():
                for index, update in enumerate(task.status_updates):
                    if index < task.progress_wakeup_index:
                        continue
                    if not update.get("wakeup") or update.get("reported"):
                        continue
                    snapshot = self._snapshot(task, include_result=False)
                    snapshot["wakeup"] = {
                        "type": "status_update",
                        "message_id": update.get("id", ""),
                        "text": update.get("text", ""),
                    }
                    wakeups.append(snapshot)
                    break
                if (
                    task.wakeup_on_complete
                    and not task.wakeup_reported
                    and task.status in {"completed", "failed"}
                ):
                    snapshot = self._snapshot(task, include_result=True)
                    snapshot["wakeup"] = {"type": "completed", "message_id": "", "text": task.result or task.error}
                    wakeups.append(snapshot)
            wakeups.sort(key=lambda item: item.get("updated_at") or "")
            return wakeups[:limit]

    def mark_wakeup_reported(self, task_id: str, message_id: str = "") -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is not None:
                if message_id:
                    for index, update in enumerate(task.status_updates):
                        if update.get("id") == message_id:
                            update["reported"] = True
                            task.progress_wakeup_index = max(task.progress_wakeup_index, index + 1)
                            break
                else:
                    task.wakeup_reported = True
                task.updated_at = time.time()
                self._persist_task_locked(task)

    def _run_task(self, task_id: str) -> None:
        while True:
            restart = self._run_task_revision(task_id)
            if not restart:
                return

    def _run_task_revision(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            active_revision = task.revision
            if active_revision > 0:
                task.result = ""
                task.error = ""
            self._update_locked(task, status="running", progress="starting GPT-5.5 task")
            self._persist_task_locked(task)

        stream: Any = None
        previous_response_id = ""
        next_input: Any = None
        try:
            while True:
                stream_state = {"active_phase": "final_answer", "commentary_parts": []}
                with self._lock:
                    task = self._tasks.get(task_id)
                    if task is None:
                        return False
                    if task.cancel_event.is_set():
                        self._finish_locked(task, "cancelled", "cancelled")
                        self._persist_task_locked(task)
                        return False
                    if task.revision != active_revision:
                        self._update_locked(task, progress="restarting with steering update")
                        self._persist_task_locked(task)
                        return True
                    request_input = next_input or background_task_prompt(self.config, task)

                kwargs = {
                    "model": self.config.task_model,
                    "instructions": background_task_instructions(self.config),
                    "input": request_input,
                    "reasoning": {
                        "effort": self.config.task_reasoning_effort,
                        "summary": self.config.task_reasoning_summary,
                    },
                    "tools": self._task_tools(),
                    "include": self._task_include(),
                    "stream": True,
                    "store": True,
                    "truncation": "auto",
                    "parallel_tool_calls": True,
                }
                if previous_response_id:
                    kwargs["previous_response_id"] = previous_response_id
                stream = self.client.responses.create(**kwargs)
                with self._lock:
                    task = self._tasks.get(task_id)
                    if task is not None:
                        task.active_stream = stream
                completed_response = None
                for event in stream:
                    with self._lock:
                        task = self._tasks.get(task_id)
                        if task is None:
                            return False
                        if task.cancel_event.is_set():
                            close = getattr(stream, "close", None)
                            if close is not None:
                                close()
                            if task.active_stream is stream:
                                task.active_stream = None
                            self._finish_locked(task, "cancelled", "cancelled")
                            self._persist_task_locked(task)
                            LOGGER.info("background task %s cancelled", task_id)
                            return False
                        if task.revision != active_revision:
                            close = getattr(stream, "close", None)
                            if close is not None:
                                close()
                            if task.active_stream is stream:
                                task.active_stream = None
                            self._update_locked(task, progress="restarting with steering update")
                            self._persist_task_locked(task)
                            return True
                    response = self._handle_stream_event(task_id, event, stream_state)
                    if response is not None:
                        completed_response = response

                with self._lock:
                    task = self._tasks.get(task_id)
                    if task is not None and task.active_stream is stream:
                        task.active_stream = None
                    if task is None:
                        return False
                    if task.revision != active_revision:
                        self._update_locked(task, progress="restarting with steering update")
                        self._persist_task_locked(task)
                        return True
                    self._flush_commentary_locked(task_id, stream_state, wakeup=False)
                function_outputs = self._function_call_outputs(task_id, completed_response)
                if function_outputs:
                    previous_response_id = object_get(completed_response, "id")
                    next_input = function_outputs
                    continue
                with self._lock:
                    task = self._tasks.get(task_id)
                    if task is not None and task.status in {"queued", "running", "cancelling"}:
                        self._finish_locked(task, "completed", "completed")
                        self._persist_task_locked(task)
                        LOGGER.info("background task %s completed", task_id)
                return False
        except BaseException as exc:
            with self._lock:
                task = self._tasks.get(task_id)
                if task is not None:
                    if task.active_stream is stream:
                        task.active_stream = None
                    if task.revision != active_revision:
                        self._update_locked(task, progress="restarting with steering update")
                        self._persist_task_locked(task)
                        return True
                    task.error = str(exc)
                    self._finish_locked(task, "failed", "failed")
                    self._persist_task_locked(task)
            LOGGER.exception("background task %s failed", task_id)
            return False

    def _task_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = [
            {
                "type": "function",
                "name": TASK_STATUS_TOOL_NAME,
                "description": (
                    "Queue a concise user-facing progress update for the realtime voice layer. "
                    "Use this for meaningful progress or a short preamble, not for private reasoning. "
                    "Do not use it for the final answer."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "One concise sentence the voice assistant can say to the user.",
                        },
                        "speak_now": {
                            "type": "boolean",
                            "description": (
                                "Set true only for important progress that should wake the voice assistant; "
                                "otherwise it is stored for the user to ask about."
                            ),
                        },
                    },
                    "required": ["text", "speak_now"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": GET_WEATHER_TOOL_NAME,
                "description": (
                    "Get current weather or a forecast from OpenWeather. Use this instead of web search for weather. "
                    "It defaults to Cambridge, GB when location is omitted, supports any named location, and caches "
                    "results for 10 minutes unless no_cache is true."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Location name such as Cambridge, GB or New York, US.",
                        },
                        "units": {
                            "type": "string",
                            "enum": ["metric", "imperial", "standard"],
                            "description": "Unit system. Use metric by default for Cambridge, UK.",
                        },
                        "forecast_days": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 5,
                            "description": "How many forecast days to return. Omit or use 0 for current-only.",
                        },
                        "no_cache": {
                            "type": "boolean",
                            "description": "Set true only when the user explicitly asks to refresh or recheck.",
                        },
                    },
                    "required": ["location", "units", "forecast_days", "no_cache"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "web_search",
                "search_context_size": self.config.web_search_context_size,
                "user_location": {
                    "type": "approximate",
                    "country": self.config.user_country,
                    "city": self.config.user_city,
                    "region": self.config.user_region,
                    "timezone": self.config.user_timezone,
                },
            }
        ]
        if self.config.task_code_execution:
            tools.append(
                {
                    "type": "code_interpreter",
                    "container": {"type": "auto"},
                }
            )
        return tools

    def _task_include(self) -> list[str]:
        include = ["web_search_call.action.sources"]
        if self.config.task_code_execution:
            include.append("code_interpreter_call.outputs")
        return include

    def _handle_stream_event(self, task_id: str, event: Any, stream_state: dict[str, Any]) -> Any | None:
        event_type = getattr(event, "type", "")
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status not in {"queued", "running", "cancelling"}:
                return None

            if event_type == "response.created":
                response = getattr(event, "response", None)
                task.response_id = getattr(response, "id", "") or task.response_id
                self._update_locked(task, progress="response created")
            elif event_type == "response.in_progress":
                self._update_locked(task, progress="model working")
            elif event_type == "response.web_search_call.in_progress":
                self._update_locked(task, progress="web search starting")
                self._append_event_locked(task, "web search starting")
            elif event_type == "response.web_search_call.searching":
                self._update_locked(task, progress="searching the web")
                self._append_event_locked(task, "searching the web")
            elif event_type == "response.web_search_call.completed":
                self._update_locked(task, progress="web search completed")
                self._append_event_locked(task, "web search completed")
            elif event_type == "response.code_interpreter_call.in_progress":
                self._update_locked(task, progress="code interpreter starting")
                self._append_event_locked(task, "code interpreter starting")
            elif event_type == "response.code_interpreter_call.interpreting":
                self._update_locked(task, progress="running code")
                self._append_event_locked(task, "running code")
            elif event_type == "response.code_interpreter_call.completed":
                self._update_locked(task, progress="code execution completed")
                self._append_event_locked(task, "code execution completed")
            elif event_type == "response.code_interpreter_call_code.delta":
                code_delta = getattr(event, "delta", "")
                if code_delta:
                    self._update_locked(task, progress=f"writing code: {_truncate(code_delta.strip(), 160)}")
            elif event_type in {"response.output_item.added", "response.output_item.created"}:
                item = getattr(event, "item", None)
                if object_get(item, "type") == "message":
                    stream_state["active_phase"] = object_get(item, "phase") or "final_answer"
                    stream_state["commentary_parts"] = []
                else:
                    stream_state["active_phase"] = ""
            elif event_type == "response.reasoning_summary_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    task.reasoning_summary = _truncate(
                        task.reasoning_summary + delta,
                        self.config.task_summary_chars,
                        keep_tail=True,
                    )
                    self._update_locked(task, progress="reasoning")
            elif event_type == "response.reasoning_summary_text.done":
                text = getattr(event, "text", "")
                if text:
                    task.reasoning_summary = _truncate(text, self.config.task_summary_chars, keep_tail=True)
                    self._update_locked(task, progress="reasoning summary updated")
            elif event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    if stream_state.get("active_phase") == "commentary":
                        stream_state.setdefault("commentary_parts", []).append(delta)
                        self._update_locked(task, progress="writing progress update")
                    else:
                        task.result = _truncate(task.result + delta, self.config.task_result_chars, keep_tail=True)
                        self._update_locked(task, progress="writing answer")
            elif event_type == "response.output_text.done":
                if stream_state.get("active_phase") == "commentary":
                    text = getattr(event, "text", "") or "".join(stream_state.get("commentary_parts", []))
                    if text:
                        self._append_status_update_locked(task, text, wakeup=False)
                    stream_state["commentary_parts"] = []
            elif event_type == "response.completed":
                response = getattr(event, "response", None)
                final_text = getattr(response, "output_text", "") or extract_responses_output_text(response)
                if final_text and not task.result.strip():
                    task.result = _truncate(final_text, self.config.task_result_chars, keep_tail=True)
                task.response_id = getattr(response, "id", "") or task.response_id
                self._persist_task_locked(task)
                return response
            elif event_type in {"response.failed", "response.incomplete", "error"}:
                error = getattr(event, "error", None)
                task.error = str(error or event_type)
                self._finish_locked(task, "failed", "failed")
                self._persist_task_locked(task)
        return None

    def _function_call_outputs(self, task_id: str, response: Any) -> list[dict[str, Any]]:
        if response is None:
            return []
        outputs: list[dict[str, Any]] = []
        for call in responses_function_calls(response):
            name = object_get(call, "name")
            call_id = object_get(call, "call_id")
            if not call_id:
                continue
            if name == TASK_STATUS_TOOL_NAME:
                arguments = parse_json_object(object_get(call, "arguments") or "{}")
                text = string_argument(arguments, "text")
                speak_now = parse_bool_argument(arguments.get("speak_now"), default=False)
                with self._lock:
                    task = self._tasks.get(task_id)
                    if task is not None:
                        self._append_status_update_locked(task, text, wakeup=speak_now or task.wakeup_on_progress)
                        self._persist_task_locked(task)
                outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps({"ok": True, "queued": bool(text)}, ensure_ascii=False),
                    }
                )
            elif name == GET_WEATHER_TOOL_NAME:
                arguments = parse_json_object(object_get(call, "arguments") or "{}")
                output = self.weather.get_weather(
                    location=string_argument(arguments, "location"),
                    units=string_argument(arguments, "units", "metric"),
                    no_cache=parse_bool_argument(arguments.get("no_cache"), default=False),
                    forecast_days=max(0, min(5, int_argument(arguments, "forecast_days", 0))),
                )
                outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(output, ensure_ascii=False),
                    }
                )
            else:
                outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps({"ok": False, "error": f"unknown background tool: {name}"}),
                    }
                )
        return outputs

    def _flush_commentary_locked(self, task_id: str, stream_state: dict[str, Any], wakeup: bool) -> None:
        if stream_state.get("active_phase") != "commentary":
            return
        text = "".join(stream_state.get("commentary_parts", [])).strip()
        if not text:
            return
        task = self._tasks.get(task_id)
        if task is None:
            return
        self._append_status_update_locked(task, text, wakeup=wakeup or task.wakeup_on_progress)
        stream_state["commentary_parts"] = []

    def _append_status_update_locked(self, task: BackgroundTask, text: str, wakeup: bool) -> None:
        clean = _truncate(text.strip(), 500)
        if not clean:
            return
        task.status_updates.append(
            {
                "id": f"status_{uuid.uuid4().hex[:10]}",
                "text": clean,
                "created_at": _iso_time(time.time()),
                "wakeup": bool(wakeup),
                "reported": False,
            }
        )
        if len(task.status_updates) > 50:
            del task.status_updates[:-50]
        self._append_event_locked(task, f"status update: {clean}")
        self._update_locked(task, progress=clean)

    def _get_locked(self, task_id: str, prefer_active: bool = False) -> BackgroundTask | None:
        if task_id:
            return self._tasks.get(task_id)
        if not self._tasks:
            return None
        if prefer_active:
            active = [
                task for task in self._tasks.values() if task.status in {"queued", "running", "cancelling"}
            ]
            if active:
                return max(active, key=lambda task: task.updated_at)
        return max(self._tasks.values(), key=lambda task: task.created_at)

    def _prune_locked(self) -> None:
        if len(self._tasks) < self.config.max_background_tasks:
            return
        done = [
            task
            for task in self._tasks.values()
            if task.status not in {"queued", "running", "cancelling"}
        ]
        done.sort(key=lambda task: task.updated_at)
        while done and len(self._tasks) >= self.config.max_background_tasks:
            task = done.pop(0)
            self._tasks.pop(task.id, None)
            self._delete_task_file(task.id)

    def _snapshot(self, task: BackgroundTask, include_result: bool) -> dict[str, Any]:
        return {
            "id": task.id,
            "title": task.title,
            "request_text": task.request_text,
            "source": task.source,
            "wakeup_on_complete": task.wakeup_on_complete,
            "wakeup_on_progress": task.wakeup_on_progress,
            "wakeup_reported": task.wakeup_reported,
            "status": task.status,
            "progress": task.progress,
            "created_at": _iso_time(task.created_at),
            "updated_at": _iso_time(task.updated_at),
            "completed_at": _iso_time(task.completed_at) if task.completed_at else None,
            "response_id": task.response_id,
            "status_updates": task.status_updates[-12:],
            "steering_messages": task.steering_messages[-12:],
            "result": task.result if include_result else _truncate(task.result, 1200),
            "error": task.error,
            "events": task.events[-12:],
        }

    def _update_locked(self, task: BackgroundTask, status: str | None = None, progress: str | None = None) -> None:
        if status is not None:
            task.status = status
        if progress is not None:
            task.progress = progress
        task.updated_at = time.time()
        self._persist_task_locked(task)

    def _finish_locked(self, task: BackgroundTask, status: str, progress: str) -> None:
        self._update_locked(task, status=status, progress=progress)
        task.completed_at = time.time()

    def _append_event_locked(self, task: BackgroundTask, text: str) -> None:
        task.events.append(f"{_iso_time(time.time())}: {text}")
        if len(task.events) > 50:
            del task.events[:-50]

    def _task_file(self, task_id: str) -> Path:
        return self.tasks_dir / f"{task_id}.json"

    def _persist_task_locked(self, task: BackgroundTask) -> None:
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "task": self._task_payload(task),
        }
        path = self._task_file(task.id)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)

    def _task_payload(self, task: BackgroundTask) -> dict[str, Any]:
        return {
            "id": task.id,
            "request_text": task.request_text,
            "history": [message.__dict__ for message in task.history],
            "memory_context": task.memory_context,
            "title": task.title,
            "source": task.source,
            "wakeup_on_complete": task.wakeup_on_complete,
            "wakeup_on_progress": task.wakeup_on_progress,
            "wakeup_reported": task.wakeup_reported,
            "progress_wakeup_index": task.progress_wakeup_index,
            "status": task.status,
            "progress": task.progress,
            "response_id": task.response_id,
            "result": task.result,
            "reasoning_summary": task.reasoning_summary,
            "error": task.error,
            "events": task.events,
            "status_updates": task.status_updates,
            "steering_messages": task.steering_messages,
            "revision": task.revision,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
            "completed_at": task.completed_at,
        }

    def _load_persisted_tasks(self) -> None:
        if not self.tasks_dir.exists():
            return
        for path in sorted(self.tasks_dir.glob("task_*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                data = payload.get("task", payload)
                task = self._task_from_payload(data)
            except Exception:
                LOGGER.exception("failed to load background task state from %s", path)
                continue
            self._tasks[task.id] = task

    def _task_from_payload(self, data: dict[str, Any]) -> BackgroundTask:
        history = []
        for item in data.get("history", []):
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                history.append(Message(role=role, content=content))
        task = BackgroundTask(
            id=str(data.get("id") or f"task_{uuid.uuid4().hex[:10]}"),
            request_text=str(data.get("request_text") or data.get("query") or ""),
            history=history,
            memory_context=str(data.get("memory_context") or ""),
            title=str(data.get("title") or ""),
            source=str(data.get("source") or "realtime"),
            wakeup_on_complete=bool(data.get("wakeup_on_complete")),
            wakeup_on_progress=bool(data.get("wakeup_on_progress")),
            wakeup_reported=bool(data.get("wakeup_reported")),
            progress_wakeup_index=int(data.get("progress_wakeup_index") or 0),
            status=str(data.get("status") or "completed"),
            progress=str(data.get("progress") or ""),
            response_id=str(data.get("response_id") or ""),
            result=str(data.get("result") or ""),
            reasoning_summary=str(data.get("reasoning_summary") or ""),
            error=str(data.get("error") or ""),
            events=[str(item) for item in data.get("events", []) if isinstance(item, str)],
            status_updates=[
                item for item in data.get("status_updates", []) if isinstance(item, dict)
            ],
            steering_messages=[
                item for item in data.get("steering_messages", []) if isinstance(item, dict)
            ],
            revision=int(data.get("revision") or 0),
            created_at=float(data.get("created_at") or time.time()),
            updated_at=float(data.get("updated_at") or time.time()),
            completed_at=data.get("completed_at"),
        )
        if task.status in {"queued", "running", "cancelling"}:
            task.status = "failed"
            task.progress = "interrupted by daemon restart"
            task.error = task.error or "interrupted by daemon restart"
            task.completed_at = time.time()
        return task

    def _delete_task_file(self, task_id: str) -> None:
        try:
            self._task_file(task_id).unlink(missing_ok=True)
        except Exception:
            LOGGER.exception("failed to delete task file for %s", task_id)


def background_task_instructions(config: Config) -> str:
    return (
        "You are a GPT-5.5 background research and code-execution worker for Max Code, a realtime general-purpose "
        "voice assistant. "
        "The realtime voice model may decide whether to start or steer a task, but it does not interpret or rewrite "
        "the user's request for you. Interpret the raw user transcript yourself using the provided conversation "
        "and steering messages. "
        "Use get_weather for weather questions, including current conditions and forecasts up to 5 days. It defaults "
        "to Cambridge, GB when location is omitted, but it can also look up any location the user names. "
        "Use web_search for other current, local, or source-dependent facts. Use code_interpreter for calculations, "
        "small programs, data processing, writing code, or checking code. Interpret relative dates like today, "
        "tomorrow, this morning, and tonight using the user's local timezone from the context. For web results, "
        "prefer sources relevant to the user's location and current local date when the query is local or time-sensitive. "
        "Use emit_status_update for concise user-facing progress updates or preambles. Do not put private reasoning "
        "or chain-of-thought in status updates. Produce a concise final answer suitable for the voice assistant to "
        "summarize aloud, with source names or URLs when web facts matter. "
        f"{local_context(config)}"
    )


def background_task_prompt(config: Config, task: BackgroundTask) -> str:
    history = task.history[-12:]
    history_lines = []
    for message in history:
        history_lines.append(f"{message.role}: {_truncate(message.content, 1000)}")
    steering_lines = []
    for item in task.steering_messages:
        text = _truncate(str(item.get("text") or ""), 1000)
        created_at = str(item.get("created_at") or "")
        steering_lines.append(f"- {created_at}: {text}")
    status_lines = []
    for item in task.status_updates[-8:]:
        status_lines.append(f"- {item.get('created_at')}: {item.get('text')}")
    return (
        "This task was started from a live push-to-talk Realtime conversation. "
        "Work independently. The text below is the raw user transcript captured by the application, not a rewritten "
        "query from the realtime model. Correctly infer what the user wants from the raw transcript and context. "
        "If steering messages are present, treat each steering message as a later raw user utterance about this same "
        "task. Look up and apply the newest steering message from the user before continuing. "
        "Use emit_status_update for useful progress messages that Max Code can say to the user. "
        "Use hosted web search for current facts and code_interpreter for calculations, code generation, code execution, "
        "or verification. Treat the following local context as authoritative for 'today', 'now', local weather, events, "
        f"and schedules. {local_context(config)} "
        f"\n\nOriginal raw user transcript:\n{task.request_text}"
        f"\n\nRecent conversation before this turn:\n{chr(10).join(history_lines) if history_lines else '[none]'}"
        f"\n\nRelevant durable memory:\n{task.memory_context.strip() if task.memory_context.strip() else '[none]'}"
        f"\n\nSteering messages from later user turns:\n{chr(10).join(steering_lines) if steering_lines else '[none]'}"
        f"\n\nPreviously queued user-facing status updates:\n{chr(10).join(status_lines) if status_lines else '[none]'}"
    )


def _iso_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _truncate(text: str, max_chars: int, keep_tail: bool = False) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n[truncated]\n"
    if keep_tail:
        return marker + text[-max(0, max_chars - len(marker)) :]
    return text[: max(0, max_chars - len(marker))] + marker


class RealtimeVoiceClient:
    def __init__(self, config: Config, settings: RuntimeSettings | None = None):
        self.config = config
        self.settings = settings or RuntimeSettings(config.settings_file, config.voice_volume, config.music_volume)
        self.settings.ensure()
        self.player = PcmPlayer(config.audio_playback_device, volume_level=self.settings.voice_volume())

    def wait_for_connectivity(self) -> None:
        host = self.config.openai_connectivity_host
        deadline = time.monotonic() + self.config.openai_connectivity_wait_seconds
        last_error: OSError | None = None
        while True:
            try:
                with socket.create_connection((host, 443), timeout=5):
                    return
            except OSError as exc:
                last_error = exc
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(f"network is not ready for {host}:443") from last_error
                LOGGER.warning("waiting for network/DNS before OpenAI realtime call: %s", exc)
                time.sleep(min(5.0, remaining))

    def respond_to_audio(self, history: Iterable[Message], audio_path: Path) -> RealtimeTurnResult:
        deadline = time.monotonic() + self.config.realtime_max_session_seconds
        ws = self._connect()
        try:
            self._send_session_update(ws)
            self._wait_for_session_updated(ws, deadline)
            self._seed_history(ws, history)
            self._send_audio_turn(ws, audio_path)
            return self._read_response(ws, deadline)
        finally:
            ws.close()

    def _connect(self) -> websocket.WebSocket:
        url = f"wss://api.openai.com/v1/realtime?model={self.config.realtime_model}"
        headers = [
            f"Authorization: Bearer {self.config.openai_api_key}",
            f"OpenAI-Safety-Identifier: {self.config.realtime_safety_identifier}",
        ]
        LOGGER.info("opening realtime websocket with %s", self.config.realtime_model)
        return websocket.create_connection(
            url,
            header=headers,
            sslopt={"ca_certs": certifi.where()},
            timeout=self.config.realtime_response_timeout_seconds,
        )

    def _send_session_update(self, ws: websocket.WebSocket, memory_context: str = "") -> None:
        session: dict[str, Any] = {
            "type": "realtime",
            "model": self.config.realtime_model,
            "output_modalities": ["audio"],
            "instructions": realtime_session_instructions(self.config, memory_context),
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": self.config.realtime_input_rate,
                    },
                    "turn_detection": realtime_turn_detection(self.config),
                    "transcription": {
                        "model": self.config.realtime_input_transcription_model,
                    },
                },
                "output": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": 24000,
                    },
                    "voice": self.config.realtime_voice,
                },
            },
            "reasoning": {
                "effort": self.config.realtime_reasoning_effort,
            },
            "tools": realtime_tools(),
            "tool_choice": "auto",
        }
        self._send(ws, {"type": "session.update", "session": session})

    def _wait_for_session_updated(self, ws: websocket.WebSocket, deadline: float) -> None:
        while True:
            event = self._recv(ws, deadline)
            event_type = event.get("type")
            if event_type == "session.updated":
                LOGGER.info("realtime session updated")
                return
            if event_type == "session.created":
                continue
            self._raise_for_error(event)

    def _seed_history(self, ws: websocket.WebSocket, history: Iterable[Message]) -> None:
        messages = list(history)[-self.config.realtime_history_messages :]
        if not messages:
            return
        LOGGER.info("seeding realtime session with %d saved messages", len(messages))
        for message in messages:
            self._send(ws, {"type": "conversation.item.create", "item": conversation_item_for_message(message)})

    def _send_audio_turn(self, ws: websocket.WebSocket, audio_path: Path) -> None:
        self._send(ws, {"type": "input_audio_buffer.clear"})
        total_bytes = 0
        for chunk in iter_wav_pcm16_chunks(audio_path, self.config.realtime_input_rate):
            total_bytes += len(chunk)
            self._send(
                ws,
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                },
            )
        if total_bytes == 0:
            raise RuntimeError(f"recording has no PCM audio: {audio_path}")
        LOGGER.info("sent %d bytes of PCM audio to realtime session", total_bytes)
        self._send(ws, {"type": "input_audio_buffer.commit"})
        self._send(
            ws,
            {
                "type": "response.create",
                "response": {
                    "output_modalities": ["audio"],
                    "instructions": realtime_turn_instructions(self.config),
                },
            },
        )

    def _read_response(self, ws: websocket.WebSocket, deadline: float) -> RealtimeTurnResult:
        input_transcript_parts: list[str] = []
        input_transcript_final = ""
        output_transcript_parts: list[str] = []
        output_transcript_final = ""
        response_done_text = ""
        requested_close = False

        with self.player.open_stream(rate=24000, channels=1) as audio:
            while True:
                event = self._recv(ws, deadline)
                event_type = event.get("type")
                self._raise_for_error(event)

                if event_type == "conversation.item.input_audio_transcription.delta":
                    input_transcript_parts.append(event.get("delta", ""))
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    input_transcript_final = event.get("transcript", "").strip()
                    if input_transcript_final:
                        LOGGER.info("realtime input transcript: %s", input_transcript_final)
                elif event_type == "response.output_audio.delta":
                    audio.write(base64.b64decode(event.get("delta", "")))
                elif event_type == "response.output_audio_transcript.delta":
                    output_transcript_parts.append(event.get("delta", ""))
                elif event_type == "response.output_audio_transcript.done":
                    output_transcript_final = event.get("transcript", "").strip()
                    if output_transcript_final:
                        LOGGER.info("realtime assistant transcript: %s", output_transcript_final)
                elif event_type == "response.done":
                    response = event.get("response", {})
                    requested_close = response_requested_close(response)
                    response_done_text = extract_response_text(response)
                    break

        user_text = input_transcript_final or "".join(input_transcript_parts).strip()
        assistant_text = output_transcript_final or "".join(output_transcript_parts).strip() or response_done_text
        if requested_close:
            LOGGER.info("model requested realtime session close")
        if not user_text:
            LOGGER.warning("realtime session completed without an input transcript")
        if not assistant_text and not requested_close:
            raise RuntimeError("realtime model returned no assistant transcript")
        return RealtimeTurnResult(user_text=user_text, assistant_text=assistant_text, requested_close=requested_close)

    def _send(self, ws: websocket.WebSocket, event: dict[str, Any]) -> None:
        ws.send(json.dumps(event))

    def _recv(self, ws: websocket.WebSocket, deadline: float) -> dict[str, Any]:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("realtime session exceeded local maximum duration")
            ws.settimeout(min(5.0, remaining))
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            event = json.loads(raw)
            if not isinstance(event, dict):
                raise RuntimeError(f"unexpected realtime event: {event!r}")
            return event

    def _raise_for_error(self, event: dict[str, Any]) -> None:
        if event.get("type") != "error":
            return
        error = event.get("error", event)
        message = error.get("message") if isinstance(error, dict) else str(error)
        raise RuntimeError(f"realtime API error: {message}")


class RealtimeConversationSession:
    def __init__(
        self,
        config: Config,
        scheduled_tasks: ScheduledTaskStore | None = None,
        memory: MemoryStore | None = None,
        music: MusicPlayer | None = None,
        settings: RuntimeSettings | None = None,
        recorder_factory: Callable[[], RawPcmRecorder] | None = None,
    ):
        self.config = config
        self.settings = settings or RuntimeSettings(config.settings_file, config.voice_volume, config.music_volume)
        self.settings.ensure()
        self.player = PcmPlayer(config.audio_playback_device, volume_level=self.settings.voice_volume())
        self._recorder_factory = recorder_factory or (lambda: RawPcmRecorder(self.config))
        self.scheduled_tasks = scheduled_tasks or ScheduledTaskStore(config)
        self.memory = memory or MemoryStore(config)
        self.memory.ensure_workspace()
        self.tasks = BackgroundTaskManager(config, self.memory)
        self.weather = OpenWeatherService(config)
        self.music = music or MusicPlayer(config, settings=self.settings)
        self.camera = CameraCapture(config)
        self._ws: websocket.WebSocket | None = None
        self._receiver_thread: threading.Thread | None = None
        self._send_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._errors: queue.Queue[BaseException] = queue.Queue()
        self._completed_turns: queue.Queue[RealtimeTurnResult] = queue.Queue()

        self._session_started_at = 0.0
        self._last_activity_at = 0.0
        self._closing = False
        self._seeded = False

        self._recorder: RawPcmRecorder | None = None
        self._capture_queue: queue.Queue[bytes | None] | None = None
        self._capture_thread: threading.Thread | None = None
        self._sender_thread: threading.Thread | None = None
        self._recording_started_at = 0.0
        self._capture_bytes = 0
        self._sent_audio_bytes = 0

        self._response_pending = False
        self._response_create_pending = False
        self._response_active = False
        self._response_wait_started_at = 0.0
        self._waiting_for_input_transcript = False
        self._tool_active = False
        self._tool_generation = 0
        self._turn_generation = 0
        self._pending_input_generation = 0
        self._active_response_id = ""
        self._playback: PcmOutputStream | None = None
        self._playback_closing = False
        self._output_item_id = ""
        self._output_audio_bytes = 0
        self._input_transcript_parts: list[str] = []
        self._input_transcript_final = ""
        self._output_transcript_parts: list[str] = []
        self._output_transcript_final = ""
        self._ignore_current_input = False
        self._ignore_next_input_transcript = False
        self._current_history: list[Message] = []
        self._continuous_camera: ContinuousCameraCapture | None = None
        self._continuous_camera_thread: threading.Thread | None = None
        self._continuous_camera_stop: threading.Event | None = None
        self._continuous_camera_interval = self._clamp_camera_interval(config.camera_continuous_interval_seconds)
        self._continuous_camera_reason = ""
        self._continuous_camera_generation = 0

    def start_live_session(self, history: Iterable[Message], capture_image: bool = False) -> dict[str, Any]:
        self._raise_background_error()
        history_list = list(history)
        started_capture = False
        with self._state_lock:
            self._current_history = history_list
            self._last_activity_at = time.monotonic()
            if self._recorder is None:
                self._interrupt_response_locked()
                self._reset_turn_text_locked()
                self._recorder = self._recorder_factory()
                self._capture_queue = queue.Queue()
                self._capture_bytes = 0
                self._sent_audio_bytes = 0
                self._recording_started_at = time.monotonic()
                recorder = self._recorder
                capture_queue = self._capture_queue
                started_capture = True
            else:
                recorder = None
                capture_queue = None

        if started_capture:
            assert recorder is not None
            assert capture_queue is not None
            LOGGER.info("starting realtime live audio capture")
            try:
                recorder.start()
            except BaseException:
                self._stop_capture_threads()
                raise
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                args=(recorder, capture_queue),
                name="realtime-capture",
                daemon=True,
            )
            self._capture_thread.start()

        try:
            self.ensure_open(history_list)
            if started_capture:
                self._send({"type": "input_audio_buffer.clear"})
                assert capture_queue is not None
                self._sender_thread = threading.Thread(
                    target=self._send_audio_loop,
                    args=(capture_queue,),
                    name="realtime-audio-sender",
                    daemon=True,
                )
                self._sender_thread.start()
                LOGGER.info("realtime live audio sender started")
            if capture_image:
                LOGGER.info("ignoring non-model camera snapshot request for live session start")
            return {"ok": True, "started": started_capture}
        except BaseException:
            if started_capture:
                self._stop_capture_threads()
            raise

    def capture_camera_snapshot(
        self,
        history: Iterable[Message] = (),
        reason: str = "",
        create_response: bool = False,
        asynchronous: bool = False,
    ) -> dict[str, Any]:
        if asynchronous:
            threading.Thread(
                target=self._capture_camera_snapshot_safe,
                args=(list(history), reason, create_response),
                name="camera-snapshot",
                daemon=True,
            ).start()
            return {"ok": True, "queued": True}
        return self._capture_camera_snapshot_safe(list(history), reason, create_response)

    def _capture_camera_snapshot_safe(
        self,
        history: list[Message],
        reason: str = "",
        create_response: bool = False,
    ) -> dict[str, Any]:
        try:
            return self._capture_camera_snapshot(history, reason, create_response)
        except CameraError as exc:
            LOGGER.warning("camera snapshot unavailable: %s", exc)
            return {"ok": False, "error": str(exc)}
        except BaseException as exc:
            LOGGER.exception("camera snapshot failed")
            return {"ok": False, "error": str(exc)}

    def _capture_camera_snapshot(
        self,
        history: list[Message],
        reason: str = "",
        create_response: bool = False,
    ) -> dict[str, Any]:
        self.ensure_open(history)
        snapshot = self.camera.capture(
            settle_seconds=self.config.camera_snapshot_settle_seconds,
            shutter_callback=self._play_camera_shutter if self.config.camera_shutter_sound_enabled else None,
        )
        note = reason.strip() or "camera snapshot"
        with self._state_lock:
            if self._recorder is not None:
                LOGGER.info("discarding camera snapshot captured during new recording")
                return {"ok": False, "cancelled": True, "error": "interrupted by new recording"}
        self._attach_camera_snapshot(snapshot, note)
        if create_response:
            with self._state_lock:
                self._response_pending = True
                self._response_create_pending = True
                self._response_wait_started_at = time.monotonic()
            self._send(
                {
                    "type": "response.create",
                    "response": {
                        "output_modalities": ["audio"],
                        "instructions": realtime_turn_instructions(self.config),
                    },
                }
            )
        return {
            "ok": True,
            "path": str(snapshot.path),
            "bytes": snapshot.size_bytes,
            "detail": self.config.camera_image_detail,
        }

    def _attach_camera_snapshot(self, snapshot: CameraSnapshot, note: str) -> None:
        self._send(
            {
                "type": "conversation.item.create",
                "item": conversation_item_for_camera_snapshot(
                    snapshot,
                    detail=self.config.camera_image_detail,
                    note=note,
                ),
            }
        )
        with self._state_lock:
            self._last_activity_at = time.monotonic()
        LOGGER.info("attached camera snapshot %s (%d bytes)", snapshot.path, snapshot.size_bytes)

    def start_continuous_camera(
        self,
        history: Iterable[Message] = (),
        reason: str = "",
        interval_seconds: float | None = None,
    ) -> dict[str, Any]:
        self.ensure_open(history)
        interval = self._clamp_camera_interval(
            self.config.camera_continuous_interval_seconds if interval_seconds is None else interval_seconds
        )
        clean_reason = reason.strip() or "continuous camera"
        with self._state_lock:
            if self._continuous_camera is not None:
                self._continuous_camera_interval = interval
                self._continuous_camera_reason = clean_reason
                self._last_activity_at = time.monotonic()
                return {
                    "ok": True,
                    "active": True,
                    "updated": True,
                    "interval_seconds": interval,
                    "min_interval_seconds": self.config.camera_continuous_min_interval_seconds,
                    "max_interval_seconds": self.config.camera_continuous_max_interval_seconds,
                }

        camera = ContinuousCameraCapture(self.config)
        camera.start()
        stop_event = threading.Event()
        with self._state_lock:
            self._continuous_camera_generation += 1
            generation = self._continuous_camera_generation
            self._continuous_camera = camera
            self._continuous_camera_stop = stop_event
            self._continuous_camera_interval = interval
            self._continuous_camera_reason = clean_reason
            self._last_activity_at = time.monotonic()
            thread = threading.Thread(
                target=self._continuous_camera_loop,
                args=(generation, camera, stop_event),
                name="continuous-camera-upload",
                daemon=True,
            )
            self._continuous_camera_thread = thread
            thread.start()
        return {
            "ok": True,
            "active": True,
            "updated": False,
            "interval_seconds": interval,
            "min_interval_seconds": self.config.camera_continuous_min_interval_seconds,
            "max_interval_seconds": self.config.camera_continuous_max_interval_seconds,
        }

    def stop_continuous_camera(self, reason: str = "") -> dict[str, Any]:
        camera, thread = self._stop_continuous_camera_locked_parts()
        if camera is None:
            return {"ok": True, "active": False, "stopped": False}
        camera.close()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)
        LOGGER.info("continuous camera stopped: %s", reason.strip() or "requested")
        return {"ok": True, "active": False, "stopped": True}

    def _continuous_camera_loop(
        self,
        generation: int,
        camera: ContinuousCameraCapture,
        stop_event: threading.Event,
    ) -> None:
        try:
            settle = self.config.camera_snapshot_settle_seconds
            if settle > 0 and stop_event.wait(settle):
                return
            while not stop_event.is_set():
                with self._state_lock:
                    if generation != self._continuous_camera_generation or self._ws is None or self._closing:
                        return
                    capture_interval = self._continuous_camera_interval
                    if self._response_create_pending or self._response_active or self._tool_active:
                        wait_interval = min(0.25, max(0.05, capture_interval))
                    else:
                        wait_interval = 0.0
                    note_reason = self._continuous_camera_reason or "continuous camera"
                if wait_interval:
                    if stop_event.wait(wait_interval):
                        return
                    continue
                snapshot = camera.snapshot(note=note_reason)
                with self._state_lock:
                    if generation != self._continuous_camera_generation or self._ws is None or self._closing:
                        return
                self._attach_camera_snapshot(snapshot, f"continuous camera: {note_reason}")
                if stop_event.wait(capture_interval):
                    return
        except BaseException as exc:
            self._errors.put(exc)
        finally:
            with self._state_lock:
                if generation == self._continuous_camera_generation and self._continuous_camera is camera:
                    self._continuous_camera = None
                    self._continuous_camera_thread = None
                    self._continuous_camera_stop = None
            camera.close()

    def _stop_continuous_camera_locked_parts(
        self,
    ) -> tuple[ContinuousCameraCapture | None, threading.Thread | None]:
        with self._state_lock:
            camera = self._continuous_camera
            thread = self._continuous_camera_thread
            stop_event = self._continuous_camera_stop
            if camera is None:
                return None, None
            self._continuous_camera_generation += 1
            self._continuous_camera = None
            self._continuous_camera_thread = None
            self._continuous_camera_stop = None
            if stop_event is not None:
                stop_event.set()
        return camera, thread

    def _clamp_camera_interval(self, interval_seconds: float) -> float:
        try:
            interval = float(interval_seconds)
        except (TypeError, ValueError):
            interval = self.config.camera_continuous_interval_seconds
        min_interval = self.config.camera_continuous_min_interval_seconds
        max_interval = max(min_interval, self.config.camera_continuous_max_interval_seconds)
        return max(min_interval, min(max_interval, interval))

    def _play_camera_shutter(self) -> None:
        try:
            with self.player.open_stream(rate=24000, channels=1) as stream:
                stream.write(camera_shutter_pcm())
        except BaseException:
            LOGGER.exception("camera shutter cue failed")

    def begin_turn(self, history: Iterable[Message]) -> None:
        self._raise_background_error()
        with self._state_lock:
            if self._recorder is not None:
                raise RuntimeError("realtime recording is already active")
            self._last_activity_at = time.monotonic()
            self._interrupt_response_locked()
            self._reset_turn_text_locked()
            self._current_history = list(history)
            self._recorder = self._recorder_factory()
            self._capture_queue = queue.Queue()
            self._capture_bytes = 0
            self._sent_audio_bytes = 0
            self._recording_started_at = time.monotonic()
            recorder = self._recorder
            capture_queue = self._capture_queue

        LOGGER.info("starting realtime turn capture")
        try:
            recorder.start()
        except BaseException:
            self._stop_capture_threads()
            raise
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            args=(recorder, capture_queue),
            name="realtime-capture",
            daemon=True,
        )
        self._capture_thread.start()

        try:
            self.ensure_open(history)
            self._send({"type": "input_audio_buffer.clear"})
            self._sender_thread = threading.Thread(
                target=self._send_audio_loop,
                args=(capture_queue,),
                name="realtime-audio-sender",
                daemon=True,
            )
            self._sender_thread.start()
            LOGGER.info("realtime audio sender started")
        except BaseException:
            self._stop_capture_threads()
            raise

    def stop_recording(self) -> float:
        return self._stop_capture_threads()

    def commit_recording(self) -> None:
        self._raise_background_error()
        with self._state_lock:
            sent_audio_bytes = self._sent_audio_bytes
        if sent_audio_bytes == 0:
            self.clear_pending_input()
            raise RuntimeError("no audio was captured for realtime turn")
        self._send({"type": "input_audio_buffer.commit"})
        with self._state_lock:
            self._turn_generation += 1
            self._pending_input_generation = self._turn_generation
            self._response_pending = True
            self._response_create_pending = False
            self._waiting_for_input_transcript = True
            self._response_wait_started_at = time.monotonic()
            self._last_activity_at = time.monotonic()
        LOGGER.info("committed realtime turn: %d PCM bytes; waiting for transcript", sent_audio_bytes)

    def end_turn(self, commit: bool) -> float:
        duration = self._stop_capture_threads()
        if not commit:
            self.clear_pending_input()
            return duration
        self.commit_recording()
        LOGGER.info("realtime recording duration: %.3fs", duration)
        return duration

    def clear_pending_input(self) -> None:
        if self.is_open:
            self._send({"type": "input_audio_buffer.clear"})

    def check_health(self) -> None:
        self._raise_background_error()

    def ensure_open(self, history: Iterable[Message]) -> None:
        self._raise_background_error()
        with self._state_lock:
            if self._ws is not None and self._receiver_thread is not None and self._receiver_thread.is_alive():
                return
        ws = RealtimeVoiceClient(self.config)._connect()
        deadline = time.monotonic() + self.config.realtime_response_timeout_seconds
        try:
            RealtimeVoiceClient(self.config)._send_session_update(ws, self.memory.bootstrap_context())
            RealtimeVoiceClient(self.config)._wait_for_session_updated(ws, deadline)
            RealtimeVoiceClient(self.config)._seed_history(ws, history)
        except BaseException:
            ws.close()
            raise

        with self._state_lock:
            self._ws = ws
            self._closing = False
            self._seeded = True
            self._session_started_at = time.monotonic()
            self._last_activity_at = self._session_started_at
        self._receiver_thread = threading.Thread(target=self._receive_loop, name="realtime-receiver", daemon=True)
        self._receiver_thread.start()
        LOGGER.info("persistent realtime session open")

    def close_if_idle(self) -> None:
        with self._state_lock:
            if self._ws is None or self._is_busy_locked() or self._continuous_camera is not None:
                return
            idle_for = time.monotonic() - self._last_activity_at
            if idle_for < self.config.realtime_idle_timeout_seconds:
                return
        LOGGER.info("closing realtime session after %.1fs idle", idle_for)
        self.close()

    def close_if_too_old(self) -> None:
        with self._state_lock:
            if self._ws is None:
                return
            age = time.monotonic() - self._session_started_at
            if age < self.config.realtime_max_session_seconds:
                return
        LOGGER.info("closing realtime session after %.1fs hard max age", age)
        self.close()

    def cool_down_if_silent(self) -> None:
        cooldown = self.config.realtime_silent_cooldown_seconds
        if cooldown <= 0:
            return
        should_start_response = False
        fallback_query = ""
        generation = 0
        with self._state_lock:
            if self._ws is None or self._recorder is not None:
                return
            if not (self._response_pending or self._response_active or self._tool_active):
                return
            if self._playback is not None or self._output_transcript_parts or self._output_transcript_final:
                return
            wait_started_at = self._response_wait_started_at or self._last_activity_at
            silent_for = time.monotonic() - wait_started_at
            if self._waiting_for_input_transcript and silent_for >= min(2.0, cooldown):
                LOGGER.info("starting realtime response before final transcript after %.1fs", silent_for)
                self._waiting_for_input_transcript = False
                generation = self._pending_input_generation
                fallback_query = self._input_transcript_final or "".join(self._input_transcript_parts).strip()
                should_start_response = True
            elif self._waiting_for_input_transcript:
                return
            if not should_start_response and silent_for < cooldown:
                return
            if not should_start_response:
                LOGGER.info("cooling down silent realtime response after %.1fs", silent_for)
                self._interrupt_response_locked()
                self._reset_turn_text_locked()
        if should_start_response:
            self._start_response_for_current_turn(fallback_query, generation)

    def close(self) -> None:
        receiver_thread: threading.Thread | None
        camera, camera_thread = self._stop_continuous_camera_locked_parts()
        with self._state_lock:
            self._closing = True
            ws = self._ws
            self._ws = None
            receiver_thread = self._receiver_thread
            self._receiver_thread = None
            self._response_pending = False
            self._response_create_pending = False
            self._response_active = False
            self._response_wait_started_at = 0.0
            self._waiting_for_input_transcript = False
            self._tool_active = False
            self._tool_generation += 1
            self._abort_playback_locked()
            self._reset_turn_text_locked()
        self._stop_capture_threads()
        if camera is not None:
            camera.close()
        if camera_thread is not None and camera_thread is not threading.current_thread():
            camera_thread.join(timeout=2)
        if ws is not None:
            ws.close()
        if receiver_thread is not None and receiver_thread is not threading.current_thread():
            receiver_thread.join(timeout=2)
        with self._state_lock:
            self._closing = False
            self._seeded = False
        LOGGER.info("persistent realtime session closed")

    @property
    def is_open(self) -> bool:
        with self._state_lock:
            return self._ws is not None

    @property
    def is_recording(self) -> bool:
        with self._state_lock:
            return self._recorder is not None

    @property
    def is_responding(self) -> bool:
        with self._state_lock:
            return self._response_pending or self._response_active or self._tool_active or self._playback is not None

    @property
    def is_voice_busy(self) -> bool:
        with self._state_lock:
            return self._is_busy_locked()

    def pop_completed_turns(self) -> list[RealtimeTurnResult]:
        turns: list[RealtimeTurnResult] = []
        while True:
            try:
                turns.append(self._completed_turns.get_nowait())
            except queue.Empty:
                return turns

    def _capture_loop(self, recorder: RawPcmRecorder, capture_queue: queue.Queue[bytes | None]) -> None:
        try:
            while True:
                chunk = recorder.read(4096)
                if not chunk:
                    return
                with self._state_lock:
                    self._capture_bytes += len(chunk)
                capture_queue.put(chunk)
        except BaseException as exc:
            self._errors.put(exc)
        finally:
            capture_queue.put(None)

    def _send_audio_loop(self, capture_queue: queue.Queue[bytes | None]) -> None:
        try:
            while True:
                chunk = capture_queue.get()
                if chunk is None:
                    return
                self._send(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                    }
                )
                with self._state_lock:
                    self._sent_audio_bytes += len(chunk)
        except BaseException as exc:
            self._errors.put(exc)

    def _stop_capture_threads(self) -> float:
        with self._state_lock:
            recorder = self._recorder
            self._recorder = None
            capture_thread = self._capture_thread
            sender_thread = self._sender_thread
            self._capture_thread = None
            self._sender_thread = None
            self._capture_queue = None
        duration = recorder.stop() if recorder is not None else 0.0
        if capture_thread is not None:
            capture_thread.join(timeout=2)
        if sender_thread is not None:
            sender_thread.join(timeout=5)
        self._raise_background_error()
        return duration

    def _receive_loop(self) -> None:
        try:
            while True:
                with self._state_lock:
                    ws = self._ws
                    closing = self._closing
                    deadline = self._session_started_at + self.config.realtime_max_session_seconds
                if ws is None or closing:
                    return
                try:
                    event = self._recv(ws, deadline)
                except websocket.WebSocketConnectionClosedException:
                    with self._state_lock:
                        if self._closing or self._ws is None:
                            return
                    raise
                should_close = self._handle_event(event)
                if should_close:
                    self.close()
                    return
        except BaseException as exc:
            with self._state_lock:
                closing = self._closing
            if not closing:
                self._errors.put(exc)

    def _handle_event(self, event: dict[str, Any]) -> bool:
        event_type = event.get("type")
        if event_type == "error":
            if event_is_ignorable_control_error(event):
                LOGGER.warning("ignoring realtime control error: %s", event)
                return False
            RealtimeVoiceClient(self.config)._raise_for_error(event)

        audio_chunk: bytes | None = None
        with self._state_lock:
            self._last_activity_at = time.monotonic()

            if event_type == "conversation.item.input_audio_transcription.delta":
                if self._ignore_current_input or self._ignore_next_input_transcript:
                    return False
                self._input_transcript_parts.append(event.get("delta", ""))
            elif event_type == "conversation.item.input_audio_transcription.completed":
                if self._ignore_current_input or self._ignore_next_input_transcript:
                    transcript = event.get("transcript", "").strip()
                    if transcript:
                        LOGGER.info("ignored realtime input transcript during assistant response: %s", transcript)
                    self._ignore_current_input = False
                    self._ignore_next_input_transcript = False
                    self._input_transcript_parts = []
                    return False
                self._input_transcript_final = event.get("transcript", "").strip()
                if self._input_transcript_final:
                    LOGGER.info("realtime input transcript: %s", self._input_transcript_final)
                should_start_response = self._waiting_for_input_transcript
                self._waiting_for_input_transcript = False
                generation = self._pending_input_generation
                transcript = self._input_transcript_final or "".join(self._input_transcript_parts).strip()
                if should_start_response:
                    threading.Thread(
                        target=self._start_response_for_current_turn,
                        args=(transcript, generation),
                        name="realtime-active-memory",
                        daemon=True,
                    ).start()
            elif event_type == "input_audio_buffer.speech_started":
                LOGGER.info("realtime speech started")
                if self._response_create_pending or self._response_active or self._playback is not None:
                    LOGGER.info("ignoring realtime speech start during assistant response")
                    self._ignore_current_input = True
                    return False
                self._ignore_current_input = False
                self._ignore_next_input_transcript = False
                self._interrupt_response_locked()
                self._reset_turn_text_locked()
            elif event_type == "input_audio_buffer.speech_stopped":
                LOGGER.info("realtime speech stopped")
            elif event_type == "input_audio_buffer.committed":
                if self._ignore_current_input:
                    LOGGER.info("ignoring realtime input audio committed during assistant response")
                    self._ignore_next_input_transcript = True
                    return False
                self._turn_generation += 1
                self._pending_input_generation = self._turn_generation
                self._response_pending = True
                self._response_create_pending = False
                self._waiting_for_input_transcript = True
                self._response_wait_started_at = time.monotonic()
                LOGGER.info("realtime input audio committed by server VAD")
            elif event_type == "response.created":
                response = event.get("response", {})
                if self._recorder is not None or not self._response_create_pending:
                    LOGGER.info("ignoring stale realtime response.created while not waiting for output")
                    return False
                self._response_create_pending = False
                self._response_pending = False
                self._response_active = True
                self._active_response_id = response.get("id", "")
                if not self._response_wait_started_at:
                    self._response_wait_started_at = time.monotonic()
            elif event_type in {"response.output_item.created", "response.output_item.added"}:
                if not self._is_current_response_event_locked(event):
                    return False
                item = event.get("item", {})
                if item.get("type") == "message":
                    self._output_item_id = item.get("id", "")
                    self._output_audio_bytes = 0
            elif event_type == "response.output_audio.delta":
                if not self._is_current_response_event_locked(event):
                    return False
                self._response_wait_started_at = 0.0
                audio_chunk = base64.b64decode(event.get("delta", ""))
            elif event_type == "response.output_audio_transcript.delta":
                if not self._is_current_response_event_locked(event):
                    return False
                self._response_wait_started_at = 0.0
                self._output_transcript_parts.append(event.get("delta", ""))
            elif event_type == "response.output_audio_transcript.done":
                if not self._is_current_response_event_locked(event):
                    return False
                self._output_transcript_final = event.get("transcript", "").strip()
                if self._output_transcript_final:
                    LOGGER.info("realtime assistant transcript: %s", self._output_transcript_final)
            elif event_type == "response.done":
                if not self._is_current_response_event_locked(event):
                    return False
                return self._complete_response_locked(event.get("response", {}))
        if audio_chunk:
            self._write_output_audio(audio_chunk)
        return False

    def _is_current_response_event_locked(self, event: dict[str, Any]) -> bool:
        if self._recorder is not None:
            LOGGER.info("ignoring realtime response event while recording: %s", event.get("type"))
            return False
        if not self._response_active:
            LOGGER.info("ignoring realtime response event without an active response: %s", event.get("type"))
            return False
        response = event.get("response") if isinstance(event.get("response"), dict) else {}
        response_id = str(event.get("response_id") or response.get("id") or "").strip()
        if response_id and self._active_response_id and response_id != self._active_response_id:
            LOGGER.info("ignoring stale realtime response event for %s", response_id)
            return False
        if response_id and not self._active_response_id and event.get("type") != "response.created":
            LOGGER.info("ignoring realtime response event before active response id is known: %s", response_id)
            return False
        return True

    def _complete_response_locked(self, response: dict[str, Any]) -> bool:
        status = response.get("status", "")
        usage = response.get("usage")
        if usage:
            LOGGER.info("realtime response usage: %s", usage)
        self._response_pending = False
        self._response_create_pending = False
        self._response_active = False
        self._response_wait_started_at = 0.0
        self._waiting_for_input_transcript = False
        self._active_response_id = ""
        self._close_playback_locked(check=True)

        if status == "cancelled":
            LOGGER.info("realtime response cancelled")
            self._reset_turn_text_locked()
            return False

        tool_calls = response_function_calls(response)
        async_tool_calls = [call for call in tool_calls if call.get("name") in ASYNC_TASK_TOOL_NAMES]
        if async_tool_calls:
            self._tool_active = True
            self._response_wait_started_at = time.monotonic()
            self._tool_generation += 1
            generation = self._tool_generation
            threading.Thread(
                target=self._run_realtime_tool_calls,
                args=(generation, async_tool_calls),
                name="realtime-tool-calls",
                daemon=True,
            ).start()
            return False

        requested_close = response_requested_close(response)
        user_text = self._input_transcript_final or "".join(self._input_transcript_parts).strip()
        assistant_text = (
            self._output_transcript_final
            or "".join(self._output_transcript_parts).strip()
            or extract_response_text(response)
        )
        if user_text or assistant_text or requested_close:
            self._completed_turns.put(
                RealtimeTurnResult(
                    user_text=user_text,
                    assistant_text=assistant_text,
                    requested_close=requested_close,
                )
            )
        self._reset_turn_text_locked()
        if requested_close:
            LOGGER.info("model requested realtime session close")
        return requested_close

    def _run_realtime_tool_calls(self, generation: int, calls: list[dict[str, Any]]) -> None:
        try:
            for call in calls:
                output = self._execute_realtime_tool_call(call)
                with self._state_lock:
                    if generation != self._tool_generation or self._ws is None or self._closing:
                        LOGGER.info("discarding stale realtime tool result")
                        return
                self._send(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": call.get("call_id", ""),
                            "output": json.dumps(output, ensure_ascii=False),
                        },
                    }
                )
            with self._state_lock:
                if generation != self._tool_generation or self._ws is None or self._closing:
                    return
                self._tool_active = False
                self._response_pending = True
                self._response_create_pending = True
                self._response_wait_started_at = time.monotonic()
            self._send(
                {
                    "type": "response.create",
                    "response": {
                        "output_modalities": ["audio"],
                        "instructions": (
                            f"{realtime_turn_instructions(self.config)} "
                            "If a background task is still running, tell the user briefly and do not wait."
                        ),
                    },
                }
            )
        except BaseException as exc:
            self._errors.put(exc)

    def _start_response_for_current_turn(self, query: str = "", generation: int = 0) -> None:
        try:
            with self._state_lock:
                if (
                    self._ws is None
                    or self._closing
                    or generation != self._pending_input_generation
                ):
                    return
            memory_context = self.memory.active_context(query) if query.strip() else ""
            with self._state_lock:
                if (
                    self._ws is None
                    or self._closing
                    or generation != self._pending_input_generation
                ):
                    LOGGER.info("discarding stale realtime response start")
                    return
                self._response_pending = True
                self._response_create_pending = True
                self._response_wait_started_at = time.monotonic()
                self._last_activity_at = time.monotonic()
                self._send(
                    {
                        "type": "response.create",
                        "response": {
                            "output_modalities": ["audio"],
                            "instructions": realtime_turn_instructions(self.config, memory_context),
                        },
                    }
                )
            LOGGER.info(
                "started realtime response with active memory chars=%d",
                len(memory_context),
            )
        except BaseException as exc:
            self._errors.put(exc)

    def _execute_realtime_tool_call(self, call: dict[str, Any]) -> dict[str, Any]:
        name = call.get("name", "")
        arguments = parse_tool_arguments(call)
        LOGGER.info("running realtime tool %s with args keys=%s", name, sorted(arguments.keys()))
        if name in {START_TASK_TOOL_NAME, WEB_SEARCH_TOOL_NAME}:
            title = string_argument(arguments, "title")
            wakeup_on_complete = parse_bool_argument(arguments.get("wakeup_on_complete"), default=False)
            wakeup_on_progress = parse_bool_argument(arguments.get("wakeup_on_progress"), default=False)
            return self.tasks.start(
                self._latest_user_transcript(),
                history=self._latest_history(),
                title=title,
                wakeup_on_complete=wakeup_on_complete,
                wakeup_on_progress=wakeup_on_progress,
            )
        if name == STEER_TASK_TOOL_NAME:
            wakeup_on_complete = (
                parse_bool_argument(arguments.get("wakeup_on_complete"), default=False)
                if "wakeup_on_complete" in arguments
                else None
            )
            return self.tasks.steer(
                self._latest_user_transcript(),
                task_id=string_argument(arguments, "task_id"),
                wakeup_on_complete=wakeup_on_complete,
            )
        if name == LIST_TASKS_TOOL_NAME:
            return self.tasks.list(
                include_completed=parse_bool_argument(arguments.get("include_completed"), default=False)
            )
        if name == GET_TASK_TOOL_NAME:
            return self.tasks.get(
                task_id=string_argument(arguments, "task_id"),
                include_result=parse_bool_argument(arguments.get("include_result"), default=True),
            )
        if name == CANCEL_TASK_TOOL_NAME:
            return self.tasks.cancel(string_argument(arguments, "task_id"))
        if name == ADD_SCHEDULED_TASK_TOOL_NAME:
            return self.scheduled_tasks.add(
                title=string_argument(arguments, "title"),
                prompt=string_argument(arguments, "prompt"),
                run_at=string_argument(arguments, "run_at"),
                action=string_argument(arguments, "action", "speak"),
                repeat=string_argument(arguments, "repeat", "once"),
            )
        if name == LIST_SCHEDULED_TASKS_TOOL_NAME:
            return self.scheduled_tasks.list(
                include_inactive=parse_bool_argument(arguments.get("include_inactive"), default=False)
            )
        if name == DELETE_SCHEDULED_TASK_TOOL_NAME:
            return self.scheduled_tasks.delete(string_argument(arguments, "task_id"))
        if name == GET_WEATHER_TOOL_NAME:
            return self.weather.get_weather(
                location=string_argument(arguments, "location"),
                units=string_argument(arguments, "units", "metric"),
                no_cache=parse_bool_argument(arguments.get("no_cache"), default=False),
                forecast_days=max(0, min(5, int_argument(arguments, "forecast_days", 0))),
            )
        if name == SET_VOICE_VOLUME_TOOL_NAME:
            volumes = self.settings.set_voice_volume(int_argument(arguments, "level", self.player.volume_level()))
            level = self.player.set_volume_level(volumes["voice_volume"])
            return {"ok": True, "volume": level, "scale": level / 10.0}
        if name == GET_VOICE_VOLUME_TOOL_NAME:
            level = self.player.volume_level()
            return {"ok": True, "volume": level, "scale": level / 10.0}
        if name == GET_MUSIC_VOLUME_TOOL_NAME:
            status = self.music.status()
            level = int_argument(status, "volume", self.settings.music_volume())
            return {"ok": True, "volume": level, "scale": level / 10.0}
        if name == LIST_MUSIC_TOOL_NAME:
            return self.music.list()
        if name == GET_MUSIC_STATUS_TOOL_NAME:
            return self.music.status()
        if name == PLAY_MUSIC_TOOL_NAME:
            return self.music.request_play(string_argument(arguments, "query"))
        if name == PAUSE_MUSIC_TOOL_NAME:
            return self.music.pause(reason="user")
        if name == RESUME_MUSIC_TOOL_NAME:
            return self.music.request_resume()
        if name == STOP_MUSIC_TOOL_NAME:
            return self.music.stop()
        if name == SET_MUSIC_VOLUME_TOOL_NAME:
            return self.music.set_volume(int_argument(arguments, "level", self.settings.music_volume()))
        if name == TAKE_PICTURE_TOOL_NAME:
            return self.capture_camera_snapshot(
                self._latest_history(),
                reason=string_argument(arguments, "reason", "model requested picture"),
            )
        if name == START_CONTINUOUS_CAMERA_TOOL_NAME:
            interval = (
                float_argument(arguments, "interval_seconds", self.config.camera_continuous_interval_seconds)
                if "interval_seconds" in arguments
                else None
            )
            return self.start_continuous_camera(
                self._latest_history(),
                reason=string_argument(arguments, "reason", "continuous camera"),
                interval_seconds=interval,
            )
        if name == STOP_CONTINUOUS_CAMERA_TOOL_NAME:
            return self.stop_continuous_camera(reason=string_argument(arguments, "reason", "model requested stop"))
        if name == MEMORY_SEARCH_TOOL_NAME:
            return self.memory.search(
                string_argument(arguments, "query"),
                max_results=int_argument(arguments, "max_results", 5),
            )
        if name == MEMORY_LIST_TOOL_NAME:
            return self.memory.list_entries(
                include_forgotten=parse_bool_argument(arguments.get("include_forgotten"), default=False)
            )
        if name == MEMORY_ADD_TOOL_NAME:
            return self.memory.add_entry(
                text=string_argument(arguments, "text"),
                kind=string_argument(arguments, "kind", "note"),
                source="user",
            )
        if name == MEMORY_UPDATE_TOOL_NAME:
            return self.memory.update_entry(
                entry_id=string_argument(arguments, "entry_id"),
                text=string_argument(arguments, "text"),
            )
        if name == MEMORY_FORGET_TOOL_NAME:
            return self.memory.forget_entry(
                entry_id=string_argument(arguments, "entry_id"),
                query=string_argument(arguments, "query"),
                reason=string_argument(arguments, "reason"),
            )
        if name == MEMORY_GET_SOURCE_TOOL_NAME:
            return self.memory.get_source(
                path=string_argument(arguments, "path"),
                from_line=int_argument(arguments, "from_line", 1),
                lines=int_argument(arguments, "lines", 12),
            )
        return {"ok": False, "error": f"unknown realtime tool: {name}"}

    def _latest_user_transcript(self) -> str:
        with self._state_lock:
            return (self._input_transcript_final or "".join(self._input_transcript_parts)).strip()

    def _latest_history(self) -> list[Message]:
        with self._state_lock:
            return list(self._current_history)

    def pause_music_for_voice(self) -> bool:
        return self.music.pause_for_voice()

    def apply_deferred_music_after_voice(self) -> None:
        self.music.apply_deferred_after_voice()

    def trigger_scheduled_speech(self, history: Iterable[Message], title: str, prompt: str) -> bool:
        self._raise_background_error()
        title = title.strip() or "Scheduled task"
        prompt = prompt.strip()
        if not prompt:
            LOGGER.warning("not starting scheduled speech with empty prompt")
            return False
        with self._state_lock:
            if self._is_busy_locked():
                return False
            self._interrupt_response_locked()
            self._reset_turn_text_locked()
            self._input_transcript_final = f"[scheduled task: {title}] {prompt}"

        self.ensure_open(history)
        self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "A scheduled item is due now. "
                                f"Title: {title}. Message/request: {prompt}"
                            ),
                        }
                    ],
                },
            }
        )
        with self._state_lock:
            self._response_pending = True
            self._response_create_pending = True
            self._response_wait_started_at = time.monotonic()
            self._last_activity_at = time.monotonic()
        self._send(
            {
                "type": "response.create",
                "response": {
                    "output_modalities": ["audio"],
                    "instructions": (
                        f"{local_context(self.config)} "
                        "This scheduled item is due now. Speak naturally and concisely. "
                        "Do not use implementation terminology."
                    ),
                },
            }
        )
        LOGGER.info("started scheduled speech: %s", title)
        return True

    def pending_background_wakeups(self, limit: int = 1) -> list[dict[str, Any]]:
        return self.tasks.pending_wakeups(limit=limit)

    def mark_background_wakeup_reported(self, task_id: str, message_id: str = "") -> None:
        self.tasks.mark_wakeup_reported(task_id, message_id)

    def trigger_background_task_wakeup(self, history: Iterable[Message], task: dict[str, Any]) -> bool:
        self._raise_background_error()
        task_id = str(task.get("id") or "").strip()
        title = str(task.get("title") or task.get("progress") or task_id or "Background task").strip()
        wakeup = task.get("wakeup") if isinstance(task.get("wakeup"), dict) else {}
        wakeup_type = str(wakeup.get("type") or "completed")
        wakeup_text = str(wakeup.get("text") or "")
        if not task_id:
            return False
        with self._state_lock:
            if self._is_busy_locked():
                return False
            self._interrupt_response_locked()
            self._reset_turn_text_locked()
            self._input_transcript_final = f"[background task completed: {title}]"

        self.ensure_open(history)
        self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                (
                                    "A background task sent a progress update for the user. "
                                    if wakeup_type == "status_update"
                                    else "A background task has finished and was marked to wake the user. "
                                )
                                + "Say the relevant update aloud now. "
                                + f"Progress update text: {wakeup_text}. Task JSON: "
                                + f"{json.dumps(task, ensure_ascii=False)}"
                            ),
                        }
                    ],
                },
            }
        )
        with self._state_lock:
            self._response_pending = True
            self._response_create_pending = True
            self._response_wait_started_at = time.monotonic()
            self._last_activity_at = time.monotonic()
        self._send(
            {
                "type": "response.create",
                "response": {
                    "output_modalities": ["audio"],
                    "instructions": (
                        f"{local_context(self.config)} "
                        "A background task needs a spoken user update. Briefly tell the user the progress, result, or failure. "
                        "Do not mention internal tool names unless useful."
                    ),
                },
            }
        )
        LOGGER.info("started background task wakeup for %s", task_id)
        return True

    def _write_output_audio(self, chunk: bytes) -> None:
        if not chunk:
            return
        with self._state_lock:
            if self._recorder is not None:
                self._output_audio_bytes += len(chunk)
                return
            if self._playback is None:
                self._playback = self.player.open_stream(rate=24000, channels=1)
                self._playback.__enter__()
                self._playback_closing = False
            if self._playback_closing:
                return
            playback = self._playback
        playback.write(chunk)
        with self._state_lock:
            if self._playback is playback and not self._playback_closing:
                self._output_audio_bytes += len(chunk)

    def _interrupt_response_locked(self) -> None:
        ws_is_open = self._ws is not None
        self._abort_playback_locked()
        if ws_is_open and (self._response_pending or self._response_active):
            self._send(
                {
                    "event_id": f"cancel_{time.monotonic_ns()}",
                    "type": "response.cancel",
                }
            )
        if ws_is_open and self._output_item_id and self._output_audio_bytes:
            audio_end_ms = int(self._output_audio_bytes / 2 / 24000 * 1000)
            self._send(
                {
                    "event_id": f"truncate_{time.monotonic_ns()}",
                    "type": "conversation.item.truncate",
                    "item_id": self._output_item_id,
                    "content_index": 0,
                    "audio_end_ms": max(0, audio_end_ms),
                }
            )
        elif self._output_item_id and self._output_audio_bytes:
            LOGGER.info("skipping audio truncate because realtime session is already closed")
        self._response_pending = False
        self._response_create_pending = False
        self._response_active = False
        self._response_wait_started_at = 0.0
        self._waiting_for_input_transcript = False
        self._active_response_id = ""
        self._tool_active = False
        self._tool_generation += 1
        self._turn_generation += 1
        self._pending_input_generation = self._turn_generation
        self._output_item_id = ""
        self._output_audio_bytes = 0

    def _abort_playback_locked(self) -> None:
        playback = self._playback
        if playback is None:
            return
        self._playback = None
        self._playback_closing = False
        playback.abort()
        self._close_playback_in_background(playback, check=False)

    def _close_playback_in_background(self, playback: PcmOutputStream, check: bool) -> None:
        def close_playback() -> None:
            try:
                playback.close(check=check)
            except BaseException:
                LOGGER.exception("PCM playback close failed")
            finally:
                with self._state_lock:
                    if self._playback is playback:
                        self._playback = None
                        self._playback_closing = False

        threading.Thread(target=close_playback, name="pcm-playback-close", daemon=True).start()

    def _close_playback_locked(self, check: bool) -> None:
        playback = self._playback
        if playback is None or self._playback_closing:
            return
        self._playback_closing = True
        self._close_playback_in_background(playback, check=check)

    def _reset_turn_text_locked(self) -> None:
        self._input_transcript_parts = []
        self._input_transcript_final = ""
        self._output_transcript_parts = []
        self._output_transcript_final = ""
        self._output_item_id = ""
        self._output_audio_bytes = 0
        self._ignore_current_input = False
        self._ignore_next_input_transcript = False

    def _is_busy_locked(self) -> bool:
        return (
            self._recorder is not None
            or self._response_pending
            or self._response_active
            or self._tool_active
            or self._playback is not None
        )

    def _send(self, event: dict[str, Any]) -> None:
        with self._state_lock:
            ws = self._ws
        if ws is None:
            raise RuntimeError("realtime session is not open")
        with self._send_lock:
            ws.send(json.dumps(event))

    def _recv(self, ws: websocket.WebSocket, deadline: float) -> dict[str, Any]:
        return RealtimeVoiceClient(self.config)._recv(ws, deadline)

    def _raise_background_error(self) -> None:
        try:
            error = self._errors.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError("realtime background worker failed") from error


def iter_wav_pcm16_chunks(path: Path, expected_rate: int, chunk_frames: int = 4096) -> Iterator[bytes]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        rate = wav.getframerate()
        if channels != 1:
            raise RuntimeError(f"realtime audio must be mono; got {channels} channels in {path}")
        if sample_width != 2:
            raise RuntimeError(f"realtime audio must be 16-bit PCM; got sample width {sample_width} in {path}")
        if rate != expected_rate:
            raise RuntimeError(f"realtime audio must be {expected_rate} Hz; got {rate} Hz in {path}")
        while True:
            data = wav.readframes(chunk_frames)
            if not data:
                return
            yield data


def conversation_item_for_message(message: Message) -> dict[str, Any]:
    content_type = "input_text" if message.role == "user" else "output_text"
    return {
        "type": "message",
        "role": message.role,
        "content": [
            {
                "type": content_type,
                "text": message.content,
            }
        ],
    }


def conversation_item_for_camera_snapshot(snapshot: CameraSnapshot, detail: str = "auto", note: str = "") -> dict[str, Any]:
    image: dict[str, Any] = {
        "type": "input_image",
        "image_url": snapshot.data_url,
    }
    if detail.strip():
        image["detail"] = detail.strip()
    text = note.strip() or "Camera snapshot attached."
    return {
        "type": "message",
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": f"[Camera snapshot: {text}]",
            },
            image,
        ],
    }


def response_requested_close(response: dict[str, Any]) -> bool:
    return bool(response_function_calls(response, CLOSE_TOOL_NAME))


def response_function_calls(response: dict[str, Any], name: str | None = None) -> list[dict[str, Any]]:
    return [
        item
        for item in response.get("output", [])
        if item.get("type") == "function_call" and (name is None or item.get("name") == name)
    ]


def responses_function_calls(response: Any, name: str | None = None) -> list[Any]:
    if response is None:
        return []
    calls = []
    for item in object_get(response, "output", default=[]) or []:
        if object_get(item, "type") == "function_call" and (name is None or object_get(item, "name") == name):
            calls.append(item)
    return calls


def parse_tool_arguments(call: dict[str, Any]) -> dict[str, Any]:
    raw = call.get("arguments") or "{}"
    return parse_json_object(raw)


def parse_json_object(raw: str) -> dict[str, Any]:
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def object_get(obj: Any, name: str, default: Any = "") -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def string_argument(arguments: dict[str, Any], name: str, default: str = "") -> str:
    value = arguments.get(name, default)
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def int_argument(arguments: dict[str, Any], name: str, default: int) -> int:
    value = arguments.get(name, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed


def float_argument(arguments: dict[str, Any], name: str, default: float) -> float:
    value = arguments.get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_bool_argument(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def extract_response_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            text = content.get("transcript") or content.get("text")
            if text:
                parts.append(text)
    return "".join(parts).strip()


def camera_shutter_pcm(rate: int = 24000) -> bytes:
    samples = array("h")
    pattern = [
        (0.035, 3200.0, 0.55),
        (0.030, 0.0, 0.0),
        (0.055, 1900.0, 0.42),
    ]
    for duration, frequency, amplitude in pattern:
        count = max(1, int(rate * duration))
        for index in range(count):
            if frequency <= 0:
                samples.append(0)
                continue
            envelope = 1.0 - (index / count)
            value = int(32767 * amplitude * envelope * math.sin(2.0 * math.pi * frequency * index / rate))
            samples.append(value)
    return samples.tobytes()


def extract_responses_output_text(response: Any) -> str:
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", "")
            if text:
                parts.append(text)
    return "".join(parts).strip()


def event_is_ignorable_control_error(event: dict[str, Any]) -> bool:
    error = event.get("error", {})
    if not isinstance(error, dict):
        return False
    event_id = str(error.get("event_id") or event.get("event_id") or "")
    return event_id.startswith(("cancel_", "truncate_"))
