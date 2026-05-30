from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str

    def to_api(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class ConversationStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> list[Message]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        messages = data.get("messages", [])
        loaded: list[Message] = []
        for item in messages:
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                loaded.append(Message(role=role, content=content))
        return loaded

    def append_pair(self, user_text: str, assistant_text: str) -> None:
        messages = self.load()
        messages.append(Message(role="user", content=user_text))
        messages.append(Message(role="assistant", content=assistant_text))
        self.save(messages)

    def save(self, messages: Iterable[Message]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "messages": [message.__dict__ for message in messages],
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(tmp, self.path)

    def clear(self) -> None:
        self.save([])
