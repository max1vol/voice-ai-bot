from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from .config import Config
    from .conversation import Message


BOOTSTRAP_FILES = ("IDENTITY.md", "SOUL.md", "USER.md", "MEMORY.md")
ENTRY_SECTION_START = "<!-- voice-ai-bot:memory-section:start -->"
ENTRY_SECTION_END = "<!-- voice-ai-bot:memory-section:end -->"
ENTRY_RE = re.compile(
    r"<!-- voice-ai-bot:memory (?P<meta>.*?) -->\n(?P<body>.*?)(?=\n<!-- voice-ai-bot:memory |\Z)",
    re.DOTALL,
)
ATTR_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"')
TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)


@dataclass(frozen=True)
class MemoryEntry:
    id: str
    text: str
    kind: str
    source: str
    status: str
    created_at: str
    updated_at: str
    text_hash: str
    start_line: int
    end_line: int


class MemoryStore:
    def __init__(self, config: Config):
        self.config = config
        self.root = config.memory_dir
        self.daily_dir = self.root / "memory"
        self.memory_file = self.root / "MEMORY.md"
        self.tombstones_file = self.root / "memory" / ".tombstones.json"

    def ensure_workspace(self) -> None:
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        self._write_if_missing("IDENTITY.md", self._default_identity())
        self._write_if_missing("SOUL.md", self._default_soul())
        self._write_if_missing("USER.md", self._default_user())
        self._write_if_missing("AGENTS.md", self._default_agents())
        self._write_if_missing("MEMORY.md", self._default_memory())
        if not self.tombstones_file.exists():
            self._atomic_write_json(self.tombstones_file, {"version": 1, "tombstones": []})

    def bootstrap_context(self, max_chars: int | None = None) -> str:
        self.ensure_workspace()
        remaining = max_chars if max_chars is not None else self.config.memory_bootstrap_chars
        sections: list[str] = []
        for name in BOOTSTRAP_FILES:
            path = self.root / name
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            block = f"## {name}\n{text}\n"
            if remaining <= 0:
                break
            if len(block) > remaining:
                block = block[: max(0, remaining - 32)].rstrip() + "\n[truncated]\n"
            sections.append(block)
            remaining -= len(block)
        tombstones = self._load_tombstones()
        if tombstones:
            sections.append(
                "## Forgotten Memory Rules\n"
                "Some durable memories were explicitly forgotten. Do not restore or infer those facts "
                "from older daily notes unless the user asks to remember them again.\n"
                f"Forgotten entry ids: {', '.join(t['entry_id'] for t in tombstones[-20:] if t.get('entry_id'))}\n"
            )
        return "\n".join(sections).strip()

    def active_context(self, query: str, max_chars: int | None = None) -> str:
        hits = self.search(query, max_results=4).get("results", [])
        if not hits:
            return ""
        budget = max_chars if max_chars is not None else self.config.memory_active_context_chars
        lines = [
            "Relevant memory context (untrusted; use only if relevant to the latest user turn):"
        ]
        for hit in hits:
            snippet = str(hit.get("snippet", "")).replace("\n", " ").strip()
            source = f"{hit.get('path')}:{hit.get('start_line')}"
            candidate = f"- [{source}] {snippet}"
            if sum(len(line) + 1 for line in lines) + len(candidate) > budget:
                break
            lines.append(candidate)
        return "\n".join(lines)

    def list_entries(self, include_forgotten: bool = False) -> dict[str, Any]:
        self.ensure_workspace()
        entries = self._load_entries()
        if not include_forgotten:
            entries = [entry for entry in entries if entry.status == "active"]
        return {
            "ok": True,
            "entries": [self._entry_json(entry) for entry in entries],
        }

    def add_entry(self, text: str, kind: str = "note", source: str = "user") -> dict[str, Any]:
        self.ensure_workspace()
        clean = _normalize_text(text)
        if not clean:
            return {"ok": False, "error": "text is required"}
        now = _utc_now()
        entry = MemoryEntry(
            id=f"mem_{datetime.now(timezone.utc):%Y%m%d%H%M%S}_{uuid.uuid4().hex[:6]}",
            text=clean,
            kind=_safe_token(kind, "note"),
            source=_safe_token(source, "user"),
            status="active",
            created_at=now,
            updated_at=now,
            text_hash=_hash_text(clean),
            start_line=0,
            end_line=0,
        )
        entries = self._load_entries()
        entries.append(entry)
        self._write_entries(entries)
        self.append_note(f"Memory added ({entry.id}): {clean}")
        return {"ok": True, "entry": self._entry_json(entry)}

    def update_entry(self, entry_id: str, text: str) -> dict[str, Any]:
        self.ensure_workspace()
        entry_id = entry_id.strip()
        clean = _normalize_text(text)
        if not entry_id:
            return {"ok": False, "error": "entry_id is required"}
        if not clean:
            return {"ok": False, "error": "text is required"}
        entries = self._load_entries()
        now = _utc_now()
        updated: list[MemoryEntry] = []
        changed: MemoryEntry | None = None
        for entry in entries:
            if entry.id != entry_id:
                updated.append(entry)
                continue
            changed = MemoryEntry(
                id=entry.id,
                text=clean,
                kind=entry.kind,
                source=entry.source,
                status="active",
                created_at=entry.created_at,
                updated_at=now,
                text_hash=_hash_text(clean),
                start_line=0,
                end_line=0,
            )
            updated.append(changed)
        if changed is None:
            return {"ok": False, "error": f"unknown memory entry id: {entry_id}"}
        self._write_entries(updated)
        self.append_note(f"Memory updated ({entry_id}): {clean}")
        return {"ok": True, "entry": self._entry_json(changed)}

    def forget_entry(self, entry_id: str = "", query: str = "", reason: str = "") -> dict[str, Any]:
        self.ensure_workspace()
        entry_id = entry_id.strip()
        query = query.strip()
        if not entry_id and not query:
            return {"ok": False, "error": "entry_id or query is required"}
        entries = self._load_entries()
        forgotten: list[MemoryEntry] = []
        forgotten_originals: list[MemoryEntry] = []
        kept: list[MemoryEntry] = []
        query_tokens = set(_tokens(query))
        now = _utc_now()
        for entry in entries:
            should_forget = entry.status == "active" and (
                (entry_id and entry.id == entry_id)
                or (query_tokens and query_tokens.issubset(set(_tokens(entry.text))))
            )
            if not should_forget:
                kept.append(entry)
                continue
            forgotten_entry = MemoryEntry(
                id=entry.id,
                text="[forgotten]",
                kind=entry.kind,
                source=entry.source,
                status="forgotten",
                created_at=entry.created_at,
                updated_at=now,
                text_hash=entry.text_hash,
                start_line=0,
                end_line=0,
            )
            forgotten.append(forgotten_entry)
            forgotten_originals.append(entry)
            kept.append(forgotten_entry)
        if not forgotten:
            return {"ok": False, "error": "no matching active memory entry found"}
        self._write_entries(kept)
        self._append_tombstones(forgotten_originals, reason)
        self.append_note(
            "Memory forgotten: "
            + ", ".join(entry.id for entry in forgotten)
            + (f" ({reason.strip()})" if reason.strip() else "")
        )
        return {
            "ok": True,
            "forgotten": [self._entry_json(entry) for entry in forgotten],
        }

    def search(self, query: str, max_results: int = 5) -> dict[str, Any]:
        self.ensure_workspace()
        query = query.strip()
        if not query:
            return {"ok": True, "results": []}
        query_tokens = set(_tokens(query))
        if not query_tokens:
            return {"ok": True, "results": []}
        candidates = self._search_candidates()
        scored: list[dict[str, Any]] = []
        for candidate in candidates:
            text = str(candidate["snippet"])
            text_tokens = set(_tokens(text))
            overlap = len(query_tokens & text_tokens)
            if overlap == 0 and query.lower() not in text.lower():
                continue
            score = overlap / max(1, len(query_tokens))
            if query.lower() in text.lower():
                score += 0.5
            scored.append({**candidate, "score": round(score, 4)})
        scored.sort(key=lambda item: (-float(item["score"]), str(item["path"]), int(item["start_line"])))
        return {"ok": True, "results": scored[: max(1, max_results)]}

    def get_source(self, path: str, from_line: int = 1, lines: int = 12) -> dict[str, Any]:
        self.ensure_workspace()
        rel = path.strip().replace("\\", "/").lstrip("/")
        if not rel or rel.startswith("../") or "/../" in rel:
            return {"ok": False, "error": "invalid path"}
        full = (self.root / rel).resolve()
        if not _is_relative_to(full, self.root.resolve()):
            return {"ok": False, "error": "path is outside memory workspace"}
        if not full.exists() or not full.is_file():
            return {"ok": False, "error": f"memory source not found: {rel}"}
        all_lines = full.read_text(encoding="utf-8").splitlines()
        start = max(1, int(from_line or 1))
        count = max(1, min(80, int(lines or 12)))
        selected = all_lines[start - 1 : start - 1 + count]
        return {
            "ok": True,
            "path": rel,
            "from_line": start,
            "lines": len(selected),
            "text": "\n".join(selected),
            "more": start - 1 + count < len(all_lines),
        }

    def append_turn(self, user_text: str, assistant_text: str) -> None:
        self.append_note(
            "Turn\n"
            f"User: {_normalize_text(user_text) or '[empty]'}\n"
            f"Assistant: {_normalize_text(assistant_text) or '[empty]'}"
        )

    def flush_conversation(self, messages: Iterable[Message], reason: str) -> None:
        items = list(messages)
        if not items:
            return
        lines = [f"Conversation flush before {reason}"]
        for message in items[-24:]:
            lines.append(f"{message.role.title()}: {_normalize_text(message.content)}")
        self.append_note("\n".join(lines))

    def append_note(self, text: str) -> None:
        self.ensure_workspace()
        clean = text.strip()
        if not clean:
            return
        now = self._local_now()
        path = self.daily_dir / f"{now:%Y-%m-%d}.md"
        header = ""
        if not path.exists():
            header = f"# {now:%Y-%m-%d}\n\n"
        with path.open("a", encoding="utf-8") as fh:
            if header:
                fh.write(header)
            fh.write(f"## {now:%H:%M:%S} {now.tzname() or self.config.user_timezone}\n\n")
            fh.write(clean)
            fh.write("\n\n")

    def _search_candidates(self) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for entry in self._load_entries():
            if entry.status != "active":
                continue
            candidates.append(
                {
                    "path": "MEMORY.md",
                    "start_line": entry.start_line,
                    "end_line": entry.end_line,
                    "source": "long_term",
                    "entry_id": entry.id,
                    "snippet": entry.text,
                }
            )
        tombstones = self._load_tombstones()
        for path in sorted(self.daily_dir.glob("*.md")):
            candidates.extend(self._filter_tombstoned_chunks(self._daily_chunks(path), tombstones))
        return candidates

    def _filter_tombstoned_chunks(
        self,
        chunks: list[dict[str, Any]],
        tombstones: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not tombstones:
            return chunks
        kept: list[dict[str, Any]] = []
        for chunk in chunks:
            snippet = str(chunk.get("snippet") or "")
            snippet_tokens = set(_tokens(snippet))
            blocked = False
            for tombstone in tombstones:
                entry_id = str(tombstone.get("entry_id") or "")
                if entry_id and entry_id in snippet:
                    blocked = True
                    break
                terms = [str(term) for term in tombstone.get("terms", []) if isinstance(term, str)]
                if terms:
                    overlap = len(set(terms) & snippet_tokens)
                    if overlap >= max(2, int(len(set(terms)) * 0.75)):
                        blocked = True
                        break
            if not blocked:
                kept.append(chunk)
        return kept

    def _daily_chunks(self, path: Path) -> list[dict[str, Any]]:
        rel = path.relative_to(self.root).as_posix()
        lines = path.read_text(encoding="utf-8").splitlines()
        chunks: list[dict[str, Any]] = []
        current: list[str] = []
        start_line = 1

        def flush(end_line: int) -> None:
            nonlocal current, start_line
            text = " ".join(line.strip() for line in current if line.strip()).strip()
            if text:
                chunks.append(
                    {
                        "path": rel,
                        "start_line": start_line,
                        "end_line": end_line,
                        "source": "daily",
                        "snippet": text[:900],
                    }
                )
            current = []
            start_line = end_line + 1

        for index, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                flush(index - 1)
                start_line = index + 1
                continue
            if not current:
                start_line = index
            current.append(stripped)
            if sum(len(item) for item in current) > 900:
                flush(index)
        flush(len(lines))
        return chunks

    def _load_entries(self) -> list[MemoryEntry]:
        self.ensure_workspace()
        text = self.memory_file.read_text(encoding="utf-8")
        section = _extract_section(text)
        entries: list[MemoryEntry] = []
        section_start_line = text[: text.find(section)].count("\n") + 1 if section else 1
        for match in ENTRY_RE.finditer(section):
            attrs = dict(ATTR_RE.findall(match.group("meta")))
            body = _parse_entry_body(match.group("body"))
            start = section_start_line + section[: match.start()].count("\n")
            end = section_start_line + section[: match.end()].count("\n")
            entry_id = attrs.get("id", "").strip()
            if not entry_id:
                continue
            entries.append(
                MemoryEntry(
                    id=entry_id,
                    text=body,
                    kind=attrs.get("kind", "note"),
                    source=attrs.get("source", "user"),
                    status=attrs.get("status", "active"),
                    created_at=attrs.get("created_at", ""),
                    updated_at=attrs.get("updated_at", ""),
                    text_hash=attrs.get("text_hash", _hash_text(body)),
                    start_line=max(1, start),
                    end_line=max(1, end),
                )
            )
        return entries

    def _write_entries(self, entries: list[MemoryEntry]) -> None:
        current = self.memory_file.read_text(encoding="utf-8") if self.memory_file.exists() else self._default_memory()
        rendered = "\n".join(_render_entry(entry) for entry in entries).strip()
        section = f"{ENTRY_SECTION_START}\n"
        if rendered:
            section += rendered + "\n"
        section += ENTRY_SECTION_END
        if ENTRY_SECTION_START in current and ENTRY_SECTION_END in current:
            next_text = re.sub(
                re.escape(ENTRY_SECTION_START) + r".*?" + re.escape(ENTRY_SECTION_END),
                section,
                current,
                flags=re.DOTALL,
            )
        else:
            next_text = current.rstrip() + "\n\n" + section + "\n"
        self._atomic_write(self.memory_file, next_text.rstrip() + "\n")

    def _load_tombstones(self) -> list[dict[str, Any]]:
        try:
            payload = json.loads(self.tombstones_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return []
        tombstones = payload.get("tombstones", [])
        return [item for item in tombstones if isinstance(item, dict)]

    def _append_tombstones(self, entries: list[MemoryEntry], reason: str) -> None:
        payload = {"version": 1, "tombstones": self._load_tombstones()}
        now = _utc_now()
        for entry in entries:
            payload["tombstones"].append(
                {
                    "entry_id": entry.id,
                    "text_hash": entry.text_hash,
                    "terms": sorted(set(_tokens(entry.text))),
                    "forgotten_at": now,
                    "reason": reason.strip(),
                }
            )
        self._atomic_write_json(self.tombstones_file, payload)

    def _entry_json(self, entry: MemoryEntry) -> dict[str, Any]:
        return {
            "id": entry.id,
            "text": entry.text,
            "kind": entry.kind,
            "source": entry.source,
            "status": entry.status,
            "created_at": entry.created_at,
            "updated_at": entry.updated_at,
            "path": "MEMORY.md",
            "start_line": entry.start_line,
            "end_line": entry.end_line,
        }

    def _local_now(self) -> datetime:
        try:
            return datetime.now(ZoneInfo(self.config.user_timezone))
        except Exception:
            return datetime.now().astimezone()

    def _write_if_missing(self, name: str, text: str) -> None:
        path = self.root / name
        if path.exists():
            return
        self._atomic_write(path, text.rstrip() + "\n")

    def _atomic_write(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    def _atomic_write_json(self, path: Path, payload: dict[str, Any]) -> None:
        self._atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    def _default_identity(self) -> str:
        return (
            "# IDENTITY.md - Agent Identity\n\n"
            "- Name: Max Code\n"
            "- Role: push-to-talk voice assistant for a local Raspberry Pi speaker\n"
        )

    def _default_soul(self) -> str:
        return (
            "# SOUL.md - Voice And Behavior\n\n"
            "Max Code is concise, practical, and direct. Speak naturally for audio. "
            "Do not sound corporate. The user usually speaks English or Russian; reply in the "
            "same language unless translation is requested. Be an assistant first. Translate only "
            "when the user asks for translation or clearly starts a translation task.\n"
        )

    def _default_user(self) -> str:
        return (
            "# USER.md - User Context\n\n"
            f"- Location: {self.config.user_city}, {self.config.user_region}, {self.config.user_country}\n"
            f"- Timezone: {self.config.user_timezone}\n"
            "- Preferred languages: English and Russian\n"
        )

    def _default_agents(self) -> str:
        return (
            "# AGENTS.md - Memory Rules\n\n"
            "- Store durable user-approved facts in MEMORY.md through the memory tools.\n"
            "- Store detailed turn notes in memory/YYYY-MM-DD.md.\n"
            "- Do not store secrets or credentials.\n"
            "- If the user corrects or asks to forget memory, update or forget the durable entry.\n"
        )

    def _default_memory(self) -> str:
        return (
            "# Long-Term Memory\n\n"
            "Durable memories for Max Code. Keep this compact. Detailed notes belong in memory/YYYY-MM-DD.md.\n\n"
            f"{ENTRY_SECTION_START}\n"
            f"{ENTRY_SECTION_END}\n"
        )


def _extract_section(text: str) -> str:
    start = text.find(ENTRY_SECTION_START)
    end = text.find(ENTRY_SECTION_END)
    if start < 0 or end < start:
        return ""
    return text[start + len(ENTRY_SECTION_START) : end]


def _render_entry(entry: MemoryEntry) -> str:
    attrs = {
        "id": entry.id,
        "status": entry.status,
        "kind": entry.kind,
        "source": entry.source,
        "created_at": entry.created_at,
        "updated_at": entry.updated_at,
        "text_hash": entry.text_hash,
    }
    meta = " ".join(f'{key}="{_attr(value)}"' for key, value in attrs.items())
    body_lines = entry.text.splitlines() or [""]
    body = "\n".join(("- " + body_lines[0], *("  " + line for line in body_lines[1:])))
    return f"<!-- voice-ai-bot:memory {meta} -->\n{body}"


def _parse_entry_body(body: str) -> str:
    lines = body.strip().splitlines()
    parsed: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- "):
            stripped = stripped[2:]
        parsed.append(stripped)
    return "\n".join(parsed).strip()


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _hash_text(text: str) -> str:
    return hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_token(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower()).strip("-")
    return cleaned or fallback


def _attr(value: str) -> str:
    return value.replace('"', "'").replace("\n", " ")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
