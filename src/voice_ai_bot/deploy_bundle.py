from __future__ import annotations

import json
import shutil
from pathlib import Path

BUNDLE_MANIFEST_NAME = "pi-deploy-manifest.json"
RUNTIME_DIRS = ("src", "systemd")
RUNTIME_FILES = ("pyproject.toml",)
IGNORED_NAMES = {".DS_Store", ".pytest_cache", "__pycache__"}


def build_pi_deploy_bundle(repo_root: Path, output_dir: Path, env_source: Path | None = None) -> dict[str, object]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for relative in RUNTIME_FILES:
        source = repo_root / relative
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(relative)

    for relative in RUNTIME_DIRS:
        source = repo_root / relative
        target = output_dir / relative
        shutil.copytree(source, target, ignore=_copy_ignore)
        copied.extend(_relative_file_paths(target, output_dir))

    if env_source is not None:
        env_source = env_source.resolve()
        shutil.copy2(env_source, output_dir / ".env")
        copied.append(".env")

    manifest = {
        "bundle": "voice-ai-bot-pi",
        "top_level_dirs": list(RUNTIME_DIRS),
        "top_level_files": list(RUNTIME_FILES),
        "env_included": env_source is not None,
        "paths": sorted(copied),
    }
    (output_dir / BUNDLE_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _relative_file_paths(root: Path, base: Path) -> list[str]:
    return sorted(path.relative_to(base).as_posix() for path in root.rglob("*") if path.is_file())


def _copy_ignore(_directory: str, entries: list[str]) -> set[str]:
    ignored = set()
    for entry in entries:
        if entry in IGNORED_NAMES:
            ignored.add(entry)
        elif entry.endswith(".egg-info"):
            ignored.add(entry)
        elif entry.endswith((".pyc", ".pyo")):
            ignored.add(entry)
    return ignored
