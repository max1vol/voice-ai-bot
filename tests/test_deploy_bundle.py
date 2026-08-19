from __future__ import annotations

import compileall
import subprocess
import sys
from pathlib import Path

from voice_ai_bot.deploy_bundle import BUNDLE_MANIFEST_NAME, build_pi_deploy_bundle

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pi_install_script_keeps_fast_deploy_path():
    script = (REPO_ROOT / "scripts" / "install_pi.sh").read_text(encoding="utf-8")

    assert "rsync -az --delete --exclude .venv" in script
    assert 'echo "system dependencies already present; skipping apt"' in script
    assert 'if [ "$needs_apt" -eq 1 ]; then' in script
    assert "apt-get update" in script
    assert "if [ ! -x .venv/bin/python ]; then" in script
    assert "pip install --upgrade --no-build-isolation ." in script
    assert "pip install --upgrade pip setuptools wheel" not in script


def test_pi_deploy_bundle_contains_only_runtime_files(tmp_path):
    env_source = tmp_path / "bundle.env"
    env_source.write_text("OPENAI_API_KEY=test-key\n")
    output_dir = tmp_path / "bundle"

    manifest = build_pi_deploy_bundle(REPO_ROOT, output_dir, env_source=env_source)

    assert set(path.name for path in output_dir.iterdir()) == {
        ".env",
        BUNDLE_MANIFEST_NAME,
        "pyproject.toml",
        "src",
        "systemd",
    }
    assert not (output_dir / "watch").exists()
    assert not (output_dir / "docs").exists()
    assert not (output_dir / "tests").exists()
    assert not (output_dir / "README.md").exists()
    assert not (output_dir / ".DS_Store").exists()
    assert not (output_dir / "src" / "voice_ai_bot.egg-info").exists()
    assert manifest["env_included"] is True
    assert "src/voice_ai_bot/debug_web.py" in manifest["paths"]
    assert "src/voice_ai_bot/realtime_voice.py" in manifest["paths"]
    assert "systemd/voice-ai-bot.service" in manifest["paths"]
    assert "systemd/voice-ai-bot-debug.service" in manifest["paths"]
    assert ".env" in manifest["paths"]


def test_pi_deploy_bundle_is_installable_and_compilable(tmp_path):
    env_source = tmp_path / "bundle.env"
    env_source.write_text("OPENAI_API_KEY=test-key\n")
    output_dir = tmp_path / "bundle"
    build_pi_deploy_bundle(REPO_ROOT, output_dir, env_source=env_source)

    assert compileall.compile_dir(str(output_dir / "src"), quiet=1)

    install_root = tmp_path / "install-root"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--target",
            str(install_root),
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert compileall.compile_dir(str(install_root / "voice_ai_bot"), quiet=1)
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.path.insert(0, sys.argv[1]); "
                "from importlib import metadata; "
                "import voice_ai_bot; "
                "import voice_ai_bot.config; "
                "import voice_ai_bot.conversation; "
                "import voice_ai_bot.debug_web; "
                "import voice_ai_bot.memory; "
                "import voice_ai_bot.music; "
                "import voice_ai_bot.realtime_voice; "
                "dist = metadata.distribution('voice-ai-bot'); "
                "scripts = {entry.name for entry in dist.entry_points if entry.group == 'console_scripts'}; "
                "assert 'voice-ai-bot-debug-web' in scripts; "
                "print(voice_ai_bot.__version__)"
            ),
            str(install_root),
        ],
        check=True,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
