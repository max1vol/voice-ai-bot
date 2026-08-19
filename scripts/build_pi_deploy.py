#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from voice_ai_bot.deploy_bundle import build_pi_deploy_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a minimal Raspberry Pi deploy bundle.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env", type=Path, default=None, help="Path to .env to include in the bundle.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    env_path = args.env.resolve() if args.env is not None else (repo_root / ".env")
    if not env_path.exists():
        raise SystemExit(f"missing deploy env file: {env_path}")
    build_pi_deploy_bundle(repo_root, args.output, env_source=env_path)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
