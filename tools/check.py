"""Run the pinned local checks without changing source or downloading tools."""

import argparse
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", action="store_true", help="also test installed Chrome")
    parser.add_argument("--coverage", action="store_true")
    args = parser.parse_args()
    node = shutil.which("node")
    if node is None:
        parser.error("Node.js is required; install the version listed in README.md")
    (PROJECT / ".cache").mkdir(exist_ok=True)
    pytest = [sys.executable, "-m", "pytest", "-q", f"--basetemp=.cache/check-{uuid.uuid4().hex}"]
    if args.coverage:
        pytest += ["--cov", "--cov-report=term-missing"]
    checks = [
        [
            sys.executable,
            "-m",
            "ruff",
            "format",
            "--check",
            "brain_model",
            "research",
            "tests",
            "tools",
            "server.py",
            "benchmark.py",
        ],
        [sys.executable, "-m", "ruff", "check", "."],
        [node, "node_modules/pyright/index.js"],
        [node, "node_modules/@biomejs/biome/bin/biome", "check", "web", "tests"],
        [node, "node_modules/typescript/bin/tsc", "--noEmit"],
        [node, "--test", "tests/memory.test.mjs", "tests/charts.test.mjs"],
        pytest,
    ]
    if args.browser:
        checks.append(
            [
                node,
                "node_modules/@playwright/test/cli.js",
                "test",
                "tests/ui.spec.mjs",
                "tests/ui-live.spec.mjs",
                "--workers=1",
            ]
        )
    failed = 0
    environment = os.environ | {"PYTHONUTF8": "1", "HUMAN_EMOTION_DEVICE": "cpu"}
    for command in checks:
        print("Running", " ".join(command), flush=True)
        failed += subprocess.run(command, cwd=PROJECT, env=environment, check=False).returncode != 0
    return int(failed > 0)


if __name__ == "__main__":
    sys.exit(main())
