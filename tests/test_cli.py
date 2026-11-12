import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def test_empty_scenario_selection_fails():
    result = subprocess.run(
        [sys.executable, str(PROJECT / "tests/check_scenarios.py"), "--only", "__nothing__"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "No scenarios" in result.stderr


def test_invalid_benchmark_size_fails():
    result = subprocess.run(
        [sys.executable, str(PROJECT / "benchmark.py"), "1", "--steps", "1", "--cpu"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0


def test_zero_benchmark_steps_fail():
    result = subprocess.run(
        [sys.executable, str(PROJECT / "benchmark.py"), "16384", "--steps", "0", "--cpu"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "steps" in result.stderr
