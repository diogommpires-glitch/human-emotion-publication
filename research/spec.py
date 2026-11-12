"""Prespecified protocol parameters and boundary validation."""

import math
from dataclasses import asdict, dataclass, fields

SEEDS = (11, 23, 37, 53, 71)
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
DATASETS = {"ds005540": "1.0.7", "ds004892": "1.0.2", "ds004872": "1.0.3"}
GAINS = ("sensory_gain", "threat_gain", "reward_gain", "social_gain", "stress_gain", "control_gain")


@dataclass(frozen=True)
class ExperimentSpec:
    protocol: str = "synthetic"
    schema_version: int = 1
    seed: int = 11
    duration_seconds: float = 1.0
    chunk_seconds: float = 1.0
    sample_hz: float = 100.0
    source_path: str | None = None
    dataset_manifest: str | None = None
    neurons: int = 256
    budget_bytes: int = 8 * 1024**3
    bootstrap_iterations: int = 10000
    permutation_iterations: int = 1000

    def __post_init__(self):
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if self.protocol not in {
            "synthetic",
            "mechanics",
            "eeg",
            "fmri",
            "ratings",
            "session_order",
        }:
            raise ValueError(f"unknown protocol {self.protocol!r}")
        if self.seed not in SEEDS:
            raise ValueError(f"seed must be one of {SEEDS}")
        self._validate_bounds()
        for name in ("source_path", "dataset_manifest"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str)
                or not value.strip()
                or "\0" in value
                or len(value) > 4096
            ):
                raise ValueError(f"{name} must be a nonempty local path string or null")
        if self.protocol in {"eeg", "fmri", "ratings", "session_order"} and not self.source_path:
            raise ValueError("empirical protocols require a prepared cohort source_path")
        if self.protocol in {"eeg", "fmri", "ratings"} and (
            self.bootstrap_iterations != 10000 or self.permutation_iterations != 1000
        ):
            raise ValueError(
                "confirmatory protocols require 10000 bootstraps and 1000 temporal permutations"
            )

    def _validate_bounds(self):
        for name, maximum in (
            ("duration_seconds", 86400),
            ("chunk_seconds", 300),
            ("sample_hz", 10000),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value <= maximum
            ):
                raise ValueError(f"{name} must be finite and in (0, {maximum}]")
        for name in ("neurons", "budget_bytes", "bootstrap_iterations", "permutation_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 64 <= self.neurons <= 131072:
            raise ValueError("CPU research neuron count must be in [64, 131072]")

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise TypeError("experiment spec must be a JSON object")
        unknown = set(values) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"unknown experiment fields: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self):
        return asdict(self)
