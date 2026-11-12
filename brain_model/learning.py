"""Versioned contact-level pair STDP and experimental dopamine eligibility."""

import math

import torch

from .config import parse_learning

__all__ = ["ContactLearner", "parse_learning"]


class ContactLearner:
    """Traces are updated after applying each step's rule; weights never alias w0."""

    def __init__(self, topology: dict, config: dict, dt_ms: float, weights: dict | None = None):
        self.config = config
        self.dt_ms = dt_ms
        self.contacts = topology["contacts"]
        self.region = topology["region_of_cell"]
        self.weights = {
            key: (weights[key] if weights is not None else c["weight"]).clone()
            for key, c in self.contacts.items()
        }
        self.pre = torch.zeros(topology["n"], device=topology["device"])
        self.post = torch.zeros_like(self.pre)
        self.eligibility = {key: torch.zeros_like(c["weight"]) for key, c in self.contacts.items()}

    def step(self, spikes: torch.Tensor, dopamine: torch.Tensor, baseline: torch.Tensor) -> None:
        cfg = self.config
        if cfg["rule"] == "none":
            return
        self.pre.mul_(math.exp(-self.dt_ms / cfg["tau_plus_ms"]))
        self.post.mul_(math.exp(-self.dt_ms / cfg["tau_minus_ms"]))
        for key, c in self.contacts.items():
            if not key.endswith("glu"):
                continue
            source, target = c["source"], c["target"]
            weight = self.weights[key]
            low, high = 0.1 * c["weight"], 2 * c["weight"]
            causal = self.pre[source] * spikes[target]
            if cfg["rule"] == "pair_stdp":
                span = (high - low).clamp(min=torch.finfo(weight.dtype).tiny)
                delta = cfg["g"] * (
                    cfg["A_plus"] * causal * (high - weight) / span
                    - cfg["A_minus"] * self.post[target] * spikes[source] * (weight - low) / span
                )
            else:
                trace = self.eligibility[key]
                trace.mul_(math.exp(-self.dt_ms / (1000 * cfg["trace_tau_s"]))).add_(causal)
                difference = (dopamine - baseline)[self.region[target]]
                delta = (
                    cfg["eta"]
                    * (self.dt_ms / 1000)
                    * trace
                    * torch.where(difference.abs() >= cfg["da_threshold"], difference, 0.0)
                )
            self.weights[key] = torch.where(c["plastic"], (weight + delta).clamp(low, high), weight)
        self.pre.add_(spikes)
        self.post.add_(spikes)

    def tensors(self) -> dict[str, torch.Tensor]:
        return {
            "pre": self.pre,
            "post": self.post,
            **{f"weights.{key}": value for key, value in self.weights.items()},
            **{f"eligibility.{key}": value for key, value in self.eligibility.items()},
        }

    def restore(self, tensors: dict[str, torch.Tensor]) -> None:
        self.pre = tensors["pre"].clone()
        self.post = tensors["post"].clone()
        for key in self.contacts:
            self.weights[key] = tensors[f"weights.{key}"].clone()
            self.eligibility[key] = tensors[f"eligibility.{key}"].clone()
