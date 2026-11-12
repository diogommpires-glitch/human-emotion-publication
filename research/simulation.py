"""Numeric fixture and continuous real-network bridge, with no spike raster."""

import copy
import json

import numpy as np


def synthetic_chunk(spec, checkpoint, start_seconds, duration_seconds):
    first = round(start_seconds * spec.sample_hz)
    stop = round((start_seconds + duration_seconds) * spec.sample_hz)
    times = np.arange(first, stop) / spec.sample_hz
    values = np.sin(2 * np.pi * times)[:, None]
    return {"time_seconds": times, "fixture_signal": values}, {"next_sample": np.array(stop)}


class SimulatorAdapter:
    def __init__(self, spec, *, stimulus_drives=None, learning_config=None, randomised_graph=False):
        import torch

        from brain_model.config import DEFAULT_INPUTS, NETWORK

        self.cfg = copy.deepcopy(NETWORK)
        self.cfg["runtime"]["min_cells_per_region"] = min(
            64, max(2, spec.neurons // len(self.cfg["regions"]))
        )
        self.cfg["runtime"]["topology_seed"] = spec.seed
        if randomised_graph:
            rng = np.random.default_rng(spec.seed)
            targets = [edge["to"] for edge in self.cfg["projections"]]
            for edge, target in zip(self.cfg["projections"], rng.permutation(targets), strict=True):
                edge["to"] = str(target)
        self.inputs = dict(DEFAULT_INPUTS)
        self.device = torch.device("cpu")
        self.stimulus_drives = stimulus_drives
        self.learning_config = learning_config or {"rule": "none"}
        self.topology = None

    def __call__(self, spec, checkpoint, start_seconds, duration_seconds):
        import torch

        from brain_model import simulator
        from brain_model.sessions import pack_topology, unpack_topology

        with simulator.RUN_LOCK:
            if checkpoint is None:
                self.topology = simulator.build_topology(self.cfg, spec.neurons, self.device)
                state = None
            else:
                metadata = json.loads(str(checkpoint["metadata"]))
                if metadata["cfg"] != self.cfg:
                    raise ValueError(
                        "research checkpoint configuration differs from requested simulation"
                    )
                topology_tensors = {
                    key.removeprefix("topology."): torch.from_numpy(value.copy())
                    for key, value in checkpoint.items()
                    if key.startswith("topology.")
                }
                self.topology = unpack_topology(metadata["topology"], topology_tensors, self.device)
                tensors = {
                    key.removeprefix("run."): torch.from_numpy(value.copy())
                    for key, value in checkpoint.items()
                    if key.startswith("run.")
                }
                state = simulator.RunState.from_checkpoint(metadata["run"], tensors, self.topology)
                expected_start = state.metadata["step"] * state.metadata["dt_ms"] / 1000
                if abs(expected_start - start_seconds) > 1e-10:
                    raise ValueError("research checkpoint and recorded time disagree")
            raw, state = simulator.run_episode(
                self.cfg,
                self.topology,
                self.inputs,
                spec.seed,
                duration_s=duration_seconds,
                state=state,
                stimulus_drives=self.stimulus_drives,
                learning_config=self.learning_config,
                record=False,
            )
            metadata, tensors = state.to_checkpoint()
            arrays = {
                "time_seconds": np.asarray(raw["timecourse"]["t"]),
                "neural_drive": np.asarray(raw["timecourse"]["neural_drive"]),
                "projection_hz": np.asarray(raw["timecourse"]["projection_hz"]),
                "occupancy": np.asarray(raw["timecourse"]["occupancy"]),
                "field_time_seconds": np.asarray(raw["field_potential"]["t"]),
                "field_potential": np.asarray(list(raw["field_potential"]["channels"].values())).T,
            }
            packed = {"run." + key: value.numpy() for key, value in tensors.items()}
            packed.update(
                {
                    "topology." + key: value.numpy()
                    for key, value in pack_topology(self.topology).items()
                }
            )
            topology_metadata = {
                "network": self.cfg,
                "neurons": self.topology["n"],
                "rows": self.topology["rows"],
            }
            packed["metadata"] = np.array(
                json.dumps(
                    {"cfg": self.cfg, "run": metadata, "topology": topology_metadata},
                    allow_nan=False,
                )
            )
            return arrays, packed


def simulate_stimulus(
    spec, stimulus_drives, *, frozen=True, randomised_graph=False, cancel=lambda: None
):
    adapter = SimulatorAdapter(
        spec,
        stimulus_drives=stimulus_drives,
        learning_config={"rule": "none" if frozen else "pair_stdp"},
        randomised_graph=randomised_graph,
    )
    checkpoint, start, chunks = None, 0.0, []
    while start < spec.duration_seconds - 1e-12:
        cancel()
        duration = min(spec.chunk_seconds, spec.duration_seconds - start)
        values, checkpoint = adapter(spec, checkpoint, start, duration)
        chunks.append(values)
        start += duration
    return {name: np.concatenate([chunk[name] for chunk in chunks], axis=0) for name in chunks[0]}


def rating_predictors(recording):
    rates = np.asarray(recording["projection_hz"])
    drive = np.asarray(recording["neural_drive"])
    occupancy = np.asarray(recording["occupancy"])
    if not len(rates):
        raise ValueError("rating predictors require recorded samples")
    # Seven named aggregate predictors, fixed before observing ratings.
    return np.array(
        [
            rates.mean(),
            rates.std(),
            rates.max(),
            drive.mean(),
            drive.std(),
            occupancy.mean(),
            occupancy.std(),
        ]
    )


RATING_PREDICTORS = (
    "rate_mean_hz",
    "rate_sd_hz",
    "rate_peak_hz",
    "drive_mean",
    "drive_sd",
    "occupancy_mean",
    "occupancy_sd",
)


def session_episode(record, topology, cfg, seed, weights, learn, chunk_seconds, cancel):
    from brain_model import simulator
    from brain_model.config import DEFAULT_INPUTS

    duration = float(record["duration_seconds"])
    levels = dict(DEFAULT_INPUTS)
    if set(record["inputs"]) - levels.keys():
        raise ValueError("session-order inputs contain unknown channels")
    levels.update(record["inputs"])
    if not np.isfinite(duration) or not cfg["runtime"]["time_step_ms"] / 1000 <= duration <= 86400:
        raise ValueError("session-order duration must fit [one integration step, 86400 seconds]")
    if any(
        not isinstance(value, (float, int)) or not np.isfinite(value) or not 0 <= value <= 1
        for value in levels.values()
    ):
        raise ValueError("session-order input levels must be finite and in [0,1]")
    drives = [levels] * int(np.ceil(duration))
    state, elapsed, raw = None, 0.0, None
    while elapsed < duration - 1e-12:
        cancel()
        chunk = min(chunk_seconds, duration - elapsed)
        with simulator.RUN_LOCK:
            raw, state = simulator.run_episode(
                cfg,
                topology,
                levels,
                seed,
                duration_s=chunk,
                state=state,
                stimulus_drives=drives,
                weights=weights,
                learning_config={"rule": "pair_stdp" if learn else "none"},
                record=False,
            )
        elapsed += chunk
    assert raw is not None and state is not None
    return raw, state


def session_order_control(order_name, episodes, probe, topology, cfg, seed, chunk_seconds, cancel):
    base = {name: contacts["weight"].clone() for name, contacts in topology["contacts"].items()}
    indices = np.arange(len(episodes))
    if order_name == "shuffled":
        indices = np.random.default_rng(seed).permutation(indices)
    weights = {name: value.clone() for name, value in base.items()}
    for index in indices:
        cancel()
        if order_name == "independent":
            weights = {name: value.clone() for name, value in base.items()}
        _, state = session_episode(
            episodes[index], topology, cfg, seed, weights, True, chunk_seconds, cancel
        )
        weights = {name: state.tensors["learning.weights." + name].clone() for name in base}
    raw, _ = session_episode(probe, topology, cfg, seed, weights, False, chunk_seconds, cancel)
    reference, _ = session_episode(probe, topology, cfg, seed, base, False, chunk_seconds, cancel)
    return {
        "seed": seed,
        "order": order_name,
        "episode_order": indices.tolist(),
        "probe_spike_count": raw["spike_count"],
        "frozen_probe_spike_count": reference["spike_count"],
        "weight_l1_change": float(
            sum((weights[name] - base[name]).abs().sum().item() for name in base)
        ),
    }


def evaluate_session_order(spec, cancel=lambda: None):
    from dataclasses import replace
    from pathlib import Path

    import torch

    from brain_model import simulator

    from .spec import SEEDS

    source = json.loads(Path(spec.source_path).read_text(encoding="utf-8"))
    episodes, probe = source["episodes"], source["probe"]
    if not 2 <= len(episodes) <= 1000:
        raise ValueError("session-order control requires 2..1000 prespecified episodes")
    results = []
    for seed in SEEDS:
        adapter = SimulatorAdapter(replace(spec, seed=seed))
        with simulator.RUN_LOCK:
            topology = simulator.build_topology(adapter.cfg, spec.neurons, torch.device("cpu"))
        results.extend(
            session_order_control(
                order, episodes, probe, topology, adapter.cfg, seed, spec.chunk_seconds, cancel
            )
            for order in ("fixed", "shuffled", "independent")
        )
    return {
        "protocol": "session_order",
        "empirical_validation": False,
        "results": results,
        "interpretation": "Mechanistic fixed/shuffled/independent controls; no participant validation.",
    }
