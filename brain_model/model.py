# HTTP and checkpoint validation use ValueError consistently for invalid public data.
# ruff: noqa: TRY004
"""Request handling, emotion readouts and metadata for Human Emotion.

The spiking network lives in simulator.py. This module validates a request, runs
the network and turns recorded activity into readouts. Every readout compares the
stimulus period with the same run's pre-stimulus baseline (the emotional response)
and adds half of the baseline's difference from a standard resting reference (the
mood set by the body and mind sliders). The mapping from activity to named
emotions is a hand-designed summary of model variables, not a validated classifier.
"""

import json
import math
import secrets

from .config import (
    DEFAULT_INPUTS,
    NETWORK,
    REFERENCE_SEED,
    configuration_hash,
    parse_learning,
)

try:
    import torch

    from . import simulator

    IMPORT_ERROR = None
except ImportError as exc:  # keep /api/info usable so the page can explain the problem
    torch = simulator = None
    IMPORT_ERROR = str(exc)

_REFERENCE = {}


class SimulationError(RuntimeError):
    """A failure the user can act on (memory, missing PyTorch); shown as-is."""


# --------------------------------------------------------------------------- requests


def _unit(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a number between 0 and 1.")
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1.")
    return float(value)


def _device_sizes(device, cfg=None):
    rt = (NETWORK if cfg is None else cfg)["runtime"]
    if device.type == "cuda":
        return rt["gpu_network_sizes"], rt["gpu_default_cells"]
    return rt["cpu_network_sizes"], rt["cpu_default_cells"]


def _apply_person(inputs, person):
    """A familiar person acts as a learned social cue: warm memories route to the oxytocin
    and reward system, threatening ones to the amygdala; familiarity lowers novelty."""
    closeness = person["closeness"]
    tone = 2 * person["tone"] - 1
    warm = closeness * max(0.0, tone) * (0.5 + 0.5 * person["trust"])
    cold = closeness * max(0.0, -tone) * (1.0 - 0.5 * person["trust"])
    inputs["social_touch"] = min(1.0, inputs["social_touch"] + 0.6 * warm)
    inputs["threat_cue"] = min(1.0, inputs["threat_cue"] + 0.6 * cold)
    inputs["novelty"] *= 1 - 0.5 * closeness


def _request_inputs(payload, cfg):
    input_channels = {row["id"]: row for row in cfg["inputs"]}
    default_inputs = {key: row["default"] for key, row in input_channels.items()}
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object.")
    raw = payload.get("inputs", {})
    if not isinstance(raw, dict):
        raise ValueError("inputs must be a JSON object.")
    unknown = sorted(set(raw) - set(input_channels))
    if unknown:
        raise ValueError(f"Unknown input: {', '.join(unknown)}.")
    return {key: _unit(raw.get(key, default), key) for key, default in default_inputs.items()}


def parse_request(payload, device, *, cfg=None):
    cfg = NETWORK if cfg is None else cfg
    inputs = _request_inputs(payload, cfg)

    person = payload.get("person")
    person_name = ""
    if person:
        if not isinstance(person, dict):
            raise ValueError("person must be a JSON object.")
        person_name = str(person.get("name", ""))[:120]
        _apply_person(
            inputs,
            {
                key: _unit(person.get(key, 0.5), f"person.{key}")
                for key in ("trust", "closeness", "tone")
            },
        )

    sizes, default = _device_sizes(device, cfg)
    if isinstance(payload.get("neurons", default), bool) or not isinstance(
        payload.get("neurons", default), int
    ):
        raise ValueError("neurons must be a whole number.")
    neurons = payload.get("neurons", default)
    if neurons not in sizes:
        raise ValueError(
            f"Choose one of these network sizes: {', '.join(f'{size:,}' for size in sizes)}."
        )

    seed = payload.get("seed")
    if seed is None or seed == "":
        seed = secrets.randbits(31)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be a whole number.")

    if not 0 <= seed < 2**31:
        raise ValueError("seed must be between 0 and 2147483647.")

    learning_config = parse_learning(payload)

    return {
        "inputs": inputs,
        "neurons": neurons,
        "seed": seed,
        "person": person_name,
        "event": str(payload.get("event", ""))[:500],
        "learning_config": learning_config,
    }


# --------------------------------------------------------------------------- running


def _fit_to_memory(neurons, device):
    """On CUDA, step down to the largest size whose estimate fits in free VRAM."""
    if simulator is None or torch is None:
        raise SimulationError("PyTorch is required to allocate a network.")
    if device.type != "cuda":
        return neurons
    sizes = sorted(size for size in _device_sizes(device)[0] if size <= neurons)
    cached = simulator._TOPOLOGY_CACHE.get((neurons, str(device), configuration_hash(NETWORK)))
    if cached is not None:
        return neurons
    simulator._TOPOLOGY_CACHE.clear()
    torch.cuda.empty_cache()
    free = torch.cuda.mem_get_info(device)[0]
    for size in reversed(sizes):
        if simulator.estimated_bytes(NETWORK, size) <= 0.9 * free:
            return size
    raise SimulationError(
        "The GPU has too little free memory for the smallest network. Close GPU-heavy apps and try again."
    )


def _reference(topology):
    if simulator is None:
        raise SimulationError("PyTorch is required to run the reference.")
    key = (topology["n"], str(topology["device"]), topology["config_hash"])
    if key not in _REFERENCE:
        duration = NETWORK["runtime"]["reference_duration_seconds"]
        run = simulator.run(
            NETWORK, topology, DEFAULT_INPUTS, REFERENCE_SEED, duration_s=duration, record=False
        )
        _REFERENCE.clear()
        _REFERENCE[key] = run["baseline"]
    return _REFERENCE[key]


def simulate(payload):
    if simulator is None or torch is None:
        raise SimulationError(f"PyTorch is required to run the network ({IMPORT_ERROR}).")
    device = simulator.pick_device()
    request = parse_request(payload, device)
    with simulator.RUN_LOCK:
        try:
            neurons = _fit_to_memory(request["neurons"], device)
            topology = simulator.build_topology(NETWORK, neurons, device)
            reference = _reference(topology)
            raw = simulator.run(
                NETWORK,
                topology,
                request["inputs"],
                request["seed"],
                learning_config=request["learning_config"],
            )
        except torch.cuda.OutOfMemoryError:
            simulator._TOPOLOGY_CACHE.clear()
            torch.cuda.empty_cache()
            raise SimulationError(
                "The GPU ran out of memory. Choose a smaller network or close GPU-heavy apps."
            ) from None
        response = build_response(request, raw, reference, topology, cfg=NETWORK)
        json.dumps(response, allow_nan=False)
        return response


# --------------------------------------------------------------------------- readouts

from .readouts import build_response


def model_info():
    """Describe the network and the local compute backend."""
    info = {
        "name": NETWORK["name"],
        "status": NETWORK["status"],
        "description": NETWORK["description"],
        "regions": [
            {key: region[key] for key in ("id", "label", "role", "transmitter")}
            for region in NETWORK["regions"]
        ],
        "projections": len(NETWORK["projections"]),
        "neuromodulators": [mod["name"] for mod in NETWORK["neuromodulators"]],
        "inputs": [
            {key: channel[key] for key in ("id", "label", "group", "timing", "default")}
            for channel in NETWORK["inputs"]
        ],
        "timeline": {
            "duration_seconds": NETWORK["runtime"]["duration_seconds"],
            "stimulus_seconds": NETWORK["runtime"]["stimulus_seconds"],
        },
        "time_step_ms": NETWORK["runtime"]["time_step_ms"],
        "learning": {
            "rules": ["none", "pair_stdp", "dopamine_eligibility"],
            "defaults": parse_learning({}),
            "weight_bounds": [0.1, 2.0],
            "description": "Contact-level pair STDP or experimental occupancy-gated eligibility; no reward prediction or value function.",
        },
    }

    if simulator is None:
        return {
            **info,
            "available": False,
            "backend": "unavailable",
            "hardware": {"available": False, "name": f"PyTorch not installed ({IMPORT_ERROR})"},
            "network_sizes": [],
            "default_neurons": None,
        }
    device = simulator.pick_device()
    sizes, default = _device_sizes(device)
    hardware = simulator.device_info(device)
    return {
        **info,
        "available": True,
        "backend": hardware["backend"],
        "hardware": hardware,
        "network_sizes": sizes,
        "default_neurons": default,
    }
