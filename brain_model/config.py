# HTTP and checkpoint validation use ValueError consistently for invalid public data.
# ruff: noqa: TRY004
"""Network constants and validated public numerical configuration."""

import hashlib
import json
import math
import pathlib

NETWORK_PATH = pathlib.Path(__file__).with_name("network.json")
NETWORK = json.loads(NETWORK_PATH.read_text(encoding="utf-8"))
REGIONS = {region["id"]: region for region in NETWORK["regions"]}
INPUTS = {channel["id"]: channel for channel in NETWORK["inputs"]}
DEFAULT_INPUTS = {key: channel["default"] for key, channel in INPUTS.items()}
MOOD_WEIGHT = 0.5
REFERENCE_SEED = 20260926


def finite_number(value, name: str, minimum=None, maximum=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must lie in [{minimum}, {maximum}], got {value}.")
    return float(value)


def _finite_tree(value, path):
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        finite_number(value, path)


def _validate_runtime(cfg: dict) -> None:
    rt = cfg["runtime"]
    dt = finite_number(rt["time_step_ms"], "time_step_ms", 1e-6)
    finite_number(rt["duration_seconds"], "duration_seconds", dt / 1000)
    if dt > rt["rate_smoothing_ms"] or dt != cfg["synapses"]["local_delay_ms"]:
        raise ValueError("time_step_ms must equal local_delay_ms and not exceed rate_smoothing_ms.")
    for name in ("baseline_window_seconds", "stimulus_seconds"):
        lo, hi = rt[name]
        if not 0 <= lo < hi <= rt["duration_seconds"]:
            raise ValueError(f"{name} must be ordered inside the episode duration.")
    if rt["baseline_window_seconds"][1] > rt["stimulus_seconds"][0]:
        raise ValueError("Baseline must end before the stimulus.")
    finite_number(
        cfg["neuromodulation"]["receptor_half_occupancy"], "receptor_half_occupancy", 1e-12
    )


def _validate_cells(cfg: dict) -> None:
    for row in cfg["regions"]:
        finite_number(row["relative_size"], "relative_size", 1e-12)
        finite_number(row["projection_fraction"], "projection_fraction", 0, 1)
        finite_number(row.get("vip_fraction", 0), "vip_fraction", 0, 1)
        if row["projection_cell_type"] not in cfg["cell_types"]:
            raise ValueError("Unknown projection cell type.")
    for cell in cfg["cell_types"].values():
        for name in ("tau_m_ms", "tau_w_ms", "recovery_ms", "delta_t_mV", "release_u"):
            finite_number(cell[name], name, 1e-12)
        finite_number(cell["release_u"], "release_u", 1e-12, 1)
        finite_number(cell["refractory_ms"], "refractory_ms", 0)
        finite_number(cell["facilitation_ms"], "facilitation_ms", 0)


def _validate_synapse_times(cfg: dict) -> None:
    for name in ("ampa_decay_ms", "nmda_decay_ms", "gaba_a_decay_ms", "gaba_b_decay_ms"):
        finite_number(cfg["synapses"][name], name, 1e-12)
    finite_number(
        cfg["synapses"]["long_range_delay_ms"],
        "long_range_delay_ms",
        cfg["runtime"]["time_step_ms"],
    )


def _validate_projections(cfg: dict, ids: set[str]) -> None:
    for edge in cfg["projections"]:
        if edge["from"] not in ids or edge["to"] not in ids:
            raise ValueError("Projection refers to an unknown region.")
        if not isinstance(edge.get("plastic_synapse", False), bool):
            raise ValueError("plastic_synapse must be boolean.")


def _validate_inputs(cfg: dict, ids: set[str]) -> None:
    for channel in cfg["inputs"]:
        finite_number(channel["assembly_fraction"], "assembly_fraction", 0, 1)
        finite_number(channel["max_rate_hz"], "max_rate_hz", 0)
        if set(channel["targets"]) - set(ids):
            raise ValueError("Input refers to an unknown region.")


def _validate_modulators(cfg: dict, ids: set[str]) -> None:
    for mod in cfg["neuromodulators"]:
        if (
            mod["source"] not in ids
            or set(mod["targets"]) - set(ids)
            or set(mod["effects"]) - set(ids)
        ):
            raise ValueError("Neuromodulator refers to an unknown region.")
        finite_number(mod["resting_source_hz"], "resting_source_hz", 1e-12)
        finite_number(mod["clearance_ms"], "clearance_ms", 1e-12)


def validate_network(cfg: dict) -> None:
    """Validate numerical units, anatomy and references before allocating the graph."""
    _finite_tree(cfg, "network")
    ids = {row["id"] for row in cfg["regions"]}
    if len(ids) != len(cfg["regions"]):
        raise ValueError("Region ids must be unique.")
    _validate_runtime(cfg)
    _validate_cells(cfg)
    _validate_synapse_times(cfg)
    _validate_projections(cfg, ids)
    _validate_inputs(cfg, ids)
    _validate_modulators(cfg, ids)


def configuration_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, allow_nan=False).encode()).hexdigest()


RULE_VERSION = 1
DEFAULTS = {
    "A_plus": 0.001,
    "A_minus": 0.0012,
    "tau_plus_ms": 20.0,
    "tau_minus_ms": 20.0,
    "g": 1.0,
    "trace_tau_s": 1.0,
    "eta": 0.0001,
    "da_threshold": 0.05,
}


LEGACY_PARAMETERS = {
    "stdp": {
        "A_plus": "A_plus",
        "A_minus": "A_minus",
        "tau_plus": "tau_plus_ms",
        "tau_minus": "tau_minus_ms",
    },
    "td_learning": {
        "trace_tau_s": "trace_tau_s",
        "learning_rate_baseline": "eta",
        "da_threshold": "da_threshold",
    },
}


def _legacy_learning(name: str, values: dict) -> dict | None:
    if not isinstance(values, dict) or not isinstance(values.get("enabled", True), bool):
        raise ValueError(f"{name} must be an object with boolean enabled.")
    mapping = LEGACY_PARAMETERS[name]
    unsupported = set(values) - {"enabled"} - mapping.keys()
    if unsupported:
        raise ValueError(
            f"Migration required: {name} parameters {sorted(unsupported)} have no coherent rule-v1 meaning; use learning."
        )
    if not values.get("enabled", True):
        return None
    rule = "pair_stdp" if name == "stdp" else "dopamine_eligibility"
    return {"rule": rule, **{mapping[k]: v for k, v in values.items() if k != "enabled"}}


def _learning_parameters(values: dict) -> dict:
    if not isinstance(values, dict):
        raise ValueError("learning must be an object.")
    if set(values) - {"rule", "version", *DEFAULTS}:
        raise ValueError("Unknown learning parameter.")
    rule = values.get("rule", "none")
    if (
        rule not in ("none", "pair_stdp", "dopamine_eligibility")
        or values.get("version", 1) != RULE_VERSION
    ):
        raise ValueError("Unsupported learning rule or version.")
    result = {"rule": rule, "version": RULE_VERSION, **DEFAULTS, **values}
    for key in DEFAULTS:
        value = finite_number(result[key], f"learning.{key}", 0)
        if key in ("tau_plus_ms", "tau_minus_ms", "trace_tau_s") and value == 0:
            raise ValueError(f"learning.{key} must be positive.")
    if result["g"] > 1 or result["da_threshold"] > 1:
        raise ValueError("g and da_threshold must be in [0,1].")
    return result


def parse_learning(payload: dict) -> dict:
    modern = payload.get("learning")
    legacy = [(key, payload[key]) for key in LEGACY_PARAMETERS if payload.get(key) is not None]
    if modern is not None and legacy:
        raise ValueError("Use learning only; do not combine with deprecated stdp/td_learning.")
    active = [
        adapted
        for name, values in legacy
        if (adapted := _legacy_learning(name, values)) is not None
    ]
    if len(active) > 1:
        raise ValueError("Learning rules are mutually exclusive.")
    if modern is not None:
        return _learning_parameters(modern)
    return _learning_parameters(active[0] if active else {"rule": "none"})
