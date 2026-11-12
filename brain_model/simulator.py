"""Spiking network with delayed synapses and projected neuromodulators.

The same code runs on an NVIDIA GPU (CUDA) or on the CPU. Everything anatomical
or biophysical comes from network.json; this module only implements dynamics:

* adaptive exponential integrate-and-fire cells (subthreshold and spike-triggered
  adaptation, 2 ms refractory period, per-cell threshold jitter);
* AMPA, voltage-dependent NMDA, GABA-A and GABA-B conductances, with Tsodyks-Markram
  short-term depression/facilitation at each presynaptic cell and a 1 ms local
  versus multi-millisecond long-range conduction delay;
* Poisson sensory, bodily and background input;
* dopamine, noradrenaline, serotonin, acetylcholine and oxytocin released by their
  source nuclei in proportion to firing, carried to target regions, cleared with
  their own time constants and acting through receptor occupancy.

It is a reduced research model, not a human brain replica.
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass, field

import torch

from .config import configuration_hash, finite_number
from .learning import ContactLearner, parse_learning
from .topology import (
    _TOPOLOGY_CACHE,
    _synapse_parts,
    build_topology,
    contact_matrix,
    estimated_bytes,
    layout,
    planned_synapses,
    region_sizes,
)

__all__ = [
    "_TOPOLOGY_CACHE",
    "RunState",
    "_synapse_parts",
    "build_topology",
    "estimated_bytes",
    "layout",
    "planned_synapses",
    "region_sizes",
    "run",
    "run_episode",
]

RUN_LOCK = threading.Lock()
LFP_AMPA_DELAY_MS = 6  # Mazzoni et al. (2015) LFP proxy: |I_AMPA|(t-6 ms) - 1.65 |I_GABA|(t)
LFP_GABA_WEIGHT = 1.65


def pick_device():
    if os.environ.get("HUMAN_EMOTION_DEVICE", "").lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def device_info(device):
    if device.type != "cuda":
        return {
            "available": False,
            "backend": "CPU",
            "name": f"CPU ({torch.get_num_threads()} threads)",
        }
    props = torch.cuda.get_device_properties(device)
    return {
        "available": True,
        "backend": "CUDA",
        "name": props.name,
        "vram_gib": round(props.total_memory / 2**30, 2),
        "free_vram_gib": round(torch.cuda.mem_get_info(device)[0] / 2**30, 2),
        "cuda": torch.version.cuda,
    }


def _spmv(matrix, vector):
    if matrix is None:
        return torch.zeros_like(vector)
    return torch.sparse.mm(matrix, vector.unsqueeze(1)).squeeze(1)


def _external_rates(cfg, topology, inputs, generator):
    """Expected Poisson events per second for each cell, without and with the stimulus."""
    device, rows = topology["device"], topology["rows"]
    background = cfg["background_input"]
    # Interneurons receive a smaller share of external drive than projection cells.
    share = torch.where(topology["interneuron"], background.get("interneuron_scale", 1.0), 1.0).to(
        device
    )
    base = float(background["rate_hz"]) * share
    for region, row in zip(cfg["regions"], rows, strict=False):
        base[row["start"] : row["start"] + row["size"]] *= region.get("background_scale", 1.0)
    stim = base.clone()
    by_id = {row["id"]: row for row in rows}
    for channel in cfg["inputs"]:
        level = float(inputs.get(channel["id"], channel["default"]))
        for region_id, weight in channel["targets"].items():
            row = by_id[region_id]
            # Draw every assembly even at zero level so assemblies never depend on other inputs.
            member = topology["assemblies"][channel["id"] + ":" + region_id]
            cells = slice(row["start"], row["start"] + row["size"])
            extra = member * share[cells] * (level * channel["max_rate_hz"] * weight)
            stim[cells] += extra
            if channel["timing"] == "tonic":
                base[cells] += extra
    return base, stim


def _tonic_current(cfg, inputs, index, device):
    nm = cfg["neuromodulation"]
    defaults = {channel["id"]: channel["default"] for channel in cfg["inputs"]}
    tonic = torch.zeros(len(index), device=device)
    for factor in cfg.get("tonic_factors", []):
        change = (
            float(inputs.get(factor["input"], defaults[factor["input"]]))
            - defaults[factor["input"]]
        )
        for region_id, effect in factor["effects"].items():
            tonic[index[region_id]] += nm["excitability_mV_per_occupancy"] * effect * change
    return tonic


def _modulator_tables(cfg, topology, inputs):
    device, rows = topology["device"], topology["rows"]
    index = {row["id"]: i for i, row in enumerate(rows)}
    mods = cfg["neuromodulators"]
    shape = (len(rows), len(mods))
    density, excitability, adaptation, nmda = (torch.zeros(shape, device=device) for _ in range(4))
    for m, mod in enumerate(mods):
        for region_id, value in mod["targets"].items():
            density[index[region_id], m] = value
        for region_id, effects in mod["effects"].items():
            excitability[index[region_id], m] = effects.get("excitability", 0.0)
            adaptation[index[region_id], m] = effects.get("adaptation", 0.0)
            nmda[index[region_id], m] = effects.get("nmda", 0.0)
    cfg["neuromodulation"]
    tonic = _tonic_current(cfg, inputs, index, device)
    mod_names = [mod["name"] for mod in mods]
    mod_idx = {name: i for i, name in enumerate(mod_names)}
    return {
        "density": density,
        "excitability": excitability,
        "adaptation": adaptation,
        "nmda": nmda,
        "source": torch.tensor([index[mod["source"]] for mod in mods], device=device),
        "resting_hz": torch.tensor([mod["resting_source_hz"] for mod in mods], device=device),
        "clearance_ms": torch.tensor(
            [mod["clearance_ms"] for mod in mods], device=device, dtype=torch.float32
        ),
        "tonic_mV": tonic,
        "names": mod_names,
        "index": mod_idx,
    }


@dataclass
class RunState:
    """Continuous neural, chemical, delay, plasticity and RNG state."""

    tensors: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_checkpoint(self):
        return dict(self.metadata), {
            key: value.detach().cpu().contiguous() for key, value in self.tensors.items()
        }

    @classmethod
    def from_checkpoint(cls, metadata, tensors, topology):
        if metadata.get("version") != 1 or metadata.get("n") != topology["n"]:
            raise ValueError("Continuous state version or neuron count does not match topology.")
        if (
            metadata.get("config_hash") != topology["config_hash"]
            or metadata.get("dt_ms") != topology["dynamics"]["dt_ms"]
        ):
            raise ValueError("Continuous checkpoint configuration does not match topology.")
        n, regions = topology["n"], topology["region_count"]
        dynamics = topology["dynamics"]
        shape_by_key: dict[str, tuple[tuple[int, ...], torch.dtype]] = dict.fromkeys(
            (
                "v",
                "w",
                "g_ampa",
                "g_nmda",
                "g_gaba_a",
                "g_gaba_b",
                "u",
                "x",
                "release",
                "learning.pre",
                "learning.post",
            ),
            ((n,), torch.float32),
        )
        shape_by_key.update(
            dict.fromkeys(("da_sum", "da_baseline", "pending_drive"), ((regions,), torch.float32))
        )
        shape_by_key.update(
            dict.fromkeys(("rates", "pending_counts"), ((regions * 3,), torch.float32))
        )
        shape_by_key["chem"] = ((regions, dynamics["modulators"]), torch.float32)
        shape_by_key["refractory"] = ((n,), torch.int16)
        shape_by_key["spike_total"] = ((), torch.int64)
        shape_by_key["lfp_tail"] = (
            (max(1, round(LFP_AMPA_DELAY_MS / dynamics["dt_ms"])), 2, dynamics["channels"]),
            torch.float32,
        )
        shape_by_key["rng"] = (
            tuple(torch.Generator(device=topology["device"]).get_state().shape),
            torch.uint8,
        )
        if dynamics["delay"] > 1:
            shape_by_key["ring"] = ((dynamics["delay"], n), torch.float32)
        for window in ("baseline", "response"):
            shape_by_key[f"window.{window}.counts"] = ((regions * 3,), torch.float32)
            shape_by_key[f"window.{window}.occ"] = (
                (regions, dynamics["modulators"]),
                torch.float32,
            )
            shape_by_key[f"window.{window}.release"] = ((2,), torch.float32)
        for group, contacts in topology["contacts"].items():
            for field_name in ("weights", "eligibility"):
                shape_by_key[f"learning.{field_name}.{group}"] = (
                    tuple(contacts["weight"].shape),
                    torch.float32,
                )
        if set(tensors) != set(shape_by_key) or any(
            tuple(tensors[key].shape) != shape or tensors[key].dtype != dtype
            for key, (shape, dtype) in shape_by_key.items()
        ):
            raise ValueError("Continuous checkpoint tensor shape or dtype mismatch.")
        if any(not torch.isfinite(value).all() for value in tensors.values()):
            raise ValueError("Continuous state contains non-finite values.")
        return cls(
            {key: value.to(topology["device"]) for key, value in tensors.items()}, dict(metadata)
        )


def run(
    cfg,
    topology,
    inputs,
    seed,
    duration_s=None,
    record=True,
    stdp_learner=None,
    stdp_config=None,
    td_learner=None,
    td_config=None,
    learning_config=None,
    weights=None,
):
    """Independent episode returning JSON-compatible data; cached base weights are immutable."""
    if stdp_learner is not None or td_learner is not None:
        raise ValueError("Pass learning_config instead of deprecated learner instances.")
    if learning_config is not None and (stdp_config is not None or td_config is not None):
        raise ValueError("Learning rules are mutually exclusive.")
    config = learning_config or parse_learning({"stdp": stdp_config, "td_learning": td_config})
    result, _ = run_episode(
        cfg,
        topology,
        inputs,
        seed,
        duration_s=duration_s,
        record=record,
        learning_config=config,
        weights=weights,
    )
    return result


def run_episode(
    cfg,
    topology,
    inputs,
    seed,
    *,
    duration_s=None,
    state=None,
    stimulus_drives=None,
    learning_config=None,
    weights=None,
    record=False,
):
    """Advance a continuous episode; per-second drive rows use absolute simulation time."""
    device, n = topology["device"], topology["n"]
    if configuration_hash(cfg) != topology["config_hash"]:
        raise ValueError("Episode configuration does not match the realised topology.")
    known_inputs = {channel["id"] for channel in cfg["inputs"]}
    if not isinstance(inputs, dict) or set(inputs) - known_inputs:
        raise ValueError("Episode inputs must be a mapping of known channel ids.")
    for key, value in inputs.items():
        finite_number(value, key, 0, 1)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError("Episode seed must be an integer in [0,2147483647].")
    rt, syn, nm = cfg["runtime"], cfg["synapses"], cfg["neuromodulation"]
    dt_ms = rt["time_step_ms"]
    dt = dt_ms / 1000
    duration_s = rt["duration_seconds"] if duration_s is None else duration_s
    finite_number(duration_s, "duration_s", dt)
    steps = round(duration_s / dt)
    if not math.isclose(steps * dt, duration_s, abs_tol=1e-10):
        raise ValueError("duration_s must be a whole number of integration steps.")
    offset = 0 if state is None else state.metadata["step"]
    learning_config = (
        parse_learning({"learning": learning_config})
        if learning_config is not None
        else parse_learning({})
    )
    if state is not None and (
        state.metadata["inputs"] != inputs
        or state.metadata["learning"] != learning_config
        or state.metadata["config_hash"] != topology["config_hash"]
        or state.metadata["seed"] != seed
    ):
        raise ValueError("Continuous chunks must preserve inputs and learning configuration.")
    step_of = lambda seconds: round(seconds / dt)
    base_from, base_to = map(step_of, rt["baseline_window_seconds"])
    stim_on, stim_off = map(step_of, rt["stimulus_seconds"])
    resp_from = stim_on + step_of(0.1)  # skip the first 100 ms of transmission
    record_every = max(1, round(rt["record_every_ms"] / dt_ms))
    generator = torch.Generator(device=device).manual_seed(int(seed))
    assembly_generator = torch.Generator(device=device).manual_seed(rt["topology_seed"] + 17)
    started = time.perf_counter()

    learner = ContactLearner(topology, learning_config, dt_ms, weights)
    if state is not None:
        learner.restore(
            {
                key.removeprefix("learning."): value
                for key, value in state.tensors.items()
                if key.startswith("learning.")
            }
        )
    mats = {
        name: contact_matrix(c, n, learner.weights[name])[0]
        for name, c in topology["contacts"].items()
    }

    p = topology["params"]
    rows, R = topology["rows"], topology["region_count"]
    region_of_cell, bin_of_cell = topology["region_of_cell"], topology["bin_of_cell"]
    gaba_f = topology["gaba_cell"].to(torch.float32)
    glu_f = 1.0 - gaba_f

    # Constants
    decay = lambda tau_ms: math.exp(-dt_ms / tau_ms)
    d_ampa, d_nmda = decay(syn["ampa_decay_ms"]), decay(syn["nmda_decay_ms"])
    d_gaba_a, d_gaba_b = decay(syn["gaba_a_decay_ms"]), decay(syn["gaba_b_decay_ms"])
    e_glu, e_gaba_a, e_gaba_b = (
        syn["glutamate_reversal_mV"],
        syn["gaba_a_reversal_mV"],
        syn["gaba_b_reversal_mV"],
    )
    mg_factor = syn["magnesium_mM"] / 3.57
    delay = max(1, round(syn["long_range_delay_ms"] / dt_ms))
    dt_over_tau = dt_ms / p["tau_m_ms"]
    w_rate = dt_ms / p["tau_w_ms"]
    ref_steps = torch.round(p["refractory_ms"] / dt_ms).to(torch.int16)
    rec_rate = 1 - torch.exp(-dt_ms / p["recovery_ms"])
    facilitating = (p["facilitation_ms"] > 0).to(torch.float32)
    facil_decay = torch.exp(-dt_ms / p["facilitation_ms"].clamp(min=1e-3)) * facilitating
    release_u = p["release_u"]
    w_external = cfg["background_input"]["weight"]
    rate_alpha = dt_ms / rt["rate_smoothing_ms"]

    lam_base, lam_stim = _external_rates(cfg, topology, inputs, assembly_generator)
    lam_base, lam_stim = lam_base * dt, lam_stim * dt
    mod = _modulator_tables(cfg, topology, inputs)
    K = nm["receptor_half_occupancy"]
    density = mod["density"]
    chem_decay = torch.exp(-dt_ms / mod["clearance_ms"])[None, :]
    occ_rest = density / (density + K)
    chem = density.clone()  # resting concentration in the same units as K
    exc_scale, gain = nm["excitability_mV_per_occupancy"], nm["gain_per_occupancy"]

    # State
    v = p["v_rest_mV"] + 4 * (torch.rand(n, device=device, generator=generator) - 0.5)
    w = torch.zeros(n, device=device)
    g_ampa, g_nmda, g_gaba_a, g_gaba_b = (torch.zeros(n, device=device) for _ in range(4))
    refractory = torch.zeros(n, device=device, dtype=torch.int16)
    u = release_u.clone()
    x = torch.ones(n, device=device)
    release = torch.zeros(n, device=device)
    ring = torch.zeros((delay, n), device=device) if delay > 1 else None
    # 3 cell classes per region: projection, SST/PV, VIP
    bin_sizes = torch.tensor(
        [[row["projection"], row["sst_count"], row["vip_count"]] for row in rows],
        device=device,
        dtype=torch.float32,
    ).reshape(-1)
    rates = torch.zeros(R * 3, device=device)
    rates.view(R, 3)[mod["source"], 0] = mod["resting_hz"]
    counts = torch.zeros(R * 3, device=device)

    # Recording buffers (all on the device; copied once at the end)
    n_records = (steps + offset % record_every) // record_every
    rec_counts = torch.zeros((n_records, R * 3), device=device)
    rec_occ = torch.zeros((n_records, R, len(mod["names"])), device=device)
    rec_drive = torch.zeros((n_records, R), device=device)
    pending_counts = torch.zeros(R * 3, device=device)
    pending_drive = torch.zeros(R, device=device)
    drive_counts = torch.zeros(R, device=device)
    da_sum = torch.zeros(R, device=device)
    da_steps = 0
    da_baseline = occ_rest[:, mod["index"]["dopamine"]].clone()
    windows = {
        name: {
            "counts": torch.zeros(R * 3, device=device),
            "occ": torch.zeros_like(density),
            "release": torch.zeros(2, device=device),
            "steps": 0,
        }
        for name in ("baseline", "response")
    }
    release_masks = torch.stack((glu_f, gaba_f), dim=1)
    spike_total = torch.zeros((), device=device, dtype=torch.int64)
    if record:
        sample_ids, sample_layout = _raster_sample(cfg, rows, device)
        raster = torch.zeros((steps, sample_ids.numel()), device=device, dtype=torch.bool)
    lfp_weights = _lfp_weights(cfg, topology)
    lfp_hist = torch.zeros((steps, 2, lfp_weights.shape[1]), device=device)
    shift = max(1, round(LFP_AMPA_DELAY_MS / dt_ms))
    lfp_tail = torch.zeros((shift, 2, lfp_weights.shape[1]), device=device)

    if state is not None:
        saved = state.tensors
        v, w, g_ampa, g_nmda, g_gaba_a, g_gaba_b, refractory, u, x, release, rates, chem = (
            saved[key].clone()
            for key in (
                "v",
                "w",
                "g_ampa",
                "g_nmda",
                "g_gaba_a",
                "g_gaba_b",
                "refractory",
                "u",
                "x",
                "release",
                "rates",
                "chem",
            )
        )
        ring = saved["ring"].clone() if delay > 1 else None
        spike_total = saved["spike_total"].clone()
        da_sum, da_baseline = saved["da_sum"].clone(), saved["da_baseline"].clone()
        da_steps = state.metadata["da_steps"]
        lfp_tail = saved["lfp_tail"]
        pending_counts = saved["pending_counts"].clone()
        pending_drive = saved["pending_drive"].clone()
        generator.set_state(saved["rng"].cpu())
        for name, window in windows.items():
            for key in ("counts", "occ", "release"):
                window[key] = saved[f"window.{name}.{key}"].clone()
            window["steps"] = state.metadata[f"window.{name}.steps"]
    drive_second = None
    for local_step in range(steps):
        step = offset + local_step
        stimulus_on = stim_on <= step < stim_off
        g_ampa.mul_(d_ampa)
        g_nmda.mul_(d_nmda)
        g_gaba_a.mul_(d_gaba_a)
        g_gaba_b.mul_(d_gaba_b)
        if stimulus_drives is not None:
            second = int(step * dt)
            if second != drive_second:
                if second >= len(stimulus_drives):
                    raise ValueError("stimulus_drives does not cover the absolute episode time.")
                levels = dict(inputs)
                for key, value in stimulus_drives[second].items():
                    if key not in levels:
                        raise ValueError(f"Unknown stimulus channel {key}.")
                    levels[key] = finite_number(value, key, 0, 1)
                _, driven = _external_rates(cfg, topology, levels, assembly_generator)
                mod["tonic_mV"] = _tonic_current(
                    cfg, levels, {row["id"]: i for i, row in enumerate(rows)}, device
                )
                lam_driven = driven * dt
                drive_second = second
            external = lam_driven
        else:
            external = lam_stim if stimulus_on else lam_base
        g_ampa.add_(torch.poisson(external, generator=generator), alpha=w_external)

        # Neuromodulators: each region's level relaxes toward source firing x projection density.
        source_rates = rates.view(R, 3)[mod["source"], 0]
        target = density * (source_rates / mod["resting_hz"])[None, :]
        chem = target + (chem - target) * chem_decay
        occ = chem / (chem + K)
        d_occ = occ - occ_rest
        i_mod = (exc_scale * (mod["excitability"] * d_occ).sum(1) + mod["tonic_mV"])[region_of_cell]
        adapt_gain = (1 + gain * (mod["adaptation"] * d_occ).sum(1)).clamp(0.2, 3.0)[region_of_cell]
        nmda_gain = (1 + gain * (mod["nmda"] * d_occ).sum(1)).clamp(0.2, 3.0)[region_of_cell]

        # Synaptic transmission: local contacts after 1 step, long-range after `delay` steps.
        if ring is not None:
            ring[(step - 1) % delay].copy_(release)
            delayed = ring[(step - delay) % delay]
        else:
            delayed = release
        incoming_e = _spmv(mats["local_glu"], release) + _spmv(mats["long_glu"], delayed)
        incoming_i = _spmv(mats["local_gaba"], release) + _spmv(mats["long_gaba"], delayed)
        g_ampa.add_(incoming_e)
        g_nmda.add_(incoming_e * syn["nmda_to_ampa_ratio"] * nmda_gain)
        g_gaba_a.add_(incoming_i)
        g_gaba_b.add_(incoming_i, alpha=syn["gaba_b_to_a_ratio"])

        # Membrane (AdEx)
        mg_block = 1 / (1 + mg_factor * torch.exp(-0.062 * v))
        i_glu = (g_ampa + g_nmda * mg_block) * (e_glu - v)
        i_gaba = g_gaba_a * (e_gaba_a - v) + g_gaba_b * (e_gaba_b - v)
        upswing = p["delta_t_mV"] * torch.exp(
            ((v - p["v_threshold_mV"]) / p["delta_t_mV"]).clamp(max=10.0)
        )
        dv = dt_over_tau * (p["v_rest_mV"] - v + upswing - w + i_glu + i_gaba + i_mod)
        w.add_(w_rate * (p["a"] * (v - p["v_rest_mV"]) - w))
        v_new = torch.where(refractory == 0, v + dv, p["v_reset_mV"])
        spike = v_new >= p["v_peak_mV"]
        spike_f = spike.to(torch.float32)
        v = torch.where(spike, p["v_reset_mV"], v_new)
        w.add_(spike_f * p["b_mV"] * adapt_gain)
        refractory = torch.where(spike, ref_steps, (refractory - 1).clamp_(min=0))

        # Short-term plasticity (Tsodyks-Markram), normalized so a rested synapse has its nominal weight.
        x.add_((1 - x) * rec_rate)
        u = release_u + (u - release_u) * facil_decay
        u_spike = release_u + facilitating * (u + release_u * (1 - u) - release_u)
        released = spike_f * u_spike * x
        x.sub_(released)
        u = torch.where(spike, u_spike, u)
        release = released / release_u

        dopamine = occ[:, mod["index"]["dopamine"]]
        if base_from <= step < base_to:
            da_sum.add_(dopamine)
            da_steps += 1
            da_baseline = da_sum / da_steps
        learner.step(spike_f, dopamine, dopamine if step < stim_on else da_baseline)
        if learning_config["rule"] != "none":
            for name, c in topology["contacts"].items():
                mats[name].values().zero_().index_add_(0, c["slot"], learner.weights[name])

        # Firing rates per region and cell class
        counts.zero_().index_add_(0, bin_of_cell, spike_f)
        rates.add_(rate_alpha * (counts / (bin_sizes.clamp(min=1) * dt) - rates))
        spike_total.add_(spike.sum())
        record_index = (local_step + offset % record_every) // record_every
        drive_counts.zero_().index_add_(0, region_of_cell, i_glu.abs() + i_gaba.abs())
        pending_drive.add_(drive_counts)
        pending_counts.add_(counts)
        if (step + 1) % record_every == 0:
            rec_drive[record_index] = pending_drive
            rec_counts[record_index] = pending_counts
            rec_occ[record_index] = occ
            pending_drive.zero_()
            pending_counts.zero_()
        for name, lo, hi in (("baseline", base_from, base_to), ("response", resp_from, stim_off)):
            if lo <= step < hi:
                window = windows[name]
                window["counts"].add_(counts)
                window["occ"].add_(occ)
                window["release"].add_(release @ release_masks)
                window["steps"] += 1
        if record:
            raster[local_step] = spike[sample_ids]
        lfp_hist[local_step, 0] = i_glu.abs() @ lfp_weights
        lfp_hist[local_step, 1] = i_gaba.abs() @ lfp_weights

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    def window_means(window):
        if not window["steps"]:
            return None
        seconds = window["steps"] * dt
        per_bin = (window["counts"] / (bin_sizes.clamp(min=1) * seconds)).view(R, 3).cpu().tolist()
        release_cells = torch.stack((glu_f.sum(), gaba_f.sum()))
        return {
            "projection_hz": [row[0] for row in per_bin],
            "interneuron_hz": [row[1] for row in per_bin],
            "occupancy": (window["occ"] / window["steps"]).cpu().tolist(),
            "release_per_cell_hz": (window["release"] / (release_cells.clamp(min=1) * seconds))
            .cpu()
            .tolist(),
        }

    rec_rates = (rec_counts / (bin_sizes.clamp(min=1)[None, :] * record_every * dt)).view(
        n_records, R, 3
    )
    result = {
        "neuron_count": n,
        "steps": steps,
        "time_step_ms": dt_ms,
        "duration_seconds": steps * dt,
        "regions": [row["id"] for row in rows],
        "modulators": mod["names"],
        "region_cells": [row["size"] for row in rows],
        "baseline": window_means(windows["baseline"]),
        "response": window_means(windows["response"]),
        "resting_occupancy": occ_rest.cpu().tolist(),
        "timecourse": {
            "t": [
                round((offset // record_every + i + 1) * record_every * dt, 6)
                for i in range(n_records)
            ],
            "neural_drive": (rec_drive / record_every).cpu().tolist(),
            "projection_hz": rec_rates[:, :, 0].cpu().tolist(),
            "occupancy": rec_occ.cpu().tolist(),
        },
        "spike_count": int(spike_total.item()),
        "elapsed_seconds": elapsed,
        "seed": int(seed),
        "device_name": device_info(device)["name"],
        "learning": learning_config,
    }
    if record:
        result["raster"] = _raster_events(raster, sample_layout, dt_ms)
        for event in result["raster"]["events"]:
            event[0] += offset * dt_ms
    combined_lfp = torch.cat((lfp_tail, lfp_hist), dim=0)
    proxy = combined_lfp[:-shift, 0] - LFP_GABA_WEIGHT * combined_lfp[shift:, 1]
    every = max(1, round(4 / dt_ms))
    first = (-offset) % every
    result["field_potential"] = {
        "sample_rate_hz": 1000 / (every * dt_ms),
        "t": [(offset + i) * dt for i in range(first, steps, every)],
        "channels": {
            name: proxy[first::every, c].cpu().tolist()
            for c, name in enumerate(cfg["field_potential_channels"])
        },
    }
    tensors = dict(
        {
            "v": v,
            "w": w,
            "g_ampa": g_ampa,
            "g_nmda": g_nmda,
            "g_gaba_a": g_gaba_a,
            "g_gaba_b": g_gaba_b,
            "refractory": refractory,
            "u": u,
            "x": x,
            "release": release,
            "rates": rates,
            "chem": chem,
            "spike_total": spike_total,
            "da_sum": da_sum,
            "da_baseline": da_baseline,
        }.items()
    )
    if ring is not None:
        tensors["ring"] = ring
    tensors["rng"] = generator.get_state()
    tensors["pending_counts"] = pending_counts
    tensors["pending_drive"] = pending_drive
    tensors["lfp_tail"] = combined_lfp[-shift:].clone()
    tensors.update({f"learning.{key}": value for key, value in learner.tensors().items()})
    metadata = {
        "version": 1,
        "n": n,
        "step": offset + steps,
        "da_steps": da_steps,
        "inputs": dict(inputs),
        "learning": learning_config,
        "seed": int(seed),
        "dt_ms": dt_ms,
        "config_hash": topology["config_hash"],
    }
    for name, window in windows.items():
        metadata[f"window.{name}.steps"] = window["steps"]
        tensors.update(
            {f"window.{name}.{key}": window[key] for key in ("counts", "occ", "release")}
        )
    if any(not torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError("Simulation produced non-finite state.")
    return result, RunState(tensors, metadata)


# --------------------------------------------------------------------------- recording helpers


def _raster_sample(cfg, rows, device):
    per_region = cfg["runtime"]["raster_cells_per_region"]
    ids, sample_layout = [], []
    for region, row in zip(cfg["regions"], rows, strict=False):
        count = min(per_region, row["size"])
        proj = max(1, min(count - 1, round(count * row["projection"] / row["size"])))
        p_ids = (
            torch.linspace(0, row["projection"] - 1, proj).round().to(torch.int64) + row["start"]
        )
        i_ids = (
            torch.linspace(0, row["sst_count"] + row["vip_count"] - 1, count - proj)
            .round()
            .to(torch.int64)
            + row["start"]
            + row["projection"]
        )
        ids.extend((p_ids, i_ids))
        sample_layout.append(
            {
                "id": row["id"],
                "cells": count,
                "projection": proj,
                "transmitter": region["transmitter"],
            }
        )
    return torch.cat(ids).to(device), sample_layout


def _raster_events(raster, sample_layout, dt_ms, limit=90_000):
    events = raster.nonzero(as_tuple=False)
    if events.shape[0] > limit:
        events = events[:: math.ceil(events.shape[0] / limit)]
    return {
        "layout": sample_layout,
        "events": [[round(step * dt_ms, 2), cell] for step, cell in events.cpu().tolist()],
    }


def _lfp_weights(cfg, topology):
    """Weights over projection cells of glutamatergic regions under each schematic electrode."""
    channels = cfg["field_potential_channels"]
    by_id = {row["id"]: row for row in topology["rows"]}
    weights = torch.zeros((topology["n"], len(channels)), device=topology["device"])
    for c, mix in enumerate(channels.values()):
        for region_id, weight in mix.items():
            row = by_id[region_id]
            if not row["gaba_projection"]:
                weights[row["start"] : row["start"] + row["projection"], c] = (
                    weight / row["projection"]
                )
    return weights
