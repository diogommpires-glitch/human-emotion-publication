"""Pure projections from recorded neural activity to descriptive readouts."""

import math

from .config import MOOD_WEIGHT, NETWORK


class _Activity:
    """Compare two activity windows (a against b) region by region."""

    def __init__(self, a, b, regions, modulators):
        self.a, self.b = a, b
        self.index = {region: i for i, region in enumerate(regions)}
        self.mod = {name: m for m, name in enumerate(modulators)}

    def rate(self, region):
        """log2 change in projection-cell firing, softened by 0.5 Hz and clipped."""
        i = self.index[region]
        ratio = (self.a["projection_hz"][i] + 0.5) / (self.b["projection_hz"][i] + 0.5)
        return max(-2.0, min(3.0, math.log2(ratio)))

    def chem(self, name, region):
        """Change in receptor occupancy, scaled so ~0.1 counts like a doubling of firing."""
        i, m = self.index[region], self.mod[name]
        return 8.0 * (self.a["occupancy"][i][m] - self.b["occupancy"][i][m])


def _components(act):
    L, C = act.rate, act.chem
    return {
        "threat": 0.35 * L("amygdala_cea")
        + 0.25 * L("pag")
        + 0.2 * L("amygdala_bla")
        + 0.2 * C("noradrenaline", "amygdala_bla"),
        "anxiety": 0.5 * L("bnst") + 0.3 * L("dacc") + 0.2 * C("noradrenaline", "bnst"),
        "reward": 0.35 * C("dopamine", "nucleus_accumbens")
        + 0.25 * L("vta")
        + 0.2 * L("nucleus_accumbens")
        + 0.2 * L("amygdala_bla_reward"),
        "loss": 0.6 * L("lateral_habenula")
        - 0.2 * C("dopamine", "nucleus_accumbens")
        - 0.2 * L("raphe"),
        "arousal": 0.3 * L("locus_coeruleus")
        + 0.3 * C("noradrenaline", "amygdala_bla")
        + 0.2 * L("hypothalamus")
        + 0.2 * L("motor_brainstem"),
        "calm": 0.35 * C("serotonin", "amygdala_bla")
        + 0.25 * C("oxytocin", "amygdala_cea")
        - 0.2 * L("amygdala_cea")
        - 0.2 * L("locus_coeruleus"),
        "social": 0.5 * L("oxytocin_pvn")
        + 0.3 * C("oxytocin", "amygdala_cea")
        + 0.2 * L("insula")
        - 0.3 * L("pag"),
        "novelty": 0.5 * L("hippocampus")
        + 0.3 * C("acetylcholine", "sensory_cortex")
        + 0.2 * L("basal_forebrain"),
        "distress": 0.5 * L("insula") + 0.5 * L("pag") - 0.25 * L("amygdala_bla"),
        "control": 0.6 * L("dlpfc") - 0.4 * L("dacc"),
    }


def _emotions(e, phasic, mood):
    """Fear answers an explicit cue (phasic threat); anxiety is sustained threat without one (mood)."""
    logits = {
        "Fear-like": 1.2 * phasic["threat"] + 0.6 * e["arousal"] - 0.5 * e["control"],
        "Anxiety-like": 1.3 * e["anxiety"]
        + 0.8 * mood["threat"]
        + 0.4 * e["arousal"]
        - 0.4 * e["control"]
        - 0.3 * phasic["threat"],
        "Anger-like": 0.8 * e["threat"]
        + 0.9 * e["control"]
        + 0.4 * e["arousal"]
        - 0.4 * e["anxiety"],
        "Sadness-like": 1.3 * e["loss"] - 0.4 * e["arousal"] - 0.4 * e["reward"],
        "Joy-like": 1.3 * e["reward"] + 0.3 * e["arousal"] - 0.6 * e["threat"] - 0.5 * e["loss"],
        "Calm-like": 1.2 * e["calm"] - 0.6 * e["arousal"] - 0.4 * e["threat"],
        "Curiosity-like": 1.2 * e["novelty"] + 0.3 * e["reward"] - 0.6 * e["threat"],
        "Social warmth-like": 1.3 * e["social"] + 0.3 * e["reward"] - 0.5 * e["threat"],
        "Distress-like": 1.2 * e["distress"] + 0.4 * e["threat"] - 0.3 * e["control"],
        "Little change": 0.5,
    }
    peak = max(logits.values())
    exps = {name: math.exp((value - peak) / 0.35) for name, value in logits.items()}
    total = sum(exps.values())
    ranked = sorted(
        ({"label": name, "score": round(100 * value / total)} for name, value in exps.items()),
        key=lambda row: row["score"],
        reverse=True,
    )
    return ranked[:3], {name: round(value, 3) for name, value in logits.items()}


def _dimensions(e):
    return {
        "valence": math.tanh(
            0.7
            * (
                e["reward"]
                + e["social"]
                + 0.5 * e["calm"]
                - e["threat"]
                - 0.8 * e["anxiety"]
                - e["loss"]
                - 0.8 * e["distress"]
            )
        ),
        "arousal": math.tanh(
            0.8 * (e["arousal"] + 0.4 * e["threat"] + 0.3 * e["reward"] + 0.3 * e["novelty"])
        ),
        "agency": math.tanh(e["control"]),
        "social_safety": math.tanh(
            0.8 * (e["social"] + 0.5 * e["calm"] - 0.6 * e["threat"] - 0.3 * e["anxiety"])
        ),
    }


def _weighted_chemistry(window, modulators, regions, cfg):
    """Density-weighted mean receptor occupancy of each chemical over its target regions."""
    index = {region: i for i, region in enumerate(regions)}
    result = {}
    for m, mod in enumerate(cfg["neuromodulators"]):
        total = sum(mod["targets"].values())
        result[mod["name"]] = (
            sum(window[index[r]][m] * d for r, d in mod["targets"].items()) / total
        )
    return result


def _filter_trace(values, sample_rate):
    # Remove slow drift for a readable EEG-like trace. No clinical filter or microvolt
    # calibration is implied.
    alpha = 1 - math.exp(-1 / (sample_rate * 0.55))
    baseline = values[0] if values else 0.0
    filtered = []
    for value in values:
        baseline += alpha * (value - baseline)
        filtered.append(value - baseline)
    return filtered


def _spectrum(values, sample_rate):
    if len(values) < 16:
        return {"frequencies_hz": [], "power": [], "bands": {}, "peak_hz": None}
    mean = sum(values) / len(values)
    count = len(values)
    window = [0.5 - 0.5 * math.cos(2 * math.pi * i / (count - 1)) for i in range(count)]
    samples = [(value - mean) * window[i] for i, value in enumerate(values)]
    frequencies = list(range(1, 46))
    power = []
    for frequency in frequencies:
        omega = 2 * math.pi * frequency / sample_rate
        real = sum(sample * math.cos(omega * i) for i, sample in enumerate(samples))
        imag = sum(sample * math.sin(omega * i) for i, sample in enumerate(samples))
        power.append(real * real + imag * imag)
    total = sum(power) or 1e-18
    bands = {}
    for name, low, high in (
        ("delta", 1, 4),
        ("theta", 4, 8),
        ("alpha", 8, 13),
        ("beta", 13, 30),
        ("gamma", 30, 46),
    ):
        bands[name] = round(
            100
            * sum(p for f, p in zip(frequencies, power, strict=False) if low <= f < high)
            / total,
            1,
        )
    peak = max(power) or 1.0
    return {
        "frequencies_hz": frequencies,
        "power": [round(value / peak, 5) for value in power],
        "bands": bands,
        "peak_hz": frequencies[power.index(max(power))],
    }


def _interpretation(raw, regions_out, chemistry, emotions, inputs, channels):
    stimuli = [
        channel["label"].lower()
        for key, channel in channels.items()
        if channel["timing"] == "stimulus" and inputs[key] > 0.05
    ]
    parts = []
    if stimuli:
        parts.append(
            f"The stimulus ({', '.join(stimuli)}) entered through its sensory and learned-association routes."
        )
    else:
        parts.append(
            "No sensory stimulus was switched on, so any change reflects ongoing network activity and noise."
        )
    changed = sorted(regions_out, key=lambda row: abs(row["change_log2"]), reverse=True)[:3]
    if changed and abs(changed[0]["change_log2"]) >= 0.3:
        described = [
            f"{row['label']} {row['baseline_hz']:.1f}→{row['response_hz']:.1f} Hz"
            for row in changed
            if abs(row["change_log2"]) >= 0.3
        ]
        parts.append("Largest firing changes: " + "; ".join(described) + ".")
    shifts = sorted(
        chemistry, key=lambda row: abs(row["response"] - row["baseline"]), reverse=True
    )[:2]
    shifts = [row for row in shifts if abs(row["response"] - row["baseline"]) >= 0.01]
    if shifts:
        parts.append(
            "Chemical signals: "
            + "; ".join(
                f"{row['name']} {'rose' if row['response'] > row['baseline'] else 'fell'} "
                f"({row['baseline']:.2f}→{row['response']:.2f} receptor occupancy)"
                for row in shifts
            )
            + "."
        )
    parts.append(
        f"The closest readout is {emotions[0]['label']}; it summarizes model activity and is not an emotion measurement."
    )
    return " ".join(parts)


def _regional_readouts(raw, cfg):
    regions = raw["regions"]
    base, resp = raw["baseline"], raw["response"]
    region_by_id = {row["id"]: row for row in cfg["regions"]}
    regions_out = []
    for i, region_id in enumerate(regions):
        region = region_by_id[region_id]
        b, r = base["projection_hz"][i], resp["projection_hz"][i]
        regions_out.append(
            {
                "id": region_id,
                "label": region["label"],
                "role": region["role"],
                "transmitter": region["transmitter"],
                "cells": raw["region_cells"][i],
                "baseline_hz": round(b, 2),
                "response_hz": round(r, 2),
                "interneuron_baseline_hz": round(base["interneuron_hz"][i], 2),
                "interneuron_response_hz": round(resp["interneuron_hz"][i], 2),
                "change_log2": round(math.log2((r + 0.5) / (b + 0.5)), 3),
            }
        )

    return regions_out


def _chemical_readouts(raw, cfg):
    regions, modulators = raw["regions"], raw["modulators"]
    base, resp = raw["baseline"], raw["response"]
    resting = _weighted_chemistry(raw["resting_occupancy"], modulators, regions, cfg)
    chem_base = _weighted_chemistry(base["occupancy"], modulators, regions, cfg)
    chem_resp = _weighted_chemistry(resp["occupancy"], modulators, regions, cfg)
    chemistry_rows = [
        {
            "name": name,
            "resting": round(resting[name], 3),
            "baseline": round(chem_base[name], 3),
            "response": round(chem_resp[name], 3),
        }
        for name in modulators
    ]
    tc = raw["timecourse"]
    stride = max(1, len(tc["t"]) // 150)
    series = {name: [] for name in modulators}
    for occupancy in tc["occupancy"][::stride]:
        for name, value in _weighted_chemistry(occupancy, modulators, regions, cfg).items():
            series[name].append(round(value, 4))

    return chemistry_rows, {"t": tc["t"][::stride], "series": series}


def build_response(request, raw, reference, topology, *, cfg=None):
    cfg = NETWORK if cfg is None else cfg
    channels_by_id = {row["id"]: row for row in cfg["inputs"]}
    regions, modulators = raw["regions"], raw["modulators"]
    base, resp = raw["baseline"], raw["response"]
    phasic = _components(_Activity(resp, base, regions, modulators))
    mood = _components(_Activity(base, reference, regions, modulators))
    evidence = {key: phasic[key] + MOOD_WEIGHT * mood[key] for key in phasic}
    emotions, logits = _emotions(evidence, phasic, mood)

    regions_out = _regional_readouts(raw, cfg)

    chemistry_rows, chemical_timecourse = _chemical_readouts(raw, cfg)

    fp = raw["field_potential"]
    channels = {
        name: _filter_trace(values, fp["sample_rate_hz"]) for name, values in fp["channels"].items()
    }
    mean_trace = [sum(values) / len(values) for values in zip(*channels.values(), strict=False)]

    rt, inputs = cfg["runtime"], request["inputs"]
    return {
        "model": cfg["name"],
        "model_status": cfg["status"],
        "event": request["event"],
        "person": request["person"],
        "inputs_used": {key: round(value, 3) for key, value in inputs.items()},
        "timeline": {
            "duration_seconds": raw["duration_seconds"],
            "stimulus_seconds": rt["stimulus_seconds"],
            "baseline_window_seconds": rt["baseline_window_seconds"],
        },
        "dimensions": {key: round(value, 3) for key, value in _dimensions(evidence).items()},
        "evidence": {
            "response": {k: round(v, 3) for k, v in phasic.items()},
            "mood": {k: round(v, 3) for k, v in mood.items()},
            "emotion_logits": logits,
        },
        "emotions": emotions,
        "regions": regions_out,
        "chemistry": {
            "units": "receptor occupancy (0-1), density-weighted over each chemical's target regions",
            "neuromodulators": chemistry_rows,
            "synaptic_release_per_cell_hz": {
                "glutamate": [
                    round(base["release_per_cell_hz"][0], 3),
                    round(resp["release_per_cell_hz"][0], 3),
                ],
                "gaba": [
                    round(base["release_per_cell_hz"][1], 3),
                    round(resp["release_per_cell_hz"][1], 3),
                ],
            },
            "tonic": {
                factor["name"]: round(inputs[factor["input"]], 3)
                for factor in cfg.get("tonic_factors", [])
            },
            "timecourse": chemical_timecourse,
        },
        "spikes": {**raw["raster"], "total_count": raw["spike_count"]},
        "eeg": {
            "sample_rate_hz": fp["sample_rate_hz"],
            "units": "arbitrary model units (LFP proxy)",
            "channels": {
                name: [round(value, 4) for value in values] for name, values in channels.items()
            },
            "spectrum": _spectrum(mean_trace, fp["sample_rate_hz"]),
        },
        "simulation": {
            "neuron_count": raw["neuron_count"],
            "requested_neuron_count": request["neurons"],
            "synapse_count": topology["synapse_count"],
            "regions": len(regions),
            "time_step_ms": raw["time_step_ms"],
            "duration_seconds": raw["duration_seconds"],
            "cell_model": "adaptive_exponential_integrate_and_fire",
            "synapse_model": "ampa_nmda_gabaa_gabab_with_tsodyks_markram_short_term_plasticity",
            "backend": "CUDA" if topology["device"].type == "cuda" else "CPU",
            "device": raw["device_name"],
            "elapsed_seconds": round(raw["elapsed_seconds"], 2),
            "topology_build_seconds": round(topology["build_seconds"], 2),
            "random_seed": raw["seed"],
            "plasticity": {
                "learning": raw.get("learning"),
            },
        },
        "interpretation": _interpretation(
            raw, regions_out, chemistry_rows, emotions, inputs, channels_by_id
        ),
        "caveat": (
            f"This is a reduced {raw['neuron_count']:,}-cell spiking model of emotion circuits, not a human brain. "
            "Its regions, wiring strengths, receptor effects and clearance times are literature-inspired priors, "
            "not measurements of your brain. Chemical values are model receptor occupancies, not concentrations. "
            "The EEG-like traces are a standard LFP proxy from simulated synaptic currents, not scalp EEG. "
            "Emotion labels are hand-designed summaries of model activity, not measurements of feelings."
        ),
    }
