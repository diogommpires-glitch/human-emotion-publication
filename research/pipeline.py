"""Scientific results from explicitly prepared, provenance-linked local cohorts."""

import json
from pathlib import Path

import numpy as np

from .spec import DATASETS, SEEDS
from .statistics import (
    cluster_bootstrap,
    crossfit_mean,
    matrix,
    nested_feature_ridge,
    nested_gain_calibration,
    scores,
    temporal_null,
)
from .store import load_arrays, sha256


def validate_cohort_identity(manifest, protocol):
    if manifest.get("schema_version") != 1 or manifest.get("protocol") != protocol:
        raise ValueError("cohort protocol/schema mismatch")
    dataset = manifest.get("dataset")
    if dataset not in DATASETS or manifest.get("version") != DATASETS[dataset]:
        raise ValueError("cohort must identify a pinned OpenNeuro release")
    expected = {"eeg": {"ds005540"}, "fmri": {"ds004892"}, "ratings": {"ds005540", "ds004872"}}
    if dataset not in expected[protocol]:
        raise ValueError("cohort dataset does not match modality; film cohorts must stay separate")
    if (
        not manifest.get("provenance")
        or not manifest.get("stimulus_encoding")
        or not manifest.get("stimuli")
    ):
        raise ValueError("cohort requires preprocessing provenance and actual stimulus records")
    if manifest.get("learning_rule", "none") not in {"none", "pair_stdp"}:
        raise ValueError("cohort learning rule must be prespecified as none or pair_stdp")


def validate_cohort_groups(values, manifest):
    trial_ids = values["trial_index"]
    if (
        trial_ids.dtype.kind not in "iu"
        or (trial_ids < 0).any()
        or (trial_ids >= len(manifest["stimuli"])).any()
    ):
        raise ValueError("trial indices must identify a declared stimulus")
    for trial in np.unique(trial_ids):
        if len(np.unique(values["stimulus_group"][trial_ids == trial])) != 1:
            raise ValueError("one stimulus cannot belong to multiple split families")
    for series in np.unique(values["series"]):
        times = values["time_seconds"][values["series"] == series]
        if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
            raise ValueError("series must have strictly increasing actual times in seconds")


def load_cohort(path, protocol, budget_bytes=8 * 1024**3):
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_cohort_identity(manifest, protocol)
    array_path = (manifest_path.parent / manifest["arrays"]).resolve()
    if (
        not array_path.is_relative_to(manifest_path.parent)
        or sha256(array_path) != manifest["sha256"]
    ):
        raise ValueError("cohort arrays must match their local manifest hash")
    values = load_arrays(array_path, budget_bytes)
    required = {
        "conditions",
        "outcomes",
        "participant",
        "stimulus_group",
        "series",
        "time_seconds",
        "trial_index",
        "sample_seconds",
    }
    if not required <= values.keys():
        raise ValueError(f"cohort is missing arrays {sorted(required - values.keys())}")
    outcomes = matrix(values["outcomes"], "outcomes")
    if any(len(values[name]) != len(outcomes) for name in required):
        raise ValueError("cohort arrays do not align with outcomes")
    validate_cohort_groups(values, manifest)
    target_names = {
        "eeg": ["delta", "theta", "alpha", "beta", "gamma"],
        "fmri": ["amygdala", "hippocampus", "insula"],
        "ratings": ["valence", "arousal"]
        if manifest["dataset"] == "ds005540"
        else ["Good", "Bad", "Alert", "Calm"],
    }[protocol]
    if manifest.get("target_names") != target_names or outcomes.shape[1] != len(target_names):
        raise ValueError("target names/dimensions must match the prespecified modality")
    return manifest, values


def fit_seed_methods(builder, values, seed, stimulus_folds, checkpoint, progress, seed_index):
    y, p, s = values["outcomes"], values["participant"], values["stimulus_group"]
    baseline = crossfit_mean(y, p, s, 5, stimulus_folds, seed)
    methods = {"training_mean": {**scores(y, baseline), "predictions": baseline}}
    names = ("conditions", "stimulus", "frozen_simulator", "randomised_graph", "simulator")
    for method_index, name in enumerate(names):
        cached = checkpoint.get(seed, name) if checkpoint is not None else None
        if (
            cached is None
            and name == "simulator"
            and builder.manifest.get("learning_rule", "none") == "none"
        ):
            cached = dict(methods["frozen_simulator"])
            if checkpoint is not None:
                checkpoint.save(seed, name, cached)
        if cached is None:
            cached = nested_feature_ridge(
                builder,
                name,
                y,
                p,
                s,
                5,
                stimulus_folds,
                seed,
                builder.cancel,
                bold=builder.spec.protocol == "fmri",
            )
            if checkpoint is not None:
                checkpoint.save(seed, name, cached)
        methods[name] = cached
        progress((seed_index * 5 + method_index + 1) / 26, f"seed {seed}: {name}")
    return methods


def add_calibration(methods, builder, values, seed, stimulus_folds, checkpoint):
    settings = builder.manifest.get("calibration_settings")
    if not settings:
        return
    calibrated = checkpoint.get(seed, "calibrated_simulator") if checkpoint is not None else None
    if calibrated is None:
        calibrated = nested_gain_calibration(
            builder,
            settings,
            values["outcomes"],
            values["participant"],
            values["stimulus_group"],
            stimulus_folds,
            seed,
            builder.cancel,
        )
        if checkpoint is not None:
            checkpoint.save(seed, "calibrated_simulator", calibrated)
    methods["calibrated_simulator"] = calibrated


def seed_uncertainty(methods, values, spec, seed):
    y = matrix(values["outcomes"], "outcomes")
    prediction, baseline = (
        methods["simulator"]["predictions"],
        methods["training_mean"]["predictions"],
    )
    if spec.protocol == "fmri":
        from .statistics import bold_cluster_bootstrap

        interval = bold_cluster_bootstrap(
            y,
            prediction,
            baseline,
            values["participant"],
            values["stimulus_group"],
            spec.bootstrap_iterations,
            seed,
        )
    else:
        baseline_error = np.abs(y - baseline) if spec.protocol == "ratings" else (y - baseline) ** 2
        model_error = (
            np.abs(y - prediction) if spec.protocol == "ratings" else (y - prediction) ** 2
        )
        interval = cluster_bootstrap(
            np.mean(baseline_error - model_error, axis=1),
            values["participant"],
            values["stimulus_group"],
            spec.bootstrap_iterations,
            seed,
        )
    metric = {"eeg": "mse", "fmri": "explained_variance", "ratings": "mae"}[spec.protocol]
    null = temporal_null(
        y, prediction, values["series"], spec.permutation_iterations, seed, metric=metric
    )
    for method in methods.values():
        del method["predictions"]
    return {
        "seed": seed,
        "methods": methods,
        "paired_improvement_vs_mean": interval,
        "temporal_null": null,
    }


def evaluate_cohort(
    spec, cancel=lambda: None, *, checkpoint_dir=None, progress=lambda fraction, stage: None
):
    from .checkpoints import EvaluationCheckpoint
    from .features import FoldFeatures
    from .provenance import scientific_identity

    identity = scientific_identity()
    manifest, values = load_cohort(spec.source_path, spec.protocol, spec.budget_bytes)
    if spec.dataset_manifest:
        from .datasets import DatasetManifest

        if (
            DatasetManifest.read(spec.dataset_manifest).to_dict()
            != manifest["provenance"]["acquisition"]
        ):
            raise ValueError(
                "experiment dataset manifest differs from cohort acquisition provenance"
            )
    stimuli = values["stimulus_group"]
    stimulus_folds = (
        len(np.unique(stimuli)) if manifest["dataset"] in {"ds004892", "ds004872"} else 3
    )
    builder = FoldFeatures(manifest, values, Path(spec.source_path).resolve().parent, spec, cancel)
    checkpoint = None
    if checkpoint_dir is not None:
        checkpoint = EvaluationCheckpoint(
            checkpoint_dir,
            {
                "cohort_sha256": sha256(spec.source_path),
                "spec": spec.to_dict(),
                "scientific_identity": identity,
            },
            spec.budget_bytes,
        )
    results = []
    for seed_index, seed in enumerate(SEEDS):
        cancel()
        methods = fit_seed_methods(
            builder, values, seed, stimulus_folds, checkpoint, progress, seed_index
        )
        add_calibration(methods, builder, values, seed, stimulus_folds, checkpoint)
        results.append(seed_uncertainty(methods, values, spec, seed))
        builder.cache.clear()
    return {
        "schema_version": 1,
        "scientific_identity": identity,
        "protocol": spec.protocol,
        "dataset": manifest["dataset"],
        "version": manifest["version"],
        "cohort_sha256": manifest["sha256"],
        "primary_metric": {"eeg": "mse", "fmri": "explained_variance", "ratings": "mae"}[
            spec.protocol
        ],
        "seeds": results,
        "empirical_validation": False,
        "interpretation": "Held-out estimates, including negative outcomes. No clinical or individual emotion inference.",
        "multiplicity": "Use research combine with all three modality results for Holm correction; seeds are sensitivity analyses, not independent tests.",
    }
