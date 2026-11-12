import json

import numpy as np
import pytest

from research import ExperimentSpec
from research.pipeline import evaluate_cohort
from research.statistics import bold_cluster_bootstrap, scores, temporal_null
from research.store import Cancelled, sha256


def test_prepared_cohort_nested_pipeline_and_persisted_resume(tmp_path, monkeypatch):
    stimuli = []
    for index in range(6):
        path = tmp_path / f"stimulus-{index}.npz"
        np.savez(
            path,
            time_seconds=np.arange(3),
            features=np.array([[index, 1], [index + 0.2, 2], [index + 0.4, 3]]),
        )
        stimuli.append(
            {
                "id": f"fixture-{index}",
                "family": f"family-{index}",
                "kind": "author_av",
                "duration_seconds": 3.0,
                "path": path.name,
                "sha256": sha256(path),
            }
        )
    participants = np.repeat(np.arange(10), 6)
    trial = np.tile(np.arange(6), 10)
    arrays = tmp_path / "cohort.npz"
    np.savez(
        arrays,
        outcomes=np.column_stack([trial / 5, 1 - trial / 5]) + participants[:, None] * 0.001,
        conditions=np.eye(6)[trial],
        participant=participants,
        stimulus_group=trial,
        series=participants,
        time_seconds=trial * 4.0,
        sample_seconds=np.zeros(60),
        trial_index=trial,
    )
    manifest = {
        "schema_version": 1,
        "protocol": "ratings",
        "dataset": "ds005540",
        "version": "1.0.7",
        "arrays": arrays.name,
        "sha256": sha256(arrays),
        "stimuli": stimuli,
        "target_names": ["valence", "arousal"],
        "learning_rule": "none",
        "stimulus_encoding": "synthetic author-feature fixture",
        "provenance": {"synthetic_fixture": True},
    }
    manifest_path = tmp_path / "cohort.json"
    manifest_path.write_text(json.dumps(manifest))
    calls = []

    def numerical_simulator_boundary(spec, drives, **kwargs):
        calls.append((spec.seed, kwargs["randomised_graph"]))
        values = np.array([list(row.values()) for row in drives])
        if kwargs["randomised_graph"]:
            values = values[:, ::-1]
        return {
            "time_seconds": np.arange(1, 4),
            "projection_hz": values[:, :3] * 20,
            "neural_drive": values[:, 3:] * 2,
            "occupancy": values[:, None, :],
        }

    monkeypatch.setattr("research.features.simulate_stimulus", numerical_simulator_boundary)
    spec = ExperimentSpec(protocol="ratings", source_path=str(manifest_path))
    checkpoint = tmp_path / "run"
    checkpoint.mkdir()

    def stop_after_frozen(fraction, stage):
        if stage == "seed 11: frozen_simulator":
            raise Cancelled("synthetic cancellation between completed fits")

    with pytest.raises(Cancelled):
        evaluate_cohort(spec, checkpoint_dir=checkpoint, progress=stop_after_frozen)
    before = json.loads((checkpoint / "evaluation-progress.json").read_text())
    frozen = before["completed"]["11:frozen_simulator"]
    first_hash = sha256(checkpoint / frozen["path"])
    result = evaluate_cohort(spec, checkpoint_dir=checkpoint)
    assert result["primary_metric"] == "mae"
    assert result["empirical_validation"] is False
    assert len(result["seeds"]) == 5
    assert all(row["temporal_null"]["metric"] == "mae" for row in result["seeds"])
    assert all(np.isfinite(row["methods"]["simulator"]["mae"]) for row in result["seeds"])
    assert sha256(checkpoint / frozen["path"]) == first_hash
    after = json.loads((checkpoint / "evaluation-progress.json").read_text())
    assert len(after["completed"]) == 25
    call_count = len(calls)
    repeated = evaluate_cohort(spec, checkpoint_dir=checkpoint)
    assert repeated == result
    assert len(calls) == call_count
    from brain_model import model

    monkeypatch.setitem(
        model.NETWORK["runtime"], "topology_seed", model.NETWORK["runtime"]["topology_seed"] + 1
    )
    with pytest.raises(ValueError, match="differs"):
        evaluate_cohort(spec, checkpoint_dir=checkpoint)


def test_primary_inference_metrics_do_not_substitute_mse():
    observed = np.tile(np.arange(5.0), 4)
    shifted = observed + 10
    flattened = np.full_like(observed, observed.mean())
    assert scores(observed, shifted)["mse"] > scores(observed, flattened)["mse"]
    assert (
        scores(observed, shifted)["explained_variance"]
        > scores(observed, flattened)["explained_variance"]
    )
    participants = np.repeat(np.arange(4), 5)
    stimuli = np.tile(np.arange(5), 4)
    interval = bold_cluster_bootstrap(
        observed[:, None],
        shifted[:, None],
        flattened[:, None],
        participants,
        stimuli,
        iterations=10000,
    )
    assert interval["estimate"] == pytest.approx(1.0)
    assert interval["lower"] == pytest.approx(1.0)
    assert interval["upper"] == pytest.approx(1.0)
    assert interval["undefined_draws"] > 0
    assert interval["valid_draws"] + interval["undefined_draws"] + interval["empty_draws"] == 10000
    assert interval["valid_draws"] >= 9000
    ev_null = temporal_null(
        observed, shifted, participants, iterations=100, metric="explained_variance"
    )
    assert ev_null["statistic"] == pytest.approx(1.0)
    assert ev_null["p_value"] < 0.05
    error = np.zeros_like(observed)
    error[0] = 20
    mae_null = temporal_null(observed, observed + error, participants, iterations=100, metric="mae")
    assert mae_null["statistic"] == pytest.approx(1.0)
    assert scores(observed, observed + error)["mse"] == 20


def test_bold_bootstrap_rejects_constant_source_and_too_many_undefined_draws():
    participants = np.repeat(np.arange(4), 2)
    stimuli = np.tile(np.arange(2), 4)
    constant = np.ones((8, 1))
    with pytest.raises(ValueError, match="constant"):
        bold_cluster_bootstrap(constant, constant, constant, participants, stimuli)
    observed = stimuli[:, None].astype(float)
    with pytest.raises(ValueError, match="90%"):
        bold_cluster_bootstrap(
            observed,
            observed,
            np.zeros_like(observed),
            participants,
            stimuli,
            iterations=100,
            seed=11,
        )
