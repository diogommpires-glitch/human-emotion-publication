import numpy as np
import pytest

from research import ExperimentSpec, ExperimentStore
from research.observations import hrf_observation, log_relative_bands
from research.statistics import cluster_bootstrap, crossed_folds, holm, nested_ridge, temporal_null


def test_crossed_folds_exclude_both_identities():
    participant = np.repeat(np.arange(10), 6)
    stimulus = np.tile(np.arange(6), 10)
    folds = list(crossed_folds(participant, stimulus, 5, 3, seed=11))
    assert len(folds) == 15
    for train, test in folds:
        assert not set(participant[train]) & set(participant[test])
        assert not set(stimulus[train]) & set(stimulus[test])
        assert len(train) and len(test)


def test_nested_ridge_fits_without_test_outcome_leakage():
    rng = np.random.default_rng(11)
    p = np.repeat(np.arange(10), 6)
    s = np.tile(np.arange(6), 10)
    x = rng.normal(size=(60, 3))
    y = x[:, :1] * 2
    result = nested_ridge(x, y, p, s, participant_folds=5, stimulus_folds=3)
    assert result["mse"] < 0.01
    assert result["predictions"].shape == y.shape
    assert set(result["alphas"]) <= {0.01, 0.1, 1.0, 10.0, 100.0}


def test_bands_and_hrf_use_physical_time():
    fs = 256
    t = np.arange(fs * 8) / fs
    bands = log_relative_bands(np.sin(2 * np.pi * 10 * t)[None, :], fs)
    assert np.argmax(bands[0]) == 2
    drive = np.zeros((200, 1))
    drive[0] = 1
    bold = hrf_observation(drive, 0.1, np.arange(0, 20, 1.0))
    assert 4 <= np.argmax(bold[:, 0]) <= 7
    assert bold.shape == (20, 1)


def test_statistics_are_seeded_and_control_families():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    p = np.repeat(np.arange(5), 4)
    s = np.tile(np.arange(4), 5)
    a = cluster_bootstrap(np.ones(20), p, s, iterations=100, seed=11)
    assert a["lower"] == a["upper"] == 1
    y = np.sin(np.arange(100) / 4)
    result = temporal_null(y, y, np.repeat(np.arange(2), 50), iterations=100, seed=11)
    assert result["p_value"] < 0.05


def test_manifest_cancel_resume_and_numeric_chunks(tmp_path):
    store = ExperimentStore(tmp_path)
    spec = ExperimentSpec(protocol="synthetic", duration_seconds=0.03, chunk_seconds=0.01)
    job = store.submit(spec)
    assert job["status"] == "queued"
    assert store.cancel(job["id"])["status"] == "cancelled"
    assert store.resume(job["id"])["status"] == "queued"
    store.run_pending()
    result = store.get(job["id"])
    assert result["status"] == "completed"
    assert result["progress"] == 1
    assert result["empirical_validation"] is False
    assert len(result["chunks"]) == 3
    for chunk in result["chunks"]:
        arrays = np.load(tmp_path / job["id"] / chunk["path"], allow_pickle=False)
        assert np.isfinite(arrays["time_seconds"]).all()
    assert ExperimentStore(tmp_path).get(job["id"]) == result


def test_specs_reject_unknown_fields_and_invalid_limits():
    with pytest.raises(ValueError):
        ExperimentSpec.from_dict({"protocol": "synthetic", "duration_seconds": float("nan")})
    with pytest.raises(ValueError):
        ExperimentSpec.from_dict({"protocol": "synthetic", "personal_notes": "private"})
    with pytest.raises(ValueError):
        ExperimentSpec(protocol="synthetic", chunk_seconds=0)


@pytest.mark.parametrize("path", [{"private": "note"}, ["cohort.json"], 12, "", "bad\0path"])
def test_source_paths_reject_non_path_payloads_at_submission(path):
    with pytest.raises(ValueError, match="path"):
        ExperimentSpec.from_dict({"protocol": "ratings", "source_path": path})


def test_hrf_maps_repeated_participant_scan_times_back_to_rows():
    drive = np.zeros((201, 3))
    drive[0] = [1, 2, 3]
    one_person = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    repeated = np.tile(one_person, 3)
    single = hrf_observation(drive, 0.1, one_person)
    crossed = hrf_observation(drive, 0.1, repeated)
    np.testing.assert_allclose(crossed, np.tile(single, (3, 1)))
    assert crossed.shape == (15, 3)
