import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from research import ExperimentSpec, ExperimentStore
from research.checkpoints import EvaluationCheckpoint
from research.datasets import DatasetFile, DatasetManifest, acquire, read_events
from research.encoders import ImageryEncoder, StimulusScaler, film_frame_features
from research.features import FoldFeatures
from research.fmri import extract_rois, fmriprep_command
from research.observations import fit_bold_observation, preprocess_eeg
from research.simulation import synthetic_chunk
from research.statistics import calibrate_gains, nested_feature_ridge
from research.store import sha256


def test_cancel_between_chunks_and_resume_preserves_times(tmp_path):
    store = ExperimentStore(tmp_path)
    spec = ExperimentSpec(duration_seconds=0.04, chunk_seconds=0.01)
    job = store.submit(spec)

    def adapter(spec, checkpoint, start, duration):
        result = synthetic_chunk(spec, checkpoint, start, duration)
        store.cancel(job["id"])
        return result

    store.run_pending(adapter)
    stopped = store.get(job["id"])
    assert stopped["status"] == "cancelled"
    assert stopped["completed_seconds"] == 0.01
    first_hash = stopped["chunks"][0]["sha256"]
    store.resume(job["id"])
    store.run_pending()
    finished = store.get(job["id"])
    assert finished["status"] == "completed"
    assert finished["chunks"][0]["sha256"] == first_hash
    time = np.concatenate(
        [
            np.load(tmp_path / job["id"] / chunk["path"])["time_seconds"]
            for chunk in finished["chunks"]
        ]
    )
    np.testing.assert_array_equal(time, np.arange(4) / 100)


def test_budget_failure_retains_manifest_and_hash_mismatch_blocks_resume(tmp_path):
    store = ExperimentStore(tmp_path)
    job = store.submit(ExperimentSpec(budget_bytes=10))
    store.run_pending()
    assert store.get(job["id"])["status"] == "failed"
    assert "budget" in store.get(job["id"])["error"]
    job = store.submit(ExperimentSpec(duration_seconds=0.02, chunk_seconds=0.01))

    def cancel(spec, checkpoint, start, duration):
        store.cancel(job["id"])
        return synthetic_chunk(spec, checkpoint, start, duration)

    store.run_pending(cancel)
    item = store.get(job["id"])["chunks"][0]
    (tmp_path / job["id"] / item["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        store.resume(job["id"])


def test_acquire_checks_budget_hash_and_existing_data(tmp_path, monkeypatch):
    payload = b"public synthetic metadata"
    file = DatasetFile(
        "dataset_description.json",
        "https://raw.githubusercontent.com/example/data",
        hashlib.sha256(payload).hexdigest(),
        len(payload),
        "CC0",
    )
    manifest = DatasetManifest("ds005540", "1.0.7", (file,))

    class Response(io.BytesIO):
        url = file.url

    class Opener:
        def open(self, *args, **kwargs):
            return Response(payload)

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    with pytest.raises(ValueError, match="budget"):
        acquire(manifest, tmp_path, max_bytes=1)
    result = acquire(manifest, tmp_path, max_bytes=100)
    saved = Path(result["root"]) / file.path
    assert saved.read_bytes() == payload
    saved.write_bytes(b"existing data must remain")
    with pytest.raises(ValueError, match="overwrite"):
        acquire(manifest, tmp_path, max_bytes=100)
    assert saved.read_bytes() == b"existing data must remain"
    with pytest.raises(ValueError, match="relative"):
        DatasetFile("../escape", file.url, file.sha256, 1, "CC0")


def test_events_require_actual_duration_and_stimulus(tmp_path):
    path = tmp_path / "events.tsv"
    path.write_text("onset\tduration\tstim_file\n0\tn/a\tclip\n")
    with pytest.raises(ValueError):
        read_events(path)
    path.write_text("onset\tduration\tstim_file\n1.2\t13.7\tclip\n")
    event = read_events(path)[0]
    assert (event["onset"], event["duration"], event["stimulus"]) == (1.2, 13.7, "clip")


def test_train_only_text_and_scaler_and_actual_film_frames():
    encoder = ImageryEncoder(components=2).fit(
        ["a forest beside water", "a quiet mountain", "people walking nearby"]
    )
    before = dict(encoder.vectorizer.vocabulary_)
    encoder.transform(["zzzz held out narrative"])
    assert encoder.vectorizer.vocabulary_ == before
    assert "zz" not in before
    scaler = StimulusScaler().fit([[0, 2], [2, 4]])
    np.testing.assert_allclose(scaler.transform([[100, 101]]), [[99, 98]])
    times, features = film_frame_features(
        [np.zeros((2, 2, 3)), np.ones((2, 2, 3)) * 255], 2, np.ones(100), 100
    )
    np.testing.assert_allclose(times, [0, 0.5])
    np.testing.assert_allclose(features, [[1, 0, 0], [1, 1, 1]])


def test_fold_stimulus_encoder_never_fits_heldout_values(tmp_path):
    stimuli = []
    for index, value in enumerate((0, 2, 1000)):
        path = tmp_path / f"{index}.npz"
        np.savez(path, time_seconds=np.arange(3), features=np.ones((3, 2)) * value)
        stimuli.append(
            {"kind": "author_av", "path": path.name, "sha256": sha256(path), "duration_seconds": 3}
        )
    values = {"trial_index": np.arange(3)}
    builder = FoldFeatures(
        {"stimuli": stimuli, "dataset": "ds005540"},
        values,
        tmp_path,
        ExperimentSpec(),
        lambda: None,
    )
    result = builder(np.array([0, 1]), np.array([2]), "stimulus", 11)
    assert result.shape == (1, 6)
    assert np.max(np.abs(result - 0.5)) > 0.49


def test_actual_mne_preprocessing_synthetic_sinusoid(tmp_path):
    import mne

    sfreq = 256
    time = np.arange(sfreq * 12) / sfreq
    data = np.array([np.sin(2 * np.pi * 10 * time), -np.sin(2 * np.pi * 10 * time)]) * 10e-6
    raw = mne.io.RawArray(data, mne.create_info(["F3", "F4"], sfreq, "eeg"), verbose="ERROR")
    path = tmp_path / "synthetic_raw.fif"
    raw.save(path, overwrite=False, verbose="ERROR")
    result = preprocess_eeg(path, [{"onset": 2, "duration": 8}])
    assert result["qc"]["accepted"] == 1
    assert np.argmax(result["features"][0]["log_relative_power"]) == 2


def test_actual_nilearn_three_roi_extraction_and_motion_qc(tmp_path):
    import nibabel as nib
    import pandas as pd

    shape, scans = (5, 5, 5), 30
    labels = np.zeros(shape, dtype=np.int16)
    labels[0:2] = 1
    labels[2:4] = 2
    labels[4] = 3
    atlas = tmp_path / "atlas.nii.gz"
    nib.save(nib.Nifti1Image(labels, np.eye(4)), atlas)
    values = 100 + np.random.default_rng(11).normal(size=(*shape, scans))
    bold = tmp_path / "sub-synthetic_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz"
    nib.save(nib.Nifti1Image(values, np.eye(4)), bold)
    confounds = tmp_path / "confounds.tsv"
    names = [
        "trans_x",
        "trans_y",
        "trans_z",
        "rot_x",
        "rot_y",
        "rot_z",
        "csf",
        "white_matter",
        "framewise_displacement",
    ]
    table = pd.DataFrame(np.zeros((scans, len(names))), columns=names)
    table.to_csv(confounds, sep="\t", index=False)
    sidecar = tmp_path / "bold.json"
    sidecar.write_text(json.dumps({"RepetitionTime": 2.0}))
    atlases = {
        "synthetic": {
            "maps": str(atlas),
            "labels": ["Background", "Amygdala", "Hippocampus", "Insular Cortex"],
        }
    }
    result = extract_rois(bold, confounds, sidecar, atlases)
    assert result["bold"].shape == (30, 3)
    np.testing.assert_array_equal(result["time_seconds"], np.arange(30) * 2)
    table["framewise_displacement"] = 1.0
    table.to_csv(confounds, sep="\t", index=False)
    with pytest.raises(ValueError, match="QC"):
        extract_rois(bold, confounds, sidecar, atlases)


def test_bold_slopes_are_fitted_only_on_training_and_fmriprep_is_pinned(tmp_path):
    train = np.arange(12).reshape(4, 3)
    test = np.array([[30, 40, 50]])
    prediction, coefficients = fit_bold_observation(train, train * 2 + 3, test)
    np.testing.assert_allclose(prediction, test * 2 + 3)
    np.testing.assert_allclose(coefficients, [[3, 2]] * 3, atol=1e-12)
    bids = tmp_path / "bids"
    bids.mkdir()
    license_path = tmp_path / "license.txt"
    license_path.write_text("synthetic test licence")
    command = fmriprep_command(bids, tmp_path / "out", tmp_path / "work", license_path, ["01"])
    assert "--fs-no-reconall" in command
    assert "--pull=never" in command
    assert "nipreps/fmriprep:25.2.7" in command


def test_fit_checkpoints_verify_identity_and_predictions(tmp_path):
    checkpoint = EvaluationCheckpoint(tmp_path, {"cohort": "fixture"}, 100000)
    checkpoint.save(11, "mean", {"mse": 1.0, "predictions": np.arange(6).reshape(3, 2)})
    resumed = EvaluationCheckpoint(tmp_path, {"cohort": "fixture"}, 100000)
    np.testing.assert_array_equal(
        resumed.get(11, "mean")["predictions"], np.arange(6).reshape(3, 2)
    )
    with pytest.raises(ValueError, match="differs"):
        EvaluationCheckpoint(tmp_path, {"cohort": "different"}, 100000)


def test_nested_features_receive_only_disjoint_training_groups():
    p, s = np.repeat(np.arange(10), 6), np.tile(np.arange(6), 10)
    x = np.random.default_rng(11).normal(size=(60, 2))
    calls = []

    def builder(train, indices, method, seed):
        calls.append((train.copy(), indices.copy()))
        return x[indices]

    result = nested_feature_ridge(builder, "stimulus", x[:, :1], p, s)
    assert result["mse"] < 1e-3
    for train, test in calls:
        if np.array_equal(train, test):
            continue
        assert not set(p[train]) & set(p[test])
        assert not set(s[train]) & set(s[test])


def test_gain_search_caps_prespecified_settings():
    from research.spec import GAINS

    settings = [dict.fromkeys(GAINS, 0.5), dict.fromkeys(GAINS, 1.0)]
    result = calibrate_gains(settings, lambda setting: (setting["sensory_gain"] - 1) ** 2)
    assert result["gains"] == settings[1]
    with pytest.raises(ValueError, match="32"):
        calibrate_gains(settings * 17, lambda setting: 0)


def test_os_worker_lock_excludes_other_process_and_releases_after_crash(tmp_path):
    import subprocess
    import sys

    from research.locking import worker_file_lock

    path = tmp_path / "worker.lock"
    script = "from research.locking import worker_file_lock; import sys; lock=worker_file_lock(sys.argv[1]); print(lock.__enter__(), flush=True)"
    with worker_file_lock(path) as acquired:
        assert acquired
        result = subprocess.run(
            [sys.executable, "-c", script, str(path)], capture_output=True, text=True, check=True
        )
        assert result.stdout.strip() == "False"
    crash = script + "; import os; os._exit(0)"
    result = subprocess.run(
        [sys.executable, "-c", crash, str(path)], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "True"
    with worker_file_lock(path) as acquired:
        assert acquired


def test_unexpected_job_failure_is_persisted_and_next_job_runs(tmp_path):
    store = ExperimentStore(tmp_path)
    first = store.submit(ExperimentSpec(duration_seconds=0.01))
    second = store.submit(ExperimentSpec(duration_seconds=0.01))
    calls = 0

    def adapter(spec, checkpoint, start, duration):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ArithmeticError("synthetic unexpected fault")
        return synthetic_chunk(spec, checkpoint, start, duration)

    store.run_pending(adapter)
    assert sorted([store.get(first["id"])["status"], store.get(second["id"])["status"]]) == [
        "completed",
        "failed",
    ]
    assert store.active_count == 0


def test_archive_declared_shape_is_rejected_before_numpy_allocation(tmp_path, monkeypatch):
    import zipfile

    from research.store import load_arrays

    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {"descr": "<f8", "fortran_order": False, "shape": (1000000,)}
    )
    path = tmp_path / "header-only.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("values.npy", header.getvalue())
    allocation_called = False

    def allocate(*args, **kwargs):
        nonlocal allocation_called
        allocation_called = True
        raise AssertionError("allocation must not be reached")

    monkeypatch.setattr(np, "load", allocate)
    with pytest.raises(ValueError, match="budget"):
        load_arrays(path, max_bytes=256)
    assert allocation_called is False


def test_archive_payload_size_and_object_dtype_fail_before_loading(tmp_path):
    import zipfile

    from research.store import load_arrays

    path = tmp_path / "short.npz"
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {"descr": "<f8", "fortran_order": False, "shape": (2,)}
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("values.npy", header.getvalue() + b"12345678")
    with pytest.raises(ValueError, match="payload"):
        load_arrays(path)
    np.savez(path, values=np.array([{"untrusted": 1}], dtype=object))
    with pytest.raises(ValueError, match="dtype"):
        load_arrays(path)


def test_fmriprep_cancellation_stops_only_its_owned_container(tmp_path, monkeypatch):
    from research.fmri import run_fmriprep
    from research.store import Cancelled

    bids = tmp_path / "bids"
    bids.mkdir()
    licence = tmp_path / "license"
    licence.write_text("synthetic")
    command = fmriprep_command(bids, tmp_path / "out", tmp_path / "work", licence, ["01"])

    class Process:
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 1

        def wait(self, timeout):
            return self.returncode

    stopped = []
    monkeypatch.setattr("research.fmri.shutil.which", lambda name: "docker")
    monkeypatch.setattr("research.fmri.subprocess.Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr("research.fmri.subprocess.run", lambda args, **kwargs: stopped.append(args))

    def cancel():
        raise Cancelled("cancel external process")

    with pytest.raises(Cancelled):
        run_fmriprep(command, tmp_path / "run.log", cancel)
    name = command[command.index("--name") + 1]
    assert stopped == [["docker", "stop", "--time", "10", name]]


def test_bids_compressed_annotations_use_sidecar_columns_and_real_time(tmp_path):
    import gzip

    from research.datasets import annotation_series

    path = tmp_path / "synthetic_physio.tsv.gz"
    with gzip.open(path, "wt") as stream:
        stream.write("1\t2\t3\t4\n2\t3\t4\t5\n")
    sidecar = tmp_path / "synthetic_physio.json"
    sidecar.write_text(
        json.dumps(
            {"Columns": ["Good", "Bad", "Alert", "Calm"], "SamplingFrequency": 2, "StartTime": 0.5}
        )
    )
    times, values = annotation_series(path, sidecar)
    np.testing.assert_array_equal(times, [0.5, 1.0])
    np.testing.assert_array_equal(values, [[1, 2, 3, 4], [2, 3, 4, 5]])
