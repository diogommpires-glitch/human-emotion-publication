"""Prepare observations from acquired files without inventing ratings or timings."""

import json
from pathlib import Path

import numpy as np

from .datasets import DatasetManifest, annotation_series, read_events, verify_local
from .fmri import extract_rois
from .observations import preprocess_eeg
from .spec import DATASETS
from .store import atomic_json, load_arrays, sha256


def eeg_observation(record, stimulus, acquired_path):
    events = read_events(acquired_path(record["events"]), stimulus_column=record["stimulus_column"])
    event = events[int(record["event_index"])]
    validate_event(event, stimulus)
    result = preprocess_eeg(
        acquired_path(record["raw"]), [event], bad_channels=record.get("bad_channels", [])
    )
    if not result["features"]:
        return [], [], [], result["qc"]
    return [result["features"][0]["log_relative_power"]], [event["onset"]], [0.0], result["qc"]


def validate_event(event, stimulus):
    if (
        event["stimulus"] != stimulus["id"]
        or abs(event["duration"] - stimulus["duration_seconds"]) > 1e-6
    ):
        raise ValueError("event identity/duration differs from the actual stimulus metadata")


def fmri_observation(record, stimulus, source, source_root):
    names = ("bold", "confounds", "sidecar")
    paths = [(source_root / record[name]).resolve() for name in names]
    if any(
        sha256(path) != record[name + "_sha256"] for name, path in zip(names, paths, strict=True)
    ):
        raise ValueError("fMRIPrep derivative content differs from recorded provenance")
    result = extract_rois(paths[0], paths[1], paths[2], source["atlases"])
    actual_times = result["time_seconds"]
    sample_times = actual_times - float(record["stimulus_onset_seconds"])
    keep = (sample_times >= 0) & (sample_times < stimulus["duration_seconds"])
    return result["bold"][keep], actual_times[keep], sample_times[keep], result["qc"]


def rating_observation(record, stimulus, source, acquired_path):
    columns = source["rating_columns"]
    if source["dataset"] == "ds005540":
        events = read_events(
            acquired_path(record["events"]), stimulus_column=record["stimulus_column"]
        )
        event = events[int(record["event_index"])]
        validate_event(event, stimulus)
        values = [[float(event["source"][columns[name]]) for name in ("valence", "arousal")]]
        return values, [event["onset"]], [0.0], {"accepted": 1}
    if source["dataset"] != "ds004872":
        raise ValueError("ratings must use EEG participants or the separate film annotation cohort")
    actual_times, values = annotation_series(
        acquired_path(record["annotations"]),
        acquired_path(record["sidecar"]),
        tuple(columns[name] for name in ("Good", "Bad", "Alert", "Calm")),
    )
    sample_times = actual_times - float(record["stimulus_onset_seconds"])
    keep = (sample_times >= 0.01) & (sample_times < stimulus["duration_seconds"])
    return (
        values[keep],
        actual_times[keep],
        sample_times[keep],
        {"accepted": int(keep.sum()), "outside_recorded_stimulus": int((~keep).sum())},
    )


def prepare_stimuli(stimuli, source_root, output):
    metadata = []
    for index, stimulus in enumerate(stimuli):
        stimulus = dict(stimulus)
        if stimulus["kind"] != "imagery":
            path = (source_root / stimulus["path"]).resolve()
            if sha256(path) != stimulus["sha256"]:
                raise ValueError("stimulus derived features fail manifest hash verification")
            target = output / f"stimulus-{index:04d}.npz"
            features = load_arrays(path)
            np.savez_compressed(
                target, time_seconds=features["time_seconds"], features=features["features"]
            )
            stimulus["path"], stimulus["sha256"] = target.name, sha256(target)
        metadata.append(stimulus)
    return metadata


def collect_observations(source, source_root, acquired_path, cancel):
    rows, qc = [], []
    for record in source["records"]:
        cancel()
        trial = int(record["stimulus_index"])
        stimulus = source["stimuli"][trial]
        protocol = source["protocol"]
        if protocol == "eeg":
            values, actual_times, sample_times, quality = eeg_observation(
                record, stimulus, acquired_path
            )
        elif protocol == "fmri":
            values, actual_times, sample_times, quality = fmri_observation(
                record, stimulus, source, source_root
            )
        elif protocol == "ratings":
            values, actual_times, sample_times, quality = rating_observation(
                record, stimulus, source, acquired_path
            )
        else:
            raise ValueError("preparation protocol must be eeg, fmri or ratings")
        qc.append(quality)
        if not np.isfinite(values).all():
            raise ValueError("missing observations must be excluded explicitly, never invented")
        family = stimulus.get("family", stimulus["id"])
        for value, actual, sample in zip(values, actual_times, sample_times, strict=True):
            rows.append(
                {
                    "outcomes": value,
                    "participant": record["participant"],
                    "stimulus_group": family,
                    "trial_index": trial,
                    "series": record["participant"] + ":" + record["session"],
                    "time_seconds": actual,
                    "sample_seconds": sample,
                    "conditions": record["condition_features"],
                }
            )
    if not rows:
        raise ValueError("no usable observations remain after QC")
    return {name: np.asarray([row[name] for row in rows]) for name in rows[0]}, qc


def prepare_cohort(source_path, destination, cancel=lambda: None):
    source_path = Path(source_path).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    dataset = source["dataset"]
    if dataset not in DATASETS or source["version"] != DATASETS[dataset]:
        raise ValueError("preparation requires a pinned dataset release")
    data_root = (source_path.parent / source["data_root"]).resolve()
    acquisition = DatasetManifest.read(source_path.parent / source["dataset_manifest"])
    if acquisition.dataset != dataset:
        raise ValueError("acquisition manifest and observation source identify different datasets")
    verify_local(acquisition, data_root)
    allowed_files = {str((data_root / item.path).resolve()) for item in acquisition.files}

    def acquired_path(relative):
        path = (data_root / relative).resolve()
        if str(path) not in allowed_files:
            raise ValueError("observation input is absent from the verified acquisition manifest")
        return path

    arrays, qc = collect_observations(source, source_path.parent, acquired_path, cancel)
    output = Path(destination).resolve()
    output.mkdir(parents=True, exist_ok=False)
    arrays_path = output / "cohort.npz"
    np.savez_compressed(
        arrays_path,
        outcomes=arrays["outcomes"],
        participant=arrays["participant"],
        stimulus_group=arrays["stimulus_group"],
        series=arrays["series"],
        trial_index=arrays["trial_index"],
        time_seconds=arrays["time_seconds"],
        sample_seconds=arrays["sample_seconds"],
        conditions=arrays["conditions"],
    )
    manifest = {
        "schema_version": 1,
        "protocol": source["protocol"],
        "dataset": dataset,
        "version": source["version"],
        "arrays": arrays_path.name,
        "sha256": sha256(arrays_path),
        "stimuli": prepare_stimuli(source["stimuli"], source_path.parent, output),
        "target_names": source["target_names"],
        "learning_rule": source.get("learning_rule", "none"),
        "stimulus_encoding": "fold-fitted TF-IDF/SVD or author AV/CC film features; fixed signed projection",
        "provenance": {
            "source_sha256": sha256(source_path),
            "acquisition": acquisition.to_dict(),
            "qc": qc,
        },
    }
    if source.get("calibration_settings"):
        manifest["calibration_settings"] = source["calibration_settings"]
    atomic_json(output / "cohort.json", manifest)
    return output / "cohort.json"
