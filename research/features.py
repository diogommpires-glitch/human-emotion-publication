"""Fit stimulus transforms within each fold before running physical-time simulations."""

import json
from pathlib import Path

import numpy as np

from .encoders import ImageryEncoder, StimulusScaler, encode_drives
from .observations import hrf_observation, log_relative_bands
from .simulation import rating_predictors, simulate_stimulus
from .spec import ExperimentSpec
from .statistics import matrix
from .store import load_arrays, sha256


class FoldFeatures:
    def __init__(self, manifest, arrays, root, spec, cancel):
        self.manifest, self.arrays, self.root = manifest, arrays, Path(root)
        self.spec, self.cancel = spec, cancel
        self.stimuli = manifest["stimuli"]
        self.cache = {}
        self.gains = None
        self.raw: list[tuple[np.ndarray, np.ndarray] | None] = [
            self._read_stimulus(item) for item in self.stimuli
        ]

    def _read_stimulus(self, item):
        if item["kind"] == "imagery":
            if not isinstance(item["narrative"], str) or not item["narrative"].strip():
                raise ValueError(
                    "imagery needs the actual guidance narrative, never its emotion category"
                )
            return None
        path = (self.root / item["path"]).resolve()
        if not path.is_relative_to(self.root.resolve()) or sha256(path) != item["sha256"]:
            raise ValueError("stimulus features must match their local manifest hash")
        data = load_arrays(path, self.spec.budget_bytes)
        times, features = (
            np.asarray(data["time_seconds"]),
            matrix(data["features"], "stimulus features"),
        )
        if (
            len(times) != len(features)
            or not np.isfinite(times).all()
            or (np.diff(times) <= 0).any()
            or times[0] != 0
        ):
            raise ValueError(
                "stimulus frames must have actual increasing timestamps starting at zero"
            )
        if item["kind"] not in {"author_av", "cc_film"}:
            raise ValueError("stimulus kind must be imagery, author_av or cc_film")
        if item["kind"] == "cc_film" and not item.get("media_license"):
            raise ValueError("film feature provenance requires the individual media licence")
        return times, features

    def __call__(self, train, indices, method, seed):
        if method == "conditions":
            return matrix(self.arrays["conditions"][indices], "conditions")
        if method == "simulator" and self.manifest.get("learning_rule", "none") == "none":
            method = "frozen_simulator"
        key = (tuple(train.tolist()), method, seed, json.dumps(self.gains, sort_keys=True))
        if key not in self.cache:
            self.cache[key] = self._build(train, method, seed)
        return self.cache[key][indices]

    def _fit_encoders(self, training_trials, seed):
        scalers, imagery = {}, None
        kinds = np.array([item["kind"] for item in self.stimuli])
        for kind in np.unique(kinds):
            selected = training_trials[kinds[training_trials] == kind]
            if len(selected) == 0:
                raise ValueError(
                    f"training fold contains no {kind} stimuli; change the prespecified family split"
                )
            if kind == "imagery":
                narratives = [self.stimuli[index]["narrative"] for index in selected]
                imagery = ImageryEncoder(seed=seed).fit(narratives)
                features = imagery.transform(narratives)
            else:
                features = np.concatenate(
                    [raw[1] for index in selected if (raw := self.raw[index]) is not None]
                )
            scalers[kind] = StimulusScaler().fit(features)
        return scalers, imagery

    def _drives(self, index, scalers, imagery):
        item = self.stimuli[index]
        kind, duration = item["kind"], float(item["duration_seconds"])
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("actual stimulus duration must be positive finite seconds")
        if kind == "imagery":
            features = imagery.transform([item["narrative"]])
            times = np.array([0.0])
        else:
            raw = self.raw[index]
            assert raw is not None
            times, features = raw
        projected = encode_drives(scalers[kind].transform(features))
        levels = np.array([list(row.values()) for row in projected])
        channels = tuple(projected[0])
        seconds = np.arange(int(np.ceil(duration)))
        interpolated = np.stack(
            [np.interp(seconds, times, levels[:, column]) for column in range(6)], axis=1
        )
        drives = [dict(zip(channels, row.tolist(), strict=True)) for row in interpolated]
        if self.gains is not None:
            from .encoders import apply_gains

            drives = apply_gains(drives, self.gains)
        return drives, interpolated, duration

    def _direct_features(self, interpolated, rows):
        if self.manifest["dataset"] in {"ds004892", "ds004872"}:
            times = np.arange(len(interpolated))
            return np.stack(
                [
                    np.interp(self.arrays["sample_seconds"][rows], times, interpolated[:, column])
                    for column in range(6)
                ],
                axis=1,
            )
        return np.repeat(interpolated.mean(axis=0)[None, :], len(rows), axis=0)

    def _build(self, train, method, seed):
        trial_ids = np.asarray(self.arrays["trial_index"], dtype=int)
        scalers, imagery = self._fit_encoders(np.unique(trial_ids[train]), seed)
        output = None
        for index in range(len(self.stimuli)):
            self.cancel()
            drives, interpolated, duration = self._drives(index, scalers, imagery)
            rows = np.flatnonzero(trial_ids == index)
            if method == "stimulus":
                predictors = self._direct_features(interpolated, rows)
            else:
                simulation_spec = ExperimentSpec(
                    protocol="mechanics",
                    seed=seed,
                    duration_seconds=duration,
                    chunk_seconds=self.spec.chunk_seconds,
                    neurons=self.spec.neurons,
                )
                recording = simulate_stimulus(
                    simulation_spec,
                    drives,
                    frozen=method == "frozen_simulator"
                    or self.manifest.get("learning_rule", "none") == "none",
                    randomised_graph=method == "randomised_graph",
                    cancel=self.cancel,
                )
                predictors = self._observe(recording, rows)
            if output is None:
                output = np.empty((len(trial_ids), predictors.shape[1]))
            output[rows] = predictors
        return output

    def _observe(self, recording, rows) -> np.ndarray:
        if self.spec.protocol == "eeg":
            times = recording["field_time_seconds"]
            bands = log_relative_bands(
                recording["field_potential"].T, 1 / np.median(np.diff(times))
            ).mean(axis=0)
            return np.repeat(bands[None, :], len(rows), axis=0)
        if self.spec.protocol == "fmri":
            return self._bold_predictors(recording, rows)
        if self.manifest["dataset"] == "ds004872":
            return self._continuous_ratings(recording, rows)
        features = rating_predictors(recording)
        return np.repeat(features[None, :], len(rows), axis=0)

    def _bold_predictors(self, recording, rows) -> np.ndarray:
        from brain_model.config import NETWORK

        names = [row["id"] for row in NETWORK["regions"]]
        groups = (
            [i for i, name in enumerate(names) if name.startswith("amygdala")],
            [names.index("hippocampus")],
            [names.index("insula")],
        )
        drive = np.stack(
            [recording["neural_drive"][:, group].sum(axis=1) for group in groups], axis=1
        )
        times = recording["time_seconds"]
        interval = float(np.median(np.diff(times)))
        # Recorded bins represent interval averages; prepend the pre-recording zero drive.
        drive = np.vstack([np.zeros((1, 3)), drive])
        return hrf_observation(drive, interval, self.arrays["sample_seconds"][rows])

    def _continuous_ratings(self, recording, rows) -> np.ndarray:
        times = recording["time_seconds"]
        predictions = []
        for sample in self.arrays["sample_seconds"][rows]:
            selected = (times > max(0, sample - 1)) & (times <= sample)
            if not selected.any():
                raise ValueError(
                    "film rating precedes the first complete simulator observation bin"
                )
            window = {
                name: recording[name][selected]
                for name in ("projection_hz", "neural_drive", "occupancy")
            }
            predictions.append(rating_predictors(window))
        return np.asarray(predictions)
