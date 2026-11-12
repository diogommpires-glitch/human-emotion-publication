"""Pinned data provenance and explicit bounded acquisition."""

import csv
import gzip
import json
import re
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from .spec import DATASETS
from .store import atomic_json, sha256


def validate_public_url(url):
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.username or parsed.password:
        raise ValueError("dataset URL must use HTTPS without credentials")
    allowed = {
        "raw.githubusercontent.com",
        "github.com",
        "openneuro.org",
        "openneuro.s3.amazonaws.com",
        "s3.amazonaws.com",
        "objects.githubusercontent.com",
        "media.githubusercontent.com",
    }
    if parsed.hostname not in allowed:
        raise ValueError("download host must be an approved public OpenNeuro/GitHub source")


class PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


@dataclass(frozen=True)
class DatasetFile:
    path: str
    url: str
    sha256: str
    size_bytes: int
    license: str
    role: str = "data"

    def __post_init__(self):
        path = PurePosixPath(self.path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in self.path
            or ":" in self.path
            or not path.parts
        ):
            raise ValueError("dataset paths must be relative and stay within the dataset directory")
        if not re.fullmatch("[a-f0-9]{64}", self.sha256):
            raise ValueError("every acquired file requires an independently recorded SHA-256")
        if not isinstance(self.size_bytes, int) or self.size_bytes < 1:
            raise ValueError("file size must be positive")
        validate_public_url(self.url)
        if self.role == "media" and self.license not in {
            "CC0",
            "CC-BY-3.0",
            "CC-BY-4.0",
            "CC-BY-SA-3.0",
            "CC-BY-SA-4.0",
        }:
            raise ValueError("film media requires an explicit supported Creative Commons licence")
        if not self.license:
            raise ValueError("file licence must be recorded")


@dataclass(frozen=True)
class DatasetManifest:
    dataset: str
    version: str
    files: tuple[DatasetFile, ...]
    schema_version: int = 1
    license: str = "CC0"

    def __post_init__(self):
        if self.dataset not in DATASETS or DATASETS[self.dataset] != self.version:
            raise ValueError(f"dataset release must match {DATASETS}")
        if self.schema_version != 1 or self.license != "CC0":
            raise ValueError("dataset metadata requires schema 1 and CC0")
        if len({file.path for file in self.files}) != len(self.files):
            raise ValueError("duplicate file paths in dataset manifest")

    @classmethod
    def read(cls, path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        value["files"] = tuple(DatasetFile(**item) for item in value["files"])
        return cls(**value)

    def to_dict(self):
        return asdict(self)


def acquire(manifest, destination, *, max_bytes, cancel=lambda: None):
    if sum(item.size_bytes for item in manifest.files) > max_bytes:
        raise ValueError("manifest exceeds explicit download byte budget")
    root = Path(destination).resolve() / manifest.dataset / manifest.version
    root.mkdir(parents=True, exist_ok=True)
    acquired = []
    for item in manifest.files:
        cancel()
        target = root.joinpath(*PurePosixPath(item.path).parts).resolve()
        if not target.is_relative_to(root.resolve()):
            raise ValueError("dataset destination escapes through a symlink")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.stat().st_size != item.size_bytes or sha256(target) != item.sha256:
                raise ValueError("existing dataset file differs; refusing to overwrite")
            acquired.append(item.path)
            continue
        acquire_file(item, target, cancel)
        acquired.append(item.path)
    receipt = root / ("acquisition-" + uuid.uuid4().hex + ".json")
    atomic_json(receipt, {"manifest": manifest.to_dict(), "acquired": acquired})
    return {"root": str(root), "receipt": str(receipt), "files": len(acquired)}


def verify_local(manifest, root):
    root = Path(root).resolve()
    for item in manifest.files:
        path = (root / item.path).resolve()
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or path.stat().st_size != item.size_bytes
            or sha256(path) != item.sha256
        ):
            raise ValueError(f"dataset manifest verification failed for {item.path}")


def read_events(path, *, stimulus_column="stim_file", condition_column="trial_type"):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    events = []
    for row in rows:
        onset, duration = float(row["onset"]), float(row["duration"])
        if not 0 <= onset < float("inf") or not 0 < duration < float("inf"):
            raise ValueError(
                "BIDS onset and duration must be finite seconds; missing timings cannot be guessed"
            )
        stimulus = row.get(stimulus_column)
        if not stimulus or stimulus == "n/a":
            raise ValueError(f"event requires actual stimulus identity in {stimulus_column}")
        events.append(
            {
                "onset": onset,
                "duration": duration,
                "stimulus": stimulus,
                "condition": row.get(condition_column),
                "source": row,
            }
        )
    return events


def eeg_records(root):
    root = Path(root)
    records = []
    for path in sorted(root.glob("sub-*/ses-*/eeg/*_events.tsv")):
        stem = path.name.removesuffix("_events.tsv")
        recordings = [
            item
            for suffix in (".edf", ".bdf", ".vhdr", ".set", ".fif")
            for item in path.parent.glob(stem + "_eeg" + suffix)
        ]
        if len(recordings) != 1:
            raise ValueError("each EEG events file requires exactly one supported raw recording")
        records.append(
            {
                "participant": path.parts[-4],
                "session": path.parts[-3],
                "recording": str(recordings[0]),
                "events": str(path),
            }
        )
    if not records:
        raise ValueError("no BIDS EEG recordings found; check ds005540 release layout")
    return records


def annotation_series(path, sidecar, columns=("Good", "Bad", "Alert", "Calm")):
    import numpy as np

    metadata = json.loads(Path(sidecar).read_text(encoding="utf-8"))
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        column_names = metadata.get("Columns")
        if column_names is None:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        else:
            rows = list(csv.DictReader(stream, fieldnames=column_names, delimiter="\t"))
    values = np.array([[float(row[key]) for key in columns] for row in rows])
    if not np.isfinite(values).all():
        raise ValueError("annotations contain missing or non-finite ratings")
    frequency = float(metadata["SamplingFrequency"])
    start = float(metadata["StartTime"])
    if frequency <= 0 or not np.isfinite([frequency, start]).all():
        raise ValueError(
            "annotation sidecar requires finite physical sampling frequency and start time"
        )
    return start + np.arange(len(values)) / frequency, values


def acquire_file(item, target, cancel):
    temporary = target.with_name(target.name + ".partial-" + uuid.uuid4().hex)
    opener = urllib.request.build_opener(PublicRedirect())
    try:
        with (
            opener.open(item.url, timeout=30) as response,
            temporary.open("xb") as stream,
        ):
            validate_public_url(response.url)
            count = 0
            while block := response.read(min(1024 * 1024, item.size_bytes - count + 1)):
                cancel()
                count += len(block)
                if count > item.size_bytes:
                    raise ValueError("download exceeds declared file size")
                stream.write(block)
        if temporary.stat().st_size != item.size_bytes or sha256(temporary) != item.sha256:
            raise ValueError("download size or SHA-256 differs from the pinned manifest")
        # Hard link publishes without overwriting a concurrent acquisition.
        target.hardlink_to(temporary)
    finally:
        temporary.unlink(missing_ok=True)
