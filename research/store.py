"""Atomic manifests and bounded numeric chunks for one local worker."""

import hashlib
import json
import logging
import os
import threading
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np

from .locking import worker_file_lock
from .spec import ExperimentSpec


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_member_bytes(archive, member, max_bytes):
    with archive.open(member) as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("numeric archive requires NPY version 1 or 2")
        header_bytes = stream.tell()
    if (
        dtype.hasobject
        or dtype.fields is not None
        or dtype.kind not in "biufSU"
        or dtype.itemsize <= 0
    ):
        raise ValueError("numeric archive contains an unsupported dtype")
    elements = 1
    for dimension in shape:
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or not 0 <= dimension <= np.iinfo(np.intp).max
        ):
            raise ValueError("numeric archive shape contains an invalid dimension")
        elements *= dimension
        if elements * dtype.itemsize > max_bytes:
            raise ValueError("numeric array shape exceeds its byte budget")
    payload_bytes = elements * dtype.itemsize
    if payload_bytes != member.file_size - header_bytes:
        raise ValueError("numeric array shape and actual payload length disagree")
    return payload_bytes


def load_arrays(path, max_bytes=8 * 1024**3):
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len({item.filename for item in members}) != len(members):
            raise ValueError("numeric archive has duplicate array entries")
        if sum(item.file_size for item in members) > max_bytes:
            raise ValueError("numeric archive exceeds its uncompressed byte budget")
        total = 0
        for member in members:
            if not member.filename.endswith(".npy"):
                raise ValueError("numeric archive contains a non-array entry")
            total += array_member_bytes(archive, member, max_bytes - total)
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


class Cancelled(RuntimeError):
    pass


class ExperimentStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._worker_lock = threading.Lock()
        self._active = 0

    @property
    def active_count(self):
        return self._active + sum(item["status"] == "queued" for item in self.list())

    def _path(self, job_id):
        if (
            not isinstance(job_id, str)
            or len(job_id) != 32
            or any(c not in "0123456789abcdef" for c in job_id)
        ):
            raise ValueError("experiment id must be 32 lowercase hexadecimal characters")
        return self.root / job_id / "manifest.json"

    def submit(self, spec):
        if isinstance(spec, dict):
            spec = ExperimentSpec.from_dict(spec)
        if not isinstance(spec, ExperimentSpec):
            raise TypeError("expected ExperimentSpec or JSON object")
        job_id = uuid.uuid4().hex
        job = {
            "schema_version": 1,
            "id": job_id,
            "spec": spec.to_dict(),
            "status": "queued",
            "created_at": time.time(),
            "updated_at": time.time(),
            "progress": 0.0,
            "completed_seconds": 0.0,
            "chunks": [],
            "cancel_requested": False,
            "empirical_validation": False,
            "error": None,
        }
        with self._lock:
            self._path(job_id).parent.mkdir()
            atomic_json(self._path(job_id), job)
        return job

    def get(self, job_id):
        with self._lock:
            return json.loads(self._path(job_id).read_text(encoding="utf-8"))

    def list(self):
        with self._lock:
            return [
                self.get(path.parent.name) for path in sorted(self.root.glob("*/manifest.json"))
            ]

    def _save(self, job):
        job["updated_at"] = time.time()
        atomic_json(self._path(job["id"]), job)
        return job

    def cancel(self, job_id):
        with self._lock:
            job = self.get(job_id)
            if job["status"] in {"queued", "running"}:
                job["cancel_requested"] = True
                if job["status"] == "queued":
                    job["status"] = "cancelled"
                self._save(job)
            return job

    def resume(self, job_id):
        with self._lock:
            job = self.get(job_id)
            if job["status"] not in {"cancelled", "failed", "interrupted"}:
                raise ValueError("only cancelled, failed or interrupted jobs can resume")
            self._verify_chunks(job)
            job.update(status="queued", cancel_requested=False, error=None)
            return self._save(job)

    def recover(self):
        with self._worker_lock, worker_file_lock(self.root / ".worker.lock") as acquired:
            if not acquired:
                raise RuntimeError(
                    "cannot recover while another process owns the experiment worker"
                )
            with self._lock:
                for job in self.list():
                    if job["status"] == "running":
                        job["status"] = "interrupted"
                        self._save(job)

    def _verify_chunks(self, job):
        for chunk in job["chunks"]:
            directory = self._path(job["id"]).parent.resolve()
            path = (directory / chunk["path"]).resolve()
            if not path.is_relative_to(directory):
                raise ValueError("chunk path escapes its experiment directory")
            if sha256(path) != chunk["sha256"]:
                raise ValueError("recorded chunk hash mismatch; refusing resume")

    def check_cancelled(self, job_id):
        if self.get(job_id)["cancel_requested"]:
            raise Cancelled("cancelled at a chunk or dataset boundary")

    def run_pending(self, adapter=None):
        if not self._worker_lock.acquire(blocking=False):
            return
        try:
            with worker_file_lock(self.root / ".worker.lock") as acquired:
                if not acquired:
                    return
                for job in self.list():
                    if job["status"] == "queued":
                        self._execute(job["id"], adapter)
        finally:
            self._worker_lock.release()

    def _execute(self, job_id, adapter):
        with self._lock:
            job = self.get(job_id)
            if job["cancel_requested"]:
                return
            job["status"] = "running"
            self._save(job)
            self._active += 1
        try:
            spec = ExperimentSpec.from_dict(job["spec"])
            if spec.protocol in {"synthetic", "mechanics"}:
                self._record(job_id, spec, adapter)
            elif spec.protocol == "session_order":
                from .simulation import evaluate_session_order

                result = evaluate_session_order(spec, lambda: self.check_cancelled(job_id))
                atomic_json(self._path(job_id).parent / "results.json", result)
            else:
                from .pipeline import evaluate_cohort

                result = evaluate_cohort(
                    spec,
                    lambda: self.check_cancelled(job_id),
                    checkpoint_dir=self._path(job_id).parent,
                    progress=lambda fraction, stage: self._progress(job_id, fraction, stage),
                )
                atomic_json(self._path(job_id).parent / "results.json", result)
            self.check_cancelled(job_id)
            with self._lock:
                job = self.get(job_id)
                job.update(status="completed", progress=1.0)
                self._save(job)
        except Cancelled:
            with self._lock:
                job = self.get(job_id)
                job["status"] = "cancelled"
                self._save(job)
        except Exception as exc:
            # A failed job must be visible and must not kill the single queue worker.
            logging.getLogger(__name__).exception("Research experiment %s failed", job_id)
            with self._lock:
                job = self.get(job_id)
                job.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                self._save(job)
        finally:
            self._active -= 1

    def _progress(self, job_id, fraction, stage):
        with self._lock:
            job = self.get(job_id)
            job.update(progress=fraction, stage=stage)
            self._save(job)

    def _record(self, job_id, spec, adapter):
        from .simulation import SimulatorAdapter, synthetic_chunk

        if adapter is None:
            adapter = synthetic_chunk if spec.protocol == "synthetic" else SimulatorAdapter(spec)
        job = self.get(job_id)
        self._verify_chunks(job)
        checkpoint = None
        if job["chunks"]:
            last = self._path(job_id).parent / job["chunks"][-1]["path"]
            archive = load_arrays(last, spec.budget_bytes)
            checkpoint = {
                name.removeprefix("state__"): value
                for name, value in archive.items()
                if name.startswith("state__")
            }
        while job["completed_seconds"] < spec.duration_seconds - 1e-12:
            self.check_cancelled(job_id)
            start = job["completed_seconds"]
            duration = min(spec.chunk_seconds, spec.duration_seconds - start)
            arrays, checkpoint = adapter(spec, checkpoint, start, duration)
            payload = dict(arrays)
            payload.update({"state__" + key: value for key, value in checkpoint.items()})
            name, destination = self._publish_chunk(job, spec, payload)
            with self._lock:
                job = self.get(job_id)
                job["chunks"].append(
                    {
                        "path": name,
                        "sha256": sha256(destination),
                        "start_seconds": start,
                        "duration_seconds": duration,
                    }
                )
                job.update(
                    completed_seconds=start + duration,
                    progress=(start + duration) / spec.duration_seconds,
                )
                self._save(job)

    def _publish_chunk(self, job, spec, payload):
        for value in payload.values():
            value = np.asarray(value)
            if value.dtype.kind not in "biufSU" or (
                value.dtype.kind == "f" and not np.isfinite(value).all()
            ):
                raise ValueError("chunk values must be finite numeric arrays or checkpoint text")
        directory = self._path(job["id"]).parent
        used = sum(path.stat().st_size for path in directory.iterdir() if path.is_file())
        needed = sum(np.asarray(value).nbytes for value in payload.values()) + 65536
        if used + needed > spec.budget_bytes:
            raise ValueError("experiment disk budget exceeded; existing results retained")
        name = f"chunk-{len(job['chunks']):06d}-{uuid.uuid4().hex}.npz"
        destination = directory / name
        temporary = destination.with_suffix(".tmp")
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        return name, destination
