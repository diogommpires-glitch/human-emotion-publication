"""Explicit local research CLI. Acquisition and heavy preprocessing are separate actions."""

import argparse
import json
import sys
from pathlib import Path

from . import ExperimentSpec, ExperimentStore
from .store import atomic_json


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--store", default=".local/research", help="local experiment directory")
    commands = result.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="submit and execute a JSON experiment specification")
    run.add_argument("spec")
    commands.add_parser("list", help="list local experiment manifests")
    for name in ("status", "cancel", "resume"):
        command = commands.add_parser(name)
        command.add_argument("id")
    commands.add_parser(
        "recover", help="mark interrupted jobs once the OS worker lock is available"
    )
    acquire = commands.add_parser("acquire", help="explicitly download only pinned manifest files")
    acquire.add_argument("manifest")
    acquire.add_argument("destination")
    acquire.add_argument("--max-bytes", required=True, type=int)
    prepare = commands.add_parser(
        "prepare", help="preprocess a verified EEG/fMRI/rating source manifest"
    )
    prepare.add_argument("source")
    prepare.add_argument("destination")
    commands.add_parser(
        "prerequisites", help="report fMRI preprocessing prerequisites without installing"
    )
    fprep = commands.add_parser(
        "fmriprep", help="run the pinned container; never pull it automatically"
    )
    for name in ("bids_root", "output_root", "work_root", "license_path", "log_path"):
        fprep.add_argument(name)
    fprep.add_argument("--participants", nargs="+", required=True)
    atlas = commands.add_parser("atlas", help="explicitly acquire Harvard-Oxford atlases")
    atlas.add_argument("destination")
    atlas.add_argument("manifest")
    combine = commands.add_parser(
        "combine", help="apply Holm correction across EEG/BOLD/ratings primary tests"
    )
    combine.add_argument("results", nargs=3)
    combine.add_argument("--output", required=True)
    av = commands.add_parser(
        "av-features",
        help="extract author audiovisual embeddings with their documented sample rate",
    )
    av.add_argument("source")
    av.add_argument("output")
    av.add_argument("--visual-key", required=True)
    av.add_argument("--audio-key", required=True)
    av.add_argument("--sample-hz", required=True, type=float)
    film = commands.add_parser(
        "film-features", help="derive RMS/luminance/frame-change features from licensed CC media"
    )
    for name in ("video", "audio", "license_manifest", "output"):
        film.add_argument(name)
    return result


def combine_results(paths):
    from .statistics import holm

    results = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    by_protocol = {item["protocol"]: item for item in results}
    if set(by_protocol) != {"eeg", "fmri", "ratings"}:
        raise ValueError("combine requires exactly one EEG, fMRI and ratings result")
    families = ["eeg", "fmri", "ratings"]
    p_values = []
    for family in families:
        seed_results = by_protocol[family]["seeds"]
        primary = next(item for item in seed_results if item["seed"] == 11)
        p_values.append(primary["temporal_null"]["p_value"])
    return {
        "primary_seed": 11,
        "families": families,
        "p_values": p_values,
        "holm_adjusted": holm(p_values).tolist(),
        "empirical_validation": False,
        "interpretation": "Family-corrected temporal-null tests; rejection alone is not biological validation.",
    }


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        output = dispatch(args)
    except (ValueError, TypeError, OSError, RuntimeError, ImportError, KeyError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(output, indent=2, allow_nan=False))
    return 0


def dispatch(args):
    if args.command in {"av-features", "film-features"}:
        return write_stimulus_features(args)
    if args.command == "acquire":
        from .datasets import DatasetManifest, acquire

        return acquire(
            DatasetManifest.read(args.manifest), args.destination, max_bytes=args.max_bytes
        )
    if args.command == "prepare":
        from .preparation import prepare_cohort

        return {"manifest": str(prepare_cohort(args.source, args.destination))}
    if args.command == "prerequisites":
        from .fmri import fmriprep_prerequisites

        return fmriprep_prerequisites()
    if args.command == "fmriprep":
        from .fmri import fmriprep_command, run_fmriprep

        command = fmriprep_command(
            args.bids_root, args.output_root, args.work_root, args.license_path, args.participants
        )
        return run_fmriprep(command, args.log_path)
    if args.command == "atlas":
        from .fmri import fetch_atlases

        result = fetch_atlases(args.destination)
        if Path(args.manifest).exists():
            raise ValueError("atlas manifest already exists; choose a new output path")
        atomic_json(args.manifest, result)
        return result
    if args.command == "combine":
        result = combine_results(args.results)
        if Path(args.output).exists():
            raise ValueError("combined result already exists; choose a new output path")
        atomic_json(args.output, result)
        return result
    return dispatch_store(args)


def dispatch_store(args):
    store = ExperimentStore(args.store)
    if args.command == "run":
        spec = ExperimentSpec.from_dict(json.loads(Path(args.spec).read_text(encoding="utf-8")))
        job = store.submit(spec)
        store.run_pending()
        result = store.get(job["id"])
        if result["status"] == "failed":
            raise RuntimeError(result["error"])
        return result
    if args.command == "list":
        return store.list()
    if args.command == "status":
        return store.get(args.id)
    if args.command == "cancel":
        return store.cancel(args.id)
    if args.command == "resume":
        store.resume(args.id)
        store.run_pending()
        return store.get(args.id)
    store.recover()
    return store.list()


def write_stimulus_features(args):
    import numpy as np

    from .encoders import author_av_embeddings, read_film_features
    from .store import sha256

    if Path(args.output).exists():
        raise ValueError("stimulus output already exists; choose a new path")
    if args.command == "av-features":
        if not np.isfinite(args.sample_hz) or args.sample_hz <= 0:
            raise ValueError("author embedding sample rate must be positive finite Hz")
        features = author_av_embeddings(
            args.source, visual_key=args.visual_key, audio_key=args.audio_key
        )
        times = np.arange(len(features)) / args.sample_hz
    else:
        licence = json.loads(Path(args.license_manifest).read_text(encoding="utf-8"))
        times, features = read_film_features(args.video, args.audio, licence)
    with Path(args.output).open("xb") as stream:
        np.savez_compressed(stream, time_seconds=times, features=features)
    return {"path": args.output, "sha256": sha256(args.output), "frames": len(times)}


if __name__ == "__main__":
    sys.exit(main())
