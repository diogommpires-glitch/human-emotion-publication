"""Pinned fMRIPrep invocation and whole-region Harvard-Oxford observations."""

import json
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np

FMRIPREP_IMAGE = "nipreps/fmriprep:25.2.7"


def fmriprep_command(bids_root, output_root, work_root, license_path, participants):
    paths = [Path(item).resolve() for item in (bids_root, output_root, work_root, license_path)]
    if not paths[0].is_dir() or not paths[3].is_file():
        raise ValueError("fMRIPrep requires an existing BIDS root and FreeSurfer licence file")
    if not participants or any(not str(item).isalnum() for item in participants):
        raise ValueError("participant labels must be nonempty BIDS alphanumeric identifiers")
    if paths[0] == paths[1] or paths[0] == paths[2]:
        raise ValueError("fMRIPrep input, derivatives and work directories must be distinct")
    return [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--name",
        "research-fmriprep-" + uuid.uuid4().hex,
        "--network=none",
        "-v",
        f"{paths[0]}:/data:ro",
        "-v",
        f"{paths[1]}:/out",
        "-v",
        f"{paths[2]}:/work",
        "-v",
        f"{paths[3]}:/license.txt:ro",
        FMRIPREP_IMAGE,
        "/data",
        "/out",
        "participant",
        "--participant-label",
        *participants,
        "--fs-license-file",
        "/license.txt",
        "--fs-no-reconall",
        "--output-spaces",
        "MNI152NLin6Asym:res-2",
        "--notrack",
        "--nthreads",
        "4",
        "--omp-nthreads",
        "2",
        "--mem-mb",
        "8192",
        "-w",
        "/work",
    ]


def fmriprep_prerequisites():
    docker = shutil.which("docker")
    return {
        "image": FMRIPREP_IMAGE,
        "docker_available": docker is not None,
        "requirements": [
            "Docker with Linux containers",
            "explicitly pulled pinned image",
            "FreeSurfer licence file",
            "TemplateFlow templates pre-cached in container",
            "BIDS T1w and BOLD files",
            "at least 8 GiB RAM and dataset-specific free disk",
        ],
        "ready": docker is not None,
        "note": "WSL alone is insufficient. No image pull, system install or full subject run is automatic.",
    }


def run_fmriprep(command, log_path, cancel=lambda: None):
    if not shutil.which("docker"):
        raise RuntimeError(
            "fMRIPrep blocked: Docker Linux containers are unavailable; WSL alone is insufficient"
        )
    if command[:3] != ["docker", "run", "--rm"] or FMRIPREP_IMAGE not in command:
        raise ValueError("use the pinned fmriprep_command builder")
    if "--name" not in command:
        raise ValueError("fMRIPrep command needs an owned container name for cancellation")
    name = command[command.index("--name") + 1]
    if not re.fullmatch(r"research-fmriprep-[a-f0-9]{32}", name):
        raise ValueError("invalid research container name")
    with Path(log_path).open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, shell=False)
        try:
            while process.poll() is None:
                cancel()
                time.sleep(0.2)
        finally:
            if process.poll() is None:
                subprocess.run(
                    ["docker", "stop", "--time", "10", name],
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
                process.terminate()
                process.wait(timeout=30)
    if process.returncode:
        raise RuntimeError(
            f"fMRIPrep exited with status {process.returncode}; inspect the local log"
        )
    return {"returncode": process.returncode, "image": FMRIPREP_IMAGE, "log": str(log_path)}


def fetch_atlases(destination):
    from nilearn.datasets import fetch_atlas_harvard_oxford

    result = {}
    for name in ("cort-maxprob-thr25-2mm", "sub-maxprob-thr25-2mm"):
        atlas = fetch_atlas_harvard_oxford(name, data_dir=str(destination))
        result[name] = {"maps": str(atlas.maps), "labels": list(atlas.labels)}
    return result


def extract_rois(
    bold_path,
    confounds_path,
    sidecar_path,
    atlases,
    *,
    fd_threshold_mm=0.5,
    max_censored_fraction=0.2,
):
    import nibabel as nib
    from nilearn.image import new_img_like
    from nilearn.maskers import NiftiLabelsMasker

    metadata = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    tr = float(metadata["RepetitionTime"])
    if not np.isfinite(tr) or tr <= 0:
        raise ValueError("BOLD metadata must contain positive RepetitionTime in seconds")
    if "space-MNI152NLin6Asym" not in Path(bold_path).name:
        raise ValueError(
            "Harvard-Oxford extraction requires the pinned MNI152NLin6Asym output space"
        )
    bold = nib.load(bold_path)
    if not isinstance(bold, nib.Nifti1Image):
        raise TypeError("BOLD data must be a NIfTI-1 image")
    if len(bold.shape) != 4:
        raise ValueError("BOLD data must contain a time dimension")
    nuisance, keep = motion_confounds(
        confounds_path, bold.shape[3], fd_threshold_mm, max_censored_fraction
    )
    label_volume = whole_region_labels(bold, atlases)
    roi_names = ("amygdala", "hippocampus", "insula")
    labels = new_img_like(bold, label_volume)
    masker = NiftiLabelsMasker(
        labels_img=labels,
        standardize=None,
        detrend=True,
        t_r=tr,
        standardize_confounds=True,
        resampling_target="data",
    )
    signals = masker.fit_transform(bold, confounds=nuisance, sample_mask=np.flatnonzero(keep))
    return {
        "time_seconds": np.arange(len(keep))[keep] * tr,
        "bold": signals,
        "roi_names": roi_names,
        "qc": {
            "tr_seconds": tr,
            "retained_volumes": int(keep.sum()),
            "censored_volumes": int((~keep).sum()),
            "fd_threshold_mm": fd_threshold_mm,
            "whole_regions_only": True,
        },
    }


def motion_confounds(confounds_path, volumes, fd_threshold_mm, max_censored_fraction):
    import pandas as pd

    confounds = pd.read_csv(confounds_path, sep="\t")
    if volumes != len(confounds):
        raise ValueError("BOLD volumes and confound rows must align")
    names = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z", "csf", "white_matter"]
    if not {*names, "framewise_displacement"} <= set(confounds.columns):
        raise ValueError("fMRIPrep motion, tissue and displacement confounds are required")
    fd = confounds["framewise_displacement"].to_numpy(dtype=float)
    if np.isnan(fd[0]):
        fd[0] = 0.0  # first displacement is undefined relative to a preceding volume
    if not np.isfinite(fd).all():
        raise ValueError("unexpected missing framewise displacement beyond the first volume")
    keep = fd < fd_threshold_mm
    for name in confounds.columns:
        if name.startswith("non_steady_state_outlier"):
            keep &= confounds[name].to_numpy() == 0
    if 1 - keep.mean() > max_censored_fraction or keep.sum() < 10:
        raise ValueError("BOLD run fails prespecified motion/usable-volume QC")
    nuisance = confounds[names].to_numpy(dtype=float)
    if not np.isfinite(nuisance).all():
        raise ValueError("BOLD nuisance confounds contain non-finite values")
    return nuisance, keep


def whole_region_labels(bold, atlases):
    import nibabel as nib
    from nilearn.image import resample_to_img

    label_volume = np.zeros(bold.shape[:3], dtype=np.int16)
    roi_names = ("amygdala", "hippocampus", "insula")
    for atlas in atlases.values():
        atlas_image = nib.load(atlas["maps"])
        aligned = resample_to_img(
            atlas_image, bold, interpolation="nearest", force_resample=True, copy_header=True
        )
        atlas_values = aligned.get_fdata()
        for label, name in enumerate(atlas["labels"]):
            text = name.lower()
            for roi_index, roi in enumerate(roi_names, start=1):
                if roi in text or (roi == "insula" and "insular" in text):
                    label_volume[atlas_values == label] = roi_index
    if not {1, 2, 3} <= set(np.unique(label_volume)):
        raise ValueError("Harvard-Oxford atlas does not resolve all three whole-region ROIs")
    return label_volume
