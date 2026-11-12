# Local research workflow

The research package runs numeric mechanics, EEG, fMRI, ratings and session-order
experiments locally. A completed job means that computation finished. It does not
mean the model predicts human emotion successfully. No empirical dataset has been
run as part of implementation, and no clinical or individual-level claim follows
from the synthetic checks.

The CPU smoke commands, from the repository root, are:

```powershell
.venv/Scripts/python.exe -m research --help
.venv/Scripts/python.exe -m research run protocols/synthetic.json
.venv/Scripts/python.exe -m research run protocols/mechanics.json
.venv/Scripts/python.exe -m research prerequisites
.venv/Scripts/python.exe -m pytest tests/test_research.py tests/test_research_boundaries.py tests/test_research_pipeline.py
```

Install the pinned optional `research` dependencies from the project environment.
MNE, Nilearn, SciPy, scikit-learn and media libraries load only in the paths which
need them. Missing dependencies fail with their import error. The default network
bridge deliberately uses CPU; it never silently allocates a large GPU experiment.
The 256-neuron smoke protocol is a mechanics test, not an anatomical resolution
claim. Its reduced minimum region size is recorded in its checkpoint configuration.

## Runs and restart

`ExperimentStore(root)` exposes `submit(spec)`, `get(id)`, `list()`, `cancel(id)`,
`resume(id)`, `recover()` and synchronous `run_pending()`. `active_count` includes
queued and currently executing jobs. One worker owns a process lock. Cancellation
is checked between simulation chunks, stimuli and fit stages. HTTP handlers may
call these methods while a background worker executes `run_pending()`.

Each run has a versioned JSON manifest and immutable, hashed numeric NPZ chunks.
Chunks include neural drive, regional rates, occupancy and field-potential traces,
plus the continuous state and realised topology. They contain no raster, names,
photos or memory notes. Neural state, delayed release, chemical state, plasticity,
RNG state and absolute time survive a restart; topology is restored from its arrays.
Manifests are flushed then atomically replaced. Unpublished chunks from an interrupted
write are retained and never treated as completed work. The default disk budget is
8 GiB. Old runs, datasets and checkpoints are never automatically deleted.

```powershell
.venv/Scripts/python.exe -m research list
.venv/Scripts/python.exe -m research status RUN_ID
.venv/Scripts/python.exe -m research cancel RUN_ID
.venv/Scripts/python.exe -m research resume RUN_ID
```

After a crashed process, `recover` acquires the same OS advisory lock as the worker
before marking running jobs interrupted. Windows byte-range locks and POSIX flock
release automatically on process exit; no PID probe or stale-file deletion is used.
`resume` verifies recorded chunk hashes before continuing.
Empirical jobs checkpoint each completed seed/method fit and resume from those
results; an interrupted incomplete method is recomputed. Resume identities include
the complete network configuration, scientific source hashes, protocol version
and numerical package versions; changed scientific code requires a new run. Progress reports those
fit stages. The CLI returns a nonzero status when a run fails. Numeric archives are checked
for dtype, shape, payload length and aggregate bytes before allocating arrays.

## Acquire an explicit subset

The pinned releases are ds005540 1.0.7 (EEG/ratings), ds004892 1.0.2 (film fMRI),
and ds004872 1.0.3 (the separate film annotation cohort). `protocols/datasets.json`
records protocol choices and public metadata links. It is not a download manifest.

A download manifest has `dataset`, `version`, `schema_version: 1`, `license: CC0`
and `files`. Each file has `path`, HTTPS `url`, independently recorded `sha256`,
`size_bytes`, `license`, and optionally `role: media`. Paths are relative to the
release directory. Resolve the exact release's content and hashes before creating
this manifest. Do not substitute the latest release or guess annex object URLs.

```powershell
.venv/Scripts/python.exe -m research acquire .local/eeg-download.json .local/data --max-bytes 1073741824
```

The command checks the complete byte budget before opening the network, streams
each file with a declared-size bound, verifies its hash and refuses to overwrite
existing different data. HTTPS redirects are restricted to public acquisition
hosts. A receipt records every acquired file and licence. Acquiring a subject,
film or atlas is always an explicit command; tests acquire no real participant data.

Dataset metadata licences do not cover every original stimulus. Use the EEG
authors' released audiovisual embeddings. For film features use only media with
its own recorded CC licence, source URL and SHA-256. Derived feature files contain
`time_seconds` and a finite `features` matrix; timestamps use source seconds.

```powershell
.venv/Scripts/python.exe -m research av-features author.mat .local/av.npz --visual-key VISUAL_KEY --audio-key AUDIO_KEY --sample-hz DOCUMENTED_RATE
.venv/Scripts/python.exe -m research film-features film.mp4 audio.wav media-license.json .local/film.npz
```

The embedding keys and rate must come from the actual author file/metadata.
Film RMS uses PCM audio; luminance and absolute frame change use decoded RGB frames.
The audio WAV must be extracted without changing the media timeline. Imagery uses
the actual guidance narrative in the source manifest, never its target emotion
label. Character TF-IDF and truncated SVD support the released narrative language.

## Prepare observations

`prepare SOURCE.json NEW_DIRECTORY` reads a source manifest and produces a hashed
cohort manifest plus numeric observations. Existing output directories are refused.
A source manifest contains:

- `schema_version`, `protocol`, `dataset`, `version`, `data_root`, `dataset_manifest`;
- `target_names`, and `rating_columns` mapping those names to actual source columns;
- `stimuli`, each with `id`, `family`, `duration_seconds`, `kind` and either actual
  `narrative` or derived feature `path` and `sha256`. Kind is `imagery`, `author_av`
  or `cc_film`; films also need `media_license`;
- `records`, each with a public participant code, session, `stimulus_index` and
  fixed condition-baseline numeric `condition_features`.

EEG records additionally identify acquired `raw` and `events` paths,
`stimulus_column`, `event_index` and optional prespecified `bad_channels`.
Ratings from ds005540 use those event records and actual valence/arousal columns.
The adapter never fills missing ratings. MNE loads supported raw EEG, applies
1-45 Hz filtering, line-frequency notch when available, average reference and
explicit bad-channel interpolation. Trials with annotations, nonfinite samples
or peak-to-peak amplitude above 150 microvolts are excluded with QC indices.
Welch power is integrated over 1-4, 4-8, 8-13, 13-30 and 30-45 Hz, normalised by
their summed power, then logged. Zero-power trials fail explicitly.

Film annotation records identify acquired `annotations` and `sidecar` files,
plus `stimulus_onset_seconds`. The sidecar supplies SamplingFrequency and StartTime.
Good, Bad, Alert and Calm are mapped to actual column names. Samples outside the
film or before the first 10 ms model bin are excluded and counted. Scanner and
annotation participant codes are never paired across datasets.

For fMRI, first preprocess explicitly:

```powershell
.venv/Scripts/python.exe -m research prerequisites
.venv/Scripts/python.exe -m research atlas .local/atlases .local/atlases.json
.venv/Scripts/python.exe -m research fmriprep BIDS_ROOT DERIVATIVES WORK_DIR LICENSE_FILE LOG_FILE --participants 01
```

The pinned container is `nipreps/fmriprep:25.2.7`, with `--fs-no-reconall`,
MNI152NLin6Asym 2 mm output, four threads, 8 GiB memory, no telemetry and no network.
`--pull=never` prevents implicit image acquisition. A Linux-container runtime,
pre-pulled image, FreeSurfer licence and available TemplateFlow templates are
required. WSL alone is insufficient. This implementation machine has no Docker,
so the full fMRIPrep workflow has not been executed here.

Each fMRI source record specifies preprocessed `bold`, `confounds`, `sidecar`,
their `_sha256` values and `stimulus_onset_seconds`. `atlases` is the atlas manifest
from the explicit atlas command. Nilearn regresses six motion parameters, CSF and
white matter, detrends, censors framewise displacement >=0.5 mm and nonsteady
volumes, and rejects runs with over 20% censored volumes or fewer than ten usable
volumes. Both hemispheres contribute to whole amygdala, hippocampus and insula
signals. Amygdala subdivisions and small nuclei are not separately measured.

```powershell
.venv/Scripts/python.exe -m research prepare .local/eeg-source.json .local/cohorts/eeg
.venv/Scripts/python.exe -m research run protocols/eeg.json
```

## Fit and evaluate

Five participant folds are crossed with three EEG stimulus-family folds. Film
analysis crosses five participant folds with leave-one-film-out. Rows sharing
either held-out identity are excluded from training. Inner folds follow the same
two-axis rule. Every learned stimulus transform, feature scaler, ridge coefficient
and observation slope is fitted within its training fold. No time compression is
used. Simulated input levels are sampled at their actual one-second boundaries.

The fixed label-blind projection uses six input channels. It is a modelling
assumption, not a measured semantic-to-neural mapping. Seven prespecified simulator
rating predictors summarise regional rate mean/SD/maximum, neural-drive mean/SD,
and occupancy mean/SD. Film ratings use the preceding second; EEG ratings use the
trial. Ridge alpha is selected from 0.01, 0.1, 1, 10 and 100 inside training folds.
Mean, condition, direct-stimulus ridge, frozen-simulator and randomised-projection-
graph baselines are retained. With primary `learning_rule: none`, the simulator
and frozen-simulator methods coincide intentionally. `pair_stdp` is an explicit
experimental alternative, never silently enabled.

BOLD predictors sum absolute excitatory and inhibitory synaptic current over each
whole region, convolve with a normalised canonical double-gamma HRF (6 s peak,
16 s undershoot, ratio 6), then sample at actual scan times. One slope/intercept per
ROI is fitted only on training observations. This is an observation proxy, not a
claim that spiking activity directly measures BOLD.

Seeds are 11, 23, 37, 53 and 71. Primary metrics are EEG MSE, BOLD explained
variance and rating MAE. Paired error improvement is resampled with independent
participant and stimulus cluster multiplicities for 10,000 draws. BOLD uncertainty
recomputes explained-variance improvement for every weighted draw. Temporal nulls
use the declared primary metric, 1,000 within-series nonzero circular shifts and
a plus-one p-value. Constant
BOLD targets have undefined explained variance, reported as null with a count.
The primary seed is 11; other seeds are sensitivity analyses, not independent
replicates. Combine all three modality results for Holm correction:

```powershell
.venv/Scripts/python.exe -m research combine EEG_RESULTS FMRI_RESULTS RATING_RESULTS --output NEW_RESULT.json
```

Optional `calibration_settings` contains at most 32 prespecified settings with
exactly sensory_gain, threat_gain, reward_gain, social_gain, stress_gain and
control_gain. Calibration runs only after the frozen analysis, chooses gains and
ridge alpha in inner training folds and reports separately as exploratory. Input
levels saturate at [0,1]. Test outcomes never choose settings. Session-order
protocols compare fixed, shuffled and independent episodes with weights-only
carryover and no-learning probes against frozen starting weights.

## Primary sources

- [EmoEEG-MC authors and stimulus embeddings](https://github.com/ncclab-sustech/EmoEEG-MC/blob/main/Readme.md)
- [Emo-FilM dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12019557/)
- [MNE filtering documentation](https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html)
- [Nilearn labelled ROI extraction](https://nilearn.github.io/stable/modules/generated/nilearn.maskers.NiftiLabelsMasker.html)
- [fMRIPrep command-line documentation](https://fmriprep.org/en/stable/usage.html)
- [SPM haemodynamic modelling](https://www.fil.ion.ucl.ac.uk/spm/docs/tutorials/beginners/haemodynamic_modelling/)
- [ImageIO FFmpeg decoding](https://imageio.readthedocs.io/en/stable/_autosummary/imageio.plugins.ffmpeg.html)
