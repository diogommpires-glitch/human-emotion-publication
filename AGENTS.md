# Project instructions

This is the authoritative agent entry point for Human Emotion. Read
`README.md`, the relevant source and tests, and `docs/ARCHITECTURE.md` before
changing behaviour. Use `docs/LEARNING.md` for learning/persistence contracts and
`docs/RESEARCH.md` for scientific protocols. Older review reports describe earlier
states and are not proof of current behaviour.

## Environment and commands

Use this repository's `.venv`, never a parent environment or base Anaconda. The
reference Windows environment is Python 3.14.4, Node.js 26.2.0 and uv 0.13.6.
Set `PYTHONUTF8=1` for Python tools on Windows.

```powershell
$env:PYTHONUTF8 = '1'
uv sync --locked --python 3.14.4 --extra cuda --extra research --dev
npm ci --ignore-scripts
.venv/Scripts/python.exe server.py
.venv/Scripts/python.exe tools/check.py
.venv/Scripts/python.exe tools/check.py --browser --coverage
```

Use `--extra cpu` instead of `--extra cuda` for CPU wheels; the extras conflict
intentionally. The RTX 5090 requires CUDA 12.8 or newer. Do not silently replace
the CUDA wheel to hide a GPU failure. `HUMAN_EMOTION_DEVICE=cpu` forces CPU
execution. The combined checks force CPU episodes and also test the small CUDA
transmission kernel when available; `--browser` uses installed Chrome.
On POSIX use `.venv/bin/python`; only Windows has been validated for this upgrade.

## Changes and style

- Prefer the smallest clear change. Read nearby code and resolve discoverable
  facts before asking questions. State material assumptions and the check that
  could disprove them.
- Keep Python/PyTorch, standard-library localhost HTTP and native JavaScript/canvas.
  Keep numerical integration together. Preserve public entry points and formats
  unless the current task explicitly changes them.
- Use typed public boundaries, precise domain names, early returns and comments
  explaining units or constraints. Avoid speculative wrappers and broad error
  hiding. Simplify touched code after correctness.
- Use British spelling in prose and no em dashes. Identifiers retain established
  library spelling. Do not reformat unrelated files.
- The optional local reference is
  `C:/Users/<user>/Desktop/coding_repos/CODE_STYLE_v1.md`, or
  `/mnt/c/Users/<user>/Desktop/coding_repos/CODE_STYLE_v1.md` under WSL. If absent,
  the plain rules in this file apply. Do not ingest or execute the reference
  library wholesale. Project instructions and the user's request take precedence.

## Scientific and persistence invariants

- Keep equations, units, seeds, configuration provenance and error tolerances
  explicit. Inputs and occupancy are dimensionless [0,1], firing is Hz, integration
  is ms and episode/timeline durations are seconds. Raster timestamps are ms.
- Distinguish three population classes from two transmitter channels. GABAergic
  projection cells are inhibitory. Preserve individual contacts and plasticity
  flags when summing canonical CSR transmission slots. Empty populations yield
  zero rates, never NaN.
- Learning rules are mutually exclusive and versioned. Only marked glutamate
  contacts change. Preserve initial weights, bounds, trace ordering, dopamine
  baseline semantics and chunk equivalence. Eligibility is an experimental
  occupancy-gated mechanism, without reward prediction or a value function.
- Independent episodes reset weights and dynamic state. Sessions carry weights
  only. Continuous research chunks retain full neural/chemical/STP/delay/learning/
  RNG state, absolute time and accumulators. Store realised topology/assemblies,
  not seeds alone.
- Keep session computation private until a finite response and complete revision
  can be published. HEAD publication, stale-revision rejection and restart-safe
  idempotency are contracts. Never replace them with best-effort writes.
- Validate external requests and imported checkpoints for bounds, shapes, finite
  values, hashes and scientific consistency. Keep JSON/safetensors sessions and
  numeric research chunks free of executable object deserialisation.
- Fit encoders, scalers, calibration and observation parameters within training
  folds. Preserve participant/stimulus separation, timing, QC, prespecified
  baselines and uncertainty. Empirical validation remains false unless new
  observed evidence justifies a documented change.

## Privacy and verification

- Keep loopback binding, Host/Origin validation, request/allocation bounds and
  resource budgets. Never bypass a scientific or security check to reduce code.
- Do not copy, print, upload or commit secrets, identifying notes, faces, scans,
  real participant data, weights or generated research outputs. Use small synthetic
  fixtures. Model session records exclude identifying browser annotations.
- Do not overwrite data or automatically delete old outputs to satisfy a budget.
  Explicit acquisition requires file provenance, hashes, byte limits and licences.
- Reproduce a bug before fixing its cause. Keep one meaningful regression per
  changed behaviour. Run the relevant focused checks, then `tools/check.py`.
  Add `--browser` for UI changes. Run only one GPU workload at a time.
- Use independent critique for substantial work. Close confirmed findings with
  executable evidence. Review complexity/dead-code findings before deleting public
  entry points. Never optimise source for a cosmetic detector score.
- Report the exact checks run, results, platform and limitations. Another agent's
  claim, an earlier report or intended behaviour is not verification. Whole-model
  scenarios are optional and currently have known threat failures; do not hide them
  behind the default unit-test result.
- Keep dependency changes pinned in the project lockfiles. Do not install admin
  tools, publish, push, change remotes or rewrite history without explicit scope.
