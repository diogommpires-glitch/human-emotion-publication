# Remaining work

The simulator, contact learning, weight sessions and local research workflows are
implemented. Their next useful test is how well prespecified predictions compare
with acquired observations. Empirical validation remains false.

Work is ordered by value relative to effort. Keep negative results and frozen
baselines visible. The protocol and provenance requirements are in
[docs/RESEARCH.md](docs/RESEARCH.md).

| Priority | Work and reason | Where it plugs in | Evidence that would close it |
| --- | --- | --- | --- |
| 1 | Diagnose the two threat scenario failures before adjusting gains or labels. Circuit tuning and readout tuning answer different questions. | `tests/check_scenarios.py`, `brain_model/readouts.py::_components`, `brain_model/network.json` | Reproduce across prespecified seeds; identify the failed circuit/readout prediction and add one causal regression before changing it. Report the full scenario set. |
| 2 | Acquire and prepare a small explicit EEG/ratings subset to expose adapter assumptions against real files. | `research/datasets.py::acquire`, `research/preparation.py::prepare_cohort` | Hash-verified licensed files, actual event/rating columns, recorded QC exclusions and a reproducible prepared cohort. Keep participant data outside Git. |
| 3 | Run the frozen EEG/ratings analyses before calibration. Synthetic checks cannot establish predictive value. | `research/pipeline.py::evaluate_cohort`, `research/statistics.py::nested_feature_ridge` | Participant/stimulus-disjoint results against all prespecified baselines, uncertainty and temporal nulls using the pinned seeds. |
| 4 | Supply the explicit fMRIPrep prerequisites and run one small fMRI preparation end to end. | `research/fmri.py::fmriprep_prerequisites`, `run_fmriprep`, `extract_rois` | A locally available pinned container, licence/templates, successful preprocessing log and hashed ROI/confound outputs with timing and censoring QC. |
| 5 | Evaluate fMRI and film ratings as separate cohorts, then combine modality-family tests. | `research/observations.py::hrf_observation`, `research/pipeline.py::evaluate_cohort`, `research/__main__.py::combine_results` | Whole-region held-out predictions at actual scan times, cohort separation and combined multiplicity correction. No claim about unresolved nuclei. |
| 6 | Run exploratory six-gain calibration after frozen results, preserving the original outputs. | `research/statistics.py::nested_gain_calibration`, `research/encoders.py::apply_gains` | At most 32 prespecified settings selected entirely within training folds. Report held-out performance separately from the frozen analysis. |
| 7 | Test whether learning adds predictive value beyond state/order effects. | `research/simulation.py::evaluate_session_order` | Fixed, shuffled and independent episodes, no-learning probes and frozen-start comparisons with uncertainty. A weight change alone is insufficient. |
| 8 | Establish the validity and sensitivity limits of the readout, LFP and BOLD proxies. | `brain_model/readouts.py::build_response`, `research/observations.py` | Prespecified ablations and sensitivity analysis against observed data, including null or failed results. Define each proxy's intended scope before interpreting it. |
| 9 | Verify a clean Linux CPU setup and benchmark larger configured GPU sizes under an explicit budget. | `README.md`, `tools/check.py`, `benchmark.py` | Fresh-environment setup, CPU/browser check logs and measured build/runtime/memory by size. Record unsupported cases rather than inferring support from source paths. |

Finer interneuron populations, additional physiology and anatomical detail need a
specific empirical question and an observable acceptance criterion before they
justify expanding the model.
