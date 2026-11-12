"""Observation models preserve seconds, Hz and the whole-region ROI resolution."""

import numpy as np

from .statistics import matrix

BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))


def log_relative_bands(samples, sample_hz):
    from scipy.signal import welch

    data = matrix(samples, "samples")
    if not np.isfinite(sample_hz) or sample_hz <= 90 or data.shape[1] < sample_hz:
        raise ValueError("EEG requires sampling above 90 Hz and at least one second of data")
    frequencies, power = welch(
        data, fs=sample_hz, nperseg=min(data.shape[1], round(sample_hz * 2)), axis=1
    )
    bands = np.stack(
        [power[:, (frequencies >= low) & (frequencies < high)].sum(axis=1) for low, high in BANDS],
        axis=1,
    )
    total = bands.sum(axis=1, keepdims=True)
    if (total <= 0).any():
        raise ValueError("zero-power EEG epochs are invalid, not zero log power")
    return np.log(np.maximum(bands / total, np.finfo(float).tiny))


def hrf_observation(neural_drive, dt_seconds, scan_times_seconds):
    from scipy.special import gamma

    drive = matrix(neural_drive, "neural_drive")
    scans = np.asarray(scan_times_seconds, dtype=float)
    if (
        not np.isfinite(dt_seconds)
        or dt_seconds <= 0
        or scans.ndim != 1
        or not np.isfinite(scans).all()
    ):
        raise ValueError("HRF needs positive sample spacing and finite scan times")
    times = np.arange(len(drive)) * dt_seconds
    if len(scans) == 0 or scans.min() < 0 or scans.max() > times[-1]:
        raise ValueError("scan times must fall inside the recorded neural drive")
    t = np.arange(0, 32, dt_seconds)
    # SPM canonical double-gamma response, 6 s peak and 16 s undershoot, ratio 6.
    hrf = t**5 * np.exp(-t) / gamma(6) - t**15 * np.exp(-t) / (6 * gamma(16))
    hrf /= hrf.sum()
    convolved = np.stack(
        [np.convolve(drive[:, index], hrf)[: len(drive)] for index in range(drive.shape[1])], axis=1
    )
    unique_scans, inverse = np.unique(scans, return_inverse=True)
    sampled = np.stack(
        [np.interp(unique_scans, times, convolved[:, index]) for index in range(drive.shape[1])],
        axis=1,
    )
    return sampled[inverse]


def fit_bold_observation(train_drive, train_bold, test_drive):
    x, y, test = (
        matrix(train_drive, "train drive"),
        matrix(train_bold, "train BOLD"),
        matrix(test_drive, "test drive"),
    )
    if x.shape != y.shape or x.shape[1] != test.shape[1]:
        raise ValueError("BOLD observation fits one slope and intercept per corresponding ROI")
    predictions = np.empty_like(test)
    coefficients = []
    for roi in range(x.shape[1]):
        design = np.column_stack([np.ones(len(x)), x[:, roi]])
        fit = np.linalg.lstsq(design, y[:, roi], rcond=None)[0]
        predictions[:, roi] = fit[0] + fit[1] * test[:, roi]
        coefficients.append(fit.tolist())
    return predictions, coefficients


def preprocess_eeg(path, events, *, line_hz=50.0, bad_channels=(), peak_to_peak_volts=150e-6):
    import mne

    raw = mne.io.read_raw(path, preload=True, verbose="ERROR")
    raw.pick("eeg")
    raw.info["bads"] = list(bad_channels)
    if bad_channels:
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    if raw.info["sfreq"] <= 2 * 45:
        raise ValueError("EEG acquisition sampling rate cannot resolve the 30-45 Hz band")
    if line_hz < raw.info["sfreq"] / 2:
        raw.notch_filter(np.array([line_hz]), verbose="ERROR")
    raw.filter(1.0, 45.0, verbose="ERROR")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    results, excluded = [], []
    for index, event in enumerate(events):
        onset, duration = float(event["onset"]), float(event["duration"])
        if onset < 0 or duration < 1 or onset + duration > raw.n_times / raw.info["sfreq"]:
            raise ValueError("EEG event exceeds recording or is shorter than one second")
        start, stop = (
            round(onset * raw.info["sfreq"]),
            round((onset + duration) * raw.info["sfreq"]),
        )
        epoch = raw.get_data(start=start, stop=stop, reject_by_annotation="NaN")
        if not np.isfinite(epoch).all() or (np.ptp(epoch, axis=1) > peak_to_peak_volts).any():
            excluded.append(index)
            continue
        results.append(
            {
                "event_index": index,
                "onset_seconds": onset,
                "duration_seconds": duration,
                "log_relative_power": log_relative_bands(epoch, raw.info["sfreq"])
                .mean(axis=0)
                .tolist(),
            }
        )
    return {
        "features": results,
        "qc": {
            "accepted": len(results),
            "excluded_event_indices": excluded,
            "sampling_hz": raw.info["sfreq"],
            "peak_to_peak_volts": peak_to_peak_volts,
            "reference": "average",
            "band_edges_hz": BANDS,
            "interpolated_channels": list(bad_channels),
        },
    }
