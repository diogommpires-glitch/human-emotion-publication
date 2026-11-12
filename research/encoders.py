"""Label-blind stimulus encoders with training-only learned transforms."""

from pathlib import Path

import numpy as np

from .statistics import matrix


class ImageryEncoder:
    def __init__(self, components=8, seed=11):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=4096)
        self.components, self.seed, self.svd = components, seed, None

    def fit(self, training_texts):
        from sklearn.decomposition import TruncatedSVD

        values = self.vectorizer.fit_transform(training_texts)
        components = min(self.components, values.shape[0] - 1, values.shape[1] - 1)
        if components < 1:
            raise ValueError("imagery encoder needs >=2 distinct training narratives")
        self.svd = TruncatedSVD(n_components=components, random_state=self.seed)
        self.svd.fit(values)
        return self

    def transform(self, texts):
        if self.svd is None:
            raise ValueError("fit the imagery encoder on training narratives first")
        return self.svd.transform(self.vectorizer.transform(texts))


class StimulusScaler:
    def fit(self, training_features):
        x = matrix(training_features, "training stimulus features")
        self.mean, self.scale = x.mean(axis=0), x.std(axis=0)
        self.scale = np.where(self.scale > 1e-12, self.scale, 1.0)
        return self

    def transform(self, features):
        if not hasattr(self, "mean"):
            raise ValueError("stimulus scaler must be fitted on training data")
        x = matrix(features, "stimulus features")
        if x.shape[1] != len(self.mean):
            raise ValueError("stimulus feature dimensions differ from training")
        return (x - self.mean) / self.scale


def author_av_embeddings(path, *, visual_key, audio_key):
    from scipy.io import loadmat

    values = loadmat(path)
    visual, audio = (
        matrix(values[visual_key], "author visual embeddings"),
        matrix(values[audio_key], "author audio embeddings"),
    )
    if len(visual) != len(audio):
        raise ValueError("author embedding timelines differ; use their actual alignment metadata")
    return np.concatenate([visual, audio], axis=1)


def film_frame_features(frames, frame_hz, audio, audio_hz):
    audio = np.asarray(audio, dtype=float)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or not np.isfinite(audio).all() or frame_hz <= 0 or audio_hz <= 0:
        raise ValueError("film features need finite PCM audio and positive source sampling rates")
    features, previous = [], None
    for index, frame in enumerate(frames):
        rgb = np.asarray(frame, dtype=float)
        if rgb.ndim != 3 or rgb.shape[2] < 3 or not np.isfinite(rgb).all():
            raise ValueError("film frames must be finite RGB images")
        luminance = np.sum(rgb[..., :3] * [0.2126, 0.7152, 0.0722], axis=2) / 255
        change = 0.0 if previous is None else float(np.abs(luminance - previous).mean())
        start, stop = round(index * audio_hz / frame_hz), round((index + 1) * audio_hz / frame_hz)
        segment = audio[start:stop]
        if len(segment) == 0:
            raise ValueError("audio does not cover the complete film timeline")
        features.append([float(np.sqrt(np.mean(segment**2))), float(luminance.mean()), change])
        previous = luminance
    if not features:
        raise ValueError("film contains no frames")
    return np.arange(len(features)) / frame_hz, np.asarray(features)


def read_film_features(video_path, audio_path, license_record):
    import imageio.v3 as iio
    from scipy.io import wavfile

    from .store import sha256

    if license_record.get("license") not in {
        "CC0",
        "CC-BY-3.0",
        "CC-BY-4.0",
        "CC-BY-SA-3.0",
        "CC-BY-SA-4.0",
    } or not license_record.get("source_url"):
        raise ValueError("film requires explicit CC licence and source URL")
    if sha256(video_path) != license_record.get("sha256"):
        raise ValueError("film content differs from licensed media manifest")
    metadata = iio.immeta(video_path, plugin="FFMPEG")
    rate, audio = wavfile.read(Path(audio_path))
    if audio.dtype.kind in "iu":
        audio = audio.astype(float) / max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
    return film_frame_features(
        iio.imiter(video_path, plugin="FFMPEG"), float(metadata["fps"]), audio, rate
    )


def encode_drives(features):
    x = matrix(features, "scaled stimulus features")
    # A fixed signed projection is label-blind; it is not an emotion classifier.
    channels = ("novelty", "threat_cue", "reward_cue", "social_touch", "stress", "control")
    basis = np.cos((np.arange(x.shape[1])[:, None] + 1) * (np.arange(6)[None, :] + 1))
    projected = x @ basis / np.sqrt(x.shape[1])
    levels = 0.5 + 0.5 * np.tanh(projected)
    return [dict(zip(channels, row.tolist(), strict=True)) for row in levels]


def apply_gains(drives, gains):
    from .spec import GAINS

    if set(gains) != set(GAINS) or any(
        not np.isfinite(value) or value < 0 for value in gains.values()
    ):
        raise ValueError("six finite nonnegative global gains are required")
    groups = {
        "novelty": "sensory_gain",
        "threat_cue": "threat_gain",
        "reward_cue": "reward_gain",
        "social_touch": "social_gain",
        "stress": "stress_gain",
        "control": "control_gain",
    }
    return [
        {key: float(np.clip(value * gains[groups[key]], 0, 1)) for key, value in row.items()}
        for row in drives
    ]
