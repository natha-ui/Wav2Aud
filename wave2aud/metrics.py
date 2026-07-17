"""Quantitative measures so the audio can be compared directly to the wave.

Three groups:

1. **Paired descriptors** -- the same acoustic statistics computed on the
   coupled *wave drive* and on the *audio*, side by side (spectral centroid,
   bandwidth, Wiener entropy, crest factor, modulation rate).
2. **Preservation scores** -- how faithfully the audio tracks the wave: the
   correlation of their amplitude envelopes (rhythm/dynamics) and the rank
   correlation of their spectral-centroid trajectories (does the tune follow
   the wave's spectral evolution?).
3. **Interpretability (ear vs FFT)** -- sensory roughness, musical-scale
   conformity and Wiener entropy for the biomimetic-ear output versus the naive
   FFT baseline (:mod:`wave2aud.baseline`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import stft
from scipy.stats import pearsonr, spearmanr

from .baseline import fft_sonify


# ---------------------------------------------------------------------------
# low-level signal statistics
# ---------------------------------------------------------------------------
def _mono(x):
    x = np.asarray(x, float)
    return x.mean(axis=1) if x.ndim == 2 else x


def _spectrum(x, fs):
    x = _mono(x)
    w = np.hanning(len(x))
    X = np.abs(np.fft.rfft(x * w)) + 1e-12
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    return f, X


def spectral_centroid(x, fs):
    f, X = _spectrum(x, fs)
    p = X / X.sum()
    c = float((f * p).sum())
    bw = float(np.sqrt(((f - c) ** 2 * p).sum()))
    return c, bw


def wiener_entropy(x, fs):
    """Spectral flatness / Wiener entropy (a birdsong staple). 0 tonal, 1 noisy."""
    _, X = _spectrum(x, fs)
    P = X ** 2
    return float(np.exp(np.mean(np.log(P))) / (np.mean(P) + 1e-20))


def crest_factor(x):
    x = _mono(x)
    return float(np.max(np.abs(x)) / (np.sqrt(np.mean(x ** 2)) + 1e-12))


def modulation_rate(x, fs):
    """Dominant amplitude-modulation rate of the envelope (Hz)."""
    e = np.abs(_mono(x))
    e = e - e.mean()
    if len(e) < 8:
        return 0.0
    E = np.abs(np.fft.rfft(e * np.hanning(len(e))))
    fr = np.fft.rfftfreq(len(e), 1.0 / fs)
    band = (fr > 0.3) & (fr < 30)
    if not band.any() or E[band].sum() == 0:
        return 0.0
    return float(fr[band][np.argmax(E[band])])


def roughness(x, fs, n_peaks=24):
    """Sethares sensory dissonance of the strongest spectral peaks (lower = smoother)."""
    f, X = _spectrum(x, fs)
    idx = np.argsort(X)[-n_peaks:]
    fr, am = f[idx], X[idx] / (X.max() + 1e-12)
    order = np.argsort(fr)
    fr, am = fr[order], am[order]
    d = 0.0
    for i in range(len(fr)):
        for j in range(i + 1, len(fr)):
            fmin = max(fr[i], 1.0)
            s = 0.24 / (0.0207 * fmin + 18.96)
            df = fr[j] - fr[i]
            d += am[i] * am[j] * (np.exp(-3.5 * s * df) - np.exp(-5.75 * s * df))
    return float(d / (am.sum() ** 2 + 1e-9))


def scale_conformity(x, fs, tol_cents=15.0):
    """Fraction of spectral energy within ``tol_cents`` of a 12-TET semitone."""
    f, X = _spectrum(x, fs)
    m = 12 * np.log2(np.maximum(f, 1.0) / 440.0) + 69
    cents = np.abs(m - np.round(m)) * 100.0
    return float(X[cents <= tol_cents].sum() / (X.sum() + 1e-12))


# ---------------------------------------------------------------------------
# preservation (wave <-> audio)
# ---------------------------------------------------------------------------
def _envelope(x, fs, n=256):
    e = np.abs(_mono(x))
    idx = np.linspace(0, len(e), n + 1).astype(int)
    return np.array([e[idx[i]:idx[i + 1]].mean() if idx[i + 1] > idx[i] else 0.0
                     for i in range(n)])


def _centroid_contour(x, fs, n=64, frame=1024):
    x = _mono(x)
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    f, tt, Z = stft(x, fs=fs, nperseg=frame, noverlap=frame // 2)
    mag = np.abs(Z) + 1e-9
    c = (f[:, None] * mag).sum(0) / mag.sum(0)
    if len(c) < 2:
        return np.zeros(n)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(c)), c)


def preservation_scores(drive, audio, fs):
    de, ae = _envelope(drive, fs), _envelope(audio, fs)
    env_r = float(pearsonr(de, ae)[0]) if de.std() > 0 and ae.std() > 0 else 0.0
    dc, ac = _centroid_contour(drive, fs), _centroid_contour(audio, fs)
    cont_rho = float(spearmanr(dc, ac).correlation) if np.std(dc) > 0 and np.std(ac) > 0 else 0.0
    return {"envelope_r": env_r, "centroid_contour_rho": cont_rho}


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------
@dataclass
class ComparisonReport:
    wave: dict
    audio: dict
    preservation: dict
    meta: dict = field(default_factory=dict)

    def rows(self):
        keys = ["centroid_hz", "bandwidth_hz", "wiener_entropy", "crest_factor", "mod_rate_hz"]
        labels = ["spectral centroid (Hz)", "bandwidth (Hz)", "Wiener entropy",
                  "crest factor", "modulation rate (Hz)"]
        return [(lab, self.wave[k], self.audio[k]) for lab, k in zip(labels, keys)]


def _descriptors(x, fs):
    c, bw = spectral_centroid(x, fs)
    return {"centroid_hz": c, "bandwidth_hz": bw, "wiener_entropy": wiener_entropy(x, fs),
            "crest_factor": crest_factor(x), "mod_rate_hz": modulation_rate(x, fs)}


def compare(result) -> ComparisonReport:
    """Paired wave/audio descriptors + preservation for a :class:`SonifyResult`."""
    drive = result.drive.signal
    audio = result.audio
    fs = result.fs
    return ComparisonReport(
        wave=_descriptors(drive, fs),
        audio=_descriptors(audio, fs),
        preservation=preservation_scores(drive, audio, fs),
        meta={"category": result.sample.wave_type, "label": result.sample.label()},
    )


def mapping_monotonicity(param_values, results, audio_feature="centroid_hz"):
    """Rank-correlation between a swept wave parameter and an audio feature.

    A high value means the audio dimension (e.g. pitch/centroid) tracks the
    physical parameter (e.g. target velocity) monotonically -- so a listener can
    read the wave off the sound. Returns ``(rho, audio_values)``.
    """
    ys = [_descriptors(r.audio, r.fs)[audio_feature] for r in results]
    xs = np.asarray(param_values, float)
    if np.std(xs) == 0 or np.std(ys) == 0:
        return 0.0, ys
    return float(spearmanr(xs, ys).correlation), ys


def _feature_vectors(audios, fs):
    V = []
    for a in audios:
        c, bw = spectral_centroid(a, fs)
        V.append([np.log2(max(c, 1)), bw, wiener_entropy(a, fs), crest_factor(a), roughness(a, fs)])
    V = np.array(V)
    return (V - V.mean(0)) / (V.std(0) + 1e-9)


def category_separability(labels, audios, fs):
    """Ratio of between-category to within-category distance in feature space.

    Higher = categories are more distinctly identifiable by ear. Computed the
    same way for ear audio and FFT-baseline audio to compare interpretability.
    """
    X = _feature_vectors(audios, fs)
    labels = np.asarray(labels)
    within, between = [], []
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            d = float(np.linalg.norm(X[i] - X[j]))
            (within if labels[i] == labels[j] else between).append(d)
    w = np.mean(within) if within else 1e-9
    b = np.mean(between) if between else 0.0
    return float(b / (w + 1e-9))


def category_tone_matrix(presets=None):
    """Pairwise perceptual distance between the category *tones*.

    Sonifies one representative per category and returns ``(labels, matrix)`` of
    Euclidean distances in feature space. Categories are laid out on a similarity
    continuum (seismic ... ultrasound), so neighbours are close and the extremes
    (seismic vs ultrasound) are the most different.
    """
    from . import simulate as sim
    from .pipeline import Sonifier

    reps = presets or {
        "seismic": "seismic_M3_near", "audio": "audio_vowel_low", "radio": "radio_fm_music",
        "infrared": "ir_body", "radar": "radar_car", "gamma": "gamma_cs137",
        "ultrasound": "us_mid_double",
    }
    labels = list(reps)
    son = Sonifier()
    audios = [son.sonify(sim.preset(reps[k])).audio for k in labels]
    X = _feature_vectors(audios, 44100)
    n = len(labels)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = float(np.linalg.norm(X[i] - X[j]))
    return labels, M


@dataclass
class InterpretabilityReport:
    ear: dict
    fft: dict
    meta: dict = field(default_factory=dict)


def interpretability(sample, fs: float = 44100.0) -> InterpretabilityReport:
    """Compare the biomimetic-ear output with the naive FFT baseline."""
    from .pipeline import Sonifier

    ear_audio = Sonifier(fs=fs).sonify(sample).audio
    fft_audio, _ = fft_sonify(sample, fs=fs)

    def _m(a):
        return {"roughness": roughness(a, fs), "scale_conformity": scale_conformity(a, fs),
                "wiener_entropy": wiener_entropy(a, fs), "crest_factor": crest_factor(a)}

    return InterpretabilityReport(ear=_m(ear_audio), fft=_m(fft_audio),
                                  meta={"category": sample.wave_type, "label": sample.label()})
