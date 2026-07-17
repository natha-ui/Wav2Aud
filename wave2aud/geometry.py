"""Geometric interpretation of a wave and its audio (birdsong-style).

Birdsong is often understood geometrically -- as an *attractor* in a delay
embedding of the pressure signal (Takens), and as a *gesture*: a trajectory
through a low-dimensional acoustic feature space (classically pitch vs Wiener
entropy, after Tchernichovski's Sound Analysis). We apply the same two lenses
to *both* the coupled wave drive and the resulting audio, so the shapes can be
laid side by side:

* :func:`delay_embedding` -- the phase portrait / attractor.
* :func:`feature_trajectory` -- the gesture in (pitch, Wiener-entropy) space.

Because the sonification maps the wave's structure onto music continuously, the
audio's gesture is a recognisable transform of the wave's gesture -- the
geometry makes the correspondence visible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import stft


def _mono(x):
    x = np.asarray(x, float)
    return x.mean(axis=1) if x.ndim == 2 else x


def auto_tau(x, fs, default=0.003):
    """Delay (samples) ~ a quarter period of the dominant frequency."""
    x = _mono(x)
    if len(x) < 16:
        return max(1, int(default * fs))
    w = np.hanning(len(x))
    X = np.abs(np.fft.rfft(x * w))
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    X[0] = 0
    fdom = f[int(np.argmax(X))] or (1.0 / default)
    return max(1, int(fs / (4.0 * max(fdom, 1.0))))


def delay_embedding(x, fs, tau=None, dims=3, max_points=3000):
    """Takens delay embedding -> the signal's phase-portrait / attractor."""
    x = _mono(x)
    x = x / (np.max(np.abs(x)) + 1e-12)
    tau = tau or auto_tau(x, fs)
    n = len(x) - (dims - 1) * tau
    if n <= 1:
        return np.zeros((1, dims))
    emb = np.stack([x[i * tau: i * tau + n] for i in range(dims)], axis=1)
    if len(emb) > max_points:
        emb = emb[:: len(emb) // max_points]
    return emb


def feature_trajectory(x, fs, frame=1024, hop=512):
    """Gesture in (pitch, Wiener-entropy) space, plus energy and time."""
    x = _mono(x)
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    f, tt, Z = stft(x, fs=fs, nperseg=frame, noverlap=frame - hop)
    mag = np.abs(Z) + 1e-9
    p = mag / mag.sum(0, keepdims=True)
    centroid = (f[:, None] * p).sum(0)
    pitch_midi = 12 * np.log2(np.maximum(centroid, 1.0) / 440.0) + 69
    P = mag ** 2
    wiener = np.exp(np.mean(np.log(P), axis=0)) / (np.mean(P, axis=0) + 1e-20)
    energy = mag.sum(0)
    energy = energy / (energy.max() + 1e-12)
    return {"time": tt, "pitch_midi": pitch_midi, "centroid_hz": centroid,
            "entropy": wiener, "energy": energy}


@dataclass
class GeometryReport:
    wave_embedding: np.ndarray
    audio_embedding: np.ndarray
    wave_trajectory: dict
    audio_trajectory: dict
    meta: dict


def geometry_report(result, dims=3) -> GeometryReport:
    """Delay embeddings + feature-space gestures for a :class:`SonifyResult`."""
    drive = result.drive.signal
    audio = result.audio
    fs = result.fs
    return GeometryReport(
        wave_embedding=delay_embedding(drive, fs, dims=dims),
        audio_embedding=delay_embedding(audio, fs, dims=dims),
        wave_trajectory=feature_trajectory(drive, fs),
        audio_trajectory=feature_trajectory(audio, fs),
        meta={"category": result.sample.wave_type, "label": result.sample.label()},
    )
