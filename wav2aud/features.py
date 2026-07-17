"""Perceptual feature extraction from the neurogram.

These are *auditory* features read off the cochlear/neural representation --
not raw FFT bins. They summarise what the ear "hears" and become the handles
the musical mapper turns:

======================  =============================================
Feature                 Drives (typical)
======================  =============================================
loudness                loudness / dynamics
brightness (centroid)   brightness / timbre / pitch register
bandwidth (spread)      timbre richness, noise content
flatness                noise_content vs harmony (tonal <-> noisy)
onset_rate              rhythm density, tempo
flux                    tremolo, panning motion, vibrato depth
tempo_bpm               tempo
roughness               distortion / edge
======================  =============================================
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .haircell import Neurogram, to_control_rate


@dataclass
class WaveFeatures:
    loudness: float          # 0..1 overall neural drive
    brightness: float        # 0..1 (spectral centroid, log-mapped)
    centroid_hz: float
    bandwidth: float         # 0..1 spectral spread
    flatness: float          # 0..1 (0 tonal, 1 noise-like)
    onset_rate: float        # onsets per second
    flux: float              # 0..1 temporal variability
    tempo_bpm: float
    roughness: float         # 0..1 fast fluctuation
    duration: float          # seconds (audio)
    cf: np.ndarray = field(default=None, repr=False)
    channel_energy: np.ndarray = field(default=None, repr=False)
    loudness_curve: np.ndarray = field(default=None, repr=False)
    control_rate: float = 0.0
    aux: dict = field(default_factory=dict)


def _detect_onsets(loud: np.ndarray, cr: float):
    if loud.size < 3:
        return np.array([], dtype=int)
    d = np.diff(loud, prepend=loud[:1])
    d = np.maximum(d, 0.0)
    if d.max() <= 0:
        return np.array([], dtype=int)
    thr = d.mean() + 1.0 * d.std()
    peaks = []
    min_gap = max(1, int(0.05 * cr))  # 50 ms refractory
    last = -min_gap
    for i in range(1, len(d) - 1):
        if d[i] > thr and d[i] >= d[i - 1] and d[i] > d[i + 1] and (i - last) >= min_gap:
            peaks.append(i)
            last = i
    return np.asarray(peaks, dtype=int)


def _estimate_tempo(onsets: np.ndarray, cr: float) -> float:
    if onsets.size < 2:
        return 0.0
    iois = np.diff(onsets) / cr  # seconds
    iois = iois[iois > 1e-3]
    if iois.size == 0:
        return 0.0
    bpm = 60.0 / float(np.median(iois))
    # fold into a musical range
    while bpm > 180:
        bpm /= 2.0
    while 0 < bpm < 50:
        bpm *= 2.0
    return bpm


def extract_features(neuro: Neurogram, aux: dict | None = None, control_rate: float = 250.0) -> WaveFeatures:
    env, cr = to_control_rate(neuro, control_rate)
    cf = neuro.cf
    n_ch, n_t = env.shape
    duration = n_t / cr if cr else 0.0

    per_ch = env.mean(axis=1)
    # Hair-cell compression (x**0.35) flattens the channel-energy distribution;
    # re-expand it (~inverse power) so tonal vs noisy and dark vs bright are
    # discriminable for the spectral statistics below.
    ch_lin = np.power(per_ch, 2.6)
    total = ch_lin.sum() + 1e-12
    p = ch_lin / total

    centroid_hz = float(np.sum(p * cf))
    spread_hz = float(np.sqrt(np.sum(p * (cf - centroid_hz) ** 2)))

    # brightness: map centroid onto log range of the bank -> 0..1
    lo, hi = float(cf.min()), float(cf.max())
    brightness = float(np.clip((np.log2(max(centroid_hz, lo)) - np.log2(lo)) / (np.log2(hi) - np.log2(lo) + 1e-9), 0, 1))
    bandwidth = float(np.clip(spread_hz / (0.5 * (hi - lo) + 1e-9), 0, 1))

    # spectral flatness across channels: tonal (low) vs noisy (high)
    geo = np.exp(np.mean(np.log(ch_lin + 1e-12)))
    ari = ch_lin.mean() + 1e-12
    flatness = float(np.clip(geo / ari, 0, 1))

    loud = env.sum(axis=0)
    loud_n = loud / (loud.max() + 1e-12)
    # loudness ~ how continuously "filled" the sound is (sustained tone -> high,
    # sparse pings -> low), a musically useful dynamic proxy.
    loudness = float(np.clip(loud_n.mean() * 1.4, 0.05, 1.0))

    flux = float(np.clip(np.mean(np.abs(np.diff(loud_n))) * cr / 8.0, 0, 1))
    roughness = float(np.clip(np.std(np.diff(loud_n)) * np.sqrt(cr) / 6.0, 0, 1))

    onsets = _detect_onsets(loud_n, cr)
    onset_rate = float(onsets.size / duration) if duration > 0 else 0.0
    tempo_bpm = _estimate_tempo(onsets, cr)

    return WaveFeatures(
        loudness=loudness,
        brightness=brightness,
        centroid_hz=centroid_hz,
        bandwidth=bandwidth,
        flatness=flatness,
        onset_rate=onset_rate,
        flux=flux,
        tempo_bpm=tempo_bpm,
        roughness=roughness,
        duration=duration,
        cf=cf,
        channel_energy=per_ch,
        loudness_curve=loud_n,
        control_rate=cr,
        aux=dict(aux or {}),
    )
