"""A naive FFT sonifier -- the baseline the biomimetic ear is compared against.

This is the "typical" approach wave2aud deliberately avoids: take the coupled
drive, STFT it, and turn the loudest spectral bins straight into sustained tones
(octave-folded into an audible band). There is **no** cochlear model, no scale,
no category identity -- so the output is inharmonic, category-agnostic and hard
to read. :mod:`wave2aud.metrics` quantifies exactly how it loses interpretability
versus the ear.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import stft

from .transduction import TransductionFrontEnd
from .waves import WaveSample


def _fold_to_band(freqs, fmin=110.0, fmax=3500.0):
    f = np.array(freqs, float)
    f[f <= 0] = fmin
    while np.any(f < fmin):
        f = np.where(f < fmin, f * 2.0, f)
    while np.any(f > fmax):
        f = np.where(f > fmax, f / 2.0, f)
    return f


def fft_sonify(sample: WaveSample, fs: float = 44100.0, n_tones: int = 16,
               frame: int = 2048, hop: int = 512, duration: float | None = None):
    """Naive spectral resynthesis of the coupled drive (the FFT baseline)."""
    drive = TransductionFrontEnd(fs_out=fs).couple(sample).signal
    if len(drive) < frame:
        drive = np.pad(drive, (0, frame - len(drive)))
    f, tt, Z = stft(drive, fs=fs, nperseg=frame, noverlap=frame - hop)
    mag = np.abs(Z)
    bin_energy = mag.sum(axis=1)
    bin_energy[0] = 0.0
    top = np.argsort(bin_energy)[-n_tones:]
    out = np.zeros(len(drive))
    t = np.arange(len(drive)) / fs
    tone_freqs = _fold_to_band(f[top])
    for bi, ft in zip(top, tone_freqs):
        amp = np.interp(t, tt, mag[bi])
        amp /= mag.max() + 1e-9
        out += amp * np.sin(2 * np.pi * ft * t)
    peak = np.max(np.abs(out)) + 1e-9
    out = 0.9 * out / peak
    stereo = np.stack([out, out], axis=1).astype(np.float32)
    return stereo, np.sort(tone_freqs)
