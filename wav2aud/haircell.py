"""Hair-cell transduction: from membrane motion to neural activity.

Vertebrate hair cells convert basilar-membrane displacement into a neural
firing rate. We reproduce the perceptually important nonlinearities:

* **Inner hair cell (IHC) rectification** -- cells respond to deflection in one
  direction, so the drive is half-wave rectified.
* **Outer hair cell (OHC) compression** -- the active cochlear amplifier
  compresses a huge dynamic range into a narrow one (instantaneous companding).
* **Transduction low-pass** -- the cell membrane cannot follow fine structure
  above ~1 kHz, so we low-pass the rectified signal into an envelope.
* **Neural adaptation** -- firing emphasises *onsets*; a slow leaky baseline is
  subtracted to sharpen attacks (this is what gives rhythm its bite).

The result is a *neurogram* / cochleagram: a channel x time map of neural
activity that is later resynthesised as music.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt, sosfilt

from .cochlea import Cochleagram


def _lowpass_sos(cutoff: float, fs: float, order: int = 2):
    wn = min(cutoff / (fs / 2.0), 0.99)
    wn = max(wn, 1e-4)
    return butter(order, wn, btype="low", output="sos")


@dataclass
class Neurogram:
    """Channel x time neural activity plus the centre frequencies."""

    activity: np.ndarray    # [n_channels, n_samples] >= 0
    cf: np.ndarray
    fs: float

    @property
    def n_channels(self) -> int:
        return self.activity.shape[0]


class HairCellTransduction:
    def __init__(
        self,
        fs: float,
        compression: float = 0.35,
        transduction_cut: float = 1200.0,
        adapt_tau: float = 0.06,
        onset_emphasis: float = 0.55,
    ):
        self.fs = float(fs)
        self.compression = float(compression)
        self.onset_emphasis = float(onset_emphasis)
        self._env_sos = _lowpass_sos(transduction_cut, fs, order=2)
        self._adapt_sos = _lowpass_sos(1.0 / (2.0 * np.pi * adapt_tau), fs, order=1)

    def process(self, coch: Cochleagram) -> Neurogram:
        bm = coch.bm
        # IHC rectification + OHC compression (instantaneous companding).
        rect = np.maximum(bm, 0.0)
        comp = np.power(rect, self.compression)
        # Transduction low-pass -> smooth envelope per channel.
        env = sosfilt(self._env_sos, comp, axis=1)
        env = np.maximum(env, 0.0)
        # Adaptation: subtract a slow leaky baseline to emphasise onsets.
        baseline = sosfilt(self._adapt_sos, env, axis=1)
        onset = np.maximum(env - baseline, 0.0)
        activity = (1.0 - self.onset_emphasis) * env + self.onset_emphasis * onset
        return Neurogram(activity=activity, cf=coch.cf.copy(), fs=coch.fs)


def to_control_rate(neuro: Neurogram, control_rate: float = 250.0) -> tuple[np.ndarray, float]:
    """Downsample the neurogram to a slow control rate for mapping & resynthesis.

    Returns ``(envelopes[n_channels, n_frames], control_rate)``. Anti-alias
    with a zero-phase low-pass before decimating so control envelopes are clean.
    """
    fs = neuro.fs
    factor = max(1, int(round(fs / control_rate)))
    act = neuro.activity
    if factor > 1:
        sos = _lowpass_sos(0.4 * fs / factor, fs, order=4)
        # zero-phase to avoid smearing onsets in time
        act = sosfiltfilt(sos, act, axis=1)
        act = act[:, ::factor]
    act = np.maximum(act, 0.0)
    return act, fs / factor
