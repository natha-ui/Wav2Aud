"""The cochlea: an ERB-spaced gammatone filterbank.

This is the mechanical heart of the biomimetic ear. The basilar membrane is
modelled as a bank of overlapping gammatone bandpass filters whose centre
frequencies are spaced uniformly on the ERB (equivalent rectangular bandwidth)
scale of Glasberg & Moore -- the standard description of mammalian frequency
analysis. Each filter's output is the displacement of one place along the
membrane.

Design notes on the *biomimicry*:

* **Human cochlea** -> ERB spacing and gammatone impulse shape.
* **Bat cochlea** -> an optional "fovea": extra channel density and sharper
  tuning over a chosen band (``sharpen_band``), mirroring the acoustic fovea
  bats devote to their echolocation frequency.
* **Insect antenna / Johnston's organ** -> :class:`AntennaResonator`, a lightly
  damped resonant front coupler used by some transduction paths to emphasise a
  mechanical resonance before the cochlea proper.

The filterbank runs at audio sample rate on whatever drive signal the
transduction front-end produces; it never sees GHz/MHz carriers directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve


def hz_to_erb_number(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=float)
    return 21.4 * np.log10(1.0 + 0.00437 * f)


def erb_number_to_hz(e: np.ndarray) -> np.ndarray:
    e = np.asarray(e, dtype=float)
    return (10.0 ** (e / 21.4) - 1.0) / 0.00437


def erb_space(f_lo: float, f_hi: float, n: int) -> np.ndarray:
    """``n`` centre frequencies from ``f_lo`` to ``f_hi``, equal on the ERB scale."""
    return erb_number_to_hz(np.linspace(hz_to_erb_number(f_lo), hz_to_erb_number(f_hi), n))


def erb_bandwidth(cf: np.ndarray) -> np.ndarray:
    """Glasberg & Moore ERB (Hz) at centre frequency ``cf``."""
    return 24.7 * (1.0 + 0.00437 * np.asarray(cf, dtype=float))


@dataclass
class Cochleagram:
    """Output of the cochlea: basilar-membrane displacement per place & time."""

    bm: np.ndarray          # [n_channels, n_samples] band-pass filtered drive
    cf: np.ndarray          # [n_channels] centre frequencies (Hz)
    fs: float               # audio sample rate

    @property
    def n_channels(self) -> int:
        return self.bm.shape[0]


class GammatoneFilterbank:
    """Bank of 4th-order gammatone filters (FIR from the analytic impulse)."""

    def __init__(
        self,
        fs: float,
        f_lo: float = 45.0,
        f_hi: float | None = None,
        n_channels: int = 36,
        order: int = 4,
        sharpen_band: tuple[float, float] | None = None,
        sharpen: float = 1.0,
    ):
        self.fs = float(fs)
        f_hi = float(f_hi) if f_hi else 0.45 * self.fs
        self.order = int(order)
        self.cf = erb_space(f_lo, f_hi, n_channels)
        # Bat-fovea sharpening: narrow the bandwidth of channels inside the band.
        self._q_scale = np.ones_like(self.cf)
        if sharpen_band is not None and sharpen != 1.0:
            lo, hi = sharpen_band
            inside = (self.cf >= lo) & (self.cf <= hi)
            self._q_scale[inside] = 1.0 / float(sharpen)
        self._irs = [self._make_ir(cf, q) for cf, q in zip(self.cf, self._q_scale)]

    def _make_ir(self, cf: float, q_scale: float) -> np.ndarray:
        b = 1.019 * erb_bandwidth(cf) * q_scale  # effective bandwidth (Hz)
        # Truncate the impulse to ~a handful of time constants (cap for speed).
        dur = min(0.10, 6.0 / (2.0 * np.pi * b))
        n = max(8, int(dur * self.fs))
        t = np.arange(n) / self.fs
        ir = t ** (self.order - 1) * np.exp(-2.0 * np.pi * b * t) * np.cos(2.0 * np.pi * cf * t)
        # Normalise to unit peak magnitude response so channels are comparable.
        mag = np.abs(np.fft.rfft(ir, 8192))
        peak = mag.max()
        if peak > 0:
            ir = ir / peak
        return ir.astype(np.float64)

    def process(self, drive: np.ndarray) -> Cochleagram:
        x = np.asarray(drive, dtype=float)
        n = len(x)
        bm = np.empty((len(self.cf), n), dtype=float)
        for i, ir in enumerate(self._irs):
            bm[i] = fftconvolve(x, ir, mode="full")[:n]
        return Cochleagram(bm=bm, cf=self.cf.copy(), fs=self.fs)


class AntennaResonator:
    """A single lightly-damped mechanical resonance (insect antenna / JO).

    Applied *before* the cochlea by some transduction paths to model the way an
    antenna's flagellum mechanically resonates at a preferred frequency,
    boosting a band before neural transduction. Implemented as a 2nd-order
    resonant bandpass in the time domain.
    """

    def __init__(self, fs: float, f0: float, q: float = 6.0, gain: float = 1.0):
        self.fs = float(fs)
        self.f0 = float(f0)
        self.q = float(q)
        self.gain = float(gain)

    def process(self, x: np.ndarray) -> np.ndarray:
        from scipy.signal import iirpeak, lfilter

        w0 = min(self.f0 / (self.fs / 2.0), 0.99)
        if w0 <= 0:
            return np.asarray(x, dtype=float)
        b, a = iirpeak(w0, self.q)
        y = lfilter(b, a, np.asarray(x, dtype=float))
        # Blend so the resonance colours rather than replaces the signal.
        return (1.0 - self.gain) * np.asarray(x, float) + self.gain * y
