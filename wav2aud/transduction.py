"""Transduction front-ends: coupling each physical wave into the ear's drive.

This is where we deliberately *avoid* generic pre-processing (no "just take an
FFT and map bins to notes"). Instead each modality is coupled into an
audio-rate mechanical drive by a physically-motivated operation -- the same way
a real ear is mechanically driven by pressure -- and the *ear* does the
analysis:

======================  ==================================================
Wave type               Coupling (physical analogue)
======================  ==================================================
seismic                 time/pitch compression (play the ground faster)
ultrasound              heterodyne down-conversion (a bat detector)
radar / microwave       coherent baseband -> Doppler drive
radio                   AM/FM demodulation (recover the message)
infrared                thermal-intensity compression + emissivity colour
gamma                   photon events -> energy-tuned impulse train
======================  ==================================================

Every coupling returns a :class:`Drive`: a real, audio-rate signal plus an
``aux`` dictionary of physically meaningful side-channels (Doppler, range,
arrival angle, event density, temperature, ...) that later steer the music.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import hilbert, resample, resample_poly

from .waves import WaveSample


@dataclass
class Drive:
    """Audio-rate mechanical drive for the biomimetic ear."""

    signal: np.ndarray          # real, audio rate, normalised ~[-1, 1]
    fs: float                   # audio sample rate
    aux: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _norm(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return x * (peak / m) if m > 0 else x


def _resample_len(x: np.ndarray, n_out: int) -> np.ndarray:
    n_out = max(2, int(n_out))
    if len(x) == n_out:
        return np.asarray(x, dtype=float)
    return resample(np.asarray(x, dtype=float), n_out)


def _to_audio_rate(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Rational resample preserving pitch (used after heterodyne/demod)."""
    fs_in = float(fs_in)
    fs_out = float(fs_out)
    if abs(fs_in - fs_out) < 1e-6:
        return np.asarray(x, dtype=float)
    from math import gcd

    up = int(round(fs_out))
    down = int(round(fs_in))
    g = gcd(up, down) or 1
    up //= g
    down //= g
    # keep the polyphase filter tractable
    if max(up, down) > 20000:
        return _resample_len(x, int(len(x) * fs_out / fs_in))
    return resample_poly(np.asarray(x, dtype=float), up, down)


def _dominant_freq(x: np.ndarray, fs: float) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 8:
        return 0.0
    w = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * w))
    freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
    if spec.sum() <= 0:
        return 0.0
    return float(freqs[int(np.argmax(spec))])


# ---------------------------------------------------------------------------
# per-modality couplings
# ---------------------------------------------------------------------------
def couple_seismic(s: WaveSample, fs_out: float, speed: float = 48.0) -> Drive:
    """Play the ground faster: pitch/time compression into the audio band."""
    x = np.real(s.data).astype(float)
    # output duration = input duration / speed, landing at fs_out
    n_out = int(round(len(x) * fs_out / (s.sample_rate * speed)))
    y = _resample_len(x, n_out)
    dom = _dominant_freq(x, s.sample_rate)
    aux = {
        "native_dominant_hz": dom,
        "shifted_dominant_hz": dom * speed,
        "speed": speed,
        "energy": float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0,
        "range_m": s.meta.get("range_m"),
        "azimuth_deg": s.meta.get("azimuth_deg", 0.0),
    }
    return Drive(_norm(y), fs_out, aux)


def couple_ultrasound(s: WaveSample, fs_out: float, center: float = 1600.0) -> Drive:
    """Heterodyne (bat-detector) down-conversion around the carrier."""
    carrier = s.carrier or _dominant_freq(np.real(s.data), s.sample_rate) or 40000.0
    t = np.arange(s.n_samples) / s.sample_rate
    ana = s.data if s.is_complex else hilbert(np.real(s.data).astype(float))
    shifted = ana * np.exp(-2j * np.pi * (carrier - center) * t)
    base = np.real(shifted)
    y = _to_audio_rate(base, s.sample_rate, fs_out)
    aux = {
        "carrier_hz": carrier,
        "shift_center_hz": center,
        "range_m": s.meta.get("range_m"),
        "azimuth_deg": s.meta.get("azimuth_deg", 0.0),
        "elevation_deg": s.meta.get("elevation_deg", 0.0),
        "n_echoes": s.meta.get("n_echoes"),
    }
    return Drive(_norm(y), fs_out, aux)


def couple_radar(s: WaveSample, fs_out: float, speed: float = 1.0) -> Drive:
    """Coherent baseband -> Doppler drive (Doppler is usually already audio)."""
    if s.is_complex:
        base = np.real(s.data).astype(float)
        doppler = _dominant_freq(np.real(s.data), s.sample_rate)
    else:
        base = np.real(s.data).astype(float)
        doppler = _dominant_freq(base, s.sample_rate)
    n_out = int(round(len(base) * fs_out / (s.sample_rate * speed)))
    y = _resample_len(base, n_out)
    aux = {
        "doppler_hz": doppler * speed,
        "prf_hz": s.sample_rate,
        "range_m": s.meta.get("range_m"),
        "azimuth_deg": s.meta.get("azimuth_deg", 0.0),
        "elevation_deg": s.meta.get("elevation_deg", 0.0),
        "radial_velocity_ms": s.meta.get("radial_velocity_ms"),
    }
    return Drive(_norm(y), fs_out, aux)


def couple_radio(s: WaveSample, fs_out: float) -> Drive:
    """Demodulate the message (AM envelope or FM discriminator)."""
    mod = str(s.meta.get("mod", "FM")).upper()
    ana = s.data if s.is_complex else hilbert(np.real(s.data).astype(float))
    if mod == "AM":
        msg = np.abs(ana)
        msg = msg - np.mean(msg)
    else:  # FM
        phase = np.unwrap(np.angle(ana))
        msg = np.diff(phase, prepend=phase[:1])
        msg = msg - np.mean(msg)
    y = _to_audio_rate(msg, s.sample_rate, fs_out)
    aux = {
        "carrier_hz": s.carrier,
        "mod": mod,
        "bandwidth_hz": s.meta.get("bandwidth_hz"),
        "modulation_index": s.meta.get("modulation_index"),
        "azimuth_deg": s.meta.get("azimuth_deg", 0.0),
    }
    return Drive(_norm(y), fs_out, aux)


def couple_infrared(s: WaveSample, fs_out: float, speed: float = 6.0) -> Drive:
    """Thermal intensity compressed into the audio band; temperature -> colour."""
    x = np.real(s.data).astype(float)
    x = x - np.mean(x)
    n_out = int(round(len(x) * fs_out / (s.sample_rate * speed)))
    y = _resample_len(x, n_out)
    temp = s.meta.get("temperature_k")
    aux = {
        "temperature_k": temp,
        "speed": speed,
        "emissivity": s.meta.get("emissivity"),
        "bands": s.meta.get("bands"),
        "azimuth_deg": s.meta.get("azimuth_deg", 0.0),
    }
    return Drive(_norm(y), fs_out, aux)


def couple_gamma(s: WaveSample, fs_out: float, duration: float | None = None) -> Drive:
    """Photon events -> an energy-tuned impulse train (each photon a tiny ping).

    Energy sets the pitch of the impulse, so the *cochlea* naturally sorts
    photons by energy -- no explicit spectral analysis required.
    """
    energies = np.real(s.data).astype(float)
    times = np.asarray(s.meta.get("event_times", np.arange(len(energies)) / max(s.sample_rate, 1e-9)), float)
    if times.size == 0:
        return Drive(np.zeros(int(fs_out * 0.5)), fs_out, {"n_events": 0})
    span = float(times.max() - times.min()) or 1.0
    dur = duration if duration else min(8.0, max(1.5, span))
    t0 = times.min()
    out_len = int(fs_out * dur)
    y = np.zeros(out_len)
    e_lo, e_hi = float(np.min(energies)), float(np.max(energies))
    e_span = (e_hi - e_lo) or 1.0
    tau = 0.006
    tvec = np.arange(int(0.05 * fs_out)) / fs_out
    for e, tt in zip(energies, times):
        # log-energy -> pitch in a pleasant band
        frac = (np.log1p(max(e, 1e-6)) - np.log1p(max(e_lo, 1e-6))) / (
            np.log1p(e_hi) - np.log1p(max(e_lo, 1e-6)) or 1.0
        )
        f = 320.0 * (4000.0 / 320.0) ** np.clip(frac, 0, 1)
        idx = int(((tt - t0) / (span)) * (out_len - len(tvec) - 1))
        idx = max(0, min(idx, out_len - len(tvec) - 1))
        ping = np.exp(-tvec / tau) * np.sin(2 * np.pi * f * tvec)
        amp = 0.4 + 0.6 * np.sqrt(max(e - e_lo, 0.0) / e_span)
        y[idx : idx + len(tvec)] += amp * ping
    aux = {
        "n_events": int(len(energies)),
        "event_rate_hz": float(len(energies) / span),
        "mean_energy": float(np.mean(energies)),
        "energy_spread": float(np.std(energies)),
    }
    return Drive(_norm(y), fs_out, aux)


def couple_audio(s: WaveSample, fs_out: float) -> Drive:
    """Ordinary sound is already in the ear's range -- pass it through directly."""
    x = np.real(s.data).astype(float)
    y = _to_audio_rate(x, s.sample_rate, fs_out)
    aux = {
        "dominant_hz": _dominant_freq(x, s.sample_rate),
        "azimuth_deg": s.meta.get("azimuth_deg", 0.0),
        "elevation_deg": s.meta.get("elevation_deg"),
    }
    return Drive(_norm(y), fs_out, aux)


_COUPLERS = {
    "seismic": couple_seismic,
    "ultrasound": couple_ultrasound,
    "radar": couple_radar,
    "radio": couple_radio,
    "infrared": couple_infrared,
    "gamma": couple_gamma,
    "audio": couple_audio,
}


class TransductionFrontEnd:
    """Dispatches a :class:`WaveSample` to its physical coupling."""

    def __init__(self, fs_out: float = 44100.0, overrides: dict | None = None):
        self.fs_out = float(fs_out)
        self.overrides = overrides or {}

    def couple(self, sample: WaveSample) -> Drive:
        fn = _COUPLERS[sample.wave_type]
        kwargs = self.overrides.get(sample.wave_type, {})
        drive = fn(sample, self.fs_out, **kwargs)
        drive.aux.setdefault("wave_type", sample.wave_type)
        drive.aux.setdefault("label", sample.label())
        return drive
