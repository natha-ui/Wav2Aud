"""Bring your own waveform.

Turn a user-supplied waveform into a :class:`~wave2aud.waves.WaveSample` and an
interactive 3-D "experience" page that shows the wave with artistic flourish
while playing its sonification. Two input modes:

* :func:`from_wav`   -- a ``.wav`` file (its samples are the waveform).
* :func:`from_image` -- a picture of a waveform (a line/curve is traced out of
  the image, background auto-detected, works for dark-on-light or light-on-dark).

The user must declare which of the six wave categories it is (radar, radio,
infrared, ultrasound, gamma, seismic), because that fixes the physical coupling
and the sonic identity.

:func:`render_experience` writes a self-contained HTML file (embedding the audio
and the trace) that renders a rotating 3-D delay-embedding "attractor" of the
wave, colour-graded in the category hue, with a playhead synced to the audio.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.io import wavfile

from .audio_io import wav_datauri
from .waves import WaveSample, WAVE_TYPES

# native sample rate assumed for an *image* trace, per modality, so the coupling
# shifts it into the audio band sensibly (a picture carries no timing of its own)
NATIVE_FS = {"seismic": 200.0, "radar": 8000.0, "radio": 8000.0,
             "infrared": 100.0, "ultrasound": 96000.0, "gamma": 60.0, "audio": 16000.0}

_TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "docs", "_experience_template.html")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _check_type(wave_type: str) -> str:
    if wave_type not in WAVE_TYPES:
        raise ValueError(f"wave_type must be one of {WAVE_TYPES}, got {wave_type!r}")
    return wave_type


def _to_float(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.int16:
        return data.astype(np.float64) / 32768.0
    if data.dtype == np.int32:
        return data.astype(np.float64) / 2147483648.0
    if data.dtype == np.uint8:
        return (data.astype(np.float64) - 128.0) / 128.0
    return data.astype(np.float64)


def _resample_trace(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, float)
    if len(x) == n:
        return x
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)


def _sample_from_trace(trace, fs, wave_type, carrier=None, meta=None) -> WaveSample:
    """Wrap a 1-D trace as a WaveSample (gamma -> derive photon events)."""
    meta = dict(meta or {})
    trace = np.asarray(trace, float)
    if wave_type == "gamma":
        from scipy.signal import find_peaks
        mag = np.abs(trace)
        height = 0.2 * (mag.max() or 1.0)
        peaks, _ = find_peaks(mag, height=height, distance=max(1, len(mag) // 200))
        if peaks.size == 0:
            peaks = np.arange(0, len(mag), max(1, len(mag) // 40))
        times = peaks / fs
        energies = 150.0 + 1300.0 * (mag[peaks] / (mag.max() + 1e-9))
        meta.setdefault("event_times", times)
        return WaveSample(energies, fs, "gamma", meta=meta)
    return WaveSample(trace, fs, wave_type, carrier, meta=meta)


# ---------------------------------------------------------------------------
# ingestion
# ---------------------------------------------------------------------------
def from_wav(path: str, wave_type: str, carrier: float | None = None,
             meta: dict | None = None, max_seconds: float | None = 10.0) -> WaveSample:
    """Load a ``.wav`` and label it as ``wave_type``."""
    _check_type(wave_type)
    fs, data = wavfile.read(path)
    data = _to_float(np.asarray(data))
    if data.ndim == 2:
        data = data.mean(axis=1)
    if max_seconds:
        data = data[: int(max_seconds * fs)]
    meta = dict(meta or {})
    meta.setdefault("label", os.path.basename(path))
    meta.setdefault("source", "wav")
    return _sample_from_trace(data, float(fs), wave_type, carrier, meta)


def trace_from_image(path: str, n_out: int = 1600, smooth: int = 3) -> np.ndarray:
    """Extract a 1-D waveform from a picture of one (auto background detection)."""
    from PIL import Image

    img = Image.open(path).convert("L")
    a = np.asarray(img, dtype=float)          # [H, W]
    if a.ndim != 2 or a.size == 0:
        raise ValueError("could not read a 2-D image")
    bg = np.median(a)
    dev0 = np.abs(a - bg)
    thr0 = 0.35 * dev0.max()
    # crop to the bounding box of the ink so margins/axes don't distort the x-axis
    ink = dev0 > thr0
    cols = np.where(ink.any(axis=0))[0]
    rowsi = np.where(ink.any(axis=1))[0]
    if cols.size >= 2 and rowsi.size >= 2:
        a = a[rowsi[0]:rowsi[-1] + 1, cols[0]:cols[-1] + 1]
    h, w = a.shape
    bg = np.median(a)
    dev = np.abs(a - bg)                       # ink stands out from background
    thr = 0.35 * dev.max()
    rows = np.arange(h)[:, None]
    mask = dev > thr
    colw = np.where(mask, dev, 0.0)
    denom = colw.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        centroid = (colw * rows).sum(axis=0) / denom
    good = denom > 0
    xs = np.arange(w)
    if good.sum() < 2:
        y = np.zeros(w)
    else:
        y = np.interp(xs, xs[good], centroid[good])
    y = -(y - np.nanmean(y))                    # image y is downward; flip
    if smooth > 1:
        k = np.ones(smooth) / smooth
        y = np.convolve(y, k, mode="same")
    y = y / (np.max(np.abs(y)) + 1e-9)
    return _resample_trace(y, n_out)


def from_image(path: str, wave_type: str, sample_rate: float | None = None,
               carrier: float | None = None, meta: dict | None = None) -> WaveSample:
    """Trace a waveform out of an image and label it as ``wave_type``."""
    _check_type(wave_type)
    trace = trace_from_image(path)
    fs = float(sample_rate or NATIVE_FS[wave_type])
    meta = dict(meta or {})
    meta.setdefault("label", os.path.basename(path))
    meta.setdefault("source", "image")
    return _sample_from_trace(trace, fs, wave_type, carrier, meta)


def load(path: str, wave_type: str, **kw) -> WaveSample:
    """Dispatch by file extension: ``.wav`` -> :func:`from_wav`, else image."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".wav", ".wave"):
        return from_wav(path, wave_type, **kw)
    return from_image(path, wave_type, **kw)


# ---------------------------------------------------------------------------
# experience (3-D visualisation + audio)
# ---------------------------------------------------------------------------
def _display_trace(sample: WaveSample, n: int = 1400) -> np.ndarray:
    if sample.wave_type == "gamma":
        times = np.asarray(sample.meta.get("event_times", []), float)
        energies = np.real(sample.data).astype(float)
        y = np.zeros(n)
        if times.size:
            span = float(times.max() - times.min()) or 1.0
            for tt, e in zip(times, energies):
                idx = int((tt - times.min()) / span * (n - 1))
                y[idx] = max(y[idx], e)
        y = y / (y.max() + 1e-9)
        return y
    raw = np.real(sample.data).astype(float)
    raw = raw / (np.max(np.abs(raw)) + 1e-9)
    return _resample_trace(raw, n)


def build_experience_data(sample: WaveSample, fs: float = 44100.0, trace_points: int = 1400) -> dict:
    """Everything the 3-D page needs: trace, audio, features."""
    from .pipeline import Sonifier

    res = Sonifier(fs=fs).sonify(sample)
    trace = _display_trace(sample, trace_points)
    f = res.features
    return {
        "wave_type": sample.wave_type,
        "label": sample.label(),
        "source": sample.meta.get("source", "array"),
        "trace": [round(float(v), 4) for v in trace],
        "audio": wav_datauri(res.audio, fs),
        "duration": round(float(res.audio.shape[0] / fs), 3),
        "features": {
            "brightness": round(float(f.brightness), 3),
            "loudness": round(float(f.loudness), 3),
            "centroid_hz": round(float(f.centroid_hz), 1),
            "onset_rate": round(float(f.onset_rate), 2),
            "flatness": round(float(f.flatness), 3),
            "tempo_bpm": round(float(f.tempo_bpm), 1),
        },
    }


def render_experience(sample: WaveSample, out_html: str, fs: float = 44100.0,
                      template: str | None = None) -> str:
    """Write a self-contained 3-D experience page for ``sample``."""
    data = build_experience_data(sample, fs=fs)
    tpl_path = template or _TEMPLATE
    with open(tpl_path, encoding="utf-8") as fh:
        tpl = fh.read()
    html = tpl.replace("__EXPERIENCE_JSON__", json.dumps(data))
    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out_html


def experience_from_wav(path, wave_type, out_html, fs=44100.0, **kw):
    return render_experience(from_wav(path, wave_type, **kw), out_html, fs=fs)


def experience_from_image(path, wave_type, out_html, fs=44100.0, **kw):
    return render_experience(from_image(path, wave_type, **kw), out_html, fs=fs)
