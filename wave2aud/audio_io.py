"""Minimal WAV I/O (stdlib scipy only, no soundfile dependency)."""
from __future__ import annotations

import base64
import io
import wave

import numpy as np
from scipy.io import wavfile


def write_wav(path: str, stereo: np.ndarray, fs: float) -> str:
    stereo = np.asarray(stereo, dtype=float)
    if stereo.ndim == 1:
        stereo = np.stack([stereo, stereo], axis=1)
    stereo = np.clip(stereo, -1.0, 1.0)
    ints = (stereo * 32767.0).astype(np.int16)
    wavfile.write(path, int(round(fs)), ints)
    return path


def wav_datauri(audio: np.ndarray, fs: float, out_fs: int = 22050) -> str:
    """Encode audio as a base64 ``data:audio/wav`` URI (mono, downsampled).

    Handy for embedding a clip directly into a self-contained HTML page.
    """
    from math import gcd
    from scipy.signal import resample_poly

    mono = audio.mean(axis=1) if np.asarray(audio).ndim == 2 else np.asarray(audio, float)
    up, down = int(out_fs), int(round(fs))
    g = gcd(up, down) or 1
    mono = resample_poly(mono, up // g, down // g)
    peak = float(np.max(np.abs(mono))) or 1.0
    ints = (np.clip(mono / peak * 0.97, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(out_fs)
        wf.writeframes(ints.tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def read_wav(path: str):
    fs, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)
    return data, float(fs)
