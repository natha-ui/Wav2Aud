"""High-level API: turn a wave into music in one call.

``Sonifier`` chains the four stages -- transduction -> biomimetic ear ->
feature/music mapping -> synthesis -- and keeps every intermediate so the
visualisation layer can show exactly how a wave became sound.

``StreamingSonifier`` wraps the same pipeline for a live sensor feed: push
chunks in, get cross-faded audio blocks out (offline core, streaming-ready).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio_io import write_wav
from .ear import AuditoryImage, BiomimeticEar
from .features import WaveFeatures
from .mapping import AudioParameters, map_features
from .synthesis import render
from .transduction import Drive, TransductionFrontEnd
from .waves import WaveSample


@dataclass
class SonifyResult:
    sample: WaveSample
    drive: Drive
    image: AuditoryImage
    params: AudioParameters
    audio: np.ndarray          # [n, 2] float32
    fs: float

    @property
    def features(self) -> WaveFeatures:
        return self.image.features

    @property
    def natural_audio(self) -> np.ndarray:
        """The coupled wave as plain audio (the 'natural' counterpart to the music)."""
        d = np.asarray(self.drive.signal, float)
        peak = np.max(np.abs(d)) or 1.0
        d = (d / peak * 0.9).astype(np.float32)
        return np.stack([d, d], axis=1)

    def write(self, path: str) -> str:
        return write_wav(path, self.audio, self.fs)

    def write_natural(self, path: str) -> str:
        """Write the wave heard *naturally* (coupled drive) to a WAV file."""
        return write_wav(path, self.natural_audio, self.drive.fs)


class Sonifier:
    def __init__(self, fs: float = 44100.0, ear: BiomimeticEar | None = None,
                 overrides: dict | None = None):
        self.fs = float(fs)
        self.front_end = TransductionFrontEnd(fs_out=fs, overrides=overrides)
        self.ear = ear or BiomimeticEar(fs=fs)

    def sonify(self, sample: WaveSample) -> SonifyResult:
        drive = self.front_end.couple(sample)
        image = self.ear.listen(drive)
        params = map_features(image.features)
        audio = render(params, image.envelopes, image.cf, image.control_rate, self.fs)
        return SonifyResult(sample, drive, image, params, audio, self.fs)

    def to_wav(self, sample: WaveSample, path: str) -> SonifyResult:
        res = self.sonify(sample)
        res.write(path)
        return res


def sonify(sample: WaveSample, fs: float = 44100.0) -> SonifyResult:
    """One-shot convenience wrapper."""
    return Sonifier(fs=fs).sonify(sample)


class StreamingSonifier:
    """Real-time-style block processor with equal-power cross-fades.

    Feed successive :class:`WaveSample` chunks (same ``wave_type``); each call
    returns a stereo block whose head is cross-faded with the previous block's
    tail to avoid seams. Suitable for driving a live audio callback or a
    robot's continuous perception stream.
    """

    def __init__(self, fs: float = 44100.0, crossfade: float = 0.08,
                 ear: BiomimeticEar | None = None, overrides: dict | None = None):
        self.sonifier = Sonifier(fs=fs, ear=ear, overrides=overrides)
        self.fs = float(fs)
        self.xf = int(crossfade * fs)
        self._tail = None

    def push(self, sample: WaveSample) -> np.ndarray:
        block = self.sonifier.sonify(sample).audio
        if self._tail is not None and self.xf > 0 and len(block) > self.xf:
            fade = np.linspace(0, 1, self.xf)[:, None]
            head = block[: self.xf]
            tail = self._tail[: self.xf]
            m = min(len(head), len(tail))
            block[:m] = tail[:m] * (1 - fade[:m]) + head[:m] * fade[:m]
        self._tail = block[-self.xf:] if self.xf > 0 else None
        return block
