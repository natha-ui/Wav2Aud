"""Real-time conversion: stream a sensor straight to the speakers.

This is the thin layer that turns :class:`~wave2aud.realtime.RealtimeSonifier`
(which converts wave chunks into seamless audio blocks) into an actual live
instrument: chunks are pulled from any :class:`~wave2aud.sources.WaveSource`,
sonified, and written to an audio output device.

Audio output needs the optional ``sounddevice`` dependency (a PortAudio
binding)::

    pip install "wave2aud[realtime]"

Everything here degrades gracefully: :func:`available` reports whether output is
possible, and the import is deferred so the rest of the package works without it.

Example::

    from wave2aud import live
    from wave2aud.sources import SimulatedSource

    live.stream(SimulatedSource("radar"), n_chunks=10)     # to the speakers

    # or your own hardware, via the WaveSource interface:
    live.stream(CallbackSource("radar", sdr.read_samples, sample_rate=2.4e6))
"""
from __future__ import annotations

import numpy as np

from .realtime import RealtimeSonifier
from .waves import WaveSample


def available() -> bool:
    """True if live audio output is usable (``sounddevice`` importable)."""
    try:
        import sounddevice  # noqa: F401
        return True
    except Exception:
        return False


def list_devices():
    """Return the available audio output devices (requires ``sounddevice``)."""
    import sounddevice as sd
    return sd.query_devices()


def _require_sounddevice():
    try:
        import sounddevice as sd
        return sd
    except Exception as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "Live audio output needs the optional 'sounddevice' dependency.\n"
            "Install it with:  pip install \"wave2aud[realtime]\"\n"
            "(sounddevice needs PortAudio; on Linux: apt install libportaudio2)"
        ) from exc


def _to_stereo(mono: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = np.asarray(mono, dtype=np.float32)
    hi = float(np.max(np.abs(m))) or 1.0
    m = m * (peak / hi)
    return np.stack([m, m], axis=1).astype(np.float32)


class LiveSonifier:
    """An open audio stream that sonifies wave chunks as they arrive.

    Parameters
    ----------
    fs:
        Output sample rate (Hz).
    mode:
        ``"musical"`` plays the sonification; ``"natural"`` plays the coupled
        wave itself (the raw wave brought into the audible range).
    device:
        Output device index/name, or ``None`` for the system default. See
        :func:`list_devices`.
    blocksize:
        PortAudio block size; ``0`` lets the backend choose.
    """

    def __init__(self, fs: float = 44100.0, mode: str = "musical",
                 device=None, blocksize: int = 0):
        if mode not in ("musical", "natural"):
            raise ValueError("mode must be 'musical' or 'natural'")
        self.fs = float(fs)
        self.mode = mode
        self.device = device
        self.blocksize = int(blocksize)
        self.engine = RealtimeSonifier(fs=fs)
        self._stream = None

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> "LiveSonifier":
        sd = _require_sounddevice()
        self._stream = sd.OutputStream(
            samplerate=self.fs, channels=2, dtype="float32",
            device=self.device, blocksize=self.blocksize,
        )
        self._stream.start()
        return self

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self) -> "LiveSonifier":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # -- streaming ---------------------------------------------------------
    def render(self, sample: WaveSample) -> np.ndarray:
        """Sonify one chunk and return the stereo block (without playing it)."""
        block = self.engine.process(sample)
        if self.mode == "natural":
            drive = getattr(self.engine, "last_drive", None)
            if drive is not None and len(drive) > 1:
                block = _to_stereo(drive)
        return block

    def push(self, sample: WaveSample) -> np.ndarray:
        """Sonify one chunk and write it to the audio device."""
        block = self.render(sample)
        if self._stream is None:
            raise RuntimeError("stream is not open - use `with LiveSonifier(...)` or call open()")
        self._stream.write(np.ascontiguousarray(block, dtype=np.float32))
        return block

    def run(self, source, n_chunks: int | None = None) -> int:
        """Pull chunks from a :class:`~wave2aud.sources.WaveSource` and play them.

        Returns the number of chunks played. Blocks until the source is
        exhausted or ``n_chunks`` have been played.
        """
        played = 0
        for sample in source.stream(n_chunks):
            self.push(sample)
            played += 1
        return played


def stream(source, fs: float = 44100.0, mode: str = "musical",
           device=None, n_chunks: int | None = None) -> int:
    """Open a device, sonify ``source`` live, and close it again."""
    with LiveSonifier(fs=fs, mode=mode, device=device) as live:
        return live.run(source, n_chunks=n_chunks)
