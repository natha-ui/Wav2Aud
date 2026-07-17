"""Sensor abstraction layer.

A :class:`WaveSource` is a hardware-agnostic producer of
:class:`~wave2aud.waves.WaveSample` chunks. The rest of the package depends only
on this interface, so a simulated source, a file, a NumPy array, or a real
driver (RTL-SDR, geophone DAQ, ultrasonic ranger, gamma spectrometer, thermal
camera) are all interchangeable. Concrete hardware drivers implement
``read()``; stubs below show the contract and provide fully working simulated
and in-memory sources for development and CI.
"""
from __future__ import annotations

from typing import Callable, Iterator, Optional

import numpy as np

from .waves import WaveSample
from . import simulate as _sim


class WaveSource:
    """Abstract sensor. Implement :meth:`read` (one chunk) or :meth:`stream`."""

    wave_type: str = "radio"

    def read(self) -> Optional[WaveSample]:
        raise NotImplementedError

    def stream(self, n_chunks: Optional[int] = None) -> Iterator[WaveSample]:
        i = 0
        while n_chunks is None or i < n_chunks:
            sample = self.read()
            if sample is None:
                return
            yield sample
            i += 1

    def close(self) -> None:  # pragma: no cover - trivial
        pass


class ArraySource(WaveSource):
    """Wrap an in-memory array (or a list of arrays for streaming)."""

    def __init__(self, data, sample_rate, wave_type, carrier=None, meta=None, chunks=None):
        self.wave_type = wave_type
        self._chunks = list(chunks) if chunks is not None else [np.asarray(data)]
        self._sr = sample_rate
        self._carrier = carrier
        self._meta = meta or {}
        self._i = 0

    def read(self) -> Optional[WaveSample]:
        if self._i >= len(self._chunks):
            return None
        d = self._chunks[self._i]
        self._i += 1
        return WaveSample(d, self._sr, self.wave_type, self._carrier, meta=dict(self._meta))


class SimulatedSource(WaveSource):
    """A source backed by :mod:`wave2aud.simulate` (great for demos/CI/robots)."""

    def __init__(self, wave_type: str, n_chunks: Optional[int] = None, **kwargs):
        self.wave_type = wave_type
        self._gen = {
            "seismic": _sim.simulate_seismic,
            "radar": _sim.simulate_radar,
            "radio": _sim.simulate_radio,
            "infrared": _sim.simulate_infrared,
            "ultrasound": _sim.simulate_ultrasound,
            "gamma": _sim.simulate_gamma,
            "audio": _sim.simulate_audio,
        }[wave_type]
        self._kwargs = kwargs
        self._n = n_chunks
        self._i = 0

    def read(self) -> Optional[WaveSample]:
        if self._n is not None and self._i >= self._n:
            return None
        kw = dict(self._kwargs)
        if "seed" in self._gen.__code__.co_varnames:
            kw.setdefault("seed", self._i)  # vary chunks over time
        self._i += 1
        return self._gen(**kw)


class CallbackSource(WaveSource):
    """Adapter around any user callable returning arrays -- e.g. an SDR driver.

    Example::

        src = CallbackSource("radar", lambda: sdr.read_samples(4096),
                             sample_rate=2.4e6, carrier=10.5e9)
    """

    def __init__(self, wave_type, fn: Callable[[], Optional[np.ndarray]],
                 sample_rate, carrier=None, meta=None):
        self.wave_type = wave_type
        self._fn = fn
        self._sr = sample_rate
        self._carrier = carrier
        self._meta = meta or {}

    def read(self) -> Optional[WaveSample]:
        d = self._fn()
        if d is None:
            return None
        return WaveSample(d, self._sr, self.wave_type, self._carrier, meta=dict(self._meta))


# ---- driver stubs (document the hardware contract; not exercised in CI) -----
class RTLSDRSource(CallbackSource):  # pragma: no cover - needs hardware
    """RTL-SDR radio/radar front-end. Requires ``pyrtlsdr``.

    Fill in ``_open`` with real device setup; ``read`` returns complex IQ.
    """

    def __init__(self, center_hz=98.5e6, sample_rate=2.4e6, wave_type="radio", meta=None):
        self._sdr = None
        super().__init__(wave_type, self._grab, sample_rate, center_hz, meta)

    def _open(self):
        from rtlsdr import RtlSdr  # type: ignore

        self._sdr = RtlSdr()
        self._sdr.sample_rate = self._sr
        self._sdr.center_freq = self._carrier
        self._sdr.gain = "auto"

    def _grab(self):
        if self._sdr is None:
            self._open()
        return self._sdr.read_samples(4096)

    def close(self):
        if self._sdr is not None:
            self._sdr.close()
