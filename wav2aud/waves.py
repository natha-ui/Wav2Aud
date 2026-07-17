"""Wave data model.

A :class:`WaveSample` is the *raw physical observable* delivered by a sensor:
Doppler/IQ for radar, a demodulatable RF frame for radio, an intensity/time
series for infrared, a heterodyne band for ultrasound, a photon event list for
gamma, and a ground-velocity trace for seismic.

Crucially, the sample carries the *native* physics (its true sample rate,
carrier frequency and any side channels such as target range or arrival angle).
The biomimetic ear never assumes the data is already audio -- the transduction
front-end (:mod:`wav2aud.transduction`) is responsible for coupling each
physical modality into the ear's audio-rate mechanical drive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

#: Canonical wave categories understood by the package.
WAVE_TYPES = (
    "radar",       # microwave / radar returns (Doppler + range)
    "radio",       # modulated RF (AM/FM baseband recoverable)
    "infrared",    # thermal / IR intensity & emissivity bands
    "ultrasound",  # acoustic > 20 kHz, echolocation-style echoes
    "gamma",       # discrete high-energy photon events
    "seismic",     # ground motion, sub-audio to a few tens of Hz
    "audio",       # ordinary audible sound (already in the ear's range)
)


@dataclass
class WaveSample:
    """A raw slice of a physical wave, as produced by a sensor.

    Parameters
    ----------
    data:
        The raw samples. May be real or complex (complex is used for IQ radar
        / heterodyned ultrasound). For ``gamma`` this holds photon *energies*
        indexed by ``meta['event_times']``.
    sample_rate:
        Native sampling rate of ``data`` in Hz (the *true* physical rate, e.g.
        250 Hz for a geophone, 2e6 for an ultrasound ADC).
    wave_type:
        One of :data:`WAVE_TYPES`.
    carrier:
        Carrier / centre frequency in Hz where meaningful (radar, radio,
        ultrasound). ``None`` otherwise.
    duration:
        Convenience: physical duration in seconds. Auto-filled if omitted.
    meta:
        Modality-specific side channels. Common keys:
        ``range_m`` (float or array), ``azimuth_deg``, ``elevation_deg``,
        ``event_times`` (gamma), ``temperature_k`` (infrared),
        ``label`` (free text).
    """

    data: np.ndarray
    sample_rate: float
    wave_type: str
    carrier: Optional[float] = None
    duration: Optional[float] = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data)
        if self.wave_type not in WAVE_TYPES:
            raise ValueError(
                f"unknown wave_type {self.wave_type!r}; expected one of {WAVE_TYPES}"
            )
        if self.duration is None:
            n = self.data.shape[-1] if self.data.ndim else len(self.data)
            self.duration = float(n) / float(self.sample_rate) if self.sample_rate else 0.0

    # -- small conveniences -------------------------------------------------
    @property
    def n_samples(self) -> int:
        return int(self.data.shape[-1]) if self.data.ndim else len(self.data)

    @property
    def is_complex(self) -> bool:
        return np.iscomplexobj(self.data)

    def label(self) -> str:
        return str(self.meta.get("label", self.wave_type))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        c = f", carrier={self.carrier:g}Hz" if self.carrier else ""
        return (
            f"WaveSample({self.wave_type}, n={self.n_samples}, "
            f"fs={self.sample_rate:g}Hz{c}, dur={self.duration:.3g}s)"
        )
