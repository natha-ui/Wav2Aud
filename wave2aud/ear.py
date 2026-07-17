"""The biomimetic ear: cochlea + hair cells + neural feature read-out.

Combines the sub-models into one organ and adds two pieces of cross-species
biomimicry that switch on per wave category:

* a **bat acoustic fovea** -- extra frequency selectivity over a chosen band
  (used for ultrasound / echolocation), and
* an **insect antenna resonance** -- a mechanical pre-emphasis applied to the
  drive before the cochlea (used where a sharp mechanical resonance aids
  detection).

``listen()`` returns an :class:`AuditoryImage`: the neurogram, the perceptual
features, and the control-rate channel envelopes the synthesiser resynthesises.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cochlea import AntennaResonator, GammatoneFilterbank, Cochleagram
from .features import WaveFeatures, extract_features
from .haircell import HairCellTransduction, Neurogram, to_control_rate
from .transduction import Drive

# per-category ear specialisations (biomimicry that varies by modality)
CATEGORY_EAR = {
    "ultrasound": {"sharpen_band": (1200.0, 3200.0), "sharpen": 2.5,
                   "antenna": (1900.0, 6.0, 0.35)},   # bat fovea + JO resonance
    "radar": {"antenna": (1300.0, 4.0, 0.30)},
    "gamma": {"sharpen_band": (2000.0, 5000.0), "sharpen": 1.8},
    "seismic": {"f_lo": 40.0},
}


@dataclass
class AuditoryImage:
    neurogram: Neurogram
    features: WaveFeatures
    envelopes: np.ndarray      # [n_channels, n_frames] at control_rate
    cf: np.ndarray
    control_rate: float
    drive: Drive


class BiomimeticEar:
    """A configurable cochlea/hair-cell ear with per-category specialisations."""

    def __init__(
        self,
        fs: float = 44100.0,
        n_channels: int = 36,
        f_lo: float = 45.0,
        f_hi: float | None = None,
        control_rate: float = 250.0,
        compression: float = 0.35,
    ):
        self.fs = float(fs)
        self.n_channels = int(n_channels)
        self.f_lo = float(f_lo)
        self.f_hi = f_hi
        self.control_rate = float(control_rate)
        self.hair = HairCellTransduction(fs, compression=compression)
        self._banks: dict = {}

    def _bank_for(self, wave_type: str) -> GammatoneFilterbank:
        cfg = CATEGORY_EAR.get(wave_type, {})
        key = (
            cfg.get("f_lo", self.f_lo),
            cfg.get("sharpen_band"),
            cfg.get("sharpen", 1.0),
        )
        if key not in self._banks:
            self._banks[key] = GammatoneFilterbank(
                self.fs,
                f_lo=cfg.get("f_lo", self.f_lo),
                f_hi=self.f_hi,
                n_channels=self.n_channels,
                sharpen_band=cfg.get("sharpen_band"),
                sharpen=cfg.get("sharpen", 1.0),
            )
        return self._banks[key]

    def listen(self, drive: Drive) -> AuditoryImage:
        wave_type = drive.aux.get("wave_type", "radio")
        signal = np.asarray(drive.signal, float)

        # insect-antenna mechanical pre-emphasis (if configured)
        cfg = CATEGORY_EAR.get(wave_type, {})
        if "antenna" in cfg:
            f0, q, gain = cfg["antenna"]
            signal = AntennaResonator(self.fs, f0, q, gain).process(signal)

        bank = self._bank_for(wave_type)
        coch: Cochleagram = bank.process(signal)
        neuro: Neurogram = self.hair.process(coch)

        envelopes, cr = to_control_rate(neuro, self.control_rate)
        feats: WaveFeatures = extract_features(neuro, aux=drive.aux, control_rate=self.control_rate)

        return AuditoryImage(
            neurogram=neuro,
            features=feats,
            envelopes=envelopes,
            cf=neuro.cf,
            control_rate=cr,
            drive=drive,
        )
