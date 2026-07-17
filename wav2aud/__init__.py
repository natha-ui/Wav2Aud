"""wav2aud -- biomimetic sonification of non-audio waves into semi-musical audio.

Quick start::

    import wav2aud as w2a
    sample = w2a.simulate.preset("radar_car")
    result = w2a.sonify(sample)
    result.write("radar_car.wav")

Pipeline: transduction -> biomimetic ear (cochlea + hair cells) -> perceptual
features -> category music mapping -> synthesis.
"""
from __future__ import annotations

from .waves import WaveSample, WAVE_TYPES
from .transduction import Drive, TransductionFrontEnd
from .cochlea import GammatoneFilterbank, AntennaResonator
from .haircell import HairCellTransduction, Neurogram
from .features import WaveFeatures, extract_features
from .mapping import AudioParameters, CategoryVoice, CATEGORY_VOICES, map_features
from .synthesis import render
from .ear import BiomimeticEar, AuditoryImage
from .pipeline import Sonifier, StreamingSonifier, SonifyResult, sonify
from .realtime import RealtimeSonifier, chunk_sample
from .audio_io import write_wav, read_wav
from . import simulate
from . import sources
from . import metrics
from . import geometry
from . import baseline
from . import ingest
from . import live

__version__ = "0.1.0"

__all__ = [
    "WaveSample", "WAVE_TYPES", "Drive", "TransductionFrontEnd",
    "GammatoneFilterbank", "AntennaResonator", "HairCellTransduction", "Neurogram",
    "WaveFeatures", "extract_features", "AudioParameters", "CategoryVoice",
    "CATEGORY_VOICES", "map_features", "render", "BiomimeticEar", "AuditoryImage",
    "Sonifier", "StreamingSonifier", "SonifyResult", "sonify",
    "RealtimeSonifier", "chunk_sample",
    "write_wav", "read_wav", "simulate", "sources",
    "metrics", "geometry", "baseline", "ingest", "live", "__version__",
]
