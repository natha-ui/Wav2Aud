"""Feature -> music mapping, with a fixed sonic identity per wave category.

The mapping has two layers:

1. **Category identity (fixed).** Each wave type owns a register band, a
   musical scale, a partial recipe (timbre family), an articulation
   (envelope) and a spatial/effect signature. This is what makes radar always
   sound like shimmering glass bells and seismic always sound like a deep
   bowed drone -- the *category fingerprint*.

2. **Within-category modulation (continuous).** The extracted
   :class:`~wave2aud.features.WaveFeatures` move parameters *within* the
   category's envelope. Because the mapping is deterministic and continuous,
   two similar waves land on nearly identical parameters (small audible
   difference) while very different waves spread far apart (large audible
   difference) -- yet both keep the category fingerprint.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import WaveFeatures

# --- musical scales (semitone classes) -------------------------------------
MAJOR = (0, 2, 4, 5, 7, 9, 11)
MINOR = (0, 2, 3, 5, 7, 8, 10)
PENTATONIC = (0, 2, 4, 7, 9)
WHOLE_TONE = (0, 2, 4, 6, 8, 10)
LYDIAN = (0, 2, 4, 6, 7, 9, 11)


@dataclass
class AudioParameters:
    """Everything the synthesiser needs -- every requested musical dimension."""

    # pitch & harmony
    register_lo_midi: float
    register_hi_midi: float
    scale: tuple
    root_midi: int
    detune_cents: float = 0.0
    # timbre / spectrum
    partials: tuple = (1.0, 0.5, 0.3)
    brightness: float = 0.5          # 0..1 -> spectral tilt / cutoff
    noise_content: float = 0.0       # 0..1
    distortion: float = 0.0          # 0..1 tanh drive
    # dynamics / amplitude
    loudness: float = 0.7            # 0..1 master gain
    attack: float = 0.02
    decay: float = 0.2
    sustain: float = 0.7
    release: float = 0.4
    tremolo_rate: float = 0.0
    tremolo_depth: float = 0.0
    vibrato_rate: float = 0.0
    vibrato_depth: float = 0.0       # semitones
    # time
    tempo_bpm: float = 90.0
    rhythm_density: float = 0.5      # 0..1
    duration: float = 4.0
    # space
    pan: float = 0.0                 # -1 L .. +1 R
    pan_motion_rate: float = 0.0
    pan_motion_depth: float = 0.0
    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0
    reverb: float = 0.2              # 0..1 wet
    spatial_3d: bool = False
    # bookkeeping
    category: str = ""
    label: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class CategoryVoice:
    """The fixed fingerprint of one wave category."""

    name: str
    register: tuple            # (lo_midi, hi_midi)
    scale: tuple
    root_midi: int
    partials: tuple
    base_brightness: float
    base_noise: float
    envelope: tuple           # (attack, decay, sustain, release)
    base_reverb: float
    base_tempo: float
    articulation: str         # 'sustained' | 'plucked' | 'pointillistic' | 'bowed'
    description: str


# --- the seven fingerprints ------------------------------------------------
# Voices are laid out on a similarity continuum so that sonic distance tracks
# physical similarity:  seismic -> audio -> radio -> infrared -> radar -> gamma
# -> ultrasound  (low/warm/mechanical ... high/bright/sharp). Neighbours cluster
# {seismic,audio} low-warm, {radio,infrared} mid-warm-EM, {radar,gamma,
# ultrasound} high-bright; the two extremes (seismic and ultrasound) are the
# most different in register, brightness and articulation.
CATEGORY_VOICES: dict[str, CategoryVoice] = {
    "seismic": CategoryVoice(
        name="seismic", register=(22, 44), scale=MINOR, root_midi=33,
        partials=(1.0, 0.72, 0.58, 0.44, 0.32, 0.24, 0.18),  # rich bowed contrabass
        base_brightness=0.2, base_noise=0.08, envelope=(0.3, 0.62, 0.92, 1.4),
        base_reverb=0.64, base_tempo=52, articulation="bowed",
        description="Deep bowed contrabass drone; magnitude -> loudness/register.",
    ),
    "audio": CategoryVoice(
        name="audio", register=(43, 65), scale=MAJOR, root_midi=48,
        partials=(1.0, 0.55, 0.4, 0.28, 0.2, 0.12),  # natural voice-like pad
        base_brightness=0.42, base_noise=0.06, envelope=(0.06, 0.4, 0.85, 0.7),
        base_reverb=0.32, base_tempo=78, articulation="sustained",
        description="Warm natural voice-like pad; the sound reinterpreted in tune.",
    ),
    "radio": CategoryVoice(
        name="radio", register=(50, 72), scale=MAJOR, root_midi=57,
        partials=(1.0, 0.6, 0.42, 0.3, 0.22, 0.16),  # warm electric-piano/pad
        base_brightness=0.5, base_noise=0.08, envelope=(0.05, 0.4, 0.85, 0.6),
        base_reverb=0.28, base_tempo=84, articulation="sustained",
        description="Warm analog pad; AM tremolo, FM vibrato.",
    ),
    "infrared": CategoryVoice(
        name="infrared", register=(56, 78), scale=LYDIAN, root_midi=53,
        partials=(1.0, 0.5, 0.18, 0.12),  # breathy flute/soft pad
        base_brightness=0.52, base_noise=0.1, envelope=(0.18, 0.5, 0.9, 0.8),
        base_reverb=0.55, base_tempo=64, articulation="sustained",
        description="Warm breathy flute/pad; temperature sets brightness.",
    ),
    "radar": CategoryVoice(
        name="radar", register=(66, 88), scale=WHOLE_TONE, root_midi=62,
        partials=(1.0, 0.15, 0.55, 0.1, 0.32),  # bell-like shimmer
        base_brightness=0.76, base_noise=0.05, envelope=(0.004, 0.35, 0.25, 0.5),
        base_reverb=0.42, base_tempo=104, articulation="plucked",
        description="Crystalline glass bells; Doppler glides, PRF shimmer.",
    ),
    "gamma": CategoryVoice(
        name="gamma", register=(76, 98), scale=PENTATONIC, root_midi=64,
        partials=(1.0, 0.25, 0.12, 0.06),  # glassy celesta sparkle
        base_brightness=0.9, base_noise=0.03, envelope=(0.001, 0.12, 0.0, 0.4),
        base_reverb=0.6, base_tempo=100, articulation="pointillistic",
        description="Sparse celesta sparkles; photon energy -> pitch.",
    ),
    "ultrasound": CategoryVoice(
        name="ultrasound", register=(84, 106), scale=PENTATONIC, root_midi=72,
        partials=(1.0, 0.03, 0.02, 0.44),  # bright glass mallet (strong 4th partial)
        base_brightness=0.99, base_noise=0.03, envelope=(0.002, 0.14, 0.0, 0.2),
        base_reverb=0.32, base_tempo=126, articulation="plucked",
        description="Bright mallet echoes; range -> reverb/delay (bat sonar).",
    ),
}


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * float(np.clip(t, 0, 1))


def map_features(feat: WaveFeatures) -> AudioParameters:
    """Turn extracted features into synthesis parameters for their category."""
    wave_type = feat.aux.get("wave_type", "radio")
    voice = CATEGORY_VOICES[wave_type]
    lo, hi = voice.register
    a, d, s, r = voice.envelope

    # Register position: brighter/hotter waves ride higher in the band, but the
    # band itself is fixed to the category -> category stays recognisable.
    center = _lerp(lo, hi, 0.35 + 0.5 * feat.brightness)
    span = _lerp(6.0, 16.0, feat.bandwidth)
    reg_lo = max(lo, center - span)
    reg_hi = min(hi, center + span)

    p = AudioParameters(
        register_lo_midi=reg_lo,
        register_hi_midi=reg_hi,
        scale=voice.scale,
        root_midi=voice.root_midi,
        partials=voice.partials,
        brightness=_lerp(voice.base_brightness * 0.7, min(1.0, voice.base_brightness + 0.2), feat.brightness),
        noise_content=float(np.clip(voice.base_noise + 0.4 * np.clip((feat.flatness - 0.5) / 0.45, 0, 1), 0, 1)),
        distortion=float(np.clip(0.3 * feat.roughness, 0, 0.6)),
        loudness=float(np.clip(0.45 + 0.5 * feat.loudness, 0.1, 1.0)),
        attack=a, decay=d, sustain=s, release=r,
        tremolo_rate=_lerp(3.0, 8.0, feat.flux),
        tremolo_depth=_lerp(0.0, 0.5, feat.flux) if voice.articulation == "sustained" else 0.0,
        vibrato_rate=_lerp(4.0, 6.5, feat.flux),
        vibrato_depth=_lerp(0.0, 0.4, feat.flux) if voice.articulation in ("sustained", "bowed") else 0.0,
        tempo_bpm=feat.tempo_bpm if feat.tempo_bpm > 0 else voice.base_tempo,
        rhythm_density=float(np.clip(0.15 + feat.onset_rate / 8.0, 0.05, 1.0)),
        duration=max(1.5, feat.duration),
        reverb=float(np.clip(voice.base_reverb + 0.2 * feat.flatness, 0, 0.95)),
        pan_motion_rate=_lerp(0.05, 0.6, feat.flux),
        pan_motion_depth=_lerp(0.0, 0.7, feat.flux),
        category=wave_type,
        label=feat.aux.get("label", wave_type),
        meta=dict(feat.aux),
    )

    _apply_category_specifics(p, voice, feat)
    return p


def _apply_category_specifics(p: AudioParameters, voice: CategoryVoice, feat: WaveFeatures) -> None:
    """Wire physically meaningful side-channels into the right musical knob."""
    aux = feat.aux
    az = float(aux.get("azimuth_deg") or 0.0)
    el = float(aux.get("elevation_deg") or 0.0)
    p.azimuth_deg, p.elevation_deg = az, el
    p.pan = float(np.clip(np.sin(np.radians(az)), -1, 1))
    p.spatial_3d = bool(aux.get("elevation_deg") is not None)

    if p.category == "radar":
        dop = float(aux.get("doppler_hz") or 0.0)
        # Doppler -> a gentle pitch glide (vibrato-like) and register lift
        p.detune_cents = float(np.clip(dop / 8.0, -600, 600))
        p.vibrato_rate = 5.5
        p.vibrato_depth = float(np.clip(abs(dop) / 4000.0, 0, 0.5))
        p.tremolo_rate = float(np.clip(feat.onset_rate or 6.0, 2, 12))  # PRF shimmer
        p.tremolo_depth = 0.35

    elif p.category == "ultrasound":
        rng = aux.get("range_m")
        if rng is not None:
            rng = float(np.mean(np.atleast_1d(rng)))
            p.reverb = float(np.clip(0.2 + rng / 20.0, 0.1, 0.9))  # farther -> more space
            p.meta["echo_delay_s"] = float(np.clip(rng / 340.0, 0.01, 0.4))

    elif p.category == "gamma":
        rate = float(aux.get("event_rate_hz") or feat.onset_rate)
        p.rhythm_density = float(np.clip(rate / 12.0, 0.03, 0.8))
        p.tempo_bpm = float(np.clip(60 + rate * 6, 60, 160))

    elif p.category == "seismic":
        # Read magnitude from the preserved raw energy (the drive is peak-
        # normalised, so absolute amplitude must come from aux). Bigger quakes
        # sit lower, louder and darker -- directly readable by ear.
        energy = float(aux.get("energy") or 0.0)
        t = float(np.clip((np.log10(energy + 1e-6) + 1.0) / 4.0, 0.0, 1.0))  # ~M3..M7
        p.loudness = float(np.clip(0.35 + 0.6 * t, 0.2, 1.0))
        shift = _lerp(2.0, -7.0, t)
        p.register_lo_midi += shift
        p.register_hi_midi += shift
        p.brightness = float(np.clip(0.45 - 0.2 * t, 0.15, 0.6))
        p.vibrato_rate = 4.5
        p.vibrato_depth = 0.25
        p.meta["magnitude_proxy"] = t

    elif p.category == "infrared":
        temp = aux.get("temperature_k")
        if temp is not None:
            t = float(np.clip((float(temp) - 250.0) / 400.0, 0, 1))
            p.brightness = float(np.clip(0.3 + 0.6 * t, 0, 1))

    elif p.category == "radio":
        idx = aux.get("modulation_index")
        if idx is not None:
            p.vibrato_depth = float(np.clip(float(idx) / 10.0, 0, 0.5))
        if str(aux.get("mod", "")).upper() == "AM":
            p.tremolo_depth = max(p.tremolo_depth, 0.4)
