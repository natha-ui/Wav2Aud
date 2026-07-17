"""Synthetic wave sources.

Physically-plausible generators for every category so the package runs with no
hardware. Each returns a :class:`~wav2aud.waves.WaveSample` carrying the native
physics (true sample rate, carrier, side channels) exactly as a real sensor
would. A library of named :data:`PRESETS` demonstrates the intended behaviour:
*comparable* waves within a category (e.g. a car vs a truck on radar) map to
*similar* audio, while *very different* waves map far apart.
"""
from __future__ import annotations

import numpy as np

from .waves import WaveSample

C_LIGHT = 2.998e8
C_SOUND = 340.0


# ---------------------------------------------------------------------------
# seismic
# ---------------------------------------------------------------------------
def simulate_seismic(magnitude=4.5, distance_km=30.0, duration=20.0, fs=200.0, seed=0) -> WaveSample:
    rng = np.random.default_rng(seed)
    n = int(duration * fs)
    t = np.arange(n) / fs
    # bigger quakes radiate lower dominant frequency
    f_dom = np.clip(6.0 - 0.7 * magnitude, 0.4, 5.0)
    p_arr = 0.15 * distance_km / 6.0        # P-wave arrival (fast)
    s_arr = 0.15 * distance_km / 3.5        # S-wave arrival (slower, stronger)
    amp = 10 ** (magnitude - 4.0)
    x = np.zeros(n)

    def wavelet(arrival, f, gain, decay):
        env = np.exp(-np.maximum(t - arrival, 0) / decay) * (t >= arrival)
        return gain * env * np.sin(2 * np.pi * f * (t - arrival))

    x += wavelet(p_arr, f_dom * 1.6, 0.4 * amp, 2.0)
    x += wavelet(s_arr, f_dom, amp, 4.0)
    x += 0.05 * amp * rng.standard_normal(n) * np.exp(-t / (duration))
    return WaveSample(
        data=x, sample_rate=fs, wave_type="seismic",
        meta={"range_m": distance_km * 1000.0, "azimuth_deg": 20.0,
              "magnitude": magnitude, "label": f"M{magnitude:g} @ {distance_km:g}km"},
    )


# ---------------------------------------------------------------------------
# radar / microwave
# ---------------------------------------------------------------------------
def simulate_radar(velocity_ms=15.0, carrier=10.5e9, prf=8000.0, duration=1.5,
                   rcs=1.0, micro_doppler=0.0, azimuth_deg=0.0, seed=1) -> WaveSample:
    rng = np.random.default_rng(seed)
    n = int(duration * prf)
    t = np.arange(n) / prf
    f_d = 2.0 * velocity_ms * carrier / C_LIGHT   # Doppler shift (Hz)
    iq = np.sqrt(rcs) * np.exp(2j * np.pi * f_d * t)
    if micro_doppler > 0:  # rotating parts (blades, wheels)
        iq += 0.4 * np.sqrt(rcs) * np.exp(2j * np.pi * (f_d + micro_doppler * np.sin(2 * np.pi * 12 * t)) * t)
    iq += 0.05 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return WaveSample(
        data=iq, sample_rate=prf, wave_type="radar", carrier=carrier,
        meta={"radial_velocity_ms": velocity_ms, "range_m": 120.0,
              "azimuth_deg": azimuth_deg, "elevation_deg": 0.0,
              "label": f"v={velocity_ms:g}m/s fd={f_d:.0f}Hz"},
    )


# ---------------------------------------------------------------------------
# radio
# ---------------------------------------------------------------------------
def simulate_radio(mod="FM", message_hz=(3.0, 5.0), carrier=98.5e6, duration=3.0,
                   fs=8000.0, modulation_index=3.0, azimuth_deg=-15.0, seed=2) -> WaveSample:
    rng = np.random.default_rng(seed)
    n = int(duration * fs)
    t = np.arange(n) / fs
    msg = np.zeros(n)
    for i, f in enumerate(message_hz):
        msg += (1.0 / (i + 1)) * np.sin(2 * np.pi * f * t)
    msg /= np.max(np.abs(msg)) + 1e-9
    if mod.upper() == "FM":
        phase = modulation_index * np.cumsum(msg) / fs * (2 * np.pi * 3.0)
        s = np.exp(1j * phase)
    else:  # AM
        s = (1.0 + 0.7 * msg) * np.exp(1j * 2 * np.pi * 0.0 * t)
    s = s + 0.02 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return WaveSample(
        data=s, sample_rate=fs, wave_type="radio", carrier=carrier,
        meta={"mod": mod.upper(), "modulation_index": modulation_index,
              "bandwidth_hz": 2 * (max(message_hz) * (modulation_index + 1)),
              "azimuth_deg": azimuth_deg, "label": f"{mod.upper()} idx={modulation_index:g}"},
    )


# ---------------------------------------------------------------------------
# infrared / thermal
# ---------------------------------------------------------------------------
def simulate_infrared(temperature_k=310.0, flicker=0.2, duration=8.0, fs=100.0,
                      bands=(0.6, 1.2, 2.4), seed=3) -> WaveSample:
    rng = np.random.default_rng(seed)
    n = int(duration * fs)
    t = np.arange(n) / fs
    x = np.zeros(n)
    for b in bands:  # slow thermal breathing across emissivity bands
        x += (1.0 / b) * np.sin(2 * np.pi * b * t + rng.uniform(0, 2 * np.pi))
    x += flicker * rng.standard_normal(n)
    x = x / (np.max(np.abs(x)) + 1e-9)
    return WaveSample(
        data=x, sample_rate=fs, wave_type="infrared",
        meta={"temperature_k": temperature_k, "emissivity": 0.95, "bands": list(bands),
              "azimuth_deg": 5.0, "label": f"T={temperature_k:g}K"},
    )


# ---------------------------------------------------------------------------
# ultrasound
# ---------------------------------------------------------------------------
def simulate_ultrasound(range_m=1.5, n_echoes=3, carrier=40000.0, fs=400000.0,
                        duration=0.03, azimuth_deg=10.0, seed=4) -> WaveSample:
    rng = np.random.default_rng(seed)
    n = int(duration * fs)
    t = np.arange(n) / fs
    burst_len = int(0.0015 * fs)
    burst = np.zeros(n)
    tb = np.arange(burst_len) / fs
    ping = np.hanning(burst_len) * np.sin(2 * np.pi * carrier * tb)
    burst[:burst_len] = ping
    x = np.zeros(n)
    for k in range(n_echoes):
        delay = 2.0 * (range_m + 0.4 * k) / C_SOUND
        idx = int(delay * fs)
        if idx + burst_len < n:
            x[idx: idx + burst_len] += ping * (0.7 ** (k + 1))
    x += 0.01 * rng.standard_normal(n)
    return WaveSample(
        data=x, sample_rate=fs, wave_type="ultrasound", carrier=carrier,
        meta={"range_m": range_m, "n_echoes": n_echoes, "azimuth_deg": azimuth_deg,
              "elevation_deg": 0.0, "label": f"{range_m:g}m x{n_echoes}"},
    )


# ---------------------------------------------------------------------------
# gamma
# ---------------------------------------------------------------------------
def simulate_gamma(rate_hz=6.0, duration=6.0, lines=(662.0,), continuum=(50.0, 400.0),
                   line_fraction=0.6, seed=5) -> WaveSample:
    rng = np.random.default_rng(seed)
    n_events = max(1, rng.poisson(rate_hz * duration))
    times = np.sort(rng.uniform(0, duration, n_events))
    energies = np.empty(n_events)
    for i in range(n_events):
        if rng.random() < line_fraction and lines:
            e = rng.choice(lines) * (1 + rng.normal(0, 0.02))  # photopeak + resolution
        else:
            e = rng.uniform(*continuum)                        # Compton continuum
        energies[i] = max(e, 1.0)
    return WaveSample(
        data=energies, sample_rate=rate_hz, wave_type="gamma",
        meta={"event_times": times, "label": f"{rate_hz:g}cps lines={lines}"},
    )


# ---------------------------------------------------------------------------
# audio (ordinary sound)
# ---------------------------------------------------------------------------
def simulate_audio(kind="vowel", f0=160.0, duration=2.5, fs=16000.0, seed=6) -> WaveSample:
    rng = np.random.default_rng(seed)
    n = int(duration * fs)
    t = np.arange(n) / fs
    if kind == "whistle":
        f = f0 * 2 + 60 * np.sin(2 * np.pi * 0.6 * t) + 40 * t
        x = np.sin(2 * np.pi * np.cumsum(f) / fs)
    else:  # vowel: harmonic source + formants
        vib = 1 + 0.01 * np.sin(2 * np.pi * 5.5 * t)
        x = np.zeros(n)
        for k in range(1, 12):
            x += (1.0 / k) * np.sin(2 * np.pi * f0 * k * np.cumsum(vib) / fs)
        # crude formant emphasis
        for fc, g in ((700, 1.0), (1220, 0.6), (2600, 0.35)):
            x += g * np.sin(2 * np.pi * fc * t) * (0.4 + 0.6 * np.abs(np.sin(2 * np.pi * f0 * t)))
    x += 0.02 * rng.standard_normal(n)
    x = x / (np.max(np.abs(x)) + 1e-9)
    return WaveSample(
        data=x, sample_rate=fs, wave_type="audio",
        meta={"kind": kind, "azimuth_deg": 0.0, "label": f"{kind} @ {f0:g}Hz"},
    )


# ---------------------------------------------------------------------------
# presets: comparable vs very-different within each category
# ---------------------------------------------------------------------------
def _presets():
    return {
        # seismic: two similar small quakes + one very different great quake
        "seismic_M3_near": lambda: simulate_seismic(3.0, 12, seed=10),
        "seismic_M3p3_near": lambda: simulate_seismic(3.3, 15, seed=11),
        "seismic_M7_far": lambda: simulate_seismic(7.0, 120, duration=30, seed=12),
        # radar: car vs truck (similar) vs drone (micro-Doppler, different)
        "radar_car": lambda: simulate_radar(14, seed=20),
        "radar_truck": lambda: simulate_radar(17, rcs=4.0, seed=21),
        "radar_drone": lambda: simulate_radar(6, micro_doppler=900, rcs=0.3, seed=22),
        # radio: two FM stations (similar) vs an AM broadcast (different)
        "radio_fm_music": lambda: simulate_radio("FM", (3.0, 5.0), seed=30),
        "radio_fm_talk": lambda: simulate_radio("FM", (2.0, 6.5), modulation_index=2.0, seed=31),
        "radio_am_beacon": lambda: simulate_radio("AM", (4.0,), modulation_index=1.0, seed=32),
        # infrared: warm body vs hot engine (different temperature)
        "ir_body": lambda: simulate_infrared(310, seed=40),
        "ir_warm_room": lambda: simulate_infrared(295, flicker=0.1, seed=41),
        "ir_engine": lambda: simulate_infrared(520, flicker=0.35, seed=42),
        # ultrasound: near single echo vs far multi-echo (clutter)
        "us_near_single": lambda: simulate_ultrasound(0.8, 1, seed=50),
        "us_mid_double": lambda: simulate_ultrasound(1.6, 2, seed=51),
        "us_far_clutter": lambda: simulate_ultrasound(4.0, 5, seed=52),
        # gamma: quiet background vs active Cs-137 source vs intense mixed
        "gamma_background": lambda: simulate_gamma(2.0, lines=(), seed=60),
        "gamma_cs137": lambda: simulate_gamma(6.0, lines=(662.0,), seed=61),
        "gamma_intense_mixed": lambda: simulate_gamma(18.0, lines=(1173.0, 1332.0), seed=62),
        # audio: two similar vowels + a very different whistle
        "audio_vowel_low": lambda: simulate_audio("vowel", 140, seed=70),
        "audio_vowel_high": lambda: simulate_audio("vowel", 190, seed=71),
        "audio_whistle": lambda: simulate_audio("whistle", 300, seed=72),
    }


PRESETS = _presets()


def preset(name: str) -> WaveSample:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; options: {sorted(PRESETS)}")
    return PRESETS[name]()


def all_presets() -> dict:
    return {name: fn() for name, fn in PRESETS.items()}
