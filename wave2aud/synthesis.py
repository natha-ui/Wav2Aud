"""Synthesiser: resynthesise the auditory image as pleasant, musical stereo.

The instrument is a **fixed bank of scale notes** for the category (every scale
degree inside the category's register). The cochlea routes each channel's neural
energy onto its nearest scale note, so the output is always in-scale and never
muddier than the ~10-20 notes of the bank. Each note is one oscillator with a
category partial recipe; summing them turns the wave's spectral evolution into
melody and harmony. Because the note bank is fixed and each note keeps its own
phase, the very same model drives the real-time engine (:mod:`wave2aud.realtime`).

Implements every requested dimension: pitch, loudness, timbre, harmony, rhythm,
tempo, duration, stereo, 3-D spatialisation, reverb, vibrato, tremolo, envelope,
noise content, brightness, distortion and panning motion.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from .features import _detect_onsets
from .mapping import AudioParameters, CATEGORY_VOICES


# ---------------------------------------------------------------------------
# pitch helpers
# ---------------------------------------------------------------------------
def midi_to_hz(m):
    return 440.0 * 2.0 ** ((np.asarray(m, float) - 69.0) / 12.0)


def scale_notes_in_band(root: int, scale, lo_midi: float, hi_midi: float) -> np.ndarray:
    """Every scale-degree MIDI note between ``lo_midi`` and ``hi_midi``."""
    sset = set(int(s) % 12 for s in scale)
    notes = [m for m in range(int(np.floor(lo_midi)), int(np.ceil(hi_midi)) + 1)
             if (m - root) % 12 in sset]
    if not notes:  # degenerate guard
        notes = [int(round(0.5 * (lo_midi + hi_midi)))]
    return np.asarray(notes, dtype=float)


def channel_to_midi(cf: np.ndarray, p: AudioParameters) -> np.ndarray:
    """Continuous MIDI pitch each cochlear channel maps to (before snapping)."""
    cf = np.asarray(cf, float)
    lo, hi = float(cf.min()), float(cf.max())
    frac = (np.log2(np.clip(cf, lo, hi)) - np.log2(lo)) / (np.log2(hi) - np.log2(lo) + 1e-9)
    return p.register_lo_midi + frac * (p.register_hi_midi - p.register_lo_midi) + p.detune_cents / 100.0


class NoteBank:
    """The fixed set of pitched voices for a category, plus channel routing."""

    def __init__(self, params: AudioParameters):
        voice = CATEGORY_VOICES[params.category]
        lo, hi = voice.register
        self.notes = scale_notes_in_band(params.root_midi, params.scale, lo, hi)
        self.freqs = midi_to_hz(self.notes)

    def route(self, cf: np.ndarray, params: AudioParameters) -> np.ndarray:
        midi = channel_to_midi(cf, params)
        return np.argmin(np.abs(self.notes[:, None] - midi[None, :]), axis=0)

    def note_envelopes(self, channel_env: np.ndarray, cf, params) -> np.ndarray:
        idx = self.route(cf, params)
        out = np.zeros((len(self.notes), channel_env.shape[1]))
        for ch in range(channel_env.shape[0]):
            out[idx[ch]] += channel_env[ch]
        return out


def retune_channels(cf: np.ndarray, p: AudioParameters) -> np.ndarray:
    """Legacy helper (used by viz): channel -> snapped output frequency."""
    bank = NoteBank(p)
    idx = bank.route(cf, p)
    return bank.freqs[idx]


# ---------------------------------------------------------------------------
# envelope generators
# ---------------------------------------------------------------------------
def adsr(n, a, d, s, r, fs):
    env = np.zeros(n)
    ai, di, ri = int(a * fs), int(d * fs), int(r * fs)
    if ai + di + ri > n:
        k = n / max(ai + di + ri + 1, 1)
        ai, di, ri = int(ai * k), int(di * k), int(ri * k)
    si = max(0, n - ai - di - ri)
    if ai > 0:
        env[:ai] = np.linspace(0, 1, ai)
    if di > 0:
        env[ai:ai + di] = np.linspace(1, s, di)
    env[ai + di:ai + di + si] = s
    if ri > 0:
        env[ai + di + si:ai + di + si + ri] = np.linspace(s, 0, ri)
    return env[:n]


def grain_envelope(onset_frames, cr, n, fs, decay, release):
    env = np.zeros(n)
    length = int((decay + release) * fs) or int(0.1 * fs)
    shape = np.exp(-np.arange(length) / (0.35 * length + 1.0))
    for oc in onset_frames:
        i = int(oc * fs / cr)
        end = min(n, i + length)
        if end > i:
            env[i:end] = np.maximum(env[i:end], shape[: end - i])
    return env


# ---------------------------------------------------------------------------
# effects
# ---------------------------------------------------------------------------
def _reverb_ir(fs, t60=1.6, seed=7):
    rng = np.random.default_rng(seed)
    n = int(t60 * fs)
    ir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
    ir[: int(0.005 * fs)] *= 0.2
    ir /= np.sqrt(np.sum(ir ** 2)) + 1e-9
    return ir


def apply_reverb(x, wet, fs):
    if wet <= 0.001:
        return x
    ir = _reverb_ir(fs)
    wetsig = fftconvolve(x, ir, mode="full")[: len(x)]
    wetsig /= np.max(np.abs(wetsig)) + 1e-9
    return (1 - wet) * x + wet * wetsig


def _lowpass(x, cutoff, fs, order=2):
    wn = min(cutoff / (fs / 2), 0.99)
    sos = butter(order, max(wn, 1e-3), btype="low", output="sos")
    return sosfilt(sos, x)


def _declick(x, fs, ms=6.0):
    k = int(ms / 1000.0 * fs)
    if k > 0 and len(x) > 2 * k:
        ramp = np.linspace(0, 1, k)
        x[:k] *= ramp
        x[-k:] *= ramp[::-1]
    return x


def equal_power_pan(mono, pan):
    theta = (np.clip(pan, -1, 1) + 1.0) * (np.pi / 4.0)
    return mono * np.cos(theta), mono * np.sin(theta)


def normalize_rms(stereo, target=0.12):
    rms = np.sqrt(np.mean(stereo ** 2)) + 1e-9
    g = target / rms
    out = stereo * min(g, 8.0)
    peak = np.max(np.abs(out))
    if peak > 0.97:
        out *= 0.97 / peak
    return out


# ---------------------------------------------------------------------------
# main render (offline)
# ---------------------------------------------------------------------------
def render(params, envelopes, cf, control_rate, fs=44100.0):
    """Render stereo float32 audio in [-1, 1], shape ``(n_samples, 2)``."""
    p = params
    voice = CATEGORY_VOICES[p.category]
    n = max(int(p.duration * fs), int(0.5 * fs))
    t = np.arange(n) / fs

    bank = NoteBank(p)
    note_env = bank.note_envelopes(envelopes, cf, p)          # [n_notes, T]
    peak = note_env.max() + 1e-12
    note_env = note_env / peak

    ctrl_t = np.arange(envelopes.shape[1]) / control_rate
    vib_lfo = np.sin(2 * np.pi * p.vibrato_rate * t) if p.vibrato_depth > 0 else None

    partials = np.asarray(p.partials, float)
    partials = partials / (partials.sum() + 1e-9)
    chorus = voice.articulation == "sustained"                # detune pads slightly

    mix = np.zeros(n)
    for j, f0 in enumerate(bank.freqs):
        e = note_env[j]
        if e.max() < 0.02 or f0 <= 0 or f0 > 0.45 * fs:
            continue
        env_up = np.interp(t, ctrl_t, e)
        if vib_lfo is not None:
            beta = (2 ** (p.vibrato_depth / 12.0) - 1.0) * f0 / max(p.vibrato_rate, 0.5)
            vib_phase = beta * vib_lfo
        else:
            vib_phase = 0.0
        osc = np.zeros(n)
        for k, w in enumerate(partials):
            fk = f0 * (k + 1)
            if fk > 0.45 * fs:
                break
            wk = w / (1.0 + 0.12 * k)                          # gentle high rolloff
            if chorus:
                osc += 0.5 * wk * np.sin(2 * np.pi * fk * 0.997 * t + (k + 1) * vib_phase)
                osc += 0.5 * wk * np.sin(2 * np.pi * fk * 1.003 * t + (k + 1) * vib_phase)
            else:
                osc += wk * np.sin(2 * np.pi * fk * t + (k + 1) * vib_phase)
        mix += env_up * osc

    if np.max(np.abs(mix)) > 0:
        mix /= np.max(np.abs(mix))

    # articulation
    loud = envelopes.sum(axis=0)
    loud /= loud.max() + 1e-12
    if voice.articulation in ("plucked", "pointillistic"):
        onsets = _detect_onsets(loud, control_rate)
        shape = (grain_envelope(onsets, control_rate, n, fs, p.decay, p.release)
                 if onsets.size else adsr(n, p.attack, p.decay, 0.0, p.release, fs))
        mix *= 0.12 + 0.88 * shape
    else:
        mix *= adsr(n, p.attack, p.decay, p.sustain, p.release, fs)

    # noise content (breath / hiss)
    if p.noise_content > 0.01:
        rng = np.random.default_rng(3)
        noise = _lowpass(rng.standard_normal(n), 200 + 4500 * p.brightness, fs)
        loud_up = np.interp(t, ctrl_t, loud)
        mix += p.noise_content * 0.14 * noise * loud_up

    # brightness -> spectral tilt (kept musical, not harsh)
    cutoff = 650 * (11000 / 650) ** p.brightness
    mix = _lowpass(mix, cutoff, fs)

    if p.tremolo_depth > 0.001:
        mix *= 1.0 - p.tremolo_depth * (0.5 + 0.5 * np.sin(2 * np.pi * p.tremolo_rate * t))
    if p.distortion > 0.001:
        drive = 1.0 + 6.0 * p.distortion
        mix = np.tanh(drive * mix) / np.tanh(drive)

    mix = _declick(mix, fs)

    pan = p.pan + p.pan_motion_depth * np.sin(2 * np.pi * p.pan_motion_rate * t)
    left, right = equal_power_pan(mix, pan)

    if p.spatial_3d:
        itd = int(np.clip(np.sin(np.radians(p.azimuth_deg)) * 0.0006 * fs, -30, 30))
        if itd > 0:
            right = np.concatenate([np.zeros(itd), right])[:n]
        elif itd < 0:
            left = np.concatenate([np.zeros(-itd), left])[:n]
        el = np.clip((p.elevation_deg + 90) / 180.0, 0, 1)
        left = _lowpass(left, 2500 + 12000 * el, fs)
        right = _lowpass(right, 2500 + 12000 * el, fs)

    left = apply_reverb(left, p.reverb, fs)
    right = apply_reverb(right, p.reverb, fs)

    # RMS-normalise for a balanced, pleasant level across categories, while
    # still letting louder waves read as a bit louder (bounded).
    stereo = np.stack([left, right], axis=1)
    stereo = normalize_rms(stereo, target=0.085 + 0.06 * float(np.clip(p.loudness, 0, 1)))
    return stereo.astype(np.float32)
