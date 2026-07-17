"""Stateful real-time synthesis engine.

Unlike the offline pipeline (which renders a whole clip and the old
``StreamingSonifier`` cross-faded independent blocks), this engine keeps *every*
DSP stage's state across blocks, so successive blocks join seamlessly with no
cross-fade:

* **Streaming complex-gammatone cochlea** (Hohmann-style): each channel is a
  cascade of complex one-pole resonators whose state persists; the magnitude of
  the complex output is the analytic envelope (no Hilbert needed).
* **Streaming hair cells**: compression + adaptation with retained filter state.
* **Persistent-phase note bank**: the fixed scale-note oscillators keep their
  phase between blocks, so pitch is continuous.
* **Stateful Schroeder reverb**: comb + all-pass delay lines carried over.
* **Parameter smoothing**: mapping parameters glide between blocks.

Everything uses ``scipy.signal.lfilter``/``sosfilt`` with retained ``zi`` state,
so it is both correct and fast.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, lfilter, sosfilt, sosfilt_zi

from .cochlea import erb_space, erb_bandwidth
from .features import WaveFeatures
from .mapping import AudioParameters, CATEGORY_VOICES, map_features
from .synthesis import NoteBank
from .transduction import TransductionFrontEnd
from .waves import WaveSample


# ---------------------------------------------------------------------------
# streaming complex gammatone cochlea
# ---------------------------------------------------------------------------
class StreamingComplexGammatone:
    """Hohmann complex gammatone filterbank with persistent per-channel state."""

    def __init__(self, fs, f_lo=45.0, f_hi=None, n_channels=36, order=4):
        self.fs = float(fs)
        f_hi = float(f_hi) if f_hi else 0.45 * self.fs
        self.cf = erb_space(f_lo, f_hi, n_channels)
        self.order = int(order)
        b = 1.019 * erb_bandwidth(self.cf)                       # bandwidth (Hz)
        self.a = np.exp(-2 * np.pi * b / self.fs) * np.exp(2j * np.pi * self.cf / self.fs)
        self.gain = (1.0 - np.abs(self.a)) ** self.order         # normalise peak to 1
        # state: [n_channels, order] complex, one sample of memory per stage
        self.zi = np.zeros((n_channels, self.order), dtype=complex)

    def process(self, x: np.ndarray) -> np.ndarray:
        """Return complex output [n_channels, n] (envelope = abs)."""
        x = np.asarray(x, dtype=complex)
        n = len(x)
        out = np.empty((len(self.cf), n), dtype=complex)
        for ch in range(len(self.cf)):
            a = self.a[ch]
            v = x
            for s in range(self.order):
                v, zf = lfilter([1.0], [1.0, -a], v, zi=[self.zi[ch, s]])
                self.zi[ch, s] = zf[0]
            out[ch] = v * self.gain[ch]
        return out


class StreamingHairCells:
    def __init__(self, fs, compression=0.4, adapt_tau=0.06, onset=0.5):
        self.compression = float(compression)
        self.onset = float(onset)
        wn = min(1.0 / (2 * np.pi * adapt_tau) / (fs / 2), 0.99)
        self.sos = butter(1, max(wn, 1e-4), btype="low", output="sos")
        self.zi = None  # lazily shaped per channel

    def process(self, envelope: np.ndarray) -> np.ndarray:
        env = np.maximum(envelope, 0.0)
        comp = np.power(env, self.compression)
        if self.zi is None:
            zi0 = sosfilt_zi(self.sos)                    # [n_sections, 2]
            self.zi = np.repeat(zi0[:, None, :], comp.shape[0], axis=1) * 0.0
        baseline, self.zi = sosfilt(self.sos, comp, axis=1, zi=self.zi)
        onset = np.maximum(comp - baseline, 0.0)
        return (1.0 - self.onset) * comp + self.onset * onset


# ---------------------------------------------------------------------------
# stateful Schroeder reverb (comb + all-pass via lfilter zi)
# ---------------------------------------------------------------------------
class _CombZi:
    def __init__(self, fs, delay_ms, g):
        D = max(1, int(delay_ms * fs / 1000.0))
        self.b = np.array([1.0])
        self.a = np.zeros(D + 1); self.a[0] = 1.0; self.a[-1] = -g
        self.zi = np.zeros(D)

    def process(self, x):
        y, self.zi = lfilter(self.b, self.a, x, zi=self.zi)
        return y


class _AllpassZi:
    def __init__(self, fs, delay_ms, g):
        D = max(1, int(delay_ms * fs / 1000.0))
        self.b = np.zeros(D + 1); self.b[0] = -g; self.b[-1] = 1.0
        self.a = np.zeros(D + 1); self.a[0] = 1.0; self.a[-1] = -g
        self.zi = np.zeros(D)

    def process(self, x):
        y, self.zi = lfilter(self.b, self.a, x, zi=self.zi)
        return y


class StreamingReverb:
    def __init__(self, fs):
        self.combs = [_CombZi(fs, d, g) for d, g in
                      [(29.7, 0.78), (37.1, 0.76), (41.1, 0.74), (43.7, 0.72)]]
        self.aps = [_AllpassZi(fs, d, 0.7) for d in (5.0, 1.7)]

    def process(self, x, wet):
        if wet <= 0.001:
            return x
        acc = np.zeros_like(x)
        for c in self.combs:
            acc += c.process(x)
        acc /= len(self.combs)
        for ap in self.aps:
            acc = ap.process(acc)
        return (1.0 - wet) * x + wet * acc


# ---------------------------------------------------------------------------
# stateful note-bank synthesiser
# ---------------------------------------------------------------------------
class StatefulSynth:
    def __init__(self, fs, params: AudioParameters):
        self.fs = float(fs)
        self.voice = CATEGORY_VOICES[params.category]
        self.bank = NoteBank(params)
        self.phase = np.zeros(len(self.bank.freqs))     # per-note oscillator phase
        self.vib_phase = 0.0
        self.trem_phase = 0.0
        self.pan_phase = 0.0
        self.reverb = StreamingReverb(fs)
        wn = 0.5
        self.bright_sos = butter(2, wn, btype="low", output="sos")
        self.bright_zi = None
        self._chorus = self.voice.articulation == "sustained"

    def _lowpass_bright(self, x, brightness):
        cutoff = 650 * (11000 / 650) ** float(np.clip(brightness, 0, 1))
        wn = min(cutoff / (self.fs / 2), 0.99)
        self.bright_sos = butter(2, max(wn, 1e-3), btype="low", output="sos")
        if self.bright_zi is None:
            self.bright_zi = sosfilt_zi(self.bright_sos) * 0.0
        y, self.bright_zi = sosfilt(self.bright_sos, x, zi=self.bright_zi)
        return y

    def process(self, note_env: np.ndarray, p: AudioParameters) -> np.ndarray:
        n = note_env.shape[1]
        two_pi = 2 * np.pi
        t_inc = np.arange(1, n + 1) / self.fs

        partials = np.asarray(p.partials, float)
        partials = partials / (partials.sum() + 1e-9)

        # LFO phase ramps (continued across blocks)
        vib = np.sin(self.vib_phase + two_pi * p.vibrato_rate * t_inc) if p.vibrato_depth > 0 else None
        self.vib_phase = (self.vib_phase + two_pi * p.vibrato_rate * n / self.fs) % two_pi

        mix = np.zeros(n)
        peak = note_env.max() + 1e-9
        for j, f0 in enumerate(self.bank.freqs):
            e = note_env[j] / peak
            if e.max() < 0.015 or f0 <= 0 or f0 > 0.45 * self.fs:
                self.phase[j] = (self.phase[j] + two_pi * f0 * n / self.fs) % two_pi
                continue
            base = self.phase[j] + two_pi * f0 * t_inc
            if vib is not None:
                beta = (2 ** (p.vibrato_depth / 12.0) - 1.0) * f0 / max(p.vibrato_rate, 0.5)
                base = base + beta * vib
            osc = np.zeros(n)
            for k, w in enumerate(partials):
                fk = f0 * (k + 1)
                if fk > 0.45 * self.fs:
                    break
                wk = w / (1.0 + 0.12 * k)
                if self._chorus:
                    osc += 0.5 * wk * np.sin((k + 1) * base * 0.997)
                    osc += 0.5 * wk * np.sin((k + 1) * base * 1.003)
                else:
                    osc += wk * np.sin((k + 1) * base)
            mix += e * osc
            self.phase[j] = (self.phase[j] + two_pi * f0 * n / self.fs) % two_pi

        m = np.max(np.abs(mix))
        if m > 0:
            mix /= m

        mix = self._lowpass_bright(mix, p.brightness)

        if p.tremolo_depth > 0.001:
            trem = 1.0 - p.tremolo_depth * (0.5 + 0.5 * np.sin(self.trem_phase + two_pi * p.tremolo_rate * t_inc))
            mix *= trem
            self.trem_phase = (self.trem_phase + two_pi * p.tremolo_rate * n / self.fs) % two_pi
        if p.distortion > 0.001:
            drive = 1.0 + 6.0 * p.distortion
            mix = np.tanh(drive * mix) / np.tanh(drive)

        pan = p.pan + p.pan_motion_depth * np.sin(self.pan_phase + two_pi * p.pan_motion_rate * t_inc)
        self.pan_phase = (self.pan_phase + two_pi * p.pan_motion_rate * n / self.fs) % two_pi
        theta = (np.clip(pan, -1, 1) + 1.0) * (np.pi / 4.0)
        left = self.reverb.process(mix * np.cos(theta), p.reverb)
        right = self.reverb.process(mix * np.sin(theta), p.reverb)

        stereo = np.stack([left, right], axis=1)
        target = 0.085 + 0.06 * float(np.clip(p.loudness, 0, 1))
        rms = np.sqrt(np.mean(stereo ** 2)) + 1e-9
        stereo *= min(target / rms, 8.0)
        pk = np.max(np.abs(stereo))
        if pk > 0.97:
            stereo *= 0.97 / pk
        return stereo.astype(np.float32)


# ---------------------------------------------------------------------------
# quick per-block features (lightweight; audio-rate activity in)
# ---------------------------------------------------------------------------
def _quick_features(activity, cf, aux, fs) -> WaveFeatures:
    per_ch = activity.mean(axis=1)
    ch_lin = np.power(per_ch, 2.6)
    p = ch_lin / (ch_lin.sum() + 1e-12)
    centroid = float(np.sum(p * cf))
    lo, hi = float(cf.min()), float(cf.max())
    brightness = float(np.clip((np.log2(max(centroid, lo)) - np.log2(lo)) /
                               (np.log2(hi) - np.log2(lo) + 1e-9), 0, 1))
    spread = float(np.sqrt(np.sum(p * (cf - centroid) ** 2)))
    bandwidth = float(np.clip(spread / (0.5 * (hi - lo) + 1e-9), 0, 1))
    geo = np.exp(np.mean(np.log(ch_lin + 1e-12)))
    flatness = float(np.clip(geo / (ch_lin.mean() + 1e-12), 0, 1))
    loud = activity.sum(axis=0)
    ln = loud / (loud.max() + 1e-12)
    loudness = float(np.clip(ln.mean() * 1.4, 0.05, 1))
    flux = float(np.clip(np.mean(np.abs(np.diff(ln))) * fs / 800.0, 0, 1))
    return WaveFeatures(
        loudness=loudness, brightness=brightness, centroid_hz=centroid,
        bandwidth=bandwidth, flatness=flatness, onset_rate=float(aux.get("event_rate_hz", 0) or 0),
        flux=flux, tempo_bpm=0.0, roughness=float(np.clip(flux * 1.2, 0, 1)),
        duration=activity.shape[1] / fs, cf=cf, channel_energy=per_ch,
        loudness_curve=ln, control_rate=fs, aux=dict(aux),
    )


def _smooth_params(prev: AudioParameters, new: AudioParameters, a=0.25) -> AudioParameters:
    if prev is None:
        return new
    for f in ("register_lo_midi", "register_hi_midi", "brightness", "reverb",
              "tremolo_depth", "vibrato_depth", "loudness", "pan", "noise_content"):
        setattr(new, f, (1 - a) * getattr(prev, f) + a * getattr(new, f))
    return new


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
class RealtimeSonifier:
    """Feed wave chunks, get seamless audio blocks. Category is fixed by the
    first chunk (a live stream is one modality)."""

    def __init__(self, fs=44100.0, f_lo=45.0, n_channels=36, overlap=1024):
        self.fs = float(fs)
        self.frontend = TransductionFrontEnd(fs_out=fs)
        self._f_lo, self._n = f_lo, n_channels
        self._overlap = int(overlap)
        self._tail = None
        self.cochlea = None
        self.hair = None
        self.synth = None
        self._params = None
        self._category = None
        self.last_features = None
        self.last_drive = None

    def _couple_continuous(self, chunk: WaveSample):
        """Overlap-save coupling so per-block resampling joins seamlessly."""
        if self._overlap <= 0 or self._tail is None:
            drive = self.frontend.couple(chunk)
            self._tail = chunk.data[-self._overlap:] if self._overlap > 0 else None
            return drive
        data = np.concatenate([self._tail, chunk.data])
        prev = len(self._tail)
        self._tail = chunk.data[-self._overlap:]
        work = WaveSample(data, chunk.sample_rate, chunk.wave_type, chunk.carrier, meta=chunk.meta)
        drive = self.frontend.couple(work)
        ratio = len(drive.signal) / max(len(data), 1)
        drop = min(int(round(prev * ratio)), max(len(drive.signal) - 2, 0))
        drive.signal = drive.signal[drop:]
        return drive

    def process(self, chunk: WaveSample) -> np.ndarray:
        if self.cochlea is None:                       # first block fixes the category
            f_lo = 40.0 if chunk.wave_type == "seismic" else self._f_lo
            self.cochlea = StreamingComplexGammatone(self.fs, f_lo=f_lo, n_channels=self._n)
            self.hair = StreamingHairCells(self.fs)
            self._category = chunk.wave_type
        drive = (self.frontend.couple(chunk) if chunk.wave_type == "gamma"
                 else self._couple_continuous(chunk))
        self.last_drive = drive.signal      # the wave itself, audible ("natural")

        activity = self.hair.process(np.abs(self.cochlea.process(drive.signal)))
        feats = _quick_features(activity, self.cochlea.cf, drive.aux, self.fs)
        self.last_features = feats
        params = map_features(feats)
        if self.synth is None:
            self.synth = StatefulSynth(self.fs, params)
        params = _smooth_params(self._params, params)
        self._params = params
        note_env = self.synth.bank.note_envelopes(activity, self.cochlea.cf, params)
        return self.synth.process(note_env, params)

    # convenience: run a whole WaveSample through the engine in chunks -------
    def render_sample(self, sample: WaveSample, block_seconds=0.12, declick_ms=4.0) -> np.ndarray:
        """Stream a whole sample through the engine and join the blocks.

        The synthesiser is phase-continuous, so a short ``declick_ms`` raised
        edge is all that is needed to remove the front-end's per-block
        resampling seam (not the 80 ms crossfade the old block engine needed to
        hide phase resets).
        """
        blocks = [self.process(b) for b in chunk_sample(sample, block_seconds)]
        if not blocks:
            return np.zeros((1, 2), np.float32)
        xf = int(declick_ms / 1000.0 * self.fs)
        out = blocks[0]
        for b in blocks[1:]:
            if xf > 0 and len(out) > xf and len(b) > xf:
                fade = np.linspace(0, 1, xf)[:, None]
                joined = out[-xf:] * (1 - fade) + b[:xf] * fade
                out = np.concatenate([out[:-xf], joined, b[xf:]], axis=0)
            else:
                out = np.concatenate([out, b], axis=0)
        return out


def chunk_sample(sample: WaveSample, block_seconds=0.12, min_native=2048):
    """Split a WaveSample into time-contiguous chunks for streaming.

    Low-rate modalities (seismic, infrared) use a minimum native-sample count so
    each block is large enough to resample cleanly; those phenomena are slow, so
    the resulting larger block latency is not a problem.
    """
    if sample.wave_type == "gamma":
        # Gamma is inherently event-based, not a continuous stream: process the
        # whole photon list as a single block through the stateful stages.
        yield sample
        return
    step = max(int(block_seconds * sample.sample_rate), min_native)
    data = sample.data
    for i in range(0, sample.n_samples, step):
        seg = data[i:i + step]
        if len(seg) < 2:
            continue
        yield WaveSample(seg, sample.sample_rate, sample.wave_type, sample.carrier,
                         meta=dict(sample.meta))
