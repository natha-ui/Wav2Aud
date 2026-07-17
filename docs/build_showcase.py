"""Build the self-contained interactive showcase HTML.

Generates the biomimetic-ear clips (one per category) and a few FFT-baseline
clips for A/B listening, computes the real interpretability / monotonicity /
separability statistics, and injects everything into
``_showcase_template.html`` -> ``wave2aud_showcase.html``.
"""
from __future__ import annotations

import base64
import io
import json
import os
import wave

import numpy as np
from scipy.signal import resample_poly

import wave2aud as w2a
from wave2aud import simulate, metrics
from wave2aud.baseline import fft_sonify

HERE = os.path.dirname(os.path.abspath(__file__))

CLIPS = {
    "seismic": "seismic_M3_near", "audio": "audio_vowel_low", "radio": "radio_fm_music",
    "infrared": "ir_body", "radar": "radar_car", "gamma": "gamma_cs137",
    "ultrasound": "us_mid_double",
}
AB_CATEGORIES = ["radar", "gamma", "radio"]     # ear-vs-FFT A/B demos
OUT_FS = 22050


def to_wav_datauri(audio, fs):
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    from math import gcd
    up, down = OUT_FS, int(round(fs))
    g = gcd(up, down)
    mono = resample_poly(mono, up // g, down // g)
    peak = np.max(np.abs(mono)) or 1.0
    ints = (np.clip(mono / peak * 0.97, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(OUT_FS)
        wf.writeframes(ints.tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    son = w2a.Sonifier()
    audio_ear, results = {}, {}
    print("Ear clips:")
    for cat, preset in CLIPS.items():
        r = son.sonify(simulate.preset(preset))
        results[cat] = r
        audio_ear[cat] = to_wav_datauri(r.audio, r.fs)
        print(f"  {cat:11s} {len(audio_ear[cat]) / 1024:6.1f} KB")

    print("FFT-baseline A/B clips:")
    audio_fft = {}
    for cat in AB_CATEGORIES:
        fftk, _ = fft_sonify(simulate.preset(CLIPS[cat]))
        audio_fft[cat] = to_wav_datauri(fftk, 44100)
        print(f"  {cat:11s} {len(audio_fft[cat]) / 1024:6.1f} KB")

    print("Example waves (for the studio 'try' buttons)...")

    def _pack(wave_type, data, fs):
        data = np.asarray(data, float)
        data = data / (np.max(np.abs(data)) + 1e-9)
        return {"wave_type": wave_type, "fs": round(float(fs), 2),
                "data": [round(float(v), 4) for v in data]}

    def example_trace(sample, n=4000):
        x = np.real(sample.data).astype(float)
        idx = np.linspace(0, len(x) - 1, min(n, len(x)))
        y = np.interp(idx, np.arange(len(x)), x)
        fs_out = len(y) * sample.sample_rate / len(x)
        return _pack(sample.wave_type, y, fs_out)

    def gamma_example(n=4000):
        s = simulate.simulate_gamma(7.0, lines=(662.0,), duration=6.0)
        times = np.asarray(s.meta.get("event_times", []), float)
        energies = np.real(s.data).astype(float)
        if times.size == 0:
            times, energies = np.array([0.5, 1.5, 2.5]), np.array([400., 600., 662.])
        span = float(times.max() - times.min()) or 1.0
        y = np.zeros(n)
        for tt, e in zip(times, energies):
            y[int((tt - times.min()) / span * (n - 1))] = e
        return _pack("gamma", y, n / span)

    def ultrasound_example(n=4000, fs=8000.0):
        y = np.zeros(n)
        for k, delay in enumerate((0.06, 0.14, 0.22)):
            i, L = int(delay * fs), int(0.012 * fs)
            if i + L < n:
                y[i:i + L] += np.hanning(L) * np.sin(2 * np.pi * 1600 * np.arange(L) / fs) * (0.7 ** k)
        return _pack("ultrasound", y, fs)

    examples = {
        "seismic (quake)": example_trace(simulate.simulate_seismic(4.5, 30, duration=15)),
        "audio (voice)": example_trace(simulate.simulate_audio("vowel", 160)),
        "radio (FM)": example_trace(simulate.simulate_radio("FM", (3.0, 5.0))),
        "infrared (thermal)": example_trace(simulate.simulate_infrared(320)),
        "radar (car)": example_trace(simulate.simulate_radar(velocity_ms=14)),
        "gamma (Cs-137)": gamma_example(),
        "ultrasound (echoes)": ultrasound_example(),
    }
    for k, e in examples.items():
        print(f"  {k:18s} {len(e['data'])} pts @ {e['fs']:.0f} Hz")

    print("Computing statistics...")
    vels = [5, 10, 15, 20, 25, 30]
    vel_rho, _ = metrics.mapping_monotonicity(
        vels, [son.sonify(simulate.simulate_radar(velocity_ms=v)) for v in vels])
    mags = [3.0, 4.0, 5.0, 6.0, 7.0]
    mag_rho, _ = metrics.mapping_monotonicity(
        mags, [son.sonify(simulate.simulate_seismic(magnitude=m, distance_km=30, duration=15)) for m in mags])

    labels, ear_a, fft_a = [], [], []
    for name, s in simulate.all_presets().items():
        labels.append(s.wave_type)
        ear_a.append(son.sonify(s).audio)
        fft_a.append(fft_sonify(s)[0])
    sep_ear = metrics.category_separability(labels, ear_a, 44100)
    sep_fft = metrics.category_separability(labels, fft_a, 44100)

    er, fr, ec, fc = [], [], [], []
    for cat, preset in CLIPS.items():
        rep = metrics.interpretability(simulate.preset(preset))
        er.append(rep.ear["roughness"]); fr.append(rep.fft["roughness"])
        ec.append(rep.ear["scale_conformity"]); fc.append(rep.fft["scale_conformity"])

    stats = {
        "vel_pitch_rho": round(float(vel_rho), 2),
        "mag_pitch_rho": round(float(mag_rho), 2),
        "sep_ear": round(sep_ear, 2), "sep_fft": round(sep_fft, 2),
        "rough_ear": round(float(np.mean(er)), 3), "rough_fft": round(float(np.mean(fr)), 3),
        "conf_ear": round(float(np.mean(ec)), 2), "conf_fft": round(float(np.mean(fc)), 2),
    }
    print("  stats:", stats)

    with open(os.path.join(HERE, "_showcase_template.html"), encoding="utf-8") as f:
        tpl = f.read()
    html = (tpl.replace("__AUDIO_JSON__", json.dumps(audio_ear))
               .replace("__AUDIO_FFT_JSON__", json.dumps(audio_fft))
               .replace("__STATS_JSON__", json.dumps(stats))
               .replace("__EXAMPLES_JSON__", json.dumps(examples)))
    out = os.path.join(HERE, "wave2aud_showcase.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {out}  ({len(html) / 1024:.0f} KB)")

    # keep the deployable copy (served as the site root) in sync
    site = os.path.join(os.path.dirname(HERE), "site", "index.html")
    os.makedirs(os.path.dirname(site), exist_ok=True)
    with open(site, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {site}")


if __name__ == "__main__":
    main()
