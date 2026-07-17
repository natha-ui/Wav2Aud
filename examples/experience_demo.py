"""Demo the 'bring your own waveform' feature end to end.

Creates a sample .wav and a sample waveform *image*, ingests each (declaring a
wave type), and renders an interactive 3-D experience page for both.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile

from wave2aud import ingest

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "experiences")
os.makedirs(OUT, exist_ok=True)


def make_demo_wav(path):
    fs = 16000
    t = np.linspace(0, 2.2, int(fs * 2.2), endpoint=False)
    # a few decaying modes -> an interesting attractor
    x = (np.exp(-1.6 * t) * np.sin(2 * np.pi * (180 + 260 * t) * t)
         + 0.5 * np.exp(-1.0 * t) * np.sin(2 * np.pi * 90 * t)
         + 0.25 * np.exp(-3.0 * t) * np.sin(2 * np.pi * 520 * t))
    x = x / np.max(np.abs(x))
    wavfile.write(path, fs, (x * 32767).astype(np.int16))
    return path


def make_demo_image(path):
    t = np.linspace(0, 1, 2000)
    y = (np.exp(-2.2 * t) * np.sin(2 * np.pi * 6 * t)
         + 0.4 * np.sin(2 * np.pi * 1.5 * t))
    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.plot(t, y, color="#111", lw=2)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main():
    wav = make_demo_wav(os.path.join(OUT, "demo_input.wav"))
    img = make_demo_image(os.path.join(OUT, "demo_input_waveform.png"))
    print("inputs:", os.path.basename(wav), "+", os.path.basename(img))

    # a .wav declared as ultrasound
    s1 = ingest.from_wav(wav, "ultrasound")
    ingest.render_experience(s1, os.path.join(OUT, "experience_wav_ultrasound.html"))
    print("  wav  -> experience_wav_ultrasound.html", s1)

    # an image of a waveform declared as seismic
    s2 = ingest.from_image(img, "seismic")
    ingest.render_experience(s2, os.path.join(OUT, "experience_image_seismic.html"))
    print("  image-> experience_image_seismic.html", s2)

    for f in ("experience_wav_ultrasound.html", "experience_image_seismic.html"):
        kb = os.path.getsize(os.path.join(OUT, f)) / 1024
        print(f"  {f}: {kb:.0f} KB")
    print("done ->", OUT)


if __name__ == "__main__":
    main()
