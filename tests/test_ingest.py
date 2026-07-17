"""Tests for the 'bring your own waveform' ingestion + experience feature."""
import os

import numpy as np
import pytest
from scipy.io import wavfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wave2aud import ingest


@pytest.fixture
def demo_wav(tmp_path):
    fs = 16000
    t = np.linspace(0, 1.0, fs, endpoint=False)
    x = np.exp(-2 * t) * np.sin(2 * np.pi * 200 * t)
    p = str(tmp_path / "w.wav")
    wavfile.write(p, fs, (x / np.max(np.abs(x)) * 32767).astype(np.int16))
    return p


@pytest.fixture
def demo_image(tmp_path):
    t = np.linspace(0, 1, 1500)
    y = np.exp(-2.2 * t) * np.sin(2 * np.pi * 6 * t) + 0.4 * np.sin(2 * np.pi * 1.5 * t)
    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.plot(t, y, color="#111", lw=2); ax.axis("off"); fig.tight_layout(pad=0)
    p = str(tmp_path / "wave.png")
    fig.savefig(p, dpi=110); plt.close(fig)
    return p, y / np.max(np.abs(y))


def test_from_wav(demo_wav):
    s = ingest.from_wav(demo_wav, "radar")
    assert s.wave_type == "radar" and s.n_samples > 1000
    assert np.isfinite(s.data).all()


def test_from_wav_rejects_bad_type(demo_wav):
    with pytest.raises(ValueError):
        ingest.from_wav(demo_wav, "not_a_wave")


def test_image_trace_recovers_shape(demo_image):
    path, y_true = demo_image
    ex = ingest.trace_from_image(path, n_out=len(y_true))
    from scipy.stats import pearsonr
    assert pearsonr(y_true, ex)[0] > 0.9        # near-perfect after bbox crop


def test_from_image_all_types(demo_image):
    path, _ = demo_image
    for wt in ["radar", "radio", "infrared", "ultrasound", "gamma", "seismic"]:
        s = ingest.from_image(path, wt)
        assert s.wave_type == wt


def test_gamma_ingestion_makes_events(demo_image):
    path, _ = demo_image
    s = ingest.from_image(path, "gamma")
    assert "event_times" in s.meta and s.n_samples >= 1


def test_render_experience(demo_wav, tmp_path):
    s = ingest.from_wav(demo_wav, "ultrasound")
    out = str(tmp_path / "exp.html")
    ingest.render_experience(s, out)
    assert os.path.exists(out)
    html = open(out, encoding="utf-8").read()
    assert "__EXPERIENCE_JSON__" not in html      # placeholder was filled
    assert "data:audio/wav;base64," in html       # audio embedded
    assert '"wave_type": "ultrasound"' in html


def test_build_experience_data(demo_wav):
    s = ingest.from_wav(demo_wav, "seismic")
    d = ingest.build_experience_data(s)
    assert len(d["trace"]) == 1400
    assert d["duration"] > 0 and "features" in d
    assert d["audio"].startswith("data:audio/wav;base64,")
