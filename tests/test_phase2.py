"""Tests for the real-time engine, metrics, geometry and FFT baseline."""
import numpy as np
import pytest

import wave2aud as w2a
from wave2aud import simulate, metrics, geometry
from wave2aud.realtime import RealtimeSonifier, chunk_sample
from wave2aud.baseline import fft_sonify


CONTINUOUS = ["seismic_M3_near", "radar_car", "radio_fm_music", "ir_body", "us_mid_double"]


@pytest.mark.parametrize("preset", CONTINUOUS + ["gamma_cs137"])
def test_realtime_runs(preset):
    s = simulate.preset(preset)
    rt = RealtimeSonifier()
    audio = rt.render_sample(s, block_seconds=0.12)
    assert audio.ndim == 2 and audio.shape[1] == 2
    assert np.isfinite(audio).all()
    assert np.abs(audio).max() <= 1.0
    assert np.abs(audio).max() > 0.01


@pytest.mark.parametrize("preset", CONTINUOUS)
def test_realtime_seamless(preset):
    """Persistent phases => block joins are no worse than interior transitions."""
    s = simulate.preset(preset)
    rt = RealtimeSonifier()
    blocks = [rt.process(b) for b in chunk_sample(s, 0.12)]
    if len(blocks) < 2:
        pytest.skip("single block")
    audio = np.concatenate(blocks, axis=0).mean(axis=1)
    jumps = np.abs(np.diff(audio))
    bnd = np.cumsum([len(b) for b in blocks])[:-1]
    boundary = jumps[np.clip(bnd - 1, 0, len(jumps) - 1)].max()
    interior = np.percentile(jumps, 99.9)
    assert boundary <= 4.0 * interior + 0.06


def test_realtime_deterministic():
    a = RealtimeSonifier().render_sample(simulate.preset("radar_car"))
    b = RealtimeSonifier().render_sample(simulate.preset("radar_car"))
    assert np.allclose(a, b)


def test_ear_more_interpretable_than_fft():
    """Aggregated over categories the ear is smoother and more scale-conformant."""
    ear_rough, fft_rough, ear_conf, fft_conf = [], [], [], []
    for preset in CONTINUOUS + ["gamma_cs137"]:
        rep = metrics.interpretability(simulate.preset(preset))
        ear_rough.append(rep.ear["roughness"]); fft_rough.append(rep.fft["roughness"])
        ear_conf.append(rep.ear["scale_conformity"]); fft_conf.append(rep.fft["scale_conformity"])
    assert np.mean(ear_rough) < np.mean(fft_rough)
    assert np.mean(ear_conf) >= np.mean(fft_conf)


def test_category_separability_favours_ear():
    labels, ear, fft = [], [], []
    for name, s in simulate.all_presets().items():
        labels.append(s.wave_type)
        ear.append(w2a.sonify(s).audio)
        fft.append(fft_sonify(s)[0])
    assert metrics.category_separability(labels, ear, 44100) > \
        metrics.category_separability(labels, fft, 44100)


def test_mapping_monotonicity():
    """Audio pitch tracks a swept physical parameter (readable by ear)."""
    vels = [5, 10, 15, 20, 25, 30]
    res = [w2a.sonify(simulate.simulate_radar(velocity_ms=v)) for v in vels]
    rho, _ = metrics.mapping_monotonicity(vels, res)
    assert rho > 0.85


def test_preservation_scores_present():
    rep = metrics.compare(w2a.sonify(simulate.preset("seismic_M3_near")))
    assert set(["envelope_r", "centroid_contour_rho"]).issubset(rep.preservation)
    assert -1.0 <= rep.preservation["envelope_r"] <= 1.0


def test_geometry_shapes():
    g = geometry.geometry_report(w2a.sonify(simulate.preset("radar_car")), dims=3)
    assert g.wave_embedding.shape[1] == 3 and g.audio_embedding.shape[1] == 3
    for tr in (g.wave_trajectory, g.audio_trajectory):
        assert len(tr["pitch_midi"]) == len(tr["entropy"]) == len(tr["time"])
