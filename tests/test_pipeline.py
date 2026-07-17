"""End-to-end tests for the wave2aud pipeline."""
import numpy as np
import pytest

import wave2aud as w2a
from wave2aud import simulate
from wave2aud.waves import WAVE_TYPES
from wave2aud.ros import WaveBridge


ONE_PER_TYPE = {
    "radar": "radar_car", "radio": "radio_fm_music", "infrared": "ir_body",
    "ultrasound": "us_mid_double", "gamma": "gamma_cs137", "seismic": "seismic_M3_near",
}


@pytest.mark.parametrize("wave_type,preset", list(ONE_PER_TYPE.items()))
def test_sonify_produces_valid_stereo(wave_type, preset):
    result = w2a.sonify(simulate.preset(preset))
    a = result.audio
    assert a.ndim == 2 and a.shape[1] == 2
    assert a.shape[0] > 1000
    assert np.isfinite(a).all()
    assert np.abs(a).max() <= 1.0
    assert np.abs(a).max() > 0.01           # not silent
    assert result.params.category == wave_type


def test_every_wave_type_has_a_voice():
    for wt in WAVE_TYPES:
        assert wt in w2a.CATEGORY_VOICES


def test_register_ordering_matches_categories():
    """Distinct sonic identity: registers are ordered seismic < ... < gamma."""
    centers = {}
    for wt, preset in ONE_PER_TYPE.items():
        p = w2a.sonify(simulate.preset(preset)).params
        centers[wt] = 0.5 * (p.register_lo_midi + p.register_hi_midi)
    assert centers["seismic"] < centers["radio"] < centers["gamma"]
    assert centers["seismic"] < centers["ultrasound"]


def test_determinism():
    a1 = w2a.sonify(simulate.preset("gamma_cs137")).audio
    a2 = w2a.sonify(simulate.preset("gamma_cs137")).audio
    assert np.allclose(a1, a2)


def test_within_category_closer_than_across():
    """Comparable waves map near each other; different categories map far apart."""
    son = w2a.Sonifier()
    labels, vecs = [], []
    for name, sample in simulate.all_presets().items():
        f = son.sonify(sample).features
        labels.append(sample.wave_type)
        vecs.append([f.brightness, f.loudness, np.log2(max(f.centroid_hz, 1)),
                     f.flatness, min(f.onset_rate, 40) / 40.0])
    X = np.array(vecs)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)   # z-score each feature
    labels = np.array(labels)
    within, across = [], []
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            d = np.linalg.norm(X[i] - X[j])
            (within if labels[i] == labels[j] else across).append(d)
    assert np.mean(within) < np.mean(across)


def test_streaming_crossfade():
    src = simulate.simulate_radar
    stream = w2a.StreamingSonifier()
    blocks = [stream.push(src(velocity_ms=10 + 3 * k, seed=k)) for k in range(3)]
    for b in blocks:
        assert b.shape[1] == 2 and np.isfinite(b).all()


def test_ros_bridge_without_rclpy():
    """The ROS core works headless (no ROS2 needed for CI)."""
    s = simulate.preset("seismic_M3_near")
    bridge = WaveBridge("seismic", s.sample_rate)
    audio, feats = bridge.process(np.real(s.data), s.meta)
    assert audio.ndim == 1 and audio.size % 2 == 0    # interleaved stereo
    assert set(["loudness", "brightness", "tempo_bpm"]).issubset(feats)


def test_complex_input_roundtrip():
    s = simulate.simulate_radar(velocity_ms=20)
    assert s.is_complex
    result = w2a.sonify(s)
    assert np.isfinite(result.audio).all()
