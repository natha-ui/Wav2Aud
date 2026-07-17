"""Tests for the real-time live-output layer.

These never open an audio device, so they run headless in CI whether or not
``sounddevice`` (and PortAudio) is installed.
"""
import numpy as np
import pytest

from wave2aud import live, simulate
from wave2aud.sources import SimulatedSource


def test_available_reports_a_bool():
    assert isinstance(live.available(), bool)


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        live.LiveSonifier(mode="bogus")


@pytest.mark.parametrize("mode", ["musical", "natural"])
def test_render_without_a_device(mode):
    """render() sonifies a chunk without needing an output device."""
    ls = live.LiveSonifier(mode=mode)
    block = ls.render(simulate.preset("radar_car"))
    assert block.ndim == 2 and block.shape[1] == 2
    assert np.isfinite(block).all()
    assert np.abs(block).max() <= 1.0


def test_natural_and_musical_differ():
    sample = simulate.preset("seismic_M3_near")
    musical = live.LiveSonifier(mode="musical").render(sample)
    natural = live.LiveSonifier(mode="natural").render(sample)
    # both are audio, but they are not the same signal
    assert musical.shape[1] == natural.shape[1] == 2
    n = min(len(musical), len(natural))
    assert not np.allclose(musical[:n], natural[:n])


def test_push_without_open_raises():
    ls = live.LiveSonifier()
    with pytest.raises(RuntimeError, match="not open"):
        ls.push(simulate.preset("radar_car"))


def test_engine_exposes_the_drive_for_natural_mode():
    ls = live.LiveSonifier(mode="natural")
    ls.render(simulate.preset("radio_fm_music"))
    assert ls.engine.last_drive is not None
    assert np.isfinite(ls.engine.last_drive).all()


def test_render_streams_successive_chunks():
    """Successive chunks keep working (engine state carries over)."""
    ls = live.LiveSonifier()
    src = SimulatedSource("radar", n_chunks=3)
    blocks = [ls.render(s) for s in src.stream()]
    assert len(blocks) == 3
    assert all(np.isfinite(b).all() for b in blocks)
