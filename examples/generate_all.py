"""Generate every demo clip, a pipeline figure per category, and the feature map.

Run::

    python examples/generate_all.py
"""
from __future__ import annotations

import os

import wav2aud as w2a
from wav2aud import simulate, viz

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO = os.path.join(HERE, "output", "audio")
FIGS = os.path.join(HERE, "output", "figures")

# one representative preset per category to render a full pipeline figure for
REPRESENTATIVE = [
    "seismic_M7_far", "radar_drone", "radio_fm_music",
    "ir_engine", "us_far_clutter", "gamma_cs137",
]


def main():
    os.makedirs(AUDIO, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)
    son = w2a.Sonifier()

    results = {}
    print("Sonifying presets ->", AUDIO)
    for name, sample in simulate.all_presets().items():
        r = son.sonify(sample)
        r.write(os.path.join(AUDIO, f"{name}.wav"))
        results[name] = r
        f = r.features
        print(f"  {name:22s} bright={f.brightness:.2f} loud={f.loudness:.2f} "
              f"onset/s={f.onset_rate:4.1f} tempo={f.tempo_bpm:3.0f}")

    print("\nRendering pipeline figures ->", FIGS)
    for name in REPRESENTATIVE:
        viz.plot_pipeline(results[name], os.path.join(FIGS, f"pipeline_{name}.png"))
        print(f"  pipeline_{name}.png")

    viz.plot_feature_map(list(results.values()), os.path.join(FIGS, "feature_map.png"))
    print("  feature_map.png")

    print("\nRendering analysis figures ->", FIGS)
    # geometry (birdsong-style) + wave<->audio comparison for a couple of waves
    for name in ("radar_car", "seismic_M7_far"):
        viz.plot_geometry(results[name], os.path.join(FIGS, f"geometry_{name}.png"))
        viz.plot_wave_audio_comparison(results[name], os.path.join(FIGS, f"compare_{name}.png"))
        print(f"  geometry_{name}.png / compare_{name}.png")
    # interpretability: ear vs FFT baseline
    for name in ("radar_drone", "gamma_cs137"):
        viz.plot_interpretability(simulate.preset(name), os.path.join(FIGS, f"interp_{name}.png"))
        print(f"  interp_{name}.png")

    print(f"\nDone: {len(results)} clips + pipeline/analysis figures in {FIGS}.")


if __name__ == "__main__":
    main()
