"""Visualisation of the wave -> audio interpretation.

``plot_pipeline`` renders the whole journey for one wave: raw physical signal ->
coupled ear drive -> cochleagram (what the ear resolves) -> the retuned musical
notes -> the output audio waveform and spectrogram. ``plot_feature_map`` shows
how the six categories separate in perceptual-feature space (distinct sonic
identities) while comparable waves cluster.
"""
from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .pipeline import SonifyResult  # noqa: E402
from .synthesis import retune_channels  # noqa: E402

CATEGORY_COLORS = {
    "radar": "#4dd0e1", "radio": "#ffb74d", "infrared": "#e57373",
    "ultrasound": "#81c784", "gamma": "#ba68c8", "seismic": "#7986cb",
}


def _spectrogram(ax, mono, fs, title):
    ax.specgram(mono, NFFT=1024, Fs=fs, noverlap=512, cmap="magma")
    ax.set_ylim(0, min(8000, fs / 2))
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("time (s)", fontsize=8)
    ax.set_ylabel("Hz", fontsize=8)


def plot_pipeline(result: SonifyResult, path: str | None = None, show_note_names=True):
    r = result
    color = CATEGORY_COLORS.get(r.sample.wave_type, "#888")
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    fig.suptitle(
        f"wave2aud  |  {r.sample.wave_type.upper()}  |  {r.sample.label()}",
        fontsize=14, fontweight="bold",
    )

    # 1. raw physical wave
    ax = axes[0, 0]
    raw = r.sample.data
    if r.sample.wave_type == "gamma":
        times = r.sample.meta.get("event_times", np.arange(len(raw)))
        ax.stem(times, np.real(raw), linefmt=color, markerfmt="o", basefmt=" ")
        ax.set_ylabel("energy (keV)", fontsize=8)
        ax.set_xlabel("time (s)", fontsize=8)
    else:
        y = np.real(raw)[:4000]
        tt = np.arange(len(y)) / r.sample.sample_rate
        ax.plot(tt, y, color=color, lw=0.7)
        ax.set_ylabel("amplitude", fontsize=8)
        ax.set_xlabel("time (s)", fontsize=8)
    ax.set_title(f"1. Raw wave  (fs={r.sample.sample_rate:g} Hz)", fontsize=9)

    # 2. coupled ear drive
    ax = axes[0, 1]
    d = r.drive.signal[: int(0.3 * r.drive.fs)]
    ax.plot(np.arange(len(d)) / r.drive.fs, d, color=color, lw=0.6)
    ax.set_title("2. Coupled drive at the eardrum (audio rate)", fontsize=9)
    ax.set_xlabel("time (s)", fontsize=8)

    # 3. cochleagram
    ax = axes[1, 0]
    env = r.image.envelopes
    cf = r.image.cf
    extent = [0, env.shape[1] / r.image.control_rate, 0, env.shape[0]]
    ax.imshow(env, aspect="auto", origin="lower", cmap="viridis", extent=extent)
    ax.set_title("3. Cochleagram (basilar-membrane / hair-cell activity)", fontsize=9)
    ax.set_xlabel("time (s)", fontsize=8)
    ax.set_ylabel("cochlear channel", fontsize=8)

    # 4. channel energy -> retuned notes
    ax = axes[1, 1]
    ch_e = env.mean(axis=1)
    ch_e = ch_e / (ch_e.max() + 1e-9)
    f_out = retune_channels(cf, r.params)
    sc = ax.scatter(cf, f_out, s=8 + 120 * ch_e, c=[color], alpha=0.7, edgecolors="k", linewidths=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("4. Retuning: cochlear channel -> musical note", fontsize=9)
    ax.set_xlabel("channel centre freq (Hz)", fontsize=8)
    ax.set_ylabel("output note (Hz)", fontsize=8)
    ax.axhspan(
        440 * 2 ** ((r.params.register_lo_midi - 69) / 12),
        440 * 2 ** ((r.params.register_hi_midi - 69) / 12),
        color=color, alpha=0.12,
    )

    # 5. output waveform
    ax = axes[2, 0]
    mono = r.audio.mean(axis=1)
    ax.plot(np.arange(len(mono)) / r.fs, mono, color=color, lw=0.4)
    ax.set_title("5. Output audio waveform (stereo mix)", fontsize=9)
    ax.set_xlabel("time (s)", fontsize=8)

    # 6. output spectrogram
    _spectrogram(axes[2, 1], mono, r.fs, "6. Output spectrogram (the music)")

    # parameter caption
    p = r.params
    cap = (
        f"register MIDI {p.register_lo_midi:.0f}-{p.register_hi_midi:.0f} | "
        f"tempo {p.tempo_bpm:.0f} bpm | reverb {p.reverb:.2f} | "
        f"noise {p.noise_content:.2f} | tremolo {p.tremolo_depth:.2f} | "
        f"vibrato {p.vibrato_depth:.2f} st | pan {p.pan:+.2f}"
    )
    fig.text(0.5, 0.005, cap, ha="center", fontsize=8, color="#444")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    if path:
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path
    return fig


def plot_geometry(result, path: str | None = None):
    """Birdsong-style geometry: delay-embedding attractors + gesture trajectories."""
    from .geometry import geometry_report
    g = geometry_report(result, dims=2)
    color = CATEGORY_COLORS.get(result.sample.wave_type, "#888")
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    fig.suptitle(f"Geometric interpretation  |  {result.sample.wave_type.upper()}  |  {result.sample.label()}",
                 fontsize=13, fontweight="bold")

    for ax, emb, name in [(axes[0, 0], g.wave_embedding, "Wave"),
                          (axes[0, 1], g.audio_embedding, "Audio")]:
        ax.plot(emb[:, 0], emb[:, 1], lw=0.35, color=color, alpha=0.8)
        ax.set_title(f"{name} attractor  (x(t) vs x(t+τ))", fontsize=10)
        ax.set_xlabel("x(t)", fontsize=8); ax.set_ylabel("x(t+τ)", fontsize=8)
        ax.set_aspect("equal", "box")

    for ax, tr, name in [(axes[1, 0], g.wave_trajectory, "Wave"),
                         (axes[1, 1], g.audio_trajectory, "Audio")]:
        sc = ax.scatter(tr["entropy"], tr["pitch_midi"], c=tr["time"], cmap="viridis",
                        s=10, alpha=0.8)
        ax.set_title(f"{name} gesture  (Wiener entropy vs pitch)", fontsize=10)
        ax.set_xlabel("Wiener entropy → noisy", fontsize=8)
        ax.set_ylabel("pitch (MIDI)", fontsize=8)
        fig.colorbar(sc, ax=ax, label="time (s)", fraction=0.046)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if path:
        fig.savefig(path, dpi=110); plt.close(fig); return path
    return fig


def plot_interpretability(sample, path: str | None = None, fs: float = 44100.0):
    """Biomimetic ear vs naive FFT baseline: spectrograms + interpretability bars."""
    from .baseline import fft_sonify
    from .metrics import interpretability
    from .pipeline import Sonifier

    ear = Sonifier(fs=fs).sonify(sample).audio.mean(axis=1)
    fftk = fft_sonify(sample, fs=fs)[0].mean(axis=1)
    rep = interpretability(sample, fs=fs)
    color = CATEGORY_COLORS.get(sample.wave_type, "#888")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Interpretability: biomimetic ear vs FFT  |  {sample.wave_type.upper()}",
                 fontsize=13, fontweight="bold")
    _spectrogram(axes[0, 0], ear, fs, "Biomimetic ear — scale-locked, category voice")
    _spectrogram(axes[0, 1], fftk, fs, "FFT baseline — raw spectral tones")

    labels = ["roughness\n(lower better)", "scale conformity\n(higher better)"]
    ear_v = [rep.ear["roughness"], rep.ear["scale_conformity"]]
    fft_v = [rep.fft["roughness"], rep.fft["scale_conformity"]]
    x = np.arange(len(labels))
    axes[1, 0].bar(x - 0.2, ear_v, 0.4, label="ear", color=color)
    axes[1, 0].bar(x + 0.2, fft_v, 0.4, label="FFT", color="#999")
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(labels, fontsize=8)
    axes[1, 0].legend(fontsize=8); axes[1, 0].set_title("Interpretability metrics", fontsize=10)

    axes[1, 1].axis("off")
    txt = ("The ear output is quantitatively:\n\n"
           f"  • smoother  (roughness {rep.ear['roughness']:.3f} vs {rep.fft['roughness']:.3f})\n"
           f"  • more in-tune  (scale conf. {rep.ear['scale_conformity']:.2f} vs {rep.fft['scale_conformity']:.2f})\n\n"
           "and keeps a fixed category timbre, so different\nwaves stay distinguishable by ear — the FFT\n"
           "baseline sounds like raw spectra with no identity.")
    axes[1, 1].text(0.0, 0.9, txt, fontsize=10, va="top", family="monospace")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if path:
        fig.savefig(path, dpi=110); plt.close(fig); return path
    return fig


def plot_wave_audio_comparison(result, path: str | None = None):
    """Quantitative side-by-side: envelope & centroid-contour tracking + descriptors."""
    from .metrics import compare, _envelope, _centroid_contour
    rep = compare(result)
    color = CATEGORY_COLORS.get(result.sample.wave_type, "#888")
    drive, audio, fs = result.drive.signal, result.audio, result.fs

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Wave ↔ audio comparison  |  {result.sample.wave_type.upper()}  |  {result.sample.label()}",
                 fontsize=13, fontweight="bold")

    de, ae = _envelope(drive, fs), _envelope(audio, fs)
    ax = axes[0, 0]
    ax.plot(de / (de.max() + 1e-9), color="#888", lw=1.2, label="wave envelope")
    ax.plot(ae / (ae.max() + 1e-9), color=color, lw=1.2, label="audio envelope")
    ax.set_title(f"Amplitude envelope  (r = {rep.preservation['envelope_r']:+.2f})", fontsize=10)
    ax.legend(fontsize=8); ax.set_xlabel("time →", fontsize=8)

    dc, ac = _centroid_contour(drive, fs), _centroid_contour(audio, fs)
    ax = axes[0, 1]
    ax.plot((dc - dc.mean()) / (dc.std() + 1e-9), color="#888", lw=1.2, label="wave centroid")
    ax.plot((ac - ac.mean()) / (ac.std() + 1e-9), color=color, lw=1.2, label="audio centroid")
    ax.set_title(f"Spectral-centroid contour  (ρ = {rep.preservation['centroid_contour_rho']:+.2f})", fontsize=10)
    ax.legend(fontsize=8); ax.set_xlabel("time →", fontsize=8)

    rows = rep.rows()
    labels = [r[0] for r in rows]
    wv = np.array([r[1] for r in rows]); av = np.array([r[2] for r in rows])
    norm = np.maximum(np.abs(wv), np.abs(av)) + 1e-9
    y = np.arange(len(labels))
    ax = axes[1, 0]
    ax.barh(y - 0.2, wv / norm, 0.4, color="#888", label="wave")
    ax.barh(y + 0.2, av / norm, 0.4, color=color, label="audio")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Descriptors (normalised per row)", fontsize=10); ax.legend(fontsize=8)

    ax = axes[1, 1]; ax.axis("off")
    lines = ["measure            wave        audio", "-" * 40]
    for lab, w, a in rows:
        lines.append(f"{lab[:16]:16s} {w:9.2f} {a:9.2f}")
    lines += ["", f"envelope r        {rep.preservation['envelope_r']:+.2f}",
              f"centroid ρ        {rep.preservation['centroid_contour_rho']:+.2f}"]
    ax.text(0.0, 0.95, "\n".join(lines), fontsize=9, va="top", family="monospace")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if path:
        fig.savefig(path, dpi=110); plt.close(fig); return path
    return fig


def plot_category_similarity(path: str | None = None):
    """Heatmap of the category-tone distance matrix (the similarity structure)."""
    from .metrics import category_tone_matrix
    labels, M = category_tone_matrix()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(M, cmap="magma")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                    color="white" if M[i, j] > M.max() * 0.5 else "black", fontsize=8)
    ax.set_title("Category tone distance\n(similar categories close; seismic ↔ ultrasound the extremes)", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, label="perceptual distance")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110); plt.close(fig); return path
    return fig


def plot_feature_map(results, path: str | None = None):
    """Scatter categories in (brightness, loudness) with onset-rate as size."""
    fig, ax = plt.subplots(figsize=(9, 7))
    seen = set()
    for r in results:
        wt = r.sample.wave_type
        f = r.features
        ax.scatter(
            f.brightness, f.centroid_hz, s=40 + 400 * f.loudness,
            c=[CATEGORY_COLORS.get(wt, "#888")], alpha=0.75, edgecolors="k",
            linewidths=0.4, label=wt if wt not in seen else None,
        )
        seen.add(wt)
        ax.annotate(r.sample.label(), (f.brightness, f.centroid_hz), fontsize=6,
                    alpha=0.6, xytext=(3, 3), textcoords="offset points")
    ax.set_yscale("log")
    ax.set_xlabel("brightness (perceived)")
    ax.set_ylabel("spectral centroid (Hz, log)")
    ax.set_title("Category identity in perceptual space\n(marker size = loudness)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path
    return fig
