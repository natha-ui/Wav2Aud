"""Command-line interface for wave2aud."""
from __future__ import annotations

import argparse
import os

from . import simulate
from .pipeline import Sonifier


def _cmd_list(_args):
    print("Wave types :", ", ".join(sorted(set(s.wave_type for s in [])) or []) or
          "radar, radio, infrared, ultrasound, gamma, seismic")
    print("Presets    :")
    for name in sorted(simulate.PRESETS):
        print("  -", name)


def _cmd_sonify(args):
    if args.preset:
        sample = simulate.preset(args.preset)
    else:
        gen = {
            "seismic": simulate.simulate_seismic, "radar": simulate.simulate_radar,
            "radio": simulate.simulate_radio, "infrared": simulate.simulate_infrared,
            "ultrasound": simulate.simulate_ultrasound, "gamma": simulate.simulate_gamma,
        }[args.type]
        sample = gen()
    son = Sonifier(fs=args.fs)
    result = son.sonify(sample)
    out = args.out or f"{args.preset or args.type}.wav"
    result.write(out)
    print(f"wrote {out}  ({result.audio.shape[0]/result.fs:.2f}s, {sample.wave_type})")
    if args.figure:
        from . import viz
        viz.plot_pipeline(result, args.figure)
        print(f"wrote {args.figure}")


def _cmd_experience(args):
    from . import ingest
    if not args.input:
        raise SystemExit("provide --input path to a .wav or an image of a waveform")
    sample = ingest.load(args.input, args.type)
    out = args.out or "experience.html"
    ingest.render_experience(sample, out, fs=args.fs)
    print(f"wrote {out}  ({sample.wave_type}: {sample.label()})")
    print("open it in a browser: a 3-D visualisation that plays the sonification.")


def _cmd_demo(args):
    from . import viz
    os.makedirs(args.out, exist_ok=True)
    son = Sonifier(fs=args.fs)
    results = []
    for name, sample in simulate.all_presets().items():
        r = son.sonify(sample)
        r.write(os.path.join(args.out, f"{name}.wav"))
        if args.figures:
            viz.plot_pipeline(r, os.path.join(args.out, f"{name}.png"))
        results.append(r)
        print(f"  {name:22s} -> {name}.wav")
    if args.figures:
        viz.plot_feature_map(results, os.path.join(args.out, "_feature_map.png"))
    print(f"done: {len(results)} clips in {args.out}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="wave2aud", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="list wave types and presets")
    p.set_defaults(func=_cmd_list)

    p = sub.add_parser("sonify", help="sonify one wave to a WAV file")
    p.add_argument("--type", choices=["radar", "radio", "infrared", "ultrasound", "gamma", "seismic"])
    p.add_argument("--preset", help="named preset (see `wave2aud list`)")
    p.add_argument("--out", help="output .wav path")
    p.add_argument("--figure", help="also write a pipeline figure to this path")
    p.add_argument("--fs", type=float, default=44100.0)
    p.set_defaults(func=_cmd_sonify)

    p = sub.add_parser("experience", help="3-D visual + audio from your own .wav or waveform image")
    p.add_argument("--type", required=True,
                   choices=["radar", "radio", "infrared", "ultrasound", "gamma", "seismic"],
                   help="which wave category your input is")
    p.add_argument("--input", required=True, help="path to a .wav or an image of a waveform")
    p.add_argument("--out", help="output .html path")
    p.add_argument("--fs", type=float, default=44100.0)
    p.set_defaults(func=_cmd_experience)

    p = sub.add_parser("demo", help="sonify every preset (and optionally figures)")
    p.add_argument("--out", default="wave2aud_out", help="output directory")
    p.add_argument("--figures", action="store_true", help="also render pipeline figures")
    p.add_argument("--fs", type=float, default=44100.0)
    p.set_defaults(func=_cmd_demo)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
