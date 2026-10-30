#!/usr/bin/env python3
"""Places kernels on a roofline using peaks cached by measure_peaks.py.

Each kernel is given as NAME:FLOPS:BYTES:MS, where FLOPS and BYTES are the kernel's total useful
floating-point operations and DRAM bytes moved (counted analytically), and MS is its measured time.

Examples:
  roofline.py --kernel matvec:2.0e8:4.0e8:0.25
  roofline.py --compute bf16_tc --kernel v1:1.1e12:3e9:4.1 --kernel v2:1.1e12:6e8:2.9 --plot r.png
"""

import argparse
import json
import math
import sys
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / ".cache"
COMPUTE = ["fp32", "fp64", "fp16", "tf32_tc", "bf16_tc", "fp16_tc", "fp32_sgemm"]


def load_peaks(path: str | None) -> tuple[dict, Path]:
    if path:
        p = Path(path)
    else:
        files = sorted(CACHE.glob("peaks-*.json"), key=lambda f: f.stat().st_mtime)
        if not files:
            sys.exit(f"No cached peaks in {CACHE}; run measure_peaks.py first.")
        p = files[-1]
    return json.loads(p.read_text()), p


def parse_kernel(spec: str) -> tuple[str, float, float, float]:
    try:
        name, flops, nbytes, ms = spec.rsplit(":", 3)
        return name, float(flops), float(nbytes), float(ms)
    except ValueError:
        sys.exit(f"Bad --kernel {spec!r}; expected NAME:FLOPS:BYTES:MS")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--kernel", action="append", required=True, help="NAME:FLOPS:BYTES:MS")
    parser.add_argument(
        "--compute", choices=COMPUTE, default="fp32", help="compute roof (default fp32)"
    )
    parser.add_argument(
        "--memory", choices=["dram", "l2"], default="dram", help="memory roof (default dram)"
    )
    parser.add_argument(
        "--theoretical", action="store_true", help="use spec-sheet peaks, not measured"
    )
    parser.add_argument("--peaks", help="peaks JSON (default: newest file in .cache/)")
    parser.add_argument("--plot", help="write a log-log roofline plot to this PNG path")
    args = parser.parse_args()

    peaks, peaks_path = load_peaks(args.peaks)
    if args.theoretical:
        if args.memory != "dram" or args.compute not in ("fp32", "fp64"):
            sys.exit("--theoretical only covers --memory dram with --compute fp32 or fp64.")
        peak_gflops = peaks[f"theoretical_{args.compute}_gflops"]
        peak_gbs = peaks["theoretical_dram_gbs"]
    else:
        peak_gflops = peaks[f"{args.compute}_gflops"]
        # The memory roof is the best bandwidth any access pattern achieved.
        keys = (
            ["dram_read_gbs", "dram_write_gbs", "dram_copy_gbs"]
            if args.memory == "dram"
            else ["l2_read_gbs"]
        )
        peak_gbs = max(peaks[k] for k in keys)
    ridge = peak_gflops / peak_gbs

    kind = "theoretical" if args.theoretical else "measured"
    print(f"{peaks['name']} ({kind} peaks from {peaks_path.name}, {peaks.get('measured_at', '?')})")
    print(
        f"  {args.compute} roof {peak_gflops:,.0f} GFLOP/s, "
        f"{args.memory} roof {peak_gbs:,.0f} GB/s, ridge {ridge:.1f} FLOP/byte\n"
    )
    header = (
        f"{'kernel':<20}{'FLOP/B':>9}{'GFLOP/s':>12}{'GB/s':>10}{'roof GFLOP/s':>14}{'% roof':>9}"
        "  bound"
    )
    print(header)
    print("-" * len(header))

    points = []
    for spec in args.kernel:
        name, flops, nbytes, ms = parse_kernel(spec)
        s = ms / 1e3
        gflops = flops / s / 1e9
        gbs = nbytes / s / 1e9
        ai = flops / nbytes if nbytes else math.inf
        roof = min(peak_gflops, ai * peak_gbs)
        bound = "memory" if ai < ridge else "compute"
        # With no FLOPs (e.g. a copy) the useful comparison is against the bandwidth roof.
        pct = 100 * gbs / peak_gbs if flops == 0 else 100 * gflops / roof
        print(
            f"{name:<20}{ai:>9.2f}{gflops:>12,.1f}{gbs:>10,.1f}{roof:>14,.1f}{pct:>8.1f}%  {bound}"
        )
        points.append((name, ai, gflops))

    if args.plot:
        plot(
            args.plot,
            points,
            peak_gflops,
            peak_gbs,
            f"{peaks['name']}: {args.compute} / {args.memory} ({kind})",
        )


def plot(path: str, points, peak_gflops: float, peak_gbs: float, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ridge = peak_gflops / peak_gbs
    finite = [ai for _, ai, _ in points if 0 < ai < math.inf]
    lo = min([ridge / 100, *[ai / 2 for ai in finite]])
    hi = max([ridge * 100, *[ai * 2 for ai in finite]])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(
        [lo, ridge, hi],
        [lo * peak_gbs, peak_gflops, peak_gflops],
        color="black",
        lw=2,
        label="roof",
    )
    ax.axvline(ridge, color="gray", ls=":", lw=1)
    for name, ai, gflops in points:
        if 0 < ai < math.inf and gflops > 0:
            ax.scatter([ai], [gflops], zorder=3)
            ax.annotate(name, (ai, gflops), textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
