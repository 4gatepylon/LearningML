"""Runs the benchmark binary and prints each kernel's time, TFLOPS and share of the roofline.

Run via `make bench ARGS="--n 4096"`; .env sets the roofline (PEAK_TFLOPS, MEM_BANDWIDTH_TBPS).
"""

import argparse
import os
import subprocess
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent / ".build" / "bench"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--n", type=int, default=8192, help="matrix size (n x n)")
parser.add_argument("--warmup-iterations", type=int, default=1, help="untimed calls before timing")
args = parser.parse_args()

# A matmul does 2n^3 FLOPs on 3n^2 bf16 values (2 bytes each): n / 3 FLOPs per byte of traffic.
peak_tflops = float(os.environ["PEAK_TFLOPS"])
roofline = min(peak_tflops, args.n / 3 * float(os.environ["MEM_BANDWIDTH_TBPS"]))

cmd = [BENCH, str(args.n), str(args.warmup_iterations)]
out = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True).stdout
row = "{:<12}{:>6}{:>10}{:>9}{:>10}{:>7}".format
print(row("kernel", "n", "ms", "TFLOPS", "roofline", "%roof"))
for line in out.splitlines():
    name, ms = line.split(",")
    tflops = 2 * args.n**3 / (float(ms) * 1e-3) / 1e12
    print(row(name, args.n, ms, f"{tflops:.1f}", f"{roofline:.1f}", f"{tflops / roofline:.0%}"))
