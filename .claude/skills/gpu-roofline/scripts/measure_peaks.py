#!/usr/bin/env python3
"""Builds and runs the GPU microbenchmarks and caches the measured peaks as JSON.

Uses nvcc from PATH and the GPU selected by CUDA_VISIBLE_DEVICES (device 0 of what is visible).
The cache goes to ../.cache/peaks-<host>-<gpu>.json, which roofline.py reads by default.

Usage: measure_peaks.py [--only specs|bandwidth|flops ...] [--dram-mib N] [--gemm-n N]
"""

import argparse
import json
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
SKILL = SCRIPTS.parent
BUILD = SKILL / ".build"
CACHE = SKILL / ".cache"

BENCHES = {
    "specs": ("gpu_specs.cu", []),
    "bandwidth": ("bench_bandwidth.cu", []),
    "flops": ("bench_flops.cu", ["-lcublas"]),
}


def build(name: str) -> Path:
    src, libs = BENCHES[name]
    out = BUILD / Path(src).stem
    if out.exists() and out.stat().st_mtime >= (SCRIPTS / src).stat().st_mtime:
        return out
    BUILD.mkdir(exist_ok=True)
    cmd = ["nvcc", "-O3", "-std=c++17", "-arch=native", str(SCRIPTS / src), "-o", str(out), *libs]
    print("+", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)
    return out


def run(binary: Path, args: list[str]) -> dict:
    proc = subprocess.run([str(binary), *args], check=True, capture_output=True, text=True)
    results = {}
    for line in proc.stdout.splitlines():
        m = re.match(r"RESULT (\S+) (.+)", line)
        if m:
            key, value = m.groups()
            try:
                results[key] = float(value)
            except ValueError:
                results[key] = value
        else:
            print(line)
    return results


def cache_path(gpu_name: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", gpu_name.lower()).strip("-")
    return CACHE / f"peaks-{socket.gethostname()}-{slug}.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--only", nargs="+", choices=list(BENCHES), help="run only these benchmarks"
    )
    parser.add_argument("--dram-mib", default="1024", help="DRAM buffer size for bench_bandwidth")
    parser.add_argument("--gemm-n", default="8192", help="GEMM size for bench_flops")
    args = parser.parse_args()

    if shutil.which("nvcc") is None:
        sys.exit("nvcc not found on PATH; source the CUDA env first (e.g. optimizing_matmul/.env).")

    selected = args.only or list(BENCHES)
    bench_args = {"specs": [], "bandwidth": [args.dram_mib], "flops": [args.gemm_n]}
    # Always run specs: the cache file is named after the GPU.
    results = run(build("specs"), [])
    for name in selected:
        if name != "specs":
            results.update(run(build(name), bench_args[name]))

    path = cache_path(str(results["name"]))
    cached = json.loads(path.read_text()) if path.exists() else {}
    cached.update(results)
    cached["host"] = socket.gethostname()
    cached["measured_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    CACHE.mkdir(exist_ok=True)
    path.write_text(json.dumps(cached, indent=2) + "\n")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
