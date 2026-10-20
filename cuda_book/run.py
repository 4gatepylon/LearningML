#!/usr/bin/env python3
"""Build and run the standalone C++ programs in this folder.

Usage:
  ./run.py                    # build and run every *.cc file
  ./run.py vector_add         # build and run vector_add.cc
  ./run.py vector_add.cc      # same; the .cc extension is optional
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / ".artifacts"
CXX = os.environ.get("CXX", "g++")
CXXFLAGS = os.environ.get("CXXFLAGS", "-std=c++17 -O2 -Wall -Wextra").split()


def build_and_run(name: str) -> bool:
    src = HERE / f"{name}.cc"
    if not src.is_file():
        print(f"error: {src.name} not found", file=sys.stderr)
        return False

    ARTIFACTS.mkdir(exist_ok=True)
    exe = ARTIFACTS / name
    compile_cmd = [CXX, *CXXFLAGS, str(src), "-o", str(exe)]
    print(" ".join(compile_cmd), flush=True)
    if subprocess.run(compile_cmd).returncode != 0:
        return False

    print(f"./{exe.relative_to(HERE)}", flush=True)
    return subprocess.run([str(exe)]).returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("names", nargs="*", help="programs to run (default: all)")
    args = parser.parse_args()

    names = [n.removesuffix(".cc") for n in args.names]
    if not names:
        names = sorted(p.stem for p in HERE.glob("*.cc"))

    failed = []
    for name in names:
        if len(names) > 1:
            print(f"\n==> {name}", flush=True)
        if not build_and_run(name):
            failed.append(name)

    if len(names) > 1:
        print(f"\n{len(names) - len(failed)}/{len(names)} passed")
    if failed:
        print(f"failed: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
