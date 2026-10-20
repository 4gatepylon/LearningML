#!/usr/bin/env python3
"""Build and test the standalone C++ programs in this folder.

Usage:
  ./test.py                    # build and test every *.cc file
  ./test.py vector_add         # build and test vector_add.cc
  ./test.py vector_add.cc      # same; the .cc extension is optional
  ./test.py vec                # same, if vector_add.cc is the only file starting with "vec"
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


def resolve(arg: str) -> str | None:
    """Maps an argument to a program name: exact match, else a unique filename prefix match."""
    files = sorted(p.name for p in HERE.glob("*.cc"))
    for exact in (arg, f"{arg}.cc"):
        if exact in files:
            return exact.removesuffix(".cc")
    matches = [f for f in files if f.startswith(arg)]
    if len(matches) == 1:
        return matches[0].removesuffix(".cc")
    if matches:
        print(
            f"error: '{arg}' matches multiple files ({', '.join(matches)}); type more of the name",
            file=sys.stderr,
        )
    else:
        print(f"error: no .cc file matches '{arg}'", file=sys.stderr)
    return None


def build_and_test(name: str) -> bool:
    src = HERE / f"{name}.cc"

    ARTIFACTS.mkdir(exist_ok=True)
    exe = ARTIFACTS / name
    compile_cmd = [CXX, *CXXFLAGS, str(src), "-o", str(exe)]
    print(" ".join(compile_cmd), flush=True)
    if subprocess.run(compile_cmd).returncode != 0:
        return False

    print(f"./{exe.relative_to(HERE)}", flush=True)
    return subprocess.run([str(exe)]).returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("names", nargs="*", help="programs to test (default: all)")
    args = parser.parse_args()

    if args.names:
        names = [resolve(a) for a in args.names]
        if None in names:
            return 1
    else:
        names = sorted(p.stem for p in HERE.glob("*.cc"))

    failed = []
    for name in names:
        if len(names) > 1:
            print(f"\n==> {name}", flush=True)
        if not build_and_test(name):
            failed.append(name)

    if len(names) > 1:
        print(f"\n{len(names) - len(failed)}/{len(names)} passed")
    if failed:
        print(f"failed: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
