#!/usr/bin/env python3
"""Build and test the standalone C++/CUDA programs in this folder (*.cc via g++, *.cu via nvcc).

Usage:
  ./test.py                    # build and test every *.cc and *.cu file
  ./test.py vector_add         # build and test vector_add.cu (or .cc)
  ./test.py vector_add.cu      # same; the extension is optional
  ./test.py vec                # same, if vector_add.* is the only file starting with "vec"
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / ".artifacts"
CXX = os.environ.get("CXX", "g++")
CXXFLAGS = os.environ.get("CXXFLAGS", "-std=c++17 -O2 -Wall -Wextra").split()
# nvcc comes from PATH (e.g. `source .env`), so no CUDA version or location is hardcoded here.
NVCC = os.environ.get("NVCC", "nvcc")
NVCCFLAGS = os.environ.get(
    "NVCCFLAGS", "-std=c++17 -O2 -Xcompiler=-Wall,-Wextra -Wno-deprecated-gpu-targets"
).split()
# GoogleTest (libgtest-dev); gtest_main supplies main().
LIBS = ["-lgtest", "-lgtest_main", "-lpthread"]
SUFFIXES = (".cc", ".cu")


def sources() -> list[Path]:
    return sorted(p for p in HERE.iterdir() if p.suffix in SUFFIXES)


def resolve(arg: str) -> Path | None:
    """Maps an argument to a source file: exact match, else a unique filename prefix match."""
    files = {p.name: p for p in sources()}
    for exact in (arg, *(arg + s for s in SUFFIXES)):
        if exact in files:
            return files[exact]
    matches = [f for f in files if f.startswith(arg)]
    if len(matches) == 1:
        return files[matches[0]]
    if matches:
        print(
            f"error: '{arg}' matches multiple files ({', '.join(matches)}); type more of the name",
            file=sys.stderr,
        )
    else:
        print(f"error: no .cc or .cu file matches '{arg}'", file=sys.stderr)
    return None


def build_and_test(src: Path) -> bool:
    ARTIFACTS.mkdir(exist_ok=True)
    exe = ARTIFACTS / src.stem
    if src.suffix == ".cu":
        if shutil.which(NVCC) is None:
            print(
                f"error: {NVCC} not found; is CUDA on your PATH (e.g. `source .env`)?",
                file=sys.stderr,
            )
            return False
        compiler = [NVCC, *NVCCFLAGS]
    else:
        compiler = [CXX, *CXXFLAGS]
    compile_cmd = [*compiler, str(src), "-o", str(exe), *LIBS]
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
        names = sources()

    failed = []
    for src in names:
        if len(names) > 1:
            print(f"\n==> {src.name}", flush=True)
        if not build_and_test(src):
            failed.append(src.name)

    if len(names) > 1:
        print(f"\n{len(names) - len(failed)}/{len(names)} passed")
    if failed:
        print(f"failed: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
