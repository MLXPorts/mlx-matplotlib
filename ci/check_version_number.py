#!/usr/bin/env python3

"""
Check that Matplotlib's version is not the explicit setuptools_scm fallback.

This fork often builds from an sdist without git tags (and sometimes without a
.git directory), so we only fail if we fell all the way back to the explicit
fallback version.

Usage:
  1) Check an installed distribution:
     $ python ci/check_version_number.py

  2) Check a built wheel without installing it:
     $ python ci/check_version_number.py dist/matplotlib-*.whl
"""

from __future__ import annotations

import sys
import zipfile

from importlib.metadata import PackageNotFoundError, version as dist_version


def _fail_if_unknown(ver: str) -> None:
    # Keep this intentionally narrow: we only want to catch the explicit
    # setuptools_scm fallback, not normal 0.x development versions.
    if ver.startswith("0.0+UNKNOWN"):
        raise SystemExit("Version is unknown (setuptools_scm fallback)")


def _wheel_version(wheel_path: str) -> str:
    with zipfile.ZipFile(wheel_path) as zf:
        meta_paths = [p for p in zf.namelist() if p.endswith(".dist-info/METADATA")]
        if not meta_paths:
            raise SystemExit(f"Could not find METADATA in wheel: {wheel_path}")
        # There should be exactly one; pick the first if multiple exist.
        meta = zf.read(meta_paths[0]).decode("utf-8", errors="replace")

    for line in meta.splitlines():
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip()

    raise SystemExit(f"Could not find Version field in wheel METADATA: {wheel_path}")


def main(argv: list[str]) -> int:
    if len(argv) == 2:
        ver = _wheel_version(argv[1])
        print(f"Wheel version {ver} in {argv[1]}")
        _fail_if_unknown(ver)
        return 0

    if len(argv) != 1:
        raise SystemExit("Usage: check_version_number.py [path/to/wheel.whl]")

    try:
        ver = dist_version("matplotlib")
    except PackageNotFoundError as e:
        raise SystemExit(
            "matplotlib is not installed (pass a wheel path instead)"
        ) from e

    print(f"Installed version {ver}")
    _fail_if_unknown(ver)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
