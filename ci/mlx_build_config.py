#!/usr/bin/env python3
"""Select build settings that must match the installed MLX wheel."""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import platform
import re
import sys
import sysconfig


_NANOBIND_BY_MLX = (
    ((0, 31, 0), "2.12.0"),
    ((0, 30, 0), "2.10.2"),
    ((0, 0, 0), "2.4.0"),
)


def _parse_version(value: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", value)
    if match is None:
        raise SystemExit(f"Cannot parse MLX version {value!r}")
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch or 0)


def _installed_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _nanobind_version_for_mlx(mlx_version: str) -> str:
    parsed = _parse_version(mlx_version)
    for minimum, nanobind_version in _NANOBIND_BY_MLX:
        if parsed >= minimum:
            return nanobind_version
    raise AssertionError("unreachable")


def _is_free_threaded() -> bool:
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def _mlx_requirement_for_this_python() -> str:
    if sys.implementation.name != "cpython":
        raise SystemExit("MLX publishes CPython wheels only; PyPy is unsupported.")
    if _is_free_threaded():
        raise SystemExit("MLX does not publish free-threaded CPython wheels.")

    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin":
        if machine not in {"arm64", "aarch64"}:
            raise SystemExit("MLX macOS wheels are arm64 only.")
        return "mlx"
    if system == "Linux":
        return "mlx[cpu]"
    raise SystemExit(f"MLX wheels are unavailable on {system}.")


def _macos_target_from_wheel() -> str | None:
    if platform.system() != "Darwin":
        return None

    for package in ("mlx-metal", "mlx"):
        try:
            wheel = metadata.distribution(package).read_text("WHEEL") or ""
        except metadata.PackageNotFoundError:
            continue
        for line in wheel.splitlines():
            if not line.startswith("Tag: "):
                continue
            match = re.search(r"macosx_(\d+)_(\d+)_", line)
            if match is not None:
                return f"{int(match.group(1))}.{int(match.group(2))}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlx-requirement", action="store_true")
    parser.add_argument("--nanobind-requirement", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--github-env")
    args = parser.parse_args()

    if args.mlx_requirement:
        print(_mlx_requirement_for_this_python())
        return

    mlx_version = _installed_version("mlx")
    if mlx_version is None:
        raise SystemExit("Install MLX before selecting the nanobind ABI.")
    nanobind_requirement = f"nanobind=={_nanobind_version_for_mlx(mlx_version)}"

    if args.nanobind_requirement:
        print(nanobind_requirement)
        return

    macos_target = _macos_target_from_wheel()
    if args.github_env:
        with open(args.github_env, "a", encoding="utf-8") as env_file:
            env_file.write(f"MLXPORTS_MLX_VERSION={mlx_version}\n")
            env_file.write(f"MLXPORTS_NANOBIND_REQUIREMENT={nanobind_requirement}\n")
            if macos_target is not None:
                env_file.write(f"MACOSX_DEPLOYMENT_TARGET={macos_target}\n")
        return

    if args.summary:
        print(f"MLX version: {mlx_version}")
        print(f"nanobind requirement: {nanobind_requirement}")
        if macos_target is not None:
            print(f"macOS deployment target: {macos_target}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
