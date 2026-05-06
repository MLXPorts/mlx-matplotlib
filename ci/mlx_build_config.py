#!/usr/bin/env python3
"""Select build settings that must match the installed MLX wheel."""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import platform
import re
import shlex
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


def _wheel_tags(package: str) -> list[str]:
    try:
        wheel = metadata.distribution(package).read_text("WHEEL") or ""
    except metadata.PackageNotFoundError:
        return []
    return [
        line.removeprefix("Tag: ").strip()
        for line in wheel.splitlines()
        if line.startswith("Tag: ")
    ]


def _mlx_wheel_tags() -> list[str]:
    tags = []
    for package in ("mlx", "mlx-metal"):
        tags.extend(_wheel_tags(package))
    return tags


def _wheel_platform_tags() -> list[str]:
    platforms = []
    for tag in _mlx_wheel_tags():
        parts = tag.split("-")
        if len(parts) >= 3 and parts[-1] not in platforms:
            platforms.append(parts[-1])
    return platforms


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

    targets = []
    for tag in _wheel_platform_tags():
        match = re.search(r"macosx_(\d+)_(\d+)_", tag)
        if match is not None:
            targets.append((int(match.group(1)), int(match.group(2))))
    if not targets:
        return None
    major, minor = max(targets)
    return f"{major}.{minor}"


def _manylinux_image_from_wheel() -> str | None:
    if platform.system() != "Linux":
        return None

    images = []
    for tag in _wheel_platform_tags():
        match = re.search(r"(manylinux_\d+_\d+)_", tag)
        if match is not None:
            major, minor = match.group(1).removeprefix("manylinux_").split("_")
            images.append((int(major), int(minor), match.group(1)))
    if not images:
        return None
    return max(images)[2]


def _env_exports(mlx_version: str, nanobind_requirement: str) -> dict[str, str]:
    exports = {
        "MLXPORTS_MLX_VERSION": mlx_version,
        "MLXPORTS_NANOBIND_REQUIREMENT": nanobind_requirement,
    }
    wheel_platforms = _wheel_platform_tags()
    if wheel_platforms:
        exports["MLXPORTS_MLX_WHEEL_PLATFORMS"] = ",".join(wheel_platforms)
    macos_target = _macos_target_from_wheel()
    if macos_target is not None:
        exports["MACOSX_DEPLOYMENT_TARGET"] = macos_target
    manylinux_image = _manylinux_image_from_wheel()
    if manylinux_image is not None:
        exports["CIBW_MANYLINUX_X86_64_IMAGE"] = manylinux_image
        exports["CIBW_MANYLINUX_AARCH64_IMAGE"] = manylinux_image
    return exports


def _write_github_env(path: str, exports: dict[str, str]) -> None:
    with open(path, "a", encoding="utf-8") as env_file:
        for key, value in exports.items():
            env_file.write(f"{key}={value}\n")


def _print_shell_env(exports: dict[str, str]) -> None:
    for key, value in exports.items():
        print(f"export {key}={shlex.quote(value)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlx-requirement", action="store_true")
    parser.add_argument("--nanobind-requirement", action="store_true")
    parser.add_argument("--shell-env", action="store_true")
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

    exports = _env_exports(mlx_version, nanobind_requirement)

    if args.shell_env:
        _print_shell_env(exports)
        return

    if args.github_env:
        _write_github_env(args.github_env, exports)
        return

    if args.summary:
        print(f"MLX version: {mlx_version}")
        print(f"nanobind requirement: {nanobind_requirement}")
        wheel_platforms = exports.get("MLXPORTS_MLX_WHEEL_PLATFORMS")
        if wheel_platforms is not None:
            print(f"MLX wheel platform tags: {wheel_platforms}")
        macos_target = exports.get("MACOSX_DEPLOYMENT_TARGET")
        if macos_target is not None:
            print(f"macOS deployment target: {macos_target}")
        manylinux_image = exports.get("CIBW_MANYLINUX_X86_64_IMAGE")
        if manylinux_image is not None:
            print(f"manylinux image: {manylinux_image}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
