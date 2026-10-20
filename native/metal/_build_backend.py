#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""PEP 517 backend for phydrax-metal: package a verified native build as a wheel.

The native plugin is built and audited by ``tools/build_metal_wheels.py``, which
calls :func:`build_wheel` with ``config_settings={"staging": DIR}``. ``DIR`` holds
``phydrax_metal/build.json``, its hashed native assets and private indexing helper,
and ``licenses/``. The helper must match its canonical core source byte for byte.
This backend never compiles, downloads or installs anything, and it refuses
sdists, editable installs, and wheels without a verified staging tree.
"""

from __future__ import annotations

import base64
import hashlib
import json
import platform
import re
import sys
import tomllib
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import NoReturn


__all__ = [
    "build_editable",
    "build_sdist",
    "build_wheel",
    "get_requires_for_build_editable",
    "get_requires_for_build_sdist",
    "get_requires_for_build_wheel",
    "prepare_metadata_for_build_wheel",
]

ROOT = Path(__file__).resolve().parent
PACKAGE = "phydrax_metal"
PLATFORM = "mps"
PHYDRAX_LICENSE = ROOT.parents[1] / "LICENSE"
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)

type ConfigSettings = Mapping[str, str | list[str]] | None


class StagingError(RuntimeError):
    """The staging tree is not a complete, self-consistent native build."""


def _project() -> tuple[str, str, str, list[str], str, str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    entry_points = project["entry-points"]["jax_plugins"]
    if entry_points != {PLATFORM: f"{PACKAGE}.plugin"}:
        raise StagingError(
            "pyproject.toml must declare exactly the mps jax_plugins entry point"
        )
    return (
        project["name"],
        project["version"],
        project["description"],
        list(project["dependencies"]),
        project["requires-python"],
        project["license"]["text"],
    )


def _license_names() -> tuple[str, ...]:
    lock = json.loads((ROOT / "sources.lock.json").read_text(encoding="utf-8"))
    names = {"Phydrax-LICENSE"}
    for source in lock["sources"]:
        for relative in source["license_files"]:
            path = PurePosixPath(source["name"], relative)
            if path.is_absolute() or ".." in path.parts:
                raise StagingError("Source licenses must have package-relative paths.")
            names.add(path.as_posix())
    return tuple(sorted(names))


def _metadata_lines(
    project: tuple[str, str, str, list[str], str, str],
    licenses: tuple[str, ...],
) -> list[str]:
    name, version, summary, dependencies, requires_python, license_text = project
    return [
        "Metadata-Version: 2.4",
        f"Name: {name}",
        f"Version: {version}",
        f"Summary: {summary}",
        "Author-email: Phydra Labs <leonard@phydra.ai>",
        f"License: {license_text}",
        "Classifier: Operating System :: MacOS",
        "Classifier: Programming Language :: Python :: 3.13",
        "Classifier: Environment :: GPU",
        f"Requires-Python: {requires_python}",
        *(f"Requires-Dist: {dependency}" for dependency in dependencies),
        *(f"License-File: {license_name}" for license_name in licenses),
    ]


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: ConfigSettings = None,
) -> str:
    """Resolve dependencies without compiling or requiring native staging assets."""
    del config_settings
    project = _project()
    directory = f"{project[0].replace('-', '_')}-{project[1]}.dist-info"
    destination = Path(metadata_directory) / directory
    destination.mkdir(parents=True, exist_ok=True)
    content = "\n".join(_metadata_lines(project, _license_names())) + "\n"
    (destination / "METADATA").write_text(content, encoding="utf-8")
    return directory


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _staging(config_settings: ConfigSettings) -> Path:
    value = (config_settings or {}).get("staging")
    if not isinstance(value, str) or not value:
        raise StagingError(
            "phydrax-metal wheels are built only by tools/build_metal_wheels.py "
            "(config setting 'staging' is required)"
        )
    return Path(value).resolve()


def _verified_manifest(
    staging: Path, version: str, dependencies: list[str]
) -> dict[str, object]:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise StagingError("phydrax-metal wheels are packaged only on macOS arm64 hosts")
    package = staging / PACKAGE
    record = json.loads((package / "build.json").read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise StagingError("build.json is not an object")
    identity = {key: value for key, value in record.items() if key != "build_id"}
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != record.get("build_id"):
        raise StagingError("build.json does not match its build_id")
    if record.get("platform") != PLATFORM or record.get("version") != version:
        raise StagingError("build.json platform/version disagree with pyproject.toml")
    expected = {
        f"jax=={record.get('jax_version')}",
        f"jaxlib=={record.get('jaxlib_version')}",
    }
    if set(dependencies) != expected:
        raise StagingError(
            f"pyproject.toml dependencies {dependencies} are not {sorted(expected)}"
        )
    assets = record.get("assets")
    if (
        not isinstance(assets, dict)
        or record.get("library") not in assets
        or "_jax_indexing.py" not in assets
    ):
        raise StagingError(
            "build.json does not identify the library and private indexing helper "
            "among its assets"
        )
    staged = {
        path.relative_to(package).as_posix()
        for path in [*(package / "lib").rglob("*"), package / "_jax_indexing.py"]
        if path.is_file()
    }
    if staged != set(assets):
        raise StagingError(f"staged files {sorted(staged)} differ from build.json assets")
    for relative, digest in assets.items():
        if _sha256(package / relative) != digest:
            raise StagingError(f"{relative} does not match build.json")
    if _sha256(package / "_jax_indexing.py") != _sha256(
        ROOT.parents[1] / "phydrax" / "_jax_indexing.py"
    ):
        raise StagingError("Staged indexing helper differs from the canonical source.")
    return record


def _platform_tag(minimum_macos: object) -> str:
    if not isinstance(minimum_macos, str) or not re.fullmatch(r"\d+\.\d+", minimum_macos):
        raise StagingError(
            f"build.json minimum_macos {minimum_macos!r} is not MAJOR.MINOR"
        )
    major, minor = minimum_macos.split(".")
    return f"py3-none-macosx_{major}_{minor}_arm64"


def _record_hash(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return "sha256=" + encoded.decode("ascii")


def _write(
    bundle: zipfile.ZipFile, records: list[str], name: str, data: bytes, mode: int
) -> None:
    info = zipfile.ZipInfo(name, ZIP_EPOCH)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (0o100000 | mode) << 16
    bundle.writestr(info, data)
    records.append(f"{name},{_record_hash(data)},{len(data)}")


def build_wheel(
    wheel_directory: str,
    config_settings: ConfigSettings = None,
    metadata_directory: str | None = None,
) -> str:
    """Package the verified staging tree named by ``config_settings['staging']``."""
    project = _project()
    name, version, _, dependencies, _, _ = project
    staging = _staging(config_settings)
    record = _verified_manifest(staging, version, dependencies)
    tag = _platform_tag(record.get("minimum_macos"))
    distribution = name.replace("-", "_")
    dist_info = f"{distribution}-{version}.dist-info"

    files: dict[str, tuple[bytes, int]] = {}
    for source in sorted((ROOT / "python" / PACKAGE).iterdir()):
        if source.suffix in (".py", ".typed") and source.is_file():
            files[f"{PACKAGE}/{source.name}"] = (source.read_bytes(), 0o644)
    files[f"{PACKAGE}/build.json"] = (
        (staging / PACKAGE / "build.json").read_bytes(),
        0o644,
    )
    assets = record["assets"]
    if not isinstance(assets, dict):
        raise StagingError("Verified build manifest has no native asset mapping.")
    for relative in sorted(assets):
        mode = 0o755 if relative.endswith(".dylib") else 0o644
        files[f"{PACKAGE}/{relative}"] = (
            (staging / PACKAGE / relative).read_bytes(),
            mode,
        )

    licenses: dict[str, bytes] = {"Phydrax-LICENSE": PHYDRAX_LICENSE.read_bytes()}
    for path in sorted((staging / "licenses").rglob("*")):
        if path.is_file():
            relative = PurePosixPath(*path.relative_to(staging / "licenses").parts)
            licenses[str(relative)] = path.read_bytes()

    declared_licenses = _license_names()
    if set(licenses) != set(declared_licenses):
        raise StagingError("Staged licenses do not match the locked source closure.")
    metadata_lines = _metadata_lines(project, declared_licenses)
    if metadata_directory is not None:
        prepared = Path(metadata_directory) / "METADATA"
        if prepared.read_text(encoding="utf-8") != "\n".join(metadata_lines) + "\n":
            raise StagingError("Prepared metadata changed before wheel packaging.")
    wheel_lines = [
        "Wheel-Version: 1.0",
        "Generator: phydrax-metal _build_backend",
        "Root-Is-Purelib: false",
        f"Tag: {tag}",
    ]
    entry_points = f"[jax_plugins]\n{PLATFORM} = {PACKAGE}.plugin\n"

    filename = f"{distribution}-{version}-{tag}.whl"
    target = Path(wheel_directory) / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    records: list[str] = []
    with zipfile.ZipFile(target, "w") as bundle:
        for archive_name, (data, mode) in sorted(files.items()):
            _write(bundle, records, archive_name, data, mode)
        for license_name, data in sorted(licenses.items()):
            _write(bundle, records, f"{dist_info}/licenses/{license_name}", data, 0o644)
        _write(
            bundle,
            records,
            f"{dist_info}/METADATA",
            ("\n".join(metadata_lines) + "\n").encode("utf-8"),
            0o644,
        )
        _write(
            bundle,
            records,
            f"{dist_info}/WHEEL",
            ("\n".join(wheel_lines) + "\n").encode("utf-8"),
            0o644,
        )
        _write(
            bundle,
            records,
            f"{dist_info}/entry_points.txt",
            entry_points.encode("utf-8"),
            0o644,
        )
        records.append(f"{dist_info}/RECORD,,")
        info = zipfile.ZipInfo(f"{dist_info}/RECORD", ZIP_EPOCH)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = (0o100000 | 0o644) << 16
        bundle.writestr(info, "\n".join(records) + "\n")
    return filename


def get_requires_for_build_wheel(config_settings: ConfigSettings = None) -> list[str]:
    """The backend needs no build dependencies."""
    del config_settings
    return []


def build_sdist(sdist_directory: str, config_settings: ConfigSettings = None) -> NoReturn:
    """Refuse source distributions: phydrax-metal is distributed as wheels only."""
    raise RuntimeError(
        "phydrax-metal does not support sdists; build with tools/build_metal_wheels.py."
    )


def get_requires_for_build_sdist(config_settings: ConfigSettings = None) -> NoReturn:
    """Refuse sdist requests before a frontend installs further build dependencies."""
    raise RuntimeError(
        "phydrax-metal does not support sdists; build with tools/build_metal_wheels.py."
    )


def build_editable(
    wheel_directory: str,
    config_settings: ConfigSettings = None,
    metadata_directory: str | None = None,
) -> NoReturn:
    """Refuse editable installs: only a verified native build may be installed."""
    raise RuntimeError("phydrax-metal does not support editable installs.")


def get_requires_for_build_editable(config_settings: ConfigSettings = None) -> NoReturn:
    """Refuse editable installs before a frontend installs build dependencies."""
    raise RuntimeError("phydrax-metal does not support editable installs.")
