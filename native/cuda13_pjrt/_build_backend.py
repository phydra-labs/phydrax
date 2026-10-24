"""Wheel-only packaging of the source recipe's verified Windows staging tree.

No compilation, dependency installation, or downloads occur in this backend.
The recipe supplies ``config_settings={"staging": directory,
"kernel-wheel": original_built_wheel}``; upstream version and kernel bytes are
authenticated against that wheel, while owned runtime files match the checkout.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
import platform
import re
import sys
import sysconfig
import tomllib
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn


ROOT = Path(__file__).resolve().parent
PACKAGE = "phydrax_cuda13_pjrt"
MANIFEST = f"{PACKAGE}/manifest.json"
TAG = "cp313-cp313-win_amd64"
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
type ConfigSettings = Mapping[str, str | list[str]] | None


class StagingError(RuntimeError):
    """Staging is incomplete, untrusted, or incompatible with this wheel."""


def _json(data: bytes) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise StagingError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(data, object_pairs_hook=unique)
    except (ValueError, UnicodeError) as error:
        raise StagingError("Invalid JSON manifest") from error
    if not isinstance(value, dict):
        raise StagingError("Manifest must be an object")
    return value


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _relative(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise StagingError("Asset path must be nonempty text")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in (".", "..") or part.endswith((".", " ")) for part in path.parts)
        or re.search(r'[\\:\x00-\x1f,<>"|?*]', value)
    ):
        raise StagingError(f"Unsafe asset path: {value!r}")
    return value


def _read(root: Path, relative: str) -> bytes:
    path = root / _relative(relative)
    for ancestor in (path, *path.parents):
        if ancestor == root:
            break
        if ancestor.is_symlink() or ancestor.is_junction():
            raise StagingError(f"Linked staging asset: {relative}")
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise StagingError(f"Missing or escaping staging asset: {relative}")
    return path.read_bytes()


def _project() -> dict[str, Any]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"]
    core = tomllib.loads((ROOT.parents[1] / "pyproject.toml").read_text("utf-8"))[
        "project"
    ]
    if (
        project["name"] != "phydrax-cuda13-pjrt"
        or project["version"] != core["version"]
        or project["requires-python"] != ">=3.13,<3.14"
        or project["entry-points"] != {"jax_plugins": {"cuda": f"{PACKAGE}.plugin"}}
    ):
        raise StagingError(
            "Project identity, Python requirement, or plugin ownership mismatch"
        )
    dependencies = project["dependencies"]
    if not {"jax==0.12.0", "jaxlib==0.12.0"}.issubset(dependencies):
        raise StagingError("Stock JAX/JAXlib 0.12.0 are required")
    return project


def _lock() -> tuple[dict[str, Any], bytes]:
    data = (ROOT / "sources.lock.json").read_bytes()
    lock = _json(data)
    revisions = {source["name"]: source.get("revision") for source in lock["sources"]}
    if (
        revisions.get("jax") != "8dcb7aeced770bd22531b42d0b925c7342787c27"
        or revisions.get("xla") != "d867969800d4a8fd045505eabc65e1e66f30794f"
        or lock["toolchain"].get("bazel_version") != "9.2.0"
    ):
        raise StagingError(
            "Source lock does not identify the matched JAX/XLA/Bazel stack"
        )
    for relative, digest in lock["recipe_inputs"].items():
        if _digest(_read(ROOT.parents[1], relative)) != digest:
            raise StagingError(f"Locked recipe input digest mismatch: {relative}")
    return lock, data


def _metadata(project: dict[str, Any], licenses: list[str]) -> bytes:
    lines = [
        "Metadata-Version: 2.4",
        f"Name: {project['name']}",
        f"Version: {project['version']}",
        f"Summary: {project['description']}",
        f"Requires-Python: {project['requires-python']}",
        f"License: {project['license']['text']}",
        "Classifier: Operating System :: Microsoft :: Windows",
        "Classifier: Programming Language :: Python :: 3.13",
        *(f"Requires-Dist: {item}" for item in project["dependencies"]),
        *(f"License-File: {item.removeprefix('licenses/')}" for item in licenses),
    ]
    return ("\n".join(lines) + "\n").encode()


def prepare_metadata_for_build_wheel(
    metadata_directory: str, config_settings: ConfigSettings = None
) -> str:
    """Expose dependency metadata without initiating a source build."""
    del config_settings
    project = _project()
    lock, _ = _lock()
    name = f"phydrax_cuda13_pjrt-{project['version']}.dist-info"
    target = Path(metadata_directory) / name
    target.mkdir(parents=True, exist_ok=True)
    (target / "METADATA").write_bytes(_metadata(project, sorted(lock["licenses"])))
    return name


def _host() -> None:
    if (
        sys.platform != "win32"
        or platform.machine().lower() not in ("amd64", "x86_64")
        or sys.implementation.name != "cpython"
        or sys.version_info[:2] != (3, 13)
        or sysconfig.get_config_var("Py_GIL_DISABLED")
    ):
        raise StagingError("Packaging requires standard CPython 3.13 on Windows x64")


def _identity(
    record: dict[str, Any],
    project: dict[str, Any],
    lock: dict[str, Any],
    lock_bytes: bytes,
) -> None:
    identity = {key: value for key, value in record.items() if key != "build_id"}
    if _digest(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ) != record.get("build_id"):
        raise StagingError("Manifest build_id mismatch")
    expected = {
        "distribution": project["name"],
        "version": project["version"],
        "cuda_major": 13,
        "jax_version": "0.12.0",
        "jaxlib_version": "0.12.0",
        "runtime_requirements": project["dependencies"],
        "python_tag": "cp313",
        "abi_tag": "cp313",
        "platform_tag": "win_amd64",
        "sources_lock_sha256": _digest(lock_bytes),
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise StagingError(f"Manifest {key} mismatch")
    for key in ("sources", "patches", "toolchain"):
        if record.get(key) != lock[key]:
            raise StagingError(f"Manifest {key} differs from source lock")
    for patch in lock["patches"]:
        if _digest(_read(ROOT, patch["path"])) != patch["sha256"]:
            raise StagingError("Locked patch digest mismatch")
    capabilities = record.get("capabilities")
    if (
        not isinstance(capabilities, list)
        or any(not isinstance(item, str) for item in capabilities)
        or capabilities != sorted(set(capabilities))
        or set(capabilities) & {"pallas_triton", "mosaic_gpu", "distributed"}
    ):
        raise StagingError("Invalid single-device compiler capabilities")
    if "sparse_qr" in capabilities and (
        "sparse_qr" not in lock["capabilities"]
        or "_solver" not in lock["required_kernels"]
    ):
        raise StagingError(
            "sparse_qr requires the source-locked capability and solver kernel"
        )


def _assets(
    staging: Path, record: dict[str, Any], lock: dict[str, Any]
) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    folded: set[str] = set()
    for group in ("assets", "licenses"):
        entries = record.get(group)
        if not isinstance(entries, list) or not entries:
            raise StagingError(f"Manifest {group} must be a nonempty list")
        for entry in entries:
            if not isinstance(entry, dict):
                raise StagingError(f"Malformed {group} entry")
            name = _relative(entry.get("path"))
            if name.casefold() in folded or name == MANIFEST:
                raise StagingError(f"Duplicate or reserved asset: {name}")
            folded.add(name.casefold())
            owner = PurePosixPath(name).parts[0]
            if owner not in (
                ("licenses",) if group == "licenses" else (PACKAGE, "jax_cuda13_plugin")
            ):
                raise StagingError(f"Foreign module ownership: {name}")
            data = _read(staging, name)
            if (
                type(entry.get("size")) is not int
                or len(data) != entry["size"]
                or _digest(data) != entry.get("sha256")
            ):
                raise StagingError(f"Asset size/hash mismatch: {name}")
            files[name] = data
    license_names = {entry["path"] for entry in record["licenses"]}
    if license_names != set(lock["licenses"]):
        raise StagingError("License inventory differs from source lock")
    for name in license_names:
        if _digest(files[name]) != lock["licenses"][name]:
            raise StagingError(f"License identity mismatch: {name}")
    if (
        files.get("licenses/Phydrax-LICENSE")
        != (ROOT.parents[1] / "LICENSE").read_bytes()
    ):
        raise StagingError("Phydrax license identity mismatch")
    staged: set[str] = set()
    for path in staging.rglob("*"):
        if path.is_symlink() or path.is_junction():
            raise StagingError("Staging may not contain links")
        if path.is_file():
            staged.add(path.relative_to(staging).as_posix())
    if staged != set(files) | {MANIFEST}:
        raise StagingError("Undeclared or missing staged files")
    return files


def _runtime_sources(files: dict[str, bytes], upstream: dict[str, bytes]) -> None:
    expected = {
        path.relative_to(ROOT / "python").as_posix(): path.read_bytes()
        for path in (ROOT / "python").rglob("*")
        if path.is_file() and path.suffix in (".py", ".typed")
    }
    expected[f"{PACKAGE}/_pe_dependencies.py"] = (
        ROOT.parents[1] / "phydrax" / "_pe_dependencies.py"
    ).read_bytes()
    if not expected or any(files.get(name) != data for name, data in expected.items()):
        raise StagingError("Runtime sources or canonical PE parser differ from staging")
    for name in files:
        if name.endswith((".py", ".pyc", ".pth")) and name not in expected:
            if not name.endswith(".py") or files[name] != upstream.get(name):
                raise StagingError(f"Unowned Python module: {name}")


def _kernel_wheel(
    files: dict[str, bytes], record: dict[str, Any], settings: ConfigSettings
) -> dict[str, bytes]:
    value = (settings or {}).get("kernel-wheel")
    identity = record.get("kernel_wheel")
    if not isinstance(value, str) or not isinstance(identity, dict):
        raise StagingError("The original built kernel-wheel is required")
    path = Path(value)
    filename = f"jax_cuda13_plugin-0.12.0-{TAG}.whl"
    if path.name != filename or identity.get("filename") != filename:
        raise StagingError("Kernel wheel filename/version/tag mismatch")
    if not path.is_file() or _digest(path.read_bytes()) != identity.get("sha256"):
        raise StagingError("Kernel wheel digest mismatch")
    try:
        with zipfile.ZipFile(path) as wheel:
            names = wheel.namelist()
            if len(names) != len(set(name.casefold() for name in names)):
                raise StagingError("Duplicate kernel wheel members")
            for name in names:
                _relative(name.rstrip("/"))
            info = "jax_cuda13_plugin-0.12.0.dist-info"
            metadata = wheel.read(f"{info}/METADATA").decode("utf-8").splitlines()
            tags = wheel.read(f"{info}/WHEEL").decode("utf-8").splitlines()
            if (
                "Version: 0.12.0" not in metadata
                or f"Tag: {TAG}" not in tags
                or not any(
                    line in ("Name: jax-cuda13-plugin", "Name: jax_cuda13_plugin")
                    for line in metadata
                )
            ):
                raise StagingError("Kernel wheel metadata identity mismatch")
            upstream = {
                name: wheel.read(name)
                for name in names
                if name.startswith("jax_cuda13_plugin/")
                and name.endswith((".py", ".pyd"))
                and name != "jax_cuda13_plugin/__init__.py"
            }
    except (zipfile.BadZipFile, KeyError, UnicodeError) as error:
        raise StagingError("Malformed kernel wheel") from error
    if "jax_cuda13_plugin/version.py" not in upstream:
        raise StagingError("Kernel wheel is missing upstream version.py")
    for name, data in upstream.items():
        if files.get(name) != data:
            raise StagingError(f"Kernel wheel member differs from staging: {name}")
    for name in files:
        if name.endswith(".pyd") and files[name] != upstream.get(name):
            raise StagingError(f"Kernel extension has no wheel provenance: {name}")
    return upstream


def _native(staging: Path, record: dict[str, Any], lock: dict[str, Any]) -> None:
    spec = importlib.util.spec_from_file_location(
        "_phydrax_packaging_pe", ROOT.parents[1] / "phydrax" / "_pe_dependencies.py"
    )
    if spec is None or spec.loader is None:
        raise StagingError("Cannot load canonical PE parser")
    parser = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = parser
    spec.loader.exec_module(parser)
    pjrt = 0
    kernels: set[str] = set()
    for asset in record["assets"]:
        name = asset["path"]
        role = asset.get("role")
        if role not in ("pjrt", "kernel", "runtime", "parser", "tool"):
            raise StagingError(f"Unknown asset role: {name}")
        if role in ("pjrt", "kernel") or name.lower().endswith((".dll", ".pyd", ".exe")):
            try:
                image = parser.inspect_pe(staging / name)
            except ValueError as error:
                raise StagingError(f"Malformed PE asset: {name}") from error
            if image.machine != 0x8664:
                raise StagingError(f"PE asset is not AMD64: {name}")
            exports = {
                item.name for item in image.exports if item.rva and item.forwarder is None
            }
            if role == "pjrt":
                pjrt += 1
                if (
                    not name.startswith(f"{PACKAGE}/")
                    or not name.endswith(".dll")
                    or "GetPjrtApi" not in exports
                ):
                    raise StagingError("PJRT DLL must export GetPjrtApi")
            elif role == "kernel":
                module = PurePosixPath(name).name.split(".")[0]
                if module in kernels:
                    raise StagingError(f"Conflicting kernel module ownership: {module}")
                kernels.add(module)
                if (
                    not name.startswith("jax_cuda13_plugin/")
                    or not name.endswith(".pyd")
                    or f"PyInit_{module}" not in exports
                ):
                    raise StagingError(f"Kernel Python export mismatch: {name}")
            elif name.endswith(".pyd"):
                raise StagingError("Python extensions must have kernel ownership")
    if pjrt != 1 or not kernels or kernels != set(lock["required_kernels"]):
        raise StagingError(
            "Staging requires one PJRT DLL and the complete locked CUDA kernel set"
        )


def build_wheel(
    wheel_directory: str,
    config_settings: ConfigSettings = None,
    metadata_directory: str | None = None,
) -> str:
    """Package only a complete authenticated staging tree."""
    value = (config_settings or {}).get("staging")
    if not isinstance(value, str) or not value:
        raise StagingError(
            "Source builds are refused; config setting 'staging' is required"
        )
    _host()
    staging = Path(value).resolve()
    project = _project()
    lock, lock_bytes = _lock()
    record = _json(_read(staging, MANIFEST))
    _identity(record, project, lock, lock_bytes)
    files = _assets(staging, record, lock)
    upstream = _kernel_wheel(files, record, config_settings)
    _runtime_sources(files, upstream)
    _native(staging, record, lock)
    info = f"phydrax_cuda13_pjrt-{project['version']}.dist-info"
    metadata = _metadata(project, sorted(lock["licenses"]))
    if (
        metadata_directory is not None
        and (Path(metadata_directory) / "METADATA").read_bytes() != metadata
    ):
        raise StagingError("Prepared metadata changed before packaging")
    for name in tuple(files):
        if name.startswith("licenses/"):
            files[f"{info}/{name}"] = files.pop(name)
    files[MANIFEST] = _read(staging, MANIFEST)
    files[f"{info}/METADATA"] = metadata
    files[f"{info}/WHEEL"] = (
        f"Wheel-Version: 1.0\nGenerator: phydrax-cuda13-pjrt\nRoot-Is-Purelib: false\nTag: {TAG}\n".encode()
    )
    files[f"{info}/entry_points.txt"] = (
        f"[jax_plugins]\ncuda = {PACKAGE}.plugin\n".encode()
    )
    records = io.StringIO(newline="")
    writer = csv.writer(records, lineterminator="\n")
    for name, data in sorted(files.items()):
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(data).digest())
            .rstrip(b"=")
            .decode("ascii")
        )
        writer.writerow((name, f"sha256={digest}", len(data)))
    writer.writerow((f"{info}/RECORD", "", ""))
    files[f"{info}/RECORD"] = records.getvalue().encode()
    destination = Path(wheel_directory)
    destination.mkdir(parents=True, exist_ok=True)
    filename = f"phydrax_cuda13_pjrt-{project['version']}-{TAG}.whl"
    with zipfile.ZipFile(
        destination / filename, "w", compression=zipfile.ZIP_DEFLATED
    ) as bundle:
        for name, data in sorted(files.items()):
            member = zipfile.ZipInfo(name, ZIP_EPOCH)
            member.compress_type = zipfile.ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o100644 << 16
            bundle.writestr(member, data)
    return filename


def get_requires_for_build_wheel(config_settings: ConfigSettings = None) -> list[str]:
    del config_settings
    return []


def build_sdist(sdist_directory: str, config_settings: ConfigSettings = None) -> NoReturn:
    raise StagingError(
        "Source distributions are refused; install a published Windows wheel"
    )


def get_requires_for_build_sdist(config_settings: ConfigSettings = None) -> NoReturn:
    raise StagingError("Source distributions are refused")


def build_editable(
    wheel_directory: str,
    config_settings: ConfigSettings = None,
    metadata_directory: str | None = None,
) -> NoReturn:
    raise StagingError("Editable installs are refused; install a published Windows wheel")


def get_requires_for_build_editable(config_settings: ConfigSettings = None) -> NoReturn:
    raise StagingError("Editable installs are refused")


def prepare_metadata_for_build_editable(
    metadata_directory: str, config_settings: ConfigSettings = None
) -> NoReturn:
    raise StagingError("Editable installs are refused")
