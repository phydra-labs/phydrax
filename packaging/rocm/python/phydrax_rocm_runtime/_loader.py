#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Owned ROCm 10 wheel inspection and deferred, absolute-path native loading.

Inspection never imports SDK modules, loads a shared object, or discovers a GPU.
The import finder acts only at the ROCm JAX plugin boundary; JAX's plugin
callbacks occur after native loading and cannot safely prepare its dependencies.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import platform
import re
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, cast


SDK_VERSION = "10.1.0"
ARCHITECTURES = ("gfx1250", "gfx942", "gfx950")
COMPONENTS = {
    "core": ("phydrax-rocm-core", "_rocm_sdk_core"),
    "libraries": ("phydrax-rocm-libraries", "_rocm_sdk_libraries"),
    "device_gfx1250": ("phydrax-rocm-device-gfx1250", "_rocm_sdk_libraries"),
    "device_gfx942": ("phydrax-rocm-device-gfx942", "_rocm_sdk_libraries"),
    "device_gfx950": ("phydrax-rocm-device-gfx950", "_rocm_sdk_libraries"),
}
COMPONENT_PAYLOAD_ROOTS = {
    "core": ("_rocm_sdk_core", "rocm_sdk_core"),
    "libraries": ("_rocm_sdk_libraries", "rocm_sdk_libraries"),
    "device_gfx1250": ("_rocm_sdk_libraries",),
    "device_gfx942": ("_rocm_sdk_libraries",),
    "device_gfx950": ("_rocm_sdk_libraries",),
}
_PAYLOAD_ROOTS = frozenset(
    root for roots in COMPONENT_PAYLOAD_ROOTS.values() for root in roots
)
DISTRIBUTIONS = ("phydrax-rocm-runtime", *(item[0] for item in COMPONENTS.values()))
_PLUGIN = "jax_plugins.xla_rocm10"
_LOCK = threading.RLock()
_HANDLES: list[object] = []
_LOADED: RocmBundle | None = None
_FAILURE: str | None = None
_INITIAL_ROCM_ENVIRONMENT: tuple[tuple[str, str], ...] | None = None
_SYSTEM_LIBRARIES = frozenset(
    {
        "libc.so.6",
        "libm.so.6",
        "libdl.so.2",
        "libpthread.so.0",
        "librt.so.1",
        "libstdc++.so.6",
        "libgcc_s.so.1",
        "ld-linux-x86-64.so.2",
        "libutil.so.1",
        "libresolv.so.2",
    }
)
_FINDER: _RocmFinder | None = None
_ROCM_LIBRARY = re.compile(
    r"lib(?:amdhip|hsa-runtime|amd_comgr|roc(?:blas|solver|fft|sparse|rand|profiler)|"
    r"hip(?:blas|solver|fft|sparse|rand|rtc)|MIOpen|rccl|roctx|roctracer).*\.so(?:\..*)?"
)


class RocmRuntimeError(RuntimeError):
    """The owned userspace cannot be admitted safely in this process."""


@dataclass(frozen=True, slots=True)
class RocmBundle:
    sdk_version: str
    architectures: tuple[str, ...]
    manifest_path: str
    components: tuple[tuple[str, str, str], ...]
    libraries: tuple[tuple[str, str, tuple[str, ...], str], ...]
    preload: tuple[str, ...]
    compiler: tuple[tuple[str, str], ...]
    kpack: tuple[tuple[str, tuple[str, ...]], ...]
    plugin_objects: tuple[tuple[str, str | None, tuple[str, ...], str], ...]
    assets: tuple[tuple[str, str], ...]


def _object(value: object, fields: set[str], name: str) -> dict[str, Any]:
    if (
        type(value) is not dict
        or any(type(key) is not str for key in value)
        or set(value) != fields
    ):
        raise RocmRuntimeError(f"Invalid {name}: expected fields {sorted(fields)}.")
    return cast(dict[str, Any], value)


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value or "\0" in value:
        raise RocmRuntimeError(f"Invalid {name}: expected nonempty text.")
    return value


def _digest(value: object) -> str:
    text = _text(value, "SHA256")
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise RocmRuntimeError("Invalid SHA256 digest.")
    return text


def _list(value: object, name: str) -> list[Any]:
    if type(value) is not list or not value:
        raise RocmRuntimeError(f"Invalid {name}: expected a nonempty list.")
    return value


def _relative(value: object) -> str:
    text = _text(value, "payload path")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or str(path) != text:
        raise RocmRuntimeError(f"Unsafe payload path: {text!r}.")
    if path.parts[0] not in _PAYLOAD_ROOTS:
        raise RocmRuntimeError(f"Unowned payload path: {text!r}.")
    return text


def _asset(root: Path, relative: str, *, directory: bool = False) -> Path:
    path = root / relative
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise RocmRuntimeError(f"Payload escapes its installation: {path}.")
    if not (resolved.is_dir() if directory else resolved.is_file()):
        raise RocmRuntimeError(f"Missing owned ROCm asset: {path}.")
    return resolved


def _record_digest(record: metadata.PackagePath, digest: str, size: int) -> None:
    expected = base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip("=")
    if (
        record.hash is None
        or record.hash.mode != "sha256"
        or record.hash.value != expected
        or record.size != size
    ):
        raise RocmRuntimeError(f"Installed RECORD identity mismatch: {record}.")


def _json(text: str, name: str) -> object:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise RocmRuntimeError(f"Duplicate {name} field: {key}.")
            result[key] = value
        return result

    try:
        return json.loads(text, object_pairs_hook=unique)
    except (ValueError, TypeError) as error:
        raise RocmRuntimeError(f"Invalid {name}: {error}") from error


def _distribution(name: str) -> metadata.Distribution:
    try:
        return metadata.distribution(name)
    except metadata.PackageNotFoundError as error:
        required = SDK_VERSION if name in DISTRIBUTIONS else "0.12.0"
        raise RocmRuntimeError(
            f"Missing {name}=={required}; install phydrax[rocm10]."
        ) from error


def _distribution_path(distribution: metadata.Distribution, relative: str) -> Path:
    location = distribution.locate_file(relative)
    if not isinstance(location, Path):
        raise RocmRuntimeError("Owned ROCm components require a filesystem installation.")
    return location


def _conflicting_distributions() -> None:
    seen = set()
    for distribution in metadata.distributions():
        name = (distribution.metadata.get("Name") or "").lower().replace("_", "-")
        if name in DISTRIBUTIONS or name in {"jax-rocm10-plugin", "jax-rocm10-pjrt"}:
            if name in seen:
                raise RocmRuntimeError(f"Duplicate native distribution: {name}.")
            seen.add(name)
        if (
            name.startswith("rocm-sdk-")
            or name == "rocm"
            or (
                name.startswith("jax-rocm")
                and name not in {"jax-rocm10-plugin", "jax-rocm10-pjrt"}
            )
        ):
            raise RocmRuntimeError(
                f"Conflicting native distribution {name!r}; use a clean phydrax[rocm10] environment."
            )


def inspect_rocm_bundle() -> RocmBundle | None:
    """Inspect metadata, or return the immutable plan bound to this process.

    Before binding, inspection checks inventories and asset presence without
    reading large payload bytes. First activation hashes every payload exactly
    once. After binding this reports that admitted snapshot, not a fresh disk
    integrity check; installation changes require a new Python process.
    """
    with _LOCK:
        if _LOADED is not None:
            return _LOADED
        try:
            runtime = metadata.distribution("phydrax-rocm-runtime")
        except metadata.PackageNotFoundError:
            return None
        try:
            return _inspect(runtime)
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise RocmRuntimeError(f"Invalid owned ROCm bundle: {error}") from error


def _inspect_manifest(
    runtime: metadata.Distribution,
) -> tuple[Path, dict[str, Any]]:
    if runtime.version != SDK_VERSION:
        raise RocmRuntimeError(
            f"Owned ROCm runtime requires {SDK_VERSION}, found {runtime.version}."
        )
    _conflicting_distributions()
    manifest_text = _distribution_path(
        runtime, "phydrax_rocm_runtime/runtime.json"
    ).read_text(encoding="utf-8")
    manifest_record = next(
        (
            item
            for item in runtime.files or ()
            if str(item) == "phydrax_rocm_runtime/runtime.json"
        ),
        None,
    )
    if manifest_record is None:
        raise RocmRuntimeError("Owned runtime manifest is missing from RECORD.")
    manifest_bytes = manifest_text.encode("utf-8")
    _record_digest(
        manifest_record, hashlib.sha256(manifest_bytes).hexdigest(), len(manifest_bytes)
    )
    manifest_path = _distribution_path(runtime, "phydrax_rocm_runtime/runtime.json")
    manifest = _object(
        _json(manifest_text, "runtime manifest"),
        {
            "sdk_version",
            "architectures",
            "components",
            "loader",
            "source_manifest",
            "third_party_identity",
            "packaging_identity",
        },
        "runtime manifest",
    )
    if manifest["sdk_version"] != SDK_VERSION or type(manifest["sdk_version"]) is not str:
        raise RocmRuntimeError("Owned ROCm SDK version mismatch.")
    if manifest["architectures"] != list(ARCHITECTURES):
        raise RocmRuntimeError(
            "Owned ROCm bundle must contain gfx1250, gfx942, and gfx950."
        )
    return manifest_path, manifest


def _component_inventory(
    distribution: metadata.Distribution,
    name: str,
    roots: tuple[str, ...],
    digest: str,
    root: Path,
) -> dict[str, dict[str, Any]]:
    inventory_text = distribution.read_text("phydrax-payload.json")
    if inventory_text is None:
        raise RocmRuntimeError(f"Missing payload inventory: {name}.")
    inventory = _list(_json(inventory_text, "payload inventory"), "payload inventory")
    canonical = json.dumps(inventory, separators=(",", ":"), ensure_ascii=True).encode()
    if hashlib.sha256(canonical).hexdigest() != digest:
        raise RocmRuntimeError(f"Payload inventory identity mismatch: {name}.")
    records = {str(item): item for item in distribution.files or ()}
    entries = {}
    for entry in inventory:
        entry = _object(entry, {"path", "sha256", "size", "mode"}, "payload entry")
        relative = _relative(entry["path"])
        if relative.split("/", 1)[0] not in roots:
            raise RocmRuntimeError(f"Component payload escapes its module: {relative}.")
        _digest(entry["sha256"])
        if (
            type(entry["size"]) is not int
            or entry["size"] < 0
            or type(entry["mode"]) is not int
        ):
            raise RocmRuntimeError("Invalid payload size or mode.")
        if relative in entries or relative not in records:
            raise RocmRuntimeError(
                f"Missing or duplicated payload RECORD entry: {relative}."
            )
        _record_digest(records[relative], entry["sha256"], entry["size"])
        asset = _asset(root, relative)
        if asset.stat().st_size != entry["size"]:
            raise RocmRuntimeError(f"Payload size mismatch: {relative}.")
        entries[relative] = entry
    return entries


def _inspect_components(
    components: dict[str, Any], root: Path
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[tuple[str, str, str]]]:
    inventories: dict[str, dict[str, dict[str, Any]]] = {}
    facts = []
    for key, (name, module) in COMPONENTS.items():
        component = _object(
            components[key],
            {
                "distribution",
                "version",
                "module",
                "upstream",
                "payload_sha256",
                "deduplicated",
            },
            key,
        )
        if (component["distribution"], component["version"], component["module"]) != (
            name,
            SDK_VERSION,
            module,
        ):
            raise RocmRuntimeError(f"Component identity mismatch: {key}.")
        upstream = _object(component["upstream"], {"url", "sha256"}, "upstream")
        if not _text(upstream["url"], "upstream URL").startswith("https://"):
            raise RocmRuntimeError("Upstream provenance requires an HTTPS URL.")
        _digest(upstream["sha256"])
        digest = _digest(component["payload_sha256"])
        distribution = _distribution(name)
        if (
            distribution.version != SDK_VERSION
            or _distribution_path(distribution, "").resolve() != root
        ):
            raise RocmRuntimeError(f"Mismatched component installation: {name}.")
        inventories[key] = _component_inventory(
            distribution, name, COMPONENT_PAYLOAD_ROOTS[key], digest, root
        )
        facts.append((name, SDK_VERSION, digest))
    return inventories, facts


def _merge_inventories(
    inventories: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    all_entries = {}
    for inventory in inventories.values():
        for relative, entry in inventory.items():
            if relative in all_entries:
                raise RocmRuntimeError(
                    f"Duplicate component payload ownership: {relative}."
                )
            all_entries[relative] = entry
    return all_entries


def _validate_provenance(
    manifest: dict[str, Any],
    loader: dict[str, Any],
    owned: Callable[..., Path],
) -> None:
    owned(manifest["source_manifest"])
    if (
        manifest["third_party_identity"] != f"AMD ROCm SDK {SDK_VERSION}"
        or manifest["packaging_identity"] != "Phydrax owned ROCm runtime packaging"
    ):
        raise RocmRuntimeError(
            "Third-party and packaging provenance identities do not match."
        )
    if (
        type(loader["system_libraries"]) is not list
        or set(loader["system_libraries"]) != _SYSTEM_LIBRARIES
    ):
        raise RocmRuntimeError("Invalid host ABI dependency allowlist.")


def _validate_deduplicated(
    components: dict[str, Any],
    inventories: dict[str, dict[str, dict[str, Any]]],
) -> None:
    for key, component in components.items():
        if type(component["deduplicated"]) is not list:
            raise RocmRuntimeError("Deduplicated payload entries must be a list.")
        for entry in component["deduplicated"]:
            entry = _object(
                entry, {"path", "owner", "sha256", "mode"}, "deduplicated entry"
            )
            relative = _relative(entry["path"])
            owner = entry["owner"]
            owner_key = next(
                (
                    name
                    for name, (distribution, _) in COMPONENTS.items()
                    if distribution == owner
                ),
                None,
            )
            if owner_key is None or owner_key == key or relative in inventories[key]:
                raise RocmRuntimeError("Invalid shared payload ownership.")
            original = inventories[owner_key].get(relative)
            if original is None or (original["sha256"], original["mode"]) != (
                entry["sha256"],
                entry["mode"],
            ):
                raise RocmRuntimeError("Deduplicated payload identity mismatch.")


def _inspect_libraries(
    loader: dict[str, Any],
    all_entries: dict[str, dict[str, Any]],
    owned: Callable[..., Path],
) -> tuple[list[tuple[str, str, tuple[str, ...], str]], set[str], dict[str, str]]:
    libraries = []
    sonames = set()
    paths = {}
    for entry in _list(loader["libraries"], "libraries"):
        entry = _object(entry, {"path", "soname", "needed", "sha256"}, "library")
        path = owned(entry["path"])
        soname = _text(entry["soname"], "SONAME")
        if "/" in soname or soname in sonames or str(path) in paths:
            raise RocmRuntimeError(f"Conflicting library identity: {soname}.")
        if type(entry["needed"]) is not list:
            raise RocmRuntimeError("Library dependencies must be a list.")
        needed = tuple(_text(value, "dependency") for value in entry["needed"])
        digest = _digest(entry["sha256"])
        if digest != all_entries[entry["path"]]["sha256"]:
            raise RocmRuntimeError(f"Library inventory digest mismatch: {path}.")
        sonames.add(soname)
        paths[str(path)] = soname
        libraries.append((str(path), soname, needed, digest))
    return libraries, sonames, paths


def _inspect_preload(
    loader: dict[str, Any],
    libraries: list[tuple[str, str, tuple[str, ...], str]],
    sonames: set[str],
    paths: dict[str, str],
    owned: Callable[..., Path],
) -> tuple[str, ...]:
    preload = tuple(str(owned(value)) for value in _list(loader["preload"], "preload"))
    if len(set(preload)) != len(preload) or set(preload) != set(paths):
        raise RocmRuntimeError("Preload must contain every owned library exactly once.")
    seen = set()
    by_path = {entry[0]: entry for entry in libraries}
    for path in preload:
        _, soname, needed, _ = by_path[path]
        if any(dependency in sonames and dependency not in seen for dependency in needed):
            raise RocmRuntimeError(f"Invalid dependency order before {soname}.")
        if any(
            dependency not in sonames and dependency not in _SYSTEM_LIBRARIES
            for dependency in needed
        ):
            raise RocmRuntimeError(f"Dependency escapes owned runtime: {soname}.")
        seen.add(soname)
    return preload


def _inspect_compiler(
    loader: dict[str, Any], owned: Callable[..., Path]
) -> tuple[dict[str, Any], tuple[tuple[str, str], ...]]:
    compiler = _object(
        loader["compiler"],
        {"llvm_path", "lld_path", "rocm_path", "bitcode_path"},
        "compiler",
    )
    compiler_paths = tuple(
        (key, str(owned(value, directory=key != "lld_path")))
        for key, value in compiler.items()
    )
    if not os.access(dict(compiler_paths)["lld_path"], os.X_OK):
        raise RocmRuntimeError("Owned ROCm linker is not executable.")
    return compiler, compiler_paths


def _inspect_kpack(
    loader: dict[str, Any],
    inventories: dict[str, dict[str, dict[str, Any]]],
    owned: Callable[..., Path],
) -> list[tuple[str, tuple[str, ...]]]:
    packs = _object(loader["kpack"], set(ARCHITECTURES), "kpack")
    kpack = []
    for arch in ARCHITECTURES:
        arch_paths = []
        for value in _list(packs[arch], arch):
            if (
                value not in inventories[f"device_{arch}"]
                or arch not in value
                or not value.endswith(".kpack")
            ):
                raise RocmRuntimeError(f"Architecture payload identity mismatch: {arch}.")
            arch_paths.append(str(owned(value)))
        kpack.append((arch, tuple(arch_paths)))
    return kpack


def _inspect_plugin_objects(
    loader: dict[str, Any], root: Path, sonames: set[str]
) -> list[tuple[str, str | None, tuple[str, ...], str]]:
    plugin_objects = []
    for entry in _list(loader["plugin_objects"], "plugin objects"):
        entry = _object(
            entry, {"wheel", "path", "sha256", "soname", "needed"}, "plugin object"
        )
        relative = _text(entry["path"], "plugin path")
        parts = PurePosixPath(relative).parts
        if (
            PurePosixPath(relative).is_absolute()
            or ".." in parts
            or not (
                parts[0] == "jax_rocm10_plugin"
                or parts[:2] == ("jax_plugins", "xla_rocm10")
            )
        ):
            raise RocmRuntimeError(f"Invalid ROCm plugin object: {relative}.")
        name = (
            "jax-rocm10-plugin" if parts[0] == "jax_rocm10_plugin" else "jax-rocm10-pjrt"
        )
        distribution = _distribution(name)
        path = _distribution_path(distribution, relative).resolve()
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or distribution.version != "0.12.0"
        ):
            raise RocmRuntimeError(f"Missing or mismatched plugin object: {relative}.")
        digest = _digest(entry["sha256"])
        record = next(
            (record for record in distribution.files or () if str(record) == relative),
            None,
        )
        if record is None:
            raise RocmRuntimeError(f"Plugin object missing from RECORD: {relative}.")
        _record_digest(record, digest, path.stat().st_size)
        if entry["soname"] is not None:
            _text(entry["soname"], "plugin SONAME")
        if type(entry["needed"]) is not list:
            raise RocmRuntimeError("Plugin dependencies must be a list.")
        needed = tuple(_text(value, "plugin dependency") for value in entry["needed"])
        if any(
            value not in sonames and value not in _SYSTEM_LIBRARIES for value in needed
        ):
            raise RocmRuntimeError("JAX plugin dependencies escape the owned closure.")
        plugin_objects.append((str(path), entry["soname"], needed, digest))
    return plugin_objects


def _inspect(runtime: metadata.Distribution) -> RocmBundle:
    manifest_path, manifest = _inspect_manifest(runtime)
    components = _object(manifest["components"], set(COMPONENTS), "components")
    root = manifest_path.parent.parent.resolve()
    inventories, facts = _inspect_components(components, root)
    loader = _object(
        manifest["loader"],
        {
            "preload",
            "compiler",
            "kpack",
            "libraries",
            "system_libraries",
            "plugin_objects",
        },
        "loader",
    )
    all_entries = _merge_inventories(inventories)

    def owned(value: object, *, directory: bool = False) -> Path:
        relative = _relative(value)
        if not directory and relative not in all_entries:
            raise RocmRuntimeError(
                f"Asset is absent from component inventories: {relative}."
            )
        path = _asset(root, relative, directory=directory)
        if not directory and path.stat().st_size != all_entries[relative]["size"]:
            raise RocmRuntimeError(f"Owned ROCm asset size mismatch: {path}.")
        return path

    _validate_provenance(manifest, loader, owned)
    _validate_deduplicated(components, inventories)
    libraries, sonames, paths = _inspect_libraries(loader, all_entries, owned)
    preload = _inspect_preload(loader, libraries, sonames, paths, owned)
    compiler, compiler_paths = _inspect_compiler(loader, owned)
    kpack = _inspect_kpack(loader, inventories, owned)
    if not any(
        name.startswith(compiler["bitcode_path"] + "/") and name.endswith(".bc")
        for name in all_entries
    ):
        raise RocmRuntimeError("Missing owned ROCm compiler bitcode inventory.")
    plugin_objects = _inspect_plugin_objects(loader, root, sonames)
    assets = tuple(
        (str((root / name).resolve()), entry["sha256"])
        for name, entry in sorted(all_entries.items())
    )
    return RocmBundle(
        SDK_VERSION,
        ARCHITECTURES,
        str(manifest_path),
        tuple(facts),
        tuple(libraries),
        preload,
        compiler_paths,
        tuple(kpack),
        tuple(plugin_objects),
        assets,
    )


def _loaded_paths() -> tuple[Path, ...]:
    """Observe Linux mappings without calling dlopen or a driver API."""
    try:
        lines = Path("/proc/self/maps").read_text().splitlines()
    except OSError as error:
        raise RocmRuntimeError(
            "Cannot establish already-loaded library identity from /proc/self/maps."
        ) from error
    paths = set()
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            paths.add(Path(fields[5].removesuffix(" (deleted)")).resolve())
    return tuple(paths)


def _refuse_foreign_libraries(bundle: RocmBundle) -> None:
    owned = {Path(entry[0]) for entry in bundle.libraries}
    names = {entry[1] for entry in bundle.libraries}
    for path in _loaded_paths():
        if (
            path.name in names or _ROCM_LIBRARY.fullmatch(path.name)
        ) and path not in owned:
            raise RocmRuntimeError(
                f"Foreign ROCm library is already loaded: {path}; start a clean process before importing Phydrax."
            )


def activate_rocm_bundle(*, _importing: str | None = None) -> RocmBundle:
    """Prepare the compiler and load the owned closure before the JAX plugin."""
    global _LOADED, _FAILURE, _INITIAL_ROCM_ENVIRONMENT
    with _LOCK:
        if _FAILURE is not None:
            raise RocmRuntimeError(_FAILURE)
        if _LOADED is not None:
            _refuse_foreign_libraries(_LOADED)
            return _LOADED
        try:
            if sys.platform != "linux" or platform.machine() != "x86_64":
                raise RocmRuntimeError("Owned ROCm requires Linux x86_64.")
            bundle = inspect_rocm_bundle()
            if bundle is None:
                raise RocmRuntimeError(
                    "Missing owned ROCm bundle; install phydrax[rocm10]."
                )
            for name in ("jax", "jaxlib", "jax-rocm10-plugin", "jax-rocm10-pjrt"):
                if _distribution(name).version != "0.12.0":
                    raise RocmRuntimeError(f"ROCm 10 requires {name}==0.12.0.")
            if any(
                name in sys.modules and name != _importing
                for name in (_PLUGIN, "jax_rocm10_plugin")
            ):
                raise RocmRuntimeError(
                    "ROCm plugin was imported before the owned loader; start a fresh Python process with site initialization enabled."
                )
            _refuse_foreign_libraries(bundle)

            # Build-time ELF inspection is bound by the manifest and each wheel's
            # RECORD. Checking those bytes here needs no JAX-dependent ELF types.
            payloads = (
                *bundle.assets,
                *((path, digest) for path, _, _, digest in bundle.plugin_objects),
            )
            for path, digest in payloads:
                with open(path, "rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != digest:
                    raise RocmRuntimeError(f"Owned ROCm payload digest mismatch: {path}.")
            compiler = dict(bundle.compiler)
            for variable, key in (
                ("LLVM_PATH", "llvm_path"),
                ("ROCM_PATH", "rocm_path"),
                ("ROCM_HOME", "rocm_path"),
                ("HIP_DEVICE_LIB_PATH", "bitcode_path"),
            ):
                existing = os.environ.get(variable)
                if existing and Path(existing).resolve() != Path(compiler[key]):
                    raise RocmRuntimeError(
                        f"{variable} conflicts with the owned ROCm bundle; unset it before starting Python."
                    )
            _INITIAL_ROCM_ENVIRONMENT = tuple(
                (name, os.environ[name])
                for name in ("ROCM_PATH", "ROCM_HOME")
                if name in os.environ
            )
            for variable, key in (
                ("LLVM_PATH", "llvm_path"),
                ("ROCM_PATH", "rocm_path"),
                ("ROCM_HOME", "rocm_path"),
                ("HIP_DEVICE_LIB_PATH", "bitcode_path"),
            ):
                os.environ[variable] = compiler[key]
            import ctypes

            for path in bundle.preload:
                # Retain every successful handle even if a later load fails.
                _HANDLES.append(ctypes.CDLL(path, mode=os.RTLD_NOW | ctypes.RTLD_GLOBAL))
            _refuse_foreign_libraries(bundle)
            _LOADED = bundle
            return bundle
        except (OSError, ValueError, RuntimeError) as error:
            _FAILURE = f"Owned ROCm activation failed: {error}"
            raise RocmRuntimeError(_FAILURE) from error


def rocm_bundle_loaded() -> bool:
    return _LOADED is not None


def initial_rocm_environment() -> tuple[tuple[str, str], ...] | None:
    """Preserve external SDK selection independently of automatic compiler setup."""
    return _INITIAL_ROCM_ENVIRONMENT


class _RocmLoader(importlib.abc.Loader):
    def __init__(self, loader: importlib.abc.Loader, fullname: str) -> None:
        self.loader = loader
        self.fullname = fullname

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return self.loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        activate_rocm_bundle(_importing=self.fullname)
        self.loader.exec_module(module)


class _RocmFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname not in ("jax_rocm10_plugin", _PLUGIN):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _RocmLoader(spec.loader, fullname)
        return spec


def install_rocm_loader() -> None:
    """Install a narrow finder; do not inspect hardware or load any SDK library."""
    global _FINDER
    with _LOCK:
        if _FINDER is None:
            _FINDER = _RocmFinder()
            sys.meta_path.insert(0, _FINDER)
