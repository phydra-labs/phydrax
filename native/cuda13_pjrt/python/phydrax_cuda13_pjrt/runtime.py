# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Windows CUDA wheel admission and driver-hybrid process-lifetime ownership.

The immutable records below are the interchange boundary to core Phydrax. No
function imports core Phydrax or JAX. ``installation_record`` only reads files
and distribution metadata. ``mapped_libraries`` and ``runtime_identity`` observe
the current process; ``cuda_runtime_api`` binds an admitted companion without
invoking CUDA. ``driver_version`` explicitly queries CUDA, and ``load_libraries``
loads native DLLs. Dependency closure is preloading, not Windows-loader emulation.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import importlib.metadata as metadata
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath
from threading import RLock
from typing import Any, Literal, NamedTuple, TYPE_CHECKING


if TYPE_CHECKING:
    from phydrax._pe_dependencies import inspect_pe, is_system_import, PeExport
else:
    from ._pe_dependencies import inspect_pe, is_system_import, PeExport

DISTRIBUTION = "phydrax-cuda13-pjrt"
_MANIFEST = "phydrax_cuda13_pjrt/manifest.json"
_INTERPRETER_DLLS = frozenset(
    ("python3.dll", "python313.dll", "vcruntime140.dll", "vcruntime140_1.dll")
)
_LOAD_FLAGS = 0x00000100 | 0x00000800  # DLL_LOAD_DIR | SYSTEM32, never cwd/PATH.
_SESSION_LOCK = RLock()
_LIBRARIES: dict[str, ctypes.CDLL] = {}
_DIRECTORIES: dict[str, Any] = {}
_KERNELS_READY = False
_CUDA_APIS: dict[str, CudaRuntimeAPI] = {}


class AssetRecord(NamedTuple):
    path: str
    sha256: str
    size: int
    role: str


class InstallationRecord(NamedTuple):
    distribution: str
    version: str
    cuda_major: int
    jax_version: str
    jaxlib_version: str
    build_id: str
    manifest_path: str
    assets: tuple[AssetRecord, ...]
    capabilities: tuple[str, ...]
    issues: tuple[str, ...]


class MappedLibrary(NamedTuple):
    path: str
    name: str
    distribution: str | None
    version: str | None
    cuda_major: int | None


class RuntimeIdentity(NamedTuple):
    """Observed driver-owned runtime cohort; ``_asdict()`` is the JSON contract.

    Paths and SHA-256 digests identify resident files, not wheel assets. This
    passive record makes no CUDA API/version or device-initialization claim.
    """

    linkage: Literal["driver-hybrid"]
    owner: Literal["display-driver"]
    runtime_path: str
    runtime_sha256: str
    driver_path: str
    driver_sha256: str
    driver_dispatch_path: str
    driver_dispatch_sha256: str


_INT = ctypes.c_int
_POINTER = ctypes.c_void_p
_CUDA_SIGNATURES: dict[str, tuple[Any, ...]] = {
    "cudaGetDevice": (ctypes.POINTER(_INT),),
    "cudaGetDeviceCount": (ctypes.POINTER(_INT),),
    "cudaSetDevice": (_INT,),
    "cudaDeviceSynchronize": (),
    "cudaDriverGetVersion": (ctypes.POINTER(_INT),),
    "cudaRuntimeGetVersion": (ctypes.POINTER(_INT),),
    "cudaMemGetInfo": (
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ),
    "cudaDeviceGetAttribute": (ctypes.POINTER(_INT), _INT, _INT),
    "cudaDeviceCanAccessPeer": (ctypes.POINTER(_INT), _INT, _INT),
    "cudaMalloc": (ctypes.POINTER(_POINTER), ctypes.c_size_t),
    "cudaFree": (_POINTER,),
    "cudaMemcpy": (_POINTER, _POINTER, ctypes.c_size_t, _INT),
    "cudaMemcpyPeer": (_POINTER, _INT, _POINTER, _INT, ctypes.c_size_t),
    "cudaGetDriverEntryPointByVersion": (
        ctypes.c_char_p,
        ctypes.POINTER(_POINTER),
        ctypes.c_uint,
        ctypes.c_ulonglong,
        ctypes.POINTER(_INT),
    ),
}


class CudaRuntimeAPI:
    """Public CUDART pointers owned by a retained, admitted numerical companion.

    This is not a DLL handle: the hybrid DLL has no public CUDART exports. Lookup
    calls only the companion's vendor-linked accessor, never a CUDA operation.
    """

    __slots__ = ("_library", "_functions", "owner_path")

    def __init__(self, library: ctypes.CDLL, owner_path: str) -> None:
        self._library = library
        self.owner_path = owner_path
        self._functions: dict[str, Any] = {}
        try:
            accessor = library.phx_cuda_runtime_function
            error = library.phx_cuda_last_error
        except AttributeError as cause:
            raise RuntimeError(
                f"CUDA companion lacks the public runtime accessor: {owner_path}"
            ) from cause
        accessor.argtypes = [ctypes.c_char_p]
        accessor.restype = ctypes.c_void_p
        error.argtypes = []
        error.restype = ctypes.c_char_p
        for name, arguments in _CUDA_SIGNATURES.items():
            address = accessor(name.encode("ascii"))
            if not address:
                detail = error()
                message = detail.decode("utf-8", errors="replace") if detail else ""
                raise RuntimeError(
                    f"CUDA companion refused runtime API {name}: {message}"
                )
            # CUDARTAPI is __stdcall on Windows; Win64 has one unified ABI.
            self._functions[name] = ctypes.WINFUNCTYPE(_INT, *arguments)(address)

    def function(self, name: str) -> Any:
        """Return only a signature-known public API; arbitrary pointer calls refuse."""
        if not isinstance(name, str) or name not in self._functions:
            raise ValueError(f"Unknown CUDA runtime API: {name!r}")
        return self._functions[name]

    def __getattr__(self, name: str) -> Any:
        try:
            return self._functions[name]
        except KeyError:
            raise AttributeError(name) from None


class NativeObject(NamedTuple):
    path: str
    machine: int
    imports: tuple[str, ...]
    delay_imports: tuple[str, ...]
    exports: tuple[PeExport, ...]


class DependencyEdge(NamedTuple):
    requester: str
    name: str
    path: str | None
    owner: str | None
    kind: Literal["owned", "resident", "system", "unresolved"]


class DependencyClosure(NamedTuple):
    root: str
    machine: int
    objects: tuple[NativeObject, ...]
    edges: tuple[DependencyEdge, ...]
    issues: tuple[str, ...]
    complete: bool


class _OwnedFile(NamedTuple):
    path: Path
    distribution: str
    version: str
    record_hash: str
    size: int
    cuda_major: int | None


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _distribution(name: str) -> metadata.Distribution:
    dist = metadata.distribution(name)
    identity = dist.metadata
    installed_name, version = identity.get("Name"), identity.get("Version")
    if (
        not isinstance(installed_name, str)
        or _normalized(installed_name) != _normalized(name)
        or not isinstance(version, str)
        or not version.strip()
    ):
        raise RuntimeError(
            f"Missing or invalid installed metadata for {name}: "
            f"Name={installed_name!r}, Version={version!r}"
        )
    return dist


def _key(path: str | os.PathLike[str]) -> str:
    return str(Path(path).resolve()).casefold()


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _relative(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or re.search(r'[\\:\x00-\x1f<>"|?*]', value)
    ):
        raise RuntimeError("Runtime asset path must be a canonical relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in (".", "..") or part.endswith((".", " ")) for part in path.parts)
    ):
        raise RuntimeError(f"Runtime asset escapes its wheel: {value!r}")
    if path.as_posix() != value:
        raise RuntimeError(f"Noncanonical runtime asset path: {value!r}")
    return value


def _record_file(dist: metadata.Distribution, relative: str) -> _OwnedFile:
    files = dist.files
    if files is None:
        raise RuntimeError(f"{dist.metadata['Name']} has no installed RECORD")
    matches = [item for item in files if item.as_posix() == relative]
    if len(matches) != 1:
        raise RuntimeError(f"Missing or duplicate RECORD ownership for {relative}")
    item = matches[0]
    if item.hash is None or item.hash.mode != "sha256" or item.size is None:
        raise RuntimeError(f"Missing SHA-256/size in RECORD for {relative}")
    located_root, located_file = dist.locate_file(""), dist.locate_file(item)
    if not isinstance(located_root, os.PathLike) or not isinstance(
        located_file, os.PathLike
    ):
        raise RuntimeError("CUDA runtime wheels must be installed as filesystem files")
    root = Path(located_root).resolve()
    path = Path(located_file).resolve()
    if not path.is_relative_to(root):
        raise RuntimeError(f"RECORD path escapes installed wheel root: {relative}")
    name = _normalized(dist.metadata["Name"])
    return _OwnedFile(
        path,
        name,
        dist.version,
        item.hash.value,
        item.size,
        13 if name != "jaxlib" else None,
    )


def _verify_owned(item: _OwnedFile) -> str:
    digest = _sha256(item.path)
    expected = (
        base64.urlsafe_b64encode(bytes.fromhex(digest)).rstrip(b"=").decode("ascii")
    )
    if item.path.stat().st_size != item.size or item.record_hash != expected:
        raise RuntimeError(f"Installed RECORD integrity failure: {item.path}")
    return digest


def _manifest_assets(
    dist: metadata.Distribution, manifest: dict[str, Any]
) -> tuple[AssetRecord, ...]:
    assets = manifest.get("assets")
    if not isinstance(assets, list) or not assets:
        raise RuntimeError("Runtime manifest has no asset inventory")
    result = []
    names = set()
    for item in assets:
        if not isinstance(item, dict):
            raise RuntimeError("Invalid runtime manifest asset")
        relative = _relative(item.get("path"))
        if relative.casefold() in names:
            raise RuntimeError(f"Duplicate runtime asset: {relative}")
        names.add(relative.casefold())
        digest, size, role = item.get("sha256"), item.get("size"), item.get("role")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeError(f"Invalid asset digest for {relative}")
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(role, str)
        ):
            raise RuntimeError(f"Invalid asset size/role for {relative}")
        owned = _record_file(dist, relative)
        actual = _verify_owned(owned)
        if owned.size != size or actual != digest:
            raise RuntimeError(f"Runtime manifest integrity failure: {relative}")
        result.append(AssetRecord(str(owned.path), digest, size, role))
    if sum(item.role == "pjrt" for item in result) != 1 or not any(
        item.role == "kernel" for item in result
    ):
        raise RuntimeError(
            "Runtime manifest requires one PJRT DLL and compiled CUDA kernels"
        )
    return tuple(sorted(result))


def _required_vendor_distributions(
    dist: metadata.Distribution,
) -> tuple[metadata.Distribution, ...]:
    result = []
    for requirement in dist.metadata.get_all("Requires-Dist") or ():
        match = re.match(r"([A-Za-z0-9_.-]+)", requirement)
        if match is None or not _normalized(match[1]).startswith("nvidia-"):
            continue
        pin = re.fullmatch(r"([A-Za-z0-9_.-]+)\s*==\s*([0-9.]+)", requirement.strip())
        if pin is None:
            raise RuntimeError(
                f"Runtime NVIDIA dependency is not exactly pinned: {requirement}"
            )
        installed = _distribution(pin[1])
        if installed.version != pin[2]:
            raise RuntimeError(
                f"Runtime requires {requirement}, found {installed.version}"
            )
        result.append(installed)
    return tuple(result)


def _manifest_json(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"Duplicate runtime manifest key: {key}")
            result[key] = value
        return result

    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    if not isinstance(result, dict):
        raise RuntimeError("Runtime manifest must be an object")
    return result


def installation_record() -> InstallationRecord:
    """Read and verify installed identity/assets without loading a native library.

    Missing/corrupt files and distribution metadata retain their concrete I/O
    errors. Unsupported identity raises RuntimeError. No capabilities are inferred.
    """
    dist = _distribution(DISTRIBUTION)
    owned = _record_file(dist, _MANIFEST)
    _verify_owned(owned)
    manifest = _manifest_json(owned.path)
    _required_vendor_distributions(dist)
    expected = {
        "distribution": DISTRIBUTION,
        "version": dist.version,
        "cuda_major": 13,
        "jax_version": "0.12.0",
        "jaxlib_version": "0.12.0",
        "python_tag": "cp313",
        "abi_tag": "cp313",
        "platform_tag": "win_amd64",
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise RuntimeError(
                f"Unsupported runtime manifest {name}: {manifest.get(name)!r}; expected {value!r}"
            )
    build_id = manifest.get("build_id")
    body = {name: value for name, value in manifest.items() if name != "build_id"}
    digest = hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()
    if build_id != digest:
        raise RuntimeError("Runtime manifest build identity does not match its contents")
    capabilities = manifest.get("capabilities")
    if not isinstance(capabilities, list) or not all(
        isinstance(item, str) for item in capabilities
    ):
        raise RuntimeError(
            "Runtime manifest is missing its explicit compiler capability record"
        )
    if capabilities != sorted(set(capabilities)):
        raise RuntimeError("Compiler capability record must be canonical and unique")
    for dependency in ("jax", "jaxlib"):
        if metadata.version(dependency) != "0.12.0":
            raise RuntimeError(f"The Windows runtime requires stock {dependency}==0.12.0")
    owners = []
    for candidate in metadata.distributions():
        if any(
            item.parts and item.parts[0] == "jax_cuda13_plugin"
            for item in candidate.files or ()
        ):
            name = candidate.metadata.get("Name")
            if not isinstance(name, str) or not name.strip():
                raise RuntimeError("CUDA kernel owner has missing Name metadata")
            owners.append(_normalized(name))
    if owners != [DISTRIBUTION]:
        raise RuntimeError(
            f"Conflicting CUDA kernel package ownership: {sorted(owners)!r}"
        )
    assets = _manifest_assets(dist, manifest)
    return InstallationRecord(
        DISTRIBUTION,
        dist.version,
        13,
        "0.12.0",
        "0.12.0",
        digest,
        str(owned.path),
        assets,
        tuple(capabilities),
        (),
    )


def _owned_inventory() -> dict[str, _OwnedFile]:
    runtime = _distribution(DISTRIBUTION)
    names = {DISTRIBUTION, "jaxlib"}
    names.update(
        _normalized(item.metadata["Name"])
        for item in _required_vendor_distributions(runtime)
    )
    try:
        companion = _distribution("phydrax-cuda13")
    except metadata.PackageNotFoundError:
        companion = None
    if companion is not None:
        if companion.version != runtime.version:
            raise RuntimeError(
                f"Windows CUDA runtime requires phydrax-cuda13=={runtime.version}, "
                f"found version {companion.version}"
            )
        names.add("phydrax-cuda13")
        for requirement in companion.metadata.get_all("Requires-Dist") or ():
            match = re.match(r"([A-Za-z0-9_.-]+)", requirement)
            if (
                match
                and _normalized(match[1]).startswith("nvidia-")
                and ";" not in requirement
            ):
                names.add(_normalized(match[1]))
    # cuTENSOR is a separately selected core extra, not a mandatory companion
    # dependency. Its wheel is an admissible owner only when actually installed.
    try:
        cutensor = _distribution("cutensor-cu13")
    except metadata.PackageNotFoundError:
        cutensor = None
    if cutensor is not None:
        if cutensor.version != "2.8.1":
            raise RuntimeError(
                f"Windows CUDA runtime requires cutensor-cu13==2.8.1 when selected, found {cutensor.version}"
            )
        names.add("cutensor-cu13")
    inventory = {}
    for name in sorted(names):
        dist = _distribution(name)
        if dist.files is None:
            raise RuntimeError(f"Missing installed RECORD for {name}")
        for item in dist.files:
            if item.suffix.lower() not in (".dll", ".pyd", ".exe", ".bc"):
                continue
            relative = _relative(item.as_posix())
            owned = _record_file(dist, relative)
            key = _key(owned.path)
            if key in inventory:
                raise RuntimeError(f"Ambiguous installed RECORD ownership: {owned.path}")
            inventory[key] = owned
    return inventory


def _kernel32() -> Any:
    if sys.platform != "win32":
        raise RuntimeError("Windows CUDA DLL observation requires native Windows")
    return ctypes.WinDLL("kernel32.dll", winmode=0x00000800, use_last_error=True)


def _system_directory() -> Path:
    kernel = _kernel32()
    function = kernel.GetSystemDirectoryW
    function.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32]
    function.restype = ctypes.c_uint32
    buffer = ctypes.create_unicode_buffer(32768)
    length = function(buffer, len(buffer))
    if length == 0 or length >= len(buffer):
        raise ctypes.WinError(ctypes.get_last_error())
    return Path(buffer.value).resolve()


def _mapped_paths() -> tuple[str, ...]:
    kernel = _kernel32()
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    enum = kernel.K32EnumProcessModules
    enum.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    enum.restype = ctypes.c_int
    filename = kernel.GetModuleFileNameW
    filename.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
    filename.restype = ctypes.c_uint32
    count = 256
    while count <= 65536:
        modules = (ctypes.c_void_p * count)()
        needed = ctypes.c_uint32()
        if not enum(
            kernel.GetCurrentProcess(),
            modules,
            ctypes.sizeof(modules),
            ctypes.byref(needed),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if needed.value <= ctypes.sizeof(modules):
            break
        count = needed.value // ctypes.sizeof(ctypes.c_void_p) + 64
    else:
        raise RuntimeError("Windows module inventory exceeds bounded capacity")
    paths = []
    for module in modules[: needed.value // ctypes.sizeof(ctypes.c_void_p)]:
        buffer = ctypes.create_unicode_buffer(32768)
        length = filename(module, buffer, len(buffer))
        if length == 0 or length >= len(buffer):
            raise ctypes.WinError(ctypes.get_last_error())
        paths.append(str(Path(buffer.value).resolve()))
    if not paths:
        raise RuntimeError("Windows module inventory returned no executable module")
    return tuple(sorted(set(paths), key=str.casefold))


def mapped_libraries() -> tuple[MappedLibrary, ...]:
    """Observe resident modules; unavailable inventory raises, never means clean."""
    owned = _owned_inventory()
    result = []
    for path in _mapped_paths():
        owner = owned.get(_key(path))
        result.append(
            MappedLibrary(
                path,
                Path(path).name,
                owner.distribution if owner else None,
                owner.version if owner else None,
                owner.cuda_major if owner else None,
            )
        )
    return tuple(result)


def _basename_index(owned: dict[str, _OwnedFile]) -> dict[str, list[_OwnedFile]]:
    # Roots are admitted by full path. PE imports need not end in ".dll", and
    # Python extension roots commonly share a basename across package directories.
    result: dict[str, list[_OwnedFile]] = {}
    for item in owned.values():
        result.setdefault(item.path.name.casefold(), []).append(item)
    return result


def _resident_index(paths: tuple[str, ...]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for path in paths:
        matches = result.setdefault(Path(path).name.casefold(), [])
        if all(_key(existing) != _key(path) for existing in matches):
            matches.append(path)
    return result


def _driver_cohort(
    resident: dict[str, list[str]], *, required: bool
) -> tuple[Path | None, Path | None, Path | None]:
    """Validate observed ownership even when lazy driver loading is incomplete."""
    names = ("nvcudart_hybrid64.dll", "nvcuda64.dll", "nvcuda.dll")
    for name in resident:
        if (name.startswith("nvcudart") and name not in names) or (
            name.startswith("cudart") and name.endswith(".dll")
        ):
            raise RuntimeError(
                f"Conflicting resident CUDA runtime module: {resident[name]!r}; "
                "the Windows CUDA 13 runtime requires the display-driver hybrid cohort"
            )
    hybrid, actual, dispatch = (
        Path(mapped).resolve() if (mapped := _resident_import(name, resident)) else None
        for name in names
    )
    paths = hybrid, actual, dispatch
    if not any(paths) and not required:
        return paths
    system = _system_directory()
    if dispatch is not None and _key(dispatch) != _key(system / "nvcuda.dll"):
        raise RuntimeError(
            f"Resident NVIDIA driver dispatch has an unexpected owner: {dispatch}"
        )
    repository = (system / "DriverStore/FileRepository").resolve()
    for path in (hybrid, actual):
        if path is None:
            continue
        in_system = _key(path.parent) == _key(system)
        in_repository = (
            path.is_relative_to(repository)
            and len(path.relative_to(repository).parts) == 2
        )
        if not (in_system or in_repository):
            raise RuntimeError(
                f"Resident NVIDIA driver module has an unexpected owner: {path}"
            )
    if (
        hybrid is not None
        and actual is not None
        and _key(hybrid.parent) != _key(actual.parent)
    ):
        raise RuntimeError(
            f"Conflicting NVIDIA driver cohort: hybrid {hybrid}, actual driver {actual}"
        )
    if required and any(path is None for path in paths):
        missing = [name for name, path in zip(names, paths) if path is None]
        raise RuntimeError(
            "Missing resident NVIDIA driver-hybrid cohort: " + ", ".join(missing)
        )
    return paths


def runtime_identity() -> RuntimeIdentity:
    """Hash the current process's complete driver cohort, without invoking CUDA.

    Missing modules are unobserved, not an inferred successful runtime. Loading
    and explicit version/device queries belong to callers, never this observer.
    """
    with _SESSION_LOCK:
        hybrid, actual, dispatch = _driver_cohort(
            _resident_index(_mapped_paths()), required=True
        )
        assert hybrid is not None and actual is not None and dispatch is not None
        return RuntimeIdentity(
            "driver-hybrid",
            "display-driver",
            str(hybrid),
            _sha256(hybrid),
            str(actual),
            _sha256(actual),
            str(dispatch),
            _sha256(dispatch),
        )


def _owned_import(name: str, owned: dict[str, list[_OwnedFile]]) -> _OwnedFile | None:
    matches = owned.get(name.casefold(), ())
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous wheel DLL basename {name}: "
            + ", ".join(str(item.path) for item in matches)
        )
    return matches[0] if matches else None


def _resident_import(name: str, resident: dict[str, list[str]]) -> str | None:
    matches = resident.get(name.casefold(), ())
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple resident DLLs share required basename {name}: "
            + ", ".join(matches)
        )
    return matches[0] if matches else None


def _check_import_owner(name: str, path: Path, resident: dict[str, list[str]]) -> None:
    if name.casefold() == "nvml.dll" and _key(path) == _key(
        _system_directory() / "nvml.dll"
    ):
        _nvml_dispatch(path.parent, resident)
        return
    mapped = _resident_import(name, resident)
    if mapped is not None and _key(mapped) != _key(path):
        raise RuntimeError(
            f"DLL import-order conflict for {name}: resident {mapped}, owned {path}; use separate processes"
        )


def _check_resident(
    owned: dict[str, list[_OwnedFile]], resident: dict[str, list[str]]
) -> None:
    _driver_cohort(resident, required=False)
    # cuDNN dispatches lazily by basename, so reject even unmatched foreign family members.
    families = (
        "cudnn",
        "cublas",
        "cudart",
        "cufft",
        "cusolver",
        "cusparse",
        "curand",
        "cudss",
        "cutensor",
        "nvrtc",
        "nvjitlink",
        "cupti",
    )
    for name in owned:
        if name.startswith(families) and name.endswith(".dll"):
            _owned_import(name, owned)
    for name in resident:
        if not (name.startswith(families) and name.endswith(".dll")):
            continue
        candidate = _owned_import(name, owned)
        path = _resident_import(name, resident)
        if candidate is None:
            raise RuntimeError(
                f"Unowned resident CUDA DLL family member {path}; use separate processes"
            )
        _check_import_owner(name, candidate.path, resident)


def _nvml_dispatch(system: Path, resident: dict[str, list[str]]) -> str | None:
    # NVIDIA's System32 NVML dispatcher loads another nvml.dll from its driver
    # package. Both may be resident; a second basename is not a second owner
    # when its exact directory belongs to the observed CUDA driver cohort.
    dispatch = system / "nvml.dll"
    _, driver, _ = _driver_cohort(resident, required=False)
    allowed = {_key(dispatch)}
    if driver is not None:
        allowed.add(_key(driver.parent / "nvml.dll"))
    mapped = resident.get("nvml.dll", ())
    for path in mapped:
        if _key(path) not in allowed:
            raise RuntimeError(
                f"Resident NVIDIA management DLL has an unexpected driver owner: {path}"
            )
    return str(dispatch) if any(_key(path) == _key(dispatch) for path in mapped) else None


def _system_edge(
    requester: str, name: str, system: Path, resident: dict[str, list[str]]
) -> DependencyEdge:
    lower = name.casefold()
    mapped = (
        _nvml_dispatch(system, resident)
        if lower == "nvml.dll"
        else _resident_import(name, resident)
    )
    if lower.startswith(("api-ms-win-", "ext-ms-win-")):
        return DependencyEdge(requester, name, None, "Windows API-set", "system")
    if lower in _INTERPRETER_DLLS:
        roots = (
            Path(sys.base_prefix).resolve(),
            Path(sys.executable).resolve().parent,
        )
        if mapped is not None and Path(mapped).resolve().parent in roots:
            return DependencyEdge(
                requester, name, str(Path(mapped).resolve()), "CPython", "resident"
            )
        for root in roots:
            candidate = root / lower
            if candidate.is_file():
                _check_import_owner(name, candidate, resident)
                inspect_pe(candidate)
                return DependencyEdge(
                    requester, name, str(candidate), "CPython", "system"
                )
    candidate = system / name
    if candidate.is_file():
        if mapped is not None and _key(mapped) != _key(candidate):
            raise RuntimeError(
                f"Resident system/driver DLL has an unexpected owner: {mapped}; expected {candidate}"
            )
        return DependencyEdge(
            requester,
            name,
            str(candidate.resolve()),
            "NVIDIA driver" if lower in ("nvcuda.dll", "nvml.dll") else "Windows",
            "system",
        )
    return DependencyEdge(requester, name, None, None, "unresolved")


def _closure(
    path: str | os.PathLike[str],
    inventory: dict[str, _OwnedFile],
    resident: dict[str, list[str]],
) -> DependencyClosure:
    root = Path(path).resolve()
    by_name = _basename_index(inventory)
    _check_resident(by_name, resident)
    if _key(root) not in inventory:
        raise RuntimeError(f"Native root has no admitted wheel RECORD owner: {root}")
    system = _system_directory()
    objects: list[NativeObject] = []
    edges: list[DependencyEdge] = []
    issues: list[str] = []
    visited: set[str] = set()

    def visit(current: Path) -> None:
        key = _key(current)
        if key in visited:
            return
        visited.add(key)
        if len(visited) > 4096:
            raise RuntimeError("Native dependency closure exceeds bounded capacity")
        _verify_owned(inventory[key])
        image = inspect_pe(current)
        forwarders = set()
        for exported in image.exports:
            if exported.forwarder is not None:
                module = exported.forwarder.rsplit(".", 1)[0]
                forwarders.add(
                    module if module.lower().endswith(".dll") else module + ".dll"
                )
        for name in sorted(
            set(image.imports + image.delay_imports) | forwarders, key=str.casefold
        ):
            item = _owned_import(name, by_name)
            if item is not None:
                _check_import_owner(name, item.path, resident)
                edge = DependencyEdge(
                    str(current), name, str(item.path), item.distribution, "owned"
                )
                edges.append(edge)
                visit(item.path)
            elif is_system_import(name) or name.casefold() in _INTERPRETER_DLLS:
                edge = _system_edge(str(current), name, system, resident)
                edges.append(edge)
                if edge.kind == "unresolved":
                    issues.append(
                        f"Unresolved system/driver import {name} required by {current}"
                    )
            else:
                _resident_import(name, resident)
                edges.append(DependencyEdge(str(current), name, None, None, "unresolved"))
                issues.append(f"Unowned DLL import {name} required by {current}")
        objects.append(
            NativeObject(
                str(current),
                image.machine,
                image.imports,
                image.delay_imports,
                image.exports,
            )
        )

    visit(root)
    return DependencyClosure(
        str(root),
        0x8664,
        tuple(objects),
        tuple(edges),
        tuple(sorted(set(issues))),
        not issues,
    )


def dependency_closure(path: str | os.PathLike[str]) -> DependencyClosure:
    """Inspect admitted PE closure and resident conflicts without loading its DLLs."""
    return _closure(path, _owned_inventory(), _resident_index(_mapped_paths()))


def load_libraries(path: str | os.PathLike[str]) -> tuple[str, ...]:
    """Preload the verified closure by absolute path, retaining all native handles.

    The returned paths include the root and are dependency-first. OS/API-set and
    driver dependencies are not attributed to wheels. Already owned handles are
    reused. Failure is retained by the caller; no partial success is returned.
    """
    with _SESSION_LOCK:
        inventory = _owned_inventory()
        closure = _closure(path, inventory, _resident_index(_mapped_paths()))
        if not closure.complete:
            raise RuntimeError(
                "Cannot preload native dependency closure: " + "; ".join(closure.issues)
            )
        # The stable-ABI python3.dll may not be resident yet. Kernel directories
        # are not interpreter directories, and loading must not consult PATH.
        for edge in closure.edges:
            if edge.owner == "CPython" and edge.path is not None:
                key = _key(edge.path)
                if key not in _LIBRARIES:
                    _LIBRARIES[key] = ctypes.WinDLL(edge.path, winmode=_LOAD_FLAGS)
        for item in closure.objects:
            key = _key(item.path)
            directory = str(Path(item.path).parent)
            if directory.casefold() not in _DIRECTORIES:
                _DIRECTORIES[directory.casefold()] = os.add_dll_directory(directory)
            if key not in _LIBRARIES:
                # DLL_LOAD_DIR is only this absolute root's directory. Dependencies
                # from other wheel directories have already been loaded above.
                _LIBRARIES[key] = ctypes.WinDLL(item.path, winmode=_LOAD_FLAGS)
        resident = _resident_index(_mapped_paths())
        _check_resident(_basename_index(inventory), resident)
        for edge in closure.edges:
            if edge.path is not None:
                _check_import_owner(edge.name, Path(edge.path), resident)
        return tuple(item.path for item in closure.objects)


def library_handle(path: str | os.PathLike[str]) -> ctypes.CDLL:
    """Return an already admitted handle; never resolve a basename implicitly."""
    with _SESSION_LOCK:
        handle = _LIBRARIES.get(_key(path))
        if handle is None:
            raise RuntimeError(f"Native library has not been admitted: {path}")
        return handle


def cuda_runtime_api(companion: ctypes.CDLL) -> CudaRuntimeAPI:
    """Bind a previously admitted companion, without loading or initializing CUDA."""
    with _SESSION_LOCK:
        keys = [key for key, library in _LIBRARIES.items() if library is companion]
        if len(keys) != 1:
            raise RuntimeError(
                "CUDA runtime API requires an already admitted companion handle"
            )
        key = keys[0]
        _driver_cohort(_resident_index(_mapped_paths()), required=False)
        if key not in _CUDA_APIS:
            owned = _owned_inventory().get(key)
            if owned is None or owned.distribution != "phydrax-cuda13":
                raise RuntimeError(
                    "CUDA runtime API requires a RECORD-owned phydrax-cuda13 companion"
                )
            dist = _distribution("phydrax-cuda13")
            manifest = _record_file(dist, "phydrax_cuda13/phx_cuda_build.json")
            _verify_owned(manifest)
            linkage = _manifest_json(manifest.path).get("runtime_linkage")
            if linkage != {
                "kind": "driver-hybrid",
                "target_dll": "nvcudart_hybrid64.dll",
                "owner": "display-driver",
            }:
                raise RuntimeError(
                    "CUDA companion does not declare the canonical driver-hybrid runtime linkage"
                )
            _CUDA_APIS[key] = CudaRuntimeAPI(companion, str(owned.path))
        return _CUDA_APIS[key]


def cuda_driver_library() -> ctypes.CDLL:
    """Return the retained NVIDIA driver from Windows System32, never a search path."""
    path = _system_directory() / "nvcuda.dll"
    with _SESSION_LOCK:
        resident = _resident_index(_mapped_paths())
        _driver_cohort(resident, required=False)
        mapped = _resident_import("nvcuda.dll", resident)
        if mapped is not None and _key(mapped) != _key(path):
            raise RuntimeError(
                f"Resident NVIDIA driver has an unexpected owner: {mapped}"
            )
        key = _key(path)
        if key not in _LIBRARIES:
            _LIBRARIES[key] = ctypes.WinDLL(str(path), winmode=_LOAD_FLAGS)
        _driver_cohort(_resident_index(_mapped_paths()), required=False)
        return _LIBRARIES[key]


def driver_version() -> int:
    """Explicitly load the OS-owned NVIDIA driver and query its CUDA API version."""
    driver = cuda_driver_library()
    function = driver.cuDriverGetVersion
    function.argtypes = [ctypes.POINTER(ctypes.c_int)]
    function.restype = ctypes.c_int
    result = ctypes.c_int()
    status = function(ctypes.byref(result))
    if status != 0:
        raise RuntimeError(f"cuDriverGetVersion failed with CUDA status {status}")
    if result.value < 13000:
        raise RuntimeError(
            f"CUDA 13 requires a compatible driver; reported API {result.value}"
        )
    return result.value


def compiler_root() -> str:
    """Verify the complete wheel-owned assembler/linker/libdevice toolchain."""
    required = ("ptxas.exe", "nvlink.exe", "libdevice.10.bc")
    tools: dict[str, _OwnedFile] = {}
    for item in _owned_inventory().values():
        name = item.path.name.casefold()
        if name not in required:
            continue
        if name in tools:
            raise RuntimeError(f"Ambiguous RECORD-owned compiler asset: {name}")
        _verify_owned(item)
        tools[name] = item
    if set(tools) != set(required):
        raise RuntimeError(
            f"Missing RECORD-owned compiler assets: {sorted(set(required) - tools.keys())}"
        )
    root = tools["ptxas.exe"].path.parent.parent
    if (
        tools["nvlink.exe"].path != root / "bin" / "nvlink.exe"
        or tools["libdevice.10.bc"].path
        != root / "nvvm" / "libdevice" / "libdevice.10.bc"
    ):
        raise RuntimeError(
            "Owned compiler assets do not share the required CUDA data root"
        )
    directory = str(root / "bin")
    path = os.environ.get("PATH", "").split(os.pathsep)
    if directory.casefold() not in {item.casefold() for item in path}:
        os.environ["PATH"] = os.pathsep.join([directory, *path])
    return str(root)


def prepare_kernels() -> None:
    """Early canonical kernel-package boundary, before importing any CUDA .pyd."""
    global _KERNELS_READY
    with _SESSION_LOCK:
        if _KERNELS_READY:
            return
        record = installation_record()
        inventory = _owned_inventory()
        # cuDNN and NVRTC dispatch some modules by bare basename at runtime.
        # Admit the entire owned family now, never a best-effort partial family.
        families = [
            item.path
            for item in inventory.values()
            if item.path.suffix.lower() == ".dll"
            and item.path.name.casefold().startswith(("cudnn", "nvrtc"))
        ]
        for path in sorted(families):
            load_libraries(path)
        for asset in record.assets:
            if asset.role == "kernel":
                load_libraries(asset.path)
        _KERNELS_READY = True
