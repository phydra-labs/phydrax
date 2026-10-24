"""Assemble an isolated Windows CUDA toolkit from verified wheel RECORDs."""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


SUPPORT = {
    "cuda_cudart": (
        "13.0.96",
        "a2ed875f9997aa24904fb70cc9db3acd9308433cde99bc8e63ec1271c9da31b4",
    ),
    "cuda_cuobjdump": (
        "13.0.85",
        "cddbc524dc229c708cc989e4ff2d636ba342c695e0f60c1026a1b5f76de4b77b",
    ),
}
TOOL_NAMES = (
    "PHX_WINDOWS_CC",
    "PHX_WINDOWS_CXX",
    "PHX_WINDOWS_CUDA_HOST_COMPILER",
    "PHX_WINDOWS_NVCC",
    "PHX_WINDOWS_LIB",
    "PHX_PATCH_EXECUTABLE",
    "PHX_WINDOWS_CMAKE",
    "PHX_WINDOWS_NINJA",
)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy_verified(source: Path, destination: Path, sha256: str) -> None:
    if destination.exists():
        if digest(destination) != sha256:
            raise RuntimeError(
                f"Conflicting/stale toolkit file: {destination}; use a fresh toolkit view"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if digest(destination) != sha256:
        raise RuntimeError(f"Toolkit copy changed during preparation: {source}")


def record_relative(parts: tuple[str, ...]) -> Path | None:
    for kind in ("nvvm", "include", "bin", "lib", "lib64"):
        if kind not in parts:
            continue
        suffix = parts[parts.index(kind) + 1 :]
        if not suffix or "__pycache__" in suffix or suffix[-1].endswith((".py", ".pyc")):
            return None
        if kind in ("lib", "lib64"):
            if suffix[0] == "x64":
                suffix = suffix[1:]
            if suffix[0] == "cmake":
                return Path("lib", *suffix)
            return Path("lib", "x64", *suffix)
        return Path(kind, *suffix)
    return None


def wheel_inventory(root: Path) -> dict[str, Any]:
    wheels: dict[str, str] = {}
    libraries: dict[str, str] = {}
    records: list[dict[str, str]] = []
    for distribution in sorted(
        metadata.distributions(), key=lambda item: item.metadata["Name"]
    ):
        name = distribution.metadata["Name"].lower().replace("_", "-")
        if not (name.startswith("nvidia-") or name == "cutensor-cu13"):
            continue
        major = re.search(r"-cu(\d+)$", name)
        if major and major.group(1) != "13":
            raise RuntimeError(f"Foreign CUDA distribution: {name}")
        selected = False
        for record in distribution.files or ():
            parts = record.parts
            if not parts or parts[0] not in ("nvidia", "cutensor"):
                continue
            if any(re.fullmatch(r"cu\d+", part) and part != "cu13" for part in parts):
                raise RuntimeError(f"Foreign CUDA RECORD: {name}: {record}")
            relative = record_relative(parts)
            if relative is None:
                continue
            source = Path(distribution.locate_file(record)).resolve()
            if (
                not source.is_file()
                or record.hash is None
                or record.hash.mode != "sha256"
            ):
                raise RuntimeError(f"Missing file or SHA256 RECORD: {name}: {record}")
            sha256 = digest(source)
            encoded = (
                base64.urlsafe_b64encode(bytes.fromhex(sha256))
                .rstrip(b"=")
                .decode("ascii")
            )
            if encoded != record.hash.value or (
                record.size is not None and source.stat().st_size != record.size
            ):
                raise RuntimeError(f"Wheel RECORD mismatch: {name}: {record}")
            if source.suffix.lower() == ".dll":
                # NVVM's compiler-private layout must remain intact.
                # Runtime DLLs may appear under lib rather than bin in vendor wheels.
                if relative.parts[0] != "nvvm":
                    relative = Path("bin", source.name)
                key = source.name.lower()
                if key in libraries and libraries[key] != str(source):
                    raise RuntimeError(f"Ambiguous DLL ownership: {key}")
                libraries[key] = str(source)
            copy_verified(source, root / relative, sha256)
            records.append(
                {
                    "distribution": name,
                    "path": str(source),
                    "sha256": sha256,
                    "view": relative.as_posix(),
                }
            )
            selected = True
        if selected:
            if name in wheels:
                raise RuntimeError(f"Duplicate installed distribution: {name}")
            wheels[name] = distribution.version
    if not wheels:
        raise RuntimeError("No RECORD-owned Windows CUDA toolkit inputs found")
    return {
        "host_architecture": platform.machine(),
        "wheels": wheels,
        "libraries": libraries,
        "records": records,
        "runpath": [],
    }


def support_files(
    root: Path, downloads: Path, inventory: dict[str, Any]
) -> dict[str, Any]:
    pins: dict[str, Any] = {}
    downloads.mkdir(parents=True, exist_ok=True)
    for family, (version, sha256) in SUPPORT.items():
        if family == "cuda_cudart" and any(
            record["view"] == "lib/x64/cuda.lib" for record in inventory["records"]
        ):
            # Prefer the authenticated wheel's driver import library. The
            # support archive is only a fallback for wheels that omit it.
            continue
        stem = f"{family}-windows-x86_64-{version}-archive"
        url = f"https://developer.download.nvidia.com/compute/cuda/redist/{family}/windows-x86_64/{stem}.zip"
        archive = downloads / f"{stem}.zip"
        if not archive.exists():
            temporary = archive.with_suffix(".download")
            with urllib.request.urlopen(url) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
            if digest(temporary) != sha256:
                raise RuntimeError(f"NVIDIA support archive SHA256 mismatch: {url}")
            temporary.replace(archive)
        if digest(archive) != sha256:
            raise RuntimeError(f"NVIDIA support archive SHA256 mismatch: {archive}")
        selected = 0
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                path = PurePosixPath(member.filename)
                if member.is_dir() or path.is_absolute() or ".." in path.parts:
                    continue
                if family == "cuda_cudart":
                    if path.name.lower() != "cuda.lib":
                        continue
                    relative = Path("lib", "x64", "cuda.lib")
                else:
                    if "bin" not in path.parts:
                        continue
                    relative = Path("bin", *path.parts[path.parts.index("bin") + 1 :])
                data = bundle.read(member)
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() and destination.read_bytes() != data:
                    raise RuntimeError(f"Conflicting support input: {destination}")
                destination.write_bytes(data)
                selected += 1
        if not selected:
            raise RuntimeError(f"Missing required support inputs in {archive}")
        pins[family] = {"url": url, "sha256": sha256, "version": version}
    return pins


def classify_link_library(
    librarian: Path, library: Path, dll_name: str, family: str
) -> dict[str, Any]:
    members = subprocess.run(
        [str(librarian), "/nologo", "/list", str(library)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    imported_names = {
        line.strip().lower()
        for line in members.splitlines()
        if line.strip().lower().endswith(".dll")
    }
    kind = "dll-import"
    target = dll_name.lower()
    dispatch_symbols: list[str] = []
    if imported_names != {target}:
        loaders = {
            "cuda": ("driver-loader", "nvcuda.dll", "cudaDriverLoader_initializeLoader"),
            "cudart": (
                "driver-hybrid",
                "nvcudart_hybrid64.dll",
                "initializeHybridRuntimeLoaderOnce",
            ),
        }
        if imported_names or family not in loaders:
            raise RuntimeError(f"Unrecognized vendor link library: {library}: {members}")
        kind, target, entrypoint = loaders[family]
        data = library.read_bytes()
        dispatch_symbols = [entrypoint, "cuGetProcAddress_v2"]
        if family == "cudart":
            dispatch_symbols.append("cuGetExportTable")
        if not data.startswith(b"!<arch>\n") or any(
            value.encode("ascii") + b"\0" not in data
            for value in [target, "nvcuda.dll", *dispatch_symbols]
        ):
            raise RuntimeError(f"Missing NVIDIA loader ABI in {library}")
    return {
        "path": str(library),
        "sha256": digest(library),
        "kind": kind,
        "target_dll": target,
        "archive_members": members.splitlines(),
        "dispatch_symbols": dispatch_symbols,
    }


def prepare_link_libraries(
    root: Path, inventory: dict[str, Any], librarian: Path, components: set[str]
) -> dict[str, dict[str, Any]]:
    parser_path = Path(__file__).resolve().parents[3] / "phydrax" / "_pe_dependencies.py"
    spec = importlib.util.spec_from_file_location("_phx_build_pe", parser_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import canonical PE inspector: {parser_path}")
    pe = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pe
    spec.loader.exec_module(pe)
    families = {"cudart"}
    if "cudss" in components:
        families.add("cudss")
    if components & {"amgx", "quda"}:
        families.update(("cublas", "cusparse", "cusolver", "curand", "nvJitLink"))
    if "cutensor" in components:
        families.add("cutensor")
    result: dict[str, dict[str, Any]] = {}
    for family in sorted(families):
        matches = [
            Path(path)
            for name, path in inventory["libraries"].items()
            if re.fullmatch(
                re.escape(family) + r"(?:64)?(?:[_-]\d+)*\.dll", name, re.IGNORECASE
            )
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one RECORD-owned {family} DLL, found {matches}"
            )
        dll = matches[0]
        image = pe.inspect_pe(dll)
        library = root / "lib" / "x64" / f"{family}.lib"
        if not library.exists():
            exports = []
            for item in image.exports:
                if not item.rva:
                    continue
                if item.name is None or not re.fullmatch(r"[^\s\";=]+", item.name):
                    raise RuntimeError(
                        f"Cannot safely generate an import library for export {item!r} in {dll}"
                    )
                exports.append(f"  {item.name} @{item.ordinal}")
            if not exports:
                raise RuntimeError(f"No real exports in {dll}")
            definition = root / "import-defs" / f"{family}.def"
            definition.parent.mkdir(parents=True, exist_ok=True)
            definition.write_text(
                f'LIBRARY "{dll.name}"\nEXPORTS\n' + "\n".join(exports) + "\n",
                encoding="utf-8",
            )
            subprocess.run(
                [
                    str(librarian),
                    "/nologo",
                    "/machine:x64",
                    f"/def:{definition}",
                    f"/out:{library}",
                ],
                check=True,
            )
        result[family] = classify_link_library(librarian, library, dll.name, family)
        # Optional providers use the common lib path; FindCUDAToolkit uses lib/x64.
        copy_verified(library, root / "lib" / library.name, digest(library))
        inventory.setdefault("component_dlls", {})[family] = str(root / "bin" / dll.name)
    if "quda" in components:
        library = root / "lib/x64/nvml.lib"
        result["nvml"] = classify_link_library(librarian, library, "nvml.dll", "nvml")
        copy_verified(library, root / "lib/nvml.lib", digest(library))
    result["cuda"] = classify_link_library(
        librarian, root / "lib/x64/cuda.lib", "nvcuda.dll", "cuda"
    )
    return result


def main() -> None:
    if sys.platform != "win32" or platform.machine().lower() not in ("amd64", "x86_64"):
        raise RuntimeError("Windows toolkit preparation requires native Windows x64")
    root, downloads = (Path(value).resolve() for value in sys.argv[1:3])
    profile = Path(os.environ["USERPROFILE"]).resolve()
    if not root.is_relative_to(profile) or not downloads.is_relative_to(profile):
        raise RuntimeError(
            "Windows toolkit and download directories must stay under USERPROFILE"
        )
    toolchain: dict[str, Any] = {}
    for variable in TOOL_NAMES:
        path = Path(os.environ[variable])
        if not path.is_absolute() or not path.is_file():
            raise RuntimeError(f"{variable} must name an explicit existing executable")
        toolchain[variable] = {"path": str(path.resolve()), "sha256": digest(path)}
    toolchain["sdk"] = {
        "root": os.environ["PHX_WINDOWS_SDK_ROOT"],
        "version": os.environ["PHX_WINDOWS_SDK_VERSION"],
        "msvc_root": os.environ["PHX_WINDOWS_MSVC_ROOT"],
    }
    msvc = Path(os.environ["PHX_WINDOWS_MSVC_ROOT"]).resolve()
    auxiliary = msvc.parents[2] / "Auxiliary/Build"
    if not (auxiliary / "vcvarsall.bat").is_file():
        raise RuntimeError(f"Missing real MSVC Auxiliary/Build layout: {auxiliary}")
    for path in sorted(auxiliary.rglob("*")):
        if not path.is_file():
            continue
        if not path.resolve().is_relative_to(profile):
            raise RuntimeError(f"MSVC auxiliary input escapes USERPROFILE: {path}")
        toolchain[f"msvc_auxiliary/{path.relative_to(auxiliary).as_posix()}"] = {
            "path": str(path),
            "sha256": digest(path),
        }
    inventory = wheel_inventory(root)
    nvcc = Path(os.environ["PHX_WINDOWS_NVCC"]).resolve()
    if not any(
        Path(row["path"]) == nvcc and Path(row["view"]).name.lower() == "nvcc.exe"
        for row in inventory["records"]
    ):
        raise RuntimeError(
            "PHX_WINDOWS_NVCC must name the installed wheel's RECORD-owned nvcc.exe"
        )
    inventory["upstream_pins"] = support_files(root, downloads, inventory)
    path = root / "bin/cuobjdump.exe"
    toolchain["cuobjdump"] = {"path": str(path), "sha256": digest(path)}
    inventory["toolchain"] = toolchain
    inventory["link_libraries"] = prepare_link_libraries(
        root, inventory, Path(os.environ["PHX_WINDOWS_LIB"]), set(sys.argv[3].split(";"))
    )
    for family, record in inventory["link_libraries"].items():
        toolchain[f"{family}_link_library"] = record
    runtime = inventory["link_libraries"]["cudart"]
    inventory["runtime_linkage"] = {
        "kind": runtime["kind"],
        "target_dll": runtime["target_dll"],
        "owner": "display-driver" if runtime["kind"] == "driver-hybrid" else "nvidia-cuda-runtime",
    }
    inventory["executables"] = {
        "nvcc": str(root / "bin" / "nvcc.exe"),
        "cuobjdump": str(root / "bin" / "cuobjdump.exe"),
    }
    (root / "discovery.json").write_text(
        json.dumps(inventory, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
