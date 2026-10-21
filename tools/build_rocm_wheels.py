#!/usr/bin/env python3
"""Repackage locked AMD payloads into wheel-only, owned ROCm10 distributions.

Only build-time Python stdlib and readelf are required. No SDK installation,
compiler execution, accelerator access, or runtime shell utility is involved.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any, NotRequired, Protocol, TypedDict


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "packaging/rocm/sources.lock.json"
ARCHITECTURES = ("gfx1250", "gfx942", "gfx950")
COMPONENTS = ("core", "libraries", *(f"device_{arch}" for arch in ARCHITECTURES))
COMPONENT_PAYLOAD_ROOTS = {
    "core": ("_rocm_sdk_core", "rocm_sdk_core"),
    "libraries": ("_rocm_sdk_libraries", "rocm_sdk_libraries"),
    **{f"device_{arch}": ("_rocm_sdk_libraries",) for arch in ARCHITECTURES},
}
TAG = "py3-none-linux_x86_64"
# These are the host ABI, not substitutes for any ROCm userspace dependency.
SYSTEM_LIBRARIES = frozenset(
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
BLOCK_SIZE = 8 * 1024 * 1024


class ByteReader(Protocol):
    """The read-only byte stream operation used by archive copying."""

    def read(self, size: int = -1, /) -> bytes: ...


class PayloadPath(TypedDict):
    path: str


class PayloadEntry(PayloadPath):
    sha256: str
    size: int
    mode: int


class OwnedPayloadEntry(PayloadEntry):
    owner: str


class SharedPayloadReference(TypedDict):
    path: str
    owner: str
    sha256: str
    mode: int


class SourceIdentity(TypedDict):
    url: str
    sha256: str


class SourceArchive(SourceIdentity):
    filename: str
    size: NotRequired[int]


class ElfDynamic(TypedDict):
    soname: str | None
    needed: list[str]


class NativeObject(ElfDynamic):
    path: str
    sha256: str


class PluginObject(NativeObject):
    wheel: str


class ComponentRecord(TypedDict):
    distribution: str
    version: str
    module: str
    upstream: SourceIdentity
    payload_sha256: str
    deduplicated: list[SharedPayloadReference]


class LoaderPlan(TypedDict):
    preload: list[str]
    libraries: list[NativeObject]
    compiler: dict[str, str]
    kpack: dict[str, list[str]]
    system_libraries: list[str]
    plugin_objects: list[PluginObject]


class RuntimeManifest(TypedDict):
    sdk_version: str
    architectures: list[str]
    components: dict[str, ComponentRecord]
    loader: LoaderPlan
    source_manifest: str
    third_party_identity: str
    packaging_identity: str


def stream_sha256(stream: ByteReader) -> str:
    digest = hashlib.sha256()
    while block := stream.read(BLOCK_SIZE):
        digest.update(block)
    return digest.hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode()


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_name(name: str) -> None:
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or "\0" in name
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in name.rstrip("/").split("/"))
        or ":" in path.parts[0]
    ):
        raise ValueError(f"unsafe archive path: {name!r}")


def checked_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    result = []
    seen = set()
    for info in archive.infolist():
        safe_name(info.filename)
        if info.filename in seen:
            raise ValueError(f"duplicate archive member: {info.filename}")
        seen.add(info.filename)
        kind = stat.S_IFMT(info.external_attr >> 16)
        if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError(f"unsupported archive member type: {info.filename}")
        if info.flag_bits & 1:
            raise ValueError("encrypted archive member")
        if not info.is_dir():
            result.append(info)
    return sorted(result, key=lambda item: item.filename)


def fetch(source: SourceArchive, directory: Path, *, download: bool) -> Path:
    filename = source["filename"]
    safe_name(filename)
    if "/" in filename:
        raise ValueError("source filename must be a basename")
    path = directory / filename
    if not path.exists():
        if not download:
            raise ValueError(f"missing locked source: {path}; use --download")
        if not source["url"].startswith("https://"):
            raise ValueError("source URL must use HTTPS")
        directory.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".partial")
        try:
            with (
                urllib.request.urlopen(source["url"]) as remote,
                temporary.open("wb") as output,
            ):
                shutil.copyfileobj(remote, output, BLOCK_SIZE)
            if file_sha256(temporary) != source["sha256"]:
                raise ValueError(f"source hash mismatch: {filename}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if file_sha256(path) != source["sha256"]:
        raise ValueError(f"source hash mismatch: {filename}")
    return path


def elf_metadata(path: Path) -> ElfDynamic:
    # Same build-time readelf inspection convention as build_cuda_wheels.py;
    # importing the frontend ELF substrate would import JAX while packaging.
    result = subprocess.run(
        ["readelf", "--file-header", "--dynamic", "--wide", str(path)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "LC_ALL": "C"},
    ).stdout
    if not re.search(r"Machine:\s+Advanced Micro Devices X86-64", result):
        raise ValueError(f"native object is not ELF x86_64: {path}")
    sonames = re.findall(r"\(SONAME\).*?\[([^\]]+)\]", result)
    return {
        "soname": sonames[0] if sonames else None,
        "needed": re.findall(r"\(NEEDED\).*?\[([^\]]+)\]", result),
    }


def native_member(name: str) -> bool:
    # GPU code objects are ELF too, but must never be host-dlopen'ed.
    return (
        ".so" in PurePosixPath(name).name
        and ".hsaco" not in name
        and not name.endswith(".co")
    )


def wheel_info(name: str, mode: int = 0o644) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | mode) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


class WheelWriter:
    def __init__(self, path: Path) -> None:
        self.archive = zipfile.ZipFile(
            path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
        )
        self.rows: list[tuple[str, str, str]] = []

    def add(
        self,
        name: str,
        stream: ByteReader,
        mode: int = 0o644,
        *,
        capture: Path | None = None,
    ) -> PayloadEntry:
        digest = hashlib.sha256()
        size = 0
        with self.archive.open(wheel_info(name, mode), "w", force_zip64=True) as output:
            cached = capture.open("wb") if capture else None
            try:
                while block := stream.read(BLOCK_SIZE):
                    digest.update(block)
                    size += len(block)
                    output.write(block)
                    if cached is not None:
                        cached.write(block)
            finally:
                if cached is not None:
                    cached.close()
        encoded = base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode()
        self.rows.append((name, "sha256=" + encoded, str(size)))
        return {"path": name, "sha256": digest.hexdigest(), "size": size, "mode": mode}

    def text(self, name: str, data: bytes) -> None:
        self.add(name, io.BytesIO(data))

    def finish(self, dist_info: str) -> None:
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerows(sorted(self.rows))
        writer.writerow((dist_info + "/RECORD", "", ""))
        self.archive.writestr(wheel_info(dist_info + "/RECORD"), buffer.getvalue())
        self.archive.close()


def metadata(
    distribution: str,
    version: str,
    requires_python: str,
    dependencies: Sequence[str] = (),
) -> bytes:
    lines = [
        "Metadata-Version: 2.4",
        f"Name: {distribution}",
        f"Version: {version}",
        "Summary: Phydrax packaging of unchanged AMD ROCm userspace payloads",
        f"Requires-Python: {requires_python}",
        "Platform: Linux x86_64",
        "License-File: licenses/NOTICE.txt",
        "License-File: licenses/PHYDRAX-LICENSE.txt",
    ]
    lines.extend(f"Requires-Dist: {dependency}" for dependency in dependencies)
    return (
        "\n".join(lines)
        + "\n\nThird-party payload identity and licenses remain AMD/upstream; "
        "Phydrax supplies packaging and loader metadata, not a replacement SDK.\n"
    ).encode()


def add_metadata(
    writer: WheelWriter,
    distribution: str,
    lock: dict[str, Any],
    dependencies: Sequence[str] = (),
) -> str:
    dist_info = distribution.replace("-", "_") + "-" + lock["sdk_version"] + ".dist-info"
    writer.text(
        dist_info + "/METADATA",
        metadata(
            distribution, lock["sdk_version"], lock["requires_python"], dependencies
        ),
    )
    writer.text(
        dist_info + "/WHEEL",
        f"Wheel-Version: 1.0\nGenerator: phydrax-rocm-repack\nRoot-Is-Purelib: false\nTag: {TAG}\n".encode(),
    )
    writer.text(
        dist_info + "/licenses/NOTICE.txt",
        (ROOT / "packaging/rocm/NOTICE.txt").read_bytes(),
    )
    writer.text(
        dist_info + "/licenses/PHYDRAX-LICENSE.txt",
        (ROOT / "LICENSE").read_bytes(),
    )
    legal_root = ROOT / "packaging/rocm"
    for path in sorted((legal_root / "licenses").rglob("*")):
        if path.is_file():
            writer.text(
                dist_info
                + "/licenses/upstream/"
                + path.relative_to(legal_root / "licenses").as_posix(),
                path.read_bytes(),
            )
    for filename in (
        "license_sources.json",
        "source-notice-audit.json",
        "architecture-coverage.json",
    ):
        path = legal_root / filename
        if path.exists():
            writer.text(dist_info + "/licenses/" + filename, path.read_bytes())
    return dist_info


def architecture_payloads(paths: set[str], architecture: str) -> list[str]:
    kpacks = sorted(
        path
        for path in paths
        if path.startswith("_rocm_sdk_libraries/.kpack/")
        and path.endswith(f"_{architecture}.kpack")
    )
    code_root = f"_rocm_sdk_libraries/lib/hipblaslt/library/{architecture}/"
    if not kpacks or not any(
        path.startswith(code_root) and path.endswith((".co", ".hsaco")) for path in paths
    ):
        raise ValueError(f"missing architecture payload: {architecture}")
    return kpacks


def repack_component(
    source_path: Path,
    source: SourceArchive,
    component: str,
    lock: dict[str, Any],
    output: Path,
    shared: dict[str, OwnedPayloadEntry],
) -> tuple[ComponentRecord, list[PayloadEntry], list[NativeObject]]:
    if component not in COMPONENTS:
        raise ValueError(f"unsupported ROCm component: {component}")
    distribution = "phydrax-rocm-" + component.replace("_", "-")
    filename = (
        distribution.replace("-", "_") + "-" + lock["sdk_version"] + "-" + TAG + ".whl"
    )
    target = output / filename
    temporary = target.with_suffix(".partial")
    inventory: list[PayloadEntry] = []
    native: list[NativeObject] = []
    deduplicated: list[SharedPayloadReference] = []
    notices: list[PayloadEntry] = []
    original_metadata: list[PayloadEntry] = []
    try:
        with (
            zipfile.ZipFile(source_path) as archive,
            tempfile.TemporaryDirectory(prefix="phydrax-rocm-elf-") as scratch,
        ):
            members = checked_members(archive)
            upstream_dist = {
                item.filename.split("/")[0]
                for item in members
                if item.filename.split("/")[0].endswith(".dist-info")
            }
            if len(upstream_dist) != 1:
                raise ValueError(
                    "upstream wheel must contain exactly one dist-info directory"
                )
            upstream_dist_info = upstream_dist.pop()
            original_record = dict(
                (row[0], row[1:])
                for row in csv.reader(
                    io.StringIO(archive.read(upstream_dist_info + "/RECORD").decode())
                )
            )
            expected_dist = f"rocm_sdk_{component}-{lock['sdk_version']}.dist-info"
            if upstream_dist_info != expected_dist:
                raise ValueError(
                    f"unexpected upstream distribution: {upstream_dist_info}"
                )
            payload_paths = {
                item.filename
                for item in members
                if not item.filename.startswith(upstream_dist_info + "/")
            }
            for name in payload_paths:
                if name.split("/")[0] not in COMPONENT_PAYLOAD_ROOTS[component]:
                    raise ValueError(f"unexpected component payload namespace: {name}")
            if component.startswith("device_"):
                architecture_payloads(payload_paths, component.removeprefix("device_"))
            writer = WheelWriter(temporary)
            try:
                dist_info = add_metadata(writer, distribution, lock)
                for info in members:
                    name = info.filename
                    mode = stat.S_IMODE(info.external_attr >> 16) or 0o644
                    is_metadata = name.startswith(upstream_dist_info + "/")
                    destination = dist_info + "/upstream/" + name if is_metadata else name
                    if name in shared and not is_metadata:
                        with archive.open(info) as stream:
                            digest = stream_sha256(stream)
                        existing = shared[name]
                        if name not in original_record:
                            raise ValueError(
                                f"member missing from upstream RECORD: {name}"
                            )
                        expected_hash, expected_size = original_record[name]
                        actual = (
                            "sha256="
                            + base64.urlsafe_b64encode(bytes.fromhex(digest))
                            .rstrip(b"=")
                            .decode()
                        )
                        if expected_hash != actual or expected_size != str(
                            info.file_size
                        ):
                            raise ValueError(f"upstream RECORD mismatch: {name}")
                        if digest != existing["sha256"] or mode != existing["mode"]:
                            raise ValueError(
                                f"conflicting cross-component payload: {name}"
                            )
                        deduplicated.append(
                            {
                                "path": name,
                                "owner": existing["owner"],
                                "sha256": digest,
                                "mode": mode,
                            }
                        )
                        continue
                    compiler_program = name in {
                        "_rocm_sdk_core/lib/llvm/bin/ld.lld",
                        "_rocm_sdk_core/lib/llvm/bin/lld",
                    }
                    cached = (
                        Path(scratch) / "native"
                        if not is_metadata and (native_member(name) or compiler_program)
                        else None
                    )
                    with archive.open(info) as stream:
                        entry = writer.add(destination, stream, mode, capture=cached)
                    if name not in original_record:
                        raise ValueError(f"member missing from upstream RECORD: {name}")
                    expected_hash, expected_size = original_record[name]
                    if expected_hash:
                        actual = (
                            "sha256="
                            + base64.urlsafe_b64encode(bytes.fromhex(entry["sha256"]))
                            .rstrip(b"=")
                            .decode()
                        )
                        if expected_hash != actual or expected_size != str(entry["size"]):
                            raise ValueError(f"upstream RECORD mismatch: {name}")
                    elif not name.endswith(".dist-info/RECORD"):
                        raise ValueError(f"unhashed upstream payload: {name}")
                    if is_metadata:
                        original_metadata.append(entry)
                    else:
                        inventory.append(entry)
                        shared[name] = {**entry, "owner": distribution}
                        if cached is not None:
                            with cached.open("rb") as stream:
                                header = stream.read(20)
                            # Some .so-like names are linker scripts, not loadable objects.
                            if header[:4] == b"\x7fELF" and header[18:20] == b">\0":
                                native.append(
                                    {
                                        "path": name,
                                        **elf_metadata(cached),
                                        "sha256": entry["sha256"],
                                    }
                                )
                            cached.unlink()
                    if re.search(
                        r"license|copying|copyright|notice", name, re.IGNORECASE
                    ):
                        notices.append(entry)
                payload = canonical(inventory)
                writer.text(dist_info + "/phydrax-payload.json", payload)
                provenance = {
                    "distribution": distribution,
                    "version": lock["sdk_version"],
                    "upstream": source,
                    "deduplicated": deduplicated,
                    "original_metadata": original_metadata,
                    "notices": notices,
                }
                writer.text(
                    dist_info + "/upstream-provenance.json", canonical(provenance)
                )
                writer.finish(dist_info)
            except BaseException:
                writer.archive.close()
                raise
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    component_data: ComponentRecord = {
        "distribution": distribution,
        "version": lock["sdk_version"],
        "module": "_rocm_sdk_core" if component == "core" else "_rocm_sdk_libraries",
        "upstream": {"url": source["url"], "sha256": source["sha256"]},
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "deduplicated": deduplicated,
    }
    return component_data, inventory, native


def dependency_order(
    libraries: list[NativeObject], required: list[str]
) -> list[NativeObject]:
    by_soname: dict[str, NativeObject] = {}
    for item in libraries:
        soname = item["soname"]
        if not soname:
            continue
        previous = by_soname.get(soname)
        if previous and previous["sha256"] != item["sha256"]:
            raise ValueError(f"conflicting owned SONAME: {soname}")
        if previous is None or item["path"] < previous["path"]:
            by_soname[soname] = item
    ordered: list[NativeObject] = []
    active: set[str] = set()
    complete: set[str] = set()

    def visit(soname: str) -> None:
        if soname in SYSTEM_LIBRARIES or soname in complete:
            return
        if soname in active:
            raise ValueError(
                f"cyclic native dependency cannot be preloaded safely: {soname}"
            )
        if soname not in by_soname:
            raise ValueError(f"missing owned native dependency: {soname}")
        active.add(soname)
        item = by_soname[soname]
        for dependency in item["needed"]:
            visit(dependency)
        active.remove(soname)
        complete.add(soname)
        ordered.append(item)

    for soname in sorted(set(required)):
        visit(soname)
    return ordered


def plugin_dependencies(paths: list[Path]) -> tuple[list[str], list[PluginObject]]:
    needed: list[str] = []
    objects: list[PluginObject] = []
    with tempfile.TemporaryDirectory(prefix="phydrax-jax-elf-") as scratch:
        for path in paths:
            with zipfile.ZipFile(path) as archive:
                for info in checked_members(archive):
                    if not native_member(info.filename):
                        continue
                    temporary = Path(scratch) / "plugin.so"
                    with archive.open(info) as source, temporary.open("wb") as target:
                        shutil.copyfileobj(source, target, BLOCK_SIZE)
                    dynamic = elf_metadata(temporary)
                    needed.extend(dynamic["needed"])
                    objects.append(
                        {
                            "wheel": path.name,
                            "path": info.filename,
                            "sha256": file_sha256(temporary),
                            **dynamic,
                        }
                    )
                    temporary.unlink()
    if not objects:
        raise ValueError("no native JAX plugin prerequisites found")
    return needed, objects


def runtime_manifest(
    lock: dict[str, Any],
    components: dict[str, ComponentRecord],
    inventory: Sequence[PayloadPath],
    libraries: list[NativeObject],
    required: list[str],
    plugin_objects: list[PluginObject],
) -> RuntimeManifest:
    if set(components) != set(COMPONENTS):
        raise ValueError("all three architectures and all five components are required")
    if lock["architectures"] != list(ARCHITECTURES):
        raise ValueError("the complete sorted three-architecture coverage is required")
    paths = {item["path"] for item in inventory}
    kpack = {
        architecture: architecture_payloads(paths, architecture)
        for architecture in ARCHITECTURES
    }
    compiler = {
        "rocm_path": "_rocm_sdk_core",
        "llvm_path": "_rocm_sdk_core/lib/llvm",
        "lld_path": "_rocm_sdk_core/lib/llvm/bin/ld.lld",
        "bitcode_path": "_rocm_sdk_core/lib/llvm/amdgcn/bitcode",
    }
    if compiler["lld_path"] not in paths:
        raise ValueError("missing owned compiler linker")
    if not any(
        path.startswith(compiler["bitcode_path"] + "/") and path.endswith(".bc")
        for path in paths
    ):
        raise ValueError("missing owned compiler bitcode")
    compiler_needed = [
        dependency
        for item in libraries
        if item["path"] in {compiler["lld_path"], compiler["llvm_path"] + "/bin/lld"}
        for dependency in item["needed"]
    ]
    selected = dependency_order(
        libraries, required + compiler_needed + lock["runtime_roots"]
    )
    return {
        "sdk_version": lock["sdk_version"],
        "architectures": list(ARCHITECTURES),
        "components": components,
        "loader": {
            "preload": [item["path"] for item in selected],
            "libraries": selected,
            "compiler": compiler,
            "kpack": kpack,
            "system_libraries": sorted(SYSTEM_LIBRARIES),
            "plugin_objects": plugin_objects,
        },
        "source_manifest": "_rocm_sdk_core/share/therock/therock_manifest.json",
        "third_party_identity": "AMD ROCm SDK " + lock["sdk_version"],
        "packaging_identity": "Phydrax owned ROCm runtime packaging",
    }


def write_runtime_wheel(
    lock: dict[str, Any], manifest: RuntimeManifest, output: Path
) -> Path:
    """Publish the loader and admitted component plan as one atomic wheel."""
    distribution = "phydrax-rocm-runtime"
    target = output / f"phydrax_rocm_runtime-{lock['sdk_version']}-{TAG}.whl"
    temporary = target.with_suffix(".partial")
    writer = WheelWriter(temporary)
    try:
        dependencies = [
            f"{item['distribution']}=={item['version']}"
            for item in manifest["components"].values()
        ]
        dist_info = add_metadata(writer, distribution, lock, dependencies)
        writer.text(
            "phydrax_rocm_runtime/__init__.py",
            b'"""Immutable metadata for the Phydrax-owned AMD userspace bundle."""\n',
        )
        writer.text(
            "phydrax_rocm_runtime/_loader.py",
            (ROOT / "packaging/rocm/python/phydrax_rocm_runtime/_loader.py").read_bytes(),
        )
        writer.text(
            "phydrax_rocm_runtime.pth",
            b"import phydrax_rocm_runtime._loader; phydrax_rocm_runtime._loader.install_rocm_loader()\n",
        )
        writer.text("phydrax_rocm_runtime/runtime.json", canonical(manifest))
        writer.text(dist_info + "/sources.lock.json", canonical(lock))
        writer.text(
            dist_info + "/source-notice-audit.json",
            (ROOT / "packaging/rocm/source-notice-audit.json").read_bytes(),
        )
        writer.finish(dist_info)
        temporary.replace(target)
    except BaseException:
        writer.archive.close()
        temporary.unlink(missing_ok=True)
        raise
    return target


def build(
    lock: dict[str, Any], sources: Path, output: Path, *, download: bool = False
) -> RuntimeManifest:
    if (
        lock["sdk_version"] != "10.1.0"
        or set(lock["components"]) != set(COMPONENTS)
        or lock["architectures"] != list(ARCHITECTURES)
    ):
        raise ValueError("only the complete pinned ROCm10 SDK 10.1.0 is supported")
    if not (ROOT / "packaging/rocm/license_sources.json").is_file():
        raise ValueError("missing supplemental license/source availability inventory")
    output.mkdir(parents=True, exist_ok=True)
    source_paths = {
        name: fetch(lock["components"][name], sources, download=download)
        for name in COMPONENTS
    }
    plugin_paths = [
        fetch(source, sources, download=download) for source in lock["plugins"]
    ]
    required, plugin_objects = plugin_dependencies(plugin_paths)
    components: dict[str, ComponentRecord] = {}
    inventory: list[PayloadEntry] = []
    libraries: list[NativeObject] = []
    shared: dict[str, OwnedPayloadEntry] = {}
    for name in COMPONENTS:
        component, entries, native = repack_component(
            source_paths[name], lock["components"][name], name, lock, output, shared
        )
        components[name] = component
        inventory.extend(entries)
        libraries.extend(native)
    manifest = runtime_manifest(
        lock, components, inventory, libraries, required, plugin_objects
    )
    write_runtime_wheel(lock, manifest, output)
    (output / "runtime.json").write_bytes(json.dumps(manifest, indent=2).encode() + b"\n")
    artifacts = []
    for name in (*COMPONENTS, "runtime"):
        path = output / f"phydrax_rocm_{name}-{lock['sdk_version']}-{TAG}.whl"
        artifacts.append(
            {
                "filename": path.name,
                "sha256": file_sha256(path),
                "size": path.stat().st_size,
                "exceeds_default_pypi_file_limit": path.stat().st_size
                > 100 * 1024 * 1024,
            }
        )
    report = {
        "sdk_version": lock["sdk_version"],
        "artifacts": artifacts,
        "sources": lock,
        "publication_performed": False,
        "publication_prerequisites": [
            "Applicable upstream license/source obligations review",
            "PyPI project-specific large-file quotas",
        ],
    }
    (output / "build-report.json").write_bytes(
        json.dumps(report, indent=2).encode() + b"\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--sources", type=Path, default=ROOT / ".tmp/rocm-upstream")
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument(
        "--download", action="store_true", help="fetch missing SHA256-locked archives"
    )
    args = parser.parse_args()
    build(
        json.loads(args.lock.read_text()),
        args.sources,
        args.wheelhouse,
        download=args.download,
    )


if __name__ == "__main__":
    main()
