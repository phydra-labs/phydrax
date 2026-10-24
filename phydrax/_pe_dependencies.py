# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Standalone, non-loading PE32+ x86-64 import/export inspection.

This file is copied byte-for-byte into the Windows PJRT runtime wheel. It must
not import Phydrax, NumPy, JAX, or an optional binary inspection package.
"""

from __future__ import annotations

import hashlib
import mmap
import os
import re
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple


class PeFormatError(ValueError):
    """An unsupported, ambiguous, malformed, or truncated PE image."""


class PeExport(NamedTuple):
    name: str | None
    ordinal: int
    rva: int
    forwarder: str | None


class PeImage(NamedTuple):
    machine: int
    imports: tuple[str, ...]
    delay_imports: tuple[str, ...]
    exports: tuple[PeExport, ...]


# The classifier only identifies OS/driver ownership, not installed availability.
_SYSTEM_IMPORTS = frozenset(
    "advapi32.dll bcrypt.dll bcryptprimitives.dll cfgmgr32.dll comctl32.dll "
    "comdlg32.dll crypt32.dll cryptbase.dll dbghelp.dll dwrite.dll dxcore.dll "
    "gdi32.dll imm32.dll iphlpapi.dll kernel32.dll kernelbase.dll mpr.dll "
    "msasn1.dll msvcrt.dll ncrypt.dll netapi32.dll normaliz.dll ntdll.dll "
    "nvcuda.dll nvml.dll ole32.dll oleaut32.dll powrprof.dll psapi.dll rpcrt4.dll "
    "secur32.dll setupapi.dll shell32.dll shlwapi.dll user32.dll userenv.dll "
    "version.dll winhttp.dll wininet.dll winmm.dll wintrust.dll ws2_32.dll "
    "wtsapi32.dll ucrtbase.dll vcruntime140.dll vcruntime140_1.dll "
    "msvcp140.dll msvcp140_1.dll msvcp140_2.dll concrt140.dll".split()
)


def is_system_import(name: str) -> bool:
    """Identify a Windows system/driver DLL basename without locating/loading it."""
    lower = name.casefold()
    if not lower or any(char in lower for char in "/\\:"):
        return False
    return lower in _SYSTEM_IMPORTS or (
        lower.endswith(".dll") and lower.startswith(("api-ms-win-", "ext-ms-win-"))
    )


class _Section(NamedTuple):
    address: int
    virtual_size: int
    offset: int
    raw_size: int


class _Reader:
    def __init__(self, data: mmap.mmap) -> None:
        self.data = data
        self.sections: tuple[_Section, ...] = ()
        self.headers = 0
        self.image_base = 0
        self.directories: tuple[tuple[int, int], ...] = ()
        self.string_bytes = 0

    def unpack(self, format: str, offset: int) -> tuple[int, ...]:
        size = struct.calcsize(format)
        self.check(offset, size)
        return struct.unpack_from(format, self.data, offset)

    def check(self, offset: int, size: int) -> None:
        if offset < 0 or size < 0 or offset > len(self.data) - size:
            raise PeFormatError(f"PE file range outside image: {offset:#x}+{size:#x}")

    def rva(self, address: int, size: int = 1) -> int:
        if address <= 0 or size < 0 or address + size > 0x100000000:
            raise PeFormatError(f"Invalid PE RVA: {address:#x}+{size:#x}")
        matches = []
        if address < self.headers and size <= self.headers - address:
            matches.append(address)
        for section in self.sections:
            delta = address - section.address
            if 0 <= delta < max(section.virtual_size, section.raw_size):
                if delta > section.raw_size or size > section.raw_size - delta:
                    raise PeFormatError("PE RVA refers to unbacked section data")
                matches.append(section.offset + delta)
        if len(matches) != 1:
            raise PeFormatError(f"Unmapped or ambiguous PE RVA: {address:#x}")
        self.check(matches[0], size)
        return matches[0]

    def string(self, address: int, *, dll: bool = False) -> str:
        start = self.rva(address)
        # Bound names independently of file size, and validate their entire RVA span.
        end = self.data.find(b"\0", start, min(start + 65536, len(self.data)))
        if end == -1:
            raise PeFormatError("Unterminated PE string")
        self.rva(address, end - start + 1)
        self.string_bytes += end - start + 1
        if self.string_bytes > 64 * 1024 * 1024:
            raise PeFormatError("PE strings exceed bounded metadata capacity")
        try:
            result = self.data[start:end].decode("ascii")
        except UnicodeDecodeError as error:
            raise PeFormatError("Non-ASCII PE import/export name") from error
        if not result or (dll and any(char in result for char in "/\\:")):
            raise PeFormatError("Invalid PE import/export name")
        return result

    def header(self) -> int:
        self.check(0, 64)
        if self.data[:2] != b"MZ":
            raise PeFormatError("Missing DOS MZ signature")
        (pe,) = self.unpack("<I", 60)
        self.check(pe, 24)
        if pe < 64 or self.data[pe : pe + 4] != b"PE\0\0":
            raise PeFormatError("Missing PE signature")
        machine, count = self.unpack("<HH", pe + 4)
        (optional_size,) = self.unpack("<H", pe + 20)
        optional = pe + 24
        self.check(optional, optional_size)
        if machine != 0x8664:
            raise PeFormatError(f"Expected x86-64 PE machine 0x8664, got {machine:#x}")
        if not 1 <= count <= 96 or optional_size < 112:
            raise PeFormatError("Invalid PE section count or optional header size")
        if self.unpack("<H", optional)[0] != 0x20B:
            raise PeFormatError("Expected PE32+ optional header")
        self.image_base = self.unpack("<Q", optional + 24)[0]
        self.headers = self.unpack("<I", optional + 60)[0]
        directories = self.unpack("<I", optional + 108)[0]
        if directories > 16 or optional_size < 112 + directories * 8:
            raise PeFormatError("Invalid PE data-directory table")
        self.directories = tuple(
            (rva, size)
            for index in range(directories)
            for rva, size in (self.unpack("<II", optional + 112 + index * 8),)
        )
        table = optional + optional_size
        self.check(table, count * 40)
        if self.headers < table + count * 40:
            raise PeFormatError("PE SizeOfHeaders does not cover section table")
        self.check(0, self.headers)
        sections = []
        for index in range(count):
            virtual_size, address, raw_size, offset = self.unpack(
                "<IIII", table + index * 40 + 8
            )
            self.check(offset, raw_size)
            if address + max(virtual_size, raw_size) > 0x100000000:
                raise PeFormatError("PE section RVA overflow")
            sections.append(_Section(address, virtual_size, offset, raw_size))
        self.sections = tuple(sections)
        for index, left in enumerate(sections):
            if left.address < self.headers or (
                left.raw_size and left.offset < self.headers
            ):
                raise PeFormatError("PE section overlaps headers")
            for right in sections[index + 1 :]:
                if max(left.address, right.address) < min(
                    left.address + max(left.virtual_size, left.raw_size),
                    right.address + max(right.virtual_size, right.raw_size),
                ):
                    raise PeFormatError("Overlapping PE virtual sections")
                if (
                    left.raw_size
                    and right.raw_size
                    and max(left.offset, right.offset)
                    < min(left.offset + left.raw_size, right.offset + right.raw_size)
                ):
                    raise PeFormatError("Overlapping PE raw sections")
        return machine

    def directory(self, index: int) -> tuple[int, int]:
        if index >= len(self.directories):
            return 0, 0
        address, size = self.directories[index]
        if bool(address) != bool(size):
            raise PeFormatError("Inconsistent PE data-directory address and size")
        if size:
            self.rva(address, size)
        return address, size

    def imports(self, *, delayed: bool) -> tuple[str, ...]:
        address, size = self.directory(13 if delayed else 1)
        if not size:
            return ()
        width = 32 if delayed else 20
        if size < width:
            raise PeFormatError("Truncated PE import descriptor")
        if size // width > 4096:
            raise PeFormatError("PE import table exceeds bounded descriptor capacity")
        names = set()
        for index in range(size // width):
            fields = self.unpack(
                "<" + "I" * (width // 4), self.rva(address + index * width, width)
            )
            if not any(fields):
                return tuple(sorted(names, key=str.casefold))
            name_rva = fields[1] if delayed else fields[3]
            if delayed:
                if fields[0] not in (0, 1):
                    raise PeFormatError("Unsupported PE delay-import attributes")
                if fields[0] == 0:
                    name_rva -= self.image_base
            names.add(self.string(name_rva, dll=True))
        raise PeFormatError("Unterminated PE import descriptor table")

    def exports(self) -> tuple[PeExport, ...]:
        address, size = self.directory(0)
        if not size:
            return ()
        if size < 40:
            raise PeFormatError("Truncated PE export directory")
        fields = self.unpack("<IIHHIIIIIII", self.rva(address, 40))
        base, count, name_count, functions, names, ordinals = fields[5:]
        if count > 1_000_000 or name_count > 1_000_000:
            raise PeFormatError("Unbounded PE export table")
        if base + count > 0x100000000:
            raise PeFormatError("PE export ordinal overflow")
        function_offset = self.rva(functions, count * 4) if count else 0
        name_offset = self.rva(names, name_count * 4) if name_count else 0
        ordinal_offset = self.rva(ordinals, name_count * 2) if name_count else 0
        by_index: dict[int, list[str]] = {}
        seen_names = set()
        for index in range(name_count):
            ordinal = self.unpack("<H", ordinal_offset + index * 2)[0]
            if ordinal >= count:
                raise PeFormatError("PE export ordinal exceeds address table")
            name = self.string(self.unpack("<I", name_offset + index * 4)[0])
            if name in seen_names:
                raise PeFormatError("Duplicate PE named export")
            seen_names.add(name)
            by_index.setdefault(ordinal, []).append(name)
        result = []
        for index in range(count):
            target = self.unpack("<I", function_offset + index * 4)[0]
            if not target:
                if index in by_index:
                    raise PeFormatError("Named PE export has no address")
                continue
            forwarder = None
            if address <= target < address + size:
                forwarder = self.string(target)
                module, _, symbol = forwarder.rpartition(".")
                if (
                    target + len(forwarder) + 1 > address + size
                    or not module
                    or not symbol
                    or any(char in module for char in "/\\:")
                    or (symbol.startswith("#") and not symbol[1:].isdigit())
                ):
                    raise PeFormatError("Invalid PE export forwarder")
            elif not any(
                section.address
                <= target
                < section.address + max(section.virtual_size, section.raw_size)
                for section in self.sections
            ):
                raise PeFormatError("PE export target lies outside the virtual image")
            for name in by_index.get(index) or (None,):
                result.append(PeExport(name, base + index, target, forwarder))
        return tuple(sorted(result, key=lambda item: (item.ordinal, item.name or "")))


def inspect_pe(path: str | os.PathLike[str]) -> PeImage:
    """Inspect imports/exports without executing code; refuse unsupported images."""
    with Path(path).open("rb") as stream:
        if os.fstat(stream.fileno()).st_size < 64:
            raise PeFormatError("Truncated PE image")
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
            reader = _Reader(data)
            machine = reader.header()
            return PeImage(
                machine,
                reader.imports(delayed=False),
                reader.imports(delayed=True),
                reader.exports(),
            )


def inspect_cuda_runtime_linkage(
    path: str | os.PathLike[str],
    map_path: str | os.PathLike[str],
    linkage: Mapping[str, str],
) -> dict[str, object]:
    """Audit hybrid dispatch bytes and linker members without loading a DLL.

    Callers authenticate the installed binary and map against wheel RECORD (or
    the build's actual outputs). Dispatch strings must reside in mapped PE
    sections, not an appended overlay. This is binary/linker evidence, not a
    claim that a CUDA API or GPU workload has executed.
    """
    expected = {
        "kind": "driver-hybrid",
        "target_dll": "nvcudart_hybrid64.dll",
        "owner": "display-driver",
    }
    if linkage != expected:
        raise PeFormatError("Expected canonical driver-hybrid CUDA runtime linkage")
    binary = Path(path).resolve()
    linker_map = Path(map_path).resolve()
    mapping_bytes = linker_map.read_bytes()
    mapping = mapping_bytes.decode("utf-8")
    if re.search(
        r"\bcudart(?:\.lib)?:cudart_static_loader\.obj\b", mapping, re.IGNORECASE
    ) is None or re.search(r"\bcudart_static(?::|\.lib\b)", mapping, re.IGNORECASE):
        raise PeFormatError(
            f"{binary.name}: missing hybrid loader or static runtime in link map"
        )
    markers = ("nvcudart_hybrid64.dll", "cuGetProcAddress_v2", "cuGetExportTable")
    with binary.open("rb") as stream:
        if os.fstat(stream.fileno()).st_size < 64:
            raise PeFormatError("Truncated PE image")
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
            reader = _Reader(data)
            machine = reader.header()
            imports = reader.imports(delayed=False)
            delayed = reader.imports(delayed=True)
            exports = reader.exports()
            names = set(imports).union(delayed)
            for export in exports:
                if export.forwarder is not None:
                    library, separator, symbol = export.forwarder.rpartition(".")
                    if not separator or not library or not symbol:
                        raise PeFormatError(
                            f"Invalid PE export forwarder: {export.forwarder}"
                        )
                    names.add(
                        library if library.lower().endswith(".dll") else library + ".dll"
                    )
            normalized = set()
            for name in names:
                if (
                    not name
                    or any(char in name for char in "/\\:")
                    or not name.lower().endswith(".dll")
                ):
                    raise PeFormatError(f"Path-bearing or invalid PE import: {name!r}")
                normalized.add(name.lower())
            if any(name.startswith("cudart") for name in normalized):
                raise PeFormatError(
                    f"{binary.name}: mixed standalone and hybrid CUDA runtimes"
                )
            for marker in markers:
                encoded = marker.encode("ascii") + b"\0"
                if not any(
                    data.find(encoded, section.offset, section.offset + section.raw_size)
                    != -1
                    for section in reader.sections
                ):
                    raise PeFormatError(
                        f"{binary.name}: missing mapped hybrid runtime dispatch evidence: {marker}"
                    )
            digest = hashlib.sha256(data).hexdigest()
    return {
        "path": str(binary),
        "sha256": digest,
        "machine": machine,
        "imports": list(imports),
        "delay_imports": list(delayed),
        "closure_imports": sorted(normalized),
        "dispatch_markers": list(markers),
        "link_map_path": str(linker_map),
        "link_map_sha256": hashlib.sha256(mapping_bytes).hexdigest(),
        "runtime_linkage": expected,
        "static_runtime_excluded": True,
    }
