"""Minimal shared objects encoded from the System V ELF64 external format."""

import os
import struct
from pathlib import Path
from typing import Literal


def write_shared_object(
    path: Path,
    *,
    needed: tuple[str, ...] = (),
    soname: str | None = None,
    rpath: tuple[str, ...] = (),
    runpath: tuple[str, ...] = (),
    machine: Literal["x86_64", "aarch64"] = "x86_64",
) -> Path:
    """Write inspectable ELF metadata, not executable code or a loadable fixture."""

    def loader_path(value: str) -> str:
        # An ELF search list uses ':' separators, so a Windows drive cannot be
        # encoded as an absolute entry. Keep fixture references archive-relative.
        if os.name == "nt" and os.path.isabs(value) and os.path.splitdrive(value)[0]:
            relative = os.path.relpath(value, path.parent).replace("\\", "/")
            return "$ORIGIN" if relative == "." else f"$ORIGIN/{relative}"
        return value

    strings = bytearray(b"\0")
    dynamic: list[tuple[int, int]] = []
    entries = [(1, name) for name in needed]
    if soname is not None:
        entries.append((14, soname))
    if rpath:
        entries.append((15, ":".join(loader_path(value) for value in rpath)))
    if runpath:
        entries.append((29, ":".join(loader_path(value) for value in runpath)))
    for tag, value in entries:
        dynamic.append((tag, len(strings)))
        strings.extend(value.encode() + b"\0")
    dynamic_offset = 64 + 2 * 56
    dynamic_size = (len(dynamic) + 3) * 16
    strings_offset = dynamic_offset + dynamic_size
    virtual_base = 0x10000
    dynamic.extend(((5, virtual_base + strings_offset), (10, len(strings)), (0, 0)))
    size = strings_offset + len(strings)
    machine_number = 62 if machine == "x86_64" else 183
    header = struct.pack(
        "<16sHHIQQQIHHHHHH",
        b"\x7fELF\x02\x01\x01" + b"\0" * 9,
        3,
        machine_number,
        1,
        0,
        64,
        0,
        0,
        64,
        56,
        2,
        0,
        0,
        0,
    )
    load_segment = struct.pack(
        "<IIQQQQQQ", 1, 4, 0, virtual_base, virtual_base, size, size, 4096
    )
    dynamic_segment = struct.pack(
        "<IIQQQQQQ",
        2,
        4,
        dynamic_offset,
        virtual_base + dynamic_offset,
        virtual_base + dynamic_offset,
        dynamic_size,
        dynamic_size,
        8,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        header
        + load_segment
        + dynamic_segment
        + b"".join(struct.pack("<qQ", *entry) for entry in dynamic)
        + strings
    )
    return path
