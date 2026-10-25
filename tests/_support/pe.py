"""Small non-executable PE metadata fixtures encoded from the PE/COFF format.

These files describe headers/imports/exports only; they are never native provider
stand-ins and must not be loaded as DLLs.
"""

from __future__ import annotations

import struct
from pathlib import Path


def pe_image(
    *,
    imports: tuple[str, ...] = (),
    delay_imports: tuple[str, ...] = (),
    exports: tuple[tuple[str | None, int, str | None], ...] = (),
    machine: int = 0x8664,
) -> bytes:
    data = bytearray(0x3200)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x80)
    data[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", data, 0x84, machine, 1, 0, 0, 0, 0xF0, 0x2022)
    optional = 0x98
    struct.pack_into("<H", data, optional, 0x20B)
    struct.pack_into("<Q", data, optional + 24, 0x180000000)
    struct.pack_into("<II", data, optional + 32, 0x1000, 0x200)
    struct.pack_into("<II", data, optional + 56, 0x4000, 0x200)
    struct.pack_into("<I", data, optional + 108, 16)
    section = optional + 0xF0
    data[section : section + 8] = b".rdata\0\0"
    struct.pack_into("<IIII", data, section + 8, 0x3000, 0x1000, 0x3000, 0x200)
    struct.pack_into("<I", data, section + 36, 0x40000040)

    def rva(offset: int) -> int:
        return offset + 0xE00

    cursor = 0x1400

    def text(value: str) -> int:
        nonlocal cursor
        encoded = value.encode("ascii") + b"\0"
        start = cursor
        data[start : start + len(encoded)] = encoded
        cursor += len(encoded)
        return rva(start)

    for delayed, names, start in ((False, imports, 0x800), (True, delay_imports, 0xC00)):
        if not names:
            continue
        width = 32 if delayed else 20
        index = 13 if delayed else 1
        struct.pack_into(
            "<II", data, optional + 112 + index * 8, rva(start), (len(names) + 1) * width
        )
        for offset, name in enumerate(names):
            if delayed:
                struct.pack_into(
                    "<IIIIIIII",
                    data,
                    start + offset * width,
                    1,
                    text(name),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                )
            else:
                struct.pack_into(
                    "<IIIII", data, start + offset * width, 0, 0, 0, text(name), 0
                )
    if exports:
        base = min(item[1] for item in exports)
        count = max(item[1] for item in exports) - base + 1
        named = sorted(
            (item for item in exports if item[0] is not None),
            key=lambda item: item[0] or "",
        )
        functions = 0x240
        names = functions + count * 4
        ordinals = names + len(named) * 4
        cursor = ordinals + len(named) * 2
        for name, ordinal, forwarder in exports:
            target = text(forwarder) if forwarder is not None else 0x3000 + ordinal
            struct.pack_into("<I", data, functions + (ordinal - base) * 4, target)
        for index, (name, ordinal, _) in enumerate(named):
            if name is None:
                raise RuntimeError("Named export fixture has no name")
            struct.pack_into("<I", data, names + index * 4, text(name))
            struct.pack_into("<H", data, ordinals + index * 2, ordinal - base)
        struct.pack_into(
            "<IIHHIIIIIII",
            data,
            0x200,
            0,
            0,
            0,
            0,
            text("fixture.dll"),
            base,
            count,
            len(named),
            rva(functions),
            rva(names),
            rva(ordinals),
        )
        struct.pack_into("<II", data, optional + 112, 0x1000, cursor - 0x200)
    return bytes(data)


def write_pe(
    path: Path,
    *,
    imports: tuple[str, ...] = (),
    delay_imports: tuple[str, ...] = (),
    exports: tuple[tuple[str | None, int, str | None], ...] = (),
    machine: int = 0x8664,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        pe_image(
            imports=imports, delay_imports=delay_imports, exports=exports, machine=machine
        )
    )
    return path
