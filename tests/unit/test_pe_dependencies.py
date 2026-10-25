"""Consumer contracts for bounded non-loading PE32+ metadata inspection."""

from __future__ import annotations

import hashlib
import importlib.util
import struct
from pathlib import Path
from types import ModuleType

import pytest

from tests._support.pe import pe_image, write_pe


@pytest.fixture(scope="module")
def pe() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "phydrax/_pe_dependencies.py"
    spec = importlib.util.spec_from_file_location("standalone_pe_reader", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reports_ordinary_delayed_and_ordinal_exports(
    pe: ModuleType, tmp_path: Path
) -> None:
    path = write_pe(
        tmp_path / "profile with spaces Ω" / "fixture.dll",
        imports=("KERNEL32.dll", "cudart64_13.dll"),
        delay_imports=("CUDNN64_9.dll",),
        exports=(
            ("GetPjrtApi", 7, None),
            (None, 9, "KERNEL32.Sleep"),
            ("Alias", 10, "OTHER.#42"),
        ),
    )
    image = pe.inspect_pe(path)
    assert image.machine == 0x8664
    assert image.imports == ("cudart64_13.dll", "KERNEL32.dll")
    assert image.delay_imports == ("CUDNN64_9.dll",)
    assert [(item.name, item.ordinal, item.forwarder) for item in image.exports] == [
        ("GetPjrtApi", 7, None),
        (None, 9, "KERNEL32.Sleep"),
        ("Alias", 10, "OTHER.#42"),
    ]


def test_empty_optional_tables_are_valid(pe: ModuleType, tmp_path: Path) -> None:
    image = pe.inspect_pe(write_pe(tmp_path / "empty.dll"))
    assert image.imports == image.delay_imports == image.exports == ()


@pytest.mark.parametrize("machine", [0x14C, 0xAA64, 0])
def test_rejects_other_machines(pe: ModuleType, tmp_path: Path, machine: int) -> None:
    with pytest.raises(pe.PeFormatError, match="x86-64"):
        pe.inspect_pe(write_pe(tmp_path / "wrong.dll", machine=machine))


@pytest.mark.parametrize("length", [0, 1, 63, 100, 255, 511, 0x31FF])
def test_truncated_file_refuses(pe: ModuleType, tmp_path: Path, length: int) -> None:
    path = tmp_path / "truncated.dll"
    path.write_bytes(pe_image()[:length])
    with pytest.raises(pe.PeFormatError):
        pe.inspect_pe(path)


@pytest.mark.parametrize(
    "offset,format,value",
    [
        (0x3C, "<I", 0xFFFFFFF0),
        (0x98, "<H", 0x10B),
        (0x98 + 108, "<I", 100),
        (0x188 + 12, "<I", 0xFFFFFFF0),
        (0x188 + 20, "<I", 0x3100),
        (0x98 + 112 + 8, "<I", 0x5000),
        (0x800 + 12, "<I", 0x5000),
        (0x98 + 112 + 8 + 4, "<I", 20),
    ],
)
def test_invalid_ranges_and_unterminated_tables_refuse(
    pe: ModuleType,
    tmp_path: Path,
    offset: int,
    format: str,
    value: int,
) -> None:
    data = bytearray(pe_image(imports=("KERNEL32.dll",)))
    struct.pack_into(format, data, offset, value)
    path = tmp_path / "invalid.dll"
    path.write_bytes(data)
    with pytest.raises(pe.PeFormatError):
        pe.inspect_pe(path)


def test_delay_virtual_address_cannot_underflow(pe: ModuleType, tmp_path: Path) -> None:
    data = bytearray(pe_image(delay_imports=("vendor.dll",)))
    struct.pack_into("<I", data, 0xC00, 0)
    path = tmp_path / "delay.dll"
    path.write_bytes(data)
    with pytest.raises(pe.PeFormatError, match="RVA"):
        pe.inspect_pe(path)


def test_name_must_be_basename(pe: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(pe.PeFormatError, match="name"):
        pe.inspect_pe(write_pe(tmp_path / "escape.dll", imports=("../vendor.dll",)))


def test_ordinal_table_must_reference_export_address_table(
    pe: ModuleType, tmp_path: Path
) -> None:
    data = bytearray(pe_image(exports=(("entry", 1, None),)))
    struct.pack_into("<H", data, 0x248, 2)
    path = tmp_path / "ordinal.dll"
    path.write_bytes(data)
    with pytest.raises(pe.PeFormatError, match="ordinal"):
        pe.inspect_pe(path)


def test_overlapping_sections_are_not_arbitrarily_selected(
    pe: ModuleType, tmp_path: Path
) -> None:
    data = bytearray(pe_image())
    struct.pack_into("<H", data, 0x86, 2)
    data[0x1B0:0x1D8] = data[0x188:0x1B0]
    path = tmp_path / "ambiguous.dll"
    path.write_bytes(data)
    with pytest.raises(pe.PeFormatError, match="Overlapping"):
        pe.inspect_pe(path)


def test_system_classification_is_case_insensitive_not_a_search_policy(
    pe: ModuleType,
) -> None:
    assert pe.is_system_import("KERNEL32.DLL")
    assert pe.is_system_import("api-ms-win-core-synch-l1-2-0.dll")
    assert pe.is_system_import("nvcuda.dll")
    assert not pe.is_system_import("C:\\Windows\\System32\\kernel32.dll")
    assert not pe.is_system_import("cudnn64_9.dll")
    assert not pe.is_system_import("unowned.dll")


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "static-archive",
        "static-library",
        "missing-loader",
        "missing-marker",
        "overlay-markers",
        "unterminated-marker",
        "ordinary-cudart",
        "delayed-cudart",
        "forwarded-cudart",
        "wrong-linkage",
        "wrong-machine",
        "malformed-pe",
    ],
)
def test_cuda_hybrid_dispatch_requires_same_binary_and_linker_evidence(
    pe: ModuleType,
    tmp_path: Path,
    fault: str | None,
) -> None:
    # These metadata fixtures cannot execute; only binary admission is exercised.
    markers = ("nvcudart_hybrid64.dll", "cuGetProcAddress_v2", "cuGetExportTable")
    data = bytearray(
        pe_image(
            imports=("CuDaRt64_13.DLL",)
            if fault == "ordinary-cudart"
            else ("kernel32.dll",),
            delay_imports=("cudart64_13.dll",) if fault == "delayed-cudart" else (),
            exports=(("Alias", 1, "cudart64_13.cudaGetDevice"),)
            if fault == "forwarded-cudart"
            else (),
            machine=0xAA64 if fault == "wrong-machine" else 0x8664,
        )
    )
    encoded = b"".join(marker.encode("ascii") + b"\0" for marker in markers)
    if fault == "missing-marker":
        encoded = encoded.replace(b"cuGetExportTable\0", b"")
    elif fault == "unterminated-marker":
        encoded = encoded.replace(b"cuGetExportTable\0", b"cuGetExportTableX")
    if fault == "overlay-markers":
        data.extend(encoded)
    else:
        data[0x1E00 : 0x1E00 + len(encoded)] = encoded
    if fault == "malformed-pe":
        data[:2] = b"NO"
    path = tmp_path / "companion.dll"
    path.write_bytes(data)
    mapping = "0001:00001000 initializeLoader 0000000180001000 cudart:cudart_static_loader.obj\n"
    if fault == "static-archive":
        mapping += (
            "0001:00002000 cudaMalloc 0000000180002000 cudart_static:cuda_runtime.obj\n"
        )
    elif fault == "static-library":
        mapping += "cudart_static.lib(cuda_runtime.obj)\n"
    elif fault == "missing-loader":
        mapping = "unrelated cudart_static_loader marker, not an archive member\n"
    map_path = tmp_path / "companion.map"
    map_path.write_text(mapping, encoding="utf-8")
    linkage = {
        "kind": "static" if fault == "wrong-linkage" else "driver-hybrid",
        "target_dll": "nvcudart_hybrid64.dll",
        "owner": "display-driver",
    }
    if fault is not None:
        with pytest.raises(pe.PeFormatError):
            pe.inspect_cuda_runtime_linkage(path, map_path, linkage)
    else:
        proof = pe.inspect_cuda_runtime_linkage(path, map_path, linkage)
        assert proof["sha256"] == hashlib.sha256(data).hexdigest()
        assert (
            proof["link_map_sha256"] == hashlib.sha256(map_path.read_bytes()).hexdigest()
        )
        assert proof["imports"] == proof["closure_imports"] == ["kernel32.dll"]
        assert proof["delay_imports"] == []
        assert proof["dispatch_markers"] == list(markers)
        assert proof["static_runtime_excluded"] is True
