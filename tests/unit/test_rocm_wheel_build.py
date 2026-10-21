"""Small archive fixtures exercise owned ROCm packaging without SDK/GPU access."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import stat
import zipfile
from collections.abc import Sequence
from pathlib import Path

import pytest

from tools import build_rocm_wheels as builder


def upstream(
    tmp_path: Path,
    component: str = "core",
    *,
    corrupt_record: bool = False,
    corrupt_record_path: str | None = None,
    payload_architecture: str | None = None,
    extra_files: dict[str, tuple[bytes, int]] | None = None,
) -> tuple[Path, builder.SourceArchive]:
    path = tmp_path / f"rocm_sdk_{component}-10.1.0-py3-none-linux_x86_64.whl"
    module = "_rocm_sdk_core" if component == "core" else "_rocm_sdk_libraries"
    files = {
        module + "/__init__.py": (b"", 0o644),
        module + "/bin/tool": (b"unchanged executable payload\0", 0o755),
        module + "/share/doc/LICENSE": (b"original third-party notice\n", 0o644),
    }
    if component.startswith("device_"):
        architecture = payload_architecture or component.removeprefix("device_")
        files[f"{module}/.kpack/blas_lib_{architecture}.kpack"] = (b"kpack", 0o644)
        files[f"{module}/lib/hipblaslt/library/{architecture}/code.co"] = (
            b"device code",
            0o644,
        )
    if extra_files:
        files.update(extra_files)
    dist = f"rocm_sdk_{component}-10.1.0.dist-info"
    files[dist + "/METADATA"] = (
        b"Metadata-Version: 2.1\nName: upstream\nVersion: 10.1.0\n",
        0o644,
    )
    files[dist + "/WHEEL"] = (b"Wheel-Version: 1.0\nTag: py3-none-linux_x86_64\n", 0o644)
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for name, (data, _) in sorted(files.items()):
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        )
        if corrupt_record or name == corrupt_record_path:
            digest = "wrong"
        writer.writerow((name, "sha256=" + digest, str(len(data))))
    writer.writerow((dist + "/RECORD", "", ""))
    files[dist + "/RECORD"] = (record.getvalue().encode(), 0o644)
    with zipfile.ZipFile(path, "w") as archive:
        for name, (data, mode) in files.items():
            archive.writestr(builder.wheel_info(name, mode), data)
    source: builder.SourceArchive = {
        "filename": path.name,
        "url": "https://example.invalid/" + path.name,
        "sha256": builder.file_sha256(path),
    }
    return path, source


def lock() -> dict[str, str | list[str]]:
    return {
        "sdk_version": "10.1.0",
        "requires_python": ">=3.13,<3.14",
        "runtime_roots": [],
        "architectures": ["gfx1250", "gfx942", "gfx950"],
    }


@pytest.mark.parametrize(
    "name", ["core", "libraries", "device_gfx1250", "device_gfx942", "device_gfx950"]
)
def test_repack_preserves_bytes_modes_notices_and_original_metadata(
    tmp_path: Path, name: str
) -> None:
    source_path, source = upstream(tmp_path, name)
    output = tmp_path / "output"
    output.mkdir()
    component, inventory, native = builder.repack_component(
        source_path, source, name, lock(), output, {}
    )
    assert native == []
    wheel = output / f"phydrax_rocm_{name}-10.1.0-py3-none-linux_x86_64.whl"
    dist = f"phydrax_rocm_{name}-10.1.0.dist-info"
    with zipfile.ZipFile(source_path) as old, zipfile.ZipFile(wheel) as new:
        for entry in inventory:
            assert new.read(entry["path"]) == old.read(entry["path"])
            assert (
                stat.S_IMODE(new.getinfo(entry["path"]).external_attr >> 16)
                == entry["mode"]
            )
        upstream_metadata = f"rocm_sdk_{name}-10.1.0.dist-info/METADATA"
        assert new.read(dist + "/upstream/" + upstream_metadata) == old.read(
            upstream_metadata
        )
        payload = new.read(dist + "/phydrax-payload.json")
        assert json.loads(payload) == inventory
        assert hashlib.sha256(payload).hexdigest() == component["payload_sha256"]
        metadata = new.read(dist + "/METADATA").decode()
        assert f"Name: phydrax-rocm-{name.replace('_', '-')}\n" in metadata
        assert "Requires-Python: >=3.13,<3.14\n" in metadata
        assert "Requires-Dist:" not in metadata
        evidence = json.loads(new.read(dist + "/licenses/architecture-coverage.json"))
        assert evidence["targets"]["gfx1250"]["vendor_release_ready"] is False
        assert (
            evidence["targets"]["gfx1250"]["phydrax_hardware_validation"] == "not tested"
        )
        rows = list(csv.reader(io.StringIO(new.read(dist + "/RECORD").decode())))
        assert {row[0] for row in rows} == set(new.namelist())
        for path, digest, size in rows:
            if not digest:
                assert path == dist + "/RECORD"
                continue
            data = new.read(path)
            assert len(data) == int(size)
            assert (
                digest
                == "sha256="
                + base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                .rstrip(b"=")
                .decode()
            )


@pytest.mark.parametrize("architecture", ["gfx1250", "gfx942", "gfx950"])
def test_identical_initializer_has_only_one_owner(
    tmp_path: Path, architecture: str
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    shared: dict[str, builder.OwnedPayloadEntry] = {}
    first_path, first = upstream(tmp_path, "libraries")
    builder.repack_component(first_path, first, "libraries", lock(), output, shared)
    second_path, second = upstream(tmp_path, f"device_{architecture}")
    component, inventory, _ = builder.repack_component(
        second_path, second, f"device_{architecture}", lock(), output, shared
    )
    assert {entry["path"] for entry in inventory} == {
        f"_rocm_sdk_libraries/.kpack/blas_lib_{architecture}.kpack",
        f"_rocm_sdk_libraries/lib/hipblaslt/library/{architecture}/code.co",
    }
    assert {entry["owner"] for entry in component["deduplicated"]} == {
        "phydrax-rocm-libraries"
    }
    with zipfile.ZipFile(
        output / f"phydrax_rocm_device_{architecture}-10.1.0-py3-none-linux_x86_64.whl"
    ) as archive:
        assert "_rocm_sdk_libraries/__init__.py" not in archive.namelist()


def test_conflicting_shared_payload_is_refused(tmp_path: Path) -> None:
    path, source = upstream(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    shared: dict[str, builder.OwnedPayloadEntry] = {
        "_rocm_sdk_core/__init__.py": {
            "path": "_rocm_sdk_core/__init__.py",
            "sha256": "0" * 64,
            "size": 0,
            "mode": 0o644,
            "owner": "phydrax-rocm-core",
        }
    }
    with pytest.raises(ValueError, match="conflicting cross-component"):
        builder.repack_component(path, source, "core", lock(), output, shared)
    assert not list(output.iterdir())


@pytest.mark.parametrize(
    "name",
    ["../escape", "/absolute", "a/../escape", "a\\escape", "C:/escape", "a//b", "./a"],
)
def test_unsafe_archive_paths_are_refused(name: str) -> None:
    with pytest.raises(ValueError, match="unsafe archive path"):
        builder.safe_name(name)


def test_symlink_archive_member_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.whl"
    info = builder.wheel_info("payload/link")
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(info, "../../outside")
    with zipfile.ZipFile(path) as archive, pytest.raises(ValueError, match="member type"):
        builder.checked_members(archive)


def test_corrupt_archive_hash_is_refused(tmp_path: Path) -> None:
    _, source = upstream(tmp_path)
    source["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source hash mismatch"):
        builder.fetch(source, tmp_path, download=False)


def test_corrupt_upstream_record_is_refused(tmp_path: Path) -> None:
    path, source = upstream(tmp_path, corrupt_record=True)
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match="upstream RECORD mismatch"):
        builder.repack_component(path, source, "core", lock(), output, {})
    assert not list(output.iterdir())


def library(
    name: str, needed: Sequence[str] = (), *, digest: str = "a"
) -> builder.NativeObject:
    return {
        "path": "_rocm_sdk_core/lib/" + name,
        "soname": name,
        "needed": list(needed),
        "sha256": digest * 64,
    }


def test_native_dependencies_are_topologically_ordered() -> None:
    leaf = library("libleaf.so", ["libc.so.6"])
    root = library("libroot.so", ["libleaf.so", "libstdc++.so.6"])
    assert builder.dependency_order([root, leaf], ["libroot.so"]) == [leaf, root]


def test_missing_native_dependency_is_not_replaced_by_host_sdk() -> None:
    with pytest.raises(ValueError, match="missing owned native dependency: libamdhip64"):
        builder.dependency_order([library("root.so", ["libamdhip64.so.7"])], ["root.so"])


def test_conflicting_sonames_are_refused() -> None:
    with pytest.raises(ValueError, match="conflicting owned SONAME"):
        builder.dependency_order(
            [library("same.so"), library("same.so", digest="b")], ["same.so"]
        )


def test_dependency_cycles_are_refused() -> None:
    with pytest.raises(ValueError, match="cyclic native dependency"):
        builder.dependency_order(
            [library("a.so", ["b.so"]), library("b.so", ["a.so"])], ["a.so"]
        )


def manifest_inventory() -> list[builder.PayloadEntry]:
    return [
        {"path": name, "sha256": "0" * 64, "size": 0, "mode": 0o644}
        for name in (
            "_rocm_sdk_core/lib/llvm/bin/ld.lld",
            "_rocm_sdk_core/lib/llvm/amdgcn/bitcode/ocml.bc",
            "_rocm_sdk_libraries/.kpack/blas_lib_gfx1250.kpack",
            "_rocm_sdk_libraries/.kpack/blas_lib_gfx942.kpack",
            "_rocm_sdk_libraries/.kpack/blas_lib_gfx950.kpack",
            "_rocm_sdk_libraries/lib/hipblaslt/library/gfx1250/code.co",
            "_rocm_sdk_libraries/lib/hipblaslt/library/gfx942/code.co",
            "_rocm_sdk_libraries/lib/hipblaslt/library/gfx950/code.co",
        )
    ]


def manifest_components() -> dict[str, builder.ComponentRecord]:
    return {
        name: {
            "distribution": "phydrax-rocm-" + name.replace("_", "-"),
            "version": "10.1.0",
            "module": "_rocm_sdk_core" if name == "core" else "_rocm_sdk_libraries",
            "upstream": {"url": "https://example.invalid/" + name, "sha256": "0" * 64},
            "payload_sha256": "0" * 64,
            "deduplicated": [],
        }
        for name in builder.COMPONENTS
    }


def test_manifest_keeps_architecture_kpack_paths_separate() -> None:
    components = manifest_components()
    manifest = builder.runtime_manifest(
        lock(), components, manifest_inventory(), [], [], []
    )
    assert manifest["architectures"] == ["gfx1250", "gfx942", "gfx950"]
    for architecture in manifest["architectures"]:
        assert all(
            architecture in path for path in manifest["loader"]["kpack"][architecture]
        )
    assert "schema" not in manifest and "generation" not in manifest


@pytest.mark.parametrize("architecture", ["gfx1250", "gfx942", "gfx950"])
@pytest.mark.parametrize("missing", ["kpack", "code", "both"])
def test_missing_architecture_is_refused(architecture: str, missing: str) -> None:
    entries = [
        entry
        for entry in manifest_inventory()
        if not (
            architecture in entry["path"]
            and (
                missing == "both"
                or (missing == "kpack" and entry["path"].endswith(".kpack"))
                or (missing == "code" and entry["path"].endswith(".co"))
            )
        )
    ]
    with pytest.raises(ValueError, match=f"missing architecture payload: {architecture}"):
        builder.runtime_manifest(lock(), manifest_components(), entries, [], [], [])


def test_missing_compiler_payload_is_refused() -> None:
    entries = [
        entry for entry in manifest_inventory() if not entry["path"].endswith("ld.lld")
    ]
    with pytest.raises(ValueError, match="missing owned compiler linker"):
        builder.runtime_manifest(lock(), manifest_components(), entries, [], [], [])


def test_runtime_metadata_has_only_exact_owned_dependencies() -> None:
    dependencies = [
        "phydrax-rocm-" + name.replace("_", "-") + "==10.1.0"
        for name in builder.COMPONENTS
    ]
    text = builder.metadata(
        "phydrax-rocm-runtime", "10.1.0", ">=3.13,<3.14", dependencies
    ).decode()
    assert [
        line.removeprefix("Requires-Dist: ")
        for line in text.splitlines()
        if line.startswith("Requires-Dist: ")
    ] == dependencies
    assert "http" not in text and "rocm-sdk" not in text


def test_repack_is_reproducible(tmp_path: Path) -> None:
    path, source = upstream(tmp_path)
    hashes = []
    for name in ("first", "second"):
        output = tmp_path / name
        output.mkdir()
        builder.repack_component(path, source, "core", lock(), output, {})
        hashes.append(builder.file_sha256(next(output.glob("*.whl"))))
    assert hashes[0] == hashes[1]


@pytest.mark.parametrize(
    "component", ["core", "libraries", "device_gfx1250", "device_gfx942", "device_gfx950"]
)
def test_missing_component_is_refused(component: str) -> None:
    components = manifest_components()
    del components[component]
    with pytest.raises(ValueError, match="all five components"):
        builder.runtime_manifest(lock(), components, manifest_inventory(), [], [], [])


@pytest.mark.parametrize("architecture", ["gfx1250", "gfx942", "gfx950"])
def test_device_component_cannot_borrow_other_architecture(
    tmp_path: Path, architecture: str
) -> None:
    component = f"device_{architecture}"
    other = "gfx942" if architecture != "gfx942" else "gfx950"
    path, source = upstream(tmp_path, component, payload_architecture=other)
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match=f"missing architecture payload: {architecture}"):
        builder.repack_component(path, source, component, lock(), output, {})
    assert not list(output.iterdir())


def test_device_payload_namespace_is_enforced(tmp_path: Path) -> None:
    path, source = upstream(
        tmp_path,
        "device_gfx1250",
        extra_files={"_rocm_sdk_core/lib/unowned": (b"wrong namespace", 0o644)},
    )
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match="unexpected component payload namespace"):
        builder.repack_component(path, source, "device_gfx1250", lock(), output, {})
    assert not list(output.iterdir())


def test_shared_payload_record_is_still_authenticated(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    shared: dict[str, builder.OwnedPayloadEntry] = {}
    path, source = upstream(tmp_path, "libraries")
    builder.repack_component(path, source, "libraries", lock(), output, shared)
    path, source = upstream(
        tmp_path,
        "device_gfx1250",
        corrupt_record_path="_rocm_sdk_libraries/__init__.py",
    )
    with pytest.raises(ValueError, match="upstream RECORD mismatch"):
        builder.repack_component(path, source, "device_gfx1250", lock(), output, shared)
    assert not (
        output / "phydrax_rocm_device_gfx1250-10.1.0-py3-none-linux_x86_64.whl"
    ).exists()


def test_all_component_payloads_reach_runtime_wheel(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    shared: dict[str, builder.OwnedPayloadEntry] = {}
    components: dict[str, builder.ComponentRecord] = {}
    inventory: list[builder.PayloadEntry] = []
    for name in builder.COMPONENTS:
        extra = (
            {
                "_rocm_sdk_core/lib/llvm/bin/ld.lld": (b"linker fixture", 0o755),
                "_rocm_sdk_core/lib/llvm/amdgcn/bitcode/ocml.bc": (b"bitcode", 0o644),
                "rocm_sdk_core/__init__.py": (b"", 0o644),
            }
            if name == "core"
            else {"rocm_sdk_libraries/__init__.py": (b"", 0o644)}
            if name == "libraries"
            else None
        )
        path, source = upstream(tmp_path, name, extra_files=extra)
        component, entries, native = builder.repack_component(
            path, source, name, lock(), output, shared
        )
        assert native == []
        components[name] = component
        inventory.extend(entries)
    manifest = builder.runtime_manifest(lock(), components, inventory, [], [], [])
    runtime_path = builder.write_runtime_wheel(lock(), manifest, output)
    assert len(list(output.glob("*.whl"))) == 6
    assert shared["_rocm_sdk_libraries/__init__.py"]["owner"] == "phydrax-rocm-libraries"
    with zipfile.ZipFile(runtime_path) as archive:
        installed = json.loads(archive.read("phydrax_rocm_runtime/runtime.json"))
        assert installed["architectures"] == ["gfx1250", "gfx942", "gfx950"]
        metadata = archive.read("phydrax_rocm_runtime-10.1.0.dist-info/METADATA").decode()
        assert [
            line.removeprefix("Requires-Dist: ")
            for line in metadata.splitlines()
            if line.startswith("Requires-Dist: ")
        ] == [
            "phydrax-rocm-core==10.1.0",
            "phydrax-rocm-libraries==10.1.0",
            "phydrax-rocm-device-gfx1250==10.1.0",
            "phydrax-rocm-device-gfx942==10.1.0",
            "phydrax-rocm-device-gfx950==10.1.0",
        ]
        for architecture in installed["architectures"]:
            assert installed["loader"]["kpack"][architecture] == [
                f"_rocm_sdk_libraries/.kpack/blas_lib_{architecture}.kpack"
            ]


@pytest.mark.parametrize(
    "architectures", [["gfx942", "gfx950"], ["gfx950", "gfx942", "gfx1250"]]
)
def test_incomplete_or_unsorted_lock_coverage_is_refused(
    architectures: list[str],
) -> None:
    configuration = lock()
    configuration["architectures"] = architectures
    with pytest.raises(ValueError, match="complete sorted three-architecture coverage"):
        builder.runtime_manifest(
            configuration, manifest_components(), manifest_inventory(), [], [], []
        )
