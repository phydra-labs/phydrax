"""Windows companion packaging admission without compiling or loading native code."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tools import build_cuda_wheels as build


if TYPE_CHECKING:
    from phydrax._pe_dependencies import PeImage


def configuration(
    tmp_path: Path, *, audit_wheel: Path | None = None
) -> build.Configuration:
    return build.Configuration(
        ("120-real", "120-virtual"),
        ("cudss",),
        Path(sys.executable),
        tmp_path / "wheels",
        tmp_path / "build",
        None,
        2,
        "none",
        audit_wheel,
    )


def cli_arguments(tmp_path: Path) -> list[str]:
    return [
        "--python",
        sys.executable,
        "--wheelhouse",
        str(tmp_path / "wheels"),
        "--build-dir",
        str(tmp_path / "build"),
    ]


@pytest.mark.parametrize(
    ("platform", "components"),
    [("win32", build.WINDOWS_COMPONENTS), ("linux", build.COMPONENTS)],
)
def test_platform_component_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    components: tuple[str, ...],
) -> None:
    monkeypatch.setattr(build.sys, "platform", platform)
    assert build.parse_configuration(cli_arguments(tmp_path)).components == components


@pytest.mark.parametrize("component", ["cutensornet", "cuest", "collective", "cudecomp"])
def test_windows_refuses_unavailable_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    component: str,
) -> None:
    monkeypatch.setattr(build.sys, "platform", "win32")
    with pytest.raises(SystemExit) as failure:
        build.parse_configuration(cli_arguments(tmp_path) + ["--components", component])
    assert failure.value.code == 2


def test_windows_refuses_linux_container(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(build.sys, "platform", "win32")
    with pytest.raises(SystemExit) as failure:
        build.parse_configuration(cli_arguments(tmp_path) + ["--container", "docker"])
    assert failure.value.code == 2


@pytest.mark.parametrize("architecture", ["AMD64", "x86_64"])
def test_windows_accepts_only_native_x64_host(
    monkeypatch: pytest.MonkeyPatch,
    architecture: str,
) -> None:
    monkeypatch.setattr(build.sys, "platform", "win32")
    monkeypatch.setattr(build.platform, "machine", lambda: architecture)
    assert build.host_platform().architecture == "x86_64"


def test_windows_arm_is_not_admitted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(build.sys, "platform", "win32")
    monkeypatch.setattr(build.platform, "machine", lambda: "ARM64")
    with pytest.raises(build.AuditFailure, match="Windows x86_64"):
        build.host_platform()


@pytest.mark.parametrize("component", build.WINDOWS_COMPONENTS)
def test_exact_public_exports_are_accepted(component: str) -> None:
    pe = build.pe_reader()
    names = build.expected_windows_exports(component)
    exports = tuple(
        pe.PeExport(name, index, 4096 + index, None)
        for index, name in enumerate(sorted(names))
    )
    image = pe.PeImage(0x8664, (), (), exports)
    assert build.verify_pe_exports(Path("component.dll"), image, component) == sorted(
        names
    )


@pytest.mark.parametrize("mutation", ["missing", "extra", "ordinal", "forwarded", "null"])
def test_windows_refuses_noncanonical_exports(mutation: str) -> None:
    pe = build.pe_reader()
    names = build.expected_windows_exports("cudss")
    exports = [
        pe.PeExport(name, index, 4096 + index, None)
        for index, name in enumerate(sorted(names))
    ]
    if mutation == "missing":
        exports.pop()
    elif mutation == "extra":
        exports.append(pe.PeExport("AMGX_initialize", 100, 8192, None))
    elif mutation == "ordinal":
        exports.append(pe.PeExport(None, 100, 8192, None))
    elif mutation == "forwarded":
        exports[0] = exports[0]._replace(forwarder="other.call")
    else:
        exports[0] = exports[0]._replace(rva=0)
    image = pe.PeImage(0x8664, (), (), tuple(exports))
    with pytest.raises(build.AuditFailure, match="export"):
        build.verify_pe_exports(Path("component.dll"), image, "cudss")


def test_pe_closure_follows_delay_imports_and_forwarders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pe = build.pe_reader()
    provider = tmp_path / "cudart64_13.dll"
    provider.write_bytes(b"provider fixture")
    delayed = tmp_path / "provider_math.dll"
    delayed.write_bytes(b"delayed provider fixture")
    root = pe.PeImage(0x8664, ("CUDART64_13.DLL",), (), ())
    images = {
        provider: pe.PeImage(0x8664, ("KERNEL32.dll",), ("provider_math.dll",), ()),
        delayed: pe.PeImage(
            0x8664, (), (), (pe.PeExport("driver", 1, 8192, "nvcuda.call"),)
        ),
    }

    def inspect(path: Path) -> PeImage:
        return images[path]

    monkeypatch.setattr(pe, "inspect_pe", inspect)
    inventory = build.NvidiaInventory({}, {path.name: path for path in images}, {})
    closure = build.verify_pe_closure({"component.dll": root}, inventory)
    assert set(closure) == {
        "cudart64_13.dll",
        "provider_math.dll",
        "kernel32.dll",
        "nvcuda.dll",
    }
    assert closure["nvcuda.dll"] == {
        "owner": "driver",
        "runtime_availability_observed": False,
    }


@pytest.mark.parametrize(
    "name", ["unknown.dll", "cudart64_12.dll", "C:/toolkit/provider.dll"]
)
def test_pe_closure_refuses_unowned_import(name: str) -> None:
    pe = build.pe_reader()
    image = pe.PeImage(0x8664, (), (name,), ())
    with pytest.raises(build.AuditFailure, match="PE import"):
        build.verify_pe_closure(
            {"component.dll": image}, build.NvidiaInventory({}, {}, {})
        )


def test_windows_refuses_static_runtime_link_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(build.sys, "platform", "win32")
    config = configuration(tmp_path)
    config.build_dir.mkdir()
    (config.build_dir / "build.ninja").write_text(
        "  LINK_LIBRARIES = C:/toolkit/CUDART_STATIC.LIB\n",
        encoding="utf-8",
    )
    with pytest.raises(build.AuditFailure, match="static cudart"):
        build.archive_link_evidence(config)


def test_windows_refuses_static_msvc_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(build.sys, "platform", "win32")
    config = configuration(tmp_path)
    config.build_dir.mkdir()
    (config.build_dir / "build.ninja").write_text("  FLAGS = /O2 /MT\n", encoding="utf-8")
    with pytest.raises(build.AuditFailure, match="static MSVC CRT"):
        build.archive_link_evidence(config)


def test_windows_audit_failure_writes_honest_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(build.sys, "platform", "win32")
    monkeypatch.setattr(build.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(build, "profile_path", lambda path: path)
    config = configuration(tmp_path, audit_wheel=tmp_path / "missing-win_amd64.whl")

    def fail_inventory() -> build.NvidiaInventory:
        raise build.AuditFailure("NVIDIA RECORD mismatch: cudart64_13.dll")

    monkeypatch.setattr(build, "inventory", fail_inventory)
    assert build.execute(config) == 1
    manifest = json.loads(
        (config.wheelhouse / "phydrax-cuda13-build.json").read_text(encoding="utf-8")
    )
    assert manifest["successful"] is False
    assert "NVIDIA RECORD mismatch" in manifest["error"]
    assert manifest["commands"] == []


def test_windows_build_emits_raw_wheel_without_linux_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(build.sys, "platform", "win32")
    config = configuration(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()
    commands: list[list[str]] = []

    def run(
        command: Sequence[str],
        environment: Mapping[str, str],
        evidence: dict[str, build.Json],
    ) -> str:
        commands.append(list(command))
        output = Path(command[command.index("--outdir") + 1])
        (output / "phydrax_cuda13-0.2.2-py3-none-win_amd64.whl").write_bytes(b"wheel")
        return ""

    monkeypatch.setattr(build, "run", run)
    wheel, policy = build.build_and_repair(
        config,
        staging,
        build.NvidiaInventory({}, {}, {}),
        {"PHX_WINDOWS_NINJA": str(tmp_path / "ninja.exe")},
        {},
    )
    assert wheel.name.endswith("-win_amd64.whl")
    assert policy == frozenset()
    assert len(commands) == 1
    assert commands[0][1:3] == ["-m", "build"]


def test_windows_host_only_dll_does_not_claim_architecture_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(build.sys, "platform", "win32")
    config = configuration(tmp_path)
    library = tmp_path / "phydrax_cuda_cudss.dll"
    diagnostic = f"cuobjdump info : File '{library}' does not contain device code"

    def run(
        command: Sequence[str],
        environment: Mapping[str, str],
        evidence: dict[str, build.Json],
        *,
        expected_failure: tuple[int, str] | None = None,
    ) -> str:
        assert expected_failure == (0xFFFFFFFF, diagnostic)
        return diagnostic

    monkeypatch.setattr(build, "run", run)
    result = build.inspect_device_code(
        library, tmp_path / "cuobjdump.exe", config, {}, {}
    )
    assert result["host_only_component"] is True
    assert result["release_architecture_coverage"] is False


def test_windows_device_component_cannot_omit_ptx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = configuration(tmp_path)

    def run(
        command: Sequence[str],
        environment: Mapping[str, str],
        evidence: dict[str, build.Json],
        *,
        expected_failure: tuple[int, str] | None = None,
    ) -> str:
        return "sm_120" if "--list-elf" in command else ""

    monkeypatch.setattr(build, "run", run)
    with pytest.raises(build.AuditFailure, match="missing SASS.*PTX"):
        build.inspect_device_code(
            tmp_path / "phydrax_cuda_quda.dll",
            tmp_path / "cuobjdump.exe",
            config,
            {},
            {},
        )


@pytest.mark.parametrize("delay_loaded", [False, True])
def test_windows_accepts_only_shared_cuda13_import(
    delay_loaded: bool, tmp_path: Path
) -> None:
    pe = build.pe_reader()
    imports = ("CUDART64_13.DLL",)
    image = pe.PeImage(
        0x8664,
        () if delay_loaded else imports,
        imports if delay_loaded else (),
        (),
    )
    assert (
        build.verify_pe_shared_cudart(
            Path("component.dll"),
            image,
            {
                "runtime_linkage": {
                    "kind": "dll-import",
                    "target_dll": "cudart64_13.dll",
                    "owner": "nvidia-cuda-runtime",
                }
            },
            configuration(tmp_path),
        )
        == "cudart64_13.dll"
    )


@pytest.mark.parametrize(
    "imports",
    [(), ("cudart64_12.dll",), ("cudart64_13.dll", "cudart64_12.dll")],
)
def test_windows_refuses_missing_or_foreign_cudart(
    imports: tuple[str, ...], tmp_path: Path
) -> None:
    pe = build.pe_reader()
    image = pe.PeImage(0x8664, imports, (), ())
    with pytest.raises(build.AuditFailure):
        build.verify_pe_shared_cudart(
            Path("component.dll"),
            image,
            {
                "runtime_linkage": {
                    "kind": "dll-import",
                    "target_dll": "cudart64_13.dll",
                    "owner": "nvidia-cuda-runtime",
                }
            },
            configuration(tmp_path),
        )
