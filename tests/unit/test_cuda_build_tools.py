"""CUDA 13 build and resolver gates refuse incompatible runtime artifacts."""

import json
import os
import platform
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Mapping, Sequence
from email.message import Message
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from tools import build_cuda_wheels as build, check_cuda_extras as extras


@pytest.mark.parametrize("failure", ["architecture", "archive-hash"])
def test_cuest_runtime_recipe_refuses_invalid_build_inputs(
    tmp_path: Path, failure: str
) -> None:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("runtime packaging coverage requires CMake")
    if sys.platform != "linux" or platform.machine() not in ("x86_64", "aarch64"):
        pytest.skip("runtime packaging coverage requires a supported Linux host")
    source = Path(__file__).resolve().parents[2] / "native/cuest_runtime"
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    archive_platform = "linux-x86_64" if platform.machine() == "x86_64" else "linux-sbsa"
    archive = downloads / f"libcuest-{archive_platform}-0.2.2.2_cuda13-archive.tar.xz"
    archive.write_bytes(b"Untrusted archive: must be refused before extraction")
    command = [
        cmake,
        "-S",
        str(source.resolve()),
        "-B",
        str(tmp_path / "build"),
        f"-DPHX_CUDA_DOWNLOAD_DIR={downloads}",
        f"-DSKBUILD_METADATA_DIR={tmp_path / 'metadata'}",
    ]
    if failure == "architecture":
        command.extend(["-DCMAKE_SYSTEM_NAME=Linux", "-DCMAKE_SYSTEM_PROCESSOR=riscv64"])
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    diagnostic = (
        "do not support cross-compilation"
        if failure == "architecture"
        else "archive SHA256 mismatch"
    )
    assert diagnostic in result.stdout + result.stderr
    assert not (tmp_path / "build/cuest-support").exists()


def cuda_metadata() -> Message:
    project = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )["project"]
    metadata = Message()
    metadata["Version"] = project["version"]
    for extra in ("cuda13", "cutensor", "cutensornet", "cuest"):
        metadata["Provides-Extra"] = extra
        for text in project["optional-dependencies"][extra]:
            requirement = Requirement(text)
            marker = f"extra == '{extra}'"
            if requirement.marker is not None:
                marker = f"({requirement.marker}) and {marker}"
            requirement.marker = None
            metadata["Requires-Dist"] = f"{requirement}; {marker}"
    return metadata


def test_cuda_metadata_records_core_and_cuest_intent() -> None:
    metadata = cuda_metadata()
    assert extras.metadata_errors(metadata) == []

    incomplete = Message()
    incomplete["Version"] = "0.2.2"
    incomplete["Provides-Extra"] = "cuda13"
    incomplete["Provides-Extra"] = "cuest"
    incomplete["Requires-Dist"] = 'phydrax-cuda-major==13.*; extra == "cuda13"'
    requirements = metadata.get_all("Requires-Dist")
    if requirements is None:
        raise AssertionError("complete metadata fixture lost its requirements")
    incomplete["Requires-Dist"] = requirements[-1]
    assert any(
        "marker requirements" in error for error in extras.metadata_errors(incomplete)
    )


@pytest.fixture
def linux_resolution() -> dict[str, str]:
    return extras.resolved_distributions(
        """
        phydrax-cuda-major==13.0
        phydrax-cuda13==0.2.2
        jax-cuda13-plugin==0.12.0
        jax-cuda13-pjrt==0.12.0
        nvidia-cuda-runtime==13.4.92
        nvidia-cublas==13.8.0.4
        nvidia-cusparse==12.8.6.72
        nvidia-cusolver==12.3.4.7
        nvidia-cudss-cu13==0.8.0
        """
    )


@pytest.mark.parametrize("architecture", ["x86_64", "aarch64"])
def test_cuda13_resolution_accepts_math_library_version12(
    linux_resolution: dict[str, str], architecture: str
) -> None:
    assert not extras.resolution_errors(
        linux_resolution, f"{architecture}-manylinux_2_28", "0.2.2"
    )


@pytest.mark.parametrize(
    ("distribution", "version", "diagnostic"),
    [
        ("phydrax-cuda-major", "12.0", "Missing or incorrect CUDA 13 marker"),
        ("phydrax-cuda12", "0.2.2", "Expected phydrax-cuda13"),
        ("jax-cuda12-plugin", "0.11.2", "Expected CUDA 13 JAX plugin"),
        ("nvidia-cuda-runtime-cu12", "12.9.79", "Foreign CUDA 12 distribution"),
        ("nvidia-cuda-runtime", "12.9.79", "Expected CUDA 13 runtime version"),
        ("cutensor-cu12", "2.8.1", "Foreign CUDA 12 distribution"),
        ("cutensornet-cu12", "2.14.0", "Foreign CUDA 12 distribution"),
    ],
)
def test_cuda13_resolution_refuses_foreign_runtime(
    linux_resolution: dict[str, str],
    distribution: str,
    version: str,
    diagnostic: str,
) -> None:
    linux_resolution[distribution] = version
    errors = extras.resolution_errors(linux_resolution, "x86_64-manylinux_2_28", "0.2.2")
    assert any(diagnostic in error for error in errors)


@pytest.mark.parametrize("platform", ["aarch64-apple-darwin", "aarch64-pc-windows-msvc"])
def test_unsupported_host_cuda_intent_requires_no_gpu_packages(platform: str) -> None:
    resolved = {"phydrax-cuda-major": "13.0"}
    assert not extras.resolution_errors(resolved, platform, "0.2.2")
    resolved["nvidia-cuda-runtime"] = "13.4.92"
    assert any(
        "Unsupported-host resolution includes" in error
        for error in extras.resolution_errors(resolved, platform, "0.2.2")
    )


@pytest.mark.parametrize("architecture", ["x86_64", "aarch64"])
@pytest.mark.parametrize("companion_version", [None, "0.2.1"])
def test_linux_resolution_requires_lockstep_native_companion(
    linux_resolution: dict[str, str],
    architecture: str,
    companion_version: str | None,
) -> None:
    if companion_version is None:
        linux_resolution.pop("phydrax-cuda13")
    else:
        linux_resolution["phydrax-cuda13"] = companion_version
    assert any(
        "Expected phydrax-cuda13==0.2.2" in error
        for error in extras.resolution_errors(
            linux_resolution, f"{architecture}-manylinux_2_28", "0.2.2"
        )
    )


@pytest.mark.parametrize(
    ("distribution", "release"),
    [
        ("cutensor-cu13", "2.8.1"),
        ("cutensornet-cu13", "2.14.0"),
        ("phydrax-cuest", "0.2.2"),
    ],
)
def test_optional_cuda_runtime_resolution_keeps_its_declared_major(
    linux_resolution: dict[str, str], distribution: str, release: str
) -> None:
    linux_resolution[distribution] = release
    assert (
        extras.resolution_errors(linux_resolution, "x86_64-manylinux_2_28", "0.2.2") == []
    )


def test_optional_runtime_release_must_match_its_companion(
    linux_resolution: dict[str, str],
) -> None:
    linux_resolution["phydrax-cuest"] = "0.2.1"
    errors = extras.resolution_errors(linux_resolution, "x86_64-manylinux_2_28", "0.2.2")
    assert any("phydrax-cuest==0.2.2" in error for error in errors)


@pytest.fixture
def windows_resolution(linux_resolution: dict[str, str]) -> dict[str, str]:
    linux_resolution.pop("jax-cuda13-plugin")
    linux_resolution.pop("jax-cuda13-pjrt")
    return {
        **linux_resolution,
        "jax": "0.12.0",
        "jaxlib": "0.12.0",
        "phydrax-cuda13-pjrt": "0.2.2",
    }


def test_windows_resolution_requires_owned_stock_stack(
    windows_resolution: dict[str, str],
) -> None:
    assert (
        extras.resolution_errors(windows_resolution, "x86_64-pc-windows-msvc", "0.2.2")
        == []
    )


@pytest.mark.parametrize(
    "name", ("jax", "jaxlib", "phydrax-cuda13", "phydrax-cuda13-pjrt")
)
@pytest.mark.parametrize("version", (None, "0.0.0"))
def test_windows_resolution_refuses_missing_or_mismatched_stack(
    windows_resolution: dict[str, str],
    name: str,
    version: str | None,
) -> None:
    if version is None:
        windows_resolution.pop(name)
    else:
        windows_resolution[name] = version
    assert any(
        name in error
        for error in extras.resolution_errors(
            windows_resolution, "x86_64-pc-windows-msvc", "0.2.2"
        )
    )


@pytest.mark.parametrize(
    "owner", ("jax-cuda13-plugin", "jax-cuda13-pjrt", "jax-cuda12-plugin", "winjax")
)
def test_windows_resolution_refuses_competing_import_owners(
    windows_resolution: dict[str, str],
    owner: str,
) -> None:
    windows_resolution[owner] = "0.12.0"
    assert any(
        "sole Windows CUDA owner" in error
        for error in extras.resolution_errors(
            windows_resolution, "x86_64-pc-windows-msvc", "0.2.2"
        )
    )


@pytest.mark.parametrize(
    "provider",
    ("cutensornet-cu13", "phydrax-cuest", "nvidia-nccl-cu13", "nvidia-nvshmem-cu13"),
)
def test_windows_resolution_does_not_qualify_unsupported_providers(
    windows_resolution: dict[str, str],
    provider: str,
) -> None:
    windows_resolution[provider] = "0.2.2"
    assert any(
        "Unsupported Windows runtime" in error
        for error in extras.resolution_errors(
            windows_resolution, "x86_64-pc-windows-msvc", "0.2.2"
        )
    )


@pytest.mark.parametrize("extra", ("cutensornet", "cuest"))
def test_metadata_refuses_silently_erased_windows_provider(extra: str) -> None:
    metadata = cuda_metadata()
    requirements = metadata.get_all("Requires-Dist", [])
    del metadata["Requires-Dist"]
    for text in requirements:
        requirement = Requirement(text)
        if requirement.marker is not None and f'extra == "{extra}"' in str(
            requirement.marker
        ):
            metadata["Requires-Dist"] = f"{text} and sys_platform == 'linux'"
        else:
            metadata["Requires-Dist"] = text
    assert any(
        f"[{extra}]" in error and "win32" in error
        for error in extras.metadata_errors(metadata)
    )


def test_build_cli_rejects_removed_major_selector(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as failure:
        build.parse_configuration(
            [
                "--major",
                "12",
                "--python",
                sys.executable,
                "--wheelhouse",
                str(tmp_path / "wheels"),
                "--build-dir",
                str(tmp_path / "build"),
            ]
        )
    assert failure.value.code == 2


@pytest.mark.parametrize(
    "needed",
    [(), ("libcudart.so.12",), ("libcudart.so.13", "libcudart.so.12")],
)
def test_native_audit_requires_only_shared_cuda13_runtime(
    tmp_path: Path, needed: tuple[str, ...]
) -> None:
    with pytest.raises(build.AuditFailure, match=r"expected NEEDED libcudart\.so\.13"):
        build.verify_shared_cudart(
            tmp_path / "component.so", build.Elf(needed, (), None), {}, {}
        )


def test_native_dependency_audit_accepts_cuda13_math_soname12(tmp_path: Path) -> None:
    name = "libcusparse.so.12"
    site = Path(build.sysconfig.get_path("platlib"))
    provider = site / "nvidia/cu13/lib" / name
    result = build.verify_dependencies(
        tmp_path / "phydrax_cuda13/lib/component.so",
        build.Elf((name,), ("$ORIGIN/../../nvidia/cu13/lib",), None),
        tmp_path,
        {},
        build.NvidiaInventory({"nvidia-cusparse": "12.8.6.72"}, {name: provider}, {}),
        frozenset(),
    )
    assert result[name] == str(provider)


@pytest.mark.parametrize("full_release", [False, True], ids=["hopper-only", "release"])
def test_artifact_coverage_requires_every_release_architecture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, full_release: bool
) -> None:
    architectures = build.RELEASE_ARCHITECTURES if full_release else ("90-real",)
    config = build.Configuration(
        architectures,
        ("amgx",),
        Path(sys.executable),
        tmp_path,
        tmp_path / "build",
        None,
        1,
        "none",
    )
    sass = (
        "\n".join(
            f"ELF file: kernel.sm_{target}.cubin" for target in (80, 86, 89, 90, 100, 120)
        )
        if full_release
        else "ELF file: kernel.sm_90.cubin"
    )
    ptx = "PTX file: kernel.compute_120.ptx" if full_release else ""

    def inspect(
        command: Sequence[str],
        environment: Mapping[str, str],
        evidence: dict[str, build.Json],
        *,
        expected_failure: tuple[int, str] | None = None,
    ) -> str:
        del environment, evidence, expected_failure
        return sass if command[1] == "--list-elf" else ptx

    monkeypatch.setattr(build, "run", inspect)
    result = build.inspect_device_code(
        tmp_path / "libphydrax_cuda_amgx.so",
        Path("cuobjdump"),
        config,
        {},
        {},
    )
    assert result["release_architecture_coverage"] is full_release


@pytest.mark.parametrize(
    ("architecture", "container_platform", "loader"),
    [
        ("x86_64", "linux/amd64", "ld-linux-x86-64.so.2"),
        ("aarch64", "linux/arm64", "ld-linux-aarch64.so.1"),
    ],
)
@pytest.mark.parametrize("container", ["podman", "docker"])
def test_native_host_selects_matching_container(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    architecture: str,
    container_platform: str,
    loader: str,
    container: str,
) -> None:
    monkeypatch.setattr(build.platform, "machine", lambda: architecture)
    monkeypatch.setattr(build.sys, "platform", "linux")
    host = build.host_platform()
    assert host.architecture == architecture
    assert host.loader == loader
    config = build.Configuration(
        ("121-real",),
        ("cudss",),
        Path(sys.executable),
        tmp_path / "wheels",
        tmp_path / "build",
        None,
        1,
        container,
    )
    if os.name != "posix":
        with pytest.raises(build.AuditFailure, match="POSIX host"):
            build.container_command(config)
        return
    command = build.container_command(config)
    assert command[command.index("--platform") + 1] == container_platform
    assert f"quay.io/pypa/manylinux_2_28_{architecture}:latest" in command
    assert command[command.index("--architectures") + 1] == "121-real"
    assert "--gpus" not in command
    assert "--device" not in command


@pytest.mark.parametrize(
    ("system", "architecture"), [("linux", "riscv64"), ("darwin", "aarch64")]
)
def test_build_rejects_unsupported_host(
    monkeypatch: pytest.MonkeyPatch, system: str, architecture: str
) -> None:
    monkeypatch.setattr(build.platform, "machine", lambda: architecture)
    monkeypatch.setattr(build.sys, "platform", system)
    with pytest.raises(build.AuditFailure):
        build.host_platform()


@pytest.mark.parametrize(
    ("architecture", "expected_tag", "loader", "foreign_loader"),
    [
        (
            "x86_64",
            "manylinux_2_35_x86_64",
            "ld-linux-x86-64.so.2",
            "ld-linux-aarch64.so.1",
        ),
        (
            "aarch64",
            "manylinux_2_28_aarch64",
            "ld-linux-aarch64.so.1",
            "ld-linux-x86-64.so.2",
        ),
    ],
)
def test_manylinux_policy_uses_host_architecture_and_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    architecture: str,
    expected_tag: str,
    loader: str,
    foreign_loader: str,
) -> None:
    policies = [
        {
            "name": name,
            "lib_whitelist": ["libc.so.6"],
            "symbol_versions": dict.fromkeys(architectures, {}),
        }
        for name, architectures in [
            ("manylinux_2_28", ("x86_64", "aarch64")),
            ("manylinux_2_35", ("x86_64",)),
            ("manylinux_2_40", ("x86_64", "aarch64")),
        ]
    ]
    policy_path = tmp_path / "manylinux-policy.json"
    policy_path.write_text(json.dumps(policies), encoding="utf-8")

    class PolicyDistribution:
        def locate_file(self, path: str) -> Path:
            assert path == "auditwheel/policy/manylinux-policy.json"
            return policy_path

    def distribution(name: str) -> PolicyDistribution:
        assert name == "auditwheel"
        return PolicyDistribution()

    monkeypatch.setattr(build.importlib.metadata, "distribution", distribution)
    monkeypatch.setattr(build.platform, "machine", lambda: architecture)
    monkeypatch.setattr(build.platform, "libc_ver", lambda: ("glibc", "2.39"))
    monkeypatch.setattr(build.sys, "platform", "linux")
    tag, allowed = build.manylinux_policy()
    assert tag == expected_tag
    assert allowed == frozenset(("libc.so.6", loader))
    path = tmp_path / "component.so"
    elf = build.Elf((loader,), (), None)
    assert build.verify_dependencies(
        path, elf, tmp_path, {}, build.NvidiaInventory({}, {}, {}), allowed
    ) == {loader: "platform"}
    with pytest.raises(build.AuditFailure, match="unresolved/non-policy"):
        build.verify_dependencies(
            path,
            build.Elf((foreign_loader,), (), None),
            tmp_path,
            {},
            build.NvidiaInventory({}, {}, {}),
            allowed,
        )


@pytest.mark.parametrize(
    ("architecture", "machine"),
    [("x86_64", "Advanced Micro Devices X86-64"), ("aarch64", "AArch64")],
)
@pytest.mark.parametrize("defect", ["none", "machine", "class", "data"])
def test_elf_audit_requires_native_64bit_little_endian(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    architecture: str,
    machine: str,
    defect: str,
) -> None:
    monkeypatch.setattr(build.platform, "machine", lambda: architecture)
    monkeypatch.setattr(build.sys, "platform", "linux")
    header = {
        "Machine": "RISC-V" if defect == "machine" else machine,
        "Class": "ELF32" if defect == "class" else "ELF64",
        "Data": "2's complement, big endian"
        if defect == "data"
        else "2's complement, little endian",
    }

    def inspect(
        command: Sequence[str],
        environment: Mapping[str, str],
        evidence: dict[str, build.Json],
    ) -> str:
        del environment, evidence
        assert "--file-header" in command
        return "\n".join(f"  {key}: {value}" for key, value in header.items()) + (
            "\n 0x0001 (NEEDED) Shared library: [libcudart.so.13]\n"
            " 0x001d (RUNPATH) Library runpath: [$ORIGIN]\n"
        )

    monkeypatch.setattr(build, "run", inspect)
    if defect == "none":
        assert build.inspect_elf(tmp_path / "component.so", {}, {}) == build.Elf(
            ("libcudart.so.13",), ("$ORIGIN",), None
        )
    else:
        with pytest.raises(build.AuditFailure, match="expected little-endian ELF64"):
            build.inspect_elf(tmp_path / "component.so", {}, {})


def test_gb10_explicit_target_does_not_change_release_qualification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = build.parse_configuration(
        [
            "--architectures",
            "121-real",
            "--components",
            "amgx",
            "--python",
            sys.executable,
            "--wheelhouse",
            str(tmp_path),
            "--build-dir",
            str(tmp_path / "build"),
        ]
    )

    def inspect(
        command: Sequence[str],
        environment: Mapping[str, str],
        evidence: dict[str, build.Json],
        *,
        expected_failure: tuple[int, str] | None = None,
    ) -> str:
        del environment, evidence, expected_failure
        return "ELF file: kernel.sm_121.cubin" if command[1] == "--list-elf" else ""

    monkeypatch.setattr(build, "run", inspect)
    result = build.inspect_device_code(
        tmp_path / "libphydrax_cuda_amgx.so", Path("cuobjdump"), config, {}, {}
    )
    assert result["sass"] == ["121"]
    assert result["release_architecture_coverage"] is False
    assert "121-real" not in build.RELEASE_ARCHITECTURES
