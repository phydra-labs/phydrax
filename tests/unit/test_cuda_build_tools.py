"""CUDA 13 build and resolver gates refuse incompatible runtime artifacts."""

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from tools import build_cuda_wheels as build, check_cuda_extras as extras


@pytest.fixture
def linux_resolution() -> dict[str, str]:
    return extras.resolved_distributions(
        """
        phydrax-cuda-major==13.0
        phydrax-cuda13==0.2.2
        jax-cuda13-plugin==0.11.2
        jax-cuda13-pjrt==0.11.2
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
    if architecture == "aarch64":
        linux_resolution.pop("phydrax-cuda13")
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


@pytest.mark.parametrize("platform", ["aarch64-apple-darwin", "x86_64-pc-windows-msvc"])
def test_non_linux_cuda_intent_requires_no_gpu_packages(platform: str) -> None:
    resolved = {"phydrax-cuda-major": "13.0"}
    assert not extras.resolution_errors(resolved, platform, "0.2.2")
    resolved["nvidia-cuda-runtime"] = "13.4.92"
    assert any(
        "Non-Linux resolution includes" in error
        for error in extras.resolution_errors(resolved, platform, "0.2.2")
    )


def test_arm_linux_refuses_native_companion(linux_resolution: dict[str, str]) -> None:
    assert any(
        "Companion resolved off Linux x86_64" in error
        for error in extras.resolution_errors(
            linux_resolution, "aarch64-manylinux_2_28", "0.2.2"
        )
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
