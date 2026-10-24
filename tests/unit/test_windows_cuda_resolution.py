"""Offline real-uv resolution tests; metadata fixtures are not GPU binaries."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import shutil
import tarfile
import zipfile
from pathlib import Path

import pytest

from tools import check_cuda_extras as extras


def metadata_wheel(
    directory: Path,
    name: str,
    version: str,
    *,
    requirements: tuple[str, ...] = (),
    tag: str = "py3-none-any",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    normalized = name.replace("-", "_")
    prefix = f"{normalized}-{version}.dist-info"
    files = {
        f"{prefix}/METADATA": (
            f"Metadata-Version: 2.3\nName: {name}\nVersion: {version}\n"
            "Requires-Python: >=3.13,<3.14\nProvides-Extra: cuda13\n"
            "Provides-Extra: cutensornet\nProvides-Extra: cuest\n"
            + "".join(f"Requires-Dist: {item}\n" for item in requirements)
            + "\n"
        ).encode(),
        f"{prefix}/WHEEL": f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: {tag}\n".encode(),
    }
    record = io.StringIO(newline="")
    writer = csv.writer(record)
    for member, payload in files.items():
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
            .rstrip(b"=")
            .decode()
        )
        writer.writerow((member, f"sha256={digest}", len(payload)))
    writer.writerow((f"{prefix}/RECORD", "", ""))
    files[f"{prefix}/RECORD"] = record.getvalue().encode()
    path = directory / f"{normalized}-{version}-{tag}.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for member, payload in files.items():
            archive.writestr(member, payload)
    return path


def resolver_fixture(
    tmp_path: Path,
    *,
    runtime_version: str | None = "0.2.2",
    runtime_tag: str = "cp313-cp313-win_amd64",
) -> tuple[Path, Path]:
    links = tmp_path / "wheels"
    for name, version in (
        ("jax", "0.12.0"),
        ("jaxlib", "0.12.0"),
        ("phydrax-cuda-major", "13.0"),
        ("phydrax-cuda13", "0.2.2"),
        ("nvidia-cuda-runtime", "13.4.92"),
    ):
        metadata_wheel(links, name, version)
    metadata_wheel(links, "nvidia-cudss-cu13", "0.8.0")
    if runtime_version is not None:
        metadata_wheel(links, "phydrax-cuda13-pjrt", runtime_version, tag=runtime_tag)
    core = metadata_wheel(
        tmp_path / "core",
        "phydrax",
        "0.2.2",
        requirements=(
            "jax==0.12.0",
            "jaxlib==0.12.0",
            "phydrax-cuda-major==13.*",
            "phydrax-cuda13==0.2.2",
            "phydrax-cuda13-pjrt==0.2.2",
            "nvidia-cuda-runtime==13.4.92",
            "nvidia-cudss-cu13==0.8.0",
            "cutensornet-cu13==2.14.0; extra == 'cutensornet'",
            "phydrax-cuest==0.2.2; extra == 'cuest'",
        ),
    )
    return core, links


def require_uv() -> None:
    if shutil.which("uv") is None:
        pytest.skip("real binary resolver coverage requires uv")


def test_actual_windows_binary_resolution(tmp_path: Path) -> None:
    require_uv()
    core, links = resolver_fixture(tmp_path)
    result = extras.compile_case(
        core,
        (links,),
        "x86_64-pc-windows-msvc",
        "0.2.2",
        work_dir=tmp_path / "resolve",
        no_index=True,
    )
    assert result["successful"], result
    assert result["resolved"]["phydrax-cuda13-pjrt"] == "0.2.2"


@pytest.mark.parametrize(
    "version,tag",
    (
        (None, "cp313-cp313-win_amd64"),
        ("0.2.1", "cp313-cp313-win_amd64"),
        ("0.2.2", "cp313-cp313-manylinux_2_28_x86_64"),
        ("0.2.2", "cp313-cp313t-win_amd64"),
    ),
)
def test_binary_resolution_refuses_missing_mismatched_or_incompatible_runtime(
    tmp_path: Path, version: str | None, tag: str
) -> None:
    require_uv()
    core, links = resolver_fixture(tmp_path, runtime_version=version, runtime_tag=tag)
    result = extras.compile_case(
        core,
        (links,),
        "x86_64-pc-windows-msvc",
        "0.2.2",
        work_dir=tmp_path / "resolve",
        no_index=True,
        refusal="phydrax-cuda13-pjrt",
    )
    assert result["successful"], result
    assert result["command"]["returncode"] != 0


def test_unsupported_windows_architecture_refuses_instead_of_installing_marker_only(
    tmp_path: Path,
) -> None:
    require_uv()
    core, links = resolver_fixture(tmp_path)
    result = extras.compile_case(
        core,
        (links,),
        "aarch64-pc-windows-msvc",
        "0.2.2",
        work_dir=tmp_path / "resolve",
        no_index=True,
        refusal="phydrax-cuda13-pjrt",
    )
    assert result["successful"], result
    assert result["command"]["returncode"] != 0


@pytest.mark.parametrize(
    "extra,distribution",
    (("cutensornet", "cutensornet-cu13"), ("cuest", "phydrax-cuest")),
)
def test_explicit_unavailable_windows_provider_is_not_noop(
    tmp_path: Path, extra: str, distribution: str
) -> None:
    require_uv()
    core, links = resolver_fixture(tmp_path)
    result = extras.compile_case(
        core,
        (links,),
        "x86_64-pc-windows-msvc",
        "0.2.2",
        work_dir=tmp_path / "resolve",
        extras=f"cuda13,{extra}",
        no_index=True,
        refusal=distribution,
    )
    assert result["successful"], result
    assert result["command"]["returncode"] != 0


def test_resolver_refuses_competing_owner_even_when_solver_accepts(
    tmp_path: Path,
) -> None:
    require_uv()
    core, links = resolver_fixture(tmp_path)
    metadata_wheel(links, "jax-cuda13-plugin", "0.12.0")
    result = extras.compile_case(
        core,
        (links,),
        "x86_64-pc-windows-msvc",
        "0.2.2",
        work_dir=tmp_path / "resolve",
        no_index=True,
        constraint="jax-cuda13-plugin==0.12.0",
    )
    assert result["command"]["returncode"] == 0
    assert not result["successful"]
    assert any("sole Windows CUDA owner" in error for error in result["errors"])


def test_binary_resolution_never_builds_runtime_sdist(tmp_path: Path) -> None:
    require_uv()
    core, links = resolver_fixture(tmp_path, runtime_version=None)
    # A genuine source archive is available, but only a compatible wheel can
    # satisfy this release gate. No backend or compiler may be invoked.
    payload = (
        b"[build-system]\nrequires = []\nbuild-backend = 'forbidden_source_backend'\n"
    )
    with tarfile.open(links / "phydrax_cuda13_pjrt-0.2.2.tar.gz", "w:gz") as archive:
        member = tarfile.TarInfo("phydrax_cuda13_pjrt-0.2.2/pyproject.toml")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    result = extras.compile_case(
        core,
        (links,),
        "x86_64-pc-windows-msvc",
        "0.2.2",
        work_dir=tmp_path / "resolve",
        no_index=True,
        refusal="phydrax-cuda13-pjrt",
    )
    assert result["successful"], result
    assert result["command"]["returncode"] != 0
    assert "forbidden_source_backend" not in result["command"]["stderr"]
