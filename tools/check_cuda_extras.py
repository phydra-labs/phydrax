#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Qualify published CUDA-extra metadata against live package-index resolutions."""

from __future__ import annotations

import argparse
import email.policy
import json
import os
import re
import sys
import tempfile
import zipfile
from datetime import datetime, UTC
from email.message import Message
from email.parser import BytesParser
from pathlib import Path
from typing import TypedDict

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from tools._accelerator_process import CommandEvidence, run as run_bounded
from tools.qualify_cuda_wheels import validate_work_path


ROOT = Path(__file__).resolve().parents[1]
# JAX >=0.11 publishes no x86_64 macOS jaxlib, so Phydrax does not support it.
# The Linux glibc tag is substituted from --linux-tag (release wheels: manylinux_2_28).
PLATFORMS = (
    "x86_64-{linux}",
    "aarch64-{linux}",
    "aarch64-apple-darwin",
    "x86_64-pc-windows-msvc",
)
# CUDA 13 math-library distribution versions are not CUDA-major identifiers.
NVIDIA_FAMILIES = (
    "cuda-runtime",
    "cublas",
    "cusparse",
    "cusolver",
    "cufft",
    "curand",
    "nvjitlink",
    "cuda-nvrtc",
    "cuda-cupti",
    "cudnn",
    "nccl",
    "nvshmem",
    "cudss",
    "cuest",
    "cuda-cccl",
    "cuda-crt",
    "cuda-nvcc",
    "nvvm",
)
UNSUFFIXED_CUDA13 = frozenset(
    f"nvidia-{family}"
    for family in (
        "cuda-runtime",
        "cublas",
        "cusparse",
        "cusolver",
        "cufft",
        "curand",
        "nvjitlink",
        "cuda-nvrtc",
        "cuda-cupti",
        "cuda-cccl",
        "cuda-crt",
        "cuda-nvcc",
        "nvvm",
    )
)
NVIDIA_MAJORS = {
    **{
        f"nvidia-{family}-cu{major}": major
        for family in NVIDIA_FAMILIES
        for major in (12, 13)
    },
    **dict.fromkeys(UNSUFFIXED_CUDA13, 13),
    **{
        f"{family}-cu{major}": major
        for family in ("cutensor", "cutensornet")
        for major in (12, 13)
    },
    "phydrax-cuest": 13,
}


class CaseEvidence(TypedDict):
    platform: str
    extras: str
    expected: str
    resolved: dict[str, str]
    successful: bool
    errors: list[str]
    resolved_nvidia: dict[str, str]
    resolved_nvidia_majors: dict[str, int]
    command: CommandEvidence


class Evidence(TypedDict):
    timestamp: str
    completed_at: str
    python: str
    uv: str
    builds: list[CommandEvidence]
    metadata_errors: list[str]
    cases: list[CaseEvidence]
    errors: list[str]
    successful: bool


def run(
    command: list[str], *, input_text: str | None = None, cwd: Path = ROOT
) -> CommandEvidence:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("UV_") and key not in ("PYTHONPATH", "PYTHONHOME")
    }
    for key, directory in (
        ("UV_CACHE_DIR", "uv-cache"),
        ("UV_PYTHON_INSTALL_DIR", "uv-python"),
        ("TEMP", "tmp"),
        ("TMP", "tmp"),
        ("TMPDIR", "tmp"),
        ("XDG_CACHE_HOME", "cache"),
    ):
        path = validate_work_path(cwd / directory, cwd)
        path.mkdir(parents=True, exist_ok=True)
        environment[key] = str(path)
    environment["UV_PYTHON_DOWNLOADS"] = "never"
    if input_text is None:
        return run_bounded(command, cwd=cwd, environment=environment, timeout=900)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".in", dir=cwd, encoding="utf-8", delete=False
    ) as stream:
        stream.write(input_text)
        requirements = Path(stream.name)
    invocation = [str(requirements) if item == "-" else item for item in command]
    return run_bounded(invocation, cwd=cwd, environment=environment, timeout=900)


def build(
    source: Path, destination: Path, evidence: Evidence, *, python: Path
) -> Path | None:
    destination.mkdir(parents=True, exist_ok=True)
    result = run(
        [
            "uv",
            "build",
            "--no-config",
            "--python",
            str(python),
            "--wheel",
            "--out-dir",
            str(destination),
            str(source),
        ],
        cwd=destination,
    )
    evidence["builds"].append(result)
    if result["returncode"]:
        evidence["errors"].append(f"Wheel build failed for {source}: {result['stderr']}")
        return None
    wheels = tuple(destination.glob("*.whl"))
    if len(wheels) != 1:
        evidence["errors"].append(
            f"Expected exactly one wheel in {destination}: {wheels}"
        )
        return None
    return wheels[0]


def wheel_metadata(wheel: Path) -> Message | str:
    with zipfile.ZipFile(wheel) as archive:
        members = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(members) != 1:
            return f"Expected one wheel METADATA, found {members}"
        return BytesParser(policy=email.policy.default).parsebytes(
            archive.read(members[0])
        )


def metadata_errors(metadata: Message) -> list[str]:
    """Evaluate published requirements, rather than their TOML spelling."""
    extras = set(metadata.get_all("Provides-Extra", []))
    version = str(metadata["Version"])
    errors: list[str] = []
    for extra in sorted(extras):
        if extra == "cudss" or (extra.startswith("cuda") and extra != "cuda13"):
            errors.append(f"Unsupported {extra} extra is still published")
    for extra in ("cuda13", "cutensor", "cutensornet", "cuest"):
        if extra not in extras:
            errors.append(f"Missing {extra} extra")
    requirements: list[Requirement] = []
    for value in metadata.get_all("Requires-Dist", []):
        try:
            requirements.append(Requirement(str(value)))
        except InvalidRequirement as error:
            errors.append(f"Invalid Requires-Dist: {error}")
    for host, machine in (
        ("linux", "x86_64"),
        ("linux", "aarch64"),
        ("win32", "AMD64"),
        ("win32", "ARM64"),
        ("darwin", "arm64"),
    ):
        requires_stack = host in ("linux", "win32")
        for extra in ("cuda13", "cutensor", "cutensornet", "cuest"):
            environment = {
                **default_environment(),
                "sys_platform": host,
                "platform_system": {
                    "linux": "Linux",
                    "win32": "Windows",
                    "darwin": "Darwin",
                }[host],
                "platform_machine": machine,
                "python_version": "3.13",
                "python_full_version": "3.13.0",
                "extra": extra,
            }
            selected = [
                item
                for item in requirements
                if item.marker is None or item.marker.evaluate(environment)
            ]
            expected: dict[str, str] = {}
            if extra in ("cuda13", "cuest"):
                expected["phydrax-cuda-major"] = "==13.*"
            if extra == "cuda13" and requires_stack:
                expected.update(
                    {
                        "jax": "==0.12.0",
                        "jaxlib": "==0.12.0",
                        "phydrax-cuda13": f"=={version}",
                    }
                )
                if host == "win32":
                    expected["phydrax-cuda13-pjrt"] = f"=={version}"
            if extra in ("cutensor", "cutensornet") and requires_stack:
                expected["cutensor-cu13"] = "==2.8.1"
            if extra == "cutensornet" and host in ("linux", "win32"):
                expected["cutensornet-cu13"] = "==2.14.0"
            if extra == "cuest" and host in ("linux", "win32"):
                expected["phydrax-cuest"] = f"=={version}"
            for name, specifier in expected.items():
                matching = [
                    item for item in selected if canonicalize_name(item.name) == name
                ]
                if not any(
                    str(item.specifier) == specifier and item.url is None
                    for item in matching
                ):
                    errors.append(
                        f"{host}/{machine} [{extra}]: expected {name}{specifier}; marker requirements do not select it"
                    )
            cuda_owned = {
                canonicalize_name(item.name)
                for item in selected
                if canonicalize_name(item.name).startswith(("phydrax-cuda", "cutensor"))
                or canonicalize_name(item.name) == "phydrax-cuest"
            }
            unexpected = cuda_owned - expected.keys()
            if unexpected:
                errors.append(
                    f"{host}/{machine} [{extra}]: unexpected accelerator requirements {sorted(unexpected)}"
                )
            if extra == "cuda13" and requires_stack:
                jax = [
                    item
                    for item in selected
                    if canonicalize_name(item.name) == "jax"
                    and str(item.specifier) == "==0.12.0"
                ]
                if host == "linux" and not any("cuda13" in item.extras for item in jax):
                    errors.append(f"{host}/{machine}: missing upstream jax[cuda13]")
                if host == "win32" and any(item.extras for item in jax):
                    errors.append(
                        "Windows requires stock JAX without upstream CUDA extras"
                    )
    return errors


def resolved_distributions(output: str) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for line in output.splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)(?:\s*;.*)?", line.strip())
        if match is not None:
            name = re.sub(r"[-_.]+", "-", match[1]).lower()
            resolved[name] = match[2]
    return dict(sorted(resolved.items()))


def resolution_errors(resolved: dict[str, str], platform: str, version: str) -> list[str]:
    errors: list[str] = []
    if resolved.get("phydrax-cuda-major", "").split(".")[0] != "13":
        errors.append("Missing or incorrect CUDA 13 marker")
    nvidia = {
        name: version
        for name, version in resolved.items()
        if name.startswith("nvidia-")
        or name == "phydrax-cuest"
        or re.fullmatch(r"(?:cutensor|cutensornet)-cu\d+", name)
    }
    plugins = [
        name
        for name in resolved
        if re.fullmatch(r"(?:jax|phydrax)-cuda\d+-(plugin|pjrt)", name)
        or "winjax" in name
    ]
    companions = {
        name: version
        for name, version in resolved.items()
        if re.fullmatch(r"phydrax-cuda\d+", name)
    }
    windows = platform == "x86_64-pc-windows-msvc"
    if "manylinux" not in platform and not windows:
        if nvidia or plugins or companions:
            errors.append(
                "Unsupported-host resolution includes NVIDIA/JAX CUDA/companion packages: "
                f"{nvidia}, {plugins}, {companions}"
            )
        return errors
    if platform.startswith(("x86_64-", "aarch64-")):
        if companions != {"phydrax-cuda13": version}:
            errors.append(f"Expected phydrax-cuda13=={version}, resolved {companions}")
    elif companions:
        errors.append(
            f"Companion resolved off supported Linux architectures: {companions}"
        )
    if "phydrax-cuest" in resolved and resolved["phydrax-cuest"] != version:
        errors.append(
            f"Expected phydrax-cuest=={version}, resolved {resolved['phydrax-cuest']}"
        )
    if windows:
        if plugins != ["phydrax-cuda13-pjrt"]:
            errors.append(
                f"Expected sole Windows CUDA owner phydrax-cuda13-pjrt, resolved {plugins}"
            )
        if resolved.get("phydrax-cuda13-pjrt") != version:
            errors.append(f"Expected phydrax-cuda13-pjrt=={version}")
        for name in ("jax", "jaxlib"):
            if resolved.get(name) != "0.12.0":
                errors.append(
                    f"Expected stock {name}==0.12.0, resolved {resolved.get(name)}"
                )
        for name in (
            "nvidia-nccl-cu13",
            "nvidia-nvshmem-cu13",
            "cutensornet-cu13",
            "phydrax-cuest",
        ):
            if name in resolved:
                errors.append(f"Unsupported Windows runtime: {name}")
    elif set(plugins) != {"jax-cuda13-plugin", "jax-cuda13-pjrt"}:
        errors.append(f"Expected CUDA 13 JAX plugin and PJRT, resolved {plugins}")
    for name in nvidia:
        declared_major = NVIDIA_MAJORS.get(name)
        if declared_major is None:
            errors.append(f"Unclassified NVIDIA distribution: {name}")
        elif declared_major != 13:
            errors.append(f"Foreign CUDA {declared_major} distribution: {name}")
    if "nvidia-cudss-cu13" not in nvidia:
        errors.append("Missing mandatory nvidia-cudss-cu13 runtime")
    runtime = "nvidia-cuda-runtime"
    if runtime not in nvidia:
        errors.append(f"Missing {runtime}")
    elif nvidia[runtime].split(".")[0] != "13":
        errors.append(f"Expected CUDA 13 runtime version, found {nvidia[runtime]}")
    return errors


def compile_case(
    wheel: Path,
    find_links: tuple[Path, ...],
    platform: str,
    version: str,
    *,
    work_dir: Path,
    extras: str = "cuda13",
    refusal: str | None = None,
    constraint: str | None = None,
    no_index: bool = False,
) -> CaseEvidence:
    work_dir.mkdir(parents=True, exist_ok=True)
    output = work_dir / "resolved.txt"
    if output.exists():
        raise FileExistsError(f"Resolver evidence already exists: {output}")
    command = run(
        [
            "uv",
            "pip",
            "compile",
            "--no-config",
            "--no-sources",
            "--only-binary",
            ":all:",
            "--output-file",
            str(output),
            "--python-platform",
            platform,
            "--python-version",
            "3.13",
            *(("--no-index",) if no_index else ()),
            *(
                argument
                for link in find_links
                for argument in ("--find-links", str(link))
            ),
            "--no-header",
            "--no-annotate",
            "-",
        ],
        input_text=f"phydrax[{extras}] @ {wheel.as_uri()}\n"
        + ("" if constraint is None else constraint + "\n"),
        cwd=work_dir,
    )
    errors: list[str] = []
    resolved = (
        resolved_distributions(output.read_text(encoding="utf-8"))
        if output.is_file()
        else {}
    )
    if command["timed_out"]:
        errors.append("Resolver timed out; refusal was not established")
    elif refusal is not None:
        if not command["returncode"]:
            errors.append(f"Expected refusal naming {refusal}, but resolution succeeded")
        elif refusal.lower() not in command["stderr"].lower() or not any(
            token in command["stderr"].lower()
            for token in (
                "no solution",
                "unsatisfiable",
                "no versions",
                "not found",
                "no wheels",
                "no matching",
            )
        ):
            errors.append(
                f"Resolver failed without establishing the requested {refusal} refusal"
            )
    elif command["returncode"]:
        errors.append(f"CUDA 13 resolution failed: {command['stderr']}")
    else:
        errors.extend(resolution_errors(resolved, platform, version))
    nvidia = {
        name: version for name, version in resolved.items() if name.startswith("nvidia-")
    }
    return {
        "platform": platform,
        "extras": extras,
        "expected": "resolved" if refusal is None else f"refused:{refusal}",
        "resolved": resolved,
        "successful": not errors,
        "errors": errors,
        "resolved_nvidia": nvidia,
        "resolved_nvidia_majors": {
            name: NVIDIA_MAJORS[name] for name in nvidia if name in NVIDIA_MAJORS
        },
        "command": command,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python", type=Path, required=True, help="Explicit build interpreter"
    )
    parser.add_argument(
        "--phydrax-wheel",
        type=Path,
        help="Use an existing core wheel instead of building source",
    )
    parser.add_argument(
        "--marker-wheel",
        type=Path,
        help="Existing CUDA-major marker wheel; required with --phydrax-wheel",
    )
    parser.add_argument(
        "--work-dir", type=Path, required=True, help="Scratch wheel directory"
    )
    parser.add_argument(
        "--evidence", type=Path, required=True, help="JSON evidence output"
    )
    parser.add_argument(
        "--companion-wheelhouse",
        type=Path,
        required=True,
        help="Directory holding the phydrax-cuda13 wheel until it is on PyPI",
    )
    parser.add_argument(
        "--linux-tag",
        default="manylinux_2_28",
        help="Linux platform tag of the companion wheels (release: manylinux_2_28)",
    )
    args = parser.parse_args()
    if (args.phydrax_wheel is None) != (args.marker_wheel is None):
        parser.error("--phydrax-wheel and --marker-wheel must be supplied together")
    root = (
        Path.home()
        if sys.platform == "win32"
        else Path(Path(args.work_dir).absolute().anchor)
    )
    work_dir = validate_work_path(args.work_dir, root)
    evidence_path = validate_work_path(args.evidence, root)
    wheelhouse = validate_work_path(args.companion_wheelhouse, root)
    python = validate_work_path(args.python, root)
    if not python.is_file():
        parser.error("--python must name an existing interpreter")
    platforms = tuple(platform.format(linux=args.linux_tag) for platform in PLATFORMS)
    work_dir.mkdir(parents=True, exist_ok=True)
    version = run(["uv", "--version"], cwd=work_dir)
    evidence: Evidence = {
        "timestamp": datetime.now(UTC).isoformat(),
        "completed_at": "",
        "python": sys.version,
        "uv": version["stdout"].strip(),
        "builds": [],
        "metadata_errors": [],
        "cases": [],
        "errors": [],
        "successful": False,
    }
    if not tuple(wheelhouse.glob("phydrax_cuda13-*.whl")):
        evidence["errors"].append(f"No phydrax-cuda13 wheel in {wheelhouse}")
    if not tuple(wheelhouse.glob("phydrax_cuda13_pjrt-*-cp313-cp313-win_amd64.whl")):
        evidence["errors"].append(
            f"No CPython 3.13 Windows phydrax-cuda13-pjrt wheel in {wheelhouse}"
        )
    if args.phydrax_wheel is None:
        marker_dir = work_dir / "markers" / "13"
        marker = build(
            ROOT / "packaging/cuda-major/13", marker_dir, evidence, python=python
        )
        wheel = build(ROOT, work_dir / "phydrax", evidence, python=python)
    else:
        wheel = validate_work_path(args.phydrax_wheel, root)
        marker = validate_work_path(args.marker_wheel, root)
        for artifact in (wheel, marker):
            if not artifact.is_file() or artifact.suffix != ".whl":
                parser.error(f"Expected an existing binary wheel: {artifact}")
        marker_dir = marker.parent
        marker_metadata = wheel_metadata(marker)
        if isinstance(marker_metadata, str) or (
            canonicalize_name(str(marker_metadata["Name"])) != "phydrax-cuda-major"
            or not str(marker_metadata["Version"]).startswith("13.")
        ):
            evidence["errors"].append(
                "Supplied marker wheel is not phydrax-cuda-major 13"
            )
    metadata = None if wheel is None else wheel_metadata(wheel)
    if isinstance(metadata, str):
        evidence["errors"].append(metadata)
    elif (
        wheel is not None
        and metadata is not None
        and not evidence["errors"]
        and marker is not None
    ):
        evidence["metadata_errors"] = metadata_errors(metadata)
        find_links = (marker_dir, wheelhouse)
        for platform in platforms:
            case = compile_case(
                wheel,
                find_links,
                platform,
                str(metadata["Version"]),
                work_dir=work_dir / platform,
            )
            evidence["cases"].append(case)
            print(
                f"{platform} [cuda13]: {'PASS' if case['successful'] else 'FAIL'}",
                flush=True,
            )
        windows = "x86_64-pc-windows-msvc"
        for extra, refusal in (
            ("cutensor", None),
            ("cutensornet", "cutensornet-cu13"),
            ("cuest", "phydrax-cuest"),
        ):
            evidence["cases"].append(
                compile_case(
                    wheel,
                    find_links,
                    windows,
                    str(metadata["Version"]),
                    work_dir=work_dir / f"windows-{extra}",
                    extras=f"cuda13,{extra}",
                    refusal=refusal,
                )
            )
        for distribution in ("jax", "jaxlib", "phydrax-cuda13", "phydrax-cuda13-pjrt"):
            evidence["cases"].append(
                compile_case(
                    wheel,
                    find_links,
                    windows,
                    str(metadata["Version"]),
                    work_dir=work_dir / f"windows-mismatch-{distribution}",
                    refusal=distribution,
                    constraint=f"{distribution}==0.0.0",
                )
            )
    evidence["successful"] = (
        not evidence["errors"]
        and not evidence["metadata_errors"]
        and len(evidence["cases"]) == len(platforms) + 7
        and all(case["successful"] for case in evidence["cases"])
    )
    evidence["completed_at"] = datetime.now(UTC).isoformat()
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for error in [*evidence["errors"], *evidence["metadata_errors"]]:
        print(error, file=sys.stderr)
    print(f"Evidence: {evidence_path}")
    return 0 if evidence["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
