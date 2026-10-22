#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Qualify published CUDA-extra metadata against live package-index resolutions."""

from __future__ import annotations

import argparse
import email.policy
import json
import re
import subprocess
import sys
import zipfile
from datetime import datetime, UTC
from email.message import Message
from email.parser import BytesParser
from pathlib import Path
from typing import TypedDict


ROOT = Path(__file__).resolve().parents[1]
# JAX >=0.11 publishes no x86_64 macOS jaxlib, so Phydrax does not support it.
# The Linux glibc tag is substituted from --linux-tag (release wheels: manylinux_2_28).
PLATFORMS = (
    "x86_64-{linux}",
    "aarch64-{linux}",
    "aarch64-apple-darwin",
    "x86_64-pc-windows-msvc",
)
COMPANION_MARKER = (
    "sys_platform=='linux'and(platform_machine=='x86_64'orplatform_machine=='aarch64')"
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


class CommandEvidence(TypedDict):
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


class CaseEvidence(TypedDict):
    platform: str
    extras: str
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


def run(command: list[str], *, input_text: str | None = None) -> CommandEvidence:
    result = subprocess.run(
        command,
        cwd=ROOT,
        input=input_text,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def build(source: Path, destination: Path, evidence: Evidence) -> Path | None:
    destination.mkdir(parents=True, exist_ok=True)
    result = run(
        [
            "uv",
            "build",
            "--wheel",
            "--out-dir",
            str(destination),
            str(source),
        ]
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


def _normalized(requirement: str) -> str:
    return re.sub(r"\s+", "", requirement).replace('"', "'")


def metadata_errors(metadata: Message) -> list[str]:
    extras = set(metadata.get_all("Provides-Extra", []))
    version = str(metadata["Version"])
    errors: list[str] = []
    for extra in sorted(extras):
        if extra == "cudss" or (extra.startswith("cuda") and extra != "cuda13"):
            errors.append(f"Unsupported {extra} extra is still published")
    if "cuda13" not in extras:
        errors.append("Missing cuda13 extra")
    requirements = metadata.get_all("Requires-Dist", [])
    # CUDA intent is platform-independent for both the complete CUDA stack and
    # the separately selectable cuEST runtime. The latter also depends on the
    # marker transitively; keeping the direct requirement makes standalone
    # resolver intent explicit and lets universal local locks resolve it.
    expected = {
        "phydrax-cuda-major==13.*;extra=='cuda13'",
        "phydrax-cuda-major==13.*;extra=='cuest'",
    }
    matching = {
        _normalized(requirement)
        for requirement in requirements
        if requirement.lower().startswith("phydrax-cuda-major")
    }
    if matching != expected:
        errors.append(
            f"Expected CUDA 13 marker requirements {sorted(expected)}: {sorted(matching)}"
        )
    companion = [
        _normalized(requirement)
        for requirement in requirements
        if re.match(r"phydrax-cuda\d", requirement.lower())
    ]
    expected_companion = (
        f"phydrax-cuda13=={version};({COMPANION_MARKER})andextra=='cuda13'"
    )
    if companion != [expected_companion]:
        errors.append(
            f"Expected lockstep companion requirement {expected_companion}: {companion}"
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
        name for name in resolved if re.fullmatch(r"jax-cuda\d+-(plugin|pjrt)", name)
    ]
    companions = {
        name: version
        for name, version in resolved.items()
        if re.fullmatch(r"phydrax-cuda\d+", name)
    }
    if "manylinux" not in platform:
        if nvidia or plugins or companions:
            errors.append(
                "Non-Linux resolution includes NVIDIA/JAX CUDA/companion packages: "
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
    if set(plugins) != {"jax-cuda13-plugin", "jax-cuda13-pjrt"}:
        errors.append(f"Expected CUDA 13 JAX plugin and PJRT, resolved {plugins}")
    for name in nvidia:
        declared_major = NVIDIA_MAJORS.get(name)
        if declared_major is None:
            errors.append(f"Unclassified NVIDIA distribution: {name}")
        elif declared_major != 13:
            errors.append(f"Foreign CUDA {declared_major} distribution: {name}")
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
) -> CaseEvidence:
    command = run(
        [
            "uv",
            "pip",
            "compile",
            "--no-config",
            "--python-platform",
            platform,
            "--python-version",
            "3.13",
            *(
                argument
                for link in find_links
                for argument in ("--find-links", str(link))
            ),
            "--no-header",
            "--no-annotate",
            "-",
        ],
        input_text=f"phydrax[cuda13] @ {wheel.as_uri()}\n",
    )
    errors: list[str] = []
    resolved = resolved_distributions(command["stdout"])
    if command["returncode"]:
        errors.append(f"CUDA 13 resolution failed: {command['stderr']}")
    else:
        errors.extend(resolution_errors(resolved, platform, version))
    nvidia = {
        name: version for name, version in resolved.items() if name.startswith("nvidia-")
    }
    return {
        "platform": platform,
        "extras": "cuda13",
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
    work_dir: Path = args.work_dir.resolve()
    evidence_path: Path = args.evidence.resolve()
    wheelhouse: Path = args.companion_wheelhouse.resolve()
    platforms = tuple(platform.format(linux=args.linux_tag) for platform in PLATFORMS)
    version = run(["uv", "--version"])
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
    marker_dir = work_dir / "markers" / "13"
    marker = build(ROOT / "packaging/cuda-major/13", marker_dir, evidence)
    wheel = build(ROOT, work_dir / "phydrax", evidence)
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
            case = compile_case(wheel, find_links, platform, str(metadata["Version"]))
            evidence["cases"].append(case)
            print(
                f"{platform} [cuda13]: {'PASS' if case['successful'] else 'FAIL'}",
                flush=True,
            )
    evidence["successful"] = (
        not evidence["errors"]
        and not evidence["metadata_errors"]
        and len(evidence["cases"]) == len(platforms)
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
