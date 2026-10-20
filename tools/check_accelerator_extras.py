# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Inspect upstream accelerator extras and resolve their cross-platform matrix.

No JAX or accelerator provider is imported. Run against an actual built core
wheel; unsupported hosts must refuse rather than resolve an empty GPU extra.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from email.message import Message
from enum import StrEnum
from pathlib import Path
from typing import Any

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from tools._accelerator_process import run
from tools.check_cuda_extras import resolved_distributions, wheel_metadata


class Accelerator(StrEnum):
    """Closed, shared selector for the two upstream accelerator installations."""

    ONEAPI = "oneapi"
    TPU = "tpu"

    @property
    def distributions(self) -> dict[str, str]:
        match self:
            case Accelerator.ONEAPI:
                return {
                    "jax": "0.12.0",
                    "jaxlib": "0.11.2",
                    "jax-oneapi-plugin": "0.11.2",
                    "jax-oneapi-pjrt": "0.11.2",
                }
            case Accelerator.TPU:
                return {"jax": "0.12.0", "jaxlib": "0.12.0", "libtpu": "0.0.50"}
        raise ValueError(f"Unsupported accelerator {self!r}")

    @property
    def requirement(self) -> str:
        match self:
            case Accelerator.ONEAPI:
                return "jax-oneapi-plugin[with-oneapi]==0.11.2"
            case Accelerator.TPU:
                return "jax[tpu]==0.12.0"
        raise ValueError(f"Unsupported accelerator {self!r}")


@dataclass(frozen=True, slots=True)
class ResolutionCase:
    name: str
    platform: str
    python: str
    extra: Accelerator | None
    expected: bool
    conflict: str | None = None
    constraint: str | None = None


def resolution_matrix() -> tuple[ResolutionCase, ...]:
    cases: list[ResolutionCase] = []
    for accelerator in Accelerator:
        for python in ("3.13",):
            cases.append(
                ResolutionCase(
                    f"{accelerator}-linux",
                    "x86_64-manylinux_2_31",
                    python,
                    accelerator,
                    True,
                )
            )
            for label, target in (
                ("arm", "aarch64-manylinux_2_31"),
                ("mac", "aarch64-apple-darwin"),
                ("windows", "x86_64-pc-windows-msvc"),
            ):
                cases.append(
                    ResolutionCase(
                        f"{accelerator}-{label}", target, python, accelerator, False
                    )
                )
    cases.append(ResolutionCase("base-cpu", "x86_64-manylinux_2_31", "3.13", None, True))
    cases.append(
        ResolutionCase(
            "tpu-old-glibc", "x86_64-manylinux_2_28", "3.13", Accelerator.TPU, False
        )
    )
    for other in ("tpu", "cuda13"):
        cases.append(
            ResolutionCase(
                f"oneapi-conflicts-{other}",
                "x86_64-manylinux_2_31",
                "3.13",
                Accelerator.ONEAPI,
                False,
                other,
            )
        )
    cases.append(
        ResolutionCase(
            "oneapi-conflicts-jaxlib-0.12.0",
            "x86_64-manylinux_2_31",
            "3.13",
            Accelerator.ONEAPI,
            False,
            constraint="jaxlib==0.12.0",
        )
    )
    return tuple(cases)


def metadata_errors(metadata: Message) -> list[str]:
    """Require provider intent on every host, including unsupported hosts."""
    errors: list[str] = []
    extras = set(metadata.get_all("Provides-Extra", []))
    requirements = [
        Requirement(str(text)) for text in metadata.get_all("Requires-Dist", [])
    ]
    for accelerator in Accelerator:
        if accelerator.value not in extras:
            errors.append(f"Missing {accelerator} extra")
        expected = Requirement(accelerator.requirement)
        for system, machine in (
            ("Linux", "x86_64"),
            ("Linux", "aarch64"),
            ("Darwin", "arm64"),
            ("Windows", "AMD64"),
        ):
            environment = {
                **default_environment(),
                "extra": accelerator.value,
                "platform_system": system,
                "sys_platform": {
                    "Linux": "linux",
                    "Darwin": "darwin",
                    "Windows": "win32",
                }[system],
                "platform_machine": machine,
                "python_version": "3.13",
                "python_full_version": "3.13.0",
            }
            active = [
                req
                for req in requirements
                if req.marker is None or req.marker.evaluate(environment)
            ]
            provider = [
                req
                for req in active
                if canonicalize_name(req.name) == canonicalize_name(expected.name)
                and expected.extras <= req.extras
                and str(req.specifier) == str(expected.specifier)
            ]
            if not provider:
                errors.append(
                    f"{accelerator}: missing {expected} on {system}/{machine}; an empty extra is forbidden"
                )
            for name, version in accelerator.distributions.items():
                direct = [req for req in active if canonicalize_name(req.name) == name]
                if any(
                    not req.specifier.contains(version, prereleases=True)
                    for req in direct
                ):
                    errors.append(
                        f"{accelerator}: direct {name} requirement excludes {version}"
                    )
    if any(
        canonicalize_name(req.name) in {"phydrax-oneapi", "phydrax-tpu"}
        for req in requirements
    ):
        errors.append("Upstream runtimes must not claim owned native companions")
    return errors


def resolution_errors(
    resolved: dict[str, str], accelerator: Accelerator | None
) -> list[str]:
    if accelerator is None:
        providers = {"libtpu", "jax-oneapi-plugin", "jax-oneapi-pjrt"} & resolved.keys()
        return (
            [f"Base CPU install includes optional runtimes: {sorted(providers)}"]
            if providers
            else []
        )
    return [
        f"{name}: expected {version}, resolved {resolved.get(name)!r}"
        for name, version in accelerator.distributions.items()
        if resolved.get(name) != version
    ]


def refusal_errors(case: ResolutionCase, stderr: str) -> list[str]:
    """Require the specific provider/platform or selected JAXlib contradiction.

    A solver's generic unsatisfiable status is not proof of the intended refusal.
    Registry omissions, unrelated dependency failures, and operational failures
    must never count as accelerator compatibility evidence.
    """
    text = " ".join(stderr.lower().split())
    operational_failures = (
        "failed to fetch",
        "failed to download",
        "failed to build",
        "failed to prepare",
        "failed to connect",
        "failed to query",
        "failed to read",
        "error sending request",
        "connection refused",
        "connection reset",
        "connection closed",
        "network is unreachable",
        "network failure",
        "transport error",
        "timed out",
        "timeout",
        "certificate",
        "tls error",
        "dns error",
        "name resolution",
        "failed to spawn",
        "command not found",
        "unrecognized argument",
        "invalid value",
        "permission denied",
        "no such file or directory",
        "was not found in the package registry",
        "was not found in the registry",
        "index error",
        "index failed",
        "index is unavailable",
        "offline mode",
        "server error",
        "failed to authenticate",
        "authentication failed",
        "unauthorized",
        "forbidden",
        "could not fetch",
        "unable to access",
    )
    if any(failure in text for failure in operational_failures) or re.search(
        r"(?:http|status(?: code)?)\s*(?:error\s*)?[45]\d\d\b", text
    ):
        return [
            "Operational/index/build failure is not accelerator incompatibility evidence"
        ]
    if "no solution found" not in text or "requirements are unsatisfiable" not in text:
        return ["Missing complete resolver incompatibility derivation"]
    oneapi_pin = re.search(r"phydrax\[oneapi\]==\S+ depends on jaxlib==0\.11\.2\b", text)
    if case.constraint == "jaxlib==0.12.0":
        if oneapi_pin and "you require jaxlib==0.12.0" in text:
            return []
        return ["Missing oneAPI JAXlib 0.11.2 versus explicit JAXlib 0.12.0 derivation"]
    if case.conflict in {"tpu", "cuda13"}:
        other = re.escape(case.conflict)
        other_pin = re.search(
            rf"jax\[{other}\]==0\.12\.0 depends on jaxlib==0\.12\.0\b", text
        )
        extra_link = re.search(
            rf"phydrax\[{other}\]==\S+ depends on jax\[{other}\](?:==|>=)0\.12\.0\b",
            text,
        )
        if oneapi_pin and other_pin and extra_link and "incompatible" in text:
            return []
        return [
            f"Missing oneAPI/{case.conflict} contradictory upstream JAXlib derivation"
        ]
    if case.conflict is not None or case.constraint is not None or case.extra is None:
        return ["This negative case does not declare a supported incompatibility proof"]
    match case.extra:
        case Accelerator.ONEAPI:
            provider, version = "jax-oneapi-plugin", "0.11.2"
        case Accelerator.TPU:
            provider, version = "libtpu", "0.0.50"
    platform_tags = {
        "aarch64-manylinux_2_31": r"manylinux_2_31_aarch64",
        "aarch64-apple-darwin": r"macosx_\d+_\d+_arm64",
        "x86_64-pc-windows-msvc": r"win_amd64",
        "x86_64-manylinux_2_28": r"manylinux_2_28_x86_64",
    }
    target_tag = platform_tags.get(case.platform)
    if target_tag and re.search(
        rf"\bbecause {re.escape(provider)}(?:\[[^\]]+\])?=={re.escape(version)} has no wheels with a matching platform tag \(e\.g\., `?{target_tag}`?\)",
        text,
    ):
        return []
    return [f"Missing {provider}=={version} wheel-tag refusal for {case.platform}"]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phydrax-wheel", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    wheel = args.phydrax_wheel.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=False)
    metadata = wheel_metadata(wheel)
    errors = [metadata] if isinstance(metadata, str) else metadata_errors(metadata)
    cases: list[dict[str, Any]] = []
    for case in resolution_matrix():
        extra = f"[{case.extra}]" if case.extra is not None else ""
        if case.conflict is not None:
            extra = f"[{case.extra},{case.conflict}]"
        requirement = f"phydrax{extra} @ {wheel.as_uri()}"
        requirements = [requirement]
        if case.constraint is not None:
            requirements.append(case.constraint)
        requirements_file = args.work_dir.resolve() / f"{case.name}.in"
        requirements_file.write_text("\n".join(requirements) + "\n")
        command = [
            args.uv,
            "pip",
            "compile",
            "--python-version",
            case.python,
            "--python-platform",
            case.platform,
            "--no-header",
            "--no-annotate",
            str(requirements_file),
        ]
        result = run(
            command, cwd=args.work_dir, environment=dict(os.environ), timeout=args.timeout
        )
        resolved = resolved_distributions(Path(result["stdout_path"]).read_text())
        case_errors = []
        if result["timed_out"]:
            case_errors.append("Resolver timed out; this is not a platform refusal")
        elif (result["returncode"] == 0) != case.expected:
            case_errors.append(
                f"Expected {'resolution' if case.expected else 'refusal'}, got exit {result['returncode']}"
            )
        elif not case.expected:
            case_errors.extend(
                refusal_errors(case, Path(result["stderr_path"]).read_text())
            )
        if case.expected and result["returncode"] == 0:
            case_errors.extend(resolution_errors(resolved, case.extra))
        cases.append(
            {
                "name": case.name,
                "platform": case.platform,
                "python": case.python,
                "expected": case.expected,
                "requirements": requirements,
                "command": result,
                "resolved": resolved,
                "errors": case_errors,
                "successful": not case_errors,
            }
        )
    report = {
        "wheel": str(wheel),
        "wheel_sha256": sha256(wheel),
        "metadata_errors": errors,
        "cases": cases,
        "successful": not errors and all(case["successful"] for case in cases),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
