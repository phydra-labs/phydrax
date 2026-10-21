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
    """Closed install selector for supported accelerator stacks."""

    ONEAPI = "oneapi"
    TPU = "tpu"
    ROCM10 = "rocm10"

    @property
    def runtime_platform(self) -> str:
        return "rocm" if self.rocm_major is not None else self.value

    @property
    def qualification_platforms(self) -> tuple[str, ...]:
        """Expose CPU for ROCm host callbacks, never for scientific placement."""
        return ("rocm", "cpu") if self.rocm_major is not None else (self.value,)

    @property
    def rocm_major(self) -> int | None:
        match self:
            case Accelerator.ROCM10:
                return 10
            case Accelerator.ONEAPI | Accelerator.TPU:
                return None
        raise ValueError(f"Unsupported accelerator {self!r}")

    @property
    def sdk_version(self) -> str | None:
        return "10.1.0" if self is Accelerator.ROCM10 else None

    @property
    def runtime_distributions(self) -> dict[str, str]:
        version = self.sdk_version
        if version is None:
            return {}
        return {
            f"phydrax-rocm-{component}": version
            for component in (
                "runtime",
                "core",
                "libraries",
                "device-gfx1250",
                "device-gfx942",
                "device-gfx950",
            )
        }

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
            case Accelerator.ROCM10:
                return {
                    "jax": "0.12.0",
                    "jaxlib": "0.12.0",
                    f"jax-{self.value}-plugin": "0.12.0",
                    f"jax-{self.value}-pjrt": "0.12.0",
                    **self.runtime_distributions,
                }
        raise ValueError(f"Unsupported accelerator {self!r}")

    @property
    def requirement(self) -> str:
        match self:
            case Accelerator.ONEAPI:
                return "jax-oneapi-plugin[with-oneapi]==0.11.2"
            case Accelerator.TPU:
                return "jax[tpu]==0.12.0"
            case Accelerator.ROCM10:
                return f"jax-{self.value}-plugin==0.12.0"
        raise ValueError(f"Unsupported accelerator {self!r}")

    def matches_rocm_runtime(self, platform: str, client: str, runtime: str) -> bool:
        """Recognize ROCm; installation family is bound by distribution provenance."""
        return (
            self.rocm_major is not None
            and platform.lower() in {"gpu", "rocm"}
            and client.lower() in {"gpu", "rocm"}
            and re.search(
                r"\brocm\b",
                runtime,
                re.IGNORECASE,
            )
            is not None
        )


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
    for accelerator in (Accelerator.ROCM10,):
        for package in (
            "jax",
            "jaxlib",
            f"jax-{accelerator}-plugin",
            f"jax-{accelerator}-pjrt",
        ):
            cases.append(
                ResolutionCase(
                    f"{accelerator}-conflicts-{package}-0.11.2",
                    "x86_64-manylinux_2_31",
                    "3.13",
                    accelerator,
                    False,
                    constraint=f"{package}==0.11.2",
                )
            )
    cases.append(
        ResolutionCase(
            "rocm10-conflicts-runtime-version",
            "x86_64-manylinux_2_31",
            "3.13",
            Accelerator.ROCM10,
            False,
            constraint="phydrax-rocm-runtime==99.0.0",
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
                and (accelerator.rocm_major is None or not req.extras)
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
                if accelerator.rocm_major is not None and name in {
                    "jax",
                    "jaxlib",
                    "phydrax-rocm-runtime",
                }:
                    if not any(str(req.specifier) == f"=={version}" for req in direct):
                        errors.append(
                            f"{accelerator}: missing explicit {name}=={version}"
                        )
    if any(
        canonicalize_name(req.name)
        in {
            "phydrax-oneapi",
            "phydrax-tpu",
            "phydrax-rocm",
            "phydrax-rocm10",
        }
        for req in requirements
    ):
        errors.append("Upstream runtimes must not claim owned native companions")
    return errors


def resolution_errors(
    resolved: dict[str, str], accelerator: Accelerator | None
) -> list[str]:
    expected = {} if accelerator is None else accelerator.distributions
    errors = [
        f"{name}: expected {version}, resolved {resolved.get(name)!r}"
        for name, version in expected.items()
        if resolved.get(name) != version
    ]
    optional = {
        name
        for selected in Accelerator
        for name in selected.distributions
        if name not in {"jax", "jaxlib"}
    }
    optional.update(
        name for name in resolved if name.startswith(("jax-rocm", "phydrax-rocm"))
    )
    unexpected = (optional & resolved.keys()) - expected.keys()
    if unexpected:
        errors.append(f"Unselected optional runtimes resolved: {sorted(unexpected)}")
    return errors


def refusal_errors(
    case: ResolutionCase,
    stderr: str,
    *,
    metadata: Message | None = None,
    baseline: dict[str, str] | None = None,
) -> list[str]:
    """Bind refusal to wheel intent and a successful exact-stack baseline.

    Dependency contradictions come from requirements, not the resolver's choice
    of explanation wording. Operational failures never establish incompatibility.
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
    if (
        case.extra is None
        or metadata is None
        or metadata_errors(metadata)
        or baseline is None
        or resolution_errors(baseline, case.extra)
    ):
        return ["Missing validated wheel metadata or successful selected-stack baseline"]
    if case.constraint is not None:
        constraint = Requirement(case.constraint)
        name = canonicalize_name(constraint.name)
        version = case.extra.distributions.get(name)
        if (
            version is not None
            and not constraint.specifier.contains(version, prereleases=True)
            and name in text
            and str(constraint.specifier) in text
        ):
            return []
        return ["Explicit constraint does not establish the selected-stack contradiction"]
    if case.conflict is not None:
        if case.extra is not Accelerator.ONEAPI or case.conflict not in {"tpu", "cuda13"}:
            return ["No supported cross-extra contradiction is declared"]
        environment = {
            **default_environment(),
            "extra": case.conflict,
            "sys_platform": "linux",
            "platform_system": "Linux",
            "platform_machine": "x86_64",
            "python_version": "3.13",
            "python_full_version": "3.13.0",
        }
        requirements = [
            Requirement(str(value)) for value in metadata.get_all("Requires-Dist", [])
        ]
        # Both upstream JAX0.12 extras pin JAXlib0.12, unlike oneAPI0.11.2.
        linked = any(
            canonicalize_name(requirement.name) == "jax"
            and case.conflict in requirement.extras
            and requirement.specifier.contains("0.12.0")
            and (requirement.marker is None or requirement.marker.evaluate(environment))
            for requirement in requirements
        )
        if (
            linked
            and baseline["jaxlib"] != "0.12.0"
            and all(name in text for name in ("oneapi", case.conflict, "jaxlib"))
        ):
            return []
        return [
            "Wheel requirements do not establish the selected extras' JAXlib contradiction"
        ]
    match case.extra:
        case Accelerator.ONEAPI:
            provider, version = "jax-oneapi-plugin", "0.11.2"
        case Accelerator.TPU:
            provider, version = "libtpu", "0.0.50"
        case Accelerator.ROCM10:
            provider, version = f"jax-{case.extra}-plugin", "0.12.0"
    platform_tags = {
        "aarch64-manylinux_2_31": r"manylinux_2_31_aarch64",
        "aarch64-apple-darwin": r"macosx_\d+_\d+_arm64",
        "x86_64-pc-windows-msvc": r"win_amd64",
        "x86_64-manylinux_2_28": r"manylinux_2_28_x86_64",
    }
    target_tag = platform_tags.get(case.platform)
    providers = {provider: version}
    if case.extra is Accelerator.ROCM10:
        providers.update(case.extra.runtime_distributions)
    if target_tag and any(
        re.search(
            rf"\bbecause {re.escape(name)}(?:\[[^\]]+\])?=={re.escape(release)} has no wheels with a matching platform tag \(e\.g\., `?{target_tag}`?\)",
            text,
        )
        for name, release in providers.items()
    ):
        return []
    return [f"Missing {provider}=={version} wheel-tag refusal for {case.platform}"]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def case_requirements(case: ResolutionCase, wheel: Path) -> list[str]:
    extra = f"[{case.extra}]" if case.extra is not None else ""
    if case.conflict is not None:
        extra = f"[{case.extra},{case.conflict}]"
    requirements = [f"phydrax{extra} @ {wheel.as_uri()}"]
    if case.constraint is not None:
        requirements.append(case.constraint)
    return requirements


def assess_resolution_report(report: dict[str, Any], wheel: Path) -> dict[str, Any]:
    """Reassess retained full command evidence without rerunning resolution."""
    wheel = wheel.resolve()
    metadata = wheel_metadata(wheel)
    errors = [metadata] if isinstance(metadata, str) else metadata_errors(metadata)
    if report.get("wheel_sha256") != sha256(wheel):
        errors.append("Retained resolver evidence belongs to a different wheel")
    matrix = {case.name: case for case in resolution_matrix()}
    records = report["cases"]
    names = [record["name"] for record in records]
    if len(names) != len(set(names)) or set(names) != set(matrix):
        errors.append("Retained resolver evidence does not cover the exact case matrix")
    baselines: dict[Accelerator, dict[str, str]] = {}
    for record in records:
        case_errors: list[str] = []
        record["errors"] = case_errors
        record["successful"] = False
        case = matrix.get(record["name"])
        if case is None:
            case_errors.append("Unknown resolver case")
            continue
        if (
            record.get("platform") != case.platform
            or record.get("python") != case.python
            or record.get("expected") != case.expected
            or record.get("requirements") != case_requirements(case, wheel)
        ):
            case_errors.append("Retained resolver target differs from the declared case")
        result = record["command"]
        command = result["command"]
        for flag, value in (
            ("--python-version", case.python),
            ("--python-platform", case.platform),
        ):
            if command.count(flag) != 1 or command[
                command.index(flag) + 1 : command.index(flag) + 2
            ] != [value]:
                case_errors.append(f"Retained resolver command differs at {flag}")
        try:
            for stream in ("stdout", "stderr"):
                path = Path(result[f"{stream}_path"])
                if sha256(path) != result[f"{stream}_sha256"]:
                    raise ValueError(f"Retained {stream} digest differs")
            record["resolved"] = resolved_distributions(
                Path(result["stdout_path"]).read_text()
            )
        except (OSError, ValueError) as error:
            case_errors.append(f"Full retained resolver output unavailable: {error}")
            continue
        if result["timed_out"]:
            case_errors.append("Resolver timed out; this is not a platform refusal")
        elif (result["returncode"] == 0) != case.expected:
            case_errors.append(
                f"Expected {'resolution' if case.expected else 'refusal'}, got exit {result['returncode']}"
            )
        elif not case.expected and result["returncode"] != 1:
            case_errors.append("Resolver did not return its incompatibility exit status")
        elif case.expected:
            case_errors.extend(resolution_errors(record["resolved"], case.extra))
            if (
                not case_errors
                and not errors
                and case.extra is not None
                and case.conflict is None
            ):
                baselines[case.extra] = record["resolved"]
    for record in records:
        case = matrix.get(record["name"])
        if case is not None and not case.expected and not record["errors"]:
            record["errors"].extend(
                refusal_errors(
                    case,
                    Path(record["command"]["stderr_path"]).read_text(),
                    metadata=metadata if isinstance(metadata, Message) else None,
                    baseline=baselines.get(case.extra)
                    if case.extra is not None
                    else None,
                )
            )
        record["successful"] = not record["errors"]
    report["metadata_errors"] = errors
    report["successful"] = not errors and all(record["successful"] for record in records)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phydrax-wheel", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--reassess-report",
        type=Path,
        help="Reassess full hash-bound retained command logs; do not run the resolver",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--find-links", type=Path, action="append", default=[])
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    wheel = args.phydrax_wheel.resolve()
    if args.reassess_report is not None:
        report = assess_resolution_report(
            json.loads(args.reassess_report.read_text()), wheel
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return 0 if report["successful"] else 1
    if args.work_dir is None:
        parser.error("--work-dir is required unless --reassess-report is supplied")
    args.work_dir.mkdir(parents=True, exist_ok=False)
    metadata = wheel_metadata(wheel)
    errors = [metadata] if isinstance(metadata, str) else metadata_errors(metadata)
    cases: list[dict[str, Any]] = []
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(
            ("UV_INDEX", "UV_DEFAULT_INDEX", "UV_EXTRA_INDEX", "UV_FIND_LINKS", "PIP_")
        )
    }
    for case in resolution_matrix():
        requirements = case_requirements(case, wheel)
        requirements_file = args.work_dir.resolve() / f"{case.name}.in"
        requirements_file.write_text("\n".join(requirements) + "\n")
        command = [
            args.uv,
            "--no-config",
            "pip",
            "compile",
            "--python-version",
            case.python,
            "--python-platform",
            case.platform,
            "--no-header",
            "--no-annotate",
            *(
                value
                for path in args.find_links
                for value in ("--find-links", str(path.resolve()))
            ),
            str(requirements_file),
        ]
        result = run(
            command, cwd=args.work_dir, environment=environment, timeout=args.timeout
        )
        resolved = resolved_distributions(Path(result["stdout_path"]).read_text())
        cases.append(
            {
                "name": case.name,
                "platform": case.platform,
                "python": case.python,
                "expected": case.expected,
                "requirements": requirements,
                "command": result,
                "resolved": resolved,
                "errors": [],
                "runtime_compatibility_claimed": False,
                "successful": False,
            }
        )
    report = {
        "wheel": str(wheel),
        "wheel_sha256": sha256(wheel),
        "find_links": [str(path.resolve()) for path in args.find_links],
        "metadata_errors": errors,
        "cases": cases,
        "successful": not errors and all(case["successful"] for case in cases),
    }
    assess_resolution_report(report, wheel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
