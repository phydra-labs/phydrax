# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Hardware-free metadata/resolver-policy tests (not accelerator evidence)."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from email.message import Message
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from tools import check_accelerator_extras as extras_tool
from tools.check_accelerator_extras import (
    Accelerator,
    metadata_errors,
    refusal_errors,
    resolution_errors,
    resolution_matrix,
    ResolutionCase,
)


def _metadata() -> Message:
    message = Message()
    message["Name"] = "phydrax"
    message["Version"] = "0.2.2"
    for accelerator in Accelerator:
        message["Provides-Extra"] = accelerator.value
    message["Provides-Extra"] = "cuda13"
    for requirement in (
        "jax>=0.12.0,<0.13",
        'jax==0.12.0; extra == "oneapi"',
        'jax==0.12.0; extra == "rocm10"',
        'jaxlib==0.11.2; extra == "oneapi"',
        'jax-oneapi-plugin[with-oneapi]==0.11.2; extra == "oneapi"',
        'jax[tpu]==0.12.0; extra == "tpu"',
        'jax[cuda13]==0.12.0; extra == "cuda13"',
        'jaxlib==0.12.0; extra == "cuda13"',
        'jaxlib==0.12.0; extra == "rocm10"',
        'jax-rocm10-plugin==0.12.0; extra == "rocm10"',
        'phydrax-rocm-runtime==10.1.0; extra == "rocm10"',
    ):
        message["Requires-Dist"] = requirement
    return message


@pytest.mark.parametrize(
    "failure",
    (
        None,
        "failed-baseline",
        "missing-baseline-provider",
        "tampered-log",
        "different-wheel",
        "command-target",
        "timeout",
        "unrelated-refusal",
    ),
)
def test_retained_resolution_assessment_requires_bound_successful_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str | None,
) -> None:
    wheel = tmp_path / "core.whl"
    wheel.write_bytes(b"metadata parser fixture")
    cases = (
        ResolutionCase(
            "positive", "x86_64-manylinux_2_31", "3.13", Accelerator.ROCM10, True
        ),
        ResolutionCase(
            "negative",
            "x86_64-manylinux_2_31",
            "3.13",
            Accelerator.ROCM10,
            False,
            constraint="jax==0.11.2",
        ),
    )
    monkeypatch.setattr(extras_tool, "resolution_matrix", Mock(return_value=cases))
    monkeypatch.setattr(extras_tool, "wheel_metadata", Mock(return_value=_metadata()))
    execute = Mock(side_effect=AssertionError("Reassessment must not rerun resolution"))
    monkeypatch.setattr(extras_tool, "run", execute)
    records: list[dict[str, Any]] = []
    for case in cases:
        installed = Accelerator.ROCM10.distributions
        if failure == "missing-baseline-provider":
            del installed["jax-rocm10-pjrt"]
        stdout = (
            "\n".join(f"{name}=={version}" for name, version in installed.items())
            if case.expected
            else ""
        )
        stderr = (
            ""
            if case.expected
            else (
                "No solution found: phydrax[rocm10], jax==0.11.2; requirements are unsatisfiable."
            )
        )
        if failure == "unrelated-refusal" and not case.expected:
            stderr = (
                "No solution found: unrelated-package; requirements are unsatisfiable."
            )
        command: dict[str, Any] = {
            "command": [
                "uv",
                "pip",
                "compile",
                "--python-version",
                "3.13",
                "--python-platform",
                case.platform,
            ],
            "returncode": 0 if case.expected else 1,
            "timed_out": failure == "timeout",
        }
        if failure == "failed-baseline" and case.expected:
            command["returncode"] = 1
        if failure == "command-target":
            command["command"][-1] = "aarch64-apple-darwin"
        for stream, value in (("stdout", stdout), ("stderr", stderr)):
            path = tmp_path / f"{case.name}.{stream}"
            path.write_text(value)
            command[f"{stream}_path"] = str(path)
            command[f"{stream}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        if failure == "tampered-log":
            Path(command["stdout_path"]).write_text("changed")
        records.append(
            {
                "name": case.name,
                "platform": case.platform,
                "python": case.python,
                "expected": case.expected,
                "requirements": [f"phydrax[rocm10] @ {wheel.as_uri()}"]
                + ([case.constraint] if case.constraint is not None else []),
                "command": command,
            }
        )
    report: dict[str, Any] = {
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "cases": records,
    }
    if failure == "different-wheel":
        wheel.write_bytes(b"replacement wheel")
    result = extras_tool.assess_resolution_report(report, wheel)
    assert result["successful"] is (failure is None)
    assert not execute.called


def test_upstream_metadata_keeps_intent_on_every_platform() -> None:
    assert metadata_errors(_metadata()) == []


@pytest.mark.parametrize("accelerator", tuple(Accelerator))
def test_platform_marked_empty_extra_is_refused(accelerator: Accelerator) -> None:
    message = _metadata()
    requirements = message.get_all("Requires-Dist", [])
    del message["Requires-Dist"]
    for requirement in requirements:
        if f'extra == "{accelerator}"' in requirement:
            requirement += ' and sys_platform == "linux" and platform_machine == "x86_64"'
        message["Requires-Dist"] = requirement
    assert any("empty extra" in error for error in metadata_errors(message))


def test_mixed_frontend_oneapi_stack_does_not_hide_jaxlib_conflict() -> None:
    message = _metadata()
    message["Requires-Dist"] = "jaxlib==0.12.0"
    assert any(
        "jaxlib" in error and "excludes" in error for error in metadata_errors(message)
    )


@pytest.mark.parametrize("accelerator", tuple(Accelerator))
def test_resolved_stack_requires_real_upstream_provider(accelerator: Accelerator) -> None:
    assert resolution_errors(accelerator.distributions, accelerator) == []
    assert resolution_errors({"jax": "0.12.0", "jaxlib": "0.12.0"}, accelerator)
    assert resolution_errors(accelerator.distributions, None)


def test_resolver_matrix_is_complete_and_unsupported_hosts_refuse() -> None:
    cases = resolution_matrix()
    assert len({case.name for case in cases}) == len(cases)
    for accelerator in Accelerator:
        selected = [case for case in cases if case.extra == accelerator]
        assert (
            len([case for case in selected if case.expected and case.conflict is None])
            == 1
        )
        assert {case.platform for case in selected if not case.expected} >= {
            "aarch64-manylinux_2_31",
            "aarch64-apple-darwin",
            "x86_64-pc-windows-msvc",
        }
    assert any(case.name == "base-cpu" and case.expected for case in cases)
    assert any(case.name == "tpu-old-glibc" and not case.expected for case in cases)
    assert not any(case.conflict == "metal" for case in cases)
    constraint = next(
        case for case in cases if case.name == "oneapi-conflicts-jaxlib-0.12.0"
    )
    assert constraint.constraint == "jaxlib==0.12.0" and not constraint.expected


@pytest.mark.parametrize(
    "name,provider,version,tag",
    (
        (
            "oneapi-arm",
            "jax-oneapi-plugin[with-oneapi]",
            "0.11.2",
            "manylinux_2_31_aarch64",
        ),
        ("oneapi-mac", "jax-oneapi-plugin[with-oneapi]", "0.11.2", "macosx_13_0_arm64"),
        ("oneapi-windows", "jax-oneapi-plugin[with-oneapi]", "0.11.2", "win_amd64"),
        ("tpu-arm", "libtpu", "0.0.50", "manylinux_2_31_aarch64"),
        ("tpu-mac", "libtpu", "0.0.50", "macosx_13_0_arm64"),
        ("tpu-windows", "libtpu", "0.0.50", "win_amd64"),
        ("tpu-old-glibc", "libtpu", "0.0.50", "manylinux_2_28_x86_64"),
        ("rocm10-arm", "jax-rocm10-plugin", "0.12.0", "manylinux_2_31_aarch64"),
        ("rocm10-mac", "jax-rocm10-plugin", "0.12.0", "macosx_13_0_arm64"),
        ("rocm10-windows", "jax-rocm10-plugin", "0.12.0", "win_amd64"),
    ),
)
def test_refusal_requires_selected_provider_wheel_derivation(
    name: str, provider: str, version: str, tag: str
) -> None:
    case = next(case for case in resolution_matrix() if case.name == name)
    text = f"No solution found when resolving dependencies: Because {provider}=={version} has no wheels with a matching platform tag (e.g., `{tag}`), your requirements are unsatisfiable."
    assert case.extra is not None
    check = partial(
        refusal_errors, case, metadata=_metadata(), baseline=case.extra.distributions
    )
    assert check(text) == []
    assert check(text.replace(provider, "unrelated-package"))
    assert check(text.replace(version, "99.0.0"))
    assert check(text.replace(tag, "unrelated_platform"))
    # An available-wheel hint must not disguise another host's refusal.
    assert check(
        text.replace(tag, "unrelated_platform") + f" Hint: available for `{tag}`."
    )


@pytest.mark.parametrize(
    "case",
    [
        case
        for case in resolution_matrix()
        if not case.expected
        and (case.constraint is not None or case.conflict is not None)
    ],
)
def test_declared_contradictions_require_metadata_and_successful_baseline(
    case: ResolutionCase,
) -> None:
    assert case.extra is not None
    text = (
        f"No solution found: phydrax[{case.extra}], "
        f"{case.constraint or case.conflict}, jaxlib; requirements are unsatisfiable."
    )
    assert not refusal_errors(
        case,
        text,
        metadata=_metadata(),
        baseline=case.extra.distributions,
    )
    assert refusal_errors(case, text, metadata=_metadata())
    assert refusal_errors(case, text, baseline=case.extra.distributions)
    assert refusal_errors(
        case,
        text,
        metadata=_metadata(),
        baseline={**case.extra.distributions, "jaxlib": "99.0.0"},
    )
    assert refusal_errors(
        case,
        "No solution found: unrelated-package; requirements are unsatisfiable.",
        metadata=_metadata(),
        baseline=case.extra.distributions,
    )


def test_compatible_constraint_cannot_be_claimed_as_known_contradiction() -> None:
    case = ResolutionCase(
        "compatible",
        "x86_64-manylinux_2_31",
        "3.13",
        Accelerator.ROCM10,
        False,
        constraint="jaxlib==0.12.0",
    )
    assert refusal_errors(
        case,
        "No solution found: jaxlib==0.12.0; requirements are unsatisfiable.",
        metadata=_metadata(),
        baseline=Accelerator.ROCM10.distributions,
    )


@pytest.mark.parametrize(
    "failure",
    (
        "Failed to fetch https://index.invalid/simple/libtpu",
        "Failed to download a wheel",
        "Failed to build unrelated-package",
        "error sending request",
        "connection refused",
        "TLS error",
        "HTTP 503 Service Unavailable",
        "status code 401 Unauthorized",
        "phydrax-metal was not found in the package registry",
        "command not found",
        "index is unavailable",
    ),
)
def test_operational_failure_never_counts_even_with_provider_refusal_text(
    failure: str,
) -> None:
    case = next(case for case in resolution_matrix() if case.name == "tpu-arm")
    derivation = "No solution found: Because libtpu==0.0.50 has no wheels with a matching platform tag (e.g., `manylinux_2_31_aarch64`), your requirements are unsatisfiable."
    assert refusal_errors(
        case,
        derivation + "\n" + failure,
        metadata=_metadata(),
        baseline=Accelerator.TPU.distributions,
    )


def test_unrelated_unsatisfiable_or_unpublished_metal_is_not_accelerator_proof() -> None:
    oneapi = next(case for case in resolution_matrix() if case.name == "oneapi-arm")
    unrelated = "No solution found: Because unrelated-package requires impossible-version, your requirements are unsatisfiable. You also requested jax-oneapi-plugin[with-oneapi]==0.11.2."
    assert refusal_errors(
        oneapi,
        unrelated,
        metadata=_metadata(),
        baseline=Accelerator.ONEAPI.distributions,
    )
    old_metal = ResolutionCase(
        "oneapi-conflicts-metal",
        "x86_64-manylinux_2_31",
        "3.13",
        Accelerator.ONEAPI,
        False,
        "metal",
    )
    missing_metal = "No solution found: Because phydrax-metal was not found in the package registry and phydrax[metal] depends on phydrax-metal, your requirements are unsatisfiable."
    assert refusal_errors(old_metal, missing_metal)


@pytest.mark.parametrize("accelerator", (Accelerator.ROCM10,))
def test_rocm_selector_separates_install_and_runtime(accelerator: Accelerator) -> None:
    assert accelerator.runtime_platform == "rocm"
    assert accelerator.qualification_platforms == ("rocm", "cpu")
    assert accelerator.runtime_distributions == {
        "phydrax-rocm-runtime": "10.1.0",
        "phydrax-rocm-core": "10.1.0",
        "phydrax-rocm-libraries": "10.1.0",
        "phydrax-rocm-device-gfx1250": "10.1.0",
        "phydrax-rocm-device-gfx942": "10.1.0",
        "phydrax-rocm-device-gfx950": "10.1.0",
    }
    assert accelerator.distributions == {
        "jax": "0.12.0",
        "jaxlib": "0.12.0",
        f"jax-{accelerator}-plugin": "0.12.0",
        f"jax-{accelerator}-pjrt": "0.12.0",
        **accelerator.runtime_distributions,
    }
    assert accelerator.requirement == f"jax-{accelerator}-plugin==0.12.0"
    cases = [
        case
        for case in resolution_matrix()
        if case.extra is accelerator and case.constraint is not None
    ]
    assert len(cases) == 5


@pytest.mark.parametrize("major", (10,))
def test_rocm_requires_explicit_core_pins_and_no_fake_plugin_companion(
    major: int,
) -> None:
    message = _metadata()
    requirements = message.get_all("Requires-Dist", [])
    del message["Requires-Dist"]
    for requirement in requirements:
        if requirement == f'jaxlib==0.12.0; extra == "rocm{major}"':
            continue
        message["Requires-Dist"] = requirement
    assert any("missing explicit jaxlib" in error for error in metadata_errors(message))
    message = _metadata()
    message["Requires-Dist"] = f'phydrax-rocm{major}==0.2.2; extra == "rocm{major}"'
    assert any("owned native" in error for error in metadata_errors(message))


@pytest.mark.parametrize("major", (10,))
def test_rocm_does_not_request_upstream_amd_index_extra(major: int) -> None:
    message = _metadata()
    requirements = message.get_all("Requires-Dist", [])
    del message["Requires-Dist"]
    for requirement in requirements:
        message["Requires-Dist"] = requirement.replace(
            f"jax-rocm{major}-plugin==", f"jax-rocm{major}-plugin[with-rocm]=="
        )
    assert metadata_errors(message)


@pytest.mark.parametrize("accelerator", (Accelerator.ROCM10,))
@pytest.mark.parametrize(
    "component",
    ("runtime", "core", "libraries", "device-gfx1250", "device-gfx942", "device-gfx950"),
)
def test_rocm_resolution_requires_complete_matching_bundle(
    accelerator: Accelerator, component: str
) -> None:
    installed = accelerator.distributions
    name = f"phydrax-rocm-{component}"
    del installed[name]
    assert resolution_errors(installed, accelerator)
    installed[name] = "99.0.0"
    assert resolution_errors(installed, accelerator)


def test_resolver_refuses_unsupported_runtime_family() -> None:
    installed = {**Accelerator.ROCM10.distributions, "jax-rocm99-plugin": "0.12.0"}
    assert resolution_errors(installed, Accelerator.ROCM10)


@pytest.mark.parametrize("major", (10,))
def test_rocm_metadata_requires_explicit_owned_runtime_pin(major: int) -> None:
    message = _metadata()
    requirements = message.get_all("Requires-Dist", [])
    del message["Requires-Dist"]
    for requirement in requirements:
        if (
            requirement.startswith("phydrax-rocm-runtime")
            and f'"rocm{major}"' in requirement
        ):
            continue
        message["Requires-Dist"] = requirement
    assert any(
        "missing explicit phydrax-rocm-runtime" in error
        for error in metadata_errors(message)
    )


@pytest.mark.parametrize(
    "module",
    (
        "tools.check_accelerator_extras",
        "tools.qualify_accelerator_wheels",
        "examples.accelerator_scientific_execution",
    ),
)
def test_tool_help_does_not_import_jax_or_provider(module: str) -> None:
    source = f"import runpy,sys; sys.argv=[{module!r},'--help']; runpy.run_module({module!r},run_name='__main__')"
    # A meta-path blocker proves CLI parsing does not initialize optional providers.
    source = (
        "import importlib.abc,sys\nclass Block(importlib.abc.MetaPathFinder):\n def find_spec(self,fullname,path=None,target=None):\n  if fullname.split('.')[0] in {'jax','jaxlib','jax_plugins','libtpu','tensorflow'}: raise RuntimeError('numerical import forbidden')\nsys.meta_path.insert(0,Block())\n"
        + source
    )
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (
        "--accelerator" if module.startswith("examples.") else "--phydrax-wheel"
    ) in result.stdout
