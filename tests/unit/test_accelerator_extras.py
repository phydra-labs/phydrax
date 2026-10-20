# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Hardware-free metadata/resolver-policy tests (not accelerator evidence)."""

from __future__ import annotations

import subprocess
import sys
from email.message import Message
from pathlib import Path

import pytest

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
    message["Provides-Extra"] = "oneapi"
    message["Provides-Extra"] = "tpu"
    for requirement in (
        "jax==0.12.0",
        'jaxlib==0.11.2; extra == "oneapi"',
        'jax-oneapi-plugin[with-oneapi]==0.11.2; extra == "oneapi"',
        'jax[tpu]==0.12.0; extra == "tpu"',
    ):
        message["Requires-Dist"] = requirement
    return message


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
        assert len([case for case in selected if case.expected]) == 1
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
    ),
)
def test_refusal_requires_selected_provider_wheel_derivation(
    name: str, provider: str, version: str, tag: str
) -> None:
    case = next(case for case in resolution_matrix() if case.name == name)
    text = f"No solution found when resolving dependencies: Because {provider}=={version} has no wheels with a matching platform tag (e.g., `{tag}`), your requirements are unsatisfiable."
    assert refusal_errors(case, text) == []
    assert refusal_errors(case, text.replace(provider, "unrelated-package"))
    assert refusal_errors(case, text.replace(version, "99.0.0"))
    assert refusal_errors(case, text.replace(tag, "unrelated_platform"))
    # An available-wheel hint must not disguise another host's refusal.
    assert refusal_errors(
        case, text.replace(tag, "unrelated_platform") + f" Hint: available for `{tag}`."
    )


@pytest.mark.parametrize("other", ("tpu", "cuda13"))
def test_refusal_requires_both_selected_extras_upstream_jaxlib_derivation(
    other: str,
) -> None:
    case = next(case for case in resolution_matrix() if case.conflict == other)
    operator = "==" if other == "tpu" else ">="
    text = f"No solution found when resolving dependencies: Because jax[{other}]==0.12.0 depends on jaxlib==0.12.0 and phydrax[{other}]==0.2.2 depends on jax[{other}]{operator}0.12.0, we conclude it depends on jaxlib==0.12.0. And because phydrax[oneapi]==0.2.2 depends on jaxlib==0.11.2, these extras are incompatible and your requirements are unsatisfiable."
    assert refusal_errors(case, text) == []
    assert refusal_errors(case, text.replace("jaxlib==0.11.2", "jaxlib==0.12.0"))
    assert refusal_errors(
        case,
        text.replace(
            f"jax[{other}]==0.12.0 depends on jaxlib==0.12.0",
            "unrelated-package depends on missing-package",
        ),
    )


def test_explicit_jaxlib_constraint_is_not_a_metal_compatibility_claim() -> None:
    case = next(case for case in resolution_matrix() if case.constraint is not None)
    text = "No solution found when resolving dependencies: Because phydrax[oneapi]==0.2.2 depends on jaxlib==0.11.2 and you require jaxlib==0.12.0, your requirements are unsatisfiable."
    assert refusal_errors(case, text) == []
    assert refusal_errors(
        case, text.replace("you require jaxlib==0.12.0", "you require missing-package")
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
    assert refusal_errors(case, derivation + "\n" + failure)


def test_unrelated_unsatisfiable_or_unpublished_metal_is_not_accelerator_proof() -> None:
    oneapi = next(case for case in resolution_matrix() if case.name == "oneapi-arm")
    unrelated = "No solution found: Because unrelated-package requires impossible-version, your requirements are unsatisfiable. You also requested jax-oneapi-plugin[with-oneapi]==0.11.2."
    assert refusal_errors(oneapi, unrelated)
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
