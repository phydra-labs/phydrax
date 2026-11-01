#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path

import pytest

from tools import audit_compile_sites as audit


_IMPORTS = """import functools
from functools import partial

import equinox as eqx
import jax
from jax.sharding import PartitionSpec

from phydrax._owned_programs import run_owned_program
"""


def _findings(tmp_path: Path, source: str, /) -> list[tuple[str, str, str]]:
    package = tmp_path / "phydrax"
    package.mkdir(parents=True, exist_ok=True)
    (package / "module.py").write_text(_IMPORTS + source, encoding="utf-8")
    return [
        (finding.kind, finding.function, finding.detail)
        for finding in audit.audit(package, base=tmp_path)
    ]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "def run(x):\n    return jax.jit(lambda y: y + 1)(x)\n",
            [("per-call-compile", "run", "jax.jit of lambda")],
        ),
        (
            "def run(x, scale):\n"
            "    def scaled(y):\n        return y * scale\n"
            "    return eqx.filter_jit(scaled)(x)\n",
            [("per-call-compile", "run", "equinox.filter_jit of closure scaled")],
        ),
        (
            "def run(x):\n"
            "    @partial(jax.jit, static_argnames=('n',))\n"
            "    def body(y, n):\n        return y * n\n"
            "    return body(x, 2)\n",
            [
                (
                    "per-call-compile",
                    "run",
                    "jax.jit decorating nested def body",
                )
            ],
        ),
        (
            "def run(x, mesh):\n"
            "    def local(y):\n        return y\n"
            "    return jax.shard_map(\n"
            "        jax.vmap(local), mesh=mesh, in_specs=PartitionSpec(),"
            " out_specs=PartitionSpec()\n    )(x)\n",
            [("per-call-compile", "run", "jax.shard_map of vmap of closure local")],
        ),
        (
            "class Plan:\n"
            "    def apply(self, x):\n"
            "        return partial(jax.jit, donate_argnums=0)(self.kernel)(x)\n",
            [
                (
                    "per-call-compile",
                    "Plan.apply",
                    "jax.jit of attribute self.kernel",
                )
            ],
        ),
    ],
)
def test_per_call_constructions_are_reported(
    tmp_path: Path, source: str, expected: list[tuple[str, str, str]]
) -> None:
    assert _findings(tmp_path, source) == expected


@pytest.mark.parametrize(
    "source",
    [
        # Module-level and decorated module functions have one stable identity.
        "step = jax.jit(lambda y: y + 1)\n"
        "@eqx.filter_jit\ndef apply(y):\n    return y\n"
        "class Plan:\n    @eqx.filter_jit\n    def __call__(self, y):\n        return y\n",
        # Constructions traced inside a compiled program compile with it.
        "@jax.jit\ndef outer(x):\n    return jax.jit(lambda y: y * 2)(x)\n",
        "def _program(owner, x):\n"
        "    def local(y):\n        return y\n"
        "    return jax.shard_map(local, mesh=owner.mesh, in_specs=PartitionSpec(),"
        " out_specs=PartitionSpec())(x)\n"
        "def run(owner, x):\n"
        "    return run_owned_program(owner, 'phase', _program, x)\n",
        # A memoized factory and a prepared attribute own their program.
        "@functools.cache\ndef factory(n):\n    return jax.jit(lambda y: y * n)\n",
        "class Plan:\n"
        "    def __init__(self, kernel):\n"
        "        self.program = jax.jit(kernel)\n",
        # Arrays and module-level functions are not closures.
        "@eqx.filter_jit\ndef apply(f, y):\n    return f(y)\n"
        "def module_function(y):\n    return y\n"
        "def run(y):\n    return apply(module_function, y)\n",
    ],
)
def test_stable_programs_are_not_reported(tmp_path: Path, source: str) -> None:
    assert _findings(tmp_path, source) == []


def test_closures_passed_as_static_arguments_are_reported(tmp_path: Path) -> None:
    source = """@eqx.filter_jit
def apply(f, y):
    return f(y)


@partial(jax.jit, static_argnames=("f",))
def apply_static(y, f):
    return f(y)


@partial(jax.jit, static_argnums=(0,))
def apply_positional(f, y):
    return f(y)


def run(y, scale):
    def scaled(value):
        return value * scale

    first = apply(lambda value: value * scale, y)
    second = apply_static(y, f=scaled)
    third = apply_positional(partial(pow, 2), y)
    return first + second + third + apply_static(scaled, y)


def owned(owner, y):
    return run_owned_program(owner, "phase", lambda bound, value: value, y)
"""
    assert _findings(tmp_path, source) == [
        ("closure-static-arg", "run", "lambda as argument 0 of apply"),
        ("closure-static-arg", "run", "closure scaled as f= of apply_static"),
        ("closure-static-arg", "run", "partial(...) as argument 0 of apply_positional"),
        ("closure-owned-method", "owned", "lambda"),
    ]


def test_allowlist_matches_by_function_and_reports_stale_entries() -> None:
    findings = [
        audit.Finding("phydrax/a.py", 3, "per-call-compile", "factory", "jax.jit"),
        audit.Finding("phydrax/b.py", 9, "per-call-compile", "step", "jax.jit"),
    ]
    entries = (
        audit.AllowlistEntry("phydrax/a.py", "factory", "per-call-compile", "owned"),
        audit.AllowlistEntry("phydrax/c.py", "gone", "per-call-compile", "removed"),
    )

    report = audit.classify(findings, entries)

    assert report.findings == (findings[1],)
    assert report.allowed == (findings[0],)
    assert report.stale == (entries[1],)


def test_package_has_no_unallowlisted_compile_sites() -> None:
    report = audit.classify(
        audit.audit(audit.PACKAGE), audit.load_allowlist(audit.ALLOWLIST)
    )

    assert report.findings == ()
    assert report.stale == ()
