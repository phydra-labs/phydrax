#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host-validated constant numeric versions and runtime-guarded traced versions."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


la = phx.linalg

_PRECONDITIONER_INVALID = "Preconditioner provenance versions are invalid."
_OPERATOR_NEGATIVE = "operator_numeric_version must be non-negative."


def _properties() -> Any:
    return la.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_definite": "construction",
            "positive_semidefinite": "construction",
        },
    )


def _system(matrix: Any, /) -> Any:
    return la.LinearSystem(
        la.DenseLinearOperator(
            matrix, properties=_properties(), operator_id="version-metadata"
        )
    )


def _provenance(**versions: Any) -> Any:
    return la.LinearSolveProvenance(
        backend="jax-dense",
        method="dense-cholesky",
        plan_id="plan",
        problem_id="problem",
        reason="test",
        prepared=True,
        **versions,
    )


def _callbacks(function: Any, /, *args: Any) -> int:
    return str(jax.make_jaxpr(function)(*args)).count("pure_callback")


def _version_leaves(provenance: Any, /) -> tuple[Any, ...]:
    return (
        provenance.preconditioner_numeric_version,
        provenance.preconditioner_built_numeric_version,
        provenance.operator_numeric_version,
    )


@pytest.mark.parametrize(
    ("versions", "message"),
    [
        ({"operator_numeric_version": -1}, _OPERATOR_NEGATIVE),
        (
            {"operator_numeric_version": np.zeros((1,), dtype=np.int64)},
            "operator_numeric_version must be scalar.",
        ),
        (
            {
                "preconditioner_numeric_version": -1,
                "preconditioner_built_numeric_version": 0,
            },
            _PRECONDITIONER_INVALID,
        ),
        (
            {
                "preconditioner_numeric_version": 1,
                "preconditioner_built_numeric_version": 2,
            },
            _PRECONDITIONER_INVALID,
        ),
        (
            {
                "preconditioner_numeric_version": -2,
                "preconditioner_built_numeric_version": -2,
            },
            _PRECONDITIONER_INVALID,
        ),
        (
            {
                "preconditioner_numeric_version": jnp.asarray([1, 1]),
                "preconditioner_built_numeric_version": 1,
            },
            "Preconditioner provenance versions must be scalar.",
        ),
    ],
)
def test_constant_invalid_versions_are_refused_on_the_host(
    versions: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _provenance(**versions)
    # A constant inside a trace is still host metadata: refused before staging.
    with pytest.raises(ValueError, match=message):
        jax.make_jaxpr(lambda: _provenance(**versions))()


def test_constant_versions_embed_canonical_int32_without_callbacks() -> None:
    cases = (
        {},
        # Matched and stale preconditioners, from Python, NumPy, and JAX scalars.
        {
            "preconditioner_numeric_version": np.int64(2),
            "preconditioner_built_numeric_version": 2,
            "operator_numeric_version": jnp.asarray(2),
        },
        {
            "preconditioner_numeric_version": jnp.asarray(3, dtype=jnp.int32),
            "preconditioner_built_numeric_version": np.int32(1),
            "operator_numeric_version": 3,
        },
    )
    for versions in cases:
        eager = _provenance(**versions)
        expected = tuple(int(value) for value in _version_leaves(eager))
        for leaf in _version_leaves(eager):
            assert isinstance(leaf, jax.Array)
            assert leaf.shape == () and leaf.dtype == jnp.int32
        assert _callbacks(lambda: _version_leaves(_provenance(**versions))) == 0
        staged = jax.jit(lambda: _version_leaves(_provenance(**versions)))()
        assert tuple(int(value) for value in staged) == expected
    assert tuple(int(value) for value in _version_leaves(_provenance())) == (-1, -1, 0)


def test_traced_versions_keep_runtime_refusals() -> None:
    operator = eqx.filter_jit(
        lambda version: _provenance(operator_numeric_version=version)
    )
    assert int(operator(jnp.asarray(4)).operator_numeric_version) == 4
    with pytest.raises(eqx.EquinoxRuntimeError, match=_OPERATOR_NEGATIVE):
        operator(jnp.asarray(-1))

    preconditioner = eqx.filter_jit(
        lambda version, built: _provenance(
            preconditioner_numeric_version=version,
            preconditioner_built_numeric_version=built,
        )
    )
    stale = preconditioner(jnp.asarray(3), jnp.asarray(1))
    assert (
        int(stale.preconditioner_numeric_version),
        int(stale.preconditioner_built_numeric_version),
    ) == (3, 1)
    # A traced partner still guards the constant one: the pair is one invariant.
    with pytest.raises(eqx.EquinoxRuntimeError, match=_PRECONDITIONER_INVALID):
        preconditioner(jnp.asarray(1), 2)
    with pytest.raises(eqx.EquinoxRuntimeError, match=_PRECONDITIONER_INVALID):
        preconditioner(jnp.asarray(-1), jnp.asarray(0))
    with pytest.raises(ValueError, match="must be scalar"):
        preconditioner(jnp.ones((2,), dtype=jnp.int32), jnp.asarray(1))


def test_prepared_numeric_version_guards_traced_values_only() -> None:
    prepared = la.prepare(_system(jnp.asarray([[4.0, 1.0], [1.0, 3.0]])))

    def rebind(version: Any) -> Any:
        return la.PreparedLinearSolve(
            prepared.problem,
            prepared.template,
            prepared.state,
            numeric_version=version,
        ).numeric_version

    with pytest.raises(ValueError, match="numeric_version must be non-negative."):
        rebind(-1)
    with pytest.raises(ValueError, match="numeric_version must be scalar."):
        rebind(np.zeros((2,), dtype=np.int32))
    assert _callbacks(lambda: rebind(5)) == 0
    assert _callbacks(rebind, jnp.asarray(5)) > 0
    assert int(eqx.filter_jit(rebind)(jnp.asarray(5))) == 5
    with pytest.raises(
        eqx.EquinoxRuntimeError, match="numeric_version must be non-negative."
    ):
        eqx.filter_jit(rebind)(jnp.asarray(-1))


def test_native_tiny_solve_with_default_metadata_exports_without_callbacks() -> None:
    def tiny(matrix: jax.Array, rhs: jax.Array) -> tuple[jax.Array, ...]:
        result = la.solve(_system(matrix), rhs)
        return (result.value, *_version_leaves(result.provenance))

    matrix = jnp.asarray([[4.0, 1.0], [1.0, 3.0]])
    rhs = jnp.asarray([1.0, 2.0])
    exported = jax.export.export(jax.jit(tiny))(matrix, rhs)
    value, *versions = exported.call(matrix, rhs)
    np.testing.assert_allclose(value, np.linalg.solve(matrix, rhs), rtol=1e-12)
    assert [int(version) for version in versions] == [-1, -1, 0]


def test_refreshed_preconditioned_and_recycled_solves_carry_versions() -> None:
    matrix = jnp.asarray([[4.0, 1.0], [1.0, 3.0]])
    rhs = jnp.asarray([1.0, 2.0])
    tolerance = la.TolerancePolicy(relative=1e-12, max_steps=10)
    for refresh, built in (("numeric", 1), ("frozen", 0)):
        policy = la.LinearSolvePolicy(
            la.PCG(),
            tolerance=tolerance,
            preconditioning=la.PreconditioningPolicy(
                la.JacobiPreconditionerBuilder(), refresh=refresh
            ),
        )
        refreshed = la.refresh(la.prepare(_system(matrix), policy), _system(2 * matrix))

        def provenance(rhs: jax.Array, prepared: Any = refreshed) -> tuple[Any, ...]:
            return _version_leaves(la.solve(prepared, rhs).provenance)

        assert [int(value) for value in provenance(rhs)] == [1, built, 1]
        assert [int(value) for value in jax.jit(provenance)(rhs)] == [1, built, 1]

    recycling = la.LinearSolvePolicy(
        la.GMRES(), tolerance=tolerance, recycling=la.RecyclingPolicy()
    )
    prepared = la.prepare(_system(matrix), recycling)
    first = la.solve_recycled(prepared, rhs)
    refreshed = la.refresh(prepared, _system(2 * matrix))
    second = la.solve_recycled(
        refreshed, rhs, recycling=la.refresh_recycling(first.recycling, refreshed)
    )
    assert int(first.result.provenance.operator_numeric_version) == 0
    assert int(second.result.provenance.operator_numeric_version) == 1
    assert int(second.recycling.operator_numeric_version) == 1
    np.testing.assert_allclose(
        second.result.value, np.linalg.solve(2 * matrix, rhs), rtol=1e-8
    )
