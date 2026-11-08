#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Every-order derivatives of prepared sparse factor solves and panels.

``A(p)`` and ``b(p)`` depend smoothly on parameters; the references are the same
functions differentiated through the dense `jnp.linalg.solve` and
`jnp.linalg.cholesky`. Supernodal Cholesky solves differentiate implicitly
(tangents and cotangents are solves with the prepared panels); the
elimination-schedule LU route differentiates through its JAX factorization.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

import phydrax as phx
from phydrax import config as pxconfig
from phydrax.linalg import _sparse_factorizations as factorizations, _sparse_supernodal


pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=True))

la = phx.linalg
_HERMITIAN = la.OperatorProperties(
    self_adjoint=True,
    positive_definite=True,
    evidence={"self_adjoint": "construction", "positive_definite": "construction"},
)
_ORDER = 3


def _grid(nx: int, ny: int) -> sp.csr_matrix:
    """Nine-point grid pattern: enough fill and several supernodes."""

    def line(count: int) -> sp.dia_matrix:
        return sp.diags([1.0, 1.0, 1.0], [-1, 0, 1], shape=(count, count))

    pattern = sp.kron(line(ny), line(nx)).tocsr()
    pattern.sort_indices()
    return pattern


class _Family:
    """Values ``v₀ + Σᵢ sin(pᵢ) vᵢ₊₁ + (Σᵢ pᵢ)² v_m`` and ``b(p)`` on one pattern."""

    def __init__(
        self,
        kind: str,
        *,
        parameters: int = 1,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        pattern = _grid(7, 6)
        size = pattern.shape[0]
        rows = np.repeat(np.arange(size), np.diff(pattern.indptr))
        columns = pattern.indices
        keys = rows * size + columns
        partner = np.searchsorted(keys, columns * size + rows)
        diagonal = rows == columns
        complex_ = kind == "cholesky-complex"
        dtype = np.complex128 if complex_ else np.float64

        def random(scale: float, *, symmetric: bool) -> np.ndarray:
            raw = rng.uniform(-scale, scale, size=rows.size)
            if complex_:
                raw = raw + 1j * rng.uniform(-scale, scale, size=rows.size)
                values = 0.5 * (raw + np.conj(raw[partner]))
                return np.where(diagonal, values.real, values).astype(dtype)
            return 0.5 * (raw + raw[partner]) if symmetric else raw

        symmetric = kind != "lu-general"
        base = np.where(diagonal, 0.0, -1.0).astype(dtype)
        if kind == "lu-symmetric-indefinite":
            # Alternating dominant diagonals: nonsingular, indefinite, and safe
            # for static-pivot LU.
            sign = np.where(np.arange(size) % 2 == 0, 1.0, -1.0)
            base = np.where(diagonal, 20.0 * sign[rows], base)
        else:
            base = np.where(diagonal, 20.0, base).astype(dtype)
        base = base + random(0.5, symmetric=symmetric)
        self.kind = kind
        self.size = size
        self.rows = rows
        self.columns = columns
        self.values = jnp.asarray(base)
        self.directions = jnp.asarray(
            np.stack([random(0.6, symmetric=symmetric) for _ in range(parameters + 1)])
        )
        rhs = rng.standard_normal((3, size))
        if complex_:
            rhs = rhs + 1j * rng.standard_normal((3, size))
        self.rhs = jnp.asarray(rhs)
        cholesky = kind.startswith("cholesky")
        properties = _HERMITIAN if cholesky else None
        matrix = sp.csr_matrix(
            (np.asarray(base), columns, pattern.indptr), shape=pattern.shape
        )
        self.operator = phx.sparse.SparseLinearMap(
            phx.sparse.EdgeRelation(
                jnp.asarray(columns, dtype=jnp.int32),
                jnp.asarray(rows, dtype=jnp.int32),
                source_size=size,
                target_size=size,
            ),
            jnp.asarray(matrix.data),
            properties=properties,
        )
        np.testing.assert_array_equal(
            np.asarray(self.operator.sparse_storage().values), np.asarray(base)
        )
        self.policy = la.SparseFactorizationPolicy(
            "cholesky" if cholesky else "lu", ordering="nested-dissection"
        )
        self.plan = la.prepare_sparse_factorization(self.operator, self.policy)
        assert (self.plan.supernodal is not None) == cholesky

    def value_function(self, p: jax.Array) -> jax.Array:
        p = jnp.atleast_1d(p)
        weights = jnp.concatenate((jnp.sin(p), jnp.sum(p)[None] ** 2))
        return self.values + jnp.tensordot(
            weights.astype(self.values.dtype), self.directions, axes=1
        )

    def rhs_function(self, p: jax.Array) -> jax.Array:
        p = jnp.atleast_1d(p)
        total = jnp.sum(p)
        return self.rhs[0] + jnp.cos(total) * self.rhs[1] + total**3 * self.rhs[2]

    def dense(self, values: jax.Array) -> jax.Array:
        return (
            jnp.zeros((self.size, self.size), values.dtype)
            .at[self.rows, self.columns]
            .set(values)
        )

    def sparse_solution(self, p: jax.Array) -> jax.Array:
        factor = la.refresh_sparse_factorization_values(self.plan, self.value_function(p))
        return factor.solve(self.rhs_function(p)).value

    def dense_solution(self, p: jax.Array) -> jax.Array:
        return jnp.linalg.solve(self.dense(self.value_function(p)), self.rhs_function(p))


_KINDS = ("cholesky-real", "cholesky-complex", "lu-general", "lu-symmetric-indefinite")


type _Function = Callable[[jax.Array], jax.Array]


def _nested(function: _Function, order: int) -> list[_Function]:
    derivatives = [function]
    for _ in range(order):
        derivatives.append(jax.jacfwd(derivatives[-1]))
    return derivatives


def _close(actual: Any, expected: Any, /, *, rtol: float = 1e-11) -> None:
    actual, expected = np.asarray(actual), np.asarray(expected)
    scale = max(float(np.max(np.abs(expected))), 1.0)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=rtol * scale)


@pytest.mark.parametrize("kind", _KINDS)
def test_solution_derivatives_through_third_order_match_dense_solve(kind: str) -> None:
    family = _Family(kind)
    p = jnp.asarray(0.3)
    sparse = _nested(family.sparse_solution, _ORDER)
    dense = _nested(family.dense_solution, _ORDER)
    for order in range(_ORDER + 1):
        _close(sparse[order](p), dense[order](p))
    jitted = jax.jit(sparse[_ORDER])(p)
    _close(jitted, dense[_ORDER](p))


@pytest.mark.parametrize("kind", _KINDS)
def test_mixed_mode_hessians_and_jvp_vjp_duality(kind: str) -> None:
    family = _Family(kind, parameters=2, seed=1)
    p = jnp.asarray([0.2, -0.4])
    weights = jnp.asarray(np.random.default_rng(5).standard_normal(family.size))

    def real(solution: _Function) -> _Function:
        def flattened(value: jax.Array) -> jax.Array:
            result = solution(value)
            return jnp.concatenate((jnp.real(result), jnp.imag(result)))

        return flattened

    sparse, dense = real(family.sparse_solution), real(family.dense_solution)

    def objective(solution: _Function) -> _Function:
        def value(parameters: jax.Array) -> jax.Array:
            return jnp.dot(jnp.tile(weights, 2), solution(parameters))

        return value

    reference = jax.hessian(objective(dense))(p)
    for transform in (
        jax.hessian,
        lambda f: jax.jacrev(jax.jacrev(f)),
        lambda f: jax.jacrev(jax.jacfwd(f)),
        lambda f: jax.jacfwd(jax.jacfwd(f)),
    ):
        _close(transform(objective(sparse))(p), reference)

    rng = np.random.default_rng(9)
    tangent = jnp.asarray(rng.standard_normal(2))
    second = jnp.asarray(rng.standard_normal(2))
    cotangent = jnp.asarray(rng.standard_normal(2 * family.size))

    def first_derivative(value: jax.Array) -> jax.Array:
        return jax.jvp(sparse, (value,), (tangent,))[1]

    for function in (sparse, first_derivative):
        forward = jax.jvp(function, (p,), (second,))[1]
        (backward,) = jax.vjp(function, p)[1](cotangent)
        np.testing.assert_allclose(
            jnp.dot(cotangent, forward), jnp.dot(backward, second), rtol=1e-12
        )


@pytest.mark.parametrize("kind", _KINDS)
def test_one_numeric_factorization_per_nested_derivative(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    family = _Family(kind)
    counts = {"factor": 0, "tangent": 0}
    if family.plan.supernodal is not None:
        factor_values = _sparse_supernodal.factor_values
        tangent_values = _sparse_supernodal.tangent_values

        def counted_factor(*args: Any, **kwargs: Any) -> Any:
            counts["factor"] += 1
            return factor_values(*args, **kwargs)

        def counted_tangent(*args: Any, **kwargs: Any) -> Any:
            counts["tangent"] += 1
            return tangent_values(*args, **kwargs)

        monkeypatch.setattr(_sparse_supernodal, "factor_values", counted_factor)
        monkeypatch.setattr(_sparse_supernodal, "tangent_values", counted_tangent)
    else:
        # The JAX factorization runs as one loop; its body is traced once per
        # factorization however many derivative levels transform it.
        eliminate = factorizations._eliminate_pivot

        def counted_eliminate(*args: Any, **kwargs: Any) -> Any:
            counts["factor"] += 1
            return eliminate(*args, **kwargs)

        monkeypatch.setattr(factorizations, "_eliminate_pivot", counted_eliminate)
    p = jnp.asarray(0.3)
    third = _nested(family.sparse_solution, _ORDER)[_ORDER]
    hessian = jax.hessian(lambda value: jnp.sum(jnp.abs(family.sparse_solution(value))))
    for evaluate in (third, jax.jit(third), hessian, jax.jit(hessian)):
        counts["factor"] = 0
        jax.block_until_ready(evaluate(p))
        assert counts["factor"] == 1
    if family.plan.supernodal is not None:
        # Panel tangents reuse the stored panels: none refactors.
        counts["factor"] = counts["tangent"] = 0
        jax.block_until_ready(
            jax.jacfwd(
                jax.jacfwd(
                    lambda value: (
                        la.refresh_sparse_factorization_values(
                            family.plan, family.value_function(value)
                        ).factor_values
                    )
                )
            )(p)
        )
        assert counts["factor"] == 1
        assert counts["tangent"] > 0


def _panel_coordinates(
    analysis: _sparse_supernodal.SupernodalAnalysis,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Elimination-order (row, column) of every lower panel position."""
    rows = np.zeros(analysis.factor_size, dtype=np.int64)
    columns = np.zeros(analysis.factor_size, dtype=np.int64)
    for node in range(analysis.supernode_count):
        first, last = analysis.supernode_columns[node : node + 2]
        structure = analysis.rows[
            analysis.row_offsets[node] : analysis.row_offsets[node + 1]
        ]
        start = analysis.value_offsets[node]
        for offset, column in enumerate(range(first, last)):
            positions = start + offset * structure.size + np.arange(structure.size)
            rows[positions] = structure
            columns[positions] = column
    return rows, columns, rows >= columns


@pytest.mark.parametrize("kind", ("cholesky-real", "cholesky-complex"))
def test_panel_and_minimum_pivot_derivatives_match_dense_cholesky(kind: str) -> None:
    family = _Family(kind, parameters=2, seed=2)
    analysis = family.plan.supernodal
    assert analysis is not None and analysis.supernode_count > 1
    rows, columns, lower = _panel_coordinates(analysis)
    elimination = analysis.elimination
    p = jnp.asarray([0.1, 0.25])

    def panels(value: jax.Array) -> jax.Array:
        factor = la.refresh_sparse_factorization_values(
            family.plan, family.value_function(value)
        )
        return jnp.where(lower, factor.factor_values, 0.0)

    def dense_panels(value: jax.Array) -> jax.Array:
        matrix = family.dense(family.value_function(value))
        ordered = matrix[elimination][:, elimination]
        # The panels hold the lower triangle; the dense factor reads it too.
        factor = jnp.linalg.cholesky(jnp.tril(ordered) + jnp.tril(ordered, -1).conj().T)
        return jnp.where(lower, factor[rows, columns], 0.0)

    sparse, dense = _nested(panels, _ORDER), _nested(dense_panels, _ORDER)
    for order in range(_ORDER + 1):
        _close(sparse[order](p), dense[order](p), rtol=1e-10)

    refreshed = la.refresh_sparse_factorization_values(
        family.plan, family.value_function(p)
    )
    position = int(
        np.argmin(
            np.where(
                lower & (rows == columns),
                np.abs(np.asarray(refreshed.factor_values)) ** 2,
                np.inf,
            )
        )
    )

    def minimum(value: jax.Array) -> jax.Array:
        return la.refresh_sparse_factorization_values(
            family.plan, family.value_function(value)
        ).diagnostics.minimum_pivot

    def dense_minimum(value: jax.Array) -> jax.Array:
        return jnp.abs(dense_panels(value)[position]) ** 2

    sparse, dense = _nested(minimum, _ORDER), _nested(dense_minimum, _ORDER)
    for order in range(_ORDER + 1):
        _close(sparse[order](p), dense[order](p), rtol=1e-10)


def test_replaced_pivot_is_constant_at_every_order() -> None:
    family = _Family("cholesky-real", seed=3)
    plan = la.prepare_sparse_factorization(
        family.operator,
        la.SparseFactorizationPolicy(
            "cholesky",
            ordering="nested-dissection",
            allow_pivot_replacement=True,
            replacement_value=2.0,
        ),
    )
    analysis = plan.supernodal
    assert analysis is not None
    last = int(analysis.elimination[-1])
    diagonal = np.flatnonzero((family.rows == last) & (family.columns == last))
    # The pivot eliminated last turns negative and is replaced; every earlier
    # pivot stays accepted, so the reference is the leading Cholesky block,
    # one forward substitution and the constant replaced root.
    values = family.values.at[diagonal].set(-40.0)
    rows, columns, lower = _panel_coordinates(analysis)
    elimination = analysis.elimination

    def value_function(p: jax.Array) -> jax.Array:
        return values + jnp.sin(p) * family.directions[0] + p**2 * family.directions[1]

    def panels(p: jax.Array) -> jax.Array:
        factor = la.refresh_sparse_factorization_values(plan, value_function(p))
        return jnp.where(lower, factor.factor_values, 0.0)

    def dense_panels(p: jax.Array) -> jax.Array:
        matrix = family.dense(value_function(p))[elimination][:, elimination]
        matrix = jnp.tril(matrix) + jnp.tril(matrix, -1).T
        leading = jnp.linalg.cholesky(matrix[:-1, :-1])
        row = jax.scipy.linalg.solve_triangular(leading, matrix[:-1, -1], lower=True)
        factor = jnp.zeros_like(matrix)
        factor = factor.at[:-1, :-1].set(leading).at[-1, :-1].set(row)
        factor = factor.at[-1, -1].set(jnp.sqrt(2.0))
        return jnp.where(lower, factor[rows, columns], 0.0)

    p = jnp.asarray(0.2)
    refreshed = la.refresh_sparse_factorization_values(plan, value_function(p))
    assert int(refreshed.diagnostics.replaced_pivots) == 1
    sparse, dense = _nested(panels, _ORDER), _nested(dense_panels, _ORDER)
    for order in range(_ORDER + 1):
        _close(sparse[order](p), dense[order](p), rtol=1e-10)


def test_reverse_mode_through_panels_is_a_typed_refusal() -> None:
    family = _Family("cholesky-real")
    p = jnp.asarray(0.3)

    def refreshed(value: jax.Array) -> Any:
        return la.refresh_sparse_factorization_values(
            family.plan, family.value_function(value)
        )

    with pytest.raises(la.SparseFactorDerivativeError, match="Reverse-mode"):
        jax.grad(lambda value: jnp.sum(refreshed(value).factor_values))(p)
    with pytest.raises(la.SparseFactorDerivativeError, match="Reverse-mode"):
        jax.grad(lambda value: refreshed(value).diagnostics.minimum_pivot)(p)
    # The solve itself differentiates in reverse mode at every order.
    gradient = jax.grad(lambda value: jnp.sum(family.sparse_solution(value)))(p)
    _close(gradient, jax.grad(lambda value: jnp.sum(family.dense_solution(value)))(p))
