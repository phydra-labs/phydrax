#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Symbolic fill bounds for sparse LU with an owned column ordering.

Independent references: SuperLU's factor of ``A[:, q]`` under its natural
ordering (the executed factorization), SuperLU's own COLAMD factorization for
ordering quality, a dense symbolic Cholesky elimination of ``(AQ)ᵀ(AQ)`` for
the column counts, and dense NumPy solves.
"""

from math import prod
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import phydrax as phx


jax.config.update("jax_enable_x64", True)

la = phx.linalg
mx = phx.solver.maxwell
_POLICY = la.LinearSolvePolicy(
    la.SparseLU(), differentiation=la.DifferentiationPolicy("none")
)


def _sparse_map(matrix: sp.csr_matrix) -> Any:
    coordinates = matrix.tocoo()
    size = coordinates.shape[0]
    relation = phx.sparse.EdgeRelation(
        jnp.asarray(coordinates.col, dtype=jnp.int32),
        jnp.asarray(coordinates.row, dtype=jnp.int32),
        source_size=size,
        target_size=size,
    )
    return phx.sparse.SparseLinearMap(relation, jnp.asarray(coordinates.data))


def _laplacian(shape: tuple[int, ...]) -> Any:
    total = sp.csr_matrix((prod(shape), prod(shape)))
    for axis, count in enumerate(shape):
        factors = [sp.identity(extent, format="csr") for extent in shape]
        factors[axis] = sp.diags(
            [-np.ones(count - 1), 2.0 * np.ones(count), -np.ones(count - 1)],
            [-1, 0, 1],
            format="csr",
        )
        term = factors[0]
        for factor in factors[1:]:
            term = sp.kron(term, factor, format="csr")
        total = total + term
    return _sparse_map(total)


def _curl_curl(
    shape: tuple[int, ...], periodic: tuple[bool, ...], widths: tuple[int, ...]
) -> Any:
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=flag)
            for count, flag in zip(shape, periodic, strict=True)
        ),
        axis_names=tuple("xyz"[: len(shape)]),
    ).prepare(jnp.asarray([[0.0] * len(shape), [1.0] * len(shape)]))
    bridge = phx.discretization.StructuredCochainBridge(grid)
    layout = mx.MaxwellCochainLayout(bridge, "tez" if len(shape) == 2 else "full_3d")
    material = mx.DiagonalMaxwellConstitutivePlan(permittivity=2.0).prepare(
        bridge.cochain, layout
    )
    operator = mx.FrequencyMaxwellOperator(
        bridge,
        layout,
        material,
        6.0,
        stretching=mx.MaxwellCPMLPlan(widths, target_reflection=1e-6),
    )
    space = la.ArraySpace((operator.size,), dtype=jnp.complex128)
    zero = jnp.zeros((operator.size,), dtype=jnp.complex128)
    return phx.sparse.compile_sparse_jacobian(
        lambda field, frozen: frozen.mv(field),
        zero,
        source=space,
        target=space,
        sample_args=operator,
        complex_semantics="holomorphic",
    ).operator(zero)


def _host_matrix(operator: Any) -> sp.csc_matrix:
    storage = operator.sparse_storage()
    return sp.csr_matrix(
        (
            np.asarray(storage.values),
            np.asarray(storage.indices),
            np.asarray(storage.indptr),
        ),
        shape=storage.shape,
    ).tocsc()


@pytest.mark.parametrize(
    "build",
    [
        lambda: _laplacian((40, 40)),
        lambda: _laplacian((10, 10, 10)),
        lambda: _curl_curl((16, 48), (True, False), (0, 6)),
        lambda: _curl_curl((5, 8, 8), (True, False, False), (0, 2, 2)),
    ],
    ids=["laplacian-2d", "laplacian-3d", "curl-curl-tez", "curl-curl-3d"],
)
def test_symbolic_fill_bounds_the_executed_factor(build: Any) -> None:
    operator = build()
    problem = la.LinearSystem(operator)
    plan = la.plan(problem, _POLICY)
    analysis = plan.sparse_lu_analysis
    assert analysis is not None
    assert analysis.structurally_nonsingular
    matrix = _host_matrix(operator)
    columns = np.asarray(analysis.column_permutation)
    assert np.array_equal(np.sort(columns), np.arange(matrix.shape[0]))
    executed = spla.splu(matrix[:, columns], permc_spec="NATURAL")
    executed_nnz = executed.L.nnz + executed.U.nnz
    # George–Ng: every pivot sequence stays inside 2 nnz(chol((AQ)ᵀ(AQ))).
    assert executed_nnz <= analysis.factor_nnz_bound <= 2.5 * executed_nnz
    colamd = spla.splu(matrix)
    assert executed_nnz <= 1.25 * (colamd.L.nnz + colamd.U.nnz)
    itemsize = matrix.dtype.itemsize
    estimate = plan.candidates[-1]
    assert executed_nnz * itemsize < estimate.factorization_bytes
    assert estimate.factorization_bytes < matrix.shape[0] ** 2 * itemsize

    right_hand_side = jnp.cos(jnp.arange(matrix.shape[0], dtype=jnp.float64)).astype(
        matrix.dtype
    )
    result = la.solve(problem, right_hand_side, policy=_POLICY)
    assert bool(result.successful)
    residual = matrix @ np.asarray(result.value) - np.asarray(right_hand_side)
    assert np.linalg.norm(residual) <= 1e-9 * np.linalg.norm(right_hand_side)


def _dense_symbolic_cholesky_nnz(gram: np.ndarray) -> int:
    filled = gram.copy()
    for pivot in range(filled.shape[0]):
        below = np.flatnonzero(filled[pivot + 1 :, pivot]) + pivot + 1
        filled[np.ix_(below, below)] = True
    return int(np.count_nonzero(np.tril(filled)))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_column_counts_equal_dense_symbolic_elimination(seed: int) -> None:
    rng = np.random.default_rng(seed)
    size = 60
    pattern = (rng.random((size, size)) < 0.04) | np.eye(size, dtype=np.bool_)
    values = np.where(pattern, rng.normal(size=(size, size)), 0.0)
    plan = la.plan(la.LinearSystem(_sparse_map(sp.csr_matrix(values))), _POLICY)
    analysis = plan.sparse_lu_analysis
    assert analysis is not None
    ordered = pattern[:, np.asarray(analysis.column_permutation)].astype(np.int64)
    gram = (ordered.T @ ordered) != 0
    assert analysis.factor_nnz_bound == 2 * _dense_symbolic_cholesky_nnz(gram)


def test_factorization_budget_refuses_above_the_symbolic_fill() -> None:
    problem = la.LinearSystem(_laplacian((20, 20)))
    required = la.plan(problem, _POLICY).candidates[-1].factorization_bytes

    def policy(budget: int) -> Any:
        return la.LinearSolvePolicy(
            la.SparseLU(),
            differentiation=la.DifferentiationPolicy("none"),
            resources=la.SolveResourcePolicy(factorization_bytes=budget),
        )

    assert la.plan(problem, policy(required)).sparse_lu_analysis is not None
    with pytest.raises(ValueError, match="factorization bytes, exceeding"):
        la.plan(problem, policy(required - 1))


def test_structurally_singular_pattern_keeps_the_dense_bound() -> None:
    matrix = sp.csr_matrix(
        np.asarray(
            [
                [2.0, 1.0, 0.0, 0.0],
                [1.0, 2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    analysis = la.plan(la.LinearSystem(_sparse_map(matrix)), _POLICY).sparse_lu_analysis
    assert analysis is not None
    assert not analysis.structurally_nonsingular
    assert analysis.factor_nnz_bound == 16


def test_transposed_and_adjoint_solves_undo_the_column_ordering() -> None:
    rng = np.random.default_rng(3)
    size = 50
    pattern = (rng.random((size, size)) < 0.08) | np.eye(size, dtype=np.bool_)
    values = np.where(
        pattern, rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size)), 0
    ) + 4.0 * np.eye(size)
    operator = _sparse_map(sp.csr_matrix(values))
    prepared = la.prepare(la.LinearSystem(operator), _POLICY)
    analysis = prepared.plan.sparse_lu_analysis
    assert analysis is not None
    assert not np.array_equal(np.asarray(analysis.column_permutation), np.arange(size))
    right_hand_side = jnp.asarray(rng.normal(size=size) + 1j * rng.normal(size=size))
    rhs = np.asarray(right_hand_side)
    np.testing.assert_allclose(
        la.solve(prepared, right_hand_side).value,
        np.linalg.solve(values, rhs),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        la.solve_transpose(prepared, right_hand_side).value,
        np.linalg.solve(values.T, rhs),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        la.solve_adjoint(prepared, right_hand_side).value,
        np.linalg.solve(values.conj().T, rhs),
        rtol=1e-10,
    )
