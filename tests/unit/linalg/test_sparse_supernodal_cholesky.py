#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Supernodal multifrontal sparse Cholesky against the pivot-by-pivot route.

Systems are P2 Dirichlet Laplacians assembled by
``problem.linear_system(storage="sparse")`` on a uniform triangle grid. The
pivot-by-pivot reference is the same complete Cholesky planned through the
elimination schedule that incomplete factorizations use.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import phydrax as phx
from phydrax import config as pxconfig
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.linalg import _sparse_factorizations as factorizations


pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=True))

la = phx.linalg
_SPD = la.OperatorProperties(
    self_adjoint=True,
    positive_definite=True,
    evidence={"self_adjoint": "construction", "positive_definite": "construction"},
)


def _triangles(count: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, 1.0, count + 1)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((x.ravel(), y.ravel()), axis=-1)
    index = np.arange((count + 1) ** 2).reshape(count + 1, count + 1)
    lower = index[:-1, :-1].ravel()
    cells = np.concatenate(
        (
            np.stack((lower, lower + count + 1, lower + count + 2), axis=-1),
            np.stack((lower, lower + count + 2, lower + 1), axis=-1),
        )
    )
    return points, cells.astype(np.int32)


@pytest.fixture(scope="module")
def fe_system() -> tuple[Any, np.ndarray]:
    """A ~1.5k-DOF P2 Dirichlet Laplacian and its right-hand side."""
    points, cells = _triangles(20)
    mesh = CellMesh.from_triangles(jnp.asarray(points), jnp.asarray(cells))
    space = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 2)
        ),
    ).prepare()
    problem = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "dirichlet-laplace",
            "u",
            (phx.equations.DiffusionAction("u", 1.0),),
            properties=_SPD,
        ),
        space,
        constraint=phx.discretization.dirichlet_constraint(space, "u"),
        dirichlet_values=lambda value: value[..., 0] + 2.0 * value[..., 1],
    )
    system, rhs = problem.linear_system(storage="sparse")
    return system.operator, np.asarray(rhs)


def _matrix(operator: Any) -> sp.csr_matrix:
    storage = operator.sparse_storage()
    return sp.csr_matrix(
        (
            np.asarray(storage.values),
            np.asarray(storage.indices),
            np.asarray(storage.indptr),
        ),
        shape=storage.shape,
    )


def _with_values(operator: Any, values: np.ndarray) -> Any:
    storage = operator.sparse_storage()
    matrix = sp.csr_matrix(
        (values, np.asarray(storage.indices), np.asarray(storage.indptr)),
        shape=storage.shape,
    ).tocoo()
    relation = phx.sparse.EdgeRelation(
        jnp.asarray(matrix.col, dtype=jnp.int32),
        jnp.asarray(matrix.row, dtype=jnp.int32),
        source_size=matrix.shape[1],
        target_size=matrix.shape[0],
    )
    return phx.sparse.SparseLinearMap(
        relation,
        jnp.asarray(matrix.data),
        properties=operator.properties,
    )


def _relative_residual(matrix: sp.csr_matrix, value: Any, rhs: np.ndarray) -> float:
    solution = np.asarray(value)
    return float(np.linalg.norm(matrix @ solution - rhs) / np.linalg.norm(rhs))


_ORDERINGS = ("nested-dissection", "approximate-minimum-degree", "reverse-cuthill-mckee")


@pytest.mark.parametrize("ordering", _ORDERINGS)
def test_supernodal_cholesky_matches_the_pivot_by_pivot_route(
    fe_system: tuple[Any, np.ndarray],
    ordering: la.SparseOrdering,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator, rhs = fe_system
    matrix = _matrix(operator)
    policy = la.SparseFactorizationPolicy("cholesky", ordering=ordering)
    plan = la.prepare_sparse_factorization(operator, policy)
    assert plan.supernodal is not None
    assert plan.elimination is None
    factor = la.refresh_sparse_factorization(plan, operator)
    solved = factor.solve(jnp.asarray(rhs))

    with monkeypatch.context() as patch:
        patch.setattr(factorizations, "_supernodal_route", lambda kind, policy: False)
        reference_plan = la.prepare_sparse_factorization(operator, policy)
    assert reference_plan.supernodal is None
    reference = la.refresh_sparse_factorization(reference_plan, operator)
    reference_solved = reference.solve(jnp.asarray(rhs))

    assert int(factor.status) == int(la.SparseFactorizationStatus.SUCCESS)
    assert bool(solved.success) and bool(reference_solved.success)
    # Same ordering, same structural fill; the postorder leaves fill unchanged.
    assert np.array_equal(plan.permutation, reference_plan.permutation)
    assert plan.factor_nnz == reference_plan.factor_nnz
    assert plan.factor_size >= plan.factor_nnz
    assert plan.factor_flops > 0
    assert int(factor.diagnostics.factor_nonzeros) <= plan.factor_nnz
    np.testing.assert_allclose(
        factor.diagnostics.minimum_pivot,
        reference.diagnostics.minimum_pivot,
        rtol=1e-12,
    )
    assert int(factor.diagnostics.rejected_pivots) == 0
    residual = _relative_residual(matrix, solved.value, rhs)
    reference_residual = _relative_residual(matrix, reference_solved.value, rhs)
    assert residual <= max(reference_residual, 4.0 * np.finfo(np.float64).eps)
    np.testing.assert_allclose(
        solved.value, reference_solved.value, rtol=1e-12, atol=1e-12
    )


def test_supernodal_multi_rhs_and_adjoint_solves(
    fe_system: tuple[Any, np.ndarray],
) -> None:
    operator, _ = fe_system
    matrix = _matrix(operator)
    factor = la.factorize_sparse(
        operator, la.SparseFactorizationPolicy("cholesky", ordering="nested-dissection")
    )
    block = np.random.default_rng(3).standard_normal((matrix.shape[0], 5))
    solved = jax.jit(lambda value: factor.solve(value))(jnp.asarray(block))
    adjoint = factor.solve(jnp.asarray(block), adjoint=True)
    assert bool(solved.success)
    residual = np.linalg.norm(matrix @ np.asarray(solved.value) - block)
    assert residual <= 1e-12 * np.linalg.norm(block)
    np.testing.assert_array_equal(solved.value, adjoint.value)
    columns = [np.asarray(factor.solve(jnp.asarray(column)).value) for column in block.T]
    # Blocked and single-column BLAS sweeps round differently near cancellation.
    np.testing.assert_allclose(
        np.stack(columns, axis=1), solved.value, rtol=1e-12, atol=1e-14
    )


def test_supernodal_cholesky_refuses_indefinite_values(
    fe_system: tuple[Any, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator, rhs = fe_system
    matrix = _matrix(operator)
    policy = la.SparseFactorizationPolicy("cholesky", ordering="nested-dissection")
    plan = la.prepare_sparse_factorization(operator, policy)
    # Negating the diagonal entry eliminated last makes the values indefinite
    # (a negative Rayleigh quotient on that unit vector) and rejects exactly the
    # final pivot: a rejected pivot is continued as one, so an earlier rejection
    # would feed growing Schur complements to every later pivot.
    rows = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
    diagonal = np.flatnonzero(rows == matrix.indices)
    indefinite_values = np.asarray(matrix.data).copy()
    indefinite_values[diagonal[int(plan.permutation[-1])]] *= -1.0

    indefinite = la.refresh_sparse_factorization_values(
        plan, jnp.asarray(indefinite_values)
    )
    assert int(indefinite.status) == int(la.SparseFactorizationStatus.NONPOSITIVE_PIVOT)
    assert int(indefinite.diagnostics.rejected_pivots) == 1
    assert int(indefinite.diagnostics.replaced_pivots) == 0
    assert bool(indefinite.diagnostics.finite)
    assert not bool(indefinite.solve(jnp.asarray(rhs)).success)
    with monkeypatch.context() as patch:
        patch.setattr(factorizations, "_supernodal_route", lambda kind, policy: False)
        reference_plan = la.prepare_sparse_factorization(operator, policy)
    reference = la.refresh_sparse_factorization_values(
        reference_plan, jnp.asarray(indefinite_values)
    )
    assert int(reference.status) == int(indefinite.status)
    assert int(reference.diagnostics.rejected_pivots) == 1

    replacing = la.prepare_sparse_factorization(
        operator,
        la.SparseFactorizationPolicy(
            "cholesky", ordering="nested-dissection", allow_pivot_replacement=True
        ),
    )
    replaced = la.refresh_sparse_factorization_values(
        replacing, jnp.asarray(indefinite_values)
    )
    assert int(replaced.status) == int(la.SparseFactorizationStatus.SUCCESS)
    assert int(replaced.diagnostics.replaced_pivots) == int(
        replaced.diagnostics.rejected_pivots
    )
    assert int(replaced.diagnostics.replaced_pivots) == 1

    nonfinite = np.asarray(matrix.data).copy()
    nonfinite[0] = np.nan
    failed = la.refresh_sparse_factorization_values(plan, jnp.asarray(nonfinite))
    assert int(failed.status) == int(la.SparseFactorizationStatus.NONFINITE)
    assert not bool(failed.diagnostics.finite)


def test_supernodal_plan_is_reused_for_refreshed_values(
    fe_system: tuple[Any, np.ndarray],
) -> None:
    operator, rhs = fe_system
    matrix = _matrix(operator)
    policy = la.SparseFactorizationPolicy("cholesky", ordering="nested-dissection")
    plan = la.prepare_sparse_factorization(operator, policy)
    refreshed_matrix = sp.csr_matrix(3.0 * matrix + sp.eye(matrix.shape[0]))
    refreshed_matrix.sort_indices()
    refreshed_operator = _with_values(operator, refreshed_matrix.data)

    traces: list[None] = []

    def traced_refresh(plan: Any, values: jax.Array) -> Any:
        traces.append(None)
        return la.refresh_sparse_factorization_values(plan, values)

    refresh = jax.jit(traced_refresh)
    first = refresh(plan, jnp.asarray(matrix.data))
    second = refresh(plan, jnp.asarray(refreshed_matrix.data))
    # New values on the frozen pattern reuse one compiled refresh.
    assert len(traces) == 1
    assert first.plan.plan_id == second.plan.plan_id == plan.plan_id
    assert second.plan.supernodal is plan.supernodal
    routed = la.refresh_sparse_factorization(plan, refreshed_operator)
    np.testing.assert_array_equal(routed.factor_values, second.factor_values)
    for factor, values in ((first, matrix), (second, refreshed_matrix)):
        solved = factor.solve(jnp.asarray(rhs))
        assert bool(solved.success)
        assert _relative_residual(values, solved.value, rhs) <= 1e-13

    batched = jax.vmap(
        lambda values: la.refresh_sparse_factorization_values(plan, values)
    )(jnp.stack((matrix.data, refreshed_matrix.data)))
    np.testing.assert_array_equal(batched.factor_values[0], first.factor_values)
    np.testing.assert_array_equal(batched.factor_values[1], second.factor_values)


def test_supernodal_factorization_is_deterministic(
    fe_system: tuple[Any, np.ndarray],
) -> None:
    operator, rhs = fe_system
    policy = la.SparseFactorizationPolicy("cholesky", ordering="nested-dissection")
    first_plan = la.prepare_sparse_factorization(operator, policy)
    second_plan = la.prepare_sparse_factorization(operator, policy)
    assert first_plan.plan_id == second_plan.plan_id
    assert first_plan.supernodal is not None and second_plan.supernodal is not None
    assert first_plan.supernodal == second_plan.supernodal
    assert first_plan.supernodal.analysis_id == second_plan.supernodal.analysis_id
    first = la.refresh_sparse_factorization(first_plan, operator)
    second = la.refresh_sparse_factorization(second_plan, operator)
    np.testing.assert_array_equal(first.factor_values, second.factor_values)
    np.testing.assert_array_equal(
        first.solve(jnp.asarray(rhs)).value, second.solve(jnp.asarray(rhs)).value
    )


def test_supernodal_factor_and_solve_derivatives_match_central_differences(
    fe_system: tuple[Any, np.ndarray],
) -> None:
    operator, rhs = fe_system
    matrix = _matrix(operator)
    plan = la.prepare_sparse_factorization(
        operator, la.SparseFactorizationPolicy("cholesky", ordering="nested-dissection")
    )
    rng = np.random.default_rng(11)
    # A symmetric direction on the stored pattern keeps the values self-adjoint:
    # average random entries with their transposed partners.
    size = matrix.shape[0]
    rows = np.repeat(np.arange(size), np.diff(matrix.indptr))
    keys = rows * size + matrix.indices
    partner = np.searchsorted(keys, matrix.indices * size + rows)
    raw = rng.standard_normal(matrix.nnz)
    direction = jnp.asarray(0.5 * (raw + raw[partner]))
    values = jnp.asarray(matrix.data)
    weights = jnp.asarray(rng.standard_normal(matrix.shape[0]))
    step = 1e-6

    def panels(scale: jax.Array) -> jax.Array:
        return la.refresh_sparse_factorization_values(
            plan, values + scale * direction
        ).factor_values

    _, tangent = jax.jvp(panels, (jnp.asarray(0.0),), (jnp.asarray(1.0),))
    central = (panels(jnp.asarray(step)) - panels(jnp.asarray(-step))) / (2 * step)
    np.testing.assert_allclose(tangent, central, rtol=1e-6, atol=1e-7)

    def observable(stored: jax.Array, right: jax.Array) -> jax.Array:
        factor = la.refresh_sparse_factorization_values(plan, stored)
        return jnp.vdot(weights, factor.solve(right).value)

    value_gradient, rhs_gradient = jax.jit(jax.grad(observable, argnums=(0, 1)))(
        values, jnp.asarray(rhs)
    )
    shifted = [
        float(observable(values + sign * step * direction, jnp.asarray(rhs)))
        for sign in (1.0, -1.0)
    ]
    np.testing.assert_allclose(
        jnp.vdot(value_gradient, direction),
        (shifted[0] - shifted[1]) / (2 * step),
        rtol=1e-6,
    )
    multiplier = spla.spsolve(sp.csc_matrix(matrix), np.asarray(weights))
    np.testing.assert_allclose(rhs_gradient, multiplier, rtol=1e-9, atol=1e-12)
    forward = jax.jvp(
        lambda scale: observable(values + scale * direction, jnp.asarray(rhs)),
        (jnp.asarray(0.0),),
        (jnp.asarray(1.0),),
    )[1]
    np.testing.assert_allclose(forward, jnp.vdot(value_gradient, direction), rtol=1e-9)


def _admission_cases(matrix: sp.csr_matrix, plan: Any) -> dict[str, np.ndarray]:
    rows = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
    diagonal = np.flatnonzero(rows == matrix.indices)
    spd = np.asarray(matrix.data).copy()
    last = spd.copy()
    last[diagonal[int(plan.permutation[-1])]] *= -1.0
    first = spd.copy()
    first[diagonal[int(plan.permutation[0])]] *= -1.0
    nonfinite = spd.copy()
    nonfinite[0] = np.nan
    return {"spd": spd, "last": last, "first": first, "nonfinite": nonfinite}


@pytest.mark.parametrize("replacement", (False, True))
def test_status_only_admission_reproduces_the_refresh_evidence(
    fe_system: tuple[Any, np.ndarray], replacement: bool
) -> None:
    operator, _ = fe_system
    matrix = _matrix(operator)
    plan = la.prepare_sparse_factorization(
        operator,
        la.SparseFactorizationPolicy(
            "cholesky",
            ordering="approximate-minimum-degree",
            allow_pivot_replacement=replacement,
        ),
    )
    cases = _admission_cases(matrix, plan)

    def refresh(values: jax.Array) -> Any:
        return la.refresh_sparse_factorization_values(plan, values)

    def admit(values: jax.Array) -> Any:
        return factorizations.admit_sparse_factorization_values(plan, values)

    batch = np.stack(tuple(cases.values()))
    for name, values, transform in (
        *((name, values, jax.jit) for name, values in cases.items()),
        ("batch", batch, jax.vmap),
    ):
        full = transform(refresh)(jnp.asarray(values))
        admitted = transform(admit)(jnp.asarray(values))
        for left, right in (
            (admitted.status, full.status),
            (admitted.minimum_pivot, full.diagnostics.minimum_pivot),
            (admitted.replaced_pivots, full.diagnostics.replaced_pivots),
            (admitted.rejected_pivots, full.diagnostics.rejected_pivots),
            (admitted.finite, full.diagnostics.finite),
        ):
            np.testing.assert_array_equal(
                np.asarray(left), np.asarray(right), err_msg=name
            )
    # The evidence carries no derivative and does not refuse to be traced by one.
    gradient = jax.grad(
        lambda values: (
            jnp.sum(values)
            + factorizations.admit_sparse_factorization_values(plan, values).minimum_pivot
        )
    )(jnp.asarray(cases["spd"]))
    np.testing.assert_array_equal(gradient, np.ones_like(cases["spd"]))


@pytest.mark.parametrize("case", ("spd", "last", "first", "nonfinite"))
def test_numeric_workspace_bound_covers_the_measured_refresh(
    fe_system: tuple[Any, np.ndarray], case: str
) -> None:
    import tracemalloc

    from phydrax.linalg import _sparse_supernodal as supernodal

    operator, _ = fe_system
    matrix = _matrix(operator)
    plan = la.prepare_sparse_factorization(
        operator,
        la.SparseFactorizationPolicy("cholesky", ordering="nested-dissection"),
    )
    analysis = plan.supernodal
    assert analysis is not None
    values = _admission_cases(matrix, plan)[case]
    policy = plan.policy
    tracemalloc.start()
    supernodal.factor_values(
        analysis,
        values,
        policy.diagonal_shift,
        policy.pivot_tolerance,
        policy.allow_pivot_replacement,
        policy.replacement_value,
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    itemsize = values.dtype.itemsize
    workspace = supernodal.numeric_workspace_bytes(analysis, itemsize)
    assert peak <= analysis.factor_size * itemsize + workspace
    assert factorizations.numeric_admission_bytes(plan) == plan.factor_bytes + workspace
