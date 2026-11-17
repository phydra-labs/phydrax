# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import gc
from pathlib import Path

import jax
import numpy as np
import pytest

from phydrax.linalg import FailurePolicy, MaterializationPolicy
from phydrax.meshing._tetra_coupled import (
    _CoupledBinding,
    _solve,
    TetraMetricProposalError,
)
from phydrax.meshing._tetra_coupled_program import (
    _CoupledRow,
    _ProgramCache,
    _RowRole,
    prepare_certificate,
    prepare_primary,
    solve_coupled,
)
from phydrax.optim import (
    ConvexProgramStatus,
    ConvexSolvePolicy,
    ConvexTermination,
    NativeHomogeneousConic,
)


_FIXTURES = Path(__file__).parents[1] / "optim"


def _policy(
    steps: int = 100, *, materialization: MaterializationPolicy | None = None
) -> ConvexSolvePolicy:
    return ConvexSolvePolicy(
        NativeHomogeneousConic(newton="factorized"),
        termination=ConvexTermination(absolute=1e-7, maximum_steps=steps),
        failure=FailurePolicy("status"),
        materialization=MaterializationPolicy()
        if materialization is None
        else materialization,
    )


def _binding(vertices: tuple[int, ...]) -> _CoupledBinding:
    return _CoupledBinding(
        ("captured-source", "captured-target", "captured-metric", "captured-model"),
        6690,
        (vertices[0], vertices[-1]),
        vertices,
        (),
    )


def _captured(
    name: str,
) -> tuple[list[_CoupledRow], list[float], _CoupledBinding, np.ndarray]:
    # Coordinate lists, not numerical nonzeros, define the captured structure.
    with np.load(_FIXTURES / name) as data:
        row_indices = np.asarray(data["rows"], dtype=np.int64)
        columns = np.asarray(data["columns"], dtype=np.int64)
        values = np.asarray(data["coefficients"], dtype=np.float64)
        rhs = np.asarray(data["rhs"], dtype=np.float64)
        vertices = tuple(int(v) for v in data["vertex_ids"])
    rows = []
    matrix = np.zeros((len(rhs), 3 * len(vertices)), dtype=np.float64)
    for i in range(len(rhs)):
        select = np.flatnonzero(row_indices == i)
        order = select[np.argsort(columns[select], kind="stable")]
        support = tuple(int(column) for column in columns[order])
        rows.append(_CoupledRow(support, values[order], _RowRole.SOURCE, identity=(i,)))
        np.add.at(matrix[i], columns[order], values[order])
    return rows, rhs.tolist(), _binding(vertices), matrix


@pytest.mark.parametrize(
    ("fixture", "primary_status", "certificate_expected"),
    [
        (
            "sparse_hsd_distance_constraints.npz",
            ConvexProgramStatus.PRIMAL_INFEASIBLE,
            False,
        ),
        ("sparse_hsd_full_defect.npz", ConvexProgramStatus.ITERATION_LIMIT, True),
    ],
    ids=["captured-81-primary-ray", "captured-156-bounded-farkas"],
)
def test_captured_models_produce_original_coordinate_proof(
    fixture: str, primary_status: ConvexProgramStatus, certificate_expected: bool
) -> None:
    rows, rhs, binding, matrix = _captured(fixture)
    solved = _solve(rows, rhs, [], _policy(), binding, cache=_ProgramCache())
    assert int(solved.result.status) == int(primary_status)
    assert solved.primal is None
    assert solved.binding == binding
    assert solved.ray_audit is not None and bool(solved.ray_audit.valid)
    ray = np.asarray(solved.ray_audit.inequality_ray)
    assert np.max(np.abs(matrix.T @ ray)) <= 1e-8
    assert np.min(ray) >= -1e-12
    assert float(np.asarray(rhs) @ ray) < -1e-8
    assert solved.primary_iterations == int(solved.result.iterations)
    if certificate_expected:
        assert not bool(solved.result.certificate.dual_ray_valid)
        assert solved.certificate_result is not None
        assert int(solved.certificate_result.status) == int(ConvexProgramStatus.OPTIMAL)
        assert bool(solved.certificate_result.successful)
        assert solved.certificate_demand is not None
        assert solved.certificate_iterations == int(solved.certificate_result.iterations)
        assert solved.certificate_iterations > 0
    else:
        assert bool(solved.result.certificate.dual_ray_valid)
        assert solved.certificate_result is None
        assert solved.certificate_demand is None
        assert solved.certificate_iterations == 0
    assert (
        solved.total_iterations
        == solved.primary_iterations + solved.certificate_iterations
    )


def _row(
    columns: tuple[int, ...],
    values: list[float],
    role: _RowRole = _RowRole.SOURCE,
    *,
    identity: tuple[int, ...],
) -> _CoupledRow:
    return _CoupledRow(
        columns, np.asarray(values, dtype=np.float64), role, identity=identity
    )


def test_feasible_primary_skips_certificate() -> None:
    solved = _solve(
        [_row((0,), [-1.0], identity=(0,))], [-1.0], [], _policy(), _binding((0,))
    )
    assert int(solved.result.status) == int(ConvexProgramStatus.OPTIMAL)
    assert bool(solved.result.successful)
    assert solved.primal is not None
    assert solved.primal[0] >= 1.0 - 1e-6
    assert solved.certificate_result is None
    assert solved.certificate_demand is None
    assert solved.ray_audit is None
    assert solved.certificate_iterations == 0
    assert solved.total_iterations == solved.primary_iterations


def test_certificate_equality_dual_has_signed_box() -> None:
    # x = 0 and x >= 1 require a negative equality multiplier. A nonnegative
    # box for that multiplier cannot produce this Farkas witness.
    policy = _policy()
    cache = _ProgramCache()
    primary = prepare_primary(
        [_row((0,), [-1.0], identity=(0,))],
        [-1.0],
        [_row((0,), [-1.0], _RowRole.CARRIER, identity=(1,))],
        3,
        "signed-equality-dual",
        policy,
        cache,
    )
    certificate = prepare_certificate(primary, policy, cache)
    result = solve_coupled(certificate)
    assert int(result.status) == int(ConvexProgramStatus.OPTIMAL)
    assert bool(result.successful)
    ray = np.asarray(result.primal)
    assert ray[0] < -1e-6 and ray[1] > 1e-6
    assert np.max(np.abs(ray)) <= 1.0 + 1e-6
    assert abs(ray[0] - ray[1] * -1.0) <= 1e-6
    assert -ray[1] < -1e-6


def test_rebind_keeps_structural_zero_and_changes_numerical_solution() -> None:
    policy = _policy()
    cache = _ProgramCache()
    first_binding = _binding((0,))._replace(source_cells=(10,))
    second_binding = first_binding._replace(source_cells=(11,))
    first = _solve(
        [_row((0, 1), [-1.0, 0.0], identity=(0,))],
        [-1.0],
        [],
        policy,
        first_binding,
        cache=cache,
    )
    second = _solve(
        [_row((0, 1), [-1.0, -1.0], identity=(0,))],
        [-1.0],
        [],
        policy,
        second_binding,
        cache=cache,
    )
    assert first.binding == first_binding and second.binding == second_binding
    assert (
        first.result.provenance.numeric_binding_id
        != second.result.provenance.numeric_binding_id
    )
    assert first.primal is not None and second.primal is not None
    np.testing.assert_allclose(first.primal[:2], [1.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(second.primal[:2], [0.5, 0.5], atol=1e-5)
    assert first.certificate_result is None and second.certificate_result is None


def _live_solver_executables() -> int:
    gc.collect()
    client = jax.devices()[0].client
    return sum(
        "solve_native_conic" in executable.hlo_modules()[0].name
        for executable in client.live_executables()
    )


def test_structures_of_one_capacity_class_share_one_compiled_solve() -> None:
    # A local search visits many sparsity patterns of similar size. They must
    # not each compile and retain a solver.
    policy = _policy()
    patterns: list[tuple[tuple[int, ...], list[float]]] = [
        ((0,), [-1.0]),
        ((1, 2), [-1.0, -1.0]),
        ((0, 2), [-2.0, -1.0]),
        ((0, 1, 2), [-1.0, -1.0, -1.0]),
    ]
    first = _solve(
        [_row(patterns[0][0], patterns[0][1], identity=(0,))],
        [-1.0],
        [],
        policy,
        _binding((0,)),
    )
    assert first.primal is not None
    compiled = _live_solver_executables()
    assert compiled >= 1
    for columns, values in patterns[1:]:
        solved = _solve(
            [_row(columns, values, identity=(0,))], [-1.0], [], policy, _binding((0,))
        )
        assert solved.primal is not None
        weights = np.asarray(values)
        # The minimum-norm point of one half-space -a.x <= -1.
        expected = np.zeros(3)
        expected[list(columns)] = -weights / float(weights @ weights)
        np.testing.assert_allclose(solved.primal, expected, atol=1e-5)
    assert _live_solver_executables() == compiled


def test_capacity_class_result_is_reported_in_original_coordinates() -> None:
    rows, rhs, binding, matrix = _captured("sparse_hsd_distance_constraints.npz")
    primary = prepare_primary(
        rows, rhs, [], matrix.shape[1], "original-coordinates", _policy(), _ProgramCache()
    )
    assert primary.layout is not None
    result = solve_coupled(primary)
    constraints, variables = matrix.shape
    assert result.primal.shape == (variables,)
    assert result.stationarity_residual.shape == (variables,)
    for value in (
        result.cone_dual,
        result.cone_slack,
        result.cone_primal_residual,
        result.certificate.inequality_dual_ray,
    ):
        assert value.shape == (constraints,)
    assert result.provenance.structure_id == primary.program.structure_id
    # The restored residual is the original program's residual, row for row.
    np.testing.assert_allclose(
        result.cone_primal_residual,
        matrix @ np.asarray(result.primal) + np.asarray(result.cone_slack) - rhs,
        atol=1e-12,
    )


def test_certificate_admission_refusal_preserves_primary_result() -> None:
    rows, rhs, binding, _ = _captured("sparse_hsd_full_defect.npz")
    # The primary's actual input demand is admitted; the larger transposed
    # Farkas LP must independently satisfy the same materialization limit.
    baseline = prepare_primary(
        rows,
        rhs,
        [],
        3 * len(binding.vertices),
        "admission-measurement",
        _policy(1),
        _ProgramCache(),
    )
    policy = _policy(
        1,
        materialization=MaterializationPolicy(max_entries=baseline.demand.input_entries),
    )
    with pytest.raises(TetraMetricProposalError) as caught:
        _solve(rows, rhs, [], policy, binding, cache=_ProgramCache())
    error = caught.value
    assert error.binding == binding
    assert error.result is not None
    assert int(error.result.status) == int(ConvexProgramStatus.ITERATION_LIMIT)
    assert not bool(error.result.certificate.dual_ray_valid)
    assert error.certificate_result is None
    assert isinstance(error.certificate_error, Exception)
    assert error.__cause__ is error.certificate_error
    assert error.certificate_demand is not None
    assert error.certificate_demand.input_entries > policy.materialization.max_entries
    assert not error.certificate_started
    assert error.certificate_iterations == 0
    assert error.total_iterations == int(error.result.iterations)


def test_unresolved_certificate_preserves_both_native_results() -> None:
    rows, rhs, binding, _ = _captured("sparse_hsd_full_defect.npz")
    with pytest.raises(TetraMetricProposalError) as caught:
        _solve(rows, rhs, [], _policy(1), binding, cache=_ProgramCache())
    error = caught.value
    assert error.binding == binding
    assert error.result is not None and error.certificate_result is not None
    assert int(error.result.status) == int(ConvexProgramStatus.ITERATION_LIMIT)
    assert not bool(error.result.certificate.dual_ray_valid)
    assert int(error.certificate_result.status) == int(
        ConvexProgramStatus.ITERATION_LIMIT
    )
    assert not bool(error.certificate_result.successful)
    assert error.certificate_error is None
    assert error.certificate_demand is not None
    assert error.ray_audit is not None and not bool(error.ray_audit.valid)
    assert int(error.result.iterations) == 1
    assert int(error.certificate_result.iterations) == 1
    assert error.certificate_started
    assert error.total_iterations == (
        int(error.result.iterations) + int(error.certificate_result.iterations)
    )
