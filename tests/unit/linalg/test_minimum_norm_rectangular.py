# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Rectangular minimum-norm solves against independent dense SVD oracles.

Oracles are host NumPy SVD pseudoinverses of explicit matrices; they share no
code with the native LSMR, rank-certificate, or implicit-derivative routes.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax.linalg as la
from phydrax.sparse import EdgeRelation, SparseCoordinateOperator


_RNG = np.random.default_rng(20261001)
# Rank-3 constraint matrix with six equations on four unknowns: three redundant
# rows and a one-dimensional source nullspace.
_LEFT = _RNG.standard_normal((6, 3))
_RIGHT = _RNG.standard_normal((4, 3))
_MATRIX = _LEFT @ _RIGHT.T
_SOLUTION = _RNG.standard_normal(4)
_CONSISTENT = _MATRIX @ _SOLUTION
_LEFT_NULL = np.linalg.svd(_MATRIX)[0][:, 3:]
_INCONSISTENT = _CONSISTENT + _LEFT_NULL @ np.asarray([0.7, -0.4, 0.2])
_SOURCE_WEIGHTS = np.asarray([2.0, 0.5, 1.5, 3.0])
_TARGET_WEIGHTS = np.asarray([1.0, 4.0, 0.25, 2.0, 1.0, 0.5])
_TOLERANCE = la.TolerancePolicy(relative=1e-12, absolute=1e-14, max_steps=80)


def _weighted_pinv(
    matrix: np.ndarray, source: np.ndarray, target: np.ndarray
) -> np.ndarray:
    root_target = np.sqrt(target)[:, None]
    root_source = np.sqrt(source)[None, :]
    weighted = root_target * matrix / root_source
    return np.linalg.pinv(weighted, rcond=1e-10) * root_target.T / root_source.T


def _operator(weighted: bool) -> la.DenseLinearOperator:
    if not weighted:
        return la.DenseLinearOperator(jnp.asarray(_MATRIX))
    return la.DenseLinearOperator(
        jnp.asarray(_MATRIX),
        source=la.ArraySpace(
            (4,),
            dtype=jnp.float64,
            pairing=la.DiagonalPairing(jnp.asarray(_SOURCE_WEIGHTS)),
        ),
        target=la.ArraySpace(
            (6,),
            dtype=jnp.float64,
            pairing=la.DiagonalPairing(jnp.asarray(_TARGET_WEIGHTS)),
        ),
    )


_ROUTES = {
    "lsmr": (la.LSMR(), False),
    "generalized-lsmr-weighted": (la.GeneralizedLSMR(), True),
    "dense-svd-weighted": (la.DenseSVD(), True),
    "craig": (la.Craig(), False),
    "craig-weighted": (la.Craig(), True),
}
_SPD_PRECONDITIONER = la.PreconditionerProperties(
    linear=True,
    stationary=True,
    self_adjoint=True,
    positive_definite=True,
    evidence={
        name: "asserted"
        for name in ("linear", "stationary", "self_adjoint", "positive_definite")
    },
)


def _sparse_operator(weighted: bool) -> SparseCoordinateOperator:
    rows, columns = np.nonzero(_MATRIX)
    source_pairing = (
        la.DiagonalPairing(jnp.asarray(_SOURCE_WEIGHTS)) if weighted else None
    )
    target_pairing = (
        la.DiagonalPairing(jnp.asarray(_TARGET_WEIGHTS)) if weighted else None
    )
    return SparseCoordinateOperator(
        EdgeRelation(
            columns.astype(np.int32), rows.astype(np.int32), source_size=4, target_size=6
        ),
        jnp.asarray(_MATRIX[rows, columns]),
        source=la.ArraySpace(
            (4,), dtype=jnp.float64, space_id="z", pairing=source_pairing
        ),
        target=la.ArraySpace(
            (6,), dtype=jnp.float64, space_id="b", pairing=target_pairing
        ),
    )


_CRAIG_PRECONDITIONED = {
    "matrix-free-jacobi": (la.Craig(), la.JacobiPreconditionerBuilder()),
    "assembled-jacobi": (
        la.Craig(assembly=la.SparseAssemblyPolicy()),
        la.JacobiPreconditionerBuilder(),
    ),
    "assembled-smoothed-aggregation": (
        la.Craig(assembly=la.SparseAssemblyPolicy()),
        la.SmoothedAggregationHierarchyBuilder(
            la.SmoothedAggregationPolicy(minimum_coarse_size=2),
            la.JacobiPreconditionerBuilder(relaxation=0.6),
            la.DenseInversePreconditionerBuilder(),
            properties=_SPD_PRECONDITIONER,
        ),
    ),
}


@pytest.mark.parametrize("weighted", (False, True), ids=("euclidean", "weighted"))
@pytest.mark.parametrize(
    "route", tuple(_CRAIG_PRECONDITIONED), ids=tuple(_CRAIG_PRECONDITIONED)
)
def test_preconditioned_craig_matches_oracle_and_classifies_incompatibility(
    route: str, weighted: bool
) -> None:
    method, builder = _CRAIG_PRECONDITIONED[route]
    policy = la.LinearSolvePolicy(
        method, tolerance=_TOLERANCE, preconditioning=la.PreconditioningPolicy(builder)
    )
    operator = _sparse_operator(weighted)
    problem = la.MinimumNormProblem(
        operator, rank_certificate=la.certify_rectangular_rank(operator, _CERTIFIED)
    )
    source = _SOURCE_WEIGHTS if weighted else np.ones(4)
    target = _TARGET_WEIGHTS if weighted else np.ones(6)

    consistent = la.solve(problem, jnp.asarray(_CONSISTENT), policy=policy)
    assert consistent.provenance.method == "craig"
    assert int(consistent.status) == int(la.LinearSolveStatus.SUCCESS)
    np.testing.assert_allclose(
        np.asarray(consistent.value),
        _weighted_pinv(_MATRIX, source, target) @ _CONSISTENT,
        rtol=1e-8,
        atol=1e-10,
    )
    assert consistent.minimum_norm is not None
    assert float(consistent.minimum_norm.constraint_residual) < 1e-9 * np.linalg.norm(
        _CONSISTENT
    )
    assert bool(consistent.minimum_norm.stationarity_verified)

    # The preconditioned iteration ends at a weighted least-squares point; its
    # witness must still be an exact unit left-null vector separating b.
    inconsistent = la.solve(problem, jnp.asarray(_INCONSISTENT), policy=policy)
    assert int(inconsistent.status) == int(la.LinearSolveStatus.INCOMPATIBLE_RHS)
    evidence = inconsistent.minimum_norm
    assert evidence is not None
    assert bool(evidence.incompatible)
    witness = np.asarray(evidence.left_null_witness)
    np.testing.assert_allclose(np.sqrt(witness @ (target * witness)), 1.0, rtol=1e-10)
    adjoint_image = (_MATRIX.T @ (target * witness)) / source
    assert np.sqrt(adjoint_image @ (source * adjoint_image)) < 1e-8
    np.testing.assert_allclose(
        float(evidence.incompatibility_margin),
        abs(witness @ (target * _INCONSISTENT)),
        rtol=1e-10,
    )
    # Any source vector misses b by at least the margin minus the left-null
    # leakage; the exact least-squares residual is the tightest such bound.
    least_squares = _weighted_pinv(_MATRIX, source, target) @ _INCONSISTENT
    residual = _INCONSISTENT - _MATRIX @ least_squares
    assert float(evidence.incompatibility_margin) <= np.sqrt(
        residual @ (target * residual)
    ) * (1.0 + 1e-8)


def test_auto_planning_selects_craig_only_with_preconditioning_and_fingerprints_assembly() -> (
    None
):
    problem = la.MinimumNormProblem(_sparse_operator(False))
    preconditioning = la.PreconditioningPolicy(la.JacobiPreconditionerBuilder())
    assert la.plan(problem, la.LinearSolvePolicy()).method != "craig"
    selected = la.plan(problem, la.LinearSolvePolicy(preconditioning=preconditioning))
    assert selected.method == "craig"
    matrix_free = la.plan(
        problem, la.LinearSolvePolicy(la.Craig(), preconditioning=preconditioning)
    )
    assembled = la.plan(
        problem,
        la.LinearSolvePolicy(
            la.Craig(assembly=la.SparseAssemblyPolicy()), preconditioning=preconditioning
        ),
    )
    assert matrix_free.plan_id != assembled.plan_id


def test_craig_classifies_ill_conditioned_incompatibility_before_roundoff_growth() -> (
    None
):
    # Redundant second differences: B B* has condition number 1.9e6 on its range,
    # so an exact-tolerance (relative = 0) solve cannot reach eps-level
    # stationarity; past its attainable floor, MINRES on the inconsistent singular
    # system only amplifies roundoff along null(B*). Oracle: dense NumPy SVD.
    size = 60
    second = np.zeros((size - 2, size))
    for row in range(size - 2):
        second[row, row : row + 3] = (1.0, -2.0, 1.0)
    matrix = np.vstack((second, 2.0 * second[: size // 4]))
    left, singular, _ = np.linalg.svd(matrix)
    rank = int(np.sum(singular > singular[0] * 1e-12))
    rng = np.random.default_rng(1)
    consistent = matrix @ rng.standard_normal(size)
    consistent /= np.linalg.norm(consistent)
    inconsistent = consistent + 0.5 * left[:, rank:] @ rng.standard_normal(
        matrix.shape[0] - rank
    ) / np.sqrt(matrix.shape[0] - rank)
    rows, columns = np.nonzero(matrix)
    operator = SparseCoordinateOperator(
        EdgeRelation(
            columns.astype(np.int32),
            rows.astype(np.int32),
            source_size=size,
            target_size=matrix.shape[0],
        ),
        jnp.asarray(matrix[rows, columns]),
        source=la.ArraySpace((size,), dtype=jnp.float64, space_id="z"),
        target=la.ArraySpace((matrix.shape[0],), dtype=jnp.float64, space_id="b"),
    )
    policy = la.LinearSolvePolicy(
        la.Craig(),
        tolerance=la.TolerancePolicy(relative=0.0, absolute=1e-10, max_steps=4 * size),
        preconditioning=la.PreconditioningPolicy(la.JacobiPreconditionerBuilder()),
    )
    pseudoinverse = np.linalg.pinv(matrix, rcond=1e-12)
    problem = la.MinimumNormProblem(
        operator, rank_certificate=la.certify_rectangular_rank(operator, _CERTIFIED)
    )

    solved = la.solve(problem, jnp.asarray(consistent), policy=policy)
    assert int(solved.status) == int(la.LinearSolveStatus.SUCCESS)
    expected = pseudoinverse @ consistent
    np.testing.assert_allclose(
        np.asarray(solved.value), expected, atol=1e-8 * np.abs(expected).max()
    )

    refused = la.solve(problem, jnp.asarray(inconsistent), policy=policy)
    assert int(refused.status) == int(la.LinearSolveStatus.INCOMPATIBLE_RHS)
    assert int(refused.diagnostics.iterations) < 4 * size
    evidence = refused.minimum_norm
    assert evidence is not None
    assert bool(evidence.incompatible)
    value = np.asarray(refused.value)
    assert np.all(np.isfinite(value))
    witness = np.asarray(evidence.left_null_witness)
    np.testing.assert_allclose(np.linalg.norm(witness), 1.0, rtol=1e-10)
    leakage = np.linalg.norm(matrix.T @ witness)
    assert leakage < 1e-5
    margin = abs(witness @ inconsistent)
    np.testing.assert_allclose(float(evidence.incompatibility_margin), margin, rtol=1e-10)
    # The certificate excludes every solution out to a radius far beyond the
    # computed point, and cannot exceed the true distance of b from range(B).
    assert (margin - 1e-10) / leakage > 1e3 * np.linalg.norm(value)
    distance = np.linalg.norm(inconsistent - matrix @ (pseudoinverse @ inconsistent))
    assert margin <= distance + leakage * np.linalg.norm(pseudoinverse @ inconsistent)


@pytest.mark.parametrize("route", tuple(_ROUTES), ids=tuple(_ROUTES))
def test_redundant_rectangular_minimum_norm_matches_dense_svd_oracle(route: str) -> None:
    method, weighted = _ROUTES[route]
    operator = _operator(weighted)
    result = la.solve(
        la.MinimumNormProblem(operator),
        jnp.asarray(_CONSISTENT),
        policy=la.LinearSolvePolicy(method, tolerance=_TOLERANCE),
    )
    source = _SOURCE_WEIGHTS if weighted else np.ones(4)
    target = _TARGET_WEIGHTS if weighted else np.ones(6)
    expected = _weighted_pinv(_MATRIX, source, target) @ _CONSISTENT

    assert int(result.status) == int(la.LinearSolveStatus.SUCCESS)
    np.testing.assert_allclose(np.asarray(result.value), expected, rtol=1e-8, atol=1e-10)
    evidence = result.minimum_norm
    assert evidence is not None
    assert float(evidence.constraint_residual) < 1e-9 * np.linalg.norm(_CONSISTENT)
    assert not bool(evidence.incompatible)
    if isinstance(method, la.DenseSVD):
        assert np.isnan(float(evidence.stationarity_residual))
        assert not bool(evidence.stationarity_verified)
    else:
        assert bool(evidence.stationarity_verified)
        assert float(evidence.stationarity_residual) < 1e-9 * np.linalg.norm(expected)
        # Witness work is charged beyond the primal recurrence.
        assert (
            int(result.diagnostics.matvec_count) > int(result.diagnostics.iterations) + 3
        )


@pytest.mark.parametrize("route", tuple(_ROUTES), ids=tuple(_ROUTES))
def test_inconsistent_rhs_is_incompatible_with_validated_left_null_witness(
    route: str,
) -> None:
    method, weighted = _ROUTES[route]
    operator = _operator(weighted)
    result = la.solve(
        la.MinimumNormProblem(
            operator, rank_certificate=la.certify_rectangular_rank(operator, _CERTIFIED)
        ),
        jnp.asarray(_INCONSISTENT),
        policy=la.LinearSolvePolicy(method, tolerance=_TOLERANCE),
    )
    target = _TARGET_WEIGHTS if weighted else np.ones(6)
    source = _SOURCE_WEIGHTS if weighted else np.ones(4)
    least_squares = _weighted_pinv(_MATRIX, source, target) @ _INCONSISTENT
    residual = _INCONSISTENT - _MATRIX @ least_squares
    residual_norm = np.sqrt(residual @ (target * residual))

    assert int(result.status) == int(la.LinearSolveStatus.INCOMPATIBLE_RHS)
    assert not bool(result.successful)
    assert not bool(result.derivative_valid)
    evidence = result.minimum_norm
    assert evidence is not None
    assert bool(evidence.incompatible)
    np.testing.assert_allclose(
        float(evidence.constraint_residual), residual_norm, rtol=1e-8
    )
    witness = np.asarray(evidence.left_null_witness)
    # Independent oracle: the witness is a unit N-norm left-null vector of A ...
    np.testing.assert_allclose(np.sqrt(witness @ (target * witness)), 1.0, rtol=1e-10)
    adjoint_image = (_MATRIX.T @ (target * witness)) / source
    assert np.sqrt(adjoint_image @ (source * adjoint_image)) < 1e-9
    np.testing.assert_allclose(float(evidence.left_null_residual), 0.0, atol=1e-9)
    # ... whose pairing with b separates b from range(A) by the residual norm.
    np.testing.assert_allclose(
        float(evidence.incompatibility_margin),
        abs(witness @ (target * _INCONSISTENT)),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        float(evidence.incompatibility_margin), residual_norm, rtol=1e-8
    )
    assert float(evidence.incompatibility_radius) > np.sqrt(
        least_squares @ (source * least_squares)
    )


@pytest.mark.parametrize(
    "method", (la.LSMR(), la.GeneralizedLSMR()), ids=("lsmr", "glsmr")
)
def test_absolute_constraint_tolerance_is_met_by_the_true_residual(
    method: la.AbstractLinearMethod,
) -> None:
    # Full-row-rank 60x100 constraint with smallest singular value 1e-2. An
    # absolute normal-residual stop ||A^T r|| <= 1e-9 would accept ||r|| up to
    # 1e-7; the exact constraint must instead reach ||r|| <= 1e-9.
    rng = np.random.default_rng(7)
    left = np.linalg.qr(rng.standard_normal((60, 60)))[0]
    right = np.linalg.qr(rng.standard_normal((100, 60)))[0]
    matrix = left @ np.diag(np.logspace(0.0, -2.0, 60)) @ right.T
    rhs = matrix @ rng.standard_normal(100)
    result = la.solve(
        la.MinimumNormProblem(la.DenseLinearOperator(jnp.asarray(matrix))),
        jnp.asarray(rhs),
        policy=la.LinearSolvePolicy(
            method,
            tolerance=la.TolerancePolicy(relative=0.0, absolute=1e-9, max_steps=2000),
        ),
    )
    assert int(result.status) == int(la.LinearSolveStatus.SUCCESS)
    assert np.linalg.norm(rhs - matrix @ np.asarray(result.value)) <= 1e-9
    np.testing.assert_allclose(
        np.asarray(result.value), np.linalg.pinv(matrix) @ rhs, rtol=0.0, atol=1e-6
    )


def test_unconverged_iteration_is_not_classified_incompatible() -> None:
    result = la.solve(
        la.MinimumNormProblem(_operator(False)),
        jnp.asarray(_CONSISTENT),
        policy=la.LinearSolvePolicy(
            la.LSMR(),
            tolerance=la.TolerancePolicy(relative=1e-12, absolute=1e-14, max_steps=1),
        ),
    )
    evidence = result.minimum_norm
    assert evidence is not None
    assert int(result.status) == int(la.LinearSolveStatus.MAXIMUM_STEPS_REACHED)
    assert not bool(evidence.incompatible)


@pytest.mark.parametrize(
    "method",
    (la.LSMR(damping=0.1), la.GeneralizedLSMR(damping=0.1), la.DenseSVD(damping=0.1)),
    ids=("lsmr", "generalized-lsmr", "dense-svd"),
)
def test_minimum_norm_refuses_damping(method: la.AbstractLinearMethod) -> None:
    with pytest.raises(ValueError, match="exact constrained minimum-norm objective"):
        la.plan(la.MinimumNormProblem(_operator(False)), la.LinearSolvePolicy(method))


def test_metric_transform_and_relaxed_stacked_objective_match_dense_oracle() -> None:
    scale = np.asarray([1.0, 2.0, 0.5, 1.5])
    row_scale = np.asarray([1.0, 2.0, 1.0, 0.5, 1.0, 3.0])
    rho = 4.0
    policy = la.LinearSolvePolicy(la.GeneralizedLSMR(), tolerance=_TOLERANCE)
    transformed = _operator(False) @ la.DiagonalLinearOperator(jnp.asarray(scale))

    exact = la.solve(
        la.MinimumNormProblem(transformed), jnp.asarray(_CONSISTENT), policy=policy
    )
    exact_expected = np.linalg.pinv(_MATRIX * scale, rcond=1e-10) @ _CONSISTENT
    assert int(exact.status) == int(la.LinearSolveStatus.SUCCESS)
    np.testing.assert_allclose(
        np.asarray(exact.value), exact_expected, rtol=1e-8, atol=1e-10
    )

    stacked = la.StackedLinearOperator(
        (
            la.DiagonalLinearOperator(jnp.asarray(np.sqrt(rho) / row_scale))
            @ transformed,
            la.IdentityLinearOperator(transformed.source),
        )
    )
    rhs = _INCONSISTENT
    relaxed = la.solve(
        la.LeastSquaresProblem(stacked),
        (jnp.asarray(np.sqrt(rho) * rhs / row_scale), jnp.zeros(4)),
        policy=policy,
    )
    # Oracle: minimize 0.5 |z|^2 + rho/2 |D^-1 (A S z - b)|^2 by normal equations.
    design = (_MATRIX * scale) / row_scale[:, None]
    relaxed_expected = np.linalg.solve(
        np.eye(4) + rho * design.T @ design, rho * design.T @ (rhs / row_scale)
    )
    assert int(relaxed.status) == int(la.LinearSolveStatus.SUCCESS)
    np.testing.assert_allclose(np.asarray(relaxed.value), relaxed_expected, rtol=1e-8)
    assert stacked.properties.certifies("rank")
    assert stacked.properties.rank == 4
    estimate = la.plan(la.LeastSquaresProblem(stacked), policy).candidates[-1]
    assert estimate.row_blocks == (6, 4)
    assert _TOLERANCE.max_steps is not None
    assert estimate.forward_actions_per_rhs == 2 * _TOLERANCE.max_steps + 3


_CERTIFIED = la.svd.SVDSolvePolicy(
    la.svd.DenseSVD(), rank=la.RankPolicy(relative_cutoff=1e-8)
)


def test_dense_svd_rank_certificate_carries_native_exact_evidence() -> None:
    operator = la.DenseLinearOperator(jnp.asarray(_MATRIX), operator_id="redundant-rows")
    certificate = la.certify_rectangular_rank(operator, _CERTIFIED)
    singular_values = np.linalg.svd(_MATRIX, compute_uv=False)
    evidence = certificate.rank_evidence
    assert certificate.route == "dense-svd"
    assert evidence is not None
    assert evidence.certificate_kind == "exact-spectrum"
    assert int(la.svd.require_exact_svd_rank(evidence)) == 3
    assert int(certificate.rank) == np.linalg.matrix_rank(_MATRIX) == 3
    assert int(certificate.right_nullity) == 1
    assert int(certificate.left_nullity) == 3
    np.testing.assert_allclose(
        float(certificate.smallest_retained_singular_value),
        singular_values[2],
        rtol=1e-10,
    )
    assert (
        float(certificate.largest_discarded_singular_value) <= 1e-12 * singular_values[0]
    )
    assert bool(certificate.fixed_rank)
    assert bool(certificate.matches(operator))
    stale = la.DenseLinearOperator(
        jnp.asarray(_MATRIX + 1e-3 * _RNG.standard_normal(_MATRIX.shape)),
        operator_id="redundant-rows",
    )
    assert not bool(certificate.matches(stale))

    # At the roundoff cutoff a redundant row cannot be separated from the cutoff.
    roundoff = la.certify_rectangular_rank(operator)
    assert not bool(roundoff.fixed_rank)

    with pytest.raises(ValueError, match="scope"):
        la.MinimumNormProblem(
            la.DenseLinearOperator(jnp.asarray(_MATRIX), operator_id="other"),
            rank_certificate=certificate,
        )


@pytest.mark.parametrize("algorithm", ("divide-and-conquer", "qr"))
def test_dense_svd_drivers_certify_exactly_clustered_spectrum(
    algorithm: la.svd.DenseSVDAlgorithm,
) -> None:
    # Kronecker-sum lattice Laplacian: exactly repeated eigenvalues, the spectrum
    # class on which divide-and-conquer drivers can fail. Oracle: closed form.
    size = 12
    line = 2.0 * np.eye(size) - np.eye(size, k=1) - np.eye(size, k=-1)
    lattice = np.kron(line, np.eye(size)) + np.kron(np.eye(size), line)
    certificate = la.certify_rectangular_rank(
        la.DenseLinearOperator(jnp.asarray(lattice)),
        la.svd.SVDSolvePolicy(
            la.svd.DenseSVD(algorithm=algorithm), rank=la.RankPolicy(relative_cutoff=1e-8)
        ),
    )
    modes = 2.0 - 2.0 * np.cos(np.pi * np.arange(1, size + 1) / (size + 1))
    assert int(certificate.svd_status) == int(la.svd.SVDSolveStatus.SUCCESS)
    assert int(certificate.rank) == size * size
    np.testing.assert_allclose(
        float(certificate.smallest_retained_singular_value), 2.0 * modes[0], rtol=1e-10
    )
    assert bool(certificate.fixed_rank)


def test_failed_svd_certificate_reports_unavailable_rank() -> None:
    # A non-finite operator fails native SVD preparation; the certificate must
    # carry that status and no rank, not a computed-looking spectrum.
    matrix = _MATRIX.copy()
    matrix[0, 0] = np.inf
    certificate = la.certify_rectangular_rank(
        la.DenseLinearOperator(jnp.asarray(matrix)), _CERTIFIED
    )
    assert certificate.route == "dense-svd"
    assert int(certificate.svd_status) == int(la.svd.SVDSolveStatus.PREPARATION_FAILED)
    assert int(certificate.rank) == -1
    assert not bool(certificate.converged)
    assert np.isnan(float(certificate.smallest_retained_singular_value))
    assert np.isnan(float(certificate.largest_discarded_singular_value))
    assert not bool(certificate.fixed_rank)


def test_rank_certificate_beyond_svd_budget_is_explicitly_unavailable() -> None:
    refused = la.certify_rectangular_rank(
        la.DenseLinearOperator(jnp.asarray(_MATRIX)),
        la.svd.SVDSolvePolicy(
            la.svd.DenseSVD(), materialization=la.MaterializationPolicy(max_entries=10)
        ),
    )
    assert refused.route == "unavailable"
    assert refused.rank_evidence is None
    assert not refused.cost.accepted
    assert "materialization limit" in refused.reason
    assert int(refused.rank) == -1
    assert not bool(refused.fixed_rank)


def test_randomized_rank_certificate_is_fixed_only_with_deterministic_coverage() -> None:
    covered = la.certify_rectangular_rank(
        la.DenseLinearOperator(jnp.asarray(_MATRIX)),
        la.svd.SVDSolvePolicy(
            la.svd.RandomizedSVD(), count=4, rank=la.RankPolicy(relative_cutoff=1e-8)
        ),
        key=jax.random.key(0),
    )
    assert covered.route == "randomized-svd"
    assert covered.rank_evidence is not None
    assert covered.rank_evidence.certificate_kind == "deterministic-frobenius"
    assert int(covered.rank) == 3
    assert bool(covered.fixed_rank)

    operator, _ = _sparse_wide(400, 600)
    sketched = la.certify_rectangular_rank(
        operator,
        la.svd.SVDSolvePolicy(la.svd.RandomizedSVD(), count=10),
        key=jax.random.key(1),
    )
    evidence = sketched.rank_evidence
    assert evidence is not None
    assert evidence.certificate_kind == "independent-gaussian"
    assert 0.0 < float(evidence.failure_probability) < 1.0
    assert int(evidence.lower_bound) <= int(evidence.upper_bound)
    assert not bool(evidence.deterministic_exact)
    assert not bool(sketched.fixed_rank)


_DIRECTION = _RNG.standard_normal((6, 3))
_RHS_DIRECTION = _RNG.standard_normal(3)
_PROBE = _RNG.standard_normal(4)


def _oracle_objective(parameter: float) -> float:
    # Fixed-rank path: A(t) = (L + t dL) R^T keeps rank three and redundant rows.
    matrix = (_LEFT + parameter * _DIRECTION) @ _RIGHT.T
    shift = np.concatenate((parameter * _RHS_DIRECTION, [0.0]))
    rhs = matrix @ (_SOLUTION + shift)
    return float(_PROBE @ (np.linalg.pinv(matrix, rcond=1e-10) @ rhs))


def _objective(policy: la.LinearSolvePolicy, certified: bool) -> Callable[[Array], Array]:
    def objective(parameter: Array) -> Array:
        matrix = (jnp.asarray(_LEFT) + parameter * jnp.asarray(_DIRECTION)) @ jnp.asarray(
            _RIGHT
        ).T
        operator = la.DenseLinearOperator(matrix, operator_id="fixed-rank-path")
        certificate = (
            la.certify_rectangular_rank(operator, _CERTIFIED) if certified else None
        )
        shift = jnp.concatenate((parameter * jnp.asarray(_RHS_DIRECTION), jnp.zeros(1)))
        rhs = matrix @ (jnp.asarray(_SOLUTION) + shift)
        result = la.solve(
            la.MinimumNormProblem(operator, rank_certificate=certificate),
            rhs,
            policy=policy,
        )
        return jnp.asarray(_PROBE) @ result.value

    return objective


def _finite_difference() -> float:
    step = 1e-6
    return (_oracle_objective(step) - _oracle_objective(-step)) / (2.0 * step)


@pytest.mark.parametrize("mode", ("reverse", "forward"))
def test_certified_minimum_norm_derivative_matches_finite_differences(mode: str) -> None:
    objective = _objective(
        la.LinearSolvePolicy(la.GeneralizedLSMR(), tolerance=_TOLERANCE), certified=True
    )
    parameter = jnp.asarray(0.0)
    derivative = (
        jax.grad(objective)(parameter)
        if mode == "reverse"
        else jax.jvp(objective, (parameter,), (jnp.asarray(1.0),))[1]
    )
    np.testing.assert_allclose(float(derivative), _finite_difference(), rtol=1e-5)


def test_dense_svd_rank_evidence_admits_the_minimum_norm_derivative() -> None:
    objective = _objective(
        la.LinearSolvePolicy(la.DenseSVD(), rank=la.RankPolicy(relative_cutoff=1e-8)),
        certified=False,
    )
    np.testing.assert_allclose(
        float(jax.grad(objective)(jnp.asarray(0.0))), _finite_difference(), rtol=1e-5
    )


def test_minimum_norm_operator_derivative_is_refused_without_rank_certificate() -> None:
    policy = la.LinearSolvePolicy(la.GeneralizedLSMR(), tolerance=_TOLERANCE)
    gradient = jax.grad(_objective(policy, certified=False))(jnp.asarray(0.0))
    assert np.isnan(float(gradient))

    result = la.solve(
        la.MinimumNormProblem(_operator(False)), jnp.asarray(_CONSISTENT), policy=policy
    )
    assert bool(result.successful)
    assert result.derivative_regular is not None
    assert not bool(result.derivative_regular)
    assert not bool(result.derivative_valid)


def test_certified_low_rank_rhs_derivative_preserves_out_of_range_action() -> None:
    policy = la.LinearSolvePolicy(
        la.GeneralizedLSMR(),
        tolerance=_TOLERANCE,
        differentiation=la.DifferentiationPolicy("rhs-only"),
    )
    operator = _operator(False)
    problem = la.MinimumNormProblem(
        operator, rank_certificate=la.certify_rectangular_rank(operator, _CERTIFIED)
    )

    def objective(rhs: Array) -> Array:
        return jnp.asarray(_PROBE) @ la.solve(problem, rhs, policy=policy).value

    gradient = jax.grad(objective)(jnp.asarray(_CONSISTENT))
    np.testing.assert_allclose(
        np.asarray(gradient), np.linalg.pinv(_MATRIX).T @ _PROBE, rtol=1e-7, atol=1e-9
    )


def _dense_wide(rows: int, columns: int) -> tuple[la.AbstractLinearOperator, np.ndarray]:
    matrix = np.random.default_rng(3).standard_normal((rows, columns))
    return la.DenseLinearOperator(jnp.asarray(matrix)), matrix


def _sparse_wide(rows: int, columns: int) -> tuple[la.AbstractLinearOperator, np.ndarray]:
    # Diagonally anchored random pattern: full row rank with five extra entries
    # per row, stored without duplicate routes.
    rng = np.random.default_rng(11)
    entries: dict[tuple[int, int], float] = {}
    for row in range(rows):
        entries[(row, row)] = 4.0
        for column in rng.choice(columns, size=5, replace=False):
            entries.setdefault((row, int(column)), float(rng.standard_normal()))
    keys = sorted(entries)
    target_rows = np.asarray([key[0] for key in keys], dtype=np.int32)
    source_columns = np.asarray([key[1] for key in keys], dtype=np.int32)
    values = np.asarray([entries[key] for key in keys])
    relation = EdgeRelation(
        source_columns, target_rows, source_size=columns, target_size=rows
    )
    operator = SparseCoordinateOperator(
        relation,
        jnp.asarray(values),
        source=la.ArraySpace((columns,), dtype=jnp.float64),
        target=la.ArraySpace((rows,), dtype=jnp.float64),
    )
    matrix = np.zeros((rows, columns))
    matrix[target_rows, source_columns] = values
    return operator, matrix


_WIDE_CASES = {
    "dense-80x110": (_dense_wide, 80, 110),
    "sparse-400x600": (_sparse_wide, 400, 600),
}


@pytest.mark.parametrize("case", tuple(_WIDE_CASES), ids=tuple(_WIDE_CASES))
def test_rhs_derivative_vjp_matches_jvp_and_pseudoinverse_transpose(case: str) -> None:
    # Cotangent columns lie outside range(A*), so the reverse pass solves an
    # inconsistent adjoint least-squares problem; it must reach its stationary
    # point rather than be rejected as nonconverged.
    builder, rows, columns = _WIDE_CASES[case]
    operator, matrix = builder(rows, columns)
    problem = la.MinimumNormProblem(
        operator, rank_certificate=la.certify_rectangular_rank(operator, _CERTIFIED)
    )
    rng = np.random.default_rng(29)
    rhs = jnp.asarray(rng.standard_normal(rows))
    probe = rng.standard_normal(columns)
    direction = rng.standard_normal(rows)
    policy = la.LinearSolvePolicy(
        la.GeneralizedLSMR(),
        tolerance=la.TolerancePolicy(relative=0.0, absolute=1e-10, max_steps=4096),
        differentiation=la.DifferentiationPolicy("rhs-only"),
        failure=la.FailurePolicy("status"),
    )

    def objective(right: Array) -> Array:
        return jnp.asarray(probe) @ la.solve(problem, right, policy=policy).value

    gradient = np.asarray(jax.grad(objective)(rhs))
    tangent = float(jax.jvp(objective, (rhs,), (jnp.asarray(direction),))[1])
    expected = np.linalg.pinv(matrix).T @ probe
    assert np.all(np.isfinite(gradient))
    # The default derivative-solve tolerance (relative 1e-8 backward error)
    # bounds the forward error by roughly cond(A) * 1e-8.
    assert np.linalg.norm(gradient - expected) <= 1e-6 * np.linalg.norm(expected)
    np.testing.assert_allclose(tangent, expected @ direction, rtol=1e-6)
    np.testing.assert_allclose(gradient @ direction, tangent, rtol=1e-5)


def test_certificate_rejects_same_probe_rank_changing_numeric_revision() -> None:
    operator = la.DenseLinearOperator(jnp.eye(2, dtype=jnp.float64), operator_id="same")
    certificate = la.certify_rectangular_rank(operator, _CERTIFIED)
    # Independent construction of the previously accepted adversarial update:
    # the rank-one projector agrees with I on the old deterministic probe.
    index = np.arange(1, 3, dtype=np.float64)
    probe = np.sin(index * 1.6180339887498949) + 0.5 * np.cos(index * 0.7071067811865476)
    probe /= np.linalg.norm(probe)
    replacement = la.DenseLinearOperator(
        jnp.asarray(np.outer(probe, probe)), operator_id="same"
    )
    assert not bool(certificate.matches(replacement))
    result = la.solve(
        la.MinimumNormProblem(replacement, rank_certificate=certificate),
        jnp.asarray(probe),
        policy=la.LinearSolvePolicy(la.GeneralizedLSMR(), tolerance=_TOLERANCE),
    )
    assert result.minimum_norm is not None
    assert int(result.diagnostics.rank) != 2
    assert not bool(result.derivative_regular)


def test_certificate_binds_numeric_pairing_state() -> None:
    matrix = jnp.eye(2, dtype=jnp.float64)
    original = la.DenseLinearOperator(
        matrix,
        target=la.ArraySpace(
            (2,),
            dtype=jnp.float64,
            space_id="target",
            pairing=la.DiagonalPairing(jnp.ones(2, dtype=jnp.float64)),
        ),
        operator_id="paired",
    )
    certificate = la.certify_rectangular_rank(original, _CERTIFIED)
    replacement = la.DenseLinearOperator(
        matrix,
        target=la.ArraySpace(
            (2,),
            dtype=jnp.float64,
            space_id="target",
            pairing=la.DiagonalPairing(jnp.asarray([1.0, 2.0], dtype=jnp.float64)),
        ),
        operator_id="paired",
    )
    assert not bool(certificate.matches(replacement))


def test_certificate_requires_exact_numeric_revision_even_below_roundoff_tolerance() -> (
    None
):
    matrix = np.eye(2, dtype=np.float64)
    original = la.DenseLinearOperator(jnp.asarray(matrix), operator_id="exact-revision")
    certificate = la.certify_rectangular_rank(original, _CERTIFIED)
    matrix[0, 0] = np.nextafter(matrix[0, 0], np.inf)
    replacement = la.DenseLinearOperator(
        jnp.asarray(matrix), operator_id="exact-revision"
    )
    assert not bool(certificate.matches(replacement))


@pytest.mark.parametrize("scale", (1.0, 1e20, 1e-20), ids=("unit", "large", "small"))
@pytest.mark.parametrize("route", ("craig", "minres"))
def test_lanczos_operator_norm_is_independent_of_rhs_scale(
    scale: float, route: str
) -> None:
    matrix = jnp.diag(jnp.asarray([1.0, 2.0], dtype=jnp.float64))
    operator = la.DenseLinearOperator(
        matrix,
        properties=la.OperatorProperties(
            self_adjoint=True, evidence={"self_adjoint": "asserted"}
        ),
    )
    problem = (
        la.MinimumNormProblem(operator) if route == "craig" else la.LinearSystem(operator)
    )
    method = la.Craig() if route == "craig" else la.MINRES()
    result = la.solve(
        problem,
        scale * jnp.ones(2, dtype=jnp.float64),
        policy=la.LinearSolvePolicy(
            method,
            tolerance=la.TolerancePolicy(relative=1e-12, absolute=0.0, max_steps=4),
        ),
    )
    assert int(result.status) == int(la.LinearSolveStatus.SUCCESS)
    np.testing.assert_allclose(np.asarray(result.value) / scale, [1.0, 0.5], rtol=1e-12)
    if result.minimum_norm is not None:
        assert not bool(result.minimum_norm.incompatible)


@pytest.mark.parametrize("certified", (False, True), ids=("uncertified", "full-row-rank"))
def test_finite_exclusion_ball_does_not_make_invertible_rhs_incompatible(
    certified: bool,
) -> None:
    matrix = np.diag(np.asarray([1.0, 1e-6], dtype=np.float64))
    operator = la.DenseLinearOperator(jnp.asarray(matrix))
    certificate = la.certify_rectangular_rank(operator, _CERTIFIED) if certified else None
    result = la.solve(
        la.MinimumNormProblem(operator, rank_certificate=certificate),
        jnp.ones(2, dtype=jnp.float64),
        policy=la.LinearSolvePolicy(
            la.LSMR(),
            tolerance=la.TolerancePolicy(relative=1e-4, absolute=0.0, max_steps=10),
        ),
    )
    evidence = result.minimum_norm
    assert evidence is not None
    assert not bool(result.successful)
    assert int(result.status) != int(la.LinearSolveStatus.INCOMPATIBLE_RHS)
    assert not bool(evidence.incompatible)
    radius = float(evidence.incompatibility_radius)
    assert np.isfinite(radius)
    assert np.linalg.norm(np.asarray(result.value)) < radius
    # The exact solution is outside the certified finite ball.
    assert radius < np.linalg.norm(np.linalg.solve(matrix, np.ones(2)))


@pytest.mark.parametrize("certified", (False, True), ids=("uncertified", "full-row-rank"))
def test_rhs_tangent_requires_range_correct_action_for_retained_singular_direction(
    certified: bool,
) -> None:
    operator = la.DenseLinearOperator(
        jnp.diag(jnp.asarray([1.0, 1e-10], dtype=jnp.float64))
    )
    certificate = (
        la.certify_rectangular_rank(
            operator,
            la.svd.SVDSolvePolicy(
                la.svd.DenseSVD(algorithm="qr"),
                rank=la.RankPolicy(relative_cutoff=1e-12),
            ),
        )
        if certified
        else None
    )
    problem = la.MinimumNormProblem(operator, rank_certificate=certificate)
    policy = la.LinearSolvePolicy(
        la.LSMR(), differentiation=la.DifferentiationPolicy("rhs-only")
    )

    def solve_rhs(rhs: Array) -> Array:
        return la.solve(problem, rhs, policy=policy).value

    value, tangent = jax.jvp(
        solve_rhs,
        (jnp.asarray([1.0, 0.0], dtype=jnp.float64),),
        (jnp.ones(2, dtype=jnp.float64),),
    )
    np.testing.assert_allclose(value, [1.0, 0.0], atol=1e-12)
    derivative = np.asarray(tangent)
    if np.all(np.isfinite(derivative)):
        np.testing.assert_allclose(
            np.diag([1.0, 1e-10]) @ derivative, np.ones(2), rtol=1e-8, atol=1e-10
        )
    else:
        assert np.all(np.isnan(derivative))


def test_exact_out_of_range_rhs_action_preserves_pseudoinverse_jvp_and_vjp() -> None:
    matrix = np.diag(np.asarray([1.0, 0.0], dtype=np.float64))
    operator = la.DenseLinearOperator(jnp.asarray(matrix))
    policy = la.LinearSolvePolicy(
        la.LSMR(), differentiation=la.DifferentiationPolicy("rhs-only")
    )

    def solve_rhs(rhs: Array) -> Array:
        return la.solve(la.MinimumNormProblem(operator), rhs, policy=policy).value

    rhs = jnp.asarray([1.0, 0.0], dtype=jnp.float64)
    direction = jnp.ones(2, dtype=jnp.float64)
    _, tangent = jax.jvp(solve_rhs, (rhs,), (direction,))
    _, pullback = jax.vjp(solve_rhs, rhs)
    cotangent = pullback(direction)[0]
    expected = np.linalg.pinv(matrix) @ np.ones(2)
    np.testing.assert_allclose(tangent, expected, atol=1e-12)
    np.testing.assert_allclose(cotangent, expected, atol=1e-12)


def test_wide_dense_svd_solution_jvp_stays_within_admitted_workspace() -> None:
    columns = 1000
    budget = 1 << 20
    matrix = np.linspace(0.5, 1.5, columns, dtype=np.float64)[None, :]
    direction = np.cos(np.arange(columns, dtype=np.float64))[None, :]
    policy = la.LinearSolvePolicy(
        la.DenseSVD(),
        resources=la.SolveResourcePolicy(
            workspace_bytes=budget, krylov_basis_bytes=budget
        ),
    )

    def solution(design: Array) -> Array:
        return la.solve(
            la.LeastSquaresProblem(
                la.DenseLinearOperator(design, operator_id="wide-factor-action")
            ),
            jnp.ones(1, dtype=jnp.float64),
            policy=policy,
        ).value

    def derivative(design: Array, tangent: Array) -> tuple[Array, Array]:
        return jax.jvp(solution, (design,), (tangent,))

    design = jnp.asarray(matrix)
    tangent = jnp.asarray(direction)
    executable = jax.jit(derivative).lower(design, tangent).compile()
    memory = executable.memory_analysis()
    assert memory is not None
    assert memory.temp_size_in_bytes + memory.output_size_in_bytes <= budget
    value, actual = executable(design, tangent)
    norm_squared = float(matrix[0] @ matrix[0])
    expected_value = matrix[0] / norm_squared
    expected_tangent = (
        direction[0] / norm_squared
        - 2.0 * matrix[0] * (matrix[0] @ direction[0]) / norm_squared**2
    )
    np.testing.assert_allclose(value, expected_value, rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(actual, expected_tangent, rtol=1e-10, atol=1e-13)


@pytest.mark.parametrize("mode", ("forward", "reverse"))
def test_weighted_rank_deficient_least_squares_solution_derivative_is_complete(
    mode: str,
) -> None:
    right_direction = np.random.default_rng(7).standard_normal(_RIGHT.shape)
    policy = la.LinearSolvePolicy(la.DenseSVD(), rank=la.RankPolicy(relative_cutoff=1e-8))
    source = la.ArraySpace(
        (4,),
        dtype=jnp.float64,
        pairing=la.DiagonalPairing(jnp.asarray(_SOURCE_WEIGHTS)),
    )
    target = la.ArraySpace(
        (6,),
        dtype=jnp.float64,
        pairing=la.DiagonalPairing(jnp.asarray(_TARGET_WEIGHTS)),
    )

    def objective(parameter: Array) -> Array:
        matrix = (jnp.asarray(_LEFT) + parameter * jnp.asarray(_DIRECTION)) @ (
            jnp.asarray(_RIGHT) + parameter * jnp.asarray(right_direction)
        ).T
        result = la.solve(
            la.LeastSquaresProblem(
                la.DenseLinearOperator(matrix, source=source, target=target)
            ),
            jnp.asarray(_INCONSISTENT),
            policy=policy,
        )
        return jnp.asarray(_PROBE) @ result.value

    def oracle(parameter: float) -> float:
        matrix = (_LEFT + parameter * _DIRECTION) @ (
            _RIGHT + parameter * right_direction
        ).T
        value = _weighted_pinv(matrix, _SOURCE_WEIGHTS, _TARGET_WEIGHTS) @ _INCONSISTENT
        return float(_PROBE @ value)

    step = 1e-6
    expected = (oracle(step) - oracle(-step)) / (2.0 * step)
    parameter = jnp.asarray(0.0, dtype=jnp.float64)
    actual = (
        jax.jvp(objective, (parameter,), (jnp.asarray(1.0, dtype=jnp.float64),))[1]
        if mode == "forward"
        else jax.grad(objective)(parameter)
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_dense_svd_rank_evidence_admits_minimum_norm_jvp() -> None:
    objective = _objective(
        la.LinearSolvePolicy(la.DenseSVD(), rank=la.RankPolicy(relative_cutoff=1e-8)),
        certified=False,
    )
    _, tangent = jax.jvp(
        objective,
        (jnp.asarray(0.0, dtype=jnp.float64),),
        (jnp.asarray(1.0, dtype=jnp.float64),),
    )
    np.testing.assert_allclose(float(tangent), _finite_difference(), rtol=1e-5)
