# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from typing import ClassVar, NoReturn

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import PyTree

import phydrax as phx
from phydrax.linalg._svd_evidence import gaussian_range_bound, version_failure_probability


la = phx.linalg
svd = la.svd


class _Shell(la.AbstractLinearOperator):
    """Real fused coordinate actions, with dense access explicitly unavailable."""

    matrix: Array
    _fused_block_action_kind: ClassVar[str] = "fused"

    def __init__(
        self, matrix: Array, /, *, operator_id: str = "rectangular-shell"
    ) -> None:
        self.matrix = matrix
        self.source = la.ArraySpace((matrix.shape[1],), dtype=matrix.dtype)
        self.target = la.ArraySpace((matrix.shape[0],), dtype=matrix.dtype)
        self.batch_shape = ()
        self.properties = la.OperatorProperties()
        self.capabilities = la.OperatorCapabilities(
            transpose=True, adjoint=True, materialize=False
        )
        self.operator_id = operator_id

    def mv(self, vector: PyTree[Array], /) -> Array:
        return self.matrix @ self.source.flatten(vector)

    def transpose_mv(self, vector: PyTree[Array], /) -> Array:
        return self.matrix.T @ self.target.flatten(vector)

    def adjoint_mv(self, vector: PyTree[Array], /) -> Array:
        return self.matrix.conj().T @ self.target.flatten(vector)

    def mv_block(self, vectors: ArrayLike, /) -> Array:
        return self.matrix @ jnp.asarray(vectors)

    def adjoint_mv_block(self, vectors: ArrayLike, /) -> Array:
        return self.matrix.conj().T @ jnp.asarray(vectors)

    def _materialize(self, /) -> NoReturn:
        raise AssertionError("Randomized SVD must never materialize the shell.")


def _matrix(rows: int, columns: int, dtype: np.dtype, /) -> Array:
    rng = np.random.default_rng(20261001)
    rank = min(rows, columns)
    left = rng.normal(size=(rows, rank))
    right = rng.normal(size=(columns, rank))
    if np.issubdtype(dtype, np.complexfloating):
        left = left + 1j * rng.normal(size=left.shape)
        right = right + 1j * rng.normal(size=right.shape)
    left, _ = np.linalg.qr(left)
    right, _ = np.linalg.qr(right)
    spectrum = np.geomspace(7, 0.02, rank)
    return jnp.asarray((left * spectrum) @ right.conj().T, dtype=dtype)


def _policy(
    count: int,
    /,
    *,
    oversampling: int = 2,
    power_iterations: int = 1,
    differentiation: svd.SVDDifferentiationMode = "none",
    refresh: la.ProbeRefresh = "reuse",
    require_leading: bool = False,
) -> svd.SVDSolvePolicy:
    return svd.SVDSolvePolicy(
        svd.RandomizedSVD(
            oversampling=oversampling,
            power_iterations=power_iterations,
            probe_refresh=refresh,
        ),
        count=count,
        tolerance=svd.SVDTolerancePolicy(residual=2, orthogonality=2e-5),
        differentiation=differentiation,
        approximation=svd.SVDApproximationPolicy(require_leading=require_leading),
    )


@pytest.mark.strict_jax
@pytest.mark.parametrize("rows,columns,count", [(14, 7, 3), (8, 20, 5)])
@pytest.mark.parametrize(
    "dtype",
    [
        np.dtype(np.float32),
        np.dtype(np.float64),
        np.dtype(np.complex64),
        np.dtype(np.complex128),
    ],
)
def test_rectangular_clipped_width_and_complex_triplets(
    rows: int, columns: int, count: int, dtype: np.dtype
) -> None:
    matrix = _matrix(rows, columns, dtype)
    policy = _policy(count, oversampling=8, power_iterations=2)
    prepared = svd.prepare_svd(
        svd.SVDProblem(la.DenseLinearOperator(matrix)), policy, key=jax.random.key(12)
    )
    assert isinstance(prepared.state, svd.RandomizedSVDState)
    result = svd.svd(prepared)
    expected = np.linalg.svd(np.asarray(matrix), compute_uv=False)[:count]
    assert prepared.plan.sketch_size == min(rows, columns)
    assert prepared.state.range_basis.shape == (rows, min(rows, columns))
    assert prepared.state.compressed_operator.shape == (min(rows, columns), columns)
    assert bool(result.successful)
    assert np.allclose(result.singular_values, expected, rtol=2e-5, atol=2e-5)
    values = result.singular_values.astype(result.left_coordinates.dtype)[None, :]
    assert np.allclose(
        matrix @ result.right_coordinates,
        result.left_coordinates * values,
        rtol=2e-5,
        atol=2e-5,
    )
    assert result.singular_values.dtype == matrix.real.dtype
    assert bool(result.leading_evidence.certified)


def test_nonmaterializing_shell_has_independent_audit_and_opaque_scratch() -> None:
    matrix = _matrix(13, 8, np.dtype(np.float64))
    problem = svd.SVDProblem(_Shell(matrix))
    prepared = svd.prepare_svd(problem, _policy(3), key=jax.random.key(11))
    assert isinstance(prepared.state, svd.RandomizedSVDState)
    result = svd.svd(prepared)
    assert bool(result.successful)
    assert result.range_evidence.certificate_kind == "independent-gaussian"
    assert result.range_evidence.total_energy is None
    assert result.range_evidence.independent_audit_assumption
    assert not result.range_evidence.provider_action_error_known
    assert not prepared.plan.cost.provider_workspace_exact
    observed_residual = np.linalg.norm(
        (
            np.eye(13)
            - np.asarray(prepared.state.range_basis)
            @ np.asarray(prepared.state.range_basis).conj().T
        )
        @ np.asarray(matrix),
        2,
    )
    assert float(result.range_evidence.spectral_upper_bound) >= observed_residual
    assert float(result.range_evidence.failure_probability) <= 1e-6 / 2
    assert prepared.plan.cost.preparation_forward_block_calls == 3
    assert prepared.plan.cost.preparation_adjoint_block_calls == 2
    assert (
        int(result.diagnostics.operator_matvec_count)
        == 2 * prepared.plan.sketch_size + 8 + 3
    )


@pytest.mark.parametrize("refresh", ["reuse", "redraw"])
def test_fixed_key_direct_prepared_refresh_identity(refresh: la.ProbeRefresh) -> None:
    matrix = _matrix(9, 7, np.dtype(np.float64))
    key = jax.random.key(6)
    problem = svd.SVDProblem(_Shell(matrix), problem_id="refreshable-range")
    policy = _policy(2, refresh=refresh)
    prepared = svd.prepare_svd(problem, policy, key=key)
    assert isinstance(prepared.state, svd.RandomizedSVDState)
    direct, repeated = svd.svd(problem, policy=policy, key=key), svd.svd(prepared)
    assert np.array_equal(direct.singular_values, repeated.singular_values)
    assert np.array_equal(
        svd.svd(prepared).range_evidence.spectral_upper_bound,
        repeated.range_evidence.spectral_upper_bound,
    )
    changed = svd.refresh_svd(
        prepared, svd.SVDProblem(_Shell(1.5 * matrix), problem_id=problem.problem_id)
    )
    assert isinstance(changed.state, svd.RandomizedSVDState)
    assert int(changed.numeric_version) == 1
    assert int(changed.state.audit_version) == 1
    assert int(changed.state.sketch_version) == (0 if refresh == "reuse" else 1)
    if refresh == "reuse":
        assert np.allclose(
            svd.svd(changed).singular_values, 1.5 * repeated.singular_values, atol=1e-11
        )
    else:
        first_projector = prepared.state.range_basis @ prepared.state.range_basis.conj().T
        second_projector = changed.state.range_basis @ changed.state.range_basis.conj().T
        assert not np.allclose(first_projector, second_projector, atol=1e-10)
    assert float(changed.state.audit_failure_probability) <= 1e-6 / 6
    assert float(changed.state.audit_failure_probability) < float(
        prepared.state.audit_failure_probability
    )
    with pytest.raises(ValueError, match="replacement key"):
        svd.svd(prepared, key=key)


def test_key_and_capacity_refusal_before_numerical_work() -> None:
    problem = svd.SVDProblem(la.DenseLinearOperator(jnp.eye(4)))
    with pytest.raises(ValueError, match="explicit scalar typed key"):
        svd.prepare_svd(problem, _policy(2))
    with pytest.raises(TypeError):
        svd.prepare_svd(problem, _policy(2), key=jax.random.PRNGKey(0))
    with pytest.raises(ValueError):
        svd.prepare_svd(problem, _policy(2), key=jax.random.split(jax.random.key(0), 2))
    with pytest.raises(ValueError, match="does not accept"):
        svd.prepare_svd(problem, key=jax.random.key(0))
    for value in [True, 2.5]:
        with pytest.raises(TypeError, match="integer"):
            svd.RandomizedSVD(oversampling=value)  # ty: ignore[invalid-argument-type]
        with pytest.raises(TypeError, match="integer"):
            svd.SVDResourcePolicy(workspace_bytes=value)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="largest"):
        svd.plan_svd(problem, svd.SVDSolvePolicy(svd.RandomizedSVD(), which="smallest"))
    with pytest.raises(ValueError, match="workspace bytes"):
        svd.plan_svd(
            problem,
            svd.SVDSolvePolicy(
                svd.RandomizedSVD(), resources=svd.SVDResourcePolicy(workspace_bytes=0)
            ),
        )
    with pytest.raises(ValueError, match="full coverage"):
        svd.plan_svd(
            problem,
            svd.SVDSolvePolicy(
                svd.RandomizedSVD(oversampling=0),
                count=2,
                rank=la.RankPolicy(require_full_rank=True),
            ),
        )


@pytest.mark.parametrize("rank", [0, 2])
def test_rank_deficient_primal_does_not_claim_algorithmic_derivative(rank: int) -> None:
    matrix = jnp.diag(
        jnp.asarray([4.0, 2.0, 0.0, 0.0]) if rank else jnp.zeros((4,), dtype=jnp.float64)
    )
    prepared = svd.prepare_svd(
        svd.SVDProblem(la.DenseLinearOperator(matrix)),
        _policy(2, oversampling=2),
        key=jax.random.key(4),
    )
    assert isinstance(prepared.state, svd.RandomizedSVDState)
    result = svd.svd(prepared)
    assert bool(result.successful)
    assert np.all(np.isfinite(result.singular_values))
    assert (
        int(result.rank_evidence.lower_bound)
        <= rank
        <= int(result.rank_evidence.upper_bound)
    )
    assert not bool(prepared.state.qr_full_rank)
    differentiated = svd.svd(
        svd.SVDProblem(la.DenseLinearOperator(matrix)),
        policy=_policy(2, oversampling=2, differentiation="projector"),
        key=jax.random.key(4),
    )
    assert int(differentiated.primal_status) == 0
    assert int(differentiated.derivative_status) == int(
        svd.SVDSolveStatus.DIFFERENTIATION_REJECTED
    )
    assert not bool(differentiated.derivative_valid)


def test_deterministic_direct_range_residual_and_projection_energy() -> None:
    matrix = _matrix(10, 7, np.dtype(np.complex128))
    prepared = svd.prepare_svd(
        svd.SVDProblem(la.DenseLinearOperator(matrix)),
        _policy(2, oversampling=1, power_iterations=0),
        key=jax.random.key(5),
    )
    assert isinstance(prepared.state, svd.RandomizedSVDState)
    result = svd.svd(prepared)
    basis = np.asarray(prepared.state.range_basis)
    residual = (np.eye(10) - basis @ basis.conj().T) @ np.asarray(matrix)
    observed = np.linalg.norm(residual, "fro")
    assert np.isclose(
        float(
            result.range_evidence.spectral_upper_bound
            - result.range_evidence.numerical_allowance
        ),
        observed,
        atol=1e-12,
    )
    assert float(result.range_evidence.failure_probability) == 0
    measured = np.sum(
        np.abs(np.asarray(matrix) @ np.asarray(result.right_coordinates)) ** 2, axis=0
    )
    assert np.allclose(result.diagnostics.projection_energies, measured, atol=1e-12)
    assert not np.allclose(
        measured, np.asarray(result.singular_values) ** 2, rtol=1e-8, atol=1e-8
    )
    factor = (
        np.asarray(result.left_coordinates * result.singular_values)
        @ np.asarray(result.right_coordinates).conj().T
    )
    actual_factor_residual = np.linalg.norm(np.asarray(matrix) - factor, 2)
    assert (
        actual_factor_residual
        <= float(result.range_evidence.factor_residual_upper_bound) + 1e-12
    )
    assert int(result.rank_evidence.upper_bound) > int(result.rank_evidence.lower_bound)
    with pytest.raises(eqx.EquinoxRuntimeError, match="Deterministic exact"):
        svd.require_exact_svd_rank(result)


@pytest.mark.parametrize("complex_probe", [False, True])
def test_rank_one_small_ball_constants_are_analytical(complex_probe: bool) -> None:
    delta, count, observed = 2e-5, 6, 0.37
    expected = (
        observed / np.sqrt(-np.log1p(-(delta ** (1 / count))))
        if complex_probe
        else np.sqrt(2 / np.pi) * observed * delta ** (-1 / count)
    )
    actual = gaussian_range_bound(
        jnp.asarray(observed, jnp.float64),
        jnp.asarray(delta, jnp.float64),
        count,
        complex_probe,
    )
    assert np.isclose(actual, expected, rtol=1e-14)
    probabilities = np.asarray(
        [
            version_failure_probability(
                1e-6, jnp.asarray(version, jnp.int32), jnp.dtype(jnp.float64)
            )
            for version in range(20)
        ]
    )
    assert np.all(probabilities <= 1e-6 / ((np.arange(20) + 1) * (np.arange(20) + 2)))
    assert probabilities.sum() < 1e-6


def test_exact_lower_triplet_cannot_certify_missed_leading_cluster() -> None:
    matrix = jnp.diag(jnp.asarray([9.0, 4.0, 1.0], dtype=jnp.float64))
    problem = svd.SVDProblem(la.DenseLinearOperator(matrix))
    prepared = svd.prepare_svd(
        problem,
        _policy(1, oversampling=0, power_iterations=0, require_leading=True),
        key=jax.random.key(0),
    )
    assert isinstance(prepared.state, svd.RandomizedSVDState)
    # Deliberately captured lower invariant range, with independent original residual evidence.
    basis = jnp.asarray([[0.0], [0.0], [1.0]], dtype=jnp.float64)
    altered = eqx.tree_at(
        lambda value: (
            value.range_basis,
            value.compressed_operator,
            value.range_bound,
            value.range_allowance,
        ),
        prepared.state,
        (
            basis,
            basis.T @ matrix,
            jnp.asarray(np.sqrt(97), jnp.float64),
            jnp.asarray(0, jnp.float64),
        ),
    )
    result = svd.svd(svd.PreparedSVDSolve(problem, prepared.plan, altered))
    assert np.allclose(result.diagnostics.relative_residuals, 0, atol=1e-14)
    assert not bool(result.leading_evidence.certified)
    assert int(result.primal_status) == int(svd.SVDSolveStatus.LEADING_UNCERTIFIED)


@pytest.mark.parametrize("left_side", [False, True])
@pytest.mark.parametrize("covariance", [False, True])
def test_fixed_sketch_nonzero_power_projector_jvp_and_vjp(
    left_side: bool, covariance: bool
) -> None:
    matrix = _matrix(7, 5, np.dtype(np.float64))
    perturbation = jnp.asarray(
        np.random.default_rng(4).normal(size=matrix.shape), dtype=matrix.dtype
    )
    dimension = matrix.shape[0] if left_side else matrix.shape[1]
    vector = jnp.asarray(
        np.random.default_rng(5).normal(size=(dimension,)), dtype=matrix.dtype
    )
    contraction = jnp.asarray(
        np.random.default_rng(6).normal(size=(dimension,)), dtype=matrix.dtype
    )
    policy, key = (
        _policy(2, oversampling=1, power_iterations=1, differentiation="projector"),
        jax.random.key(14),
    )

    def action(value: Array) -> Array:
        result = svd.svd(
            svd.SVDProblem(la.DenseLinearOperator(value)), policy=policy, key=key
        )
        frame = result.left_coordinates if left_side else result.right_coordinates
        response = result.left_response if left_side else result.right_response
        if covariance:
            return svd.covariance_action(frame, result.singular_values, response, vector)
        return svd.projector_action(frame, response, vector)

    _, tangent = jax.jvp(action, (matrix,), (perturbation,))
    epsilon = 1e-5
    finite_difference = (
        action(matrix + epsilon * perturbation) - action(matrix - epsilon * perturbation)
    ) / (2 * epsilon)
    assert np.allclose(tangent, finite_difference, rtol=2e-4, atol=2e-6)
    reverse = jax.grad(lambda value: jnp.vdot(contraction, action(value)).real)(matrix)
    assert np.isclose(
        jnp.vdot(reverse, perturbation).real,
        jnp.vdot(contraction, finite_difference).real,
        rtol=2e-4,
        atol=2e-6,
    )


@pytest.mark.parametrize("randomized", [False, True])
def test_status_preparation_failure_eager_jit_fixed_shapes(randomized: bool) -> None:
    method = svd.RandomizedSVD(oversampling=0) if randomized else svd.DenseSVD()
    policy = svd.SVDSolvePolicy(method, count=2, differentiation="projector")
    key = jax.random.key(1) if randomized else None

    def solve(matrix: Array) -> svd.SVDSolveResult:
        return svd.svd(
            svd.SVDProblem(la.DenseLinearOperator(matrix)), policy=policy, key=key
        )

    invalid = jnp.full((4, 3), jnp.nan, dtype=jnp.float64)
    eager, compiled = solve(invalid), jax.jit(solve)(invalid)
    for result in [eager, compiled]:
        assert int(result.primal_status) == int(svd.SVDSolveStatus.PREPARATION_FAILED)
        assert not bool(result.successful)
        assert np.all(np.isnan(result.singular_values))
        assert result.left_coordinates.shape == (4, 2)
        assert result.right_coordinates.shape == (3, 2)
        assert not bool(result.rank_evidence.available)
        assert not bool(result.range_evidence.available)
        assert np.all(result.right_response.frame_correction == 0)
    with pytest.raises(eqx.EquinoxRuntimeError, match="preparation failed"):
        svd.svd(
            svd.SVDProblem(la.DenseLinearOperator(invalid)),
            policy=svd.SVDSolvePolicy(method, count=2, failure=la.FailurePolicy("error")),
            key=key,
        )


@pytest.mark.parametrize("composite", [False, True])
def test_supported_diagonal_and_composite_pairings(composite: bool) -> None:
    matrix = _matrix(6, 4, np.dtype(np.float64))
    source_weights = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=matrix.dtype)
    target_weights = jnp.asarray([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=matrix.dtype)
    if composite:
        source = la.BlockSpace(
            (
                la.ArraySpace(
                    (2,),
                    dtype=matrix.dtype,
                    pairing=la.DiagonalPairing(source_weights[:2]),
                ),
                la.ArraySpace(
                    (2,),
                    dtype=matrix.dtype,
                    pairing=la.DiagonalPairing(source_weights[2:]),
                ),
            ),
            names=("source-first", "source-second"),
        )
        target = la.BlockSpace(
            (
                la.ArraySpace(
                    (3,),
                    dtype=matrix.dtype,
                    pairing=la.DiagonalPairing(target_weights[:3]),
                ),
                la.ArraySpace(
                    (3,),
                    dtype=matrix.dtype,
                    pairing=la.DiagonalPairing(target_weights[3:]),
                ),
            ),
            names=("target-first", "target-second"),
        )
    else:
        source = la.ArraySpace(
            (4,), dtype=matrix.dtype, pairing=la.DiagonalPairing(source_weights)
        )
        target = la.ArraySpace(
            (6,), dtype=matrix.dtype, pairing=la.DiagonalPairing(target_weights)
        )
    operator = la.DenseLinearOperator(matrix, source=source, target=target)
    result = svd.svd(
        svd.SVDProblem(operator), policy=_policy(2, oversampling=2), key=jax.random.key(8)
    )
    transformed = (
        jnp.sqrt(target_weights)[:, None] * matrix / jnp.sqrt(source_weights)[None, :]
    )
    assert bool(result.successful)
    assert np.allclose(
        result.singular_values,
        np.linalg.svd(transformed, compute_uv=False)[:2],
        atol=1e-11,
    )
    assert np.allclose(
        result.right_coordinates.conj().T
        @ (source_weights[:, None] * result.right_coordinates),
        np.eye(2),
        atol=1e-12,
    )
    assert np.allclose(
        result.left_coordinates.conj().T
        @ (target_weights[:, None] * result.left_coordinates),
        np.eye(2),
        atol=1e-12,
    )
    assert np.allclose(
        matrix @ result.right_coordinates,
        result.left_coordinates * result.singular_values,
        atol=1e-12,
    )


def test_randomized_singular_values_differentiate_compressed_algorithm() -> None:
    matrix = _matrix(7, 5, np.dtype(np.float64))
    direction = jnp.asarray(
        np.random.default_rng(11).normal(size=matrix.shape), dtype=matrix.dtype
    )
    policy = _policy(
        2, oversampling=1, power_iterations=1, differentiation="singular-values"
    )
    key = jax.random.key(17)
    probe = jnp.asarray([0.4, -0.7], dtype=matrix.dtype)

    def action(value: Array) -> Array:
        result = svd.svd(
            svd.SVDProblem(la.DenseLinearOperator(value)), policy=policy, key=key
        )
        return jnp.vdot(probe, result.singular_values).real

    _, tangent = jax.jvp(action, (matrix,), (direction,))
    epsilon = 1e-5
    finite_difference = (
        action(matrix + epsilon * direction) - action(matrix - epsilon * direction)
    ) / (2 * epsilon)
    assert np.isclose(tangent, finite_difference, rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize("amplitude", [1e-30, 1.0, 1e30], ids=["tiny", "unit", "large"])
@pytest.mark.parametrize("dtype", [np.float32, np.complex64], ids=["real", "complex"])
def test_shell_audit_and_rank_bounds_are_scale_safe(
    amplitude: float, dtype: type[np.float32] | type[np.complex64]
) -> None:
    reference = np.diag(np.asarray([9.0, 4.0, 1.0], dtype=np.float64))
    matrix = jnp.asarray(amplitude * reference, dtype=dtype)
    problem = svd.SVDProblem(
        _Shell(matrix, operator_id=f"scaled-audit:{np.dtype(dtype).name}")
    )
    prepared = svd.prepare_svd(
        problem, _policy(1, oversampling=0, power_iterations=0), key=jax.random.key(29)
    )
    assert isinstance(prepared.state, svd.RandomizedSVDState)
    basis = np.asarray(
        prepared.state.range_basis,
        dtype=np.complex128 if dtype is np.complex64 else np.float64,
    )
    omitted = reference - basis @ (basis.conj().T @ reference)
    independent_norm = np.linalg.norm(omitted, ord=2)
    result = svd.svd(prepared)
    assert (
        np.asarray(result.range_evidence.spectral_upper_bound) / amplitude
        >= independent_norm
    )
    assert (
        int(result.rank_evidence.lower_bound)
        <= 3
        <= int(result.rank_evidence.upper_bound)
    )
    assert not bool(result.rank_evidence.deterministic_exact)
