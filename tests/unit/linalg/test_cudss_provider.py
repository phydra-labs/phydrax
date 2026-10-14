#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import DTypeLike

import phydrax as phx


la = phx.linalg


def _problem(matrix: jax.Array) -> la.LinearSystem:
    size = matrix.shape[-1]
    rows, columns = np.indices((size, size), dtype=np.int32)
    relation = phx.sparse.EdgeRelation(
        columns.reshape(-1), rows.reshape(-1), source_size=size, target_size=size
    )
    operator = phx.sparse.SparseLinearMap(
        relation,
        matrix.reshape(matrix.shape[:-2] + (-1,)),
        operator_id="cudss-regression",
        properties=la.OperatorProperties(
            self_adjoint=True, evidence={"self_adjoint": "asserted"}
        ),
    )
    return la.LinearSystem(operator)


def _require_cudss() -> None:
    if not la.sparse_provider_availability("spineax-cudss").available:
        pytest.skip("Spineax cuDSS is unavailable")


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_cudss_numeric_refresh_implicit_matrix_derivatives(dtype: DTypeLike) -> None:
    _require_cudss()
    with jax.enable_x64():
        matrix = jnp.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=dtype)
        rhs = jnp.asarray([1.0, -2.0], dtype=dtype)
        system = _problem(matrix)
        prepared = la.prepare(
            system,
            la.LinearSolvePolicy(la.SparseLDLT(provider="spineax-cudss")),
        )
        try:

            def solve(values: jax.Array, forcing: jax.Array) -> jax.Array:
                refreshed_system = eqx.tree_at(
                    lambda item: item.operator.coefficients,
                    system,
                    values.reshape(-1),
                )
                return la.solve(la.refresh(prepared, refreshed_system), forcing).value

            def reference(values: jax.Array, forcing: jax.Array) -> jax.Array:
                return jnp.linalg.solve(values, forcing)

            tangent_matrix = jnp.eye(2, dtype=dtype) * 0.1
            tangent_rhs = jnp.ones(2, dtype=dtype) * 0.2

            @eqx.filter_jit
            def jvp(values: jax.Array, forcing: jax.Array) -> jax.Array:
                return jax.jvp(solve, (values, forcing), (tangent_matrix, tangent_rhs))[1]

            expected = jax.jvp(reference, (matrix, rhs), (tangent_matrix, tangent_rhs))[1]
            assert jnp.allclose(jvp(matrix, rhs), expected, rtol=1e-5, atol=1e-6)
            _, pullback = jax.vjp(solve, matrix, rhs)
            _, dense_pullback = jax.vjp(reference, matrix, rhs)
            for actual, expected in zip(
                pullback(jnp.ones(2, dtype=dtype)),
                dense_pullback(jnp.ones(2, dtype=dtype)),
                strict=True,
            ):
                assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-6)
        finally:
            la.release(prepared)


@pytest.mark.parametrize("batch_shape", [(), (2,), (1, 2)])
def test_cudss_transformed_batch_and_refinement_evidence(
    batch_shape: tuple[int, ...],
) -> None:
    _require_cudss()
    with jax.enable_x64():
        matrix = jnp.broadcast_to(
            jnp.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=jnp.float64),
            batch_shape + (2, 2),
        )
        rhs = jnp.broadcast_to(
            jnp.asarray([[1.0, 2.0], [-2.0, 1.0]], dtype=jnp.float64),
            batch_shape + (2, 2),
        )
        prepared = la.prepare(
            _problem(matrix),
            la.LinearSolvePolicy(la.SparseLDLT(refinement_steps=2)),
            rhs_layout=la.RHSLayout((2,)),
        )
        try:
            expected = jnp.linalg.solve(matrix, rhs)
            results = (
                eqx.filter_jit(la.solve)(prepared, rhs),
                eqx.filter_jit(la.solve_transpose)(prepared, rhs),
                eqx.filter_jit(la.solve_adjoint)(prepared, rhs),
            )
            for result in results:
                assert jnp.all(result.successful)
                assert jnp.allclose(result.value, expected, rtol=1e-12, atol=1e-12)
                assert jnp.all(result.diagnostics.refinement_steps == 2)
                assert result.diagnostics.refinement_steps.shape == batch_shape + (2,)
        finally:
            la.release(prepared)


def test_cudss_batched_factor_resource_bound() -> None:
    _require_cudss()
    with jax.enable_x64():
        matrix = jnp.broadcast_to(jnp.eye(2, dtype=jnp.float64), (1, 2, 2, 2))
        system = _problem(matrix)
        plan = la.plan(
            system,
            la.LinearSolvePolicy(
                la.SparseLDLT(), differentiation=la.DifferentiationPolicy("none")
            ),
        )
        expected_bound = 2 * 2 * 2 * jnp.dtype(jnp.float64).itemsize
        assert plan.candidates[-1].factorization_bytes == expected_bound
        assert plan.candidates[-1].preparation_workspace_bytes == expected_bound
        with pytest.raises(ValueError, match="factorization bytes"):
            la.plan(
                system,
                la.LinearSolvePolicy(
                    la.SparseLDLT(),
                    differentiation=la.DifferentiationPolicy("none"),
                    resources=la.SolveResourcePolicy(
                        factorization_bytes=expected_bound - 1
                    ),
                ),
            )


def test_cuda_sparse_qr_auto_plan_prepares_and_solves() -> None:
    if not la.sparse_provider_availability("jax-cuda").available:
        pytest.skip("JAX CUDA sparse QR is unavailable")
    with jax.enable_x64():
        matrix = jnp.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=jnp.float64)
        rhs = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
        system = _problem(matrix)
        policy = la.LinearSolvePolicy(failure=la.FailurePolicy("error"))
        selected_plan = la.plan(system, policy)
        assert selected_plan.backend == "jax-sparse"
        assert selected_plan.method == "sparse-qr"
        result = la.solve(system, rhs, policy=policy)
        assert jnp.all(result.successful)
        assert jnp.allclose(result.value, jnp.linalg.solve(matrix, rhs))
        prepared = la.prepare(system, selected_plan)
        result = eqx.filter_jit(la.solve)(prepared, rhs)
        assert jnp.all(result.successful)
        assert jnp.allclose(result.value, jnp.linalg.solve(matrix, rhs))


def test_cuda_sparse_qr_auto_plan_rejects_int64_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not la.sparse_provider_availability("jax-cuda").available:
        pytest.skip("JAX CUDA sparse QR is unavailable")
    import phydrax.sparse._linear as sparse_linear

    with jax.enable_x64():
        monkeypatch.setattr(
            sparse_linear, "_storage_index_dtype", lambda shape, groups: np.int64
        )
        system = _problem(jnp.eye(2, dtype=jnp.float64))
        operator = system.operator
        assert isinstance(operator, phx.sparse.SparseLinearMap)
        assert operator.sparse_storage().index_width == 64
        with pytest.raises(ValueError, match="32-bit CSR indices"):
            la.prepare(system, la.LinearSolvePolicy(failure=la.FailurePolicy("error")))


def test_cuda_sparse_qr_declares_complex_support() -> None:
    assert la.sparse_provider_capabilities("jax-cuda").complex


@pytest.mark.parametrize("dtype", [jnp.complex64, jnp.complex128])
@pytest.mark.parametrize("automatic", [False, True])
def test_cuda_sparse_qr_complex_nonsymmetric_solve(
    dtype: DTypeLike, automatic: bool
) -> None:
    if not la.sparse_provider_availability("jax-cuda").available:
        pytest.skip("JAX CUDA sparse QR is unavailable")
    with jax.enable_x64():
        matrix = jnp.asarray(
            [[3.0 + 1.0j, 2.0 - 1.0j], [0.5 + 2.0j, 4.0 - 0.5j]],
            dtype=dtype,
        )
        rhs = jnp.asarray(
            [[1.0 + 2.0j, 0.5 - 1.0j], [-2.0 + 0.5j, 1.0 + 1.0j]],
            dtype=dtype,
        )
        rows, columns = np.indices((2, 2), dtype=np.int32)
        relation = phx.sparse.EdgeRelation(
            columns.reshape(-1), rows.reshape(-1), source_size=2, target_size=2
        )
        operator = phx.sparse.SparseLinearMap(
            relation,
            matrix.reshape(-1),
            operator_id="cuda-qr-complex-nonsymmetric",
        )
        policy = (
            la.LinearSolvePolicy(failure=la.FailurePolicy("error"))
            if automatic
            else la.LinearSolvePolicy(la.SparseQR(provider="jax-cuda"))
        )
        prepared = la.prepare(
            la.LinearSystem(operator), policy, rhs_layout=la.RHSLayout((2,))
        )
        assert prepared.plan.backend == "jax-sparse"
        assert prepared.plan.method == "sparse-qr"
        result = eqx.filter_jit(la.solve)(prepared, rhs)
        tolerance = 1e-5 if dtype == jnp.complex64 else 1e-12
        assert jnp.all(result.successful)
        assert result.value.dtype == matrix.dtype
        assert jnp.allclose(
            result.value, jnp.linalg.solve(matrix, rhs), rtol=tolerance, atol=tolerance
        )


@pytest.mark.parametrize("curvature", [1.0, -1.0], ids=["convex", "nonconvex"])
def test_structured_ipm_cudss_refusal_includes_convex_problems(
    curvature: float,
) -> None:
    opt = phx.optim
    with jax.enable_x64():
        problem = opt.MinimizationProblem(
            lambda value, target: curvature * jnp.sum((value - target) ** 2),
            bounds=opt.Bounds(0.0, 1.0),
            constraints=(
                opt.NonlinearConstraint(
                    lambda value, _: jnp.asarray([jnp.sum(value)]),
                    lower=1.0,
                    upper=1.0,
                    constraint_id="sum-one",
                ),
            ),
            problem_id="cudss-ipm-inertia-refusal",
        )
        compilation = opt.compile_structured_minimization(
            problem,
            jnp.asarray([0.5, 0.5], dtype=jnp.float64),
            sample_args=jnp.asarray([0.25, 0.75], dtype=jnp.float64),
        )
        method = opt.PrimalDualInteriorPoint(
            mode="sparse-augmented",
            linear_policy=la.LinearSolvePolicy(la.SparseLDLT()),
        )
        with pytest.raises(
            ValueError, match="structured IPM, including convex problems,.*zero-inertia"
        ):
            opt.solve_structured_minimization(compilation, method=method)


@pytest.mark.parametrize("singular", [False, True])
def test_sparse_auto_status_uses_status_capable_route(singular: bool) -> None:
    with jax.enable_x64():
        matrix = jnp.asarray(
            [[1.0, 2.0], [2.0, 4.0]] if singular else [[4.0, 1.0], [1.0, 3.0]],
            dtype=jnp.float64,
        )
        rhs = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
        system = _problem(matrix)
        selected_plan = la.plan(system)
        assert selected_plan.policy.failure.mode == "status"
        assert selected_plan.method == "minres"
        assert any(
            "singular CUDA QR raises instead of returning failure status" in reason
            for reason in selected_plan.rejected
        )
        prepared = la.prepare(system, selected_plan)
        result = eqx.filter_jit(la.solve)(prepared, rhs)
        assert bool(result.successful) == (not singular)
        if not singular:
            assert jnp.allclose(result.value, jnp.linalg.solve(matrix, rhs))
        if la.sparse_provider_availability("jax-cuda").available:
            explicit = la.plan(
                system, la.LinearSolvePolicy(la.SparseQR(provider="jax-cuda"))
            )
            assert explicit.backend == "jax-sparse"
            assert explicit.method == "sparse-qr"
