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
from phydrax import config as pxconfig


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
    availability = la.sparse_provider_availability("cudss")
    if not availability.available:
        pytest.skip(availability.reason)


@pytest.mark.cuda_native
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_cudss_numeric_refresh_implicit_matrix_derivatives(dtype: DTypeLike) -> None:
    _require_cudss()
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        matrix = jnp.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=dtype)
        rhs = jnp.asarray([1.0, -2.0], dtype=dtype)
        system = _problem(matrix)
        prepared = la.prepare(
            system,
            la.LinearSolvePolicy(la.SparseLDLT(provider="cudss")),
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


@pytest.mark.cuda_native
@pytest.mark.parametrize("batch_shape", [(), (2,), (1, 2)])
def test_cudss_transformed_batch_and_refinement_evidence(
    batch_shape: tuple[int, ...],
) -> None:
    _require_cudss()
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
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


@pytest.mark.cuda_native
def test_cudss_batched_factor_resource_bound() -> None:
    _require_cudss()
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        matrix = jnp.broadcast_to(jnp.eye(2, dtype=jnp.float64), (1, 2, 2, 2))
        system = _problem(matrix)
        plan = la.plan(
            system,
            la.LinearSolvePolicy(
                la.SparseLDLT(), differentiation=la.DifferentiationPolicy("none")
            ),
        )
        expected_bound = 2 * 2 * 2 * jnp.dtype(jnp.float64).itemsize
        assert plan.candidates[-1].factorization_bytes == 8 * expected_bound
        assert plan.candidates[-1].preparation_workspace_bytes == expected_bound
        with pytest.raises(ValueError, match="factorization bytes"):
            la.plan(
                system,
                la.LinearSolvePolicy(
                    la.SparseLDLT(),
                    differentiation=la.DifferentiationPolicy("none"),
                    resources=la.SolveResourcePolicy(
                        factorization_bytes=8 * expected_bound - 1
                    ),
                ),
            )


def test_cuda_sparse_qr_auto_plan_prepares_and_solves() -> None:
    if not la.sparse_provider_availability("jax-cuda").available:
        pytest.skip("JAX CUDA sparse QR is unavailable")
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
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

    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
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
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
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
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
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
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
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


@pytest.mark.parametrize("capacity", [0, -1])
def test_cudss_cache_capacity_requires_positive_value(capacity: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        la.SparseLDLT(factor_cache_capacity=capacity)


@pytest.mark.parametrize("capacity", [True, 1.5, "2"])
def test_cudss_cache_capacity_requires_integer(capacity: object) -> None:
    with pytest.raises(TypeError, match="positive integer"):
        la.SparseLDLT(factor_cache_capacity=capacity)  # ty: ignore[invalid-argument-type]


def test_cudss_availability_agrees_with_sparse_provider() -> None:
    direct = phx.backends.cudss_availability()
    sparse = la.sparse_provider_availability("cudss")
    assert (direct.available, direct.reason) == (sparse.available, sparse.reason)


def _prepare(matrix: jax.Array, capacity: int = 8) -> la.PreparedLinearSolve:
    return la.prepare(
        _problem(matrix),
        la.LinearSolvePolicy(
            la.SparseLDLT(factor_cache_capacity=capacity, refinement_steps=0)
        ),
    )


def _partition(prepared: la.PreparedLinearSolve) -> tuple[int, int, int]:
    from phydrax._cuda_native import cuda_partition_key

    key = cuda_partition_key(prepared.plan.plan_id)
    partitions = [
        entry
        for entry in phx.backends.cuda_native_resources().partitions
        if entry.component == "cudss" and entry.partition == key
    ]
    assert len(partitions) == 1
    entry = partitions[0]
    return entry.live, entry.capacity, entry.rebuilds


@pytest.mark.cuda_native
def test_cudss_traced_refresh_bounds_partition_and_preserves_superseded_factor() -> None:
    _require_cudss()
    matrix = jnp.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=jnp.float32)
    rhs = jnp.asarray([1.0, -2.0], dtype=jnp.float32)
    original = _prepare(matrix, 1)
    current = original

    @eqx.filter_jit
    def refresh(
        current: la.PreparedLinearSolve, values: jax.Array
    ) -> la.PreparedLinearSolve:
        updated = eqx.tree_at(
            lambda item: item.operator.coefficients, original.problem, values.reshape(-1)
        )
        return la.refresh(current, updated)

    try:
        for index in range(12):
            updated_matrix = matrix + jnp.eye(2, dtype=jnp.float32) * (index + 1)
            current = refresh(current, updated_matrix)
            result = eqx.filter_jit(la.solve)(current, rhs)
            np.testing.assert_allclose(
                result.value, jnp.linalg.solve(updated_matrix, rhs), rtol=1e-6
            )
            assert _partition(current)[:2] == (1, 1)
        result = eqx.filter_jit(la.solve)(original, rhs)
        np.testing.assert_allclose(result.value, jnp.linalg.solve(matrix, rhs), rtol=1e-6)
        assert int(result.provenance.native_rebuilds) > 0
        assert _partition(original)[:2] == (1, 1)
    finally:
        la.release(original)
        la.release(current)


@pytest.mark.cuda_native
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_cudss_capacity_one_alternating_preparations_exact_rebuilds(
    dtype: DTypeLike,
) -> None:
    _require_cudss()
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        first_matrix = jnp.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=dtype)
        second_matrix = first_matrix + jnp.eye(2, dtype=dtype)
        rhs = jnp.asarray([1.0, -2.0], dtype=dtype)
        first = _prepare(first_matrix, 1)
        resident = la.solve(first, rhs).value
        second = _prepare(second_matrix, 1)
        try:
            assert first.plan.plan_id == second.plan.plan_id
            for prepared, matrix in (
                (first, first_matrix),
                (second, second_matrix),
                (first, first_matrix),
                (second, second_matrix),
            ):
                result = eqx.filter_jit(la.solve)(prepared, rhs)
                np.testing.assert_allclose(
                    result.value, jnp.linalg.solve(matrix, rhs), rtol=1e-6
                )
                assert int(result.provenance.native_rebuilds) > 0
                assert _partition(prepared)[:2] == (1, 1)
            rebuilt = la.solve(first, rhs)
            np.testing.assert_array_equal(rebuilt.value, resident)
        finally:
            la.release(first)
            la.release(second)


@pytest.mark.cuda_native
def test_cudss_plan_partitions_never_evict_each_other() -> None:
    _require_cudss()
    matrix = jnp.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=jnp.float32)
    rhs = jnp.ones((2,), dtype=jnp.float32)
    small, large = _prepare(matrix, 1), _prepare(matrix, 8)
    original_problem = small.problem

    @eqx.filter_jit
    def refresh(
        current: la.PreparedLinearSolve, values: jax.Array
    ) -> la.PreparedLinearSolve:
        updated = eqx.tree_at(
            lambda item: item.operator.coefficients, original_problem, values.reshape(-1)
        )
        return la.refresh(current, updated)

    try:
        assert small.plan.plan_id != large.plan.plan_id
        for index in range(10):
            small = refresh(small, matrix * (index + 1))
            result = eqx.filter_jit(la.solve)(large, rhs)
            assert int(result.provenance.native_rebuilds) == 0
            np.testing.assert_allclose(
                result.value, jnp.linalg.solve(matrix, rhs), rtol=1e-6
            )
            assert _partition(small)[:2] == (1, 1)
            assert _partition(large)[0] <= 8
    finally:
        la.release(small)
        la.release(large)


@pytest.mark.cuda_native
def test_cudss_release_inside_jit_and_idempotence() -> None:
    _require_cudss()
    prepared = _prepare(jnp.eye(2, dtype=jnp.float32), 1)
    from phydrax._cuda_native import cuda_partition_key

    key = cuda_partition_key(prepared.plan.plan_id)

    def live() -> int:
        return sum(
            entry.live
            for entry in phx.backends.cuda_native_resources().partitions
            if entry.component == "cudss" and entry.partition == key
        )

    assert _partition(prepared)[0] == 1
    assert bool(eqx.filter_jit(la.release)(prepared))
    jax.effects_barrier()
    assert live() == 0
    assert not bool(la.release(prepared))
    assert not bool(eqx.filter_jit(la.release)(prepared))
    jax.effects_barrier()
    assert live() == 0


@pytest.mark.cuda_native
def test_cudss_same_plan_isolated_across_two_devices() -> None:
    from phydrax._cuda_native import cuda_partition_key
    from phydrax._jax_config import default_device

    _require_cudss()
    devices = jax.devices("gpu")[:2]
    if len(devices) < 2:
        pytest.skip("Same-plan cuDSS qualification requires two CUDA devices")
    prepared: list[la.PreparedLinearSolve] = []
    matrices: list[jax.Array] = []
    forcings: list[jax.Array] = []
    try:
        for index, device in enumerate(devices):
            with default_device(device):
                matrix = jnp.asarray(
                    [[4.0 + index, 1.0], [1.0, 3.0 + index]], dtype=jnp.float32
                )
                matrices.append(matrix)
                forcings.append(jnp.asarray([1.0, -2.0], dtype=jnp.float32))
                prepared.append(_prepare(matrix, 1))
        assert prepared[0].plan.plan_id == prepared[1].plan.plan_id
        key = cuda_partition_key(prepared[0].plan.plan_id)
        for index, device in enumerate(devices):
            with default_device(device):
                result = eqx.filter_jit(la.solve)(prepared[index], forcings[index])
                np.testing.assert_allclose(
                    result.value,
                    np.linalg.solve(
                        np.asarray(matrices[index]), np.asarray(forcings[index])
                    ),
                    rtol=1e-6,
                )
                assert result.value.devices() == {device}
                assert int(result.provenance.native_rebuilds) == 0
        partitions = [
            entry
            for entry in phx.backends.cuda_native_resources().partitions
            if entry.component == "cudss" and entry.partition == key
        ]
        assert {(entry.device, entry.live, entry.capacity) for entry in partitions} == {
            (device.id, 1, 1) for device in devices
        }
        la.release(prepared[0])
        with default_device(devices[1]):
            result = la.solve(prepared[1], forcings[1])
            np.testing.assert_allclose(
                result.value,
                np.linalg.solve(np.asarray(matrices[1]), np.asarray(forcings[1])),
                rtol=1e-6,
            )
            assert int(result.provenance.native_rebuilds) == 0
    finally:
        for item in prepared:
            la.release(item)
    assert not any(
        entry.component == "cudss" and entry.partition == key
        for entry in phx.backends.cuda_native_resources().partitions
    )


@pytest.mark.cuda_native
@pytest.mark.parametrize("singular", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_cudss_partial_inertia_never_certifies_zero(
    singular: bool, dtype: DTypeLike
) -> None:
    _require_cudss()
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        matrix = jnp.diag(jnp.asarray([2.0, 0.0 if singular else -3.0], dtype=dtype))
        prepared = _prepare(matrix)
        try:
            evidence = la.factorization_inertia(
                prepared, la.InertiaPolicy(source="provider")
            )
            assert int(evidence.positive) >= 1
            if not singular:
                assert int(evidence.negative) == 1
            assert int(evidence.zero) == -1
            assert not bool(evidence.zero_count_reliable)
            assert not bool(evidence.certified)
        finally:
            la.release(prepared)


@pytest.mark.cuda_native
def test_cudss_public_batch_values_and_nested_vmap() -> None:
    _require_cudss()
    matrix = jnp.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=jnp.float32)
    matrices = jnp.stack((matrix, matrix * 2, matrix * 3, matrix * 4)).reshape(2, 2, 2, 2)
    rhs = jnp.asarray([1.0, -2.0], dtype=jnp.float32)

    system = _problem(matrix)
    policy = la.LinearSolvePolicy(la.SparseLDLT(factor_cache_capacity=1))

    def solve_one(values: jax.Array) -> jax.Array:
        updated = eqx.tree_at(
            lambda item: item.operator.coefficients, system, values.reshape(-1)
        )
        return la.solve(la.prepare(updated, policy), rhs).value

    result = eqx.filter_jit(jax.vmap(jax.vmap(solve_one)))(matrices)
    np.testing.assert_allclose(
        result,
        jax.vmap(jax.vmap(lambda value: jnp.linalg.solve(value, rhs)))(matrices),
        rtol=1e-6,
    )


@pytest.mark.cuda_native
def test_cudss_token_batching_rules_and_trace_time_refusal() -> None:
    _require_cudss()
    from phydrax._cuda_native import cuda_partition_key
    from phydrax.linalg.backends import _cudss_ffi as ffi

    values = jnp.asarray([4.0, 1.0, 1.0, 3.0], dtype=jnp.float32)
    offsets = jnp.asarray([0, 2, 4], dtype=jnp.int32)
    columns = jnp.asarray([0, 1, 0, 1], dtype=jnp.int32)
    partition = cuda_partition_key("cudss-token-batching")

    def analyze(coefficients: jax.Array) -> ffi.CudssToken:
        return ffi.analyze(
            coefficients,
            offsets,
            columns,
            partition=partition,
            capacity=2,
            reordering="default",
            memory="device",
        )

    token = ffi.factorize(analyze(values), values)
    try:
        with pytest.raises(ValueError, match="unbatched token.*batched values"):
            jax.make_jaxpr(
                jax.vmap(lambda coefficients: ffi.factorize(token, coefficients))
            )(jnp.stack((values, values * 2)))
        tampered = eqx.tree_at(lambda item: item.values, token, values * 2)
        with pytest.raises(ValueError, match="unbatched token.*batched values"):
            jax.make_jaxpr(
                jax.vmap(
                    lambda coefficients: ffi.solve(
                        eqx.tree_at(lambda item: item.values, tampered, coefficients),
                        jnp.ones((2,), dtype=jnp.float32),
                        refinement_steps=0,
                    )[0]
                )
            )(jnp.stack((values, values * 2)))
        right_hand_sides = jnp.asarray([[1.0, -2.0], [2.0, 1.0]], dtype=jnp.float32)
        solve_rhs = jax.jit(
            jax.vmap(lambda rhs: ffi.solve(token, rhs, refinement_steps=1)[0])
        )
        np.testing.assert_allclose(
            solve_rhs(right_hand_sides),
            jnp.linalg.solve(values.reshape(2, 2), right_hand_sides.T).T,
            rtol=1e-6,
        )
        tokens = jax.jit(
            jax.vmap(
                lambda coefficients: ffi.factorize(analyze(coefficients), coefficients)
            )
        )(jnp.stack((values, values * 2)))
        try:
            assert bool(jnp.all(tokens.id == tokens.id[0]))
            diagonal, _ = jax.jit(jax.vmap(ffi.diagonal))(tokens)
            assert diagonal.shape == (2, 2)
            assert bool(jnp.all(diagonal > 0))
            released = jax.jit(jax.vmap(ffi.release))(tokens)
            assert bool(jnp.all(released))
            assert not bool(ffi.release(tokens))
        finally:
            ffi.release(tokens)
    finally:
        ffi.release(token)


@pytest.mark.cuda_native
def test_cudss_non_default_gpu() -> None:
    from phydrax._jax_config import default_device

    _require_cudss()
    devices = jax.devices("gpu")
    if len(devices) < 2:
        pytest.skip("requires two CUDA devices")
    with default_device(devices[1]):
        matrix = jnp.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=jnp.float32)
        rhs = jnp.asarray([1.0, -2.0], dtype=jnp.float32)
        prepared = _prepare(matrix)
        try:
            result = eqx.filter_jit(la.solve)(prepared, rhs)
            assert result.value.device == devices[1]
            np.testing.assert_allclose(
                result.value, jnp.linalg.solve(matrix, rhs), rtol=1e-6
            )
        finally:
            la.release(prepared)
