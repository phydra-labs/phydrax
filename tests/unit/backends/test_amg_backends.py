#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
import phydrax.backends.amg as amg


la = phx.linalg


def _problem(matrix: Any, *, operator_id: Any = "amg-system") -> Any:
    size = matrix.shape[0]
    structure = {
        "velocity": jax.ShapeDtypeStruct((size - 1,), matrix.dtype),
        "pressure": jax.ShapeDtypeStruct((1,), matrix.dtype),
    }
    space = la.PyTreeSpace(structure, space_id="amg-pytree-space")
    rows, columns = jnp.nonzero(matrix)
    relation = phx.sparse.EdgeRelation(
        columns,
        rows,
        source_size=size,
        target_size=size,
    )
    operator = phx.sparse.SparseCoordinateOperator(
        relation,
        matrix[rows, columns],
        source=space,
        target=space,
        operator_id=operator_id,
    )
    return la.LinearSystem(operator)


def _rhs() -> Any:
    return {
        "velocity": jnp.asarray([[1.0, -0.5], [2.0, 1.5]]),
        "pressure": jnp.asarray([[0.25, -1.0]]),
    }


def _available(capabilities: Any) -> Any:
    return phx.backends.BackendAvailability(
        capabilities=capabilities,
        available=True,
        requirement="test provider",
        reason="source-shaped fake provider",
    )


def test_pyamgcl_dependency_absence_is_precise(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "phydrax.backends._availability.importlib.util.find_spec",
        lambda name: None,
    )

    cpu = amg.PyAMGCLBackend().availability()

    assert not cpu.available
    assert "not installed" in cpu.reason
    with pytest.raises(phx.backends.BackendUnavailableError, match="pyamgcl"):
        cpu.require("linear-system")


def test_policy_configuration_is_canonical_and_fingerprinted() -> None:
    left = amg.AmgXPolicy(
        {
            "solver": {"tolerance": 1e-9, "solver": "FGMRES"},
            "config_version": 2,
        }
    )
    right = amg.AmgXPolicy(
        {
            "config_version": 2,
            "solver": {"solver": "FGMRES", "tolerance": 1e-9},
        }
    )

    assert left.config == right.config
    assert left.policy_id == right.policy_id
    assert left.hierarchy_cache_capacity == 2
    assert (
        amg.AmgXPolicy(hierarchy_cache_capacity=1).policy_id
        != amg.AmgXPolicy(hierarchy_cache_capacity=2).policy_id
    )


@pytest.mark.parametrize("capacity", [0, -1])
def test_amgx_cache_capacity_must_be_positive(capacity: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        amg.AmgXPolicy(hierarchy_cache_capacity=capacity)


class _FakePyAMGCLSolver:
    def __init__(self, matrix: Any) -> None:
        self.matrix = matrix.toarray()
        self.calls = 0

    def __call__(self, right_hand_side: Any, initial_guess: Any = None) -> Any:
        del initial_guess
        self.calls += 1
        return np.linalg.solve(self.matrix, right_hand_side), {
            "iterations": 4,
            "reason": "converged",
        }


class _FakePyAMGCL:
    def __init__(self) -> None:
        self.arguments = None
        self.solver = None

    def make_solver(self, matrix: Any, *, solver: Any, prm: Any) -> Any:
        self.arguments = (solver, prm)
        self.solver = _FakePyAMGCLSolver(matrix)
        return self.solver


def test_pyamgcl_source_api_preserves_pytree_rhs_and_refreshes(monkeypatch: Any) -> None:
    provider = _FakePyAMGCL()
    monkeypatch.setattr(
        amg.PyAMGCLBackend,
        "availability",
        lambda self: _available(self.capabilities),
    )
    monkeypatch.setattr(amg, "import_backend_module", lambda *args: provider)
    matrix = jnp.asarray([[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 3.0]])
    problem = _problem(matrix)
    policy = amg.PyAMGCLPolicy(
        solver="bicgstab",
        config={"solver": {"maxiter": 40, "tol": 1e-6}},
    )
    plan = amg.plan_pyamgcl(problem, policy)
    prepared = amg.prepare_pyamgcl(problem, plan)
    result = amg.solve_pyamgcl(prepared, _rhs())

    coordinates = jax.vmap(problem.operator.source.flatten, in_axes=1, out_axes=1)(
        result.value
    )
    expected_rhs = jax.vmap(problem.operator.target.flatten, in_axes=1, out_axes=1)(
        _rhs()
    )
    assert provider.arguments == (
        "bicgstab",
        {"solver": {"maxiter": 40, "tol": 1e-6}},
    )
    # ty: ignore[unresolved-attribute]
    assert provider.solver.calls == 2
    assert jnp.allclose(matrix @ coordinates, expected_rhs)
    assert jnp.all(result.success)
    assert jnp.all(result.diagnostics.iterations == 4)
    assert result.diagnostics.provider_reasons == ("converged", "converged")
    assert result.provenance.plan_id == plan.plan_id
    assert result.transfers.host_to_device_bytes == 0

    refreshed_matrix = matrix + 0.5 * jnp.eye(3)
    refreshed = amg.refresh_pyamgcl(prepared, _problem(refreshed_matrix))
    refreshed_result = amg.solve_pyamgcl(refreshed, _rhs())
    refreshed_coordinates = jax.vmap(
        refreshed.problem.operator.source.flatten, in_axes=1, out_axes=1
    )(refreshed_result.value)
    assert refreshed.plan.plan_id == prepared.plan.plan_id
    assert refreshed.prepared_id == prepared.prepared_id
    assert int(refreshed.numeric_version) == 1
    assert jnp.allclose(refreshed_matrix @ refreshed_coordinates, expected_rhs)

    changed_pattern = refreshed_matrix.at[0, 1].set(0.0)
    with pytest.raises(ValueError, match="unchanged CSR pattern"):
        amg.refresh_pyamgcl(prepared, _problem(changed_pattern))


@pytest.fixture
def native_amgx() -> None:
    availability = amg.amgx_availability()
    if not availability.available:
        pytest.skip(availability.reason)


def _poisson_problem(
    *, dtype: Any = jnp.float64, operator_id: str = "amgx-native-poisson"
) -> tuple[Any, jax.Array, jax.Array]:
    # Large enough to exercise multigrid rather than exact tiny Krylov breakdown.
    size = 64
    matrix = (
        2.0 * jnp.eye(size, dtype=dtype)
        - jnp.eye(size, k=1, dtype=dtype)
        - jnp.eye(size, k=-1, dtype=dtype)
    )
    rows, columns = np.nonzero(np.asarray(matrix))
    space = la.ArraySpace((size,), dtype=dtype)
    operator = phx.sparse.SparseCoordinateOperator(
        phx.sparse.EdgeRelation(columns, rows, source_size=size, target_size=size),
        matrix[rows, columns],
        source=space,
        target=space,
        operator_id=operator_id,
    )
    rhs = jnp.stack(
        (
            jnp.ones((size,), dtype=dtype),
            jnp.linspace(-1.0, 1.0, size, dtype=dtype),
        ),
        axis=1,
    )
    return la.LinearSystem(operator), matrix, rhs


def _assert_dense_parity(
    result: Any, matrix: jax.Array, rhs: jax.Array, *, rtol: float = 1e-5
) -> None:
    reference = np.linalg.solve(np.asarray(matrix), np.asarray(rhs))
    np.testing.assert_allclose(result.value, reference, rtol=rtol, atol=1e-7)
    assert jnp.all(result.success)
    assert jnp.all(result.diagnostics.native_status == 0)
    assert result.diagnostics.native_status.shape == rhs.shape[1:]
    assert result.diagnostics.iterations.shape == rhs.shape[1:]
    assert result.diagnostics.native_residual_norm.shape == rhs.shape[1:]
    assert jnp.all(jnp.isfinite(result.diagnostics.native_residual_norm))
    assert jnp.all(result.diagnostics.iterations >= 0)


def _amgx_partition(prepared: Any) -> Any:
    from phydrax._cuda_native import cuda_partition_key

    jax.effects_barrier()
    partition = cuda_partition_key(prepared.plan.plan_id)
    matches = tuple(
        entry
        for entry in phx.backends.cuda_native_resources().partitions
        if entry.component == "amgx" and entry.partition == partition
    )
    assert len(matches) == 1
    return matches[0]


def _amgx_live_count(prepared: amg.PreparedAmgX) -> int:
    jax.effects_barrier()
    return sum(
        entry.live
        for entry in phx.backends.cuda_native_resources().partitions
        if entry.component == "amgx" and entry.partition == prepared.partition
    )


@pytest.mark.cuda_native
@pytest.mark.parametrize("multi_rhs", [False, True], ids=("single-rhs", "multi-rhs"))
def test_amgx_native_dense_parity(native_amgx: None, multi_rhs: bool) -> None:
    problem, matrix, rhs = _poisson_problem()
    if not multi_rhs:
        rhs = rhs[:, 0]
    prepared = amg.prepare_amgx(problem, amg.plan_amgx(problem))
    try:
        result = amg.solve_amgx(prepared, rhs)
        _assert_dense_parity(result, matrix, rhs)
        assert result.diagnostics.native_rebuilds is not None
        assert int(result.diagnostics.native_rebuilds) == 0
        assert not result.provenance.promoted_to_float64
        assert int(result.transfers.host_to_device_bytes) == 0
        assert int(result.transfers.device_to_host_bytes) == 0
    finally:
        amg.release_amgx(prepared)


@pytest.mark.cuda_native
def test_amgx_native_jit_matches_eager(native_amgx: None) -> None:
    problem, matrix, rhs = _poisson_problem(operator_id="amgx-jit-parity")
    prepared = amg.prepare_amgx(problem, amg.plan_amgx(problem))
    try:
        eager = amg.solve_amgx(prepared, rhs)
        compiled = eqx.filter_jit(amg.solve_amgx)(prepared, rhs)
        _assert_dense_parity(compiled, matrix, rhs)
        np.testing.assert_allclose(compiled.value, eager.value, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(
            compiled.diagnostics.native_status, eager.diagnostics.native_status
        )
        np.testing.assert_array_equal(compiled.status, eager.status)
    finally:
        amg.release_amgx(prepared)


@pytest.mark.cuda_native
def test_amgx_refresh_is_independent_and_old_coefficients_rebuild(
    native_amgx: None,
) -> None:
    problem, matrix, rhs = _poisson_problem(operator_id="amgx-refresh-independent")
    prepared = amg.prepare_amgx(problem, amg.plan_amgx(problem))
    changed = eqx.tree_at(
        lambda item: item.operator.coefficients,
        problem,
        2.0 * problem.operator.coefficients,
    )
    refreshed = amg.refresh_amgx(prepared, changed)
    fresh = amg.prepare_amgx(changed, amg.plan_amgx(changed))
    try:
        assert refreshed is not prepared
        assert not jnp.array_equal(refreshed.native_id, prepared.native_id)
        assert int(refreshed.native_rebuilds) == 0
        assert int(refreshed.numeric_version) == int(prepared.numeric_version) + 1
        refreshed_result = amg.solve_amgx(refreshed, rhs)
        fresh_result = amg.solve_amgx(fresh, rhs)
        _assert_dense_parity(refreshed_result, 2.0 * matrix, rhs)
        np.testing.assert_allclose(
            refreshed_result.value, fresh_result.value, rtol=1e-12, atol=1e-12
        )
        before = _amgx_partition(prepared).rebuilds
        original_result = amg.solve_amgx(prepared, rhs)
        _assert_dense_parity(original_result, matrix, rhs)
        assert original_result.diagnostics.native_rebuilds is not None
        assert int(original_result.diagnostics.native_rebuilds) == 1
        assert _amgx_partition(prepared).rebuilds == before + 1
    finally:
        amg.release_amgx(fresh)
        amg.release_amgx(refreshed)
        amg.release_amgx(prepared)


@pytest.mark.cuda_native
def test_amgx_nonconvergence_preserves_per_rhs_provider_failure(
    native_amgx: None,
) -> None:
    problem, _, rhs = _poisson_problem(operator_id="amgx-nonconvergence")
    policy = amg.AmgXPolicy(
        {
            "config_version": 2,
            "solver": {
                "solver": "CG",
                "preconditioner": {"solver": "NOSOLVER"},
                "max_iters": 1,
                "tolerance": 1e-14,
                "norm": "L2",
                "monitor_residual": 1,
                "convergence": "RELATIVE_INI_CORE",
            },
        }
    )
    prepared = amg.prepare_amgx(problem, amg.plan_amgx(problem, policy))
    try:
        result = amg.solve_amgx(prepared, rhs)
        assert result.status.shape == (2,)
        assert jnp.all(result.status == int(amg.AMGSolveStatus.PROVIDER_FAILURE))
        assert jnp.all(~result.success)
        assert result.diagnostics.native_status is not None
        assert result.diagnostics.native_status.shape == (2,)
        assert jnp.all(result.diagnostics.native_status != 0)
        assert result.diagnostics.iterations is not None
        assert jnp.all(result.diagnostics.iterations <= 1)
        assert jnp.all(result.diagnostics.relative_residual_norm > 1e-14)
    finally:
        amg.release_amgx(prepared)


@pytest.mark.cuda_native
@pytest.mark.strict_jax
def test_amgx_float32_promotion_is_recorded(native_amgx: None) -> None:
    problem, matrix, rhs = _poisson_problem(
        dtype=jnp.float32, operator_id="amgx-float32-promotion"
    )
    policy = amg.AmgXPolicy(
        {
            "config_version": 2,
            "solver": {
                "solver": "CG",
                "max_iters": 100,
                "tolerance": 1e-4,
                "norm": "L2",
                "monitor_residual": 1,
                "convergence": "RELATIVE_INI_CORE",
            },
        }
    )
    prepared = amg.prepare_amgx(problem, amg.plan_amgx(problem, policy))
    try:
        result = amg.solve_amgx(prepared, rhs)
        _assert_dense_parity(result, matrix, rhs, rtol=1e-2)
        assert prepared.values.dtype == jnp.float64
        assert result.provenance.promoted_to_float64
        assert result.diagnostics.native_residual_norm is not None
        assert result.diagnostics.native_residual_norm.dtype == jnp.float64
    finally:
        amg.release_amgx(prepared)


@pytest.mark.cuda_native
def test_amgx_cuda13_classical_amg_pr386_regression(native_amgx: None) -> None:
    from phydrax._cuda_native import load_cuda_component

    if load_cuda_component("amgx").major != 13:
        pytest.skip("Classical AMG PR #386 regression requires the CUDA 13 companion")
    problem, matrix, rhs = _poisson_problem(operator_id="amgx-classical-cuda13")
    policy = amg.AmgXPolicy(
        {
            "config_version": 2,
            "solver": {
                "solver": "FGMRES",
                "preconditioner": {
                    "solver": "AMG",
                    "algorithm": "CLASSICAL",
                    "selector": "PMIS",
                    "interpolator": "D2",
                    "coarse_solver": "DENSE_LU_SOLVER",
                    "max_iters": 1,
                    "max_levels": 10,
                    "min_coarse_rows": 2,
                },
                "max_iters": 100,
                "tolerance": 1e-8,
                "norm": "L2",
                "monitor_residual": 1,
                "convergence": "RELATIVE_INI_CORE",
            },
        }
    )
    prepared = amg.prepare_amgx(problem, amg.plan_amgx(problem, policy))
    try:
        _assert_dense_parity(amg.solve_amgx(prepared, rhs), matrix, rhs)
    finally:
        amg.release_amgx(prepared)


@pytest.mark.cuda_native
@pytest.mark.parametrize("capacity", [1, 2], ids=("capacity-one", "capacity-two"))
def test_amgx_repeated_traced_refreshes_bound_native_resources(
    native_amgx: None, capacity: int
) -> None:
    problem, matrix, rhs = _poisson_problem(
        operator_id=f"amgx-traced-capacity-{capacity}"
    )
    policy = amg.AmgXPolicy(hierarchy_cache_capacity=capacity)
    prepared = amg.prepare_amgx(problem, amg.plan_amgx(problem, policy))
    refresh = eqx.filter_jit(amg.refresh_amgx)
    current = prepared
    try:
        for step in range(8):
            scale = 1.0 + (step + 1) / 8.0
            changed = eqx.tree_at(
                lambda item: item.operator.coefficients,
                problem,
                scale * problem.operator.coefficients,
            )
            current = refresh(current, changed)
            result = amg.solve_amgx(current, rhs)
            _assert_dense_parity(result, scale * matrix, rhs)
            resources = _amgx_partition(current)
            assert resources.capacity == capacity
            assert 0 < resources.live <= capacity
        old_result = amg.solve_amgx(prepared, rhs)
        _assert_dense_parity(old_result, matrix, rhs)
        assert old_result.diagnostics.native_rebuilds is not None
        assert int(old_result.diagnostics.native_rebuilds) == 1
        assert _amgx_partition(prepared).live <= capacity
    finally:
        amg.release_amgx(current)
        amg.release_amgx(prepared)


@pytest.mark.cuda_native
def test_amgx_arrays_on_nondefault_gpu(native_amgx: None) -> None:
    from phydrax._jax_config import default_device

    devices = tuple(device for device in jax.devices() if device.platform == "gpu")
    if len(devices) < 2:
        pytest.skip("Non-default GPU AmgX qualification requires at least two GPUs")
    device = devices[1]
    with default_device(device):
        problem, matrix, rhs = _poisson_problem(operator_id="amgx-nondefault-gpu")
        prepared = amg.prepare_amgx(problem, amg.plan_amgx(problem))
        try:
            result = eqx.filter_jit(amg.solve_amgx)(prepared, rhs)
            _assert_dense_parity(result, matrix, rhs)
            assert result.value.devices() == {device}
            assert _amgx_partition(prepared).device == device.id
        finally:
            amg.release_amgx(prepared)


@pytest.mark.cuda_native
@pytest.mark.parametrize("traced", [False, True], ids=("eager", "traced"))
def test_amgx_release_is_idempotent_and_preparation_rebuilds(
    native_amgx: None, traced: bool
) -> None:
    problem, matrix, rhs = _poisson_problem(
        operator_id=f"amgx-release-{'traced' if traced else 'eager'}"
    )
    prepared = amg.prepare_amgx(problem, amg.plan_amgx(problem))
    release = eqx.filter_jit(amg.release_amgx) if traced else amg.release_amgx
    try:
        assert _amgx_partition(prepared).live == 1
        release(prepared)
        assert _amgx_live_count(prepared) == 0
        release(prepared)
        assert _amgx_live_count(prepared) == 0
        result = amg.solve_amgx(prepared, rhs)
        _assert_dense_parity(result, matrix, rhs)
        assert result.diagnostics.native_rebuilds is not None
        assert int(result.diagnostics.native_rebuilds) == 1
        assert _amgx_partition(prepared).live == 1
    finally:
        amg.release_amgx(prepared)


@pytest.mark.cuda_native
def test_amgx_same_plan_isolated_across_two_devices(native_amgx: None) -> None:
    from phydrax._jax_config import default_device

    devices = jax.devices("gpu")[:2]
    if len(devices) < 2:
        pytest.skip("Same-plan AmgX qualification requires two CUDA devices")
    prepared: list[amg.PreparedAmgX] = []
    matrices: list[jax.Array] = []
    forcings: list[jax.Array] = []
    policy = amg.AmgXPolicy(hierarchy_cache_capacity=1)
    try:
        for index, device in enumerate(devices):
            with default_device(device):
                problem, matrix, rhs = _poisson_problem(
                    operator_id="amgx-same-plan-two-devices"
                )
                scale = index + 1
                problem = eqx.tree_at(
                    lambda item: item.operator.coefficients,
                    problem,
                    problem.operator.coefficients * scale,
                )
                matrices.append(matrix * scale)
                forcings.append(rhs)
                prepared.append(amg.prepare_amgx(problem, amg.plan_amgx(problem, policy)))
        assert prepared[0].plan.plan_id == prepared[1].plan.plan_id
        for index, device in enumerate(devices):
            with default_device(device):
                result = eqx.filter_jit(amg.solve_amgx)(prepared[index], forcings[index])
                _assert_dense_parity(result, matrices[index], forcings[index])
                assert result.value.devices() == {device}
                assert result.diagnostics.native_rebuilds is not None
                assert int(result.diagnostics.native_rebuilds) == 0
        partitions = [
            entry
            for entry in phx.backends.cuda_native_resources().partitions
            if entry.component == "amgx" and entry.partition == prepared[0].partition
        ]
        assert {(entry.device, entry.live, entry.capacity) for entry in partitions} == {
            (device.id, 1, 1) for device in devices
        }
        amg.release_amgx(prepared[0])
        with default_device(devices[1]):
            result = amg.solve_amgx(prepared[1], forcings[1])
            _assert_dense_parity(result, matrices[1], forcings[1])
            assert result.diagnostics.native_rebuilds is not None
            assert int(result.diagnostics.native_rebuilds) == 0
    finally:
        for item in prepared:
            amg.release_amgx(item)
    assert _amgx_live_count(prepared[0]) == 0
