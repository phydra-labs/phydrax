#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
import phydrax.backends.amg as amg


la = phx.linalg


def _problem(matrix, *, operator_id="amg-system"):
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


def _rhs():
    return {
        "velocity": jnp.asarray([[1.0, -0.5], [2.0, 1.5]]),
        "pressure": jnp.asarray([[0.25, -1.0]]),
    }


def _available(capabilities):
    return phx.backends.BackendAvailability(
        capabilities=capabilities,
        available=True,
        requirement="test provider",
        reason="source-shaped fake provider",
    )


def test_dependency_absence_is_precise_and_importing_phydrax_remains_lazy(monkeypatch):
    monkeypatch.setattr(
        "phydrax.backends._availability.importlib.util.find_spec",
        lambda name: None,
    )

    cpu = amg.PyAMGCLBackend().availability()
    gpu = amg.AmgXBackend().availability()

    assert not cpu.available
    assert not gpu.available
    assert "not installed" in cpu.reason
    assert "not installed" in gpu.reason
    with pytest.raises(phx.backends.BackendUnavailableError, match="pyamgcl"):
        cpu.require("linear-system")


def test_policy_configuration_is_canonical_and_fingerprinted():
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


class _FakePyAMGCLSolver:
    def __init__(self, matrix):
        self.matrix = matrix.toarray()
        self.calls = 0

    def __call__(self, right_hand_side, initial_guess=None):
        del initial_guess
        self.calls += 1
        return np.linalg.solve(self.matrix, right_hand_side), {
            "iterations": 4,
            "reason": "converged",
        }


class _FakePyAMGCL:
    def __init__(self):
        self.arguments = None
        self.solver = None

    def make_solver(self, matrix, *, solver, prm):
        self.arguments = (solver, prm)
        self.solver = _FakePyAMGCLSolver(matrix)
        return self.solver


def test_pyamgcl_source_api_preserves_pytree_rhs_and_refreshes(monkeypatch):
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


class _FakeHandle:
    def __init__(self, provider):
        self.provider = provider
        self.destroyed = 0

    def destroy(self):
        self.destroyed += 1


class _FakeConfig(_FakeHandle):
    def create_from_dict(self, config):
        self.provider.config = config
        return self


class _FakeResources(_FakeHandle):
    def create_simple(self, config):
        self.config = config
        return self


class _FakeMatrix(_FakeHandle):
    def create(self, resources):
        self.resources = resources
        return self

    def upload_CSR(self, matrix):
        self.array = matrix.toarray()
        self.provider.matrix_uploads += 1


class _FakeVector(_FakeHandle):
    def create(self, resources):
        self.resources = resources
        self.provider.vectors.append(self)
        return self

    def upload(self, value):
        self.array = np.array(value, copy=True)

    def download(self):
        return np.array(self.array, copy=True)


class _FakeSolver(_FakeHandle):
    def create(self, resources, config):
        self.resources = resources
        self.config = config
        return self

    def setup(self, matrix):
        self.matrix = matrix
        self.provider.setup_calls += 1

    def solve(self, right_hand_side, solution):
        solution.array = np.linalg.solve(self.matrix.array, right_hand_side.array)
        self.provider.solve_calls += 1

    def get_iterations_number(self):
        return 5

    def get_status(self):
        return "converged"


class _FakeAmgX:
    def __init__(self):
        self.initialized = 0
        self.finalized = 0
        self.matrix_uploads = 0
        self.setup_calls = 0
        self.solve_calls = 0
        self.vectors = []
        self.config = None
        self.Config = lambda: _FakeConfig(self)
        self.Resources = lambda: _FakeResources(self)
        self.Matrix = lambda: _FakeMatrix(self)
        self.Vector = lambda: _FakeVector(self)
        self.Solver = lambda: _FakeSolver(self)

    def initialize(self):
        self.initialized += 1

    def finalize(self):
        self.finalized += 1


def test_amgx_official_lifecycle_reuses_hierarchy_for_multiple_rhs(monkeypatch):
    provider = _FakeAmgX()
    monkeypatch.setattr(
        amg.AmgXBackend,
        "availability",
        lambda self: _available(self.capabilities),
    )
    monkeypatch.setattr(amg, "import_backend_module", lambda *args: provider)
    matrix = jnp.asarray([[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 3.0]])
    problem = _problem(matrix)
    plan = amg.plan_amgx(problem)
    prepared = amg.prepare_amgx(problem, plan)
    result = amg.solve_amgx(prepared, _rhs())

    coordinates = jax.vmap(problem.operator.source.flatten, in_axes=1, out_axes=1)(
        result.value
    )
    expected_rhs = jax.vmap(problem.operator.target.flatten, in_axes=1, out_axes=1)(
        _rhs()
    )
    assert provider.initialized == 1
    assert provider.matrix_uploads == 1
    assert provider.setup_calls == 1
    assert provider.solve_calls == 2
    assert jnp.allclose(matrix @ coordinates, expected_rhs)
    assert jnp.all(result.diagnostics.iterations == 5)
    assert result.diagnostics.provider_reasons == ("converged", "converged")
    assert int(result.transfers.host_to_device_bytes) == 2 * expected_rhs.nbytes
    assert int(result.transfers.device_to_host_bytes) == expected_rhs.nbytes
    assert int(result.transfers.synchronization_count) == 2

    persistent_handles = (
        prepared.runtime.solver,
        prepared.runtime.matrix,
        prepared.runtime.resources,
        prepared.runtime.config,
    )
    amg.release_amgx(prepared)
    amg.release_amgx(prepared)
    assert prepared.released
    assert provider.finalized == 1
    assert all(handle.destroyed == 1 for handle in persistent_handles)
    with pytest.raises(ValueError, match="released"):
        amg.solve_amgx(prepared, _rhs())
