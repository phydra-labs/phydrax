#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
import phydrax.backends.petsc as pet


la = phx.linalg


def _available(capabilities):
    return phx.backends.BackendAvailability(
        capabilities=capabilities,
        available=True,
        requirement="source-shaped fake PETSc",
        reason="available in test",
    )


def _operator(matrix, *, operator_id="petsc-amat", space=None):
    size = matrix.shape[0]
    if space is None:
        space = la.ArraySpace((size,), dtype=matrix.dtype)
    rows, columns = jnp.nonzero(matrix)
    relation = phx.sparse.EdgeRelation(
        columns,
        rows,
        source_size=size,
        target_size=size,
    )
    return phx.sparse.SparseCoordinateOperator(
        relation,
        matrix[rows, columns],
        source=space,
        target=space,
        operator_id=operator_id,
    )


def _problem(matrix, *, operator_id="petsc-amat", problem_id="petsc-system", space=None):
    return la.LinearSystem(
        _operator(matrix, operator_id=operator_id, space=space),
        problem_id=problem_id,
    )


class _FakeVec:
    def __init__(self, array=None):
        self.array = None if array is None else np.asarray(array)

    def createWithArray(self, array, comm=None):
        del comm
        return _FakeVec(np.asarray(array))

    def createSeq(self, size, comm=None):
        del comm
        return _FakeVec(np.zeros(size))

    def duplicate(self):
        return _FakeVec(np.zeros_like(self.array))

    def set(self, value):
        self.array.fill(value)

    def getArray(self, readonly=False):
        del readonly
        return self.array


class _FakeMat:
    def __init__(self, provider):
        self.provider = provider
        self.dense = None
        self.shape = None

    def createAIJ(self, *, size, csr, comm=None):
        del comm
        matrix = _FakeMat(self.provider)
        matrix.shape = tuple(size)
        matrix._set_csr(*csr)
        self.provider.matrices.append(matrix)
        return matrix

    def createDense(self, *, size, comm=None):
        del comm
        matrix = _FakeMat(self.provider)
        matrix.shape = tuple(size)
        matrix.dense = np.zeros(size)
        self.provider.dense_creations += 1
        self.provider.matrices.append(matrix)
        return matrix

    def _set_csr(self, indptr, indices, values):
        self.dense = np.zeros(self.shape, dtype=np.asarray(values).dtype)
        for row in range(self.shape[0]):
            begin, end = int(indptr[row]), int(indptr[row + 1])
            self.dense[row, np.asarray(indices[begin:end], dtype=int)] = values[begin:end]

    def assemble(self):
        pass

    def setUp(self):
        pass

    def zeroEntries(self):
        self.dense.fill(0)

    def setValuesCSR(self, indptr, indices, values):
        self._set_csr(indptr, indices, values)

    def setValues(self, rows, columns, values):
        self.dense[np.ix_(np.asarray(rows), np.asarray(columns))] = values


class _FakePC:
    def __init__(self):
        self.pc_type = None

    def setType(self, value):
        self.pc_type = value


class _FakeKSP:
    def __init__(self, provider):
        self.provider = provider
        self.pc = _FakePC()
        self.reason = 2
        self.iterations = 3
        self.setup_count = 0
        self.preconditioner_setups = 0
        self.reuse = False
        self.amat = None
        self.pmat = None

    def create(self, comm=None):
        del comm
        self.provider.ksp = self
        return self

    def setOperators(self, amat, pmat):
        self.amat, self.pmat = amat, pmat

    def setType(self, value):
        self.ksp_type = value

    def getPC(self):
        return self.pc

    def setReusePreconditioner(self, value):
        self.reuse = bool(value)

    def setTolerances(self, **values):
        self.tolerances = values

    def setInitialGuessNonzero(self, value):
        self.initial_nonzero = value

    def setOptionsPrefix(self, prefix):
        self.prefix = prefix

    def setFromOptions(self):
        pass

    def setUp(self):
        self.setup_count += 1
        if self.setup_count == 1 or not self.reuse:
            self.preconditioner_setups += 1

    def solve(self, rhs, value):
        value.array[:] = np.linalg.solve(self.amat.dense, rhs.array)

    def getIterationNumber(self):
        return self.iterations

    def getConvergedReason(self):
        return self.reason


class _FakeSNES:
    def __init__(self, provider):
        self.provider = provider
        self.ksp = _FakeKSP(provider)
        self.used_matrix_free = False

    def create(self, comm=None):
        del comm
        self.provider.snes = self
        return self

    def setFunction(self, callback, residual):
        self.function = callback
        self.residual = residual

    def setUseMF(self, value):
        self.used_matrix_free = bool(value)

    def setJacobian(self, callback, *, J, P):
        self.jacobian_callback = callback
        self.jacobian = (J, P)

    def setType(self, value):
        self.snes_type = value

    def setTolerances(self, **values):
        self.tolerances = values

    def getKSP(self):
        return self.ksp

    def setOptionsPrefix(self, prefix):
        self.prefix = prefix

    def setFromOptions(self):
        pass

    def setUp(self):
        pass


class _FakeOptions(dict):
    pass


class _FakePETSc:
    ScalarType = np.float64
    IntType = np.int32
    COMM_SELF = object()

    def __init__(self):
        self.matrices = []
        self.dense_creations = 0
        self.ksp = None
        self.snes = None
        self.options = _FakeOptions()
        self.Mat = lambda: _FakeMat(self)
        self.Vec = _FakeVec
        self.KSP = lambda: _FakeKSP(self)
        self.SNES = lambda: _FakeSNES(self)
        self.Options = lambda: self.options


def _install_fake(monkeypatch):
    provider = _FakePETSc()
    monkeypatch.setattr(
        pet.PETScBackend,
        "availability",
        lambda self: _available(self.capabilities),
    )
    monkeypatch.setattr(pet, "import_backend_module", lambda *args: provider)
    return provider


def test_dependency_absence_is_lazy_and_precise(monkeypatch):
    monkeypatch.setattr(
        "phydrax.backends._availability.importlib.util.find_spec",
        lambda name: None,
    )

    availability = pet.petsc_availability()

    assert not availability.available
    assert "not installed" in availability.reason
    with pytest.raises(phx.backends.BackendUnavailableError, match="petsc4py"):
        availability.require("linear-system")


def test_ksp_preserves_pytree_multi_rhs_and_uses_reason_plus_true_residual(monkeypatch):
    provider = _install_fake(monkeypatch)
    matrix = jnp.asarray(
        [[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 3.0]],
        dtype=jnp.float64,
    )
    structure = {
        "velocity": jax.ShapeDtypeStruct((2,), jnp.float64),
        "pressure": jax.ShapeDtypeStruct((1,), jnp.float64),
    }
    space = la.PyTreeSpace(structure, space_id="petsc-pytree-space")
    rhs = {
        "velocity": jnp.asarray([[1.0, -0.5], [2.0, 1.5]]),
        "pressure": jnp.asarray([[0.25, -1.0]]),
    }
    plan = pet.plan_petsc_linear(
        _problem(matrix, space=space),
        pet.PETScKSPPolicy(
            ksp_type="gmres",
            pc_type="jacobi",
            options={"ksp_monitor": None},
        ),
    )
    prepared = pet.prepare_petsc_linear(plan)
    result = pet.solve_petsc_linear(prepared, rhs)
    solution = jax.vmap(space.flatten, in_axes=1, out_axes=1)(result.value)
    targets = jax.vmap(space.flatten, in_axes=1, out_axes=1)(rhs)

    assert jnp.allclose(matrix @ solution, targets)
    assert jnp.all(result.successful)
    assert jnp.all(result.diagnostics.relative_residual < 1e-12)
    assert result.provenance.operator_id == plan.problem.operator.operator_id
    assert (
        result.provenance.preconditioner_operator_id
        == plan.preconditioner_operator.operator_id
    )
    assert int(result.provenance.setup_transfer.device_to_host_bytes) > 0
    assert int(result.provenance.solve_transfer.host_to_device_bytes) > 0
    assert provider.ksp.setup_count == 1

    provider.ksp.reason = -3
    failed = pet.solve_petsc_linear(prepared, rhs)
    assert not jnp.any(failed.successful)
    assert jnp.all(failed.diagnostics.relative_residual < 1e-12)


def test_ksp_rejects_noncanonical_sources_without_dense_materialization(monkeypatch):
    matrix = jnp.eye(3, dtype=jnp.float64)
    dense = la.LinearSystem(la.DenseLinearOperator(matrix))
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("dense materialization must not be called")

    monkeypatch.setattr(la, "materialize", forbidden)
    with pytest.raises(ValueError, match="canonical CSR"):
        pet.plan_petsc_linear(dense)
    assert not called


def test_ksp_refresh_preserves_distinct_amat_pmat_and_explicit_pc_reuse(monkeypatch):
    provider = _install_fake(monkeypatch)
    matrix = jnp.asarray([[4.0, -1.0], [-1.0, 3.0]], dtype=jnp.float64)
    surrogate = jnp.asarray([[4.0, 0.0], [0.0, 3.0]], dtype=jnp.float64)
    problem = _problem(matrix)
    pmat = _operator(surrogate, operator_id="petsc-pmat")
    policy = pet.PETScKSPPolicy(pc_type="jacobi", reuse_preconditioner=False)
    prepared = pet.prepare_petsc_linear(
        pet.plan_petsc_linear(problem, policy, preconditioner_operator=pmat)
    )
    original_amat = prepared.operator_matrix
    original_pmat = prepared.preconditioner_matrix
    refreshed = pet.refresh_petsc_linear(
        prepared,
        _problem(matrix + 0.5 * jnp.eye(2)),
        preconditioner_operator=_operator(
            surrogate + 0.25 * jnp.eye(2),
            operator_id="petsc-pmat",
        ),
    )

    assert refreshed.operator_matrix is original_amat
    assert refreshed.preconditioner_matrix is original_pmat
    assert refreshed.operator_matrix is not refreshed.preconditioner_matrix
    assert refreshed.prepared_id == prepared.prepared_id
    assert refreshed.numeric_version == 1
    assert provider.ksp.preconditioner_setups == 2

    changed_pattern = matrix.at[0, 1].set(0.0)
    with pytest.raises(ValueError, match="unchanged.*patterns"):
        pet.refresh_petsc_linear(
            prepared,
            _problem(changed_pattern),
            preconditioner_operator=pmat,
        )
    with pytest.raises(ValueError, match="Pmat identity"):
        pet.refresh_petsc_linear(
            prepared,
            _problem(matrix),
            preconditioner_operator=_operator(surrogate, operator_id="other-pmat"),
        )


def test_explicit_reuse_policy_preserves_numeric_pc_across_refresh(monkeypatch):
    provider = _install_fake(monkeypatch)
    matrix = jnp.asarray([[3.0, -1.0], [-1.0, 2.0]], dtype=jnp.float64)
    policy = pet.PETScKSPPolicy(pc_type="jacobi", reuse_preconditioner=True)
    prepared = pet.prepare_petsc_linear(pet.plan_petsc_linear(_problem(matrix), policy))
    refreshed = pet.refresh_petsc_linear(
        prepared,
        _problem(matrix + 0.25 * jnp.eye(2)),
    )

    assert refreshed.plan.policy.reuse_preconditioner
    assert provider.ksp.preconditioner_setups == 1


def test_matrix_free_snes_never_builds_or_autodifferentiates_dense_jacobian(monkeypatch):
    provider = _install_fake(monkeypatch)
    problem = phx.nonlinear.NonlinearSystemProblem(
        lambda state, args: {
            "left": state["left"] ** 2 - args,
            "right": state["right"] - 1.0,
        },
        problem_id="petsc-nonlinear",
    )
    initial = {
        "left": jnp.asarray([2.0], dtype=jnp.float64),
        "right": jnp.asarray([0.0], dtype=jnp.float64),
    }
    monkeypatch.setattr(
        pet.jax,
        "jacfwd",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("matrix-free SNES must not request a dense Jacobian")
        ),
    )

    plan = pet.plan_petsc_nonlinear(problem, initial, args=jnp.asarray([4.0]))
    prepared = pet.prepare_petsc_nonlinear(plan)

    assert plan.policy.jacobian_mode == "matrix-free"
    assert provider.snes.used_matrix_free
    assert provider.dense_creations == 0
    assert prepared.jacobian is None
    assert prepared.plan.plan_id == plan.plan_id


def test_dense_autodiff_snes_is_explicit_and_resource_guarded():
    problem = phx.nonlinear.NonlinearSystemProblem(
        lambda state, args: state - args,
        problem_id="petsc-dense-autodiff",
    )
    state = jnp.ones((3,), dtype=jnp.float64)
    policy = pet.PETScSNESPolicy(
        jacobian_mode="dense-autodiff",
        maximum_dense_dimension=2,
    )

    with pytest.raises(ValueError, match="dense-autodiff"):
        pet.plan_petsc_nonlinear(problem, state, policy, args=jnp.zeros_like(state))
