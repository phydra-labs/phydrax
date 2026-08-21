#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.backends import (
    BackendAvailability,
    BackendUnavailableError,
    slepc as slepc_backend,
)
from phydrax.backends.slepc import (
    plan_slepc_eigensolve,
    prepare_slepc_eigensolve,
    SLEPC_CAPABILITIES,
    SLEPcEigenPolicy,
    SLEPcSTOptions,
)
from phydrax.linalg import (
    AbstractLinearOperator,
    ArraySpace,
    OperatorCapabilities,
    OperatorProperties,
)
from phydrax.linalg.eigen import (
    CayleyTransform,
    GeneralEigenproblem,
    GeneralEigenSelection,
    ShiftInvertTransform,
)


class _NeverMaterializedOperator(AbstractLinearOperator):
    matrix: jax.Array

    def __init__(self, matrix, /, *, operator_id):
        matrix_ = jnp.asarray(matrix)
        space = ArraySpace((matrix_.shape[0],), dtype=matrix_.dtype)
        self.source = space
        self.target = space
        self.matrix = matrix_
        self.properties = OperatorProperties()
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=False,
        )
        self.batch_shape = ()
        self.operator_id = operator_id

    def mv(self, vector):
        return self.matrix @ self.source.validate(vector)

    def transpose_mv(self, vector):
        return self.matrix.T @ self.target.validate(vector)

    def adjoint_mv(self, vector):
        return jnp.conj(self.matrix.T) @ self.target.validate(vector)

    def _materialize(self):
        raise AssertionError("SLEPc shell planning must not materialize operators")


def _problem(*, generalized=False, dtype=jnp.float64):
    matrix = jnp.diag(jnp.asarray([2.0, 6.0, 15.0], dtype=dtype))
    operator = _NeverMaterializedOperator(matrix, operator_id="slepc-shell-A")
    if not generalized:
        return GeneralEigenproblem(operator, problem_id="slepc-shell-standard")
    mass = _NeverMaterializedOperator(
        jnp.diag(jnp.asarray([1.0, 2.0, 3.0], dtype=dtype)),
        operator_id="slepc-shell-B",
    )
    return GeneralEigenproblem(
        operator,
        mass,
        problem_id="slepc-shell-generalized",
    )


def test_slepc_shell_plan_is_dependency_free_and_never_materializes(monkeypatch):
    problem = _problem(generalized=True)

    def unexpected_probe():
        raise AssertionError("symbolic planning must not probe optional dependencies")

    monkeypatch.setattr(slepc_backend, "slepc_availability", unexpected_probe)
    policy = SLEPcEigenPolicy(
        GeneralEigenSelection("largest-real", count=1),
        operator_mode="shell",
    )
    first = plan_slepc_eigensolve(problem, policy)
    second = plan_slepc_eigensolve(problem, policy)

    assert first.plan_id == second.plan_id
    assert first.problem_id == problem.problem_id
    assert first.requested_count == 1
    assert first.policy.operator_mode == "shell"


def test_slepc_prepare_reports_missing_optional_dependencies(monkeypatch):
    problem = _problem()
    unavailable = BackendAvailability(
        capabilities=SLEPC_CAPABILITIES,
        available=False,
        requirement="slepc4py and petsc4py for this test",
        reason="dependency intentionally absent",
    )
    monkeypatch.setattr(slepc_backend, "slepc_availability", lambda: unavailable)

    with pytest.raises(BackendUnavailableError, match="dependency intentionally absent"):
        prepare_slepc_eigensolve(
            problem,
            SLEPcEigenPolicy(GeneralEigenSelection("largest-magnitude", count=1)),
        )


def test_slepc_transform_rejects_shell_instead_of_materializing():
    problem = _problem()
    selection = GeneralEigenSelection.closest(6.25, 1)

    with pytest.raises(ValueError, match="explicit CSR"):
        plan_slepc_eigensolve(
            problem,
            SLEPcEigenPolicy(
                selection,
                transform=ShiftInvertTransform(6.25),
                operator_mode="shell",
                st_options=SLEPcSTOptions(
                    "sinvert",
                    ksp_type="preonly",
                    pc_type="lu",
                ),
            ),
        )

    with pytest.raises(ValueError, match="explicit CSR"):
        plan_slepc_eigensolve(
            problem,
            SLEPcEigenPolicy(
                selection,
                transform=CayleyTransform(6.25),
                operator_mode="shell",
                st_options=SLEPcSTOptions(
                    "cayley",
                    ksp_type="gmres",
                    pc_type="ilu",
                ),
            ),
        )


def test_slepc_csr_mode_requires_sparse_operators_and_declared_transform_options():
    problem = _problem(generalized=True)
    closest = GeneralEigenSelection.closest(3.0, 1)

    with pytest.raises(ValueError, match="requires declared"):
        plan_slepc_eigensolve(
            problem,
            SLEPcEigenPolicy(
                closest,
                transform=ShiftInvertTransform(3.0),
                operator_mode="csr",
            ),
        )

    with pytest.raises(ValueError, match="requires declared"):
        plan_slepc_eigensolve(
            problem,
            SLEPcEigenPolicy(
                closest,
                transform=ShiftInvertTransform(3.0),
                operator_mode="csr",
                st_options=SLEPcSTOptions(
                    "cayley",
                    ksp_type="preonly",
                    pc_type="lu",
                ),
            ),
        )

    with pytest.raises(TypeError, match="AbstractSparseLinearOperator"):
        plan_slepc_eigensolve(
            problem,
            SLEPcEigenPolicy(
                closest,
                transform=ShiftInvertTransform(3.0),
                operator_mode="csr",
                st_options=SLEPcSTOptions(
                    "sinvert",
                    ksp_type="preonly",
                    pc_type="lu",
                    factor_solver_type="mumps",
                ),
            ),
        )


def test_slepc_plan_rejects_unsupported_selection_without_provider_imports():
    problem = _problem()

    with pytest.raises(ValueError, match="largest-real"):
        plan_slepc_eigensolve(
            problem,
            SLEPcEigenPolicy(GeneralEigenSelection("smallest-magnitude", count=1)),
        )


def test_slepc_original_generalized_pencil_verification_pairs_and_normalizes():
    problem = _problem(generalized=True)
    values = np.asarray([3.0 + 0.0j, 5.0 + 0.0j])
    right = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.complex128,
    )
    left = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    selection = GeneralEigenSelection("largest-magnitude", count=2)

    ordered_values, ordered_right, ordered_left = slepc_backend._postprocess_eigenpairs(
        problem,
        values,
        right,
        left,
        selection,
        2,
    )
    (
        right_residuals,
        left_residuals,
        right_relative,
        left_relative,
        pairing,
        normalized_right,
        normalized_left,
        action_count,
    ) = slepc_backend._verify_and_normalize_pairs(
        problem,
        ordered_values,
        ordered_right,
        ordered_left,
    )

    assert np.allclose(ordered_values, np.asarray([5.0, 3.0]))
    assert np.allclose(right_residuals, 0.0)
    assert np.allclose(left_residuals, 0.0)
    assert np.allclose(right_relative, 0.0)
    assert np.allclose(left_relative, 0.0)
    assert np.allclose(pairing, np.eye(2))
    assert np.allclose(np.linalg.norm(normalized_right, axis=0), 1.0)
    assert np.all(np.isfinite(normalized_left))
    assert action_count > 0


def test_slepc_complex_original_pencil_residuals_use_adjoint_actions():
    matrix = jnp.asarray(
        [
            [1.0 + 2.0j, 0.0],
            [0.0, 3.0 - 4.0j],
        ],
        dtype=jnp.complex128,
    )
    problem = GeneralEigenproblem(
        _NeverMaterializedOperator(matrix, operator_id="slepc-complex-shell")
    )
    values = np.asarray([3.0 - 4.0j])
    vector = np.asarray([[0.0], [1.0]], dtype=np.complex128)

    right_residuals, left_residuals, _, _, pairing, _, _, _ = (
        slepc_backend._verify_and_normalize_pairs(
            problem,
            values,
            vector,
            vector,
        )
    )

    assert np.allclose(right_residuals, 0.0)
    assert np.allclose(left_residuals, 0.0)
    assert np.allclose(pairing, np.ones((1, 1)))
