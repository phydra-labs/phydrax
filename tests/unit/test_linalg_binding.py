#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


la = phx.linalg


def _spd_properties():
    return la.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_definite": "construction",
            "positive_semidefinite": "construction",
        },
    )


def _problem(matrix):
    return la.LinearSystem(
        la.DenseLinearOperator(
            matrix,
            properties=_spd_properties(),
            operator_id="binding-test-operator",
        ),
        problem_id="binding-test-problem",
    )


def test_template_binds_changing_coefficients_inside_compiled_scan():
    matrices = jnp.asarray(
        [
            [[4.0, 1.0], [1.0, 3.0]],
            [[5.0, 0.5], [0.5, 2.0]],
            [[3.0, -0.25], [-0.25, 2.5]],
        ]
    )
    right_hand_sides = jnp.asarray([[1.0, 2.0], [2.0, -1.0], [-3.0, 0.5]])
    template = la.prepare_template(
        _problem(matrices[0]),
        la.LinearSolvePolicy(
            la.DenseCholesky(),
            require_device_binding=True,
        ),
    )

    def run(matrix_batch, rhs_batch):
        def step(_, payload):
            version, matrix, rhs = payload
            prepared = la.bind_numeric(
                template,
                _problem(matrix),
                numeric_version=version,
            )
            result = la.solve(prepared, rhs)
            return None, (result.value, prepared.numeric_version)

        _, outputs = jax.lax.scan(
            step,
            None,
            (jnp.arange(matrix_batch.shape[0]), matrix_batch, rhs_batch),
        )
        return outputs

    values, versions = jax.jit(run)(matrices, right_hand_sides)
    expected = jax.vmap(jnp.linalg.solve)(matrices, right_hand_sides)
    assert template.device_bindable
    assert jnp.allclose(values, expected)
    assert jnp.array_equal(versions, jnp.arange(3, dtype=jnp.int32))


def test_template_rejects_symbolic_structure_changes():
    template = la.prepare_template(_problem(jnp.eye(2)))
    changed = la.LinearSystem(
        la.DenseLinearOperator(jnp.eye(3), operator_id="binding-test-operator"),
        problem_id="binding-test-problem",
    )
    with pytest.raises(ValueError, match="symbolic problem structure"):
        la.bind_numeric(template, changed)


def test_refresh_preserves_template_identity_and_increments_dynamic_version():
    first = _problem(jnp.asarray([[4.0, 1.0], [1.0, 3.0]]))
    second = _problem(jnp.asarray([[6.0, -1.0], [-1.0, 4.0]]))
    prepared = la.prepare(first, la.LinearSolvePolicy(la.DenseCholesky()))
    refreshed = la.refresh(prepared, second)

    assert refreshed.template.template_id == prepared.template.template_id
    assert int(refreshed.numeric_version) == 1
    assert jnp.allclose(
        la.solve(refreshed, jnp.asarray([1.0, 2.0])).value,
        jnp.linalg.solve(second.operator.matrix, jnp.asarray([1.0, 2.0])),
    )


def test_solve_supports_transformation_generated_empty_operator_batches():
    matrices = jnp.empty((0, 2, 2))
    right_hand_sides = jnp.empty((0, 2))
    result = la.solve(_problem(matrices), right_hand_sides)

    assert result.value.shape == (0, 2)
    assert result.status.shape == (0,)
