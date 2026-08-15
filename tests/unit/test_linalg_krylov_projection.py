#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


la = phx.linalg


def _self_adjoint_properties(*, positive_definite=False):
    evidence = {"self_adjoint": "construction"}
    if positive_definite:
        evidence["positive_definite"] = "construction"
    return la.OperatorProperties(
        self_adjoint=True,
        positive_definite=positive_definite,
        evidence=evidence,
    )


def test_krylov_projection_plan_costs_and_resource_rejection():
    operator = la.DiagonalLinearOperator(
        jnp.arange(1.0, 6.0),
        operator_id="projection-cost-operator",
    )
    policy = la.KrylovProjectionPolicy(max_dimension=4)
    plan = la.plan_krylov_projection(operator, policy)

    assert plan.selected_method == "lanczos"
    assert plan.dimension == 4
    assert plan.cost.matvec_count == 4
    assert plan.cost.storage_bytes > 0
    assert plan.cost.workspace_bytes >= plan.cost.operator_action_workspace_bytes
    assert plan.cost.exact

    constrained = la.KrylovProjectionPolicy(
        max_dimension=4,
        resources=la.KrylovProjectionResourcePolicy(
            max_storage_bytes=plan.cost.storage_bytes - 1,
        ),
    )
    with pytest.raises(ValueError, match="storage estimate"):
        la.plan_krylov_projection(operator, constrained)


def test_prepared_krylov_projection_preserves_relation_and_projects_vectors():
    matrix = jnp.asarray(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 1.0],
            [0.0, 0.0, 1.0, 2.0],
        ]
    )
    operator = la.DenseLinearOperator(
        matrix,
        properties=_self_adjoint_properties(positive_definite=True),
        operator_id="bound-krylov-operator",
    )
    initial = jnp.asarray([1.0, -2.0, 0.5, 3.0])
    plan = la.plan_krylov_projection(
        operator,
        la.KrylovProjectionPolicy("lanczos", max_dimension=4),
    )
    prepared = la.prepare_krylov_projection(operator, initial, plan)
    basis = prepared.decomposition.basis.T
    projected = prepared.decomposition.projected

    assert prepared.plan is plan
    assert prepared.method == "lanczos"
    assert prepared.capacity == 4
    assert prepared.effective_dimension == 4
    assert jnp.allclose(matrix @ basis[:, :-1], basis @ projected, atol=1e-11)
    assert jnp.allclose(jnp.conj(prepared.basis.T) @ prepared.basis, jnp.eye(4))

    vector = jnp.asarray([2.0, -1.0, 4.0, 0.5])
    coefficients = prepared.coefficients(vector)
    assert jnp.allclose(prepared.lift(coefficients), prepared.project(vector))
    assert jnp.allclose(
        prepared.project(vector) + prepared.residual(vector),
        vector,
        atol=1e-12,
    )


def test_bound_projection_is_reused_without_operator_actions_and_under_jit():
    actions = []
    matrix = jnp.asarray(
        [
            [3.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    space = la.ArraySpace((3,), dtype=jnp.float64)

    def action(vector):
        actions.append(None)
        return matrix @ vector

    operator = la.FunctionLinearOperator(
        action,
        source=space,
        target=space,
        operator_id="counted-krylov-operator",
    )
    initial = jnp.asarray([1.0, -0.5, 2.0])
    prepared = la.prepare_krylov_projection(
        operator,
        initial,
        la.KrylovProjectionPolicy("arnoldi", max_dimension=3),
    )
    actions.clear()

    eager = la.matrix_exponential_action(
        operator,
        prepared.initial,
        decomposition=prepared,
    )
    compiled = eqx.filter_jit(
        lambda projection: la.matrix_exponential_action(
            projection.operator,
            projection.initial,
            decomposition=projection,
        )
    )(prepared)

    expected = jax.scipy.linalg.expm(matrix) @ initial
    assert actions == []
    assert eager.matvec_count == 0
    assert compiled.matvec_count == 0
    assert "reused bound arnoldi" in eager.provenance
    assert jnp.allclose(eager.value, expected, rtol=1e-11, atol=1e-11)
    assert jnp.allclose(compiled.value, expected, rtol=1e-11, atol=1e-11)


def test_krylov_projection_reuse_rejects_wrong_operator_start_and_unbound_state():
    matrix = jnp.asarray([[2.0, 1.0], [0.0, 3.0]])
    operator = la.DenseLinearOperator(matrix, operator_id="projection-binding")
    initial = jnp.asarray([1.0, -1.0])
    prepared = la.prepare_krylov_projection(
        operator,
        initial,
        la.KrylovProjectionPolicy("arnoldi", max_dimension=2),
    )
    changed = la.DenseLinearOperator(
        matrix.at[0, 0].set(4.0),
        operator_id="projection-binding",
    )

    with pytest.raises(ValueError, match="operator state does not match"):
        la.matrix_exponential_action(
            changed,
            initial,
            decomposition=prepared,
        )
    with pytest.raises(ValueError, match="starting vector does not match"):
        la.matrix_exponential_action(
            operator,
            initial + 1.0,
            decomposition=prepared,
        )
    with pytest.raises(ValueError, match="Unbound Krylov decompositions"):
        la.matrix_exponential_action(
            operator,
            initial,
            decomposition=prepared.decomposition,
        )


def test_krylov_projection_refresh_preserves_plan_identity_and_rebuilds_state():
    first_matrix = jnp.asarray([[3.0, 1.0], [0.0, 2.0]])
    first_operator = la.DenseLinearOperator(
        first_matrix,
        operator_id="refreshable-krylov",
    )
    initial = jnp.asarray([1.0, 2.0])
    prepared = la.prepare_krylov_projection(
        first_operator,
        initial,
        la.KrylovProjectionPolicy("arnoldi", max_dimension=2),
    )
    second_matrix = jnp.asarray([[4.0, -1.0], [0.5, 2.0]])
    second_operator = la.DenseLinearOperator(
        second_matrix,
        operator_id="refreshable-krylov",
    )
    next_initial = jnp.asarray([-1.0, 0.5])
    refreshed = la.refresh_krylov_projection(
        prepared,
        second_operator,
        next_initial,
    )

    assert refreshed.plan is prepared.plan
    assert refreshed.projection_id == prepared.projection_id
    assert refreshed.operator_fingerprint != prepared.operator_fingerprint
    assert refreshed.initial_fingerprint != prepared.initial_fingerprint
    assert refreshed.numeric_version == 1
    assert refreshed.refresh_count == 1
    result = la.matrix_exponential_action(
        second_operator,
        refreshed.initial,
        decomposition=refreshed,
    )
    assert jnp.allclose(
        result.value,
        jax.scipy.linalg.expm(second_matrix) @ next_initial,
        rtol=1e-11,
        atol=1e-11,
    )


def test_complex_arnoldi_projection_is_metric_correct_and_reusable():
    matrix = jnp.asarray([[1.0 + 1.0j, 2.0 - 0.5j], [0.25j, 3.0 - 1.0j]])
    operator = la.DenseLinearOperator(
        matrix,
        operator_id="complex-krylov-projection",
    )
    initial = jnp.asarray([1.0 - 0.5j, 2.0 + 1.0j])
    prepared = la.prepare_krylov_projection(
        operator,
        initial,
        la.KrylovProjectionPolicy("arnoldi", max_dimension=2),
    )

    assert prepared.method == "arnoldi"
    assert jnp.allclose(
        jnp.conj(prepared.basis.T) @ prepared.basis,
        jnp.eye(2),
        atol=1e-12,
    )
    result = la.matrix_exponential_action(
        operator,
        prepared.initial,
        decomposition=prepared,
    )
    assert jnp.allclose(
        result.value,
        jax.scipy.linalg.expm(matrix) @ initial,
        rtol=1e-11,
        atol=1e-11,
    )
