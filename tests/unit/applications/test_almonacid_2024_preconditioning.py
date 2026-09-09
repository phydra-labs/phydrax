# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Defend mixed-coordinate permutation, row scaling, and exact condensation."""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from phydrax.applications.skeletal_muscle.continuum import (
    Almonacid2024Control,
    Almonacid2024MuscleAponeurosisPlan,
    almonacid_2024_repository_case,
)
from phydrax.applications.skeletal_muscle.continuum._almonacid_2024_preconditioning import (
    almonacid_2024_linear_policy,
    almonacid_2024_setup_operator,
)
from phydrax.linalg import MaterializationPolicy, TransposeLinearOperator
from phydrax.nonlinear import NewtonKrylov


INPUTS = Path(__file__).resolve().parents[2] / "fixtures/flexodeal_0698e3d"


@eqx.filter_jit
def _actions(model, coordinates, control, direction, cotangent):
    setup = almonacid_2024_setup_operator(model, coordinates, control)
    flat, restore = ravel_pytree(coordinates)

    def residual(value):
        return ravel_pytree(model._root_residual(restore(value), control))[0]

    _, expected = jax.jvp(residual, (flat,), (direction,))
    _, pullback = jax.vjp(residual, flat)
    blocks = setup.block_operator.blocks
    mechanical = direction[: setup.mechanical_schur.source.size]
    # Exact local elimination must annihilate BOTH scalar rows, despite the
    # pressure pivot's identically zero diagonal. It is not scalar Jacobi.
    scalar = -setup.scalar_inverse.apply(blocks[0][1].mv(mechanical))
    scalar_residual, mechanical_residual = setup.block_operator.mv((scalar, mechanical))
    builder = model.plan.method.linear_policy.preconditioning.builder
    forward = builder.prepare(setup, materialization=MaterializationPolicy())
    reverse = builder.prepare(
        TransposeLinearOperator(setup), materialization=MaterializationPolicy()
    )
    forward_pairing = jnp.vdot(
        cotangent, ravel_pytree(forward.apply(restore(direction)))[0]
    )
    reverse_pairing = jnp.vdot(
        ravel_pytree(reverse.apply(restore(cotangent)))[0], direction
    )
    storage = setup.full_storage
    positions = jnp.arange(storage.nnz, dtype=storage.indptr.dtype)
    rows = jnp.searchsorted(storage.indptr[1:], positions, side="right")
    sparse_forward = (
        jnp.zeros_like(direction)
        .at[rows]
        .add(storage.values * direction[storage.indices])
    )
    sparse_transpose = (
        jnp.zeros_like(cotangent)
        .at[storage.indices]
        .add(storage.values * cotangent[rows])
    )
    return (
        ravel_pytree(setup.mv(restore(direction)))[0],
        expected,
        ravel_pytree(setup.transpose_mv(restore(cotangent)))[0],
        pullback(cotangent)[0],
        ravel_pytree(setup.adjoint_mv(restore(cotangent)))[0],
        scalar_residual,
        mechanical_residual,
        setup.mechanical_schur.mv(mechanical),
        forward_pairing,
        reverse_pairing,
        sparse_forward,
        sparse_transpose,
    )


@pytest.mark.parametrize(
    ("dynamic", "mechanical_solver"),
    (
        (False, None),
        (True, "condensed-jax-cpu"),
        (True, "full-mixed-jax-cpu"),
    ),
    ids=("quasistatic-triangular", "dynamic-condensed", "dynamic-full-mixed"),
)
def test_source_scaled_tangent_adjoint_and_local_schur_under_numeric_refresh(
    dynamic, mechanical_solver
):
    with jax.enable_x64(True):
        source_plan, parameters, _, _, _ = almonacid_2024_repository_case(
            INPUTS, refinement=0
        )
        plan = Almonacid2024MuscleAponeurosisPlan(
            source_plan.geometry,
            control_source_id=source_plan.control_source_id,
            dynamic=dynamic,
            pulling_face_id=source_plan.pulling_face_id,
            stress_scale_Pa=source_plan.stress_scale_Pa,
            method=NewtonKrylov(
                linear_policy=almonacid_2024_linear_policy(mechanical_solver)
            ),
        )
        model = plan.prepare(parameters)
        state = model.state
        initial = (
            state.displacement_m[model.geometry.free_dofs]
            / plan.geometry.muscle_length_m,
            state.pressure_coefficients_Pa / plan.stress_scale_Pa,
            state.dilation_coefficients,
        )
        flat, _ = ravel_pytree(initial)
        direction = 0.1 * jnp.sin(jnp.arange(flat.size, dtype=flat.dtype) + 0.4)
        cotangent = 0.1 * jnp.cos(jnp.arange(flat.size, dtype=flat.dtype) + 0.7)
        # Reuse one traced setup with distinct current material and inertia
        # values; a frozen initial/reference tangent must fail this comparison.
        for amplitude, dt in ((1.0, 0.012), (1.7, 0.019)):
            coordinates = (
                initial[0]
                + amplitude
                * 1e-5
                * jnp.sin(jnp.arange(initial[0].size).reshape(initial[0].shape)),
                initial[1]
                + amplitude
                * 0.02
                * jnp.cos(jnp.arange(initial[1].size).reshape(initial[1].shape)),
                initial[2]
                + amplitude
                * 0.003
                * jnp.sin(jnp.arange(initial[2].size).reshape(initial[2].shape) + 0.2),
            )
            control = Almonacid2024Control(
                dt, 0.2, 0.001, source_id=plan.control_source_id
            )
            (
                actual,
                expected,
                transposed,
                vjp,
                adjoint,
                eliminated,
                full_u,
                condensed_u,
                forward_pairing,
                reverse_pairing,
                sparse_forward,
                sparse_transpose,
            ) = _actions(model, coordinates, control, direction, cotangent)
            np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-8)
            np.testing.assert_allclose(transposed, vjp, rtol=2e-10, atol=2e-8)
            np.testing.assert_allclose(adjoint, vjp, rtol=2e-10, atol=2e-8)
            np.testing.assert_allclose(eliminated, 0.0, atol=2e-8)
            np.testing.assert_allclose(full_u, condensed_u, rtol=2e-10, atol=2e-8)
            np.testing.assert_allclose(
                forward_pairing, reverse_pairing, rtol=2e-10, atol=2e-8
            )
            np.testing.assert_allclose(sparse_forward, expected, rtol=2e-10, atol=2e-8)
            np.testing.assert_allclose(sparse_transpose, vjp, rtol=2e-10, atol=2e-8)
