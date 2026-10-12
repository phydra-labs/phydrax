#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_oriented_general_plane_stress_response_and_tangent() -> None:
    rotation = phx.applications.solid_mechanics.MPMMaterialOrientation(
        jnp.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    )
    base = phx.applications.solid_mechanics.OrientedMPMConstitutivePlan(
        phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(3),
        rotation,
    )
    material = phx.applications.solid_mechanics.GeneralPlaneStressMPMConstitutivePlan(
        base
    )
    parameters = phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(
        3.0, 11.0
    )
    state = material.initialize_state((1,), jnp.float64)
    deformation = jnp.asarray([[[1.12, 0.06], [0.02, 0.93]]])
    response = material.evaluate(
        deformation, state, jnp.asarray((2.0,)), parameters, 0.0, 0.01
    )
    linearized = material.evaluate_linearized(
        deformation, state, jnp.asarray((2.0,)), parameters, 0.0, 0.01
    )

    assert bool(response.successful[0])
    assert jnp.linalg.norm(response.diagnostics["plane_stress_residual"][0]) < 1e-9
    assert response.diagnostics["transverse_director"][0, 2] > 0.0
    assert bool(linearized.tangent_successful[0])


def test_nonlocal_softening_regularizes_history() -> None:
    softening = phx.applications.solid_mechanics.NonlocalSofteningPlan(viscosity=0.01)
    regularized = softening.regularize(
        jnp.asarray((1.0, 2.0)), jnp.asarray((1.5, 1.5)), 0.01
    )
    assert jnp.all(regularized >= 1.0)
    assert jnp.all(regularized <= 2.0)


def test_biot_thermal_operator_has_coupled_jvp_vjp_and_boundaries() -> None:
    shape = (8, 8)
    boundary = phx.applications.solid_mechanics.MPMCoupledBoundaryPlan(
        pressure_mask=jnp.zeros(shape, dtype="bool").at[0, :].set(True),
        pressure_values=0.0,
        pressure_flux=0.0,
        temperature_mask=jnp.zeros(shape, dtype="bool").at[-1, :].set(True),
        temperature_values=300.0,
        heat_flux=0.0,
    )
    operator = phx.applications.solid_mechanics.PreparedMPMCoupledFieldOperator(
        shape,
        (0.1, 0.1),
        (False, False),
        phx.applications.solid_mechanics.BiotPoromechanicsParameters(
            0.8, 0.1, 1e-4, 1e-3
        ),
        phx.applications.solid_mechanics.ThermalMPMParameters(
            1.0, 2.0, 1e-5, 0.9, 293.15
        ),
        boundary,
    )
    state = phx.applications.solid_mechanics.MPMCoupledFieldState(
        jnp.ones(shape),
        jnp.ones(shape),
        293.15 * jnp.ones(shape),
        jnp.zeros(shape),
        jnp.asarray(0.0),
    )
    direction = phx.applications.solid_mechanics.MPMCoupledFieldState(
        0.1 * jnp.ones(shape),
        jnp.zeros(shape),
        0.2 * jnp.ones(shape),
        jnp.zeros(shape),
        jnp.asarray(0.0),
    )
    linearized = operator.linearize(
        state,
        direction,
        jnp.zeros(shape),
        jnp.zeros(shape),
        jnp.zeros(shape),
        jnp.ones(shape),
        (jnp.ones(shape), jnp.ones(shape)),
    )
    assert bool(linearized.successful)
    assert jnp.all(jnp.isfinite(linearized.jvp.pressure))
    assert jnp.all(jnp.isfinite(linearized.transpose[0]))
    np.testing.assert_allclose(linearized.residual.pressure[0], 1.0)
