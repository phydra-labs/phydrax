#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _arguments(gradient, model_state=(0.1, 1.0)):
    return phx.equations.TurbulenceArguments(
        jnp.asarray(gradient),
        jnp.asarray(0.02),
        jnp.asarray(0.1),
        jnp.asarray(model_state),
    )


def test_les_and_rans_eddy_viscosities_are_nonnegative_and_finite():
    gradient = jnp.asarray(((0.2, -0.4, 0.1), (0.3, -0.1, 0.2), (-0.2, 0.1, -0.1)))
    arguments = _arguments(gradient)
    state = jnp.asarray((1.0,))
    for plan in (
        phx.equations.WALEPlan(),
        phx.equations.VremanPlan(),
        phx.equations.SpalartAllmarasPlan(),
        phx.equations.KOmegaSSTPlan(),
    ):
        viscosity = plan.kinematic_viscosity(state, arguments)
        assert jnp.isfinite(viscosity)
        assert viscosity >= 0.0
    assert (
        phx.equations.WALEPlan().kinematic_viscosity(state, _arguments(jnp.zeros((3, 3))))
        == 0.0
    )
    assert jnp.isfinite(phx.equations.SpalartAllmarasPlan().source(arguments))
    assert jnp.all(jnp.isfinite(phx.equations.KOmegaSSTPlan().source(arguments)))


def test_turbulent_transport_wall_model_and_synthetic_inflow_are_operational():
    molecular = phx.equations.ConstantTransport(1.0e-5, 0.02)
    closure = phx.equations.TurbulentTransportClosure(
        molecular,
        phx.equations.WALEPlan(),
        specific_heat_cp=1004.5,
    )
    state = jnp.asarray((1.2, 0.0, 0.0, 0.0, 3.0e5))
    arguments = _arguments(
        jnp.asarray(((0.1, 0.3, 0.0), (0.0, -0.1, 0.2), (0.0, 0.0, 0.0)))
    )
    properties = closure.properties(jnp.asarray(300.0), state, arguments)
    assert properties.dynamic_viscosity >= 1.0e-5
    assert properties.thermal_conductivity >= 0.02

    wall = phx.equations.EquilibriumWallModel().evaluate(10.0, 0.01, 1.2, 1.5e-5)
    assert wall.friction_velocity > 0.0
    assert wall.shear_stress > 0.0
    assert jnp.isfinite(wall.residual)

    waves = jnp.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 1.0)))
    amplitudes = jnp.asarray(((1.0, 2.0, 0.0), (0.5, 1.0, -0.5)))
    inflow = phx.equations.SyntheticTurbulenceInflowPlan(
        waves, amplitudes, jnp.asarray((0.0, 0.3))
    )
    np.testing.assert_allclose(
        jnp.sum(inflow.wavevectors * inflow.amplitudes, axis=-1),
        0.0,
        atol=2.0e-12,
    )
    velocity = inflow.velocity(jnp.asarray(((0.1, 0.2, 0.3),)), 0.4)
    assert velocity.shape == (1, 3)
    assert jnp.all(jnp.isfinite(velocity))
