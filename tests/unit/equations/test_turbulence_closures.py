#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _arguments(gradient, model_state=(0.1, 1.0)):
    return phx.equations.RANSTurbulenceArguments(
        jnp.asarray(gradient),
        jnp.asarray(0.02),
        jnp.asarray(model_state),
    )


def test_rans_eddy_viscosities_are_nonnegative_and_finite():
    gradient = jnp.asarray(((0.2, -0.4, 0.1), (0.3, -0.1, 0.2), (-0.2, 0.1, -0.1)))
    arguments = _arguments(gradient)
    state = jnp.asarray((1.0,))
    for plan in (
        phx.equations.SpalartAllmarasPlan(),
        phx.equations.KOmegaSSTPlan(),
    ):
        viscosity = plan.kinematic_viscosity(state, arguments)
        assert jnp.isfinite(viscosity)
        assert viscosity >= 0.0
    assert jnp.isfinite(phx.equations.SpalartAllmarasPlan().source(arguments))
    assert jnp.all(jnp.isfinite(phx.equations.KOmegaSSTPlan().source(arguments)))


def test_turbulent_transport_is_operational():
    molecular = phx.equations.ConstantTransport(1.0e-5, 0.02)
    closure = phx.equations.TurbulentTransportClosure(
        molecular,
        phx.equations.SpalartAllmarasPlan(),
        specific_heat_cp=1004.5,
    )
    state = jnp.asarray((1.2, 0.0, 0.0, 0.0, 3.0e5))
    arguments = _arguments(
        jnp.asarray(((0.1, 0.3, 0.0), (0.0, -0.1, 0.2), (0.0, 0.0, 0.0)))
    )
    properties = closure.properties(jnp.asarray(300.0), state, arguments)
    assert properties.dynamic_viscosity >= 1.0e-5
    assert properties.thermal_conductivity >= 0.02
