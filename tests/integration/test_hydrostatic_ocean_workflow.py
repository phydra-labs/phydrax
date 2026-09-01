#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _geometry(*, periodic_x=True, vertical_coordinate="zstar", depth=None):
    shape = (6, 4, 3)
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(6, periodic=periodic_x),
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
            phx.discretization.UniformCellAxisSpec(3, periodic=False),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, -10.0), (6.0, 4.0, 0.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("hydrostatic",)
    ).prepare()
    rest_depth = jnp.full(shape[:2], 10.0) if depth is None else jnp.asarray(depth)
    return phx.discretization.TensorZHydrostaticGridPlan(
        discretization,
        rest_depth,
        vertical_coordinate=vertical_coordinate,
    ).prepare()


def _state(ocean, eta):
    return ocean.initialize_state(
        eta,
        tracers={
            "absolute_salinity": jnp.full(ocean.geometry.cell_shape, 35.0),
            "conservative_temperature": jnp.full(ocean.geometry.cell_shape, 10.0),
        },
    )


def test_implicit_hydrostatic_external_wave_is_volume_conservative():
    geometry = _geometry()
    ocean = phx.applications.ocean.HydrostaticPrimitiveEquationPlan(
        geometry,
        coriolis_f0=1.0e-4,
        mixing=phx.applications.ocean.HydrostaticMixingPlan(
            "prescribed",
            background_viscosity=1.0e-5,
            background_diffusivity=1.0e-6,
        ),
    ).prepare()
    x = jnp.arange(geometry.horizontal_shape[0])[:, None]
    eta = 1.0e-3 * jnp.sin(2.0 * jnp.pi * x / geometry.horizontal_shape[0])
    eta = jnp.broadcast_to(eta, geometry.horizontal_shape)
    state = _state(ocean, eta)
    continuation = phx.applications.ocean.HydrostaticContinuationState.initialize(state)
    method = phx.applications.ocean.HydrostaticIMEXMidpointMethod(ocean)

    result = method.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        continuation,
        jnp.asarray(0.02),
        None,
    )

    assert bool(result.successful)
    np.testing.assert_allclose(
        jnp.sum(geometry.cell_area * (result.accepted_state.state.eta - state.eta)),
        0.0,
        atol=2e-9,
    )
    assert jnp.all(jnp.isfinite(result.accepted_state.state.transports[0]))


def test_split_explicit_wetdry_keeps_nonnegative_depth_and_inventory():
    x = jnp.linspace(0.0, 1.0, 6)[:, None]
    depth = jnp.broadcast_to(0.02 + 0.98 * x, (6, 4))
    geometry = _geometry(depth=depth)
    ocean = phx.applications.ocean.HydrostaticPrimitiveEquationPlan(
        geometry,
        external_mode="split-explicit",
        wetting_and_drying=True,
        wet_depth=1.0e-4,
        split_substeps=10,
    ).prepare()
    eta = jnp.zeros(geometry.horizontal_shape)
    state = _state(ocean, eta)
    x_transport = state.transports[0].at[1, :, :].set(-2.0e-4)
    state = phx.applications.ocean.HydrostaticOceanState(
        state.eta,
        (x_transport, state.transports[1]),
        state.tracer_inventory,
        state.tke_inventory,
    )
    continuation = phx.applications.ocean.HydrostaticContinuationState.initialize(state)

    result = phx.applications.ocean.HydrostaticIMEXMidpointMethod(ocean).step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        continuation,
        jnp.asarray(0.01),
        None,
    )
    final = result.accepted_state.state
    epoch = geometry.metric_epoch(final.eta)

    assert bool(result.successful)
    assert jnp.all(epoch.total_depth >= 0.0)
    assert jnp.all(final.tracer_inventory["absolute_salinity"] >= 0.0)
    assert result.accepted_state.ledger.limiter_correction >= 0.0


def test_flather_boundary_and_freshwater_share_volume_ledger():
    geometry = _geometry(periodic_x=False)
    boundary = phx.applications.ocean.HydrostaticOpenBoundary(
        0,
        "lower",
        "flather",
        target_eta=0.0,
        target_transport=0.0,
    )
    freshwater = phx.applications.ocean.FreshwaterVolumeFluxPlan(2.0e-5)
    ocean = phx.applications.ocean.HydrostaticPrimitiveEquationPlan(
        geometry,
        boundaries=(boundary,),
        freshwater=freshwater,
    ).prepare()
    state = _state(ocean, jnp.full(geometry.horizontal_shape, 1.0e-3))
    continuation = phx.applications.ocean.HydrostaticContinuationState.initialize(state)

    result = phx.applications.ocean.HydrostaticIMEXMidpointMethod(ocean).step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        continuation,
        jnp.asarray(0.01),
        None,
    )

    assert bool(result.successful)
    assert jnp.isfinite(result.accepted_state.ledger.open_boundary_volume)
    assert jnp.isfinite(result.accepted_state.ledger.freshwater_volume)


def test_nonlinear_eos_and_kpp_like_closure_advance_finitely():
    geometry = _geometry(vertical_coordinate="partial-z")
    ocean = phx.applications.ocean.HydrostaticPrimitiveEquationPlan(
        geometry,
        eos=phx.applications.ocean.NonlinearSeawaterPolynomialEOS(),
        mixing=phx.applications.ocean.HydrostaticMixingPlan(
            "kpp", maximum_coefficient=1.0e-3
        ),
    ).prepare()
    state = _state(ocean, jnp.zeros(geometry.horizontal_shape))
    continuation = phx.applications.ocean.HydrostaticContinuationState.initialize(state)

    result = phx.applications.ocean.HydrostaticIMEXMidpointMethod(ocean).step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        continuation,
        jnp.asarray(0.01),
        None,
    )

    assert bool(result.successful)
    assert jnp.all(
        jnp.isfinite(
            result.accepted_state.state.tracer_inventory["conservative_temperature"]
        )
    )
