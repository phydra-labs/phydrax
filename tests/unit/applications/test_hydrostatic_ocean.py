#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _grid(*, vertical_coordinate="zstar", rest_depth=10.0):
    shape = (4, 4, 3)
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
            phx.discretization.UniformCellAxisSpec(3, periodic=False),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, -10.0), (4.0, 4.0, 0.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("hydrostatic",)
    ).prepare()
    depth = jnp.full(shape[:2], rest_depth)
    return phx.discretization.TensorZHydrostaticGridPlan(
        discretization,
        depth,
        vertical_coordinate=vertical_coordinate,
    ).prepare()


def _ocean(**kwargs):
    geometry = _grid(vertical_coordinate=kwargs.pop("vertical_coordinate", "zstar"))
    return phx.applications.ocean.HydrostaticPrimitiveEquationPlan(
        geometry, **kwargs
    ).prepare()


def _state(ocean, *, eta=0.0, salinity=35.0, temperature=10.0):
    eta_ = jnp.full(ocean.geometry.horizontal_shape, eta)
    return ocean.initialize_state(
        eta_,
        tracers={
            "absolute_salinity": jnp.full(ocean.geometry.cell_shape, salinity),
            "conservative_temperature": jnp.full(ocean.geometry.cell_shape, temperature),
        },
    )


def test_checked_tridiagonal_line_solve():
    lower = jnp.asarray(((0.0, -1.0, -1.0), (0.0, -2.0, -2.0)))
    diagonal = jnp.asarray(((2.0, 2.0, 2.0), (4.0, 4.0, 4.0)))
    upper = jnp.asarray(((-1.0, -1.0, 0.0), (-2.0, -2.0, 0.0)))
    expected = jnp.asarray(((1.0, 1.0, 1.0), (2.0, 2.0, 2.0)))
    rhs = (
        lower * jnp.roll(expected, 1, axis=-1)
        + diagonal * expected
        + upper * jnp.roll(expected, -1, axis=-1)
    )
    rhs = rhs.at[:, 0].set(diagonal[:, 0] * expected[:, 0] + upper[:, 0] * expected[:, 1])
    rhs = rhs.at[:, -1].set(
        lower[:, -1] * expected[:, -2] + diagonal[:, -1] * expected[:, -1]
    )

    result = phx.linalg.solve_tridiagonal_lines(lower, diagonal, upper, rhs, -1)

    assert bool(result.successful)
    np.testing.assert_allclose(result.value, expected, atol=1e-12)
    assert result.residual_norm <= 1e-12


def test_zstar_geometry_volume_and_continuity_identities():
    geometry = _grid()
    eta = jnp.full(geometry.horizontal_shape, 0.5)
    epoch = geometry.metric_epoch(eta)
    x = jnp.zeros(geometry.x_face_shape)
    y = jnp.zeros(geometry.y_face_shape)
    x = x.at[1, :, :].set(0.25)

    layer_net = geometry.net_cell_flux((x, y))
    barotropic_net = geometry.surface_net_flux(geometry.depth_integrate((x, y)))
    vertical = geometry.diagnose_vertical_flux((x, y))

    assert bool(epoch.valid)
    np.testing.assert_allclose(jnp.sum(epoch.layer_thickness, axis=-1), epoch.total_depth)
    np.testing.assert_allclose(jnp.sum(layer_net, axis=-1), barotropic_net)
    np.testing.assert_allclose(vertical[..., 0], 0.0)
    np.testing.assert_allclose(vertical[..., -1], -barotropic_net)


def test_partial_cell_geometry_preserves_column_depth():
    geometry = _grid(vertical_coordinate="partial-z")
    eta = jnp.zeros(geometry.horizontal_shape)
    epoch = geometry.metric_epoch(eta)

    assert bool(epoch.valid)
    np.testing.assert_allclose(
        jnp.sum(epoch.cell_volume, axis=-1),
        geometry.cell_area * geometry.rest_depth,
        atol=1e-12,
    )


def test_linear_and_nonlinear_eos_have_consistent_derivatives():
    salinity = jnp.asarray((34.0, 35.0, 36.0))
    temperature = jnp.asarray((5.0, 10.0, 15.0))
    pressure = jnp.asarray((0.0, 1000.0, 5000.0))
    linear = phx.applications.ocean.LinearHydrostaticEOS().evaluate(
        salinity, temperature, pressure
    )
    nonlinear = phx.applications.ocean.NonlinearSeawaterPolynomialEOS().evaluate(
        salinity, temperature, pressure
    )

    assert bool(linear.successful)
    assert bool(nonlinear.successful)
    assert jnp.all(linear.alpha > 0.0)
    assert jnp.all(linear.beta > 0.0)
    assert jnp.any(jnp.abs(nonlinear.density - linear.density) > 0.0)
    assert jnp.all(jnp.isfinite(nonlinear.density_pressure_derivative))


def test_implicit_free_surface_preserves_rest():
    ocean = _ocean()
    state = _state(ocean)
    epoch = ocean.geometry.metric_epoch(state.eta)

    result = ocean.free_surface.solve(
        state.eta, state.transports, epoch, jnp.asarray(0.1)
    )

    assert bool(result.successful)
    np.testing.assert_allclose(result.eta, state.eta, atol=1e-12)
    np.testing.assert_allclose(result.transports[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(result.transports[1], 0.0, atol=1e-12)
    assert result.residual_norm <= 1e-10


def test_freshwater_changes_volume_and_conserves_salt_inventory():
    freshwater = phx.applications.ocean.FreshwaterVolumeFluxPlan(
        1.0e-4,
        absolute_salinity=0.0,
        conservative_temperature=10.0,
    )
    ocean = _ocean(freshwater=freshwater)
    state = _state(ocean)
    continuation = phx.applications.ocean.HydrostaticContinuationState.initialize(
        ocean, state
    )
    method = phx.applications.ocean.HydrostaticIMEXMidpointMethod(ocean)

    result = method.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        continuation,
        jnp.asarray(0.1),
        None,
    )

    before_salt = jnp.sum(state.tracer_inventory["absolute_salinity"])
    after_salt = jnp.sum(
        result.accepted_state.state.tracer_inventory["absolute_salinity"]
    )
    assert bool(result.successful)
    assert jnp.sum(result.accepted_state.state.eta) > 0.0
    np.testing.assert_allclose(after_salt, before_salt, atol=1e-10)
    assert (
        abs(
            float(
                result.accepted_state.ledger.volume_change
                - result.accepted_state.ledger.freshwater_volume
            )
        )
        <= 1e-10
    )


def test_beta_plane_and_latitude_longitude_metrics():
    lon = jnp.linspace(0.0, 0.2, 5)
    lat = jnp.linspace(-0.3, 0.3, 5)
    z = jnp.linspace(-10.0, 0.0, 4)
    geometry = phx.discretization.LatitudeLongitudeHydrostaticGridPlan(
        lon,
        lat,
        z,
        jnp.full((4, 4), 10.0),
    ).prepare()
    epoch = geometry.metric_epoch(jnp.zeros((4, 4)))

    assert bool(epoch.valid)
    assert jnp.all(geometry.cell_area > 0.0)
    assert jnp.all(jnp.isfinite(geometry.coriolis))
    np.testing.assert_allclose(
        jnp.sum(epoch.cell_volume, axis=-1),
        geometry.cell_area * geometry.rest_depth,
    )


def test_vertical_closure_modes_return_finite_coefficients():
    for kind in ("prescribed", "ri", "kpp", "tke", "redi-gm"):
        ocean = _ocean(mixing=phx.applications.ocean.HydrostaticMixingPlan(kind))
        state = _state(ocean)
        epoch = ocean.geometry.metric_epoch(state.eta)

        viscosity, diffusivity = ocean._mixing_coefficients(state, epoch)

        assert jnp.all(jnp.isfinite(viscosity))
        assert jnp.all(jnp.isfinite(diffusivity))
        assert jnp.all(viscosity >= 0.0)
        assert jnp.all(diffusivity >= 0.0)
