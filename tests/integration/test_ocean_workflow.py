#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _ocean(*, coriolis=0.0, temperature_flux=None):
    shape = (4, 4, 4)
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(shape[0], periodic=True),
            phx.discretization.UniformCellAxisSpec(shape[1], periodic=True),
            phx.discretization.UniformCellAxisSpec(shape[2], periodic=False),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, -1.0), (1.0, 1.0, 0.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("ocean",)
    ).prepare()
    plan = phx.applications.ocean.CartesianBoussinesqOceanPlan(
        phx.applications.ocean.OceanAxisConvention(),
        phx.applications.ocean.LinearSeawaterReference(),
        coriolis_parameter=coriolis,
        temperature_surface_flux=temperature_flux,
    )
    return plan.prepare(discretization)


def _state(ocean, *, u=0.0, v=0.0, temperature=None):
    discretization = ocean.operators.discretization
    velocity = (
        jnp.full(discretization.face_layouts[0].shape, u),
        jnp.full(discretization.face_layouts[1].shape, v),
        jnp.zeros(discretization.face_layouts[2].shape),
    )
    reference = ocean.plan.reference
    temperature_ = (
        jnp.full(discretization.cell_shape, reference.reference_temperature)
        if temperature is None
        else temperature
    )
    salinity = jnp.full(discretization.cell_shape, reference.reference_salinity)
    return ocean.initial_state(velocity, temperature_, salinity)


def test_ocean_fixed_step_inertial_oscillation():
    ocean = _ocean(coriolis=0.5)
    continuation = phx.applications.ocean.OceanBoussinesqContinuationState.initialize(
        _state(ocean, u=1.0)
    )
    method = phx.applications.ocean.OceanBoussinesqSSPRK33Method(ocean)
    problem = phx.solver.FixedStepProblem(
        method,
        continuation,
        t0=0.0,
        t1=0.1,
        step_size=0.01,
        state_geometry=phx.metrix.EuclideanStateGeometry(),
    )

    solution = phx.solver.solve_fixed_step(problem)
    final = jax_tree_last(solution.states)
    view = ocean.state_view(final.coordinates)

    assert bool(solution.successful)
    np.testing.assert_allclose(view.velocity[0], np.cos(0.05), atol=3e-8)
    np.testing.assert_allclose(view.velocity[1], -np.sin(0.05), atol=3e-8)
    np.testing.assert_allclose(final.coriolis_work, 0.0, atol=2e-12)


def test_stratification_restriction_detects_internal_wave_scale():
    ocean = _ocean()
    z = ocean.operators.discretization.grid.structured_axes[2].interval_centers
    temperature = 10.0 + jnp.broadcast_to(
        z.reshape((1, 1, z.size)), ocean.operators.discretization.cell_shape
    )
    state = _state(ocean, temperature=temperature)

    restriction = ocean.dynamics.step_restriction(0.0, state)

    assert jnp.isfinite(restriction.stratification)
    assert restriction.stratification > 0.0


def test_surface_heat_flux_updates_only_accepted_temperature_inventory():
    flux = phx.discretization.MACScalarBoundaryCondition("flux", 1.0e-5)
    ocean = _ocean(temperature_flux=flux)
    coordinates = _state(ocean)
    continuation = phx.applications.ocean.OceanBoussinesqContinuationState.initialize(
        coordinates
    )
    method = phx.applications.ocean.OceanBoussinesqSSPRK33Method(ocean)
    dt = 0.01

    result = method.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        continuation,
        jnp.asarray(dt),
        None,
    )
    before = ocean.state_view(continuation.coordinates).temperature
    after = ocean.state_view(result.accepted_state.coordinates).temperature
    volumes = ocean.operators.discretization.cell_volumes
    content_change = jnp.sum(volumes * (after - before))
    top_area = jnp.sum(
        jnp.take(ocean.operators.discretization.face_measures[2], -1, axis=2)
    )

    assert bool(result.successful)
    np.testing.assert_allclose(
        content_change,
        -dt * 1.0e-5 * top_area,
        rtol=2e-10,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        result.accepted_state.temperature_boundary_content,
        content_change,
        rtol=2e-10,
        atol=2e-13,
    )
    assert jnp.isfinite(result.accepted_state.boundary_potential_energy)
    assert result.accepted_state.boundary_potential_energy != 0.0
    assert jnp.isfinite(result.accepted_state.molecular_potential_energy_mixing)
    np.testing.assert_allclose(
        result.accepted_state.sgs_potential_energy_mixing,
        0.0,
        atol=2e-13,
    )


def jax_tree_last(tree):
    import jax

    return jax.tree.map(lambda leaf: leaf[-1], tree)
