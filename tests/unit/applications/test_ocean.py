#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _prepared_ocean(
    *,
    shape=(4, 4, 3),
    coriolis=0.0,
    temperature_diffusivity=0.0,
    salinity_diffusivity=0.0,
    temperature_flux=None,
    surface_stress=None,
):
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
    reference = phx.applications.ocean.LinearSeawaterReference()
    plan = phx.applications.ocean.CartesianBoussinesqOceanPlan(
        phx.applications.ocean.OceanAxisConvention(),
        reference,
        coriolis_parameter=coriolis,
        temperature_diffusivity=temperature_diffusivity,
        salinity_diffusivity=salinity_diffusivity,
        temperature_surface_flux=temperature_flux,
        surface_stress=surface_stress,
    )
    return plan.prepare(discretization)


def _rest_state(ocean):
    discretization = ocean.operators.discretization
    velocity = tuple(
        jnp.zeros(layout.shape, dtype=discretization.cell_volumes.dtype)
        for layout in discretization.face_layouts
    )
    temperature = jnp.full(
        discretization.cell_shape,
        ocean.plan.reference.reference_temperature,
    )
    salinity = jnp.full(
        discretization.cell_shape,
        ocean.plan.reference.reference_salinity,
    )
    return ocean.initial_state(velocity, temperature, salinity)


def test_linear_seawater_reference_density_compensation():
    reference = phx.applications.ocean.LinearSeawaterReference()
    temperature = jnp.asarray((11.0, 9.0))
    salinity = reference.reference_salinity + (
        reference.thermal_expansion
        / reference.haline_contraction
        * (temperature - reference.reference_temperature)
    )

    np.testing.assert_allclose(
        reference.density_anomaly(temperature, salinity), 0.0, atol=1e-13
    )
    np.testing.assert_allclose(
        reference.temperature_flux_from_heat_flux(
            reference.reference_density * reference.heat_capacity
        ),
        1.0,
    )


def test_scalar_cfl_uses_oriented_face_flux_not_canceling_average():
    ocean = _prepared_ocean()
    discretization = ocean.operators.discretization
    x_layout = discretization.face_layouts[0]
    alternating = jnp.where(jnp.arange(x_layout.shape[0]) % 2 == 0, 1.0, -1.0).reshape(
        (x_layout.shape[0], 1, 1)
    )
    velocity = (
        jnp.broadcast_to(alternating, x_layout.shape),
        jnp.zeros(discretization.face_layouts[1].shape),
        jnp.zeros(discretization.face_layouts[2].shape),
    )

    restriction = ocean.transport.step_restriction(velocity)

    assert jnp.isfinite(restriction.advective["temperature"])
    assert restriction.advective["temperature"] > 0.0


def test_directional_scalar_diffusion_and_surface_flux_are_conservative():
    flux = phx.discretization.MACScalarBoundaryCondition("flux", 2.0e-6)
    ocean = _prepared_ocean(
        temperature_diffusivity=jnp.asarray((1.0e-4, 1.0e-4, 1.0e-5)),
        salinity_diffusivity=jnp.asarray((1.0e-5, 1.0e-5, 1.0e-6)),
        temperature_flux=flux,
    )
    state = _rest_state(ocean)
    velocity, scalars = ocean.dynamics.unpack_state(state)
    results = ocean.transport.evaluate(0.0, scalars, velocity)
    diagnostics = ocean.transport.diagnostics_from_fluxes(scalars, results)
    top_area = jnp.sum(
        jnp.take(
            ocean.operators.discretization.face_measures[2],
            -1,
            axis=2,
        )
    )

    np.testing.assert_allclose(
        diagnostics.fields["temperature"].diffusive_content_rate,
        -2.0e-6 * top_area,
        rtol=1e-11,
        atol=1e-13,
    )
    assert jnp.isfinite(
        ocean.transport.step_restriction(velocity).diffusive["temperature"]
    )


def test_mac_coriolis_is_weighted_power_neutral():
    ocean = _prepared_ocean(coriolis=0.5)
    discretization = ocean.operators.discretization
    x = jnp.arange(np.prod(discretization.face_layouts[0].shape)).reshape(
        discretization.face_layouts[0].shape
    )
    y = jnp.arange(np.prod(discretization.face_layouts[1].shape)).reshape(
        discretization.face_layouts[1].shape
    )
    velocity = (
        jnp.sin(x),
        jnp.cos(y),
        jnp.zeros(discretization.face_layouts[2].shape),
    )

    evidence = ocean.dynamics.ocean_forcing.evaluate(0.0, velocity)

    assert bool(evidence.success)
    assert (
        evidence.normalized_coriolis_work_defect
        <= evidence.coriolis_work_scale * 0 + 1e-12
    )
    np.testing.assert_allclose(evidence.surface_stress_power, 0.0)


def test_ocean_stage_and_wave_restrictions_are_finite():
    ocean = _prepared_ocean(coriolis=0.25)
    state = _rest_state(ocean)

    stage = ocean.dynamics.stage(0.0, state)
    restriction = ocean.dynamics.step_restriction(state)

    assert bool(stage.success)
    assert jnp.isfinite(restriction.ocean_forcing)
    assert restriction.ocean_forcing > 0.0
    assert jnp.isinf(restriction.stratification)
    np.testing.assert_allclose(stage.buoyancy.normalized_exchange_defect, 0.0)


def test_ocean_checkpoint_round_trip(tmp_path):
    ocean = _prepared_ocean()
    continuation = phx.applications.ocean.OceanBoussinesqContinuationState.initialize(
        _rest_state(ocean)
    )
    target = tmp_path / "ocean.chk"

    phx.applications.ocean.write_ocean_checkpoint(
        target,
        ocean,
        jnp.asarray(0.25),
        jnp.asarray(4, dtype=jnp.int32),
        continuation,
    )
    time, step, restored = phx.applications.ocean.read_ocean_checkpoint(
        target, ocean, continuation
    )

    np.testing.assert_allclose(time, 0.25)
    assert int(step) == 4
    np.testing.assert_allclose(restored.coordinates, continuation.coordinates)
    assert jax_tree_allclose(restored, continuation)


def jax_tree_allclose(left, right):
    return all(
        np.allclose(np.asarray(a), np.asarray(b))
        for a, b in zip(jax_tree_leaves(left), jax_tree_leaves(right), strict=True)
    )


def jax_tree_leaves(value):
    import jax

    return jax.tree.leaves(value)


def test_dynamic_surface_scalar_flux_uses_stage_time_and_args():
    condition = phx.discretization.MACScalarBoundaryCondition(
        "flux",
        lambda time, coordinates, args: args * time * jnp.ones(coordinates.shape[:-1]),
        function_id="time-scaled-temperature-flux",
    )
    ocean = _prepared_ocean(temperature_flux=condition)
    state = _rest_state(ocean)
    velocity, scalars = ocean.dynamics.unpack_state(state)
    result = ocean.transport.evaluate(2.0, scalars, velocity, 3.0)
    diagnostics = ocean.transport.diagnostics_from_fluxes(scalars, result)
    top_area = jnp.sum(
        jnp.take(ocean.operators.discretization.face_measures[2], -1, axis=2)
    )

    np.testing.assert_allclose(
        diagnostics.fields["temperature"].diffusive_content_rate,
        -6.0 * top_area,
        rtol=1e-12,
        atol=1e-12,
    )


def test_surface_stress_is_tangential_and_top_layer_owned():
    ocean = _prepared_ocean(surface_stress=(2.0, 0.0, 0.0))
    discretization = ocean.operators.discretization
    velocity = (
        jnp.ones(discretization.face_layouts[0].shape),
        jnp.zeros(discretization.face_layouts[1].shape),
        jnp.zeros(discretization.face_layouts[2].shape),
    )

    evidence = ocean.dynamics.ocean_forcing.evaluate(0.0, velocity)
    top = jnp.take(evidence.surface_stress_force[0], -1, axis=2)
    below = jnp.take(evidence.surface_stress_force[0], 0, axis=2)

    assert bool(evidence.success)
    assert jnp.all(top > 0.0)
    np.testing.assert_allclose(below, 0.0)
    np.testing.assert_allclose(evidence.surface_stress_force[2], 0.0)
    assert evidence.surface_stress_power > 0.0


def test_ocean_diagnostic_output_contains_named_fields(tmp_path):
    from phydrax._array_archive import read_array_archive

    ocean = _prepared_ocean(coriolis=0.25)
    continuation = phx.applications.ocean.OceanBoussinesqContinuationState.initialize(
        _rest_state(ocean)
    )
    target = tmp_path / "ocean-output.zip"

    phx.applications.ocean.write_ocean_output(
        target,
        ocean,
        jnp.asarray(0.0),
        continuation,
    )
    manifest, arrays = read_array_archive(target)

    assert manifest["kind"] == "ocean-boussinesq-output"
    assert manifest["ocean_id"] == ocean.prepared_id
    assert {
        "temperature",
        "salinity",
        "density_anomaly",
        "buoyancy",
        "pressure",
        "velocity/0",
        "velocity/1",
        "velocity/2",
    }.issubset(arrays)


def test_coupled_stage_propagates_dynamic_boundary_data():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(3, periodic=True),
            phx.discretization.UniformCellAxisSpec(3, periodic=True),
            phx.discretization.UniformCellAxisSpec(3, periodic=False),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, -1.0), (1.0, 1.0, 0.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("ocean",)
    ).prepare()
    operators = phx.discretization.MACOperatorPlan(discretization).prepare()
    provider = phx.discretization.MACBoundaryProvider(
        function=lambda time, coordinates, args: (
            jnp.stack(
                (
                    jnp.full(coordinates[0].shape, time),
                    jnp.zeros_like(coordinates[0]),
                    jnp.zeros_like(coordinates[0]),
                ),
                axis=0,
            ),
            jnp.stack(
                (
                    jnp.ones_like(coordinates[0]),
                    jnp.zeros_like(coordinates[0]),
                    jnp.zeros_like(coordinates[0]),
                ),
                axis=0,
            ),
        ),
        provider_id="moving-ocean-lid",
    )
    boundaries = phx.discretization.MACBoundaryPlan(
        operators,
        (
            phx.discretization.MACBoundarySide("z", "lower", "no-slip"),
            phx.discretization.MACBoundarySide(
                "z", "upper", "no-slip", provider=provider
            ),
        ),
    )
    ocean = phx.applications.ocean.CartesianBoussinesqOceanPlan(
        phx.applications.ocean.OceanAxisConvention(),
        phx.applications.ocean.LinearSeawaterReference(),
        viscosity=0.1,
    ).prepare(discretization, boundaries=boundaries)
    state = _rest_state(ocean)

    stage = ocean.dynamics.stage(0.5, state)

    assert bool(stage.success)
    assert jnp.max(jnp.abs(stage.unconstrained_velocity_rate[0])) > 0.0
