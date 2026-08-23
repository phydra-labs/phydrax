#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _grid(shape):
    return phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(count) for count in shape),
        axis_names=tuple("xy"[: len(shape)]),
    ).prepare(jnp.stack((jnp.zeros(len(shape)), jnp.ones(len(shape)))))


def test_ideal_gas_material_and_euler_roundtrip_are_consistent():
    material = phx.equations.IdealGasMaterial(1.4, 287.0)
    system = phx.equations.EulerSystem(2, material=material)
    primitive = jnp.asarray([[1.2, 30.0, -4.0, 101325.0]])
    state = system.primitive_to_conserved(primitive)

    np.testing.assert_allclose(
        system.conserved_to_primitive(state), primitive, rtol=2e-12
    )
    np.testing.assert_allclose(
        system.temperature(state),
        primitive[..., -1] / (primitive[..., 0] * 287.0),
        rtol=1e-12,
    )
    assert jnp.all(system.admissible(state))
    assert system.system_id != phx.equations.EulerSystem(2).system_id


def test_stiffened_gas_pressure_energy_and_temperature_roundtrip():
    material = phx.equations.StiffenedGasMaterial(
        4.4, 6.0e8, 1816.0, reference_energy=2.0e5
    )
    density = jnp.asarray([1000.0, 950.0])
    pressure = jnp.asarray([1.0e5, 2.0e5])
    energy = material.specific_internal_energy(density, pressure)

    np.testing.assert_allclose(
        material.pressure(density, energy), pressure, rtol=2e-12
    )
    assert jnp.all(material.temperature(density, pressure) > 0.0)
    assert jnp.all(material.sound_speed(density, pressure) > 0.0)
    assert jnp.all(material.admissible(density, pressure))


def test_constant_and_sutherland_transport_have_physical_values_and_gradients():
    temperature = jnp.asarray([250.0, 300.0, 600.0])
    state = jnp.ones((3, 3))
    constant = phx.equations.ConstantTransport(1.8e-5, 0.026)
    constant_properties = constant.properties(temperature, state)
    np.testing.assert_allclose(constant_properties.dynamic_viscosity, 1.8e-5)
    np.testing.assert_allclose(constant_properties.thermal_conductivity, 0.026)

    sutherland = phx.equations.SutherlandTransport(
        1.8e-5, 300.0, 110.4, 1004.5, 0.71
    )
    properties = sutherland.properties(temperature, state)
    np.testing.assert_allclose(properties.dynamic_viscosity[1], 1.8e-5)
    assert jnp.all(jnp.diff(properties.dynamic_viscosity) > 0.0)
    gradient = jax.grad(
        lambda value: sutherland.properties(
            jnp.asarray([value]), jnp.ones((1, 3))
        ).dynamic_viscosity[0]
    )(jnp.asarray(300.0))
    assert jnp.isfinite(gradient) and gradient > 0.0


def test_material_owned_viscous_flux_resolves_couette_shear():
    grid = _grid((12, 10))
    system = phx.equations.CompressibleNavierStokesSystem(
        phx.equations.ConstantTransport(0.2, 0.0), 2
    )
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    y = grid.structured_axes[1].interval_centers
    velocity_x = jnp.broadcast_to(y[None, :], grid.shape)
    primitive = jnp.stack(
        (
            jnp.ones(grid.shape),
            velocity_x,
            jnp.zeros(grid.shape),
            jnp.ones(grid.shape),
        ),
        axis=-1,
    )
    state = system.primitive_to_conserved(primitive)
    pair = phx.discretization.FiniteVolumeBoundaryPair(
        phx.discretization.ExtrapolationBoundary(),
        phx.discretization.ExtrapolationBoundary(),
    )
    halo = phx.discretization.FiniteVolumeHaloPlan(
        discretization,
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.FiniteVolumeBoundarySet(
            ("x", "y"), (pair, pair)
        ),
    ).prepare()
    fluxes = phx.discretization.ViscousFluxPlan().face_fluxes(
        system, 0.0, state, discretization, halo
    )

    np.testing.assert_allclose(fluxes[1][1:-1, 2:-2, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(fluxes[1][1:-1, 2:-2, 1], 0.2, atol=2e-12)


def test_mapped_viscous_flux_is_zero_for_uniform_state():
    system = phx.equations.CompressibleNavierStokesSystem(
        phx.equations.ConstantTransport(0.1, 0.2), 2
    )
    base = phx.discretization.FiniteVolumePlan(
        _grid((5, 4)), component_names=system.component_names
    ).prepare()
    mapped = phx.discretization.MappedFiniteVolumePlan(
        base,
        lambda point: jnp.stack(
            (point[0] + 0.1 * point[0] * point[1], point[1])
        ),
        mapping_id="viscous-warp",
    ).prepare()
    state = system.primitive_to_conserved(
        jnp.broadcast_to(jnp.asarray([1.0, 0.2, -0.1, 1.0]), mapped.state_shape)
    )
    pair = phx.discretization.FiniteVolumeBoundaryPair(
        phx.discretization.ExtrapolationBoundary(),
        phx.discretization.ExtrapolationBoundary(),
    )
    halo = phx.discretization.FiniteVolumeHaloPlan(
        mapped,
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.FiniteVolumeBoundarySet(
            ("x", "y"), (pair, pair)
        ),
    ).prepare()
    fluxes = phx.discretization.ViscousFluxPlan().face_fluxes(
        system, 0.0, state, mapped, halo
    )
    assert all(jnp.max(jnp.abs(flux)) < 1e-11 for flux in fluxes)


def test_viscous_stability_bound_scales_with_spacing_and_transport():
    def reported_step(cells, viscosity):
        grid = phx.discretization.TensorGridPlan(
            (
                phx.discretization.UniformCellAxisSpec(
                    cells, periodic=True
                ),
            ),
            axis_names=("x",),
        ).prepare(jnp.asarray([[0.0], [1.0]]))
        system = phx.equations.CompressibleNavierStokesSystem(
            phx.equations.ConstantTransport(viscosity, 0.02)
        )
        discretization = phx.discretization.FiniteVolumePlan(
            grid, component_names=system.component_names
        ).prepare()
        primitive = jnp.broadcast_to(
            jnp.asarray([1.0, 0.0, 1.0]), (cells, 3)
        )
        state = system.primitive_to_conserved(primitive)
        return phx.discretization.ViscousFluxPlan().stability_report(
            system, state, discretization
        )

    coarse = reported_step(16, 0.1)
    fine = reported_step(32, 0.1)
    viscous = reported_step(16, 0.2)

    np.testing.assert_allclose(
        coarse.momentum_step / fine.momentum_step, 4.0, rtol=1e-12
    )
    np.testing.assert_allclose(
        coarse.momentum_step / viscous.momentum_step, 2.0, rtol=1e-12
    )
    assert coarse.selected_step > 0.0
