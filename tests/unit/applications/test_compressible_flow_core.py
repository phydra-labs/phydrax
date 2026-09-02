import jax.numpy as jnp
import numpy as np

from phydrax.applications.compressible_flow._boundary import (
    CharacteristicNonreflectingBoundaryPlan,
    CompressibleSpongePlan,
)
from phydrax.applications.compressible_flow._contracts import (
    CompressibleFlowCaseSpec,
    FiniteXBoundaryLayerCaseSpec,
    FiniteXBoundaryLayerInflowPlan,
)
from phydrax.applications.compressible_flow._diagnostics import (
    CompressibleBudgetPlan,
    CompressiblePlaneStatisticsPlan,
)
from phydrax.applications.compressible_flow._forcing import CompressibleForcingPlan
from phydrax.applications.compressible_flow._qualification import (
    CompressibleReferenceWavePlan,
)
from phydrax.equations._hyperbolic_systems import EulerSystem
from phydrax.equations._materials import IdealGasMaterial


def test_reference_isentropic_acoustic_entropy_and_vorticity_waves():
    x = jnp.stack(
        jnp.meshgrid(jnp.linspace(0.0, 1.0, 8), jnp.linspace(0.0, 1.0, 6), indexing="ij"),
        axis=-1,
    )
    system = EulerSystem(2)
    for kind in ("isentropic", "acoustic", "entropy", "vorticity"):
        plan = CompressibleReferenceWavePlan(
            kind,
            (0.2, -0.1),
            (2.0 * jnp.pi, 0.0),
            polarization=(0.0, 1.0),
        )
        evidence = plan.evaluate(system, x, 0.25)
        assert bool(evidence.finite)
        assert bool(jnp.all(evidence.admissible))
        np.testing.assert_allclose(evidence.pressure_relation_residual, 0.0, atol=1e-6)
        np.testing.assert_allclose(evidence.transverse_velocity_residual, 0.0, atol=1e-6)


def test_conservative_forcing_work_and_named_budget_decomposition():
    density = jnp.asarray((1.0, 2.0, 3.0))
    velocity = jnp.stack(
        (jnp.asarray((0.5, 1.0, -0.25)), jnp.asarray((0.2, 0.1, 0.3))), axis=-1
    )
    pressure = jnp.ones_like(density)
    total_energy = pressure / 0.4 + 0.5 * density * jnp.sum(velocity**2, axis=-1)
    state = jnp.concatenate(
        (density[:, None], density[:, None] * velocity, total_energy[:, None]), axis=-1
    )
    forcing = CompressibleForcingPlan(
        2,
        acceleration=(0.3, -0.2),
        mass_rate=0.1,
        injection_velocity=(0.4, 0.0),
        injection_specific_internal_energy=2.0,
        volumetric_heating=0.05,
    ).evaluate(state)
    np.testing.assert_allclose(forcing.work_identity_residual, 0.0, atol=1e-7)
    np.testing.assert_allclose(forcing.source[..., 0], forcing.mass_source)
    np.testing.assert_allclose(forcing.source[..., 1:3], forcing.momentum_source)
    np.testing.assert_allclose(forcing.source[..., -1], forcing.total_energy_source)

    zeros = jnp.zeros_like(density)
    gradient = jnp.zeros(density.shape + (2, 2))
    stress = jnp.zeros_like(gradient)
    budget = CompressibleBudgetPlan(2).evaluate(
        state,
        forcing.source,
        pressure,
        velocity_gradient=gradient,
        viscous_stress=stress,
        thermal_rate=zeros,
        entropy_rate=zeros,
        interface_rate=zeros,
        filter_rate=zeros,
        limiter_rate=zeros,
        sponge_rate=zeros,
        forcing_rate=forcing.total_energy_source,
        boundary_rate=zeros,
    )
    assert bool(budget.complete)
    np.testing.assert_allclose(budget.decomposition_residual, 0.0, atol=1e-6)
    np.testing.assert_allclose(
        budget.total_energy, budget.kinetic_energy + budget.internal_energy
    )


def test_favre_raw_moments_spectra_and_wall_thermal_statistics():
    nx, ny = 6, 4
    y = jnp.linspace(0.0, 1.0, ny)
    density = 1.0 + 0.1 * jnp.arange(nx)[:, None] + jnp.zeros((nx, ny))
    u = jnp.broadcast_to(y[None, :], (nx, ny))
    velocity = jnp.stack((u, jnp.zeros_like(u)), axis=-1)
    pressure = jnp.ones((nx, ny))
    temperature = jnp.ones((nx, ny))
    energy = pressure / 0.4 + 0.5 * density * u**2
    state = jnp.concatenate(
        (density[..., None], density[..., None] * velocity, energy[..., None]), axis=-1
    )
    gradient = jnp.zeros((nx, ny, 2, 2)).at[..., 0, 1].set(1.0)
    temperature_gradient = jnp.zeros((nx, ny, 2)).at[..., 1].set(2.0)
    viscosity = 0.1 * jnp.ones((nx, ny))
    conductivity = 0.2 * jnp.ones((nx, ny))
    plan = CompressiblePlaneStatisticsPlan(
        2,
        wall_normal_axis=1,
        wall_normal_coordinates=y,
        periodic_lengths=(2.0 * jnp.pi,),
    )
    statistics = plan.evaluate(
        state,
        pressure,
        temperature,
        jnp.sqrt(1.4) * jnp.ones((nx, ny)),
        viscosity,
        velocity_gradient=gradient,
        thermal_conductivity=conductivity,
        temperature_gradient=temperature_gradient,
    )
    assert bool(statistics.finite)
    np.testing.assert_allclose(statistics.favre_identity_residual, 0.0, atol=1e-6)
    np.testing.assert_allclose(statistics.favre_mean_velocity[..., 0], y, atol=1e-6)
    np.testing.assert_allclose(
        statistics.wall_heat_flux, jnp.asarray((0.4, -0.4)), atol=1e-6
    )
    np.testing.assert_allclose(
        statistics.raw_moments.merge(statistics.raw_moments).weight, 2.0 * nx
    )
    assert statistics.solenoidal_spectrum.shape == (nx, ny)
    assert statistics.dilatational_spectrum.shape == (nx, ny)


def test_characteristic_boundary_has_zero_outgoing_reflection_and_sponge_ledgers():
    system = EulerSystem(1)
    interior = system.primitive_to_conserved(jnp.asarray((1.0, 0.2, 1.0)))
    far_field = system.primitive_to_conserved(jnp.asarray((1.0, 0.0, 1.0)))
    boundary = CharacteristicNonreflectingBoundaryPlan().apply(
        system, interior, far_field, jnp.asarray((1.0,))
    )
    assert bool(boundary.ledger.admissible)
    np.testing.assert_allclose(boundary.ledger.reflected_energy, 0.0, atol=1e-10)

    coordinates = jnp.linspace(0.0, 1.0, 6)
    state = jnp.broadcast_to(interior, (6, 3)).at[:, 0].add(0.02)
    sponge = CompressibleSpongePlan(
        far_field,
        strength=2.0,
        start_coordinate=0.5,
        end_coordinate=1.0,
    ).apply(system, state, coordinates, step_size=0.1)
    assert bool(sponge.ledger.finite)
    assert (
        sponge.ledger.fluctuation_energy_after < sponge.ledger.fluctuation_energy_before
    )
    assert 0.0 <= sponge.ledger.reflection_coefficient < 1.0
    np.testing.assert_allclose(sponge.source[:, 0].sum(), sponge.ledger.mass_rate)
    np.testing.assert_allclose(
        sponge.source[:, -1].sum(), sponge.ledger.total_energy_rate
    )


def test_finite_x_boundary_layer_owns_inflow_outflow_and_wall_contracts():
    material = IdealGasMaterial()
    inflow = FiniteXBoundaryLayerInflowPlan(
        free_stream_density=1.0,
        free_stream_velocity=2.0,
        free_stream_pressure=1.0,
        boundary_layer_thickness=0.2,
        wall_temperature=1.0,
    )
    boundary_layer = FiniteXBoundaryLayerCaseSpec(
        (0.0, 4.0),
        (0.0, 1.0),
        inflow,
    )
    case = CompressibleFlowCaseSpec(
        "finite-x-boundary-layer",
        2,
        "navier_stokes",
        "structured-fv",
        material,
        boundary_layer=boundary_layer,
    )
    primitive = inflow.primitive(jnp.asarray((0.0, 0.2, 1.0)), material, case.dimension)
    np.testing.assert_allclose(primitive[0, 1], 0.0)
    assert primitive[-1, 1] > primitive[1, 1]
    assert boundary_layer.outflow_kind == "characteristic-nonreflecting"
    assert boundary_layer.wall_kind == "no-slip-thermal"
