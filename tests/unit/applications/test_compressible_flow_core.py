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
from phydrax.equations import (
    ChemicalPhaseKind,
    ChemicalSpeciesSchema,
    HomogeneousHelmholtzPlan,
    HomogeneousMixtureEulerSystem,
    IdealGasReferenceHelmholtzTerm,
    PolynomialSpeciesThermodynamicsPlan,
    UNIVERSAL_GAS_CONSTANT,
    ZeroResidualHelmholtzTerm,
)


def _model(species_count=2):
    names = tuple(chr(ord("A") + index) for index in range(species_count))
    schema = ChemicalSpeciesSchema.from_unique_species(
        names,
        (ChemicalPhaseKind.GAS,) * species_count,
        jnp.linspace(0.020, 0.030, species_count),
        names,
        jnp.eye(species_count, dtype=jnp.int32),
        jnp.zeros((species_count,), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
    )
    species_thermodynamics = PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.full((species_count, 1), 2.5 * UNIVERSAL_GAS_CONSTANT),
        jnp.linspace(1.0e3, 2.0e3, species_count),
        reference_molar_entropy=jnp.linspace(100.0, 120.0, species_count),
        reference_temperature=300.0,
        minimum_temperature=150.0,
        maximum_temperature=2000.0,
    )
    ideal = IdealGasReferenceHelmholtzTerm(schema, species_thermodynamics)
    return HomogeneousHelmholtzPlan(ideal, ZeroResidualHelmholtzTerm(schema))


def _uniform_primitive(system, shape=()):
    species = jnp.broadcast_to(
        jnp.asarray((0.35, 0.65), dtype=jnp.float32), shape + (system.species_count,)
    )
    velocity = jnp.broadcast_to(
        jnp.linspace(0.15, -0.05, system.dimension), shape + (system.dimension,)
    )
    temperature = jnp.full(shape + (1,), 500.0)
    return jnp.concatenate((species, velocity, temperature), axis=-1)


def test_reference_characteristic_acoustic_entropy_and_vorticity_waves():
    system = HomogeneousMixtureEulerSystem(_model(), 2)
    points = jnp.stack(
        jnp.meshgrid(jnp.linspace(0.0, 1.0, 5), jnp.linspace(0.0, 1.0, 4)),
        axis=-1,
    )
    base = _uniform_primitive(system)
    for kind in ("isentropic", "acoustic", "entropy", "vorticity"):
        plan = CompressibleReferenceWavePlan(
            kind, base, (2.0 * np.pi, 0.0), amplitude=1.0e-5
        )
        evidence = plan.evaluate(system, points, jnp.asarray(0.1))
        assert bool(jnp.all(evidence.admissible & evidence.entropy_supported))
        assert float(evidence.characteristic_identity_residual) < 2.0e-5
        np.testing.assert_allclose(
            evidence.transverse_velocity_residual, 0.0, atol=2.0e-5
        )


def test_full_species_normal_flux_reflection_and_characteristic_modes():
    system = HomogeneousMixtureEulerSystem(_model(), 2)
    state = system.primitive_to_conserved(_uniform_primitive(system))
    normal = jnp.asarray((0.6, 0.8))
    normal_flux = system.physical_normal_flux(state, normal)
    axis_flux = sum(
        normal[axis] * system.physical_flux(state, axis)
        for axis in range(system.dimension)
    )
    np.testing.assert_allclose(normal_flux, axis_flux, rtol=1.0e-6)
    reflected = system.reflect_normal_state(state, normal)
    np.testing.assert_array_equal(reflected[: system.species_count], state[:2])
    np.testing.assert_allclose(reflected[-1], state[-1])
    momentum = state[system.species_count : -1]
    reflected_momentum = reflected[system.species_count : -1]
    np.testing.assert_allclose(
        jnp.dot(reflected_momentum, normal), -jnp.dot(momentum, normal)
    )
    left, right, speeds = system.normal_eigensystem(state, state, normal)
    np.testing.assert_allclose(
        left @ right, jnp.eye(system.component_count), rtol=2.0e-5, atol=2.0e-5
    )
    primitive = system.conserved_to_primitive(state)
    convective = jnp.dot(primitive[system.species_count : -1], normal)
    np.testing.assert_allclose(
        speeds[1:-1],
        jnp.full((system.species_count + system.dimension - 1,), convective),
        rtol=1.0e-5,
    )


def test_conservative_mixture_forcing_work_and_named_budget_decomposition():
    system = HomogeneousMixtureEulerSystem(_model(), 2)
    state = system.primitive_to_conserved(_uniform_primitive(system, (3,)))
    forcing = CompressibleForcingPlan(
        system,
        acceleration=(0.5, -0.25),
        mass_rate=0.2,
        injection_mass_fractions=(0.7, 0.3),
        injection_velocity=(1.0, 0.5),
        injection_density=0.9,
        injection_temperature=450.0,
        volumetric_heating=0.75,
    ).evaluate(state)
    np.testing.assert_allclose(
        jnp.sum(forcing.species_mass_source, axis=-1), forcing.mass_source
    )
    np.testing.assert_allclose(forcing.work_identity_residual, 0.0, atol=1.0e-6)
    budget = CompressibleBudgetPlan(system).evaluate(
        state,
        forcing.source,
        velocity_gradient=jnp.zeros((3, 2, 2)),
        viscous_stress=jnp.zeros((3, 2, 2)),
        thermal_rate=jnp.zeros((3,)),
        entropy_rate=jnp.zeros((3,)),
        interface_rate=jnp.zeros((3,)),
        filter_rate=jnp.zeros((3,)),
        limiter_rate=jnp.zeros((3,)),
        sponge_rate=jnp.zeros((3,)),
        forcing_rate=forcing.total_energy_source,
        boundary_rate=jnp.zeros_like(state),
    )
    np.testing.assert_allclose(jnp.sum(budget.species_mass), budget.mass)
    np.testing.assert_allclose(jnp.sum(budget.species_mass_rate), budget.mass_rate)
    np.testing.assert_allclose(budget.decomposition_residual, 0.0, atol=1.0e-6)
    assert bool(budget.complete)


def test_favre_raw_moments_spectra_and_wall_thermal_statistics():
    nx, ny = 6, 4
    system = HomogeneousMixtureEulerSystem(_model(), 2)
    primitive = _uniform_primitive(system, (nx, ny))
    primitive = primitive.at[..., system.species_count].add(
        jnp.linspace(0.0, 0.2, nx)[:, None]
    )
    state = system.primitive_to_conserved(primitive)
    coordinates = jnp.linspace(0.0, 1.0, ny)
    plan = CompressiblePlaneStatisticsPlan(
        system,
        wall_normal_axis=1,
        wall_normal_coordinates=coordinates,
        periodic_lengths=(2.0,),
    )
    statistics = plan.evaluate(
        state,
        jnp.full((nx, ny), 1.0e-3),
        velocity_gradient=jnp.zeros((nx, ny, 2, 2)),
        thermal_conductivity=jnp.full((nx, ny), 0.03),
        temperature_gradient=jnp.zeros((nx, ny, 2)),
    )
    np.testing.assert_allclose(statistics.favre_identity_residual, 0.0, atol=1.0e-6)
    assert bool(statistics.finite)
    assert not bool(jnp.any(statistics.wall_units_available))
    assert statistics.wall_shear.shape == (2, 2)
    assert statistics.solenoidal_spectrum.shape == (nx, ny)


def test_full_species_characteristic_boundary_and_sponge_ledgers():
    system = HomogeneousMixtureEulerSystem(_model(), 1)
    far_field = system.primitive_to_conserved(_uniform_primitive(system))
    interior_primitive = _uniform_primitive(system).at[system.species_count].set(-0.1)
    interior = system.primitive_to_conserved(interior_primitive)
    result = CharacteristicNonreflectingBoundaryPlan().apply(
        system, interior, far_field, jnp.asarray((1.0,))
    )
    assert result.boundary_state.shape == (system.component_count,)
    assert bool(result.ledger.admissible)
    state = jnp.broadcast_to(interior, (6, system.component_count))
    sponge = CompressibleSpongePlan(
        system,
        far_field,
        strength=2.0,
        start_coordinate=0.5,
        end_coordinate=1.0,
    )
    sponge_result = sponge.apply(state, jnp.linspace(0.0, 1.0, 6), step_size=0.1)
    assert sponge_result.ledger.species_mass_rate.shape == (system.species_count,)
    np.testing.assert_allclose(
        jnp.sum(sponge_result.ledger.species_mass_rate),
        sponge_result.ledger.mass_rate,
    )
    assert bool(sponge_result.ledger.finite)


def test_finite_x_boundary_layer_owns_canonical_composition_and_temperature():
    model = _model()
    system = HomogeneousMixtureEulerSystem(model, 2)
    inflow = FiniteXBoundaryLayerInflowPlan(
        free_stream_density=1.2,
        free_stream_mass_fractions=(0.25, 0.75),
        free_stream_velocity=5.0,
        free_stream_temperature=500.0,
        boundary_layer_thickness=0.2,
        wall_temperature=350.0,
    )
    primitive = inflow.primitive(jnp.asarray((0.0, 0.2, 1.0)), system)
    assert primitive.shape == (3, system.component_count)
    np.testing.assert_allclose(jnp.sum(primitive[..., :2], axis=-1), 1.2)
    np.testing.assert_allclose(primitive[0, -1], 350.0)
    boundary_layer = FiniteXBoundaryLayerCaseSpec((0.0, 10.0), (0.0, 2.0), inflow)
    case = CompressibleFlowCaseSpec(
        "finite-x",
        2,
        "euler",
        "structured-fv",
        model,
        boundary_layer=boundary_layer,
    )
    assert case.prepare_system().component_count == 5
    assert boundary_layer.wall_kind == "no-slip-thermal"
