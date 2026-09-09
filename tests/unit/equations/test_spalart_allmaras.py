import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications.compressible_flow import (
    FlatWallDistancePlan,
    SpalartAllmarasManufacturedPlan,
    SpalartAllmarasWallBoundary,
)


def _base_system():
    schema = phx.equations.ChemicalSpeciesSchema.from_unique_species(
        ("air",),
        (phx.equations.ChemicalPhaseKind.GAS,),
        jnp.asarray((0.02897,)),
        ("air",),
        jnp.asarray(((1,),), dtype=jnp.int32),
        jnp.asarray((0,), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
    )
    calorics = phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray((2.5 * phx.equations.UNIVERSAL_GAS_CONSTANT,)),
        jnp.asarray((0.0,)),
        reference_temperature=300.0,
        minimum_temperature=100.0,
        maximum_temperature=3000.0,
    )
    thermodynamics = phx.equations.HomogeneousHelmholtzPlan(
        phx.equations.IdealGasReferenceHelmholtzTerm(schema, calorics),
        phx.equations.ZeroResidualHelmholtzTerm(schema),
    )
    return phx.equations.HomogeneousMixtureCompressibleNavierStokesSystem(
        thermodynamics,
        phx.equations.ConstantTransport(1.0e-5, 0.02),
        2,
    )


def test_sa_negative_closure_has_positive_and_recovery_branches():
    plan = phx.equations.SpalartAllmarasNegativePlan()
    gradient = jnp.asarray(((0.0, 2.0), (0.0, 0.0)))
    working_gradient = jnp.asarray((0.01, -0.02))
    positive = plan.evaluate(1.0, 1.0e-5, 2.0e-4, gradient, working_gradient, 0.1)
    negative = plan.evaluate(1.0, 1.0e-5, -5.0e-6, gradient, working_gradient, 0.1)

    assert bool(positive.successful)
    assert bool(negative.successful)
    assert positive.eddy_viscosity > 0.0
    np.testing.assert_allclose(negative.eddy_viscosity, 0.0, atol=0.0)
    assert negative.source > 0.0
    assert positive.diffusion_coefficient > 0.0
    assert negative.diffusion_coefficient > 0.0


def test_sa_system_roundtrip_diffusion_and_source_ledger():
    base = _base_system()
    system = phx.equations.SpalartAllmarasCompressibleSystem(base)

    def state_at(y):
        primitive = jnp.asarray((1.0, y, 0.0, 500.0, 2.0e-4))
        return system.primitive_to_conserved(primitive)

    state = state_at(jnp.asarray(0.1))
    derivative = jax.jacfwd(state_at)(jnp.asarray(0.1))
    gradient = jnp.stack((jnp.zeros_like(derivative), derivative), axis=-1)
    arguments = phx.equations.SpalartAllmarasArguments(jnp.asarray(0.1))
    diffusion = system.diffusion_evaluation(state, gradient, arguments)

    assert bool(diffusion.successful)
    assert diffusion.flux.shape == (system.component_count, system.dimension)
    assert diffusion.source.shape == (system.component_count,)
    np.testing.assert_allclose(diffusion.source[system.energy_index], 0.0, atol=0.0)
    assert diffusion.source[system.turbulence_index] != 0.0
    np.testing.assert_allclose(
        system.conserved_to_primitive(state),
        jnp.asarray((1.0, 0.1, 0.0, 500.0, 2.0e-4)),
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    assert bool(system.admissible(state))
    assert system.maximum_diffusivity(state, arguments) > 0.0


def test_sa_wall_sets_zero_face_working_variable_and_flat_distance_is_exact():
    base = _base_system()
    system = phx.equations.SpalartAllmarasCompressibleSystem(base)
    wall = SpalartAllmarasWallBoundary(
        phx.discretization.NoSlipAdiabaticWallBoundary(jnp.asarray((0.0, 0.0)))
    )
    interior = system.primitive_to_conserved(jnp.asarray((1.0, 2.0, 0.0, 500.0, 3.0e-4)))
    exterior = wall.exterior_state(
        system,
        jnp.asarray(0.0),
        interior,
        jnp.asarray((0.5, 0.1)),
        jnp.asarray((0.0, -1.0)),
        1,
        None,
    )
    np.testing.assert_allclose(
        0.5 * (system.working_variable(interior) + system.working_variable(exterior)),
        0.0,
        atol=1.0e-12,
    )

    centers = jnp.asarray((((0.25, 0.1), (0.25, 0.3)), ((0.75, 0.1), (0.75, 0.3))))
    distance = FlatWallDistancePlan(1, 0.0, "lower").prepare(
        centers, geometry_id="flat-test"
    )
    np.testing.assert_allclose(distance.distance, centers[..., 1], atol=0.0)


def test_sa_manufactured_plan_returns_complete_finite_rate():
    system = phx.equations.SpalartAllmarasCompressibleSystem(_base_system())
    plan = SpalartAllmarasManufacturedPlan(
        system,
        lambda point, args: jnp.asarray(
            (
                1.0 + 0.02 * point[0],
                0.3 + 0.1 * point[1],
                0.05 * point[0],
                500.0 + 3.0 * point[1],
                2.0e-4 + 1.0e-5 * point[0],
            )
        ),
        lambda point, args: point[1] + 0.5,
        case_id="sa-negative-manufactured",
    )
    points = jnp.asarray(((0.2, 0.3), (0.7, 0.6)))
    evidence = plan.evaluate(points)

    assert bool(jnp.all(evidence.successful))
    assert evidence.exact_rate.shape == (2, system.component_count)
    assert jnp.all(jnp.isfinite(evidence.exact_rate))
