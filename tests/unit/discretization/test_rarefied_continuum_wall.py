#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _system(dimension=2):
    schema = phx.equations.ChemicalSpeciesSchema.from_unique_species(
        ("A",),
        (phx.equations.ChemicalPhaseKind.GAS,),
        jnp.asarray((0.028,)),
        ("A",),
        jnp.asarray(((1,),), dtype=jnp.int32),
        jnp.asarray((0,), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
    )
    calorics = phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.full((1, 1), 2.5 * phx.equations.UNIVERSAL_GAS_CONSTANT),
        jnp.asarray((0.0,)),
        reference_molar_entropy=jnp.asarray((100.0,)),
        reference_temperature=300.0,
        minimum_temperature=100.0,
        maximum_temperature=1000.0,
    )
    thermodynamics = phx.equations.HomogeneousHelmholtzPlan(
        phx.equations.IdealGasReferenceHelmholtzTerm(schema, calorics),
        phx.equations.ZeroResidualHelmholtzTerm(schema),
    )
    return phx.equations.HomogeneousMixtureCompressibleNavierStokesSystem(
        thermodynamics,
        phx.equations.ConstantTransport(1.0e-5, 0.02),
        dimension,
    )


def _wall(
    *,
    tmac=1.0,
    thermal=1.0,
    temperature=290.0,
    heat_flux=None,
    dimension=2,
):
    coefficients = phx.discretization.MaxwellSmoluchowskiWallCoefficients(
        slip_prefactor=1.0,
        temperature_jump_prefactor=1.0,
        thermal_creep_prefactor=0.75,
        convention_id="test-first-order-convention",
    )
    material = phx.discretization.ContinuumGasWallMaterial(
        jnp.zeros((dimension,)),
        tangential_momentum_accommodation=tmac,
        thermal_accommodation=thermal,
        wall_temperature=None if heat_flux is not None else temperature,
        outward_heat_flux=heat_flux,
    )
    regime = phx.discretization.WallRegimePolicy(
        characteristic_length=1.0,
        maximum_knudsen_number=1.0,
        maximum_lambda_to_wall_distance=1.0,
    )
    return phx.discretization.MaxwellSmoluchowskiContinuumWallPlan(
        coefficients, material, regime
    )


def _state_and_gradient(system):
    def state_at(y):
        primitive = jnp.asarray((1.0, 2.0 + 4.0 * y, 0.0, 300.0 + 10.0 * y))
        return system.primitive_to_conserved(primitive)

    coordinate = jnp.asarray(0.1)
    state = state_at(coordinate)
    derivative = jax.jacfwd(state_at)(coordinate)
    gradient = jnp.zeros((system.component_count, 2)).at[:, 1].set(derivative)
    return state[None, :], gradient[None, :, :]


def test_slip_jump_and_thermal_creep_have_independent_accommodation():
    system = _system()
    state, gradient = _state_and_gradient(system)
    normal = jnp.asarray((0.0, -1.0))
    baseline = _wall().evaluate_normal_flux(
        system, state, gradient, jnp.asarray((0.1,)), normal
    )
    lower_tmac = _wall(tmac=0.5).evaluate_normal_flux(
        system, state, gradient, jnp.asarray((0.1,)), normal
    )
    lower_thermal = _wall(thermal=0.5).evaluate_normal_flux(
        system, state, gradient, jnp.asarray((0.1,)), normal
    )

    assert bool(baseline.header.globally_eligible)
    assert baseline.gas_velocity_trace[0, 0] > 0.0
    assert baseline.gas_temperature_trace[0] > 290.0
    assert lower_tmac.slip_length[0] > baseline.slip_length[0]
    np.testing.assert_allclose(
        lower_tmac.temperature_jump_length, baseline.temperature_jump_length
    )
    assert lower_thermal.temperature_jump_length[0] > baseline.temperature_jump_length[0]
    np.testing.assert_allclose(lower_thermal.slip_length, baseline.slip_length)


def test_prescribed_outward_heat_flux_sets_physical_normal_flux_and_wall_refuses_ale():
    system = _system()
    state, gradient = _state_and_gradient(system)
    wall = _wall(heat_flux=2.0)
    evaluation = wall.evaluate_normal_flux(
        system,
        state,
        gradient,
        jnp.asarray((0.1,)),
        jnp.asarray((0.0, -1.0)),
    )

    assert bool(evaluation.header.globally_eligible)
    np.testing.assert_allclose(evaluation.outward_thermal_flux, 2.0, rtol=1.0e-12)
    with pytest.raises(ValueError, match="ALE"):
        wall.ale_exterior_state(system, state, None, 1)


def test_rarefied_wall_rejects_unsupported_gas_system():
    system = phx.equations.CompressibleNavierStokesSystem(
        phx.equations.ConstantTransport(0.1, 0.2), 2
    )
    state = system.primitive_to_conserved(jnp.asarray(((1.0, 0.0, 0.0, 1.0),)))
    with pytest.raises(TypeError, match="ideal-mixture"):
        _wall().evaluate_normal_flux(
            system,
            state,
            jnp.zeros((1, system.component_count, 2)),
            jnp.asarray((0.1,)),
            jnp.asarray((0.0, -1.0)),
        )


def test_structured_viscous_flux_consumes_rarefied_wall_normal_flux():
    system = _system(dimension=1)
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8),), axis_names=("x",)
    ).prepare(jnp.asarray(((0.0,), (1.0,))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    wall = _wall(dimension=1)
    boundaries = phx.discretization.FiniteVolumeBoundarySet(
        ("x",),
        (phx.discretization.FiniteVolumeBoundaryPair(wall, wall),),
    )
    halo = phx.discretization.FiniteVolumeHaloPlan(
        discretization,
        phx.discretization.PiecewiseConstantReconstruction(),
        boundaries,
    ).prepare()
    primitive = jnp.stack(
        (
            jnp.ones((8,)),
            jnp.linspace(-0.1, 0.1, 8),
            jnp.linspace(290.0, 310.0, 8),
        ),
        axis=-1,
    )
    state = system.primitive_to_conserved(primitive)
    evaluation = phx.discretization.ViscousFluxPlan().evaluate(
        system, jnp.asarray(0.0), state, discretization, halo
    )

    assert bool(evaluation.successful)
    assert bool(jnp.all(jnp.isfinite(evaluation.face_fluxes[0])))
    assert evaluation.face_fluxes[0].shape == (9, system.component_count)
