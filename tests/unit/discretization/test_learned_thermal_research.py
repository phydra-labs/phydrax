#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.discrete_velocity._learned_thermal_research import (
    IntegerVelocityFrameShiftPlan,
    LearnedThermalResearchStatus,
    MatchedThermalCrossRelaxationPlan,
    PositiveLearnedThermalEnergyPlan,
    PressureExtendedParticleEquilibriumPlan,
)
from phydrax.discretization.discrete_velocity._quadrature import (
    CertifiedDiscreteVelocityQuadrature,
    d2v17_quadrature,
    d2v37_off_lattice_quadrature,
)


jax.config.update("jax_enable_x64", True)


def _d2q9_quadrature():
    velocities = np.asarray(
        (
            (0, 0),
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ),
        dtype=np.float64,
    )
    directional_weights = np.where(velocities == 0.0, 2.0 / 3.0, 1.0 / 6.0)
    return CertifiedDiscreteVelocityQuadrature(
        "D2Q9-learned-thermal-research",
        velocities,
        np.prod(directional_weights, axis=-1),
        reference_temperature=1.0 / 3.0,
        certified_degree=4,
        transport_kind="integer_lattice",
    )


def _thermal_statistics(quadrature):
    velocities = quadrature.velocities
    cx = velocities[:, 0]
    cy = velocities[:, 1]
    return jnp.stack(
        (
            jnp.ones_like(cx),
            cx,
            cy,
            cx * cx + cy * cy,
            cx * cy,
        ),
        axis=-1,
    )


def _energy_equilibrium(quadrature, total_energy=2.4):
    plan = PositiveLearnedThermalEnergyPlan(quadrature, 5)
    result = plan.evaluate(
        jnp.asarray(total_energy),
        _thermal_statistics(quadrature),
        jnp.asarray((0.0, 0.08, -0.04, -0.025, 0.015)),
    )
    assert bool(result.successful)
    return result


def test_pressure_extended_particle_equilibrium_has_exact_pressure_stress_moments():
    quadrature = _d2q9_quadrature()
    plan = PressureExtendedParticleEquilibriumPlan(quadrature)
    density = jnp.asarray((1.2, 0.85))
    velocity = jnp.asarray(((0.06, -0.03), (-0.04, 0.05)))
    pressure = jnp.asarray((0.54, 0.46))

    result = plan.evaluate(density, velocity, pressure)

    assert bool(jnp.all(result.successful))
    target_stress = density[..., None, None] * velocity[..., :, None] * velocity[
        ..., None, :
    ] + pressure[..., None, None] * jnp.eye(2)
    np.testing.assert_allclose(
        jnp.sum(result.populations, axis=-1), density, rtol=2.0e-13, atol=2.0e-13
    )
    np.testing.assert_allclose(
        jnp.einsum("...q,qd->...d", result.populations, quadrature.velocities),
        density[..., None] * velocity,
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    np.testing.assert_allclose(
        jnp.einsum(
            "...q,qa,qb->...ab",
            result.populations,
            quadrature.velocities,
            quadrature.velocities,
        ),
        target_stress,
        rtol=4.0e-13,
        atol=4.0e-13,
    )
    np.testing.assert_allclose(
        result.evidence.particle_stress_residual, 0.0, atol=4.0e-13
    )
    assert bool(jnp.all(result.evidence.minimum_population > 0.0))


def test_pressure_extension_reference_state_has_zero_moment_correction():
    quadrature = _d2q9_quadrature()
    plan = PressureExtendedParticleEquilibriumPlan(quadrature)

    result = plan.evaluate(
        jnp.asarray(1.0),
        jnp.zeros((2,)),
        jnp.asarray(quadrature.reference_temperature),
    )

    assert bool(result.successful)
    np.testing.assert_allclose(
        result.populations, quadrature.weights, rtol=2.0e-13, atol=2.0e-13
    )
    np.testing.assert_allclose(result.evidence.raw_moment_residual, 0.0, atol=2.0e-13)
    np.testing.assert_allclose(result.evidence.invariant_correction, 0.0, atol=2.0e-13)
    np.testing.assert_allclose(result.evidence.stress_correction, 0.0, atol=2.0e-13)
    refused = plan.evaluate(jnp.asarray(1.0), jnp.zeros((2,)), jnp.asarray(1.0))
    assert int(refused.status) == int(
        LearnedThermalResearchStatus.OUTSIDE_DECLARED_SUPPORT
    )
    np.testing.assert_array_equal(
        refused.populations, jnp.zeros_like(refused.populations)
    )


def test_positive_learned_family_preserves_native_energy_and_supplied_moments():
    quadrature = _d2q9_quadrature()
    statistics = _thermal_statistics(quadrature)
    parameters = jnp.asarray((0.3, 0.12, -0.07, -0.04, 0.025))
    plan = PositiveLearnedThermalEnergyPlan(quadrature, statistics.shape[-1])
    total_energy = jnp.asarray((1.75, 3.25))

    result = plan.evaluate(total_energy, statistics, parameters)

    assert bool(jnp.all(result.successful))
    assert bool(jnp.all(result.populations > 0.0))
    np.testing.assert_allclose(
        jnp.sum(result.populations, axis=-1),
        total_energy,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(result.evidence.total_energy_residual, 0.0, atol=2.0e-13)
    np.testing.assert_allclose(
        result.evidence.recovered_sufficient_moments,
        jnp.einsum("...q,qk->...k", result.populations, statistics),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert plan.plan_id == result.plan_id == result.evidence.plan_id


def test_positive_learned_family_differentiates_trunk_values_and_natural_parameters():
    quadrature = _d2q9_quadrature()
    plan = PositiveLearnedThermalEnergyPlan(quadrature, 5)
    statistics = _thermal_statistics(quadrature)
    parameters = jnp.asarray((0.0, 0.09, -0.06, -0.03, 0.02))
    probe = quadrature.velocities[:, 0] + 0.3 * quadrature.velocities[:, 1] ** 2

    def observable(statistic_values, natural_values):
        populations = plan.evaluate(
            jnp.asarray(2.2), statistic_values, natural_values
        ).populations
        return jnp.sum(populations * probe)

    statistic_gradient, natural_gradient = jax.grad(observable, argnums=(0, 1))(
        statistics, parameters
    )

    assert bool(jnp.all(jnp.isfinite(statistic_gradient)))
    assert bool(jnp.all(jnp.isfinite(natural_gradient)))
    assert float(jnp.linalg.norm(statistic_gradient[:, 1:])) > 1.0e-6
    assert float(jnp.linalg.norm(natural_gradient[1:])) > 1.0e-6


def test_learned_family_refuses_values_outside_its_declared_support():
    quadrature = _d2q9_quadrature()
    statistics = _thermal_statistics(quadrature)
    plan = PositiveLearnedThermalEnergyPlan(
        quadrature,
        5,
        maximum_absolute_natural_parameter=0.25,
        maximum_logit_span=4.0,
    )

    result = plan.evaluate(
        jnp.asarray(2.0),
        statistics,
        jnp.asarray((0.0, 0.3, 0.0, 0.0, 0.0)),
    )

    assert int(result.status) == int(
        LearnedThermalResearchStatus.OUTSIDE_DECLARED_SUPPORT
    )
    assert not bool(result.successful)
    assert not bool(result.evidence.within_declared_support)
    np.testing.assert_array_equal(result.populations, jnp.zeros_like(result.populations))


def test_g_star_has_zero_limit_and_nonzero_stress_correction_without_energy_defect():
    quadrature = _d2q9_quadrature()
    particle_plan = PressureExtendedParticleEquilibriumPlan(quadrature)
    thermal_plan = MatchedThermalCrossRelaxationPlan(quadrature)
    velocity = jnp.asarray((0.08, -0.04))
    particle = particle_plan.evaluate(jnp.asarray(1.0), velocity, jnp.asarray(0.5))
    energy = _energy_equilibrium(quadrature)

    zero = thermal_plan.quasi_equilibrium(
        energy.populations,
        particle.populations,
        particle.populations,
        velocity,
        jnp.asarray(0.5),
    )
    anisotropic_perturbation = (
        0.02
        * quadrature.weights
        * (quadrature.velocities[:, 0] ** 2 - quadrature.velocities[:, 1] ** 2)
    )
    nonequilibrium_particles = particle.populations + anisotropic_perturbation
    corrected = thermal_plan.quasi_equilibrium(
        energy.populations,
        nonequilibrium_particles,
        particle.populations,
        velocity,
        jnp.asarray(0.5),
    )

    assert bool(zero.successful)
    np.testing.assert_array_equal(
        zero.raw_correction, jnp.zeros_like(zero.raw_correction)
    )
    np.testing.assert_array_equal(zero.correction, jnp.zeros_like(zero.correction))
    assert bool(corrected.successful)
    assert float(jnp.linalg.norm(corrected.correction)) > 1.0e-8
    np.testing.assert_allclose(
        jnp.sum(corrected.populations),
        jnp.sum(energy.populations),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        corrected.evidence.total_energy_residual, 0.0, atol=2.0e-13
    )
    np.testing.assert_allclose(
        corrected.evidence.correction_energy_residual, 0.0, atol=2.0e-13
    )
    np.testing.assert_allclose(
        corrected.evidence.weighted_velocity_sum, 0.0, atol=2.0e-13
    )


def test_prandtl_cross_relaxation_conserves_g_star_energy_and_has_unit_pr_zero_limit():
    quadrature = _d2q9_quadrature()
    particle_plan = PressureExtendedParticleEquilibriumPlan(quadrature)
    thermal_plan = MatchedThermalCrossRelaxationPlan(quadrature)
    velocity = jnp.asarray((0.07, -0.025))
    particle = particle_plan.evaluate(jnp.asarray(1.0), velocity, jnp.asarray(0.5))
    energy = _energy_equilibrium(quadrature, total_energy=2.8)
    anisotropy = (
        0.015
        * quadrature.weights
        * (quadrature.velocities[:, 0] ** 2 - quadrature.velocities[:, 1] ** 2)
    )
    g_star = thermal_plan.quasi_equilibrium(
        energy.populations,
        particle.populations + anisotropy,
        particle.populations,
        velocity,
        jnp.asarray(0.5),
    )
    current = (
        energy.populations + 0.002 * quadrature.weights * quadrature.velocities[:, 0]
    )

    result = thermal_plan.cross_relax(
        current,
        energy.populations,
        g_star.populations,
        jnp.asarray(0.8),
        jnp.asarray(0.71),
    )
    unit_prandtl = thermal_plan.cross_relax(
        current,
        energy.populations,
        g_star.populations,
        jnp.asarray(0.8),
        jnp.asarray(1.0),
    )

    assert bool(result.successful)
    np.testing.assert_allclose(
        result.evidence.effective_prandtl_number, 0.71, rtol=2.0e-13, atol=2.0e-13
    )
    np.testing.assert_allclose(
        jnp.sum(result.populations), jnp.sum(current), rtol=3.0e-13, atol=3.0e-13
    )
    np.testing.assert_allclose(result.evidence.conservation_residual, 0.0, atol=3.0e-13)
    assert bool(unit_prandtl.successful)
    np.testing.assert_allclose(
        unit_prandtl.cross_relaxation_increment,
        0.0,
        atol=1.0e-15,
    )


def test_integer_frame_shift_roundtrips_and_obeys_raw_moment_transformations():
    quadrature = _d2q9_quadrature()
    particle_plan = PressureExtendedParticleEquilibriumPlan(quadrature)
    relative_velocity = jnp.asarray((0.09, -0.04))
    particles = particle_plan.evaluate(
        jnp.asarray(1.1), relative_velocity, jnp.asarray(0.55)
    ).populations
    energy = _energy_equilibrium(quadrature, total_energy=2.6).populations
    shift = jnp.asarray((1, -1))
    plan = IntegerVelocityFrameShiftPlan(quadrature, shift)

    result = plan.forward(particles, energy)
    source = result.evidence.source_moments
    target = result.moments
    roundtrip = plan.inverse(target)

    assert bool(result.successful)
    np.testing.assert_allclose(
        target.particle_momentum,
        source.particle_momentum + source.density * shift,
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    expected_stress = (
        source.particle_stress
        + shift[:, None] * source.particle_momentum[None, :]
        + source.particle_momentum[:, None] * shift[None, :]
        + source.density * shift[:, None] * shift[None, :]
    )
    np.testing.assert_allclose(
        target.particle_stress, expected_stress, rtol=4.0e-13, atol=4.0e-13
    )
    np.testing.assert_allclose(target.total_energy, source.total_energy, atol=2.0e-13)
    np.testing.assert_allclose(
        target.total_energy_flux,
        source.total_energy_flux + source.total_energy * shift,
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    for field in (
        "density",
        "particle_momentum",
        "particle_stress",
        "total_energy",
        "total_energy_flux",
    ):
        np.testing.assert_allclose(
            getattr(roundtrip, field), getattr(source, field), rtol=5.0e-13, atol=5.0e-13
        )
    np.testing.assert_allclose(
        result.evidence.maximum_identity_residual, 0.0, atol=5.0e-13
    )


def test_integer_frame_plan_reports_hull_refusal_and_rejects_unsupported_rules():
    quadrature = _d2q9_quadrature()
    plan = IntegerVelocityFrameShiftPlan(quadrature, jnp.asarray((1, 0)))

    outside = plan.assess(jnp.asarray((3.1, 0.0)))

    assert not bool(outside.admissible)
    assert int(outside.status) == int(
        LearnedThermalResearchStatus.INADMISSIBLE_FRAME_SHIFT
    )
    with pytest.raises(ValueError, match="components must be integers"):
        IntegerVelocityFrameShiftPlan(quadrature, jnp.asarray((0.5, 0.0)))
    with pytest.raises(ValueError, match="integer-lattice"):
        IntegerVelocityFrameShiftPlan(d2v37_off_lattice_quadrature(), jnp.asarray((1, 0)))
    with pytest.raises(ValueError, match="D2Q9"):
        PressureExtendedParticleEquilibriumPlan(d2v17_quadrature())
    with pytest.raises(ValueError, match="D2Q9"):
        MatchedThermalCrossRelaxationPlan(d2v17_quadrature())
