#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import TestCase

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks._io import write_json_atomic
from phydrax._fingerprint import canonical_fingerprint
from phydrax.discretization.discrete_velocity._learned_thermal_research import (
    IntegerVelocityFrameShiftPlan,
    LearnedThermalResearchStatus,
    MatchedThermalCrossRelaxationPlan,
    PositiveLearnedThermalEnergyPlan,
    PressureExtendedParticleEquilibriumPlan,
    ThermalFrameMoments,
)
from phydrax.discretization.discrete_velocity._quadrature import (
    CertifiedDiscreteVelocityQuadrature,
    d2v17_quadrature,
    d2v37_off_lattice_quadrature,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPOSITORY_ROOT / "benchmarks/learned_thermal_kinetic_shocks.json"

THRESHOLDS = {
    "maximum_moment_residual": 5.0e-13,
    "maximum_native_energy_residual": 5.0e-13,
    "minimum_gradient_norm": 1.0e-8,
    "maximum_gradient_energy_identity_residual": 5.0e-13,
    "maximum_quasi_energy_residual": 5.0e-13,
    "maximum_cross_relaxation_energy_residual": 8.0e-13,
    "maximum_effective_prandtl_residual": 5.0e-13,
    "maximum_frame_identity_residual": 8.0e-13,
    "maximum_frame_roundtrip_residual": 8.0e-13,
}


def _d2q9_quadrature() -> CertifiedDiscreteVelocityQuadrature:
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
        "D2Q9-learned-thermal-kinetic-qualification",
        velocities,
        np.prod(directional_weights, axis=-1),
        reference_temperature=1.0 / 3.0,
        certified_degree=4,
        transport_kind="integer_lattice",
    )


def _thermal_statistics(quadrature: CertifiedDiscreteVelocityQuadrature, /) -> jax.Array:
    velocities = quadrature.velocities
    cx = velocities[:, 0]
    cy = velocities[:, 1]
    radius_squared = cx * cx + cy * cy
    return jnp.stack(
        (
            jnp.ones_like(cx),
            cx,
            cy,
            radius_squared,
            cx * cy,
            cx * cx - cy * cy,
            cx * radius_squared,
            cy * radius_squared,
            cx * cx * cy * cy,
        ),
        axis=-1,
    )


def _vector(values: jax.Array, /) -> list[float]:
    return [float(value) for value in np.asarray(values).reshape(-1)]


def _matrix(values: jax.Array, /) -> list[list[float]]:
    return [
        [float(value) for value in row]
        for row in np.asarray(values).reshape((-1, values.shape[-1]))
    ]


def _maximum_absolute(*values: jax.Array) -> float:
    return max(float(jnp.max(jnp.abs(value))) for value in values)


def _all_successful(status: jax.Array, /) -> bool:
    return bool(jnp.all(status == int(LearnedThermalResearchStatus.SUCCESS)))


def _frame_roundtrip_residual(
    source: ThermalFrameMoments, roundtrip: ThermalFrameMoments, /
) -> float:
    return _maximum_absolute(
        roundtrip.density - source.density,
        roundtrip.particle_momentum - source.particle_momentum,
        roundtrip.particle_stress - source.particle_stress,
        roundtrip.total_energy - source.total_energy,
        roundtrip.total_energy_flux - source.total_energy_flux,
    )


def _qualification() -> dict[str, Any]:
    quadrature = _d2q9_quadrature()
    particle_plan = PressureExtendedParticleEquilibriumPlan(quadrature)
    thermal_plan = MatchedThermalCrossRelaxationPlan(quadrature)

    density = jnp.asarray((1.20, 0.85, 1.05, 0.95), dtype=jnp.float64)
    velocity = jnp.asarray(
        ((0.060, -0.030), (-0.040, 0.050), (0.075, 0.025), (-0.055, -0.035)),
        dtype=jnp.float64,
    )
    pressure = jnp.asarray((0.54, 0.36, 0.42, 0.48), dtype=jnp.float64)
    temperature = pressure / density
    total_energy = jnp.asarray((2.20, 2.65, 3.10, 2.45), dtype=jnp.float64)

    particle = particle_plan.evaluate(density, velocity, pressure)
    independently_recovered_density = jnp.sum(particle.populations, axis=-1)
    independently_recovered_momentum = jnp.einsum(
        "...q,qd->...d", particle.populations, quadrature.velocities
    )
    independently_recovered_stress = jnp.einsum(
        "...q,qa,qb->...ab",
        particle.populations,
        quadrature.velocities,
        quadrature.velocities,
    )
    target_stress = density[..., None, None] * velocity[..., :, None] * velocity[
        ..., None, :
    ] + pressure[..., None, None] * jnp.eye(2, dtype=jnp.float64)
    particle_moment_residual = _maximum_absolute(
        independently_recovered_density - density,
        independently_recovered_momentum - density[..., None] * velocity,
        independently_recovered_stress - target_stress,
        particle.evidence.maximum_absolute_moment_residual,
    )

    statistics = _thermal_statistics(quadrature)
    natural_parameters = jnp.asarray(
        (0.20, 0.11, -0.07, -0.06, 0.025, 0.018, -0.012, 0.009, -0.015),
        dtype=jnp.float64,
    )
    energy_plan = PositiveLearnedThermalEnergyPlan(
        quadrature,
        statistics.shape[-1],
        maximum_absolute_natural_parameter=1.0,
        maximum_logit_span=8.0,
    )
    learned_energy = energy_plan.evaluate(total_energy, statistics, natural_parameters)
    independent_energy_residual = (
        jnp.sum(learned_energy.populations, axis=-1) - total_energy
    )
    sufficient_moment_residual = (
        learned_energy.evidence.recovered_sufficient_moments
        - jnp.einsum("...q,qk->...k", learned_energy.populations, statistics)
    )

    probe = (
        quadrature.velocities[:, 0]
        + 0.31 * quadrature.velocities[:, 1] ** 2
        - 0.17 * quadrature.velocities[:, 0] * quadrature.velocities[:, 1]
    )

    def learned_observable(
        statistic_values: jax.Array, parameter_values: jax.Array
    ) -> jax.Array:
        populations = energy_plan.evaluate(
            jnp.asarray(2.35, dtype=jnp.float64),
            statistic_values,
            parameter_values,
        ).populations
        return jnp.sum(populations * probe)

    statistic_gradient, parameter_gradient = jax.grad(learned_observable, argnums=(0, 1))(
        statistics, natural_parameters
    )

    def native_energy_sum(energy: jax.Array) -> jax.Array:
        return jnp.sum(
            energy_plan.evaluate(energy, statistics, natural_parameters).populations
        )

    native_energy_gradient = jax.grad(native_energy_sum)(
        jnp.asarray(2.35, dtype=jnp.float64)
    )
    statistic_gradient_norm = float(jnp.linalg.norm(statistic_gradient[:, 1:]))
    parameter_gradient_norm = float(jnp.linalg.norm(parameter_gradient[1:]))
    gradients_finite = bool(
        jnp.all(jnp.isfinite(statistic_gradient))
        & jnp.all(jnp.isfinite(parameter_gradient))
        & jnp.isfinite(native_energy_gradient)
    )

    zero_quasi = thermal_plan.quasi_equilibrium(
        learned_energy.populations,
        particle.populations,
        particle.populations,
        velocity,
        temperature,
    )
    anisotropy_basis = quadrature.weights * (
        quadrature.velocities[:, 0] ** 2 - quadrature.velocities[:, 1] ** 2
    )
    anisotropy_scale = jnp.asarray((0.020, 0.014, 0.018, 0.016), dtype=jnp.float64)
    nonequilibrium_particles = (
        particle.populations + anisotropy_scale[:, None] * anisotropy_basis
    )
    corrected_quasi = thermal_plan.quasi_equilibrium(
        learned_energy.populations,
        nonequilibrium_particles,
        particle.populations,
        velocity,
        temperature,
    )
    zero_correction_norm = float(jnp.linalg.norm(zero_quasi.correction))
    zero_stress_defect_norm = float(jnp.linalg.norm(zero_quasi.evidence.stress_defect))
    corrected_stress_defect_norms = jnp.linalg.norm(
        corrected_quasi.evidence.stress_defect, axis=(-2, -1)
    )
    corrected_correction_norms = jnp.linalg.norm(corrected_quasi.correction, axis=-1)
    quasi_energy_residual = _maximum_absolute(
        corrected_quasi.evidence.total_energy_residual,
        corrected_quasi.evidence.correction_energy_residual,
        jnp.sum(corrected_quasi.populations, axis=-1) - total_energy,
    )

    current_energy = learned_energy.populations + 1.5e-3 * quadrature.weights * (
        quadrature.velocities[:, 0] + 0.25 * quadrature.velocities[:, 1]
    )
    particle_relaxation_time = jnp.asarray((0.80, 0.92, 0.74, 0.86), dtype=jnp.float64)
    prescribed_prandtl = jnp.asarray((0.71, 1.20, 0.83, 1.00), dtype=jnp.float64)
    cross_relaxation = thermal_plan.cross_relax(
        current_energy,
        learned_energy.populations,
        corrected_quasi.populations,
        particle_relaxation_time,
        prescribed_prandtl,
    )
    cross_energy_residual = _maximum_absolute(
        cross_relaxation.evidence.conservation_residual,
        jnp.sum(cross_relaxation.populations, axis=-1) - jnp.sum(current_energy, axis=-1),
    )
    effective_prandtl_residual = float(
        jnp.max(
            jnp.abs(
                cross_relaxation.evidence.effective_prandtl_number - prescribed_prandtl
            )
        )
    )
    unit_prandtl_cross_increment = float(
        jnp.max(jnp.abs(cross_relaxation.cross_relaxation_increment[-1]))
    )

    frame_shift = jnp.asarray((1.0, -1.0), dtype=jnp.float64)
    frame_plan = IntegerVelocityFrameShiftPlan(quadrature, frame_shift)
    frame = frame_plan.forward(particle.populations, learned_energy.populations)
    source_moments = frame.evidence.source_moments
    shifted_moments = frame.moments
    expected_shifted_momentum = (
        source_moments.particle_momentum + source_moments.density[..., None] * frame_shift
    )
    expected_shifted_stress = (
        source_moments.particle_stress
        + frame_shift[None, :, None] * source_moments.particle_momentum[:, None, :]
        + source_moments.particle_momentum[:, :, None] * frame_shift[None, None, :]
        + source_moments.density[:, None, None]
        * frame_shift[None, :, None]
        * frame_shift[None, None, :]
    )
    expected_shifted_energy_flux = (
        source_moments.total_energy_flux
        + source_moments.total_energy[..., None] * frame_shift
    )
    frame_identity_residual = _maximum_absolute(
        shifted_moments.density - source_moments.density,
        shifted_moments.particle_momentum - expected_shifted_momentum,
        shifted_moments.particle_stress - expected_shifted_stress,
        shifted_moments.total_energy - source_moments.total_energy,
        shifted_moments.total_energy_flux - expected_shifted_energy_flux,
        frame.evidence.maximum_identity_residual,
    )
    frame_roundtrip = frame_plan.inverse(shifted_moments)
    frame_roundtrip_residual = _frame_roundtrip_residual(source_moments, frame_roundtrip)

    outside_particle = particle_plan.evaluate(
        jnp.asarray(1.0, dtype=jnp.float64),
        jnp.zeros((2,), dtype=jnp.float64),
        jnp.asarray(1.0, dtype=jnp.float64),
    )
    restrictive_energy_plan = PositiveLearnedThermalEnergyPlan(
        quadrature,
        statistics.shape[-1],
        maximum_absolute_natural_parameter=0.25,
        maximum_logit_span=4.0,
    )
    outside_energy = restrictive_energy_plan.evaluate(
        jnp.asarray(2.0, dtype=jnp.float64),
        statistics,
        natural_parameters.at[1].set(0.30),
    )
    outside_quasi = thermal_plan.quasi_equilibrium(
        learned_energy.populations[0],
        particle.populations[0],
        particle.populations[0],
        velocity[0],
        jnp.asarray(1.0, dtype=jnp.float64),
    )
    outside_frame = frame_plan.assess(jnp.asarray((3.1, -1.0), dtype=jnp.float64))

    exception_assertions = TestCase()
    with exception_assertions.assertRaisesRegex(
        ValueError, "components must be integers"
    ):
        IntegerVelocityFrameShiftPlan(
            quadrature, jnp.asarray((0.5, 0.0), dtype=jnp.float64)
        )
    fractional_shift_refused = True
    with exception_assertions.assertRaisesRegex(ValueError, "integer-lattice"):
        IntegerVelocityFrameShiftPlan(
            d2v37_off_lattice_quadrature(dtype=jnp.float64),
            jnp.asarray((1.0, 0.0), dtype=jnp.float64),
        )
    off_lattice_frame_refused = True
    with exception_assertions.assertRaisesRegex(ValueError, "D2Q9"):
        PressureExtendedParticleEquilibriumPlan(d2v17_quadrature(dtype=jnp.float64))
    non_d2q9_particle_refused = True
    with exception_assertions.assertRaisesRegex(ValueError, "D2Q9"):
        MatchedThermalCrossRelaxationPlan(d2v17_quadrature(dtype=jnp.float64))
    non_d2q9_cross_relaxation_refused = True

    refusals = {
        "pressure_outside_support": {
            "status": int(outside_particle.status),
            "expected_status": int(LearnedThermalResearchStatus.OUTSIDE_DECLARED_SUPPORT),
            "populations_exactly_zero": bool(
                jnp.array_equal(
                    outside_particle.populations,
                    jnp.zeros_like(outside_particle.populations),
                )
            ),
        },
        "learned_energy_outside_support": {
            "status": int(outside_energy.status),
            "expected_status": int(LearnedThermalResearchStatus.OUTSIDE_DECLARED_SUPPORT),
            "populations_exactly_zero": bool(
                jnp.array_equal(
                    outside_energy.populations,
                    jnp.zeros_like(outside_energy.populations),
                )
            ),
        },
        "quasi_equilibrium_outside_temperature_support": {
            "status": int(outside_quasi.status),
            "expected_status": int(LearnedThermalResearchStatus.OUTSIDE_DECLARED_SUPPORT),
            "populations_exactly_zero": bool(
                jnp.array_equal(
                    outside_quasi.populations,
                    jnp.zeros_like(outside_quasi.populations),
                )
            ),
        },
        "frame_outside_convex_support": {
            "status": int(outside_frame.status),
            "expected_status": int(LearnedThermalResearchStatus.INADMISSIBLE_FRAME_SHIFT),
            "admissible": bool(outside_frame.admissible),
        },
        "fractional_frame_shift_rejected_at_construction": (fractional_shift_refused),
        "off_lattice_frame_rule_rejected_at_construction": (off_lattice_frame_refused),
        "non_d2q9_particle_rule_rejected_at_construction": (non_d2q9_particle_refused),
        "non_d2q9_cross_relaxation_rule_rejected_at_construction": (
            non_d2q9_cross_relaxation_refused
        ),
    }

    gates = {
        "pressure_extended_particle_mass_momentum_stress": bool(
            _all_successful(particle.status)
            and bool(jnp.all(particle.populations > 0.0))
            and particle_moment_residual <= THRESHOLDS["maximum_moment_residual"]
        ),
        "richer_positive_learned_energy_exact_native_moment": bool(
            statistics.shape[-1] == 9
            and _all_successful(learned_energy.status)
            and bool(jnp.all(learned_energy.populations > 0.0))
            and _maximum_absolute(
                independent_energy_residual,
                learned_energy.evidence.total_energy_residual,
                sufficient_moment_residual,
            )
            <= THRESHOLDS["maximum_native_energy_residual"]
        ),
        "learned_energy_gradients_are_real_and_nontrivial": bool(
            gradients_finite
            and statistic_gradient_norm >= THRESHOLDS["minimum_gradient_norm"]
            and parameter_gradient_norm >= THRESHOLDS["minimum_gradient_norm"]
            and abs(float(native_energy_gradient) - 1.0)
            <= THRESHOLDS["maximum_gradient_energy_identity_residual"]
        ),
        "g_star_zero_stress_limit": bool(
            _all_successful(zero_quasi.status)
            and zero_stress_defect_norm == 0.0
            and zero_correction_norm == 0.0
            and bool(jnp.array_equal(zero_quasi.populations, learned_energy.populations))
        ),
        "g_star_nonzero_stress_correction_conserves_native_energy": bool(
            _all_successful(corrected_quasi.status)
            and bool(
                jnp.all(
                    corrected_stress_defect_norms >= THRESHOLDS["minimum_gradient_norm"]
                )
            )
            and bool(
                jnp.all(corrected_correction_norms >= THRESHOLDS["minimum_gradient_norm"])
            )
            and quasi_energy_residual <= THRESHOLDS["maximum_quasi_energy_residual"]
        ),
        "independently_prescribed_prandtl_cross_relaxation": bool(
            _all_successful(cross_relaxation.status)
            and len(set(_vector(prescribed_prandtl))) == prescribed_prandtl.size
            and bool(jnp.any(prescribed_prandtl != 1.0))
            and effective_prandtl_residual
            <= THRESHOLDS["maximum_effective_prandtl_residual"]
            and unit_prandtl_cross_increment
            <= THRESHOLDS["maximum_native_energy_residual"]
        ),
        "cross_relaxation_conserves_native_energy": bool(
            cross_energy_residual
            <= THRESHOLDS["maximum_cross_relaxation_energy_residual"]
            and bool(jnp.all(cross_relaxation.populations > 0.0))
        ),
        "integer_frame_shift_moment_transformation": bool(
            _all_successful(frame.status)
            and bool(jnp.all(frame.evidence.admissibility.admissible))
            and frame_identity_residual <= THRESHOLDS["maximum_frame_identity_residual"]
        ),
        "integer_frame_shift_roundtrip": bool(
            frame_roundtrip_residual <= THRESHOLDS["maximum_frame_roundtrip_residual"]
        ),
        "outside_support_and_off_lattice_misuse_refused": bool(
            refusals["pressure_outside_support"]["status"]
            == refusals["pressure_outside_support"]["expected_status"]
            and refusals["pressure_outside_support"]["populations_exactly_zero"]
            and refusals["learned_energy_outside_support"]["status"]
            == refusals["learned_energy_outside_support"]["expected_status"]
            and refusals["learned_energy_outside_support"]["populations_exactly_zero"]
            and refusals["quasi_equilibrium_outside_temperature_support"]["status"]
            == refusals["quasi_equilibrium_outside_temperature_support"][
                "expected_status"
            ]
            and refusals["quasi_equilibrium_outside_temperature_support"][
                "populations_exactly_zero"
            ]
            and refusals["frame_outside_convex_support"]["status"]
            == refusals["frame_outside_convex_support"]["expected_status"]
            and not refusals["frame_outside_convex_support"]["admissible"]
            and fractional_shift_refused
            and off_lattice_frame_refused
            and non_d2q9_particle_refused
            and non_d2q9_cross_relaxation_refused
        ),
    }

    report: dict[str, Any] = {
        "tool": "learned_thermal_kinetic_qualification",
        "qualification": "deterministic isolated learned-thermal kinetic ingredients",
        "scope": {
            "research_only": True,
            "included": [
                "local pressure-extended D2Q9 particle equilibrium",
                "local positive supplied learned thermal exponential family",
                "local stress-coupled thermal quasi-equilibrium",
                "local independently prescribed-Pr cross relaxation",
                "local integer velocity-frame moment translation",
                "explicit support and quadrature misuse refusal",
            ],
            "excluded": [
                "spatial transport",
                "boundary conditions",
                "TVD regularization",
                "complete shock runs",
                "complete cylinder runs",
                "shock ownership or shock-solver claims",
                "production readiness",
            ],
            "claim": "isolated research ingredients only; not a spatial shock solver",
        },
        "provenance": {
            "deterministic": True,
            "external_code": False,
            "external_data": False,
            "external_weights": False,
            "case_count": density.size,
            "case_order": [
                "positive-x-negative-y",
                "negative-x-positive-y",
                "positive-x-positive-y",
                "negative-x-negative-y",
            ],
        },
        "contracts": {
            "quadrature": "certified tensor-product D2Q9 integer lattice",
            "particle_moments": [
                "sum(f) = rho",
                "sum(c f) = rho u",
                "sum(c c f) = rho u u + p I",
            ],
            "native_energy": "sum(g) = total energy",
            "learned_family": (
                "caller-supplied differentiable sufficient statistics and natural parameters"
            ),
            "cross_relaxation": ("tau_2 = 1/2 + (tau_1 - 1/2) / prescribed_Pr"),
            "frame_shift": (
                "unchanged populations with exact raw-moment translation; no remap or streaming"
            ),
        },
        "identities": {
            "quadrature": quadrature.quadrature_id,
            "particle_plan": particle_plan.plan_id,
            "learned_energy_plan": energy_plan.plan_id,
            "cross_relaxation_plan": thermal_plan.plan_id,
            "frame_shift_plan": frame_plan.plan_id,
        },
        "thresholds": THRESHOLDS,
        "states": {
            "density": _vector(density),
            "velocity": _matrix(velocity),
            "pressure": _vector(pressure),
            "temperature": _vector(temperature),
            "total_energy": _vector(total_energy),
        },
        "pressure_extended_particle": {
            "all_successful": _all_successful(particle.status),
            "all_populations_positive": bool(jnp.all(particle.populations > 0.0)),
            "minimum_population": float(jnp.min(particle.populations)),
            "minimum_extended_support_margin": float(
                jnp.min(particle.evidence.extended_support_margin)
            ),
            "maximum_mass_momentum_stress_residual": particle_moment_residual,
        },
        "learned_thermal_energy": {
            "statistic_count": statistics.shape[-1],
            "natural_parameters": _vector(natural_parameters),
            "all_successful": _all_successful(learned_energy.status),
            "all_populations_positive": bool(jnp.all(learned_energy.populations > 0.0)),
            "minimum_population": float(jnp.min(learned_energy.populations)),
            "maximum_native_energy_residual": _maximum_absolute(
                independent_energy_residual,
                learned_energy.evidence.total_energy_residual,
            ),
            "maximum_sufficient_moment_residual": _maximum_absolute(
                sufficient_moment_residual
            ),
            "gradients": {
                "finite": gradients_finite,
                "nonconstant_statistic_gradient_norm": statistic_gradient_norm,
                "nonconstant_parameter_gradient_norm": parameter_gradient_norm,
                "native_energy_derivative": float(native_energy_gradient),
            },
        },
        "thermal_quasi_equilibrium": {
            "zero_stress_defect": {
                "all_successful": _all_successful(zero_quasi.status),
                "correction_norm": zero_correction_norm,
                "stress_defect_norm": zero_stress_defect_norm,
                "populations_match_equilibrium_exactly": bool(
                    jnp.array_equal(zero_quasi.populations, learned_energy.populations)
                ),
            },
            "nonzero_stress_defect": {
                "all_successful": _all_successful(corrected_quasi.status),
                "stress_defect_norms": _vector(corrected_stress_defect_norms),
                "correction_norms": _vector(corrected_correction_norms),
                "maximum_native_energy_residual": quasi_energy_residual,
                "minimum_population": float(jnp.min(corrected_quasi.populations)),
            },
        },
        "cross_relaxation": {
            "all_successful": _all_successful(cross_relaxation.status),
            "particle_relaxation_time": _vector(particle_relaxation_time),
            "prescribed_prandtl": _vector(prescribed_prandtl),
            "effective_prandtl": _vector(
                cross_relaxation.evidence.effective_prandtl_number
            ),
            "maximum_effective_prandtl_residual": effective_prandtl_residual,
            "maximum_native_energy_conservation_residual": cross_energy_residual,
            "unit_prandtl_cross_increment_maximum_absolute": (
                unit_prandtl_cross_increment
            ),
            "minimum_population": float(jnp.min(cross_relaxation.populations)),
        },
        "integer_frame_shift": {
            "shift": _vector(frame_shift),
            "all_successful": _all_successful(frame.status),
            "all_relative_velocities_inside_support": bool(
                jnp.all(frame.evidence.admissibility.admissible)
            ),
            "minimum_interior_margin": float(
                jnp.min(frame.evidence.admissibility.interior_margin)
            ),
            "maximum_forward_identity_residual": frame_identity_residual,
            "maximum_inverse_roundtrip_residual": frame_roundtrip_residual,
        },
        "refusals": refusals,
        "gates": gates,
    }
    report["passed"] = all(gates.values())
    report["report_id"] = canonical_fingerprint(report)
    return report


def main() -> int:
    with jax.enable_x64(True):
        report = _qualification()
    if not report["passed"]:
        return 1
    write_json_atomic(REPORT_PATH, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
