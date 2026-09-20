#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax.equations as equations
from benchmarks._io import write_json_atomic
from phydrax._fingerprint import canonical_fingerprint
from phydrax.closure_data._kinetic_equilibrium_artifact import (
    LearnedEnergyEquilibriumArtifact,
    read_learned_energy_equilibrium_artifact,
)
from phydrax.discretization.discrete_velocity._smooth_compressible import (
    SmoothCompressibleD2VKineticMethod,
    SmoothCompressibleKineticState,
)
from phydrax.discretization.discrete_velocity._spatial import (
    PreparedSmoothCompressibleD2V17SpatialDynamics,
    SmoothCompressibleD2V17SpatialPlan,
)
from phydrax.discretization.discrete_velocity._spatial_forcing import (
    SmoothCompressibleD2VBodyForcingPlan,
    SmoothCompressibleD2VForcingResult,
    SmoothCompressibleD2VForcingStatus,
    ZeroSmoothCompressibleD2VForcingPlan,
)


SPATIAL_SHAPE = (16, 16)
DOMAIN_EXTENT = (1.0, 1.0)
CELL_SPACING = tuple(
    extent / count for extent, count in zip(DOMAIN_EXTENT, SPATIAL_SHAPE, strict=True)
)
TIME_STEP = CELL_SPACING[0]
PARTICLE_RELAXATION_TIME = 0.03
TOTAL_ENERGY_RELAXATION_TIME = 0.04
CONSERVATION_TOLERANCE = 1.0e-11

ACCELERATION = (2.0e-3, -1.5e-3)
VOLUMETRIC_HEATING = 2.0e-3
COMBINED_ACCELERATION = (1.25e-3, -7.5e-4)
COMBINED_HEATING = 1.5e-3

THRESHOLDS = {
    "source_maximum_absolute_residual": 5.0e-12,
    "content_ledger_maximum_absolute_residual": 5.0e-11,
    "runtime_conservation_maximum_absolute_residual": 5.0e-11,
}


def _state_equal(
    left: SmoothCompressibleKineticState,
    right: SmoothCompressibleKineticState,
    /,
) -> bool:
    return bool(
        jnp.array_equal(left.particle_populations, right.particle_populations)
        & jnp.array_equal(left.total_energy_populations, right.total_energy_populations)
    )


def _maximum_absolute(values: jax.Array, /) -> float:
    return float(jnp.max(jnp.abs(values)))


def _vector(values: jax.Array, /) -> list[float]:
    return [float(value) for value in np.asarray(values).reshape(-1)]


def _read_parent_spatial(path: Path, /) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("Parent spatial qualification must be a JSON object.")
    required = {"tool", "qualification_id", "identities", "passed"}
    if not required.issubset(report):
        raise ValueError("Parent spatial qualification is missing identity fields.")
    if report["tool"] != "learned_energy_spatial_qualification":
        raise ValueError("Parent report is not the learned-energy spatial qualification.")
    if not isinstance(report["qualification_id"], str) or not report["qualification_id"]:
        raise ValueError("Parent spatial qualification identity is invalid.")
    identities = report["identities"]
    if not isinstance(identities, dict):
        raise ValueError("Parent spatial identities must be a JSON object.")
    required_identities = {
        "artifact",
        "model_numeric_revision",
        "support",
        "spatial_plan",
        "prepared_spatial",
        "quadrature",
        "material",
        "transport_closure",
        "kinetic_method",
        "energy_plan",
    }
    if not required_identities.issubset(identities):
        raise ValueError("Parent spatial qualification has an incomplete identity chain.")
    return report


def _domain_content(
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    state: SmoothCompressibleKineticState,
    /,
) -> jax.Array:
    conserved = runtime.method.moments(state).conserved
    volume = jnp.asarray(
        runtime.transport.cell_volume, dtype=state.particle_populations.dtype
    )
    return jnp.sum(conserved, axis=(0, 1)) * volume


def _transport_state(
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    state: SmoothCompressibleKineticState,
    /,
) -> SmoothCompressibleKineticState:
    return SmoothCompressibleKineticState(
        runtime.transport.transport(state.particle_populations),
        runtime.transport.transport(state.total_energy_populations),
    )


def _deterministic_learned_state(
    artifact: LearnedEnergyEquilibriumArtifact,
    method: SmoothCompressibleD2VKineticMethod,
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    /,
) -> tuple[SmoothCompressibleKineticState, dict[str, Any]]:
    binding = artifact.binding
    plan = binding.plan
    support = plan.support
    model = binding.model.as_trainable()

    x = jnp.arange(SPATIAL_SHAPE[0], dtype=jnp.float64)[:, None]
    y = jnp.arange(SPATIAL_SHAPE[1], dtype=jnp.float64)[None, :]
    phase_x = 2.0 * jnp.pi * x / SPATIAL_SHAPE[0]
    phase_y = 2.0 * jnp.pi * y / SPATIAL_SHAPE[1]

    def interior_field(
        bounds: tuple[float, float], pattern: jax.Array, amplitude: float
    ) -> jax.Array:
        lower, upper = bounds
        midpoint = 0.5 * (lower + upper)
        return midpoint + amplitude * (upper - lower) * pattern

    density = interior_field(
        support.rho_bounds, jnp.sin(phase_x) * jnp.cos(phase_y), 0.10
    )
    velocity_x = interior_field(
        support.u_x_bounds, jnp.cos(phase_x) * jnp.sin(phase_y), 0.08
    )
    velocity_y = interior_field(support.u_y_bounds, jnp.sin(phase_x + phase_y), 0.08)
    temperature = interior_field(
        support.temperature_bounds, jnp.cos(phase_x - phase_y), 0.10
    )
    pressure = density * plan.material.gas_constant * temperature
    specific_internal_energy = plan.material.specific_internal_energy(density, pressure)
    total_energy = density * specific_internal_energy + 0.5 * density * (
        velocity_x**2 + velocity_y**2
    )
    conserved = jnp.stack(
        (
            density,
            density * velocity_x,
            density * velocity_y,
            total_energy,
        ),
        axis=-1,
    )

    dual, support_evidence = plan.predict_dual_with_evidence(model, conserved)
    equilibrium, equilibrium_evidence = method.equilibrium_from_energy_dual_with_evidence(
        conserved, dual, plan.equilibrium_plan
    )
    runtime_result = runtime.step_with_energy_dual(
        equilibrium, jnp.asarray(TIME_STEP, dtype=jnp.float64), dual
    )
    record = {
        "construction": "deterministic interior-support periodic field",
        "explicit_unfrozen_model_evaluation": True,
        "all_inputs_finite": bool(jnp.all(jnp.isfinite(conserved))),
        "all_lanes_inside_declared_support": bool(
            jnp.all(support_evidence.primitive_supported)
        ),
        "all_model_outputs_finite": bool(jnp.all(support_evidence.model_output_finite)),
        "all_support_evaluations_successful": bool(jnp.all(support_evidence.successful)),
        "all_learned_equilibria_successful": bool(
            jnp.all(equilibrium_evidence.successful)
        ),
        "minimum_particle_population": float(jnp.min(equilibrium.particle_populations)),
        "minimum_total_energy_population": float(
            jnp.min(equilibrium.total_energy_populations)
        ),
        "runtime_successful": bool(runtime_result.successful),
        "runtime_rollback_applied": bool(runtime_result.rollback_applied),
        "runtime_maximum_absolute_conservation_residual": float(
            runtime_result.evidence.conservation.maximum_absolute_residual
        ),
    }
    return runtime_result.accepted_state, record


def _zero_source_case(
    plan: ZeroSmoothCompressibleD2VForcingPlan,
    state: SmoothCompressibleKineticState,
    /,
) -> dict[str, Any]:
    result = plan.apply(state, jnp.asarray(TIME_STEP, dtype=jnp.float64))
    zero_particle_increment = jnp.zeros_like(result.particle_population_increment)
    zero_energy_increment = jnp.zeros_like(result.total_energy_population_increment)
    zero_moment_increment = jnp.zeros_like(result.mass_momentum_energy_increment)
    return {
        "plan_id": plan.plan_id,
        "successful": bool(result.successful),
        "status": int(result.evidence.status),
        "rollback_applied": bool(result.rollback_applied),
        "candidate_matches_input_exactly": _state_equal(result.candidate_state, state),
        "accepted_matches_input_exactly": _state_equal(result.accepted_state, state),
        "particle_increment_exactly_zero": bool(
            jnp.array_equal(result.particle_population_increment, zero_particle_increment)
        ),
        "energy_increment_exactly_zero": bool(
            jnp.array_equal(
                result.total_energy_population_increment, zero_energy_increment
            )
        ),
        "accepted_conserved_increment_exactly_zero": bool(
            jnp.array_equal(result.mass_momentum_energy_increment, zero_moment_increment)
        ),
        "maximum_absolute_source_moment_residual": float(
            result.evidence.maximum_absolute_source_moment_residual
        ),
    }


def _uniform_acceleration_case(
    method: SmoothCompressibleD2VKineticMethod,
    plan: SmoothCompressibleD2VBodyForcingPlan,
    state: SmoothCompressibleKineticState,
    /,
) -> dict[str, Any]:
    before = method.moments(state)
    result = plan.apply(state, jnp.asarray(TIME_STEP, dtype=jnp.float64))
    after = method.moments(result.accepted_state)
    actual_increment = after.conserved - before.conserved
    acceleration = jnp.asarray(ACCELERATION, dtype=jnp.float64)
    body_force = before.density[..., None] * acceleration
    momentum_impulse = TIME_STEP * body_force
    midpoint_velocity = (before.momentum + 0.5 * momentum_impulse) / before.density[
        ..., None
    ]
    midpoint_work = TIME_STEP * jnp.sum(midpoint_velocity * body_force, axis=-1)
    return {
        "plan_id": plan.plan_id,
        "acceleration": list(ACCELERATION),
        "successful": bool(result.successful),
        "status": int(result.evidence.status),
        "rollback_applied": bool(result.rollback_applied),
        "target_mass_increment_exactly_zero": bool(
            jnp.array_equal(
                result.evidence.target_source_moments[..., 0],
                jnp.zeros_like(result.evidence.target_source_moments[..., 0]),
            )
        ),
        "maximum_absolute_recovered_mass_increment": _maximum_absolute(
            actual_increment[..., 0]
        ),
        "maximum_absolute_momentum_impulse_residual": _maximum_absolute(
            actual_increment[..., 1:3] - momentum_impulse
        ),
        "maximum_absolute_midpoint_velocity_residual": _maximum_absolute(
            result.evidence.midpoint_velocity - midpoint_velocity
        ),
        "maximum_absolute_midpoint_energy_work_residual": _maximum_absolute(
            actual_increment[..., 3] - midpoint_work
        ),
        "volumetric_heating_increment_exactly_zero": bool(
            jnp.array_equal(
                result.evidence.volumetric_heating_increment,
                jnp.zeros_like(result.evidence.volumetric_heating_increment),
            )
        ),
        "maximum_absolute_source_moment_residual": float(
            result.evidence.maximum_absolute_source_moment_residual
        ),
    }


def _pure_heating_case(
    method: SmoothCompressibleD2VKineticMethod,
    plan: SmoothCompressibleD2VBodyForcingPlan,
    state: SmoothCompressibleKineticState,
    /,
) -> dict[str, Any]:
    before = method.moments(state)
    result = plan.apply(state, jnp.asarray(TIME_STEP, dtype=jnp.float64))
    after = method.moments(result.accepted_state)
    actual_increment = after.conserved - before.conserved
    expected_heating = jnp.full(
        before.density.shape,
        TIME_STEP * VOLUMETRIC_HEATING,
        dtype=jnp.float64,
    )
    return {
        "plan_id": plan.plan_id,
        "volumetric_heating": VOLUMETRIC_HEATING,
        "successful": bool(result.successful),
        "status": int(result.evidence.status),
        "rollback_applied": bool(result.rollback_applied),
        "target_mass_increment_exactly_zero": bool(
            jnp.array_equal(
                result.evidence.target_source_moments[..., 0],
                jnp.zeros_like(result.evidence.target_source_moments[..., 0]),
            )
        ),
        "maximum_absolute_recovered_mass_increment": _maximum_absolute(
            actual_increment[..., 0]
        ),
        "maximum_absolute_recovered_momentum_increment": _maximum_absolute(
            actual_increment[..., 1:3]
        ),
        "acceleration_work_increment_exactly_zero": bool(
            jnp.array_equal(
                result.evidence.acceleration_work_increment,
                jnp.zeros_like(result.evidence.acceleration_work_increment),
            )
        ),
        "maximum_absolute_heating_increment_residual": _maximum_absolute(
            actual_increment[..., 3] - expected_heating
        ),
        "maximum_absolute_source_moment_residual": float(
            result.evidence.maximum_absolute_source_moment_residual
        ),
    }


def _combined_transport_case(
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    plan: SmoothCompressibleD2VBodyForcingPlan,
    state: SmoothCompressibleKineticState,
    /,
) -> dict[str, Any]:
    before_content = _domain_content(runtime, state)
    result = plan.apply(state, jnp.asarray(TIME_STEP, dtype=jnp.float64))
    after_source_content = _domain_content(runtime, result.accepted_state)
    transported = _transport_state(runtime, result.accepted_state)
    after_transport_content = _domain_content(runtime, transported)
    volume = jnp.asarray(
        runtime.transport.cell_volume, dtype=state.particle_populations.dtype
    )
    target_source_content = (
        jnp.sum(result.evidence.target_source_moments, axis=(0, 1)) * volume
    )
    accepted_source_content = (
        jnp.sum(result.mass_momentum_energy_increment, axis=(0, 1)) * volume
    )
    source_application_residual = (
        after_source_content - before_content - target_source_content
    )
    accepted_source_residual = accepted_source_content - target_source_content
    transport_residual = after_transport_content - after_source_content
    complete_ledger_residual = (
        after_transport_content - before_content - target_source_content
    )
    transport_realizability = runtime.method.realizability(transported)
    return {
        "plan_id": plan.plan_id,
        "acceleration": list(COMBINED_ACCELERATION),
        "volumetric_heating": COMBINED_HEATING,
        "successful": bool(result.successful),
        "status": int(result.evidence.status),
        "rollback_applied": bool(result.rollback_applied),
        "periodic_transport_executed": True,
        "target_mass_source_exactly_zero": bool(
            jnp.array_equal(
                result.evidence.target_source_moments[..., 0],
                jnp.zeros_like(result.evidence.target_source_moments[..., 0]),
            )
        ),
        "pre_source_content": _vector(before_content),
        "target_source_content": _vector(target_source_content),
        "post_source_content": _vector(after_source_content),
        "post_transport_content": _vector(after_transport_content),
        "maximum_absolute_source_application_residual": _maximum_absolute(
            source_application_residual
        ),
        "maximum_absolute_accepted_source_residual": _maximum_absolute(
            accepted_source_residual
        ),
        "maximum_absolute_periodic_transport_residual": _maximum_absolute(
            transport_residual
        ),
        "maximum_absolute_complete_content_source_ledger_residual": (
            _maximum_absolute(complete_ledger_residual)
        ),
        "post_transport_realizable": bool(transport_realizability.realizable),
        "post_transport_minimum_particle_population": float(
            transport_realizability.minimum_particle_population.min()
        ),
        "post_transport_minimum_total_energy_population": float(
            transport_realizability.minimum_total_energy_population.min()
        ),
    }


def _refusal_record(
    result: SmoothCompressibleD2VForcingResult,
    state: SmoothCompressibleKineticState,
    expected_status: SmoothCompressibleD2VForcingStatus,
    /,
) -> dict[str, Any]:
    candidate_finite = jnp.all(
        jnp.isfinite(result.candidate_state.particle_populations)
    ) & jnp.all(jnp.isfinite(result.candidate_state.total_energy_populations))
    candidate_negative = jnp.any(
        result.candidate_state.particle_populations < 0.0
    ) | jnp.any(result.candidate_state.total_energy_populations < 0.0)
    zero_accepted_increment = jnp.zeros_like(result.mass_momentum_energy_increment)
    return {
        "expected_status": int(expected_status),
        "status": int(result.evidence.status),
        "successful": bool(result.successful),
        "rollback_applied": bool(result.rollback_applied),
        "candidate_contains_nonfinite_population": bool(~candidate_finite),
        "candidate_contains_negative_population": bool(candidate_negative),
        "accepted_state_matches_input_exactly": _state_equal(
            result.accepted_state, state
        ),
        "accepted_conserved_increment_exactly_zero": bool(
            jnp.array_equal(
                result.mass_momentum_energy_increment, zero_accepted_increment
            )
        ),
    }


def _qualification(
    model_artifact_path: Path,
    parent_spatial_path: Path,
    /,
) -> dict[str, Any]:
    parent = _read_parent_spatial(parent_spatial_path)
    parent_identities = parent["identities"]
    artifact = read_learned_energy_equilibrium_artifact(model_artifact_path)
    binding = artifact.binding
    transport_closure = equations.ConstantTransport(
        PARTICLE_RELAXATION_TIME, TOTAL_ENERGY_RELAXATION_TIME
    )
    method = equations.smooth_compressible_d2v17_method(
        binding.plan.material, transport_closure, dtype=jnp.float64
    )
    spatial_plan = SmoothCompressibleD2V17SpatialPlan(
        method,
        binding.plan.equilibrium_plan,
        SPATIAL_SHAPE,
        CELL_SPACING,
        TIME_STEP,
        conservation_tolerance=CONSERVATION_TOLERANCE,
    )
    runtime = spatial_plan.prepare()
    state, initialization = _deterministic_learned_state(artifact, method, runtime)

    zero_plan = ZeroSmoothCompressibleD2VForcingPlan(method)
    acceleration_plan = SmoothCompressibleD2VBodyForcingPlan(
        method, acceleration=ACCELERATION
    )
    heating_plan = SmoothCompressibleD2VBodyForcingPlan(
        method, volumetric_heating=VOLUMETRIC_HEATING
    )
    combined_plan = SmoothCompressibleD2VBodyForcingPlan(
        method,
        acceleration=COMBINED_ACCELERATION,
        volumetric_heating=COMBINED_HEATING,
    )
    nonfinite_plan = SmoothCompressibleD2VBodyForcingPlan(
        method, acceleration=(np.finfo(np.float64).max, 0.0)
    )
    negative_plan = SmoothCompressibleD2VBodyForcingPlan(
        method, volumetric_heating=-1.0e6
    )

    zero = _zero_source_case(zero_plan, state)
    acceleration = _uniform_acceleration_case(method, acceleration_plan, state)
    heating = _pure_heating_case(method, heating_plan, state)
    combined = _combined_transport_case(runtime, combined_plan, state)
    nonfinite = _refusal_record(
        nonfinite_plan.apply(state, jnp.asarray(TIME_STEP, dtype=jnp.float64)),
        state,
        SmoothCompressibleD2VForcingStatus.NONFINITE_CANDIDATE,
    )
    negative = _refusal_record(
        negative_plan.apply(state, jnp.asarray(TIME_STEP, dtype=jnp.float64)),
        state,
        SmoothCompressibleD2VForcingStatus.NONPOSITIVE_CANDIDATE,
    )

    forcing_plan_ids = (
        zero_plan.plan_id,
        acceleration_plan.plan_id,
        heating_plan.plan_id,
        combined_plan.plan_id,
        nonfinite_plan.plan_id,
        negative_plan.plan_id,
    )
    forcing_id = canonical_fingerprint(
        {
            "kind": "learned-energy-conserved-source-qualification",
            "method": method.method_id,
            "plans": list(forcing_plan_ids),
        }
    )

    source_tolerance = THRESHOLDS["source_maximum_absolute_residual"]
    ledger_tolerance = THRESHOLDS["content_ledger_maximum_absolute_residual"]
    gates = {
        "parent_spatial_qualification": bool(parent["passed"]),
        "parent_identity_chain": bool(
            parent_identities["artifact"] == artifact.artifact_id
            and parent_identities["model_numeric_revision"]
            == binding.numeric_revision.revision_id
            and parent_identities["support"] == binding.plan.support.support_id
            and parent_identities["quadrature"] == method.quadrature.quadrature_id
            and parent_identities["material"] == method.material.material_id
            and parent_identities["transport_closure"] == transport_closure.closure_id
            and parent_identities["kinetic_method"] == method.method_id
            and parent_identities["energy_plan"] == binding.plan.equilibrium_plan.plan_id
            and tuple(parent["configuration"]["coarse_resolution"]) == SPATIAL_SHAPE
            and tuple(parent["configuration"]["coarse_cell_spacing"]) == CELL_SPACING
            and parent["configuration"]["coarse_time_step"] == TIME_STEP
        ),
        "explicit_model_and_runtime_execution": bool(
            initialization["explicit_unfrozen_model_evaluation"]
            and initialization["all_inputs_finite"]
            and initialization["all_lanes_inside_declared_support"]
            and initialization["all_model_outputs_finite"]
            and initialization["all_support_evaluations_successful"]
            and initialization["all_learned_equilibria_successful"]
            and initialization["minimum_particle_population"] > 0.0
            and initialization["minimum_total_energy_population"] > 0.0
            and initialization["runtime_successful"]
            and not initialization["runtime_rollback_applied"]
            and initialization["runtime_maximum_absolute_conservation_residual"]
            <= THRESHOLDS["runtime_conservation_maximum_absolute_residual"]
        ),
        "zero_source_identity": bool(
            zero["successful"]
            and zero["status"] == int(SmoothCompressibleD2VForcingStatus.SUCCESS)
            and not zero["rollback_applied"]
            and zero["candidate_matches_input_exactly"]
            and zero["accepted_matches_input_exactly"]
            and zero["particle_increment_exactly_zero"]
            and zero["energy_increment_exactly_zero"]
            and zero["accepted_conserved_increment_exactly_zero"]
            and zero["maximum_absolute_source_moment_residual"] == 0.0
        ),
        "uniform_acceleration_mass_impulse_and_midpoint_work": bool(
            acceleration["successful"]
            and acceleration["status"] == int(SmoothCompressibleD2VForcingStatus.SUCCESS)
            and not acceleration["rollback_applied"]
            and acceleration["target_mass_increment_exactly_zero"]
            and acceleration["volumetric_heating_increment_exactly_zero"]
            and acceleration["maximum_absolute_recovered_mass_increment"]
            <= source_tolerance
            and acceleration["maximum_absolute_momentum_impulse_residual"]
            <= source_tolerance
            and acceleration["maximum_absolute_midpoint_velocity_residual"]
            <= source_tolerance
            and acceleration["maximum_absolute_midpoint_energy_work_residual"]
            <= source_tolerance
            and acceleration["maximum_absolute_source_moment_residual"]
            <= source_tolerance
        ),
        "pure_volumetric_heating": bool(
            heating["successful"]
            and heating["status"] == int(SmoothCompressibleD2VForcingStatus.SUCCESS)
            and not heating["rollback_applied"]
            and heating["target_mass_increment_exactly_zero"]
            and heating["acceleration_work_increment_exactly_zero"]
            and heating["maximum_absolute_recovered_mass_increment"] <= source_tolerance
            and heating["maximum_absolute_recovered_momentum_increment"]
            <= source_tolerance
            and heating["maximum_absolute_heating_increment_residual"] <= source_tolerance
            and heating["maximum_absolute_source_moment_residual"] <= source_tolerance
        ),
        "combined_periodic_content_source_ledger": bool(
            combined["successful"]
            and combined["status"] == int(SmoothCompressibleD2VForcingStatus.SUCCESS)
            and not combined["rollback_applied"]
            and combined["periodic_transport_executed"]
            and combined["target_mass_source_exactly_zero"]
            and combined["post_transport_realizable"]
            and combined["post_transport_minimum_particle_population"] > 0.0
            and combined["post_transport_minimum_total_energy_population"] > 0.0
            and combined["maximum_absolute_source_application_residual"]
            <= ledger_tolerance
            and combined["maximum_absolute_accepted_source_residual"] <= ledger_tolerance
            and combined["maximum_absolute_periodic_transport_residual"]
            <= ledger_tolerance
            and combined["maximum_absolute_complete_content_source_ledger_residual"]
            <= ledger_tolerance
        ),
        "nonfinite_candidate_refusal_and_complete_rollback": bool(
            not nonfinite["successful"]
            and nonfinite["rollback_applied"]
            and nonfinite["status"] == nonfinite["expected_status"]
            and nonfinite["candidate_contains_nonfinite_population"]
            and nonfinite["accepted_state_matches_input_exactly"]
            and nonfinite["accepted_conserved_increment_exactly_zero"]
        ),
        "negative_candidate_refusal_and_complete_rollback": bool(
            not negative["successful"]
            and negative["rollback_applied"]
            and negative["status"] == negative["expected_status"]
            and negative["candidate_contains_negative_population"]
            and negative["accepted_state_matches_input_exactly"]
            and negative["accepted_conserved_increment_exactly_zero"]
        ),
        "distinct_forcing_plan_identities": len(set(forcing_plan_ids))
        == len(forcing_plan_ids),
    }

    report = {
        "tool": "learned_energy_forcing_qualification",
        "scope": {
            "stage": "deterministic learned-energy forcing qualification",
            "included": [
                "D2V17 conserved source transactions",
                "body acceleration momentum impulse",
                "midpoint acceleration energy work",
                "volumetric heating",
                "periodic transport content/source ledger",
                "atomic refusal and rollback",
            ],
            "excluded": [
                "boundary conditions",
                "shock qualification",
                "constitutive stress-source claims",
                "higher-order source-moment claims",
                "continuum-limit equivalence",
                "production-readiness claims",
            ],
            "claim": (
                "conservation/work-consistent body acceleration and volumetric heating only"
            ),
        },
        "contracts": {
            "conserved_order": [
                "mass",
                "x_momentum",
                "y_momentum",
                "total_energy",
            ],
            "acceleration_energy_work": "midpoint velocity",
            "volumetric_heating": "direct total-energy density increment",
            "transport": "exact periodic pull after accepted source transaction",
            "transaction": (
                "candidate is accepted atomically or the complete state and accepted conserved increment roll back"
            ),
        },
        "configuration": {
            "spatial_shape": list(SPATIAL_SHAPE),
            "domain_extent": list(DOMAIN_EXTENT),
            "cell_spacing": list(CELL_SPACING),
            "time_step": TIME_STEP,
            "particle_relaxation_time": PARTICLE_RELAXATION_TIME,
            "total_energy_relaxation_time": TOTAL_ENERGY_RELAXATION_TIME,
            "dtype": "float64",
        },
        "identities": {
            "parent_spatial_qualification": parent["qualification_id"],
            "parent_spatial_plan": parent_identities["spatial_plan"],
            "parent_prepared_spatial": parent_identities["prepared_spatial"],
            "model_artifact": artifact.artifact_id,
            "model_numeric_revision": binding.numeric_revision.revision_id,
            "model_semantic": binding.plan.semantic_id,
            "support": binding.plan.support.support_id,
            "normalizer": binding.plan.normalizer.normalizer_id,
            "quadrature": method.quadrature.quadrature_id,
            "material": method.material.material_id,
            "transport_closure": transport_closure.closure_id,
            "kinetic_method": method.method_id,
            "energy_plan": binding.plan.equilibrium_plan.plan_id,
            "spatial_plan": spatial_plan.plan_id,
            "prepared_spatial": runtime.prepared_id,
            "forcing": forcing_id,
            "zero_forcing_plan": zero_plan.plan_id,
            "acceleration_forcing_plan": acceleration_plan.plan_id,
            "heating_forcing_plan": heating_plan.plan_id,
            "combined_forcing_plan": combined_plan.plan_id,
            "nonfinite_candidate_forcing_plan": nonfinite_plan.plan_id,
            "negative_candidate_forcing_plan": negative_plan.plan_id,
        },
        "thresholds": THRESHOLDS,
        "initialization_and_runtime": initialization,
        "zero_source_identity": zero,
        "uniform_acceleration": acceleration,
        "pure_heating": heating,
        "combined_periodic_transport": combined,
        "nonfinite_candidate_refusal": nonfinite,
        "negative_candidate_refusal": negative,
        "gates": gates,
    }
    report["passed"] = all(gates.values())
    report["qualification_id"] = canonical_fingerprint(report)
    return report


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Qualify conserved D2V17 body-acceleration and volumetric-heating sources."
        )
    )
    parser.add_argument(
        "--model-artifact",
        type=Path,
        default=Path("benchmarks/learned_energy_equilibrium.phxml"),
        help="checksum-validated learned energy-equilibrium artifact",
    )
    parser.add_argument(
        "--parent-spatial",
        type=Path,
        default=Path("benchmarks/learned_energy_spatial.json"),
        help="passing parent learned-energy spatial qualification",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/learned_energy_forcing.json"),
        help="canonical atomic JSON output path",
    )
    return parser.parse_args()


def _validate_output_path(arguments: argparse.Namespace, /) -> None:
    output = arguments.output.resolve()
    protected = (
        arguments.model_artifact.resolve(),
        arguments.parent_spatial.resolve(),
        Path("benchmarks/learned_energy_equilibrium.json").resolve(),
    )
    if output in protected:
        raise ValueError("Forcing qualification output must be a distinct artifact.")


def main() -> int:
    arguments = _parse_arguments()
    _validate_output_path(arguments)
    with jax.enable_x64(True):
        report = _qualification(arguments.model_artifact, arguments.parent_spatial)
    if not report["passed"]:
        print(json.dumps(report, ensure_ascii=True, sort_keys=True))
        return 1
    write_json_atomic(arguments.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
