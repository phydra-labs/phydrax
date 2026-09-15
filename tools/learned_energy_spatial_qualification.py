#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks._io import atomic_write
from benchmarks._runtime import (
    capture_environment,
    CompilationTiming,
    DurationDistribution,
    measure_lower_and_compile,
    measure_synchronized,
)
from phydrax._array_archive import (
    array_collection_digest,
    read_array_archive,
    write_array_archive,
)
from phydrax._fingerprint import canonical_fingerprint, canonical_json
from phydrax.closure_data._kinetic_equilibrium import (
    PreparedLearnedEnergyEquilibriumBinding,
)
from phydrax.closure_data._kinetic_equilibrium_artifact import (
    LearnedEnergyEquilibriumArtifact,
    read_learned_energy_equilibrium_artifact,
)
from phydrax.discretization.discrete_velocity._energy_equilibrium import (
    PositiveEnergyEquilibriumPlan,
)
from phydrax.discretization.discrete_velocity._quadrature import d2v17_quadrature
from phydrax.discretization.discrete_velocity._smooth_compressible import (
    SmoothCompressibleD2VKineticMethod,
    SmoothCompressibleKineticState,
)
from phydrax.discretization.discrete_velocity._spatial import (
    D2V17PeriodicTransportPlan,
    PreparedSmoothCompressibleD2V17SpatialDynamics,
    SmoothCompressibleD2VStepStatus,
)
from phydrax.equations._materials import IdealGasMaterial
from phydrax.equations._transport_closures import ConstantTransport


DEFAULT_PARENT = Path("benchmarks/learned_energy_equilibrium.json")
DEFAULT_OUTPUT = Path("benchmarks/learned_energy_spatial.json")
DEFAULT_COARSE_RESOLUTION = 16
DEFAULT_HORIZON = 0.25
DEFAULT_WARMUPS = 2
DEFAULT_REPETITIONS = 10

THRESHOLDS = {
    "uniform_maximum_population_error": 5.0e-12,
    "maximum_conservation_residual": 1.0e-10,
    "maximum_f_stress_relative_l2_error": 8.0e-2,
    "maximum_g_energy_relative_l2_error": 4.0e-2,
    "maximum_g_equilibrium_flux_error": 1.0e-9,
    "maximum_g_flux_relative_l2_error": 8.0e-2,
    "maximum_conserved_relative_l2_error": 4.0e-2,
    "minimum_population": 0.0,
    "minimum_oracle_hull_margin": 1.0e-8,
    "minimum_learned_support_margin": 0.0,
    "maximum_checkpoint_resume_error": 0.0,
    "maximum_learned_f_stress_relative_l2_error": 8.0e-2,
    "maximum_learned_g_energy_relative_l2_error": 5.0e-2,
    "maximum_learned_g_flux_relative_l2_error": 8.0e-2,
}


def _relative_l2(candidate: jax.Array, reference: jax.Array, /) -> float:
    candidate_ = jnp.asarray(candidate)
    reference_ = jnp.asarray(reference)
    numerator = jnp.sqrt(jnp.sum((candidate_ - reference_) ** 2))
    denominator = jnp.maximum(
        jnp.sqrt(jnp.sum(reference_**2)),
        jnp.finfo(reference_.dtype).tiny,
    )
    return float(numerator / denominator)


def _state_maximum_error(
    candidate: SmoothCompressibleKineticState,
    reference: SmoothCompressibleKineticState,
    /,
) -> float:
    return float(
        jnp.maximum(
            jnp.max(
                jnp.abs(candidate.particle_populations - reference.particle_populations)
            ),
            jnp.max(
                jnp.abs(
                    candidate.total_energy_populations
                    - reference.total_energy_populations
                )
            ),
        )
    )


def _state_exactly_equal(
    candidate: SmoothCompressibleKineticState,
    reference: SmoothCompressibleKineticState,
    /,
) -> bool:
    return bool(
        np.array_equal(
            np.asarray(candidate.particle_populations),
            np.asarray(reference.particle_populations),
        )
        and np.array_equal(
            np.asarray(candidate.total_energy_populations),
            np.asarray(reference.total_energy_populations),
        )
    )


def _observables(
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    state: SmoothCompressibleKineticState,
    /,
) -> dict[str, jax.Array]:
    velocities = runtime.method.quadrature.velocities
    moments = runtime.method.moments(state)
    return {
        "conserved": moments.conserved,
        "f_stress": jnp.einsum(
            "...q,qi,qj->...ij",
            state.particle_populations,
            velocities,
            velocities,
        ),
        "g_energy": jnp.sum(state.total_energy_populations, axis=-1),
        "g_flux": jnp.einsum("...q,qd->...d", state.total_energy_populations, velocities),
    }


def _domain_content(
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    state: SmoothCompressibleKineticState,
    /,
) -> jax.Array:
    moments = runtime.method.moments(state)
    return jnp.sum(moments.conserved, axis=(0, 1)) * runtime.transport.cell_volume


def _minimum_populations(state: SmoothCompressibleKineticState, /) -> tuple[float, float]:
    return (
        float(jnp.min(state.particle_populations)),
        float(jnp.min(state.total_energy_populations)),
    )


def _primitive_conserved(
    resolution: int,
    material: IdealGasMaterial,
    /,
    *,
    uniform: bool,
) -> jax.Array:
    dtype = jnp.float64
    if uniform:
        density = jnp.ones((resolution, resolution), dtype=dtype)
        velocity_x = jnp.full((resolution, resolution), 0.04, dtype=dtype)
        velocity_y = jnp.full((resolution, resolution), -0.025, dtype=dtype)
        temperature = jnp.full(
            (resolution, resolution),
            0.5,
            dtype=dtype,
        )
    else:
        coordinate = (jnp.arange(resolution, dtype=dtype) + 0.5) / resolution
        x, y = jnp.meshgrid(coordinate, coordinate, indexing="ij")
        two_pi = 2.0 * jnp.pi
        density = 1.0 + 0.01 * jnp.sin(two_pi * x) * jnp.cos(two_pi * y)
        velocity_x = 0.04 + 0.008 * jnp.cos(two_pi * x) * jnp.sin(two_pi * y)
        velocity_y = -0.025 + 0.006 * jnp.sin(two_pi * (x + y))
        temperature = 0.5 + 0.01 * jnp.cos(two_pi * x) * jnp.cos(two_pi * y)
    pressure = density * material.gas_constant * temperature
    momentum_x = density * velocity_x
    momentum_y = density * velocity_y
    kinetic = 0.5 * density * (velocity_x**2 + velocity_y**2)
    total_energy = pressure / (material.gamma - 1.0) + kinetic
    return jnp.stack(
        (density, momentum_x, momentum_y, total_energy),
        axis=-1,
    )


def _initial_state(
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    conserved: jax.Array,
    /,
) -> tuple[SmoothCompressibleKineticState, dict[str, Any]]:
    moments = conserved
    density = moments[..., 0]
    momentum = moments[..., 1:3]
    total_energy = moments[..., -1]
    velocity = momentum / density[..., None]
    kinetic = 0.5 * jnp.sum(momentum * velocity, axis=-1)
    pressure = (runtime.method.material.gamma - 1.0) * (total_energy - kinetic)
    target_flux = (total_energy + pressure)[..., None] * velocity
    oracle = runtime.energy_plan.solve(total_energy, target_flux)
    state, evidence = runtime.method.equilibrium_from_energy_dual_with_evidence(
        conserved,
        oracle.dual,
        runtime.energy_plan,
    )
    return state, {
        "all_oracle_successful": bool(jnp.all(oracle.successful)),
        "all_equilibrium_successful": bool(evidence.successful),
        "minimum_hull_margin": float(jnp.min(oracle.evidence.interior_margin)),
        "minimum_f_population": float(jnp.min(state.particle_populations)),
        "minimum_g_population": float(jnp.min(state.total_energy_populations)),
        "maximum_f_stress_residual": float(
            evidence.maximum_absolute_particle_momentum_flux_residual
        ),
        "maximum_g_energy_residual": float(
            jnp.max(jnp.abs(oracle.evidence.total_energy_residual))
        ),
        "maximum_g_flux_error": float(jnp.max(oracle.evidence.flux_error_norm)),
    }


def _build_runtime(
    resolution: int,
    binding: PreparedLearnedEnergyEquilibriumBinding | None,
    /,
) -> PreparedSmoothCompressibleD2V17SpatialDynamics:
    if binding is None:
        material = IdealGasMaterial(1.4, 1.0)
        energy_plan = PositiveEnergyEquilibriumPlan(d2v17_quadrature(dtype=jnp.float64))
    else:
        material = binding.plan.material
        energy_plan = binding.plan.equilibrium_plan
    method = SmoothCompressibleD2VKineticMethod(
        energy_plan.quadrature,
        material,
        ConstantTransport(0.03, 0.04),
    )
    spacing = 1.0 / resolution
    transport = D2V17PeriodicTransportPlan(
        energy_plan.quadrature,
        (resolution, resolution),
        (spacing, spacing),
        spacing,
    )
    return PreparedSmoothCompressibleD2V17SpatialDynamics(
        method,
        energy_plan,
        transport,
        conservation_tolerance=THRESHOLDS["maximum_conservation_residual"],
    )


def _empty_rollout_accumulator(
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    initial_state: SmoothCompressibleKineticState,
    step_count: int,
    /,
) -> dict[str, Any]:
    minimum_f, minimum_g = _minimum_populations(initial_state)
    return {
        "content_component_order": ["mass", "momentum_x", "momentum_y", "energy"],
        "requested_steps": step_count,
        "stable_steps": 0,
        "all_steps_successful": True,
        "maximum_step_conservation_residual": 0.0,
        "maximum_f_equilibrium_stress_residual": 0.0,
        "maximum_g_equilibrium_energy_residual": 0.0,
        "maximum_g_equilibrium_flux_error": 0.0,
        "minimum_oracle_hull_margin": math.inf,
        "minimum_f_population": minimum_f,
        "minimum_g_population": minimum_g,
        "initial_content": [
            float(value) for value in _domain_content(runtime, initial_state)
        ],
    }


def _finish_rollout_record(
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    state: SmoothCompressibleKineticState,
    accumulator: dict[str, Any],
    /,
) -> dict[str, Any]:
    final_content = _domain_content(runtime, state)
    initial_content = jnp.asarray(
        accumulator["initial_content"], dtype=final_content.dtype
    )
    accumulator["final_content"] = [float(value) for value in final_content]
    accumulator["maximum_global_conservation_residual"] = float(
        jnp.max(jnp.abs(final_content - initial_content))
    )
    accumulator["stable_horizon"] = (
        accumulator["stable_steps"] * runtime.required_step_size
    )
    accumulator["requested_horizon"] = (
        accumulator["requested_steps"] * runtime.required_step_size
    )
    return accumulator


def _record_step(
    accumulator: dict[str, Any],
    result: Any,
    /,
    *,
    hull_margin: float,
    stable_prefix: bool,
) -> bool:
    step_successful = bool(result.successful)
    stable_prefix = stable_prefix and step_successful
    accumulator["all_steps_successful"] = bool(
        accumulator["all_steps_successful"] and step_successful
    )
    if stable_prefix:
        accumulator["stable_steps"] += 1
    conservation = result.evidence.conservation
    equilibrium = result.evidence.equilibrium
    accumulator["maximum_step_conservation_residual"] = max(
        accumulator["maximum_step_conservation_residual"],
        float(conservation.maximum_absolute_residual),
    )
    accumulator["maximum_f_equilibrium_stress_residual"] = max(
        accumulator["maximum_f_equilibrium_stress_residual"],
        float(equilibrium.maximum_absolute_particle_momentum_flux_residual),
    )
    accumulator["maximum_g_equilibrium_energy_residual"] = max(
        accumulator["maximum_g_equilibrium_energy_residual"],
        float(jnp.max(jnp.abs(equilibrium.energy.total_energy_residual))),
    )
    accumulator["maximum_g_equilibrium_flux_error"] = max(
        accumulator["maximum_g_equilibrium_flux_error"],
        float(jnp.max(equilibrium.energy.flux_error_norm)),
    )
    accumulator["minimum_oracle_hull_margin"] = min(
        accumulator["minimum_oracle_hull_margin"], hull_margin
    )
    minimum_f, minimum_g = _minimum_populations(result.accepted_state)
    accumulator["minimum_f_population"] = min(
        accumulator["minimum_f_population"], minimum_f
    )
    accumulator["minimum_g_population"] = min(
        accumulator["minimum_g_population"], minimum_g
    )
    return stable_prefix


def _rollout_oracle(
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    initial_state: SmoothCompressibleKineticState,
    step_count: int,
    /,
) -> tuple[SmoothCompressibleKineticState, dict[str, Any]]:
    state = initial_state
    accumulator = _empty_rollout_accumulator(runtime, initial_state, step_count)
    stable_prefix = True
    for _ in range(step_count):
        result, oracle = runtime.step_oracle(
            state,
            jnp.asarray(
                runtime.required_step_size,
                dtype=state.particle_populations.dtype,
            ),
        )
        stable_prefix = _record_step(
            accumulator,
            result,
            hull_margin=float(jnp.min(oracle.interior_margin)),
            stable_prefix=stable_prefix,
        )
        state = result.accepted_state
    return state, _finish_rollout_record(runtime, state, accumulator)


def _support_margin_record(evidence: Any, /) -> dict[str, float]:
    return {
        "rho": float(jnp.min(evidence.rho_margin)),
        "u_x": float(jnp.min(evidence.u_x_margin)),
        "u_y": float(jnp.min(evidence.u_y_margin)),
        "temperature": float(jnp.min(evidence.temperature_margin)),
        "mach": float(jnp.min(evidence.mach_margin)),
        "hull": float(jnp.min(evidence.hull_margin)),
        "particle_equilibrium": float(jnp.min(evidence.particle_equilibrium_margin)),
    }


def _rollout_learned(
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    binding: PreparedLearnedEnergyEquilibriumBinding,
    initial_state: SmoothCompressibleKineticState,
    step_count: int,
    /,
) -> tuple[SmoothCompressibleKineticState, dict[str, Any]]:
    state = initial_state
    accumulator = _empty_rollout_accumulator(runtime, initial_state, step_count)
    accumulator["all_support_successful"] = True
    support_margins = {
        "rho": math.inf,
        "u_x": math.inf,
        "u_y": math.inf,
        "temperature": math.inf,
        "mach": math.inf,
        "hull": math.inf,
        "particle_equilibrium": math.inf,
    }
    stable_prefix = True
    for _ in range(step_count):
        conserved = runtime.method.moments(state).conserved
        dual, support = binding.predict_dual_with_evidence(conserved)
        support_successful = bool(jnp.all(support.successful))
        accumulator["all_support_successful"] = bool(
            accumulator["all_support_successful"] and support_successful
        )
        for name, value in _support_margin_record(support).items():
            support_margins[name] = min(support_margins[name], value)
        result = runtime.step_with_energy_dual(
            state,
            jnp.asarray(
                runtime.required_step_size,
                dtype=state.particle_populations.dtype,
            ),
            dual,
        )
        stable_prefix = _record_step(
            accumulator,
            result,
            hull_margin=float(
                jnp.min(result.evidence.equilibrium.energy.interior_margin)
            ),
            stable_prefix=stable_prefix and support_successful,
        )
        state = result.accepted_state
    accumulator["minimum_support_margins"] = support_margins
    return state, _finish_rollout_record(runtime, state, accumulator)


def _restrict_two_to_one(value: jax.Array, coarse_resolution: int, /) -> jax.Array:
    trailing = value.shape[2:]
    return value.reshape(
        coarse_resolution,
        2,
        coarse_resolution,
        2,
        *trailing,
    ).mean(axis=(1, 3))


def _comparison_record(
    candidate: Mapping[str, jax.Array],
    reference: Mapping[str, jax.Array],
    /,
) -> dict[str, float]:
    return {
        "conserved_relative_l2_error": _relative_l2(
            candidate["conserved"], reference["conserved"]
        ),
        "f_stress_relative_l2_error": _relative_l2(
            candidate["f_stress"], reference["f_stress"]
        ),
        "g_energy_relative_l2_error": _relative_l2(
            candidate["g_energy"], reference["g_energy"]
        ),
        "g_flux_relative_l2_error": _relative_l2(
            candidate["g_flux"], reference["g_flux"]
        ),
    }


def _two_resolution_comparison(
    coarse_runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    coarse_state: SmoothCompressibleKineticState,
    fine_runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    fine_state: SmoothCompressibleKineticState,
    /,
) -> dict[str, float]:
    coarse = _observables(coarse_runtime, coarse_state)
    fine = {
        name: _restrict_two_to_one(value, coarse_runtime.transport.spatial_shape[0])
        for name, value in _observables(fine_runtime, fine_state).items()
    }
    return _comparison_record(coarse, fine)


def _learned_oracle_comparison(
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    learned_state: SmoothCompressibleKineticState,
    oracle_state: SmoothCompressibleKineticState,
    /,
) -> dict[str, float]:
    return _comparison_record(
        _observables(runtime, learned_state),
        _observables(runtime, oracle_state),
    )


def _rollback_checks(
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    state: SmoothCompressibleKineticState,
    /,
) -> dict[str, Any]:
    dtype = state.particle_populations.dtype
    wrong_step, _ = runtime.step_oracle(
        state,
        jnp.asarray(0.5 * runtime.required_step_size, dtype=dtype),
    )
    conserved = runtime.method.moments(state).conserved
    injected_dual = jnp.full(conserved.shape[:-1] + (2,), jnp.nan, dtype=dtype)
    injected = runtime.step_with_energy_dual(
        state,
        jnp.asarray(runtime.required_step_size, dtype=dtype),
        injected_dual,
    )
    return {
        "wrong_dt": {
            "successful": bool(wrong_step.successful),
            "rollback_applied": bool(wrong_step.rollback_applied),
            "status": SmoothCompressibleD2VStepStatus(
                int(wrong_step.evidence.status)
            ).name,
            "accepted_state_exactly_predecessor": _state_exactly_equal(
                wrong_step.accepted_state, state
            ),
        },
        "injected_nonfinite_dual": {
            "successful": bool(injected.successful),
            "rollback_applied": bool(injected.rollback_applied),
            "status": SmoothCompressibleD2VStepStatus(int(injected.evidence.status)).name,
            "accepted_state_exactly_predecessor": _state_exactly_equal(
                injected.accepted_state, state
            ),
        },
    }


def _advance_without_report(
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    binding: PreparedLearnedEnergyEquilibriumBinding | None,
    state: SmoothCompressibleKineticState,
    step_count: int,
    /,
) -> SmoothCompressibleKineticState:
    current = state
    for _ in range(step_count):
        if binding is None:
            result, _ = runtime.step_oracle(
                current,
                jnp.asarray(
                    runtime.required_step_size,
                    dtype=current.particle_populations.dtype,
                ),
            )
        else:
            dual, _ = binding.predict_dual_with_evidence(
                runtime.method.moments(current).conserved
            )
            result = runtime.step_with_energy_dual(
                current,
                jnp.asarray(
                    runtime.required_step_size,
                    dtype=current.particle_populations.dtype,
                ),
                dual,
            )
        current = result.accepted_state
    return current


def _checkpoint_resume(
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    binding: PreparedLearnedEnergyEquilibriumBinding | None,
    learned_artifact_id: str | None,
    initial_state: SmoothCompressibleKineticState,
    uninterrupted_final: SmoothCompressibleKineticState,
    step_count: int,
    /,
) -> dict[str, Any]:
    accepted_step = step_count // 2
    checkpoint_state = _advance_without_report(
        runtime,
        binding,
        initial_state,
        accepted_step,
    )
    arrays = {
        "accepted/particle_populations": np.asarray(
            checkpoint_state.particle_populations
        ),
        "accepted/total_energy_populations": np.asarray(
            checkpoint_state.total_energy_populations
        ),
    }
    payload_id = array_collection_digest(arrays)
    checkpoint_id = canonical_fingerprint(
        {
            "kind": "learned-energy-spatial-accepted-state-checkpoint",
            "runtime": runtime.prepared_id,
            "learned_energy_artifact": learned_artifact_id,
            "accepted_step": accepted_step,
            "payload": payload_id,
        }
    )
    manifest = {
        "kind": "learned-energy-spatial-accepted-state-checkpoint",
        "checkpoint_id": checkpoint_id,
        "runtime_id": runtime.prepared_id,
        "learned_energy_artifact_id": learned_artifact_id,
        "accepted_step": accepted_step,
        "payload_id": payload_id,
        "state_kind": "accepted_only",
    }
    with tempfile.TemporaryDirectory(
        prefix="phydrax-learned-energy-spatial-"
    ) as directory:
        checkpoint_path = Path(directory) / "checkpoint.phx"
        write_array_archive(checkpoint_path, manifest=manifest, arrays=arrays)
        restored_manifest, restored_arrays = read_array_archive(checkpoint_path)
    expected_manifest = {**manifest, "arrays": restored_manifest["arrays"]}
    if restored_manifest != expected_manifest:
        raise ValueError("Spatial checkpoint metadata changed during restoration.")
    if array_collection_digest(restored_arrays) != payload_id:
        raise ValueError(
            "Spatial checkpoint payload identity changed during restoration."
        )
    restored_state = SmoothCompressibleKineticState(
        jnp.asarray(restored_arrays["accepted/particle_populations"]),
        jnp.asarray(restored_arrays["accepted/total_energy_populations"]),
    )
    resumed_final = _advance_without_report(
        runtime,
        binding,
        restored_state,
        step_count - accepted_step,
    )
    return {
        "checkpoint_id": checkpoint_id,
        "payload_id": payload_id,
        "runtime_id": runtime.prepared_id,
        "learned_energy_artifact_id": learned_artifact_id,
        "state_kind": "accepted_only",
        "accepted_step": accepted_step,
        "resumed_steps": step_count - accepted_step,
        "restored_state_exactly_checkpointed": _state_exactly_equal(
            restored_state, checkpoint_state
        ),
        "resumed_final_exactly_uninterrupted": _state_exactly_equal(
            resumed_final, uninterrupted_final
        ),
        "maximum_resume_error": _state_maximum_error(resumed_final, uninterrupted_final),
    }


def _compiled_rollout(
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    binding: PreparedLearnedEnergyEquilibriumBinding | None,
    step_count: int,
    /,
) -> Callable[[SmoothCompressibleKineticState], Any]:
    step_size = runtime.required_step_size

    def rollout(initial_state: SmoothCompressibleKineticState):
        dtype = initial_state.particle_populations.dtype

        def step(state, _):
            if binding is None:
                result, _ = runtime.step_oracle(
                    state, jnp.asarray(step_size, dtype=dtype)
                )
            else:
                dual = binding.predict_dual(runtime.method.moments(state).conserved)
                result = runtime.step_with_energy_dual(
                    state,
                    jnp.asarray(step_size, dtype=dtype),
                    dual,
                )
            return result.accepted_state, result.successful

        final_state, successful = jax.lax.scan(
            step,
            initial_state,
            xs=None,
            length=step_count,
        )
        return final_state, jnp.all(successful)

    return jax.jit(rollout)


def _compilation_record(timing: CompilationTiming, /) -> dict[str, float]:
    return {
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
    }


def _paired_runtime(
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    binding: PreparedLearnedEnergyEquilibriumBinding,
    initial_state: SmoothCompressibleKineticState,
    step_count: int,
    /,
    *,
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    oracle_jit = _compiled_rollout(runtime, None, step_count)
    learned_jit = _compiled_rollout(runtime, binding, step_count)
    oracle_compiled, oracle_compilation = measure_lower_and_compile(
        lambda: oracle_jit.lower(initial_state),
        lambda lowered: lowered.compile(),
    )
    learned_compiled, learned_compilation = measure_lower_and_compile(
        lambda: learned_jit.lower(initial_state),
        lambda lowered: lowered.compile(),
    )
    oracle_operation = lambda: oracle_compiled(initial_state)
    learned_operation = lambda: learned_compiled(initial_state)
    for index in range(warmups):
        first, second = (
            (oracle_operation, learned_operation)
            if index % 2 == 0
            else (learned_operation, oracle_operation)
        )
        measure_synchronized(first)
        measure_synchronized(second)
    oracle_samples: list[float] = []
    learned_samples: list[float] = []
    oracle_result: Any = None
    learned_result: Any = None
    pair_ids: list[str] = []
    for index in range(repetitions):
        pair_id = f"spatial-runtime-pair-{index:04d}"
        pair_ids.append(pair_id)
        if index % 2 == 0:
            oracle_result, oracle_seconds = measure_synchronized(oracle_operation)
            learned_result, learned_seconds = measure_synchronized(learned_operation)
        else:
            learned_result, learned_seconds = measure_synchronized(learned_operation)
            oracle_result, oracle_seconds = measure_synchronized(oracle_operation)
        oracle_samples.append(oracle_seconds)
        learned_samples.append(learned_seconds)
    oracle_duration = DurationDistribution(tuple(oracle_samples))
    learned_duration = DurationDistribution(tuple(learned_samples))
    oracle_median = oracle_duration.median_seconds
    learned_median = learned_duration.median_seconds
    speedup = (
        None
        if oracle_median is None or learned_median is None or learned_median == 0.0
        else oracle_median / learned_median
    )
    return {
        "performed": True,
        "paired": True,
        "pair_ids": pair_ids,
        "warmups": warmups,
        "repetitions": repetitions,
        "step_count_per_sample": step_count,
        "oracle": {
            "operation": "compiled D2V17 oracle collide-stream rollout",
            "compilation": _compilation_record(oracle_compilation),
            "duration": oracle_duration.to_seconds_dict(),
            "last_result_successful": bool(oracle_result[1]),
        },
        "learned": {
            "operation": "compiled D2V17 learned-dual collide-stream rollout",
            "compilation": _compilation_record(learned_compilation),
            "duration": learned_duration.to_seconds_dict(),
            "last_result_successful": bool(learned_result[1]),
        },
        "oracle_over_learned_median_speedup": speedup,
    }


def _read_parent_stage_one(path: Path, /) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Stage-one qualification artifact must contain a JSON object.")
    identities = payload["identities"]
    if not isinstance(identities, dict):
        raise TypeError("Stage-one qualification identities must be a JSON object.")
    artifact_id = payload.get("artifact_id")
    if artifact_id is not None and (not isinstance(artifact_id, str) or not artifact_id):
        raise ValueError(
            "Stage-one artifact identity must be a non-empty string or null."
        )
    record = {
        "path": str(path),
        "tool": payload["tool"],
        "passed": bool(payload["passed"]),
        "artifact_id": artifact_id,
        "support_ids": {
            "dataset_preparation": identities["dataset_preparation"],
            "normalizer": identities["normalizer"],
            "normalizer_provenance": identities["normalizer_provenance"],
        },
        "runtime_ids": {
            "binding_plan": identities["binding_plan"],
            "numeric_revision": identities["numeric_revision"],
            "prepared_binding": identities["prepared_binding"],
        },
        "scientific_ids": {
            "quadrature": identities["quadrature"],
            "material": identities["material"],
            "flow_schema": identities["flow_schema"],
            "oracle_plan": identities["oracle_plan"],
        },
    }
    return payload, record


def _learned_artifact_record(
    artifact: LearnedEnergyEquilibriumArtifact,
    parent_payload: Mapping[str, Any],
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    /,
) -> dict[str, Any]:
    binding = artifact.binding
    plan = binding.plan
    parent_identities = parent_payload["identities"]
    compatibility = {
        "training_preparation_matches_parent": (
            plan.training_preparation_id == parent_identities["dataset_preparation"]
        ),
        "flow_schema_matches_parent": (
            plan.schema.schema_id == parent_identities["flow_schema"]
        ),
        "normalizer_matches_parent": (
            plan.normalizer.normalizer_id == parent_identities["normalizer"]
            and plan.normalizer.provenance.provenance_id
            == parent_identities["normalizer_provenance"]
        ),
        "semantic_matches_parent": (
            plan.semantic_id == parent_identities["semantic_provenance"]
        ),
        "numeric_revision_matches_parent": (
            binding.numeric_revision.revision_id == parent_identities["numeric_revision"]
        ),
        "quadrature_matches_parent": (
            plan.equilibrium_plan.quadrature.quadrature_id
            == parent_identities["quadrature"]
        ),
        "material_matches_parent": (
            plan.material.material_id == parent_identities["material"]
        ),
        "oracle_plan_matches_parent": (
            plan.equilibrium_plan.plan_id == parent_identities["oracle_plan"]
        ),
        "runtime_uses_loaded_binding_plan": (
            runtime.energy_plan.plan_id == plan.equilibrium_plan.plan_id
        ),
    }
    return {
        "artifact_id": artifact.artifact_id,
        "support_id": plan.support.support_id,
        "runtime_id": runtime.prepared_id,
        "prepared_binding_id": binding.prepared_id,
        "binding_plan_id": plan.plan_id,
        "numeric_revision_id": binding.numeric_revision.revision_id,
        "semantic_id": plan.semantic_id,
        "training_preparation_id": plan.training_preparation_id,
        "parent_artifact_id": plan.parent_artifact_id,
        "compatibility": compatibility,
        "all_compatible": all(compatibility.values()),
    }


def _ids_present(value: Mapping[str, Any], /) -> bool:
    return all(isinstance(item, str) and bool(item) for item in value.values())


def build_report(
    *,
    configuration: Mapping[str, Any],
    parent_stage_one: Mapping[str, Any],
    identities: Mapping[str, Any],
    uniform_free_stream: Mapping[str, Any],
    smooth_periodic: Mapping[str, Any],
    checkpoint_resume: Mapping[str, Any],
    rollback: Mapping[str, Any],
    learned: Mapping[str, Any],
    runtime: Mapping[str, Any],
    environment: Mapping[str, Any],
    thresholds: Mapping[str, float] = THRESHOLDS,
) -> dict[str, Any]:
    """Build deterministic JSON qualification evidence from pure report inputs."""

    limits = dict(thresholds)
    coarse = smooth_periodic["coarse_oracle"]
    fine = smooth_periodic["fine_oracle"]
    resolution = smooth_periodic["two_resolution_reference"]
    learned_requested = bool(learned["requested"])
    learned_passed = bool(
        not learned_requested
        or (
            learned["performed"]
            and learned["artifact"]["all_compatible"]
            and learned["coarse"]["all_steps_successful"]
            and learned["coarse"]["all_support_successful"]
            and learned["coarse"]["stable_steps"] == learned["coarse"]["requested_steps"]
            and min(learned["coarse"]["minimum_support_margins"].values())
            >= limits["minimum_learned_support_margin"]
            and learned["coarse"]["minimum_f_population"] >= limits["minimum_population"]
            and learned["coarse"]["minimum_g_population"] >= limits["minimum_population"]
            and learned["comparison_to_oracle"]["f_stress_relative_l2_error"]
            <= limits["maximum_learned_f_stress_relative_l2_error"]
            and learned["comparison_to_oracle"]["g_energy_relative_l2_error"]
            <= limits["maximum_learned_g_energy_relative_l2_error"]
            and learned["comparison_to_oracle"]["g_flux_relative_l2_error"]
            <= limits["maximum_learned_g_flux_relative_l2_error"]
        )
    )
    runtime_passed = bool(
        not learned_requested
        or (
            runtime["performed"]
            and runtime["paired"]
            and runtime["repetitions"] == len(runtime["pair_ids"])
            and len(set(runtime["pair_ids"])) == len(runtime["pair_ids"])
            and runtime["oracle"]["duration"]["count"] == runtime["repetitions"]
            and runtime["learned"]["duration"]["count"] == runtime["repetitions"]
            and runtime["oracle"]["last_result_successful"]
            and runtime["learned"]["last_result_successful"]
        )
    )
    gates = {
        "parent_stage_one": bool(
            parent_stage_one["tool"] == "learned_energy_equilibrium_qualification"
            and parent_stage_one["passed"]
            and _ids_present(parent_stage_one["support_ids"])
            and _ids_present(parent_stage_one["runtime_ids"])
            and _ids_present(parent_stage_one["scientific_ids"])
        ),
        "spatial_runtime_binds_parent": bool(
            identities["quadrature"] == parent_stage_one["scientific_ids"]["quadrature"]
            and identities["material"] == parent_stage_one["scientific_ids"]["material"]
            and identities["energy_plan"]
            == parent_stage_one["scientific_ids"]["oracle_plan"]
        ),
        "fixed_lattice_step_no_reduction": bool(
            not configuration["step_reduction_allowed"]
            and configuration["coarse_time_step"]
            == configuration["coarse_cell_spacing"][0]
            and configuration["coarse_time_step"]
            == configuration["coarse_cell_spacing"][1]
            and configuration["fine_time_step"] == configuration["fine_cell_spacing"][0]
            and configuration["fine_time_step"] == configuration["fine_cell_spacing"][1]
        ),
        "uniform_free_stream": bool(
            uniform_free_stream["initialization"]["all_oracle_successful"]
            and uniform_free_stream["initialization"]["all_equilibrium_successful"]
            and uniform_free_stream["rollout"]["all_steps_successful"]
            and uniform_free_stream["maximum_population_error"]
            <= limits["uniform_maximum_population_error"]
        ),
        "exact_conservation": bool(
            uniform_free_stream["rollout"]["maximum_step_conservation_residual"]
            <= limits["maximum_conservation_residual"]
            and uniform_free_stream["rollout"]["maximum_global_conservation_residual"]
            <= limits["maximum_conservation_residual"]
            and coarse["maximum_step_conservation_residual"]
            <= limits["maximum_conservation_residual"]
            and coarse["maximum_global_conservation_residual"]
            <= limits["maximum_conservation_residual"]
            and fine["maximum_step_conservation_residual"]
            <= limits["maximum_conservation_residual"]
            and fine["maximum_global_conservation_residual"]
            <= limits["maximum_conservation_residual"]
            and (
                not learned_requested
                or (
                    learned["coarse"]["maximum_step_conservation_residual"]
                    <= limits["maximum_conservation_residual"]
                    and learned["coarse"]["maximum_global_conservation_residual"]
                    <= limits["maximum_conservation_residual"]
                )
            )
        ),
        "f_stress": bool(
            resolution["f_stress_relative_l2_error"]
            <= limits["maximum_f_stress_relative_l2_error"]
            and coarse["maximum_f_equilibrium_stress_residual"]
            <= limits["maximum_conservation_residual"]
            and fine["maximum_f_equilibrium_stress_residual"]
            <= limits["maximum_conservation_residual"]
        ),
        "g_energy_and_flux": bool(
            resolution["g_energy_relative_l2_error"]
            <= limits["maximum_g_energy_relative_l2_error"]
            and resolution["g_flux_relative_l2_error"]
            <= limits["maximum_g_flux_relative_l2_error"]
            and coarse["maximum_g_equilibrium_energy_residual"]
            <= limits["maximum_conservation_residual"]
            and fine["maximum_g_equilibrium_energy_residual"]
            <= limits["maximum_conservation_residual"]
            and coarse["maximum_g_equilibrium_flux_error"]
            <= limits["maximum_g_equilibrium_flux_error"]
            and fine["maximum_g_equilibrium_flux_error"]
            <= limits["maximum_g_equilibrium_flux_error"]
        ),
        "two_resolution_oracle_reference": bool(
            coarse["all_steps_successful"]
            and fine["all_steps_successful"]
            and resolution["conserved_relative_l2_error"]
            <= limits["maximum_conserved_relative_l2_error"]
        ),
        "positivity": bool(
            min(
                coarse["minimum_f_population"],
                coarse["minimum_g_population"],
                fine["minimum_f_population"],
                fine["minimum_g_population"],
            )
            >= limits["minimum_population"]
        ),
        "hull_and_support_margins": bool(
            min(
                coarse["minimum_oracle_hull_margin"],
                fine["minimum_oracle_hull_margin"],
            )
            >= limits["minimum_oracle_hull_margin"]
            and learned_passed
        ),
        "stable_horizon": bool(
            coarse["stable_steps"] == coarse["requested_steps"]
            and fine["stable_steps"] == fine["requested_steps"]
            and math.isclose(coarse["stable_horizon"], configuration["horizon"])
            and math.isclose(fine["stable_horizon"], configuration["horizon"])
        ),
        "checkpoint_resume": bool(
            checkpoint_resume["state_kind"] == "accepted_only"
            and checkpoint_resume["restored_state_exactly_checkpointed"]
            and checkpoint_resume["resumed_final_exactly_uninterrupted"]
            and checkpoint_resume["maximum_resume_error"]
            <= limits["maximum_checkpoint_resume_error"]
            and checkpoint_resume["runtime_id"] == identities["prepared_spatial"]
            and (
                not learned_requested
                or checkpoint_resume["learned_energy_artifact_id"]
                == learned["artifact"]["artifact_id"]
            )
        ),
        "wrong_dt_refusal": bool(
            not rollback["wrong_dt"]["successful"]
            and rollback["wrong_dt"]["rollback_applied"]
            and rollback["wrong_dt"]["status"] == "INVALID_INPUT_STATE"
            and rollback["wrong_dt"]["accepted_state_exactly_predecessor"]
        ),
        "injected_rollback": bool(
            not rollback["injected_nonfinite_dual"]["successful"]
            and rollback["injected_nonfinite_dual"]["rollback_applied"]
            and rollback["injected_nonfinite_dual"]["accepted_state_exactly_predecessor"]
        ),
        "loaded_learned_artifact_comparison": learned_passed,
        "paired_oracle_learned_runtime": runtime_passed,
    }
    report = {
        "tool": "learned_energy_spatial_qualification",
        "scope": {
            "stage": "deterministic stage-2 spatial qualification",
            "claim": "periodic D2V17 smooth-flow spatial runtime only",
            "included": [
                "D2V17 exact periodic pull transport",
                "uniform free stream",
                "smooth low-amplitude periodic perturbation",
                "two-resolution oracle reference",
                "optional frozen learned-artifact comparison",
                "accepted-state checkpoint and resume",
                "wrong-step refusal and injected rollback",
                "paired oracle and learned runtime when learned artifact is supplied",
            ],
            "excluded": [
                "boundaries",
                "forcing",
                "shocks",
                "D2V37",
                "entropy qualification",
                "production qualification",
            ],
        },
        "contracts": {
            "transport": "exact fixed-step periodic D2V17 pull",
            "step_reduction": "forbidden",
            "conserved_content_order": [
                "mass",
                "momentum_x",
                "momentum_y",
                "total_energy",
            ],
            "f_stress": "sum_q(f_q c_q tensor c_q)",
            "g_energy": "sum_q(g_q)",
            "g_flux": "sum_q(g_q c_q)",
            "fine_to_coarse_reference": "2x2 conservative cell average",
            "checkpoint_state": "accepted populations only",
        },
        "configuration": dict(configuration),
        "parent_stage_one": dict(parent_stage_one),
        "identities": dict(identities),
        "thresholds": limits,
        "uniform_free_stream": dict(uniform_free_stream),
        "smooth_periodic": dict(smooth_periodic),
        "learned": dict(learned),
        "checkpoint_resume": dict(checkpoint_resume),
        "rollback": dict(rollback),
        "runtime": dict(runtime),
        "environment": dict(environment),
        "gates": gates,
    }
    report["passed"] = all(gates.values())
    report["qualification_id"] = canonical_fingerprint(report)
    return report


def qualification_report(
    *,
    parent_stage_one_path: Path = DEFAULT_PARENT,
    learned_artifact_path: Path | None = None,
    coarse_resolution: int = DEFAULT_COARSE_RESOLUTION,
    horizon: float = DEFAULT_HORIZON,
    warmups: int = DEFAULT_WARMUPS,
    repetitions: int = DEFAULT_REPETITIONS,
) -> dict[str, Any]:
    """Execute the deterministic spatial cases and build their pure report."""

    resolution = int(coarse_resolution)
    horizon_ = float(horizon)
    warmups_ = int(warmups)
    repetitions_ = int(repetitions)
    if resolution < 5:
        raise ValueError("coarse_resolution must be at least five.")
    if not math.isfinite(horizon_) or horizon_ <= 0.0:
        raise ValueError("horizon must be finite and positive.")
    if warmups_ < 0 or repetitions_ < 1:
        raise ValueError("runtime warmups must be nonnegative and repetitions positive.")
    coarse_step_count = round(horizon_ * resolution)
    fine_resolution = 2 * resolution
    fine_step_count = round(horizon_ * fine_resolution)
    if (
        coarse_step_count < 2
        or fine_step_count != 2 * coarse_step_count
        or not math.isclose(coarse_step_count / resolution, horizon_, abs_tol=1.0e-14)
    ):
        raise ValueError(
            "horizon must contain an integer number of coarse and fine lattice steps."
        )

    parent_payload, parent_record = _read_parent_stage_one(parent_stage_one_path)
    artifact = (
        None
        if learned_artifact_path is None
        else read_learned_energy_equilibrium_artifact(learned_artifact_path)
    )
    binding = None if artifact is None else artifact.binding
    coarse_runtime = _build_runtime(resolution, binding)
    fine_runtime = _build_runtime(fine_resolution, binding)

    uniform_conserved = _primitive_conserved(
        resolution, coarse_runtime.method.material, uniform=True
    )
    uniform_initial, uniform_initialization = _initial_state(
        coarse_runtime, uniform_conserved
    )
    uniform_final, uniform_rollout = _rollout_oracle(
        coarse_runtime, uniform_initial, coarse_step_count
    )
    uniform_record = {
        "initialization": uniform_initialization,
        "rollout": uniform_rollout,
        "maximum_population_error": _state_maximum_error(uniform_final, uniform_initial),
    }

    coarse_conserved = _primitive_conserved(
        resolution, coarse_runtime.method.material, uniform=False
    )
    fine_conserved = _primitive_conserved(
        fine_resolution, fine_runtime.method.material, uniform=False
    )
    coarse_initial, coarse_initialization = _initial_state(
        coarse_runtime, coarse_conserved
    )
    fine_initial, fine_initialization = _initial_state(fine_runtime, fine_conserved)
    coarse_oracle_final, coarse_oracle = _rollout_oracle(
        coarse_runtime, coarse_initial, coarse_step_count
    )
    fine_oracle_final, fine_oracle = _rollout_oracle(
        fine_runtime, fine_initial, fine_step_count
    )
    smooth_record = {
        "amplitude_regime": "low",
        "periodicity": "two-dimensional unit torus",
        "coarse_initialization": coarse_initialization,
        "fine_initialization": fine_initialization,
        "reference_restriction": "2x2 conservative cell average",
        "coarse_oracle": coarse_oracle,
        "fine_oracle": fine_oracle,
        "two_resolution_reference": _two_resolution_comparison(
            coarse_runtime,
            coarse_oracle_final,
            fine_runtime,
            fine_oracle_final,
        ),
    }

    if artifact is None:
        learned_record: dict[str, Any] = {
            "requested": False,
            "path": None,
            "performed": False,
            "reason": "no learned artifact supplied",
            "artifact": None,
            "coarse": None,
            "comparison_to_oracle": None,
        }
        runtime_record: dict[str, Any] = {
            "performed": False,
            "paired": False,
            "reason": "no learned artifact supplied",
        }
        checkpoint_binding = None
        checkpoint_final = coarse_oracle_final
        learned_artifact_id = None
    else:
        learned_final, learned_rollout = _rollout_learned(
            coarse_runtime,
            binding,
            coarse_initial,
            coarse_step_count,
        )
        learned_record = {
            "requested": True,
            "path": str(learned_artifact_path),
            "performed": True,
            "reason": None,
            "artifact": _learned_artifact_record(
                artifact,
                parent_payload,
                coarse_runtime,
            ),
            "coarse": learned_rollout,
            "comparison_to_oracle": _learned_oracle_comparison(
                coarse_runtime,
                learned_final,
                coarse_oracle_final,
            ),
        }
        runtime_record = _paired_runtime(
            coarse_runtime,
            binding,
            coarse_initial,
            coarse_step_count,
            warmups=warmups_,
            repetitions=repetitions_,
        )
        checkpoint_binding = binding
        checkpoint_final = learned_final
        learned_artifact_id = artifact.artifact_id

    checkpoint_record = _checkpoint_resume(
        coarse_runtime,
        checkpoint_binding,
        learned_artifact_id,
        coarse_initial,
        checkpoint_final,
        coarse_step_count,
    )
    rollback_record = _rollback_checks(coarse_runtime, coarse_initial)
    configuration = {
        "deterministic": True,
        "coarse_resolution": [resolution, resolution],
        "fine_resolution": [fine_resolution, fine_resolution],
        "domain_extent": [1.0, 1.0],
        "coarse_cell_spacing": list(coarse_runtime.transport.cell_spacing),
        "fine_cell_spacing": list(fine_runtime.transport.cell_spacing),
        "coarse_time_step": coarse_runtime.required_step_size,
        "fine_time_step": fine_runtime.required_step_size,
        "coarse_step_count": coarse_step_count,
        "fine_step_count": fine_step_count,
        "horizon": horizon_,
        "runtime_warmups": warmups_,
        "step_reduction_allowed": coarse_runtime.allows_step_reduction,
        "runtime_repetitions": repetitions_,
        "learned_artifact_supplied": artifact is not None,
    }
    identities = {
        "quadrature": coarse_runtime.method.quadrature.quadrature_id,
        "material": coarse_runtime.method.material.material_id,
        "transport_closure": coarse_runtime.method.transport.closure_id,
        "kinetic_method": coarse_runtime.method.method_id,
        "energy_plan": coarse_runtime.energy_plan.plan_id,
        "spatial_plan": coarse_runtime.transport.plan_id,
        "fine_spatial_plan": fine_runtime.transport.plan_id,
        "prepared_spatial": coarse_runtime.prepared_id,
        "fine_prepared_spatial": fine_runtime.prepared_id,
        "artifact": learned_artifact_id,
        "model_numeric_revision": (
            None if binding is None else binding.numeric_revision.revision_id
        ),
        "support": (
            None if artifact is None else artifact.binding.plan.support.support_id
        ),
        "prepared_binding": None if binding is None else binding.prepared_id,
    }
    return build_report(
        configuration=configuration,
        parent_stage_one=parent_record,
        identities=identities,
        uniform_free_stream=uniform_record,
        smooth_periodic=smooth_record,
        checkpoint_resume=checkpoint_record,
        rollback=rollback_record,
        learned=learned_record,
        runtime=runtime_record,
        environment=capture_environment().to_dict(),
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Qualify periodic D2V17 oracle and optional frozen learned-energy "
            "spatial runtime execution."
        )
    )
    parser.add_argument(
        "--parent-stage-one",
        type=Path,
        default=DEFAULT_PARENT,
        help="stage-one learned-energy qualification JSON",
    )
    parser.add_argument(
        "--learned-artifact",
        type=Path,
        default=None,
        help="optional frozen learned-energy deployment artifact",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="canonical JSON output written only when every gate passes",
    )
    parser.add_argument(
        "--coarse-resolution",
        type=int,
        default=DEFAULT_COARSE_RESOLUTION,
    )
    parser.add_argument("--horizon", type=float, default=DEFAULT_HORIZON)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    return parser.parse_args()


def _assert_output_is_not_stage_one(output: Path, parent: Path, /) -> None:
    destination = output.resolve()
    forbidden = (parent.resolve(), DEFAULT_PARENT.resolve())
    aliases_stage_one = destination in forbidden or (
        destination.exists()
        and any(
            stage_one.exists() and destination.samefile(stage_one)
            for stage_one in forbidden
        )
    )
    if aliases_stage_one:
        raise ValueError("Spatial qualification must never overwrite stage-one JSON.")


def main() -> int:
    arguments = _parse_arguments()
    _assert_output_is_not_stage_one(arguments.output, arguments.parent_stage_one)
    with jax.enable_x64(True):
        report = qualification_report(
            parent_stage_one_path=arguments.parent_stage_one,
            learned_artifact_path=arguments.learned_artifact,
            coarse_resolution=arguments.coarse_resolution,
            horizon=arguments.horizon,
            warmups=arguments.warmups,
            repetitions=arguments.repetitions,
        )
    if not report["passed"]:
        failed = sorted(name for name, passed in report["gates"].items() if not passed)
        print("failed qualification gates: " + ", ".join(failed), file=sys.stderr)
        return 1
    payload = canonical_json(report) + "\n"
    atomic_write(
        arguments.output,
        lambda temporary: temporary.write_text(payload, encoding="utf-8"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
