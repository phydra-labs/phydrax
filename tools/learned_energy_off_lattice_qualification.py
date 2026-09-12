#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import math
import subprocess
import sys
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
    compiler_evidence,
    DurationDistribution,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_synchronized,
)
from phydrax._fingerprint import canonical_fingerprint, canonical_json
from phydrax.discretization.discrete_velocity._quadrature import (
    CertifiedDiscreteVelocityQuadrature,
    d2v17_quadrature,
    d2v37_off_lattice_quadrature,
)
from phydrax.discretization.discrete_velocity._semi_lagrangian import (
    CoupledD2V37TransportStatus,
    PreparedCoupledD2V37OffLatticeTransport,
)
from phydrax.discretization.discrete_velocity._smooth_compressible import (
    SmoothCompressibleD2VKineticMethod,
    SmoothCompressibleKineticState,
)
from phydrax.discretization.discrete_velocity._spatial import (
    D2V17PeriodicTransportPlan,
)
from phydrax.equations._materials import IdealGasMaterial
from phydrax.equations._transport_closures import ConstantTransport


DEFAULT_OUTPUT = Path("benchmarks/learned_energy_off_lattice.json")
DEFAULT_COARSE_RESOLUTION = 16
DEFAULT_RUNTIME_RESOLUTION = 64
DEFAULT_WARMUPS = 2
DEFAULT_REPETITIONS = 10
DOMAIN_EXTENT = (1.0, 1.0)
FRACTIONAL_COURANT = 0.31
PARTICLE_RELAXATION_TIME = 0.03
TOTAL_ENERGY_RELAXATION_TIME = 0.04
CONSERVATION_TOLERANCE = 1.0e-11
_D2V17_REJECTION_PROBE = "--internal-d2v17-rejection-probe"
_D2V17_REJECTION_MESSAGE = (
    "Coupled off-lattice transport requires the declared D2V37 quadrature identity."
)
_PROTECTED_ARTIFACT_NAMES = frozenset(
    {
        "learned_energy_equilibrium.json",
        "learned_energy_equilibrium.phxml",
        "learned_energy_boundaries.json",
        "learned_energy_boundary.json",
        "learned_energy_export.json",
        "learned_energy_forcing.json",
        "learned_energy_spatial.json",
        "learned_energy_stage_two.json",
        "learned_energy_stage_two.phxml",
    }
)
_PROTECTED_ARTIFACT_PATHS = tuple(
    Path("benchmarks") / name for name in sorted(_PROTECTED_ARTIFACT_NAMES)
)

THRESHOLDS = {
    "maximum_constant_population_error": 2.0e-12,
    "maximum_population_integral_residual": 5.0e-11,
    "maximum_declared_moment_residual": 5.0e-11,
    "minimum_population": 0.0,
    "maximum_gradient_directional_relative_error": 5.0e-11,
    "maximum_finest_fluctuation_relative_l2_error": 2.0e-2,
    "maximum_refinement_error_ratio": 0.40,
    "minimum_observed_binary_refinement_order": 1.50,
    "maximum_forward_to_reverse_direction_error_ratio": 0.25,
}


def _maximum_absolute(values: jax.Array, /) -> float:
    return float(jnp.max(jnp.abs(values)))


def _vector(values: jax.Array, /) -> list[float]:
    return [float(value) for value in np.asarray(values).reshape(-1)]


def _population_fluctuation_relative_l2(
    candidate: jax.Array, reference: jax.Array, /
) -> float:
    reference_fluctuation = reference - jnp.mean(reference, axis=(0, 1), keepdims=True)
    numerator = jnp.sqrt(jnp.sum((candidate - reference) ** 2))
    denominator = jnp.sqrt(jnp.sum(reference_fluctuation**2))
    return float(numerator / denominator)


def _state_fluctuation_relative_l2(
    candidate: SmoothCompressibleKineticState,
    reference: SmoothCompressibleKineticState,
    /,
) -> float:
    reference_f = reference.particle_populations
    reference_g = reference.total_energy_populations
    fluctuation_f = reference_f - jnp.mean(reference_f, axis=(0, 1), keepdims=True)
    fluctuation_g = reference_g - jnp.mean(reference_g, axis=(0, 1), keepdims=True)
    error_square = jnp.sum((candidate.particle_populations - reference_f) ** 2)
    error_square += jnp.sum((candidate.total_energy_populations - reference_g) ** 2)
    scale_square = jnp.sum(fluctuation_f**2) + jnp.sum(fluctuation_g**2)
    return float(jnp.sqrt(error_square / scale_square))


def _methods() -> tuple[
    SmoothCompressibleD2VKineticMethod,
    SmoothCompressibleD2VKineticMethod,
    IdealGasMaterial,
    ConstantTransport,
]:
    material = IdealGasMaterial(1.4, 1.0)
    closure = ConstantTransport(PARTICLE_RELAXATION_TIME, TOTAL_ENERGY_RELAXATION_TIME)
    d2v37_method = SmoothCompressibleD2VKineticMethod(
        d2v37_off_lattice_quadrature(dtype=jnp.float64), material, closure
    )
    d2v17_method = SmoothCompressibleD2VKineticMethod(
        d2v17_quadrature(dtype=jnp.float64), material, closure
    )
    return d2v37_method, d2v17_method, material, closure


def _prepare_d2v37(
    method: SmoothCompressibleD2VKineticMethod, resolution: int, /
) -> PreparedCoupledD2V37OffLatticeTransport:
    spacing = tuple(extent / resolution for extent in DOMAIN_EXTENT)
    time_step = FRACTIONAL_COURANT * spacing[0]
    return PreparedCoupledD2V37OffLatticeTransport.prepare(
        method,
        (resolution, resolution),
        spacing,
        time_step,
        conservation_tolerance=CONSERVATION_TOLERANCE,
    )


def _analytic_state(
    quadrature: CertifiedDiscreteVelocityQuadrature,
    resolution: int,
    /,
    *,
    displacement_time: float = 0.0,
) -> SmoothCompressibleKineticState:
    dtype = quadrature.velocities.dtype
    coordinate = (jnp.arange(resolution, dtype=dtype) + 0.5) / resolution
    x = coordinate[:, None, None]
    y = coordinate[None, :, None]
    velocity_x = quadrature.velocities[None, None, :, 0]
    velocity_y = quadrature.velocities[None, None, :, 1]
    population = jnp.arange(quadrature.population_count, dtype=dtype)[None, None, :]
    shifted_x = x - velocity_x * displacement_time
    shifted_y = y - velocity_y * displacement_time
    two_pi = jnp.asarray(2.0 * math.pi, dtype=dtype)

    f_base = 0.35 + 0.008 * population
    f_modulation = (
        0.10 * jnp.sin(two_pi * (shifted_x + 0.017 * population))
        + 0.07 * jnp.cos(two_pi * (shifted_y - 0.023 * population))
        + 0.04 * jnp.sin(two_pi * (shifted_x + shifted_y + 0.011 * population))
    )
    g_base = 0.55 + 0.009 * population
    g_modulation = (
        0.09 * jnp.cos(two_pi * (shifted_x - 0.019 * population))
        + 0.08 * jnp.sin(two_pi * (shifted_y + 0.013 * population))
        + 0.035 * jnp.cos(two_pi * (shifted_x - shifted_y + 0.007 * population))
    )
    return SmoothCompressibleKineticState(
        f_base * (1.0 + f_modulation),
        g_base * (1.0 + g_modulation),
    )


def _fractional_shift_record(
    prepared: PreparedCoupledD2V37OffLatticeTransport, /
) -> dict[str, Any]:
    transfers = prepared.population_transport.departure_transfers
    offsets = np.asarray([transfer.offset_in_cells for transfer in transfers])
    velocities = np.asarray(prepared.quadrature.velocities)
    moving = np.linalg.norm(velocities, axis=1) > 0.0
    distance_from_integer = np.max(np.abs(offsets - np.rint(offsets)), axis=1)
    fractional_axes = np.abs(offsets - np.rint(offsets)) > 1.0e-12
    expected = (
        velocities
        * prepared.required_step_size
        / np.asarray(prepared.cell_spacing)[None, :]
    )
    return {
        "offsets_in_cells": offsets.tolist(),
        "moving_population_count": int(np.count_nonzero(moving)),
        "fractional_moving_population_count": int(
            np.count_nonzero(moving & (distance_from_integer > 1.0e-12))
        ),
        "two_axis_fractional_population_count": int(
            np.count_nonzero(np.all(fractional_axes, axis=1))
        ),
        "minimum_moving_distance_from_integer": float(
            np.min(distance_from_integer[moving])
        ),
        "maximum_absolute_offset_in_cells": float(np.max(np.abs(offsets))),
        "maximum_prepared_offset_residual": float(np.max(np.abs(offsets - expected))),
        "all_transfers_periodic": bool(
            all(all(transfer.periodic_axes) for transfer in transfers)
        ),
        "transfer_count": len(transfers),
    }


def _constant_case(
    prepared: PreparedCoupledD2V37OffLatticeTransport, /
) -> dict[str, Any]:
    quadrature = prepared.quadrature
    shape = prepared.population_transport.source_shape
    population = jnp.arange(
        quadrature.population_count, dtype=quadrature.velocities.dtype
    )
    f_constant = 0.20 + 0.01 * population
    g_constant = 0.40 + 0.015 * population
    state = SmoothCompressibleKineticState(
        jnp.broadcast_to(f_constant, shape + (quadrature.population_count,)),
        jnp.broadcast_to(g_constant, shape + (quadrature.population_count,)),
    )
    result = prepared.transport_with_evidence(state, prepared.required_step_size)
    return {
        "successful": bool(result.successful),
        "status": int(result.status),
        "transport_id": result.transport_id,
        "prepared_id": result.prepared_id,
        "maximum_f_population_error": _maximum_absolute(
            result.candidate_state.particle_populations - state.particle_populations
        ),
        "maximum_g_population_error": _maximum_absolute(
            result.candidate_state.total_energy_populations
            - state.total_energy_populations
        ),
        "minimum_candidate_f_population": float(
            jnp.min(result.candidate_state.particle_populations)
        ),
        "minimum_candidate_g_population": float(
            jnp.min(result.candidate_state.total_energy_populations)
        ),
    }


def _smooth_transport_case(
    prepared: PreparedCoupledD2V37OffLatticeTransport,
    resolution: int,
    /,
) -> dict[str, Any]:
    initial = _analytic_state(prepared.quadrature, resolution)
    forward_exact = _analytic_state(
        prepared.quadrature,
        resolution,
        displacement_time=prepared.required_step_size,
    )
    reverse_exact = _analytic_state(
        prepared.quadrature,
        resolution,
        displacement_time=-prepared.required_step_size,
    )
    result = prepared.transport_with_evidence(initial, prepared.required_step_size)
    candidate = result.candidate_state
    evidence = result.evidence
    maximum_change = max(
        _maximum_absolute(candidate.particle_populations - initial.particle_populations),
        _maximum_absolute(
            candidate.total_energy_populations - initial.total_energy_populations
        ),
    )
    return {
        "resolution": [resolution, resolution],
        "cell_spacing": list(prepared.cell_spacing),
        "time_step": prepared.required_step_size,
        "successful": bool(result.successful),
        "status": int(result.status),
        "transport_id": result.transport_id,
        "prepared_id": result.prepared_id,
        "maximum_population_change": maximum_change,
        "f": {
            "declared_moment_names": list(evidence.f.declared_moment_names),
            "source_population_integrals": _vector(
                evidence.f.source_population_integrals
            ),
            "target_population_integrals": _vector(
                evidence.f.target_population_integrals
            ),
            "population_integral_residuals": _vector(
                evidence.f.population_conservation_residual
            ),
            "maximum_absolute_population_integral_residual": float(
                evidence.f.maximum_absolute_population_residual
            ),
            "source_declared_moments": _vector(evidence.f.source_moments),
            "target_declared_moments": _vector(evidence.f.target_moments),
            "declared_moment_residuals": _vector(evidence.f.conservation_residual),
            "maximum_absolute_declared_moment_residual": float(
                evidence.f.maximum_absolute_residual
            ),
            "minimum_source_population": float(evidence.f.minimum_source_population),
            "minimum_target_population": float(evidence.f.minimum_target_population),
            "positivity_preserved": bool(evidence.f.positivity_preserved),
        },
        "g": {
            "declared_moment_names": list(evidence.g.declared_moment_names),
            "source_population_integrals": _vector(
                evidence.g.source_population_integrals
            ),
            "target_population_integrals": _vector(
                evidence.g.target_population_integrals
            ),
            "population_integral_residuals": _vector(
                evidence.g.population_conservation_residual
            ),
            "maximum_absolute_population_integral_residual": float(
                evidence.g.maximum_absolute_population_residual
            ),
            "source_declared_moments": _vector(evidence.g.source_moments),
            "target_declared_moments": _vector(evidence.g.target_moments),
            "declared_moment_residuals": _vector(evidence.g.conservation_residual),
            "maximum_absolute_declared_moment_residual": float(
                evidence.g.maximum_absolute_residual
            ),
            "minimum_source_population": float(evidence.g.minimum_source_population),
            "minimum_target_population": float(evidence.g.minimum_target_population),
            "positivity_preserved": bool(evidence.g.positivity_preserved),
        },
        "finite": bool(evidence.finite),
        "populations_nonnegative": bool(evidence.populations_nonnegative),
        "positivity_preserved": bool(evidence.positivity_preserved),
        "maximum_absolute_population_integral_residual": float(
            evidence.maximum_absolute_population_residual
        ),
        "maximum_absolute_declared_moment_residual": float(
            evidence.maximum_absolute_declared_moment_residual
        ),
        "direction": {
            "forward_f_fluctuation_relative_l2_error": (
                _population_fluctuation_relative_l2(
                    candidate.particle_populations,
                    forward_exact.particle_populations,
                )
            ),
            "forward_g_fluctuation_relative_l2_error": (
                _population_fluctuation_relative_l2(
                    candidate.total_energy_populations,
                    forward_exact.total_energy_populations,
                )
            ),
            "forward_combined_fluctuation_relative_l2_error": (
                _state_fluctuation_relative_l2(candidate, forward_exact)
            ),
            "reverse_f_fluctuation_relative_l2_error": (
                _population_fluctuation_relative_l2(
                    candidate.particle_populations,
                    reverse_exact.particle_populations,
                )
            ),
            "reverse_g_fluctuation_relative_l2_error": (
                _population_fluctuation_relative_l2(
                    candidate.total_energy_populations,
                    reverse_exact.total_energy_populations,
                )
            ),
            "reverse_combined_fluctuation_relative_l2_error": (
                _state_fluctuation_relative_l2(candidate, reverse_exact)
            ),
            "analytic_forward_map": "population_q(x-c_q*dt)",
            "analytic_reverse_discriminator": "population_q(x+c_q*dt)",
        },
    }


def _refinement_record(
    coarse: Mapping[str, Any], fine: Mapping[str, Any], /
) -> dict[str, float]:
    rows: dict[str, float] = {}
    for population in ("f", "g", "combined"):
        coarse_error = float(
            coarse["direction"][f"forward_{population}_fluctuation_relative_l2_error"]
        )
        fine_error = float(
            fine["direction"][f"forward_{population}_fluctuation_relative_l2_error"]
        )
        rows[f"coarse_{population}_fluctuation_relative_l2_error"] = coarse_error
        rows[f"fine_{population}_fluctuation_relative_l2_error"] = fine_error
        rows[f"{population}_fine_over_coarse_error_ratio"] = fine_error / coarse_error
        rows[f"{population}_observed_binary_refinement_order"] = math.log2(
            coarse_error / fine_error
        )
    return rows


def _gradient_case(
    prepared: PreparedCoupledD2V37OffLatticeTransport,
    state: SmoothCompressibleKineticState,
    /,
) -> dict[str, Any]:
    f = state.particle_populations
    g = state.total_energy_populations
    f_index = jnp.arange(f.size, dtype=f.dtype).reshape(f.shape)
    g_index = jnp.arange(g.size, dtype=g.dtype).reshape(g.shape)
    f_weight = 1.0 + 0.17 * jnp.sin(0.013 * f_index)
    g_weight = 0.8 + 0.19 * jnp.cos(0.011 * g_index)
    f_direction = 0.25 + 0.07 * jnp.cos(0.017 * f_index)
    g_direction = 0.30 + 0.06 * jnp.sin(0.019 * g_index)

    def objective(f_values: jax.Array, g_values: jax.Array) -> jax.Array:
        candidate = prepared.transport_with_evidence(
            SmoothCompressibleKineticState(f_values, g_values),
            prepared.required_step_size,
        ).candidate_state
        return jnp.sum(candidate.particle_populations * f_weight) + jnp.sum(
            candidate.total_energy_populations * g_weight
        )

    gradient_f, gradient_g = jax.grad(objective, argnums=(0, 1))(f, g)
    transported_direction = prepared.transport_with_evidence(
        SmoothCompressibleKineticState(f_direction, g_direction),
        prepared.required_step_size,
    ).candidate_state
    f_gradient_action = jnp.sum(gradient_f * f_direction)
    g_gradient_action = jnp.sum(gradient_g * g_direction)
    f_expected_action = jnp.sum(transported_direction.particle_populations * f_weight)
    g_expected_action = jnp.sum(transported_direction.total_energy_populations * g_weight)
    f_relative_error = jnp.abs(f_gradient_action - f_expected_action) / jnp.abs(
        f_expected_action
    )
    g_relative_error = jnp.abs(g_gradient_action - g_expected_action) / jnp.abs(
        g_expected_action
    )
    return {
        "operation": "reverse-mode gradient through coupled prepared transport",
        "f_gradient_shape": list(gradient_f.shape),
        "g_gradient_shape": list(gradient_g.shape),
        "input_shape": list(f.shape),
        "all_f_gradient_finite": bool(jnp.all(jnp.isfinite(gradient_f))),
        "all_g_gradient_finite": bool(jnp.all(jnp.isfinite(gradient_g))),
        "f_gradient_l2_norm": float(jnp.sqrt(jnp.sum(gradient_f**2))),
        "g_gradient_l2_norm": float(jnp.sqrt(jnp.sum(gradient_g**2))),
        "f_gradient_directional_action": float(f_gradient_action),
        "g_gradient_directional_action": float(g_gradient_action),
        "f_direct_linear_transport_action": float(f_expected_action),
        "g_direct_linear_transport_action": float(g_expected_action),
        "f_directional_relative_error": float(f_relative_error),
        "g_directional_relative_error": float(g_relative_error),
        "maximum_directional_relative_error": float(
            jnp.maximum(f_relative_error, g_relative_error)
        ),
    }


def _fixed_step_refusal(
    prepared: PreparedCoupledD2V37OffLatticeTransport,
    state: SmoothCompressibleKineticState,
    /,
) -> dict[str, Any]:
    supplied = 0.5 * prepared.required_step_size
    result = prepared.transport_with_evidence(state, supplied)
    return {
        "required_step_size": prepared.required_step_size,
        "supplied_step_size": supplied,
        "allows_step_reduction": prepared.allows_step_reduction,
        "successful": bool(result.successful),
        "status": int(result.status),
        "expected_status": int(CoupledD2V37TransportStatus.FIXED_STEP_MISMATCH),
        "evidence_successful": bool(result.evidence.successful),
        "evidence_status": int(result.evidence.status),
    }


def _run_d2v17_rejection_probe() -> None:
    with jax.enable_x64(True):
        material = IdealGasMaterial(1.4, 1.0)
        closure = ConstantTransport(
            PARTICLE_RELAXATION_TIME, TOTAL_ENERGY_RELAXATION_TIME
        )
        d2v17_method = SmoothCompressibleD2VKineticMethod(
            d2v17_quadrature(dtype=jnp.float64), material, closure
        )
        PreparedCoupledD2V37OffLatticeTransport.prepare(
            d2v17_method,
            (8, 8),
            (0.125, 0.125),
            0.125,
            conservation_tolerance=CONSERVATION_TOLERANCE,
        )


def _d2v17_rejection_record() -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.learned_energy_off_lattice_qualification",
            _D2V17_REJECTION_PROBE,
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    return {
        "probe": "PreparedCoupledD2V37OffLatticeTransport.prepare with D2V17 method",
        "executed_in_isolated_process": True,
        "return_code": completed.returncode,
        "uncaught_value_error_observed": "ValueError:" in completed.stderr,
        "expected_identity_rejection_observed": (
            _D2V17_REJECTION_MESSAGE in completed.stderr
        ),
        "expected_message": _D2V17_REJECTION_MESSAGE,
    }


def _compilation_record(timing: CompilationTiming, /) -> dict[str, float]:
    return {
        "lowering_seconds": float(timing.lowering_seconds),
        "compilation_seconds": float(timing.compilation_seconds),
    }


def _compiler_record(compiled: Any, /) -> dict[str, Any]:
    cost = compiled.cost_analysis()
    memory = compiled.memory_analysis()
    unavailable_reason = (
        "compiled executable did not provide memory analysis" if memory is None else None
    )
    evidence = compiler_evidence(
        cost,
        memory,
        source="jax-compiled-executable",
        unavailable_reason=unavailable_reason,
    )
    return {
        "flops": evidence.flops,
        "bytes_accessed": evidence.bytes_accessed,
        "argument_bytes": evidence.argument_bytes,
        "output_bytes": evidence.output_bytes,
        "temporary_bytes": evidence.temporary_bytes,
        "generated_code_bytes": evidence.generated_code_bytes,
        "estimated_device_memory_bytes": evidence.estimated_device_memory_bytes,
        "source": evidence.source,
        "unavailable_reason": evidence.unavailable_reason,
    }


def _paired_measurement(
    candidate: Callable[[], Any],
    reference: Callable[[], Any],
    /,
    *,
    warmups: int,
    repetitions: int,
) -> tuple[Any, DurationDistribution, Any, DurationDistribution]:
    for index in range(warmups):
        first, second = (
            (candidate, reference) if index % 2 == 0 else (reference, candidate)
        )
        measure_synchronized(first)
        measure_synchronized(second)

    candidate_samples: list[float] = []
    reference_samples: list[float] = []
    candidate_result = None
    reference_result = None
    for index in range(repetitions):
        if index % 2 == 0:
            candidate_result, candidate_seconds = measure_synchronized(candidate)
            reference_result, reference_seconds = measure_synchronized(reference)
        else:
            reference_result, reference_seconds = measure_synchronized(reference)
            candidate_result, candidate_seconds = measure_synchronized(candidate)
        candidate_samples.append(candidate_seconds)
        reference_samples.append(reference_seconds)
    if candidate_result is None or reference_result is None:
        raise RuntimeError("Paired runtime measurement produced no samples.")
    return (
        candidate_result,
        DurationDistribution(tuple(candidate_samples)),
        reference_result,
        DurationDistribution(tuple(reference_samples)),
    )


def _runtime_record(
    d2v37_method: SmoothCompressibleD2VKineticMethod,
    d2v17_method: SmoothCompressibleD2VKineticMethod,
    resolution: int,
    /,
    *,
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    d2v37 = _prepare_d2v37(d2v37_method, resolution)
    spacing = tuple(extent / resolution for extent in DOMAIN_EXTENT)
    d2v17 = D2V17PeriodicTransportPlan(
        d2v17_method.quadrature,
        (resolution, resolution),
        spacing,
        spacing[0],
    )
    d2v37_state = _analytic_state(d2v37_method.quadrature, resolution)
    d2v17_state = _analytic_state(d2v17_method.quadrature, resolution)

    def d2v37_operation(f: jax.Array, g: jax.Array) -> SmoothCompressibleKineticState:
        return d2v37.transport(
            SmoothCompressibleKineticState(f, g), d2v37.required_step_size
        )

    def d2v17_operation(f: jax.Array, g: jax.Array) -> SmoothCompressibleKineticState:
        return SmoothCompressibleKineticState(
            d2v17.transport(f),
            d2v17.transport(g),
        )

    d2v37_jit = jax.jit(d2v37_operation)
    d2v17_jit = jax.jit(d2v17_operation)
    d2v37_compiled, d2v37_compilation = measure_lower_and_compile(
        lambda: d2v37_jit.lower(
            d2v37_state.particle_populations,
            d2v37_state.total_energy_populations,
        ),
        lambda lowered: lowered.compile(),
    )
    d2v17_compiled, d2v17_compilation = measure_lower_and_compile(
        lambda: d2v17_jit.lower(
            d2v17_state.particle_populations,
            d2v17_state.total_energy_populations,
        ),
        lambda lowered: lowered.compile(),
    )
    (
        d2v37_result,
        d2v37_duration,
        d2v17_result,
        d2v17_duration,
    ) = _paired_measurement(
        lambda: d2v37_compiled(
            d2v37_state.particle_populations,
            d2v37_state.total_energy_populations,
        ),
        lambda: d2v17_compiled(
            d2v17_state.particle_populations,
            d2v17_state.total_energy_populations,
        ),
        warmups=warmups,
        repetitions=repetitions,
    )
    d2v37_median = d2v37_duration.median_seconds
    d2v17_median = d2v17_duration.median_seconds
    if (
        d2v37_median is None
        or d2v17_median is None
        or d2v37_median <= 0.0
        or d2v17_median <= 0.0
    ):
        raise RuntimeError("Runtime distributions require positive finite medians.")
    d2v37_compiler = _compiler_record(d2v37_compiled)
    d2v17_compiler = _compiler_record(d2v17_compiled)
    d2v37_device_memory = d2v37_compiler["estimated_device_memory_bytes"]
    d2v17_device_memory = d2v17_compiler["estimated_device_memory_bytes"]
    device_memory_ratio = (
        None
        if d2v37_device_memory is None or d2v17_device_memory is None
        else d2v37_device_memory / d2v17_device_memory
    )
    pair_ids = [f"paired-repetition-{index:02d}" for index in range(repetitions)]
    return {
        "collection": "interleaved pairs with alternating execution order",
        "warmups": warmups,
        "repetitions": repetitions,
        "pair_ids": pair_ids,
        "spatial_shape": [resolution, resolution],
        "dtype": str(d2v37_state.particle_populations.dtype),
        "candidate": {
            "operation": (
                "JIT PreparedCoupledD2V37OffLatticeTransport.transport on coupled f/g"
            ),
            "population_count_per_field": 37,
            "population_field_count": 2,
            "time_step": d2v37.required_step_size,
            "quadrature_id": d2v37.quadrature.quadrature_id,
            "prepared_id": d2v37.prepared_id,
            "logical_prepared_bytes": logical_array_bytes(d2v37),
            "logical_input_state_bytes": logical_array_bytes(d2v37_state),
            "logical_output_state_bytes": logical_array_bytes(d2v37_result),
            "compilation": _compilation_record(d2v37_compilation),
            "steady_state": d2v37_duration.to_seconds_dict(),
            "compiler": d2v37_compiler,
            "output_shape": list(d2v37_result.particle_populations.shape),
            "output_finite": bool(
                jnp.all(jnp.isfinite(d2v37_result.particle_populations))
                & jnp.all(jnp.isfinite(d2v37_result.total_energy_populations))
            ),
        },
        "reference": {
            "operation": (
                "JIT D2V17PeriodicTransportPlan.transport applied independently "
                "to f and g"
            ),
            "population_count_per_field": 17,
            "population_field_count": 2,
            "time_step": d2v17.time_step,
            "quadrature_id": d2v17.quadrature.quadrature_id,
            "plan_id": d2v17.plan_id,
            "logical_prepared_bytes": logical_array_bytes(d2v17),
            "logical_input_state_bytes": logical_array_bytes(d2v17_state),
            "logical_output_state_bytes": logical_array_bytes(d2v17_result),
            "compilation": _compilation_record(d2v17_compilation),
            "steady_state": d2v17_duration.to_seconds_dict(),
            "compiler": d2v17_compiler,
            "output_shape": list(d2v17_result.particle_populations.shape),
            "output_finite": bool(
                jnp.all(jnp.isfinite(d2v17_result.particle_populations))
                & jnp.all(jnp.isfinite(d2v17_result.total_energy_populations))
            ),
        },
        "comparison": {
            "same_spatial_shape": True,
            "same_dtype": bool(
                d2v37_state.particle_populations.dtype
                == d2v17_state.particle_populations.dtype
            ),
            "same_population_field_count": True,
            "same_population_count": False,
            "same_transport_semantics": False,
            "candidate_over_reference_median_runtime_ratio": (
                d2v37_median / d2v17_median
            ),
            "candidate_over_reference_compiler_device_memory_ratio": (
                device_memory_ratio
            ),
            "performance_claim": {
                "made": False,
                "statement": (
                    "no speed or memory-superiority claim; D2V37 conservative "
                    "bilinear transfers and D2V17 exact pulls have different Q "
                    "and semantics"
                ),
            },
        },
    }


def build_report(
    *,
    configuration: Mapping[str, Any],
    identities: Mapping[str, Any],
    fractional_shifts: Mapping[str, Any],
    constant_preservation: Mapping[str, Any],
    coarse_transport: Mapping[str, Any],
    fine_transport: Mapping[str, Any],
    refinement: Mapping[str, Any],
    gradient: Mapping[str, Any],
    fixed_step_refusal: Mapping[str, Any],
    d2v17_rejection: Mapping[str, Any],
    runtime: Mapping[str, Any],
    environment: Mapping[str, Any],
    thresholds: Mapping[str, float] = THRESHOLDS,
) -> dict[str, Any]:
    """Build canonical qualification evidence from explicit execution records."""

    limits = dict(thresholds)
    smooth_levels = (coarse_transport, fine_transport)
    moving_count = fractional_shifts["moving_population_count"]
    forward_to_reverse_ratios = tuple(
        level["direction"][f"forward_{population}_fluctuation_relative_l2_error"]
        / level["direction"][f"reverse_{population}_fluctuation_relative_l2_error"]
        for level in smooth_levels
        for population in ("f", "g", "combined")
    )
    candidate_runtime = runtime["candidate"]
    reference_runtime = runtime["reference"]
    candidate_memory = candidate_runtime["compiler"]["estimated_device_memory_bytes"]
    reference_memory = reference_runtime["compiler"]["estimated_device_memory_bytes"]
    gates = {
        "separate_d2v37_quadrature_and_model_identity": bool(
            identities["d2v37_quadrature"] != identities["d2v17_quadrature"]
            and identities["d2v37_kinetic_method"] != identities["d2v17_kinetic_method"]
            and identities["d2v37_kinetic_model"] == identities["d2v37_kinetic_method"]
            and identities["d2v17_kinetic_model"] == identities["d2v17_kinetic_method"]
            and identities["d2v37_program_manifest"]
            != identities["d2v17_program_manifest"]
            and identities["prepared_quadrature"] == identities["d2v37_quadrature"]
            and identities["prepared_method"] == identities["d2v37_kinetic_method"]
            and identities["d2v37_population_count"] == 37
            and identities["d2v17_population_count"] == 17
            and identities["d2v37_transport_kind"] == "off_lattice"
            and identities["d2v17_transport_kind"] == "integer_lattice"
            and identities["stage_one_or_d2v17_model_artifact"] is None
            and configuration["dtype"] == "float64"
            and configuration["deterministic"]
            and not configuration["step_reduction_allowed"]
        ),
        "periodic_fractional_departure_shifts": bool(
            fractional_shifts["transfer_count"] == 37
            and moving_count == 36
            and fractional_shifts["fractional_moving_population_count"] == moving_count
            and fractional_shifts["two_axis_fractional_population_count"] > 0
            and fractional_shifts["minimum_moving_distance_from_integer"] > 1.0e-12
            and fractional_shifts["maximum_prepared_offset_residual"] <= 1.0e-14
            and fractional_shifts["all_transfers_periodic"]
        ),
        "constant_preservation": bool(
            constant_preservation["successful"]
            and constant_preservation["status"]
            == int(CoupledD2V37TransportStatus.SUCCESS)
            and constant_preservation["maximum_f_population_error"]
            <= limits["maximum_constant_population_error"]
            and constant_preservation["maximum_g_population_error"]
            <= limits["maximum_constant_population_error"]
        ),
        "per_population_integral_conservation": bool(
            all(
                level["successful"]
                and level["maximum_absolute_population_integral_residual"]
                <= limits["maximum_population_integral_residual"]
                and level["f"]["maximum_absolute_population_integral_residual"]
                <= limits["maximum_population_integral_residual"]
                and level["g"]["maximum_absolute_population_integral_residual"]
                <= limits["maximum_population_integral_residual"]
                and len(level["f"]["population_integral_residuals"]) == 37
                and len(level["g"]["population_integral_residuals"]) == 37
                for level in smooth_levels
            )
        ),
        "declared_f_mass_momentum_and_g_energy_conservation": bool(
            all(
                level["f"]["declared_moment_names"]
                == ["mass", "momentum_x", "momentum_y"]
                and level["g"]["declared_moment_names"] == ["total_energy"]
                and level["f"]["maximum_absolute_declared_moment_residual"]
                <= limits["maximum_declared_moment_residual"]
                and level["g"]["maximum_absolute_declared_moment_residual"]
                <= limits["maximum_declared_moment_residual"]
                and level["maximum_absolute_declared_moment_residual"]
                <= limits["maximum_declared_moment_residual"]
                for level in smooth_levels
            )
        ),
        "positive_nontrivial_transport_execution": bool(
            all(
                level["successful"]
                and level["status"] == int(CoupledD2V37TransportStatus.SUCCESS)
                and level["finite"]
                and level["populations_nonnegative"]
                and level["positivity_preserved"]
                and level["f"]["minimum_source_population"] > limits["minimum_population"]
                and level["f"]["minimum_target_population"] > limits["minimum_population"]
                and level["g"]["minimum_source_population"] > limits["minimum_population"]
                and level["g"]["minimum_target_population"] > limits["minimum_population"]
                and level["maximum_population_change"] > 0.0
                for level in smooth_levels
            )
            and coarse_transport["transport_id"] == identities["transport"]
            and fine_transport["transport_id"] == identities["fine_transport"]
        ),
        "population_gradient": bool(
            gradient["f_gradient_shape"] == gradient["input_shape"]
            and gradient["g_gradient_shape"] == gradient["input_shape"]
            and gradient["all_f_gradient_finite"]
            and gradient["all_g_gradient_finite"]
            and gradient["f_gradient_l2_norm"] > 0.0
            and gradient["g_gradient_l2_norm"] > 0.0
            and gradient["maximum_directional_relative_error"]
            <= limits["maximum_gradient_directional_relative_error"]
        ),
        "forward_transport_direction": bool(
            all(
                ratio <= limits["maximum_forward_to_reverse_direction_error_ratio"]
                for ratio in forward_to_reverse_ratios
            )
        ),
        "binary_spatial_refinement": bool(
            all(
                refinement[f"fine_{population}_fluctuation_relative_l2_error"]
                <= limits["maximum_finest_fluctuation_relative_l2_error"]
                and refinement[f"{population}_fine_over_coarse_error_ratio"]
                <= limits["maximum_refinement_error_ratio"]
                and refinement[f"{population}_observed_binary_refinement_order"]
                >= limits["minimum_observed_binary_refinement_order"]
                for population in ("f", "g", "combined")
            )
        ),
        "fixed_step_rejection": bool(
            not fixed_step_refusal["allows_step_reduction"]
            and not fixed_step_refusal["successful"]
            and not fixed_step_refusal["evidence_successful"]
            and fixed_step_refusal["status"]
            == int(CoupledD2V37TransportStatus.FIXED_STEP_MISMATCH)
            and fixed_step_refusal["evidence_status"]
            == int(CoupledD2V37TransportStatus.FIXED_STEP_MISMATCH)
        ),
        "d2v17_identity_rejection": bool(
            d2v17_rejection["executed_in_isolated_process"]
            and d2v17_rejection["return_code"] != 0
            and d2v17_rejection["uncaught_value_error_observed"]
            and d2v17_rejection["expected_identity_rejection_observed"]
        ),
        "paired_runtime_and_memory_reporting": bool(
            runtime["repetitions"] == len(runtime["pair_ids"])
            and len(set(runtime["pair_ids"])) == runtime["repetitions"]
            and candidate_runtime["steady_state"]["count"] == runtime["repetitions"]
            and reference_runtime["steady_state"]["count"] == runtime["repetitions"]
            and candidate_runtime["output_finite"]
            and reference_runtime["output_finite"]
            and candidate_runtime["logical_prepared_bytes"] > 0
            and reference_runtime["logical_prepared_bytes"] > 0
            and candidate_runtime["logical_input_state_bytes"] > 0
            and reference_runtime["logical_input_state_bytes"] > 0
            and candidate_runtime["logical_output_state_bytes"] > 0
            and reference_runtime["logical_output_state_bytes"] > 0
            and candidate_runtime["quadrature_id"] == identities["d2v37_quadrature"]
            and reference_runtime["quadrature_id"] == identities["d2v17_quadrature"]
            and candidate_runtime["output_shape"] == runtime["spatial_shape"] + [37]
            and reference_runtime["output_shape"] == runtime["spatial_shape"] + [17]
            and candidate_memory is not None
            and reference_memory is not None
            and candidate_memory > 0
            and reference_memory > 0
            and runtime["comparison"]["same_spatial_shape"]
            and runtime["comparison"]["same_dtype"]
            and runtime["comparison"]["same_population_field_count"]
            and not runtime["comparison"]["same_population_count"]
            and not runtime["comparison"]["same_transport_semantics"]
            and not runtime["comparison"]["performance_claim"]["made"]
        ),
    }
    report = {
        "tool": "learned_energy_off_lattice_qualification",
        "scope": {
            "stage": "deterministic D2V37 off-lattice transport qualification",
            "claim": (
                "periodic conservative positive D2V37 off-lattice population "
                "transport only"
            ),
            "included": [
                "prepared periodic D2V37 bilinear departure transfers",
                "constant preservation",
                "per-population integral conservation",
                "declared f mass/momentum and g total-energy conservation",
                "positive-population preservation",
                "population differentiation",
                "analytic direction and binary spatial refinement",
                "fixed-step and D2V17 identity refusal",
                "descriptive runtime and compiler-memory comparison to D2V17 pulls",
            ],
            "excluded": [
                "boundary conditions",
                "shocks or finite-volume shock ownership",
                "high-Mach accuracy or stability",
                "collision or equilibrium accuracy",
                "integer streaming or D2V17 equivalence",
                "runtime or memory superiority",
                "production qualification",
            ],
        },
        "contracts": {
            "geometry": "two-dimensional periodic uniform grid",
            "transport": (
                "fixed-step positive conservative multilinear departure transfer"
            ),
            "step_reduction": "forbidden",
            "f_declared_moments": ["mass", "momentum_x", "momentum_y"],
            "g_declared_moments": ["total_energy"],
            "analytic_direction": "population_q(x-c_q*dt)",
            "refinement": "one fixed-Courant transport step on N and 2N grids",
            "runtime_reference": (
                "two independent D2V17 exact periodic pulls, one each for f and g"
            ),
            "runtime_interpretation": (
                "descriptive only because D2V37 and D2V17 differ in population count "
                "and transport semantics"
            ),
        },
        "configuration": dict(configuration),
        "identities": dict(identities),
        "thresholds": limits,
        "fractional_shifts": dict(fractional_shifts),
        "constant_preservation": dict(constant_preservation),
        "smooth_transport": {
            "coarse": dict(coarse_transport),
            "fine": dict(fine_transport),
            "refinement": dict(refinement),
        },
        "gradient": dict(gradient),
        "refusals": {
            "fixed_step": dict(fixed_step_refusal),
            "d2v17_identity": dict(d2v17_rejection),
        },
        "runtime_and_memory": dict(runtime),
        "environment": dict(environment),
        "gates": gates,
    }
    report["passed"] = all(gates.values())
    report["qualification_id"] = canonical_fingerprint(report)
    return report


def qualification_report(
    *,
    coarse_resolution: int = DEFAULT_COARSE_RESOLUTION,
    runtime_resolution: int = DEFAULT_RUNTIME_RESOLUTION,
    warmups: int = DEFAULT_WARMUPS,
    repetitions: int = DEFAULT_REPETITIONS,
) -> dict[str, Any]:
    """Execute deterministic D2V37 transport cases and return canonical evidence."""

    coarse_resolution_ = int(coarse_resolution)
    runtime_resolution_ = int(runtime_resolution)
    warmups_ = int(warmups)
    repetitions_ = int(repetitions)
    if coarse_resolution_ < 8:
        raise ValueError("coarse_resolution must be at least eight.")
    if runtime_resolution_ < 8:
        raise ValueError("runtime_resolution must be at least eight.")
    if warmups_ < 0 or repetitions_ < 1:
        raise ValueError("warmups must be nonnegative and repetitions must be positive.")

    d2v17_rejection = _d2v17_rejection_record()
    d2v37_method, d2v17_method, material, closure = _methods()
    coarse = _prepare_d2v37(d2v37_method, coarse_resolution_)
    fine_resolution = 2 * coarse_resolution_
    fine = _prepare_d2v37(d2v37_method, fine_resolution)
    fractional_shifts = _fractional_shift_record(coarse)
    constant_preservation = _constant_case(coarse)
    coarse_transport = _smooth_transport_case(coarse, coarse_resolution_)
    fine_transport = _smooth_transport_case(fine, fine_resolution)
    refinement = _refinement_record(coarse_transport, fine_transport)
    coarse_state = _analytic_state(d2v37_method.quadrature, coarse_resolution_)
    gradient = _gradient_case(coarse, coarse_state)
    fixed_step_refusal = _fixed_step_refusal(coarse, coarse_state)
    runtime = _runtime_record(
        d2v37_method,
        d2v17_method,
        runtime_resolution_,
        warmups=warmups_,
        repetitions=repetitions_,
    )
    identities = {
        "d2v37_quadrature": d2v37_method.quadrature.quadrature_id,
        "d2v37_kinetic_model": d2v37_method.method_id,
        "d2v37_kinetic_method": d2v37_method.method_id,
        "d2v37_program_manifest": d2v37_method.program_manifest.manifest_id,
        "d2v37_population_count": d2v37_method.quadrature.population_count,
        "d2v37_transport_kind": d2v37_method.quadrature.transport_kind,
        "d2v17_quadrature": d2v17_method.quadrature.quadrature_id,
        "d2v17_kinetic_model": d2v17_method.method_id,
        "d2v17_kinetic_method": d2v17_method.method_id,
        "d2v17_program_manifest": d2v17_method.program_manifest.manifest_id,
        "d2v17_population_count": d2v17_method.quadrature.population_count,
        "d2v17_transport_kind": d2v17_method.quadrature.transport_kind,
        "material": material.material_id,
        "transport_closure": closure.closure_id,
        "prepared_quadrature": coarse.quadrature.quadrature_id,
        "prepared_method": coarse.method_id,
        "population_transport": coarse.population_transport.prepared_id,
        "f_declared_moment_map": coarse.f_declared_moments.map_id,
        "g_declared_moment_map": coarse.g_declared_moments.map_id,
        "transport": coarse.transport_id,
        "prepared_transport": coarse.prepared_id,
        "fine_transport": fine.transport_id,
        "fine_prepared_transport": fine.prepared_id,
        "stage_one_or_d2v17_model_artifact": None,
    }
    configuration = {
        "deterministic": True,
        "dtype": str(d2v37_method.quadrature.velocities.dtype),
        "domain_extent": list(DOMAIN_EXTENT),
        "coarse_resolution": [coarse_resolution_, coarse_resolution_],
        "fine_resolution": [fine_resolution, fine_resolution],
        "coarse_cell_spacing": list(coarse.cell_spacing),
        "fine_cell_spacing": list(fine.cell_spacing),
        "coarse_time_step": coarse.required_step_size,
        "fine_time_step": fine.required_step_size,
        "fractional_courant": FRACTIONAL_COURANT,
        "conservation_tolerance": CONSERVATION_TOLERANCE,
        "step_reduction_allowed": coarse.allows_step_reduction,
        "runtime_resolution": [runtime_resolution_, runtime_resolution_],
        "runtime_warmups": warmups_,
        "runtime_repetitions": repetitions_,
    }
    return build_report(
        configuration=configuration,
        identities=identities,
        fractional_shifts=fractional_shifts,
        constant_preservation=constant_preservation,
        coarse_transport=coarse_transport,
        fine_transport=fine_transport,
        refinement=refinement,
        gradient=gradient,
        fixed_step_refusal=fixed_step_refusal,
        d2v17_rejection=d2v17_rejection,
        runtime=runtime,
        environment=capture_environment().to_dict(),
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Qualify deterministic periodic D2V37 off-lattice population transport."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="canonical JSON output written atomically only when every gate passes",
    )
    parser.add_argument(
        "--coarse-resolution", type=int, default=DEFAULT_COARSE_RESOLUTION
    )
    parser.add_argument(
        "--runtime-resolution", type=int, default=DEFAULT_RUNTIME_RESOLUTION
    )
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    return parser.parse_args()


def _validate_output_path(path: Path, /) -> Path:
    output = Path(path)
    if output.suffix != ".json":
        raise ValueError("Off-lattice qualification output must be a JSON artifact.")
    destination = output.resolve()
    protected = tuple(path.resolve() for path in _PROTECTED_ARTIFACT_PATHS)
    aliases_protected = destination in protected or (
        destination.exists()
        and any(
            candidate.exists() and destination.samefile(candidate)
            for candidate in protected
        )
    )
    if output.name in _PROTECTED_ARTIFACT_NAMES or aliases_protected:
        raise ValueError(
            "Off-lattice qualification cannot overwrite stage-one or D2V17 artifacts."
        )
    return output


def main() -> int:
    arguments = _parse_arguments()
    output = _validate_output_path(arguments.output)
    with jax.enable_x64(True):
        report = qualification_report(
            coarse_resolution=arguments.coarse_resolution,
            runtime_resolution=arguments.runtime_resolution,
            warmups=arguments.warmups,
            repetitions=arguments.repetitions,
        )
    if not report["passed"]:
        failed = sorted(name for name, passed in report["gates"].items() if not passed)
        print("failed qualification gates: " + ", ".join(failed), file=sys.stderr)
        return 1
    payload = canonical_json(report) + "\n"
    atomic_write(
        output,
        lambda temporary: temporary.write_text(payload, encoding="utf-8"),
    )
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == [_D2V17_REJECTION_PROBE]:
        _run_d2v17_rejection_probe()
        raise SystemExit(0)
    raise SystemExit(main())
