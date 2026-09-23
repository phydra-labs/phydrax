#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Matrix-free zero- and finite-temperature fixed-sector response workflows."""

from __future__ import annotations

from math import comb, pi

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._strict import StrictModule
from ..linalg import (
    matrix_exponential_action,
    ShiftedLinearSystemFamily,
    solve_shifted,
)
from ..operators.quantum._response import (
    FiniteTemperatureResponsePlan,
    QuantumSectorProbe,
    ZeroTemperatureResponsePlan,
)
from ..operators.quantum.lattice import QuantumSectorOperator
from ._thermal_pure_quantum import ThermalPureQuantumResult


class ZeroTemperatureResponseEvidence(StrictModule):
    source_norm_residual: Array
    source_eigen_residual: Array
    source_certified: Array
    moments: Array
    positivity_violation: Array
    nonnegative: Array
    shifted_status: Array
    shifted_residuals: Array
    valid: Array
    source_basis_id: str = eqx.field(static=True)
    target_basis_id: str = eqx.field(static=True)
    probe_id: str = eqx.field(static=True)


class ZeroTemperatureResponseResult(StrictModule):
    frequencies: Array
    retarded: Array
    spectral: Array
    source_vector: Array
    excited_source: Array
    evidence: ZeroTemperatureResponseEvidence


class FiniteTemperatureResponseEvidence(StrictModule):
    forward_moments: Array
    reverse_moments: Array
    forward_positivity_violation: Array
    reverse_positivity_violation: Array
    kms_residual: Array
    partition_ratio: Array
    maximum_numerical_error: Array
    numerical_converged: Array
    positivity_satisfied: Array
    kms_satisfied: Array
    valid: Array
    source_basis_id: str = eqx.field(static=True)
    target_basis_id: str = eqx.field(static=True)
    probe_id: str = eqx.field(static=True)


class FiniteTemperatureResponseResult(StrictModule):
    frequencies: Array
    times: Array
    forward_raw_correlations: Array
    reverse_raw_correlations: Array
    forward_numerical_error_estimates: Array
    reverse_numerical_error_estimates: Array
    forward_correlation: Array
    reverse_correlation: Array
    forward_correlation_standard_error: Array
    reverse_correlation_standard_error: Array
    forward_spectrum: Array
    reverse_spectrum: Array
    forward_spectrum_standard_error: Array
    reverse_spectrum_standard_error: Array
    evidence: FiniteTemperatureResponseEvidence


def _validate_channel(
    source: QuantumSectorOperator,
    target: QuantumSectorOperator,
    probe: QuantumSectorProbe,
    /,
) -> None:
    if not isinstance(source, QuantumSectorOperator) or not isinstance(
        target, QuantumSectorOperator
    ):
        raise TypeError(
            "source and target Hamiltonians must be QuantumSectorOperator values."
        )
    if not source.properties.certifies("self_adjoint") or not target.properties.certifies(
        "self_adjoint"
    ):
        raise ValueError("Response Hamiltonians must be certified self-adjoint.")
    if probe.source_basis_id != source.charge_map.source.basis_id:
        raise ValueError("Probe source sector does not match the source Hamiltonian.")
    if probe.target_basis_id != target.charge_map.source.basis_id:
        raise ValueError("Probe target sector does not match the target Hamiltonian.")


def _moments_ground(
    target: QuantumSectorOperator,
    excited: Array,
    source_energy: Array,
    count: int,
    /,
) -> Array:
    values = []
    current = excited
    for _ in range(count):
        values.append(target.source.inner(excited, current))
        current = target.mv(current) - source_energy * current
    return jnp.stack(values)


def zero_temperature_response(
    plan: ZeroTemperatureResponsePlan,
    source_hamiltonian: QuantumSectorOperator,
    target_hamiltonian: QuantumSectorOperator,
    probe: QuantumSectorProbe,
    ground_state: ArrayLike,
    ground_energy: ArrayLike,
    /,
) -> ZeroTemperatureResponseResult:
    """Evaluate <Aψ|(ω+E₀+iη-H_target)⁻¹|Aψ> through linalg shifted solve."""
    if not isinstance(plan, ZeroTemperatureResponsePlan):
        raise TypeError("plan must be ZeroTemperatureResponsePlan.")
    if not isinstance(probe, QuantumSectorProbe):
        raise TypeError("probe must be QuantumSectorProbe.")
    _validate_channel(source_hamiltonian, target_hamiltonian, probe)
    state = source_hamiltonian.source.validate(ground_state)
    energy = jnp.asarray(ground_energy, dtype=state.real.dtype)
    if energy.shape != () or jnp.iscomplexobj(energy):
        raise ValueError("ground_energy must be one real scalar.")
    norm_residual = jnp.abs(source_hamiltonian.source.inner(state, state) - 1.0)
    eigen_error = source_hamiltonian.mv(state) - energy * state
    eigen_residual = jnp.sqrt(
        jnp.real(source_hamiltonian.source.inner(eigen_error, eigen_error))
    )
    excited = probe.operator.mv(state)
    shifts = plan.frequencies.astype(jnp.complex128) + energy + 1.0j * plan.broadening
    family = ShiftedLinearSystemFamily(target_hamiltonian, shifts)
    shifted = solve_shifted(family, excited, policy=plan.shifted_solve)
    solutions = jnp.asarray(shifted.value)
    retarded = contract("i,wi->w", jnp.conj(excited), solutions)
    spectral = -jnp.imag(retarded) / pi
    positivity_violation = jnp.max(jnp.maximum(-spectral, 0.0))
    moments = _moments_ground(target_hamiltonian, excited, energy, plan.moment_count)
    finite = (
        jnp.all(jnp.isfinite(retarded))
        & jnp.all(jnp.isfinite(spectral))
        & jnp.all(jnp.isfinite(moments))
        & jnp.isfinite(norm_residual)
        & jnp.isfinite(eigen_residual)
    )
    nonnegative = positivity_violation <= plan.positivity_tolerance
    source_certified = (norm_residual <= plan.source_tolerance) & (
        eigen_residual <= plan.source_tolerance
    )
    evidence = ZeroTemperatureResponseEvidence(
        source_norm_residual=norm_residual,
        source_eigen_residual=eigen_residual,
        source_certified=source_certified,
        moments=moments,
        positivity_violation=positivity_violation,
        nonnegative=nonnegative,
        shifted_status=shifted.status,
        shifted_residuals=shifted.diagnostics.residual_norm,
        valid=finite & source_certified & nonnegative & jnp.all(shifted.successful),
        source_basis_id=probe.source_basis_id,
        target_basis_id=probe.target_basis_id,
        probe_id=probe.probe_id,
    )
    return ZeroTemperatureResponseResult(
        frequencies=plan.frequencies,
        retarded=retarded,
        spectral=spectral,
        source_vector=state,
        excited_source=excited,
        evidence=evidence,
    )


def _ratio_and_jackknife(numerators: Array, weights: Array, /) -> tuple[Array, Array]:
    count = weights.shape[0]
    total_numerator = jnp.sum(numerators, axis=0)
    total_weight = jnp.sum(weights)
    value = total_numerator / jnp.where(total_weight > 0.0, total_weight, 1.0)
    leave_weights = total_weight - weights
    leave = (total_numerator[None, ...] - numerators) / jnp.where(
        leave_weights > 0.0, leave_weights, 1.0
    ).reshape((count,) + (1,) * (numerators.ndim - 1))
    center = jnp.mean(leave, axis=0)
    error = jnp.sqrt(
        (count - 1) / count * jnp.sum(jnp.abs(leave - center[None, ...]) ** 2, axis=0)
    )
    return value, error


def _time_weights(times: Array, window: Array, /) -> Array:
    interior = 0.5 * (times[2:] - times[:-2])
    trapezoid = jnp.concatenate(
        (
            (0.5 * (times[1] - times[0]))[None],
            interior,
            (0.5 * (times[-1] - times[-2]))[None],
        )
    )
    return trapezoid * window


def _spectrum(raw_correlation: Array, weights: Array, plan, /) -> Array:
    phases = jnp.exp(1.0j * plan.frequencies[:, None] * plan.times[None, :])
    return jnp.real(contract("wt,...t,t->...w", phases, raw_correlation, weights)) / (
        2.0 * pi
    )


def _thermal_channel_moments(
    source_hamiltonian: QuantumSectorOperator,
    target_hamiltonian: QuantumSectorOperator,
    probe_action,
    thermal_vectors: Array,
    norm_weights: Array,
    count: int,
    /,
) -> Array:
    raw = []
    for thermal in thermal_vectors:
        source_powers = [thermal]
        for _ in range(1, count):
            source_powers.append(source_hamiltonian.mv(source_powers[-1]))
        mapped = [probe_action(value) for value in source_powers]
        target_powers = [mapped[0]]
        for _ in range(1, count):
            target_powers.append(target_hamiltonian.mv(target_powers[-1]))
        moments = []
        for order in range(count):
            value = jnp.asarray(0.0 + 0.0j)
            for source_order in range(order + 1):
                value = value + (
                    (-1) ** source_order
                    * comb(order, source_order)
                    * target_hamiltonian.source.inner(
                        mapped[source_order], target_powers[order - source_order]
                    )
                )
            moments.append(value)
        raw.append(jnp.stack(moments))
    return jnp.sum(jnp.stack(raw), axis=0) / jnp.sum(norm_weights)


def finite_temperature_response(
    plan: FiniteTemperatureResponsePlan,
    source_hamiltonian: QuantumSectorOperator,
    target_hamiltonian: QuantumSectorOperator,
    probe: QuantumSectorProbe,
    source_tpq: ThermalPureQuantumResult,
    target_tpq: ThermalPureQuantumResult,
    /,
) -> FiniteTemperatureResponseResult:
    """TPQ time correlation in both charge sectors with canonical KMS evidence."""
    if not isinstance(plan, FiniteTemperatureResponsePlan):
        raise TypeError("plan must be FiniteTemperatureResponsePlan.")
    if not isinstance(probe, QuantumSectorProbe):
        raise TypeError("probe must be QuantumSectorProbe.")
    if not isinstance(source_tpq, ThermalPureQuantumResult) or not isinstance(
        target_tpq, ThermalPureQuantumResult
    ):
        raise TypeError("source_tpq and target_tpq must be ThermalPureQuantumResult.")
    _validate_channel(source_hamiltonian, target_hamiltonian, probe)
    if source_tpq.sector_basis_id != probe.source_basis_id:
        raise ValueError("Source TPQ result belongs to a different sector.")
    if target_tpq.sector_basis_id != probe.target_basis_id:
        raise ValueError("Target TPQ result belongs to a different sector.")
    if source_tpq.hamiltonian_id != source_hamiltonian.operator_id:
        raise ValueError("Source TPQ result belongs to a different Hamiltonian.")
    if target_tpq.hamiltonian_id != target_hamiltonian.operator_id:
        raise ValueError("Target TPQ result belongs to a different Hamiltonian.")
    if source_tpq.beta != target_tpq.beta:
        raise ValueError("Source and target TPQ results must use the same beta.")
    complex_bytes = np.dtype(np.complex128).itemsize
    action_workspaces = []
    for hamiltonian in (source_hamiltonian, target_hamiltonian):
        dimension = hamiltonian.source.size
        krylov_dimension = min(plan.matrix_function.max_dimension, dimension)
        action_workspaces.append(
            (
                (krylov_dimension + 8) * dimension
                + (krylov_dimension + 1) * krylov_dimension
            )
            * complex_bytes
            + hamiltonian.action_workspace_bytes
        )
    workspace = (
        max(action_workspaces)
        + 4
        * (source_hamiltonian.source.size + target_hamiltonian.source.size)
        * complex_bytes
    )
    if workspace > plan.maximum_workspace_bytes:
        raise ValueError(
            f"Finite-temperature response requires {workspace} workspace bytes, "
            f"exceeding maximum_workspace_bytes {plan.maximum_workspace_bytes}."
        )
    probe_time_count = (
        source_tpq.thermal_vectors.shape[0] + target_tpq.thermal_vectors.shape[0]
    ) * plan.times.size
    raw_bytes = probe_time_count * (
        np.dtype(np.complex128).itemsize
        + np.dtype(np.float64).itemsize
        + np.dtype(np.bool_).itemsize
    )
    if raw_bytes + plan.fixed_result_bytes > plan.maximum_result_bytes:
        raise ValueError(
            "Raw finite-temperature correlations exceed maximum_result_bytes."
        )

    forward_rows = []
    forward_converged = []
    forward_errors = []
    for thermal in source_tpq.thermal_vectors:
        left = probe.operator.mv(thermal)
        row = []
        flags = []
        errors = []
        for time in plan.times:
            source_action = matrix_exponential_action(
                source_hamiltonian,
                thermal,
                1.0j * time,
                policy=plan.matrix_function,
            )
            mapped = probe.operator.mv(source_action.value)
            target_action = matrix_exponential_action(
                target_hamiltonian,
                mapped,
                -1.0j * time,
                policy=plan.matrix_function,
            )
            row.append(target_hamiltonian.source.inner(left, target_action.value))
            flags.append(source_action.successful & target_action.successful)
            errors.append(source_action.diagnostics.error_estimate + target_action.diagnostics.error_estimate)
        forward_rows.append(jnp.stack(row))
        forward_converged.append(jnp.stack(flags))
        forward_errors.append(jnp.stack(errors))

    reverse_rows = []
    reverse_converged = []
    reverse_errors = []
    for thermal in target_tpq.thermal_vectors:
        left = probe.operator.adjoint_mv(thermal)
        row = []
        flags = []
        errors = []
        for time in plan.times:
            target_action = matrix_exponential_action(
                target_hamiltonian,
                thermal,
                1.0j * time,
                policy=plan.matrix_function,
            )
            mapped = probe.operator.adjoint_mv(target_action.value)
            source_action = matrix_exponential_action(
                source_hamiltonian,
                mapped,
                -1.0j * time,
                policy=plan.matrix_function,
            )
            row.append(source_hamiltonian.source.inner(left, source_action.value))
            flags.append(target_action.successful & source_action.successful)
            errors.append(target_action.diagnostics.error_estimate + source_action.diagnostics.error_estimate)
        reverse_rows.append(jnp.stack(row))
        reverse_converged.append(jnp.stack(flags))
        reverse_errors.append(jnp.stack(errors))

    forward_raw = jnp.stack(forward_rows)
    reverse_raw = jnp.stack(reverse_rows)
    forward_numerical_errors = jnp.stack(forward_errors)
    reverse_numerical_errors = jnp.stack(reverse_errors)
    forward_correlation, forward_correlation_error = _ratio_and_jackknife(
        forward_raw, source_tpq.norm_weights
    )
    reverse_correlation, reverse_correlation_error = _ratio_and_jackknife(
        reverse_raw, target_tpq.norm_weights
    )
    integration_weights = _time_weights(plan.times, plan.window)
    forward_raw_spectrum = _spectrum(forward_raw, integration_weights, plan)
    reverse_raw_spectrum = _spectrum(reverse_raw, integration_weights, plan)
    forward_spectrum, forward_spectrum_error = _ratio_and_jackknife(
        forward_raw_spectrum, source_tpq.norm_weights
    )
    reverse_spectrum, reverse_spectrum_error = _ratio_and_jackknife(
        reverse_raw_spectrum, target_tpq.norm_weights
    )
    forward_moments = _thermal_channel_moments(
        source_hamiltonian,
        target_hamiltonian,
        probe.operator.mv,
        source_tpq.thermal_vectors,
        source_tpq.norm_weights,
        plan.moment_count,
    )
    reverse_moments = _thermal_channel_moments(
        target_hamiltonian,
        source_hamiltonian,
        probe.operator.adjoint_mv,
        target_tpq.thermal_vectors,
        target_tpq.norm_weights,
        plan.moment_count,
    )
    partition_ratio = source_tpq.partition_estimate / target_tpq.partition_estimate
    kms_expected = (
        partition_ratio * jnp.exp(-source_tpq.beta * plan.frequencies) * forward_spectrum
    )
    kms_difference = reverse_spectrum[::-1] - kms_expected
    kms_scale = jnp.maximum(
        jnp.maximum(jnp.abs(reverse_spectrum[::-1]), jnp.abs(kms_expected)), 1.0
    )
    kms_residual = jnp.max(jnp.abs(kms_difference) / kms_scale)
    forward_violation = jnp.max(jnp.maximum(-forward_spectrum, 0.0))
    reverse_violation = jnp.max(jnp.maximum(-reverse_spectrum, 0.0))
    numerical_converged = (
        jnp.all(jnp.stack(forward_converged))
        & jnp.all(jnp.stack(reverse_converged))
        & source_tpq.valid
        & target_tpq.valid
    )
    maximum_numerical_error = jnp.maximum(
        jnp.max(forward_numerical_errors), jnp.max(reverse_numerical_errors)
    )
    positivity = (forward_violation <= plan.positivity_tolerance) & (
        reverse_violation <= plan.positivity_tolerance
    )
    kms_satisfied = kms_residual <= plan.kms_tolerance
    finite = (
        jnp.all(jnp.isfinite(forward_raw))
        & jnp.all(jnp.isfinite(reverse_raw))
        & jnp.all(jnp.isfinite(forward_spectrum))
        & jnp.all(jnp.isfinite(reverse_spectrum))
        & jnp.all(jnp.isfinite(forward_moments))
        & jnp.all(jnp.isfinite(reverse_moments))
        & jnp.isfinite(kms_residual)
        & jnp.all(jnp.isfinite(forward_numerical_errors))
        & jnp.all(jnp.isfinite(reverse_numerical_errors))
    )
    evidence = FiniteTemperatureResponseEvidence(
        forward_moments=forward_moments,
        reverse_moments=reverse_moments,
        forward_positivity_violation=forward_violation,
        reverse_positivity_violation=reverse_violation,
        kms_residual=kms_residual,
        partition_ratio=partition_ratio,
        numerical_converged=numerical_converged,
        maximum_numerical_error=maximum_numerical_error,
        positivity_satisfied=positivity,
        kms_satisfied=kms_satisfied,
        valid=finite & numerical_converged & positivity & kms_satisfied,
        source_basis_id=probe.source_basis_id,
        target_basis_id=probe.target_basis_id,
        probe_id=probe.probe_id,
    )
    return FiniteTemperatureResponseResult(
        frequencies=plan.frequencies,
        times=plan.times,
        forward_raw_correlations=forward_raw,
        reverse_raw_correlations=reverse_raw,
        forward_numerical_error_estimates=forward_numerical_errors,
        reverse_numerical_error_estimates=reverse_numerical_errors,
        forward_correlation=forward_correlation,
        reverse_correlation=reverse_correlation,
        forward_correlation_standard_error=forward_correlation_error,
        reverse_correlation_standard_error=reverse_correlation_error,
        forward_spectrum=forward_spectrum,
        reverse_spectrum=reverse_spectrum,
        forward_spectrum_standard_error=forward_spectrum_error,
        reverse_spectrum_standard_error=reverse_spectrum_error,
        evidence=evidence,
    )


__all__ = [
    "FiniteTemperatureResponseEvidence",
    "FiniteTemperatureResponseResult",
    "ZeroTemperatureResponseEvidence",
    "ZeroTemperatureResponseResult",
    "finite_temperature_response",
    "zero_temperature_response",
]
