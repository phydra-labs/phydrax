#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical random-phase thermal pure quantum estimates in one fixed sector."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite, pi

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Key

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg import (
    AbstractLinearOperator,
    matrix_exponential_action,
    MatrixFunctionPolicy,
)
from ..operators.quantum.lattice import QuantumSectorOperator


class ThermalPureQuantumPlan(StrictModule):
    """Fixed beta, random probes, Lanczos policy, and explicit byte admission."""

    matrix_function: MatrixFunctionPolicy
    beta: float = eqx.field(static=True)
    probe_count: int = eqx.field(static=True)
    observable_count: int = eqx.field(static=True)
    maximum_retained_bytes: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        beta: float,
        /,
        *,
        probe_count: int,
        observable_count: int,
        matrix_function: MatrixFunctionPolicy,
        maximum_retained_bytes: int,
        maximum_workspace_bytes: int,
    ):
        beta_ = float(beta)
        probes = int(probe_count)
        observables = int(observable_count)
        byte_limit = int(maximum_retained_bytes)
        workspace_limit = int(maximum_workspace_bytes)
        if not isfinite(beta_) or beta_ < 0.0:
            raise ValueError("beta must be finite and non-negative.")
        if probes < 2 or observables < 0 or byte_limit < 1 or workspace_limit < 1:
            raise ValueError("TPQ probe/observable counts and byte limits are invalid.")
        if not isinstance(matrix_function, MatrixFunctionPolicy):
            raise TypeError("matrix_function must be MatrixFunctionPolicy.")
        if matrix_function.method != "lanczos":
            raise ValueError(
                "Canonical fixed-sector TPQ requires Lanczos matrix functions."
            )
        if matrix_function.differentiation.mode != "none":
            raise ValueError("Canonical TPQ requires a stopped Krylov AD boundary.")
        self.matrix_function = matrix_function
        self.beta = beta_
        self.probe_count = probes
        self.observable_count = observables
        self.maximum_retained_bytes = byte_limit
        self.maximum_workspace_bytes = workspace_limit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-sector-thermal-pure-quantum-plan",
                "beta": beta_,
                "probe_count": probes,
                "observable_count": observables,
                "maximum_retained_bytes": byte_limit,
                "maximum_workspace_bytes": workspace_limit,
                "matrix_function": {
                    "method": matrix_function.method,
                    "max_dimension": matrix_function.max_dimension,
                    "orthogonalization": matrix_function.orthogonalization,
                    "error_tolerance": matrix_function.error_tolerance,
                    "differentiation": matrix_function.differentiation.mode,
                },
            }
        )


class PreparedThermalPureQuantum(StrictModule):
    plan: ThermalPureQuantumPlan = eqx.field(static=True)
    hamiltonian: QuantumSectorOperator
    retained_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class ThermalPureQuantumResult(StrictModule):
    """Raw probes/thermal vectors and ratio-of-sums statistical evidence."""

    raw_probes: Array
    thermal_vectors: Array
    norm_weights: Array
    observable_numerators: Array
    partition_estimate: Array
    partition_standard_error: Array
    observable_estimates: Array
    observable_standard_errors: Array
    numerical_error_estimates: Array
    numerical_converged: Array
    valid: Array
    root_key: Array
    beta: float = eqx.field(static=True)
    hamiltonian_id: str = eqx.field(static=True)
    sector_basis_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


def prepare_thermal_pure_quantum(
    plan: ThermalPureQuantumPlan,
    hamiltonian: QuantumSectorOperator,
    /,
) -> PreparedThermalPureQuantum:
    if not isinstance(plan, ThermalPureQuantumPlan):
        raise TypeError("plan must be ThermalPureQuantumPlan.")
    if not isinstance(hamiltonian, QuantumSectorOperator):
        raise TypeError("hamiltonian must be QuantumSectorOperator.")
    if not hamiltonian.properties.certifies("self_adjoint"):
        raise ValueError(
            "TPQ requires a certified self-adjoint fixed-sector Hamiltonian."
        )
    dimension = hamiltonian.source.size
    complex_bytes = np.dtype(np.complex128).itemsize
    real_bytes = np.dtype(np.float64).itemsize
    retained = plan.probe_count * dimension * 2 * complex_bytes
    retained += plan.probe_count * (
        2 * real_bytes
        + np.dtype(np.bool_).itemsize
        + plan.observable_count * complex_bytes
    )
    retained += real_bytes + plan.observable_count * (complex_bytes + real_bytes)
    krylov_dimension = min(plan.matrix_function.max_dimension, dimension)
    workspace = (
        (krylov_dimension + 4) * dimension + (krylov_dimension + 1) * krylov_dimension
    ) * complex_bytes + hamiltonian.action_workspace_bytes
    if retained > plan.maximum_retained_bytes:
        raise ValueError(
            f"TPQ retained state requires {retained} bytes, exceeding "
            f"maximum_retained_bytes {plan.maximum_retained_bytes}."
        )
    if workspace > plan.maximum_workspace_bytes:
        raise ValueError(
            f"TPQ Krylov action requires {workspace} bytes, exceeding "
            f"maximum_workspace_bytes {plan.maximum_workspace_bytes}."
        )
    return PreparedThermalPureQuantum(
        plan=plan,
        hamiltonian=hamiltonian,
        retained_bytes=retained,
        workspace_bytes=workspace,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-fixed-sector-thermal-pure-quantum",
                "plan": plan.plan_id,
                "hamiltonian": hamiltonian.operator_id,
                "sector": hamiltonian.charge_map.source.basis_id,
            }
        ),
    )


def _validate_observables(
    prepared: PreparedThermalPureQuantum,
    observables: Sequence[AbstractLinearOperator],
    /,
) -> tuple[AbstractLinearOperator, ...]:
    values = tuple(observables)
    if len(values) != prepared.plan.observable_count:
        raise ValueError("observables must match plan.observable_count.")
    for observable in values:
        if not isinstance(observable, AbstractLinearOperator):
            raise TypeError("Every TPQ observable must be an AbstractLinearOperator.")
        if not observable.source.compatible(
            prepared.hamiltonian.source
        ) or not observable.target.compatible(prepared.hamiltonian.target):
            raise ValueError(
                "Every TPQ observable must be an endomorphism of the sector."
            )
    return values


def _jackknife_ratio_error(numerators: Array, denominators: Array, /) -> Array:
    count = denominators.shape[0]
    total_numerator = jnp.sum(numerators, axis=0)
    total_denominator = jnp.sum(denominators)
    leave_denominator = total_denominator - denominators
    safe = jnp.where(leave_denominator > 0.0, leave_denominator, 1.0)
    leave = (total_numerator[None, :] - numerators) / safe[:, None]
    center = jnp.mean(leave, axis=0)
    return jnp.sqrt((count - 1) / count * jnp.sum(jnp.abs(leave - center) ** 2, axis=0))


def thermal_pure_quantum(
    prepared: PreparedThermalPureQuantum,
    observables: Sequence[AbstractLinearOperator],
    /,
    *,
    key: Key[Array, ""],
) -> ThermalPureQuantumResult:
    """Execute deterministic semantic probe splitting and preserve every raw probe."""
    if not isinstance(prepared, PreparedThermalPureQuantum):
        raise TypeError("prepared must be PreparedThermalPureQuantum.")
    selected = _validate_observables(prepared, observables)
    root_key = jnp.asarray(key)
    dimension = prepared.hamiltonian.source.size
    probes = []
    thermal = []
    errors = []
    converged = []
    for index in range(prepared.plan.probe_count):
        probe_key = jr.fold_in(root_key, index)
        phases = jr.uniform(
            probe_key,
            (dimension,),
            minval=0.0,
            maxval=2.0 * pi,
            dtype=jnp.float64,
        )
        probe = jnp.exp(1.0j * phases).astype(jnp.complex128)
        action = matrix_exponential_action(
            prepared.hamiltonian,
            probe,
            -0.5 * prepared.plan.beta,
            policy=prepared.plan.matrix_function,
        )
        probes.append(probe)
        thermal.append(jnp.asarray(action.value))
        errors.append(action.diagnostics.error_estimate)
        converged.append(action.successful)
    raw_probes = jnp.stack(probes)
    thermal_vectors = jnp.stack(thermal)
    numerical_errors = jnp.stack(errors)
    numerical_converged = jnp.stack(converged)
    weights = jnp.stack(
        tuple(
            jnp.real(prepared.hamiltonian.source.inner(vector, vector))
            for vector in thermal_vectors
        )
    )
    numerators = []
    for vector in thermal_vectors:
        numerators.append(
            jnp.stack(
                tuple(
                    prepared.hamiltonian.source.inner(vector, observable.mv(vector))
                    for observable in selected
                )
            )
            if selected
            else jnp.zeros((0,), dtype=jnp.complex128)
        )
    raw_numerators = jnp.stack(numerators)
    total_weight = jnp.sum(weights)
    safe_weight = jnp.where(total_weight > 0.0, total_weight, 1.0)
    observable_estimates = jnp.sum(raw_numerators, axis=0) / safe_weight
    partition_estimate = jnp.mean(weights)
    partition_error = jnp.std(weights, ddof=1) / jnp.sqrt(prepared.plan.probe_count)
    observable_errors = _jackknife_ratio_error(raw_numerators, weights)
    finite = (
        jnp.all(jnp.isfinite(raw_probes))
        & jnp.all(jnp.isfinite(thermal_vectors))
        & jnp.all(jnp.isfinite(weights))
        & jnp.all(jnp.isfinite(raw_numerators))
    )
    valid = finite & jnp.all(numerical_converged) & jnp.all(weights > 0.0)
    return ThermalPureQuantumResult(
        raw_probes=raw_probes,
        thermal_vectors=thermal_vectors,
        norm_weights=weights,
        observable_numerators=raw_numerators,
        partition_estimate=partition_estimate,
        partition_standard_error=partition_error,
        observable_estimates=observable_estimates,
        observable_standard_errors=observable_errors,
        numerical_error_estimates=numerical_errors,
        numerical_converged=numerical_converged,
        valid=valid,
        root_key=root_key,
        beta=prepared.plan.beta,
        hamiltonian_id=prepared.hamiltonian.operator_id,
        sector_basis_id=prepared.hamiltonian.charge_map.source.basis_id,
        prepared_id=prepared.prepared_id,
    )


__all__ = [
    "PreparedThermalPureQuantum",
    "ThermalPureQuantumPlan",
    "ThermalPureQuantumResult",
    "prepare_thermal_pure_quantum",
    "thermal_pure_quantum",
]
