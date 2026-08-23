#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..solver._memory_kernel import DynamicalMapPhysicality


def _superoperator_from_kraus(kraus: Array) -> Array:
    dimension = kraus.shape[-1]
    basis = jnp.eye(dimension * dimension, dtype=kraus.dtype).reshape(
        (dimension * dimension, dimension, dimension)
    )
    outputs = jnp.stack(
        [
            sum(
                operator @ value @ jnp.conj(operator.T)
                for operator in kraus
            ).reshape(-1)
            for value in basis
        ]
    )
    return jnp.swapaxes(outputs, -1, -2)


class QuantumIntervention(StrictModule):
    kraus_operators: Array
    superoperator: Array
    completeness_residual: Array
    trace_nonincreasing: Array
    valid: Array
    intervention_id: str = eqx.field(static=True)

    def __init__(self, kraus_operators: ArrayLike, /, *, intervention_id: str):
        kraus = jnp.asarray(kraus_operators)
        if kraus.ndim != 3 or kraus.shape[-2] != kraus.shape[-1]:
            raise ValueError("Kraus operators require shape (count,n,n).")
        dimension = kraus.shape[-1]
        completeness = sum(jnp.conj(operator.T) @ operator for operator in kraus)
        difference = jnp.eye(dimension, dtype=kraus.dtype) - completeness
        minimum = jnp.min(jnp.linalg.eigvalsh(0.5 * (difference + jnp.conj(difference.T))))
        self.kraus_operators = kraus
        self.superoperator = _superoperator_from_kraus(kraus)
        self.completeness_residual = jnp.max(jnp.abs(completeness - jnp.eye(dimension)))
        self.trace_nonincreasing = minimum >= -1e-9
        self.valid = jnp.all(jnp.isfinite(kraus)) & self.trace_nonincreasing
        self.intervention_id = str(intervention_id)

    @property
    def dimension(self) -> int:
        return int(self.kraus_operators.shape[-1])

    def apply(self, density: ArrayLike, /) -> tuple[Array, Array]:
        rho = jnp.asarray(density)
        output = sum(
            operator @ rho @ jnp.conj(operator.T)
            for operator in self.kraus_operators
        )
        probability = jnp.real(jnp.trace(output))
        normalized = jnp.where(probability > 0.0, output / probability, output)
        return normalized, probability


class ProcessTensorPhysicality(StrictModule):
    local_cp_margins: Array
    local_tp_residuals: Array
    causality_residual: Array
    valid: Array
    status: str = eqx.field(static=True)

    def __init__(
        self,
        local_cp_margins: ArrayLike,
        local_tp_residuals: ArrayLike,
        /,
        *,
        status: str,
    ):
        self.local_cp_margins = jnp.asarray(local_cp_margins)
        self.local_tp_residuals = jnp.asarray(local_tp_residuals)
        self.causality_residual = jnp.max(self.local_tp_residuals)
        self.valid = (
            jnp.all(self.local_cp_margins >= -1e-8)
            & jnp.all(self.local_tp_residuals <= 1e-8)
        )
        self.status = str(status)


class ProcessTensorMPO(StrictModule):
    tensors: tuple[Array, ...]
    initial_density: Array
    dimension: int = eqx.field(static=True)
    temporal_bond_dimensions: tuple[int, ...] = eqx.field(static=True)
    process_id: str = eqx.field(static=True)

    def __init__(
        self,
        tensors: Sequence[ArrayLike],
        initial_density: ArrayLike,
        /,
        *,
        process_id: str,
    ):
        values = tuple(jnp.asarray(tensor) for tensor in tensors)
        density = jnp.asarray(initial_density)
        if density.ndim != 2 or density.shape[0] != density.shape[1]:
            raise ValueError("Initial process state must be square.")
        physical = density.size
        if not values or any(
            tensor.ndim != 4 or tensor.shape[1:3] != (physical, physical)
            for tensor in values
        ):
            raise ValueError("Process tensors require (left,out,in,right) tensors.")
        if values[0].shape[0] != 1 or values[-1].shape[-1] != 1:
            raise ValueError("Process-tensor edge bonds must be one.")
        for left, right in zip(values[:-1], values[1:], strict=True):
            if left.shape[-1] != right.shape[0]:
                raise ValueError("Adjacent process-tensor bonds must match.")
        self.tensors = values
        self.initial_density = density
        self.dimension = density.shape[0]
        self.temporal_bond_dimensions = tuple(
            int(tensor.shape[-1]) for tensor in values[:-1]
        )
        self.process_id = str(process_id)

    def contract(
        self,
        interventions: Sequence[QuantumIntervention] | None = None,
        /,
    ) -> tuple[Array, Array]:
        operations = (
            tuple(interventions)
            if interventions is not None
            else tuple(
                QuantumIntervention(
                    jnp.eye(self.dimension, dtype=self.initial_density.dtype)[None, ...],
                    intervention_id=f"identity:{index}",
                )
                for index in range(len(self.tensors))
            )
        )
        if len(operations) != len(self.tensors):
            raise ValueError("One intervention is required per process time slot.")
        state = self.initial_density.reshape((1, -1))
        probability = jnp.asarray(1.0)
        for tensor, intervention in zip(self.tensors, operations, strict=True):
            state = jnp.einsum("li,loir->ro", state, tensor)
            state = jnp.einsum("oi,ri->ro", intervention.superoperator, state)
            density = state.sum(axis=0).reshape((self.dimension, self.dimension))
            trace = jnp.real(jnp.trace(density))
            probability = probability * trace
            state = state / jnp.maximum(trace, 1e-30)
        return state[0].reshape((self.dimension, self.dimension)), probability

    def physicality(self) -> ProcessTensorPhysicality:
        if any(tensor.shape[0] != 1 or tensor.shape[-1] != 1 for tensor in self.tensors):
            return ProcessTensorPhysicality(
                jnp.asarray([jnp.nan]),
                jnp.asarray([jnp.nan]),
                status="unknown-general-temporal-bond",
            )
        reports = [
            DynamicalMapPhysicality(tensor[0, :, :, 0], self.dimension)
            for tensor in self.tensors
        ]
        return ProcessTensorPhysicality(
            jnp.stack([report.cp_margin for report in reports]),
            jnp.stack([report.trace_preservation_residual for report in reports]),
            status="local-markov-factorization",
        )


def markov_process_tensor(
    superoperators: Sequence[ArrayLike],
    initial_density: ArrayLike,
    /,
    *,
    process_id: str = "markov-process",
) -> ProcessTensorMPO:
    tensors = tuple(jnp.asarray(operator)[None, :, :, None] for operator in superoperators)
    return ProcessTensorMPO(tensors, initial_density, process_id=process_id)


class ProcessTomographyResult(StrictModule):
    process_tensor: ProcessTensorMPO
    reconstruction_residual: Array
    valid: Array

    def __init__(
        self,
        process_tensor: ProcessTensorMPO,
        reconstruction_residual: ArrayLike,
        /,
    ):
        self.process_tensor = process_tensor
        self.reconstruction_residual = jnp.asarray(reconstruction_residual)
        self.valid = (
            process_tensor.physicality().valid
            & jnp.isfinite(self.reconstruction_residual)
        )


def reconstruct_markov_process_tensor(
    observed_superoperators: Sequence[ArrayLike],
    initial_density: ArrayLike,
    /,
    *,
    process_id: str = "reconstructed-markov-process",
) -> ProcessTomographyResult:
    process = markov_process_tensor(
        observed_superoperators, initial_density, process_id=process_id
    )
    residual = jnp.max(
        jnp.stack(
            [
                jnp.linalg.norm(
                    process.tensors[index][0, :, :, 0]
                    - jnp.asarray(observed_superoperators[index])
                )
                for index in range(len(process.tensors))
            ]
        )
    )
    return ProcessTomographyResult(process, residual)


__all__ = [
    "ProcessTensorMPO",
    "ProcessTensorPhysicality",
    "ProcessTomographyResult",
    "QuantumIntervention",
    "markov_process_tensor",
    "reconstruct_markov_process_tensor",
]
