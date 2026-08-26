#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._geometry_precision import GeometryPrecisionPolicy
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ..linalg import HermitianPrecisionPolicy, HermitianSpectrum
from ..solver._memory_kernel import DynamicalMapPhysicality
from ._precision import TensorNetworkPrecisionPolicy


def _superoperator_from_kraus(
    kraus: Array,
    precision: TensorNetworkPrecisionPolicy,
    /,
) -> Array:
    dimension = kraus.shape[-1]
    basis = jnp.eye(dimension * dimension, dtype=kraus.dtype).reshape(
        (dimension * dimension, dimension, dimension)
    )
    operators = precision.contraction(kraus)
    outputs = jnp.stack(
        [
            precision.sum(
                jnp.stack(
                    [operator @ value @ jnp.conj(operator.T) for operator in operators]
                ),
                axis=0,
            ).reshape(-1)
            for value in basis
        ]
    )
    return precision.output(jnp.swapaxes(outputs, -1, -2))


class QuantumIntervention(StrictModule):
    kraus_operators: Array
    superoperator: Array
    completeness_residual: Array
    trace_nonincreasing: Array
    valid: Array
    precision: TensorNetworkPrecisionPolicy
    hermitian_precision: HermitianPrecisionPolicy
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    intervention_id: str = eqx.field(static=True)

    def __init__(
        self,
        kraus_operators: ArrayLike,
        /,
        *,
        intervention_id: str,
        precision: TensorNetworkPrecisionPolicy | None = None,
        hermitian_precision: HermitianPrecisionPolicy | None = None,
    ):
        precision_ = TensorNetworkPrecisionPolicy() if precision is None else precision
        hermitian_ = (
            HermitianPrecisionPolicy()
            if hermitian_precision is None
            else hermitian_precision
        )
        if not isinstance(precision_, TensorNetworkPrecisionPolicy):
            raise TypeError("precision must be TensorNetworkPrecisionPolicy or None.")
        if not isinstance(hermitian_, HermitianPrecisionPolicy):
            raise TypeError(
                "hermitian_precision must be HermitianPrecisionPolicy or None."
            )
        kraus = precision_.storage(jnp.asarray(kraus_operators))
        precision_.validate_storage(kraus)
        if kraus.ndim != 3 or kraus.shape[-2] != kraus.shape[-1]:
            raise ValueError("Kraus operators require shape (count,n,n).")
        dimension = kraus.shape[-1]
        operators = precision_.contraction(kraus)
        completeness = precision_.sum(
            jnp.stack([jnp.conj(operator.T) @ operator for operator in operators]),
            axis=0,
        )
        difference = jnp.eye(dimension, dtype=kraus.dtype) - completeness
        spectrum = HermitianSpectrum(difference, precision=hermitian_)
        minimum = precision_.decision(spectrum.minimum_eigenvalue)
        residual = precision_.decision(
            jnp.max(
                jnp.abs(
                    precision_.accumulation(
                        completeness - jnp.eye(dimension, dtype=kraus.dtype)
                    )
                )
            )
        )
        self.kraus_operators = kraus
        self.superoperator = _superoperator_from_kraus(kraus, precision_)
        self.completeness_residual = residual
        self.trace_nonincreasing = minimum >= -1e-9
        self.valid = (
            jnp.all(jnp.isfinite(kraus)) & spectrum.valid & self.trace_nonincreasing
        )
        self.precision = precision_
        self.hermitian_precision = hermitian_
        self.precision_evidence = precision_.evidence_for(
            kraus,
            children={"completeness-spectrum": spectrum.precision_evidence},
            output_value=self.superoperator,
        )
        self.intervention_id = str(intervention_id)

    @property
    def dimension(self) -> int:
        return int(self.kraus_operators.shape[-1])

    def apply(self, density: ArrayLike, /) -> tuple[Array, Array]:
        rho = self.precision.contraction(density)
        output = self.precision.sum(
            jnp.stack(
                [
                    operator @ rho @ jnp.conj(operator.T)
                    for operator in self.precision.contraction(self.kraus_operators)
                ]
            ),
            axis=0,
        )
        probability = self.precision.decision(
            jnp.real(self.precision.sum(jnp.diag(output)))
        )
        normalized = jnp.where(probability > 0.0, output / probability, output)
        return self.precision.output(normalized), probability


class ProcessTensorPhysicality(StrictModule):
    local_cp_margins: Array
    local_tp_residuals: Array
    initial_state_valid: Array
    causality_residual: Array
    valid: Array
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    status: str = eqx.field(static=True)

    def __init__(
        self,
        local_cp_margins: ArrayLike,
        local_tp_residuals: ArrayLike,
        initial_state_valid: ArrayLike,
        causality_residual: ArrayLike,
        /,
        *,
        status: str,
        precision_evidence: PrecisionEvidenceEnvelope,
    ):
        if not isinstance(precision_evidence, PrecisionEvidenceEnvelope):
            raise TypeError("precision_evidence must be PrecisionEvidenceEnvelope.")
        cp_margins = jnp.asarray(local_cp_margins)
        tp_residuals = jnp.asarray(local_tp_residuals)
        initial_valid = jnp.asarray(initial_state_valid)
        causality = jnp.asarray(causality_residual)
        if initial_valid.shape != () or initial_valid.dtype != jnp.bool_:
            raise TypeError("initial_state_valid must be one scalar Boolean.")
        if causality.shape != ():
            raise ValueError("causality_residual must be one scalar.")
        self.local_cp_margins = cp_margins
        self.local_tp_residuals = tp_residuals
        self.initial_state_valid = initial_valid
        self.causality_residual = causality
        self.valid = (
            jnp.all(jnp.isfinite(cp_margins))
            & jnp.all(jnp.isfinite(tp_residuals))
            & jnp.isfinite(causality)
            & jnp.all(cp_margins >= -1e-8)
            & jnp.all(tp_residuals <= 1e-8)
            & (causality <= 1e-8)
            & initial_valid
        )
        self.precision_evidence = precision_evidence
        self.status = str(status)


class ProcessTensorMPO(StrictModule):
    tensors: tuple[Array, ...]
    initial_density: Array
    precision: TensorNetworkPrecisionPolicy
    geometry_precision: GeometryPrecisionPolicy
    hermitian_precision: HermitianPrecisionPolicy
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
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
        precision: TensorNetworkPrecisionPolicy | None = None,
        geometry_precision: GeometryPrecisionPolicy | None = None,
        hermitian_precision: HermitianPrecisionPolicy | None = None,
    ):
        precision_ = TensorNetworkPrecisionPolicy() if precision is None else precision
        geometry_ = (
            GeometryPrecisionPolicy()
            if geometry_precision is None
            else geometry_precision
        )
        hermitian_ = (
            HermitianPrecisionPolicy()
            if hermitian_precision is None
            else hermitian_precision
        )
        if not isinstance(precision_, TensorNetworkPrecisionPolicy):
            raise TypeError("precision must be TensorNetworkPrecisionPolicy or None.")
        if not isinstance(geometry_, GeometryPrecisionPolicy):
            raise TypeError("geometry_precision must be GeometryPrecisionPolicy or None.")
        if not isinstance(hermitian_, HermitianPrecisionPolicy):
            raise TypeError(
                "hermitian_precision must be HermitianPrecisionPolicy or None."
            )
        density = precision_.storage(jnp.asarray(initial_density))
        values = tuple(
            precision_.storage(jnp.asarray(tensor, dtype=density.dtype))
            for tensor in tensors
        )
        precision_.validate_storage(values + (density,))
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
        density_spectrum = HermitianSpectrum(density, precision=hermitian_)
        self.tensors = values
        self.initial_density = density
        self.precision = precision_
        self.geometry_precision = geometry_
        self.hermitian_precision = hermitian_
        self.precision_evidence = precision_.evidence_for(
            values + (density,),
            children={"initial-density": density_spectrum.precision_evidence},
        )
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
                    jnp.eye(
                        self.dimension,
                        dtype=self.initial_density.dtype,
                    )[None, ...],
                    intervention_id=f"identity:{index}",
                    precision=self.precision,
                    hermitian_precision=self.hermitian_precision,
                )
                for index in range(len(self.tensors))
            )
        )
        if len(operations) != len(self.tensors):
            raise ValueError("One intervention is required per process time slot.")
        state = self.precision.contraction(self.initial_density.reshape((1, -1)))
        probability = self.precision.decision(1.0)
        for tensor, intervention in zip(self.tensors, operations, strict=True):
            tensor_ = self.precision.contraction(tensor)
            superoperator = self.precision.contraction(intervention.superoperator)
            state = oe.contract("li,loir->ro", state, tensor_)
            state = oe.contract("oi,ri->ro", superoperator, state)
            density = self.precision.sum(state, axis=0).reshape(
                (self.dimension, self.dimension)
            )
            trace = self.precision.decision(
                jnp.real(self.precision.sum(jnp.diag(density)))
            )
            probability = self.precision.decision(probability * trace)
            state = state / jnp.maximum(trace, 1e-30)
        return (
            self.precision.output(state[0].reshape((self.dimension, self.dimension))),
            probability,
        )

    def physicality(self) -> ProcessTensorPhysicality:
        hermitian = 0.5 * (self.initial_density + jnp.conj(self.initial_density.T))
        hermiticity_residual = jnp.max(
            jnp.abs(self.initial_density - jnp.conj(self.initial_density.T))
        )
        initial_valid = (
            jnp.all(jnp.isfinite(self.initial_density))
            & (jnp.abs(jnp.trace(self.initial_density) - 1.0) <= 1e-8)
            & (hermiticity_residual <= 1e-8)
            & (
                HermitianSpectrum(
                    hermitian,
                    precision=self.hermitian_precision,
                ).minimum_eigenvalue
                >= -1e-8
            )
        )
        if any(tensor.shape[0] != 1 or tensor.shape[-1] != 1 for tensor in self.tensors):
            return ProcessTensorPhysicality(
                jnp.asarray([jnp.nan]),
                jnp.asarray([jnp.nan]),
                initial_valid,
                jnp.asarray(jnp.nan),
                status="unknown-general-temporal-bond",
                precision_evidence=self.precision_evidence,
            )
        reports = [
            DynamicalMapPhysicality(
                tensor[0, :, :, 0],
                self.dimension,
                geometry_precision=self.geometry_precision,
                hermitian_precision=self.hermitian_precision,
            )
            for tensor in self.tensors
        ]
        evidence = self.precision.evidence_for(
            self.tensors,
            children={
                f"local-map-{index}": report.precision_evidence
                for index, report in enumerate(reports)
            },
        )
        _, identity_probability = self.contract()
        causality_residual = jnp.abs(identity_probability - 1.0)
        return ProcessTensorPhysicality(
            jnp.stack([report.cp_margin for report in reports]),
            jnp.stack([report.trace_preservation_residual for report in reports]),
            initial_valid,
            causality_residual,
            status="local-markov-factorization",
            precision_evidence=evidence,
        )


def markov_process_tensor(
    superoperators: Sequence[ArrayLike],
    initial_density: ArrayLike,
    /,
    *,
    process_id: str = "markov-process",
    precision: TensorNetworkPrecisionPolicy | None = None,
    geometry_precision: GeometryPrecisionPolicy | None = None,
    hermitian_precision: HermitianPrecisionPolicy | None = None,
) -> ProcessTensorMPO:
    tensors = tuple(
        jnp.asarray(operator)[None, :, :, None] for operator in superoperators
    )
    return ProcessTensorMPO(
        tensors,
        initial_density,
        process_id=process_id,
        precision=precision,
        geometry_precision=geometry_precision,
        hermitian_precision=hermitian_precision,
    )


__all__ = [
    "ProcessTensorMPO",
    "ProcessTensorPhysicality",
    "QuantumIntervention",
    "markov_process_tensor",
]
