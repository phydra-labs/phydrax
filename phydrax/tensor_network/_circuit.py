#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared nearest-neighbor tensor-network execution for canonical QuantumProgram."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array

from .._strict import StrictModule
from ..operators.quantum._operations import (
    LocalKrausChannelOperation,
    LocalUnitaryOperation,
    QuantumProgram,
)
from ..operators.quantum._propagation import (
    kraus_trace_preservation_residual,
    unitarity_residual,
)
from ._core import LocallyPurifiedDensity, MatrixProductState
from ._evolution import apply_two_site_gate


class TensorNetworkQuantumProgramPolicy(StrictModule):
    maximum_operations: int = eqx.field(static=True)
    maximum_bond_dimension: int = eqx.field(static=True)
    maximum_purification_dimension: int = eqx.field(static=True)
    unitarity_tolerance: float = eqx.field(static=True)
    trace_preservation_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_operations: int,
        maximum_bond_dimension: int,
        maximum_purification_dimension: int,
        unitarity_tolerance: float = 1e-6,
        trace_preservation_tolerance: float = 1e-6,
    ):
        operations = int(maximum_operations)
        bond = int(maximum_bond_dimension)
        purification = int(maximum_purification_dimension)
        tolerances = (
            float(unitarity_tolerance),
            float(trace_preservation_tolerance),
        )
        if operations < 0 or bond <= 0 or purification <= 0:
            raise ValueError("Tensor-network quantum-program capacities are invalid.")
        if any(not isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError(
                "Tensor-network quantum-program tolerances must be finite and non-negative."
            )
        self.maximum_operations = operations
        self.maximum_bond_dimension = bond
        self.maximum_purification_dimension = purification
        self.unitarity_tolerance = tolerances[0]
        self.trace_preservation_tolerance = tolerances[1]


class _TensorNetworkQuantumOperationEvidence(StrictModule):
    finite: Array
    physicality_residual: Array
    valid: Array
    operation_kind: str = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)


class PreparedTensorNetworkQuantumProgram(StrictModule):
    program: QuantumProgram
    policy: TensorNetworkQuantumProgramPolicy
    target_sites: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    operation_kinds: tuple[str, ...] = eqx.field(static=True)
    operation_evidence: tuple[_TensorNetworkQuantumOperationEvidence, ...]
    valid: Array
    claim: str = eqx.field(static=True)


class TensorNetworkQuantumProgramResult(StrictModule):
    state: MatrixProductState | LocallyPurifiedDensity
    discarded_weights: Array
    retained_ranks: Array
    operation_valid: Array
    valid: Array
    program_id: str = eqx.field(static=True)
    state_kind: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def _operation_evidence(
    program: QuantumProgram,
    policy: TensorNetworkQuantumProgramPolicy,
    /,
) -> tuple[_TensorNetworkQuantumOperationEvidence, ...]:
    evidence = []
    for operation in program.operations:
        if isinstance(operation, LocalUnitaryOperation):
            values = operation.unitary
            residual = unitarity_residual(values)
            tolerance = policy.unitarity_tolerance
            kind = "unitary"
        else:
            values = operation.kraus
            residual = kraus_trace_preservation_residual(values)
            tolerance = policy.trace_preservation_tolerance
            kind = "kraus"
        finite = jnp.all(jnp.isfinite(values))
        evidence.append(
            _TensorNetworkQuantumOperationEvidence(
                finite=finite,
                physicality_residual=residual,
                valid=finite & jnp.isfinite(residual) & (residual <= tolerance),
                operation_kind=kind,
                schema_id=operation.schema_id,
            )
        )
    return tuple(evidence)


def prepare_tensor_network_quantum_program(
    program: QuantumProgram,
    policy: TensorNetworkQuantumProgramPolicy,
    /,
) -> PreparedTensorNetworkQuantumProgram:
    """Reject nonlocal operations; no implicit SWAP routing or densification."""
    if not isinstance(program, QuantumProgram) or not isinstance(
        policy, TensorNetworkQuantumProgramPolicy
    ):
        raise TypeError("program/policy types are invalid.")
    if len(program.operations) > policy.maximum_operations:
        raise ValueError("Tensor-network operation capacity exceeded.")
    targets = []
    kinds = []
    for operation in program.operations:
        sites = program.layout.target_indices(operation.target_wire_ids)
        if len(sites) > 2:
            raise ValueError(
                "Tensor-network device supports only one/two-site operations."
            )
        if len(sites) == 2 and sites[1] != sites[0] + 1:
            raise ValueError(
                "Nonlocal gates require an explicit caller-visible SWAP rewrite."
            )
        if isinstance(operation, LocalKrausChannelOperation):
            if program.state_kind != "density-matrix" or len(sites) != 1:
                raise ValueError("Tensor-network Kraus operations require one LPDO site.")
            kinds.append("kraus")
        else:
            kinds.append("unitary")
        targets.append(sites)
    evidence = _operation_evidence(program, policy)
    valid = (
        jnp.all(jnp.stack([item.valid for item in evidence]))
        if evidence
        else jnp.asarray(True)
    )
    return PreparedTensorNetworkQuantumProgram(
        program=program,
        policy=policy,
        target_sites=tuple(targets),
        operation_kinds=tuple(kinds),
        operation_evidence=evidence,
        valid=valid,
        claim="nearest-neighbor-static-no-implicit-swap-or-densification",
    )


def _mps_one_site(
    state: MatrixProductState, site: int, unitary: Array
) -> MatrixProductState:
    tensors = list(state.tensors)
    tensors[site] = oe.contract("oi,lir->lor", unitary, tensors[site])
    return MatrixProductState(tensors, precision=state.precision)


def _lpdo_one_site(
    state: LocallyPurifiedDensity, site: int, unitary: Array
) -> LocallyPurifiedDensity:
    tensors = list(state.tensors)
    tensors[site] = oe.contract("oi,likr->lokr", unitary, tensors[site])
    return LocallyPurifiedDensity(tensors, precision=state.precision)


def execute_tensor_network_quantum_program(
    prepared: PreparedTensorNetworkQuantumProgram,
    initial_state: MatrixProductState | LocallyPurifiedDensity,
    /,
) -> TensorNetworkQuantumProgramResult:
    if not isinstance(prepared, PreparedTensorNetworkQuantumProgram):
        raise TypeError("prepared must be PreparedTensorNetworkQuantumProgram.")
    if prepared.program.state_kind == "state-vector":
        if not isinstance(initial_state, MatrixProductState):
            raise TypeError("state-vector tensor execution requires MatrixProductState.")
        if initial_state.physical_dimensions != prepared.program.layout.local_dimensions:
            raise ValueError("MPS physical dimensions do not match program layout.")
    else:
        if not isinstance(initial_state, LocallyPurifiedDensity):
            raise TypeError(
                "density-matrix tensor execution requires LocallyPurifiedDensity."
            )
        if initial_state.physical_dimensions != prepared.program.layout.local_dimensions:
            raise ValueError("LPDO physical dimensions do not match program layout.")
    state = initial_state
    discarded = []
    ranks = []
    validity = []
    for operation, sites, physicality in zip(
        prepared.program.operations,
        prepared.target_sites,
        prepared.operation_evidence,
        strict=True,
    ):
        if isinstance(operation, LocalUnitaryOperation):
            if isinstance(state, MatrixProductState):
                if len(sites) == 1:
                    state = _mps_one_site(state, sites[0], operation.unitary)
                    discarded.append(jnp.asarray(0.0))
                    ranks.append(jnp.asarray(0, dtype=jnp.int32))
                    validity.append(physicality.valid)
                else:
                    left_dimension = state.physical_dimensions[sites[0]]
                    right_dimension = state.physical_dimensions[sites[1]]
                    gate = operation.unitary.reshape(
                        left_dimension, right_dimension, left_dimension, right_dimension
                    )
                    state, evidence = apply_two_site_gate(
                        state,
                        sites[0],
                        gate,
                        maximum_bond_dimension=prepared.policy.maximum_bond_dimension,
                        normalize=False,
                    )
                    discarded.append(evidence.discarded_weight)
                    ranks.append(jnp.asarray(evidence.retained_rank, dtype=jnp.int32))
                    validity.append(physicality.valid & evidence.valid)
            else:
                if len(sites) == 1:
                    state = _lpdo_one_site(state, sites[0], operation.unitary)
                    discarded.append(jnp.asarray(0.0))
                    ranks.append(jnp.asarray(0, dtype=jnp.int32))
                    validity.append(physicality.valid)
                else:
                    from ..solver._purified_tebd import apply_lpdo_two_site_unitary

                    left_dimension = state.physical_dimensions[sites[0]]
                    right_dimension = state.physical_dimensions[sites[1]]
                    gate = operation.unitary.reshape(
                        left_dimension, right_dimension, left_dimension, right_dimension
                    )
                    state, evidence = apply_lpdo_two_site_unitary(
                        state,
                        sites[0],
                        gate,
                        maximum_bond_dimension=prepared.policy.maximum_bond_dimension,
                    )
                    discarded.append(evidence.discarded_weight)
                    ranks.append(jnp.asarray(evidence.retained_rank, dtype=jnp.int32))
                    validity.append(physicality.valid & evidence.valid)
        else:
            if not isinstance(state, LocallyPurifiedDensity):
                raise RuntimeError("Prepared Kraus route lost its LPDO state kind.")
            from ..solver._purified_lindblad import (
                apply_local_kraus_channel,
                LocalKrausChannel,
            )

            channel = LocalKrausChannel(
                sites[0], operation.kraus, channel_id=operation.schema_id
            )
            state, evidence = apply_local_kraus_channel(
                state,
                channel,
                maximum_purification_dimension=prepared.policy.maximum_purification_dimension,
            )
            discarded.append(evidence.truncation.discarded_weight)
            ranks.append(jnp.asarray(evidence.truncation.retained_rank, dtype=jnp.int32))
            validity.append(physicality.valid & evidence.valid)
    operation_valid = jnp.stack(validity) if validity else jnp.empty((0,), dtype=bool)
    return TensorNetworkQuantumProgramResult(
        state=state,
        discarded_weights=jnp.stack(discarded) if discarded else jnp.empty((0,)),
        retained_ranks=jnp.stack(ranks) if ranks else jnp.empty((0,), dtype=jnp.int32),
        operation_valid=operation_valid,
        valid=prepared.valid & jnp.all(operation_valid),
        program_id=prepared.program.program_id,
        state_kind=prepared.program.state_kind,
        claim="finite-tensor-execution-with-explicit-truncation-evidence",
    )


__all__ = [
    "PreparedTensorNetworkQuantumProgram",
    "TensorNetworkQuantumProgramPolicy",
    "TensorNetworkQuantumProgramResult",
    "execute_tensor_network_quantum_program",
    "prepare_tensor_network_quantum_program",
]
