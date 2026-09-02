#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fail-closed quantum admission, run records, and provider-neutral payloads."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..operators.quantum._operations import (
    LocalKrausChannelOperation,
    LocalUnitaryOperation,
    QuantumProgram,
)
from ..operators.quantum._register import HilbertRegisterLayout
from ._quantum_compilation import HardwareTopology
from ._quantum_experiment import (
    QuantumExperimentExactResult,
    QuantumExperimentProgram,
    QuantumShotBatchResult,
)


class QuantumProgramInterchange(StrictModule, NonTrainableState):
    """Strict provider-neutral numerical payload for the canonical map IR."""

    numerical_operations: tuple[Array, ...]
    finite: Array
    valid: Array
    wire_ids: tuple[str, ...] = eqx.field(static=True)
    local_dimensions: tuple[int, ...] = eqx.field(static=True)
    state_kind: str = eqx.field(static=True)
    operation_kinds: tuple[str, ...] = eqx.field(static=True)
    operation_targets: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    payload_id: str = eqx.field(static=True)

    def __init__(self, program: QuantumProgram, /):
        if not isinstance(program, QuantumProgram):
            raise TypeError("program must be QuantumProgram.")
        numerical: list[Array] = []
        kinds: list[str] = []
        targets: list[tuple[str, ...]] = []
        for operation in program.operations:
            if isinstance(operation, LocalUnitaryOperation):
                numerical.append(operation.unitary)
                kinds.append("unitary")
            else:
                numerical.append(operation.kraus)
                kinds.append("kraus")
            targets.append(operation.target_wire_ids)
        self.numerical_operations = tuple(numerical)
        self.wire_ids = program.layout.wire_ids
        self.local_dimensions = program.layout.local_dimensions
        self.state_kind = program.state_kind
        self.operation_kinds = tuple(kinds)
        self.operation_targets = tuple(targets)
        finite = (
            jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in numerical]))
            if numerical
            else jnp.asarray(True)
        )
        self.finite = finite
        self.valid = finite
        self.payload_id = canonical_fingerprint(
            {
                "kind": "quantum-program-interchange",
                "wires": self.wire_ids,
                "dimensions": self.local_dimensions,
                "state_kind": self.state_kind,
                "operation_kinds": self.operation_kinds,
                "operation_targets": self.operation_targets,
                "operation_shapes": tuple(value.shape for value in numerical),
                "operation_dtypes": tuple(str(value.dtype) for value in numerical),
                "numerical_content": array_tree_fingerprint(tuple(numerical)),
            }
        )

    def materialize(self) -> QuantumProgram:
        layout = HilbertRegisterLayout(self.wire_ids, self.local_dimensions)
        operations: list[LocalUnitaryOperation | LocalKrausChannelOperation] = []
        for kind, targets, numerical in zip(
            self.operation_kinds,
            self.operation_targets,
            self.numerical_operations,
            strict=True,
        ):
            if kind == "unitary":
                operations.append(LocalUnitaryOperation(numerical, targets))
            elif kind == "kraus":
                operations.append(LocalKrausChannelOperation(numerical, targets))
            else:
                raise ValueError("Interchange operation kind is not canonical.")
        return QuantumProgram(layout, operations, state_kind=self.state_kind)


class QuantumResultInterchange(StrictModule, NonTrainableState):
    """Provider-neutral fixed-capacity experiment evidence payload."""

    probabilities: Array
    counts: Array
    branch_status: Array
    zero_probability: Array
    probability_sum_residual: Array
    negative_probability_residual: Array
    finite: Array
    valid: Array
    experiment_id: str = eqx.field(static=True)
    payload_id: str = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        probabilities: ArrayLike,
        counts: ArrayLike,
        branch_status: ArrayLike,
        zero_probability: ArrayLike,
        valid: ArrayLike,
        /,
        *,
        experiment_id: str,
        tolerance: float = 1e-6,
    ):
        probabilities_ = jnp.asarray(probabilities)
        counts_ = jnp.asarray(counts)
        status = jnp.asarray(branch_status)
        zero = jnp.asarray(zero_probability, dtype=bool)
        valid_ = jnp.asarray(valid, dtype=bool).reshape(())
        tolerance_ = float(tolerance)
        if probabilities_.ndim != 1 or probabilities_.shape[0] < 1:
            raise ValueError("Result probabilities require shape (outcomes,).")
        if counts_.shape != probabilities_.shape or status.shape != probabilities_.shape:
            raise ValueError("Counts/status must match the fixed outcome capacity.")
        if zero.shape != probabilities_.shape:
            raise ValueError("zero_probability must match the fixed outcome capacity.")
        if not jnp.issubdtype(counts_.dtype, jnp.integer) or not jnp.issubdtype(
            status.dtype, jnp.integer
        ):
            raise TypeError("Interchange counts/status must use integer coordinates.")
        if not isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("Interchange tolerance must be finite and nonnegative.")
        identifier = str(experiment_id)
        if not identifier:
            raise ValueError("experiment_id must be nonempty.")
        probability_sum_residual = jnp.abs(jnp.sum(probabilities_) - 1.0)
        negative_probability_residual = jnp.maximum(-jnp.min(probabilities_), 0.0)
        finite = (
            jnp.all(jnp.isfinite(probabilities_))
            & jnp.all(jnp.isfinite(counts_))
            & jnp.all(jnp.isfinite(status))
        )
        evidence_valid = (
            valid_
            & finite
            & jnp.all(counts_ >= 0)
            & (probability_sum_residual <= tolerance_)
            & (negative_probability_residual <= tolerance_)
        )
        self.probabilities = probabilities_
        self.counts = counts_
        self.branch_status = status
        self.zero_probability = zero
        self.probability_sum_residual = probability_sum_residual
        self.negative_probability_residual = negative_probability_residual
        self.finite = finite
        self.valid = evidence_valid
        self.experiment_id = identifier
        self.tolerance = tolerance_
        self.payload_id = canonical_fingerprint(
            {
                "kind": "quantum-result-interchange",
                "experiment": identifier,
                "probability_shape": probabilities_.shape,
                "probability_dtype": str(probabilities_.dtype),
                "count_dtype": str(counts_.dtype),
                "status_dtype": str(status.dtype),
                "numerical_content": array_tree_fingerprint(
                    (probabilities_, counts_, status, zero, valid_)
                ),
            }
        )


class QuantumServicePolicy(StrictModule, NonTrainableState):
    maximum_wires: int = eqx.field(static=True)
    maximum_operations: int = eqx.field(static=True)
    maximum_branches: int = eqx.field(static=True)
    maximum_shots: int = eqx.field(static=True)
    maximum_classical_bits: int = eqx.field(static=True)
    allowed_topology_ids: tuple[str, ...] = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_wires: int,
        maximum_operations: int,
        maximum_branches: int,
        maximum_shots: int,
        maximum_classical_bits: int,
        allowed_topology_ids: Sequence[str],
    ):
        capacities = tuple(
            int(value)
            for value in (
                maximum_wires,
                maximum_operations,
                maximum_branches,
                maximum_shots,
                maximum_classical_bits,
            )
        )
        topologies = tuple(str(value) for value in allowed_topology_ids)
        if any(value < 0 for value in capacities):
            raise ValueError("Quantum service capacities must be nonnegative.")
        if not topologies or any(not value for value in topologies):
            raise ValueError("At least one allowed topology ID is required.")
        self.maximum_wires = capacities[0]
        self.maximum_operations = capacities[1]
        self.maximum_branches = capacities[2]
        self.maximum_shots = capacities[3]
        self.maximum_classical_bits = capacities[4]
        self.allowed_topology_ids = topologies
        self.policy_id = canonical_fingerprint(
            {
                "kind": "quantum-service-policy",
                "capacities": capacities,
                "allowed_topologies": topologies,
            }
        )


class QuantumServiceRequest(StrictModule, NonTrainableState):
    experiment: QuantumExperimentProgram
    topology: HardwareTopology
    requested_shots: int = eqx.field(static=True)
    request_id: str = eqx.field(static=True)

    def __init__(
        self,
        experiment: QuantumExperimentProgram,
        topology: HardwareTopology,
        /,
        *,
        requested_shots: int,
    ):
        if not isinstance(experiment, QuantumExperimentProgram) or not isinstance(
            topology, HardwareTopology
        ):
            raise TypeError("experiment/topology types are invalid.")
        shots = int(requested_shots)
        if shots < 0:
            raise ValueError("requested_shots must be nonnegative.")
        self.experiment = experiment
        self.topology = topology
        self.requested_shots = shots
        self.request_id = canonical_fingerprint(
            {
                "kind": "quantum-service-request",
                "experiment": experiment.experiment_id,
                "topology": topology.topology_id,
                "shots": shots,
            }
        )


class QuantumServiceAdmission(StrictModule, NonTrainableState):
    accepted: Array
    required_wires: int = eqx.field(static=True)
    required_operations: int = eqx.field(static=True)
    required_branches: int = eqx.field(static=True)
    required_shots: int = eqx.field(static=True)
    required_classical_bits: int = eqx.field(static=True)
    refusal_codes: tuple[str, ...] = eqx.field(static=True)
    experiment_id: str = eqx.field(static=True)
    request_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    admission_id: str = eqx.field(static=True)


def admit_quantum_service_request(
    request: QuantumServiceRequest,
    policy: QuantumServicePolicy,
    /,
) -> QuantumServiceAdmission:
    """Compute a fail-closed decision; admission never invokes a provider."""
    if not isinstance(request, QuantumServiceRequest) or not isinstance(
        policy, QuantumServicePolicy
    ):
        raise TypeError("request/policy types are invalid.")
    experiment = request.experiment
    wires = len(experiment.prefix.layout.wire_ids)
    operations = len(experiment.prefix.operations) + sum(
        len(branch.operations) for branch in experiment.branch_programs
    )
    branches = len(experiment.branch_programs)
    bits = experiment.classical_layout.total_bits
    refusal: list[str] = []
    if wires > policy.maximum_wires:
        refusal.append("wire-capacity-exceeded")
    if operations > policy.maximum_operations:
        refusal.append("operation-capacity-exceeded")
    if branches > policy.maximum_branches:
        refusal.append("branch-capacity-exceeded")
    if request.requested_shots > policy.maximum_shots:
        refusal.append("shot-capacity-exceeded")
    if bits > policy.maximum_classical_bits:
        refusal.append("classical-bit-capacity-exceeded")
    if len(request.topology.physical_wire_ids) != wires:
        refusal.append("topology-wire-count-mismatch")
    if request.topology.topology_id not in policy.allowed_topology_ids:
        refusal.append("topology-not-allowed")
    identifier = canonical_fingerprint(
        {
            "kind": "quantum-service-admission",
            "request": request.request_id,
            "policy": policy.policy_id,
            "requirements": (wires, operations, branches, request.requested_shots, bits),
            "refusal_codes": tuple(refusal),
        }
    )
    return QuantumServiceAdmission(
        jnp.asarray(not refusal),
        wires,
        operations,
        branches,
        request.requested_shots,
        bits,
        tuple(refusal),
        experiment.experiment_id,
        request.request_id,
        policy.policy_id,
        identifier,
    )


class QuantumServiceRunRecord(StrictModule, NonTrainableState):
    admission: QuantumServiceAdmission
    result_payload: QuantumResultInterchange | None
    executed: Array
    successful: Array
    logical_start_tick: int = eqx.field(static=True)
    logical_finish_tick: int = eqx.field(static=True)
    status: str = eqx.field(static=True)
    run_id: str = eqx.field(static=True)


def record_quantum_service_run(
    admission: QuantumServiceAdmission,
    exact: QuantumExperimentExactResult | None,
    shots: QuantumShotBatchResult | None,
    /,
    *,
    logical_start_tick: int,
    logical_finish_tick: int,
) -> QuantumServiceRunRecord:
    """Record caller-supplied execution evidence; this function performs no provider call."""
    if not isinstance(admission, QuantumServiceAdmission):
        raise TypeError("admission must be QuantumServiceAdmission.")
    start = int(logical_start_tick)
    finish = int(logical_finish_tick)
    if start < 0 or finish < start:
        raise ValueError("Logical run ticks must be ordered and nonnegative.")
    if not bool(admission.accepted):
        if exact is not None or shots is not None:
            raise ValueError("A refused request cannot carry execution evidence.")
        payload = None
        executed = jnp.asarray(False)
        successful = jnp.asarray(False)
        status = "refused"
    else:
        if not isinstance(exact, QuantumExperimentExactResult) or not isinstance(
            shots, QuantumShotBatchResult
        ):
            raise TypeError("Accepted runs require exact and shot evidence.")
        if (
            exact.experiment_id != shots.experiment_id
            or exact.experiment_id != admission.experiment_id
        ):
            raise ValueError(
                "Admission, exact, and shot evidence identify different experiments."
            )
        if shots.shot_count != admission.required_shots:
            raise ValueError("Shot evidence count differs from the admitted request.")
        payload = QuantumResultInterchange(
            exact.instrument_result.probabilities,
            shots.counts,
            exact.branch_status,
            exact.zero_probability,
            exact.valid & shots.valid,
            experiment_id=exact.experiment_id,
        )
        executed = jnp.asarray(True)
        successful = exact.valid & shots.valid
        status = "succeeded" if bool(successful) else "failed-evidence"
    identifier = canonical_fingerprint(
        {
            "kind": "quantum-service-run-record",
            "admission": admission.admission_id,
            "start": start,
            "finish": finish,
            "status": status,
            "payload": None if payload is None else payload.payload_id,
        }
    )
    return QuantumServiceRunRecord(
        admission,
        payload,
        executed,
        successful,
        start,
        finish,
        status,
        identifier,
    )


__all__ = [
    "QuantumProgramInterchange",
    "QuantumResultInterchange",
    "QuantumServiceAdmission",
    "QuantumServicePolicy",
    "QuantumServiceRequest",
    "QuantumServiceRunRecord",
    "admit_quantum_service_request",
    "record_quantum_service_run",
]
