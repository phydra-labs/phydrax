"""Independent rod members with bounded memory on prepared execution worksets."""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
from jax import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._validation import canonical_identifier, positive_finite_float
from ...execution import (
    evaluate_execution_worksets_grouped,
    ExecutionGroup,
    ExecutionRuntime,
    ExecutionWorksetPlan,
    PoolExecutionSignature,
    PreparedExecutionWorksets,
)
from ._acceptance import FuelAcceptancePolicy
from ._bundles import FuelModelBundle
from ._execution import fuel_numerical_execution_id, FuelNumericalExecution
from ._problem import PreparedFuelRod
from ._result import FuelRodResult
from ._sensitivity import array_tree_bytes
from ._simulation import FuelRodSimulation
from ._step import evidence_prototype


# Rod-state copies charged to one live member's working set, in addition to
# its dense unknown-by-unknown joint factorization.
MEMBER_STATE_COPY_COUNT = 3
# Bytes charged per retained step-evidence entry (the widest 64-bit scalar).
EVIDENCE_ENTRY_BYTES = 8
# Conservative multiplier on the retained evidence of every step attempt.
RETAINED_EVIDENCE_COPY_COUNT = 2


def _member_working_set_bytes(state_bytes: int, unknown: Array, /) -> int:
    """Live rod-state copies plus the dense joint factorization of one member."""
    return (
        MEMBER_STATE_COPY_COUNT * state_bytes + unknown.size**2 * unknown.dtype.itemsize
    )


def _member_output_bytes(member: FuelEnsembleMember, state_bytes: int, /) -> int:
    """Accepted states plus step evidence of every possible attempt of one member."""
    policy = FuelAcceptancePolicy() if member.policy is None else member.policy
    evidence_entries = sum(
        leaf.size for leaf in jax.tree.leaves(evidence_prototype(member.prepared, policy))
    )
    evidence_bytes = EVIDENCE_ENTRY_BYTES * evidence_entries
    accepted_state_bytes = state_bytes * (member.maximum_steps + 1)
    attempt_evidence_bytes = (
        RETAINED_EVIDENCE_COPY_COUNT
        * evidence_bytes
        * member.maximum_steps
        * (member.maximum_retries + 1)
    )
    return accepted_state_bytes + attempt_evidence_bytes


class FuelEnsembleMember(StrictModule):
    """Prepared rod physics and host lifecycle settings; no store is open yet."""

    prepared: PreparedFuelRod
    parameters: FuelModelBundle | None
    policy: FuelAcceptancePolicy | None
    member_id: str = eqx.field(static=True)
    checkpoint_directory: Path = eqx.field(static=True)
    time_step: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    maximum_retries: int = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedFuelRod,
        /,
        *,
        member_id: str,
        checkpoint_directory: str | Path,
        time_step: float,
        maximum_steps: int,
        maximum_retries: int = 8,
        parameters: FuelModelBundle | None = None,
        acceptance_policy: FuelAcceptancePolicy | None = None,
    ) -> None:
        if not isinstance(prepared, PreparedFuelRod):
            raise TypeError("Ensemble member requires a PreparedFuelRod.")
        step = positive_finite_float(time_step, "Ensemble time_step")
        if maximum_steps < 1 or maximum_retries < 0:
            raise ValueError(
                "Ensemble maximum_steps must be at least one and maximum_retries nonnegative."
            )
        self.prepared = prepared
        self.parameters = parameters
        self.policy = acceptance_policy
        self.member_id = canonical_identifier(member_id, "ensemble member_id")
        self.time_step = step
        self.maximum_steps = maximum_steps
        self.maximum_retries = maximum_retries


class FuelEnsembleResult(StrictModule):
    """One `FuelRodResult` per member, in member order, with the execution worksets."""

    results: tuple[FuelRodResult, ...]
    worksets: PreparedExecutionWorksets
    member_ids: tuple[str, ...] = eqx.field(static=True)
    maximum_live_member_bytes: int = eqx.field(static=True)
    maximum_retained_output_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class FuelEnsemblePlan(StrictModule):
    """Independent rod members; no member-by-site materialization or vmap.

    Every member is charged for all possible retry evidence and accepted-state
    retention before launch. Grouped execution runs independent members on
    disjoint device groups; worksets are grouped by the explicit execution
    signature, never by equal array shapes or numerical rod identity.
    """

    members: tuple[FuelEnsembleMember, ...]
    worksets: PreparedExecutionWorksets
    maximum_working_set_bytes: int = eqx.field(static=True)
    maximum_output_bytes: int = eqx.field(static=True)
    maximum_member_bytes: int = eqx.field(static=True)
    output_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        members: tuple[FuelEnsembleMember, ...],
        /,
        *,
        maximum_working_set_bytes: int,
        maximum_output_bytes: int,
        bucket_capacity: int = 1,
    ) -> None:
        if not members or len({member.member_id for member in members}) != len(members):
            raise ValueError("Ensemble members must be nonempty with unique member_id.")
        if len({member.checkpoint_directory for member in members}) != len(members):
            raise ValueError("Ensemble members must not share a checkpoint directory.")
        if maximum_working_set_bytes <= 0 or maximum_output_bytes <= 0:
            raise ValueError("Ensemble memory bounds must be positive.")
        canonical = tuple(sorted(members, key=lambda member: member.member_id))
        signatures: list[PoolExecutionSignature] = []
        member_bytes: list[int] = []
        output_bytes = 0
        for member in canonical:
            state = member.prepared.initial_state(parameters=member.parameters)
            state_bytes = array_tree_bytes(state)
            unknown = member.prepared.pack_unknown(state)
            member_bytes.append(_member_working_set_bytes(state_bytes, unknown))
            output_bytes += _member_output_bytes(member, state_bytes)
            signatures.append(
                PoolExecutionSignature(
                    topology_id=fuel_numerical_execution_id(
                        member.prepared, policy=member.policy
                    ),
                    method_id="fuel-production-backward-euler",
                    precision_id=str(unknown.dtype),
                    backend_id=jax.default_backend(),
                )
            )
        if (
            max(member_bytes) > maximum_working_set_bytes
            or output_bytes > maximum_output_bytes
        ):
            raise ValueError(
                "Ensemble member working set or retained output exceeds its memory bound."
            )
        self.members = canonical
        self.worksets = ExecutionWorksetPlan(
            tuple(member.member_id for member in canonical),
            tuple(signatures),
            bucket_capacity=bucket_capacity,
        ).prepare()
        self.maximum_working_set_bytes = maximum_working_set_bytes
        self.maximum_output_bytes = maximum_output_bytes
        self.maximum_member_bytes = max(member_bytes)
        self.output_bytes = output_bytes
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fuel-independent-rod-ensemble",
                "worksets": self.worksets.prepared_id,
                "prepared": tuple(member.prepared.prepared_id for member in canonical),
                "lifecycle": tuple(
                    (
                        member.time_step,
                        member.maximum_steps,
                        member.maximum_retries,
                        str(member.checkpoint_directory),
                    )
                    for member in canonical
                ),
                "working_set_bound": maximum_working_set_bytes,
                "retained_output_bound": maximum_output_bytes,
            }
        )

    def run(self, /, *, runtime: ExecutionRuntime | None = None) -> FuelEnsembleResult:
        live = ExecutionRuntime.current() if runtime is None else runtime
        concurrent_count = min(live.root_group.spec.device_count, len(self.members))
        if concurrent_count * self.maximum_member_bytes > self.maximum_working_set_bytes:
            raise ValueError(
                "Concurrent rod member working sets exceed the ensemble memory bound."
            )
        # One numerical execution per execution signature, shared by its members.
        numerical_executions: dict[str, FuelNumericalExecution] = {}

        def evaluate(
            index: int,
            semantic_id: str,
            signature: PoolExecutionSignature,
            _rng_index: int,
            group: ExecutionGroup,
        ) -> FuelRodResult:
            member = self.members[index]
            if (
                member.member_id != semantic_id
                or signature.signature_id
                != self.worksets.plan.signatures[index].signature_id
            ):
                raise ValueError("Workset item identity does not match its rod member.")
            prepared = jax.device_put(member.prepared, group.devices[0])
            parameters = (
                None
                if member.parameters is None
                else jax.device_put(member.parameters, group.devices[0])
            )
            with jax.default_device(group.devices[0]):
                if signature.signature_id not in numerical_executions:
                    numerical_executions[signature.signature_id] = FuelNumericalExecution(
                        prepared, policy=member.policy
                    )
                simulation = FuelRodSimulation(
                    prepared,
                    time_step=member.time_step,
                    checkpoint_directory=member.checkpoint_directory,
                    maximum_steps=member.maximum_steps,
                    maximum_retries=member.maximum_retries,
                    acceptance_policy=member.policy,
                    numerical_execution=numerical_executions[signature.signature_id],
                )
                try:
                    return simulation.run(parameters=parameters)
                finally:
                    simulation.close()

        results = evaluate_execution_worksets_grouped(self.worksets, live, evaluate)
        if any(not isinstance(result, FuelRodResult) for result in results):
            raise TypeError("Rod ensemble worksets must return FuelRodResult objects.")
        return FuelEnsembleResult(
            results=results,
            worksets=self.worksets,
            member_ids=self.worksets.plan.semantic_ids,
            maximum_live_member_bytes=concurrent_count * self.maximum_member_bytes,
            maximum_retained_output_bytes=self.output_bytes,
            plan_id=self.plan_id,
        )
