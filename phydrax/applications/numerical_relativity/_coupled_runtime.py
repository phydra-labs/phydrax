#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntFlag
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...metrix._adm_exchange import ADMGridGeometry, StressEnergyProjection
from ._matter_coupling import (
    CoupledBudget,
    CoupledStageAddress,
    CoupledStepLedgers,
    MatterCouplingPolicy,
    MatterStageProposal,
    RelativisticMatterKind,
    Z4cStageProposal,
)


_SSPRK33_STAGE_TIME_FRACTIONS = (0.0, 1.0, 0.5)
_SSPRK33_STAGE_COUNT = 3


class CoupledEvolutionStatus(IntFlag):
    """Composable, fixed-width reasons why an atomic coupled step was rejected."""

    SUCCESS = 0
    INVALID_STEP = 1
    STAGE_IDENTITY_MISMATCH = 2
    Z4C_REJECTED = 4
    MATTER_REJECTED = 8
    NONFINITE = 16
    LEDGER_LIMIT_EXCEEDED = 32
    FAILURE_LIMIT_REACHED = 64
    TERMINAL = 128
    DERIVATIVE_INVALID = 256


_STATUS_MESSAGES = {
    CoupledEvolutionStatus.INVALID_STEP: "step size is non-finite or non-positive",
    CoupledEvolutionStatus.STAGE_IDENTITY_MISMATCH: (
        "stage, time, topology, or matter-kind identity does not match"
    ),
    CoupledEvolutionStatus.Z4C_REJECTED: "the Z4c participant rejected a stage proposal",
    CoupledEvolutionStatus.MATTER_REJECTED: (
        "the matter participant rejected a stage proposal"
    ),
    CoupledEvolutionStatus.NONFINITE: "the coupled proposal contains non-finite evidence",
    CoupledEvolutionStatus.LEDGER_LIMIT_EXCEEDED: (
        "a source, constraint, conservation, or floor budget was exceeded"
    ),
    CoupledEvolutionStatus.FAILURE_LIMIT_REACHED: (
        "the configured consecutive failure limit was reached"
    ),
    CoupledEvolutionStatus.TERMINAL: "the coupled runtime is terminal",
    CoupledEvolutionStatus.DERIVATIVE_INVALID: (
        "one or both participant derivatives lack valid evidence"
    ),
}


def coupled_evolution_status_message(
    status: int | CoupledEvolutionStatus, /
) -> str:
    value = CoupledEvolutionStatus(int(status))
    if value == CoupledEvolutionStatus.SUCCESS:
        return "successful"
    return "; ".join(message for flag, message in _STATUS_MESSAGES.items() if value & flag)


def _scalar(value: ArrayLike, role: str, /, *, dtype: Any | None = None) -> Array:
    result = jnp.asarray(value, dtype=dtype)
    if result.shape != ():
        raise ValueError(f"{role} must be scalar.")
    return result


def _select_tree(condition: Array, proposed: Any, current: Any, /) -> Any:
    if jax.tree.structure(proposed) != jax.tree.structure(current):
        raise ValueError("A participant proposal must preserve its state PyTree structure.")
    return jax.tree.map(
        lambda candidate, accepted: jnp.where(condition, candidate, accepted),
        proposed,
        current,
    )


def _select_budget(
    condition: Array, proposed: CoupledBudget, current: CoupledBudget, /
) -> CoupledBudget:
    selected = _select_tree(condition, proposed, current)
    if not isinstance(selected, CoupledBudget):
        raise TypeError("Coupled budget selection did not preserve CoupledBudget.")
    return selected


def _status_bit(predicate: Array, bit: CoupledEvolutionStatus, /) -> Array:
    return jnp.where(predicate, jnp.int32(int(bit)), jnp.int32(0))


class CoupledEvolutionState(StrictModule, NonTrainableState):
    """Authoritative accepted Z4c/matter pair at one macro-step boundary."""

    z4c: Any
    matter: Any
    budget: CoupledBudget
    time: Array
    next_step_id: Array
    accepted_steps: Array
    rejected_steps: Array
    consecutive_failures: Array
    terminal: Array
    topology_id: str = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)

    def __init__(
        self,
        z4c: Any,
        matter: Any,
        budget: CoupledBudget,
        time: ArrayLike,
        next_step_id: ArrayLike,
        accepted_steps: ArrayLike,
        rejected_steps: ArrayLike,
        consecutive_failures: ArrayLike,
        terminal: ArrayLike,
        /,
        *,
        topology_id: str,
        runtime_id: str,
    ):
        if not isinstance(budget, CoupledBudget):
            raise TypeError("budget must be CoupledBudget.")
        if not isinstance(topology_id, str) or not topology_id:
            raise ValueError("topology_id must be non-empty.")
        if not isinstance(runtime_id, str) or not runtime_id:
            raise ValueError("runtime_id must be non-empty.")
        self.z4c = z4c
        self.matter = matter
        self.budget = budget
        self.time = _scalar(time, "coupled state time")
        self.next_step_id = _scalar(
            next_step_id, "coupled state next_step_id", dtype=jnp.int32
        )
        self.accepted_steps = _scalar(
            accepted_steps, "coupled state accepted_steps", dtype=jnp.int32
        )
        self.rejected_steps = _scalar(
            rejected_steps, "coupled state rejected_steps", dtype=jnp.int32
        )
        self.consecutive_failures = _scalar(
            consecutive_failures,
            "coupled state consecutive_failures",
            dtype=jnp.int32,
        )
        self.terminal = _scalar(terminal, "coupled state terminal", dtype=bool)
        self.topology_id = topology_id
        self.runtime_id = runtime_id


class _CoupledStageEvaluation(StrictModule):
    address: CoupledStageAddress
    geometry: ADMGridGeometry
    stress_energy: StressEnergyProjection
    z4c_proposal: Z4cStageProposal
    matter_proposal: MatterStageProposal
    ledgers: CoupledStepLedgers
    proposed_budget: CoupledBudget
    status: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    successful: Array


class CoupledStepResult(StrictModule):
    """Three retained SSPRK stages and one all-or-nothing authoritative commit."""

    candidate: CoupledEvolutionState
    accepted: CoupledEvolutionState
    addresses: tuple[CoupledStageAddress, CoupledStageAddress, CoupledStageAddress]
    geometries: tuple[ADMGridGeometry, ADMGridGeometry, ADMGridGeometry]
    stress_energy: tuple[
        StressEnergyProjection,
        StressEnergyProjection,
        StressEnergyProjection,
    ]
    z4c_proposals: tuple[Z4cStageProposal, Z4cStageProposal, Z4cStageProposal]
    matter_proposals: tuple[
        MatterStageProposal,
        MatterStageProposal,
        MatterStageProposal,
    ]
    stage_ledgers: tuple[CoupledStepLedgers, CoupledStepLedgers, CoupledStepLedgers]
    ledgers: CoupledStepLedgers
    stage_status: Array
    stage_successful: Array
    status: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    successful: Array
    accepted_step: Array
    attempted: Array
    operator_id: str = eqx.field(static=True)


class Z4cMatterCoupledRuntime(StrictModule, NonTrainableState):
    """Atomic same-stage SSPRK33 evolution of Z4c with GRHD or GRMHD.

    Geometry and stress-energy are rebuilt from the paired working states at
    times ``t``, ``t + dt``, and ``t + dt/2``. Stage callbacks receive both the
    immutable pre-step state and the current SSPRK stage state; they own the
    standard SSPRK33 recurrence and return ledgers already weighted for that
    stage. This coordinator alone commits, and only after all three paired stage
    proposals and the cumulative budget pass.
    """

    geometry_at_stage: Callable = eqx.field(static=True)
    stress_energy_at_stage: Callable = eqx.field(static=True)
    propose_z4c_stage: Callable = eqx.field(static=True)
    propose_matter_stage: Callable = eqx.field(static=True)
    policy: MatterCouplingPolicy
    topology_id: str = eqx.field(static=True)
    matter_kind: RelativisticMatterKind = eqx.field(static=True)
    z4c_runtime_id: str = eqx.field(static=True)
    matter_runtime_id: str = eqx.field(static=True)
    stage_time_fractions: tuple[float, float, float] = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry_at_stage: Callable,
        stress_energy_at_stage: Callable,
        propose_z4c_stage: Callable,
        propose_matter_stage: Callable,
        policy: MatterCouplingPolicy,
        /,
        *,
        topology_id: str,
        matter_kind: RelativisticMatterKind,
        z4c_runtime_id: str,
        matter_runtime_id: str,
    ):
        callbacks = (
            geometry_at_stage,
            stress_energy_at_stage,
            propose_z4c_stage,
            propose_matter_stage,
        )
        if any(not callable(callback) for callback in callbacks):
            raise TypeError("Every coupled stage adapter must be callable.")
        if not isinstance(policy, MatterCouplingPolicy):
            raise TypeError("policy must be MatterCouplingPolicy.")
        if not isinstance(topology_id, str) or not topology_id:
            raise ValueError("topology_id must be non-empty.")
        if matter_kind not in ("grhd", "grmhd"):
            raise ValueError("matter_kind must be 'grhd' or 'grmhd'.")
        if not isinstance(z4c_runtime_id, str) or not z4c_runtime_id:
            raise ValueError("z4c_runtime_id must be non-empty.")
        if not isinstance(matter_runtime_id, str) or not matter_runtime_id:
            raise ValueError("matter_runtime_id must be non-empty.")
        self.geometry_at_stage = geometry_at_stage
        self.stress_energy_at_stage = stress_energy_at_stage
        self.propose_z4c_stage = propose_z4c_stage
        self.propose_matter_stage = propose_matter_stage
        self.policy = policy
        self.topology_id = topology_id
        self.matter_kind = matter_kind
        self.z4c_runtime_id = z4c_runtime_id
        self.matter_runtime_id = matter_runtime_id
        self.stage_time_fractions = _SSPRK33_STAGE_TIME_FRACTIONS
        self.operator_id = canonical_fingerprint(
            {
                "kind": "same-stage-z4c-relativistic-matter-runtime",
                "integrator": "ssprk33",
                "stage_time_fractions": list(_SSPRK33_STAGE_TIME_FRACTIONS),
                "topology": topology_id,
                "matter_kind": matter_kind,
                "z4c_runtime": z4c_runtime_id,
                "matter_runtime": matter_runtime_id,
                "policy": policy.policy_id,
            }
        )

    def initialize(
        self,
        z4c: Any,
        matter: Any,
        /,
        *,
        time: ArrayLike = 0.0,
        next_step_id: ArrayLike = 0,
        budget: CoupledBudget | None = None,
    ) -> CoupledEvolutionState:
        time_ = _scalar(time, "initial coupled time")
        budget_ = CoupledBudget.zeros(dtype=time_.dtype) if budget is None else budget
        if not isinstance(budget_, CoupledBudget):
            raise TypeError("budget must be CoupledBudget or None.")
        return CoupledEvolutionState(
            z4c,
            matter,
            budget_,
            time_,
            next_step_id,
            0,
            0,
            0,
            False,
            topology_id=self.topology_id,
            runtime_id=self.operator_id,
        )

    def _evaluate_stage(
        self,
        base_z4c: Any,
        working_z4c: Any,
        base_matter: Any,
        working_matter: Any,
        budget: CoupledBudget,
        address: CoupledStageAddress,
        args: Any,
        step_valid: Array,
        step_finite: Array,
        active: Array,
        /,
    ) -> _CoupledStageEvaluation:
        geometry = self.geometry_at_stage(working_z4c, address, args)
        if not isinstance(geometry, ADMGridGeometry):
            raise TypeError("geometry_at_stage must return ADMGridGeometry.")
        stress_energy = self.stress_energy_at_stage(
            working_matter, geometry, address, args
        )
        if not isinstance(stress_energy, StressEnergyProjection):
            raise TypeError(
                "stress_energy_at_stage must return StressEnergyProjection."
            )
        z4c = self.propose_z4c_stage(
            base_z4c,
            working_z4c,
            geometry,
            stress_energy,
            address,
            args,
        )
        matter = self.propose_matter_stage(
            base_matter,
            working_matter,
            geometry,
            stress_energy,
            address,
            args,
        )
        if not isinstance(z4c, Z4cStageProposal):
            raise TypeError("propose_z4c_stage must return Z4cStageProposal.")
        if not isinstance(matter, MatterStageProposal):
            raise TypeError("propose_matter_stage must return MatterStageProposal.")

        static_exchange_identity = (
            geometry.topology_id == self.topology_id
            and z4c.geometry.geometry_lineage_id
            == geometry.geometry_lineage_id
            and z4c.geometry.convention_id == geometry.convention_id
            and z4c.geometry.scale_id == geometry.scale_id
            and z4c.geometry.topology_id == geometry.topology_id
            and matter.stress_energy.projection_id == stress_energy.projection_id
        )
        dynamic_exchange_identity = (
            stress_energy.compatible_with(geometry)
            & matter.stress_energy.compatible_with(geometry)
            & (z4c.geometry.snapshot_token == geometry.snapshot_token)
        )
        active_masks_match = jnp.all(stress_energy.active == geometry.active)
        address_valid = address.matches(z4c.address) & address.matches(matter.address)
        identity_valid = (
            address_valid
            & active_masks_match
            & dynamic_exchange_identity
            & jnp.asarray(static_exchange_identity)
            & jnp.asarray(matter.matter_kind == self.matter_kind)
        )
        ledgers = CoupledStepLedgers(
            z4c.source,
            z4c.constraint,
            matter.conservation,
            matter.floor,
            matter.horizon_flux,
        )
        proposed_budget = budget.accumulate(ledgers)
        step_limits = ledgers.within_step_limits(self.policy)
        budget_limits = proposed_budget.within_limits(self.policy)
        exchange_limits = (
            jnp.max(jnp.abs(stress_energy.projection_defect))
            <= self.policy.source_consistency_tolerance
        ) & (
            jnp.max(jnp.abs(stress_energy.conservation_defect))
            <= self.policy.conservation_tolerance
        )
        finite = (
            step_finite
            & z4c.evidence.finite
            & matter.evidence.finite
            & jnp.all(geometry.finite)
            & jnp.all(stress_energy.finite)
            & ledgers.finite
        )
        converged = z4c.evidence.converged & matter.evidence.converged
        physically_valid = (
            z4c.evidence.physically_valid
            & matter.evidence.physically_valid
            & geometry.all_active_valid
            & stress_energy.all_active_valid
            & step_limits
            & budget_limits
            & exchange_limits
        )
        qualified = (
            z4c.evidence.qualified & matter.evidence.qualified & identity_valid
        )
        derivative_valid = (
            z4c.evidence.derivative_valid & matter.evidence.derivative_valid
        )
        derivatives_admitted = (
            derivative_valid if self.policy.require_derivative_valid else jnp.asarray(True)
        )
        successful = (
            active
            & step_valid
            & finite
            & converged
            & physically_valid
            & qualified
            & derivatives_admitted
            & z4c.evidence.forward_successful
            & matter.evidence.forward_successful
        )

        status = jnp.int32(0)
        status = status | _status_bit(~step_valid, CoupledEvolutionStatus.INVALID_STEP)
        status = status | _status_bit(
            ~identity_valid, CoupledEvolutionStatus.STAGE_IDENTITY_MISMATCH
        )
        status = status | _status_bit(
            ~z4c.evidence.forward_successful, CoupledEvolutionStatus.Z4C_REJECTED
        )
        status = status | _status_bit(
            ~matter.evidence.forward_successful,
            CoupledEvolutionStatus.MATTER_REJECTED,
        )
        status = status | _status_bit(~finite, CoupledEvolutionStatus.NONFINITE)
        status = status | _status_bit(
            ~(step_limits & budget_limits & exchange_limits),
            CoupledEvolutionStatus.LEDGER_LIMIT_EXCEEDED,
        )
        status = status | _status_bit(
            self.policy.require_derivative_valid & ~derivative_valid,
            CoupledEvolutionStatus.DERIVATIVE_INVALID,
        )
        status = status | _status_bit(~active, CoupledEvolutionStatus.TERMINAL)
        return _CoupledStageEvaluation(
            address,
            geometry,
            stress_energy,
            z4c,
            matter,
            ledgers,
            proposed_budget,
            status,
            finite,
            converged,
            physically_valid,
            qualified,
            derivative_valid,
            successful,
        )

    def advance(
        self,
        state: CoupledEvolutionState,
        step_size: ArrayLike,
        args: Any = None,
        /,
    ) -> CoupledStepResult:
        if not isinstance(state, CoupledEvolutionState):
            raise TypeError("state must be CoupledEvolutionState.")
        if state.runtime_id != self.operator_id or state.topology_id != self.topology_id:
            raise ValueError("Coupled state runtime/topology identity does not match.")
        step = _scalar(step_size, "coupled step_size", dtype=state.time.dtype)
        step_finite = jnp.isfinite(state.time) & jnp.isfinite(step)
        step_valid = (
            step_finite
            & (state.next_step_id >= 0)
            & (step > 0.0)
        )
        safe_step = jnp.where(step_valid, step, jnp.zeros_like(step))
        step_end = state.time + safe_step
        active = ~state.terminal

        working_z4c = state.z4c
        working_matter = state.matter
        working_budget = state.budget
        stages: list[_CoupledStageEvaluation] = []
        for stage_id, fraction in enumerate(self.stage_time_fractions):
            address = CoupledStageAddress(
                state.time,
                state.time + fraction * safe_step,
                step_end,
                state.next_step_id,
                stage_id,
                topology_id=self.topology_id,
            )
            stage = self._evaluate_stage(
                state.z4c,
                working_z4c,
                state.matter,
                working_matter,
                working_budget,
                address,
                args,
                step_valid,
                step_finite,
                active,
            )
            stages.append(stage)
            working_z4c = stage.z4c_proposal.candidate
            working_matter = stage.matter_proposal.candidate
            working_budget = stage.proposed_budget

        stage_values = tuple(stages)
        if len(stage_values) != _SSPRK33_STAGE_COUNT:
            raise RuntimeError("The fixed SSPRK33 stage count changed.")
        addresses = tuple(value.address for value in stage_values)
        geometries = tuple(value.geometry for value in stage_values)
        stress_energy = tuple(value.stress_energy for value in stage_values)
        z4c_proposals = tuple(value.z4c_proposal for value in stage_values)
        matter_proposals = tuple(value.matter_proposal for value in stage_values)
        stage_ledgers = tuple(value.ledgers for value in stage_values)
        ledgers = CoupledStepLedgers.combine(stage_ledgers)
        proposed_budget = state.budget.accumulate(ledgers)
        stage_status = jnp.stack(tuple(value.status for value in stage_values))
        stage_successful = jnp.stack(
            tuple(value.successful for value in stage_values)
        )
        snapshot_tokens = jnp.stack(
            tuple(value.snapshot_token for value in geometries)
        )
        snapshot_identity_valid = (
            jnp.all(snapshot_tokens != 0)
            & (snapshot_tokens[0] != snapshot_tokens[1])
            & (snapshot_tokens[0] != snapshot_tokens[2])
            & (snapshot_tokens[1] != snapshot_tokens[2])
        )
        status = stage_status[0]
        for stage_value in stage_status[1:]:
            status = status | stage_value
        status = status | _status_bit(
            ~snapshot_identity_valid,
            CoupledEvolutionStatus.STAGE_IDENTITY_MISMATCH,
        )
        finite = jnp.all(jnp.stack(tuple(value.finite for value in stage_values)))
        converged = jnp.all(
            jnp.stack(tuple(value.converged for value in stage_values))
        )
        physically_valid = jnp.all(
            jnp.stack(tuple(value.physically_valid for value in stage_values))
        )
        qualified = (
            jnp.all(jnp.stack(tuple(value.qualified for value in stage_values)))
            & snapshot_identity_valid
        )
        derivative_valid = jnp.all(
            jnp.stack(tuple(value.derivative_valid for value in stage_values))
        )
        successful = jnp.all(stage_successful) & snapshot_identity_valid

        candidate = CoupledEvolutionState(
            working_z4c,
            working_matter,
            proposed_budget,
            step_end,
            state.next_step_id + jnp.int32(1),
            state.accepted_steps + jnp.int32(1),
            state.rejected_steps,
            jnp.int32(0),
            False,
            topology_id=self.topology_id,
            runtime_id=self.operator_id,
        )
        rejected_attempt = active & ~successful
        failures = jnp.where(
            successful,
            jnp.int32(0),
            state.consecutive_failures + rejected_attempt.astype(jnp.int32),
        )
        failure_limit = failures >= self.policy.maximum_consecutive_failures
        accepted = CoupledEvolutionState(
            _select_tree(successful, working_z4c, state.z4c),
            _select_tree(successful, working_matter, state.matter),
            _select_budget(successful, proposed_budget, state.budget),
            jnp.where(successful, step_end, state.time),
            jnp.where(
                successful,
                state.next_step_id + jnp.int32(1),
                state.next_step_id,
            ),
            state.accepted_steps + successful.astype(jnp.int32),
            state.rejected_steps + rejected_attempt.astype(jnp.int32),
            failures,
            state.terminal | (rejected_attempt & failure_limit),
            topology_id=self.topology_id,
            runtime_id=self.operator_id,
        )
        status = status | _status_bit(
            rejected_attempt & failure_limit,
            CoupledEvolutionStatus.FAILURE_LIMIT_REACHED,
        )
        status = status | _status_bit(state.terminal, CoupledEvolutionStatus.TERMINAL)
        return CoupledStepResult(
            candidate,
            accepted,
            addresses,
            geometries,
            stress_energy,
            z4c_proposals,
            matter_proposals,
            stage_ledgers,
            ledgers,
            stage_status,
            stage_successful,
            status,
            finite,
            converged,
            physically_valid,
            qualified,
            derivative_valid,
            successful,
            successful,
            active,
            self.operator_id,
        )
