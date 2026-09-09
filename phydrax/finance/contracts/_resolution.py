#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import IntFlag
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..core import CalendarSnapshot, FinancialTimestamp
from ..market._snapshots import ReferenceDataSnapshot
from ._base import AbstractContract, AbstractResolvedContract
from ._cashflows import CashflowBatch, CashflowStatus, PreparedCashflowBatch
from ._exercise import ExerciseSchedule, SettlementTerms


class ContractResolutionStatus(IntFlag):
    """Composable fail-closed status bits for contract host resolution."""

    SUCCESS = 0
    INVALID_REFERENCE = 1
    CALENDAR_MISMATCH = 2
    SCHEDULE_OVERFLOW = 4
    DUPLICATE_OBLIGATION = 8
    CAUSAL_TIME_VIOLATION = 16
    INCOMPATIBLE_SETTLEMENT = 32
    REPLAY_MISMATCH = 64


class ContractResolutionContext(StrictModule, NonTrainableState):
    """Pinned host resources for deterministic contract resolution."""

    reference_data: ReferenceDataSnapshot
    calendars: tuple[CalendarSnapshot, ...]
    as_of: FinancialTimestamp = eqx.field(static=True)
    cashflow_capacity: int = eqx.field(static=True)
    context_status: ContractResolutionStatus = eqx.field(static=True)
    context_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_data: ReferenceDataSnapshot,
        calendars: Sequence[CalendarSnapshot],
        as_of: FinancialTimestamp,
        /,
        *,
        cashflow_capacity: int,
    ):
        if not isinstance(reference_data, ReferenceDataSnapshot):
            raise TypeError("reference_data must be a ReferenceDataSnapshot.")
        calendar_values = tuple(calendars)
        if not calendar_values or not all(
            isinstance(value, CalendarSnapshot) for value in calendar_values
        ):
            raise TypeError("calendars must contain at least one CalendarSnapshot.")
        calendar_ids = tuple(value.calendar_id for value in calendar_values)
        if len(set(calendar_ids)) != len(calendar_ids):
            raise ValueError("Contract resolution calendar IDs must be unique.")
        if not isinstance(as_of, FinancialTimestamp):
            raise TypeError("as_of must be a FinancialTimestamp.")
        if (
            isinstance(cashflow_capacity, bool)
            or not isinstance(cashflow_capacity, int)
            or cashflow_capacity < 1
        ):
            raise ValueError("cashflow_capacity must be a positive integer.")
        status = ContractResolutionStatus.SUCCESS
        if reference_data.as_of.available_ns > as_of.epoch_nanoseconds:
            status |= ContractResolutionStatus.CAUSAL_TIME_VIOLATION
        ordered = tuple(sorted(calendar_values, key=lambda value: value.calendar_id))
        self.reference_data = reference_data
        self.calendars = ordered
        self.as_of = as_of
        self.cashflow_capacity = cashflow_capacity
        self.context_status = status
        self.context_id = canonical_fingerprint(
            {
                "kind": "financial-contract-resolution-context",
                "reference_data": reference_data.snapshot_id,
                "calendars": [value.snapshot_id for value in ordered],
                "as_of_ns": int(as_of.epoch_nanoseconds),
                "as_of_available_ns": int(as_of.available_ns),
                "as_of_vintage_id": as_of.vintage_id,
                "cashflow_capacity": cashflow_capacity,
                "context_status": int(status),
            }
        )

    @property
    def accepted(self) -> bool:
        return self.context_status == ContractResolutionStatus.SUCCESS

    def calendar(self, calendar_id: str, /) -> CalendarSnapshot:
        if not isinstance(calendar_id, str) or not calendar_id.strip():
            raise ValueError("calendar_id must be a non-empty string.")
        normalized = calendar_id.strip()
        for value in self.calendars:
            if value.calendar_id == normalized:
                return value
        raise KeyError(
            f"Calendar {normalized!r} is not pinned in the resolution context."
        )


class ResolvedContract(AbstractResolvedContract):
    """Generic resolved carrier for known obligations and contractual conventions."""

    definition: AbstractContract
    _cashflows: CashflowBatch
    settlement: SettlementTerms
    exercise: ExerciseSchedule | None
    resolution_status: Array
    contract_id: str = eqx.field(static=True)
    resolved_id: str = eqx.field(static=True)

    def __init__(
        self,
        definition: AbstractContract,
        cashflows: CashflowBatch,
        settlement: SettlementTerms,
        /,
        *,
        exercise: ExerciseSchedule | None = None,
        resolution_status: int
        | ContractResolutionStatus = ContractResolutionStatus.SUCCESS,
    ):
        if not isinstance(definition, AbstractContract):
            raise TypeError("definition must be an AbstractContract.")
        if not isinstance(cashflows, CashflowBatch):
            raise TypeError("cashflows must be a CashflowBatch.")
        if not isinstance(settlement, SettlementTerms):
            raise TypeError("settlement must be SettlementTerms.")
        if exercise is not None and not isinstance(exercise, ExerciseSchedule):
            raise TypeError("exercise must be ExerciseSchedule or None.")
        status = ContractResolutionStatus(int(resolution_status))
        self.definition = definition
        self._cashflows = cashflows
        self.settlement = settlement
        self.exercise = exercise
        self.resolution_status = jnp.asarray(int(status), dtype=jnp.int32)
        self.contract_id = definition.contract_id
        self.resolved_id = canonical_fingerprint(
            {
                "kind": "resolved-financial-contract",
                "contract": definition.contract_id,
                "cashflows": cashflows.batch_id,
                "settlement": settlement.terms_id,
                "exercise": None if exercise is None else exercise.schedule_id,
                "resolution_status": int(status),
            }
        )

    @property
    def known_cashflows(self) -> CashflowBatch:
        return self._cashflows

    @property
    def accepted(self) -> Array:
        return self.resolution_status == int(ContractResolutionStatus.SUCCESS)

    def prepare(self, /, *, capacity: int) -> PreparedResolvedContract:
        prepared_cashflows = self._cashflows.prepare(capacity)
        status = self.resolution_status
        status = status | jnp.where(
            prepared_cashflows.preparation_status == int(CashflowStatus.SUCCESS),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(int(ContractResolutionStatus.SCHEDULE_OVERFLOW), dtype=jnp.int32),
        )
        return PreparedResolvedContract(
            prepared_cashflows,
            self.settlement,
            self.exercise,
            status,
            self.resolved_id,
        )

    def to_record(self) -> Mapping[str, Any]:
        return {
            "contract_id": self.contract_id,
            "cashflow_batch_id": self._cashflows.batch_id,
            "settlement_terms_id": self.settlement.terms_id,
            "exercise_schedule_id": None
            if self.exercise is None
            else self.exercise.schedule_id,
            "resolution_status": int(self.resolution_status),
            "resolved_id": self.resolved_id,
        }


class PreparedResolvedContract(StrictModule, NonTrainableState):
    """Fixed-shape known contract obligations ready for device computation."""

    cashflows: PreparedCashflowBatch
    settlement: SettlementTerms
    exercise: ExerciseSchedule | None
    status: Array
    resolved_id: str = eqx.field(static=True)

    @property
    def accepted(self) -> Array:
        return (self.status == int(ContractResolutionStatus.SUCCESS)) & jnp.all(
            ~self.cashflows.cashflows.valid_mask | self.cashflows.accepted
        )


class ContractResolutionPlan(StrictModule, NonTrainableState):
    """Pinned deterministic invocation of a product-specific host resolver."""

    definition: AbstractContract
    context: ContractResolutionContext
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        definition: AbstractContract,
        context: ContractResolutionContext,
        /,
    ):
        if not isinstance(definition, AbstractContract):
            raise TypeError("definition must be an AbstractContract.")
        if not isinstance(context, ContractResolutionContext):
            raise TypeError("context must be a ContractResolutionContext.")
        self.definition = definition
        self.context = context
        self.plan_id = canonical_fingerprint(
            {
                "kind": "financial-contract-resolution-plan",
                "contract": definition.contract_id,
                "context": context.context_id,
            }
        )

    def resolve(self) -> AbstractResolvedContract:
        return resolve_contract(self.definition, self.context)

    def replay(self, expected_resolved_id: str, /) -> ContractResolutionReplay:
        if not isinstance(expected_resolved_id, str) or not expected_resolved_id:
            raise ValueError("expected_resolved_id must be a non-empty string.")
        resolved = self.resolve()
        matches = resolved.resolved_id == expected_resolved_id
        status = (
            ContractResolutionStatus.SUCCESS
            if matches
            else ContractResolutionStatus.REPLAY_MISMATCH
        )
        return ContractResolutionReplay(
            resolved,
            jnp.asarray(matches),
            jnp.asarray(int(status), dtype=jnp.int32),
            self.plan_id,
        )


class ContractResolutionReplay(StrictModule, NonTrainableState):
    resolved: AbstractResolvedContract
    matches: Array
    status: Array
    plan_id: str = eqx.field(static=True)


def resolve_contract(
    definition: AbstractContract,
    context: ContractResolutionContext,
    /,
) -> AbstractResolvedContract:
    """Execute one product resolver and validate its semantic identity boundary."""

    if not isinstance(definition, AbstractContract):
        raise TypeError("definition must be an AbstractContract.")
    if not isinstance(context, ContractResolutionContext):
        raise TypeError("context must be a ContractResolutionContext.")
    resolved = definition.resolve(context)
    if not isinstance(resolved, AbstractResolvedContract):
        raise TypeError("A contract resolver must return an AbstractResolvedContract.")
    if resolved.contract_id != definition.contract_id:
        raise ValueError("Resolved contract identity does not match its definition.")
    return resolved


__all__ = [
    "ContractResolutionContext",
    "ContractResolutionPlan",
    "ContractResolutionReplay",
    "ContractResolutionStatus",
    "PreparedResolvedContract",
    "ResolvedContract",
    "resolve_contract",
]
