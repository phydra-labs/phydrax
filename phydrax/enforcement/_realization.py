#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from enum import Enum
from typing import Any, TYPE_CHECKING, TypeAlias

import equinox as eqx

from .._frozendict import frozendict
from .._strict import StrictModule


if TYPE_CHECKING:
    from phydrax.conditions import Condition, ConditionRealizationStamp

    from ._lifecycle import RealizationLifecycleState


FieldMap: TypeAlias = Mapping[str, Any]


class RealizationStatus(Enum):
    """Terminal status of one deterministic field-realization attempt."""

    SUCCESS = "success"
    UNCHANGED = "unchanged"
    INVALID_INPUT = "invalid_input"
    SOURCE_UNAVAILABLE = "source_unavailable"
    REFRESH_FAILED = "refresh_failed"
    SOLVE_FAILED = "solve_failed"
    VALIDATION_FAILED = "validation_failed"
    NONFINITE = "nonfinite"
    UNSUPPORTED = "unsupported"

    @property
    def successful(self) -> bool:
        return self in (RealizationStatus.SUCCESS, RealizationStatus.UNCHANGED)


class ConditionEvaluationContext(StrictModule):
    """Explicit, replayable inputs to a condition-realization attempt.

    ``accepted_step`` is the only step coordinate used to address randomized
    realization sources. Retries therefore reuse the same random address and do
    not consume randomness. ``attempt`` remains available for diagnostics only.
    """

    condition: Condition
    accepted_step: int = eqx.field(static=True)
    attempt: int = eqx.field(static=True)
    time: Any
    caller_sources: frozendict[str, Any]
    parameters: frozendict[str, Any]
    parameter_revision: int = eqx.field(static=True)
    adaptive_sources: frozenset[str] = eqx.field(static=True)
    prng_key: Any
    exact_required: bool = eqx.field(static=True)

    def __init__(
        self,
        condition: Condition,
        /,
        *,
        accepted_step: int = 0,
        attempt: int = 0,
        time: Any = None,
        caller_sources: Mapping[str, Any] = frozendict(),
        parameters: Mapping[str, Any] = frozendict(),
        parameter_revision: int = 0,
        adaptive_sources: frozenset[str] = frozenset(),
        prng_key: Any = None,
        exact_required: bool = False,
    ):
        step = int(accepted_step)
        attempt_ = int(attempt)
        revision = int(parameter_revision)
        if step < 0 or attempt_ < 0 or revision < 0:
            raise ValueError(
                "Accepted step, attempt, and parameter revision must be nonnegative."
            )
        caller = frozendict(caller_sources)
        parameter_values = frozendict(parameters)
        requested = frozenset(str(name) for name in adaptive_sources)
        if any(not name for name in (*caller, *parameter_values, *requested)):
            raise ValueError("Condition evaluation source names must be nonempty.")
        self.condition = condition
        self.accepted_step = step
        self.attempt = attempt_
        self.time = time
        self.caller_sources = caller
        self.parameters = parameter_values
        self.parameter_revision = revision
        self.adaptive_sources = requested
        self.prng_key = prng_key
        self.exact_required = bool(exact_required)

    @property
    def condition_id(self) -> str:
        return self.condition.condition_id

    @property
    def quantifier(self):
        return self.condition.quantifier


class FieldRealizationResult(StrictModule):
    """Outcome of realizing fields, with failed candidates deliberately withheld."""

    status: RealizationStatus = eqx.field(static=True)
    fields: frozendict[str, Any] | None
    state: RealizationLifecycleState
    stamp: ConditionRealizationStamp | None
    evidence: Any
    message: str = eqx.field(static=True)

    def __init__(
        self,
        status: RealizationStatus,
        fields: Mapping[str, Any] | None,
        state: RealizationLifecycleState,
        /,
        *,
        stamp: ConditionRealizationStamp | None = None,
        evidence: Any = None,
        message: str = "",
    ):
        if not isinstance(status, RealizationStatus):
            raise TypeError("Field realization status must be a RealizationStatus.")
        successful = status.successful
        if successful and fields is None:
            raise ValueError("A successful realization must return committed fields.")
        if successful and stamp is None:
            raise ValueError("A successful realization must carry its realization stamp.")
        if not successful and fields is not None:
            raise ValueError("A failed realization cannot expose a failed field iterate.")
        message_ = str(message)
        if not successful and not message_:
            raise ValueError("A failed realization must explain its failure.")
        self.status = status
        self.fields = None if fields is None else frozendict(fields)
        self.state = state
        self.stamp = stamp
        self.evidence = evidence
        self.message = message_

    @property
    def successful(self) -> bool:
        return self.status.successful

    @classmethod
    def success(
        cls,
        fields: Mapping[str, Any],
        /,
        *,
        state: RealizationLifecycleState,
        stamp: ConditionRealizationStamp,
        evidence: Any = None,
        unchanged: bool = False,
        message: str = "",
    ) -> FieldRealizationResult:
        status = RealizationStatus.UNCHANGED if unchanged else RealizationStatus.SUCCESS
        return cls(
            status,
            fields,
            state,
            stamp=stamp,
            evidence=evidence,
            message=message,
        )

    @classmethod
    def failure(
        cls,
        status: RealizationStatus,
        /,
        *,
        state: RealizationLifecycleState,
        message: str,
        evidence: Any = None,
    ) -> FieldRealizationResult:
        if status.successful:
            raise ValueError("FieldRealizationResult.failure requires a failure status.")
        return cls(status, None, state, evidence=evidence, message=message)


class AbstractFieldRealization(StrictModule):
    """Deterministic realization of declarative conditions into accepted fields.

    Probabilistic conditioning is intentionally not part of this interface.
    Implementations return explicit state and evidence and may never mutate the
    supplied fields or lifecycle state.
    """

    @abstractmethod
    def realize(
        self,
        fields: FieldMap,
        state: RealizationLifecycleState | None = None,
        *,
        context: ConditionEvaluationContext,
    ) -> FieldRealizationResult:
        raise NotImplementedError


__all__ = [
    "AbstractFieldRealization",
    "ConditionEvaluationContext",
    "FieldMap",
    "FieldRealizationResult",
    "RealizationStatus",
]
