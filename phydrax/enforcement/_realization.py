#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, TypeAlias

import equinox as eqx

from .._frozendict import frozendict
from .._strict import StrictModule
from ..conditions._evidence import ConditionRealizationStamp
from ..conditions._ir import Condition, ConditionQuantifier
from ..typing import checked


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

    @checked
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
    ) -> None:
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
    def quantifier(self) -> ConditionQuantifier:
        return self.condition.quantifier


class FieldRealizationResult(StrictModule):
    """Outcome of realizing fields, with failed candidates deliberately withheld."""

    status: RealizationStatus = eqx.field(static=True)
    fields: frozendict[str, Any] | None
    state: _lifecycle.RealizationLifecycleState
    stamp: ConditionRealizationStamp | None
    evidence: Any
    message: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        status: RealizationStatus,
        fields: Mapping[str, Any] | None,
        state: _lifecycle.RealizationLifecycleState,
        /,
        *,
        stamp: ConditionRealizationStamp | None = None,
        evidence: Any = None,
        message: str = "",
    ) -> None:
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
    @checked
    def success(
        cls,
        fields: Mapping[str, Any],
        /,
        *,
        state: _lifecycle.RealizationLifecycleState,
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
    @checked
    def failure(
        cls,
        status: RealizationStatus,
        /,
        *,
        state: _lifecycle.RealizationLifecycleState,
        message: str,
        evidence: Any = None,
    ) -> FieldRealizationResult:
        if status.successful:
            raise ValueError("FieldRealizationResult.failure requires a failure status.")
        return cls(status, None, state, evidence=evidence, message=message)


class RealizationAdmission(StrictModule):
    """Composition contract of one field realization.

    ``reads`` and ``writes`` name the solver fields the realization reads and may
    replace. ``writes_unknown`` marks a realization whose correction chart can
    replace any field it receives; composition treats it as writing every field.
    ``establishes`` lists the condition identities the realization enforces and
    ``preserves`` the identities of other established contracts it provably keeps
    satisfied. Condition sources are not a substitute for the write set.
    """

    reads: tuple[str, ...] = eqx.field(static=True)
    writes: tuple[str, ...] = eqx.field(static=True)
    writes_unknown: bool = eqx.field(static=True)
    establishes: tuple[str, ...] = eqx.field(static=True)
    preserves: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        reads: Sequence[str],
        writes: Sequence[str],
        establishes: Sequence[str],
        writes_unknown: bool = False,
        preserves: Sequence[str] = (),
    ) -> None:
        def names(values: Sequence[str], label: str, /) -> tuple[str, ...]:
            result = tuple(str(value) for value in values)
            if any(not value for value in result) or len(set(result)) != len(result):
                raise ValueError(f"Admission {label} must be unique nonempty names.")
            return result

        if not isinstance(writes_unknown, bool):
            raise TypeError("writes_unknown must be boolean.")
        self.reads = names(reads, "reads")
        self.writes = names(writes, "writes")
        self.writes_unknown = writes_unknown
        self.establishes = names(establishes, "established conditions")
        self.preserves = names(preserves, "preserved conditions")

    def may_write(self, field: str, /) -> bool:
        """Whether this realization may replace ``field``."""
        return self.writes_unknown or field in self.writes


class AbstractFieldRealization(StrictModule):
    """Deterministic realization of declarative conditions into accepted fields.

    Probabilistic conditioning is intentionally not part of this interface.
    Implementations return explicit state and evidence and may never mutate the
    supplied fields or lifecycle state. Every realization declares its
    composition contract through `admission`.
    """

    @abstractmethod
    def realize(
        self,
        fields: FieldMap,
        state: _lifecycle.RealizationLifecycleState | None = None,
        *,
        context: ConditionEvaluationContext,
    ) -> FieldRealizationResult:
        raise NotImplementedError

    @abstractmethod
    def admission(self, condition: Condition, /) -> RealizationAdmission:
        """Return the read/write and preservation contract for ``condition``."""
        raise NotImplementedError


# Lifecycle imports the context and result above; resolve its nominal annotations
# through the module after those definitions exist, including during circular import.
from . import _lifecycle


__all__ = [
    "AbstractFieldRealization",
    "ConditionEvaluationContext",
    "FieldMap",
    "FieldRealizationResult",
    "RealizationAdmission",
    "RealizationStatus",
]
