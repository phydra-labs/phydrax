#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from typing import Any, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax.random as jr

from .._frozendict import frozendict
from .._strict import AbstractAttribute, StrictModule
from ._realization import (
    ConditionEvaluationContext,
    FieldRealizationResult,
    RealizationStatus,
)


if TYPE_CHECKING:
    from phydrax.conditions import ConditionRealizationStamp


SourceProvider: TypeAlias = Callable[..., Any]
RefreshValidator: TypeAlias = Callable[
    [Mapping[str, Any], "RefreshProposal"], "RefreshValidation"
]


class RealizationLifecyclePhase(Enum):
    UNINITIALIZED = "uninitialized"
    READY = "ready"
    FAILED = "failed"


class RealizationSourceKind(Enum):
    FIXED = "fixed"
    CALLER = "caller"
    PER_STEP = "per_step"
    ADAPTIVE = "adaptive"
    PARAMETERIZED = "parameterized"
    RANDOMIZED = "randomized"


class RefreshProposalStatus(Enum):
    READY = "ready"
    UNCHANGED = "unchanged"
    FAILED = "failed"


class RealizationFailure(StrictModule):
    status: RealizationStatus = eqx.field(static=True)
    message: str = eqx.field(static=True)
    accepted_step: int = eqx.field(static=True)
    attempt: int = eqx.field(static=True)
    evidence: Any

    def __init__(
        self,
        status: RealizationStatus,
        message: str,
        /,
        *,
        accepted_step: int,
        attempt: int,
        evidence: Any = None,
    ):
        if status.successful:
            raise ValueError("A realization failure requires a failure status.")
        message_ = str(message)
        if not message_:
            raise ValueError("A realization failure requires a message.")
        step = int(accepted_step)
        attempt_ = int(attempt)
        if step < 0 or attempt_ < 0:
            raise ValueError("Failure coordinates must be nonnegative.")
        self.status = status
        self.message = message_
        self.accepted_step = step
        self.attempt = attempt_
        self.evidence = evidence


class RealizationSourceStamp(StrictModule):
    name: str = eqx.field(static=True)
    kind: RealizationSourceKind = eqx.field(static=True)
    accepted_step: int = eqx.field(static=True)
    parameter_revision: int = eqx.field(static=True)
    generation: int = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        kind: RealizationSourceKind,
        /,
        *,
        accepted_step: int,
        parameter_revision: int,
        generation: int,
    ):
        name_ = str(name)
        step = int(accepted_step)
        parameter_revision_ = int(parameter_revision)
        generation_ = int(generation)
        if not name_:
            raise ValueError("A realization source stamp requires a name.")
        if step < 0 or parameter_revision_ < 0 or generation_ < 0:
            raise ValueError("Realization source stamp coordinates must be nonnegative.")
        self.name = name_
        self.kind = kind
        self.accepted_step = step
        self.parameter_revision = parameter_revision_
        self.generation = generation_


class _SourceResolution(StrictModule):
    available: bool = eqx.field(static=True)
    value: Any
    message: str = eqx.field(static=True)

    def __init__(self, available: bool, value: Any = None, message: str = ""):
        available_ = bool(available)
        message_ = str(message)
        if not available_ and not message_:
            raise ValueError("An unavailable source requires a message.")
        self.available = available_
        self.value = value
        self.message = message_


class AbstractRealizationSource(StrictModule):
    """Immutable declaration of when and how a realization input is refreshed."""

    name: AbstractAttribute[str]
    kind: AbstractAttribute[RealizationSourceKind]

    @abstractmethod
    def needs_refresh(
        self,
        state: RealizationLifecycleState,
        context: ConditionEvaluationContext,
        /,
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def resolve(
        self,
        state: RealizationLifecycleState,
        context: ConditionEvaluationContext,
        /,
    ) -> _SourceResolution:
        raise NotImplementedError


def _source_name(name: str, /) -> str:
    value = str(name)
    if not value:
        raise ValueError("A realization source requires a nonempty name.")
    return value


def _step_changed(
    name: str,
    state: RealizationLifecycleState,
    context: ConditionEvaluationContext,
    /,
) -> bool:
    stamp = state.source_stamps.get(name)
    return stamp is None or stamp.accepted_step != context.accepted_step


class FixedRealizationSource(AbstractRealizationSource):
    name: str = eqx.field(static=True)
    value: Any
    kind: RealizationSourceKind = eqx.field(static=True)

    def __init__(self, name: str, value: Any, /):
        self.name = _source_name(name)
        self.value = value
        self.kind = RealizationSourceKind.FIXED

    def needs_refresh(self, state, context, /) -> bool:
        del context
        return self.name not in state.values

    def resolve(self, state, context, /) -> _SourceResolution:
        del state, context
        return _SourceResolution(True, self.value)


class CallerRealizationSource(AbstractRealizationSource):
    name: str = eqx.field(static=True)
    required: bool = eqx.field(static=True)
    has_default: bool = eqx.field(static=True)
    default: Any
    kind: RealizationSourceKind = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        /,
        *,
        required: bool = True,
        default: Any = None,
        has_default: bool = False,
    ):
        required_ = bool(required)
        has_default_ = bool(has_default)
        if required_ and has_default_:
            raise ValueError("A required caller source cannot also declare a default.")
        self.name = _source_name(name)
        self.required = required_
        self.has_default = has_default_
        self.default = default
        self.kind = RealizationSourceKind.CALLER

    def needs_refresh(self, state, context, /) -> bool:
        return self.name in context.caller_sources or self.name not in state.values

    def resolve(self, state, context, /) -> _SourceResolution:
        if self.name in context.caller_sources:
            return _SourceResolution(True, context.caller_sources[self.name])
        if self.name in state.values:
            return _SourceResolution(True, state.values[self.name])
        if self.has_default:
            return _SourceResolution(True, self.default)
        if not self.required:
            return _SourceResolution(True, None)
        return _SourceResolution(
            False,
            message=f"Required caller realization source {self.name!r} is unavailable.",
        )


class PerStepRealizationSource(AbstractRealizationSource):
    name: str = eqx.field(static=True)
    provider: SourceProvider = eqx.field(static=True)
    kind: RealizationSourceKind = eqx.field(static=True)

    def __init__(self, name: str, provider: SourceProvider, /):
        if not callable(provider):
            raise TypeError("A per-step realization source provider must be callable.")
        self.name = _source_name(name)
        self.provider = provider
        self.kind = RealizationSourceKind.PER_STEP

    def needs_refresh(self, state, context, /) -> bool:
        return _step_changed(self.name, state, context)

    def resolve(self, state, context, /) -> _SourceResolution:
        del state
        return _SourceResolution(True, self.provider(context))


class AdaptiveRealizationSource(AbstractRealizationSource):
    name: str = eqx.field(static=True)
    provider: SourceProvider = eqx.field(static=True)
    kind: RealizationSourceKind = eqx.field(static=True)

    def __init__(self, name: str, provider: SourceProvider, /):
        if not callable(provider):
            raise TypeError("An adaptive realization source provider must be callable.")
        self.name = _source_name(name)
        self.provider = provider
        self.kind = RealizationSourceKind.ADAPTIVE

    def needs_refresh(self, state, context, /) -> bool:
        return self.name not in state.values or self.name in context.adaptive_sources

    def resolve(self, state, context, /) -> _SourceResolution:
        previous = state.values.get(self.name)
        return _SourceResolution(True, self.provider(previous, context))


class ParameterizedRealizationSource(AbstractRealizationSource):
    name: str = eqx.field(static=True)
    provider: SourceProvider = eqx.field(static=True)
    kind: RealizationSourceKind = eqx.field(static=True)

    def __init__(self, name: str, provider: SourceProvider, /):
        if not callable(provider):
            raise TypeError(
                "A parameterized realization source provider must be callable."
            )
        self.name = _source_name(name)
        self.provider = provider
        self.kind = RealizationSourceKind.PARAMETERIZED

    def needs_refresh(self, state, context, /) -> bool:
        stamp = state.source_stamps.get(self.name)
        return stamp is None or stamp.parameter_revision != context.parameter_revision

    def resolve(self, state, context, /) -> _SourceResolution:
        del state
        return _SourceResolution(True, self.provider(context.parameters, context))


def address_accepted_step_key(key: Any, accepted_step: int, stream: int = 0, /):
    """Derive a stable PRNG address without consuming state or counting retries."""

    step = int(accepted_step)
    stream_ = int(stream)
    if step < 0 or stream_ < 0:
        raise ValueError("Accepted-step PRNG coordinates must be nonnegative.")
    return jr.fold_in(jr.fold_in(key, stream_), step)


class RandomizedRealizationSource(AbstractRealizationSource):
    """Seed-addressed numerical randomization, not probabilistic conditioning."""

    name: str = eqx.field(static=True)
    provider: SourceProvider = eqx.field(static=True)
    stream: int = eqx.field(static=True)
    kind: RealizationSourceKind = eqx.field(static=True)

    def __init__(self, name: str, provider: SourceProvider, /, *, stream: int = 0):
        stream_ = int(stream)
        if not callable(provider):
            raise TypeError("A randomized realization source provider must be callable.")
        if stream_ < 0:
            raise ValueError(
                "A randomized realization source stream must be nonnegative."
            )
        self.name = _source_name(name)
        self.provider = provider
        self.stream = stream_
        self.kind = RealizationSourceKind.RANDOMIZED

    def needs_refresh(self, state, context, /) -> bool:
        return _step_changed(self.name, state, context)

    def resolve(self, state, context, /) -> _SourceResolution:
        del state
        if context.prng_key is None:
            return _SourceResolution(
                False,
                message=f"Randomized realization source {self.name!r} requires a PRNG key.",
            )
        key = address_accepted_step_key(
            context.prng_key, context.accepted_step, self.stream
        )
        return _SourceResolution(True, self.provider(key, context))


RealizationSource: TypeAlias = (
    FixedRealizationSource
    | CallerRealizationSource
    | PerStepRealizationSource
    | AdaptiveRealizationSource
    | ParameterizedRealizationSource
    | RandomizedRealizationSource
)


class RealizationLifecycleState(StrictModule):
    """Checkpointable committed source state for one field realization."""

    phase: RealizationLifecyclePhase = eqx.field(static=True)
    generation: int = eqx.field(static=True)
    accepted_step: int = eqx.field(static=True)
    parameter_revision: int = eqx.field(static=True)
    values: frozendict[str, Any]
    source_stamps: frozendict[str, RealizationSourceStamp]
    realization_stamp: ConditionRealizationStamp | None
    last_failure: RealizationFailure | None

    def __init__(
        self,
        *,
        phase: RealizationLifecyclePhase = RealizationLifecyclePhase.UNINITIALIZED,
        generation: int = 0,
        accepted_step: int = 0,
        parameter_revision: int = 0,
        values: Mapping[str, Any] = frozendict(),
        source_stamps: Mapping[str, RealizationSourceStamp] = frozendict(),
        realization_stamp: ConditionRealizationStamp | None = None,
        last_failure: RealizationFailure | None = None,
    ):
        generation_ = int(generation)
        step = int(accepted_step)
        revision = int(parameter_revision)
        if generation_ < 0 or step < 0 or revision < 0:
            raise ValueError("Lifecycle coordinates must be nonnegative.")
        values_ = frozendict(values)
        stamps_ = frozendict(source_stamps)
        if set(values_) != set(stamps_):
            raise ValueError(
                "Committed source values and stamps must have identical names."
            )
        if phase is RealizationLifecyclePhase.UNINITIALIZED and values_:
            raise ValueError("An uninitialized lifecycle cannot contain source values.")
        if phase is RealizationLifecyclePhase.FAILED and last_failure is None:
            raise ValueError("A failed lifecycle requires explicit failure evidence.")
        self.phase = phase
        self.generation = generation_
        self.accepted_step = step
        self.parameter_revision = revision
        self.values = values_
        self.source_stamps = stamps_
        self.realization_stamp = realization_stamp
        self.last_failure = last_failure

    @classmethod
    def initial(cls) -> RealizationLifecycleState:
        return cls()


class RefreshProposal(StrictModule):
    status: RefreshProposalStatus = eqx.field(static=True)
    base_generation: int = eqx.field(static=True)
    accepted_step: int = eqx.field(static=True)
    attempt: int = eqx.field(static=True)
    parameter_revision: int = eqx.field(static=True)
    values: frozendict[str, Any] | None
    source_stamps: frozendict[str, RealizationSourceStamp] | None
    refreshed: tuple[str, ...] = eqx.field(static=True)
    message: str = eqx.field(static=True)

    def __init__(
        self,
        status: RefreshProposalStatus,
        /,
        *,
        base_generation: int,
        accepted_step: int,
        attempt: int,
        parameter_revision: int,
        values: Mapping[str, Any] | None,
        source_stamps: Mapping[str, RealizationSourceStamp] | None,
        refreshed: Sequence[str] = (),
        message: str = "",
    ):
        successful = status is not RefreshProposalStatus.FAILED
        if successful and (values is None or source_stamps is None):
            raise ValueError(
                "A successful refresh proposal requires complete candidates."
            )
        if not successful and (values is not None or source_stamps is not None):
            raise ValueError(
                "A failed refresh proposal cannot expose partial candidates."
            )
        message_ = str(message)
        if not successful and not message_:
            raise ValueError("A failed refresh proposal requires a message.")
        refreshed_ = tuple(str(name) for name in refreshed)
        if len(set(refreshed_)) != len(refreshed_):
            raise ValueError("Refreshed source names must be unique.")
        self.status = status
        self.base_generation = int(base_generation)
        self.accepted_step = int(accepted_step)
        self.attempt = int(attempt)
        self.parameter_revision = int(parameter_revision)
        self.values = None if values is None else frozendict(values)
        self.source_stamps = None if source_stamps is None else frozendict(source_stamps)
        self.refreshed = refreshed_
        self.message = message_

    @property
    def successful(self) -> bool:
        return self.status is not RefreshProposalStatus.FAILED


class RefreshValidation(StrictModule):
    status: RealizationStatus = eqx.field(static=True)
    message: str = eqx.field(static=True)
    evidence: Any

    def __init__(
        self,
        status: RealizationStatus,
        /,
        *,
        message: str = "",
        evidence: Any = None,
    ):
        message_ = str(message)
        if not status.successful and not message_:
            raise ValueError("A rejected refresh requires a message.")
        self.status = status
        self.message = message_
        self.evidence = evidence

    @property
    def accepted(self) -> bool:
        return self.status.successful

    @classmethod
    def accept(
        cls,
        *,
        unchanged: bool = False,
        message: str = "",
        evidence: Any = None,
    ) -> RefreshValidation:
        status = RealizationStatus.UNCHANGED if unchanged else RealizationStatus.SUCCESS
        return cls(status, message=message, evidence=evidence)

    @classmethod
    def reject(
        cls,
        status: RealizationStatus = RealizationStatus.VALIDATION_FAILED,
        /,
        *,
        message: str,
        evidence: Any = None,
    ) -> RefreshValidation:
        if status.successful:
            raise ValueError("RefreshValidation.reject requires a failure status.")
        return cls(status, message=message, evidence=evidence)


def propose_refresh(
    declarations: Sequence[RealizationSource],
    state: RealizationLifecycleState | None,
    /,
    *,
    context: ConditionEvaluationContext,
) -> RefreshProposal:
    """Materialize a complete candidate without changing committed state."""

    current = RealizationLifecycleState.initial() if state is None else state
    names = tuple(source.name for source in declarations)
    if len(set(names)) != len(names):
        raise ValueError("Realization source declaration names must be unique.")
    if context.accepted_step < current.accepted_step:
        return RefreshProposal(
            RefreshProposalStatus.FAILED,
            base_generation=current.generation,
            accepted_step=context.accepted_step,
            attempt=context.attempt,
            parameter_revision=context.parameter_revision,
            values=None,
            source_stamps=None,
            message="A refresh proposal cannot rewind the accepted-step coordinate.",
        )

    candidate_values = dict(current.values)
    candidate_stamps = dict(current.source_stamps)
    refreshed: list[str] = []
    next_generation = current.generation + 1
    for source in declarations:
        if not source.needs_refresh(current, context):
            continue
        resolution = source.resolve(current, context)
        if not resolution.available:
            return RefreshProposal(
                RefreshProposalStatus.FAILED,
                base_generation=current.generation,
                accepted_step=context.accepted_step,
                attempt=context.attempt,
                parameter_revision=context.parameter_revision,
                values=None,
                source_stamps=None,
                refreshed=tuple(refreshed),
                message=resolution.message,
            )
        candidate_values[source.name] = resolution.value
        candidate_stamps[source.name] = RealizationSourceStamp(
            source.name,
            source.kind,
            accepted_step=context.accepted_step,
            parameter_revision=context.parameter_revision,
            generation=next_generation,
        )
        refreshed.append(source.name)

    declared = frozenset(names)
    removed = tuple(name for name in current.values if name not in declared)
    refreshed.extend(removed)
    candidate_values = {
        name: value for name, value in candidate_values.items() if name in declared
    }
    candidate_stamps = {
        name: stamp for name, stamp in candidate_stamps.items() if name in declared
    }
    status = RefreshProposalStatus.READY if refreshed else RefreshProposalStatus.UNCHANGED
    return RefreshProposal(
        status,
        base_generation=current.generation,
        accepted_step=context.accepted_step,
        attempt=context.attempt,
        parameter_revision=context.parameter_revision,
        values=candidate_values,
        source_stamps=candidate_stamps,
        refreshed=tuple(refreshed),
    )


def validate_refresh(
    proposal: RefreshProposal,
    validator: RefreshValidator | None = None,
    /,
) -> RefreshValidation:
    """Validate a complete proposal without modifying either proposal or state."""

    if not proposal.successful:
        return RefreshValidation.reject(
            RealizationStatus.SOURCE_UNAVAILABLE,
            message=proposal.message,
        )
    if proposal.status is RefreshProposalStatus.UNCHANGED:
        return RefreshValidation.accept(unchanged=True)
    if validator is None:
        return RefreshValidation.accept()
    if proposal.values is None:
        raise RuntimeError("A successful refresh proposal lost its candidate values.")
    validation = validator(proposal.values, proposal)
    if not isinstance(validation, RefreshValidation):
        raise TypeError("A refresh validator must return RefreshValidation.")
    return validation


def _failed_lifecycle(
    state: RealizationLifecycleState,
    validation: RefreshValidation,
    proposal: RefreshProposal,
    /,
) -> RealizationLifecycleState:
    return RealizationLifecycleState(
        phase=RealizationLifecyclePhase.FAILED,
        generation=state.generation,
        accepted_step=state.accepted_step,
        parameter_revision=state.parameter_revision,
        values=state.values,
        source_stamps=state.source_stamps,
        realization_stamp=state.realization_stamp,
        last_failure=RealizationFailure(
            validation.status,
            validation.message,
            accepted_step=proposal.accepted_step,
            attempt=proposal.attempt,
            evidence=validation.evidence,
        ),
    )


def commit_refresh(
    state: RealizationLifecycleState | None,
    proposal: RefreshProposal,
    validation: RefreshValidation,
    /,
) -> RealizationLifecycleState:
    """Atomically accept all refreshed sources or preserve all committed sources."""

    current = RealizationLifecycleState.initial() if state is None else state
    if (
        proposal.base_generation != current.generation
        or proposal.accepted_step < current.accepted_step
    ):
        stale = RefreshValidation.reject(
            RealizationStatus.REFRESH_FAILED,
            message="Refresh proposal is stale relative to committed lifecycle state.",
        )
        return _failed_lifecycle(current, stale, proposal)
    if not proposal.successful and validation.accepted:
        inconsistent = RefreshValidation.reject(
            RealizationStatus.REFRESH_FAILED,
            message="A failed refresh proposal cannot be committed.",
        )
        return _failed_lifecycle(current, inconsistent, proposal)
    if not validation.accepted:
        return _failed_lifecycle(current, validation, proposal)
    if proposal.values is None or proposal.source_stamps is None:
        inconsistent = RefreshValidation.reject(
            RealizationStatus.REFRESH_FAILED,
            message="Accepted refresh is missing complete candidate state.",
        )
        return _failed_lifecycle(current, inconsistent, proposal)
    sources_changed = proposal.status is RefreshProposalStatus.READY
    coordinates_changed = (
        proposal.accepted_step != current.accepted_step
        or proposal.parameter_revision != current.parameter_revision
    )
    return RealizationLifecycleState(
        phase=RealizationLifecyclePhase.READY,
        generation=current.generation + int(sources_changed or coordinates_changed),
        accepted_step=proposal.accepted_step,
        parameter_revision=proposal.parameter_revision,
        values=proposal.values,
        source_stamps=proposal.source_stamps,
        realization_stamp=None if sources_changed else current.realization_stamp,
    )


def record_realization_stamp(
    state: RealizationLifecycleState,
    stamp: ConditionRealizationStamp,
    /,
) -> RealizationLifecycleState:
    """Record the stamp only after a successful field realization."""

    if state.phase is not RealizationLifecyclePhase.READY:
        raise ValueError("Only a ready lifecycle can accept a realization stamp.")
    return RealizationLifecycleState(
        phase=state.phase,
        generation=state.generation,
        accepted_step=state.accepted_step,
        parameter_revision=state.parameter_revision,
        values=state.values,
        source_stamps=state.source_stamps,
        realization_stamp=stamp,
    )


class EnforcementState(StrictModule):
    """Checkpointable accepted root for a collection of field realizations."""

    fields: frozendict[str, Any]
    realizations: frozendict[str, RealizationLifecycleState]
    accepted_step: int = eqx.field(static=True)
    generation: int = eqx.field(static=True)
    last_failure: RealizationFailure | None

    def __init__(
        self,
        fields: Mapping[str, Any] = frozendict(),
        realizations: Mapping[str, RealizationLifecycleState] = frozendict(),
        /,
        *,
        accepted_step: int = 0,
        generation: int = 0,
        last_failure: RealizationFailure | None = None,
    ):
        step = int(accepted_step)
        generation_ = int(generation)
        if step < 0 or generation_ < 0:
            raise ValueError("Enforcement state coordinates must be nonnegative.")
        realization_states = frozendict(realizations)
        if any(
            not isinstance(value, RealizationLifecycleState)
            for value in realization_states.values()
        ):
            raise TypeError("Enforcement realizations must be lifecycle states.")
        self.fields = frozendict(fields)
        self.realizations = realization_states
        self.accepted_step = step
        self.generation = generation_
        self.last_failure = last_failure


class PreparedEnforcementStep(StrictModule):
    """All-or-nothing candidate for advancing the accepted enforcement root."""

    status: RealizationStatus = eqx.field(static=True)
    base_generation: int = eqx.field(static=True)
    accepted_step: int = eqx.field(static=True)
    fields: frozendict[str, Any] | None
    realizations: frozendict[str, RealizationLifecycleState] | None
    results: frozendict[str, FieldRealizationResult]
    message: str = eqx.field(static=True)

    def __init__(
        self,
        status: RealizationStatus,
        /,
        *,
        base_generation: int,
        accepted_step: int,
        fields: Mapping[str, Any] | None,
        realizations: Mapping[str, RealizationLifecycleState] | None,
        results: Mapping[str, FieldRealizationResult] = frozendict(),
        message: str = "",
    ):
        successful = status.successful
        base_generation_ = int(base_generation)
        accepted_step_ = int(accepted_step)
        if base_generation_ < 0 or accepted_step_ < 0:
            raise ValueError("Prepared enforcement coordinates must be nonnegative.")
        results_ = frozendict(results)
        if any(
            not isinstance(value, FieldRealizationResult) for value in results_.values()
        ):
            raise TypeError("Prepared enforcement results must be field realizations.")
        if successful and (fields is None or realizations is None):
            raise ValueError(
                "A successful enforcement step requires complete candidates."
            )
        if successful and any(not result.successful for result in results_.values()):
            raise ValueError(
                "A successful enforcement step cannot contain failed results."
            )
        if not successful and (fields is not None or realizations is not None):
            raise ValueError(
                "A failed enforcement step cannot expose candidate iterates."
            )
        message_ = str(message)
        if not successful and not message_:
            raise ValueError("A failed enforcement step requires a message.")
        self.status = status
        self.base_generation = base_generation_
        self.accepted_step = accepted_step_
        self.fields = None if fields is None else frozendict(fields)
        self.realizations = None if realizations is None else frozendict(realizations)
        self.results = results_
        self.message = message_

    @property
    def successful(self) -> bool:
        return self.status.successful

    @classmethod
    def success(
        cls,
        fields: Mapping[str, Any],
        realizations: Mapping[str, RealizationLifecycleState],
        /,
        *,
        state: EnforcementState,
        accepted_step: int,
        results: Mapping[str, FieldRealizationResult] = frozendict(),
        unchanged: bool = False,
    ) -> PreparedEnforcementStep:
        status = RealizationStatus.UNCHANGED if unchanged else RealizationStatus.SUCCESS
        return cls(
            status,
            base_generation=state.generation,
            accepted_step=accepted_step,
            fields=fields,
            realizations=realizations,
            results=results,
        )

    @classmethod
    def failure(
        cls,
        status: RealizationStatus,
        /,
        *,
        state: EnforcementState,
        accepted_step: int,
        message: str,
        results: Mapping[str, FieldRealizationResult] = frozendict(),
    ) -> PreparedEnforcementStep:
        if status.successful:
            raise ValueError("PreparedEnforcementStep.failure requires a failure status.")
        return cls(
            status,
            base_generation=state.generation,
            accepted_step=accepted_step,
            fields=None,
            realizations=None,
            results=results,
            message=message,
        )


def commit_enforcement_step(
    state: EnforcementState,
    prepared: PreparedEnforcementStep,
    /,
) -> EnforcementState:
    """Commit a successful prepared step, otherwise retain the accepted root."""

    if (
        prepared.base_generation != state.generation
        or prepared.accepted_step < state.accepted_step
    ):
        failure = RealizationFailure(
            RealizationStatus.REFRESH_FAILED,
            "Prepared enforcement step is stale relative to committed state.",
            accepted_step=prepared.accepted_step,
            attempt=0,
        )
        return EnforcementState(
            state.fields,
            state.realizations,
            accepted_step=state.accepted_step,
            generation=state.generation,
            last_failure=failure,
        )
    if not prepared.successful:
        failure = RealizationFailure(
            prepared.status,
            prepared.message,
            accepted_step=prepared.accepted_step,
            attempt=0,
            evidence=prepared.results,
        )
        return EnforcementState(
            state.fields,
            state.realizations,
            accepted_step=state.accepted_step,
            generation=state.generation,
            last_failure=failure,
        )
    if prepared.fields is None or prepared.realizations is None:
        raise RuntimeError("A successful prepared enforcement step lost its candidates.")
    transaction_changed = (
        prepared.status is RealizationStatus.SUCCESS
        or prepared.accepted_step != state.accepted_step
    )
    return EnforcementState(
        prepared.fields,
        prepared.realizations,
        accepted_step=prepared.accepted_step,
        generation=state.generation + int(transaction_changed),
    )


__all__ = [
    "AbstractRealizationSource",
    "AdaptiveRealizationSource",
    "CallerRealizationSource",
    "EnforcementState",
    "FixedRealizationSource",
    "ParameterizedRealizationSource",
    "PerStepRealizationSource",
    "PreparedEnforcementStep",
    "RandomizedRealizationSource",
    "RealizationFailure",
    "RealizationLifecyclePhase",
    "RealizationLifecycleState",
    "RealizationSource",
    "RealizationSourceKind",
    "RealizationSourceStamp",
    "RefreshProposal",
    "RefreshProposalStatus",
    "RefreshValidation",
    "address_accepted_step_key",
    "commit_enforcement_step",
    "commit_refresh",
    "propose_refresh",
    "record_realization_stamp",
    "validate_refresh",
]
