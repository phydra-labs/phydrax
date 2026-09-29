#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit host orchestration of mixed host/native partitioned coupling.

A host participant, such as an FMI co-simulation slave, executes outside JAX with
visible process I/O. Native preparation and the native window runtime refuse it.
This module is the one boundary that executes it: participants, ports,
measurements, exchanges, temporal conversions, conservative-transfer
certificates, ledger rows, window status, and the accepted/candidate state
transition are the canonical partitioned-coupling records. Only the host
lifecycle is added: capturing, restoring, and releasing the live host state
around each window.

An implicit window re-executes every host participant from its accepted
checkpoint and therefore requires a real save/restore of its complete state. A
host participant without it is admitted only on a declared explicit route; a
rejected explicit window then cannot be rolled back, is reported as
unrecoverable, and the orchestrator refuses to continue from it.

Host coupling forms no derivatives. A forward co-simulation has no adjoint, and a
staged `ExternalAdjointAction` is never chained through host windows implicitly.
"""

from __future__ import annotations

import abc
import math
from collections.abc import Callable, Mapping
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..._external_runtime import _host_only, ExternalDerivativeSupport
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...nonlinear import FixedPointIteration, NonlinearStatus, NonlinearTermination
from ...units import UnitDefinition
from .._partitioned_coupling_graph import (
    _prepare_coupling_plan,
    CouplingGraph,
    PreparedCoupling,
)
from .._partitioned_coupling_runtime import (
    _apply_exchange,
    _CouplingEvaluation,
    _global_gauss_seidel_evaluation,
    _global_jacobi_evaluation,
    _pack_interface,
    _ParticipantWork,
    _require_state_identity,
    _stagewise_evaluation,
    _unpack_interface,
    _window_result,
)
from .._partitioned_coupling_types import (
    AbstractCouplingSubsystem,
    CouplingDifferentiationPolicy,
    CouplingExchange,
    CouplingPort,
    CouplingState,
    CouplingSubsystemCapabilities,
    CouplingSubsystemResult,
    CouplingWindow,
    CouplingWindowResult,
    ExplicitCouplingPolicy,
    ImplicitCouplingPolicy,
)
from .._partitioned_coupling_waveform import CouplingWaveform, validate_coupling_signal
from ._lower_partitioned import PartitionedCouplingDeclaration
from ._method_participants import AbstractMethodCouplingParticipant


HostRollback: TypeAlias = Literal["restore", "none"]
HostWindowCommit: TypeAlias = Literal["accepted", "rejected", "unrecoverable"]


def _scalar(value: Any, role: str, /, *, dtype: Any | None = None) -> Array:
    array = jnp.asarray(value, dtype=dtype)
    if array.shape != ():
        raise ValueError(f"{role} must be a scalar array.")
    return array


class HostParticipantState(StrictModule):
    """Accepted lifecycle marker of one host participant inside a `CouplingState`.

    The physical state lives in the host process. The marker records the accepted
    communication point and committed window count, which the orchestrator checks
    against the live participant before every window.
    """

    time: Array
    accepted_windows: Array

    def __init__(self, time: Any, accepted_windows: Any, /) -> None:
        time_ = _scalar(time, "Host participant time")
        if not jnp.issubdtype(time_.dtype, jnp.floating):
            raise TypeError("Host participant time must be a real floating scalar.")
        self.time = time_
        self.accepted_windows = _scalar(
            accepted_windows, "Host participant accepted_windows", dtype=jnp.int32
        )


class AbstractHostCouplingParticipant(AbstractCouplingSubsystem):
    """Coupling participant executed eagerly on the host with explicit process I/O.

    Host participants are never JIT-capable: `prepare_coupling` and
    `lower_partitioned_coupling` refuse them, and only `prepare_host_coupling`
    admits them. `advance_window` advances the live process from the accepted
    communication point to the window end with concrete inputs, returns concrete
    port values, and refuses JAX transformations.

    `rollback == "restore"` declares a real save/restore of the complete host
    state through `capture`, `restore`, and `release`; only such participants may
    be re-executed by an implicit window or rolled back after a rejection.
    `derivative_support` states the provider's derivative route.
    """

    rollback: eqx.AbstractVar[HostRollback]

    @property
    def capabilities(self) -> CouplingSubsystemCapabilities:
        """Host execution: never traced, replayable only through real restore."""
        return CouplingSubsystemCapabilities(
            jit=False,
            differentiable=False,
            deterministic_replay=self.rollback == "restore",
            fixed_topology=True,
            supports_endpoint=True,
            supports_waveform=False,
        )

    @property
    def discretization_bundle_id(self) -> str | None:
        return None

    @property
    @abc.abstractmethod
    def derivative_support(self) -> ExternalDerivativeSupport:
        """The provider's derivative route; host coupling never selects one."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def time_unit(self) -> UnitDefinition:
        """Unit of the host process time coordinate, which is the coupling clock."""
        raise NotImplementedError

    @abc.abstractmethod
    def communication_time(self) -> float:
        """Current live communication point of the host process."""
        raise NotImplementedError

    @abc.abstractmethod
    def initial_outputs(self) -> tuple[Any, ...]:
        """Output-port values at the live communication point.

        Instantaneous ports observe the live state; whole-window ports carry
        zero amount.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def capture(self) -> Any:
        """Capture the complete live host state as an owned checkpoint."""
        raise NotImplementedError

    @abc.abstractmethod
    def restore(self, checkpoint: Any, /) -> None:
        """Return the live host state exactly to one captured checkpoint."""
        raise NotImplementedError

    @abc.abstractmethod
    def release(self, checkpoint: Any, /) -> None:
        """Free one captured checkpoint."""
        raise NotImplementedError

    @abc.abstractmethod
    def evidence_id(self) -> str:
        """Identity of the host's retained execution evidence."""
        raise NotImplementedError


class _CompiledParticipant(AbstractCouplingSubsystem):
    """One native participant executed as a compiled window map on the host route.

    Eager execution would retrace the participant's window map for every window
    and interface iterate; the compiled map is cached by the participant's
    structure instead. Ports, capabilities, and identity are the participant's.
    """

    participant: AbstractCouplingSubsystem

    def __init__(self, participant: AbstractCouplingSubsystem, /) -> None:
        if not participant.capabilities.jit:
            raise RuntimeError("Only JIT-capable participants execute compiled.")
        self.participant = participant

    @property
    def subsystem_id(self) -> str:
        return self.participant.subsystem_id

    @property
    def input_ports(self) -> tuple[CouplingPort, ...]:
        return self.participant.input_ports

    @property
    def output_ports(self) -> tuple[CouplingPort, ...]:
        return self.participant.output_ports

    @property
    def capabilities(self) -> CouplingSubsystemCapabilities:
        return self.participant.capabilities

    @property
    def discretization_bundle_id(self) -> str | None:
        return self.participant.discretization_bundle_id

    def advance_window(
        self,
        window: CouplingWindow,
        start_state: Any,
        inputs: tuple[Any, ...],
        args: Any,
        /,
    ) -> CouplingSubsystemResult:
        return _compiled_window(self.participant, window, start_state, inputs, args)


@eqx.filter_jit
def _compiled_window(
    participant: AbstractCouplingSubsystem,
    window: CouplingWindow,
    start_state: Any,
    inputs: tuple[Any, ...],
    args: Any,
    /,
) -> CouplingSubsystemResult:
    return participant.advance_window(window, start_state, inputs, args)


class PreparedHostCoupling(StrictModule, NonTrainableState):
    """Mixed host/native coupling plan with its initial accepted state.

    `plan` is the canonical `PreparedCoupling` of the declared graph; its report
    is not JIT-eligible, so the native runtime refuses it. `execution` is the same
    plan with every native participant executed as a compiled window map.
    `window_size` is the fixed coupling window of every host window.
    """

    plan: PreparedCoupling
    execution: PreparedCoupling
    initial_state: CouplingState
    window_size: float = eqx.field(static=True)
    host_indices: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @property
    def host_participants(self) -> tuple[AbstractHostCouplingParticipant, ...]:
        return tuple(
            _host_participant(self.plan.subsystems[index]) for index in self.host_indices
        )


class HostCouplingWindowResult(StrictModule):
    """One host window: canonical window evidence and its host commit decision.

    `commit` is `"accepted"` when the candidate became the accepted state,
    `"rejected"` when every host process is back at (or never left) the accepted
    checkpoint, and `"unrecoverable"` when a rejected window advanced a host
    participant that has no state restore; that coupling cannot continue.
    `host_evaluations` counts executions of the host participant set and
    `restores` every restore of their checkpoints.
    """

    window: CouplingWindowResult
    commit: HostWindowCommit = eqx.field(static=True)
    host_evaluations: int = eqx.field(static=True)
    restores: int = eqx.field(static=True)
    host_evidence_ids: tuple[str, ...] = eqx.field(static=True)

    @property
    def accepted_state(self) -> CouplingState:
        return self.window.accepted_state


class HostCouplingSolution(StrictModule):
    """Fixed-window host rollout stopped at its first rejected window."""

    windows: tuple[HostCouplingWindowResult, ...]
    final_state: CouplingState
    successful: bool = eqx.field(static=True)


def _host_participant(
    subsystem: AbstractCouplingSubsystem, /
) -> AbstractHostCouplingParticipant:
    if not isinstance(subsystem, AbstractHostCouplingParticipant):
        raise RuntimeError("Prepared host index does not name a host participant.")
    return subsystem


# Preparation ------------------------------------------------------------------


def _classify_participants(
    subsystems: tuple[AbstractCouplingSubsystem, ...], /
) -> tuple[tuple[AbstractHostCouplingParticipant, ...], tuple[str, ...]]:
    """Host participants and native participant IDs of one mixed graph."""
    host: list[AbstractHostCouplingParticipant] = []
    native: list[str] = []
    for subsystem in subsystems:
        if isinstance(subsystem, AbstractHostCouplingParticipant):
            host.append(subsystem)
        elif subsystem.capabilities.jit:
            native.append(subsystem.subsystem_id)
        else:
            raise ValueError(
                f"Participant {subsystem.subsystem_id!r} is neither JIT-capable nor "
                "a host participant with an explicit lifecycle."
            )
    if not host:
        raise ValueError(
            "Host coupling requires a host participant; an all-native graph runs on "
            "the native runtime through lower_partitioned_coupling."
        )
    if not native:
        raise ValueError(
            "Host coupling orchestrates a mixed host/native graph and requires at "
            "least one native participant."
        )
    return tuple(host), tuple(native)


def _refuse_host_derivatives(
    differentiation: CouplingDifferentiationPolicy,
    host: tuple[AbstractHostCouplingParticipant, ...],
    /,
) -> None:
    if differentiation.mode == "none":
        return
    routes = []
    for participant in host:
        support = participant.derivative_support
        match support.route:
            case "none":
                routes.append(
                    f"{participant.subsystem_id!r} has no adjoint (derivative-free "
                    f"alternatives: {', '.join(support.alternatives)})"
                )
            case "external-adjoint":
                routes.append(
                    f"{participant.subsystem_id!r} stages its own adjoint actions, "
                    "which host windows never chain implicitly"
                )
            case _:
                raise ValueError(f"Unknown external derivative route {support.route!r}.")
    raise ValueError(
        f"Coupling differentiation {differentiation.mode!r} cannot cross host "
        "participants: " + "; ".join(routes) + ". A forward co-simulation does not "
        "acquire an adjoint by wrapping it."
    )


def _admit_host_policy(
    declaration: PartitionedCouplingDeclaration,
    host: tuple[AbstractHostCouplingParticipant, ...],
    /,
) -> None:
    policy = declaration.policy
    if isinstance(policy, ExplicitCouplingPolicy):
        return
    if not isinstance(policy, ImplicitCouplingPolicy):
        raise TypeError("Unsupported coupling policy type.")
    method = policy.method
    if not isinstance(method, FixedPointIteration):
        raise ValueError(
            "Implicit host windows iterate the interface with a damped fixed-point "
            "sweep; general-root methods trace the participant map and cannot "
            "execute host participants."
        )
    if method.acceleration is not None:
        raise ValueError(
            "Implicit host windows iterate a damped fixed-point sweep; Anderson "
            "acceleration is a traced native method and is not applied on the host."
        )
    unrestorable = [
        participant.subsystem_id
        for participant in host
        if participant.rollback != "restore"
    ]
    if unrestorable:
        raise ValueError(
            "Implicit host windows re-execute every host participant from its "
            f"accepted checkpoint; {unrestorable} have no real state save/restore, "
            "so only a declared explicit non-retrying route is admitted."
        )


def _admit_host_clock(
    graph: CouplingGraph, host: tuple[AbstractHostCouplingParticipant, ...], /
) -> None:
    """Every host process runs on the graph's one declared coupling clock."""
    clock = graph.time_unit
    if clock is None:
        raise ValueError(
            "Host participants advance a physical clock; declare the coupling "
            "time_unit on the PartitionedCouplingDeclaration."
        )
    mismatched = [
        participant.subsystem_id
        for participant in host
        if participant.time_unit.unit_id != clock.unit_id
    ]
    if mismatched:
        raise ValueError(
            f"Host participants {mismatched} use a time unit other than the coupling "
            f"clock {clock.symbol!r}."
        )


def _host_markers(
    host: tuple[AbstractHostCouplingParticipant, ...], t0: float, /
) -> dict[str, HostParticipantState]:
    markers: dict[str, HostParticipantState] = {}
    for participant in host:
        live = participant.communication_time()
        if not math.isclose(live, t0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                f"Host participant {participant.subsystem_id!r} is at communication "
                f"point {live!r}, not the coupling start {t0!r}."
            )
        markers[participant.subsystem_id] = HostParticipantState(
            jnp.asarray(t0), jnp.asarray(0, dtype=jnp.int32)
        )
    return markers


def _zero_signal(port: CouplingPort, /) -> Any:
    zeros = jax.tree.map(
        lambda spec: jnp.zeros(spec.shape, spec.dtype), port.space.structure()
    )
    plan = port.waveform_plan
    if plan is None:
        return zeros
    return CouplingWaveform.constant(plan.initial_grid(), zeros, port.space)


def _observable(subsystem: AbstractCouplingSubsystem, /) -> bool:
    return isinstance(
        subsystem, (AbstractMethodCouplingParticipant, AbstractHostCouplingParticipant)
    )


def _explicit_values(
    subsystems: tuple[AbstractCouplingSubsystem, ...],
    exchanges: tuple[CouplingExchange, ...],
    exchange_values: Mapping[str, Any],
    /,
) -> dict[str, Any]:
    """Explicit initial values only where no source observes its checkpoint."""
    sources = {
        port.port_id: subsystem
        for subsystem in subsystems
        for port in subsystem.output_ports
    }
    unknown = sorted(set(exchange_values) - {e.exchange_id for e in exchanges})
    if unknown:
        raise ValueError("Unknown initial exchange values: " + ", ".join(unknown))
    missing: list[str] = []
    for exchange in exchanges:
        observable = _observable(sources[exchange.source_port_id])
        supplied = exchange.exchange_id in exchange_values
        if observable and supplied:
            raise ValueError(
                f"Exchange {exchange.exchange_id!r} is derived from its source "
                "checkpoint and cannot be overridden."
            )
        if not observable and not supplied:
            missing.append(exchange.exchange_id)
    if missing:
        raise ValueError(
            "Exchanges from participants without checkpoint observations need "
            "explicit initial values: " + ", ".join(sorted(missing))
        )
    return dict(exchange_values)


def _source_outputs(
    subsystem: AbstractCouplingSubsystem, state: Any, args: Any, /
) -> tuple[Any, ...]:
    if isinstance(subsystem, AbstractHostCouplingParticipant):
        outputs = tuple(subsystem.initial_outputs())
        if len(outputs) != len(subsystem.output_ports):
            raise ValueError(
                f"Host participant {subsystem.subsystem_id!r} observed the wrong "
                "number of output ports."
            )
        return tuple(
            validate_coupling_signal(port, value)
            for port, value in zip(subsystem.output_ports, outputs, strict=True)
        )
    if isinstance(subsystem, AbstractMethodCouplingParticipant):
        return subsystem.initial_outputs(state, args)
    raise RuntimeError("Only observable sources derive initial exchange values.")


def _derived_values(
    prepared: PreparedCoupling, window: CouplingWindow, args: Any, /
) -> dict[str, Any]:
    """Map every observable source checkpoint through its declared exchange."""
    values: dict[str, Any] = {}
    outputs: dict[int, tuple[Any, ...]] = {}
    for index, exchange in enumerate(prepared.exchanges):
        source_index = prepared.exchange_source_subsystems[index]
        source = prepared.subsystems[source_index]
        if not _observable(source):
            continue
        if source_index not in outputs:
            outputs[source_index] = _source_outputs(
                source, prepared.reference_state.participant_states[source_index], args
            )
        output = outputs[source_index][prepared.exchange_source_output_indices[index]]
        values[exchange.exchange_id] = _apply_exchange(prepared, index, output, window)
    return values


def prepare_host_coupling(
    declaration: PartitionedCouplingDeclaration,
    native_states: Mapping[str, Any],
    /,
    *,
    t0: float,
    window_size: float,
    args: Any = None,
    exchange_values: Mapping[str, Any] | None = None,
    problem_id: str = "host-coupling",
) -> PreparedHostCoupling:
    """Prepare one mixed host/native declaration for explicit host execution.

    The declaration is the one used by `lower_partitioned_coupling`; native
    lowering refuses its host participants. Preparation applies the canonical
    route, physical-exchange, temporal-conversion, conservative-certificate, and
    policy validation, then admits the host lifecycle: every host process runs on
    the declaration's coupling clock `time_unit`, an implicit policy requires
    a damped `FixedPointIteration` and real state restore of every host
    participant, and any derivative request is refused. `native_states` holds one
    initial checkpoint per native participant; host participants start from their
    live communication point, which must equal `t0`. Exchanges from native method
    and host participants start from their checkpoint observations mapped through
    the declared exchange; other sources need explicit `exchange_values`.
    """
    _host_only(native_states, args, exchange_values)
    if not isinstance(declaration, PartitionedCouplingDeclaration):
        raise TypeError("declaration must be PartitionedCouplingDeclaration.")
    graph = declaration.graph
    host, native = _classify_participants(graph.subsystems)
    _refuse_host_derivatives(declaration.differentiation, host)
    _admit_host_policy(declaration, host)
    _admit_host_clock(graph, host)
    size = float(window_size)
    if not math.isfinite(size) or size <= 0.0:
        raise ValueError("Host coupling window_size must be finite and positive.")
    if set(native_states) != set(native):
        raise ValueError(
            "Host coupling requires exactly one initial checkpoint per native "
            f"participant: {sorted(native)}."
        )
    explicit = _explicit_values(
        graph.subsystems,
        graph.exchanges,
        {} if exchange_values is None else exchange_values,
    )
    states = {**native_states, **_host_markers(host, float(t0))}
    ordered_states = tuple(
        states[subsystem.subsystem_id] for subsystem in graph.subsystems
    )
    targets = {
        port.port_id: port
        for subsystem in graph.subsystems
        for port in subsystem.input_ports
    }

    def plan(values: tuple[Any, ...], /) -> PreparedCoupling:
        return _prepare_coupling_plan(
            graph,
            ordered_states,
            values,
            policy=declaration.policy,
            differentiation=declaration.differentiation,
            time=t0,
            args=args,
            problem_id=problem_id,
            resources=declaration.resources,
            host_execution=True,
        )

    provisional = plan(
        tuple(
            explicit.get(
                exchange.exchange_id, _zero_signal(targets[exchange.target_port_id])
            )
            for exchange in graph.exchanges
        )
    )
    derived = _derived_values(provisional, CouplingWindow(0, t0, t0 + size), args)
    prepared = plan(
        tuple(
            explicit[exchange.exchange_id]
            if exchange.exchange_id in explicit
            else derived[exchange.exchange_id]
            for exchange in graph.exchanges
        )
    )
    host_indices = tuple(
        index
        for index, subsystem in enumerate(prepared.subsystems)
        if isinstance(subsystem, AbstractHostCouplingParticipant)
    )
    return PreparedHostCoupling(
        plan=prepared,
        execution=eqx.tree_at(
            lambda value: value.subsystems,
            prepared,
            tuple(
                subsystem
                if isinstance(subsystem, AbstractHostCouplingParticipant)
                else _CompiledParticipant(subsystem)
                for subsystem in prepared.subsystems
            ),
        ),
        initial_state=prepared.reference_state,
        window_size=size,
        host_indices=host_indices,
        plan_id=canonical_fingerprint(
            {
                "kind": "prepared-host-coupling",
                "plan": prepared.plan_id,
                "window_size": size.hex(),
            }
        ),
    )


# Window execution ---------------------------------------------------------------


def _require_live(prepared: PreparedHostCoupling, state: CouplingState, /) -> None:
    """Refuse a window whose host processes are not at the accepted state."""
    accepted = float(state.time)
    for index in prepared.host_indices:
        participant = _host_participant(prepared.plan.subsystems[index])
        marker = state.participant_states[index]
        if not isinstance(marker, HostParticipantState):
            raise TypeError(
                f"Host participant {participant.subsystem_id!r} state must be "
                "HostParticipantState."
            )
        live = participant.communication_time()
        if not (
            math.isclose(live, accepted, rel_tol=1e-12, abs_tol=1e-12)
            and math.isclose(float(marker.time), accepted, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise ValueError(
                f"Host participant {participant.subsystem_id!r} is at communication "
                f"point {live!r} but the accepted coupling state is at {accepted!r}; "
                "a host process without state restore cannot return to an accepted "
                "checkpoint after a rejected window."
            )


# Restorable host participants paired with their captured checkpoints.
type _Captured = tuple[tuple[AbstractHostCouplingParticipant, Any], ...]


def _capture(prepared: PreparedHostCoupling, /) -> _Captured:
    """Checkpoints of every restorable host participant, captured individually."""
    captured: list[tuple[AbstractHostCouplingParticipant, Any]] = []
    try:
        for participant in prepared.host_participants:
            if participant.rollback == "restore":
                captured.append((participant, participant.capture()))
    except BaseException as error:
        for failure in _rollback(tuple(captured), restore=False):
            error.add_note(_failure_note(failure))
        raise
    return tuple(captured)


def _failure_note(failure: Exception, /) -> str:
    # `_rollback` names the participant and action of every failure in its notes.
    context = " ".join(failure.__notes__)
    return f"{context} {type(failure).__name__}: {failure}"


def _rollback(captured: _Captured, /, *, restore: bool) -> list[Exception]:
    """Restore (when asked) and release every checkpoint independently.

    One failed host call, such as a dead session, must neither strand the other
    participants ahead of the accepted state nor leak their checkpoints, so each
    call is attempted and its failure returned to the caller, which reports it.
    """
    failures: list[Exception] = []
    for participant, checkpoint in captured:
        if restore:
            try:
                participant.restore(checkpoint)
            except Exception as failure:
                failure.add_note(
                    f"Restoring host participant {participant.subsystem_id!r} failed."
                )
                failures.append(failure)
        try:
            participant.release(checkpoint)
        except Exception as failure:
            failure.add_note(
                f"Releasing the checkpoint of {participant.subsystem_id!r} failed."
            )
            failures.append(failure)
    return failures


def _raise_failures(failures: list[Exception], /) -> None:
    if failures:
        raise ExceptionGroup(
            "Host checkpoint restore or release failed; affected host processes may "
            "not be at the accepted state.",
            failures,
        )


def _restore(captured: _Captured, /) -> None:
    """Restore every captured participant before an implicit re-execution."""
    failures: list[Exception] = []
    for participant, checkpoint in captured:
        try:
            participant.restore(checkpoint)
        except Exception as failure:
            failure.add_note(
                f"Restoring host participant {participant.subsystem_id!r} failed."
            )
            failures.append(failure)
    _raise_failures(failures)


def _norm(method: FixedPointIteration, value: Array, /) -> float:
    precision = method.precision
    return float(precision.decision(jnp.linalg.norm(precision.accumulation(value))))


def _fixed_point_status(
    evaluation: _CouplingEvaluation,
    residual_norm: float,
    step_norm: float,
    state_norm: float,
    initial_norm: float,
    termination: NonlinearTermination,
    /,
) -> NonlinearStatus:
    """Native damped fixed-point termination classes for one host iterate."""
    if not bool(evaluation.successful):
        return NonlinearStatus.UNRECOVERABLE_DOMAIN_FAILURE
    if not (bool(evaluation.finite) and math.isfinite(residual_norm)):
        return NonlinearStatus.NONFINITE_EVALUATION
    if residual_norm <= float(termination.residual_threshold(initial_norm)):
        return NonlinearStatus.SUCCESS
    if step_norm <= float(termination.step_threshold(state_norm)):
        return NonlinearStatus.RESIDUAL_STAGNATION
    if residual_norm > termination.divergence_factor * max(initial_norm, 1e-30):
        return NonlinearStatus.DIVERGENCE
    return NonlinearStatus.ITERATING


class _HostIterate(StrictModule):
    evaluation: _CouplingEvaluation
    status: NonlinearStatus = eqx.field(static=True)
    iterations: int = eqx.field(static=True)
    evaluations: int = eqx.field(static=True)
    work: _ParticipantWork


def _host_fixed_point(
    plan: PreparedCoupling,
    policy: ImplicitCouplingPolicy,
    state: CouplingState,
    window: CouplingWindow,
    args: Any,
    restore: Callable[[], None],
    /,
) -> _HostIterate:
    """Damped fixed-point interface sweep with host restore before every iterate.

    The candidate is the last executed evaluation, whose used inputs and mapped
    values certify the window exactly as the native final re-evaluation does; it
    is not executed a second time. Host synchronization of the iterate norms is
    intentional: the loop drives host processes.
    """
    method = policy.method
    sweep = policy.fixed_point_sweep
    if not isinstance(method, FixedPointIteration) or sweep is None:
        raise RuntimeError("Prepared host fixed-point policy is incomplete.")
    order = None if sweep.kind == "jacobi" else sweep.subsystem_order
    termination = policy.termination

    def evaluate(coordinates: Array) -> tuple[_CouplingEvaluation, Array]:
        values = _unpack_interface(plan, coordinates, state.exchange_values)
        evaluation = _stagewise_evaluation(
            plan, window, state, values, args, gauss_seidel_order=order
        )
        return evaluation, _pack_interface(plan, evaluation.exchange_values) - coordinates

    coordinates = _pack_interface(plan, state.exchange_values)
    evaluation, residual = evaluate(coordinates)
    initial_norm = _norm(method, residual)
    status = _fixed_point_status(
        evaluation, initial_norm, math.inf, 0.0, initial_norm, termination
    )
    iterations, evaluations = 0, 1
    work = jnp.zeros_like(evaluation.participant_work)
    counted = jnp.zeros_like(evaluation.participant_iterations)
    maximum_evaluations = termination.maximum_evaluations
    while (
        status == NonlinearStatus.ITERATING
        and iterations < termination.maximum_steps
        and (maximum_evaluations is None or evaluations < maximum_evaluations)
    ):
        work = work + evaluation.participant_work
        counted = counted + evaluation.participant_iterations
        proposed = coordinates + method.damping * residual
        restore()
        evaluation, next_residual = evaluate(proposed)
        iterations, evaluations = iterations + 1, evaluations + 1
        status = _fixed_point_status(
            evaluation,
            _norm(method, next_residual),
            _norm(method, proposed - coordinates),
            _norm(method, coordinates),
            initial_norm,
            termination,
        )
        coordinates, residual = proposed, next_residual
    if status == NonlinearStatus.ITERATING:
        status = (
            NonlinearStatus.MAXIMUM_STEPS_REACHED
            if iterations >= termination.maximum_steps
            else NonlinearStatus.MAXIMUM_EVALUATIONS_REACHED
        )
    return _HostIterate(
        evaluation, status, iterations, evaluations, _ParticipantWork(work, counted)
    )


def _explicit_window(
    plan: PreparedCoupling,
    policy: ExplicitCouplingPolicy,
    state: CouplingState,
    window: CouplingWindow,
    args: Any,
    /,
) -> CouplingWindowResult:
    match policy.sweep.kind:
        case "jacobi":
            evaluation = _global_jacobi_evaluation(
                plan, window, state, state.exchange_values, args
            )
        case "gauss-seidel":
            evaluation = _global_gauss_seidel_evaluation(
                plan,
                window,
                state,
                state.exchange_values,
                policy.sweep.subsystem_order,
                args,
            )
        case _:
            raise ValueError(f"Unknown coupling sweep {policy.sweep.kind!r}.")
    return _window_result(
        plan,
        state,
        window,
        evaluation,
        nonlinear_status=jnp.asarray(-1, dtype=jnp.int32),
        coupling_iterations=jnp.asarray(1, dtype=jnp.int32),
        nonlinear_residual_evaluations=jnp.asarray(0, dtype=jnp.int32),
        implicit=False,
    )


def _implicit_window(
    plan: PreparedCoupling,
    policy: ImplicitCouplingPolicy,
    state: CouplingState,
    window: CouplingWindow,
    args: Any,
    restore: Callable[[], None],
    /,
) -> tuple[CouplingWindowResult, int]:
    iterate = _host_fixed_point(plan, policy, state, window, args, restore)
    result = _window_result(
        plan,
        state,
        window,
        iterate.evaluation,
        nonlinear_status=jnp.asarray(int(iterate.status), dtype=jnp.int32),
        coupling_iterations=jnp.asarray(iterate.iterations, dtype=jnp.int32),
        nonlinear_residual_evaluations=jnp.asarray(
            iterate.evaluations - 1, dtype=jnp.int32
        ),
        implicit=True,
        iterate_work=iterate.work,
    )
    return result, iterate.evaluations


def _commit(
    prepared: PreparedHostCoupling,
    state: CouplingState,
    result: CouplingWindowResult,
    captured: _Captured,
    /,
) -> tuple[HostWindowCommit, int]:
    """Keep an accepted host state, or return every host process to its checkpoint.

    Restorable participants are restored individually. A participant without
    restore keeps a rejected window recoverable only if it never left the
    accepted communication point, which by the host participant contract
    identifies the accepted state.
    """
    if bool(result.successful):
        _raise_failures(_rollback(captured, restore=False))
        return "accepted", 0
    _raise_failures(_rollback(captured, restore=True))
    accepted = float(state.time)
    unmoved = all(
        math.isclose(
            participant.communication_time(), accepted, rel_tol=1e-12, abs_tol=1e-12
        )
        for participant in prepared.host_participants
        if participant.rollback != "restore"
    )
    return ("rejected" if unmoved else "unrecoverable"), int(bool(captured))


def _execute_window(
    prepared: PreparedHostCoupling,
    state: CouplingState,
    window: CouplingWindow,
    args: Any,
    captured: _Captured,
    /,
) -> tuple[CouplingWindowResult, int, int]:
    """Window result, host evaluations, and iterate restores of one policy."""
    plan = prepared.execution
    policy = plan.policy
    if isinstance(policy, ExplicitCouplingPolicy):
        return _explicit_window(plan, policy, state, window, args), 1, 0
    if not isinstance(policy, ImplicitCouplingPolicy):
        raise TypeError("Unsupported prepared coupling policy.")
    if len(captured) != len(prepared.host_indices):
        raise RuntimeError("Implicit host windows were admitted without restore.")
    restores = 0

    def restore() -> None:
        nonlocal restores
        _restore(captured)
        restores += 1

    result, evaluations = _implicit_window(plan, policy, state, window, args, restore)
    return result, evaluations, restores


def _scheduled_end(prepared: PreparedHostCoupling, state: CouplingState, /) -> float:
    """End `t0 + (k + 1)·h` of the state's next window on the fixed host grid.

    Accumulating `t += h` drifts off the grid (0.1 + 0.1 + 0.1 > 0.3) and can
    step a host process past its declared stop time.
    """
    initial = prepared.initial_state
    t0 = float(initial.time)
    size = prepared.window_size
    completed = int(state.window_index) - int(initial.window_index)
    if completed < 0 or not math.isclose(
        float(state.time), t0 + completed * size, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError(
            "Host coupling state is not on the fixed window grid of its preparation."
        )
    return t0 + (completed + 1) * size


def _advance_window(
    prepared: PreparedHostCoupling,
    state: CouplingState,
    args: Any,
    end: float,
    /,
) -> HostCouplingWindowResult:
    window = CouplingWindow(
        state.window_index, state.time, jnp.asarray(end, dtype=state.time.dtype)
    )
    _require_live(prepared, state)
    captured = _capture(prepared)
    try:
        result, evaluations, restores = _execute_window(
            prepared, state, window, args, captured
        )
    except BaseException as error:
        # A failed host call leaves the processes at an unknown point: return every
        # restorable one to the accepted checkpoint independently and report any
        # rollback failure on the original error, which propagates unmasked.
        for failure in _rollback(captured, restore=True):
            error.add_note(_failure_note(failure))
        raise
    commit, rollback_restores = _commit(prepared, state, result, captured)
    return HostCouplingWindowResult(
        window=result,
        commit=commit,
        host_evaluations=evaluations,
        restores=restores + rollback_restores,
        host_evidence_ids=tuple(
            participant.evidence_id() for participant in prepared.host_participants
        ),
    )


def advance_host_coupling_window(
    prepared: PreparedHostCoupling,
    state: CouplingState,
    args: Any = None,
    /,
) -> HostCouplingWindowResult:
    """Advance one fixed host window and commit or roll back every host process.

    Window `k` spans `[t0 + k·h, t0 + (k + 1)·h]` on the prepared grid.
    Participants execute eagerly in the declared sweep; native participants run
    their pure window maps and host participants perform explicit process I/O.
    The window status, certification, ledger rows, and accepted state follow the
    native window contract. Before the window every host process must sit at the
    accepted communication point. Restorable host participants are captured
    first: implicit iterates restore them before every re-execution, and a
    rejected or failed window restores each of them. A rejected window that
    advanced a participant without restore is `"unrecoverable"`, and the next
    window refuses to start.
    """
    _host_only(state, args)
    if not isinstance(prepared, PreparedHostCoupling):
        raise TypeError("prepared must be PreparedHostCoupling.")
    _require_state_identity(prepared.plan, state)
    return _advance_window(prepared, state, args, _scheduled_end(prepared, state))


def solve_host_coupling(
    prepared: PreparedHostCoupling,
    /,
    *,
    t1: float,
    args: Any = None,
    state: CouplingState | None = None,
) -> HostCouplingSolution:
    """Advance fixed host windows from `state` (default: the initial state) to `t1`.

    Windows follow the prepared grid and the last one ends exactly at `t1`. The
    rollout stops at the first window that is not accepted: a rejected window
    replays identically from the same checkpoint, so it is reported, never retried.
    """
    _host_only(args, state)
    if not isinstance(prepared, PreparedHostCoupling):
        raise TypeError("prepared must be PreparedHostCoupling.")
    current = prepared.initial_state if state is None else state
    _require_state_identity(prepared.plan, current)
    end = float(t1)
    raw_count = (end - float(current.time)) / prepared.window_size
    count = round(raw_count)
    if count <= 0 or not math.isclose(raw_count, count, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("t1 - t0 must be a positive whole number of host windows.")
    windows: list[HostCouplingWindowResult] = []
    for index in range(count):
        scheduled = _scheduled_end(prepared, current)
        result = _advance_window(
            prepared, current, args, end if index == count - 1 else scheduled
        )
        windows.append(result)
        current = result.accepted_state
        if result.commit != "accepted":
            break
    return HostCouplingSolution(
        windows=tuple(windows),
        final_state=current,
        successful=all(result.commit == "accepted" for result in windows)
        and len(windows) == count,
    )


__all__ = [
    "AbstractHostCouplingParticipant",
    "HostCouplingSolution",
    "HostCouplingWindowResult",
    "HostParticipantState",
    "HostRollback",
    "HostWindowCommit",
    "PreparedHostCoupling",
    "advance_host_coupling_window",
    "prepare_host_coupling",
    "solve_host_coupling",
]
