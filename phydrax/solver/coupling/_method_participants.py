#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native method owners bound as transactional partitioned-coupling participants.

A participant carries the owner's complete continuation, explicit model state, and
PRNG key data in its accepted checkpoint. Every window evaluation, implicit
interface iterate, and adaptive retry starts from that checkpoint, so a rejected
candidate never leaks advanced history, model state, or randomness, and replay
reproduces it bit for bit.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any, assert_never, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...typing import parse, PRNGKey
from .._conservation_temporal import ConservationIMEXMethod
from .._differential_algebraic import (
    DAEContinuation,
    DifferentialAlgebraicSolution,
    PreparedDAESolve,
    solve_dae,
)
from .._fixed_step import AbstractFixedStepMethod, FixedStepStatus
from .._partitioned_coupling_types import (
    AbstractCouplingSubsystem,
    CouplingPort,
    CouplingSubsystemCapabilities,
    CouplingSubsystemResult,
    CouplingWindow,
    CouplingWindowErrorEstimate,
)
from .._partitioned_coupling_waveform import (
    coupling_signal_structure,
    CouplingWaveform,
    CouplingWaveformPlan,
    validate_coupling_signal,
)


MethodParticipantRandomness: TypeAlias = Literal["none", "carried-key", "external"]
_Native: TypeAlias = AbstractFixedStepMethod | ConservationIMEXMethod


class MethodParticipantState(StrictModule):
    """Accepted checkpoint of one bound native method participant.

    `native` is the owner's complete continuation, `model_state` its explicit
    component state, and `key_data` the data of the carried typed PRNG key (absent
    for deterministic owners). `accepted_windows` counts committed windows and
    `native_steps` is the owner's native step index at which `native` resumes: a
    fixed-step owner advances it by its substeps per window, while DAE and steady
    owners, whose schedules live in their own continuation, keep it unchanged. A
    candidate advances every field exactly once per window evaluation.
    """

    native: Any
    model_state: Any
    key_data: Array | None
    accepted_windows: Array
    native_steps: Array


class MethodWindowBinding(StrictModule):
    """Native owner arguments and advanced model state for one evaluation."""

    method_args: Any
    model_state: Any


class SteadyResponse(StrictModule):
    """Instantaneous response of a steady owner with its native evidence."""

    outputs: tuple[Any, ...]
    model_state: Any
    successful: Array
    status: Array
    residual_norm: Array
    iterations: Array
    work: Array


def _identifier(value: str, role: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{role} must be a non-empty string.")
    return value


def _ports(
    input_ports: tuple[CouplingPort, ...], output_ports: tuple[CouplingPort, ...], /
) -> tuple[tuple[CouplingPort, ...], tuple[CouplingPort, ...]]:
    inputs = tuple(input_ports)
    outputs = tuple(output_ports)
    if any(not isinstance(port, CouplingPort) for port in (*inputs, *outputs)):
        raise TypeError("Participant ports must contain CouplingPort values.")
    if any(port.direction != "input" for port in inputs) or any(
        port.direction != "output" for port in outputs
    ):
        raise ValueError("Participant ports have inconsistent directions.")
    identifiers = tuple(port.port_id for port in (*inputs, *outputs))
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Participant port IDs must be unique.")
    return inputs, outputs


def _capabilities(
    ports: tuple[CouplingPort, ...],
    randomness: MethodParticipantRandomness,
    differentiable: bool,
    /,
) -> CouplingSubsystemCapabilities:
    """Truthful capabilities: external randomness cannot replay a checkpoint."""
    return CouplingSubsystemCapabilities(
        jit=True,
        differentiable=bool(differentiable),
        deterministic_replay=randomness != "external",
        fixed_topology=True,
        supports_endpoint=any(port.waveform_plan is None for port in ports),
        supports_waveform=any(port.waveform_plan is not None for port in ports),
        counts_complete=True,
    )


def _carried_key_data(
    randomness: MethodParticipantRandomness, key: Any, /
) -> Array | None:
    match randomness:
        case "carried-key":
            return jax.random.key_data(parse(key, PRNGKey, "key"))
        case "none" | "external":
            if key is not None:
                raise ValueError(
                    "Only a carried-key participant stores a PRNG key in its checkpoint."
                )
            return None
        case _:
            assert_never(randomness)


def _split_window_key(
    state: MethodParticipantState, impl: str | None, /
) -> tuple[Array | None, Array | None]:
    """Window key and next checkpoint key data, derived from the accepted key."""
    if state.key_data is None or impl is None:
        return None, None
    key = jax.random.wrap_key_data(state.key_data, impl=impl)
    window_key, next_key = jax.random.split(key)
    return window_key, jax.random.key_data(next_key)


def _checked_state(state: Any, /) -> MethodParticipantState:
    if not isinstance(state, MethodParticipantState):
        raise TypeError("Method participant state must be MethodParticipantState.")
    return state


def _checked_binding(binding: Any, /) -> MethodWindowBinding:
    if not isinstance(binding, MethodWindowBinding):
        raise TypeError("Participant bind must return MethodWindowBinding.")
    return binding


def _uniform_plan(plan: CouplingWaveformPlan, nodes: np.ndarray, role: str, /) -> None:
    initial = np.asarray(plan.initial_nodes, dtype=np.float64)
    if (
        plan.adaptation is not None
        or plan.sample_capacity != nodes.size
        or not np.array_equal(initial, nodes)
    ):
        raise ValueError(
            f"{role} waveform ports must sample exactly the participant's fixed "
            "native time nodes; use an explicit interpolating exchange to change grids."
        )


def _instantaneous_outputs(
    ports: tuple[CouplingPort, ...], /
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Endpoint, waveform, and whole-window output port positions."""
    endpoint = tuple(
        index
        for index, port in enumerate(ports)
        if port.waveform_plan is None and port.temporal_kind == "instantaneous"
    )
    waveform = tuple(
        index for index, port in enumerate(ports) if port.waveform_plan is not None
    )
    amounts = tuple(
        index
        for index, port in enumerate(ports)
        if port.temporal_kind == "interval_integral"
    )
    return endpoint, waveform, amounts


def _assemble_outputs(
    ports: tuple[CouplingPort, ...],
    endpoint_values: tuple[Any, ...],
    waveform_values: tuple[Any, ...],
    amount_values: tuple[Any, ...],
    /,
) -> tuple[Any, ...]:
    """Place observed values back into the declared output-port order."""
    endpoint, waveform, amounts = _instantaneous_outputs(ports)
    values: dict[int, Any] = {}
    for index, value in zip(endpoint, endpoint_values, strict=True):
        values[index] = ports[index].space.validate(value)
    for index, value in zip(waveform, waveform_values, strict=True):
        values[index] = value
    for index, value in zip(amounts, amount_values, strict=True):
        values[index] = ports[index].space.validate(value)
    return tuple(values[index] for index in range(len(ports)))


def _stacked_waveforms(
    ports: tuple[CouplingPort, ...],
    positions: tuple[int, ...],
    samples: tuple[tuple[Any, ...], ...],
    /,
) -> tuple[CouplingWaveform, ...]:
    """Waveform outputs from observations at every fixed native time node."""
    waveforms: list[CouplingWaveform] = []
    for local, index in enumerate(positions):
        port = ports[index]
        plan = port.waveform_plan
        if not isinstance(plan, CouplingWaveformPlan):
            raise RuntimeError("Waveform output port lost its plan after preparation.")
        values = jax.tree.map(
            lambda *leaves: jnp.stack(leaves),
            *(port.space.validate(node[local]) for node in samples),
        )
        waveforms.append(CouplingWaveform(plan.initial_grid(), values, port.space))
    return tuple(waveforms)


class AbstractMethodCouplingParticipant(AbstractCouplingSubsystem):
    """Native owner binding that publishes outputs consistent with a checkpoint."""

    randomness: eqx.AbstractVar[MethodParticipantRandomness]

    @abc.abstractmethod
    def initial_outputs(
        self, state: MethodParticipantState, args: Any, /
    ) -> tuple[Any, ...]:
        """Output-port values consistent with an accepted checkpoint.

        Instantaneous ports observe the checkpoint, waveform ports hold that
        observation across their grid, and whole-window ports carry zero amount.
        """
        raise NotImplementedError


def _held_outputs(
    ports: tuple[CouplingPort, ...], observed: tuple[Any, ...], /
) -> tuple[Any, ...]:
    """Initial output values: observations held, whole-window amounts zero."""
    endpoint, waveform, amounts = _instantaneous_outputs(ports)
    instantaneous = tuple(sorted((*endpoint, *waveform)))
    by_index = dict(zip(instantaneous, observed, strict=True))
    values: list[Any] = []
    for index, port in enumerate(ports):
        if index in amounts:
            values.append(
                jax.tree.map(
                    lambda spec: jnp.zeros(spec.shape, spec.dtype),
                    coupling_signal_structure(port),
                )
            )
        elif port.waveform_plan is None:
            values.append(port.space.validate(by_index[index]))
        else:
            values.append(
                CouplingWaveform.constant(
                    port.waveform_plan.initial_grid(), by_index[index], port.space
                )
            )
    return tuple(values)


def _substep_input(
    port: CouplingPort, value: Any, substep: Array, substeps: int, /
) -> Any:
    """Declared per-substep view of one window input.

    Instantaneous endpoints stay frozen over the window, waveform inputs give their
    `(start, end)` samples on the participant's native nodes, and whole-window
    amounts are spent at a uniform rate, so the substep shares sum exactly.
    """
    signal = validate_coupling_signal(port, value)
    if port.waveform_plan is not None:
        return (
            signal.sample(substep, port.space),
            signal.sample(substep + 1, port.space),
        )
    if port.temporal_kind == "interval_integral":
        return jax.tree.map(lambda leaf: leaf / substeps, signal)
    return signal


class _NativeStep(StrictModule):
    state: Any
    successful: Array
    residual: Array
    iterations: Array
    work: Array


def _native_step(
    method: _Native,
    index: Array,
    time: Array,
    state: Any,
    step_size: Array,
    method_args: Any,
    /,
) -> _NativeStep:
    """One native owner step with its own evidence, never a reimplementation."""
    match method:
        case AbstractFixedStepMethod():
            result = method.step(index, time, state, step_size, method_args)
            return _NativeStep(
                result.accepted_state,
                jnp.asarray(result.successful, dtype=jnp.bool_),
                jnp.asarray(result.residual, dtype=time.dtype),
                jnp.asarray(result.iterations, dtype=jnp.int32),
                jnp.asarray(result.work, dtype=jnp.int32),
            )
        case ConservationIMEXMethod():
            imex = method.step(time, state, step_size, method_args)
            return _NativeStep(
                imex.accepted_state,
                jnp.asarray(imex.successful, dtype=jnp.bool_),
                jnp.asarray(imex.maximum_implicit_residual, dtype=time.dtype),
                jnp.asarray(imex.implicit_iterations, dtype=jnp.int32),
                jnp.asarray(method.tableau.stage_count, dtype=jnp.int32),
            )
        case _:
            raise TypeError("Unsupported native fixed-step owner.")


class _SubstepCarry(StrictModule):
    native: Any
    model_state: Any
    successful: Array
    residual: Array
    iterations: Array
    work: Array


class FixedStepCouplingParticipant(AbstractMethodCouplingParticipant, NonTrainableState):
    """Bind a native fixed-step or conservative IMEX owner as a coupling participant.

    The owner advances `substeps` uniform native steps per window. `bind` maps the
    substep window, declared per-substep input views, model state, and substep key
    to native method arguments. `observe(native, args)` returns every
    instantaneous output in declared port order (waveform ports at every node) and
    `amounts(start_native, end_native, args)` every whole-window output, which the
    owner must itself have spent, for example through a native flux accumulator.
    Waveform ports sample exactly the native nodes `linspace(0, 1, substeps + 1)`.
    """

    method: _Native
    bind: Callable[..., MethodWindowBinding]
    observe: Callable[[Any, Any], tuple[Any, ...]]
    amounts: Callable[[Any, Any, Any], tuple[Any, ...]] | None
    estimate_error: Callable[[Any, Any, Any], CouplingWindowErrorEstimate] | None
    input_ports: tuple[CouplingPort, ...]
    output_ports: tuple[CouplingPort, ...]
    capabilities: CouplingSubsystemCapabilities
    substeps: int = eqx.field(static=True)
    randomness: MethodParticipantRandomness = eqx.field(static=True)
    key_impl: str | None = eqx.field(static=True)
    subsystem_id: str = eqx.field(static=True)
    discretization_bundle_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        method: _Native,
        bind: Callable[..., MethodWindowBinding],
        observe: Callable[[Any, Any], tuple[Any, ...]],
        /,
        *,
        subsystem_id: str,
        substeps: int,
        input_ports: tuple[CouplingPort, ...] = (),
        output_ports: tuple[CouplingPort, ...] = (),
        amounts: Callable[[Any, Any, Any], tuple[Any, ...]] | None = None,
        estimate_error: Callable[[Any, Any, Any], CouplingWindowErrorEstimate]
        | None = None,
        randomness: MethodParticipantRandomness = "none",
        key_impl: str | None = None,
        differentiable: bool = True,
        discretization_bundle_id: str | None = None,
    ) -> None:
        if not isinstance(method, (AbstractFixedStepMethod, ConservationIMEXMethod)):
            raise TypeError(
                "method must be an AbstractFixedStepMethod or ConservationIMEXMethod."
            )
        if not callable(bind) or not callable(observe):
            raise TypeError("bind and observe must be callable.")
        if estimate_error is not None and not callable(estimate_error):
            raise TypeError("estimate_error must be callable or None.")
        count = int(substeps)
        if count < 1:
            raise ValueError("A fixed-step participant takes at least one substep.")
        randomness = parse(randomness, MethodParticipantRandomness, "randomness")
        inputs, outputs = _ports(input_ports, output_ports)
        nodes = np.linspace(0.0, 1.0, count + 1)
        for port in (*inputs, *outputs):
            if port.waveform_plan is not None:
                _uniform_plan(port.waveform_plan, nodes, "Fixed-step participant")
        has_amounts = any(port.temporal_kind == "interval_integral" for port in outputs)
        if has_amounts != (amounts is not None):
            raise ValueError(
                "amounts is required exactly when whole-window output ports exist."
            )
        if amounts is not None and not callable(amounts):
            raise TypeError("amounts must be callable or None.")
        if (randomness == "carried-key") != (key_impl is not None):
            raise ValueError("key_impl names the carried key implementation only.")
        self.method = method
        self.bind = bind
        self.observe = observe
        self.amounts = amounts
        self.estimate_error = estimate_error
        self.input_ports = inputs
        self.output_ports = outputs
        self.capabilities = _capabilities((*inputs, *outputs), randomness, differentiable)
        self.substeps = count
        self.randomness = randomness
        self.key_impl = None if key_impl is None else _identifier(key_impl, "key_impl")
        self.subsystem_id = _identifier(subsystem_id, "subsystem_id")
        self.discretization_bundle_id = (
            None
            if discretization_bundle_id is None
            else _identifier(discretization_bundle_id, "discretization_bundle_id")
        )

    def initial_state(
        self,
        native: Any,
        /,
        *,
        model_state: Any = None,
        key: Any = None,
        step_index: ArrayLike = 0,
    ) -> MethodParticipantState:
        """Checkpoint holding the owner state, model state, and carried key.

        `step_index` is the native step index at which `native` resumes; the
        owner's step `k` of every later window receives `step_index + k` plus the
        substeps of all previously accepted windows.
        """
        key_data = _carried_key_data(self.randomness, key)
        if key_data is not None and str(jax.random.key_impl(key)) != self.key_impl:
            raise ValueError("Carried key implementation differs from key_impl.")
        index = jnp.asarray(step_index)
        if index.shape != () or not jnp.issubdtype(index.dtype, jnp.integer):
            raise ValueError("step_index must be an integer scalar.")
        index = eqx.error_if(
            index.astype(jnp.int32), index < 0, "step_index must be non-negative."
        )
        return MethodParticipantState(
            native, model_state, key_data, jnp.asarray(0, dtype=jnp.int32), index
        )

    def initial_outputs(
        self, state: MethodParticipantState, args: Any, /
    ) -> tuple[Any, ...]:
        checkpoint = _checked_state(state)
        return _held_outputs(
            self.output_ports, tuple(self.observe(checkpoint.native, args))
        )

    def _substep(
        self,
        window: CouplingWindow,
        inputs: tuple[Any, ...],
        window_key: Array | None,
        args: Any,
        carry: _SubstepCarry,
        substep: Array,
        step_index: Array,
    ) -> tuple[_NativeStep, Any]:
        size = window.size / self.substeps
        start = window.start + size * substep
        sub_window = CouplingWindow(substep, start, start + size)
        views = tuple(
            _substep_input(port, value, substep, self.substeps)
            for port, value in zip(self.input_ports, inputs, strict=True)
        )
        key = None if window_key is None else jax.random.fold_in(window_key, substep)
        binding = _checked_binding(
            self.bind(sub_window, views, carry.model_state, key, args)
        )
        step = _native_step(
            self.method, step_index, start, carry.native, size, binding.method_args
        )
        return step, binding.model_state

    def advance_window(
        self,
        window: CouplingWindow,
        start_state: Any,
        inputs: tuple[Any, ...],
        args: Any,
        /,
    ) -> CouplingSubsystemResult:
        state = _checked_state(start_state)
        window_key, next_key_data = _split_window_key(state, self.key_impl)
        dtype = window.start.dtype

        def body(carry: _SubstepCarry, substep: Array) -> tuple[_SubstepCarry, Any]:
            def execute(_: None) -> _SubstepCarry:
                step, model_state = self._substep(
                    window,
                    inputs,
                    window_key,
                    args,
                    carry,
                    substep,
                    state.native_steps + substep,
                )
                return _SubstepCarry(
                    step.state,
                    model_state,
                    carry.successful & step.successful,
                    jnp.maximum(carry.residual, step.residual),
                    carry.iterations + step.iterations,
                    carry.work + step.work,
                )

            # After a failed native step no further owner work is performed.
            advanced = jax.lax.cond(
                carry.successful, execute, lambda _: carry, operand=None
            )
            return advanced, tuple(self.observe(advanced.native, args))

        initial = _SubstepCarry(
            state.native,
            state.model_state,
            jnp.asarray(True),
            jnp.asarray(0.0, dtype=dtype),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )
        final, trajectory = jax.lax.scan(
            body, initial, jnp.arange(self.substeps, dtype=jnp.int32)
        )
        outputs = self._outputs(state.native, final.native, trajectory, args)
        candidate = MethodParticipantState(
            final.native,
            final.model_state,
            next_key_data,
            state.accepted_windows + 1,
            state.native_steps + self.substeps,
        )
        estimate = (
            None
            if self.estimate_error is None
            else self.estimate_error(state.native, final.native, args)
        )
        return CouplingSubsystemResult(
            candidate,
            outputs,
            successful=final.successful,
            status=jnp.where(
                final.successful,
                int(FixedStepStatus.SUCCESS),
                int(FixedStepStatus.STEP_FAILURE),
            ),
            residual_norm=final.residual,
            iterations=final.iterations,
            error_estimate=estimate,
            work=final.work,
        )

    def _outputs(
        self, start: Any, end: Any, trajectory: tuple[Any, ...], args: Any, /
    ) -> tuple[Any, ...]:
        endpoint, waveform, _ = _instantaneous_outputs(self.output_ports)
        initial = tuple(self.observe(start, args))
        final = tuple(self.observe(end, args))
        instantaneous = tuple(sorted((*endpoint, *waveform)))
        local = {index: position for position, index in enumerate(instantaneous)}
        samples = (
            tuple(initial[local[index]] for index in waveform),
            *(
                tuple(
                    jax.tree.map(lambda leaf, k=node: leaf[k], trajectory[local[index]])
                    for index in waveform
                )
                for node in range(self.substeps)
            ),
        )
        amounts = () if self.amounts is None else tuple(self.amounts(start, end, args))
        return _assemble_outputs(
            self.output_ports,
            tuple(final[local[index]] for index in endpoint),
            _stacked_waveforms(
                self.output_ports,
                waveform,
                tuple(tuple(node) for node in samples),
            ),
            amounts,
        )


class DAEParticipantNative(StrictModule):
    """Array part of the owner's `DAEContinuation` and whether it has started."""

    continuation: Any
    started: Array


def _neutral_solution(prepared: PreparedDAESolve, /) -> DifferentialAlgebraicSolution:
    shape = eqx.filter_eval_shape(lambda: solve_dae(prepared, args=prepared.problem.args))
    return jax.tree.map(
        lambda value: (
            jnp.zeros(value.shape, value.dtype)
            if isinstance(value, jax.ShapeDtypeStruct)
            else value
        ),
        shape,
    )


class _DAEWindow(StrictModule):
    """Continuation array part, samples, and native evidence of one window solve."""

    continuation: Any
    accepted_order: Array
    states: Array
    state_rates: Array
    successful: Array
    status: Array
    residual_norm: Array
    iterations: Array
    attempts: Array
    accepted_error: Array


def _dae_window(solution: DifferentialAlgebraicSolution, /) -> _DAEWindow:
    attempts = solution.attempt_history
    steps = solution.step_history
    return _DAEWindow(
        eqx.filter(solution.continuation, eqx.is_array),
        solution.continuation.accepted_order,
        solution.states,
        solution.state_rates,
        solution.successful,
        solution.termination_status,
        jnp.max(solution.residual_norm),
        jnp.sum(jnp.where(attempts.valid, attempts.nonlinear_iterations, 0)),
        # Every accepted and rejected native attempt is owner work.
        attempts.count,
        jnp.max(jnp.where(steps.valid, steps.error_ratios, 0.0)),
    )


class DAECouplingParticipant(AbstractMethodCouplingParticipant, NonTrainableState):
    """Bind a prepared adaptive DAE solve and its exact `DAEContinuation`.

    The prepared time grid is a window template: its normalized save times are
    rebound to each window, and the accepted-history continuation, including the
    retained nonlinear solve, is carried in the participant checkpoint. A fixed-grid
    or event-driven DAE cannot resume its history across windows and is refused.
    Inputs enter through `bind` as native DAE arguments; waveform inputs are
    refused because the owner's residual takes time-dependent inputs only through
    its own input policy.
    """

    prepared: PreparedDAESolve
    continuation_static: Any
    bind: Callable[..., MethodWindowBinding]
    observe: Callable[[Array, Array, Any], tuple[Any, ...]]
    amounts: Callable[[Array, Array, Any], tuple[Any, ...]] | None
    input_ports: tuple[CouplingPort, ...]
    output_ports: tuple[CouplingPort, ...]
    capabilities: CouplingSubsystemCapabilities
    normalized_times: tuple[float, ...] = eqx.field(static=True)
    randomness: MethodParticipantRandomness = eqx.field(static=True)
    key_impl: str | None = eqx.field(static=True)
    subsystem_id: str = eqx.field(static=True)
    discretization_bundle_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedDAESolve,
        bind: Callable[..., MethodWindowBinding],
        observe: Callable[[Array, Array, Any], tuple[Any, ...]],
        /,
        *,
        subsystem_id: str,
        input_ports: tuple[CouplingPort, ...] = (),
        output_ports: tuple[CouplingPort, ...] = (),
        amounts: Callable[[Array, Array, Any], tuple[Any, ...]] | None = None,
        randomness: MethodParticipantRandomness = "none",
        key_impl: str | None = None,
        differentiable: bool = False,
    ) -> None:
        if not isinstance(prepared, PreparedDAESolve):
            raise TypeError("prepared must be PreparedDAESolve.")
        if prepared.plan.policy.adaptive is None or prepared.events is not None:
            raise ValueError(
                "Only an event-free adaptive DAE solve resumes its exact history "
                "across coupling windows; rollback would otherwise be incomplete."
            )
        if not callable(bind) or not callable(observe):
            raise TypeError("bind and observe must be callable.")
        randomness = parse(randomness, MethodParticipantRandomness, "randomness")
        inputs, outputs = _ports(input_ports, output_ports)
        if any(port.waveform_plan is not None for port in inputs):
            raise ValueError(
                "DAE participant inputs are endpoint or whole-window values."
            )
        times = np.asarray(prepared.time_grid.times, dtype=np.float64)
        normalized = (times - times[0]) / (times[-1] - times[0])
        for port in outputs:
            if port.waveform_plan is not None:
                _uniform_plan(port.waveform_plan, normalized, "DAE participant")
        has_amounts = any(port.temporal_kind == "interval_integral" for port in outputs)
        if has_amounts != (amounts is not None):
            raise ValueError(
                "amounts is required exactly when whole-window output ports exist."
            )
        if (randomness == "carried-key") != (key_impl is not None):
            raise ValueError("key_impl names the carried key implementation only.")
        _, static = eqx.partition(_neutral_solution(prepared).continuation, eqx.is_array)
        self.prepared = prepared
        self.continuation_static = static
        self.bind = bind
        self.observe = observe
        self.amounts = amounts
        self.input_ports = inputs
        self.output_ports = outputs
        self.capabilities = _capabilities((*inputs, *outputs), randomness, differentiable)
        self.normalized_times = tuple(float(value) for value in normalized)
        self.randomness = randomness
        self.key_impl = None if key_impl is None else _identifier(key_impl, "key_impl")
        self.subsystem_id = _identifier(subsystem_id, "subsystem_id")
        self.discretization_bundle_id = prepared.problem.discretization_bundle_id

    def initial_state(
        self, /, *, model_state: Any = None, key: Any = None
    ) -> MethodParticipantState:
        """Unstarted checkpoint; the first window runs the owner's initialization."""
        key_data = _carried_key_data(self.randomness, key)
        if key_data is not None and str(jax.random.key_impl(key)) != self.key_impl:
            raise ValueError("Carried key implementation differs from key_impl.")
        arrays = eqx.filter(_neutral_solution(self.prepared).continuation, eqx.is_array)
        native = DAEParticipantNative(arrays, jnp.asarray(False))
        zero = jnp.asarray(0, dtype=jnp.int32)
        return MethodParticipantState(native, model_state, key_data, zero, zero)

    def initial_outputs(
        self, state: MethodParticipantState, args: Any, /
    ) -> tuple[Any, ...]:
        checkpoint = _checked_state(state)
        native = checkpoint.native
        continuation: DAEContinuation = eqx.combine(
            native.continuation, self.continuation_static
        )
        value = jnp.where(
            native.started, continuation.state, self.prepared.problem.initial_state
        )
        rate = jnp.where(
            native.started,
            continuation.state_rate,
            self.prepared.problem.initial_state_rate,
        )
        return _held_outputs(self.output_ports, tuple(self.observe(value, rate, args)))

    def _solve(
        self, window: CouplingWindow, native: DAEParticipantNative, dae_args: Any, /
    ) -> _DAEWindow:
        times = jnp.asarray(self.normalized_times, dtype=window.start.dtype)
        prepared = eqx.tree_at(
            lambda value: value.time_grid.times,
            self.prepared,
            window.start + window.size * times,
        )
        continuation = eqx.combine(native.continuation, self.continuation_static)
        # The branches differ in initialization evidence; both publish the same
        # window record so the continuation and native work reach the checkpoint.
        return jax.lax.cond(
            native.started,
            lambda _: _dae_window(
                solve_dae(prepared, args=dae_args, continuation=continuation)
            ),
            lambda _: _dae_window(solve_dae(prepared, args=dae_args)),
            operand=None,
        )

    def advance_window(
        self,
        window: CouplingWindow,
        start_state: Any,
        inputs: tuple[Any, ...],
        args: Any,
        /,
    ) -> CouplingSubsystemResult:
        state = _checked_state(start_state)
        window_key, next_key_data = _split_window_key(state, self.key_impl)
        views = tuple(
            _substep_input(port, value, jnp.asarray(0, dtype=jnp.int32), 1)
            for port, value in zip(self.input_ports, inputs, strict=True)
        )
        binding = _checked_binding(
            self.bind(window, views, state.model_state, window_key, args)
        )
        solved = self._solve(window, state.native, binding.method_args)
        candidate = MethodParticipantState(
            DAEParticipantNative(solved.continuation, jnp.asarray(True)),
            binding.model_state,
            next_key_data,
            state.accepted_windows + 1,
            state.native_steps,
        )
        dtype = window.start.dtype
        return CouplingSubsystemResult(
            candidate,
            self._outputs(solved, args),
            successful=solved.successful,
            status=solved.status,
            residual_norm=solved.residual_norm.astype(dtype),
            iterations=solved.iterations,
            error_estimate=CouplingWindowErrorEstimate(
                solved.accepted_error.astype(dtype),
                jnp.asarray(1.0, dtype=dtype),
                jnp.maximum(solved.accepted_order, 1),
                solved.successful,
            ),
            work=solved.attempts,
        )

    def _outputs(self, solved: _DAEWindow, args: Any, /) -> tuple[Any, ...]:
        endpoint, waveform, _ = _instantaneous_outputs(self.output_ports)
        instantaneous = tuple(sorted((*endpoint, *waveform)))
        local = {index: position for position, index in enumerate(instantaneous)}
        nodes = tuple(
            tuple(self.observe(solved.states[node], solved.state_rates[node], args))
            for node in range(len(self.normalized_times))
        )
        final = nodes[-1]
        amounts = (
            ()
            if self.amounts is None
            else tuple(self.amounts(solved.states[0], solved.states[-1], args))
        )
        return _assemble_outputs(
            self.output_ports,
            tuple(final[local[index]] for index in endpoint),
            _stacked_waveforms(
                self.output_ports,
                waveform,
                tuple(tuple(node[local[index]] for index in waveform) for node in nodes),
            ),
            amounts,
        )


class SteadyResponseCouplingParticipant(
    AbstractMethodCouplingParticipant, NonTrainableState
):
    """Bind a steady field-response owner as an instantaneous participant.

    `respond(inputs, model_state, key, args)` returns a `SteadyResponse`. The owner
    never receives the window: a steady response does not integrate in time and
    therefore publishes no whole-window amount. Endpoint ports respond once at the
    window end; waveform ports sharing one plan respond at every sample node. Its
    temporal truncation error is zero, not an estimate of transient physics. The
    checkpoint's `native` is the accepted end-of-window response, starting from
    `initial_response`, so restarts and re-preparation observe the accepted state.
    """

    respond: Callable[..., SteadyResponse]
    initial_response: tuple[Any, ...]
    input_ports: tuple[CouplingPort, ...]
    output_ports: tuple[CouplingPort, ...]
    capabilities: CouplingSubsystemCapabilities
    randomness: MethodParticipantRandomness = eqx.field(static=True)
    key_impl: str | None = eqx.field(static=True)
    subsystem_id: str = eqx.field(static=True)
    discretization_bundle_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        respond: Callable[..., SteadyResponse],
        /,
        *,
        subsystem_id: str,
        input_ports: tuple[CouplingPort, ...],
        output_ports: tuple[CouplingPort, ...],
        initial_response: tuple[Any, ...],
        randomness: MethodParticipantRandomness = "none",
        key_impl: str | None = None,
        differentiable: bool = True,
        discretization_bundle_id: str | None = None,
    ) -> None:
        if not callable(respond):
            raise TypeError("respond must be callable.")
        randomness = parse(randomness, MethodParticipantRandomness, "randomness")
        inputs, outputs = _ports(input_ports, output_ports)
        ports = (*inputs, *outputs)
        if any(port.temporal_kind == "interval_integral" for port in ports):
            raise ValueError(
                "A steady response has no temporal accumulation; whole-window "
                "amounts belong to a transient owner."
            )
        # Plans carry arrays; their canonical capacity identity is the plan ID.
        plans = {
            None if port.waveform_plan is None else port.waveform_plan.plan_id
            for port in ports
        }
        if len(plans) != 1:
            raise ValueError(
                "Steady response ports are all endpoints or all share one waveform plan."
            )
        if (randomness == "carried-key") != (key_impl is not None):
            raise ValueError("key_impl names the carried key implementation only.")
        if len(tuple(initial_response)) != len(outputs):
            raise ValueError("initial_response needs one value per output port.")
        self.respond = respond
        self.initial_response = tuple(
            port.space.validate(value)
            for port, value in zip(outputs, initial_response, strict=True)
        )
        self.input_ports = inputs
        self.output_ports = outputs
        self.capabilities = _capabilities(ports, randomness, differentiable)
        self.randomness = randomness
        self.key_impl = None if key_impl is None else _identifier(key_impl, "key_impl")
        self.subsystem_id = _identifier(subsystem_id, "subsystem_id")
        self.discretization_bundle_id = (
            None
            if discretization_bundle_id is None
            else _identifier(discretization_bundle_id, "discretization_bundle_id")
        )

    def initial_state(
        self, /, *, model_state: Any = None, key: Any = None
    ) -> MethodParticipantState:
        """Checkpoint of the initial response, model state, key, and window count."""
        key_data = _carried_key_data(self.randomness, key)
        if key_data is not None and str(jax.random.key_impl(key)) != self.key_impl:
            raise ValueError("Carried key implementation differs from key_impl.")
        zero = jnp.asarray(0, dtype=jnp.int32)
        return MethodParticipantState(
            self.initial_response, model_state, key_data, zero, zero
        )

    def initial_outputs(
        self, state: MethodParticipantState, args: Any, /
    ) -> tuple[Any, ...]:
        checkpoint = _checked_state(state)
        response = checkpoint.native
        if not isinstance(response, tuple) or len(response) != len(self.output_ports):
            raise ValueError(
                "Steady checkpoint native must hold one accepted response per output."
            )
        return _held_outputs(self.output_ports, response)

    def _response(
        self, inputs: tuple[Any, ...], model_state: Any, key: Array | None, args: Any
    ) -> SteadyResponse:
        response = self.respond(inputs, model_state, key, args)
        if not isinstance(response, SteadyResponse):
            raise TypeError("respond must return SteadyResponse.")
        return response

    def advance_window(
        self,
        window: CouplingWindow,
        start_state: Any,
        inputs: tuple[Any, ...],
        args: Any,
        /,
    ) -> CouplingSubsystemResult:
        state = _checked_state(start_state)
        window_key, next_key_data = _split_window_key(state, self.key_impl)
        ports = (*self.input_ports, *self.output_ports)
        plan = ports[0].waveform_plan
        if plan is None:
            response = self._response(
                tuple(
                    validate_coupling_signal(port, value)
                    for port, value in zip(self.input_ports, inputs, strict=True)
                ),
                state.model_state,
                window_key,
                args,
            )
            outputs = tuple(
                port.space.validate(value)
                for port, value in zip(self.output_ports, response.outputs, strict=True)
            )
            accepted = outputs
        else:
            response, waveforms = self._sampled_response(
                plan, inputs, state.model_state, window_key, args
            )
            outputs = waveforms
            accepted = tuple(
                waveform.sample(waveform.grid.sample_count - 1, port.space)
                for waveform, port in zip(waveforms, self.output_ports, strict=True)
            )
        dtype = window.start.dtype
        return CouplingSubsystemResult(
            MethodParticipantState(
                accepted,
                response.model_state,
                next_key_data,
                state.accepted_windows + 1,
                state.native_steps,
            ),
            outputs,
            successful=response.successful,
            status=response.status,
            residual_norm=jnp.asarray(response.residual_norm, dtype=dtype),
            iterations=response.iterations,
            error_estimate=CouplingWindowErrorEstimate(
                jnp.asarray(0.0, dtype=dtype), jnp.asarray(1.0, dtype=dtype), 1, True
            ),
            work=response.work,
        )

    def _sampled_response(
        self,
        plan: CouplingWaveformPlan,
        inputs: tuple[Any, ...],
        model_state: Any,
        window_key: Array | None,
        args: Any,
        /,
    ) -> tuple[SteadyResponse, tuple[CouplingWaveform, ...]]:
        """Quasi-static response at every waveform node, threaded through model state."""
        waveforms = tuple(
            validate_coupling_signal(port, value)
            for port, value in zip(self.input_ports, inputs, strict=True)
        )
        grid = plan.initial_grid()
        nodes: list[tuple[Any, ...]] = []
        successful = jnp.asarray(True)
        residual = jnp.asarray(0.0)
        iterations = jnp.asarray(0, dtype=jnp.int32)
        work = jnp.asarray(0, dtype=jnp.int32)
        status = jnp.asarray(0, dtype=jnp.int32)
        state = model_state
        for node in range(plan.sample_capacity):
            key = None if window_key is None else jax.random.fold_in(window_key, node)
            response = self._response(
                tuple(
                    waveform.sample(node, port.space)
                    for waveform, port in zip(waveforms, self.input_ports, strict=True)
                ),
                state,
                key,
                args,
            )
            active = node < grid.sample_count
            state = jax.tree.map(
                lambda new, old, a=active: jnp.where(a, new, old),
                response.model_state,
                state,
            )
            nodes.append(tuple(response.outputs))
            successful = successful & (~active | response.successful)
            status = jnp.where(active & ~response.successful, response.status, status)
            residual = jnp.maximum(
                residual, jnp.where(active, response.residual_norm, 0.0)
            )
            iterations = iterations + jnp.where(active, response.iterations, 0)
            work = work + jnp.where(active, response.work, 0)
        outputs = tuple(
            CouplingWaveform(
                grid,
                jax.tree.map(
                    lambda *leaves: jnp.stack(leaves),
                    *(port.space.validate(node[index]) for node in nodes),
                ),
                port.space,
            )
            for index, port in enumerate(self.output_ports)
        )
        return (
            SteadyResponse(
                outputs, state, successful, status, residual, iterations, work
            ),
            outputs,
        )


__all__ = [
    "AbstractMethodCouplingParticipant",
    "DAECouplingParticipant",
    "DAEParticipantNative",
    "FixedStepCouplingParticipant",
    "MethodParticipantRandomness",
    "MethodParticipantState",
    "MethodWindowBinding",
    "SteadyResponse",
    "SteadyResponseCouplingParticipant",
]
