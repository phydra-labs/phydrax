#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

from ..._strict import StrictModule
from ...linalg._gaussian_chain import associative_freeze
from ...nonlinear import (
    CausalLevenbergMarquardt,
    CausalNewton,
    CausalRecurrenceDiagnostics,
    CausalRecurrenceProblem,
    NonlinearStatus,
    NonlinearTermination,
    solve_causal_recurrence,
)
from .._keys import EvalKey
from ._recurrent import (
    _recurrent_output_from_state,
    _resolve_initial_state,
    _tree_where,
    _tree_zero_where_invalid,
    AbstractRecurrentCell,
    RecurrentBatch,
    RecurrentResult,
    run_recurrent,
)


CausalRecurrentFailurePolicy: TypeAlias = Literal["raise", "serial"]


class CausalRecurrentConfig(StrictModule):
    """Causal nonlinear method, termination, and explicit failure behavior."""

    method: CausalNewton | CausalLevenbergMarquardt
    termination: NonlinearTermination
    failure_policy: CausalRecurrentFailurePolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        method: CausalNewton | CausalLevenbergMarquardt | None = None,
        termination: NonlinearTermination | None = None,
        failure_policy: CausalRecurrentFailurePolicy = "raise",
    ):
        method_ = CausalLevenbergMarquardt() if method is None else method
        termination_ = NonlinearTermination() if termination is None else termination
        if not isinstance(method_, (CausalNewton, CausalLevenbergMarquardt)):
            raise TypeError("method must be a causal recurrence method or None.")
        if not isinstance(termination_, NonlinearTermination):
            raise TypeError("termination must be NonlinearTermination or None.")
        if failure_policy not in ("raise", "serial"):
            raise ValueError("failure_policy must be 'raise' or 'serial'.")
        self.method = method_
        self.termination = termination_
        self.failure_policy = failure_policy


class CausalRecurrentDiagnostics(StrictModule):
    """Per-case causal solve diagnostics and explicit fallback record."""

    causal: CausalRecurrenceDiagnostics
    status: Array
    fallback_used: Array


class CausalRecurrentResult(StrictModule):
    """Recurrent outputs with causal convergence and fallback provenance."""

    states: Any
    outputs: Any
    final_state: Any
    final_output: Any
    diagnostics: CausalRecurrentDiagnostics
    approximation_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.diagnostics.status == int(NonlinearStatus.SUCCESS)

    @property
    def recurrent(self) -> RecurrentResult:
        return RecurrentResult(
            states=self.states,
            outputs=self.outputs,
            final_state=self.final_state,
            final_output=self.final_output,
        )


def _case_count(batch: RecurrentBatch, /) -> int:
    return prod(batch.case_shape) if batch.case_shape else 1


def _flatten_state_cases(tree: Any, case_shape: tuple[int, ...], /) -> Any:
    count = prod(case_shape) if case_shape else 1
    rank = len(case_shape)
    return jax.tree.map(
        lambda leaf: jnp.asarray(leaf).reshape((count,) + jnp.asarray(leaf).shape[rank:]),
        tree,
    )


def _flatten_sequence_cases(
    tree: Any,
    case_shape: tuple[int, ...],
    sequence_length: int,
    /,
) -> Any:
    count = prod(case_shape) if case_shape else 1
    rank = len(case_shape)

    def flatten(leaf):
        value = jnp.asarray(leaf)
        moved = jnp.moveaxis(value, rank, 0)
        return jnp.moveaxis(
            moved.reshape((sequence_length, count) + value.shape[rank + 1 :]),
            0,
            1,
        )

    return jax.tree.map(flatten, tree)


def _restore_state_cases(tree: Any, case_shape: tuple[int, ...], /) -> Any:
    return jax.tree.map(
        lambda leaf: leaf.reshape(case_shape + leaf.shape[1:]),
        tree,
    )


def _restore_sequence_cases(tree: Any, case_shape: tuple[int, ...], /) -> Any:
    return jax.tree.map(
        lambda leaf: leaf.reshape(case_shape + leaf.shape[1:]),
        tree,
    )


def _latest_valid_output(initial: Any, values: Any, valid: Array, /) -> Any:
    return jax.tree.map(
        lambda initial_leaf, value_leaf: associative_freeze(
            initial_leaf,
            value_leaf,
            valid,
        )[-1],
        initial,
        values,
    )


def run_causal_recurrent(
    cell: AbstractRecurrentCell,
    batch: RecurrentBatch,
    /,
    *,
    initial_state: Any | None = None,
    reset_state: Any | None = None,
    initial_trajectory: Any | None = None,
    key: EvalKey = None,
    probe_key: Array | None = None,
    config: CausalRecurrentConfig | None = None,
) -> CausalRecurrentResult:
    """Evaluate a recurrent cell through a certified causal nonlinear solve."""

    if not isinstance(cell, AbstractRecurrentCell):
        raise TypeError("cell must be an AbstractRecurrentCell.")
    if not isinstance(batch, RecurrentBatch):
        raise TypeError("batch must be a RecurrentBatch.")
    configuration = CausalRecurrentConfig() if config is None else config
    if not isinstance(configuration, CausalRecurrentConfig):
        raise TypeError("config must be CausalRecurrentConfig or None.")

    canonical_state = _resolve_initial_state(cell, batch, None)
    state0 = (
        canonical_state
        if initial_state is None
        else _resolve_initial_state(cell, batch, initial_state)
    )
    restart_state = (
        canonical_state
        if reset_state is None
        else _resolve_initial_state(cell, batch, reset_state)
    )
    count = _case_count(batch)
    flat_initial = _flatten_state_cases(state0, batch.case_shape)
    flat_restart = _flatten_state_cases(restart_state, batch.case_shape)
    flat_inputs = _flatten_sequence_cases(
        batch.inputs,
        batch.case_shape,
        batch.sequence_length,
    )
    flat_valid = batch.valid.reshape((count, batch.sequence_length))
    flat_reset = batch.reset.reshape((count, batch.sequence_length))
    flat_initial_trajectory = (
        None
        if initial_trajectory is None
        else _flatten_sequence_cases(
            initial_trajectory,
            batch.case_shape,
            batch.sequence_length,
        )
    )
    step_keys = None if key is None else jr.split(key, batch.sequence_length)
    if (
        configuration.method.linearization.mode == "diagonal-hutchinson"
        and probe_key is None
    ):
        raise ValueError("Hutchinson causal recurrence requires probe_key.")
    probe_root = jr.key(0) if probe_key is None else probe_key
    case_probe_keys = jr.split(probe_root, count)

    def solve_case(
        current_cell,
        case_initial,
        case_restart,
        case_inputs,
        case_valid,
        case_reset,
        case_initial_trajectory,
        case_probe_key,
    ):
        if step_keys is None:
            drivers = (case_inputs, case_valid, case_reset)

            def transition(parameters, previous, driver):
                recurrent_cell, recurrent_restart = parameters
                inputs, valid, reset = driver
                restarted = _tree_where(reset & valid, recurrent_restart, previous)
                safe_inputs = _tree_zero_where_invalid(valid, inputs)
                proposed, _ = recurrent_cell.step(restarted, safe_inputs, key=None)
                return _tree_where(valid, proposed, previous)

        else:
            drivers = (case_inputs, case_valid, case_reset, step_keys)

            def transition(parameters, previous, driver):
                recurrent_cell, recurrent_restart = parameters
                inputs, valid, reset, step_key = driver
                restarted = _tree_where(reset & valid, recurrent_restart, previous)
                safe_inputs = _tree_zero_where_invalid(valid, inputs)
                proposed, _ = recurrent_cell.step(
                    restarted,
                    safe_inputs,
                    key=step_key,
                )
                return _tree_where(valid, proposed, previous)

        problem = CausalRecurrenceProblem(
            transition,
            case_initial,
            drivers,
            parameters=(current_cell, case_restart),
            problem_id="causal-recurrent-cell",
        )
        result = solve_causal_recurrence(
            problem,
            initial_trajectory=case_initial_trajectory,
            method=configuration.method,
            termination=configuration.termination,
            probe_key=case_probe_key,
        )
        return result.states, result.status, result.diagnostics

    if flat_initial_trajectory is None:
        flat_initial_trajectory = jax.tree.map(
            lambda leaf: jnp.broadcast_to(
                leaf[:, None, ...],
                (count, batch.sequence_length) + leaf.shape[1:],
            ),
            flat_initial,
        )
    if case_probe_keys is None:
        case_probe_keys = jnp.zeros((count, 2), dtype=jnp.uint32)

    flat_states, flat_status, flat_diagnostics = jax.vmap(
        solve_case,
        in_axes=(None, 0, 0, 0, 0, 0, 0, 0),
    )(
        cell,
        flat_initial,
        flat_restart,
        flat_inputs,
        flat_valid,
        flat_reset,
        flat_initial_trajectory,
        case_probe_keys,
    )

    def output_case(
        current_cell,
        case_initial,
        case_restart,
        case_inputs,
        case_valid,
        case_reset,
        case_states,
    ):
        predecessors = jax.tree.map(
            lambda initial_leaf, state_leaf: jnp.concatenate(
                (initial_leaf[None, ...], state_leaf[:-1]),
                axis=0,
            ),
            case_initial,
            case_states,
        )
        output_initial = _recurrent_output_from_state(current_cell, case_initial)

        if step_keys is None:

            def evaluate(previous, inputs, valid, reset):
                restarted = _tree_where(reset & valid, case_restart, previous)
                safe_inputs = _tree_zero_where_invalid(valid, inputs)
                _, output = current_cell.step(restarted, safe_inputs, key=None)
                return output

            raw_outputs = jax.vmap(evaluate)(
                predecessors,
                case_inputs,
                case_valid,
                case_reset,
            )
        else:

            def evaluate(previous, inputs, valid, reset, step_key):
                restarted = _tree_where(reset & valid, case_restart, previous)
                safe_inputs = _tree_zero_where_invalid(valid, inputs)
                _, output = current_cell.step(restarted, safe_inputs, key=step_key)
                return output

            raw_outputs = jax.vmap(evaluate)(
                predecessors,
                case_inputs,
                case_valid,
                case_reset,
                step_keys,
            )
        masked_outputs = _tree_zero_where_invalid(case_valid, raw_outputs)
        final_output = _latest_valid_output(output_initial, raw_outputs, case_valid)
        final_state = jax.tree.map(lambda leaf: leaf[-1], case_states)
        return masked_outputs, final_state, final_output

    flat_outputs, flat_final_state, flat_final_output = jax.vmap(
        output_case,
        in_axes=(None, 0, 0, 0, 0, 0, 0),
    )(
        cell,
        flat_initial,
        flat_restart,
        flat_inputs,
        flat_valid,
        flat_reset,
        flat_states,
    )

    successful = flat_status == int(NonlinearStatus.SUCCESS)
    fallback_used = ~successful
    states = _restore_sequence_cases(flat_states, batch.case_shape)
    outputs = _restore_sequence_cases(flat_outputs, batch.case_shape)
    final_state = _restore_state_cases(flat_final_state, batch.case_shape)
    final_output = _restore_state_cases(flat_final_output, batch.case_shape)

    if configuration.failure_policy == "raise":
        checked = eqx.error_if(
            jax.tree.leaves(states)[0],
            jnp.any(~successful),
            "Causal recurrent evaluation failed; inspect a record-mode core solve or use explicit serial fallback.",
        )
        states = eqx.tree_at(lambda tree: jax.tree.leaves(tree)[0], states, checked)
    else:
        serial = run_recurrent(
            cell,
            batch,
            initial_state=initial_state,
            reset_state=reset_state,
            key=key,
        )
        case_success = successful.reshape(batch.case_shape)
        states = _tree_where(case_success, states, serial.states)
        outputs = _tree_where(case_success, outputs, serial.outputs)
        final_state = _tree_where(case_success, final_state, serial.final_state)
        final_output = _tree_where(case_success, final_output, serial.final_output)

    diagnostics = CausalRecurrentDiagnostics(
        causal=jax.tree.map(
            lambda leaf: leaf.reshape(batch.case_shape + leaf.shape[1:]),
            flat_diagnostics,
        ),
        status=flat_status.reshape(batch.case_shape),
        fallback_used=fallback_used.reshape(batch.case_shape),
    )
    approximation = (
        "causal-recurrent"
        if configuration.failure_policy == "raise"
        else "causal-recurrent-with-explicit-serial-fallback"
    )
    return CausalRecurrentResult(
        states=states,
        outputs=outputs,
        final_state=final_state,
        final_output=final_output,
        diagnostics=diagnostics,
        approximation_id=approximation,
    )


__all__ = [
    "CausalRecurrentConfig",
    "CausalRecurrentDiagnostics",
    "CausalRecurrentFailurePolicy",
    "CausalRecurrentResult",
    "run_causal_recurrent",
]
