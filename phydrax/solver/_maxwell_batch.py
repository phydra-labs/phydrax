#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._maxwell import (
    _fixed_step,
    _PreparedMaxwellFixedStep,
    CompatibleMaxwellDiagnostics,
    CompatibleMaxwellState,
    PreparedCompatibleMaxwell,
    solve_compatible_maxwell,
)


class PreparedCompatibleMaxwellCaseBatch(StrictModule, NonTrainableState):
    """Independent cases sharing one static Maxwell executable signature."""

    runtimes: tuple[PreparedCompatibleMaxwell, ...]
    initial_states: tuple[CompatibleMaxwellState, ...]
    fixed_steps: tuple[_PreparedMaxwellFixedStep, ...]
    case_shape: tuple[int, ...] = eqx.field(static=True)
    signature_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class CompatibleMaxwellCaseBatchResult(StrictModule):
    final_states: CompatibleMaxwellState
    observations: tuple[Array, ...]
    diagnostics: CompatibleMaxwellDiagnostics
    status: Array
    step_count: Array
    case_shape: tuple[int, ...] = eqx.field(static=True)


def _stack_cases(values: Sequence[Any], case_shape: tuple[int, ...], /) -> Any:
    return jax.tree_util.tree_map(
        lambda *leaves: jnp.stack(leaves, axis=0).reshape(
            (*case_shape, *leaves[0].shape)
        ),
        *values,
    )


def prepare_compatible_maxwell_case_batch(
    runtimes: Sequence[PreparedCompatibleMaxwell],
    initial_states: Sequence[CompatibleMaxwellState],
    case_shape: Sequence[int],
    step_size: ArrayLike,
    /,
) -> PreparedCompatibleMaxwellCaseBatch:
    """Bind independent cases while retaining one shared topology/signature."""

    runtimes_ = tuple(runtimes)
    states = tuple(initial_states)
    shape = tuple(int(value) for value in case_shape)
    if not shape or any(value <= 0 for value in shape):
        raise ValueError("Maxwell case_shape must contain positive dimensions.")
    count = int(np.prod(shape))
    if len(runtimes_) != count or len(states) != count:
        raise ValueError("Maxwell case count does not match case_shape.")
    if any(not isinstance(value, PreparedCompatibleMaxwell) for value in runtimes_):
        raise TypeError("Case batches require prepared Maxwell runtimes.")
    checked_states = tuple(
        runtime._state(state) for runtime, state in zip(runtimes_, states, strict=True)
    )
    fixed = tuple(
        _fixed_step(runtime, step_size, state.primary.electric_displacement.dtype)
        for runtime, state in zip(runtimes_, checked_states, strict=True)
    )
    signature = fixed[0].signature.signature_id
    if any(value.signature.signature_id != signature for value in fixed[1:]):
        raise ValueError("Maxwell cases do not share one executable signature.")
    total_bytes = sum(value.resource_estimate.total_bytes for value in runtimes_)
    if total_bytes > runtimes_[0].plan.resources.maximum_total_bytes:
        raise ValueError("Maxwell case batch exceeds the declared total resource budget.")
    return PreparedCompatibleMaxwellCaseBatch(
        runtimes_,
        checked_states,
        fixed,
        shape,
        signature,
        canonical_fingerprint(
            {
                "kind": "prepared-compatible-maxwell-case-batch",
                "signature": signature,
                "case_shape": shape,
                "runtimes": [value.prepared_id for value in runtimes_],
                "step_size": float(np.asarray(step_size)),
            }
        ),
    )


def solve_compatible_maxwell_case_batch(
    batch: PreparedCompatibleMaxwellCaseBatch,
    start_time: ArrayLike,
    steps: int,
    case_args: Sequence[Any] | None = None,
    /,
) -> CompatibleMaxwellCaseBatchResult:
    """Execute independent cases without cross-case reductions."""

    if not isinstance(batch, PreparedCompatibleMaxwellCaseBatch):
        raise TypeError("batch must be PreparedCompatibleMaxwellCaseBatch.")
    args = (None,) * len(batch.runtimes) if case_args is None else tuple(case_args)
    if len(args) != len(batch.runtimes):
        raise ValueError("Maxwell case argument count does not match the batch.")
    results = tuple(
        solve_compatible_maxwell(
            runtime,
            state,
            start_time,
            fixed.parameters.step_size,
            steps,
            argument,
        )
        for runtime, state, fixed, argument in zip(
            batch.runtimes,
            batch.initial_states,
            batch.fixed_steps,
            args,
            strict=True,
        )
    )
    states = _stack_cases(tuple(value.final_state for value in results), batch.case_shape)
    diagnostics = _stack_cases(
        tuple(value.diagnostics for value in results), batch.case_shape
    )
    observer_count = len(results[0].observations)
    observations = tuple(
        _stack_cases(
            tuple(value.observations[index] for value in results),
            batch.case_shape,
        )
        for index in range(observer_count)
    )
    status = jnp.stack(tuple(value.status for value in results)).reshape(batch.case_shape)
    step_count = jnp.stack(tuple(value.step_count for value in results)).reshape(
        batch.case_shape
    )
    return CompatibleMaxwellCaseBatchResult(
        states,
        observations,
        diagnostics,
        status,
        step_count,
        batch.case_shape,
    )


__all__ = [
    "CompatibleMaxwellCaseBatchResult",
    "PreparedCompatibleMaxwellCaseBatch",
    "prepare_compatible_maxwell_case_batch",
    "solve_compatible_maxwell_case_batch",
]
