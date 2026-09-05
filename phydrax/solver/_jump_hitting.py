#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Event-ledger first passage and finite, closed CTMC reference analysis."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .. import linalg as la
from .._strict import StrictModule
from ..stochastic import JUMP_MAX_EVENTS, JUMP_SUCCESS
from ._jump import FiniteStateGenerator, JumpSolution


class JumpFirstHit(StrictModule):
    """Exact observed hits; failures are not right-censored observations.

    ``time`` is infinity for an unobserved hit. ``observation_end`` is the
    requested horizon for successful paths and the last verified event time
    for failed paths. A hit preceding capacity exhaustion remains exact.
    """

    time: Array
    hit: Array
    initially_in_target: Array
    censored: Array
    incomplete: Array
    capacity_failure: Array
    observation_end: Array
    status: Array


def event_first_hit(
    solution: JumpSolution,
    initial_state: ArrayLike,
    target: Callable[[Array], ArrayLike],
    /,
    *,
    t0: ArrayLike,
    t1: ArrayLike,
) -> JumpFirstHit:
    """Scan initial/post-event states of a native pure-jump solution.

    ``solution`` must span ``t0`` through ``t1`` (or explicit solver failure).
    These bounds and the actual solve initial state are explicit because the
    native solution may save neither endpoint. Hybrid differential solutions
    and unqualified raw event batches are refused: continuous target crossings
    cannot be recovered from a jump ledger. The target is a scalar pure
    predicate. This reduction supports JIT, not discrete-path differentiation.
    """
    if not isinstance(solution, JumpSolution):
        raise TypeError(
            "Event-exact first passage requires a native pure-jump JumpSolution."
        )
    events = solution.events
    if events.post_states is None:
        raise ValueError("First passage requires event post_states.")
    start, end = jnp.asarray(t0), jnp.asarray(t1)
    if start.shape != () or end.shape != ():
        raise ValueError("First-passage bounds must be scalars.")
    end = eqx.error_if(
        end,
        ~jnp.isfinite(start) | ~jnp.isfinite(end) | (end < start),
        "First-passage bounds must be finite and ordered.",
    )
    initial = jnp.broadcast_to(
        jnp.asarray(initial_state), events.batch_shape + events.state_shape
    )
    after = events.post_states.reshape((-1,) + events.state_shape)
    post_target = jax.vmap(target)(after).reshape(events.times.shape)
    initial_target = jax.vmap(target)(
        initial.reshape((-1,) + events.state_shape)
    ).reshape(events.batch_shape)
    eligible = events.valid & (events.times >= start) & (events.times <= end)
    post_hit = jnp.min(jnp.where(eligible & post_target, events.times, jnp.inf), axis=-1)
    first = jnp.where(initial_target, start, post_hit)
    hit = jnp.isfinite(first)
    successful = events.status == JUMP_SUCCESS
    last_event = jnp.max(jnp.where(eligible, events.times, start), axis=-1)
    return JumpFirstHit(
        time=first,
        hit=hit,
        initially_in_target=initial_target,
        censored=~hit & successful,
        incomplete=~hit & ~successful,
        capacity_failure=events.status == JUMP_MAX_EVENTS,
        observation_end=jnp.where(successful, end, last_event),
        status=events.status,
    )


class FiniteHittingResult(StrictModule):
    """Absorbing-target probabilities and unconditional MFPT in generator time.

    Unreachable states have hitting probability zero. Any state with a positive
    probability of avoiding the target forever has infinite unconditional MFPT.
    Failed numeric solves return NaN rather than a regularized finite answer.
    """

    hitting_probability: Array
    mean_first_passage_time: Array
    reachable: Array
    almost_sure: Array
    communicating_class: Array
    closed_avoiding_class: Array
    probability_residual: Array
    mfpt_residual: Array
    successful: Array


def finite_generator_hitting(
    generator: FiniteStateGenerator,
    target: ArrayLike,
    /,
) -> FiniteHittingResult:
    """Host-prepare reachability, then solve nonsingular absorbing subproblems.

    Only closed generators are admitted, including explicit zero escaped-rate
    evidence. Positive rates define reachability without a numerical rate
    cutoff. No pseudo-inverse, diagonal regularization or reflecting truncation.
    """
    matrix = jnp.asarray(generator.matrix)
    host = np.asarray(jax.device_get(matrix))
    mask = np.asarray(jax.device_get(jnp.asarray(target)))
    count = host.shape[0]
    if (
        count == 0
        or host.shape != (count, count)
        or mask.shape != (count,)
        or mask.dtype != bool
    ):
        raise ValueError("A square generator and boolean state target mask are required.")
    if not np.all(np.isfinite(host)):
        raise ValueError("Generator rates must be finite.")
    offdiag = host.copy()
    np.fill_diagonal(offdiag, 0.0)
    scale = max(1.0, float(np.max(np.abs(host))))
    tolerance = np.finfo(host.dtype).eps * count * scale * 16
    if (
        np.any(offdiag < 0)
        or np.any(np.diag(host) > 0)
        or np.any(np.abs(host.sum(axis=1)) > tolerance)
        or np.any(np.asarray(jax.device_get(generator.escaped_rates)) != 0)
    ):
        raise ValueError(
            "Absorbing analysis requires a closed valid generator; escaped rates refuse."
        )
    # Make targets absorbing before classifying paths that can avoid the target.
    adjacency = offdiag > 0
    adjacency[mask] = False
    reach = adjacency | np.eye(count, dtype=bool)
    for k in range(count):
        reach |= reach[:, k, None] & reach[None, k, :]
    reachable = np.any(reach[:, mask], axis=1)
    communicating = reach & reach.T
    classes = np.min(np.where(communicating, np.arange(count)[None, :], count), axis=1)
    bad_closed = np.zeros(count, dtype=bool)
    for label in np.unique(classes):
        members = classes == label
        closed = not np.any(adjacency[np.ix_(members, ~members)])
        if closed and not np.any(mask & members):
            bad_closed |= members
    can_avoid = np.any(reach[:, bad_closed], axis=1)
    almost_sure = reachable & ~can_avoid
    probability = jnp.asarray(mask, dtype=matrix.dtype)
    mfpt = jnp.where(jnp.asarray(mask), 0.0, jnp.inf).astype(matrix.dtype)
    success = jnp.asarray(True)
    probability_residual = jnp.asarray(0.0, dtype=matrix.dtype)
    mfpt_residual = jnp.asarray(0.0, dtype=matrix.dtype)
    policy = la.LinearSolvePolicy(la.DenseLU(), failure=la.FailurePolicy("status"))
    candidates = np.flatnonzero(reachable & ~mask)
    if candidates.size:
        index = jnp.asarray(candidates)
        block = -matrix[index[:, None], index[None, :]]
        rhs = jnp.sum(matrix[index][:, jnp.asarray(np.flatnonzero(mask))], axis=1)
        solved = la.solve(
            la.LinearSystem(la.DenseLinearOperator(block)), rhs, policy=policy
        )
        values = jnp.where(solved.successful, solved.value, jnp.nan)
        probability = probability.at[index].set(values)
        probability_residual = jnp.max(jnp.abs(block @ values - rhs))
        success = success & solved.successful
    certain = np.flatnonzero(almost_sure & ~mask)
    if certain.size:
        index = jnp.asarray(certain)
        block = -matrix[index[:, None], index[None, :]]
        rhs = jnp.ones((certain.size,), dtype=matrix.dtype)
        solved = la.solve(
            la.LinearSystem(la.DenseLinearOperator(block)), rhs, policy=policy
        )
        values = jnp.where(solved.successful, solved.value, jnp.nan)
        mfpt = mfpt.at[index].set(values)
        mfpt_residual = jnp.max(jnp.abs(block @ values - rhs))
        success = success & solved.successful
    return FiniteHittingResult(
        hitting_probability=probability,
        mean_first_passage_time=mfpt,
        reachable=jnp.asarray(reachable),
        almost_sure=jnp.asarray(almost_sure),
        communicating_class=jnp.asarray(classes),
        closed_avoiding_class=jnp.asarray(bad_closed),
        probability_residual=probability_residual,
        mfpt_residual=mfpt_residual,
        successful=success,
    )


__all__ = [
    "FiniteHittingResult",
    "JumpFirstHit",
    "event_first_hit",
    "finite_generator_hitting",
]
