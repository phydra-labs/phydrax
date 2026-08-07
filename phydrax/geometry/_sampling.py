#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jaxtyping import Array, Key

from .._strict import StrictModule


if TYPE_CHECKING:
    from ._atlas import BoundaryAtlas


@dataclass(frozen=True, slots=True)
class RejectionSamplingPlan:
    """Static limits for fixed-shape rejection sampling."""

    proposals_per_round: int = 256
    maximum_rounds: int = 64

    def __post_init__(self):
        if self.proposals_per_round <= 0:
            raise ValueError("proposals_per_round must be positive.")
        if self.maximum_rounds <= 0:
            raise ValueError("maximum_rounds must be positive.")

    @property
    def maximum_proposals(self) -> int:
        return self.proposals_per_round * self.maximum_rounds


@dataclass(frozen=True, slots=True)
class AtlasSamplingPlan:
    """Static candidate budget for Jacobian-weighted chart resampling."""

    candidates_per_sample: int = 8
    minimum_candidates: int = 64

    def __post_init__(self):
        if self.candidates_per_sample <= 0 or self.minimum_candidates <= 0:
            raise ValueError("Atlas sampling candidate counts must be positive.")


class SamplingReport(StrictModule):
    """JAX-compatible diagnostics for one bounded sampling attempt."""

    requested: int = eqx.field(static=True)
    proposed: Array
    accepted: Array
    rounds: Array
    complete: Array
    acceptance_rate: Array

    def __init__(
        self,
        *,
        requested: int,
        proposed: Array,
        accepted: Array,
        rounds: Array,
    ):
        proposed_ = jnp.asarray(proposed, dtype=jnp.int32).reshape(())
        accepted_ = jnp.asarray(accepted, dtype=jnp.int32).reshape(())
        rounds_ = jnp.asarray(rounds, dtype=jnp.int32).reshape(())
        self.requested = int(requested)
        self.proposed = proposed_
        self.accepted = accepted_
        self.rounds = rounds_
        self.complete = accepted_ == int(requested)
        self.acceptance_rate = jnp.where(
            proposed_ > 0,
            accepted_.astype(float) / proposed_.astype(float),
            jnp.asarray(0.0),
        )


class SamplingResult(StrictModule):
    """Fixed-shape samples, validity mask, and completion diagnostics."""

    points: Array
    valid: Array
    report: SamplingReport
    weights: Array
    strata: Array

    def __init__(
        self,
        points: Array,
        valid: Array,
        report: SamplingReport,
        *,
        weights: Array | None = None,
        strata: Array | None = None,
    ):
        points_ = jnp.asarray(points)
        if points_.ndim != 2:
            raise ValueError("SamplingResult.points must have shape (num_points, dim).")
        valid_ = jnp.asarray(valid, dtype=bool).reshape((points_.shape[0],))
        if report.requested != points_.shape[0]:
            raise ValueError("SamplingReport.requested must match the sample count.")
        count = points_.shape[0]
        if weights is None:
            denominator = jnp.maximum(jnp.sum(valid_, dtype=jnp.int32), 1)
            weights_ = jnp.where(valid_, 1.0 / denominator, 0.0)
        else:
            weights_ = jnp.asarray(weights, dtype=float).reshape((count,))
        strata_ = (
            -jnp.ones((count,), dtype=jnp.int32)
            if strata is None
            else jnp.asarray(strata, dtype=jnp.int32).reshape((count,))
        )
        self.points = points_
        self.valid = valid_
        self.report = report
        self.weights = weights_
        self.strata = strata_


def complete_sampling_result(
    points: Array,
    /,
    *,
    weights: Array | None = None,
    strata: Array | None = None,
) -> SamplingResult:
    """Wrap direct samples in a complete result."""
    points_ = jnp.asarray(points)
    if points_.ndim != 2:
        raise ValueError("Direct samples must have shape (num_points, dim).")
    count = points_.shape[0]
    report = SamplingReport(
        requested=count,
        proposed=jnp.asarray(count, dtype=jnp.int32),
        accepted=jnp.asarray(count, dtype=jnp.int32),
        rounds=jnp.asarray(1 if count else 0, dtype=jnp.int32),
    )
    return SamplingResult(
        points_,
        jnp.ones((count,), dtype=bool),
        report,
        weights=weights,
        strata=strata,
    )


def bounded_rejection_sample(
    proposal: Callable[[Key[Array, ""], int], Array],
    accept: Callable[[Array], Array],
    *,
    num_points: int,
    point_dimension: int,
    key: Key[Array, ""],
    plan: RejectionSamplingPlan = RejectionSamplingPlan(),
    dtype: jnp.dtype | None = None,
) -> SamplingResult:
    """Collect accepted proposals with bounded JAX control flow.

    The result always has shape ``(num_points, point_dimension)``. Callers inspect
    ``result.report.complete`` or use :func:`require_complete` before exposing only
    the point array.
    """
    count = int(num_points)
    dimension = int(point_dimension)
    if count < 0:
        raise ValueError("num_points must be non-negative.")
    if dimension <= 0:
        raise ValueError("point_dimension must be positive.")

    sample_dtype = jnp.dtype(float if dtype is None else dtype)
    initial_points = jnp.zeros((count, dimension), dtype=sample_dtype)
    requested_count = jnp.asarray(count, dtype=jnp.int32)
    maximum_rounds = jnp.asarray(plan.maximum_rounds, dtype=jnp.int32)
    proposals_per_round = jnp.asarray(plan.proposals_per_round, dtype=jnp.int32)
    initial = (
        key,
        initial_points,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )

    def condition(state):
        _, _, accepted_count, _, rounds = state
        return (accepted_count < requested_count) & (rounds < maximum_rounds)

    def body(state):
        loop_key, points, accepted_count, proposed_count, rounds = state
        loop_key, proposal_key = jr.split(loop_key)
        candidates = jnp.asarray(
            proposal(proposal_key, plan.proposals_per_round),
            dtype=sample_dtype,
        )
        if candidates.shape != (plan.proposals_per_round, dimension):
            raise ValueError(
                "proposal must return shape "
                f"({plan.proposals_per_round}, {dimension}), got {candidates.shape}."
            )
        accepted_mask = jnp.asarray(accept(candidates), dtype=bool)
        if accepted_mask.shape != (plan.proposals_per_round,):
            raise ValueError(
                "accept must return shape "
                f"({plan.proposals_per_round},), got {accepted_mask.shape}."
            )

        ranks = jnp.cumsum(
            accepted_mask.astype(jnp.int32), dtype=jnp.int32
        ) - jnp.asarray(1, dtype=jnp.int32)
        remaining = requested_count - accepted_count
        selected = accepted_mask & (ranks < remaining)
        targets = jnp.where(selected, accepted_count + ranks, requested_count)
        points = points.at[targets].set(candidates, mode="drop")
        accepted_this_round = jnp.minimum(
            jnp.sum(accepted_mask.astype(jnp.int32), dtype=jnp.int32),
            remaining,
        )
        return (
            loop_key,
            points,
            accepted_count + accepted_this_round,
            proposed_count + proposals_per_round,
            rounds + jnp.asarray(1, dtype=jnp.int32),
        )

    _, points, accepted_count, proposed_count, rounds = lax.while_loop(
        condition,
        body,
        initial,
    )
    valid = jnp.arange(count, dtype=jnp.int32) < accepted_count
    report = SamplingReport(
        requested=count,
        proposed=proposed_count,
        accepted=accepted_count,
        rounds=rounds,
    )
    return SamplingResult(points, valid, report)


def sample_boundary_atlas(
    atlas: BoundaryAtlas,
    num_points: int,
    /,
    *,
    key: Key[Array, ""],
    plan: AtlasSamplingPlan = AtlasSamplingPlan(),
) -> SamplingResult:
    """Sample charts in physical measure using Jacobian-weighted candidates."""
    from ._atlas import BoundaryAtlas

    if not isinstance(atlas, BoundaryAtlas):
        raise TypeError("sample_boundary_atlas requires a BoundaryAtlas.")
    count = int(num_points)
    if count < 0:
        raise ValueError("num_points must be non-negative.")
    if count == 0:
        return complete_sampling_result(
            jnp.empty((0, atlas.ambient_dimension), dtype=float),
            strata=jnp.empty((0,), dtype=jnp.int32),
        )
    candidate_count = max(
        plan.minimum_candidates,
        plan.candidates_per_sample * count,
    )
    chart_key, reference_key, selection_key = jr.split(key, 3)
    charts = jr.randint(
        chart_key,
        (candidate_count,),
        0,
        atlas.num_charts,
        dtype=jnp.int32,
    )
    reference = jr.uniform(
        reference_key,
        (candidate_count, atlas.reference_dimension),
    )
    density = atlas.jacobian(charts, reference)
    density = jnp.where(
        atlas.seam_owner[charts] & atlas.reference_mask(charts, reference),
        density,
        0.0,
    )
    total_density = jnp.sum(density)
    available = total_density > 0.0
    probabilities = jnp.where(
        available,
        density / jnp.maximum(total_density, jnp.finfo(density.dtype).tiny),
        jnp.full_like(density, 1.0 / candidate_count),
    )
    selected = jr.choice(
        selection_key,
        candidate_count,
        (count,),
        replace=True,
        p=probabilities,
    )
    selected_charts = charts[selected]
    accepted = jnp.where(available, count, 0).astype(jnp.int32)
    valid = jnp.full((count,), available, dtype=bool)
    report = SamplingReport(
        requested=count,
        proposed=jnp.asarray(candidate_count, dtype=jnp.int32),
        accepted=accepted,
        rounds=jnp.asarray(1, dtype=jnp.int32),
    )
    return SamplingResult(
        atlas.map(selected_charts, reference[selected]),
        valid,
        report,
        weights=jnp.where(valid, 1.0 / count, 0.0),
        strata=atlas.source_entity_ids[selected_charts],
    )


def require_complete(result: SamplingResult, /, *, context: str) -> Array:
    """Return points while preserving a JIT-compatible underfill error."""
    return eqx.error_if(
        result.points,
        pred=~result.report.complete,
        msg=(
            f"{context} exhausted its bounded proposal budget before filling the "
            "requested fixed-shape batch. Inspect SamplingReport for diagnostics."
        ),
    )


__all__ = [
    "AtlasSamplingPlan",
    "RejectionSamplingPlan",
    "SamplingReport",
    "SamplingResult",
    "bounded_rejection_sample",
    "complete_sampling_result",
    "sample_boundary_atlas",
    "require_complete",
]
