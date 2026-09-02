#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule


class NestedSamplingCapacity(StrictModule):
    """Hard fixed capacities for prepared dynamic nested execution."""

    max_live: int = eqx.field(static=True)
    max_dead_points: int = eqx.field(static=True)
    max_likelihood_evaluations: int = eqx.field(static=True)
    max_dynamic_batches: int = eqx.field(static=True)
    max_clusters: int = eqx.field(static=True)
    max_phantoms: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_live: int,
        max_dead_points: int,
        max_likelihood_evaluations: int,
        max_dynamic_batches: int,
        max_clusters: int,
        max_phantoms: int,
    ):
        values = tuple(
            int(value)
            for value in (
                max_live,
                max_dead_points,
                max_likelihood_evaluations,
                max_dynamic_batches,
                max_clusters,
                max_phantoms,
            )
        )
        if any(value <= 0 for value in values):
            raise ValueError("Every nested-sampling capacity must be positive.")
        if values[0] < 2 or values[1] < values[0] or values[2] < values[0]:
            raise ValueError(
                "Nested capacities require max_live >= 2 and dead/evaluation caps >= max_live."
            )
        (
            self.max_live,
            self.max_dead_points,
            self.max_likelihood_evaluations,
            self.max_dynamic_batches,
            self.max_clusters,
            self.max_phantoms,
        ) = values


class PeriodicNestedCoordinate(StrictModule):
    """Explicit one-dimensional periodic parameter topology."""

    path: str = eqx.field(static=True)
    origin: float = eqx.field(static=True)
    period: float = eqx.field(static=True)

    def __init__(self, path: str, origin: float, period: float, /):
        path_ = str(path)
        origin_ = float(origin)
        period_ = float(period)
        if not path_:
            raise ValueError("Periodic nested coordinate path must be non-empty.")
        if not math.isfinite(origin_) or not math.isfinite(period_) or period_ <= 0.0:
            raise ValueError(
                "Periodic origin/period must be finite with positive period."
            )
        self.path = path_
        self.origin = origin_
        self.period = period_

    def wrap(self, value: ArrayLike, /) -> Array:
        array = jnp.asarray(value)
        return self.origin + jnp.mod(array - self.origin, self.period)

    def displacement(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        raw = jnp.asarray(right) - jnp.asarray(left)
        return jnp.mod(raw + 0.5 * self.period, self.period) - 0.5 * self.period

    def embedding(self, value: ArrayLike, /) -> Array:
        angle = 2.0 * jnp.pi * (self.wrap(value) - self.origin) / self.period
        return jnp.stack((jnp.cos(angle), jnp.sin(angle)), axis=-1)


class NestedPriorPlan(StrictModule):
    """Declared Lebesgue, finite-counting, and periodic coordinate blocks."""

    continuous_paths: tuple[str, ...] = eqx.field(static=True)
    finite_supports: tuple[tuple[str, Array, Array], ...]
    periodic: tuple[PeriodicNestedCoordinate, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        continuous_paths: Sequence[str],
        finite_supports: Mapping[str, tuple[ArrayLike, ArrayLike]] | None = None,
        periodic: Sequence[PeriodicNestedCoordinate] = (),
    ):
        continuous = tuple(str(path) for path in continuous_paths)
        if any(not path for path in continuous) or len(set(continuous)) != len(
            continuous
        ):
            raise ValueError("continuous_paths must contain distinct nonempty paths.")
        support_values = []
        for path, (support, masses) in (
            {} if finite_supports is None else finite_supports
        ).items():
            path_ = str(path)
            values = jnp.asarray(support)
            probabilities = jnp.asarray(masses, dtype=float)
            if not path_ or values.ndim != 1 or values.size == 0:
                raise ValueError(
                    "Finite counting supports must be named nonempty vectors."
                )
            if (
                probabilities.shape != values.shape
                or bool(jnp.any(~jnp.isfinite(probabilities)))
                or bool(jnp.any(probabilities <= 0.0))
                or not bool(jnp.isclose(jnp.sum(probabilities), 1.0))
            ):
                raise ValueError(
                    "Finite counting masses must be positive and sum to one."
                )
            support_values.append((path_, values, probabilities))
        periodic_ = tuple(periodic)
        if any(not isinstance(item, PeriodicNestedCoordinate) for item in periodic_):
            raise TypeError("periodic entries must be PeriodicNestedCoordinate.")
        paths = continuous + tuple(item[0] for item in support_values)
        if len(paths) != len(set(paths)):
            raise ValueError("Nested prior continuous/counting blocks must be disjoint.")
        if any(item.path not in continuous for item in periodic_):
            raise ValueError("Periodic coordinates must name declared continuous paths.")
        if len({item.path for item in periodic_}) != len(periodic_):
            raise ValueError("Periodic coordinate paths must be distinct.")
        self.continuous_paths = continuous
        self.finite_supports = tuple(support_values)
        self.periodic = periodic_
        self.plan_id = canonical_fingerprint(
            {
                "continuous": list(continuous),
                "supports": [
                    {
                        "path": path,
                        "content": array_tree_fingerprint(
                            {"values": values, "probabilities": probabilities}
                        )["sha256"],
                    }
                    for path, values, probabilities in support_values
                ],
                "periodic": [
                    {
                        "path": item.path,
                        "origin": item.origin,
                        "period": item.period,
                    }
                    for item in periodic_
                ],
            }
        )


class NestedProposalPlan(StrictModule):
    """Composed bounded constrained-prior replacement kernels."""

    base: Literal["hit-and-run", "slice-within-gibbs"] = eqx.field(static=True)
    ellipsoid: bool = eqx.field(static=True)
    discrete_gibbs: bool = eqx.field(static=True)
    periodic_slice: bool = eqx.field(static=True)
    phantom_recycling: bool = eqx.field(static=True)
    learned_flow: bool = eqx.field(static=True)
    gradient_guided: bool = eqx.field(static=True)
    ellipsoid_enlargement: float = eqx.field(static=True)
    gradient_step_size: float = eqx.field(static=True)
    maximum_attempts: int = eqx.field(static=True)
    slice_scale: float = eqx.field(static=True)
    gradient_barrier_scale: float = eqx.field(static=True)
    rejection_fallback: bool = eqx.field(static=True)

    def __init__(
        self,
        base: Literal["hit-and-run", "slice-within-gibbs"] = "hit-and-run",
        /,
        *,
        ellipsoid: bool = False,
        discrete_gibbs: bool = False,
        periodic_slice: bool = False,
        phantom_recycling: bool = False,
        learned_flow: bool = False,
        gradient_guided: bool = False,
        ellipsoid_enlargement: float = 1.1,
        gradient_step_size: float = 0.1,
        maximum_attempts: int = 100,
        slice_scale: float = 1.0,
        gradient_barrier_scale: float = 0.1,
        rejection_fallback: bool = False,
    ):
        if base not in ("hit-and-run", "slice-within-gibbs"):
            raise ValueError("Unknown nested base proposal.")
        enlargement = float(ellipsoid_enlargement)
        gradient_step = float(gradient_step_size)
        slice_scale_ = float(slice_scale)
        barrier_scale = float(gradient_barrier_scale)
        attempts = int(maximum_attempts)
        if not math.isfinite(enlargement) or enlargement <= 1.0:
            raise ValueError("ellipsoid_enlargement must be finite and exceed one.")
        if not math.isfinite(gradient_step) or gradient_step <= 0.0:
            raise ValueError("gradient_step_size must be finite and positive.")
        if not math.isfinite(slice_scale_) or slice_scale_ <= 0.0:
            raise ValueError("slice_scale must be finite and positive.")
        if not math.isfinite(barrier_scale) or barrier_scale <= 0.0:
            raise ValueError("gradient_barrier_scale must be finite and positive.")
        if attempts <= 0:
            raise ValueError("maximum_attempts must be positive.")
        self.base = base
        self.ellipsoid = bool(ellipsoid)
        self.discrete_gibbs = bool(discrete_gibbs)
        self.periodic_slice = bool(periodic_slice)
        self.phantom_recycling = bool(phantom_recycling)
        self.learned_flow = bool(learned_flow)
        self.gradient_guided = bool(gradient_guided)
        self.ellipsoid_enlargement = enlargement
        self.gradient_step_size = gradient_step
        self.slice_scale = slice_scale_
        self.gradient_barrier_scale = barrier_scale
        self.rejection_fallback = bool(rejection_fallback)
        self.maximum_attempts = attempts


class DynamicNestedPolicy(StrictModule):
    """Bounded pilot/allocation schedule updated only between epochs."""

    pilot_dead_points: int = eqx.field(static=True)
    additional_live_per_batch: int = eqx.field(static=True)
    allocation_cadence: int = eqx.field(static=True)
    evidence_fraction: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        pilot_dead_points: int,
        additional_live_per_batch: int,
        allocation_cadence: int,
        evidence_fraction: float = 0.5,
    ):
        pilot, additional, cadence = map(
            int, (pilot_dead_points, additional_live_per_batch, allocation_cadence)
        )
        fraction = float(evidence_fraction)
        if pilot <= 0 or additional <= 0 or cadence <= 0:
            raise ValueError("Dynamic nested counts and cadence must be positive.")
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("evidence_fraction must lie in [0, 1].")
        self.pilot_dead_points = pilot
        self.additional_live_per_batch = additional
        self.allocation_cadence = cadence
        self.evidence_fraction = fraction

    def allocation_priority(
        self,
        posterior_importance: ArrayLike,
        evidence_importance: ArrayLike,
        /,
    ) -> Array:
        posterior = jnp.asarray(posterior_importance)
        evidence = jnp.asarray(evidence_importance)
        if posterior.shape != evidence.shape:
            raise ValueError("Dynamic nested importance arrays must align.")
        return (
            1.0 - self.evidence_fraction
        ) * posterior + self.evidence_fraction * evidence


class EllipsoidalNestedBounds(StrictModule):
    """Fixed-capacity enlarged ellipsoids with exact union-overlap density."""

    centers: Array
    factors: Array
    active: Array
    log_volumes: Array
    enlargement: float = eqx.field(static=True)

    def overlap_count(self, points: ArrayLike, /) -> Array:
        value = jnp.asarray(points)
        differences = value[..., None, :] - self.centers

        coordinates = jnp.stack(
            tuple(
                jsp.linalg.solve_triangular(
                    self.factors[index],
                    differences[..., index, :][..., None],
                    lower=True,
                )[..., 0]
                for index in range(self.centers.shape[0])
            ),
            axis=-2,
        )
        inside = jnp.sum(coordinates**2, axis=-1) <= 1.0
        return jnp.sum(inside & self.active, axis=-1)

    def sample(self, key: Array, /) -> tuple[Array, Array]:
        probabilities = jax.nn.softmax(jnp.where(self.active, self.log_volumes, -jnp.inf))
        cluster_key, radius_key, direction_key = jr.split(key, 3)
        cluster = jr.choice(cluster_key, self.active.size, p=probabilities)
        dimension = self.centers.shape[-1]
        direction = jr.normal(direction_key, (dimension,), dtype=self.centers.dtype)
        direction = direction / jnp.linalg.norm(direction)
        radius = jr.uniform(radius_key, (), dtype=self.centers.dtype) ** (1.0 / dimension)
        point = self.centers[cluster] + self.factors[cluster] @ (radius * direction)
        overlap = self.overlap_count(point)
        log_density = jnp.log(jnp.asarray(overlap, dtype=point.dtype)) - (
            jsp.special.logsumexp(jnp.where(self.active, self.log_volumes, -jnp.inf))
        )
        return point, log_density


class PhantomNestedState(StrictModule):
    """Fixed ring of constrained states that never enters quadrature on creation."""

    positions: Array
    log_likelihood: Array
    birth_log_likelihood: Array
    proposal_epoch: Array
    ancestry: Array
    mask: Array
    cursor: Array

    @classmethod
    def initialize(
        cls, capacity: int, dimension: int, /, *, dtype: Any
    ) -> PhantomNestedState:
        capacity_, dimension_ = int(capacity), int(dimension)
        if capacity_ <= 0 or dimension_ <= 0:
            raise ValueError("Phantom capacity and dimension must be positive.")
        return cls(
            positions=jnp.zeros((capacity_, dimension_), dtype=dtype),
            log_likelihood=jnp.full((capacity_,), -jnp.inf, dtype=dtype),
            birth_log_likelihood=jnp.full((capacity_,), -jnp.inf, dtype=dtype),
            proposal_epoch=jnp.zeros((capacity_,), dtype=jnp.int32),
            ancestry=-jnp.ones((capacity_,), dtype=jnp.int32),
            mask=jnp.zeros((capacity_,), dtype=bool),
            cursor=jnp.asarray(0, dtype=jnp.int32),
        )

    def add(
        self,
        position: ArrayLike,
        /,
        *,
        log_likelihood: ArrayLike,
        birth_log_likelihood: ArrayLike,
        proposal_epoch: int | Array,
        ancestry: int | Array,
    ) -> PhantomNestedState:
        value = jnp.asarray(position, dtype=self.positions.dtype)
        if value.shape != self.positions.shape[1:]:
            raise ValueError("Phantom position layout changed.")
        index = self.cursor
        return eqx.tree_at(
            lambda state: (
                state.positions,
                state.log_likelihood,
                state.birth_log_likelihood,
                state.proposal_epoch,
                state.ancestry,
                state.mask,
                state.cursor,
            ),
            self,
            (
                self.positions.at[index].set(value),
                self.log_likelihood.at[index].set(log_likelihood),
                self.birth_log_likelihood.at[index].set(birth_log_likelihood),
                self.proposal_epoch.at[index].set(proposal_epoch),
                self.ancestry.at[index].set(ancestry),
                self.mask.at[index].set(True),
                (index + 1) % self.positions.shape[0],
            ),
        )

    def eligible(self, threshold: ArrayLike, /) -> Array:
        threshold_ = jnp.asarray(threshold, dtype=self.log_likelihood.dtype)
        return (
            self.mask
            & (self.log_likelihood > threshold_)
            & (self.birth_log_likelihood <= threshold_)
        )


class NestedSamplingPlan(StrictModule):
    """Single immutable lifecycle joining prior, proposals, dynamic policy, and caps."""

    capacity: NestedSamplingCapacity
    prior: NestedPriorPlan
    proposal: NestedProposalPlan
    dynamic: DynamicNestedPolicy | None
    initial_live: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        capacity: NestedSamplingCapacity,
        prior: NestedPriorPlan,
        proposal: NestedProposalPlan,
        /,
        *,
        initial_live: int,
        dynamic: DynamicNestedPolicy | None = None,
    ):
        if not isinstance(capacity, NestedSamplingCapacity):
            raise TypeError("capacity must be NestedSamplingCapacity.")
        if not isinstance(prior, NestedPriorPlan):
            raise TypeError("prior must be NestedPriorPlan.")
        if not isinstance(proposal, NestedProposalPlan):
            raise TypeError("proposal must be NestedProposalPlan.")
        if dynamic is not None and not isinstance(dynamic, DynamicNestedPolicy):
            raise TypeError("dynamic must be DynamicNestedPolicy or None.")
        live = int(initial_live)
        if live < 2 or live > capacity.max_live:
            raise ValueError("initial_live must lie in [2, max_live].")
        if proposal.discrete_gibbs != bool(prior.finite_supports):
            raise ValueError(
                "Finite counting priors and discrete_gibbs must be declared together."
            )
        if proposal.periodic_slice != bool(prior.periodic):
            raise ValueError(
                "Periodic coordinates and periodic_slice must be declared together."
            )
        self.capacity = capacity
        self.prior = prior
        self.proposal = proposal
        self.dynamic = dynamic
        self.initial_live = live
        self.plan_id = canonical_fingerprint(
            {
                "capacity": {
                    "max_live": capacity.max_live,
                    "max_dead_points": capacity.max_dead_points,
                    "max_likelihood_evaluations": (capacity.max_likelihood_evaluations),
                    "max_dynamic_batches": capacity.max_dynamic_batches,
                    "max_clusters": capacity.max_clusters,
                    "max_phantoms": capacity.max_phantoms,
                },
                "prior": prior.plan_id,
                "proposal": {
                    "base": proposal.base,
                    "ellipsoid": proposal.ellipsoid,
                    "discrete_gibbs": proposal.discrete_gibbs,
                    "periodic_slice": proposal.periodic_slice,
                    "phantom_recycling": proposal.phantom_recycling,
                    "learned_flow": proposal.learned_flow,
                    "gradient_guided": proposal.gradient_guided,
                    "ellipsoid_enlargement": proposal.ellipsoid_enlargement,
                    "gradient_step_size": proposal.gradient_step_size,
                    "maximum_attempts": proposal.maximum_attempts,
                    "slice_scale": proposal.slice_scale,
                    "gradient_barrier_scale": proposal.gradient_barrier_scale,
                    "rejection_fallback": proposal.rejection_fallback,
                },
                "dynamic": (
                    None
                    if dynamic is None
                    else {
                        "pilot_dead_points": dynamic.pilot_dead_points,
                        "additional_live_per_batch": (dynamic.additional_live_per_batch),
                        "allocation_cadence": dynamic.allocation_cadence,
                        "evidence_fraction": dynamic.evidence_fraction,
                    }
                ),
                "initial_live": live,
            }
        )


__all__ = [
    "DynamicNestedPolicy",
    "EllipsoidalNestedBounds",
    "NestedPriorPlan",
    "NestedProposalPlan",
    "NestedSamplingCapacity",
    "NestedSamplingPlan",
    "PeriodicNestedCoordinate",
    "PhantomNestedState",
]
