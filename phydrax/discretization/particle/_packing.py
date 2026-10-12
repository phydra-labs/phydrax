#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded dense-reference random sequential packing through DEM shape geometry.

The immutable preparation preflights ``capacity * tries * pair_capacity`` before
allocating the canonical all-pair relation. Runtime placement visits physical
IDs in increasing order and addresses independent pose streams by the full
64-bit ID and try ordinal. Bounding balls certify domain containment and prune
pairs; only the shape's narrow-phase geometry admits an overlapping-ball pair.
Candidate poses are diagnostic. Incompleteness admits no body: accepted poses
are NaN and the entire accepted mask is false, not a partial usable packing.
"""

from __future__ import annotations

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...sparse import EdgeRelation
from ...typing import checked, PRNGKey
from ._pair_state import ParticlePairKeySpace
from ._pairwise import ParticlePairRelation
from ._rigid_body import RigidBodyKinematics
from ._rigid_shape import AbstractPreparedRigidContactShape


@final
class RigidDEMPackingResult(StrictModule):
    candidate_kinematics: RigidBodyKinematics
    accepted_kinematics: RigidBodyKinematics
    attempt_count: Array
    placed_mask: Array
    accepted_mask: Array
    geometry_valid: Array
    bounds_margin: Array
    requested_count: Array
    placed_count: Array
    candidate_evaluation_count: Array
    complete: Array
    successful: Array
    pair_capacity: int = eqx.field(static=True)
    contact_capacity: int = eqx.field(static=True)
    required_evaluation_budget: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    shape_id: str = eqx.field(static=True)


@final
class _PackingState(StrictModule):
    position: Array
    orientation: Array
    attempts: Array
    placed: Array
    geometry_valid: Array
    bounds_margin: Array
    evaluations: Array


def _random_pose(
    key: PRNGKey, particle_id: Array, ordinal: Array, lower: Array, upper: Array, /
) -> tuple[Array, Array]:
    # fold_in consumes 32 bits; both words are needed for physical int64 IDs.
    stream = jax.random.fold_in(key, particle_id.astype(jnp.uint32))
    stream = jax.random.fold_in(stream, (particle_id >> 32).astype(jnp.uint32))
    stream = jax.random.fold_in(stream, ordinal.astype(jnp.uint32))
    position = jax.random.uniform(
        jax.random.fold_in(stream, 0),
        lower.shape,
        dtype=lower.dtype,
        minval=lower,
        maxval=upper,
    )
    if lower.shape[0] == 2:
        orientation = jax.random.uniform(
            jax.random.fold_in(stream, 1),
            (1,),
            dtype=lower.dtype,
            minval=0.0,
            maxval=2.0 * jnp.pi,
        )
    else:
        # Uniform measure on SO(3), using the unit-quaternion double cover.
        uniform = jax.random.uniform(
            jax.random.fold_in(stream, 1), (3,), dtype=lower.dtype
        )
        a, b = jnp.sqrt(1.0 - uniform[0]), jnp.sqrt(uniform[0])
        theta, phi = 2.0 * jnp.pi * uniform[1], 2.0 * jnp.pi * uniform[2]
        orientation = jnp.stack(
            (b * jnp.cos(phi), a * jnp.sin(theta), a * jnp.cos(theta), b * jnp.sin(phi))
        )
    return position, orientation


def _packing_attempt(
    plan: RandomSequentialPackingPlan,
    shapes: AbstractPreparedRigidContactShape,
    key: PRNGKey,
    pairs: ParticlePairRelation,
    key_space: ParticlePairKeySpace,
    contact_capacity: int,
    body: Array,
    state: _PackingState,
    /,
) -> _PackingState:
    particles = shapes.bodies.particles
    radius = shapes.bounding_radii[body]
    lower = plan.lower.astype(state.position.dtype)
    upper = plan.upper.astype(state.position.dtype)
    point, orientation = _random_pose(
        key,
        particles.particle_ids[body],
        state.attempts[body],
        lower + radius,
        upper - radius,
    )
    position = state.position.at[body].set(point)
    orientations = state.orientation.at[body].set(orientation)
    margin = jnp.min(jnp.minimum(point - radius - lower, upper - point - radius))
    in_bounds = jnp.isfinite(margin) & (margin >= 0.0)
    left, right = pairs.left_indices, pairs.right_indices
    connected = ((left == body) & state.placed[right]) | (
        (right == body) & state.placed[left]
    )
    delta = position[left] - position[right]
    radius_sum = shapes.bounding_radii[left] + shapes.bounding_radii[right]
    nearby = jnp.sum(delta * delta, axis=-1) <= radius_sum * radius_sum
    candidate = connected & nearby & in_bounds
    runtime_pairs = eqx.tree_at(lambda item: item.relation.valid, pairs, candidate)
    dimension = particles.ambient_dimension
    kinematics = RigidBodyKinematics(
        position,
        jnp.zeros_like(position),
        orientations,
        jnp.zeros((particles.capacity, 1 if dimension == 2 else 3), dtype=position.dtype),
    )

    def narrow_phase(_: None) -> tuple[Array, Array]:
        batch = shapes.pair_geometry(
            kinematics,
            jnp.ones((particles.capacity,), dtype=position.dtype),
            runtime_pairs,
            key_space,
            maximum_contact_routes=contact_capacity,
        )
        geometry = batch.geometry
        valid = batch.successful & jnp.all(~batch.keys.valid | geometry.valid)
        separated = jnp.all(~geometry.valid | (geometry.gap >= 0.0))
        return valid, separated

    queried = jnp.any(candidate)
    if pairs.capacity == 0:
        valid, separated = (
            jnp.asarray(True, dtype=jnp.bool_),
            jnp.asarray(True, dtype=jnp.bool_),
        )
    else:
        valid, separated = jax.lax.cond(
            queried,
            narrow_phase,
            lambda _: (
                jnp.asarray(True, dtype=jnp.bool_),
                jnp.asarray(True, dtype=jnp.bool_),
            ),
            None,
        )
    admitted = in_bounds & valid & separated
    return _PackingState(
        position,
        orientations,
        state.attempts.at[body].add(1),
        state.placed.at[body].set(admitted),
        state.geometry_valid.at[body].set(valid),
        state.bounds_margin.at[body].set(margin),
        state.evaluations + jnp.where(queried, pairs.capacity, 0).astype(jnp.int64),
    )


@final
class RandomSequentialPackingPlan(StrictModule, NonTrainableState):
    lower: Array
    upper: Array
    maximum_tries_per_body: int = eqx.field(static=True)
    maximum_candidate_evaluations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        lower: ArrayLike,
        upper: ArrayLike,
        maximum_tries_per_body: int,
        /,
        *,
        maximum_candidate_evaluations: int,
        plan_id: str | None = None,
    ) -> None:
        lower_host, upper_host = np.asarray(lower), np.asarray(upper)
        if lower_host.ndim != 1 or lower_host.shape not in ((2,), (3,)):
            raise ValueError("Packing bounds require ambient dimension 2 or 3.")
        if upper_host.shape != lower_host.shape:
            raise ValueError("Packing bounds must have matching shapes.")
        if not np.all(np.isfinite(lower_host)) or not np.all(np.isfinite(upper_host)):
            raise ValueError("Packing bounds must be finite.")
        if np.any(upper_host <= lower_host):
            raise ValueError("Packing upper bounds must exceed lower bounds.")
        if type(maximum_tries_per_body) is not int or maximum_tries_per_body <= 0:
            raise ValueError("maximum_tries_per_body must be a positive integer.")
        if maximum_tries_per_body > np.iinfo(np.int32).max:
            raise ValueError("maximum_tries_per_body exceeds the int32 attempt capacity.")
        if (
            type(maximum_candidate_evaluations) is not int
            or maximum_candidate_evaluations < 0
        ):
            raise ValueError(
                "maximum_candidate_evaluations must be a nonnegative integer."
            )
        dtype = np.result_type(lower_host.dtype, upper_host.dtype, np.float32)
        lower_host, upper_host = lower_host.astype(dtype), upper_host.astype(dtype)
        generated = canonical_fingerprint(
            {
                "kind": "random-sequential-dem-packing",
                "oracle": "bounded-dense-shape-pairs",
                "bounds": array_tree_fingerprint(
                    {"lower": lower_host, "upper": upper_host}
                ),
                "maximum_tries_per_body": maximum_tries_per_body,
                "maximum_candidate_evaluations": maximum_candidate_evaluations,
                "rng_address": "physical-int64-id-try-pose-component",
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be nonempty.")
        self.lower, self.upper = jnp.asarray(lower_host), jnp.asarray(upper_host)
        self.maximum_tries_per_body = maximum_tries_per_body
        self.maximum_candidate_evaluations = maximum_candidate_evaluations
        self.plan_id = identifier

    @checked
    def initialize(
        self, shapes: AbstractPreparedRigidContactShape, key: PRNGKey, /
    ) -> RigidDEMPackingResult:
        """Prepare a bounded reference relation, then execute device-only tries."""
        particles = shapes.bodies.particles
        capacity, dimension = particles.capacity, particles.ambient_dimension
        if self.lower.shape != (dimension,):
            raise ValueError("Packing bounds must match the shape ambient dimension.")
        pair_capacity = capacity * (capacity - 1) // 2
        required = capacity * self.maximum_tries_per_body * pair_capacity
        if required > np.iinfo(np.int64).max:
            raise ValueError(
                "Packing evaluation count exceeds the int64 evidence capacity."
            )
        if required > self.maximum_candidate_evaluations:
            raise ValueError(
                f"Packing candidate-evaluation budget requires {required}, "
                f"exceeding {self.maximum_candidate_evaluations}."
            )
        contact_capacity = shapes.contact_capacity(
            pair_capacity, pair_capacity * shapes.routes_per_pair
        )
        key_space = ParticlePairKeySpace(particles)
        # Canonical host-only topology preparation precedes all numerical loops.
        ids = np.asarray(particles.particle_ids, dtype=np.int64)
        order_host = np.argsort(ids, kind="stable")
        left_rank, right_rank = np.triu_indices(capacity, 1)
        left, right = order_host[left_rank], order_host[right_rank]
        pairs = ParticlePairRelation(
            EdgeRelation(left, right, source_size=capacity, target_size=capacity),
            ids[left],
            ids[right],
            source_support_id=particles.support.support_id,
            target_support_id=particles.support.support_id,
            same_set=True,
            unordered=True,
        )
        dtype = particles.safe_masses.dtype
        orientation = jnp.zeros((capacity, 1 if dimension == 2 else 4), dtype=dtype)
        if dimension == 3:
            orientation = orientation.at[:, 0].set(1.0)
        initial = _PackingState(
            jnp.zeros((capacity, dimension), dtype=dtype),
            orientation,
            jnp.zeros((capacity,), dtype=jnp.int32),
            jnp.zeros((capacity,), dtype=jnp.bool_),
            jnp.ones((capacity,), dtype=jnp.bool_),
            jnp.full((capacity,), jnp.inf, dtype=dtype),
            jnp.asarray(0, dtype=jnp.int64),
        )
        order = jnp.asarray(order_host, dtype=jnp.int32)

        def place(rank: Array, state: _PackingState) -> _PackingState:
            body = order[rank]

            def pending(value: _PackingState) -> Array:
                return (
                    particles.active_mask[body]
                    & ~value.placed[body]
                    & (value.attempts[body] < self.maximum_tries_per_body)
                )

            def attempt(value: _PackingState) -> _PackingState:
                return _packing_attempt(
                    self, shapes, key, pairs, key_space, contact_capacity, body, value
                )

            return jax.lax.while_loop(pending, attempt, state)

        state = jax.lax.fori_loop(0, capacity, place, initial)
        complete = jnp.all(~particles.active_mask | state.placed)
        successful = complete & jnp.all(~particles.active_mask | state.geometry_valid)
        candidate = RigidBodyKinematics(
            state.position,
            jnp.zeros_like(state.position),
            state.orientation,
            jnp.zeros((capacity, 1 if dimension == 2 else 3), dtype=dtype),
        )
        accepted = jax.tree.map(
            lambda value: jnp.where(successful, value, jnp.nan), candidate
        )
        return RigidDEMPackingResult(
            candidate,
            accepted,
            state.attempts,
            state.placed,
            state.placed & successful,
            state.geometry_valid,
            state.bounds_margin,
            jnp.sum(particles.active_mask, dtype=jnp.int32),
            jnp.sum(state.placed, dtype=jnp.int32),
            state.evaluations,
            complete,
            successful,
            pair_capacity,
            contact_capacity,
            required,
            self.plan_id,
            shapes.shape_id,
        )


__all__ = ["RandomSequentialPackingPlan", "RigidDEMPackingResult"]
