#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-capacity active chromatin, cytoskeletal, and adhesion dynamics.

Every discrete transition is addressed deterministically, evaluated by the
identity-safe pair-relation substrate, and committed atomically. Conservative
link forces are exact gradients of scalar spring energies. Differentiation
through a step is conditional on its realized fixed event branch.
"""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.particle._relations import (
    DynamicPairRelationPlan,
    PairRelationCommitResult,
    PairRelationEventBatch,
    PairRelationEventKind,
    PairRelationEvidence,
    PairRelationState,
    PairSpringEvaluation,
    PairSpringPlan,
    PreparedDynamicPairRelations,
    PreparedPairSpringEnergy,
)


def _probability(rate: float, dt: Array) -> Array:
    return -jnp.expm1(-jnp.asarray(rate, dtype=dt.dtype) * dt)


def _keys(key: Array, realization: int, step: Array, process: int, size: int) -> Array:
    root = jr.fold_in(key, jnp.asarray(realization, dtype=jnp.uint32))
    root = jr.fold_in(root, jnp.asarray(step, dtype=jnp.uint32))
    root = jr.fold_in(root, jnp.asarray(process, dtype=jnp.uint32))
    return jax.vmap(lambda index: jr.fold_in(root, index))(
        jnp.arange(size, dtype=jnp.uint32)
    )


def _uniforms(
    key: Array, realization: int, step: Array, process: int, size: int
) -> Array:
    return jax.vmap(lambda local: jr.uniform(local, ()))(
        _keys(key, realization, step, process, size)
    )


def _batch(
    event_ids: Array,
    event_kind: Array,
    valid: Array,
    relation_ids: Array,
    incarnations: Array,
    left: Array,
    right: Array,
    relation_kind: Array,
    parameters: Array,
) -> PairRelationEventBatch:
    return PairRelationEventBatch(
        event_ids.astype(jnp.int32),
        event_kind.astype(jnp.int32),
        valid.astype(bool),
        relation_ids.astype(jnp.int32),
        incarnations.astype(jnp.int32),
        left.astype(jnp.int32),
        right.astype(jnp.int32),
        relation_kind.astype(jnp.int32),
        parameters,
    )


def _empty(
    relations: PreparedDynamicPairRelations, dtype: np.dtype | type
) -> PairRelationEventBatch:
    capacity = relations.event_capacity
    return _batch(
        jnp.arange(capacity),
        jnp.zeros((capacity,), dtype=jnp.int32),
        jnp.zeros((capacity,), dtype=bool),
        jnp.full((capacity,), -1, dtype=jnp.int32),
        jnp.full((capacity,), -1, dtype=jnp.int32),
        jnp.full((capacity,), -1, dtype=jnp.int32),
        jnp.full((capacity,), -1, dtype=jnp.int32),
        jnp.full((capacity,), -1, dtype=jnp.int32),
        jnp.zeros((capacity, relations.parameter_width), dtype=dtype),
    )


def _one(
    relations: PreparedDynamicPairRelations,
    dtype: np.dtype | type,
    *,
    event_id: ArrayLike,
    operation: PairRelationEventKind,
    valid: ArrayLike = True,
    relation_id: ArrayLike = -1,
    incarnation: ArrayLike = -1,
    left: ArrayLike = -1,
    right: ArrayLike = -1,
    relation_kind: ArrayLike = -1,
    parameters: ArrayLike | None = None,
) -> PairRelationEventBatch:
    empty = _empty(relations, dtype)
    values = (
        jnp.zeros((relations.parameter_width,), dtype=dtype)
        if parameters is None
        else jnp.asarray(parameters, dtype=dtype)
    )
    if values.shape != (relations.parameter_width,):
        raise ValueError("parameters must have the prepared relation parameter width.")
    return PairRelationEventBatch(
        empty.event_ids.at[0].set(jnp.asarray(event_id, dtype=empty.event_ids.dtype)),
        empty.event_kind.at[0].set(
            jnp.asarray(int(operation), dtype=empty.event_kind.dtype)
        ),
        empty.valid.at[0].set(jnp.asarray(valid, dtype=empty.valid.dtype)),
        empty.relation_ids.at[0].set(
            jnp.asarray(relation_id, dtype=empty.relation_ids.dtype)
        ),
        empty.relation_incarnations.at[0].set(
            jnp.asarray(incarnation, dtype=empty.relation_incarnations.dtype)
        ),
        empty.left.at[0].set(jnp.asarray(left, dtype=empty.left.dtype)),
        empty.right.at[0].set(jnp.asarray(right, dtype=empty.right.dtype)),
        empty.relation_kind.at[0].set(
            jnp.asarray(relation_kind, dtype=empty.relation_kind.dtype)
        ),
        empty.parameters.at[0].set(values),
    )


def _occupancy(state: PairRelationState, endpoint_capacity: int) -> Array:
    left = jnp.clip(state.left, 0, endpoint_capacity - 1)
    right = jnp.clip(state.right, 0, endpoint_capacity - 1)
    counts = jnp.zeros((endpoint_capacity,), dtype=jnp.int32)
    counts = counts.at[left].add(state.active.astype(jnp.int32))
    return counts.at[right].add(state.active.astype(jnp.int32))


class ChromatinDynamicsPlan(StrictModule, NonTrainableState):
    """Plan two-foot diffusion capture and collision-aware loop extrusion."""

    site_positions: Array
    roadblocks: Array
    relations: DynamicPairRelationPlan
    spring: PairSpringPlan
    binding_rate: float = eqx.field(static=True)
    unbinding_rate: float = eqx.field(static=True)
    extrusion_rate: float = eqx.field(static=True)
    capture_distance: float = eqx.field(static=True)
    spring_stiffness: float = eqx.field(static=True)
    spring_rest_length: float = eqx.field(static=True)
    realization_id: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        site_positions: ArrayLike,
        relation_capacity: int,
        /,
        *,
        roadblocks: ArrayLike | None = None,
        binding_rate: float = 0.1,
        unbinding_rate: float = 0.01,
        extrusion_rate: float = 1.0,
        capture_distance: float = 1.0,
        spring_stiffness: float = 1.0,
        spring_rest_length: float = 0.0,
        realization_id: int = 0,
        plan_id: str | None = None,
    ):
        positions = np.asarray(site_positions)
        capacity = int(relation_capacity)
        barriers = (
            np.zeros((positions.shape[0],), dtype=bool)
            if roadblocks is None and positions.ndim == 2
            else np.asarray(roadblocks, dtype=bool)
        )
        rates = (float(binding_rate), float(unbinding_rate), float(extrusion_rate))
        capture = float(capture_distance)
        stiffness = float(spring_stiffness)
        rest = float(spring_rest_length)
        realization = int(realization_id)
        if positions.ndim != 2 or positions.shape[0] < 2 or positions.shape[1] == 0:
            raise ValueError(
                "site_positions must have shape (at least two sites, positive dimension)."
            )
        if not np.issubdtype(positions.dtype, np.inexact):
            raise TypeError("site_positions must have an inexact dtype.")
        if not np.all(np.isfinite(positions)):
            raise ValueError("site_positions must be finite.")
        if barriers.shape != (positions.shape[0],):
            raise ValueError("roadblocks must have site-capacity shape.")
        if capacity <= 0:
            raise ValueError("relation_capacity must be positive.")
        if any(not isfinite(value) or value < 0.0 for value in rates):
            raise ValueError("Chromatin event rates must be finite and nonnegative.")
        if not isfinite(capture) or capture <= 0.0:
            raise ValueError("capture_distance must be finite and positive.")
        if not isfinite(stiffness) or stiffness < 0.0 or not isfinite(rest) or rest < 0.0:
            raise ValueError(
                "Chromatin spring parameters must be finite and nonnegative."
            )
        if realization < 0 or realization > np.iinfo(np.uint32).max:
            raise ValueError("realization_id must fit uint32.")
        relations = DynamicPairRelationPlan(
            np.zeros((positions.shape[0],), dtype=np.int32),
            capacity,
            2,
            compatibility=np.ones((1, 1, 1), dtype=bool),
            exclusion=np.ones((1, 1), dtype=bool),
            symmetric_kinds=np.ones((1,), dtype=bool),
            event_capacity=capacity,
        )
        spring = PairSpringPlan(0, 1)
        generated = canonical_fingerprint(
            {
                "kind": "chromatin-dynamics-plan",
                "sites": array_tree_fingerprint(positions),
                "roadblocks": array_tree_fingerprint(barriers),
                "relations": relations.plan_id,
                "rates": rates,
                "capture": capture,
                "stiffness": stiffness,
                "rest": rest,
                "realization": realization,
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be nonempty.")
        self.site_positions = jnp.asarray(positions)
        self.roadblocks = jnp.asarray(barriers)
        self.relations = relations
        self.spring = spring
        self.binding_rate, self.unbinding_rate, self.extrusion_rate = rates
        self.capture_distance = capture
        self.spring_stiffness = stiffness
        self.spring_rest_length = rest
        self.realization_id = realization
        self.plan_id = identifier

    def prepare(self, /) -> PreparedChromatinDynamics:
        return PreparedChromatinDynamics(self)


class ChromatinState(StrictModule):
    """Chromatin loop graph and reproducible addressed-event clock."""

    relations: PairRelationState
    time: Array
    step_index: Array


class ChromatinObservables(StrictModule):
    """Joint genomic, spatial, occupancy, and energetic loop observables."""

    occupied_sites: Array
    roadblock_occupancy: Array
    loop_genomic_span: Array
    loop_spatial_distance: Array
    loop_active: Array
    loop_count: Array
    total_genomic_span: Array
    mean_genomic_span: Array
    mean_spatial_distance: Array
    bound_fraction: Array
    spring_energy: Array


class ChromatinStepEvidence(StrictModule):
    """Relation, spring, capture, and collision evidence for one step."""

    relation: PairRelationEvidence
    springs: PairSpringEvaluation
    capture_candidate_count: Array
    collision_count: Array
    bound_count: Array
    unbound_count: Array
    extruded_count: Array
    finite: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class ChromatinStepResult(StrictModule):
    """Candidate and fail-closed accepted chromatin states."""

    candidate_state: ChromatinState
    accepted_state: ChromatinState
    evidence: ChromatinStepEvidence
    observables: ChromatinObservables
    successful: Array


class PreparedChromatinDynamics(StrictModule, NonTrainableState):
    """Prepared fixed-capacity chromatin event and conservative-force runtime."""

    plan: ChromatinDynamicsPlan
    relations: PreparedDynamicPairRelations
    springs: PreparedPairSpringEnergy
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: ChromatinDynamicsPlan, /):
        if not isinstance(plan, ChromatinDynamicsPlan):
            raise TypeError("plan must be a ChromatinDynamicsPlan.")
        relations = plan.relations.prepare(prepared_scope_id=plan.plan_id)
        springs = plan.spring.prepare(
            relations, ambient_dimension=plan.site_positions.shape[1]
        )
        self.plan = plan
        self.relations = relations
        self.springs = springs
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-chromatin-dynamics", "plan": plan.plan_id}
        )

    def initialize(
        self,
        /,
        *,
        left: ArrayLike | None = None,
        right: ArrayLike | None = None,
    ) -> ChromatinState:
        capacity = self.relations.relation_capacity
        if left is None and right is None:
            relations = self.relations.initialize()
        else:
            if left is None or right is None:
                raise ValueError("left and right must be provided together.")
            left_ = np.asarray(left, dtype=np.int32)
            right_ = np.asarray(right, dtype=np.int32)
            if left_.shape != (capacity,) or right_.shape != (capacity,):
                raise ValueError("Initial endpoints must have relation-capacity shape.")
            occupied = (left_ >= 0) & (right_ >= 0)
            canonical_left = np.where(occupied, np.minimum(left_, right_), left_)
            canonical_right = np.where(occupied, np.maximum(left_, right_), right_)
            parameters = np.zeros((capacity, 2), dtype=self.plan.site_positions.dtype)
            parameters[:, 0] = self.plan.spring_stiffness
            parameters[:, 1] = self.plan.spring_rest_length
            relations = self.relations.initialize(
                left=canonical_left,
                right=canonical_right,
                relation_kind=np.where(occupied, 0, -1),
                occupied=occupied,
                active=occupied,
                parameters=parameters,
            )
        return ChromatinState(
            relations,
            jnp.zeros((), dtype=self.plan.site_positions.dtype),
            jnp.zeros((), dtype=jnp.int32),
        )

    def bind(
        self,
        state: ChromatinState,
        left: ArrayLike,
        right: ArrayLike,
        /,
        *,
        event_id: ArrayLike = 0,
    ) -> PairRelationCommitResult:
        """Evaluate and commit one addressed two-foot diffusion capture."""

        first = jnp.asarray(left, dtype=jnp.int32)
        second = jnp.asarray(right, dtype=jnp.int32)
        left_ = jnp.minimum(first, second)
        right_ = jnp.maximum(first, second)
        site_count = self.plan.site_positions.shape[0]
        safe_left = jnp.clip(left_, 0, site_count - 1)
        safe_right = jnp.clip(right_, 0, site_count - 1)
        displacement = (
            self.plan.site_positions[safe_right] - self.plan.site_positions[safe_left]
        )
        distance = jnp.sqrt(jnp.sum(displacement * displacement))
        valid = (
            (left_ >= 0)
            & (left_ < site_count)
            & (right_ >= 0)
            & (right_ < site_count)
            & ~self.plan.roadblocks[safe_left]
            & ~self.plan.roadblocks[safe_right]
            & (distance <= self.plan.capture_distance)
        )
        return self.relations.apply(
            state.relations,
            _one(
                self.relations,
                state.time.dtype,
                event_id=event_id,
                operation=PairRelationEventKind.BIND,
                valid=valid,
                left=left_,
                right=right_,
                relation_kind=0,
                parameters=jnp.asarray(
                    [self.plan.spring_stiffness, self.plan.spring_rest_length],
                    dtype=state.time.dtype,
                ),
            ),
        )

    def unbind(
        self,
        state: ChromatinState,
        relation_id: ArrayLike,
        incarnation: ArrayLike,
        /,
        *,
        event_id: ArrayLike = 0,
    ) -> PairRelationCommitResult:
        """Commit one identity-addressed chromatin unbinding event."""

        return self.relations.apply(
            state.relations,
            _one(
                self.relations,
                state.time.dtype,
                event_id=event_id,
                operation=PairRelationEventKind.UNBIND,
                relation_id=relation_id,
                incarnation=incarnation,
            ),
        )

    def observables(self, state: ChromatinState, /) -> ChromatinObservables:
        relation = state.relations
        active = relation.active & relation.occupied
        site_count = self.plan.site_positions.shape[0]
        left = jnp.clip(relation.left, 0, site_count - 1)
        right = jnp.clip(relation.right, 0, site_count - 1)
        span = jnp.where(active, jnp.abs(relation.right - relation.left), 0)
        delta = self.plan.site_positions[right] - self.plan.site_positions[left]
        distance = jnp.where(active, jnp.sqrt(jnp.sum(delta * delta, axis=-1)), 0.0)
        occupied_sites = _occupancy(relation, site_count) > 0
        count = jnp.sum(active, dtype=jnp.int32)
        denominator = jnp.maximum(count, 1)
        return ChromatinObservables(
            occupied_sites,
            self.plan.roadblocks,
            span,
            distance,
            active,
            count,
            jnp.sum(span, dtype=self.plan.site_positions.dtype),
            jnp.sum(span, dtype=self.plan.site_positions.dtype) / denominator,
            jnp.sum(distance) / denominator,
            count.astype(self.plan.site_positions.dtype)
            / self.relations.relation_capacity,
            self.springs.energy(relation, self.plan.site_positions),
        )

    def _collisions(self, relation: PairRelationState, moving: Array) -> Array:
        site_count = self.plan.site_positions.shape[0]
        proposed_left = relation.left - 1
        proposed_right = relation.right + 1
        left = jnp.clip(proposed_left, 0, site_count - 1)
        right = jnp.clip(proposed_right, 0, site_count - 1)
        occupied = _occupancy(relation, site_count)
        owns_left = (left == relation.left) | (left == relation.right)
        owns_right = (right == relation.left) | (right == relation.right)
        blocked = (
            (proposed_left < 0)
            | (proposed_right >= site_count)
            | self.plan.roadblocks[left]
            | self.plan.roadblocks[right]
            | (occupied[left] > owns_left.astype(jnp.int32))
            | (occupied[right] > owns_right.astype(jnp.int32))
        )
        indices = jnp.arange(self.relations.relation_capacity)
        proposed_conflict = jax.vmap(
            lambda index: jnp.any(
                moving
                & (indices < index)
                & (
                    (left == left[index])
                    | (left == right[index])
                    | (right == left[index])
                    | (right == right[index])
                )
            )
        )(indices)
        return moving & (blocked | proposed_conflict)

    def _finish(
        self,
        state: ChromatinState,
        commit: PairRelationCommitResult,
        dt: Array,
        operations: Array,
        capture: Array,
        collisions: Array,
    ) -> ChromatinStepResult:
        age = self.relations.advance_age(commit.accepted_state, dt)
        candidate = ChromatinState(
            age.candidate_state, state.time + dt, state.step_index + 1
        )
        springs = self.springs.evaluate(age.accepted_state, self.plan.site_positions)
        finite = jnp.isfinite(dt) & (dt >= 0.0) & age.finite & springs.finite
        successful = commit.successful & springs.successful & finite
        accepted = jax.tree.map(
            lambda new, old: jnp.where(successful, new, old), candidate, state
        )
        applied = commit.evidence.applied
        evidence = ChromatinStepEvidence(
            commit.evidence,
            springs,
            jnp.sum(capture, dtype=jnp.int32),
            jnp.sum(collisions, dtype=jnp.int32),
            jnp.sum(
                applied & (operations == int(PairRelationEventKind.BIND)), dtype=jnp.int32
            ),
            jnp.sum(
                applied & (operations == int(PairRelationEventKind.UNBIND)),
                dtype=jnp.int32,
            ),
            jnp.sum(
                applied & (operations == int(PairRelationEventKind.MOVE)), dtype=jnp.int32
            ),
            finite,
            successful,
            self.prepared_id,
        )
        return ChromatinStepResult(
            candidate, accepted, evidence, self.observables(accepted), successful
        )

    def extrude(self, state: ChromatinState, /) -> ChromatinStepResult:
        relation = state.relations
        moving = relation.active
        collisions = self._collisions(relation, moving)
        capacity = self.relations.event_capacity
        operations = jnp.full((capacity,), int(PairRelationEventKind.MOVE))
        events = _batch(
            state.step_index * (capacity + 1) + jnp.arange(capacity),
            operations,
            moving & ~collisions,
            relation.relation_ids,
            relation.incarnations,
            relation.left - 1,
            relation.right + 1,
            relation.kind,
            relation.parameters,
        )
        return self._finish(
            state,
            self.relations.apply(relation, events),
            jnp.zeros((), dtype=state.time.dtype),
            operations,
            jnp.zeros_like(moving),
            collisions,
        )

    def step(
        self, state: ChromatinState, key: Array, dt: ArrayLike, /
    ) -> ChromatinStepResult:
        step = jnp.asarray(dt, dtype=state.time.dtype)
        if step.shape != ():
            raise ValueError("dt must be scalar.")
        relation = state.relations
        capacity = self.relations.event_capacity
        indices = jnp.arange(capacity, dtype=jnp.int32)
        unbind = relation.active & (
            _uniforms(key, self.plan.realization_id, state.step_index, 0, capacity)
            < _probability(self.plan.unbinding_rate, step)
        )
        moving = (
            relation.active
            & ~unbind
            & (
                _uniforms(key, self.plan.realization_id, state.step_index, 1, capacity)
                < _probability(self.plan.extrusion_rate, step)
            )
        )
        collisions = self._collisions(relation, moving)
        moving = moving & ~collisions
        site_count = self.plan.site_positions.shape[0]
        bind_keys = _keys(key, self.plan.realization_id, state.step_index, 2, capacity)
        first = jax.vmap(lambda local: jr.randint(local, (), 0, site_count))(bind_keys)
        second = jax.vmap(
            lambda local: jr.randint(jr.fold_in(local, 1), (), 0, site_count)
        )(bind_keys)
        left = jnp.minimum(first, second)
        right = jnp.maximum(first, second)
        distance = jnp.sqrt(
            jnp.sum(
                (self.plan.site_positions[right] - self.plan.site_positions[left]) ** 2,
                axis=-1,
            )
        )
        site_occupied = _occupancy(relation, site_count) > 0
        move_destinations = jnp.zeros((site_count,), dtype=bool)
        move_destinations = move_destinations.at[
            jnp.clip(relation.left - 1, 0, site_count - 1)
        ].max(moving)
        move_destinations = move_destinations.at[
            jnp.clip(relation.right + 1, 0, site_count - 1)
        ].max(moving)
        capture = (
            ~relation.occupied
            & (left != right)
            & ~self.plan.roadblocks[left]
            & ~self.plan.roadblocks[right]
            & ~site_occupied[left]
            & ~site_occupied[right]
            & ~move_destinations[left]
            & ~move_destinations[right]
            & (distance <= self.plan.capture_distance)
            & (
                _uniforms(key, self.plan.realization_id, state.step_index, 3, capacity)
                < _probability(self.plan.binding_rate, step)
            )
        )
        conflict = jax.vmap(
            lambda index: jnp.any(
                capture
                & (indices < index)
                & (
                    (left == left[index])
                    | (left == right[index])
                    | (right == left[index])
                    | (right == right[index])
                )
            )
        )(indices)
        bind = capture & ~conflict
        operations = jnp.where(
            unbind,
            int(PairRelationEventKind.UNBIND),
            jnp.where(
                moving, int(PairRelationEventKind.MOVE), int(PairRelationEventKind.BIND)
            ),
        )
        parameters = jnp.broadcast_to(
            jnp.asarray(
                [self.plan.spring_stiffness, self.plan.spring_rest_length],
                dtype=state.time.dtype,
            ),
            (capacity, 2),
        )
        events = _batch(
            state.step_index * (4 * capacity) + indices,
            operations,
            unbind | moving | bind,
            jnp.where(bind, -1, relation.relation_ids),
            jnp.where(bind, -1, relation.incarnations),
            jnp.where(bind, left, relation.left - 1),
            jnp.where(bind, right, relation.right + 1),
            jnp.where(bind, 0, relation.kind),
            parameters,
        )
        return self._finish(
            state,
            self.relations.apply(relation, events),
            step,
            operations,
            capture,
            collisions,
        )


class ActinNetworkPlan(StrictModule, NonTrainableState):
    """Plan actin polymerization, turnover, branching, capping, and severing."""

    relations: DynamicPairRelationPlan
    spring: PairSpringPlan
    node_capacity: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    initial_monomer_pool: float = eqx.field(static=True)
    monomer_mass: float = eqx.field(static=True)
    segment_length: float = eqx.field(static=True)
    spring_stiffness: float = eqx.field(static=True)
    polymerization_rate: float = eqx.field(static=True)
    depolymerization_rate: float = eqx.field(static=True)
    branching_rate: float = eqx.field(static=True)
    capping_rate: float = eqx.field(static=True)
    severing_rate: float = eqx.field(static=True)
    realization_id: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        node_capacity: int,
        edge_capacity: int,
        /,
        *,
        ambient_dimension: int = 3,
        initial_monomer_pool: float = 100.0,
        monomer_mass: float = 1.0,
        segment_length: float = 1.0,
        spring_stiffness: float = 10.0,
        polymerization_rate: float = 1.0,
        depolymerization_rate: float = 0.1,
        branching_rate: float = 0.1,
        capping_rate: float = 0.05,
        severing_rate: float = 0.01,
        realization_id: int = 0,
        plan_id: str | None = None,
    ):
        nodes = int(node_capacity)
        edges = int(edge_capacity)
        dimension = int(ambient_dimension)
        pool = float(initial_monomer_pool)
        mass = float(monomer_mass)
        length = float(segment_length)
        stiffness = float(spring_stiffness)
        rates = tuple(
            float(value)
            for value in (
                polymerization_rate,
                depolymerization_rate,
                branching_rate,
                capping_rate,
                severing_rate,
            )
        )
        realization = int(realization_id)
        if nodes <= 0 or edges <= 0 or dimension <= 0:
            raise ValueError("Actin capacities and dimension must be positive.")
        if not isfinite(pool) or pool < 0.0 or not isfinite(mass) or mass <= 0.0:
            raise ValueError("Actin monomer masses are invalid.")
        if (
            not isfinite(length)
            or length <= 0.0
            or not isfinite(stiffness)
            or stiffness < 0.0
        ):
            raise ValueError("Actin segment length and stiffness are invalid.")
        if any(not isfinite(value) or value < 0.0 for value in rates):
            raise ValueError("Actin event rates must be finite and nonnegative.")
        if realization < 0 or realization > np.iinfo(np.uint32).max:
            raise ValueError("realization_id must fit uint32.")
        relations = DynamicPairRelationPlan(
            np.zeros((nodes,), dtype=np.int32),
            edges,
            2,
            compatibility=np.ones((2, 1, 1), dtype=bool),
            exclusion=np.zeros((2, 2), dtype=bool),
            symmetric_kinds=np.zeros((2,), dtype=bool),
            kind_count=2,
            event_capacity=edges,
        )
        spring = PairSpringPlan(0, 1)
        generated = canonical_fingerprint(
            {
                "kind": "actin-network-plan",
                "nodes": nodes,
                "dimension": dimension,
                "relations": relations.plan_id,
                "pool": pool,
                "mass": mass,
                "length": length,
                "stiffness": stiffness,
                "rates": rates,
                "realization": realization,
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be nonempty.")
        self.relations = relations
        self.spring = spring
        self.node_capacity = nodes
        self.ambient_dimension = dimension
        self.initial_monomer_pool = pool
        self.monomer_mass = mass
        self.segment_length = length
        self.spring_stiffness = stiffness
        (
            self.polymerization_rate,
            self.depolymerization_rate,
            self.branching_rate,
            self.capping_rate,
            self.severing_rate,
        ) = rates
        self.realization_id = realization
        self.plan_id = identifier

    def prepare(self, /) -> PreparedActinNetwork:
        return PreparedActinNetwork(self)


class ActinNetworkState(StrictModule):
    """Actin nodes, edges, monomer reservoir, and stable lineage identities."""

    relations: PairRelationState
    node_position: Array
    node_active: Array
    node_mass: Array
    capped: Array
    lineage_id: Array
    next_lineage_id: Array
    monomer_pool: Array
    time: Array
    step_index: Array


class ActinNetworkEvidence(StrictModule):
    """Mass, lineage, capacity, identity, and finite-state evidence."""

    relation: PairRelationEvidence
    springs: PairSpringEvaluation
    node_overflow: Array
    invalid_request: Array
    stale_identity: Array
    mass_before: Array
    mass_after: Array
    mass_residual: Array
    lineage_valid: Array
    finite: Array
    successful: Array
    event_code: Array
    prepared_id: str = eqx.field(static=True)


class ActinStepResult(StrictModule):
    """Candidate and fail-closed accepted actin network states."""

    candidate_state: ActinNetworkState
    accepted_state: ActinNetworkState
    evidence: ActinNetworkEvidence
    successful: Array


class PreparedActinNetwork(StrictModule, NonTrainableState):
    """Prepared fixed-shape actin node-edge transaction runtime."""

    plan: ActinNetworkPlan
    relations: PreparedDynamicPairRelations
    springs: PreparedPairSpringEnergy
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: ActinNetworkPlan, /):
        if not isinstance(plan, ActinNetworkPlan):
            raise TypeError("plan must be an ActinNetworkPlan.")
        relations = plan.relations.prepare(prepared_scope_id=plan.plan_id)
        self.plan = plan
        self.relations = relations
        self.springs = plan.spring.prepare(
            relations, ambient_dimension=plan.ambient_dimension
        )
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-actin-network", "plan": plan.plan_id}
        )

    def initialize(self, seed_positions: ArrayLike, /) -> ActinNetworkState:
        seeds = np.asarray(seed_positions)
        if (
            seeds.ndim != 2
            or seeds.shape[0] == 0
            or seeds.shape[0] > self.plan.node_capacity
            or seeds.shape[1] != self.plan.ambient_dimension
        ):
            raise ValueError("seed_positions must fit the prepared node shape.")
        if not np.issubdtype(seeds.dtype, np.inexact):
            raise TypeError("seed_positions must have an inexact dtype.")
        if not np.all(np.isfinite(seeds)):
            raise ValueError("seed_positions must be finite.")
        count = seeds.shape[0]
        seed_mass = count * self.plan.monomer_mass
        if self.plan.initial_monomer_pool < seed_mass:
            raise ValueError("initial_monomer_pool cannot supply the seed mass.")
        positions = np.zeros(
            (self.plan.node_capacity, self.plan.ambient_dimension), dtype=seeds.dtype
        )
        positions[:count] = seeds
        active = np.arange(self.plan.node_capacity) < count
        return ActinNetworkState(
            self.relations.initialize(),
            jnp.asarray(positions),
            jnp.asarray(active),
            jnp.asarray(np.where(active, self.plan.monomer_mass, 0.0), dtype=seeds.dtype),
            jnp.zeros((self.plan.node_capacity,), dtype=bool),
            jnp.asarray(
                np.where(active, np.arange(self.plan.node_capacity), -1),
                dtype=jnp.int32,
            ),
            jnp.asarray(count, dtype=jnp.int32),
            jnp.asarray(self.plan.initial_monomer_pool - seed_mass, dtype=seeds.dtype),
            jnp.zeros((), dtype=seeds.dtype),
            jnp.zeros((), dtype=jnp.int32),
        )

    def total_mass(self, state: ActinNetworkState, /) -> Array:
        return state.monomer_pool + jnp.sum(
            jnp.where(state.node_active, state.node_mass, 0.0)
        )

    def _lineage_valid(self, state: ActinNetworkState) -> Array:
        relation = state.relations
        left = jnp.clip(relation.left, 0, self.plan.node_capacity - 1)
        right = jnp.clip(relation.right, 0, self.plan.node_capacity - 1)
        incoming = (
            jnp.zeros((self.plan.node_capacity,), dtype=jnp.int32)
            .at[right]
            .add(relation.active.astype(jnp.int32))
        )
        return (
            jnp.all(
                ~relation.active | (state.node_active[left] & state.node_active[right])
            )
            & jnp.all(incoming <= 1)
            & jnp.all(
                jnp.where(
                    state.node_active, state.lineage_id >= 0, state.lineage_id == -1
                )
            )
        )

    def _finish(
        self,
        state: ActinNetworkState,
        candidate: ActinNetworkState,
        commit: PairRelationCommitResult,
        valid: Array,
        overflow: Array,
        invalid: Array,
        event_code: int,
    ) -> ActinStepResult:
        mass_before = self.total_mass(state)
        mass_after = self.total_mass(candidate)
        residual = mass_after - mass_before
        tolerance = (
            32.0
            * jnp.finfo(candidate.node_mass.dtype).eps
            * jnp.maximum(jnp.abs(mass_before), 1.0)
        )
        lineage_valid = self._lineage_valid(candidate)
        springs = self.springs.evaluate(candidate.relations, candidate.node_position)
        finite = (
            jnp.all(jnp.isfinite(candidate.node_position))
            & jnp.all(jnp.isfinite(candidate.node_mass))
            & jnp.isfinite(candidate.monomer_pool)
            & springs.finite
        )
        successful = (
            valid
            & commit.successful
            & ~overflow
            & ~invalid
            & (jnp.abs(residual) <= tolerance)
            & lineage_valid
            & springs.successful
            & finite
        )
        accepted = jax.tree.map(
            lambda new, old: jnp.where(successful, new, old), candidate, state
        )
        evidence = ActinNetworkEvidence(
            commit.evidence,
            springs,
            overflow,
            invalid,
            commit.evidence.stale_identity_count > 0,
            mass_before,
            mass_after,
            residual,
            lineage_valid,
            finite,
            successful,
            jnp.asarray(event_code, dtype=jnp.int32),
            self.prepared_id,
        )
        return ActinStepResult(candidate, accepted, evidence, successful)

    def _idle(self, state: ActinNetworkState) -> ActinStepResult:
        commit = self.relations.apply(
            state.relations, _empty(self.relations, state.node_position.dtype)
        )
        return self._finish(
            state,
            state,
            commit,
            jnp.asarray(True),
            jnp.asarray(False),
            jnp.asarray(False),
            0,
        )

    def _grow(
        self,
        state: ActinNetworkState,
        parent: ArrayLike,
        direction: ArrayLike,
        *,
        event_id: ArrayLike,
        branch: bool,
    ) -> ActinStepResult:
        parent_ = jnp.asarray(parent, dtype=jnp.int32)
        vector = jnp.asarray(direction, dtype=state.node_position.dtype)
        if vector.shape != (self.plan.ambient_dimension,):
            raise ValueError("direction must have ambient-dimension shape.")
        free = ~state.node_active
        available = jnp.any(free)
        child = jnp.argmax(free.astype(jnp.int32))
        safe_parent = jnp.clip(parent_, 0, self.plan.node_capacity - 1)
        norm = jnp.sqrt(jnp.sum(vector * vector))
        finite_direction = jnp.all(jnp.isfinite(vector)) & (norm > 0.0)
        valid = (
            available
            & (parent_ >= 0)
            & (parent_ < self.plan.node_capacity)
            & state.node_active[safe_parent]
            & (branch | ~state.capped[safe_parent])
            & (state.monomer_pool >= self.plan.monomer_mass)
            & finite_direction
        )
        endpoint_active = state.node_active.at[child].set(valid)
        events = _one(
            self.relations,
            state.node_position.dtype,
            event_id=event_id,
            operation=PairRelationEventKind.BIND,
            valid=valid,
            left=parent_,
            right=child,
            relation_kind=1 if branch else 0,
            parameters=jnp.asarray(
                [self.plan.spring_stiffness, self.plan.segment_length],
                dtype=state.node_position.dtype,
            ),
        )
        commit = self.relations.apply(
            state.relations, events, endpoint_active=endpoint_active
        )
        use = valid & commit.successful
        unit = vector / jnp.where(norm > 0.0, norm, 1.0)
        candidate = ActinNetworkState(
            commit.accepted_state,
            state.node_position.at[child].set(
                jnp.where(
                    use,
                    state.node_position[safe_parent] + self.plan.segment_length * unit,
                    state.node_position[child],
                )
            ),
            state.node_active.at[child].set(use | state.node_active[child]),
            state.node_mass.at[child].set(
                jnp.where(use, self.plan.monomer_mass, state.node_mass[child])
            ),
            state.capped.at[child].set(jnp.where(use, False, state.capped[child])),
            state.lineage_id.at[child].set(
                jnp.where(use, state.lineage_id[safe_parent], state.lineage_id[child])
            ),
            state.next_lineage_id,
            state.monomer_pool - jnp.where(use, self.plan.monomer_mass, 0.0),
            state.time,
            state.step_index,
        )
        return self._finish(
            state,
            candidate,
            commit,
            valid,
            ~available,
            ~valid & available,
            2 if branch else 1,
        )

    def polymerize(
        self,
        state: ActinNetworkState,
        parent: ArrayLike,
        direction: ArrayLike,
        /,
        *,
        event_id: ArrayLike = 0,
    ) -> ActinStepResult:
        """Append an uncapped filament segment while conserving monomer mass."""

        return self._grow(state, parent, direction, event_id=event_id, branch=False)

    def branch(
        self,
        state: ActinNetworkState,
        parent: ArrayLike,
        direction: ArrayLike,
        /,
        *,
        event_id: ArrayLike = 0,
    ) -> ActinStepResult:
        """Append a daughter branch carrying the parent lineage identity."""

        return self._grow(state, parent, direction, event_id=event_id, branch=True)

    def depolymerize(
        self,
        state: ActinNetworkState,
        node: ArrayLike,
        /,
        *,
        event_id: ArrayLike = 0,
    ) -> ActinStepResult:
        """Remove one terminal non-seed node and return its mass to solution."""

        node_ = jnp.asarray(node, dtype=jnp.int32)
        safe_node = jnp.clip(node_, 0, self.plan.node_capacity - 1)
        relation = state.relations
        incoming = relation.active & (relation.right == node_)
        outgoing = relation.active & (relation.left == node_)
        slot = jnp.argmax(incoming.astype(jnp.int32))
        valid = (
            (node_ >= 0)
            & (node_ < self.plan.node_capacity)
            & state.node_active[safe_node]
            & (jnp.sum(incoming, dtype=jnp.int32) == 1)
            & ~jnp.any(outgoing)
        )
        commit = self.relations.apply(
            relation,
            _one(
                self.relations,
                state.node_position.dtype,
                event_id=event_id,
                operation=PairRelationEventKind.UNBIND,
                valid=valid,
                relation_id=relation.relation_ids[slot],
                incarnation=relation.incarnations[slot],
            ),
        )
        use = valid & commit.successful
        released = jnp.where(use, state.node_mass[safe_node], 0.0)
        candidate = ActinNetworkState(
            commit.accepted_state,
            state.node_position.at[safe_node].set(
                jnp.where(
                    use,
                    jnp.zeros_like(state.node_position[safe_node]),
                    state.node_position[safe_node],
                )
            ),
            state.node_active.at[safe_node].set(
                jnp.where(use, False, state.node_active[safe_node])
            ),
            state.node_mass.at[safe_node].set(
                jnp.where(use, 0.0, state.node_mass[safe_node])
            ),
            state.capped.at[safe_node].set(
                jnp.where(use, False, state.capped[safe_node])
            ),
            state.lineage_id.at[safe_node].set(
                jnp.where(use, -1, state.lineage_id[safe_node])
            ),
            state.next_lineage_id,
            state.monomer_pool + released,
            state.time,
            state.step_index,
        )
        return self._finish(
            state, candidate, commit, valid, jnp.asarray(False), ~valid, 3
        )

    def set_capped(
        self, state: ActinNetworkState, node: ArrayLike, capped: ArrayLike, /
    ) -> ActinStepResult:
        """Atomically set the capping state of one active node."""

        node_ = jnp.asarray(node, dtype=jnp.int32)
        safe_node = jnp.clip(node_, 0, self.plan.node_capacity - 1)
        valid = (
            (node_ >= 0)
            & (node_ < self.plan.node_capacity)
            & state.node_active[safe_node]
        )
        commit = self.relations.apply(
            state.relations, _empty(self.relations, state.node_position.dtype)
        )
        candidate = eqx.tree_at(
            lambda value: value.capped,
            state,
            state.capped.at[safe_node].set(
                jnp.where(valid, jnp.asarray(capped, dtype=bool), state.capped[safe_node])
            ),
        )
        return self._finish(
            state, candidate, commit, valid, jnp.asarray(False), ~valid, 4
        )

    def sever(
        self,
        state: ActinNetworkState,
        relation_id: ArrayLike,
        relation_incarnation: ArrayLike,
        /,
        *,
        event_id: ArrayLike = 0,
    ) -> ActinStepResult:
        """Sever an addressed edge and reidentify the daughter subtree lineage."""

        relation = state.relations
        target = jnp.asarray(relation_id, dtype=jnp.int32)
        slot = jnp.argmax((relation.relation_ids == target).astype(jnp.int32))
        commit = self.relations.apply(
            relation,
            _one(
                self.relations,
                state.node_position.dtype,
                event_id=event_id,
                operation=PairRelationEventKind.UNBIND,
                relation_id=target,
                incarnation=relation_incarnation,
            ),
        )
        child = jnp.clip(relation.right[slot], 0, self.plan.node_capacity - 1)
        descendants = jnp.arange(self.plan.node_capacity) == child

        def closure(_: int, marked: Array) -> Array:
            left = jnp.clip(relation.left, 0, self.plan.node_capacity - 1)
            right = jnp.clip(relation.right, 0, self.plan.node_capacity - 1)
            reached = relation.active & (relation.relation_ids != target) & marked[left]
            return marked.at[right].max(reached)

        descendants = jax.lax.fori_loop(
            0, self.relations.relation_capacity, closure, descendants
        )
        overflow = state.next_lineage_id == jnp.iinfo(jnp.int32).max
        use = commit.successful & ~overflow
        candidate = ActinNetworkState(
            jax.tree.map(
                lambda new, old: jnp.where(use, new, old),
                commit.accepted_state,
                relation,
            ),
            state.node_position,
            state.node_active,
            state.node_mass,
            state.capped,
            jnp.where(
                use & descendants & state.node_active,
                state.next_lineage_id,
                state.lineage_id,
            ),
            state.next_lineage_id + use.astype(jnp.int32),
            state.monomer_pool,
            state.time,
            state.step_index,
        )
        return self._finish(
            state, candidate, commit, use, jnp.asarray(False), overflow, 5
        )

    def step(
        self, state: ActinNetworkState, key: Array, dt: ArrayLike, /
    ) -> ActinStepResult:
        """Execute one addressed event by priority and advance the event clock."""

        step = jnp.asarray(dt, dtype=state.time.dtype)
        if step.shape != ():
            raise ValueError("dt must be scalar.")
        relation = state.relations
        left = jnp.clip(relation.left, 0, self.plan.node_capacity - 1)
        right = jnp.clip(relation.right, 0, self.plan.node_capacity - 1)
        incoming = (
            jnp.zeros((self.plan.node_capacity,), dtype=jnp.int32)
            .at[right]
            .add(relation.active.astype(jnp.int32))
        )
        outgoing = (
            jnp.zeros((self.plan.node_capacity,), dtype=jnp.int32)
            .at[left]
            .add(relation.active.astype(jnp.int32))
        )
        tips = state.node_active & (outgoing == 0)
        free = jnp.any(~state.node_active)
        supply = state.monomer_pool >= self.plan.monomer_mass
        polymer = (
            tips
            & ~state.capped
            & free
            & supply
            & (
                _uniforms(
                    key,
                    self.plan.realization_id,
                    state.step_index,
                    10,
                    self.plan.node_capacity,
                )
                < _probability(self.plan.polymerization_rate, step)
            )
        )
        branch = (
            state.node_active
            & free
            & supply
            & (
                _uniforms(
                    key,
                    self.plan.realization_id,
                    state.step_index,
                    11,
                    self.plan.node_capacity,
                )
                < _probability(self.plan.branching_rate, step)
            )
        )
        cap = (
            tips
            & ~state.capped
            & (
                _uniforms(
                    key,
                    self.plan.realization_id,
                    state.step_index,
                    12,
                    self.plan.node_capacity,
                )
                < _probability(self.plan.capping_rate, step)
            )
        )
        depoly = (
            tips
            & (incoming == 1)
            & (
                _uniforms(
                    key,
                    self.plan.realization_id,
                    state.step_index,
                    13,
                    self.plan.node_capacity,
                )
                < _probability(self.plan.depolymerization_rate, step)
            )
        )
        sever = relation.active & (
            _uniforms(
                key,
                self.plan.realization_id,
                state.step_index,
                14,
                self.relations.relation_capacity,
            )
            < _probability(self.plan.severing_rate, step)
        )
        polymer_node = jnp.argmax(polymer.astype(jnp.int32))
        branch_node = jnp.argmax(branch.astype(jnp.int32))
        cap_node = jnp.argmax(cap.astype(jnp.int32))
        depoly_node = jnp.argmax(depoly.astype(jnp.int32))
        sever_slot = jnp.argmax(sever.astype(jnp.int32))
        directions = jax.vmap(
            lambda local: jr.normal(
                local,
                (self.plan.ambient_dimension,),
                dtype=state.node_position.dtype,
            )
        )(
            _keys(
                key,
                self.plan.realization_id,
                state.step_index,
                15,
                self.plan.node_capacity,
            )
        )
        base = state.step_index * 16
        result = jax.lax.cond(
            jnp.any(sever),
            lambda _: self.sever(
                state,
                relation.relation_ids[sever_slot],
                relation.incarnations[sever_slot],
                event_id=base + 5,
            ),
            lambda _: jax.lax.cond(
                jnp.any(depoly),
                lambda __: self.depolymerize(state, depoly_node, event_id=base + 3),
                lambda __: jax.lax.cond(
                    jnp.any(cap),
                    lambda ___: self.set_capped(state, cap_node, True),
                    lambda ___: jax.lax.cond(
                        jnp.any(branch),
                        lambda ____: self.branch(
                            state,
                            branch_node,
                            directions[branch_node],
                            event_id=base + 2,
                        ),
                        lambda ____: jax.lax.cond(
                            jnp.any(polymer),
                            lambda _____: self.polymerize(
                                state,
                                polymer_node,
                                directions[polymer_node],
                                event_id=base + 1,
                            ),
                            lambda _____: self._idle(state),
                            operand=None,
                        ),
                        operand=None,
                    ),
                    operand=None,
                ),
                operand=None,
            ),
            operand=None,
        )
        finite = jnp.isfinite(step) & (step > 0.0)
        successful = result.successful & finite
        candidate = eqx.tree_at(
            lambda value: (value.time, value.step_index),
            result.accepted_state,
            (result.accepted_state.time + step, result.accepted_state.step_index + 1),
        )
        accepted = jax.tree.map(
            lambda new, old: jnp.where(successful, new, old), candidate, state
        )
        evidence = eqx.tree_at(
            lambda value: (value.finite, value.successful),
            result.evidence,
            (result.evidence.finite & finite, successful),
        )
        return ActinStepResult(candidate, accepted, evidence, successful)


class MotorCrosslinkerPlan(StrictModule, NonTrainableState):
    """Plan passive crosslinks and directed load-stalling motor relations."""

    relations: DynamicPairRelationPlan
    spring: PairSpringPlan
    endpoint_capacity: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    spring_stiffness: float = eqx.field(static=True)
    rest_length: float = eqx.field(static=True)
    stepping_rate: float = eqx.field(static=True)
    stall_force: float = eqx.field(static=True)
    realization_id: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        endpoint_capacity: int,
        relation_capacity: int,
        /,
        *,
        ambient_dimension: int = 3,
        spring_stiffness: float = 1.0,
        rest_length: float = 0.0,
        stepping_rate: float = 1.0,
        stall_force: float = 10.0,
        realization_id: int = 0,
        plan_id: str | None = None,
    ):
        endpoints = int(endpoint_capacity)
        capacity = int(relation_capacity)
        dimension = int(ambient_dimension)
        stiffness = float(spring_stiffness)
        rest = float(rest_length)
        rate = float(stepping_rate)
        stall = float(stall_force)
        realization = int(realization_id)
        if endpoints < 2 or capacity <= 0 or dimension <= 0:
            raise ValueError("Motor/crosslinker capacities and dimension are invalid.")
        if (
            not isfinite(stiffness)
            or stiffness < 0.0
            or not isfinite(rest)
            or rest < 0.0
            or not isfinite(rate)
            or rate < 0.0
            or not isfinite(stall)
            or stall <= 0.0
        ):
            raise ValueError("Motor/crosslinker physical parameters are invalid.")
        if realization < 0 or realization > np.iinfo(np.uint32).max:
            raise ValueError("realization_id must fit uint32.")
        relations = DynamicPairRelationPlan(
            np.zeros((endpoints,), dtype=np.int32),
            capacity,
            2,
            compatibility=np.ones((2, 1, 1), dtype=bool),
            exclusion=np.zeros((2, 2), dtype=bool),
            symmetric_kinds=np.asarray([True, False]),
            kind_count=2,
            event_capacity=capacity,
        )
        generated = canonical_fingerprint(
            {
                "kind": "motor-crosslinker-plan",
                "endpoints": endpoints,
                "dimension": dimension,
                "relations": relations.plan_id,
                "stiffness": stiffness,
                "rest": rest,
                "rate": rate,
                "stall": stall,
                "realization": realization,
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be nonempty.")
        self.relations = relations
        self.spring = PairSpringPlan(0, 1)
        self.endpoint_capacity = endpoints
        self.ambient_dimension = dimension
        self.spring_stiffness = stiffness
        self.rest_length = rest
        self.stepping_rate = rate
        self.stall_force = stall
        self.realization_id = realization
        self.plan_id = identifier

    def prepare(self, /) -> PreparedMotorCrosslinkers:
        return PreparedMotorCrosslinkers(self)


class MotorCrosslinkerState(StrictModule):
    """Passive and motor relation state with an addressed-event clock."""

    relations: PairRelationState
    time: Array
    step_index: Array


class MotorCrosslinkerEvidence(StrictModule):
    """Motor stepping, stalling, endpoint, relation, and spring evidence."""

    relation: PairRelationEvidence
    springs: PairSpringEvaluation
    stepped_count: Array
    stalled_count: Array
    endpoint_blocked_count: Array
    finite: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class MotorCrosslinkerStepResult(StrictModule):
    """Candidate and fail-closed accepted motor/crosslinker states."""

    candidate_state: MotorCrosslinkerState
    accepted_state: MotorCrosslinkerState
    evidence: MotorCrosslinkerEvidence
    successful: Array


class PreparedMotorCrosslinkers(StrictModule, NonTrainableState):
    """Prepared crosslinker/motor binding and endpoint-stepping runtime."""

    plan: MotorCrosslinkerPlan
    relations: PreparedDynamicPairRelations
    springs: PreparedPairSpringEnergy
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: MotorCrosslinkerPlan, /):
        if not isinstance(plan, MotorCrosslinkerPlan):
            raise TypeError("plan must be a MotorCrosslinkerPlan.")
        relations = plan.relations.prepare(prepared_scope_id=plan.plan_id)
        self.plan = plan
        self.relations = relations
        self.springs = plan.spring.prepare(
            relations, ambient_dimension=plan.ambient_dimension
        )
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-motor-crosslinkers", "plan": plan.plan_id}
        )

    def initialize(
        self,
        dtype: np.dtype | type = float,
        /,
        *,
        left: ArrayLike | None = None,
        right: ArrayLike | None = None,
        motor: ArrayLike | None = None,
    ) -> MotorCrosslinkerState:
        capacity = self.relations.relation_capacity
        if left is None and right is None:
            relations = self.relations.initialize(
                parameters=np.zeros((capacity, 2), dtype=dtype)
            )
        else:
            if left is None or right is None:
                raise ValueError("left and right must be provided together.")
            left_ = np.asarray(left, dtype=np.int32)
            right_ = np.asarray(right, dtype=np.int32)
            if left_.shape != (capacity,) or right_.shape != (capacity,):
                raise ValueError("Initial endpoints must have relation-capacity shape.")
            occupied = (left_ >= 0) & (right_ >= 0)
            motor_ = (
                np.zeros((capacity,), dtype=bool)
                if motor is None
                else np.asarray(motor, dtype=bool)
            )
            if motor_.shape != (capacity,):
                raise ValueError("motor must have relation-capacity shape.")
            parameters = np.zeros((capacity, 2), dtype=dtype)
            parameters[:, 0] = self.plan.spring_stiffness
            parameters[:, 1] = self.plan.rest_length
            relations = self.relations.initialize(
                left=left_,
                right=right_,
                relation_kind=np.where(occupied, motor_.astype(np.int32), -1),
                occupied=occupied,
                active=occupied,
                parameters=parameters,
            )
        return MotorCrosslinkerState(
            relations, jnp.zeros((), dtype=dtype), jnp.zeros((), dtype=jnp.int32)
        )

    def bind(
        self,
        state: MotorCrosslinkerState,
        left: ArrayLike,
        right: ArrayLike,
        /,
        *,
        motor: bool,
        event_id: ArrayLike = 0,
    ) -> PairRelationCommitResult:
        return self.relations.apply(
            state.relations,
            _one(
                self.relations,
                state.time.dtype,
                event_id=event_id,
                operation=PairRelationEventKind.BIND,
                left=left,
                right=right,
                relation_kind=1 if motor else 0,
                parameters=jnp.asarray(
                    [self.plan.spring_stiffness, self.plan.rest_length],
                    dtype=state.time.dtype,
                ),
            ),
        )

    def unbind(
        self,
        state: MotorCrosslinkerState,
        relation_id: ArrayLike,
        incarnation: ArrayLike,
        /,
        *,
        event_id: ArrayLike = 0,
    ) -> PairRelationCommitResult:
        return self.relations.apply(
            state.relations,
            _one(
                self.relations,
                state.time.dtype,
                event_id=event_id,
                operation=PairRelationEventKind.UNBIND,
                relation_id=relation_id,
                incarnation=incarnation,
            ),
        )

    def step(
        self,
        state: MotorCrosslinkerState,
        key: Array,
        dt: ArrayLike,
        positions: ArrayLike,
        successor: ArrayLike,
        /,
    ) -> MotorCrosslinkerStepResult:
        """Step motor right endpoints along a prepared-size successor table."""

        coordinates = jnp.asarray(positions)
        next_endpoint = jnp.asarray(successor, dtype=jnp.int32)
        if coordinates.shape != (
            self.plan.endpoint_capacity,
            self.plan.ambient_dimension,
        ):
            raise ValueError("positions has the wrong prepared shape.")
        if next_endpoint.shape != (self.plan.endpoint_capacity,):
            raise ValueError("successor must have endpoint-capacity shape.")
        dt_ = jnp.asarray(dt, dtype=state.time.dtype)
        if dt_.shape != ():
            raise ValueError("dt must be scalar.")
        relation = state.relations
        capacity = self.relations.relation_capacity
        left = jnp.clip(relation.left, 0, self.plan.endpoint_capacity - 1)
        right = jnp.clip(relation.right, 0, self.plan.endpoint_capacity - 1)
        distance = jnp.sqrt(
            jnp.sum((coordinates[right] - coordinates[left]) ** 2, axis=-1)
        )
        load = relation.parameters[:, 0] * jnp.abs(distance - relation.parameters[:, 1])
        motor = relation.active & (relation.kind == 1)
        stalled = motor & (load >= self.plan.stall_force)
        proposed = next_endpoint[right]
        blocked = motor & (
            (proposed < 0)
            | (proposed >= self.plan.endpoint_capacity)
            | (proposed == relation.left)
        )
        moving = (
            motor
            & ~stalled
            & ~blocked
            & (
                _uniforms(
                    key,
                    self.plan.realization_id,
                    state.step_index,
                    20,
                    capacity,
                )
                < _probability(self.plan.stepping_rate, dt_)
            )
        )
        proposed_safe = jnp.clip(proposed, 0, self.plan.endpoint_capacity - 1)
        indices = jnp.arange(capacity)
        converged_duplicate = jax.vmap(
            lambda index: jnp.any(
                moving
                & (indices < index)
                & (relation.left == relation.left[index])
                & (proposed_safe == proposed_safe[index])
            )
        )(indices)
        moving = moving & ~converged_duplicate
        events = _batch(
            state.step_index * (capacity + 1) + indices,
            jnp.full((capacity,), int(PairRelationEventKind.MOVE)),
            moving,
            relation.relation_ids,
            relation.incarnations,
            relation.left,
            proposed,
            relation.kind,
            relation.parameters,
        )
        commit = self.relations.apply(relation, events)
        age = self.relations.advance_age(commit.accepted_state, dt_)
        springs = self.springs.evaluate(age.accepted_state, coordinates)
        finite = jnp.isfinite(dt_) & (dt_ > 0.0) & age.finite & springs.finite
        successful = commit.successful & springs.successful & finite
        candidate = MotorCrosslinkerState(
            age.candidate_state, state.time + dt_, state.step_index + 1
        )
        accepted = jax.tree.map(
            lambda new, old: jnp.where(successful, new, old), candidate, state
        )
        evidence = MotorCrosslinkerEvidence(
            commit.evidence,
            springs,
            jnp.sum(commit.evidence.applied, dtype=jnp.int32),
            jnp.sum(stalled, dtype=jnp.int32),
            jnp.sum(blocked, dtype=jnp.int32),
            finite,
            successful,
            self.prepared_id,
        )
        return MotorCrosslinkerStepResult(candidate, accepted, evidence, successful)


class FocalAdhesionPlan(StrictModule, NonTrainableState):
    """Plan cell-to-substrate adhesion turnover and conservative traction."""

    relations: DynamicPairRelationPlan
    spring: PairSpringPlan
    cell_endpoint_count: int = eqx.field(static=True)
    substrate_endpoint_count: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    binding_rate: float = eqx.field(static=True)
    unbinding_rate: float = eqx.field(static=True)
    capture_distance: float = eqx.field(static=True)
    spring_stiffness: float = eqx.field(static=True)
    rest_length: float = eqx.field(static=True)
    force_scale: float = eqx.field(static=True)
    realization_id: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_endpoint_count: int,
        substrate_endpoint_count: int,
        relation_capacity: int,
        /,
        *,
        ambient_dimension: int = 3,
        binding_rate: float = 1.0,
        unbinding_rate: float = 0.1,
        capture_distance: float = 1.0,
        spring_stiffness: float = 1.0,
        rest_length: float = 0.0,
        force_scale: float = 1.0,
        realization_id: int = 0,
        plan_id: str | None = None,
    ):
        cells = int(cell_endpoint_count)
        substrate = int(substrate_endpoint_count)
        capacity = int(relation_capacity)
        dimension = int(ambient_dimension)
        bind_rate = float(binding_rate)
        unbind_rate = float(unbinding_rate)
        capture = float(capture_distance)
        stiffness = float(spring_stiffness)
        rest = float(rest_length)
        force = float(force_scale)
        realization = int(realization_id)
        if cells <= 0 or substrate <= 0 or capacity <= 0 or dimension <= 0:
            raise ValueError("Focal-adhesion capacities and dimension must be positive.")
        if any(not isfinite(value) or value < 0.0 for value in (bind_rate, unbind_rate)):
            raise ValueError("Focal-adhesion rates must be finite and nonnegative.")
        if (
            not isfinite(capture)
            or capture <= 0.0
            or not isfinite(stiffness)
            or stiffness < 0.0
            or not isfinite(rest)
            or rest < 0.0
            or not isfinite(force)
            or force <= 0.0
        ):
            raise ValueError("Focal-adhesion physical controls are invalid.")
        if realization < 0 or realization > np.iinfo(np.uint32).max:
            raise ValueError("realization_id must fit uint32.")
        endpoint_types = np.concatenate(
            (np.zeros((cells,), dtype=np.int32), np.ones((substrate,), dtype=np.int32))
        )
        compatibility = np.zeros((1, 2, 2), dtype=bool)
        compatibility[0, 0, 1] = True
        relations = DynamicPairRelationPlan(
            endpoint_types,
            capacity,
            2,
            compatibility=compatibility,
            exclusion=np.ones((1, 1), dtype=bool),
            event_capacity=capacity,
        )
        generated = canonical_fingerprint(
            {
                "kind": "focal-adhesion-plan",
                "cells": cells,
                "substrate": substrate,
                "dimension": dimension,
                "relations": relations.plan_id,
                "rates": (bind_rate, unbind_rate),
                "capture": capture,
                "stiffness": stiffness,
                "rest": rest,
                "force_scale": force,
                "realization": realization,
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be nonempty.")
        self.relations = relations
        self.spring = PairSpringPlan(0, 1)
        self.cell_endpoint_count = cells
        self.substrate_endpoint_count = substrate
        self.ambient_dimension = dimension
        self.binding_rate = bind_rate
        self.unbinding_rate = unbind_rate
        self.capture_distance = capture
        self.spring_stiffness = stiffness
        self.rest_length = rest
        self.force_scale = force
        self.realization_id = realization
        self.plan_id = identifier

    def prepare(self, /) -> PreparedFocalAdhesions:
        return PreparedFocalAdhesions(self)


class FocalAdhesionState(StrictModule):
    """Focal-adhesion relation state with an addressed-event clock."""

    relations: PairRelationState
    time: Array
    step_index: Array


class FocalAdhesionEvidence(StrictModule):
    """Adhesion turnover, traction, relation, and spring evidence."""

    relation: PairRelationEvidence
    springs: PairSpringEvaluation
    cell_traction: Array
    total_traction: Array
    traction_norm: Array
    bound_count: Array
    unbound_count: Array
    finite: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class FocalAdhesionStepResult(StrictModule):
    """Candidate and fail-closed accepted focal-adhesion states."""

    candidate_state: FocalAdhesionState
    accepted_state: FocalAdhesionState
    evidence: FocalAdhesionEvidence
    successful: Array


class PreparedFocalAdhesions(StrictModule, NonTrainableState):
    """Prepared focal-adhesion turnover and energy-derived traction runtime."""

    plan: FocalAdhesionPlan
    relations: PreparedDynamicPairRelations
    springs: PreparedPairSpringEnergy
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: FocalAdhesionPlan, /):
        if not isinstance(plan, FocalAdhesionPlan):
            raise TypeError("plan must be a FocalAdhesionPlan.")
        relations = plan.relations.prepare(prepared_scope_id=plan.plan_id)
        self.plan = plan
        self.relations = relations
        self.springs = plan.spring.prepare(
            relations, ambient_dimension=plan.ambient_dimension
        )
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-focal-adhesions", "plan": plan.plan_id}
        )

    @property
    def endpoint_capacity(self) -> int:
        return self.plan.cell_endpoint_count + self.plan.substrate_endpoint_count

    def initialize(
        self,
        dtype: np.dtype | type = float,
        /,
        *,
        cell_endpoints: ArrayLike | None = None,
        substrate_endpoints: ArrayLike | None = None,
    ) -> FocalAdhesionState:
        capacity = self.relations.relation_capacity
        if cell_endpoints is None and substrate_endpoints is None:
            relations = self.relations.initialize(
                parameters=np.zeros((capacity, 2), dtype=dtype)
            )
        else:
            if cell_endpoints is None or substrate_endpoints is None:
                raise ValueError(
                    "cell_endpoints and substrate_endpoints must be provided together."
                )
            cell = np.asarray(cell_endpoints, dtype=np.int32)
            substrate = np.asarray(substrate_endpoints, dtype=np.int32)
            if cell.shape != (capacity,) or substrate.shape != (capacity,):
                raise ValueError("Initial endpoints must have relation-capacity shape.")
            occupied = (cell >= 0) & (substrate >= 0)
            if np.any(occupied & (substrate >= self.plan.substrate_endpoint_count)):
                raise ValueError("Initial substrate endpoint is out of range.")
            right = np.where(occupied, substrate + self.plan.cell_endpoint_count, -1)
            parameters = np.zeros((capacity, 2), dtype=dtype)
            parameters[:, 0] = self.plan.spring_stiffness
            parameters[:, 1] = self.plan.rest_length
            relations = self.relations.initialize(
                left=cell,
                right=right,
                relation_kind=np.where(occupied, 0, -1),
                occupied=occupied,
                active=occupied,
                parameters=parameters,
            )
        return FocalAdhesionState(
            relations, jnp.zeros((), dtype=dtype), jnp.zeros((), dtype=jnp.int32)
        )

    def bind(
        self,
        state: FocalAdhesionState,
        cell_endpoint: ArrayLike,
        substrate_endpoint: ArrayLike,
        /,
        *,
        event_id: ArrayLike = 0,
    ) -> PairRelationCommitResult:
        substrate = jnp.asarray(substrate_endpoint, dtype=jnp.int32)
        valid = (substrate >= 0) & (substrate < self.plan.substrate_endpoint_count)
        return self.relations.apply(
            state.relations,
            _one(
                self.relations,
                state.time.dtype,
                event_id=event_id,
                operation=PairRelationEventKind.BIND,
                valid=valid,
                left=cell_endpoint,
                right=substrate + self.plan.cell_endpoint_count,
                relation_kind=0,
                parameters=jnp.asarray(
                    [self.plan.spring_stiffness, self.plan.rest_length],
                    dtype=state.time.dtype,
                ),
            ),
        )

    def unbind(
        self,
        state: FocalAdhesionState,
        relation_id: ArrayLike,
        incarnation: ArrayLike,
        /,
        *,
        event_id: ArrayLike = 0,
    ) -> PairRelationCommitResult:
        """Commit one identity-addressed focal-adhesion rupture."""

        return self.relations.apply(
            state.relations,
            _one(
                self.relations,
                state.time.dtype,
                event_id=event_id,
                operation=PairRelationEventKind.UNBIND,
                relation_id=relation_id,
                incarnation=incarnation,
            ),
        )

    def traction(
        self,
        state: FocalAdhesionState,
        cell_positions: ArrayLike,
        substrate_positions: ArrayLike,
        /,
    ) -> FocalAdhesionEvidence:
        cell = jnp.asarray(cell_positions)
        substrate = jnp.asarray(substrate_positions)
        if cell.shape != (
            self.plan.cell_endpoint_count,
            self.plan.ambient_dimension,
        ):
            raise ValueError("cell_positions has the wrong prepared shape.")
        if substrate.shape != (
            self.plan.substrate_endpoint_count,
            self.plan.ambient_dimension,
        ):
            raise ValueError("substrate_positions has the wrong prepared shape.")
        positions = jnp.concatenate((cell, substrate), axis=0)
        springs = self.springs.evaluate(state.relations, positions)
        traction = springs.forces[: self.plan.cell_endpoint_count]
        commit = self.relations.apply(
            state.relations, _empty(self.relations, positions.dtype)
        )
        successful = springs.successful & commit.successful
        return FocalAdhesionEvidence(
            commit.evidence,
            springs,
            traction,
            jnp.sum(traction, axis=0),
            jnp.sqrt(jnp.sum(traction * traction)),
            jnp.zeros((), dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.int32),
            springs.finite,
            successful,
            self.prepared_id,
        )

    def step(
        self,
        state: FocalAdhesionState,
        key: Array,
        dt: ArrayLike,
        cell_positions: ArrayLike,
        substrate_positions: ArrayLike,
        /,
    ) -> FocalAdhesionStepResult:
        """Advance diffusion capture and force-accelerated adhesion rupture."""

        cell = jnp.asarray(cell_positions)
        substrate = jnp.asarray(substrate_positions)
        if cell.shape != (
            self.plan.cell_endpoint_count,
            self.plan.ambient_dimension,
        ):
            raise ValueError("cell_positions has the wrong prepared shape.")
        if substrate.shape != (
            self.plan.substrate_endpoint_count,
            self.plan.ambient_dimension,
        ):
            raise ValueError("substrate_positions has the wrong prepared shape.")
        dt_ = jnp.asarray(dt, dtype=state.time.dtype)
        if dt_.shape != ():
            raise ValueError("dt must be scalar.")
        positions = jnp.concatenate((cell, substrate), axis=0)
        relation = state.relations
        capacity = self.relations.relation_capacity
        indices = jnp.arange(capacity)
        initial_springs = self.springs.evaluate(relation, positions)
        force = jnp.sqrt(
            jnp.maximum(
                2.0 * relation.parameters[:, 0] * initial_springs.relation_energy,
                0.0,
            )
        )
        rupture_rate = self.plan.unbinding_rate * jnp.exp(
            jnp.minimum(force / self.plan.force_scale, 40.0)
        )
        unbind = relation.active & (
            _uniforms(key, self.plan.realization_id, state.step_index, 30, capacity)
            < -jnp.expm1(-rupture_rate * dt_)
        )
        keys = _keys(key, self.plan.realization_id, state.step_index, 31, capacity)
        proposed_cell = jax.vmap(
            lambda local: jr.randint(local, (), 0, self.plan.cell_endpoint_count)
        )(keys)
        proposed_substrate_local = jax.vmap(
            lambda local: jr.randint(
                jr.fold_in(local, 1), (), 0, self.plan.substrate_endpoint_count
            )
        )(keys)
        proposed_substrate = proposed_substrate_local + self.plan.cell_endpoint_count
        distance = jnp.sqrt(
            jnp.sum(
                (cell[proposed_cell] - substrate[proposed_substrate_local]) ** 2,
                axis=-1,
            )
        )
        occupied = _occupancy(relation, self.endpoint_capacity) > 0
        capture = (
            ~relation.occupied
            & ~occupied[proposed_cell]
            & ~occupied[proposed_substrate]
            & (distance <= self.plan.capture_distance)
            & (
                _uniforms(key, self.plan.realization_id, state.step_index, 32, capacity)
                < _probability(self.plan.binding_rate, dt_)
            )
        )
        conflicts = jax.vmap(
            lambda index: jnp.any(
                capture
                & (indices < index)
                & (
                    (proposed_cell == proposed_cell[index])
                    | (proposed_substrate == proposed_substrate[index])
                )
            )
        )(indices)
        bind = capture & ~conflicts
        operations = jnp.where(
            unbind,
            int(PairRelationEventKind.UNBIND),
            int(PairRelationEventKind.BIND),
        )
        parameters = jnp.broadcast_to(
            jnp.asarray(
                [self.plan.spring_stiffness, self.plan.rest_length],
                dtype=positions.dtype,
            ),
            (capacity, 2),
        )
        events = _batch(
            state.step_index * (capacity + 1) + indices,
            operations,
            unbind | bind,
            jnp.where(bind, -1, relation.relation_ids),
            jnp.where(bind, -1, relation.incarnations),
            jnp.where(bind, proposed_cell, relation.left),
            jnp.where(bind, proposed_substrate, relation.right),
            jnp.where(bind, 0, relation.kind),
            parameters,
        )
        commit = self.relations.apply(relation, events)
        age = self.relations.advance_age(commit.accepted_state, dt_)
        springs = self.springs.evaluate(age.accepted_state, positions)
        traction = springs.forces[: self.plan.cell_endpoint_count]
        finite = jnp.isfinite(dt_) & (dt_ > 0.0) & age.finite & springs.finite
        successful = commit.successful & springs.successful & finite
        candidate = FocalAdhesionState(
            age.candidate_state, state.time + dt_, state.step_index + 1
        )
        accepted = jax.tree.map(
            lambda new, old: jnp.where(successful, new, old), candidate, state
        )
        evidence = FocalAdhesionEvidence(
            commit.evidence,
            springs,
            traction,
            jnp.sum(traction, axis=0),
            jnp.sqrt(jnp.sum(traction * traction)),
            jnp.sum(commit.evidence.applied & bind, dtype=jnp.int32),
            jnp.sum(commit.evidence.applied & unbind, dtype=jnp.int32),
            finite,
            successful,
            self.prepared_id,
        )
        return FocalAdhesionStepResult(candidate, accepted, evidence, successful)


__all__ = [
    "ActinNetworkEvidence",
    "ActinNetworkPlan",
    "ActinNetworkState",
    "ActinStepResult",
    "ChromatinDynamicsPlan",
    "ChromatinObservables",
    "ChromatinState",
    "ChromatinStepEvidence",
    "ChromatinStepResult",
    "FocalAdhesionEvidence",
    "FocalAdhesionPlan",
    "FocalAdhesionState",
    "FocalAdhesionStepResult",
    "MotorCrosslinkerEvidence",
    "MotorCrosslinkerPlan",
    "MotorCrosslinkerState",
    "MotorCrosslinkerStepResult",
    "PreparedActinNetwork",
    "PreparedChromatinDynamics",
    "PreparedFocalAdhesions",
    "PreparedMotorCrosslinkers",
]
