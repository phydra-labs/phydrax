#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-capacity, identity-safe dynamic relations between particle endpoints.

A relation identity is the pair ``(relation_id, incarnation)``. Relation IDs are
stable prepared slot identities; incarnations advance whenever a vacated slot is
bound again. Candidate event batches are evaluated in deterministic event-ID
order and committed atomically, so a capacity or identity failure cannot leave a
partially modified graph.
"""

from __future__ import annotations

from enum import IntEnum
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class PairRelationEventKind(IntEnum):
    """Discrete mutations understood by a dynamic pair-relation runtime."""

    BIND = 0
    UNBIND = 1
    MOVE = 2
    ACTIVATE = 3
    DEACTIVATE = 4


class PairRelationStatus(IntEnum):
    """Fail-closed status for one relation event or atomic event batch."""

    SUCCESS = 0
    CAPACITY_EXCEEDED = 1
    INVALID_ENDPOINT = 2
    DUPLICATE = 3
    EXCLUDED = 4
    STALE_IDENTITY = 5
    INVALID_REQUEST = 6
    NONFINITE = 7
    INCARNATION_OVERFLOW = 8


class PairRelationState(StrictModule):
    """Fixed-shape runtime graph and stable slot identities."""

    relation_ids: Array
    incarnations: Array
    left: Array
    right: Array
    kind: Array
    occupied: Array
    active: Array
    age: Array
    parameters: Array
    ever_occupied: Array
    numeric_version: Array


class PairRelationEventBatch(StrictModule):
    """One fixed-capacity batch of addressed relation mutations."""

    event_ids: Array
    event_kind: Array
    valid: Array
    relation_ids: Array
    relation_incarnations: Array
    left: Array
    right: Array
    relation_kind: Array
    parameters: Array


class PairRelationEvidence(StrictModule):
    """Auditable event-level and aggregate rejection evidence."""

    event_status: Array
    applied: Array
    requested_count: Array
    applied_count: Array
    overflow_count: Array
    invalid_endpoint_count: Array
    duplicate_count: Array
    exclusion_count: Array
    stale_identity_count: Array
    invalid_request_count: Array
    nonfinite_count: Array
    incarnation_overflow_count: Array
    invalid_state_count: Array
    source_state_match: Array
    finite: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class PairRelationEvaluation(StrictModule):
    """Candidate graph and evidence prior to atomic commit."""

    source_state: PairRelationState
    candidate_state: PairRelationState
    evidence: PairRelationEvidence
    prepared_id: str = eqx.field(static=True)


class PairRelationCommitResult(StrictModule):
    """Atomic commit result; ``accepted_state`` is unchanged on any failure."""

    candidate_state: PairRelationState
    accepted_state: PairRelationState
    evidence: PairRelationEvidence
    successful: Array
    prepared_id: str = eqx.field(static=True)


class PairRelationAgeResult(StrictModule):
    """Fail-closed active-relation age advance."""

    candidate_state: PairRelationState
    accepted_state: PairRelationState
    finite: Array
    successful: Array


def make_pair_relation_events(
    event_capacity: int,
    parameter_width: int,
    /,
    *,
    event_ids: ArrayLike = (),
    event_kind: ArrayLike = (),
    valid: ArrayLike | None = None,
    relation_ids: ArrayLike = (),
    relation_incarnations: ArrayLike = (),
    left: ArrayLike = (),
    right: ArrayLike = (),
    relation_kind: ArrayLike = (),
    parameters: ArrayLike | None = None,
    dtype: np.dtype | type = float,
) -> PairRelationEventBatch:
    """Pad host event data into one canonical fixed-capacity event batch.

    Omitted target identities and endpoints are filled with ``-1``. This helper
    is intentionally a preparation-time convenience; compiled paths consume the
    returned fixed-shape arrays directly.
    """

    capacity = int(event_capacity)
    width = int(parameter_width)
    if capacity <= 0 or width <= 0:
        raise ValueError("Event capacity and parameter width must be positive.")
    if not np.issubdtype(np.dtype(dtype), np.inexact):
        raise TypeError("Relation parameters must use an inexact dtype.")
    event_values = np.asarray(event_kind, dtype=np.int32)
    count = int(event_values.size)
    if event_values.ndim != 1 or count > capacity:
        raise ValueError("event_kind must be rank one and fit event_capacity.")

    def padded(values: ArrayLike, fill: int, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.int32)
        if array.size == 0 and count:
            array = np.full((count,), fill, dtype=np.int32)
        if array.shape != (count,):
            raise ValueError(f"{name} must have the event-count shape.")
        return np.pad(array, (0, capacity - count), constant_values=fill)

    identifiers = padded(event_ids, 0, "event_ids")
    target_ids = padded(relation_ids, -1, "relation_ids")
    target_incarnations = padded(relation_incarnations, -1, "relation_incarnations")
    left_ = padded(left, -1, "left")
    right_ = padded(right, -1, "right")
    kinds = padded(relation_kind, -1, "relation_kind")
    if valid is None:
        valid_ = np.ones((count,), dtype=bool)
    else:
        valid_ = np.asarray(valid, dtype=bool)
        if valid_.shape != (count,):
            raise ValueError("valid must have the event-count shape.")
    valid_ = np.pad(valid_, (0, capacity - count), constant_values=False)
    if parameters is None:
        parameters_ = np.zeros((count, width), dtype=dtype)
    else:
        parameters_ = np.asarray(parameters, dtype=dtype)
        if parameters_.shape != (count, width):
            raise ValueError("parameters must have shape (event count, parameter width).")
    parameters_ = np.pad(parameters_, ((0, capacity - count), (0, 0)))
    event_values = np.pad(
        event_values,
        (0, capacity - count),
        constant_values=int(PairRelationEventKind.BIND),
    )
    return PairRelationEventBatch(
        jnp.asarray(identifiers),
        jnp.asarray(event_values),
        jnp.asarray(valid_),
        jnp.asarray(target_ids),
        jnp.asarray(target_incarnations),
        jnp.asarray(left_),
        jnp.asarray(right_),
        jnp.asarray(kinds),
        jnp.asarray(parameters_),
    )


class DynamicPairRelationPlan(StrictModule, NonTrainableState):
    """Plan a reusable, fixed-capacity relation table.

    ``compatibility[k, a, b]`` declares whether relation kind ``k`` may connect
    endpoint types ``a`` and ``b``. ``exclusion[k, q]`` rejects an active new or
    moved kind-``k`` relation when it shares either endpoint with an active
    kind-``q`` relation. Exact duplicate relations are always rejected.
    """

    endpoint_types: Array
    endpoint_active: Array
    compatibility: Array
    exclusion: Array
    symmetric_kinds: Array
    relation_ids: Array
    relation_capacity: int = eqx.field(static=True)
    parameter_width: int = eqx.field(static=True)
    endpoint_capacity: int = eqx.field(static=True)
    endpoint_type_count: int = eqx.field(static=True)
    kind_count: int = eqx.field(static=True)
    event_capacity: int = eqx.field(static=True)
    incarnation_maximum: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        endpoint_types: ArrayLike,
        relation_capacity: int,
        parameter_width: int,
        /,
        *,
        endpoint_active: ArrayLike | None = None,
        compatibility: ArrayLike | None = None,
        exclusion: ArrayLike | None = None,
        symmetric_kinds: ArrayLike | None = None,
        relation_ids: ArrayLike | None = None,
        kind_count: int = 1,
        event_capacity: int | None = None,
        incarnation_maximum: int = 2**31 - 1,
        plan_id: str | None = None,
    ):
        types = np.asarray(endpoint_types)
        relation_count = int(relation_capacity)
        width = int(parameter_width)
        kinds = int(kind_count)
        events = relation_count if event_capacity is None else int(event_capacity)
        incarnation_limit = int(incarnation_maximum)
        if (
            types.ndim != 1
            or types.size == 0
            or not np.issubdtype(types.dtype, np.integer)
        ):
            raise TypeError("endpoint_types must be a nonempty rank-1 integer array.")
        types = types.astype(np.int32)
        if np.any(types < 0):
            raise ValueError("endpoint_types must be nonnegative.")
        if relation_count <= 0 or width <= 0 or kinds <= 0 or events <= 0:
            raise ValueError(
                "Relation, parameter, kind, and event capacities must be positive."
            )
        if incarnation_limit <= 0 or incarnation_limit > np.iinfo(np.int32).max:
            raise ValueError("incarnation_maximum is invalid.")
        endpoint_count = int(types.size)
        type_count = int(types.max()) + 1
        endpoint_mask = (
            np.ones((endpoint_count,), dtype=bool)
            if endpoint_active is None
            else np.asarray(endpoint_active, dtype=bool)
        )
        if endpoint_mask.shape != (endpoint_count,):
            raise ValueError("endpoint_active must have endpoint-capacity shape.")
        compatible = (
            np.ones((kinds, type_count, type_count), dtype=bool)
            if compatibility is None
            else np.asarray(compatibility, dtype=bool)
        )
        if compatible.shape != (kinds, type_count, type_count):
            raise ValueError(
                "compatibility must have shape (kind count, endpoint type count, endpoint type count)."
            )
        excluded = (
            np.zeros((kinds, kinds), dtype=bool)
            if exclusion is None
            else np.asarray(exclusion, dtype=bool)
        )
        if excluded.shape != (kinds, kinds) or not np.array_equal(excluded, excluded.T):
            raise ValueError("exclusion must be a symmetric kind-by-kind matrix.")
        symmetric = (
            np.zeros((kinds,), dtype=bool)
            if symmetric_kinds is None
            else np.asarray(symmetric_kinds, dtype=bool)
        )
        if symmetric.shape != (kinds,):
            raise ValueError("symmetric_kinds must have kind-count shape.")
        identifiers = (
            np.arange(relation_count, dtype=np.int32)
            if relation_ids is None
            else np.asarray(relation_ids)
        )
        if identifiers.shape != (relation_count,) or not np.issubdtype(
            identifiers.dtype, np.integer
        ):
            raise TypeError("relation_ids must be a relation-capacity integer array.")
        identifiers = identifiers.astype(np.int32)
        if np.any(identifiers < 0) or np.unique(identifiers).size != relation_count:
            raise ValueError("relation_ids must be unique and nonnegative.")
        generated = canonical_fingerprint(
            {
                "kind": "dynamic-pair-relation-plan",
                "topology": array_tree_fingerprint(
                    {
                        "endpoint_types": types,
                        "endpoint_active": endpoint_mask,
                        "compatibility": compatible,
                        "exclusion": excluded,
                        "symmetric": symmetric,
                        "relation_ids": identifiers,
                    }
                ),
                "relation_capacity": relation_count,
                "parameter_width": width,
                "event_capacity": events,
                "incarnation_maximum": incarnation_limit,
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be nonempty.")
        self.endpoint_types = jnp.asarray(types)
        self.endpoint_active = jnp.asarray(endpoint_mask)
        self.compatibility = jnp.asarray(compatible)
        self.exclusion = jnp.asarray(excluded)
        self.symmetric_kinds = jnp.asarray(symmetric)
        self.relation_ids = jnp.asarray(identifiers)
        self.relation_capacity = relation_count
        self.parameter_width = width
        self.endpoint_capacity = endpoint_count
        self.endpoint_type_count = type_count
        self.kind_count = kinds
        self.event_capacity = events
        self.incarnation_maximum = incarnation_limit
        self.plan_id = identifier

    def prepare(
        self, /, *, prepared_scope_id: str = "dynamic-pair-relations"
    ) -> PreparedDynamicPairRelations:
        return PreparedDynamicPairRelations(self, prepared_scope_id=prepared_scope_id)


class PreparedDynamicPairRelations(StrictModule, NonTrainableState):
    """Prepared evaluator for deterministic, fixed-shape graph transitions."""

    plan: DynamicPairRelationPlan
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: DynamicPairRelationPlan,
        /,
        *,
        prepared_scope_id: str = "dynamic-pair-relations",
    ):
        if not isinstance(plan, DynamicPairRelationPlan):
            raise TypeError("plan must be a DynamicPairRelationPlan.")
        scope = str(prepared_scope_id)
        if not scope:
            raise ValueError("prepared_scope_id must be nonempty.")
        self.plan = plan
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-dynamic-pair-relations",
                "plan": plan.plan_id,
                "scope": scope,
            }
        )

    @property
    def relation_capacity(self) -> int:
        return self.plan.relation_capacity

    @property
    def event_capacity(self) -> int:
        return self.plan.event_capacity

    @property
    def parameter_width(self) -> int:
        return self.plan.parameter_width

    def initialize(
        self,
        /,
        *,
        left: ArrayLike | None = None,
        right: ArrayLike | None = None,
        relation_kind: ArrayLike | None = None,
        occupied: ArrayLike | None = None,
        active: ArrayLike | None = None,
        parameters: ArrayLike | None = None,
        incarnations: ArrayLike | None = None,
    ) -> PairRelationState:
        capacity = self.plan.relation_capacity
        width = self.plan.parameter_width
        occupied_ = (
            np.zeros((capacity,), dtype=bool)
            if occupied is None
            else np.asarray(occupied, dtype=bool)
        )
        active_ = occupied_.copy() if active is None else np.asarray(active, dtype=bool)
        left_ = (
            np.full((capacity,), -1, dtype=np.int32)
            if left is None
            else np.asarray(left, dtype=np.int32)
        )
        right_ = (
            np.full((capacity,), -1, dtype=np.int32)
            if right is None
            else np.asarray(right, dtype=np.int32)
        )
        kind_ = (
            np.full((capacity,), -1, dtype=np.int32)
            if relation_kind is None
            else np.asarray(relation_kind, dtype=np.int32)
        )
        parameters_ = (
            np.zeros((capacity, width), dtype=float)
            if parameters is None
            else np.asarray(parameters)
        )
        incarnations_ = (
            occupied_.astype(np.int32)
            if incarnations is None
            else np.asarray(incarnations, dtype=np.int32)
        )
        if any(
            value.shape != (capacity,)
            for value in (occupied_, active_, left_, right_, kind_, incarnations_)
        ) or parameters_.shape != (capacity, width):
            raise ValueError(
                "Initial relation arrays must have their prepared fixed shapes."
            )
        if not np.issubdtype(parameters_.dtype, np.inexact):
            raise TypeError("Initial relation parameters must use an inexact dtype.")
        valid_endpoint = (
            (left_ >= 0)
            & (left_ < self.plan.endpoint_capacity)
            & (right_ >= 0)
            & (right_ < self.plan.endpoint_capacity)
            & (left_ != right_)
        )
        valid_kind = (kind_ >= 0) & (kind_ < self.plan.kind_count)
        if (
            np.any(active_ & ~occupied_)
            or np.any(occupied_ & ~(valid_endpoint & valid_kind))
            or np.any(~occupied_ & ((left_ != -1) | (right_ != -1) | (kind_ != -1)))
            or np.any(incarnations_ < 0)
            or np.any(incarnations_ > self.plan.incarnation_maximum)
            or not np.all(np.isfinite(parameters_))
        ):
            raise ValueError("Initial relation state is structurally invalid.")
        state = PairRelationState(
            self.plan.relation_ids,
            jnp.asarray(incarnations_),
            jnp.asarray(left_),
            jnp.asarray(right_),
            jnp.asarray(kind_),
            jnp.asarray(occupied_),
            jnp.asarray(active_),
            jnp.zeros((capacity,), dtype=jnp.asarray(parameters_).dtype),
            jnp.asarray(parameters_),
            jnp.asarray(occupied_),
            jnp.zeros((), dtype=jnp.int32),
        )
        structural = self._state_structure(state, self.plan.endpoint_active)
        finite = self._state_finite(state)
        if not bool(np.asarray(structural & finite)):
            raise ValueError(
                "Initial relations violate finite, compatibility, duplicate, or exclusion rules."
            )
        return state

    def _validate_shapes(
        self,
        state: PairRelationState,
        events: PairRelationEventBatch,
        endpoint_active: Array,
    ) -> None:
        relation_shape = (self.plan.relation_capacity,)
        if not isinstance(state, PairRelationState):
            raise TypeError("state must be a PairRelationState.")
        if not isinstance(events, PairRelationEventBatch):
            raise TypeError("events must be a PairRelationEventBatch.")
        if any(
            value.shape != relation_shape
            for value in (
                state.relation_ids,
                state.incarnations,
                state.left,
                state.right,
                state.kind,
                state.occupied,
                state.active,
                state.age,
                state.ever_occupied,
            )
        ) or state.parameters.shape != (
            self.plan.relation_capacity,
            self.plan.parameter_width,
        ):
            raise ValueError("state does not have the prepared relation shape.")
        if state.numeric_version.shape != ():
            raise ValueError("state numeric_version must be scalar.")
        if not jnp.issubdtype(state.parameters.dtype, jnp.inexact):
            raise TypeError("state parameters must use an inexact dtype.")
        event_shape = (self.plan.event_capacity,)
        if any(
            value.shape != event_shape
            for value in (
                events.event_ids,
                events.event_kind,
                events.valid,
                events.relation_ids,
                events.relation_incarnations,
                events.left,
                events.right,
                events.relation_kind,
            )
        ) or events.parameters.shape != (
            self.plan.event_capacity,
            self.plan.parameter_width,
        ):
            raise ValueError("events does not have the prepared event shape.")
        if not jnp.issubdtype(events.parameters.dtype, jnp.inexact):
            raise TypeError("event parameters must use an inexact dtype.")
        if endpoint_active.shape != (self.plan.endpoint_capacity,):
            raise ValueError("endpoint_active must have endpoint-capacity shape.")

    def _state_structure(self, state: PairRelationState, endpoint_active: Array) -> Array:
        safe_left = jnp.clip(state.left, 0, self.plan.endpoint_capacity - 1)
        safe_right = jnp.clip(state.right, 0, self.plan.endpoint_capacity - 1)
        safe_kind = jnp.clip(state.kind, 0, self.plan.kind_count - 1)
        left_type = self.plan.endpoint_types[safe_left]
        right_type = self.plan.endpoint_types[safe_right]
        endpoint_valid = (
            (state.left >= 0)
            & (state.left < self.plan.endpoint_capacity)
            & (state.right >= 0)
            & (state.right < self.plan.endpoint_capacity)
            & (state.left != state.right)
            & endpoint_active[safe_left]
            & endpoint_active[safe_right]
        )
        kind_valid = (state.kind >= 0) & (state.kind < self.plan.kind_count)
        compatible = self.plan.compatibility[safe_kind, left_type, right_type]
        structure = (
            jnp.array_equal(state.relation_ids, self.plan.relation_ids)
            & jnp.all(~state.active | state.occupied)
            & jnp.all(~state.occupied | (endpoint_valid & kind_valid & compatible))
            & jnp.all(
                state.occupied
                | ((state.left == -1) & (state.right == -1) & (state.kind == -1))
            )
            & jnp.all(
                (state.incarnations >= 0)
                & (state.incarnations <= self.plan.incarnation_maximum)
            )
            & jnp.all(state.age >= 0.0)
            & (state.numeric_version >= 0)
        )
        indices = jnp.arange(self.plan.relation_capacity)

        def row_valid(index: Array) -> Array:
            same_kind = state.kind == state.kind[index]
            direct = (state.left == state.left[index]) & (
                state.right == state.right[index]
            )
            reverse = (state.left == state.right[index]) & (
                state.right == state.left[index]
            )
            duplicate = (
                state.occupied[index]
                & state.occupied
                & (indices < index)
                & same_kind
                & (direct | (self.plan.symmetric_kinds[safe_kind[index]] & reverse))
            )
            shares = (
                (state.left == state.left[index])
                | (state.left == state.right[index])
                | (state.right == state.left[index])
                | (state.right == state.right[index])
            )
            excluded = (
                state.active[index]
                & state.active
                & (indices < index)
                & shares
                & self.plan.exclusion[safe_kind[index], safe_kind]
            )
            return ~jnp.any(duplicate | excluded)

        return structure & jnp.all(jax.vmap(row_valid)(indices))

    def _state_finite(self, state: PairRelationState) -> Array:
        return jnp.all(jnp.isfinite(state.age)) & jnp.all(jnp.isfinite(state.parameters))

    def _same_state(self, left: PairRelationState, right: PairRelationState) -> Array:
        comparisons = tuple(
            jnp.array_equal(left_leaf, right_leaf)
            for left_leaf, right_leaf in zip(
                jax.tree.leaves(left), jax.tree.leaves(right), strict=True
            )
        )
        result = comparisons[0]
        for comparison in comparisons[1:]:
            result = result & comparison
        return result

    def evaluate(
        self,
        state: PairRelationState,
        events: PairRelationEventBatch,
        /,
        *,
        endpoint_active: ArrayLike | None = None,
    ) -> PairRelationEvaluation:
        endpoint_mask = (
            self.plan.endpoint_active
            if endpoint_active is None
            else jnp.asarray(endpoint_active, dtype=bool)
        )
        self._validate_shapes(state, events, endpoint_mask)
        order = jnp.argsort(
            jnp.where(events.valid, events.event_ids, jnp.iinfo(jnp.int32).max),
            stable=True,
        )
        ordered = jax.tree.map(lambda value: value[order], events)
        event_id_duplicate = jax.vmap(
            lambda identifier, valid: (
                valid & (jnp.sum(ordered.valid & (ordered.event_ids == identifier)) > 1)
            )
        )(ordered.event_ids, ordered.valid)
        initial_state_structural = self._state_structure(state, endpoint_mask)
        initial_state_finite = self._state_finite(state)
        initial_state_valid = initial_state_structural & initial_state_finite
        initial_status = jnp.zeros((self.plan.event_capacity,), dtype=jnp.int32)
        initial_applied = jnp.zeros((self.plan.event_capacity,), dtype=bool)
        indices = jnp.arange(self.plan.relation_capacity, dtype=jnp.int32)

        def apply_event(index: int, carry: tuple[Array, ...]) -> tuple[Array, ...]:
            (
                left,
                right,
                kinds,
                occupied,
                active,
                age,
                parameters,
                incarnations,
                ever_occupied,
                statuses,
                applied,
            ) = carry
            requested = ordered.valid[index]
            operation = ordered.event_kind[index]
            bind = operation == int(PairRelationEventKind.BIND)
            unbind = operation == int(PairRelationEventKind.UNBIND)
            move = operation == int(PairRelationEventKind.MOVE)
            activate = operation == int(PairRelationEventKind.ACTIVATE)
            deactivate = operation == int(PairRelationEventKind.DEACTIVATE)
            targets_existing = unbind | move | activate | deactivate
            known_operation = bind | targets_existing

            target_matches = state.relation_ids == ordered.relation_ids[index]
            target_found = jnp.any(target_matches)
            target_slot = jnp.argmax(target_matches.astype(jnp.int32))
            safe_target = jnp.clip(target_slot, 0, self.plan.relation_capacity - 1)
            stale = targets_existing & (
                ~target_found
                | ~occupied[safe_target]
                | (incarnations[safe_target] != ordered.relation_incarnations[index])
            )

            available_mask = ~occupied
            has_capacity = jnp.any(available_mask)
            free_slot = jnp.argmax(available_mask.astype(jnp.int32))
            slot = jnp.where(bind, free_slot, safe_target)
            safe_slot = jnp.clip(slot, 0, self.plan.relation_capacity - 1)
            candidate_left = jnp.where(move | bind, ordered.left[index], left[safe_slot])
            candidate_right = jnp.where(
                move | bind, ordered.right[index], right[safe_slot]
            )
            candidate_kind = jnp.where(
                bind, ordered.relation_kind[index], kinds[safe_slot]
            )
            safe_left = jnp.clip(candidate_left, 0, self.plan.endpoint_capacity - 1)
            safe_right = jnp.clip(candidate_right, 0, self.plan.endpoint_capacity - 1)
            safe_kind = jnp.clip(candidate_kind, 0, self.plan.kind_count - 1)
            needs_endpoints = bind | move | activate
            endpoint_valid = (
                (candidate_left >= 0)
                & (candidate_left < self.plan.endpoint_capacity)
                & (candidate_right >= 0)
                & (candidate_right < self.plan.endpoint_capacity)
                & (candidate_left != candidate_right)
                & endpoint_mask[safe_left]
                & endpoint_mask[safe_right]
                & (candidate_kind >= 0)
                & (candidate_kind < self.plan.kind_count)
                & self.plan.compatibility[
                    safe_kind,
                    self.plan.endpoint_types[safe_left],
                    self.plan.endpoint_types[safe_right],
                ]
            )
            other = occupied & (indices != safe_slot)
            same_kind = kinds == candidate_kind
            direct = (left == candidate_left) & (right == candidate_right)
            reverse = (left == candidate_right) & (right == candidate_left)
            duplicate = (bind | move | activate) & jnp.any(
                other
                & same_kind
                & (direct | (self.plan.symmetric_kinds[safe_kind] & reverse))
            )
            shares = (
                (left == candidate_left)
                | (left == candidate_right)
                | (right == candidate_left)
                | (right == candidate_right)
            )
            exclusion = (bind | activate | (move & active[safe_slot])) & jnp.any(
                active
                & (indices != safe_slot)
                & shares
                & self.plan.exclusion[
                    safe_kind, jnp.clip(kinds, 0, self.plan.kind_count - 1)
                ]
            )
            finite = ~bind | jnp.all(jnp.isfinite(ordered.parameters[index]))
            incarnation_overflow = (
                bind
                & has_capacity
                & (incarnations[safe_slot] >= self.plan.incarnation_maximum)
            )
            next_incarnation = jnp.where(
                incarnation_overflow,
                incarnations[safe_slot],
                incarnations[safe_slot] + jnp.asarray(bind, dtype=jnp.int32),
            )
            invalid_request = ~known_operation | event_id_duplicate[index]
            invalid_endpoint = needs_endpoints & ~endpoint_valid
            overflow = bind & ~has_capacity
            status = jnp.where(
                invalid_request,
                int(PairRelationStatus.INVALID_REQUEST),
                jnp.where(
                    stale,
                    int(PairRelationStatus.STALE_IDENTITY),
                    jnp.where(
                        ~finite,
                        int(PairRelationStatus.NONFINITE),
                        jnp.where(
                            invalid_endpoint,
                            int(PairRelationStatus.INVALID_ENDPOINT),
                            jnp.where(
                                duplicate,
                                int(PairRelationStatus.DUPLICATE),
                                jnp.where(
                                    exclusion,
                                    int(PairRelationStatus.EXCLUDED),
                                    jnp.where(
                                        overflow,
                                        int(PairRelationStatus.CAPACITY_EXCEEDED),
                                        jnp.where(
                                            incarnation_overflow,
                                            int(PairRelationStatus.INCARNATION_OVERFLOW),
                                            int(PairRelationStatus.SUCCESS),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ).astype(jnp.int32)
            use = requested & (status == int(PairRelationStatus.SUCCESS))
            use_bind = use & bind
            use_unbind = use & unbind
            use_move = use & move
            use_activate = use & activate
            use_deactivate = use & deactivate
            left = left.at[safe_slot].set(
                jnp.where(
                    use_bind | use_move,
                    candidate_left,
                    jnp.where(use_unbind, -1, left[safe_slot]),
                )
            )
            right = right.at[safe_slot].set(
                jnp.where(
                    use_bind | use_move,
                    candidate_right,
                    jnp.where(use_unbind, -1, right[safe_slot]),
                )
            )
            kinds = kinds.at[safe_slot].set(
                jnp.where(
                    use_bind, candidate_kind, jnp.where(use_unbind, -1, kinds[safe_slot])
                )
            )
            parameters = parameters.at[safe_slot].set(
                jnp.where(
                    use_bind,
                    ordered.parameters[index],
                    jnp.where(
                        use_unbind,
                        jnp.zeros_like(parameters[safe_slot]),
                        parameters[safe_slot],
                    ),
                )
            )
            incarnations = incarnations.at[safe_slot].set(
                jnp.where(use_bind, next_incarnation, incarnations[safe_slot])
            )
            ever_occupied = ever_occupied.at[safe_slot].set(
                ever_occupied[safe_slot] | use_bind
            )
            occupied = occupied.at[safe_slot].set(
                jnp.where(
                    use_bind, True, jnp.where(use_unbind, False, occupied[safe_slot])
                )
            )
            active = active.at[safe_slot].set(
                jnp.where(
                    use_bind | use_activate,
                    True,
                    jnp.where(use_unbind | use_deactivate, False, active[safe_slot]),
                )
            )
            age = age.at[safe_slot].set(
                jnp.where(use_bind | use_unbind, 0.0, age[safe_slot])
            )
            statuses = statuses.at[index].set(
                jnp.where(requested, status, int(PairRelationStatus.SUCCESS))
            )
            applied = applied.at[index].set(use)
            return (
                left,
                right,
                kinds,
                occupied,
                active,
                age,
                parameters,
                incarnations,
                ever_occupied,
                statuses,
                applied,
            )

        initial = (
            state.left,
            state.right,
            state.kind,
            state.occupied,
            state.active,
            state.age,
            state.parameters,
            state.incarnations,
            state.ever_occupied,
            initial_status,
            initial_applied,
        )
        final = jax.lax.fori_loop(0, self.plan.event_capacity, apply_event, initial)
        (
            left,
            right,
            kinds,
            occupied,
            active,
            age,
            parameters,
            incarnations,
            ever_occupied,
            ordered_status,
            ordered_applied,
        ) = final
        inverse_order = jnp.argsort(order)
        event_status = ordered_status[inverse_order]
        applied = ordered_applied[inverse_order]
        state_changed = jnp.any(applied)
        version_available = ~state_changed | (
            state.numeric_version < jnp.iinfo(jnp.int32).max
        )
        candidate = PairRelationState(
            state.relation_ids,
            incarnations,
            left,
            right,
            kinds,
            occupied,
            active,
            age,
            parameters,
            ever_occupied,
            state.numeric_version + (state_changed & version_available).astype(jnp.int32),
        )
        requested = events.valid
        finite_parameters = events.valid & (
            events.event_kind == int(PairRelationEventKind.BIND)
        )
        finite = initial_state_finite & jnp.all(
            ~finite_parameters[:, None] | jnp.isfinite(events.parameters)
        )
        successful = (
            initial_state_valid
            & version_available
            & jnp.all(~requested | (event_status == int(PairRelationStatus.SUCCESS)))
        )

        def count(status: PairRelationStatus) -> Array:
            return jnp.sum(requested & (event_status == int(status)), dtype=jnp.int32)

        evidence = PairRelationEvidence(
            event_status,
            applied,
            jnp.sum(requested, dtype=jnp.int32),
            jnp.sum(applied, dtype=jnp.int32),
            count(PairRelationStatus.CAPACITY_EXCEEDED),
            count(PairRelationStatus.INVALID_ENDPOINT),
            count(PairRelationStatus.DUPLICATE),
            count(PairRelationStatus.EXCLUDED),
            count(PairRelationStatus.STALE_IDENTITY),
            count(PairRelationStatus.INVALID_REQUEST),
            count(PairRelationStatus.NONFINITE)
            + (~initial_state_finite).astype(jnp.int32),
            count(PairRelationStatus.INCARNATION_OVERFLOW),
            ((~initial_state_structural) | ~version_available).astype(jnp.int32),
            jnp.asarray(True),
            finite,
            successful,
            self.prepared_id,
        )
        return PairRelationEvaluation(state, candidate, evidence, self.prepared_id)

    def commit(
        self,
        state: PairRelationState,
        evaluation: PairRelationEvaluation,
        /,
    ) -> PairRelationCommitResult:
        if not isinstance(evaluation, PairRelationEvaluation):
            raise TypeError("evaluation must be a PairRelationEvaluation.")
        if evaluation.prepared_id != self.prepared_id:
            raise ValueError(
                "evaluation belongs to a different prepared relation runtime."
            )
        source_matches = self._same_state(state, evaluation.source_state)
        successful = evaluation.evidence.successful & source_matches
        candidate = evaluation.candidate_state
        accepted = jax.tree.map(
            lambda new, old: jnp.where(successful, new, old), candidate, state
        )
        evidence = eqx.tree_at(
            lambda value: (
                value.stale_identity_count,
                value.source_state_match,
                value.successful,
            ),
            evaluation.evidence,
            (
                evaluation.evidence.stale_identity_count
                + (~source_matches).astype(jnp.int32),
                source_matches,
                successful,
            ),
        )
        return PairRelationCommitResult(
            candidate,
            accepted,
            evidence,
            successful,
            self.prepared_id,
        )

    def apply(
        self,
        state: PairRelationState,
        events: PairRelationEventBatch,
        /,
        *,
        endpoint_active: ArrayLike | None = None,
    ) -> PairRelationCommitResult:
        """Evaluate and atomically commit one event batch."""

        return self.commit(
            state, self.evaluate(state, events, endpoint_active=endpoint_active)
        )

    def advance_age(
        self, state: PairRelationState, dt: ArrayLike, /
    ) -> PairRelationAgeResult:
        """Advance active relation ages while rejecting nonfinite/nonpositive steps."""

        step = jnp.asarray(dt, dtype=state.age.dtype)
        if step.shape != ():
            raise ValueError("dt must be scalar.")
        changed = (step > 0.0) & jnp.any(state.active)
        version_available = ~changed | (state.numeric_version < jnp.iinfo(jnp.int32).max)
        candidate = eqx.tree_at(
            lambda value: (value.age, value.numeric_version),
            state,
            (
                jnp.where(state.active, state.age + step, state.age),
                state.numeric_version + (changed & version_available).astype(jnp.int32),
            ),
        )
        finite = (
            jnp.isfinite(step)
            & (step >= 0.0)
            & jnp.all(jnp.isfinite(candidate.age))
            & version_available
        )
        accepted = jax.tree.map(
            lambda new, old: jnp.where(finite, new, old), candidate, state
        )
        return PairRelationAgeResult(candidate, accepted, finite, finite)


class PairSpringPlan(StrictModule, NonTrainableState):
    """Parameter-column mapping for conservative pair springs."""

    stiffness_parameter: int = eqx.field(static=True)
    rest_length_parameter: int = eqx.field(static=True)
    minimum_length: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        stiffness_parameter: int = 0,
        rest_length_parameter: int = 1,
        /,
        *,
        minimum_length: float = 1.0e-12,
        plan_id: str | None = None,
    ):
        stiffness = int(stiffness_parameter)
        rest = int(rest_length_parameter)
        minimum = float(minimum_length)
        if stiffness < 0 or rest < 0 or stiffness == rest:
            raise ValueError("Spring parameter columns must be distinct and nonnegative.")
        if not isfinite(minimum) or minimum <= 0.0:
            raise ValueError("minimum_length must be positive and finite.")
        generated = canonical_fingerprint(
            {
                "kind": "pair-spring-plan",
                "stiffness_parameter": stiffness,
                "rest_length_parameter": rest,
                "minimum_length": minimum,
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be nonempty.")
        self.stiffness_parameter = stiffness
        self.rest_length_parameter = rest
        self.minimum_length = minimum
        self.plan_id = identifier

    def prepare(
        self, relations: PreparedDynamicPairRelations, /, *, ambient_dimension: int
    ) -> PreparedPairSpringEnergy:
        return PreparedPairSpringEnergy(
            self, relations, ambient_dimension=ambient_dimension
        )


class PairSpringEvaluation(StrictModule):
    """Energy-derived relation forces and their numerical evidence."""

    energy: Array
    forces: Array
    relation_energy: Array
    extension: Array
    degenerate_count: Array
    finite: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class PreparedPairSpringEnergy(StrictModule, NonTrainableState):
    """Prepared conservative spring energy over active dynamic relations."""

    plan: PairSpringPlan
    relations: PreparedDynamicPairRelations
    ambient_dimension: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: PairSpringPlan,
        relations: PreparedDynamicPairRelations,
        /,
        *,
        ambient_dimension: int,
    ):
        if not isinstance(plan, PairSpringPlan):
            raise TypeError("plan must be a PairSpringPlan.")
        if not isinstance(relations, PreparedDynamicPairRelations):
            raise TypeError("relations must be PreparedDynamicPairRelations.")
        dimension = int(ambient_dimension)
        if dimension <= 0:
            raise ValueError("ambient_dimension must be positive.")
        if (
            max(plan.stiffness_parameter, plan.rest_length_parameter)
            >= relations.parameter_width
        ):
            raise ValueError(
                "Spring parameter columns exceed the relation parameter width."
            )
        self.plan = plan
        self.relations = relations
        self.ambient_dimension = dimension
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-pair-spring-energy",
                "plan": plan.plan_id,
                "relations": relations.prepared_id,
                "ambient_dimension": dimension,
            }
        )

    def _terms(
        self, state: PairRelationState, positions: Array
    ) -> tuple[Array, Array, Array]:
        safe_left = jnp.clip(state.left, 0, self.relations.plan.endpoint_capacity - 1)
        safe_right = jnp.clip(state.right, 0, self.relations.plan.endpoint_capacity - 1)
        displacement = positions[safe_right] - positions[safe_left]
        squared_length = jnp.sum(displacement * displacement, axis=-1)
        length = jnp.sqrt(jnp.maximum(squared_length, self.plan.minimum_length**2))
        stiffness = state.parameters[:, self.plan.stiffness_parameter]
        rest_length = state.parameters[:, self.plan.rest_length_parameter]
        extension = length - rest_length
        relation_energy = jnp.where(
            state.occupied & state.active,
            0.5 * stiffness * extension * extension,
            0.0,
        )
        return relation_energy, extension, squared_length

    def energy(self, state: PairRelationState, positions: ArrayLike, /) -> Array:
        """Return the scalar conservative spring energy."""

        coordinates = jnp.asarray(positions)
        if coordinates.shape != (
            self.relations.plan.endpoint_capacity,
            self.ambient_dimension,
        ):
            raise ValueError(
                "positions must have shape (endpoint capacity, ambient dimension)."
            )
        if not jnp.issubdtype(coordinates.dtype, jnp.inexact):
            raise TypeError("positions must use an inexact dtype.")
        relation_energy, _, _ = self._terms(state, coordinates)
        return jnp.sum(relation_energy)

    def forces(self, state: PairRelationState, positions: ArrayLike, /) -> Array:
        """Return forces as the exact negative gradient of ``energy``."""

        coordinates = jnp.asarray(positions)
        return -jax.grad(lambda value: self.energy(state, value))(coordinates)

    def evaluate(
        self, state: PairRelationState, positions: ArrayLike, /
    ) -> PairSpringEvaluation:
        coordinates = jnp.asarray(positions)
        energy = self.energy(state, coordinates)
        forces = self.forces(state, coordinates)
        relation_energy, extension, squared_length = self._terms(state, coordinates)
        active = state.occupied & state.active
        stiffness = state.parameters[:, self.plan.stiffness_parameter]
        rest_length = state.parameters[:, self.plan.rest_length_parameter]
        valid_parameters = jnp.all(
            ~active
            | (
                jnp.isfinite(stiffness)
                & (stiffness >= 0.0)
                & jnp.isfinite(rest_length)
                & (rest_length >= 0.0)
            )
        )
        degenerate = active & (squared_length <= self.plan.minimum_length**2)
        finite = (
            jnp.all(jnp.isfinite(coordinates))
            & jnp.isfinite(energy)
            & jnp.all(jnp.isfinite(forces))
            & jnp.all(jnp.isfinite(relation_energy))
        )
        successful = finite & valid_parameters & ~jnp.any(degenerate)
        return PairSpringEvaluation(
            energy,
            jnp.where(successful, forces, jnp.zeros_like(forces)),
            relation_energy,
            extension,
            jnp.sum(degenerate, dtype=jnp.int32),
            finite,
            successful,
            self.prepared_id,
        )


__all__ = [
    "DynamicPairRelationPlan",
    "PairRelationAgeResult",
    "PairRelationCommitResult",
    "PairRelationEvaluation",
    "PairRelationEventBatch",
    "PairRelationEventKind",
    "PairRelationEvidence",
    "PairRelationState",
    "PairRelationStatus",
    "PairSpringEvaluation",
    "PairSpringPlan",
    "PreparedDynamicPairRelations",
    "PreparedPairSpringEnergy",
    "make_pair_relation_events",
]
