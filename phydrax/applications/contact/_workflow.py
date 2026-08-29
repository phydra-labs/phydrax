#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._bvh import beam_select_leaf_items, build_packed_bvh, PackedBVH
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations import FiniteElementMaterialState, FiniteElementMaterialTransaction
from ...solver import FiniteElementAttemptResult


class FrictionlessContactLaw(StrictModule, NonTrainableState):
    penalty: Array
    law_id: str = eqx.field(static=True)

    def __init__(self, penalty: ArrayLike, /):
        penalty_ = jnp.asarray(penalty)
        if penalty_.shape != () or not bool(jnp.isfinite(penalty_)) or penalty_ <= 0.0:
            raise ValueError("Contact penalty must be one positive finite scalar.")
        self.penalty = penalty_
        self.law_id = canonical_fingerprint(
            {"kind": "frictionless-contact-law", "penalty": float(penalty_)}
        )

    def response(
        self,
        gap: ArrayLike,
        normal: ArrayLike,
        /,
    ) -> tuple[Array, Array, Array]:
        gap_ = jnp.asarray(gap)
        normal_ = jnp.asarray(normal)
        if normal_.shape != gap_.shape + (normal_.shape[-1],):
            raise ValueError("Contact normals must append one coordinate axis to gap.")
        active = gap_ < 0.0
        pressure = self.penalty * jnp.maximum(-gap_, 0.0)
        traction = pressure[..., None] * normal_
        tangent = self.penalty * active
        return traction, tangent, active


class ContactPairState(StrictModule, NonTrainableState):
    pair_ids: Array
    slave_ids: Array
    master_ids: Array
    reference_gap: Array
    normals: Array
    active: Array
    state_version: int = eqx.field(static=True)
    pair_state_id: str = eqx.field(static=True)

    def __init__(
        self,
        pair_ids: ArrayLike,
        slave_ids: ArrayLike,
        master_ids: ArrayLike,
        reference_gap: ArrayLike,
        normals: ArrayLike,
        /,
        *,
        active: ArrayLike | None = None,
        state_version: int = 0,
    ):
        pairs = jnp.asarray(pair_ids, dtype=jnp.int64)
        slave = jnp.asarray(slave_ids, dtype=jnp.int64)
        master = jnp.asarray(master_ids, dtype=jnp.int64)
        gap = jnp.asarray(reference_gap)
        normals_ = jnp.asarray(normals)
        version = int(state_version)
        if (
            pairs.ndim != 1
            or slave.shape != pairs.shape
            or master.shape != pairs.shape
            or gap.shape != pairs.shape
            or normals_.shape[:-1] != pairs.shape
            or version < 0
        ):
            raise ValueError("Contact pair-state shapes or version are invalid.")
        if len(set(pairs.tolist())) != pairs.size:
            raise ValueError("Persistent contact pair IDs must be unique.")
        norm = jnp.sqrt(jnp.sum(normals_**2, axis=-1))
        if bool(jnp.any(~jnp.isfinite(norm) | (norm <= 0.0))):
            raise ValueError("Contact normals must be finite and nonzero.")
        normals_ = normals_ / norm[:, None]
        active_ = gap < 0.0 if active is None else jnp.asarray(active, dtype=bool)
        if active_.shape != pairs.shape:
            raise ValueError("Contact active mask must match pair IDs.")
        self.pair_ids = pairs
        self.slave_ids = slave
        self.master_ids = master
        self.reference_gap = gap
        self.normals = normals_
        self.active = active_
        self.state_version = version
        self.pair_state_id = canonical_fingerprint(
            {
                "kind": "contact-pair-state",
                "pair_ids": pairs.tolist(),
                "slave_ids": slave.tolist(),
                "master_ids": master.tolist(),
                "dimension": int(normals_.shape[-1]),
                "version": version,
            }
        )

    def packed(self, gap: ArrayLike | None = None, /) -> Array:
        gap_ = self.reference_gap if gap is None else jnp.asarray(gap)
        return jnp.concatenate(
            (gap_[:, None], self.normals, self.active[:, None]), axis=1
        )


class ContactSearchPlan(StrictModule, NonTrainableState):
    """Deterministic fixed-mesh segment search with persistent pair identities."""

    master_segments: Array
    master_ids: Array
    bvh: PackedBVH
    beam_width: int = eqx.field(static=True)
    search_id: str = eqx.field(static=True)

    def __init__(
        self,
        master_segments: ArrayLike,
        master_ids: ArrayLike,
        /,
        *,
        leaf_size: int = 8,
        beam_width: int = 4,
    ):
        segments = np.asarray(master_segments)
        identifiers = np.asarray(master_ids, dtype=np.int64)
        width = int(beam_width)
        if segments.ndim != 3 or segments.shape[1:] != (2, 2):
            raise ValueError("Contact search master segments require shape (n, 2, 2).")
        if (
            identifiers.shape != (segments.shape[0],)
            or len(set(identifiers.tolist())) != identifiers.size
        ):
            raise ValueError("Contact master IDs must be one unique value per segment.")
        if width <= 0:
            raise ValueError("Contact search beam_width must be positive.")
        order = np.argsort(identifiers, kind="stable")
        segments = segments[order]
        identifiers = identifiers[order]
        minimum = np.min(segments, axis=1)
        maximum = np.max(segments, axis=1)
        self.master_segments = jnp.asarray(segments)
        self.master_ids = jnp.asarray(identifiers)
        self.bvh = build_packed_bvh(
            minimum,
            maximum,
            leaf_size=int(leaf_size),
            dtype=self.master_segments.dtype,
        )
        self.beam_width = width
        self.search_id = canonical_fingerprint(
            {
                "kind": "contact-segment-search",
                "master_ids": identifiers.tolist(),
                "leaf_size": int(leaf_size),
                "beam_width": width,
            }
        )

    def search(
        self,
        slave_points: ArrayLike,
        slave_ids: ArrayLike,
        /,
    ) -> ContactPairState:
        points = jnp.asarray(slave_points)
        identifiers = jnp.asarray(slave_ids, dtype=jnp.int64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Contact slave points require shape (n, 2).")
        if identifiers.shape != (points.shape[0],):
            raise ValueError("Contact slave IDs must match slave points.")
        candidates, valid = beam_select_leaf_items(
            points,
            bvh=self.bvh,
            beam_width=self.beam_width,
            steps=self.bvh.max_depth + 1,
        )
        segments = self.master_segments[candidates]
        start = segments[..., 0, :]
        tangent = segments[..., 1, :] - start
        denominator = jnp.sum(tangent**2, axis=-1)
        parameter = jnp.clip(
            jnp.sum((points[:, None, :] - start) * tangent, axis=-1) / denominator,
            0.0,
            1.0,
        )
        closest = start + parameter[..., None] * tangent
        distance_squared = jnp.sum((points[:, None, :] - closest) ** 2, axis=-1)
        distance_squared = jnp.where(valid, distance_squared, jnp.inf)
        choice = jnp.argmin(distance_squared, axis=1)
        selected = jnp.take_along_axis(candidates, choice[:, None], axis=1)[:, 0]
        selected_segment = self.master_segments[selected]
        selected_tangent = selected_segment[:, 1] - selected_segment[:, 0]
        length = jnp.sqrt(jnp.sum(selected_tangent**2, axis=-1))
        normal = (
            jnp.stack((selected_tangent[:, 1], -selected_tangent[:, 0]), axis=-1)
            / length[:, None]
        )
        selected_start = selected_segment[:, 0]
        selected_parameter = parameter[jnp.arange(points.shape[0]), choice]
        selected_closest = selected_start + selected_parameter[:, None] * selected_tangent
        gap = jnp.sum((points - selected_closest) * normal, axis=-1)
        master_ids = self.master_ids[selected]
        pair_ids = (identifiers << jnp.int64(32)) ^ master_ids
        return ContactPairState(
            pair_ids,
            identifiers,
            master_ids,
            gap,
            normal,
        )


class ContactEvaluation(StrictModule):
    gap: Array
    traction: Array
    tangent: Array
    active: Array
    pair_ids: Array


class ContactWorkflow(StrictModule, NonTrainableState):
    law: FrictionlessContactLaw
    pairs: ContactPairState
    workflow_id: str = eqx.field(static=True)

    def __init__(self, law: FrictionlessContactLaw, pairs: ContactPairState, /):
        if not isinstance(law, FrictionlessContactLaw) or not isinstance(
            pairs, ContactPairState
        ):
            raise TypeError("ContactWorkflow requires a law and persistent pair state.")
        self.law = law
        self.pairs = pairs
        self.workflow_id = canonical_fingerprint(
            {
                "kind": "contact-workflow",
                "law": law.law_id,
                "pairs": pairs.pair_state_id,
            }
        )

    def evaluate(
        self,
        slave_displacement: ArrayLike,
        master_displacement: ArrayLike,
        /,
    ) -> ContactEvaluation:
        slave = jnp.asarray(slave_displacement)
        master = jnp.asarray(master_displacement)
        if slave.shape != self.pairs.normals.shape or master.shape != slave.shape:
            raise ValueError("Contact displacement traces must match pair normals.")
        normal_motion = jnp.sum((slave - master) * self.pairs.normals, axis=-1)
        gap = self.pairs.reference_gap + normal_motion
        traction, tangent, active = self.law.response(gap, self.pairs.normals)
        return ContactEvaluation(
            gap=gap,
            traction=traction,
            tangent=tangent,
            active=active,
            pair_ids=self.pairs.pair_ids,
        )

    def attempt(
        self,
        fields,
        evaluation: ContactEvaluation,
        accepted: ArrayLike,
        /,
        *,
        retry_requested: ArrayLike = False,
        suggested_step: ArrayLike = 0.0,
    ) -> FiniteElementAttemptResult:
        if not isinstance(evaluation, ContactEvaluation):
            raise TypeError("evaluation must be ContactEvaluation.")
        pair_state = FiniteElementMaterialState(
            "contact-pairs",
            self.pairs.packed(),
            trial=jnp.concatenate(
                (
                    evaluation.gap[:, None],
                    self.pairs.normals,
                    evaluation.active[:, None],
                ),
                axis=1,
            ),
            state_version=self.pairs.state_version,
        )
        return FiniteElementAttemptResult(
            fields,
            accepted,
            materials=FiniteElementMaterialTransaction((pair_state,)),
            retry_requested=retry_requested,
            suggested_step=suggested_step,
            diagnostics=evaluation,
        )


__all__ = [
    "ContactSearchPlan",
    "ContactEvaluation",
    "ContactPairState",
    "ContactWorkflow",
    "FrictionlessContactLaw",
]
