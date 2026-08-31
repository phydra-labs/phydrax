#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ._blocks import AbstractMemberBlock, MemberBlockEvaluation
from ._reference import MemberKinematics, MemberNetworkDefinition


def _empty_evaluation(energy: Array, valid: Array, dtype, dimension: int):
    empty = jnp.empty((0,), dtype=dtype)
    return MemberBlockEvaluation(
        energy,
        jnp.empty((0,), dtype=jnp.int32),
        empty,
        jnp.empty((0, max(dimension - 1, 1)), dtype=dtype),
        jnp.empty((0, max(dimension - 1, 1)), dtype=dtype),
        empty,
        jnp.empty((0,), dtype=bool),
        jnp.empty((0,), dtype=bool),
        empty,
        valid,
    )


class LinearConnectionSpringBlock(AbstractMemberBlock):
    """Translational and rotational springs between explicit node pairs."""

    node_pairs: Array
    translation_stiffness: Array
    rotation_stiffness: Array
    rest_translation: Array
    rest_rotation: Array

    def __init__(
        self,
        node_pairs: ArrayLike,
        translation_stiffness: ArrayLike,
        rotation_stiffness: ArrayLike,
        /,
        *,
        rest_translation: ArrayLike | None = None,
        rest_rotation: ArrayLike | None = None,
        block_id: str | None = None,
    ):
        pairs = jnp.asarray(node_pairs, dtype=jnp.int32)
        translation = jnp.asarray(translation_stiffness)
        rotation = jnp.asarray(rotation_stiffness, dtype=translation.dtype)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("node_pairs must have shape (connections, 2).")
        if translation.ndim != 2 or translation.shape[0] != pairs.shape[0]:
            raise ValueError("translation_stiffness must align with connections.")
        if rotation.ndim != 2 or rotation.shape[0] != pairs.shape[0]:
            raise ValueError("rotation_stiffness must align with connections.")
        rest_t = (
            jnp.zeros_like(translation)
            if rest_translation is None
            else jnp.asarray(rest_translation, dtype=translation.dtype)
        )
        rest_r = (
            jnp.zeros_like(rotation)
            if rest_rotation is None
            else jnp.asarray(rest_rotation, dtype=rotation.dtype)
        )
        if rest_t.shape != translation.shape or rest_r.shape != rotation.shape:
            raise ValueError("Connection rest values must match stiffness arrays.")
        if bool(jnp.any(translation < 0.0) | jnp.any(rotation < 0.0)):
            raise ValueError("Connection stiffness values must be nonnegative.")
        self.node_pairs = pairs
        self.translation_stiffness = translation
        self.rotation_stiffness = rotation
        self.rest_translation = rest_t
        self.rest_rotation = rest_r
        self.member_indices = jnp.empty((0,), dtype=jnp.int32)
        self.block_id = str(
            block_id
            or canonical_fingerprint(
                {
                    "kind": "linear-connection-spring",
                    "pairs": array_tree_fingerprint(pairs),
                    "translation": array_tree_fingerprint(translation),
                    "rotation": array_tree_fingerprint(rotation),
                }
            )
        )

    def evaluate(
        self,
        definition: MemberNetworkDefinition,
        kinematics: MemberKinematics,
        /,
    ) -> MemberBlockEvaluation:
        first, second = self.node_pairs[:, 0], self.node_pairs[:, 1]
        translation = kinematics.positions[second] - kinematics.positions[first]
        rotation = (
            kinematics.rotation_vectors[second] - kinematics.rotation_vectors[first]
        )
        if translation.shape != self.translation_stiffness.shape:
            raise ValueError(
                "Connection translation dimension does not match the structure."
            )
        if rotation.shape != self.rotation_stiffness.shape:
            raise ValueError(
                "Connection rotation dimension does not match the structure."
            )
        translation_delta = translation - self.rest_translation
        rotation_delta = rotation - self.rest_rotation
        energy = 0.5 * jnp.sum(
            self.translation_stiffness * translation_delta**2
            + self.rotation_stiffness * rotation_delta**2
        )
        valid = jnp.isfinite(energy)
        return _empty_evaluation(
            energy, valid, kinematics.positions.dtype, definition.structure.dimension
        )


class NonlinearMomentRotationBlock(AbstractMemberBlock):
    """Bilinear rotational connection with hardening and rotation capacity."""

    node_pairs: Array
    axis: Array
    elastic_stiffness: Array
    yield_moment: Array
    hardening_ratio: Array
    rotation_capacity: Array

    def __init__(
        self,
        node_pairs: ArrayLike,
        axis: ArrayLike,
        elastic_stiffness: ArrayLike,
        yield_moment: ArrayLike,
        hardening_ratio: ArrayLike,
        rotation_capacity: ArrayLike,
        /,
        *,
        block_id: str | None = None,
    ):
        pairs = jnp.asarray(node_pairs, dtype=jnp.int32)
        axis_ = jnp.asarray(axis)
        stiffness = jnp.asarray(elastic_stiffness, dtype=axis_.dtype)
        yield_ = jnp.asarray(yield_moment, dtype=axis_.dtype)
        hardening = jnp.asarray(hardening_ratio, dtype=axis_.dtype)
        capacity = jnp.asarray(rotation_capacity, dtype=axis_.dtype)
        count = pairs.shape[0]
        if pairs.shape != (count, 2) or axis_.shape[0] != count:
            raise ValueError("Connection axis and node pairs are inconsistent.")
        for value in (stiffness, yield_, hardening, capacity):
            if value.shape != (count,):
                raise ValueError(
                    "Connection scalar properties must match connection count."
                )
        if bool(
            jnp.any(stiffness <= 0.0)
            | jnp.any(yield_ <= 0.0)
            | jnp.any(hardening < 0.0)
            | jnp.any(capacity <= 0.0)
        ):
            raise ValueError("Nonlinear connection properties are inadmissible.")
        norm = jnp.sqrt(jnp.sum(axis_ * axis_, axis=-1))
        self.node_pairs = pairs
        self.axis = axis_ / norm[:, None]
        self.elastic_stiffness = stiffness
        self.yield_moment = yield_
        self.hardening_ratio = hardening
        self.rotation_capacity = capacity
        self.member_indices = jnp.empty((0,), dtype=jnp.int32)
        self.block_id = str(block_id or "nonlinear-moment-rotation")

    def evaluate(self, definition, kinematics, /):
        first, second = self.node_pairs[:, 0], self.node_pairs[:, 1]
        delta = kinematics.rotation_vectors[second] - kinematics.rotation_vectors[first]
        rotation = jnp.sum(delta * self.axis, axis=-1)
        yield_rotation = self.yield_moment / self.elastic_stiffness
        absolute = jnp.abs(rotation)
        elastic = jnp.minimum(absolute, yield_rotation)
        plastic = jnp.maximum(absolute - yield_rotation, 0.0)
        energy = jnp.sum(
            0.5 * self.elastic_stiffness * elastic**2
            + self.yield_moment * plastic
            + 0.5 * self.hardening_ratio * self.elastic_stiffness * plastic**2
        )
        valid = jnp.all(absolute <= self.rotation_capacity) & jnp.isfinite(energy)
        return _empty_evaluation(
            energy, valid, kinematics.positions.dtype, definition.structure.dimension
        )


class GapSupportState(eqx.Module):
    gap: Array
    reaction: Array
    active: Array
    complementarity: Array


def gap_support_response(
    displacement: ArrayLike,
    normal: ArrayLike,
    gap_offset: ArrayLike,
    stiffness: ArrayLike,
    /,
) -> GapSupportState:
    displacement_ = jnp.asarray(displacement)
    normal_ = jnp.asarray(normal, dtype=displacement_.dtype)
    offset = jnp.asarray(gap_offset, dtype=displacement_.dtype)
    stiffness_ = jnp.asarray(stiffness, dtype=displacement_.dtype)
    norm = jnp.sqrt(jnp.sum(normal_ * normal_, axis=-1))
    unit = normal_ / norm[:, None]
    signed_gap = offset + jnp.sum(displacement_ * unit, axis=-1)
    penetration = jnp.maximum(-signed_gap, 0.0)
    reaction = stiffness_ * penetration
    complementarity = reaction * jnp.maximum(signed_gap, 0.0)
    return GapSupportState(signed_gap, reaction, penetration > 0.0, complementarity)


class FrictionSupportState(eqx.Module):
    normal_reaction: Array
    tangential_reaction: Array
    sticking: Array
    cone_residual: Array


def friction_support_response(
    normal_reaction: ArrayLike,
    trial_tangential_reaction: ArrayLike,
    friction_coefficient: ArrayLike,
    /,
) -> FrictionSupportState:
    normal = jnp.asarray(normal_reaction)
    trial = jnp.asarray(trial_tangential_reaction, dtype=normal.dtype)
    coefficient = jnp.asarray(friction_coefficient, dtype=normal.dtype)
    norm = jnp.sqrt(jnp.sum(trial * trial, axis=-1))
    limit = coefficient * jnp.maximum(normal, 0.0)
    scale = jnp.minimum(1.0, limit / jnp.maximum(norm, jnp.finfo(norm.dtype).tiny))
    reaction = scale[:, None] * trial
    return FrictionSupportState(
        normal,
        reaction,
        norm <= limit,
        jnp.maximum(norm - limit, 0.0),
    )


__all__ = [
    "FrictionSupportState",
    "GapSupportState",
    "LinearConnectionSpringBlock",
    "NonlinearMomentRotationBlock",
    "friction_support_response",
    "gap_support_response",
]
