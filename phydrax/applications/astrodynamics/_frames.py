#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._context import AstrodynamicsContext, AstrodynamicsFrame
from ._state import CartesianOrbitState
from ._status import AstrodynamicsStatus


class KinematicTransformEvaluation(StrictModule):
    rotation: Array
    rotation_rate: Array
    translation: Array
    translation_velocity: Array
    valid: Array
    status: Array
    transform_id: str = eqx.field(static=True)


class KinematicFrameTransform(StrictModule, NonTrainableState):
    """Pure source-to-target rotation and moving-origin transform."""

    evaluator: Callable
    source_frame: AstrodynamicsFrame
    target_frame: AstrodynamicsFrame
    transform_id: str = eqx.field(static=True)

    def __init__(
        self,
        evaluator: Callable[
            [Array, Any], tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]
        ],
        source_frame: AstrodynamicsFrame,
        target_frame: AstrodynamicsFrame,
        /,
        *,
        transform_id: str,
    ):
        if not callable(evaluator):
            raise TypeError("evaluator must be callable.")
        if not isinstance(source_frame, AstrodynamicsFrame) or not isinstance(
            target_frame, AstrodynamicsFrame
        ):
            raise TypeError(
                "Frame transform endpoints must be AstrodynamicsFrame objects."
            )
        identifier = str(transform_id).strip()
        if not identifier:
            raise ValueError("transform_id must be non-empty.")
        self.evaluator = evaluator
        self.source_frame = source_frame
        self.target_frame = target_frame
        self.transform_id = canonical_fingerprint(
            {
                "kind": "kinematic-frame-transform",
                "declared_id": identifier,
                "source": source_frame.frame_id,
                "target": target_frame.frame_id,
            }
        )

    def evaluate(
        self, relative_seconds: ArrayLike, args: Any = None, /
    ) -> KinematicTransformEvaluation:
        time = jnp.asarray(relative_seconds).reshape(())
        rotation, rotation_rate, translation, translation_velocity = self.evaluator(
            time, args
        )
        rotation_ = jnp.asarray(rotation)
        rate_ = jnp.asarray(rotation_rate, dtype=rotation_.dtype)
        translation_ = jnp.asarray(translation, dtype=rotation_.dtype)
        velocity_ = jnp.asarray(translation_velocity, dtype=rotation_.dtype)
        if rotation_.shape != (3, 3) or rate_.shape != (3, 3):
            raise ValueError("Frame rotation and derivative must have shape (3,3).")
        if translation_.shape != (3,) or velocity_.shape != (3,):
            raise ValueError("Frame translation and velocity must have shape (3,).")
        orthogonality = rotation_ @ rotation_.T - jnp.eye(3, dtype=rotation_.dtype)
        valid = (
            jnp.isfinite(time)
            & jnp.all(jnp.isfinite(rotation_))
            & jnp.all(jnp.isfinite(rate_))
            & jnp.all(jnp.isfinite(translation_))
            & jnp.all(jnp.isfinite(velocity_))
            & (jnp.max(jnp.abs(orthogonality)) <= 1.0e-9)
            & (jnp.sum(rotation_[0] * jnp.cross(rotation_[1], rotation_[2])) > 0.0)
        )
        status = jnp.where(
            valid,
            int(AstrodynamicsStatus.SUCCESS),
            int(AstrodynamicsStatus.SINGULAR_GEOMETRY),
        ).astype(jnp.int32)
        return KinematicTransformEvaluation(
            rotation_, rate_, translation_, velocity_, valid, status, self.transform_id
        )

    def apply(
        self,
        state: CartesianOrbitState,
        relative_seconds: ArrayLike,
        target_context: AstrodynamicsContext,
        args: Any = None,
        /,
    ) -> tuple[CartesianOrbitState, KinematicTransformEvaluation]:
        if not isinstance(state, CartesianOrbitState):
            raise TypeError("state must be a CartesianOrbitState.")
        if not isinstance(target_context, AstrodynamicsContext):
            raise TypeError("target_context must be an AstrodynamicsContext.")
        if state.context.frame.frame_id != self.source_frame.frame_id:
            raise ValueError("State frame does not match transform source.")
        if target_context.frame.frame_id != self.target_frame.frame_id:
            raise ValueError("Target context frame does not match transform target.")
        if state.context.scale.scale_id != target_context.scale.scale_id:
            raise ValueError("Frame transformation requires matching scale contracts.")
        evaluation = self.evaluate(relative_seconds, args)
        relative_position = state.position - evaluation.translation
        position = evaluation.rotation @ relative_position
        velocity = (
            evaluation.rotation @ (state.velocity - evaluation.translation_velocity)
            + evaluation.rotation_rate @ relative_position
        )
        position = jnp.where(evaluation.valid, position, jnp.zeros_like(position))
        velocity = jnp.where(evaluation.valid, velocity, jnp.zeros_like(velocity))
        return CartesianOrbitState(position, velocity, target_context), evaluation

    def apply_inverse(
        self,
        state: CartesianOrbitState,
        relative_seconds: ArrayLike,
        source_context: AstrodynamicsContext,
        args: Any = None,
        /,
    ) -> tuple[CartesianOrbitState, KinematicTransformEvaluation]:
        if state.context.frame.frame_id != self.target_frame.frame_id:
            raise ValueError("State frame does not match transform target.")
        if source_context.frame.frame_id != self.source_frame.frame_id:
            raise ValueError("Source context frame does not match transform source.")
        evaluation = self.evaluate(relative_seconds, args)
        source_relative = evaluation.rotation.T @ state.position
        position = evaluation.translation + source_relative
        velocity = evaluation.translation_velocity + evaluation.rotation.T @ (
            state.velocity - evaluation.rotation_rate @ source_relative
        )
        position = jnp.where(evaluation.valid, position, jnp.zeros_like(position))
        velocity = jnp.where(evaluation.valid, velocity, jnp.zeros_like(velocity))
        return CartesianOrbitState(position, velocity, source_context), evaluation


class ConstantKinematicEvaluator(StrictModule):
    rotation: Array
    rotation_rate: Array
    translation: Array
    translation_velocity: Array

    def __init__(
        self,
        rotation: ArrayLike,
        /,
        *,
        rotation_rate: ArrayLike | None = None,
        translation: ArrayLike | tuple[float, float, float] = (0.0, 0.0, 0.0),
        translation_velocity: ArrayLike | tuple[float, float, float] = (
            0.0,
            0.0,
            0.0,
        ),
    ):
        rotation_ = jnp.asarray(rotation)
        self.rotation = rotation_
        self.rotation_rate = (
            jnp.zeros_like(rotation_)
            if rotation_rate is None
            else jnp.asarray(rotation_rate, dtype=rotation_.dtype)
        )
        self.translation = jnp.asarray(translation, dtype=rotation_.dtype)
        self.translation_velocity = jnp.asarray(
            translation_velocity, dtype=rotation_.dtype
        )

    def __call__(self, time: Array, args: Any, /):
        del time, args
        return (
            self.rotation,
            self.rotation_rate,
            self.translation,
            self.translation_velocity,
        )


class PreparedFramePath(StrictModule, NonTrainableState):
    transforms: tuple[KinematicFrameTransform, ...]
    path_id: str = eqx.field(static=True)

    def __init__(self, transforms: tuple[KinematicFrameTransform, ...], /):
        items = tuple(transforms)
        if not items:
            raise ValueError("Prepared frame path requires at least one transform.")
        for left, right in pairwise(items):
            if left.target_frame.frame_id != right.source_frame.frame_id:
                raise ValueError("Prepared frame path is disconnected.")
        self.transforms = items
        self.path_id = canonical_fingerprint(
            {
                "kind": "prepared-frame-path",
                "transforms": [item.transform_id for item in items],
            }
        )


__all__ = [
    "ConstantKinematicEvaluator",
    "KinematicFrameTransform",
    "KinematicTransformEvaluation",
    "PreparedFramePath",
]
