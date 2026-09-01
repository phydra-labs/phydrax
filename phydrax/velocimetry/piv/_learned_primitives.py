#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..imaging._warp import backward_warp


class BackwardWarpResult(StrictModule):
    """A nonperiodic pullback together with its interpolation support."""

    values: Array
    valid: Array


class CostVolumePlan(StrictModule, NonTrainableState):
    """Fixed local displacement lattice and bounded working chunk capacity."""

    offsets_rc: Array
    padded_offsets_rc: Array
    offset_valid: Array
    radius: int = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)
    chunk_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, radius: int, /, *, chunk_size: int = 16):
        radius_ = int(radius)
        chunk_size_ = int(chunk_size)
        if radius_ < 0:
            raise ValueError("radius must be non-negative.")
        if chunk_size_ <= 0:
            raise ValueError("chunk_size must be positive.")
        side = 2 * radius_ + 1
        count = side * side
        chunk_count = (count + chunk_size_ - 1) // chunk_size_
        capacity = chunk_count * chunk_size_
        row_offsets = jnp.repeat(jnp.arange(-radius_, radius_ + 1), side)
        column_offsets = jnp.tile(jnp.arange(-radius_, radius_ + 1), side)
        offsets = jnp.stack((row_offsets, column_offsets), axis=-1)
        padding = capacity - count
        padded = jnp.pad(offsets, ((0, padding), (0, 0)))
        valid = jnp.arange(capacity) < count

        self.offsets_rc = offsets
        self.padded_offsets_rc = padded.reshape((chunk_count, chunk_size_, 2))
        self.offset_valid = valid.reshape((chunk_count, chunk_size_))
        self.radius = radius_
        self.chunk_size = chunk_size_
        self.chunk_count = chunk_count
        self.plan_id = canonical_fingerprint(
            {
                "kind": "piv-local-cost-volume-plan",
                "radius": radius_,
                "chunk_size": chunk_size_,
                "offset_order": "row-major-row-then-column",
            }
        )

    @property
    def offset_count(self) -> int:
        return int(self.offsets_rc.shape[0])


class CostVolumeResult(StrictModule, NonTrainableState):
    """Local correlation scores, support, and their exact offset lattice."""

    values: Array
    valid: Array
    offsets_rc: Array
    plan_id: str = eqx.field(static=True)


def backward_warp_2d(
    image: ArrayLike,
    displacement_rc: ArrayLike,
    /,
    *,
    valid_mask: ArrayLike | None = None,
) -> BackwardWarpResult:
    """Pull an image back without wrapping.

    At output site ``(r, c)``, a displacement ``(dr, dc)`` samples the input at
    ``(r - dr, c - dc)``. Samples whose bilinear footprint leaves the image or
    touches an invalid source site are zero and explicitly invalid.
    """

    image_ = jnp.asarray(image)
    displacement = jnp.asarray(displacement_rc)
    if image_.ndim not in (2, 3):
        raise ValueError(
            "image must have shape (rows, columns) or (rows, columns, channels)."
        )
    if displacement.shape != image_.shape[:2] + (2,):
        raise ValueError(
            "displacement_rc must have shape image.shape[:2] + (2,); "
            f"got {displacement.shape} for image shape {image_.shape}."
        )
    sampled = backward_warp(
        image_,
        displacement,
        valid_mask=valid_mask,
        fill_value=0.0,
    )
    return BackwardWarpResult(values=sampled.values, valid=sampled.valid)


def build_cost_volume_2d(
    reference_features: ArrayLike,
    target_features: ArrayLike,
    plan: CostVolumePlan,
    /,
    *,
    base_displacement_rc: ArrayLike | None = None,
    reference_valid: ArrayLike | None = None,
    target_valid: ArrayLike | None = None,
) -> CostVolumeResult:
    """Build a memory-planned local correlation volume.

    Channel ``k`` evaluates the candidate first-to-second displacement
    ``base_displacement_rc + plan.offsets_rc[k]``. Thus a positive column offset
    correlates a reference pixel with a target pixel to its right. Offset chunks
    cap temporary warped feature storage; padded chunk lanes are masked away.
    """

    reference = jnp.asarray(reference_features)
    target = jnp.asarray(target_features)
    if reference.ndim != 3 or target.ndim != 3:
        raise ValueError("Cost-volume features must be channel-last 2D arrays.")
    if reference.shape != target.shape:
        raise ValueError("reference_features and target_features must have equal shape.")
    rows, columns, channels = reference.shape
    if channels <= 0:
        raise ValueError("Cost-volume features must have at least one channel.")
    if not isinstance(plan, CostVolumePlan):
        raise TypeError("plan must be a CostVolumePlan.")

    base = (
        jnp.zeros((rows, columns, 2), dtype=jnp.result_type(reference.dtype, float))
        if base_displacement_rc is None
        else jnp.asarray(base_displacement_rc)
    )
    if base.shape != (rows, columns, 2):
        raise ValueError("base_displacement_rc must have shape (rows, columns, 2).")
    reference_support = (
        jnp.ones((rows, columns), dtype=bool)
        if reference_valid is None
        else jnp.asarray(reference_valid, dtype=bool)
    )
    if reference_support.shape != (rows, columns):
        raise ValueError("reference_valid must have shape (rows, columns).")
    if target_valid is not None and jnp.asarray(target_valid).shape != (rows, columns):
        raise ValueError("target_valid must have shape (rows, columns).")

    normalization = jnp.asarray(channels, dtype=jnp.result_type(reference.dtype, float))

    def evaluate_offset(offset_rc: Array) -> tuple[Array, Array]:
        candidate = base + offset_rc
        sampled = backward_warp_2d(
            target,
            -candidate,
            valid_mask=target_valid,
        )
        correlation = contract("hwc,hwc->hw", reference, sampled.values) / normalization
        valid = reference_support & sampled.valid
        return jnp.where(valid, correlation, 0.0), valid

    def evaluate_chunk(
        carry: None,
        chunk: tuple[Array, Array],
    ) -> tuple[None, tuple[Array, Array]]:
        offsets, lanes_valid = chunk
        values, valid = jax.vmap(evaluate_offset)(offsets)
        lane_mask = lanes_valid[:, None, None]
        return None, (
            jnp.where(lane_mask, values, 0.0),
            valid & lane_mask,
        )

    _, (chunk_values, chunk_valid) = jax.lax.scan(
        evaluate_chunk,
        None,
        (plan.padded_offsets_rc, plan.offset_valid),
    )
    capacity = plan.chunk_count * plan.chunk_size
    values = chunk_values.reshape((capacity, rows, columns))[: plan.offset_count]
    valid = chunk_valid.reshape((capacity, rows, columns))[: plan.offset_count]
    return CostVolumeResult(
        values=jnp.moveaxis(values, 0, -1),
        valid=jnp.moveaxis(valid, 0, -1),
        offsets_rc=plan.offsets_rc,
        plan_id=plan.plan_id,
    )


def resize_displacement_2d(
    displacement_rc: ArrayLike,
    output_shape: tuple[int, int],
    /,
) -> Array:
    """Resize a row/column displacement and preserve its pixel-unit meaning."""

    displacement = jnp.asarray(displacement_rc)
    if displacement.ndim != 3 or displacement.shape[-1] != 2:
        raise ValueError("displacement_rc must have shape (rows, columns, 2).")
    output_rows, output_columns = (int(output_shape[0]), int(output_shape[1]))
    if output_rows <= 0 or output_columns <= 0:
        raise ValueError("output_shape entries must be positive.")
    input_rows, input_columns = displacement.shape[:2]
    resized = jax.image.resize(
        displacement,
        (output_rows, output_columns, 2),
        method="linear",
        antialias=True,
    )
    component_scale = jnp.asarray(
        (output_rows / input_rows, output_columns / input_columns),
        dtype=resized.dtype,
    )
    return resized * component_scale


def _masked_mean(values: Array, valid: Array) -> tuple[Array, Array]:
    mask = jnp.asarray(valid, dtype=bool)
    if values.shape != mask.shape:
        mask = jnp.broadcast_to(mask, values.shape)
    safe_values = jnp.where(mask, values, 0.0)
    count = jnp.sum(mask)
    denominator = jnp.maximum(count, jnp.asarray(1, dtype=count.dtype))
    mean = jnp.sum(safe_values) / denominator.astype(safe_values.dtype)
    return jnp.where(count > 0, mean, 0.0), count


def _robust_penalty(residual: Array, epsilon: float, exponent: float) -> Array:
    epsilon_ = jnp.asarray(epsilon, dtype=residual.dtype)
    return jnp.power(jnp.square(residual) + jnp.square(epsilon_), exponent) - jnp.power(
        jnp.square(epsilon_), exponent
    )


def _resize_image(image: Array, shape: tuple[int, int]) -> Array:
    return jax.image.resize(
        image,
        shape + (int(image.shape[-1]),),
        method="linear",
        antialias=True,
    )


def _resize_mask(mask: Array, shape: tuple[int, int]) -> Array:
    resized = jax.image.resize(mask.astype(float), shape, method="nearest")
    return resized > 0.5


def _supervised_loss(
    predicted: Array,
    target: Array,
    valid: Array,
    epsilon: float,
    exponent: float,
) -> tuple[Array, Array]:
    residual = predicted - target
    endpoint = jnp.mean(_robust_penalty(residual, epsilon, exponent), axis=-1)
    return _masked_mean(endpoint, valid)


def _photometric_loss(
    first: Array,
    second: Array,
    forward: Array,
    backward: Array,
    first_valid: Array,
    second_valid: Array,
    epsilon: float,
    exponent: float,
) -> tuple[Array, Array]:
    second_to_first = backward_warp_2d(second, -forward, valid_mask=second_valid)
    first_to_second = backward_warp_2d(first, -backward, valid_mask=first_valid)
    first_residual = jnp.mean(
        _robust_penalty(first - second_to_first.values, epsilon, exponent), axis=-1
    )
    second_residual = jnp.mean(
        _robust_penalty(second - first_to_second.values, epsilon, exponent), axis=-1
    )
    first_loss, first_count = _masked_mean(
        first_residual, first_valid & second_to_first.valid
    )
    second_loss, second_count = _masked_mean(
        second_residual, second_valid & first_to_second.valid
    )
    count = first_count + second_count
    numerator = first_loss * first_count + second_loss * second_count
    denominator = jnp.maximum(count, jnp.asarray(1, dtype=count.dtype))
    return jnp.where(count > 0, numerator / denominator, 0.0), count


def _consistency_loss(
    forward: Array,
    backward: Array,
    first_valid: Array,
    second_valid: Array,
    epsilon: float,
    exponent: float,
) -> tuple[Array, Array]:
    backward_at_forward = backward_warp_2d(backward, -forward, valid_mask=second_valid)
    forward_at_backward = backward_warp_2d(forward, -backward, valid_mask=first_valid)
    first_residual = jnp.mean(
        _robust_penalty(forward + backward_at_forward.values, epsilon, exponent),
        axis=-1,
    )
    second_residual = jnp.mean(
        _robust_penalty(backward + forward_at_backward.values, epsilon, exponent),
        axis=-1,
    )
    first_loss, first_count = _masked_mean(
        first_residual, first_valid & backward_at_forward.valid
    )
    second_loss, second_count = _masked_mean(
        second_residual, second_valid & forward_at_backward.valid
    )
    count = first_count + second_count
    numerator = first_loss * first_count + second_loss * second_count
    denominator = jnp.maximum(count, jnp.asarray(1, dtype=count.dtype))
    return jnp.where(count > 0, numerator / denominator, 0.0), count


def _smoothness_loss(
    displacement: Array,
    valid: Array,
    epsilon: float,
    exponent: float,
) -> tuple[Array, Array]:
    row_residual = jnp.mean(
        _robust_penalty(
            displacement[1:, :, :] - displacement[:-1, :, :], epsilon, exponent
        ),
        axis=-1,
    )
    column_residual = jnp.mean(
        _robust_penalty(
            displacement[:, 1:, :] - displacement[:, :-1, :], epsilon, exponent
        ),
        axis=-1,
    )
    row_loss, row_count = _masked_mean(row_residual, valid[1:, :] & valid[:-1, :])
    column_loss, column_count = _masked_mean(
        column_residual, valid[:, 1:] & valid[:, :-1]
    )
    count = row_count + column_count
    numerator = row_loss * row_count + column_loss * column_count
    denominator = jnp.maximum(count, jnp.asarray(1, dtype=count.dtype))
    return jnp.where(count > 0, numerator / denominator, 0.0), count


PIV_LOSS_SUCCESS = 0
PIV_LOSS_INSUFFICIENT_SUPPORT = 1


class PIVLossResult(StrictModule):
    """Masked loss terms and the evidence count supporting each one."""

    total: Array
    supervised: Array
    photometric: Array
    consistency: Array
    smoothness: Array
    supervised_valid_count: Array
    photometric_valid_count: Array
    consistency_valid_count: Array
    smoothness_valid_count: Array
    valid: Array
    status: Array


class MultiScaleRobustPIVLoss(StrictModule, NonTrainableState):
    """Masked robust supervision, reconstruction, cycle, and regularity loss."""

    scale_weights: tuple[float, ...] = eqx.field(static=True)
    supervised_weight: float = eqx.field(static=True)
    photometric_weight: float = eqx.field(static=True)
    consistency_weight: float = eqx.field(static=True)
    smoothness_weight: float = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    exponent: float = eqx.field(static=True)
    loss_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        scale_weights: Sequence[float] = (1.0,),
        supervised_weight: float = 1.0,
        photometric_weight: float = 1.0,
        consistency_weight: float = 0.1,
        smoothness_weight: float = 0.01,
        epsilon: float = 1e-3,
        exponent: float = 0.5,
    ):
        scale_weights_ = tuple(float(weight) for weight in scale_weights)
        term_weights = (
            float(supervised_weight),
            float(photometric_weight),
            float(consistency_weight),
            float(smoothness_weight),
        )
        epsilon_ = float(epsilon)
        exponent_ = float(exponent)
        if not scale_weights_ or any(
            not math.isfinite(weight) or weight < 0.0 for weight in scale_weights_
        ):
            raise ValueError(
                "scale_weights must be a nonempty sequence of finite nonnegative values."
            )
        if sum(scale_weights_) <= 0.0:
            raise ValueError("At least one scale weight must be positive.")
        if any(not math.isfinite(weight) or weight < 0.0 for weight in term_weights):
            raise ValueError("PIV loss term weights must be finite and nonnegative.")
        if not math.isfinite(epsilon_) or epsilon_ <= 0.0:
            raise ValueError("epsilon must be finite and positive.")
        if not math.isfinite(exponent_) or not 0.0 < exponent_ <= 1.0:
            raise ValueError("exponent must lie in (0, 1].")

        self.scale_weights = scale_weights_
        self.supervised_weight = term_weights[0]
        self.photometric_weight = term_weights[1]
        self.consistency_weight = term_weights[2]
        self.smoothness_weight = term_weights[3]
        self.epsilon = epsilon_
        self.exponent = exponent_
        self.loss_id = canonical_fingerprint(
            {
                "kind": "multiscale-robust-piv-loss",
                "scale_weights": scale_weights_,
                "term_weights": term_weights,
                "epsilon": epsilon_,
                "exponent": exponent_,
            }
        )

    def __call__(
        self,
        first_image: ArrayLike,
        second_image: ArrayLike,
        forward_pyramid_rc: Sequence[ArrayLike],
        backward_pyramid_rc: Sequence[ArrayLike],
        /,
        *,
        first_valid: ArrayLike | None = None,
        second_valid: ArrayLike | None = None,
        target_forward_rc: ArrayLike | None = None,
        target_backward_rc: ArrayLike | None = None,
        target_valid: ArrayLike | None = None,
    ) -> PIVLossResult:
        first = jnp.asarray(first_image)
        second = jnp.asarray(second_image)
        if first.ndim != 3 or second.shape != first.shape:
            raise ValueError(
                "Loss images must have the same NHWC-case shape (rows, columns, channels)."
            )
        forward_levels = tuple(jnp.asarray(level) for level in forward_pyramid_rc)
        backward_levels = tuple(jnp.asarray(level) for level in backward_pyramid_rc)
        if not forward_levels or len(forward_levels) != len(backward_levels):
            raise ValueError(
                "Forward and backward displacement pyramids must be nonempty and equally sized."
            )
        if len(self.scale_weights) not in (1, len(forward_levels)):
            raise ValueError(
                "scale_weights must contain one value or one value per pyramid level."
            )
        full_shape = first.shape[:2]
        first_mask = (
            jnp.ones(full_shape, dtype=bool)
            if first_valid is None
            else jnp.asarray(first_valid, dtype=bool)
        )
        second_mask = (
            jnp.ones(full_shape, dtype=bool)
            if second_valid is None
            else jnp.asarray(second_valid, dtype=bool)
        )
        if first_mask.shape != full_shape or second_mask.shape != full_shape:
            raise ValueError("Image validity masks must match the image spatial shape.")
        target_forward = (
            None if target_forward_rc is None else jnp.asarray(target_forward_rc)
        )
        target_backward = (
            None if target_backward_rc is None else jnp.asarray(target_backward_rc)
        )
        target_mask = (
            first_mask if target_valid is None else jnp.asarray(target_valid, dtype=bool)
        )
        if target_forward is not None and target_forward.shape != full_shape + (2,):
            raise ValueError(
                "target_forward_rc must match the full-resolution displacement shape."
            )
        if target_backward is not None and target_backward.shape != full_shape + (2,):
            raise ValueError(
                "target_backward_rc must match the full-resolution displacement shape."
            )
        if target_mask.shape != full_shape:
            raise ValueError("target_valid must match the image spatial shape.")

        weights = (
            self.scale_weights
            if len(self.scale_weights) > 1
            else self.scale_weights * len(forward_levels)
        )
        weight_sum = sum(weights)
        supervised = jnp.asarray(0.0, dtype=first.dtype)
        photometric = jnp.asarray(0.0, dtype=first.dtype)
        consistency = jnp.asarray(0.0, dtype=first.dtype)
        smoothness = jnp.asarray(0.0, dtype=first.dtype)
        supervised_count = jnp.asarray(0, dtype=jnp.int32)
        photometric_count = jnp.asarray(0, dtype=jnp.int32)
        consistency_count = jnp.asarray(0, dtype=jnp.int32)
        smoothness_count = jnp.asarray(0, dtype=jnp.int32)

        for forward, backward, raw_weight in zip(
            forward_levels, backward_levels, weights, strict=True
        ):
            if (
                forward.shape != backward.shape
                or forward.ndim != 3
                or forward.shape[-1] != 2
            ):
                raise ValueError(
                    "Every displacement level must have equal shape (rows, columns, 2)."
                )
            shape = (int(forward.shape[0]), int(forward.shape[1]))
            scale_weight = raw_weight / weight_sum
            first_level = _resize_image(first, shape)
            second_level = _resize_image(second, shape)
            first_valid_level = _resize_mask(first_mask, shape)
            second_valid_level = _resize_mask(second_mask, shape)

            photo_value, photo_count = _photometric_loss(
                first_level,
                second_level,
                forward,
                backward,
                first_valid_level,
                second_valid_level,
                self.epsilon,
                self.exponent,
            )
            consistency_value, consistency_level_count = _consistency_loss(
                forward,
                backward,
                first_valid_level,
                second_valid_level,
                self.epsilon,
                self.exponent,
            )
            forward_smoothness, forward_smoothness_count = _smoothness_loss(
                forward,
                first_valid_level,
                self.epsilon,
                self.exponent,
            )
            backward_smoothness, backward_smoothness_count = _smoothness_loss(
                backward,
                second_valid_level,
                self.epsilon,
                self.exponent,
            )
            smoothness_level_count = forward_smoothness_count + backward_smoothness_count
            smoothness_numerator = (
                forward_smoothness * forward_smoothness_count
                + backward_smoothness * backward_smoothness_count
            )
            smoothness_denominator = jnp.maximum(
                smoothness_level_count,
                jnp.asarray(1, dtype=smoothness_level_count.dtype),
            )
            smoothness_value = jnp.where(
                smoothness_level_count > 0,
                smoothness_numerator / smoothness_denominator,
                0.0,
            )

            supervised_value = jnp.asarray(0.0, dtype=first.dtype)
            supervised_level_count = jnp.asarray(0, dtype=jnp.int32)
            if target_forward is not None:
                forward_target = resize_displacement_2d(target_forward, shape)
                target_valid_level = _resize_mask(target_mask, shape)
                supervised_value, supervised_level_count = _supervised_loss(
                    forward,
                    forward_target,
                    target_valid_level & first_valid_level,
                    self.epsilon,
                    self.exponent,
                )
            if target_backward is not None:
                backward_target = resize_displacement_2d(target_backward, shape)
                target_valid_level = _resize_mask(target_mask, shape)
                backward_supervised, backward_supervised_count = _supervised_loss(
                    backward,
                    backward_target,
                    target_valid_level & second_valid_level,
                    self.epsilon,
                    self.exponent,
                )
                combined_count = supervised_level_count + backward_supervised_count
                combined_numerator = (
                    supervised_value * supervised_level_count
                    + backward_supervised * backward_supervised_count
                )
                combined_denominator = jnp.maximum(
                    combined_count, jnp.asarray(1, dtype=combined_count.dtype)
                )
                supervised_value = jnp.where(
                    combined_count > 0,
                    combined_numerator / combined_denominator,
                    0.0,
                )
                supervised_level_count = combined_count

            supervised = supervised + scale_weight * supervised_value
            photometric = photometric + scale_weight * photo_value
            consistency = consistency + scale_weight * consistency_value
            smoothness = smoothness + scale_weight * smoothness_value
            supervised_count = supervised_count + supervised_level_count
            photometric_count = photometric_count + photo_count
            consistency_count = consistency_count + consistency_level_count
            smoothness_count = smoothness_count + smoothness_level_count

        total = (
            self.supervised_weight * supervised
            + self.photometric_weight * photometric
            + self.consistency_weight * consistency
            + self.smoothness_weight * smoothness
        )
        supporting_samples = (
            supervised_count + photometric_count + consistency_count + smoothness_count
        )
        valid_result = supporting_samples > 0
        status = jnp.where(
            valid_result,
            PIV_LOSS_SUCCESS,
            PIV_LOSS_INSUFFICIENT_SUPPORT,
        ).astype(jnp.int32)
        return PIVLossResult(
            total=total,
            supervised=supervised,
            photometric=photometric,
            consistency=consistency,
            smoothness=smoothness,
            supervised_valid_count=supervised_count,
            photometric_valid_count=photometric_count,
            consistency_valid_count=consistency_count,
            smoothness_valid_count=smoothness_count,
            valid=valid_result,
            status=status,
        )


__all__ = [
    "BackwardWarpResult",
    "CostVolumePlan",
    "CostVolumeResult",
    "PIV_LOSS_INSUFFICIENT_SUPPORT",
    "PIV_LOSS_SUCCESS",
    "MultiScaleRobustPIVLoss",
    "PIVLossResult",
    "backward_warp_2d",
    "build_cost_volume_2d",
    "resize_displacement_2d",
]
