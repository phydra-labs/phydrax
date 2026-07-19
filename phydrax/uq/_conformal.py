#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal

import coordax as cx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._predictive import PredictionInterval


class SplitConformal(StrictModule):
    """Exact finite-sample split conformal intervals for scalar exchangeable cases."""

    radius: Array
    alpha: float

    def __init__(self, radius: ArrayLike, alpha: float):
        radius_array = jnp.asarray(radius, dtype=float).reshape(())
        _validate_radius(radius_array)
        self.radius = radius_array
        self.alpha = _validate_alpha(alpha)

    @classmethod
    def calibrate(
        cls,
        center: cx.Field | ArrayLike,
        target: cx.Field | ArrayLike,
        /,
        *,
        alpha: float,
        case_dim: int | str = 0,
        mask: ArrayLike | None = None,
    ) -> "SplitConformal":
        _require_matching_field_structure(center, target)
        center_array, axis = _array_and_case_axis(center, case_dim)
        target_array, target_axis = _array_and_case_axis(target, case_dim)
        if center_array.shape != target_array.shape or axis != target_axis:
            raise ValueError("center and target must have matching case-aligned shapes.")
        scores = jnp.moveaxis(jnp.abs(target_array - center_array), axis, 0)
        if scores.ndim != 1:
            raise ValueError(
                "SplitConformal requires one scalar score per case; use "
                "FunctionalConformal for field-valued cases."
            )
        scores = _masked_case_scores(scores, mask, original_axis=axis)
        return cls(_finite_sample_quantile(scores, alpha), alpha)

    def interval(self, center: cx.Field | ArrayLike, /) -> PredictionInterval:
        center_field = _as_field(center)
        center_data = jnp.asarray(center_field.data)
        return PredictionInterval(
            _field_like(center_field, center_data - self.radius),
            _field_like(center_field, center_data + self.radius),
            nominal_coverage=1.0 - self.alpha,
            simultaneous=False,
            calibrated=True,
        )


class NormalizedConformal(StrictModule):
    """Split conformal intervals normalized by a predicted positive scale."""

    radius: Array
    alpha: float
    min_scale: float

    def __init__(self, radius: ArrayLike, alpha: float, *, min_scale: float = 1e-8):
        radius_array = jnp.asarray(radius, dtype=float).reshape(())
        _validate_radius(radius_array)
        minimum = float(min_scale)
        if not math.isfinite(minimum) or minimum <= 0.0:
            raise ValueError("min_scale must be finite and positive.")
        self.radius = radius_array
        self.alpha = _validate_alpha(alpha)
        self.min_scale = minimum

    @classmethod
    def calibrate(
        cls,
        center: cx.Field | ArrayLike,
        scale: cx.Field | ArrayLike,
        target: cx.Field | ArrayLike,
        /,
        *,
        alpha: float,
        case_dim: int | str = 0,
        min_scale: float = 1e-8,
        mask: ArrayLike | None = None,
    ) -> "NormalizedConformal":
        _require_matching_field_structure(center, scale, target)
        center_array, axis = _array_and_case_axis(center, case_dim)
        scale_array, scale_axis = _array_and_case_axis(scale, case_dim)
        target_array, target_axis = _array_and_case_axis(target, case_dim)
        if (
            center_array.shape != target_array.shape
            or scale_array.shape != center_array.shape
            or axis != scale_axis
            or axis != target_axis
        ):
            raise ValueError("center, scale, and target must have matching shapes.")
        _validate_scale(scale_array)
        scores = jnp.moveaxis(
            jnp.abs(target_array - center_array) / jnp.maximum(scale_array, min_scale),
            axis,
            0,
        )
        if scores.ndim != 1:
            raise ValueError(
                "NormalizedConformal requires one scalar score per case; use "
                "FunctionalConformal for field-valued cases."
            )
        scores = _masked_case_scores(scores, mask, original_axis=axis)
        return cls(
            _finite_sample_quantile(scores, alpha),
            alpha,
            min_scale=min_scale,
        )

    def interval(
        self,
        center: cx.Field | ArrayLike,
        scale: cx.Field | ArrayLike,
        /,
    ) -> PredictionInterval:
        center_field = _as_field(center)
        scale_field = _as_field(scale)
        if (
            center_field.dims != scale_field.dims
            or center_field.data.shape != scale_field.data.shape
        ):
            raise ValueError("center and scale fields must have matching structure.")
        _validate_scale(jnp.asarray(scale_field.data, dtype=float))
        center_data = jnp.asarray(center_field.data)
        width = self.radius * jnp.maximum(jnp.asarray(scale_field.data), self.min_scale)
        return PredictionInterval(
            _field_like(center_field, center_data - width),
            _field_like(center_field, center_data + width),
            nominal_coverage=1.0 - self.alpha,
            simultaneous=False,
            calibrated=True,
        )


class FunctionalConformal(StrictModule):
    """Case-level conformal bands calibrated on whole fields or trajectories."""

    radius: Array
    alpha: float
    min_scale: float
    normalized: bool
    score: Literal["max", "l2"]

    def __init__(
        self,
        radius: ArrayLike,
        alpha: float,
        *,
        min_scale: float = 1e-8,
        normalized: bool,
        score: Literal["max", "l2"] = "max",
    ):
        radius_array = jnp.asarray(radius, dtype=float).reshape(())
        _validate_radius(radius_array)
        minimum = float(min_scale)
        if not math.isfinite(minimum) or minimum <= 0.0:
            raise ValueError("min_scale must be finite and positive.")
        if score not in ("max", "l2"):
            raise ValueError("score must be 'max' or 'l2'.")
        self.radius = radius_array
        self.alpha = _validate_alpha(alpha)
        self.min_scale = minimum
        self.normalized = bool(normalized)
        self.score = score

    @classmethod
    def calibrate(
        cls,
        center: cx.Field | ArrayLike,
        target: cx.Field | ArrayLike,
        /,
        *,
        alpha: float,
        case_dim: int | str = 0,
        scale: cx.Field | ArrayLike | None = None,
        min_scale: float = 1e-8,
        mask: ArrayLike | None = None,
        weights: ArrayLike | None = None,
        score: Literal["max", "l2"] = "max",
    ) -> "FunctionalConformal":
        _require_matching_field_structure(center, target, scale)
        if score == "max" and weights is not None:
            raise ValueError("weights are supported only for score='l2'.")
        center_array, axis = _array_and_case_axis(center, case_dim)
        target_array, target_axis = _array_and_case_axis(target, case_dim)
        if center_array.shape != target_array.shape or axis != target_axis:
            raise ValueError("center and target must have matching field-case shapes.")
        residual = jnp.abs(target_array - center_array)
        normalized = scale is not None
        if scale is not None:
            scale_array, scale_axis = _array_and_case_axis(scale, case_dim)
            if scale_array.shape != residual.shape or scale_axis != axis:
                raise ValueError("scale must match center and target structure.")
            _validate_scale(scale_array)
            residual = residual / jnp.maximum(scale_array, min_scale)
        residual = jnp.moveaxis(residual, axis, 0)
        if residual.ndim < 2:
            raise ValueError(
                "FunctionalConformal requires at least one physical dimension per case."
            )
        flat = residual.reshape((int(residual.shape[0]), -1))
        mask_flat = jnp.ones_like(flat, dtype=bool)
        if mask is not None:
            mask_array = jnp.asarray(mask, dtype=bool)
            if mask_array.shape != center_array.shape:
                mask_array = jnp.broadcast_to(mask_array, center_array.shape)
            mask_flat = jnp.moveaxis(mask_array, axis, 0).reshape(flat.shape)
        valid_case = jnp.any(mask_flat, axis=1)
        if score == "max":
            case_scores = jnp.max(jnp.where(mask_flat, flat, -jnp.inf), axis=1)
        elif score == "l2":
            weight_flat = jnp.ones_like(flat)
            if weights is not None:
                weight_array = jnp.asarray(weights, dtype=float)
                if bool(jnp.any(~jnp.isfinite(weight_array))) or bool(
                    jnp.any(weight_array < 0.0)
                ):
                    raise ValueError("weights must be finite and non-negative.")
                weight_array = jnp.broadcast_to(weight_array, center_array.shape)
                weight_flat = jnp.moveaxis(weight_array, axis, 0).reshape(flat.shape)
            effective = weight_flat * mask_flat
            denominator = jnp.sum(effective, axis=1)
            valid_case = valid_case & (denominator > 0.0)
            case_scores = jnp.sqrt(jnp.sum(effective * flat**2, axis=1) / denominator)
        else:
            raise ValueError("score must be 'max' or 'l2'.")
        case_scores = case_scores[valid_case]
        return cls(
            _finite_sample_quantile(case_scores, alpha),
            alpha,
            min_scale=min_scale,
            normalized=normalized,
            score=score,
        )

    def interval(
        self,
        center: cx.Field | ArrayLike,
        scale: cx.Field | ArrayLike | None = None,
        /,
    ) -> PredictionInterval:
        if self.score == "l2":
            raise ValueError(
                "An L2 conformal score defines a norm ball, not pointwise bounds."
            )
        center_field = _as_field(center)
        if self.normalized:
            if scale is None:
                raise ValueError("This functional calibrator requires a scale field.")
            scale_field = _as_field(scale)
            if (
                center_field.dims != scale_field.dims
                or center_field.data.shape != scale_field.data.shape
            ):
                raise ValueError("center and scale fields must have matching structure.")
            _validate_scale(jnp.asarray(scale_field.data, dtype=float))
            width = self.radius * jnp.maximum(
                jnp.asarray(scale_field.data), self.min_scale
            )
        else:
            if scale is not None:
                raise ValueError("This functional calibrator was fitted without scale.")
            width = self.radius
        center_data = jnp.asarray(center_field.data)
        return PredictionInterval(
            _field_like(center_field, center_data - width),
            _field_like(center_field, center_data + width),
            nominal_coverage=1.0 - self.alpha,
            simultaneous=self.score == "max",
            calibrated=True,
        )


def _finite_sample_quantile(scores: ArrayLike, alpha: float) -> Array:
    level = _validate_alpha(alpha)
    score_array = jnp.asarray(scores, dtype=float).reshape((-1,))
    count = int(score_array.shape[0])
    if count <= 0:
        raise ValueError("Calibration scores must be non-empty.")
    if bool(jnp.any(~jnp.isfinite(score_array))):
        raise ValueError("Calibration scores must be finite.")
    rank = int(math.ceil((count + 1) * (1.0 - level)))
    if rank > count:
        raise ValueError(
            "Requested miscoverage is too small for the calibration sample size: "
            f"rank={rank}, num_scores={count}."
        )
    return jnp.sort(score_array)[rank - 1]


def _validate_alpha(alpha: float) -> float:
    value = float(alpha)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("alpha must lie strictly between zero and one.")
    return value


def _validate_radius(radius: Array) -> None:
    if not bool(jnp.isfinite(radius)) or bool(radius < 0.0):
        raise ValueError("Conformal radius must be finite and non-negative.")


def _array_and_case_axis(
    value: cx.Field | ArrayLike,
    case_dim: int | str,
) -> tuple[Array, int]:
    if isinstance(value, cx.Field):
        array = jnp.asarray(value.data, dtype=float)
        if isinstance(case_dim, str):
            matches = [index for index, dim in enumerate(value.dims) if dim == case_dim]
            if len(matches) != 1:
                raise ValueError(
                    f"case_dim {case_dim!r} must identify one field dimension."
                )
            axis = matches[0]
        else:
            axis = int(case_dim)
    else:
        array = jnp.asarray(value, dtype=float)
        if isinstance(case_dim, str):
            raise TypeError("String case_dim requires a coordax.Field input.")
        axis = int(case_dim)
    if array.ndim == 0:
        raise ValueError("Calibration inputs require a case dimension.")
    if axis < -array.ndim or axis >= array.ndim:
        raise ValueError(
            f"case_dim axis {axis} is out of bounds for rank-{array.ndim} input."
        )
    axis %= array.ndim
    return array, axis


def _masked_case_scores(
    scores: Array, mask: ArrayLike | None, *, original_axis: int
) -> Array:
    if mask is None:
        return scores
    mask_array = jnp.asarray(mask, dtype=bool)
    if mask_array.ndim != 1:
        mask_array = jnp.moveaxis(mask_array, original_axis, 0).reshape(scores.shape)
    if mask_array.shape != scores.shape:
        raise ValueError("mask must select the scalar calibration cases.")
    return scores[mask_array]


def _require_matching_field_structure(
    *values: cx.Field | ArrayLike | None,
) -> None:
    fields = tuple(value for value in values if isinstance(value, cx.Field))
    if len(fields) < 2:
        return
    reference = fields[0]
    if any(
        field.dims != reference.dims or field.data.shape != reference.data.shape
        for field in fields[1:]
    ):
        raise ValueError("Calibration fields must have matching shapes and dimensions.")


def _validate_scale(scale: Array) -> None:
    if bool(jnp.any(~jnp.isfinite(scale))) or bool(jnp.any(scale < 0.0)):
        raise ValueError("scale must be finite and non-negative.")


def _as_field(value: cx.Field | ArrayLike) -> cx.Field:
    if isinstance(value, cx.Field):
        return value
    array = jnp.asarray(value, dtype=float)
    return cx.Field(array, dims=(None,) * array.ndim)


def _field_like(template: cx.Field, data: ArrayLike) -> cx.Field:
    return cx.Field(jnp.asarray(data, dtype=float), dims=template.dims)


__all__ = ["FunctionalConformal", "NormalizedConformal", "SplitConformal"]
