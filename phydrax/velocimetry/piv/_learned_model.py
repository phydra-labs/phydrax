#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key
from opt_einsum import contract

from ..._doc import DOC_KEY0
from ..._fingerprint import canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState
from ..imaging._types import DenseDisplacementField2D, ImagePair2D
from ..imaging._warp import image_coordinates
from ._learned_primitives import (
    backward_warp_2d,
    build_cost_volume_2d,
    CostVolumePlan,
    resize_displacement_2d,
)


class LearnedDensePIVPlan(StrictModule, NonTrainableState):
    """Static image pyramid, correlation lattice, and memory execution contract."""

    cost_volume: CostVolumePlan
    image_shape: tuple[int, int] = eqx.field(static=True)
    level_shapes: tuple[tuple[int, int], ...] = eqx.field(static=True)
    input_channels: int = eqx.field(static=True)
    level_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        image_shape: tuple[int, int],
        /,
        *,
        input_channels: int = 1,
        level_count: int = 3,
        search_radius: int = 2,
        cost_volume_chunk_size: int = 16,
    ):
        rows, columns = (int(image_shape[0]), int(image_shape[1]))
        channels = int(input_channels)
        levels = int(level_count)
        if rows <= 0 or columns <= 0:
            raise ValueError("image_shape entries must be positive.")
        if channels <= 0:
            raise ValueError("input_channels must be positive.")
        if levels <= 0:
            raise ValueError("level_count must be positive.")
        if min(rows, columns) // (2 ** (levels - 1)) < 2:
            raise ValueError(
                "Every requested pyramid level must contain at least two rows and columns."
            )
        level_shapes = tuple(
            (
                max(2, rows // (2**level)),
                max(2, columns // (2**level)),
            )
            for level in range(levels)
        )
        cost_plan = CostVolumePlan(search_radius, chunk_size=cost_volume_chunk_size)
        self.cost_volume = cost_plan
        self.image_shape = (rows, columns)
        self.level_shapes = level_shapes
        self.input_channels = channels
        self.level_count = levels
        self.plan_id = canonical_fingerprint(
            {
                "kind": "learned-dense-piv-plan",
                "image_shape": self.image_shape,
                "input_channels": channels,
                "level_shapes": level_shapes,
                "cost_volume_plan_id": cost_plan.plan_id,
            }
        )

    def prepare(
        self,
        first_image: ArrayLike,
        second_image: ArrayLike,
        /,
        *,
        first_valid: ArrayLike | None = None,
        second_valid: ArrayLike | None = None,
    ) -> "PreparedLearnedDensePIV":
        """Build one immutable fine-to-coarse image and strict-mask pyramid."""

        first = jnp.asarray(first_image)
        second = jnp.asarray(second_image)
        expected = self.image_shape + (self.input_channels,)
        if first.shape != expected or second.shape != expected:
            raise ValueError(f"Prepared images must both have shape {expected}.")
        if jnp.issubdtype(first.dtype, jnp.complexfloating) or jnp.issubdtype(
            second.dtype, jnp.complexfloating
        ):
            raise TypeError("Prepared PIV images must be real-valued.")
        if not jnp.issubdtype(first.dtype, jnp.inexact) or not jnp.issubdtype(
            second.dtype, jnp.inexact
        ):
            first = first.astype(float)
            second = second.astype(float)
        dtype = jnp.result_type(first.dtype, second.dtype)
        first = first.astype(dtype)
        second = second.astype(dtype)
        first_mask = (
            jnp.ones(self.image_shape, dtype=bool)
            if first_valid is None
            else jnp.asarray(first_valid, dtype=bool)
        )
        second_mask = (
            jnp.ones(self.image_shape, dtype=bool)
            if second_valid is None
            else jnp.asarray(second_valid, dtype=bool)
        )
        if first_mask.shape != self.image_shape or second_mask.shape != self.image_shape:
            raise ValueError("Prepared image masks must match plan.image_shape.")
        first_mask = first_mask & jnp.all(jnp.isfinite(first), axis=-1)
        second_mask = second_mask & jnp.all(jnp.isfinite(second), axis=-1)
        first = jnp.where(first_mask[..., None], first, 0.0)
        second = jnp.where(second_mask[..., None], second, 0.0)

        first_pyramid: list[Array] = []
        second_pyramid: list[Array] = []
        first_valid_pyramid: list[Array] = []
        second_valid_pyramid: list[Array] = []
        tolerance = 8.0 * jnp.finfo(dtype).eps
        for shape in self.level_shapes:
            output_shape = shape + (self.input_channels,)
            first_level = (
                first
                if shape == self.image_shape
                else jax.image.resize(
                    first, output_shape, method="linear", antialias=True
                )
            )
            second_level = (
                second
                if shape == self.image_shape
                else jax.image.resize(
                    second, output_shape, method="linear", antialias=True
                )
            )
            if shape == self.image_shape:
                first_level_valid = first_mask
                second_level_valid = second_mask
            else:
                first_support = jax.image.resize(
                    first_mask.astype(dtype), shape, method="linear", antialias=True
                )
                second_support = jax.image.resize(
                    second_mask.astype(dtype), shape, method="linear", antialias=True
                )
                first_level_valid = first_support >= 1.0 - tolerance
                second_level_valid = second_support >= 1.0 - tolerance
            first_pyramid.append(
                jnp.where(first_level_valid[..., None], first_level, 0.0)
            )
            second_pyramid.append(
                jnp.where(second_level_valid[..., None], second_level, 0.0)
            )
            first_valid_pyramid.append(first_level_valid)
            second_valid_pyramid.append(second_level_valid)

        return PreparedLearnedDensePIV(
            first_pyramid=tuple(first_pyramid),
            second_pyramid=tuple(second_pyramid),
            first_valid_pyramid=tuple(first_valid_pyramid),
            second_valid_pyramid=tuple(second_valid_pyramid),
            plan_id=self.plan_id,
        )


class PreparedLearnedDensePIV(StrictModule, NonTrainableState):
    """One pair realized on the exact fine-to-coarse execution pyramid."""

    first_pyramid: tuple[Array, ...]
    second_pyramid: tuple[Array, ...]
    first_valid_pyramid: tuple[Array, ...]
    second_valid_pyramid: tuple[Array, ...]
    plan_id: str = eqx.field(static=True)


class DensePIVPrediction(StrictModule):
    """Coarse-to-fine first-to-second row/column displacements and support."""

    displacement_pyramid_rc: tuple[Array, ...]
    valid_pyramid: tuple[Array, ...]
    architecture_id: str = eqx.field(static=True)

    @property
    def displacement_rc(self) -> Array:
        return self.displacement_pyramid_rc[-1]

    @property
    def valid(self) -> Array:
        return self.valid_pyramid[-1]


class LearnedDensePIVResult(StrictModule):
    """A learned prediction neutrally adapted to the canonical dense field type."""

    prediction: DensePIVPrediction
    field: DenseDisplacementField2D
    pair_id: str = eqx.field(static=True)


class AbstractDensePIVModel(StrictModule):
    """Dense PIV model contract over an explicitly prepared fixed pyramid."""

    plan: AbstractAttribute[LearnedDensePIVPlan]
    architecture_id: AbstractAttribute[str]

    @abstractmethod
    def __call__(self, prepared: PreparedLearnedDensePIV, /) -> DensePIVPrediction:
        raise NotImplementedError

    def prepare_pair(self, pair: ImagePair2D, /) -> PreparedLearnedDensePIV:
        if not isinstance(pair, ImagePair2D):
            raise TypeError("pair must be an ImagePair2D.")
        if pair.geometry.image_shape != self.plan.image_shape:
            raise ValueError("Image pair geometry does not match the learned PIV plan.")
        first = pair.first
        second = pair.second
        if first.ndim == 2:
            first = first[..., None]
        if second.ndim == 2:
            second = second[..., None]
        return self.plan.prepare(
            first,
            second,
            first_valid=pair.first_mask,
            second_valid=pair.second_mask,
        )

    def predict_pair(self, pair: ImagePair2D, /) -> LearnedDensePIVResult:
        prediction = self(self.prepare_pair(pair))
        provenance = tuple(pair.provenance) + (
            f"image-pair:{pair.pair_id}",
            f"learned-piv-architecture:{self.architecture_id}",
        )
        field = DenseDisplacementField2D(
            image_coordinates(pair.geometry),
            prediction.displacement_rc,
            prediction.valid,
            geometry_id=pair.geometry.geometry_id,
            provenance=provenance,
        )
        return LearnedDensePIVResult(
            prediction=prediction,
            field=field,
            pair_id=pair.pair_id,
        )


class _ChannelLastConv2D(StrictModule):
    weight: Array
    bias: Array
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        /,
        *,
        key: Key[Array, ""],
        final_scale: float = 1.0,
    ):
        input_count = int(in_channels)
        output_count = int(out_channels)
        kernel = int(kernel_size)
        if input_count <= 0 or output_count <= 0 or kernel <= 0 or kernel % 2 == 0:
            raise ValueError(
                "Convolution channel counts and odd kernel size must be positive."
            )
        fan_in = kernel * kernel * input_count
        fan_out = kernel * kernel * output_count
        limit = math.sqrt(6.0 / (fan_in + fan_out)) * float(final_scale)
        self.weight = jr.uniform(
            key,
            (kernel, kernel, input_count, output_count),
            minval=-limit,
            maxval=limit,
        )
        self.bias = jnp.zeros((output_count,), dtype=self.weight.dtype)
        self.in_channels = input_count
        self.out_channels = output_count
        self.kernel_size = kernel

    def __call__(self, values: Array, /) -> Array:
        if values.ndim != 3 or values.shape[-1] != self.in_channels:
            raise ValueError(
                f"Convolution input must have shape (rows, columns, {self.in_channels})."
            )
        convolved = jax.lax.conv_general_dilated(
            values[None, ...],
            self.weight,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )[0]
        return convolved + self.bias


class _SharedFeaturePyramid(StrictModule):
    input_projection: _ChannelLastConv2D
    hidden_projection: _ChannelLastConv2D
    output_projection: _ChannelLastConv2D

    def __init__(
        self,
        input_channels: int,
        feature_channels: int,
        /,
        *,
        key: Key[Array, ""],
    ):
        keys = jr.split(key, 3)
        self.input_projection = _ChannelLastConv2D(
            input_channels, feature_channels, 3, key=keys[0]
        )
        self.hidden_projection = _ChannelLastConv2D(
            feature_channels, feature_channels, 3, key=keys[1]
        )
        self.output_projection = _ChannelLastConv2D(
            feature_channels, feature_channels, 3, key=keys[2]
        )

    def __call__(self, image: Array, /) -> Array:
        features = jax.nn.silu(self.input_projection(image))
        features = jax.nn.silu(self.hidden_projection(features))
        features = self.output_projection(features)
        squared_norm = contract("hwc,hwc->hw", features, features)
        denominator = jnp.sqrt(
            jnp.maximum(
                squared_norm,
                jnp.asarray(jnp.finfo(features.dtype).tiny, dtype=features.dtype),
            )
        )
        normalized = features / denominator[..., None]
        return jnp.where((squared_norm > 0.0)[..., None], normalized, 0.0)


class _SharedResidualRefinement(StrictModule):
    input_projection: _ChannelLastConv2D
    hidden_projection: _ChannelLastConv2D
    output_projection: _ChannelLastConv2D
    radius: int = eqx.field(static=True)

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        radius: int,
        /,
        *,
        key: Key[Array, ""],
    ):
        keys = jr.split(key, 3)
        self.input_projection = _ChannelLastConv2D(
            input_channels, hidden_channels, 3, key=keys[0]
        )
        self.hidden_projection = _ChannelLastConv2D(
            hidden_channels, hidden_channels, 3, key=keys[1]
        )
        self.output_projection = _ChannelLastConv2D(
            hidden_channels, 2, 3, key=keys[2], final_scale=0.05
        )
        self.radius = int(radius)

    def __call__(self, inputs: Array, /) -> Array:
        hidden = jax.nn.silu(self.input_projection(inputs))
        hidden = jax.nn.silu(self.hidden_projection(hidden))
        radius = jnp.asarray(self.radius, dtype=hidden.dtype)
        return radius * jnp.tanh(self.output_projection(hidden))


class CorrelationPyramidPIV(AbstractDensePIVModel):
    """Shared-feature coarse-to-fine local-correlation displacement model.

    This is a native compact architecture: one feature extractor and one residual
    refiner are shared across all physical pyramid levels. Local cost volumes are
    evaluated directly around the upsampled first-to-second displacement, without
    periodic padding or an imported architecture, parameter source, or weights.
    """

    plan: LearnedDensePIVPlan
    feature_pyramid: _SharedFeaturePyramid
    residual_refinement: _SharedResidualRefinement
    feature_channels: int = eqx.field(static=True)
    refinement_channels: int = eqx.field(static=True)
    architecture_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: LearnedDensePIVPlan,
        /,
        *,
        feature_channels: int = 16,
        refinement_channels: int = 32,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        if not isinstance(plan, LearnedDensePIVPlan):
            raise TypeError("plan must be a LearnedDensePIVPlan.")
        features = int(feature_channels)
        refinement = int(refinement_channels)
        if features <= 0 or refinement <= 0:
            raise ValueError("Feature and refinement channel counts must be positive.")
        keys = jr.split(key, 2)
        cost_channels = plan.cost_volume.offset_count
        self.plan = plan
        self.feature_pyramid = _SharedFeaturePyramid(
            plan.input_channels, features, key=keys[0]
        )
        self.residual_refinement = _SharedResidualRefinement(
            cost_channels + features + 2,
            refinement,
            plan.cost_volume.radius,
            key=keys[1],
        )
        self.feature_channels = features
        self.refinement_channels = refinement
        self.architecture_id = canonical_fingerprint(
            {
                "kind": "native-correlation-pyramid-piv",
                "plan_id": plan.plan_id,
                "feature_channels": features,
                "refinement_channels": refinement,
                "feature_layers": 3,
                "feature_kernel_size": 3,
                "feature_activation": "silu",
                "feature_normalization": "per-pixel-l2",
                "refinement_layers": 3,
                "refinement_kernel_size": 3,
                "refinement_activation": "silu-tanh-output",
                "feature_sharing": "all-levels",
                "refinement_sharing": "all-levels",
                "displacement_components": "row-column",
            }
        )

    def parameter_state_tree(self, /) -> Any:
        return {
            "feature_pyramid": self.feature_pyramid,
            "residual_refinement": self.residual_refinement,
        }

    def __call__(self, prepared: PreparedLearnedDensePIV, /) -> DensePIVPrediction:
        if not isinstance(prepared, PreparedLearnedDensePIV):
            raise TypeError("prepared must be a PreparedLearnedDensePIV.")
        if prepared.plan_id != self.plan.plan_id:
            raise ValueError(
                "Prepared inputs and model must share the exact learned PIV plan."
            )
        if len(prepared.first_pyramid) != self.plan.level_count:
            raise ValueError(
                "Prepared pyramid level count does not match the model plan."
            )

        displacement: Array | None = None
        displacements: list[Array] = []
        validity: list[Array] = []
        for level in range(self.plan.level_count - 1, -1, -1):
            first = prepared.first_pyramid[level]
            second = prepared.second_pyramid[level]
            first_valid = prepared.first_valid_pyramid[level]
            second_valid = prepared.second_valid_pyramid[level]
            shape = self.plan.level_shapes[level]
            if displacement is None:
                displacement = jnp.zeros(shape + (2,), dtype=first.dtype)
            else:
                displacement = resize_displacement_2d(displacement, shape)

            first_features = self.feature_pyramid(first)
            second_features = self.feature_pyramid(second)
            cost_volume = build_cost_volume_2d(
                first_features,
                second_features,
                self.plan.cost_volume,
                base_displacement_rc=displacement,
                reference_valid=first_valid,
                target_valid=second_valid,
            )
            cost_valid = jnp.any(cost_volume.valid, axis=-1)
            refinement_input = jnp.concatenate(
                (
                    first_features,
                    cost_volume.values,
                    displacement,
                ),
                axis=-1,
            )
            residual = self.residual_refinement(refinement_input)
            displacement = displacement + jnp.where(cost_valid[..., None], residual, 0.0)
            target_at_prediction = backward_warp_2d(
                second,
                -displacement,
                valid_mask=second_valid,
            )
            level_valid = first_valid & target_at_prediction.valid
            displacement = jnp.where(level_valid[..., None], displacement, 0.0)
            displacements.append(displacement)
            validity.append(level_valid)

        return DensePIVPrediction(
            displacement_pyramid_rc=tuple(displacements),
            valid_pyramid=tuple(validity),
            architecture_id=self.architecture_id,
        )


__all__ = [
    "AbstractDensePIVModel",
    "CorrelationPyramidPIV",
    "DensePIVPrediction",
    "LearnedDensePIVPlan",
    "LearnedDensePIVResult",
    "PreparedLearnedDensePIV",
]
