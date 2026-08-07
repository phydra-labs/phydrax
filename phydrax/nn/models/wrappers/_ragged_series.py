#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

import coordax as cx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from phydrax.domain import BatchEvaluator, PointBatch

from ...._callable import _ensure_special_kwonly_args
from ...._doc import DOC_KEY0
from ...._strict import StrictModule
from ..core._keys import EvalKey, split_eval_key


MaskedSeriesReduction = Literal["mean", "sum"]


class RaggedSeriesBatchInput(StrictModule):
    """Array payload passed from `RaggedSeriesModel` to user encoders."""

    static: Any | None
    series: Any
    time: Array
    mask: Array
    length: Array
    sample_index: Array | None
    sample_scale: Array | None

    def __init__(
        self,
        *,
        static: Any | None,
        series: Any,
        time: Array,
        mask: Array,
        length: Array,
        sample_index: Array | None = None,
        sample_scale: Array | None = None,
    ):
        self.static = static
        self.series = series
        self.time = jnp.asarray(time, dtype=float)
        self.mask = jnp.asarray(mask, dtype=bool)
        self.length = jnp.asarray(length, dtype=jnp.int32)
        self.sample_index = (
            None if sample_index is None else jnp.asarray(sample_index, dtype=jnp.int32)
        )
        self.sample_scale = (
            None if sample_scale is None else jnp.asarray(sample_scale, dtype=float)
        )


def _tree_to_feature_array(tree: Any, /, *, axis_rank: int, name: str) -> Array:
    flat = []
    for leaf in jax.tree_util.tree_leaves(tree):
        flat.append(jnp.asarray(leaf))
    if not flat:
        raise ValueError(f"{name} requires at least one array leaf.")

    first = flat[0]
    if first.ndim < int(axis_rank):
        raise ValueError(
            f"{name} leaves must have at least {axis_rank} leading axes; "
            f"got {first.shape}."
        )
    leading_shape = tuple(int(n) for n in first.shape[:axis_rank])
    parts: list[Array] = []
    for arr in flat:
        if arr.ndim < int(axis_rank):
            raise ValueError(
                f"{name} leaves must have at least {axis_rank} leading axes; "
                f"got {arr.shape}."
            )
        if tuple(int(n) for n in arr.shape[:axis_rank]) != leading_shape:
            raise ValueError(f"{name} leaves must share leading axes.")
        feature_size = 1
        for dim in arr.shape[axis_rank:]:
            feature_size *= int(dim)
        parts.append(arr.reshape(leading_shape + (feature_size,)))
    return jnp.concatenate(parts, axis=-1)


def _extract_payload(batch: PointBatch, label: str, /) -> RaggedSeriesBatchInput:
    payload = batch.points[label]
    if not isinstance(payload, Mapping):
        raise TypeError("RaggedSeriesModel expects a mapping payload for its label.")
    if "series" not in payload:
        raise KeyError("Ragged series payload is missing 'series'.")
    if "time" not in payload:
        raise KeyError("Ragged series payload is missing 'time'.")
    if "mask" not in payload:
        raise KeyError("Ragged series payload is missing 'mask'.")
    if "length" not in payload:
        raise KeyError("Ragged series payload is missing 'length'.")

    static = None
    if "static" in payload:
        static = jax.tree_util.tree_map(
            lambda x: x.data if isinstance(x, cx.Field) else x,
            payload["static"],
            is_leaf=lambda x: isinstance(x, cx.Field),
        )
    series = jax.tree_util.tree_map(
        lambda x: x.data if isinstance(x, cx.Field) else x,
        payload["series"],
        is_leaf=lambda x: isinstance(x, cx.Field),
    )
    time = payload["time"]
    mask = payload["mask"]
    length = payload["length"]
    if not isinstance(time, cx.Field):
        raise TypeError("Ragged series 'time' payload must be a coordax.Field.")
    if not isinstance(mask, cx.Field):
        raise TypeError("Ragged series 'mask' payload must be a coordax.Field.")
    if not isinstance(length, cx.Field):
        raise TypeError("Ragged series 'length' payload must be a coordax.Field.")
    sample_index = None
    if "sample_index" in payload:
        sample_index_field = payload["sample_index"]
        if not isinstance(sample_index_field, cx.Field):
            raise TypeError(
                "Ragged series 'sample_index' payload must be a coordax.Field."
            )
        sample_index = sample_index_field.data
    sample_scale = None
    if "sample_scale" in payload:
        sample_scale_field = payload["sample_scale"]
        if not isinstance(sample_scale_field, cx.Field):
            raise TypeError(
                "Ragged series 'sample_scale' payload must be a coordax.Field."
            )
        sample_scale = sample_scale_field.data

    return RaggedSeriesBatchInput(
        static=static,
        series=series,
        time=time.data,
        mask=mask.data,
        length=length.data,
        sample_index=sample_index,
        sample_scale=sample_scale,
    )


class RaggedSeriesModel(StrictModule, BatchEvaluator):
    """Wrap a ragged-series encoder as a Phydrax batch-aware `DomainFunction`."""

    model: Callable
    label: str

    def __init__(self, model: Callable, /, *, label: str = "data"):
        self.model = _ensure_special_kwonly_args(model)
        self.label = str(label)

    def __call_batch__(
        self,
        batch: Any,
        /,
        *,
        key: EvalKey = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        if not isinstance(batch, PointBatch):
            raise TypeError("RaggedSeriesModel requires PointBatch evaluation.")
        if self.label not in batch.points:
            raise KeyError(f"RaggedSeriesModel label {self.label!r} missing from batch.")
        axis = batch.structure.axis_for(self.label)
        if axis is None:
            raise ValueError(
                f"RaggedSeriesModel label {self.label!r} must be sampled on an axis."
            )

        payload = _extract_payload(batch, self.label)
        y = jnp.asarray(self.model(payload, key=key, **kwargs), dtype=float)
        if y.ndim == 0:
            raise ValueError("RaggedSeriesModel output must retain a leading case axis.")
        if int(y.shape[0]) != int(payload.length.shape[0]):
            raise ValueError(
                "RaggedSeriesModel output leading axis must match sampled case count."
            )
        return cx.Field(y, dims=(axis,) + (None,) * (y.ndim - 1))


class MaskedSeriesPoolingModel(StrictModule):
    """Encode variable-length series with a per-step model and masked reduction."""

    step_model: Callable
    readout_model: Callable
    reduction: MaskedSeriesReduction
    include_time: bool
    include_static_in_steps: bool
    include_static_in_readout: bool
    scale_sampled_sum: bool

    def __init__(
        self,
        *,
        step_model: Callable,
        readout_model: Callable,
        reduction: MaskedSeriesReduction = "mean",
        include_time: bool = True,
        include_static_in_steps: bool = False,
        include_static_in_readout: bool = True,
        scale_sampled_sum: bool = False,
    ):
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be either 'mean' or 'sum'.")
        self.step_model = _ensure_special_kwonly_args(step_model)
        self.readout_model = _ensure_special_kwonly_args(readout_model)
        self.reduction = reduction
        self.include_time = bool(include_time)
        self.include_static_in_steps = bool(include_static_in_steps)
        self.include_static_in_readout = bool(include_static_in_readout)
        self.scale_sampled_sum = bool(scale_sampled_sum)

    def __call__(
        self,
        x: RaggedSeriesBatchInput,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        series_features = _tree_to_feature_array(
            x.series,
            axis_rank=2,
            name="series",
        )
        parts = [series_features]
        if self.include_time:
            parts.append(jnp.asarray(x.time, dtype=float)[..., None])

        static_features: Array | None = None
        if x.static is not None:
            static_features = _tree_to_feature_array(
                x.static,
                axis_rank=1,
                name="static",
            )
        if self.include_static_in_steps:
            if static_features is None:
                raise ValueError("include_static_in_steps=True requires static data.")
            repeated_static = jnp.broadcast_to(
                static_features[:, None, :],
                series_features.shape[:2] + (int(static_features.shape[-1]),),
            )
            parts.append(repeated_static)

        step_input = jnp.concatenate(parts, axis=-1)
        key_step, key_readout = split_eval_key(key, 2)
        step_output = jnp.asarray(self.step_model(step_input, key=key_step), dtype=float)
        if step_output.ndim == 2:
            step_output = step_output[..., None]
        if step_output.ndim < 3:
            raise ValueError("step_model must return shape (N, L, ...) outputs.")
        if step_output.shape[:2] != step_input.shape[:2]:
            raise ValueError("step_model output must preserve case and time axes.")

        latent = step_output.reshape(step_output.shape[:2] + (-1,))
        mask = jnp.asarray(x.mask, dtype=bool)
        if mask.shape != step_input.shape[:2]:
            raise ValueError("mask must have shape (N, Lmax).")
        mask_f = mask.astype(latent.dtype)[..., None]
        pooled = jnp.sum(latent * mask_f, axis=1)
        if self.reduction == "mean":
            denom = jnp.maximum(
                jnp.sum(mask_f, axis=1), jnp.asarray(1.0, dtype=pooled.dtype)
            )
            pooled = pooled / denom
        elif self.scale_sampled_sum and x.sample_scale is not None:
            scale = jnp.asarray(x.sample_scale, dtype=pooled.dtype)
            pooled = pooled * scale[:, None]

        readout_parts = [pooled]
        if self.include_static_in_readout:
            if static_features is None:
                raise ValueError("include_static_in_readout=True requires static data.")
            readout_parts.append(static_features)
        readout_input = jnp.concatenate(readout_parts, axis=-1)
        return self.readout_model(readout_input, key=key_readout)


__all__ = [
    "MaskedSeriesPoolingModel",
    "MaskedSeriesReduction",
    "RaggedSeriesBatchInput",
    "RaggedSeriesModel",
]
