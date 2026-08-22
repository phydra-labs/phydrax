#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, cast

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._frozendict import frozendict
from .._strict import StrictModule


class PointIntegrationBatch(StrictModule):
    """Coupled deterministic or sampled points with an explicit weight field."""

    points: Any
    weights: cx.Field
    axes: tuple[str, ...] = eqx.field(static=True)
    mask: cx.Field | None
    target_mass: Array | None
    stratum_indices: Array | None
    num_strata: int | None = eqx.field(static=True)
    provenance: str = eqx.field(static=True)

    def __init__(
        self,
        points: Any,
        weights: cx.Field,
        /,
        *,
        axes: tuple[str, ...],
        mask: cx.Field | None = None,
        target_mass: Array | None = None,
        stratum_indices: Array | None = None,
        num_strata: int | None = None,
        provenance: str = "fixed",
    ):
        if not isinstance(weights, cx.Field):
            raise TypeError("PointIntegrationBatch weights must be a coordax.Field.")
        missing = tuple(axis for axis in axes if axis not in weights.named_dims)
        if missing:
            raise ValueError(f"Point integration weights are missing axes {missing!r}.")
        if mask is not None and not isinstance(mask, cx.Field):
            raise TypeError("PointIntegrationBatch mask must be a coordax.Field or None.")
        self.points = points
        self.weights = weights
        self.axes = tuple(axes)
        self.mask = mask
        self.target_mass = target_mass
        if stratum_indices is not None:
            indices = jnp.asarray(stratum_indices, dtype=jnp.int32).reshape((-1,))
            if indices.size != weights.data.size:
                raise ValueError("stratum_indices must have one entry per point weight.")
            if num_strata is None or int(num_strata) < 1:
                raise ValueError(
                    "num_strata must be positive when stratum_indices are supplied."
                )
            self.stratum_indices = indices
            self.num_strata = int(num_strata)
        else:
            self.stratum_indices = None
            self.num_strata = None
        self.provenance = str(provenance)


class SeparableIntegrationBatch(StrictModule):
    """Named-axis tensor weights without materializing their Cartesian product."""

    points: Any
    weights_by_axis: frozendict[str, cx.Field]
    axes: tuple[str, ...] = eqx.field(static=True)
    coupled_weight: cx.Field | None
    mask: cx.Field | None
    target_mass: Array | None
    provenance: str = eqx.field(static=True)

    def __init__(
        self,
        points: Any,
        weights_by_axis: dict[str, cx.Field] | frozendict[str, cx.Field],
        /,
        *,
        axes: tuple[str, ...] | None = None,
        coupled_weight: cx.Field | None = None,
        mask: cx.Field | None = None,
        target_mass: Array | None = None,
        provenance: str = "fixed-separable",
    ):
        weights = frozendict(weights_by_axis)
        axes_ = tuple(weights) if axes is None else tuple(axes)
        missing = tuple(axis for axis in axes_ if axis not in weights)
        if missing:
            raise ValueError(
                f"Separable integration weights are missing axes {missing!r}."
            )
        for axis, weight in weights.items():
            if not isinstance(weight, cx.Field) or weight.dims != (axis,):
                raise ValueError(
                    f"weights_by_axis[{axis!r}] must have exactly dims={(axis,)!r}."
                )
        self.points = points
        self.weights_by_axis = weights
        self.axes = axes_
        self.coupled_weight = coupled_weight
        self.mask = mask
        self.target_mass = target_mass
        self.provenance = str(provenance)

    def total_weight(self) -> cx.Field:
        total = cx.Field(jnp.asarray(1.0), dims=())
        for axis in self.axes:
            total = total * self.weights_by_axis[axis]
        if self.coupled_weight is not None:
            total = total * self.coupled_weight
        return total


class MappedIntegrationBatch(StrictModule):
    """Reference points mapped into physical coordinates with Jacobian weights."""

    reference_points: Array
    points: Any
    weights: Array
    mask: Array
    target_mass: Array | None
    axis: str = eqx.field(static=True)
    cell: str = eqx.field(static=True)
    provenance: str = eqx.field(static=True)

    def __init__(
        self,
        reference_points: Array,
        points: Any,
        weights: Array,
        /,
        *,
        mask: Array | None = None,
        target_mass: Array | None = None,
        axis: str = "__integration_point__",
        cell: str = "mapped",
        provenance: str = "mapped",
    ):
        reference = jnp.asarray(reference_points, dtype=float)
        weights_ = jnp.asarray(weights, dtype=float).reshape((-1,))
        if reference.ndim < 2 or reference.shape[0] != weights_.shape[0]:
            raise ValueError(
                "Mapped reference points and weights must share a point axis."
            )
        mask_ = (
            jnp.ones(weights_.shape, dtype=bool)
            if mask is None
            else jnp.asarray(mask, dtype=bool).reshape(weights_.shape)
        )
        self.reference_points = reference
        self.points = points
        self.weights = weights_
        self.mask = mask_
        self.target_mass = target_mass
        self.axis = str(axis)
        self.cell = str(cell)
        self.provenance = str(provenance)


class WeightedSampleBatch(StrictModule):
    """Masked log-weighted samples with explicit axes and design provenance."""

    samples: Any
    log_weights: Array | cx.Field
    mask: Array | cx.Field | None
    target_mass: Array | None
    support_valid: Array | None
    stratum_ids: Array | None
    pair_ids: Array | None
    replicate_ids: Array | None
    ancestry_ids: Array | cx.Field | None
    sample_axes: tuple[int, ...] | tuple[str, ...] = eqx.field(static=True)
    provenance: str = eqx.field(static=True)
    independent: bool = eqx.field(static=True)

    def __init__(
        self,
        samples: Any,
        log_weights: Array | cx.Field,
        /,
        *,
        mask: Array | cx.Field | None = None,
        target_mass: Array | None = None,
        support_valid: Array | None = None,
        stratum_ids: Array | None = None,
        pair_ids: Array | None = None,
        replicate_ids: Array | None = None,
        ancestry_ids: Array | cx.Field | None = None,
        sample_axes: int | str | tuple[int, ...] | tuple[str, ...] = 0,
        provenance: str = "external",
        independent: bool = False,
    ):
        raw_axes = sample_axes if isinstance(sample_axes, tuple) else (sample_axes,)
        if not raw_axes:
            raise ValueError("sample_axes must contain at least one axis.")
        if isinstance(log_weights, cx.Field):
            if not all(isinstance(axis, str) and axis for axis in raw_axes):
                raise TypeError("Named log-weight fields require named sample_axes.")
            axes: tuple[int, ...] | tuple[str, ...] = tuple(
                str(axis) for axis in raw_axes
            )
            missing = tuple(axis for axis in axes if axis not in log_weights.named_dims)
            if missing:
                raise ValueError(f"log_weights is missing sample axes {missing!r}.")
            if any(dim is None for dim in log_weights.dims):
                raise ValueError("Named log-weight fields must name every dimension.")
            log_weights_ = log_weights
            if mask is not None:
                if isinstance(mask, cx.Field):
                    if set(mask.named_dims) - set(log_weights.named_dims):
                        raise ValueError(
                            "mask dimensions must be present in log_weights."
                        )
                    mask = cx.Field(
                        jnp.asarray(mask.broadcast_like(log_weights).data, dtype=bool),
                        dims=log_weights.dims,
                    )
                else:
                    mask = cx.Field(
                        jnp.broadcast_to(
                            jnp.asarray(mask, dtype=bool), log_weights.shape
                        ),
                        dims=log_weights.dims,
                    )
            if ancestry_ids is not None:
                if isinstance(ancestry_ids, cx.Field):
                    if set(ancestry_ids.named_dims) - set(log_weights.named_dims):
                        raise ValueError(
                            "ancestry_ids dimensions must be present in log_weights."
                        )
                    ancestry_ids = cx.Field(
                        jnp.asarray(
                            ancestry_ids.broadcast_like(log_weights).data,
                            dtype=jnp.int32,
                        ),
                        dims=log_weights.dims,
                    )
                else:
                    ancestry_ids = cx.Field(
                        jnp.broadcast_to(
                            jnp.asarray(ancestry_ids, dtype=jnp.int32),
                            log_weights.shape,
                        ),
                        dims=log_weights.dims,
                    )
            sample_count = 1
            for axis in axes:
                sample_count *= int(log_weights.named_shape[axis])
        else:
            log_weights_ = jnp.asarray(log_weights, dtype=float)
            if log_weights_.ndim < 1:
                raise ValueError("Weighted samples require at least one weight axis.")
            if not all(isinstance(axis, int) for axis in raw_axes):
                raise TypeError("Raw log-weight arrays require integer sample_axes.")
            resolved = tuple(
                int(axis) + log_weights_.ndim if int(axis) < 0 else int(axis)
                for axis in raw_axes
            )
            if any(axis < 0 or axis >= log_weights_.ndim for axis in resolved):
                raise ValueError("sample_axes contains an out-of-range axis.")
            if len(set(resolved)) != len(resolved):
                raise ValueError("sample_axes must not contain duplicates.")
            axes = resolved
            if mask is not None:
                mask_data = mask.data if isinstance(mask, cx.Field) else mask
                mask = jnp.broadcast_to(
                    jnp.asarray(mask_data, dtype=bool), log_weights_.shape
                )
            if ancestry_ids is not None:
                ancestry_data = (
                    ancestry_ids.data
                    if isinstance(ancestry_ids, cx.Field)
                    else ancestry_ids
                )
                ancestry_ids = jnp.broadcast_to(
                    jnp.asarray(ancestry_data, dtype=jnp.int32),
                    log_weights_.shape,
                )
            sample_count = 1
            for axis in axes:
                sample_count *= int(log_weights_.shape[axis])

        def identifiers(value: Array | None, name: str) -> Array | None:
            if value is None:
                return None
            result = jnp.asarray(value, dtype=jnp.int32).reshape((-1,))
            if result.shape != (sample_count,):
                raise ValueError(f"{name} must have one entry per sampled unit.")
            return result

        provenance_ = str(provenance)
        if not provenance_:
            raise ValueError("provenance must be non-empty.")
        self.samples = samples
        self.log_weights = log_weights_
        self.mask = mask
        if target_mass is None:
            self.target_mass = None
        else:
            mass = jnp.asarray(target_mass, dtype=float)
            if bool(jnp.any(~jnp.isfinite(mass) | (mass <= 0.0))):
                raise ValueError("target_mass must be finite and strictly positive.")
            self.target_mass = mass
        self.support_valid = (
            None if support_valid is None else jnp.asarray(support_valid, dtype=bool)
        )
        self.stratum_ids = identifiers(stratum_ids, "stratum_ids")
        self.pair_ids = identifiers(pair_ids, "pair_ids")
        self.replicate_ids = identifiers(replicate_ids, "replicate_ids")
        self.ancestry_ids = ancestry_ids
        self.sample_axes = axes
        self.provenance = provenance_
        self.independent = bool(independent)

    @property
    def num_samples(self) -> int:
        count = 1
        if isinstance(self.log_weights, cx.Field):
            axes = cast(tuple[str, ...], self.sample_axes)
            for axis in axes:
                count *= int(self.log_weights.named_shape[axis])
        else:
            axes = cast(tuple[int, ...], self.sample_axes)
            for axis in axes:
                count *= int(self.log_weights.shape[axis])
        return count


IntegrationBatch = (
    PointIntegrationBatch
    | SeparableIntegrationBatch
    | MappedIntegrationBatch
    | WeightedSampleBatch
)


__all__ = [
    "IntegrationBatch",
    "MappedIntegrationBatch",
    "PointIntegrationBatch",
    "SeparableIntegrationBatch",
    "WeightedSampleBatch",
]
