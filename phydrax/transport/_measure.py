#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Any, Callable, cast

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array

from .._numerics import log_normalize
from .._strict import StrictModule
from ..integration._targets import DiscreteMeasureTarget, WeightedSampleTarget


EventEncoder = Callable[[Any], Array | cx.Field]


class _FiniteTransportMeasure(StrictModule):
    """Canonical finite measure used internally by native transport."""

    points: Array
    probabilities: Array
    mass: Array
    active: Array
    event_shape: tuple[int, ...] = eqx.field(static=True)
    normalized: bool = eqx.field(static=True)
    provenance: str = eqx.field(static=True)

    def __init__(
        self,
        points: Array,
        probabilities: Array,
        mass: Array,
        active: Array,
        /,
        *,
        event_shape: tuple[int, ...],
        normalized: bool,
        provenance: str,
    ):
        points_ = jnp.asarray(points, dtype=float)
        probabilities_ = jnp.asarray(probabilities, dtype=float)
        active_ = jnp.asarray(active, dtype=bool)
        mass_ = jnp.asarray(mass, dtype=float).reshape(())
        if points_.ndim != 2 or points_.shape[0] == 0 or points_.shape[1] == 0:
            raise ValueError("Transport points must have nonempty shape (atom, feature).")
        if probabilities_.shape != (points_.shape[0],):
            raise ValueError("Transport probabilities must have one value per atom.")
        if active_.shape != probabilities_.shape:
            raise ValueError("Transport active mask must have one value per atom.")
        probabilities_ = eqx.error_if(
            probabilities_,
            jnp.any(~jnp.isfinite(probabilities_))
            | jnp.any(probabilities_ < 0.0)
            | ~jnp.isclose(jnp.sum(probabilities_), 1.0),
            "Transport probabilities must be finite, nonnegative, and sum to one.",
        )
        mass_ = eqx.error_if(
            mass_,
            ~jnp.isfinite(mass_) | (mass_ <= 0.0),
            "Transport mass must be finite and positive.",
        )
        points_ = eqx.error_if(
            points_,
            jnp.any(active_[:, None] & ~jnp.isfinite(points_)),
            "Active transport atoms must contain only finite coordinates.",
        )
        self.points = jnp.where(active_[:, None], points_, 0.0)
        self.probabilities = jnp.where(active_, probabilities_, 0.0)
        self.mass = mass_
        self.active = active_
        self.event_shape = tuple(int(size) for size in event_shape)
        self.normalized = bool(normalized)
        self.provenance = str(provenance)

    @property
    def physical_weights(self) -> Array:
        """Return physical atom weights."""
        return self.mass * self.probabilities

    @property
    def num_atoms(self) -> int:
        return int(self.points.shape[0])

    @property
    def feature_size(self) -> int:
        return int(self.points.shape[1])


def lower_transport_measure(
    target: DiscreteMeasureTarget | WeightedSampleTarget,
    /,
    *,
    encoder: EventEncoder | None,
    name: str,
) -> _FiniteTransportMeasure:
    """Lower a supported PhydraX measure to one unbatched finite measure."""
    if isinstance(target, DiscreteMeasureTarget):
        return _lower_discrete(target, encoder=encoder, name=name)
    if isinstance(target, WeightedSampleTarget):
        return _lower_weighted(target, encoder=encoder, name=name)
    raise TypeError(
        f"{name} must be a DiscreteMeasureTarget or WeightedSampleTarget; "
        f"got {type(target).__name__}."
    )


def _lower_discrete(
    target: DiscreteMeasureTarget,
    /,
    *,
    encoder: EventEncoder | None,
    name: str,
) -> _FiniteTransportMeasure:
    weights = _discrete_weight_field(target)
    axes = target.axes
    if any(dim is None for dim in weights.dims) or set(weights.dims) != set(axes):
        raise ValueError(
            f"{name} discrete weights must contain exactly axes={axes!r} with no "
            "retained case dimensions."
        )
    positions = tuple(weights.dims.index(axis) for axis in axes)
    atom_shape = tuple(int(weights.shape[position]) for position in positions)
    weight_values = jnp.transpose(jnp.asarray(weights.data, dtype=float), positions)
    if target.mask is None:
        included = jnp.ones(atom_shape, dtype=bool)
    else:
        mask_field = target.mask.broadcast_like(weights)
        included = jnp.transpose(jnp.asarray(mask_field.data, dtype=bool), positions)
    raw_points = encoder(target.points) if encoder is not None else target.points
    points, event_shape = _canonical_points_named_or_raw(
        raw_points,
        axes=axes,
        atom_shape=atom_shape,
        raw_weight_dims=weights.dims,
        name=f"{name} points",
    )
    probabilities, mass, active = _linear_probabilities_and_mass(
        weight_values.reshape((-1,)),
        included.reshape((-1,)),
        normalized=target.normalized,
        target_mass=target.target_mass,
        name=name,
    )
    return _FiniteTransportMeasure(
        points,
        probabilities,
        mass,
        active,
        event_shape=event_shape,
        normalized=target.normalized,
        provenance=target.provenance,
    )


def _lower_weighted(
    target: WeightedSampleTarget,
    /,
    *,
    encoder: EventEncoder | None,
    name: str,
) -> _FiniteTransportMeasure:
    if isinstance(target.log_weights, cx.Field):
        weights = target.log_weights
        axes = cast(tuple[str, ...], target.sample_axes)
        if any(dim is None for dim in weights.dims) or set(weights.dims) != set(axes):
            raise ValueError(
                f"{name} log weights must contain exactly sample_axes={axes!r} "
                "with no retained case dimensions."
            )
        positions = tuple(weights.dims.index(axis) for axis in axes)
        atom_shape = tuple(int(weights.shape[position]) for position in positions)
        log_weights = jnp.transpose(jnp.asarray(weights.data, dtype=float), positions)
        if target.mask is None:
            included = jnp.ones(atom_shape, dtype=bool)
        else:
            mask_field = cast(cx.Field, target.mask).broadcast_like(weights)
            included = jnp.transpose(
                jnp.asarray(mask_field.data, dtype=bool), positions
            )
        raw_samples = encoder(target.samples) if encoder is not None else target.samples
        points, event_shape = _canonical_points_named_or_raw(
            raw_samples,
            axes=axes,
            atom_shape=atom_shape,
            raw_weight_dims=weights.dims,
            name=f"{name} samples",
        )
    else:
        weights_array = jnp.asarray(target.log_weights, dtype=float)
        axes = cast(tuple[int, ...], target.sample_axes)
        if set(axes) != set(range(weights_array.ndim)):
            raise ValueError(
                f"{name} sample_axes must cover every log-weight dimension for an "
                "unbatched transport measure."
            )
        atom_shape = tuple(int(weights_array.shape[axis]) for axis in axes)
        log_weights = jnp.transpose(weights_array, axes)
        if target.mask is None:
            included_raw = jnp.ones(weights_array.shape, dtype=bool)
        elif isinstance(target.mask, cx.Field):
            included_raw = jnp.asarray(target.mask.data, dtype=bool)
        else:
            included_raw = jnp.asarray(target.mask, dtype=bool)
        included_raw = jnp.broadcast_to(included_raw, weights_array.shape)
        included = jnp.transpose(included_raw, axes)
        raw_samples = encoder(target.samples) if encoder is not None else target.samples
        points, event_shape = _canonical_points_raw(
            raw_samples,
            weight_shape=weights_array.shape,
            sample_positions=axes,
            name=f"{name} samples",
        )
    included = _combine_support_validity(
        included,
        target.support_valid,
        atom_shape=atom_shape,
        name=name,
    )
    probabilities, mass, active = _log_probabilities_and_mass(
        log_weights.reshape((-1,)),
        included.reshape((-1,)),
        normalized=target.normalized,
        target_mass=target.target_mass,
        name=name,
    )
    return _FiniteTransportMeasure(
        points,
        probabilities,
        mass,
        active,
        event_shape=event_shape,
        normalized=target.normalized,
        provenance=target.provenance,
    )


def _discrete_weight_field(target: DiscreteMeasureTarget, /) -> cx.Field:
    if isinstance(target.weights, cx.Field):
        return target.weights
    total = cx.Field(jnp.asarray(1.0), dims=())
    for axis in target.axes:
        total = total * target.weights[axis]
    return total


def _canonical_points_named_or_raw(
    value: Any,
    /,
    *,
    axes: tuple[str, ...],
    atom_shape: tuple[int, ...],
    raw_weight_dims: tuple[str | None, ...],
    name: str,
) -> tuple[Array, tuple[int, ...]]:
    leaf = _single_encoded_leaf(value, name=name)
    if isinstance(leaf, cx.Field):
        missing = tuple(axis for axis in axes if axis not in leaf.named_dims)
        if missing:
            raise ValueError(f"{name} is missing atom axes {missing!r}.")
        positions = tuple(leaf.dims.index(axis) for axis in axes)
        remaining = tuple(
            position for position in range(leaf.ndim) if position not in positions
        )
        canonical = jnp.transpose(jnp.asarray(leaf.data), positions + remaining)
        observed = tuple(int(size) for size in canonical.shape[: len(axes)])
        if observed != atom_shape:
            raise ValueError(
                f"{name} atom shape must be {atom_shape}; got {observed}."
            )
        return _flatten_events(canonical, atom_shape=atom_shape)
    data = jnp.asarray(leaf)
    if data.ndim < len(raw_weight_dims) or tuple(data.shape[: len(raw_weight_dims)]) != tuple(
        int(size) for size in atom_shape
    ):
        # Raw arrays paired with named weights follow the target axis order.
        if data.ndim < len(atom_shape) or tuple(data.shape[: len(atom_shape)]) != atom_shape:
            raise ValueError(
                f"{name} raw arrays must begin with atom shape {atom_shape}."
            )
        canonical = data
    else:
        positions = tuple(raw_weight_dims.index(axis) for axis in axes)
        outputs = tuple(range(len(raw_weight_dims), data.ndim))
        canonical = jnp.transpose(data, positions + outputs)
    return _flatten_events(canonical, atom_shape=atom_shape)


def _canonical_points_raw(
    value: Any,
    /,
    *,
    weight_shape: tuple[int, ...],
    sample_positions: tuple[int, ...],
    name: str,
) -> tuple[Array, tuple[int, ...]]:
    leaf = _single_encoded_leaf(value, name=name)
    data = jnp.asarray(leaf.data if isinstance(leaf, cx.Field) else leaf)
    if data.ndim < len(weight_shape) or tuple(data.shape[: len(weight_shape)]) != tuple(
        int(size) for size in weight_shape
    ):
        raise ValueError(
            f"{name} raw arrays must begin with complete weight shape {weight_shape}."
        )
    outputs = tuple(range(len(weight_shape), data.ndim))
    canonical = jnp.transpose(data, sample_positions + outputs)
    atom_shape = tuple(int(weight_shape[position]) for position in sample_positions)
    return _flatten_events(canonical, atom_shape=atom_shape)


def _flatten_events(
    canonical: Array,
    /,
    *,
    atom_shape: tuple[int, ...],
) -> tuple[Array, tuple[int, ...]]:
    atom_count = prod(atom_shape)
    event_shape = tuple(int(size) for size in canonical.shape[len(atom_shape) :])
    if event_shape:
        return canonical.reshape((atom_count, prod(event_shape))), event_shape
    return canonical.reshape((atom_count, 1)), ()


def _single_encoded_leaf(value: Any, /, *, name: str) -> Array | cx.Field:
    if isinstance(value, cx.Field) or eqx.is_array(value):
        return value
    leaves = jtu.tree_leaves(value, is_leaf=lambda item: isinstance(item, cx.Field))
    if len(leaves) != 1:
        raise ValueError(
            f"{name} must be one array/field or use an explicit event encoder; "
            f"found {len(leaves)} leaves."
        )
    leaf = leaves[0]
    if not isinstance(leaf, cx.Field):
        leaf = jnp.asarray(leaf)
    return leaf


def _linear_probabilities_and_mass(
    weights: Array,
    included: Array,
    /,
    *,
    normalized: bool,
    target_mass: Array | None,
    name: str,
) -> tuple[Array, Array, Array]:
    values = jnp.asarray(weights, dtype=float)
    included_ = jnp.asarray(included, dtype=bool)
    values = eqx.error_if(
        values,
        jnp.any(included_ & (~jnp.isfinite(values) | (values < 0.0))),
        f"{name} active weights must be finite and nonnegative.",
    )
    positive = included_ & (values > 0.0)
    active_weights = jnp.where(positive, values, 0.0)
    raw_mass = jnp.sum(active_weights)
    raw_mass = eqx.error_if(
        raw_mass,
        ~jnp.isfinite(raw_mass) | (raw_mass <= 0.0),
        f"{name} must contain positive finite mass.",
    )
    probabilities = active_weights / raw_mass
    mass = _resolved_mass(
        raw_mass,
        normalized=normalized,
        target_mass=target_mass,
        name=name,
    )
    return probabilities, mass, positive


def _log_probabilities_and_mass(
    log_weights: Array,
    included: Array,
    /,
    *,
    normalized: bool,
    target_mass: Array | None,
    name: str,
) -> tuple[Array, Array, Array]:
    values = jnp.asarray(log_weights, dtype=float)
    included_ = jnp.asarray(included, dtype=bool)
    admissible = jnp.isfinite(values) | jnp.isneginf(values)
    values = eqx.error_if(
        values,
        jnp.any(included_ & ~admissible),
        f"{name} active log weights must be finite or negative infinity.",
    )
    probabilities, log_mass, valid = log_normalize(
        values,
        axes=0,
        mask=included_,
    )
    probabilities = eqx.error_if(
        probabilities,
        ~valid,
        f"{name} must contain positive finite mass.",
    )
    raw_mass = jnp.exp(log_mass)
    mass = _resolved_mass(
        raw_mass,
        normalized=normalized,
        target_mass=target_mass,
        name=name,
    )
    active = included_ & jnp.isfinite(values)
    return probabilities, mass, active


def _resolved_mass(
    raw_mass: Array,
    /,
    *,
    normalized: bool,
    target_mass: Array | None,
    name: str,
) -> Array:
    if target_mass is not None:
        mass = jnp.asarray(target_mass, dtype=float).reshape(())
    elif normalized:
        mass = jnp.asarray(1.0, dtype=raw_mass.dtype)
    else:
        mass = raw_mass
    return eqx.error_if(
        mass,
        ~jnp.isfinite(mass) | (mass <= 0.0),
        f"{name} target mass must be finite and positive.",
    )


def _combine_support_validity(
    included: Array,
    support_valid: Array | None,
    /,
    *,
    atom_shape: tuple[int, ...],
    name: str,
) -> Array:
    if support_valid is None:
        return included
    valid = jnp.asarray(support_valid, dtype=bool)
    atom_count = prod(atom_shape)
    if valid.shape == atom_shape:
        canonical = valid
    elif valid.size == atom_count:
        canonical = valid.reshape(atom_shape)
    else:
        raise ValueError(
            f"{name} support_valid must have atom shape {atom_shape} or "
            f"{atom_count} entries."
        )
    return included & canonical


__all__ = ["EventEncoder"]
