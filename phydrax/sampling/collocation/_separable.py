#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TYPE_CHECKING

import coordax as cx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Key

from phydrax.domain import DomainComponent, GridBatch

from ..._doc import DOC_KEY0
from ..._frozendict import frozendict
from ..._strict import StrictModule
from ._adaptive import AbstractCollocationPolicy


if TYPE_CHECKING:
    from phydrax.domain import DomainFunction

    from ._adaptive import PointwiseSamplingTerm


class SeparableCollocationPopulation(StrictModule):
    """Persistent state for one fixed-shape coordinate-separable population."""

    batch: GridBatch
    axis_age_by_axis: frozendict[str, cx.Field]
    axis_active_by_axis: frozendict[str, cx.Field]
    refresh_count: Array
    last_refresh: Array
    logical_point_count: Array
    active_logical_point_count: Array

    def __init__(
        self,
        batch: GridBatch,
        *,
        axis_age_by_axis: Mapping[str, cx.Field] | None = None,
        axis_active_by_axis: Mapping[str, cx.Field] | None = None,
        refresh_count: int | Array = 0,
        last_refresh: int | Array = 0,
    ):
        axis_fields = _axis_fields(batch)
        if axis_age_by_axis is None:
            ages = {
                axis: cx.Field(
                    jnp.zeros(field.data.shape, dtype=jnp.int32),
                    dims=(axis,),
                )
                for axis, field in axis_fields.items()
            }
        else:
            ages = dict(axis_age_by_axis)
        if axis_active_by_axis is None:
            active_by_axis = _axis_active_fields(batch)
        else:
            active_by_axis = dict(axis_active_by_axis)
        if frozenset(active_by_axis) != frozenset(axis_fields):
            raise ValueError(
                "axis_active_by_axis must contain exactly the coordinate-separable axes."
            )
        for axis, active in active_by_axis.items():
            expected = axis_fields[axis].data.shape
            if active.dims != (axis,) or active.data.shape != expected:
                raise ValueError(
                    f"Active mask for axis {axis!r} must have dims {(axis,)!r} "
                    f"and shape {expected!r}."
                )
        if frozenset(ages) != frozenset(axis_fields):
            raise ValueError(
                "axis_age_by_axis must contain exactly the coordinate-separable axes."
            )
        for axis, age in ages.items():
            expected = axis_fields[axis].data.shape
            if age.dims != (axis,) or age.data.shape != expected:
                raise ValueError(
                    f"Age for axis {axis!r} must have dims {(axis,)!r} and shape "
                    f"{expected!r}."
                )
        logical_count, active_count = _logical_counts(batch, active_by_axis)
        self.batch = batch
        self.axis_age_by_axis = frozendict(ages)
        self.axis_active_by_axis = frozendict(active_by_axis)
        self.refresh_count = jnp.asarray(refresh_count, dtype=jnp.int32)
        self.last_refresh = jnp.asarray(last_refresh, dtype=jnp.int32)
        self.logical_point_count = jnp.asarray(logical_count, dtype=jnp.int32)
        self.active_logical_point_count = jnp.asarray(active_count, dtype=jnp.int32)

    def loss_weight(self) -> cx.Field:
        return _axis_active_weight(self.axis_active_by_axis)


def _separable_population_metrics(
    population: SeparableCollocationPopulation,
    /,
) -> dict[str, Array]:
    axis_node_count = sum(
        int(field.data.shape[0]) for field in _axis_fields(population.batch).values()
    )
    active_axis_node_count = sum(
        jnp.sum(jnp.asarray(field.data, dtype=float))
        for field in population.axis_active_by_axis.values()
    )
    return {
        "refresh_count": jnp.asarray(population.refresh_count, dtype=float),
        "last_refresh": jnp.asarray(population.last_refresh, dtype=float),
        "logical_point_count": jnp.asarray(
            population.logical_point_count,
            dtype=float,
        ),
        "active_logical_point_count": jnp.asarray(
            population.active_logical_point_count,
            dtype=float,
        ),
        "axis_node_count": jnp.asarray(axis_node_count, dtype=float),
        "active_axis_node_count": jnp.asarray(
            active_axis_node_count,
            dtype=float,
        ),
        "effective_sample_size": jnp.asarray(
            population.active_logical_point_count,
            dtype=float,
        ),
    }


class SeparableCollocationPolicy(AbstractCollocationPolicy):
    """Periodic replacement for fixed-shape coordinate-separable populations."""

    refresh_every: int

    def __init__(self, *, refresh_every: int = 100):
        if int(refresh_every) <= 0:
            raise ValueError("refresh_every must be positive.")
        self.refresh_every = int(refresh_every)

    def should_refresh(
        self,
        population: SeparableCollocationPopulation,
        iter_: int | Array,
    ) -> Array:
        step = jnp.asarray(iter_, dtype=jnp.int32)
        return (step - population.last_refresh) >= self.refresh_every

    def initialize(
        self,
        constraint: PointwiseSamplingTerm,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> SeparableCollocationPopulation:
        batch = constraint.sample(key=key)
        if not isinstance(batch, GridBatch):
            raise TypeError(
                "Separable adaptive collocation requires coord-separable sampling."
            )
        _single_component(constraint)
        return SeparableCollocationPopulation(batch)

    def loss_batch_and_weight(
        self,
        population: SeparableCollocationPopulation,
        /,
    ) -> tuple[GridBatch, cx.Field]:
        return population.batch, population.loss_weight()

    def data_metrics(
        self,
        population: SeparableCollocationPopulation,
        /,
    ) -> dict[str, Array]:
        return _separable_population_metrics(population)

    def refresh_residual_evaluations(
        self,
        population: SeparableCollocationPopulation,
        /,
    ) -> int:
        return 0

    def refresh(
        self,
        constraint: PointwiseSamplingTerm,
        functions: Mapping[str, DomainFunction],
        population: SeparableCollocationPopulation,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array,
    ) -> SeparableCollocationPopulation:
        del functions
        sampled = constraint.sample(key=key)
        if not isinstance(sampled, GridBatch):
            raise TypeError("Periodic grid refresh requires GridBatch.")
        batch = _replace_coordinate_blocks(population.batch, sampled)
        return SeparableCollocationPopulation(
            batch,
            refresh_count=population.refresh_count + 1,
            last_refresh=iter_,
        )


class HierarchicalAxisPolicy(AbstractCollocationPolicy):
    """Fixed-capacity residual-guided activation of nested coordinate nodes."""

    refresh_every: int
    refinement_fraction: Array
    epsilon: Array

    def __init__(
        self,
        *,
        refresh_every: int = 100,
        refinement_fraction: float = 0.1,
        epsilon: float = 1e-12,
    ):
        if int(refresh_every) <= 0:
            raise ValueError("refresh_every must be positive.")
        if not 0.0 < float(refinement_fraction) <= 1.0:
            raise ValueError("refinement_fraction must lie in (0, 1].")
        if float(epsilon) <= 0.0:
            raise ValueError("epsilon must be positive.")
        self.refresh_every = int(refresh_every)
        self.refinement_fraction = jnp.asarray(refinement_fraction, dtype=float)
        self.epsilon = jnp.asarray(epsilon, dtype=float)

    def should_refresh(
        self,
        population: SeparableCollocationPopulation,
        iter_: int | Array,
    ) -> Array:
        step = jnp.asarray(iter_, dtype=jnp.int32)
        return (step - population.last_refresh) >= self.refresh_every

    def initialize(
        self,
        constraint: PointwiseSamplingTerm,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> SeparableCollocationPopulation:
        batch = constraint.sample(key=key)
        if not isinstance(batch, GridBatch):
            raise TypeError(
                "Hierarchical axis refinement requires coord-separable sampling."
            )
        _single_component(constraint)
        for axis in _axis_fields(batch):
            discretization = batch.axis_discretization_by_axis.get(axis)
            if (
                discretization is None
                or discretization.basis != "nested"
                or discretization.active is None
            ):
                raise ValueError(
                    "Every adaptive coordinate axis must use NestedDyadicAxisSpec."
                )
        return SeparableCollocationPopulation(batch)

    def loss_batch_and_weight(
        self,
        population: SeparableCollocationPopulation,
        /,
    ) -> tuple[GridBatch, cx.Field]:
        return population.batch, population.loss_weight()

    def data_metrics(
        self,
        population: SeparableCollocationPopulation,
        /,
    ) -> dict[str, Array]:
        return _separable_population_metrics(population)

    def refresh_residual_evaluations(
        self,
        population: SeparableCollocationPopulation,
        /,
    ) -> int:
        return int(population.logical_point_count)

    def refresh(
        self,
        constraint: PointwiseSamplingTerm,
        functions: Mapping[str, DomainFunction],
        population: SeparableCollocationPopulation,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array,
    ) -> SeparableCollocationPopulation:
        marginals = _axis_residual_marginals(
            constraint,
            functions,
            population.batch,
            key=key,
            epsilon=self.epsilon,
        )
        discretizations = dict(population.batch.axis_discretization_by_axis)
        active_by_axis: dict[str, cx.Field] = {}
        ages: dict[str, cx.Field] = {}
        for axis, field in _axis_fields(population.batch).items():
            discretization = discretizations[axis]
            old_active = jnp.asarray(
                population.axis_active_by_axis[axis].data,
                dtype=bool,
            )
            inactive_count = int(jnp.sum(~old_active))
            if inactive_count == 0:
                new_active = old_active
            else:
                add_n = min(
                    max(
                        1,
                        int(round(field.data.shape[0] * float(self.refinement_fraction))),
                    ),
                    inactive_count,
                )
                score = jnp.asarray(marginals[axis].data, dtype=float)
                ranked = jnp.argsort(jnp.where(old_active, -jnp.inf, score))[::-1]
                new_active = old_active.at[ranked[:add_n]].set(True)
            discretizations[axis] = discretization.with_active(new_active)
            active_by_axis[axis] = cx.Field(
                new_active.astype(float),
                dims=(axis,),
            )
            old_age = jnp.asarray(
                population.axis_age_by_axis[axis].data,
                dtype=jnp.int32,
            )
            retained_age = jnp.where(old_active, old_age + 1, 0)
            ages[axis] = cx.Field(retained_age, dims=(axis,))
        batch = GridBatch(
            points=population.batch.points,
            dense_structure=population.batch.dense_structure,
            coord_axes_by_label=population.batch.coord_axes_by_label,
            coord_mask_by_label=population.batch.coord_mask_by_label,
            coord_geometry_weight_by_label=(
                population.batch.coord_geometry_weight_by_label
            ),
            coord_geometry_order_by_label=(
                population.batch.coord_geometry_order_by_label
            ),
            axis_discretization_by_axis=frozendict(discretizations),
        )
        return SeparableCollocationPopulation(
            batch,
            axis_age_by_axis=ages,
            axis_active_by_axis=active_by_axis,
            refresh_count=population.refresh_count + 1,
            last_refresh=iter_,
        )


def HierarchicalAxisCollocation(**kwargs: Any) -> HierarchicalAxisPolicy:
    return HierarchicalAxisPolicy(**kwargs)


def PeriodicSeparableCollocation(**kwargs: Any) -> SeparableCollocationPolicy:
    return SeparableCollocationPolicy(**kwargs)


def _single_component(constraint: PointwiseSamplingTerm) -> DomainComponent:
    component = constraint.component
    if not isinstance(component, DomainComponent):
        raise TypeError(
            "Separable adaptive collocation does not support component unions."
        )
    return component


def _axis_fields(batch: GridBatch) -> dict[str, cx.Field]:
    fields: dict[str, cx.Field] = {}
    for label, axes in batch.coord_axes_by_label.items():
        values = batch.points[label]
        if not isinstance(values, tuple):
            raise TypeError(f"Coordinate-separable label {label!r} must store a tuple.")
        for axis, field in zip(axes, values, strict=True):
            if not isinstance(field, cx.Field):
                raise TypeError(
                    f"Coordinate-separable axis {axis!r} must store a coordax.Field."
                )
            fields[axis] = field
    return fields


def _axis_active_fields(batch: GridBatch) -> dict[str, cx.Field]:
    active: dict[str, cx.Field] = {}
    for axis, field in _axis_fields(batch).items():
        discretization = batch.axis_discretization_by_axis.get(axis)
        if discretization is None or discretization.active is None:
            data = jnp.ones(field.data.shape, dtype=float)
        else:
            data = jnp.asarray(discretization.active, dtype=float)
        active[axis] = cx.Field(data, dims=(axis,))
    return active


def _axis_active_weight(active_by_axis: Mapping[str, cx.Field]) -> cx.Field:
    weight = cx.Field(jnp.asarray(1.0, dtype=float), dims=())
    for active in active_by_axis.values():
        weight = weight * cx.Field(
            jnp.asarray(active.data, dtype=float),
            dims=active.dims,
        )
    return weight


def _axis_sizes(batch: GridBatch) -> dict[str, int]:
    sizes = {
        axis: int(field.data.shape[0]) for axis, field in _axis_fields(batch).items()
    }
    dense_axes = batch.dense_structure.axis_names
    if dense_axes is None:
        raise ValueError("GridBatch dense structure must be canonicalized.")
    for block, axis in zip(
        batch.dense_structure.blocks,
        dense_axes,
        strict=True,
    ):
        leaves = jtu.tree_leaves(
            batch.points[block[0]],
            is_leaf=lambda x: isinstance(x, cx.Field),
        )
        fields = [leaf for leaf in leaves if isinstance(leaf, cx.Field)]
        if not fields:
            raise ValueError(f"Dense block {block!r} has no coordinate field.")
        sizes[axis] = int(fields[0].named_shape[axis])
    return sizes


def _logical_counts(
    batch: GridBatch,
    active_by_axis: Mapping[str, cx.Field],
) -> tuple[int, int]:
    sizes = _axis_sizes(batch)
    logical = 1
    for size in sizes.values():
        logical *= size
    mask = _logical_mask_field(batch) * _axis_active_weight(active_by_axis)
    masked_axes = frozenset(dim for dim in mask.dims if dim is not None)
    unmasked = 1
    for axis, size in sizes.items():
        if axis not in masked_axes:
            unmasked *= size
    active = int(jnp.sum(jnp.asarray(mask.data))) * unmasked
    return logical, active


def _logical_mask_field(batch: GridBatch) -> cx.Field:
    result = cx.Field(jnp.asarray(1.0, dtype=float), dims=())
    for mask in batch.coord_mask_by_label.values():
        result = result * cx.Field(jnp.asarray(mask.data, dtype=float), dims=mask.dims)
    return result


def _axis_residual_marginals(
    constraint: PointwiseSamplingTerm,
    functions: Mapping[str, DomainFunction],
    batch: GridBatch,
    /,
    *,
    key: Key[Array, ""],
    epsilon: Array,
) -> frozendict[str, cx.Field]:
    _single_component(constraint)
    score = constraint.pointwise_score(functions, batch, key=key)
    data = jax.lax.stop_gradient(jnp.asarray(score.data, dtype=float))
    data = jnp.nan_to_num(
        data,
        nan=0.0,
        posinf=jnp.finfo(data.dtype).max,
        neginf=0.0,
    )
    score = cx.Field(jnp.maximum(data, 0.0), dims=score.dims)
    masked = score * _logical_mask_field(batch)
    marginals: dict[str, cx.Field] = {}
    for axis in _axis_fields(batch):
        marginal = masked
        reduce_axes = tuple(
            dim for dim in marginal.dims if dim is not None and dim != axis
        )
        for reduce_axis in reduce_axes:
            marginal = _sum_or_max_named(
                marginal,
                reduce_axis,
                maximum=True,
            )
        marginals[axis] = cx.Field(
            jnp.maximum(jnp.asarray(marginal.data), epsilon),
            dims=marginal.dims,
        )
    return frozendict(marginals)


def _replace_coordinate_blocks(old, sampled):
    if old.coord_axes_by_label != sampled.coord_axes_by_label:
        raise ValueError("Periodic refresh changed coordinate-separable axis metadata.")
    points = dict(old.points)
    for label in old.coord_axes_by_label:
        old_values = old.points[label]
        new_values = sampled.points[label]
        if not isinstance(old_values, tuple) or not isinstance(new_values, tuple):
            raise TypeError("Coordinate-separable points must be tuples.")
        if tuple(field.data.shape for field in old_values) != tuple(
            field.data.shape for field in new_values
        ):
            raise ValueError("Periodic refresh must preserve coordinate axis shapes.")
        points[label] = new_values
    return GridBatch(
        points=frozendict(points),
        dense_structure=old.dense_structure,
        coord_axes_by_label=old.coord_axes_by_label,
        coord_mask_by_label=sampled.coord_mask_by_label,
        coord_geometry_weight_by_label=sampled.coord_geometry_weight_by_label,
        coord_geometry_order_by_label=sampled.coord_geometry_order_by_label,
        axis_discretization_by_axis=sampled.axis_discretization_by_axis,
    )


def _sum_or_max_named(field, axis, *, maximum):
    position = field.dims.index(axis)
    if maximum:
        data = jnp.max(field.data, axis=position)
    else:
        data = jnp.sum(field.data, axis=position)
    dims = field.dims[:position] + field.dims[position + 1 :]
    return cx.Field(data, dims=dims)


__all__ = [
    "HierarchicalAxisCollocation",
    "HierarchicalAxisPolicy",
    "PeriodicSeparableCollocation",
    "SeparableCollocationPolicy",
    "SeparableCollocationPopulation",
]
