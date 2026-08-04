#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import coordax as cx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key, PyTree

from .._callable import _ensure_special_kwonly_args
from .._doc import DOC_KEY0
from .._frozendict import frozendict
from .._numerics import QuadratureRuleData
from ..domain._base import _AbstractGeometry
from ..domain._components import (
    Boundary,
    DomainComponent,
    DomainComponentUnion,
    Fixed,
    FixedEnd,
    FixedStart,
    Interior,
)
from ..domain._dataset import DatasetDomain
from ..domain._domain import RelabeledDomain
from ..domain._grid import AbstractAxisSpec, AxisDiscretization
from ..domain._probability import ProbabilityDomain
from ..domain._scalar import _AbstractScalarDomain
from ..domain._structure import (
    CoordSeparableBatch,
    PointsBatch,
    ProductStructure,
)
from ._batches import PointIntegrationBatch, SeparableIntegrationBatch
from ._plans import FixedQuadraturePlan
from ._rules import interval_rule_data, IntervalRule
from ._targets import ComponentTarget


def first_field_leaf(tree: PyTree[Any], /) -> cx.Field:
    leaves = jax.tree_util.tree_leaves(
        tree, is_leaf=lambda value: isinstance(value, cx.Field)
    )
    for leaf in leaves:
        if isinstance(leaf, cx.Field):
            return leaf
    raise ValueError("Expected at least one coordax.Field leaf.")


def sum_over(field: cx.Field, axis: str, /) -> cx.Field:
    if axis not in field.named_dims:
        raise ValueError(f"Cannot reduce missing axis {axis!r} from dims={field.dims!r}.")
    position = field.dims.index(axis)
    values = jnp.sum(jnp.asarray(field.data), axis=position)
    return cx.Field(values, dims=field.dims[:position] + field.dims[position + 1 :])


def axes_for_over(
    structure: ProductStructure,
    over: str | tuple[str, ...] | None,
    /,
) -> tuple[str, ...]:
    axis_names = structure.axis_names
    if axis_names is None:
        raise ValueError("ProductStructure must be canonicalized before integration.")
    if over is None:
        return tuple(axis_names)
    requested = (over,) if isinstance(over, str) else tuple(over)
    if not requested:
        raise ValueError("over must select at least one active label.")
    requested_set = frozenset(requested)
    if len(requested_set) != len(requested):
        raise ValueError(f"over={requested!r} contains duplicate labels.")
    selected: list[str] = []
    covered: set[str] = set()
    for block, axis in zip(structure.blocks, axis_names, strict=True):
        block_set = frozenset(block)
        overlap = requested_set & block_set
        if not overlap:
            continue
        if overlap != block_set:
            raise ValueError(
                f"over={requested!r} partially selects coupled block {block!r}."
            )
        selected.append(axis)
        covered.update(block)
    missing = requested_set - covered
    if missing:
        raise ValueError(
            f"over={requested!r} includes inactive or unknown labels "
            f"{tuple(sorted(missing))!r}."
        )
    return tuple(selected)


def label_measure(component: DomainComponent, label: str, /) -> Array:
    selector = component.spec.component_for(label)
    factor = component.domain.factor(label)
    if isinstance(factor, RelabeledDomain):
        factor = factor.base
    if isinstance(factor, _AbstractGeometry):
        if isinstance(selector, Interior):
            return jnp.asarray(factor.volume, dtype=float)
        if isinstance(selector, Boundary):
            return jnp.asarray(factor.boundary_measure_value, dtype=float)
        if isinstance(selector, Fixed):
            return jnp.asarray(1.0)
        raise TypeError(f"Unsupported geometry component {type(selector).__name__}.")
    if isinstance(factor, _AbstractScalarDomain):
        if isinstance(selector, Interior):
            return jnp.asarray(factor.measure, dtype=float)
        if isinstance(selector, Boundary):
            return jnp.asarray(2.0)
        if isinstance(selector, (FixedStart, FixedEnd, Fixed)):
            return jnp.asarray(1.0)
        raise TypeError(f"Unsupported scalar component {type(selector).__name__}.")
    if isinstance(factor, DatasetDomain):
        if isinstance(selector, Interior):
            return jnp.asarray(factor.measure, dtype=float)
        raise TypeError(f"Unsupported dataset component {type(selector).__name__}.")
    from ..domain._ragged_series_dataset import RaggedSeriesDatasetDomain
    from ..domain.graph._dataset import GraphDatasetDomain
    from ..domain.graph._domain import GraphDomain

    if isinstance(factor, RaggedSeriesDatasetDomain):
        if isinstance(selector, Interior):
            return jnp.asarray(factor.measure, dtype=float)
        raise TypeError(f"Unsupported dataset component {type(selector).__name__}.")
    if isinstance(factor, (GraphDomain, GraphDatasetDomain)):
        return jnp.asarray(factor.component_measure(selector), dtype=float)
    raise TypeError(f"Unsupported unary domain type {type(factor).__name__}.")


def component_factor_fields(
    component: DomainComponent,
    points: PointsBatch | CoordSeparableBatch | Any,
    /,
    *,
    key: Key[Array, ""],
    kwargs: dict[str, Any],
) -> tuple[cx.Field, cx.Field]:
    """Evaluate dynamic selection and measure-modifier fields."""
    mask = cx.Field(jnp.asarray(1.0), dims=())
    if isinstance(points, CoordSeparableBatch):
        for coordinate_mask in points.coord_mask_by_label.values():
            values = jnp.asarray(coordinate_mask.data)
            mask = mask * cx.Field(values.astype(float), dims=coordinate_mask.dims)
    for label, where_function in component.where.items():
        if (
            isinstance(points, CoordSeparableBatch)
            and label in points.coord_axes_by_label
        ):
            continue
        wrapped = _ensure_special_kwonly_args(where_function)
        value = cx.cmap(wrapped, out_axes="leading")(points[label], key=key)
        if not isinstance(value, cx.Field):
            raise TypeError("Per-label component filters must return coordax.Field.")
        data = jnp.asarray(value.data)
        mask = mask * cx.Field(
            data.astype(float) if data.dtype == jnp.bool_ else data,
            dims=value.dims,
        )
    if component.where_all is not None:
        value = component.where_all(points, key=key, **kwargs)
        data = jnp.asarray(value.data)
        mask = mask * cx.Field(
            data.astype(float) if data.dtype == jnp.bool_ else data,
            dims=value.dims,
        )
    modifier = cx.Field(jnp.asarray(1.0), dims=())
    if component.weight_all is not None:
        value = component.weight_all(points, key=key, **kwargs)
        if not isinstance(value, cx.Field):
            raise TypeError("component.weight_all must return coordax.Field.")
        modifier = modifier * value
    return mask, modifier


def _custom_total_weight(component: DomainComponent, points: Any, /) -> cx.Field | None:
    from ..domain._irregular_trajectory_dataset import (
        irregular_trajectory_default_quadrature_total_weight,
    )
    from ..domain._trajectory_dataset import trajectory_default_quadrature_total_weight
    from ..domain.graph._trajectory import (
        graph_trajectory_default_quadrature_total_weight,
    )

    graph = graph_trajectory_default_quadrature_total_weight(component, points)
    if graph is not None:
        return graph
    trajectory = trajectory_default_quadrature_total_weight(component, points)
    if trajectory is not None:
        return trajectory
    return irregular_trajectory_default_quadrature_total_weight(component, points)


def _point_weight(
    component: DomainComponent,
    points: Any,
    axes: tuple[str, ...],
    /,
) -> cx.Field:
    custom = _custom_total_weight(component, points)
    if custom is not None:
        unwanted = tuple(axis for axis in custom.named_dims if axis not in axes)
        if unwanted:
            raise ValueError(
                "Custom trajectory integration cannot partially weight unintegrated axes."
            )
        return custom
    structure = points.structure
    axis_names = structure.axis_names
    if axis_names is None:
        raise ValueError("Point batch structure must be canonicalized.")
    total = cx.Field(jnp.asarray(1.0), dims=())
    for block, axis in zip(structure.blocks, axis_names, strict=True):
        if axis not in axes:
            continue
        mass = jnp.asarray(1.0)
        for label in block:
            mass = mass * label_measure(component, label)
        reference = first_field_leaf(points[block[0]])
        count = int(reference.named_shape[axis])
        total = total * cx.Field(
            jnp.full((count,), mass / float(count), dtype=float), dims=(axis,)
        )
    return total


def _coord_axes(
    points: CoordSeparableBatch,
    component: DomainComponent,
    over: str | tuple[str, ...] | None,
    /,
) -> tuple[str, ...]:
    if over is None:
        dense = points.dense_structure.axis_names
        if dense is None:
            raise ValueError("Dense product structure must be canonicalized.")
        return tuple(
            axis
            for label in component.domain.labels
            for axis in points.coord_axes_by_label.get(label, ())
        ) + tuple(dense)
    if isinstance(over, str) and over in points.coord_axes_by_label:
        return tuple(points.coord_axes_by_label[over])
    return axes_for_over(points.dense_structure, over)


def _coord_weights(
    component: DomainComponent,
    points: CoordSeparableBatch,
    axes: tuple[str, ...],
    /,
) -> dict[str, cx.Field]:
    weights: dict[str, cx.Field] = {}
    for label, coordinate_axes in points.coord_axes_by_label.items():
        factor = component.domain.factor(label)
        if isinstance(factor, RelabeledDomain):
            factor = factor.base
        for coordinate_index, axis in enumerate(coordinate_axes):
            if axis not in axes:
                continue
            field = first_field_leaf(points.points[label][coordinate_index])
            count = int(field.named_shape[axis])
            discretization = points.axis_discretization_by_axis.get(axis)
            if discretization is not None and discretization.quad_weights is not None:
                values = discretization.quad_weights
            elif isinstance(factor, _AbstractGeometry):
                bounds = jnp.asarray(factor.mesh_bounds, dtype=float)
                values = jnp.full(
                    (count,),
                    (bounds[1, coordinate_index] - bounds[0, coordinate_index])
                    / float(count),
                )
            elif isinstance(factor, _AbstractScalarDomain):
                values = jnp.full(
                    (count,), label_measure(component, label) / float(count)
                )
            else:
                raise TypeError(
                    "Coord-separable weights require geometry or scalar factors."
                )
            weights[axis] = cx.Field(jnp.asarray(values), dims=(axis,))
    dense_names = points.dense_structure.axis_names
    if dense_names is None:
        raise ValueError("Dense product structure must be canonicalized.")
    for block, axis in zip(points.dense_structure.blocks, dense_names, strict=True):
        if axis not in axes:
            continue
        mass = jnp.asarray(1.0)
        for label in block:
            mass = mass * label_measure(component, label)
        reference = first_field_leaf(points[block[0]])
        count = int(reference.named_shape[axis])
        weights[axis] = cx.Field(jnp.full((count,), mass / float(count)), dims=(axis,))
    return weights


def materialize_sampled_component(
    target: ComponentTarget,
    points: Any,
    /,
) -> PointIntegrationBatch | SeparableIntegrationBatch:
    """Attach authoritative component-measure weights to existing sampled points."""
    if isinstance(target.component, DomainComponentUnion):
        raise TypeError("Materialize each union term against its aligned point batch.")
    component = target.component
    if isinstance(points, CoordSeparableBatch):
        axes = _coord_axes(points, component, target.axes)
        weights = _coord_weights(component, points, axes)
        coupled = cx.Field(jnp.asarray(1.0), dims=())
        for label, geometry_weight in points.coord_geometry_weight_by_label.items():
            coordinate_axes = points.coord_axes_by_label.get(label, ())
            if coordinate_axes and all(axis in axes for axis in coordinate_axes):
                coupled = coupled * geometry_weight
        return SeparableIntegrationBatch(
            points,
            weights,
            axes=axes,
            coupled_weight=coupled,
            target_mass=component.measure(),
            provenance="component-sampled-separable",
        )
    axes = axes_for_over(points.structure, target.axes)
    return PointIntegrationBatch(
        points,
        _point_weight(component, points, axes),
        axes=axes,
        target_mass=component.measure(),
        provenance="component-sampled",
    )


class IntegrationAxisSpec(AbstractAxisSpec):
    """Domain-grid adapter for a canonical interval quadrature rule."""

    rule: IntervalRule

    def __init__(self, rule: IntervalRule):
        data = interval_rule_data(rule)
        super().__init__(int(data.nodes.shape[0]))
        self.rule = rule

    def materialize(self, lower: Array, upper: Array, /) -> AxisDiscretization:
        data = interval_rule_data(self.rule)
        half = 0.5 * (jnp.asarray(upper) - jnp.asarray(lower))
        center = 0.5 * (jnp.asarray(upper) + jnp.asarray(lower))
        return AxisDiscretization(
            nodes=center + half * data.nodes,
            quad_weights=half * data.weights,
            basis="legendre",
            periodic=False,
        )


def _scalar_interior_rule_data(
    factor: _AbstractScalarDomain,
    data: QuadratureRuleData,
    /,
) -> tuple[Array, Array]:
    if isinstance(factor, ProbabilityDomain):
        unit = 0.5 * (data.nodes + 1.0)
        if (
            bool(jnp.any(unit <= 0.0)) or bool(jnp.any(unit >= 1.0))
        ) and factor.distribution.support is None:
            raise ValueError(
                "Endpoint-inclusive quadrature cannot map an unbounded probability "
                "component; use GaussLegendreRule or stochastic integration."
            )
        return jnp.asarray(factor.distribution.icdf(unit)), 0.5 * data.weights
    lower = jnp.asarray(factor.fixed("start"))
    upper = jnp.asarray(factor.fixed("end"))
    half = 0.5 * (upper - lower)
    center = 0.5 * (upper + lower)
    return center + half * data.nodes, half * data.weights


def _materialize_scalar_interiors(
    component: DomainComponent,
    target: ComponentTarget,
    rule: IntervalRule,
    /,
) -> PointIntegrationBatch:
    data = interval_rule_data(rule)
    fixed_labels = frozenset(
        label
        for label in component.domain.labels
        if isinstance(
            component.spec.component_for(label),
            (Fixed, FixedStart, FixedEnd),
        )
    )
    blocks = tuple(
        (label,) for label in component.domain.labels if label not in fixed_labels
    )
    structure = ProductStructure(blocks).canonicalize(
        component.domain.labels,
        fixed_labels=fixed_labels,
    )
    points: dict[str, cx.Field] = {}
    weights_by_axis: dict[str, cx.Field] = {}
    for label in component.domain.labels:
        factor = component.domain.factor(label)
        if isinstance(factor, RelabeledDomain):
            factor = factor.base
        selector = component.spec.component_for(label)
        if not isinstance(factor, _AbstractScalarDomain):
            raise TypeError("Scalar fixed quadrature requires scalar domain factors.")
        if isinstance(selector, Interior):
            axis = structure.axis_for(label)
            if axis is None:
                raise RuntimeError("Interior scalar factor has no integration axis.")
            values, weights = _scalar_interior_rule_data(factor, data)
            points[label] = cx.Field(values, dims=(axis,))
            weights_by_axis[axis] = cx.Field(weights, dims=(axis,))
        elif isinstance(selector, Fixed):
            points[label] = cx.Field(jnp.asarray(selector.value).reshape(()), dims=())
        elif isinstance(selector, FixedStart):
            points[label] = cx.Field(
                jnp.asarray(factor.fixed("start")).reshape(()),
                dims=(),
            )
        elif isinstance(selector, FixedEnd):
            points[label] = cx.Field(
                jnp.asarray(factor.fixed("end")).reshape(()),
                dims=(),
            )
        else:
            raise TypeError(f"Unsupported scalar selector {type(selector).__name__}.")
    axes = axes_for_over(structure, target.axes)
    total = cx.Field(jnp.asarray(1.0), dims=())
    for axis in axes:
        total = total * weights_by_axis[axis]
    return PointIntegrationBatch(
        PointsBatch(frozendict(points), structure),
        total,
        axes=axes,
        target_mass=component.measure(),
        provenance=f"{type(rule).__name__}:{data.nodes.shape[0]}",
    )


def _materialize_scalar_boundaries(
    component: DomainComponent,
    target: ComponentTarget,
    rule: IntervalRule,
    /,
) -> PointIntegrationBatch:
    data = interval_rule_data(rule)
    points: dict[str, cx.Field] = {}
    blocks: list[tuple[str, ...]] = []
    weights_by_axis: dict[str, cx.Field] = {}
    fixed_labels: set[str] = set()
    for label in component.domain.labels:
        factor = component.domain.factor(label)
        if isinstance(factor, RelabeledDomain):
            factor = factor.base
        selector = component.spec.component_for(label)
        if isinstance(selector, (Fixed, FixedStart, FixedEnd)):
            fixed_labels.add(label)
            if isinstance(selector, Fixed):
                value = selector.value
            elif isinstance(selector, FixedStart):
                if not isinstance(factor, _AbstractScalarDomain):
                    raise TypeError("FixedStart requires a scalar domain factor.")
                value = factor.fixed("start")
            else:
                if not isinstance(factor, _AbstractScalarDomain):
                    raise TypeError("FixedEnd requires a scalar domain factor.")
                value = factor.fixed("end")
            points[label] = cx.Field(jnp.asarray(value), dims=())
            continue
        if not isinstance(factor, _AbstractScalarDomain):
            raise ValueError("Scalar boundary quadrature requires scalar factors.")
        axis = f"__phydra_blk__{label}"
        if isinstance(selector, Boundary):
            values = jnp.stack(
                (
                    jnp.asarray(factor.fixed("start")),
                    jnp.asarray(factor.fixed("end")),
                )
            )
            weights = jnp.ones((2,), dtype=float)
        elif isinstance(selector, Interior):
            values, weights = _scalar_interior_rule_data(factor, data)
        else:
            raise TypeError(f"Unsupported scalar selector {type(selector).__name__}.")
        points[label] = cx.Field(values, dims=(axis,))
        blocks.append((label,))
        weights_by_axis[axis] = cx.Field(weights, dims=(axis,))
    structure = ProductStructure(tuple(blocks)).canonicalize(
        component.domain.labels, fixed_labels=frozenset(fixed_labels)
    )
    batch = PointsBatch(frozendict(points), structure)
    axes = axes_for_over(structure, target.axes)
    total = cx.Field(jnp.asarray(1.0), dims=())
    for axis in axes:
        total = total * weights_by_axis[axis]
    return PointIntegrationBatch(
        batch,
        total,
        axes=axes,
        target_mass=component.measure(),
        provenance=f"scalar-boundary:{type(rule).__name__}",
    )


def _materialize_cad_boundary(
    component: DomainComponent,
    target: ComponentTarget,
    rule: IntervalRule,
    /,
) -> PointIntegrationBatch:
    from ..domain.geometry2d._from_cad import Geometry2DFromCAD
    from ..domain.geometry3d._mesh import Geometry3DFromCAD
    from ._rules import ReferenceIntervalRule, ReferenceQuadrilateralRule

    varying = tuple(
        label
        for label in component.domain.labels
        if not isinstance(
            component.spec.component_for(label), (Fixed, FixedStart, FixedEnd)
        )
    )
    if len(varying) != 1:
        raise ValueError(
            "CAD boundary quadrature supports one varying geometry factor; "
            "use ProductIntegrationPlan for mixed factors."
        )
    label = varying[0]
    selector = component.spec.component_for(label)
    factor = component.domain.factor(label)
    if isinstance(factor, RelabeledDomain):
        factor = factor.base
    if not isinstance(selector, Boundary) or not isinstance(
        factor, (Geometry2DFromCAD, Geometry3DFromCAD)
    ):
        raise ValueError("Fixed boundary quadrature requires a CAD geometry boundary.")
    atlas = factor.boundary_chart_atlas
    reference_rule = (
        ReferenceIntervalRule(rule)
        if atlas.reference_dim == 1
        else ReferenceQuadrilateralRule(rule)
    )
    reference_data = reference_rule.materialize()
    count = int(reference_data.points.shape[0])
    charts = atlas.num_charts
    reference = jnp.broadcast_to(
        reference_data.points[None, ...],
        (charts, count, atlas.reference_dim),
    )
    chart_indices = jnp.broadcast_to(
        jnp.arange(charts, dtype=jnp.int32)[:, None],
        (charts, count),
    )
    physical = atlas.map(chart_indices, reference)
    jacobian = atlas.jacobian(chart_indices, reference)
    weights = jacobian * reference_data.weights[None, :]
    fixed_labels = frozenset(other for other in component.domain.labels if other != label)
    structure = ProductStructure(((label,),)).canonicalize(
        component.domain.labels, fixed_labels=fixed_labels
    )
    axis = structure.axis_for(label)
    if axis is None:
        raise RuntimeError("CAD boundary structure has no integration axis.")
    points: dict[str, cx.Field] = {
        label: cx.Field(physical.reshape((-1, factor.var_dim)), dims=(axis, None))
    }
    for other in fixed_labels:
        selector_ = component.spec.component_for(other)
        factor_ = component.domain.factor(other)
        if isinstance(factor_, RelabeledDomain):
            factor_ = factor_.base
        if isinstance(selector_, Fixed):
            value = selector_.value
        elif isinstance(selector_, FixedStart):
            if not isinstance(factor_, _AbstractScalarDomain):
                raise TypeError("FixedStart requires a scalar domain factor.")
            value = factor_.fixed("start")
        else:
            if not isinstance(factor_, _AbstractScalarDomain):
                raise TypeError("FixedEnd requires a scalar domain factor.")
            value = factor_.fixed("end")
        dimensions = (None,) if isinstance(factor_, _AbstractGeometry) else ()
        points[other] = cx.Field(jnp.asarray(value), dims=dimensions)
    axes = axes_for_over(structure, target.axes)
    return PointIntegrationBatch(
        PointsBatch(
            frozendict({name: points[name] for name in component.domain.labels}),
            structure,
        ),
        cx.Field(weights.reshape((-1,)), dims=(axis,)),
        axes=axes,
        target_mass=component.measure(),
        provenance=f"cad-boundary:{type(rule).__name__}",
    )


def materialize_fixed_component(
    target: ComponentTarget,
    plan: FixedQuadraturePlan,
    /,
) -> PointIntegrationBatch | SeparableIntegrationBatch | tuple[Any, ...]:
    """Lower component geometry and interval rules into typed batches."""
    if isinstance(target.component, DomainComponentUnion):
        return tuple(
            materialize_fixed_component(
                ComponentTarget(term, axes=target.axes, normalized=target.normalized),
                plan,
            )
            for term in target.component.terms
        )
    component = target.component
    selectors = tuple(
        component.spec.component_for(label) for label in component.domain.labels
    )
    rule = plan.rule
    interval_rule_data(rule)
    if any(isinstance(selector, Boundary) for selector in selectors):
        factors = tuple(
            component.domain.factor(label) for label in component.domain.labels
        )
        unwrapped = tuple(
            factor.base if isinstance(factor, RelabeledDomain) else factor
            for factor in factors
        )
        if all(isinstance(factor, _AbstractScalarDomain) for factor in unwrapped):
            return _materialize_scalar_boundaries(component, target, rule)
        return _materialize_cad_boundary(component, target, rule)
    factors = tuple(component.domain.factor(label) for label in component.domain.labels)
    unwrapped_factors = tuple(
        factor.base if isinstance(factor, RelabeledDomain) else factor
        for factor in factors
    )
    if all(isinstance(factor, _AbstractScalarDomain) for factor in unwrapped_factors):
        return _materialize_scalar_interiors(component, target, rule)
    spec = IntegrationAxisSpec(rule)
    coord_separable: dict[str, Any] = {}
    for label in component.domain.labels:
        selector = component.spec.component_for(label)
        if isinstance(selector, (Fixed, FixedStart, FixedEnd)):
            continue
        if not isinstance(selector, Interior):
            raise ValueError(
                f"Unsupported fixed component selector {type(selector).__name__}."
            )
        factor = component.domain.factor(label)
        if isinstance(factor, RelabeledDomain):
            factor = factor.base
        if isinstance(factor, _AbstractGeometry):
            coord_separable[label] = (spec,) * int(factor.var_dim)
        elif isinstance(factor, _AbstractScalarDomain):
            coord_separable[label] = spec
        else:
            raise TypeError(
                "Fixed quadrature currently supports scalar and geometric domain factors."
            )
    points = component.sample_coord_separable(
        coord_separable,
        num_points=(),
        dense_structure=ProductStructure(()),
        sampler="uniform",
        key=DOC_KEY0,
    )
    return materialize_sampled_component(target, points)


__all__ = [
    "IntegrationAxisSpec",
    "axes_for_over",
    "component_factor_fields",
    "first_field_leaf",
    "label_measure",
    "materialize_fixed_component",
    "materialize_sampled_component",
    "sum_over",
]
