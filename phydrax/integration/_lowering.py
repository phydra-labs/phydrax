#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import coordax as cx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key, PyTree

from phydrax.discretization import (
    AbstractAxisSpec,
    AxisDiscretization,
    AxisDomain,
)
from phydrax.domain import (
    AbstractGeometry,
    AbstractScalarDomain,
    Boundary,
    ComponentSum,
    DomainComponent,
    Fixed,
    FixedEnd,
    FixedStart,
    GridBatch,
    GridSampling,
    Interior,
    open_unit_interval,
    PointBatch,
    ProbabilityDomain,
    require_exact_mass,
    SampleLayout,
)

from .._callable import _ensure_special_kwonly_args
from .._doc import DOC_KEY0
from .._frozendict import frozendict
from .._precision import complex_precision_dtype, real_precision_dtype_name
from ..geometry import BoundaryAtlasProvider, CubatureAtlasProvider
from ._batches import PointIntegrationBatch, SeparableIntegrationBatch
from ._plans import FixedQuadraturePlan
from ._rules import (
    CubatureRule,
    CubatureRuleData,
    GaussHermiteRule,
    GaussianCubatureRule,
    interval_rule_data,
    IntervalRule,
    OrthogonalRuleData,
    probability_rule_data,
    ProbabilityRule,
)
from ._targets import ComponentTarget


def first_field_leaf(tree: PyTree[Any], /) -> cx.Field:
    leaves = jax.tree_util.tree_leaves(
        tree, is_leaf=lambda value: isinstance(value, cx.Field)
    )
    for leaf in leaves:
        if isinstance(leaf, cx.Field):
            return leaf
    raise ValueError("Expected at least one coordax.Field leaf.")


def sum_over(
    field: cx.Field,
    axis: str,
    /,
    *,
    accumulation_dtype: Any | None = None,
) -> cx.Field:
    if axis not in field.named_dims:
        raise ValueError(f"Cannot reduce missing axis {axis!r} from dims={field.dims!r}.")
    position = field.dims.index(axis)
    data = jnp.asarray(field.data)
    if accumulation_dtype is not None and jnp.issubdtype(data.dtype, jnp.inexact):
        real_dtype = real_precision_dtype_name(accumulation_dtype)
        target_dtype = (
            complex_precision_dtype(real_dtype)
            if jnp.issubdtype(data.dtype, jnp.complexfloating)
            else real_dtype
        )
        data = data.astype(target_dtype)
    values = jnp.sum(data, axis=position)
    return cx.Field(values, dims=field.dims[:position] + field.dims[position + 1 :])


def axes_for_over(
    structure: SampleLayout,
    over: str | tuple[str, ...] | None,
    /,
) -> tuple[str, ...]:
    axis_names = structure.axis_names
    if axis_names is None:
        raise ValueError("SampleLayout must be canonicalized before integration.")
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


def _block_measure(component: DomainComponent, block: tuple[str, ...], /) -> Array:
    """Return one exact factor mass per complete factor represented by ``block``."""
    block_labels = frozenset(block)
    mass = jnp.asarray(1.0, dtype=float)
    for factor_component in component.factor_components:
        factor_labels = frozenset(factor_component.factor.labels)
        overlap = block_labels & factor_labels
        if not overlap:
            continue
        if overlap != factor_labels:
            raise ValueError(
                f"SampleLayout block {block!r} splits coupled factor "
                f"{factor_component.factor.labels!r}."
            )
        mass = mass * require_exact_mass(
            factor_component.measure.mass,
            operation=f"sampling block {block!r} weight construction",
        )
    return mass


def _component_base_mass(component: DomainComponent | ComponentSum, /) -> Array:
    """Realize an exact component base mass without allocating mass descriptors."""
    if isinstance(component, DomainComponent):
        return _block_measure(component, component.domain.labels)
    mass = jnp.asarray(0.0, dtype=float)
    for term in component.terms:
        mass = mass + _block_measure(term, term.domain.labels)
    return mass


def component_factor_fields(
    component: DomainComponent,
    points: PointBatch | GridBatch | Any,
    /,
    *,
    key: Key[Array, ""],
    kwargs: dict[str, Any],
) -> tuple[cx.Field, cx.Field]:
    """Evaluate dynamic selection and measure-modifier fields."""
    mask = cx.Field(jnp.asarray(1.0), dims=())
    if isinstance(points, GridBatch):
        for coordinate_mask in points.coord_mask_by_label.values():
            values = jnp.asarray(coordinate_mask.data)
            mask = mask * cx.Field(values.astype(float), dims=coordinate_mask.dims)
    for label, where_function in component.where.items():
        if isinstance(points, GridBatch) and label in points.coord_axes_by_label:
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
    from phydrax.domain import (
        irregular_trajectory_default_quadrature_total_weight,
        trajectory_default_quadrature_total_weight,
    )
    from phydrax.domain.graph import (
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
        mass = _block_measure(component, block)
        reference = first_field_leaf(points[block[0]])
        count = int(reference.named_shape[axis])
        total = total * cx.Field(
            jnp.full((count,), mass / float(count), dtype=float), dims=(axis,)
        )
    return total


def _coord_axes(
    points: GridBatch,
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
    points: GridBatch,
    axes: tuple[str, ...],
    /,
) -> dict[str, cx.Field]:
    weights: dict[str, cx.Field] = {}
    for label, coordinate_axes in points.coord_axes_by_label.items():
        factor = component.domain.factor(label)

        for coordinate_index, axis in enumerate(coordinate_axes):
            if axis not in axes:
                continue
            field = first_field_leaf(points.points[label][coordinate_index])
            count = int(field.named_shape[axis])
            discretization = points.axis_discretization_by_axis.get(axis)
            if discretization is not None and discretization.quad_weights is not None:
                values = discretization.quad_weights
            elif isinstance(factor, AbstractGeometry):
                bounds = jnp.asarray(factor.mesh_bounds, dtype=float)
                values = jnp.full(
                    (count,),
                    (bounds[1, coordinate_index] - bounds[0, coordinate_index])
                    / float(count),
                )
            elif isinstance(factor, AbstractScalarDomain):
                values = jnp.full(
                    (count,), _block_measure(component, (label,)) / float(count)
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
        mass = _block_measure(component, block)
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
    if isinstance(target.component, ComponentSum):
        raise TypeError("Materialize each union term against its aligned point batch.")
    component = target.component
    if isinstance(points, GridBatch):
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
            target_mass=_component_base_mass(component),
            provenance="component-sampled-separable",
        )
    axes = axes_for_over(points.structure, target.axes)
    return PointIntegrationBatch(
        points,
        _point_weight(component, points, axes),
        axes=axes,
        target_mass=_component_base_mass(component),
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
        lower_ = jnp.asarray(lower)
        upper_ = jnp.asarray(upper)
        half = 0.5 * (upper_ - lower_)
        center = 0.5 * (upper_ + lower_)
        return AxisDiscretization(
            nodes=center + half * data.nodes,
            quad_weights=half * data.weights,
            basis="legendre",
            domain=AxisDomain.interval(lower_, upper_),
            primary_entity="point",
            lower_endpoint_included=bool(jnp.isclose(data.nodes[0], -1.0)),
            upper_endpoint_included=bool(jnp.isclose(data.nodes[-1], 1.0)),
        )


def _fixed_rule_node_count(rule: IntervalRule | ProbabilityRule, /) -> int:
    if isinstance(rule, GaussianCubatureRule):
        return rule.num_points
    if isinstance(rule, GaussHermiteRule):
        data = probability_rule_data(rule)
        if not isinstance(data, OrthogonalRuleData):
            raise RuntimeError("GaussHermiteRule resolved non-orthogonal rule data.")
        return int(data.nodes.shape[0])
    return int(interval_rule_data(rule).nodes.shape[0])


def _scalar_interior_rule_data(
    factor: AbstractScalarDomain,
    rule: IntervalRule | ProbabilityRule,
    /,
) -> tuple[Array, Array]:
    if isinstance(rule, (GaussHermiteRule, GaussianCubatureRule)):
        owner = type(rule).__name__
        if not isinstance(factor, ProbabilityDomain):
            raise TypeError(f"{owner} requires a standard-normal probability factor.")
        data = probability_rule_data(rule)
        if isinstance(rule, GaussianCubatureRule):
            if not isinstance(data, CubatureRuleData):
                raise RuntimeError(
                    "GaussianCubatureRule resolved non-cubature rule data."
                )
            if rule.dimension != 1:
                raise ValueError(
                    "Direct scalar probability integration requires a "
                    "one-dimensional GaussianCubatureRule."
                )
            nodes = data.points[:, 0]
        else:
            if not isinstance(data, OrthogonalRuleData):
                raise RuntimeError("GaussHermiteRule resolved non-orthogonal rule data.")
            nodes = data.nodes
        transport = factor.reference_transport
        if transport.reference_measure != data.integration_measure:
            raise ValueError(
                f"{owner} requires a probability factor with a "
                "standard-normal reference transport."
            )
        return transport.from_reference(nodes), data.weights

    data = interval_rule_data(rule)
    if isinstance(factor, ProbabilityDomain):
        transport = factor.reference_transport
        if transport.reference_measure == "uniform":
            reference_nodes = data.nodes
        elif transport.reference_measure == "standard-normal":
            probabilities = open_unit_interval(0.5 * (data.nodes + 1.0))
            reference_nodes = jax.scipy.special.ndtri(probabilities)
        else:
            raise ValueError("Unsupported probability reference measure.")
        return transport.from_reference(reference_nodes), 0.5 * data.weights
    lower = jnp.asarray(factor.fixed("start"))
    upper = jnp.asarray(factor.fixed("end"))
    half = 0.5 * (upper - lower)
    center = 0.5 * (upper + lower)
    return center + half * data.nodes, half * data.weights


def _materialize_scalar_interiors(
    component: DomainComponent,
    target: ComponentTarget,
    rule: IntervalRule | ProbabilityRule,
    /,
) -> PointIntegrationBatch:
    rule_node_count = _fixed_rule_node_count(rule)
    fixed_labels = frozenset(
        label
        for label in component.domain.labels
        if isinstance(
            component.spec.selection_for(label),
            (Fixed, FixedStart, FixedEnd),
        )
    )
    blocks = tuple(
        (label,) for label in component.domain.labels if label not in fixed_labels
    )
    structure = SampleLayout(blocks).canonicalize(
        component.domain.labels,
        fixed_labels=fixed_labels,
    )
    points: dict[str, cx.Field] = {}
    weights_by_axis: dict[str, cx.Field] = {}
    for label in component.domain.labels:
        factor = component.domain.factor(label)

        selector = component.spec.selection_for(label)
        if not isinstance(factor, AbstractScalarDomain):
            raise TypeError("Scalar fixed quadrature requires scalar domain factors.")
        if isinstance(selector, Interior):
            axis = structure.axis_for(label)
            if axis is None:
                raise RuntimeError("Interior scalar factor has no integration axis.")
            values, weights = _scalar_interior_rule_data(factor, rule)
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
        PointBatch(frozendict(points), structure),
        total,
        axes=axes,
        target_mass=_component_base_mass(component),
        provenance=f"{type(rule).__name__}:{rule_node_count}",
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

        selector = component.spec.selection_for(label)
        if isinstance(selector, (Fixed, FixedStart, FixedEnd)):
            fixed_labels.add(label)
            if isinstance(selector, Fixed):
                value = selector.value
            elif isinstance(selector, FixedStart):
                if not isinstance(factor, AbstractScalarDomain):
                    raise TypeError("FixedStart requires a scalar domain factor.")
                value = factor.fixed("start")
            else:
                if not isinstance(factor, AbstractScalarDomain):
                    raise TypeError("FixedEnd requires a scalar domain factor.")
                value = factor.fixed("end")
            points[label] = cx.Field(jnp.asarray(value), dims=())
            continue
        if not isinstance(factor, AbstractScalarDomain):
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
            values, weights = _scalar_interior_rule_data(factor, rule)
        else:
            raise TypeError(f"Unsupported scalar selector {type(selector).__name__}.")
        points[label] = cx.Field(values, dims=(axis,))
        blocks.append((label,))
        weights_by_axis[axis] = cx.Field(weights, dims=(axis,))
    structure = SampleLayout(tuple(blocks)).canonicalize(
        component.domain.labels, fixed_labels=frozenset(fixed_labels)
    )
    batch = PointBatch(frozendict(points), structure)
    axes = axes_for_over(structure, target.axes)
    total = cx.Field(jnp.asarray(1.0), dims=())
    for axis in axes:
        total = total * weights_by_axis[axis]
    return PointIntegrationBatch(
        batch,
        total,
        axes=axes,
        target_mass=_component_base_mass(component),
        provenance=f"scalar-boundary:{type(rule).__name__}",
    )


def _materialize_boundary_atlas(
    component: DomainComponent,
    target: ComponentTarget,
    rule: IntervalRule,
    /,
) -> PointIntegrationBatch:
    from ._rules import ReferenceIntervalRule, ReferenceQuadrilateralRule

    varying = tuple(
        label
        for label in component.domain.labels
        if not isinstance(
            component.spec.selection_for(label), (Fixed, FixedStart, FixedEnd)
        )
    )
    if len(varying) != 1:
        raise ValueError(
            "Boundary-atlas quadrature supports one varying geometry factor; "
            "use ProductIntegrationPlan for mixed factors."
        )
    label = varying[0]
    selector = component.spec.selection_for(label)
    factor = component.domain.factor(label)

    if not isinstance(selector, Boundary) or not isinstance(
        factor, BoundaryAtlasProvider
    ):
        raise ValueError(
            "Fixed boundary quadrature requires a geometry with a boundary atlas."
        )
    atlas = factor.boundary_atlas
    if selector.tags is not None or selector.entity_ids is not None:
        atlas = atlas.select(tags=selector.tags, entity_ids=selector.entity_ids)
    if atlas.reference_dim == 1:
        reference_rule = ReferenceIntervalRule(rule)
    elif atlas.reference_dim == 2:
        reference_rule = ReferenceQuadrilateralRule(rule)
    else:
        raise ValueError(
            "Boundary-atlas quadrature supports reference dimensions one and two."
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
    active = (
        atlas.reference_mask(chart_indices, reference) & atlas.seam_owner[chart_indices]
    )
    weights = jnp.where(
        active,
        jacobian * reference_data.weights[None, :],
        0.0,
    )
    fixed_labels = frozenset(other for other in component.domain.labels if other != label)
    structure = SampleLayout(((label,),)).canonicalize(
        component.domain.labels, fixed_labels=fixed_labels
    )
    axis = structure.axis_for(label)
    if axis is None:
        raise RuntimeError("Boundary-atlas structure has no integration axis.")
    event_size = component.domain.coordinate(label).event_size
    points: dict[str, cx.Field] = {
        label: cx.Field(physical.reshape((-1, event_size)), dims=(axis, None))
    }
    for other in fixed_labels:
        selector_ = component.spec.selection_for(other)
        factor_ = component.domain.factor(other)

        if isinstance(selector_, Fixed):
            value = selector_.value
        elif isinstance(selector_, FixedStart):
            if not isinstance(factor_, AbstractScalarDomain):
                raise TypeError("FixedStart requires a scalar domain factor.")
            value = factor_.fixed("start")
        else:
            if not isinstance(factor_, AbstractScalarDomain):
                raise TypeError("FixedEnd requires a scalar domain factor.")
            value = factor_.fixed("end")
        dimensions = (None,) if isinstance(factor_, AbstractGeometry) else ()
        points[other] = cx.Field(jnp.asarray(value), dims=dimensions)
    axes = axes_for_over(structure, target.axes)
    return PointIntegrationBatch(
        PointBatch(
            frozendict({name: points[name] for name in component.domain.labels}),
            structure,
        ),
        cx.Field(weights.reshape((-1,)), dims=(axis,)),
        axes=axes,
        target_mass=_component_base_mass(component),
        provenance=f"boundary-atlas:{type(rule).__name__}",
    )


def _cubature_factor_data(
    factor: Any,
    selector: Any,
    rule: CubatureRule,
    /,
) -> tuple[Array, Array]:
    if not isinstance(factor, CubatureAtlasProvider):
        raise TypeError("The selected geometry does not expose native cubature.")
    if isinstance(selector, Interior):
        component_kind = "interior"
    elif isinstance(selector, Boundary):
        component_kind = "boundary"
    else:
        raise TypeError("Native cubature requires Interior() or Boundary().")
    atlas = factor.cubature_atlas(component_kind)
    if isinstance(selector, Boundary) and (
        selector.tags is not None or selector.entity_ids is not None
    ):
        atlas = atlas.select(tags=selector.tags, entity_ids=selector.entity_ids)
    if atlas.reference_domain != rule.reference_domain:
        raise ValueError(
            f"Cubature rule reference {rule.reference_domain!r} does not match "
            f"geometry reference {atlas.reference_domain!r}."
        )
    reference_data = rule.materialize()
    count = int(reference_data.points.shape[0])
    charts = atlas.num_charts
    reference = jnp.broadcast_to(
        reference_data.points[None, ...],
        (charts, count, atlas.reference_dimension),
    )
    chart_indices = jnp.broadcast_to(
        jnp.arange(charts, dtype=jnp.int32)[:, None],
        (charts, count),
    )
    evaluation = atlas.evaluate(chart_indices, reference)
    weights = jnp.where(
        evaluation.admissible,
        evaluation.measure_scale * reference_data.weights[None, :],
        jnp.asarray(jnp.nan, dtype=evaluation.measure_scale.dtype),
    )
    return (
        evaluation.points.reshape((-1, factor.spatial_dim)),
        weights.reshape((-1,)),
    )


def _materialize_cubature_atlas(
    component: DomainComponent,
    target: ComponentTarget,
    rule: CubatureRule,
    /,
) -> PointIntegrationBatch:
    varying = tuple(
        label
        for label in component.domain.labels
        if not isinstance(
            component.spec.selection_for(label), (Fixed, FixedStart, FixedEnd)
        )
    )
    if len(varying) != 1:
        raise ValueError(
            "Native cubature supports one varying geometry factor; "
            "use ProductIntegrationPlan for mixed factors."
        )
    label = varying[0]
    selector = component.spec.selection_for(label)
    factor = component.domain.factor(label)
    physical, weights = _cubature_factor_data(factor, selector, rule)
    fixed_labels = frozenset(other for other in component.domain.labels if other != label)
    structure = SampleLayout(((label,),)).canonicalize(
        component.domain.labels, fixed_labels=fixed_labels
    )
    axis = structure.axis_for(label)
    if axis is None:
        raise RuntimeError("Native cubature structure has no integration axis.")
    points: dict[str, cx.Field] = {label: cx.Field(physical, dims=(axis, None))}
    for other in fixed_labels:
        selector_ = component.spec.selection_for(other)
        factor_ = component.domain.factor(other)
        if isinstance(selector_, Fixed):
            value = selector_.value
        elif isinstance(selector_, FixedStart):
            if not isinstance(factor_, AbstractScalarDomain):
                raise TypeError("FixedStart requires a scalar domain factor.")
            value = factor_.fixed("start")
        else:
            if not isinstance(factor_, AbstractScalarDomain):
                raise TypeError("FixedEnd requires a scalar domain factor.")
            value = factor_.fixed("end")
        dimensions = (None,) if isinstance(factor_, AbstractGeometry) else ()
        points[other] = cx.Field(jnp.asarray(value), dims=dimensions)
    axes = axes_for_over(structure, target.axes)
    return PointIntegrationBatch(
        PointBatch(
            frozendict({name: points[name] for name in component.domain.labels}),
            structure,
        ),
        cx.Field(weights, dims=(axis,)),
        axes=axes,
        target_mass=jnp.sum(weights),
        provenance=rule.rule_id,
    )


def materialize_fixed_component(
    target: ComponentTarget,
    plan: FixedQuadraturePlan,
    /,
) -> PointIntegrationBatch | SeparableIntegrationBatch | tuple[Any, ...]:
    """Lower component geometry and interval rules into typed batches."""
    if isinstance(target.component, ComponentSum):
        return tuple(
            materialize_fixed_component(
                ComponentTarget(term, axes=target.axes, normalized=target.normalized),
                plan,
            )
            for term in target.component.terms
        )
    component = target.component
    selectors = tuple(
        component.spec.selection_for(label) for label in component.domain.labels
    )
    rule = plan.rule
    if isinstance(rule, CubatureRule):
        return _materialize_cubature_atlas(component, target, rule)
    if any(isinstance(selector, Boundary) for selector in selectors):
        factors = tuple(
            component.domain.factor(label) for label in component.domain.labels
        )
        unwrapped = tuple(factor for factor in factors)
        if all(isinstance(factor, AbstractScalarDomain) for factor in unwrapped):
            return _materialize_scalar_boundaries(component, target, rule)
        interval_rule_data(rule)
        return _materialize_boundary_atlas(component, target, rule)
    factors = tuple(component.domain.factor(label) for label in component.domain.labels)
    unwrapped_factors = tuple(factor for factor in factors)
    if all(isinstance(factor, AbstractScalarDomain) for factor in unwrapped_factors):
        return _materialize_scalar_interiors(component, target, rule)
    interval_rule_data(rule)
    spec = IntegrationAxisSpec(rule)
    coord_separable: dict[str, Any] = {}
    for label in component.domain.labels:
        selector = component.spec.selection_for(label)
        if isinstance(selector, (Fixed, FixedStart, FixedEnd)):
            continue
        if not isinstance(selector, Interior):
            raise ValueError(
                f"Unsupported fixed component selector {type(selector).__name__}."
            )
        factor = component.domain.factor(label)

        if isinstance(factor, AbstractGeometry):
            coord_separable[label] = (spec,) * int(factor.spatial_dim)
        elif isinstance(factor, AbstractScalarDomain):
            coord_separable[label] = spec
        else:
            raise TypeError(
                "Fixed quadrature currently supports scalar and geometric domain factors."
            )
    points = component.sample(
        GridSampling(coord_separable, design="uniform"),
        key=DOC_KEY0,
    )
    return materialize_sampled_component(target, points)


__all__ = [
    "IntegrationAxisSpec",
    "axes_for_over",
    "component_factor_fields",
    "first_field_leaf",
    "materialize_fixed_component",
    "materialize_sampled_component",
    "sum_over",
]
