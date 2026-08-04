#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools
from typing import Any

import coordax as cx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Key

from .._doc import DOC_KEY0
from .._frozendict import frozendict
from .._numerics import (
    axis_level,
    normalize_axis_rules,
    smolyak_axis_data,
    smolyak_terms,
    SmolyakAxisRule,
)
from .._strict import StrictModule
from ..domain._base import _AbstractGeometry
from ..domain._components import (
    DomainComponentUnion,
    Fixed,
    FixedEnd,
    FixedStart,
    Interior,
)
from ..domain._domain import RelabeledDomain
from ..domain._probability import ProbabilityDomain
from ..domain._scalar import _AbstractScalarDomain
from ..domain._structure import PointsBatch, ProductStructure
from ._batches import PointIntegrationBatch
from ._estimates import (
    IntegrationEstimate,
    IntegrationProvenance,
    SparseGridDiagnostics,
)
from ._fixed import integrate_fixed_component, integrate_fixed_density
from ._plans import SparseGridPlan
from ._status import IntegrationStatus
from ._targets import ComponentTarget, DensityTarget


class SparseGridRealization(StrictModule):
    """A Smolyak batch and its immediately coarser comparison batch."""

    batch: PointIntegrationBatch
    previous: PointIntegrationBatch | None
    level: int
    num_unique_nodes: int
    previous_num_unique_nodes: int
    num_terms: int
    axis_rules: tuple[SmolyakAxisRule, ...]

    def __init__(
        self,
        batch: PointIntegrationBatch,
        previous: PointIntegrationBatch | None,
        /,
        *,
        level: int,
        num_terms: int,
        axis_rules: tuple[SmolyakAxisRule, ...],
    ):
        self.batch = batch
        self.previous = previous
        self.level = int(level)
        self.num_unique_nodes = int(batch.weights.data.size)
        self.previous_num_unique_nodes = (
            0 if previous is None else int(previous.weights.data.size)
        )
        self.num_terms = int(num_terms)
        self.axis_rules = axis_rules


def _unwrap(factor: Any, /) -> Any:
    return factor.base if isinstance(factor, RelabeledDomain) else factor


def _smolyak_rule(
    dimension: int,
    level: int,
    anisotropy: tuple[float, ...] | None,
    /,
    axis_rules: SmolyakAxisRule | tuple[SmolyakAxisRule, ...] = "clenshaw-curtis",
) -> tuple[np.ndarray, np.ndarray]:
    dimension_ = int(dimension)
    rules = normalize_axis_rules(
        dimension_,
        axis_rules,
        default="clenshaw-curtis",
        allowed=("clenshaw-curtis", "gauss-hermite"),
    )
    terms = smolyak_terms(dimension_, int(level), anisotropy)
    table: dict[
        tuple[tuple[str, int, int], ...],
        tuple[tuple[float, ...], float],
    ] = {}
    for term in terms:
        one_dimensional = tuple(
            smolyak_axis_data(rule, axis_level(term.index, axis))
            for axis, rule in enumerate(rules)
        )
        if any(data.quadrature_weights is None for data in one_dimensional):
            raise ValueError(
                "Every sparse-grid integration rule needs quadrature weights."
            )
        node_ranges = tuple(range(data.nodes.shape[0]) for data in one_dimensional)
        for position in itertools.product(*node_ranges):
            identifier = tuple(
                one_dimensional[axis].node_ids[point]
                for axis, point in enumerate(position)
            )
            node = tuple(
                float(one_dimensional[axis].nodes[point])
                for axis, point in enumerate(position)
            )
            weight = float(term.coefficient)
            for axis, point in enumerate(position):
                quadrature_weights = one_dimensional[axis].quadrature_weights
                assert quadrature_weights is not None
                weight *= float(quadrature_weights[point])
            if identifier in table:
                existing_node, existing_weight = table[identifier]
                table[identifier] = (existing_node, existing_weight + weight)
            else:
                table[identifier] = (node, weight)
    ordered = tuple(sorted(table))
    nodes = np.asarray(tuple(table[key][0] for key in ordered), dtype=float).reshape(
        (-1, dimension_)
    )
    weights = np.asarray(tuple(table[key][1] for key in ordered), dtype=float)
    weight_scale = max(1.0, float(np.max(np.abs(weights), initial=0.0)))
    active = np.abs(weights) > 64.0 * np.finfo(float).eps * weight_scale
    return nodes[active], weights[active]


def _fixed_field(factor: Any, selector: Any, /) -> cx.Field:
    factor = _unwrap(factor)
    if isinstance(factor, _AbstractScalarDomain):
        if isinstance(selector, FixedStart):
            value = factor.fixed("start")
        elif isinstance(selector, FixedEnd):
            value = factor.fixed("end")
        elif isinstance(selector, Fixed):
            value = selector.value
        else:
            raise TypeError("Expected a fixed scalar selector.")
        return cx.Field(jnp.asarray(value, dtype=float).reshape(()), dims=())
    if isinstance(factor, _AbstractGeometry) and isinstance(selector, Fixed):
        return cx.Field(
            jnp.asarray(selector.value, dtype=float).reshape((factor.var_dim,)),
            dims=(None,),
        )
    raise TypeError("Unsupported fixed sparse-grid factor.")


def _materialize_level(
    target: ComponentTarget,
    plan: SparseGridPlan,
    level: int,
    /,
) -> PointIntegrationBatch:
    component = target.component
    if isinstance(component, DomainComponentUnion):
        raise TypeError("Sparse-grid component unions must be integrated term by term.")
    fixed_labels = frozenset(
        label
        for label in component.domain.labels
        if isinstance(component.spec.component_for(label), (FixedStart, FixedEnd, Fixed))
    )
    varying = tuple(
        label for label in component.domain.labels if label not in fixed_labels
    )
    if target.axes is not None:
        requested = (target.axes,) if isinstance(target.axes, str) else target.axes
        if len(requested) != len(varying) or frozenset(requested) != frozenset(varying):
            raise ValueError(
                "Sparse-grid targets use one coupled axis; axes must select every "
                f"non-fixed label {varying!r}."
            )
    unsupported = tuple(
        label
        for label in varying
        if not isinstance(component.spec.component_for(label), Interior)
    )
    if unsupported:
        raise TypeError(
            "Sparse grids support only Interior() or fixed component selectors; "
            f"unsupported labels: {unsupported!r}."
        )
    if len(varying) != plan.dimension:
        raise ValueError(
            f"SparseGridPlan dimension={plan.dimension} but target has "
            f"{len(varying)} non-fixed factors."
        )
    factors = tuple(_unwrap(component.domain.factor(label)) for label in varying)
    if any(not isinstance(factor, _AbstractScalarDomain) for factor in factors):
        raise TypeError("Sparse grids currently support scalar and probability factors.")
    canonical_nodes, canonical_weights = _smolyak_rule(
        plan.dimension,
        level,
        plan.anisotropy,
        plan.axis_rules,
    )
    mapped_columns: list[Array] = []
    scale = jnp.asarray(1.0, dtype=float)
    for axis, (label, factor, rule) in enumerate(
        zip(varying, factors, plan.axis_rules, strict=True)
    ):
        coordinate = jnp.asarray(canonical_nodes[:, axis], dtype=float)
        if rule == "gauss-hermite":
            if not isinstance(factor, ProbabilityDomain):
                raise TypeError(
                    f"Gauss--Hermite axis {label!r} requires a probability factor."
                )
            if (
                not factor.supports_reference_transform
                or factor.reference_measure != "standard-normal"
            ):
                raise ValueError(
                    f"Gauss--Hermite axis {label!r} requires a standard-normal "
                    "reference transform."
                )
            mapped = factor.from_reference(coordinate)
        elif isinstance(factor, ProbabilityDomain):
            support = factor.distribution.support
            if support is None:
                raise ValueError(
                    f"Clenshaw--Curtis axis {label!r} requires bounded "
                    "probability support."
                )
            mapped = factor.distribution.icdf(0.5 * (coordinate + 1.0))
            scale = scale * 0.5
        else:
            lower = factor.fixed("start")
            upper = factor.fixed("end")
            mapped = 0.5 * (upper - lower) * coordinate + 0.5 * (upper + lower)
            scale = scale * 0.5 * (upper - lower)
        mapped_columns.append(jnp.asarray(mapped))
    structure = ProductStructure((varying,)).canonicalize(
        component.domain.labels, fixed_labels=fixed_labels
    )
    axis_name = structure.axis_for(varying[0])
    if axis_name is None:
        raise RuntimeError("Sparse-grid structure has no sample axis.")
    points: dict[str, cx.Field] = {}
    varying_index = {label: index for index, label in enumerate(varying)}
    for label in component.domain.labels:
        if label in fixed_labels:
            points[label] = _fixed_field(
                component.domain.factor(label), component.spec.component_for(label)
            )
        else:
            points[label] = cx.Field(
                mapped_columns[varying_index[label]], dims=(axis_name,)
            )
    point_batch = PointsBatch(frozendict(points), structure)
    weights = cx.Field(
        scale * jnp.asarray(canonical_weights, dtype=float), dims=(axis_name,)
    )
    return PointIntegrationBatch(
        point_batch,
        weights,
        axes=(axis_name,),
        target_mass=component.measure(),
        provenance=(f"smolyak:level-{level}:rules-{'+'.join(plan.axis_rules)}"),
    )


def materialize_sparse_grid(
    target: ComponentTarget | DensityTarget,
    plan: SparseGridPlan,
    /,
) -> SparseGridRealization:
    """Materialize a Smolyak rule and its immediately coarser comparison."""
    base = target.base if isinstance(target, DensityTarget) else target
    if not isinstance(base, ComponentTarget):
        raise TypeError("Sparse grids require a component-based target.")
    batch = _materialize_level(base, plan, plan.level)
    previous = None if plan.level == 1 else _materialize_level(base, plan, plan.level - 1)
    num_terms = len(smolyak_terms(plan.dimension, plan.level, plan.anisotropy))
    return SparseGridRealization(
        batch,
        previous,
        level=plan.level,
        num_terms=num_terms,
        axis_rules=plan.axis_rules,
    )


def integrate_sparse_grid(
    integrand: Any,
    target: ComponentTarget | DensityTarget,
    realization: SparseGridRealization,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    kwargs: dict[str, Any] | None = None,
) -> IntegrationEstimate:
    """Reduce a Smolyak batch and report its deterministic level difference."""
    if isinstance(target, DensityTarget):
        current = integrate_fixed_density(
            integrand, target, realization.batch, key=key, kwargs=kwargs
        )
        previous = (
            None
            if realization.previous is None
            else integrate_fixed_density(
                integrand, target, realization.previous, key=key, kwargs=kwargs
            )
        )
    else:
        current = integrate_fixed_component(
            integrand, target, realization.batch, key=key, kwargs=kwargs
        )
        previous = (
            None
            if realization.previous is None
            else integrate_fixed_component(
                integrand, target, realization.previous, key=key, kwargs=kwargs
            )
        )
    if previous is None:
        level_difference = None
        error = None
    else:
        level_difference = current.value.data - previous.value.data
        error = jnp.max(jnp.abs(level_difference))
    num_evaluations = current.num_evaluations + (
        0 if previous is None else previous.num_evaluations
    )
    status = jnp.where(
        jnp.all(jnp.isfinite(current.value.data)),
        current.status,
        int(IntegrationStatus.NONFINITE_INTEGRAND),
    )
    diagnostics = SparseGridDiagnostics(
        status=status,
        num_evaluations=num_evaluations,
        level_difference=level_difference,
        level=realization.level,
        num_unique_nodes=realization.num_unique_nodes,
        previous_num_unique_nodes=realization.previous_num_unique_nodes,
        num_terms=realization.num_terms,
        axis_rules=realization.axis_rules,
    )
    return IntegrationEstimate(
        current.value,
        status=status,
        num_evaluations=num_evaluations,
        error_estimate=error,
        error_kind="sparse-grid-level-difference" if error is not None else None,
        diagnostics=diagnostics,
        provenance=IntegrationProvenance(
            "sparse-grid", "component", realization.batch.provenance
        ),
    )


__all__ = [
    "SparseGridRealization",
    "integrate_sparse_grid",
    "materialize_sparse_grid",
]
