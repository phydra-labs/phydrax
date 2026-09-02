#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Key

from phydrax.domain import (
    AbstractScalarDomain,
    ComponentSum,
    Fixed,
    FixedEnd,
    FixedStart,
    Interior,
    Interval1d,
    PointBatch,
    SampleLayout,
)

from .._doc import DOC_KEY0
from .._frozendict import frozendict
from .._strict import StrictModule
from ._batches import PointIntegrationBatch
from ._estimates import IntegrationEstimate, IntegrationProvenance
from ._fixed import integrate_fixed_component, integrate_fixed_density
from ._lowering import _component_base_mass
from ._plans import DiffraxCollocationQuadraturePlan
from ._precision import IntegrationPrecisionPolicy
from ._status import IntegrationStatus
from ._targets import ComponentTarget, DensityTarget


class DiffraxCollocationDiagnostics(StrictModule):
    status: Array
    num_evaluations: Array
    active_collocation: Array
    solver_successful: Array
    solver_id: str = eqx.field(static=True)


def _fixed_field(factor: Any, selector: Any, /) -> cx.Field:
    if isinstance(factor, Interval1d):
        if not isinstance(selector, Fixed):
            raise TypeError("A non-collocated Interval1d factor must select Fixed().")
        return cx.Field(jnp.asarray(selector.value).reshape((1,)), dims=(None,))
    if not isinstance(factor, AbstractScalarDomain):
        raise TypeError("Fixed Diffrax collocation factors must be scalar or Interval1d.")
    if isinstance(selector, FixedStart):
        value = factor.fixed("start")
    elif isinstance(selector, FixedEnd):
        value = factor.fixed("end")
    elif isinstance(selector, Fixed):
        value = selector.value
    else:
        raise TypeError("Non-collocated factors must be fixed.")
    return cx.Field(jnp.asarray(value).reshape(()), dims=())


def materialize_diffrax_collocation(
    target: ComponentTarget | DensityTarget,
    plan: DiffraxCollocationQuadraturePlan,
    /,
) -> PointIntegrationBatch:
    base = target.base if isinstance(target, DensityTarget) else target
    if not isinstance(base, ComponentTarget) or isinstance(base.component, ComponentSum):
        raise TypeError("Diffrax collocation requires one component target.")
    component = base.component
    varying = tuple(
        label
        for label in component.domain.labels
        if not isinstance(
            component.spec.selection_for(label), (Fixed, FixedStart, FixedEnd)
        )
    )
    if len(varying) != 1:
        raise ValueError("Diffrax collocation requires exactly one varying scalar label.")
    label = varying[0]
    factor = component.domain.factor(label)
    if not isinstance(factor, (AbstractScalarDomain, Interval1d)) or not isinstance(
        component.spec.selection_for(label), Interior
    ):
        raise TypeError(
            "Diffrax collocation varies one scalar or Interval1d Interior factor."
        )
    fixed_labels = frozenset(name for name in component.domain.labels if name != label)
    structure = SampleLayout(((label,),)).canonicalize(
        component.domain.labels, fixed_labels=fixed_labels
    )
    axis = structure.axis_for(label)
    if axis is None:
        raise RuntimeError("Diffrax collocation layout has no integration axis.")
    first_active = jnp.argmax(plan.active)
    safe_nodes = jnp.where(plan.active, plan.nodes, plan.nodes[first_active])
    point_values = safe_nodes[:, None] if isinstance(factor, Interval1d) else safe_nodes
    point_dims = (axis, None) if isinstance(factor, Interval1d) else (axis,)
    points = {
        label: cx.Field(point_values, dims=point_dims),
        **{
            name: _fixed_field(
                component.domain.factor(name), component.spec.selection_for(name)
            )
            for name in fixed_labels
        },
    }
    weights = jnp.where(plan.active, plan.weights, 0.0)
    return PointIntegrationBatch(
        PointBatch(frozendict(points), structure),
        cx.Field(weights, dims=(axis,)),
        axes=(axis,),
        mask=cx.Field(plan.active, dims=(axis,)),
        target_mass=_component_base_mass(component),
        provenance=f"diffrax-collocation:{plan.solver_id}",
    )


def integrate_diffrax_collocation(
    integrand: Any,
    target: ComponentTarget | DensityTarget,
    batch: PointIntegrationBatch,
    plan: DiffraxCollocationQuadraturePlan,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    kwargs: dict[str, Any] | None = None,
    precision: IntegrationPrecisionPolicy | None = None,
) -> IntegrationEstimate:
    precision_ = IntegrationPrecisionPolicy() if precision is None else precision
    estimate = (
        integrate_fixed_density(
            integrand, target, batch, key=key, kwargs=kwargs, precision=precision_
        )
        if isinstance(target, DensityTarget)
        else integrate_fixed_component(
            integrand, target, batch, key=key, kwargs=kwargs, precision=precision_
        )
    )
    status = jnp.where(
        plan.solver_successful,
        estimate.status,
        int(IntegrationStatus.DIFFRAX_SOLVE_FAILED),
    ).astype(jnp.int32)
    value = estimate.value
    if plan.throw:
        data = eqx.error_if(
            value.data,
            status != int(IntegrationStatus.CONVERGED),
            "Diffrax collocation solve or reduction failed.",
        )
        value = cx.Field(data, dims=value.dims)
    diagnostics = DiffraxCollocationDiagnostics(
        status=status,
        num_evaluations=jnp.sum(plan.active, dtype=jnp.int32),
        active_collocation=jnp.sum(plan.active, dtype=jnp.int32),
        solver_successful=plan.solver_successful,
        solver_id=plan.solver_id,
    )
    return IntegrationEstimate(
        value,
        status=status,
        num_evaluations=diagnostics.num_evaluations,
        error_estimate=None,
        error_kind=None,
        diagnostics=diagnostics,
        provenance=IntegrationProvenance(
            "diffrax-collocation", "component", plan.solver_id
        ),
    )


__all__ = [
    "DiffraxCollocationDiagnostics",
    "DiffraxCollocationQuadraturePlan",
    "integrate_diffrax_collocation",
    "materialize_diffrax_collocation",
]
