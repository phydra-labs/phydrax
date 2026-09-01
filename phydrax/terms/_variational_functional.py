#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import jax
import jax.numpy as jnp

from ..domain import Boundary, DomainFunction
from ..integration import (
    CallerIntegration,
    ComponentTarget,
    FixedIntegration,
    IntegrationSource,
    PerStepIntegration,
)
from ..operators import grad
from ..variational import (
    Functional,
    FunctionalContext,
    LocalFieldJet,
    LocalGeometry,
)
from ._integral_functional import IntegralFunctional


def _source_target(source: IntegrationSource, /):
    if isinstance(source, PerStepIntegration | CallerIntegration):
        return source.target
    if isinstance(source, FixedIntegration):
        return source.realization.target
    raise TypeError(
        "Functional bindings support fixed, per-step, or caller integration sources."
    )


def _stop_parameter_gradient(function: DomainFunction, /) -> DomainFunction:
    def _stopped(*args, key=None, **kwargs):
        return jax.lax.stop_gradient(function.func(*args, key=key, **kwargs))

    return DomainFunction(
        domain=function.domain,
        deps=function.deps,
        func=_stopped,
        metadata=function.metadata,
    )


def _term_integrand(
    term,
    fields: Mapping[str, DomainFunction],
    /,
    *,
    geometry_variable: str,
    pullback_fields: frozenset[str],
    source: IntegrationSource,
    context: FunctionalContext,
) -> DomainFunction:
    operands: list[DomainFunction] = []
    jet_indices: dict[str, tuple[int | None, int | None]] = {}
    for specification in term.fields:
        field = fields[specification.field_name]
        value_index = None
        gradient_index = None
        if specification.value:
            value = field
            if specification.field_name not in pullback_fields:
                value = _stop_parameter_gradient(value)
            value_index = len(operands)
            operands.append(value)
        if specification.gradient:
            gradient = grad(field, var=geometry_variable)
            if specification.field_name not in pullback_fields:
                gradient = _stop_parameter_gradient(gradient)
            gradient_index = len(operands)
            operands.append(gradient)
        jet_indices[specification.field_name] = (value_index, gradient_index)

    joined = operands[0].domain
    for operand in operands[1:]:
        joined = joined.join(operand.domain)
    coordinate = joined.Function(geometry_variable)(lambda point: point)
    coordinate_index = len(operands)
    operands.append(coordinate)

    normal_index = None
    target = _source_target(source)
    if term.normal:
        if not isinstance(target, ComponentTarget) or not isinstance(
            target.component.spec.selection_for(geometry_variable),
            Boundary,
        ):
            raise ValueError("A functional normal requires a boundary ComponentTarget.")
        normal_index = len(operands)
        operands.append(target.component.normal(var=geometry_variable))

    joined = operands[0].domain
    for operand in operands[1:]:
        joined = joined.join(operand.domain)
    promoted = tuple(operand.promote(joined) for operand in operands)
    dependencies = tuple(
        label
        for label in joined.labels
        if any(label in operand.deps for operand in promoted)
    )
    positions = {label: index for index, label in enumerate(dependencies)}
    operand_positions = tuple(
        tuple(positions[label] for label in operand.deps) for operand in promoted
    )

    def _density(*args, key=None, **kwargs):
        arrays = tuple(
            operand.func(
                *(args[index] for index in selected),
                key=key,
                **kwargs,
            )
            for operand, selected in zip(promoted, operand_positions, strict=True)
        )
        local_fields = {
            name: LocalFieldJet(
                value=None if value_index is None else arrays[value_index],
                gradient=(None if gradient_index is None else arrays[gradient_index]),
            )
            for name, (value_index, gradient_index) in jet_indices.items()
        }
        geometry = LocalGeometry(
            arrays[coordinate_index],
            normal=None if normal_index is None else arrays[normal_index],
        )
        return jnp.asarray(term.density(local_fields, geometry, context))

    return DomainFunction(domain=joined, deps=dependencies, func=_density, metadata={})


def bind_functional(
    functional: Functional,
    fields: Mapping[str, DomainFunction],
    sources: Mapping[str, IntegrationSource],
    /,
    *,
    geometry_variables: Mapping[str, str],
    pullback_fields: Sequence[str] | None = None,
    context: FunctionalContext | None = None,
    nonfinite_integrand: Literal["raise", "propagate"] = "raise",
) -> tuple[IntegralFunctional, ...]:
    """Bind one physical functional to DomainFunction integral terms.

    Resulting solver gradients are parameter pullbacks through the selected fields;
    they are not ambient physical-space functional derivatives.
    """
    if not isinstance(functional, Functional):
        raise TypeError("functional must be a variational.Functional.")
    expected_fields = set(functional.field_names)
    actual_fields = set(fields)
    if actual_fields != expected_fields:
        raise KeyError(
            "Functional field bindings must match exactly; "
            f"missing={tuple(sorted(expected_fields - actual_fields))}, "
            f"extra={tuple(sorted(actual_fields - expected_fields))}."
        )
    if any(not isinstance(field, DomainFunction) for field in fields.values()):
        raise TypeError("Functional field bindings must be DomainFunction values.")
    expected_regions = set(functional.region_names)
    actual_regions = set(sources)
    if actual_regions != expected_regions:
        raise KeyError(
            "Functional integration sources must match regions exactly; "
            f"missing={tuple(sorted(expected_regions - actual_regions))}, "
            f"extra={tuple(sorted(actual_regions - expected_regions))}."
        )
    if set(geometry_variables) != expected_regions:
        raise KeyError("geometry_variables must provide exactly one label per region.")
    selected_pullbacks = (
        functional.variable_fields
        if pullback_fields is None
        else tuple(str(name) for name in pullback_fields)
    )
    unknown_pullbacks = tuple(
        name for name in selected_pullbacks if name not in expected_fields
    )
    if unknown_pullbacks:
        raise KeyError(f"Unknown pullback fields {unknown_pullbacks}.")
    pullback_set = frozenset(selected_pullbacks)
    context_ = FunctionalContext() if context is None else context
    if not isinstance(context_, FunctionalContext):
        raise TypeError("context must be a FunctionalContext or None.")

    bound: list[IntegralFunctional] = []
    for term in functional.terms:
        source = sources[term.region]
        geometry_variable = str(geometry_variables[term.region])
        if not geometry_variable:
            raise ValueError("Functional geometry labels must be non-empty.")

        def integrand(
            current_fields: Mapping[str, DomainFunction],
            /,
            *,
            _term=term,
            _source=source,
            _geometry_variable=geometry_variable,
        ) -> DomainFunction:
            return _term_integrand(
                _term,
                current_fields,
                geometry_variable=_geometry_variable,
                pullback_fields=pullback_set,
                source=_source,
                context=context_,
            )

        term_pullbacks = tuple(
            name
            for name in selected_pullbacks
            if any(spec.field_name == name for spec in term.fields)
        )
        bound.append(
            IntegralFunctional(
                source=source,
                integrand=integrand,
                objective_vars=term_pullbacks,
                weight=term.weight,
                label=term.identifier,
                nonfinite_integrand=nonfinite_integrand,
            )
        )
    return tuple(bound)


__all__ = ["bind_functional"]
