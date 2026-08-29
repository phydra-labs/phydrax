#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from phydrax.conditions import AbstractResidualCondition
from phydrax.domain import DomainFunction
from phydrax.operators.differential import (
    level_set_coarea_density,
    level_set_phase_indicator,
)

from ._integral_functional import IntegralFunctional
from ._residual import _squared_frobenius_field


def implicit_interface_penalty(
    condition: AbstractResidualCondition,
    target: Any,
    /,
    *,
    level_set_field: str,
    width: float,
    plan: Any = None,
    spatial_var: str = "x",
    mode: Literal["reverse", "forward"] = "reverse",
    scale: float = 1.0,
    label: str | None = None,
    materialization_policy: Literal["fixed", "per_step", "caller"] = "per_step",
) -> IntegralFunctional:
    """Integrate a squared residual over an implicit interface via coarea."""

    if not isinstance(condition, AbstractResidualCondition):
        raise TypeError("condition must be an AbstractResidualCondition.")
    name = str(level_set_field)
    if not name:
        raise ValueError("level_set_field must be non-empty.")

    def integrand(functions) -> DomainFunction:
        if name not in functions:
            raise KeyError(f"Missing level-set field {name!r}.")
        residual = condition.residual(functions)
        density = level_set_coarea_density(
            functions[name],
            width=width,
            var=spatial_var,
            mode=mode,
        )
        return density * _squared_frobenius_field(residual)

    fields = tuple(dict.fromkeys(condition.fields + (name,)))
    return IntegralFunctional(
        target=target,
        integrand=integrand,
        plan=plan,
        objective_vars=fields,
        weight=scale,
        label=condition.label if label is None else label,
        materialization_policy=materialization_policy,
    )


def implicit_phase_penalty(
    condition: AbstractResidualCondition,
    target: Any,
    /,
    *,
    level_set_field: str,
    width: float,
    phase: Literal["inside", "outside"],
    plan: Any = None,
    scale: float = 1.0,
    label: str | None = None,
    materialization_policy: Literal["fixed", "per_step", "caller"] = "per_step",
) -> IntegralFunctional:
    """Integrate a squared residual over one diffuse phase of an ambient domain."""

    if not isinstance(condition, AbstractResidualCondition):
        raise TypeError("condition must be an AbstractResidualCondition.")
    name = str(level_set_field)
    if not name:
        raise ValueError("level_set_field must be non-empty.")
    if phase not in ("inside", "outside"):
        raise ValueError("phase must be 'inside' or 'outside'.")

    def integrand(functions) -> DomainFunction:
        if name not in functions:
            raise KeyError(f"Missing level-set field {name!r}.")
        residual = condition.residual(functions)
        indicator = level_set_phase_indicator(
            functions[name],
            width=width,
            phase=phase,
        )
        return indicator * _squared_frobenius_field(residual)

    fields = tuple(dict.fromkeys(condition.fields + (name,)))
    return IntegralFunctional(
        target=target,
        integrand=integrand,
        plan=plan,
        objective_vars=fields,
        weight=scale,
        label=condition.label if label is None else label,
        materialization_policy=materialization_policy,
    )


def free_boundary_term_suite(
    phase_terms: Sequence[IntegralFunctional],
    interface_terms: Sequence[IntegralFunctional],
    /,
) -> tuple[IntegralFunctional, ...]:
    """Validate and concatenate phase and interface functionals in stable order."""

    terms = tuple(phase_terms) + tuple(interface_terms)
    if not terms or any(not isinstance(term, IntegralFunctional) for term in terms):
        raise TypeError(
            "A free-boundary term suite requires one or more IntegralFunctional terms."
        )
    return terms


__all__ = [
    "free_boundary_term_suite",
    "implicit_interface_penalty",
    "implicit_phase_penalty",
]
