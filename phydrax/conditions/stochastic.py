#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import TypeAlias

from phydrax.domain import ComponentSum, DomainComponent, DomainFunction

from ..metrix import RiemannianMetric
from ..operators.differential import (
    dt,
    fokker_planck_operator,
    kolmogorov_generator,
    probability_current,
    StochasticInterpretation,
)
from ._base import ConditionSupport, Residual
from ._field_ops import dot
from .boundary import _condition_value, ConditionValue


CoefficientField: TypeAlias = DomainFunction | str


def _validate_coefficients(
    diffusion: CoefficientField | None,
    covariance: CoefficientField | None,
    interpretation: StochasticInterpretation,
    /,
) -> None:
    if diffusion is not None and covariance is not None:
        raise ValueError("Provide either diffusion or covariance, not both.")
    if interpretation not in ("ito", "stratonovich"):
        raise ValueError("interpretation must be 'ito' or 'stratonovich'.")
    if interpretation == "stratonovich" and diffusion is None:
        raise ValueError(
            "Stratonovich conditions require diffusion; covariance alone cannot "
            "determine the drift correction."
        )


def _fields(
    primary: str,
    *coefficients: CoefficientField | None,
) -> tuple[str, ...]:
    names = [primary]
    names.extend(value for value in coefficients if isinstance(value, str))
    return tuple(dict.fromkeys(names))


def _coefficient(
    value: CoefficientField | None,
    functions: dict[str, DomainFunction],
    /,
    *,
    name: str,
) -> DomainFunction | None:
    if value is None:
        return None
    if isinstance(value, DomainFunction):
        return value
    if value not in functions:
        raise KeyError(f"Unknown {name} field {value!r}.")
    return functions[value]


def Kolmogorov(
    field: str,
    on: ConditionSupport,
    /,
    *,
    drift: CoefficientField,
    evolution_var: str | None,
    diffusion: CoefficientField | None = None,
    covariance: CoefficientField | None = None,
    metric: RiemannianMetric | None = None,
    interpretation: StochasticInterpretation = "ito",
    state_var: str = "x",
    label: str | None = None,
) -> Residual:
    """Backward Kolmogorov residual, stationary or time dependent."""
    if not isinstance(drift, (DomainFunction, str)):
        raise TypeError("drift must be a DomainFunction or field name.")
    _validate_coefficients(diffusion, covariance, interpretation)
    names = _fields(field, drift, diffusion, covariance)

    def residual(*values: DomainFunction) -> DomainFunction:
        functions = dict(zip(names, values, strict=True))
        observable = functions[field]
        drift_field = _coefficient(drift, functions, name="drift")
        diffusion_field = _coefficient(diffusion, functions, name="diffusion")
        covariance_field = _coefficient(covariance, functions, name="covariance")
        if drift_field is None:
            raise ValueError("Kolmogorov drift cannot be None.")
        generator = kolmogorov_generator(
            observable,
            drift_field,
            diffusion=diffusion_field,
            covariance=covariance_field,
            interpretation=interpretation,
            metric=metric,
            var=state_var,
        )
        if evolution_var is None:
            return generator
        return dt(observable, var=evolution_var) + generator

    return Residual(names, on, residual, label=label)


def FokkerPlanck(
    density_field: str,
    on: ConditionSupport,
    /,
    *,
    drift: CoefficientField,
    evolution_var: str | None,
    diffusion: CoefficientField | None = None,
    covariance: CoefficientField | None = None,
    metric: RiemannianMetric | None = None,
    interpretation: StochasticInterpretation = "ito",
    state_var: str = "x",
    label: str | None = None,
) -> Residual:
    """Fokker–Planck residual, stationary or time dependent."""
    if not isinstance(drift, (DomainFunction, str)):
        raise TypeError("drift must be a DomainFunction or field name.")
    _validate_coefficients(diffusion, covariance, interpretation)
    names = _fields(density_field, drift, diffusion, covariance)

    def residual(*values: DomainFunction) -> DomainFunction:
        functions = dict(zip(names, values, strict=True))
        density = functions[density_field]
        drift_field = _coefficient(drift, functions, name="drift")
        diffusion_field = _coefficient(diffusion, functions, name="diffusion")
        covariance_field = _coefficient(covariance, functions, name="covariance")
        if drift_field is None:
            raise ValueError("Fokker–Planck drift cannot be None.")
        adjoint = fokker_planck_operator(
            density,
            drift_field,
            diffusion=diffusion_field,
            covariance=covariance_field,
            interpretation=interpretation,
            metric=metric,
            var=state_var,
        )
        if evolution_var is None:
            return adjoint
        return dt(density, var=evolution_var) - adjoint

    return Residual(names, on, residual, label=label)


def ProbabilityFlux(
    density_field: str,
    on: DomainComponent,
    /,
    *,
    drift: CoefficientField,
    diffusion: CoefficientField | None = None,
    covariance: CoefficientField | None = None,
    target: ConditionValue | None = None,
    metric: RiemannianMetric | None = None,
    interpretation: StochasticInterpretation = "ito",
    state_var: str = "x",
    label: str | None = None,
) -> Residual:
    """Outward probability-current flux on one boundary component."""
    if isinstance(on, ComponentSum) or not isinstance(on, DomainComponent):
        raise TypeError("ProbabilityFlux requires one DomainComponent.")
    if not isinstance(drift, (DomainFunction, str)):
        raise TypeError("drift must be a DomainFunction or field name.")
    _validate_coefficients(diffusion, covariance, interpretation)
    names = _fields(density_field, drift, diffusion, covariance)
    normal = on.normal(var=state_var)
    target_value = _condition_value(target, on, 0.0)

    def residual(*values: DomainFunction) -> DomainFunction:
        functions = dict(zip(names, values, strict=True))
        density = functions[density_field]
        drift_field = _coefficient(drift, functions, name="drift")
        diffusion_field = _coefficient(diffusion, functions, name="diffusion")
        covariance_field = _coefficient(covariance, functions, name="covariance")
        if drift_field is None:
            raise ValueError("Probability-flux drift cannot be None.")
        current = probability_current(
            density,
            drift_field,
            diffusion=diffusion_field,
            covariance=covariance_field,
            metric=metric,
            interpretation=interpretation,
            var=state_var,
        )
        return dot(current, normal) - target_value

    return Residual(names, on, residual, label=label)


__all__ = ["CoefficientField", "FokkerPlanck", "Kolmogorov", "ProbabilityFlux"]
