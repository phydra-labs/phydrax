#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any, TypeAlias

from jaxtyping import ArrayLike

from phydrax.domain import (
    ComponentSum,
    DomainComponent,
    DomainFunction,
    ReferencedDensityField,
)

from .._strict import StrictModule
from ..metrix import (
    RiemannianHypersurface,
    RiemannianMetric,
    WeightedRiemannianMeasure,
)
from ..operators.differential import (
    dt,
    fokker_planck_operator,
    kolmogorov_generator,
    probability_current,
    StochasticInterpretation,
    weighted_fokker_planck_operator,
    weighted_kolmogorov_generator,
    weighted_probability_current,
)
from ._base import ConditionSupport, Residual
from ._field_ops import dot
from .boundary import _condition_value, ConditionValue


CoefficientField: TypeAlias = DomainFunction | str


class _StochasticBoundaryOperator(StrictModule):
    expression: Any
    target: ConditionValue
    realization: Any
    context: Any

    def __call__(self, field: DomainFunction, /) -> DomainFunction:
        result = self.expression(field, self.realization, self.context)
        if not isinstance(result, DomainFunction):
            raise TypeError(
                "Stochastic boundary expression must return a DomainFunction."
            )
        return result - self.target


def StochasticBoundaryResidual(
    field: str,
    on: ConditionSupport,
    /,
    *,
    expression: Any,
    target: ConditionValue = 0.0,
    realization: Any = None,
    context: Any = None,
    label: str | None = None,
) -> Residual:
    """Build a replayable nonhomogeneous stochastic boundary residual.

    ``expression`` receives ``(field, realization, context)`` and must return a
    typed ``DomainFunction``.  The realization is fixed in the condition, so an
    exact deterministic objective never resamples a random boundary target.
    Nonlinear expressions remain residuals; this function makes no hard-enforcement
    claim.
    """
    if not isinstance(field, str) or not field:
        raise ValueError("field must be a non-empty function name.")
    if not callable(expression):
        raise TypeError("expression must be callable.")
    return Residual(
        field,
        on,
        _StochasticBoundaryOperator(
            expression,
            _condition_value(target, on, 0.0),
            realization,
            context,
        ),
        label=label,
    )


class _RiemannianNormalCallable(StrictModule):
    boundary: RiemannianHypersurface

    def __init__(self, boundary: RiemannianHypersurface, /):
        self.boundary = boundary

    def __call__(self, coordinates, /, *, key=None, **kwargs: Any):
        del key, kwargs
        return self.boundary.unit_normal(coordinates)


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
    boundary_geometry: RiemannianHypersurface | None = None,
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
    if boundary_geometry is None:
        normal = on.normal(var=state_var)
        resolved_metric = metric
    else:
        if not isinstance(boundary_geometry, RiemannianHypersurface):
            raise TypeError("boundary_geometry must be a RiemannianHypersurface.")
        if metric is not None and metric is not boundary_geometry.metric:
            raise ValueError(
                "ProbabilityFlux metric and boundary geometry must share one metric."
            )
        normal = DomainFunction(
            domain=on.domain,
            deps=(state_var,),
            func=_RiemannianNormalCallable(boundary_geometry),
            metadata={},
        )
        resolved_metric = boundary_geometry.metric
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
            metric=resolved_metric,
            interpretation=interpretation,
            var=state_var,
        )
        return dot(current, normal) - target_value

    return Residual(names, on, residual, label=label)


def WeightedKolmogorov(
    field: str,
    on: ConditionSupport,
    /,
    *,
    drift: CoefficientField,
    measure: WeightedRiemannianMeasure,
    evolution_var: str | None,
    diffusivity: ArrayLike = 1.0,
    state_var: str = "x",
    label: str | None = None,
) -> Residual:
    """Backward Kolmogorov residual relative to a weighted measure."""
    if not isinstance(drift, (DomainFunction, str)):
        raise TypeError("drift must be a DomainFunction or field name.")
    if not isinstance(measure, WeightedRiemannianMeasure):
        raise TypeError("measure must be a WeightedRiemannianMeasure.")
    names = _fields(field, drift)

    def residual(*values: DomainFunction) -> DomainFunction:
        functions = dict(zip(names, values, strict=True))
        observable = functions[field]
        drift_field = _coefficient(drift, functions, name="drift")
        if drift_field is None:
            raise ValueError("Weighted Kolmogorov drift cannot be None.")
        generator = weighted_kolmogorov_generator(
            observable,
            drift_field,
            measure,
            diffusivity=diffusivity,
            var=state_var,
        )
        if evolution_var is None:
            return generator
        return dt(observable, var=evolution_var) + generator

    return Residual(names, on, residual, label=label)


def WeightedFokkerPlanck(
    density_field: str,
    on: ConditionSupport,
    /,
    *,
    drift: CoefficientField,
    measure: WeightedRiemannianMeasure,
    evolution_var: str | None,
    diffusivity: ArrayLike = 1.0,
    state_var: str = "x",
    label: str | None = None,
) -> Residual:
    """Fokker–Planck residual for density relative to ``dmu``."""
    if not isinstance(drift, (DomainFunction, str)):
        raise TypeError("drift must be a DomainFunction or field name.")
    if not isinstance(measure, WeightedRiemannianMeasure):
        raise TypeError("measure must be a WeightedRiemannianMeasure.")
    names = _fields(density_field, drift)

    def residual(*values: DomainFunction) -> DomainFunction:
        functions = dict(zip(names, values, strict=True))
        density = functions[density_field]
        drift_field = _coefficient(drift, functions, name="drift")
        if drift_field is None:
            raise ValueError("Weighted Fokker–Planck drift cannot be None.")
        referenced = ReferencedDensityField(
            density,
            reference="weighted-riemannian-volume",
            state_var=state_var,
            measure=measure,
        )
        adjoint = weighted_fokker_planck_operator(
            referenced,
            drift_field,
            measure,
            diffusivity=diffusivity,
            var=state_var,
        )
        if evolution_var is None:
            return adjoint
        return dt(density, var=evolution_var) - adjoint

    return Residual(names, on, residual, label=label)


def WeightedProbabilityFlux(
    density_field: str,
    on: DomainComponent,
    /,
    *,
    drift: CoefficientField,
    measure: WeightedRiemannianMeasure,
    boundary_geometry: RiemannianHypersurface,
    diffusivity: ArrayLike = 1.0,
    target: ConditionValue | None = None,
    state_var: str = "x",
    label: str | None = None,
) -> Residual:
    """Intrinsic weighted probability-current flux."""
    if isinstance(on, ComponentSum) or not isinstance(on, DomainComponent):
        raise TypeError("WeightedProbabilityFlux requires one DomainComponent.")
    if not isinstance(boundary_geometry, RiemannianHypersurface):
        raise TypeError("boundary_geometry must be a RiemannianHypersurface.")
    if boundary_geometry.metric is not measure.metric:
        raise ValueError("Boundary geometry and measure must share one metric.")
    if not isinstance(drift, (DomainFunction, str)):
        raise TypeError("drift must be a DomainFunction or field name.")
    names = _fields(density_field, drift)
    normal = DomainFunction(
        domain=on.domain,
        deps=(state_var,),
        func=_RiemannianNormalCallable(boundary_geometry),
        metadata={},
    )
    target_value = _condition_value(target, on, 0.0)

    def residual(*values: DomainFunction) -> DomainFunction:
        functions = dict(zip(names, values, strict=True))
        density = ReferencedDensityField(
            functions[density_field],
            reference="weighted-riemannian-volume",
            state_var=state_var,
            measure=measure,
        )
        drift_field = _coefficient(drift, functions, name="drift")
        if drift_field is None:
            raise ValueError("Weighted probability-flux drift cannot be None.")
        current = weighted_probability_current(
            density,
            drift_field,
            measure,
            diffusivity=diffusivity,
            var=state_var,
        )
        return dot(current, normal) - target_value

    return Residual(names, on, residual, label=label)


__all__ = [
    "CoefficientField",
    "FokkerPlanck",
    "Kolmogorov",
    "StochasticBoundaryResidual",
    "ProbabilityFlux",
    "WeightedFokkerPlanck",
    "WeightedKolmogorov",
    "WeightedProbabilityFlux",
]
