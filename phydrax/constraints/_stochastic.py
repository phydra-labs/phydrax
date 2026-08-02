#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from jaxtyping import Array, ArrayLike, Key

from .._doc import DOC_KEY0
from ..domain._components import DomainComponent, DomainComponentUnion
from ..domain._function import DomainFunction
from ..domain._structure import CoordSeparableBatch, PointsBatch, ProductStructure
from ..metrix import RiemannianMetric
from ..operators.differential import (
    dt,
    fokker_planck_operator,
    kolmogorov_generator,
    StochasticInterpretation,
)
from ._adaptive import AbstractCollocationPolicy
from ._functional import FunctionalConstraint
from ._sampling_spec import SamplingNumPoints


CoefficientField = DomainFunction | str


def _validate_coefficients(
    *,
    diffusion: CoefficientField | None,
    covariance: CoefficientField | None,
    interpretation: StochasticInterpretation,
) -> None:
    if diffusion is not None and covariance is not None:
        raise ValueError("Provide either diffusion or covariance, not both.")
    if interpretation not in ("ito", "stratonovich"):
        raise ValueError("interpretation must be 'ito' or 'stratonovich'.")
    if interpretation == "stratonovich" and diffusion is None:
        raise ValueError(
            "Stratonovich constraints require diffusion; covariance alone cannot "
            "determine the drift correction."
        )


def _resolve_coefficient(
    value: CoefficientField | None,
    functions: Mapping[str, DomainFunction],
    /,
    *,
    name: str,
) -> DomainFunction | None:
    if value is None:
        return None
    if isinstance(value, DomainFunction):
        return value
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a DomainFunction, field name, or None.")
    try:
        resolved = functions[value]
    except KeyError as exc:
        raise KeyError(f"Unknown {name} field {value!r}.") from exc
    if not isinstance(resolved, DomainFunction):
        raise TypeError(f"Named {name} field {value!r} must be a DomainFunction.")
    return resolved


def _constraint_vars(
    primary: str,
    *coefficients: CoefficientField | None,
) -> tuple[str, ...]:
    names = [str(primary)]
    names.extend(value for value in coefficients if isinstance(value, str))
    return tuple(dict.fromkeys(names))


def ContinuousKolmogorovConstraint(
    constraint_var: str,
    component: DomainComponent | DomainComponentUnion,
    /,
    *,
    drift: CoefficientField,
    evolution_var: str | None,
    diffusion: CoefficientField | None = None,
    covariance: CoefficientField | None = None,
    metric: RiemannianMetric | None = None,
    interpretation: StochasticInterpretation = "ito",
    state_var: str = "x",
    num_points: SamplingNumPoints,
    structure: ProductStructure,
    dense_structure: ProductStructure | None = None,
    sampler: str = "latin_hypercube",
    weight: DomainFunction | ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    reduction: Literal["mean", "integral"] = "mean",
    sampling_mode: Literal["resample", "fixed"] = "resample",
    fixed_batch: (
        PointsBatch | CoordSeparableBatch | tuple[PointsBatch, ...] | None
    ) = None,
    fixed_batch_key: Key[Array, ""] = DOC_KEY0,
    collocation_policy: AbstractCollocationPolicy | None = None,
) -> FunctionalConstraint:
    r"""Create a sampled backward Kolmogorov residual constraint.

    With ``evolution_var`` set, the residual is
    :math:`\partial_t u+\mathcal{L}u`. With ``evolution_var=None``, it is the
    stationary residual :math:`\mathcal{L}u`. Coefficients may be fixed
    ``DomainFunction`` objects or names in the solver's function mapping; named
    coefficients remain jointly trainable. ``metric`` selects the covariant
    generator while preserving coordinate Itô drift inputs.
    """
    if not isinstance(constraint_var, str) or not constraint_var:
        raise ValueError("constraint_var must be a non-empty field name.")
    if not isinstance(drift, (DomainFunction, str)):
        raise TypeError("drift must be a DomainFunction or field name.")
    _validate_coefficients(
        diffusion=diffusion,
        covariance=covariance,
        interpretation=interpretation,
    )

    def residual(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
        observable = functions[constraint_var]
        drift_field = _resolve_coefficient(drift, functions, name="drift")
        diffusion_field = _resolve_coefficient(diffusion, functions, name="diffusion")
        covariance_field = _resolve_coefficient(covariance, functions, name="covariance")
        assert drift_field is not None
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

    return FunctionalConstraint(
        component=component,
        residual=residual,
        constraint_vars=_constraint_vars(constraint_var, drift, diffusion, covariance),
        num_points=num_points,
        structure=structure,
        dense_structure=dense_structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
        reduction=reduction,
        sampling_mode=sampling_mode,
        fixed_batch=fixed_batch,
        fixed_batch_key=fixed_batch_key,
        collocation_policy=collocation_policy,
    )


def ContinuousFokkerPlanckConstraint(
    constraint_var: str,
    component: DomainComponent | DomainComponentUnion,
    /,
    *,
    drift: CoefficientField,
    evolution_var: str | None,
    diffusion: CoefficientField | None = None,
    covariance: CoefficientField | None = None,
    metric: RiemannianMetric | None = None,
    interpretation: StochasticInterpretation = "ito",
    state_var: str = "x",
    num_points: SamplingNumPoints,
    structure: ProductStructure,
    dense_structure: ProductStructure | None = None,
    sampler: str = "latin_hypercube",
    weight: DomainFunction | ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    reduction: Literal["mean", "integral"] = "mean",
    sampling_mode: Literal["resample", "fixed"] = "resample",
    fixed_batch: (
        PointsBatch | CoordSeparableBatch | tuple[PointsBatch, ...] | None
    ) = None,
    fixed_batch_key: Key[Array, ""] = DOC_KEY0,
    collocation_policy: AbstractCollocationPolicy | None = None,
) -> FunctionalConstraint:
    r"""Create a sampled Fokker--Planck residual constraint.

    With ``evolution_var`` set, the residual is
    :math:`\partial_t p-\mathcal{L}^*p`. With ``evolution_var=None``, it is the
    stationary residual :math:`\mathcal{L}^*p`. When ``metric`` is supplied,
    density is relative to Riemannian volume. Positivity, normalization, initial
    data, and boundary conditions are deliberately separate constraints or ansatzes.
    """
    if not isinstance(constraint_var, str) or not constraint_var:
        raise ValueError("constraint_var must be a non-empty field name.")
    if not isinstance(drift, (DomainFunction, str)):
        raise TypeError("drift must be a DomainFunction or field name.")
    _validate_coefficients(
        diffusion=diffusion,
        covariance=covariance,
        interpretation=interpretation,
    )

    def residual(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
        density = functions[constraint_var]
        drift_field = _resolve_coefficient(drift, functions, name="drift")
        diffusion_field = _resolve_coefficient(diffusion, functions, name="diffusion")
        covariance_field = _resolve_coefficient(covariance, functions, name="covariance")
        assert drift_field is not None
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

    return FunctionalConstraint(
        component=component,
        residual=residual,
        constraint_vars=_constraint_vars(constraint_var, drift, diffusion, covariance),
        num_points=num_points,
        structure=structure,
        dense_structure=dense_structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
        reduction=reduction,
        sampling_mode=sampling_mode,
        fixed_batch=fixed_batch,
        fixed_batch_key=fixed_batch_key,
        collocation_policy=collocation_policy,
    )


__all__ = [
    "CoefficientField",
    "ContinuousFokkerPlanckConstraint",
    "ContinuousKolmogorovConstraint",
]
