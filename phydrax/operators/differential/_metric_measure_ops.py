#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike

import phydrax.ein as ein
from phydrax.domain import DomainFunction, ReferencedDensityField

from ..._strict import StrictModule
from ...metrix import WeightedRiemannianMeasure
from ._domain_ops import directional_derivative, grad
from ._riemannian_ops import (
    _dependencies,
    _geometry_contract,
    _positions,
    intrinsic_laplace_beltrami,
    riemannian_div,
    riemannian_grad,
)


class _WeightedLaplacianCallable(StrictModule):
    base: DomainFunction
    differential: DomainFunction
    measure: WeightedRiemannianMeasure
    base_positions: tuple[int, ...]
    differential_positions: tuple[int, ...]
    coordinate_position: int = eqx.field(static=True)

    def __init__(
        self,
        base: DomainFunction,
        differential: DomainFunction,
        measure: WeightedRiemannianMeasure,
        base_positions: tuple[int, ...],
        differential_positions: tuple[int, ...],
        coordinate_position: int,
        /,
    ):
        self.base = base
        self.differential = differential
        self.measure = measure
        self.base_positions = base_positions
        self.differential_positions = differential_positions
        self.coordinate_position = int(coordinate_position)

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        base = self.base.func(
            *[args[position] for position in self.base_positions], key=key, **kwargs
        )
        differential = jnp.asarray(
            self.differential.func(
                *[args[position] for position in self.differential_positions],
                key=key,
                **kwargs,
            )
        )
        coordinates = args[self.coordinate_position]
        log_weight_differential = jax.grad(self.measure._log_weight_point)(coordinates)
        correction = ein.contract(
            "...i,...ij,...j->...",
            log_weight_differential,
            self.measure.metric.inverse(coordinates),
            differential,
        )
        return base + correction


class _WeightedDivergenceCallable(StrictModule):
    base: DomainFunction
    field: DomainFunction
    measure: WeightedRiemannianMeasure
    base_positions: tuple[int, ...]
    field_positions: tuple[int, ...]
    coordinate_position: int = eqx.field(static=True)

    def __init__(
        self,
        base: DomainFunction,
        field: DomainFunction,
        measure: WeightedRiemannianMeasure,
        base_positions: tuple[int, ...],
        field_positions: tuple[int, ...],
        coordinate_position: int,
        /,
    ):
        self.base = base
        self.field = field
        self.measure = measure
        self.base_positions = base_positions
        self.field_positions = field_positions
        self.coordinate_position = int(coordinate_position)

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        base = self.base.func(
            *[args[position] for position in self.base_positions], key=key, **kwargs
        )
        vector = jnp.asarray(
            self.field.func(
                *[args[position] for position in self.field_positions],
                key=key,
                **kwargs,
            )
        )
        coordinates = args[self.coordinate_position]
        differential = jax.grad(self.measure._log_weight_point)(coordinates)
        return base + jnp.sum(vector * differential, axis=-1)


def weighted_riemannian_grad(
    function: DomainFunction,
    measure: WeightedRiemannianMeasure,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    """Return the metric gradient under a weighted Riemannian measure."""
    if not isinstance(measure, WeightedRiemannianMeasure):
        raise TypeError("measure must be a WeightedRiemannianMeasure.")
    return riemannian_grad(function, measure.metric, var=var, mode=mode)


def weighted_laplacian(
    function: DomainFunction,
    measure: WeightedRiemannianMeasure,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    """Return ``div_mu grad_g function``."""
    if not isinstance(measure, WeightedRiemannianMeasure):
        raise TypeError("measure must be a WeightedRiemannianMeasure.")
    var_, _ = _geometry_contract(function, measure.metric, var)
    base = intrinsic_laplace_beltrami(function, measure.metric, var=var_, mode=mode)
    differential = grad(function, var=var_, mode=mode)
    deps = _dependencies(function.domain.labels, (base, differential), var_)
    return DomainFunction(
        domain=function.domain,
        deps=deps,
        func=_WeightedLaplacianCallable(
            base,
            differential,
            measure,
            _positions(deps, base),
            _positions(deps, differential),
            deps.index(var_),
        ),
        metadata=base.metadata,
    )


def weighted_riemannian_div(
    field: DomainFunction,
    measure: WeightedRiemannianMeasure,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    """Return divergence relative to ``exp(log_weight) dvol_g``."""
    if not isinstance(measure, WeightedRiemannianMeasure):
        raise TypeError("measure must be a WeightedRiemannianMeasure.")
    var_, _ = _geometry_contract(field, measure.metric, var)
    base = riemannian_div(field, measure.metric, var=var_, mode=mode)
    deps = _dependencies(field.domain.labels, (base, field), var_)
    return DomainFunction(
        domain=field.domain,
        deps=deps,
        func=_WeightedDivergenceCallable(
            base,
            field,
            measure,
            _positions(deps, base),
            _positions(deps, field),
            deps.index(var_),
        ),
        metadata=base.metadata,
    )


def _weighted_density_contract(
    density: ReferencedDensityField,
    measure: WeightedRiemannianMeasure,
) -> DomainFunction:
    if not isinstance(density, ReferencedDensityField):
        raise TypeError("density must be a ReferencedDensityField.")
    if density.reference != "weighted-riemannian-volume":
        raise ValueError("Weighted forward operators require density relative to dmu.")
    if density.measure is not measure:
        raise ValueError("Density and operator must share one weighted measure.")
    return density.field


def weighted_kolmogorov_generator(
    observable: DomainFunction,
    drift: DomainFunction,
    measure: WeightedRiemannianMeasure,
    /,
    *,
    diffusivity: ArrayLike = 1.0,
    var: str | None = None,
) -> DomainFunction:
    """Return ``drift·df + diffusivity Δ_mu f``."""
    if not isinstance(measure, WeightedRiemannianMeasure):
        raise TypeError("measure must be a WeightedRiemannianMeasure.")
    var_, _ = _geometry_contract(observable, measure.metric, var)
    first = directional_derivative(observable, drift, var=var_)
    second = weighted_laplacian(observable, measure, var=var_)
    return first + jnp.asarray(diffusivity) * second


def weighted_probability_current(
    density: ReferencedDensityField,
    drift: DomainFunction,
    measure: WeightedRiemannianMeasure,
    /,
    *,
    diffusivity: ArrayLike = 1.0,
    var: str | None = None,
) -> DomainFunction:
    """Return current relative to ``dmu``."""
    field = _weighted_density_contract(density, measure)
    gradient = weighted_riemannian_grad(field, measure, var=var)
    return field * drift - jnp.asarray(diffusivity) * gradient


def weighted_fokker_planck_operator(
    density: ReferencedDensityField,
    drift: DomainFunction,
    measure: WeightedRiemannianMeasure,
    /,
    *,
    diffusivity: ArrayLike = 1.0,
    var: str | None = None,
) -> DomainFunction:
    """Return the forward operator for density relative to ``dmu``."""
    current = weighted_probability_current(
        density,
        drift,
        measure,
        diffusivity=diffusivity,
        var=var,
    )
    return -weighted_riemannian_div(current, measure, var=var)


__all__ = [
    "weighted_fokker_planck_operator",
    "weighted_kolmogorov_generator",
    "weighted_laplacian",
    "weighted_probability_current",
    "weighted_riemannian_div",
    "weighted_riemannian_grad",
]
