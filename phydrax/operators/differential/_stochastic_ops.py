#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import jax
import jax.numpy as jnp
import opt_einsum as oe

from ..._strict import StrictModule
from ...domain._base import _AbstractGeometry
from ...domain._function import DomainFunction
from ...domain._scalar import _AbstractScalarDomain
from ...metrix import LeviCivitaConnection, RiemannianMetric
from ._domain_ops import (
    _factor_and_dim,
    _strip_derivative_hook_metadata,
    div,
    div_tensor,
    grad,
    hessian,
)
from ._riemannian_ops import (
    covariant_hessian as riemannian_covariant_hessian,
    riemannian_div,
    riemannian_div_tensor,
)
from ._stochastic_estimators import (
    directional_stratonovich_correction,
    factor_hvp_contraction,
)


StochasticInterpretation = Literal["ito", "stratonovich"]
GeneratorContraction: TypeAlias = Literal["auto", "factor_hvp", "dense"]


def _require_function(value: Any, name: str, /) -> DomainFunction:
    if not isinstance(value, DomainFunction):
        raise TypeError(f"{name} must be a DomainFunction.")
    return value


def _require_interpretation(value: str, /) -> StochasticInterpretation:
    if value not in ("ito", "stratonovich"):
        raise ValueError("interpretation must be 'ito' or 'stratonovich'.")
    return value


def _join_fields(*fields: DomainFunction) -> tuple[Any, tuple[DomainFunction, ...]]:
    domain = fields[0].domain
    for field in fields[1:]:
        if domain.labels != field.domain.labels:
            domain = domain.join(field.domain)
    return domain, tuple(field.promote(domain) for field in fields)


def _union_deps(domain: Any, *fields: DomainFunction) -> tuple[str, ...]:
    used = {label for field in fields for label in field.deps}
    return tuple(label for label in domain.labels if label in used)


def _positions(deps: tuple[str, ...], field: DomainFunction, /) -> tuple[int, ...]:
    index = {label: i for i, label in enumerate(deps)}
    return tuple(index[label] for label in field.deps)


def _field_args(args: tuple[Any, ...], positions: tuple[int, ...], /) -> list[Any]:
    return [args[index] for index in positions]


def _batch_ndim(field: DomainFunction, args: list[Any], /) -> int:
    """Infer sampled axes while excluding a geometry coordinate's value axis."""
    ranks: list[int] = []
    for label, value in zip(field.deps, args, strict=True):
        factor = field.domain.factor(label)
        factor = getattr(factor, "base", factor)
        if isinstance(value, tuple):
            ranks.extend(jnp.asarray(item).ndim for item in value)
            continue
        rank = jnp.asarray(value).ndim
        if isinstance(factor, _AbstractGeometry):
            rank = max(0, rank - 1)
        elif not isinstance(factor, _AbstractScalarDomain):
            continue
        ranks.append(rank)
    return max(ranks, default=0)


class _DiffusionCovarianceCallable(StrictModule):
    diffusion: DomainFunction

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        sigma = jnp.asarray(self.diffusion.func(*args, key=key, **kwargs))
        if sigma.ndim < 2:
            raise ValueError("diffusion must have trailing shape (state_dim, noise_dim).")
        return sigma @ jnp.swapaxes(sigma, -1, -2)


class _StratonovichCorrectionCallable(StrictModule):
    diffusion: DomainFunction
    state_position: int
    state_dim: int

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        state = jnp.asarray(args[self.state_position])
        if state.shape != (self.state_dim,):
            raise ValueError(
                f"state coordinate must have shape ({self.state_dim},); got {state.shape}."
            )

        def evaluate(value):
            local_args = list(args)
            local_args[self.state_position] = value
            return self.diffusion.func(*local_args, key=key, **kwargs)

        return directional_stratonovich_correction(evaluate, state)


class _CoordinateToCovariantDriftCallable(StrictModule):
    drift: DomainFunction
    covariance: DomainFunction
    metric: RiemannianMetric
    drift_positions: tuple[int, ...]
    covariance_positions: tuple[int, ...]
    coordinate_position: int
    state_dim: int

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        drift = jnp.asarray(
            self.drift.func(
                *_field_args(args, self.drift_positions),
                key=key,
                **kwargs,
            )
        )
        covariance = jnp.asarray(
            self.covariance.func(
                *_field_args(args, self.covariance_positions),
                key=key,
                **kwargs,
            )
        )
        expected = (self.state_dim, self.state_dim)
        if drift.shape[-1:] != (self.state_dim,):
            raise ValueError(
                f"drift must have trailing shape ({self.state_dim},); got {drift.shape}."
            )
        if covariance.shape[-2:] != expected:
            raise ValueError(
                f"covariance must have trailing shape {expected}; got {covariance.shape}."
            )
        coefficients = LeviCivitaConnection(self.metric).coefficients(
            args[self.coordinate_position]
        )
        return drift + 0.5 * oe.contract(
            "...kij,...ij->...k",
            coefficients,
            covariance,
        )


class _KolmogorovCallable(StrictModule):
    observable_gradient: DomainFunction
    observable_hessian: DomainFunction | None
    drift: DomainFunction
    covariance: DomainFunction | None
    gradient_positions: tuple[int, ...]
    hessian_positions: tuple[int, ...]
    drift_positions: tuple[int, ...]
    covariance_positions: tuple[int, ...]
    state_dim: int

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        gradient = jnp.asarray(
            self.observable_gradient.func(
                *_field_args(args, self.gradient_positions), key=key, **kwargs
            )
        )
        drift = jnp.asarray(
            self.drift.func(*_field_args(args, self.drift_positions), key=key, **kwargs)
        )
        if gradient.ndim < 1 or gradient.shape[-1] != self.state_dim:
            raise ValueError(
                f"grad(observable) must end in state dimension {self.state_dim}; "
                f"got {gradient.shape}."
            )
        if drift.ndim < 1 or drift.shape[-1] != self.state_dim:
            raise ValueError(
                f"drift must have trailing shape ({self.state_dim},); got {drift.shape}."
            )
        extra = gradient.ndim - drift.ndim
        if extra < 0:
            raise ValueError("drift and observable gradient have incompatible ranks.")
        drift_expanded = drift.reshape(
            drift.shape[:-1] + (1,) * extra + (self.state_dim,)
        )
        out = jnp.sum(drift_expanded * gradient, axis=-1)

        if self.covariance is None:
            return out
        assert self.observable_hessian is not None
        hessian_value = jnp.asarray(
            self.observable_hessian.func(
                *_field_args(args, self.hessian_positions), key=key, **kwargs
            )
        )
        covariance = jnp.asarray(
            self.covariance.func(
                *_field_args(args, self.covariance_positions), key=key, **kwargs
            )
        )
        expected = (self.state_dim, self.state_dim)
        if hessian_value.ndim < 2 or hessian_value.shape[-2:] != expected:
            raise ValueError(
                f"hessian(observable) must end in {expected}; got {hessian_value.shape}."
            )
        if covariance.ndim < 2 or covariance.shape[-2:] != expected:
            raise ValueError(
                f"covariance must have trailing shape {expected}; got {covariance.shape}."
            )
        extra = hessian_value.ndim - covariance.ndim
        if extra < 0:
            raise ValueError("covariance and observable Hessian have incompatible ranks.")
        covariance_expanded = covariance.reshape(
            covariance.shape[:-2] + (1,) * extra + expected
        )
        return out + 0.5 * jnp.sum(covariance_expanded * hessian_value, axis=(-2, -1))


class _FactorHVPKolmogorovCallable(StrictModule):
    observable: DomainFunction
    drift: DomainFunction
    diffusion: DomainFunction
    observable_positions: tuple[int, ...]
    drift_positions: tuple[int, ...]
    diffusion_positions: tuple[int, ...]
    state_position: int
    observable_state_position: int
    state_dim: int

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        observable_args = _field_args(args, self.observable_positions)
        state = jnp.asarray(args[self.state_position])
        if state.shape != (self.state_dim,):
            raise ValueError(
                f"state coordinate must have shape ({self.state_dim},); got {state.shape}."
            )
        drift = jnp.asarray(
            self.drift.func(*_field_args(args, self.drift_positions), key=key, **kwargs)
        )
        diffusion = jnp.asarray(
            self.diffusion.func(
                *_field_args(args, self.diffusion_positions), key=key, **kwargs
            )
        )
        if drift.shape != state.shape:
            raise ValueError(
                f"drift must have trailing shape ({self.state_dim},); got {drift.shape}."
            )
        if diffusion.ndim != 2 or diffusion.shape[0] != self.state_dim:
            raise ValueError(
                "diffusion must have trailing shape "
                f"({self.state_dim}, noise_dim); got {diffusion.shape}."
            )

        def evaluate(value):
            if self.observable_state_position < 0:
                return self.observable.func(*observable_args, key=key, **kwargs)
            local_args = list(observable_args)
            local_args[self.observable_state_position] = value
            return self.observable.func(*local_args, key=key, **kwargs)

        gradient = jax.jacrev(evaluate)(state)
        drift_expanded = drift.reshape((1,) * (gradient.ndim - state.ndim) + state.shape)
        first_order = jnp.sum(gradient * drift_expanded, axis=-1)
        second_order = factor_hvp_contraction(evaluate, state, diffusion)
        return first_order + 0.5 * second_order


class _DensityCoefficientProductCallable(StrictModule):
    density: DomainFunction
    coefficient: DomainFunction
    density_positions: tuple[int, ...]
    coefficient_positions: tuple[int, ...]
    state_dim: int
    tensor: bool

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        density_args = _field_args(args, self.density_positions)
        coefficient_args = _field_args(args, self.coefficient_positions)
        density = jnp.asarray(self.density.func(*density_args, key=key, **kwargs))
        expected_batch_ndim = _batch_ndim(self.density, density_args)
        if density.ndim not in (0, expected_batch_ndim):
            raise ValueError("density must be scalar-valued at each state-time point.")
        coefficient = jnp.asarray(
            self.coefficient.func(*coefficient_args, key=key, **kwargs)
        )
        trailing = (self.state_dim, self.state_dim) if self.tensor else (self.state_dim,)
        if (
            coefficient.ndim < len(trailing)
            or coefficient.shape[-len(trailing) :] != trailing
        ):
            name = "covariance" if self.tensor else "drift"
            raise ValueError(
                f"{name} must have trailing shape {trailing}; got {coefficient.shape}."
            )
        return coefficient * density.reshape(density.shape + (1,) * len(trailing))


def diffusion_covariance(diffusion: DomainFunction, /) -> DomainFunction:
    r"""Return the covariance field :math:`a=\sigma\sigma^\mathsf{T}`.

    ``diffusion`` must be matrix-valued with trailing shape
    ``(state_dim, noise_dim)``. Rectangular diffusion fields are supported.
    """
    sigma = _require_function(diffusion, "diffusion")
    return DomainFunction(
        domain=sigma.domain,
        deps=sigma.deps,
        func=_DiffusionCovarianceCallable(sigma),
        metadata=_strip_derivative_hook_metadata(sigma.metadata),
    )


def stratonovich_to_ito_drift(
    drift: DomainFunction,
    diffusion: DomainFunction,
    /,
    *,
    var: str = "x",
) -> DomainFunction:
    r"""Convert a Euclidean Stratonovich drift to its Itô drift.

    For diffusion columns :math:`\sigma_{\cdot k}`, the correction is

    .. math::

        b_i^I=b_i^S+\tfrac12\sum_{j,k}\sigma_{jk}\,\partial_j\sigma_{ik}.
    """
    drift_field = _require_function(drift, "drift")
    sigma = _require_function(diffusion, "diffusion")
    _, state_dim = _factor_and_dim(sigma, var)
    domain, (drift_promoted, sigma_promoted) = _join_fields(drift_field, sigma)
    if var not in sigma_promoted.deps:
        return drift_promoted
    correction = DomainFunction(
        domain=domain,
        deps=sigma_promoted.deps,
        func=_StratonovichCorrectionCallable(
            sigma_promoted,
            sigma_promoted.deps.index(var),
            int(state_dim),
        ),
        metadata=_strip_derivative_hook_metadata(sigma_promoted.metadata),
    )
    return drift_promoted + correction


def _coordinate_to_covariant_drift(
    drift: DomainFunction,
    covariance: DomainFunction,
    metric: RiemannianMetric,
    /,
    *,
    var: str,
    state_dim: int,
) -> DomainFunction:
    domain, (drift_promoted, covariance_promoted) = _join_fields(drift, covariance)
    deps = tuple(
        label
        for label in domain.labels
        if label == var
        or label in drift_promoted.deps
        or label in covariance_promoted.deps
    )
    return DomainFunction(
        domain=domain,
        deps=deps,
        func=_CoordinateToCovariantDriftCallable(
            drift=drift_promoted,
            covariance=covariance_promoted,
            metric=metric,
            drift_positions=_positions(deps, drift_promoted),
            covariance_positions=_positions(deps, covariance_promoted),
            coordinate_position=deps.index(var),
            state_dim=int(state_dim),
        ),
        metadata=_strip_derivative_hook_metadata(drift_promoted.metadata),
    )


def _resolved_coefficients(
    drift: DomainFunction,
    *,
    diffusion: DomainFunction | None,
    covariance: DomainFunction | None,
    interpretation: StochasticInterpretation,
    var: str,
    metric: RiemannianMetric | None,
    state_dim: int,
) -> tuple[DomainFunction, DomainFunction | None]:
    if diffusion is not None and covariance is not None:
        raise ValueError("Provide either diffusion or covariance, not both.")
    if interpretation == "stratonovich":
        if diffusion is None:
            raise ValueError(
                "Stratonovich operators require diffusion; covariance alone cannot "
                "determine the drift correction."
            )
        resolved_drift = stratonovich_to_ito_drift(drift, diffusion, var=var)
        resolved_covariance = diffusion_covariance(diffusion)
    else:
        resolved_drift = drift
        resolved_covariance = (
            diffusion_covariance(diffusion) if diffusion is not None else covariance
        )
    if metric is not None and resolved_covariance is not None:
        resolved_drift = _coordinate_to_covariant_drift(
            resolved_drift,
            resolved_covariance,
            metric,
            var=var,
            state_dim=state_dim,
        )
    return resolved_drift, resolved_covariance


def kolmogorov_generator(
    observable: DomainFunction,
    drift: DomainFunction,
    /,
    *,
    diffusion: DomainFunction | None = None,
    covariance: DomainFunction | None = None,
    metric: RiemannianMetric | None = None,
    interpretation: StochasticInterpretation = "ito",
    contraction: GeneratorContraction = "auto",
    var: str = "x",
) -> DomainFunction:
    r"""Apply a backward Kolmogorov generator to an observable.

    Scalar, vector, and tensor observables are handled componentwise. With a
    diffusion factor, ``contraction="auto"`` uses exact factor-Hessian-vector
    products and never materializes a Hessian or covariance. Dense contraction is
    retained for an explicitly supplied covariance and for Riemannian covariant
    Hessians. For ``interpretation="stratonovich"``, coordinate diffusion fields
    are first converted to Itô coefficients.
    """
    observable_field = _require_function(observable, "observable")
    drift_field = _require_function(drift, "drift")
    diffusion_field = (
        None if diffusion is None else _require_function(diffusion, "diffusion")
    )
    covariance_field = (
        None if covariance is None else _require_function(covariance, "covariance")
    )
    interpretation_value = _require_interpretation(interpretation)
    if contraction not in ("auto", "factor_hvp", "dense"):
        raise ValueError("contraction must be 'auto', 'factor_hvp', or 'dense'.")
    if contraction == "factor_hvp" and diffusion_field is None:
        raise ValueError("factor_hvp contraction requires a diffusion factor.")
    if contraction == "factor_hvp" and metric is not None:
        raise ValueError(
            "factor_hvp contraction is currently Euclidean; use dense for a metric."
        )
    factor, state_dim = _factor_and_dim(observable_field, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError("var must name a geometry state variable.")
    if metric is not None and metric.chart.dimension != state_dim:
        raise ValueError(
            f"Metric chart dimension {metric.chart.dimension} does not match "
            f"state dimension {state_dim}."
        )
    drift_ito, covariance_ito = _resolved_coefficients(
        drift_field,
        diffusion=diffusion_field,
        covariance=covariance_field,
        interpretation=interpretation_value,
        var=var,
        metric=metric,
        state_dim=int(state_dim),
    )
    use_factor_hvp = (
        diffusion_field is not None
        and metric is None
        and contraction in ("auto", "factor_hvp")
    )
    if use_factor_hvp:
        factor_domain, factor_fields = _join_fields(
            observable_field,
            drift_ito,
            diffusion_field,
        )
        observable_factor, drift_factor, diffusion_factor = factor_fields
        factor_deps = _union_deps(
            factor_domain,
            observable_factor,
            drift_factor,
            diffusion_factor,
        )
        if var in factor_deps:
            return DomainFunction(
                domain=factor_domain,
                deps=factor_deps,
                func=_FactorHVPKolmogorovCallable(
                    observable=observable_factor,
                    drift=drift_factor,
                    diffusion=diffusion_factor,
                    observable_positions=_positions(factor_deps, observable_factor),
                    drift_positions=_positions(factor_deps, drift_factor),
                    diffusion_positions=_positions(factor_deps, diffusion_factor),
                    state_position=factor_deps.index(var),
                    observable_state_position=(
                        observable_factor.deps.index(var)
                        if var in observable_factor.deps
                        else -1
                    ),
                    state_dim=int(state_dim),
                ),
                metadata=_strip_derivative_hook_metadata(observable_factor.metadata),
            )
    fields = [observable_field, drift_ito]
    if covariance_ito is not None:
        fields.append(covariance_ito)
    domain, promoted = _join_fields(*fields)
    observable_promoted = promoted[0]
    drift_promoted = promoted[1]
    covariance_promoted = promoted[2] if covariance_ito is not None else None
    observable_gradient = grad(observable_promoted, var=var)
    observable_hessian = (
        (
            hessian(observable_promoted, var=var)
            if metric is None
            else riemannian_covariant_hessian(observable_promoted, metric, var=var)
        )
        if covariance_promoted is not None
        else None
    )
    used_fields = [observable_gradient, drift_promoted]
    if observable_hessian is not None and covariance_promoted is not None:
        used_fields.extend((observable_hessian, covariance_promoted))
    deps = _union_deps(domain, *used_fields)
    return DomainFunction(
        domain=domain,
        deps=deps,
        func=_KolmogorovCallable(
            observable_gradient=observable_gradient,
            observable_hessian=observable_hessian,
            drift=drift_promoted,
            covariance=covariance_promoted,
            gradient_positions=_positions(deps, observable_gradient),
            hessian_positions=(
                () if observable_hessian is None else _positions(deps, observable_hessian)
            ),
            drift_positions=_positions(deps, drift_promoted),
            covariance_positions=(
                ()
                if covariance_promoted is None
                else _positions(deps, covariance_promoted)
            ),
            state_dim=int(state_dim),
        ),
        metadata=_strip_derivative_hook_metadata(observable_promoted.metadata),
    )


def _density_product(
    density: DomainFunction,
    coefficient: DomainFunction,
    /,
    *,
    state_dim: int,
    tensor: bool,
) -> DomainFunction:
    domain, (density_promoted, coefficient_promoted) = _join_fields(density, coefficient)
    deps = _union_deps(domain, density_promoted, coefficient_promoted)
    return DomainFunction(
        domain=domain,
        deps=deps,
        func=_DensityCoefficientProductCallable(
            density=density_promoted,
            coefficient=coefficient_promoted,
            density_positions=_positions(deps, density_promoted),
            coefficient_positions=_positions(deps, coefficient_promoted),
            state_dim=int(state_dim),
            tensor=bool(tensor),
        ),
        metadata=_strip_derivative_hook_metadata(density_promoted.metadata),
    )


def probability_current(
    density: DomainFunction,
    drift: DomainFunction,
    /,
    *,
    diffusion: DomainFunction | None = None,
    covariance: DomainFunction | None = None,
    metric: RiemannianMetric | None = None,
    interpretation: StochasticInterpretation = "ito",
    var: str = "x",
) -> DomainFunction:
    r"""Return the Fokker--Planck probability current.

    In Euclidean coordinates this is
    ``J_i = b_i p - 1/2 ∂_j(a_ij p)`` and the forward operator is ``-div(J)``.
    With ``metric`` the products and divergence are relative to Riemannian volume.
    """
    density_field = _require_function(density, "density")
    drift_field = _require_function(drift, "drift")
    diffusion_field = (
        None if diffusion is None else _require_function(diffusion, "diffusion")
    )
    covariance_field = (
        None if covariance is None else _require_function(covariance, "covariance")
    )
    interpretation_value = _require_interpretation(interpretation)
    factor, state_dim = _factor_and_dim(density_field, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError("var must name a geometry state variable.")
    if metric is not None and metric.chart.dimension != state_dim:
        raise ValueError(
            f"Metric chart dimension {metric.chart.dimension} does not match "
            f"state dimension {state_dim}."
        )
    drift_ito, covariance_ito = _resolved_coefficients(
        drift_field,
        diffusion=diffusion_field,
        covariance=covariance_field,
        interpretation=interpretation_value,
        var=var,
        metric=metric,
        state_dim=int(state_dim),
    )
    current = _density_product(
        density_field,
        drift_ito,
        state_dim=int(state_dim),
        tensor=False,
    )
    if covariance_ito is None:
        return current
    covariance_times_density = _density_product(
        density_field,
        covariance_ito,
        state_dim=int(state_dim),
        tensor=True,
    )
    diffusive_current = (
        div_tensor(covariance_times_density, var=var)
        if metric is None
        else riemannian_div_tensor(covariance_times_density, metric, var=var)
    )
    return current - 0.5 * diffusive_current


def fokker_planck_operator(
    density: DomainFunction,
    drift: DomainFunction,
    /,
    *,
    diffusion: DomainFunction | None = None,
    covariance: DomainFunction | None = None,
    metric: RiemannianMetric | None = None,
    interpretation: StochasticInterpretation = "ito",
    var: str = "x",
) -> DomainFunction:
    r"""Apply the forward Kolmogorov/Fokker--Planck operator to a density.

    Without ``metric`` this is the coordinate-density operator
    ``-∂_i(b^i p) + 1/2 ∂_i∂_j(a^ij p)``. With ``metric`` it is the
    Riemannian-volume-density operator
    ``-∇_i(b^i p) + 1/2 ∇_i∇_j(a^ij p)``. Coordinate Itô drift is converted to
    the equivalent covariant vector drift. The result does not add a time
    derivative or impose positivity, normalization, initial, or boundary data.
    """
    current = probability_current(
        density,
        drift,
        diffusion=diffusion,
        covariance=covariance,
        metric=metric,
        interpretation=interpretation,
        var=var,
    )
    return (
        -div(current, var=var)
        if metric is None
        else -riemannian_div(current, metric, var=var)
    )


__all__ = [
    "GeneratorContraction",
    "StochasticInterpretation",
    "diffusion_covariance",
    "fokker_planck_operator",
    "kolmogorov_generator",
    "probability_current",
    "stratonovich_to_ito_drift",
]
