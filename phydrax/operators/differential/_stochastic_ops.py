#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp

from ..._strict import StrictModule
from ...domain._base import _AbstractGeometry
from ...domain._function import DomainFunction
from ...domain._scalar import _AbstractScalarDomain
from ._domain_ops import (
    _factor_and_dim,
    _strip_derivative_hook_metadata,
    div,
    div_tensor,
    grad,
    hessian,
)


StochasticInterpretation = Literal["ito", "stratonovich"]


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
    derivative: DomainFunction
    state_dim: int

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        sigma = jnp.asarray(self.diffusion.func(*args, key=key, **kwargs))
        derivative = jnp.asarray(self.derivative.func(*args, key=key, **kwargs))
        if sigma.ndim < 2 or sigma.shape[-2] != self.state_dim:
            raise ValueError(
                "diffusion must have trailing shape "
                f"({self.state_dim}, noise_dim); got {sigma.shape}."
            )
        expected = (self.state_dim, int(sigma.shape[-1]), self.state_dim)
        if derivative.ndim < 3 or derivative.shape[-3:] != expected:
            raise ValueError(
                "grad(diffusion) must have trailing shape "
                f"{expected}; got {derivative.shape}."
            )
        return 0.5 * jnp.einsum("...jk,...ikj->...i", sigma, derivative)


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
    derivative = grad(sigma_promoted, var=var)
    correction = DomainFunction(
        domain=domain,
        deps=sigma_promoted.deps,
        func=_StratonovichCorrectionCallable(sigma_promoted, derivative, int(state_dim)),
        metadata=_strip_derivative_hook_metadata(sigma_promoted.metadata),
    )
    return drift_promoted + correction


def _resolved_coefficients(
    drift: DomainFunction,
    *,
    diffusion: DomainFunction | None,
    covariance: DomainFunction | None,
    interpretation: StochasticInterpretation,
    var: str,
) -> tuple[DomainFunction, DomainFunction | None]:
    if diffusion is not None and covariance is not None:
        raise ValueError("Provide either diffusion or covariance, not both.")
    if interpretation == "stratonovich":
        if diffusion is None:
            raise ValueError(
                "Stratonovich operators require diffusion; covariance alone cannot "
                "determine the drift correction."
            )
        corrected = stratonovich_to_ito_drift(drift, diffusion, var=var)
        return corrected, diffusion_covariance(diffusion)
    if diffusion is not None:
        return drift, diffusion_covariance(diffusion)
    return drift, covariance


def kolmogorov_generator(
    observable: DomainFunction,
    drift: DomainFunction,
    /,
    *,
    diffusion: DomainFunction | None = None,
    covariance: DomainFunction | None = None,
    interpretation: StochasticInterpretation = "ito",
    var: str = "x",
) -> DomainFunction:
    r"""Apply a backward Kolmogorov generator to an observable.

    The returned field is :math:`\mathcal{L}u`, not the full evolution residual.
    Scalar, vector, and tensor observables are handled componentwise. For
    ``interpretation="stratonovich"``, the diffusion vector fields are converted to
    their equivalent Itô drift before applying the generator.
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
    factor, state_dim = _factor_and_dim(observable_field, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError("var must name a Euclidean geometry state variable.")
    drift_ito, covariance_ito = _resolved_coefficients(
        drift_field,
        diffusion=diffusion_field,
        covariance=covariance_field,
        interpretation=interpretation_value,
        var=var,
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
        hessian(observable_promoted, var=var) if covariance_promoted is not None else None
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


def fokker_planck_operator(
    density: DomainFunction,
    drift: DomainFunction,
    /,
    *,
    diffusion: DomainFunction | None = None,
    covariance: DomainFunction | None = None,
    interpretation: StochasticInterpretation = "ito",
    var: str = "x",
) -> DomainFunction:
    r"""Apply the forward Kolmogorov/Fokker--Planck operator to a density.

    This returns

    .. math::

        \mathcal{L}^*p=-\nabla\!\cdot(bp)
        +\tfrac12\partial_i\partial_j(a_{ij}p),

    including derivatives of state-dependent covariance fields. It does not add a
    time derivative or impose positivity, normalization, initial, or boundary data.
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
        raise ValueError("var must name a Euclidean geometry state variable.")
    drift_ito, covariance_ito = _resolved_coefficients(
        drift_field,
        diffusion=diffusion_field,
        covariance=covariance_field,
        interpretation=interpretation_value,
        var=var,
    )
    drift_times_density = _density_product(
        density_field,
        drift_ito,
        state_dim=int(state_dim),
        tensor=False,
    )
    out = -div(drift_times_density, var=var)
    if covariance_ito is None:
        return out
    covariance_times_density = _density_product(
        density_field,
        covariance_ito,
        state_dim=int(state_dim),
        tensor=True,
    )
    return out + 0.5 * div(
        div_tensor(covariance_times_density, var=var),
        var=var,
    )


__all__ = [
    "StochasticInterpretation",
    "diffusion_covariance",
    "fokker_planck_operator",
    "kolmogorov_generator",
    "stratonovich_to_ito_drift",
]
