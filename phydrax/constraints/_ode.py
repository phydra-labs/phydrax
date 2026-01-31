#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

import interpax
import jax.numpy as jnp
from jaxtyping import ArrayLike

from ..domain._components import FixedStart
from ..domain._function import DomainFunction
from ..domain._structure import NumPoints, ProductStructure
from ..domain._time import TimeInterval
from ..operators.differential._domain_ops import dt_n as _dt_n
from ._functional import FunctionalConstraint
from ._pointset import PointSetConstraint


def _as_domain_function(
    value: DomainFunction | ArrayLike | Callable | None,
    domain,
    *,
    deps: tuple[str, ...],
) -> DomainFunction:
    if value is None:
        return DomainFunction(domain=domain, deps=(), func=0.0, metadata={})
    if isinstance(value, DomainFunction):
        return value
    if callable(value):
        return DomainFunction(domain=domain, deps=deps, func=value, metadata={})
    return DomainFunction(domain=domain, deps=(), func=value, metadata={})


def _coerce_operator_output(out: Any, base: DomainFunction, /) -> DomainFunction:
    if isinstance(out, DomainFunction):
        return out
    if callable(out):
        return DomainFunction(domain=base.domain, deps=base.deps, func=out, metadata={})
    return DomainFunction(domain=base.domain, deps=(), func=out, metadata={})


def ContinuousODEConstraint(
    constraint_var: str,
    domain: TimeInterval,
    operator: Callable[[DomainFunction], DomainFunction],
    /,
    *,
    num_points: NumPoints | tuple[Any, ...],
    structure: ProductStructure | None = None,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    reduction: Literal["mean", "integral"] = "mean",
) -> FunctionalConstraint:
    r"""Continuous ODE residual constraint on a `TimeInterval`.

    Given an ODE residual operator $\mathcal{R}$ (provided by `operator`), this
    enforces $\mathcal{R}(u)(t)=0$ in a sampled/weak sense by minimizing

    $$
    \ell = w\,\frac{1}{|[t_0,t_1]|}\int_{t_0}^{t_1}\|\mathcal{R}(u)(t)\|_F^2\,dt
    $$

    (or the unnormalized integral when `reduction="integral"`).
    """
    if structure is None:
        structure = ProductStructure(((domain.label,),))
    if len(structure.blocks) == 0:
        raise ValueError(
            "ContinuousODEConstraint requires a non-empty sampling structure."
        )

    def op(u: DomainFunction, /) -> DomainFunction:
        return _coerce_operator_output(operator(u), u)

    return FunctionalConstraint.from_operator(
        component=domain.component(),
        operator=op,
        constraint_vars=constraint_var,
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
        reduction=reduction,
    )


def DiscreteODEConstraint(
    constraint_var: str,
    domain: TimeInterval,
    operator: Callable[[DomainFunction], DomainFunction],
    /,
    *,
    times: ArrayLike,
    weight: ArrayLike = 1.0,
    label: str | None = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> PointSetConstraint:
    r"""Discrete ODE residual constraint evaluated at explicit times.

    Evaluates the ODE residual $\mathcal{R}(u)(t_i)$ at the provided times `times`
    and applies a mean/sum reduction of $\|\mathcal{R}(u)(t_i)\|_F^2$.
    """
    pts = {domain.label: times}

    def residual(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
        u = functions[constraint_var]
        return _coerce_operator_output(operator(u), u)

    return PointSetConstraint.from_points(
        component=domain.component(),
        points=pts,
        residual=residual,
        weight=weight,
        label=label,
        reduction=reduction,
    )


def InitialODEConstraint(
    constraint_var: str,
    domain: TimeInterval,
    /,
    *,
    func: DomainFunction | ArrayLike | Callable | None = None,
    time_derivative_order: int = 0,
    time_derivative_backend: Literal["ad", "jet"] = "ad",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    structure: ProductStructure | None = None,
    sampler: str = "latin_hypercube",
    reduction: Literal["mean", "integral"] = "mean",
) -> FunctionalConstraint:
    r"""Initial-condition constraint on the slice $t=t_0$.

    Enforces

    $$
    \left.\frac{\partial^n u}{\partial t^n}\right|_{t=t_0} = g,
    $$

    where `n = time_derivative_order` and `g` is given by `func` (defaults to $0$).
    """
    component = domain.component({domain.label: FixedStart()})
    structure = ProductStructure(()) if structure is None else structure
    order = int(time_derivative_order)
    target = _as_domain_function(func, domain, deps=(domain.label,))

    def operator(u: DomainFunction, /) -> DomainFunction:
        return (
            _dt_n(
                u,
                var=domain.label,
                order=order,
                backend=time_derivative_backend,
            )
            - target
        )

    num_points = () if len(structure.blocks) == 0 else 1
    return FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=constraint_var,
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        reduction=reduction,
    )


def DiscreteTimeDataConstraint(
    constraint_var: str,
    domain: TimeInterval,
    /,
    *,
    times: ArrayLike,
    values: ArrayLike,
    weight: ArrayLike = 1.0,
    label: str | None = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> PointSetConstraint:
    r"""Discrete data-fit constraint $u(t_i)\approx y_i$ with smooth time interpolation.

    Constructs a cubic Hermite spline target $g(t)$ through the samples `(times, values)`
    and enforces $u(t_i)\approx g(t_i)$ at the provided times.
    """
    pts = {domain.label: times}
    t = jnp.asarray(times, dtype=float).reshape((-1,))
    y = jnp.asarray(values, dtype=float)
    if y.ndim == 0:
        raise ValueError("values must have shape (T, ...) matching times.")
    if y.shape[0] != t.shape[0]:
        raise ValueError("values must have shape (T, ...) matching times.")

    slopes = interpax.approx_df(t, y, axis=0)
    spline = interpax.CubicHermiteSpline(t, y, slopes, axis=0, check=False)

    def target_fn(t_eval):
        return spline(t_eval)

    target = DomainFunction(
        domain=domain, deps=(domain.label,), func=target_fn, metadata={}
    )

    def residual(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
        return functions[constraint_var] - target

    return PointSetConstraint.from_points(
        component=domain.component(),
        points=pts,
        residual=residual,
        weight=weight,
        label=label,
        reduction=reduction,
    )
