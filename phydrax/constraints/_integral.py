#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._callable import _ensure_special_kwonly_args
from ..domain._components import (
    Boundary,
    DomainComponent,
    DomainComponentUnion,
    FixedStart,
)
from ..domain._function import DomainFunction
from ..domain._structure import NumPoints, ProductStructure
from ..operators.differential import cauchy_stress
from ..operators.linalg import einsum
from ._functional_integral import IntegralEqualityConstraint


def _default_structure(
    component: DomainComponent | DomainComponentUnion, /
) -> ProductStructure:
    labels = component.domain.labels
    return ProductStructure((labels,)).canonicalize(component.domain.labels)


def _as_domain_function(
    value: DomainFunction | ArrayLike,
    domain,
    *,
    name: str,
) -> DomainFunction:
    if isinstance(value, DomainFunction):
        return value
    if callable(value):
        return DomainFunction(domain=domain, deps=domain.labels, func=value, metadata={})
    return DomainFunction(domain=domain, deps=(), func=value, metadata={})


def _as_array_target(
    value: DomainFunction | ArrayLike | None, *, name: str
) -> Array | None:
    if value is None:
        return None
    return jnp.asarray(value, dtype=float)


def _normal(
    component: DomainComponent | DomainComponentUnion, *, var: str
) -> DomainFunction:
    if isinstance(component, DomainComponentUnion):
        raise TypeError("Boundary normals require a single DomainComponent, not a union.")
    return component.normal(var=var)


def _dot(a: DomainFunction, b: DomainFunction) -> DomainFunction:
    return einsum("...i,...i->...", a, b)


def _matvec(a: DomainFunction, b: DomainFunction) -> DomainFunction:
    return einsum("...ij,...j->...i", a, b)


def _cross(a: DomainFunction, b: DomainFunction) -> DomainFunction:
    return a._binary_op(b, lambda x, y: jnp.cross(x, y))


def _ensure_operator(
    operator: Callable[..., DomainFunction] | DomainFunction,
) -> Callable[..., DomainFunction]:
    if isinstance(operator, DomainFunction):
        return lambda *args: operator

    op = _ensure_special_kwonly_args(operator)

    def _operator(*funcs: DomainFunction) -> DomainFunction:
        out = op(*funcs)
        if isinstance(out, DomainFunction):
            return out
        base = funcs[0]
        if callable(out):
            return DomainFunction(
                domain=base.domain, deps=base.deps, func=out, metadata={}
            )
        return DomainFunction(domain=base.domain, deps=(), func=out, metadata={})

    return _operator


def ContinuousIntegralInteriorConstraint(
    constraint_vars: str | Sequence[str],
    domain,
    operator: Callable[..., DomainFunction] | DomainFunction,
    /,
    *,
    num_points: NumPoints | tuple[Any, ...],
    structure: ProductStructure | None = None,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    equal_to: ArrayLike | None = None,
    where: Mapping[str, Callable] | None = None,
    where_all: DomainFunction | None = None,
) -> IntegralEqualityConstraint:
    r"""Integral equality constraint over an interior component.

    Enforces an interior integral condition of the form

    $$
    \int_{\Omega} f(z)\,d\mu(z) = c,
    $$

    where the integrand $f$ is produced by `operator` applied to the named
    `constraint_vars`.
    """
    component = domain.component(where=where, where_all=where_all)
    structure = _default_structure(component) if structure is None else structure
    op = _ensure_operator(operator)

    return IntegralEqualityConstraint.from_operator(
        component=component,
        operator=op,
        constraint_vars=constraint_vars,
        equal_to=0.0 if equal_to is None else equal_to,
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
    )


def ContinuousIntegralBoundaryConstraint(
    constraint_vars: str | Sequence[str],
    domain,
    operator: Callable[..., DomainFunction] | DomainFunction,
    /,
    *,
    num_points: NumPoints | tuple[Any, ...],
    structure: ProductStructure | None = None,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    equal_to: ArrayLike | None = None,
    where: Mapping[str, Callable] | None = None,
    where_all: DomainFunction | None = None,
) -> IntegralEqualityConstraint:
    r"""Integral equality constraint over a boundary component.

    Enforces a boundary integral condition

    $$
    \int_{\partial\Omega} f(x,n(x))\,dS = c,
    $$

    where `operator` receives the boundary normal field $n(x)$ as an additional
    argument.
    """
    component = domain.component(
        {domain.label: Boundary()}, where=where, where_all=where_all
    )
    structure = _default_structure(component) if structure is None else structure
    op = _ensure_operator(operator)

    def operator_with_normals(*funcs: DomainFunction) -> DomainFunction:
        return op(*funcs, _normal(component, var="x"))

    return IntegralEqualityConstraint.from_operator(
        component=component,
        operator=operator_with_normals,
        constraint_vars=constraint_vars,
        equal_to=0.0 if equal_to is None else equal_to,
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
    )


def ContinuousIntegralInitialConstraint(
    constraint_vars: str | Sequence[str],
    domain,
    operator: Callable[..., DomainFunction] | DomainFunction,
    /,
    *,
    evolution_var: str = "t",
    num_points: NumPoints | tuple[Any, ...],
    structure: ProductStructure | None = None,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    equal_to: ArrayLike | None = None,
    where: Mapping[str, Callable] | None = None,
    where_all: DomainFunction | None = None,
) -> IntegralEqualityConstraint:
    r"""Integral equality constraint over the initial surface.

    Constructs a component slice `evolution_var = FixedStart()` (e.g. $t=t_0$) and
    enforces an integral condition on that slice.
    """
    if evolution_var not in domain.labels:
        raise KeyError(f"Label {evolution_var!r} not in domain {domain.labels}.")
    component = domain.component(
        {evolution_var: FixedStart()}, where=where, where_all=where_all
    )
    structure = _default_structure(component) if structure is None else structure
    op = _ensure_operator(operator)

    return IntegralEqualityConstraint.from_operator(
        component=component,
        operator=op,
        constraint_vars=constraint_vars,
        equal_to=0.0 if equal_to is None else equal_to,
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
    )


def EMBoundaryChargeConstraint(
    field_var: str,
    domain,
    /,
    *,
    total_free_charge: ArrayLike,
    var: str = "x",
    num_points: NumPoints | tuple[Any, ...],
    structure: ProductStructure | None = None,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    where: Mapping[str, Callable] | None = None,
    where_all: DomainFunction | None = None,
) -> IntegralEqualityConstraint:
    r"""Boundary integral constraint for total free charge (Gauss law).

    For an electric displacement-like field $D$, Gauss' law implies

    $$
    \int_{\partial\Omega} D\cdot n\,dS = Q_{\text{free}}.
    $$

    This enforces that equality with `total_free_charge`.
    """
    q = _as_array_target(total_free_charge, name="total_free_charge")
    if q is None:
        raise ValueError("total_free_charge must be array-like.")

    component = domain.component(
        {domain.label: Boundary()}, where=where, where_all=where_all
    )
    structure = _default_structure(component) if structure is None else structure
    n = _normal(component, var=var)

    def operator(u: DomainFunction, /) -> DomainFunction:
        return _dot(u, n)

    return IntegralEqualityConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=field_var,
        equal_to=q,
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
    )


def MagneticFluxZeroConstraint(
    field_var: str,
    domain,
    /,
    *,
    var: str = "x",
    num_points: NumPoints | tuple[Any, ...],
    structure: ProductStructure | None = None,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    where: Mapping[str, Callable] | None = None,
    where_all: DomainFunction | None = None,
) -> IntegralEqualityConstraint:
    r"""Boundary integral constraint enforcing zero net magnetic flux.

    For a magnetic flux density field $B$, one form of $\nabla\cdot B = 0$ is

    $$
    \int_{\partial\Omega} B\cdot n\,dS = 0.
    $$
    """
    component = domain.component(
        {domain.label: Boundary()}, where=where, where_all=where_all
    )
    structure = _default_structure(component) if structure is None else structure
    n = _normal(component, var=var)

    def operator(u: DomainFunction, /) -> DomainFunction:
        return _dot(u, n)

    return IntegralEqualityConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=field_var,
        equal_to=0.0,
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
    )


def CFDBoundaryFlowRateConstraint(
    velocity_var: str,
    domain,
    /,
    *,
    flow_rate: ArrayLike,
    var: str = "x",
    num_points: NumPoints | tuple[Any, ...],
    structure: ProductStructure | None = None,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    where: Mapping[str, Callable] | None = None,
    where_all: DomainFunction | None = None,
) -> IntegralEqualityConstraint:
    r"""Boundary integral constraint for volumetric flow rate.

    Enforces

    $$
    \int_{\partial\Omega} u\cdot n\,dS = \dot{V},
    $$

    where $\dot{V}$ is the prescribed flow rate.
    """
    target = _as_array_target(flow_rate, name="flow_rate")
    if target is None:
        raise ValueError("flow_rate must be array-like.")

    component = domain.component(
        {domain.label: Boundary()}, where=where, where_all=where_all
    )
    structure = _default_structure(component) if structure is None else structure
    n = _normal(component, var=var)

    def operator(u: DomainFunction, /) -> DomainFunction:
        return _dot(u, n)

    return IntegralEqualityConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=velocity_var,
        equal_to=target,
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
    )


def CFDKineticEnergyFluxBoundaryConstraint(
    velocity_var: str,
    domain,
    /,
    *,
    target_total_power: ArrayLike,
    var: str = "x",
    num_points: NumPoints | tuple[Any, ...],
    structure: ProductStructure | None = None,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    where: Mapping[str, Callable] | None = None,
    where_all: DomainFunction | None = None,
) -> IntegralEqualityConstraint:
    r"""Boundary integral constraint for kinetic energy flux (power).

    Uses the flux density $\tfrac12\|u\|_2^2(u\cdot n)$ and enforces

    $$
    \int_{\partial\Omega} \tfrac12\|u\|_2^2(u\cdot n)\,dS = P.
    $$
    """
    target = _as_array_target(target_total_power, name="target_total_power")
    if target is None:
        raise ValueError("target_total_power must be array-like.")

    component = domain.component(
        {domain.label: Boundary()}, where=where, where_all=where_all
    )
    structure = _default_structure(component) if structure is None else structure
    n = _normal(component, var=var)

    def operator(u: DomainFunction, /) -> DomainFunction:
        return 0.5 * _dot(u, u) * _dot(u, n)

    return IntegralEqualityConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=velocity_var,
        equal_to=target,
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
    )


def SolidTotalReactionBoundaryConstraint(
    displacement_var: str,
    domain,
    /,
    *,
    lambda_: ArrayLike,
    mu: ArrayLike,
    target_reaction: ArrayLike,
    var: str = "x",
    num_points: NumPoints | tuple[Any, ...],
    structure: ProductStructure | None = None,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    where: Mapping[str, Callable] | None = None,
    where_all: DomainFunction | None = None,
) -> IntegralEqualityConstraint:
    r"""Boundary integral constraint for total reaction force.

    For Cauchy stress $\sigma(u)$ and outward normal $n$, the traction is
    $t=\sigma n$. This enforces a net reaction force condition

    $$
    \int_{\partial\Omega} \sigma(u)\,n\,dS = F_{\text{target}}.
    $$
    """
    target = _as_array_target(target_reaction, name="target_reaction")
    if target is None:
        raise ValueError("target_reaction must be array-like.")

    component = domain.component(
        {domain.label: Boundary()}, where=where, where_all=where_all
    )
    structure = _default_structure(component) if structure is None else structure
    lambda_fn = _as_domain_function(lambda_, domain, name="lambda_")
    mu_fn = _as_domain_function(mu, domain, name="mu")
    n = _normal(component, var=var)

    def operator(u: DomainFunction, /) -> DomainFunction:
        sigma = cauchy_stress(u, lambda_=lambda_fn, mu=mu_fn, var=var)
        return _matvec(sigma, n)

    return IntegralEqualityConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=displacement_var,
        equal_to=target,
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
    )


def AveragePressureBoundaryConstraint(
    pressure_var: str,
    domain,
    /,
    *,
    mean_pressure: ArrayLike,
    num_points: NumPoints | tuple[Any, ...],
    structure: ProductStructure | None = None,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    where: Mapping[str, Callable] | None = None,
    where_all: DomainFunction | None = None,
) -> IntegralEqualityConstraint:
    r"""Boundary integral constraint for pressure.

    Enforces an integral of the form

    $$
    \int_{\partial\Omega} p\,dS = c.
    $$

    If you want to enforce an *average* pressure $\bar p$, set `mean_pressure` to
    $\bar p\,|\partial\Omega|$.
    """
    target = _as_array_target(mean_pressure, name="mean_pressure")
    if target is None:
        raise ValueError("mean_pressure must be array-like.")

    def operator(p: DomainFunction, /) -> DomainFunction:
        return p

    component = domain.component(
        {domain.label: Boundary()}, where=where, where_all=where_all
    )
    structure = _default_structure(component) if structure is None else structure

    return IntegralEqualityConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=pressure_var,
        equal_to=target,
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
    )


def EMPoyntingFluxBoundaryConstraint(
    e_var: str,
    h_var: str,
    domain,
    /,
    *,
    target_total_power: ArrayLike,
    num_points: NumPoints | tuple[Any, ...],
    structure: ProductStructure | None = None,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    where: Mapping[str, Callable] | None = None,
    where_all: DomainFunction | None = None,
) -> IntegralEqualityConstraint:
    r"""Boundary integral constraint for Poynting flux (electromagnetic power).

    The Poynting vector is $S = E\times H$. This enforces

    $$
    \int_{\partial\Omega} (E\times H)\cdot n\,dS = P.
    $$
    """
    target = _as_array_target(target_total_power, name="target_total_power")
    if target is None:
        raise ValueError("target_total_power must be array-like.")

    component = domain.component(
        {domain.label: Boundary()}, where=where, where_all=where_all
    )
    structure = _default_structure(component) if structure is None else structure
    n = _normal(component, var="x")

    def operator(E: DomainFunction, H: DomainFunction, /) -> DomainFunction:
        s = _cross(E, H)
        return _dot(s, n)

    return IntegralEqualityConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=(e_var, h_var),
        equal_to=target,
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
    )
