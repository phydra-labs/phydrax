#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from jaxtyping import ArrayLike

from ..domain._components import DomainComponentUnion, Fixed, FixedEnd, FixedStart
from ..domain._function import DomainFunction
from ..domain._structure import NumPoints, ProductStructure
from ..operators.differential._domain_ops import dt_n
from ._functional import FunctionalConstraint


def ContinuousPointwiseInteriorConstraint(
    constraint_vars: str | Sequence[str],
    domain,
    operator,
    *,
    num_points: NumPoints | tuple[Any, ...],
    structure: ProductStructure,
    coord_separable: Mapping[str, Any] | None = None,
    dense_structure: ProductStructure | None = None,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    reduction: Literal["mean", "integral"] = "mean",
    where: Mapping[str, Any] | None = None,
    where_all: DomainFunction | None = None,
) -> FunctionalConstraint:
    r"""Pointwise residual constraint over an interior domain component.

    Let `operator` define a residual `DomainFunction` $r$ from one or more fields:

    $$
    r = \mathcal{N}(u_1,\dots,u_k),
    $$

    where $r(z)$ is evaluated at sampled interior points $z\in\Omega$.

    The resulting objective is the mean or integral of the squared residual.

    For `reduction="mean"`:

    $$
    \ell = w\,\frac{1}{\mu(\Omega)}\int_{\Omega} \|r(z)\|_F^2\,d\mu(z),
    $$

    For `reduction="integral"`:

    $$
    \ell = w\int_{\Omega} \|r(z)\|_F^2\,d\mu(z).
    $$

    **Arguments:**

    - `constraint_vars`: Name (or names) of the field functions passed into `operator`.
    - `domain`: A `phydrax.domain` object to sample from.
    - `operator`: Callable mapping one or more `DomainFunction` objects to a residual
      `DomainFunction`.
    - `num_points`: Number of interior points to sample (paired or structured; see `structure`).
    - `structure`: A `ProductStructure` describing how variables are sampled/blocked.
    - `coord_separable`: Optional coord-separable sampling spec (per label).
    - `dense_structure`: Optional dense structure used when sampling produces dense batches.
    - `sampler`: Sampling scheme (e.g. `"latin_hypercube"`).
    - `weight`: Scalar multiplier $w$ applied to this term.
    - `over`: Optional subset of labels to reduce/integrate over.
    - `reduction`: Reduction mode: `"mean"` (measure-normalized) or `"integral"` (unnormalized).
    - `where`: Optional per-label filters, treated as indicator functions.
    - `where_all`: Optional global filter, evaluated on the full point tuple.
    """
    component = domain.component(where=where, where_all=where_all)
    if isinstance(component, DomainComponentUnion):
        raise TypeError(
            "ContinuousPointwiseInteriorConstraint requires a DomainComponent, not a union."
        )
    return FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=constraint_vars,
        num_points=num_points,
        structure=structure,
        coord_separable=coord_separable,
        dense_structure=dense_structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
        reduction=reduction,
    )


def ContinuousInitialFunctionConstraint(
    constraint_vars: str | Sequence[str],
    domain,
    /,
    *,
    func: DomainFunction | ArrayLike | None = None,
    evolution_var: str = "t",
    time_derivative_order: int = 0,
    mode: Literal["reverse", "forward"] = "reverse",
    time_derivative_backend: Literal["ad", "jet"] = "ad",
    num_points: NumPoints | tuple[Any, ...],
    structure: ProductStructure,
    coord_separable: Mapping[str, Any] | None = None,
    dense_structure: ProductStructure | None = None,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    reduction: Literal["mean", "integral"] = "mean",
    where: Mapping[str, Any] | None = None,
    where_all: DomainFunction | None = None,
) -> FunctionalConstraint:
    r"""Initial-condition constraint matching time derivatives on the initial slice.

    Builds a constraint on the component slice `evolution_var = FixedStart()` and
    enforces

    $$
    \left.\frac{\partial^n u}{\partial t^n}\right|_{t=t_0} = g,
    $$

    where `n = time_derivative_order`, $t_0$ is the start of the time interval, and
    `g` is provided by `func` (defaults to $0$).

    The loss is formed by sampling points on the initial slice and reducing the
    squared residual as in `FunctionalConstraint`.

    **Arguments:**

    - `constraint_vars`: Name (or names) of the field functions passed into the residual.
    - `domain`: A `phydrax.domain` object to sample from.
    - `func`: Target value $g$ as a `DomainFunction`, callable, or array-like. Defaults to `0.0`.
    - `evolution_var`: Name of the time-like label used for the initial slice (default `"t"`).
    - `time_derivative_order`: Derivative order $n$ for $\partial^n/\partial t^n$.
    - `mode`: Differentiation mode (`"reverse"` or `"forward"`).
    - `time_derivative_backend`: Backend for time derivatives (`"ad"` or `"jet"`).
    - `num_points`: Number of initial-slice points to sample (paired or structured; see `structure`).
    - `structure`: A `ProductStructure` describing how variables are sampled/blocked.
    - `coord_separable`: Optional coord-separable sampling spec (per label).
    - `dense_structure`: Optional dense structure used when sampling produces dense batches.
    - `sampler`: Sampling scheme (e.g. `"latin_hypercube"`).
    - `weight`: Scalar multiplier applied to this term.
    - `over`: Optional subset of labels to reduce/integrate over.
    - `reduction`: `"mean"` or `"integral"`.
    - `where`: Optional per-label filters, treated as indicator functions.
    - `where_all`: Optional global filter, evaluated on the full point tuple.
    """
    component = domain.component(
        {evolution_var: FixedStart()}, where=where, where_all=where_all
    )
    if isinstance(component, DomainComponentUnion):
        raise TypeError(
            "ContinuousInitialFunctionConstraint requires a DomainComponent, not a union."
        )
    if evolution_var not in component.domain.labels:
        raise KeyError(
            f"Label {evolution_var!r} not in domain {component.domain.labels}."
        )
    if not isinstance(component.spec.component_for(evolution_var), FixedStart):
        raise ValueError(
            "ContinuousInitialFunctionConstraint requires a component with "
            f"{evolution_var!r}: FixedStart()."
        )

    order = int(time_derivative_order)
    if order < 0:
        raise ValueError("time_derivative_order must be non-negative.")

    fixed_labels = {
        lbl
        for lbl in component.domain.labels
        if isinstance(component.spec.component_for(lbl), (FixedStart, FixedEnd, Fixed))
    }
    deps = tuple(lbl for lbl in component.domain.labels if lbl not in fixed_labels)
    if func is None:
        value = 0.0
    elif isinstance(func, DomainFunction):
        value = func
    elif callable(func):
        value = DomainFunction(domain=component.domain, deps=deps, func=func, metadata={})
    else:
        value = func

    def operator(*args: DomainFunction) -> DomainFunction:
        if len(args) != 1:
            raise ValueError(
                "ContinuousInitialFunctionConstraint expects a single function."
            )
        return (
            dt_n(
                args[0],
                var=evolution_var,
                order=order,
                mode=mode,
                backend=time_derivative_backend,
            )
            - value
        )

    if coord_separable is not None:
        if coord_separable.keys() & fixed_labels:
            raise ValueError(
                "coord_separable must not include fixed labels for initial constraints."
            )

    return FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=constraint_vars,
        num_points=num_points,
        structure=structure,
        coord_separable=coord_separable,
        dense_structure=dense_structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
        reduction=reduction,
    )
