#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
import quadax
from jaxtyping import Array, ArrayLike, Key

from ..._doc import DOC_KEY0
from ..._frozendict import frozendict
from ..._strict import StrictModule
from ...domain._base import _AbstractGeometry
from ...domain._components import (
    Boundary,
    DomainComponent,
    Fixed,
    FixedEnd,
    FixedStart,
    Interior,
)
from ...domain._domain import RelabeledDomain
from ...domain._function import DomainFunction
from ...domain._scalar import _AbstractScalarDomain, ScalarInterval
from ...domain._structure import PointsBatch, ProductStructure
from ...domain.geometry1d._primitives import Interval1d
from ._batch_ops import _sum_over, _weight_product, _where_product


AdaptiveQuadratureMethod = Literal[
    "gauss_kronrod",
    "clenshaw_curtis",
    "tanh_sinh",
]

_DEFAULT_ORDERS: dict[AdaptiveQuadratureMethod, int] = {
    "gauss_kronrod": 21,
    "clenshaw_curtis": 32,
    "tanh_sinh": 61,
}
_ALLOWED_ORDERS: dict[AdaptiveQuadratureMethod, frozenset[int]] = {
    "gauss_kronrod": frozenset((15, 21, 31, 41, 51, 61)),
    "clenshaw_curtis": frozenset((8, 16, 32, 64, 128, 256)),
    "tanh_sinh": frozenset((41, 61, 81, 101)),
}


class AdaptiveQuadratureConfig(StrictModule):
    """Static controls for one-dimensional globally adaptive quadrature."""

    method: AdaptiveQuadratureMethod = eqx.field(static=True)
    absolute_tolerance: float | None = eqx.field(static=True)
    relative_tolerance: float | None = eqx.field(static=True)
    max_intervals: int = eqx.field(static=True)
    order: int = eqx.field(static=True)
    norm: float | int | Callable[[Array], Array] = eqx.field(static=True)
    breakpoints: tuple[float, ...] = eqx.field(static=True)
    collect_subintervals: bool = eqx.field(static=True)
    throw: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        method: AdaptiveQuadratureMethod = "gauss_kronrod",
        absolute_tolerance: float | None = None,
        relative_tolerance: float | None = None,
        max_intervals: int = 50,
        order: int | None = None,
        norm: float | int | Callable[[Array], Array] = math.inf,
        breakpoints: Sequence[float] = (),
        collect_subintervals: bool = False,
        throw: bool = True,
    ):
        method_ = str(method)
        if method_ not in _DEFAULT_ORDERS:
            raise ValueError(
                "method must be 'gauss_kronrod', 'clenshaw_curtis', or 'tanh_sinh'."
            )
        method_typed: AdaptiveQuadratureMethod = method_  # type: ignore[assignment]

        order_ = _DEFAULT_ORDERS[method_typed] if order is None else int(order)
        if order_ not in _ALLOWED_ORDERS[method_typed]:
            allowed = tuple(sorted(_ALLOWED_ORDERS[method_typed]))
            raise ValueError(f"order for {method_!r} must be one of {allowed}.")

        max_intervals_ = int(max_intervals)
        if max_intervals_ <= 0:
            raise ValueError("max_intervals must be positive.")

        absolute_tolerance_ = _validate_tolerance(
            absolute_tolerance,
            "absolute_tolerance",
        )
        relative_tolerance_ = _validate_tolerance(
            relative_tolerance,
            "relative_tolerance",
        )

        breakpoints_ = tuple(float(point) for point in breakpoints)
        if any(not math.isfinite(point) for point in breakpoints_):
            raise ValueError("breakpoints must be finite.")
        if any(
            right <= left
            for left, right in zip(breakpoints_[:-1], breakpoints_[1:], strict=True)
        ):
            raise ValueError("breakpoints must be strictly increasing and unique.")
        if max_intervals_ < len(breakpoints_) + 1:
            raise ValueError(
                "max_intervals must be at least the number of initial intervals."
            )

        if callable(norm):
            norm_ = norm
        else:
            norm_ = float(norm)
            if norm_ <= 0.0 or math.isnan(norm_):
                raise ValueError("norm must be positive or callable.")

        self.method = method_typed
        self.absolute_tolerance = absolute_tolerance_
        self.relative_tolerance = relative_tolerance_
        self.max_intervals = max_intervals_
        self.order = order_
        self.norm = norm_
        self.breakpoints = breakpoints_
        self.collect_subintervals = bool(collect_subintervals)
        self.throw = bool(throw)


def _validate_tolerance(value: float | None, name: str, /) -> float | None:
    if value is None:
        return None
    value_ = float(value)
    if not math.isfinite(value_) or value_ < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return value_


class AdaptiveSubintervals(StrictModule):
    """Padded accepted-subinterval diagnostics from Quadax."""

    count: Array
    lower_bounds: Array
    upper_bounds: Array
    integral_estimates: Array
    estimated_errors: Array


class AdaptiveIntegralResult(StrictModule):
    """Adaptive integral value and JAX-compatible convergence diagnostics."""

    value: cx.Field
    estimated_error: Array
    num_evaluations: Array
    status: Array
    subintervals: AdaptiveSubintervals | None
    method: AdaptiveQuadratureMethod = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        """Whether Quadax terminated without a numerical failure status."""
        return self.status == 0


def _unwrap_factor(factor: Any, /) -> Any:
    return factor.base if isinstance(factor, RelabeledDomain) else factor


def _fixed_field(factor: Any, component: Any, /) -> cx.Field:
    factor = _unwrap_factor(factor)
    if isinstance(factor, _AbstractScalarDomain):
        if isinstance(component, FixedStart):
            value = factor.fixed("start")
        elif isinstance(component, FixedEnd):
            value = factor.fixed("end")
        elif isinstance(component, Fixed):
            value = component.value
        else:
            raise TypeError(
                "Non-integrated scalar factors must use Fixed, FixedStart, or FixedEnd."
            )
        return cx.Field(jnp.asarray(value, dtype=float).reshape(()), dims=())

    if isinstance(factor, _AbstractGeometry) and isinstance(component, Fixed):
        value = jnp.asarray(component.value, dtype=float).reshape((factor.var_dim,))
        return cx.Field(value, dims=(None,))

    raise TypeError(
        "Non-integrated factors must be fixed scalar domains or fixed geometries."
    )


class _DomainAdaptiveIntegrand(StrictModule):
    integrand: DomainFunction
    component: DomainComponent
    fixed_points: frozendict[str, cx.Field]
    structure: ProductStructure
    key: Key[Array, ""]
    kwargs: frozendict[str, Any]
    variable: str = eqx.field(static=True)
    axis: str = eqx.field(static=True)
    geometry_variable: bool = eqx.field(static=True)
    cache_token: tuple[int, int, int, str, str, bool] = eqx.field(static=True)

    def __hash__(self) -> int:
        return hash((type(self), self.cache_token))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _DomainAdaptiveIntegrand) and (
            self.cache_token == other.cache_token
        )

    def __call__(self, coordinate: Array, /) -> Array:
        coordinate_ = jnp.asarray(coordinate, dtype=float)
        if self.geometry_variable:
            variable_field = cx.Field(
                coordinate_.reshape((1, 1)),
                dims=(self.axis, None),
            )
        else:
            variable_field = cx.Field(coordinate_.reshape((1,)), dims=(self.axis,))

        points = dict(self.fixed_points.items())
        points[self.variable] = variable_field
        ordered_points = frozendict(
            {label: points[label] for label in self.component.domain.labels}
        )
        batch = PointsBatch(ordered_points, self.structure)

        value = self.integrand(batch, key=self.key, **self.kwargs)
        if not isinstance(value, cx.Field):
            raise TypeError("integrand must evaluate to a coordax.Field.")
        mask = _where_product(
            self.component,
            batch,
            key=self.key,
            **self.kwargs,
        )
        weight = _weight_product(
            self.component,
            batch,
            key=self.key,
            **self.kwargs,
        )
        value = value * mask * weight

        unexpected_axes = tuple(name for name in value.named_dims if name != self.axis)
        if unexpected_axes:
            raise ValueError(
                "Adaptive integration produced unexpected named output axes "
                f"{unexpected_axes!r}."
            )
        if self.axis in value.named_dims:
            value = _sum_over(value, self.axis)
        return jnp.asarray(value.data)


def _resolve_adaptive_domain(
    component: DomainComponent,
    variable: str | None,
    /,
) -> tuple[str, ScalarInterval | Interval1d, ProductStructure, frozendict[str, cx.Field]]:
    if not isinstance(component, DomainComponent):
        raise TypeError("component must be a DomainComponent.")

    labels = component.domain.labels
    if variable is None:
        free_labels = tuple(
            label
            for label in labels
            if isinstance(component.spec.component_for(label), Interior)
        )
        if len(free_labels) != 1:
            raise ValueError(
                "Adaptive integration requires exactly one interior label; "
                "specify all other labels as fixed."
            )
        variable_ = free_labels[0]
    else:
        variable_ = str(variable)
        if variable_ not in labels:
            raise ValueError(
                f"Unknown integration variable {variable_!r}; expected one of {labels!r}."
            )

    if not isinstance(component.spec.component_for(variable_), Interior):
        raise ValueError("The adaptive integration variable must select Interior().")

    factor = _unwrap_factor(component.domain.factor(variable_))
    if not isinstance(factor, (ScalarInterval, Interval1d)):
        raise TypeError(
            "Adaptive integration currently supports ScalarInterval and Interval1d "
            "integration factors."
        )

    fixed_points: dict[str, cx.Field] = {}
    fixed_labels: set[str] = set()
    for label in labels:
        if label == variable_:
            continue
        selector = component.spec.component_for(label)
        if isinstance(selector, (Interior, Boundary)):
            raise ValueError(
                "Adaptive integration supports one free label; all other labels "
                "must be fixed."
            )
        fixed_points[label] = _fixed_field(
            component.domain.factor(label),
            selector,
        )
        fixed_labels.add(label)

    structure = ProductStructure(((variable_,),)).canonicalize(
        labels,
        fixed_labels=frozenset(fixed_labels),
    )
    return variable_, factor, structure, frozendict(fixed_points)


def adaptive_integral(
    integrand: DomainFunction | ArrayLike,
    /,
    *,
    component: DomainComponent,
    variable: str | None = None,
    quadrature: AdaptiveQuadratureConfig | None = None,
    key: Key[Array, ""] = DOC_KEY0,
    **kwargs: Any,
) -> AdaptiveIntegralResult:
    """Adaptively integrate over one interval label of a domain component.

    Exactly one component label may select ``Interior``. Other product-domain
    labels must be fixed, and therefore contribute unit-mass Dirac factors. Domain
    filters and ``weight_all`` are evaluated at every adaptive node.
    """
    config = AdaptiveQuadratureConfig() if quadrature is None else quadrature
    if not isinstance(config, AdaptiveQuadratureConfig):
        raise TypeError("quadrature must be an AdaptiveQuadratureConfig or None.")

    variable_, factor, structure, fixed_points = _resolve_adaptive_domain(
        component,
        variable,
    )
    function = (
        integrand
        if isinstance(integrand, DomainFunction)
        else DomainFunction(domain=component.domain, deps=(), func=integrand)
    )
    missing_labels = tuple(
        label for label in function.domain.labels if label not in component.domain.labels
    )
    if missing_labels:
        raise ValueError(
            f"Integrand domain labels {missing_labels!r} are absent from the component."
        )

    axis = structure.axis_for(variable_)
    if axis is None:
        raise RuntimeError("Adaptive integration variable has no sampling axis.")
    callback_kwargs = frozendict(kwargs)
    callback = _DomainAdaptiveIntegrand(
        integrand=function,
        component=component,
        fixed_points=fixed_points,
        structure=structure,
        key=key,
        kwargs=callback_kwargs,
        variable=variable_,
        axis=axis,
        geometry_variable=isinstance(factor, Interval1d),
        cache_token=(
            id(function),
            id(component),
            0 if not kwargs else id(callback_kwargs),
            variable_,
            axis,
            isinstance(factor, Interval1d),
        ),
    )

    interval = jnp.asarray(
        (factor.start, *config.breakpoints, factor.end),
        dtype=float,
    )
    interval = eqx.error_if(
        interval,
        jnp.any(jnp.diff(interval) <= 0.0),
        "Adaptive quadrature breakpoints must lie strictly inside the interval.",
    )
    backend = {
        "gauss_kronrod": quadax.quadgk,
        "clenshaw_curtis": quadax.quadcc,
        "tanh_sinh": quadax.quadts,
    }[config.method]
    value, info = backend(
        callback,
        interval,
        full_output=config.collect_subintervals,
        epsabs=config.absolute_tolerance,
        epsrel=config.relative_tolerance,
        max_ninter=config.max_intervals,
        order=config.order,
        norm=config.norm,
    )

    value_array = jnp.asarray(value)
    status = jnp.asarray(info.status, dtype=jnp.int32)
    if config.throw:
        value_array = eqx.error_if(
            value_array,
            status != 0,
            "Adaptive quadrature failed to meet its numerical contract.",
        )

    subintervals = None
    if config.collect_subintervals:
        backend_info: Mapping[str, Any] = info.info
        subintervals = AdaptiveSubintervals(
            count=jnp.asarray(backend_info["ninter"], dtype=jnp.int32),
            lower_bounds=jnp.asarray(backend_info["a_arr"]),
            upper_bounds=jnp.asarray(backend_info["b_arr"]),
            integral_estimates=jnp.asarray(backend_info["r_arr"]),
            estimated_errors=jnp.asarray(backend_info["e_arr"]),
        )

    return AdaptiveIntegralResult(
        value=cx.Field(value_array, dims=(None,) * value_array.ndim),
        estimated_error=jnp.asarray(info.err),
        num_evaluations=jnp.asarray(info.neval, dtype=jnp.int32),
        status=status,
        subintervals=subintervals,
        method=config.method,
    )


__all__ = [
    "AdaptiveIntegralResult",
    "AdaptiveQuadratureConfig",
    "AdaptiveQuadratureMethod",
    "AdaptiveSubintervals",
    "adaptive_integral",
]
