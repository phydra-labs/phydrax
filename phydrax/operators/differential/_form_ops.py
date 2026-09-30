#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, final, Literal, TYPE_CHECKING

import equinox as eqx
from jax import Array

from phydrax.domain import AbstractGeometry, DomainFunction

from ..._strict import StrictModule
from ...exterior import _algebra as algebra
from ...exterior._basis import exterior_indices
from ...exterior._form_type import FiberProduct, FormTwist, FormType
from ...metrix import AbstractSemiRiemannianMetric, CoordinateChart, LorentzianMetric
from ._domain_ops import _factor_and_dim, _resolve_var
from ._form_derivative import (
    _DerivativeOptions,
    _FormDerivativeProvider,
    _function_value,
    _gradient_function,
)


if TYPE_CHECKING:
    from ...nn._keys import EvalKey


def _positions(deps: tuple[str, ...], function: DomainFunction, /) -> tuple[int, ...]:
    return tuple(deps.index(label) for label in function.deps)


def _dependencies(
    domain_labels: tuple[str, ...], functions: tuple[DomainFunction, ...], var: str, /
) -> tuple[str, ...]:
    return tuple(
        label
        for label in domain_labels
        if label == var or any(label in function.deps for function in functions)
    )


def _evaluate(
    function: DomainFunction,
    positions: tuple[int, ...],
    args: tuple[Any, ...],
    form_type: FormType,
    /,
    *,
    key: EvalKey,
    kwargs: dict[str, Any],
) -> Array:
    values = _function_value(
        function,
        tuple(args[position] for position in positions),
        key=key,
        kwargs=kwargs,
    )
    expected = form_type.value_shape
    if values.ndim < len(expected) or values.shape[-len(expected) :] != expected:
        raise ValueError(
            f"Form coefficients require explicit trailing shape {expected}; got {values.shape}."
        )
    return values


@final
class DomainDifferentialForm(StrictModule):
    """A labeled smooth form with coefficients ``(*batch, components, *fiber)``."""

    coefficients: DomainFunction
    chart: CoordinateChart
    var: str = eqx.field(static=True)
    form_type: FormType = eqx.field(static=True)

    def __init__(
        self,
        coefficients: DomainFunction,
        /,
        *,
        chart: CoordinateChart,
        degree: int,
        twist: FormTwist = "untwisted",
        fiber_shape: tuple[int, ...] = (),
        var: str | None = None,
    ) -> None:
        if not isinstance(coefficients, DomainFunction):
            raise TypeError("coefficients must be a DomainFunction.")
        if not isinstance(chart, CoordinateChart):
            raise TypeError("chart must be a CoordinateChart.")
        form_type = FormType(
            chart.dimension, degree, twist=twist, fiber_shape=fiber_shape
        )
        variable = _resolve_var(coefficients, var)
        _, dimension = _factor_and_dim(coefficients, variable)
        if not isinstance(coefficients.domain.factor(variable), AbstractGeometry):
            raise ValueError("Domain differential forms require a geometry variable.")
        if dimension != chart.dimension:
            raise ValueError(
                f"Chart dimension {chart.dimension} does not match domain variable {variable!r} dimension {dimension}."
            )
        self.coefficients = coefficients
        self.chart = chart
        self.var = variable
        self.form_type = form_type

    @property
    def degree(self) -> int:
        return self.form_type.degree

    @property
    def twist(self) -> FormTwist:
        return self.form_type.twist

    @property
    def fiber_shape(self) -> tuple[int, ...]:
        return self.form_type.fiber_shape

    @property
    def indices(self) -> tuple[tuple[int, ...], ...]:
        return exterior_indices(self.chart.dimension, self.degree)

    @property
    def coefficient_count(self) -> int:
        return self.form_type.component_count


@final
class DomainMaxwellResiduals(StrictModule):
    """Covariant Maxwell field strength and its two form-valued residuals."""

    field_strength: DomainDifferentialForm
    homogeneous: DomainDifferentialForm
    inhomogeneous: DomainDifferentialForm

    def __init__(
        self,
        field_strength: DomainDifferentialForm,
        homogeneous: DomainDifferentialForm,
        inhomogeneous: DomainDifferentialForm,
        /,
    ) -> None:
        self.field_strength = field_strength
        self.homogeneous = homogeneous
        self.inhomogeneous = inhomogeneous


@final
class _DomainWedgeCallable(StrictModule, _FormDerivativeProvider):
    left: DomainDifferentialForm
    right: DomainDifferentialForm
    left_positions: tuple[int, ...] = eqx.field(static=True)
    right_positions: tuple[int, ...] = eqx.field(static=True)
    product: FiberProduct = eqx.field(static=True)

    def __init__(
        self,
        left: DomainDifferentialForm,
        right: DomainDifferentialForm,
        deps: tuple[str, ...],
        product: FiberProduct,
        /,
    ) -> None:
        self.left = left
        self.right = right
        self.left_positions = _positions(deps, left.coefficients)
        self.right_positions = _positions(deps, right.coefficients)
        self.product = product

    def __call__(self, *args: Any, key: EvalKey = None, **kwargs: Any) -> Array:
        left = _evaluate(
            self.left.coefficients,
            self.left_positions,
            args,
            self.left.form_type,
            key=key,
            kwargs=kwargs,
        )
        right = _evaluate(
            self.right.coefficients,
            self.right_positions,
            args,
            self.right.form_type,
            key=key,
            kwargs=kwargs,
        )
        return algebra.wedge(
            left, right, self.left.form_type, self.right.form_type, product=self.product
        )


@final
class _DomainExteriorCallable(StrictModule, _FormDerivativeProvider):
    form_type: FormType = eqx.field(static=True)
    derivative: DomainFunction
    positions: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        form: DomainDifferentialForm,
        derivative: DomainFunction,
        deps: tuple[str, ...],
        /,
    ) -> None:
        self.form_type = form.form_type
        self.derivative = derivative
        self.positions = _positions(deps, derivative)

    def __call__(self, *args: Any, key: EvalKey = None, **kwargs: Any) -> Array:
        jacobian = _function_value(
            self.derivative,
            tuple(args[position] for position in self.positions),
            key=key,
            kwargs=kwargs,
        )
        return algebra.exterior_derivative_from_jacobian(jacobian, self.form_type)


@final
class _DomainInteriorCallable(StrictModule, _FormDerivativeProvider):
    vector: DomainFunction
    form: DomainDifferentialForm
    vector_positions: tuple[int, ...] = eqx.field(static=True)
    form_positions: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        vector: DomainFunction,
        form: DomainDifferentialForm,
        deps: tuple[str, ...],
        /,
    ) -> None:
        self.vector = vector
        self.form = form
        self.vector_positions = _positions(deps, vector)
        self.form_positions = _positions(deps, form.coefficients)

    def __call__(self, *args: Any, key: EvalKey = None, **kwargs: Any) -> Array:
        vector = _function_value(
            self.vector,
            tuple(args[position] for position in self.vector_positions),
            key=key,
            kwargs=kwargs,
        )
        values = _evaluate(
            self.form.coefficients,
            self.form_positions,
            args,
            self.form.form_type,
            key=key,
            kwargs=kwargs,
        )
        return algebra.interior(vector, values, self.form.form_type)


@final
class _DomainHodgeCallable(StrictModule, _FormDerivativeProvider):
    form: DomainDifferentialForm
    metric: AbstractSemiRiemannianMetric
    positions: tuple[int, ...] = eqx.field(static=True)
    coordinate_position: int = eqx.field(static=True)

    def __init__(
        self,
        form: DomainDifferentialForm,
        metric: AbstractSemiRiemannianMetric,
        deps: tuple[str, ...],
        /,
    ) -> None:
        self.form = form
        self.metric = metric
        self.positions = _positions(deps, form.coefficients)
        self.coordinate_position = deps.index(form.var)

    def __call__(self, *args: Any, key: EvalKey = None, **kwargs: Any) -> Array:
        values = _evaluate(
            self.form.coefficients,
            self.positions,
            args,
            self.form.form_type,
            key=key,
            kwargs=kwargs,
        )
        coordinates = args[self.coordinate_position]
        return algebra.hodge_star(
            values,
            self.form.form_type,
            self.metric.inverse(coordinates),
            self.metric.volume_density(coordinates),
        )


@final
class _ScaleCallable(StrictModule, _FormDerivativeProvider):
    function: DomainFunction
    scale: float = eqx.field(static=True)

    def __init__(self, function: DomainFunction, scale: float, /) -> None:
        self.function = function
        self.scale = scale

    def __call__(self, *args: Any, key: EvalKey = None, **kwargs: Any) -> Array:
        return self.scale * _function_value(self.function, args, key=key, kwargs=kwargs)


@final
class _DomainTwistCallable(StrictModule, _FormDerivativeProvider):
    form: DomainDifferentialForm
    orientation: int = eqx.field(static=True)
    twist: FormTwist = eqx.field(static=True)

    def __init__(
        self, form: DomainDifferentialForm, orientation: int, twist: FormTwist, /
    ) -> None:
        if orientation not in (-1, 1):
            raise ValueError("orientation must be +1 or -1.")
        self.form = form
        self.orientation = orientation
        self.twist = twist

    def __call__(self, *args: Any, key: EvalKey = None, **kwargs: Any) -> Array:
        values = _evaluate(
            self.form.coefficients,
            tuple(range(len(args))),
            args,
            self.form.form_type,
            key=key,
            kwargs=kwargs,
        )
        if self.twist == "twisted":
            return algebra.to_twisted(values, self.form.form_type, self.orientation)
        return algebra.to_untwisted(values, self.form.form_type, self.orientation)


@final
class _DomainPullbackCallable(StrictModule, _FormDerivativeProvider):
    form: DomainDifferentialForm
    mapping: DomainFunction
    mapping_positions: tuple[int, ...] = eqx.field(static=True)
    form_positions: tuple[int, ...] = eqx.field(static=True)
    source_var: str = eqx.field(static=True)
    options: _DerivativeOptions = eqx.field(static=True)
    coorientation: int | None = eqx.field(static=True)

    def __init__(
        self,
        form: DomainDifferentialForm,
        mapping: DomainFunction,
        deps: tuple[str, ...],
        source_var: str,
        options: _DerivativeOptions,
        coorientation: int | None,
        /,
    ) -> None:
        self.form = form
        self.mapping = mapping
        self.mapping_positions = _positions(deps, mapping)
        self.form_positions = tuple(
            -1 if label == form.var else deps.index(label)
            for label in form.coefficients.deps
        )
        self.source_var = source_var
        self.options = options
        self.coorientation = coorientation

    def __call__(self, *args: Any, key: EvalKey = None, **kwargs: Any) -> Array:
        mapping_args = tuple(args[position] for position in self.mapping_positions)
        target = _function_value(self.mapping, mapping_args, key=key, kwargs=kwargs)
        if target.shape[-1:] != (self.form.chart.dimension,):
            raise ValueError("Pullback mapping has the wrong target dimension.")
        bound = tuple(
            target if position < 0 else args[position] for position in self.form_positions
        )
        values = _evaluate(
            self.form.coefficients,
            tuple(range(len(bound))),
            bound,
            self.form.form_type,
            key=key,
            kwargs=kwargs,
        )
        derivative = _gradient_function(self.mapping, self.source_var, self.options)
        jacobian = _function_value(derivative, mapping_args, key=key, kwargs=kwargs)
        return algebra.pullback(
            values, self.form.form_type, jacobian, coorientation=self.coorientation
        )


def _with_coefficients(
    form: DomainDifferentialForm, coefficients: DomainFunction, form_type: FormType, /
) -> DomainDifferentialForm:
    return DomainDifferentialForm(
        coefficients,
        chart=form.chart,
        degree=form_type.degree,
        twist=form_type.twist,
        fiber_shape=form_type.fiber_shape,
        var=form.var,
    )


def domain_differential_form(
    coefficients: DomainFunction,
    /,
    *,
    chart: CoordinateChart,
    degree: int,
    twist: FormTwist = "untwisted",
    fiber_shape: tuple[int, ...] = (),
    var: str | None = None,
) -> DomainDifferentialForm:
    return DomainDifferentialForm(
        coefficients,
        chart=chart,
        degree=degree,
        twist=twist,
        fiber_shape=fiber_shape,
        var=var,
    )


def _require_same_base(
    left: DomainDifferentialForm, right: DomainDifferentialForm, /
) -> None:
    if (
        left.coefficients.domain is not right.coefficients.domain
        or left.var != right.var
        or not left.chart.compatible_with(right.chart)
    ):
        raise ValueError("Domain forms must share one domain, variable, and chart.")


def domain_wedge(
    left: DomainDifferentialForm,
    right: DomainDifferentialForm,
    /,
    *,
    product: FiberProduct = "scalar",
) -> DomainDifferentialForm:
    if not isinstance(left, DomainDifferentialForm) or not isinstance(
        right, DomainDifferentialForm
    ):
        raise TypeError("domain_wedge requires two DomainDifferentialForm instances.")
    _require_same_base(left, right)
    output_type = left.form_type.wedge_type(right.form_type, product=product)
    deps = _dependencies(
        left.coefficients.domain.labels, (left.coefficients, right.coefficients), left.var
    )
    coefficients = DomainFunction(
        domain=left.coefficients.domain,
        deps=deps,
        func=_DomainWedgeCallable(left, right, deps, product),
        metadata=left.coefficients.metadata,
    )
    return _with_coefficients(left, coefficients, output_type)


def domain_exterior_derivative(
    form: DomainDifferentialForm,
    /,
    *,
    mode: Literal["reverse", "forward"] = "forward",
    backend: Literal["ad", "jet", "fd", "basis"] = "ad",
) -> DomainDifferentialForm:
    if not isinstance(form, DomainDifferentialForm):
        raise TypeError("domain_exterior_derivative requires a DomainDifferentialForm.")
    output_type = form.form_type.exterior_derivative_type()
    derivative = _gradient_function(
        form.coefficients, form.var, (mode, backend, "poly", False)
    )
    deps = _dependencies(form.coefficients.domain.labels, (derivative,), form.var)
    coefficients = DomainFunction(
        domain=form.coefficients.domain,
        deps=deps,
        func=_DomainExteriorCallable(form, derivative, deps),
        metadata=derivative.metadata,
    )
    return _with_coefficients(form, coefficients, output_type)


def domain_interior_product(
    vector: DomainFunction, form: DomainDifferentialForm, /
) -> DomainDifferentialForm:
    if not isinstance(vector, DomainFunction):
        raise TypeError("vector must be a DomainFunction.")
    if not isinstance(form, DomainDifferentialForm):
        raise TypeError("form must be a DomainDifferentialForm.")
    if vector.domain is not form.coefficients.domain:
        raise ValueError("Domain interior product requires one shared Domain instance.")
    output_type = form.form_type.interior_type()
    deps = _dependencies(vector.domain.labels, (vector, form.coefficients), form.var)
    coefficients = DomainFunction(
        domain=vector.domain,
        deps=deps,
        func=_DomainInteriorCallable(vector, form, deps),
        metadata=form.coefficients.metadata,
    )
    return _with_coefficients(form, coefficients, output_type)


def _add_domain_forms(
    left: DomainDifferentialForm, right: DomainDifferentialForm, /
) -> DomainDifferentialForm:
    _require_same_base(left, right)
    if left.form_type.form_type_id != right.form_type.form_type_id:
        raise ValueError("Only domain forms with identical form types can be added.")
    return _with_coefficients(
        left, left.coefficients + right.coefficients, left.form_type
    )


def _scale_domain_form(
    form: DomainDifferentialForm, scale: float, /
) -> DomainDifferentialForm:
    coefficients = DomainFunction(
        domain=form.coefficients.domain,
        deps=form.coefficients.deps,
        func=_ScaleCallable(form.coefficients, scale),
        metadata=form.coefficients.metadata,
    )
    return _with_coefficients(form, coefficients, form.form_type)


def domain_lie_derivative(
    vector: DomainFunction,
    form: DomainDifferentialForm,
    /,
    *,
    mode: Literal["reverse", "forward"] = "forward",
    backend: Literal["ad", "jet", "fd", "basis"] = "ad",
) -> DomainDifferentialForm:
    if not isinstance(vector, DomainFunction):
        raise TypeError("vector must be a DomainFunction.")
    if not isinstance(form, DomainDifferentialForm):
        raise TypeError("form must be a DomainDifferentialForm.")
    if form.degree == 0:
        return domain_interior_product(
            vector, domain_exterior_derivative(form, mode=mode, backend=backend)
        )
    first = domain_exterior_derivative(
        domain_interior_product(vector, form), mode=mode, backend=backend
    )
    if form.degree == form.chart.dimension:
        return first
    second = domain_interior_product(
        vector, domain_exterior_derivative(form, mode=mode, backend=backend)
    )
    return _add_domain_forms(first, second)


def _require_metric(
    form: DomainDifferentialForm, metric: AbstractSemiRiemannianMetric, /
) -> None:
    if not isinstance(form, DomainDifferentialForm):
        raise TypeError("A DomainDifferentialForm is required.")
    if not isinstance(metric, AbstractSemiRiemannianMetric):
        raise TypeError("A nondegenerate metric is required.")
    if not form.chart.compatible_with(metric.chart):
        raise ValueError("Domain form and metric charts must match.")


def domain_hodge_star(
    form: DomainDifferentialForm, metric: AbstractSemiRiemannianMetric, /
) -> DomainDifferentialForm:
    """Apply the metric-only Hodge star, flipping the orientation-line twist."""
    _require_metric(form, metric)
    output_type = form.form_type.hodge_dual()
    deps = _dependencies(form.coefficients.domain.labels, (form.coefficients,), form.var)
    coefficients = DomainFunction(
        domain=form.coefficients.domain,
        deps=deps,
        func=_DomainHodgeCallable(form, metric, deps),
        metadata=form.coefficients.metadata,
    )
    return _with_coefficients(form, coefficients, output_type)


def domain_codifferential(
    form: DomainDifferentialForm,
    metric: AbstractSemiRiemannianMetric,
    /,
    *,
    mode: Literal["reverse", "forward"] = "forward",
    backend: Literal["ad", "jet", "fd", "basis"] = "ad",
) -> DomainDifferentialForm:
    _require_metric(form, metric)
    sign = algebra.codifferential_sign(form.form_type, metric.signature.index)
    derivative = domain_exterior_derivative(
        domain_hodge_star(form, metric), mode=mode, backend=backend
    )
    result = domain_hodge_star(derivative, metric)
    return _scale_domain_form(result, sign)


def domain_hodge_laplacian(
    form: DomainDifferentialForm,
    metric: AbstractSemiRiemannianMetric,
    /,
    *,
    mode: Literal["reverse", "forward"] = "forward",
    backend: Literal["ad", "jet", "fd", "basis"] = "ad",
) -> DomainDifferentialForm:
    _require_metric(form, metric)
    if form.degree == 0:
        derivative = domain_exterior_derivative(form, mode=mode, backend=backend)
        return domain_codifferential(derivative, metric, mode=mode, backend=backend)
    codifferential = domain_codifferential(form, metric, mode=mode, backend=backend)
    first = domain_exterior_derivative(codifferential, mode=mode, backend=backend)
    if form.degree == form.chart.dimension:
        return first
    derivative = domain_exterior_derivative(form, mode=mode, backend=backend)
    second = domain_codifferential(derivative, metric, mode=mode, backend=backend)
    return _add_domain_forms(first, second)


def _convert_twist(
    form: DomainDifferentialForm, orientation: int, twist: FormTwist, /
) -> DomainDifferentialForm:
    if not isinstance(form, DomainDifferentialForm):
        raise TypeError("A DomainDifferentialForm is required.")
    if form.twist == twist:
        raise ValueError(f"Form is already {twist}.")
    coefficients = DomainFunction(
        domain=form.coefficients.domain,
        deps=form.coefficients.deps,
        func=_DomainTwistCallable(form, orientation, twist),
        metadata=form.coefficients.metadata,
    )
    return _with_coefficients(form, coefficients, form.form_type.with_twist(twist))


def domain_to_untwisted(
    form: DomainDifferentialForm, orientation: int, /
) -> DomainDifferentialForm:
    """Trivialize the orientation line with the explicit orientation ±1."""
    return _convert_twist(form, orientation, "untwisted")


def domain_to_twisted(
    form: DomainDifferentialForm, orientation: int, /
) -> DomainDifferentialForm:
    return _convert_twist(form, orientation, "twisted")


def domain_pullback_form(
    form: DomainDifferentialForm,
    mapping: DomainFunction,
    /,
    *,
    source_chart: CoordinateChart,
    source_var: str | None = None,
    coorientation: int | None = None,
    mode: Literal["reverse", "forward"] = "forward",
    backend: Literal["ad", "jet", "fd", "basis"] = "ad",
) -> DomainDifferentialForm:
    """Pull a form back along a labeled map from its source domain."""
    if not isinstance(form, DomainDifferentialForm) or not isinstance(
        mapping, DomainFunction
    ):
        raise TypeError(
            "Pullback requires a DomainDifferentialForm and DomainFunction map."
        )
    if not isinstance(source_chart, CoordinateChart):
        raise TypeError("source_chart must be a CoordinateChart.")
    var = _resolve_var(mapping, source_var)
    _, dimension = _factor_and_dim(mapping, var)
    if not isinstance(mapping.domain.factor(var), AbstractGeometry):
        raise ValueError("Domain pullbacks require a geometry variable.")
    if dimension != source_chart.dimension:
        raise ValueError(
            "Pullback source chart dimension does not match its domain variable."
        )
    if coorientation is not None and coorientation not in (-1, 1):
        raise ValueError("coorientation must be +1 or -1.")
    if (
        form.twist == "twisted"
        and dimension != form.chart.dimension
        and coorientation is None
    ):
        raise ValueError("A non-square twisted pullback requires a coorientation.")
    output_type = FormType(
        dimension, form.degree, twist=form.twist, fiber_shape=form.fiber_shape
    )
    other_deps = tuple(label for label in form.coefficients.deps if label != form.var)
    if any(label not in mapping.domain.labels for label in other_deps):
        raise ValueError(
            "Pullback source domain is missing a form coefficient dependency."
        )
    deps = tuple(
        label
        for label in mapping.domain.labels
        if label == var or label in mapping.deps or label in other_deps
    )
    coefficients = DomainFunction(
        domain=mapping.domain,
        deps=deps,
        func=_DomainPullbackCallable(
            form,
            mapping,
            deps,
            var,
            (mode, backend, "poly", False),
            coorientation,
        ),
        metadata=form.coefficients.metadata,
    )
    return DomainDifferentialForm(
        coefficients,
        chart=source_chart,
        degree=output_type.degree,
        twist=output_type.twist,
        fiber_shape=output_type.fiber_shape,
        var=var,
    )


def domain_maxwell_residuals(
    field_strength: DomainDifferentialForm,
    metric: LorentzianMetric,
    /,
    *,
    electric_current: DomainDifferentialForm | None = None,
    magnetic_current: DomainDifferentialForm | None = None,
) -> DomainMaxwellResiduals:
    r"""Compose ``dF - M`` and ``delta F + J_flat`` on four-dimensional spacetime.

    ``F`` and ``M`` are untwisted scalar-fiber forms. ``J_flat`` is the
    untwisted covector representation of the twisted current three-form.
    """
    if not isinstance(field_strength, DomainDifferentialForm):
        raise TypeError("domain_maxwell_residuals requires a DomainDifferentialForm.")
    expected = FormType(4, 2)
    if field_strength.form_type.form_type_id != expected.form_type_id:
        raise ValueError(
            "Maxwell field strength must be an untwisted scalar-fiber degree-2 form on a four-dimensional chart."
        )
    if not isinstance(metric, LorentzianMetric):
        raise TypeError("domain_maxwell_residuals requires a LorentzianMetric.")
    _require_metric(field_strength, metric)
    for current, current_type, name in (
        (electric_current, expected.codifferential_type(), "electric_current"),
        (magnetic_current, expected.exterior_derivative_type(), "magnetic_current"),
    ):
        if current is None:
            continue
        if not isinstance(current, DomainDifferentialForm):
            raise TypeError(f"{name} must be a DomainDifferentialForm.")
        _require_same_base(field_strength, current)
        if current.form_type.form_type_id != current_type.form_type_id:
            raise ValueError(
                f"{name} must have the matching untwisted scalar-fiber degree-{current_type.degree} form type."
            )
    homogeneous = domain_exterior_derivative(field_strength)
    if magnetic_current is not None:
        homogeneous = _add_domain_forms(
            homogeneous, _scale_domain_form(magnetic_current, -1.0)
        )
    inhomogeneous = domain_codifferential(field_strength, metric)
    if electric_current is not None:
        inhomogeneous = _add_domain_forms(inhomogeneous, electric_current)
    return DomainMaxwellResiduals(field_strength, homogeneous, inhomogeneous)


__all__ = [
    "DomainDifferentialForm",
    "DomainMaxwellResiduals",
    "domain_codifferential",
    "domain_differential_form",
    "domain_exterior_derivative",
    "domain_hodge_laplacian",
    "domain_hodge_star",
    "domain_interior_product",
    "domain_lie_derivative",
    "domain_maxwell_residuals",
    "domain_pullback_form",
    "domain_to_twisted",
    "domain_to_untwisted",
    "domain_wedge",
]
