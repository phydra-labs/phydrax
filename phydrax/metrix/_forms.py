"""Chart-callable smooth forms over the canonical exterior kernel."""

from __future__ import annotations

from collections.abc import Callable
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .._strict import StrictModule
from ..exterior import _algebra
from ..exterior._basis import exterior_indices
from ..exterior._form_type import FiberProduct, FormTwist, FormType
from ._chart import CoordinateChart
from ._map import DifferentiableMap
from ._metric import AbstractSemiRiemannianMetric
from ._utils import _pointwise_array


def _require_same_chart(left: CoordinateChart, right: CoordinateChart, /) -> None:
    if not left.compatible_with(right):
        raise ValueError(
            f"Differential-form charts do not match: {left.name!r} and {right.name!r}."
        )


@final
class DifferentialForm(StrictModule):
    """Coefficients have shape ``(*batch,C(n,k),*fiber)`` at every degree."""

    coefficient_function: Callable[[Array], Array]
    chart: CoordinateChart
    form_type: FormType = eqx.field(static=True)

    def __init__(
        self,
        coefficients: Callable[[Array], Array],
        /,
        *,
        chart: CoordinateChart,
        degree: int,
        twist: FormTwist = "untwisted",
        fiber_shape: tuple[int, ...] = (),
    ) -> None:
        if not callable(coefficients):
            raise TypeError("Differential-form coefficients must be callable.")
        if not isinstance(chart, CoordinateChart):
            raise TypeError("Differential-form chart must be a CoordinateChart.")
        form_type = FormType(
            chart.dimension, degree, twist=twist, fiber_shape=fiber_shape
        )
        self.coefficient_function = coefficients
        self.chart = chart
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

    def _coefficients_point(self, coordinates: Array, /) -> Array:
        values = jnp.asarray(self.coefficient_function(coordinates))
        if values.shape != self.form_type.value_shape:
            raise ValueError(
                f"Degree-{self.degree} form coefficients require shape {self.form_type.value_shape}; got {values.shape}."
            )
        return values

    def __call__(self, coordinates: ArrayLike, /) -> Array:
        return _pointwise_array(
            self._coefficients_point, coordinates, self.chart.dimension
        )


@final
class _WedgeCoefficient(StrictModule):
    left: DifferentialForm
    right: DifferentialForm
    product: FiberProduct = eqx.field(static=True)

    def __init__(
        self, left: DifferentialForm, right: DifferentialForm, product: FiberProduct, /
    ) -> None:
        self.left, self.right, self.product = left, right, product

    def __call__(self, coordinates: Array, /) -> Array:
        return _algebra.wedge(
            self.left._coefficients_point(coordinates),
            self.right._coefficients_point(coordinates),
            self.left.form_type,
            self.right.form_type,
            product=self.product,
        )


@final
class _ExteriorDerivativeCoefficient(StrictModule):
    form: DifferentialForm

    def __init__(self, form: DifferentialForm, /) -> None:
        self.form = form

    def __call__(self, coordinates: Array, /) -> Array:
        return _algebra.exterior_derivative_from_jacobian(
            jax.jacfwd(self.form._coefficients_point)(coordinates), self.form.form_type
        )


@final
class _PullbackFormCoefficient(StrictModule):
    form: DifferentialForm
    map: DifferentiableMap
    coorientation: int | None = eqx.field(static=True)

    def __init__(
        self, form: DifferentialForm, map: DifferentiableMap, coorientation: int | None, /
    ) -> None:
        self.form, self.map, self.coorientation = form, map, coorientation

    def __call__(self, coordinates: Array, /) -> Array:
        return _algebra.pullback(
            self.form._coefficients_point(self.map.map_function(coordinates)),
            self.form.form_type,
            self.map.jacobian(coordinates),
            coorientation=self.coorientation,
        )


@final
class _InteriorProductCoefficient(StrictModule):
    vector_field: Callable[[Array], Array]
    form: DifferentialForm

    def __init__(
        self, vector_field: Callable[[Array], Array], form: DifferentialForm, /
    ) -> None:
        self.vector_field, self.form = vector_field, form

    def __call__(self, coordinates: Array, /) -> Array:
        vector = jnp.asarray(self.vector_field(coordinates))
        if vector.shape != (self.form.chart.dimension,):
            raise ValueError(
                "Interior vector field shape does not match chart dimension."
            )
        return _algebra.interior(
            vector, self.form._coefficients_point(coordinates), self.form.form_type
        )


@final
class _SumFormCoefficient(StrictModule):
    left: DifferentialForm
    right: DifferentialForm

    def __init__(self, left: DifferentialForm, right: DifferentialForm, /) -> None:
        self.left, self.right = left, right

    def __call__(self, coordinates: Array, /) -> Array:
        return self.left._coefficients_point(
            coordinates
        ) + self.right._coefficients_point(coordinates)


@final
class _ScaledFormCoefficient(StrictModule):
    form: DifferentialForm
    scale: int = eqx.field(static=True)

    def __init__(self, form: DifferentialForm, scale: int, /) -> None:
        self.form, self.scale = form, scale

    def __call__(self, coordinates: Array, /) -> Array:
        return self.scale * self.form._coefficients_point(coordinates)


@final
class _TwistCoefficient(StrictModule):
    form: DifferentialForm
    orientation: int = eqx.field(static=True)

    def __init__(self, form: DifferentialForm, orientation: int, /) -> None:
        self.form, self.orientation = form, orientation

    def __call__(self, coordinates: Array, /) -> Array:
        values = self.form._coefficients_point(coordinates)
        match self.form.twist:
            case "untwisted":
                return _algebra.to_twisted(values, self.form.form_type, self.orientation)
            case "twisted":
                return _algebra.to_untwisted(
                    values, self.form.form_type, self.orientation
                )


@final
class _HodgeStarCoefficient(StrictModule):
    form: DifferentialForm
    metric: AbstractSemiRiemannianMetric

    def __init__(
        self, form: DifferentialForm, metric: AbstractSemiRiemannianMetric, /
    ) -> None:
        self.form, self.metric = form, metric

    def __call__(self, coordinates: Array, /) -> Array:
        return _algebra.hodge_star(
            self.form._coefficients_point(coordinates),
            self.form.form_type,
            self.metric.inverse(coordinates),
            self.metric.volume_density(coordinates),
        )


def _form(
    coefficients: Callable[[Array], Array], chart: CoordinateChart, form_type: FormType, /
) -> DifferentialForm:
    return DifferentialForm(
        coefficients,
        chart=chart,
        degree=form_type.degree,
        twist=form_type.twist,
        fiber_shape=form_type.fiber_shape,
    )


def wedge(
    left: DifferentialForm,
    right: DifferentialForm,
    /,
    *,
    product: FiberProduct = "scalar",
) -> DifferentialForm:
    if not isinstance(left, DifferentialForm) or not isinstance(right, DifferentialForm):
        raise TypeError("wedge requires two DifferentialForm instances.")
    _require_same_chart(left.chart, right.chart)
    form_type = left.form_type.wedge_type(right.form_type, product=product)
    return _form(_WedgeCoefficient(left, right, product), left.chart, form_type)


def exterior_derivative(form: DifferentialForm, /) -> DifferentialForm:
    if not isinstance(form, DifferentialForm):
        raise TypeError("exterior_derivative requires a DifferentialForm.")
    return _form(
        _ExteriorDerivativeCoefficient(form),
        form.chart,
        form.form_type.exterior_derivative_type(),
    )


def pullback_form(
    form: DifferentialForm, map: DifferentiableMap, /, *, coorientation: int | None = None
) -> DifferentialForm:
    if not isinstance(form, DifferentialForm) or not isinstance(map, DifferentiableMap):
        raise TypeError("pullback_form requires a form and a DifferentiableMap.")
    _require_same_chart(map.target, form.chart)
    if form.twist == "twisted" and map.source.dimension != map.target.dimension:
        _algebra._coorientation(coorientation)
    form_type = FormType(
        map.source.dimension, form.degree, twist=form.twist, fiber_shape=form.fiber_shape
    )
    return _form(
        _PullbackFormCoefficient(form, map, coorientation), map.source, form_type
    )


def interior_product(
    vector_field: Callable[[Array], Array], form: DifferentialForm, /
) -> DifferentialForm:
    if not callable(vector_field) or not isinstance(form, DifferentialForm):
        raise TypeError(
            "Interior product requires a callable vector and a DifferentialForm."
        )
    return _form(
        _InteriorProductCoefficient(vector_field, form),
        form.chart,
        form.form_type.interior_type(),
    )


def _add_forms(left: DifferentialForm, right: DifferentialForm, /) -> DifferentialForm:
    _require_same_chart(left.chart, right.chart)
    if left.form_type.form_type_id != right.form_type.form_type_id:
        raise ValueError("Only forms with identical scientific types can be added.")
    return _form(_SumFormCoefficient(left, right), left.chart, left.form_type)


def lie_derivative(
    vector_field: Callable[[Array], Array], form: DifferentialForm, /
) -> DifferentialForm:
    if not callable(vector_field) or not isinstance(form, DifferentialForm):
        raise TypeError(
            "Lie derivative requires a callable vector and a DifferentialForm."
        )
    if form.degree == 0:
        return interior_product(vector_field, exterior_derivative(form))
    first = exterior_derivative(interior_product(vector_field, form))
    if form.degree == form.chart.dimension:
        return first
    return _add_forms(first, interior_product(vector_field, exterior_derivative(form)))


def hodge_star(
    form: DifferentialForm, metric: AbstractSemiRiemannianMetric, /
) -> DifferentialForm:
    """Orientation-free Hodge star; tensor with the orientation line."""
    if not isinstance(form, DifferentialForm) or not isinstance(
        metric, AbstractSemiRiemannianMetric
    ):
        raise TypeError(
            "hodge_star requires a DifferentialForm and a nondegenerate metric."
        )
    _require_same_chart(form.chart, metric.chart)
    return _form(
        _HodgeStarCoefficient(form, metric), form.chart, form.form_type.hodge_dual()
    )


def codifferential(
    form: DifferentialForm, metric: AbstractSemiRiemannianMetric, /
) -> DifferentialForm:
    """Metric adjoint delta, preserving twist and refusing degree zero."""
    if not isinstance(form, DifferentialForm) or not isinstance(
        metric, AbstractSemiRiemannianMetric
    ):
        raise TypeError(
            "codifferential requires a DifferentialForm and a nondegenerate metric."
        )
    _require_same_chart(form.chart, metric.chart)
    form_type = form.form_type.codifferential_type()
    result = hodge_star(exterior_derivative(hodge_star(form, metric)), metric)
    sign = _algebra.codifferential_sign(form.form_type, metric.signature.index)
    return _form(_ScaledFormCoefficient(result, sign), form.chart, form_type)


def hodge_laplacian(
    form: DifferentialForm, metric: AbstractSemiRiemannianMetric, /
) -> DifferentialForm:
    if not isinstance(form, DifferentialForm) or not isinstance(
        metric, AbstractSemiRiemannianMetric
    ):
        raise TypeError("hodge_laplacian requires a form and a nondegenerate metric.")
    _require_same_chart(form.chart, metric.chart)
    if form.degree == 0:
        return codifferential(exterior_derivative(form), metric)
    first = exterior_derivative(codifferential(form, metric))
    if form.degree == form.chart.dimension:
        return first
    return _add_forms(first, codifferential(exterior_derivative(form), metric))


def to_untwisted(form: DifferentialForm, orientation: int, /) -> DifferentialForm:
    if not isinstance(form, DifferentialForm):
        raise TypeError("to_untwisted requires a DifferentialForm.")
    if form.twist != "twisted":
        raise ValueError("to_untwisted requires a twisted form.")
    sign = _algebra._coorientation(orientation)
    return _form(
        _TwistCoefficient(form, sign), form.chart, form.form_type.with_twist("untwisted")
    )


def to_twisted(form: DifferentialForm, orientation: int, /) -> DifferentialForm:
    if not isinstance(form, DifferentialForm):
        raise TypeError("to_twisted requires a DifferentialForm.")
    if form.twist != "untwisted":
        raise ValueError("to_twisted requires an untwisted form.")
    sign = _algebra._coorientation(orientation)
    return _form(
        _TwistCoefficient(form, sign), form.chart, form.form_type.with_twist("twisted")
    )


__all__ = [
    "DifferentialForm",
    "codifferential",
    "exterior_derivative",
    "hodge_laplacian",
    "hodge_star",
    "interior_product",
    "lie_derivative",
    "pullback_form",
    "to_untwisted",
    "to_twisted",
    "wedge",
]
