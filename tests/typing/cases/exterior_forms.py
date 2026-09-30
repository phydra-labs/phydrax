"""Public scientific form types and smooth carrier inference."""

from collections.abc import Callable

from jax import Array
from typing_extensions import assert_type

from phydrax import metrix
from phydrax.exterior import (
    form_to_vector,
    FormProxy,
    FormTwist,
    FormType,
    FormValueSpec,
    hodge_star,
    interior,
    map_reference_values,
    pullback,
    vector_to_form,
    wedge,
)


def exterior_types(
    values: Array, metric_inverse: Array, jacobian: Array, vector: Array
) -> None:
    one = FormType(3, 1, twist="untwisted")
    assert_type(one, FormType)
    assert_type(one.twist, FormTwist)
    assert_type(one.degree, int)
    assert_type(one.value_shape, tuple[int, ...])
    assert_type(one.hodge_dual(), FormType)
    assert_type(one.exterior_derivative_type(), FormType)
    assert_type(one.codifferential_type(), FormType)
    assert_type(one.trace_type(), FormType)
    spec = FormValueSpec(one, proxy="circulation")
    assert_type(spec.proxy, FormProxy)
    assert_type(spec.form_type, FormType)
    assert_type(spec.value_spec_id, str)
    assert_type(vector_to_form(values, spec), Array)
    assert_type(form_to_vector(values, spec), Array)
    assert_type(map_reference_values(values, spec, jacobian), Array)
    assert_type(pullback(values, one, jacobian), Array)
    assert_type(interior(vector, values, one), Array)
    assert_type(hodge_star(values, one, metric_inverse, 1.0), Array)
    assert_type(wedge(values, values, one, one), Array)
    FormType(3, 1, twist="density")  # ty: ignore[invalid-argument-type]
    FormValueSpec(one, proxy="hdiv")  # ty: ignore[invalid-argument-type]
    wedge(values, values, one, one, product="tensor")  # ty: ignore[invalid-argument-type]


def chart_types(
    coefficients: Callable[[Array], Array],
    chart: metrix.CoordinateChart,
    metric: metrix.AbstractSemiRiemannianMetric,
) -> None:
    form = metrix.DifferentialForm(coefficients, chart=chart, degree=1, twist="twisted")
    assert_type(form.form_type, FormType)
    assert_type(form.twist, FormTwist)
    assert_type(form.fiber_shape, tuple[int, ...])
    assert_type(metrix.exterior_derivative(form), metrix.DifferentialForm)
    assert_type(metrix.hodge_star(form, metric), metrix.DifferentialForm)
    assert_type(metrix.codifferential(form, metric), metrix.DifferentialForm)
    assert_type(metrix.to_untwisted(form, 1), metrix.DifferentialForm)
    metrix.hodge_star(form, metric, orientation=1)  # ty: ignore[unknown-argument]
