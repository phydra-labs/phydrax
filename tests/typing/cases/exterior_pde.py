"""Static public contracts for exterior-aware equation construction."""

from typing import assert_type

from phydrax.equations import PDEExpression, PDEField, PDEFormGeometry, PDEValueType
from phydrax.exterior import FormType, FormValueSpec


spec = FormValueSpec(FormType(3, 0), proxy="scalar")
field = PDEField("u", coordinates=("x",), form=spec)
assert_type(field.form, FormValueSpec | None)
u = PDEExpression.field("u")
assert_type(u.exterior_derivative(), PDEExpression)
assert_type(u.codifferential(), PDEExpression)
assert_type(u.hodge_star(), PDEExpression)
assert_type(u.wedge(u), PDEExpression)
assert_type(u.interior_product(PDEExpression.field("v")), PDEExpression)
assert_type(u.lie_derivative(PDEExpression.field("v")), PDEExpression)
assert_type(u.trace("wall"), PDEExpression)
assert_type(PDEFormGeometry(), PDEFormGeometry)


def inferred_form(value_type: PDEValueType, /) -> FormValueSpec | None:
    return assert_type(value_type.form, FormValueSpec | None)
