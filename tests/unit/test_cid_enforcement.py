import pytest

import phydrax as phx
from phydrax.domain import Interval1d


def test_cid01_builtin_condition_lowers_to_typed_affine_equation() -> None:
    domain = Interval1d(0.0, 1.0)
    boundary = domain.component({"x": phx.domain.Boundary()})
    spec = phx.enforcement.EnforcementSpec(
        phx.conditions.Dirichlet("u", boundary, target=2.0)
    )
    # ty: ignore[unresolved-attribute]
    jet = spec.transform.equation.lhs.terms[0][1]
    # ty: ignore[unresolved-attribute]
    assert (jet.field, jet.variable, jet.order, jet.normal) == (
        "u",
        "x",
        0,
        False,
    )
    # ty: ignore[unresolved-attribute]
    assert spec.transform.proof.provider_certified


def test_cid01_untyped_callable_escape_hatch_is_rejected() -> None:
    domain = Interval1d(0.0, 1.0)
    boundary = domain.component({"x": phx.domain.Boundary()})
    with pytest.raises(TypeError, match="AffineEnforcementTransform"):
        phx.enforcement.EnforcementSpec(
            phx.conditions.Dirichlet("u", boundary, target=0.0),
            # ty: ignore[invalid-argument-type]
            transform=lambda value, fields: value,
        )
