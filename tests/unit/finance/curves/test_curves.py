#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.finance.core import Currency, FinanceDate
from phydrax.finance.curves._core import (
    CurveDefinition,
    CurveGrid,
    CurveRepresentation,
    CurveSet,
    InterpolationPolicy,
    PreparedCurve,
)


USD = Currency("USD", 2)
VALUATION_DATE = FinanceDate.from_iso("2026-09-08")
EXPLICIT_LINEAR = InterpolationPolicy(
    "linear", left_extrapolation="forbid", right_extrapolation="flat_forward"
)


def _definition(curve_id: Any, representation: Any) -> Any:
    return CurveDefinition(
        curve_id=curve_id,
        role="test",
        valuation_date=VALUATION_DATE,
        currency=USD,
        representation=representation,
        grid=CurveGrid(jnp.array([0.0, 1.0, 2.0, 5.0])),
        interpolation=EXPLICIT_LINEAR,
    )


def test_curves_scenario_1() -> None:
    zero = PreparedCurve(
        _definition("zero", CurveRepresentation.ZERO_RATE),
        jnp.full((4,), 0.0275),
    )
    forward = PreparedCurve(
        _definition("forward", CurveRepresentation.FORWARD_RATE),
        jnp.full((4,), 0.0275),
    )
    times = jnp.array([0.0, 0.25, 1.5, 4.0, 8.0])

    expected = jnp.exp(-0.0275 * times)
    np.testing.assert_allclose(zero.discount_factor(times), expected, rtol=2e-6)
    np.testing.assert_allclose(forward.discount_factor(times), expected, rtol=2e-6)
    curve = PreparedCurve(
        _definition("negative", CurveRepresentation.LOG_DISCOUNT),
        jnp.array([0.0, 0.01, 0.02, 0.05]),
    )

    discount = curve.discount_factor(jnp.array([0.5, 1.5, 6.0]))
    assert bool(jnp.all(discount > 1.0))
    assert bool(jnp.all(discount > 0.0))
    discount_definition = _definition("discount", CurveRepresentation.LOG_DISCOUNT)
    survival_definition = _definition("survival", CurveRepresentation.LOG_SURVIVAL)

    with pytest.raises(ValueError, match="strictly positive"):
        PreparedCurve.from_discount_factors(
            discount_definition, jnp.array([1.0, 0.99, 0.0, 0.9])
        )
    with pytest.raises(ValueError, match="strictly positive"):
        PreparedCurve.from_survival_probabilities(
            survival_definition, jnp.array([1.0, 0.98, -0.1, 0.8])
        )


def test_curves_scenario_2() -> None:
    definition = CurveDefinition(
        curve_id="bounded",
        role="test",
        valuation_date=VALUATION_DATE,
        currency=USD,
        representation="zero_rate",
        grid=CurveGrid(jnp.array([0.0, 1.0, 2.0])),
        interpolation=InterpolationPolicy(
            "linear",
            left_extrapolation="forbid",
            right_extrapolation="forbid",
        ),
    )
    curve = PreparedCurve(definition, jnp.array([0.01, 0.02, 0.03]))

    with pytest.raises(Exception, match="above the last node"):
        curve.discount_factor(3.0)
    definition = _definition("sensitive", CurveRepresentation.ZERO_RATE)
    node_quote = jnp.diag(jnp.array([1.0, 2.0, 3.0, 4.0]))
    curve = PreparedCurve(
        definition,
        jnp.array([0.01, 0.02, 0.03, 0.04]),
        node_quote_jacobian=node_quote,
        quote_ids=("q0", "q1", "q2", "q3"),
    )

    node = curve.node_sensitivity(jnp.array([0.5, 1.5]), quantity="discount_factor")
    quote = curve.quote_sensitivity(jnp.array([0.5, 1.5]), quantity="discount_factor")

    np.testing.assert_allclose(quote.jacobian, node.jacobian @ node_quote)
    assert node.input_ids == tuple(f"sensitive:node:{index}" for index in range(4))
    assert quote.input_ids == ("q0", "q1", "q2", "q3")
    first = PreparedCurve(
        _definition("discount", CurveRepresentation.ZERO_RATE), jnp.zeros((4,))
    )
    second = PreparedCurve(
        _definition("projection", CurveRepresentation.FORWARD_RATE), jnp.zeros((4,))
    )
    curves = CurveSet((first, second))

    assert curves.curve("projection") is second
    refreshed = curves.refresh_numeric(
        (first.refresh_numeric(jnp.full((4,), 0.01)), second)
    )
    np.testing.assert_allclose(refreshed.curve("discount").node_values, 0.01)

    changed_definition = CurveDefinition(
        curve_id="discount",
        role="test",
        valuation_date=VALUATION_DATE,
        currency=USD,
        representation="zero_rate",
        grid=CurveGrid(jnp.array([0.0, 1.0, 3.0, 5.0])),
        interpolation=EXPLICIT_LINEAR,
    )
    with pytest.raises(ValueError, match="topology"):
        curves.refresh_numeric(
            (PreparedCurve(changed_definition, jnp.zeros((4,))), second)
        )
