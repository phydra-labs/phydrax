# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""IGA fixture migrations retain declared axes, fields, and authored weight scale."""

from __future__ import annotations

import numpy as np
import pytest

from tests._support.iga_fields import quarter_annulus_query, quarter_annulus_trace


def test_query_fixture_retains_parameter_axis_identity_and_weight_scale() -> None:
    fixture = quarter_annulus_query()

    assert fixture.plan.basis.axis_names == ("r", "theta")
    assert fixture.prepared.basis.axis_names == fixture.plan.basis.axis_names
    assert fixture.prepared.default_runtime.numeric_version == "annulus"
    assert fixture.plan.fields[0].name == "u"
    assert fixture.plan.fields[0].component_shape == ()
    np.testing.assert_array_equal(fixture.plan.geometry.control_points, fixture.controls)
    # Boehm insertion at one half averages the unit and sqrt(1/2) weights.
    expected = np.asarray(
        (1.0, 0.5 * (1.0 + np.sqrt(0.5)), 0.5 * (1.0 + np.sqrt(0.5)), 1.0)
    )
    np.testing.assert_array_equal(
        fixture.plan.geometry.weights, np.broadcast_to(expected, (4, 4))
    )


@pytest.mark.parametrize(
    ("vector", "field_name", "component_shape"),
    (
        pytest.param(False, "u", (), id="scalar-value"),
        pytest.param(True, "v", (2,), id="vector-value"),
    ),
)
def test_trace_fixture_retains_parameter_and_field_identities(
    vector: bool, field_name: str, component_shape: tuple[int, ...]
) -> None:
    fixture = quarter_annulus_trace(vector=vector)

    assert fixture.plan.basis.axis_names == ("r", "t")
    assert fixture.prepared.basis.axis_names == fixture.plan.basis.axis_names
    assert fixture.plan.fields[0].name == field_name
    assert fixture.plan.fields[0].component_shape == component_shape
    np.testing.assert_array_equal(fixture.plan.geometry.control_points, fixture.controls)
    assert fixture.prepared.default_runtime.numeric_version == "annulus"
