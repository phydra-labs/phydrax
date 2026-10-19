#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np
import pytest

from phydrax.geometry.analytic import Tube
from phydrax.geometry.brep._intersection_curve import CurveRange
from phydrax.geometry.brep._parameter_range import refusal_parameter_range
from phydrax.geometry.brep._patches import CircleCurve, EllipseCurve, LineCurve


def test_full_circle_exact_open_radius_range() -> None:
    # The center has multiple nearest points at distance R; all distances < R
    # have unique projection, so the reach is exactly R independently of Tube.
    radius = 3.0
    path = CircleCurve((1, 2, 3), (1, 0, 0), (0, 1, 0), radius)
    assert Tube(path, 0.25).parameter_range is None
    source = Tube(path, np.nextafter(radius, 0.0), include_parameter_range=True)
    certificate = source.parameter_range
    assert certificate is not None
    assert certificate.parameter == "radius"
    assert certificate.interval == (0.0, radius)
    assert certificate.end_status == ("attained_exactly", "attained_exactly")
    assert certificate.constraints == ("parameter_domain", "reach")
    assert certificate.entity_ids == ((), ())
    assert certificate.evidence
    with pytest.raises(ValueError) as caught:
        Tube(path, radius)
    assert refusal_parameter_range(caught.value).interval == certificate.interval


def test_convex_segment_has_proved_infinite_reach() -> None:
    # A closed convex set has unique Euclidean projection at every distance.
    path = CurveRange(LineCurve((1, 2, 3), (0, 0, 2)), -1.0, 4.0)
    source = Tube(path, 100.0, include_parameter_range=True)
    certificate = source.parameter_range
    assert certificate is not None
    assert certificate.interval == (0.0, np.inf)
    assert certificate.end_status == ("attained_exactly", "attained_exactly")
    assert certificate.constraints == ("parameter_domain", "reach")
    assert Tube(path, 0.25).parameter_range is None


@pytest.mark.parametrize("radius", [0.0, -1.0, 3.0, 4.0, np.inf, np.nan, [0.25]])
def test_invalid_radius_keeps_exact_refusal_and_range(
    radius: float | list[float],
) -> None:
    path = CircleCurve((0, 0, 0), (1, 0, 0), (0, 1, 0), 3.0)
    value = np.asarray(radius, dtype=np.float64)
    message = (
        "Tube.radius must be finite, positive, and strictly below reach; "
        f"got {value!r} with reach 3.0."
    )
    with pytest.raises(ValueError) as caught:
        Tube(path, value)
    assert type(caught.value) is ValueError
    assert caught.value.args == (message,)
    certificate = refusal_parameter_range(caught.value)
    assert certificate.interval == (0.0, 3.0)
    assert certificate.constraints == ("parameter_domain", "reach")
    assert certificate.evidence


@pytest.mark.parametrize("include", [False, True])
def test_unsupported_curve_range_is_explicitly_unknown(include: bool) -> None:
    path = CurveRange(EllipseCurve((0, 0, 0), (1, 0, 0), (0, 1, 0), 2.0, 1.0))
    with pytest.raises(ValueError) as caught:
        Tube(path, 0.25, include_parameter_range=include)
    assert type(caught.value) is ValueError
    assert caught.value.args == (
        "Tube.path must use a supported line segment or full circle; got unsupported CurveRange.",
    )
    certificate = refusal_parameter_range(caught.value)
    assert certificate.interval == (None, None)
    assert certificate.end_status == ("unknown", "unknown")
    assert certificate.constraints == ("unknown", "unknown")
    assert certificate.evidence == (str(caught.value),)


def test_nonexact_circle_axes_refuse_with_unknown_range() -> None:
    component = np.sqrt(0.5)
    path = CircleCurve(
        (0, 0, 0), (component, component, 0), (-component, component, 0), 2.0
    )
    with pytest.raises(ValueError) as caught:
        Tube(path, 0.25, include_parameter_range=True)
    assert type(caught.value) is ValueError
    assert caught.value.args == (
        "Tube.path must have orthonormal binary circle axes; got unsupported nonorthonormal axes.",
    )
    certificate = refusal_parameter_range(caught.value)
    assert certificate.interval == (None, None)
    assert certificate.end_status == ("unknown", "unknown")
    assert certificate.evidence == (str(caught.value),)
