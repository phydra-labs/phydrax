from fractions import Fraction as F

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.geometry._circle_fan_material import CircleFanMaterialMap
from phydrax.geometry.brep._patches import CircleCurve
from phydrax.geometry.brep._trim_helpers import PeriodOffset


def test_source_circle_fan_points_jacobian_and_edge_trace() -> None:
    curve = CircleCurve((0, 0, 0), (1, 0, 0), (0, 1, 0), 2)
    fan = CircleFanMaterialMap(
        curve, (F(0), F(0), F(0)), F(0), PeriodOffset(F(0), F(1, 4))
    )
    reference = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.2, 0.3)))
    points = fan.evaluate_points(reference)
    _, jacobian = fan.evaluate(reference[3:])
    np.testing.assert_allclose(
        np.asarray(points[:3]), ((0, 0, 0), (2, 0, 0), (0, 2, 0)), atol=1e-6
    )
    independent = jax.jacfwd(lambda p: fan.evaluate_points(p))(reference[3])
    np.testing.assert_allclose(
        np.asarray(jacobian[0]), np.asarray(independent), atol=1e-6
    )
    assert np.linalg.det(np.asarray(jacobian[0]).T @ np.asarray(jacobian[0])) > 0
    t = jnp.asarray((0.2, 0.5, 0.8))
    edge_points = jnp.stack((1 - t, t), axis=-1)
    source_points = jax.vmap(curve.evaluate)(t * (jnp.pi / 2))
    np.testing.assert_allclose(
        np.asarray(fan.evaluate_points(edge_points)), np.asarray(source_points), atol=1e-6
    )
    assert fan.orientation == 1 and fan.star_lower > 0
    reverse = CircleFanMaterialMap(
        curve, (F(0), F(0), F(0)), PeriodOffset(F(0), F(1, 4)), F(0)
    )
    assert reverse.orientation == -1
    np.testing.assert_allclose(
        np.asarray(reverse.evaluate_points(edge_points)),
        np.asarray(fan.evaluate_points(edge_points[::-1])),
        atol=1e-6,
    )


def test_circle_fan_refuses_singular_and_unauthorized_geometry() -> None:
    curve = CircleCurve((0, 0, 0), (1, 0, 0), (0, 1, 0), 1)
    with pytest.raises(ValueError, match="exact plane"):
        CircleFanMaterialMap(curve, (F(0), F(0), F(1)), F(0), F(1))
    with pytest.raises(ValueError, match="native turn"):
        CircleFanMaterialMap(curve, (F(0), F(0), F(0)), F(0), PeriodOffset(F(0), F(1)))
    with pytest.raises(ValueError, match="Jacobian|orientation"):
        CircleFanMaterialMap(
            curve,
            (F(2), F(0), F(0)),
            PeriodOffset(F(0), F(-1, 4)),
            PeriodOffset(F(0), F(1, 4)),
        )
