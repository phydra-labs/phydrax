import jax.numpy as jnp

from phydrax.conditions import LinearFunctional, PointJetAction
from phydrax.domain import Interval1d


def test_point_jet_actions_compose_value_and_derivative_rows():
    domain = Interval1d(0.0, 1.0)
    points = domain.component().points({"x": jnp.asarray([[0.0], [1.0]])})

    @domain.Function("x")
    def field(x):
        return x[0] ** 2

    value = PointJetAction(
        "u",
        points,
        jnp.asarray([[1.0, -1.0], [0.0, 0.0]]),
    )
    derivative = PointJetAction(
        "u",
        points,
        jnp.asarray([[0.0, 0.0], [1.0, -1.0]]),
        derivatives=(("x", 0, 1),),
    )
    action = LinearFunctional((value, derivative))
    result = action.linear_action({"u": field})
    assert jnp.allclose(result, jnp.asarray([-1.0, -2.0]))
