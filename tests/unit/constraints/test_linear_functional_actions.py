import jax.numpy as jnp

from phydrax.conditions import LinearFunctional, MatrixLinearFunctional, PointJetAction
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


def test_matrix_linear_functional_content_participates_in_composite_identity():
    first = MatrixLinearFunctional(("u",), ((1,),), (jnp.asarray([[1.0]]),))
    second = MatrixLinearFunctional(("u",), ((1,),), (jnp.asarray([[2.0]]),))

    assert (
        LinearFunctional((first,)).operator_id != LinearFunctional((second,)).operator_id
    )
