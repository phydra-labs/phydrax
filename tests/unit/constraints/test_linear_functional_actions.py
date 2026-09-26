from typing import Any

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
import phydrax.axes as cx
from phydrax.conditions import (
    LinearFunctional,
    LinearReductionAction,
    MatrixLinearFunctional,
    PointJetAction,
)
from phydrax.domain import BatchEvaluator, Interval1d


class _KeyedBatchValue(BatchEvaluator):
    def __call_batch__(
        self, batch: Any, /, *, key: Any = jr.key(0), **kwargs: Any
    ) -> Any:
        del kwargs
        reference = batch["x"]
        return cx.AxisArray(
            jnp.broadcast_to(jr.uniform(key), reference.data.shape[:-1]),
            dims=reference.dims[:-1],
        )


def test_point_jet_action_without_key_matches_default_field_evaluation() -> None:
    domain = Interval1d(0.0, 1.0)
    points = domain.component().points({"x": jnp.asarray([[0.0], [1.0]])})
    field = domain.Function("x")(_KeyedBatchValue())
    action = PointJetAction("u", points, jnp.asarray([[1.0, 2.0]]))

    result = action.apply({"u": field})

    expected = 3.0 * jnp.asarray(field(points).data)[0]
    assert jnp.allclose(result, jnp.asarray([expected]))


def test_linear_reduction_action_without_key_uses_default_reduction_key() -> None:
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    realization = phx.integration.materialize(
        phx.integration.over(domain.component()),
        phx.integration.QuasiMonteCarloPlan(16, num_replicates=4),
        key=jr.key(10),
    )
    prepared = phx.integration.prepare_linear_reduction(realization)
    field = domain.Function("x")(lambda x: x)
    action = LinearReductionAction("u", prepared, jnp.asarray([2.0]))

    result = action.apply({"u": field})

    assert jnp.allclose(result, 2.0 * jnp.asarray(prepared.apply(field).data))


def test_point_jet_actions_compose_value_and_derivative_rows() -> None:
    domain = Interval1d(0.0, 1.0)
    points = domain.component().points({"x": jnp.asarray([[0.0], [1.0]])})

    @domain.Function("x")
    def field(x: Any) -> Any:
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


def test_matrix_linear_functional_content_participates_in_composite_identity() -> None:
    first = MatrixLinearFunctional(("u",), ((1,),), (jnp.asarray([[1.0]]),))
    second = MatrixLinearFunctional(("u",), ((1,),), (jnp.asarray([[2.0]]),))

    assert (
        LinearFunctional((first,)).operator_id != LinearFunctional((second,)).operator_id
    )
