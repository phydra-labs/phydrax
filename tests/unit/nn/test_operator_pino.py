#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp

import phydrax as phx
from phydrax.nn.models.core._base import _AbstractOperatorModel


class _QuadraticQueryOperator(_AbstractOperatorModel):
    in_size: int = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

    def __init__(self):
        self.in_size = 1
        self.out_size = "scalar"

    def __call_operator_batch__(self, batch, *, key=None):
        del key
        coordinates = batch.require_single_query().coordinates_array(
            case_shape=batch.case_shape
        )
        return coordinates[..., 0] ** 2

    def __call__(self, x, *, key=None):
        if not isinstance(x, phx.nn.OperatorBatch):
            raise TypeError("_QuadraticQueryOperator requires an OperatorBatch.")
        return self.__call_operator_batch__(x, key=key)


def _batch(resolution):
    axis = phx.nn.OperatorAxis("x", jnp.linspace(-1.0, 1.0, resolution))
    source = phx.nn.FunctionSamples(
        values=jnp.ones((2, resolution)),
        axes=(axis,),
    )
    return phx.nn.OperatorBatch(
        inputs={"forcing": source},
        queries={"query": phx.nn.FunctionSamples(values=None, axes=(axis,))},
        case_axes=("case",),
        case_shape=(2,),
    )


def test_operator_context_is_coordinate_aware_and_differentiable():
    model = _QuadraticQueryOperator()
    context = phx.nn.bind_operator_context(model, _batch(9))
    values = context(jnp.asarray([[-0.4], [0.2], [0.7]]))
    assert values.shape == (2, 3)
    assert jnp.allclose(values, jnp.asarray([0.16, 0.04, 0.49])[None, :])

    domain = phx.domain.Interval1d(-1.0, 1.0)
    prediction = context.domain_function(domain, "x")
    second_derivative = phx.operators.laplacian(prediction, var="x")
    assert jnp.allclose(second_derivative.func(jnp.asarray([0.3])), 2.0)


def test_native_differential_pino_residual_holds_across_resolutions():
    domain = phx.domain.Interval1d(-1.0, 1.0)
    model = _QuadraticQueryOperator()
    function = domain.Model("x")(model)
    constraint = phx.terms.DifferentialPhysicsInformedOperatorTerm(
        "u",
        (_batch(9), _batch(17)),
        domain,
        "x",
        lambda u: phx.operators.laplacian(u, var="x") - 2.0,
        loss="l2",
    )
    loss = constraint.loss({"u": function})
    assert loss < 1e-20
