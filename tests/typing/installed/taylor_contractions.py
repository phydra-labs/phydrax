from typing import assert_type

import jax.numpy as jnp
from jax import Array

from phydrax.operators.differential import (
    evaluate_taylor_contractions,
    plan_taylor_contractions,
    TaylorContractionPlan,
    TaylorContractionPolicy,
    TaylorContractionRequest,
    TaylorContractionResult,
)


def polynomial(point: Array) -> Array:
    return point**3


request = TaylorContractionRequest(("direction",), (3,))
plan = plan_taylor_contractions(
    (request,), policy=TaylorContractionPolicy(strategy="linear")
)
result = evaluate_taylor_contractions(
    polynomial,
    (jnp.asarray(0.5, dtype=jnp.float32),),
    {"direction": (jnp.asarray(1.0, dtype=jnp.float32),)},
    plan,
)
assert_type(plan, TaylorContractionPlan)
assert_type(result, TaylorContractionResult)
assert_type(result.plan, TaylorContractionPlan)
assert_type(plan.required_regularity_order, int)
assert_type(result.values[0], Array)
