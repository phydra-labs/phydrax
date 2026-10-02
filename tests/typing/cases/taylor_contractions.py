from __future__ import annotations

from typing import assert_type

import jax.numpy as jnp
from jax import Array

from phydrax.operators.differential import (
    evaluate_taylor_contractions,
    plan_taylor_contractions,
    TaylorContractionPlan,
    TaylorContractionPolicy,
    TaylorContractionRequest,
    TaylorContractionResources,
    TaylorContractionResult,
)
from phydrax.operators.differential._taylor_contracts import (
    TaylorDirectionDim,
    TaylorEventDims,
    TaylorRequestDim,
)
from phydrax.typing import Bool, Identifier, Identifiers, Inexact


request = TaylorContractionRequest(("first", "second"), (2, 1), request_id="mixed")
assert_type(request, TaylorContractionRequest)
assert_type(request.direction_ids, Identifiers[TaylorDirectionDim])
assert_type(request.request_id, Identifier)
resources = TaylorContractionResources(max_order=16, workset_size=4)
policy = TaylorContractionPolicy(strategy="linear", resources=resources)
plan = plan_taylor_contractions((request,), policy=policy)
assert_type(plan, TaylorContractionPlan)
assert_type(plan.required_regularity_order, int)
assert_type(plan.direction_ids, tuple[str, ...])


def polynomial(first: Array, second: Array) -> Array:
    return first**2 * second


result = evaluate_taylor_contractions(
    polynomial,
    (jnp.asarray(1.0, dtype=jnp.float32), jnp.asarray(2.0, dtype=jnp.float32)),
    {
        "first": (
            jnp.asarray(1.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        ),
        "second": (
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(1.0, dtype=jnp.float32),
        ),
    },
    plan,
)
assert_type(result, TaylorContractionResult)
assert_type(result.primal, Inexact[TaylorEventDims])
assert_type(result.values, Inexact[TaylorRequestDim, TaylorEventDims])
assert_type(result.finite, Bool[TaylorRequestDim])
assert_type(result.derivative_valid, Bool[TaylorRequestDim])
TaylorContractionPolicy(strategy="unchecked")  # ty: ignore[invalid-argument-type]
TaylorContractionRequest(("first",), ("twice",))  # ty: ignore[invalid-argument-type]
TaylorContractionResources(max_order="large")  # ty: ignore[invalid-argument-type]
plan_taylor_contractions((request,), policy="linear")  # ty: ignore[invalid-argument-type]
