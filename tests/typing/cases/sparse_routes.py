"""Sparse forward accumulation selectors remain explicit at typed boundaries."""

from jax import Array

from phydrax.sparse import EdgeRelation, linear_apply, route_reduce, RowRelation


def ordered_routes(
    relation: EdgeRelation | RowRelation, coefficients: Array, values: Array
) -> None:
    route_reduce(relation, values, reduction="mean", accumulation="compensated")
    route_reduce(relation, values, accumulation="deterministic")
    linear_apply(relation, coefficients, values, accumulation="deterministic")
    linear_apply(relation, coefficients, values, accumulation="compensated")
    route_reduce(relation, values, accumulation="unordered")  # ty: ignore[invalid-argument-type]
    linear_apply(relation, coefficients, values, accumulation="unordered")  # ty: ignore[invalid-argument-type]
