#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..sparse import EdgeRelation, SparseLinearMap


class LinearRelationCostEstimate(StrictModule):
    source_size: int = eqx.field(static=True)
    target_size: int = eqx.field(static=True)
    structural_entries: int = eqx.field(static=True)
    route_bytes: int = eqx.field(static=True)
    coefficient_bytes: int = eqx.field(static=True)
    dense_bytes: int = eqx.field(static=True)


class LinearRoutePlan(StrictModule):
    """Static source-to-equation routes shared by circuit lowerings."""

    relation: EdgeRelation
    cost: LinearRelationCostEstimate
    plan_id: str = eqx.field(static=True)


class PreparedLinearRelation(StrictModule):
    """Frequency/case-batched coefficients bound to one immutable route plan."""

    plan: LinearRoutePlan
    operator: SparseLinearMap
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)

    def apply(self, value: ArrayLike, /) -> Array:
        return self.operator.mv(jnp.asarray(value))

    def materialize(self, /, *, maximum_bytes: int) -> Array:
        if maximum_bytes <= 0:
            raise ValueError("maximum_bytes must be positive.")
        batch_count = prod(self.operator.batch_shape) if self.operator.batch_shape else 1
        required = (
            batch_count
            * self.plan.cost.source_size
            * self.plan.cost.target_size
            * self.operator.coefficients.dtype.itemsize
        )
        if required > int(maximum_bytes):
            raise MemoryError("Linear relation materialization exceeds maximum_bytes.")
        return self.operator.as_dense()


def plan_linear_routes(
    source_size: int,
    target_size: int,
    source_indices: Sequence[int] | ArrayLike,
    target_indices: Sequence[int] | ArrayLike,
    /,
    *,
    plan_id: str | None = None,
) -> LinearRoutePlan:
    source_count, target_count = int(source_size), int(target_size)
    if source_count <= 0 or target_count <= 0:
        raise ValueError("Linear relation spaces must be nonempty.")
    relation = EdgeRelation(
        jnp.asarray(source_indices, dtype=jnp.int32),
        jnp.asarray(target_indices, dtype=jnp.int32),
        source_size=source_count,
        target_size=target_count,
    )
    route_bytes = sum(
        int(value.size * value.dtype.itemsize)
        for value in (
            relation.source_indices,
            relation.target_indices,
            relation.valid,
        )
    )
    entries = relation.capacity
    identifier = (
        canonical_fingerprint(
            {
                "kind": "circuit-linear-route-plan",
                "source_size": source_count,
                "target_size": target_count,
                "source_indices": list(map(int, relation.source_indices.tolist())),
                "target_indices": list(map(int, relation.target_indices.tolist())),
            }
        )
        if plan_id is None
        else str(plan_id)
    )
    if not identifier:
        raise ValueError("plan_id must be non-empty.")
    cost = LinearRelationCostEstimate(
        source_count,
        target_count,
        entries,
        route_bytes,
        entries * jnp.dtype(jnp.complex128).itemsize,
        source_count * target_count * jnp.dtype(jnp.complex128).itemsize,
    )
    return LinearRoutePlan(relation, cost, identifier)


def plan_block_diagonal_routes(
    block_sizes: Sequence[int],
    /,
    *,
    plan_id: str | None = None,
) -> LinearRoutePlan:
    sizes = tuple(int(value) for value in block_sizes)
    if not sizes or any(value <= 0 for value in sizes):
        raise ValueError("block_sizes must contain positive sizes.")
    sources: list[int] = []
    targets: list[int] = []
    offset = 0
    for size in sizes:
        for row in range(size):
            for column in range(size):
                sources.append(offset + column)
                targets.append(offset + row)
        offset += size
    return plan_linear_routes(
        offset,
        offset,
        sources,
        targets,
        plan_id=plan_id,
    )


def bind_linear_relation(
    plan: LinearRoutePlan,
    coefficients: ArrayLike,
    /,
    *,
    numeric_version: ArrayLike = 0,
    operator_id: str | None = None,
) -> PreparedLinearRelation:
    if not isinstance(plan, LinearRoutePlan):
        raise TypeError("plan must be LinearRoutePlan.")
    values = jnp.asarray(coefficients)
    if values.shape[-1:] != (plan.cost.structural_entries,):
        raise ValueError("Linear relation coefficients have the wrong route count.")
    if not jnp.issubdtype(values.dtype, jnp.inexact):
        values = values.astype(float)
    version = jnp.asarray(numeric_version, dtype=jnp.int32)
    if version.shape != () or bool(version < 0):
        raise ValueError("numeric_version must be one nonnegative scalar.")
    identifier = f"{plan.plan_id}/operator" if operator_id is None else str(operator_id)
    if not identifier:
        raise ValueError("operator_id must be non-empty.")
    operator = SparseLinearMap(plan.relation, values, operator_id=identifier)
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-circuit-linear-relation",
            "plan": plan.plan_id,
            "operator": operator.operator_id,
        }
    )
    return PreparedLinearRelation(plan, operator, version, prepared_id)


def bind_block_diagonal_relation(
    plan: LinearRoutePlan,
    blocks: Sequence[ArrayLike],
    /,
    *,
    numeric_version: ArrayLike = 0,
    operator_id: str | None = None,
) -> PreparedLinearRelation:
    arrays = tuple(jnp.asarray(block) for block in blocks)
    if not arrays or any(
        block.ndim < 2 or block.shape[-2] != block.shape[-1] for block in arrays
    ):
        raise ValueError("blocks must contain square matrices.")
    batch_shape = arrays[0].shape[:-2]
    if any(block.shape[:-2] != batch_shape for block in arrays):
        raise ValueError("Block relation batches must match.")
    expected = sum(int(block.shape[-1]) ** 2 for block in arrays)
    if expected != plan.cost.structural_entries:
        raise ValueError("Block sizes do not match the route plan.")
    coefficients = jnp.concatenate(
        tuple(block.reshape(batch_shape + (-1,)) for block in arrays), axis=-1
    )
    return bind_linear_relation(
        plan,
        coefficients,
        numeric_version=numeric_version,
        operator_id=operator_id,
    )


__all__ = [
    "LinearRelationCostEstimate",
    "LinearRoutePlan",
    "PreparedLinearRelation",
    "bind_block_diagonal_relation",
    "bind_linear_relation",
    "plan_block_diagonal_routes",
    "plan_linear_routes",
]
