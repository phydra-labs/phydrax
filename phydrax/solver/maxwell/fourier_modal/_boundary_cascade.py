#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._factorization import _dense_solve
from ._layer import PreparedLayerOperator


class BoundaryCascadePolicy(StrictModule, NonTrainableState):
    """Static short-interval Taylor and stable-doubling policy."""

    doublings: int = eqx.field(static=True)
    initializer_order: int = eqx.field(static=True)
    paired_error: bool = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        doublings: int = 12,
        initializer_order: int = 6,
        paired_error: bool = True,
        relative_tolerance: float = 1e-8,
        absolute_tolerance: float = 1e-10,
    ):
        doublings_ = int(doublings)
        order = int(initializer_order)
        relative = float(relative_tolerance)
        absolute = float(absolute_tolerance)
        if doublings_ < 0:
            raise ValueError("doublings must be non-negative.")
        if order < 2:
            raise ValueError("initializer_order must be at least two.")
        if relative < 0.0 or absolute < 0.0:
            raise ValueError("Boundary-cascade tolerances must be non-negative.")
        self.doublings = doublings_
        self.initializer_order = order
        self.paired_error = bool(paired_error)
        self.relative_tolerance = relative
        self.absolute_tolerance = absolute
        self.policy_id = canonical_fingerprint(
            {
                "kind": "boundary-cascade-policy",
                "doublings": doublings_,
                "initializer_order": order,
                "paired_error": self.paired_error,
                "relative_tolerance": relative,
                "absolute_tolerance": absolute,
            }
        )


class BoundaryRelationDiagnostics(StrictModule):
    solve_residual: Array
    initializer_remainder: Array
    paired_error: Array
    finite: Array
    converged: Array


class BoundaryRelation(StrictModule):
    """Mixed field map [E_left, H_right] to [E_right, H_left]."""

    a: Array
    b: Array
    c: Array
    d: Array
    diagnostics: BoundaryRelationDiagnostics

    @property
    def tangential_size(self) -> int:
        return int(self.a.shape[0])


def identity_boundary_relation(size: int, dtype: jnp.dtype, /) -> BoundaryRelation:
    identity = jnp.eye(int(size), dtype=dtype)
    zero = jnp.zeros_like(identity)
    diagnostics = BoundaryRelationDiagnostics(
        jnp.asarray(0.0, dtype=identity.real.dtype),
        jnp.asarray(0.0, dtype=identity.real.dtype),
        jnp.asarray(0.0, dtype=identity.real.dtype),
        jnp.asarray(True),
        jnp.asarray(True),
    )
    return BoundaryRelation(identity, zero, zero, identity, diagnostics)


def _matrix_relative_residual(matrix: Array, solution: Array, rhs: Array) -> Array:
    residual = matrix @ solution - rhs
    denominator = jnp.maximum(jnp.sqrt(jnp.sum(jnp.abs(rhs) ** 2)), 1.0)
    return jnp.sqrt(jnp.sum(jnp.abs(residual) ** 2)) / denominator


def _transfer_to_boundary(transfer: Array, /) -> BoundaryRelation:
    size = transfer.shape[0] // 2
    t11 = transfer[:size, :size]
    t12 = transfer[:size, size:]
    t21 = transfer[size:, :size]
    t22 = transfer[size:, size:]
    identity = jnp.eye(size, dtype=transfer.dtype)
    right_hand_side = jnp.concatenate((t21, identity), axis=1)
    solution = _dense_solve(t22, right_hand_side)
    solve_t21 = solution[:, :size]
    inverse_t22 = solution[:, size:]
    relation = BoundaryRelation(
        t11 - t12 @ solve_t21,
        t12 @ inverse_t22,
        -solve_t21,
        inverse_t22,
        BoundaryRelationDiagnostics(
            _matrix_relative_residual(t22, solution, right_hand_side),
            jnp.asarray(0.0, dtype=transfer.real.dtype),
            jnp.asarray(0.0, dtype=transfer.real.dtype),
            jnp.all(jnp.isfinite(transfer)),
            jnp.asarray(True),
        ),
    )
    return relation


def compose_boundary_relations(
    left: BoundaryRelation,
    right: BoundaryRelation,
    /,
) -> BoundaryRelation:
    """Compose adjacent left and right boundary relations without transfer growth."""
    if left.a.shape != right.a.shape:
        raise ValueError("Boundary relations must act on the same tangential space.")
    size = left.tangential_size
    identity = jnp.eye(size, dtype=left.a.dtype)
    system = identity - left.b @ right.c
    rhs = jnp.concatenate((left.a, left.b @ right.d), axis=1)
    middle = _dense_solve(system, rhs)
    from_left = middle[:, :size]
    from_right = middle[:, size:]
    a = right.a @ from_left
    b = right.a @ from_right + right.b
    c = left.c + left.d @ right.c @ from_left
    d = left.d @ (right.c @ from_right + right.d)
    solve_residual = jnp.maximum(
        jnp.maximum(left.diagnostics.solve_residual, right.diagnostics.solve_residual),
        _matrix_relative_residual(system, middle, rhs),
    )
    initializer_remainder = jnp.maximum(
        left.diagnostics.initializer_remainder,
        right.diagnostics.initializer_remainder,
    )
    paired_error = jnp.maximum(
        left.diagnostics.paired_error,
        right.diagnostics.paired_error,
    )
    finite = (
        left.diagnostics.finite
        & right.diagnostics.finite
        & jnp.all(jnp.isfinite(a))
        & jnp.all(jnp.isfinite(b))
        & jnp.all(jnp.isfinite(c))
        & jnp.all(jnp.isfinite(d))
    )
    converged = left.diagnostics.converged & right.diagnostics.converged & finite
    return BoundaryRelation(
        a,
        b,
        c,
        d,
        BoundaryRelationDiagnostics(
            solve_residual,
            initializer_remainder,
            paired_error,
            finite,
            converged,
        ),
    )


def _taylor_transfer(
    matrix: Array,
    thickness: Array,
    doublings: int,
    order: int,
    /,
) -> tuple[Array, Array]:
    scaled = matrix * (thickness / (2**doublings))
    identity = jnp.eye(matrix.shape[0], dtype=matrix.dtype)
    transfer = identity
    term = identity
    for degree in range(1, order + 1):
        term = term @ scaled / degree
        transfer = transfer + term
    next_term = term @ scaled / (order + 1)
    remainder = jnp.sqrt(jnp.sum(jnp.abs(next_term) ** 2))
    return transfer, remainder


def _prepare_at_doublings(
    layer: PreparedLayerOperator,
    thickness: Array,
    policy: BoundaryCascadePolicy,
    doublings: int,
    /,
) -> BoundaryRelation:
    transfer, remainder = _taylor_transfer(
        layer.matrix,
        thickness,
        doublings,
        policy.initializer_order,
    )
    relation = _transfer_to_boundary(transfer)
    diagnostics = BoundaryRelationDiagnostics(
        relation.diagnostics.solve_residual,
        remainder,
        relation.diagnostics.paired_error,
        relation.diagnostics.finite,
        relation.diagnostics.converged,
    )
    relation = BoundaryRelation(
        relation.a, relation.b, relation.c, relation.d, diagnostics
    )
    for _ in range(doublings):
        relation = compose_boundary_relations(relation, relation)
    return relation


def _boundary_difference(left: BoundaryRelation, right: BoundaryRelation) -> Array:
    numerator = jnp.sqrt(
        jnp.sum(jnp.abs(left.a - right.a) ** 2)
        + jnp.sum(jnp.abs(left.b - right.b) ** 2)
        + jnp.sum(jnp.abs(left.c - right.c) ** 2)
        + jnp.sum(jnp.abs(left.d - right.d) ** 2)
    )
    denominator = jnp.maximum(
        jnp.sqrt(
            jnp.sum(jnp.abs(right.a) ** 2)
            + jnp.sum(jnp.abs(right.b) ** 2)
            + jnp.sum(jnp.abs(right.c) ** 2)
            + jnp.sum(jnp.abs(right.d) ** 2)
        ),
        1.0,
    )
    return numerator / denominator


def prepare_layer_boundary(
    layer: PreparedLayerOperator,
    thickness: ArrayLike,
    policy: BoundaryCascadePolicy,
    /,
) -> BoundaryRelation:
    value = jnp.asarray(thickness, dtype=layer.matrix.dtype)
    if value.ndim > 0:
        raise ValueError("thickness must be scalar.")
    primary = _prepare_at_doublings(layer, value, policy, policy.doublings)
    paired_error = jnp.asarray(0.0, dtype=layer.matrix.real.dtype)
    if policy.paired_error:
        refined = _prepare_at_doublings(layer, value, policy, policy.doublings + 1)
        paired_error = _boundary_difference(primary, refined)
        primary = refined
    tolerance = policy.absolute_tolerance + policy.relative_tolerance
    converged = (
        primary.diagnostics.finite
        & (primary.diagnostics.initializer_remainder <= tolerance)
        & ((paired_error <= tolerance) if policy.paired_error else jnp.asarray(True))
    )
    diagnostics = BoundaryRelationDiagnostics(
        primary.diagnostics.solve_residual,
        primary.diagnostics.initializer_remainder,
        paired_error,
        primary.diagnostics.finite,
        converged,
    )
    return BoundaryRelation(primary.a, primary.b, primary.c, primary.d, diagnostics)


__all__ = [
    "BoundaryCascadePolicy",
    "BoundaryRelation",
    "BoundaryRelationDiagnostics",
    "compose_boundary_relations",
    "identity_boundary_relation",
    "prepare_layer_boundary",
]
