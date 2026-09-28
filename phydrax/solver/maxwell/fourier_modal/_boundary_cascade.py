#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

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
    ) -> None:
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
    """Power-wave scattering relation of a slab between two boundary planes.

    At each plane the tangential harmonics ``e = [Eₓ, Eᵧ]`` and ``h = [Hₓ, Hᵧ]`` split
    into forward and backward power waves ``f = (e + Jh)/2`` and ``g = (e − Jh)/2``,
    with ``Jh = [Hᵧ, −Hₓ]`` and the unit reference admittance of the relative
    constitutive units. ``|f|² − |g|²`` is the harmonic sum ``Re(eᴴJh)`` of +z Poynting
    flux. Inputs ``[f_left, g_right]`` map to outputs ``[f_right, g_left]``:
    ``f_right = s11 f_left + s12 g_right`` and ``g_left = s21 f_left + s22 g_right``.

    A passive slab with a real Bloch wavevector has a contractive relation, so
    composition never divides by exponentially small evanescent transmission and a
    cavity resonance of one sub-slab never makes the cascade singular.
    """

    s11: Array
    s12: Array
    s21: Array
    s22: Array
    diagnostics: BoundaryRelationDiagnostics

    @property
    def tangential_size(self) -> int:
        return self.s11.shape[-1]


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


def _rotate_magnetic(magnetic: Array, /) -> Array:
    """Return ``Jh = [Hᵧ, −Hₓ]`` along the leading tangential axis."""
    count = magnetic.shape[0] // 2
    return jnp.concatenate((magnetic[count:], -magnetic[:count]), axis=0)


def _fields_to_waves(electric: Array, magnetic: Array, /) -> tuple[Array, Array]:
    """Forward and backward power waves of tangential harmonic fields."""
    rotated = _rotate_magnetic(magnetic)
    return 0.5 * (electric + rotated), 0.5 * (electric - rotated)


def _waves_to_fields(forward: Array, backward: Array, /) -> tuple[Array, Array]:
    """Tangential harmonic fields of forward and backward power waves."""
    rotated = forward - backward
    count = rotated.shape[0] // 2
    magnetic = jnp.concatenate((-rotated[count:], rotated[:count]), axis=0)
    return forward + backward, magnetic


def _matrix_relative_residual(matrix: Array, solution: Array, rhs: Array) -> Array:
    residual = matrix @ solution - rhs
    denominator = jnp.maximum(jnp.sqrt(jnp.sum(jnp.abs(rhs) ** 2)), 1.0)
    return jnp.sqrt(jnp.sum(jnp.abs(residual) ** 2)) / denominator


def _transfer_to_boundary(transfer: Array, /) -> BoundaryRelation:
    """Convert a short field transfer ``[e, h](0) ↦ [e, h](L)`` to power waves."""
    size = transfer.shape[0] // 2
    forward_rows, backward_rows = _fields_to_waves(transfer[:size], transfer[size:])
    # Columns: the field transfer applied to the fields of unit forward/backward waves.
    identity = jnp.eye(size, dtype=transfer.dtype)
    zero = jnp.zeros_like(identity)
    forward_fields = jnp.concatenate(_waves_to_fields(identity, zero), axis=0)
    backward_fields = jnp.concatenate(_waves_to_fields(zero, identity), axis=0)
    t11 = forward_rows @ forward_fields
    t12 = forward_rows @ backward_fields
    t21 = backward_rows @ forward_fields
    t22 = backward_rows @ backward_fields
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
    """Redheffer star product of adjacent left and right power-wave relations."""
    if left.s11.shape != right.s11.shape:
        raise ValueError("Boundary relations must act on the same tangential space.")
    size = left.tangential_size
    identity = jnp.eye(size, dtype=left.s11.dtype)
    system = identity - left.s12 @ right.s21
    rhs = jnp.concatenate((left.s11, left.s12 @ right.s22), axis=1)
    middle = _dense_solve(system, rhs)
    from_left = middle[:, :size]
    from_right = middle[:, size:]
    s11 = right.s11 @ from_left
    s12 = right.s11 @ from_right + right.s12
    s21 = left.s21 + left.s22 @ right.s21 @ from_left
    s22 = left.s22 @ (right.s21 @ from_right + right.s22)
    solve_residual = jnp.maximum(
        jnp.maximum(left.diagnostics.solve_residual, right.diagnostics.solve_residual),
        _matrix_relative_residual(system, middle, rhs),
    )
    initializer_remainder = (
        left.diagnostics.initializer_remainder + right.diagnostics.initializer_remainder
    )
    paired_error = left.diagnostics.paired_error + right.diagnostics.paired_error
    finite = (
        left.diagnostics.finite
        & right.diagnostics.finite
        & jnp.all(jnp.isfinite(s11))
        & jnp.all(jnp.isfinite(s12))
        & jnp.all(jnp.isfinite(s21))
        & jnp.all(jnp.isfinite(s22))
    )
    converged = left.diagnostics.converged & right.diagnostics.converged & finite
    return BoundaryRelation(
        s11,
        s12,
        s21,
        s22,
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
        relation.s11, relation.s12, relation.s21, relation.s22, diagnostics
    )
    for _ in range(doublings):
        relation = compose_boundary_relations(relation, relation)
    return relation


def _boundary_difference(left: BoundaryRelation, right: BoundaryRelation) -> Array:
    numerator = jnp.sqrt(
        jnp.sum(jnp.abs(left.s11 - right.s11) ** 2)
        + jnp.sum(jnp.abs(left.s12 - right.s12) ** 2)
        + jnp.sum(jnp.abs(left.s21 - right.s21) ** 2)
        + jnp.sum(jnp.abs(left.s22 - right.s22) ** 2)
    )
    denominator = jnp.maximum(
        jnp.sqrt(
            jnp.sum(jnp.abs(right.s11) ** 2)
            + jnp.sum(jnp.abs(right.s12) ** 2)
            + jnp.sum(jnp.abs(right.s21) ** 2)
            + jnp.sum(jnp.abs(right.s22) ** 2)
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
    return BoundaryRelation(
        primary.s11, primary.s12, primary.s21, primary.s22, diagnostics
    )


__all__ = [
    "BoundaryCascadePolicy",
    "BoundaryRelation",
    "BoundaryRelationDiagnostics",
    "compose_boundary_relations",
    "identity_boundary_relation",
    "prepare_layer_boundary",
]
