#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Complete transformation rules of the accelerated MACE fragment coupling.

On one prepared edge fragment of ``F`` lanes, ``R`` receiver slots and ``S``
fragment-local source rows, the accelerated operation is one role of the
five-way multilinear form

    Phi(C, R, H, Y, G) = sum_l a_l sum_q C_q sum_k R[l, p(q), k]
                         H[s(l), k, i(q)] Y[l, j(q)] G[r(l), k, o(q)]

over fixed coupling rows ``q = (p, o, i, j)``, live lanes ``a_l``, local
sources ``s(l)`` and receiver slots ``r(l)``. The role named ``X`` returns
``dPhi/dX`` from the other four slots: ``"receiver"`` is the forward receiver
aggregate, ``"source"`` the source-owned reverse, ``"radial"`` and
``"harmonic"`` the lane radial/geometry cotangents and ``"coefficient"`` the
fixed-table cotangent.

Every role is linear in each input slot, and its transpose in slot ``Z`` is
the role ``Z`` with the cotangent placed in slot ``X``. The family is therefore
closed under JVP, transposition and batching: arbitrary nested derivatives
execute the same Pallas kernels and are never first-order-only rules.

The roles that reduce over lanes (``"receiver"``, ``"source"`` and
``"coefficient"``) accept a *seeded* bind: the incoming ``(high, correction)``
accumulator of their result, into which every lane event is streamed, so the
seed is added before the fragment's events rather than to a collapsed
fragment subtotal. Its value is ``high + correction``; following the
error-free ``two_sum`` derivative, the tangent is ``(d high_in + dPhi,
d correction_in)``, and its transpose returns the pair cotangent to the seed
and the ``high`` cotangent to the one linear slot.

The JVP of a role is the sum over its active slots of the same role with that
slot replaced by its tangent. For a reducing role every active slot's lane
events stream into ONE declared accumulator: the first slot's bind is seeded
with the incoming tangent pair (zero when there is none) and each further
slot's bind is seeded with the previous pair, so ``deterministic`` adds every
event in one stable order and ``compensated`` carries every ``two_sum``
residual through all slots; slot subtotals are never collapsed and then
added. Lane-owned roles (``"radial"``, ``"harmonic"``) reduce nothing across
lanes; their per-lane slot terms add elementwise.

Integer routing and lane masks are non-differentiable schedule operands. A
declared maximum derivative order is enforced when a tangent kernel is staged.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Any, get_args

import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.extend import core as jax_core
from jax.interpreters import ad, batching

from ...backends.atomistic import MACECouplingRole
from ...typing import parse


MACE_COUPLING_SCHEDULE_OPERANDS = 8
"""Fragment routing operands leading every bind, in kernel order."""
MACE_COUPLING_INPUT_SLOTS = 4
MACE_REDUCING_ROLES: tuple[MACECouplingRole, ...] = ("receiver", "source", "coefficient")
"""Roles that reduce lane events and therefore accept a seeded accumulator."""
_ROLES: tuple[MACECouplingRole, ...] = get_args(MACECouplingRole)


def coupling_input_roles(role: MACECouplingRole, /) -> tuple[MACECouplingRole, ...]:
    """Canonical ordered input slots of one coupling role."""
    role_ = parse(role, MACECouplingRole, "role")
    return tuple(slot for slot in _ROLES if slot != role_)


class MACECouplingDerivativeError(ValueError):
    """A requested transformation leaves the admitted accelerated envelope."""


def _staged_order(params: dict[str, Any], seeded: bool, /) -> dict[str, Any]:
    order = params["derivative_order"] + 1
    admitted = params["plan"].maximum_derivative_order
    if order > admitted:
        raise MACECouplingDerivativeError(
            f"The accelerated MACE coupling admits derivatives through order {admitted}; "
            f"this transformation requests order {order}. Use the native reference "
            "route for higher orders."
        )
    return {**params, "derivative_order": order, "seeded": seeded}


def _jvp(
    primitive: jax_core.Primitive,
    primals: Sequence[Array],
    tangents: Sequence[Array | ad.Zero],
    **params: Any,
) -> tuple[Array, Array | ad.Zero]:
    schedule = tuple(primals[:MACE_COUPLING_SCHEDULE_OPERANDS])
    end = MACE_COUPLING_SCHEDULE_OPERANDS + MACE_COUPLING_INPUT_SLOTS
    inputs = tuple(primals[MACE_COUPLING_SCHEDULE_OPERANDS:end])
    output = primitive.bind(*primals, **params)
    seeded: bool = params["seeded"]
    seed_tangent = tangents[end] if seeded else ad.Zero(jax.typeof(output))
    active = tuple(
        (index, tangent)
        for index, tangent in enumerate(tangents[MACE_COUPLING_SCHEDULE_OPERANDS:end])
        if not isinstance(tangent, ad.Zero)
    )
    if not active:
        if isinstance(seed_tangent, ad.Zero):
            return output, ad.Zero(jax.typeof(output))
        return output, seed_tangent

    def term(index: int, tangent: Array, seed: Array | None) -> Array:
        operands = list(inputs)
        operands[index] = tangent
        extra = () if seed is None else (seed,)
        return primitive.bind(
            *schedule, *operands, *extra, **_staged_order(params, seed is not None)
        )

    if params["role"] not in MACE_REDUCING_ROLES:
        # Lane-owned results: each lane's slot terms add elementwise.
        total = term(*active[0], None)
        for index, tangent in active[1:]:
            total = total + term(index, tangent, None)
        return output, total
    if not seeded and len(active) == 1:
        return output, term(*active[0], None)
    # One accumulator streams every active slot's lane events in order.
    pair_shape = jax.typeof(output).shape if seeded else (2, *jax.typeof(output).shape)
    carry = (
        jnp.zeros(pair_shape, dtype=output.dtype)
        if isinstance(seed_tangent, ad.Zero)
        else seed_tangent
    )
    for index, tangent in active:
        carry = term(index, tangent, carry)
    if seeded:
        return output, carry
    if params["plan"].accumulation == "compensated":
        return output, carry[0] + carry[1]
    return output, carry[0]


def _transpose(
    primitive: jax_core.Primitive,
    cotangent: Array | ad.Zero,
    *operands: Array | ad.UndefinedPrimal,
    **params: Any,
) -> tuple[Array | ad.Zero | None, ...]:
    seeded: bool = params["seeded"]
    schedule = operands[:MACE_COUPLING_SCHEDULE_OPERANDS]
    end = MACE_COUPLING_SCHEDULE_OPERANDS + MACE_COUPLING_INPUT_SLOTS
    inputs = operands[MACE_COUPLING_SCHEDULE_OPERANDS:end]
    seed = operands[end:]
    if any(ad.is_undefined_primal(value) for value in schedule):
        raise MACECouplingDerivativeError(
            "MACE fragment routing and lane masks are fixed schedule operands; they "
            "have no transpose."
        )
    undefined = tuple(
        index for index, value in enumerate(inputs) if ad.is_undefined_primal(value)
    )
    if len(undefined) != 1:
        raise MACECouplingDerivativeError(
            "A MACE coupling role is multilinear; transposition requires exactly one "
            f"linear input slot, got {len(undefined)}. A seeded accumulator is "
            "transposed only together with the tangent slot streamed into it."
        )
    linear_seed = seeded and ad.is_undefined_primal(seed[0])
    # A seeded tangent bind's linear map is (seed_h + Phi, seed_c): the seed
    # receives the pair cotangent and the linear slot the ``high`` cotangent.
    seed_cotangent: tuple[Array | ad.Zero | None, ...] = ()
    if seeded:
        if not linear_seed:
            seed_cotangent = (None,)
        elif isinstance(cotangent, ad.Zero):
            seed_cotangent = (ad.Zero(seed[0].aval),)
        else:
            seed_cotangent = (cotangent,)
    (index,) = undefined
    role: MACECouplingRole = params["role"]
    slots = coupling_input_roles(role)
    target = slots[index]
    linear = inputs[index]
    if not isinstance(linear, ad.UndefinedPrimal):
        raise RuntimeError("The transposed coupling slot lost its linear status.")
    if isinstance(cotangent, ad.Zero):
        result: Array | ad.Zero = ad.Zero(linear.aval)
    else:
        values: dict[MACECouplingRole, Array] = {}
        for slot, value in zip(slots, inputs, strict=True):
            if slot == target:
                continue
            if isinstance(value, ad.UndefinedPrimal):
                raise RuntimeError("Only one coupling slot may be linear.")
            values[slot] = value
        values[role] = cotangent[0] if seeded else cotangent
        result = primitive.bind(
            *schedule,
            *(values[slot] for slot in coupling_input_roles(target)),
            **{**params, "role": target, "seeded": False},
        )
    return (
        *(None for _ in schedule),
        *(result if position == index else None for position in range(len(inputs))),
        *seed_cotangent,
    )


def _batch(
    primitive: jax_core.Primitive,
    operands: Sequence[Array],
    axes: Sequence[int | None],
    **params: Any,
) -> tuple[Array, int | None]:
    mapped = tuple(index for index, axis in enumerate(axes) if axis is not None)
    if not mapped:
        return primitive.bind(*operands, **params), None
    moved = tuple(
        value if axis is None else jnp.moveaxis(value, axis, 0)
        for value, axis in zip(operands, axes, strict=True)
    )

    # Each batch lane is one complete kernel execution over a bounded fragment;
    # mapping sequentially keeps the per-program workspace of the admitted plan.
    def lane(values: tuple[Array, ...]) -> Array:
        current = list(moved)
        for index, value in zip(mapped, values, strict=True):
            current[index] = value
        return primitive.bind(*current, **params)

    return lax.map(lane, tuple(moved[index] for index in mapped)), 0


def register_mace_coupling_derivatives(primitive: jax_core.Primitive, /) -> None:
    """Install the closed JVP, transpose and batching rules for ``primitive``."""
    ad.primitive_jvps[primitive] = partial(_jvp, primitive)
    ad.primitive_transposes[primitive] = partial(_transpose, primitive)
    batching.primitive_batchers[primitive] = partial(_batch, primitive)


__all__ = [
    "coupling_input_roles",
    "MACE_COUPLING_INPUT_SLOTS",
    "MACE_COUPLING_SCHEDULE_OPERANDS",
    "MACE_REDUCING_ROLES",
    "MACECouplingDerivativeError",
    "register_mace_coupling_derivatives",
]
