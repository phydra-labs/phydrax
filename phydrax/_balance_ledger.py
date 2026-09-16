#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._strict import StrictModule


class BalanceTerm(StrictModule):
    name: str = eqx.field(static=True)
    value: Array
    sign: int = eqx.field(static=True)
    owner_id: str = eqx.field(static=True)
    unit_id: str = eqx.field(static=True)

    def __init__(
        self, name: str, value: ArrayLike, sign: int, owner_id: str, unit_id: str, /
    ):
        if sign not in (-1, 1):
            raise ValueError("BalanceTerm sign must be -1 or 1.")
        if not name or not owner_id or not unit_id:
            raise ValueError("BalanceTerm identities must be non-empty.")
        self.name = str(name)
        self.value = jnp.asarray(value)
        self.sign = int(sign)
        self.owner_id = str(owner_id)
        self.unit_id = str(unit_id)


class BalanceEvidence(StrictModule):
    residual: Array
    scale: Array
    relative_residual: Array
    finite: Array
    successful: Array
    quantity_id: str = eqx.field(static=True)
    unit_id: str = eqx.field(static=True)


def evaluate_balance(
    quantity_id: str,
    storage_change: ArrayLike,
    terms: Sequence[BalanceTerm],
    /,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> BalanceEvidence:
    terms_ = tuple(terms)
    if not quantity_id or not terms_:
        raise ValueError("A balance requires a quantity identity and at least one term.")
    units = {term.unit_id for term in terms_}
    if len(units) != 1:
        raise ValueError("Every balance term must use one common unit.")
    owners = [(term.owner_id, term.name) for term in terms_]
    if len(set(owners)) != len(owners):
        raise ValueError("A balance cannot contain duplicate owner/name terms.")
    residual = jnp.asarray(storage_change)
    scale = jnp.maximum(jnp.abs(residual), 1.0)
    for term in terms_:
        residual = residual + term.sign * term.value
        scale = jnp.maximum(scale, jnp.abs(term.value))
    relative = jnp.abs(residual) / scale
    finite = jnp.all(jnp.isfinite(residual)) & jnp.all(jnp.isfinite(scale))
    successful = finite & jnp.all(
        jnp.abs(residual) <= absolute_tolerance + relative_tolerance * scale
    )
    return BalanceEvidence(
        residual,
        scale,
        relative,
        finite,
        successful,
        str(quantity_id),
        next(iter(units)),
    )


__all__ = ["BalanceEvidence", "BalanceTerm", "evaluate_balance"]
