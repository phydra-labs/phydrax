#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite bosonic/fermionic exchange sectors for periodic many-body paths."""

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


class ExchangePathPlan(StrictModule):
    """Fixed-capacity permutation sector with explicit completeness evidence."""

    permutations: Array
    active: Array
    parity: Array
    particle_count: int = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    statistics: Literal["boson", "fermion"] = eqx.field(static=True)
    sector: Literal["full-enumeration", "restricted"] = eqx.field(static=True)
    claim: str = eqx.field(static=True)

    def __init__(
        self,
        permutations: ArrayLike,
        /,
        *,
        statistics: Literal["boson", "fermion"],
        active: ArrayLike | None = None,
        require_full_enumeration: bool = False,
    ):
        table = np.asarray(permutations)
        if (
            table.ndim != 2
            or min(table.shape) < 1
            or not np.issubdtype(table.dtype, np.integer)
        ):
            raise ValueError(
                "permutations must be a nonempty integer (capacity, particles) table."
            )
        capacity, particles = map(int, table.shape)
        mask = (
            np.ones((capacity,), dtype=bool)
            if active is None
            else np.asarray(active, dtype=bool)
        )
        if mask.shape != (capacity,) or not np.any(mask):
            raise ValueError("active must select at least one permutation row.")
        reference = np.arange(particles)
        for row in table[mask]:
            if not np.array_equal(np.sort(row), reference):
                raise ValueError(
                    "Every active row must be a permutation of all particles."
                )
        active_rows = table[mask]
        unique_count = int(np.unique(active_rows, axis=0).shape[0])
        if unique_count != int(mask.sum()):
            raise ValueError("Active permutation rows must be unique.")
        full = unique_count == math.factorial(particles)
        if require_full_enumeration and not full:
            raise ValueError(
                "The active table does not cover the complete exchange group."
            )
        if statistics not in ("boson", "fermion"):
            raise ValueError("statistics must be 'boson' or 'fermion'.")
        parity = np.ones((capacity,), dtype=np.int8)
        for index, row in enumerate(table):
            inversions = sum(
                int(row[left] > row[right])
                for left in range(particles)
                for right in range(left + 1, particles)
            )
            parity[index] = -1 if inversions % 2 else 1
        parity[~mask] = 0
        self.permutations = jnp.asarray(table, dtype=jnp.int32)
        self.active = jnp.asarray(mask)
        self.parity = jnp.asarray(parity)
        self.particle_count = particles
        self.capacity = capacity
        self.statistics = statistics
        self.sector = "full-enumeration" if full else "restricted"
        self.claim = "finite-exchange-sector"

    @property
    def characters(self) -> Array:
        if self.statistics == "boson":
            return self.active.astype(jnp.int8)
        return self.parity


def exchange_path_action(
    paths: ArrayLike,
    permutation: ArrayLike,
    /,
    *,
    inverse_temperature: float,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> Array:
    """Ring kinetic action closed through one declared particle permutation."""
    q = jnp.asarray(paths)
    permutation_ = jnp.asarray(permutation, dtype=jnp.int32)
    if q.ndim < 3:
        raise ValueError(
            "paths must have trailing shape (particles, beads, state_dimension)."
        )
    particles, beads, dimension = map(int, q.shape[-3:])
    if particles < 1 or beads < 2 or dimension < 1 or permutation_.shape != (particles,):
        raise ValueError("Invalid path/permutation shape.")
    beta, mass_, hbar_ = float(inverse_temperature), float(mass), float(hbar)
    if any(not np.isfinite(value) or value <= 0.0 for value in (beta, mass_, hbar_)):
        raise ValueError(
            "inverse_temperature, mass, and hbar must be finite and positive."
        )
    internal = q[..., :, 1:, :] - q[..., :, :-1, :]
    closure = q[..., permutation_, 0, :] - q[..., :, -1, :]
    squared = jnp.sum(internal * internal, axis=(-3, -2, -1)) + jnp.sum(
        closure * closure, axis=(-2, -1)
    )
    step = beta / beads
    return mass_ * squared / (2.0 * hbar_**2 * step)


class ExchangePathEstimate(StrictModule):
    value: Array
    standard_error: Array
    average_sign: Array
    signed_effective_sample_size: Array
    valid: Array
    unresolved_sign_problem: Array
    statistics: str = eqx.field(static=True)
    sector: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def estimate_exchange_observable(
    observable: ArrayLike,
    log_absolute_weight: ArrayLike,
    character: ArrayLike,
    /,
    *,
    plan: ExchangePathPlan,
    minimum_average_sign: float = 1e-3,
) -> ExchangePathEstimate:
    """Signed reweighting for a finite admitted exchange sector."""
    values = jnp.asarray(observable)
    logs = jnp.asarray(log_absolute_weight)
    signs = jnp.asarray(character)
    if values.shape != logs.shape or values.shape != signs.shape or values.ndim < 1:
        raise ValueError(
            "observable, log_absolute_weight, and character shapes must match."
        )
    active = signs != 0
    count = jnp.sum(active)
    maximum = jnp.max(jnp.where(active, logs, -jnp.inf))
    maximum = jnp.where(count > 0, maximum, 0.0)
    absolute = jnp.where(active, jnp.exp(logs - maximum), 0.0)
    signed = absolute * signs
    denominator = jnp.sum(signed)
    numerator = jnp.sum(signed * jnp.where(active, values, 0.0))
    value = numerator / denominator
    average_sign = denominator / jnp.sum(absolute)
    signed_ess = denominator**2 / jnp.sum(absolute**2)
    contributions = jnp.where(active, signed * (values - value), 0.0)
    standard_error = (
        jnp.sqrt(jnp.sum(jnp.abs(contributions) ** 2))
        / jnp.maximum(jnp.abs(denominator), jnp.finfo(absolute.dtype).tiny)
        * jnp.sqrt(count / (count - 1))
    )
    standard_error = jnp.where(count >= 2, standard_error, jnp.nan)
    finite_samples = (
        jnp.all(jnp.where(active, jnp.isfinite(values), True))
        & jnp.all(jnp.where(active, jnp.isfinite(logs), True))
        & jnp.all(jnp.where(active, jnp.isfinite(signs), True))
    )
    valid = finite_samples & (count >= 2) & (denominator != 0.0)
    return ExchangePathEstimate(
        value=value,
        standard_error=standard_error,
        average_sign=average_sign,
        signed_effective_sample_size=signed_ess,
        valid=valid,
        unresolved_sign_problem=jnp.abs(average_sign) < minimum_average_sign,
        statistics=plan.statistics,
        sector=plan.sector,
        claim="finite-sector-signed-reweighting-no-sign-problem-cure",
    )


__all__ = [
    "ExchangePathEstimate",
    "ExchangePathPlan",
    "estimate_exchange_observable",
    "exchange_path_action",
]
