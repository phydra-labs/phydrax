#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Sectional population balances with growth, aggregation, and breakage."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..qualification import CapabilityProfile, SupportTuple


class SectionalPopulationPlan(StrictModule, NonTrainableState):
    edges: Array
    centers: Array
    widths: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, edges: ArrayLike, /):
        values = np.asarray(edges, dtype=float)
        if (
            values.ndim != 1
            or values.size < 3
            or np.any(np.diff(values) <= 0.0)
            or values[0] < 0.0
        ):
            raise ValueError(
                "Population edges must be non-negative and strictly increasing."
            )
        self.edges = jnp.asarray(values)
        self.centers = jnp.asarray(0.5 * (values[:-1] + values[1:]))
        self.widths = jnp.asarray(np.diff(values))
        self.plan_id = canonical_fingerprint(
            {"kind": "sectional-population-plan", "edges": values.tolist()}
        )

    def moments(self, number_density: ArrayLike, orders: ArrayLike, /) -> Array:
        density = jnp.asarray(number_density)
        powers = jnp.asarray(orders)
        if density.shape[-1] != self.centers.size:
            raise ValueError("Population density must end with the sectional axis.")
        return jnp.sum(
            density[..., None, :] * self.centers ** powers[..., None] * self.widths,
            axis=-1,
        )

    def growth_rate(self, number_density: ArrayLike, growth: ArrayLike, /) -> Array:
        density = jnp.asarray(number_density)
        velocity = jnp.asarray(growth)
        if density.shape != velocity.shape or density.shape[-1] != self.centers.size:
            raise ValueError("Growth and density arrays must align with sections.")
        face_velocity = jnp.concatenate(
            (
                velocity[..., :1],
                0.5 * (velocity[..., :-1] + velocity[..., 1:]),
                velocity[..., -1:],
            ),
            axis=-1,
        )
        face_density = jnp.concatenate(
            (
                density[..., :1],
                jnp.where(
                    face_velocity[..., 1:-1] >= 0.0, density[..., :-1], density[..., 1:]
                ),
                density[..., -1:],
            ),
            axis=-1,
        )
        flux = face_velocity * face_density
        return -jnp.diff(flux, axis=-1) / self.widths

    def aggregation_rate(self, number_density: ArrayLike, kernel: ArrayLike, /) -> Array:
        density = jnp.asarray(number_density)
        coefficients = jnp.asarray(kernel)
        bins = int(self.centers.size)
        if density.shape != (bins,) or coefficients.shape != (bins, bins):
            raise ValueError(
                "This bounded aggregation route requires one sectional vector and square kernel."
            )
        loss = density * (coefficients @ density)
        birth = jnp.zeros_like(density)
        for left in range(bins):
            for right in range(bins):
                product_size = self.centers[left] + self.centers[right]
                target = jnp.clip(
                    jnp.searchsorted(self.edges, product_size) - 1, 0, bins - 1
                )
                contribution = (
                    0.5 * coefficients[left, right] * density[left] * density[right]
                )
                birth = birth.at[target].add(contribution)
        return birth - loss

    def breakage_rate(
        self, number_density: ArrayLike, rate: ArrayLike, daughter_matrix: ArrayLike, /
    ) -> Array:
        density = jnp.asarray(number_density)
        rate_ = jnp.asarray(rate)
        daughters = jnp.asarray(daughter_matrix)
        bins = int(self.centers.size)
        if (
            density.shape != (bins,)
            or rate_.shape != (bins,)
            or daughters.shape != (bins, bins)
        ):
            raise ValueError("Breakage arrays do not match the sectional plan.")
        return daughters @ (rate_ * density) - rate_ * density

    def realizable(self, number_density: ArrayLike, /) -> Array:
        density = jnp.asarray(number_density)
        return jnp.all(jnp.isfinite(density)) & jnp.all(density >= 0.0)


def population_balance_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    specs = (
        (
            "population-balance.sectional-growth",
            {"coordinates": "one", "method": "upwind-sectional"},
        ),
        (
            "population-balance.aggregation",
            {"coordinates": "one", "method": "fixed-pivot-nearest-bin"},
        ),
        (
            "population-balance.breakage",
            {"coordinates": "one", "method": "daughter-matrix"},
        ),
    )
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(name, attrs),),
            required_gates=("moment-balance", "positivity", "refinement"),
        )
        for name, attrs in specs
    )


__all__ = ["SectionalPopulationPlan", "population_balance_candidate_profiles"]
