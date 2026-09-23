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


class SectionalAggregationRate(StrictModule):
    cell_number_density_rate: Array
    overflow_number_rate: Array
    overflow_first_moment_rate: Array
    first_moment_residual: Array


class SectionalPopulationPlan(StrictModule, NonTrainableState):
    edges: Array
    centers: Array
    widths: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, edges: ArrayLike, /):
        values = np.asarray(edges, dtype=np.float64)
        if (
            values.ndim != 1
            or values.size < 3
            or not np.all(np.isfinite(values))
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
        density = eqx.error_if(
            density,
            jnp.any(~jnp.isfinite(density) | (density < 0)),
            "Population density must be finite and nonnegative.",
        )
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

    def aggregation_rate(
        self, number_density: ArrayLike, kernel: ArrayLike, /
    ) -> SectionalAggregationRate:
        density = jnp.asarray(number_density)
        coefficients = jnp.asarray(kernel)
        bins = self.centers.size
        if density.shape != (bins,) or coefficients.shape != (bins, bins):
            raise ValueError(
                "This bounded aggregation route requires one sectional vector and square kernel."
            )
        density = eqx.error_if(
            density,
            jnp.any(
                ~jnp.isfinite(density)
                | ~jnp.isfinite(coefficients)
                | (density < 0)
                | (coefficients < 0)
            ),
            "Aggregation density and kernel must be finite and nonnegative.",
        )
        density = eqx.error_if(
            density,
            ~jnp.allclose(coefficients, jnp.swapaxes(coefficients, -1, -2)),
            "Aggregation kernel must be symmetric for moment conservation.",
        )
        number = density * self.widths
        loss_number = number * (coefficients @ number)
        birth_number = jnp.zeros_like(number)
        overflow_number = jnp.asarray(0.0, dtype=number.dtype)
        overflow_moment = jnp.asarray(0.0, dtype=number.dtype)
        for left in range(bins):
            for right in range(bins):
                product_size = self.centers[left] + self.centers[right]
                event_rate = (
                    0.5 * coefficients[left, right] * number[left] * number[right]
                )
                is_overflow = product_size > self.centers[-1]
                upper = jnp.clip(
                    jnp.searchsorted(self.centers, product_size, side="left"),
                    1,
                    bins - 1,
                )
                lower = upper - 1
                denominator = self.centers[upper] - self.centers[lower]
                upper_weight = (product_size - self.centers[lower]) / denominator
                upper_weight = jnp.clip(upper_weight, 0.0, 1.0)
                bounded = jnp.where(is_overflow, 0.0, event_rate)
                birth_number = birth_number.at[lower].add((1.0 - upper_weight) * bounded)
                birth_number = birth_number.at[upper].add(upper_weight * bounded)
                overflow_number = overflow_number + jnp.where(
                    is_overflow, event_rate, 0.0
                )
                overflow_moment = overflow_moment + jnp.where(
                    is_overflow, product_size * event_rate, 0.0
                )
        cell_number_rate = birth_number - loss_number
        first_residual = jnp.sum(self.centers * cell_number_rate) + overflow_moment
        return SectionalAggregationRate(
            cell_number_rate / self.widths,
            overflow_number,
            overflow_moment,
            first_residual,
        )

    def breakage_rate(
        self, number_density: ArrayLike, rate: ArrayLike, daughter_matrix: ArrayLike, /
    ) -> Array:
        density = jnp.asarray(number_density)
        rate_ = jnp.asarray(rate)
        daughters = jnp.asarray(daughter_matrix)
        bins = self.centers.size
        if (
            density.shape != (bins,)
            or rate_.shape != (bins,)
            or daughters.shape != (bins, bins)
        ):
            raise ValueError("Breakage arrays do not match the sectional plan.")
        density = eqx.error_if(
            density,
            jnp.any(
                ~jnp.isfinite(density)
                | ~jnp.isfinite(rate_)
                | ~jnp.isfinite(daughters)
                | (density < 0)
                | (rate_ < 0)
                | (daughters < 0)
            ),
            "Breakage data must be finite and nonnegative.",
        )
        daughter_moment = self.centers @ daughters
        density = eqx.error_if(
            density,
            jnp.any(
                (rate_ > 0)
                & (
                    jnp.abs(daughter_moment - self.centers)
                    > 1e-10 * jnp.maximum(self.centers, 1.0)
                )
            ),
            "Active daughter distributions must conserve first moment.",
        )
        return daughters @ (rate_ * density) - rate_ * density

    def realizable(self, number_density: ArrayLike, /) -> Array:
        density = jnp.asarray(number_density)
        return density.shape == self.centers.shape and jnp.all(
            jnp.isfinite(density)
        ) & jnp.all(density >= 0.0)


def population_balance_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    specs = (
        (
            "population-balance.sectional-growth",
            {"coordinates": "one", "method": "upwind-sectional"},
        ),
        (
            "population-balance.aggregation",
            {"coordinates": "one", "method": "fixed-pivot-moment-preserving"},
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


__all__ = [
    "SectionalAggregationRate",
    "SectionalPopulationPlan",
    "population_balance_candidate_profiles",
]
