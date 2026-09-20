#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared Schwarz screening with explicit geometry and omitted-integral bounds."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._basis import PreparedGaussianBasis
from ._integrals import contracted_electron_repulsion_element


class GaussianScreeningPlan(StrictModule, NonTrainableState):
    threshold: float = eqx.field(static=True)
    maximum_displacement: float = eqx.field(static=True)
    maximum_quartets: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        threshold: float = 0.0,
        /,
        *,
        maximum_displacement: float = 0.25,
        maximum_quartets: int = 16_777_216,
    ):
        threshold_ = float(threshold)
        displacement = float(maximum_displacement)
        capacity = int(maximum_quartets)
        if (
            not isfinite(threshold_)
            or threshold_ < 0.0
            or not isfinite(displacement)
            or displacement <= 0.0
            or capacity <= 0
        ):
            raise ValueError(
                "Gaussian screening threshold, displacement, or capacity is invalid."
            )
        self.threshold = threshold_
        self.maximum_displacement = displacement
        self.maximum_quartets = capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gaussian-screening-plan",
                "threshold": threshold_,
                "maximum_displacement": displacement,
                "maximum_quartets": capacity,
            }
        )

    def prepare(
        self,
        basis: PreparedGaussianBasis,
        positions: ArrayLike,
        /,
    ) -> PreparedGaussianScreening:
        return PreparedGaussianScreening(self, basis, positions)


class PreparedGaussianScreening(StrictModule, NonTrainableState):
    plan: GaussianScreeningPlan
    basis_id: str = eqx.field(static=True)
    reference_positions: Array
    pair_bounds: Array
    quartets: tuple[tuple[int, int, int, int], ...] = eqx.field(static=True)
    omitted_bound_sum: float = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: GaussianScreeningPlan,
        basis: PreparedGaussianBasis,
        positions: ArrayLike,
        /,
    ):
        if not isinstance(plan, GaussianScreeningPlan):
            raise TypeError("plan must be GaussianScreeningPlan.")
        if not isinstance(basis, PreparedGaussianBasis):
            raise TypeError("basis must be PreparedGaussianBasis.")
        coordinate = np.asarray(positions, dtype=np.float64)
        if (
            coordinate.ndim != 2
            or coordinate.shape[1] != 3
            or np.any(~np.isfinite(coordinate))
        ):
            raise ValueError(
                "Screening reference positions must have finite shape (N, 3)."
            )
        count = basis.cartesian_basis_function_count
        if count**4 > plan.maximum_quartets:
            raise ValueError("Gaussian screening quartet capacity is exceeded.")
        bounds = np.zeros((count, count), dtype=coordinate.dtype)
        for left in range(count):
            for right in range(count):
                diagonal = float(
                    contracted_electron_repulsion_element(
                        basis, coordinate, left, right, left, right
                    )
                )
                if not np.isfinite(diagonal) or diagonal < -1.0e-12:
                    raise ValueError(
                        "Gaussian Schwarz diagonal must be finite and non-negative."
                    )
                bounds[left, right] = np.sqrt(max(diagonal, 0.0))
        quartets: list[tuple[int, int, int, int]] = []
        omitted = 0.0
        for a in range(count):
            for b in range(count):
                for c in range(count):
                    for d in range(count):
                        bound = bounds[a, b] * bounds[c, d]
                        if bound >= plan.threshold:
                            quartets.append((a, b, c, d))
                        else:
                            omitted += bound
        self.plan = plan
        self.basis_id = basis.prepared_id
        self.reference_positions = jnp.asarray(coordinate)
        self.pair_bounds = jnp.asarray(bounds)
        self.quartets = tuple(quartets)
        self.omitted_bound_sum = float(omitted)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-gaussian-screening",
                "plan": plan.plan_id,
                "basis": basis.prepared_id,
                "quartets": [list(value) for value in quartets],
                "omitted_bound_sum": omitted,
                "arrays": array_tree_fingerprint(
                    {"reference_positions": coordinate, "pair_bounds": bounds}
                ),
            }
        )

    @property
    def retained_quartet_count(self) -> int:
        return len(self.quartets)

    @property
    def total_quartet_count(self) -> int:
        return self.pair_bounds.shape[0] ** 4

    def geometry_valid(self, positions: ArrayLike, /) -> Array:
        coordinate = jnp.asarray(positions, dtype=self.reference_positions.dtype)
        if coordinate.shape != self.reference_positions.shape:
            raise ValueError("Screening geometry shape differs from preparation.")
        displacement = jnp.sqrt(
            jnp.sum((coordinate - self.reference_positions) ** 2, axis=1)
        )
        return jnp.max(displacement, initial=0.0) <= self.plan.maximum_displacement


__all__ = ["GaussianScreeningPlan", "PreparedGaussianScreening"]
