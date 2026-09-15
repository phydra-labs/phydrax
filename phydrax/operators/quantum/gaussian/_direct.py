#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Integral-direct Coulomb and exchange contractions over fixed screened quartets."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....ein import contract
from ._basis import PreparedGaussianBasis
from ._integrals import contracted_electron_repulsion_element
from ._screening import GaussianScreeningPlan, PreparedGaussianScreening


class DirectJKResult(StrictModule):
    coulomb: Array
    exchange: Array
    retained_quartets: Array
    omitted_quartets: Array
    omitted_contribution_bound: Array
    geometry_valid: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class DirectJKPlan(StrictModule, NonTrainableState):
    screening: GaussianScreeningPlan
    plan_id: str = eqx.field(static=True)

    def __init__(self, screening: GaussianScreeningPlan | None = None, /):
        screening_ = GaussianScreeningPlan() if screening is None else screening
        if not isinstance(screening_, GaussianScreeningPlan):
            raise TypeError("screening must be GaussianScreeningPlan or None.")
        self.screening = screening_
        self.plan_id = canonical_fingerprint(
            {"kind": "direct-jk-plan", "screening": screening_.plan_id}
        )

    def prepare(
        self,
        basis: PreparedGaussianBasis,
        positions: ArrayLike,
        /,
    ) -> PreparedDirectJK:
        return PreparedDirectJK(self, basis, self.screening.prepare(basis, positions))


class PreparedDirectJK(StrictModule, NonTrainableState):
    plan: DirectJKPlan
    basis: PreparedGaussianBasis
    screening: PreparedGaussianScreening
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: DirectJKPlan,
        basis: PreparedGaussianBasis,
        screening: PreparedGaussianScreening,
        /,
    ):
        if screening.basis_id != basis.prepared_id:
            raise ValueError("Direct J/K screening belongs to another Gaussian basis.")
        self.plan = plan
        self.basis = basis
        self.screening = screening
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-direct-jk",
                "plan": plan.plan_id,
                "basis": basis.prepared_id,
                "screening": screening.prepared_id,
            }
        )

    def evaluate(
        self,
        positions: ArrayLike,
        density: ArrayLike,
        /,
    ) -> DirectJKResult:
        coordinate = jnp.asarray(positions)
        density_ = jnp.asarray(density, dtype=coordinate.dtype)
        output_count = self.basis.basis_function_count
        if density_.shape != (output_count, output_count):
            raise ValueError(
                "Density matrix must align with the requested Gaussian basis."
            )
        transform = self.basis.transformation
        cartesian_density = contract("pa,ab,qb->pq", transform, density_, transform)
        count = self.basis.cartesian_basis_function_count
        coulomb = jnp.zeros((count, count), dtype=coordinate.dtype)
        exchange = jnp.zeros((count, count), dtype=coordinate.dtype)
        for a, b, c, d in self.screening.quartets:
            integral = contracted_electron_repulsion_element(
                self.basis, coordinate, a, b, c, d
            )
            coulomb = coulomb.at[a, b].add(cartesian_density[c, d] * integral)
            exchange = exchange.at[a, c].add(cartesian_density[b, d] * integral)
        geometry_valid = self.screening.geometry_valid(coordinate)
        finite = jnp.all(jnp.isfinite(coulomb)) & jnp.all(jnp.isfinite(exchange))
        omitted_bound = (
            jnp.max(jnp.abs(cartesian_density), initial=0.0)
            * self.screening.omitted_bound_sum
        )
        return DirectJKResult(
            self.basis.transform_one_body(coulomb),
            self.basis.transform_one_body(exchange),
            jnp.asarray(self.screening.retained_quartet_count, dtype=jnp.int64),
            jnp.asarray(
                self.screening.total_quartet_count
                - self.screening.retained_quartet_count,
                dtype=jnp.int64,
            ),
            omitted_bound,
            geometry_valid,
            geometry_valid & finite,
            self.prepared_id,
        )


__all__ = ["DirectJKPlan", "DirectJKResult", "PreparedDirectJK"]
