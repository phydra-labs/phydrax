#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class AngularQuadratureEvidence(StrictModule):
    zeroth_moment_residual: Array
    first_moment_residual: Array
    second_moment_residual: Array
    symmetric: Array
    positive: Array
    successful: Array
    quadrature_id: str = eqx.field(static=True)


class CertifiedSlabAngularQuadrature(StrictModule, NonTrainableState):
    ordinates: Array
    weights: Array
    opposite_indices: Array
    evidence: AngularQuadratureEvidence
    quadrature_id: str = eqx.field(static=True)

    def __init__(
        self,
        ordinates: ArrayLike,
        weights: ArrayLike,
        /,
        *,
        tolerance: float = 1.0e-12,
        quadrature_id: str | None = None,
    ):
        mu = np.asarray(ordinates, dtype=np.float64)
        weight = np.asarray(weights, dtype=np.float64)
        tolerance_ = float(tolerance)
        if (
            mu.ndim != 1
            or mu.size < 2
            or weight.shape != mu.shape
            or np.any(~np.isfinite(mu))
            or np.any(np.abs(mu) > 1.0)
            or np.any(mu == 0.0)
            or np.any(~np.isfinite(weight))
            or np.any(weight <= 0.0)
            or not np.isfinite(tolerance_)
            or tolerance_ <= 0.0
        ):
            raise ValueError("Slab angular ordinates, weights, or tolerance are invalid.")
        opposite = []
        for value in mu:
            candidates = np.flatnonzero(np.abs(mu + value) <= tolerance_)
            if candidates.size != 1:
                raise ValueError("Slab quadrature must contain one opposite ordinate.")
            opposite.append(int(candidates[0]))
        opposite_ = np.asarray(opposite, dtype=np.int32)
        zero = abs(float(np.sum(weight)) - 2.0)
        first = abs(float(np.sum(weight * mu)))
        second = abs(float(np.sum(weight * mu**2)) - 2.0 / 3.0)
        symmetric = np.array_equal(
            opposite_[opposite_], np.arange(mu.size)
        ) and np.allclose(weight, weight[opposite_], rtol=0.0, atol=tolerance_)
        positive = bool(np.all(weight > 0.0))
        passed = positive and symmetric and max(zero, first, second) <= tolerance_
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "certified-slab-angular-quadrature",
                    "ordinates": array_tree_fingerprint(mu),
                    "weights": array_tree_fingerprint(weight),
                    "tolerance": tolerance_,
                }
            )
            if quadrature_id is None
            else str(quadrature_id)
        )
        if not identifier or not passed:
            raise ValueError("Slab angular quadrature failed moment certification.")
        self.ordinates = jnp.asarray(mu)
        self.weights = jnp.asarray(weight)
        self.opposite_indices = jnp.asarray(opposite_)
        self.quadrature_id = identifier
        self.evidence = AngularQuadratureEvidence(
            jnp.asarray(zero),
            jnp.asarray(first),
            jnp.asarray(second),
            jnp.asarray(symmetric),
            jnp.asarray(positive),
            jnp.asarray(passed),
            identifier,
        )

    @classmethod
    def gauss_legendre(
        cls, order: int, /, *, tolerance: float = 1.0e-12
    ) -> "CertifiedSlabAngularQuadrature":
        count = int(order)
        if count < 2 or count % 2:
            raise ValueError("Slab Gauss-Legendre order must be positive and even.")
        ordinates, weights = np.polynomial.legendre.leggauss(count)
        return cls(ordinates, weights, tolerance=tolerance)

    @property
    def angle_count(self) -> int:
        return self.ordinates.size


__all__ = ["AngularQuadratureEvidence", "CertifiedSlabAngularQuadrature"]
