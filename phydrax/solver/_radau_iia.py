#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import Legendre

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class RadauIIAMethod(StrictModule, NonTrainableState):
    """Immutable right-Radau collocation tableau on the unit interval."""

    nodes: Array
    matrix: Array
    weights: Array
    stage_count: int = eqx.field(static=True)
    order: int = eqx.field(static=True)
    stiffly_accurate: bool = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(self, stage_count: int, /):
        if not isinstance(stage_count, int) or isinstance(stage_count, bool):
            raise TypeError("stage_count must be an integer.")
        if stage_count < 1 or stage_count > 8:
            raise ValueError(
                "Radau IIA supports declared stage counts from one through eight."
            )
        if stage_count == 1:
            nodes = np.asarray([1.0])
        else:
            radau = Legendre.basis(stage_count) - Legendre.basis(stage_count - 1)
            roots = np.sort(np.real_if_close(radau.roots()).astype(float))
            nodes = 0.5 * (roots + 1.0)
            nodes[-1] = 1.0
        matrix = np.zeros((stage_count, stage_count), dtype=float)
        weights = np.zeros((stage_count,), dtype=float)
        for column in range(stage_count):
            basis = Polynomial([1.0])
            denominator = 1.0
            for other in range(stage_count):
                if other == column:
                    continue
                basis = basis * Polynomial([-nodes[other], 1.0])
                denominator *= nodes[column] - nodes[other]
            basis = basis / denominator
            integral = basis.integ()
            for row in range(stage_count):
                matrix[row, column] = integral(nodes[row]) - integral(0.0)
            weights[column] = integral(1.0) - integral(0.0)
        tolerance = 128.0 * np.finfo(float).eps
        stiff = bool(np.allclose(matrix[-1], weights, rtol=0.0, atol=tolerance))
        if not stiff:
            raise ValueError(
                "Constructed Radau tableau failed stiff-accuracy verification."
            )
        for degree in range(2 * stage_count - 1):
            exact = 1.0 / (degree + 1)
            represented = float(np.sum(weights * nodes**degree))
            if not np.isclose(
                represented, exact, rtol=0.0, atol=2048 * np.finfo(float).eps
            ):
                raise ValueError(
                    "Constructed Radau tableau failed polynomial-order verification."
                )
        self.nodes = jnp.asarray(nodes)
        self.matrix = jnp.asarray(matrix)
        self.weights = jnp.asarray(weights)
        self.stage_count = stage_count
        self.order = 2 * stage_count - 1
        self.stiffly_accurate = stiff
        self.method_id = canonical_fingerprint(
            {
                "kind": "radau-iia",
                "stage_count": stage_count,
                "order": self.order,
            }
        )

    @property
    def A(self) -> Array:
        return self.matrix

    @property
    def b(self) -> Array:
        return self.weights

    @property
    def c(self) -> Array:
        return self.nodes


__all__ = ["RadauIIAMethod"]
