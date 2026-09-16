#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native arbitrary-quadrature zero-spin SL(2,C) B4 booster reference."""

from __future__ import annotations

from math import pi

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule


class SL2CBoosterReferencePlan(StrictModule):
    """Guarded EPRL B4 reference for j_a=l_a=i=k=0 only."""

    radial_nodes: Array
    radial_weights: Array
    radial_cutoff: float = eqx.field(static=True)
    quadrature_order: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        radial_cutoff: float = 16.0,
        quadrature_order: int = 256,
        tolerance: float = 1e-10,
    ):
        cutoff = float(radial_cutoff)
        order = int(quadrature_order)
        tolerance_value = float(tolerance)
        if (
            not np.isfinite(cutoff)
            or not 2.0 <= cutoff <= 30.0
            or order < 16
            or order > 4096
            or not np.isfinite(tolerance_value)
            or tolerance_value < 0.0
        ):
            raise ValueError("SL2C booster quadrature parameters are invalid.")
        canonical_nodes, canonical_weights = np.polynomial.legendre.leggauss(order)
        nodes = 0.5 * cutoff * (canonical_nodes + 1.0)
        weights = 0.5 * cutoff * canonical_weights
        self.radial_nodes = jnp.asarray(nodes)
        self.radial_weights = jnp.asarray(weights)
        self.radial_cutoff = cutoff
        self.quadrature_order = order
        self.tolerance = tolerance_value
        self.plan_id = canonical_fingerprint(
            {
                "kind": "zero-spin-sl2c-b4-booster-reference-plan",
                "radial_cutoff": cutoff,
                "quadrature_order": order,
                "nodes": array_tree_fingerprint(nodes),
                "weights": array_tree_fingerprint(weights),
                "tolerance": tolerance_value,
                "matrix_element": "d_(rho=0,k=0;j=0,l=0,m=0)=r/sinh(r)",
                "normalization": "one-over-four-pi",
            }
        )


class SL2CBoosterReferenceEvidence(StrictModule):
    value: Array
    exact_value: Array
    absolute_error: Array
    relative_error: Array
    cutoff_tail_upper_bound: Array
    finite: Array
    accepted: Array
    plan_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def _tail_upper_bound(cutoff: float, /) -> float:
    radius = float(cutoff)
    polynomial = radius**4 / 2.0 + radius**3 + 1.5 * radius**2 + 1.5 * radius + 0.75
    return 4.0 * np.exp(-2.0 * radius) * polynomial / (1.0 - np.exp(-2.0 * radius)) ** 2


def evaluate_zero_spin_b4_booster(
    plan: SL2CBoosterReferencePlan, /
) -> SL2CBoosterReferenceEvidence:
    """Evaluate B4(0,0;0,0) = (4π)^-1 integral r^4/sinh(r)^2 dr."""
    if not isinstance(plan, SL2CBoosterReferencePlan):
        raise TypeError("plan must be SL2CBoosterReferencePlan.")
    radius = plan.radial_nodes
    integrand = radius**4 / jnp.sinh(radius) ** 2
    value = jnp.sum(plan.radial_weights * integrand) / (4.0 * pi)
    exact = jnp.asarray(pi**3 / 120.0, dtype=value.dtype)
    error = jnp.abs(value - exact)
    relative = error / jnp.abs(exact)
    tail = jnp.asarray(_tail_upper_bound(plan.radial_cutoff) / (4.0 * pi))
    finite = jnp.isfinite(value) & jnp.isfinite(error) & jnp.isfinite(tail)
    accepted = finite & (error <= plan.tolerance + tail)
    return SL2CBoosterReferenceEvidence(
        value=value,
        exact_value=exact,
        absolute_error=error,
        relative_error=relative,
        cutoff_tail_upper_bound=tail,
        finite=finite,
        accepted=accepted,
        plan_id=plan.plan_id,
        claim="zero-spin-sl2c-b4-reference-only-no-nonzero-spin-eprl-amplitude-claim",
    )


__all__ = [
    "SL2CBoosterReferenceEvidence",
    "SL2CBoosterReferencePlan",
    "evaluate_zero_spin_b4_booster",
]
