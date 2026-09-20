#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule


class ScalarBlockPlan(StrictModule):
    """Finite radial-series plan for one-dimensional global scalar blocks."""

    cross_ratios: Array
    external_dimension: float = eqx.field(static=True)
    radial_order: int = eqx.field(static=True)
    maximum_evaluations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cross_ratios: ArrayLike,
        /,
        *,
        external_dimension: float,
        radial_order: int = 64,
        maximum_evaluations: int = 1_000_000,
    ):
        points = np.asarray(cross_ratios, dtype=np.float64)
        external = float(external_dimension)
        order = int(radial_order)
        maximum = int(maximum_evaluations)
        if points.ndim != 1 or points.size == 0:
            raise ValueError("cross_ratios must be one nonempty vector.")
        if not np.all(np.isfinite(points)) or np.any((points <= 0.0) | (points >= 1.0)):
            raise ValueError("cross_ratios must be finite and lie strictly in (0, 1).")
        if external <= 0.0 or not np.isfinite(external):
            raise ValueError("external_dimension must be finite and positive.")
        if order < 1 or maximum < 1 or points.size * order > maximum:
            raise ValueError("Scalar-block series exceeds maximum_evaluations.")
        self.cross_ratios = jnp.asarray(points)
        self.external_dimension = external
        self.radial_order = order
        self.maximum_evaluations = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-sl2-scalar-block-plan",
                "cross_ratios": array_tree_fingerprint(points),
                "external_dimension": external,
                "radial_order": order,
                "maximum_evaluations": maximum,
            }
        )


class ScalarBlockEvidence(StrictModule):
    block_values: Array
    crossing_values: Array
    final_terms: Array
    tail_proxies: Array
    finite: Array
    scaling_dimension: Array
    plan_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class PreparedScalarBlocks(StrictModule):
    """Prepared crossing prefactors separated from runtime scaling dimensions."""

    plan: ScalarBlockPlan
    direct_prefactor: Array
    reflected_prefactor: Array
    prepared_id: str = eqx.field(static=True)

    def block(self, scaling_dimension: ArrayLike, /) -> tuple[Array, Array, Array]:
        delta = jnp.asarray(scaling_dimension, dtype=self.plan.cross_ratios.dtype)
        if delta.ndim != 0:
            raise ValueError("scaling_dimension must be scalar.")
        delta = eqx.error_if(
            delta,
            ~jnp.isfinite(delta) | (delta <= 0.0),
            "Non-identity SL(2,R) scaling dimensions must be finite and positive.",
        )
        return _sl2_series(delta, self.plan.cross_ratios, self.plan.radial_order)

    def crossing_vector(self, scaling_dimension: ArrayLike, /) -> Array:
        direct, _, _ = self.block(scaling_dimension)
        reflected, _, _ = _sl2_series(
            jnp.asarray(scaling_dimension, dtype=self.plan.cross_ratios.dtype),
            1.0 - self.plan.cross_ratios,
            self.plan.radial_order,
        )
        return self.direct_prefactor * direct - self.reflected_prefactor * reflected

    def identity_crossing_vector(self) -> Array:
        return self.direct_prefactor - self.reflected_prefactor

    def evidence(self, scaling_dimension: ArrayLike, /) -> ScalarBlockEvidence:
        direct, final_direct, tail_direct = self.block(scaling_dimension)
        reflected, final_reflected, tail_reflected = _sl2_series(
            jnp.asarray(scaling_dimension, dtype=self.plan.cross_ratios.dtype),
            1.0 - self.plan.cross_ratios,
            self.plan.radial_order,
        )
        crossing = self.direct_prefactor * direct - self.reflected_prefactor * reflected
        final_terms = jnp.stack((final_direct, final_reflected), axis=0)
        tails = jnp.stack((tail_direct, tail_reflected), axis=0)
        finite = jnp.all(jnp.isfinite(direct)) & jnp.all(jnp.isfinite(crossing))
        return ScalarBlockEvidence(
            block_values=direct,
            crossing_values=crossing,
            final_terms=final_terms,
            tail_proxies=tails,
            finite=finite,
            scaling_dimension=jnp.asarray(scaling_dimension),
            plan_id=self.plan.plan_id,
            claim="finite-radial-series-sl2-reference-only",
        )


def _sl2_series(delta: Array, points: Array, order: int, /) -> tuple[Array, Array, Array]:
    term = points**delta
    value = term
    for index in range(order - 1):
        numerator = (delta + index) * (delta + index)
        denominator = (2.0 * delta + index) * (index + 1.0)
        term = term * (numerator / denominator) * points
        value = value + term
    tail_proxy = jnp.abs(term) * points / (1.0 - points)
    return value, term, tail_proxy


def prepare_scalar_blocks(plan: ScalarBlockPlan, /) -> PreparedScalarBlocks:
    if not isinstance(plan, ScalarBlockPlan):
        raise TypeError("plan must be ScalarBlockPlan.")
    points = plan.cross_ratios
    direct = (1.0 - points) ** (2.0 * plan.external_dimension)
    reflected = points ** (2.0 * plan.external_dimension)
    return PreparedScalarBlocks(
        plan=plan,
        direct_prefactor=direct,
        reflected_prefactor=reflected,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-finite-sl2-scalar-crossing-blocks",
                "plan": plan.plan_id,
                "convention": "(1-z)^2DeltaPhi*g(z)-z^2DeltaPhi*g(1-z)",
            }
        ),
    )


__all__ = [
    "PreparedScalarBlocks",
    "ScalarBlockEvidence",
    "ScalarBlockPlan",
    "prepare_scalar_blocks",
]
