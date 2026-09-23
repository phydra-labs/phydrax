#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Second-order effective interactions from virtual Landau-level manifolds."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._identity import MonopoleLandauLevel


class LandauLevelMixingEffectivePlan(StrictModule, NonTrainableState):
    active: MonopoleLandauLevel
    virtual: tuple[MonopoleLandauLevel, ...]
    channel_ids: tuple[str, ...] = eqx.field(static=True)
    three_body_ids: tuple[str, ...] = eqx.field(static=True)
    energy_denominators: Array
    two_body_vertices: Array
    three_body_vertices: Array
    kappa: float = eqx.field(static=True)
    denominator_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        active: MonopoleLandauLevel,
        virtual: Sequence[MonopoleLandauLevel],
        channel_ids: Sequence[str],
        energy_denominators: ArrayLike,
        two_body_vertices: ArrayLike,
        /,
        *,
        kappa: float,
        three_body_ids: Sequence[str] = (),
        three_body_vertices: ArrayLike | None = None,
        denominator_tolerance: float = 1.0e-12,
    ):
        virtual_ = tuple(virtual)
        channels = tuple(str(value).strip() for value in channel_ids)
        three_ids = tuple(str(value).strip() for value in three_body_ids)
        denominators = np.asarray(energy_denominators, dtype=np.float64)
        two_vertices = np.asarray(two_body_vertices, dtype=np.complex128)
        three_vertices = (
            np.zeros((len(virtual_), 0), dtype=np.complex128)
            if three_body_vertices is None
            else np.asarray(three_body_vertices, dtype=np.complex128)
        )
        mixing = float(kappa)
        tolerance = float(denominator_tolerance)
        if not isinstance(active, MonopoleLandauLevel) or any(
            not isinstance(value, MonopoleLandauLevel) for value in virtual_
        ):
            raise TypeError("active and virtual values must be MonopoleLandauLevel.")
        if (
            not virtual_
            or not channels
            or any(not value for value in channels + three_ids)
            or len(set(channels)) != len(channels)
            or len(set(three_ids)) != len(three_ids)
            or denominators.shape != (len(virtual_),)
            or two_vertices.shape != (len(virtual_), len(channels))
            or three_vertices.shape != (len(virtual_), len(three_ids))
            or np.any(~np.isfinite(denominators))
            or np.any(~np.isfinite(two_vertices))
            or np.any(~np.isfinite(three_vertices))
            or not isfinite(mixing)
            or mixing < 0.0
            or not isfinite(tolerance)
            or tolerance <= 0.0
            or np.any(np.abs(denominators) <= tolerance)
        ):
            raise ValueError(
                "Effective Landau-level-mixing inputs are invalid or singular."
            )
        if any(
            value.twice_monopole_strength != active.twice_monopole_strength
            for value in virtual_
        ):
            raise ValueError(
                "Active and virtual manifolds must share physical monopole strength."
            )
        self.active = active
        self.virtual = virtual_
        self.channel_ids = channels
        self.three_body_ids = three_ids
        self.energy_denominators = jnp.asarray(denominators)
        self.two_body_vertices = jnp.asarray(two_vertices)
        self.three_body_vertices = jnp.asarray(three_vertices)
        self.kappa = mixing
        self.denominator_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "landau-level-mixing-effective-plan",
                "active": active.manifold_id,
                "virtual": tuple(value.manifold_id for value in virtual_),
                "channel_ids": channels,
                "three_body_ids": three_ids,
                "arrays": array_tree_fingerprint(
                    {
                        "denominators": denominators,
                        "two_body_vertices": two_vertices,
                        "three_body_vertices": three_vertices,
                    }
                ),
                "kappa": mixing,
                "denominator_tolerance": tolerance,
            }
        )


class EffectiveLandauLevelInteractionResult(StrictModule, NonTrainableState):
    two_body_correction: Array
    three_body_correction: Array
    minimum_denominator: Array
    hermiticity_residual: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


def evaluate_effective_landau_level_interaction(
    plan: LandauLevelMixingEffectivePlan,
    /,
) -> EffectiveLandauLevelInteractionResult:
    if not isinstance(plan, LandauLevelMixingEffectivePlan):
        raise TypeError("plan must be LandauLevelMixingEffectivePlan.")
    inverse = 1.0 / plan.energy_denominators
    two = -(plan.kappa**2) * contract(
        "vc,v,vd->cd",
        jnp.conj(plan.two_body_vertices),
        inverse,
        plan.two_body_vertices,
        backend="jax",
    )
    three = -(plan.kappa**2) * contract(
        "vt,v->t",
        jnp.abs(plan.three_body_vertices) ** 2,
        inverse,
        backend="jax",
    )
    hermiticity = jnp.max(jnp.abs(two - jnp.conj(two.T)), initial=0.0)
    minimum = jnp.min(jnp.abs(plan.energy_denominators))
    successful = (
        jnp.all(jnp.isfinite(two))
        & jnp.all(jnp.isfinite(three))
        & (hermiticity <= 1.0e-10)
        & (minimum > plan.denominator_tolerance)
    )
    return EffectiveLandauLevelInteractionResult(
        two,
        three,
        minimum,
        hermiticity,
        successful,
        plan.plan_id,
        canonical_fingerprint(
            {
                "kind": "effective-landau-level-interaction-result",
                "plan": plan.plan_id,
                "two": array_tree_fingerprint(np.asarray(two)),
                "three": array_tree_fingerprint(np.asarray(three)),
            }
        ),
    )


__all__ = [
    "EffectiveLandauLevelInteractionResult",
    "LandauLevelMixingEffectivePlan",
    "evaluate_effective_landau_level_interaction",
]
