#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._phase_space import PhaseSpacePoint, TwoBodyPhaseSpaceMap


class TwoBodyDecayPlan(StrictModule, NonTrainableState):
    parent_pdg_id: int = eqx.field(static=True)
    daughter_pdg_ids: tuple[int, int] = eqx.field(static=True)
    phase_space: TwoBodyPhaseSpaceMap
    branching_fraction: Array
    decay_model_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        parent_pdg_id: int,
        daughter_pdg_ids: Sequence[int],
        daughter_masses: Sequence[float],
        /,
        *,
        branching_fraction: float,
        decay_model_id: str = "isotropic-two-body",
    ):
        daughters = tuple(daughter_pdg_ids)
        masses = tuple(float(value) for value in daughter_masses)
        fraction = float(branching_fraction)
        model = str(decay_model_id).strip()
        if len(daughters) != 2 or len(masses) != 2 or not model:
            raise ValueError(
                "Two-body decay requires two daughters, masses, and a model ID."
            )
        if any(not math.isfinite(value) or value < 0.0 for value in masses):
            raise ValueError("Daughter masses must be finite and nonnegative.")
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("branching_fraction must lie in (0, 1].")
        self.parent_pdg_id = int(parent_pdg_id)
        self.daughter_pdg_ids = daughters
        self.phase_space = TwoBodyPhaseSpaceMap(*masses)
        self.branching_fraction = jnp.asarray(fraction)
        self.decay_model_id = model
        self.plan_id = canonical_fingerprint(
            {
                "kind": "bounded-two-body-decay",
                "parent": self.parent_pdg_id,
                "daughters": list(daughters),
                "phase_space": self.phase_space.map_id,
                "branching_fraction": fraction,
                "model": model,
            }
        )


class TwoBodyDecayResult(StrictModule, NonTrainableState):
    point: PhaseSpacePoint
    weight: Array
    plan_id: str = eqx.field(static=True)


def decay_two_body(
    plan: TwoBodyDecayPlan,
    parent_momentum: ArrayLike,
    unit_coordinates: ArrayLike,
    /,
) -> TwoBodyDecayResult:
    if not isinstance(plan, TwoBodyDecayPlan):
        raise TypeError("plan must be TwoBodyDecayPlan.")
    point = plan.phase_space.map(unit_coordinates, parent_momentum)
    return TwoBodyDecayResult(
        point, point.jacobian * plan.branching_fraction, plan.plan_id
    )


__all__ = ["TwoBodyDecayPlan", "TwoBodyDecayResult", "decay_two_body"]
