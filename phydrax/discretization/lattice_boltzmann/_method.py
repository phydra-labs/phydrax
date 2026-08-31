#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._collision import (
    BGKCollisionPlan,
    CentralMomentCollisionPlan,
    collide_detailed,
    CumulantCollisionPlan,
    EntropicCollisionPlan,
    KBCCollisionPlan,
    LatticeBoltzmannCollisionPlan,
    LatticeBoltzmannCollisionResult,
    MRTCollisionPlan,
    prepare_lattice_boltzmann_collision,
    PreparedLatticeBoltzmannCollision,
    quadratic_equilibrium,
    RegularizedCollisionPlan,
    SmagorinskyCollisionPlan,
    TRTCollisionPlan,
)
from ._forcing import guo_raw_source, GuoForcingPlan, zero_force_source
from ._lattice import LatticeBoltzmannVelocitySet
from ._precision import LatticeBoltzmannPrecisionPolicy


_COLLISION_TYPES = (
    BGKCollisionPlan,
    TRTCollisionPlan,
    MRTCollisionPlan,
    RegularizedCollisionPlan,
    SmagorinskyCollisionPlan,
    CentralMomentCollisionPlan,
    CumulantCollisionPlan,
    KBCCollisionPlan,
    EntropicCollisionPlan,
)


class LatticeBoltzmannMethodPlan(StrictModule, NonTrainableState):
    """Athermal equilibrium, collision, and explicitly compatible force plan."""

    collision: LatticeBoltzmannCollisionPlan
    forcing: GuoForcingPlan | None
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        collision: LatticeBoltzmannCollisionPlan,
        /,
        *,
        forcing: GuoForcingPlan | None = None,
    ):
        if not isinstance(collision, _COLLISION_TYPES):
            raise TypeError("collision must be a supported LBM collision plan.")
        if forcing is not None and not isinstance(forcing, GuoForcingPlan):
            raise TypeError("forcing must be GuoForcingPlan or None.")
        if forcing is not None and not forcing.supports(collision.family):
            raise ValueError(
                f"Guo forcing is not certified for {collision.family!r} collision."
            )
        self.collision = collision
        self.forcing = forcing
        self.method_id = canonical_fingerprint(
            {
                "kind": "athermal-lattice-boltzmann-method",
                "collision": collision.collision_id,
                "forcing": None if forcing is None else forcing.forcing_id,
                "equilibrium": "quadratic-athermal",
            }
        )

    def prepare(
        self,
        velocity_set: LatticeBoltzmannVelocitySet,
        precision: LatticeBoltzmannPrecisionPolicy,
        /,
    ) -> "PreparedLatticeBoltzmannMethodPlan":
        collision = prepare_lattice_boltzmann_collision(
            self.collision, velocity_set, precision
        )
        return PreparedLatticeBoltzmannMethodPlan(
            collision,
            self.forcing,
            canonical_fingerprint(
                {
                    "kind": "prepared-lattice-boltzmann-method",
                    "method": self.method_id,
                    "collision": collision.prepared_id,
                    "lattice": velocity_set.lattice_id,
                    "precision": precision.policy_id,
                }
            ),
        )

    def collide(
        self,
        populations: Array,
        density: Array,
        velocity: Array,
        force_density: Array,
        even_rate: Array,
        velocity_set: LatticeBoltzmannVelocitySet,
        precision: LatticeBoltzmannPrecisionPolicy,
        /,
    ) -> LatticeBoltzmannCollisionResult:
        return self.prepare(velocity_set, precision).collide(
            populations,
            density,
            velocity,
            force_density,
            even_rate,
            velocity_set,
            precision,
        )

    def collide_detailed(self, *args, **kwargs) -> LatticeBoltzmannCollisionResult:
        return self.collide(*args, **kwargs)


class PreparedLatticeBoltzmannMethodPlan(StrictModule, NonTrainableState):
    """Lattice-bound collision and forcing with no preparation inside cell kernels."""

    collision: PreparedLatticeBoltzmannCollision
    forcing: GuoForcingPlan | None
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        collision: PreparedLatticeBoltzmannCollision,
        forcing: GuoForcingPlan | None,
        method_id: str,
        /,
    ):
        if not isinstance(collision, PreparedLatticeBoltzmannCollision):
            raise TypeError("collision must be PreparedLatticeBoltzmannCollision.")
        if forcing is not None and not isinstance(forcing, GuoForcingPlan):
            raise TypeError("forcing must be GuoForcingPlan or None.")
        identifier = str(method_id)
        if not identifier:
            raise ValueError("method_id must be nonempty.")
        self.collision = collision
        self.forcing = forcing
        self.method_id = identifier

    def collide(
        self,
        populations: Array,
        density: Array,
        velocity: Array,
        force_density: Array,
        even_rate: Array,
        velocity_set: LatticeBoltzmannVelocitySet,
        precision: LatticeBoltzmannPrecisionPolicy,
        /,
    ) -> LatticeBoltzmannCollisionResult:
        equilibrium = quadratic_equilibrium(density, velocity, velocity_set, precision)
        raw_force = (
            zero_force_source(populations)
            if self.forcing is None
            else guo_raw_source(velocity, force_density, velocity_set, precision)
        )
        return collide_detailed(
            self.collision,
            populations,
            equilibrium,
            raw_force,
            even_rate,
            velocity,
            velocity_set,
            precision,
        )

    def collide_detailed(self, *args, **kwargs) -> LatticeBoltzmannCollisionResult:
        return self.collide(*args, **kwargs)


__all__ = [
    "LatticeBoltzmannMethodPlan",
    "PreparedLatticeBoltzmannMethodPlan",
]
