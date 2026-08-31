#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jaxtyping import Array

from ..._strict import StrictModule
from ._pair_state import (
    INTERACTION_KEY_WIDTH,
    ParticlePairRemap,
    remap_particle_pair_values,
)


class DEMNormalHistory(StrictModule):
    maximum_overlap: Array
    plastic_overlap: Array
    previous_overlap: Array


class DEMCohesionHistory(StrictModule):
    components: tuple[Any, ...]


class DEMTangentialHistory(StrictModule):
    sliding: Array
    previous_normal: Array
    displacement: Array


class DEMRotationalHistory(StrictModule):
    rolling_displacement: Array
    torsional_displacement: Array
    previous_normal: Array
    rolling_yielded: Array
    torsional_yielded: Array


class DEMContactEvaluationContext(StrictModule):
    current_keys: Array
    current_valid: Array
    continued: Array
    left_inverse_mass: Array
    right_inverse_mass: Array
    left_radius: Array
    right_radius: Array
    left_material: Array
    right_material: Array
    step_size: Array
    step_index: Array


class DEMContactHistory(StrictModule):
    """Law-compositional contact state aligned to stable pair keys."""

    pair_keys: Array
    valid: Array
    active: Array
    normal: DEMNormalHistory
    cohesion: DEMCohesionHistory
    tangential: DEMTangentialHistory
    rotational: DEMRotationalHistory

    @classmethod
    def empty(
        cls, capacity: int, ambient_dimension: int, dtype: Any, /
    ) -> DEMContactHistory:
        count = int(capacity)
        dimension = int(ambient_dimension)
        if count < 0 or dimension not in (2, 3):
            raise ValueError("Contact history capacity/dimension is invalid.")
        angular_dimension = 1 if dimension == 2 else 3
        scalar = jnp.zeros((count,), dtype=dtype)
        mask = jnp.zeros((count,), dtype=bool)
        vector = jnp.zeros((count, dimension), dtype=dtype)
        angular = jnp.zeros((count, angular_dimension), dtype=dtype)
        return cls(
            -jnp.ones((count, INTERACTION_KEY_WIDTH), dtype=jnp.int64),
            mask,
            mask,
            DEMNormalHistory(scalar, scalar, scalar),
            DEMCohesionHistory(()),
            DEMTangentialHistory(mask, vector, vector),
            DEMRotationalHistory(angular, angular, vector, mask, mask),
        )

    @property
    def values(self):
        return (
            self.active,
            self.normal,
            self.cohesion,
            self.tangential,
            self.rotational,
        )

    def with_routes(self, pair_keys: Array, valid: Array, /) -> DEMContactHistory:
        return DEMContactHistory(
            jnp.asarray(pair_keys, dtype=jnp.int64),
            jnp.asarray(valid, dtype=bool),
            self.active,
            self.normal,
            self.cohesion,
            self.tangential,
            self.rotational,
        )


def remap_dem_contact_history(
    history: DEMContactHistory,
    remap: ParticlePairRemap,
    pair_keys: Array,
    valid: Array,
    /,
) -> DEMContactHistory:
    """Remap every law-owned history leaf and install the new route identity."""

    if not isinstance(history, DEMContactHistory):
        raise TypeError("history must be a DEMContactHistory.")
    active, normal, cohesion, tangential, rotational = remap_particle_pair_values(
        remap, history.values
    )
    return DEMContactHistory(
        jnp.asarray(pair_keys, dtype=jnp.int64),
        jnp.asarray(valid, dtype=bool),
        active.astype(bool),
        normal,
        cohesion,
        DEMTangentialHistory(
            tangential.sliding.astype(bool),
            tangential.previous_normal,
            tangential.displacement,
        ),
        DEMRotationalHistory(
            rotational.rolling_displacement,
            rotational.torsional_displacement,
            rotational.previous_normal,
            rotational.rolling_yielded.astype(bool),
            rotational.torsional_yielded.astype(bool),
        ),
    )


__all__ = [
    "DEMCohesionHistory",
    "DEMContactEvaluationContext",
    "DEMContactHistory",
    "DEMNormalHistory",
    "DEMRotationalHistory",
    "DEMTangentialHistory",
    "remap_dem_contact_history",
]
