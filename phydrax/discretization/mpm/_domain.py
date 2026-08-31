#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class MPMParticleDomainPlan(StrictModule, NonTrainableState):
    """Admissible material-point box inside a complete computational halo."""

    bounds: Array
    periodic: tuple[bool, ...] = eqx.field(static=True)
    support_margin: tuple[float, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        bounds: ArrayLike,
        /,
        *,
        periodic: Sequence[bool] | None = None,
        support_margin: float | Sequence[float],
    ):
        bounds_ = np.asarray(bounds, dtype=float)
        if bounds_.ndim != 2 or bounds_.shape[0] != 2 or bounds_.shape[1] not in (2, 3):
            raise ValueError("MPM particle bounds must have shape (2, 2) or (2, 3).")
        if np.any(~np.isfinite(bounds_)) or np.any(bounds_[1] <= bounds_[0]):
            raise ValueError("MPM particle bounds must be finite and strictly ordered.")
        dimension = int(bounds_.shape[1])
        periodic_ = (
            (False,) * dimension
            if periodic is None
            else tuple(bool(value) for value in periodic)
        )
        if len(periodic_) != dimension:
            raise ValueError("periodic must contain one flag per MPM dimension.")
        if np.isscalar(support_margin):
            margins = (float(support_margin),) * dimension
        else:
            margins = tuple(float(value) for value in support_margin)
        if len(margins) != dimension or any(
            not np.isfinite(value) or value < 0.0 for value in margins
        ):
            raise ValueError("support_margin must contain finite nonnegative values.")
        self.bounds = jnp.asarray(bounds_)
        self.periodic = periodic_
        self.support_margin = margins
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mpm-particle-domain",
                "bounds": array_tree_fingerprint(bounds_),
                "periodic": periodic_,
                "support_margin": margins,
                "position_storage": "unwrapped",
            }
        )

    @property
    def dimension(self) -> int:
        return int(self.bounds.shape[1])

    def contains(self, position: ArrayLike, /) -> Array:
        value = jnp.asarray(position)
        if value.ndim < 1 or value.shape[-1] != self.dimension:
            raise ValueError("Particle positions must end in the MPM domain dimension.")
        contained = jnp.ones(value.shape[:-1], dtype=bool)
        for axis, periodic in enumerate(self.periodic):
            if not periodic:
                contained = contained & (value[..., axis] >= self.bounds[0, axis])
                contained = contained & (value[..., axis] <= self.bounds[1, axis])
        return contained


__all__ = ["MPMParticleDomainPlan"]
