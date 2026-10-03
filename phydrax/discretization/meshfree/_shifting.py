# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Tangential mesh redistribution and its conservative relative transport.

Shifting is a mesh velocity, not a physical motion law. A shift moves the
mesh tangentially with ``u_s`` while the material keeps its own velocity, so
material crosses the moving mesh with the relative velocity ``-u_s``. The
relative content flux is the native low-order upwind rate of
:class:`MeshfreeAdvection` on the surface exterior graph; a native temporal
method integrates it together with the coordinates and the measure-rate
source ``w div_G u_s`` of the moving-surface runtime.
"""

from __future__ import annotations

from typing import final, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...typing import checked, Dim, Float64, Int32
from ._exterior import PreparedMeshfreeExteriorCalculus
from ._exterior_transport import MeshfreeAdvection, MeshfreeAdvectionRate


class ShiftAmbientDim(Dim):
    """Ambient Cartesian components."""


class ShiftEdgeDim(Dim):
    """Canonical surface-graph edges."""


@final
class SurfaceShiftPolicy(StrictModule):
    """Tangential pair repulsion toward a declared target separation."""

    strength: float = eqx.field(static=True)
    target_separation: float = eqx.field(static=True)

    def __init__(self, *, strength: float = 0.1, target_separation: float) -> None:
        if (
            not np.isfinite(strength)
            or not np.isfinite(target_separation)
            or strength < 0
            or target_separation <= 0
        ):
            raise ValueError(
                "Shift strength must be nonnegative and the target separation "
                "positive and finite."
            )
        self.strength = float(strength)
        self.target_separation = float(target_separation)

    def velocity(self, points: ArrayLike, normals: ArrayLike, pairs: ArrayLike) -> Array:
        """Tangential mesh velocity pushing pairs closer than the target apart."""
        x, n = jnp.asarray(points), jnp.asarray(normals)
        edges = jnp.asarray(pairs, dtype=jnp.int32)
        if x.ndim != 2 or n.shape != x.shape or edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("Shifting needs compact points/normals and endpoint pairs.")
        delta = x[edges[:, 0]] - x[edges[:, 1]]
        length = jnp.linalg.norm(delta, axis=1)
        safe = jnp.maximum(length, jnp.finfo(x.dtype).tiny)
        force = (
            self.strength
            * jnp.maximum(self.target_separation - length, 0)[:, None]
            * delta
            / safe[:, None]
        )
        velocity = (
            jnp.zeros_like(x).at[edges[:, 0]].add(force).at[edges[:, 1]].add(-force)
        )
        return velocity - jnp.sum(velocity * n, axis=1)[:, None] * n


@final
class SurfaceMeshShift(StrictModule):
    """Mesh redistribution law with its conservative relative transport rate.

    The prepared surface exterior owns the graph, its canonical orientation and
    its metric weights (frozen at the support epoch). The relative volume flux
    of edge ``e = (i, j)`` at the stage points is
    ``w_e (r_i + r_j)/2 . (x_j - x_i)``; with moment-exact weights its outgoing
    sum is the discrete ``w div_G r``. Upwind positivity is certified only for a
    nonnegative accepted metric under the outgoing CFL bound.
    """

    __strict_contract__ = True
    policy: SurfaceShiftPolicy
    advection: MeshfreeAdvection
    pairs: Int32[ShiftEdgeDim, Literal[2]]
    weights: Float64[ShiftEdgeDim]
    metric_nonnegative: bool = eqx.field(static=True)
    shift_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        policy: SurfaceShiftPolicy,
        exterior: PreparedMeshfreeExteriorCalculus,
        /,
    ) -> None:
        if not bool(np.asarray(exterior.metric_result.accepted)):
            raise ValueError("Mesh shifting needs an accepted surface exterior metric.")
        weights = jnp.asarray(exterior.metric_result.weights, dtype=jnp.float64)
        self.policy = policy
        self.advection = MeshfreeAdvection(exterior)
        self.pairs = jnp.asarray(exterior.pairs, dtype=jnp.int32)
        self.weights = weights
        self.metric_nonnegative = bool(np.all(np.asarray(weights) >= 0))
        self.shift_id = canonical_fingerprint(
            {
                "kind": "surface-mesh-shift",
                "graph": exterior.incidence.source.space_id,
                "strength": policy.strength,
                "target_separation": policy.target_separation,
            }
        )

    def velocity(self, points: ArrayLike, normals: ArrayLike, /) -> Array:
        return self.policy.velocity(points, normals, self.pairs)

    def volume_flux(self, points: ArrayLike, relative_velocity: ArrayLike, /) -> Array:
        """Oriented relative volume flux of the stage points, positive ``i -> j``."""
        x, r = jnp.asarray(points), jnp.asarray(relative_velocity)
        if r.shape != x.shape:
            raise ValueError("Relative velocity must have one vector per point.")
        first, second = self.pairs[:, 0], self.pairs[:, 1]
        mean = 0.5 * (r[first] + r[second])
        return self.weights * jnp.sum(mean * (x[second] - x[first]), axis=1)

    def rate(
        self,
        concentration: ArrayLike,
        points: ArrayLike,
        relative_velocity: ArrayLike,
        /,
    ) -> MeshfreeAdvectionRate:
        """Native upwind content rate of material crossing the moving mesh."""
        return self.advection.rate(
            concentration, self.volume_flux(points, relative_velocity)
        )


__all__ = [
    "SurfaceMeshShift",
    "SurfaceShiftPolicy",
]
