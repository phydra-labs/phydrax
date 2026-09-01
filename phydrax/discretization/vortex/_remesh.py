#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class VortexRemeshResult2D(StrictModule):
    position: Array
    circulation: Array
    core_radius: Array
    active: Array
    circulation_residual: Array
    first_moment_residual: Array
    dropped_circulation: Array
    successful: Array
    remesh_id: str = eqx.field(static=True)


class ConservativeVortexRemeshPlan2D(StrictModule, NonTrainableState):
    """Bilinear projection to a fixed nodal lattice preserving zeroth/first moments."""

    lower: Array
    upper: Array
    shape: tuple[int, int] = eqx.field(static=True)
    grid_position: Array
    core_radius: float = eqx.field(static=True)
    boundary: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        lower: ArrayLike,
        upper: ArrayLike,
        shape: tuple[int, int],
        core_radius: float,
        /,
        *,
        boundary: str = "reject",
    ):
        lower_ = np.asarray(lower, dtype=float)
        upper_ = np.asarray(upper, dtype=float)
        shape_ = tuple(int(value) for value in shape)
        if lower_.shape != (2,) or upper_.shape != (2,) or np.any(upper_ <= lower_):
            raise ValueError("Remesh bounds must be increasing two-vectors.")
        if len(shape_) != 2 or any(value < 2 for value in shape_):
            raise ValueError("Remesh shape requires at least two nodes per axis.")
        if core_radius <= 0.0 or boundary not in ("reject", "drop"):
            raise ValueError("Remesh core radius/boundary policy is invalid.")
        axes = tuple(
            jnp.linspace(lower_[axis], upper_[axis], shape_[axis]) for axis in range(2)
        )
        xx, yy = jnp.meshgrid(*axes, indexing="ij")
        grid = jnp.stack((xx.reshape(-1), yy.reshape(-1)), axis=-1)
        self.lower = jnp.asarray(lower_)
        self.upper = jnp.asarray(upper_)
        self.shape = shape_
        self.grid_position = grid
        self.core_radius = float(core_radius)
        self.boundary = boundary
        self.plan_id = canonical_fingerprint(
            {
                "kind": "conservative-vortex-remesh-2d",
                "lower": lower_.tolist(),
                "upper": upper_.tolist(),
                "shape": shape_,
                "core_radius": float(core_radius),
                "boundary": boundary,
            }
        )

    @property
    def capacity(self) -> int:
        return self.shape[0] * self.shape[1]

    def apply(
        self,
        position: ArrayLike,
        circulation: ArrayLike,
        active: ArrayLike | None = None,
        /,
    ) -> VortexRemeshResult2D:
        points = jnp.asarray(position)
        gamma = jnp.asarray(circulation, dtype=points.dtype)
        mask = (
            jnp.ones(gamma.shape, dtype=bool)
            if active is None
            else jnp.asarray(active, dtype=bool)
        )
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or gamma.shape != points.shape[:1]
            or mask.shape != gamma.shape
        ):
            raise ValueError("Remesh particle arrays have incompatible shapes.")
        spacing = (self.upper - self.lower) / jnp.asarray(
            (self.shape[0] - 1, self.shape[1] - 1), dtype=points.dtype
        )
        coordinate = (points - self.lower) / spacing
        base = jnp.floor(coordinate).astype(jnp.int32)
        fraction = coordinate - base
        in_domain = jnp.all((points >= self.lower) & (points <= self.upper), axis=-1)
        valid_source = (
            mask
            & in_domain
            & jnp.all(jnp.isfinite(points), axis=-1)
            & jnp.isfinite(gamma)
        )
        base = jnp.clip(
            base, 0, jnp.asarray((self.shape[0] - 2, self.shape[1] - 2), dtype=jnp.int32)
        )
        fraction = jnp.where((points == self.upper), 1.0, fraction)
        offsets = jnp.asarray(((0, 0), (1, 0), (0, 1), (1, 1)), dtype=jnp.int32)
        indices = base[:, None, :] + offsets[None, :, :]
        flat = indices[..., 0] * self.shape[1] + indices[..., 1]
        fx, fy = fraction[:, 0], fraction[:, 1]
        weights = jnp.stack(
            ((1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy), axis=-1
        )
        payload = jnp.where(valid_source[:, None], gamma[:, None] * weights, 0.0)
        target_gamma = (
            jnp.zeros((self.capacity,), dtype=gamma.dtype)
            .at[flat.reshape(-1)]
            .add(payload.reshape(-1))
        )
        target_active = jnp.abs(target_gamma) > 0.0
        total_before = jnp.sum(jnp.where(mask, gamma, 0.0))
        total_after = jnp.sum(target_gamma)
        first_after = jnp.sum(target_gamma[:, None] * self.grid_position, axis=0)
        dropped = jnp.sum(jnp.where(mask & ~in_domain, gamma, 0.0))
        circulation_residual = total_after + dropped - total_before
        first_residual = first_after - jnp.sum(
            jnp.where(valid_source[:, None], gamma[:, None] * points, 0.0), axis=0
        )
        tolerance = (
            256 * jnp.finfo(gamma.dtype).eps * jnp.maximum(jnp.sum(jnp.abs(gamma)), 1.0)
        )
        boundary_ok = (
            jnp.all(~mask | in_domain) if self.boundary == "reject" else jnp.asarray(True)
        )
        successful = (
            boundary_ok
            & (jnp.abs(circulation_residual) <= tolerance)
            & (
                jnp.max(jnp.abs(first_residual))
                <= tolerance * jnp.maximum(jnp.max(jnp.abs(points)), 1.0)
            )
        )
        return VortexRemeshResult2D(
            self.grid_position,
            target_gamma,
            jnp.full((self.capacity,), self.core_radius, dtype=gamma.dtype),
            target_active,
            circulation_residual,
            first_residual,
            dropped,
            successful,
            self.plan_id,
        )


__all__ = ["ConservativeVortexRemeshPlan2D", "VortexRemeshResult2D"]
