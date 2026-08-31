#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ....discretization.vortex._filament import VortexFilamentState


class FilamentVelocityDiagnostics(StrictModule):
    active_segments: Array
    minimum_segment_length: Array
    coincident_endpoint_count: Array
    finite: Array


class FilamentVelocityEvaluation(StrictModule):
    velocity: Array
    successful: Array
    diagnostics: FilamentVelocityDiagnostics
    evaluation_id: str = eqx.field(static=True)


def regularized_filament_velocity_3d(
    target: ArrayLike,
    start: ArrayLike,
    end: ArrayLike,
    circulation: ArrayLike,
    core_radius: ArrayLike,
    /,
) -> Array:
    """Velocity of oriented finite segments with a Rosenhead-type core."""

    targets = jnp.asarray(target)
    starts = jnp.asarray(start, dtype=targets.dtype)
    ends = jnp.asarray(end, dtype=targets.dtype)
    gamma = jnp.asarray(circulation, dtype=targets.dtype)
    core = jnp.asarray(core_radius, dtype=targets.dtype)
    if (
        targets.shape[-1:] != (3,)
        or starts.shape[-1:] != (3,)
        or ends.shape != starts.shape
    ):
        raise ValueError(
            "Filament target/start/end arrays require trailing dimension three."
        )
    if gamma.shape != starts.shape[:-1] or core.shape != gamma.shape:
        raise ValueError(
            "Filament circulation/core shapes must match segment leading shape."
        )
    r1 = targets[..., None, :] - starts
    r2 = targets[..., None, :] - ends
    r0 = ends - starts
    length = jnp.sqrt(jnp.sum(r0 * r0, axis=-1))
    norm1 = jnp.sqrt(jnp.sum(r1 * r1, axis=-1))
    norm2 = jnp.sqrt(jnp.sum(r2 * r2, axis=-1))
    tiny = jnp.finfo(targets.dtype).tiny
    safe1 = jnp.maximum(norm1, tiny)
    safe2 = jnp.maximum(norm2, tiny)
    cross = jnp.cross(r1, r2)
    cross_squared = jnp.sum(cross * cross, axis=-1)
    denominator = cross_squared + (core * length) ** 2
    safe_denominator = jnp.maximum(denominator, tiny)
    axial = jnp.sum(r0 * (r1 / safe1[..., None] - r2 / safe2[..., None]), axis=-1)
    coefficient = gamma * axial / (4.0 * math.pi * safe_denominator)
    valid = (length > 0.0) & (core >= 0.0) & jnp.isfinite(coefficient)
    return jnp.sum(
        jnp.where(valid[..., None], coefficient[..., None] * cross, 0.0), axis=-2
    )


class PreparedFilamentVelocity3D(StrictModule):
    """Arbitrary-target velocity evaluation for one fixed filament topology."""

    filament: VortexFilamentState
    evaluator_id: str = eqx.field(static=True)

    def __init__(self, filament: VortexFilamentState, /):
        if not isinstance(filament, VortexFilamentState):
            raise TypeError("filament must be VortexFilamentState.")
        self.filament = filament
        self.evaluator_id = canonical_fingerprint(
            {
                "kind": "prepared-filament-velocity-3d",
                "topology": filament.topology.topology_id,
            }
        )

    def evaluate(self, targets: ArrayLike, /) -> FilamentVelocityEvaluation:
        target = jnp.asarray(targets)
        if target.ndim != 2 or target.shape[1] != 3:
            raise ValueError("Filament targets must have shape (target_count, 3).")
        geometry = self.filament.geometry()
        active = self.filament.topology.active
        circulation = jnp.where(active, self.filament.circulation, 0.0)
        core = jnp.where(active, self.filament.core_radius, 1.0)
        velocity = regularized_filament_velocity_3d(
            target,
            geometry.start,
            geometry.end,
            circulation,
            core,
        )
        finite = (
            geometry.finite & geometry.nondegenerate & jnp.all(jnp.isfinite(velocity))
        )
        diagnostics = FilamentVelocityDiagnostics(
            jnp.sum(active, dtype=jnp.int32),
            geometry.minimum_active_length,
            jnp.sum(
                (
                    jnp.linalg.norm(
                        target[:, None, :] - geometry.start[None, :, :], axis=-1
                    )
                    == 0.0
                )
                | (
                    jnp.linalg.norm(
                        target[:, None, :] - geometry.end[None, :, :], axis=-1
                    )
                    == 0.0
                ),
                dtype=jnp.int32,
            ),
            finite,
        )
        return FilamentVelocityEvaluation(
            velocity,
            finite,
            diagnostics,
            canonical_fingerprint(
                {
                    "kind": "filament-velocity-evaluation",
                    "evaluator": self.evaluator_id,
                    "target_count": int(target.shape[0]),
                }
            ),
        )


__all__ = [
    "FilamentVelocityDiagnostics",
    "FilamentVelocityEvaluation",
    "PreparedFilamentVelocity3D",
    "regularized_filament_velocity_3d",
]
