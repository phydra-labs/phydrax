#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ....discretization.vortex._interfaces import (
    DEFAULT_VORTEX_FIELD_REQUEST,
    VortexFieldRequest,
    VortexVelocityDiagnostics,
    VortexVelocityEvaluation,
)
from ._gaussian2d import gaussian_vortex_velocity_2d


class BarnesHutDiagnostics2D(StrictModule):
    far_cluster_count: Array
    direct_interaction_count: Array
    truncation_bound: Array
    maximum_reference_displacement: Array
    stale_topology: Array
    opening_angle: float = eqx.field(static=True)


class BarnesHutVortexPlan2D(StrictModule):
    """Fixed-leaf Barnes--Hut treecode with direct near interactions.

    The leaf partition is built from reference positions. Numeric source positions may
    vary until the declared refresh displacement is exceeded; evaluation then fails
    closed with ``stale_topology`` rather than silently using an invalid hierarchy.
    """

    reference_position: Array
    groups: Array
    group_valid: Array
    reference_center: Array
    reference_radius: Array
    opening_angle: float = eqx.field(static=True)
    maximum_reference_displacement: float = eqx.field(static=True)
    source_capacity: int = eqx.field(static=True)
    leaf_size: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_position: ArrayLike,
        /,
        *,
        leaf_size: int = 32,
        opening_angle: float = 0.5,
        maximum_reference_displacement: float = 0.1,
    ):
        reference = np.asarray(reference_position, dtype=float)
        leaf = int(leaf_size)
        angle = float(opening_angle)
        displacement = float(maximum_reference_displacement)
        if (
            reference.ndim != 2
            or reference.shape[1] != 2
            or reference.shape[0] == 0
            or np.any(~np.isfinite(reference))
        ):
            raise ValueError("Barnes-Hut reference positions require finite shape (N,2).")
        if leaf <= 0 or not 0.0 < angle < 1.0 or displacement <= 0.0:
            raise ValueError("Barnes-Hut leaf/opening/refresh controls are invalid.")
        order = np.lexsort((reference[:, 1], reference[:, 0]))
        group_count = (reference.shape[0] + leaf - 1) // leaf
        groups = -np.ones((group_count, leaf), dtype=np.int32)
        valid = np.zeros((group_count, leaf), dtype=bool)
        centers, radii = [], []
        for group in range(group_count):
            indices = order[group * leaf : (group + 1) * leaf]
            groups[group, : indices.size] = indices
            valid[group, : indices.size] = True
            values = reference[indices]
            center = np.mean(values, axis=0)
            centers.append(center)
            radii.append(
                max(
                    float(np.max(np.linalg.norm(values - center, axis=1))),
                    np.finfo(float).eps,
                )
            )
        self.reference_position = jnp.asarray(reference)
        self.groups = jnp.asarray(groups)
        self.group_valid = jnp.asarray(valid)
        self.reference_center = jnp.asarray(centers)
        self.reference_radius = jnp.asarray(radii)
        self.opening_angle = angle
        self.maximum_reference_displacement = displacement
        self.source_capacity = int(reference.shape[0])
        self.leaf_size = leaf
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-leaf-barnes-hut-vortex-2d",
                "reference": array_tree_fingerprint(reference),
                "leaf_size": leaf,
                "opening_angle": angle,
                "maximum_reference_displacement": displacement,
            }
        )

    def evaluate(
        self,
        position: ArrayLike,
        circulation: ArrayLike,
        core_radius: ArrayLike,
        targets: ArrayLike,
        /,
        *,
        request: VortexFieldRequest = DEFAULT_VORTEX_FIELD_REQUEST,
    ) -> VortexVelocityEvaluation:
        if request.velocity_gradient or request.vorticity:
            raise ValueError(
                "Barnes-Hut backend currently supports velocity requests only."
            )
        source = jnp.asarray(position)
        gamma = jnp.asarray(circulation, dtype=source.dtype)
        core = jnp.asarray(core_radius, dtype=source.dtype)
        target = jnp.asarray(targets, dtype=source.dtype)
        if (
            source.shape != (self.source_capacity, 2)
            or gamma.shape != (self.source_capacity,)
            or core.shape != (self.source_capacity,)
        ):
            raise ValueError("Barnes-Hut sources do not match reference capacity.")
        if target.ndim != 2 or target.shape[1] != 2:
            raise ValueError("Barnes-Hut targets require shape (M,2).")
        displacement = jnp.max(jnp.linalg.norm(source - self.reference_position, axis=-1))
        stale = displacement > self.maximum_reference_displacement
        output = jnp.zeros_like(target)
        far_count = jnp.asarray(0, dtype=jnp.int32)
        direct_count = jnp.asarray(0, dtype=jnp.int32)
        error_bound = jnp.asarray(0.0, dtype=source.dtype)
        for group in range(int(self.groups.shape[0])):
            indices = self.groups[group]
            valid = self.group_valid[group]
            safe_indices = jnp.where(valid, indices, 0)
            points = source[safe_indices]
            strengths = jnp.where(valid, gamma[safe_indices], 0.0)
            cores = jnp.where(valid, core[safe_indices], 1.0)
            center = jnp.sum(
                jnp.where(valid[:, None], points, 0.0), axis=0
            ) / jnp.maximum(jnp.sum(valid), 1)
            radius = jnp.max(
                jnp.where(valid, jnp.linalg.norm(points - center, axis=-1), 0.0)
            )
            distance = jnp.linalg.norm(target - center, axis=-1)
            far = radius <= self.opening_angle * jnp.maximum(
                distance, jnp.finfo(source.dtype).tiny
            )
            total_gamma = jnp.sum(strengths)
            absolute_gamma = jnp.sum(jnp.abs(strengths))
            coherent = jnp.abs(total_gamma) >= 0.5 * absolute_gamma
            far = far & coherent
            effective_core = jnp.max(jnp.where(valid, cores, 0.0))
            far_velocity = gaussian_vortex_velocity_2d(
                target - center,
                total_gamma,
                jnp.maximum(effective_core, jnp.finfo(source.dtype).eps),
            )
            pair_displacement = target[:, None, :] - points[None, :, :]
            pair_shape = pair_displacement.shape[:-1]
            pair_velocity = gaussian_vortex_velocity_2d(
                pair_displacement,
                jnp.broadcast_to(strengths[None, :], pair_shape),
                jnp.broadcast_to(cores[None, :], pair_shape),
            )
            direct_velocity = jnp.sum(
                jnp.where(valid[None, :, None], pair_velocity, 0.0), axis=1
            )
            output = output + jnp.where(far[:, None], far_velocity, direct_velocity)
            far_count = far_count + jnp.sum(far, dtype=jnp.int32)
            direct_count = direct_count + jnp.sum(~far, dtype=jnp.int32) * jnp.sum(
                valid, dtype=jnp.int32
            )
            separation = jnp.maximum(distance - radius, jnp.finfo(source.dtype).eps)
            bound = jnp.sum(jnp.abs(strengths)) * radius / (2.0 * jnp.pi * separation**2)
            error_bound = error_bound + jnp.max(jnp.where(far, bound, 0.0))
        finite = (
            jnp.all(jnp.isfinite(output))
            & jnp.all(jnp.isfinite(source))
            & jnp.all(jnp.isfinite(gamma))
            & jnp.all(jnp.isfinite(core))
            & jnp.all(core > 0.0)
        )
        successful = finite & ~stale
        backend = BarnesHutDiagnostics2D(
            far_count, direct_count, error_bound, displacement, stale, self.opening_angle
        )
        diagnostics = VortexVelocityDiagnostics(
            jnp.asarray(self.source_capacity, dtype=jnp.int32),
            jnp.asarray(target.shape[0], dtype=jnp.int32),
            far_count + direct_count,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.min(core),
            finite,
            jnp.all(jnp.isfinite(output)),
            ~stale,
            successful,
            backend,
        )
        return VortexVelocityEvaluation(
            output,
            None,
            None,
            successful,
            self.plan_id,
            canonical_fingerprint(
                {
                    "kind": "barnes-hut-vortex-evaluation-2d",
                    "plan": self.plan_id,
                    "target_count": int(target.shape[0]),
                }
            ),
            diagnostics,
        )


__all__ = ["BarnesHutDiagnostics2D", "BarnesHutVortexPlan2D"]
