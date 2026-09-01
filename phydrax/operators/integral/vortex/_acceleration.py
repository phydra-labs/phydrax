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
from ....discretization.vortex._capabilities import VortexVelocityCapabilities
from ....discretization.vortex._interfaces import (
    DEFAULT_VORTEX_FIELD_REQUEST,
    VortexFieldRequest,
    VortexVelocityDiagnostics,
    VortexVelocityEvaluation,
)
from ....discretization.vortex._precision import VortexPrecisionPolicy
from ....discretization.vortex._source import VortexSourceState, VortexTargetState
from ._gaussian2d import gaussian_vortex_velocity_2d


class FixedClusterDiagnostics2D(StrictModule):
    far_cluster_count: Array
    direct_interaction_count: Array
    truncation_bound: Array
    maximum_reference_displacement: Array
    stale_topology: Array
    opening_angle: float = eqx.field(static=True)


class FixedClusterVortexPlan2D(StrictModule):
    """Fixed-leaf cluster approximation with direct near interactions.

    This intentionally is not an FMM. The leaf partition is built from reference
    positions and becomes stale after the declared displacement bound.
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
    capabilities: VortexVelocityCapabilities

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
        self.capabilities = VortexVelocityCapabilities(
            2,
            required_source_fields=(
                "positions",
                "strength",
                "active_mask",
                "core_radius",
            ),
            supported_fields=("velocity",),
            domain="free-space",
            precision=VortexPrecisionPolicy(),
            derivatives=(
                "source-position",
                "source-strength",
                "source-core-radius",
                "target-position",
            ),
            target_topologies=("arbitrary-targets",),
            acceleration="fixed-cluster",
        )
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
        source: VortexSourceState,
        target: VortexTargetState,
        /,
        *,
        request: VortexFieldRequest = DEFAULT_VORTEX_FIELD_REQUEST,
    ) -> VortexVelocityEvaluation:
        if not isinstance(source, VortexSourceState):
            raise TypeError("source must be VortexSourceState.")
        if not isinstance(target, VortexTargetState):
            raise TypeError("target must be VortexTargetState.")
        if request.velocity_gradient or request.vorticity:
            raise ValueError(
                "Fixed-cluster backend currently supports velocity requests only."
            )
        if (
            source.dimension != 2
            or source.capacity != self.source_capacity
            or source.core_radius is None
        ):
            raise ValueError(
                "Fixed-cluster source does not match the reference capacity."
            )
        if target.dimension != 2:
            raise ValueError("Fixed-cluster targets must be two-dimensional.")
        positions = source.safe_positions()
        gamma = source.safe_strength()
        core = source.safe_core_radius()
        targets = target.positions
        active_source = source.active_mask
        displacement = jnp.max(
            jnp.linalg.norm(positions - self.reference_position, axis=-1)
        )
        stale = displacement > self.maximum_reference_displacement
        output = jnp.zeros_like(targets)
        far_count = jnp.asarray(0, dtype=jnp.int32)
        direct_count = jnp.asarray(0, dtype=jnp.int32)
        error_bound = jnp.asarray(0.0, dtype=positions.dtype)
        for group in range(int(self.groups.shape[0])):
            indices = self.groups[group]
            valid = self.group_valid[group]
            safe_indices = jnp.where(valid, indices, 0)
            points = positions[safe_indices]
            source_active = valid & active_source[safe_indices]
            strengths = jnp.where(source_active, gamma[safe_indices], 0.0)
            cores = jnp.where(source_active, core[safe_indices], 1.0)
            center = jnp.sum(
                jnp.where(source_active[:, None], points, 0.0), axis=0
            ) / jnp.maximum(jnp.sum(source_active), 1)
            radius = jnp.max(
                jnp.where(
                    source_active,
                    jnp.linalg.norm(points - center, axis=-1),
                    0.0,
                )
            )
            distance = jnp.linalg.norm(targets - center, axis=-1)
            far = radius <= self.opening_angle * jnp.maximum(
                distance, jnp.finfo(positions.dtype).tiny
            )
            total_gamma = jnp.sum(strengths)
            absolute_gamma = jnp.sum(jnp.abs(strengths))
            coherent = jnp.abs(total_gamma) >= 0.5 * absolute_gamma
            far = far & coherent
            effective_core = jnp.max(jnp.where(valid, cores, 0.0))
            far_velocity = gaussian_vortex_velocity_2d(
                targets - center,
                total_gamma,
                jnp.maximum(
                    effective_core,
                    jnp.finfo(positions.dtype).eps,
                ),
            )
            pair_displacement = targets[:, None, :] - points[None, :, :]
            pair_shape = pair_displacement.shape[:-1]
            pair_velocity = gaussian_vortex_velocity_2d(
                pair_displacement,
                jnp.broadcast_to(strengths[None, :], pair_shape),
                jnp.broadcast_to(cores[None, :], pair_shape),
            )
            direct_velocity = jnp.sum(
                jnp.where(source_active[None, :, None], pair_velocity, 0.0),
                axis=1,
            )
            output = output + jnp.where(far[:, None], far_velocity, direct_velocity)
            far_count = far_count + jnp.sum(far, dtype=jnp.int32)
            direct_count = direct_count + jnp.sum(~far, dtype=jnp.int32) * jnp.sum(
                source_active, dtype=jnp.int32
            )
            separation = jnp.maximum(
                distance - radius,
                jnp.finfo(positions.dtype).eps,
            )
            bound = jnp.sum(jnp.abs(strengths)) * radius / (2.0 * jnp.pi * separation**2)
            error_bound = error_bound + jnp.max(jnp.where(far, bound, 0.0))
        finite = (
            jnp.all(jnp.isfinite(output))
            & jnp.all(jnp.isfinite(positions))
            & jnp.all(jnp.isfinite(gamma))
            & jnp.all(jnp.where(active_source, jnp.isfinite(core) & (core > 0.0), True))
        )
        successful = finite & ~stale
        backend = FixedClusterDiagnostics2D(
            far_count,
            direct_count,
            error_bound,
            displacement,
            stale,
            self.opening_angle,
        )
        diagnostics = VortexVelocityDiagnostics(
            jnp.asarray(self.source_capacity, dtype=jnp.int32),
            jnp.asarray(target.capacity, dtype=jnp.int32),
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
                    "kind": "fixed-cluster-vortex-evaluation-2d",
                    "plan": self.plan_id,
                    "target_count": target.capacity,
                }
            ),
            diagnostics,
        )


__all__ = ["FixedClusterDiagnostics2D", "FixedClusterVortexPlan2D"]
