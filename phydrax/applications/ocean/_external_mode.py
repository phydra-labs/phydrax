#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.finite_volume._hydrostatic_grid import PreparedHydrostaticGrid


ExternalModeSubcycleKind: TypeAlias = Literal["fixed-count", "adaptive-cfl"]


class ExternalModeSubcycleSchedule(StrictModule):
    substep_sizes: Array
    active_mask: Array
    count: Array
    maximum_courant: Array
    capacity_valid: Array
    finite: Array
    successful: Array
    policy_id: str = eqx.field(static=True)


class ExternalModeSubcyclePolicy(StrictModule, NonTrainableState):
    """Fixed-capacity split-explicit schedule with fixed or adaptive count."""

    kind: ExternalModeSubcycleKind = eqx.field(static=True)
    fixed_count: int = eqx.field(static=True)
    maximum_substeps: int = eqx.field(static=True)
    target_courant: float = eqx.field(static=True)
    minimum_spacing: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: ExternalModeSubcycleKind,
        /,
        *,
        fixed_count: int = 1,
        maximum_substeps: int | None = None,
        target_courant: float = 0.8,
        minimum_spacing: float = 0.0,
    ):
        if kind not in ("fixed-count", "adaptive-cfl"):
            raise ValueError("Unknown external-mode subcycle policy.")
        fixed = int(fixed_count)
        maximum = fixed if maximum_substeps is None else int(maximum_substeps)
        courant = float(target_courant)
        spacing = float(minimum_spacing)
        if (
            fixed <= 0
            or maximum <= 0
            or fixed > maximum
            or not np.isfinite(courant)
            or courant <= 0.0
            or not np.isfinite(spacing)
            or spacing < 0.0
        ):
            raise ValueError("External-mode subcycle policy is invalid.")
        self.kind = kind
        self.fixed_count = fixed
        self.maximum_substeps = maximum
        self.target_courant = courant
        self.minimum_spacing = spacing
        self.policy_id = canonical_fingerprint(
            {
                "kind": "external-mode-subcycle-policy",
                "mode": kind,
                "fixed_count": fixed,
                "maximum_substeps": maximum,
                "target_courant": courant,
                "minimum_spacing": spacing,
            }
        )

    @classmethod
    def fixed(cls, count: int, /) -> "ExternalModeSubcyclePolicy":
        return cls("fixed-count", fixed_count=count, maximum_substeps=count)

    @classmethod
    def adaptive_cfl(
        cls,
        maximum_substeps: int,
        /,
        *,
        target_courant: float = 0.8,
        minimum_substeps: int = 1,
        minimum_spacing: float = 0.0,
    ) -> "ExternalModeSubcyclePolicy":
        return cls(
            "adaptive-cfl",
            fixed_count=minimum_substeps,
            maximum_substeps=maximum_substeps,
            target_courant=target_courant,
            minimum_spacing=minimum_spacing,
        )

    def empty(self, dtype, /) -> ExternalModeSubcycleSchedule:
        return ExternalModeSubcycleSchedule(
            substep_sizes=jnp.zeros((self.maximum_substeps,), dtype=dtype),
            active_mask=jnp.zeros((self.maximum_substeps,), dtype=bool),
            count=jnp.asarray(0, dtype=jnp.int32),
            maximum_courant=jnp.asarray(0.0, dtype=dtype),
            capacity_valid=jnp.asarray(True),
            finite=jnp.asarray(True),
            successful=jnp.asarray(True),
            policy_id=self.policy_id,
        )

    def schedule(
        self,
        geometry: PreparedHydrostaticGrid,
        eta: ArrayLike,
        step_size: ArrayLike,
        gravity: ArrayLike,
        /,
        *,
        barotropic_transport: tuple[ArrayLike, ArrayLike] | None = None,
    ) -> ExternalModeSubcycleSchedule:
        if not isinstance(geometry, PreparedHydrostaticGrid):
            raise TypeError("geometry must be a PreparedHydrostaticGrid.")
        eta_ = jnp.asarray(eta, dtype=geometry.cell_area.dtype)
        dt = jnp.asarray(step_size, dtype=eta_.dtype).reshape(())
        gravity_ = jnp.asarray(gravity, dtype=eta_.dtype).reshape(())
        epoch = geometry.metric_epoch(eta_)
        x_spacing = jnp.maximum(geometry.x_center_distance, self.minimum_spacing)
        y_spacing = jnp.maximum(geometry.y_center_distance, self.minimum_spacing)
        x_cross_section = jnp.sum(epoch.x_face_area, axis=-1)
        y_cross_section = jnp.sum(epoch.y_face_area, axis=-1)
        x_depth = jnp.where(
            geometry.x_edge_length > 0.0,
            x_cross_section
            / jnp.where(geometry.x_edge_length > 0.0, geometry.x_edge_length, 1.0),
            0.0,
        )
        y_depth = jnp.where(
            geometry.y_edge_length > 0.0,
            y_cross_section
            / jnp.where(geometry.y_edge_length > 0.0, geometry.y_edge_length, 1.0),
            0.0,
        )
        x_advective_speed = jnp.zeros_like(x_depth)
        y_advective_speed = jnp.zeros_like(y_depth)
        if barotropic_transport is not None:
            x = jnp.asarray(barotropic_transport[0], dtype=eta_.dtype)
            y = jnp.asarray(barotropic_transport[1], dtype=eta_.dtype)
            if (
                x.shape != geometry.x_face_shape[:-1]
                or y.shape != geometry.y_face_shape[:-1]
            ):
                raise ValueError("Barotropic transport shapes are invalid.")
            x_advective_speed = jnp.where(
                x_cross_section > 0.0,
                jnp.abs(x) / jnp.where(x_cross_section > 0.0, x_cross_section, 1.0),
                0.0,
            )
            y_advective_speed = jnp.where(
                y_cross_section > 0.0,
                jnp.abs(y) / jnp.where(y_cross_section > 0.0, y_cross_section, 1.0),
                0.0,
            )
        x_wave_speed = jnp.sqrt(jnp.maximum(gravity_ * x_depth, 0.0))
        y_wave_speed = jnp.sqrt(jnp.maximum(gravity_ * y_depth, 0.0))
        x_courant = jnp.abs(dt) * (x_wave_speed + x_advective_speed) / x_spacing
        y_courant = jnp.abs(dt) * (y_wave_speed + y_advective_speed) / y_spacing
        unsplit_courant = jnp.maximum(jnp.max(x_courant), jnp.max(y_courant))
        if self.kind == "fixed-count":
            requested = jnp.asarray(self.fixed_count, dtype=jnp.int32)
        else:
            finite_courant = jnp.isfinite(unsplit_courant) & (unsplit_courant >= 0.0)
            count_courant = jnp.where(
                finite_courant,
                unsplit_courant,
                self.target_courant * (self.maximum_substeps + 1),
            )
            bounded_requested = jnp.minimum(
                jnp.ceil(count_courant / self.target_courant),
                jnp.asarray(self.maximum_substeps + 1, dtype=count_courant.dtype),
            ).astype(jnp.int32)
            requested = jnp.maximum(self.fixed_count, bounded_requested)
        capacity_valid = requested <= self.maximum_substeps
        safe_count = jnp.minimum(requested, self.maximum_substeps)
        active = jnp.arange(self.maximum_substeps) < safe_count
        substep = dt / jnp.maximum(safe_count, 1)
        substep_sizes = jnp.where(active, substep, 0.0)
        maximum_courant = unsplit_courant / jnp.maximum(safe_count, 1)
        finite = (
            epoch.finite
            & jnp.isfinite(dt)
            & jnp.isfinite(gravity_)
            & (gravity_ > 0.0)
            & jnp.isfinite(maximum_courant)
            & jnp.all(jnp.isfinite(x_spacing))
            & jnp.all(jnp.isfinite(y_spacing))
            & jnp.all(x_spacing > 0.0)
            & jnp.all(y_spacing > 0.0)
        )
        successful = finite & capacity_valid & epoch.valid
        return ExternalModeSubcycleSchedule(
            substep_sizes=substep_sizes,
            active_mask=active,
            count=requested,
            maximum_courant=maximum_courant,
            capacity_valid=capacity_valid,
            finite=finite,
            successful=successful,
            policy_id=self.policy_id,
        )


__all__ = [
    "ExternalModeSubcycleKind",
    "ExternalModeSubcyclePolicy",
    "ExternalModeSubcycleSchedule",
]
