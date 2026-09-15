#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._admissibility import AdmissibilityHeader, AdmissibilityReason
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry._eroded_domain import (
    AbstractFiniteRadiusWallPlan,
    FiniteRadiusErosionEvaluation,
    FiniteRadiusErosionReason,
)


class DLDTopology(StrictModule, NonTrainableState):
    row_count: int = eqx.field(static=True)
    column_count: int = eqx.field(static=True)
    period_rows: int = eqx.field(static=True)
    outlet_count: int = eqx.field(static=True)
    particle_capacity: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        row_count: int,
        column_count: int,
        period_rows: int,
        outlet_count: int,
        particle_capacity: int,
    ) -> None:
        values = tuple(
            int(value)
            for value in (
                row_count,
                column_count,
                period_rows,
                outlet_count,
                particle_capacity,
            )
        )
        if (
            values[0] <= 0
            or values[1] <= 0
            or values[2] <= 1
            or values[3] <= 0
            or values[4] <= 0
        ):
            raise ValueError("DLD topology capacities must be positive.")
        if values[0] < values[2]:
            raise ValueError("DLD row count must cover at least one shift period.")
        (
            self.row_count,
            self.column_count,
            self.period_rows,
            self.outlet_count,
            self.particle_capacity,
        ) = values
        self.topology_id = canonical_fingerprint(
            {"kind": "dld-topology", "values": values}
        )


class DLDDesign(StrictModule, NonTrainableState):
    post_radius: float = eqx.field(static=True)
    axial_pitch: float = eqx.field(static=True)
    lateral_pitch: float = eqx.field(static=True)
    row_shift: float = eqx.field(static=True)
    channel_lower: float = eqx.field(static=True)
    channel_upper: float = eqx.field(static=True)
    first_row_x: float = eqx.field(static=True)
    outlet_x: float = eqx.field(static=True)
    depth: float = eqx.field(static=True)
    length_unit_id: str = eqx.field(static=True)
    design_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        post_radius: float,
        axial_pitch: float,
        lateral_pitch: float,
        row_shift: float,
        channel_lower: float,
        channel_upper: float,
        first_row_x: float,
        outlet_x: float,
        depth: float,
        length_unit_id: str,
    ) -> None:
        values = tuple(
            float(value)
            for value in (
                post_radius,
                axial_pitch,
                lateral_pitch,
                row_shift,
                channel_lower,
                channel_upper,
                first_row_x,
                outlet_x,
                depth,
            )
        )
        unit = str(length_unit_id)
        if (
            any(not np.isfinite(value) for value in values)
            or min(values[0], values[1], values[2], values[8]) <= 0.0
            or values[3] <= 0.0
            or values[5] <= values[4]
            or values[7] <= values[6]
            or 2.0 * values[0] >= min(values[1], values[2])
            or not unit
        ):
            raise ValueError("DLD geometry values are invalid.")
        (
            self.post_radius,
            self.axial_pitch,
            self.lateral_pitch,
            self.row_shift,
            self.channel_lower,
            self.channel_upper,
            self.first_row_x,
            self.outlet_x,
            self.depth,
        ) = values
        self.length_unit_id = unit
        self.design_id = canonical_fingerprint(
            {"kind": "dld-design", "values": values, "length_unit": unit}
        )


class DLDGeometryPlan(AbstractFiniteRadiusWallPlan, NonTrainableState):
    """Exact disjoint-circle and straight-sidewall DLD center geometry."""

    topology: DLDTopology
    design: DLDDesign
    post_centers: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, topology: DLDTopology, design: DLDDesign, /) -> None:
        if not isinstance(topology, DLDTopology) or not isinstance(design, DLDDesign):
            raise TypeError("DLD geometry requires topology and design.")
        expected_shift = design.lateral_pitch / topology.period_rows
        if not np.isclose(design.row_shift, expected_shift, rtol=1.0e-10, atol=1.0e-14):
            raise ValueError("DLD row shift must close exactly over period_rows.")
        rows = np.arange(topology.row_count, dtype=float)
        columns = np.arange(topology.column_count, dtype=float)
        x = design.first_row_x + rows * design.axial_pitch
        offsets = (
            np.arange(topology.row_count) % topology.period_rows
        ) * design.row_shift
        base_y = design.channel_lower + design.post_radius
        y = base_y + columns[None, :] * design.lateral_pitch + offsets[:, None]
        centers = np.stack(
            (
                np.broadcast_to(x[:, None], y.shape),
                y,
            ),
            axis=-1,
        ).reshape((-1, 2))
        if (
            np.min(centers[:, 1] - design.post_radius) < design.channel_lower
            or np.max(centers[:, 1] + design.post_radius) > design.channel_upper
            or np.max(centers[:, 0] + design.post_radius) >= design.outlet_x
        ):
            raise ValueError("DLD posts intersect sidewalls or the outlet plane.")
        self.topology = topology
        self.design = design
        self.post_centers = jnp.asarray(centers)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dld-exact-circular-post-geometry",
                "topology": topology.topology_id,
                "design": design.design_id,
                "post_centers": array_tree_fingerprint(centers),
            }
        )

    @property
    def dimension(self) -> int:
        return 2

    @property
    def post_count(self) -> int:
        return self.post_centers.shape[0]

    @property
    def minimum_gap(self) -> float:
        return min(
            self.design.axial_pitch - 2.0 * self.design.post_radius,
            self.design.lateral_pitch - 2.0 * self.design.post_radius,
        )

    def evaluate(
        self, points: ArrayLike, radii: ArrayLike, /
    ) -> FiniteRadiusErosionEvaluation:
        point = jnp.asarray(points)
        radius = jnp.asarray(radii, dtype=point.dtype)
        if point.ndim < 2 or point.shape[-1] != 2 or radius.shape != point.shape[:-1]:
            raise ValueError("DLD points and radii have incompatible shapes.")
        relative = point[..., None, :] - self.post_centers.astype(point.dtype)
        center_distance = jnp.sqrt(jnp.sum(relative**2, axis=-1))
        safe_distance = jnp.maximum(center_distance, jnp.finfo(point.dtype).tiny)
        post_clearance = center_distance - self.design.post_radius - radius[..., None]
        lower_clearance = point[..., 1] - self.design.channel_lower - radius
        upper_clearance = self.design.channel_upper - point[..., 1] - radius
        all_clearance = jnp.concatenate(
            (
                post_clearance,
                lower_clearance[..., None],
                upper_clearance[..., None],
            ),
            axis=-1,
        )
        selected = jnp.argmin(all_clearance, axis=-1)
        clearance = jnp.take_along_axis(all_clearance, selected[..., None], axis=-1)[
            ..., 0
        ]
        sorted_clearance = jnp.sort(all_clearance, axis=-1)
        tie_margin = sorted_clearance[..., 1] - sorted_clearance[..., 0]
        selected_post = jnp.clip(selected, 0, self.post_count - 1)
        selected_relative = jnp.take_along_axis(
            relative,
            selected_post[..., None, None],
            axis=-2,
        )[..., 0, :]
        selected_distance = jnp.take_along_axis(
            safe_distance, selected_post[..., None], axis=-1
        )[..., 0]
        post_normal = selected_relative / selected_distance[..., None]
        lower_normal = jnp.broadcast_to(
            jnp.asarray((0.0, 1.0), dtype=point.dtype), point.shape
        )
        upper_normal = jnp.broadcast_to(
            jnp.asarray((0.0, -1.0), dtype=point.dtype), point.shape
        )
        inward_normal = jnp.where(
            (selected < self.post_count)[..., None],
            post_normal,
            jnp.where(
                (selected == self.post_count)[..., None], lower_normal, upper_normal
            ),
        )
        selected_center = self.post_centers[selected_post]
        post_closest = selected_center + self.design.post_radius * post_normal
        lower_closest = point.at[..., 1].set(self.design.channel_lower)
        upper_closest = point.at[..., 1].set(self.design.channel_upper)
        closest = jnp.where(
            (selected < self.post_count)[..., None],
            post_closest,
            jnp.where(
                (selected == self.post_count)[..., None], lower_closest, upper_closest
            ),
        )
        scale = jnp.maximum(jnp.max(jnp.abs(point), axis=-1), 1.0)
        tolerance = 128.0 * jnp.finfo(point.dtype).eps * scale
        unique = tie_margin > tolerance
        finite = (
            jnp.all(jnp.isfinite(point), axis=-1)
            & jnp.isfinite(radius)
            & jnp.isfinite(clearance)
            & jnp.all(jnp.isfinite(inward_normal), axis=-1)
        )
        supported = finite & (radius >= 0.0) & unique & (clearance >= 0.0)
        reasons = jnp.zeros(radius.shape, dtype=jnp.uint32)
        reasons = jnp.where(
            finite,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.NONFINITE), jnp.uint32),
        )
        reasons = jnp.where(
            unique,
            reasons,
            reasons
            | jnp.asarray(
                int(FiniteRadiusErosionReason.NONUNIQUE_CLOSEST_POINT), jnp.uint32
            ),
        )
        reasons = jnp.where(
            (radius >= 0.0) & (clearance >= 0.0),
            reasons,
            reasons
            | jnp.asarray(
                int(FiniteRadiusErosionReason.CENTER_OUTSIDE_ERODED_DOMAIN), jnp.uint32
            ),
        )
        header = AdmissibilityHeader(
            jnp.where(supported, jnp.minimum(clearance, tie_margin), -1.0),
            reasons,
            self.plan_id,
            canonical_fingerprint(
                {"kind": "dld-geometry-evidence", "geometry": self.plan_id}
            ),
        )
        return FiniteRadiusErosionEvaluation(
            clearance,
            closest,
            inward_normal,
            selected.astype(jnp.int32),
            tie_margin,
            header,
            self.plan_id,
        )


__all__ = ["DLDDesign", "DLDGeometryPlan", "DLDTopology"]
