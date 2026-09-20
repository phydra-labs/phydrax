#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host-prepared logical patch boxes and bounded static execution buckets.

Logical patch identity is independent of an execution lane, a bucket, and a device.
The bucket layer bounds compiled shapes without making ragged arrays part of the
JAX runtime contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class LogicalPatchBox(StrictModule, NonTrainableState):
    """Half-open reference-cell box at one physical AMR level."""

    level: int = eqx.field(static=True)
    lower: tuple[int, ...] = eqx.field(static=True)
    upper: tuple[int, ...] = eqx.field(static=True)
    box_id: str = eqx.field(static=True)

    def __init__(
        self,
        level: int,
        lower: Sequence[int],
        upper: Sequence[int],
        /,
    ):
        level_ = int(level)
        lower_ = tuple(lower)
        upper_ = tuple(upper)
        if level_ < 0 or not lower_ or len(lower_) != len(upper_):
            raise ValueError("Logical patch box level and bounds must be valid.")
        if any(
            start < 0 or stop <= start for start, stop in zip(lower_, upper_, strict=True)
        ):
            raise ValueError("Logical patch boxes require non-negative half-open bounds.")
        self.level = level_
        self.lower = lower_
        self.upper = upper_
        self.box_id = canonical_fingerprint(
            {
                "kind": "amr-logical-patch-box",
                "level": level_,
                "lower": lower_,
                "upper": upper_,
            }
        )

    @property
    def dimension(self) -> int:
        return len(self.lower)

    @property
    def extent(self) -> tuple[int, ...]:
        return tuple(
            stop - start for start, stop in zip(self.lower, self.upper, strict=True)
        )

    @property
    def cell_count(self) -> int:
        return prod(self.extent)

    def contains_cell(self, coordinate: Sequence[int], /) -> bool:
        values = tuple(coordinate)
        return len(values) == self.dimension and all(
            start <= value < stop
            for value, start, stop in zip(values, self.lower, self.upper, strict=True)
        )

    def contains_box(self, other: "LogicalPatchBox", /) -> bool:
        return (
            self.level == other.level
            and self.dimension == other.dimension
            and all(
                left <= other_left and other_right <= right
                for left, right, other_left, other_right in zip(
                    self.lower,
                    self.upper,
                    other.lower,
                    other.upper,
                    strict=True,
                )
            )
        )

    def overlaps(self, other: "LogicalPatchBox", /) -> bool:
        return (
            self.level == other.level
            and self.dimension == other.dimension
            and all(
                max(left, other_left) < min(right, other_right)
                for left, right, other_left, other_right in zip(
                    self.lower,
                    self.upper,
                    other.lower,
                    other.upper,
                    strict=True,
                )
            )
        )

    def intersection(self, other: "LogicalPatchBox", /) -> "LogicalPatchBox | None":
        if self.level != other.level or self.dimension != other.dimension:
            return None
        lower = tuple(
            max(left, right) for left, right in zip(self.lower, other.lower, strict=True)
        )
        upper = tuple(
            min(left, right) for left, right in zip(self.upper, other.upper, strict=True)
        )
        if any(stop <= start for start, stop in zip(lower, upper, strict=True)):
            return None
        return LogicalPatchBox(self.level, lower, upper)

    def touches_face(self, other: "LogicalPatchBox", axis: int, side: int, /) -> bool:
        axis_ = int(axis)
        side_ = int(side)
        if (
            self.level != other.level
            or self.dimension != other.dimension
            or axis_ < 0
            or axis_ >= self.dimension
            or side_ not in (-1, 1)
        ):
            return False
        face_match = (
            self.lower[axis_] == other.upper[axis_]
            if side_ < 0
            else self.upper[axis_] == other.lower[axis_]
        )
        if not face_match:
            return False
        return all(
            axis == axis_
            or max(self.lower[axis], other.lower[axis])
            < min(self.upper[axis], other.upper[axis])
            for axis in range(self.dimension)
        )

    def grow(self, widths: int | Sequence[int], /) -> "LogicalPatchBox":
        widths_ = (
            (int(widths),) * self.dimension if isinstance(widths, int) else tuple(widths)
        )
        if len(widths_) != self.dimension or any(value < 0 for value in widths_):
            raise ValueError("Patch growth widths must be non-negative and axis-aligned.")
        return LogicalPatchBox(
            self.level,
            tuple(
                max(0, start - width)
                for start, width in zip(self.lower, widths_, strict=True)
            ),
            tuple(stop + width for stop, width in zip(self.upper, widths_, strict=True)),
        )

    def refine(self, ratio: int, /) -> "LogicalPatchBox":
        ratio_ = int(ratio)
        if ratio_ <= 1:
            raise ValueError("Logical patch refinement ratio must exceed one.")
        return LogicalPatchBox(
            self.level + 1,
            tuple(value * ratio_ for value in self.lower),
            tuple(value * ratio_ for value in self.upper),
        )

    def coarsen(self, ratio: int, /) -> "LogicalPatchBox":
        ratio_ = int(ratio)
        if (
            self.level == 0
            or ratio_ <= 1
            or any(value % ratio_ for value in (*self.lower, *self.upper))
        ):
            raise ValueError(
                "Logical patch box is not exactly coarsenable at this ratio."
            )
        return LogicalPatchBox(
            self.level - 1,
            tuple(value // ratio_ for value in self.lower),
            tuple(value // ratio_ for value in self.upper),
        )


class PatchShapeSignature(StrictModule, NonTrainableState):
    """One finite static envelope signature admitted at an AMR level."""

    envelope_shape: tuple[int, ...] = eqx.field(static=True)
    halo_width: tuple[int, ...] = eqx.field(static=True)
    alignment: tuple[int, ...] = eqx.field(static=True)
    signature_id: str = eqx.field(static=True)

    def __init__(
        self,
        envelope_shape: Sequence[int],
        /,
        *,
        halo_width: int | Sequence[int] = 1,
        alignment: int | Sequence[int] = 1,
    ):
        shape = tuple(envelope_shape)
        halo = (
            (int(halo_width),) * len(shape)
            if isinstance(halo_width, int)
            else tuple(halo_width)
        )
        alignment_ = (
            (int(alignment),) * len(shape)
            if isinstance(alignment, int)
            else tuple(alignment)
        )
        if (
            not shape
            or any(value <= 0 for value in shape)
            or len(halo) != len(shape)
            or any(value < 0 for value in halo)
            or len(alignment_) != len(shape)
            or any(value <= 0 for value in alignment_)
            or any(
                value % aligned for value, aligned in zip(shape, alignment_, strict=True)
            )
        ):
            raise ValueError("Patch shape envelope, halo, and alignment must be valid.")
        self.envelope_shape = shape
        self.halo_width = halo
        self.alignment = alignment_
        self.signature_id = canonical_fingerprint(
            {
                "kind": "amr-patch-shape-signature",
                "envelope": shape,
                "halo": halo,
                "alignment": alignment_,
            }
        )

    def admits(self, extent: Sequence[int], /) -> bool:
        values = tuple(extent)
        return len(values) == len(self.envelope_shape) and all(
            0 < value <= envelope and value % alignment == 0
            for value, envelope, alignment in zip(
                values,
                self.envelope_shape,
                self.alignment,
                strict=True,
            )
        )


class PatchBucketPlan(StrictModule, NonTrainableState):
    """Static lane capacity for one shape signature."""

    signature: PatchShapeSignature
    lane_capacity: int = eqx.field(static=True)
    bucket_id: str = eqx.field(static=True)

    def __init__(self, signature: PatchShapeSignature, lane_capacity: int, /):
        capacity = int(lane_capacity)
        if not isinstance(signature, PatchShapeSignature) or capacity <= 0:
            raise ValueError(
                "Patch bucket requires a signature and positive lane capacity."
            )
        self.signature = signature
        self.lane_capacity = capacity
        self.bucket_id = canonical_fingerprint(
            {
                "kind": "amr-patch-bucket",
                "signature": signature.signature_id,
                "lane_capacity": capacity,
            }
        )


class BlockHierarchyCapacityPlan(StrictModule, NonTrainableState):
    """Bounded canonical entity and route capacities for one hierarchy."""

    entity_capacities: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    route_capacities: tuple[int, ...] = eqx.field(static=True)
    capacity_id: str = eqx.field(static=True)

    def __init__(
        self,
        entity_capacities: Sequence[Sequence[int]],
        route_capacities: Sequence[int],
        /,
    ):
        entities = tuple(tuple(row) for row in entity_capacities)
        routes = tuple(route_capacities)
        if (
            not entities
            or any(not row or any(value <= 0 for value in row) for row in entities)
            or not routes
            or any(value <= 0 for value in routes)
        ):
            raise ValueError("Hierarchy entity and route capacities must be positive.")
        self.entity_capacities = entities
        self.route_capacities = routes
        self.capacity_id = canonical_fingerprint(
            {
                "kind": "amr-hierarchy-capacity",
                "entities": entities,
                "routes": routes,
            }
        )


__all__ = [
    "BlockHierarchyCapacityPlan",
    "LogicalPatchBox",
    "PatchBucketPlan",
    "PatchShapeSignature",
]
