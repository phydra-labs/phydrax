#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Material-neutral final-state vertical-extrusion process recipes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ..simplicial import PlanarMeshRegion


@dataclass(frozen=True, slots=True, order=True)
class ZInterval:
    """Finite, nonempty vertical interval in the stack coordinate unit."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.lower, bool)
            or not isinstance(self.lower, Real)
            or isinstance(self.upper, bool)
            or not isinstance(self.upper, Real)
        ):
            raise TypeError("Z interval bounds must be real numbers.")
        lower = float(self.lower)
        upper = float(self.upper)
        if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
            raise ValueError("A Z interval requires finite bounds with upper > lower.")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def height(self) -> float:
        return self.upper - self.lower


@dataclass(frozen=True, slots=True)
class StackRegion:
    """Named vertical extrusion and its explicit overlap precedence."""

    region_id: str
    footprint: PlanarMeshRegion
    z_interval: ZInterval
    precedence: int
    stack_region_id: str = field(init=False)

    def __post_init__(self) -> None:
        identifier = str(self.region_id)
        if not identifier:
            raise ValueError("region_id must be nonempty.")
        if not isinstance(self.footprint, PlanarMeshRegion):
            raise TypeError("footprint must be a PlanarMeshRegion.")
        if not isinstance(self.z_interval, ZInterval):
            raise TypeError("z_interval must be a ZInterval.")
        if isinstance(self.precedence, bool) or not isinstance(self.precedence, Integral):
            raise TypeError("precedence must be an integer.")
        precedence = int(self.precedence)
        geometry = _footprint_payload(self.footprint)
        object.__setattr__(self, "region_id", identifier)
        object.__setattr__(self, "precedence", precedence)
        object.__setattr__(
            self,
            "stack_region_id",
            canonical_fingerprint(
                {
                    "kind": "process-stack-region",
                    "region_id": identifier,
                    "footprint": geometry,
                    "z_interval": [self.z_interval.lower, self.z_interval.upper],
                    "precedence": precedence,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class StackVoid:
    """Named vertical void applied only to explicitly targeted regions."""

    void_id: str
    footprint: PlanarMeshRegion
    z_interval: ZInterval
    target_region_ids: tuple[str, ...]
    stack_void_id: str = field(init=False)

    def __post_init__(self) -> None:
        identifier = str(self.void_id)
        if not identifier:
            raise ValueError("void_id must be nonempty.")
        if not isinstance(self.footprint, PlanarMeshRegion):
            raise TypeError("footprint must be a PlanarMeshRegion.")
        if not isinstance(self.z_interval, ZInterval):
            raise TypeError("z_interval must be a ZInterval.")
        targets = tuple(str(value) for value in self.target_region_ids)
        if (
            not targets
            or any(not value for value in targets)
            or len(set(targets)) != len(targets)
        ):
            raise ValueError(
                "target_region_ids must contain unique nonempty region identifiers."
            )
        object.__setattr__(self, "void_id", identifier)
        object.__setattr__(self, "target_region_ids", targets)
        object.__setattr__(
            self,
            "stack_void_id",
            canonical_fingerprint(
                {
                    "kind": "process-stack-void",
                    "void_id": identifier,
                    "footprint": _footprint_payload(self.footprint),
                    "z_interval": [self.z_interval.lower, self.z_interval.upper],
                    "target_region_ids": list(targets),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class ProcessStack:
    """Closed final-state Boolean recipe; it contains no material or process physics."""

    coordinate_contract: SpatialCoordinateContract
    regions: tuple[StackRegion, ...]
    voids: tuple[StackVoid, ...] = ()
    stack_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be a SpatialCoordinateContract.")
        if self.coordinate_contract.coordinate_system != "cartesian":
            raise ValueError(
                "Vertical process stacks require a Cartesian coordinate contract."
            )
        regions = tuple(self.regions)
        voids = tuple(self.voids)
        if not regions or not all(isinstance(value, StackRegion) for value in regions):
            raise ValueError("A process stack requires at least one StackRegion.")
        if not all(isinstance(value, StackVoid) for value in voids):
            raise TypeError("voids must contain StackVoid values.")
        region_ids = tuple(value.region_id for value in regions)
        void_ids = tuple(value.void_id for value in voids)
        if len(set(region_ids)) != len(region_ids):
            raise ValueError("Process stack region IDs must be unique.")
        if len(set(void_ids)) != len(void_ids):
            raise ValueError("Process stack void IDs must be unique.")
        if set(region_ids).intersection(void_ids):
            raise ValueError(
                "Process stack region and void operand IDs must be disjoint."
            )
        precedences = tuple(value.precedence for value in regions)
        if len(set(precedences)) != len(precedences):
            raise ValueError("Every process stack region requires distinct precedence.")
        known_regions = frozenset(region_ids)
        for void in voids:
            unknown = tuple(
                target for target in void.target_region_ids if target not in known_regions
            )
            if unknown:
                raise ValueError(
                    f"Process void {void.void_id!r} targets unknown regions {unknown!r}."
                )
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "voids", voids)
        object.__setattr__(
            self,
            "stack_id",
            canonical_fingerprint(
                {
                    "kind": "process-stack",
                    "coordinate_contract": self.coordinate_contract.spatial_id,
                    "regions": [value.stack_region_id for value in regions],
                    "voids": [value.stack_void_id for value in voids],
                    "region_precedence": list(self.region_precedence),
                }
            ),
        )

    @property
    def region_precedence(self) -> tuple[str, ...]:
        """Region IDs from highest to lowest overlap precedence."""

        return tuple(
            value.region_id
            for value in sorted(
                self.regions, key=lambda region: region.precedence, reverse=True
            )
        )


def _footprint_payload(region: PlanarMeshRegion, /) -> dict[str, object]:
    vertices = np.asarray(region.vertices, dtype=float)
    edges = np.asarray(region.edges, dtype=np.int64)
    offsets = np.asarray(region.loop_offsets, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or not np.all(np.isfinite(vertices)):
        raise ValueError("Process stack footprints require finite planar vertices.")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("Process stack footprint edges must have shape (N, 2).")
    if offsets.ndim != 1 or offsets.size < 2:
        raise ValueError("Process stack footprint loop offsets are invalid.")
    loops = [
        edges[int(offsets[index]) : int(offsets[index + 1]), 0].tolist()
        for index in range(offsets.size - 1)
    ]
    return {
        "vertices": vertices.tolist(),
        "loops": loops,
    }


__all__ = ["ProcessStack", "StackRegion", "StackVoid", "ZInterval"]
