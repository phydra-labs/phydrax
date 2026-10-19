"""Millimeter rectangle partitions and identity-bound controls for band tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from phydrax import SpatialCoordinateContract, units
from phydrax.geometry import (
    BRepPartitionPatch,
    BRepPartitionPolicy,
    BRepPartitionResult,
    BRepPartitionRole,
    partition_planar,
    PlanarEmbedding,
    PlanarMeshRegion,
    PlanarPartitionOperand,
    PlanarPartitionPlan,
)
from phydrax.meshing import (
    PatchControl,
    PlanarBandResult,
    RegionControl,
    RegionRole,
)


if TYPE_CHECKING:
    from phydrax.meshing import GmshProvider


BAND_EMBEDDING = PlanarEmbedding(
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
BAND_COORDINATES = SpatialCoordinateContract(units.MILLIMETER)


def rectangular_partition(
    directory: Path, widths: tuple[float, ...] = (1.0, 1.0)
) -> BRepPartitionResult:
    """Partition adjacent unit-height rectangles in the declared millimeter plane."""
    cursor = 0.0
    operands = []
    names = []
    for index, width in enumerate(widths):
        name = f"region-{index}"
        names.append(name)
        rectangle = PlanarMeshRegion(
            np.asarray(
                (
                    (cursor, 0.0),
                    (cursor + width, 0.0),
                    (cursor + width, 1.0),
                    (cursor, 1.0),
                ),
                dtype=np.float64,
            ),
            ((0, 1, 2, 3),),
            feature_id=name,
        )
        operands.append(PlanarPartitionOperand(name, rectangle, BRepPartitionRole.REGION))
        cursor += width
    return partition_planar(
        PlanarPartitionPlan(
            BAND_COORDINATES,
            BAND_EMBEDDING,
            tuple(operands),
            BRepPartitionPolicy(tuple(names)),
        ),
        destination=directory / "partition.brep",
        linear_deflection=0.05,
        angular_deflection=0.2,
    )


def interface_patch(
    partition: BRepPartitionResult, first: str, second: str
) -> BRepPartitionPatch:
    """Find the patch with the two declared region identities."""
    adjacent = tuple(sorted((first, second)))
    return next(
        patch
        for patch in partition.patches
        if tuple(sorted(patch.adjacent_region_ids)) == adjacent
    )


def band_meshing_controls(
    provider: GmshProvider, source: PlanarBandResult
) -> tuple[tuple[RegionControl, ...], tuple[PatchControl, ...]]:
    """Bind region and patch controls to the published band partition."""
    regions = tuple(
        RegionControl(
            provider.entity_scope(source, region.entity_ids),
            region.name,
            f"material:{region.name}",
            RegionRole.SOLID,
        )
        for region in source.partition.regions
    )
    patches = tuple(
        PatchControl(
            patch.name,
            provider.entity_scope(source, patch.entity_ids),
            patch.adjacent_region_ids,
        )
        for patch in source.partition.patches
        if len(set(patch.adjacent_region_ids)) == len(patch.adjacent_region_ids)
    )
    return regions, patches
