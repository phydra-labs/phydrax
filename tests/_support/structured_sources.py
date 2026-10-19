"""Build oriented tensor-block boundary charts for generation tests."""

from __future__ import annotations

import numpy as np

import phydrax as phx
from phydrax.geometry.brep import CircleCurve, LineCurve
from phydrax.meshing import TransfiniteBlock, TransfiniteCurveControl
from phydrax.meshing.qualification import prepare_family_reversed_multiblock_source


def square(n: int, /) -> TransfiniteBlock:
    return TransfiniteBlock(
        "square",
        tuple(
            TransfiniteCurveControl(LineCurve(origin, direction), name, (0.0, 1.0), n)
            for name, origin, direction in (
                ("left", (0.0, 0.0), (0.0, 1.0)),
                ("right", (1.0, 0.0), (0.0, 1.0)),
                ("bottom", (0.0, 0.0), (1.0, 0.0)),
                ("top", (0.0, 1.0), (1.0, 0.0)),
            )
        ),
    )


def annulus_sector(index: int, /) -> TransfiniteBlock:
    first = index * np.pi / 2.0
    last = (index + 1) * np.pi / 2.0
    name = f"sector:{index}"
    first_ray = np.asarray((np.cos(first), np.sin(first)), dtype=np.float64)
    last_ray = np.asarray((np.cos(last), np.sin(last)), dtype=np.float64)
    return TransfiniteBlock(
        name,
        (
            TransfiniteCurveControl(
                CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 1.0),
                "inner",
                (first, last),
                6,
            ),
            TransfiniteCurveControl(
                CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 2.0),
                "outer",
                (first, last),
                6,
            ),
            TransfiniteCurveControl(
                LineCurve(first_ray, first_ray), f"ray:{index}", (0.0, 1.0), 2
            ),
            TransfiniteCurveControl(
                LineCurve(last_ray, last_ray), f"ray:{(index + 1) % 4}", (0.0, 1.0), 2
            ),
        ),
    )


def material_multiblock_request() -> tuple[
    phx.meshing.NativeStructuredSource,
    phx.meshing.VolumeMeshingSpec,
    phx.meshing.NativeMeshingOptions,
]:
    """Reproduce the reversed two-material box request at resolution four."""
    source, interface_ids = prepare_family_reversed_multiblock_source(4, True)
    domain = source.domain
    assert isinstance(domain, phx.geometry.PiecewiseLinearDomain)
    capacity = 20000
    # Keep the qualification request's limits, including its live
    # storage and work allowances, when transferring fixture ownership.
    limits = phx.meshing.MeshingLimits(
        maximum_vertices=capacity,
        maximum_edges=12 * capacity,
        maximum_faces=12 * capacity,
        maximum_cells=capacity,
        maximum_connectivity_entries=64 * capacity,
        maximum_data_bytes=1024 * capacity,
        maximum_work_units=100 * capacity,
        maximum_cavity_cells=capacity,
        maximum_geometry_queries=100 * capacity,
        maximum_scratch_bytes=4096 * capacity,
        maximum_wall_seconds=120.0,
    )
    scope = phx.meshing.MeshingScope(
        source.source_id,
        source.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        "family-boundary",
        np.arange(domain.facets.shape[0], dtype=np.int64),
    )
    regions = tuple(
        phx.meshing.RegionControl(
            phx.meshing.MeshingScope(
                source.source_id,
                source.source_revision,
                phx.meshing.MeshingEntityKind.GEOMETRY,
                3,
                "reversed-source-materials",
                np.asarray((index,), dtype=np.int64),
            ),
            name,
            f"material-{name}",
            phx.meshing.RegionRole.SOLID,
        )
        for index, name in enumerate(("left", "right"))
    )
    patch = phx.meshing.PatchControl(
        "material-interface",
        phx.meshing.MeshingScope(
            source.source_id,
            source.source_revision,
            phx.meshing.MeshingEntityKind.GEOMETRY,
            2,
            "family-boundary",
            interface_ids,
        ),
        ("left", "right"),
    )
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3, 3, phx.meshing.CellFamilyPolicy(required=("hexahedron",)), geometry_order=1
        ),
        scope,
        phx.meshing.VolumeFillStrategy.MULTIZONE,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope,
                0.25,
                maximum_size=0.5,
                strength=phx.meshing.SizeControlStrength.HARD,
            ),
        ),
        size_compliance=phx.meshing.SizeCompliancePolicy(
            relative_tolerance=0.95, target_statistics=("p50",)
        ),
        limits=limits,
        region_controls=regions,
        patch_controls=(patch,),
    )
    return (
        source,
        specification,
        phx.meshing.NativeMeshingOptions("structured_transfinite"),
    )
