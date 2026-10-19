"""Periodic provider request fixtures with declared lattice and material strata."""

from itertools import product

import numpy as np

import phydrax as phx
from phydrax.discretization import PeriodicCell
from phydrax.meshing._periodic import require_periodic_topology, simplex_entity_corners


def periodic_lattice_points(count: int) -> np.ndarray:
    axis = np.arange(count) / count
    return np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape((-1, 2))


def skew_periodic_material_request(
    limits: phx.meshing.MeshingLimits,
) -> tuple[
    phx.meshing.NativePeriodicSource,
    phx.meshing.SurfaceMeshingSpec,
    phx.meshing.UniformSizeControl,
]:
    M = phx.meshing
    vectors = np.asarray(((1.0, 0.0), (0.25, 1.0)), dtype=np.float64)
    cell = PeriodicCell(vectors)
    count = 4
    lattice = np.stack(
        np.meshgrid(np.arange(count), np.arange(count), indexing="ij"), axis=-1
    ).reshape((-1, 2))
    corners, regions = [], []
    for i, j in product(range(count), range(count)):
        for triangle in (
            ((i, j), (i + 1, j), (i + 1, j + 1)),
            ((i, j), (i + 1, j + 1), (i, j + 1)),
        ):
            corners.append(triangle)
            regions.append(0 if i < count // 2 else 1)
    integer = np.asarray(corners, dtype=np.int64)
    domain = M.publish_periodic_simplices(
        (lattice / count) @ vectors,
        (integer[..., 0] % count) * count + integer[..., 1] % count,
        integer // count,
        cell,
    )
    source = M.NativePeriodicSource(
        domain, "skew-material", "r1", cell_regions=np.asarray(regions)
    )
    edges = simplex_entity_corners(domain, 1)
    fractional = np.linalg.solve(vectors.T, np.asarray(domain.coordinates).T).T
    x = fractional[edges, 0]
    interface = (x[:, 0] == x[:, 1]) & ((x[:, 0] % 1.0 == 0.0) | (x[:, 0] % 1.0 == 0.5))
    quotient_edges = np.unique(
        np.asarray(require_periodic_topology(domain).orbits(1)[0])[interface]
    )
    feature = M.ProtectedFeature(
        M.MeshingScope(
            source.source_id,
            source.source_revision,
            M.MeshingEntityKind.GEOMETRY,
            1,
            "interface-orbits",
            quotient_edges,
        ),
        M.FeatureKind.MATERIAL_INTERFACE,
    )
    scope = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        "regions",
        np.asarray((0, 1), dtype=np.int64),
    )
    control = M.UniformSizeControl(
        scope, 0.25, maximum_size=0.375, strength=M.SizeControlStrength.HARD
    )
    materials = tuple(
        M.RegionControl(
            M.MeshingScope(
                source.source_id,
                source.source_revision,
                M.MeshingEntityKind.GEOMETRY,
                2,
                "regions",
                np.asarray((region,), dtype=np.int64),
            ),
            f"material-{region}",
            f"material-{region}",
            M.RegionRole.USER,
        )
        for region in (0, 1)
    )
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 2, M.CellFamilyPolicy(required=("triangle",))),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        size_controls=(control,),
        protected_features=(feature,),
        region_controls=materials,
        limits=limits,
    )
    return source, specification, control


def periodic_seed_request() -> tuple[
    phx.meshing.NativePeriodicSource, phx.meshing.SurfaceMeshingSpec
]:
    """Return the unit-square torus soft-size request used for publication checks."""
    M = phx.meshing
    source = M.NativePeriodicSource(
        M.PeriodicPointOrbits(periodic_lattice_points(3), PeriodicCell(np.eye(2))),
        "torus",
        "r1",
    )
    scope = M.MeshingScope(
        "torus", "r1", M.MeshingEntityKind.GEOMETRY, 2, "regions", np.asarray((0,))
    )
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 2, M.CellFamilyPolicy(required=("triangle",))),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
        ),
        size_controls=(
            M.UniformSizeControl(
                scope, 0.25, maximum_size=0.35, strength=M.SizeControlStrength.SOFT
            ),
        ),
    )
    return source, specification
