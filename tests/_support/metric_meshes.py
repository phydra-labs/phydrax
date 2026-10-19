"""Deterministically perturbed planar carriers and constant tensor requests."""

from __future__ import annotations

import numpy as np

from phydrax import SpatialCoordinateContract
from phydrax.discretization import CellMesh, PolygonalConnectivity
from phydrax.meshing import (
    CellMeshingResult,
    certify_cell_mesh,
    MeshingEntityKind,
    MeshingScope,
    MeshMetricField,
    MeshPatch,
    MeshZone,
    MeshZoneRole,
)
from tests._support.meshing_fixtures import mesh_scope


# Source: the numerical regression fixtures in
# a6827a9e5:tests/unit/meshing/test_tetra_metric.py. These distinct allowances
# belong to geometry, finite-volume conservation, FE projection, or the
# one-sided trial-image reference; their values remain those of that source.
TETRA_GEOMETRY_TOLERANCE = 1.0e-12
TETRA_VOLUME_TOLERANCE = 1.0e-14
TETRA_DG_TOLERANCE = 1.0e-11
TETRA_FIELD_TOLERANCE = 1.0e-10
TETRA_TRIAL_IMAGE_TOLERANCE = 1.0e-15
TETRA_RADIUS_DIFFERENCE_STEP = 1.0e-6
TETRA_RADIUS_DIFFERENCE_TOLERANCE = 1.0e-5
# Source: d2dfcb941:tests/unit/meshing/test_metric.py. These tolerances
# distinguish spectral roundoff, iterative gradation, and interpolation.
METRIC_ROUNDOFF_TOLERANCE = 1.0e-12
METRIC_ITERATION_TOLERANCE = 1.0e-9
METRIC_PSD_TOLERANCE = 1.0e-8
METRIC_INTERPOLATION_TOLERANCE = 1.0e-10
# The isotropic differential fixture permits 2e-12, a 9008-eps envelope
# for spectral log/exp and their differentiated composition at unit scale.
METRIC_DIFFERENTIAL_TOLERANCE = 2.0e-12


def sinusoidal_triangle_grid(
    count: int, *, perturbation: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Retain the local-metric grid's row order, boundary, and sinusoidal offsets."""
    x, y = np.meshgrid(
        np.linspace(0.0, 1.0, count + 1, dtype=np.float64),
        np.linspace(0.0, 1.0, count + 1, dtype=np.float64),
    )
    points = np.stack((x.ravel(), y.ravel()), axis=1)
    interior = np.all((points > 0.0) & (points < 1.0), axis=1)
    offsets = np.where(
        interior[:, None],
        perturbation * np.asarray((1.0, -0.6), dtype=np.float64),
        0.0,
    )
    shift = np.sin(7.0 * points[:, :1] + 3.0 * points[:, 1:])
    points = points + offsets * shift / count
    index = np.arange((count + 1) ** 2).reshape((count + 1, count + 1))
    a = index[:-1, :-1].ravel()
    b = index[:-1, 1:].ravel()
    c = index[1:, 1:].ravel()
    d = index[1:, :-1].ravel()
    triangles = np.concatenate((np.stack((a, b, c), 1), np.stack((a, c, d), 1)))
    return points, triangles


def planar_metric_source(
    count: int = 4, *, perturbation: float = 0.0, organized: bool = False
) -> CellMeshingResult:
    """Certify a planar carrier, optionally retaining its two regions and bottom patch."""
    points, triangles = sinusoidal_triangle_grid(count, perturbation=perturbation)
    mesh = CellMesh.from_triangles(points, triangles)
    mesh = certify_cell_mesh(mesh, SpatialCoordinateContract.si()).mesh
    if not organized:
        return certify_cell_mesh(mesh, SpatialCoordinateContract.si())
    coordinates = np.asarray(mesh.coordinates)
    cells = np.asarray(mesh.entity_set(2).entity_ids)
    centroids = np.concatenate(
        [coordinates[np.asarray(block.vertices)].mean(axis=1) for block in mesh.blocks]
    )
    assert isinstance(mesh.connectivity, PolygonalConnectivity)
    edges = np.asarray(mesh.connectivity.edges)
    bottom = np.all(coordinates[edges][:, :, 1] == 0.0, axis=1)
    zones = (
        MeshZone(
            "left", MeshZoneRole.REGION, mesh_scope(mesh, 2, cells[centroids[:, 0] < 0.5])
        ),
        MeshZone(
            "right",
            MeshZoneRole.REGION,
            mesh_scope(mesh, 2, cells[centroids[:, 0] > 0.5]),
        ),
    )
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids)
    patches = (MeshPatch("bottom", mesh_scope(mesh, 1, edge_ids[bottom])),)
    return certify_cell_mesh(
        mesh, SpatialCoordinateContract.si(), patches=patches, zones=zones
    )


def constant_mesh_metric(
    source: CellMeshingResult, tensor: np.ndarray
) -> MeshMetricField:
    """Bind one constant tensor to every vertex of the certified carrier."""
    mesh = source.mesh
    vertices = mesh.entity_set(0)
    values = np.broadcast_to(tensor, (vertices.entity_ids.shape[0], 2, 2))
    return MeshMetricField(
        mesh_scope(mesh, 0), values, minimum_size=1.0e-3, maximum_size=10.0
    )


def tensor_metric_scope(count: int) -> MeshingScope:
    """Bind tensor rows to the metric-only fixture's stable scope identifiers."""
    return MeshingScope(
        "mesh",
        "r1",
        MeshingEntityKind.MESH,
        0,
        "vertices",
        np.arange(count, dtype=np.int64),
    )


def metric_grid(count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a Cartesian metric grid with horizontal, vertical, and diagonal edges."""
    axis = np.linspace(0.0, 1.0, count, dtype=np.float64)
    first, second = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((first.ravel(), second.ravel()), axis=1)
    rows = np.arange(count * count, dtype=np.int64).reshape(count, count)
    edges = np.concatenate(
        (
            np.stack((rows[:-1, :].ravel(), rows[1:, :].ravel()), axis=1),
            np.stack((rows[:, :-1].ravel(), rows[:, 1:].ravel()), axis=1),
            np.stack((rows[:-1, :-1].ravel(), rows[1:, 1:].ravel()), axis=1),
        )
    )
    return points, edges


def anisotropic_metric_samples(count: int, seed: int) -> np.ndarray:
    """Return the seeded rotated two-dimensional tensors used by metric fixtures."""
    generator = np.random.default_rng(seed)
    angles = generator.uniform(0.0, np.pi, count)
    sizes = generator.uniform(0.05, 0.4, count)
    sizes[0] = 0.01
    rotation = np.stack(
        (
            np.stack((np.cos(angles), -np.sin(angles)), axis=-1),
            np.stack((np.sin(angles), np.cos(angles)), axis=-1),
        ),
        axis=-2,
    )
    eigenvalues = np.stack((sizes**-2, (3.0 * sizes) ** -2), axis=-1)
    return (rotation * eigenvalues[:, None, :]) @ np.swapaxes(rotation, -1, -2)
