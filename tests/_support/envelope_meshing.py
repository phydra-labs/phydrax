# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Envelope source builders and independent affine-field qualification references."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx
from phydrax.geometry import SurfaceMetadata
from phydrax.meshing import (
    EnvelopeFeaturePermission,
    EnvelopeTopologyPermission,
    NativeSurfaceEnvelopeSource,
    RawTriangleSoup,
    SurfaceEnvelopePolicy,
)


M = phx.meshing
_SI = phx.SpatialCoordinateContract.si()


def envelope_metadata() -> SurfaceMetadata:
    return SurfaceMetadata(
        source_id="sheet-source",
        source_revision="scan-17",
        coordinate_contract=_SI,
        provenance=("synthetic-open-sheet",),
    )


def envelope_sheet() -> RawTriangleSoup:
    return RawTriangleSoup(
        np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=np.float64),
        np.array([[0, 1, 2]], dtype=np.int64),
        envelope_metadata(),
        feature_edges=np.array([[0, 1]], dtype=np.int64),
    )


def envelope_policy(
    *,
    max_samples: int = 100000,
    max_work_units: int = 10000000,
    max_vertices: int = 10000,
    max_triangles: int = 20000,
    max_volume_vertices: int | None = None,
    max_tetrahedra: int | None = None,
    feature_permission: EnvelopeFeaturePermission = EnvelopeFeaturePermission.ALLOW_ROUNDING,
    topology_permission: EnvelopeTopologyPermission = EnvelopeTopologyPermission.ALLOW_CHANGE,
) -> SurfaceEnvelopePolicy:
    return SurfaceEnvelopePolicy(
        offset=0.263,
        maximum_deviation=0.55,
        spacing=0.11,
        certificate_spacing=0.05,
        feature_permission=feature_permission,
        topology_permission=topology_permission,
        max_samples=max_samples,
        max_work_units=max_work_units,
        max_vertices=max_vertices,
        max_triangles=max_triangles,
        max_volume_vertices=max_volume_vertices,
        max_tetrahedra=max_tetrahedra,
    )


def envelope_specification(
    source: NativeSurfaceEnvelopeSource,
    *,
    original_identity: bool = False,
    maximum_wall_seconds: float = 3600.0,
    maximum_scratch_bytes: int = 8000000000,
) -> M.VolumeMeshingSpec:
    scope = M.MeshingScope(
        source.envelope.source.metadata.source_id
        if original_identity
        else source.source_id,
        source.envelope.source.metadata.source_revision
        if original_identity
        else source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        "envelope-facets",
        np.arange(source.plc_source.complex.facet_count, dtype=np.int64),
    )
    return M.VolumeMeshingSpec(
        M.CellMeshingTarget(3, 3, M.CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        M.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            M.UniformSizeControl(scope, 2.0, strength=M.SizeControlStrength.SOFT),
        ),
        limits=M.MeshingLimits(
            maximum_vertices=20000,
            maximum_cells=100000,
            maximum_work_units=20000000,
            maximum_wall_seconds=maximum_wall_seconds,
            maximum_scratch_bytes=maximum_scratch_bytes,
        ),
    )


def envelope_failure(
    field: str, expectation: str, values: dict[str, object]
) -> RuntimeError:
    return RuntimeError(f"{field} must {expectation}; got {values!r}")


def tetrahedron_measures(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    # The determinant is a host-only independent signed-volume reference.
    corners = points[cells]
    return np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0


def quadratic_value(points: Array) -> Array:
    return jnp.sum(points * points, axis=-1)


def quadratic_source(points: Array, parameters: object) -> Array:
    del parameters
    # -Delta(|x|^2) = -2*d on the three-dimensional repaired carrier.
    return jnp.full(points.shape[:-1], -2.0 * points.shape[-1], dtype=points.dtype)


def quadratic_l2_error(mesh: phx.discretization.CellMesh, values: Array) -> float:
    """Integrate the squared P1-minus-quadratic error using simplex moments."""
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    cells = np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
    corners = points[cells]
    nodal = np.asarray(values)[cells]
    coefficients = -(corners @ np.swapaxes(corners, -1, -2))
    coefficients += 0.5 * (nodal[:, :, None] + nodal[:, None, :])
    dimension = mesh.topological_dimension
    count = dimension + 1
    integral = np.zeros(cells.shape[0], dtype=np.float64)
    # Normalized barycentric moment: d!*prod(alpha!)/(d+sum(alpha))!.
    denominator = math.factorial(dimension + 4) / math.factorial(dimension)
    for first in range(count):
        for second in range(count):
            for third in range(count):
                for fourth in range(count):
                    exponents = np.bincount(
                        (first, second, third, fourth), minlength=count
                    )
                    moment = (
                        np.prod([math.factorial(int(value)) for value in exponents])
                        / denominator
                    )
                    integral += (
                        coefficients[:, first, second]
                        * coefficients[:, third, fourth]
                        * moment
                    )
    return float(
        np.sqrt(max(0.0, float(np.sum(tetrahedron_measures(points, cells) * integral))))
    )
