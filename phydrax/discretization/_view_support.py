#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Support geometry and location status shared by discrete field views.

Structured reconstructions cover a tensor box; affine simplicial meshes derive
their region from the mesh; an explicit geometry is admitted only with evidence
that the reconstruction covers it.
"""

from __future__ import annotations

from itertools import product
from math import factorial, isfinite
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array

from ._reference_cell import reference_cell_topology
from ._simplicial_locator import AbstractCellLocator, CellLocationStatus
from ._views import FieldQueryStatus


def cell_location_status(status: Array, /) -> Array:
    """Map `CellLocationStatus` codes onto field query statuses."""
    return jnp.where(
        status == int(CellLocationStatus.LOCATED),
        int(FieldQueryStatus.VALID),
        jnp.where(
            status == int(CellLocationStatus.OUTSIDE),
            int(FieldQueryStatus.OUTSIDE_SUPPORT),
            jnp.where(
                status == int(CellLocationStatus.NONFINITE),
                int(FieldQueryStatus.NONFINITE),
                int(FieldQueryStatus.LOCATION_FAILED),
            ),
        ),
    ).astype(jnp.int32)


def simplicial_mesh_support_geometry(
    locator: AbstractCellLocator, support_id: str, /
) -> Any:
    """Compile the exact region covered by affine simplicial or planar-facet tensor cells."""
    from ..geometry.simplicial import MeshRegion
    from ..geometry.simplicial._io import planar_region_from_triangles

    cell_map = locator.cell_map
    if cell_map.coordinate_element.degree != 1:
        raise ValueError(
            "Curved FE cell maps require an explicit support_geometry with evidence."
        )
    coordinates = np.asarray(locator.coordinates, dtype=np.float64)
    cells = np.asarray(cell_map.coordinate_dofs, dtype=np.int64)
    feature_id = f"finite-element-support:{support_id}"
    match cell_map.coordinate_element.cell_kind:
        case "triangle":
            source = planar_region_from_triangles(
                coordinates, cells, recenter=False, feature_id=feature_id
            )
        case "tetrahedron":
            faces = np.concatenate(
                (
                    cells[:, [1, 2, 3]],
                    cells[:, [0, 3, 2]],
                    cells[:, [0, 1, 3]],
                    cells[:, [0, 2, 1]],
                )
            )
            keys = np.sort(faces, axis=1)
            _, inverse, counts = np.unique(
                keys, axis=0, return_inverse=True, return_counts=True
            )
            boundary = faces[counts[inverse.reshape((-1,))] == 1]
            used, compact = np.unique(boundary, return_inverse=True)
            source = MeshRegion(
                coordinates[used],
                compact.reshape(boundary.shape).astype(np.int32),
                feature_id=feature_id,
            )
        case "quadrilateral" | "hexahedron":
            return _tensor_mesh_support_geometry(
                coordinates,
                cells,
                cell_map.coordinate_element.cell_kind,
                feature_id,
                locator,
            )
        case "interval":
            from ..geometry.simplicial._support_region import compile_simplicial_support

            return compile_simplicial_support(locator, support_id)
        case kind:
            if kind.startswith("simplex:"):
                from ..geometry.simplicial._support_region import (
                    compile_simplicial_support,
                )

                return compile_simplicial_support(locator, support_id)
            if kind.startswith("tensor:"):
                return _tensor_mesh_support_geometry(
                    coordinates, cells, kind, feature_id, locator
                )
            raise ValueError(
                f"The support of {kind!r} FE cells is not derived; pass an explicit support_geometry."
            )
    return source.compile()


def _tensor_mesh_support_geometry(
    coordinates: np.ndarray,
    cells: np.ndarray,
    cell_kind: str,
    feature_id: str,
    locator: AbstractCellLocator,
    /,
) -> Any:
    from ..geometry.simplicial import MeshRegion
    from ..geometry.simplicial._io import planar_region_from_triangles

    reference = reference_cell_topology(cell_kind)
    if reference.dimension == 1:
        from ..geometry.simplicial._support_region import compile_simplicial_support

        return compile_simplicial_support(locator, feature_id)
    if reference.dimension == 2:
        triangles = cells[:, np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)]
        return planar_region_from_triangles(
            coordinates,
            triangles.reshape((-1, 3)),
            recenter=False,
            feature_id=feature_id,
        ).compile()
    if reference.dimension != 3:
        raise ValueError("Tensor support regions are prepared for 2-D/3-D mesh cells.")
    faces = cells[:, np.asarray(reference.entities[2], dtype=np.int32)]
    keys = np.sort(faces.reshape((-1, 4)), axis=1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("Tensor support requires manifold faces.")
    exterior = counts[inverse] == 1
    boundary = faces.reshape((-1, 4))[exterior].copy()
    parent = np.repeat(np.arange(cells.shape[0]), faces.shape[1])[exterior]
    points = coordinates[boundary]
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    norm = np.linalg.norm(normals, axis=1)
    scale = np.max(np.linalg.norm(points - points[:, :1], axis=2), axis=1)
    if np.any(norm <= np.finfo(np.float64).eps * np.maximum(scale**2, 1.0)):
        raise ValueError("Tensor support boundary faces must be nondegenerate.")
    residual = np.abs(np.sum((points[:, 3] - points[:, 0]) * normals, axis=1)) / norm
    if np.any(residual > 1e-10 * np.maximum(scale, 1.0)):
        raise ValueError("Warped tensor faces need an explicit curved support_geometry.")
    centers = np.mean(coordinates[cells[parent]], axis=1)
    inward = np.sum(normals * (np.mean(points, axis=1) - centers), axis=1) < 0.0
    boundary[inward] = boundary[inward][:, (0, 3, 2, 1)]
    triangles = boundary[:, np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)].reshape(
        (-1, 3)
    )
    used, compact = np.unique(triangles, return_inverse=True)
    return MeshRegion(
        coordinates[used],
        compact.reshape(triangles.shape).astype(np.int32),
        feature_id=feature_id,
    ).compile()


def verify_mesh_support_geometry(
    geometry: Any, locator: AbstractCellLocator, /, *, tolerance: float
) -> None:
    from ..geometry import GeometryCapability

    cell_map = locator.cell_map
    coordinates = jnp.asarray(locator.coordinates)
    field = np.asarray(geometry.boundary_field(coordinates))
    if np.any(field > tolerance):
        raise ValueError("FE mesh vertices lie outside the support geometry.")
    if not geometry.has_capability(GeometryCapability.INTERIOR_MEASURE):
        raise ValueError(
            "An explicit support_geometry needs an interior measure to evidence that "
            "the FE mesh covers it."
        )
    reference = jnp.full(
        (cell_map.cell_count, cell_map.reference_dimension),
        1.0 / (cell_map.reference_dimension + 1),
        dtype=coordinates.dtype,
    )
    evaluation = cell_map.evaluate(
        coordinates, jnp.arange(cell_map.cell_count), reference
    )
    if cell_map.coordinate_element.degree != 1:
        raise ValueError(
            "Covering evidence is exact only for affine FE cell maps; curved maps "
            "need an explicit inverse provider with its own support evidence."
        )
    reference_topology = reference_cell_topology(cell_map.coordinate_element.cell_kind)
    tensor = reference_topology.name in (
        "quadrilateral",
        "hexahedron",
    ) or reference_topology.name.startswith("tensor:")
    if tensor:
        from .._polynomial._orthogonal import legendre_rule_data

        axis_rule = legendre_rule_data(2)
        nodes = 0.5 * (np.asarray(axis_rule.nodes) + 1.0)
        weights = 0.5 * np.asarray(axis_rule.weights)
        indices = np.asarray(
            tuple(product(range(2), repeat=cell_map.reference_dimension)), dtype=np.int32
        )
        references = np.tile(nodes[indices], (cell_map.cell_count, 1))
        cell_indices = np.repeat(
            np.arange(cell_map.cell_count, dtype=np.int32), indices.shape[0]
        )
        evaluation = cell_map.evaluate(
            coordinates, jnp.asarray(cell_indices), jnp.asarray(references)
        )
        quadrature_weights = np.tile(
            np.prod(weights[indices], axis=1), cell_map.cell_count
        )
        mesh_measure = float(
            np.sum(np.abs(np.asarray(evaluation.determinant)) * quadrature_weights)
        )
    else:
        mesh_measure = float(
            np.sum(np.abs(np.asarray(evaluation.determinant)))
        ) / factorial(cell_map.reference_dimension)
    geometry_measure = float(np.asarray(geometry.measure))
    if abs(mesh_measure - geometry_measure) > tolerance * max(1.0, geometry_measure):
        raise ValueError(
            "The FE mesh measure differs from the support geometry measure; the "
            "mesh does not cover the geometry."
        )


def tensor_box_support_geometry(
    lower: np.ndarray,
    upper: np.ndarray,
    support_geometry: Any,
    /,
    *,
    feature_id: str,
    tolerance: float,
) -> Any:
    """Return the compiled support box, or verify an explicit box geometry.

    `support_geometry=None` compiles an `Orthotope` spanning `[lower, upper]`.
    An explicit region is admitted only when its bounds equal the box and its
    interior measure equals the box measure: a region inside the box's bounding
    box with the box's measure is the box.
    """
    from ..geometry import CompiledGeometry, GeometryCapability, GeometryKind, Orthotope

    lower_ = np.asarray(lower, dtype=np.float64)
    upper_ = np.asarray(upper, dtype=np.float64)
    if (
        lower_.ndim != 1
        or upper_.shape != lower_.shape
        or not np.all(np.isfinite(lower_) & np.isfinite(upper_))
        or np.any(upper_ <= lower_)
    ):
        raise ValueError("A support box needs finite, ordered lower and upper corners.")
    limit = float(tolerance)
    if not isfinite(limit) or limit < 0.0:
        raise ValueError("support_tolerance must be finite and non-negative.")
    if support_geometry is None:
        return Orthotope(
            0.5 * (lower_ + upper_), upper_ - lower_, feature_id=feature_id
        ).compile()
    if not isinstance(support_geometry, CompiledGeometry):
        raise TypeError("support_geometry must be a CompiledGeometry or None.")
    if support_geometry.kind is not GeometryKind.REGION:
        raise ValueError("support_geometry must be a region geometry.")
    if support_geometry.ambient_dimension != lower_.size:
        raise ValueError("support_geometry dimension must equal the box dimension.")
    if not support_geometry.has_capability(GeometryCapability.INTERIOR_MEASURE):
        raise ValueError(
            "An explicit support_geometry needs an interior measure to evidence "
            "that it is the reconstruction's tensor box."
        )
    box = np.stack((lower_, upper_))
    bounds = np.asarray(support_geometry.bounds, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(box))))
    if bounds.shape != box.shape or np.max(np.abs(bounds - box)) > limit * scale:
        raise ValueError("support_geometry bounds differ from the reconstruction box.")
    box_measure = float(np.prod(upper_ - lower_))
    measure = float(np.asarray(support_geometry.measure))
    if abs(measure - box_measure) > limit * max(1.0, box_measure):
        raise ValueError(
            "support_geometry measure differs from the reconstruction box; the "
            "geometry is not the tensor box the reconstruction covers."
        )
    return support_geometry


__all__ = [
    "cell_location_status",
    "simplicial_mesh_support_geometry",
    "tensor_box_support_geometry",
    "verify_mesh_support_geometry",
]
