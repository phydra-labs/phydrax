#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Support geometry and location status shared by discrete field views.

Structured reconstructions cover a tensor box; affine simplicial meshes derive
their region from the mesh; an explicit geometry is admitted only with evidence
that the reconstruction covers it.
"""

from __future__ import annotations

from math import isfinite
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

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
    """Compile the region covered by an affine triangle or tetrahedron mesh."""
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
        case kind:
            raise ValueError(
                f"The support of {kind!r} FE cells is not derived; pass an explicit "
                "support_geometry."
            )
    return source.compile()


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
    simplex_volume = {1: 1.0, 2: 0.5, 3: 1.0 / 6.0}[cell_map.reference_dimension]
    mesh_measure = (
        float(np.sum(np.abs(np.asarray(evaluation.determinant)))) * simplex_volume
    )
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
