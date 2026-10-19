#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Rational physical corner views shared by independent mapped root declarations."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from ..discretization import _coordinate_enclosure as algebra
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..discretization._exact_algebraic_geometry import rational_coefficient


if TYPE_CHECKING:
    from ..discretization._cell_geometry import CellGeometrySpec
    from ..discretization._cell_mesh import CellMesh


def mapped_source_corner_coordinates(
    reference_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    *,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Validate rational physical corner agreement across independent root maps."""

    elements, routes, _ = source_geometry.resolve(reference_mesh)
    values = source_geometry.source_coordinates(budget=budget)
    ids = np.asarray(reference_mesh.vertex_global_ids, dtype=np.int64)
    positions: dict[int, tuple[Fraction, ...]] = {}
    for block, element, route in zip(
        reference_mesh.blocks, elements, routes, strict=True
    ):
        local_values = tuple(
            tuple(values[index] for index in row)
            for row in np.asarray(route, dtype=np.int64)
        )
        for local, row in zip(local_values, np.asarray(block.vertices), strict=True):
            images = algebra.coordinate_corner_images(element, local, budget=budget)
            if images is None:
                raise ValueError(
                    f"source_geometry.corner_images must have rational coordinate expressions; got {images!r} for element {element!r}"
                )
            for vertex, image in zip(ids[row], images, strict=True):
                image = tuple(
                    rational_coefficient(value, role="source_geometry.corner_image")
                    for value in image
                )
                if int(vertex) in positions and positions[int(vertex)] != image:
                    raise ValueError(
                        f"source_geometry.corner_images must agree at shared vertex {int(vertex)}; got {image!r} and {positions[int(vertex)]!r}"
                    )
                positions[int(vertex)] = image
    if any(int(vertex) not in positions for vertex in ids):
        raise ValueError(
            f"reference_mesh.vertex_global_ids must all have source chart images; got unmapped IDs {tuple(int(vertex) for vertex in ids if int(vertex) not in positions)!r}"
        )
    return np.asarray(
        tuple(tuple(float(value) for value in positions[int(vertex)]) for vertex in ids),
        dtype=np.float64,
    )
