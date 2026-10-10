#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact affine reference charts, retaining the original coordinate source."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from .._meshcore import charge_native_geometry_queries
from ..discretization._coordinate_enclosure import _solve_exact
from ..discretization._reference_cell import reference_cell_topology
from ..discretization.fem import FiniteElementSpec
from ._quad_generation import _family_host_array


def affine_reference_root_frame(
    root: np.ndarray,
    root_kind: str,
    /,
) -> tuple[list[list[Fraction]], tuple[Fraction, ...]]:
    """Validate the original exact affine chart, including every root corner."""
    match root_kind:
        case "tetrahedron" | "prism":
            axes = (1, 2, 3)
        case "hexahedron":
            axes = (1, 3, 4)
        case _:
            raise ValueError(
                "Reference composition requires an affine tetrahedron, prism, or hexahedron root."
            )
    topology = reference_cell_topology(root_kind)
    if root.shape != (len(topology.vertices), 3) or not np.all(np.isfinite(root)):
        raise ValueError(
            "Reference composition requires every finite original three-dimensional root corner."
        )
    origin = tuple(Fraction(float(value)) for value in root[0])
    matrix = [
        [Fraction(float(root[index, axis])) - origin[axis] for index in axes]
        for axis in range(3)
    ]
    for point, corner in zip(root, topology.vertices, strict=True):
        if any(
            Fraction(float(point[axis]))
            != origin[axis]
            + sum(
                (matrix[axis][column] * Fraction(corner[column]) for column in range(3)),
                Fraction(0),
            )
            for axis in range(3)
        ):
            raise ValueError(
                "The independently declared reference root is not exactly affine."
            )
    return matrix, origin


def exact_reference_chart_controls(
    target: np.ndarray,
    root: np.ndarray,
    root_kind: str,
    chart: FiniteElementSpec,
    /,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Solve the actual affine reference root exactly, never a physical inverse.

    Corner containment proves whole-cell containment because the canonical
    degree-one target shape functions are nonnegative on their reference cell.
    Original source coefficients are not fitted, rounded, or reconstructed.
    """
    target_topology = reference_cell_topology(chart.cell_kind)
    if (
        target.shape != (len(target_topology.vertices), 3)
        or chart.degree != 1
        or chart.cell_kind not in ("tetrahedron", "prism", "hexahedron")
        or not np.all(np.isfinite(target))
    ):
        raise ValueError(
            "Reference composition requires complete finite three-dimensional linear corner charts."
        )
    matrix, origin = affine_reference_root_frame(root, root_kind)
    right = [
        [Fraction(float(point[axis])) - origin[axis] for point in target]
        for axis in range(3)
    ]
    charge_native_geometry_queries(1, work_units=1)
    coordinates = _solve_exact(matrix, right)
    for corner in zip(*coordinates, strict=True):
        if any(value < 0 or value > 1 for value in corner):
            return None
        if root_kind == "tetrahedron" and sum(corner, Fraction(0)) > 1:
            return None
        if root_kind == "prism" and corner[0] + corner[1] > 1:
            return None
    maximum = np.iinfo(np.int64).max
    bounded = all(
        abs(value.numerator) <= maximum and value.denominator <= maximum
        for axis in coordinates
        for value in axis
    )
    dtype = np.int64 if bounded else np.object_
    numerators = _family_host_array((chart.local_dof_count, 3), dtype)
    denominators = _family_host_array(numerators.shape, dtype)
    for corner, dofs in enumerate(chart.entity_dofs[0]):
        if len(dofs) != 1:
            raise ValueError(
                "A canonical linear reference chart must have one DOF per corner."
            )
        for axis in range(3):
            value = coordinates[axis][corner]
            numerators[dofs[0], axis] = value.numerator
            denominators[dofs[0], axis] = value.denominator
    return numerators, denominators
