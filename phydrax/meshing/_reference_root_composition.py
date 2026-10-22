#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Construct affine reference charts with rational arithmetic and retained source coefficients."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from phydrax.linalg._exact import solve_exact

from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..discretization._reference_cell import reference_cell_topology
from ..discretization.fem import FiniteElementSpec
from ..typing import checked


def affine_reference_root_frame(
    root: np.ndarray,
    root_kind: str,
    /,
) -> tuple[list[list[Fraction]], tuple[Fraction, ...]]:
    """Validate the source affine chart equations at every root corner."""
    match root_kind:
        case "tetrahedron" | "prism":
            axes = (1, 2, 3)
        case "hexahedron":
            axes = (1, 3, 4)
        case _:
            raise ValueError(
                f"ReferenceRootComposition.root_kind must be tetrahedron, prism, or hexahedron; got {root_kind!r}"
            )
    topology = reference_cell_topology(root_kind)
    if root.shape != (len(topology.vertices), 3) or not np.all(np.isfinite(root)):
        raise ValueError(
            f"ReferenceRootComposition.root must contain finite three-dimensional corners; got shape {root.shape!r}, expected {(len(topology.vertices), 3)!r}, finite {bool(np.all(np.isfinite(root)))}"
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
                f"ReferenceRootComposition.root must satisfy its affine corner equations; got point {point!r} at reference corner {corner!r}"
            )
    return matrix, origin


@checked
def exact_reference_chart_controls(
    target: np.ndarray | tuple[tuple[Fraction, ...], ...],
    root: np.ndarray,
    root_kind: str,
    chart: FiniteElementSpec,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Solve the affine reference chart with rational arithmetic.

    Corner containment proves whole-cell containment because degree-one target
    shape functions are nonnegative on the reference cell. The chart retains
    source coefficients without fitting or rounding them.

    Immutable rational target rows are placement provenance earned before
    carrier rounding; callers authenticate their round-to-nearest carrier.
    Numerical arrays retain the original exact lift of their current values.
    """
    target_topology = reference_cell_topology(chart.cell_kind)
    if chart.degree != 1 or chart.cell_kind not in ("tetrahedron", "prism", "hexahedron"):
        raise ValueError(
            "ReferenceRootComposition.target must use a supported degree-one chart."
        )
    matrix, origin = affine_reference_root_frame(root, root_kind)
    if isinstance(target, np.ndarray):
        if target.shape != (len(target_topology.vertices), 3) or not np.all(np.isfinite(target)):
            raise ValueError(
                f"ReferenceRootComposition.target must contain finite degree-one three-dimensional corners; got shape {target.shape!r}."
            )
        right = [
            [Fraction(float(point[axis])) - origin[axis] for point in target]
            for axis in range(3)
        ]
    else:
        if len(target) != len(target_topology.vertices) or any(
            len(point) != 3 or any(type(value) is not Fraction for value in point)
            for point in target
        ):
            raise ValueError(
                "ReferenceRootComposition.target must contain complete immutable rational corner rows."
            )
        right = [
            [point[axis] - origin[axis] for point in target]
            for axis in range(3)
        ]

    ticket = budget.ticket
    if ticket.remaining_work_units < 1 or ticket.remaining_geometry_queries < 1:
        ticket.refuse(1, queries_upper=1)
    ticket.consume(1, geometry_queries=1)
    coordinates = solve_exact(matrix, right)
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
    entries = chart.local_dof_count * 3
    storage_upper = 2 * entries * np.dtype(dtype).itemsize
    if not bounded:
        storage_upper += sum(
            64
            + (abs(value.numerator).bit_length() + value.denominator.bit_length() + 7)
            // 8
            for axis in coordinates
            for value in axis
        )
    budget.reserve(0, storage_upper)
    numerators = np.empty((chart.local_dof_count, 3), dtype=dtype)
    denominators = np.empty(numerators.shape, dtype=dtype)
    for corner, dofs in enumerate(chart.entity_dofs[0]):
        if len(dofs) != 1:
            raise ValueError(
                f"ReferenceRootComposition.chart.entity_dofs must assign one DOF to each corner; got corner {corner}, DOFs {dofs!r}"
            )
        for axis in range(3):
            value = coordinates[axis][corner]
            numerators[dofs[0], axis] = value.numerator
            denominators[dofs[0], axis] = value.denominator
    return numerators, denominators
