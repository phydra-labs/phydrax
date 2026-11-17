# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np

from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh, lagrange_element
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._coordinate_enclosure import Expression, Polynomial
from phydrax.geometry._embedding_cells import EmbeddingCell, prepare_embedding_cells


def _endpoint(value: Fraction, *, upper: bool) -> float:
    result = float(value)
    if (Fraction(result) < value if upper else Fraction(result) > value):
        result = float(np.nextafter(result, np.inf if upper else -np.inf))
    return result


def _interval_bounds(
    nodes: np.ndarray, values: np.ndarray
) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]:
    """Independent P2 Bernstein controls and outward binary64 accumulation."""
    barycentric = []
    for row in range(nodes.shape[0]):
        point = tuple(Fraction(float(nodes[row, axis])) for axis in range(3))
        barycentric.append((Fraction(1) - sum(point, Fraction(0)), *point))
    controls = [(i, i) for i in range(4)] + [
        (i, j) for i in range(4) for j in range(i + 1, 4)
    ]
    outer = []
    inner = []
    for axis in range(3):
        lows = []
        highs = []
        for left, right in controls:
            low = high = 0.0
            for node, row in zip(barycentric, values, strict=True):
                active = [i for i, value in enumerate(node) if value]
                if len(active) == 1:
                    vertex = active[0]
                    control = Fraction(int(left == right == vertex))
                    if left != right and vertex in (left, right):
                        control = Fraction(-1, 2)
                else:
                    control = Fraction(2 if active == [left, right] else 0)
                product = control * Fraction(float(row[axis]))
                # All P2 basis controls and fixture values are dyadic and these
                # products are exactly representable in binary64.
                part = float(product)
                assert Fraction(part) == product
                for lower in (True, False):
                    previous = low if lower else high
                    exact = Fraction(previous) + product
                    rounded = float(exact)
                    if Fraction(rounded) != exact:
                        rounded = float(np.nextafter(rounded, -np.inf if lower else np.inf))
                    if lower:
                        low = rounded
                    else:
                        high = rounded
            lows.append(low)
            highs.append(high)
        outer.append((min(lows), max(highs)))
        inner.append((min(highs), max(lows)))
    return tuple(outer), tuple(inner)

def test_p2_tetrahedra_match_exact_fraction_reference() -> None:
    element = lagrange_element("tetrahedron", 2)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    edge = next(
        i for i in range(nodes.shape[0])
        if tuple(float(nodes[i, axis]) for axis in range(3)) == (0.5, 0.0, 0.0)
    )
    rows = []
    for index in range(3):
        points = nodes.copy()
        points[:, 0] += 4 * index
        if index == 1:
            points[edge, 2] += 0.125
        if index == 2:
            points[edge, 0] = np.nextafter(points[edge, 0], np.inf)
        rows.append(points)
    bank = np.concatenate(rows)
    # Mesh corners are in reference order, independent of the P2 DOF order.
    reference = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    vertices = np.asarray(
        [[4 * i + x, y, z] for i in range(3) for x, y, z in reference],
        dtype=np.float64,
    )
    mesh = CellMesh(
        vertices,
        (CellBlock("cells", "tetrahedron", np.arange(12, dtype=np.int32).reshape(3, 4),
                   global_ids=np.asarray((31, 17, 42), dtype=np.int64)),),
        vertex_global_ids=np.arange(100, 112, dtype=np.int64),
    )
    geometry = CellGeometrySpec(
        {"cells": element}, {"cells": np.arange(30, dtype=np.int64).reshape(3, 10)}, bank
    )
    cells = prepare_embedding_cells(mesh, geometry)
    assert len(cells) == 3
    reference_vertices = tuple(tuple(Fraction(v) for v in point) for point in reference)
    facets = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
    zero = (0, 0, 0)
    units = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    bump = {(1, 0, 0): 4, (2, 0, 0): -4, (1, 1, 0): -4, (1, 0, 1): -4}
    for index, cell in enumerate(cells):
        delta = [Fraction(0), Fraction(0), Fraction(0)]
        delta[2] = Fraction(1, 8) if index == 1 else Fraction(0)
        if index == 2:
            delta[0] = Fraction(float(rows[index][edge, 0])) - Fraction(17, 2)
        expressions: list[Expression] = []
        for axis in range(3):
            expression: Polynomial = {units[axis]: Fraction(1)}
            if axis == 0 and index:
                expression[zero] = Fraction(4 * index)
            for exponent, factor in bump.items():
                value = expression.get(exponent, Fraction(0)) + factor * delta[axis]
                if value:
                    expression[exponent] = value
                else:
                    expression.pop(exponent, None)
            expressions.append(expression)
        corners = tuple(
            (Fraction(4 * index + x), Fraction(y), Fraction(z)) for x, y, z in reference
        )
        affine = index == 0
        # The degree-two Bernstein controls are vertex values and, on each edge,
        # twice the midpoint value minus half the sum of its endpoints.
        exact_bounds = []
        for axis in range(3):
            controls = [point[axis] for point in corners]
            for left in range(4):
                for right in range(left + 1, 4):
                    midpoint = (corners[left][axis] + corners[right][axis]) / 2
                    if (left, right) == (0, 1):
                        midpoint += delta[axis]
                    controls.append(2 * midpoint - (corners[left][axis] + corners[right][axis]) / 2)
            exact_bounds.append((min(controls), max(controls)))
        expected = EmbeddingCell(
            (31, 17, 42)[index], "cells", index, "tetrahedron", 3, "simplex",
            tuple(range(4 * index, 4 * index + 4)),
            tuple(range(100 + 4 * index, 104 + 4 * index)),
            reference_vertices, tuple(expressions), corners,
            corners[0] if affine else None,
            tuple(tuple(Fraction(int(axis == column)) for column in range(3)) for axis in range(3)) if affine else None,
            tuple(_endpoint(lo, upper=False) for lo, _ in exact_bounds),
            tuple(_endpoint(hi, upper=True) for _, hi in exact_bounds),
            facets, None, None, mesh.mesh_id, cell_geometry_id(geometry), None,
        )
        if not affine and cell.image_inner_bounds is not None:
            outer, inner = _interval_bounds(nodes, rows[index])
            expected = replace(
                expected,
                image_lower=tuple(bound[0] for bound in outer),
                image_upper=tuple(bound[1] for bound in outer),
                image_inner_bounds=inner,
            )
        assert cell == expected
        for axis, (lo, hi) in enumerate(exact_bounds):
            assert Fraction(cell.image_lower[axis]) <= lo
            assert Fraction(cell.image_upper[axis]) >= hi
            if cell.image_inner_bounds is not None:
                inner_lo, inner_hi = cell.image_inner_bounds[axis]
                assert Fraction(inner_lo) >= lo
                assert Fraction(inner_hi) <= hi
