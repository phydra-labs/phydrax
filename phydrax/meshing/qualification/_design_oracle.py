#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical design qualification cases and independent affine coordinate oracles."""

from __future__ import annotations

import equinox as eqx
import jax
import numpy as np
from jax import Array

from ...discretization._cell_geometry import (
    BarycentricCellGeometryElement,
    coordinate_tabulation,
    PolynomialComposedCellGeometryElement,
    require_full_p1_source,
    require_p1_cardinal_source,
    require_scalar_coordinate_element,
    RestrictedCellGeometryElement,
    ScalarCoordinateElement,
    ScalarCoordinateTabulation,
)
from ...discretization._reference_cell_types import CellGeometryElement
from ...discretization.fem._reference_types import FiniteElementSpec


DESIGN_QUALIFICATION_DEFAULT_CASE = "routine-sphere"


def design_qualification_cases() -> dict[str, dict[str, object]]:
    """Return a fresh copy of the canonical native design-loop case table.

    Every case executes the complete fixed-epoch derivative, trusted proposal,
    native event, all-state transfer, rollback and reanalysis chain under the
    caller's frozen campaign controls. The lattice has
    ``lattice_factor * resolution + 1`` points per axis over
    ``[-lattice_half_width, lattice_half_width]**3``. The combined case also
    solves the represented-QoI design equation inside the event-free epoch and
    requalifies the derivatives at its republished native epoch.
    """
    return {
        "routine-sphere": {
            "scale": "routine",
            "radius": 0.75,
            "lattice_factor": 2,
            "lattice_half_width": 1.4,
            "design_target": None,
            "demanding_controls": [],
        },
        "larger-sphere": {
            "scale": "larger",
            "radius": 0.75,
            "lattice_factor": 4,
            "lattice_half_width": 1.4,
            "design_target": None,
            "demanding_controls": [],
        },
        "combined-design-target": {
            "scale": "combined",
            "radius": 0.75,
            "lattice_factor": 2,
            "lattice_half_width": 1.4,
            "design_target": "authored-continuous-sphere-x2-moment",
            "demanding_controls": [
                "fixed-epoch-adjoint-design-root",
                "geometry-realization-event-requalification",
                "observed-error-learned-marker",
                "lattice-event-crossing-refusal",
                "failed-reanalysis-rollback-reuse",
            ],
        },
    }


def _require_affine_triangle(element: CellGeometryElement, /) -> ScalarCoordinateElement:
    """Prove the affine P1 triangle law of every source layer; a degree is not a proof."""
    coordinate = require_scalar_coordinate_element(element, "Design physical oracle")
    if coordinate.cell_kind != "triangle":
        raise ValueError(
            f"affine_triangle_corner_basis.cell_kind must select a triangle coordinate map; got {coordinate.cell_kind!r}"
        )
    match coordinate:
        case FiniteElementSpec():
            require_p1_cardinal_source(coordinate)
        case BarycentricCellGeometryElement():
            require_full_p1_source(coordinate)
        case RestrictedCellGeometryElement():
            _require_affine_triangle(coordinate.source_element)
        case PolynomialComposedCellGeometryElement():
            _require_affine_triangle(coordinate.source_element)
            require_p1_cardinal_source(coordinate.chart_element)
        case _:
            raise ValueError(
                f"affine_triangle_corner_basis.source_law must be affine P1 triangular with affine reference actions; got {type(coordinate).__name__!r}"
            )
    return coordinate


@eqx.filter_jit
def _corner_basis(element: ScalarCoordinateTabulation, reference: Array, /) -> Array:
    # Numerical source actions stay dynamic across the tabulation boundary.
    return element.tabulate(reference)[0]


def affine_triangle_corner_basis(element: CellGeometryElement, /) -> Array:
    """Return the coordinate basis at the reference triangle corners.

    A straight-corner physical oracle is independent of the represented map only
    when the coordinate law is proved affine: a P1 cardinal finite element, a full
    P1 barycentric source, or a restriction or polynomial composition whose source
    and chart layers are themselves affine P1. Any other element refuses.
    """
    coordinate = _require_affine_triangle(element)
    reference = jax.device_put(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
    )
    return _corner_basis(coordinate_tabulation(coordinate), reference)


__all__ = [
    "DESIGN_QUALIFICATION_DEFAULT_CASE",
    "affine_triangle_corner_basis",
    "design_qualification_cases",
]
