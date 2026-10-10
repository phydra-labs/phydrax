#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact original-root face support for mixed coordinate-map restrictions."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from ..discretization import _coordinate_enclosure as algebra, CellMesh
from ..discretization._cell_geometry import CellGeometryElement, CellGeometrySpec
from ..discretization._reference_cell import reference_cell_topology
from ._mapped_coverage import nonnegative, SubdivisionLedger
from ._mapped_reference_coverage import _polynomial_reference_chart
from ._restricted_embedding import _inside_reference, _reference_constraints


if TYPE_CHECKING:
    from ..meshing._plc_mapped_support import MappedPlcSupport


def _cross(
    first: tuple[Fraction, ...], second: tuple[Fraction, ...]
) -> tuple[Fraction, ...]:
    return tuple(
        first[(axis + 1) % 3] * second[(axis + 2) % 3]
        - first[(axis + 2) % 3] * second[(axis + 1) % 3]
        for axis in range(3)
    )


@dataclass(frozen=True)
class _MappedFacetRoot:
    kind: str
    vertices: tuple[int, ...]
    element: CellGeometryElement
    controls: algebra.CoordinateSourceBank


@dataclass(frozen=True)
class MappedSourceFacetAuthority:
    """Actual original cell expressions and their declared source-face incidence."""

    mesh: CellMesh
    geometry: CellGeometrySpec
    roots: dict[int, _MappedFacetRoot]
    facet_vertex_ids: tuple[tuple[int, ...], ...]

    @classmethod
    def prepare(
        cls,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        triangle_vertices: np.ndarray,
        triangle_facets: np.ndarray,
        /,
        *,
        ledger: algebra.CoordinateEnclosureBudget,
    ) -> MappedSourceFacetAuthority:
        """Bind actual original cell expressions under the owning support ledger.

        ``ledger`` is the support preparation's own exact-expression ledger; its
        remaining allowance is consumed here and by every later face query.
        """
        elements, routes, _ = geometry.resolve(mesh)
        values = geometry.source_coordinates()
        vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        roots: dict[int, _MappedFacetRoot] = {}
        with ledger.activate(), ledger.temporary_scope():
            facet_count = mesh.entity_set(2).count
            if (
                triangle_vertices.ndim != 2
                or triangle_vertices.shape[1] != 3
                or triangle_facets.shape != (triangle_vertices.shape[0],)
                or np.any(triangle_facets < 0)
                or np.any(triangle_facets >= facet_count)
            ):
                raise ValueError(
                    "Mapped facet authority requires original face-row triangle incidence."
                )
            # Build the source-face index once. Repeated full-bank masks would
            # perform quadratic preparation work before any support query.
            ledger.reserve(
                triangle_facets.size + facet_count,
                16 * triangle_facets.size + 16 * (facet_count + 1),
            )
            order = np.argsort(triangle_facets, kind="stable")
            bounds = np.concatenate(
                (
                    np.zeros((1,), dtype=np.int64),
                    np.cumsum(
                        np.bincount(triangle_facets, minlength=facet_count),
                        dtype=np.int64,
                    ),
                )
            )
            for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
                for identifier, vertices, dofs in zip(
                    np.asarray(block.global_ids),
                    np.asarray(block.vertices),
                    np.asarray(route),
                    strict=True,
                ):
                    controls = tuple(values[int(dof)] for dof in dofs)
                    with ledger.temporary_scope():
                        resolved = (
                            algebra._prepare_coordinate_expressions(element, controls)
                            is not None
                        )
                    if not resolved:
                        raise ValueError(
                            "Mapped source face requires its actual original coordinate expression."
                        )
                    root = _MappedFacetRoot(
                        block.cell_kind,
                        tuple(int(vertex_ids[row]) for row in vertices),
                        element,
                        controls,
                    )
                    # Each retained root record and its exact controls are charged once.
                    ledger.retain_basis(root.vertices)
                    ledger.retain_basis(root.controls)
                    roots[int(identifier)] = root
            facets: list[tuple[int, ...]] = []
            for facet in range(facet_count):
                triangle_rows = order[bounds[facet] : bounds[facet + 1]]
                ledger.reserve(triangle_rows.size + 1, 24 * triangle_rows.size)
                pieces = triangle_vertices[triangle_rows]
                if pieces.shape == (1, 3):
                    rows = tuple(int(row) for row in pieces[0])
                elif (
                    pieces.shape == (2, 3)
                    and pieces[0, 0] == pieces[1, 0]
                    and pieces[0, 2] == pieces[1, 1]
                ):
                    rows = (*tuple(int(row) for row in pieces[0]), int(pieces[1, 2]))
                else:
                    raise ValueError(
                        "Mapped facet authority requires its exact original oriented face declaration."
                    )
                facets.append(tuple(int(vertex_ids[row]) for row in rows))
            facet_vertex_ids = tuple(facets)
            ledger.retain_basis(facet_vertex_ids)
        ledger.reserve(len(roots), 128 * len(roots))
        return cls(mesh, geometry, roots, facet_vertex_ids)

    def support(
        self,
        proof: MappedPlcSupport,
        dimension: int,
        row: int,
        facet: int,
        work: list[int],
        /,
    ) -> tuple[bool, int]:
        """Exact source-face support, charged to the support preparation's ledger."""
        with proof.ledger.activate(), proof.ledger.temporary_scope():
            return self._support(proof, dimension, row, facet, work)

    def _support(
        self,
        proof: MappedPlcSupport,
        dimension: int,
        row: int,
        facet: int,
        work: list[int],
        /,
    ) -> tuple[bool, int]:
        from ..meshing._plc_mapped_support import _CellMap, _charts, _spend

        if dimension not in (0, 1, 2) or facet < 0 or facet >= len(self.facet_vertex_ids):
            raise ValueError(
                "Mapped source-face query requires exact declared source/target strata."
            )
        source_vertices = self.facet_vertex_ids[facet]
        source_key = set(source_vertices)
        orientations: set[int] = set()
        checked = False
        for parent in proof.parents(dimension, row):
            current = proof.cells[parent]
            root = self.roots.get(current.root_cell_id)
            if root is None:
                continue
            checked = True
            _spend(work, 1)
            proof.ledger.reserve(
                len(current.root_vertex_ids) + sum(len(row) for row in current.controls)
            )
            if (
                current.root_vertex_ids != root.vertices
                or current.controls != root.controls
            ):
                raise ValueError(
                    "Mapped source face altered its original root vertices or exact controls."
                )
            topology = reference_cell_topology(root.kind)
            candidates = tuple(
                local
                for local in topology.entities[2]
                if {root.vertices[index] for index in local} == source_key
            )
            if len(candidates) != 1:
                return False, 0
            reference = _polynomial_reference_chart(current.element, root.element)
            if reference is None:
                raise ValueError(
                    "Mapped source face lost its actual original root chart chain."
                )
            reference_element, controls = reference
            values = tuple(
                tuple(Fraction(float(value)) for value in coordinate)
                for coordinate in controls
            )
            coordinates = algebra.coordinate_expressions(reference_element, values)
            if coordinates is None:
                raise ValueError("Mapped source-face reference expression is unresolved.")
            ordered = tuple(
                root.vertices.index(identifier) for identifier in source_vertices
            )
            source_points = tuple(
                tuple(Fraction(float(value)) for value in topology.vertices[index])
                for index in ordered
            )
            first = tuple(
                b - a for a, b in zip(source_points[0], source_points[1], strict=True)
            )
            second = tuple(
                b - a for a, b in zip(source_points[0], source_points[-1], strict=True)
            )
            normal = _cross(first, second)
            plane_offset = -sum(
                (a * b for a, b in zip(normal, source_points[0], strict=True)),
                Fraction(0),
            )
            if not any(normal):
                raise ValueError(
                    "Original mapped source face has a collapsed reference plane."
                )
            if dimension == 0:
                corners = algebra.coordinate_corner_images(reference_element, values)
                if corners is None:
                    raise ValueError("Mapped source-face point pullback is unresolved.")
                vertex = int(proof.entity_vertices[0][row][0])
                point = corners[current.vertices.index(vertex)]
                if not _inside_reference(point, root.kind) or sum(
                    (a * b for a, b in zip(normal, point, strict=True)), plane_offset
                ):
                    return False, 0
                continue
            chart_cell = _CellMap(
                current.kind,
                current.vertices,
                coordinates,
                current.element,
                current.controls,
                current.root_cell_id,
                current.root_vertex_ids,
            )
            for chart, chart_domain in _charts(
                (chart_cell,), proof.entity_vertices[dimension][row], dimension
            ):
                plane = algebra.constant(plane_offset, dimension)
                for coefficient, coordinate in zip(normal, chart, strict=True):
                    plane = algebra.expression_add(
                        plane, algebra.expression_scale(coordinate, coefficient)
                    )
                if plane:
                    return False, 0
                if any(isinstance(value, algebra.RationalPolynomial) for value in chart):
                    raise ValueError(
                        "Mapped source-face reference inequalities require their exact polynomial chart."
                    )
                polynomials = tuple(
                    value
                    for value in chart
                    if not isinstance(value, algebra.RationalPolynomial)
                )
                constraints = _reference_constraints(polynomials, root.kind, dimension)
                for constraint in constraints:
                    used = SubdivisionLedger()
                    status = nonnegative(
                        constraint,
                        chart_domain,
                        dimension,
                        proof.limits.maximum_bernstein_nodes,
                        proof.limits.maximum_subdivision_depth,
                        min(proof.limits.maximum_subdivision_pieces, work[0]),
                        used,
                    )
                    _spend(work, used.pieces)
                    if not status:
                        return False, 0
                if dimension == 2:
                    zero = (Fraction(0),) * dimension
                    tangents = tuple(
                        tuple(
                            algebra.expression_evaluate(
                                algebra.expression_derivative(coordinate, axis), zero
                            )
                            for coordinate in chart
                        )
                        for axis in range(2)
                    )
                    product = sum(
                        (a * b for a, b in zip(normal, _cross(*tangents), strict=True)),
                        Fraction(0),
                    )
                    if product == 0:
                        raise ValueError(
                            "Mapped source-face chart orientation is unresolved."
                        )
                    orientations.add(1 if product > 0 else -1)
        if not checked:
            return False, 0
        return len(orientations) <= 1, next(iter(orientations)) if orientations else 0
