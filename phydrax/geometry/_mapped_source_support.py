#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Root-face support for mixed coordinate-map restrictions."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._reference_cell_types import CellGeometryElement
from phydrax.geometry._embedding_cells import inside_reference, reference_constraints

from ..discretization import _coordinate_enclosure as algebra, CellMesh
from ..discretization._cell_geometry import CellGeometrySpec, equivalent_coordinate_root
from ..discretization._reference_cell import reference_cell_topology
from ..meshing._support_work import spend_support_work
from ..typing import checked
from ._mapped_coverage import nonnegative, SubdivisionLedger
from ._mapped_reference_coverage import PreparedPolynomialReferenceCharts
from ._rational import rational_cross, rational_vector


@dataclass(frozen=True)
class _MappedFacetRoot:
    kind: str
    vertices: tuple[int, ...]
    element: CellGeometryElement
    controls: algebra.CoordinateSourceBank


def _source_face_plane(
    root: _MappedFacetRoot,
    source_vertices: tuple[int, ...],
) -> tuple[tuple[Fraction, ...], Fraction]:
    topology = reference_cell_topology(root.kind)
    ordered = tuple(root.vertices.index(identifier) for identifier in source_vertices)
    source_points = tuple(rational_vector(topology.vertices[index]) for index in ordered)
    first = tuple(b - a for a, b in zip(source_points[0], source_points[1], strict=True))
    second = tuple(
        b - a for a, b in zip(source_points[0], source_points[-1], strict=True)
    )
    normal = rational_cross(first, second)
    plane_offset = -sum(
        (a * b for a, b in zip(normal, source_points[0], strict=True)), Fraction(0)
    )
    if not any(normal):
        raise ValueError(
            "MappedSourceFacetAuthority.face must span a reference plane; got collapsed face"
        )
    return normal, plane_offset


def _point_on_source_face(
    proof: plc_mapped_support_owner.MappedPlcSupport,
    row: int,
    current: plc_mapped_support_owner.CellMap,
    root: _MappedFacetRoot,
    corners: algebra.CoordinateSourceBank | None,
    normal: tuple[Fraction, ...],
    plane_offset: Fraction,
) -> bool:
    if corners is None:
        raise ValueError(
            "MappedSourceFacetAuthority.point must resolve its pullback; got unresolved corner images"
        )
    vertex = int(proof.entity_vertices[0][row][0])
    point = tuple(
        algebra.require_rational_reference_scalar(value)
        for value in corners[current.vertices.index(vertex)]
    )
    return inside_reference(point, root.kind) and not sum(
        (a * b for a, b in zip(normal, point, strict=True)), plane_offset
    )


def _chart_on_source_face(
    chart: tuple[algebra.Expression, ...],
    chart_domain: algebra.BernsteinDomain,
    normal: tuple[Fraction, ...],
    plane_offset: Fraction,
    root_kind: str,
    dimension: int,
    proof: plc_mapped_support_owner.MappedPlcSupport,
    work: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[bool, int, int]:
    plane = algebra.constant(plane_offset, dimension, budget=budget)
    for coefficient, coordinate in zip(normal, chart, strict=True):
        plane = algebra.expression_add(
            plane,
            algebra.expression_scale(coordinate, coefficient, budget=budget),
            budget=budget,
        )
    if plane:
        return False, 0, work
    if any(isinstance(value, algebra.RationalPolynomial) for value in chart):
        raise ValueError(
            "MappedSourceFacetAuthority.chart must have polynomial reference inequalities; got rational chart"
        )
    polynomials = tuple(
        value for value in chart if not isinstance(value, algebra.RationalPolynomial)
    )
    constraints = reference_constraints(polynomials, root_kind, dimension, budget=budget)
    for constraint in constraints:
        used = SubdivisionLedger()
        status = nonnegative(
            constraint,
            chart_domain,
            dimension,
            proof.limits.maximum_bernstein_nodes,
            proof.limits.maximum_subdivision_depth,
            min(proof.limits.maximum_subdivision_pieces, work),
            used,
            budget=budget,
        )
        work = spend_support_work(
            work,
            used.pieces,
            "MappedSourceFacetAuthority.work must cover source-support queries; got exhausted query budget",
        )
        if not status:
            return False, 0, work
    if dimension != 2:
        return True, 0, work
    zero = (Fraction(0),) * dimension
    tangents = tuple(
        tuple(
            algebra.expression_evaluate(
                algebra.expression_derivative(coordinate, axis, budget=budget),
                zero,
                budget=budget,
            )
            for coordinate in chart
        )
        for axis in range(2)
    )
    product = sum(
        (a * b for a, b in zip(normal, rational_cross(*tangents), strict=True)),
        Fraction(0),
    )
    if product == 0:
        raise ValueError(
            "MappedSourceFacetAuthority.orientation must be nonzero; got unresolved tangent orientation"
        )
    return True, 1 if product > 0 else -1, work


@dataclass(frozen=True)
class MappedSourceFacetAuthority:
    """Measured source cell expressions and their declared source-face incidence."""

    mesh: CellMesh
    geometry: CellGeometrySpec
    roots: dict[int, _MappedFacetRoot]
    facet_vertex_ids: tuple[tuple[int, ...], ...]
    reference_charts: PreparedPolynomialReferenceCharts = field(
        default_factory=PreparedPolynomialReferenceCharts
    )

    @classmethod
    @checked
    def prepare(
        cls,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        triangle_vertices: np.ndarray,
        triangle_facets: np.ndarray,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> MappedSourceFacetAuthority:
        """Bind measured source cell expressions under the bound support ledger.

        ``ledger`` is the support preparation's own rational-expression ledger; its
        remaining allowance is consumed here and by every later face query.
        """
        elements, routes, _ = geometry.resolve(mesh)
        values = geometry.source_coordinates(budget=budget)
        vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        roots: dict[int, _MappedFacetRoot] = {}
        with budget.activate(), budget.temporary_scope():
            facet_count = mesh.entity_set(2).count
            if (
                triangle_vertices.ndim != 2
                or triangle_vertices.shape[1] != 3
                or triangle_facets.shape != (triangle_vertices.shape[0],)
                or np.any(triangle_facets < 0)
                or np.any(triangle_facets >= facet_count)
            ):
                raise ValueError(
                    "MappedSourceFacetAuthority.triangle_facets must index source faces for every triangle; got invalid incidence shape or rows"
                )
            # Build the source-face index once. Repeated full-bank masks would
            # perform quadratic preparation work before any support query.
            budget.reserve(
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
                    with budget.temporary_scope():
                        resolved = (
                            algebra.prepare_coordinate_expressions(
                                element, controls, budget=budget
                            )
                            is not None
                        )
                    if not resolved:
                        raise ValueError(
                            "MappedSourceFacetAuthority.root must resolve its coordinate expression; got unresolved source map"
                        )
                    root = _MappedFacetRoot(
                        block.cell_kind,
                        tuple(int(vertex_ids[row]) for row in vertices),
                        element,
                        controls,
                    )
                    # Each retained root record and its rational controls are charged once.
                    budget.retain_basis(root.vertices)
                    budget.retain_basis(root.controls)
                    roots[int(identifier)] = root
            facets: list[tuple[int, ...]] = []
            for facet in range(facet_count):
                triangle_rows = order[bounds[facet] : bounds[facet + 1]]
                budget.reserve(triangle_rows.size + 1, 24 * triangle_rows.size)
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
                        "MappedSourceFacetAuthority.triangle_vertices must encode one oriented source face; got incompatible triangle pieces"
                    )
                facets.append(tuple(int(vertex_ids[row]) for row in rows))
            facet_vertex_ids = tuple(facets)
            budget.retain_basis(facet_vertex_ids)
        budget.reserve(len(roots), 128 * len(roots))
        return cls(mesh, geometry, roots, facet_vertex_ids)

    @checked
    def support(
        self,
        proof: plc_mapped_support_owner.MappedPlcSupport,
        dimension: int,
        row: int,
        facet: int,
        work: int,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[bool, int, int]:
        """Return source-face support, orientation, and remaining query work."""
        with proof.ledger.activate(), proof.ledger.temporary_scope():
            return self._support(proof, dimension, row, facet, work, budget=budget)

    def _support(
        self,
        proof: plc_mapped_support_owner.MappedPlcSupport,
        dimension: int,
        row: int,
        facet: int,
        work: int,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[bool, int, int]:

        if dimension not in (0, 1, 2) or facet < 0 or facet >= len(self.facet_vertex_ids):
            raise ValueError(
                "MappedSourceFacetAuthority.query must name declared strata and a source facet; got invalid dimension or facet"
            )
        source_vertices = self.facet_vertex_ids[facet]
        source_key = set(source_vertices)
        orientations: set[int] = set()
        checked_root = False
        for parent in proof.parents(dimension, row):
            current = proof.cells[parent]
            root = self.roots.get(current.root_cell_id)
            if root is None:
                continue
            checked_root = True
            work = spend_support_work(
                work,
                1,
                "MappedSourceFacetAuthority.work must cover source-support queries; got exhausted query budget",
            )
            proof.ledger.reserve(
                len(current.root_vertex_ids) + sum(len(row) for row in current.controls)
            )
            equivalent = (
                None
                if current.root_vertex_ids != root.vertices
                else equivalent_coordinate_root(
                    current.element,
                    current.controls,
                    root.element,
                    root.controls,
                    budget=budget,
                )
            )
            if equivalent is None:
                raise ValueError(
                    "MappedSourceFacetAuthority.root must retain source vertices and controls; got changed root data"
                )
            topology = reference_cell_topology(root.kind)
            candidates = tuple(
                local
                for local in topology.entities[2]
                if {root.vertices[index] for index in local} == source_key
            )
            if len(candidates) != 1:
                return False, 0, work
            reference = self.reference_charts.prepare(
                current.element, equivalent, budget=budget
            )
            if reference is None:
                raise ValueError(
                    "MappedSourceFacetAuthority.reference must retain the root chart chain; got unresolved chart"
                )
            reference_element, controls = reference
            values = tuple(rational_vector(coordinate) for coordinate in controls)
            corners = None
            if dimension == 0:
                coordinates, corners = algebra.coordinate_map_images(
                    reference_element, values, budget=budget
                )
            else:
                coordinates = algebra.coordinate_expressions(
                    reference_element, values, budget=budget
                )
            if coordinates is None:
                raise ValueError(
                    "MappedSourceFacetAuthority.reference must resolve coordinate expressions; got unresolved expressions"
                )
            normal, plane_offset = _source_face_plane(root, source_vertices)
            if dimension == 0:
                if not _point_on_source_face(
                    proof,
                    row,
                    current,
                    root,
                    corners,
                    normal,
                    plane_offset,
                ):
                    return False, 0, work
                continue
            chart_cell = plc_mapped_support_owner.CellMap(
                current.kind,
                current.vertices,
                coordinates,
                current.element,
                current.controls,
                current.root_cell_id,
                current.root_vertex_ids,
            )
            for chart, chart_domain in plc_mapped_support_owner.mapped_reference_charts(
                (chart_cell,),
                proof.entity_vertices[dimension][row],
                dimension,
                budget=budget,
            ):
                supported, orientation, work = _chart_on_source_face(
                    chart,
                    chart_domain,
                    normal,
                    plane_offset,
                    root.kind,
                    dimension,
                    proof,
                    work,
                    budget=budget,
                )
                if not supported:
                    return False, 0, work
                if orientation:
                    orientations.add(orientation)
        if not checked_root:
            return False, 0, work
        return (
            len(orientations) <= 1,
            next(iter(orientations)) if orientations else 0,
            work,
        )


# Mapped PLC preparation consumes facet support; nominal checking resolves after both declarations load.
import phydrax.meshing._plc_mapped_support as plc_mapped_support_owner
