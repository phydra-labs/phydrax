#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Exact original-root strata of mixed mapped coordinate restrictions."""

from __future__ import annotations

import sys
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from .._meshcore import charge_native_geometry_queries
from ..discretization import _coordinate_enclosure as algebra
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._cell_mesh import CellMesh
from ..discretization._reference_cell import reference_cell_topology
from ..geometry._mapped_coverage import map_domain, nonnegative, SubdivisionLedger
from ..geometry._mapped_reference_coverage import _roots
from ..geometry._mapped_reference_domain import MappedReferenceDomain
from ..geometry._mesh_certificates import MeshCertificateLimits
from ..geometry._planar_coverage import _reserve_fraction_work
from ..geometry._restricted_embedding import _reference_constraints
from ._contracts import MeshingFailure, MeshingFailureCategory


if TYPE_CHECKING:
    from ._association import GeometryAssociation


def _chart_orientation(
    source: tuple[tuple[Fraction, ...], ...],
    target: tuple[tuple[Fraction, ...], ...],
    degree: int,
    /,
) -> int:
    def direction(points: tuple[tuple[Fraction, ...], ...]) -> tuple[Fraction, ...]:
        first = tuple(points[1][axis] - points[0][axis] for axis in range(3))
        if degree == 1:
            return first
        second = tuple(points[-1][axis] - points[0][axis] for axis in range(3))
        return tuple(
            first[(axis + 1) % 3] * second[(axis + 2) % 3]
            - first[(axis + 2) % 3] * second[(axis + 1) % 3]
            for axis in range(3)
        )

    _reserve_fraction_work(
        (*source, *target), 18 if degree == 2 else 6, 1, 20 if degree == 2 else 4
    )
    product = sum(
        (a * b for a, b in zip(direction(source), direction(target), strict=True)),
        Fraction(0),
    )
    if product == 0:
        raise ValueError("Mapped source and target chart orientations are unresolved.")
    return 1 if product > 0 else -1


def mapped_reference_associations(
    domain: MappedReferenceDomain,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
    *,
    maximum_support_queries: int,
) -> tuple[GeometryAssociation, ...]:
    """Classify complete exact chart images in their original mixed root strata."""
    limits = MeshCertificateLimits()
    ledger = algebra.coordinate_enclosure_budget(
        maximum_support_queries, limits.maximum_scratch_bytes
    )
    work_start = ledger.work_units
    try:
        with (
            ledger.activate(),
            ledger.bound_stage(
                maximum_support_queries,
                limits.maximum_scratch_bytes,
                starting_work_units=work_start,
            ),
            ledger.temporary_scope(),
        ):
            return _mapped_reference_associations(
                domain, mesh, geometry, maximum_support_queries, limits
            )
    except algebra.CoordinateEnclosureResourceError as error:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            f"Mapped original-root association proof exhausted its resource budget: {error}",
            stage="mapped-source-association",
        ) from error
    finally:
        ledger.charge_native_work(ledger.work_units - ledger.native_charged_work_units)


def _mapped_reference_associations(
    domain: MappedReferenceDomain,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    maximum_support_queries: int,
    limits: MeshCertificateLimits,
) -> tuple[GeometryAssociation, ...]:
    from ._association import GeometryAssociation, GeometryAssociationKind
    from ._plc_mapped_support import _CellMap, _charts, plc_mapped_vertices
    from ._quad_generation import _family_host_array

    record = geometry.restriction_source
    if (
        record is None
        or record.source_geometry_id != cell_geometry_id(domain.source_geometry)
        or record.source_topology_id != domain.reference_mesh.topology_id
    ):
        raise ValueError(
            "Mapped associations require the original source-root restriction record."
        )
    supported = ("tetrahedron", "prism", "hexahedron")
    if any(
        block.cell_kind not in supported
        for block in (*mesh.blocks, *domain.reference_mesh.blocks)
    ):
        raise ValueError(
            "Mapped association transport requires canonical tetrahedral, prismatic or hexahedral charts."
        )
    required = sum(
        sum(
            len(entities)
            for entities in reference_cell_topology(block.cell_kind).entities
        )
        * block.cell_count
        for block in mesh.blocks
    )
    if required > maximum_support_queries:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Mapped source-stratum queries exceed the declared support budget.",
            stage="mapped-source-association",
        )
    charge_native_geometry_queries(required, work_units=required)
    root_mesh = domain.reference_mesh
    root_entities = tuple(plc_mapped_vertices(root_mesh, degree) for degree in range(4))
    target_entities = tuple(plc_mapped_vertices(mesh, degree) for degree in range(4))
    root_vertices = np.asarray(root_mesh.vertex_global_ids, dtype=np.int64)
    target_vertices = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    root_lookup = tuple(
        {
            tuple(sorted(root_vertices[row[row >= 0]].tolist())): index
            for index, row in enumerate(rows)
        }
        for rows in root_entities
    )
    target_lookup = tuple(
        {
            tuple(sorted(target_vertices[row[row >= 0]].tolist())): index
            for index, row in enumerate(rows)
        }
        for rows in target_entities
    )
    choices: list[list[set[tuple[int, int, int]]]] = [
        [set() for _ in rows] for rows in target_entities
    ]
    roots = _roots(domain)
    elements, routes, _ = geometry.resolve(mesh)
    values = algebra.prepared_coordinate_source_bank(geometry)
    used = SubdivisionLedger()
    ledger = algebra._COORDINATE_BUDGET.get()
    if ledger is None:
        raise RuntimeError(
            "Mapped associations require their owning source-support ledger."
        )
    ledger.retain_basis((*root_lookup, *target_lookup))
    ledger.reserve(
        sum(len(rows) for rows in choices),
        sys.getsizeof(choices)
        + sum(
            sys.getsizeof(rows) + sum(sys.getsizeof(row) for row in rows)
            for rows in choices
        )
        + sum(
            sys.getsizeof(array)
            if array.flags.owndata
            else array.nbytes + sys.getsizeof(array)
            for array in (*root_entities, *target_entities)
        ),
    )
    root_identities: dict[int, tuple[str, str]] = {}
    for root in roots.values():
        ledger.retain_basis((root.vertices, root.controls, root.polynomial))
        ledger.reserve(1, sys.getsizeof(root) + sys.getsizeof(root.__dict__))
        root_identities[root.cell_id] = algebra._element_identity(root.element)
    ledger.retain_basis(tuple(root_identities.items()))
    chains: dict[
        tuple[tuple[str, str], tuple[str, str]], tuple[algebra.Polynomial, ...]
    ] = {}
    reference_templates: dict[
        str, tuple[tuple[algebra.Polynomial, ...], tuple[tuple[Fraction, ...], ...]]
    ] = {}
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        topology = reference_cell_topology(block.cell_kind)
        element_identity = algebra._element_identity(element)
        ledger.retain_basis(element_identity)
        parents = np.asarray(record.block_parent_cell_ids[block.name], dtype=np.int64)
        parent_vertices = np.asarray(
            record.block_parent_vertex_ids[block.name], dtype=np.int64
        )
        cells = np.asarray(block.vertices, dtype=np.int64)
        route_rows = np.asarray(route, dtype=np.int64)
        if (
            cells.ndim != 2
            or route_rows.ndim != 2
            or parents.ndim != 1
            or parent_vertices.ndim != 2
        ):
            raise ValueError(
                "Mapped source incidence requires canonical rank-two cell and coefficient routes."
            )
        for index in range(block.cell_count):
            with ledger.temporary_scope():
                cell = cells[index]
                controls = tuple(values[int(dof)] for dof in route_rows[index])
                parent = int(parents[index])
                root = roots.get(parent)
                if root is None:
                    raise ValueError(
                        "Mapped restriction names an undeclared source root."
                    )
                chain_key = (element_identity, root_identities[parent])
                coordinates = chains.get(chain_key)
                if coordinates is None:
                    _, expressions = algebra.coordinate_reference_chain(
                        element, ancestor=root.element
                    )
                    if any(
                        isinstance(value, algebra.RationalPolynomial)
                        for value in expressions
                    ):
                        raise ValueError(
                            "Mapped reference strata require exact polynomial charts."
                        )
                    coordinates = tuple(
                        value
                        for value in expressions
                        if not isinstance(value, algebra.RationalPolynomial)
                    )
                    chains[chain_key] = coordinates
                    ledger.retain_basis((chain_key, coordinates))
                ledger.reserve(len(controls) + len(root.vertices))
                if (
                    controls != root.controls
                    or tuple(int(value) for value in parent_vertices[index] if value >= 0)
                    != root.vertices
                ):
                    raise ValueError(
                        "Mapped restriction altered its original root expression or controls."
                    )
                constraints = _reference_constraints(coordinates, root.kind, 3)
                for constraint in constraints:
                    if not nonnegative(
                        constraint,
                        map_domain(block.cell_kind),
                        3,
                        limits.maximum_bernstein_nodes,
                        limits.maximum_subdivision_depth,
                        min(limits.maximum_subdivision_pieces, maximum_support_queries),
                        used,
                    ):
                        if used.pieces > min(
                            limits.maximum_subdivision_pieces, maximum_support_queries
                        ):
                            raise MeshingFailure(
                                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                                "Mapped root containment exhausted its subdivision budget.",
                                stage="mapped-source-association",
                            )
                        raise ValueError(
                            "Mapped target chart containment in its original root is invalid or unresolved."
                        )
                template = reference_templates.get(root.kind)
                if template is None:
                    root_expressions = _reference_constraints(
                        algebra.axes(3), root.kind, 3
                    )
                    if any(
                        isinstance(value, algebra.RationalPolynomial)
                        for value in root_expressions
                    ):
                        raise RuntimeError(
                            "Reference-cell facet equations changed to rational quotients."
                        )
                    root_constraints = tuple(
                        value
                        for value in root_expressions
                        if not isinstance(value, algebra.RationalPolynomial)
                    )
                    root_points = tuple(
                        tuple(Fraction(float(value)) for value in position)
                        for position in reference_cell_topology(root.kind).vertices
                    )
                    template = (root_constraints, root_points)
                    reference_templates[root.kind] = template
                    ledger.retain_basis((root.kind, template))
                root_constraints, root_points = template
                corners = tuple(
                    tuple(
                        algebra.evaluate(
                            value, tuple(Fraction(float(x)) for x in position)
                        )
                        for value in coordinates
                    )
                    for position in topology.vertices
                )
                chart_cell = _CellMap(
                    block.cell_kind,
                    tuple(cell.tolist()),
                    coordinates,
                    element,
                    controls,
                    parent,
                    root.vertices,
                )
                for degree in range(4):
                    for local in topology.entities[degree]:
                        key = tuple(sorted(target_vertices[cell[list(local)]].tolist()))
                        target_row = target_lookup[degree][key]
                        if degree == 0:
                            charts = (
                                (
                                    tuple(
                                        algebra.constant(value, 0)
                                        for value in corners[local[0]]
                                    ),
                                    "box",
                                ),
                            )
                        elif degree == 3:
                            charts = ((coordinates, map_domain(block.cell_kind)),)
                        else:
                            # Canonical incidence order owns orientation; full
                            # polynomial restrictions own edge and face support.
                            charts = _charts(
                                (chart_cell,), target_entities[degree][target_row], degree
                            )
                        if not charts:
                            raise ValueError(
                                "Mapped target entity lost its complete original-root chart."
                            )
                        fixed: set[int] | None = None
                        for chart, _ in charts:
                            if any(
                                isinstance(value, algebra.RationalPolynomial)
                                for value in chart
                            ):
                                raise ValueError(
                                    "Mapped reference strata require their exact polynomial chart."
                                )
                            polynomials = tuple(
                                value
                                for value in chart
                                if not isinstance(value, algebra.RationalPolynomial)
                            )
                            vanishing = {
                                facet
                                for facet, constraint in enumerate(
                                    _reference_constraints(polynomials, root.kind, degree)
                                )
                                if not constraint
                            }
                            fixed = vanishing if fixed is None else fixed & vanishing
                        if fixed is None:
                            raise RuntimeError(
                                "Mapped root stratum proof has no chart equations."
                            )
                        source_key = tuple(
                            sorted(
                                root.vertices[vertex]
                                for vertex, point in enumerate(root_points)
                                if all(
                                    algebra.evaluate(root_constraints[facet], point) == 0
                                    for facet in fixed
                                )
                            )
                        )
                        source = next(
                            (
                                (source_degree, lookup[source_key])
                                for source_degree, lookup in enumerate(root_lookup)
                                if source_key in lookup
                            ),
                            None,
                        )
                        if source is None or source[0] < degree:
                            raise ValueError(
                                "Mapped chart stratum has no dimensionally valid original source entity."
                            )
                        source_degree, source_row = source
                        orientation = 0
                        if degree == source_degree and degree in (1, 2):
                            source_rows = root_entities[source_degree][source_row]
                            source_points = tuple(
                                root_points[root.vertices.index(int(identifier))]
                                for identifier in root_vertices[
                                    source_rows[source_rows >= 0]
                                ]
                            )
                            chart_orientations: set[int] = set()
                            for chart, _ in charts:
                                zero = (Fraction(0),) * degree
                                tangents = tuple(
                                    tuple(
                                        algebra.expression_evaluate(
                                            algebra.expression_derivative(value, axis),
                                            zero,
                                        )
                                        for value in chart
                                    )
                                    for axis in range(degree)
                                )
                                chart_orientations.add(
                                    _chart_orientation(
                                        source_points,
                                        ((Fraction(0),) * 3, *tangents),
                                        degree,
                                    )
                                )
                            if len(chart_orientations) != 1:
                                raise ValueError(
                                    "Mapped entity charts have contradictory original-source orientations."
                                )
                            orientation = next(iter(chart_orientations))
                        candidate = (source_degree, source_row, orientation)
                        if candidate not in choices[degree][target_row]:
                            ledger.retain_basis(candidate)
                            choices[degree][target_row].add(candidate)
    output = []
    for degree, rows in enumerate(choices):
        dimensions = _family_host_array((len(rows),), np.int64)
        identifiers = _family_host_array((len(rows),), np.int64)
        orientations = _family_host_array((len(rows),), np.int8)
        for row, candidates in enumerate(rows):
            if not candidates:
                raise ValueError("Mapped target entity has no source-root incidence.")
            smallest = min(dimension for dimension, _, _ in candidates)
            selected = {
                (dimension, index, orientation)
                for dimension, index, orientation in candidates
                if dimension == smallest
            }
            if len(selected) != 1:
                raise ValueError(
                    "Mapped target entity has contradictory original source strata."
                )
            dimensions[row], index, orientations[row] = next(iter(selected))
            identifiers[row] = np.asarray(
                root_mesh.topology.entities(smallest).entity_ids
            )[index]
        names = tuple(
            domain.image_entity_id(int(dimension), int(identifier))
            for dimension, identifier in zip(dimensions, identifiers, strict=True)
        )
        target_ids = np.asarray(mesh.topology.entities(degree).entity_ids, dtype=np.int64)
        output.append(
            GeometryAssociation(
                GeometryAssociationKind.MAPPED_REFERENCE,
                domain.source_id,
                domain.source_revision,
                mesh.topology.entities(degree).entity_set_id,
                target_ids,
                names,
                np.zeros(target_ids.shape, dtype=np.float64),
                exact=True,
                parent_dimensions=dimensions,
                parent_ids=identifiers,
                orientations=orientations,
            )
        )
    return tuple(output)
