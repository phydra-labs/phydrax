#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Coverage by exact root restrictions, including collapsed rational charts."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ..discretization import _coordinate_enclosure as algebra
from ..discretization._cell_geometry import (
    CellGeometryElement,
    CellGeometrySpec,
    coordinate_lagrange_element,
    PolynomialComposedCellGeometryElement,
    RationalComposedCellGeometryElement,
    RestrictedCellGeometryElement,
)
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    certify_cell_geometry_validity,
)
from ..discretization._cell_mesh import CellBlock, CellMesh
from ..discretization._reference_cell import reference_cell_topology
from ._mapped_coverage import integrate, map_domain
from ._mapped_reference_domain import (
    mapped_source_corner_coordinates,
    MappedReferenceDomain,
)
from ._mesh_certificates import (
    _active_expression_work,
    _coverage_result,
    _CoverageOverlapKey,
    _EmbeddingState,
    _facet_entity_ids,
    _mapped_coverage_measures,
    _mesh_facets,
    certify_domain_coverage,
    certify_global_embedding,
    DomainCoverageCertificate,
    GlobalEmbeddingCertificate,
    MeshCertificateBinding,
    MeshCertificateLimits,
    PiecewiseLinearDomain,
)
from ._planar_coverage import project, rational_points, signed_measure


@dataclass(frozen=True)
class _Root:
    cell_id: int
    kind: str
    region: int
    vertices: tuple[int, ...]
    element: CellGeometryElement
    controls: algebra.CoordinateSourceBank
    polynomial: tuple[algebra.Polynomial, ...]


@dataclass(frozen=True)
class _Chart:
    block: str
    cell_id: int
    kind: str
    vertices: tuple[int, ...]
    matrix: tuple[tuple[Fraction, ...], ...] | None
    offset: tuple[Fraction, ...] | None
    reference_element: CellGeometryElement | None = None
    reference_controls: np.ndarray | None = None


def _roots(domain: MappedReferenceDomain) -> dict[int, _Root]:
    elements, routes, _ = domain.source_geometry.resolve(domain.reference_mesh)
    values = algebra.prepared_coordinate_source_bank(domain.source_geometry)
    vertex_ids = np.asarray(domain.reference_mesh.vertex_global_ids, dtype=np.int64)
    result = {}
    cursor = 0
    for block, element, route in zip(
        domain.reference_mesh.blocks, elements, routes, strict=True
    ):
        local_values = tuple(
            tuple(values[index] for index in row)
            for row in np.asarray(route, dtype=np.int64)
        )
        for local, cell_id, row in zip(
            local_values,
            np.asarray(block.global_ids),
            np.asarray(block.vertices),
            strict=True,
        ):
            polynomial = algebra.coordinate_polynomials(element, local)
            if polynomial is None:
                raise ValueError("Mapped source root expression is absent.")
            result[int(cell_id)] = _Root(
                int(cell_id),
                block.cell_kind,
                int(domain.cell_regions[cursor]),
                tuple(int(value) for value in vertex_ids[row]),
                element,
                local,
                polynomial,
            )
            cursor += 1
    return result


def physical_source_mesh(domain: MappedReferenceDomain) -> CellMesh:
    """An exact corner view of the independently declared source coordinate maps."""
    points = mapped_source_corner_coordinates(
        domain.reference_mesh, domain.source_geometry
    )
    return domain.reference_mesh.with_coordinates(
        points, numeric_version=domain.source_revision
    )


def certify_mapped_reference_source(
    domain: MappedReferenceDomain, limits: MeshCertificateLimits
) -> tuple[CellMesh, GlobalEmbeddingCertificate, DomainCoverageCertificate]:
    """Independently establish the source reference partition and physical embedding."""
    reference_geometry = CellGeometrySpec.affine(domain.reference_mesh)
    reference_validity = certify_cell_geometry_validity(
        reference_geometry, mesh=domain.reference_mesh
    )
    reference_embedding = certify_global_embedding(
        domain.reference_mesh, reference_geometry, reference_validity, limits=limits
    )
    reference_coverage = certify_domain_coverage(
        domain.reference_mesh,
        reference_geometry,
        domain.reference_domain,
        domain.cell_regions,
        embedding=reference_embedding,
        limits=limits,
    )
    physical_mesh = physical_source_mesh(domain)
    physical_validity = certify_cell_geometry_validity(
        domain.source_geometry, mesh=physical_mesh
    )
    physical_embedding = certify_global_embedding(
        physical_mesh, domain.source_geometry, physical_validity, limits=limits
    )
    return physical_mesh, physical_embedding, reference_coverage


def _restriction(
    element: CellGeometryElement, root: _Root, dimension: int
) -> tuple[tuple[tuple[Fraction, ...], ...], tuple[Fraction, ...]] | None:
    matrix = tuple(
        tuple(Fraction(int(i == j)) for j in range(dimension)) for i in range(dimension)
    )
    offset = (Fraction(0),) * dimension
    while element.element_id != root.element.element_id and isinstance(
        element,
        (
            RestrictedCellGeometryElement,
            PolynomialComposedCellGeometryElement,
            RationalComposedCellGeometryElement,
        ),
    ):
        if isinstance(element, RationalComposedCellGeometryElement):
            # Preserve the physical-reference rational law through its actual
            # owner in _polynomial_reference_chart, never an affine surrogate.
            return None
        if isinstance(element, RestrictedCellGeometryElement):
            transform = tuple(
                tuple(Fraction(float(value)) for value in row)
                for row in np.asarray(element.matrix)
            )
            translation = tuple(
                Fraction(float(value)) for value in np.asarray(element.offset)
            )
        else:
            ancestor, expressions = algebra.coordinate_reference_chain(
                element, ancestor=element.source_element
            )
            if (
                ancestor.element_id != element.source_element.element_id
                or len(expressions) != dimension
                or any(
                    isinstance(value, algebra.RationalPolynomial) for value in expressions
                )
            ):
                return None
            arguments = tuple(
                value
                for value in expressions
                if not isinstance(value, algebra.RationalPolynomial)
            )
            if any(
                len(exponent) != dimension or sum(exponent) > 1
                for argument in arguments
                for exponent in argument
            ):
                return None
            transform = tuple(
                tuple(
                    argument.get(
                        tuple(int(axis == column) for axis in range(dimension)),
                        Fraction(),
                    )
                    for column in range(dimension)
                )
                for argument in arguments
            )
            translation = tuple(
                argument.get((0,) * dimension, Fraction()) for argument in arguments
            )
        matrix = tuple(
            tuple(
                sum(
                    (transform[i][k] * matrix[k][j] for k in range(dimension)),
                    Fraction(0),
                )
                for j in range(dimension)
            )
            for i in range(dimension)
        )
        offset = tuple(
            sum((transform[i][k] * offset[k] for k in range(dimension)), translation[i])
            for i in range(dimension)
        )
        element = element.source_element
    if (
        element.element_id != root.element.element_id
        or algebra.coordinate_source_signature(element)
        != algebra.coordinate_source_signature(root.element)
    ):
        return None
    return matrix, offset


def _polynomial_reference_chart(
    element: CellGeometryElement,
    root_element: CellGeometryElement,
) -> tuple[CellGeometryElement, np.ndarray] | None:
    """Retain each exact chart owner and coefficients, replacing only the root by identity."""
    from ..discretization.fem._reference import FiniteElementSpec

    if element.element_id == root_element.element_id:
        identity = coordinate_lagrange_element(root_element.cell_kind, 1)
        return identity, np.asarray(identity.reference_nodes, dtype=np.float64)
    if not isinstance(
        element,
        (
            RestrictedCellGeometryElement,
            PolynomialComposedCellGeometryElement,
            RationalComposedCellGeometryElement,
        ),
    ):
        return None
    source = _polynomial_reference_chart(element.source_element, root_element)
    if source is None:
        return None
    reference, controls = source
    if not isinstance(
        reference,
        (
            FiniteElementSpec,
            RestrictedCellGeometryElement,
            PolynomialComposedCellGeometryElement,
            RationalComposedCellGeometryElement,
        ),
    ):
        raise TypeError("Reference chart reconstruction lost its scalar source owner.")
    if isinstance(element, RestrictedCellGeometryElement):
        return RestrictedCellGeometryElement(
            reference, element.cell_kind, element.matrix, element.offset
        ), controls
    numerator_dtype = (
        np.int64
        if any(a < 0 for row in element.chart_coefficients for a, _ in row)
        else np.uint64
    )
    numerators = np.asarray(
        [[a for a, _ in row] for row in element.chart_coefficients], dtype=numerator_dtype
    )
    denominators = np.asarray(
        [[b for _, b in row] for row in element.chart_coefficients], dtype=np.uint64
    )
    return type(element)(
        reference, element.chart_element, numerators, denominators
    ), controls


def _target_charts(
    state: _EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    regions: np.ndarray,
    domain: MappedReferenceDomain,
    roots: dict[int, _Root],
) -> dict[int, list[_Chart]] | None:
    record = geometry.restriction_source
    if record is None:
        # Exact independently declared identity is a root restriction theorem,
        # not an inference from matching dimensions, nodes or fitted values.
        if (
            cell_geometry_id(geometry) != cell_geometry_id(domain.source_geometry)
            or mesh.topology_id != domain.reference_mesh.topology_id
        ):
            state.add("mapped_domain_restriction_source", "unresolved", "mesh")
            return None
        if not np.array_equal(regions, domain.cell_regions):
            state.add("mapped_domain_region_identity", "violated", "mesh")
            return None
        dimension = domain.ambient_dimension
        matrix = tuple(
            tuple(Fraction(i == j) for j in range(dimension)) for i in range(dimension)
        )
        offset = (Fraction(0),) * dimension
        vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        return {
            int(cell_id): [
                _Chart(
                    block.name,
                    int(cell_id),
                    block.cell_kind,
                    tuple(int(value) for value in vertex_ids[row]),
                    matrix,
                    offset,
                )
            ]
            for block in mesh.blocks
            for cell_id, row in zip(
                np.asarray(block.global_ids), np.asarray(block.vertices), strict=True
            )
        }
    if (
        record.source_geometry_id != cell_geometry_id(domain.source_geometry)
        or record.source_topology_id != domain.reference_mesh.topology_id
    ):
        state.add("mapped_domain_source_identity", "violated", "mesh")
        return None
    elements, routes, _ = geometry.resolve(mesh)
    values = geometry.source_coordinates()
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    parents = record.block_parent_cell_ids
    corners = record.block_parent_vertex_ids
    result: dict[int, list[_Chart]] = {cell_id: [] for cell_id in roots}
    cursor = 0
    valid = True
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        local_values = tuple(
            tuple(values[index] for index in row)
            for row in np.asarray(route, dtype=np.int64)
        )
        for local, cell_id, row, parent, parent_vertices in zip(
            local_values,
            np.asarray(block.global_ids),
            np.asarray(block.vertices),
            np.asarray(parents[block.name]),
            np.asarray(corners[block.name]),
            strict=True,
        ):
            root = roots.get(int(parent))
            if root is None:
                state.add(
                    "mapped_domain_parent_identity", "violated", "cell", (int(cell_id),)
                )
                valid = False
                cursor += 1
                continue
            chart = _restriction(element, root, domain.ambient_dimension)
            chain = element
            composed = False
            while chain.element_id != root.element.element_id and isinstance(
                chain,
                (
                    RestrictedCellGeometryElement,
                    PolynomialComposedCellGeometryElement,
                    RationalComposedCellGeometryElement,
                ),
            ):
                composed |= isinstance(
                    chain,
                    (
                        PolynomialComposedCellGeometryElement,
                        RationalComposedCellGeometryElement,
                    ),
                )
                chain = chain.source_element
            if composed:
                chart = None
            reference = (
                _polynomial_reference_chart(element, root.element)
                if chart is None
                else None
            )
            if (
                (chart is None and reference is None)
                or tuple(int(value) for value in parent_vertices if value >= 0)
                != root.vertices
                or local != root.controls
            ):
                state.add(
                    "mapped_domain_root_expression", "violated", "cell", (int(cell_id),)
                )
                valid = False
            elif int(regions[cursor]) != root.region:
                state.add(
                    "mapped_domain_region_identity", "violated", "cell", (int(cell_id),)
                )
                valid = False
            else:
                vertices = tuple(int(value) for value in vertex_ids[row])
                if chart is not None:
                    result[root.cell_id].append(
                        _Chart(
                            block.name, int(cell_id), block.cell_kind, vertices, *chart
                        )
                    )
                elif reference is not None:
                    result[root.cell_id].append(
                        _Chart(
                            block.name,
                            int(cell_id),
                            block.cell_kind,
                            vertices,
                            None,
                            None,
                            *reference,
                        )
                    )
            cursor += 1
    return result if valid else None


def _chart_mesh(
    charts: list[_Chart], dimension: int
) -> tuple[CellMesh, CellGeometrySpec] | None:
    if any(chart.reference_element is not None for chart in charts):
        return _exact_chart_mesh(charts, dimension)
    positions: dict[int, tuple[Fraction, ...]] = {}
    grouped: dict[str, list[_Chart]] = {}
    for chart in charts:
        grouped.setdefault(chart.block, []).append(chart)
        if chart.matrix is None or chart.offset is None:
            raise RuntimeError("An affine reference chart lost its exact transform.")
        for vertex, corner in zip(
            chart.vertices, reference_cell_topology(chart.kind).vertices, strict=True
        ):
            point = tuple(
                sum(
                    (
                        weight * Fraction(value)
                        for weight, value in zip(row, corner, strict=True)
                    ),
                    offset,
                )
                for row, offset in zip(chart.matrix, chart.offset, strict=True)
            )
            if any(Fraction(float(value)) != value for value in point) or (
                vertex in positions and positions[vertex] != point
            ):
                return None
            positions[vertex] = point
    ids = tuple(sorted(positions))
    index = {vertex: row for row, vertex in enumerate(ids)}
    points = np.asarray(
        tuple(tuple(float(value) for value in positions[vertex]) for vertex in ids),
        dtype=np.float64,
    )
    blocks = tuple(
        CellBlock(
            name,
            members[0].kind,
            np.asarray(
                tuple(
                    tuple(index[vertex] for vertex in chart.vertices) for chart in members
                ),
                dtype=np.int32,
            ),
            global_ids=np.asarray(
                tuple(chart.cell_id for chart in members), dtype=np.int64
            ),
        )
        for name, members in sorted(
            grouped.items(),
            key=lambda item: (
                len(reference_cell_topology(item[1][0].kind).vertices),
                item[0],
            ),
        )
    )
    mesh = CellMesh(points, blocks, vertex_global_ids=np.asarray(ids, dtype=np.int64))
    elements = {
        block.name: coordinate_lagrange_element(block.cell_kind, 1) for block in blocks
    }
    geometry = CellGeometrySpec(
        elements, {block.name: block.vertices for block in blocks}, points
    )
    variables = algebra.axes(dimension)
    for block in blocks:
        for chart, row in zip(
            grouped[block.name], np.asarray(block.vertices), strict=True
        ):
            if chart.matrix is None or chart.offset is None:
                raise RuntimeError("An affine reference chart lost its exact transform.")
            arguments = variables
            if chart.kind == "pyramid":
                u, v, w = variables
                collapse = algebra.add(algebra.constant(1, 3), algebra.scale(w, -1))
                arguments = (
                    algebra.add(
                        algebra.multiply(u, collapse), algebra.scale(w, Fraction(1, 2))
                    ),
                    algebra.add(
                        algebra.multiply(v, collapse), algebra.scale(w, Fraction(1, 2))
                    ),
                    w,
                )
            expected = tuple(
                algebra.add(
                    algebra.constant(offset, dimension),
                    algebra.sum_polynomials(
                        tuple(
                            algebra.scale(variable, weight)
                            for variable, weight in zip(
                                arguments, row_matrix, strict=True
                            )
                        )
                    ),
                )
                for row_matrix, offset in zip(chart.matrix, chart.offset, strict=True)
            )
            actual = algebra.coordinate_polynomials(elements[block.name], points[row])
            if actual != expected:
                return None
    return mesh, geometry


def _exact_chart_mesh(
    charts: list[_Chart], dimension: int
) -> tuple[CellMesh, CellGeometrySpec] | None:
    """RNE carrier plus the actual source-coordinate chart, never interpolation."""
    positions: dict[int, tuple[Fraction, ...]] = {}
    grouped: dict[str, list[_Chart]] = {}
    elements: dict[str, CellGeometryElement] = {}
    controls: list[np.ndarray] = []
    routes: dict[str, np.ndarray] = {}
    count = 0
    for chart in charts:
        element, values = chart.reference_element, chart.reference_controls
        if element is None or values is None:
            # Affine neighbors retain exact chart coefficients, not the
            # rounded coordinate carrier of the reconstructed reference mesh.
            if chart.matrix is None or chart.offset is None:
                raise RuntimeError(
                    "Reference chart has neither an affine nor an exact composed source."
                )
            identity = coordinate_lagrange_element(
                "triangle" if dimension == 2 else "tetrahedron", 1
            )
            nodal = coordinate_lagrange_element(chart.kind, 1)
            coefficients = tuple(
                tuple(
                    sum(
                        (
                            weight * Fraction(float(x))
                            for weight, x in zip(row, corner, strict=True)
                        ),
                        shift,
                    )
                    for row, shift in zip(chart.matrix, chart.offset, strict=True)
                )
                for corner in np.asarray(nodal.reference_nodes)
            )
            owner = (
                RationalComposedCellGeometryElement
                if chart.kind == "pyramid"
                else PolynomialComposedCellGeometryElement
            )
            element = owner(
                identity,
                nodal,
                np.asarray(
                    [[x.numerator for x in row] for row in coefficients], dtype=np.int64
                ),
                np.asarray(
                    [[x.denominator for x in row] for row in coefficients],
                    dtype=np.uint64,
                ),
            )
            values = np.asarray(identity.reference_nodes, dtype=np.float64)
        if chart.block in elements:
            if elements[chart.block].element_id != element.element_id:
                return None
        else:
            elements[chart.block] = element
        images = algebra.coordinate_corner_images(element, values)
        if images is None:
            return None
        for vertex, point in zip(chart.vertices, images, strict=True):
            if vertex in positions and positions[vertex] != point:
                return None
            positions[vertex] = point
        grouped.setdefault(chart.block, []).append(chart)
        route = np.arange(count, count + values.shape[0], dtype=np.int64)
        routes[chart.block] = (
            np.vstack((routes[chart.block], route))
            if chart.block in routes
            else route[None]
        )
        controls.append(values)
        count += values.shape[0]
    ids = tuple(sorted(positions))
    index = {vertex: row for row, vertex in enumerate(ids)}
    points = np.asarray(
        tuple(tuple(float(value) for value in positions[vertex]) for vertex in ids),
        dtype=np.float64,
    )
    blocks = tuple(
        CellBlock(
            name,
            members[0].kind,
            np.asarray(
                tuple(
                    tuple(index[vertex] for vertex in chart.vertices) for chart in members
                ),
                dtype=np.int32,
            ),
            global_ids=np.asarray(
                tuple(chart.cell_id for chart in members), dtype=np.int64
            ),
        )
        for name, members in sorted(
            grouped.items(),
            key=lambda item: (
                len(reference_cell_topology(item[1][0].kind).vertices),
                item[0],
            ),
        )
    )
    mesh = CellMesh(points, blocks, vertex_global_ids=np.asarray(ids, dtype=np.int64))
    return mesh, CellGeometrySpec(elements, routes, np.concatenate(controls))


def _root_domain(
    domain: MappedReferenceDomain, root: _Root
) -> tuple[PiecewiseLinearDomain, tuple[int, ...]]:
    topology = reference_cell_topology(root.kind)
    facets = []
    owners = []
    for index, face in enumerate(topology.entities[domain.ambient_dimension - 1]):
        pieces = (
            (face,)
            if len(face) <= 3
            else ((face[0], face[1], face[2]), (face[0], face[2], face[3]))
        )
        facets.extend(pieces)
        owners.extend(index for _ in pieces)
    declared = PiecewiseLinearDomain(
        np.asarray(topology.vertices, dtype=np.float64),
        np.asarray(facets, dtype=np.int64),
        np.tile(np.asarray(((0, -1),), dtype=np.int64), (len(facets), 1)),
        ("root",),
        source_id=f"{domain.source_id}:root:{root.cell_id}",
    )
    return declared, tuple(owners)


def _facet_identities(
    mesh: CellMesh,
) -> tuple[dict[int, tuple[int, ...]], dict[tuple[int, ...], int]]:
    facets = _mesh_facets(mesh)
    entities = _facet_entity_ids(mesh, facets.rows)
    vertices = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    identities = {
        int(entity): tuple(sorted(int(value) for value in vertices[row[row >= 0]]))
        for entity, row in zip(entities, facets.rows, strict=True)
    }
    return identities, {key: entity for entity, key in identities.items()}


def _compose_fragment_relations(
    relations: dict[_CoverageOverlapKey, Fraction],
    chart_mesh: CellMesh,
    chart_coverage: DomainCoverageCertificate,
    face_owners: tuple[int, ...],
    root: _Root,
    domain: MappedReferenceDomain,
    source_coverage: DomainCoverageCertificate,
    source_facets: dict[tuple[int, ...], int],
    target_facets: dict[tuple[int, ...], int],
    covering: set[int],
) -> None:
    chart_facets, _ = _facet_identities(chart_mesh)
    root_faces = reference_cell_topology(root.kind).entities[domain.ambient_dimension - 1]
    source_rows: dict[int, list[tuple[int, tuple[int, ...], float]]] = {}
    for facet, source, axes, _, upper, _, _ in source_coverage.facet_source_overlaps:
        source_rows.setdefault(facet, []).append((source, axes, upper))
    for chart_facet, piece, _, _, _, _, _ in chart_coverage.facet_source_overlaps:
        target = target_facets[chart_facets[chart_facet]]
        if target not in covering:
            continue
        source_face = tuple(
            sorted(root.vertices[vertex] for vertex in root_faces[face_owners[piece]])
        )
        for source, axes, upper in source_rows.get(source_facets[source_face], ()):
            total = abs(
                signed_measure(
                    project(
                        rational_points(
                            np.asarray(domain.reference_domain.vertices)[
                                domain.reference_domain.facets[source]
                            ]
                        ),
                        axes,
                    )
                )
            )
            key: _CoverageOverlapKey = (target, source, axes, "candidate", "reference")
            relations[key] = min(total, relations.get(key, Fraction(0)) + Fraction(upper))


def _add_measure(
    totals: list[Fraction | None], region: int, measure: Fraction | None
) -> None:
    current = totals[region]
    totals[region] = None if current is None or measure is None else current + measure


def certify_mapped_reference_coverage(
    state: _EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: MappedReferenceDomain,
    regions: np.ndarray,
    limits: MeshCertificateLimits,
    binding: MeshCertificateBinding,
    embedding: GlobalEmbeddingCertificate,
) -> DomainCoverageCertificate:
    requested = domain.exact_region_measures()
    _, source_embedding, reference_coverage = certify_mapped_reference_source(
        domain, limits
    )
    premises = [source_embedding.certificate_id, reference_coverage.certificate_id]
    source_periodic = domain.reference_mesh.periodic_topology
    target_periodic = mesh.periodic_topology
    if (source_periodic is None) != (target_periodic is None):
        state.add("mapped_domain_periodic_source_identity", "violated", "mesh")
    elif source_periodic is not None and target_periodic is not None:
        from ..discretization._periodic_cell import PeriodicCell
        from ..discretization._periodic_topology import PeriodicIsometryGroup

        source_cell, target_cell = source_periodic.cell, target_periodic.cell
        same_group = (
            isinstance(source_cell, PeriodicCell)
            and isinstance(target_cell, PeriodicCell)
            and source_cell.cell_id == target_cell.cell_id
        ) or (
            isinstance(source_cell, PeriodicIsometryGroup)
            and isinstance(target_cell, PeriodicIsometryGroup)
            and source_cell.group_id == target_cell.group_id
        )
        if not same_group:
            state.add("mapped_domain_periodic_source_identity", "violated", "mesh")
        for certificate in (source_embedding, embedding):
            if "periodic_mapped_trace_equivariance" not in certificate.evaluated_checks:
                state.add("mapped_domain_periodic_trace_premise", "unresolved", "mesh")
        # The complete source reference partition below covers every authored
        # facet, including seam gauges. Physical walls/material traces and
        # quotient-glued seams are not silently omitted from continuous bounds.
    for check, certificate in (
        ("mapped_domain_source_embedding_premise", source_embedding),
        ("mapped_domain_reference_coverage_premise", reference_coverage),
    ):
        if certificate.status != "certified":
            state.add(check, certificate.status, "mesh")
    roots = _roots(domain)
    charts = _target_charts(state, mesh, geometry, regions, domain, roots)
    achieved: list[Fraction | None] = [Fraction(0) for _ in domain.region_ids]
    relations: dict[_CoverageOverlapKey, Fraction] = {}
    _, source_facets = _facet_identities(domain.reference_mesh)
    _, target_facets = _facet_identities(mesh)
    facets = _mesh_facets(mesh)
    entities = _facet_entity_ids(mesh, facets.rows)
    covering = set(int(value) for value in entities[facets.boundary])
    for group in np.unique(facets.group):
        rows = np.flatnonzero(facets.group == group)
        if (
            rows.size == 2
            and regions[facets.cells[rows[0]]] != regions[facets.cells[rows[1]]]
        ):
            covering.add(int(entities[rows[0]]))
    if charts is None:
        reported, _ = _mapped_coverage_measures(state, mesh, geometry, regions)
        actual = (
            *reported,
            *(Fraction(0) for _ in range(len(domain.region_ids) - len(reported))),
        )
        return _coverage_result(
            binding,
            embedding,
            domain,
            state,
            requested,
            actual,
            0,
            relations,
            tuple(premises),
            expression_work=_active_expression_work(),
        )
    target_elements, target_routes, _ = geometry.resolve(mesh)
    target_values = geometry.source_coordinates()
    direct: dict[int, Fraction | None] = {cell_id: Fraction(0) for cell_id in roots}
    record = geometry.restriction_source
    parents = (
        {block.name: np.asarray(block.global_ids) for block in mesh.blocks}
        if record is None
        else record.block_parent_cell_ids
    )
    for block, element, route in zip(
        mesh.blocks, target_elements, target_routes, strict=True
    ):
        local_values = tuple(
            tuple(target_values[index] for index in row)
            for row in np.asarray(route, dtype=np.int64)
        )
        for parent, local in zip(
            np.asarray(parents[block.name]), local_values, strict=True
        ):
            polynomial = algebra.coordinate_polynomials(element, local)
            current = direct[int(parent)]
            if polynomial is None:
                direct[int(parent)] = None
            elif current is not None:
                jacobian = tuple(
                    tuple(
                        algebra.derivative(value, axis)
                        for axis in range(domain.ambient_dimension)
                    )
                    for value in polynomial
                )
                direct[int(parent)] = current + integrate(
                    algebra.determinant(jacobian),
                    map_domain(block.cell_kind),
                    domain.ambient_dimension,
                )
    complete = (
        source_embedding.status
        == reference_coverage.status
        == embedding.status
        == "certified"
    )
    for cell_id, root in sorted(roots.items()):
        if not charts[cell_id]:
            state.add("mapped_domain_missing_root", "violated", "source_cell", (cell_id,))
            _add_measure(achieved, root.region, direct[cell_id])
            complete = False
            continue
        chart_view = _chart_mesh(charts[cell_id], domain.ambient_dimension)
        if chart_view is None:
            state.add(
                "mapped_domain_chart_representation",
                "unresolved",
                "cell",
                tuple(chart.cell_id for chart in charts[cell_id]),
            )
            _add_measure(achieved, root.region, direct[cell_id])
            complete = False
            continue
        chart_mesh, chart_geometry = chart_view
        chart_validity = certify_cell_geometry_validity(chart_geometry, mesh=chart_mesh)
        chart_embedding = certify_global_embedding(
            chart_mesh, chart_geometry, chart_validity, limits=limits
        )
        declared, face_owners = _root_domain(domain, root)
        chart_coverage = certify_domain_coverage(
            chart_mesh,
            chart_geometry,
            declared,
            np.zeros(
                sum(block.cell_count for block in chart_mesh.blocks), dtype=np.int64
            ),
            embedding=chart_embedding,
            limits=limits,
        )
        premises.append(chart_coverage.certificate_id)
        if chart_coverage.status != "certified":
            state.add(
                "mapped_domain_root_coverage_premise",
                chart_coverage.status,
                "cell",
                tuple(chart.cell_id for chart in charts[cell_id]),
            )
            _add_measure(achieved, root.region, direct[cell_id])
            complete = False
            continue
        _compose_fragment_relations(
            relations,
            chart_mesh,
            chart_coverage,
            face_owners,
            root,
            domain,
            reference_coverage,
            source_facets,
            target_facets,
            covering,
        )
        actual = direct[cell_id]
        if actual is None:
            # Exact coefficient/root identity plus the independently certified
            # reference partition permits integration over the original root,
            # including rational source restrictions whose child polynomial is
            # absent. This is the source partition/change-of-variables theorem.
            jacobian = tuple(
                tuple(
                    algebra.derivative(value, axis)
                    for axis in range(domain.ambient_dimension)
                )
                for value in root.polynomial
            )
            actual = integrate(
                algebra.determinant(jacobian),
                map_domain(root.kind),
                domain.ambient_dimension,
            )
        _add_measure(achieved, root.region, actual)
    return _coverage_result(
        binding,
        embedding,
        domain,
        state,
        requested,
        tuple(achieved),
        domain.facets.shape[0] if complete else 0,
        relations,
        tuple(premises),
        expression_work=_active_expression_work(),
    )
