#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Source-root strata of mixed mapped coordinate restrictions."""

from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
from typing import Protocol

import numpy as np

from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.geometry._embedding_cells import (
    reference_constraints,
    vanishing_reference_constraints,
)
from phydrax.geometry._embedding_state import MeshCertificateLimits
from phydrax.geometry._planar_primitives import reserve_fraction_work
from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory

from .._strict import StrictModule
from ..discretization import _coordinate_enclosure as algebra
from ..discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    CellGeometrySpec,
    equivalent_coordinate_root,
)
from ..discretization._cell_geometry_validity import (
    _ConstructedTensorGrid,
    cell_geometry_id,
)
from ..discretization._cell_mesh import CellMesh
from ..discretization._reference_cell import reference_cell_topology
from ..discretization._reference_cell_types import CellGeometryElement
from ..geometry._mapped_coverage import map_domain, nonnegative, SubdivisionLedger
from ..geometry._mapped_reference_coverage import source_roots
from ..geometry._mapped_reference_domain import MappedReferenceDomain
from ..typing import checked

# Module references defer symbol resolution across mutually dependent execution owners.
from . import (
    _association_state as association_state_owner,
    _plc_mapped_support as plc_mapped_support_owner,
)
from ._association_state import GeometryAssociation


def _chart_orientation(
    source: tuple[tuple[Fraction, ...], ...],
    target: tuple[tuple[Fraction, ...], ...],
    degree: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
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

    reserve_fraction_work(
        (*source, *target),
        18 if degree == 2 else 6,
        1,
        20 if degree == 2 else 4,
        budget=budget,
    )
    product = sum(
        (a * b for a, b in zip(direction(source), direction(target), strict=True)),
        Fraction(0),
    )
    if product == 0:
        raise ValueError(
            f"MappedReferenceAssociation.orientation must have a nonzero direction product; got {product!r}"
        )
    return 1 if product > 0 else -1


@checked
def mapped_reference_associations(
    domain: MappedReferenceDomain,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
    *,
    maximum_support_queries: int,
    budget: CoordinateEnclosureBudget,
) -> tuple[GeometryAssociation, ...]:
    """Classify whole chart images against their mixed source-root strata."""
    limits = MeshCertificateLimits(maximum_scratch_bytes=budget.maximum_memory_bytes)
    ledger = budget
    work_start = ledger.work_units
    try:
        with (
            ledger.measurement(),
            ledger.activate(),
            ledger.bound_stage(
                maximum_support_queries,
                limits.maximum_scratch_bytes,
                starting_work_units=work_start,
            ),
            ledger.temporary_scope(),
        ):
            return _mapped_reference_associations(
                domain,
                mesh,
                geometry,
                maximum_support_queries,
                limits,
                budget=budget,
            )
    except CoordinateEnclosureResourceError as error:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            f"MappedReferenceAssociation.resource_budget must admit the source-stratum proof; got {error}",
            stage="mapped-source-association",
        ) from error


class _MappedRoot(Protocol):
    @property
    def cell_id(self) -> int: ...
    @property
    def kind(self) -> str: ...
    @property
    def vertices(self) -> tuple[int, ...]: ...
    @property
    def element(self) -> CellGeometryElement: ...
    @property
    def controls(self) -> algebra.CoordinateSourceBank: ...


class _RootChart(StrictModule):
    coordinates: tuple[algebra.Polynomial, ...]
    points: tuple[tuple[Fraction, ...], ...]
    # Root reference vertices on each ``reference_constraints`` facet.
    facet_vertices: tuple[frozenset[int], ...]
    corners: tuple[tuple[Fraction, ...], ...]
    cell: plc_mapped_support_owner.CellMap
    # The chart is multi-affine on a box cell: its restriction to every edge
    # and face is determined by ``corners``.
    multi_affine: bool


class _RootStratum(StrictModule):
    degree: int
    row: int


class _ChartOrientation(StrictModule):
    value: int


type _ReferenceTemplates = dict[
    str, tuple[tuple[tuple[Fraction, ...], ...], tuple[frozenset[int], ...]]
]
type _CoordinateChains = dict[
    tuple[tuple[str, str], tuple[str, str]], tuple[algebra.Polynomial, ...]
]


def _prepare_root_chart(
    root: _MappedRoot,
    element: CellGeometryElement,
    controls: algebra.CoordinateSourceBank,
    parent_vertices: np.ndarray,
    cell: np.ndarray,
    cell_kind: str,
    root_identity: tuple[str, str],
    element_identity: tuple[str, str],
    chains: _CoordinateChains,
    reference_templates: _ReferenceTemplates,
    used: SubdivisionLedger,
    limits: MeshCertificateLimits,
    maximum_support_queries: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> _RootChart:
    """Certify root controls and whole-chart containment, reusing admitted polynomial chains."""
    ledger = budget
    topology = reference_cell_topology(cell_kind)
    ledger.reserve(len(controls) + len(root.vertices))
    equivalent = (
        None
        if tuple(int(value) for value in parent_vertices if value >= 0) != root.vertices
        else equivalent_coordinate_root(
            element, controls, root.element, root.controls, budget=budget
        )
    )
    if equivalent is None:
        raise ValueError(
            "MappedReferenceAssociation.restriction must prove the root law and retain ordered source vertex identities; got no equivalent root expression"
        )
    chain_key = (element_identity, root_identity)
    coordinates = chains.get(chain_key)
    if coordinates is None:
        _, expressions = algebra.coordinate_reference_chain(
            element,
            ancestor=equivalent,
            budget=budget,
        )
        if any(isinstance(value, algebra.RationalPolynomial) for value in expressions):
            raise ValueError(
                f"MappedReferenceAssociation.reference_chain must contain polynomial charts; got expression types {tuple(type(value).__name__ for value in expressions)!r}"
            )
        coordinates = tuple(
            value
            for value in expressions
            if not isinstance(value, algebra.RationalPolynomial)
        )
        chains[chain_key] = coordinates
        ledger.retain_basis((chain_key, coordinates))
    constraints = reference_constraints(coordinates, root.kind, 3, budget=budget)
    for constraint in constraints:
        if not nonnegative(
            constraint,
            map_domain(cell_kind),
            3,
            limits.maximum_bernstein_nodes,
            limits.maximum_subdivision_depth,
            min(limits.maximum_subdivision_pieces, maximum_support_queries),
            used,
            budget=budget,
        ):
            if used.pieces > min(
                limits.maximum_subdivision_pieces, maximum_support_queries
            ):
                raise MeshingFailure(
                    MeshingFailureCategory.RESOURCE_EXHAUSTED,
                    f"MappedReferenceAssociation.subdivision_pieces must fit the containment budget; got {used.pieces}, limit {min(limits.maximum_subdivision_pieces, maximum_support_queries)}",
                    stage="mapped-source-association",
                )
            raise ValueError(
                f"MappedReferenceAssociation.containment must certify each root constraint; got constraint {constraint!r}, target kind {cell_kind!r}, subdivision pieces {used.pieces}"
            )
    template = reference_templates.get(root.kind)
    if template is None:
        root_expressions = reference_constraints(
            algebra.axes(3, budget=budget),
            root.kind,
            3,
            budget=budget,
        )
        if any(
            isinstance(value, algebra.RationalPolynomial) for value in root_expressions
        ):
            raise RuntimeError(
                f"MappedReferenceAssociation.facet_equations must be polynomials; got expression types {tuple(type(value).__name__ for value in root_expressions)!r}"
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
        facet_vertices = tuple(
            frozenset(
                vertex
                for vertex, point in enumerate(root_points)
                if algebra.evaluate(constraint, point, budget=budget) == 0
            )
            for constraint in root_constraints
        )
        template = (root_points, facet_vertices)
        reference_templates[root.kind] = template
        ledger.retain_basis((root.kind, template))
    root_points, facet_vertices = template
    corners = tuple(
        tuple(
            algebra.require_rational_reference_scalar(
                algebra.evaluate(
                    value,
                    tuple(Fraction(float(x)) for x in position),
                    budget=budget,
                )
            )
            for value in coordinates
        )
        for position in topology.vertices
    )
    chart_cell = plc_mapped_support_owner.CellMap(
        cell_kind,
        tuple(cell.tolist()),
        coordinates,
        element,
        controls,
        root.cell_id,
        root.vertices,
    )
    # Box charts with at most linear dependence on each reference axis.
    ledger.reserve(sum(len(value) for value in coordinates))
    multi_affine = cell_kind == "hexahedron" and all(
        exponent <= 1 for value in coordinates for index in value for exponent in index
    )
    return _RootChart(
        coordinates, root_points, facet_vertices, corners, chart_cell, multi_affine
    )


def _chart_vanishing_facets(
    local_charts: tuple[
        tuple[tuple[algebra.Expression, ...], algebra.BernsteinDomain], ...
    ],
    root: _MappedRoot,
    degree: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> frozenset[int]:
    """Return the root facets whose equations vanish on every entity chart."""
    fixed: frozenset[int] | None = None
    for chart, _ in local_charts:
        if any(isinstance(value, algebra.RationalPolynomial) for value in chart):
            raise ValueError(
                f"MappedReferenceAssociation.entity_chart must contain polynomials; got expression types {tuple(type(value).__name__ for value in chart)!r}"
            )
        vanishing = vanishing_reference_constraints(
            chart, root.kind, degree, budget=budget
        )
        fixed = vanishing if fixed is None else fixed & vanishing
    if fixed is None:
        raise RuntimeError(
            f"MappedReferenceAssociation.entity_charts must provide facet equations; got chart count {len(local_charts)}"
        )
    return fixed


def _corner_vanishing_facets(
    corners: tuple[tuple[Fraction, ...], ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> frozenset[int]:
    """Return the box-root facets vanishing on a multi-affine entity restriction.

    A multi-affine chart restricted to a cell edge or face is multi-affine in
    that entity's free parameters, so ``x_i`` (or ``1 - x_i``) vanishes on the
    whole entity exactly when it vanishes at every entity corner.
    """
    ambient = len(corners[0])
    budget.reserve(2 * ambient * len(corners))
    return frozenset(
        (
            *(
                axis
                for axis in range(ambient)
                if all(corner[axis] == 0 for corner in corners)
            ),
            *(
                ambient + axis
                for axis in range(ambient)
                if all(corner[axis] == 1 for corner in corners)
            ),
        )
    )


def _identify_root_stratum(
    fixed: frozenset[int],
    root: _MappedRoot,
    facet_vertices: tuple[frozenset[int], ...],
    root_lookup: tuple[dict[tuple[int, ...], int], ...],
    degree: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> _RootStratum:
    """Return the root stratum on every facet that vanishes on the entity."""
    # The root's facet/vertex incidence is the exact zero set of its facet
    # equations at its reference vertices, earned once per root kind.
    budget.reserve(len(fixed) * len(root.vertices))
    source_key = tuple(
        sorted(
            root.vertices[vertex]
            for vertex in range(len(root.vertices))
            if all(vertex in facet_vertices[facet] for facet in fixed)
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
            f"MappedReferenceAssociation.source_stratum must identify an entity of at least target dimension; got source {source!r}, target dimension {degree}, vertex key {source_key!r}"
        )
    source_degree, source_row = source
    return _RootStratum(source_degree, source_row)


def _resolve_chart_orientation(
    source_degree: int,
    source_row: int,
    degree: int,
    root_entities: tuple[np.ndarray, ...],
    root_vertices: np.ndarray,
    root_points: tuple[tuple[Fraction, ...], ...],
    root: _MappedRoot,
    local_charts: tuple[
        tuple[tuple[algebra.Expression, ...], algebra.BernsteinDomain], ...
    ],
    *,
    budget: CoordinateEnclosureBudget,
) -> _ChartOrientation:
    """Return a common orientation across entity charts using rational arithmetic."""
    orientation = 0
    if degree == source_degree and degree in (1, 2):
        source_rows = root_entities[source_degree][source_row]
        source_points = tuple(
            root_points[root.vertices.index(int(identifier))]
            for identifier in root_vertices[source_rows[source_rows >= 0]]
        )
        chart_orientations: set[int] = set()
        for chart, _ in local_charts:
            zero = (Fraction(0),) * degree
            tangents = tuple(
                tuple(
                    algebra.require_rational_reference_scalar(
                        algebra.expression_evaluate(
                            algebra.expression_derivative(value, axis, budget=budget),
                            zero,
                            budget=budget,
                        )
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
                    budget=budget,
                )
            )
        if len(chart_orientations) != 1:
            raise ValueError(
                f"MappedReferenceAssociation.orientation must agree across entity charts; got directions {chart_orientations!r}"
            )
        orientation = next(iter(chart_orientations))
    return _ChartOrientation(orientation)


class _RestrictionAdmission(StrictModule):
    record: CellGeometryRestrictionSource


class _MappedAssociationResult(StrictModule):
    associations: tuple[association_state_owner.GeometryAssociation, ...]


def _admit_mapped_restriction(
    domain: MappedReferenceDomain,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    maximum_support_queries: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> _RestrictionAdmission:
    """Validate root identity and admit all source-stratum queries."""
    record = geometry.restriction_source
    if (
        record is None
        or record.source_geometry_id != cell_geometry_id(domain.source_geometry)
        or record.source_topology_id != domain.reference_mesh.topology_id
    ):
        raise ValueError(
            f"MappedReferenceAssociation.restriction_source must bind the domain root geometry and topology; got {record!r}, expected topology {domain.reference_mesh.topology_id!r}"
        )
    supported = ("tetrahedron", "prism", "hexahedron")
    if any(
        block.cell_kind not in supported
        for block in (*mesh.blocks, *domain.reference_mesh.blocks)
    ):
        raise ValueError(
            f"MappedReferenceAssociation.cell_kinds must be tetrahedron, prism, or hexahedron; got {tuple(block.cell_kind for block in (*mesh.blocks, *domain.reference_mesh.blocks))!r}"
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
            f"MappedReferenceAssociation.support_queries must fit the support budget; got required {required}, limit {maximum_support_queries}",
            stage="mapped-source-association",
        )

    if (
        required > budget.ticket.remaining_work_units
        or required > budget.ticket.remaining_geometry_queries
    ):
        budget.ticket.refuse(required, queries_upper=required)
    budget.ticket.consume(required, geometry_queries=required)
    return _RestrictionAdmission(record)


def _assemble_mapped_associations(
    domain: MappedReferenceDomain,
    mesh: CellMesh,
    root_mesh: CellMesh,
    choices: Iterable[list[set[tuple[int, int, int]]]],
    *,
    budget: CoordinateEnclosureBudget,
) -> _MappedAssociationResult:
    """Publish the unique lowest-dimensional root claims in target entity order."""
    ledger = budget
    output = []
    for degree, rows in enumerate(choices):
        ledger.reserve(0, 3 * 128 + 17 * len(rows))
        dimensions = np.empty((len(rows),), dtype=np.int64)
        identifiers = np.empty((len(rows),), dtype=np.int64)
        orientations = np.empty((len(rows),), dtype=np.int8)
        for row, candidates in enumerate(rows):
            if not candidates:
                raise ValueError(
                    f"MappedReferenceAssociation.source_incidence must provide a root claim for every target row; got dimension {degree}, row {row}, candidate count {len(candidates)}"
                )
            smallest = min(dimension for dimension, _, _ in candidates)
            selected = {
                (dimension, index, orientation)
                for dimension, index, orientation in candidates
                if dimension == smallest
            }
            if len(selected) != 1:
                raise ValueError(
                    f"MappedReferenceAssociation.source_stratum must select one lowest-dimensional root claim; got dimension {degree}, row {row}, claims {selected!r}"
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
            association_state_owner.GeometryAssociation(
                association_state_owner.GeometryAssociationKind.MAPPED_REFERENCE,
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
    return _MappedAssociationResult(tuple(output))


class _MappedEntityClaim(StrictModule):
    candidate: tuple[int, int, int]


def _classify_mapped_entity(
    degree: int,
    local: tuple[int, ...],
    target_row: int,
    prepared: _RootChart,
    root: _MappedRoot,
    root_entities: tuple[np.ndarray, ...],
    target_entities: tuple[np.ndarray, ...],
    root_vertices: np.ndarray,
    root_lookup: tuple[dict[tuple[int, ...], int], ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> _MappedEntityClaim:
    """Classify the whole entity chart and certify a consistent root orientation."""
    corners, coordinates, chart_cell = (
        prepared.corners,
        prepared.coordinates,
        prepared.cell,
    )
    root_points, facet_vertices = prepared.points, prepared.facet_vertices
    # A multi-affine box chart decides its box-root strata on exact entity
    # corners; only entities on a root stratum of their own dimension need
    # charts, for their orientation.
    stratum: _RootStratum | None = None
    if degree in (1, 2) and prepared.multi_affine and root.kind == "hexahedron":
        stratum = _identify_root_stratum(
            _corner_vanishing_facets(
                tuple(corners[index] for index in local), budget=budget
            ),
            root,
            facet_vertices,
            root_lookup,
            degree,
            budget=budget,
        )
        if stratum.degree != degree:
            return _MappedEntityClaim((stratum.degree, stratum.row, 0))
    if degree == 0:
        local_charts = (
            (
                tuple(
                    algebra.constant(value, 0, budget=budget)
                    for value in corners[local[0]]
                ),
                "box",
            ),
        )
    elif degree == 3:
        local_charts = ((coordinates, map_domain(prepared.cell.kind)),)
    else:
        # Incidence vertex order determines orientation; full
        # polynomial restrictions own edge and face support.
        local_charts = plc_mapped_support_owner.mapped_reference_charts(
            (chart_cell,),
            target_entities[degree][target_row],
            degree,
            budget=budget,
        )
    if not local_charts:
        raise ValueError(
            f"MappedReferenceAssociation.entity_charts must cover the target entity; got dimension {degree}, row {target_row}, chart count {len(local_charts)}"
        )
    if stratum is None:
        stratum = _identify_root_stratum(
            _chart_vanishing_facets(local_charts, root, degree, budget=budget),
            root,
            facet_vertices,
            root_lookup,
            degree,
            budget=budget,
        )
    source_degree, source_row = stratum.degree, stratum.row
    orientation = _resolve_chart_orientation(
        source_degree,
        source_row,
        degree,
        root_entities,
        root_vertices,
        root_points,
        root,
        local_charts,
        budget=budget,
    ).value
    candidate = (source_degree, source_row, orientation)
    return _MappedEntityClaim(candidate)


def _mapped_reference_associations(
    domain: MappedReferenceDomain,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    maximum_support_queries: int,
    limits: MeshCertificateLimits,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[association_state_owner.GeometryAssociation, ...]:
    ledger = budget

    record = _admit_mapped_restriction(
        domain, mesh, geometry, maximum_support_queries, budget=budget
    ).record
    root_mesh = domain.reference_mesh
    root_entities = tuple(
        plc_mapped_support_owner.plc_mapped_vertices(root_mesh, degree)
        for degree in range(4)
    )
    target_entities = tuple(
        plc_mapped_support_owner.plc_mapped_vertices(mesh, degree) for degree in range(4)
    )
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
    local_roots = source_roots(domain, budget=budget)
    elements, routes, _ = geometry.resolve(mesh, budget=budget)
    values = algebra.prepared_coordinate_source_bank(geometry, budget=budget)
    used = SubdivisionLedger()
    ledger.retain_basis((*root_lookup, *target_lookup))
    ledger.reserve(
        sum(len(rows) for rows in choices),
        128
        + 128 * sum(len(rows) for rows in choices)
        + sum(array.nbytes + 128 for array in (*root_entities, *target_entities)),
    )
    root_identities: dict[int, tuple[str, str]] = {}
    for root in local_roots.values():
        ledger.retain_basis((root.vertices, root.controls, root.polynomial))
        ledger.reserve(1, 256)
        root_identities[root.cell_id] = algebra.element_identity(root.element, budget=budget)
    ledger.retain_basis(tuple(root_identities.items()))
    chains: dict[
        tuple[tuple[str, str], tuple[str, str]], tuple[algebra.Polynomial, ...]
    ] = {}
    reference_templates: _ReferenceTemplates = {}
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        topology = reference_cell_topology(block.cell_kind)
        element_identity = algebra.element_identity(element, budget=budget)
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
                f"MappedReferenceAssociation.incidence must use rank-two cell, coefficient, and parent-vertex routes with vector parent IDs; got shapes {cells.shape!r}, {route_rows.shape!r}, {parents.shape!r}, {parent_vertices.shape!r}"
            )
        for index in range(block.cell_count):
            with ledger.temporary_scope():
                cell = cells[index]
                controls = tuple(values[int(dof)] for dof in route_rows[index])
                parent = int(parents[index])
                root = local_roots.get(parent)
                if root is None:
                    raise ValueError(
                        f"MappedReferenceAssociation.parent_cell_id must name a declared source root; got {parent}"
                    )
                prepared_chart = _prepare_root_chart(
                    root,
                    element,
                    controls,
                    parent_vertices[index],
                    cell,
                    block.cell_kind,
                    root_identities[parent],
                    element_identity,
                    chains,
                    reference_templates,
                    used,
                    limits,
                    maximum_support_queries,
                    budget=budget,
                )
                for degree in range(4):
                    for local in topology.entities[degree]:
                        key = tuple(sorted(target_vertices[cell[list(local)]].tolist()))
                        target_row = target_lookup[degree][key]
                        candidate = _classify_mapped_entity(
                            degree,
                            local,
                            target_row,
                            prepared_chart,
                            root,
                            root_entities,
                            target_entities,
                            root_vertices,
                            root_lookup,
                            budget=budget,
                        ).candidate
                        if candidate not in choices[degree][target_row]:
                            ledger.retain_basis(candidate)
                            choices[degree][target_row].add(candidate)
    return _assemble_mapped_associations(
        domain, mesh, root_mesh, choices, budget=budget
    ).associations


def _tensor_reference_associations(
    domain: MappedReferenceDomain,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    construction: _ConstructedTensorGrid,
    *,
    maximum_support_queries: int,
    budget: CoordinateEnclosureBudget,
) -> tuple[GeometryAssociation, ...]:
    with (
        budget.measurement(), budget.activate(), budget.temporary_scope(),
        budget.bound_stage(maximum_support_queries, budget.maximum_memory_bytes),
    ):
        return _classify_tensor_reference(
            domain, mesh, geometry, construction,
            maximum_support_queries=maximum_support_queries, budget=budget,
        )


def _classify_tensor_reference(
    domain: MappedReferenceDomain,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    construction: _ConstructedTensorGrid,
    *,
    maximum_support_queries: int,
    budget: CoordinateEnclosureBudget,
) -> tuple[GeometryAssociation, ...]:
    """Consume exact tensor inputs, not a retained association verdict."""
    construction.consume(
        domain.reference_mesh, domain.source_geometry, mesh, geometry,
        construction.theorem, budget,
    )
    theorem = construction.theorem
    counts = np.asarray(theorem.counts, dtype=np.int64)
    order = np.asarray(theorem.order, dtype=np.int64)
    parents = np.asarray(construction.parents, dtype=np.int64)
    cells = sum(block.cell_count for block in mesh.blocks)
    budget.reserve(cells, 4096 + 128 * cells)
    if (
        counts.ndim != 2 or counts.shape[1] != 3 or np.any(counts <= 0)
        or order.shape != (cells,) or parents.shape != (cells,)
        or not np.array_equal(np.sort(order), np.arange(cells))
    ):
        raise ValueError("MappedReferenceAssociation.tensor_order must cover every child exactly once")
    offsets = np.concatenate(([0], np.cumsum(np.prod(counts, axis=1))))
    if offsets[-1] != cells:
        raise ValueError("MappedReferenceAssociation.tensor_coverage must cover all roots")
    expected_parents = np.searchsorted(offsets[1:], order, side="right")
    if not np.array_equal(parents, expected_parents):
        raise ValueError("MappedReferenceAssociation.tensor_parent must agree with C-order")
    record = geometry.restriction_source
    if (
        record is None
        or record.source_geometry_id != cell_geometry_id(domain.source_geometry)
        or record.source_topology_id != domain.reference_mesh.topology_id
    ):
        raise ValueError("MappedReferenceAssociation.restriction_source must bind current roots")
    root_mesh = domain.reference_mesh
    roots = source_roots(domain, budget=budget)
    root_ids = np.concatenate([np.asarray(block.global_ids) for block in root_mesh.blocks])
    if len(root_ids) != len(counts):
        raise ValueError("MappedReferenceAssociation.tensor_coverage must cover every source root")
    actual_parents = np.concatenate([
        np.asarray(record.block_parent_cell_ids[block.name]) for block in mesh.blocks
    ])
    if not np.array_equal(actual_parents, root_ids[parents]):
        raise ValueError("MappedReferenceAssociation.tensor_parent must bind current target parents")
    root_entities = tuple(
        plc_mapped_support_owner.plc_mapped_vertices(root_mesh, degree) for degree in range(4)
    )
    target_entities = tuple(
        plc_mapped_support_owner.plc_mapped_vertices(mesh, degree) for degree in range(4)
    )
    root_vertices = np.asarray(root_mesh.vertex_global_ids)
    target_vertices = np.asarray(mesh.vertex_global_ids)
    row_count = sum(len(rows) for rows in target_entities)
    # A fixed bound accounts for the dictionaries and candidate sets without
    # recursively walking their integer keys as a polynomial basis.
    budget.reserve(row_count, 4096 + 1024 * max(len(rows) for rows in target_entities))
    root_lookup = tuple({
        tuple(sorted(root_vertices[row[row >= 0]].tolist())): index
        for index, row in enumerate(rows)
    } for rows in root_entities)
    topology = reference_cell_topology("hexahedron")
    vertices = np.asarray(topology.vertices, dtype=np.int64)

    query_count = 0
    orientations: dict[tuple[int, tuple[int, ...], tuple[int, ...]], int] = {}

    def candidates() -> Iterable[list[set[tuple[int, int, int]]]]:
        nonlocal query_count
        for degree, local_entities in enumerate(topology.entities):
            rows = target_entities[degree]
            target_lookup = {
                tuple(sorted(target_vertices[row[row >= 0]].tolist())): index
                for index, row in enumerate(rows)
            }
            choices: list[set[tuple[int, int, int]]] = [set() for _ in rows]
            seen: set[tuple[int, int]] = set()
            cursor = 0
            for block in mesh.blocks:
                if block.cell_kind != "hexahedron":
                    raise ValueError("MappedReferenceAssociation.tensor_cells must be hexahedra")
                parent_vertices = np.asarray(record.block_parent_vertex_ids[block.name])
                for local_row, cell in enumerate(np.asarray(block.vertices)):
                    parent = int(parents[cursor])
                    root = roots[int(root_ids[parent])]
                    if tuple(int(value) for value in parent_vertices[local_row] if value >= 0) != root.vertices:
                        raise ValueError("MappedReferenceAssociation.tensor_parent must retain ordered root vertices")
                    position = np.asarray(np.unravel_index(
                        int(order[cursor] - offsets[parent]), tuple(counts[parent]),
                    ))
                    corners = position + vertices
                    for local in local_entities:
                        target_row = target_lookup[
                            tuple(sorted(target_vertices[cell[list(local)]].tolist()))
                        ]
                        claim = (parent, target_row)
                        if claim in seen:
                            continue
                        seen.add(claim)
                        query_count += 1
                        if query_count > maximum_support_queries:
                            raise MeshingFailure(
                                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                                "MappedReferenceAssociation.support_queries must admit unique tensor claims",
                                stage="mapped-source-association",
                            )
                        budget.reserve_geometry_queries(1)
                        budget.reserve(1)
                        points = corners[list(local)]
                        lower = np.all(points == 0, axis=0)
                        upper = np.all(points == counts[parent], axis=0)
                        selected = np.all(
                            (~lower | (vertices == 0)) & (~upper | (vertices == 1)), axis=1
                        )
                        key = tuple(sorted(root.vertices[index] for index in np.flatnonzero(selected)))
                        source = next(
                            ((d, lookup[key]) for d, lookup in enumerate(root_lookup) if key in lookup),
                            None,
                        )
                        if source is None or source[0] < degree:
                            raise ValueError("MappedReferenceAssociation.source_stratum must contain target entity")
                        source_degree, source_row = source
                        orientation = 0
                        if source_degree == degree and degree in (1, 2):
                            source_rows = root_entities[degree][source_row]
                            source_points = vertices[[
                                root.vertices.index(int(identifier))
                                for identifier in root_vertices[source_rows[source_rows >= 0]]
                            ]]
                            entity = target_entities[degree][target_row]
                            ordered = corners[[
                                int(np.flatnonzero(cell == vertex)[0])
                                for vertex in entity[entity >= 0]
                            ]]
                            # Positive diagonal tensor scaling preserves sign;
                            # the unchanged rational orientation owner validates
                            # each distinct ordered root/entity direction.
                            directions = ordered - ordered[0]
                            orientation_key = (
                                degree, tuple(int(n) for n in source_points.flat),
                                tuple(int(n) for n in directions.flat),
                            )
                            orientation = orientations.get(orientation_key, 0)
                            if orientation == 0:
                                orientation = _chart_orientation(
                                    tuple(tuple(Fraction(int(n)) for n in point) for point in source_points),
                                    tuple(tuple(Fraction(int(n)) for n in point) for point in directions),
                                    degree, budget=budget,
                                )
                                orientations[orientation_key] = orientation
                        choices[target_row].add((source_degree, source_row, orientation))
                    cursor += 1
            yield choices
            # Assembly has consumed this degree; do not overlap candidate banks.
            del choices, target_lookup, seen
    return _assemble_mapped_associations(
        domain, mesh, root_mesh, candidates(), budget=budget,
    ).associations
