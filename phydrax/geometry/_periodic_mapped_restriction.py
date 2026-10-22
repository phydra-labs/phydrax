# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Exact conforming subdivisions of an authenticated periodic mapped complex.

The canonical nested-reference owner proves full oriented reference tiling and
whole-map equality.  Vertex keys below prove that these tilings form one common
subcomplex, including all lower-dimensional source strata.  Unpaired fine
facets must lie on actual physical (not periodic-cut) coarse boundary facets.
Thus every translated child is a subset of its certified source parent image;
parent image intersections pull back to common subdivision entities.  Inverse
unions use the same partition in reverse, never a transplanted ancestor proof.

Mesh certificate checks
-----------------------
[
  {
    "owner": "phydrax.geometry._periodic_mapped_restriction.certify_periodic_mapped_restriction",
    "checks": [
      "periodic_mapped_restriction_source_embedding",
      "periodic_mapped_restriction_parent_partition",
      "periodic_mapped_restriction_common_subcomplex",
      "periodic_mapped_restriction_derived_region_assignment"
    ],
    "premises": [
      {"tag": "X", "statement": "The complete current source quotient embedding and positive source validity bind the actual source mesh and full coordinate maps. Source and target have the identical exact isometry group; current full target facet traces are rechecked under its actual actions."},
      {"tag": "X", "statement": "The canonical nested-reference owner proves every exact affine child map, full oriented parent tiling and zero whole-map deviation, with complete source and target cell coverage."},
      {"tag": "X", "statement": "Exact corner-weight keys are bijective with fine vertex IDs, retain every old corner identity and occur in every coarse cell incident on their minimal source stratum. Every unpaired fine quotient facet is a subdivision of an actual physical coarse boundary facet, never an unmatched interior or periodic-cut facet."},
      {"tag": "N", "statement": "Current target full-cell validity is positive. The source material assignment and exact namespace are independently proved by canonical whole-domain coverage; paired reference pieces derive a target assignment. The prepared-publication owner separately binds that derived assignment to the actual target request."}
    ],
    "conclusion": "The fine complex is a common conforming subdivision of the coarse complex under every group action. Refinement subsets, or inverse exact unions, inherit source-image interior disjointness and common-subcomplex contacts; the derived target material assignment preserves the proved source assignment.",
    "citation": {
      "kind": "mathematical",
      "reference": "De Loera, Rambau and Santos, Triangulations: Structures for Algorithms and Applications",
      "locator": "Chapter 2: subdivisions and the common-face intersection property, with the canonical oriented reference-chain tiling proof",
      "url": "https://doi.org/10.1007/978-3-642-12971-1"
    },
    "failure_statuses": ["unresolved"]
  }
]
End mesh certificate checks
"""
from __future__ import annotations

from fractions import Fraction
from typing import final
from weakref import WeakValueDictionary

import equinox as eqx
import numpy as np

from phydrax._fingerprint import canonical_fingerprint
from phydrax._meshcore import current_native_execution_scope
from phydrax._strict import StrictModule
from phydrax._validation import canonical_identifier, nonnegative_integer
from phydrax.typing import parse
from phydrax.discretization._cell_geometry_predecessor import CellGeometryPredecessor
from phydrax.discretization._cell_geometry_transition_types import CellGeometryTransition
from phydrax.discretization._cell_geometry_validity import cell_geometry_id, CellValidityCertificate
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._nested_reference import prepare_nested_reference
from phydrax.discretization._nested_reference_types import NestedReferencePair, rational_pair_affine, reference_pair_record
from phydrax.discretization._periodic_exact import (
    identification_id, identification_orders, reduce_periodic_exponents,
)
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax._trainable import NonTrainableState

from ._embedding_state import (
    facet_entity_ids, mesh_facets, MeshCertificateLimits, PeriodicPartitionDirection,
)
from ._planar_primitives import reserve_fraction_work


type _Key = tuple[tuple[int, Fraction], ...]


def _cells(mesh: CellMesh) -> tuple[tuple[str, tuple[int, ...]], ...]:
    ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    return tuple(
        (block.cell_kind, tuple(int(ids[vertex]) for vertex in row))
        for block in mesh.blocks
        for row in np.asarray(block.vertices, dtype=np.int64)
    )


def _weights(kind: str, point: tuple[Fraction, ...]) -> tuple[Fraction, ...] | None:
    x, y, z = point
    if kind == "tetrahedron":
        return 1 - x - y - z, x, y, z
    if kind == "prism":
        return (1 - x - y) * (1 - z), x * (1 - z), y * (1 - z), (1 - x - y) * z, x * z, y * z
    return None


def _physical_boundary(mesh: CellMesh, budget: CoordinateEnclosureBudget) -> set[frozenset[int]] | None:
    topology = mesh.periodic_topology
    if topology is None:
        return None
    facets = mesh_facets(mesh, budget=budget)
    count = facets.rows.shape[0]
    budget.reserve(
        count * (8 + max(1, count.bit_length())),
        1024 + 1024 * count,
    )
    entity_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    lookup = {int(identifier): row for row, identifier in enumerate(entity_ids)}
    ids = facet_entity_ids(mesh, facets.rows)
    orbits = np.asarray(topology.orbits(2)[0], dtype=np.int64)
    orbit_rows = np.asarray([orbits[lookup[int(identifier)]] for identifier in ids])
    unique, counts = np.unique(orbit_rows, return_counts=True)
    if np.any(counts > 2):
        return None
    single = set(unique[counts == 1].tolist())
    vertices = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    return {
        frozenset(int(vertices[vertex]) for vertex in row if vertex >= 0)
        for row, orbit in zip(facets.rows, orbit_rows, strict=True)
        if int(orbit) in single
    }


def _subcomplex(
    coarse: CellMesh,
    fine: CellMesh,
    pieces: tuple[tuple[int, int, NestedReferencePair], ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> str | None:
    """Prove compatible subdivisions on every source stratum, not just volumes."""
    count = sum(block.cell_count for mesh in (coarse, fine) for block in mesh.blocks)
    budget.reserve(8 * count, 1024 + 2048 * count)
    coarse_cells, fine_cells = _cells(coarse), _cells(fine)
    incident: dict[frozenset[int], set[int]] = {}
    for parent, (kind, vertices) in enumerate(coarse_cells):
        topology = reference_cell_topology(kind)
        for entities in topology.entities:
            for entity in entities:
                support = frozenset(vertices[index] for index in entity)
                incident.setdefault(support, set()).add(parent)
    keys: dict[int, _Key] = {}
    key_vertices: dict[_Key, int] = {}
    key_parents: dict[_Key, set[int]] = {}
    child_parents: dict[int, int] = {}
    for parent, child, pair in pieces:
        coarse_kind, parent_vertices = coarse_cells[parent]
        fine_kind, child_vertices = fine_cells[child]
        matrix, offset = rational_pair_affine(pair)
        budget.reserve(len(child_vertices) * 16, 512 + 512 * len(child_vertices))
        if child in child_parents:
            return None
        child_parents[child] = parent
        for identifier, vertex in zip(
            child_vertices, reference_cell_topology(fine_kind).vertices, strict=True
        ):
            exact_vertex = tuple(Fraction(float(value)) for value in vertex)
            reserve_fraction_work((*matrix, offset, exact_vertex), 16, 4, 32, budget=budget)
            point = tuple(
                shift + sum((value * coordinate for value, coordinate in zip(row, exact_vertex, strict=True)), Fraction(0))
                for row, shift in zip(matrix, offset, strict=True)
            )
            weights = _weights(coarse_kind, point)
            if weights is None or any(value < 0 for value in weights) or sum(weights, Fraction(0)) != 1:
                return None
            key = tuple(sorted((identifier_, value) for identifier_, value in zip(parent_vertices, weights, strict=True) if value))
            support = frozenset(identifier_ for identifier_, _ in key)
            if support not in incident:
                return None
            previous = keys.setdefault(identifier, key)
            if previous != key or key_vertices.setdefault(key, identifier) != identifier:
                return None
            # Old coarse corners keep their actual global identity.
            if len(key) == 1 and (key[0][1] != 1 or identifier != key[0][0]):
                return None
            key_parents.setdefault(key, set()).add(parent)
    if len(child_parents) != len(fine_cells):
        return None
    if set(keys) != set(np.asarray(fine.vertex_global_ids, dtype=np.int64).tolist()):
        return None
    for key, parents in key_parents.items():
        if parents != incident[frozenset(identifier for identifier, _ in key)]:
            return None
    coarse_periodic, fine_periodic = coarse.periodic_topology, fine.periodic_topology
    if coarse_periodic is None or fine_periodic is None:
        return None
    coarse_ids = np.asarray(coarse.vertex_global_ids, dtype=np.int64)
    coarse_rows = {int(identifier): row for row, identifier in enumerate(coarse_ids)}
    coarse_roots = np.asarray(coarse_periodic.vertex_representatives, dtype=np.int64)
    coarse_shifts = np.asarray(coarse_periodic.vertex_shifts, dtype=np.int64)
    coarse_fixed = np.asarray(coarse_periodic.vertex_fixed_generators, dtype=np.bool_)
    orders = identification_orders(coarse_periodic.cell)
    fine_ids = np.asarray(fine.vertex_global_ids, dtype=np.int64)
    fine_roots = np.asarray(fine_periodic.vertex_representatives, dtype=np.int64)
    fine_shifts = np.asarray(fine_periodic.vertex_shifts, dtype=np.int64)
    quotient_keys: dict[int, tuple[tuple[int, tuple[int, ...], Fraction], ...]] = {}
    quotient_vertices: dict[tuple[tuple[int, tuple[int, ...], Fraction], ...], int] = {}
    for row, identifier in enumerate(fine_ids):
        key = keys[int(identifier)]
        budget.reserve((8 + 2 * orders.size) * len(key), 256 + 128 * len(key))
        normalized = tuple(sorted(
            (
                int(coarse_ids[coarse_roots[coarse_rows[vertex]]]),
                tuple(int(value) for value in np.where(
                    coarse_fixed[coarse_rows[vertex]], 0,
                    reduce_periodic_exponents(
                        coarse_shifts[coarse_rows[vertex]] - fine_shifts[row], orders
                    ),
                )),
                weight,
            )
            for vertex, weight in key
        ))
        representative = int(fine_ids[fine_roots[row]])
        if (
            quotient_keys.setdefault(representative, normalized) != normalized
            or quotient_vertices.setdefault(normalized, representative) != representative
        ):
            return None
    # Canonical reference tiling pairs within-parent faces. Across parents the
    # current quotient facets must pair too; a lone interior cut is a T-junction.
    physical_coarse = _physical_boundary(coarse, budget)
    physical_fine = _physical_boundary(fine, budget)
    if physical_coarse is None or physical_fine is None:
        return None
    for face in physical_fine:
        support = frozenset(identifier for vertex in face for identifier, _ in keys[vertex])
        budget.reserve(len(physical_coarse), 128)
        if not any(support <= coarse_face for coarse_face in physical_coarse):
            return None
    records = tuple(
        (vertex, tuple((identifier, value.numerator, value.denominator) for identifier, value in key))
        for vertex, key in sorted(keys.items())
    )
    return canonical_fingerprint({"kind": "periodic-mapped-common-subcomplex", "vertices": records, "parents": tuple(sorted(child_parents.items())), "physical_boundary": tuple(sorted(tuple(sorted(face)) for face in physical_fine))})


def _trace_binding(
    predecessor: CellGeometryPredecessor,
    validity: CellValidityCertificate,
    budget: CoordinateEnclosureBudget,
) -> str | None:
    """Earn current complete quotient traces again when admitting stored evidence."""
    from . import _global_embedding as global_owner, _periodic_embedding as owner
    from ._embedding_state import EmbeddingState

    mesh, geometry, limits = (
        predecessor.target_mesh, predecessor.target_geometry, predecessor.source_limits
    )
    if mesh.periodic_topology is None or not isinstance(limits, MeshCertificateLimits):
        return None
    identifiers = np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks])
    state = EmbeddingState([], [])
    global_owner._validity_findings(state, validity, identifiers)
    progress = owner._PeriodicProgress()
    request = owner._PeriodicRequest(
        state, mesh, geometry, mesh.periodic_topology, identifiers, limits, budget,
        progress, validity,
    )
    group = owner._prepare_periodic_group(request)
    prepared = owner._prepare_periodic_cells(request, group)
    if prepared is None:
        return None
    facets = owner._prepare_periodic_facets(request, prepared, group)
    owner._certify_periodic_traces(request, prepared, group, facets)
    return progress.facet_trace_certificate_id if not state.findings else None


def _region_domain_id(domain: object) -> str:
    """Retain the exact domain owner; never lower CAD or approximate domains."""
    from ._domain_coverage import PiecewiseLinearDomain
    from ._mapped_reference_domain import MappedReferenceDomain

    if not isinstance(domain, (PiecewiseLinearDomain, MappedReferenceDomain)):
        raise TypeError(
            "PeriodicMappedRestrictionEvidence.source_domain must be an exact PL or mapped domain"
        )
    return domain.domain_id


def _prove(
    predecessor: CellGeometryPredecessor,
    transition: CellGeometryTransition,
    validity: CellValidityCertificate,
    source_regions: tuple[int, ...],
    target_regions: tuple[int, ...],
    source_domain: object,
    region_namespace_id: str,
    facet_trace_certificate_id: str,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[str, str, PeriodicPartitionDirection] | None:
    source, source_geometry = predecessor.source_mesh, predecessor.source_geometry
    target, geometry = predecessor.target_mesh, predecessor.target_geometry
    first, second = source.periodic_topology, target.periodic_topology
    certificate = predecessor.source_embedding
    if (
        first is None or second is None or source.storage is not None or target.storage is not None
        or source.topological_dimension != 3 or source.ambient_dimension != 3
        or target.topological_dimension != 3 or target.ambient_dimension != 3
        or identification_id(first.cell) != identification_id(second.cell)
        or certificate is None or certificate.status != "certified" or certificate.findings
        or not predecessor.source_validity.all_certified or not validity.all_certified
        or certificate.validity_certificate_id != predecessor.source_validity.certificate_id
    ):
        return None
    predecessor.require_bound(target, geometry, budget=budget)
    certificate.binding.require(source, source_geometry)
    predecessor.source_validity.require_bound(source_geometry, mesh=source)
    validity.require_bound(geometry, mesh=target)
    from . import _global_embedding as global_owner
    from ._domain_coverage import certify_domain_coverage, PiecewiseLinearDomain
    from ._mapped_reference_domain import MappedReferenceDomain
    from ._periodic_restriction import PeriodicRestrictionPremise

    # Reuse complete-route authentication, never merely a stored pass flag.
    global_owner._periodic_restriction_premise(
        PeriodicRestrictionPremise(
            source, source_geometry, predecessor.source_validity, certificate, ()
        ),
        target,
        budget=budget,
    )
    if (
        not isinstance(source_domain, (PiecewiseLinearDomain, MappedReferenceDomain))
        or source_domain.domain_id != region_namespace_id
    ):
        return None
    source_coverage = certify_domain_coverage(
        source, source_geometry, source_domain, np.asarray(source_regions, dtype=np.int64),
        embedding=certificate, limits=predecessor.source_limits, budget=budget,
    )
    if source_coverage.status != "certified" or source_coverage.findings:
        return None
    kinds = {block.cell_kind for mesh in (source, target) for block in mesh.blocks}
    if not kinds <= {"prism", "tetrahedron"}:
        return None
    if _trace_binding(predecessor, validity, budget) != facet_trace_certificate_id:
        return None
    nested = prepare_nested_reference(
        source, source_geometry, target, geometry,
        geometry_transition=transition, rooted=False, budget=budget,
    )
    if nested.certified_bound() != 0.0:
        return None
    pairs = nested.affine_pieces()
    inverse = any(pair.fine_is_source for pair in pairs)
    normalized = []
    for pair in pairs:
        if inverse and not pair.fine_is_source:
            matrix, offset = rational_pair_affine(pair)
            if any(value for value in offset) or any(
                value != int(row == column)
                for row, values in enumerate(matrix)
                for column, value in enumerate(values)
            ):
                # Simultaneous refine/coarsen is outside this union theorem.
                return None
            pair = pair._replace(fine_is_source=True)
        normalized.append(pair)
    source_count, target_count = (
        sum(block.cell_count for block in mesh.blocks) for mesh in (source, target)
    )
    if len(source_regions) != source_count or len(target_regions) != target_count:
        return None
    for pair in pairs:
        if source_regions[pair.source_cell] != target_regions[pair.target_cell]:
            return None
    coarse, fine = (target, source) if inverse else (source, target)
    pieces = tuple(
        (pair.target_cell, pair.source_cell, pair) if inverse
        else (pair.source_cell, pair.target_cell, pair)
        for pair in normalized
    )
    subcomplex = _subcomplex(coarse, fine, pieces, budget=budget)
    if subcomplex is None:
        return None
    partition = canonical_fingerprint({
        "kind": "periodic-mapped-oriented-reference-partition",
        "binding": nested.binding,
        "pieces": tuple(reference_pair_record(pair) for pair in pairs),
        "source_material_coverage": source_coverage.certificate_id,
        "derived_target_regions": target_regions,
    })
    direction: PeriodicPartitionDirection = "coarsen" if inverse else "refine"
    return partition, subcomplex, direction


def _evidence_identity(
    predecessor: CellGeometryPredecessor,
    transition: CellGeometryTransition,
    validity: CellValidityCertificate,
    source_regions: tuple[int, ...],
    target_regions: tuple[int, ...],
    source_domain: object,
    region_namespace_id: str,
    facet_trace_certificate_id: str,
    partition_id: str,
    subcomplex_id: str,
    direction: PeriodicPartitionDirection,
) -> str:
    certificate = predecessor.source_embedding
    return canonical_fingerprint({
        "kind": "periodic-mapped-restriction-evidence",
        "source_mesh": predecessor.source_mesh.mesh_id,
        "source_geometry": cell_geometry_id(predecessor.source_geometry),
        "source_validity": predecessor.source_validity.certificate_id,
        "source_embedding": None if certificate is None else certificate.certificate_id,
        "target_mesh": predecessor.target_mesh.mesh_id,
        "target_geometry": cell_geometry_id(predecessor.target_geometry),
        "target_validity": validity.certificate_id,
        "transition": transition.transition_id,
        "source_regions": source_regions,
        "target_regions": target_regions,
        "source_domain": _region_domain_id(source_domain),
        "region_namespace": region_namespace_id,
        "facet_traces": facet_trace_certificate_id,
        "partition": partition_id,
        "subcomplex": subcomplex_id,
        "direction": direction,
    })


@final
class PeriodicMappedRestrictionEvidence(StrictModule, NonTrainableState):
    """Retain actual source/target owners and exact partition/subcomplex witnesses."""

    predecessor: CellGeometryPredecessor
    transition: CellGeometryTransition
    validity: CellValidityCertificate
    source_regions: tuple[int, ...] = eqx.field(static=True)
    target_regions: tuple[int, ...] = eqx.field(static=True)
    source_domain: object
    region_namespace_id: str = eqx.field(static=True)
    facet_trace_certificate_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)
    subcomplex_id: str = eqx.field(static=True)
    direction: PeriodicPartitionDirection = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self, predecessor: CellGeometryPredecessor, transition: CellGeometryTransition,
        validity: CellValidityCertificate, source_regions: tuple[int, ...],
        target_regions: tuple[int, ...], source_domain: object, region_namespace_id: str,
        facet_trace_certificate_id: str, partition_id: str, subcomplex_id: str,
        direction: PeriodicPartitionDirection, evidence_id: str | None = None,
    ) -> None:
        prefix = "PeriodicMappedRestrictionEvidence"
        direction = parse(direction, PeriodicPartitionDirection, f"{prefix}.direction")
        source_regions = tuple(
            nonnegative_integer(value, f"{prefix}.source_regions")
            for value in source_regions
        )
        target_regions = tuple(
            nonnegative_integer(value, f"{prefix}.target_regions")
            for value in target_regions
        )
        if _region_domain_id(source_domain) != region_namespace_id:
            raise ValueError(
                f"{prefix}.region_namespace_id must bind its actual exact source domain"
            )
        region_namespace_id, facet_trace_certificate_id, partition_id, subcomplex_id = (
            canonical_identifier(value, f"{prefix}.{name}")
            for name, value in (
                ("region_namespace_id", region_namespace_id),
                ("facet_trace_certificate_id", facet_trace_certificate_id),
                ("partition_id", partition_id),
                ("subcomplex_id", subcomplex_id),
            )
        )
        expected = _evidence_identity(
            predecessor, transition, validity, source_regions, target_regions,
            source_domain, region_namespace_id, facet_trace_certificate_id, partition_id,
            subcomplex_id, direction,
        )
        if evidence_id is not None and evidence_id != expected:
            raise ValueError(
                f"{prefix}.evidence_id must bind every exact proof component; got changed evidence"
            )
        self.predecessor, self.transition, self.validity = predecessor, transition, validity
        self.source_regions, self.target_regions = source_regions, target_regions
        self.source_domain = source_domain
        self.region_namespace_id = region_namespace_id
        self.facet_trace_certificate_id = facet_trace_certificate_id
        self.partition_id, self.subcomplex_id, self.direction = partition_id, subcomplex_id, direction
        self.evidence_id = expected

    def _identity(self) -> str:
        return _evidence_identity(
            self.predecessor, self.transition, self.validity, self.source_regions,
            self.target_regions, self.source_domain, self.region_namespace_id, self.facet_trace_certificate_id,
            self.partition_id, self.subcomplex_id, self.direction,
        )

    @classmethod
    def from_stored_fields(
        cls: type[PeriodicMappedRestrictionEvidence],
        value: PeriodicMappedRestrictionEvidence,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> PeriodicMappedRestrictionEvidence:
        """Admit a live earned proof or reprove it under the caller's real budget."""
        if type(value) is not cls or value._identity() != value.evidence_id:
            raise ValueError(
                "PeriodicMappedRestrictionEvidence must retain its exact owning record and bindings"
            )
        if budget is not None:
            return _admit(value, budget=budget)
        scope = current_native_execution_scope()
        if scope is None:
            return _admit(value, budget=None)
        with scope.ticket_view() as ticket:
            view = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=scope.limits.maximum_work_units,
                maximum_memory_bytes=scope.limits.maximum_managed_bytes,
            )
            with view.activate():
                return _admit(value, budget=view)



_live_proofs: WeakValueDictionary[str, PeriodicMappedRestrictionEvidence] = WeakValueDictionary()


def _full_content(
    value: PeriodicMappedRestrictionEvidence,
    budget: CoordinateEnclosureBudget | None,
) -> str:
    from phydrax.lifecycle._meshing_source_records import stored_record_content_id

    return stored_record_content_id(value, excluded="", budget=budget)


def _admit(
    value: PeriodicMappedRestrictionEvidence,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> PeriodicMappedRestrictionEvidence:
    content = _full_content(value, budget)
    issued = _live_proofs.get(content)
    if issued is not None and (
        issued is value or _full_content(issued, budget) == content
    ):
        # Both actual records, including every source/target bank, contract and
        # policy, match an independently earned still-live proof. A fresh cold
        # process has no such owner and cannot gain authority from this digest.
        return value
    if budget is None:
        raise ValueError(
            "Unissued PeriodicMappedRestrictionEvidence requires the owning archive/request budget"
        )
    proved = _prove(
        value.predecessor, value.transition, value.validity, value.source_regions,
        value.target_regions, value.source_domain, value.region_namespace_id,
        value.facet_trace_certificate_id, budget=budget,
    )
    if proved != (value.partition_id, value.subcomplex_id, value.direction):
        raise ValueError(
            "PeriodicMappedRestrictionEvidence must reprove full source coverage, exact partition and common subcomplex"
        )
    budget.retain_basis((content,))
    _live_proofs[content] = value
    return value


def certify_periodic_mapped_restriction(
    predecessor: CellGeometryPredecessor | None, transition: CellGeometryTransition | None,
    validity: CellValidityCertificate | None, source_regions: tuple[int, ...] | None,
    target_regions: tuple[int, ...] | None, source_domain: object | None,
    region_namespace_id: str | None,
    facet_trace_certificate_id: str | None, *, budget: CoordinateEnclosureBudget,
) -> PeriodicMappedRestrictionEvidence | None:
    if current_native_execution_scope() is None or predecessor is None or transition is None or validity is None or source_regions is None or target_regions is None or source_domain is None or region_namespace_id is None or facet_trace_certificate_id is None:
        return None
    proved = _prove(predecessor, transition, validity, source_regions, target_regions, source_domain, region_namespace_id, facet_trace_certificate_id, budget=budget)
    if proved is None:
        return None
    partition, subcomplex, direction = proved
    value = PeriodicMappedRestrictionEvidence(
        predecessor, transition, validity, source_regions, target_regions,
        source_domain, region_namespace_id, facet_trace_certificate_id, partition, subcomplex, direction,
    )
    content = _full_content(value, budget)
    budget.retain_basis((content,))
    _live_proofs[content] = value
    return value
