#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Retain immutable source requests for successor recertification."""

from __future__ import annotations

from dataclasses import replace
from typing import final

import equinox as eqx
import jax
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.discretization._cell_geometry as _dependency_phydrax_discretization__cell_geometry
import phydrax.discretization._cell_geometry_validity as _dependency_phydrax_discretization__cell_geometry_validity
import phydrax.discretization._cell_mesh as _dependency_phydrax_discretization__cell_mesh
import phydrax.discretization._coordinate_budget as _dependency_phydrax_discretization__coordinate_budget
import phydrax.geometry._certified_implicit as _owner_phydrax_geometry__certified_implicit
import phydrax.geometry._domain_boundary as domain_boundary_owner
import phydrax.geometry._domain_coverage as domain_coverage_owner
import phydrax.geometry._domain_planar_boundary as cad_domain_owner
import phydrax.geometry._embedding_state as _dependency_phydrax_geometry__embedding_state
import phydrax.geometry._mapped_reference_coverage as mapped_reference_coverage_owner
import phydrax.geometry._mapped_reference_domain as _dependency_phydrax_geometry__mapped_reference_domain
import phydrax.geometry._source_boundary_records as source_boundary_records_owner
import phydrax.geometry._source_fidelity as source_fidelity_owner
import phydrax.geometry._surface_source_support as surface_source_support_owner
import phydrax.geometry.implicit._analytic_profile as _owner_phydrax_geometry_implicit__analytic_profile
import phydrax.meshing._audit as _dependency_phydrax_meshing__audit
import phydrax.meshing._certification as _owner_phydrax_meshing__certification
import phydrax.meshing._certification_schedule as _leaf_phydrax_meshing__certification_schedule
import phydrax.meshing._lineage_records as _owner_phydrax_meshing__lineage
import phydrax.meshing._surface_boundary_sources as surface_boundary_sources_owner

from .._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    tree_content_equal,
)
from .._host_storage import immutable_host_array
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import positive_integer
from ..typing import checked


@final
class _PreparedFidelityScopes(StrictModule, NonTrainableState):
    scopes: tuple[
        tuple[
            domain_boundary_owner.MeshingDomainBoundarySource, Array | np.ndarray, float
        ],
        ...,
    ]


def _prepare_fidelity_scopes(
    mesh: _dependency_phydrax_discretization__cell_mesh.CellMesh,
    scoped_fidelity: tuple[
        tuple[source_boundary_records_owner.SourceBoundaryQuery, ArrayLike, float], ...
    ],
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
    | None,
) -> _PreparedFidelityScopes:
    scopes: list[
        tuple[
            domain_boundary_owner.MeshingDomainBoundarySource,
            Array | np.ndarray,
            float,
        ]
    ] = []
    facet_ids = np.asarray(
        mesh.entity_set(mesh.topological_dimension - 1).entity_ids, dtype=np.int64
    )
    for query, selected, tolerance in scoped_fidelity:
        if not isinstance(
            query,
            domain_boundary_owner.MeshingDomainBoundarySource,
        ):
            raise TypeError(
                f"MeshCertificationInputs.scoped_fidelity.query must be MeshingDomainBoundarySource; got {type(query).__name__}"
            )
        indices = np.asarray(selected)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise TypeError(
                f"MeshCertificationInputs.scoped_fidelity.target_facets must be rank-one integer global IDs; got shape={indices.shape}, dtype={indices.dtype}"
            )
        if (
            indices.size == 0
            or np.unique(indices).size != indices.size
            or np.setdiff1d(indices, facet_ids).size
        ):
            raise ValueError(
                f"MeshCertificationInputs.scoped_fidelity.target_facets must select distinct mesh facets; got {indices.size} selected entries"
            )
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                f"MeshCertificationInputs.scoped_fidelity.tolerance must be finite and nonnegative; got {tolerance!r}"
            )
        _validate_query_integrity(query, budget=budget)
        scopes.append(
            (
                jax.tree.map(_immutable_host_leaf, query),
                np.asarray(np.sort(indices), dtype=np.int64),
                float(tolerance),
            )
        )
    scopes.sort(
        key=lambda item: (
            item[0].source_scope_id,
            tuple(np.asarray(item[1]).tolist()),
        )
    )
    if len(
        {
            (query.source_scope_id, tuple(np.asarray(indices).tolist()))
            for query, indices, _ in scopes
        }
    ) != len(scopes):
        raise ValueError(
            f"MeshCertificationInputs.scoped_fidelity must select distinct source/target traces; got {len(scopes)} requests with repeated traces"
        )
    return _PreparedFidelityScopes(tuple(scopes))


@final
class MeshCertificationInputs(StrictModule, NonTrainableState):
    """Retain immutable source data and certification assignments.

    Source fidelity and declared approximation-domain coverage have distinct
    identities. Renewal re-evaluates both obligations on the target geometry.
    ``cell_global_ids`` is a read-only C-contiguous int64 host array of the
    bound mesh's cell IDs in block order; ``request_id`` hashes its content.
    """

    schedule: _leaf_phydrax_meshing__certification_schedule.MeshCertificationSchedule
    domain: (
        domain_coverage_owner.PiecewiseLinearDomain
        | cad_domain_owner.CadSourceDomain
        | _dependency_phydrax_geometry__mapped_reference_domain.MappedReferenceDomain
        | None
    )
    source: source_boundary_records_owner.SourceBoundaryQuery | None
    cell_regions: Array | None
    junction_vertices: Array | None
    scoped_fidelity: tuple[
        tuple[
            domain_boundary_owner.MeshingDomainBoundarySource,
            Array,
            float,
        ],
        ...,
    ]
    limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits
    mapped_source_evidence: (
        mapped_reference_coverage_owner.MappedReferenceSourceEvidence | None
    )
    fidelity_tolerance: float | None = eqx.field(static=True)
    fidelity_sample_order: int = eqx.field(static=True)
    mesh_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    cell_global_ids: np.ndarray
    entity_set_ids: tuple[tuple[int, str], ...] = eqx.field(static=True)
    source_id: str | None = eqx.field(static=True)
    source_revision: str | None = eqx.field(static=True)
    domain_id: str | None = eqx.field(static=True)
    request_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        mesh: _dependency_phydrax_discretization__cell_mesh.CellMesh,
        geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
        schedule: _leaf_phydrax_meshing__certification_schedule.MeshCertificationSchedule,
        /,
        *,
        domain: domain_coverage_owner.PiecewiseLinearDomain
        | _dependency_phydrax_geometry__mapped_reference_domain.MappedReferenceDomain
        | cad_domain_owner.CadSourceDomain
        | None = None,
        cell_regions: ArrayLike | None = None,
        source: source_boundary_records_owner.SourceBoundaryQuery | None = None,
        fidelity_tolerance: float | None = None,
        fidelity_sample_order: int = 4,
        limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits
        | None = None,
        junction_vertices: ArrayLike | None = None,
        scoped_fidelity: tuple[
            tuple[
                source_boundary_records_owner.SourceBoundaryQuery,
                ArrayLike,
                float,
            ],
            ...,
        ] = (),
        mapped_source_evidence: mapped_reference_coverage_owner.MappedReferenceSourceEvidence
        | None = None,
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
        | None = None,
    ) -> None:

        if source is not None and (
            not isinstance(
                source,
                source_boundary_records_owner.SourceBoundaryQuery,
            )
            or not isinstance(source, StrictModule)
        ):
            raise TypeError(
                f"MeshCertificationInputs.source must be an immutable StrictModule source query; got {type(source).__name__}"
            )
        limits_ = (
            _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits()
            if limits is None
            else limits
        )
        identifiers = np.concatenate(
            [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
        )
        regions = (
            None if cell_regions is None else np.asarray(cell_regions, dtype=np.int64)
        )
        if regions is not None and regions.shape != identifiers.shape:
            raise ValueError(
                f"MeshCertificationInputs.cell_regions must have one entry per mesh cell; got shape={regions.shape}"
            )
        junctions = (
            None
            if junction_vertices is None
            else np.unique(np.asarray(junction_vertices, dtype=np.int64))
        )
        if (
            junctions is not None
            and np.setdiff1d(
                junctions, np.asarray(mesh.vertex_global_ids, dtype=np.int64)
            ).size
        ):
            raise ValueError(
                f"MeshCertificationInputs.junction_vertices must select mesh vertex global IDs; got {junctions.size} entries including foreign IDs"
            )
        if fidelity_tolerance is not None and (
            not np.isfinite(fidelity_tolerance) or fidelity_tolerance < 0.0
        ):
            raise ValueError(
                f"MeshCertificationInputs.fidelity_tolerance must be finite and nonnegative; got {fidelity_tolerance!r}"
            )
        fidelity_sample_order = positive_integer(
            fidelity_sample_order, "fidelity_sample_order"
        )

        scopes = _prepare_fidelity_scopes(mesh, scoped_fidelity, budget).scopes
        self.schedule = schedule
        self.domain = (
            None if domain is None else jax.tree.map(_immutable_host_leaf, domain)
        )
        self.source = (
            None if source is None else jax.tree.map(_immutable_host_leaf, source)
        )
        device_regions, device_junctions, device_scopes = jax.device_put(
            (
                None if regions is None else np.asarray(regions, dtype=np.int64),
                None if junctions is None else np.asarray(junctions, dtype=np.int64),
                tuple(selected for _, selected, _ in scopes),
            )
        )
        self.cell_regions = device_regions
        self.junction_vertices = device_junctions
        self.scoped_fidelity = tuple(
            (query, selected, tolerance)
            for (query, _, tolerance), selected in zip(scopes, device_scopes, strict=True)
        )
        self.limits = limits_
        if isinstance(
            self.domain,
            _dependency_phydrax_geometry__mapped_reference_domain.MappedReferenceDomain,
        ):
            if mapped_source_evidence is None and budget is not None:
                mapped_source_evidence = (
                    mapped_reference_coverage_owner.retained_mapped_reference_source_evidence(
                        self.domain, limits_, budget=budget
                    )
                )
            if mapped_source_evidence is not None:
                mapped_source_evidence.require_live(self.domain, limits_)
                if budget is not None:
                    mapped_reference_coverage_owner.retain_mapped_reference_source_evidence(
                        mapped_source_evidence, self.domain, limits_, budget=budget
                    )
        elif mapped_source_evidence is not None:
            raise ValueError(
                "MeshCertificationInputs.mapped_source_evidence must bind a mapped "
                "reference domain; got another domain kind."
            )
        self.mapped_source_evidence = mapped_source_evidence
        self.fidelity_tolerance = fidelity_tolerance
        self.fidelity_sample_order = int(fidelity_sample_order)
        self.mesh_id = mesh.mesh_id
        self.topology_id = mesh.topology_id
        self.geometry_id = (
            _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                geometry
            )
        )
        self.entity_set_ids = tuple(
            (dimension, mesh.entity_set(dimension).entity_set_id)
            for dimension in range(mesh.topological_dimension + 1)
        )
        self.cell_global_ids = immutable_host_array(
            np.ascontiguousarray(identifiers, dtype=np.int64)
        )
        self.source_id = None if source is None else source.source_id
        self.source_revision = None if source is None else source.source_revision
        self.domain_id = None if domain is None else domain.domain_id
        self.request_id = _request_identity(self)

    @checked
    def validate_source_integrity(
        self,
        /,
        *,
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
        | None = None,
    ) -> None:
        """Rebuild owned source facts; stored fingerprints alone are not proof."""

        schedule = (
            _leaf_phydrax_meshing__certification_schedule.MeshCertificationSchedule(
                self.schedule.route
            )
        )
        limits = replace(self.limits)
        if not tree_content_equal(self.schedule, schedule) or not tree_content_equal(
            self.limits, limits
        ):
            raise ValueError(
                "MeshCertificationInputs.policy must match its reconstructed policy; got changed policy fields"
            )
        cell_ids = self.cell_global_ids
        if (
            type(cell_ids) is not np.ndarray
            or cell_ids.dtype != np.int64
            or cell_ids.ndim != 1
        ):
            raise TypeError(
                "MeshCertificationInputs.cell_global_ids must be a one-dimensional int64 host array; got another cell-ID representation"
            )
        if _request_identity(self) != self.request_id:
            raise ValueError(
                f"MeshCertificationInputs.request_id must match its assignments and bindings; got {self.request_id!r}"
            )
        domain = self.domain
        if domain is not None:
            _validate_domain_integrity(domain, budget=budget)
        if self.mapped_source_evidence is not None:
            if not isinstance(
                domain,
                _dependency_phydrax_geometry__mapped_reference_domain.MappedReferenceDomain,
            ):
                raise ValueError(
                    "MeshCertificationInputs.mapped_source_evidence must retain its "
                    "mapped reference domain."
                )
            self.mapped_source_evidence.require_live(domain, self.limits)
        source = self.source
        if source is not None:
            _validate_query_integrity(source, budget=budget)
            if (
                source.source_id != self.source_id
                or source.source_revision != self.source_revision
            ):
                raise ValueError(
                    f"MeshCertificationInputs.source must retain its source identity; got {(source.source_id, source.source_revision)!r}"
                )
        if domain is not None and domain.domain_id != self.domain_id:
            raise ValueError(
                f"MeshCertificationInputs.domain must retain its domain identity; got {domain.domain_id!r}"
            )
        for query, _, _ in self.scoped_fidelity:
            _validate_query_integrity(query, budget=budget)

    @checked
    def transition_request(
        self,
        mesh: _dependency_phydrax_discretization__cell_mesh.CellMesh,
        geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
        /,
        *,
        lineage: _owner_phydrax_meshing__lineage.MeshLineage,
        cell_regions: ArrayLike | None = None,
        junction_vertices: ArrayLike | None = None,
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
        | None = None,
    ) -> MeshCertificationInputs:
        """Transfer assignments and retain unchanged source obligations."""

        self.validate_source_integrity(budget=budget)

        source_sets = dict(self.entity_set_ids)
        for relation in lineage.entities:
            if (
                relation.source_entity_set_id != source_sets.get(relation.dimension)
                or relation.target_entity_set_id
                != mesh.entity_set(relation.dimension).entity_set_id
            ):
                raise ValueError(
                    f"MeshCertificationInputs.lineage must bind the source and target entity sets; got {(relation.source_entity_set_id, relation.target_entity_set_id)!r}"
                )
        if (
            lineage.source_topology_id != self.topology_id
            or lineage.target_topology_id != mesh.topology_id
        ):
            raise ValueError(
                f"MeshCertificationInputs.lineage must bind the source and target topology; got {(lineage.source_topology_id, lineage.target_topology_id)!r}"
            )
        if self.source is not None and (
            self.source.source_id != self.source_id
            or self.source.source_revision != self.source_revision
        ):
            raise ValueError(
                f"MeshCertificationInputs.source must retain its prepared revision; got {self.source.source_revision!r}"
            )
        regions = _renew_transition_regions(self, mesh, lineage, cell_regions).regions
        junctions = junction_vertices
        if self.junction_vertices is not None and junctions is None:
            relations = lineage.entity_lineage(0)
            old_junctions = set(
                np.asarray(self.junction_vertices, dtype=np.int64).tolist()
            )

            identity_kinds = {
                int(_owner_phydrax_meshing__lineage.EntityLineageKind.PRESERVED),
                int(_owner_phydrax_meshing__lineage.EntityLineageKind.RELOCATED),
                int(_owner_phydrax_meshing__lineage.EntityLineageKind.COLLAPSED_INTO),
                int(_owner_phydrax_meshing__lineage.EntityLineageKind.MERGED_INTO),
            }
            mapped = [
                (source, target)
                for source, target, kind in zip(
                    np.asarray(relations.source_global_ids).tolist(),
                    np.asarray(relations.target_global_ids).tolist(),
                    np.asarray(relations.relation_kinds).tolist(),
                    strict=True,
                )
                if source in old_junctions and kind in identity_kinds
            ]
            if {source for source, _ in mapped} != old_junctions:
                raise ValueError(
                    f"MeshCertificationInputs.junction_vertices must have identity-preserving target relations or explicit revalidated junctions; got {len(mapped)} retained relations"
                )
            junctions = np.asarray(
                sorted({target for _, target in mapped}), dtype=np.int64
            )

        source = self.source
        if isinstance(source, source_fidelity_owner.MappedDomainBoundarySource):
            if regions is None:
                raise ValueError(
                    "MeshCertificationInputs.cell_regions must supply mapped-source target assignments; got None"
                )
            source = source.for_target_regions(regions)
        scoped = _renew_transition_scopes(self, mesh, lineage).scopes
        return MeshCertificationInputs(
            mesh,
            geometry,
            self.schedule,
            domain=self.domain,
            cell_regions=regions,
            source=source,
            fidelity_tolerance=self.fidelity_tolerance,
            fidelity_sample_order=self.fidelity_sample_order,
            limits=self.limits,
            junction_vertices=junctions,
            scoped_fidelity=tuple(scoped),
            mapped_source_evidence=self.mapped_source_evidence,
            budget=budget,
        )

    @checked
    def recertify_transition(
        self,
        mesh: _dependency_phydrax_discretization__cell_mesh.CellMesh,
        geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
        audit: _dependency_phydrax_meshing__audit.CellMeshAuditReport,
        /,
        *,
        lineage: _owner_phydrax_meshing__lineage.MeshLineage,
        cell_regions: ArrayLike | None = None,
        junction_vertices: ArrayLike | None = None,
        prepared: _owner_phydrax_meshing__certification.PreparedMeshingAcceptance
        | None = None,
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
        | None = None,
    ) -> _owner_phydrax_meshing__certification.MeshCertificationReport:
        """Renew acceptance using the source request and target evidence."""

        with _dependency_phydrax_geometry__embedding_state.coordinate_certificate_request(
            self.limits, budget=budget
        ) as ledger:
            request = self.transition_request(
                mesh,
                geometry,
                lineage=lineage,
                cell_regions=cell_regions,
                junction_vertices=junction_vertices,
                budget=ledger,
            )
            return _owner_phydrax_meshing__certification.certify_meshing_acceptance(
                mesh,
                geometry,
                audit,
                schedule=request.schedule,
                domain=request.domain,
                cell_regions=request.cell_regions,
                source=request.source,
                fidelity_tolerance=request.fidelity_tolerance,
                fidelity_sample_order=request.fidelity_sample_order,
                limits=request.limits,
                junction_vertices=request.junction_vertices,
                scoped_fidelity=request.scoped_fidelity,
                prepared=prepared,
                budget=ledger,
            )


def _renew_transition_scopes(
    self: MeshCertificationInputs,
    mesh: _dependency_phydrax_discretization__cell_mesh.CellMesh,
    lineage: _owner_phydrax_meshing__lineage.MeshLineage,
) -> _PreparedFidelityScopes:
    scoped = []
    if self.scoped_fidelity:
        relations = lineage.entity_lineage(mesh.topological_dimension - 1)
        sources = np.asarray(relations.source_global_ids, dtype=np.int64)
        targets = np.asarray(relations.target_global_ids, dtype=np.int64)
        memberships: dict[int, set[int]] = {}
        for index, (_, identifiers, _) in enumerate(self.scoped_fidelity):
            selected = set(np.asarray(identifiers, dtype=np.int64).tolist())
            addressed = set(
                sources[
                    np.isin(sources, np.asarray(sorted(selected), dtype=np.int64))
                ].tolist()
            )
            if addressed != selected:
                raise ValueError(
                    f"MeshCertificationInputs.scoped_fidelity must retain every target trace facet; got {len(addressed)} addressed of {len(selected)} selected facets"
                )
            for before, after in zip(sources.tolist(), targets.tolist(), strict=True):
                if before in selected:
                    memberships.setdefault(after, set()).add(index)
        for index, (query, _, tolerance) in enumerate(self.scoped_fidelity):
            identifiers = np.asarray(
                sorted(
                    target for target, classes in memberships.items() if index in classes
                ),
                dtype=np.int64,
            )
            if identifiers.size == 0:
                raise ValueError(
                    f"MeshCertificationInputs.scoped_fidelity must retain each source stratum; got no target facets for scope {query.source_scope_id!r}"
                )
            for facet in identifiers.tolist():
                ancestors = set(sources[targets == facet].tolist())
                selected = set(
                    np.asarray(self.scoped_fidelity[index][1], dtype=np.int64).tolist()
                )
                if not ancestors <= selected:
                    raise ValueError(
                        f"MeshCertificationInputs.scoped_fidelity must keep coarsened facets within one source stratum; got target facet {facet!r} with mixed ancestry"
                    )
            scoped.append((query, identifiers, tolerance))
    return _PreparedFidelityScopes(tuple(scoped))


@final
class _TransitionRegions(StrictModule, NonTrainableState):
    regions: ArrayLike | None


def _renew_transition_regions(
    self: MeshCertificationInputs,
    mesh: _dependency_phydrax_discretization__cell_mesh.CellMesh,
    lineage: _owner_phydrax_meshing__lineage.MeshLineage,
    cell_regions: ArrayLike | None,
) -> _TransitionRegions:
    regions = cell_regions
    if self.cell_regions is not None and regions is None:
        relations = lineage.entity_lineage(mesh.topological_dimension)
        source_ids = np.asarray(self.cell_global_ids, dtype=np.int64)
        source_regions = np.asarray(self.cell_regions, dtype=np.int64)
        if source_regions.shape != source_ids.shape:
            raise ValueError(
                f"MeshCertificationInputs.cell_regions must have one entry per retained cell ID; got shape={source_regions.shape}"
            )
        # A repeated source ID keeps its last assignment, as a mapping update would.
        known_ids, reversed_rows = np.unique(source_ids[::-1], return_index=True)
        known_regions = source_regions[source_ids.size - 1 - reversed_rows]
        relation_sources = np.asarray(relations.source_global_ids, dtype=np.int64)
        relation_targets = np.asarray(relations.target_global_ids, dtype=np.int64)
        if relation_sources.shape != relation_targets.shape:
            raise ValueError(
                "MeshCertificationInputs.lineage must pair every source and target cell; got unequal relation rows"
            )
        source_rows, known = _sorted_rows(known_ids, relation_sources)
        if not np.all(known):
            unknown = int(relation_sources[np.flatnonzero(~known)[0]])
            raise ValueError(
                f"MeshCertificationInputs.cell_regions must use known source-cell ancestry; got {unknown!r}"
            )
        pairs = np.unique(
            np.stack((relation_targets, known_regions[source_rows]), axis=1), axis=0
        )
        targets, first_pair, pair_counts = np.unique(
            pairs[:, 0], return_index=True, return_counts=True
        )
        identifiers = np.concatenate(
            [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
        )
        target_rows, found = _sorted_rows(targets, identifiers)
        single = np.zeros(found.shape, dtype=np.bool_)
        single[found] = pair_counts[target_rows[found]] == 1
        if not np.all(single):
            raise ValueError(
                f"MeshCertificationInputs.cell_regions must provide explicit assignments for new or cross-material cells; got {identifiers.size} targets with missing or ambiguous ancestry"
            )
        regions = pairs[first_pair[target_rows], 1]
    return _TransitionRegions(regions)


def _sorted_rows(keys: np.ndarray, queries: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    """Rows of sorted unique ``keys`` equal to each query and whether one exists."""
    if keys.size == 0:
        return np.zeros(queries.shape, dtype=np.int64), np.zeros(
            queries.shape, dtype=np.bool_
        )
    rows = np.minimum(np.searchsorted(keys, queries), keys.size - 1)
    return rows, keys[rows] == queries


def _immutable_host_leaf(value: object) -> object:
    if isinstance(value, np.ndarray):
        if (
            type(value) is np.ndarray
            and not value.dtype.hasobject
            and value.dtype.kind in "biufc"
        ):
            return immutable_host_array(value)
        snapshot = np.array(value, copy=True)
        snapshot.setflags(write=False)
        return snapshot
    return value


def _validate_domain_integrity(
    domain: domain_coverage_owner.PiecewiseLinearDomain
    | _dependency_phydrax_geometry__mapped_reference_domain.MappedReferenceDomain
    | cad_domain_owner.CadSourceDomain,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
    | None,
) -> None:
    match domain:
        case domain_coverage_owner.PiecewiseLinearDomain():
            rebuilt = domain_coverage_owner.PiecewiseLinearDomain.from_stored_fields(
                domain
            )
        case _dependency_phydrax_geometry__mapped_reference_domain.MappedReferenceDomain():
            _validate_domain_integrity(domain.reference_domain, budget=budget)
            rebuilt = _dependency_phydrax_geometry__mapped_reference_domain.MappedReferenceDomain(
                domain.reference_domain,
                domain.reference_mesh,
                domain.source_geometry,
                domain.cell_regions,
                source_id=domain.source_id,
                source_revision=domain.source_revision,
                certificate_limits=domain.certificate_limits,
                budget=budget,
            )
        case cad_domain_owner.CadSourceDomain():
            from ..lifecycle._meshing_sources import (
                meshing_source_owner_reconstruction_scope,
                reconstruct_meshing_source,
                register_meshing_source_artifacts,
            )

            register_meshing_source_artifacts()
            with meshing_source_owner_reconstruction_scope(budget=budget):
                restored = reconstruct_meshing_source(domain)
            rebuilt = domain if restored is None else restored
        case _:
            raise TypeError(
                "MeshCertificationInputs.domain must have a supported restoration kind; got unsupported domain"
            )
    if rebuilt.domain_id != domain.domain_id or not tree_content_equal(domain, rebuilt):
        raise ValueError(
            f"MeshCertificationInputs.domain must match its restored numerical facts; got stored domain_id={domain.domain_id!r}, rebuilt domain_id={rebuilt.domain_id!r}"
        )


def _validate_query_integrity(
    source: source_boundary_records_owner.SourceBoundaryQuery,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
    | None,
) -> None:

    match source:
        case source_fidelity_owner.ImplicitBoundarySource():
            if (
                _owner_phydrax_geometry__certified_implicit.implicit_state_id(
                    source.geometry
                )
                != source.source_revision
            ):
                raise ValueError(
                    f"MeshCertificationInputs.source must match its stored implicit revision; got {source.source_revision!r}"
                )
            rebuilt = source_fidelity_owner.ImplicitBoundarySource(
                source.geometry, source_id=source.source_id, spacing=source.spacing
            )
        case _owner_phydrax_geometry_implicit__analytic_profile.AnalyticImplicitProfile():
            source.require_bound(
                source.geometry, source.coordinate_contract, budget=budget
            )
            rebuilt = _owner_phydrax_geometry_implicit__analytic_profile.AnalyticImplicitProfile(
                source.geometry,
                source.coordinate_contract,
                tube_radius=source.tube_radius,
                cover_radius=source.cover_radius,
                source_id=source.source_id,
                source_revision=source.source_revision,
                budget=budget,
            )
        case source_fidelity_owner.ImplicitProjectionBoundarySource():
            _validate_query_integrity(source.profile, budget=budget)
            rebuilt = source_fidelity_owner.ImplicitProjectionBoundarySource(
                source.profile, budget=budget
            )
        case source_fidelity_owner.MappedDomainBoundarySource():
            domain = source.domain
            if not isinstance(
                domain,
                _dependency_phydrax_geometry__mapped_reference_domain.MappedReferenceDomain,
            ):
                raise TypeError(
                    f"MeshCertificationInputs.source.domain must be MappedReferenceDomain; got {type(domain).__name__}"
                )
            _validate_domain_integrity(domain, budget=budget)
            rebuilt = source_fidelity_owner.MappedDomainBoundarySource(
                domain,
                source.cell_regions,
                limits=source.limits,
                covering_radius=source.covering_radius,
            )
        case source_fidelity_owner.ParametricCurveBoundarySource():
            rebuilt = source_fidelity_owner.ParametricCurveBoundarySource(
                source.curves,
                source.parameter_ranges,
                source_id=source.source_id,
                source_revision=source.source_revision,
                covering_radius=source.covering_radius,
                maximum_samples=source.maximum_samples,
            )
        case domain_boundary_owner.MeshingDomainBoundarySource():
            source.domain.require_current(source.source_id, source.source_revision)
            rebuilt = domain_boundary_owner.MeshingDomainBoundarySource(
                source.domain,
                source.patches,
                resolution=source.resolution,
                chart_triangulations=source.chart_triangulations,
                boundary_occurrences=source.boundary_occurrences,
                boundary_cell_ids=source.boundary_cell_ids,
                boundary_patch_ids=source.boundary_patch_ids,
            )
        case surface_source_support_owner.SurfaceNativeRestrictionBoundarySource():
            _validate_query_integrity(source.root, budget=budget)
            surface_source_support_owner.SurfaceNativeRestrictionBoundarySource.require_current(source)
            geometry = _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec.from_stored_fields(
                source.geometry, budget=budget
            )
            _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec.resolve(
                geometry, source.mesh, budget=budget
            )
            rebuilt = surface_source_support_owner.SurfaceNativeRestrictionBoundarySource(
                source.root, source.support, source.mesh, geometry
            )
        case (
            surface_boundary_sources_owner.SurfaceSubdivisionBoundarySource()
            | surface_boundary_sources_owner.SurfaceChartBoundarySource()
        ):
            _validate_query_integrity(source.root, budget=budget)
            source.require_current()
            return
        case _:
            raise TypeError(
                f"MeshCertificationInputs.source must have a supported restoration kind; got {type(source).__name__}"
            )
    if not tree_content_equal(source, rebuilt):
        raise ValueError(
            f"MeshCertificationInputs.source must match its rebuilt numerical facts; got changed facts for source_id={source.source_id!r}"
        )


def _request_identity(inputs: MeshCertificationInputs) -> str:
    return canonical_fingerprint(
        {
            "kind": "mesh-certification-inputs",
            "schedule": inputs.schedule.schedule_id,
            "mesh": inputs.mesh_id,
            "topology": inputs.topology_id,
            "geometry": inputs.geometry_id,
            "source": (inputs.source_id, inputs.source_revision),
            "domain": inputs.domain_id,
            "regions": None
            if inputs.cell_regions is None
            else array_tree_fingerprint(inputs.cell_regions),
            "entity_sets": inputs.entity_set_ids,
            "cell_ids": array_tree_fingerprint(inputs.cell_global_ids),
            "junctions": None
            if inputs.junction_vertices is None
            else array_tree_fingerprint(inputs.junction_vertices),
            "tolerance": inputs.fidelity_tolerance,
            "sample_order": inputs.fidelity_sample_order,
            "limits": inputs.limits.limits_id,
            "mapped_source_evidence": None
            if inputs.mapped_source_evidence is None
            else inputs.mapped_source_evidence.evidence_id,
            "scoped_fidelity": tuple(
                (query.source_scope_id, array_tree_fingerprint(identifiers), tolerance)
                for query, identifiers, tolerance in inputs.scoped_fidelity
            ),
        }
    )
