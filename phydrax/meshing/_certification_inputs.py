#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Immutable source requests retained for actual successor recertification."""

from __future__ import annotations

from typing import final, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._cell_mesh import CellMesh
from ..geometry._mapped_reference_domain import MappedReferenceDomain
from ..geometry._mesh_certificates import (
    MeshCertificateLimits,
    PiecewiseLinearDomain,
    SourceBoundaryQuery,
)
from ..geometry._meshing_domain import MeshingDomainBoundarySource
from ._audit import CellMeshAuditReport


if TYPE_CHECKING:
    from ..geometry._mesh_certificates import (
        GlobalEmbeddingCertificate,
        SourceFidelityCertificate,
    )
    from ._certification import (
        MeshCertificationPreparedEvidence,
        MeshCertificationReport,
        MeshCertificationSchedule,
    )
    from ._lineage import MeshLineage


@final
class MeshCertificationInputs(StrictModule, NonTrainableState):
    """Owning immutable source data, never an external CAD handle or callback.

    Original-source fidelity and separately declared approximation-domain
    coverage retain distinct identities. Renewal reruns their actual theorems.
    """

    schedule: MeshCertificationSchedule
    domain: PiecewiseLinearDomain | MappedReferenceDomain | None
    source: SourceBoundaryQuery | None
    cell_regions: Array | None
    junction_vertices: Array | None
    scoped_fidelity: tuple[tuple[MeshingDomainBoundarySource, Array, float], ...]
    limits: MeshCertificateLimits
    fidelity_tolerance: float | None = eqx.field(static=True)
    fidelity_sample_order: int = eqx.field(static=True)
    mesh_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    cell_global_ids: tuple[int, ...] = eqx.field(static=True)
    entity_set_ids: tuple[tuple[int, str], ...] = eqx.field(static=True)
    source_id: str | None = eqx.field(static=True)
    source_revision: str | None = eqx.field(static=True)
    domain_id: str | None = eqx.field(static=True)
    request_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        schedule: MeshCertificationSchedule,
        /,
        *,
        domain: PiecewiseLinearDomain | MappedReferenceDomain | None = None,
        cell_regions: ArrayLike | None = None,
        source: SourceBoundaryQuery | None = None,
        fidelity_tolerance: float | None = None,
        fidelity_sample_order: int = 4,
        limits: MeshCertificateLimits | None = None,
        junction_vertices: ArrayLike | None = None,
        scoped_fidelity: tuple[tuple[SourceBoundaryQuery, ArrayLike, float], ...] = (),
    ) -> None:
        from ._certification import MeshCertificationSchedule

        if not isinstance(schedule, MeshCertificationSchedule):
            raise TypeError("schedule must be MeshCertificationSchedule.")
        if domain is not None and not isinstance(
            domain, (PiecewiseLinearDomain, MappedReferenceDomain)
        ):
            raise TypeError(
                "domain must be a canonical declared coverage source or None."
            )
        if source is not None and (
            not isinstance(source, SourceBoundaryQuery)
            or not isinstance(source, StrictModule)
        ):
            raise TypeError(
                "Retained source queries must be immutable owning StrictModule objects."
            )
        limits_ = MeshCertificateLimits() if limits is None else limits
        if not isinstance(limits_, MeshCertificateLimits):
            raise TypeError("limits must be MeshCertificateLimits or None.")
        identifiers = np.concatenate(
            [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
        )
        regions = (
            None if cell_regions is None else np.asarray(cell_regions, dtype=np.int64)
        )
        if regions is not None and regions.shape != identifiers.shape:
            raise ValueError(
                "Retained region assignments require one entry per actual cell."
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
            raise ValueError("Retained junctions must be actual mesh vertex global IDs.")
        if fidelity_tolerance is not None and (
            not np.isfinite(fidelity_tolerance) or fidelity_tolerance < 0.0
        ):
            raise ValueError("fidelity_tolerance must be finite and nonnegative.")
        if (
            isinstance(fidelity_sample_order, bool)
            or not isinstance(fidelity_sample_order, (int, np.integer))
            or fidelity_sample_order <= 0
        ):
            raise ValueError("fidelity_sample_order must be a positive integer.")

        scopes: list[tuple[MeshingDomainBoundarySource, Array, float]] = []
        facet_ids = np.asarray(
            mesh.entity_set(mesh.topological_dimension - 1).entity_ids, dtype=np.int64
        )
        for query, selected, tolerance in scoped_fidelity:
            if not isinstance(query, MeshingDomainBoundarySource):
                raise TypeError(
                    "Scoped fidelity requires its nominal original parametric face query."
                )
            indices = np.asarray(selected)
            if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
                raise TypeError(
                    "Scoped fidelity target facets must be rank-one integer global IDs."
                )
            if (
                indices.size == 0
                or np.unique(indices).size != indices.size
                or np.setdiff1d(indices, facet_ids).size
            ):
                raise ValueError(
                    "Scoped fidelity must select distinct actual target facets."
                )
            if not np.isfinite(tolerance) or tolerance < 0.0:
                raise ValueError(
                    "Scoped source tolerances must be finite and nonnegative."
                )
            _validate_query_integrity(query)
            scopes.append(
                (
                    jax.tree.map(_immutable_host_leaf, query),
                    jnp.asarray(np.sort(indices), dtype=jnp.int64),
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
                "Scoped source requests must not duplicate the same source/target trace."
            )
        self.schedule = schedule
        self.domain = (
            None if domain is None else jax.tree.map(_immutable_host_leaf, domain)
        )
        self.source = (
            None if source is None else jax.tree.map(_immutable_host_leaf, source)
        )
        self.cell_regions = (
            None if regions is None else jnp.asarray(regions, dtype=jnp.int64)
        )
        self.junction_vertices = (
            None if junctions is None else jnp.asarray(junctions, dtype=jnp.int64)
        )
        self.scoped_fidelity = tuple(scopes)
        self.limits = limits_
        self.fidelity_tolerance = fidelity_tolerance
        self.fidelity_sample_order = int(fidelity_sample_order)
        self.mesh_id = mesh.mesh_id
        self.topology_id = mesh.topology_id
        self.geometry_id = cell_geometry_id(geometry)
        self.entity_set_ids = tuple(
            (dimension, mesh.entity_set(dimension).entity_set_id)
            for dimension in range(mesh.topological_dimension + 1)
        )
        self.cell_global_ids = tuple(identifiers.tolist())
        self.source_id = None if source is None else source.source_id
        self.source_revision = None if source is None else source.source_revision
        self.domain_id = None if domain is None else domain.domain_id
        self.request_id = _request_identity(self)

    def validate_source_integrity(self) -> None:
        """Rebuild owned source facts; stored fingerprints alone are not proof."""
        from ._certification import MeshCertificationSchedule

        schedule = MeshCertificationSchedule(self.schedule.route)
        limits = MeshCertificateLimits(
            maximum_candidate_pairs=self.limits.maximum_candidate_pairs,
            maximum_ray_tests=self.limits.maximum_ray_tests,
            maximum_source_samples=self.limits.maximum_source_samples,
            maximum_distance_evaluations=self.limits.maximum_distance_evaluations,
            maximum_subdivision_depth=self.limits.maximum_subdivision_depth,
            maximum_subdivision_pieces=self.limits.maximum_subdivision_pieces,
            maximum_bernstein_nodes=self.limits.maximum_bernstein_nodes,
            maximum_periodic_images=self.limits.maximum_periodic_images,
            maximum_work_units=self.limits.maximum_work_units,
            maximum_scratch_bytes=self.limits.maximum_scratch_bytes,
        )
        if not bool(eqx.tree_equal(self.schedule, schedule, typematch=True)) or not bool(
            eqx.tree_equal(self.limits, limits, typematch=True)
        ):
            raise ValueError(
                "Retained certification policy differs from its owning request."
            )
        if _request_identity(self) != self.request_id:
            raise ValueError(
                "Retained certification numerical assignments or bindings changed."
            )
        domain = self.domain
        if domain is not None:
            _validate_domain_integrity(domain)
        source = self.source
        if source is not None:
            _validate_query_integrity(source)
            if (
                source.source_id != self.source_id
                or source.source_revision != self.source_revision
            ):
                raise ValueError("Retained certification source identity changed.")
        if domain is not None and domain.domain_id != self.domain_id:
            raise ValueError("Retained certification domain identity changed.")
        for query, _, _ in self.scoped_fidelity:
            _validate_query_integrity(query)

    def _transition_request(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        /,
        *,
        lineage: MeshLineage,
        cell_regions: ArrayLike | None = None,
        junction_vertices: ArrayLike | None = None,
    ) -> MeshCertificationInputs:
        """Own complete target assignments and every unchanged source obligation."""
        from ._lineage import MeshLineage

        self.validate_source_integrity()

        if not isinstance(lineage, MeshLineage):
            raise TypeError("lineage must be MeshLineage.")
        source_sets = dict(self.entity_set_ids)
        for relation in lineage.entities:
            if (
                relation.source_entity_set_id != source_sets.get(relation.dimension)
                or relation.target_entity_set_id
                != mesh.entity_set(relation.dimension).entity_set_id
            ):
                raise ValueError(
                    "Certification lineage must bind actual source and target entity sets."
                )
        if (
            lineage.source_topology_id != self.topology_id
            or lineage.target_topology_id != mesh.topology_id
        ):
            raise ValueError(
                "Certification renewal lineage must bind source and target topology."
            )
        if self.source is not None and (
            self.source.source_id != self.source_id
            or self.source.source_revision != self.source_revision
        ):
            raise ValueError("Certification source revision changed after preparation.")
        regions = cell_regions
        if self.cell_regions is not None and regions is None:
            relations = lineage.entity_lineage(mesh.topological_dimension)
            old = dict(
                zip(
                    self.cell_global_ids,
                    np.asarray(self.cell_regions, dtype=np.int64).tolist(),
                    strict=True,
                )
            )
            target: dict[int, set[int]] = {}
            for source_id, target_id in zip(
                np.asarray(relations.source_global_ids).tolist(),
                np.asarray(relations.target_global_ids).tolist(),
                strict=True,
            ):
                if source_id not in old:
                    raise ValueError(
                        "Region renewal ancestry addresses an unknown source cell."
                    )
                target.setdefault(target_id, set()).add(old[source_id])
            identifiers = np.concatenate(
                [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
            )
            if any(
                identifier not in target or len(target[identifier]) != 1
                for identifier in identifiers.tolist()
            ):
                raise ValueError(
                    "New or cross-material cells need explicit independently revalidated region assignments."
                )
            regions = np.asarray(
                [next(iter(target[identifier])) for identifier in identifiers.tolist()],
                dtype=np.int64,
            )
        junctions = junction_vertices
        if self.junction_vertices is not None and junctions is None:
            relations = lineage.entity_lineage(0)
            old_junctions = set(
                np.asarray(self.junction_vertices, dtype=np.int64).tolist()
            )
            from ._lineage import EntityLineageKind

            identity_kinds = {
                int(EntityLineageKind.PRESERVED),
                int(EntityLineageKind.RELOCATED),
                int(EntityLineageKind.COLLAPSED_INTO),
                int(EntityLineageKind.MERGED_INTO),
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
                    "Every retained junction needs an identity-preserving target relation or explicit revalidated junctions."
                )
            junctions = np.asarray(
                sorted({target for _, target in mapped}), dtype=np.int64
            )
        from ..geometry._mesh_certificates import MappedDomainBoundarySource

        source = self.source
        if isinstance(source, MappedDomainBoundarySource):
            if regions is None:
                raise ValueError(
                    "Mapped-source renewal requires actual target region assignments."
                )
            source = source.for_target_regions(regions)
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
                        "Scoped fidelity renewal lost an original target trace facet."
                    )
                for before, after in zip(sources.tolist(), targets.tolist(), strict=True):
                    if before in selected:
                        memberships.setdefault(after, set()).add(index)
            for index, (query, _, tolerance) in enumerate(self.scoped_fidelity):
                identifiers = np.asarray(
                    sorted(
                        target
                        for target, classes in memberships.items()
                        if index in classes
                    ),
                    dtype=np.int64,
                )
                if identifiers.size == 0:
                    raise ValueError(
                        "Scoped fidelity renewal lost a complete source stratum."
                    )
                for facet in identifiers.tolist():
                    ancestors = set(sources[targets == facet].tolist())
                    selected = set(
                        np.asarray(
                            self.scoped_fidelity[index][1], dtype=np.int64
                        ).tolist()
                    )
                    if not ancestors <= selected:
                        raise ValueError(
                            "Scoped fidelity cannot coarsen across different original source strata."
                        )
                scoped.append((query, identifiers, tolerance))
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
        )

    def recertify_transition(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        audit: CellMeshAuditReport,
        /,
        *,
        lineage: MeshLineage,
        cell_regions: ArrayLike | None = None,
        junction_vertices: ArrayLike | None = None,
        prepared: MeshCertificationPreparedEvidence | None = None,
        prepared_embedding: GlobalEmbeddingCertificate | None = None,
        prepared_fidelity: SourceFidelityCertificate | None = None,
    ) -> MeshCertificationReport:
        """Renew acceptance or consume exactly governed current target premises."""
        from ._certification import certify_meshing_acceptance

        request = self._transition_request(
            mesh,
            geometry,
            lineage=lineage,
            cell_regions=cell_regions,
            junction_vertices=junction_vertices,
        )
        return certify_meshing_acceptance(
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
            prepared_fidelity=prepared_fidelity,
            prepared_embedding=prepared_embedding,
        )


def _immutable_host_leaf(value: object) -> object:
    if isinstance(value, np.ndarray):
        snapshot = np.array(value, copy=True)
        snapshot.setflags(write=False)
        return snapshot
    return value


def _validate_domain_integrity(
    domain: PiecewiseLinearDomain | MappedReferenceDomain,
) -> None:
    if isinstance(domain, PiecewiseLinearDomain):
        rebuilt = PiecewiseLinearDomain(
            domain.vertices,
            domain.facets,
            domain.facet_regions,
            domain.region_ids,
            source_id=domain.source_id,
        )
    else:
        _validate_domain_integrity(domain.reference_domain)
        rebuilt = MappedReferenceDomain(
            domain.reference_domain,
            domain.reference_mesh,
            domain.source_geometry,
            domain.cell_regions,
            source_id=domain.source_id,
            source_revision=domain.source_revision,
        )
    if rebuilt.domain_id != domain.domain_id or not bool(
        eqx.tree_equal(domain, rebuilt, typematch=True)
    ):
        raise ValueError(
            "Declared source domain differs from its restored numerical facts."
        )


def _validate_query_integrity(source: SourceBoundaryQuery) -> None:
    from ..geometry._certified_implicit import implicit_state_id
    from ..geometry._mesh_certificates import (
        ImplicitBoundarySource,
        ImplicitProjectionBoundarySource,
        MappedDomainBoundarySource,
        ParametricCurveBoundarySource,
    )
    from ..geometry._meshing_domain import MeshingDomainBoundarySource
    from ..geometry.implicit._analytic_profile import AnalyticImplicitProfile
    from ._surface_association_transfer import (
        SurfaceChartBoundarySource,
        SurfaceSubdivisionBoundarySource,
    )

    if isinstance(source, ImplicitBoundarySource):
        if implicit_state_id(source.geometry) != source.source_revision:
            raise ValueError("Implicit source state differs from its stored revision.")
        rebuilt = ImplicitBoundarySource(
            source.geometry, source_id=source.source_id, spacing=source.spacing
        )
    elif isinstance(source, AnalyticImplicitProfile):
        source.require_bound(source.geometry, source.coordinate_contract)
        rebuilt = AnalyticImplicitProfile(
            source.geometry,
            source.coordinate_contract,
            tube_radius=source.tube_radius,
            cover_radius=source.cover_radius,
            source_id=source.source_id,
            source_revision=source.source_revision,
        )
    elif isinstance(source, ImplicitProjectionBoundarySource):
        _validate_query_integrity(source.profile)
        rebuilt = ImplicitProjectionBoundarySource(source.profile)
    elif isinstance(source, MappedDomainBoundarySource):
        domain = source.domain
        if not isinstance(domain, MappedReferenceDomain):
            raise TypeError(
                "Mapped boundary restoration requires its actual owning declared domain."
            )
        _validate_domain_integrity(domain)
        rebuilt = MappedDomainBoundarySource(
            domain,
            source.cell_regions,
            limits=source.limits,
            covering_radius=source.covering_radius,
        )
    elif isinstance(source, ParametricCurveBoundarySource):
        rebuilt = ParametricCurveBoundarySource(
            source.curves,
            source.parameter_ranges,
            source_id=source.source_id,
            source_revision=source.source_revision,
            covering_radius=source.covering_radius,
            maximum_samples=source.maximum_samples,
        )
    elif isinstance(source, MeshingDomainBoundarySource):
        source.domain.require_current(source.source_id, source.source_revision)
        rebuilt = MeshingDomainBoundarySource(
            source.domain,
            source.patches,
            resolution=source.resolution,
            chart_triangulations=source.chart_triangulations,
        )
    elif isinstance(
        source, (SurfaceSubdivisionBoundarySource, SurfaceChartBoundarySource)
    ):
        # The retained accepted root carries NaN quality leaves, so generic
        # tree equality cannot compare it; the owner re-runs its bindings.
        _validate_query_integrity(source.root)
        source.require_current()
        return
    else:
        raise TypeError(
            "Source query type has no owning restoration integrity validator."
        )
    if not bool(eqx.tree_equal(source, rebuilt, typematch=True)):
        raise ValueError("Source query differs from its rebuilt numerical facts.")


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
            "cell_ids": inputs.cell_global_ids,
            "junctions": None
            if inputs.junction_vertices is None
            else array_tree_fingerprint(inputs.junction_vertices),
            "tolerance": inputs.fidelity_tolerance,
            "sample_order": inputs.fidelity_sample_order,
            "limits": inputs.limits.limits_id,
            "scoped_fidelity": tuple(
                (query.source_scope_id, array_tree_fingerprint(identifiers), tolerance)
                for query, identifiers, tolerance in inputs.scoped_fidelity
            ),
        }
    )
