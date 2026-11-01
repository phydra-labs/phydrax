#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Compose native mesh acceptance from actual source and coordinate operands."""

from __future__ import annotations

from typing import assert_never, final

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellGeometrySpec, CellMesh
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    certify_cell_geometry_validity,
)
from ..discretization._surface_chart_records import SurfaceChartWitness
from ..geometry._domain_certification import (
    certify_domain_coverage,
    coverage_region_occurrence_paths,
    DomainCoverageCertificate,
)
from ..geometry._domain_coverage import PiecewiseLinearDomain
from ..geometry._embedding_state import MeshCertificateFinding
from ..geometry._global_embedding import (
    _junction_ids,
    certify_global_embedding,
    GlobalEmbeddingCertificate,
)
from ..geometry._mesh_certificates import (
    _same_tree_content,
    certificate_status,
    MeshCertificateStatus,
)
from ..geometry._nested_restriction_embedding import NestedRestrictionPremise
from ..geometry._source_boundary_records import (
    SourceBoundaryChartWitness,
    SourceBoundaryDistanceWitness,
    SourceBoundaryPointWitness,
    SourceBoundaryQuery,
    SurfaceEnvelopeDistanceWitness,
)
from ..geometry._source_fidelity import (
    certify_source_fidelity,
    validate_source_entity_scope,
    ValidatedSourceFidelityWitness,
)
from ..geometry._source_fidelity_certificate import (
    SourceFidelityCertificate,
    SourceFidelityWitness,
)
from ..geometry.surface import SurfaceModel
from ._association import GeometryAssociation
from ._audit import (
    audit_cell_mesh,
    CellMeshAuditDisposition,
    CellMeshAuditPolicy,
    CellMeshAuditReport,
)
from ._certification_inputs import MeshCertificationInputs, validated_fidelity_witness
from ._contracts import MeshingFailure, MeshingFailureCategory
from ._native_construction import NativeMeshOrganization
from ._organization import MeshAttribute, MeshLabel, MeshPatch, MeshZone


def _require_route_geometry(inputs: MeshCertificationInputs, /) -> None:
    mesh = inputs.mesh
    dimensions = (mesh.topological_dimension, mesh.ambient_dimension)
    route = inputs.schedule.route
    match route:
        case (
            "volume_plc"
            | "volume_implicit"
            | "volume_layer"
            | "volume_cad"
            | "mapped_volume"
        ):
            supported = dimensions == (3, 3)
        case "surface":
            supported = dimensions == (2, 3)
        case "planar":
            supported = dimensions == (2, 2)
        case "curve":
            supported = dimensions[0] == 1 and dimensions[1] in (1, 2, 3)
        case "periodic":
            supported = (
                dimensions in ((2, 2), (3, 3)) and mesh.periodic_topology is not None
            )
        case _:
            assert_never(route)
    if not supported:
        raise ValueError(
            "The mesh dimensions and quotient topology must match its certification route."
        )


def _declared_junction_findings(
    inputs: MeshCertificationInputs, /
) -> tuple[MeshCertificateFinding, ...]:
    if inputs.junction_vertices is None or inputs.mesh.topological_dimension != 1:
        return ()
    mesh = inputs.mesh
    rows = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64).reshape(-1) for block in mesh.blocks]
    )
    valence = np.bincount(rows, minlength=mesh.coordinates.shape[0])
    identifiers = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    undeclared = identifiers[
        (valence > 2) & ~np.isin(identifiers, inputs.junction_vertices)
    ]
    if not undeclared.size:
        return ()
    return (
        MeshCertificateFinding(
            "undeclared_curve_junction",
            "violated",
            "vertex",
            tuple(int(value) for value in undeclared),
            detail="A curve junction joins more than two intervals without a source declaration.",
        ),
    )


def _acceptance_audit_policy(inputs: MeshCertificationInputs, /) -> CellMeshAuditPolicy:
    disposition = CellMeshAuditDisposition
    declared_curve = (
        inputs.mesh.topological_dimension == 1 and inputs.junction_vertices is not None
    )
    return CellMeshAuditPolicy(
        nonmanifold=disposition.RECORD if declared_curve else disposition.REJECT,
        watertight_boundary=(
            disposition.REJECT
            if "open_boundary" in inputs.schedule.required_audit_checks
            else disposition.SKIP
        ),
    )


def _audit_findings(
    inputs: MeshCertificationInputs, audit: CellMeshAuditReport, /
) -> tuple[MeshCertificateFinding, ...]:
    missing = tuple(
        name
        for name in inputs.schedule.required_audit_checks
        if name not in audit.evaluated_checks
    )
    if missing:
        raise ValueError(
            f"Native acceptance requires evaluated audit checks: {missing!r}."
        )
    junctions = _declared_junction_findings(inputs)
    excused = (
        ("nonmanifold_facets", "nonmanifold_vertices")
        if inputs.mesh.topological_dimension == 1
        and inputs.junction_vertices is not None
        and not junctions
        else ()
    )
    findings = list(junctions)
    required = set(inputs.schedule.required_audit_checks)
    rejected = audit.issues
    counted = tuple(
        name
        for name, count in audit.check_counts
        if count and name in required and name not in excused
    )
    for name in dict.fromkeys((*rejected, *counted)):
        status = "unresolved" if name.startswith("unresolved_") else "violated"
        findings.append(MeshCertificateFinding(name, status, "mesh"))
    for name in audit.unresolved:
        findings.append(MeshCertificateFinding(name, "unresolved", "mesh"))
    return tuple(findings)


def _coverage_operands(
    inputs: MeshCertificationInputs,
    fidelity: SourceFidelityCertificate | None,
    /,
) -> tuple[SourceBoundaryQuery | None, SourceFidelityCertificate | None]:
    """Return the source query and fidelity proof that material coverage consumes."""
    if (
        isinstance(inputs.domain, PiecewiseLinearDomain)
        and inputs.source is not None
        and (inputs.source.source_id, inputs.source.source_revision)
        != (inputs.domain.source_id, inputs.domain.source_revision)
    ):
        # A derived affine domain carries its own content revision. Its
        # coverage is an exact chain comparison that needs no continuous
        # source query; fidelity is certified separately.
        return None, None
    return inputs.source, fidelity


def _feature_witness_scope(
    witness: SourceFidelityWitness, /
) -> tuple[int, set[tuple[str, tuple[str, ...]]]] | None:
    """Interpret actual entity correspondence, never a distance-only bound."""
    match witness:
        case SourceBoundaryChartWitness():
            dimension = witness.chart_dimension
            identifiers = witness.source_entity_ids
        case SourceBoundaryPointWitness():
            dimension = 0
            identifiers = witness.source_entity_ids
        case SurfaceChartWitness():
            dimension = 2
            identifiers = witness.geometry_entity_ids
        case SurfaceEnvelopeDistanceWitness() | SourceBoundaryDistanceWitness():
            return None
        case _:
            assert_never(witness)
    return dimension, set(zip(identifiers, witness.occurrence_paths, strict=True))


def _feature_findings(
    inputs: MeshCertificationInputs,
    certificates: tuple[SourceFidelityCertificate, ...],
    /,
) -> tuple[tuple[MeshCertificateFinding, ...], tuple[MeshCertificateFinding, ...]]:
    failures: list[MeshCertificateFinding] = []
    recorded: list[MeshCertificateFinding] = []
    for feature in inputs.protected_features or ():
        owner = inputs.feature_source
        if owner is None:
            raise ValueError(
                "Protected feature certification requires its source entity owner."
            )
        scope = feature.scope
        identifiers = np.asarray(scope.entity_ids, dtype=np.int64)
        expected = {
            owner.source_entity_key_for_scope(scope.entity_dimension, int(identifier))
            for identifier in identifiers
        }
        matches: list[SourceFidelityCertificate] = []
        for certificate, (_, witness, _) in zip(
            certificates, inputs.scoped_fidelity, strict=True
        ):
            actual = _feature_witness_scope(witness)
            if actual == (scope.entity_dimension, expected):
                validate_source_entity_scope(
                    certificate.source, owner, scope.entity_dimension, identifiers
                )
                matches.append(certificate)
        if not matches:
            finding = MeshCertificateFinding(
                "protected_feature_coverage",
                "unresolved",
                "source_entity",
                tuple(int(value) for value in identifiers),
                detail="No continuous scoped witness covers the complete authored feature.",
            )
        elif any(
            certificate.status == "certified"
            and max(certificate.mesh_to_source_upper, certificate.source_to_mesh_upper)
            <= feature.maximum_deviation
            for certificate in matches
        ):
            continue
        else:
            status = (
                "violated"
                if any(
                    max(
                        certificate.mesh_to_source_lower, certificate.source_to_mesh_lower
                    )
                    > feature.maximum_deviation
                    for certificate in matches
                )
                else "unresolved"
            )
            finding = MeshCertificateFinding(
                "protected_feature_deviation",
                status,
                "source_entity",
                tuple(int(value) for value in identifiers),
                detail="Continuous source feature deviation does not establish the requested bound.",
            )
        (failures if feature.hard else recorded).append(finding)
    return tuple(failures), tuple(recorded)


@final
class MeshCertificationReport(StrictModule, NonTrainableState):
    """Computed acceptance, retaining its source premises for independent renewal.

    Audit flags supplied by a caller are not premises. Construction reruns the
    audit and composes embedding, source fidelity, material coverage, and feature
    checks. Unsupported mathematical proofs remain explicit unresolved findings.
    """

    inputs: MeshCertificationInputs
    audit_policy: CellMeshAuditPolicy
    boundary: SurfaceModel | None
    patches: tuple[MeshPatch, ...]
    zones: tuple[MeshZone, ...]
    labels: tuple[MeshLabel, ...]
    attributes: tuple[MeshAttribute, ...]
    associations: tuple[GeometryAssociation, ...]
    audit: CellMeshAuditReport
    embedding: GlobalEmbeddingCertificate
    coverage: DomainCoverageCertificate | None
    fidelity: SourceFidelityCertificate | None
    scoped_fidelity: tuple[SourceFidelityCertificate, ...]
    findings: tuple[MeshCertificateFinding, ...]
    recorded: tuple[MeshCertificateFinding, ...]
    status: MeshCertificateStatus = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        inputs: MeshCertificationInputs,
        /,
        *,
        audit_policy: CellMeshAuditPolicy | None = None,
        organization: NativeMeshOrganization | None = None,
        embedding: GlobalEmbeddingCertificate | None = None,
        embedding_premise: NestedRestrictionPremise | None = None,
        validated: tuple[ValidatedSourceFidelityWitness, ...] = (),
    ) -> None:
        """Compute acceptance from ``inputs``.

        ``embedding`` is an optional prepared global-embedding certificate. It
        is reused only after it is shown to bind these exact mesh, coordinate
        map, local-validity and junction inputs; otherwise it is computed here,
        from ``embedding_premise`` (a certified source embedding and the exact
        nested transition to this mesh) when that theorem's premises hold.
        ``validated`` holds fidelity-witness validations already made for the
        retained mesh, coordinate map and source objects of ``inputs``, as
        returned by ``inputs.require_current``; those are not repeated.
        """
        if not isinstance(inputs, MeshCertificationInputs):
            raise TypeError("inputs must be MeshCertificationInputs.")
        if organization is not None and not isinstance(
            organization, NativeMeshOrganization
        ):
            raise TypeError("organization must be NativeMeshOrganization or None.")
        if embedding is not None and not isinstance(
            embedding, GlobalEmbeddingCertificate
        ):
            raise TypeError("embedding must be GlobalEmbeddingCertificate or None.")
        _require_route_geometry(inputs)
        organization_ = NativeMeshOrganization() if organization is None else organization
        policy = (
            _acceptance_audit_policy(inputs) if audit_policy is None else audit_policy
        )
        validated_ = inputs.require_current(
            inputs.mesh,
            inputs.geometry,
            domain=inputs.domain,
            source=inputs.source,
            validated=validated,
        )
        validity = certify_cell_geometry_validity(
            inputs.geometry,
            mesh=inputs.mesh,
            policy=policy.validity_policy,
        )
        if embedding is None:
            embedding = certify_global_embedding(
                inputs.mesh,
                inputs.geometry,
                validity,
                junction_vertices=inputs.junction_vertices,
                nested_restriction=embedding_premise,
            )
        else:
            embedding.require_inputs(
                inputs.mesh,
                inputs.geometry,
                validity,
                junction_vertices=inputs.junction_vertices,
            )
        audit = audit_cell_mesh(
            inputs.mesh,
            inputs.geometry,
            policy=policy,
            boundary=organization_.boundary,
            patches=organization_.patches,
            zones=organization_.zones,
            labels=organization_.labels,
            attributes=organization_.attributes,
            associations=organization_.associations,
            validity=validity,
            global_embedding=embedding,
        )
        findings = list(_audit_findings(inputs, audit))
        findings.extend(embedding.findings)
        fidelity = None
        if inputs.source is not None:
            if inputs.fidelity_tolerance is None:
                raise ValueError("Fidelity source requires its physical tolerance.")
            fidelity = certify_source_fidelity(
                inputs.mesh,
                inputs.geometry,
                inputs.source,
                tolerance=inputs.fidelity_tolerance,
                maximum_normal_deviation=inputs.maximum_normal_deviation,
                witness=None
                if inputs.fidelity_witness is None
                else validated_fidelity_witness(
                    validated_,
                    inputs.mesh,
                    inputs.geometry,
                    inputs.source,
                    inputs.fidelity_witness,
                ),
            )
            findings.extend(fidelity.findings)
        scoped = tuple(
            certify_source_fidelity(
                inputs.mesh,
                inputs.geometry,
                query,
                tolerance=tolerance,
                witness=validated_fidelity_witness(
                    validated_, inputs.mesh, inputs.geometry, query, witness
                ),
            )
            for query, witness, tolerance in inputs.scoped_fidelity
        )
        for certificate in scoped:
            findings.extend(certificate.findings)
        coverage = None
        if inputs.domain is not None:
            if inputs.cell_regions is None:
                raise ValueError("Domain coverage requires its material assignments.")
            if embedding.status == "certified":
                coverage_source, coverage_fidelity = _coverage_operands(inputs, fidelity)
                coverage = certify_domain_coverage(
                    inputs.mesh,
                    inputs.geometry,
                    inputs.domain,
                    inputs.cell_regions,
                    embedding=embedding,
                    source=coverage_source,
                    fidelity=coverage_fidelity,
                    source_coverage_witness=inputs.source_coverage_witness,
                )
                findings.extend(coverage.findings)
            else:
                findings.append(
                    MeshCertificateFinding(
                        "domain_embedding_premise",
                        "unresolved",
                        "mesh",
                        detail="Material coverage requires an established global target embedding.",
                    )
                )
        feature_findings, recorded = _feature_findings(inputs, scoped)
        findings.extend(feature_findings)
        self.inputs = inputs
        self.audit_policy = policy
        self.boundary = organization_.boundary
        self.patches = organization_.patches
        self.zones = organization_.zones
        self.labels = organization_.labels
        self.attributes = organization_.attributes
        self.associations = organization_.associations
        self.audit = audit
        self.embedding = embedding
        self.coverage = coverage
        self.fidelity = fidelity
        self.scoped_fidelity = scoped
        self.findings = tuple(
            sorted(
                {value.finding_id: value for value in findings}.values(),
                key=lambda value: value.finding_id,
            )
        )
        self.recorded = recorded
        self.status = certificate_status(self.findings)
        self.report_id = self._identity()

    def _identity(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "mesh-certification-report",
                "inputs": self.inputs.request_id,
                "audit": self.audit.report_id,
                "embedding": self.embedding.certificate_id,
                "coverage": None
                if self.coverage is None
                else self.coverage.certificate_id,
                "fidelity": None
                if self.fidelity is None
                else self.fidelity.certificate_id,
                "scoped_fidelity": tuple(
                    value.certificate_id for value in self.scoped_fidelity
                ),
                "findings": tuple(value.finding_id for value in self.findings),
                "recorded": tuple(value.finding_id for value in self.recorded),
                "status": self.status,
            }
        )

    @property
    def passed(self) -> bool:
        return self.status == "certified"

    def require_passed(self) -> None:
        if (
            self.status != certificate_status(self.findings)
            or self.report_id != self._identity()
        ):
            raise ValueError(
                "Mesh acceptance status and identity must match its retained evidence."
            )
        if not self.passed:
            raise MeshingFailure(
                MeshingFailureCategory.AUDIT_FAILED,
                "; ".join(
                    f"{value.check}: {value.status}"
                    + (f": {value.detail}" if value.detail else "")
                    for value in self.findings
                ),
                stage="mesh_certification",
                entity_ids=tuple(
                    sorted(
                        {
                            identifier
                            for value in self.findings
                            for identifier in value.entity_ids
                        }
                    )
                ),
            )

    def require_premises(
        self,
        inputs: MeshCertificationInputs,
        /,
        *,
        audit_policy: CellMeshAuditPolicy,
        organization: NativeMeshOrganization,
    ) -> None:
        """Refuse a stored report whose identity-bound claims differ from its premises.

        A restore calls this before replaying any proof: every compared value is
        a content identity determined by ``inputs`` and ``organization`` alone
        (request, mesh and map bindings, junctions, source revisions, coverage
        region IDs and occurrence paths, material assignment, organization IDs,
        report identity). Passing accepts nothing; acceptance is recomputed
        afterwards from the same premises.
        """
        if not isinstance(inputs, MeshCertificationInputs):
            raise TypeError("inputs must be MeshCertificationInputs.")
        if not isinstance(audit_policy, CellMeshAuditPolicy):
            raise TypeError("audit_policy must be CellMeshAuditPolicy.")
        if not isinstance(organization, NativeMeshOrganization):
            raise TypeError("organization must be NativeMeshOrganization.")
        if (
            self.status != certificate_status(self.findings)
            or self.report_id != self._identity()
        ):
            raise ValueError(
                "Stored mesh acceptance status and identity differ from its retained evidence."
            )
        if self.inputs.request_id != inputs.request_id or not _same_tree_content(
            self.audit_policy, audit_policy
        ):
            raise ValueError(
                "Stored mesh acceptance differs from its rebuilt request or audit policy."
            )
        stored = (
            None if self.boundary is None else self.boundary.model_id,
            tuple(value.patch_id for value in self.patches),
            tuple(value.zone_id for value in self.zones),
            tuple(value.label_id for value in self.labels),
            tuple(value.attribute_id for value in self.attributes),
            tuple(value.association_id for value in self.associations),
        )
        rebuilt = (
            None if organization.boundary is None else organization.boundary.model_id,
            tuple(value.patch_id for value in organization.patches),
            tuple(value.zone_id for value in organization.zones),
            tuple(value.label_id for value in organization.labels),
            tuple(value.attribute_id for value in organization.attributes),
            tuple(value.association_id for value in organization.associations),
        )
        if stored != rebuilt:
            raise ValueError(
                "Stored mesh acceptance differs from its rebuilt organization."
            )
        mesh, geometry = inputs.mesh, inputs.geometry
        binding = self.embedding.binding
        if (
            binding.mesh_id != mesh.mesh_id
            or binding.topology_id != mesh.topology_id
            or binding.geometry_layout_id != geometry.geometry_layout_id
            or binding.geometry_id != cell_geometry_id(geometry)
            or binding.junction_vertices != _junction_ids(mesh, inputs.junction_vertices)
        ):
            raise ValueError(
                "Stored global embedding binds another mesh, coordinate map, or junction set."
            )
        if (inputs.source is None) != (self.fidelity is None) or len(
            inputs.scoped_fidelity
        ) != len(self.scoped_fidelity):
            raise ValueError("Stored source fidelity differs from its rebuilt premises.")
        if (
            self.fidelity is not None
            and inputs.source is not None
            and (
                self.fidelity.binding.source_id,
                self.fidelity.binding.source_revision,
            )
            != (inputs.source.source_id, inputs.source.source_revision)
        ):
            raise ValueError("Stored source fidelity binds another source revision.")
        self._require_coverage_premises(inputs)

    def _require_coverage_premises(self, inputs: MeshCertificationInputs, /) -> None:
        coverage = self.coverage
        domain = inputs.domain
        if domain is None or self.embedding.status != "certified":
            if coverage is not None:
                raise ValueError(
                    "Stored material coverage has no admissible domain or embedding premise."
                )
            return
        if coverage is None or inputs.cell_regions is None:
            raise ValueError(
                "Stored mesh acceptance omitted its material coverage proof."
            )
        coverage_source, _ = _coverage_operands(inputs, self.fidelity)
        witness = inputs.source_coverage_witness
        stored_witness = coverage.source_coverage_witness
        if (
            coverage.domain_id != domain.domain_id
            or coverage.source_id != domain.source_id
            or coverage.region_ids != tuple(domain.region_ids)
            or coverage.region_occurrence_paths
            != coverage_region_occurrence_paths(domain)
            or coverage.embedding_certificate_id != self.embedding.certificate_id
            or (None if coverage.source is None else coverage.source.source_id)
            != (None if coverage_source is None else coverage_source.source_id)
            or (None if coverage.source is None else coverage.source.source_revision)
            != (None if coverage_source is None else coverage_source.source_revision)
            or (None if stored_witness is None else stored_witness.witness_id)
            != (None if witness is None else witness.witness_id)
        ):
            raise ValueError(
                "Stored material coverage differs from its rebuilt domain, regions, source, or witness."
            )
        regions = np.asarray(inputs.cell_regions)
        if regions.dtype.kind not in "iu" or not np.array_equal(
            regions, np.asarray(coverage.cell_regions)
        ):
            raise ValueError(
                "Stored material coverage binds another dense region assignment."
            )

    def require_current(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        /,
    ) -> tuple[ValidatedSourceFidelityWitness, ...]:
        """Require this computed acceptance to answer ``mesh`` and ``geometry``.

        Nothing is recomputed: the retained inputs must be current for these
        operands (their fidelity witnesses are validated, and the validations
        returned for reuse on the same operands), the report must have passed,
        and its status, identity and every identity-bound claim must match its
        retained premises (:meth:`require_premises`). :meth:`revalidate`
        recomputes the proofs instead.
        """
        validated = self.inputs.require_current(
            mesh, geometry, domain=self.inputs.domain, source=self.inputs.source
        )
        self.require_premises(
            self.inputs,
            audit_policy=self.audit_policy,
            organization=NativeMeshOrganization(
                boundary=self.boundary,
                patches=self.patches,
                zones=self.zones,
                labels=self.labels,
                attributes=self.attributes,
                associations=self.associations,
            ),
        )
        self.require_passed()
        return validated

    def revalidate(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        /,
        *,
        embedding: GlobalEmbeddingCertificate | None = None,
    ) -> None:
        """Recompute scientific acceptance; stored identities are not proof authority.

        ``embedding`` may supply an in-hand global-embedding certificate; it is
        reused only when it binds exactly the current inputs, and every other
        proof obligation is recomputed.
        """
        validated = self.inputs.require_current(
            mesh, geometry, domain=self.inputs.domain, source=self.inputs.source
        )
        renewed = MeshCertificationReport(
            self.inputs,
            audit_policy=self.audit_policy,
            organization=NativeMeshOrganization(
                boundary=self.boundary,
                patches=self.patches,
                zones=self.zones,
                labels=self.labels,
                attributes=self.attributes,
                associations=self.associations,
            ),
            embedding=embedding,
            embedding_premise=self.embedding.nested_restriction_premise,
            validated=validated,
        )
        if not _same_tree_content(self, renewed):
            raise ValueError(
                "Stored mesh acceptance does not reproduce its actual scientific inputs."
            )


__all__ = ["MeshCertificationReport"]
