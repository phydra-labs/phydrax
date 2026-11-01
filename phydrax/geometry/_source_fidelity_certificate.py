#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""The source-fidelity certificate record and its witness union.

This module is a dependency leaf: source models such as ``BRepModel`` accept
an in-hand certificate at checked boundaries, so the record type must resolve
at runtime without importing the proof machinery that consumes those models.
"""

from __future__ import annotations

import math
from typing import Any, final

import numpy as np
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import reconstruct_strict, register_strict_reconstruction, StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._cell_mesh import CellMesh
from ..discretization._surface_chart_records import SurfaceChartWitness
from ._embedding_state import MeshCertificateFinding
from ._mesh_certificates import (
    _same_tree_content,
    certificate_status,
    MeshCertificateBinding,
    MeshCertificateStatus,
)
from ._source_boundary_records import (
    SourceBoundaryChartCover,
    SourceBoundaryChartWitness,
    SourceBoundaryDistanceWitness,
    SourceBoundaryPointWitness,
    SourceBoundaryQuery,
    SurfaceEnvelopeDistanceWitness,
)
from .implicit._chart_restriction import (
    ImplicitChartPremise,
    ImplicitChartRestrictionEvidence,
    ImplicitChartRows,
)


type SourceFidelityWitness = (
    SurfaceChartWitness
    | SourceBoundaryChartWitness
    | SourceBoundaryPointWitness
    | SourceBoundaryDistanceWitness
    | SurfaceEnvelopeDistanceWitness
)


def _source_domain_id(source: SourceBoundaryQuery, /) -> str:
    from ._domain_boundary import MeshingDomainBoundarySource

    if isinstance(source, MeshingDomainBoundarySource):
        return source.domain.domain_id
    from ._surface_source_support import SurfaceNativeRestrictionBoundarySource

    if isinstance(source, SurfaceNativeRestrictionBoundarySource):
        return source.domain.domain_id
    from ._curve_boundary import ParametricCurveBoundarySource

    if isinstance(source, ParametricCurveBoundarySource):
        return source.query_id
    from ._mapped_reference_boundary import MappedReferenceBoundarySource

    if isinstance(source, MappedReferenceBoundarySource):
        return source.source_query_id
    from .reconstruction._boundary import ReconstructedTriangleBoundarySource

    if isinstance(source, ReconstructedTriangleBoundarySource):
        domain = source.source.meshing_domain
        return source.source_revision if domain is None else domain.domain_id
    from .implicit._analytic_profile import AnalyticImplicitProfile

    if isinstance(source, AnalyticImplicitProfile):
        return source.profile_id
    return source.source_revision


def _fidelity_certificate_identity(fields: dict[str, Any], /) -> str:
    values = tuple(float(fields[name]) for name in (
        "mesh_to_source_lower", "mesh_to_source_upper",
        "source_to_mesh_lower", "source_to_mesh_upper",
    ))
    witness, coverage = fields["witness"], fields["coverage"]
    normal = fields["normal_deviation_bounds"]
    identity = {
        "kind": "source-fidelity-certificate",
        "binding": fields["binding"].binding_id,
        "source": (fields["source_id"], fields["source_revision"], fields["domain_id"]),
        "tolerance": fields["tolerance"],
        "maximum_normal_deviation": fields["maximum_normal_deviation"],
        "deviation_bounds": array_tree_fingerprint(fields["deviation_bounds"]),
        "directed_bounds": tuple(value if math.isfinite(value) else value.hex() for value in values),
        "normal_bounds": None if normal is None else array_tree_fingerprint(normal),
        "normal_lower": fields["maximum_normal_deviation_lower"],
        "cells": fields["source_cell_global_ids"],
        "patches": fields["source_patch_ids"],
        "occurrences": fields["source_occurrence_paths"],
        "witness": None if witness is None else witness.witness_id,
        "coverage": None if coverage is None else coverage.cover_id,
        "findings": tuple(finding.finding_id for finding in fields["findings"]),
        "status": fields["status"],
    }
    restriction = fields["chart_restriction"]
    if restriction is not None:
        identity["chart_restriction"] = restriction.evidence_id
    return canonical_fingerprint(identity)


@final
class SourceFidelityCertificate(StrictModule, NonTrainableState):
    """Continuous two-directed bounds and optional oriented-normal evidence."""

    __strict_contract__ = True

    binding: MeshCertificateBinding
    source: SourceBoundaryQuery
    source_id: str
    source_revision: str
    domain_id: str
    mesh_id: str
    topology_id: str
    geometry_id: str
    tolerance: float
    maximum_normal_deviation: float | None
    deviation_bounds: np.ndarray
    mesh_to_source_lower: float
    mesh_to_source_upper: float
    source_to_mesh_lower: float
    source_to_mesh_upper: float
    normal_deviation_bounds: np.ndarray | None
    maximum_normal_deviation_lower: float | None
    maximum_normal_deviation_bound: float | None
    source_cell_global_ids: tuple[int, ...]
    source_patch_ids: tuple[int, ...]
    source_occurrence_paths: tuple[tuple[str, ...], ...]
    witness: SourceFidelityWitness | None
    coverage: SourceBoundaryChartCover | None
    status: MeshCertificateStatus
    findings: tuple[MeshCertificateFinding, ...]
    # Implicit root-chart evidence: every row's own proof, keyed by its proof
    # inputs (not part of the certificate identity; checked by its own); and
    # the restriction theorem with its premise when the bounds were inherited.
    chart_rows: ImplicitChartRows | None
    chart_restriction: ImplicitChartRestrictionEvidence | None
    chart_premise: ImplicitChartPremise | None
    certificate_id: str

    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        source: SourceBoundaryQuery,
        /,
        *,
        tolerance: float,
        maximum_normal_deviation: float | None = None,
        witness: SourceFidelityWitness | None = None,
    ) -> None:
        """Compute fidelity from scientific inputs, never from supplied claims."""
        from ._source_fidelity import certify_source_fidelity

        computed = certify_source_fidelity(
            mesh,
            geometry,
            source,
            tolerance=tolerance,
            maximum_normal_deviation=maximum_normal_deviation,
            witness=witness,
        )
        self.binding = computed.binding
        self.source = computed.source
        self.source_id = computed.source_id
        self.source_revision = computed.source_revision
        self.domain_id = computed.domain_id
        self.mesh_id = computed.mesh_id
        self.topology_id = computed.topology_id
        self.geometry_id = computed.geometry_id
        self.tolerance = computed.tolerance
        self.maximum_normal_deviation = computed.maximum_normal_deviation
        self.deviation_bounds = computed.deviation_bounds
        self.mesh_to_source_lower = computed.mesh_to_source_lower
        self.mesh_to_source_upper = computed.mesh_to_source_upper
        self.source_to_mesh_lower = computed.source_to_mesh_lower
        self.source_to_mesh_upper = computed.source_to_mesh_upper
        self.normal_deviation_bounds = computed.normal_deviation_bounds
        self.maximum_normal_deviation_lower = computed.maximum_normal_deviation_lower
        self.maximum_normal_deviation_bound = computed.maximum_normal_deviation_bound
        self.source_cell_global_ids = computed.source_cell_global_ids
        self.source_patch_ids = computed.source_patch_ids
        self.source_occurrence_paths = computed.source_occurrence_paths
        self.witness = computed.witness
        self.coverage = computed.coverage
        self.status = computed.status
        self.findings = computed.findings
        self.chart_rows = computed.chart_rows
        self.chart_restriction = computed.chart_restriction
        self.chart_premise = computed.chart_premise
        self.certificate_id = computed.certificate_id

    @classmethod
    def _from_computed(
        cls,
        binding: MeshCertificateBinding,
        source: SourceBoundaryQuery,
        tolerance: float,
        maximum_normal_deviation: float | None,
        deviation_bounds: ArrayLike,
        mesh_to_source_lower: float,
        mesh_to_source_upper: float,
        source_to_mesh_lower: float,
        source_to_mesh_upper: float,
        normal_deviation_bounds: ArrayLike | None,
        maximum_normal_deviation_lower: float | None,
        source_cell_global_ids: tuple[int, ...],
        source_patch_ids: tuple[int, ...],
        source_occurrence_paths: tuple[tuple[str, ...], ...],
        witness: SourceFidelityWitness | None,
        coverage: SourceBoundaryChartCover | None,
        findings: tuple[MeshCertificateFinding, ...],
        /,
        *,
        chart_rows: ImplicitChartRows | None = None,
        chart_restriction: ImplicitChartRestrictionEvidence | None = None,
        chart_premise: ImplicitChartPremise | None = None,
    ) -> SourceFidelityCertificate:
        if (chart_restriction is None) != (chart_premise is None):
            raise ValueError(
                "An inherited implicit-chart certificate needs its restriction "
                "evidence and premise together."
            )
        if chart_restriction is not None and chart_rows is not None:
            raise ValueError("Inherited implicit-chart bounds carry no proved rows.")
        bounds = np.asarray(deviation_bounds, dtype=np.float64)
        if (
            bounds.ndim != 2
            or bounds.shape[1] != 2
            or np.any(np.isnan(bounds))
            or np.any(bounds < 0)
        ):
            raise ValueError(
                "SourceFidelityCertificate.deviation_bounds must be nonnegative lower/upper pairs."
            )
        if np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError(
                "SourceFidelityCertificate deviation lower bounds exceed upper bounds."
            )
        bounds = np.array(bounds, copy=True)
        bounds.setflags(write=False)
        normal_bounds = None
        if normal_deviation_bounds is not None:
            normal_bounds = np.asarray(normal_deviation_bounds, dtype=np.float64)
            if (
                normal_bounds.ndim != 1
                or np.any(np.isnan(normal_bounds))
                or np.any(normal_bounds < 0)
            ):
                raise ValueError(
                    "SourceFidelityCertificate normal bounds must be nonnegative radians."
                )
            normal_bounds = np.array(normal_bounds, copy=True)
            normal_bounds.setflags(write=False)
        tolerance_ = float(tolerance)
        maximum_normal_ = (
            None if maximum_normal_deviation is None else float(maximum_normal_deviation)
        )
        if not math.isfinite(tolerance_) or tolerance_ < 0:
            raise ValueError("Source fidelity tolerance must be finite and nonnegative.")
        if maximum_normal_ is not None and (
            not math.isfinite(maximum_normal_)
            or maximum_normal_ < 0
            or maximum_normal_ > math.pi
        ):
            raise ValueError(
                "maximum_normal_deviation must be None or finite radians in [0, pi]."
            )
        values = tuple(
            float(value)
            for value in (
                mesh_to_source_lower,
                mesh_to_source_upper,
                source_to_mesh_lower,
                source_to_mesh_upper,
            )
        )
        if any(math.isnan(value) or value < 0 for value in values):
            raise ValueError(
                "Source fidelity directed bounds must be nonnegative and non-NaN."
            )
        if values[0] > values[1] or values[2] > values[3]:
            raise ValueError("Source fidelity directed lower bounds exceed upper bounds.")
        if (
            len(source_cell_global_ids) != bounds.shape[0]
            or len(source_patch_ids) != bounds.shape[0]
            or len(source_occurrence_paths) != bounds.shape[0]
        ):
            raise ValueError(
                "Source fidelity cell identities must align with per-cell bounds."
            )
        status = certificate_status(findings)
        attributes = {
            "binding": binding,
            "source": source,
            "source_id": binding.source_id or source.source_id,
            "source_revision": binding.source_revision or source.source_revision,
            "domain_id": _source_domain_id(source),
            "mesh_id": binding.mesh_id,
            "topology_id": binding.topology_id,
            "geometry_id": binding.geometry_id,
            "tolerance": tolerance_,
            "maximum_normal_deviation": maximum_normal_,
            "deviation_bounds": bounds,
            "mesh_to_source_lower": values[0],
            "mesh_to_source_upper": values[1],
            "source_to_mesh_lower": values[2],
            "source_to_mesh_upper": values[3],
            "normal_deviation_bounds": normal_bounds,
            "maximum_normal_deviation_lower": maximum_normal_deviation_lower,
            "maximum_normal_deviation_bound": (
                None
                if normal_bounds is None or normal_bounds.size == 0
                else float(np.max(normal_bounds))
            ),
            "source_cell_global_ids": tuple(
                int(value) for value in source_cell_global_ids
            ),
            "source_patch_ids": tuple(int(value) for value in source_patch_ids),
            "source_occurrence_paths": tuple(
                tuple(path) for path in source_occurrence_paths
            ),
            "witness": witness,
            "coverage": coverage,
            "status": status,
            "findings": tuple(findings),
            "chart_rows": chart_rows,
            "chart_restriction": chart_restriction,
            "chart_premise": chart_premise,
        }
        attributes["certificate_id"] = _fidelity_certificate_identity(attributes)
        return reconstruct_strict(cls, attributes)

    def _derived_certificate_id(self) -> str:
        """The content identity these stored claims determine."""
        return _fidelity_certificate_identity({
            "binding": self.binding,
            "source_id": self.source_id,
            "source_revision": self.source_revision,
            "domain_id": self.domain_id,
            "tolerance": self.tolerance,
            "maximum_normal_deviation": self.maximum_normal_deviation,
            "deviation_bounds": self.deviation_bounds,
            "mesh_to_source_lower": self.mesh_to_source_lower,
            "mesh_to_source_upper": self.mesh_to_source_upper,
            "source_to_mesh_lower": self.source_to_mesh_lower,
            "source_to_mesh_upper": self.source_to_mesh_upper,
            "normal_deviation_bounds": self.normal_deviation_bounds,
            "maximum_normal_deviation_lower": self.maximum_normal_deviation_lower,
            "source_cell_global_ids": self.source_cell_global_ids,
            "source_patch_ids": self.source_patch_ids,
            "source_occurrence_paths": self.source_occurrence_paths,
            "witness": self.witness,
            "coverage": self.coverage,
            "findings": self.findings,
            "status": self.status,
            "chart_restriction": self.chart_restriction,
        })

    def require_current(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        source: SourceBoundaryQuery,
        /,
    ) -> None:
        """Require this computed proof to answer the current mesh, map and source.

        Nothing is recomputed: the binding must match the current operands, the
        operands must be the content this proof was computed from
        (:meth:`require_inputs` with the retained tolerance, normal limit and
        witness), and the stored status and identity must be the ones the
        stored claims determine. :meth:`revalidate` recomputes the proof.
        """
        self.binding.require(mesh, geometry, source=source)
        self.require_inputs(
            mesh,
            geometry,
            source,
            tolerance=self.tolerance,
            maximum_normal_deviation=self.maximum_normal_deviation,
            witness=self.witness,
        )
        if (
            self.status != certificate_status(self.findings)
            or self.certificate_id != self._derived_certificate_id()
        ):
            raise ValueError(
                "Source fidelity certificate claims differ from its stored identity."
            )
        if self.chart_rows is not None:
            self.chart_rows.require_current()

    def revalidate(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        source: SourceBoundaryQuery,
        /,
    ) -> None:
        """Recompute the proof; stored identities are not proof authority."""
        from ._source_fidelity import certify_source_fidelity

        self.require_current(mesh, geometry, source)
        if self.chart_premise is not None:
            premise = self.chart_premise
            if premise.transition is not None:
                type(premise.transition).from_stored_fields(premise.transition)
        fresh = certify_source_fidelity(
            mesh,
            geometry,
            source,
            tolerance=self.tolerance,
            maximum_normal_deviation=self.maximum_normal_deviation,
            witness=self.witness,
            chart_parent=self.chart_premise,
        )
        if (
            fresh.certificate_id != self.certificate_id
            or not _same_tree_content(fresh.chart_rows, self.chart_rows)
            or not _same_tree_content(fresh.chart_restriction, self.chart_restriction)
        ):
            raise ValueError(
                "Source fidelity proof does not reproduce its current source and map inputs."
            )

    def require_inputs(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        source: SourceBoundaryQuery,
        /,
        *,
        tolerance: float,
        maximum_normal_deviation: float | None,
        witness: SourceFidelityWitness | None,
    ) -> None:
        """Require this computed proof to be the one these inputs determine.

        The certificate is a deterministic function of the mesh, coordinate
        map, source content, witness, tolerance and normal limit. When every
        content identity equals the bound one, a consumer may reuse this
        in-hand certificate instead of recomputing it; nothing is looked up.
        A meshing-domain source is compared by its content domain identity
        and selected patches: a domain rebuilt from an authority that now
        retains this derived tessellation names the same source content.
        """
        from ._domain_boundary import MeshingDomainBoundarySource

        binding = self.binding
        if (
            not _same_tree_content(mesh, binding.mesh)
            or not _same_tree_content(geometry, binding.geometry)
            or mesh.mesh_id != binding.mesh_id
            or mesh.topology_id != binding.topology_id
            or geometry.geometry_layout_id != binding.geometry_layout_id
            or cell_geometry_id(geometry) != binding.geometry_id
        ):
            raise ValueError(
                "Source fidelity certificate binds another mesh or coordinate map."
            )
        if isinstance(source, MeshingDomainBoundarySource) and isinstance(
            self.source, MeshingDomainBoundarySource
        ):
            # The query keeps its domain validation: no fingerprint for its own domain.
            source.require_current()
            same_source = (
                source.domain.domain_id == self.domain_id
                and source.patches == self.source.patches
                and source.source_id == self.source_id
                and source.source_revision == self.source_revision
            )
        else:
            same_source = _same_tree_content(source, self.source)
        if not same_source:
            raise ValueError("Source fidelity certificate binds another source.")
        normal_limit = (
            None if maximum_normal_deviation is None else float(maximum_normal_deviation)
        )
        if (
            float(tolerance) != self.tolerance
            or normal_limit != self.maximum_normal_deviation
        ):
            raise ValueError(
                "Source fidelity certificate answers another tolerance or normal limit."
            )
        if (None if witness is None else witness.witness_id) != (
            None if self.witness is None else self.witness.witness_id
        ):
            raise ValueError("Source fidelity certificate binds another witness.")


register_strict_reconstruction(SourceFidelityCertificate)


__all__ = ["SourceFidelityCertificate", "SourceFidelityWitness"]
