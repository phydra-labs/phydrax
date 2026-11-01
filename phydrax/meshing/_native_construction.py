#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Transient host construction data awaiting canonical scientific acceptance.

These records are not executable plans, persisted results, or compiled numerical
state. A route supplies its actual coordinate geometry and source premises;
canonical acceptance, not construction of these records, establishes validity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import numpy as np
    from jax.typing import ArrayLike

    from .._physical import SpatialCoordinateContract
    from ..discretization import CellGeometrySpec, CellMesh
    from ..discretization._cell_geometry_validity import CellValidityCertificate
    from ..discretization._nested_reference_types import ReferenceCellMapWitnesses
    from ..geometry._domain_certification import CoverageDomain, SourceCoverageWitness
    from ..geometry._global_embedding import GlobalEmbeddingCertificate
    from ..geometry._planar_embedding import PlanarEmbedding
    from ..geometry._source_boundary_records import SourceBoundaryQuery
    from ..geometry._source_fidelity_certificate import SourceFidelityWitness
    from ..geometry._surface_source_support import SurfaceSourceCharts
    from ..geometry.multiregion_surface import MultiRegionSurfaceTopology
    from ..geometry.surface import SurfaceModel
    from ._association import GeometryAssociation
    from ._certification_inputs import ScopedFidelityInput
    from ._certification_schedule import MeshCertificationSchedule
    from ._controls import ProtectedFeature
    from ._layer_curving_records import LayerCurvingSourceEvidence
    from ._organization import (
        MeshAttribute,
        MeshLabel,
        MeshPatch,
        MeshZone,
        RegionBoundaryEvidence,
    )
    from ._result import MeshingComplianceReport
    from ._scope import _SourceEntityScopes
    from ._trace import MeshingStageReport


@dataclass(frozen=True, slots=True)
class NativeCertificationRequest:
    """Mathematical source premises and obligations for one constructed mesh."""

    schedule: MeshCertificationSchedule
    source_id: str
    source_revision: str
    domain: CoverageDomain | None = None
    cell_regions: np.ndarray | None = None
    fidelity_source: SourceBoundaryQuery | None = None
    fidelity_tolerance: float | None = None
    junction_vertices: ArrayLike | None = None
    scoped_fidelity: tuple[ScopedFidelityInput, ...] = ()
    protected_features: tuple[ProtectedFeature, ...] | None = None
    maximum_normal_deviation: float | None = None
    fidelity_witness: SourceFidelityWitness | None = None
    feature_source: _SourceEntityScopes | None = None
    source_coverage_witness: SourceCoverageWitness | None = None
    # Global embedding the route already computed for exactly this mesh and
    # coordinate map; acceptance checks its bound inputs and reuses it.
    global_embedding: GlobalEmbeddingCertificate | None = None
    # Validity certificates the route already proved (e.g. its layer cells);
    # acceptance reuses their proofs of identical exact cell maps.
    validity_parent: tuple[CellValidityCertificate, ...] = ()


@dataclass(frozen=True, slots=True)
class NativeMeshOrganization:
    """Authored physical organization, independent of execution bookkeeping."""

    boundary: SurfaceModel | None = None
    patches: tuple[MeshPatch, ...] = ()
    zones: tuple[MeshZone, ...] = ()
    labels: tuple[MeshLabel, ...] = ()
    attributes: tuple[MeshAttribute, ...] = ()
    associations: tuple[GeometryAssociation, ...] = ()
    region_boundary_evidence: tuple[RegionBoundaryEvidence, ...] = ()
    surface_source: SurfaceSourceCharts | None = None
    # Region-labeled topology of a non-manifold multiregion surface mesh.
    multiregion_surface: MultiRegionSurfaceTopology | None = None


@dataclass(frozen=True, slots=True)
class NativeMeshConstruction:
    """A candidate and its coordinate map, not an accepted meshing result."""

    mesh: CellMesh
    coordinate_contract: SpatialCoordinateContract
    compliance: MeshingComplianceReport
    certification: NativeCertificationRequest
    geometry: CellGeometrySpec = field(kw_only=True)
    organization: NativeMeshOrganization = field(
        default_factory=NativeMeshOrganization, kw_only=True
    )
    stages: tuple[MeshingStageReport, ...] = field(default=(), kw_only=True)
    reference_maps: tuple[ReferenceCellMapWitnesses, ...] = field(
        default=(), kw_only=True
    )
    source_mesh: CellMesh | None = field(default=None, kw_only=True)
    source_geometry: CellGeometrySpec | None = field(default=None, kw_only=True)
    planar_embedding: PlanarEmbedding | None = field(default=None, kw_only=True)
    layer_curving_evidence: LayerCurvingSourceEvidence | None = field(
        default=None, kw_only=True
    )
