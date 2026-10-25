#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fill a native B-Rep solid from its certified whole-boundary surface mesh.

`NativeCadVolumeSource` composes three owners without changing any of them:
the B-Rep `MeshingDomain` remains the scientific authority, an accepted
``parametric_surface`` result of that whole domain supplies the immutable
boundary triangles and their exact B-Rep lineage, and a fixed-boundary
`PiecewiseLinearComplex` with its own derived identity is the declared
piecewise-linear approximation consumed by native constrained tetrahedral
generation. Facets are B-Rep faces and regions are B-Rep solids; both come from
the surface face lineage and the domain's sided face/solid incidence, never from
coordinates.

The ``cad_tetrahedral`` route publishes under the B-Rep source identity only
after the ``volume_cad`` schedule has certified per-cell validity, topology,
global embedding, coverage of the declared approximation and continuous
fidelity of the tetrahedral boundary to the B-Rep, and after exact B-Rep
associations of vertices, boundary faces and cells have been rebuilt from that
lineage.
"""

from __future__ import annotations

from typing import final

import equinox as eqx
import numpy as np

from phydrax.geometry._domain_boundary import MeshingDomainBoundarySource
from phydrax.geometry._surface_source_support import (
    SurfaceNativeRestrictionBoundarySource,
)
from phydrax.meshing._certification_schedule import MeshCertificationSchedule
from phydrax.meshing._contracts import (
    MeshingProviderInfo,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing.providers._native_route_policy import publication_policy

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._meshcore import (
    NativeBatchTicket,
    NativeExecutionScope,
    NativeHostStorageWorkspace,
)
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import positive_finite_float
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...discretization._coordinate_operands import CoordinateOperandCollector
from ...geometry._meshing_domain import MeshingDomain
from ...geometry.brep._model import BRepModel
from ...typing import checked
from .._association_state import (
    entity_rows,
    GeometryAssociation,
    GeometryAssociationKind,
)
from .._controls import FeatureKind, RegionControl
from .._host_banks import entities as host_entities
from .._measurements import NativeMeshingPhaseRecorder
from .._plc_source import PiecewiseLinearComplex
from .._result import CellMeshingResult, MeshingComplianceReport
from .._scope import MeshingScope
from .._sizing import UniformSizeControl
from .._trace import MeshingStageKind, MeshingStageReport, MeshingStageStatus
from .._volume_generation import (
    declared_plc_domain,
    generate_plc_volume,
    NativeVolumeSchedule,
    VolumeConstruction,
)
from . import _native_plc_geometry as plc_geometry_owner
from ._native_publication import (
    NativeCertificationRequest,
    NativePublicationExecution,
    NativePublicationOrganization,
    publish_native_result,
)
from ._native_route_checks import check_deadline, simplex_entity_limits
from ._native_sources import NativePlcSource, NativeSurfaceSource
from ._native_volume import plc_volume_audit_policy, volume_compliance


def _surface_association(
    surface: CellMeshingResult, dimension: int, /
) -> GeometryAssociation:
    """The one B-Rep association of a surface stratum dimension."""
    entity_set = surface.mesh.entity_set(dimension).entity_set_id
    found = tuple(
        value
        for value in surface.associations
        if value.target_entity_set_id == entity_set
    )
    if len(found) != 1 or found[0].association_kind is not GeometryAssociationKind.BREP:
        raise ValueError(
            f"NativeCadVolumeSource.surface must publish one B-Rep association of dimension {dimension}; got {len(found)}"
        )
    association = found[0]
    if not bool(np.all(np.asarray(association.resolved))) or bool(
        np.any(np.asarray(association.ambiguous))
    ):
        raise ValueError(
            f"NativeCadVolumeSource.surface associations must be resolved and unambiguous; got unresolved dimension {dimension} rows"
        )
    return association


def _approximation_identity(
    domain: MeshingDomain, surface_result_id: str, /
) -> tuple[str, str]:
    """Derived identity of the PL approximation; never the B-Rep identity."""
    source_id = canonical_fingerprint(
        {"kind": "native-cad-volume-approximation", "source": domain.source_id}
    )
    revision = canonical_fingerprint(
        {
            "kind": "native-cad-volume-approximation-revision",
            "domain": domain.domain_id,
            "authority": domain.authority_id,
            "surface": surface_result_id,
        }
    )
    return source_id, revision


@final
class _CadLineage(StrictModule):
    """Boundary complex and B-Rep lineage derived from the surface.

    Face residuals are aligned with approximation polygons (surface triangle
    rows); vertex banks are aligned with approximation vertices.
    """

    complex: PiecewiseLinearComplex
    vertex_dimensions: np.ndarray
    vertex_indices: np.ndarray
    vertex_parameters: np.ndarray
    vertex_residuals: np.ndarray
    face_residuals: np.ndarray
    vertex_entity_ids: tuple[str, ...] = eqx.field(static=True)
    vertex_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    vertex_exact: bool = eqx.field(static=True)
    face_exact: bool = eqx.field(static=True)


def _cad_lineage(
    domain: MeshingDomain, surface: CellMeshingResult, /
) -> _CadLineage:
    """Orient faces by their exact B-Rep face lineage and sided solid incidence."""
    mesh = surface.mesh
    if mesh.topological_dimension != 2 or mesh.ambient_dimension != 3:
        raise ValueError(
            "NativeCadVolumeSource.surface must be an embedded triangle mesh; got another dimension"
        )
    if any(block.cell_kind != "triangle" for block in mesh.blocks):
        raise ValueError(
            "NativeCadVolumeSource.surface must contain only triangles; got another cell kind"
        )
    if not domain.regions:
        raise ValueError(
            "NativeCadVolumeSource.domain must declare B-Rep solids; got no regions"
        )
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    triangles = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
    )
    faces = _surface_association(surface, 2)
    face_rows = entity_rows(mesh, 2, np.asarray(faces.target_global_ids, np.int64))
    if (
        face_rows.size != triangles.shape[0]
        or np.any(face_rows < 0)
        or np.unique(face_rows).size != triangles.shape[0]
        or np.any(np.asarray(faces.source_dimensions) != 2)
    ):
        raise ValueError(
            "NativeCadVolumeSource.surface must associate every triangle with one B-Rep face; got incomplete face lineage"
        )
    # Entity identities carry the B-Rep index and its occurrence path.
    patch_rows = {domain.entity_id(2, row): row for row in range(len(domain.patches))}
    source_faces = faces.source_entities.entity_ids()
    if any(value not in patch_rows for value in source_faces):
        raise ValueError(
            "NativeCadVolumeSource.surface face lineage must name domain faces; got an unknown face"
        )
    orientations = np.asarray(faces.orientations, dtype=np.int64)
    if not np.all(np.abs(orientations) == 1):
        raise ValueError(
            "NativeCadVolumeSource.surface face lineage must orient every triangle against its B-Rep face; got undefined orientation"
        )
    facets = np.empty((triangles.shape[0],), dtype=np.int64)
    oriented = np.empty_like(triangles)
    facets[face_rows] = [patch_rows[value] for value in source_faces]
    face_residuals = np.empty((triangles.shape[0],), dtype=np.float64)
    face_residuals[face_rows] = np.asarray(faces.residuals, dtype=np.float64)
    oriented[face_rows] = np.where(
        (orientations > 0)[:, None],
        triangles[face_rows],
        triangles[face_rows][:, ::-1],
    )
    if np.unique(facets).size != len(domain.patches):
        raise ValueError(
            "NativeCadVolumeSource.surface must cover every B-Rep face; got uncovered faces"
        )
    # patch_regions holds (negative side, positive side); PLC facets declare
    # (right-hand normal side, opposite side).
    sides = np.asarray(domain.patch_regions, dtype=np.int64)[:, ::-1]
    corners = points[oriented]
    determinants = np.einsum(
        "ij,ij->i", corners[:, 0], np.cross(corners[:, 1], corners[:, 2])
    )
    for region in range(len(domain.regions)):
        outward = np.isin(facets, np.flatnonzero(sides[:, 1] == region))
        inward = np.isin(facets, np.flatnonzero(sides[:, 0] == region))
        volume = (
            float(np.sum(determinants[outward])) - float(np.sum(determinants[inward]))
        ) / 6.0
        if not volume > 0.0:
            raise ValueError(
                "NativeCadVolumeSource.surface must bound every B-Rep solid with positive oriented volume; got "
                f"{volume!r} for region {region}"
            )
    vertices = _surface_association(surface, 0)
    vertex_rows = entity_rows(mesh, 0, np.asarray(vertices.target_global_ids, np.int64))
    if (
        vertex_rows.size != points.shape[0]
        or np.any(vertex_rows < 0)
        or np.unique(vertex_rows).size != points.shape[0]
    ):
        raise ValueError(
            "NativeCadVolumeSource.surface must associate every vertex with a B-Rep stratum; got incomplete vertex lineage"
        )
    order = np.argsort(vertex_rows)
    entity_ids = vertices.source_entities.entity_ids()
    paths = vertices.source_entities.occurrence_paths()
    complex_ = PiecewiseLinearComplex(
        points,
        list(oriented),
        facets,
        sides,
        tuple(region.name for region in domain.regions),
        boundary="fixed",
    )
    return _CadLineage(
        complex=complex_,
        vertex_dimensions=np.asarray(vertices.source_dimensions, np.int64)[order],
        vertex_indices=np.asarray(vertices.source_indices, np.int64)[order],
        vertex_parameters=np.asarray(vertices.parameters, np.float64)[order],
        vertex_residuals=np.asarray(vertices.residuals, np.float64)[order],
        face_residuals=face_residuals,
        vertex_entity_ids=tuple(entity_ids[int(row)] for row in order),
        vertex_occurrence_paths=tuple(paths[int(row)] for row in order),
        vertex_exact=vertices.exact,
        face_exact=faces.exact,
    )


@final
class NativeCadVolumeSource(StrictModule, NonTrainableState):
    """A B-Rep solid domain bound to its certified whole-boundary surface mesh.

    ``surface`` must be an accepted, certified result of the ``parametric_surface``
    route for every face of ``source.domain``. Its certified continuous fidelity
    tolerance is ``maximum_deviation``; the volume route recertifies its
    tetrahedral boundary against the B-Rep at that tolerance or at a tighter
    requested whole-boundary surface deviation, against the surface's certified
    root boundary source and its explicit chart chain. ``plc_source`` is the
    fixed boundary approximation with a derived identity distinct from the B-Rep.
    """

    domain: MeshingDomain
    surface: CellMeshingResult
    plc_source: NativePlcSource
    fidelity_source: MeshingDomainBoundarySource
    vertex_dimensions: np.ndarray
    vertex_indices: np.ndarray
    vertex_parameters: np.ndarray
    vertex_residuals: np.ndarray
    face_residuals: np.ndarray
    vertex_entity_ids: tuple[str, ...] = eqx.field(static=True)
    vertex_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    vertex_exact: bool = eqx.field(static=True)
    face_exact: bool = eqx.field(static=True)
    maximum_deviation: float = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self, source: NativeSurfaceSource, surface: CellMeshingResult, /
    ) -> None:
        domain = source.domain
        certification = surface.certification
        if (
            certification is None
            or not certification.passed
            or not surface.audit.passed
            or certification.schedule.route != "surface"
        ):
            raise ValueError(
                "NativeCadVolumeSource.surface must be a certified native surface result; got missing or failed certification"
            )
        binding = surface.trace.binding
        if binding is None or (binding.source_id, binding.source_revision) != (
            domain.source_id,
            domain.source_revision,
        ):
            raise ValueError(
                "NativeCadVolumeSource.surface must bind the B-Rep domain revision; got another source binding"
            )
        query = certification.request.source
        if isinstance(query, SurfaceNativeRestrictionBoundarySource):
            query = query.root
        tolerance = certification.request.fidelity_tolerance
        if (
            not isinstance(query, MeshingDomainBoundarySource)
            or query.domain.domain_id != domain.domain_id
            or set(query.patches) != set(range(len(domain.patches)))
            or tolerance is None
        ):
            raise ValueError(
                "NativeCadVolumeSource.surface must certify continuous fidelity to every face of this B-Rep domain; got another fidelity source"
            )
        authority = domain.brep_authority
        if not isinstance(authority, BRepModel) or domain.authority_id != authority.model_id:
            raise ValueError(
                "NativeCadVolumeSource.domain must retain its B-Rep model authority; got a domain without B-Rep model ancestry"
            )
        lineage = _cad_lineage(domain, surface)
        approximation_id, approximation_revision = _approximation_identity(
            domain, surface.result_id
        )
        plc_source = NativePlcSource(
            lineage.complex, approximation_id, approximation_revision
        )
        # The certified root boundary source keeps the explicit chart chain that
        # the fidelity owner re-verifies; a chain-free source would only sample.
        fidelity_source = query
        self.domain = domain
        self.surface = surface
        self.plc_source = plc_source
        self.fidelity_source = fidelity_source
        self.vertex_dimensions = lineage.vertex_dimensions
        self.vertex_indices = lineage.vertex_indices
        self.vertex_parameters = lineage.vertex_parameters
        self.vertex_residuals = lineage.vertex_residuals
        self.face_residuals = lineage.face_residuals
        self.vertex_entity_ids = lineage.vertex_entity_ids
        self.vertex_occurrence_paths = lineage.vertex_occurrence_paths
        self.vertex_exact = lineage.vertex_exact
        self.face_exact = lineage.face_exact
        self.maximum_deviation = positive_finite_float(
            float(tolerance), "NativeCadVolumeSource.maximum_deviation"
        )
        self.source_id = domain.source_id
        self.source_revision = domain.source_revision
        self.binding_id = canonical_fingerprint(
            {
                "kind": "native-cad-volume-source",
                "domain": domain.domain_id,
                "authority": domain.authority_id,
                "surface": surface.result_id,
                "approximation": plc_source.binding_id,
                "fidelity_source": fidelity_source.source_id,
                "maximum_deviation": self.maximum_deviation,
                "vertex_lineage": array_tree_fingerprint(
                    (
                        self.vertex_dimensions,
                        self.vertex_indices,
                        self.vertex_parameters,
                        self.vertex_residuals,
                    )
                ),
                "face_residuals": array_tree_fingerprint(self.face_residuals),
                "vertex_entities": list(self.vertex_entity_ids),
                "vertex_paths": [list(path) for path in self.vertex_occurrence_paths],
                "lineage_exact": (self.vertex_exact, self.face_exact),
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[NativeCadVolumeSource], value: NativeCadVolumeSource, /
    ) -> NativeCadVolumeSource:
        """Rederive every bank from the stored domain and certified surface."""
        if type(value) is not cls:
            raise TypeError(
                f"{cls.__name__}.stored_fields must have the registered type; got {type(value).__name__}"
            )
        fresh = cls(NativeSurfaceSource(value.domain), value.surface)
        if fresh.binding_id != value.binding_id:
            raise ValueError(
                f"{cls.__name__}.binding_id must match its stored domain and surface; got {value.binding_id}"
            )
        return fresh


def _rebind(scope: MeshingScope, source: NativeCadVolumeSource, /) -> MeshingScope:
    """Map a B-Rep face/solid scope onto the approximation's facet/region rows."""
    domain = source.domain
    dimension = scope.entity_dimension
    if (scope.source_id, scope.source_revision) != (
        source.source_id,
        source.source_revision,
    ) or scope.entity_set_id != domain.entity_set_id(dimension):
        raise ValueError(
            "cad_tetrahedral.scope must bind the B-Rep domain entity set; got another scope"
        )
    match dimension:
        case 2:
            rows = {int(index): row for row, index in enumerate(domain.scope_indices(2))}
            entity_set = "facets"
        case 3:
            rows = {int(index): row for row, index in enumerate(domain.scope_indices(3))}
            entity_set = "regions"
        case _:
            raise ValueError(
                f"cad_tetrahedral.scope must select B-Rep faces or solids; got dimension {dimension}"
            )
    identifiers = np.asarray(scope.entity_ids, dtype=np.int64)
    if any(int(value) not in rows for value in identifiers):
        raise ValueError(
            "cad_tetrahedral.scope must select declared B-Rep entities; got unknown entities"
        )
    plc = source.plc_source
    return MeshingScope(
        plc.source_id,
        plc.source_revision,
        scope.entity_kind,
        dimension,
        f"{plc.source_id}:{entity_set}",
        np.asarray([rows[int(value)] for value in identifiers], dtype=np.int64),
    )


@checked
def cad_volume_support_issues(
    source: NativeCadVolumeSource, specification: VolumeMeshingSpec, /
) -> list[str]:
    """Return unmet CAD volume requirements in validation order."""
    issues: list[str] = []
    target = specification.target
    families = target.cell_families
    if set((*families.required, *families.preferred)) != {"tetrahedron"}:
        issues.append("a tetrahedral volume target")
    if (
        target.topological_dimension != 3
        or target.ambient_dimension != 3
        or target.geometry_order != 1
        or families.allowed_transitions
        or families.allow_mixed
    ):
        issues.append("affine tetrahedra in ambient dimension three")
    if specification.fill_strategy is not VolumeFillStrategy.SIMPLEX:
        issues.append("the simplex fill strategy")
    boundary = specification.boundary_scope
    domain = source.domain
    if (
        boundary.entity_dimension != 2
        or (boundary.source_id, boundary.source_revision)
        != (source.source_id, source.source_revision)
        or boundary.entity_set_id != domain.entity_set_id(2)
        or not np.array_equal(
            np.sort(np.asarray(boundary.entity_ids, dtype=np.int64)),
            np.sort(np.asarray(domain.scope_indices(2), dtype=np.int64)),
        )
    ):
        issues.append("a boundary scope of every B-Rep face")
    if any(not isinstance(value, UniformSizeControl) for value in specification.size_controls):
        issues.append("uniform size controls on B-Rep faces")
    for feature in specification.protected_features:
        if (
            feature.feature_kind is not FeatureKind.SURFACE
            or feature.scope.scope_id != boundary.scope_id
        ):
            issues.append("whole-boundary B-Rep surface deviation requests only")
    if (
        specification.patch_controls
        or specification.hole_seeds
        or specification.region_seeds
        or specification.periodic_constraints
        or specification.layer_controls
    ):
        issues.append(
            "B-Rep solids without patch, seed, periodic or layer controls"
        )
    for control in specification.region_controls:
        if control.scope.entity_dimension != 3 or not control.meshing_enabled:
            issues.append("enabled region controls on B-Rep solids")
    return issues


def _approximation_specification(
    source: NativeCadVolumeSource, specification: VolumeMeshingSpec, /
) -> VolumeMeshingSpec:
    """The same controls rebound to the approximation's facet/region rows."""
    complex_ = source.plc_source.complex
    region_controls = []
    for control in specification.region_controls:
        rebound = _rebind(control.scope, source)
        row = int(np.asarray(rebound.entity_ids)[0])
        if np.asarray(rebound.entity_ids).size != 1:
            raise ValueError(
                "cad_tetrahedral.region_controls must each select one B-Rep solid; got several"
            )
        region_controls.append(
            RegionControl(
                rebound,
                complex_.region_ids[row],
                control.material_id,
                control.role,
                meshing_enabled=control.meshing_enabled,
            )
        )
    return VolumeMeshingSpec(
        specification.target,
        _rebind(specification.boundary_scope, source),
        specification.fill_strategy,
        size_controls=tuple(
            UniformSizeControl(
                _rebind(control.scope, source),
                control.target_size,
                minimum_size=control.minimum_size,
                maximum_size=control.maximum_size,
                maximum_growth_rate=control.maximum_growth_rate,
                strength=control.strength,
                priority=control.priority,
            )
            for control in specification.size_controls
            if isinstance(control, UniformSizeControl)
        ),
        region_controls=tuple(region_controls),
        size_combination=specification.size_combination,
        size_compliance=specification.size_compliance,
        limits=specification.limits,
        deterministic=specification.deterministic,
    )


@final
class PreparedCadVolume(StrictModule, NonTrainableState):
    """Admitted CAD volume request, its rebound approximation request and schedule."""

    schedule: NativeVolumeSchedule
    approximation: VolumeMeshingSpec
    fidelity_tolerance: float = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)
    source_binding_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: NativeCadVolumeSource,
        specification: VolumeMeshingSpec,
        schedule: NativeVolumeSchedule,
        /,
    ) -> None:
        issues = cad_volume_support_issues(source, specification)
        if issues:
            raise ValueError(
                "PreparedCadVolume.constraints must be supported; got "
                + "; ".join(issues)
            )
        self.schedule = schedule
        self.approximation = _approximation_specification(source, specification)
        self.fidelity_tolerance = min(
            (feature.maximum_deviation for feature in specification.protected_features),
            default=source.maximum_deviation,
        )
        self.specification_id = specification.specification_id
        self.source_binding_id = source.binding_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-native-cad-volume",
                "source": source.binding_id,
                "specification": specification.specification_id,
                "approximation": self.approximation.specification_id,
                "schedule": schedule.schedule_id,
                "fidelity_tolerance": self.fidelity_tolerance,
            }
        )


def _cad_associations(
    source: NativeCadVolumeSource, construction: VolumeConstruction, /
) -> tuple[GeometryAssociation, ...]:
    """B-Rep vertex, boundary-face and cell lineage of the tetrahedra.

    Identities are exact; boundary faces keep the certified chord residuals of
    the surface triangles they reproduce, never a zero-residual claim.
    """
    mesh = construction.mesh
    domain = source.domain
    lineage = {
        mesh.entity_set(dimension).entity_set_id: dimension for dimension in (0, 2, 3)
    }
    banks = {
        lineage[value.target_entity_set_id]: value
        for value in construction.associations
        if value.target_entity_set_id in lineage
    }
    if set(banks) != {0, 2, 3}:
        raise ValueError(
            "cad_tetrahedral.construction must publish vertex, face and cell approximation lineage; got missing banks"
        )
    vertices, faces, cells = banks[0], banks[2], banks[3]
    vertex_dimensions = np.asarray(vertices.source_dimensions, dtype=np.int64)
    vertex_sources = np.asarray(vertices.source_indices, dtype=np.int64)
    boundary = vertex_dimensions == 0
    if np.any((vertex_dimensions != 0) & (vertex_dimensions != 3)):
        raise ValueError(
            "cad_tetrahedral.construction must keep the fixed B-Rep boundary without Steiner vertices; got vertices on approximation edges or facets"
        )
    region_indices = np.asarray(domain.region_source_indices, dtype=np.int64)
    rows = vertex_sources
    dimensions = np.where(
        boundary, source.vertex_dimensions[np.where(boundary, rows, 0)], 3
    )
    indices = np.where(
        boundary,
        source.vertex_indices[np.where(boundary, rows, 0)],
        region_indices[np.where(boundary, 0, rows)],
    )
    interior = rows[~boundary]
    region_ids = iter(
        domain.entity_ids(np.full(interior.shape, 3, dtype=np.int64), interior)
    )
    entity_ids = tuple(
        source.vertex_entity_ids[int(row)] if on_boundary else next(region_ids)
        for row, on_boundary in zip(rows.tolist(), boundary.tolist(), strict=True)
    )
    paths = tuple(
        source.vertex_occurrence_paths[int(row)]
        if on_boundary
        else domain.region_source_occurrences[int(row)]
        for row, on_boundary in zip(rows.tolist(), boundary.tolist(), strict=True)
    )
    parameters = np.where(
        boundary[:, None],
        source.vertex_parameters[np.where(boundary, rows, 0)],
        np.nan,
    )
    residuals = np.where(
        boundary, source.vertex_residuals[np.where(boundary, rows, 0)], 0.0
    )
    # Every approximation facet is one B-Rep face and every fixed polygon is one
    # boundary face, so each face reproduces exactly one surface triangle.
    face_dimensions = np.asarray(faces.source_dimensions, dtype=np.int64)
    face_sources = np.asarray(faces.source_indices, dtype=np.int64)
    on_boundary_face = face_dimensions == 2
    face_ids = np.asarray(faces.target_global_ids, dtype=np.int64)[on_boundary_face]
    face_rows = face_sources[on_boundary_face]
    complex_ = source.plc_source.complex
    polygons = np.asarray(complex_.polygon_vertices, dtype=np.int64).reshape((-1, 3))
    polygon_rows = {
        key: row for row, key in enumerate(map(tuple, np.sort(polygons, axis=1).tolist()))
    }
    approximation_vertices = np.full((mesh.coordinates.shape[0],), -1, dtype=np.int64)
    vertex_rows = entity_rows(mesh, 0, np.asarray(vertices.target_global_ids, np.int64))
    approximation_vertices[vertex_rows[boundary]] = vertex_sources[boundary]
    keys = np.sort(
        approximation_vertices[host_entities(mesh, 2)[entity_rows(mesh, 2, face_ids)]],
        axis=1,
    )
    face_polygons = np.asarray(
        [polygon_rows.get(tuple(key), -1) for key in keys.tolist()], dtype=np.int64
    )
    if np.any(face_polygons < 0) or not np.array_equal(
        np.asarray(complex_.polygon_facets, dtype=np.int64)[face_polygons], face_rows
    ):
        raise ValueError(
            "cad_tetrahedral.construction must reproduce every fixed approximation polygon as one facet face; got another boundary face"
        )
    face_indices = np.asarray(domain.source_indices[2], dtype=np.int64)[face_rows]
    cell_regions = np.asarray(cells.source_indices, dtype=np.int64)
    return (
        GeometryAssociation(
            GeometryAssociationKind.BREP,
            source.source_id,
            source.source_revision,
            vertices.target_entity_set_id,
            np.asarray(vertices.target_global_ids, dtype=np.int64),
            entity_ids,
            residuals,
            exact=source.vertex_exact,
            source_dimensions=dimensions,
            source_indices=indices,
            source_occurrence_paths=paths,
            parameters=parameters,
        ),
        GeometryAssociation(
            GeometryAssociationKind.BREP,
            source.source_id,
            source.source_revision,
            faces.target_entity_set_id,
            face_ids,
            domain.entity_ids(np.full(face_rows.shape, 2, dtype=np.int64), face_rows),
            source.face_residuals[face_polygons],
            exact=source.face_exact,
            orientations=np.asarray(faces.orientations, dtype=np.int64)[on_boundary_face],
            source_dimensions=np.full(face_ids.shape, 2, dtype=np.int64),
            source_indices=face_indices,
            source_occurrence_paths=tuple(
                domain.source_occurrences[2][int(row)] for row in face_rows.tolist()
            ),
        ),
        GeometryAssociation(
            GeometryAssociationKind.BREP,
            source.source_id,
            source.source_revision,
            cells.target_entity_set_id,
            np.asarray(cells.target_global_ids, dtype=np.int64),
            domain.entity_ids(
                np.full(cell_regions.shape, 3, dtype=np.int64), cell_regions
            ),
            np.zeros(cell_regions.shape, dtype=np.float64),
            exact=True,
            source_dimensions=np.full(cell_regions.shape, 3, dtype=np.int64),
            source_indices=region_indices[cell_regions],
            source_occurrence_paths=tuple(
                domain.region_source_occurrences[int(row)]
                for row in cell_regions.tolist()
            ),
        ),
    )


@checked
def execute_cad_volume_route(
    source: NativeCadVolumeSource,
    specification: VolumeMeshingSpec,
    prepared: PreparedCadVolume,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    coordinate_operands: CoordinateOperandCollector | None = None,
) -> CellMeshingResult:
    """Construct the fixed-boundary fill and publish it only under B-Rep acceptance."""
    if (
        prepared.source_binding_id != source.binding_id
        or prepared.specification_id != specification.specification_id
    ):
        raise ValueError(
            "PreparedCadVolume.binding must match source and specification IDs; got a different prepared route binding"
        )
    authority = source.domain.brep_authority
    contract = (
        authority.coordinate_contract if isinstance(authority, BRepModel) else None
    )
    if contract is None or contract.spatial_id != coordinate_contract.spatial_id:
        raise ValueError(
            "cad_tetrahedral.coordinate_contract must preserve the B-Rep spatial contract; got another spatial ID"
        )
    workspace.retain_owner((source, specification, prepared))
    started = scope.operation_started
    limits = specification.limits
    approximation = prepared.approximation
    plc = source.plc_source
    audit_policy = plc_volume_audit_policy()
    construction = generate_plc_volume(
        plc.complex,
        approximation,
        prepared.schedule,
        validity_policy=audit_policy.validity_policy,
        source_id=plc.source_id,
        source_revision=plc.source_revision,
        input_id=prepared.prepared_id,
        scope=scope,
        record_phase=record_phase,
    )
    check_deadline(started, limits, MeshingStageKind.VOLUME_FILL)
    mesh = construction.mesh
    vertices = np.asarray(mesh.coordinates, dtype=np.float64)
    cells = np.asarray(mesh.blocks[0].vertices, dtype=np.int64)
    simplex_entity_limits(
        vertices,
        cells,
        limits,
        MeshingStageKind.VOLUME_FILL,
        cell_kind=mesh.blocks[0].cell_kind,
    )
    approximate = volume_compliance(
        approximation, prepared.schedule, construction, vertices, cells, scope=scope
    )
    compliance = MeshingComplianceReport(
        specification.specification_id,
        issues=approximate.issues,
        requested=(
            *approximate.requested,
            ("cad:maximum_boundary_deviation", prepared.fidelity_tolerance),
        ),
        achieved=approximate.achieved,
    )
    associations = _cad_associations(source, construction)
    check_deadline(started, limits, MeshingStageKind.GEOMETRY_AUDIT)
    allowance = scope.remaining()
    with scope.batch(
        min(limits.maximum_work_units, allowance.remaining_work_units),
        queries_upper=min(
            limits.maximum_geometry_queries, allowance.remaining_geometry_queries
        ),
        bytes_upper=min(limits.maximum_scratch_bytes, allowance.remaining_managed_bytes),
    ) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        budget.capture_signed_determinant_operands = (
            coordinate_operands is not None and coordinate_operands.capture_signed_determinants
        )
        operands = None if coordinate_operands is None else coordinate_operands.source_operands
        if operands is not None:
            if not any(owner is source.domain for owner, _, _, _ in operands._input_owners):
                raise ValueError("CAD surface operands must bind this actual source domain.")
            receipt = source.surface.execution_evidence
            if receipt is None:
                raise ValueError("CAD surface operands require their actual earning receipt.")
            # This batch owns the remaining root allowance. Receipt ancestry
            # checks must debit it, not the already-reserved root balance.
            with ticket.native_target():
                operands.require(
                    source.surface.mesh, source.surface.geometry, receipt, budget,
                    coordinate_contract=source.surface.coordinate_contract,
                )
        with budget.activate():
            geometry = (
                plc_geometry_owner.exact_plc_geometry(construction, plc, budget=budget)
                if construction.geometry is None
                else construction.geometry
            )
    stages = (
        MeshingStageReport(
            MeshingStageKind.SOURCE_INSPECTION,
            MeshingStageStatus.PASSED,
            input_ids=(source.binding_id, source.surface.result_id),
            output_ids=(plc.binding_id, prepared.prepared_id),
        ),
        *construction.stages,
    )

    def retain_coordinate_storage(bytes_upper: int) -> NativeBatchTicket:
        return scope.batch(0, bytes_upper=bytes_upper)

    with budget.retained_storage(retain_coordinate_storage):
        return publish_native_result(
            mesh,
            coordinate_contract,
            compliance,
            stages,
            provider,
            {
                "kind": "native-cad-volume-mesh",
                "route": "cad_tetrahedral",
                "source": source.binding_id,
                "approximation": plc.binding_id,
                "plan": plan_id,
                "specification": specification.specification_id,
                "schedule": prepared.schedule.schedule_id,
            },
            NativeCertificationRequest(
                MeshCertificationSchedule("volume_cad"),
                source.source_id,
                source.source_revision,
                limits,
                domain=declared_plc_domain(plc.complex, plc.source_id),
                cell_regions=construction.cell_regions,
                fidelity_source=source.fidelity_source,
                fidelity_tolerance=prepared.fidelity_tolerance,
            ),
            policy=publication_policy("volume", audit_policy=audit_policy),
            organization=NativePublicationOrganization(
                patches=construction.patches,
                zones=construction.zones,
                labels=construction.labels,
                associations=associations,
            ),
            execution=NativePublicationExecution(scope, workspace, record_phase, budget),
            geometry=geometry,
        )




__all__ = [
    "NativeCadVolumeSource",
    "PreparedCadVolume",
    "cad_volume_support_issues",
    "execute_cad_volume_route",
]
