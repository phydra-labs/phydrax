#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native tetrahedral construction of explicit unsigned-distance envelopes."""

from __future__ import annotations

from fractions import Fraction
from numbers import Integral
from typing import final

import equinox as eqx
import numpy as np

import phydrax.geometry._source_fidelity as source_fidelity_owner

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import MeshcoreError, surface_envelope, SurfaceEnvelopeResult
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier, finite_real_scalar
from ..discretization import CellGeometrySpec, CellMesh, reference_cell_topology
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..geometry._domain_boundary import MeshingDomainBoundarySource
from ..geometry._domain_coverage import PiecewiseLinearDomain
from ..geometry._piecewise_linear_boundary import PiecewiseLinearBoundarySource
from ..geometry._source_boundary_records import (
    rational_chart_tokens,
    SourceBoundaryChartWitness,
    SourceBoundaryPointWitness,
    SurfaceEnvelopeDistanceWitness,
)
from ..geometry.reconstruction._boundary import ReconstructedTriangleBoundarySource
from ..geometry.reconstruction._core import ReconstructedGeometrySource
from ..geometry.reconstruction._sheet import AuthoredTriangleSheet
from ..geometry.simplicial._regions import MeshRegion
from ..geometry.surface import SurfaceMetadata, SurfaceModel
from ..typing import checked
from ._association import GeometrySourceEntityRole
from ._certification_schedule import MeshCertificationSchedule
from ._contracts import (
    MeshingFailure,
    MeshingFailureCategory,
    VolumeMeshingSpec,
)
from ._controls import ProtectedFeature
from ._native_construction import (
    NativeCertificationRequest,
    NativeMeshConstruction,
    NativeMeshOrganization,
)
from ._organization import MeshLabel
from ._plc_source import _numeric_entity_ids, PiecewiseLinearComplex
from ._result import MeshingComplianceReport
from ._scope import MeshingEntityKind, MeshingScope
from ._sizing import UniformSizeControl
from ._volume_generation import NativeVolumeSchedule
from .providers._native_options import NativeMeshingOptions
from .providers._native_plc_geometry import (
    simplex_volume_entities,
    volume_cell_entity_rows,
)
from .providers._native_sources import NativePlcSource
from .providers._native_volume import PreparedPlcVolume, volume_support_issues


def _failure(
    message: str,
    /,
    *,
    category: MeshingFailureCategory = MeshingFailureCategory.UNSUPPORTED_COMBINATION,
    stage: str = "surface_envelope",
) -> MeshingFailure:
    return MeshingFailure(
        category,
        message,
        provider_code="native_surface_envelope",
        stage=stage,
    )


def _real_positive(value: object, name: str, /) -> float:
    result = finite_real_scalar(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be a positive finite physical distance.")
    return result


type _EnvelopeOriginal = ReconstructedGeometrySource | AuthoredTriangleSheet


def _original_domain_query(source: _EnvelopeOriginal, /) -> MeshingDomainBoundarySource:
    domain = source.meshing_domain
    if domain is None:
        raise _failure(
            "A surface envelope requires retained persistent triangle incidence.",
            category=MeshingFailureCategory.INVALID_SOURCE,
            stage="source_inspection",
        )
    domain.require_current(source.source_id, source.source_revision)
    return MeshingDomainBoundarySource(domain, tuple(range(len(domain.patches))))


def _fresh_original(source: _EnvelopeOriginal, /) -> _EnvelopeOriginal:
    if isinstance(source, AuthoredTriangleSheet):
        fresh: _EnvelopeOriginal = AuthoredTriangleSheet(
            source.vertices, source.triangles, feature_id=source.feature_id
        )
    else:
        fresh = ReconstructedGeometrySource(
            source.source,
            source.report,
            persistent_vertex_ids=np.asarray(
                source.persistent_vertex_ids, dtype=np.int64
            ),
            persistent_edge_ids=np.asarray(source.persistent_edge_ids, dtype=np.int64),
            persistent_facet_ids=np.asarray(source.persistent_facet_ids, dtype=np.int64),
            persistent_region_id=source.persistent_region_id,
        )
    if (
        fresh.source_id != source.source_id
        or fresh.source_revision != source.source_revision
        or fresh.binding_id != source.binding_id
        or not np.array_equal(fresh.persistent_vertex_ids, source.persistent_vertex_ids)
        or not np.array_equal(fresh.persistent_edge_ids, source.persistent_edge_ids)
        or not np.array_equal(fresh.persistent_facet_ids, source.persistent_facet_ids)
    ):
        raise _failure(
            "The represented original source no longer reproduces its retained identity.",
            category=MeshingFailureCategory.INVALID_SOURCE,
            stage="source_validation",
        )
    return fresh


def _original_triangles(source: _EnvelopeOriginal, /) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact triangle complex handed to the unsigned-distance kernel."""
    if isinstance(source, AuthoredTriangleSheet):
        # The repaired complex is the bitwise-identical authored point set.
        return (
            np.asarray(source.repaired_vertices, dtype=np.float64),
            np.asarray(source.repaired_triangles, dtype=np.int64),
        )
    region = source.source
    if not isinstance(region, MeshRegion):
        raise _failure(
            "Surface envelopes require an actual reconstructed MeshRegion source.",
            category=MeshingFailureCategory.INVALID_SOURCE,
            stage="source_validation",
        )
    return (
        np.asarray(region.vertices, dtype=np.float64),
        np.asarray(region.faces, dtype=np.int64),
    )


def _envelope_plc_source(
    domain: PiecewiseLinearDomain,
    vertex_entity_ids: np.ndarray,
    facet_entity_ids: np.ndarray,
    /,
) -> tuple[NativePlcSource, tuple[np.ndarray, ...], np.ndarray]:
    vertices = np.asarray(domain.vertices, dtype=np.float64)
    triangles = np.asarray(domain.facets, dtype=np.int64)
    vertex_ids = np.asarray(vertex_entity_ids, dtype=np.int64)
    facet_ids = np.asarray(facet_entity_ids, dtype=np.int64)
    if (
        len(domain.region_ids) != 1
        or vertex_ids.shape != (vertices.shape[0],)
        or facet_ids.shape != (triangles.shape[0],)
    ):
        raise ValueError(
            "The derived envelope PLC requires one region and complete persistent vertex/facet IDs."
        )
    exact_vertices = tuple(
        tuple(Fraction.from_float(float(value)) for value in point) for point in vertices
    )
    signed_six_volume = Fraction(0)
    edge_rows = set()
    for triangle in triangles.tolist():
        first, second, third = (exact_vertices[int(index)] for index in triangle)
        signed_six_volume += (
            first[0] * (second[1] * third[2] - second[2] * third[1])
            + first[1] * (second[2] * third[0] - second[0] * third[2])
            + first[2] * (second[0] * third[1] - second[1] * third[0])
        )
        edge_rows.update(
            tuple(sorted(pair))
            for pair in (
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            )
        )
    if signed_six_volume <= 0:
        raise _failure(
            "The derived envelope does not retain its declared positive region orientation.",
            category=MeshingFailureCategory.INVALID_SOURCE,
            stage="source_construction",
        )
    edge_vertices = tuple(sorted(edge_rows))
    segment_ids = tuple(
        canonical_fingerprint(
            {
                "kind": "surface-envelope-edge",
                "source": (domain.source_id, domain.source_revision),
                "endpoints": tuple(
                    sorted((int(vertex_ids[first]), int(vertex_ids[last])))
                ),
            }
        )
        for first, last in edge_vertices
    )
    region_sides = (-1, 0)
    complex_ = PiecewiseLinearComplex(
        vertices,
        tuple(np.asarray(face, dtype=np.int64) for face in triangles),
        np.arange(triangles.shape[0], dtype=np.int64),
        np.tile(np.asarray(region_sides, dtype=np.int64), (triangles.shape[0], 1)),
        domain.region_ids,
        segments=np.asarray(edge_vertices, dtype=np.int64).reshape((-1, 2)),
        vertex_ids=tuple(
            f"surface-envelope-vertex:{int(value)}" for value in vertex_ids.tolist()
        ),
        vertex_occurrence_paths=tuple(() for _ in vertex_ids),
        facet_ids=tuple(
            f"surface-envelope-facet:{int(value)}" for value in facet_ids.tolist()
        ),
        facet_occurrence_paths=tuple(() for _ in facet_ids),
        segment_ids=segment_ids,
        segment_occurrence_paths=tuple(() for _ in segment_ids),
        region_occurrence_paths=tuple(() for _ in domain.region_ids),
        boundary="fixed",
    )
    source = NativePlcSource(complex_, domain.source_id, domain.source_revision)
    row_ids = tuple(
        np.asarray(values, dtype=np.int64)
        for values in (
            complex_.vertex_entity_ids,
            complex_.plc_edge_entity_ids,
            complex_.facet_entity_ids,
            complex_.region_entity_ids,
        )
    )
    for values in row_ids:
        values.setflags(write=False)
    edges = np.asarray(complex_.plc_edge_vertices, dtype=np.int64)
    edges.setflags(write=False)
    return source, row_ids, edges


def _allocate_envelope_birth_ids(
    source_id: str,
    original: _EnvelopeOriginal,
    controls: tuple[float, float, float, float],
    coordinate_contract: SpatialCoordinateContract,
    vertex_count: int,
    facet_count: int,
    /,
) -> tuple[str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Allocate persistent entity IDs once in native birth order."""
    allocation_id = canonical_fingerprint(
        {
            "kind": "surface-envelope-entity-allocation",
            "source_id": source_id,
            "original": (original.source_id, original.source_revision),
            "controls": controls,
            "coordinate_contract": coordinate_contract.spatial_id,
        }
    )

    def allocated_bank(kind: str, count: int, /) -> np.ndarray:
        birth_keys = tuple(
            canonical_fingerprint(
                {
                    "kind": f"surface-envelope-{kind}-birth",
                    "allocation": allocation_id,
                    "sequence": birth,
                }
            )
            for birth in range(count)
        )
        values = np.asarray(
            _numeric_entity_ids(
                f"surface-envelope-{kind}-birth",
                birth_keys,
                tuple(() for _ in birth_keys),
            ),
            dtype=np.int64,
        )
        values.setflags(write=False)
        return values

    region_id = canonical_fingerprint(
        {"kind": "surface-envelope-region-birth", "allocation": allocation_id}
    )
    vertex_ids = allocated_bank("vertex", vertex_count)
    facet_ids = allocated_bank("facet", facet_count)
    vertex_rows = np.arange(vertex_count, dtype=np.int64)
    facet_rows = np.arange(facet_count, dtype=np.int64)
    vertex_rows.setflags(write=False)
    facet_rows.setflags(write=False)
    return allocation_id, region_id, vertex_ids, facet_ids, vertex_rows, facet_rows


def _envelope_domain(
    source_id: str,
    original: _EnvelopeOriginal,
    result: SurfaceEnvelopeResult,
    native_vertex_rows: np.ndarray,
    native_facet_rows: np.ndarray,
    native_vertex_entity_ids: np.ndarray,
    native_facet_entity_ids: np.ndarray,
    region_id: str,
    /,
) -> tuple[PiecewiseLinearDomain, np.ndarray, np.ndarray]:
    native_vertices = np.asarray(result.surface_vertices, dtype=np.float64)
    native_facets = np.asarray(result.surface_triangles, dtype=np.int64)
    vertex_rows = np.asarray(native_vertex_rows, dtype=np.int64)
    facet_rows = np.asarray(native_facet_rows, dtype=np.int64)
    vertex_birth_ids = np.asarray(native_vertex_entity_ids, dtype=np.int64)
    facet_birth_ids = np.asarray(native_facet_entity_ids, dtype=np.int64)
    if (
        vertex_rows.shape != (native_vertices.shape[0],)
        or facet_rows.shape != (native_facets.shape[0],)
        or vertex_birth_ids.shape != vertex_rows.shape
        or facet_birth_ids.shape != facet_rows.shape
        or np.any(vertex_birth_ids < 0)
        or np.any(facet_birth_ids < 0)
        or np.unique(vertex_birth_ids).size != vertex_birth_ids.size
        or np.unique(facet_birth_ids).size != facet_birth_ids.size
        or not np.array_equal(
            np.sort(vertex_rows), np.arange(vertex_rows.size, dtype=np.int64)
        )
        or not np.array_equal(
            np.sort(facet_rows), np.arange(facet_rows.size, dtype=np.int64)
        )
    ):
        raise ValueError(
            "Persisted native envelope lineage must be bijective over its source rows."
        )
    vertices = np.empty_like(native_vertices)
    vertices[vertex_rows] = native_vertices
    facets = np.empty_like(native_facets)
    vertex_entity_ids = np.empty_like(vertex_birth_ids)
    facet_entity_ids = np.empty_like(facet_birth_ids)
    vertex_entity_ids[vertex_rows] = vertex_birth_ids
    facet_entity_ids[facet_rows] = facet_birth_ids
    for native_facet_row, domain_facet_row in enumerate(facet_rows.tolist()):
        facets[domain_facet_row] = vertex_rows[native_facets[native_facet_row]]
    facet_regions = np.column_stack(
        (
            np.zeros((facets.shape[0],), dtype=np.int64),
            -np.ones((facets.shape[0],), dtype=np.int64),
        )
    )
    domain = PiecewiseLinearDomain(
        vertices,
        facets,
        facet_regions,
        (region_id,),
        source_id=source_id,
    )
    if domain.source_revision == original.source_revision:
        raise _failure(
            "The derived envelope revision must be distinct from its original source revision.",
            category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            stage="source_construction",
        )
    return domain, vertex_entity_ids, facet_entity_ids


@final
class SurfaceEnvelopeSource(StrictModule, NonTrainableState):
    """Explicit unsigned-distance envelope with an independently owned PWL domain.

    ``original`` is either a closed reconstructed ``MeshRegion`` or an authored,
    possibly open and imperfect, ``AuthoredTriangleSheet`` whose topology repair
    interpretation and counts stay on ``original.report``/``original.repair``.
    ``result.bounds`` is the native deviation proof: ``bounds[0]`` bounds the
    distance from every original point to the envelope surface and
    ``bounds[2] <= dist(envelope point, original) <= bounds[1]``.
    """

    original: _EnvelopeOriginal
    original_boundary_query: ReconstructedTriangleBoundarySource
    result: SurfaceEnvelopeResult
    domain: PiecewiseLinearDomain
    plc_source: NativePlcSource
    coordinate_contract: SpatialCoordinateContract
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    region_id: str = eqx.field(static=True)
    allocation_id: str = eqx.field(static=True)
    native_vertex_rows: np.ndarray
    native_vertex_entity_ids: np.ndarray
    native_facet_rows: np.ndarray
    native_facet_entity_ids: np.ndarray
    offset: float = eqx.field(static=True)
    spacing: float = eqx.field(static=True)
    certificate_spacing: float = eqx.field(static=True)
    maximum_deviation: float = eqx.field(static=True)
    original_query_id: str = eqx.field(static=True)
    envelope_id: str = eqx.field(static=True)
    scope_entity_id_banks: tuple[np.ndarray, ...]
    scope_entity_key_banks: tuple[tuple[tuple[str, tuple[str, ...]], ...], ...] = (
        eqx.field(static=True)
    )
    scope_entity_set_ids: tuple[str, ...] = eqx.field(static=True)
    scope_entity_ids_by_mesh_row: tuple[np.ndarray, ...]
    edges: np.ndarray

    @checked
    def __init__(
        self,
        original: _EnvelopeOriginal,
        /,
        *,
        source_id: str,
        coordinate_contract: SpatialCoordinateContract,
        offset: float,
        spacing: float,
        certificate_spacing: float,
        maximum_deviation: float,
    ) -> None:
        if not isinstance(original, (ReconstructedGeometrySource, AuthoredTriangleSheet)):
            raise TypeError(
                "original must be a ReconstructedGeometrySource or AuthoredTriangleSheet."
            )
        if isinstance(original, ReconstructedGeometrySource) and not isinstance(
            original.source, MeshRegion
        ):
            raise TypeError(
                "Surface envelopes require an actual reconstructed MeshRegion source."
            )
        if not isinstance(coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
        if coordinate_contract.coordinate_system != "cartesian":
            raise ValueError(
                "Surface-envelope lengths require an orthonormal Cartesian coordinate contract."
            )
        source_id_ = canonical_identifier(source_id, "source_id")
        controls: tuple[float, float, float, float] = (
            _real_positive(offset, "offset"),
            _real_positive(spacing, "spacing"),
            _real_positive(certificate_spacing, "certificate_spacing"),
            _real_positive(maximum_deviation, "maximum_deviation"),
        )
        # The native kernel certifies an envelope only when the offset dominates
        # its sampling error; refuse the declaration before running the kernel.
        if not controls[0] > np.sqrt(3.0) * controls[1] + controls[2]:
            raise ValueError(
                "offset must exceed sqrt(3) * spacing + certificate_spacing, "
                "the native envelope sampling error it has to dominate."
            )
        fresh_original = _fresh_original(original)
        source_vertices, source_triangles = _original_triangles(fresh_original)
        original_domain_query = _original_domain_query(fresh_original)
        if isinstance(fresh_original, ReconstructedGeometrySource) and (
            np.asarray(fresh_original.scope_entity_ids(3), dtype=np.int64).shape != (1,)
            or fresh_original.persistent_region_id is None
        ):
            raise _failure(
                "A reconstructed MeshRegion must retain one source-owned region identity.",
                category=MeshingFailureCategory.INVALID_SOURCE,
                stage="source_inspection",
            )
        try:
            result = surface_envelope(
                source_vertices,
                source_triangles,
                controls[0],
                controls[1],
                controls[2],
            )
        except MeshcoreError as error:
            raise _failure(
                f"The native surface-envelope kernel refused the represented source and declared controls: {error}",
                category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                stage="surface_envelope_execution",
            ) from error
        if max(float(result.bounds[0]), float(result.bounds[1])) > controls[3]:
            raise _failure(
                "The native upper bound cannot certify the requested maximum_deviation at these envelope controls.",
                category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                stage="source_fidelity",
            )
        if not np.array_equal(
            np.asarray(result.volume_vertices, dtype=np.float64)[
                : result.surface_vertices.shape[0]
            ],
            np.asarray(result.surface_vertices, dtype=np.float64),
        ):
            raise _failure(
                "The native envelope kernel did not retain its exact boundary-vertex prefix in the tetrahedral carrier.",
                category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                stage="surface_envelope_execution",
            )
        (
            allocation_id,
            region_id,
            vertex_entity_ids,
            facet_entity_ids,
            native_vertex_rows,
            native_facet_rows,
        ) = _allocate_envelope_birth_ids(
            source_id_,
            fresh_original,
            controls,
            coordinate_contract,
            int(result.surface_vertices.shape[0]),
            int(result.surface_triangles.shape[0]),
        )
        domain, domain_vertex_ids, domain_facet_ids = _envelope_domain(
            source_id_,
            fresh_original,
            result,
            native_vertex_rows,
            native_facet_rows,
            vertex_entity_ids,
            facet_entity_ids,
            region_id,
        )
        plc_source, row_ids, edges = _envelope_plc_source(
            domain, domain_vertex_ids, domain_facet_ids
        )
        id_banks = plc_source.scope_entity_id_banks
        key_banks = plc_source.scope_entity_key_banks
        set_ids = plc_source.scope_entity_set_ids
        original_query = ReconstructedTriangleBoundarySource(fresh_original)
        self.plc_source = plc_source
        self.original = fresh_original
        self.original_boundary_query = original_query
        self.result = result
        self.domain = domain
        self.coordinate_contract = coordinate_contract
        self.source_id = source_id_
        self.source_revision = domain.source_revision
        self.region_id = region_id
        self.allocation_id = allocation_id
        self.offset, self.spacing, self.certificate_spacing, self.maximum_deviation = (
            controls
        )
        self.original_query_id = original_domain_query.source_scope_id
        self.scope_entity_id_banks = id_banks
        self.scope_entity_key_banks = key_banks
        self.scope_entity_set_ids = set_ids
        self.scope_entity_ids_by_mesh_row = row_ids
        self.edges = edges
        self.native_vertex_rows = native_vertex_rows
        self.native_vertex_entity_ids = vertex_entity_ids
        self.native_facet_rows = native_facet_rows
        self.native_facet_entity_ids = facet_entity_ids
        self.envelope_id = canonical_fingerprint(
            {
                "kind": "native-surface-envelope-source",
                "source": (source_id_, domain.source_revision),
                "original": (fresh_original.source_id, fresh_original.source_revision),
                "controls": controls,
                "entity_allocation": allocation_id,
                "native_vertex_lineage": array_tree_fingerprint(
                    (native_vertex_rows, vertex_entity_ids)
                ),
                "native_facet_lineage": array_tree_fingerprint(
                    (native_facet_rows, facet_entity_ids)
                ),
                "bounds": array_tree_fingerprint(result.bounds),
                "envelope": array_tree_fingerprint(
                    (result.surface_vertices, result.surface_triangles)
                ),
                "carrier": array_tree_fingerprint(
                    (result.volume_vertices, result.tetrahedra)
                ),
                "plc_carrier": plc_source.binding_id,
                "coordinate_contract": coordinate_contract.spatial_id,
            }
        )

    def scope_entity_ids(self, dimension: int, /) -> np.ndarray:
        if isinstance(dimension, (bool, np.bool_)) or not isinstance(dimension, Integral):
            raise TypeError("dimension must be one integer source dimension.")
        index = int(dimension)
        if index not in (0, 1, 2, 3):
            raise ValueError("Surface-envelope source dimensions must be in 0..3.")
        return self.scope_entity_id_banks[index]

    def scope_entity_set_id(self, dimension: int, /) -> str:
        if isinstance(dimension, (bool, np.bool_)) or not isinstance(dimension, Integral):
            raise TypeError("dimension must be one integer source dimension.")
        index = int(dimension)
        if index not in (0, 1, 2, 3):
            raise ValueError("Surface-envelope source dimensions must be in 0..3.")
        return self.scope_entity_set_ids[index]

    def source_entity_key_for_scope(
        self, dimension: int, entity_id: int, /
    ) -> tuple[str, tuple[str, ...]]:
        if isinstance(dimension, (bool, np.bool_)) or not isinstance(dimension, Integral):
            raise TypeError("dimension must be one integer source dimension.")
        index = int(dimension)
        if index not in (0, 1, 2, 3):
            raise ValueError("Surface-envelope source dimensions must be in 0..3.")
        if isinstance(entity_id, (bool, np.bool_)) or not isinstance(entity_id, Integral):
            raise TypeError("entity_id must be one stable integer scope ID.")
        identifiers = self.scope_entity_id_banks[index]
        position = int(np.searchsorted(identifiers, int(entity_id)))
        if position >= identifiers.size or int(identifiers[position]) != int(entity_id):
            raise ValueError(
                "entity_id is absent from this surface-envelope source dimension."
            )
        return self.scope_entity_key_banks[index][position]

    def source_entity_key_for_mesh_row(
        self, dimension: int, row: int, /
    ) -> tuple[str, tuple[str, ...]]:
        if isinstance(dimension, (bool, np.bool_)) or not isinstance(dimension, Integral):
            raise TypeError("dimension must be one integer source dimension.")
        index = int(dimension)
        if index not in (0, 1, 2, 3):
            raise ValueError("Surface-envelope source dimensions must be in 0..3.")
        if isinstance(row, (bool, np.bool_)) or not isinstance(row, Integral):
            raise TypeError("row must be one source row index.")
        row_index = int(row)
        bank = self.scope_entity_ids_by_mesh_row[index]
        if row_index < 0 or row_index >= bank.size:
            raise ValueError("row is outside the surface-envelope source stratum.")
        return self.source_entity_key_for_scope(index, int(bank[row_index]))

    def require_current(self, /) -> None:
        fresh_original = _fresh_original(self.original)
        source_vertices, source_triangles = _original_triangles(fresh_original)
        original_domain_query = _original_domain_query(fresh_original)
        try:
            fresh_result = surface_envelope(
                source_vertices,
                source_triangles,
                self.offset,
                self.spacing,
                self.certificate_spacing,
            )
        except MeshcoreError as error:
            raise ValueError(
                f"The retained native envelope no longer reproduces: {error}"
            ) from error
        if (
            not np.array_equal(fresh_result.bounds, self.result.bounds)
            or not np.array_equal(
                fresh_result.surface_vertices, self.result.surface_vertices
            )
            or not np.array_equal(
                fresh_result.surface_triangles, self.result.surface_triangles
            )
            or not np.array_equal(
                fresh_result.volume_vertices, self.result.volume_vertices
            )
            or not np.array_equal(fresh_result.tetrahedra, self.result.tetrahedra)
            or original_domain_query.source_scope_id != self.original_query_id
        ):
            raise ValueError(
                "The retained unsigned-distance envelope does not reproduce from its current source."
            )
        domain, domain_vertex_ids, domain_facet_ids = _envelope_domain(
            self.source_id,
            fresh_original,
            fresh_result,
            self.native_vertex_rows,
            self.native_facet_rows,
            self.native_vertex_entity_ids,
            self.native_facet_entity_ids,
            self.region_id,
        )
        fresh_plc_source, row_ids, edges = _envelope_plc_source(
            domain, domain_vertex_ids, domain_facet_ids
        )
        id_banks = fresh_plc_source.scope_entity_id_banks
        key_banks = fresh_plc_source.scope_entity_key_banks
        set_ids = fresh_plc_source.scope_entity_set_ids
        if (
            domain.domain_id != self.domain.domain_id
            or fresh_plc_source.binding_id != self.plc_source.binding_id
            or domain.source_revision != self.source_revision
            or domain.region_ids != (self.region_id,)
            or any(
                not np.array_equal(first, second)
                for first, second in zip(
                    id_banks, self.scope_entity_id_banks, strict=True
                )
            )
            or key_banks != self.scope_entity_key_banks
            or set_ids != self.scope_entity_set_ids
            or any(
                not np.array_equal(first, second)
                for first, second in zip(
                    row_ids, self.scope_entity_ids_by_mesh_row, strict=True
                )
            )
            or not np.array_equal(edges, self.edges)
        ):
            raise ValueError(
                "The derived envelope source identity, domain, or scopes no longer reproduce."
            )


def _mesh_scope(
    mesh: CellMesh, dimension: int, identifiers: np.ndarray, /
) -> MeshingScope:
    entity_set = mesh.entity_set(dimension)
    return MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        dimension,
        entity_set.entity_set_id,
        np.asarray(identifiers, dtype=np.int64),
    )


def _association_target_ids(
    source: SurfaceEnvelopeSource,
    construction: NativeMeshConstruction,
    dimension: int,
    source_entity_ids: np.ndarray,
    role: GeometrySourceEntityRole,
    /,
) -> np.ndarray:
    organization = construction.organization
    if organization is None:
        raise _failure(
            "The derived PLC fill omitted its source-identity associations.",
            category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            stage="source_witness",
        )
    entity_set_id = construction.mesh.entity_set(dimension).entity_set_id
    matches = tuple(
        value
        for value in organization.associations
        if value.target_entity_set_id == entity_set_id
    )
    if len(matches) != 1:
        raise _failure(
            "The derived PLC fill lacks one canonical association for a source stratum.",
            category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            stage="source_witness",
        )
    association = matches[0]
    target_set = construction.mesh.entity_set(dimension)
    target_ids = np.asarray(association.target_global_ids, dtype=np.int64)
    resolved = np.asarray(association.resolved, dtype=np.bool_)
    if (
        association.source_id != source.source_id
        or association.source_revision != source.source_revision
        or association.target_entity_set_id != target_set.entity_set_id
        or not np.array_equal(
            np.sort(target_ids),
            np.sort(np.asarray(target_set.entity_ids, dtype=np.int64)),
        )
        or not np.all(resolved)
        or np.any(np.asarray(association.ambiguous, dtype=np.bool_))
    ):
        raise _failure(
            "The derived PLC association does not cover this exact source revision and target stratum.",
            category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            stage="source_witness",
        )
    target_by_key: dict[tuple[str, tuple[str, ...]], list[int]] = {}
    for target, key, occurrence, source_role, is_resolved in zip(
        target_ids.tolist(),
        association.source_entity_ids,
        association.source_occurrence_paths,
        association.source_roles,
        resolved.tolist(),
        strict=True,
    ):
        if is_resolved and source_role is role:
            target_by_key.setdefault((key, occurrence), []).append(int(target))
    output = []
    for identifier in np.asarray(source_entity_ids, dtype=np.int64).tolist():
        key = source.source_entity_key_for_scope(dimension, int(identifier))
        candidates = target_by_key.get(key, ())
        if len(candidates) != 1:
            raise _failure(
                "A derived envelope entity does not have one identity-bound target entity.",
                category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                stage="source_witness",
            )
        output.append(candidates[0])
    return np.asarray(output, dtype=np.int64)


def _envelope_target_boundary(
    source: SurfaceEnvelopeSource,
    construction: NativeMeshConstruction,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    mesh = construction.mesh
    vertex_ids = _association_target_ids(
        source,
        construction,
        0,
        source.scope_entity_ids_by_mesh_row[0],
        GeometrySourceEntityRole.VERTEX,
    )
    facet_ids = _association_target_ids(
        source,
        construction,
        2,
        source.scope_entity_ids_by_mesh_row[2],
        GeometrySourceEntityRole.FACET,
    )
    target_vertex_rows = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(mesh.vertex_global_ids, dtype=np.int64).tolist()
        )
    }
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    for source_row, target_id in enumerate(vertex_ids.tolist()):
        target_row = target_vertex_rows.get(int(target_id))
        if target_row is None or not np.array_equal(
            coordinates[target_row],
            np.asarray(source.domain.vertices[source_row], dtype=np.float64),
        ):
            raise _failure(
                "The derived PLC fill changed an identity-bound envelope vertex.",
                category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                stage="source_witness",
            )
    _, face_rows = simplex_volume_entities(mesh)
    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    vertex_global_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    boundary_mask = np.asarray(mesh.boundary_masks[2], dtype=np.bool_)
    if face_rows.shape != (face_ids.size, 3) or boundary_mask.shape != face_ids.shape:
        raise _failure(
            "The derived PLC fill has malformed target boundary incidence.",
            category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            stage="source_witness",
        )
    row_by_face_id = {
        int(identifier): row for row, identifier in enumerate(face_ids.tolist())
    }
    target_face_vertices = {
        int(identifier): tuple(sorted(int(value) for value in vertex_global_ids[face]))
        for face, identifier in zip(face_rows, face_ids.tolist(), strict=True)
    }
    for source_row, facet_id in enumerate(facet_ids.tolist()):
        face_row = row_by_face_id.get(int(facet_id))
        if face_row is None or not boundary_mask[face_row]:
            raise _failure(
                "A fixed envelope facet is absent from the generated volume boundary.",
                category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                stage="source_witness",
            )
        expected = tuple(
            sorted(
                int(vertex_ids[int(vertex)])
                for vertex in source.domain.facets[source_row]
            )
        )
        if target_face_vertices[int(facet_id)] != expected:
            raise _failure(
                "A generated boundary facet changed its envelope vertex ancestry.",
                category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                stage="source_witness",
            )
    return vertex_ids, facet_ids


def _envelope_boundary_chart_witness(
    source: SurfaceEnvelopeSource,
    source_query: PiecewiseLinearBoundarySource,
    construction: NativeMeshConstruction,
    geometry: CellGeometrySpec,
    vertex_ids: np.ndarray,
    facet_ids: np.ndarray,
    /,
) -> SourceBoundaryChartWitness:
    """Bind each selected derived PWL facet to its exact target cell-reference face."""
    mesh = construction.mesh
    source_facet_ids = np.asarray(source.scope_entity_ids_by_mesh_row[2], dtype=np.int64)
    source_vertex_ids = np.asarray(vertex_ids, dtype=np.int64)
    target_facet_ids = np.asarray(facet_ids, dtype=np.int64)
    if (
        source_query.domain.domain_id != source.domain.domain_id
        or (source_query.source_id, source_query.source_revision)
        != (source.source_id, source.source_revision)
        or source_query.entity_dimension != 2
        or not set(source_query.entity_ids.tolist()) <= set(source_facet_ids.tolist())
        or source_facet_ids.shape != (source.domain.facets.shape[0],)
        or target_facet_ids.shape != source_facet_ids.shape
        or source_vertex_ids.shape != (source.domain.vertices.shape[0],)
    ):
        raise _failure(
            "The derived PWL boundary query lost its source-to-facet identity bank.",
            category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            stage="source_witness",
        )

    face_global_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    cell_global_ids = np.sort(np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64))
    _, cell_face_rows = volume_cell_entity_rows(mesh, cell_global_ids)
    boundary_mask = np.asarray(mesh.boundary_masks[2], dtype=np.bool_)
    mesh_vertex_global_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    if (
        cell_face_rows.ndim != 2
        or cell_face_rows.shape[0] != cell_global_ids.size
        or face_global_ids.ndim != 1
        or boundary_mask.shape != face_global_ids.shape
    ):
        raise _failure(
            "The envelope fill has incomplete target cell-to-face incidence.",
            category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            stage="source_witness",
        )
    cell_row_by_id = {
        int(identifier): row for row, identifier in enumerate(cell_global_ids.tolist())
    }
    target_faces: dict[int, tuple[int, int, np.ndarray, tuple[int, int, int]]] = {}
    for block in mesh.blocks:
        topology = reference_cell_topology(block.cell_kind)
        local_faces = tuple(
            tuple(int(value) for value in row) for row in topology.entities[2]
        )
        reference_vertices = np.asarray(topology.vertices, dtype=np.float64)
        block_vertices = np.asarray(block.vertices, dtype=np.int64)
        block_cell_ids = np.asarray(block.global_ids, dtype=np.int64)
        for block_row, raw_cell_id in enumerate(block_cell_ids.tolist()):
            cell_id = int(raw_cell_id)
            cell_position = cell_row_by_id.get(cell_id)
            if cell_position is None:
                raise _failure(
                    "An envelope target cell is absent from the canonical cell identity bank.",
                    category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                    stage="source_witness",
                )
            if len(local_faces) > cell_face_rows.shape[1]:
                raise _failure(
                    "The target cell-face incidence omits a local reference facet.",
                    category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                    stage="source_witness",
                )
            for local_face_index, local_face in enumerate(local_faces):
                face_row = int(cell_face_rows[cell_position, local_face_index])
                if face_row < 0 or face_row >= face_global_ids.size:
                    raise _failure(
                        "The target cell-face incidence names an absent face.",
                        category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                        stage="source_witness",
                    )
                if not boundary_mask[face_row]:
                    continue
                face_id = int(face_global_ids[face_row])
                if face_id in target_faces:
                    raise _failure(
                        "One target boundary face has multiple incident source-cell charts.",
                        category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                        stage="source_witness",
                    )
                local_vertex_rows = block_vertices[
                    block_row, np.asarray(local_face, dtype=np.int64)
                ]
                target_vertex_ids = tuple(
                    int(value)
                    for value in mesh_vertex_global_ids[local_vertex_rows].tolist()
                )
                if len(target_vertex_ids) != 3 or len(set(target_vertex_ids)) != 3:
                    raise _failure(
                        "An envelope target boundary chart must be one triangle.",
                        category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                        stage="source_witness",
                    )
                target_faces[face_id] = (
                    cell_id,
                    local_face_index,
                    reference_vertices[np.asarray(local_face, dtype=np.int64)],
                    target_vertex_ids,
                )

    expected_boundary_ids = {
        int(value) for value in face_global_ids[np.flatnonzero(boundary_mask)].tolist()
    }
    if set(target_faces) != expected_boundary_ids:
        raise _failure(
            "The envelope witness cannot cover every actual target boundary face.",
            category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            stage="source_witness",
        )

    target_face_by_source_id = {
        int(source_id): int(target_id)
        for source_id, target_id in zip(
            source_facet_ids.tolist(), target_facet_ids.tolist(), strict=True
        )
    }
    if len(target_face_by_source_id) != source_facet_ids.size:
        raise _failure(
            "Derived PWL facet identities are not unique.",
            category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            stage="source_witness",
        )

    standard_triangle = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    target_cell_ids: list[int] = []
    target_entity_ids: list[int] = []
    local_entity_indices: list[int] = []
    target_reference_rows: list[np.ndarray] = []
    target_reference_exact_rows = []
    source_chart_ids: list[str] = []
    source_entity_keys: list[str] = []
    source_occurrences: list[tuple[str, ...]] = []
    source_parameter_rows = []
    orientations: list[int] = []
    for source_entity_id, source_facet_row in zip(
        source_query.entity_ids.tolist(),
        source_query.source_rows.tolist(),
        strict=True,
    ):
        chart_id = source_query.chart_id_for_entity_id(int(source_entity_id))
        target_face_id = target_face_by_source_id[int(source_entity_id)]
        target_face = target_faces.get(target_face_id)
        if target_face is None:
            raise _failure(
                "A derived PWL facet has no identity-bound target boundary face.",
                category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                stage="source_witness",
            )
        target_cell_id, local_face_index, target_reference, target_vertex_ids = (
            target_face
        )
        source_facet_vertices = np.asarray(
            source.domain.facets[source_facet_row], dtype=np.int64
        )
        source_target_vertex_ids = tuple(
            int(source_vertex_ids[int(row)]) for row in source_facet_vertices.tolist()
        )
        source_local_by_target_id = {
            identifier: local for local, identifier in enumerate(source_target_vertex_ids)
        }
        if len(source_local_by_target_id) != 3:
            raise _failure(
                "A derived PWL facet has duplicate persistent vertex correspondences.",
                category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                stage="source_witness",
            )
        try:
            local_source_vertices = tuple(
                source_local_by_target_id[int(identifier)]
                for identifier in target_vertex_ids
            )
        except KeyError as error:
            raise _failure(
                "An output boundary face changes its exact PWL vertex ancestry.",
                category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                stage="source_witness",
            ) from error
        source_parameters = tuple(
            standard_triangle[index] for index in local_source_vertices
        )
        signed_area = (source_parameters[1][0] - source_parameters[0][0]) * (
            source_parameters[2][1] - source_parameters[0][1]
        ) - (source_parameters[1][1] - source_parameters[0][1]) * (
            source_parameters[2][0] - source_parameters[0][0]
        )
        if signed_area == 0:
            raise _failure(
                "An output boundary chart collapses in its exact PWL source triangle.",
                category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                stage="source_witness",
            )
        source_entity_key, occurrence_path = source_query.chart_source_key(2, chart_id)
        target_cell_ids.append(target_cell_id)
        target_entity_ids.append(target_face_id)
        local_entity_indices.append(local_face_index)
        target_reference_rows.append(target_reference)
        target_reference_exact_rows.append(
            tuple(
                tuple(Fraction.from_float(float(value)) for value in point)
                for point in target_reference
            )
        )
        source_chart_ids.append(chart_id)
        source_entity_keys.append(source_entity_key)
        source_occurrences.append(occurrence_path)
        source_parameter_rows.append(source_parameters)
        orientations.append(1 if signed_area > 0 else -1)

    expected_query_face_ids = (
        expected_boundary_ids
        if source_query.is_full_scope
        else {
            target_face_by_source_id[int(identifier)]
            for identifier in source_query.entity_ids.tolist()
        }
    )
    if set(target_entity_ids) != expected_query_face_ids:
        raise _failure(
            "The exact PWL witness adds or omits a target boundary face.",
            category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            stage="source_witness",
        )
    target_reference_values = np.asarray(target_reference_rows, dtype=np.float64)
    source_parameter_values = np.asarray(
        [
            [tuple(float(value) for value in point) for point in row]
            for row in source_parameter_rows
        ],
        dtype=np.float64,
    )
    witness = SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=source_query.source_id,
        source_revision=source_query.source_revision,
        source_query_id=source_query.query_id,
        chart_dimension=2,
        reference_dimension=3,
        cell_global_ids=np.asarray(target_cell_ids, dtype=np.int64),
        entity_global_ids=np.asarray(target_entity_ids, dtype=np.int64),
        local_entity_indices=np.asarray(local_entity_indices, dtype=np.int32),
        target_reference_vertices=target_reference_values,
        target_reference_rational_tokens=rational_chart_tokens(
            tuple(target_reference_exact_rows)
        ),
        source_chart_ids=tuple(source_chart_ids),
        source_entity_ids=tuple(source_entity_keys),
        occurrence_paths=tuple(source_occurrences),
        source_parameters=source_parameter_values,
        source_parameter_rational_tokens=rational_chart_tokens(
            tuple(source_parameter_rows)
        ),
        orientations=np.asarray(orientations, dtype=np.int8),
    )
    source_query.validate_chart_witness(witness)
    return witness


def _envelope_feature_witness(
    source: SurfaceEnvelopeSource,
    source_query: PiecewiseLinearBoundarySource,
    construction: NativeMeshConstruction,
    geometry: CellGeometrySpec,
    vertex_ids: np.ndarray,
    facet_ids: np.ndarray,
    /,
) -> SourceBoundaryPointWitness | SourceBoundaryChartWitness:
    """Bind a selected PWL point, edge, or face scope to actual target ancestry."""
    mesh = construction.mesh
    if source_query.entity_dimension == 0:
        rows = np.asarray(source_query.source_rows, dtype=np.int64)
        all_vertex_ids = np.asarray(vertex_ids, dtype=np.int64)
        if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= all_vertex_ids.size):
            raise _failure(
                "A protected PWL point query names absent retained source vertices.",
                category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                stage="source_witness",
            )
        witness = SourceBoundaryPointWitness(
            mesh_id=mesh.mesh_id,
            geometry_id=cell_geometry_id(geometry),
            topology_id=mesh.topology_id,
            source_id=source_query.source_id,
            source_revision=source_query.source_revision,
            source_query_id=source_query.query_id,
            vertex_global_ids=all_vertex_ids[rows],
            source_entity_ids=source_query.source_entity_keys,
            occurrence_paths=source_query.source_occurrence_paths,
        )
        witness.require_current(mesh, geometry, source_query)
        return witness
    if source_query.entity_dimension == 2:
        return _envelope_boundary_chart_witness(
            source, source_query, construction, geometry, vertex_ids, facet_ids
        )
    if source_query.entity_dimension != 1:
        raise _failure(
            "Protected envelope source features must name PWL point, curve, or surface strata.",
            category=MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            stage="source_witness",
        )

    source_edge_ids = np.asarray(source.scope_entity_ids_by_mesh_row[1], dtype=np.int64)
    target_edge_ids = _association_target_ids(
        source,
        construction,
        1,
        source_edge_ids,
        GeometrySourceEntityRole.EDGE,
    )
    mesh_edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    mesh_edge_rows = {
        int(identifier): row for row, identifier in enumerate(mesh_edge_ids.tolist())
    }
    boundary_edges = np.asarray(mesh.boundary_masks[1], dtype=np.bool_)
    cell_ids = np.sort(np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64))
    cell_row_by_id = {
        int(identifier): row for row, identifier in enumerate(cell_ids.tolist())
    }
    cell_edge_rows, _ = volume_cell_entity_rows(mesh, cell_ids)
    if (
        boundary_edges.shape != mesh_edge_ids.shape
        or cell_edge_rows.ndim != 2
        or cell_edge_rows.shape[0] != cell_ids.size
    ):
        raise _failure(
            "The envelope fill omitted canonical target edge incidence.",
            category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            stage="source_witness",
        )
    target_edge_charts: dict[int, tuple[int, int, np.ndarray, tuple[int, int]]] = {}
    mesh_vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    for block in mesh.blocks:
        topology = reference_cell_topology(block.cell_kind)
        local_edges = tuple(
            tuple(int(value) for value in row) for row in topology.entities[1]
        )
        reference_vertices = np.asarray(topology.vertices, dtype=np.float64)
        block_vertices = np.asarray(block.vertices, dtype=np.int64)
        for block_row, raw_cell_id in enumerate(
            np.asarray(block.global_ids, dtype=np.int64).tolist()
        ):
            cell_id = int(raw_cell_id)
            cell_position = cell_row_by_id.get(cell_id)
            if cell_position is None or len(local_edges) > cell_edge_rows.shape[1]:
                raise _failure(
                    "An envelope cell lost its canonical local-edge ancestry.",
                    category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                    stage="source_witness",
                )
            for local_edge_index, local_edge in enumerate(local_edges):
                edge_row = int(cell_edge_rows[cell_position, local_edge_index])
                if (
                    edge_row < 0
                    or edge_row >= mesh_edge_ids.size
                    or not boundary_edges[edge_row]
                ):
                    continue
                edge_id = int(mesh_edge_ids[edge_row])
                local_vertex_rows = block_vertices[
                    block_row, np.asarray(local_edge, dtype=np.int64)
                ]
                target_vertices = tuple(
                    int(value) for value in mesh_vertex_ids[local_vertex_rows].tolist()
                )
                if len(target_vertices) != 2 or target_vertices[0] == target_vertices[1]:
                    raise _failure(
                        "An envelope target edge chart must have two distinct persistent vertices.",
                        category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                        stage="source_witness",
                    )
                candidate = (
                    cell_id,
                    local_edge_index,
                    reference_vertices[np.asarray(local_edge, dtype=np.int64)],
                    target_vertices,
                )
                previous = target_edge_charts.get(edge_id)
                if previous is None or candidate[:2] < previous[:2]:
                    target_edge_charts[edge_id] = candidate

    target_cell_ids: list[int] = []
    target_entity_ids: list[int] = []
    local_entity_indices: list[int] = []
    target_reference_rows: list[np.ndarray] = []
    target_reference_exact_rows = []
    source_chart_ids: list[str] = []
    source_entity_keys: list[str] = []
    source_occurrences: list[tuple[str, ...]] = []
    source_parameter_rows = []
    orientations: list[int] = []
    source_edges = np.asarray(source.edges, dtype=np.int64)
    for source_id, source_row in zip(
        source_query.entity_ids.tolist(), source_query.source_rows.tolist(), strict=True
    ):
        target_edge_id = int(target_edge_ids[int(source_row)])
        target_edge_row = mesh_edge_rows.get(target_edge_id)
        chart = target_edge_charts.get(target_edge_id)
        if (
            target_edge_row is None
            or not boundary_edges[target_edge_row]
            or chart is None
        ):
            raise _failure(
                "A protected PWL curve lacks an identity-bound target boundary edge chart.",
                category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                stage="source_witness",
            )
        cell_id, local_edge_index, target_reference, target_vertices = chart
        source_vertex_rows = source_edges[int(source_row)]
        source_target_vertices = tuple(
            int(vertex_ids[int(row)]) for row in source_vertex_rows.tolist()
        )
        if len(set(source_target_vertices)) != 2 or set(source_target_vertices) != set(
            target_vertices
        ):
            raise _failure(
                "A protected PWL curve changed its exact source vertex ancestry.",
                category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                stage="source_witness",
            )
        source_parameter_by_vertex = {
            source_target_vertices[0]: Fraction(0),
            source_target_vertices[1]: Fraction(1),
        }
        source_parameters = tuple(
            (source_parameter_by_vertex[int(vertex)],) for vertex in target_vertices
        )
        chart_id = source_query.chart_id_for_entity_id(int(source_id))
        entity_key, occurrence_path = source_query.chart_source_key(1, chart_id)
        target_cell_ids.append(cell_id)
        target_entity_ids.append(target_edge_id)
        local_entity_indices.append(local_edge_index)
        target_reference_rows.append(target_reference)
        target_reference_exact_rows.append(
            tuple(
                tuple(Fraction.from_float(float(value)) for value in point)
                for point in target_reference
            )
        )
        source_chart_ids.append(chart_id)
        source_entity_keys.append(entity_key)
        source_occurrences.append(occurrence_path)
        source_parameter_rows.append(source_parameters)
        orientations.append(
            1 if source_parameters[1][0] > source_parameters[0][0] else -1
        )

    witness = SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=source_query.source_id,
        source_revision=source_query.source_revision,
        source_query_id=source_query.query_id,
        chart_dimension=1,
        reference_dimension=3,
        cell_global_ids=np.asarray(target_cell_ids, dtype=np.int64),
        entity_global_ids=np.asarray(target_entity_ids, dtype=np.int64),
        local_entity_indices=np.asarray(local_entity_indices, dtype=np.int32),
        target_reference_vertices=np.asarray(target_reference_rows, dtype=np.float64),
        target_reference_rational_tokens=rational_chart_tokens(
            tuple(target_reference_exact_rows)
        ),
        source_chart_ids=tuple(source_chart_ids),
        source_entity_ids=tuple(source_entity_keys),
        occurrence_paths=tuple(source_occurrences),
        source_parameters=np.asarray(
            [
                [tuple(float(value) for value in point) for point in row]
                for row in source_parameter_rows
            ],
            dtype=np.float64,
        ),
        source_parameter_rational_tokens=rational_chart_tokens(
            tuple(source_parameter_rows)
        ),
        orientations=np.asarray(orientations, dtype=np.int8),
    )
    source_query.validate_chart_witness(witness)
    return witness


def _tetrahedron_quality(
    coordinates: np.ndarray, tetrahedra: np.ndarray, /
) -> tuple[float, float]:
    points = np.asarray(coordinates, dtype=np.float64)[
        np.asarray(tetrahedra, dtype=np.int64)
    ]
    edge_pairs = np.asarray(
        ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)), dtype=np.int64
    )
    edge_vectors = points[:, edge_pairs[:, 1]] - points[:, edge_pairs[:, 0]]
    edge_lengths = np.linalg.norm(edge_vectors, axis=2)
    minimum_edges = np.min(edge_lengths, axis=1)
    if (
        points.ndim != 3
        or points.shape[1:] != (4, 3)
        or np.any(~np.isfinite(edge_lengths))
        or np.any(minimum_edges <= 0.0)
    ):
        raise _failure(
            "Native envelope tetrahedra must have finite nonzero edges.",
            category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            stage="specification_compliance",
        )
    offsets = points[:, 1:] - points[:, :1]
    # Circumcenter offset (|a|^2 b x c + |b|^2 c x a + |c|^2 a x b) / (2 a . b x c);
    # a tetrahedron whose binary64 determinant vanishes has an unbounded ratio.
    first, second, third = offsets[:, 0], offsets[:, 1], offsets[:, 2]
    determinant = 2.0 * np.sum(first * np.cross(second, third), axis=1)
    numerator = (
        np.sum(first * first, axis=1)[:, None] * np.cross(second, third)
        + np.sum(second * second, axis=1)[:, None] * np.cross(third, first)
        + np.sum(third * third, axis=1)[:, None] * np.cross(first, second)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.linalg.norm(numerator, axis=1) / (np.abs(determinant) * minimum_edges)
    ratios = np.where((determinant != 0.0) & np.isfinite(ratios), ratios, np.inf)
    radius_edge = float(np.max(ratios, initial=0.0))
    face_vertices = np.asarray(
        ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), dtype=np.int64
    )
    triangles = points[:, face_vertices]
    normals = np.cross(
        triangles[:, :, 1] - triangles[:, :, 0], triangles[:, :, 2] - triangles[:, :, 0]
    )
    face_centers = np.mean(triangles, axis=2)
    cell_centers = np.mean(points, axis=1)
    inward = np.sum(normals * (face_centers - cell_centers[:, None, :]), axis=2) < 0.0
    normals = np.where(inward[:, :, None], -normals, normals)
    lengths = np.linalg.norm(normals, axis=2)
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= 0.0):
        raise _failure(
            "Native envelope tetrahedra must have finite face normals.",
            category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            stage="specification_compliance",
        )
    normals /= lengths[:, :, None]
    face_pairs = np.asarray(
        ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)), dtype=np.int64
    )
    cosines = np.sum(normals[:, face_pairs[:, 0]] * normals[:, face_pairs[:, 1]], axis=2)
    dihedral = 180.0 - np.rad2deg(np.arccos(np.clip(cosines, -1.0, 1.0)))
    return radius_edge, float(np.min(dihedral))


def _compliance(
    source: SurfaceEnvelopeSource,
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    construction: NativeMeshConstruction,
    /,
    soft_feature_outcomes: tuple[tuple[ProtectedFeature, float, bool], ...] = (),
) -> MeshingComplianceReport:
    base = construction.compliance
    if not isinstance(base, MeshingComplianceReport):
        raise TypeError("The derived PLC fill must retain its native compliance report.")
    coordinates = np.asarray(construction.mesh.coordinates, dtype=np.float64)
    tetrahedra = np.asarray(construction.mesh.blocks[0].vertices, dtype=np.int64)
    radius_edge, minimum_dihedral = _tetrahedron_quality(coordinates, tetrahedra)
    source_deviation = max(float(source.result.bounds[0]), float(source.result.bounds[1]))
    requested = list(base.requested)
    requested.extend(
        (
            ("radius_edge_bound", schedule.radius_edge_bound),
            ("minimum_dihedral_degrees", schedule.minimum_dihedral_degrees),
            ("source_envelope_maximum_deviation", source.maximum_deviation),
        )
    )
    achieved = list(base.achieved)
    achieved.extend(
        (
            ("maximum_radius_edge_ratio", radius_edge),
            ("minimum_dihedral_degrees", minimum_dihedral),
            ("source_envelope_maximum_deviation", source_deviation),
            ("source_envelope_input_to_envelope_upper", float(source.result.bounds[0])),
            ("source_envelope_envelope_to_input_upper", float(source.result.bounds[1])),
            ("source_envelope_envelope_to_input_lower", float(source.result.bounds[2])),
        )
    )
    if isinstance(source.original, AuthoredTriangleSheet):
        achieved.extend(
            (f"source_topology_repair:{name}", float(value))
            for name, value in source.original.repair.quantities
        )
    for feature, deviation_upper, warning in soft_feature_outcomes:
        requested.append(
            (
                f"soft_feature:{feature.feature_id}:maximum_deviation",
                feature.maximum_deviation,
            )
        )
        achieved.extend(
            (
                (f"soft_feature:{feature.feature_id}:deviation_upper", deviation_upper),
                (f"soft_feature:{feature.feature_id}:warning", float(warning)),
            )
        )
    issues = list(base.issues)
    if radius_edge > schedule.radius_edge_bound:
        issues.append("radius_edge_bound")
    if minimum_dihedral < schedule.minimum_dihedral_degrees:
        issues.append("minimum_dihedral_degrees")
    if source_deviation > source.maximum_deviation:
        issues.append("source_envelope_maximum_deviation")
    if specification.maximum_deviation is not None:
        requested.append(
            ("derived_envelope_maximum_deviation", specification.maximum_deviation)
        )
        achieved.append(("derived_envelope_maximum_deviation", 0.0))
    return MeshingComplianceReport(
        specification.specification_id,
        issues=tuple(dict.fromkeys(issues)),
        requested=tuple(requested),
        achieved=tuple(achieved),
    )


def _organization(
    source: SurfaceEnvelopeSource,
    mesh: CellMesh,
    boundary_facet_ids: np.ndarray,
    plc_organization: NativeMeshOrganization,
    /,
) -> NativeMeshOrganization:
    domain = source.domain
    boundary_faces = np.asarray(domain.facets, dtype=np.int64)
    boundary_vertex_count = np.asarray(domain.vertices).shape[0]
    boundary = SurfaceModel.from_triangles(
        np.asarray(domain.vertices, dtype=np.float64),
        boundary_faces,
        SurfaceMetadata(
            source_id=source.source_id,
            source_revision=source.source_revision,
            coordinate_contract=source.coordinate_contract,
            provenance=(
                "native_unsigned_distance_surface_envelope",
                source.original.binding_id,
                source.original.source_revision,
                source.envelope_id,
            ),
            cell_tags=("unsigned-distance-envelope",) * boundary_faces.shape[0],
        ),
        vertex_global_ids=np.asarray(mesh.vertex_global_ids, dtype=np.int64)[
            :boundary_vertex_count
        ],
        cell_global_ids=np.asarray(boundary_facet_ids, dtype=np.int64),
        numeric_version=source.source_revision,
    )
    cell_scope = _mesh_scope(
        mesh, 3, np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
    )
    labels = (
        MeshLabel(domain.region_ids[0], cell_scope),
        *(label for label in plc_organization.labels if label.name != "internal_curve"),
        MeshLabel(
            "unsigned-distance-envelope",
            _mesh_scope(mesh, 2, np.asarray(boundary_facet_ids, dtype=np.int64)),
        ),
    )
    return NativeMeshOrganization(
        boundary=boundary,
        patches=plc_organization.patches,
        zones=plc_organization.zones,
        labels=labels,
        attributes=plc_organization.attributes,
        associations=plc_organization.associations,
    )


@final
class _PreparedSurfaceEnvelopeRoute(StrictModule, NonTrainableState):
    source: SurfaceEnvelopeSource
    specification: VolumeMeshingSpec
    options: NativeMeshingOptions
    coordinate_contract: SpatialCoordinateContract
    volume_route: PreparedPlcVolume
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: SurfaceEnvelopeSource,
        specification: VolumeMeshingSpec,
        options: NativeMeshingOptions,
        coordinate_contract: SpatialCoordinateContract,
        volume_route: PreparedPlcVolume,
        /,
    ) -> None:
        self.source = source
        self.specification = specification
        self.options = options
        self.coordinate_contract = coordinate_contract
        self.volume_route = volume_route
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "native-surface-envelope-preparation",
                "source": source.envelope_id,
                "specification": specification.specification_id,
                "options": options.options_id,
                "coordinate_contract": coordinate_contract.spatial_id,
                "volume_route": volume_route.prepared_id,
            }
        )

    def execute(self) -> NativeMeshConstruction:
        source = self.source
        source_query = PiecewiseLinearBoundarySource(source, 2)
        if source.coordinate_contract.spatial_id != self.coordinate_contract.spatial_id:
            raise _failure(
                "The prepared coordinate contract no longer matches the envelope source.",
                category=MeshingFailureCategory.INVALID_SOURCE,
                stage="source_validation",
            )
        construction = self.volume_route.execute()
        mesh = construction.mesh
        geometry = construction.geometry
        if geometry is None or construction.organization is None:
            raise _failure(
                "The derived PLC fill omitted geometry or source organization.",
                category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                stage="volume_fill",
            )
        vertex_ids, facet_ids = _envelope_target_boundary(source, construction)
        derived_witness = _envelope_boundary_chart_witness(
            source,
            source_query,
            construction,
            geometry,
            vertex_ids,
            facet_ids,
        )
        distance_witness = SurfaceEnvelopeDistanceWitness(
            mesh_id=mesh.mesh_id,
            geometry_id=cell_geometry_id(geometry),
            topology_id=mesh.topology_id,
            source_id=source.original.source_id,
            source_revision=source.original.source_revision,
            source_query_id=source.original_query_id,
            offset=source.offset,
            spacing=source.spacing,
            certificate_spacing=source.certificate_spacing,
            bounds=np.asarray(source.result.bounds, dtype=np.float64),
            envelope_vertices=np.asarray(source.domain.vertices, dtype=np.float64),
            envelope_triangles=np.asarray(source.domain.facets, dtype=np.int64),
            target_vertex_global_ids=vertex_ids,
            target_facet_global_ids=facet_ids,
        )
        try:
            distance_witness.require_current(
                mesh, geometry, source.original_boundary_query
            )
        except (TypeError, ValueError, IndexError, ArithmeticError) as error:
            raise _failure(
                f"The original-source envelope witness did not revalidate: {error}",
                category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                stage="source_witness",
            ) from error
        feature_scoped_fidelity = []
        soft_feature_outcomes = []
        for feature in self.specification.protected_features:
            feature_query = PiecewiseLinearBoundarySource(
                source,
                feature.scope.entity_dimension,
                entity_ids=feature.scope.entity_ids,
            )
            feature_witness = _envelope_feature_witness(
                source,
                feature_query,
                construction,
                geometry,
                vertex_ids,
                facet_ids,
            )
            if feature.hard:
                feature_scoped_fidelity.append(
                    (feature_query, feature_witness, feature.maximum_deviation)
                )
                continue
            certificate = source_fidelity_owner.certify_source_fidelity(
                mesh,
                geometry,
                feature_query,
                tolerance=feature.maximum_deviation,
                witness=feature_witness,
            )
            deviation_upper = max(
                certificate.mesh_to_source_upper,
                certificate.source_to_mesh_upper,
            )
            warning = (
                certificate.status != "certified"
                or deviation_upper > feature.maximum_deviation
            )
            soft_feature_outcomes.append((feature, deviation_upper, warning))
        base_request = construction.certification
        if base_request.cell_regions is None:
            raise _failure(
                "The derived PLC fill omitted its actual volume region assignments.",
                category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                stage="volume_fill",
            )
        certification = NativeCertificationRequest(
            MeshCertificationSchedule("volume_plc"),
            source.source_id,
            source.source_revision,
            domain=source.domain,
            cell_regions=np.asarray(base_request.cell_regions, dtype=np.int64),
            fidelity_source=source_query,
            fidelity_tolerance=0.0,
            fidelity_witness=derived_witness,
            protected_features=self.specification.protected_features,
            scoped_fidelity=(
                (
                    source.original_boundary_query,
                    distance_witness,
                    source.maximum_deviation,
                ),
                *feature_scoped_fidelity,
            ),
            feature_source=source if self.specification.protected_features else None,
        )
        schedule = self.options.volume_schedule
        if not isinstance(schedule, NativeVolumeSchedule):
            raise _failure(
                "surface-envelope filling requires a resolved native volume schedule.",
                category=MeshingFailureCategory.INVALID_SPECIFICATION,
                stage="specification_inspection",
            )
        compliance = _compliance(
            source,
            self.specification,
            schedule,
            construction,
            soft_feature_outcomes=tuple(soft_feature_outcomes),
        )
        organization = _organization(
            source,
            mesh,
            facet_ids,
            construction.organization,
        )
        return NativeMeshConstruction(
            mesh,
            self.coordinate_contract,
            compliance,
            certification,
            geometry=geometry,
            organization=organization,
            stages=construction.stages,
        )


def _support_issues(
    source: SurfaceEnvelopeSource,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> list[str]:
    if not isinstance(source, SurfaceEnvelopeSource):
        return [
            "an explicit SurfaceEnvelopeSource over a represented ReconstructedGeometrySource or AuthoredTriangleSheet"
        ]
    if not isinstance(specification, VolumeMeshingSpec):
        return ["a VolumeMeshingSpec"]
    issues: list[str] = []
    if (
        not isinstance(options, NativeMeshingOptions)
        or options.route != "surface_envelope_tetrahedral"
    ):
        issues.append("the surface_envelope_tetrahedral native route")
    if not isinstance(coordinate_contract, SpatialCoordinateContract):
        issues.append("a SpatialCoordinateContract")
    elif (
        coordinate_contract.coordinate_system != "cartesian"
        or coordinate_contract.spatial_id != source.coordinate_contract.spatial_id
    ):
        issues.append("the retained orthonormal Cartesian coordinate contract")
    issues.extend(volume_support_issues(source.plc_source, specification))
    for feature in specification.protected_features:
        scope = feature.scope
        dimension = scope.entity_dimension
        expected_dimension = {
            "corner": 0,
            "curve": 1,
            "surface": 2,
            "material_interface": 2,
        }.get(feature.feature_kind.value)
        if dimension not in (0, 1, 2) or expected_dimension != dimension:
            issues.append(
                f"protected feature {feature.feature_id} has an unsupported PWL stratum"
            )
            continue
        if scope.entity_kind is not MeshingEntityKind.GEOMETRY or (
            scope.source_id,
            scope.source_revision,
            scope.entity_set_id,
        ) != (
            source.source_id,
            source.source_revision,
            source.scope_entity_set_id(dimension),
        ):
            issues.append(
                f"protected feature {feature.feature_id} changes envelope source identity"
            )
            continue
        try:
            PiecewiseLinearBoundarySource(source, dimension, entity_ids=scope.entity_ids)
        except (TypeError, ValueError) as error:
            issues.append(
                f"protected feature {feature.feature_id} is outside the envelope source scope: {error}"
            )
    if specification.layer_controls or specification.periodic_constraints:
        issues.append(
            "boundary layers or periodic constraints outside the native PLC fill route"
        )
    if any(
        isinstance(control, UniformSizeControl)
        and control.maximum_growth_rate is not None
        for control in specification.size_controls
    ):
        issues.append(
            "maximum-growth controls without a source-geodesic envelope size field"
        )
    if len(source.domain.region_ids) != 1:
        issues.append("one exact derived PWL envelope region")
    if not isinstance(options.volume_schedule, NativeVolumeSchedule):
        issues.append("a resolved NativeVolumeSchedule")
    return list(dict.fromkeys(issues))


@checked
def prepare_native_surface_envelope(
    source: SurfaceEnvelopeSource,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> _PreparedSurfaceEnvelopeRoute:
    """Prepare a native constrained fill against the derived envelope PLC."""
    if not isinstance(source, SurfaceEnvelopeSource):
        raise TypeError("source must be SurfaceEnvelopeSource.")
    if not isinstance(specification, VolumeMeshingSpec):
        raise TypeError("specification must be VolumeMeshingSpec.")
    if not isinstance(options, NativeMeshingOptions):
        raise TypeError("options must be NativeMeshingOptions.")
    if not isinstance(coordinate_contract, SpatialCoordinateContract):
        raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
    issues = _support_issues(source, specification, options, coordinate_contract)
    if issues:
        raise _failure(
            "; ".join(issues),
            category=MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            stage="specification_inspection",
        )
    source.require_current()
    schedule = options.volume_schedule
    if not isinstance(schedule, NativeVolumeSchedule):
        raise _failure(
            "surface-envelope filling requires a resolved native volume schedule.",
            category=MeshingFailureCategory.INVALID_SPECIFICATION,
            stage="specification_inspection",
        )
    volume_route = PreparedPlcVolume(
        source.plc_source,
        specification,
        coordinate_contract,
        schedule,
    )
    return _PreparedSurfaceEnvelopeRoute(
        source,
        specification,
        options,
        coordinate_contract,
        volume_route,
    )


__all__ = ["SurfaceEnvelopeSource", "prepare_native_surface_envelope"]
