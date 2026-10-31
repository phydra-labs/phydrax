#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from numbers import Integral
from typing import (
    assert_never,
    Literal,
    NamedTuple,
    NoReturn,
    Protocol,
    runtime_checkable,
    TYPE_CHECKING,
    TypeAlias,
)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._mass import Mass
from ..._meshcore import delaunay_2d
from ..._trainable import NonTrainableState
from ..._validation import positive_finite_float, positive_integer
from ...measurement.lidar import LidarPointProduct
from ...typing import checked, parse, PRNGKey
from .._capabilities import (
    ClosestPointProvider,
    ContactCurvatureProvider,
    GeometryCapability,
    SeamDiagnosticsProvider,
    SupportMapProvider,
)
from .._certificate import FieldCertificate
from .._contracts import (
    ClosestPointResult,
    ContactCurvatureResult,
    GeometryKernel,
    GeometryKind,
    GeometrySource,
)
from .._cubature import CubatureAtlas, CubatureComponent
from .._meshing_domain import MeshingDomain
from .._validity import GeometryValidityEvidence, representation_validity
from ..brep._patches import AbstractSurfacePatch, LineCurve
from ..design._schema import _ParameterCollector, DesignState
from ..simplicial._io import (
    _canonical_triangle_arrays,
    planar_region_from_triangles,
)
from ..simplicial._regions import MeshRegion, PlanarMeshRegion
from ..simplicial._topology import TriangleTopology
from ._normals import (
    estimate_point_normals,
    NormalEstimationEvidence,
    NormalOrientation,
)
from ._poisson import (
    extract_indicator_surface,
    PoissonDiscretization,
    PoissonSolveEvidence,
    sample_areas,
    sampled_surface_deviation,
    SampledSurfaceDeviation,
    solve_regular_poisson,
)
from ._poisson_octree import solve_octree_poisson
from ._robustness import (
    component_fit,
    ComponentFitEvidence,
    OutlierRemovalEvidence,
    ReconstructionRobustness,
    remove_statistical_outliers,
    sampling_coverage,
    SamplingCoverageEvidence,
    thin_features,
    ThinFeatureEvidence,
)


_SPACING_PER_SAMPLE = 2.0
_PADDING_CELLS = 4
_RESOLUTION_CELLS = 2.0
_DEFAULT_ROBUSTNESS = ReconstructionRobustness()
RepairInterpretation: TypeAlias = Literal[
    "sample_supported_poisson_fit",
    "poisson_inferred_closure",
    "incomplete_sampling_refused",
    "authored_sheet_topology_repair",
]
"""Interpretation of closure beyond observed point-cloud coverage.

``authored_sheet_topology_repair`` is a geometry-preserving topology repair of
an authored open triangle sheet (exact vertex welding, exact duplicate removal,
per-component orientation); its boundary stays open and any closure is an
explicit derived wrap such as a surface envelope.
"""


if TYPE_CHECKING:
    from .._atlas import BoundaryAtlas
    from .._sampling import RejectionSamplingPlan, SamplingResult


@dataclass(frozen=True, slots=True)
class ReconstructionReport:
    """Provenance, topology, fit, coverage, and approximation evidence.

    ``repair_interpretation`` distinguishes a sample-supported fit from a
    Poisson-implied completion of regions outside the measured coverage. It is
    not a claim that the inferred patch was observed or physically repaired.
    """

    source_kind: str
    algorithm: str
    input_digest: str
    input_points: int
    retained_points: int
    output_vertices: int
    output_cells: int
    connected_components: int
    watertight: bool
    winding_consistent: bool
    recenter_offset: tuple[float, ...]
    parameters: tuple[tuple[str, str], ...]
    warnings: tuple[str, ...] = ()
    source_product_id: str | None = None
    euler_characteristic: int | None = None
    normal_evidence: NormalEstimationEvidence | None = None
    poisson_evidence: PoissonSolveEvidence | None = None
    deviation_evidence: SampledSurfaceDeviation | None = None
    outlier_evidence: OutlierRemovalEvidence | None = None
    thin_feature_evidence: ThinFeatureEvidence | None = None
    component_evidence: ComponentFitEvidence | None = None
    coverage_evidence: SamplingCoverageEvidence | None = None
    repair_interpretation: RepairInterpretation | None = None


@runtime_checkable
class ReconstructionReportProvider(Protocol):
    report: ReconstructionReport


class ReconstructionFailure(ValueError):
    """Reconstruction failure retaining the diagnostics produced before rejection."""

    report: ReconstructionReport

    def __init__(self, message: str, report: ReconstructionReport) -> None:
        super().__init__(message)
        self.report = report


class ReconstructedGeometrySourceArchiveRecipe(NamedTuple):
    """Portable source inputs and persistent topology IDs for a reconstruction.

    As a NamedTuple the recipe is a PyTree, so archives inventory its mesh and ID
    arrays as dynamic leaves. Restores validate the IDs by rebuilding the owning
    ``ReconstructedGeometrySource`` and comparing its expected identity.
    """

    source: GeometrySource
    report: ReconstructionReport
    persistent_vertex_ids: np.ndarray
    persistent_edge_ids: np.ndarray
    persistent_facet_ids: np.ndarray
    persistent_region_id: int | None
    expected_source_id: str
    expected_source_revision: str


def _persistent_int64_id(payload: object, /) -> int:
    return int(canonical_fingerprint(payload)[:16], 16) & np.iinfo(np.int64).max


def _persistent_id_vector(
    values: ArrayLike | None,
    count: int,
    source_id: str,
    kind: str,
    /,
) -> np.ndarray:
    if values is None:
        result = np.asarray(
            [
                _persistent_int64_id(
                    {
                        "kind": "reconstructed-source-entity",
                        "source_id": source_id,
                        "entity_kind": kind,
                        "creation_ordinal": ordinal,
                    }
                )
                for ordinal in range(count)
            ],
            dtype=np.int64,
        )
    else:
        raw = np.asarray(values)
        if raw.shape != (count,) or raw.dtype.kind not in "iu":
            raise ValueError(f"{kind} persistent IDs must be an aligned integer vector.")
        if raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
            raise ValueError(f"{kind} persistent IDs must fit signed int64.")
        result = np.array(raw, dtype=np.int64, copy=True)
    if np.any(result < 0) or np.unique(result).size != count:
        raise ValueError(f"{kind} persistent IDs must be unique non-negative integers.")
    result.setflags(write=False)
    return result


def _reconstructed_mesh_identity_records(
    source: GeometrySource,
    source_id: str,
    /,
    *,
    vertex_ids: ArrayLike | None,
    edge_ids: ArrayLike | None,
    facet_ids: ArrayLike | None,
    region_id: int | None,
) -> tuple[tuple[np.ndarray, ...], int | None, np.ndarray | None]:
    if not isinstance(source, MeshRegion):
        if any(
            value is not None for value in (vertex_ids, edge_ids, facet_ids, region_id)
        ):
            raise ValueError(
                "Persistent mesh entity IDs require a reconstructed MeshRegion."
            )
        empty = np.empty((0,), dtype=np.int64)
        empty.setflags(write=False)
        return (empty, empty, empty), None, None

    faces = np.asarray(source.faces, dtype=np.int64)
    vertex_count = int(np.asarray(source.vertices).shape[0])
    edges = np.asarray(source.triangle_mesh.topology.edges, dtype=np.int64)
    vertices_ = _persistent_id_vector(
        vertex_ids, vertex_count, source_id, "reconstructed_vertex"
    )
    if edge_ids is None:
        edge_values = np.asarray(
            [
                _persistent_int64_id(
                    {
                        "kind": "reconstructed-edge-entity",
                        "source_id": source_id,
                        "vertices": tuple(
                            sorted((int(vertices_[first]), int(vertices_[last])))
                        ),
                    }
                )
                for first, last in edges.tolist()
            ],
            dtype=np.int64,
        )
        edge_values.setflags(write=False)
    else:
        edge_values = _persistent_id_vector(
            edge_ids, int(edges.shape[0]), source_id, "reconstructed_edge"
        )
    if np.unique(edge_values).size != edge_values.size:
        raise ValueError("Reconstructed edge persistent IDs collide.")
    if facet_ids is None:
        facet_values = np.asarray(
            [
                _persistent_int64_id(
                    {
                        "kind": "reconstructed-facet-entity",
                        "source_id": source_id,
                        "vertices": tuple(
                            sorted(int(vertices_[int(vertex)]) for vertex in face)
                        ),
                    }
                )
                for face in faces.tolist()
            ],
            dtype=np.int64,
        )
        facet_values.setflags(write=False)
    else:
        facet_values = _persistent_id_vector(
            facet_ids, int(faces.shape[0]), source_id, "reconstructed_facet"
        )
    if np.unique(facet_values).size != facet_values.size:
        raise ValueError("Reconstructed facet persistent IDs collide.")
    if region_id is None:
        region_id_ = _persistent_int64_id(
            {"kind": "reconstructed-region-entity", "source_id": source_id}
        )
    else:
        if isinstance(region_id, (bool, np.bool_)) or not isinstance(
            region_id, (int, np.integer)
        ):
            raise TypeError("region_id must be one persistent integer ID.")
        region_id_ = int(region_id)
        if region_id_ < 0 or region_id_ > np.iinfo(np.int64).max:
            raise ValueError("region_id must fit a non-negative signed int64.")
    return (vertices_, edge_values, facet_values), region_id_, edges


def _canonical_oriented_cycle(values: tuple[int, int, int], /) -> tuple[int, int, int]:
    rotations = (values, values[1:] + values[:1], values[2:] + values[:2])
    return min(rotations)


def _reconstructed_source_geometry_payload(
    source: GeometrySource,
    entity_ids: tuple[np.ndarray, ...],
    region_id: int | None,
    /,
) -> object:
    if not isinstance(source, MeshRegion) or region_id is None:
        return array_tree_fingerprint(source)
    vertices = np.asarray(source.vertices, dtype=np.float64)
    faces = np.asarray(source.faces, dtype=np.int64)
    edges = np.asarray(source.triangle_mesh.topology.edges, dtype=np.int64)
    vertex_ids, edge_ids, facet_ids = entity_ids
    vertex_records = tuple(
        sorted(
            (
                int(vertex_ids[row]),
                tuple(float(value) for value in vertices[row]),
            )
            for row in range(vertices.shape[0])
        )
    )
    edge_records = tuple(
        sorted(
            (
                int(edge_ids[row]),
                tuple(sorted((int(vertex_ids[first]), int(vertex_ids[last])))),
            )
            for row, (first, last) in enumerate(edges.tolist())
        )
    )
    facet_records = tuple(
        sorted(
            (
                int(facet_ids[row]),
                _canonical_oriented_cycle(
                    (
                        int(vertex_ids[int(face[0])]),
                        int(vertex_ids[int(face[1])]),
                        int(vertex_ids[int(face[2])]),
                    )
                ),
            )
            for row, face in enumerate(faces.tolist())
        )
    )
    return {
        "kind": "reconstructed-mesh-region-persistent-topology",
        "vertices": vertex_records,
        "edges": edge_records,
        "facets": facet_records,
        "region_id": region_id,
    }


type _ReconstructedTrianglePoint = tuple[Fraction, Fraction, Fraction]
type _ReconstructedTriangleCorners = tuple[
    _ReconstructedTrianglePoint,
    _ReconstructedTrianglePoint,
    _ReconstructedTrianglePoint,
]


def _reconstructed_triangle_point(values: np.ndarray, /) -> _ReconstructedTrianglePoint:
    return (
        Fraction.from_float(float(values[0])),
        Fraction.from_float(float(values[1])),
        Fraction.from_float(float(values[2])),
    )


class _ReconstructedTrianglePatch(AbstractSurfacePatch, NonTrainableState):
    """Affine unit-square chart trimmed to one retained source triangle.

    The chart evaluates in barycentric form, so every chart corner maps to its
    authored vertex bit for bit. The edge-vector form ``p0 + u (p1 - p0)``
    rounds ``p1 - p0`` and can miss the shared vertex by one unit in the last
    place, which breaks exact corner identity between adjacent triangles.
    """

    corners: Array
    exact_corners: _ReconstructedTriangleCorners = eqx.field(static=True)

    def __init__(self, corners: np.ndarray, /) -> None:
        points = np.asarray(corners, dtype=np.float64)
        if points.shape != (3, 3) or not np.all(np.isfinite(points)):
            raise ValueError(
                "Reconstructed triangle chart corners must be finite 3-vectors."
            )
        self.exact_corners = (
            _reconstructed_triangle_point(points[0]),
            _reconstructed_triangle_point(points[1]),
            _reconstructed_triangle_point(points[2]),
        )
        self.corners = jnp.asarray(points, dtype=jnp.float64)

    @property
    def parameter_domain(self, /) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((0.0, 0.0), (1.0, 1.0))

    def evaluate(self, parameters: Array, /) -> Array:
        values = jnp.asarray(parameters, dtype=jnp.float64)
        first = values[..., 0, None]
        second = values[..., 1, None]
        return (
            (1.0 - first - second) * self.corners[0]
            + first * self.corners[1]
            + second * self.corners[2]
        )


def _reconstructed_meshing_domain(
    source: GeometrySource,
    source_id: str,
    source_revision: str,
    binding_id: str,
    entity_ids: tuple[np.ndarray, ...],
    region_id: int | None,
    edges: np.ndarray | None,
    /,
) -> MeshingDomain | None:
    if not isinstance(source, MeshRegion) or region_id is None or edges is None:
        return None
    return _triangle_complex_meshing_domain(
        np.asarray(source.vertices, dtype=np.float64),
        np.asarray(source.faces, dtype=np.int64),
        edges,
        source_id,
        source_revision,
        binding_id,
        entity_ids,
        region_id,
        ("reconstructed_vertex", "reconstructed_edge", "reconstructed_triangle"),
    )


def _triangle_complex_meshing_domain(
    vertices: np.ndarray,
    faces: np.ndarray,
    edges: np.ndarray,
    source_id: str,
    source_revision: str,
    binding_id: str,
    entity_ids: tuple[np.ndarray, ...],
    region_id: int | None,
    source_kinds: tuple[str, str, str],
    /,
) -> MeshingDomain:
    """Exact affine-chart domain of a triangle complex; open sheets own no region."""
    from .._meshing_domain import (
        MeshingDomain,
        MeshingDomainCurve,
        MeshingDomainRegion,
        MeshingSurfacePatch,
        PatchCurveUse,
    )

    edge_rows = {
        tuple(sorted((int(first), int(last)))): row
        for row, (first, last) in enumerate(edges.tolist())
    }
    curves = tuple(
        MeshingDomainCurve(int(first), int(last)) for first, last in edges.tolist()
    )
    chart_corners = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    patches = []
    for face in faces.tolist():
        uses = []
        for local in range(3):
            first_vertex = int(face[local])
            last_vertex = int(face[(local + 1) % 3])
            edge_row = edge_rows[tuple(sorted((first_vertex, last_vertex)))]
            curve_first, curve_last = (int(value) for value in edges[edge_row])
            uv_first = chart_corners[face.index(curve_first)]
            uv_last = chart_corners[face.index(curve_last)]
            pcurve = LineCurve(uv_first, uv_last - uv_first)
            sense = 1 if (first_vertex, last_vertex) == (curve_first, curve_last) else -1
            uses.append(PatchCurveUse(edge_row, pcurve, 0.0, 1.0, sense=sense))
        patches.append(
            MeshingSurfacePatch(
                _ReconstructedTrianglePatch(vertices[face]), (tuple(uses),)
            )
        )
    return MeshingDomain(
        tuple(patches),
        curves,
        vertices.shape[0],
        source_id=source_id,
        source_revision=source_revision,
        regions=()
        if region_id is None
        else (
            MeshingDomainRegion(
                "reconstructed_region",
                tuple((row, 1) for row in range(faces.shape[0])),
            ),
        ),
        source_indices=tuple(
            tuple(int(value) for value in identifiers.tolist())
            for identifiers in entity_ids
        ),
        region_source_indices=() if region_id is None else (region_id,),
        source_kinds=source_kinds,
        authority_id=binding_id,
        accuracy="exact_represented",
        corner_points=vertices,
    )


def _reconstructed_scope_banks(
    domain: MeshingDomain | None,
    source_id: str,
    source_revision: str,
    /,
) -> tuple[
    tuple[np.ndarray, ...],
    tuple[tuple[tuple[str, tuple[str, ...]], ...], ...],
    tuple[str, ...],
]:
    if domain is not None:
        return (
            domain.scope_entity_id_banks,
            domain.scope_entity_key_banks,
            domain.scope_entity_set_ids,
        )
    identifiers = tuple(np.empty((0,), dtype=np.int64) for _ in range(4))
    for values in identifiers:
        values.setflags(write=False)
    keys: tuple[tuple[tuple[str, tuple[str, ...]], ...], ...] = ((), (), (), ())
    set_ids = tuple(
        canonical_fingerprint(
            {
                "kind": "reconstructed-source-entity-set",
                "source_id": source_id,
                "source_revision": source_revision,
                "dimension": dimension,
                "entities": (),
            }
        )
        for dimension in range(4)
    )
    return identifiers, keys, set_ids


def _scope_entity_ids_by_mesh_row(
    domain: MeshingDomain | None, /
) -> tuple[np.ndarray, ...]:
    if domain is None:
        rows = tuple(np.empty((0,), dtype=np.int64) for _ in range(4))
        for values in rows:
            values.setflags(write=False)
        return rows
    row_banks: list[np.ndarray] = []
    for dimension in range(4):
        id_by_key = {
            key: int(identifier)
            for identifier, key in zip(
                domain.scope_entity_ids(dimension),
                domain.scope_entity_key_banks[dimension],
                strict=True,
            )
        }
        row_ids = np.asarray(
            [id_by_key[(key, ())] for key in domain.stratum_entity_ids(dimension)],
            dtype=np.int64,
        )
        row_ids.setflags(write=False)
        row_banks.append(row_ids)
    return tuple(row_banks)


class ReconstructedGeometrySource(GeometrySource):
    """Geometry, persistent topology identity, and provenance of a reconstruction."""

    source: GeometrySource
    report: ReconstructionReport = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)
    persistent_vertex_ids: np.ndarray
    persistent_edge_ids: np.ndarray
    persistent_facet_ids: np.ndarray
    persistent_region_id: int | None = eqx.field(static=True)
    meshing_domain: MeshingDomain | None
    scope_entity_id_banks: tuple[np.ndarray, ...]
    scope_entity_key_banks: tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]
    scope_entity_ids_by_mesh_row: tuple[np.ndarray, ...]
    scope_entity_set_ids: tuple[str, ...] = eqx.field(static=True)

    if not TYPE_CHECKING:

        def __getattribute__(self, name: str, /) -> object:
            # Equinox binds each method as a BoundMethod pytree whose construction
            # flattens the whole receiver, including its MeshingDomain. Meshing
            # routes resolve source keys once per row, so those loops became
            # quadratic. Methods bind as plain Python methods.
            return object.__getattribute__(self, name)

    @checked
    def __init__(
        self,
        source: GeometrySource,
        report: ReconstructionReport,
        *,
        persistent_vertex_ids: ArrayLike | None = None,
        persistent_edge_ids: ArrayLike | None = None,
        persistent_facet_ids: ArrayLike | None = None,
        persistent_region_id: int | None = None,
    ) -> None:
        if not isinstance(source, GeometrySource):
            raise TypeError("source must be a GeometrySource.")
        if not isinstance(report, ReconstructionReport):
            raise TypeError("report must be a ReconstructionReport.")
        feature = getattr(source, "feature_id", None)
        if not isinstance(feature, str) or not feature.strip():
            feature = f"{type(source).__module__}.{type(source).__qualname__}"
        source_id = f"reconstructed:{feature.strip()}"
        entity_ids, region_id, edges = _reconstructed_mesh_identity_records(
            source,
            source_id,
            vertex_ids=persistent_vertex_ids,
            edge_ids=persistent_edge_ids,
            facet_ids=persistent_facet_ids,
            region_id=persistent_region_id,
        )
        source_representation = (
            None if isinstance(source, (MeshRegion, PlanarMeshRegion)) else repr(source)
        )
        source_revision = canonical_fingerprint(
            {
                "kind": "reconstructed-geometry-source-revision",
                "source_id": source_id,
                "source_feature_id": feature,
                "source_type": (f"{type(source).__module__}.{type(source).__qualname__}"),
                "source_geometry": _reconstructed_source_geometry_payload(
                    source, entity_ids, region_id
                ),
                "source_representation": source_representation,
                "reconstruction_report": asdict(report),
                "persistent_entity_ids": tuple(
                    tuple(sorted(int(value) for value in values.tolist()))
                    for values in entity_ids
                ),
                "persistent_region_id": region_id,
            }
        )
        binding_id = canonical_fingerprint(
            {
                "kind": "reconstructed-geometry-binding",
                "source_id": source_id,
                "source_revision": source_revision,
            }
        )
        domain = _reconstructed_meshing_domain(
            source,
            source_id,
            source_revision,
            binding_id,
            entity_ids,
            region_id,
            edges,
        )
        id_banks, key_banks, set_ids = _reconstructed_scope_banks(
            domain, source_id, source_revision
        )
        row_scope_ids = _scope_entity_ids_by_mesh_row(domain)
        self.source = source
        self.report = report
        self.source_id = source_id
        self.source_revision = source_revision
        self.binding_id = binding_id
        self.persistent_vertex_ids = entity_ids[0]
        self.persistent_edge_ids = entity_ids[1]
        self.persistent_facet_ids = entity_ids[2]
        self.persistent_region_id = region_id
        self.meshing_domain = domain
        self.scope_entity_id_banks = id_banks
        self.scope_entity_key_banks = key_banks
        self.scope_entity_ids_by_mesh_row = row_scope_ids
        self.scope_entity_set_ids = set_ids

    def scope_entity_ids(self, dimension: int, /) -> np.ndarray:
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, Integral)
            or int(dimension) not in (0, 1, 2, 3)
        ):
            raise ValueError("Reconstructed source dimension must be in 0..3.")
        return self.scope_entity_id_banks[int(dimension)]

    def scope_entity_set_id(self, dimension: int, /) -> str:
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, Integral)
            or int(dimension) not in (0, 1, 2, 3)
        ):
            raise ValueError("Reconstructed source dimension must be in 0..3.")
        return self.scope_entity_set_ids[int(dimension)]

    def source_entity_key_for_scope(
        self, dimension: int, entity_id: int, /
    ) -> tuple[str, tuple[str, ...]]:
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, Integral)
            or int(dimension) not in (0, 1, 2, 3)
        ):
            raise ValueError("Reconstructed source dimension must be in 0..3.")
        if isinstance(entity_id, bool) or not isinstance(entity_id, Integral):
            raise TypeError("entity_id must be one stable integer scope ID.")
        dimension_ = int(dimension)
        entity_id_ = int(entity_id)
        identifiers = self.scope_entity_id_banks[dimension_]
        position = int(np.searchsorted(identifiers, entity_id_))
        if position >= identifiers.size or int(identifiers[position]) != entity_id_:
            raise ValueError(
                "entity_id is absent from this reconstructed source dimension."
            )
        return self.scope_entity_key_banks[dimension_][position]

    def scope_entity_id_for_mesh_row(self, dimension: int, row: int, /) -> int:
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, Integral)
            or int(dimension) not in (0, 1, 2, 3)
        ):
            raise ValueError("Reconstructed source dimension must be in 0..3.")
        if isinstance(row, bool) or not isinstance(row, Integral):
            raise TypeError("row must be one source mesh row index.")
        bank = self.scope_entity_ids_by_mesh_row[int(dimension)]
        row_ = int(row)
        if row_ < 0 or row_ >= bank.size:
            raise ValueError("row is outside the reconstructed source stratum.")
        return int(bank[row_])

    def source_entity_key_for_mesh_row(
        self, dimension: int, row: int, /
    ) -> tuple[str, tuple[str, ...]]:
        """Resolve a source row through its stored persistent entity mapping."""
        entity_id = self.scope_entity_id_for_mesh_row(dimension, row)
        return self.source_entity_key_for_scope(dimension, entity_id)

    def source_row_for_scope(self, dimension: int, entity_id: int, /) -> int:
        if self.meshing_domain is None:
            raise ValueError("This reconstructed source has no triangle MeshingDomain.")
        entity_key, occurrence_path = self.source_entity_key_for_scope(
            dimension, entity_id
        )
        return self.meshing_domain.resolve_entity_id(
            dimension, entity_key, occurrence_path=occurrence_path
        )

    @property
    def archive_recipe(self) -> ReconstructedGeometrySourceArchiveRecipe:
        """Return source inputs and persistent IDs, excluding derived chart banks."""
        return ReconstructedGeometrySourceArchiveRecipe(
            self.source,
            self.report,
            self.persistent_vertex_ids,
            self.persistent_edge_ids,
            self.persistent_facet_ids,
            self.persistent_region_id,
            self.source_id,
            self.source_revision,
        )

    def _compile(self, context: _ParameterCollector, /) -> GeometryKernel:
        return _ReconstructedGeometryKernel(self.source._compile(context), self.report)


class _ReconstructedGeometryKernel(GeometryKernel):
    child: GeometryKernel
    report: ReconstructionReport = eqx.field(static=True)

    def __init__(self, child: GeometryKernel, report: ReconstructionReport) -> None:
        self.child = child
        self.report = report

    @property
    def ambient_dimension(self) -> int:
        return self.child.ambient_dimension

    @property
    def intrinsic_dimension(self) -> int:
        return self.child.intrinsic_dimension

    @property
    def kind(self) -> GeometryKind:
        return self.child.kind

    @property
    def capabilities(self) -> frozenset[GeometryCapability]:
        return self.child.capabilities

    @property
    def field_certificate(self) -> FieldCertificate:
        return self.child.field_certificate

    def geometry_validity(self, state: DesignState, /) -> GeometryValidityEvidence:
        return representation_validity(self.child, state)

    def boundary_field(self, state: DesignState, points: Array, /) -> Array:
        return self.child.boundary_field(state, points)

    def contains(self, state: DesignState, points: Array, /) -> Array:
        return self.child.contains(state, points)

    def boundary_normal(self, state: DesignState, points: Array, /) -> Array:
        return self.child.boundary_normal(state, points)

    def closest_point(self, state: DesignState, points: Array, /) -> ClosestPointResult:
        if not isinstance(self.child, ClosestPointProvider):
            raise TypeError("Reconstructed child lacks a closest-point provider.")
        result = self.child.closest_point(state, points)
        if not isinstance(result, ClosestPointResult):
            raise TypeError("Child closest-point query returned an invalid result.")
        return result

    def contact_curvature(
        self, state: DesignState, points: Array, /
    ) -> ContactCurvatureResult:
        if not isinstance(self.child, ContactCurvatureProvider):
            raise TypeError("Reconstructed child lacks a contact-curvature provider.")
        result = self.child.contact_curvature(state, points)
        if not isinstance(result, ContactCurvatureResult):
            raise TypeError("Child contact-curvature query returned an invalid result.")
        return result

    def support_map(self, state: DesignState, directions: Array, /) -> Array:
        if not isinstance(self.child, SupportMapProvider):
            raise TypeError("Reconstructed child lacks a support-map provider.")
        return self.child.support_map(state, directions)

    def bounds(self, state: DesignState, /) -> Array:
        return self.child.bounds(state)

    def measure(self, state: DesignState, /) -> Array:
        return self.child.measure(state)

    def boundary_measure(self, state: DesignState, /) -> Array:
        return self.child.boundary_measure(state)

    def interior_mass(self, state: DesignState, /) -> Mass:
        return self.child.interior_mass(state)

    def boundary_mass(self, state: DesignState, /) -> Mass:
        return self.child.boundary_mass(state)

    def sample_interior(
        self,
        state: DesignState,
        num_points: int,
        /,
        *,
        key: PRNGKey,
        plan: RejectionSamplingPlan | None = None,
    ) -> SamplingResult:
        return self.child.sample_interior(
            state,
            num_points,
            key=key,
            plan=plan,
        )

    def sample_boundary(
        self, state: DesignState, num_points: int, /, *, key: PRNGKey
    ) -> SamplingResult:
        return self.child.sample_boundary(state, num_points, key=key)

    def boundary_atlas(self, state: DesignState, /) -> BoundaryAtlas:
        return self.child.boundary_atlas(state)

    def cubature_atlas(
        self, state: DesignState, component: CubatureComponent, /
    ) -> CubatureAtlas:
        return self.child.cubature_atlas(state, component)

    def seam_residual(self, state: DesignState, /) -> Array:
        if not isinstance(self.child, SeamDiagnosticsProvider):
            raise TypeError("Reconstructed child lacks a seam-diagnostics provider.")
        return self.child.seam_residual(state)


def _point_digest(points: np.ndarray) -> str:
    canonical = np.ascontiguousarray(points, dtype=np.float64)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _validated_points(points: ArrayLike, dimension: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < dimension:
        raise ValueError(f"points must have shape (num_points, >= {dimension}).")
    values = values[:, :dimension]
    if values.shape[0] < dimension + 1:
        raise ValueError("The point cloud has too few points for reconstruction.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Reconstruction points must all be finite.")
    return values


def _recenter(points: np.ndarray, enabled: bool) -> tuple[np.ndarray, np.ndarray]:
    offset = (
        0.5 * (np.min(points, axis=0) + np.max(points, axis=0))
        if enabled
        else np.zeros((points.shape[1],), dtype=np.float64)
    )
    return points - offset, offset


def _planar_triangulation(
    points: np.ndarray,
    /,
    *,
    tolerance: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = max(float(np.max(np.ptp(points, axis=0))), 1.0)
    if tolerance > 0.0:
        quantized = np.rint(points / (tolerance * scale)).astype(np.int64)
        _, retained = np.unique(quantized, axis=0, return_index=True)
    else:
        _, retained = np.unique(points, axis=0, return_index=True)
    retained = np.sort(retained)
    vertices = points[retained]
    if vertices.shape[0] < 3:
        raise ValueError("Planar reconstruction retained fewer than three points.")
    # Direct native entry avoids importing the optional Qhull provider.
    faces, _ = delaunay_2d(vertices)
    triangles = vertices[faces]
    doubled_area = np.abs(
        (triangles[:, 1, 0] - triangles[:, 0, 0])
        * (triangles[:, 2, 1] - triangles[:, 0, 1])
        - (triangles[:, 1, 1] - triangles[:, 0, 1])
        * (triangles[:, 2, 0] - triangles[:, 0, 0])
    )
    if alpha > 0.0:
        first = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
        second = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
        third = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)
        radius = first * second * third / np.maximum(2.0 * doubled_area, 1.0e-300)
        faces = faces[radius <= alpha]
    if faces.shape[0] == 0:
        raise ValueError("Planar reconstruction produced no retained triangles.")
    return vertices, faces, retained


def _clean_surface_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, TriangleTopology]:
    vertices_, faces_ = _canonical_triangle_arrays(vertices, faces)
    # ty: ignore[invalid-argument-type]
    topology = TriangleTopology(faces_, num_vertices=vertices_.shape[0])
    return vertices_, faces_, topology


def _parameter_records(**parameters: object) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((name, repr(value)) for name, value in parameters.items()))


def reconstruct_planar_region(
    points: ArrayLike,
    *,
    recenter: bool = True,
    alpha: float = 0.0,
    tolerance: float = 1e-5,
    offset: float = 1.0,
    bound: bool = False,
    progress_bar: bool = False,
    feature_id: str | None = None,
) -> ReconstructedGeometrySource:
    """Reconstruct one planar region and report every host-side approximation."""

    points_ = _validated_points(points, 2)
    if alpha < 0.0 or tolerance < 0.0 or offset <= 0.0:
        raise ValueError("alpha/tolerance must be non-negative and offset positive.")
    vertices_2d, faces, _ = _planar_triangulation(
        points_,
        tolerance=float(tolerance),
        alpha=float(alpha),
    )
    planar = planar_region_from_triangles(
        vertices_2d,
        faces,
        recenter=False,
        feature_id=feature_id,
    )
    planar_vertices = np.asarray(planar.vertices, dtype=np.float64)
    vertices, center = _recenter(planar_vertices, recenter)
    offsets = np.asarray(planar.loop_offsets, dtype=np.int32)
    loops = tuple(
        np.arange(offsets[index], offsets[index + 1], dtype=np.int32)
        for index in range(offsets.shape[0] - 1)
    )
    source = PlanarMeshRegion(vertices, loops, feature_id=feature_id)
    algorithm = "meshcore_delaunay_2d_native_boundary"
    parameters = _parameter_records(
        alpha=float(alpha),
        tolerance=float(tolerance),
        offset=float(offset),
        bound=bool(bound),
    )
    report = ReconstructionReport(
        source_kind="planar_point_cloud",
        algorithm=algorithm,
        input_digest=_point_digest(points_),
        input_points=points_.shape[0],
        retained_points=points_.shape[0],
        output_vertices=vertices.shape[0],
        output_cells=faces.shape[0],
        connected_components=1,
        watertight=True,
        winding_consistent=True,
        recenter_offset=tuple(float(value) for value in center),
        parameters=parameters,
    )
    return ReconstructedGeometrySource(source, report)


def _surface_source(
    points: np.ndarray,
    surface: tuple[np.ndarray, np.ndarray],
    *,
    source_kind: str,
    algorithm: str,
    recenter: bool,
    parameters: tuple[tuple[str, str], ...],
    input_points: int,
    warnings: Sequence[str] = (),
    feature_id: str | None = None,
    source_product_id: str | None = None,
    input_digest: str | None = None,
    normal_evidence: NormalEstimationEvidence | None = None,
    poisson_evidence: PoissonSolveEvidence | None = None,
    outlier_evidence: OutlierRemovalEvidence | None = None,
    thin_feature_evidence: ThinFeatureEvidence | None = None,
    component_evidence: ComponentFitEvidence | None = None,
    coverage_evidence: SamplingCoverageEvidence | None = None,
    repair_interpretation: RepairInterpretation | None = None,
    measure_deviation: bool = False,
) -> ReconstructedGeometrySource:
    vertices, faces = surface
    digest = _point_digest(points) if input_digest is None else input_digest
    try:
        vertices_clean, faces_clean, topology = _clean_surface_mesh(vertices, faces)
        winding_consistent = True
    except ValueError as error:
        report = ReconstructionReport(
            source_kind=source_kind,
            algorithm=algorithm,
            input_digest=digest,
            input_points=input_points,
            retained_points=points.shape[0],
            output_vertices=vertices.shape[0],
            output_cells=faces.shape[0],
            connected_components=0,
            watertight=False,
            winding_consistent=False,
            recenter_offset=(0.0, 0.0, 0.0),
            parameters=parameters,
            warnings=(*warnings, str(error)),
            source_product_id=source_product_id,
            normal_evidence=normal_evidence,
            poisson_evidence=poisson_evidence,
            outlier_evidence=outlier_evidence,
            thin_feature_evidence=thin_feature_evidence,
            component_evidence=component_evidence,
            coverage_evidence=coverage_evidence,
            repair_interpretation=repair_interpretation,
        )
        raise ReconstructionFailure(
            "Surface reconstruction produced invalid triangle topology.", report
        ) from error
    deviation = (
        sampled_surface_deviation(points, vertices_clean, faces_clean)
        if measure_deviation and topology.watertight
        else None
    )
    vertices_clean, center = _recenter(vertices_clean, recenter)
    report = ReconstructionReport(
        source_kind=source_kind,
        algorithm=algorithm,
        input_digest=digest,
        input_points=input_points,
        retained_points=points.shape[0],
        output_vertices=vertices_clean.shape[0],
        output_cells=faces_clean.shape[0],
        connected_components=topology.num_face_components,
        watertight=topology.watertight,
        winding_consistent=winding_consistent,
        recenter_offset=tuple(float(value) for value in center),
        parameters=parameters,
        warnings=tuple(warnings),
        source_product_id=source_product_id,
        euler_characteristic=topology.euler_characteristic,
        normal_evidence=normal_evidence,
        poisson_evidence=poisson_evidence,
        deviation_evidence=deviation,
        outlier_evidence=outlier_evidence,
        thin_feature_evidence=thin_feature_evidence,
        component_evidence=component_evidence,
        coverage_evidence=coverage_evidence,
        repair_interpretation=repair_interpretation,
    )
    if not report.watertight or not report.winding_consistent:
        raise ReconstructionFailure(
            "Surface reconstruction did not produce a watertight consistently wound solid.",
            report,
        )
    # ty: ignore[invalid-argument-type]
    source = MeshRegion(vertices_clean, faces_clean, feature_id=feature_id)
    return ReconstructedGeometrySource(source, report)


def _normal_warnings(evidence: NormalEstimationEvidence, /) -> tuple[str, ...]:
    warnings: list[str] = []
    if evidence.ambiguous_direction_count:
        warnings.append(
            f"{evidence.ambiguous_direction_count} neighborhoods have ambiguous PCA normals."
        )
    if evidence.ambiguous_orientation_edges:
        warnings.append(
            f"{evidence.ambiguous_orientation_edges} orientation edges join nearly perpendicular normals."
        )
    if evidence.conflicting_neighbor_edges:
        warnings.append(
            f"{evidence.conflicting_neighbor_edges} neighbor pairs remain oppositely oriented."
        )
    return tuple(warnings)


def _poisson_surface_source(
    points: np.ndarray,
    *,
    source_kind: str,
    input_points: int,
    input_digest: str,
    recenter: bool,
    normals: ArrayLike | None,
    normal_orientation: NormalOrientation,
    neighborhood_size: int,
    sample_spacing: float | None,
    screening: float,
    discretization: PoissonDiscretization,
    robustness: ReconstructionRobustness,
    initial_warnings: Sequence[str] = (),
    base_parameters: dict[str, object] | None = None,
    feature_id: str | None = None,
    source_product_id: str | None = None,
) -> ReconstructedGeometrySource:
    if not isinstance(robustness, ReconstructionRobustness):
        raise TypeError("robustness must be a ReconstructionRobustness.")
    size = positive_integer(neighborhood_size, "neighborhood_size")
    spacing = (
        None
        if sample_spacing is None
        else positive_finite_float(sample_spacing, "sample_spacing")
    )
    screening_ = positive_finite_float(screening, "screening")
    orientation = parse(normal_orientation, NormalOrientation, "normal_orientation")
    discretization_ = parse(discretization, PoissonDiscretization, "discretization")
    parameter_values = dict(base_parameters or {})
    parameter_values.update(
        normals="supplied" if normals is not None else "estimated",
        normal_orientation=orientation,
        neighborhood_size=size,
        sample_spacing=spacing,
        screening=screening_,
        discretization=discretization_,
        outlier_std_ratio=robustness.outlier_std_ratio,
        coverage_distance=robustness.coverage_distance,
        incomplete_sampling=robustness.incomplete_sampling,
        thin_feature_distance=robustness.thin_feature_distance,
        component_fit_tolerance=robustness.component_fit_tolerance,
    )
    parameters = _parameter_records(**parameter_values)
    warnings = tuple(initial_warnings)
    retained_points = points
    retained_normals = None if normals is None else np.asarray(normals, dtype=np.float64)
    outlier_evidence: OutlierRemovalEvidence | None = None
    normal_evidence: NormalEstimationEvidence | None = None
    poisson_evidence: PoissonSolveEvidence | None = None
    thin_evidence: ThinFeatureEvidence | None = None
    fit_evidence: ComponentFitEvidence | None = None
    coverage_evidence: SamplingCoverageEvidence | None = None
    repair_interpretation: RepairInterpretation = "sample_supported_poisson_fit"

    def fail(reason: str, *, vertices: int = 0, cells: int = 0) -> NoReturn:
        report = ReconstructionReport(
            source_kind=source_kind,
            algorithm="native_screened_poisson",
            input_digest=input_digest,
            input_points=input_points,
            retained_points=retained_points.shape[0],
            output_vertices=vertices,
            output_cells=cells,
            connected_components=0,
            watertight=False,
            winding_consistent=False,
            recenter_offset=(0.0, 0.0, 0.0),
            parameters=parameters,
            warnings=(*warnings, reason),
            source_product_id=source_product_id,
            normal_evidence=normal_evidence,
            poisson_evidence=poisson_evidence,
            outlier_evidence=outlier_evidence,
            thin_feature_evidence=thin_evidence,
            component_evidence=fit_evidence,
            coverage_evidence=coverage_evidence,
            repair_interpretation=repair_interpretation,
        )
        raise ReconstructionFailure(reason, report)

    def fail_allocation(stage: str, error: MemoryError, /) -> NoReturn:
        detail = str(error) or "allocator supplied no detail"
        fail(
            f"Native reconstruction allocation failed during {stage} "
            f"({type(error).__name__}): {detail}"
        )

    if robustness.outlier_std_ratio is not None:
        try:
            retained_mask, outliers = remove_statistical_outliers(
                retained_points, size, robustness.outlier_std_ratio
            )
        except MemoryError as error:
            fail_allocation("outlier removal", error)
        outlier_evidence = outliers
        retained_points = retained_points[retained_mask]
        if retained_normals is not None:
            if retained_normals.shape != points.shape:
                raise ValueError(
                    f"normals must have shape {points.shape}; got {retained_normals.shape}."
                )
            retained_normals = retained_normals[retained_mask]
        if outliers.removed_indices:
            warnings = (
                *warnings,
                f"Statistical outlier removal dropped {len(outliers.removed_indices)} samples.",
            )
        if retained_points.shape[0] <= size:
            fail(
                "Outlier removal retained too few points for the requested neighborhood."
            )

    try:
        oriented = estimate_point_normals(
            retained_points,
            normals=retained_normals,
            orientation=orientation,
            neighborhood_size=size,
        )
    except MemoryError as error:
        fail_allocation("normal estimation", error)
    normal_evidence = oriented.evidence
    try:
        areas = sample_areas(oriented.neighbor_distances)
    except MemoryError as error:
        fail_allocation("sample-area estimation", error)
    spacing_ = (
        _SPACING_PER_SAMPLE * float(np.sqrt(np.median(areas)))
        if spacing is None
        else spacing
    )
    try:
        thin = thin_features(
            retained_points,
            oriented.normals,
            size,
            _RESOLUTION_CELLS * spacing_
            if robustness.thin_feature_distance is None
            else robustness.thin_feature_distance,
        )
    except MemoryError as error:
        fail_allocation("thin-feature analysis", error)
    thin_evidence = thin
    warnings = (*warnings, *_normal_warnings(normal_evidence))
    if thin.thin_samples:
        warnings = (
            *warnings,
            f"{thin.thin_samples} samples have a stacked sheet closer than "
            f"{thin.thin_feature_distance:.6g}; Poisson smoothing cannot separate it.",
        )
    match discretization_:
        case "octree":
            solver = solve_octree_poisson
        case "regular":
            solver = solve_regular_poisson
        case _:
            assert_never(discretization_)
    try:
        indicator = solver(
            retained_points,
            oriented.normals,
            areas,
            spacing=spacing_,
            screening=screening_,
            padding_cells=_PADDING_CELLS,
        )
    except MemoryError as error:
        fail_allocation("Poisson grid and solver", error)
    except (ValueError, OverflowError) as error:
        fail(str(error))
    poisson = indicator.evidence
    poisson_evidence = poisson
    try:
        fit = component_fit(
            indicator.sample_values,
            areas,
            oriented.components,
            robustness.component_fit_tolerance,
        )
    except MemoryError as error:
        fail_allocation("component fitting", error)
    fit_evidence = fit
    if not poisson.solver_converged:
        fail(f"Screened Poisson solve did not converge: {poisson.solver_message}")
    if fit.unresolved_components:
        fail(
            f"Screened Poisson level set does not fit sample components {fit.unresolved_components!r}."
        )
    try:
        vertices, faces = extract_indicator_surface(indicator)
    except MemoryError as error:
        fail_allocation("level-set extraction", error)
    try:
        _, coverage = sampling_coverage(
            retained_points,
            vertices,
            faces,
            _RESOLUTION_CELLS * spacing_
            if robustness.coverage_distance is None
            else robustness.coverage_distance,
        )
    except MemoryError as error:
        fail_allocation("surface coverage analysis", error)
    coverage_evidence = coverage
    if coverage.unsupported_triangles:
        repair_interpretation = "poisson_inferred_closure"
        warnings = (
            *warnings,
            f"{coverage.unsupported_patches} output patches covering "
            f"{coverage.unsupported_area:.6g} area exceed sample coverage; "
            "these are Poisson-inferred closures, not observed surface.",
        )
    match robustness.incomplete_sampling:
        case "close":
            pass
        case "refuse":
            if coverage.unsupported_triangles:
                repair_interpretation = "incomplete_sampling_refused"
                fail(
                    f"Coverage policy refused {coverage.unsupported_triangles} unsupported triangles."
                )
        case _:
            assert_never(robustness.incomplete_sampling)
    try:
        return _surface_source(
            retained_points,
            (vertices, faces),
            source_kind=source_kind,
            algorithm="native_screened_poisson",
            recenter=recenter,
            parameters=parameters,
            input_points=input_points,
            warnings=warnings,
            feature_id=feature_id,
            source_product_id=source_product_id,
            input_digest=input_digest,
            normal_evidence=normal_evidence,
            poisson_evidence=poisson_evidence,
            outlier_evidence=outlier_evidence,
            thin_feature_evidence=thin_evidence,
            component_evidence=fit_evidence,
            coverage_evidence=coverage_evidence,
            repair_interpretation=repair_interpretation,
            measure_deviation=True,
        )
    except MemoryError as error:
        fail_allocation("source geometry construction", error)


def reconstruct_surface_region(
    points: ArrayLike,
    *,
    normals: ArrayLike | None = None,
    normal_orientation: NormalOrientation = "propagate",
    recenter: bool = True,
    neighborhood_size: int | None = None,
    sample_spacing: float | None = None,
    screening: float = 4.0,
    discretization: PoissonDiscretization = "octree",
    robustness: ReconstructionRobustness = _DEFAULT_ROBUSTNESS,
    feature_id: str | None = None,
) -> ReconstructedGeometrySource:
    """Reconstruct a closed surface by native screened Poisson.

    Missing normals are estimated from bounded k-neighborhood PCA and oriented
    over a minimum-spanning forest. The adaptive octree is the default solve.
    Features below the finest grid width are smoothed; unsupported closed
    patches are reported as Poisson-inferred completion, not measured surface.
    """

    points_ = _validated_points(points, 3)
    neighborhood = (
        min(16, points_.shape[0] - 1) if neighborhood_size is None else neighborhood_size
    )
    return _poisson_surface_source(
        points_,
        source_kind="surface_point_cloud",
        input_points=points_.shape[0],
        input_digest=_point_digest(points_),
        recenter=recenter,
        normals=normals,
        normal_orientation=normal_orientation,
        neighborhood_size=neighborhood,
        sample_spacing=sample_spacing,
        screening=screening,
        discretization=discretization,
        robustness=robustness,
        feature_id=feature_id,
    )


def _terrain_points(
    points_or_grid: ArrayLike,
    *,
    x: ArrayLike | None,
    y: ArrayLike | None,
) -> np.ndarray:
    values = np.asarray(points_or_grid, dtype=np.float64)
    if values.ndim == 2 and values.shape[1] != 3:
        rows, columns = values.shape
        x_values = (
            np.arange(columns, dtype=np.float64)
            if x is None
            else np.asarray(x, dtype=np.float64)
        )
        y_values = (
            np.arange(rows, dtype=np.float64)
            if y is None
            else np.asarray(y, dtype=np.float64)
        )
        if x_values.shape != (columns,) or y_values.shape != (rows,):
            raise ValueError("x and y coordinate vectors must match the height grid.")
        x_grid, y_grid = np.meshgrid(x_values, y_values)
        points = np.column_stack((x_grid.ravel(), y_grid.ravel(), values.ravel()))
    else:
        if values.ndim != 2:
            raise ValueError("Terrain points must have shape (num_points, 3).")
        # Terrain triangulates in the plane: one triangle needs three samples.
        points = values
        if points.shape[0] < 3:
            raise ValueError("A terrain needs at least three samples to triangulate.")
    if not np.all(np.isfinite(points)):
        raise ValueError("Terrain samples must all be finite.")
    return points


def reconstruct_dem_region(
    points_or_grid: ArrayLike,
    *,
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
    recenter: bool = True,
    alpha: float = 0.0,
    tolerance: float = 1e-5,
    bound: bool = False,
    extrude_depth: float = 1.0,
    progress_bar: bool = False,
    feature_id: str | None = None,
) -> ReconstructedGeometrySource:
    """Triangulate a terrain and cap a downward extrusion as a reported solid."""

    points = _terrain_points(points_or_grid, x=x, y=y)
    if alpha < 0.0 or tolerance < 0.0 or extrude_depth <= 0.0:
        raise ValueError(
            "alpha/tolerance must be non-negative and extrude_depth positive."
        )
    planar_vertices, top_faces, retained_indices = _planar_triangulation(
        points[:, :2],
        tolerance=float(tolerance),
        alpha=float(alpha),
    )
    top_vertices = points[retained_indices].copy()
    top_vertices[:, :2] = planar_vertices
    # ty: ignore[invalid-argument-type]
    topology = TriangleTopology(top_faces, num_vertices=top_vertices.shape[0])
    boundary = np.asarray(topology.boundary_halfedges, dtype=np.int32)
    origins = np.asarray(topology.halfedge_origin, dtype=np.int32)[boundary]
    destinations = np.asarray(topology.halfedge_destination, dtype=np.int32)[boundary]
    count = top_vertices.shape[0]
    bottom_vertices = top_vertices.copy()
    bottom_vertices[:, 2] -= float(extrude_depth)
    side_first = np.stack((origins + count, destinations + count, destinations), axis=1)
    side_second = np.stack((origins + count, destinations, origins), axis=1)
    solid_vertices = np.concatenate((top_vertices, bottom_vertices), axis=0)
    solid_faces = np.concatenate(
        (
            top_faces,
            top_faces[:, [0, 2, 1]] + count,
            side_first,
            side_second,
        ),
        axis=0,
    )
    return _surface_source(
        points,
        (solid_vertices, solid_faces),
        source_kind="digital_elevation_model",
        algorithm="native_delaunay_2d_capped_extrusion",
        recenter=recenter,
        parameters=_parameter_records(
            alpha=float(alpha),
            tolerance=float(tolerance),
            bound=bool(bound),
            extrude_depth=float(extrude_depth),
        ),
        input_points=points.shape[0],
        feature_id=feature_id,
    )


def reconstruct_point_region(
    points: ArrayLike,
    *,
    normals: ArrayLike | None = None,
    normal_orientation: NormalOrientation = "propagate",
    recenter: bool = True,
    roi: tuple[float, float, float, float, float, float] | None = None,
    voxel_size: float | None = None,
    neighborhood_size: int | None = None,
    sample_spacing: float | None = None,
    screening: float = 4.0,
    discretization: PoissonDiscretization = "octree",
    robustness: ReconstructionRobustness = _DEFAULT_ROBUSTNESS,
    feature_id: str | None = None,
    source_product_id: str | None = None,
) -> ReconstructedGeometrySource:
    """Filter Cartesian samples and fit a native screened-Poisson solid.

    Unsupported portions of the closed level set are model-implied completion,
    not observed surface. Their coverage and repair interpretation are reported.
    """

    original = _validated_points(points, 3)
    retained = original
    retained_normals = None if normals is None else np.asarray(normals, dtype=np.float64)
    if retained_normals is not None and retained_normals.shape != original.shape:
        raise ValueError(
            f"normals must have shape {original.shape}; got {retained_normals.shape}."
        )
    if retained_normals is not None and not np.all(np.isfinite(retained_normals)):
        raise ValueError("normals must all be finite.")
    warnings: list[str] = []
    if roi is not None:
        bounds = np.asarray(roi, dtype=np.float64)
        if bounds.shape != (6,) or not np.all(np.isfinite(bounds)):
            raise ValueError("roi must contain six finite bounds.")
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        if not (x_min < x_max and y_min < y_max and z_min < z_max):
            raise ValueError("roi minima must be strictly below maxima.")
        mask = (
            (retained[:, 0] >= x_min)
            & (retained[:, 0] <= x_max)
            & (retained[:, 1] >= y_min)
            & (retained[:, 1] <= y_max)
            & (retained[:, 2] >= z_min)
            & (retained[:, 2] <= z_max)
        )
        retained = retained[mask]
        if retained_normals is not None:
            retained_normals = retained_normals[mask]
    if voxel_size is not None:
        voxel_width = positive_finite_float(voxel_size, "voxel_size")
        voxel = np.floor(retained / voxel_width).astype(np.int64)
        _, indices = np.unique(voxel, axis=0, return_index=True)
        indices = np.sort(indices)
        retained = retained[indices]
        if retained_normals is not None:
            retained_normals = retained_normals[indices]
    if retained.shape[0] < 4:
        raise ValueError("Filtering retained too few points for reconstruction.")
    if retained.shape[0] < original.shape[0] / 10:
        warnings.append("Filtering retained fewer than ten percent of input points.")
    neighborhood = (
        min(16, retained.shape[0] - 1) if neighborhood_size is None else neighborhood_size
    )
    return _poisson_surface_source(
        retained,
        source_kind="point_cloud",
        input_points=original.shape[0],
        input_digest=_point_digest(original),
        recenter=recenter,
        normals=retained_normals,
        normal_orientation=normal_orientation,
        neighborhood_size=neighborhood,
        sample_spacing=sample_spacing,
        screening=screening,
        discretization=discretization,
        robustness=robustness,
        initial_warnings=warnings,
        base_parameters={"roi": roi, "voxel_size": voxel_size},
        feature_id=feature_id,
        source_product_id=source_product_id,
    )


def reconstruct_lidar_region(
    product: LidarPointProduct,
    *,
    recenter: bool = True,
    roi: tuple[float, float, float, float, float, float] | None = None,
    voxel_size: float | None = None,
    neighborhood_size: int | None = None,
    sample_spacing: float | None = None,
    screening: float = 4.0,
    discretization: PoissonDiscretization = "octree",
    robustness: ReconstructionRobustness = _DEFAULT_ROBUSTNESS,
    feature_id: str | None = None,
) -> ReconstructedGeometrySource:
    """Reconstruct active LiDAR points while retaining acquisition lineage."""

    if not isinstance(product, LidarPointProduct):
        raise TypeError("product must be LidarPointProduct.")
    active = np.asarray(product.support.active_mask, dtype=np.bool_)
    return reconstruct_point_region(
        np.asarray(product.support.points)[active],
        recenter=recenter,
        roi=roi,
        voxel_size=voxel_size,
        neighborhood_size=neighborhood_size,
        sample_spacing=sample_spacing,
        screening=screening,
        discretization=discretization,
        robustness=robustness,
        feature_id=feature_id,
        source_product_id=product.point_product_id,
    )


__all__ = [
    "ReconstructedGeometrySource",
    "ReconstructedGeometrySourceArchiveRecipe",
    "ReconstructionReportProvider",
    "ReconstructionFailure",
    "ReconstructionReport",
    "reconstruct_dem_region",
    "reconstruct_lidar_region",
    "reconstruct_point_region",
    "reconstruct_planar_region",
    "reconstruct_surface_region",
]
