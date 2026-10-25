#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Revision-bound source bindings admitted by the native meshing provider.

Each binding attaches one immutable source revision to a geometry owner.
Represented geometry, entity identity, and queries remain with
``phydrax.geometry``. Source entity identifiers used in published geometry
associations are ``<revision>:<kind>:<index>`` strings of the kinds named on
each binding.
"""

from __future__ import annotations

from typing import final

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

import phydrax.discretization._cell_geometry as cell_geometry_owner
import phydrax.discretization._cell_mesh as cell_mesh_owner
import phydrax.discretization._tensor_support as tensor_support_owner
import phydrax.geometry._atlas as atlas_owner
import phydrax.geometry._contracts as contracts_owner
import phydrax.geometry.implicit._adaptive_discovery as adaptive_discovery_owner
import phydrax.geometry.simplicial._regions as regions_owner
from phydrax.geometry._source_boundary_records import SourceBoundaryQuery
from phydrax.meshing._host_banks import family_host_array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._meshcore import NativeExecutionScope
from ..._strict import StrictModule
from ..._trainable import fixed_field, NonTrainableState
from ..._validation import normalized_identifier, positive_finite_float
from ...discretization._cell_geometry import CellGeometryRestrictionSource
from ...discretization._cell_geometry_validity import cell_geometry_id
from ...geometry._domain_coverage import PiecewiseLinearDomain
from ...geometry._mapped_reference_domain import MappedReferenceDomain
from ...geometry._meshing_domain import MeshingDomain
from ...geometry.brep._model import BRepGeometry, BRepModel
from ...typing import checked
from .. import _surface_envelope as envelope_owner
from .._boundary_layer import BoundaryLayerMesh
from .._controls import BlockInterfaceControl
from .._layer_core_regions import prepare_layer_region_identity
from .._plc_source import PiecewiseLinearComplex
from .._scope import MeshingEntityKind, MeshingScope
from .._structured import TransfiniteBlock
from .._sweep import SweepControl
from .._volume_generation import declared_plc_domain


def source_entity_id(revision: str, kind: str, index: int, /) -> str:
    """Return the revision-bound identifier of one native source entity."""

    return f"{revision}:{kind}:{index}"


@final
class NativePlanarSource(StrictModule, NonTrainableState):
    """Planar piecewise-linear domain: region loops plus embedded segments.

    Entity identities: the region is ``region:0``; loop edges are ``edge:i``
    for the rows of ``region.edges``, followed by embedded segment edges
    ``edge:E + j``; region vertices are ``vertex:i`` followed by embedded
    vertices ``vertex:V + j``. Embedded segments are protected constraints of
    the generated mesh and must lie inside the region.
    """

    region: regions_owner.PlanarMeshRegion
    embedded: regions_owner.SegmentMesh | None
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        region: regions_owner.PlanarMeshRegion,
        source_revision: str,
        /,
        *,
        embedded: regions_owner.SegmentMesh | None = None,
    ) -> None:

        revision = normalized_identifier(source_revision, "source_revision")
        self.region = region
        self.embedded = embedded
        self.source_id = region.feature_id
        self.source_revision = revision
        self.binding_id = canonical_fingerprint(
            {
                "kind": "native-planar-source",
                "source_id": region.feature_id,
                "source_revision": revision,
                "vertices": array_tree_fingerprint(np.asarray(region.vertices)),
                "edges": array_tree_fingerprint(np.asarray(region.edges)),
                "loops": array_tree_fingerprint(np.asarray(region.loop_offsets)),
                "embedded": (
                    None
                    if embedded is None
                    else [
                        embedded.source_id,
                        array_tree_fingerprint(np.asarray(embedded.vertices)),
                        array_tree_fingerprint(np.asarray(embedded.edges)),
                    ]
                ),
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[NativePlanarSource], value: NativePlanarSource, /
    ) -> NativePlanarSource:
        """Revalidate a stored planar source through its constructor."""
        if type(value) is not cls:
            raise TypeError(
                f"{cls.__name__}.stored_fields must have the registered type; got {type(value).__name__}"
            )
        fresh = cls(value.region, value.source_revision, embedded=value.embedded)
        if (fresh.source_id, fresh.binding_id) != (value.source_id, value.binding_id):
            raise ValueError(
                f"{cls.__name__}.binding_id must match its stored source fields; got {value.binding_id}"
            )
        return fresh


@final
class NativeImplicitSource(StrictModule, NonTrainableState):
    """Compiled implicit geometry with its discovery lattice and identity.

    The discovered zero set is the source entity ``implicit-zero-set``.
    """

    geometry: contracts_owner.CompiledGeometry
    grid: tensor_support_owner.PreparedTensorGrid
    discovery: adaptive_discovery_owner.ImplicitVolumeQuery | None
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        geometry: contracts_owner.CompiledGeometry,
        grid: tensor_support_owner.PreparedTensorGrid,
        source_id: str,
        source_revision: str,
        /,
        *,
        discovery: adaptive_discovery_owner.ImplicitVolumeQuery | None = None,
    ) -> None:
        source = normalized_identifier(source_id, "source_id")
        self.geometry = geometry
        self.grid = grid
        self.source_id = source
        self.source_revision = normalized_identifier(source_revision, "source_revision")
        if discovery is not None:
            discovery.require_bound(geometry)
        self.discovery = discovery


@final
class NativeCurveSource(StrictModule, NonTrainableState):
    """Parametric curve charts or a straight segment network.

    A ``BoundaryAtlas`` with one-dimensional charts supplies curves on the unit
    reference interval, identified by chart index (several charts may share
    one atlas source entity, e.g. the arcs of a circle); a ``SegmentMesh``
    supplies straight curves identified by edge row. Curve entities are
    ``curve:<id>``.
    """

    curves: atlas_owner.BoundaryAtlas | regions_owner.SegmentMesh
    ambient_dimension: int = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    gap_domain: MeshingDomain | None

    @checked
    def __init__(
        self,
        curves: atlas_owner.BoundaryAtlas | regions_owner.SegmentMesh,
        source_revision: str,
        /,
        *,
        gap_domain: MeshingDomain | None = None,
    ) -> None:
        match curves:
            case atlas_owner.BoundaryAtlas():
                if curves.reference_dimension != 1:
                    raise ValueError(
                        "NativeCurveSource.curves.reference_dimension must equal 1; got a different chart dimension."
                    )
                if curves.num_charts == 0:
                    raise ValueError(
                        "NativeCurveSource.curves.num_charts must be positive; got zero charts."
                    )
                ambient = curves.ambient_dimension
            case regions_owner.SegmentMesh():
                ambient = curves.vertices.shape[1]
            case _:
                raise TypeError(
                    "NativeCurveSource.curves must be BoundaryAtlas or SegmentMesh; got an unsupported source kind."
                )
        if gap_domain is not None:
            match gap_domain.brep_authority:
                case BRepGeometry() as geometry:
                    pass
                case BRepModel() as model:
                    geometry = model.require_geometry()
                case None:
                    geometry = None
            if geometry is None or (
                geometry.edge_vertex_gap_bounds is None
                and geometry.edge_face_gap_bounds is None
            ):
                gap_domain = None
            elif (
                gap_domain.domain_id != gap_domain.content_domain_id()
                or array_tree_fingerprint(curves)
                != array_tree_fingerprint(gap_domain.curve_atlas)
                or source_revision != gap_domain.source_revision
            ):
                raise ValueError(
                    "NativeCurveSource.gap_domain must bind the current curve atlas and revision; got incompatible source"
                )
        self.curves = curves
        self.ambient_dimension = ambient
        self.source_id = curves.source_id
        self.source_revision = normalized_identifier(source_revision, "source_revision")
        self.gap_domain = gap_domain


@final
class NativeSurfaceSource(StrictModule, NonTrainableState):
    """A prepared parametric meshing domain of patches, curves and corners.

    Entity identities are the domain strata ``corner:i``, ``curve:i`` and
    ``surface:i``; identity and revision are the domain's own.
    """

    domain: MeshingDomain
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    @checked
    def __init__(self, domain: MeshingDomain, /) -> None:

        self.domain = domain
        self.source_id = domain.source_id
        self.source_revision = domain.source_revision
        self.binding_id = canonical_fingerprint(
            {"kind": "native-surface-source", "domain": domain.domain_id}
        )


@final
class NativePlcSource(StrictModule, NonTrainableState):
    """Oriented 3D piecewise-linear complex bound to one source revision.

    Entity identities: regions ``region:r`` (``complex.region_ids[r]``),
    facets ``facet:f``, PLC edges ``edge:e`` (the explicit segments first,
    then the boundary edges of the coplanar facet groups) and vertices
    ``vertex:v``.
    """

    complex: PiecewiseLinearComplex
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self, complex_: PiecewiseLinearComplex, source_id: str, source_revision: str, /
    ) -> None:

        source = normalized_identifier(source_id, "source_id")
        revision = normalized_identifier(source_revision, "source_revision")
        self.complex = complex_
        self.source_id = source
        self.source_revision = revision
        self.binding_id = canonical_fingerprint(
            {
                "kind": "native-plc-source",
                "source_id": source,
                "source_revision": revision,
                "complex": complex_.complex_id,
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[NativePlcSource], value: NativePlcSource, /
    ) -> NativePlcSource:
        """Revalidate a stored PLC source through its constructor."""
        if type(value) is not cls:
            raise TypeError(
                f"{cls.__name__}.stored_fields must have the registered type; got {type(value).__name__}"
            )
        fresh = cls(value.complex, value.source_id, value.source_revision)
        if fresh.binding_id != value.binding_id:
            raise ValueError(
                f"{cls.__name__}.binding_id must match its stored source fields; got {value.binding_id}"
            )
        return fresh


@final
class NativePolyhedralSource(StrictModule, NonTrainableState):
    """A represented PLC and immutable physical power sites and weights."""

    complex: PiecewiseLinearComplex
    sites: np.ndarray
    weights: np.ndarray
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        complex_: PiecewiseLinearComplex,
        source_id: str,
        source_revision: str,
        /,
        *,
        sites: ArrayLike,
        weights: ArrayLike | None = None,
    ) -> None:

        source = normalized_identifier(source_id, "source_id")
        revision = normalized_identifier(source_revision, "source_revision")
        points = np.asarray(sites, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError(
                "NativePolyhedralSource.sites must have nonempty shape (N, 3); got an incompatible shape."
            )
        if not np.all(np.isfinite(points)):
            raise ValueError(
                "NativePolyhedralSource.sites must be finite; got nonfinite coordinates."
            )
        values = (
            np.zeros(points.shape[0], dtype=np.float64)
            if weights is None
            else np.asarray(weights, dtype=np.float64)
        )
        if values.shape != (points.shape[0],) or not np.all(np.isfinite(values)):
            raise ValueError(
                "NativePolyhedralSource.weights must be finite with one value per site; got an incompatible shape or nonfinite values."
            )
        points = np.array(points, dtype=np.float64, copy=True)
        values = np.array(values, dtype=np.float64, copy=True)
        points.setflags(write=False)
        values.setflags(write=False)
        self.complex, self.sites, self.weights = complex_, points, values
        self.source_id, self.source_revision = source, revision
        self.binding_id = canonical_fingerprint(
            {
                "kind": "native-polyhedral-source",
                "source": source,
                "revision": revision,
                "complex": complex_.complex_id,
                "power_sites": array_tree_fingerprint((points, values)),
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[NativePolyhedralSource], value: NativePolyhedralSource, /
    ) -> NativePolyhedralSource:
        """Revalidate stored power sites and their PLC source binding."""
        if type(value) is not cls:
            raise TypeError(
                f"{cls.__name__}.stored_fields must have the registered type; got {type(value).__name__}"
            )
        fresh = cls(
            value.complex,
            value.source_id,
            value.source_revision,
            sites=value.sites,
            weights=value.weights,
        )
        if fresh.binding_id != value.binding_id:
            raise ValueError(
                f"{cls.__name__}.binding_id must match its stored source fields; got {value.binding_id}"
            )
        return fresh


def _layer_source_facets(
    complex_: PiecewiseLinearComplex,
    cap_polygons: np.ndarray,
    boundary: MeshingScope | None,
    domain: MeshingDomain | PiecewiseLinearDomain | None,
    query: SourceBoundaryQuery | None,
    facet_ids: ArrayLike | None,
    source_id: str,
    source_revision: str,
    /,
) -> np.ndarray | None:
    supplied = (
        boundary is not None,
        domain is not None,
        query is not None,
        facet_ids is not None,
    )
    if not any(supplied):
        return None
    if not all(supplied):
        raise ValueError(
            "NativeLayerCoreSource.source_boundary_scope must be supplied with source_domain, fidelity_source, and core_facet_source_ids; got an incomplete source binding."
        )
    match (boundary, domain):
        case (MeshingScope(), MeshingDomain() | PiecewiseLinearDomain()):
            pass
        case _:
            raise TypeError(
                "NativeLayerCoreSource.source_boundary_scope must be paired with a "
                "source_domain; got an incomplete source binding."
            )
    if not isinstance(query, SourceBoundaryQuery):
        raise TypeError(
            "NativeMeshingSource.fidelity_source must implement SourceBoundaryQuery; got an unsupported query."
        )
    binding = (source_id, source_revision)
    if (
        (boundary.source_id, boundary.source_revision) != binding
        or (domain.source_id, domain.source_revision) != binding
        or (query.source_id, query.source_revision) != binding
        or boundary.entity_kind is not MeshingEntityKind.GEOMETRY
        or boundary.entity_dimension != 2
        or query.ambient_dimension != 3
    ):
        raise ValueError(
            "NativeLayerCoreSource.source_boundary_scope must share the source revision and 3D geometry frame with source_domain and fidelity_source; got mismatched source facts."
        )
    values = np.asarray(facet_ids)
    if values.shape != (complex_.facet_count,) or not np.issubdtype(
        values.dtype, np.integer
    ):
        raise TypeError(
            "NativeLayerCoreSource.core_facet_source_ids must identify every core facet with integer source IDs; got an incompatible shape or dtype."
        )
    if np.any(values < -1) or not np.all(
        np.isin(values[values >= 0], np.asarray(boundary.entity_ids))
    ):
        raise ValueError(
            "NativeLayerCoreSource.core_facet_source_ids must name source boundary faces or generated caps; got out-of-scope facet IDs."
        )
    cap_facets = np.unique(np.asarray(complex_.polygon_facets)[cap_polygons])
    if not np.array_equal(np.flatnonzero(values == -1), cap_facets):
        raise ValueError(
            "NativeLayerCoreSource.core_facet_source_ids must omit source face IDs only for generated cap facets; got mismatched cap ancestry."
        )
    for facet in cap_facets:
        rows = np.flatnonzero(np.asarray(complex_.polygon_facets) == facet)
        if not np.all(np.isin(rows, cap_polygons)):
            raise ValueError(
                "NativeLayerCoreSource.cap_polygon_ids must include every polygon of each generated cap facet; got a mixed cap/source facet."
            )
    result = np.array(values, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


@final
class _LayerIdentityMaps(StrictModule):
    vertices: np.ndarray
    polygons: np.ndarray
    regions: np.ndarray
    region_ids: tuple[str, ...] = eqx.field(static=True)
    core_regions: np.ndarray


def _prepare_layer_identity_maps(
    layers: BoundaryLayerMesh,
    complex_: PiecewiseLinearComplex,
    vertex_layer_ids: ArrayLike,
    cap_polygon_ids: ArrayLike,
    layer_regions: ArrayLike,
    region_ids: tuple[str, ...] | None,
    core_region_map: ArrayLike | None,
    /,
) -> _LayerIdentityMaps:
    """Prepare identity arrays after validating layer/core incidence."""
    vertices = np.asarray(vertex_layer_ids)
    polygons = np.asarray(cap_polygon_ids)
    regions = np.asarray(layer_regions)
    for name, values in (
        ("NativeLayerCoreSource.vertex_layer_ids", vertices),
        ("NativeLayerCoreSource.cap_polygon_ids", polygons),
        ("NativeLayerCoreSource.layer_regions", regions),
    ):
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise TypeError(
                f"{name} must be a rank-one integer array; "
                f"got shape {values.shape} and dtype {values.dtype}."
            )
    cap_count = (
        0 if layers.cap is None else sum(block.cell_count for block in layers.cap.blocks)
    )
    cell_count = sum(block.cell_count for block in layers.mesh.blocks)
    if vertices.shape != (complex_.vertices.shape[0],) or np.any(
        (vertices < -1) | (vertices >= layers.mesh.coordinates.shape[0])
    ):
        raise ValueError(
            "NativeLayerCoreSource.vertex_layer_ids must map every PLC vertex to a "
            f"layer vertex or -1; got shape {vertices.shape} or out-of-range IDs."
        )
    if polygons.shape != (cap_count,) or np.any(
        (polygons < 0) | (polygons >= complex_.polygon_offsets.size - 1)
    ):
        raise ValueError(
            "NativeLayerCoreSource.cap_polygon_ids must identify every cap polygon "
            f"in block order; got shape {polygons.shape} or out-of-range IDs."
        )
    if regions.shape != (cell_count,):
        raise ValueError(
            "NativeLayerCoreSource.layer_regions must identify the composite region "
            f"of every layer cell; got shape {regions.shape}, expected {(cell_count,)}."
        )
    composite_regions, core_regions = prepare_layer_region_identity(
        complex_, regions, region_ids, core_region_map
    )
    return _LayerIdentityMaps(
        vertices, polygons, regions, composite_regions, core_regions
    )


def _validate_layer_reference(
    layers: BoundaryLayerMesh,
    regions: np.ndarray,
    composite_regions: tuple[str, ...],
    reference_source: NativeLayerCoreSource | None,
    mapped_domain: MappedReferenceDomain | None,
    source_domain: MeshingDomain | PiecewiseLinearDomain | None,
    fidelity_source: SourceBoundaryQuery | None,
    source: str,
    revision: str,
    /,
) -> None:
    """Validate the independently declared reference columns and physical roots."""
    if (reference_source is None) != (mapped_domain is None):
        raise ValueError(
            "NativeLayerCoreSource.reference_source must be supplied with "
            "mapped_domain; got an incomplete reference binding."
        )
    if reference_source is None or mapped_domain is None:
        return
    if reference_source.reference_source is not None:
        raise ValueError(
            "NativeLayerCoreSource.reference_source must retain its reference partition "
            "directly; got a nested reference source."
        )
    if (mapped_domain.source_id, mapped_domain.source_revision) != (source, revision):
        raise ValueError(
            "NativeLayerCoreSource.mapped_domain must bind the physical source identity "
            f"and revision; got {(mapped_domain.source_id, mapped_domain.source_revision)!r}."
        )
    if (
        reference_source.region_ids != composite_regions
        or mapped_domain.region_ids != composite_regions
    ):
        raise ValueError(
            "NativeLayerCoreSource.reference_source.region_ids must match the composite "
            f"materials together with mapped_domain.region_ids; got {reference_source.region_ids!r} "
            f"and {mapped_domain.region_ids!r}."
        )
    if source_domain is None or fidelity_source is None:
        raise ValueError(
            "NativeLayerCoreSource.source_domain must be supplied with fidelity_source "
            "for mapped composition; got incomplete physical geometry."
        )
    if not np.array_equal(reference_source.layer_regions, regions):
        raise ValueError(
            "NativeLayerCoreSource.reference_source.layer_regions must match the "
            "physical layer assignments; got different material arrays."
        )
    if not np.array_equal(
        reference_source.layers.mesh.vertex_global_ids, layers.mesh.vertex_global_ids
    ):
        raise ValueError(
            "NativeLayerCoreSource.reference_source.layers.mesh.vertex_global_ids must "
            "match the physical registry; got different vertex IDs."
        )
    if not np.array_equal(
        reference_source.layers.layer_index, layers.layer_index
    ) or not np.array_equal(reference_source.layers.column_index, layers.column_index):
        raise ValueError(
            "NativeLayerCoreSource.reference_source.layers.layer_index must match the "
            "physical intervals together with column_index; got different index arrays."
        )


@final
class _LayerPeriodicAncestry(StrictModule):
    representatives: np.ndarray | None
    shifts: np.ndarray | None
    polygon_pairs: np.ndarray | None


def _prepare_layer_periodic_ancestry(
    representatives: ArrayLike | None,
    shifts: ArrayLike | None,
    polygon_pairs: ArrayLike | None,
    /,
) -> _LayerPeriodicAncestry:
    """Freeze the jointly supplied signed-integer seam ancestry."""
    values = (representatives, shifts, polygon_pairs)
    if any(value is None for value in values) and any(
        value is not None for value in values
    ):
        raise ValueError(
            "NativeLayerCoreSource.core_periodic_ancestry must include representatives, "
            "shifts, and seam polygon pairs together; got an incomplete ancestry."
        )
    arrays: list[np.ndarray | None] = []
    for value in values:
        if value is None:
            arrays.append(None)
            continue
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError(
                "NativeLayerCoreSource.core_periodic_ancestry must use integer arrays; "
                f"got dtype {array.dtype}."
            )
        storage = np.iinfo(np.int64)
        if np.any(array < storage.min) or np.any(array > storage.max):
            raise ValueError(
                "NativeLayerCoreSource.core_periodic_ancestry must fit signed int64; "
                "got out-of-range integers."
            )
        array = np.array(array, dtype=np.int64, copy=True)
        array.setflags(write=False)
        arrays.append(array)
    return _LayerPeriodicAncestry(*arrays)


@final
class NativeLayerCoreSource(StrictModule, NonTrainableState):
    """Bind certified layers to a PLC core through shared entity identities.

    ``region_ids`` defines the composite material namespace;
    ``core_region_map`` maps each bounded core-local region into it, while
    ``layer_regions`` indexes it directly. Supplying neither declares the
    explicit identity composition of the core namespace. Wall-only materials
    belong here, never as unbounded regions in the core PLC.

    Periodic layers require ``core_vertex_representatives`` and
    ``core_vertex_shifts`` for every original PLC vertex in the accepted layer
    identification, together with ``core_seam_polygon_pairs`` indexing the
    distinct triangles on the two lifted copies of each seam. These integer
    construction facts are checked against cap ancestry, oriented facet
    permutations, and material incidence before native fill. Boundary Steiner
    nodes are forbidden; generated ordinary interior vertices are singleton
    orbits, not nearest-neighbor assignments to boundary seams.

    ``reference_source`` and ``mapped_domain`` jointly declare independent
    source roots for curved composition. Native fill retains every internal
    root incidence sheet in the reference PLC; reference charts then pull back
    the physical root maps. ``layers`` retains the physical wall/column
    realization independently of the reference mesh.

    ``fidelity_source`` owns fixed scientific numerical state. Its arrays are
    dynamic PyTree leaves, not static structure or discoverable training
    parameters; the containing source is ``NonTrainableState``.
    """

    layers: BoundaryLayerMesh
    complex: PiecewiseLinearComplex
    vertex_layer_ids: np.ndarray
    cap_polygon_ids: np.ndarray
    layer_regions: np.ndarray
    region_ids: tuple[str, ...] = eqx.field(static=True)
    core_region_map: np.ndarray
    source_boundary_scope: MeshingScope | None
    source_domain: MeshingDomain | PiecewiseLinearDomain | None
    fidelity_source: SourceBoundaryQuery | None = fixed_field()
    core_facet_source_ids: np.ndarray | None
    core_vertex_representatives: np.ndarray | None
    core_vertex_shifts: np.ndarray | None
    core_seam_polygon_pairs: np.ndarray | None
    reference_source: NativeLayerCoreSource | None
    mapped_domain: MappedReferenceDomain | None
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        layers: BoundaryLayerMesh,
        complex_: PiecewiseLinearComplex,
        source_id: str,
        source_revision: str,
        /,
        *,
        vertex_layer_ids: ArrayLike,
        cap_polygon_ids: ArrayLike,
        layer_regions: ArrayLike,
        region_ids: tuple[str, ...] | None = None,
        core_region_map: ArrayLike | None = None,
        source_boundary_scope: MeshingScope | None = None,
        source_domain: MeshingDomain | PiecewiseLinearDomain | None = None,
        fidelity_source: SourceBoundaryQuery | None = None,
        core_facet_source_ids: ArrayLike | None = None,
        core_vertex_representatives: ArrayLike | None = None,
        core_vertex_shifts: ArrayLike | None = None,
        core_seam_polygon_pairs: ArrayLike | None = None,
        reference_source: NativeLayerCoreSource | None = None,
        mapped_domain: MappedReferenceDomain | None = None,
    ) -> None:
        source = normalized_identifier(source_id, "source_id")
        revision = normalized_identifier(source_revision, "source_revision")
        identity = _prepare_layer_identity_maps(
            layers,
            complex_,
            vertex_layer_ids,
            cap_polygon_ids,
            layer_regions,
            region_ids,
            core_region_map,
        )
        vertices, polygons, regions = (
            identity.vertices,
            identity.polygons,
            identity.regions,
        )
        composite_regions, core_regions = identity.region_ids, identity.core_regions
        _validate_layer_reference(
            layers,
            regions,
            composite_regions,
            reference_source,
            mapped_domain,
            source_domain,
            fidelity_source,
            source,
            revision,
        )
        original_facets = _layer_source_facets(
            complex_,
            polygons,
            source_boundary_scope,
            source_domain,
            fidelity_source,
            core_facet_source_ids,
            source,
            revision,
        )
        vertices = np.array(vertices, dtype=np.int64, copy=True)
        polygons = np.array(polygons, dtype=np.int64, copy=True)
        regions = np.array(regions, dtype=np.int32, copy=True)
        for values in (vertices, polygons, regions):
            values.setflags(write=False)
        periodic = _prepare_layer_periodic_ancestry(
            core_vertex_representatives, core_vertex_shifts, core_seam_polygon_pairs
        )
        self.layers = layers
        self.complex = complex_
        self.vertex_layer_ids = vertices
        self.cap_polygon_ids = polygons
        self.layer_regions = regions
        self.region_ids = composite_regions
        self.core_region_map = core_regions
        self.source_boundary_scope = source_boundary_scope
        self.source_domain = source_domain
        self.fidelity_source = fidelity_source
        self.core_facet_source_ids = original_facets
        self.core_vertex_representatives = periodic.representatives
        self.core_vertex_shifts = periodic.shifts
        self.core_seam_polygon_pairs = periodic.polygon_pairs
        self.reference_source = reference_source
        self.mapped_domain = mapped_domain
        self.source_id = source
        self.source_revision = revision
        self.binding_id = canonical_fingerprint(
            {
                "kind": "native-layer-core-source",
                "source": source,
                "revision": revision,
                "layers": layers.result_id,
                "complex": complex_.complex_id,
                "identity_maps": array_tree_fingerprint((vertices, polygons, regions)),
                "region_ids": composite_regions,
                "core_region_map": array_tree_fingerprint(core_regions),
                "original_boundary": None
                if source_boundary_scope is None
                else source_boundary_scope.scope_id,
                "original_domain": None
                if source_domain is None
                else source_domain.domain_id,
                "core_facet_source_ids": array_tree_fingerprint(original_facets),
                "core_periodic_ancestry": array_tree_fingerprint(
                    (periodic.representatives, periodic.shifts, periodic.polygon_pairs)
                ),
                "reference_source": None
                if reference_source is None
                else reference_source.binding_id,
                "mapped_domain": None
                if mapped_domain is None
                else mapped_domain.domain_id,
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[NativeLayerCoreSource], value: NativeLayerCoreSource, /
    ) -> NativeLayerCoreSource:
        """Revalidate stored layer/core incidence and source bindings."""
        if type(value) is not cls:
            raise TypeError(
                f"{cls.__name__}.stored_fields must have the registered type; got {type(value).__name__}"
            )
        fresh = cls(
            value.layers,
            value.complex,
            value.source_id,
            value.source_revision,
            vertex_layer_ids=value.vertex_layer_ids,
            cap_polygon_ids=value.cap_polygon_ids,
            layer_regions=value.layer_regions,
            region_ids=value.region_ids,
            core_region_map=value.core_region_map,
            source_boundary_scope=value.source_boundary_scope,
            source_domain=value.source_domain,
            fidelity_source=value.fidelity_source,
            core_facet_source_ids=value.core_facet_source_ids,
            core_vertex_representatives=value.core_vertex_representatives,
            core_vertex_shifts=value.core_vertex_shifts,
            core_seam_polygon_pairs=value.core_seam_polygon_pairs,
            reference_source=value.reference_source,
            mapped_domain=value.mapped_domain,
        )
        if fresh.binding_id != value.binding_id:
            raise ValueError(
                f"{cls.__name__}.binding_id must match its stored source fields; got {value.binding_id}"
            )
        return fresh


@final
class NativeSurfaceEnvelopeSource(StrictModule, NonTrainableState):
    """A certified envelope plus explicit solid-region identity for PLC fill."""

    envelope: envelope_owner.SurfaceEnvelope
    plc_source: NativePlcSource
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self, envelope: envelope_owner.SurfaceEnvelope, region_id: str, /
    ) -> None:
        if not isinstance(region_id, str):
            raise TypeError(
                "NativeSurfaceEnvelopeSource.region_id must be a string; "
                f"got {type(region_id).__name__}."
            )
        if not region_id:
            raise ValueError(
                "NativeSurfaceEnvelopeSource.region_id must be nonempty; got ''."
            )

        with (
            NativeExecutionScope(envelope.policy.native_limits()) as scope,
            scope.host_workspace() as workspace,
        ):
            workspace.retain_owner(envelope)
            model = envelope.repaired.model
            points = np.asarray(model.mesh.coordinates, dtype=np.float64)
            triangles = np.asarray(model.mesh.blocks[0].vertices, dtype=np.int64)
            owners = family_host_array((triangles.shape[0],), np.int64, scope=scope)
            incidence = family_host_array((triangles.shape[0], 2), np.int64, scope=scope)
            with scope.batch(triangles.shape[0]) as ticket:
                for row in range(owners.size):
                    owners[row] = row
                ticket.consume(triangles.shape[0])
                incidence[:, 0] = -1
                incidence[:, 1] = 0
            complex_ = PiecewiseLinearComplex(
                points,
                tuple(triangles),
                owners,
                incidence,
                (region_id,),
                boundary="conforming",
            )
            plc_source = NativePlcSource(
                complex_, model.metadata.source_id, model.metadata.source_revision
            )
            workspace.retain_owner(plc_source)
        self.envelope, self.plc_source = envelope, plc_source
        self.source_id = plc_source.source_id
        self.source_revision = plc_source.source_revision
        self.binding_id = canonical_fingerprint(
            {
                "kind": "native-surface-envelope-source",
                "envelope": envelope.envelope_id,
                "plc": plc_source.binding_id,
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[NativeSurfaceEnvelopeSource], value: NativeSurfaceEnvelopeSource, /
    ) -> NativeSurfaceEnvelopeSource:
        """Revalidate an envelope source and its declared solid region."""
        if type(value) is not cls:
            raise TypeError(
                f"{cls.__name__}.stored_fields must have the registered type; got {type(value).__name__}"
            )
        regions = value.plc_source.complex.region_ids
        if len(regions) != 1:
            raise ValueError(
                f"{cls.__name__}.region_ids must contain one solid region; got {regions}"
            )
        fresh = cls(value.envelope, regions[0])
        if (fresh.source_id, fresh.source_revision, fresh.binding_id) != (
            value.source_id,
            value.source_revision,
            value.binding_id,
        ):
            raise ValueError(
                f"{cls.__name__}.binding_id must match its stored source fields; got {value.binding_id}"
            )
        return fresh


def _fidelity_binding(
    source: SourceBoundaryQuery,
    source_id: str,
    source_revision: str,
    maximum_deviation: float,
    /,
) -> float:
    if not isinstance(source, SourceBoundaryQuery):
        raise TypeError(
            "NativeMeshingSource.fidelity_source must implement SourceBoundaryQuery; got an unsupported query."
        )
    if (source.source_id, source.source_revision) != (source_id, source_revision):
        raise ValueError(
            "NativeMeshingSource.fidelity_source must bind the source identity and revision; got a different query binding."
        )
    return positive_finite_float(maximum_deviation, "maximum_deviation")


def _approximation_binding(
    domain: PiecewiseLinearDomain | MappedReferenceDomain | None,
    block_regions: tuple[tuple[str, str], ...],
    block_names: tuple[str, ...],
    source_id: str,
    /,
    *,
    mapped: bool = False,
) -> tuple[tuple[str, str], ...]:
    match domain:
        case None:
            if block_regions:
                raise ValueError(
                    "NativeMeshingSource.block_regions must have a declared approximation "
                    "domain; got region declarations without a domain."
                )
            return ()
        case MappedReferenceDomain() if not mapped:
            raise TypeError(
                "NativeStructuredSource.domain must be PiecewiseLinearDomain or None; "
                "got a mapped reference domain."
            )
        case PiecewiseLinearDomain() | MappedReferenceDomain():
            pass
        case _:
            raise TypeError(
                "NativeMeshingSource.domain must be PiecewiseLinearDomain, "
                "MappedReferenceDomain, or None; got an unsupported domain kind."
            )
    if domain.source_id != source_id:
        raise ValueError(
            "NativeMeshingSource.domain.source_id must match the source binding; got a different approximation source."
        )
    if any(
        not isinstance(pair, tuple)
        or len(pair) != 2
        or not all(isinstance(value, str) for value in pair)
        for pair in block_regions
    ):
        raise TypeError(
            "NativeMeshingSource.block_regions must contain pairs of block and region identifiers; got a malformed declaration."
        )
    mapping = tuple(sorted(block_regions))
    if len({name for name, _ in mapping}) != len(mapping):
        raise ValueError(
            "NativeMeshingSource.block_regions must identify each block once; got duplicate block declarations."
        )
    if any(
        name not in block_names or region not in domain.region_ids
        for name, region in mapping
    ):
        raise ValueError(
            "NativeMeshingSource.block_regions must name declared blocks and source regions; got unknown identifiers."
        )
    if len(domain.region_ids) > 1 and {name for name, _ in mapping} != set(block_names):
        raise ValueError(
            "NativeMeshingSource.block_regions must identify every block for a multimaterial approximation; got missing block declarations."
        )
    return mapping


@final
class NativeStructuredSource(StrictModule, NonTrainableState):
    """Bind blocks and gluing to an independently declared boundary query."""

    blocks: tuple[TransfiniteBlock, ...]
    interfaces: tuple[BlockInterfaceControl, ...]
    fidelity_source: SourceBoundaryQuery
    maximum_deviation: float = eqx.field(static=True)
    domain: PiecewiseLinearDomain | None
    block_regions: tuple[tuple[str, str], ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        blocks: tuple[TransfiniteBlock, ...],
        interfaces: tuple[BlockInterfaceControl, ...],
        source_id: str,
        source_revision: str,
        /,
        *,
        fidelity_source: SourceBoundaryQuery,
        maximum_deviation: float,
        domain: PiecewiseLinearDomain | None = None,
        block_regions: tuple[tuple[str, str], ...] = (),
    ) -> None:
        if not blocks or any(not isinstance(block, TransfiniteBlock) for block in blocks):
            raise TypeError(
                "NativeStructuredSource.blocks must contain at least one TransfiniteBlock; got empty blocks or an unsupported block kind."
            )
        if any(
            not isinstance(interface, BlockInterfaceControl) for interface in interfaces
        ):
            raise TypeError(
                "NativeStructuredSource.interfaces must contain BlockInterfaceControl values; got an unsupported interface kind."
            )
        if len({block.name for block in blocks}) != len(blocks):
            raise ValueError(
                "NativeStructuredSource.blocks.name must be unique; got duplicate block names."
            )
        source = normalized_identifier(source_id, "source_id")
        revision = normalized_identifier(source_revision, "source_revision")
        deviation = _fidelity_binding(
            fidelity_source, source, revision, maximum_deviation
        )
        regions = _approximation_binding(
            domain, block_regions, tuple(block.name for block in blocks), source
        )
        self.blocks = tuple(sorted(blocks, key=lambda block: block.name))
        self.interfaces = tuple(
            sorted(interfaces, key=lambda interface: interface.control_id)
        )
        self.fidelity_source = fidelity_source
        self.maximum_deviation = deviation
        self.domain, self.block_regions = domain, regions
        self.source_id, self.source_revision = source, revision
        self.binding_id = canonical_fingerprint(
            {
                "kind": "native-structured-source",
                "source": source,
                "revision": revision,
                "blocks": tuple(block.block_id for block in self.blocks),
                "interfaces": tuple(
                    interface.control_id for interface in self.interfaces
                ),
                "maximum_deviation": deviation,
                "approximation_domain": None if domain is None else domain.domain_id,
                "block_regions": regions,
            }
        )


@final
class NativeSweepSource(StrictModule, NonTrainableState):
    """Bind a profile and sweep declaration to a source boundary query."""

    profile: cell_mesh_owner.CellMesh
    profile_geometry: cell_geometry_owner.CellGeometrySpec
    control: SweepControl
    fidelity_source: SourceBoundaryQuery
    maximum_deviation: float = eqx.field(static=True)
    domain: PiecewiseLinearDomain | MappedReferenceDomain | None
    root_correspondence: CellGeometryRestrictionSource | None
    block_regions: tuple[tuple[str, str], ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        profile: cell_mesh_owner.CellMesh,
        control: SweepControl,
        source_id: str,
        source_revision: str,
        /,
        *,
        fidelity_source: SourceBoundaryQuery,
        maximum_deviation: float,
        domain: PiecewiseLinearDomain | MappedReferenceDomain | None = None,
        block_regions: tuple[tuple[str, str], ...] = (),
        profile_geometry: cell_geometry_owner.CellGeometrySpec | None = None,
        root_correspondence: CellGeometryRestrictionSource | None = None,
    ) -> None:

        geometry = (
            cell_geometry_owner.CellGeometrySpec.affine(profile)
            if profile_geometry is None
            else profile_geometry
        )

        geometry.resolve(profile)
        source = normalized_identifier(source_id, "source_id")
        revision = normalized_identifier(source_revision, "source_revision")
        deviation = _fidelity_binding(
            fidelity_source, source, revision, maximum_deviation
        )
        match domain:
            case MappedReferenceDomain():
                if domain.source_revision != revision:
                    raise ValueError(
                        "NativeSweepSource.domain.source_revision must match source_revision; "
                        f"got {domain.source_revision!r}, expected {revision!r}."
                    )
                if root_correspondence is None:
                    raise TypeError(
                        "NativeSweepSource.root_correspondence must be supplied for a "
                        "mapped domain; got None."
                    )
                if (
                    root_correspondence.source_geometry_id
                    != cell_geometry_id(domain.source_geometry)
                    or root_correspondence.source_topology_id
                    != domain.reference_mesh.topology_id
                ):
                    raise ValueError(
                        "NativeSweepSource.root_correspondence.source_geometry_id must "
                        "match the mapped domain together with source_topology_id; got a different root binding."
                    )
                expected_names = {f"sweep:{block.name}" for block in profile.blocks}
                if set(root_correspondence.block_names) != expected_names:
                    raise ValueError(
                        "NativeSweepSource.root_correspondence.block_names must name every "
                        "generated block; got a different block registry."
                    )
                parents = root_correspondence.block_parent_cell_ids
                if any(
                    parents[f"sweep:{block.name}"].shape[0]
                    != control.schedule.layer_count * block.cell_count
                    for block in profile.blocks
                ):
                    raise ValueError(
                        "NativeSweepSource.root_correspondence.block_parent_cell_ids must "
                        "identify every layer-major column cell; got different cell counts."
                    )
            case PiecewiseLinearDomain() | None:
                if root_correspondence is not None:
                    raise ValueError(
                        "NativeSweepSource.root_correspondence must have a mapped domain; "
                        "got correspondence with an unmapped domain."
                    )
            case _:
                raise TypeError(
                    "NativeSweepSource.domain must be PiecewiseLinearDomain, "
                    f"MappedReferenceDomain, or None; got {type(domain).__name__}."
                )
        regions = _approximation_binding(
            domain,
            block_regions,
            tuple(block.name for block in profile.blocks),
            source,
            mapped=True,
        )
        self.profile, self.control = profile, control
        self.profile_geometry = geometry
        self.root_correspondence = root_correspondence
        self.fidelity_source = fidelity_source
        self.maximum_deviation = deviation
        self.domain, self.block_regions = domain, regions
        self.source_id, self.source_revision = source, revision
        self.binding_id = canonical_fingerprint(
            {
                "kind": "native-sweep-source",
                "source": source,
                "revision": revision,
                "profile": profile.mesh_id,
                "control": control.control_id,
                "profile_geometry": cell_geometry_id(geometry),
                "root_correspondence": None
                if root_correspondence is None
                else root_correspondence.restriction_source_id,
                "maximum_deviation": deviation,
                "approximation_domain": None if domain is None else domain.domain_id,
                "block_regions": regions,
            }
        )


@final
class _MappedHexBinding(StrictModule):
    binding_id: str = eqx.field(static=True)


def _prepare_mapped_hex_binding(
    reference: NativePlcSource, domain: MappedReferenceDomain, /
) -> _MappedHexBinding:
    """Validate the declared reference PLC and derive the mapped binding."""
    declared = declared_plc_domain(reference.complex, reference.source_id)
    if declared.domain_id != domain.reference_domain.domain_id:
        raise ValueError(
            "NativeMappedHexSource.domain.reference_domain.domain_id must match the "
            f"declared reference PLC; got {domain.reference_domain.domain_id!r}, "
            f"expected {declared.domain_id!r}."
        )
    return _MappedHexBinding(
        canonical_fingerprint(
            {
                "kind": "native-mapped-hex-source",
                "reference": reference.binding_id,
                "mapped_domain": domain.domain_id,
            }
        )
    )


@final
class NativeMappedHexSource(StrictModule, NonTrainableState):
    """Bind physical root maps to an independently declared reference PLC."""

    reference: NativePlcSource
    domain: MappedReferenceDomain
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self, reference: NativePlcSource, domain: MappedReferenceDomain, /
    ) -> None:
        binding = _prepare_mapped_hex_binding(reference, domain)
        self.reference, self.domain = reference, domain
        self.source_id, self.source_revision = domain.source_id, domain.source_revision
        self.binding_id = binding.binding_id




__all__ = [
    "NativeCurveSource",
    "NativeImplicitSource",
    "NativeLayerCoreSource",
    "NativeMappedHexSource",
    "NativePlcSource",
    "NativePolyhedralSource",
    "NativePlanarSource",
    "NativeSurfaceSource",
    "NativeStructuredSource",
    "NativeSweepSource",
    "NativeSurfaceEnvelopeSource",
    "source_entity_id",
]
