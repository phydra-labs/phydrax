#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Revision-bound source bindings admitted by the native meshing provider.

Each binding attaches one immutable source revision to an existing geometry
owner. It adds no geometry of its own: represented geometry, entity identity,
and queries remain with ``phydrax.geometry``. Source entity identifiers used in
published geometry associations are ``<revision>:<kind>:<index>`` strings of
the kinds named on each binding.
"""

from __future__ import annotations

from typing import final

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import fixed_field, NonTrainableState
from ...discretization import CellGeometrySpec, CellMesh, PreparedTensorGrid
from ...discretization._cell_geometry import CellGeometryRestrictionSource
from ...discretization._cell_geometry_validity import cell_geometry_id
from ...geometry import BoundaryAtlas, CompiledGeometry, PlanarMeshRegion, SegmentMesh
from ...geometry._compartments import CompartmentMeshingSource
from ...geometry._mapped_reference_domain import MappedReferenceDomain
from ...geometry._mesh_certificates import PiecewiseLinearDomain, SourceBoundaryQuery
from ...geometry._meshing_domain import MeshingDomain
from ...geometry.multiregion_surface._label_extraction import LabelFieldVolumeBinding
from ...typing import checked
from .._boundary_layer import BoundaryLayerMesh
from .._controls import BlockInterfaceControl
from .._scope import MeshingEntityKind, MeshingScope
from .._structured import TransfiniteBlock
from .._surface_envelope import SurfaceEnvelope
from .._sweep import SweepControl
from .._volume_generation import declared_plc_domain, PiecewiseLinearComplex
from ._native_periodic import NativePeriodicSource


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    identifier = value.strip()
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


def source_entity_id(revision: str, kind: str, index: int, /) -> str:
    """Canonical identifier of one native source entity."""

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

    region: PlanarMeshRegion
    embedded: SegmentMesh | None
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        region: PlanarMeshRegion,
        source_revision: str,
        /,
        *,
        embedded: SegmentMesh | None = None,
    ) -> None:
        if not isinstance(region, PlanarMeshRegion):
            raise TypeError("region must be PlanarMeshRegion.")
        if embedded is not None and not isinstance(embedded, SegmentMesh):
            raise TypeError("embedded must be SegmentMesh or None.")
        revision = _identifier(source_revision, "source_revision")
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


@final
class NativeImplicitSource(StrictModule, NonTrainableState):
    """Compiled implicit geometry with its discovery lattice and identity.

    The discovered zero set is the source entity ``implicit-zero-set``.
    """

    geometry: CompiledGeometry
    grid: PreparedTensorGrid
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        geometry: CompiledGeometry,
        grid: PreparedTensorGrid,
        source_id: str,
        source_revision: str,
        /,
    ) -> None:
        source = _identifier(source_id, "source_id")
        self.geometry = geometry
        self.grid = grid
        self.source_id = source
        self.source_revision = _identifier(source_revision, "source_revision")


@final
class NativeCurveSource(StrictModule, NonTrainableState):
    """Parametric curve charts or a straight segment network.

    A ``BoundaryAtlas`` with one-dimensional charts supplies curves on the unit
    reference interval, identified by chart index (several charts may share
    one atlas source entity, e.g. the arcs of a circle); a ``SegmentMesh``
    supplies straight curves identified by edge row. Curve entities are
    ``curve:<id>``.
    """

    curves: BoundaryAtlas | SegmentMesh
    ambient_dimension: int = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)

    def __init__(
        self, curves: BoundaryAtlas | SegmentMesh, source_revision: str, /
    ) -> None:
        match curves:
            case BoundaryAtlas():
                if curves.reference_dimension != 1:
                    raise ValueError("Curve atlases must have one-dimensional charts.")
                if curves.num_charts == 0:
                    raise ValueError("Curve atlases must contain at least one chart.")
                ambient = curves.ambient_dimension
            case SegmentMesh():
                ambient = curves.vertices.shape[1]
            case _:
                raise TypeError("curves must be a BoundaryAtlas or SegmentMesh.")
        self.curves = curves
        self.ambient_dimension = ambient
        self.source_id = curves.source_id
        self.source_revision = _identifier(source_revision, "source_revision")


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

    def __init__(self, domain: MeshingDomain, /) -> None:
        if not isinstance(domain, MeshingDomain):
            raise TypeError("domain must be MeshingDomain.")
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

    def __init__(
        self, complex_: PiecewiseLinearComplex, source_id: str, source_revision: str, /
    ) -> None:
        if not isinstance(complex_, PiecewiseLinearComplex):
            raise TypeError("complex_ must be PiecewiseLinearComplex.")
        source = _identifier(source_id, "source_id")
        revision = _identifier(source_revision, "source_revision")
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


@final
class NativePolyhedralSource(StrictModule, NonTrainableState):
    """A represented PLC and immutable physical power sites and weights."""

    complex: PiecewiseLinearComplex
    sites: np.ndarray
    weights: np.ndarray
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

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
        if not isinstance(complex_, PiecewiseLinearComplex):
            raise TypeError("complex_ must be PiecewiseLinearComplex.")
        source = _identifier(source_id, "source_id")
        revision = _identifier(source_revision, "source_revision")
        points = np.asarray(sites, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError("sites must have nonempty shape (N, 3).")
        if not np.all(np.isfinite(points)):
            raise ValueError("sites must be finite.")
        values = (
            np.zeros(points.shape[0], dtype=np.float64)
            if weights is None
            else np.asarray(weights, dtype=np.float64)
        )
        if values.shape != (points.shape[0],) or not np.all(np.isfinite(values)):
            raise ValueError("weights must be finite with one value per site.")
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
            "Original layer source scope, domain, query and facet ancestry are required together."
        )
    if not isinstance(boundary, MeshingScope):
        raise TypeError("source_boundary_scope must be MeshingScope.")
    if not isinstance(domain, (MeshingDomain, PiecewiseLinearDomain)):
        raise TypeError("source_domain must be MeshingDomain or PiecewiseLinearDomain.")
    if not isinstance(query, SourceBoundaryQuery):
        raise TypeError("fidelity_source must implement SourceBoundaryQuery.")
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
            "Original layer source facts must share authoritative geometry identity and frame dimension."
        )
    values = np.asarray(facet_ids)
    if values.shape != (complex_.facet_count,) or not np.issubdtype(
        values.dtype, np.integer
    ):
        raise TypeError(
            "core_facet_source_ids must identify every core facet with integer source IDs."
        )
    if np.any(values < -1) or not np.all(
        np.isin(values[values >= 0], np.asarray(boundary.entity_ids))
    ):
        raise ValueError(
            "Core facet ancestry must name original boundary faces or generated caps."
        )
    cap_facets = np.unique(np.asarray(complex_.polygon_facets)[cap_polygons])
    if not np.array_equal(np.flatnonzero(values == -1), cap_facets):
        raise ValueError(
            "Only exact generated cap facets may lack original source face IDs."
        )
    for facet in cap_facets:
        rows = np.flatnonzero(np.asarray(complex_.polygon_facets) == facet)
        if not np.all(np.isin(rows, cap_polygons)):
            raise ValueError(
                "Generated cap facets cannot include original boundary polygons."
            )
    result = np.array(values, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


@final
class NativeLayerCoreSource(StrictModule, NonTrainableState):
    """Certified layers and an exact PLC core with explicit shared identities.

    ``region_ids`` owns the complete composite material namespace;
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
    root incidence sheet in the reference PLC; exact reference charts then
    pull back the original root maps. ``layers`` remains the actual physical
    wall/column realization, not an inverse-fitted or relabeled reference mesh.

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
        if not isinstance(layers, BoundaryLayerMesh):
            raise TypeError("layers must be BoundaryLayerMesh.")
        if not isinstance(complex_, PiecewiseLinearComplex):
            raise TypeError("complex_ must be PiecewiseLinearComplex.")
        source = _identifier(source_id, "source_id")
        revision = _identifier(source_revision, "source_revision")
        vertices = np.asarray(vertex_layer_ids)
        polygons = np.asarray(cap_polygon_ids)
        regions = np.asarray(layer_regions)
        for name, values in (
            ("vertex_layer_ids", vertices),
            ("cap_polygon_ids", polygons),
            ("layer_regions", regions),
        ):
            if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
                raise TypeError(f"{name} must be a rank-one integer array.")
        cap_count = (
            0
            if layers.cap is None
            else sum(block.cell_count for block in layers.cap.blocks)
        )
        cell_count = sum(block.cell_count for block in layers.mesh.blocks)
        if vertices.shape != (complex_.vertices.shape[0],) or np.any(
            (vertices < -1) | (vertices >= layers.mesh.coordinates.shape[0])
        ):
            raise ValueError(
                "vertex_layer_ids must map every PLC vertex to a layer vertex or -1."
            )
        if polygons.shape != (cap_count,) or np.any(
            (polygons < 0) | (polygons >= complex_.polygon_offsets.size - 1)
        ):
            raise ValueError(
                "cap_polygon_ids must identify every cap polygon in block order."
            )
        if regions.shape != (cell_count,):
            raise ValueError(
                "layer_regions must identify the composite region of every layer cell."
            )
        from .._layer_core_regions import prepare_layer_region_identity

        composite_regions, core_regions = prepare_layer_region_identity(
            complex_, regions, region_ids, core_region_map
        )
        if (reference_source is None) != (mapped_domain is None):
            raise ValueError(
                "Mapped layer/core composition requires its complete reference source and mapped domain together."
            )
        if reference_source is not None:
            if not isinstance(reference_source, NativeLayerCoreSource) or not isinstance(
                mapped_domain, MappedReferenceDomain
            ):
                raise TypeError(
                    "Mapped layer/core composition requires NativeLayerCoreSource and MappedReferenceDomain."
                )
            if reference_source.reference_source is not None:
                raise ValueError(
                    "A layer/core reference source must own its original reference partition directly."
                )
            if (mapped_domain.source_id, mapped_domain.source_revision) != (
                source,
                revision,
            ):
                raise ValueError(
                    "Mapped layer/core roots must bind the original physical source identity and revision."
                )
            if (
                reference_source.region_ids != composite_regions
                or mapped_domain.region_ids != composite_regions
            ):
                raise ValueError(
                    "Mapped layer/core composition must retain every original material identity."
                )
            if source_domain is None or fidelity_source is None:
                raise ValueError(
                    "Mapped layer/core composition requires independent original physical geometry and fidelity."
                )
            if not np.array_equal(reference_source.layer_regions, regions):
                raise ValueError(
                    "Reference source columns must retain the actual physical layer material assignments."
                )
            if not np.array_equal(
                reference_source.layers.mesh.vertex_global_ids,
                layers.mesh.vertex_global_ids,
            ):
                raise ValueError(
                    "Reference source columns must retain the actual physical layer vertex registry."
                )
            if not np.array_equal(
                reference_source.layers.layer_index, layers.layer_index
            ) or not np.array_equal(
                reference_source.layers.column_index,
                layers.column_index,
            ):
                raise ValueError(
                    "Reference source columns must retain every actual physical interval and column identity."
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
        periodic_values = (
            core_vertex_representatives,
            core_vertex_shifts,
            core_seam_polygon_pairs,
        )
        if any(value is None for value in periodic_values) and any(
            value is not None for value in periodic_values
        ):
            raise ValueError(
                "Core periodic ancestry requires representatives, shifts, and explicit seam polygon pairs together."
            )
        periodic_arrays: list[np.ndarray | None] = []
        for value in periodic_values:
            if value is None:
                periodic_arrays.append(None)
            else:
                array = np.asarray(value)
                if not np.issubdtype(array.dtype, np.integer):
                    raise TypeError("Core periodic ancestry must use integer arrays.")
                storage = np.iinfo(np.int64)
                if np.any(array < storage.min) or np.any(array > storage.max):
                    raise ValueError(
                        "Core periodic ancestry exceeds the canonical signed integer representation."
                    )
                array = np.array(array, dtype=np.int64, copy=True)
                array.setflags(write=False)
                periodic_arrays.append(array)
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
        self.core_vertex_representatives = periodic_arrays[0]
        self.core_vertex_shifts = periodic_arrays[1]
        self.core_seam_polygon_pairs = periodic_arrays[2]
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
                "core_periodic_ancestry": array_tree_fingerprint(tuple(periodic_arrays)),
                "reference_source": None
                if reference_source is None
                else reference_source.binding_id,
                "mapped_domain": None
                if mapped_domain is None
                else mapped_domain.domain_id,
            }
        )


@final
class NativeSurfaceEnvelopeSource(StrictModule, NonTrainableState):
    """A certified envelope plus explicit solid-region identity for PLC fill."""

    envelope: SurfaceEnvelope
    plc_source: NativePlcSource
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(self, envelope: SurfaceEnvelope, region_id: str, /) -> None:
        if not isinstance(envelope, SurfaceEnvelope):
            raise TypeError("envelope must be SurfaceEnvelope.")
        if not isinstance(region_id, str) or not region_id:
            raise ValueError("region_id must be an explicit nonempty identifier.")
        from ... import _meshcore
        from .._quad_generation import _family_host_array

        storage = _meshcore.current_native_host_workspace()
        if storage is not None:
            storage.retain_owner(envelope)
        model = envelope.repaired.model
        points = np.asarray(model.mesh.coordinates, dtype=np.float64)
        triangles = np.asarray(model.mesh.blocks[0].vertices, dtype=np.int64)
        owners = _family_host_array((triangles.shape[0],), np.int64)
        incidence = _family_host_array((triangles.shape[0], 2), np.int64)
        budget = _meshcore.current_native_execution_budget()
        if budget is not None:
            budget.charge(work=triangles.shape[0])
        for row in range(owners.size):
            owners[row] = row
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
        if storage is not None:
            storage.retain_owner(plc_source)
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


def _fidelity_binding(
    source: SourceBoundaryQuery,
    source_id: str,
    source_revision: str,
    maximum_deviation: float,
    /,
) -> float:
    if not isinstance(source, SourceBoundaryQuery):
        raise TypeError("fidelity_source must implement SourceBoundaryQuery.")
    if (source.source_id, source.source_revision) != (source_id, source_revision):
        raise ValueError(
            "The fidelity query must bind the authoritative source revision."
        )
    deviation = float(maximum_deviation)
    if not np.isfinite(deviation) or deviation <= 0:
        raise ValueError("maximum_deviation must be finite and positive.")
    return deviation


def _approximation_binding(
    domain: PiecewiseLinearDomain | MappedReferenceDomain | None,
    block_regions: tuple[tuple[str, str], ...],
    block_names: tuple[str, ...],
    source_id: str,
    /,
    *,
    mapped: bool = False,
) -> tuple[tuple[str, str], ...]:
    if domain is not None and not isinstance(
        domain, (PiecewiseLinearDomain, MappedReferenceDomain)
    ):
        raise TypeError(
            "domain must be PiecewiseLinearDomain, MappedReferenceDomain, or None."
        )
    if isinstance(domain, MappedReferenceDomain) and not mapped:
        raise TypeError("This block source requires PiecewiseLinearDomain or None.")
    if domain is None:
        if block_regions:
            raise ValueError("block_regions requires a declared approximation domain.")
        return ()
    if domain.source_id != source_id:
        raise ValueError(
            "The approximation domain must bind the declared source identity."
        )
    if any(
        not isinstance(pair, tuple)
        or len(pair) != 2
        or not all(isinstance(value, str) for value in pair)
        for pair in block_regions
    ):
        raise TypeError(
            "block_regions must contain pairs of block and region identifiers."
        )
    mapping = tuple(sorted(block_regions))
    if len({name for name, _ in mapping}) != len(mapping):
        raise ValueError("Block region declarations must identify each block once.")
    if any(
        name not in block_names or region not in domain.region_ids
        for name, region in mapping
    ):
        raise ValueError("Block regions must name declared blocks and source regions.")
    if len(domain.region_ids) > 1 and {name for name, _ in mapping} != set(block_names):
        raise ValueError("A multimaterial approximation requires every block's region.")
    return mapping


@final
class NativeStructuredSource(StrictModule, NonTrainableState):
    """Explicit blocks/gluing and an independently authoritative boundary query."""

    blocks: tuple[TransfiniteBlock, ...]
    interfaces: tuple[BlockInterfaceControl, ...]
    fidelity_source: SourceBoundaryQuery
    maximum_deviation: float = eqx.field(static=True)
    domain: PiecewiseLinearDomain | None
    block_regions: tuple[tuple[str, str], ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

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
            raise TypeError("blocks must contain TransfiniteBlock values.")
        if any(
            not isinstance(interface, BlockInterfaceControl) for interface in interfaces
        ):
            raise TypeError("interfaces must contain BlockInterfaceControl values.")
        if len({block.name for block in blocks}) != len(blocks):
            raise ValueError("Structured block names must be unique.")
        source = _identifier(source_id, "source_id")
        revision = _identifier(source_revision, "source_revision")
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
    """A represented profile, sweep declaration and authoritative boundary query."""

    profile: CellMesh
    profile_geometry: CellGeometrySpec
    control: SweepControl
    fidelity_source: SourceBoundaryQuery
    maximum_deviation: float = eqx.field(static=True)
    domain: PiecewiseLinearDomain | MappedReferenceDomain | None
    root_correspondence: CellGeometryRestrictionSource | None
    block_regions: tuple[tuple[str, str], ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        profile: CellMesh,
        control: SweepControl,
        source_id: str,
        source_revision: str,
        /,
        *,
        fidelity_source: SourceBoundaryQuery,
        maximum_deviation: float,
        domain: PiecewiseLinearDomain | MappedReferenceDomain | None = None,
        block_regions: tuple[tuple[str, str], ...] = (),
        profile_geometry: CellGeometrySpec | None = None,
        root_correspondence: CellGeometryRestrictionSource | None = None,
    ) -> None:
        if not isinstance(profile, CellMesh) or not isinstance(control, SweepControl):
            raise TypeError("A sweep source requires CellMesh and SweepControl.")
        geometry = (
            CellGeometrySpec.affine(profile)
            if profile_geometry is None
            else profile_geometry
        )
        if not isinstance(geometry, CellGeometrySpec):
            raise TypeError("profile_geometry must be CellGeometrySpec or None.")
        geometry.resolve(profile)
        source = _identifier(source_id, "source_id")
        revision = _identifier(source_revision, "source_revision")
        deviation = _fidelity_binding(
            fidelity_source, source, revision, maximum_deviation
        )
        if (
            isinstance(domain, MappedReferenceDomain)
            and domain.source_revision != revision
        ):
            raise ValueError(
                "The mapped sweep domain must bind the exact authored source revision."
            )
        if isinstance(domain, MappedReferenceDomain):
            if not isinstance(root_correspondence, CellGeometryRestrictionSource):
                raise TypeError(
                    "Mapped sweeps require explicit canonical root_correspondence."
                )
            if (
                root_correspondence.source_geometry_id
                != cell_geometry_id(domain.source_geometry)
                or root_correspondence.source_topology_id
                != domain.reference_mesh.topology_id
            ):
                raise ValueError(
                    "Sweep root correspondence must bind the independently authored mapped domain."
                )
            expected_names = {f"sweep:{block.name}" for block in profile.blocks}
            if set(root_correspondence.block_names) != expected_names:
                raise ValueError(
                    "Sweep root correspondence must name every exact generated block."
                )
            parents = root_correspondence.block_parent_cell_ids
            if any(
                parents[f"sweep:{block.name}"].shape[0]
                != control.schedule.layer_count * block.cell_count
                for block in profile.blocks
            ):
                raise ValueError(
                    "Sweep root correspondence must identify every layer-major source column cell."
                )
        elif root_correspondence is not None:
            raise ValueError(
                "Sweep root correspondence requires an independently authored mapped domain."
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
class NativeMappedHexSource(StrictModule, NonTrainableState):
    """Exact physical root maps over an independently declared reference PLC."""

    reference: NativePlcSource
    domain: MappedReferenceDomain
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self, reference: NativePlcSource, domain: MappedReferenceDomain, /
    ) -> None:
        if not isinstance(reference, NativePlcSource):
            raise TypeError("reference must be NativePlcSource.")
        if not isinstance(domain, MappedReferenceDomain):
            raise TypeError("domain must be MappedReferenceDomain.")
        declared = declared_plc_domain(reference.complex, reference.source_id)
        if declared.domain_id != domain.reference_domain.domain_id:
            raise ValueError(
                "Mapped source maps must bind the independently declared reference PLC domain."
            )
        self.reference, self.domain = reference, domain
        self.source_id, self.source_revision = domain.source_id, domain.source_revision
        self.binding_id = canonical_fingerprint(
            {
                "kind": "native-mapped-hex-source",
                "reference": reference.binding_id,
                "mapped_domain": domain.domain_id,
            }
        )


type NativeMeshingSource = (
    NativeCurveSource
    | CompartmentMeshingSource
    | LabelFieldVolumeBinding
    | NativeImplicitSource
    | NativeLayerCoreSource
    | NativeMappedHexSource
    | NativePlanarSource
    | NativePeriodicSource
    | NativePlcSource
    | NativePolyhedralSource
    | NativeSurfaceSource
    | NativeStructuredSource
    | NativeSweepSource
    | NativeSurfaceEnvelopeSource
)


__all__ = [
    "NativeCurveSource",
    "NativeImplicitSource",
    "NativeLayerCoreSource",
    "NativeMappedHexSource",
    "NativeMeshingSource",
    "NativePlcSource",
    "NativePolyhedralSource",
    "NativePlanarSource",
    "NativeSurfaceSource",
    "NativeStructuredSource",
    "NativeSweepSource",
    "NativeSurfaceEnvelopeSource",
    "source_entity_id",
]
