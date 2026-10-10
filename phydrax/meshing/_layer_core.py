#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Immutable advancing-layer/PLC composition, not a second tetrahedralizer.

The PLC describes the *entire* remaining core, including holes and material
interfaces. Its vertices meet layers only through the supplied integer identity
map. Coordinates never establish identity. Original cap polygons are matched
with their orientation, and native recovery must preserve every fixed polygon.
The returned domain includes the layer exterior and every material interface;
publication still requires the independent combined-domain certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import SpatialCoordinateContract
from ..discretization import CellBlock, CellMesh, reference_cell_topology
from ..discretization._cell_geometry_validity import CellValidityPolicy
from ..geometry._mapped_reference_domain import MappedReferenceDomain
from ..geometry._mesh_certificates import PiecewiseLinearDomain
from ._association import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)
from ._boundary_layer import BoundaryLayerMesh
from ._canonical import _entity_vertex_keys, canonicalize_cell_mesh
from ._contracts import (
    MeshingDerivativeMode,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingProviderInfo,
    VolumeMeshingSpec,
)
from ._controls import BoundaryLayerControl
from ._layer_core_controls import compose_layer_controls
from ._layer_core_periodic import (
    bind_core_periodic,
    core_periodic_constraint_evidence,
    prepare_core_periodic,
)
from ._layer_core_regions import prepare_layer_region_identity
from ._layer_core_resources import (
    _row_entity_vertex_keys,
    LayerCoreSourceWork,
    reserve_layer_storage,
)
from ._lineage import EntityLineageKind, MeshLineage
from ._measurements import measure_phase, NativeMeshingPhaseRecorder
from ._organization import (
    MeshAttribute,
    MeshAttributeRole,
    MeshLabel,
    MeshPatch,
    MeshZone,
    MeshZoneRole,
)
from ._result import CellMeshingResult, MeshingComplianceReport
from ._scope import MeshingEntityKind, MeshingScope
from ._sizing import SizeControlStrength, UniformSizeControl
from ._trace import MeshingStageKind, MeshingStageReport, MeshingStageStatus
from ._volume_generation import (
    _entity,
    _native_live_preparation_is_active,
    generate_plc_volume,
    NativeVolumeSchedule,
    PiecewiseLinearComplex,
    VolumeConstruction,
)
from .providers._native_sources import NativeLayerCoreSource


if TYPE_CHECKING:
    from .providers._native_layer import PreparedLayerCore


def _cycle(row: tuple[int, ...], /) -> tuple[int, ...]:
    offset = row.index(min(row))
    return row[offset:] + row[:offset]


def _bitwise_equal_points(left: np.ndarray, right: np.ndarray, /) -> bool:
    return left.shape == right.shape and np.array_equal(
        left.view(np.uint64), right.view(np.uint64)
    )


def _triangles(row: tuple[int, ...], /) -> tuple[tuple[int, int, int], ...]:
    if len(row) == 3:
        return ((row[0], row[1], row[2]),)
    if len(row) != 4:
        raise ValueError(
            "Layer/core composition supports triangular or quadrilateral faces."
        )
    a, b, c, d = _cycle(row)
    return ((a, b, c), (a, c, d))


def _integer_array(value: ArrayLike, shape: tuple[int, ...], name: str, /) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must be an integer array of shape {shape}.")
    return array.astype(np.int64, copy=False)


def _failure(message: str, /) -> MeshingFailure:
    return MeshingFailure(
        MeshingFailureCategory.COMPLIANCE_FAILED,
        message,
        stage=MeshingStageKind.VOLUME_FILL.value,
    )


@dataclass(frozen=True, slots=True)
class _Face:
    vertices: tuple[int, ...]
    cell: int
    region: int


def _faces(
    mesh: CellMesh,
    regions: np.ndarray,
    /,
    *,
    work: LayerCoreSourceWork | None = None,
) -> dict[tuple[int, ...], list[_Face]]:
    """Reference-oriented faces in global-cell-ID order, with exact ownership."""
    identifiers = np.sort(np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64))
    table: dict[tuple[int, ...], list[_Face]] = {}
    for block in mesh.blocks:
        topology = reference_cell_topology(block.cell_kind)
        rows = np.asarray(block.vertices, dtype=np.int64)
        cell_ids = np.asarray(block.global_ids, dtype=np.int64)
        for row_index in range(block.cell_count):
            identifier = cell_ids[row_index]
            index = int(np.searchsorted(identifiers, identifier))
            for local in topology.entities[2]:
                if work is not None:
                    work.charge(1)
                vertices = tuple(int(rows[row_index, vertex]) for vertex in local)
                key = tuple(sorted(vertices))
                table.setdefault(key, []).append(
                    _Face(vertices, index, int(regions[index]))
                )
    for incidents in table.values():
        if len(incidents) > 2:
            raise _failure("Layer/core faces have more than two incident cells.")
        if len(incidents) == 2 and _cycle(incidents[0].vertices) != _cycle(
            tuple(reversed(incidents[1].vertices))
        ):
            raise _failure("Layer/core shared faces have inconsistent orientation.")
    return table


@dataclass(frozen=True, slots=True)
class LayerCoreConstruction:
    """Combined mesh and complete coverage inputs for native result publication."""

    mesh: CellMesh
    domain: PiecewiseLinearDomain | MappedReferenceDomain
    cell_regions: np.ndarray
    zones: tuple[MeshZone, ...]
    patches: tuple[MeshPatch, ...]
    labels: tuple[MeshLabel, ...]
    attributes: tuple[MeshAttribute, ...]
    associations: tuple[GeometryAssociation, ...]
    stages: tuple[MeshingStageReport, ...]
    core: VolumeConstruction
    construction_id: str
    work_units: int


def _prepare_identity(
    layers: BoundaryLayerMesh,
    complex_: PiecewiseLinearComplex,
    vertex_layer_ids: ArrayLike,
    cap_polygon_ids: ArrayLike,
    /,
    *,
    work: LayerCoreSourceWork | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if complex_.boundary != "fixed":
        raise ValueError("A layer core requires the immutable PLC boundary policy.")
    layer_points = np.asarray(layers.mesh.coordinates, dtype=np.float64)
    mapping = _integer_array(
        vertex_layer_ids, (complex_.vertices.shape[0],), "vertex_layer_ids"
    )
    if work is not None:
        work.charge(mapping.size)
    if np.any(mapping < -1) or np.any(mapping >= layer_points.shape[0]):
        raise ValueError("vertex_layer_ids index absent layer vertices.")
    shared = mapping >= 0
    if np.unique(mapping[shared]).size != np.count_nonzero(shared):
        raise ValueError("Distinct PLC vertices cannot identify one layer vertex.")
    if work is not None:
        work.charge(int(np.count_nonzero(shared)))
    if not _bitwise_equal_points(
        complex_.vertices[shared], layer_points[mapping[shared]]
    ):
        raise ValueError(
            "Authoritatively shared cap/rim coordinates must be bitwise identical."
        )
    if layers.cap is None:
        cap_rows = np.empty((0, 3), dtype=np.int64)
    else:
        if any(block.cell_kind != "triangle" for block in layers.cap.blocks):
            raise ValueError(
                "A tetrahedral core needs a triangular cap; enable transition pyramids."
            )
        cap_rows = np.asarray(layers.cap_vertices, dtype=np.int64)[
            np.concatenate(
                [
                    np.asarray(block.vertices, dtype=np.int64)
                    for block in layers.cap.blocks
                ]
            )
        ]
    cap_ids = _integer_array(cap_polygon_ids, (cap_rows.shape[0],), "cap_polygon_ids")
    polygon_count = complex_.polygon_facets.shape[0]
    if (
        np.any(cap_ids < 0)
        or np.any(cap_ids >= polygon_count)
        or np.unique(cap_ids).size != cap_ids.size
    ):
        raise ValueError("cap_polygon_ids must select each cap polygon exactly once.")
    lengths = np.diff(complex_.polygon_offsets)
    if np.any(lengths != 3):
        raise ValueError("Fixed layer-core PLC polygons must all be triangles.")
    polygons = complex_.polygon_vertices.reshape(-1, 3)
    for cap_row, polygon in zip(cap_rows, polygons[cap_ids], strict=True):
        if work is not None:
            work.charge(1)
        mapped = mapping[polygon]
        if np.any(mapped < 0) or _cycle(tuple(mapped.tolist())) != _cycle(
            tuple(cap_row.tolist())
        ):
            raise ValueError(
                "Each cap polygon must match exact layer vertex IDs and orientation."
            )
    cap_incidence = complex_.facet_regions[complex_.polygon_facets[cap_ids]]
    if np.any(cap_incidence[:, 0] < 0) or np.any(cap_incidence[:, 1] != -1):
        raise ValueError(
            "The oriented cap must have core on its positive side and void on its negative side."
        )
    combined_map = mapping.copy()
    combined_map[~shared] = layer_points.shape[0] + np.arange(np.count_nonzero(~shared))
    return (
        combined_map,
        cap_ids,
        polygons,
        np.concatenate((layer_points, complex_.vertices[~shared])),
    )


def _require_fixed(
    complex_: PiecewiseLinearComplex,
    core: VolumeConstruction,
    polygons: np.ndarray,
    /,
    *,
    work: LayerCoreSourceWork,
) -> None:
    work.charge(complex_.vertices.shape[0])
    points = np.asarray(core.mesh.coordinates, dtype=np.float64)
    if not _bitwise_equal_points(points[: complex_.vertices.shape[0]], complex_.vertices):
        raise _failure(
            "Native core recovery moved, renumbered, or dropped fixed vertices."
        )
    table = _faces(core.mesh, core.cell_regions, work=work)
    constrained = {tuple(sorted(row.tolist())) for row in polygons}
    for polygon, facet in zip(polygons, complex_.polygon_facets, strict=True):
        work.charge(1)
        incidents = table.get(tuple(sorted(polygon.tolist())), [])
        expected = complex_.facet_regions[facet]
        if len(incidents) != np.count_nonzero(expected >= 0):
            raise _failure("Native core recovery split or dropped a fixed facet.")
        for side, region in enumerate(expected):
            if region < 0:
                continue
            oriented = (
                tuple(polygon.tolist()) if side == 1 else tuple(polygon[::-1].tolist())
            )
            if not any(
                face.region == region and _cycle(face.vertices) == _cycle(oriented)
                for face in incidents
            ):
                raise _failure(
                    "Native core recovery changed a fixed oriented facet or its material incidence."
                )
    if any(
        len(incidents) == 1 and key not in constrained for key, incidents in table.items()
    ):
        raise _failure("Native core recovery introduced an undeclared external boundary.")


def _combined_domain(
    layers: BoundaryLayerMesh,
    complex_: PiecewiseLinearComplex,
    layer_regions: np.ndarray,
    mapping: np.ndarray,
    cap_ids: np.ndarray,
    polygons: np.ndarray,
    points: np.ndarray,
    source_id: str,
    region_ids: tuple[str, ...],
    core_region_map: np.ndarray,
    /,
) -> tuple[PiecewiseLinearDomain, dict[tuple[int, ...], list[_Face]]]:
    layer_faces = _faces(layers.mesh, layer_regions)
    cap_keys = {
        tuple(sorted(mapping[polygons[index]].tolist())): int(index) for index in cap_ids
    }
    facets: list[tuple[int, int, int]] = []
    incidence: list[tuple[int, int]] = []
    for key, incidents in layer_faces.items():
        if key in cap_keys:
            if len(incidents) != 1:
                raise _failure("A declared cap is not an exterior face of the layers.")
            polygon = mapping[polygons[cap_keys[key]]]
            if _cycle(incidents[0].vertices) != _cycle(tuple(polygon.tolist())):
                raise _failure(
                    "The declared cap orientation disagrees with the layer cell boundary."
                )
            continue
        first = incidents[0]
        second_region = -1 if len(incidents) == 1 else incidents[1].region
        if first.region != second_region:
            for triangle in _triangles(first.vertices):
                facets.append(triangle)
                incidence.append((first.region, second_region))
    if not set(cap_keys) <= set(layer_faces):
        raise _failure("The core includes a cap facet absent from the layer boundary.")
    for index, (polygon, facet) in enumerate(
        zip(polygons, complex_.polygon_facets, strict=True)
    ):
        positive, negative = (int(value) for value in complex_.facet_regions[facet])
        positive = int(core_region_map[positive]) if positive >= 0 else -1
        negative = int(core_region_map[negative]) if negative >= 0 else -1
        mapped = tuple(mapping[polygon].tolist())
        if index in cap_ids:
            negative = layer_faces[tuple(sorted(mapped))][0].region
        if positive != negative:
            facets.append((mapped[0], mapped[1], mapped[2]))
            # PLC incidence is (positive, negative); domain incidence follows normal.
            incidence.append((negative, positive))
    return PiecewiseLinearDomain(
        points,
        np.asarray(facets, dtype=np.int64),
        np.asarray(incidence, dtype=np.int64),
        region_ids,
        source_id=source_id,
    ), layer_faces


def _scope(mesh: CellMesh, dimension: int, identifiers: np.ndarray, /) -> MeshingScope:
    entities = mesh.entity_set(dimension)
    return MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        dimension,
        entities.entity_set_id,
        identifiers,
    )


def _organization(
    mesh: CellMesh,
    regions: np.ndarray,
    layer_count: int,
    layers: BoundaryLayerMesh,
    complex_: PiecewiseLinearComplex,
    mapping: np.ndarray,
    polygons: np.ndarray,
    cap_ids: np.ndarray,
    layer_faces: dict[tuple[int, ...], list[_Face]],
    region_ids: tuple[str, ...],
    /,
) -> tuple[
    tuple[MeshZone, ...],
    tuple[MeshPatch, ...],
    tuple[MeshLabel, ...],
    tuple[MeshAttribute, ...],
]:
    cells = np.sort(np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64))
    zones = tuple(
        MeshZone(name, MeshZoneRole.REGION, _scope(mesh, 3, cells[regions == index]))
        for index, name in enumerate(region_ids)
        if np.any(regions == index)
    )
    labels = (
        MeshLabel("boundary-layer", _scope(mesh, 3, cells[:layer_count])),
        MeshLabel("core", _scope(mesh, 3, cells[layer_count:])),
    )
    attributes = tuple(
        MeshAttribute(
            name,
            MeshAttributeRole.MARKER,
            _scope(mesh, 3, cells),
            np.concatenate(
                (
                    np.asarray(values, dtype=np.int64),
                    np.full(cells.size - layer_count, -1, dtype=np.int64),
                )
            ),
        )
        for name, values in (
            ("layer_index", layers.layer_index),
            ("layer_column", layers.column_index),
        )
    )
    # Canonical identity keys deliberately retain distinct periodic/lifted vertices.
    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    keys = dict(zip(_row_entity_vertex_keys(mesh, 2), face_ids.tolist(), strict=True))
    patches: list[MeshPatch] = []
    for facet in range(complex_.facet_count):
        selected = np.flatnonzero(complex_.polygon_facets == facet)
        ids = np.asarray(
            [
                keys[tuple(sorted(mapping[polygons[index]].tolist()))]
                for index in selected
            ],
            dtype=np.int64,
        )
        patches.append(MeshPatch(f"facet:{facet}", _scope(mesh, 2, ids), connected=False))
    cap_keys = {tuple(sorted(mapping[polygons[index]].tolist())) for index in cap_ids}
    if cap_keys:
        patches.append(
            MeshPatch(
                "layer-core-interface",
                _scope(
                    mesh,
                    2,
                    np.asarray([keys[key] for key in sorted(cap_keys)], dtype=np.int64),
                ),
                connected=False,
            )
        )
    wall_ids = np.asarray(
        [
            keys[key]
            for key, incidents in layer_faces.items()
            if len(incidents) == 1 and key not in cap_keys
        ],
        dtype=np.int64,
    )
    if wall_ids.size:
        patches.append(MeshPatch("wall", _scope(mesh, 2, wall_ids), connected=False))
    return zones, tuple(patches), labels, attributes


def _core_request(
    specification: VolumeMeshingSpec, layer_control: BoundaryLayerControl, /
) -> VolumeMeshingSpec:
    """Combine the physical core cap with sizing without rewriting its evidence."""
    bounds = tuple(
        control.core_maximum_size
        for control in (*specification.layer_controls, layer_control)
        if control.core_maximum_size is not None
    )
    if not bounds:
        return specification
    original = specification.size_controls[0]
    if len(specification.size_controls) != 1 or not isinstance(
        original, UniformSizeControl
    ):
        raise ValueError(
            "Native layer/core generation requires one uniform core size request."
        )
    cap = min(bounds)
    if original.strength is SizeControlStrength.HARD and (
        (original.minimum_size is not None and original.minimum_size > cap)
        or original.target_size
        > cap
        + specification.size_compliance.absolute_tolerance
        + specification.size_compliance.relative_tolerance * original.target_size
    ):
        raise MeshingFailure(
            MeshingFailureCategory.CONTROL_CONFLICT,
            "The hard core size request conflicts with the layer control's core maximum size.",
            stage=MeshingStageKind.CONTROL_RESOLUTION.value,
            requested=(("core_maximum_size", cap), ("target_size", original.target_size)),
        )
    target = min(original.target_size, cap)
    minimum = original.minimum_size
    if (
        original.strength is SizeControlStrength.SOFT
        and minimum is not None
        and minimum > target
    ):
        minimum = None
    size = UniformSizeControl(
        original.scope,
        target,
        minimum_size=minimum,
        maximum_size=min(cap, original.maximum_size)
        if original.maximum_size is not None
        else cap,
        maximum_growth_rate=original.maximum_growth_rate,
        strength=original.strength,
        priority=original.priority,
    )
    return VolumeMeshingSpec(
        specification.target,
        specification.boundary_scope,
        specification.fill_strategy,
        size_controls=(size,),
        protected_features=specification.protected_features,
        region_controls=specification.region_controls,
        patch_controls=specification.patch_controls,
        region_seeds=specification.region_seeds,
        hole_seeds=specification.hole_seeds,
        layer_controls=specification.layer_controls,
        periodic_constraints=specification.periodic_constraints,
        size_combination=specification.size_combination,
        size_compliance=specification.size_compliance,
        limits=specification.limits,
        deterministic=specification.deterministic,
    )


def _rebind_core_associations(
    core: VolumeConstruction, target: CellMesh, vertex_map: np.ndarray, /
) -> tuple[GeometryAssociation, ...]:
    """Rebind immutable PLC vertex/edge/facet identities and canonical orientation."""
    output: list[GeometryAssociation] = []
    source_vertex_rows = {
        int(identifier): row
        for row, identifier in enumerate(np.asarray(core.mesh.vertex_global_ids))
    }
    target_vertex_ids = np.asarray(target.vertex_global_ids)
    for association in core.associations:
        if association.target_entity_set_id == core.mesh.entity_set(3).entity_set_id:
            continue
        dimensions = tuple(
            dimension
            for dimension in (0, 1, 2)
            if association.target_entity_set_id
            == core.mesh.entity_set(dimension).entity_set_id
        )
        if len(dimensions) != 1:
            raise ValueError(
                "Native core association must bind canonical cells, vertices, faces or edges."
            )
        dimension = dimensions[0]
        source_ids = np.asarray(
            core.mesh.entity_set(dimension).entity_ids, dtype=np.int64
        )
        target_ids = np.asarray(target.entity_set(dimension).entity_ids, dtype=np.int64)
        source_keys = (
            tuple((int(identifier),) for identifier in source_ids)
            if dimension == 0
            else _entity_vertex_keys(core.mesh, dimension)
        )
        target_keys = (
            tuple((int(identifier),) for identifier in target_ids)
            if dimension == 0
            else _entity_vertex_keys(target, dimension)
        )
        source_rows = dict(zip(source_ids.tolist(), source_keys, strict=True))
        target_rows = dict(zip(target_keys, target_ids.tolist(), strict=True))
        remapped: list[int] = []
        signs: list[int] = []
        for identifier in np.asarray(association.target_global_ids, dtype=np.int64):
            row = tuple(
                int(target_vertex_ids[vertex_map[source_vertex_rows[identifier_]]])
                for identifier_ in source_rows[int(identifier)]
            )
            remapped.append(target_rows[tuple(sorted(row))])
            inversions = sum(
                first > second
                for index, first in enumerate(row)
                for second in row[index + 1 :]
            )
            signs.append(-1 if inversions % 2 else 1)
        output.append(
            GeometryAssociation(
                association.association_kind,
                association.source_id,
                association.source_revision,
                target.entity_set(dimension).entity_set_id,
                np.asarray(remapped, dtype=np.int64),
                association.source_entity_ids,
                association.residuals,
                resolved=association.resolved,
                ambiguous=association.ambiguous,
                exact=association.exact,
                source_dimensions=association.source_dimensions,
                source_indices=association.source_indices,
                source_occurrence_paths=association.source_occurrence_paths,
                source_entity_roles=association.source_entity_roles,
                parameters=association.parameters,
                orientations=np.asarray(association.orientations, dtype=np.int8)
                * np.asarray(signs, dtype=np.int8),
                parent_dimensions=association.parent_dimensions,
                parent_ids=association.parent_ids,
                parent_association_id=association.parent_association_id,
                provenance=association.provenance,
            )
        )
    return tuple(output)


def generate_layer_core(
    layers: BoundaryLayerMesh,
    complex_: PiecewiseLinearComplex,
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    /,
    *,
    validity_policy: CellValidityPolicy,
    vertex_layer_ids: ArrayLike,
    cap_polygon_ids: ArrayLike,
    layer_regions: ArrayLike,
    source_id: str,
    source_revision: str,
    input_id: str,
    source_binding: NativeLayerCoreSource | None = None,
    operation_started: float | None = None,
    source_work_units: int = 0,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> LayerCoreConstruction:
    """Compose immutable layers with native filling of the complete remaining PLC.

    ``vertex_layer_ids[i]`` identifies PLC vertex i with a layer vertex, or -1
    declares a new vertex. ``cap_polygon_ids`` follows cap block order.
    ``layer_regions`` follows sorted layer global cell IDs and indexes the
    source binding's composite region namespace. The core PLC retains only
    bounded core-local regions and its exact map into that namespace.
    ``validity_policy`` is the publishing audit's cell validity policy, whose
    determinant floor the native core improvement enforces.
    """
    if not isinstance(layers, BoundaryLayerMesh) or not isinstance(
        complex_, PiecewiseLinearComplex
    ):
        raise TypeError(
            "layers and complex_ must be BoundaryLayerMesh and PiecewiseLinearComplex."
        )
    count = sum(block.cell_count for block in layers.mesh.blocks)
    regions = _integer_array(layer_regions, (count,), "layer_regions")
    work = LayerCoreSourceWork(specification.limits.maximum_work_units, source_work_units)
    work.charge(0)
    if source_binding is not None and (
        source_binding.layers.result_id != layers.result_id
        or source_binding.complex.complex_id != complex_.complex_id
        or (source_binding.source_id, source_binding.source_revision)
        != (source_id, source_revision)
        or not np.array_equal(
            source_binding.vertex_layer_ids, np.asarray(vertex_layer_ids)
        )
        or not np.array_equal(source_binding.cap_polygon_ids, np.asarray(cap_polygon_ids))
        or not np.array_equal(source_binding.layer_regions, regions)
    ):
        raise ValueError(
            "The core construction must bind the exact authored source, layer realization, PLC, and integer ancestry."
        )
    region_ids, core_regions = prepare_layer_region_identity(
        complex_,
        regions,
        None if source_binding is None else source_binding.region_ids,
        None if source_binding is None else source_binding.core_region_map,
    )
    core_specification = reserve_layer_storage(
        layers,
        complex_,
        _core_request(specification, layers.control),
        _integer_array(
            vertex_layer_ids, (complex_.vertices.shape[0],), "vertex_layer_ids"
        ),
    )
    mapping, cap_ids, polygons, initial_points = _prepare_identity(
        layers,
        complex_,
        vertex_layer_ids,
        cap_polygon_ids,
        work=work,
    )
    if layers.mesh.periodic_topology is not None and source_binding is None:
        raise ValueError(
            "Periodic core fill requires its source-authored seam anatomy before construction."
        )
    ancestry = (
        None
        if source_binding is None
        else prepare_core_periodic(
            source_binding,
            mapping,
            polygons,
            initial_points,
            work=work,
        )
    )
    core = generate_plc_volume(
        complex_,
        core_specification,
        schedule,
        validity_policy=validity_policy,
        source_id=complex_.complex_id,
        source_revision=complex_.complex_id,
        input_id=input_id,
        record_phase=record_phase,
        operation_started=operation_started,
        source_work_units=work.work_units,
    )
    work.work_units = core.work_units
    _require_fixed(complex_, core, polygons, work=work)
    with measure_phase(record_phase, "topology_construction"):
        core_points = np.asarray(core.mesh.coordinates, dtype=np.float64)
        work.charge(
            core_points.shape[0]
            + sum(block.cell_count for block in layers.mesh.blocks)
            + core.cell_regions.size
        )
        points = np.concatenate(
            (initial_points, core_points[complex_.vertices.shape[0] :])
        )
        core_map = np.concatenate(
            (
                mapping,
                initial_points.shape[0]
                + np.arange(core_points.shape[0] - complex_.vertices.shape[0]),
            )
        )
        blocks = tuple(
            CellBlock(
                f"layer:{block.name}",
                block.cell_kind,
                block.vertices,
                global_ids=block.global_ids,
            )
            for block in layers.mesh.blocks
        )
        layer_ids = np.sort(
            np.asarray(layers.mesh.entity_set(3).entity_ids, dtype=np.int64)
        )
        core_start = int(layer_ids[-1]) + 1
        if core_start + core.cell_regions.size - 1 > np.iinfo(np.int64).max:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "The immutable layer cell registry has no room for the core's exact identities.",
                stage=MeshingStageKind.CANONICALIZATION.value,
            )
        blocks += (
            CellBlock(
                "core",
                "tetrahedron",
                core_map[np.asarray(core.mesh.blocks[0].vertices, dtype=np.int64)],
                global_ids=core_start + np.arange(core.cell_regions.size),
            ),
        )
        layer_vertex_ids = np.asarray(layers.mesh.vertex_global_ids, dtype=np.int64)
        extra = points.shape[0] - layer_vertex_ids.size
        next_vertex = int(np.max(layer_vertex_ids)) + 1
        if next_vertex + extra - 1 > np.iinfo(np.int64).max:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "The immutable layer vertex registry has no room for the core's exact identities.",
                stage=MeshingStageKind.CANONICALIZATION.value,
            )
        new_ids = (
            next_vertex + np.arange(extra, dtype=np.int64)
            if extra
            else np.empty(0, dtype=np.int64)
        )
        vertex_ids = np.concatenate((layer_vertex_ids, new_ids))
        plain = CellMesh(
            points, blocks, vertex_global_ids=vertex_ids, numeric_version=source_revision
        )
        if source_binding is not None:
            plain = bind_core_periodic(source_binding, plain, ancestry)
        mesh = canonicalize_cell_mesh(plain)
    combined_regions = np.concatenate((regions, core_regions[core.cell_regions]))
    domain, layer_faces = _combined_domain(
        layers,
        complex_,
        regions,
        mapping,
        cap_ids,
        polygons,
        initial_points,
        source_id,
        region_ids,
        core_regions,
    )
    combined_faces = _faces(mesh, combined_regions, work=work)
    for index in cap_ids:
        work.charge(1)
        key = tuple(sorted(mapping[polygons[index]].tolist()))
        if len(combined_faces[key]) != 2:
            raise _failure("Layer and core do not share the exact immutable interface.")
    with measure_phase(record_phase, "organization"):
        zones, patches, labels, attributes = _organization(
            mesh,
            combined_regions,
            count,
            layers,
            complex_,
            mapping,
            polygons,
            cap_ids,
            layer_faces,
            region_ids,
        )
    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    face_keys = dict(
        zip(_row_entity_vertex_keys(mesh, 2), face_ids.tolist(), strict=True)
    )
    interface_ids = np.asarray(
        [
            face_keys[key]
            for key, incidents in combined_faces.items()
            if len(incidents) == 2 and incidents[0].region != incidents[1].region
        ],
        dtype=np.int64,
    )
    if interface_ids.size:
        labels += (MeshLabel("interface", _scope(mesh, 2, interface_ids)),)
    for label in core.labels:
        if label.name == "interface":
            continue
        dimension = label.scope.entity_dimension
        old_ids = np.asarray(core.mesh.entity_set(dimension).entity_ids, dtype=np.int64)
        old_keys = _row_entity_vertex_keys(core.mesh, dimension)
        new_ids = dict(
            zip(
                _row_entity_vertex_keys(mesh, dimension),
                np.asarray(
                    mesh.entity_set(dimension).entity_ids, dtype=np.int64
                ).tolist(),
                strict=True,
            )
        )
        selected = np.isin(old_ids, np.asarray(label.scope.entity_ids))
        remapped = np.asarray(
            [
                new_ids[tuple(sorted(int(core_map[vertex]) for vertex in key))]
                for key, keep in zip(old_keys, selected, strict=True)
                if keep
            ],
            dtype=np.int64,
        )
        labels += (MeshLabel(label.name, _scope(mesh, dimension, remapped)),)
    cells = np.sort(np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64))
    with measure_phase(record_phase, "geometry_association"):
        associations = (
            GeometryAssociation(
                GeometryAssociationKind.PIECEWISE_LINEAR,
                source_id,
                source_revision,
                mesh.entity_set(3).entity_set_id,
                cells,
                tuple(
                    f"{source_revision}:region:{int(region)}"
                    for region in combined_regions
                ),
                np.zeros(cells.size, dtype=np.float64),
                exact=True,
                source_dimensions=np.full(cells.size, 3, dtype=np.int8),
                source_indices=combined_regions,
                source_entity_roles=(GeometrySourceEntityRole.REGION,) * cells.size,
            ),
        )
        associations += _rebind_core_associations(core, mesh, core_map)
        from ._layer_core_association import _layer_source_associations

        cap_association = None
        if layers.cap is not None and cap_ids.size:
            cap_face_ids = np.concatenate(
                [
                    np.asarray(block.global_ids, dtype=np.int64)
                    for block in layers.cap.blocks
                ]
            )
            rows = mapping[polygons[cap_ids]]
            target_ids = np.asarray(
                [face_keys[tuple(sorted(row.tolist()))] for row in rows], dtype=np.int64
            )
            signs = np.asarray(
                [
                    -1
                    if sum(
                        first > second
                        for index, first in enumerate(row)
                        for second in row[index + 1 :]
                    )
                    % 2
                    else 1
                    for row in rows
                ],
                dtype=np.int8,
            )
            cap_association = GeometryAssociation(
                GeometryAssociationKind.PIECEWISE_LINEAR,
                layers.result_id,
                layers.result_id,
                mesh.entity_set(2).entity_set_id,
                target_ids,
                tuple(
                    _entity(
                        layers.result_id,
                        GeometrySourceEntityRole.FACET.value,
                        int(identifier),
                    )
                    for identifier in cap_face_ids
                ),
                np.zeros(cap_ids.size, dtype=np.float64),
                exact=True,
                source_dimensions=np.full(cap_ids.size, 2, dtype=np.int8),
                source_indices=cap_face_ids,
                orientations=signs,
                source_entity_roles=(GeometrySourceEntityRole.FACET,) * cap_ids.size,
            )
        associations += _layer_source_associations(
            layers,
            mesh,
            regions,
            region_ids,
            cap_association=cap_association,
            work=work,
        )
    construction_id = canonical_fingerprint(
        {
            "kind": "native-layer-core",
            "layers": layers.result_id,
            "complex": complex_.complex_id,
            "mesh": mesh.mesh_id,
            "domain": domain.domain_id,
            "vertex_layer_ids": array_tree_fingerprint(
                np.asarray(vertex_layer_ids, dtype=np.int64)
            ),
            "cap_polygon_ids": array_tree_fingerprint(cap_ids),
            "layer_regions": array_tree_fingerprint(regions),
            "region_ids": region_ids,
            "core_region_map": array_tree_fingerprint(core_regions),
        }
    )
    stages = (
        MeshingStageReport(
            MeshingStageKind.LAYER_GENERATION,
            MeshingStageStatus.PASSED,
            input_ids=(layers.control_id,),
            output_ids=(layers.result_id,),
            created_count=count,
        ),
        *core.stages,
        MeshingStageReport(
            MeshingStageKind.CANONICALIZATION,
            MeshingStageStatus.PASSED,
            input_ids=(layers.result_id, core.mesh.mesh_id),
            output_ids=(construction_id,),
        ),
    )
    block_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    block_regions = combined_regions[np.searchsorted(cells, block_ids)]
    return LayerCoreConstruction(
        mesh,
        domain,
        block_regions,
        zones,
        patches,
        labels,
        attributes,
        associations,
        stages,
        core,
        construction_id,
        work.work_units,
    )


def execute_layer_core_route(
    layers: BoundaryLayerMesh,
    complex_: PiecewiseLinearComplex,
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    vertex_layer_ids: ArrayLike,
    cap_polygon_ids: ArrayLike,
    layer_regions: ArrayLike,
    source_id: str,
    source_revision: str,
    generation_specification: VolumeMeshingSpec,
    original_source: NativeLayerCoreSource,
    source_preparation_work_units: int,
    prepared_owner: PreparedLayerCore,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> CellMeshingResult:
    """Generate and independently certify the entire native layer/core domain."""
    if not _native_live_preparation_is_active(prepared_owner):
        raise ValueError(
            "Layer/core publication requires its original full-route native budget; "
            "standalone prepared execution must use execute_layer_route with its actual preparation receipt."
        )
    if (prepared_owner.source_binding_id, prepared_owner.specification_id) != (
        original_source.binding_id,
        specification.specification_id,
    ):
        raise ValueError(
            "The live layer preparation belongs to another original source or specification."
        )
    from time import monotonic

    from ._audit import CellMeshAuditDisposition, CellMeshAuditPolicy
    from ._layer_source_certification import layer_source_certification
    from .providers._native_publication import (
        check_deadline,
        publish_native_result,
    )
    from .providers._native_volume import _volume_compliance

    started = monotonic()
    work = LayerCoreSourceWork(
        specification.limits.maximum_work_units, source_preparation_work_units
    )
    periodic_requested, periodic_achieved = core_periodic_constraint_evidence(
        original_source, specification, work=work
    )
    # The native core determinant-floor repair and the publication audit share one policy.
    audit_policy = CellMeshAuditPolicy(
        watertight_boundary=CellMeshAuditDisposition.REJECT
    )
    construction = generate_layer_core(
        layers,
        complex_,
        generation_specification,
        schedule,
        validity_policy=audit_policy.validity_policy,
        vertex_layer_ids=vertex_layer_ids,
        cap_polygon_ids=cap_polygon_ids,
        layer_regions=layer_regions,
        source_id=source_id,
        source_revision=source_revision,
        input_id=plan_id,
        record_phase=record_phase,
        source_binding=original_source,
        operation_started=started,
        source_work_units=work.work_units,
    )
    mesh = construction.mesh
    limits = specification.limits
    cells = sum(block.cell_count for block in mesh.blocks)
    entries = sum(block.vertices.size for block in mesh.blocks)
    checks = (
        ("vertices", mesh.coordinates.shape[0], limits.maximum_vertices),
        ("edges", mesh.entity_set(1).count, limits.maximum_edges),
        ("faces", mesh.entity_set(2).count, limits.maximum_faces),
        ("cells", cells, limits.maximum_cells),
        ("connectivity_entries", entries, limits.maximum_connectivity_entries),
        (
            "data_bytes",
            mesh.coordinates.size * 8 + entries * 4,
            limits.maximum_data_bytes,
        ),
    )
    if any(actual > bound for _, actual, bound in checks):
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "The combined native layer/core mesh exceeds its declared resource limits.",
            stage=MeshingStageKind.VOLUME_FILL.value,
            requested=tuple((f"maximum_{name}", bound) for name, _, bound in checks),
            achieved=tuple((name, actual) for name, actual, _ in checks),
        )
    check_deadline(started, limits, MeshingStageKind.VOLUME_FILL)
    core = construction.core
    with measure_phase(record_phase, "compliance"):
        core_compliance = _volume_compliance(
            generation_specification,
            schedule,
            core,
            np.asarray(core.mesh.coordinates, dtype=np.float64),
            np.asarray(core.mesh.blocks[0].vertices, dtype=np.int64),
        )
    bounds = tuple(
        control.core_maximum_size
        for control in (*specification.layer_controls, layers.control)
        if control.core_maximum_size is not None
    )
    core_size_requested: tuple[tuple[str, float], ...] = ()
    core_size_achieved: tuple[tuple[str, float], ...] = ()
    core_size_issues: tuple[str, ...] = ()
    if bounds:
        cap = min(bounds)
        core_points = np.asarray(core.mesh.coordinates, dtype=np.float64)
        core_edges = _row_entity_vertex_keys(core.mesh, 1)
        lengths = np.linalg.norm(
            core_points[np.asarray(core_edges, dtype=np.int64)[:, 1]]
            - core_points[np.asarray(core_edges, dtype=np.int64)[:, 0]],
            axis=1,
        )
        maximum = float(np.max(lengths))
        core_size_requested = (("core_maximum_size", cap),)
        core_size_achieved = (("core_maximum_edge", maximum),)
        if (
            maximum
            > cap
            + specification.size_compliance.absolute_tolerance
            + specification.size_compliance.relative_tolerance * cap
        ):
            core_size_issues = ("core_maximum_size",)
    evidence = layers.evidence
    work.work_units = construction.work_units
    zones, patches = compose_layer_controls(
        original_source, specification, construction, work=work
    )
    certification = layer_source_certification(
        original_source, specification, construction, work=work
    )
    compliance = MeshingComplianceReport(
        specification.specification_id,
        issues=(*core_compliance.issues, *core_size_issues),
        requested=(
            *core_compliance.requested,
            *core_size_requested,
            *periodic_requested,
            *(
                (f"layer:{index}:thickness", value)
                for index, value in enumerate(evidence.requested_thicknesses)
            ),
        ),
        achieved=(
            *core_compliance.achieved,
            *core_size_achieved,
            *periodic_achieved,
            ("fixed_cap_vertices_bitwise", 1.0),
            ("fixed_cap_oriented_facets", 1.0),
            ("layer_core_prepublication_work_units", work.work_units),
            *(
                (f"layer:{index}:active", float(value))
                for index, value in enumerate(np.asarray(evidence.layer_active))
            ),
            *(
                (f"layer:{index}:mean_thickness", float(value))
                for index, value in enumerate(np.asarray(evidence.achieved_thicknesses))
                if np.isfinite(value)
            ),
            *(
                (f"layer:{index}:active_column_count", int(value))
                for index, value in enumerate(np.asarray(evidence.active_column_counts))
            ),
            *(
                (f"layer:{index}:minimum_thickness", float(value))
                for index, value in enumerate(np.asarray(evidence.minimum_thicknesses))
                if np.isfinite(value)
            ),
            *(
                (f"layer:{index}:maximum_thickness", float(value))
                for index, value in enumerate(np.asarray(evidence.maximum_thicknesses))
                if np.isfinite(value)
            ),
            *(
                (f"layer:{index}:growth_rate", float(value))
                for index, value in enumerate(np.asarray(evidence.achieved_growth_rates))
                if np.isfinite(value)
            ),
            ("layer:terminated_vertices", evidence.terminated_vertex_count),
            ("layer:reduced_vertices", evidence.reduced_vertex_count),
            ("layer:merged_vertices", evidence.merged_vertex_count),
        ),
    )
    result = publish_native_result(
        mesh,
        coordinate_contract,
        compliance,
        construction.stages,
        provider,
        {
            "kind": "native-layer-core",
            "plan": plan_id,
            "source": source_id,
            "specification": specification.specification_id,
            "source_revision": source_revision,
            "source_binding": original_source.binding_id,
            "layers": layers.result_id,
            "construction": construction.construction_id,
        },
        certification,
        audit_policy=audit_policy,
        derivative_mode=MeshingDerivativeMode.NONDIFFERENTIABLE,
        enforced_limits=(
            "vertices",
            "edges",
            "faces",
            "cells",
            "connectivity_entries",
            "data_bytes",
            "wall_time",
        ),
        unenforced_limits=("native_workspace",),
        zones=zones,
        patches=patches,
        labels=construction.labels,
        associations=construction.associations,
        attributes=construction.attributes,
        record_phase=record_phase,
    )
    check_deadline(started, limits, MeshingStageKind.CERTIFICATION)
    return result


def _reserved_layer_attributes(source: CellMeshingResult, /) -> tuple[MeshAttribute, ...]:
    """Select the reserved physical-layer pair without admitting unrelated data."""
    return tuple(
        attribute
        for attribute in source.attributes
        if attribute.name in ("layer_index", "layer_column")
    )


def validate_layer_index_attributes(source: CellMeshingResult, /) -> None:
    """Validate the reserved layer pair; unrelated scientific attributes stay separate."""
    reserved = _reserved_layer_attributes(source)
    if not reserved:
        return
    if len(reserved) != 2 or {attribute.name for attribute in reserved} != {
        "layer_index",
        "layer_column",
    }:
        raise ValueError(
            "Native layer adaptation requires canonical layer_index and layer_column attributes."
        )
    if source.mesh.topological_dimension != 3:
        raise ValueError("Physical layer markers require volumetric cell identities.")
    cells = source.mesh.entity_set(3)
    for attribute in reserved:
        scope = attribute.scope
        values = np.asarray(attribute.values)
        if (
            attribute.role is not MeshAttributeRole.MARKER
            or attribute.unit is not None
            or attribute.component_shape
            or scope.entity_kind is not MeshingEntityKind.MESH
            or scope.entity_dimension != 3
            or scope.source_id != source.mesh.mesh_id
            or scope.source_revision != source.mesh.numeric_version
            or scope.entity_set_id != cells.entity_set_id
            or not np.array_equal(
                np.asarray(scope.entity_ids), np.sort(np.asarray(cells.entity_ids))
            )
            or values.ndim != 1
            or not np.issubdtype(values.dtype, np.integer)
            or np.any(values < -1)
        ):
            raise ValueError(
                "Layer markers must bind every source cell with exact integer ancestry."
            )
    attributes = {attribute.name: attribute for attribute in reserved}
    if not np.array_equal(
        np.asarray(attributes["layer_index"].values) < 0,
        np.asarray(attributes["layer_column"].values) < 0,
    ):
        raise ValueError("Physical layer interval and column support must coincide.")


def layer_interval_classes(source: CellMeshingResult, /) -> np.ndarray:
    """Physical-interval discriminators in mesh block order for template closure."""
    validate_layer_index_attributes(source)
    ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in source.mesh.blocks]
    )
    reserved = _reserved_layer_attributes(source)
    if not reserved:
        return np.zeros(ids.size, dtype=np.int64)
    attributes = {attribute.name: attribute for attribute in reserved}
    pairs = np.stack(
        [
            np.asarray(attributes[name].values, dtype=np.int64)[
                np.searchsorted(
                    np.asarray(attributes[name].scope.entity_ids, dtype=np.int64), ids
                )
            ]
            for name in ("layer_index", "layer_column")
        ],
        axis=1,
    )
    _, classes = np.unique(pairs, axis=0, return_inverse=True)
    return classes.astype(np.int64)


def remap_layer_index_attribute(
    source: CellMeshingResult,
    lineage: MeshLineage,
    target: CellMesh,
    /,
) -> tuple[MeshAttribute, ...]:
    """Inherit physical intervals only through complete, unambiguous cell ancestry."""
    validate_layer_index_attributes(source)
    reserved = _reserved_layer_attributes(source)
    if not reserved:
        return ()
    if (
        lineage.source_topology_id != source.mesh.topology_id
        or lineage.target_topology_id != target.topology_id
    ):
        raise ValueError(
            "Layer interval lineage must bind the actual source and target topologies."
        )
    relation = lineage.entity_lineage(3)
    if (
        relation.source_entity_set_id != source.mesh.entity_set(3).entity_set_id
        or relation.target_entity_set_id != target.entity_set(3).entity_set_id
    ):
        raise ValueError(
            "Layer interval lineage must bind the actual source and target cell sets."
        )
    attributes = {attribute.name: attribute for attribute in reserved}
    old_ids = np.asarray(attributes["layer_index"].scope.entity_ids, dtype=np.int64)
    old_values = np.stack(
        [
            np.asarray(attributes[name].values, dtype=np.int64)
            for name in ("layer_index", "layer_column")
        ],
        axis=1,
    )
    targets = np.sort(np.asarray(target.entity_set(3).entity_ids, dtype=np.int64))
    inherited: dict[int, tuple[int, int]] = {}
    for parent, child, kind in zip(
        np.asarray(relation.source_global_ids, dtype=np.int64),
        np.asarray(relation.target_global_ids, dtype=np.int64),
        np.asarray(relation.relation_kinds, dtype=np.int32),
        strict=True,
    ):
        index = int(np.searchsorted(old_ids, parent))
        if (
            index == old_ids.size
            or old_ids[index] != parent
            or not np.any(targets == child)
        ):
            raise ValueError(
                "Layer lineage names cells outside its bound source or target."
            )
        value = (int(old_values[index, 0]), int(old_values[index, 1]))
        if kind in (EntityLineageKind.UNKNOWN, EntityLineageKind.GENERATED_ON_GEOMETRY):
            raise ValueError(
                "Unknown or generated cell ancestry cannot preserve a physical layer interval."
            )
        if kind == EntityLineageKind.SWAPPED_FROM and value[0] >= 0:
            raise ValueError(
                "A layer swap requires an explicitly reconstructed physical schedule."
            )
        identifier = int(child)
        if identifier in inherited and inherited[identifier] != value:
            raise ValueError(
                "Coarsening or merging different physical layer intervals or columns is forbidden."
            )
        inherited[identifier] = value
    if set(inherited) != set(targets.tolist()):
        raise ValueError(
            "Every target layer/core cell requires complete physical-interval lineage."
        )
    values = np.asarray(
        [inherited[int(identifier)] for identifier in targets], dtype=np.int64
    )
    return tuple(
        MeshAttribute(
            name, MeshAttributeRole.MARKER, _scope(target, 3, targets), values[:, index]
        )
        for index, name in enumerate(("layer_index", "layer_column"))
    )


__all__ = [
    "LayerCoreConstruction",
    "generate_layer_core",
    "execute_layer_core_route",
    "validate_layer_index_attributes",
    "layer_interval_classes",
    "remap_layer_index_attribute",
]
