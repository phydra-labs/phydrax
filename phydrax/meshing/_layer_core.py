#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Immutable advancing-layer/PLC composition, not a second tetrahedralizer.

The PLC describes the *entire* remaining core, including holes and material
interfaces. Its vertices meet layers only through the supplied integer identity
map. Coordinates never establish identity. Source cap polygons are matched
with their orientation, and native recovery must preserve every fixed polygon.
The returned domain includes the layer exterior and every material interface;
publication still requires the independent combined-domain certificate.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import final

import numpy as np
from jax.typing import ArrayLike

import phydrax.discretization._cell_geometry_validity as _dependency_phydrax_discretization__cell_geometry_validity
import phydrax.discretization._coordinate_budget as _dependency_phydrax_discretization__coordinate_budget
import phydrax.geometry._domain_coverage as domain_coverage_owner
import phydrax.meshing._association_state as association_state_owner
import phydrax.meshing._boundary_layer as _dependency_phydrax_meshing__boundary_layer
import phydrax.meshing._canonical as _dependency_phydrax_meshing__canonical
import phydrax.meshing._contracts as _dependency_phydrax_meshing__contracts
import phydrax.meshing._controls as _dependency_phydrax_meshing__controls
import phydrax.meshing._failures as _leaf_phydrax_meshing__failures
import phydrax.meshing._host_banks as _dependency_phydrax_meshing__host_banks
import phydrax.meshing._layer_core_association as _owner_phydrax_meshing__layer_core_association
import phydrax.meshing._layer_core_controls as _dependency_phydrax_meshing__layer_core_controls
import phydrax.meshing._layer_core_periodic as _dependency_phydrax_meshing__layer_core_periodic
import phydrax.meshing._layer_core_publication as layer_publication
import phydrax.meshing._layer_core_regions as _dependency_phydrax_meshing__layer_core_regions
import phydrax.meshing._layer_core_resources as _dependency_phydrax_meshing__layer_core_resources
import phydrax.meshing._layer_core_topology as layer_topology
import phydrax.meshing._layer_source_certification as _owner_phydrax_meshing__layer_source_certification
import phydrax.meshing._lineage_records as _dependency_phydrax_meshing__lineage
import phydrax.meshing._organization_records as _dependency_phydrax_meshing__organization
import phydrax.meshing._plc_source as plc_source_owner
import phydrax.meshing._result as _dependency_phydrax_meshing__result
import phydrax.meshing._sizing as _dependency_phydrax_meshing__sizing
import phydrax.meshing._trace as _dependency_phydrax_meshing__trace
import phydrax.meshing._volume_generation as _dependency_phydrax_meshing__volume_generation
import phydrax.meshing.providers._native_publication as _owner_phydrax_meshing_providers__native_publication
import phydrax.meshing.providers._native_route_checks as route_checks
import phydrax.meshing.providers._native_sources as _dependency_phydrax_meshing_providers__native_sources
import phydrax.meshing.providers._native_volume as _owner_phydrax_meshing_providers__native_volume

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import NativeExecutionScope, NativeHostStorageWorkspace
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellBlock, CellMesh, reference_cell_topology
from ..typing import checked
from ._compliance_failure import compliance_failure
from ._measurements import measure_phase, NativeMeshingPhaseRecorder
from ._native_preparation import native_preparation_ledger
from ._scope import mesh_entity_scope, MeshingEntityKind


def _require_fixed(
    complex_: plc_source_owner.PiecewiseLinearComplex,
    core: _dependency_phydrax_meshing__volume_generation.VolumeConstruction,
    polygons: np.ndarray,
    /,
    *,
    work: _dependency_phydrax_meshing__layer_core_resources.LayerCoreSourceWork,
) -> None:
    work.charge(complex_.vertices.shape[0])
    points = np.asarray(core.mesh.coordinates, dtype=np.float64)
    if not layer_topology.bitwise_equal_points(
        points[: complex_.vertices.shape[0]], complex_.vertices
    ):
        raise compliance_failure(
            "generate_layer_core.complex_ must retain fixed PLC vertices, facets, orientation, and region incidence; got native core recovery moved, renumbered, or dropped fixed vertices",
            stage=_dependency_phydrax_meshing__trace.MeshingStageKind.VOLUME_FILL.value,
        )
    table = layer_topology.faces(core.mesh, core.cell_regions, work=work)
    constrained = {tuple(sorted(row.tolist())) for row in polygons}
    for polygon, facet in zip(polygons, complex_.polygon_facets, strict=True):
        work.charge(1)
        incidents = table.get(tuple(sorted(polygon.tolist())), [])
        expected = complex_.facet_regions[facet]
        if len(incidents) != np.count_nonzero(expected >= 0):
            raise compliance_failure(
                "generate_layer_core.complex_ must retain fixed PLC vertices, facets, orientation, and region incidence; got native core recovery split or dropped a fixed facet",
                stage=_dependency_phydrax_meshing__trace.MeshingStageKind.VOLUME_FILL.value,
            )
        for side, region in enumerate(expected):
            if region < 0:
                continue
            oriented = (
                tuple(polygon.tolist()) if side == 1 else tuple(polygon[::-1].tolist())
            )
            if not any(
                face.region == region
                and layer_topology.cycle(face.vertices) == layer_topology.cycle(oriented)
                for face in incidents
            ):
                raise compliance_failure(
                    "generate_layer_core.complex_ must retain fixed PLC vertices, facets, orientation, and region incidence; got native core recovery changed a fixed oriented facet or its material incidence",
                    stage=_dependency_phydrax_meshing__trace.MeshingStageKind.VOLUME_FILL.value,
                )
    if any(
        len(incidents) == 1 and key not in constrained for key, incidents in table.items()
    ):
        raise compliance_failure(
            "generate_layer_core.complex_ must retain fixed PLC vertices, facets, orientation, and region incidence; got native core recovery introduced an undeclared external boundary",
            stage=_dependency_phydrax_meshing__trace.MeshingStageKind.VOLUME_FILL.value,
        )


def _organization(
    mesh: CellMesh,
    regions: np.ndarray,
    layer_count: int,
    layers: _dependency_phydrax_meshing__boundary_layer.BoundaryLayerMesh,
    complex_: plc_source_owner.PiecewiseLinearComplex,
    mapping: np.ndarray,
    polygons: np.ndarray,
    cap_ids: np.ndarray,
    layer_faces: dict[tuple[int, ...], list[layer_topology.LayerCoreFace]],
    region_ids: tuple[str, ...],
    /,
) -> tuple[
    tuple[_dependency_phydrax_meshing__organization.MeshZone, ...],
    tuple[_dependency_phydrax_meshing__organization.MeshPatch, ...],
    tuple[_dependency_phydrax_meshing__organization.MeshLabel, ...],
    tuple[_dependency_phydrax_meshing__organization.MeshAttribute, ...],
]:
    cells = np.sort(np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64))
    zones = tuple(
        _dependency_phydrax_meshing__organization.MeshZone(
            name,
            _dependency_phydrax_meshing__organization.MeshZoneRole.REGION,
            mesh_entity_scope(mesh, 3, cells[regions == index]),
        )
        for index, name in enumerate(region_ids)
        if np.any(regions == index)
    )
    labels = (
        _dependency_phydrax_meshing__organization.MeshLabel(
            "boundary-layer", mesh_entity_scope(mesh, 3, cells[:layer_count])
        ),
        _dependency_phydrax_meshing__organization.MeshLabel(
            "core", mesh_entity_scope(mesh, 3, cells[layer_count:])
        ),
    )
    attributes = tuple(
        _dependency_phydrax_meshing__organization.MeshAttribute(
            name,
            _dependency_phydrax_meshing__organization.MeshAttributeRole.MARKER,
            mesh_entity_scope(mesh, 3, cells),
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
    # Standard identity keys deliberately retain distinct periodic/lifted vertices.
    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    keys = dict(
        zip(
            _dependency_phydrax_meshing__layer_core_resources.row_entity_vertex_keys(
                mesh, 2
            ),
            face_ids.tolist(),
            strict=True,
        )
    )
    patches: list[_dependency_phydrax_meshing__organization.MeshPatch] = []
    for facet in range(complex_.facet_count):
        selected = np.flatnonzero(complex_.polygon_facets == facet)
        ids = np.asarray(
            [
                keys[tuple(sorted(mapping[polygons[index]].tolist()))]
                for index in selected
            ],
            dtype=np.int64,
        )
        patches.append(
            _dependency_phydrax_meshing__organization.MeshPatch(
                f"facet:{facet}", mesh_entity_scope(mesh, 2, ids), connected=False
            )
        )
    cap_keys = {tuple(sorted(mapping[polygons[index]].tolist())) for index in cap_ids}
    if cap_keys:
        patches.append(
            _dependency_phydrax_meshing__organization.MeshPatch(
                "layer-core-interface",
                mesh_entity_scope(
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
        patches.append(
            _dependency_phydrax_meshing__organization.MeshPatch(
                "wall", mesh_entity_scope(mesh, 2, wall_ids), connected=False
            )
        )
    return zones, tuple(patches), labels, attributes


@checked
def core_request(
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    layer_control: _dependency_phydrax_meshing__controls.BoundaryLayerControl,
    /,
) -> _dependency_phydrax_meshing__contracts.VolumeMeshingSpec:
    """Combine the physical core cap with sizing without rewriting its evidence."""
    bounds = tuple(
        control.core_maximum_size
        for control in (*specification.layer_controls, layer_control)
        if control.core_maximum_size is not None
    )
    if not bounds:
        return specification
    original = specification.size_controls[0]
    match original:
        case _dependency_phydrax_meshing__sizing.UniformSizeControl() if (
            len(specification.size_controls) == 1
        ):
            pass
        case _:
            raise ValueError(
                "core_request.specification must provide one uniform core size request; got missing or incompatible one uniform core size request"
            )
    cap = min(bounds)
    if (
        original.strength is _dependency_phydrax_meshing__sizing.SizeControlStrength.HARD
        and (
            (original.minimum_size is not None and original.minimum_size > cap)
            or original.target_size
            > cap
            + specification.size_compliance.absolute_tolerance
            + specification.size_compliance.relative_tolerance * original.target_size
        )
    ):
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.CONTROL_CONFLICT,
            "core_request.specification must declare a uniform size compatible with the layer core-size cap; got the hard core size request conflicts with the layer control's core maximum size",
            stage=_dependency_phydrax_meshing__trace.MeshingStageKind.CONTROL_RESOLUTION.value,
            requested=(("core_maximum_size", cap), ("target_size", original.target_size)),
        )
    target = min(original.target_size, cap)
    minimum = original.minimum_size
    if (
        original.strength is _dependency_phydrax_meshing__sizing.SizeControlStrength.SOFT
        and minimum is not None
        and minimum > target
    ):
        minimum = None
    size = _dependency_phydrax_meshing__sizing.UniformSizeControl(
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
    return _dependency_phydrax_meshing__contracts.VolumeMeshingSpec(
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
    core: _dependency_phydrax_meshing__volume_generation.VolumeConstruction,
    target: CellMesh,
    vertex_map: np.ndarray,
    /,
) -> tuple[association_state_owner.GeometryAssociation, ...]:
    """Rebind immutable PLC vertex/edge/facet identities and standard orientation."""
    output: list[association_state_owner.GeometryAssociation] = []
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
                "CellMeshingResult.associations must bind standard cells, vertices, faces or edges; got incompatible Native core association"
            )
        dimension = dimensions[0]
        source_ids = np.asarray(
            core.mesh.entity_set(dimension).entity_ids, dtype=np.int64
        )
        target_ids = np.asarray(target.entity_set(dimension).entity_ids, dtype=np.int64)
        source_keys = (
            tuple((int(identifier),) for identifier in source_ids)
            if dimension == 0
            else _dependency_phydrax_meshing__canonical.entity_vertex_keys(
                core.mesh, dimension
            )
        )
        target_keys = (
            tuple((int(identifier),) for identifier in target_ids)
            if dimension == 0
            else _dependency_phydrax_meshing__canonical.entity_vertex_keys(
                target, dimension
            )
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
            association_state_owner.GeometryAssociation(
                association.association_kind,
                association.source_id,
                association.source_revision,
                target.entity_set(dimension).entity_set_id,
                np.asarray(remapped, dtype=np.int64),
                association.source_entities.entity_ids(),
                association.residuals,
                resolved=association.resolved,
                ambiguous=association.ambiguous,
                exact=association.exact,
                source_dimensions=association.source_dimensions,
                source_indices=association.source_indices,
                source_occurrence_paths=association.source_entities.occurrence_paths(),
                source_entity_roles=association.source_entities.roles(),
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


@final
class LayerCoreCarrier(StrictModule, NonTrainableState):
    """Standard core-local regions, fixed interface identity, and seam ancestry."""

    region_ids: tuple[str, ...]
    core_regions: np.ndarray
    core_specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec
    mapping: np.ndarray
    cap_ids: np.ndarray
    polygons: np.ndarray
    initial_points: np.ndarray
    ancestry: tuple[np.ndarray, np.ndarray] | None


@final
class LayerCoreTopology(StrictModule, NonTrainableState):
    """Combined standard carrier and the core's vertex-row correspondence."""

    mesh: CellMesh
    core_map: np.ndarray


@final
class LayerCoreLabels(StrictModule, NonTrainableState):
    """Ordered inherited labels and the combined facet identity table."""

    labels: tuple[_dependency_phydrax_meshing__organization.MeshLabel, ...]
    face_keys: dict[tuple[int, ...], int]


@final
class LayerCoreAssociations(StrictModule, NonTrainableState):
    associations: tuple[association_state_owner.GeometryAssociation, ...]


def _prepare_layer_carrier(
    layers: _dependency_phydrax_meshing__boundary_layer.BoundaryLayerMesh,
    complex_: plc_source_owner.PiecewiseLinearComplex,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    vertex_layer_ids: ArrayLike,
    cap_polygon_ids: ArrayLike,
    regions: np.ndarray,
    source_id: str,
    source_revision: str,
    source_binding: _dependency_phydrax_meshing_providers__native_sources.NativeLayerCoreSource
    | None,
    work: _dependency_phydrax_meshing__layer_core_resources.LayerCoreSourceWork,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> LayerCoreCarrier:
    """Bind authored identity before admitting native fill."""
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
            "generate_layer_core.source_binding must bind the authored source, layer realization, PLC, and integer ancestry; got incompatible core construction"
        )
    region_ids, core_regions = (
        _dependency_phydrax_meshing__layer_core_regions.prepare_layer_region_identity(
            complex_,
            regions,
            None if source_binding is None else source_binding.region_ids,
            None if source_binding is None else source_binding.core_region_map,
        )
    )
    core_specification = (
        _dependency_phydrax_meshing__layer_core_resources.reserve_layer_storage(
            layers,
            complex_,
            core_request(specification, layers.control),
            layer_topology.integer_array(
                vertex_layer_ids, (complex_.vertices.shape[0],), "vertex_layer_ids"
            ),
        )
    )
    mapping, cap_ids, polygons, initial_points = layer_topology.prepare_identity(
        layers,
        complex_,
        vertex_layer_ids,
        cap_polygon_ids,
        work=work,
    )
    if layers.mesh.periodic_topology is not None and source_binding is None:
        raise ValueError(
            "generate_layer_core.source_binding must provide its source-authored seam anatomy before construction; got missing or incompatible its source-authored seam anatomy before construction"
        )
    ancestry = (
        None
        if source_binding is None
        else _dependency_phydrax_meshing__layer_core_periodic.prepare_core_periodic(
            source_binding,
            mapping,
            polygons,
            initial_points,
            work=work,
            budget=budget,
        )
    )
    return LayerCoreCarrier(
        region_ids,
        core_regions,
        core_specification,
        mapping,
        cap_ids,
        polygons,
        initial_points,
        ancestry,
    )


def _admit_layer_composition(
    layers: _dependency_phydrax_meshing__boundary_layer.BoundaryLayerMesh,
    complex_: plc_source_owner.PiecewiseLinearComplex,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    core: _dependency_phydrax_meshing__volume_generation.VolumeConstruction,
    carrier: LayerCoreCarrier,
    scope: NativeExecutionScope,
    resources: ExitStack,
    work: _dependency_phydrax_meshing__layer_core_resources.LayerCoreSourceWork,
) -> _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget:
    """Reserve composition work and certify the recovered immutable core boundary."""
    polygons, cap_ids = carrier.polygons, carrier.cap_ids
    work.ticket = resources.enter_context(
        scope.batch(
            min(
                specification.limits.maximum_work_units,
                scope.remaining().remaining_work_units,
            )
            if layers.mesh.periodic_topology is not None
            else (
                complex_.vertices.shape[0]
                + polygons.shape[0]
                + cap_ids.size
                + core.mesh.coordinates.shape[0]
                + 9 * core.cell_regions.size
                + layers.mesh.coordinates.shape[0]
                + sum(
                    block.cell_count
                    * (
                        1
                        + 4 * len(reference_cell_topology(block.cell_kind).entities[2])
                        + len(reference_cell_topology(block.cell_kind).entities[1])
                        + sum(
                            len(face) - 2
                            for face in reference_cell_topology(block.cell_kind).entities[
                                2
                            ]
                        )
                    )
                    for block in layers.mesh.blocks
                )
            ),
            bytes_upper=min(
                specification.limits.maximum_scratch_bytes,
                scope.remaining().remaining_managed_bytes,
            ),
        )
    )
    budget = (
        _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget(
            work.ticket,
            maximum_work_units=specification.limits.maximum_work_units,
            maximum_memory_bytes=specification.limits.maximum_scratch_bytes,
        )
    )
    resources.enter_context(budget.activate())
    _require_fixed(complex_, core, polygons, work=work)
    return budget


def _compose_layer_topology(
    layers: _dependency_phydrax_meshing__boundary_layer.BoundaryLayerMesh,
    complex_: plc_source_owner.PiecewiseLinearComplex,
    core: _dependency_phydrax_meshing__volume_generation.VolumeConstruction,
    carrier: LayerCoreCarrier,
    source_revision: str,
    source_binding: _dependency_phydrax_meshing_providers__native_sources.NativeLayerCoreSource
    | None,
    work: _dependency_phydrax_meshing__layer_core_resources.LayerCoreSourceWork,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> LayerCoreTopology:
    """Preserve layer IDs while appending recovered core incidence."""
    mapping, initial_points, ancestry = (
        carrier.mapping,
        carrier.initial_points,
        carrier.ancestry,
    )
    core_points = np.asarray(core.mesh.coordinates, dtype=np.float64)
    work.charge(
        core_points.shape[0]
        + sum(block.cell_count for block in layers.mesh.blocks)
        + core.cell_regions.size
    )
    points = np.concatenate((initial_points, core_points[complex_.vertices.shape[0] :]))
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
    layer_ids = np.sort(np.asarray(layers.mesh.entity_set(3).entity_ids, dtype=np.int64))
    core_start = int(layer_ids[-1]) + 1
    if core_start + core.cell_regions.size - 1 > np.iinfo(np.int64).max:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "BoundaryLayerMesh.mesh must leave sufficient int64 identity capacity for recovered core entities; got the immutable layer cell registry has no room for the core's identities",
            stage=_dependency_phydrax_meshing__trace.MeshingStageKind.CANONICALIZATION.value,
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
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "BoundaryLayerMesh.mesh must leave sufficient int64 identity capacity for recovered core entities; got the immutable layer vertex registry has no room for the core's identities",
            stage=_dependency_phydrax_meshing__trace.MeshingStageKind.CANONICALIZATION.value,
        )
    new_ids = (
        next_vertex + np.arange(extra, dtype=np.int64)
        if extra
        else np.empty(0, dtype=np.int64)
    )
    vertex_ids = np.concatenate((layer_vertex_ids, new_ids))
    plain = CellMesh(
        points,
        blocks,
        vertex_global_ids=vertex_ids,
        numeric_version=source_revision,
    )
    if source_binding is not None:
        plain = _dependency_phydrax_meshing__layer_core_periodic.bind_core_periodic(
            source_binding, plain, ancestry, budget=budget
        )
    mesh = _dependency_phydrax_meshing__canonical.canonicalize_cell_mesh(plain)
    return LayerCoreTopology(mesh, core_map)


def _compose_layer_labels(
    mesh: CellMesh,
    core: _dependency_phydrax_meshing__volume_generation.VolumeConstruction,
    core_map: np.ndarray,
    combined_faces: dict[tuple[int, ...], list[layer_topology.LayerCoreFace]],
    labels: tuple[_dependency_phydrax_meshing__organization.MeshLabel, ...],
) -> LayerCoreLabels:
    """Transport interface and core labels using vertex correspondence."""
    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    face_keys = dict(
        zip(
            _dependency_phydrax_meshing__layer_core_resources.row_entity_vertex_keys(
                mesh, 2
            ),
            face_ids.tolist(),
            strict=True,
        )
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
        labels += (
            _dependency_phydrax_meshing__organization.MeshLabel(
                "interface", mesh_entity_scope(mesh, 2, interface_ids)
            ),
        )
    for label in core.labels:
        if label.name == "interface":
            continue
        dimension = label.scope.entity_dimension
        old_ids = np.asarray(core.mesh.entity_set(dimension).entity_ids, dtype=np.int64)
        old_keys = (
            _dependency_phydrax_meshing__layer_core_resources.row_entity_vertex_keys(
                core.mesh, dimension
            )
        )
        new_ids = dict(
            zip(
                _dependency_phydrax_meshing__layer_core_resources.row_entity_vertex_keys(
                    mesh, dimension
                ),
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
        labels += (
            _dependency_phydrax_meshing__organization.MeshLabel(
                label.name, mesh_entity_scope(mesh, dimension, remapped)
            ),
        )
    return LayerCoreLabels(labels, face_keys)


def _compose_layer_associations(
    layers: _dependency_phydrax_meshing__boundary_layer.BoundaryLayerMesh,
    mesh: CellMesh,
    core: _dependency_phydrax_meshing__volume_generation.VolumeConstruction,
    core_map: np.ndarray,
    combined_regions: np.ndarray,
    regions: np.ndarray,
    region_ids: tuple[str, ...],
    mapping: np.ndarray,
    polygons: np.ndarray,
    cap_ids: np.ndarray,
    face_keys: dict[tuple[int, ...], int],
    source_id: str,
    source_revision: str,
    work: _dependency_phydrax_meshing__layer_core_resources.LayerCoreSourceWork,
) -> LayerCoreAssociations:
    """Bind region, core, cap, and source layer source ancestry in order."""
    cells = np.sort(np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64))
    associations = (
        association_state_owner.GeometryAssociation(
            association_state_owner.GeometryAssociationKind.PIECEWISE_LINEAR,
            source_id,
            source_revision,
            mesh.entity_set(3).entity_set_id,
            cells,
            tuple(
                f"{source_revision}:region:{int(region)}" for region in combined_regions
            ),
            np.zeros(cells.size, dtype=np.float64),
            exact=True,
            source_dimensions=np.full(cells.size, 3, dtype=np.int8),
            source_indices=combined_regions,
            source_entity_roles=(association_state_owner.GeometrySourceEntityRole.REGION,)
            * cells.size,
        ),
    )
    associations += _rebind_core_associations(core, mesh, core_map)

    cap_association = None
    if layers.cap is not None and cap_ids.size:
        cap_face_ids = np.concatenate(
            [np.asarray(block.global_ids, dtype=np.int64) for block in layers.cap.blocks]
        )
        rows = mapping[polygons[cap_ids]]
        target_ids = np.asarray(
            [face_keys[tuple(sorted(row.tolist()))] for row in rows],
            dtype=np.int64,
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
        cap_association = association_state_owner.GeometryAssociation(
            association_state_owner.GeometryAssociationKind.PIECEWISE_LINEAR,
            layers.result_id,
            layers.result_id,
            mesh.entity_set(2).entity_set_id,
            target_ids,
            tuple(
                _dependency_phydrax_meshing__host_banks.entity_key(
                    layers.result_id,
                    association_state_owner.GeometrySourceEntityRole.FACET.value,
                    int(identifier),
                )
                for identifier in cap_face_ids
            ),
            np.zeros(cap_ids.size, dtype=np.float64),
            exact=True,
            source_dimensions=np.full(cap_ids.size, 2, dtype=np.int8),
            source_indices=cap_face_ids,
            orientations=signs,
            source_entity_roles=(association_state_owner.GeometrySourceEntityRole.FACET,)
            * cap_ids.size,
        )
    associations += (
        _owner_phydrax_meshing__layer_core_association.layer_source_associations(
            layers,
            mesh,
            regions,
            region_ids,
            cap_association=cap_association,
            work=work,
        )
    )
    return LayerCoreAssociations(associations)


def _compose_layer_domain(
    layers: _dependency_phydrax_meshing__boundary_layer.BoundaryLayerMesh,
    complex_: plc_source_owner.PiecewiseLinearComplex,
    mesh: CellMesh,
    regions: np.ndarray,
    combined_regions: np.ndarray,
    carrier: LayerCoreCarrier,
    source_id: str,
    work: _dependency_phydrax_meshing__layer_core_resources.LayerCoreSourceWork,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[
    domain_coverage_owner.PiecewiseLinearDomain,
    dict[tuple[int, ...], list[layer_topology.LayerCoreFace]],
    dict[tuple[int, ...], list[layer_topology.LayerCoreFace]],
]:
    """Construct the declared domain and require two incidents at every fixed cap."""

    # Domain degeneracy predicates draw from the admitted composition ticket.
    with work.ticket.native_target():
        domain, layer_faces = layer_topology.combined_domain(
            layers,
            complex_,
            regions,
            carrier.mapping,
            carrier.cap_ids,
            carrier.polygons,
            carrier.initial_points,
            source_id,
            carrier.region_ids,
            carrier.core_regions,
            work=work,
        )
    combined_faces = layer_topology.faces(mesh, combined_regions, work=work)
    for index in carrier.cap_ids:
        work.charge(1)
        key = tuple(sorted(carrier.mapping[carrier.polygons[index]].tolist()))
        if len(combined_faces[key]) != 2:
            raise compliance_failure(
                "generate_layer_core.complex_ must share the immutable cap interface with the accepted layers; got layer and core do not share the immutable interface",
                stage=_dependency_phydrax_meshing__trace.MeshingStageKind.VOLUME_FILL.value,
            )
    return domain, layer_faces, combined_faces


def _layer_core_identity(
    layers: _dependency_phydrax_meshing__boundary_layer.BoundaryLayerMesh,
    complex_: plc_source_owner.PiecewiseLinearComplex,
    mesh: CellMesh,
    domain: domain_coverage_owner.PiecewiseLinearDomain,
    vertex_layer_ids: ArrayLike,
    regions: np.ndarray,
    carrier: LayerCoreCarrier,
) -> str:
    """Fingerprint explicit interface and region identities, never coordinates."""
    return canonical_fingerprint(
        {
            "kind": "native-layer-core",
            "layers": layers.result_id,
            "complex": complex_.complex_id,
            "mesh": mesh.mesh_id,
            "domain": domain.domain_id,
            "vertex_layer_ids": array_tree_fingerprint(
                np.asarray(vertex_layer_ids, dtype=np.int64)
            ),
            "cap_polygon_ids": array_tree_fingerprint(carrier.cap_ids),
            "layer_regions": array_tree_fingerprint(regions),
            "region_ids": carrier.region_ids,
            "core_region_map": array_tree_fingerprint(carrier.core_regions),
        }
    )


# Retire preparation resources before native fill; compose only after readmission.
@checked
def generate_layer_core(
    layers: _dependency_phydrax_meshing__boundary_layer.BoundaryLayerMesh,
    complex_: plc_source_owner.PiecewiseLinearComplex,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    schedule: _dependency_phydrax_meshing__volume_generation.NativeVolumeSchedule,
    /,
    *,
    scope: NativeExecutionScope,
    validity_policy: _dependency_phydrax_discretization__cell_geometry_validity.CellValidityPolicy,
    vertex_layer_ids: ArrayLike,
    cap_polygon_ids: ArrayLike,
    layer_regions: ArrayLike,
    source_id: str,
    source_revision: str,
    input_id: str,
    source_binding: _dependency_phydrax_meshing_providers__native_sources.NativeLayerCoreSource
    | None = None,
    source_work_units: int = 0,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> layer_topology.LayerCoreConstruction:
    """Compose immutable layers with native filling of the full remaining PLC.

    ``vertex_layer_ids[i]`` identifies PLC vertex i with a layer vertex, or -1
    declares a new vertex. ``cap_polygon_ids`` follows cap block order.
    ``layer_regions`` follows sorted layer global cell IDs and indexes the
    source binding's composite region namespace. The core PLC retains only
    bounded core-local regions and its map into that namespace.
    ``validity_policy`` is the publishing audit's cell validity policy, whose
    determinant floor the native core improvement enforces.
    """
    with ExitStack() as resources:
        count = sum(block.cell_count for block in layers.mesh.blocks)
        regions = layer_topology.integer_array(layer_regions, (count,), "layer_regions")
        allowance = scope.remaining()
        source_upper = (
            2 * complex_.vertices.shape[0] + np.asarray(cap_polygon_ids).size
            if source_binding is None
            else _dependency_phydrax_meshing__layer_core_resources.layer_source_work_upper(
                source_binding, specification
            )
        )
        if layers.mesh.periodic_topology is not None:
            source_upper = min(
                specification.limits.maximum_work_units, allowance.remaining_work_units
            )
        work = _dependency_phydrax_meshing__layer_core_resources.LayerCoreSourceWork(
            specification.limits.maximum_work_units,
            source_work_units,
            ticket=resources.enter_context(
                scope.batch(
                    source_upper,
                    bytes_upper=min(
                        specification.limits.maximum_scratch_bytes,
                        allowance.remaining_managed_bytes,
                    ),
                )
            ),
        )
        budget = _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget(
            work.ticket,
            maximum_work_units=specification.limits.maximum_work_units,
            maximum_memory_bytes=specification.limits.maximum_scratch_bytes,
        )
        resources.enter_context(budget.activate())
        carrier = _prepare_layer_carrier(
            layers,
            complex_,
            specification,
            vertex_layer_ids,
            cap_polygon_ids,
            regions,
            source_id,
            source_revision,
            source_binding,
            work,
            budget,
        )
        region_ids, core_regions = carrier.region_ids, carrier.core_regions
        core_specification = carrier.core_specification
        mapping, cap_ids, polygons = carrier.mapping, carrier.cap_ids, carrier.polygons
        resources.close()
        del budget
        core = _dependency_phydrax_meshing__volume_generation.generate_plc_volume(
            complex_,
            core_specification,
            schedule,
            validity_policy=validity_policy,
            scope=scope,
            source_id=complex_.complex_id,
            source_revision=complex_.complex_id,
            input_id=input_id,
            record_phase=record_phase,
            source_work_units=work.work_units,
        )
        work.work_units = core.work_units
        budget = _admit_layer_composition(
            layers, complex_, specification, core, carrier, scope, resources, work
        )
        with measure_phase(record_phase, "topology_construction"):
            topology = _compose_layer_topology(
                layers,
                complex_,
                core,
                carrier,
                source_revision,
                source_binding,
                work,
                budget,
            )
            mesh, core_map = topology.mesh, topology.core_map
        combined_regions = np.concatenate((regions, core_regions[core.cell_regions]))
        domain, layer_faces, combined_faces = _compose_layer_domain(
            layers,
            complex_,
            mesh,
            regions,
            combined_regions,
            carrier,
            source_id,
            work,
            budget,
        )
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
        inherited = _compose_layer_labels(mesh, core, core_map, combined_faces, labels)
        labels, face_keys = inherited.labels, inherited.face_keys
        cells = np.sort(np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64))

        with (
            measure_phase(record_phase, "geometry_association"),
            work.ticket.native_target(),
        ):
            associations = _compose_layer_associations(
                layers,
                mesh,
                core,
                core_map,
                combined_regions,
                regions,
                region_ids,
                mapping,
                polygons,
                cap_ids,
                face_keys,
                source_id,
                source_revision,
                work,
            ).associations
        construction_id = _layer_core_identity(
            layers, complex_, mesh, domain, vertex_layer_ids, regions, carrier
        )
        stages = (
            _dependency_phydrax_meshing__trace.MeshingStageReport(
                _dependency_phydrax_meshing__trace.MeshingStageKind.LAYER_GENERATION,
                _dependency_phydrax_meshing__trace.MeshingStageStatus.PASSED,
                input_ids=(layers.control_id,),
                output_ids=(layers.result_id,),
                created_count=count,
            ),
            *core.stages,
            _dependency_phydrax_meshing__trace.MeshingStageReport(
                _dependency_phydrax_meshing__trace.MeshingStageKind.CANONICALIZATION,
                _dependency_phydrax_meshing__trace.MeshingStageStatus.PASSED,
                input_ids=(layers.result_id, core.mesh.mesh_id),
                output_ids=(construction_id,),
            ),
        )
        block_ids = np.concatenate(
            [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
        )
        block_regions = combined_regions[np.searchsorted(cells, block_ids)]
        return layer_topology.LayerCoreConstruction(
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


@final
class LayerCoreSizeEvidence(StrictModule, NonTrainableState):
    requested: tuple[tuple[str, float], ...]
    achieved: tuple[tuple[str, float], ...]
    issues: tuple[str, ...]


def _core_size_compliance(
    layers: _dependency_phydrax_meshing__boundary_layer.BoundaryLayerMesh,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    core: _dependency_phydrax_meshing__volume_generation.VolumeConstruction,
) -> LayerCoreSizeEvidence:
    """Measure the authored core-size cap on recovered core edges."""
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
        core_edges = (
            _dependency_phydrax_meshing__layer_core_resources.row_entity_vertex_keys(
                core.mesh, 1
            )
        )
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
    return LayerCoreSizeEvidence(
        core_size_requested, core_size_achieved, core_size_issues
    )


def _layer_core_compliance(
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    evidence: _dependency_phydrax_meshing__boundary_layer.BoundaryLayerEvidence,
    core_compliance: _dependency_phydrax_meshing__result.MeshingComplianceReport,
    core_size: LayerCoreSizeEvidence,
    preparation: layer_publication.LayerRoutePreparation,
) -> _dependency_phydrax_meshing__result.MeshingComplianceReport:
    core_size_requested, core_size_achieved, core_size_issues = (
        core_size.requested,
        core_size.achieved,
        core_size.issues,
    )
    periodic_requested, periodic_achieved = (
        preparation.periodic_requested,
        preparation.periodic_achieved,
    )
    work = preparation.work
    compliance = _dependency_phydrax_meshing__result.MeshingComplianceReport(
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
    return compliance


def _layer_publication_work_upper(
    source: _dependency_phydrax_meshing_providers__native_sources.NativeLayerCoreSource,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    construction: layer_topology.LayerCoreConstruction,
    /,
) -> int:
    """Bound control and fidelity-request visits from prepared incidence counts.

    Each ancestry pass charges mapping and shared rows, cap polygons, PLC
    polygons, and source-wall cells once. Lateral ancestry charges each half-cell
    corner twice at most: once for the predecessor table and once while tracing
    roots, which are retained after the first visit. The extra visit bounds an
    invalid cyclic trace before its refusal. Cap/PLC edges and layer facets have
    their declared cardinalities; no loop multiplies them by all layer cells.

    Controls charge one combined facet pass and at most two visits per authored
    face per patch. Scoped fidelity repeats ancestry once, then visits authored
    faces and requested source IDs. Numeric improvement is already settled and
    is absent from this publication ticket.
    """
    layers = source.layers
    polygons = source.complex.polygon_facets.size
    cap_polygons = source.cap_polygon_ids.size
    wall_cells = sum(block.cell_count for block in layers.source_wall.blocks)
    layer_facets = sum(
        block.cell_count * len(reference_cell_topology(block.cell_kind).entities[2])
        for block in layers.mesh.blocks
    )
    ancestry = (
        source.vertex_layer_ids.size
        + int(np.count_nonzero(source.vertex_layer_ids >= 0))
        + cap_polygons
        + polygons
        + wall_cells
    )
    authored_faces = polygons + wall_cells
    if source.core_facet_source_ids is not None:
        half_corners = sum(block.vertices.size // 2 for block in layers.mesh.blocks)
        ancestry += (
            2 * half_corners
            + 1
            + 3 * cap_polygons
            + source.complex.polygon_vertices.size
            + 2 * layer_facets
        )
        authored_faces += layer_facets
    bound = len(construction.zones)
    if specification.patch_controls:
        combined_facets = sum(
            block.cell_count * len(reference_cell_topology(block.cell_kind).entities[2])
            for block in construction.mesh.blocks
        )
        bound += (
            ancestry
            + combined_facets
            + 2 * len(specification.patch_controls) * authored_faces
        )
    if source.source_domain is not None:
        scope = source.source_boundary_scope
        bound += (
            ancestry + authored_faces + (0 if scope is None else scope.entity_ids.size)
        )
    return bound


@checked
def execute_layer_core_route(
    layers: _dependency_phydrax_meshing__boundary_layer.BoundaryLayerMesh,
    complex_: plc_source_owner.PiecewiseLinearComplex,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    schedule: _dependency_phydrax_meshing__volume_generation.NativeVolumeSchedule,
    coordinate_contract: SpatialCoordinateContract,
    provider: _dependency_phydrax_meshing__contracts.MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
    vertex_layer_ids: ArrayLike,
    cap_polygon_ids: ArrayLike,
    layer_regions: ArrayLike,
    source_id: str,
    source_revision: str,
    generation_specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    original_source: _dependency_phydrax_meshing_providers__native_sources.NativeLayerCoreSource,
    source_preparation_work_units: int,
    prepared_owner: layer_publication.PreparedLayerIdentity,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> _dependency_phydrax_meshing__result.CellMeshingResult:
    """Generate and independently certify the entire native layer/core domain."""
    with ExitStack() as resources:
        if not native_preparation_ledger(scope).live_contains(prepared_owner):
            raise ValueError(
                f"execute_layer_core_route.prepared_owner must be registered in this live native scope; got unregistered preparation {prepared_owner.prepared_id!r} (standalone execution requires execute_layer_route)"
            )
        if (prepared_owner.source_binding_id, prepared_owner.specification_id) != (
            original_source.binding_id,
            specification.specification_id,
        ):
            raise ValueError(
                "execute_layer_core_route.prepared_owner must bind the live source/specification preparation within publication limits; got the live layer preparation belongs to another source or specification"
            )

        started = scope.operation_started
        preparation = layer_publication.prepare_layer_route(
            original_source,
            specification,
            source_preparation_work_units,
            scope,
            resources,
            started=started,
        )
        work = preparation.work
        audit_policy = preparation.policy.audit_policy
        construction = generate_layer_core(
            layers,
            complex_,
            generation_specification,
            schedule,
            validity_policy=audit_policy.validity_policy,
            scope=scope,
            vertex_layer_ids=vertex_layer_ids,
            cap_polygon_ids=cap_polygon_ids,
            layer_regions=layer_regions,
            source_id=source_id,
            source_revision=source_revision,
            input_id=plan_id,
            record_phase=record_phase,
            source_binding=original_source,
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
            raise _leaf_phydrax_meshing__failures.MeshingFailure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "execute_layer_core_route.prepared_owner must bind the live source/specification preparation within publication limits; got the combined native layer/core mesh exceeds its declared resource limits",
                stage=_dependency_phydrax_meshing__trace.MeshingStageKind.VOLUME_FILL.value,
                requested=tuple((f"maximum_{name}", bound) for name, _, bound in checks),
                achieved=tuple((name, actual) for name, actual, _ in checks),
            )
        route_checks.check_deadline(
            started,
            limits,
            _dependency_phydrax_meshing__trace.MeshingStageKind.VOLUME_FILL,
        )
        core = construction.core
        with measure_phase(record_phase, "compliance"):
            core_compliance = (
                _owner_phydrax_meshing_providers__native_volume.volume_compliance(
                    generation_specification,
                    schedule,
                    core,
                    np.asarray(core.mesh.coordinates, dtype=np.float64),
                    np.asarray(core.mesh.blocks[0].vertices, dtype=np.int64),
                )
            )
        core_size = _core_size_compliance(layers, specification, core)
        work.work_units = construction.work_units
        work.ticket = resources.enter_context(
            scope.batch(
                _layer_publication_work_upper(
                    original_source, specification, construction
                )
            )
        )
        zones, patches = (
            _dependency_phydrax_meshing__layer_core_controls.compose_layer_controls(
                original_source, specification, construction, work=work
            )
        )
        certification = (
            _owner_phydrax_meshing__layer_source_certification.layer_source_certification(
                original_source, specification, construction, work=work
            )
        )
        resources.close()
        compliance = _layer_core_compliance(
            specification, layers.evidence, core_compliance, core_size, preparation
        )
        publication = layer_publication.LayerCorePublication(
            mesh=mesh,
            compliance=compliance,
            stages=construction.stages,
            provenance={
                "kind": "native-layer-core",
                "plan": plan_id,
                "source": source_id,
                "specification": specification.specification_id,
                "source_revision": source_revision,
                "source_binding": original_source.binding_id,
                "layers": layers.result_id,
                "construction": construction.construction_id,
            },
            certification=certification,
            organization=_owner_phydrax_meshing_providers__native_publication.NativePublicationOrganization(
                zones=zones,
                patches=patches,
                labels=construction.labels,
                associations=construction.associations,
                attributes=construction.attributes,
            ),
        )
        return layer_publication.publish_layer_route(
            publication,
            preparation,
            coordinate_contract,
            provider,
            scope,
            workspace,
            record_phase=record_phase,
        )


def _reserved_layer_attributes(
    source: _dependency_phydrax_meshing__result.CellMeshingResult, /
) -> tuple[_dependency_phydrax_meshing__organization.MeshAttribute, ...]:
    """Select the reserved physical-layer pair without admitting unrelated data."""
    return tuple(
        attribute
        for attribute in source.attributes
        if attribute.name in ("layer_index", "layer_column")
    )


@checked
def validate_layer_index_attributes(
    source: _dependency_phydrax_meshing__result.CellMeshingResult, /
) -> None:
    """Validate the reserved layer pair; unrelated scientific attributes stay separate."""
    reserved = _reserved_layer_attributes(source)
    if not reserved:
        return
    if len(reserved) != 2 or {attribute.name for attribute in reserved} != {
        "layer_index",
        "layer_column",
    }:
        raise ValueError(
            "CellMeshingResult.attributes must provide standard layer_index and layer_column attributes; got missing or incompatible standard layer_index and layer_column attributes"
        )
    if source.mesh.topological_dimension != 3:
        raise ValueError(
            "CellMeshingResult.attributes must provide volumetric cell identities; got missing or incompatible volumetric cell identities"
        )
    cells = source.mesh.entity_set(3)
    for attribute in reserved:
        scope = attribute.scope
        values = np.asarray(attribute.values)
        if (
            attribute.role
            is not _dependency_phydrax_meshing__organization.MeshAttributeRole.MARKER
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
                "CellMeshingResult.attributes must bind every source cell with integer ancestry; got incompatible Layer markers"
            )
    attributes = {attribute.name: attribute for attribute in reserved}
    if not np.array_equal(
        np.asarray(attributes["layer_index"].values) < 0,
        np.asarray(attributes["layer_column"].values) < 0,
    ):
        raise ValueError(
            "CellMeshingResult.attributes must coincide; got incompatible Physical layer interval and column support"
        )


@checked
def layer_interval_classes(
    source: _dependency_phydrax_meshing__result.CellMeshingResult, /
) -> np.ndarray:
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


@checked
def remap_layer_index_attribute(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    lineage: _dependency_phydrax_meshing__lineage.MeshLineage,
    target: CellMesh,
    /,
) -> tuple[_dependency_phydrax_meshing__organization.MeshAttribute, ...]:
    """Inherit physical intervals only through full, unambiguous cell ancestry."""
    validate_layer_index_attributes(source)
    reserved = _reserved_layer_attributes(source)
    if not reserved:
        return ()
    if (
        lineage.source_topology_id != source.mesh.topology_id
        or lineage.target_topology_id != target.topology_id
    ):
        raise ValueError(
            "remap_layer_index_attribute.lineage must bind the realized source and target topologies; got incompatible Layer interval lineage"
        )
    relation = lineage.entity_lineage(3)
    if (
        relation.source_entity_set_id != source.mesh.entity_set(3).entity_set_id
        or relation.target_entity_set_id != target.entity_set(3).entity_set_id
    ):
        raise ValueError(
            "remap_layer_index_attribute.lineage must bind the realized source and target cell sets; got incompatible Layer interval lineage"
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
                "remap_layer_index_attribute.lineage must retain unambiguous physical interval and column ancestry for every target cell; got layer lineage names cells outside its bound source or target"
            )
        value = (int(old_values[index, 0]), int(old_values[index, 1]))
        if kind in (
            _dependency_phydrax_meshing__lineage.EntityLineageKind.UNKNOWN,
            _dependency_phydrax_meshing__lineage.EntityLineageKind.GENERATED_ON_GEOMETRY,
        ):
            raise ValueError(
                "remap_layer_index_attribute.lineage must retain unambiguous physical interval and column ancestry for every target cell; got unknown or generated cell ancestry cannot preserve a physical layer interval"
            )
        if (
            kind == _dependency_phydrax_meshing__lineage.EntityLineageKind.SWAPPED_FROM
            and value[0] >= 0
        ):
            raise ValueError(
                "remap_layer_index_attribute.lineage must provide an explicitly reconstructed physical schedule; got missing or incompatible an explicitly reconstructed physical schedule"
            )
        identifier = int(child)
        if identifier in inherited and inherited[identifier] != value:
            raise ValueError(
                "remap_layer_index_attribute.lineage must retain unambiguous physical interval and column ancestry for every target cell; got coarsening or merging different physical layer intervals or columns is forbidden"
            )
        inherited[identifier] = value
    if set(inherited) != set(targets.tolist()):
        raise ValueError(
            "remap_layer_index_attribute.lineage must provide full physical-interval lineage; got missing or incompatible full physical-interval lineage"
        )
    values = np.asarray(
        [inherited[int(identifier)] for identifier in targets], dtype=np.int64
    )
    return tuple(
        _dependency_phydrax_meshing__organization.MeshAttribute(
            name,
            _dependency_phydrax_meshing__organization.MeshAttributeRole.MARKER,
            mesh_entity_scope(target, 3, targets),
            values[:, index],
        )
        for index, name in enumerate(("layer_index", "layer_column"))
    )


__all__ = [
    "generate_layer_core",
    "execute_layer_core_route",
    "validate_layer_index_attributes",
    "layer_interval_classes",
    "remap_layer_index_attribute",
]
