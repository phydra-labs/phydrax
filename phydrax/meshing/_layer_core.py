#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Immutable advancing-layer/PLC composition, not a second tetrahedralizer."""

from __future__ import annotations

import numpy as np

import phydrax.discretization._cell_complex as cell_complex_owner
import phydrax.discretization._cell_geometry as cell_geometry_owner
import phydrax.discretization._periodic_exact as periodic_exact_owner
import phydrax.geometry._mapped_reference_domain as mapped_domain_owner
import phydrax.geometry._source_identity as source_identity_owner
import phydrax.meshing._canonical as canonical_owner
import phydrax.meshing._contracts as contracts_owner
import phydrax.meshing._layer_core_association as association_composition
import phydrax.meshing._layer_core_controls as controls_owner
import phydrax.meshing._layer_core_mapped as mapped_core_owner
import phydrax.meshing._layer_core_periodic as periodic_owner
import phydrax.meshing._layer_core_regions as regions_owner
import phydrax.meshing._layer_core_topology as layer_topology
import phydrax.meshing._organization as organization_owner
import phydrax.meshing._plc_source as plc_source_owner
import phydrax.meshing._polyhedral_generation as polyhedral_generation_owner
import phydrax.meshing._sizing as sizing_owner
import phydrax.meshing._volume_generation as volume_generation_owner
import phydrax.meshing.providers._native_sources as sources_owner

from ..discretization import CellBlock, CellMesh
from ..typing import checked
from ._contracts import VolumeMeshingSpec


def _require_fixed(
    complex_: plc_source_owner.PiecewiseLinearComplex,
    core: volume_generation_owner.VolumeConstruction,
    /,
) -> np.ndarray:
    """Require exact fixed PLC vertices, triangles, oriented sides and closure."""
    if complex_.boundary != "fixed":
        raise ValueError("Layer/core fill requires a fixed PLC boundary.")
    input_vertices = np.asarray(core.input_vertices, dtype=np.int64)
    core_points = np.asarray(core.mesh.coordinates, dtype=np.float64)
    vertex_count = complex_.vertices.shape[0]
    if (
        input_vertices.shape != (core_points.shape[0],)
        or np.any(input_vertices < -1)
        or np.any(input_vertices >= vertex_count)
    ):
        raise ValueError(
            "Native volume input-vertex ancestry is incompatible with the PLC."
        )
    retained = input_vertices >= 0
    if not np.array_equal(np.sort(input_vertices[retained]), np.arange(vertex_count)):
        raise ValueError("Fixed PLC fill must retain every input vertex exactly once.")
    if not layer_topology.bitwise_equal_points(
        core_points[retained],
        np.asarray(complex_.vertices, dtype=np.float64)[input_vertices[retained]],
    ):
        raise ValueError("Fixed PLC fill moved an authored input vertex.")
    core_regions = np.asarray(core.cell_regions, dtype=np.int64)
    if (
        core_regions.shape
        != np.sort(np.asarray(core.mesh.entity_set(3).entity_ids)).shape
    ):
        raise ValueError("Native volume regions must cover every recovered core cell.")
    if np.any(core_regions < 0) or np.any(core_regions >= complex_.region_count):
        raise ValueError("Native volume fill introduced an undeclared core region.")
    polygons = np.asarray(complex_.polygon_vertices, dtype=np.int64).reshape(-1, 3)
    boundary_faces = np.asarray(core.boundary_faces, dtype=np.int64)
    face_source_facets = np.asarray(core.face_source_facets, dtype=np.int64)
    face_orientations = np.asarray(core.face_orientations, dtype=np.int8)
    if (
        boundary_faces.ndim != 2
        or boundary_faces.shape[1] != 3
        or face_source_facets.shape != (boundary_faces.shape[0],)
        or face_orientations.shape != face_source_facets.shape
    ):
        raise ValueError(
            "Native volume boundary ancestry arrays have incompatible shapes."
        )
    if (
        np.any(boundary_faces < 0)
        or np.any(boundary_faces >= core_points.shape[0])
        or np.any(face_source_facets < 0)
        or np.any(face_source_facets >= complex_.facet_count)
        or np.any(~np.isin(face_orientations, (-1, 1)))
    ):
        raise ValueError(
            "Native volume boundary ancestry contains undeclared identities."
        )
    polygon_by_key: dict[tuple[int, ...], int] = {}
    for polygon_index, polygon in enumerate(polygons):
        key = tuple(sorted(int(value) for value in polygon))
        if key in polygon_by_key:
            raise ValueError("The fixed PLC contains duplicate oriented polygons.")
        polygon_by_key[key] = polygon_index
    polygon_rows = np.full(polygons.shape[0], -1, dtype=np.int64)
    for face_row, face in enumerate(boundary_faces):
        inherited = input_vertices[face]
        if np.any(inherited < 0):
            raise ValueError(
                "Fixed PLC boundary facets cannot contain generated vertices."
            )
        polygon_index = polygon_by_key.get(
            tuple(sorted(int(value) for value in inherited))
        )
        if polygon_index is None or polygon_rows[polygon_index] >= 0:
            raise ValueError("A recovered boundary triangle changed its source polygon.")
        if int(face_source_facets[face_row]) != int(
            complex_.polygon_facets[polygon_index]
        ):
            raise ValueError("A recovered boundary triangle changed its source facet.")
        polygon = polygons[polygon_index]
        oriented = tuple(int(value) for value in inherited)
        original = tuple(int(value) for value in polygon)
        if layer_topology.cycle(oriented) == layer_topology.cycle(original):
            sign = 1
        elif layer_topology.cycle(oriented) == layer_topology.cycle(
            tuple(reversed(original))
        ):
            sign = -1
        else:
            raise ValueError("A recovered boundary triangle changed its source vertices.")
        if sign != int(face_orientations[face_row]):
            raise ValueError(
                "Native boundary orientation disagrees with its source facet."
            )
        polygon_rows[polygon_index] = face_row
    if np.any(polygon_rows < 0):
        raise ValueError("Fixed PLC fill must retain every source polygon exactly once.")
    input_rows = np.full(vertex_count, -1, dtype=np.int64)
    input_rows[input_vertices[retained]] = np.flatnonzero(retained)
    core_regions = np.asarray(core.cell_regions, dtype=np.int64)
    face_table = layer_topology.faces(core.mesh, core_regions)
    source_keys: set[tuple[int, ...]] = set()
    for polygon_index, polygon in enumerate(polygons):
        core_vertices = tuple(int(input_rows[int(vertex)]) for vertex in polygon)
        key = tuple(sorted(core_vertices))
        source_keys.add(key)
        incidents = face_table.get(key, [])
        facet = int(complex_.polygon_facets[polygon_index])
        expected = np.asarray(complex_.facet_regions[facet], dtype=np.int64)
        if len(incidents) != int(np.count_nonzero(expected >= 0)):
            raise ValueError("Native core recovery split or dropped a fixed PLC facet.")
        for side, region in enumerate(expected):
            if region < 0:
                continue
            oriented = (
                tuple(polygon.tolist()) if side == 1 else tuple(polygon[::-1].tolist())
            )
            if not any(
                face.region == int(region)
                and layer_topology.cycle(
                    tuple(int(input_vertices[vertex]) for vertex in face.vertices)
                )
                == layer_topology.cycle(oriented)
                for face in incidents
            ):
                raise ValueError("Native core recovery changed an oriented PLC side.")
    if any(
        len(incidents) == 1 and key not in source_keys
        for key, incidents in face_table.items()
    ):
        raise ValueError(
            "Native core recovery introduced an undeclared external boundary."
        )
    return input_vertices


def core_request(
    specification: VolumeMeshingSpec,
    layer_controls: tuple[contracts_owner.BoundaryLayerControl, ...] = (),
    /,
    *,
    physical_scale: float = 1.0,
) -> VolumeMeshingSpec:
    """Combine requested core caps without rewriting the source size semantics."""
    if not np.isfinite(physical_scale) or physical_scale <= 0.0:
        raise ValueError("physical_scale must be positive and finite.")
    bounds = tuple(
        control.core_maximum_size / physical_scale
        for control in (*specification.layer_controls, *layer_controls)
        if control.core_maximum_size is not None
    )
    if not bounds:
        return specification
    original = specification.size_controls[0]
    match original:
        case sizing_owner.UniformSizeControl() if len(specification.size_controls) == 1:
            pass
        case _:
            raise ValueError(
                "A layer/core request requires one uniform core size control."
            )
    cap = min(bounds)
    if original.strength is sizing_owner.SizeControlStrength.HARD and (
        (original.minimum_size is not None and original.minimum_size > cap)
        or original.target_size
        > cap
        + specification.size_compliance.absolute_tolerance
        + specification.size_compliance.relative_tolerance * original.target_size
    ):
        raise contracts_owner.MeshingFailure(
            contracts_owner.MeshingFailureCategory.CONTROL_CONFLICT,
            "The hard core size request conflicts with the layer control's core maximum size.",
        )
    target = min(original.target_size, cap)
    minimum = original.minimum_size
    if (
        original.strength is sizing_owner.SizeControlStrength.SOFT
        and minimum is not None
        and minimum > target
    ):
        minimum = None
    size = sizing_owner.UniformSizeControl(
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
        maximum_deviation=specification.maximum_deviation,
        deterministic=specification.deterministic,
    )


def _compose_polyhedral_layer_topology(
    layers: sources_owner.NativeLayerCoreSource,
    core: polyhedral_generation_owner.PolyhedralConstruction,
    mapping: np.ndarray,
    initial_points: np.ndarray,
    input_vertices: np.ndarray,
    source_revision: str,
    /,
) -> tuple[CellMesh, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compose exact polyhedral core loops beside immutable standard layer blocks."""
    core_mesh = core.mesh
    connectivity = core_mesh.connectivity
    if not isinstance(connectivity, cell_complex_owner.PolyhedralConnectivity):
        raise TypeError("Native restricted-power core lost polyhedral connectivity.")
    core_points = np.asarray(core_mesh.coordinates, dtype=np.float64)
    vertex_carriers = np.asarray(core.restricted_cells.vertex_carriers, dtype=np.int64)
    carrier_ids = np.asarray(core.carrier.mesh.vertex_global_ids, dtype=np.int64)
    if (
        vertex_carriers.shape != (core_points.shape[0], 4)
        or input_vertices.shape != carrier_ids.shape
        or mapping.shape != (layers.complex.vertices.shape[0],)
        or np.any(mapping < 0)
    ):
        raise ValueError(
            "Polyhedral source-carrier ancestry does not align with the PLC."
        )
    carrier_rows = {int(identifier): row for row, identifier in enumerate(carrier_ids)}
    source_vertex_rows = np.full((layers.complex.vertices.shape[0],), -1, dtype=np.int64)
    core_vertex_map = np.full((core_points.shape[0],), -1, dtype=np.int64)
    for core_row in range(vertex_carriers.shape[0]):
        carriers = vertex_carriers[core_row]
        active = carriers[carriers >= 0]
        if active.size != 1:
            continue
        carrier_row = carrier_rows.get(int(active[0]))
        if carrier_row is None:
            raise ValueError("A power vertex names an absent fixed-carrier vertex.")
        source_vertex = int(input_vertices[carrier_row])
        if source_vertex < 0:
            continue
        if source_vertex_rows[source_vertex] >= 0:
            raise ValueError("A fixed PLC vertex has duplicate power-cell descendants.")
        source_vertex_rows[source_vertex] = core_row
        core_vertex_map[core_row] = mapping[source_vertex]
    if np.any(source_vertex_rows < 0):
        raise ValueError("Restricted-power core omitted a fixed PLC source vertex.")
    layer_mesh = layers.layers.mesh
    generated = np.flatnonzero(core_vertex_map < 0)
    core_vertex_map[generated] = initial_points.shape[0] + np.arange(
        generated.size, dtype=np.int64
    )
    points = np.concatenate((initial_points, core_points[generated]))
    layer_cells = np.sort(np.asarray(layer_mesh.entity_set(3).entity_ids, dtype=np.int64))
    core_cells = np.sort(np.asarray(core_mesh.entity_set(3).entity_ids, dtype=np.int64))
    core_start = int(layer_cells[-1]) + 1
    if core_cells.size and core_start + core_cells.size - 1 > np.iinfo(np.int64).max:
        raise ValueError(
            "The accepted layer cell identity range cannot contain core cells."
        )
    initial_extra = initial_points.shape[0] - layer_mesh.coordinates.shape[0]
    if initial_extra < 0:
        raise ValueError("Layer/core identity preparation lost accepted layer vertices.")
    next_vertex = (
        int(np.max(np.asarray(layer_mesh.vertex_global_ids, dtype=np.int64))) + 1
        if layer_mesh.vertex_global_ids.size
        else 0
    )
    extra_vertices = initial_extra + generated.size
    if extra_vertices and next_vertex + extra_vertices - 1 > np.iinfo(np.int64).max:
        raise ValueError(
            "The accepted layer vertex identity range cannot contain core vertices."
        )
    vertex_ids = np.concatenate(
        (
            np.asarray(layer_mesh.vertex_global_ids, dtype=np.int64),
            next_vertex + np.arange(extra_vertices, dtype=np.int64),
        )
    )
    standard_blocks = tuple(
        CellBlock(
            f"layer:{block.name}",
            block.cell_kind,
            np.asarray(block.vertices, dtype=np.int64),
            global_ids=np.asarray(block.global_ids, dtype=np.int64),
        )
        for block in layer_mesh.blocks
    )
    core_cell_ids = core_start + np.arange(core_cells.size, dtype=np.int64)
    cell_rows = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
    oriented_loops = layer_topology.polyhedral_cell_faces(core_mesh)
    loops_by_id = {
        int(identifier): oriented_loops[row]
        for row, identifier in enumerate(cell_rows.tolist())
    }
    polyhedral_cells = tuple(
        tuple(
            np.asarray([core_vertex_map[vertex] for vertex in loop], dtype=np.int64)
            for loop in loops_by_id[int(identifier)]
        )
        for identifier in core_cells.tolist()
    )
    if len(core_mesh.blocks) == 0 or any(
        block.cell_kind != "polyhedron" for block in core_mesh.blocks
    ):
        raise TypeError("Restricted-power core blocks must all be polyhedral.")
    mesh = CellMesh.from_mixed_3d(
        points,
        standard_blocks,
        polyhedra={"core:power": polyhedral_cells},
        vertex_global_ids=vertex_ids,
        polyhedral_cell_global_ids={"core:power": core_cell_ids},
        numeric_version=source_revision,
    )
    layer_regions = np.asarray(layers.layer_regions, dtype=np.int64)
    core_region_by_id = {
        int(identifier): int(region)
        for identifier, region in zip(
            np.asarray(core_mesh.entity_set(3).entity_ids, dtype=np.int64),
            np.asarray(core.cell_regions, dtype=np.int64),
            strict=True,
        )
    }
    core_regions = np.asarray(
        [core_region_by_id[int(identifier)] for identifier in core_cells],
        dtype=np.int64,
    )
    region_map = np.asarray(layers.core_region_map, dtype=np.int64)
    combined_regions = np.concatenate((layer_regions, region_map[core_regions]))
    if (
        combined_regions.shape
        != np.sort(np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)).shape
    ):
        raise ValueError("Layer and core region records do not cover the combined mesh.")
    return mesh, core_vertex_map, combined_regions, layer_cells, core_cell_ids


def _polyhedral_periodic_ancestry(
    source: sources_owner.NativeLayerCoreSource,
    core: polyhedral_generation_owner.PolyhedralConstruction,
    mapping: np.ndarray,
    initial_points: np.ndarray,
    input_vertices: np.ndarray,
    core_vertex_map: np.ndarray,
    source_ancestry: tuple[np.ndarray, np.ndarray] | None,
    /,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Align the restricted-power quotient with authored layer/core seam winding."""
    core_periodic = core.mesh.periodic_topology
    if source_ancestry is None:
        if core_periodic is not None:
            raise ValueError(
                "Polyhedral core introduced periodic topology without layer ancestry."
            )
        return None
    layer_periodic = source.layers.mesh.periodic_topology
    if layer_periodic is None or core_periodic is None:
        raise ValueError(
            "Periodic polyhedral layer/core fill requires exact layer and core quotient topology."
        )
    base_roots, base_shifts = source_ancestry
    point_count = max(
        initial_points.shape[0], int(np.max(core_vertex_map, initial=-1)) + 1
    )
    if base_roots.shape != (initial_points.shape[0],) or base_shifts.shape != (
        initial_points.shape[0],
        layer_periodic.cell.rank,
    ):
        raise ValueError(
            "Authored layer/core periodic ancestry has incompatible extents."
        )
    if periodic_exact_owner.identification_id(
        core_periodic.cell
    ) != periodic_exact_owner.identification_id(layer_periodic.cell):
        raise ValueError(
            "Polyhedral core and accepted layers use different periodic cells."
        )
    roots = np.arange(point_count, dtype=np.int64)
    shifts = np.zeros((point_count, layer_periodic.cell.rank), dtype=np.int64)
    roots[: base_roots.size] = base_roots
    shifts[: base_shifts.shape[0]] = base_shifts
    connectivity = core.mesh.connectivity
    if not isinstance(connectivity, cell_complex_owner.PolyhedralConnectivity):
        raise TypeError("Periodic power core must retain polyhedral connectivity.")
    carrier_ids = np.asarray(core.carrier.mesh.vertex_global_ids, dtype=np.int64)
    carrier_rows = {int(identifier): row for row, identifier in enumerate(carrier_ids)}
    source_vertex_rows = np.full((source.complex.vertices.shape[0],), -1, dtype=np.int64)
    periodic_carriers = np.asarray(core.restricted_cells.vertex_carriers, dtype=np.int64)
    for core_row in range(periodic_carriers.shape[0]):
        carriers = periodic_carriers[core_row]
        active = carriers[carriers >= 0]
        if active.size != 1:
            continue
        carrier_row = carrier_rows.get(int(active[0]))
        if carrier_row is None:
            raise ValueError(
                "A periodic power vertex names an absent PLC carrier vertex."
            )
        source_vertex = int(input_vertices[carrier_row])
        if source_vertex >= 0:
            if source_vertex_rows[source_vertex] >= 0:
                raise ValueError("A periodic PLC vertex has duplicate power descendants.")
            source_vertex_rows[source_vertex] = core_row
    if np.any(source_vertex_rows < 0):
        raise ValueError("Periodic power core omitted an authored PLC vertex.")
    core_roots = np.asarray(core_periodic.vertex_representatives, dtype=np.int64)
    core_shifts = np.asarray(core_periodic.vertex_shifts, dtype=np.int64)
    if (
        core_roots.shape != core_vertex_map.shape
        or core_shifts.shape != (core_vertex_map.size, layer_periodic.cell.rank)
        or np.any(core_roots < 0)
        or np.any(core_roots >= core_vertex_map.size)
    ):
        raise ValueError("Periodic power-core quotient arrays do not match its vertices.")
    orders = periodic_exact_owner.identification_orders(layer_periodic.cell)
    offsets: dict[int, np.ndarray] = {}
    for source_vertex, core_row in enumerate(source_vertex_rows.tolist()):
        base_row = int(mapping[source_vertex])
        core_root = int(core_roots[core_row])
        target_root = int(core_vertex_map[core_root])
        if not 0 <= base_row < base_roots.size or not 0 <= target_root < base_roots.size:
            raise ValueError(
                "A periodic PLC seam root is absent from the authored layer bank."
            )
        if base_roots[base_row] != base_roots[target_root]:
            raise ValueError("Power-core periodic roots contradict authored layer seams.")
        offset = periodic_exact_owner.reduce_periodic_exponents(
            base_shifts[base_row] - core_shifts[core_row], orders
        )
        previous = offsets.get(core_root)
        if previous is not None and not np.array_equal(previous, offset):
            raise ValueError(
                "Power-core periodic winding contradicts authored source ancestry."
            )
        offsets[core_root] = offset
    for core_row, core_root_value in enumerate(core_roots.tolist()):
        target_row = int(core_vertex_map[core_row])
        target_root = int(core_vertex_map[int(core_root_value)])
        roots[target_row] = roots[target_root]
        offset = offsets.get(
            int(core_root_value), np.zeros((layer_periodic.cell.rank,), dtype=np.int64)
        )
        shifts[target_row] = periodic_exact_owner.reduce_periodic_exponents(
            core_shifts[core_row] + offset, orders
        )
    for source_vertex, core_row in enumerate(source_vertex_rows.tolist()):
        target_row = int(core_vertex_map[core_row])
        base_row = int(mapping[source_vertex])
        if roots[target_row] != base_roots[base_row] or not np.array_equal(
            shifts[target_row], base_shifts[base_row]
        ):
            raise ValueError("Power-core quotient changed an authored PLC vertex orbit.")
    return roots, shifts


def _compose_layer_topology(
    layers: sources_owner.NativeLayerCoreSource,
    core: (
        volume_generation_owner.VolumeConstruction
        | polyhedral_generation_owner.PolyhedralConstruction
    ),
    mapping: np.ndarray,
    initial_points: np.ndarray,
    input_vertices: np.ndarray,
    source_revision: str,
    /,
) -> tuple[CellMesh, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep accepted layer topology and PLC identities, appending only native interior nodes."""
    if isinstance(core, polyhedral_generation_owner.PolyhedralConstruction):
        return _compose_polyhedral_layer_topology(
            layers,
            core,
            mapping,
            initial_points,
            input_vertices,
            source_revision,
        )
    core_points = np.asarray(core.mesh.coordinates, dtype=np.float64)
    generated = np.flatnonzero(input_vertices < 0)
    core_vertex_map = np.empty(core_points.shape[0], dtype=np.int64)
    retained = input_vertices >= 0
    core_vertex_map[retained] = mapping[input_vertices[retained]]
    core_vertex_map[generated] = initial_points.shape[0] + np.arange(
        generated.size, dtype=np.int64
    )
    points = np.concatenate((initial_points, core_points[generated]))
    layer_mesh = layers.layers.mesh
    layer_cells = np.sort(np.asarray(layer_mesh.entity_set(3).entity_ids, dtype=np.int64))
    core_cells = np.sort(np.asarray(core.mesh.entity_set(3).entity_ids, dtype=np.int64))
    core_start = int(layer_cells[-1]) + 1
    if core_cells.size and core_start + core_cells.size - 1 > np.iinfo(np.int64).max:
        raise ValueError(
            "The accepted layer cell identity range cannot contain core cells."
        )
    initial_extra = initial_points.shape[0] - layer_mesh.coordinates.shape[0]
    if initial_extra < 0:
        raise ValueError("Layer/core identity preparation lost accepted layer vertices.")
    if layer_mesh.vertex_global_ids.size:
        next_vertex = (
            int(np.max(np.asarray(layer_mesh.vertex_global_ids, dtype=np.int64))) + 1
        )
    else:
        next_vertex = 0
    extra_vertices = initial_extra + generated.size
    if extra_vertices and next_vertex + extra_vertices - 1 > np.iinfo(np.int64).max:
        raise ValueError(
            "The accepted layer vertex identity range cannot contain core vertices."
        )
    vertex_ids = np.concatenate(
        (
            np.asarray(layer_mesh.vertex_global_ids, dtype=np.int64),
            next_vertex + np.arange(extra_vertices, dtype=np.int64),
        )
    )
    blocks = [
        CellBlock(
            f"layer:{block.name}",
            block.cell_kind,
            np.asarray(block.vertices, dtype=np.int64),
            global_ids=np.asarray(block.global_ids, dtype=np.int64),
        )
        for block in layer_mesh.blocks
    ]
    core_cell_ids = core_start + np.arange(core_cells.size, dtype=np.int64)
    for block in core.mesh.blocks:
        old_ids = np.asarray(block.global_ids, dtype=np.int64)
        positions = np.searchsorted(core_cells, old_ids)
        blocks.append(
            CellBlock(
                f"core:{block.name}",
                block.cell_kind,
                core_vertex_map[np.asarray(block.vertices, dtype=np.int64)],
                global_ids=core_cell_ids[positions],
            )
        )
    mesh = CellMesh(
        points,
        tuple(blocks),
        vertex_global_ids=vertex_ids,
        numeric_version=source_revision,
    )
    layer_regions = np.asarray(layers.layer_regions, dtype=np.int64)
    core_map = np.asarray(layers.core_region_map, dtype=np.int64)
    combined_regions = np.concatenate(
        (layer_regions, core_map[np.asarray(core.cell_regions, dtype=np.int64)])
    )
    if combined_regions.shape != np.sort(np.asarray(mesh.entity_set(3).entity_ids)).shape:
        raise ValueError("Layer and core region records do not cover the combined mesh.")
    return mesh, core_vertex_map, combined_regions, layer_cells, core_cell_ids


def _compose_organization(
    mesh: CellMesh,
    regions: np.ndarray,
    layer_ids: np.ndarray,
    core_ids: np.ndarray,
    layers: sources_owner.NativeLayerCoreSource,
    complex_: plc_source_owner.PiecewiseLinearComplex,
    mapping: np.ndarray,
    cap_ids: np.ndarray,
    polygons: np.ndarray,
    layer_faces: dict[tuple[int, ...], list[layer_topology.LayerCoreFace]],
    combined_faces: dict[tuple[int, ...], list[layer_topology.LayerCoreFace]],
    region_ids: tuple[str, ...],
    /,
    *,
    polyhedral_core: polyhedral_generation_owner.PolyhedralConstruction | None = None,
    core_vertex_map: np.ndarray | None = None,
) -> tuple[
    tuple[organization_owner.MeshZone, ...],
    tuple[organization_owner.MeshPatch, ...],
    tuple[organization_owner.MeshLabel, ...],
    tuple[organization_owner.MeshAttribute, ...],
]:
    """Bind region zones, exact core patches, layer intervals and combined interfaces."""
    cells = np.sort(np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64))
    order = np.argsort(regions, kind="stable")
    bounds = np.searchsorted(
        regions[order], np.arange(len(region_ids) + 1, dtype=np.int64)
    )
    zones = tuple(
        organization_owner.MeshZone(
            name,
            organization_owner.MeshZoneRole.REGION,
            layer_topology.mesh_entity_scope(mesh, 3, cells[order[start:stop]]),
        )
        for name, start, stop in zip(region_ids, bounds[:-1], bounds[1:], strict=True)
        if stop > start
    )
    labels = (
        organization_owner.MeshLabel(
            "boundary-layer", layer_topology.mesh_entity_scope(mesh, 3, layer_ids)
        ),
        organization_owner.MeshLabel(
            "core", layer_topology.mesh_entity_scope(mesh, 3, core_ids)
        ),
    )
    layer_by_id = dict(
        zip(
            np.sort(
                np.asarray(layers.layers.mesh.entity_set(3).entity_ids, dtype=np.int64)
            ).tolist(),
            np.asarray(layers.layers.layer_index, dtype=np.int64).tolist(),
            strict=True,
        )
    )
    layer_values = np.asarray(
        [layer_by_id[int(identifier)] for identifier in layer_ids], dtype=np.int64
    )
    column_values = np.asarray(layers.layers.column_index, dtype=np.int64)
    if column_values.shape != layer_ids.shape:
        raise ValueError(
            "Source column identities must identify each accepted layer cell."
        )
    attributes = (
        organization_owner.MeshAttribute(
            "layer_index",
            organization_owner.MeshAttributeRole.MARKER,
            layer_topology.mesh_entity_scope(mesh, 3, cells),
            np.concatenate((layer_values, np.full(core_ids.size, -1, dtype=np.int64))),
        ),
        organization_owner.MeshAttribute(
            "layer_column",
            organization_owner.MeshAttributeRole.MARKER,
            layer_topology.mesh_entity_scope(mesh, 3, cells),
            np.concatenate((column_values, np.full(core_ids.size, -1, dtype=np.int64))),
        ),
    )
    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    face_keys = dict(
        zip(canonical_owner._entity_vertex_keys(mesh, 2), face_ids.tolist(), strict=True)
    )
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)

    def face_id(row_vertices: tuple[int, ...], /) -> int:
        vertices = np.asarray(row_vertices, dtype=np.int64)
        key = tuple(sorted(int(value) for value in vertex_ids[vertices]))
        return face_keys[key]

    faces_by_facet: dict[int, set[int]] = {
        facet: set() for facet in range(complex_.facet_count)
    }
    if polyhedral_core is None:
        for polygon, facet_value in zip(polygons, complex_.polygon_facets, strict=True):
            facet = int(facet_value)
            faces_by_facet[facet].add(face_id(tuple(mapping[polygon].tolist())))
    else:
        if core_vertex_map is None:
            raise ValueError(
                "Polyhedral organization requires exact core vertex ancestry."
            )
        connectivity = polyhedral_core.mesh.connectivity
        if not isinstance(connectivity, cell_complex_owner.PolyhedralConnectivity):
            raise TypeError(
                "Polyhedral organization requires canonical face connectivity."
            )
        offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
        values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
        source_facets = np.asarray(polyhedral_core.face_source_facets, dtype=np.int64)
        if source_facets.shape != (connectivity.face_count,):
            raise ValueError(
                "Power-core face ancestry does not align with its connectivity."
            )
        for face, facet_value in enumerate(source_facets.tolist()):
            if facet_value < 0:
                continue
            first, last = offsets[face : face + 2]
            loop = tuple(
                int(value) for value in core_vertex_map[values[first:last]].tolist()
            )
            faces_by_facet[facet_value].add(face_id(loop))
    cap_keys = {tuple(sorted(mapping[polygons[index]].tolist())) for index in cap_ids}
    # An open wall also exposes layer side strips. The wall patch is only the
    # exterior layer faces lying wholly on the wall; a strip rising from rim
    # edge (a, b) joins the facet of the fixed PLC polygon that holds the tops
    # of the columns grown from a and b.
    layer_count = int(np.asarray(layers.layers.mesh.coordinates).shape[0])
    column_sources = np.asarray(layers.layers.vertex_source_vertex_ids, dtype=np.int64)
    wall_vertices = np.asarray(layers.layers.wall_vertices, dtype=np.int64)
    on_wall = frozenset(wall_vertices[wall_vertices >= 0].tolist())
    cap_polygons = frozenset(np.asarray(cap_ids, dtype=np.int64).tolist())
    rim_facets: dict[tuple[int, int], set[int]] = {}
    for index, (polygon, facet_value) in enumerate(
        zip(polygons, complex_.polygon_facets, strict=True)
    ):
        if index in cap_polygons:
            continue
        rows = mapping[polygon].tolist()
        for corner, row in enumerate(rows):
            following = rows[(corner + 1) % len(rows)]
            if row >= layer_count or following >= layer_count:
                continue
            ends = (int(column_sources[row]), int(column_sources[following]))
            if min(ends) >= 0 and ends[0] != ends[1]:
                rim_facets.setdefault((min(ends), max(ends)), set()).add(int(facet_value))
    wall_rows: list[int] = []
    side_rows: list[int] = []
    for key, incidents in layer_faces.items():
        if len(incidents) != 1 or key in cap_keys:
            continue
        if on_wall.issuperset(key):
            wall_rows.append(face_id(key))
            continue
        ends = sorted({int(column_sources[row]) for row in key})
        facets = rim_facets.get((ends[0], ends[-1]), set()) if len(ends) == 2 else set()
        if len(facets) == 1 and ends[0] >= 0:
            faces_by_facet[next(iter(facets))].add(face_id(key))
        else:
            side_rows.append(face_id(key))
    patches: list[organization_owner.MeshPatch] = [
        organization_owner.MeshPatch(
            "facet:"
            + source_identity_owner.source_entity_id(
                layers.source_id,
                layers.source_revision,
                "face",
                complex_.facet_ids[facet],
                occurrence_path=complex_.facet_occurrence_paths[facet],
            ),
            layer_topology.mesh_entity_scope(
                mesh, 2, np.asarray(sorted(face_rows), dtype=np.int64)
            ),
            connected=False,
        )
        for facet, face_rows in sorted(faces_by_facet.items())
        if face_rows
    ]
    if cap_keys:
        patches.append(
            organization_owner.MeshPatch(
                "layer-core-interface",
                layer_topology.mesh_entity_scope(
                    mesh,
                    2,
                    np.asarray(
                        [face_id(key) for key in sorted(cap_keys)], dtype=np.int64
                    ),
                ),
                connected=False,
            )
        )
    for name, rows in (("wall", wall_rows), ("layer-side", side_rows)):
        if rows:
            patches.append(
                organization_owner.MeshPatch(
                    name,
                    layer_topology.mesh_entity_scope(
                        mesh, 2, np.asarray(rows, dtype=np.int64)
                    ),
                    connected=False,
                )
            )
    interface_ids = np.asarray(
        [
            face_id(key)
            for key, incidents in combined_faces.items()
            if len(incidents) == 2 and incidents[0].region != incidents[1].region
        ],
        dtype=np.int64,
    )
    if interface_ids.size:
        labels += (
            organization_owner.MeshLabel(
                "interface", layer_topology.mesh_entity_scope(mesh, 2, interface_ids)
            ),
        )
    return zones, tuple(patches), labels, attributes


@checked
def _generate_layer_core_unmapped(
    source: sources_owner.NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    core_specification: VolumeMeshingSpec,
    schedule: volume_generation_owner.NativeVolumeSchedule,
    /,
    *,
    frame_id: str,
    source_domain: mapped_domain_owner.MappedReferenceDomain | None = None,
    polyhedral_schedule: polyhedral_generation_owner.NativePolyhedralSchedule
    | None = None,
) -> layer_topology.LayerCoreConstruction:
    """Fill the remaining fixed PLC and return the complete combined construction."""
    layers = source.layers
    complex_ = source.complex
    layer_count = sum(block.cell_count for block in layers.mesh.blocks)
    layer_regions = np.asarray(source.layer_regions, dtype=np.int64)
    if layer_regions.shape != (layer_count,):
        raise ValueError("Native layer regions must identify every accepted layer cell.")
    if any(
        controls_owner.realized_layer_control(source, control).control_id
        != layers.control_id
        for control in specification.layer_controls
    ):
        raise ValueError(
            "Layer controls must describe the accepted immutable layer mesh."
        )
    region_ids, core_region_map = regions_owner.prepare_layer_region_identity(
        complex_, layer_regions, source.region_ids, source.core_region_map
    )
    power_request: (
        tuple[
            sources_owner.NativePolyhedralSource,
            polyhedral_generation_owner.NativePolyhedralSchedule,
        ]
        | None
    ) = None
    if core_specification.fill_strategy is contracts_owner.VolumeFillStrategy.POLYHEDRAL:
        if source_domain is not None:
            raise contracts_owner.MeshingFailure(
                contracts_owner.MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "Mapped polyhedral layer/core composition requires a certified mixed-cell root map.",
                provider_code="native_layer_core_polyhedral",
                stage="route_execution",
            )
        if polyhedral_schedule is None:
            raise contracts_owner.MeshingFailure(
                contracts_owner.MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "Polyhedral layer/core fill requires an explicit power schedule.",
                provider_code="native_layer_core_polyhedral",
                stage="route_execution",
            )
        power_source = source.core_power_source
        if power_source is None:
            raise contracts_owner.MeshingFailure(
                contracts_owner.MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
                "Polyhedral layer/core fill requires an authored core power source.",
                provider_code="native_layer_core_polyhedral",
                stage="route_execution",
            )
        power_request = (
            power_source,
            polyhedral_generation_owner.effective_polyhedral_schedule(
                core_specification, polyhedral_schedule
            ),
        )
    mapping, cap_ids, polygons, initial_points = layer_topology.prepare_identity(
        layers, complex_, source.vertex_layer_ids, source.cap_polygon_ids
    )
    if power_request is not None:
        unmapped_vertices = mapping < 0
        if np.any(unmapped_vertices):
            mapping = mapping.copy()
            mapping[unmapped_vertices] = layers.mesh.coordinates.shape[0] + np.arange(
                np.count_nonzero(unmapped_vertices), dtype=np.int64
            )
    periodic_ancestry = periodic_owner.prepare_core_periodic(
        source, mapping, polygons, initial_points
    )
    periodic_requested, periodic_achieved = (
        periodic_owner.core_periodic_constraint_evidence(source, specification)
    )
    coordinate_map_domain = (
        source_domain if source_domain is not None else source.mapped_domain
    )
    mapped_geometry = (
        None if coordinate_map_domain is None else coordinate_map_domain.source_geometry
    )
    source_map_id = (
        ""
        if mapped_geometry is None or mapped_geometry.source_map_id is None
        else mapped_geometry.source_map_id
    )
    polyhedral_core = None
    if power_request is not None:
        power_source, power_schedule = power_request
        polyhedral_core = polyhedral_generation_owner.generate_polyhedral_volume(
            power_source,
            core_specification,
            power_schedule,
            frame_id=frame_id,
            volume_schedule=schedule,
        )
        carrier_core = polyhedral_core.carrier
        construction_core: (
            volume_generation_owner.VolumeConstruction
            | polyhedral_generation_owner.PolyhedralConstruction
        ) = polyhedral_core
    else:
        carrier_core = volume_generation_owner.generate_plc_volume(
            complex_,
            core_specification,
            schedule,
            source_id=source.source_id,
            source_revision=source.source_revision,
            source_map_id=source_map_id,
            frame_id=frame_id,
            source_domain=source_domain,
        )
        construction_core = carrier_core
    input_vertices = _require_fixed(complex_, carrier_core)
    mesh, core_vertex_map, combined_regions, layer_ids, core_cell_ids = (
        _compose_layer_topology(
            source,
            construction_core,
            mapping,
            initial_points,
            input_vertices,
            source.source_revision,
        )
    )
    if polyhedral_core is not None:
        periodic_ancestry = _polyhedral_periodic_ancestry(
            source,
            polyhedral_core,
            mapping,
            initial_points,
            input_vertices,
            core_vertex_map,
            periodic_ancestry,
        )
    mesh = canonical_owner.canonicalize_cell_mesh(mesh)
    periodic_topology = periodic_owner.bind_core_periodic(source, mesh, periodic_ancestry)
    if periodic_topology is not None:
        mesh = CellMesh(
            mesh.coordinates,
            mesh.blocks,
            vertex_global_ids=mesh.vertex_global_ids,
            entity_global_ids={
                degree: np.asarray(mesh.entity_set(degree).entity_ids, dtype=np.int64)
                for degree in range(1, mesh.topological_dimension)
            },
            numeric_version=mesh.numeric_version,
            periodic_topology=periodic_topology,
        )
    domain, layer_faces = layer_topology.combined_domain(
        layers,
        complex_,
        layer_regions,
        mapping,
        cap_ids,
        polygons,
        np.asarray(mesh.coordinates, dtype=np.float64)[: initial_points.shape[0]],
        source.source_id,
        region_ids,
        core_region_map,
    )
    combined_faces = layer_topology.faces(mesh, combined_regions)
    layer_face_keys = set(layer_faces)
    for index in cap_ids:
        key = tuple(sorted(mapping[polygons[index]].tolist()))
        incidents = combined_faces.get(key, [])
        if key not in layer_face_keys or len(incidents) != 2:
            raise ValueError(
                "The fixed layer cap must be shared by one layer and one core cell."
            )
        incident_ids = {
            int(np.sort(np.asarray(mesh.entity_set(3).entity_ids))[face.cell])
            for face in incidents
        }
        if not incident_ids.intersection(
            set(layer_ids.tolist())
        ) or not incident_ids.intersection(set(core_cell_ids.tolist())):
            raise ValueError(
                "A fixed cap facet is not shared across the layer/core transition."
            )
    zones, patches, labels, attributes = _compose_organization(
        mesh,
        combined_regions,
        layer_ids,
        core_cell_ids,
        source,
        complex_,
        mapping,
        cap_ids,
        polygons,
        layer_faces,
        combined_faces,
        region_ids,
        polyhedral_core=polyhedral_core,
        core_vertex_map=core_vertex_map,
    )
    associations = (
        *association_composition.layer_wall_associations(source, mesh),
        *association_composition.layer_source_associations(
            layers, mesh, layer_regions, region_ids
        ),
        *association_composition.core_source_associations(
            source,
            polyhedral_core if polyhedral_core is not None else carrier_core,
            mesh,
            core_vertex_map,
            combined_regions,
            region_ids,
            core_cell_ids,
        ),
    )
    preliminary = layer_topology.LayerCoreConstruction(
        mesh,
        domain,
        combined_regions,
        zones,
        patches,
        labels,
        attributes,
        associations,
        cell_geometry_owner.CellGeometrySpec.affine(mesh),
        carrier_core,
        core_vertex_map,
        core_cell_ids,
        region_ids,
        source.source_id,
        source.source_revision,
        periodic_topology,
        periodic_requested,
        periodic_achieved,
        (),
        None,
        None,
        polyhedral_core=polyhedral_core,
    )
    zones, patches = controls_owner.compose_layer_controls(
        source, specification, preliminary
    )
    return layer_topology.LayerCoreConstruction(
        mesh,
        domain,
        combined_regions,
        zones,
        patches,
        labels,
        attributes,
        associations,
        cell_geometry_owner.CellGeometrySpec.affine(mesh),
        carrier_core,
        core_vertex_map,
        core_cell_ids,
        region_ids,
        source.source_id,
        source.source_revision,
        periodic_topology,
        periodic_requested,
        periodic_achieved,
        (),
        None,
        None,
        polyhedral_core=polyhedral_core,
    )


@checked
def generate_layer_core(
    source: sources_owner.NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    core_specification: VolumeMeshingSpec,
    schedule: volume_generation_owner.NativeVolumeSchedule,
    /,
    *,
    frame_id: str,
    polyhedral_schedule: polyhedral_generation_owner.NativePolyhedralSchedule
    | None = None,
) -> layer_topology.LayerCoreConstruction:
    """Fill a fixed PLC directly or compose its reference fill through exact source roots."""
    reference = source.reference_source
    if reference is None:
        if source.mapped_domain is not None:
            raise ValueError(
                "A mapped layer/core source must retain its exact reference source."
            )
        return _generate_layer_core_unmapped(
            source,
            specification,
            core_specification,
            schedule,
            frame_id=frame_id,
            polyhedral_schedule=polyhedral_schedule,
        )
    domain = source.mapped_domain
    if not isinstance(domain, mapped_domain_owner.MappedReferenceDomain):
        raise TypeError("Mapped layer/core execution requires MappedReferenceDomain.")
    if (
        reference.reference_source is not None
        or reference.mapped_domain is not None
        or (reference.source_id, reference.source_revision)
        != (source.source_id, source.source_revision)
        or (domain.source_id, domain.source_revision)
        != (source.source_id, source.source_revision)
        or domain.region_ids != reference.complex.region_ids
        or source.complex.region_ids != reference.complex.region_ids
        or source.complex.vertex_ids != reference.complex.vertex_ids
        or source.complex.vertex_occurrence_paths
        != reference.complex.vertex_occurrence_paths
        or source.complex.segment_ids != reference.complex.segment_ids
        or source.complex.segment_occurrence_paths
        != reference.complex.segment_occurrence_paths
        or source.complex.facet_ids != reference.complex.facet_ids
        or source.complex.facet_occurrence_paths
        != reference.complex.facet_occurrence_paths
        or source.complex.region_ids != reference.complex.region_ids
        or source.complex.region_entity_ids != reference.complex.region_entity_ids
        or source.complex.region_occurrence_paths
        != reference.complex.region_occurrence_paths
        or source.region_ids != reference.region_ids
        or source.region_entity_ids.shape != reference.region_entity_ids.shape
        or not np.array_equal(source.region_entity_ids, reference.region_entity_ids)
        or source.region_occurrence_paths != reference.region_occurrence_paths
    ):
        raise ValueError(
            "Mapped layer/core source and reference PLC must retain the same authored identities, occurrences, and regions."
        )
    reference_specification = VolumeMeshingSpec(
        specification.target,
        specification.boundary_scope,
        specification.fill_strategy,
        size_controls=specification.size_controls,
        protected_features=(),
        region_controls=(),
        patch_controls=(),
        region_seeds=specification.region_seeds,
        hole_seeds=specification.hole_seeds,
        layer_controls=(),
        periodic_constraints=specification.periodic_constraints,
        size_combination=specification.size_combination,
        size_compliance=specification.size_compliance,
        maximum_deviation=specification.maximum_deviation,
        deterministic=specification.deterministic,
    )
    physical_scale = mapped_core_owner.mapped_source_lipschitz_bound(domain)
    reference_core_specification = core_request(
        core_specification,
        specification.layer_controls,
        physical_scale=physical_scale,
    )
    reference_construction = _generate_layer_core_unmapped(
        reference,
        reference_specification,
        reference_core_specification,
        schedule,
        source_domain=domain,
        frame_id=frame_id,
        polyhedral_schedule=polyhedral_schedule,
    )
    mapped = mapped_core_owner.compose_mapped_layer_core(source, reference_construction)
    zones, patches = controls_owner.compose_layer_controls(source, specification, mapped)
    return layer_topology.LayerCoreConstruction(
        mapped.mesh,
        mapped.domain,
        mapped.cell_regions,
        zones,
        patches,
        mapped.labels,
        mapped.attributes,
        mapped.associations,
        mapped.geometry,
        mapped.core,
        mapped.core_vertex_map,
        mapped.core_cell_ids,
        mapped.region_ids,
        mapped.source_id,
        mapped.source_revision,
        mapped.periodic_topology,
        mapped.periodic_requested,
        mapped.periodic_achieved,
        mapped.reference_maps,
        mapped.source_mesh,
        mapped.source_geometry,
        reference_mesh=mapped.reference_mesh,
        reference_geometry=mapped.reference_geometry,
        polyhedral_core=reference_construction.polyhedral_core,
    )


__all__ = ["core_request", "generate_layer_core"]
