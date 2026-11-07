"""Author a compact affine box from a selected wall and one genuine layer."""

from typing import Any, cast

import numpy as np

import phydrax as phx
from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain
from phydrax.meshing import _layer_core_topology as topology
from phydrax.meshing._layer_core_association_transfer import (
    NativeLayerCoreAssociationTransfer,
)
from phydrax.meshing._plc_source import PiecewiseLinearComplex
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.meshing.providers._native_sources import (
    NativeLayerCoreSource,
    NativePlcSource,
)
from tests.unit.meshing.test_periodic_layer_core_source import _periodic_wall_source
from tools.meshing_qualification.scenarios import native_hybrid_profiles as s


def tiny_affine_mapped_lifecycle() -> tuple[Any, Any, Any, Any, Any]:
    wall, wall_domain, _, _ = _periodic_wall_source()
    source_id = wall_domain.source_id
    revision = wall_domain.source_revision
    control = phx.meshing.BoundaryLayerControl(
        phx.meshing.MeshingScope.whole(wall, 2),
        phx.meshing.LayerSchedule((0.1,)),
        route=phx.meshing.BoundaryLayerRoute.ADVANCING,
    )
    policy = phx.meshing.BoundaryLayerPolicy()
    layers = phx.meshing.prepare_boundary_layers(wall, control, policy=policy)
    cap = layers.cap
    if cap is None:
        raise ValueError("Tiny layer source lost its cap")
    cap_rows = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in cap.blocks]
    )
    cap_layer = np.asarray(layers.cap_vertices, dtype=np.int64)
    source_ids = np.asarray(layers.vertex_source_vertex_ids, dtype=np.int64)[cap_layer]
    cap_by_source = {int(key): row for row, key in enumerate(source_ids)}
    if set(cap_by_source) != {20, 21, 22, 23}:
        raise ValueError("Tiny layer changed its authored wall ancestry")
    c0, c1, c2, c3 = (cap_by_source[k] for k in (20, 21, 22, 23))
    n = len(cap_layer)
    t0, t1, t2, t3 = (n + cap_by_source[k] for k in (20, 21, 22, 23))
    polygons = (
        *tuple(row for row in cap_rows),
        *tuple(row for row in cap_rows[:, ::-1] + n),
        np.asarray((c0, t2, t0)),
        np.asarray((c0, c2, t2)),
        np.asarray((c1, t1, t3)),
        np.asarray((c1, t3, c3)),
        np.asarray((c0, t1, c1)),
        np.asarray((c0, t0, t1)),
        np.asarray((c2, t3, t2)),
        np.asarray((c2, c3, t3)),
    )
    core_vertices = np.concatenate(
        (
            np.asarray(cap.coordinates, dtype=np.float64),
            np.asarray(cap.coordinates, dtype=np.float64) + np.asarray((0.0, 0.0, 0.8)),
        )
    )
    core = PiecewiseLinearComplex(
        core_vertices,
        polygons,
        np.arange(len(polygons), dtype=np.int64),
        np.tile(np.asarray((0, -1), dtype=np.int64), (len(polygons), 1)),
        ("solid",),
        vertex_ids=tuple(f"tiny-core-v{i}" for i in range(len(core_vertices))),
        vertex_occurrence_paths=((),) * len(core_vertices),
        facet_ids=tuple(f"tiny-core-f{i}" for i in range(len(polygons))),
        facet_occurrence_paths=((),) * len(polygons),
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((),),
        boundary="fixed",
    )
    vertex_layer = np.concatenate((cap_layer, np.full(n, -1, dtype=np.int64)))
    cap_ids = np.arange(len(cap_rows), dtype=np.int64)
    layer_regions = np.zeros(
        sum(b.cell_count for b in layers.mesh.blocks), dtype=np.int64
    )
    region_map = np.asarray((0,), dtype=np.int64)
    mapping, cap_ids, core_polygons, points = topology.prepare_identity(
        layers, core, vertex_layer, cap_ids
    )
    combined, _ = topology.combined_domain(
        layers,
        core,
        layer_regions,
        mapping,
        cap_ids,
        core_polygons,
        points,
        source_id,
        ("solid",),
        region_map,
        ((),),
    )
    if len(points) != len(layers.mesh.coordinates) + n:
        raise ValueError("Tiny authored vertex bank changed")
    layer_ids = np.asarray(layers.mesh.vertex_global_ids, dtype=np.int64)
    root_ids = np.concatenate(
        (
            layer_ids,
            np.arange(
                int(layer_ids.max()) + 1, int(layer_ids.max()) + 1 + n, dtype=np.int64
            ),
        )
    )
    ancestry = np.asarray(layers.vertex_source_vertex_ids, dtype=np.int64)
    wall_rows = np.asarray(layers.wall_vertices, dtype=np.int64)
    wall_by_source = {int(ancestry[row]): int(row) for row in wall_rows if row >= 0}
    outer = [wall_by_source[k] for k in (20, 21, 23, 22)]
    inner = [int(mapping[cap_by_source[k]]) for k in (20, 21, 23, 22)]
    upper = [int(mapping[n + cap_by_source[k]]) for k in (20, 21, 23, 22)]
    root = phx.discretization.CellMesh(
        points,
        (
            phx.discretization.CellBlock(
                "authored-affine-root",
                "hexahedron",
                np.asarray((outer + inner, inner + upper), dtype=np.int64),
                global_ids=np.asarray((1001, 1002), dtype=np.int64),
            ),
        ),
        vertex_global_ids=root_ids,
    )
    geometry = phx.discretization.CellGeometrySpec.affine(root)
    domain = MappedReferenceDomain(
        combined,
        root,
        geometry,
        np.asarray((0, 0), dtype=np.int64),
        source_id=source_id,
        source_revision=revision,
        reference_domain_vertex_root_ids=root_ids,
        coordinate_contract=phx.SpatialCoordinateContract.si(),
    )
    facets = np.asarray(combined.facets, dtype=np.int64)
    sides = np.asarray(combined.facet_regions, dtype=np.int64)

    def plane_group(face: list[int]) -> int:
        xyz = points[face]
        planes = (
            (0, 0.0),
            (0, 1.0),
            (1, 0.0),
            (1, 1.0),
            (2, 0.0),
            (2, float(points[upper[0], 2])),
        )
        matches = [
            i for i, (axis, value) in enumerate(planes) if np.all(xyz[:, axis] == value)
        ]
        if len(matches) != 1:
            raise ValueError("Tiny exterior facet is not on one authored box face.")
        return matches[0]

    keys = tuple(
        (plane_group(face), tuple(int(v) for v in sides[row]))
        for row, face in enumerate(cast(list[list[int]], facets.tolist()))
    )
    unique_keys = tuple(sorted(set(keys)))
    lookup = {key: i for i, key in enumerate(unique_keys)}
    groups = np.asarray([lookup[key] for key in keys], dtype=np.int64)
    facet_sides = np.asarray([key[1][::-1] for key in unique_keys], dtype=np.int64)
    segments = np.asarray(
        [(outer[i], outer[(i + 1) % 4]) for i in range(4)]
        + [(inner[i], inner[(i + 1) % 4]) for i in range(4)]
        + [(upper[i], upper[(i + 1) % 4]) for i in range(4)]
        + [(outer[i], inner[i]) for i in range(4)]
        + [(inner[i], upper[i]) for i in range(4)],
        dtype=np.int64,
    )
    full = PiecewiseLinearComplex(
        points,
        tuple(row for row in facets),
        groups,
        facet_sides,
        ("solid",),
        segments=segments,
        vertex_ids=tuple(f"tiny-source-v{int(i)}" for i in root_ids),
        vertex_occurrence_paths=((),) * len(points),
        facet_ids=tuple(f"tiny-box-face-{i}" for i in range(len(unique_keys))),
        facet_occurrence_paths=((),) * len(unique_keys),
        segment_ids=tuple(f"tiny-box-edge-{i}" for i in range(len(segments))),
        segment_occurrence_paths=((),) * len(segments),
        region_occurrence_paths=((),),
        boundary="fixed",
    )
    authority = NativePlcSource(
        full,
        source_id,
        revision,
        mapped_domain=domain,
        plc_vertex_root_global_ids=root_ids,
    )
    query = authority.boundary_query()
    face_rows = {
        tuple(sorted(int(v) for v in face)): row
        for row, face in enumerate(cast(list[list[int]], facets.tolist()))
    }
    source_faces = []
    source_paths = []
    for index, polygon in enumerate(core_polygons):
        if index in set(cap_ids.tolist()):
            source_faces.append(None)
            source_paths.append(None)
            continue
        key = tuple(sorted(int(v) for v in mapping[polygon]))
        face = face_rows[key]
        group = int(groups[face])
        source_faces.append(full.facet_ids[group])
        source_paths.append(full.facet_occurrence_paths[group])
    kwargs: dict[str, Any] = dict(
        vertex_layer_ids=vertex_layer,
        cap_polygon_ids=cap_ids,
        cap_refinement="fixed",
        noncap_refinement="conforming",
        layer_control=control,
        layer_policy=policy,
        layer_regions=layer_regions,
        region_entity_ids=np.asarray(core.region_entity_ids, dtype=np.int64),
        region_occurrence_paths=core.region_occurrence_paths,
        region_ids=core.region_ids,
        core_region_map=region_map,
        core_facet_source_ids=tuple(source_faces),
        core_facet_source_occurrence_paths=tuple(source_paths),
    )
    reference = NativeLayerCoreSource(layers, core, source_id, revision, **kwargs)
    source = NativeLayerCoreSource(
        layers,
        core,
        source_id,
        revision,
        **kwargs,
        fidelity_source=query,
        maximum_deviation=0.0,
        reference_source=reference,
        mapped_domain=domain,
    )
    from phydrax.meshing._layer_core_source_restore import (
        validate_mapped_layer_core_source,
    )

    validate_mapped_layer_core_source(source)
    family = phx.meshing.CellFamilyPolicy(
        required=("tetrahedron",), preferred=("prism",), allow_mixed=True
    )
    from phydrax.meshing._layer_core_controls import (
        mapped_wall_source_scope,
        realized_layer_control,
    )

    volume = phx.meshing.MeshingScope(
        source.source_id,
        source.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        3,
        source.scope_entity_set_id(3),
        np.asarray(source.region_entity_ids, dtype=np.int64)[
            np.unique(np.asarray(source.layer_regions, dtype=np.int64))
        ],
    )
    source_control = phx.meshing.BoundaryLayerControl(
        mapped_wall_source_scope(source),
        control.schedule,
        route=control.route,
        volume_scope=volume,
        collision=control.collision,
        corner=control.corner,
        feature_angle=control.feature_angle,
        minimum_thickness_fraction=control.minimum_thickness_fraction,
        growth_rate_bounds=control.growth_rate_bounds,
        maximum_corner_stretch=control.maximum_corner_stretch,
        smoothing_iterations=control.smoothing_iterations,
        core_maximum_size=control.core_maximum_size,
    )
    if realized_layer_control(source, source_control).control_id != layers.control_id:
        raise ValueError("Tiny source control differs from its actual realized layer.")
    spec = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(3, 3, family, geometry_order=1),
        volume,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                volume, 0.75, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
        region_seeds=(
            phx.meshing.RegionSeed(
                np.asarray((0.5, 0.5, 0.5)),
                core.region_ids[0],
                "tiny-mapped-core",
                phx.meshing.RegionRole.SOLID,
            ),
        ),
        layer_controls=(source_control,),
        maximum_deviation=0.0,
    )
    result = (
        phx.meshing.NativeMeshingProvider(NativeMeshingOptions("layer_core"))
        .plan(source, spec, coordinate_contract=phx.SpatialCoordinateContract.si())
        .execute()
    )
    if result.certification is None or not result.certification.passed:
        raise ValueError("Tiny real affine mapped producer did not certify.")
    columns = s._accepted_layer_columns(result)
    adapter = NativeLayerCoreAssociationTransfer(source, spec)
    prepared, refined, _ = s._periodic_mixed_refinement(result, columns, adapter, None)
    return source, spec, result, prepared, refined
