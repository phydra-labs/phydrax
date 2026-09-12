#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _contract() -> phx.SpatialCoordinateContract:
    return phx.SpatialCoordinateContract(phx.units.MILLIMETER)


def _planar_region(name: str, x0: float, x1: float) -> phx.geometry.PlanarMeshRegion:
    return phx.geometry.PlanarMeshRegion(
        np.asarray(((x0, 0.0), (x1, 0.0), (x1, 1.0), (x0, 1.0))),
        ((0, 1, 2, 3),),
        feature_id=name,
    )


def _planar_partition(destination: Path):
    contract = _contract()
    embedding = phx.geometry.PlanarEmbedding(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    operands = (
        phx.geometry.PlanarPartitionOperand(
            "left",
            _planar_region("left", 0.0, 1.0),
            phx.geometry.BRepPartitionRole.REGION,
        ),
        phx.geometry.PlanarPartitionOperand(
            "right",
            _planar_region("right", 1.0, 2.0),
            phx.geometry.BRepPartitionRole.REGION,
        ),
    )
    plan = phx.geometry.PlanarPartitionPlan(
        contract,
        embedding,
        operands,
        phx.geometry.BRepPartitionPolicy(region_precedence=("right", "left")),
    )
    return embedding, phx.geometry.partition_planar(plan, destination=destination)


def _persist_partition_operand(
    shape, path: Path, contract: phx.SpatialCoordinateContract
):
    return phx.geometry.persist_occt_shape(
        shape,
        path,
        coordinate_contract=contract,
    )


def _three_dimensional_partition(destination: Path):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    contract = _contract()
    left_shape = BRepPrimAPI_MakeBox(gp_Pnt(0.0, 0.0, 0.0), 1.0, 1.0, 1.0).Shape()
    right_shape = BRepPrimAPI_MakeBox(gp_Pnt(1.0, 0.0, 0.0), 1.0, 1.0, 1.0).Shape()
    operands = (
        phx.geometry.BRepPartitionOperand(
            "left",
            _persist_partition_operand(left_shape, destination / "left.brep", contract),
            phx.geometry.BRepPartitionRole.REGION,
        ),
        phx.geometry.BRepPartitionOperand(
            "right",
            _persist_partition_operand(right_shape, destination / "right.brep", contract),
            phx.geometry.BRepPartitionRole.REGION,
        ),
    )
    plan = phx.geometry.BRepPartitionPlan(
        contract,
        operands,
        phx.geometry.BRepPartitionPolicy(region_precedence=("right", "left")),
    )
    return phx.geometry.partition_brep(plan, destination=destination / "composite.brep")


def _region_controls(provider, partition, model=None):
    source = partition.model if model is None else model
    return tuple(
        phx.meshing.RegionControl(
            provider.entity_scope(source, region.entity_ids),
            region.name,
            f"material:{region.name}",
            phx.meshing.RegionRole.SOLID,
        )
        for region in partition.regions
    )


def _patch_controls(provider, partition, model=None):
    source = partition.model if model is None else model
    return tuple(
        phx.meshing.PatchControl(
            patch.name,
            provider.entity_scope(source, patch.entity_ids),
            patch.adjacent_region_ids,
        )
        for patch in partition.patches
    )


def _semantic_gmsh_specification(
    provider,
    partition,
    model=None,
    /,
    *,
    scoped_size: bool,
):
    source = partition.model if model is None else model
    whole = provider.whole_scope(source, 3)
    controls = (
        phx.meshing.UniformSizeControl(
            whole,
            0.35,
            strength=(
                phx.meshing.SizeControlStrength.SOFT
                if scoped_size
                else phx.meshing.SizeControlStrength.HARD
            ),
        ),
    )
    if scoped_size:
        local = provider.entity_scope(source, partition.regions[0].entity_ids)
        controls = (
            *controls,
            phx.meshing.UniformSizeControl(local, 0.16),
        )
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(required=("tetrahedron",)),
            geometry_order=1,
        ),
        whole,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=controls,
        region_controls=_region_controls(provider, partition, source),
        patch_controls=_patch_controls(provider, partition, source),
    )
    return specification, controls


def _require_semantic_gmsh_result(result, partition) -> None:
    if not result.audit.passed or not result.compliance.passed:
        raise RuntimeError("Semantic Gmsh result failed audit or compliance.")
    zones = {
        zone.name: zone
        for zone in result.zones
        if zone.role is phx.meshing.MeshZoneRole.REGION
    }
    expected_regions = {region.name for region in partition.regions}
    if set(zones) != expected_regions:
        raise RuntimeError("Semantic Gmsh result lost exact region ownership.")
    expected_cells = np.asarray(result.mesh.entity_set(3).entity_ids)
    observed_cells = np.concatenate(
        tuple(np.asarray(zone.scope.entity_ids) for zone in zones.values())
    )
    if np.unique(observed_cells).size != observed_cells.size or not np.array_equal(
        np.sort(observed_cells), np.sort(expected_cells)
    ):
        raise RuntimeError("Semantic Gmsh region zones are not exclusive and exhaustive.")
    patches = {patch.name: patch for patch in result.patches}
    expected_patches = {patch.name for patch in partition.patches}
    if set(patches) != expected_patches:
        raise RuntimeError("Semantic Gmsh result lost exact declared patches.")


def qualify_gmsh() -> dict[str, object]:
    import build123d as bd

    with TemporaryDirectory(prefix="phydrax-gmsh-semantic-") as temporary:
        root = Path(temporary)
        partition = _three_dimensional_partition(root)
        provider = phx.meshing.GmshProvider()
        specification, _controls = _semantic_gmsh_specification(
            provider,
            partition,
            scoped_size=False,
        )
        plan = provider.plan(partition.model, specification)
        result = plan.execute()
        _require_semantic_gmsh_result(result, partition)

        replayed_model = phx.geometry.import_brep(
            partition.model.source_id,
            coordinate_contract=partition.model.coordinate_contract,
        )
        replayed_specification, _ = _semantic_gmsh_specification(
            provider,
            partition,
            replayed_model,
            scoped_size=False,
        )
        replayed_plan = provider.plan(replayed_model, replayed_specification)
        replayed = replayed_plan.execute()
        _require_semantic_gmsh_result(replayed, partition)
        replay_preserved = (
            partition.model.source_revision == replayed_model.source_revision
            and plan.plan_id == replayed_plan.plan_id
            and result.result_id == replayed.result_id
        )
        if not replay_preserved:
            raise RuntimeError("Native BRep replay changed semantic Gmsh identity.")

        scoped_specification, scoped_controls = _semantic_gmsh_specification(
            provider,
            partition,
            scoped_size=True,
        )
        scoped = provider.plan(partition.model, scoped_specification).execute()
        _require_semantic_gmsh_result(scoped, partition)
        scoped_evidence = dict(scoped.compliance.achieved)
        if any(
            f"size:{control.control_id}:minimum_edge" not in scoped_evidence
            or f"size:{control.control_id}:maximum_edge" not in scoped_evidence
            for control in scoped_controls
        ):
            raise RuntimeError("Scoped Gmsh sizing lacks per-control evidence.")

        phx.geometry.persist_occt_shape(
            bd.Box(3.0, 1.0, 1.0).wrapped,
            partition.model.source_id,
            coordinate_contract=partition.model.coordinate_contract,
            overwrite=True,
        )
        stale_rejected = False
        try:
            plan.execute()
        except phx.meshing.MeshingFailure as failure:
            if failure.category is not phx.meshing.MeshingFailureCategory.INVALID_SOURCE:
                raise
            stale_rejected = True
        if not stale_rejected:
            raise RuntimeError("Semantic Gmsh plan accepted replaced BRep bytes.")

    return {
        "provider": result.provider.name,
        "version": result.runtime.actual_version,
        "result_id": result.result_id,
        "plan_id": plan.plan_id,
        "source_revision": partition.model.source_revision,
        "regions": len(partition.regions),
        "patches": len(partition.patches),
        "cells": result.mesh.entity_set(3).count,
        "replay_preserved": replay_preserved,
        "stale_rejected": stale_rejected,
        "scoped_size_controls": [control.control_id for control in scoped_controls],
    }


def _layout_bytes() -> bytes:
    import gdstk

    with TemporaryDirectory(prefix="phydrax-layout-fixture-") as temporary:
        path = Path(temporary) / "qualification.gds"
        library = gdstk.Library(unit=1.0e-3, precision=1.0e-9)
        top = library.new_cell("TOP")
        top.add(gdstk.rectangle((0.0, 0.0), (2.0, 1.0), layer=1, datatype=0))
        library.write_gds(path)
        return path.read_bytes()


def _decode_layout():
    contract = _contract()
    limits = phx.interchange.ResourceLimits(
        max_bytes=1_000_000,
        max_depth=4,
        max_nodes=1_000,
        max_attributes=1_000,
        max_losses=0,
    )
    policy = phx.interchange.LayoutImportPolicy(
        phx.interchange.LayoutFormat.GDSII,
        contract,
        limits,
        top_cell="TOP",
    )
    return phx.interchange.decode_layout_bytes(
        _layout_bytes(), policy, source_path="qualification.gds"
    )


def _hybrid_face_indices(source, axis: int, coordinate: float):
    points = np.asarray(source.mesh_vertices)
    triangles = points[np.asarray(source.mesh_faces)]
    face_ids = np.asarray(source.triangle_face_ids)
    return tuple(
        int(value)
        for value in np.unique(
            face_ids[
                np.all(
                    np.isclose(triangles[:, :, axis], coordinate),
                    axis=1,
                )
            ]
        )
    )


def _hybrid_gmsh_result(destination: Path, *, schedule=None):
    from OCP.BOPAlgo import BOPAlgo_Splitter
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    lower = BRepPrimAPI_MakeBox(gp_Pnt(0.0, 0.0, 0.0), 1.0, 1.0, 0.3).Shape()
    upper = BRepPrimAPI_MakeBox(gp_Pnt(0.0, 0.0, 0.3), 1.0, 1.0, 0.7).Shape()
    splitter = BOPAlgo_Splitter()
    splitter.AddArgument(lower)
    splitter.AddArgument(upper)
    splitter.Perform()
    if splitter.HasErrors():
        raise RuntimeError("OCCT failed to build the hybrid qualification fixture.")
    source = phx.geometry.persist_occt_shape(
        splitter.Shape(),
        destination,
        coordinate_contract=_contract(),
        linear_deflection=0.03,
        angular_deflection=0.15,
    )
    source_face = _hybrid_face_indices(source, 2, 0.0)
    target_face = _hybrid_face_indices(source, 2, 0.3)
    if len(source_face) != 1 or len(target_face) != 1:
        raise RuntimeError("Hybrid qualification could not resolve its exact caps.")
    swept = source.topology.face_solids[source_face[0]][0]
    core = next(index for index in range(source.topology.num_solids) if index != swept)
    provider = phx.meshing.GmshProvider()
    swept_scope = provider.entity_scope(source, source.solid_ids[swept])
    core_scope = provider.entity_scope(source, source.solid_ids[core])
    layer_schedule = (
        phx.meshing.LayerSchedule((0.08, 0.10, 0.12)) if schedule is None else schedule
    )
    if not isinstance(layer_schedule, phx.meshing.LayerSchedule):
        raise TypeError("schedule must be LayerSchedule or None.")
    control = phx.meshing.SweptLayerControl(
        provider.entity_scope(source, source.face_ids[source_face[0]]),
        provider.entity_scope(source, source.face_ids[target_face[0]]),
        swept_scope,
        layer_schedule,
    )
    whole = provider.whole_scope(source, 3)
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(
                required=("prism", "tetrahedron"),
                allow_mixed=True,
            ),
        ),
        whole,
        phx.meshing.VolumeFillStrategy.SWEEP,
        size_controls=(phx.meshing.UniformSizeControl(whole, 0.24),),
        region_controls=(
            phx.meshing.RegionControl(
                swept_scope,
                "swept-slab",
                "swept-material",
                phx.meshing.RegionRole.SOLID,
            ),
            phx.meshing.RegionControl(
                core_scope,
                "tetra-core",
                "core-material",
                phx.meshing.RegionRole.SOLID,
            ),
        ),
        patch_controls=(
            phx.meshing.PatchControl(
                "slab-core",
                provider.entity_scope(source, source.face_ids[target_face[0]]),
                ("swept-slab", "tetra-core"),
            ),
        ),
        layer_controls=(control,),
    )
    return provider.plan(source, specification).execute(), control


def _mesh_scope(mesh: phx.discretization.CellMesh, dimension: int, ids: np.ndarray):
    entities = mesh.entity_set(dimension)
    return phx.meshing.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        dimension,
        entities.entity_set_id,
        ids,
    )


def qualify_manifold() -> dict[str, object]:
    import manifold3d

    def cube(offset):
        arrays = manifold3d.Manifold.cube().translate(offset).to_mesh64()
        return phx.geometry.SurfaceModel.from_triangles(
            arrays.vert_properties[:, :3],
            arrays.tri_verts,
            phx.geometry.SurfaceMetadata(
                source_id=str(offset),
                source_revision="0",
                coordinate_contract=phx.SpatialCoordinateContract.si(),
                provenance=("qualification",),
            ),
        )

    result = phx.meshing.ManifoldProvider().execute(
        cube((0.0, 0.0, 0.0)),
        cube((0.5, 0.0, 0.0)),
        phx.meshing.SurfaceBooleanOperation.DIFFERENCE,
    )
    points = np.asarray(result.mesh.coordinates)[
        np.asarray(result.mesh.blocks[0].vertices)
    ]
    volume = float(
        np.sum(np.sum(points[:, 0] * np.cross(points[:, 1], points[:, 2]), axis=1)) / 6.0
    )
    if not np.isclose(volume, 0.5, atol=1e-12) or not result.audit.passed:
        raise RuntimeError("Manifold difference failed solid-volume qualification.")
    covered = np.concatenate(
        [np.asarray(item.target_global_ids) for item in result.associations]
    )
    if not np.array_equal(
        np.sort(covered), np.sort(np.asarray(result.mesh.entity_set(2).entity_ids))
    ):
        raise RuntimeError("Manifold result lacks complete source-face ancestry.")
    return {
        "provider": result.provider.name,
        "version": result.runtime.actual_version,
        "volume": volume,
        "source_associations": len(result.associations),
        "audit_passed": result.audit.passed,
    }


def qualify_volume(
    provider_name: str, executable: str, extractor: str
) -> dict[str, object]:
    contract = phx.SpatialCoordinateContract.si()
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    faces = np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), dtype=np.int32)
    expected_volume = 1.0 / 6.0
    surface = phx.geometry.SurfaceModel.from_triangles(
        points,
        faces,
        phx.geometry.SurfaceMetadata(
            source_id="qualification-tetrahedron",
            source_revision="0",
            coordinate_contract=contract,
            provenance=("qualification",),
        ),
    )
    if provider_name == "mmg":
        mesh = phx.discretization.CellMesh.from_tetrahedra(
            points, np.asarray(((0, 1, 2, 3),), dtype=np.int32)
        )
        provider = phx.meshing.MmgProvider()
        metric = phx.meshing.MeshMetricField(
            provider.vertex_scope(mesh),
            np.tile(np.eye(3) * 16.0, (4, 1, 1)),
            minimum_size=0.25,
            maximum_size=0.25,
        )
        result = provider.adapt(mesh, metric, contract)
    elif provider_name == "ftetwild":
        provider = phx.meshing.FTetWildProvider(
            phx.meshing.FTetWildOptions(envelope_distance=0.005, maximum_iterations=20)
        )
        scope = provider.whole_scope(surface)
        specification = phx.meshing.VolumeMeshingSpec(
            phx.meshing.CellMeshingTarget(
                3, 3, phx.meshing.CellFamilyPolicy(required=("tetrahedron",))
            ),
            scope,
            phx.meshing.VolumeFillStrategy.SIMPLEX,
            deterministic=False,
            size_controls=(
                phx.meshing.UniformSizeControl(
                    scope, 0.3, strength=phx.meshing.SizeControlStrength.SOFT
                ),
            ),
        )
        result = provider.plan(surface, specification).execute()
    elif provider_name == "vorocrust":
        result = phx.meshing.VoroCrustProvider(executable, extractor).execute(
            surface, phx.meshing.VoroCrustOptions(1.0)
        )
    else:
        import build123d as bd

        with TemporaryDirectory(prefix="phydrax-gmsh-qualification-") as temporary:
            path = Path(temporary) / "cube.step"
            coordinate_contract = _contract()
            bd.export_step(bd.Box(1.0, 1.0, 1.0), path, unit=bd.Unit.MM)
            source = phx.geometry.BRepSource(
                phx.geometry.import_brep(
                    path,
                    coordinate_contract=coordinate_contract,
                    linear_deflection=0.05,
                    angular_deflection=0.2,
                )
            )
            provider = phx.meshing.GmshProvider()
            scope = provider.whole_scope(source, 3)
            specification = phx.meshing.VolumeMeshingSpec(
                phx.meshing.CellMeshingTarget(
                    3,
                    3,
                    phx.meshing.CellFamilyPolicy(required=("tetrahedron",)),
                    geometry_order=2,
                ),
                scope,
                phx.meshing.VolumeFillStrategy.SIMPLEX,
                size_controls=(
                    phx.meshing.UniformSizeControl(scope, 0.3, maximum_growth_rate=2.0),
                ),
            )
            result = provider.plan(source, specification).execute()
            expected_volume = 1.0
    volume = float(np.sum(np.asarray(result.quality.evaluation.measures)))
    if not np.isclose(volume, expected_volume, rtol=0.03) or not result.audit.passed:
        raise RuntimeError(f"{provider_name} failed enclosed-volume qualification.")
    return {
        "provider": result.provider.name,
        "version": result.runtime.actual_version,
        "cells": result.mesh.entity_set(3).count,
        "volume": volume,
        "audit_passed": result.audit.passed,
        "quality_scope": result.audit.quality_scope,
    }


def qualify_poisson() -> dict[str, object]:
    count = 400
    z = 1.0 - 2.0 * (np.arange(count) + 0.5) / count
    angles = np.arange(count) * np.pi * (3.0 - np.sqrt(5.0))
    normals = np.column_stack(
        (np.sqrt(1.0 - z * z) * np.cos(angles), np.sqrt(1.0 - z * z) * np.sin(angles), z)
    )
    source = phx.meshing.OrientedPointCloud(
        normals,
        normals,
        phx.SpatialCoordinateContract.si(),
        source_id="sphere",
        source_revision="0",
    )
    result = phx.meshing.PoissonProvider().execute(
        source, phx.meshing.PoissonReconstructionSpec(depth=5)
    )
    error = float(
        np.max(np.abs(np.linalg.norm(np.asarray(result.mesh.coordinates), axis=1) - 1.0))
    )
    if error > 0.1 or not result.audit.passed:
        raise RuntimeError("Poisson reconstruction failed sphere qualification.")
    return {
        "provider": result.provider.name,
        "maximum_radius_error": error,
        "audit_passed": result.audit.passed,
    }


def qualify_cad_partition() -> dict[str, object]:
    with TemporaryDirectory(prefix="phydrax-cad-partition-") as temporary:
        result = _three_dimensional_partition(Path(temporary))
    if result.topological_dimension != 3 or not result.named_solid_entity_ids:
        raise RuntimeError("CAD partition did not produce exact solid regions.")
    if not result.association_graph.graph_id or not result.revision.revision_id:
        raise RuntimeError("CAD partition omitted revision association evidence.")
    return {
        "coordinate_contract": result.model.coordinate_contract.spatial_id,
        "revision": result.revision.revision_id,
        "association_graph": result.association_graph.graph_id,
        "solid_regions": {name: len(ids) for name, ids in result.named_solid_entity_ids},
        "face_patches": {name: len(ids) for name, ids in result.named_face_entity_ids},
    }


def qualify_composite_heat_3d() -> dict[str, object]:
    with TemporaryDirectory(prefix="phydrax-composite-heat-3d-") as temporary:
        partition = _three_dimensional_partition(Path(temporary))
        provider = phx.meshing.GmshProvider()
        boundary_scope = provider.whole_scope(partition.model, 3)
        specification = phx.meshing.VolumeMeshingSpec(
            phx.meshing.CellMeshingTarget(
                3, 3, phx.meshing.CellFamilyPolicy(required=("tetrahedron",))
            ),
            boundary_scope,
            phx.meshing.VolumeFillStrategy.SIMPLEX,
            size_controls=(phx.meshing.UniformSizeControl(boundary_scope, 0.2),),
            region_controls=_region_controls(provider, partition),
            patch_controls=_patch_controls(provider, partition),
        )
        result = provider.plan(partition.model, specification).execute()
    if not result.audit.passed:
        raise RuntimeError("Composite three-dimensional heat mesh failed audit.")
    zones = {
        zone.name: zone
        for zone in result.zones
        if zone.role is phx.meshing.MeshZoneRole.REGION
    }
    field = phx.discretization.FiniteElementFieldSpec(
        "temperature",
        phx.discretization.lagrange_element("tetrahedron", 1),
    )
    discretization = phx.discretization.FiniteElementPlan(
        result.mesh,
        field,
    ).prepare()
    domains = {
        name: discretization.integration_domain(
            "cell",
            phx.meshing.resolve_mesh_scope(result.mesh, zone.scope),
        )
        for name, zone in zones.items()
    }
    exterior_patches = tuple(
        patch for patch in result.patches if len(patch.adjacent_zone_ids) == 1
    )
    exterior_scope = exterior_patches[0].scope
    for patch in exterior_patches[1:]:
        exterior_scope = exterior_scope | patch.scope
    constraint = phx.discretization.dirichlet_constraint(
        discretization,
        "temperature",
        boundary_selection=phx.meshing.resolve_mesh_scope(
            result.mesh,
            exterior_scope,
        ),
    )
    form = phx.equations.FiniteElementForm(
        "two-region-volume-diffusion",
        "temperature",
        (
            phx.equations.DiffusionAction(
                "temperature",
                1.0,
                action_id="left-volume-conductivity",
                domain=domains["left"],
            ),
            phx.equations.DiffusionAction(
                "temperature",
                2.0,
                action_id="right-volume-conductivity",
                domain=domains["right"],
            ),
        ),
    )
    compiled = phx.equations.compile_finite_element_problem(
        form,
        discretization,
        constraint=constraint,
        dirichlet_values=lambda points: jnp.where(
            points[..., 0] <= 1.0,
            points[..., 0],
            1.0 + 0.5 * (points[..., 0] - 1.0),
        ),
    )
    system, right_hand_side = compiled.linear_system()
    solved = phx.linalg.solve(system, right_hand_side)
    solution = np.asarray(compiled.expand(solved.value))
    coordinates = np.asarray(result.mesh.coordinates)
    expected = np.where(
        coordinates[:, 0] <= 1.0,
        coordinates[:, 0],
        1.0 + 0.5 * (coordinates[:, 0] - 1.0),
    )
    error = float(np.max(np.abs(solution - expected)))
    if not bool(np.all(np.asarray(solved.successful))) or error > 1.0e-6:
        raise RuntimeError(
            "Composite volume diffusion failed its exact interface solution."
        )
    return {
        "mesh_id": result.mesh.mesh_id,
        "cells": result.mesh.entity_set(3).count,
        "minimum_quality": float(result.quality.minimum_mean_ratio),
        "revision": partition.revision.revision_id,
        "association_graph": partition.association_graph.graph_id,
        "region_control_count": len(specification.region_controls),
        "patch_control_count": len(specification.patch_controls),
        "dofs": discretization.dof_maps[0].global_dof_count,
        "maximum_solution_error": error,
    }


def qualify_planar_semantic() -> dict[str, object]:
    with TemporaryDirectory(prefix="phydrax-planar-semantic-") as temporary:
        embedding, result = _planar_partition(Path(temporary) / "planar.brep")
    if (
        result.topological_dimension != 2
        or result.model.topology.num_solids != 0
        or not result.named_region_entity_ids
        or not result.named_patch_entity_ids
    ):
        raise RuntimeError("Planar partition did not preserve two-dimensional semantics.")
    return {
        "embedding": embedding.embedding_id,
        "revision": result.revision.revision_id,
        "association_graph": result.association_graph.graph_id,
        "region_entity_kind": result.region_entity_kind,
        "patch_entity_kind": result.patch_entity_kind,
        "regions": {name: len(ids) for name, ids in result.named_region_entity_ids},
        "patches": {name: len(ids) for name, ids in result.named_patch_entity_ids},
    }


def qualify_composite_heat_2d() -> dict[str, object]:
    with TemporaryDirectory(prefix="phydrax-composite-heat-2d-") as temporary:
        embedding, partition = _planar_partition(Path(temporary) / "planar.brep")
        provider = phx.meshing.GmshProvider()
        scope = provider.whole_scope(partition.model, 2)
        specification = phx.meshing.SurfaceMeshingSpec(
            phx.meshing.CellMeshingTarget(
                2,
                2,
                phx.meshing.CellFamilyPolicy(required=("triangle",)),
            ),
            scope,
            planar_embedding=embedding,
            size_controls=(phx.meshing.UniformSizeControl(scope, 0.1),),
            region_controls=_region_controls(provider, partition),
            patch_controls=_patch_controls(provider, partition),
        )
        result = provider.plan(partition.model, specification).execute()
    if not result.audit.passed or result.boundary is not None:
        raise RuntimeError("Composite two-dimensional heat mesh failed planar audit.")
    zones = {
        zone.name: zone
        for zone in result.zones
        if zone.role is phx.meshing.MeshZoneRole.REGION
    }
    if set(zones) != {"left", "right"}:
        raise RuntimeError("Planar meshing lost exact material-region zones.")
    field = phx.discretization.FiniteElementFieldSpec(
        "temperature",
        phx.discretization.lagrange_element("triangle", 1),
    )
    discretization = phx.discretization.FiniteElementPlan(
        result.mesh,
        field,
    ).prepare()
    domains = {
        name: discretization.integration_domain(
            "cell",
            phx.meshing.resolve_mesh_scope(result.mesh, zone.scope),
        )
        for name, zone in zones.items()
    }
    exterior_patches = tuple(
        patch for patch in result.patches if len(patch.adjacent_zone_ids) == 1
    )
    exterior_scope = exterior_patches[0].scope
    for patch in exterior_patches[1:]:
        exterior_scope = exterior_scope | patch.scope
    constraint = phx.discretization.dirichlet_constraint(
        discretization,
        "temperature",
        boundary_selection=phx.meshing.resolve_mesh_scope(
            result.mesh,
            exterior_scope,
        ),
    )
    form = phx.equations.FiniteElementForm(
        "two-region-planar-diffusion",
        "temperature",
        (
            phx.equations.DiffusionAction(
                "temperature",
                1.0,
                action_id="left-conductivity",
                domain=domains["left"],
            ),
            phx.equations.DiffusionAction(
                "temperature",
                2.0,
                action_id="right-conductivity",
                domain=domains["right"],
            ),
        ),
    )
    compiled = phx.equations.compile_finite_element_problem(
        form,
        discretization,
        constraint=constraint,
        dirichlet_values=lambda points: jnp.where(
            points[..., 0] <= 1.0,
            points[..., 0],
            1.0 + 0.5 * (points[..., 0] - 1.0),
        ),
    )
    system, right_hand_side = compiled.linear_system()
    solved = phx.linalg.solve(system, right_hand_side)
    solution = np.asarray(compiled.expand(solved.value))
    coordinates = np.asarray(result.mesh.coordinates)
    expected = np.where(
        coordinates[:, 0] <= 1.0,
        coordinates[:, 0],
        1.0 + 0.5 * (coordinates[:, 0] - 1.0),
    )
    error = float(np.max(np.abs(solution - expected)))
    if not bool(np.all(np.asarray(solved.successful))) or error > 1.0e-6:
        raise RuntimeError(
            "Composite planar diffusion failed its exact interface solution."
        )
    return {
        "mesh_id": result.mesh.mesh_id,
        "cells": result.mesh.entity_set(2).count,
        "minimum_quality": float(result.quality.minimum_mean_ratio),
        "revision": partition.revision.revision_id,
        "association_graph": partition.association_graph.graph_id,
        "dofs": discretization.dof_maps[0].global_dof_count,
        "maximum_solution_error": error,
    }


def qualify_layout_observable() -> dict[str, object]:
    result = _decode_layout()
    region = result.model.regions[0]
    expected_layer = phx.interchange.LayoutLayerKey(1, 0)
    if (
        not result.report.valid
        or region.layer != expected_layer
        or result.source_digest != result.resource_manifest.content_sha256
        or result.model.coordinate_contract != _contract()
    ):
        raise RuntimeError(
            "Layout decoder omitted observable identity or layer evidence."
        )
    return {
        "layout_model_id": result.model.model_id,
        "source_digest": result.source_digest,
        "layer": [region.layer.layer, region.layer.datatype],
        "occurrence_path": list(region.occurrence_path),
        "provenance_id": region.provenance_id,
        "coordinate_contract": result.model.coordinate_contract.spatial_id,
    }


def qualify_layout_stack() -> dict[str, object]:
    decoded = _decode_layout()
    with TemporaryDirectory(prefix="phydrax-layout-stack-") as temporary:
        result = phx.geometry.lower_process_stack(
            phx.geometry.ProcessStack(
                decoded.model.coordinate_contract,
                (
                    phx.geometry.StackRegion(
                        "metal",
                        decoded.model.regions[0].geometry,
                        phx.geometry.ZInterval(0.0, 0.8),
                        20,
                    ),
                ),
            ),
            Path(temporary) / "stack.brep",
            operand_directory=Path(temporary) / "operands",
        )
    if not result.named_solid_entity_ids or not result.association_graph.graph_id:
        raise RuntimeError("Process stack omitted persisted CAD identity evidence.")
    return {
        "revision": result.revision.revision_id,
        "association_graph": result.association_graph.graph_id,
        "solids": {name: len(ids) for name, ids in result.named_solid_entity_ids},
        "faces": {name: len(ids) for name, ids in result.named_face_entity_ids},
    }


def qualify_planar_bands() -> dict[str, object]:
    with TemporaryDirectory(prefix="phydrax-planar-bands-") as temporary:
        embedding, partition = _planar_partition(Path(temporary) / "planar.brep")
        patch = next(
            value for value in partition.patches if len(value.adjacent_region_ids) == 2
        )
        schedule = phx.meshing.LayerSchedule.geometric(2, 0.05, growth_rate=1.0)
        control = phx.meshing.PlanarBandControl(
            patch.name,
            {name: schedule for name in patch.adjacent_region_ids},
            0.1,
        )
        source = phx.meshing.prepare_planar_bands(
            phx.meshing.PlanarBandPlan(partition, embedding, (control,)),
            destination=Path(temporary) / "bands.brep",
        )
        provider = phx.meshing.GmshProvider()
        scope = provider.whole_scope(source, 2)
        band_patch_controls = tuple(
            phx.meshing.PatchControl(
                band_patch.name,
                provider.entity_scope(source, band_patch.entity_ids),
                band_patch.adjacent_region_ids,
            )
            for band_patch in source.partition.patches
            if len(set(band_patch.adjacent_region_ids))
            == len(band_patch.adjacent_region_ids)
        )
        specification = phx.meshing.SurfaceMeshingSpec(
            phx.meshing.CellMeshingTarget(
                2,
                2,
                phx.meshing.CellFamilyPolicy(required=("quadrilateral",)),
            ),
            scope,
            planar_embedding=embedding,
            size_controls=(phx.meshing.UniformSizeControl(scope, 0.1),),
            region_controls=_region_controls(provider, source.partition),
            patch_controls=band_patch_controls,
        )
        mesh_result = provider.plan(source, specification).execute()
    if not mesh_result.audit.passed or not mesh_result.compliance.passed:
        raise RuntimeError("Planar band mesh failed audit or compliance.")
    achieved = dict(mesh_result.compliance.achieved)
    front_residuals = {}
    for layer in source.layer_partitions:
        key = (
            f"planar_band:{layer.control_id}:{layer.region_name}:"
            f"front:{layer.layer_index + 1}"
        )
        residual = achieved[f"{key}:maximum_residual"]
        if residual > 1.0e-12:
            raise RuntimeError("Planar band front departed from its exact CAD level.")
        front_residuals[key] = residual
    return {
        "plan_id": source.plan_id,
        "result_id": source.result_id,
        "mesh_result_id": mesh_result.result_id,
        "partition_revision": source.partition.revision.revision_id,
        "layer_partitions": len(source.layer_partitions),
        "front_residuals": front_residuals,
    }


def qualify_hybrid_layer() -> dict[str, object]:
    with TemporaryDirectory(prefix="phydrax-hybrid-layer-") as temporary:
        result, control = _hybrid_gmsh_result(Path(temporary) / "hybrid-slab.brep")
    if not result.audit.passed or not result.compliance.passed:
        raise RuntimeError("Generated hybrid layer mesh failed audit or compliance.")
    block_kinds = tuple(block.cell_kind for block in result.mesh.blocks)
    if block_kinds != ("prism", "tetrahedron"):
        raise RuntimeError("Generated hybrid layer mesh lost its required cell families.")
    achieved = dict(result.compliance.achieved)
    return {
        "mesh_id": result.mesh.mesh_id,
        "result_id": result.result_id,
        "control_id": control.control_id,
        "schedule_id": control.schedule.schedule_id,
        "thicknesses": list(control.schedule.thicknesses),
        "block_kinds": block_kinds,
        "cells": result.mesh.entity_set(3).count,
        "maximum_layer_thickness_error": achieved[
            f"layer:{control.control_id}:maximum_thickness_residual"
        ],
        "audit_passed": result.audit.passed,
    }


def qualify_hybrid_diffusion() -> dict[str, object]:
    with TemporaryDirectory(prefix="phydrax-hybrid-diffusion-") as temporary:
        generated, _ = _hybrid_gmsh_result(Path(temporary) / "hybrid-slab.brep")
    mesh = generated.mesh
    field = phx.discretization.FiniteElementFieldSpec(
        "u",
        {
            block.name: phx.discretization.lagrange_element(
                block.cell_kind,
                1,
            )
            for block in mesh.blocks
        },
    )
    discretization = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    cells = mesh.entity_set(3)
    scope = _mesh_scope(mesh, 3, np.asarray(cells.entity_ids))
    domain = discretization.integration_domain(
        "cell", phx.meshing.resolve_mesh_scope(mesh, scope)
    )
    form = phx.equations.FiniteElementForm(
        "hybrid-diffusion",
        "u",
        (phx.equations.DiffusionAction("u", 1.0, domain=domain),),
    )
    compiled = phx.equations.compile_finite_element_problem(form, discretization)
    system, rhs = compiled.linear_system()
    dof_count = discretization.dof_maps[0].global_dof_count
    constant = np.ones(dof_count)
    null_residual = float(np.max(np.abs(np.asarray(system.operator.mv(constant)))))
    coordinates = np.asarray(discretization.vertices)
    values = coordinates[:, 0]
    energy = float(np.vdot(values, np.asarray(system.operator.mv(values))))
    rhs_norm = float(np.max(np.abs(np.asarray(rhs))))
    if not (
        np.isclose(null_residual, 0.0, atol=1.0e-12)
        and np.isclose(rhs_norm, 0.0, atol=1.0e-12)
        and np.isclose(energy, 1.0, atol=1.0e-10)
    ):
        raise RuntimeError("Hybrid diffusion lost its P1 null mode or volume energy.")
    return {
        "mesh_id": mesh.mesh_id,
        "scope": scope.scope_id,
        "dofs": dof_count,
        "null_residual": null_residual,
        "rhs_norm": rhs_norm,
        "x_gradient_energy": energy,
        "generated_result_id": generated.result_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=(
            "manifold",
            "mmg",
            "ftetwild",
            "gmsh",
            "vorocrust",
            "poisson",
            "cad-partition",
            "composite-heat-3d",
            "planar-semantic",
            "composite-heat-2d",
            "layout-stack",
            "layout-observable",
            "planar-bands",
            "hybrid-layer",
            "hybrid-diffusion",
        ),
        default="manifold",
    )
    parser.add_argument("--executable", default="vc_mesh")
    parser.add_argument("--extractor", default="phydrax-vorocrust")
    args = parser.parse_args()
    if args.scenario == "manifold":
        result = qualify_manifold()
    elif args.scenario == "poisson":
        result = qualify_poisson()
    elif args.scenario == "gmsh":
        result = qualify_gmsh()
    elif args.scenario in {"mmg", "ftetwild", "vorocrust"}:
        result = qualify_volume(args.scenario, args.executable, args.extractor)
    elif args.scenario == "cad-partition":
        result = qualify_cad_partition()
    elif args.scenario == "composite-heat-3d":
        result = qualify_composite_heat_3d()
    elif args.scenario == "planar-semantic":
        result = qualify_planar_semantic()
    elif args.scenario == "composite-heat-2d":
        result = qualify_composite_heat_2d()
    elif args.scenario == "layout-stack":
        result = qualify_layout_stack()
    elif args.scenario == "layout-observable":
        result = qualify_layout_observable()
    elif args.scenario == "planar-bands":
        result = qualify_planar_bands()
    elif args.scenario == "hybrid-layer":
        result = qualify_hybrid_layer()
    else:
        result = qualify_hybrid_diffusion()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
