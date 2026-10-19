#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from phydrax._fingerprint import canonical_fingerprint
from phydrax.discretization._cell_geometry import (
    RestrictedCellGeometryElement,
)
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization.fem._reference_types import FiniteElementSpec
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.lifecycle._chunk_repository import RepositoryCorruptionError
from phydrax.lifecycle._distributed_checkpoint import (
    assemble_distributed_checkpoint_from_repository,
    publish_process_meshing_checkpoint,
    restore_meshing_checkpoint,
)
from phydrax.lifecycle._models import CheckpointManifest
from phydrax.lifecycle._restart_topology import (
    TopologyRestartPolicy,
)
from phydrax.meshing._contracts import VolumeMeshingSpec
from phydrax.meshing._mesh_part import MeshPart
from phydrax.meshing._result import CellMeshingResult
from phydrax.meshing.providers._native_sources import (
    NativePlcSource,
    NativePolyhedralSource,
)
from phydrax.units import METER
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.distributed_meshing.restart import run_restart_workers
from tests._support.distributed_meshing.restart_builders import (
    build_repository,
    checkpoint_state,
    native_source_arrays,
    native_source_closure,
    native_source_state,
    restart_relation,
    scientific_arrays,
)
from tests._support.native_family_generation import native_family_generation


# Empirical `test_native_source_registry_roundtrip_requeries_and_recertifies`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
RESTORED_SPHERE_DISTANCE_ATOL = 1e-12
# Empirical `test_restored_native_root_reestablishes_source_theorem`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
RESTORED_SPATIAL_ROOT_POINT_ATOL = 1e-12
# Empirical `test_native_brep_compiled_source_restores_and_reprepares_actual_queries`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
RESTORED_BREP_CLOSEST_POINT_ATOL = 1e-9
# Empirical `test_native_family_durable_restart_prepares_finite_volume`: matched max |delta|=1.11022e-16; joint allowance fraction=1.11021e-09, margin=100%.
RESTORED_POLYHEDRAL_VOLUME_SUM_ATOL = 1e-12
# Empirical `test_native_family_durable_restart_reconstructs_volume_field`: matched max |delta|=1.77636e-15; joint allowance fraction=4.08105e-09, margin=100%.
RESTORED_VOLUME_AFFINE_FIELD_ATOL = 2e-10
# Empirical `test_native_family_durable_restart_evaluates_intrinsic_surface_field`: matched max |delta|=8.88178e-16; joint allowance fraction=1.91061e-09, margin=100%.
RESTORED_SURFACE_AFFINE_FIELD_ATOL = 2e-10
# Empirical `test_implicit_association_transfer_archive_restores_current_source_authority`: matched max |delta|=9.99201e-16; joint allowance fraction=0.0999201, margin=90.008%.
RESTORED_ANALYTIC_PROFILE_DISTANCE_ATOL = 1e-14


def _assert_restored_scientific_arrays(record: dict[str, Any]) -> None:
    arrays = record["arrays"]
    assert tuple(arrays) == tuple(name for name, _ in checkpoint_state().array_roles)
    for evidence in arrays.values():
        assert evidence["logical_jax_array"]
        assert len(evidence["shards"]) == 1
        for shard in evidence["shards"]:
            np.testing.assert_array_equal(shard["actual"], shard["expected"])


def test_process_publication_changed_placement_preserves_scientific_state(
    tmp_path: Path,
) -> None:
    devices = tuple(jax.local_devices()[:2])
    if len(devices) != 2:
        pytest.skip("Changed-placement restart requires two actual local devices.")
    source = NamedSharding(
        Mesh(np.asarray(devices, dtype=np.object_), ("owner",)), PartitionSpec("owner")
    )
    destination = NamedSharding(
        Mesh(np.asarray(devices[::-1], dtype=np.object_), ("owner",)),
        PartitionSpec("owner"),
    )
    state = checkpoint_state()
    reference = scientific_arrays()
    arrays = {name: jax.device_put(value, source) for name, value in reference.items()}
    repository = build_repository(tmp_path / "checkpoint")
    publication = publish_process_meshing_checkpoint(
        repository,
        "mesh-accepted",
        "source-execution",
        state,
        arrays,
        analysis_plan_id="mesh-analysis",
        numeric_revision_id="accepted-revision",
        writer_id="process-writer",
    )
    assert publication.process_index == jax.process_index()
    manifest = assemble_distributed_checkpoint_from_repository(
        repository,
        "mesh-accepted",
        "mesh-analysis",
        "accepted-revision",
        "source-execution",
        expected_process_count=1,
    )
    checkpoint = restore_meshing_checkpoint(
        repository,
        manifest,
        {name: destination for name in arrays},
        restart_relation(),
        TopologyRestartPolicy(allow_topology_change=True),
        expected_state_id=state.checkpoint_state_id,
    )
    restored_state = checkpoint.state
    restored = checkpoint.arrays
    admission = checkpoint.admission
    assert admission.admitted
    assert restored_state.to_payload() == state.to_payload()
    for name, expected in reference.items():
        np.testing.assert_array_equal(np.asarray(restored[name]), expected)
    # A device now owns the other half of stable IDs, not its former slots.
    source_ids = arrays["vertex_ids"].addressable_shards
    restored_ids = restored["vertex_ids"]
    assert isinstance(restored_ids, jax.Array)
    target_ids = restored_ids.addressable_shards
    source_by_device = {shard.device: np.asarray(shard.data) for shard in source_ids}
    target_by_device = {shard.device: np.asarray(shard.data) for shard in target_ids}
    np.testing.assert_array_equal(target_by_device[devices[0]], [1003, 2401])
    np.testing.assert_array_equal(source_by_device[devices[0]], [101, 509])
    assert (
        sum(float(value) for value in np.asarray(restored["material_inventory"])) == 17.0
    )


def test_meshing_checkpoint_refuses_execution_cache_publication(tmp_path: Path) -> None:
    arrays = {name: jax.device_put(value) for name, value in scientific_arrays().items()}
    arrays["compiled_cache"] = jax.device_put(np.asarray([1], dtype=np.int32))
    with pytest.raises(ValueError, match=r"publication\.arrays"):
        publish_process_meshing_checkpoint(
            build_repository(tmp_path / "checkpoint"),
            "mesh-accepted",
            "source-execution",
            checkpoint_state(),
            arrays,
            analysis_plan_id="mesh-analysis",
            numeric_revision_id="accepted-revision",
            writer_id="process-writer",
        )


def test_meshing_restart_refuses_unqualified_changed_placement(tmp_path: Path) -> None:
    arrays = {name: jax.device_put(value) for name, value in scientific_arrays().items()}
    repository = build_repository(tmp_path / "checkpoint")
    state = checkpoint_state()
    publish_process_meshing_checkpoint(
        repository,
        "mesh-accepted",
        "source-execution",
        state,
        arrays,
        analysis_plan_id="mesh-analysis",
        numeric_revision_id="accepted-revision",
        writer_id="process-writer",
    )
    manifest = assemble_distributed_checkpoint_from_repository(
        repository,
        "mesh-accepted",
        "mesh-analysis",
        "accepted-revision",
        "source-execution",
        expected_process_count=1,
    )
    with pytest.raises(ValueError, match=r"restore_plan\.admission"):
        restore_meshing_checkpoint(
            repository,
            manifest,
            {name: value.sharding for name, value in arrays.items()},
            restart_relation(),
            TopologyRestartPolicy(allow_topology_change=False),
            expected_state_id=state.checkpoint_state_id,
        )


def test_meshing_restart_rejects_scientific_record_substitution(tmp_path: Path) -> None:
    arrays = {name: jax.device_put(value) for name, value in scientific_arrays().items()}
    repository = build_repository(tmp_path / "checkpoint")
    state = checkpoint_state()
    publish_process_meshing_checkpoint(
        repository,
        "mesh-accepted",
        "source-execution",
        state,
        arrays,
        analysis_plan_id="mesh-analysis",
        numeric_revision_id="accepted-revision",
        writer_id="process-writer",
    )
    manifest = assemble_distributed_checkpoint_from_repository(
        repository,
        "mesh-accepted",
        "mesh-analysis",
        "accepted-revision",
        "source-execution",
        expected_process_count=1,
    )
    with pytest.raises(
        RepositoryCorruptionError,
        match=r"checkpoint_state\.checkpoint_state_id",
    ):
        restore_meshing_checkpoint(
            repository,
            manifest,
            {name: value.sharding for name, value in arrays.items()},
            restart_relation(),
            TopologyRestartPolicy(allow_topology_change=True),
            expected_state_id=canonical_fingerprint({"different": "accepted-state"}),
        )


def test_meshing_restart_keeps_exact_shard_coverage_validation(tmp_path: Path) -> None:
    devices = tuple(jax.local_devices()[:2])
    if len(devices) != 2:
        pytest.skip("Partial-shard coverage refusal requires two actual local devices.")
    source = NamedSharding(
        Mesh(np.asarray(devices, dtype=np.object_), ("owner",)), PartitionSpec("owner")
    )
    arrays = {
        name: jax.device_put(value, source) for name, value in scientific_arrays().items()
    }
    repository = build_repository(tmp_path / "checkpoint")
    state = checkpoint_state()
    publish_process_meshing_checkpoint(
        repository,
        "mesh-accepted",
        "source-execution",
        state,
        arrays,
        analysis_plan_id="mesh-analysis",
        numeric_revision_id="accepted-revision",
        writer_id="process-writer",
    )
    manifest = assemble_distributed_checkpoint_from_repository(
        repository,
        "mesh-accepted",
        "mesh-analysis",
        "accepted-revision",
        "source-execution",
        expected_process_count=1,
    )
    coordinate_shard = next(
        shard
        for shard in manifest.shards
        if dict(shard.metadata)["array_path"] == "['arrays']['coordinates']"
    )
    incomplete = CheckpointManifest(
        manifest.checkpoint_id,
        manifest.analysis_plan_id,
        manifest.numeric_revision_id,
        manifest.execution_plan_id,
        tuple(
            shard
            for shard in manifest.shards
            if shard.shard_id != coordinate_shard.shard_id
        ),
        complete=True,
    )
    with pytest.raises(RepositoryCorruptionError, match=r"checkpoint\.array\.coverage"):
        restore_meshing_checkpoint(
            repository,
            incomplete,
            {name: source for name in arrays},
            restart_relation(),
            TopologyRestartPolicy(allow_topology_change=True),
            expected_state_id=state.checkpoint_state_id,
        )


@pytest.mark.parametrize(
    ("fault", "published"),
    (("roles", (True, False)), ("metadata", (True, True))),
    ids=("one-rank-role-refusal", "incoherent-scientific-records"),
)
def test_process_local_failure_cannot_commit_complete_mesh_checkpoint(
    tmp_path: Path,
    fault: Literal["roles", "metadata"],
    published: tuple[bool, bool],
) -> None:
    results = run_restart_workers(tmp_path, "rejected_checkpoint", (fault,))
    assert results == [
        {
            "process_index": 0,
            "local_published": published[0],
            "complete_manifest_returned": False,
            "publication_error": None,
            "assembly_error": (
                "ObjectNotFoundError" if fault == "roles" else "RepositoryCorruptionError"
            ),
        },
        {
            "process_index": 1,
            "local_published": published[1],
            "complete_manifest_returned": False,
            "publication_error": "ValueError" if fault == "roles" else None,
            "assembly_error": (
                "ObjectNotFoundError" if fault == "roles" else "RepositoryCorruptionError"
            ),
        },
    ]


def test_nonaddressable_process_publication_restarts_on_different_owner(
    tmp_path: Path,
) -> None:
    results = run_restart_workers(tmp_path, "process_checkpoint_restart")
    assert [
        {name: record[name] for name in ("process_index", "local_ids", "mesh_epoch")}
        for record in results
    ] == [
        {"process_index": 0, "local_ids": [1003, 2401], "mesh_epoch": 7},
        {"process_index": 1, "local_ids": [101, 509], "mesh_epoch": 7},
    ]
    for record in results:
        assert record["nonaddressable"]
        assert record["admitted"]
        assert record["state_preserved"]
        _assert_restored_scientific_arrays(record)


# SOURCE REGISTRY


def test_native_source_registry_roundtrip_requeries_and_recertifies() -> None:
    import phydrax as phx
    from phydrax._model._structure import (
        model_from_array_recipe,
        model_recipe_array_values,
        model_structure_recipe,
    )
    from phydrax.lifecycle._meshing_sources import (
        register_meshing_source_artifacts,
        validate_meshing_source_closure,
    )

    register_meshing_source_artifacts()
    mesh, geometry, audit, closure = native_source_closure()
    validate_meshing_source_closure(closure)
    recipe = model_structure_recipe(closure)
    arrays = model_recipe_array_values(closure, recipe, prefix="native-source")
    restored = model_from_array_recipe(recipe, arrays, prefix="native-source")
    validate_meshing_source_closure(restored)
    np.testing.assert_allclose(
        restored["source"]
        .boundary_distance(np.asarray([[2.0, 0.0, 0.0]], dtype=np.float64))
        .upper,
        [1.0],
        atol=RESTORED_SPHERE_DISTANCE_ATOL,
    )
    inputs = restored["certification_inputs"]
    renewed = phx.meshing.certify_meshing_acceptance(
        mesh,
        geometry,
        audit,
        schedule=inputs.schedule,
        source=inputs.source,
        fidelity_tolerance=inputs.fidelity_tolerance,
        fidelity_sample_order=inputs.fidelity_sample_order,
        limits=inputs.limits,
    )
    renewed.require_passed()
    assert renewed.request.request_id == restored["report"].request.request_id
    assert renewed.fidelity is not None
    assert renewed.fidelity.status == "certified"
    assert restored["associations"][0].source_entities.occurrence_paths() == ((),) * 6


def test_native_source_registry_refuses_explicit_callback() -> None:
    from phydrax.lifecycle._meshing_sources import validate_meshing_source_closure

    def callback() -> None:
        return None

    with pytest.raises(TypeError, match="source"):
        validate_meshing_source_closure({"source": callback})


def test_native_source_registry_refuses_numerical_substitution_under_cached_ids() -> None:
    import equinox as eqx

    from phydrax.lifecycle._meshing_sources import (
        register_meshing_source_artifacts,
        validate_meshing_source_closure,
    )

    register_meshing_source_artifacts()
    _, _, _, closure = native_source_closure()
    source = closure["source"]
    changed = eqx.tree_at(
        lambda value: value.geometry.state.values,
        source,
        replace_fn=lambda values: tuple(value + 0.125 for value in values),
    )
    assert changed.source_revision == source.source_revision
    closure["source"] = changed
    with pytest.raises(ValueError, match="stored_fields"):
        validate_meshing_source_closure(closure)


def test_native_curve_source_retains_plain_dataclass_coefficients() -> None:
    import jax.numpy as jnp

    from phydrax._model._structure import (
        model_from_array_recipe,
        model_recipe_array_values,
        model_structure_recipe,
    )
    from phydrax.geometry._source_fidelity import ParametricCurveBoundarySource
    from phydrax.geometry.brep._patches import BSplineCurve
    from phydrax.lifecycle._meshing_sources import (
        register_meshing_source_artifacts,
        validate_meshing_source_closure,
    )

    register_meshing_source_artifacts()
    curve = BSplineCurve(
        np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        np.asarray([1.0, 1.0], dtype=np.float64),
        np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
        1,
    )
    source = ParametricCurveBoundarySource(
        (curve,),
        ((0.0, 1.0),),
        source_id="native-line",
        source_revision="exact-line",
        covering_radius=0.1,
    )
    value = source, curve.bezier_pieces()
    validate_meshing_source_closure(value)
    recipe = model_structure_recipe(value)
    arrays = model_recipe_array_values(value, recipe, prefix="native-curve")
    restored_source, pieces = model_from_array_recipe(
        recipe, arrays, prefix="native-curve"
    )
    np.testing.assert_allclose(
        restored_source.curves[0].evaluate(jnp.asarray(0.5, dtype=jnp.float64)),
        [0.5, 0.0],
    )
    np.testing.assert_array_equal(
        pieces[0].homogeneous_controls, [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
    )
    assert not pieces[0].homogeneous_controls.flags.writeable
    distance = restored_source.boundary_distance(
        np.asarray([[0.5, 1.0]], dtype=np.float64)
    )
    assert distance.semantics == "certified"
    assert distance.lower[0] <= 1.0 <= distance.upper[0]


def _native_occurrence_authority(*, budget: CoordinateEnclosureBudget) -> tuple[Any, Any]:
    from phydrax.geometry.brep._constructors import (
        Builder,
        PlanarProfile,
        ProfileLoop,
        ProfilePlane,
        stage_extrusion,
    )
    from phydrax.geometry.brep._model import BRepGeometry, BRepOccurrence

    builder = Builder()
    stage_extrusion(
        builder,
        PlanarProfile(
            ProfilePlane(),
            ProfileLoop.polygon(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        ),
        np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        budget=budget,
    )
    occurrence = BRepOccurrence(
        ("assembly", "placed-solid"),
        0,
        translation=np.asarray((3.0, 4.0, 5.0), dtype=np.float64),
    )
    face_count = len(builder.faces)
    geometry = BRepGeometry(
        vertex_points=np.asarray(builder.vertices, dtype=np.float64),
        curves=tuple(builder.curves),
        edge_curves=tuple(builder.edge_curves),
        edge_ranges=np.asarray(builder.edge_ranges, dtype=np.float64),
        edge_vertices=tuple(builder.edge_vertices),
        pcurves=tuple(builder.pcurves),
        coedge_edges=tuple(builder.coedge_edges),
        coedge_senses=tuple(builder.coedge_senses),
        face_loops=tuple(
            tuple(tuple(loop) for loop in face.loops) for face in builder.faces
        ),
        shell_faces=(tuple(range(face_count)),),
        shell_orientations=((1,) * face_count,),
        solid_shells=((0,),),
        occurrences=(occurrence,),
    )
    return builder, geometry


def test_native_brep_source_retains_occurrence_qualified_association() -> None:
    import jax.numpy as jnp

    from phydrax._model._structure import (
        model_from_array_recipe,
        model_recipe_array_values,
        model_structure_recipe,
    )
    from phydrax.discretization._reference_cell_types import CellBlock
    from phydrax.geometry.brep._projection_contracts import brep_entity_id
    from phydrax.lifecycle._meshing_sources import (
        register_meshing_source_artifacts,
        validate_meshing_source_closure,
    )
    from phydrax.meshing._association_state import (
        GeometryAssociation,
        GeometryAssociationKind,
    )

    with coordinate_reference_budget() as budget:
        _, geometry = _native_occurrence_authority(budget=budget)
    occurrence = geometry.occurrences[0]
    revision = geometry.geometry_id
    entity_id = brep_entity_id(revision, 1, 0, occurrence_path=occurrence.path)
    edge_points = np.asarray(geometry.vertex_points)[list(geometry.edge_vertices[0])]
    target = CellMesh(
        occurrence.place(edge_points),
        (
            CellBlock(
                "placed-edge",
                "interval",
                np.asarray([[0, 1]], dtype=np.int32),
                global_ids=np.asarray([101], dtype=np.int64),
            ),
        ),
    )
    association = GeometryAssociation(
        GeometryAssociationKind.BREP,
        "native-placed-box",
        revision,
        target.entity_set(1).entity_set_id,
        target.entity_set(1).entity_ids,
        (entity_id,),
        np.asarray([0.0], dtype=np.float64),
        exact=True,
        source_dimensions=np.asarray([1], dtype=np.int8),
        source_indices=np.asarray([0], dtype=np.int64),
        source_occurrence_paths=(occurrence.path,),
    )
    value = geometry, association, ((occurrence.path, 1, 0, entity_id),)
    register_meshing_source_artifacts()
    validate_meshing_source_closure(value)
    recipe = model_structure_recipe(value)
    arrays = model_recipe_array_values(value, recipe, prefix="placed-source")
    restored, restored_association, qualified = model_from_array_recipe(
        recipe,
        arrays,
        prefix="placed-source",
    )
    assert restored_association.source_entities.occurrence_paths() == (
        ("assembly", "placed-solid"),
    )
    assert qualified == ((occurrence.path, 1, 0, entity_id),)
    first, last = restored.edge_ranges[0]
    midpoint = restored.curves[restored.edge_curves[0]].evaluate(
        jnp.asarray(0.5 * (first + last), dtype=jnp.float64),
    )
    start, end = geometry.edge_vertices[0]
    reference = 0.5 * (
        np.asarray(geometry.vertex_points[start])
        + np.asarray(geometry.vertex_points[end])
    )
    reference += np.asarray([3.0, 4.0, 5.0], dtype=np.float64)
    np.testing.assert_allclose(restored.occurrences[0].place(midpoint), reference)


def _native_spatial_root() -> Any:
    from phydrax.geometry.brep._intersection import CurveSurfaceIntersectionRoot
    from phydrax.geometry.brep._intersection_curve import SurfaceRegion
    from phydrax.geometry.brep._patches import LineCurve, PlanePatch

    plane = SurfaceRegion(
        PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        np.asarray([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float64),
    )
    return CurveSurfaceIntersectionRoot(
        LineCurve((0.0, 0.0, -1.0), (0.0, 0.0, 2.0)),
        plane,
        parameter_lower=np.asarray([0.4, -0.1, -0.1], dtype=np.float64),
        parameter_upper=np.asarray([0.6, 0.1, 0.1], dtype=np.float64),
    )


def test_restored_native_root_reestablishes_source_theorem() -> None:
    from phydrax._model._structure import (
        model_from_array_recipe,
        model_recipe_array_values,
        model_structure_recipe,
    )
    from phydrax.lifecycle._meshing_sources import (
        register_meshing_source_artifacts,
        validate_meshing_source_closure,
    )

    register_meshing_source_artifacts()
    root = _native_spatial_root()
    recipe = model_structure_recipe(root)
    restored = model_from_array_recipe(
        recipe,
        model_recipe_array_values(root, recipe, prefix="source-root"),
        prefix="source-root",
    )
    validate_meshing_source_closure(restored)
    point, error, certified = restored.evaluate()
    np.testing.assert_allclose(
        point, [0.0, 0.0, 0.0], atol=RESTORED_SPATIAL_ROOT_POINT_ATOL
    )
    assert certified and np.isfinite(error)


def test_native_root_refuses_cached_identity_with_changed_defining_ancestor() -> None:
    import equinox as eqx
    import jax.numpy as jnp

    from phydrax.lifecycle._meshing_sources import validate_meshing_source_closure

    root = _native_spatial_root()
    changed = eqx.tree_at(
        lambda value: value.surface.patch.origin,
        root,
        jnp.asarray([0.0, 0.0, 2.0], dtype=jnp.float64),
    )
    assert changed.root_id == root.root_id
    with pytest.raises(ValueError, match=r"CurveSurfaceIntersectionRoot\.parameter_box"):
        validate_meshing_source_closure(changed)


def _native_brep_compiled_source() -> Any:
    import jax.numpy as jnp

    from phydrax._physical import SpatialCoordinateContract
    from phydrax.geometry.brep._constructors import (
        brep_trim_domain,
        BRepTessellationPolicy,
    )
    from phydrax.geometry.brep._model import BRepImportReport, BRepModel
    from phydrax.geometry.brep._source import BRepSource

    with coordinate_reference_budget() as budget:
        builder, authority = _native_occurrence_authority(budget=budget)
    topology = authority.topology()
    contract = SpatialCoordinateContract.si()
    policy = BRepTessellationPolicy(
        linear_deflection=0.01,
        angular_deflection=0.1,
        trim_samples_per_edge=4,
    )
    vertices, triangles, face_ids, parameters = [], [], [], []
    for index, face in enumerate(builder.faces):
        lo, hi = face.box
        chart = np.asarray(
            (lo, (hi[0], lo[1]), hi, (lo[0], hi[1])),
            dtype=np.float64,
        )
        local = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
        if face.orientation < 0:
            local = local[:, ::-1]
        vertices.append(
            authority.occurrences[0].place(face.patch.evaluate(jnp.asarray(chart)))
        )
        triangles.append(local + 4 * index)
        face_ids.extend((index, index))
        parameters.append(chart[local])
    report = BRepImportReport(
        "native-placed-box",
        authority.geometry_id,
        "native",
        contract,
        policy.policy_id,
        topology.num_solids,
        topology.num_faces,
        topology.num_edges,
        topology.num_vertices,
        12,
        0.01,
        0.1,
        4,
        0,
    )
    model = BRepModel(
        patches=tuple(face.patch for face in builder.faces),
        parameter_bounds=np.asarray(
            [face.box for face in builder.faces], dtype=np.float64
        ),
        orientation=np.asarray(
            [face.orientation for face in builder.faces], dtype=np.float64
        ),
        trim_domains=tuple(
            brep_trim_domain(
                authority, index, face.patch, tolerance=0.01, parameter_bounds=face.box
            )
            for index, face in enumerate(builder.faces)
        ),
        topology=topology,
        coordinate_contract=contract,
        mesh_vertices=np.concatenate(vertices),
        mesh_faces=np.concatenate(triangles),
        triangle_face_ids=np.asarray(face_ids, dtype=np.int32),
        triangle_parameters=np.concatenate(parameters),
        tessellation_deviation_bounds=np.zeros((12,), dtype=np.float64),
        tessellation_normal_bounds=np.zeros((12,), dtype=np.float64),
        triangle_occurrence_ids=np.zeros((12,), dtype=np.int32),
        vertex_occurrence_ids=np.zeros((24,), dtype=np.int32),
        physical_tags=tuple(face.tag for face in builder.faces),
        report=report,
        geometry=authority,
    )
    return BRepSource(model).compile(coordinate_unit=METER)


def test_native_brep_compiled_source_restores_and_reprepares_actual_queries() -> None:
    from phydrax._array_archive import ArrayArchiveLimits
    from phydrax._model._structure import (
        model_from_array_recipe,
        model_recipe_array_values,
        model_structure_recipe,
    )
    from phydrax.lifecycle._meshing_sources import (
        register_meshing_source_artifacts,
        validate_meshing_source_closure,
    )

    register_meshing_source_artifacts()
    source = _native_brep_compiled_source()
    limits = ArrayArchiveLimits(max_members=1024, max_manifest_nesting=64)
    recipe = model_structure_recipe(source)
    restored = model_from_array_recipe(
        recipe,
        model_recipe_array_values(source, recipe, prefix="native-brep", limits=limits),
        prefix="native-brep",
        limits=limits,
    )
    validate_meshing_source_closure(restored, limits=limits)
    query = restored.kernel.query
    points = np.asarray([[3.2, 4.3, 5.4], [5.0, 4.3, 5.4]], dtype=np.float64)
    np.testing.assert_array_equal(query.contains(points).inside, [True, False])
    closest = query.closest_point(points[1:])
    np.testing.assert_allclose(
        closest.points, [[4.0, 4.3, 5.4]], atol=RESTORED_BREP_CLOSEST_POINT_ATOL
    )
    assert query.occurrences[0].path == ("assembly", "placed-solid")


def test_durable_native_source_archive_requeries_and_recertifies(tmp_path: Path) -> None:
    from phydrax._array_archive import ArrayArchiveLimits
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        recertify_restored_meshing_source,
        write_meshing_source_closure,
    )

    mesh, geometry, audit, records = native_source_closure()
    limits = ArrayArchiveLimits(max_members=512, max_manifest_nesting=64)
    receipt = write_meshing_source_closure(
        tmp_path / "source.zip", records, limits=limits
    )
    restored = read_meshing_source_closure(
        receipt.path,
        expected_content_id=receipt.content_id,
        limits=limits,
    )
    np.testing.assert_allclose(
        restored["source"]
        .boundary_distance(np.asarray([[2.0, 0.0, 0.0]], dtype=np.float64))
        .upper,
        [1.0],
        atol=RESTORED_SPHERE_DISTANCE_ATOL,
    )
    renewed = recertify_restored_meshing_source(
        restored["certification_inputs"],
        mesh,
        geometry,
        audit,
        archive_limits=limits,
    )
    assert renewed.passed
    assert renewed.fidelity is not None
    assert renewed.fidelity.status == "certified"
    assert (
        renewed.request.source_revision == records["certification_inputs"].source_revision
    )


def test_durable_source_archive_refuses_different_scientific_content(
    tmp_path: Path,
) -> None:
    from phydrax._array_archive import ArrayArchiveCorruptionError, ArrayArchiveLimits
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )

    _, _, _, records = native_source_closure()
    limits = ArrayArchiveLimits(max_members=512, max_manifest_nesting=64)
    receipt = write_meshing_source_closure(
        tmp_path / "source.zip", records, limits=limits
    )
    with pytest.raises(ArrayArchiveCorruptionError):
        read_meshing_source_closure(
            receipt.path,
            expected_content_id=canonical_fingerprint({"scientific-source": "other"}),
            limits=limits,
        )


# SOURCE CHECKPOINT ------------------------------------------------------------


def test_nonaddressable_native_source_checkpoint_restores_and_recertifies(
    tmp_path: Path,
) -> None:
    results = run_restart_workers(tmp_path, "native_source_checkpoint_restart")
    assert [
        {
            name: record[name]
            for name in ("process_index", "local_ids", "mesh_epoch", "source_recertified")
        }
        for record in results
    ] == [
        {
            "process_index": 0,
            "local_ids": [3, 4, 5],
            "mesh_epoch": 7,
            "source_recertified": True,
        },
        {
            "process_index": 1,
            "local_ids": [0, 1, 2],
            "mesh_epoch": 7,
            "source_recertified": True,
        },
    ]
    for record in results:
        assert record["nonaddressable"]
        assert record["admitted"]
        assert record["source_closure_id_preserved"]
        assert record["source_owner_indices"] == [0, 1]
        assert record["source_owner_state_ids"] == record["expected_owner_state_ids"]
        assert record["owner_closures_retained"]
        assert record["source_restored"]
        np.testing.assert_allclose(record["boundary_distance"], [1.0])
        assert record["report_id_preserved"]
        _assert_restored_scientific_arrays(record)


def test_native_numpy_source_checkpoint_restores_immutable_authored_coefficients(
    tmp_path: Path,
) -> None:
    import phydrax as phx
    from phydrax._model._structure import model_recipe_array_inventory
    from phydrax.geometry._domain_coverage import PiecewiseLinearDomain

    points = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64
    )
    facets = np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=np.int64)
    domain = PiecewiseLinearDomain(
        points,
        facets,
        np.asarray(((0, -1),) * 4, dtype=np.int64),
        ("native-material",),
        source_id="native-planar-domain",
    )
    mesh = phx.discretization.CellMesh.from_triangles(
        points, np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
    )
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    audit = phx.meshing.audit_cell_mesh(
        mesh,
        geometry,
        policy=phx.meshing.CellMeshAuditPolicy(
            watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT
        ),
    )
    report = phx.meshing.certify_meshing_acceptance(
        mesh,
        geometry,
        audit,
        schedule=phx.meshing.MeshCertificationSchedule("volume_plc"),
        domain=domain,
        cell_regions=np.asarray((0, 0), dtype=np.int64),
    )
    report.require_passed()
    closure = {
        "certification_inputs": report.request,
        "report": report,
        "associations": (),
    }
    state = native_source_state(closure)
    reference = native_source_arrays(mesh, state.mesh_epoch)
    arrays = {name: jax.device_put(value) for name, value in reference.items()}
    assert state.source_recipe_json is not None
    inventory = model_recipe_array_inventory(
        json.loads(state.source_recipe_json), prefix="source-record"
    )
    host_entries = tuple(entry for entry in inventory if entry.backend == "numpy")
    assert host_entries
    sharding = arrays["vertex_ids"].sharding
    shardings = {name: sharding for name in arrays}
    shardings.update(
        {
            state.source_array_name(entry.name): sharding
            for entry in inventory
            if entry.backend != "numpy"
        }
    )
    repository = build_repository(
        tmp_path / "native-domain-checkpoint",
        maximum_chunk_bytes=16 * 1024,
        maximum_metadata_bytes=1024 * 1024,
    )
    publish_process_meshing_checkpoint(
        repository,
        "native-domain",
        "domain-execution",
        state,
        arrays,
        analysis_plan_id="domain-analysis",
        numeric_revision_id="domain-revision",
        writer_id="domain-writer",
    )
    manifest = assemble_distributed_checkpoint_from_repository(
        repository,
        "native-domain",
        "domain-analysis",
        "domain-revision",
        "domain-execution",
        expected_process_count=1,
    )
    checkpoint = restore_meshing_checkpoint(
        repository,
        manifest,
        shardings,
        restart_relation(),
        TopologyRestartPolicy(allow_topology_change=True),
        expected_state_id=state.checkpoint_state_id,
    )
    restored_state = checkpoint.state
    restored_arrays = checkpoint.arrays
    admission = checkpoint.admission
    assert admission.admitted
    restored = restored_state.source_closure
    assert restored is not None
    retained = restored["certification_inputs"]
    np.testing.assert_array_equal(retained.domain.vertices, points)
    np.testing.assert_array_equal(retained.domain.facets, facets)
    assert not retained.domain.vertices.flags.writeable
    coverage = phx.geometry.certify_domain_coverage(
        mesh,
        geometry,
        retained.domain,
        retained.cell_regions,
        embedding=restored["report"].embedding,
    )
    assert coverage.status == "certified"
    np.testing.assert_allclose(
        np.asarray(coverage.achieved_region_measures),
        np.asarray((1.0,), dtype=np.float64),
    )
    renewed = phx.meshing.certify_meshing_acceptance(
        mesh,
        geometry,
        audit,
        schedule=retained.schedule,
        domain=retained.domain,
        cell_regions=retained.cell_regions,
        limits=retained.limits,
    )
    renewed.require_passed()
    assert renewed.report_id == report.report_id
    for entry in host_entries:
        value = restored_arrays[state.source_array_name(entry.name)]
        assert isinstance(value, np.ndarray) and not value.flags.writeable
    for name, expected in reference.items():
        np.testing.assert_array_equal(np.asarray(restored_arrays[name]), expected)


def test_native_projection_source_checkpoint_preserves_raw_evidence_and_recertifies(
    tmp_path: Path,
) -> None:
    import phydrax as phx
    from phydrax._model._structure import model_recipe_array_inventory
    from phydrax.geometry._source_fidelity import ImplicitProjectionBoundarySource
    from phydrax.geometry.implicit._analytic_profile import AnalyticImplicitProfile

    mesh, geometry, audit, original = native_source_closure()
    profile = AnalyticImplicitProfile(
        original["source"].geometry,
        phx.SpatialCoordinateContract.si(),
        tube_radius=0.49,
        source_id="native-profile-sphere",
    )
    source = ImplicitProjectionBoundarySource(profile)
    report = phx.meshing.certify_meshing_acceptance(
        mesh,
        geometry,
        audit,
        schedule=phx.meshing.MeshCertificationSchedule("surface"),
        source=source,
        fidelity_tolerance=0.8,
    )
    report.require_passed()
    closure = {
        "source": source,
        "certification_inputs": report.request,
        "report": report,
        "associations": (),
    }
    state = native_source_state(closure)
    reference = native_source_arrays(mesh, state.mesh_epoch)
    arrays = {name: jax.device_put(value) for name, value in reference.items()}
    assert state.source_recipe_json is not None
    inventory = model_recipe_array_inventory(
        json.loads(state.source_recipe_json), prefix="source-record"
    )
    sharding = arrays["vertex_ids"].sharding
    shardings = {name: sharding for name in arrays}
    shardings.update(
        {
            state.source_array_name(entry.name): sharding
            for entry in inventory
            if entry.backend != "numpy"
        }
    )
    repository = build_repository(
        tmp_path / "native-projection-checkpoint",
        maximum_chunk_bytes=16 * 1024,
        maximum_metadata_bytes=1024 * 1024,
    )
    publish_process_meshing_checkpoint(
        repository,
        "native-projection",
        "projection-execution",
        state,
        arrays,
        analysis_plan_id="projection-analysis",
        numeric_revision_id="projection-revision",
        writer_id="projection-writer",
    )
    manifest = assemble_distributed_checkpoint_from_repository(
        repository,
        "native-projection",
        "projection-analysis",
        "projection-revision",
        "projection-execution",
        expected_process_count=1,
    )
    checkpoint = restore_meshing_checkpoint(
        repository,
        manifest,
        shardings,
        restart_relation(),
        TopologyRestartPolicy(allow_topology_change=True),
        expected_state_id=state.checkpoint_state_id,
    )
    restored_state = checkpoint.state
    restored_arrays = checkpoint.arrays
    admission = checkpoint.admission
    assert admission.admitted
    restored = restored_state.source_closure
    assert restored is not None
    retained = restored["certification_inputs"]
    np.testing.assert_array_equal(retained.source.profile.center, profile.center)
    assert not retained.source.profile.center.flags.writeable
    old_fidelity = report.fidelity
    new_fidelity = restored["report"].fidelity
    assert old_fidelity is not None and new_fidelity is not None
    old_evidence = old_fidelity.projection_coverage
    new_evidence = new_fidelity.projection_coverage
    assert old_evidence is not None and new_evidence is not None
    assert new_evidence.evidence_id == old_evidence.evidence_id
    np.testing.assert_array_equal(new_evidence.facets, old_evidence.facets)
    np.testing.assert_array_equal(new_evidence.field_bounds, old_evidence.field_bounds)
    np.testing.assert_array_equal(
        new_evidence.gradient_bounds, old_evidence.gradient_bounds
    )
    assert new_evidence.fiber_endpoints == old_evidence.fiber_endpoints
    assert new_evidence.fiber_crossing_classes == old_evidence.fiber_crossing_classes
    assert (
        new_evidence.fiber_crossing_parameters == old_evidence.fiber_crossing_parameters
    )
    distance = retained.source.boundary_distance(
        np.asarray(((2.0, 0.0, 0.0),), dtype=np.float64)
    )
    assert distance.semantics == "certified"
    np.testing.assert_allclose(
        np.asarray(distance.upper), np.asarray((1.0,), dtype=np.float64)
    )
    renewed = phx.meshing.certify_meshing_acceptance(
        mesh,
        geometry,
        audit,
        schedule=retained.schedule,
        source=retained.source,
        fidelity_tolerance=retained.fidelity_tolerance,
        fidelity_sample_order=retained.fidelity_sample_order,
        limits=retained.limits,
    )
    renewed.require_passed()
    assert renewed.report_id == report.report_id
    for name, expected in reference.items():
        np.testing.assert_array_equal(np.asarray(restored_arrays[name]), expected)


# NATIVE FAMILY ---------------------------------------------------------------


@pytest.fixture(
    scope="module", params=("plc", "structured", "mapped", "polyhedral", "surface")
)
def durable_native_family(
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
) -> tuple[Any, ...]:
    family = request.param
    tmp_path = tmp_path_factory.mktemp(family)
    from phydrax._array_archive import ArrayArchiveLimits
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        recertify_restored_meshing_source,
        write_meshing_source_closure,
    )

    part, source, specification, options = native_family_generation(family)
    original_carrier = part.carrier
    assert isinstance(original_carrier, CellMeshingResult)
    report = original_carrier.certification
    assert report is not None
    closure = {
        "certification_inputs": report.request,
        "report": report,
        "associations": original_carrier.associations,
        "generation_part": part,
        "generation_source": source,
        "generation_specification": specification,
        "generation_options": options,
    }
    limits = ArrayArchiveLimits(
        max_members=4096, max_manifest_nesting=128, max_manifest_bytes=1 << 26
    )
    receipt = write_meshing_source_closure(
        tmp_path / f"{family}.zip", closure, limits=limits
    )
    restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id, limits=limits
    )
    result = restored["generation_part"].carrier
    assert isinstance(result, CellMeshingResult)
    renewed = recertify_restored_meshing_source(
        restored["certification_inputs"],
        result.mesh,
        result.geometry,
        result.audit,
        archive_limits=limits,
    )
    return family, source, original_carrier, report, restored, result, renewed


def test_native_family_durable_restart_recertifies(
    durable_native_family: tuple[Any, ...],
) -> None:
    _, _, _, report, restored, result, renewed = durable_native_family
    assert renewed.report_id == report.report_id
    assert (
        restored["generation_specification"].specification_id
        == result.compliance.specification_id
    )


def test_native_family_durable_restart_requeries_source(
    durable_native_family: tuple[Any, ...],
) -> None:
    _, source, _, report, _, _, renewed = durable_native_family
    if renewed.request.source is not None:
        source_query = renewed.request.source
        probe = np.full((1, source_query.ambient_dimension), 3.0, dtype=np.float64)
        original_source = report.request.source
        assert original_source is not None
        original_distance = original_source.boundary_distance(probe)
        restored_distance = source_query.boundary_distance(probe)
        np.testing.assert_array_equal(restored_distance.lower, original_distance.lower)
        np.testing.assert_array_equal(restored_distance.upper, original_distance.upper)
    else:
        # PLC has an independently authored oriented domain, not an invented
        # implicit/planar distance query. Fresh coverage uses its declared facets.
        assert renewed.coverage is not None
        assert isinstance(renewed.request.domain, PiecewiseLinearDomain)
        assert isinstance(source, (NativePlcSource, NativePolyhedralSource))
        assert renewed.coverage.status == "certified"
        np.testing.assert_array_equal(
            renewed.request.domain.vertices, source.complex.vertices
        )


@pytest.mark.parametrize("durable_native_family", ("polyhedral",), indirect=True)
def test_native_family_durable_restart_prepares_finite_volume(
    durable_native_family: tuple[Any, ...],
) -> None:
    from phydrax.discretization.finite_volume._polyhedral import (
        prepare_polyhedral_finite_volume_geometry,
    )

    _, source, original_carrier, _, restored, result, _ = durable_native_family
    assert isinstance(source, NativePolyhedralSource)
    fv = prepare_polyhedral_finite_volume_geometry(result.mesh)
    np.testing.assert_allclose(
        np.sum(fv.cell_volumes), 1.0, atol=RESTORED_POLYHEDRAL_VOLUME_SUM_ATOL
    )
    np.testing.assert_array_equal(restored["generation_source"].weights, source.weights)
    assert not restored["generation_source"].weights.flags.writeable
    connectivity = result.mesh.connectivity
    original = original_carrier.mesh.connectivity
    for name in (
        "face_vertex_offsets",
        "face_vertex_values",
        "cell_face_values",
        "cell_face_sign_values",
    ):
        np.testing.assert_array_equal(
            getattr(connectivity, name), getattr(original, name)
        )


@pytest.fixture(scope="module")
def durable_native_field(durable_native_family: tuple[Any, ...]) -> tuple[Any, ...]:
    import phydrax as phx

    family, _, _, _, _, result, _ = durable_native_family
    # Mapped Q2 coordinates require a Q2 field; the remaining carriers are P1.
    degree = {"mapped": 2, "plc": 1, "structured": 1, "surface": 1}[family]
    field = phx.discretization.FiniteElementFieldSpec(
        "u",
        {
            block.name: phx.discretization.lagrange_element(block.cell_kind, degree)
            for block in result.mesh.blocks
        },
    )
    space = phx.discretization.FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry
    ).prepare()
    nodes = np.asarray(space.dof_maps[0].dof_coordinates)
    weights = np.arange(1, nodes.shape[1] + 1, dtype=np.float64)
    return result, space, weights, 2.0 + nodes @ weights


@pytest.mark.parametrize(
    "durable_native_family", ("plc", "structured", "mapped"), indirect=True
)
def test_native_family_durable_restart_reconstructs_volume_field(
    durable_native_field: tuple[Any, ...],
) -> None:
    from phydrax.discretization.fem._point_interpolation import (
        prepare_finite_element_field_reconstruction,
    )

    result, space, weights, coefficients = durable_native_field
    for block in result.mesh.blocks:
        sites = np.mean(
            np.asarray(result.mesh.coordinates)[np.asarray(block.vertices)], axis=1
        )
        query = prepare_finite_element_field_reconstruction(
            space,
            "u",
            block_name=block.name,
        ).evaluate(coefficients, sites)
        np.testing.assert_allclose(
            query.values, 2.0 + sites @ weights, atol=RESTORED_VOLUME_AFFINE_FIELD_ATOL
        )


@pytest.mark.parametrize("durable_native_family", ("surface",), indirect=True)
def test_native_family_durable_restart_evaluates_intrinsic_surface_field(
    durable_native_field: tuple[Any, ...],
) -> None:
    from phydrax.discretization._reference_cell import reference_cell_topology

    result, space, weights, coefficients = durable_native_field
    for index, block in enumerate(result.mesh.blocks):
        reference = np.mean(
            np.asarray(reference_cell_topology(block.cell_kind).vertices), axis=0
        )[None]
        local = space.evaluate_block_geometry(
            "u",
            index,
            space.default_runtime.coordinates,
            reference,
            np.ones((1,)),
        )
        values = np.einsum(
            "ql,cl->cq",
            np.asarray(local.basis_values),
            coefficients[np.asarray(space.dof_maps[0].cell_dofs[index])],
        )
        np.testing.assert_allclose(
            values,
            2.0 + np.asarray(local.physical_points) @ weights,
            atol=RESTORED_SURFACE_AFFINE_FIELD_ATOL,
        )


@pytest.mark.parametrize("durable_native_family", ("mapped",), indirect=True)
def test_native_family_durable_restart_preserves_mapped_coordinate_source(
    durable_native_family: tuple[Any, ...],
) -> None:
    _, _, original_carrier, _, _, result, _ = durable_native_family
    assert result.geometry.restriction_source is not None
    np.testing.assert_array_equal(
        result.geometry.coordinates, original_carrier.geometry.coordinates
    )


def test_native_family_refuses_changed_source_power_weights_and_hard_request() -> None:
    import equinox as eqx

    from phydrax._array_archive import ArrayArchiveLimits
    from phydrax.lifecycle._meshing_sources import (
        register_meshing_source_artifacts,
        validate_meshing_source_closure,
        validate_registered_native_part,
    )

    register_meshing_source_artifacts()
    part, source, specification, options = native_family_generation("polyhedral")
    assert isinstance(source, NativePolyhedralSource)
    assert isinstance(specification, VolumeMeshingSpec)
    forged = eqx.tree_at(lambda value: value.weights, source, source.weights + 0.1)
    with pytest.raises(
        ValueError,
        match=r"NativePolyhedralSource\.binding_id",
    ):
        validate_meshing_source_closure(forged)
    import phydrax as phx

    altered = phx.meshing.VolumeMeshingSpec(
        specification.target,
        specification.boundary_scope,
        specification.fill_strategy,
        size_controls=(
            phx.meshing.UniformSizeControl(
                specification.boundary_scope,
                5.0,
                strength=phx.meshing.SizeControlStrength.SOFT,
            ),
        ),
    )
    with pytest.raises(ValueError, match=r"source_closure\.generation_options"):
        validate_registered_native_part(
            part, source, altered, options, limits=ArrayArchiveLimits()
        )


def test_native_family_refuses_changed_original_algorithm_schedule() -> None:
    import phydrax as phx
    from phydrax._array_archive import ArrayArchiveLimits
    from phydrax.lifecycle._meshing_sources import validate_registered_native_part

    part, source, specification, _ = native_family_generation("mapped")
    options = phx.meshing.NativeMeshingOptions(
        "mapped_balanced_grid_hex",
        grid_schedule=phx.meshing.NativeHexGridSchedule("balanced_grid", maximum_depth=9),
    )
    with pytest.raises(ValueError, match=r"source_closure\.generation_options"):
        validate_registered_native_part(
            part,
            source,
            specification,
            options,
            limits=ArrayArchiveLimits(max_members=4096, max_manifest_nesting=128),
        )


@pytest.mark.parametrize("carrier_kind", ("mixed-q2", "polyhedral-widths"))
def test_native_whole_carrier_restart_preserves_exact_maps_faces_and_queries(
    tmp_path: Path,
    carrier_kind: str,
) -> None:
    import jax.numpy as jnp

    import phydrax as phx
    from phydrax._array_archive import ArrayArchiveLimits
    from phydrax.discretization._reference_cell import reference_cell_topology
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )

    part: MeshPart

    if carrier_kind == "mixed-q2":
        from tests._support.overset_mapped_sources import mapped_source

        part = mapped_source(("hexahedron", "prism"))
    else:
        box, tet = (
            reference_cell_topology(kind) for kind in ("hexahedron", "tetrahedron")
        )
        points = np.concatenate(
            (np.asarray(box.vertices), np.asarray(tet.vertices) + (2.0, 0.0, 0.0))
        )
        faces = {
            "box": (tuple(np.asarray(face, dtype=np.int64) for face in box.entities[2]),),
            "tetra": (
                tuple(np.asarray(face, dtype=np.int64) + 8 for face in tet.entities[2]),
            ),
        }
        mesh = CellMesh.from_mixed_3d(
            points,
            (),
            polyhedra=faces,
            polyhedral_cell_global_ids={
                "box": np.asarray((107,)),
                "tetra": np.asarray((901,)),
            },
        )
        part = phx.meshing.MeshPart(
            "original-polyhedra",
            phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si()),
        )
    original_carrier = part.carrier
    assert isinstance(original_carrier, CellMeshingResult)
    limits = ArrayArchiveLimits(
        max_members=4096, max_manifest_nesting=128, max_manifest_bytes=1 << 26
    )
    # These represented carriers are distinct from provider
    # requests. Source-free carrier certification has a separate role.
    receipt = write_meshing_source_closure(
        tmp_path / f"{carrier_kind}.zip", part, limits=limits
    )
    restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id, limits=limits
    )
    assert isinstance(restored, MeshPart)
    result = restored.carrier
    assert isinstance(result, CellMeshingResult)
    audit = phx.meshing.audit_cell_mesh(result.mesh, result.geometry)
    audit.require_passed()
    audit.require_decided()
    assert audit.report_id == original_carrier.audit.report_id
    np.testing.assert_array_equal(
        result.geometry.coordinates, original_carrier.geometry.coordinates
    )
    for before, after in zip(
        original_carrier.geometry.geometry_dofs,
        result.geometry.geometry_dofs,
        strict=True,
    ):
        np.testing.assert_array_equal(after, before)
    if carrier_kind == "mixed-q2":
        from phydrax.discretization.fem._point_interpolation import (
            prepare_finite_element_field_reconstruction,
        )

        field = phx.discretization.FiniteElementFieldSpec(
            "u",
            {
                block.name: phx.discretization.lagrange_element(block.cell_kind, 2)
                for block in result.mesh.blocks
            },
        )
        prepared = phx.discretization.FiniteElementPlan(
            result.mesh,
            field,
            coordinate_spec=result.geometry,
        ).prepare()
        nodes = np.asarray(prepared.dof_maps[0].dof_coordinates)
        coefficients = jnp.asarray(
            2.0 + nodes[:, 0] - 3.0 * nodes[:, 1] + 0.5 * nodes[:, 2]
        )
        for block in result.mesh.blocks:
            points = np.mean(
                np.asarray(result.mesh.coordinates)[np.asarray(block.vertices)], axis=1
            )
            evaluated = prepare_finite_element_field_reconstruction(
                prepared,
                "u",
                block_name=block.name,
            ).evaluate(coefficients, points)
            np.testing.assert_allclose(
                evaluated.values,
                2.0 + points[:, 0] - 3.0 * points[:, 1] + 0.5 * points[:, 2],
                atol=RESTORED_VOLUME_AFFINE_FIELD_ATOL,
            )
        for element in result.geometry.elements:
            assert isinstance(element, (FiniteElementSpec, RestrictedCellGeometryElement))
            assert element.degree == 2
    else:
        from phydrax.discretization._polyhedral_locator import (
            PreparedPolyhedralCellLocator,
        )
        from phydrax.discretization.finite_volume._polyhedral import (
            prepare_polyhedral_finite_volume_geometry,
        )

        fv = prepare_polyhedral_finite_volume_geometry(result.mesh)
        np.testing.assert_allclose(
            np.sum(fv.cell_volumes), 7.0 / 6.0, atol=RESTORED_POLYHEDRAL_VOLUME_SUM_ATOL
        )
        location = PreparedPolyhedralCellLocator(result.mesh).locate(
            np.asarray(((0.5, 0.5, 0.5), (2.25, 0.25, 0.25))),
        )
        np.testing.assert_array_equal(location.successful, (True, True))
        scientific_ids = np.concatenate(
            tuple(np.asarray(block.global_ids) for block in result.mesh.blocks)
        )
        np.testing.assert_array_equal(
            scientific_ids[np.asarray(location.cell_ids)], (107, 901)
        )
        assert {element.local_dof_count for element in result.geometry.elements} == {4, 8}
        for name in (
            "face_vertex_offsets",
            "face_vertex_values",
            "cell_face_values",
            "cell_face_sign_values",
        ):
            np.testing.assert_array_equal(
                getattr(result.mesh.connectivity, name),
                getattr(original_carrier.mesh.connectivity, name),
            )


def _implicit_association_transfer() -> Any:
    import phydrax as phx
    from phydrax.geometry.implicit._analytic_profile import AnalyticImplicitProfile
    from phydrax.meshing._implicit_association_transfer import ImplicitAssociationTransfer

    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 0.371, feature_id="archive-sphere"
    ).compile(coordinate_unit=METER)
    profile = AnalyticImplicitProfile(
        geometry, phx.SpatialCoordinateContract.si(), cover_radius=0.05
    )
    return ImplicitAssociationTransfer(profile, maximum_queries=20000)


def test_implicit_association_transfer_archive_restores_current_source_authority(
    tmp_path: Path,
) -> None:
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )
    from phydrax.meshing._implicit_association_transfer import ImplicitAssociationTransfer

    transfer = _implicit_association_transfer()
    receipt = write_meshing_source_closure(tmp_path / "implicit-transfer", transfer)
    restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )

    assert type(restored) is ImplicitAssociationTransfer
    restored.profile.validate_source_integrity()
    assert restored.transfer_id == transfer.transfer_id
    points = np.asarray(
        ((0.0, 0.0, 0.0), (0.371, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=np.float64
    )
    np.testing.assert_allclose(
        restored.profile.boundary_distance(points).upper,
        (0.371, 0.0, 0.629),
        rtol=0.0,
        atol=RESTORED_ANALYTIC_PROFILE_DISTANCE_ATOL,
    )


@pytest.mark.parametrize(
    "malformation",
    (
        "cycle",
        "forward-reference",
        "unknown-type",
        "extra-item",
        "array-rank",
        "array-extent",
        "wrong-bank-dtype",
    ),
)
def test_source_archive_refuses_malformed_manifest_before_reading_members(
    tmp_path: Path,
    malformation: str,
) -> None:
    import zipfile

    from phydrax._array_archive import (
        ArrayArchiveCorruptionError,
        DEFAULT_ARRAY_ARCHIVE_LIMITS,
    )
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )

    receipt = write_meshing_source_closure(
        tmp_path / "source", _implicit_association_transfer()
    )
    with zipfile.ZipFile(receipt.path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(members["manifest.json"])
    recipe = json.loads(manifest["recipe"])
    root = recipe["nodes"][recipe["root"]]
    match malformation:
        case "cycle":
            root["items"][0] = recipe["root"]
        case "forward-reference":
            root["items"][0] = len(recipe["nodes"])
        case "unknown-type":
            root["type"] = "unregistered:source"
        case "extra-item":
            # An entry no registered source field owns.
            root["items"].append(0)
        case "array-rank" | "array-extent":
            array = next(node for node in recipe["nodes"] if node["kind"] == "array")
            array["shape"] = (
                [1] * (DEFAULT_ARRAY_ARCHIVE_LIMITS.max_array_rank + 1)
                if malformation == "array-rank"
                else [DEFAULT_ARRAY_ARCHIVE_LIMITS.max_axis_length + 1]
            )
        case "wrong-bank-dtype":
            first_bank = next(iter(manifest["arrays"].values()))
            first_bank["dtype"] = "uint8" if first_bank["dtype"] != "uint8" else "float64"
        case _:
            raise AssertionError(
                f"fixture.malformation must name a manifest corruption; got {malformation!r}"
            )
    manifest["recipe"] = json.dumps(recipe, separators=(",", ":"), sort_keys=True)
    members["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode(
        "utf-8"
    )
    # A numerical payload is deliberately corrupt too. Manifest admission must
    # report its own refusal, not reach this member's checksum or NPY decoder.
    numerical_member = next(iter(manifest["arrays"].values()))["member"]
    members[numerical_member] = b"invalid numerical payload"
    with zipfile.ZipFile(receipt.path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)

    expected = (
        "bank inventory is invalid"
        if malformation == "wrong-bank-dtype"
        else "recipe is invalid"
    )
    with pytest.raises(ArrayArchiveCorruptionError, match=expected):
        read_meshing_source_closure(receipt.path, expected_content_id=receipt.content_id)


def test_plc_restart_reprepares_with_canonical_limits_and_fresh_scope_ownership(
    tmp_path: Path,
) -> None:
    from phydrax._array_archive import ArrayArchiveLimits
    from phydrax._meshcore import meshcore_available, MeshcoreStatus, NativeExecutionScope
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )
    from phydrax.meshing._native_preparation import native_preparation_ledger
    from phydrax.meshing._volume_generation import prepare_plc_source

    if not meshcore_available():
        pytest.skip("native meshcore library is unavailable")
    part, source, specification, options = native_family_generation("plc")
    assert isinstance(source, NativePlcSource)
    assert isinstance(specification, VolumeMeshingSpec)
    assert isinstance(part.carrier, CellMeshingResult)
    report = part.carrier.certification
    assert report is not None
    archive_limits = ArrayArchiveLimits(
        max_members=4096, max_manifest_nesting=128, max_manifest_bytes=1 << 26
    )
    receipt = write_meshing_source_closure(
        tmp_path / "scoped-plc",
        {
            "certification_inputs": report.request,
            "report": report,
            "associations": part.carrier.associations,
            "generation_part": part,
            "generation_source": source,
            "generation_specification": specification,
            "generation_options": options,
        },
        limits=archive_limits,
    )
    restored = read_meshing_source_closure(
        receipt.path,
        expected_content_id=receipt.content_id,
        limits=archive_limits,
    )
    restored_source = restored["generation_source"]
    restored_specification = restored["generation_specification"]
    assert isinstance(restored_source, NativePlcSource)
    assert isinstance(restored_specification, VolumeMeshingSpec)
    assert restored_source.source_id == source.source_id
    assert restored_source.source_revision == source.source_revision
    assert restored_source.complex.complex_id == source.complex.complex_id
    assert restored_specification.specification_id == specification.specification_id
    assert restored_specification.limits.limits_id == specification.limits.limits_id
    limits = specification.limits
    lowered = limits.native_limits()
    assert (
        lowered.maximum_work_units,
        lowered.maximum_geometry_queries,
        lowered.maximum_cavity_cells,
        lowered.maximum_managed_bytes,
        lowered.maximum_seconds,
    ) == (
        limits.maximum_work_units,
        limits.maximum_geometry_queries,
        limits.maximum_cavity_cells,
        limits.maximum_scratch_bytes,
        limits.maximum_wall_seconds,
    )
    predecessor = NativeExecutionScope(lowered)
    with predecessor, predecessor.host_workspace():
        original_preparation = prepare_plc_source(
            source.complex,
            source.source_id,
            source.source_revision,
            part.carrier.coordinate_contract,
            limits=limits,
            scope=predecessor,
        )
        original_ledger = native_preparation_ledger(predecessor)
        original_ledger.remember_live(predecessor, original_preparation)
        assert original_ledger.live_contains(original_preparation)
        assert predecessor.remaining().remaining_work_units < limits.maximum_work_units
    assert predecessor.completed_evidence().status is MeshcoreStatus.OK
    assert not original_ledger.live_contains(original_preparation)

    successor = NativeExecutionScope(restored_specification.limits.native_limits())
    with successor, successor.host_workspace():
        restored_ledger = native_preparation_ledger(successor)
        assert restored_ledger is not original_ledger
        assert not restored_ledger.live_contains(original_preparation)
        renewed = prepare_plc_source(
            restored_source.complex,
            restored_source.source_id,
            restored_source.source_revision,
            part.carrier.coordinate_contract,
            limits=restored_specification.limits,
            scope=successor,
        )
        assert renewed is not original_preparation
        assert renewed.prepared_id == original_preparation.prepared_id
        assert (
            renewed.association_transfer.transfer_id
            == original_preparation.association_transfer.transfer_id
        )
        np.testing.assert_array_equal(
            renewed.input_polygons, original_preparation.input_polygons
        )
        np.testing.assert_array_equal(
            renewed.association_transfer.triangle_vertices,
            original_preparation.association_transfer.triangle_vertices,
        )
        restored_ledger.remember_live(successor, renewed)
        assert restored_ledger.live_contains(renewed)
        assert not restored_ledger.live_contains(original_preparation)
        assert (
            successor.remaining().remaining_work_units
            < restored_specification.limits.maximum_work_units
        )
        assert successor.parent is None
    assert successor.completed_evidence().status is MeshcoreStatus.OK
    assert not restored_ledger.live_contains(renewed)
    assert predecessor.limits is lowered
