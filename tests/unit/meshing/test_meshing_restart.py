#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

import jax
import numpy as np
import pytest
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from phydrax._fingerprint import canonical_fingerprint
from phydrax.discretization._cell_geometry import (
    CellGeometrySpec,
    RestrictedCellGeometryElement,
)
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization.fem._reference import FiniteElementSpec
from phydrax.geometry._mesh_certificates import PiecewiseLinearDomain
from phydrax.lifecycle._chunk_repository import RepositoryCorruptionError
from phydrax.lifecycle._distributed_checkpoint import (
    assemble_distributed_checkpoint_from_repository,
    publish_process_meshing_checkpoint,
    restore_meshing_checkpoint,
)
from phydrax.lifecycle._meshing_source_families import NativeGenerationSource
from phydrax.lifecycle._models import CheckpointManifest
from phydrax.lifecycle._repository import (
    HPCFilesystemProfile,
    ObjectNotFoundError,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
)
from phydrax.lifecycle._restart_topology import (
    MeshingCheckpointState,
    TopologyRestartPolicy,
    TopologyRestartRelation,
)
from phydrax.meshing._assembly import MeshPart
from phydrax.meshing._audit import CellMeshAuditReport
from phydrax.meshing._contracts import SurfaceMeshingSpec, VolumeMeshingSpec
from phydrax.meshing._result import CellMeshingResult
from phydrax.meshing.providers._native_options import (
    NativeMeshingOptions,
    NativeMeshingRoute,
)
from phydrax.meshing.providers._native_sources import (
    NativePlcSource,
    NativePolyhedralSource,
)


def _repository(
    path: Path,
    *,
    maximum_chunk_bytes: int = 64,
    maximum_metadata_bytes: int = 64 * 1024,
) -> POSIXArtifactRepository:
    profile = HPCFilesystemProfile(
        "posix.meshing-restart",
        "local-posix",
        atomic_rename_same_filesystem=True,
        file_fsync=True,
        directory_fsync=True,
        advisory_locking=True,
        attempt_private_staging=True,
    )
    return POSIXArtifactRepository(
        path,
        POSIXRepositoryPolicy(
            profile,
            maximum_chunk_bytes=maximum_chunk_bytes,
            maximum_metadata_bytes=maximum_metadata_bytes,
        ),
    )


def _state() -> MeshingCheckpointState:
    return MeshingCheckpointState(
        source_topology_id="source-topology",
        source_geometry_id="source-geometry",
        topology_id="accepted-topology",
        geometry_id="accepted-geometry",
        result_id="accepted-result",
        source_revision_id="source-revision",
        coordinate_contract_id="affine-triangle-coordinates",
        lineage_id="accepted-lineage",
        epoch_id="accepted-epoch",
        evidence_id="collective-evidence",
        transfer_id="accepted-transfer",
        request_id="meshing-request",
        mesh_epoch=7,
        array_roles=(
            ("vertex_ids", "entity-ids"),
            ("coordinates", "geometry"),
            ("cell_vertex_ids", "topology"),
            ("parent_ids", "lineage"),
            ("accepted_epochs", "epoch"),
            ("certificate_verdicts", "evidence"),
            ("material_inventory", "state"),
        ),
    )


def _scientific_arrays() -> dict[str, np.ndarray]:
    # Nonconsecutive IDs make accidental reuse of old local slots observable.
    return {
        "vertex_ids": np.asarray([101, 509, 1003, 2401], dtype=np.int64),
        "coordinates": np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64
        ),
        "cell_vertex_ids": np.asarray(
            [[101, 509, 1003], [509, 2401, 1003]], dtype=np.int64
        ),
        "parent_ids": np.asarray(
            [[101, 11], [509, 11], [1003, 23], [2401, 23]], dtype=np.int64
        ),
        "accepted_epochs": np.asarray([6, 7], dtype=np.int64),
        "certificate_verdicts": np.asarray([1, 1], dtype=np.uint8),
        "material_inventory": np.asarray([2.0, 3.0, 5.0, 7.0], dtype=np.float64),
    }


def _relation() -> TopologyRestartRelation:
    return TopologyRestartRelation(
        canonical_fingerprint({"placement": "source"}),
        canonical_fingerprint({"placement": "destination"}),
        "bitwise",
    )


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
    state = _state()
    reference = _scientific_arrays()
    arrays = {name: jax.device_put(value, source) for name, value in reference.items()}
    repository = _repository(tmp_path / "checkpoint")
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
    restored_state, restored, admission = restore_meshing_checkpoint(
        repository,
        manifest,
        {name: destination for name in arrays},
        _relation(),
        TopologyRestartPolicy(allow_topology_change=True),
        expected_state_id=state.checkpoint_state_id,
    )
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
    arrays = {name: jax.device_put(value) for name, value in _scientific_arrays().items()}
    arrays["compiled_cache"] = jax.device_put(np.asarray([1], dtype=np.int32))
    with pytest.raises(ValueError):
        publish_process_meshing_checkpoint(
            _repository(tmp_path / "checkpoint"),
            "mesh-accepted",
            "source-execution",
            _state(),
            arrays,
            analysis_plan_id="mesh-analysis",
            numeric_revision_id="accepted-revision",
            writer_id="process-writer",
        )


def test_meshing_restart_refuses_unqualified_changed_placement(tmp_path: Path) -> None:
    arrays = {name: jax.device_put(value) for name, value in _scientific_arrays().items()}
    repository = _repository(tmp_path / "checkpoint")
    state = _state()
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
    with pytest.raises(ValueError, match="Topology-changing restart is disabled"):
        restore_meshing_checkpoint(
            repository,
            manifest,
            {name: value.sharding for name, value in arrays.items()},
            _relation(),
            TopologyRestartPolicy(allow_topology_change=False),
            expected_state_id=state.checkpoint_state_id,
        )


def test_meshing_restart_rejects_scientific_record_substitution(tmp_path: Path) -> None:
    arrays = {name: jax.device_put(value) for name, value in _scientific_arrays().items()}
    repository = _repository(tmp_path / "checkpoint")
    state = _state()
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
    with pytest.raises(RepositoryCorruptionError, match="scientific binding changed"):
        restore_meshing_checkpoint(
            repository,
            manifest,
            {name: value.sharding for name, value in arrays.items()},
            _relation(),
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
        name: jax.device_put(value, source)
        for name, value in _scientific_arrays().items()
    }
    repository = _repository(tmp_path / "checkpoint")
    state = _state()
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
    with pytest.raises(RepositoryCorruptionError, match="covers"):
        restore_meshing_checkpoint(
            repository,
            incomplete,
            {name: source for name in arrays},
            _relation(),
            TopologyRestartPolicy(allow_topology_change=True),
            expected_state_id=state.checkpoint_state_id,
        )


def _multiprocess_restart_worker(
    process_index: int,
    repository_root: str,
) -> None:
    devices = tuple(jax.devices())
    source = NamedSharding(
        Mesh(np.asarray(devices, dtype=np.object_), ("owner",)), PartitionSpec("owner")
    )
    destination = NamedSharding(
        Mesh(np.asarray(devices[::-1], dtype=np.object_), ("owner",)),
        PartitionSpec("owner"),
    )
    reference = _scientific_arrays()
    arrays = {
        name: jax.make_array_from_process_local_data(
            source,
            value[
                process_index * (value.shape[0] // 2) : (process_index + 1)
                * (value.shape[0] // 2)
            ],
            global_shape=value.shape,
        )
        for name, value in reference.items()
    }
    if any(array.is_fully_addressable for array in arrays.values()):
        raise RuntimeError("The distributed scenario requires nonaddressable arrays.")
    repository = _repository(Path(repository_root))
    state = _state()
    publication = publish_process_meshing_checkpoint(
        repository,
        "mesh-accepted",
        "source-execution",
        state,
        arrays,
        analysis_plan_id="mesh-analysis",
        numeric_revision_id="accepted-revision",
        writer_id=f"process-writer-{process_index}",
    )
    multihost_utils.sync_global_devices("meshing-checkpoint-published")
    manifest = assemble_distributed_checkpoint_from_repository(
        repository,
        "mesh-accepted",
        "mesh-analysis",
        "accepted-revision",
        "source-execution",
        expected_process_count=2,
    )
    restored_state, restored, admission = restore_meshing_checkpoint(
        repository,
        manifest,
        {name: destination for name in arrays},
        _relation(),
        TopologyRestartPolicy(allow_topology_change=True),
        expected_state_id=state.checkpoint_state_id,
    )
    if restored_state.to_payload() != state.to_payload() or not admission.admitted:
        raise RuntimeError("Accepted scientific binding or restart admission changed.")
    for name, expected in reference.items():
        value = restored[name]
        if not isinstance(value, jax.Array):
            raise TypeError("Accepted scientific arrays must retain logical JAX storage.")
        for shard in value.addressable_shards:
            np.testing.assert_array_equal(np.asarray(shard.data), expected[shard.index])
    restored_ids = restored["vertex_ids"]
    if not isinstance(restored_ids, jax.Array):
        raise TypeError("Stable IDs must retain logical JAX storage.")
    print(
        json.dumps(
            {
                "process_index": publication.process_index,
                "local_ids": np.asarray(restored_ids.addressable_shards[0].data).tolist(),
                "mesh_epoch": restored_state.mesh_epoch,
            }
        ),
        flush=True,
    )
    jax.distributed.shutdown()


def _run_meshing_restart_workers(
    tmp_path: Path,
    worker_name: str,
    worker_arguments: tuple[str, ...] = (),
) -> list[dict[str, object]]:
    repository_root = tmp_path / "multiprocess-repository"
    # Provision shared repository root before concurrent writers open it.
    repository_root.mkdir()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        coordinator = f"127.0.0.1:{listener.getsockname()[1]}"
    environment = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        JAX_ENABLE_X64="true",
        XLA_FLAGS="--xla_force_host_platform_device_count=1",
        PHYDRAX_CPU_COLLECTIVES="gloo",
    )
    root = Path(__file__).parents[3]
    script = (
        "import runpy, sys, jax; "
        "jax.distributed.initialize(coordinator_address=sys.argv[4], "
        "num_processes=2, process_id=int(sys.argv[3]), local_device_ids=[0]); "
        "worker = runpy.run_path(sys.argv[1])[sys.argv[2]]; "
        "worker(int(sys.argv[3]), sys.argv[5], *sys.argv[6:])"
    )
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                str(Path(__file__).resolve()),
                worker_name,
                str(index),
                coordinator,
                str(repository_root),
                *worker_arguments,
            ],
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for index in range(2)
    ]
    results: list[dict[str, object]] = []
    try:
        for process in processes:
            stdout, stderr = process.communicate(timeout=120)
            assert process.returncode == 0, stderr
            results.append(json.loads(stdout))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    return results


def _multiprocess_rejected_checkpoint_worker(
    process_index: int,
    repository_root: str,
    fault: Literal["roles", "metadata"],
) -> None:
    devices = tuple(jax.devices())
    placement = NamedSharding(
        Mesh(np.asarray(devices, dtype=np.object_), ("owner",)), PartitionSpec("owner")
    )
    reference = _scientific_arrays()
    arrays = {
        name: jax.make_array_from_process_local_data(
            placement,
            value[
                process_index * (value.shape[0] // 2) : (process_index + 1)
                * (value.shape[0] // 2)
            ],
            global_shape=value.shape,
        )
        for name, value in reference.items()
    }
    state = _state()
    if process_index == 1:
        if fault == "roles":
            arrays.pop("material_inventory")
        else:
            payload = state.to_payload()
            payload["result_id"] = "different-accepted-scientific-result"
            state = MeshingCheckpointState.from_payload(payload)
    repository = _repository(Path(repository_root))
    published = False
    try:
        publish_process_meshing_checkpoint(
            repository,
            "mesh-rejected",
            "source-execution",
            state,
            arrays,
            analysis_plan_id="mesh-analysis",
            numeric_revision_id="accepted-revision",
            writer_id=f"process-writer-{process_index}",
        )
        published = True
    except ValueError:
        if fault != "roles" or process_index != 1:
            raise
    multihost_utils.sync_global_devices("meshing-checkpoint-local-refusal")
    try:
        assemble_distributed_checkpoint_from_repository(
            repository,
            "mesh-rejected",
            "mesh-analysis",
            "accepted-revision",
            "source-execution",
            expected_process_count=2,
        )
    except (ObjectNotFoundError, RepositoryCorruptionError) as error:
        expected = ObjectNotFoundError if fault == "roles" else RepositoryCorruptionError
        if type(error) is not expected:
            raise
    else:
        raise RuntimeError(
            "A partial or scientifically incoherent checkpoint was accepted."
        )
    print(
        json.dumps(
            {
                "process_index": process_index,
                "local_published": published,
                "complete_manifest_returned": False,
            }
        ),
        flush=True,
    )
    jax.distributed.shutdown()


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
    results = _run_meshing_restart_workers(
        tmp_path,
        "_multiprocess_rejected_checkpoint_worker",
        (fault,),
    )
    assert results == [
        {
            "process_index": 0,
            "local_published": published[0],
            "complete_manifest_returned": False,
        },
        {
            "process_index": 1,
            "local_published": published[1],
            "complete_manifest_returned": False,
        },
    ]


def test_nonaddressable_process_publication_restarts_on_different_owner(
    tmp_path: Path,
) -> None:
    results = _run_meshing_restart_workers(tmp_path, "_multiprocess_restart_worker")
    assert results == [
        {"process_index": 0, "local_ids": [1003, 2401], "mesh_epoch": 7},
        {"process_index": 1, "local_ids": [101, 509], "mesh_epoch": 7},
    ]


# SOURCE REGISTRY


def _native_source_closure() -> tuple[
    CellMesh, CellGeometrySpec, CellMeshAuditReport, dict[str, Any]
]:
    import phydrax as phx

    points = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(
        (
            (0, 2, 4),
            (2, 1, 4),
            (1, 3, 4),
            (3, 0, 4),
            (2, 0, 5),
            (1, 2, 5),
            (3, 1, 5),
            (0, 3, 5),
        ),
        dtype=np.int32,
    )
    mesh = phx.discretization.CellMesh.from_triangles(points, faces)
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    audit = phx.meshing.audit_cell_mesh(mesh, geometry)
    source = phx.geometry.ImplicitBoundarySource(
        phx.geometry.Sphere((0.0, 0.0, 0.0), 1.0, feature_id="restart-sphere").compile(),
        source_id="restart-sphere",
        spacing=0.2,
    )
    report = phx.meshing.certify_meshing_acceptance(
        mesh,
        geometry,
        audit,
        schedule=phx.meshing.MeshCertificationSchedule("surface"),
        source=source,
        fidelity_tolerance=0.8,
    )
    association = phx.meshing.GeometryAssociation(
        phx.meshing.GeometryAssociationKind.IMPLICIT,
        source.source_id,
        source.source_revision,
        mesh.entity_set(0).entity_set_id,
        mesh.vertex_global_ids,
        tuple("implicit-zero-set" for _ in points),
        np.zeros((points.shape[0],), dtype=np.float64),
        exact=True,
    )
    return (
        mesh,
        geometry,
        audit,
        {
            "source": source,
            "certification_inputs": report.request,
            "report": report,
            "associations": (association,),
        },
    )


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
    mesh, geometry, audit, closure = _native_source_closure()
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
        atol=1e-12,
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
    assert restored["associations"][0].source_occurrence_paths == ((),) * 6


def test_native_source_registry_refuses_explicit_callback() -> None:
    from phydrax.lifecycle._meshing_sources import validate_meshing_source_closure

    def callback() -> None:
        return None

    with pytest.raises(TypeError):
        validate_meshing_source_closure({"source": callback})


def test_native_source_registry_refuses_numerical_substitution_under_cached_ids() -> None:
    import equinox as eqx

    from phydrax.lifecycle._meshing_sources import (
        register_meshing_source_artifacts,
        validate_meshing_source_closure,
    )

    register_meshing_source_artifacts()
    _, _, _, closure = _native_source_closure()
    source = closure["source"]
    changed = eqx.tree_at(
        lambda value: value.geometry.state.values,
        source,
        replace_fn=lambda values: tuple(value + 0.125 for value in values),
    )
    assert changed.source_revision == source.source_revision
    closure["source"] = changed
    with pytest.raises(ValueError):
        validate_meshing_source_closure(closure)


def test_native_curve_source_retains_plain_dataclass_coefficients() -> None:
    import jax.numpy as jnp

    from phydrax._model._structure import (
        model_from_array_recipe,
        model_recipe_array_values,
        model_structure_recipe,
    )
    from phydrax.geometry._mesh_certificates import ParametricCurveBoundarySource
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


def _native_occurrence_authority() -> tuple[Any, Any]:
    from phydrax.geometry.brep._constructors import (
        _Builder,
        _stage_extrusion,
        PlanarProfile,
        ProfileLoop,
        ProfilePlane,
    )
    from phydrax.geometry.brep._model import BRepGeometry, BRepOccurrence

    builder = _Builder()
    _stage_extrusion(
        builder,
        PlanarProfile(
            ProfilePlane(),
            ProfileLoop.polygon(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        ),
        np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
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
    from phydrax.discretization._cell_mesh import CellBlock
    from phydrax.geometry.brep._projection_contracts import brep_entity_id
    from phydrax.lifecycle._meshing_sources import (
        register_meshing_source_artifacts,
        validate_meshing_source_closure,
    )
    from phydrax.meshing._association import GeometryAssociation, GeometryAssociationKind

    _, geometry = _native_occurrence_authority()
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
    assert restored_association.source_occurrence_paths == (("assembly", "placed-solid"),)
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
    np.testing.assert_allclose(point, [0.0, 0.0, 0.0], atol=1e-12)
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
    with pytest.raises(ValueError):
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

    builder, authority = _native_occurrence_authority()
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
    return BRepSource(model).compile()


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
    np.testing.assert_allclose(closest.points, [[4.0, 4.3, 5.4]], atol=1e-9)
    assert query.occurrences[0].path == ("assembly", "placed-solid")


def test_durable_native_source_archive_requeries_and_recertifies(tmp_path: Path) -> None:
    from phydrax._array_archive import ArrayArchiveLimits
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        recertify_restored_meshing_source,
        write_meshing_source_closure,
    )

    mesh, geometry, audit, records = _native_source_closure()
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
        atol=1e-12,
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

    _, _, _, records = _native_source_closure()
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


def _native_source_state(
    closure: dict[str, Any],
    *,
    source_array_bindings: dict[str, str] | None = None,
    source_logical_arrays: dict[str, jax.Array] | None = None,
) -> MeshingCheckpointState:
    payload = _state().to_payload()
    inputs = closure["certification_inputs"]
    payload.update(
        topology_id=inputs.topology_id,
        geometry_id=inputs.geometry_id,
        source_revision_id=inputs.source_revision or payload["source_revision_id"],
        source_closure=closure,
        source_owner_index=jax.process_index(),
        source_array_bindings=source_array_bindings,
        source_logical_arrays=source_logical_arrays,
    )
    return MeshingCheckpointState(**payload)


def _native_source_arrays(mesh: Any, mesh_epoch: int) -> dict[str, np.ndarray]:
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    cell_ids = np.asarray(mesh.blocks[0].global_ids, dtype=np.int64)
    return {
        "vertex_ids": vertex_ids,
        "coordinates": np.asarray(mesh.coordinates),
        "cell_vertex_ids": vertex_ids[np.asarray(mesh.blocks[0].vertices)],
        "parent_ids": np.stack((vertex_ids, vertex_ids), axis=-1),
        "accepted_epochs": np.full(cell_ids.shape, mesh_epoch, dtype=np.int64),
        "certificate_verdicts": np.ones(cell_ids.shape, dtype=np.uint8),
        "material_inventory": np.arange(1, len(vertex_ids) + 1, dtype=np.float64),
    }


def test_native_source_checkpoint_changed_placement_requeries_and_recertifies(
    tmp_path: Path,
) -> None:
    import phydrax as phx
    from phydrax._model._structure import (
        model_from_logical_array_recipe,
        model_recipe_array_inventory,
        model_recipe_array_values,
    )

    devices = tuple(jax.local_devices()[:2])
    if len(devices) != 2:
        pytest.skip("Source checkpoint placement scenario requires two local devices.")
    mesh, geometry, audit, closure = _native_source_closure()
    state = _native_source_state(closure)
    source = NamedSharding(
        Mesh(np.asarray(devices, dtype=np.object_), ("owner",)), PartitionSpec("owner")
    )
    destination = NamedSharding(
        Mesh(np.asarray(devices[::-1], dtype=np.object_), ("owner",)),
        PartitionSpec("owner"),
    )
    replicated = NamedSharding(
        Mesh(np.asarray(devices[::-1], dtype=np.object_), ("owner",)), PartitionSpec()
    )
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    reference = _native_source_arrays(mesh, state.mesh_epoch)
    arrays = {name: jax.device_put(value, source) for name, value in reference.items()}
    assert state.source_recipe_json is not None
    recipe = json.loads(state.source_recipe_json)
    inventory = model_recipe_array_inventory(recipe, prefix="source-record")
    shardings = {name: destination for name in arrays}
    for entry in inventory:
        if entry.backend != "numpy":
            shardings[state.source_array_name(entry.name)] = (
                destination if entry.shape and entry.shape[0] % 2 == 0 else replicated
            )
    repository = _repository(
        tmp_path / "native-source-checkpoint",
        maximum_chunk_bytes=16 * 1024,
        maximum_metadata_bytes=1024 * 1024,
    )
    publish_process_meshing_checkpoint(
        repository,
        "native-source",
        "source-execution",
        state,
        arrays,
        analysis_plan_id="source-analysis",
        numeric_revision_id="source-revision",
        writer_id="source-writer",
    )
    manifest = assemble_distributed_checkpoint_from_repository(
        repository,
        "native-source",
        "source-analysis",
        "source-revision",
        "source-execution",
        expected_process_count=1,
    )
    restored_state, restored_arrays, admission = restore_meshing_checkpoint(
        repository,
        manifest,
        shardings,
        _relation(),
        TopologyRestartPolicy(allow_topology_change=True),
        expected_state_id=state.checkpoint_state_id,
    )
    assert admission.admitted
    assert restored_state.source_closure_id == state.source_closure_id
    assert restored_state.to_payload() == state.to_payload()
    restored = restored_state.source_closure
    assert restored is not None
    inputs = restored["certification_inputs"]
    report = restored["report"]
    assert report.request.request_id == inputs.request_id
    assert restored_state.request_id == _state().request_id
    assert restored_state.evidence_id == _state().evidence_id
    assert inputs.source_revision == state.source_revision_id
    assert inputs.fidelity_tolerance == closure["certification_inputs"].fidelity_tolerance
    np.testing.assert_allclose(
        np.asarray(
            inputs.source.boundary_distance(
                np.asarray(((2.0, 0.0, 0.0),), dtype=np.float64)
            ).upper
        ),
        np.asarray((1.0,), dtype=np.float64),
    )
    renewed = phx.meshing.certify_meshing_acceptance(
        mesh,
        geometry,
        audit,
        schedule=inputs.schedule,
        domain=inputs.domain,
        source=inputs.source,
        cell_regions=inputs.cell_regions,
        fidelity_tolerance=inputs.fidelity_tolerance,
        fidelity_sample_order=inputs.fidelity_sample_order,
        limits=inputs.limits,
        junction_vertices=inputs.junction_vertices,
    )
    renewed.require_passed()
    assert renewed.report_id == report.report_id
    for old, new in zip(closure["associations"], restored["associations"], strict=True):
        assert new.source_occurrence_paths == old.source_occurrence_paths
        assert new.source_entity_ids == old.source_entity_ids
        np.testing.assert_array_equal(
            np.asarray(new.target_global_ids), np.asarray(old.target_global_ids)
        )
    for name, expected in reference.items():
        np.testing.assert_array_equal(np.asarray(restored_arrays[name]), expected)
    assert np.sum(np.asarray(restored_arrays["material_inventory"])) == np.sum(
        reference["material_inventory"]
    )
    restored_ids = restored_arrays["vertex_ids"]
    assert isinstance(restored_ids, jax.Array)
    np.testing.assert_array_equal(
        np.asarray(
            next(
                shard.data
                for shard in restored_ids.addressable_shards
                if shard.device == devices[0]
            )
        ),
        vertex_ids[len(vertex_ids) // 2 :],
    )
    for entry in inventory:
        value = restored_arrays[state.source_array_name(entry.name)]
        if entry.backend == "numpy":
            assert isinstance(value, np.ndarray) and not value.flags.writeable

    with pytest.raises(RepositoryCorruptionError, match="scientific binding"):
        restore_meshing_checkpoint(
            repository,
            manifest,
            shardings,
            _relation(),
            TopologyRestartPolicy(allow_topology_change=True),
            expected_state_id=canonical_fingerprint({"substitute": "source"}),
        )
    source_shard = next(
        shard
        for shard in manifest.shards
        if "source-record:" in dict(shard.metadata)["array_path"]
    )
    incomplete = CheckpointManifest(
        manifest.checkpoint_id,
        manifest.analysis_plan_id,
        manifest.numeric_revision_id,
        manifest.execution_plan_id,
        tuple(
            shard for shard in manifest.shards if shard.shard_id != source_shard.shard_id
        ),
        complete=True,
    )
    with pytest.raises(RepositoryCorruptionError):
        restore_meshing_checkpoint(
            repository,
            incomplete,
            shardings,
            _relation(),
            TopologyRestartPolicy(allow_topology_change=True),
            expected_state_id=state.checkpoint_state_id,
        )
    substituted_values = model_recipe_array_values(
        closure, recipe, prefix="source-record"
    )
    numerical = next(
        entry
        for entry in inventory
        if np.issubdtype(np.dtype(entry.dtype), np.floating) and np.prod(entry.shape) > 0
    )
    original = substituted_values[numerical.name]
    substituted_values[numerical.name] = original + np.asarray(
        0.125, dtype=original.dtype
    )
    substituted_closure = model_from_logical_array_recipe(
        recipe, substituted_values, prefix="source-record"
    )
    with pytest.raises(ValueError):
        MeshingCheckpointState.from_payload(
            state.to_payload(), source_closure=substituted_closure
        )


def _multiprocess_source_restart_worker(
    process_index: int,
    repository_root: str,
) -> None:
    import phydrax as phx
    from phydrax._model._structure import (
        model_from_logical_array_recipe,
        model_recipe_array_inventory,
        model_recipe_array_values,
        model_structure_recipe,
    )
    from phydrax.lifecycle._meshing_sources import register_meshing_source_artifacts

    devices = tuple(jax.devices())
    source = NamedSharding(
        Mesh(np.asarray(devices, dtype=np.object_), ("owner",)), PartitionSpec("owner")
    )
    destination = NamedSharding(
        Mesh(np.asarray(devices[::-1], dtype=np.object_), ("owner",)),
        PartitionSpec("owner"),
    )
    replicated_source = NamedSharding(source.mesh, PartitionSpec())
    replicated_destination = NamedSharding(destination.mesh, PartitionSpec())
    mesh, geometry, audit, local_closure = _native_source_closure()
    register_meshing_source_artifacts()
    recipe = model_structure_recipe(local_closure)
    inventory = model_recipe_array_inventory(recipe, prefix="source-record")
    source_values = model_recipe_array_values(
        local_closure, recipe, prefix="source-record"
    )
    destination_shardings: dict[str, NamedSharding] = {}
    source_bindings: dict[str, str] = {}
    logical_bank: dict[str, jax.Array] = {}
    destination_by_alias: dict[str, NamedSharding] = {}
    for entry in inventory:
        if entry.backend == "numpy":
            continue
        value = np.asarray(source_values[entry.name])
        partitioned = bool(entry.shape) and entry.shape[0] % 2 == 0
        local_value = (
            value[
                process_index * (entry.shape[0] // 2) : (process_index + 1)
                * (entry.shape[0] // 2)
            ]
            if partitioned
            else value
        )
        logical = jax.make_array_from_process_local_data(
            source if partitioned else replicated_source,
            local_value,
            global_shape=entry.shape,
        )
        source_values[entry.name] = logical
        alias = f"source/{entry.path}"
        source_bindings[entry.name] = alias
        logical_bank[alias] = logical
        destination_by_alias[alias] = (
            destination if partitioned else replicated_destination
        )
    closure = model_from_logical_array_recipe(
        recipe, source_values, prefix="source-record"
    )
    state = _native_source_state(
        closure, source_array_bindings=source_bindings, source_logical_arrays=logical_bank
    )
    destination_shardings.update(
        {
            state.source_array_name(entry.name): destination_by_alias[
                source_bindings[entry.name]
            ]
            for entry in inventory
            if entry.backend != "numpy"
        }
    )
    trusted_ids = multihost_utils.process_allgather(
        np.frombuffer(bytes.fromhex(state.checkpoint_state_id), dtype=np.uint8),
        tiled=False,
    )
    expected_ids = {
        owner: np.asarray(identifier).tobytes().hex()
        for owner, identifier in enumerate(np.asarray(trusted_ids))
    }
    reference = _native_source_arrays(mesh, state.mesh_epoch)
    arrays = {
        name: jax.make_array_from_process_local_data(
            source,
            value[
                process_index * (value.shape[0] // 2) : (process_index + 1)
                * (value.shape[0] // 2)
            ],
            global_shape=value.shape,
        )
        for name, value in reference.items()
    }
    if any(array.is_fully_addressable for array in arrays.values()):
        raise RuntimeError(
            "Source checkpoint requires actual nonaddressable accepted state."
        )
    for name in arrays:
        destination_shardings[name] = destination
    repository = _repository(
        Path(repository_root),
        maximum_chunk_bytes=16 * 1024,
        maximum_metadata_bytes=1024 * 1024,
    )
    publish_process_meshing_checkpoint(
        repository,
        "native-source",
        "source-execution",
        state,
        arrays,
        analysis_plan_id="source-analysis",
        numeric_revision_id="source-revision",
        writer_id=f"source-writer-{process_index}",
    )
    multihost_utils.sync_global_devices("native-source-checkpoint-published")
    manifest = assemble_distributed_checkpoint_from_repository(
        repository,
        "native-source",
        "source-analysis",
        "source-revision",
        "source-execution",
        expected_process_count=2,
    )
    restored_state, restored, admission = restore_meshing_checkpoint(
        repository,
        manifest,
        destination_shardings,
        _relation(),
        TopologyRestartPolicy(allow_topology_change=True),
        expected_state_id=expected_ids,
        source_owner_index=process_index,
    )
    if (
        not admission.admitted
        or restored_state.source_closure_id != state.source_closure_id
    ):
        raise RuntimeError("Source scientific binding or restart admission changed.")
    if tuple(
        owner.source_owner_index for owner in restored_state.source_owner_states
    ) != (0, 1):
        raise RuntimeError("Cold source restoration lost an exact original owner record.")
    if {
        owner.source_owner_index: owner.checkpoint_state_id
        for owner in restored_state.source_owner_states
    } != expected_ids:
        raise RuntimeError("Cold source restoration changed a committed owner identity.")
    for owner in restored_state.source_owner_states:
        if owner.source_closure is None:
            raise RuntimeError(
                "Cold source owner bank retained metadata but lost actual records."
            )
    retained = restored_state.source_closure
    if retained is None:
        raise RuntimeError("Actual native source records were not restored.")
    inputs = retained["certification_inputs"]
    np.testing.assert_allclose(
        np.asarray(
            inputs.source.boundary_distance(
                np.asarray(((2.0, 0.0, 0.0),), dtype=np.float64)
            ).upper
        ),
        np.asarray((1.0,), dtype=np.float64),
    )
    renewed = phx.meshing.certify_meshing_acceptance(
        mesh,
        geometry,
        audit,
        schedule=inputs.schedule,
        source=inputs.source,
        domain=inputs.domain,
        cell_regions=inputs.cell_regions,
        fidelity_tolerance=inputs.fidelity_tolerance,
        fidelity_sample_order=inputs.fidelity_sample_order,
        limits=inputs.limits,
        junction_vertices=inputs.junction_vertices,
    )
    renewed.require_passed()
    if renewed.report_id != retained["report"].report_id:
        raise RuntimeError("Restored native source recertification changed.")
    for name, expected in reference.items():
        value = restored[name]
        if not isinstance(value, jax.Array):
            raise TypeError(
                "Accepted numerical state must retain its logical JAX backend."
            )
        for shard in value.addressable_shards:
            np.testing.assert_array_equal(np.asarray(shard.data), expected[shard.index])
    restored_ids = restored["vertex_ids"]
    if not isinstance(restored_ids, jax.Array):
        raise TypeError("Restored stable IDs must be a logical JAX array.")
    print(
        json.dumps(
            {
                "process_index": process_index,
                "local_ids": np.asarray(restored_ids.addressable_shards[0].data).tolist(),
                "mesh_epoch": restored_state.mesh_epoch,
                "source_recertified": renewed.passed,
            }
        ),
        flush=True,
    )
    jax.distributed.shutdown()


def test_nonaddressable_native_source_checkpoint_restores_and_recertifies(
    tmp_path: Path,
) -> None:
    results = _run_meshing_restart_workers(
        tmp_path, "_multiprocess_source_restart_worker"
    )
    assert results == [
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


def test_native_numpy_source_checkpoint_restores_immutable_authored_coefficients(
    tmp_path: Path,
) -> None:
    import phydrax as phx
    from phydrax._model._structure import model_recipe_array_inventory
    from phydrax.geometry._mesh_certificates import PiecewiseLinearDomain

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
    state = _native_source_state(closure)
    reference = _native_source_arrays(mesh, state.mesh_epoch)
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
    repository = _repository(
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
    restored_state, restored_arrays, admission = restore_meshing_checkpoint(
        repository,
        manifest,
        shardings,
        _relation(),
        TopologyRestartPolicy(allow_topology_change=True),
        expected_state_id=state.checkpoint_state_id,
    )
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
    from phydrax.geometry._mesh_certificates import ImplicitProjectionBoundarySource
    from phydrax.geometry.implicit._analytic_profile import AnalyticImplicitProfile

    mesh, geometry, audit, original = _native_source_closure()
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
    state = _native_source_state(closure)
    reference = _native_source_arrays(mesh, state.mesh_epoch)
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
    repository = _repository(
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
    restored_state, restored_arrays, admission = restore_meshing_checkpoint(
        repository,
        manifest,
        shardings,
        _relation(),
        TopologyRestartPolicy(allow_topology_change=True),
        expected_state_id=state.checkpoint_state_id,
    )
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


def _native_family_generation(
    family: str,
) -> tuple[
    MeshPart,
    NativeGenerationSource,
    SurfaceMeshingSpec | VolumeMeshingSpec,
    NativeMeshingOptions,
]:
    import phydrax as phx
    from phydrax.discretization._cell_geometry import coordinate_lagrange_element
    from phydrax.discretization._reference_cell import reference_cell_topology
    from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain
    from phydrax.geometry._mesh_certificates import ParametricCurveBoundarySource
    from phydrax.geometry.brep._patches import LineCurve
    from phydrax.meshing._volume_generation import declared_plc_domain

    m = phx.meshing
    source: NativeGenerationSource
    specification: SurfaceMeshingSpec | VolumeMeshingSpec
    if family == "structured":
        boundaries = tuple(
            m.TransfiniteCurveControl(LineCurve(origin, direction), name, (0.0, 1.0), 2)
            for name, origin, direction in (
                ("left", (0.0, 0.0), (0.0, 1.0)),
                ("right", (1.0, 0.0), (0.0, 1.0)),
                ("bottom", (0.0, 0.0), (1.0, 0.0)),
                ("top", (0.0, 1.0), (1.0, 0.0)),
            )
        )
        block = m.TransfiniteBlock("square", boundaries)
        query = ParametricCurveBoundarySource(
            tuple(control.curve for control in boundaries),
            tuple(control.parameter_range for control in boundaries),
            source_id="restart-structured",
            source_revision="r1",
        )
        source = m.NativeStructuredSource(
            (block,),
            (),
            query.source_id,
            query.source_revision,
            fidelity_source=query,
            maximum_deviation=0.08,
        )
        scope = m.MeshingScope(
            source.source_id,
            source.source_revision,
            m.MeshingEntityKind.GEOMETRY,
            2,
            "square-domain",
            np.asarray((0,), dtype=np.int64),
        )
        specification = m.SurfaceMeshingSpec(
            m.CellMeshingTarget(2, 2, m.CellFamilyPolicy(required=("quadrilateral",))),
            scope,
            planar_embedding=phx.geometry.PlanarEmbedding(
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            size_controls=(
                m.UniformSizeControl(scope, 0.5, strength=m.SizeControlStrength.SOFT),
            ),
        )
        options = m.NativeMeshingOptions("structured_transfinite")
    elif family == "surface":
        from examples._native_surface_sources import cylinder_sheet

        source = m.NativeSurfaceSource(cylinder_sheet().domain)
        scope = m.MeshingScope(
            source.source_id,
            source.source_revision,
            m.MeshingEntityKind.GEOMETRY,
            2,
            source.domain.entity_set_id(2),
            np.asarray((0,), dtype=np.int64),
        )
        specification = m.SurfaceMeshingSpec(
            m.CellMeshingTarget(2, 3, m.CellFamilyPolicy(required=("triangle",))),
            scope,
            size_controls=(
                m.UniformSizeControl(scope, 0.5, strength=m.SizeControlStrength.SOFT),
            ),
        )
        options = m.NativeMeshingOptions("parametric_surface")
    else:
        kind = "tetrahedron" if family == "plc" else "hexahedron"
        topology = reference_cell_topology(kind)
        corners = np.asarray(topology.vertices, dtype=np.float64)
        loops = topology.entities[2]
        complex_ = m.PiecewiseLinearComplex(
            corners,
            loops,
            np.arange(len(loops), dtype=np.int64),
            np.tile(np.asarray((-1, 0), dtype=np.int64), (len(loops), 1)),
            ("material",),
        )
        reference_source = m.NativePlcSource(complex_, "restart-reference", "r1")
        source = reference_source
        if family == "mapped":
            mesh = phx.discretization.CellMesh(
                corners,
                (phx.discretization.CellBlock("root", kind, np.arange(8)[None]),),
                numeric_version="r1",
            )
            element = coordinate_lagrange_element(kind, 2)
            nodes = np.asarray(element.reference_nodes)
            physical = nodes.copy()
            physical[:, 2] += 0.125 * nodes[:, 0] ** 2 * nodes[:, 1]
            geometry = CellGeometrySpec(
                {"root": element}, {"root": np.arange(len(nodes))[None]}, physical
            )
            domain = MappedReferenceDomain(
                declared_plc_domain(complex_, reference_source.source_id),
                mesh,
                geometry,
                np.asarray((0,), dtype=np.int64),
                source_id="restart-curved",
                source_revision="q2-r1",
            )
            source = m.NativeMappedHexSource(reference_source, domain)
            scope = m.MeshingScope(
                source.source_id,
                source.source_revision,
                m.MeshingEntityKind.GEOMETRY,
                2,
                domain.entity_set_id(2),
                mesh.entity_set(2).entity_ids,
            )
        else:
            if family == "polyhedral":
                source = m.NativePolyhedralSource(
                    complex_,
                    "restart-power",
                    "r1",
                    sites=np.asarray(
                        (
                            (0.25, 0.25, 0.25),
                            (0.75, 0.25, 0.25),
                            (0.25, 0.75, 0.25),
                            (0.25, 0.25, 0.75),
                        )
                    ),
                    weights=np.asarray((0.03125, -0.015625, 0.0625, 0.0)),
                )
            scope = m.MeshingScope(
                source.source_id,
                source.source_revision,
                m.MeshingEntityKind.GEOMETRY,
                2,
                "source-facets",
                np.arange(len(loops), dtype=np.int64),
            )
        cell_kind = {
            "plc": "tetrahedron",
            "polyhedral": "polyhedron",
            "mapped": "hexahedron",
        }[family]
        fill = {
            "plc": m.VolumeFillStrategy.SIMPLEX,
            "polyhedral": m.VolumeFillStrategy.POLYHEDRAL,
            "mapped": m.VolumeFillStrategy.MULTIZONE,
        }[family]
        specification = m.VolumeMeshingSpec(
            m.CellMeshingTarget(
                3,
                3,
                m.CellFamilyPolicy(required=(cell_kind,)),
                geometry_order=2 if family == "mapped" else 1,
            ),
            scope,
            fill,
            size_controls=(
                m.UniformSizeControl(
                    scope,
                    0.5 if family == "mapped" else 10.0,
                    strength=m.SizeControlStrength.SOFT,
                ),
            ),
        )
        routes: dict[str, NativeMeshingRoute] = {
            "plc": "plc_tetrahedral",
            "polyhedral": "plc_restricted_power",
            "mapped": "mapped_balanced_grid_hex",
        }
        options = m.NativeMeshingOptions(routes[family])
    result = (
        m.NativeMeshingProvider(options)
        .plan(
            source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )
    return m.MeshPart("generation", result), source, specification, options


@pytest.mark.parametrize(
    "family", ("plc", "structured", "mapped", "polyhedral", "surface")
)
def test_native_family_durable_restart_requeries_recertifies_and_prepares(
    tmp_path: Path,
    family: str,
) -> None:
    import phydrax as phx
    from phydrax._array_archive import ArrayArchiveLimits
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        recertify_restored_meshing_source,
        write_meshing_source_closure,
    )

    part, source, specification, options = _native_family_generation(family)
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
    assert renewed.report_id == report.report_id
    assert (
        restored["generation_specification"].specification_id
        == result.compliance.specification_id
    )
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
        # implicit/planar distance query. Fresh coverage uses its actual facets.
        assert renewed.coverage is not None
        assert isinstance(renewed.request.domain, PiecewiseLinearDomain)
        assert isinstance(source, (NativePlcSource, NativePolyhedralSource))
        assert renewed.coverage.status == "certified"
        np.testing.assert_array_equal(
            renewed.request.domain.vertices, source.complex.vertices
        )
    if family == "polyhedral":
        assert isinstance(source, NativePolyhedralSource)
        from phydrax.discretization.finite_volume._polyhedral import (
            prepare_polyhedral_finite_volume_geometry,
        )

        fv = prepare_polyhedral_finite_volume_geometry(result.mesh)
        np.testing.assert_allclose(np.sum(fv.cell_volumes), 1.0, atol=1e-12)
        np.testing.assert_array_equal(
            restored["generation_source"].weights, source.weights
        )
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
    else:
        degree = 2 if family == "mapped" else 1
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
        from phydrax.discretization.fem._point_interpolation import (
            prepare_finite_element_field_reconstruction,
        )

        weights = np.arange(1, nodes.shape[1] + 1, dtype=np.float64)
        coefficients = 2.0 + nodes @ weights
        for index, block in enumerate(result.mesh.blocks):
            if result.mesh.ambient_dimension != result.mesh.topological_dimension:
                # Embedded surface FE owns intrinsic charts, not a Cartesian
                # volume-support reconstruction or a projected planar region.
                from phydrax.discretization._reference_cell import reference_cell_topology

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
                    values, 2.0 + np.asarray(local.physical_points) @ weights, atol=2e-10
                )
            else:
                sites = np.mean(
                    np.asarray(result.mesh.coordinates)[np.asarray(block.vertices)],
                    axis=1,
                )
                query = prepare_finite_element_field_reconstruction(
                    space,
                    "u",
                    block_name=block.name,
                ).evaluate(coefficients, sites)
                np.testing.assert_allclose(
                    query.values, 2.0 + sites @ weights, atol=2e-10
                )
        if family == "mapped":
            assert result.geometry.restriction_source is not None
            np.testing.assert_array_equal(
                result.geometry.coordinates, original_carrier.geometry.coordinates
            )


def test_native_family_refuses_changed_source_power_weights_and_hard_request() -> None:
    import equinox as eqx

    from phydrax._array_archive import ArrayArchiveLimits
    from phydrax.lifecycle._meshing_source_families import validate_registered_native_part
    from phydrax.lifecycle._meshing_sources import (
        register_meshing_source_artifacts,
        validate_meshing_source_closure,
    )

    register_meshing_source_artifacts()
    part, source, specification, options = _native_family_generation("polyhedral")
    assert isinstance(source, NativePolyhedralSource)
    assert isinstance(specification, VolumeMeshingSpec)
    forged = eqx.tree_at(lambda value: value.weights, source, source.weights + 0.1)
    with pytest.raises(ValueError, match="freshly certified native source authority"):
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
    with pytest.raises(
        ValueError, match="hard generation request|actual authored source|source revision"
    ):
        validate_registered_native_part(
            part, source, altered, options, limits=ArrayArchiveLimits()
        )


def test_native_family_refuses_changed_original_algorithm_schedule() -> None:
    import phydrax as phx
    from phydrax._array_archive import ArrayArchiveLimits
    from phydrax.lifecycle._meshing_source_families import validate_registered_native_part

    part, source, specification, _ = _native_family_generation("mapped")
    options = phx.meshing.NativeMeshingOptions(
        "mapped_balanced_grid_hex",
        grid_schedule=phx.meshing.NativeHexGridSchedule("balanced_grid", maximum_depth=9),
    )
    with pytest.raises(ValueError, match="original source-bound generation plan"):
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
        from tests.unit.meshing.test_native_overset_fields import _mapped_source

        part = _mapped_source(("hexahedron", "prism"))
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
    # These are actual represented carriers, not fabricated original provider
    # requests. Source-free carrier certification is an honest separate role.
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
                atol=2e-10,
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
        np.testing.assert_allclose(np.sum(fv.cell_volumes), 7.0 / 6.0, atol=1e-12)
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


# FIELD DECLARATIONS: physical owners survive a cold native source restart.
@pytest.mark.parametrize(
    "field_kind",
    (
        "mapped-scalar",
        "mapped-vector",
        "rt",
        "bdm2",
        "nedelec",
        "nedelec1",
        "fv-scalar",
        "fv-euler-history",
    ),
)
def test_cold_native_physical_field_declarations_reprepare_exact_owners(
    tmp_path: Path,
    field_kind: str,
) -> None:
    from dataclasses import replace
    from math import prod

    import jax.numpy as jnp

    import phydrax as phx
    from phydrax._array_archive import ArrayArchiveLimits
    from phydrax._differentiation import BranchDifferentiationPolicy
    from phydrax._frozendict import frozendict
    from phydrax.discretization import FiniteElementDiscretization
    from phydrax.discretization._conservation_boundary import ExtrapolationBoundary
    from phydrax.discretization._views import FieldTracePolicy
    from phydrax.discretization.fem import form_element
    from phydrax.discretization.fem._generic import FiniteElementFieldSpec
    from phydrax.discretization.fem._point_interpolation import (
        prepare_finite_element_field_reconstruction,
    )
    from phydrax.discretization.fem._precision import FiniteElementPrecisionPolicy
    from phydrax.discretization.finite_volume import (
        PiecewiseConstantReconstruction,
        UnstructuredFiniteVolumeDiscretization,
        UnstructuredFiniteVolumePlan,
    )
    from phydrax.discretization.finite_volume._riemann import RusanovFluxPlan
    from phydrax.equations._hyperbolic_systems import EulerSystem
    from phydrax.lifecycle._meshing_field_records import (
        MeshingFieldDeclaration,
        MeshingFieldStateRole,
        prepare_meshing_field_dynamics,
        prepare_meshing_field_index_binding,
        prepare_meshing_field_owner,
    )
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        recertify_restored_meshing_source,
        validate_meshing_source_closure,
        write_meshing_source_closure,
    )
    from phydrax.linalg import ArraySpace
    from phydrax.meshing._overset import (
        OversetPartSpec,
        OversetPolicy,
        prepare_overset_connectivity,
    )
    from phydrax.meshing._result import CellMeshingResult

    family = (
        "mapped"
        if field_kind.startswith("mapped")
        else "polyhedral"
        if field_kind.startswith("fv")
        else "plc"
    )
    part, source, specification, options = _native_family_generation(family)
    carrier = part.carrier
    assert isinstance(carrier, CellMeshingResult)
    wall = OversetPartSpec(part.name)
    companion = phx.meshing.MeshPart("companion", carrier)
    companion_wall = OversetPartSpec(companion.name)
    registration = prepare_overset_connectivity(
        phx.meshing.MeshAssembly((part, companion)),
        (wall, companion_wall),
        policy=OversetPolicy(fringe_layers=1),
    ).registration()
    fields = {}
    name = "physical"
    finite_element_fields: tuple[FiniteElementFieldSpec, ...] = ()
    precision_policy: FiniteElementPrecisionPolicy | None = None
    finite_volume_field_name: str | None = None
    finite_volume_component_names: tuple[str, ...] = ()
    coordinate_policy: str | None = None
    reconstruction_policy: PiecewiseConstantReconstruction | None = None
    reconstruction_policy_id: str | None = None
    system: EulerSystem | None = None
    flux_policy: RusanovFluxPlan | None = None
    boundary_policies: frozendict[str, ExtrapolationBoundary] = frozendict()
    if field_kind.startswith("fv"):
        system = (
            phx.equations.EulerSystem(
                3, material=phx.equations.IdealGasMaterial(1.67, 287.0)
            )
            if field_kind.endswith("history")
            else None
        )
        components = system.component_names if system is not None else ("temperature",)
        original = UnstructuredFiniteVolumePlan.from_cell_mesh(
            carrier.mesh,
            field_name=name,
            component_names=components,
        ).prepare(numeric_version="accepted-7")
        assert isinstance(original, UnstructuredFiniteVolumeDiscretization)
        fv_original = original
        owner = "finite_volume"
        finite_volume_field_name = name
        finite_volume_component_names = components
        coordinate_policy = "represented-polyhedral"
        reconstruction_policy = PiecewiseConstantReconstruction()
        reconstruction_policy_id = reconstruction_policy.plan_id
        coefficient_space = original.field_spaces[0].vector_space
        assert isinstance(coefficient_space, ArraySpace)
        shape = coefficient_space.shape
        coefficients = np.arange(prod(shape), dtype=np.float64).reshape(shape) + 2.0
        if system is not None:
            primitive = np.tile(
                np.asarray((1.0, 0.0, 0.0, 0.0, 1.0)), (original.cell_count, 1)
            )
            primitive[:, 0] += 0.1 * np.asarray(fv_original.cell_centers)[:, 0]
            coefficients = np.asarray(
                system.primitive_to_conserved(jnp.asarray(primitive))
            ).copy()
            flux_policy = RusanovFluxPlan()
            boundary_policies = frozendict(
                {
                    patch: ExtrapolationBoundary()
                    for patch in original.boundary_patch_names
                }
            )
        maximum_order = 0
    else:
        if field_kind == "rt":
            element = form_element("tetrahedron", 2, 1, twist="twisted", proxy="flux")
        elif field_kind == "bdm2":
            element = form_element(
                "tetrahedron", 2, 2, family="full", twist="twisted", proxy="flux"
            )
        elif field_kind.startswith("nedelec"):
            element = form_element(
                "tetrahedron",
                1,
                2 if field_kind == "nedelec1" else 1,
                proxy="circulation",
            )
        else:
            element = phx.discretization.lagrange_element("hexahedron", 2)
        field = phx.discretization.FiniteElementFieldSpec(
            name,
            {block.name: element for block in carrier.mesh.blocks},
            component_shape=(2,) if field_kind == "mapped-vector" else (),
        )
        precision = FiniteElementPrecisionPolicy()
        original = phx.discretization.FiniteElementPlan(
            carrier.mesh,
            field,
            coordinate_spec=carrier.geometry,
            precision_policy=precision,
        ).prepare(numeric_version="accepted-7")
        owner = "finite_element"
        finite_element_fields = (field,)
        precision_policy = precision
        coefficient_space = original.field_spaces[0].vector_space
        assert isinstance(coefficient_space, ArraySpace)
        shape = coefficient_space.shape
        coefficients = (
            np.arange(prod(shape), dtype=np.float64).reshape(shape) * 0.125 + 1.0
        )
        maximum_order = 1
    coefficients.flags.writeable = False
    fields["physical-coefficients"] = coefficients
    coefficient_binding = prepare_meshing_field_index_binding(
        carrier,
        "mesh/cell_ids"
        if field_kind.startswith("fv")
        else "field/global_coefficient_ids",
        field_space=original.field_spaces[0],
    )
    roles = [
        MeshingFieldStateRole(
            "physical-coefficients",
            name,
            "coefficients",
            shape,
            coefficients.dtype,
            7,
            index_binding=coefficient_binding,
        )
    ]
    history_policy = "none"
    if field_kind == "mapped-scalar":
        assert isinstance(original, FiniteElementDiscretization)
        boundary_values = np.asarray(original.dof_maps[0].dof_coordinates) @ np.asarray(
            (1.0, 2.0, 3.0)
        )
        boundary_mask = np.asarray(original.dof_maps[0].boundary_dof_mask).copy()
        boundary_values.flags.writeable = boundary_mask.flags.writeable = False
        fields["essential-values"] = boundary_values
        fields["essential-mask"] = boundary_mask
        roles.extend(
            (
                MeshingFieldStateRole(
                    "essential-values",
                    name,
                    "boundary-values",
                    boundary_values.shape,
                    boundary_values.dtype,
                    7,
                    index_binding=coefficient_binding,
                ),
                MeshingFieldStateRole(
                    "essential-mask",
                    name,
                    "boundary-mask",
                    boundary_mask.shape,
                    boundary_mask.dtype,
                    7,
                    index_binding=coefficient_binding,
                ),
            )
        )
    if field_kind.endswith("history"):
        if not isinstance(original, UnstructuredFiniteVolumeDiscretization):
            raise TypeError(
                "Material-history restart requires finite-volume cell centers."
            )
        fraction = 0.2 + 0.1 * np.asarray(original.cell_centers)[:, 0]
        history = np.column_stack((fraction, 1.0 - fraction))
        history.flags.writeable = False
        fields["material-history"] = history
        fields["state-epoch"] = np.asarray(7, dtype=np.int64)
        roles.extend(
            (
                MeshingFieldStateRole(
                    "material-history",
                    name,
                    "material-history",
                    history.shape,
                    history.dtype,
                    7,
                    (phx.units.ONE,),
                    index_binding=prepare_meshing_field_index_binding(
                        carrier, "mesh/cell_ids"
                    ),
                ),
                MeshingFieldStateRole(
                    "state-epoch", name, "state-epoch", (), np.dtype("int64"), 7
                ),
            )
        )
        history_policy = "material-history"
    units = (phx.units.KELVIN,)
    if field_kind == "fv-euler-history":
        density_unit = phx.units.derived_unit(
            "kg/m^3", ((phx.units.KILOGRAM, 1), (phx.units.METER, -3))
        )
        momentum_unit = phx.units.derived_unit(
            "kg/(m^2*s)",
            ((phx.units.KILOGRAM, 1), (phx.units.METER, -2), (phx.units.SECOND, -1)),
        )
        units = (
            density_unit,
            momentum_unit,
            momentum_unit,
            momentum_unit,
            phx.units.PASCAL,
        )
    declaration = MeshingFieldDeclaration(
        part_name=part.name,
        topology_id=carrier.mesh.topology_id,
        geometry_layout_id=carrier.geometry.geometry_layout_id,
        field_space_ids=frozendict({name: original.field_spaces[0].field_space_id}),
        value_units=frozendict({name: units}),
        maximum_derivative_orders=frozendict({name: maximum_order}),
        branch_policy=BranchDifferentiationPolicy.SMOOTH,
        trace_policy=FieldTracePolicy("cell-sided"),
        state_roles=tuple(roles),
        history_policy=history_policy,
        numeric_version="accepted-7",
        source_wall_policy=wall,
        owner=owner,
        finite_element_fields=finite_element_fields,
        precision_policy=precision_policy,
        finite_volume_field_name=finite_volume_field_name,
        finite_volume_component_names=finite_volume_component_names,
        finite_volume_coordinate_policy=coordinate_policy,
        reconstruction_policy=reconstruction_policy,
        reconstruction_policy_id=reconstruction_policy_id,
        finite_volume_system=system,
        finite_volume_flux_policy=flux_policy,
        finite_volume_boundaries=boundary_policies,
    )
    companion_roles = tuple(
        replace(role, entry_name=f"companion/{role.entry_name}") for role in roles
    )
    companion_declaration = replace(
        declaration,
        part_name=companion.name,
        source_wall_policy=companion_wall,
        state_roles=companion_roles,
    )
    fields.update(
        {f"companion/{key}": value.copy() for key, value in tuple(fields.items())}
    )
    report = carrier.certification
    assert report is not None
    closure = {
        "certification_inputs": report.request,
        "report": report,
        "associations": carrier.associations,
        "registration": registration,
        "generation_sources": {part.name: source, companion.name: source},
        "primary_generation_part": part.name,
        "generation_specifications": {
            part.name: specification,
            companion.name: specification,
        },
        "generation_options": {part.name: options, companion.name: options},
        "field_declarations": {
            part.name: declaration,
            companion.name: companion_declaration,
        },
        "accepted_data": {"fields": fields},
    }
    limits = ArrayArchiveLimits(
        max_members=8192, max_manifest_nesting=128, max_manifest_bytes=1 << 26
    )
    if field_kind.endswith("history"):
        corrupted = dict(fields)
        corrupted["state-epoch"] = np.asarray(8, dtype=np.int64)
        with pytest.raises(ValueError, match="state epoch"):
            validate_meshing_source_closure(
                {**closure, "accepted_data": {"fields": corrupted}},
                limits=limits,
            )
    receipt = write_meshing_source_closure(
        tmp_path / f"{field_kind}.zip", closure, limits=limits
    )
    restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id, limits=limits
    )
    cold_registration = restored["registration"].prepare()
    cold_registration.require_complete()
    cold_carrier = cold_registration.assembly.part(part.name).carrier
    assert isinstance(cold_carrier, CellMeshingResult)
    fresh_report = recertify_restored_meshing_source(
        restored["certification_inputs"],
        cold_carrier.mesh,
        cold_carrier.geometry,
        cold_carrier.audit,
        archive_limits=limits,
    )
    assert fresh_report.report_id == report.report_id
    cold_declaration = restored["field_declarations"][part.name]
    prepared, policy = prepare_meshing_field_owner(cold_declaration, cold_carrier)
    actual_coefficients = restored["accepted_data"]["fields"]["physical-coefficients"]
    np.testing.assert_array_equal(actual_coefficients, coefficients)
    assert not actual_coefficients.flags.writeable
    assert (
        cold_declaration.value_units[name][0].unit_id
        == declaration.value_units[name][0].unit_id
    )
    if field_kind.startswith("fv"):
        assert isinstance(original, UnstructuredFiniteVolumeDiscretization)
        assert isinstance(prepared, UnstructuredFiniteVolumeDiscretization)
        np.testing.assert_array_equal(prepared.cell_global_ids, original.cell_global_ids)
        np.testing.assert_allclose(
            np.sum(
                np.asarray(prepared.cell_volumes)[:, None] * actual_coefficients, axis=0
            ),
            np.sum(np.asarray(original.cell_volumes)[:, None] * coefficients, axis=0),
            atol=1e-13,
        )
        assert type(policy) is PiecewiseConstantReconstruction
        from phydrax.discretization.finite_volume import CellPolynomialReconstructionPlan

        with pytest.raises(ValueError, match="reconstruction family"):
            replace(
                cold_declaration,
                reconstruction_policy=CellPolynomialReconstructionPlan(1),
            )
        if field_kind.endswith("history"):
            np.testing.assert_array_equal(
                restored["accepted_data"]["fields"]["material-history"],
                fields["material-history"],
            )
            assert int(restored["accepted_data"]["fields"]["state-epoch"]) == 7
            history_role = next(
                role
                for role in cold_declaration.state_roles
                if role.role == "material-history"
            )
            history_binding = history_role.index_binding
            assert history_binding is not None
            reordered_bank = replace(
                history_binding,
                index_values=np.asarray(history_binding.index_values)[::-1].copy(),
            )
            wrong_roles = tuple(
                replace(role, index_binding=reordered_bank)
                if role is history_role
                else role
                for role in cold_declaration.state_roles
            )
            with pytest.raises(ValueError, match="row identity/order"):
                prepare_meshing_field_owner(
                    replace(cold_declaration, state_roles=wrong_roles), cold_carrier
                )
            dynamics = prepare_meshing_field_dynamics(cold_declaration, cold_carrier)
            original_dynamics = prepare_meshing_field_dynamics(declaration, carrier)
            cold_rate = np.asarray(dynamics(0.0, actual_coefficients))
            original_rate = np.asarray(original_dynamics(0.0, coefficients))
            np.testing.assert_allclose(cold_rate, original_rate, atol=1e-13)
            volumes = np.asarray(prepared.cell_volumes)
            np.testing.assert_allclose(np.sum(volumes * cold_rate[:, 0]), 0.0, atol=1e-13)
            np.testing.assert_allclose(
                np.sum(volumes * cold_rate[:, -1]), 0.0, atol=1e-13
            )
            next_state = actual_coefficients + 1e-3 * cold_rate
            restored_system = cold_declaration.finite_volume_system
            assert isinstance(restored_system, EulerSystem)
            assert bool(np.all(restored_system.admissible(next_state)))
            np.testing.assert_allclose(
                np.sum(volumes[:, None] * next_state[:, (0, -1)], axis=0),
                np.sum(volumes[:, None] * actual_coefficients[:, (0, -1)], axis=0),
                atol=1e-13,
            )
    else:
        assert isinstance(original, FiniteElementDiscretization)
        assert isinstance(prepared, FiniteElementDiscretization)
        for block in carrier.mesh.blocks:
            sites = np.mean(
                np.asarray(carrier.mesh.coordinates)[np.asarray(block.vertices)], axis=1
            )
            before = prepare_finite_element_field_reconstruction(
                original, name, block_name=block.name
            )
            after = prepare_finite_element_field_reconstruction(
                prepared, name, block_name=block.name
            )
            np.testing.assert_allclose(
                after.evaluate(actual_coefficients, sites).values,
                before.evaluate(coefficients, sites).values,
                atol=1e-12,
            )
            derivative = (1, 0, 0)
            np.testing.assert_allclose(
                after.derivative(actual_coefficients, sites, derivative).values,
                before.derivative(coefficients, sites, derivative).values,
                atol=1e-12,
            )
        wrong = phx.discretization.FiniteElementFieldSpec(
            name,
            {
                block.name: phx.discretization.lagrange_element(block.cell_kind, 1)
                for block in cold_carrier.mesh.blocks
            },
        )
        with pytest.raises(ValueError):
            prepare_meshing_field_owner(
                replace(cold_declaration, finite_element_fields=(wrong,)), cold_carrier
            )
        if field_kind == "mapped-scalar":

            def harmonic(points: jax.Array, /) -> jax.Array:
                return points @ jnp.asarray((1.0, 2.0, 3.0))

            form = phx.equations.FiniteElementForm(
                "cold-physical-diffusion",
                name,
                (phx.equations.DiffusionAction(name, 1.0),),
            )
            problem = phx.equations.compile_finite_element_problem(
                form,
                prepared,
                constraint=phx.discretization.dirichlet_constraint(
                    prepared,
                    name,
                    boundary_mask=restored["accepted_data"]["fields"]["essential-mask"],
                ),
                dirichlet_values=restored["accepted_data"]["fields"]["essential-values"],
            )
            operator, rhs = problem.linear_system()
            solved = phx.linalg.solve(
                operator,
                rhs,
                policy=phx.linalg.LinearSolvePolicy(
                    tolerance=phx.linalg.TolerancePolicy(relative=1e-12, absolute=1e-13),
                ),
            )
            assert bool(jnp.all(solved.successful))
            np.testing.assert_allclose(
                problem.expand(solved.value),
                harmonic(prepared.dof_maps[0].dof_coordinates),
                atol=1e-10,
            )


def test_single_native_part_physical_archive_reprepares_without_registration(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    import phydrax as phx
    from phydrax._array_archive import ArrayArchiveLimits
    from phydrax._differentiation import BranchDifferentiationPolicy
    from phydrax._frozendict import frozendict
    from phydrax.discretization._views import FieldTracePolicy
    from phydrax.discretization.fem._generic import (
        FiniteElementFieldSpec,
        FiniteElementPlan,
    )
    from phydrax.discretization.fem._precision import FiniteElementPrecisionPolicy
    from phydrax.discretization.fem._reference import lagrange_element
    from phydrax.lifecycle._meshing_field_records import (
        MeshingFieldDeclaration,
        MeshingFieldStateRole,
        prepare_meshing_field_index_binding,
        prepare_meshing_field_owner,
        validate_meshing_field_declarations,
    )
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )

    part, source, specification, options = _native_family_generation("plc")
    carrier = part.carrier
    assert isinstance(carrier, phx.meshing.CellMeshingResult)
    field = FiniteElementFieldSpec("temperature", lagrange_element("tetrahedron", 1))
    precision = FiniteElementPrecisionPolicy()
    original = FiniteElementPlan(
        carrier.mesh,
        (field,),
        coordinate_spec=carrier.geometry,
        precision_policy=precision,
    ).prepare(numeric_version="accepted-7")
    coefficient_space = original.field_spaces[0].vector_space
    assert isinstance(coefficient_space, phx.linalg.ArraySpace)
    shape = coefficient_space.shape
    coefficients = np.full(shape, 42.0, dtype=np.float64)
    history = np.full(shape, 37.0, dtype=np.float64)
    coefficient_binding = prepare_meshing_field_index_binding(
        carrier, "mesh/vertex_ids", field_space=original.field_spaces[0]
    )
    history_binding = prepare_meshing_field_index_binding(carrier, "mesh/vertex_ids")
    roles = (
        MeshingFieldStateRole(
            "temperature",
            "temperature",
            "coefficients",
            shape,
            np.dtype(np.float64),
            7,
            index_binding=coefficient_binding,
        ),
        MeshingFieldStateRole(
            "material-temperature",
            "temperature",
            "material-history",
            shape,
            np.dtype(np.float64),
            7,
            value_units=(phx.units.KELVIN,),
            index_binding=history_binding,
        ),
        MeshingFieldStateRole(
            "accepted-epoch", "temperature", "state-epoch", (), np.dtype(np.int64), 7
        ),
    )
    declaration = MeshingFieldDeclaration(
        part_name=part.name,
        owner="finite_element",
        topology_id=carrier.mesh.topology_id,
        geometry_layout_id=carrier.geometry.geometry_layout_id,
        field_space_ids=frozendict(
            {"temperature": original.field_spaces[0].field_space_id}
        ),
        value_units=frozendict({"temperature": (phx.units.KELVIN,)}),
        maximum_derivative_orders=frozendict({"temperature": 1}),
        branch_policy=BranchDifferentiationPolicy.SMOOTH,
        trace_policy=FieldTracePolicy("cell-sided"),
        state_roles=roles,
        history_policy="material-history",
        numeric_version="accepted-7",
        finite_element_fields=(field,),
        precision_policy=precision,
    )
    report = carrier.certification
    assert report is not None
    closure = {
        "certification_inputs": report.request,
        "report": report,
        "associations": carrier.associations,
        "generation_part": part,
        "generation_source": source,
        "generation_specification": specification,
        "generation_options": options,
        "field_declarations": {part.name: declaration},
        "accepted_data": {
            "fields": {
                "temperature": coefficients,
                "material-temperature": history,
                "accepted-epoch": np.asarray(7, dtype=np.int64),
            }
        },
    }
    validate_meshing_field_declarations(closure)
    limits = ArrayArchiveLimits(
        max_members=8192, max_manifest_nesting=128, max_manifest_bytes=1 << 26
    )
    receipt = write_meshing_source_closure(
        tmp_path / "native-physical.zip", closure, limits=limits
    )
    restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id, limits=limits
    )
    assert "registration" not in restored
    prepared, _ = prepare_meshing_field_owner(
        restored["field_declarations"][part.name], restored["generation_part"].carrier
    )
    assert (
        prepared.field_spaces[0].field_space_id == original.field_spaces[0].field_space_id
    )
    np.testing.assert_array_equal(
        restored["accepted_data"]["fields"]["temperature"], coefficients
    )
    np.testing.assert_array_equal(
        restored["accepted_data"]["fields"]["material-temperature"], history
    )
    assert int(np.asarray(restored["accepted_data"]["fields"]["accepted-epoch"])) == 7
    stale = dict(restored)
    stale["field_declarations"] = {
        part.name: replace(
            restored["field_declarations"][part.name],
            topology_id="other-accepted-topology",
        )
    }
    with pytest.raises(ValueError):
        validate_meshing_field_declarations(stale)
    wrong_epoch = dict(restored)
    wrong_epoch["accepted_data"] = {
        "fields": {
            **restored["accepted_data"]["fields"],
            "accepted-epoch": np.asarray(8, dtype=np.int64),
        }
    }
    with pytest.raises(ValueError):
        validate_meshing_field_declarations(wrong_epoch)


def _implicit_association_transfer() -> Any:
    import phydrax as phx
    from phydrax.geometry.implicit._analytic_profile import AnalyticImplicitProfile
    from phydrax.meshing._implicit_association_transfer import ImplicitAssociationTransfer

    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 0.371, feature_id="archive-sphere"
    ).compile()
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
        atol=1.0e-14,
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
            # An authority entry no registered source field owns.
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
            raise AssertionError(f"Uncovered manifest malformation {malformation!r}.")
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
