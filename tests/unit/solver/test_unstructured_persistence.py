#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import hashlib
import json
import zipfile
from importlib import import_module
from importlib.util import find_spec

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _unstructured_runtime():
    vertices = np.asarray(
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (2.0, 1.0),
        )
    )
    triangles = np.asarray(((1, 2, 5), (1, 5, 4)), dtype=np.int32)
    quadrilaterals = np.asarray(((0, 1, 4, 3),), dtype=np.int32)
    system = phx.equations.EulerSystem(2)
    plan = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        triangles=triangles,
        quadrilaterals=quadrilaterals,
        vertex_global_ids=np.arange(100, 106, dtype=np.int64),
        cell_global_ids=np.asarray((901, 903, 907), dtype=np.int64),
        component_names=system.component_names,
    )
    discretization = plan.prepare()
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {
            name: phx.discretization.ExtrapolationBoundary()
            for name in discretization.boundary_patch_names
        },
    )
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.HLLCFluxPlan(),
    )
    problem = phx.equations.ConservationProblemIR(
        "persistent-unstructured", "state", system, boundaries
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem, discretization, method
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics, phx.discretization.FluxPositivityPlan()
    )
    primitive = jnp.broadcast_to(
        jnp.asarray((1.0, 0.1, -0.05, 1.0)), discretization.state_shape
    )
    runtime_state = runtime.initialize_state(
        system.primitive_to_conserved(primitive),
        0.2,
        0.001,
        accepted_step=7,
        controller_state=jnp.asarray((0.3, 0.4)),
        integrator_state=jnp.asarray((1.2,)),
        forcing_state=jnp.asarray((2.5,)),
        random_state=jnp.asarray((11, 17), dtype=jnp.uint32),
        output_cursor=4,
    )
    execution = phx.solver.FiniteVolumeExecutionSpec(1.0, 1000)
    case = phx.solver.FiniteVolumeCaseSpec("persistent-unstructured", runtime, execution)
    return plan, discretization, runtime, runtime_state, case


def _moving_sliding_runtime():
    system = phx.equations.EulerSystem(2)
    vertices = np.asarray([(i / 2.0, j / 2.0) for j in range(3) for i in range(3)])
    cells = np.asarray(
        (
            (0, 1, 4, 3),
            (1, 2, 5, 4),
            (3, 4, 7, 6),
            (4, 5, 8, 7),
        ),
        dtype=np.int32,
    )
    mesh_plan = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        quadrilaterals=cells,
        vertex_global_ids=np.arange(100, 109),
        cell_global_ids=np.arange(500, 504),
        component_names=system.component_names,
    )
    discretization = mesh_plan.prepare()
    face_ids = np.asarray((9, 7, 10), dtype=np.int32)
    face_cells = np.asarray((2, 2, 3), dtype=np.int32)
    owners = np.asarray(discretization.owner_cells)[face_ids]
    neighbours = np.asarray(discretization.neighbour_cells)[face_ids]
    orientation = np.where(owners == face_cells, 1.0, -1.0)
    unit_normals = (
        np.asarray(discretization.area_vectors)[face_ids]
        / np.asarray(discretization.face_measures)[face_ids, None]
    )
    face_points = np.asarray(discretization.face_quadrature_points)[face_ids]
    face_normals = np.broadcast_to(
        orientation[:, None, None] * unit_normals[:, None, :],
        face_points.shape,
    )
    overset = phx.discretization.UnstructuredOversetPlan(
        discretization,
        discretization,
        np.asarray((2, 3), dtype=np.int32),
        np.asarray((0, 1, 2), dtype=np.int32),
        np.asarray((0, 1), dtype=np.int32),
        np.asarray((0.25, 0.25)),
        epoch_id="checkpoint-sliding-epoch",
        receptor_face_ids=face_ids,
        receptor_face_points=face_points,
        receptor_face_normals=face_normals,
        receptor_face_measures=np.asarray(discretization.face_quadrature_weights)[
            face_ids
        ],
        receptor_face_cells=face_cells,
    )

    def translation(time, points, args):
        del args
        return points.at[:, 0].add(0.2 * time)

    motion = phx.discretization.FixedConnectivityMotionPlan(
        mesh_plan,
        translation,
        mapping_id="checkpoint-sliding-motion",
    )
    sliding = phx.discretization.PeriodicSlidingInterfacePlan(
        np.asarray((0.0, 0.5, 1.0)),
        np.asarray((0.0, 0.25, 0.75, 1.0)),
        1.0,
        interface_id="checkpoint-seam",
    )
    coupling = phx.discretization.UnstructuredFiniteVolumeCouplingPlan(
        motion=motion,
        overset=overset,
        sliding=sliding,
        topology_event_capacity=8,
        topology_event_policy="accepted_step",
    )

    def wall_velocity(time, points, normals, args):
        del time, points, normals, args
        return jnp.asarray((0.2, 0.0))

    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {
            name: phx.discretization.MovingSlipWallBoundary(
                wall_velocity,
                wall_velocity_provider_id=f"checkpoint-sliding-motion:{name}",
            )
            for name in discretization.boundary_patch_names
        },
    )
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    problem = phx.equations.ConservationProblemIR(
        "checkpoint-sliding-motion",
        "state",
        system,
        boundaries,
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem,
        discretization,
        method,
        coupling=coupling,
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(),
    )
    primitive = jnp.asarray(
        (
            (1.0, 0.25, 0.0, 1.0),
            (2.0, -0.15, 0.0, 1.2),
            (1.4, 0.0, 0.0, 1.1),
            (0.8, 0.0, 0.0, 0.9),
        )
    )
    initial = runtime.initialize_state(
        system.primitive_to_conserved(primitive),
        0.0,
        1.0e-3,
    )
    return discretization, runtime, initial


def test_unstructured_mesh_archive_preserves_stable_identity(tmp_path):
    plan, _, _, _, _ = _unstructured_runtime()
    path = tmp_path / "mesh.fvmesh"
    phx.discretization.write_unstructured_fv_archive(
        path,
        plan,
        provenance={"importer": "native-test", "source": "generated"},
    )
    restored = phx.discretization.read_unstructured_fv_archive(path)

    assert restored.plan_id == plan.plan_id
    assert restored.topology_id == plan.topology_id
    assert restored.geometry_id == plan.geometry_id
    np.testing.assert_array_equal(restored.vertices, plan.vertices)
    np.testing.assert_array_equal(restored.vertex_global_ids, plan.vertex_global_ids)
    np.testing.assert_array_equal(restored.cell_global_ids, plan.cell_global_ids)
    with pytest.raises(ValueError, match="unique"):
        phx.discretization.UnstructuredFiniteVolumePlan(
            plan.vertices,
            triangles=plan.triangles,
            quadrilaterals=plan.quadrilaterals,
            vertex_global_ids=np.asarray((1, 1, 2, 3, 4, 5)),
        )


def test_unstructured_case_loader_checkpoint_and_mesh_compatibility(tmp_path):
    plan, _, _, state, _ = _unstructured_runtime()
    mesh_path = tmp_path / "mesh.fvmesh"
    phx.discretization.write_unstructured_fv_archive(mesh_path, plan)
    checksum = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
    payload = {
        "schema_version": 2,
        "name": "loaded-unstructured",
        "mesh": {"path": mesh_path.name, "sha256": checksum},
        "equation": {
            "type": "ideal_gas_euler",
            "gamma": 1.4,
            "gas_constant": 1.0,
        },
        "method": {"reconstruction": "piecewise_constant", "flux": "hllc"},
        "boundary": {"boundary": {"type": "extrapolation"}},
        "execution": {"end_time": 0.1, "maximum_steps": 100},
        "precision": {"dtype": "float64"},
        "initial_state": {
            "type": "constant_primitive",
            "values": [1.0, 0.1, -0.05, 1.0],
        },
    }
    prepared = phx.solver.load_finite_volume_case(
        payload, source_path=tmp_path / "case.json"
    )
    assert prepared.case.schema_version == 2
    assert prepared.case.mesh_topology_id == plan.topology_id
    assert prepared.case.mesh_geometry_id == plan.geometry_id
    assert prepared.initial_state is not None
    assert prepared.initial_state.shape == prepared.discretization.state_shape

    checkpoint_plan = phx.solver.FiniteVolumeCheckpointPlan(prepared.case)
    checkpoint_path = tmp_path / "restart.fvckpt"
    phx.solver.write_finite_volume_checkpoint(checkpoint_path, checkpoint_plan, state)
    restored = phx.solver.read_finite_volume_checkpoint(checkpoint_path, checkpoint_plan)
    np.testing.assert_array_equal(
        restored.runtime_state.content_state.conservative_content,
        state.content_state.conservative_content,
    )
    np.testing.assert_array_equal(
        restored.runtime_state.cell_average(), state.cell_average()
    )
    np.testing.assert_array_equal(
        restored.runtime_state.content_state.effective_cell_volumes,
        state.content_state.effective_cell_volumes,
    )
    np.testing.assert_array_equal(
        restored.runtime_state.content_state.active_cell_mask,
        state.content_state.active_cell_mask,
    )
    assert (
        restored.runtime_state.content_state.topology_epoch_id
        == state.content_state.topology_epoch_id
    )
    assert (
        restored.runtime_state.content_state.geometry_layout_id
        == state.content_state.geometry_layout_id
    )
    assert (
        restored.runtime_state.content_state.geometry_version
        == state.content_state.geometry_version
    )
    assert (
        restored.runtime_state.content_state.evidence_policy_id
        == state.content_state.evidence_policy_id
    )
    assert (
        restored.runtime_state.content_state.evidence_version
        == state.content_state.evidence_version
    )
    with zipfile.ZipFile(checkpoint_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["schema_version"] == 5
    assert manifest["runtime_state_schema_version"] == 4
    assert manifest["content_state_schema_version"] == 2
    assert manifest["mesh"]["topology_id"] == plan.topology_id

    bad_payload = {
        **payload,
        "mesh": {"path": mesh_path.name, "sha256": "0" * 64},
    }
    with pytest.raises(ValueError, match="sha256"):
        phx.solver.load_finite_volume_case(
            bad_payload, source_path=tmp_path / "case.json"
        )


def test_unstructured_hdf5_xdmf_and_vtk_outputs_are_self_describing(tmp_path):
    _, discretization, _, state, _ = _unstructured_runtime()
    output = phx.solver.FiniteVolumeOutputPlan(tmp_path / "solution.h5", discretization)
    if find_spec("h5py") is None:
        with pytest.raises(ImportError, match="h5py"):
            output.write_snapshot(discretization, state)
        return

    index = output.write_snapshot(discretization, state)
    assert index == 0
    h5py = import_module("h5py")
    with h5py.File(output.hdf5_path, "r") as handle:
        assert handle.attrs["schema_version"] == 4
        assert handle.attrs["topology_id"] == discretization.topology_id
        np.testing.assert_array_equal(handle["mesh/points"], discretization.vertices)
        np.testing.assert_array_equal(
            handle["mesh/cell_global_ids"], discretization.cell_global_ids
        )
        step = handle["steps/00000000"]
        assert "conservative_state" not in step
        np.testing.assert_array_equal(
            step["conservative_content"],
            state.content_state.conservative_content,
        )
        np.testing.assert_array_equal(step["cell_average"], state.cell_average())
        np.testing.assert_array_equal(
            step["effective_cell_volumes"],
            state.content_state.effective_cell_volumes,
        )
        np.testing.assert_array_equal(
            step["active_cell_mask"],
            state.content_state.active_cell_mask,
        )
        assert step.attrs["topology_epoch_id"] == state.content_state.topology_epoch_id
        assert step.attrs["geometry_layout_id"] == state.content_state.geometry_layout_id
        assert step.attrs["geometry_version"] == state.content_state.geometry_version
        assert step.attrs["evidence_policy_id"] == state.content_state.evidence_policy_id
        assert step.attrs["evidence_version"] == state.content_state.evidence_version
        geometry_path = step.attrs["geometry_points_path"]
        np.testing.assert_array_equal(
            handle[geometry_path],
            discretization.vertices,
        )
    xdmf = (tmp_path / "solution.xdmf").read_text()
    assert 'TopologyType="Triangle"' in xdmf
    assert 'TopologyType="Quadrilateral"' in xdmf
    for component in discretization.component_names:
        assert f'Attribute Name="{component}" AttributeType="Scalar"' in xdmf
    assert "/cell_average" in xdmf
    assert "conservative_state" not in xdmf

    vtk_path = output.write_vtk_snapshot(tmp_path / "solution.vtu", discretization, state)
    meshio = import_module("meshio")
    mesh = meshio.read(vtk_path)
    assert sum(block.data.shape[0] for block in mesh.cells) == discretization.cell_count
    assert "cell_global_id" in mesh.cell_data
    assert "vertex_global_id" in mesh.point_data
    average = np.asarray(state.cell_average())
    for component_index, component in enumerate(discretization.component_names):
        np.testing.assert_array_equal(
            np.concatenate(mesh.cell_data[component]),
            average[:, component_index],
        )


def test_sliding_checkpoint_preserves_successor_identity_and_advance(tmp_path):
    discretization, runtime, initial = _moving_sliding_runtime()
    accepted = runtime.advance(initial, {"sliding_shift": 0.2})
    assert bool(np.asarray(accepted.accepted))
    assert accepted.successor_runtime is not None
    state = accepted.runtime_state
    assert state.sliding_coupling is not None
    assert state.sliding_event_id is not None
    case = phx.solver.FiniteVolumeCaseSpec(
        "checkpoint-sliding-motion",
        runtime,
        phx.solver.FiniteVolumeExecutionSpec(1.0, 1000),
    )
    checkpoint_plan = phx.solver.FiniteVolumeCheckpointPlan(
        case,
        runtime=runtime,
    )
    path = tmp_path / "sliding.fvckpt"

    with pytest.raises(ValueError, match="originating prepared runtime"):
        phx.solver.write_finite_volume_checkpoint(
            tmp_path / "sliding-without-plan.fvckpt",
            phx.solver.FiniteVolumeCheckpointPlan(case),
            state,
        )

    phx.solver.write_finite_volume_checkpoint(path, checkpoint_plan, state)
    restored = phx.solver.read_finite_volume_checkpoint(
        path,
        checkpoint_plan,
    ).runtime_state

    assert restored.sliding_event_id == state.sliding_event_id
    assert restored.sliding_coupling_id == state.sliding_coupling_id
    np.testing.assert_array_equal(restored.sliding_shift, state.sliding_shift)
    for name in (
        "left_routes",
        "right_routes",
        "overlap_measures",
        "left_measures",
        "right_measures",
    ):
        np.testing.assert_array_equal(
            getattr(restored.sliding_coupling, name),
            getattr(state.sliding_coupling, name),
        )
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["sliding"]["event_id"] == state.sliding_event_id
    assert manifest["sliding"]["coupling_id"] == state.sliding_coupling_id
    assert manifest["sliding"]["evidence_id"] == state.sliding_coupling.evidence_id
    assert "sliding/overlap_measures" in manifest["arrays"]

    continued = accepted.successor_runtime.advance(
        restored,
        {"sliding_shift": 0.2},
    )
    assert bool(np.asarray(continued.accepted))
    assert continued.runtime_state.content_state.conservative_content.shape == (
        discretization.state_shape
    )


def test_ale_outputs_use_accepted_points_and_reject_stale_geometry(tmp_path):
    h5py = pytest.importorskip("h5py")
    meshio = import_module("meshio")
    discretization, runtime, initial = _moving_sliding_runtime()
    accepted = runtime.advance(initial, {"sliding_shift": 0.2})
    assert bool(np.asarray(accepted.accepted))
    assert accepted.ale is not None
    state = accepted.runtime_state
    geometry = accepted.ale.geometry.end_geometry
    moved_points = np.asarray(geometry.vertices)
    assert not np.array_equal(moved_points, np.asarray(discretization.vertices))
    output = phx.solver.FiniteVolumeOutputPlan(
        tmp_path / "ale-solution.h5",
        discretization,
    )

    with pytest.raises(ValueError, match="requires accepted geometry"):
        output.write_snapshot(discretization, state)
    with pytest.raises(ValueError, match="requires accepted geometry"):
        output.write_vtk_snapshot(
            tmp_path / "missing-geometry.vtu",
            discretization,
            state,
        )
    stale = runtime.dynamics.coupling.motion.geometry_state(
        initial.time,
        geometry_version=initial.content_state.geometry_version,
    )
    with pytest.raises(ValueError, match="stale"):
        output.write_snapshot(
            discretization,
            state,
            accepted_geometry=stale,
        )

    output.write_snapshot(
        discretization,
        state,
        accepted_geometry=geometry,
    )
    with h5py.File(output.hdf5_path, "r") as handle:
        step = handle["steps/00000000"]
        geometry_path = step.attrs["geometry_points_path"]
        np.testing.assert_array_equal(handle[geometry_path], moved_points)
        assert (
            handle[geometry_path].parent.attrs["geometry_version"]
            == state.content_state.geometry_version
        )
        assert (
            handle[geometry_path].parent.parent.parent.attrs["topology_epoch_id"]
            == state.content_state.topology_epoch_id
        )
    xdmf = (tmp_path / "ale-solution.xdmf").read_text()
    assert f"ale-solution.h5:{geometry_path}" in xdmf
    assert "ale-solution.h5:/mesh/points" not in xdmf

    vtk_path = output.write_vtk_snapshot(
        tmp_path / "ale-solution.vtu",
        discretization,
        state,
        accepted_geometry=moved_points,
    )
    vtk = meshio.read(vtk_path)
    np.testing.assert_array_equal(vtk.points[:, : moved_points.shape[1]], moved_points)
