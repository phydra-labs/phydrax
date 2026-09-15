#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._array_archive import read_array_archive


def _identity(points, time, args):
    del time, args
    return points


def _plane(points, time, args):
    del time, args
    return points[:, 0] - 0.37


def _cut_plan():
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(1) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    signature = phx.discretization.PatchShapeSignature((1, 1, 1), halo_width=1)
    hierarchy = phx.discretization.VariablePatchHierarchyPlan(
        grid,
        (
            phx.discretization.VariablePatchLevelPlan(
                0,
                (phx.discretization.PatchBucketPlan(signature, 1),),
            ),
        ),
        (phx.discretization.LogicalPatchBox(0, (0, 0, 0), (1, 1, 1)),),
    )
    topology = phx.discretization.VariablePatchTopologyCompiler(
        hierarchy
    ).initial_topology()
    body = phx.discretization.EmbeddedLevelSetBody(_plane, "restart-plane", 8)
    resources = phx.discretization.BlockAMRResourcePlan(
        maximum_components_per_cell=2,
        maximum_apertures_per_face=16,
        maximum_embedded_faces_per_cell=32,
    )
    return phx.discretization.MultivaluedCutCellPlan(
        topology,
        _identity,
        "identity-map",
        phx.discretization.EmbeddedLevelSetBodySet((body,)),
        resources,
    )


def test_checkpoint_reconstructs_topology_without_caller_plan(tmp_path):
    cut_plan = _cut_plan()
    moving = phx.solver.MovingMultivaluedCutCellPlan(cut_plan)
    state = moving.initialize(jnp.asarray(((2.0, 3.0),)), 0.25)
    checkpoint_plan = phx.solver.MultivaluedBlockAMRCheckpointPlan(
        cut_plan,
        ("first", "second"),
    )
    path = tmp_path / "cut-checkpoint"

    written = phx.solver.write_multivalued_block_amr_checkpoint(
        path,
        checkpoint_plan,
        state,
    )
    restored = phx.solver.read_multivalued_block_amr_checkpoint(
        path,
        phx.solver.CutCellRestartRegistry(
            {"identity-map": _identity},
            {"restart-plane": _plane},
        ),
    )

    assert restored.payload_id == written.payload_id
    assert restored.state.complex.topology_id == state.complex.topology_id
    assert restored.state.complex.geometry_id == state.complex.geometry_id
    np.testing.assert_array_equal(restored.state.content, state.content)
    np.testing.assert_array_equal(restored.state.time, state.time)
    np.testing.assert_array_equal(restored.state.revision, state.revision)


def test_output_snapshot_preserves_multivalued_connectivity(tmp_path):
    state = phx.solver.MovingMultivaluedCutCellPlan(_cut_plan()).initialize(
        jnp.asarray(((2.0, 3.0),)),
        0.5,
    )
    path = tmp_path / "cut-output"

    snapshot = phx.solver.write_multivalued_cut_cell_output(
        path,
        state,
        ("first", "second"),
    )
    manifest, arrays = read_array_archive(path)

    assert manifest["payload_id"] == snapshot.payload_id
    assert manifest["topology_id"] == state.complex.topology_id
    assert manifest["geometry_id"] == state.complex.geometry_id
    assert manifest["component_names"] == ["first", "second"]
    np.testing.assert_array_equal(arrays["content"], state.content)
    assert arrays["mesh_cell_face_offsets"].size == state.complex.component_count + 1
