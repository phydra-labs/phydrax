#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._execution_resources import ExecutionGroupSpec
from phydrax._execution_runtime import ExecutionGroup
from phydrax.discretization._cell_mesh import CellBlock, CellMesh, CellMeshStorage
from phydrax.discretization._distributed_field import DistributedHaloPlan
from phydrax.discretization.finite_volume._unstructured_dynamics import (
    execute_owner_local_finite_volume_advance,
    PreparedUnstructuredFiniteVolumeDynamics,
)


def _compiled(cells: Any = 16) -> Any:
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(cells, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    system = phx.equations.ScalarConservationSystem(
        1,
        lambda state, axis, args: state,
        lambda left, right, axis, args: jnp.ones(left.shape[:-1]),
        system_id="distributed-advection",
    )
    problem = phx.equations.ConservationProblemIR(
        "distributed-advection",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(("x",)),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    return phx.equations.compile_conservation_problem(problem, discretization, method)


def test_finite_volume_distributed_scenario_1() -> None:
    plan = phx.discretization.FiniteVolumeDecompositionPlan(
        (16,), (1,), ("x",), halo_width=2
    )
    prepared = plan.prepare((jax.devices()[0],))
    state = prepared.shard_state(jnp.ones((16, 1)))

    assert prepared.local_shape == (16,)
    assert prepared.report.device_count == 1
    assert state.sharding == prepared.cell_sharding
    compiled = _compiled()
    decomposition = phx.discretization.FiniteVolumeDecompositionPlan(
        (16,), (1,), ("x",), halo_width=1
    ).prepare((jax.devices()[0],))
    x = compiled.discretization.grid.structured_axes[0].interval_centers
    state = jnp.sin(2.0 * jnp.pi * x)[:, None]
    sharded = decomposition.shard_state(state)
    distributed = decomposition.residual(compiled.dynamics, 0.0, sharded)
    local = compiled(0.0, state)

    np.testing.assert_allclose(distributed, local, rtol=1e-12, atol=1e-12)
    decomposition = phx.discretization.FiniteVolumeDecompositionPlan(
        (8,), (1,), ("x",), halo_width=2
    ).prepare((jax.devices()[0],))
    state = decomposition.shard_state(jnp.arange(8.0)[:, None])
    halo = decomposition.periodic_halo(state, 0)

    np.testing.assert_allclose(halo[:2, 0], [6.0, 7.0])
    np.testing.assert_allclose(halo[-2:, 0], [0.0, 1.0])


def test_decomposition_rejects_nondivisible_or_halo_dominated_domains() -> None:
    with pytest.raises(ValueError, match="divide exactly"):
        phx.discretization.FiniteVolumeDecompositionPlan(
            (10,), (3,), ("x",), halo_width=1
        )
    with pytest.raises(ValueError, match="smaller"):
        phx.discretization.FiniteVolumeDecompositionPlan((8,), (4,), ("x",), halo_width=2)


@pytest.mark.skipif(
    len(jax.devices()) < 2,
    reason="requires at least two JAX devices",
)
def test_two_device_routes_and_residual_match_single_device() -> None:
    compiled = _compiled(16)
    decomposition = phx.discretization.FiniteVolumeDecompositionPlan(
        (16,), (2,), ("x",), halo_width=1
    ).prepare(jax.devices()[:2])
    x = compiled.discretization.grid.structured_axes[0].interval_centers
    state = jnp.sin(2.0 * jnp.pi * x)[:, None]
    sharded = decomposition.shard_state(state)
    distributed = decomposition.residual(compiled.dynamics, 0.0, sharded)

    assert len(decomposition.halo_routes) == 2
    assert {route.side for route in decomposition.halo_routes} == {
        "lower",
        "upper",
    }
    np.testing.assert_allclose(distributed, compiled(0.0, state), rtol=1e-12, atol=1e-12)


def _owner_local_fv(part_count: int, part: int) -> tuple[Any, Any, Any]:
    system = phx.equations.EulerSystem(2)
    dense = phx.discretization.UnstructuredFiniteVolumePlan(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5))),
        triangles=np.asarray(
            ((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)), dtype=np.int32
        ),
        vertex_global_ids=np.asarray((101, 103, 107, 109, 113), dtype=np.int64),
        cell_global_ids=np.asarray((211, 223, 227, 229), dtype=np.int64),
        component_names=system.component_names,
    ).prepare()
    ids = tuple(entity.entity_ids for entity in dense.mesh.topology.entity_sets)
    cells_owner = np.arange(4, dtype=np.int32) % part_count
    face_owner = cells_owner[np.asarray(dense.owner_cells)]
    storage = CellMeshStorage(
        tuple(entity.count for entity in dense.mesh.topology.entity_sets),
        ids,
        (np.zeros((5,), dtype=np.int32), face_owner, cells_owner),
        partition_index=part,
        partition_count=part_count,
        logical_topology_id=dense.mesh.topology_id,
        logical_geometry_id=dense.mesh.geometry_id,
        evidence_id="a" * 64,
        logical_arrays=(("coordinates", dense.mesh.coordinates),),
        local_coordinates=dense.mesh.coordinates,
        local_blocks=dense.mesh.blocks,
    )
    mesh = CellMesh(
        dense.mesh.coordinates,
        (
            CellBlock(
                "triangles", "triangle", dense.triangles, global_ids=dense.cell_global_ids
            ),
        ),
        storage=storage,
    )
    geometry = phx.discretization.UnstructuredFiniteVolumePlan.from_cell_mesh(
        mesh,
        component_names=system.component_names,
        neighborhood_complete=np.ones((4,), dtype=np.bool_),
        physical_boundary_faces=np.asarray(dense.neighbor_cells) < 0,
        closure_evidence_id=storage.evidence_id,
    ).prepare(owned_face_capacity=None if part_count == 1 else dense.face_measures.size)
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        geometry.boundary_patch_names,
        {"boundary": phx.discretization.SlipWallBoundary()},
    )
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    dynamics = PreparedUnstructuredFiniteVolumeDynamics(
        system, geometry, method, boundaries
    )
    if part_count == 1:
        sends = receives = np.empty((0, 1), dtype=np.int32)
        valid = np.empty((0, 1), dtype=np.bool_)
        permutations: tuple[tuple[tuple[int, int], ...], ...] = ()
    else:
        sends = np.tile(np.asarray((part, part + 2), dtype=np.int32), (2, 1))
        receives = np.tile(np.asarray((1 - part, 3 - part), dtype=np.int32), (2, 1))
        valid = np.asarray(((part == 0,) * 2, (part == 1,) * 2), dtype=np.bool_)
        permutations = (((0, 1),), ((1, 0),))
    receive_valid = valid if part_count == 1 else ~valid
    halo = DistributedHaloPlan(
        None,
        None,
        part_count,
        local_global_ids=geometry.cell_global_ids,
        local_owned=geometry.cell_owned,
        global_entity_count=4,
        partition_index=part,
        phase_send_indices=sends,
        phase_receive_indices=receives,
        phase_send_valid=valid,
        phase_receive_valid=receive_valid,
        permutations=permutations,
        evidence_id=storage.evidence_id,
    )
    state = system.primitive_to_conserved(
        jnp.asarray(
            (
                (1.0, 0.0, 0.0, 1.0),
                (2.0, 0.0, 0.0, 1.0),
                (1.0, 0.0, 0.0, 1.0),
                (2.0, 0.0, 0.0, 1.0),
            )
        )
    )
    return dynamics, halo, state


def test_owner_local_fv_flux_advance_conserves_mass() -> None:
    dynamics, halo, state = _owner_local_fv(1, 0)
    geometry = dynamics.discretization
    advance = jax.pmap(
        lambda value: dynamics.owner_local_advance(
            jnp.asarray(0.0), value, jnp.asarray(0.01), halo, axis_name="parts"
        ),
        axis_name="parts",
        devices=(jax.devices()[0],),
    )
    updated = advance(state[None, ...])[0]
    expected = state + 0.01 * dynamics(jnp.asarray(0.0), state)
    np.testing.assert_allclose(updated, expected, rtol=1e-12, atol=1e-12)
    assert float(updated[0, 0]) > float(state[0, 0])
    np.testing.assert_allclose(
        jnp.sum(geometry.cell_volumes * updated[:, 0]),
        jnp.sum(geometry.cell_volumes * state[:, 0]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_owner_local_fv_rejects_missing_or_incomplete_closure() -> None:
    dynamics, _, _ = _owner_local_fv(2, 0)
    geometry = dynamics.discretization
    plan = phx.discretization.UnstructuredFiniteVolumePlan
    with pytest.raises(ValueError, match="closure evidence"):
        plan.from_cell_mesh(geometry.mesh)
    with pytest.raises(ValueError, match="incomplete"):
        plan.from_cell_mesh(
            geometry.mesh,
            neighborhood_complete=np.asarray((True, False, True, False), dtype=np.bool_),
            physical_boundary_faces=np.asarray(geometry.neighbor_cells) < 0,
            closure_evidence_id=geometry.mesh.storage.evidence_id,
        )
    with pytest.raises(ValueError, match="frontier"):
        plan.from_cell_mesh(
            geometry.mesh,
            neighborhood_complete=np.ones((4,), dtype=np.bool_),
            physical_boundary_faces=np.zeros(
                geometry.neighbor_cells.shape, dtype=np.bool_
            ),
            closure_evidence_id=geometry.mesh.storage.evidence_id,
        )
    with pytest.raises(ValueError, match="named-axis"):
        dynamics(jnp.asarray(0.0), jnp.ones(geometry.state_shape))


@pytest.mark.skipif(jax.local_device_count() < 2, reason="requires two local JAX devices")
def test_two_device_owner_local_fv_updates_only_owned_cells_and_conserves_mass() -> None:
    left, left_halo, state = _owner_local_fv(2, 0)
    right, right_halo, _ = _owner_local_fv(2, 1)
    serial, _, _ = _owner_local_fv(1, 0)

    def execution_data(dynamics: Any, halo: Any) -> Any:
        geometry = dynamics.discretization
        return (
            geometry.cell_owned,
            geometry.partition_index,
            geometry.owned_face_indices,
            geometry.owned_face_valid,
            halo,
        )

    # Only local execution leaves enter the device map. Canonical logical
    # checkpoint arrays remain outside map arguments and are never replicated.
    prototype = execution_data(left, left_halo)
    static = eqx.filter(prototype, lambda value: not eqx.is_array(value))
    left_numeric, structure = jax.tree_util.tree_flatten(
        eqx.filter(prototype, eqx.is_array)
    )
    right_numeric = jax.tree_util.tree_leaves(
        eqx.filter(execution_data(right, right_halo), eqx.is_array)
    )
    numeric = tuple(
        jnp.stack((first, second))
        for first, second in zip(left_numeric, right_numeric, strict=True)
    )

    def advance(local_numeric: Any, value: Any) -> Any:
        owned_mask, part, faces, face_valid, halo = eqx.combine(
            jax.tree_util.tree_unflatten(structure, local_numeric), static
        )
        geometry = eqx.tree_at(
            lambda value: (
                value.cell_owned,
                value.partition_index,
                value.owned_face_indices,
                value.owned_face_valid,
            ),
            left.discretization,
            (owned_mask, part, faces, face_valid),
        )
        dynamics = eqx.tree_at(lambda value: value.discretization, left, geometry)
        return dynamics.owner_local_advance(
            jnp.asarray(0.0), value, jnp.asarray(0.01), halo, axis_name="parts"
        )

    # Ghost seeds differ from their owner values, so the result also proves
    # the forward exchange precedes the exactly-once interface flux solve.
    seeds = jnp.stack((state.at[1::2].set(state[::2]), state.at[::2].set(state[1::2])))
    updated = jax.pmap(advance, axis_name="parts", devices=jax.devices()[:2])(
        numeric, seeds
    )
    owned = updated[0].at[1::2].set(updated[1, 1::2])
    expected = state + 0.01 * serial(jnp.asarray(0.0), state)
    np.testing.assert_allclose(owned, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(updated[0, 1::2], seeds[0, 1::2])
    np.testing.assert_array_equal(updated[1, ::2], seeds[1, ::2])
    volumes = serial.discretization.cell_volumes
    np.testing.assert_allclose(
        jnp.sum(volumes * owned[:, 0]), jnp.sum(volumes * state[:, 0]), atol=1e-12
    )


@pytest.mark.skipif(jax.local_device_count() < 2, reason="requires two local JAX devices")
def test_owner_local_fv_executor_respects_reversed_placement_and_owned_inventory() -> (
    None
):
    left, left_halo, state = _owner_local_fv(2, 0)
    right, right_halo, _ = _owner_local_fv(2, 1)
    devices = tuple(reversed(jax.devices()[:2]))
    group = ExecutionGroup(
        ExecutionGroupSpec(
            "fv-reversed-placement",
            (jax.process_index(),),
            tuple((device.process_index, device.id) for device in devices),
            mesh_axes=(("parts", 2),),
        ),
        devices,
    )
    seeds = jnp.stack((state.at[1::2].set(state[::2]), state.at[::2].set(state[1::2])))
    result = execute_owner_local_finite_volume_advance(
        ((right, right_halo), (left, left_halo)),
        seeds,
        jnp.asarray(0.0),
        jnp.asarray(0.01),
        group,
        axis_name="parts",
    )
    serial, _, _ = _owner_local_fv(1, 0)
    expected = state + 0.01 * serial(jnp.asarray(0.0), state)
    owned = result.state[0].at[1::2].set(result.state[1, 1::2])
    np.testing.assert_allclose(owned, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.inventory_before[0], 1.5, atol=1e-12)
    np.testing.assert_allclose(result.inventory_after[0], 1.5, atol=1e-12)
    assert bool(result.finite)
    assert result.admissibility_checked and bool(result.admissible)
    np.testing.assert_array_equal(result.state[0, 1::2], seeds[0, 1::2])
    np.testing.assert_array_equal(result.state[1, ::2], seeds[1, ::2])


def test_owner_local_fv_uses_original_curved_coordinate_bank_by_default() -> None:
    from phydrax.discretization._cell_geometry import (
        CellGeometrySpec,
        coordinate_lagrange_element,
    )
    from tests.unit.discretization.test_cell_geometry_storage import _storage

    coordinate_element = coordinate_lagrange_element("triangle", 2)
    coefficients = np.asarray(coordinate_element.reference_nodes, dtype=np.float64).copy()
    coefficients[:, 1] += 0.125 * coefficients[:, 0] * coefficients[:, 1]
    root = CellMesh(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64),
        (
            CellBlock(
                "curved",
                "triangle",
                np.asarray(((0, 1, 2),), dtype=np.int32),
                global_ids=np.asarray((37,), dtype=np.int64),
            ),
        ),
    )
    source_geometry = CellGeometrySpec(
        {"curved": coordinate_element},
        {
            "curved": np.arange(coordinate_element.local_dof_count, dtype=np.int32)[
                None, :
            ]
        },
        coefficients,
    )
    storage = _storage(root, source_geometry)
    local_mesh = CellMesh(root.coordinates, root.blocks, storage=storage)
    geometry = phx.discretization.UnstructuredFiniteVolumePlan.from_cell_mesh(
        local_mesh,
        neighborhood_complete=np.ones((1,), dtype=np.bool_),
        physical_boundary_faces=np.ones((3,), dtype=np.bool_),
        closure_evidence_id=storage.evidence_id,
    ).prepare()
    # det(Dx) = 1 + reference_x/8. The actual coordinate bank gives 25/48;
    # using its unchanged sampled corners would incorrectly give 1/2.
    np.testing.assert_allclose(geometry.cell_volumes, [25.0 / 48.0], atol=1.0e-13)
