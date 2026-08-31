#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.discretization import CellBlock, CellMesh
from phydrax.discretization.fem import (
    balanced_hp_refinement_ids,
    certify_finite_element_hp_geometry,
    close_finite_element_hp_decision,
    coarsen_tensor_hp_cells,
    finite_element_hp_constraint,
    finite_element_hp_decision,
    finite_element_hp_domains,
    finite_element_hp_interface_plan,
    finite_element_hp_trace_constraint_plan,
    finite_element_hp_transfer_plan,
    FiniteElementHPEpoch,
    FiniteElementHPErrorEstimate,
    FiniteElementHPResidualJumpLedger,
    FiniteElementHPStateTransferPolicy,
    FiniteElementHPTransaction,
    hp_active_cell_mesh,
    initial_finite_element_hp_topology,
    prepare_finite_element_hp_epoch,
    refine_tensor_hp_cells,
    tensor_modal_decay_estimate,
    tensor_trace_interpolation,
)


def _quad_mesh():
    coordinates = jnp.asarray(
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (2.0, 1.0),
        )
    )
    return CellMesh(
        coordinates,
        (
            CellBlock(
                "quads",
                "quadrilateral",
                jnp.asarray(((0, 1, 4, 3), (1, 2, 5, 4)), dtype=jnp.int32),
                global_ids=jnp.asarray((10, 20), dtype=jnp.int64),
            ),
        ),
    )


def _hex_mesh():
    coordinates = jnp.asarray(
        tuple(
            (float(x), float(y), float(z))
            for z in range(2)
            for y in range(2)
            for x in range(2)
        )
    )
    return CellMesh(
        coordinates,
        (
            CellBlock(
                "hexes",
                "hexahedron",
                jnp.asarray(((0, 1, 3, 2, 4, 5, 7, 6),), dtype=jnp.int32),
                global_ids=jnp.asarray((30,), dtype=jnp.int64),
            ),
        ),
    )


def test_quad_refinement_builds_stable_forest_mortars_and_coarsens():
    topology, geometry = initial_finite_element_hp_topology(_quad_mesh(), (2, 3), 16)
    initial_interfaces = finite_element_hp_interface_plan(topology, geometry)
    balanced_ids, closure = balanced_hp_refinement_ids(
        topology,
        initial_interfaces,
        jnp.asarray((10,), dtype=jnp.int64),
    )
    assert tuple(np.asarray(balanced_ids)) == (10,)
    assert np.asarray(closure).size == 0

    refined = refine_tensor_hp_cells(topology, geometry, balanced_ids)
    assert refined.topology.active_count == 5
    assert refined.topology.allocated_count == 6
    np.testing.assert_array_equal(
        np.asarray(refined.topology.path_codes)[2:6],
        np.asarray((5, 6, 7, 8)),
    )
    interfaces = finite_element_hp_interface_plan(refined.topology, refined.geometry)
    assert np.count_nonzero(np.asarray(interfaces.relation_mask("mortar"))) == 2
    evidence = certify_finite_element_hp_geometry(
        refined.topology,
        refined.geometry,
        interfaces,
    )
    assert evidence.passed

    active_mesh, degrees, routes = hp_active_cell_mesh(
        refined.topology,
        refined.geometry,
    )
    assert sum(block.cell_count for block in active_mesh.blocks) == 5
    assert degrees == ((2, 3),)
    assert routes.shape == (5,)

    coarsened = coarsen_tensor_hp_cells(
        refined.topology,
        refined.geometry,
        jnp.asarray((10,), dtype=jnp.int64),
    )
    assert coarsened.topology.active_count == 2
    np.testing.assert_array_equal(
        np.asarray(coarsened.topology.cell_degrees)[:2],
        np.asarray(((2, 3), (2, 3))),
    )


def test_hex_refinement_allocates_eight_curved_children():
    topology, geometry = initial_finite_element_hp_topology(_hex_mesh(), 2, 12)

    def curved(_slot, points):
        mapped = points.copy()
        mapped[:, 2] += 0.1 * points[:, 0] * (1.0 - points[:, 0]) * points[:, 1]
        return mapped

    refined = refine_tensor_hp_cells(
        topology,
        geometry,
        jnp.asarray((30,), dtype=jnp.int64),
        coordinate_evaluator=curved,
    )
    assert refined.topology.active_count == 8
    assert refined.topology.child_capacity == 8
    assert np.all(np.asarray(refined.topology.levels)[1:9] == 1)
    assert np.all(np.isfinite(np.asarray(refined.geometry.cell_vertices)[1:9]))


def test_modal_decay_and_hp_decision_separate_p_from_h():
    nodes = np.linspace(-1.0, 1.0, 4)
    x, y = np.meshgrid(nodes, nodes, indexing="ij")
    decay = tensor_modal_decay_estimate(
        x**2 + y**2,
        (3, 3),
        nodes_by_axis=(nodes, nodes),
    )
    np.testing.assert_array_less(np.asarray(decay), 1.0e-20)

    topology, _ = initial_finite_element_hp_topology(_quad_mesh(), 2, 8)
    estimate = FiniteElementHPErrorEstimate(
        topology,
        jnp.asarray((4.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        smoothness=jnp.asarray(
            (
                (1.0e-8, 1.0e-3),
                (1.0, 1.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
            )
        ),
    )
    decision = finite_element_hp_decision(topology, estimate, maximum_degree=5)
    assert tuple(np.asarray(decision.target_degrees)[0]) == (3, 2)
    assert not bool(np.asarray(decision.refine)[0])


def test_hp_epoch_requires_matching_prepared_components():
    topology, geometry = initial_finite_element_hp_topology(_quad_mesh(), 2, 8)
    interfaces = finite_element_hp_interface_plan(topology, geometry)
    epoch = FiniteElementHPEpoch(_quad_mesh(), topology, geometry, interfaces)

    assert epoch.topology.topology_id == topology.topology_id
    assert epoch.worksets.topology_id == topology.topology_id


def test_prepared_hp_epoch_uses_bucket_elements_and_overlay_domains():
    topology, geometry = initial_finite_element_hp_topology(_quad_mesh(), 2, 16)
    refined = refine_tensor_hp_cells(
        topology,
        geometry,
        jnp.asarray((10,), dtype=jnp.int64),
        target_degrees=jnp.asarray(((3, 2),), dtype=jnp.int32),
    )
    epoch = prepare_finite_element_hp_epoch(
        refined.topology,
        refined.geometry,
        "u",
        conformity="L2",
    )
    interior, exterior = finite_element_hp_domains(epoch)

    assert len(epoch.mesh.blocks) == 2
    assert epoch.discretization is not None
    assert interior.kind == "interior_facet"
    assert exterior.kind == "exterior_facet"
    assert np.count_nonzero(np.asarray(epoch.interfaces.relation_mask("mortar"))) == 2


def test_p_and_h_trace_constraints_preserve_polynomials_and_raw_duals():
    master_nodes = jnp.asarray(((0.0,), (0.5,), (1.0,)))
    p_slave_nodes = jnp.linspace(0.0, 1.0, 5)[:, None]
    interpolation = tensor_trace_interpolation(master_nodes, p_slave_nodes)
    plan = finite_element_hp_trace_constraint_plan(
        8,
        jnp.arange(3, 8, dtype=jnp.int32),
        jnp.broadcast_to(jnp.arange(3, dtype=jnp.int32), (5, 3)),
        interpolation,
    )
    master_values = master_nodes[:, 0] ** 2
    full_values = plan.expand(master_values)

    np.testing.assert_allclose(np.asarray(full_values[:3]), np.asarray(master_values))
    np.testing.assert_allclose(
        np.asarray(full_values[3:]),
        np.asarray(p_slave_nodes[:, 0] ** 2),
        atol=2.0e-14,
    )
    full_dual = jnp.linspace(0.2, 1.6, 8)
    reduced = jnp.linspace(-0.5, 0.75, 3)
    np.testing.assert_allclose(
        jnp.vdot(plan.expand(reduced), full_dual),
        jnp.vdot(reduced, plan.pullback_raw(full_dual)),
        atol=2.0e-14,
    )

    child_points = jnp.asarray(((0.0,), (0.25,), (0.5,)))
    child_interpolation = tensor_trace_interpolation(master_nodes, child_points)
    np.testing.assert_allclose(
        np.asarray(child_interpolation @ master_values),
        np.asarray(child_points[:, 0] ** 2),
        atol=2.0e-14,
    )


def test_prepared_h1_epoch_builds_master_trace_constraint_and_uniform_limit():
    topology, geometry = initial_finite_element_hp_topology(_quad_mesh(), 2, 16)
    refined = refine_tensor_hp_cells(
        topology,
        geometry,
        jnp.asarray((10,), dtype=jnp.int64),
        target_degrees=jnp.asarray(((3, 4),), dtype=jnp.int32),
    )
    epoch = prepare_finite_element_hp_epoch(refined.topology, refined.geometry, "u")
    trace_plan = dict(epoch.constraints)["u"]
    assert trace_plan.reduced_dof_count < trace_plan.full_dof_count
    np.testing.assert_allclose(
        np.asarray(trace_plan.expand(jnp.ones((trace_plan.reduced_dof_count,)))),
        1.0,
        atol=2.0e-13,
    )
    constraint = finite_element_hp_constraint(epoch.discretization, "u", trace_plan)
    assert constraint.full_space.size == trace_plan.full_dof_count
    assert constraint.reduced_space.size == trace_plan.reduced_dof_count

    uniform_epoch = prepare_finite_element_hp_epoch(topology, geometry, "u")
    uniform_plan = dict(uniform_epoch.constraints)["u"]
    assert uniform_plan.full_dof_count == uniform_plan.reduced_dof_count
    np.testing.assert_allclose(np.asarray(uniform_plan.prolongation), np.eye(15))


def test_h_transfer_roles_and_epoch_transaction_are_distinct_and_conservative(
    tmp_path,
):
    topology, geometry = initial_finite_element_hp_topology(_quad_mesh(), 2, 16)
    source = prepare_finite_element_hp_epoch(
        topology,
        geometry,
        "u",
        conformity="L2",
    )
    refined = refine_tensor_hp_cells(
        topology,
        geometry,
        jnp.asarray((10,), dtype=jnp.int64),
        target_degrees=jnp.asarray(((3, 2),), dtype=jnp.int32),
    )
    target = prepare_finite_element_hp_epoch(
        refined.topology,
        refined.geometry,
        "u",
        conformity="L2",
    )
    transfer = finite_element_hp_transfer_plan(
        source,
        target,
        refined.lineage,
        "u",
        "h-refinement",
    )
    source_values = jnp.zeros((topology.capacity, transfer.primal.shape[2]))
    source_values = source_values.at[0, :9].set(1.0)
    target_values = transfer.apply_primal(source_values)
    for slot, count in zip(
        np.asarray(transfer.target_slots),
        np.asarray(transfer.target_dof_count),
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(target_values[slot, :count]),
            1.0,
            atol=2.0e-13,
        )
    target_dual = jnp.arange(target_values.size, dtype=float).reshape(target_values.shape)
    np.testing.assert_allclose(
        jnp.vdot(target_values, target_dual),
        jnp.vdot(source_values, transfer.pullback_raw(target_dual)),
        atol=2.0e-12,
    )

    transaction = FiniteElementHPTransaction(
        source,
        target,
        refined.lineage,
        h_transfers=(transfer,),
        diagnostics=("geometry-certified",),
    )
    assert transaction.rollback().epoch_id == source.epoch_id
    assert transaction.promote(True).epoch_id == target.epoch_id
    assert transaction.promote(False).epoch_id == source.epoch_id
    primal_policy = FiniteElementHPStateTransferPolicy("u", "primal")
    np.testing.assert_allclose(
        np.asarray(primal_policy.apply(transfer, source_values)),
        np.asarray(target_values),
    )
    accepted_state = phx.solver.FiniteElementAcceptedState(
        (source_values,),
        0.0,
        0,
        source.topology.topology_id,
        source.epoch_id,
        "compiled-source",
    )
    executor = phx.solver.FiniteElementTopologyTransaction(
        lambda epoch, fields, materials, candidate, args: (
            epoch.epoch_id == candidate.candidate.epoch_id
            and len(fields) == 1
            and materials is None
        )
    )
    result = executor.execute_hp(accepted_state, transaction)
    assert bool(result.committed)
    assert result.state.topology_id == target.topology.topology_id
    np.testing.assert_allclose(np.asarray(result.state.fields[0]), target_values)

    coarsened = coarsen_tensor_hp_cells(
        refined.topology,
        refined.geometry,
        jnp.asarray((10,), dtype=jnp.int64),
    )
    coarsened_epoch = prepare_finite_element_hp_epoch(
        coarsened.topology,
        coarsened.geometry,
        "u",
        conformity="L2",
    )
    coarsening_transfer = finite_element_hp_transfer_plan(
        target,
        coarsened_epoch,
        coarsened.lineage,
        "u",
        "h-coarsening",
    )
    reconstructed = coarsening_transfer.apply_mass_projection(target_values)
    parent_count = int(np.asarray(coarsening_transfer.target_dof_count)[0])
    parent_slot = int(np.asarray(coarsening_transfer.target_slots)[0])
    np.testing.assert_allclose(
        np.asarray(reconstructed[parent_slot, :parent_count]),
        1.0,
        atol=2.0e-12,
    )

    archive = tmp_path / "hp-epoch.npz"
    phx.solver.write_finite_element_hp_epoch(archive, target)
    restored = phx.solver.read_finite_element_hp_epoch(archive)
    assert restored.topology.plan_id == target.topology.plan_id
    assert restored.geometry.geometry_id == target.geometry.geometry_id


def test_residual_jump_ledger_budgets_hysteresis_and_balance_are_deterministic():
    topology, geometry = initial_finite_element_hp_topology(_quad_mesh(), 2, 16)
    interfaces = finite_element_hp_interface_plan(topology, geometry)
    ledger = FiniteElementHPResidualJumpLedger(
        topology,
        interfaces,
        jnp.asarray((2.0, 0.25) + (0.0,) * 14),
        jnp.asarray((1.0, 1.0) + (0.0,) * 14),
        jnp.ones((interfaces.capacity,)),
        jnp.ones((interfaces.capacity,)),
    )
    assert ledger.estimate.global_estimate > 0.0
    rough = FiniteElementHPErrorEstimate(
        topology,
        ledger.estimate.cell_indicators,
        smoothness=jnp.ones((topology.capacity, topology.dimension)),
    )
    limited = finite_element_hp_decision(
        topology,
        rough,
        maximum_active_cells=2,
    )
    assert not np.any(np.asarray(limited.refine))

    unlimited = finite_element_hp_decision(
        topology,
        rough,
        maximum_active_cells=8,
    )
    closed = close_finite_element_hp_decision(topology, interfaces, unlimited)
    np.testing.assert_array_equal(
        np.asarray(closed.requested_refine),
        np.asarray(unlimited.refine),
    )

    low = FiniteElementHPErrorEstimate(
        topology,
        jnp.asarray((1.0, 0.001) + (0.0,) * 14),
        smoothness=jnp.zeros((topology.capacity, topology.dimension)),
    )
    first = finite_element_hp_decision(
        topology,
        low,
        coarsen_epochs=2,
    )
    second = finite_element_hp_decision(
        topology,
        low,
        coarsen_history=first.coarsen_history,
        coarsen_epochs=2,
    )
    assert int(np.asarray(first.coarsen_history)[1]) == 1
    assert int(np.asarray(second.coarsen_history)[1]) == 2
