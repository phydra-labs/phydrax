#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.discretization.lattice_boltzmann._amr import (
    LatticeBoltzmannAMRTransferPlan,
)
from phydrax.discretization.lattice_boltzmann._geometry import (
    LatticeBoltzmannGeometryEpoch,
    LatticeBoltzmannLinkEpoch,
    LatticeBoltzmannPopulationTransferPlan,
)
from phydrax.discretization.lattice_boltzmann._geometry_sensitivity import (
    lattice_boltzmann_geometry_jvp,
    LatticeBoltzmannGeometrySensitivityMargins,
    LatticeBoltzmannGeometrySensitivityPolicy,
)


def _discretization(shape=(8, 8)):
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(n, periodic=True) for n in shape),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    return phx.discretization.LatticeBoltzmannPlan(
        grid, phx.discretization.D2Q9()
    ).prepare()


def test_geometry_epoch_and_population_transfer_are_conservative():
    discretization = _discretization()
    source_snapshot = phx.discretization.LatticeBoltzmannGeometrySnapshot.all_fluid(
        discretization
    )
    source_links = LatticeBoltzmannLinkEpoch(discretization, source_snapshot)
    source = LatticeBoltzmannGeometryEpoch(discretization, source_snapshot, source_links)
    target_mask = np.ones(discretization.grid.shape, dtype=bool)
    target_mask[3, 4] = False
    target_snapshot = phx.discretization.LatticeBoltzmannGeometrySnapshot(
        discretization, target_mask
    )
    target_links = LatticeBoltzmannLinkEpoch(
        discretization, target_snapshot, topology_epoch=1
    )
    target = LatticeBoltzmannGeometryEpoch(discretization, target_snapshot, target_links)
    transfer = LatticeBoltzmannPopulationTransferPlan(source, target)
    populations = jnp.broadcast_to(
        jnp.asarray(discretization.velocity_set.weights),
        discretization.population_shape,
    )
    result = transfer.transfer(populations)

    assert result.evidence.passed
    assert not bool(target.fluid_mask[3, 4])
    assert jnp.all(jnp.isfinite(result.populations))


def test_amr_restriction_and_prolongation_preserve_moments():
    lattice = phx.discretization.D2Q9()
    transfer = LatticeBoltzmannAMRTransferPlan(lattice)
    fine = jnp.broadcast_to(jnp.asarray(lattice.weights), (8, 8, 9))
    coarse, restricted = transfer.restrict(fine)
    prolonged, prolonged_evidence = transfer.prolong(coarse)

    assert coarse.shape == (4, 4, 9)
    assert prolonged.shape == fine.shape
    assert restricted.successful
    assert prolonged_evidence.successful
    np.testing.assert_allclose(prolonged, fine, atol=1e-14)


def test_ratio_two_amr_subcycles_only_active_fine_blocks():
    lattice = phx.discretization.D2Q9()
    transfer = LatticeBoltzmannAMRTransferPlan(lattice)
    plan = phx.discretization.LatticeBoltzmannAMRPlan(transfer)
    coarse = jnp.broadcast_to(lattice.weights, (4, 4, 9))
    fine = jnp.broadcast_to(lattice.weights, (8, 8, 9))
    fine_active = jnp.zeros((8, 8), dtype=bool).at[:4, :4].set(True)
    state = phx.discretization.LatticeBoltzmannAMRState(
        (coarse, fine),
        (jnp.ones((4, 4), dtype=bool), fine_active),
    )
    increment = jnp.asarray(1.0e-4)
    result = plan.advance_two_level(
        state,
        lambda values, amount: values + amount,
        lambda values, amount: values + amount,
        args=increment,
    )

    assert result.successful
    np.testing.assert_allclose(
        result.state.level_populations[0][:2, :2],
        coarse[:2, :2] + 2.0 * increment,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        result.state.level_populations[0][2:, 2:],
        coarse[2:, 2:] + increment,
        atol=1e-14,
    )
    np.testing.assert_array_equal(
        result.state.level_populations[1][~fine_active],
        fine[~fine_active],
    )


def test_fixed_branch_geometry_jvp_has_explicit_validity():
    policy = LatticeBoltzmannGeometrySensitivityPolicy(
        mode=phx.solver.HybridSensitivityMode.SHARP_BRANCHWISE
    )
    margins = LatticeBoltzmannGeometrySensitivityMargins(
        jnp.asarray(1.0),
        jnp.asarray(1.0),
        jnp.asarray(1.0),
        topology_unchanged=jnp.asarray(True),
        event_requested=jnp.asarray(False),
        event_localized=jnp.asarray(False),
        forward_successful=jnp.asarray(True),
    )
    result = lattice_boltzmann_geometry_jvp(
        lambda parameters: parameters**2,
        jnp.asarray(2.0),
        jnp.asarray(1.0),
        margins,
        policy,
    )

    assert result.usable
    np.testing.assert_allclose(result.sensitivity, 4.0)


def test_immersed_direct_forcing_balances_body_load_and_target_velocity():
    discretization = _discretization()
    plan = phx.discretization.ImmersedBoundaryForcingPlan(
        discretization,
        iteration_count=12,
    )
    result = plan.apply(
        jnp.zeros(discretization.grid.shape + (2,)),
        jnp.ones(discretization.grid.shape),
        jnp.asarray(((0.5, 0.5),)),
        jnp.asarray(((0.01, 0.0),)),
        jnp.asarray((float(discretization.cell_size),)),
        jnp.asarray(0.01),
        body_indices=jnp.asarray((0,)),
        body_centers=jnp.asarray(((0.5, 0.5),)),
    )

    assert result.evidence.successful
    assert result.evidence.converged
    np.testing.assert_allclose(result.ledger.force_balance_residual, 0.0, atol=1e-14)
    np.testing.assert_allclose(
        result.evidence.interpolated_velocity,
        result.evidence.target_velocity,
        atol=1e-10,
    )


def test_multiblock_exchange_is_same_step_and_orientation_reciprocal():
    def block(bounds):
        grid = phx.discretization.TensorGridPlan(
            (
                phx.discretization.UniformCellAxisSpec(4),
                phx.discretization.UniformCellAxisSpec(4),
            ),
            axis_names=("x", "y"),
        ).prepare(jnp.asarray(bounds))
        return phx.discretization.LatticeBoltzmannPlan(
            grid, phx.discretization.D2Q9()
        ).prepare()

    left = block(((0.0, 0.0), (1.0, 1.0)))
    right = block(((1.0, 0.0), (2.0, 1.0)))
    interface = phx.discretization.LatticeBoltzmannBlockInterfacePlan(
        left,
        right,
        0,
        0,
        phx.discretization.InterfaceOrientation(1),
    )
    coupling = phx.discretization.LatticeBoltzmannMultiblockCouplingPlan(
        (left, right),
        (phx.discretization.LatticeBoltzmannBlockConnection(0, 1, interface),),
    )
    left_populations = jnp.arange(
        np.prod(left.population_shape), dtype=jnp.float64
    ).reshape(left.population_shape)
    right_populations = 1000.0 + jnp.arange(
        np.prod(right.population_shape), dtype=jnp.float64
    ).reshape(right.population_shape)
    result = coupling.exchange(
        phx.discretization.LatticeBoltzmannMultiblockState(
            (left_populations, right_populations)
        )
    )

    assert result.evidence.successful
    assert result.evidence.interface_count == 1
    assert result.evidence.incoming_write_count > 0
    np.testing.assert_allclose(
        result.evidence.maximum_reciprocity_residual, 0.0, atol=1e-14
    )
    np.testing.assert_array_equal(
        result.state.populations[0][:-1],
        left_populations[:-1],
    )
    np.testing.assert_array_equal(
        result.state.populations[1][1:],
        right_populations[1:],
    )


def test_identity_mapped_lattice_preserves_a_constant_free_stream():
    count = 8
    cell_grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(count),
            phx.discretization.UniformCellAxisSpec(count),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.LatticeBoltzmannPlan(
        cell_grid, phx.discretization.D2Q9()
    ).prepare()
    half_cell = 0.5 / count
    metric_grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(count),
            phx.discretization.UniformAxisSpec(count),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((half_cell, half_cell), (1.0 - half_cell, 1.0 - half_cell))))
    mapped_grid = phx.discretization.MappedTensorGridPlan(
        metric_grid,
        lambda point: point,
        sbp_order=2,
    ).prepare()
    plan = phx.discretization.MappedLatticeBoltzmannPlan(
        discretization,
        mapped_grid,
        lambda time, populations, grid, args: jnp.zeros_like(populations),
        source_id="identity-zero-metric-source",
    )
    populations = jnp.broadcast_to(
        discretization.velocity_set.weights,
        discretization.population_shape,
    )
    result = plan.advance(
        jnp.asarray(0.0),
        populations,
        lambda values, args: values,
    )

    assert result.successful
    np.testing.assert_allclose(result.evidence.free_stream_residual, 0.0, atol=1e-14)
    np.testing.assert_array_equal(result.populations, populations)


def test_moving_sdf_refreshes_links_and_stages_topology_at_accepted_step():
    discretization = _discretization((16, 16))

    def translating_circle(time, coordinates, parameters):
        del parameters
        center = jnp.asarray((0.5 + time, 0.5))
        return jnp.sqrt(jnp.sum((coordinates - center) ** 2, axis=-1)) - 0.2

    plan = phx.discretization.MovingSDFGeometryPlan(
        discretization,
        translating_circle,
        sdf_id="translating-circle",
        body_names=("circle",),
    )
    accepted, _ = plan.initialize(jnp.asarray(0.0))
    numeric = plan.update(accepted, jnp.asarray(1.0e-3), 1)

    assert not numeric.topology_changed
    assert numeric.refresh is not None
    assert numeric.transaction is None
    assert numeric.refresh.evidence.topology_unchanged
    assert numeric.refresh.evidence.refreshed_link_count > 0

    topology = plan.update(numeric.refresh.epoch, jnp.asarray(0.12), 2)
    assert topology.topology_changed
    assert topology.refresh is None
    assert topology.transaction is not None
    populations = jnp.broadcast_to(
        discretization.velocity_set.weights,
        discretization.population_shape,
    )
    rolled_back = topology.transaction.rollback(populations)
    assert not rolled_back.committed
    np.testing.assert_array_equal(rolled_back.populations, populations)
    committed = topology.transaction.commit(populations, 2)
    assert committed.committed
    assert committed.transfer_evidence is not None
    assert committed.transfer_evidence.passed
