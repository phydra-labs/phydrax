#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.finite_volume._flux_ledger import (
    FiniteVolumeStageFluxRateLedger,
)
from phydrax.discretization.finite_volume._small_cell import (
    ConservativeSmallCellRedistributionPlan,
)
from phydrax.discretization.finite_volume._unstructured import (
    UnstructuredFiniteVolumePlan,
)
from phydrax.discretization.finite_volume._unstructured_embedded_boundary import (
    EmbeddedBoundaryPlan,
    EmbeddedBoundaryStabilizationPolicy,
)


def _quadrilateral_grid(nx, ny, *, cell_global_ids=None):
    vertices = np.asarray(
        [(float(i), float(j)) for j in range(ny + 1) for i in range(nx + 1)]
    )
    cells = []
    for j in range(ny):
        for i in range(nx):
            lower_left = j * (nx + 1) + i
            lower_right = lower_left + 1
            upper_left = lower_left + nx + 1
            upper_right = upper_left + 1
            cells.append((lower_left, lower_right, upper_right, upper_left))
    return UnstructuredFiniteVolumePlan(
        vertices,
        quadrilaterals=np.asarray(cells),
        cell_global_ids=cell_global_ids,
    ).prepare()


def _metrics(discretization, level_set, field_id, policy):
    return EmbeddedBoundaryPlan(
        discretization,
        level_set,
        field_id=field_id,
        stabilization_policy=policy,
    ).prepare()


def _policy(
    minimum_volume_fraction=0.5,
    maximum_recipients=2,
    *,
    absolute_tolerance=1.0e-14,
    relative_tolerance=1.0e-14,
):
    return EmbeddedBoundaryStabilizationPolicy(
        minimum_volume_fraction=minimum_volume_fraction,
        maximum_recipients=maximum_recipients,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )


def _scatter_rate_block(block, cell_count):
    scattered = jnp.zeros(
        (cell_count,) + block.component_shape, dtype=block.flux_rate.dtype
    )
    scattered = scattered.at[block.owner_cells].add(-block.flux_rate)
    return scattered.at[block.neighbour_cells].add(block.flux_rate)


def _ledger_scatter(plan, block, source_rate):
    return FiniteVolumeStageFluxRateLedger(
        (block,),
        source_rate,
        plan.active_cells,
        geometry_family_id=plan.evidence.geometry_id,
        geometry_layout_id=plan.evidence.prepared_geometry_id,
        geometry_version=jnp.asarray(0, dtype=jnp.int32),
        evidence_policy_id=plan.evidence.policy_id,
        evidence_version=jnp.asarray(0, dtype=jnp.int32),
        topology_epoch_id=plan.evidence.topology_id,
    ).scatter_content_rate()


def _four_recipient_plan():
    discretization = _quadrilateral_grid(3, 3)
    nodal_level_set = jnp.asarray(
        (
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 1.0, 1.0, 100.0),
            (100.0, -10.0, 1.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
        )
    )

    def level_set(points, args):
        del args
        vertex_indices = jnp.rint(points).astype(jnp.int32)
        return nodal_level_set[vertex_indices[:, 1], vertex_indices[:, 0]]

    policy = _policy(minimum_volume_fraction=0.8, maximum_recipients=4)
    metrics = _metrics(
        discretization,
        level_set,
        "four-recipient-interior-sliver",
        policy,
    )
    plan = ConservativeSmallCellRedistributionPlan(discretization, metrics, policy)
    np.testing.assert_array_equal(plan.source_cells, (4,))
    np.testing.assert_array_equal(plan.recipient_mask, ((True, True, True, True),))
    return plan


def test_one_sliver_retains_threshold_scaled_rate_and_conserves_constant_source():
    discretization = _quadrilateral_grid(3, 1)
    policy = _policy(minimum_volume_fraction=0.5, maximum_recipients=1)
    metrics = _metrics(
        discretization,
        lambda points, args: points[:, 0] - 0.8,
        "one-vertical-sliver",
        policy,
    )
    plan = ConservativeSmallCellRedistributionPlan(discretization, metrics, policy)

    np.testing.assert_array_equal(plan.small_cells, (True, False, False))
    np.testing.assert_array_equal(plan.source_cells, (0,))
    np.testing.assert_array_equal(plan.recipient_cells, ((1,),))
    np.testing.assert_allclose(plan.weights, ((1.0,),), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        plan.local_retention_fractions, (0.4, 1.0, 1.0), atol=2.0e-15
    )

    constant_source_rate = 2.0 * metrics.fluid_cell_volumes
    result = plan.redistribute_rate(constant_source_rate)
    np.testing.assert_allclose(result.redistributed_rate, (0.16, 2.24, 2.0))
    np.testing.assert_allclose(
        jnp.sum(result.redistributed_rate), jnp.sum(constant_source_rate)
    )
    np.testing.assert_allclose(result.conservation_defect, 0.0, atol=2.0e-15)
    assert result.activated
    assert result.plan_id == plan.plan_id
    assert result.evidence.metrics_id == metrics.metrics_id
    assert result.evidence.policy_id == policy.policy_id


def test_redistribution_flux_block_scatter_matches_delta_and_keeps_source_separate():
    discretization = _quadrilateral_grid(3, 1)
    policy = _policy(minimum_volume_fraction=0.5, maximum_recipients=1)
    metrics = _metrics(
        discretization,
        lambda points, args: points[:, 0] - 0.8,
        "ledger-visible-sliver",
        policy,
    )
    plan = ConservativeSmallCellRedistributionPlan(discretization, metrics, policy)
    rate = jnp.asarray(((2.0, -4.0), (3.0, 5.0), (-7.0, 1.0)))

    block = plan.redistribution_flux_rate_block(rate)
    assert block is not None
    assert block.block_kind == "small-cell-redistribution"
    assert block.block_id == plan.redistribution_block_id
    np.testing.assert_array_equal(block.owner_cells, (0,))
    np.testing.assert_array_equal(block.neighbour_cells, (1,))
    np.testing.assert_array_equal(block.active_mask, (True,))
    np.testing.assert_allclose(block.flux_rate, ((1.2, -2.4),), atol=2.0e-15)

    physical_source_rate = jnp.asarray(((0.25, -0.5), (0.75, 1.25), (-1.75, 0.25)))
    scattered = _ledger_scatter(plan, block, physical_source_rate)
    block_delta = scattered - physical_source_rate
    expected_delta = plan.redistribute_rate(rate).redistributed_rate - rate
    np.testing.assert_allclose(block_delta, expected_delta, atol=2.0e-15)
    np.testing.assert_allclose(jnp.sum(block_delta, axis=0), 0.0, atol=2.0e-15)

    same_routes = plan.redistribution_flux_rate_block(2.0 * rate)
    assert same_routes is not None
    assert same_routes.block_id == block.block_id
    assert same_routes.route_id == block.route_id
    assert same_routes.rate_block_id == block.rate_block_id


def test_adjacent_slivers_route_only_to_stable_non_small_recipients():
    discretization = _quadrilateral_grid(3, 2)
    policy = _policy(minimum_volume_fraction=0.5, maximum_recipients=2)
    metrics = _metrics(
        discretization,
        lambda points, args: points[:, 0] - 0.8,
        "two-vertical-slivers",
        policy,
    )
    plan = ConservativeSmallCellRedistributionPlan(discretization, metrics, policy)

    np.testing.assert_array_equal(plan.source_cells, (0, 3))
    np.testing.assert_array_equal(plan.recipient_mask, ((True, False), (True, False)))
    recipient_mask = np.asarray(plan.recipient_mask)
    recipients = np.asarray(plan.recipient_cells)[recipient_mask]
    np.testing.assert_array_equal(recipients, (1, 4))
    np.testing.assert_allclose(plan.weights, ((1.0, 0.0), (1.0, 0.0)), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(plan.local_retention_fractions)[recipients],
        np.ones(recipients.size),
        rtol=0.0,
        atol=0.0,
    )
    assert int(plan.report.small_cell_count) == 2
    assert int(plan.report.route_count) == 2

    rate = jnp.arange(1.0, 7.0)
    result = plan.redistribute_rate(rate)
    np.testing.assert_allclose(result.redistributed_rate, (0.4, 2.6, 3.0, 1.6, 7.4, 6.0))
    np.testing.assert_allclose(jnp.sum(result.redistributed_rate), jnp.sum(rate))
    np.testing.assert_allclose(result.conservation_defect, 0.0, atol=2.0e-15)

    block = plan.redistribution_flux_rate_block(rate)
    assert block is not None
    np.testing.assert_array_equal(block.owner_cells, (0, 3))
    np.testing.assert_array_equal(block.neighbour_cells, (1, 4))
    np.testing.assert_array_equal(block.active_mask, (True, True))
    np.testing.assert_allclose(
        _scatter_rate_block(block, discretization.cell_count),
        result.redistributed_rate - rate,
        atol=2.0e-15,
    )


def test_all_sliver_chain_fails_instead_of_routing_excess_between_small_cells():
    discretization = _quadrilateral_grid(3, 1)
    policy = _policy(minimum_volume_fraction=0.5, maximum_recipients=2)
    metrics = _metrics(
        discretization,
        lambda points, args: points[:, 1] - 0.8,
        "horizontal-sliver-chain",
        policy,
    )
    np.testing.assert_array_equal(metrics.active_fluid_cells, (True, True, True))
    np.testing.assert_allclose(metrics.volume_fraction, (0.2, 0.2, 0.2))

    with pytest.raises(ValueError, match="no non-small open-face recipient"):
        ConservativeSmallCellRedistributionPlan(discretization, metrics, policy)


def test_equal_measure_ties_are_broken_by_stable_cell_id():
    stable_ids = np.asarray((40, 30, 10, 20), dtype=np.int64)
    discretization = _quadrilateral_grid(2, 2, cell_global_ids=stable_ids)
    policy = _policy(minimum_volume_fraction=0.1, maximum_recipients=1)
    metrics = _metrics(
        discretization,
        lambda points, args: points[:, 0] + points[:, 1] - 1.8,
        "symmetric-corner-sliver",
        policy,
    )
    plan = ConservativeSmallCellRedistributionPlan(discretization, metrics, policy)

    np.testing.assert_array_equal(plan.source_cells, (0,))
    np.testing.assert_array_equal(plan.recipient_cells, ((2,),))
    assert int(discretization.cell_global_ids[plan.recipient_cells[0, 0]]) == 10

    reversed_ids = np.asarray((40, 10, 30, 20), dtype=np.int64)
    reversed_geometry = _quadrilateral_grid(2, 2, cell_global_ids=reversed_ids)
    reversed_metrics = _metrics(
        reversed_geometry,
        lambda points, args: points[:, 0] + points[:, 1] - 1.8,
        "symmetric-corner-sliver",
        policy,
    )
    reversed_plan = ConservativeSmallCellRedistributionPlan(
        reversed_geometry, reversed_metrics, policy
    )
    np.testing.assert_array_equal(reversed_plan.recipient_cells, ((1,),))


def test_vector_rates_are_componentwise_conservative_under_jit_and_grad():
    discretization = _quadrilateral_grid(3, 1)
    policy = _policy(minimum_volume_fraction=0.5, maximum_recipients=1)
    metrics = _metrics(
        discretization,
        lambda points, args: points[:, 0] - 0.8,
        "differentiable-sliver",
        policy,
    )
    plan = ConservativeSmallCellRedistributionPlan(discretization, metrics, policy)
    rate = jnp.asarray(((1.0, -2.0), (3.0, 4.0), (-5.0, 6.0)))
    redistribute = eqx.filter_jit(plan.redistribute_rate)

    result = redistribute(rate)
    np.testing.assert_allclose(
        jnp.sum(result.redistributed_rate, axis=0),
        jnp.sum(rate, axis=0),
        atol=2.0e-15,
    )
    np.testing.assert_allclose(result.conservation_defect, jnp.zeros((2,)), atol=2.0e-15)
    gradient = jax.grad(lambda value: jnp.sum(redistribute(value).redistributed_rate))(
        rate
    )
    np.testing.assert_allclose(gradient, jnp.ones_like(rate), atol=2.0e-15)


def test_float32_extreme_weights_renormalize_under_jit_and_conserve_gradient():
    plan = _four_recipient_plan()
    prepared_weights = np.asarray(
        ((0.31141971, 0.48647018, 0.20211012, 1.0e-30),),
        dtype=np.float64,
    )
    prepared_weights /= np.sum(prepared_weights, axis=1, keepdims=True)
    assert np.sum(prepared_weights.astype(np.float32), dtype=np.float32) != np.float32(
        1.0
    )
    plan = eqx.tree_at(
        lambda candidate: candidate.weights,
        plan,
        jnp.asarray(prepared_weights),
    )
    rate = jnp.linspace(-4.0, 5.0, 18, dtype=jnp.float32).reshape((9, 2))
    redistribute = eqx.filter_jit(plan.redistribute_rate)

    result = redistribute(rate)
    assert result.redistributed_rate.dtype == jnp.float32
    component_scale = jnp.sum(jnp.abs(rate), axis=0)
    tolerance = 32.0 * jnp.finfo(jnp.float32).eps * component_scale
    np.testing.assert_array_less(
        jnp.abs(result.conservation_defect),
        tolerance,
    )
    assert jnp.allclose(
        jnp.sum(result.redistributed_rate, axis=0),
        jnp.sum(rate, axis=0),
        rtol=32.0 * jnp.finfo(jnp.float32).eps,
        atol=0.0,
    )
    gradient = jax.grad(lambda value: jnp.sum(redistribute(value).redistributed_rate))(
        rate
    )
    np.testing.assert_allclose(
        gradient,
        jnp.ones_like(rate),
        rtol=0.0,
        atol=float(8.0 * jnp.finfo(jnp.float32).eps),
    )


def test_float32_flux_block_uses_normalized_route_weights_under_jit_and_grad():
    plan = _four_recipient_plan()
    prepared_weights = np.asarray(
        ((0.31141971, 0.48647018, 0.20211012, 1.0e-30),),
        dtype=np.float64,
    )
    prepared_weights /= np.sum(prepared_weights, axis=1, keepdims=True)
    plan = eqx.tree_at(
        lambda candidate: candidate.weights,
        plan,
        jnp.asarray(prepared_weights),
    )
    rate = jnp.linspace(-4.0, 5.0, 18, dtype=jnp.float32).reshape((9, 2))

    build_block = eqx.filter_jit(plan.redistribution_flux_rate_block)
    block = build_block(rate)
    assert block is not None
    assert block.flux_rate.dtype == jnp.float32
    source = int(plan.source_cells[0])
    retention = plan.local_retention_fractions[source].astype(jnp.float32)
    excess = (1.0 - retention) * rate[source]
    route_weights = block.flux_rate / excess[None, :]
    expected_weights = prepared_weights[0].astype(np.float32)
    expected_weights /= np.sum(expected_weights, dtype=np.float32)
    closure_route = int(np.argmax(expected_weights))
    expected_weights[closure_route] = np.float32(1.0) - np.sum(
        np.delete(expected_weights, closure_route), dtype=np.float32
    )
    np.testing.assert_allclose(
        route_weights,
        np.broadcast_to(expected_weights[:, None], route_weights.shape),
        rtol=float(8.0 * jnp.finfo(jnp.float32).eps),
        atol=0.0,
    )

    def block_delta(value):
        built = plan.redistribution_flux_rate_block(value)
        assert built is not None
        return _scatter_rate_block(built, plan.active_cells.size)

    jitted_delta = eqx.filter_jit(block_delta)
    delta = jitted_delta(rate)
    direct_delta = plan.redistribute_rate(rate).redistributed_rate - rate
    tolerance = (
        32.0 * jnp.finfo(jnp.float32).eps * jnp.maximum(jnp.max(jnp.abs(rate)), 1.0)
    )
    np.testing.assert_allclose(delta, direct_delta, rtol=0.0, atol=float(tolerance))
    np.testing.assert_allclose(
        jnp.sum(delta, axis=0), 0.0, rtol=0.0, atol=float(tolerance)
    )

    dual = jnp.linspace(-2.0, 3.0, rate.size, dtype=jnp.float32).reshape(rate.shape)
    block_gradient = jax.grad(lambda value: jnp.sum(jitted_delta(value) * dual))(rate)
    direct_gradient = jax.grad(
        lambda value: jnp.sum(
            (plan.redistribute_rate(value).redistributed_rate - value) * dual
        )
    )(rate)
    np.testing.assert_allclose(
        block_gradient,
        direct_gradient,
        rtol=0.0,
        atol=float(8.0 * jnp.finfo(jnp.float32).eps),
    )


def test_float32_recipient_weight_underflow_fails_explicitly_under_jit():
    plan = _four_recipient_plan()
    prepared_weights = jnp.asarray(
        ((1.0e-50, 0.25, 0.25, 0.5 - 1.0e-50),),
        dtype=jnp.float64,
    )
    plan = eqx.tree_at(
        lambda candidate: candidate.weights,
        plan,
        prepared_weights,
    )
    redistribute = eqx.filter_jit(plan.redistribute_rate)

    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="recipient weights underflow",
    ):
        result = redistribute(jnp.ones((9,), dtype=jnp.float32))
        jax.block_until_ready(result.redistributed_rate)


def test_float32_nonfinite_conservation_defect_fails_explicitly():
    plan = _four_recipient_plan()
    source = int(plan.source_cells[0])
    recipient = int(plan.recipient_cells[0, 0])
    balancer = next(
        cell for cell in range(plan.active_cells.size) if cell not in (source, recipient)
    )
    maximum = jnp.finfo(jnp.float32).max
    rate = jnp.zeros((plan.active_cells.size,), dtype=jnp.float32)
    rate = rate.at[source].set(maximum)
    rate = rate.at[recipient].set(maximum)
    rate = rate.at[balancer].set(-maximum)

    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="content-rate dtype conservation tolerance",
    ):
        result = eqx.filter_jit(plan.redistribute_rate)(rate)
        jax.block_until_ready(result.redistributed_rate)


def test_inactive_rates_are_zero_and_nonzero_inactive_content_fails():
    discretization = _quadrilateral_grid(3, 1)
    policy = _policy(minimum_volume_fraction=0.5, maximum_recipients=1)
    metrics = _metrics(
        discretization,
        lambda points, args: points[:, 0] - 1.1,
        "one-solid-and-no-small-cells",
        policy,
    )
    plan = ConservativeSmallCellRedistributionPlan(discretization, metrics, policy)

    unchanged = plan.redistribute_rate(jnp.asarray((0.0, 2.0, 3.0)))
    np.testing.assert_array_equal(unchanged.redistributed_rate, (0.0, 2.0, 3.0))
    assert not unchanged.activated
    assert plan.redistribution_flux_rate_block(jnp.asarray((0.0, 2.0, 3.0))) is None
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="Inactive-cell content rates must be exactly zero",
    ):
        result = plan.redistribute_rate(jnp.asarray((1.0, 2.0, 3.0)))
        jax.block_until_ready(result.redistributed_rate)


def test_sliver_with_only_inactive_or_closed_neighbours_fails_preparation():
    discretization = _quadrilateral_grid(2, 1)
    policy = _policy(minimum_volume_fraction=0.5, maximum_recipients=2)
    metrics = _metrics(
        discretization,
        lambda points, args: 0.2 - points[:, 0],
        "isolated-boundary-sliver",
        policy,
    )
    np.testing.assert_array_equal(metrics.active_fluid_cells, (True, False))

    with pytest.raises(ValueError, match="no non-small open-face recipient"):
        ConservativeSmallCellRedistributionPlan(discretization, metrics, policy)


def test_policy_identity_binds_every_stabilization_choice():
    discretization = _quadrilateral_grid(3, 1)
    baseline = _policy(
        minimum_volume_fraction=0.5,
        maximum_recipients=2,
        absolute_tolerance=1.0e-14,
        relative_tolerance=2.0e-14,
    )
    changed_threshold = _policy(
        minimum_volume_fraction=0.6,
        maximum_recipients=2,
        absolute_tolerance=1.0e-14,
        relative_tolerance=2.0e-14,
    )
    changed_routes = _policy(
        minimum_volume_fraction=0.5,
        maximum_recipients=1,
        absolute_tolerance=1.0e-14,
        relative_tolerance=2.0e-14,
    )
    changed_tolerance = _policy(
        minimum_volume_fraction=0.5,
        maximum_recipients=2,
        absolute_tolerance=2.0e-14,
        relative_tolerance=2.0e-14,
    )
    policies = (baseline, changed_threshold, changed_routes, changed_tolerance)
    metrics = tuple(
        _metrics(
            discretization,
            lambda points, args: points[:, 0] - 0.8,
            "identity-sliver",
            policy,
        )
        for policy in policies
    )
    plans = tuple(
        ConservativeSmallCellRedistributionPlan(discretization, metric, policy)
        for metric, policy in zip(metrics, policies, strict=True)
    )

    assert len({policy.policy_id for policy in policies}) == len(policies)
    assert len({plan.plan_id for plan in plans}) == len(plans)
    blocks = tuple(
        plan.redistribution_flux_rate_block(
            jnp.where(
                plan.active_cells,
                jnp.ones_like(plan.active_cells, dtype=float),
                0.0,
            )
        )
        for plan in plans
    )
    assert all(block is not None for block in blocks)
    assert len({block.block_id for block in blocks if block is not None}) == len(blocks)
    assert all(
        block.block_id == plan.redistribution_block_id
        for plan, block in zip(plans, blocks, strict=True)
        if block is not None
    )
    for policy, metrics_, plan in zip(policies, metrics, plans, strict=True):
        assert plan.report.policy_id == policy.policy_id
        assert plan.report.prepared_geometry_id == discretization.prepared_id
        assert plan.report.geometry_id == discretization.geometry_id
        assert plan.report.metrics_id == metrics_.metrics_id
        assert plan.evidence.policy_id == policy.policy_id
    with pytest.raises(ValueError, match="bind the stabilization policy"):
        ConservativeSmallCellRedistributionPlan(
            discretization, metrics[0], changed_threshold
        )
