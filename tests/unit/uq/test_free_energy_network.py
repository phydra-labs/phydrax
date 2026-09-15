import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _edge(source, destination, value, variance, index):
    return phx.uq.FreeEnergyEdgeObservation(
        value,
        variance,
        source_state_id=source,
        destination_state_id=destination,
        observation_id=f"edge-{index}",
        dataset_id=f"dataset-{index}",
        selection_id=f"selection-{index}",
    )


def test_network_gls_consumes_full_joint_covariance_and_reports_cycle_closure():
    observations = (
        _edge("a", "b", 1.0, 0.04, 0),
        _edge("b", "c", 2.0, 0.09, 1),
        _edge("a", "c", 3.0, 0.16, 2),
    )
    joint = jnp.asarray(
        [
            [0.04, 0.012, 0.008],
            [0.012, 0.09, 0.018],
            [0.008, 0.018, 0.16],
        ]
    )
    plan = phx.uq.FreeEnergyNetworkPlan(("a", "b", "c"), reference_state_id="a")
    result = plan.analyze(observations, joint_covariance=joint)
    np.testing.assert_allclose(result.free_energies, [0.0, 1.0, 3.0], atol=1.0e-12)
    np.testing.assert_allclose(result.joint_edge_covariance, joint)
    assert result.cycle_matrix.shape == (1, 3)
    np.testing.assert_allclose(result.cycle_residuals, 0.0, atol=1.0e-12)
    assert bool(result.successful)


def test_network_cycle_residual_retains_inconsistent_edge_evidence():
    observations = (
        _edge("a", "b", 1.0, 0.04, 0),
        _edge("b", "c", 2.0, 0.04, 1),
        _edge("a", "c", 4.0, 0.04, 2),
    )
    plan = phx.uq.FreeEnergyNetworkPlan(
        ("a", "b", "c"),
        reference_state_id="a",
        maximum_cycle_z_score=2.0,
    )
    result = phx.uq.analyze_free_energy_network(
        plan, observations, joint_covariance=0.04 * jnp.eye(3)
    )
    np.testing.assert_allclose(jnp.abs(result.cycle_residuals), [1.0], atol=1.0e-12)
    np.testing.assert_allclose(
        result.cycle_covariance,
        result.cycle_matrix @ result.joint_edge_covariance @ result.cycle_matrix.T,
    )
    assert int(result.statistical_status) & int(
        phx.uq.FreeEnergyStatus.CYCLE_INCONSISTENT
    )


def test_shared_result_influences_supply_joint_edge_covariance_without_independence_assumption():
    covariance = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.04, 0.01],
            [0.0, 0.01, 0.09],
        ]
    )
    influence = jnp.asarray(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [0.05, jnp.sqrt(0.0875)],
        ]
    )
    source = phx.uq.FreeEnergyResult(
        [0.0, 1.0, 3.0],
        covariance,
        jnp.eye(3),
        jnp.ones((3, 3), dtype=bool),
        [20.0, 20.0, 20.0],
        [20.0, 20.0, 20.0],
        influence,
        1,
        0.0,
        2,
        phx.uq.FreeEnergyStatus.SUCCESS,
        phx.uq.FreeEnergyStatus.SUCCESS,
        state_ids=("a", "b", "c"),
        gauge_state_id="a",
        method="analytic-joint-fixture",
        dataset_id="joint-dataset",
        selection_id="joint-selection",
    )
    observations = (
        phx.uq.FreeEnergyEdgeObservation.from_result(source, "a", "b"),
        phx.uq.FreeEnergyEdgeObservation.from_result(source, "a", "c"),
    )
    plan = phx.uq.FreeEnergyNetworkPlan(("a", "b", "c"), reference_state_id="a")
    result = plan.analyze(observations)
    np.testing.assert_allclose(result.joint_edge_covariance, covariance[1:, 1:])
    np.testing.assert_allclose(result.free_energies, [0.0, 1.0, 3.0])


def test_network_refuses_marginal_variances_when_joint_covariance_is_unknown():
    observations = (
        _edge("a", "b", 1.0, 0.04, 0),
        _edge("b", "c", 2.0, 0.09, 1),
    )
    plan = phx.uq.FreeEnergyNetworkPlan(("a", "b", "c"), reference_state_id="a")
    with pytest.raises(ValueError, match="joint_covariance is required"):
        plan.analyze(observations)
    with pytest.raises(ValueError, match="diagonal"):
        plan.analyze(observations, joint_covariance=jnp.eye(2))
