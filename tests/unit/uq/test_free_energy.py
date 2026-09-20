import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _lineage(capacity, *, active=None):
    active_ = (
        jnp.ones((capacity,), dtype="bool") if active is None else jnp.asarray(active)
    )
    return {
        "sample_active": active_,
        "chain_index": jnp.where(active_, 0, -1),
        "draw_index": jnp.where(active_, jnp.arange(capacity), -1),
        "repeat_index": jnp.where(active_, 0, -1),
        "dependence_group_index": jnp.where(active_, 0, -1),
    }


def _work_dataset(values, sources):
    values = jnp.asarray(values, dtype="float64")
    source = jnp.asarray(sources, dtype=jnp.int32)
    lineage = _lineage(values.size)
    return phx.uq.ReducedWorkDataset(
        values,
        lineage["sample_active"],
        lineage["sample_active"],
        source,
        1 - source,
        lineage["chain_index"],
        lineage["draw_index"],
        lineage["repeat_index"],
        lineage["dependence_group_index"],
        potential_ids=("potential-a", "potential-b"),
        state_ids=("state-a", "state-b"),
        measure_ids=("common-measure", "common-measure"),
        producer_id="analytic-work-producer",
        run_id="analytic-run",
        work_id="directed-reduced-work",
        work_kind="equilibrium-difference",
        qualification_id="exact-work-sampling",
        sampling_exact=True,
        sampling_bias_bound=0.0,
        unit_id="1",
    )


def _potential_dataset(values, origins, *, active=None, coverage=None):
    values = jnp.asarray(values, dtype="float64")
    lineage = _lineage(values.shape[1], active=active)
    covered = (
        jnp.broadcast_to(lineage["sample_active"][None, :], values.shape)
        if coverage is None
        else jnp.asarray(coverage)
    )
    state_ids = tuple(f"state-{index}" for index in range(values.shape[0]))
    return phx.uq.ReducedPotentialDataset(
        values,
        covered,
        lineage["sample_active"],
        origins,
        lineage["chain_index"],
        lineage["draw_index"],
        lineage["repeat_index"],
        lineage["dependence_group_index"],
        state_ids=state_ids,
        potential_ids=tuple(f"potential-{index}" for index in range(values.shape[0])),
        inverse_temperatures=jnp.ones((values.shape[0],)),
        reduced_convention_id="analytic-test-reduced-potential",
        qualification_id="analytic-test-sampling",
        sampling_exact=True,
        sampling_bias_bound=0.0,
        measure_id="common-measure",
        producer_id="analytic-potential-producer",
        run_id="analytic-run",
        unit_id="1",
    )


def test_reduced_potential_dataset_derives_counts_and_canonicalizes_padding():
    values = jnp.asarray([[0.0, 1.0, jnp.nan], [1.0, 2.0, jnp.nan]])
    dataset = _potential_dataset(
        values,
        [0, 1, -7],
        active=[True, True, False],
        coverage=jnp.ones_like(values, dtype="bool"),
    )
    np.testing.assert_array_equal(dataset.state_counts, [1, 1])
    np.testing.assert_array_equal(dataset.coverage[:, -1], [False, False])
    np.testing.assert_allclose(dataset.values[:, -1], 0.0)
    assert int(dataset.origin_state[-1]) == -1

    with pytest.raises(ValueError, match="Dense MBAR coverage"):
        _potential_dataset(
            jnp.zeros((2, 2)),
            [0, 1],
            coverage=[[True, False], [True, True]],
        )
    with pytest.raises(ValueError, match="unit_id"):
        phx.uq.ReducedPotentialDataset(
            jnp.zeros((2, 2)),
            jnp.ones((2, 2), dtype="bool"),
            jnp.ones((2,), dtype="bool"),
            [0, 1],
            [0, 0],
            [0, 1],
            [0, 0],
            [0, 0],
            state_ids=("a", "b"),
            potential_ids=("u-a", "u-b"),
            inverse_temperatures=jnp.ones((2,)),
            reduced_convention_id="analytic-test-reduced-potential",
            qualification_id="analytic-test-sampling",
            sampling_exact=True,
            sampling_bias_bound=0.0,
            measure_id="measure",
            producer_id="producer",
            run_id="run",
            unit_id="kJ/mol",
        )


def test_fep_and_bar_use_directed_work_and_report_block_covariance():
    delta = 1.7
    dataset = _work_dataset(
        [delta] * 8 + [-delta] * 8,
        [0] * 8 + [1] * 8,
    )
    fep = phx.uq.free_energy_perturbation(dataset)
    bar = phx.uq.bennett_acceptance_ratio(dataset)
    np.testing.assert_allclose(fep.free_energies, [0.0, delta], atol=1.0e-12)
    np.testing.assert_allclose(bar.free_energies, [0.0, delta], atol=1.0e-12)
    assert bool(fep.successful)
    assert bool(bar.successful)
    np.testing.assert_allclose(
        fep.covariance, fep.influence_values @ fep.influence_values.T
    )
    np.testing.assert_allclose(
        bar.covariance, bar.influence_values @ bar.influence_values.T
    )

    correlated = _work_dataset(
        [0.0] * 4 + [2.0] * 4 + [0.0] * 4 + [2.0] * 4,
        [0] * 16,
    )
    singleton = phx.uq.free_energy_perturbation(
        correlated,
        phx.uq.FreeEnergySelectionPlan(block_length=1),
    )
    blocked = phx.uq.free_energy_perturbation(
        correlated,
        phx.uq.FreeEnergySelectionPlan(block_length=4),
    )
    assert blocked.standard_errors[0, 1] > singleton.standard_errors[0, 1]


def test_block_bootstrap_is_explicitly_keyed_and_replayable():
    dataset = _work_dataset(
        [0.0, 0.1, -0.1, 0.2, 0.0, -0.2, 0.1, -0.1],
        [0] * 8,
    )
    plan = phx.uq.FreeEnergySelectionPlan(
        block_length=2,
        uncertainty_method="block-bootstrap",
        bootstrap_replicates=32,
    )
    with pytest.raises(ValueError, match="explicit JAX random key"):
        phx.uq.free_energy_perturbation(dataset, plan)
    first = phx.uq.free_energy_perturbation(dataset, plan, key=jax.random.key(13))
    replay = phx.uq.free_energy_perturbation(dataset, plan, key=jax.random.key(13))
    np.testing.assert_array_equal(first.covariance, replay.covariance)


def test_ti_consumes_raw_complete_path_derivatives_and_joint_blocks():
    state_count, capacity = 3, 8
    active = jnp.ones((state_count, capacity), dtype="bool")
    draw = jnp.broadcast_to(jnp.arange(capacity), active.shape)
    dataset = phx.uq.ThermodynamicDerivativeDataset(
        jnp.full((state_count, capacity), 2.0),
        active,
        active,
        jnp.zeros_like(draw),
        draw,
        jnp.zeros_like(draw),
        jnp.zeros_like(draw),
        [0.0, 0.5, 1.0],
        state_ids=("lambda-0", "lambda-half", "lambda-1"),
        potential_ids=("u-0", "u-half", "u-1"),
        measure_id="union-topology-measure",
        producer_id="analytic-derivative-producer",
        run_id="ti-run",
        derivative_id="complete-du-dlambda",
        control_path_id="linear-control-path",
        qualification_id="exact-derivative-sampling",
        sampling_exact=True,
        sampling_bias_bound=0.0,
        unit_id="1",
    )
    result = phx.uq.thermodynamic_integration(dataset)
    np.testing.assert_allclose(result.free_energies, [0.0, 1.0, 2.0], atol=1.0e-12)
    np.testing.assert_allclose(dataset.quadrature_weights, [0.25, 0.5, 0.25])
    assert bool(result.successful)


def test_mbar_has_fixed_gauge_rank_aware_covariance_and_connectivity_evidence():
    delta = 0.8
    values = jnp.stack((jnp.zeros(12), jnp.full(12, delta)))
    dataset = _potential_dataset(values, [0] * 6 + [1] * 6)
    result = phx.uq.multistate_bennett_acceptance_ratio(dataset)
    np.testing.assert_allclose(result.free_energies, [0.0, delta], atol=1.0e-10)
    np.testing.assert_allclose(result.covariance[0], 0.0, atol=1.0e-12)
    assert int(result.covariance_rank) <= 1
    assert bool(jnp.all(result.connectivity))
    assert bool(result.successful)

    separated = _potential_dataset(
        jnp.asarray(
            [
                [0.0] * 6 + [1000.0] * 6,
                [1000.0] * 6 + [0.0] * 6,
            ]
        ),
        [0] * 6 + [1] * 6,
    )
    disconnected = phx.uq.multistate_bennett_acceptance_ratio(separated)
    assert int(disconnected.statistical_status) & int(
        phx.uq.FreeEnergyStatus.DISCONNECTED
    )
    assert not bool(disconnected.successful)


def test_mbar_admits_a_dense_cross_evaluated_unsampled_connected_state():
    values = jnp.zeros((3, 12))
    dataset = _potential_dataset(values, [0] * 6 + [1] * 6)
    result = phx.uq.multistate_bennett_acceptance_ratio(dataset)
    np.testing.assert_allclose(result.free_energies, 0.0, atol=1.0e-12)
    assert int(dataset.state_counts[2]) == 0
    assert bool(jnp.all(result.connectivity))
    assert bool(result.successful)


def test_selection_evidence_is_content_bound_and_blocks_dependence_groups_synchronously():
    values = jnp.zeros((2, 8))
    dataset = _potential_dataset(values, [0, 1] * 4)
    evidence = phx.uq.FreeEnergySelectionPlan(block_length=2).select(dataset)
    assert evidence.dataset_id == dataset.dataset_id
    np.testing.assert_array_equal(evidence.block_index, [0, 0, 1, 1, 2, 2, 3, 3])

    changed = _potential_dataset(values.at[0, 0].set(1.0), [0, 1] * 4)
    with pytest.raises(ValueError, match="not bound"):
        phx.uq.multistate_bennett_acceptance_ratio(changed, evidence)
