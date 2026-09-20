import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _sparse_dataset(state_count, edge_specs, *, omit_reverse_cross=False):
    per_direction = 8
    sample_count = 2 * per_direction * len(edge_specs)
    values = np.zeros((state_count, sample_count), dtype="float64")
    coverage = np.zeros((state_count, sample_count), dtype="bool")
    origins = np.zeros((sample_count,), dtype=np.int32)
    chains = np.zeros((sample_count,), dtype=np.int32)
    draws = np.zeros((sample_count,), dtype=np.int32)
    repeats = np.zeros((sample_count,), dtype=np.int32)
    dependence = np.zeros((sample_count,), dtype=np.int32)
    cursor = 0
    for edge_index, (source, destination, delta, group) in enumerate(edge_specs):
        forward = slice(cursor, cursor + per_direction)
        reverse = slice(cursor + per_direction, cursor + 2 * per_direction)
        origins[forward] = source
        origins[reverse] = destination
        chains[forward] = 2 * edge_index
        chains[reverse] = 2 * edge_index + 1
        draws[forward] = np.arange(per_direction)
        draws[reverse] = np.arange(per_direction)
        dependence[forward] = group
        dependence[reverse] = group
        coverage[source, forward] = True
        coverage[destination, forward] = True
        coverage[destination, reverse] = True
        if not omit_reverse_cross:
            coverage[source, reverse] = True
        values[destination, forward] = delta
        values[destination, reverse] = delta
        cursor += 2 * per_direction
    return phx.uq.SparseReducedPotentialDataset(
        values,
        coverage,
        jnp.ones((sample_count,), dtype="bool"),
        origins,
        chains,
        draws,
        repeats,
        dependence,
        state_ids=tuple(f"state-{index}" for index in range(state_count)),
        potential_ids=tuple(f"potential-{index}" for index in range(state_count)),
        inverse_temperatures=jnp.ones((state_count,)),
        measure_id="sparse-common-measure",
        producer_id="sparse-analytic-producer",
        run_id="sparse-analytic-run",
        reduced_convention_id="analytic-reduced-potential",
        qualification_id="exact-independent-sparse-sampling",
        sampling_exact=True,
        sampling_bias_bound=0.0,
        bias_ids=(None,) * state_count,
        unit_id="1",
    )


def test_sparse_pairwise_bar_builds_connected_network_from_disjoint_groups():
    dataset = _sparse_dataset(3, ((0, 1, 1.0, 0), (1, 2, 2.0, 1)))
    result = phx.uq.sparse_pairwise_free_energy_network(
        dataset,
        (("state-0", "state-1"), ("state-1", "state-2")),
        reference_state_id="state-0",
    )
    np.testing.assert_allclose(result.network_result.free_energies, [0.0, 1.0, 3.0])
    assert len(result.edge_observations) == 2
    assert bool(result.successful)


def test_sparse_pairwise_network_reports_disconnected_explicit_graph():
    dataset = _sparse_dataset(4, ((0, 1, 1.0, 0), (2, 3, 1.5, 1)))
    result = phx.uq.sparse_pairwise_free_energy_network(
        dataset,
        (("state-0", "state-1"), ("state-2", "state-3")),
        reference_state_id="state-0",
    )
    assert int(result.network_result.statistical_status) & int(
        phx.uq.FreeEnergyStatus.DISCONNECTED
    )
    assert not bool(result.successful)


def test_sparse_pairwise_bar_rejects_missing_reverse_cross_coverage():
    dataset = _sparse_dataset(
        2,
        ((0, 1, 1.0, 0),),
        omit_reverse_cross=True,
    )
    with pytest.raises(ValueError, match="qualified BAR evidence"):
        phx.uq.sparse_pairwise_free_energy_network(
            dataset,
            (("state-0", "state-1"),),
            reference_state_id="state-0",
        )


def test_sparse_pairwise_network_rejects_shared_dependence_groups():
    dataset = _sparse_dataset(3, ((0, 1, 1.0, 0), (1, 2, 2.0, 0)))
    with pytest.raises(ValueError, match="share dependence groups"):
        phx.uq.sparse_pairwise_free_energy_network(
            dataset,
            (("state-0", "state-1"), ("state-1", "state-2")),
            reference_state_id="state-0",
        )
