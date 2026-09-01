#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.bioinformatics.foundation import DifferentiationKind, MethodKind
from phydrax.bioinformatics.phylogenetics._ancestral import ancestral_marginals
from phydrax.bioinformatics.phylogenetics._clock import (
    relaxed_clock,
    strict_clock,
    strict_clock_likelihood,
)
from phydrax.bioinformatics.phylogenetics._pruning import (
    felsenstein_pruning,
    LikelihoodPartition,
    tip_partials_from_sequence,
)
from phydrax.bioinformatics.phylogenetics._rates import (
    discrete_rate_mixture,
    invariant_rate_mixture,
    with_invariant_sites,
)
from phydrax.bioinformatics.phylogenetics._search import (
    nni_topology_search,
    NNISearchPlan,
    NNISearchStatus,
)
from phydrax.bioinformatics.phylogenetics._substitution import (
    general_substitution_model,
    gtr,
    hky85,
    jc69,
    k80,
)
from phydrax.bioinformatics.phylogenetics._tree import (
    tree_topology,
    tree_topology_batch,
    TreeTopologyStatus,
)
from phydrax.bioinformatics.sequence import DNA_IUPAC, SequenceBatch


def _two_tip_tree():
    return tree_topology(jnp.asarray((2, 2, -1)))


def _one_hot_patterns(first, second, state_count=4):
    first_values = jax.nn.one_hot(jnp.asarray(first), state_count)
    second_values = jax.nn.one_hot(jnp.asarray(second), state_count)
    return jnp.stack((first_values, second_values), axis=1)


def _caterpillar_tree(tip_count):
    parent = np.full((2 * tip_count - 1,), -2, dtype=np.int32)
    current = 0
    for tip in range(1, tip_count):
        ancestor = tip_count + tip - 1
        parent[current] = ancestor
        parent[tip] = ancestor
        current = ancestor
    parent[current] = -1
    return tree_topology(parent)


def test_numeric_topology_traversals_batch_polytomy_and_capacity_failure():
    star = tree_topology(jnp.asarray((4, 4, 4, 4, -1)), child_capacity=4)
    assert bool(star.valid)
    assert int(star.evidence.maximum_out_degree) == 4
    assert int(star.postorder[-1]) == int(star.root_index)
    np.testing.assert_array_equal(star.preorder, jnp.asarray((4, 0, 1, 2, 3)))
    too_small = tree_topology(jnp.asarray((4, 4, 4, 4, -1)), child_capacity=3)
    assert not bool(too_small.valid)
    assert int(too_small.status) == int(TreeTopologyStatus.CHILD_CAPACITY_EXCEEDED)

    batch = tree_topology_batch(jnp.asarray(((2, 2, -1), (2, 2, -1))), child_capacity=2)
    assert batch.parent_indices.shape == (2, 3)
    assert bool(jnp.all(batch.valid))
    np.testing.assert_array_equal(batch.postorder[:, -1], jnp.asarray((2, 2)))


def test_jc69_transition_is_analytic_at_zero_finite_and_long_branches():
    model = jc69(dtype=jnp.float64)
    for time in (0.0, 0.7, 100.0):
        transition = model.transition_matrix(time)
        exponential = np.exp(-4.0 * time / 3.0)
        expected_same = 0.25 + 0.75 * exponential
        expected_different = 0.25 - 0.25 * exponential
        expected = np.full((4, 4), expected_different)
        np.fill_diagonal(expected, expected_same)
        np.testing.assert_allclose(transition, expected, rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(jnp.sum(transition, axis=-1), 1.0, atol=2e-10)


def test_k80_hky85_gtr_and_general_generators_are_conservative_and_normalized():
    models = (
        k80(3.0),
        hky85(jnp.asarray((0.3, 0.2, 0.25, 0.25)), 4.0),
        gtr(
            jnp.asarray((0.3, 0.2, 0.25, 0.25)),
            jnp.asarray((1.0, 2.0, 0.5, 1.5, 3.0, 0.75)),
        ),
    )
    for model in models:
        assert bool(model.valid)
        np.testing.assert_allclose(jnp.sum(model.rate_matrix, axis=-1), 0.0, atol=2e-6)
        np.testing.assert_allclose(
            -jnp.sum(model.root_distribution * jnp.diag(model.rate_matrix)),
            1.0,
            atol=2e-6,
        )
        flux = model.root_distribution[:, None] * model.rate_matrix
        np.testing.assert_allclose(flux, flux.T, atol=2e-6)

    two_state = general_substitution_model(
        jnp.asarray(((-2.0, 2.0), (1.0, -1.0))),
        root_distribution=jnp.asarray((1.0 / 3.0, 2.0 / 3.0)),
    )
    time = 0.4
    transition = two_state.transition_matrix(time)
    decay = np.exp(-3.0 * time)
    expected = np.asarray(
        (
            (1.0 / 3.0 + 2.0 * decay / 3.0, 2.0 / 3.0 - 2.0 * decay / 3.0),
            (1.0 / 3.0 - decay / 3.0, 2.0 / 3.0 + decay / 3.0),
        )
    )
    np.testing.assert_allclose(transition, expected, atol=2e-6)


def test_discrete_and_invariant_rate_mixtures_have_observable_normalization():
    discrete = discrete_rate_mixture(
        jnp.asarray((0.5, 1.0, 2.0)),
        jnp.asarray((0.2, 0.3, 0.5)),
        normalize_mean=True,
    )
    assert bool(discrete.valid)
    np.testing.assert_allclose(jnp.sum(discrete.weights), 1.0)
    np.testing.assert_allclose(jnp.sum(discrete.weights * discrete.rates), 1.0)

    invariant = invariant_rate_mixture(0.25)
    np.testing.assert_allclose(invariant.rates, jnp.asarray((0.0, 4.0 / 3.0)))
    np.testing.assert_allclose(invariant.weights, jnp.asarray((0.25, 0.75)))
    combined = with_invariant_sites(discrete, 0.2)
    np.testing.assert_allclose(jnp.sum(combined.weights * combined.rates), 1.0)
    invalid = invariant_rate_mixture(1.0)
    assert not bool(invalid.valid)


def test_two_tip_pruning_matches_direct_enumeration_and_pattern_weights():
    topology = _two_tip_tree()
    model = jc69(dtype=jnp.float64)
    partials = _one_hot_patterns((0, 0, 1), (0, 2, 3))
    weights = jnp.asarray((3.0, 2.0, 5.0))
    partition = LikelihoodPartition(jnp.ones((3,), dtype=bool), model)
    lengths = jnp.asarray((0.2, 0.35, 0.0))
    result = felsenstein_pruning(
        topology, partials, lengths, (partition,), pattern_weights=weights
    )
    first_transition = model.transition_matrix(lengths[0])
    second_transition = model.transition_matrix(lengths[1])
    direct = []
    for first, second in ((0, 0), (0, 2), (1, 3)):
        direct.append(
            jnp.sum(
                model.root_distribution
                * first_transition[:, first]
                * second_transition[:, second]
            )
        )
    direct = jnp.asarray(direct)
    assert bool(result.valid)
    np.testing.assert_allclose(result.pattern_log_likelihood, jnp.log(direct), atol=2e-7)
    np.testing.assert_allclose(
        result.log_likelihood, jnp.sum(weights * jnp.log(direct)), atol=2e-7
    )


def test_sequence_lowering_preserves_iupac_ambiguity_gaps_and_missingness():
    symbols = DNA_IUPAC.symbol_to_code
    tokens = jnp.asarray(
        (
            (symbols["A"], symbols["R"], symbols["-"], symbols["."]),
            (symbols["C"], symbols["N"], symbols["?"], symbols["#"]),
        )
    )
    batch = SequenceBatch(
        jnp.asarray((10, 11)),
        tokens,
        jnp.ones(tokens.shape, dtype=bool),
        jnp.asarray((True, True)),
        jnp.zeros(tokens.shape, dtype=bool),
        DNA_IUPAC,
    )
    result = tip_partials_from_sequence(batch)
    assert bool(result.valid)
    np.testing.assert_array_equal(result.tip_partials[0, 0], (1.0, 0.0, 0.0, 0.0))
    np.testing.assert_array_equal(result.tip_partials[1, 0], (1.0, 0.0, 1.0, 0.0))
    np.testing.assert_array_equal(result.tip_partials[1, 1], (1.0, 1.0, 1.0, 1.0))
    np.testing.assert_array_equal(
        result.tip_partials[2:], jnp.ones_like(result.tip_partials[2:])
    )


def test_zero_branches_ambiguity_missingness_and_child_order_are_exact():
    topology = _two_tip_tree()
    model = jc69(dtype=jnp.float64)
    partition = LikelihoodPartition(jnp.ones((3,), dtype=bool), model)
    partials = jnp.asarray(
        (
            ((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            ((1.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)),
            ((1.0, 1.0, 1.0, 1.0), (0.0, 1.0, 0.0, 0.0)),
        ),
        dtype=jnp.float64,
    )
    lengths = jnp.asarray((0.0, 0.0, 0.0))
    result = felsenstein_pruning(topology, partials, lengths, (partition,))
    np.testing.assert_allclose(
        result.pattern_log_likelihood,
        jnp.asarray((jnp.log(0.25), -jnp.inf, jnp.log(0.25))),
    )
    assert bool(result.valid)

    positive_lengths = jnp.asarray((0.4, 0.7, 0.0))
    baseline = felsenstein_pruning(topology, partials, positive_lengths, (partition,))
    reversed_children = eqx.tree_at(
        lambda tree: (tree.child_indices, tree.child_mask),
        topology,
        (topology.child_indices[:, ::-1], topology.child_mask[:, ::-1]),
    )
    reordered = felsenstein_pruning(
        reversed_children, partials, positive_lengths, (partition,)
    )
    np.testing.assert_allclose(
        reordered.pattern_log_likelihood, baseline.pattern_log_likelihood
    )


def test_polytomy_and_rate_mixture_match_direct_state_enumeration():
    topology = tree_topology(jnp.asarray((3, 3, 3, -1)), child_capacity=3)
    model = jc69(dtype=jnp.float64)
    mixture = discrete_rate_mixture(jnp.asarray((0.25, 2.0)), jnp.asarray((0.4, 0.6)))
    partition = LikelihoodPartition(jnp.asarray((True,)), model, rate_mixture=mixture)
    partials = jax.nn.one_hot(jnp.asarray((0, 1, 2)), 4)[None, :, :]
    lengths = jnp.asarray((0.1, 0.2, 0.3, 0.0))
    result = felsenstein_pruning(topology, partials, lengths, (partition,))
    direct_categories = []
    for rate in mixture.rates:
        transitions = [
            model.transition_matrix(lengths[index] * rate) for index in range(3)
        ]
        direct_categories.append(
            jnp.sum(
                model.root_distribution
                * transitions[0][:, 0]
                * transitions[1][:, 1]
                * transitions[2][:, 2]
            )
        )
    expected = jnp.sum(mixture.weights * jnp.asarray(direct_categories))
    np.testing.assert_allclose(result.log_likelihood, jnp.log(expected), atol=2e-7)


def test_partitions_and_variable_ascertainment_correction_are_exact():
    topology = _two_tip_tree()
    model = jc69(dtype=jnp.float64)
    partials = _one_hot_patterns((0, 0), (1, 2))
    lengths = jnp.asarray((0.15, 0.25, 0.0))
    uncorrected_partition = LikelihoodPartition(jnp.ones((2,), dtype=bool), model)
    uncorrected = felsenstein_pruning(
        topology, partials, lengths, (uncorrected_partition,)
    )
    constant_partials = jnp.broadcast_to(jnp.eye(4)[:, None, :], (4, 2, 4))
    constants = felsenstein_pruning(
        topology,
        constant_partials,
        lengths,
        (LikelihoodPartition(jnp.ones((4,), dtype=bool), model),),
    )
    variable_probability = 1.0 - jnp.sum(jnp.exp(constants.pattern_log_likelihood))
    corrected = felsenstein_pruning(
        topology,
        partials,
        lengths,
        (
            LikelihoodPartition(
                jnp.ones((2,), dtype=bool), model, ascertainment="variable"
            ),
        ),
    )
    np.testing.assert_allclose(
        corrected.pattern_log_likelihood,
        uncorrected.pattern_log_likelihood - jnp.log(variable_probability),
        atol=2e-7,
    )

    split = felsenstein_pruning(
        topology,
        partials,
        lengths,
        (
            LikelihoodPartition(jnp.asarray((True, False)), model),
            LikelihoodPartition(
                jnp.asarray((False, True)),
                model,
                rate_mixture=discrete_rate_mixture(
                    jnp.asarray((2.0,)), jnp.asarray((1.0,))
                ),
            ),
        ),
    )
    assert bool(split.valid)
    assert split.partition_log_likelihood.shape == (2,)


def test_deep_tree_log_scaling_remains_finite():
    topology = _caterpillar_tree(128)
    model = jc69()
    partials = jnp.broadcast_to(
        jnp.asarray((1.0, 0.0, 0.0, 0.0)), (1, topology.tip_count, 4)
    )
    lengths = jnp.full((topology.node_count,), 20.0).at[topology.root_index].set(0.0)
    result = felsenstein_pruning(
        topology,
        partials,
        lengths,
        (LikelihoodPartition(jnp.asarray((True,)), model),),
    )
    assert bool(result.valid)
    assert bool(jnp.isfinite(result.log_likelihood))
    np.testing.assert_allclose(result.log_likelihood, -128.0 * np.log(4.0), rtol=2e-5)


def test_ancestral_marginals_match_tiny_enumeration():
    topology = _two_tip_tree()
    model = jc69(dtype=jnp.float64)
    lengths = jnp.asarray((0.2, 0.4, 0.0))
    partials = _one_hot_patterns((0,), (2,))
    result = ancestral_marginals(
        topology,
        partials,
        lengths,
        (LikelihoodPartition(jnp.asarray((True,)), model),),
    )
    first = model.transition_matrix(lengths[0])[:, 0]
    second = model.transition_matrix(lengths[1])[:, 2]
    expected_root = model.root_distribution * first * second
    expected_root = expected_root / jnp.sum(expected_root)
    assert bool(result.valid)
    np.testing.assert_allclose(
        result.marginals[0, topology.root_index], expected_root, atol=2e-7
    )
    np.testing.assert_allclose(
        result.marginals[0, 0], jnp.asarray((1.0, 0.0, 0.0, 0.0)), atol=2e-7
    )
    np.testing.assert_allclose(jnp.sum(result.marginals, axis=-1), 1.0, atol=2e-7)


def test_branch_and_clock_rate_gradients_are_finite_and_nonzero():
    topology = _two_tip_tree()
    model = jc69(dtype=jnp.float64)
    partials = _one_hot_patterns((0,), (2,))
    partition = LikelihoodPartition(jnp.asarray((True,)), model)

    def branch_objective(nonroot_lengths):
        lengths = jnp.concatenate((nonroot_lengths, jnp.zeros((1,))))
        return felsenstein_pruning(
            topology, partials, lengths, (partition,)
        ).log_likelihood

    branch_gradient = jax.grad(branch_objective)(jnp.asarray((0.2, 0.4)))
    assert bool(jnp.all(jnp.isfinite(branch_gradient)))
    assert bool(jnp.all(jnp.abs(branch_gradient) > 1e-8))

    node_times = jnp.asarray((0.0, 0.0, 1.0))

    def rate_objective(rate):
        return strict_clock_likelihood(
            topology, partials, node_times, rate, (partition,)
        ).log_likelihood

    rate_gradient = jax.grad(rate_objective)(jnp.asarray(0.3))
    assert bool(jnp.isfinite(rate_gradient))
    assert bool(jnp.abs(rate_gradient) > 1e-8)


def test_strict_and_relaxed_clocks_preserve_temporal_rate_semantics():
    topology = _two_tip_tree()
    times = jnp.asarray((0.0, 0.25, 1.0))
    strict = strict_clock(topology, times, 2.0)
    relaxed = relaxed_clock(topology, times, jnp.asarray((2.0, 3.0, 99.0)))
    assert bool(strict.valid)
    assert bool(relaxed.valid)
    np.testing.assert_allclose(strict.branch_lengths, jnp.asarray((2.0, 1.5, 0.0)))
    np.testing.assert_allclose(relaxed.branch_lengths, jnp.asarray((2.0, 2.25, 0.0)))
    invalid = strict_clock(topology, jnp.asarray((0.0, 1.0, 0.5)), 1.0)
    assert not bool(invalid.valid)


def test_bounded_nni_is_explicitly_heuristic_nondifferentiable_and_preflighted():
    topology = tree_topology(jnp.asarray((4, 5, 4, 5, 6, 6, -1)))
    model = jc69()
    # Four tip observations in one pattern: A,A,C,C.
    partials = jax.nn.one_hot(jnp.asarray((0, 0, 1, 1)), 4)[None, :, :]
    partition = LikelihoodPartition(jnp.asarray((True,)), model)
    lengths = jnp.full((7,), 0.2).at[6].set(0.0)
    initial = felsenstein_pruning(topology, partials, lengths, (partition,))
    capacity_failure = nni_topology_search(
        topology,
        partials,
        lengths,
        (partition,),
        NNISearchPlan(2, 1),
    )
    assert not bool(capacity_failure.valid)
    assert int(capacity_failure.status) == int(
        NNISearchStatus.CANDIDATE_CAPACITY_EXCEEDED
    )
    assert int(capacity_failure.evidence.evaluated_candidates) == 0

    result = nni_topology_search(
        topology,
        partials,
        lengths,
        (partition,),
        NNISearchPlan(2, 16),
    )
    assert result.method_contract.method_kind is MethodKind.HEURISTIC
    assert result.method_contract.differentiation_kind is DifferentiationKind.NONE
    assert result.claim_kind == "bounded_nni_heuristic"
    assert float(result.log_likelihood) >= float(initial.log_likelihood)
