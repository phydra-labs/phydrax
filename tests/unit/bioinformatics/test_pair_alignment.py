#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.bioinformatics.sequence._alignment import (
    AffineGapPenalties,
    align_affine,
    ALIGNMENT_CAPACITY_EXCEEDED,
    ALIGNMENT_CONDITIONAL_BAND,
    ALIGNMENT_IMPOSSIBLE,
    AlignmentExecutionPlan,
    INSERT,
    MATCH,
)
from phydrax.bioinformatics.sequence._pair_hmm import (
    PAIR_HMM_CONDITIONAL_BAND,
    pair_hmm_forward_backward,
    pair_hmm_forward_backward_from_potentials,
    PAIR_HMM_IMPOSSIBLE,
    PairHMM,
    PairHMMExecutionPlan,
)
from phydrax.bioinformatics.sequence._scoring import nucleotide_substitution_table


def _one_state_alphabet_pair_hmm() -> PairHMM:
    return PairHMM(
        jnp.zeros((3,)),
        jnp.zeros((3, 3)),
        jnp.zeros((3,)),
        jnp.zeros((1, 1)),
        jnp.zeros((1,)),
        jnp.zeros((1,)),
    )


def test_iupac_ambiguity_scores_are_distribution_expectations():
    table = nucleotide_substitution_table(match_score=2.0, mismatch_score=-3.0)
    assert table.symbols[14] == "N"
    np.testing.assert_allclose(table.score_codes(14, 0), -1.75)
    np.testing.assert_allclose(table.score_codes(4, 0), -0.5)


def test_global_affine_open_extend_padding_and_traceback_score():
    table = nucleotide_substitution_table(match_score=2.0, mismatch_score=-3.0)
    plan = AlignmentExecutionPlan.full(4, 3, traceback_capacity=7)
    result = align_affine(
        jnp.asarray((0, 0, 0, 99)),
        jnp.asarray((0, 99, 99)),
        table,
        AffineGapPenalties(-4.0, -1.0),
        plan,
        query_mask=jnp.asarray((True, True, True, False)),
        target_mask=jnp.asarray((True, False, False)),
    )
    assert bool(result.valid)
    np.testing.assert_allclose(result.score, 2.0 - 4.0 - 1.0)
    np.testing.assert_allclose(result.traceback_score, result.score)
    length = int(result.alignment_length)
    assert length == 3
    operations = np.asarray(result.operations[:length])
    assert np.count_nonzero(operations == MATCH) == 1
    assert np.count_nonzero(operations == INSERT) == 2


def test_local_and_semiglobal_have_distinct_terminal_semantics():
    table = nucleotide_substitution_table(match_score=3.0, mismatch_score=-4.0)
    penalties = AffineGapPenalties(-5.0, -1.0)
    plan = AlignmentExecutionPlan.full(3, 2, traceback_capacity=5)
    local = align_affine(
        jnp.asarray((0, 1)),
        jnp.asarray((3, 1)),
        table,
        penalties,
        plan,
        mode="local",
    )
    semiglobal = align_affine(
        jnp.asarray((3, 0)),
        jnp.asarray((0,)),
        table,
        penalties,
        plan,
        mode="semiglobal",
    )
    np.testing.assert_allclose(local.score, 3.0)
    np.testing.assert_allclose(local.traceback_score, local.score)
    assert int(local.alignment_length) == 1
    np.testing.assert_allclose(semiglobal.score, 3.0)
    np.testing.assert_allclose(semiglobal.traceback_score, semiglobal.score)
    assert int(semiglobal.alignment_length) == 1


def test_alignment_empty_impossible_and_capacity_failures_are_observable():
    table = nucleotide_substitution_table()
    penalties = AffineGapPenalties(-4.0, -1.0)
    empty = align_affine(
        jnp.asarray((0,)),
        jnp.asarray((0,)),
        table,
        penalties,
        AlignmentExecutionPlan.full(1, 1, traceback_capacity=2),
        query_mask=jnp.asarray((False,)),
        target_mask=jnp.asarray((False,)),
    )
    np.testing.assert_allclose(empty.score, 0.0)
    assert bool(empty.valid)
    assert int(empty.alignment_length) == 0

    capacity = align_affine(
        jnp.asarray((0, 0)),
        jnp.asarray((0, 0)),
        table,
        penalties,
        AlignmentExecutionPlan.full(2, 2, traceback_capacity=3),
    )
    assert not bool(capacity.valid)
    assert int(capacity.status) == ALIGNMENT_CAPACITY_EXCEEDED

    banded = align_affine(
        jnp.asarray((0, 0)),
        jnp.asarray((0,)),
        table,
        penalties,
        AlignmentExecutionPlan.diagonal_band(2, 1, traceback_capacity=3, band_radius=0),
    )
    assert not bool(banded.valid)
    assert int(banded.status) == ALIGNMENT_IMPOSSIBLE
    assert bool(banded.truncated)
    assert bool(banded.expansion_required)


def test_alignment_band_is_explicitly_conditional_when_feasible():
    table = nucleotide_substitution_table()
    result = align_affine(
        jnp.asarray((0, 0)),
        jnp.asarray((0, 0)),
        table,
        AffineGapPenalties(-4.0, -1.0),
        AlignmentExecutionPlan.diagonal_band(2, 2, traceback_capacity=4, band_radius=0),
    )
    assert bool(result.valid)
    assert int(result.status) == ALIGNMENT_CONDITIONAL_BAND
    assert not bool(result.exact)
    assert bool(result.truncated)
    assert bool(result.boundary_hit)


def test_pair_hmm_tiny_partition_matches_complete_path_enumeration():
    model = _one_state_alphabet_pair_hmm()
    plan = PairHMMExecutionPlan.full(2, 2, traceback_capacity=4)
    result = pair_hmm_forward_backward_from_potentials(
        model,
        jnp.zeros((1, 1)),
        jnp.zeros((1,)),
        jnp.zeros((1,)),
        plan,
    )
    # Paths are M, I→D, and D→I with probabilities 1/9, 1/27, and 1/27.
    np.testing.assert_allclose(result.log_partition, np.log(5.0 / 27.0), rtol=1e-6)
    np.testing.assert_allclose(result.viterbi_score, np.log(1.0 / 9.0), rtol=1e-6)
    np.testing.assert_allclose(
        result.forward_log_partition, result.backward_log_partition, atol=2e-6
    )
    np.testing.assert_allclose(result.initial_marginals.sum(), 1.0, atol=2e-6)
    np.testing.assert_allclose(result.terminal_marginals.sum(), 1.0, atol=2e-6)
    np.testing.assert_allclose(result.expected_state_counts.sum(), 7.0 / 5.0, atol=2e-6)
    np.testing.assert_allclose(
        result.expected_transition_counts.sum(), 2.0 / 5.0, atol=2e-6
    )
    assert result.state_marginals.shape == (3, 3, 3)
    assert result.transition_marginals.shape == (3, 3, 3, 3)
    assert float(result.posterior_conservation_error) < 2e-6


def test_pair_hmm_potential_gradient_is_state_occupancy():
    model = _one_state_alphabet_pair_hmm()
    plan = PairHMMExecutionPlan.full(1, 1, traceback_capacity=2)
    insertion = jnp.zeros((1,))
    deletion = jnp.zeros((1,))

    def partition(match):
        return pair_hmm_forward_backward_from_potentials(
            model, match, insertion, deletion, plan
        ).log_partition

    match = jnp.zeros((1, 1))
    result = pair_hmm_forward_backward_from_potentials(
        model, match, insertion, deletion, plan
    )
    gradient = jax.grad(partition)(match)
    np.testing.assert_allclose(
        gradient[0, 0], result.state_marginals[1, 1, MATCH], atol=2e-6
    )


def test_normalized_transition_logit_gradient_matches_transition_occupancy():
    plan = PairHMMExecutionPlan.full(1, 1, traceback_capacity=2)
    initial = jnp.zeros((3,))
    terminal = jnp.zeros((3,))
    emission = jnp.zeros((1,))

    def partition(transition_logits):
        model = PairHMM(
            initial,
            transition_logits,
            terminal,
            jnp.zeros((1, 1)),
            emission,
            emission,
        )
        return pair_hmm_forward_backward_from_potentials(
            model,
            jnp.zeros((1, 1)),
            jnp.zeros((1,)),
            jnp.zeros((1,)),
            plan,
        ).log_partition

    transition_logits = jnp.asarray(
        ((0.3, -0.2, 0.1), (-0.4, 0.5, 0.0), (0.2, -0.1, 0.4))
    )
    model = PairHMM(
        initial,
        transition_logits,
        terminal,
        jnp.zeros((1, 1)),
        emission,
        emission,
    )
    result = pair_hmm_forward_backward_from_potentials(
        model,
        jnp.zeros((1, 1)),
        jnp.zeros((1,)),
        jnp.zeros((1,)),
        plan,
    )
    transition_probability = jnp.exp(model.normalized_log_parameters()[1])
    occupancy = result.expected_transition_counts
    expected_gradient = occupancy - transition_probability * jnp.sum(
        occupancy, axis=1, keepdims=True
    )
    np.testing.assert_allclose(
        jax.grad(partition)(transition_logits), expected_gradient, atol=3e-6
    )


def test_pair_hmm_ambiguity_emissions_and_padding_preserve_likelihood():
    negative_infinity = -jnp.inf
    model = PairHMM(
        jnp.asarray((0.0, negative_infinity, negative_infinity)),
        jnp.asarray(
            (
                (0.0, negative_infinity, negative_infinity),
                (negative_infinity, 0.0, negative_infinity),
                (negative_infinity, negative_infinity, 0.0),
            )
        ),
        jnp.asarray((0.0, negative_infinity, negative_infinity)),
        jnp.asarray(((2.0, -2.0), (-2.0, 1.0))),
        jnp.zeros((2,)),
        jnp.zeros((2,)),
    )
    plan = PairHMMExecutionPlan.full(2, 2, traceback_capacity=4)
    padded = pair_hmm_forward_backward(
        model,
        jnp.asarray(((0.5, 0.5), (0.0, 0.0))),
        jnp.asarray(((1.0, 0.0), (0.0, 0.0))),
        plan,
        left_mask=jnp.asarray((True, False)),
        right_mask=jnp.asarray((True, False)),
    )
    cropped = pair_hmm_forward_backward(
        model,
        jnp.asarray(((0.5, 0.5),)),
        jnp.asarray(((1.0, 0.0),)),
        plan,
    )
    assert bool(padded.valid)
    np.testing.assert_allclose(padded.log_partition, cropped.log_partition, atol=2e-6)
    (log_match,) = (model.normalized_log_parameters()[3],)
    expected = jnp.log(0.5 * jnp.exp(log_match[0, 0]) + 0.5 * jnp.exp(log_match[1, 0]))
    np.testing.assert_allclose(padded.log_partition, expected, atol=2e-6)


def test_pair_hmm_empty_and_impossible_paths_are_distinguished():
    model = _one_state_alphabet_pair_hmm()
    plan = PairHMMExecutionPlan.full(1, 1, traceback_capacity=2)
    empty = pair_hmm_forward_backward_from_potentials(
        model,
        jnp.zeros((0, 0)),
        jnp.zeros((0,)),
        jnp.zeros((0,)),
        plan,
    )
    np.testing.assert_allclose(empty.log_partition, 0.0)
    assert bool(empty.valid)
    assert int(empty.viterbi_length) == 0

    negative_infinity = -jnp.inf
    match_only = PairHMM(
        jnp.asarray((0.0, negative_infinity, negative_infinity)),
        jnp.asarray(
            (
                (0.0, negative_infinity, negative_infinity),
                (negative_infinity, 0.0, negative_infinity),
                (negative_infinity, negative_infinity, 0.0),
            )
        ),
        jnp.asarray((0.0, negative_infinity, negative_infinity)),
        jnp.zeros((1, 1)),
        jnp.zeros((1,)),
        jnp.zeros((1,)),
    )
    impossible = pair_hmm_forward_backward_from_potentials(
        match_only,
        jnp.zeros((1, 0)),
        jnp.zeros((1,)),
        jnp.zeros((0,)),
        plan,
    )
    assert not bool(impossible.valid)
    assert int(impossible.status) == PAIR_HMM_IMPOSSIBLE
    assert bool(jnp.isneginf(impossible.log_partition))


def test_checkpointed_is_exact_and_band_is_conditional_with_diagnostics():
    model = _one_state_alphabet_pair_hmm()
    full = pair_hmm_forward_backward_from_potentials(
        model,
        jnp.zeros((1, 1)),
        jnp.zeros((1,)),
        jnp.zeros((1,)),
        PairHMMExecutionPlan.full(1, 1, traceback_capacity=2),
    )
    checkpointed = pair_hmm_forward_backward_from_potentials(
        model,
        jnp.zeros((1, 1)),
        jnp.zeros((1,)),
        jnp.zeros((1,)),
        PairHMMExecutionPlan.checkpointed(
            1, 1, traceback_capacity=2, checkpoint_stride=1
        ),
    )
    np.testing.assert_allclose(checkpointed.log_partition, full.log_partition, atol=2e-6)
    np.testing.assert_allclose(
        checkpointed.state_marginals, full.state_marginals, atol=2e-6
    )
    assert bool(checkpointed.exact)

    banded = pair_hmm_forward_backward_from_potentials(
        model,
        jnp.zeros((1, 1)),
        jnp.zeros((1,)),
        jnp.zeros((1,)),
        PairHMMExecutionPlan.diagonal_band(
            1,
            1,
            traceback_capacity=2,
            band_radius=0,
            boundary_mass_tolerance=0.0,
        ),
    )
    np.testing.assert_allclose(banded.log_partition, np.log(1.0 / 9.0), atol=2e-6)
    assert bool(banded.valid)
    assert int(banded.status) == PAIR_HMM_CONDITIONAL_BAND
    assert bool(banded.truncated)
    assert 0.0 <= float(banded.boundary_mass) <= 1.0
    assert bool(banded.expansion_required)
