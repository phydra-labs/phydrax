#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.bioinformatics.foundation import MethodKind
from phydrax.bioinformatics.sequence import decode_sequences, DNA_IUPAC, encode_sequences
from phydrax.bioinformatics.sequence._motifs import (
    MotifScanPlan,
    PositionWeightMatrix,
    scan_motif,
)
from phydrax.bioinformatics.sequence._multiple_alignment import (
    GuideTree,
    MSA_STATUS_CAPACITY_EXCEEDED,
    progressive_multiple_alignment,
    ProgressiveMSAPlan,
)
from phydrax.bioinformatics.sequence._poa import (
    align_partial_order,
    PartialOrderAlignmentPlan,
    PartialOrderGraph,
    POA_MATCH,
)
from phydrax.bioinformatics.sequence._profile_hmm import (
    DELETE_STATE,
    profile_forward_backward,
    profile_viterbi,
    ProfileHMM,
    ProfileHMMPlan,
)
from phydrax.sparse import EdgeRelation


def test_pwm_scans_ambiguity_both_strands_gaps_and_soft_masks() -> None:
    pwm = PositionWeightMatrix(
        jnp.asarray(
            (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
            )
        ),
        DNA_IUPAC,
        motif_id="ACG",
    )
    sequences = encode_sequences(["ACGANGCGT-ACG", "acgACG"], DNA_IUPAC)
    result = scan_motif(pwm, sequences, MotifScanPlan(13))

    assert bool(result.valid[0])
    np.testing.assert_allclose(result.forward_scores[0, 0], 3.0 * np.log(4.0))
    # N marginalizes over all bases and contributes a neutral log-odds term.
    np.testing.assert_allclose(result.forward_scores[0, 3], 2.0 * np.log(4.0))
    assert int(result.strand[0, 6]) == -1  # CGT is the reverse complement of ACG.
    assert not bool(result.window_valid[0, 8])  # Every window spanning '-' is invalid.
    assert not bool(result.window_valid[1, 0])  # Lowercase positions are soft masked.
    assert int(result.best_position[1]) == 3
    assert int(result.best_strand[1]) == 1
    assert int(result.evidence.ambiguity_positions[0]) == 1
    assert int(result.evidence.masked_positions[1]) == 3


def test_pwm_forward_strand_wins_exact_reverse_complement_ties() -> None:
    palindrome = PositionWeightMatrix(
        jnp.asarray(((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))),
        DNA_IUPAC,
        motif_id="AT",
    )
    result = scan_motif(palindrome, encode_sequences(["AT"], DNA_IUPAC))
    np.testing.assert_allclose(result.forward_scores, result.reverse_scores)
    assert int(result.best_strand[0]) == 1
    assert result.method_contract.method_kind is MethodKind.EXACT_MODEL


def test_profile_hmm_forward_backward_covers_match_insert_delete_and_terminal_mass() -> (
    None
):
    model = ProfileHMM(
        jnp.asarray(((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0))),
        DNA_IUPAC,
        profile_id="AC-profile",
    )
    sequences = encode_sequences(["A", "AN"], DNA_IUPAC)
    result = profile_forward_backward(model, sequences, ProfileHMMPlan(2, 2))

    assert np.all(np.asarray(result.valid))
    np.testing.assert_allclose(result.evidence.forward_backward_residual, 0.0, atol=2e-6)
    np.testing.assert_allclose(result.terminal_marginal, 1.0, atol=2e-6)
    np.testing.assert_allclose(result.evidence.emitted_mass_residual, 0.0, atol=2e-6)
    assert np.all(np.asarray(result.class_posterior_mass[:, 0]) > 0.0)
    assert np.all(np.asarray(result.class_posterior_mass[:, 1]) > 0.0)
    assert np.all(np.asarray(result.class_posterior_mass[:, 2]) > 0.0)
    assert np.any(np.asarray(result.delete_marginals[0]) > 0.0)
    assert np.all(np.asarray(model.terminal_probabilities) > 0.0)

    path = profile_viterbi(model, encode_sequences(["A"], DNA_IUPAC))
    length = int(path.path.length[0])
    assert bool(path.valid[0])
    assert DELETE_STATE in np.asarray(path.path.state_class[0, :length])
    assert int(path.evidence.query_positions_consumed[0]) == 1


def test_profile_hmm_gap_observation_is_infeasible_not_treated_as_padding() -> None:
    model = ProfileHMM(jnp.asarray(((1.0, 0.0, 0.0, 0.0),)), DNA_IUPAC)
    result = profile_forward_backward(model, encode_sequences(["-"], DNA_IUPAC))
    assert not bool(result.valid[0])
    assert not np.isfinite(float(result.log_likelihood[0]))


def _branch_graph(edge_order: np.ndarray | None = None) -> PartialOrderGraph:
    sources = np.asarray((0, 0, 1, 2), dtype=np.int32)
    targets = np.asarray((1, 2, 3, 3), dtype=np.int32)
    if edge_order is not None:
        sources = sources[edge_order]
        targets = targets[edge_order]
    relation = EdgeRelation(sources, targets, source_size=4, target_size=4)
    return PartialOrderGraph(
        jnp.asarray(
            (
                DNA_IUPAC.code("A"),
                DNA_IUPAC.code("C"),
                DNA_IUPAC.code("C"),
                DNA_IUPAC.code("T"),
            )
        ),
        relation,
        DNA_IUPAC,
    )


def test_partial_order_alignment_rejects_cycles_and_decodes_feasible_dag_path() -> None:
    cyclic = EdgeRelation(
        jnp.asarray((0, 1)),
        jnp.asarray((1, 0)),
        source_size=2,
        target_size=2,
    )
    with pytest.raises(ValueError, match="acyclic"):
        PartialOrderGraph(
            jnp.asarray((DNA_IUPAC.code("A"), DNA_IUPAC.code("C"))),
            cyclic,
            DNA_IUPAC,
        )

    plan = PartialOrderAlignmentPlan(4, 4, 2, 3)
    query = encode_sequences(["ACT"], DNA_IUPAC)
    first = align_partial_order(_branch_graph(), query, plan)
    permuted = align_partial_order(_branch_graph(np.asarray((3, 1, 2, 0))), query, plan)
    assert bool(first.valid[0])
    assert bool(first.evidence.acyclic[0])
    assert bool(first.evidence.reaches_source[0])
    assert bool(first.evidence.reaches_sink[0])
    assert bool(first.evidence.graph_edges_feasible[0])
    assert int(first.evidence.query_consumed[0]) == 3
    np.testing.assert_allclose(first.evidence.score_residual[0], 0.0, atol=2e-6)
    length = int(first.path.length[0])
    matched_nodes = first.path.node_indices[0, :length][
        first.path.operations[0, :length] == POA_MATCH
    ]
    np.testing.assert_array_equal(matched_nodes, (0, 1, 3))
    np.testing.assert_array_equal(first.path.node_indices, permuted.path.node_indices)


def test_progressive_msa_is_permutation_stable_and_never_globally_exact() -> None:
    strings = ("AC", "AG", "AT")
    plan = ProgressiveMSAPlan(3, 2, 6)
    first = progressive_multiple_alignment(encode_sequences(strings, DNA_IUPAC), plan)
    permutation = (2, 0, 1)
    second = progressive_multiple_alignment(
        encode_sequences(tuple(strings[index] for index in permutation), DNA_IUPAC), plan
    )
    inverse = np.argsort(np.asarray(permutation))
    second_rows = decode_sequences(second.alignment)

    assert bool(first.valid) and bool(second.valid)
    assert not plan.globally_exact
    assert not bool(first.evidence.globally_exact)
    assert bool(first.evidence.guide_tree_heuristic)
    assert bool(first.evidence.profile_alignment_heuristic)
    assert first.method_contract.method_kind is MethodKind.HEURISTIC
    assert int(first.evidence.permutation_tie_count) > 0
    assert decode_sequences(first.alignment) == tuple(
        second_rows[index] for index in inverse
    )


def test_progressive_msa_preserves_ambiguity_gaps_soft_masks_and_supplied_tree_claim() -> (
    None
):
    sequences = encode_sequences(["aN-C", "ANCC", "A-C"], DNA_IUPAC)
    tree = GuideTree(
        jnp.asarray(((0, 1), (3, 2))),
        jnp.asarray((0.1, 0.2)),
        3,
    )
    plan = ProgressiveMSAPlan(3, 4, 12, guide_tree_method="supplied")
    result = progressive_multiple_alignment(sequences, plan, tree)
    assert bool(result.valid)
    assert not bool(result.evidence.guide_tree_heuristic)
    assert bool(result.evidence.profile_alignment_heuristic)
    assert not bool(result.evidence.globally_exact)
    assert np.any(np.asarray(result.alignment.soft_mask[0]))
    gap = DNA_IUPAC.code(DNA_IUPAC.gap_symbol)
    active = np.asarray(result.alignment.token_codes[:, : int(result.alignment_length)])
    assert np.any(active == gap)
    assert np.all(np.any(active != gap, axis=0))

    too_small = ProgressiveMSAPlan(3, 4, 3, guide_tree_method="supplied")
    failure = progressive_multiple_alignment(sequences, too_small, tree)
    assert not bool(failure.valid)
    assert int(failure.status) == MSA_STATUS_CAPACITY_EXCEEDED
    assert int(failure.alignment_length) == 0
    assert not np.any(np.asarray(failure.column_mask))
