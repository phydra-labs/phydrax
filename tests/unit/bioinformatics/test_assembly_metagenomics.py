#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from phydrax.bioinformatics.foundation import FeatureDictionary, FeatureMapping
from phydrax.bioinformatics.genomics._assembly import (
    AssemblyStatus,
    certify_supplied_dag,
    decode_supplied_paths,
    decode_supplied_unitigs,
    score_supplied_overlaps,
    SuppliedAssemblyPaths,
    SuppliedOverlapCandidates,
)
from phydrax.bioinformatics.genomics._variation_graph import (
    build_supplied_variation_graph,
    index_graph_paths,
    locate_graph_path_positions,
)
from phydrax.bioinformatics.metagenomics._assignment import (
    assign_taxonomy,
    AssignmentStatus,
    SuppliedTaxonomicCandidates,
)
from phydrax.bioinformatics.metagenomics._binning import (
    evaluate_binning_markers,
    supplied_contig_binning,
)
from phydrax.bioinformatics.metagenomics._community import (
    estimate_community_abundance,
)
from phydrax.bioinformatics.metagenomics._functional import quantify_functional_profile
from phydrax.bioinformatics.metagenomics._sketches import (
    count_kmers,
    KmerCountingPlan,
    minhash_sketch,
    MinHashPlan,
    SketchStatus,
)
from phydrax.bioinformatics.metagenomics._taxonomy import (
    build_taxonomy_tree,
    resolve_taxon_ids,
    TaxonomyStatus,
    TaxonomyVersion,
)
from phydrax.bioinformatics.sequence import DNA_IUPAC, SequenceBatch
from phydrax.sparse import EdgeRelation


def _dna_batch(sequences: tuple[str, ...]) -> SequenceBatch:
    width = max(1, *(len(sequence) for sequence in sequences))
    pad = DNA_IUPAC.code(DNA_IUPAC.pad_symbol)
    tokens = np.full((len(sequences), width), pad, dtype=np.int32)
    valid = np.zeros(tokens.shape, dtype=bool)
    for row, sequence in enumerate(sequences):
        for column, symbol in enumerate(sequence):
            tokens[row, column] = DNA_IUPAC.code(symbol)
            valid[row, column] = True
    return SequenceBatch(
        jnp.arange(len(sequences), dtype=jnp.int32),
        jnp.asarray(tokens),
        jnp.asarray(valid),
        jnp.ones((len(sequences),), dtype=bool),
        jnp.zeros(tokens.shape, dtype=bool),
        DNA_IUPAC,
    )


def _taxonomy():
    version = TaxonomyVersion("fixture", "2026-08", "fixture-content-sha256")
    result = build_taxonomy_tree(
        jnp.asarray((1, 2, 3), dtype=jnp.int32),
        jnp.asarray((0, 1, 1), dtype=jnp.int32),
        jnp.asarray((0, 0), dtype=jnp.int32),
        jnp.asarray((1, 2), dtype=jnp.int32),
        jnp.asarray((True, True)),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray((0, 1, 2), dtype=jnp.int32),
        jnp.asarray((True, True, True)),
        version,
        merged_taxon_ids=jnp.asarray((20,), dtype=jnp.int32),
        merged_into_ids=jnp.asarray((2,), dtype=jnp.int32),
        deleted_taxon_ids=jnp.asarray((30,), dtype=jnp.int32),
    )
    assert bool(result.valid)
    return result.taxonomy, version


def test_reverse_complement_overlap_and_repeat_cycle_certificates():
    reads = _dna_batch(("AAC", "GTT"))
    candidates = SuppliedOverlapCandidates(
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray((1,), dtype=jnp.int32),
        jnp.asarray((False,)),
        jnp.asarray((True,)),
        read_capacity=2,
    )
    scored = score_supplied_overlaps(reads, candidates, minimum_overlap=3)
    assert bool(scored.valid)
    assert int(scored.graph.overlap_lengths[0]) == 3
    assert isinstance(scored.graph.relation, EdgeRelation)

    certificate = certify_supplied_dag(
        scored.graph.relation,
        scored.graph.node_valid,
        jnp.asarray((0, 1), dtype=jnp.int32),
        jnp.asarray((True, True)),
        graph_id=scored.graph.graph_id,
    )
    assert bool(certificate.valid)
    unitigs = SuppliedAssemblyPaths(
        jnp.asarray((6,), dtype=jnp.int32),
        jnp.asarray(((0, 1),), dtype=jnp.int32),
        jnp.asarray(((True, True),)),
        jnp.asarray(((False, True),)),
        jnp.asarray((True,)),
    )
    decoded = decode_supplied_unitigs(
        scored.graph,
        certificate,
        reads,
        unitigs,
        output_capacity=3,
    )
    assert bool(decoded.valid)
    assert int(decoded.path_lengths[0]) == 3
    assert bool(decoded.sequences.case_mask[0])

    repeated_path = SuppliedAssemblyPaths(
        jnp.asarray((7,), dtype=jnp.int32),
        jnp.asarray(((0, 1, 0),), dtype=jnp.int32),
        jnp.asarray(((True, True, True),)),
        jnp.asarray(((False, True, False),)),
        jnp.asarray((True,)),
    )
    repeated = decode_supplied_paths(
        scored.graph,
        certificate,
        reads,
        repeated_path,
        output_capacity=8,
    )
    assert not bool(repeated.valid)
    assert int(repeated.status) == int(AssemblyStatus.REPEATED_NODE)

    cyclic_candidates = SuppliedOverlapCandidates(
        jnp.asarray((0, 1), dtype=jnp.int32),
        jnp.asarray((1, 0), dtype=jnp.int32),
        jnp.asarray((False, False)),
        jnp.asarray((False, False)),
        read_capacity=2,
    )
    cyclic_reads = _dna_batch(("AAA", "AAA"))
    cyclic_graph = score_supplied_overlaps(
        cyclic_reads, cyclic_candidates, minimum_overlap=1
    ).graph
    cyclic = certify_supplied_dag(
        cyclic_graph.relation,
        cyclic_graph.node_valid,
        jnp.asarray((0, 1), dtype=jnp.int32),
        jnp.asarray((True, True)),
        graph_id=cyclic_graph.graph_id,
    )
    assert not bool(cyclic.valid)
    assert int(cyclic.status) == int(AssemblyStatus.CYCLE)


def test_variation_graph_path_coordinates_use_native_relation():
    nodes = _dna_batch(("AC", "G", "TT"))
    built = build_supplied_variation_graph(
        nodes,
        jnp.asarray((0, 1), dtype=jnp.int32),
        jnp.asarray((1, 2), dtype=jnp.int32),
        jnp.asarray((True, True)),
        jnp.asarray((0, 1, 2), dtype=jnp.int32),
        jnp.asarray((True, True, True)),
    )
    assert bool(built.valid)
    assert isinstance(built.graph.relation, EdgeRelation)
    coordinates = index_graph_paths(
        built.graph,
        built.certificate,
        jnp.asarray((42,), dtype=jnp.int32),
        jnp.asarray(((0, 1, 2),), dtype=jnp.int32),
        jnp.asarray(((True, True, True),)),
        jnp.asarray((True,)),
    )
    located = locate_graph_path_positions(
        coordinates,
        jnp.asarray((0, 0, 0), dtype=jnp.int32),
        jnp.asarray((0, 2, 4), dtype=jnp.int32),
    )
    np.testing.assert_array_equal(np.asarray(located.node_indices), (0, 1, 2))
    np.testing.assert_array_equal(np.asarray(located.node_offsets), (0, 0, 1))


def test_canonical_kmers_report_hash_collisions_and_empty_reads():
    reads = _dna_batch(("ACGT", ""))
    counts = count_kmers(reads, KmerCountingPlan(DNA_IUPAC, 1, canonical=False))
    assert bool(counts.valid[0])
    assert not bool(counts.valid[1])
    assert int(counts.status[1]) == int(SketchStatus.EMPTY_READ)
    assert int(counts.distinct_counts[0]) == 4

    result = minhash_sketch(counts, MinHashPlan(4, hash_bits=1))
    assert bool(jnp.any(result.sketch.collision_counts[0] > 0))
    assert int(result.status[0]) == int(SketchStatus.HASH_COLLISION)
    assert int(result.status[1]) == int(SketchStatus.EMPTY_READ)

    reverse_pair = _dna_batch(("AC", "GT"))
    pair_counts = count_kmers(reverse_pair, KmerCountingPlan(DNA_IUPAC, 2))
    np.testing.assert_array_equal(
        np.asarray(pair_counts.counts[0]), np.asarray(pair_counts.counts[1])
    )


def test_taxonomy_version_merged_deleted_and_multi_assignment():
    taxonomy, version = _taxonomy()
    resolution = resolve_taxon_ids(
        taxonomy,
        jnp.asarray((2, 20, 30, 99), dtype=jnp.int32),
        database_version=version,
    )
    np.testing.assert_array_equal(
        np.asarray(resolution.status),
        (
            int(TaxonomyStatus.SUCCESS),
            int(TaxonomyStatus.SUCCESS),
            int(TaxonomyStatus.DELETED_TAXON),
            int(TaxonomyStatus.UNKNOWN_TAXON),
        ),
    )
    assert bool(resolution.was_merged[1])

    wrong_version = TaxonomyVersion("fixture", "2026-09", "different-content")
    mismatch = resolve_taxon_ids(
        taxonomy,
        jnp.asarray((2, 20), dtype=jnp.int32),
        database_version=wrong_version,
    )
    assert bool(jnp.all(mismatch.status == int(TaxonomyStatus.VERSION_MISMATCH)))
    assert not bool(jnp.any(mismatch.valid))

    candidates = SuppliedTaxonomicCandidates(
        jnp.asarray((10, 11), dtype=jnp.int32),
        jnp.asarray(((2, 3), (30, 99)), dtype=jnp.int32),
        jnp.asarray(((0.9, 0.85), (1.0, 0.5)), dtype=jnp.float32),
        jnp.asarray(((True, True), (True, True))),
        jnp.asarray((True, True)),
        version,
    )
    assignment = assign_taxonomy(taxonomy, candidates, relative_score_threshold=0.9)
    assert int(assignment.status[0]) == int(AssignmentStatus.ASSIGNED_AMBIGUOUS)
    assert int(jnp.sum(assignment.assigned_valid[0])) == 2
    assert np.isclose(float(jnp.sum(assignment.weights[0])), 1.0)
    assert int(assignment.status[1]) == int(AssignmentStatus.UNCLASSIFIED)
    assert float(assignment.unclassified_mass[1]) == 1.0

    mismatched_candidates = SuppliedTaxonomicCandidates(
        jnp.asarray((12,), dtype=jnp.int32),
        jnp.asarray(((2,),), dtype=jnp.int32),
        jnp.asarray(((1.0,),), dtype=jnp.float32),
        jnp.asarray(((True,),)),
        jnp.asarray((True,)),
        wrong_version,
    )
    mismatched = assign_taxonomy(taxonomy, mismatched_candidates)
    assert int(mismatched.status[0]) == int(AssignmentStatus.VERSION_MISMATCH)


def test_community_normalization_preserves_unclassified_mass():
    taxonomy, version = _taxonomy()
    candidates = SuppliedTaxonomicCandidates(
        jnp.asarray((10, 11), dtype=jnp.int32),
        jnp.asarray(((2, 3), (99, 99)), dtype=jnp.int32),
        jnp.asarray(((1.0, 1.0), (1.0, 0.0)), dtype=jnp.float32),
        jnp.asarray(((True, True), (True, False))),
        jnp.asarray((True, True)),
        version,
    )
    assignment = assign_taxonomy(taxonomy, candidates)
    community = estimate_community_abundance(
        taxonomy,
        assignment,
        read_weights=jnp.asarray((2.0, 1.0), dtype=jnp.float32),
    )
    assert bool(community.valid)
    total = jnp.sum(community.taxon_abundance) + community.unclassified_abundance
    assert np.isclose(float(total), 1.0)
    assert np.isclose(float(community.unclassified_abundance), 1.0 / 3.0)


def test_binning_completeness_contamination_and_functional_profile():
    supplied = supplied_contig_binning(
        jnp.asarray((100, 101, 102), dtype=jnp.int32),
        jnp.asarray((0, 0, 1), dtype=jnp.int32),
        jnp.asarray((True, True, True)),
        bin_capacity=2,
    )
    assert bool(supplied.valid)
    metrics = evaluate_binning_markers(
        supplied.binning,
        jnp.asarray(
            ((1, 1, 0, 0), (0, 1, 1, 0), (1, 1, 1, 1)),
            dtype=jnp.int32,
        ),
    )
    np.testing.assert_allclose(np.asarray(metrics.completeness), (0.75, 1.0))
    np.testing.assert_allclose(np.asarray(metrics.contamination), (0.25, 0.0))

    source = FeatureDictionary(
        jnp.asarray((1, 2, 3), dtype=jnp.int32),
        namespace="genes",
        version="1",
        species="fixture",
        reference="fixture",
        annotation="fixture",
    )
    functions = FeatureDictionary(
        jnp.asarray((10, 11), dtype=jnp.int32),
        namespace="functions",
        version="1",
        species="fixture",
        reference="fixture",
        annotation="fixture",
    )
    mapping = FeatureMapping(
        source,
        functions,
        jnp.asarray((0, 1, 1), dtype=jnp.int32),
        jnp.asarray((0, 0, 1), dtype=jnp.int32),
        confidence=jnp.asarray((1.0, 0.5, 0.5), dtype=jnp.float32),
    )
    profile = quantify_functional_profile(
        mapping, jnp.asarray(((1.0, 2.0, 1.0),), dtype=jnp.float32)
    )
    assert bool(profile.valid[0])
    np.testing.assert_allclose(np.asarray(profile.function_abundance[0]), (0.5, 0.25))
    np.testing.assert_allclose(np.asarray(profile.unannotated_abundance), (0.25,))
