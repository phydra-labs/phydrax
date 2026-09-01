#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.bioinformatics import (
    foundation,
    genomics,
    interchange,
    models,
    omics,
    phylogenetics,
    rna,
    sequence,
    spatial,
    structure,
    systems,
)


def _hard_gc_count(codes: jax.Array, valid: jax.Array) -> jax.Array:
    gc = (codes == sequence.DNA_IUPAC.code("G")) | (codes == sequence.DNA_IUPAC.code("C"))
    return jnp.sum(gc & valid[None, ...], axis=-1).astype(jnp.float32)


def _relaxed_gc_count(probabilities: jax.Array, valid: jax.Array) -> jax.Array:
    gc = (
        probabilities[..., sequence.DNA_IUPAC.code("G")]
        + probabilities[..., sequence.DNA_IUPAC.code("C")]
    )
    return jnp.sum(gc * valid, axis=-1)


def _two_donor_lineage() -> foundation.BiospecimenLineage:
    kinds = (
        foundation.BiospecimenLineage.SUBJECT,
        foundation.BiospecimenLineage.SPECIMEN,
        foundation.BiospecimenLineage.ALIQUOT,
        foundation.BiospecimenLineage.LIBRARY,
        foundation.BiospecimenLineage.OBSERVATION,
    ) * 2
    return foundation.BiospecimenLineage(
        np.arange(10, dtype=np.int32),
        np.asarray(kinds, dtype=np.int32),
        np.asarray((0, 1, 2, 3, 5, 6, 7, 8), dtype=np.int32),
        np.asarray((1, 2, 3, 4, 6, 7, 8, 9), dtype=np.int32),
        np.asarray((-1, -1, -1, -1, 0, -1, -1, -1, -1, 1), dtype=np.int32),
        np.asarray((-1, -1, -1, -1, 10, -1, -1, -1, -1, 11), dtype=np.int32),
        study_id="tiny-spatial-study",
    )


def test_fasta_sequence_batch_affine_and_pair_hmm_workflow() -> None:
    records = interchange.parse_fasta(">query\nACGT\n>target one-mismatch\nACGA\n")
    batch, lowering = interchange.lower_fasta(
        records,
        sequence.SequenceLoweringPlan(sequence.DNA_IUPAC, 2, 4),
        numeric_record_ids=jnp.asarray((101, 102), dtype=jnp.int32),
    )
    assert sequence.decode_sequences(batch) == ("ACGT", "ACGA")
    assert int(lowering.input_record_count) == 2
    assert int(lowering.retained_record_count) == 2
    assert int(lowering.record_overflow_count) == 0
    assert int(jnp.sum(lowering.truncated_symbol_counts)) == 0
    assert int(jnp.sum(lowering.mapped_invalid_counts)) == 0

    alignment = sequence.align_affine(
        batch.token_codes[0],
        batch.token_codes[1],
        sequence.nucleotide_substitution_table(match_score=2.0, mismatch_score=-3.0),
        sequence.AffineGapPenalties(-4.0, -1.0),
        sequence.AlignmentExecutionPlan.full(4, 4, traceback_capacity=8),
        query_mask=batch.valid_mask[0],
        target_mask=batch.valid_mask[1],
    )
    assert bool(alignment.valid)
    np.testing.assert_allclose(alignment.score, 3.0)
    assert int(alignment.alignment_length) == 4
    np.testing.assert_array_equal(
        np.asarray(alignment.operations[:4]), np.full((4,), sequence.MATCH)
    )

    negative_infinity = -jnp.inf
    pair_hmm = sequence.PairHMM(
        jnp.asarray((0.0, negative_infinity, negative_infinity)),
        jnp.asarray(
            (
                (0.0, negative_infinity, negative_infinity),
                (negative_infinity, 0.0, negative_infinity),
                (negative_infinity, negative_infinity, 0.0),
            )
        ),
        jnp.asarray((0.0, negative_infinity, negative_infinity)),
        jnp.where(jnp.eye(4, dtype=bool), 2.0, -2.0),
        jnp.zeros((4,)),
        jnp.zeros((4,)),
    )
    plan = sequence.PairHMMExecutionPlan.full(4, 4, traceback_capacity=8)
    query_probabilities = jax.nn.one_hot(batch.token_codes[0], 4)
    target_probabilities = jax.nn.one_hot(batch.token_codes[1], 4)
    mismatch = sequence.pair_hmm_forward_backward(
        pair_hmm,
        query_probabilities,
        target_probabilities,
        plan,
        left_mask=batch.valid_mask[0],
        right_mask=batch.valid_mask[1],
    )
    identical = sequence.pair_hmm_forward_backward(
        pair_hmm,
        query_probabilities,
        query_probabilities,
        plan,
        left_mask=batch.valid_mask[0],
        right_mask=batch.valid_mask[0],
    )
    assert bool(mismatch.valid) and bool(identical.valid)
    assert float(identical.log_partition) > float(mismatch.log_partition)
    assert int(mismatch.viterbi_length) == 4
    np.testing.assert_array_equal(
        np.asarray(mismatch.viterbi_states[:4]), np.full((4,), sequence.MATCH)
    )


def test_annotation_reference_transcript_translation_workflow() -> None:
    annotation = interchange.parse_gff3(
        "##gff-version 3\n"
        "chr1\ttiny\tgene\t1\t9\t.\t+\t.\tID=gene1\n"
        "chr1\ttiny\tmRNA\t1\t9\t.\t+\t.\tID=tx1;Parent=gene1\n"
        "chr1\ttiny\tCDS\t1\t9\t.\t+\t0\tID=cds1;Parent=tx1\n"
    )
    relations = interchange.gff3_parent_relations(annotation)
    assert relations.unresolved == ()
    assert relations.child_rows == (1, 2)
    assert relations.parent_rows == (0, 1)
    features = tuple(
        item for item in annotation if isinstance(item, interchange.GFF3FeatureLine)
    )
    transcript_feature = next(item for item in features if item.feature_type == "mRNA")
    cds_feature = next(item for item in features if item.feature_type == "CDS")
    genome = genomics.ReferenceGenome.from_sequences(
        {"chr1": "ATGAAATAG"}, assembly_id="tiny-reference"
    )
    transcript = genomics.TranscriptModel(
        1,
        genome.dictionary.resolve(transcript_feature.seqid),
        [transcript_feature.start],
        [transcript_feature.end],
        [True],
        strand=genomics.Strand.FORWARD,
    )
    coding_sequence = genomics.CDSModel(
        transcript,
        [cds_feature.start],
        [cds_feature.end],
        [True],
        [cds_feature.phase],
    )
    assembly = genomics.assemble_cds(genome, coding_sequence, capacity=cds_feature.length)
    translated = genomics.translate_cds(assembly)
    assert bool(assembly.valid) and bool(assembly.phase_consistent)
    assert sequence.decode_sequences(assembly.sequence) == ("ATGAAATAG",)
    assert bool(translated.valid) and bool(translated.exact)
    assert sequence.decode_sequences(translated.translation.sequences) == ("MK*",)


def test_read_evidence_genotype_posterior_vcf_workflow() -> None:
    states = genomics.enumerate_genotype_states(2, 2, 3)
    calls = jnp.asarray((0, 1, 0, 1, 0, 1, 0, 1), dtype=jnp.int32)
    evidence = genomics.local_allele_evidence_from_calls(
        calls, jnp.full((8,), 50.0), jnp.full((8,), 60.0), allele_count=2
    )
    likelihoods = genomics.genotype_likelihoods_from_reads(evidence, states)
    inference = genomics.infer_genotype(
        likelihoods,
        genomics.uniform_genotype_prior(states),
        states,
        min_posterior=0.9,
    )
    assert bool(inference.valid) and bool(inference.hard_call.called)
    np.testing.assert_array_equal(inference.hard_call.alleles, (0, 1))
    assert int(inference.evidence.depth) == 8
    assert float(inference.posterior.probabilities[1]) > 0.99

    normalized = genomics.normalize_small_variant(
        "GAC",
        1,
        "A",
        ("C",),
        reference_index=7,
        contig_index=0,
        max_alleles=2,
        max_allele_length=1,
    )
    assert bool(normalized.valid)
    pl = tuple(
        int(value)
        for value in np.asarray(
            genomics.genotype_likelihoods_to_pl(inference.likelihoods)
        )
    )
    sample = interchange.VCFSample(
        genotype=tuple(int(value) for value in np.asarray(inference.hard_call.alleles)),
        genotype_quality=float(inference.hard_call.genotype_quality),
        depth=int(inference.hard_call.depth),
        allele_depths=(4, 4),
        phred_likelihoods=pl,
    )
    record = interchange.vcf_record_from_small_variant(
        normalized.site,
        "chr1",
        quality=50.0,
        filters=("PASS",),
        format_keys=("GT", "GQ", "DP", "AD", "PL"),
        samples=(sample,),
    )
    written = interchange.write_vcf(
        interchange.VCFHeader(sample_names=("sample-1",)), (record,), max_records=1
    )
    reparsed = interchange.parse_vcf(
        written.text, max_records=1, max_samples=1, max_alleles=2
    )
    assert written.valid and reparsed.valid
    assert reparsed.records[0].position == 2
    assert reparsed.records[0].samples[0].genotype == (0, 1)
    assert reparsed.records[0].samples[0].phred_likelihoods == pl


def test_multiple_alignment_fixed_tree_likelihood_workflow() -> None:
    unaligned = sequence.encode_sequences(("AC", "AG"), sequence.DNA_IUPAC)
    alignment = sequence.progressive_multiple_alignment(
        unaligned, sequence.ProgressiveMSAPlan(2, 2, 4)
    )
    assert bool(alignment.valid)
    assert int(alignment.alignment_length) == 2
    assert alignment.method_contract.method_kind is foundation.MethodKind.HEURISTIC
    tip_data = phylogenetics.tip_partials_from_sequence(alignment.alignment)
    topology = phylogenetics.tree_topology(jnp.asarray((2, 2, -1)))
    model = phylogenetics.jc69(dtype=jnp.float64)
    likelihood = phylogenetics.felsenstein_pruning(
        topology,
        tip_data.tip_partials,
        jnp.asarray((0.1, 0.1, 0.0), dtype=jnp.float64),
        (phylogenetics.LikelihoodPartition(jnp.ones_like(tip_data.site_mask), model),),
        pattern_weights=tip_data.site_mask.astype(jnp.float64),
    )
    assert bool(tip_data.valid) and bool(likelihood.valid)
    assert float(likelihood.pattern_log_likelihood[0]) > float(
        likelihood.pattern_log_likelihood[1]
    )
    assert likelihood.method_contract.method_kind is foundation.MethodKind.EXACT_MODEL


def test_count_assay_pseudobulk_nb_wald_bh_workflow() -> None:
    cells = omics.CountAssay(
        jnp.asarray(
            (
                (5, 10),
                (5, 10),
                (5, 10),
                (6, 11),
                (4, 9),
                (5, 10),
                (5, 10),
                (5, 10),
                (5, 10),
                (6, 11),
                (4, 9),
                (5, 10),
            )
        )
    )
    pseudobulk = omics.pseudobulk_counts(
        cells, jnp.repeat(jnp.arange(6, dtype=jnp.int32), 2), num_units=6
    )
    np.testing.assert_array_equal(
        pseudobulk.assay.dense_values,
        jnp.asarray(((10, 20), (11, 21), (9, 19)) * 2),
    )
    np.testing.assert_array_equal(pseudobulk.contributing_cells, jnp.full((6,), 2))
    design = omics.build_experimental_design(
        jnp.asarray((0, 0, 0, 1, 1, 1), dtype=jnp.int32), num_conditions=2
    )
    fit = omics.fit_negative_binomial_glm(
        pseudobulk.assay, design, jnp.asarray((0.1, 0.1)), maximum_steps=100
    )
    wald = omics.wald_test(fit, omics.pairwise_condition_contrast(design, 1, 0))
    adjusted = omics.benjamini_hochberg(wald.p_value, wald.valid)
    assert bool(jnp.all(fit.valid)) and bool(jnp.all(wald.valid))
    np.testing.assert_allclose(wald.log2_fold_change, 0.0, atol=2.0e-5)
    assert bool(jnp.all(wald.p_value > 0.99))
    assert bool(adjusted.valid)
    assert int(adjusted.family_size) == 2
    assert bool(jnp.all(adjusted.adjusted_p_values > 0.99))


def test_spatial_assay_neighbor_statistic_workflow() -> None:
    lineage = _two_donor_lineage()
    frame = spatial.SpatialFrame("tiny-slide", ("x", "y"), spatial.MICROMETRE)
    records = (
        spatial.SpatialSampleRecord(
            "sample-0", "specimen-0", "donor-0", "section-0", frame, lineage
        ),
        spatial.SpatialSampleRecord(
            "sample-1", "specimen-1", "donor-1", "section-1", frame, lineage
        ),
    )
    assay = spatial.SpatialAssay(
        records,
        spatial.SpatialAssayData(
            spatial.SpatialCoordinates(
                jnp.asarray(((0.0, 0.0), (1.0, 0.0), (10.0, 0.0), (11.0, 0.0))),
                frame,
            ),
            jnp.asarray(((0.0,), (1.0,), (10.0,), (11.0,))),
            jnp.asarray((0, 0, 1, 1), dtype=jnp.int32),
        ),
    )
    graph = spatial.assay_neighbor_graph(
        assay, spatial.SpatialNeighborPlan("knn", capacity=1, k=1)
    )
    statistic = spatial.assay_autocorrelation_test(
        assay, 0, graph, jax.random.key(9), permutations=31
    )
    assert bool(graph.valid) and bool(statistic.valid)
    np.testing.assert_array_equal(graph.indices[:, 0], (1, 0, 3, 2))
    assert int(graph.evidence.section_count) == 2
    assert int(statistic.evidence.donor_count) == 2
    assert float(statistic.statistic) > 0.9
    assert 0.0 < float(statistic.p_value) <= 1.0


def test_mmcif_host_record_structure_atomistic_lowering_workflow() -> None:
    record = structure.MacromolecularRecord(
        "tiny",
        (
            structure.EntityRecord(
                "1",
                structure.EntityKind.POLYMER,
                polymer_kind=structure.PolymerKind.PROTEIN_L,
                sequence_components=("ALA",),
            ),
        ),
        (structure.ChainRecord("A", "A", "1"),),
        (structure.ResidueRecord(0, "ALA", "ALA", 1, 1, entity_sequence_index=1),),
        (structure.AtomRecord("1", 0, 1, "CA", "CA", "C", 6, (1.0, 2.0, 3.0)),),
        (
            structure.ChemicalComponent(
                "ALA",
                "alanine",
                "peptide linking",
                (structure.ChemicalComponentAtom("CA", "C", 6),),
            ),
        ),
    )
    restored = interchange.parse_mmcif(interchange.dumps_mmcif(record))
    lowered = structure.lower_macromolecular_record(
        restored, structure.StructureLoweringPlan.for_record(restored)
    )
    assert restored.record_id == record.record_id
    assert bool(lowered.valid)
    assert lowered.structure is not None
    assert lowered.atomistic_structure is not None
    assert lowered.atomistic_topology is not None
    np.testing.assert_array_equal(lowered.structure.atomic_numbers, (6,))
    np.testing.assert_allclose(lowered.atomistic_structure.positions[0], (1.0, 2.0, 3.0))
    np.testing.assert_allclose(lowered.atomistic_structure.masses[0], 12.011)
    assert lowered.structure.source_record_id == restored.record_id


def test_rna_sequence_partition_marginals_workflow() -> None:
    encoded = sequence.encode_sequences(("AUGC",), sequence.RNA_IUPAC)
    codes = encoded.token_codes[0, : int(encoded.lengths[0])]
    model = rna.nussinov_energy_model(
        pair_energy=-1.3,
        wobble_energy=-0.4,
        unpaired_energy=0.2,
        minimum_hairpin_length=0,
    )
    partition = rna.partition_function(codes, model)
    assert bool(partition.valid)
    assert bool(jnp.isfinite(partition.log_partition))
    assert float(partition.pair_marginals[0, 1]) > 0.0
    np.testing.assert_allclose(partition.pair_marginals[0, 2], 0.0)
    np.testing.assert_allclose(
        partition.unpaired_marginals + jnp.sum(partition.pair_marginals, axis=1),
        1.0,
        atol=2.0e-6,
    )
    assert partition.model_id == model.model_id
    assert partition.method_contract.method_kind is foundation.MethodKind.EXACT_MODEL


def test_stoichiometric_network_fba_workflow() -> None:
    compartment = systems.Compartment("cell")
    species = (
        systems.Species("A", "cell", composition=systems.ChemicalComposition({"C": 1})),
    )
    reactions = (
        systems.Reaction(
            "source",
            ("A",),
            jnp.asarray((1.0,)),
            lower_bound=0.0,
            upper_bound=10.0,
            exchange=True,
        ),
        systems.Reaction(
            "biomass",
            ("A",),
            jnp.asarray((-1.0,)),
            lower_bound=0.0,
            upper_bound=10.0,
            objective_coefficient=1.0,
            exchange=True,
        ),
    )
    network = systems.StoichiometricNetwork((compartment,), species, reactions)
    result = systems.flux_balance_analysis(network)
    assert bool(result.successful)
    np.testing.assert_allclose(result.fluxes, (10.0, 10.0), atol=1.0e-5)
    np.testing.assert_allclose(result.objective_value, 10.0, atol=1.0e-5)
    np.testing.assert_allclose(
        network.steady_state_matrix @ result.fluxes, 0.0, atol=1.0e-6
    )
    assert result.network_id == network.network_id
    assert result.method_contract.method_kind is foundation.MethodKind.EXACT_MODEL


def test_sequence_model_artifact_inverse_design_workflow() -> None:
    batch = sequence.encode_sequences(("ACG",), sequence.DNA_IUPAC)
    tokenizer = models.TokenizerProvenance("dna-iupac", "a" * 64, sequence.DNA_IUPAC)
    encoder = models.AttentionSequenceEncoder(
        sequence.DNA_IUPAC,
        4,
        depth=1,
        num_heads=1,
        tokenizer_fingerprint=tokenizer.fingerprint,
        key=jax.random.key(1),
    )
    manifest = models.FoundationModelManifest(
        "tiny-dna-encoder",
        "attention-sequence-encoder",
        "c" * 64,
        models.native_model_parameter_sha256(encoder),
        models.native_model_structure_fingerprint(encoder),
        tokenizer,
        models.LicenseProvenance(
            "Apache-2.0",
            "b" * 64,
            status="verified",
            inference_allowed=True,
            adaptation_allowed=True,
            redistribution_allowed=True,
        ),
        models.PretrainingOverlapProvenance(
            "no-detected-overlap",
            evaluation_split_id="homology-test",
            homology_partition_id="identity-30",
            search_method="global-identity",
            identity_threshold=0.3,
            maximum_identity=0.1,
        ),
    )
    bound = models.bind_native_foundation_model(
        encoder,
        manifest,
        artifact_sha256=manifest.artifact_sha256,
        tokenizer_fingerprint=tokenizer.fingerprint,
        alphabet_fingerprint=sequence.DNA_IUPAC.fingerprint,
        evaluation_split_id="homology-test",
        homology_partition_id="identity-30",
    )
    encoded = bound.export_callable()(batch)
    prediction = models.TokenPredictionHead(
        4,
        sequence.DNA_IUPAC,
        tokenizer_fingerprint=tokenizer.fingerprint,
        key=jax.random.key(2),
    )(encoded)
    probabilities = jnp.where(
        encoded.valid_mask[..., None],
        jax.nn.softmax(prediction.logits, axis=-1),
        0.0,
    )
    distribution = sequence.SequenceDistribution(
        batch.record_ids,
        probabilities,
        encoded.valid_mask,
        batch.case_mask,
        sequence.DNA_IUPAC,
    )
    fixed = models.FixedTokenConstraint(
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray((sequence.DNA_IUPAC.code("A"),), dtype=jnp.int32),
        case_capacity=1,
        length_capacity=3,
        alphabet_size=sequence.DNA_IUPAC.size,
    )
    canonical_codes = jnp.asarray(
        tuple(sequence.DNA_IUPAC.code(symbol) for symbol in "ACGT")
    )
    allowed = jnp.broadcast_to(
        jnp.isin(jnp.arange(sequence.DNA_IUPAC.size), canonical_codes),
        probabilities.shape,
    )
    design = models.solve_sequence_design(
        models.SequenceDesignProblem(
            distribution,
            _hard_gc_count,
            _relaxed_gc_count,
            constraints=(fixed, models.AllowedTokenConstraint(allowed)),
            sample_count=8,
            sample_capacity=8,
        ),
        key=jax.random.key(3),
    )
    assert bool(bound.binding.valid)
    assert bound.binding.manifest.fingerprint == manifest.fingerprint
    assert encoded.method_contract.method_kind is foundation.MethodKind.LEARNED
    assert bool(design.valid) and bool(jnp.all(design.constraint_satisfied))
    assert int(design.selected_codes[0, 0]) == sequence.DNA_IUPAC.code("A")
    assert bool(
        jnp.all(jnp.isin(design.selected_codes[encoded.valid_mask], canonical_codes))
    )
    assert (
        design.relaxed_method_contract.method_kind
        is foundation.MethodKind.RELAXED_OBJECTIVE
    )
    assert design.method_contract.method_kind is foundation.MethodKind.HEURISTIC
