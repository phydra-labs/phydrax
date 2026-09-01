#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._integration_guardrails import CoreAbstractionRegistry
from phydrax.bioinformatics import (
    foundation,
    genomics,
    interchange,
    models,
    phylogenetics,
    rna,
    sequence,
)
from phydrax.interchange import (
    AdapterError,
    AdapterReport,
    AdapterStatus,
    require_lossless,
)


def _feature_dictionary(*, reference: str = "GRCh38") -> foundation.FeatureDictionary:
    return foundation.FeatureDictionary(
        np.asarray((101, 102), dtype=np.int32),
        namespace="ensembl_gene",
        version="release-115",
        species="Homo sapiens",
        reference=reference,
        annotation="Ensembl 115",
        labels=("GENE1", "GENE2"),
    )


def _two_observation_lineage() -> foundation.BiospecimenLineage:
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
        study_id="identity-study",
    )


def _tiny_design_result() -> models.SequenceDesignResult:
    batch = sequence.encode_sequences(("AC",), sequence.DNA_IUPAC)
    probabilities = jnp.where(
        batch.valid_mask[..., None],
        jnp.full(
            (1, 2, sequence.DNA_IUPAC.size),
            1.0 / sequence.DNA_IUPAC.size,
        ),
        0.0,
    )
    distribution = sequence.SequenceDistribution(
        batch.record_ids,
        probabilities,
        batch.valid_mask,
        batch.case_mask,
        sequence.DNA_IUPAC,
    )

    def hard(codes: jax.Array, valid: jax.Array) -> jax.Array:
        return jnp.sum(
            (codes == sequence.DNA_IUPAC.code("A")) & valid[None, ...],
            axis=-1,
        ).astype(jnp.float32)

    def relaxed(values: jax.Array, valid: jax.Array) -> jax.Array:
        return jnp.sum(
            values[..., sequence.DNA_IUPAC.code("A")] * valid,
            axis=-1,
        )

    return models.solve_sequence_design(
        models.SequenceDesignProblem(
            distribution,
            hard,
            relaxed,
            sample_count=4,
            sample_capacity=4,
        ),
        key=jax.random.key(4),
    )


def test_scientific_method_claims_remain_orthogonal_across_domains() -> None:
    exact = rna.partition_function(
        jnp.asarray((0, 3), dtype=jnp.int32),
        rna.nussinov_energy_model(minimum_hairpin_length=0),
    ).method_contract
    approximate = genomics.somatic_likelihoods(
        genomics.TumorNormalAlleleCounts((5,), (10,), (0,), (10,), (True,)),
        genomics.SomaticCopyContext((1.0,), (0.0,), (2.0,), (2.0,), (1.0,), (1.0,)),
        genomics.SomaticPanelProvenance("tiny-panel", 10, (True,), (0.01,)),
        genomics.SomaticLikelihoodPlan(maximum_candidates=1),
    ).method_contract
    heuristic = sequence.ProgressiveMSAPlan(2, 2, 4).method_contract
    learned = models.AttentionSequenceEncoder(
        sequence.DNA_IUPAC,
        4,
        depth=1,
        num_heads=1,
        tokenizer_fingerprint="tiny-tokenizer",
        key=jax.random.key(1),
    )(sequence.encode_sequences(("AC",), sequence.DNA_IUPAC)).method_contract
    design = _tiny_design_result()
    relaxed = design.relaxed_method_contract

    assert exact.method_kind is foundation.MethodKind.EXACT_MODEL
    assert approximate.method_kind is foundation.MethodKind.APPROXIMATE_MODEL
    assert relaxed.method_kind is foundation.MethodKind.RELAXED_OBJECTIVE
    assert heuristic.method_kind is foundation.MethodKind.HEURISTIC
    assert learned.method_kind is foundation.MethodKind.LEARNED
    assert exact.execution_kind is foundation.ExecutionKind.EXACT_DISCRETE
    assert heuristic.execution_kind is foundation.ExecutionKind.EXACT_DISCRETE
    assert learned.execution_kind is foundation.ExecutionKind.FLOATING_POINT_DIRECT
    assert relaxed.differentiation_kind is foundation.DifferentiationKind.EXACT_AD
    assert design.method_contract.method_kind is foundation.MethodKind.HEURISTIC
    assert (
        len(
            {
                exact.contract_id,
                approximate.contract_id,
                relaxed.contract_id,
                heuristic.contract_id,
                learned.contract_id,
            }
        )
        == 5
    )


def test_padding_masks_are_consistent_between_sequence_models_and_phylogenetics() -> None:
    compact = sequence.encode_sequences(("AC",), sequence.DNA_IUPAC)
    pad = sequence.DNA_IUPAC.code(sequence.DNA_IUPAC.pad_symbol)
    padded = sequence.SequenceBatch(
        compact.record_ids,
        jnp.asarray(
            (
                (
                    sequence.DNA_IUPAC.code("A"),
                    sequence.DNA_IUPAC.code("C"),
                    pad,
                    pad,
                    pad,
                ),
            ),
            dtype=jnp.int32,
        ),
        jnp.asarray(((True, True, False, False, False),)),
        compact.case_mask,
        jnp.zeros((1, 5), dtype=bool),
        sequence.DNA_IUPAC,
    )
    encoder = models.AttentionSequenceEncoder(
        sequence.DNA_IUPAC,
        4,
        depth=1,
        num_heads=1,
        tokenizer_fingerprint="mask-contract",
        key=jax.random.key(8),
    )
    compact_result = encoder(compact)
    padded_result = encoder(padded)
    np.testing.assert_allclose(
        padded_result.token_embeddings[:, :2],
        compact_result.token_embeddings,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        padded_result.pooled_embedding,
        compact_result.pooled_embedding,
        atol=2.0e-6,
    )
    np.testing.assert_array_equal(
        padded_result.token_embeddings[:, 2:], jnp.zeros((1, 3, 4))
    )

    compact_tips = phylogenetics.tip_partials_from_sequence(compact)
    padded_tips = phylogenetics.tip_partials_from_sequence(padded)
    np.testing.assert_array_equal(padded_tips.tip_partials[:2], compact_tips.tip_partials)
    np.testing.assert_array_equal(
        padded_tips.site_mask, (True, True, False, False, False)
    )
    with pytest.raises(ValueError, match="Invalid positions"):
        sequence.SequenceBatch(
            padded.record_ids,
            padded.token_codes.at[0, 4].set(sequence.DNA_IUPAC.code("A")),
            padded.valid_mask,
            padded.case_mask,
            padded.soft_mask,
            sequence.DNA_IUPAC,
        )


def test_content_fingerprints_are_reproducible_and_content_sensitive() -> None:
    first_features = _feature_dictionary()
    repeated_features = _feature_dictionary()
    different_features = _feature_dictionary(reference="T2T-CHM13v2.0")
    assert first_features.dictionary_id == repeated_features.dictionary_id
    assert first_features.dictionary_id != different_features.dictionary_id

    first_reference = genomics.ReferenceGenome.from_sequences(
        {"chr1": "ACGT"}, assembly_id="assembly-a"
    )
    repeated_reference = genomics.ReferenceGenome.from_sequences(
        {"chr1": "ACGT"}, assembly_id="assembly-a"
    )
    changed_reference = genomics.ReferenceGenome.from_sequences(
        {"chr1": "ACGA"}, assembly_id="assembly-a"
    )
    assert first_reference.dictionary.digest == repeated_reference.dictionary.digest
    assert first_reference.dictionary.digest != changed_reference.dictionary.digest

    first_tree = sequence.GuideTree(
        jnp.asarray(((0, 1),), dtype=jnp.int32),
        jnp.asarray((0.2,)),
        2,
    )
    repeated_tree = sequence.GuideTree(
        jnp.asarray(((0, 1),), dtype=jnp.int32),
        jnp.asarray((0.2,)),
        2,
    )
    changed_tree = sequence.GuideTree(
        jnp.asarray(((0, 1),), dtype=jnp.int32),
        jnp.asarray((0.3,)),
        2,
    )
    assert first_tree.fingerprint == repeated_tree.fingerprint
    assert first_tree.fingerprint != changed_tree.fingerprint
    assert all(
        len(value) == 64
        for value in (
            first_features.dictionary_id,
            first_reference.dictionary.digest,
            first_tree.fingerprint,
        )
    )


def test_reference_and_feature_identities_cannot_be_inferred_from_labels() -> None:
    digest = genomics.reference_digest("ACGT")
    dictionary = genomics.ReferenceDictionary(
        (genomics.ReferenceContig("chr1", 4, digest, aliases=("1",)),),
        assembly_id="assembly-a",
    )
    genome = genomics.ReferenceGenome(dictionary, {"chr1": "ACGT"})
    assert genome.dictionary.resolve("1") == 0
    with pytest.raises(ValueError, match="digest"):
        genomics.ReferenceGenome(dictionary, {"chr1": "ACGA"})

    grch38 = _feature_dictionary(reference="GRCh38")
    chm13 = _feature_dictionary(reference="T2T-CHM13v2.0")
    assert grch38.labels == chm13.labels
    assert grch38.dictionary_id != chm13.dictionary_id
    grch38_mapping = foundation.FeatureMapping(grch38, grch38, (0, 1), (0, 1))
    cross_reference_mapping = foundation.FeatureMapping(grch38, chm13, (0, 1), (0, 1))
    assert grch38_mapping.source.dictionary_id == grch38.dictionary_id
    assert cross_reference_mapping.target.dictionary_id == chm13.dictionary_id
    assert grch38_mapping.mapping_id != cross_reference_mapping.mapping_id


def test_biospecimen_ancestry_and_coarse_split_leakage_are_observable() -> None:
    lineage = _two_observation_lineage()
    units = foundation.ExperimentalUnitPlan(
        lineage,
        np.asarray((4, 9), dtype=np.int32),
        np.asarray((0, 5), dtype=np.int32),
        np.asarray((0, 1), dtype=np.int32),
        block_group_ids=np.asarray((7, 7), dtype=np.int32),
    )
    assert units.lineage.lineage_id == lineage.lineage_id
    with pytest.raises(ValueError, match="ancestor"):
        foundation.ExperimentalUnitPlan(
            lineage,
            np.asarray((4, 9), dtype=np.int32),
            np.asarray((5, 0), dtype=np.int32),
            np.asarray((0, 1), dtype=np.int32),
        )

    grouping = foundation.BiologicalGrouping(
        np.asarray((100, 101, 102, 103), dtype=np.int32),
        np.asarray(((0, 0), (0, 1), (1, 2), (1, 3)), dtype=np.int32),
        group_names=("subject", "specimen"),
    )
    split = foundation.BiologicalSplit(
        grouping,
        np.asarray((0, 2), dtype=np.int32),
        np.asarray((), dtype=np.int32),
        np.asarray((1, 3), dtype=np.int32),
    )
    audit = foundation.LeakageAudit(split)
    assert bool(audit.has_leakage) and not bool(audit.passed)
    np.testing.assert_array_equal(audit.leaking_group_counts, (2, 0))
    np.testing.assert_array_equal(
        audit.leaking_observation_mask[:, 0], (True, True, True, True)
    )
    np.testing.assert_array_equal(
        audit.leaking_observation_mask[:, 1], (False, False, False, False)
    )


def test_sbml_adapter_loss_report_uses_canonical_report_contract() -> None:
    document = interchange.SBMLDocumentAST(
        3,
        2,
        interchange.SBMLModelAST(
            "toy",
            compartments=(interchange.SBMLCompartmentAST("cell"),),
            species=(
                interchange.SBMLSpeciesAST("A", "cell", elements=(("C", 1),)),
                interchange.SBMLSpeciesAST("B", "cell", elements=(("C", 1),)),
            ),
            reactions=(
                interchange.SBMLReactionAST(
                    "R",
                    reactants=(interchange.SBMLSpeciesReferenceAST("A"),),
                    products=(interchange.SBMLSpeciesReferenceAST("B"),),
                ),
            ),
            events=(interchange.SBMLEventAST("event-1", "A > 0", (("B", "1"),)),),
        ),
        source_id="tiny.xml",
    )
    validation = interchange.validate_sbml_document(document)
    report = validation.report

    assert isinstance(report, AdapterReport)
    assert report.status is AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC
    assert not report.valid
    assert len(report.losses) == 1
    assert report.losses[0].path == "model.events[0]"
    assert report.losses[0].direction == "import"
    assert report.losses[0].category == "unsupported"
    assert report.losses[0].changes_interpretation
    with pytest.raises(AdapterError) as error:
        require_lossless(report)
    assert error.value.status is AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC


def test_public_bioinformatics_facades_do_not_eagerly_import_optional_packages() -> None:
    script = """
import sys
from phydrax.bioinformatics import genomics, interchange, spectrometry

assert genomics.ReadBatch is not None
assert interchange.load_pysam_records is not None
assert interchange.read_pyteomics_mzml is not None
assert spectrometry.SpectrumBatch is not None
assert "pysam" not in sys.modules
assert "pyteomics" not in sys.modules
assert "pyteomics.mzml" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_canonical_owner_registry_resolves_every_bioinformatics_abstraction() -> None:
    expected = {
        "biological_sequence": "phydrax.bioinformatics.sequence.SequenceBatch",
        "biological_feature_dictionary": "phydrax.bioinformatics.foundation.FeatureDictionary",
        "biospecimen_lineage": "phydrax.bioinformatics.foundation.BiospecimenLineage",
        "genomic_coordinate": "phydrax.bioinformatics.genomics.IntervalSet",
        "biological_assay": "phydrax.bioinformatics.omics.CountAssay",
        "phylogenetic_tree": "phydrax.bioinformatics.phylogenetics.TreeTopology",
        "macromolecular_structure": "phydrax.bioinformatics.structure.MacromolecularStructure",
        "mass_spectrum": "phydrax.bioinformatics.spectrometry.SpectrumBatch",
        "biochemical_network": "phydrax.bioinformatics.systems.StoichiometricNetwork",
    }
    registry = CoreAbstractionRegistry()
    for kind, owner in expected.items():
        assert registry.owner(kind) == owner
        module_name, _, symbol_name = owner.rpartition(".")
        assert getattr(importlib.import_module(module_name), symbol_name) is not None

    repeated = CoreAbstractionRegistry(expected)
    assert repeated.registry_id == registry.registry_id
    with pytest.raises(ValueError, match="already belongs"):
        CoreAbstractionRegistry({"biological_sequence": "third_party.SequenceBatch"})
