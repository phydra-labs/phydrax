#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import numpy as np
import pytest

from phydrax.bioinformatics.foundation import (
    BioinformaticsMethodContract,
    BiologicalGrouping,
    BiologicalSplit,
    BiospecimenLineage,
    DifferentiationKind,
    ExchangeabilityPlan,
    ExecutionKind,
    ExperimentalUnitPlan,
    FeatureDictionary,
    FeatureMapping,
    LeakageAudit,
    MethodKind,
    OntologyGraph,
    OutputKind,
)
from phydrax.sparse import EdgeRelation


def _dictionary(feature_ids, *, labels=None):
    return FeatureDictionary(
        np.asarray(feature_ids, dtype=np.int32),
        namespace="ensembl_gene",
        version="release-115",
        species="Homo sapiens",
        reference="GRCh38.p14",
        annotation="Ensembl 115",
        labels=labels,
    )


def _lineage(*, parent_indices=None, child_indices=None):
    parents = np.asarray(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] if parent_indices is None else parent_indices,
        dtype=np.int32,
    )
    children = np.asarray(
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11] if child_indices is None else child_indices,
        dtype=np.int32,
    )
    return BiospecimenLineage(
        np.arange(100, 112, dtype=np.int32),
        np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=np.int32),
        parents,
        children,
        np.asarray([-1] * 10 + [20, 21], dtype=np.int32),
        np.asarray([-1] * 10 + [30, 31], dtype=np.int32),
        study_id="study-alpha",
    )


def test_method_contract_separates_scientific_numerical_and_derivative_claims():
    contract = BioinformaticsMethodContract(
        "banded_global_alignment",
        MethodKind.APPROXIMATE_MODEL,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.NONE,
        OutputKind.SEQUENCE,
        conditioning_statement="Conditioned on a fixed substitution and gap model.",
        truncation_statement="Candidates outside the fixed alignment band are excluded.",
        capacity_semantics="Input and traceback capacities are fixed before execution.",
        assumptions=("The supplied alphabet is fixed.",),
        nondifferentiable_outputs=("traceback",),
        input_dtype="int32",
        output_dtype="int32",
    )
    repeated = BioinformaticsMethodContract(
        "banded_global_alignment",
        MethodKind.APPROXIMATE_MODEL,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.NONE,
        OutputKind.SEQUENCE,
        conditioning_statement="Conditioned on a fixed substitution and gap model.",
        truncation_statement="Candidates outside the fixed alignment band are excluded.",
        capacity_semantics="Input and traceback capacities are fixed before execution.",
        assumptions=("The supplied alphabet is fixed.",),
        nondifferentiable_outputs=("traceback",),
        input_dtype="int32",
        output_dtype="int32",
    )

    assert contract.method_kind is MethodKind.APPROXIMATE_MODEL
    assert contract.execution_kind is ExecutionKind.EXACT_DISCRETE
    assert contract.differentiation_kind is DifferentiationKind.NONE
    assert contract.output_kind is OutputKind.SEQUENCE
    assert contract.contract_id == repeated.contract_id
    assert len(contract.contract_id) == 64
    with pytest.raises(AttributeError):
        contract.method_name = "local_alignment"


def test_method_contract_rejects_negative_tolerances():
    with pytest.raises(ValueError, match="non-negative"):
        BioinformaticsMethodContract(
            "iterative_latent_model",
            MethodKind.EXACT_MODEL,
            ExecutionKind.ITERATIVE_TOLERANCE,
            DifferentiationKind.IMPLICIT,
            OutputKind.ARRAY,
            conditioning_statement="Conditioned on the supplied design matrix.",
            truncation_statement="No state-space truncation is applied.",
            capacity_semantics="Latent capacity is fixed by the input shape.",
            compute_dtype="float32",
            relative_tolerance=-1.0,
        )


def test_feature_dictionary_and_mapping_validate_shapes_and_one_to_many_routes():
    source = _dictionary([101, 102], labels=("A", "B"))
    target = _dictionary([201, 202, 203], labels=("X", "Y", "Z"))
    mapping = FeatureMapping(
        source,
        target,
        np.asarray([0, 0, 1], dtype=np.int32),
        np.asarray([0, 1, 2], dtype=np.int32),
        confidence=np.asarray([1.0, 0.75, 0.9], dtype=np.float32),
    )
    repeated = FeatureMapping(
        source,
        target,
        np.asarray([0, 0, 1], dtype=np.int32),
        np.asarray([0, 1, 2], dtype=np.int32),
        confidence=np.asarray([1.0, 0.75, 0.9], dtype=np.float32),
    )

    assert isinstance(mapping.relation, EdgeRelation)
    assert np.array_equal(mapping.relation.source_indices, [0, 0, 1])
    assert mapping.mapping_id == repeated.mapping_id
    assert all(not isinstance(leaf, str) for leaf in jax.tree_util.tree_leaves(mapping))

    with pytest.raises(ValueError, match="shape"):
        FeatureDictionary(
            np.asarray([1, 2], dtype=np.int32),
            namespace="gene",
            version="one",
            species="Homo sapiens",
            reference="GRCh38",
            annotation="test",
            active=np.asarray([[True, True]]),
        )
    with pytest.raises(ValueError, match="shapes must match"):
        FeatureMapping(
            source,
            target,
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
        )


def test_ontology_reuses_edge_relation_and_rejects_cycles():
    features = _dictionary([1, 2, 3])
    ontology = OntologyGraph(
        features,
        np.asarray([1, 2], dtype=np.int32),
        np.asarray([0, 1], dtype=np.int32),
        relation_codes=np.asarray([0, 0], dtype=np.int32),
    )

    assert isinstance(ontology.relation, EdgeRelation)
    assert ontology.relation_names == ("is_a",)
    with pytest.raises(ValueError, match="acyclic"):
        OntologyGraph(
            features,
            np.asarray([0, 1, 2], dtype=np.int32),
            np.asarray([1, 2, 0], dtype=np.int32),
        )


def test_lineage_rejects_cycles_and_invalid_indices():
    lineage = _lineage()
    assert lineage.entity_count == 12
    assert len(lineage.lineage_id) == 64

    with pytest.raises(ValueError, match="acyclic"):
        _lineage(
            parent_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            child_indices=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0],
        )
    with pytest.raises(ValueError, match="within the entity space"):
        _lineage(
            parent_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            child_indices=[2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
        )


def test_experimental_units_and_exchangeability_follow_lineage_ancestry():
    lineage = _lineage()
    units = ExperimentalUnitPlan(
        lineage,
        np.asarray([10, 11], dtype=np.int32),
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([0, 1], dtype=np.int32),
        block_group_ids=np.asarray([0, 0], dtype=np.int32),
    )
    exchangeability = ExchangeabilityPlan(
        units,
        np.asarray([0, 0], dtype=np.int32),
    )

    assert len(units.plan_id) == 64
    assert bool(exchangeability.permutation_mask.all())
    assert len(exchangeability.exchangeability_id) == 64
    with pytest.raises(ValueError, match="ancestor"):
        ExperimentalUnitPlan(
            lineage,
            np.asarray([10], dtype=np.int32),
            np.asarray([1], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
        )


def test_biological_grouping_rejects_nontransitive_levels():
    with pytest.raises(ValueError, match="transitively nested"):
        BiologicalGrouping(
            np.asarray([10, 11], dtype=np.int32),
            np.asarray([[0, 7], [1, 7]], dtype=np.int32),
            group_names=("subject", "specimen"),
        )


def test_biological_split_rejects_overlap_and_audit_finds_coarse_leakage():
    grouping = BiologicalGrouping(
        np.asarray([10, 11, 12], dtype=np.int32),
        np.asarray([[0, 0], [0, 1], [1, 2]], dtype=np.int32),
        group_names=("subject", "specimen"),
    )
    with pytest.raises(ValueError, match="disjoint"):
        BiologicalSplit(
            grouping,
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([1], dtype=np.int32),
            np.asarray([2], dtype=np.int32),
        )

    split = BiologicalSplit(
        grouping,
        np.asarray([0], dtype=np.int32),
        np.asarray([2], dtype=np.int32),
        np.asarray([1], dtype=np.int32),
    )
    audit = LeakageAudit(split)

    assert np.array_equal(audit.leaking_group_counts, [1, 0])
    assert np.array_equal(
        audit.leaking_observation_mask,
        np.asarray([[True, False], [True, False], [False, False]]),
    )
    assert bool(audit.has_leakage)
    assert not bool(audit.passed)
    assert len(grouping.grouping_id) == 64
    assert len(split.split_id) == 64
    assert len(audit.audit_id) == 64
