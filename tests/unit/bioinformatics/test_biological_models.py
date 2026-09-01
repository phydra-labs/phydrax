#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax.bioinformatics.models._design import (
    AllowedTokenConstraint,
    FixedTokenConstraint,
    SequenceDesignProblem,
    solve_sequence_design,
)
from phydrax.bioinformatics.models._foundation import (
    bind_low_rank_foundation_adapter,
    bind_native_foundation_model,
    ExternalFoundationRuntime,
    FoundationBindingStatus,
    FoundationModelManifest,
    LicenseProvenance,
    low_rank_adapter_parameter_sha256,
    LowRankAdapterProvenance,
    native_model_parameter_sha256,
    native_model_structure_fingerprint,
    PretrainingOverlapProvenance,
    TokenizerProvenance,
)
from phydrax.bioinformatics.models._objectives import (
    CausalTokenObjective,
    ContactObjective,
    MaskedTokenObjective,
    PairObjective,
    PairPrediction,
    TokenLabelObjective,
    TokenLabelPrediction,
    TokenPrediction,
)
from phydrax.bioinformatics.models._sequence import (
    AttentionSequenceEncoder,
    RecurrentSequenceEncoder,
    TokenPredictionHead,
)
from phydrax.bioinformatics.models._structure import (
    EquivariantStructureEncoder,
    MacromolecularBatch,
)
from phydrax.bioinformatics.sequence import DNA_IUPAC, SequenceBatch, SequenceDistribution
from phydrax.nn.layers import Linear
from phydrax.nn.parameters import adapt_low_rank, LowRankSpec


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64


def _batch() -> SequenceBatch:
    pad = DNA_IUPAC.code(DNA_IUPAC.pad_symbol)
    return SequenceBatch(
        jnp.asarray((11, 12), dtype=jnp.int32),
        jnp.asarray(
            (
                (DNA_IUPAC.code("A"), DNA_IUPAC.code("C"), DNA_IUPAC.code("G")),
                (DNA_IUPAC.code("T"), pad, pad),
            ),
            dtype=jnp.int32,
        ),
        jnp.asarray(((True, True, True), (True, False, False))),
        jnp.asarray((True, True)),
        jnp.zeros((2, 3), dtype=bool),
        DNA_IUPAC,
    )


def _tokenizer() -> TokenizerProvenance:
    return TokenizerProvenance("iupac-code", _SHA_A, DNA_IUPAC)


def _license(
    *, status: str = "verified", inference: bool = True, adaptation: bool = True
) -> LicenseProvenance:
    return LicenseProvenance(
        "Apache-2.0",
        _SHA_B,
        status=status,
        inference_allowed=inference,
        adaptation_allowed=adaptation,
        redistribution_allowed=True,
    )


def _overlap(assessment: str = "no-detected-overlap") -> PretrainingOverlapProvenance:
    return PretrainingOverlapProvenance(
        assessment,
        evaluation_split_id="homology-test-split",
        homology_partition_id="mmseqs-30-partition",
        search_method="" if assessment == "unknown" else "global-identity",
        identity_threshold=0.3,
        maximum_identity=None if assessment == "unknown" else 0.29,
    )


def _manifest(
    model,
    *,
    artifact_hash: str = _SHA_C,
    license_provenance: LicenseProvenance | None = None,
    overlap: PretrainingOverlapProvenance | None = None,
    base_hash: str | None = None,
) -> FoundationModelManifest:
    return FoundationModelManifest(
        "tiny-dna-model",
        "linear-token-head",
        artifact_hash,
        native_model_parameter_sha256(model),
        native_model_structure_fingerprint(model),
        _tokenizer(),
        _license() if license_provenance is None else license_provenance,
        _overlap() if overlap is None else overlap,
        base_model_sha256=base_hash,
    )


def _bind(model, manifest: FoundationModelManifest, **overrides):
    arguments = {
        "artifact_sha256": manifest.artifact_sha256,
        "tokenizer_fingerprint": manifest.tokenizer.fingerprint,
        "alphabet_fingerprint": DNA_IUPAC.fingerprint,
        "evaluation_split_id": "homology-test-split",
        "homology_partition_id": "mmseqs-30-partition",
        "base_model_sha256": manifest.base_model_sha256,
    }
    arguments.update(overrides)
    return bind_native_foundation_model(model, manifest, **arguments)


def _hard_a_count(codes, valid):
    return jnp.sum((codes == DNA_IUPAC.code("A")) & valid[None, ...], axis=-1).astype(
        jnp.float32
    )


def _relaxed_a_count(probabilities, valid):
    return jnp.sum(probabilities[..., DNA_IUPAC.code("A")] * valid, axis=-1)


def test_sequence_wrappers_zero_padding_and_use_deterministic_keys() -> None:
    batch = _batch()
    key = jr.PRNGKey(4)
    first = AttentionSequenceEncoder(
        DNA_IUPAC,
        8,
        depth=2,
        num_heads=2,
        tokenizer_fingerprint="tokenizer-v1",
        key=key,
    )
    second = AttentionSequenceEncoder(
        DNA_IUPAC,
        8,
        depth=2,
        num_heads=2,
        tokenizer_fingerprint="tokenizer-v1",
        key=key,
    )
    leaves_first = [leaf for leaf in jax.tree.leaves(first) if eqx.is_array(leaf)]
    leaves_second = [leaf for leaf in jax.tree.leaves(second) if eqx.is_array(leaf)]
    assert all(
        np.array_equal(a, b) for a, b in zip(leaves_first, leaves_second, strict=True)
    )

    encoded = first(batch)
    assert bool(encoded.valid)
    np.testing.assert_array_equal(encoded.token_embeddings[1, 1:], 0.0)
    recurrent = RecurrentSequenceEncoder(
        DNA_IUPAC,
        8,
        8,
        tokenizer_fingerprint="tokenizer-v1",
        key=jr.PRNGKey(5),
    )(batch)
    np.testing.assert_array_equal(recurrent.token_embeddings[1, 1:], 0.0)


def test_tokenization_mismatch_is_rejected_and_masked_loss_excludes_padding() -> None:
    batch = _batch()
    encoded = AttentionSequenceEncoder(
        DNA_IUPAC,
        6,
        num_heads=2,
        tokenizer_fingerprint="tokenizer-v1",
        key=jr.PRNGKey(0),
    )(batch)
    prediction = TokenPredictionHead(
        6,
        DNA_IUPAC,
        tokenizer_fingerprint="tokenizer-v1",
        key=jr.PRNGKey(1),
    )(encoded)
    selected = jnp.asarray(((False, True, False), (True, True, True)))
    result = MaskedTokenObjective("tokenizer-v1")(prediction, batch, selected)
    assert int(result.evidence[0]) == 2
    assert bool(result.valid)
    with pytest.raises(ValueError, match="tokenizer"):
        MaskedTokenObjective("other-tokenizer")(prediction, batch, selected)


def test_all_typed_objectives_obey_their_support_masks() -> None:
    batch = _batch()
    vocab = DNA_IUPAC.size
    contract = AttentionSequenceEncoder(
        DNA_IUPAC,
        4,
        num_heads=1,
        tokenizer_fingerprint="tokenizer-v1",
        key=jr.PRNGKey(91),
    )(batch).method_contract
    token_prediction = TokenPrediction(
        jnp.zeros((2, 3, vocab)),
        batch.valid_mask,
        alphabet_fingerprint=DNA_IUPAC.fingerprint,
        tokenizer_fingerprint="tokenizer-v1",
        method_contract=contract,
    )
    causal = CausalTokenObjective("tokenizer-v1")(token_prediction, batch)
    assert int(causal.evidence[0]) == 2

    labels = jnp.zeros((2, 3), dtype=jnp.int32)
    label_prediction = TokenLabelPrediction(
        jnp.zeros((2, 3, 2)),
        batch.valid_mask,
        label_space_id="secondary-structure",
        method_contract=token_prediction.method_contract,
    )
    label_result = TokenLabelObjective("secondary-structure")(
        label_prediction, labels, batch.valid_mask
    )
    assert int(label_result.evidence[0]) == 4

    pair_mask = batch.valid_mask[:, :, None] & batch.valid_mask[:, None, :]
    pair_prediction = PairPrediction(
        jnp.zeros((2, 3, 3, 2)),
        pair_mask,
        pair_space_id="distance-bin",
        symmetric=True,
        method_contract=token_prediction.method_contract,
    )
    pair_result = PairObjective("distance-bin")(
        pair_prediction, jnp.zeros((2, 3, 3), dtype=jnp.int32)
    )
    assert int(pair_result.evidence[0]) == 10

    contact_prediction = PairPrediction(
        jnp.zeros((2, 3, 3)),
        pair_mask,
        pair_space_id="contact",
        symmetric=True,
        method_contract=token_prediction.method_contract,
    )
    contact_result = ContactObjective()(contact_prediction, jnp.zeros((2, 3, 3)))
    assert int(contact_result.evidence[0]) == 3


def test_foundation_binding_checks_hash_model_tokenizer_license_base_and_homology_split() -> (
    None
):
    model = Linear(in_size=3, out_size=2, rwf=False, key=jr.PRNGKey(0))
    manifest = _manifest(model)
    bound = _bind(model, manifest)
    assert bool(bound.binding.valid)
    assert int(bound.binding.status) == FoundationBindingStatus.SUCCESS

    with pytest.raises(ValueError, match="artifact hash"):
        _bind(model, manifest, artifact_sha256=_SHA_D)
    different_model = Linear(in_size=3, out_size=2, rwf=False, key=jr.PRNGKey(1))
    with pytest.raises(ValueError, match="parameter hash"):
        _bind(different_model, manifest)
    with pytest.raises(ValueError, match="Tokenizer"):
        _bind(model, manifest, tokenizer_fingerprint="different-tokenizer")
    with pytest.raises(ValueError, match="Evaluation split"):
        _bind(model, manifest, evaluation_split_id="random-split")
    with pytest.raises(ValueError, match="Homology partition"):
        _bind(model, manifest, homology_partition_id="different-partition")

    unknown_license = _manifest(
        model,
        license_provenance=_license(status="unknown", inference=True),
    )
    with pytest.raises(PermissionError, match="license"):
        _bind(model, unknown_license)

    based = _manifest(model, base_hash=_SHA_A)
    with pytest.raises(ValueError, match="Base model"):
        _bind(model, based, base_model_sha256=_SHA_B)


def test_unknown_pretraining_overlap_remains_observable_without_invalidating_binding() -> (
    None
):
    model = Linear(in_size=3, out_size=2, rwf=False, key=jr.PRNGKey(0))
    manifest = _manifest(model, overlap=_overlap("unknown"))
    bound = _bind(model, manifest)
    assert bool(bound.binding.valid)
    assert (
        int(bound.binding.status) == FoundationBindingStatus.PRETRAINING_OVERLAP_UNKNOWN
    )
    assert int(bound.binding.evidence[2]) == 0


def test_native_binding_is_export_ready_while_external_runtime_is_host_only() -> None:
    model = Linear(in_size=3, out_size=2, rwf=False, key=jr.PRNGKey(0))
    bound = _bind(model, _manifest(model))
    pure = bound.export_callable()
    expected = pure(jnp.ones((3,)))
    actual = jax.jit(pure)(jnp.ones((3,)))
    np.testing.assert_allclose(actual, expected)

    external = ExternalFoundationRuntime(lambda value: value + 1, _manifest(model))
    assert not isinstance(external, eqx.Module)
    assert external.run_host(2) == 3


def test_low_rank_adapter_is_bound_to_exact_base_and_license() -> None:
    model = Linear(in_size=3, out_size=2, rwf=False, key=jr.PRNGKey(0))
    base = _bind(model, _manifest(model))
    adapted, _ = adapt_low_rank(
        model,
        {".weight": LowRankSpec(rank=1)},
        key=jr.PRNGKey(8),
    )
    adapter = LowRankAdapterProvenance(
        "adapter-1",
        _SHA_D,
        low_rank_adapter_parameter_sha256(adapted),
        base.binding.manifest.artifact_sha256,
        base.binding.manifest.parameter_sha256,
        rank=1,
        target_paths=(".weight",),
    )
    rebound = bind_low_rank_foundation_adapter(
        adapted, base, adapter, adapter_sha256=_SHA_D
    )
    assert rebound.binding.manifest.fingerprint == base.binding.manifest.fingerprint
    assert rebound.adapter_fingerprint == adapter.fingerprint
    with pytest.raises(ValueError, match="adapter artifact"):
        bind_low_rank_foundation_adapter(adapted, base, adapter, adapter_sha256=_SHA_A)
    tampered = eqx.tree_at(
        lambda value: value.weight.left,
        adapted,
        adapted.weight.left.at[0, 0].set(1.0),
    )
    with pytest.raises(ValueError, match="parameter hash"):
        bind_low_rank_foundation_adapter(
            tampered,
            base,
            adapter,
            adapter_sha256=_SHA_D,
        )


def _structure_batch(positions) -> MacromolecularBatch:
    return MacromolecularBatch(
        jnp.asarray((1,), dtype=jnp.int32),
        positions,
        jnp.asarray([[[1.0, -0.5], [0.2, 0.7], [-0.3, 0.4]]]),
        jnp.asarray([[True, True, True]]),
        jnp.asarray([True]),
        jnp.asarray([[0, 1, 1, 2, 2, 0]], dtype=jnp.int32),
        jnp.asarray([[1, 0, 2, 1, 0, 2]], dtype=jnp.int32),
        jnp.ones((1, 6), dtype=bool),
    )


def test_structure_encoder_is_translation_invariant_and_rotation_equivariant() -> None:
    positions = jnp.asarray([[[0.0, 0.0], [1.0, 0.0], [0.2, 1.1]]])
    rotation = jnp.asarray([[0.0, -1.0], [1.0, 0.0]])
    transformed = positions @ rotation.T + jnp.asarray([2.5, -3.0])
    model = EquivariantStructureEncoder(2, 5, depth=2, key=jr.PRNGKey(9))
    reference = model(_structure_batch(positions))
    rotated = model(_structure_batch(transformed))
    np.testing.assert_allclose(
        rotated.node_embeddings, reference.node_embeddings, atol=2e-6
    )
    np.testing.assert_allclose(
        rotated.coordinate_updates,
        reference.coordinate_updates @ rotation.T,
        atol=2e-6,
    )


def test_inverse_design_preserves_hard_constraints_and_separates_relaxed_objective() -> (
    None
):
    batch = _batch()
    active = batch.valid_mask
    probabilities = jnp.where(
        active[..., None],
        jnp.full((2, 3, DNA_IUPAC.size), 1.0 / DNA_IUPAC.size),
        0.0,
    )
    distribution = SequenceDistribution(
        batch.record_ids,
        probabilities,
        active,
        batch.case_mask,
        DNA_IUPAC,
    )
    fixed = FixedTokenConstraint(
        jnp.asarray((0, 1), dtype=jnp.int32),
        jnp.asarray((1, 0), dtype=jnp.int32),
        jnp.asarray((DNA_IUPAC.code("A"), DNA_IUPAC.code("C")), dtype=jnp.int32),
        case_capacity=2,
        length_capacity=3,
        alphabet_size=DNA_IUPAC.size,
    )
    allowed = jnp.ones((2, 3, DNA_IUPAC.size), dtype=bool)
    allowed = allowed.at[0, 0, DNA_IUPAC.code("T")].set(False)
    problem = SequenceDesignProblem(
        distribution,
        _hard_a_count,
        _relaxed_a_count,
        constraints=(fixed, AllowedTokenConstraint(allowed)),
        sample_count=12,
        sample_capacity=12,
    )
    first = solve_sequence_design(problem, key=jr.PRNGKey(123))
    second = solve_sequence_design(problem, key=jr.PRNGKey(123))
    np.testing.assert_array_equal(first.candidate_codes, second.candidate_codes)
    assert bool(first.valid)
    assert bool(jnp.all(first.constraint_satisfied))
    assert int(first.selected_codes[0, 1]) == DNA_IUPAC.code("A")
    assert int(first.selected_codes[1, 0]) == DNA_IUPAC.code("C")
    assert first.relaxed_method_contract.method_kind.value == "relaxed_objective"
    assert first.method_contract.method_kind.value == "heuristic"
    assert not np.array_equal(first.relaxed_objective, first.hard_objective)
    np.testing.assert_allclose(
        first.relaxation_gap,
        first.relaxed_objective - first.hard_objective,
    )
    assert bool(jnp.all((first.unique_fraction > 0.0) & (first.unique_fraction <= 1.0)))


def test_inverse_design_preflights_sample_capacity() -> None:
    batch = _batch()
    probabilities = jnp.where(
        batch.valid_mask[..., None],
        jnp.full((2, 3, DNA_IUPAC.size), 1.0 / DNA_IUPAC.size),
        0.0,
    )
    distribution = SequenceDistribution(
        batch.record_ids,
        probabilities,
        batch.valid_mask,
        batch.case_mask,
        DNA_IUPAC,
    )
    with pytest.raises(ValueError, match="exceed"):
        SequenceDesignProblem(
            distribution,
            _hard_a_count,
            _relaxed_a_count,
            sample_count=9,
            sample_capacity=8,
        )
