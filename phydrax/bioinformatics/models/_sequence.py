#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from phydrax._doc import DOC_KEY0
from phydrax._strict import StrictModule
from phydrax.nn.layers import Linear, MeasureAwareAttention, RecurrentBatch
from phydrax.nn.models import SelectiveSequenceModel

from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ..sequence import AlphabetPlan, SequenceBatch
from ._objectives import PairPrediction, TokenLabelPrediction, TokenPrediction


class BiologicalModelStatus(IntEnum):
    """Compiled status codes shared by native biological model wrappers."""

    SUCCESS = 0
    EMPTY_BATCH = 1
    NONFINITE = 2


_SEQUENCE_CONTRACT = BioinformaticsMethodContract(
    "learned-sequence-representation",
    MethodKind.LEARNED,
    ExecutionKind.FLOATING_POINT_DIRECT,
    DifferentiationKind.EXACT_AD,
    OutputKind.ARRAY,
    conditioning_statement="Finite affine, recurrent, and attention operations are evaluated directly.",
    truncation_statement="No valid sequence positions are truncated.",
    capacity_semantics="The caller-provided padded length is evaluated exactly under its mask.",
    assumptions=("Model alphabet identity equals the encoded batch alphabet identity.",),
    input_dtype="integer token codes",
    compute_dtype="model parameter dtype",
    output_dtype="model parameter dtype",
)

_TOKEN_PREDICTION_CONTRACT = BioinformaticsMethodContract(
    "learned-token-prediction",
    MethodKind.LEARNED,
    ExecutionKind.FLOATING_POINT_DIRECT,
    DifferentiationKind.EXACT_AD,
    OutputKind.PROBABILISTIC,
    conditioning_statement="Returned values are unnormalized finite categorical logits.",
    truncation_statement="No valid token is truncated.",
    capacity_semantics="One vocabulary logit is returned per alphabet symbol and valid position.",
    assumptions=("Tokenizer and alphabet fingerprints are exact static identities.",),
    input_dtype="real embeddings",
    compute_dtype="model parameter dtype",
    output_dtype="model parameter dtype",
)

_LABEL_PREDICTION_CONTRACT = BioinformaticsMethodContract(
    "learned-token-label-prediction",
    MethodKind.LEARNED,
    ExecutionKind.FLOATING_POINT_DIRECT,
    DifferentiationKind.EXACT_AD,
    OutputKind.ARRAY,
    conditioning_statement="Returned values are unnormalized label logits.",
    truncation_statement="No valid token is truncated.",
    capacity_semantics="The declared label count is evaluated at every valid position.",
)

_PAIR_PREDICTION_CONTRACT = BioinformaticsMethodContract(
    "learned-token-pair-prediction",
    MethodKind.LEARNED,
    ExecutionKind.FLOATING_POINT_DIRECT,
    DifferentiationKind.EXACT_AD,
    OutputKind.ARRAY,
    conditioning_statement="Pair features are projected directly from finite token embeddings.",
    truncation_statement="The complete padded pair square is represented and masked.",
    capacity_semantics="Quadratic pair storage is explicit and never silently truncated.",
)


class SequenceEncoderResult(StrictModule):
    """Masked token and pooled embeddings with compiled validity evidence."""

    token_embeddings: Array
    pooled_embedding: Array
    valid_mask: Array
    valid: Array
    status: Array
    evidence: Array
    alphabet_fingerprint: str = eqx.field(static=True)
    tokenizer_fingerprint: str = eqx.field(static=True)
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)

    def __init__(
        self,
        token_embeddings: Array,
        valid_mask: Array,
        *,
        alphabet_fingerprint: str,
        tokenizer_fingerprint: str,
    ):
        values = jnp.asarray(token_embeddings)
        mask = jnp.asarray(valid_mask, dtype=bool)
        if values.ndim != 3 or values.shape[:2] != mask.shape:
            raise ValueError("Token embeddings require shape (batch, length, channels).")
        values = jnp.where(mask[..., None], values, 0.0)
        count = jnp.sum(mask, axis=1, dtype=jnp.int32)
        pooled = jnp.sum(values, axis=1) / jnp.maximum(count, 1)[:, None].astype(
            values.dtype
        )
        finite = jnp.all(jnp.isfinite(values))
        support = jnp.any(mask)
        self.token_embeddings = values
        self.pooled_embedding = pooled
        self.valid_mask = mask
        self.valid = finite & support
        self.status = jnp.where(
            ~support,
            jnp.asarray(BiologicalModelStatus.EMPTY_BATCH, dtype=jnp.int32),
            jnp.where(
                finite,
                jnp.asarray(BiologicalModelStatus.SUCCESS, dtype=jnp.int32),
                jnp.asarray(BiologicalModelStatus.NONFINITE, dtype=jnp.int32),
            ),
        )
        self.evidence = jnp.stack(
            (jnp.sum(mask, dtype=jnp.int32), jnp.asarray(mask.size, dtype=jnp.int32))
        )
        self.alphabet_fingerprint = str(alphabet_fingerprint)
        self.tokenizer_fingerprint = str(tokenizer_fingerprint)
        self.method_contract = _SEQUENCE_CONTRACT


class SequenceEmbedding(StrictModule):
    """Native alphabet-bound embedding table with exact padding zeroing."""

    table: Array
    alphabet_fingerprint: str = eqx.field(static=True)
    vocabulary_size: int = eqx.field(static=True)
    embedding_size: int = eqx.field(static=True)

    def __init__(
        self,
        alphabet: AlphabetPlan,
        embedding_size: int,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        if not isinstance(alphabet, AlphabetPlan):
            raise TypeError("alphabet must be an AlphabetPlan.")
        width = int(embedding_size)
        if width <= 0:
            raise ValueError("embedding_size must be positive.")
        scale = 1.0 / max(width, 1) ** 0.5
        self.table = jr.normal(key, (alphabet.size, width)) * scale
        self.alphabet_fingerprint = alphabet.fingerprint
        self.vocabulary_size = alphabet.size
        self.embedding_size = width

    def __call__(self, batch: SequenceBatch, /) -> Array:
        if not isinstance(batch, SequenceBatch):
            raise TypeError("batch must be a SequenceBatch.")
        if batch.alphabet.fingerprint != self.alphabet_fingerprint:
            raise ValueError(
                "Sequence batch alphabet does not match the embedding table."
            )
        embedded = self.table[batch.token_codes]
        active = batch.valid_mask & batch.case_mask[:, None]
        return jnp.where(active[..., None], embedded, 0.0)


class RecurrentSequenceEncoder(StrictModule):
    """Alphabet-bound selective state-space encoder over padded sequence batches."""

    embedding: SequenceEmbedding
    recurrent: SelectiveSequenceModel
    tokenizer_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        alphabet: AlphabetPlan,
        embedding_size: int,
        state_size: int,
        /,
        *,
        depth: int = 1,
        tokenizer_fingerprint: str,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        embedding_key, recurrent_key = jr.split(key)
        self.embedding = SequenceEmbedding(alphabet, embedding_size, key=embedding_key)
        self.recurrent = SelectiveSequenceModel(
            embedding_size,
            state_size,
            inner_size=embedding_size,
            depth=depth,
            return_mode="sequence",
            key=recurrent_key,
        )
        identifier = str(tokenizer_fingerprint)
        if not identifier:
            raise ValueError("tokenizer_fingerprint must be non-empty.")
        self.tokenizer_fingerprint = identifier

    def __call__(self, batch: SequenceBatch, /) -> SequenceEncoderResult:
        embedded = self.embedding(batch)
        active = batch.valid_mask & batch.case_mask[:, None]
        output = self.recurrent(RecurrentBatch(embedded, active))
        return SequenceEncoderResult(
            output,
            active,
            alphabet_fingerprint=self.embedding.alphabet_fingerprint,
            tokenizer_fingerprint=self.tokenizer_fingerprint,
        )


class AttentionSequenceEncoder(StrictModule):
    """Measure-aware self-attention encoder with exact source and query masks."""

    embedding: SequenceEmbedding
    attention_layers: tuple[MeasureAwareAttention, ...]
    tokenizer_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        alphabet: AlphabetPlan,
        embedding_size: int,
        /,
        *,
        depth: int = 1,
        num_heads: int = 4,
        head_dim: int | None = None,
        tokenizer_fingerprint: str,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        depth_ = int(depth)
        heads = int(num_heads)
        if depth_ <= 0 or heads <= 0:
            raise ValueError("depth and num_heads must be positive.")
        dimension = (
            max(1, int(embedding_size) // heads) if head_dim is None else int(head_dim)
        )
        keys = jr.split(key, depth_ + 1)
        self.embedding = SequenceEmbedding(alphabet, embedding_size, key=keys[0])
        self.attention_layers = tuple(
            MeasureAwareAttention(
                source_channels=embedding_size,
                query_channels=embedding_size,
                out_channels=embedding_size,
                num_heads=heads,
                head_dim=dimension,
                execution="dense",
                key=layer_key,
            )
            for layer_key in keys[1:]
        )
        identifier = str(tokenizer_fingerprint)
        if not identifier:
            raise ValueError("tokenizer_fingerprint must be non-empty.")
        self.tokenizer_fingerprint = identifier

    def __call__(self, batch: SequenceBatch, /) -> SequenceEncoderResult:
        values = self.embedding(batch)
        active = batch.valid_mask & batch.case_mask[:, None]
        weights = active.astype(values.dtype)
        for attention in self.attention_layers:
            update = attention(
                values,
                values,
                weights,
                source_mask=active,
                query_mask=active,
            )
            values = jnp.where(active[..., None], values + update, 0.0)
        return SequenceEncoderResult(
            values,
            active,
            alphabet_fingerprint=self.embedding.alphabet_fingerprint,
            tokenizer_fingerprint=self.tokenizer_fingerprint,
        )


class TokenPredictionHead(StrictModule):
    """Native affine vocabulary head preserving encoder identity metadata."""

    projection: Linear
    alphabet_fingerprint: str = eqx.field(static=True)
    tokenizer_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        alphabet: AlphabetPlan,
        /,
        *,
        tokenizer_fingerprint: str,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.projection = Linear(
            in_size=int(input_size),
            out_size=alphabet.size,
            activation=None,
            rwf=False,
            key=key,
        )
        self.alphabet_fingerprint = alphabet.fingerprint
        self.tokenizer_fingerprint = str(tokenizer_fingerprint)

    def __call__(self, encoded: SequenceEncoderResult, /) -> TokenPrediction:
        if encoded.alphabet_fingerprint != self.alphabet_fingerprint:
            raise ValueError("Encoder and vocabulary head alphabets differ.")
        if encoded.tokenizer_fingerprint != self.tokenizer_fingerprint:
            raise ValueError("Encoder and vocabulary head tokenizers differ.")
        logits = self.projection(encoded.token_embeddings)
        logits = jnp.where(encoded.valid_mask[..., None], logits, 0.0)
        return TokenPrediction(
            logits,
            encoded.valid_mask,
            alphabet_fingerprint=self.alphabet_fingerprint,
            tokenizer_fingerprint=self.tokenizer_fingerprint,
            method_contract=_TOKEN_PREDICTION_CONTRACT,
        )


class TokenLabelHead(StrictModule):
    """Native per-token label head."""

    projection: Linear
    label_space_id: str = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        label_count: int,
        /,
        *,
        label_space_id: str,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        if int(label_count) <= 0:
            raise ValueError("label_count must be positive.")
        self.projection = Linear(
            in_size=int(input_size),
            out_size=int(label_count),
            activation=None,
            rwf=False,
            key=key,
        )
        self.label_space_id = str(label_space_id)

    def __call__(self, encoded: SequenceEncoderResult, /) -> TokenLabelPrediction:
        logits = self.projection(encoded.token_embeddings)
        return TokenLabelPrediction(
            jnp.where(encoded.valid_mask[..., None], logits, 0.0),
            encoded.valid_mask,
            label_space_id=self.label_space_id,
            method_contract=_LABEL_PREDICTION_CONTRACT,
        )


class PairPredictionHead(StrictModule):
    """Explicit full-capacity pair head with optional exact symmetrization."""

    projection: Linear
    pair_space_id: str = eqx.field(static=True)
    symmetric: bool = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        output_size: int,
        /,
        *,
        pair_space_id: str,
        symmetric: bool = False,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        width = int(input_size)
        outputs = int(output_size)
        if width <= 0 or outputs <= 0:
            raise ValueError("input_size and output_size must be positive.")
        self.projection = Linear(
            in_size=3 * width,
            out_size=outputs,
            activation=None,
            rwf=False,
            key=key,
        )
        self.pair_space_id = str(pair_space_id)
        self.symmetric = bool(symmetric)

    def __call__(self, encoded: SequenceEncoderResult, /) -> PairPrediction:
        values = encoded.token_embeddings
        length = int(values.shape[1])
        left = jnp.broadcast_to(
            values[:, :, None, :], values.shape[:2] + (length, values.shape[-1])
        )
        right = jnp.broadcast_to(
            values[:, None, :, :], values.shape[:1] + (length,) + values.shape[1:]
        )
        features = jnp.concatenate((left, right, left * right), axis=-1)
        logits = self.projection(features)
        pair_mask = encoded.valid_mask[:, :, None] & encoded.valid_mask[:, None, :]
        return PairPrediction(
            jnp.where(pair_mask[..., None], logits, 0.0),
            pair_mask,
            pair_space_id=self.pair_space_id,
            symmetric=self.symmetric,
            method_contract=_PAIR_PREDICTION_CONTRACT,
        )


__all__ = [
    "AttentionSequenceEncoder",
    "BiologicalModelStatus",
    "PairPredictionHead",
    "RecurrentSequenceEncoder",
    "SequenceEmbedding",
    "SequenceEncoderResult",
    "TokenLabelHead",
    "TokenPredictionHead",
]
