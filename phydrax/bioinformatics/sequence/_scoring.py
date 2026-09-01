#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule


class SubstitutionScoreTable(StrictModule):
    """Canonical scores with explicit distributions for encoded ambiguity symbols."""

    canonical_symbols: tuple[str, ...] = eqx.field(static=True)
    symbols: tuple[str, ...] = eqx.field(static=True)
    canonical_scores: Array
    symbol_probabilities: Array
    encoded_scores: Array
    table_id: str = eqx.field(static=True)

    def __init__(
        self,
        canonical_symbols: Sequence[str],
        canonical_scores: ArrayLike,
        /,
        *,
        symbols: Sequence[str] | None = None,
        symbol_probabilities: ArrayLike | None = None,
        name: str = "custom",
    ):
        canonical = tuple(str(symbol) for symbol in canonical_symbols)
        encoded = (
            canonical if symbols is None else tuple(str(symbol) for symbol in symbols)
        )
        if not canonical or len(set(canonical)) != len(canonical):
            raise ValueError("canonical_symbols must be non-empty and unique.")
        if not encoded or len(set(encoded)) != len(encoded):
            raise ValueError("symbols must be non-empty and unique.")
        raw_scores = jnp.asarray(canonical_scores)
        dtype = (
            raw_scores.dtype
            if jnp.issubdtype(raw_scores.dtype, jnp.floating)
            else jnp.asarray(0.0).dtype
        )
        scores = jnp.asarray(raw_scores, dtype=dtype)
        count = len(canonical)
        if scores.shape != (count, count):
            raise ValueError(
                "canonical_scores must have shape "
                f"({count}, {count}); got {scores.shape}."
            )
        if symbol_probabilities is None:
            if encoded != canonical:
                raise ValueError(
                    "symbol_probabilities are required when symbols include ambiguities."
                )
            probabilities = jnp.eye(count, dtype=scores.dtype)
        else:
            probabilities = jnp.asarray(symbol_probabilities, dtype=scores.dtype)
        if probabilities.shape != (len(encoded), count):
            raise ValueError(
                "symbol_probabilities must have shape "
                f"({len(encoded)}, {count}); got {probabilities.shape}."
            )
        probability_sum = jnp.sum(probabilities, axis=-1, keepdims=True)
        probabilities = probabilities / jnp.where(
            probability_sum > 0.0, probability_sum, 1.0
        )
        encoded_scores = (probabilities @ scores) @ probabilities.T
        fingerprint = canonical_fingerprint(
            {
                "kind": "substitution-score-table",
                "name": str(name),
                "canonical_symbols": canonical,
                "symbols": encoded,
                "arrays": array_tree_fingerprint((scores, probabilities)),
            }
        )
        self.canonical_symbols = canonical
        self.symbols = encoded
        self.canonical_scores = scores
        self.symbol_probabilities = probabilities
        self.encoded_scores = encoded_scores
        self.table_id = fingerprint

    @property
    def canonical_size(self) -> int:
        return len(self.canonical_symbols)

    @property
    def symbol_count(self) -> int:
        return len(self.symbols)

    def score_codes(self, left_codes: ArrayLike, right_codes: ArrayLike, /) -> Array:
        """Score broadcast-compatible encoded symbols, including ambiguities."""
        left = jnp.asarray(left_codes, dtype=jnp.int32)
        right = jnp.asarray(right_codes, dtype=jnp.int32)
        safe_left = jnp.clip(left, 0, self.symbol_count - 1)
        safe_right = jnp.clip(right, 0, self.symbol_count - 1)
        scores = self.encoded_scores[safe_left, safe_right]
        valid = (
            (left >= 0)
            & (left < self.symbol_count)
            & (right >= 0)
            & (right < self.symbol_count)
        )
        return jnp.where(valid, scores, -jnp.inf)

    def pairwise_scores(self, left_codes: ArrayLike, right_codes: ArrayLike, /) -> Array:
        """Return the dense pairwise score matrix for two rank-one code arrays."""
        left = jnp.asarray(left_codes, dtype=jnp.int32)
        right = jnp.asarray(right_codes, dtype=jnp.int32)
        if left.ndim != 1 or right.ndim != 1:
            raise ValueError("pairwise_scores requires two rank-one code arrays.")
        return self.score_codes(left[:, None], right[None, :])

    def expected_pairwise_scores(
        self,
        left_probabilities: ArrayLike,
        right_probabilities: ArrayLike,
        /,
    ) -> Array:
        """Score canonical symbol distributions without collapsing ambiguities."""
        left = jnp.asarray(left_probabilities, dtype=self.canonical_scores.dtype)
        right = jnp.asarray(right_probabilities, dtype=self.canonical_scores.dtype)
        if (
            left.ndim != 2
            or right.ndim != 2
            or left.shape[-1] != self.canonical_size
            or right.shape[-1] != self.canonical_size
        ):
            raise ValueError(
                "probability arrays must be rank two with trailing canonical size."
            )
        return (left @ self.canonical_scores) @ right.T


def identity_substitution_table(
    symbols: Sequence[str],
    /,
    *,
    match_score: float = 1.0,
    mismatch_score: float = -1.0,
    name: str = "identity",
) -> SubstitutionScoreTable:
    """Construct a symmetric match/mismatch score table."""
    symbols_ = tuple(str(symbol) for symbol in symbols)
    count = len(symbols_)
    scores = jnp.full((count, count), float(mismatch_score))
    scores = scores.at[jnp.diag_indices(count)].set(float(match_score))
    return SubstitutionScoreTable(symbols_, scores, name=name)


def nucleotide_substitution_table(
    *,
    rna: bool = False,
    match_score: float = 2.0,
    mismatch_score: float = -3.0,
) -> SubstitutionScoreTable:
    """Return an IUPAC nucleotide table with expectation-valued ambiguity scores."""
    canonical = ("A", "C", "G", "U" if rna else "T")
    symbols = canonical + ("R", "Y", "S", "W", "K", "M", "B", "D", "H", "V", "N")
    scores = jnp.full((4, 4), float(mismatch_score))
    scores = scores.at[jnp.diag_indices(4)].set(float(match_score))
    probabilities = jnp.asarray(
        (
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
            (1, 0, 1, 0),
            (0, 1, 0, 1),
            (0, 1, 1, 0),
            (1, 0, 0, 1),
            (0, 0, 1, 1),
            (1, 1, 0, 0),
            (0, 1, 1, 1),
            (1, 0, 1, 1),
            (1, 1, 0, 1),
            (1, 1, 1, 0),
            (1, 1, 1, 1),
        ),
        dtype=scores.dtype,
    )
    return SubstitutionScoreTable(
        canonical,
        scores,
        symbols=symbols,
        symbol_probabilities=probabilities,
        name="rna-iupac" if rna else "dna-iupac",
    )


def blosum62_substitution_table() -> SubstitutionScoreTable:
    """Return BLOSUM62 with protein-IUPAC U, O, B, Z, J, and X support."""
    canonical = tuple("ARNDCQEGHILKMFPSTWYV")
    scores = jnp.asarray(
        (
            (4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0),
            (-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3),
            (-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3),
            (-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3),
            (
                0,
                -3,
                -3,
                -3,
                9,
                -3,
                -4,
                -3,
                -3,
                -1,
                -1,
                -3,
                -1,
                -2,
                -3,
                -1,
                -1,
                -2,
                -2,
                -1,
            ),
            (-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2),
            (-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2),
            (0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3),
            (-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3),
            (-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3),
            (-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1),
            (-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2),
            (-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1),
            (-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1),
            (
                -1,
                -2,
                -2,
                -1,
                -3,
                -1,
                -1,
                -2,
                -2,
                -3,
                -3,
                -1,
                -2,
                -4,
                7,
                -1,
                -1,
                -4,
                -3,
                -2,
            ),
            (1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2),
            (0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0),
            (
                -3,
                -3,
                -4,
                -4,
                -2,
                -2,
                -3,
                -2,
                -2,
                -3,
                -2,
                -3,
                -1,
                1,
                -4,
                -3,
                -2,
                11,
                2,
                -3,
            ),
            (-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1),
            (0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4),
        ),
        dtype=jnp.float32,
    )
    probabilities = jnp.concatenate(
        (
            jnp.eye(20, dtype=scores.dtype),
            jnp.zeros((6, 20), dtype=scores.dtype),
        ),
        axis=0,
    )
    probabilities = probabilities.at[20, 4].set(1.0)
    probabilities = probabilities.at[21, 11].set(1.0)
    probabilities = probabilities.at[22, jnp.asarray((2, 3))].set(1.0)
    probabilities = probabilities.at[23, jnp.asarray((5, 6))].set(1.0)
    probabilities = probabilities.at[24, jnp.asarray((9, 10))].set(1.0)
    probabilities = probabilities.at[25, :].set(1.0)
    return SubstitutionScoreTable(
        canonical,
        scores,
        symbols=canonical + ("U", "O", "B", "Z", "J", "X"),
        symbol_probabilities=probabilities,
        name="blosum62",
    )


__all__ = [
    "SubstitutionScoreTable",
    "blosum62_substitution_table",
    "identity_substitution_table",
    "nucleotide_substitution_table",
]
