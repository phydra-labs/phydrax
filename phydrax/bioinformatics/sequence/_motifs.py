#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from numbers import Integral

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import core as jax_core
from jaxtyping import Array, ArrayLike

from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax._strict import StrictModule

from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._alphabet import AlphabetPlan, DNA_IUPAC
from ._batch import SequenceBatch


MOTIF_STATUS_VALID = 0
MOTIF_STATUS_CAPACITY_EXCEEDED = 1
MOTIF_STATUS_NO_VALID_WINDOW = 2


def _observation_support(alphabet: AlphabetPlan, /) -> tuple[Array, Array]:
    """Canonical-symbol support and scorable-code mask for an encoded alphabet."""
    canonical_index = {
        symbol: index for index, symbol in enumerate(alphabet.canonical_symbols)
    }
    support = np.zeros((alphabet.size, len(canonical_index)), dtype=np.float64)
    scorable = np.zeros((alphabet.size,), dtype=bool)
    ambiguity = alphabet.ambiguity_map
    uninformative = {
        alphabet.unknown_symbol,
        alphabet.missing_symbol,
    }
    for code, symbol in enumerate(alphabet.symbols):
        if symbol in canonical_index:
            support[code, canonical_index[symbol]] = 1.0
            scorable[code] = True
        elif symbol in ambiguity:
            for value in ambiguity[symbol]:
                support[code, canonical_index[value]] = 1.0
            scorable[code] = True
        elif symbol in uninformative:
            support[code, :] = 1.0
            scorable[code] = True
    return jnp.asarray(support), jnp.asarray(scorable)


def _method_contract(reverse_complement: bool) -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "position-weight-matrix motif scan",
        MethodKind.EXACT_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.ALMOST_EVERYWHERE,
        OutputKind.STRUCTURED,
        conditioning_statement=(
            "Exact log-odds evaluation for the supplied PWM, background, alphabet, "
            "position and soft masks, and both strands."
            if reverse_complement
            else "Exact log-odds evaluation for the supplied PWM, background, alphabet, position mask, and soft mask."
        ),
        truncation_statement="Every in-capacity window is evaluated; no hit truncation is performed.",
        capacity_semantics="The sequence capacity must not exceed the scan plan bound.",
        assumptions=(
            "PWM columns are conditionally independent.",
            "Ambiguity codes marginalize over their canonical support.",
        ),
        nondifferentiable_outputs=("best_position", "strand", "status"),
    )


class PositionWeightMatrix(StrictModule):
    """A normalized canonical-symbol PWM with ambiguity-aware log-odds tables."""

    probabilities: Array
    background: Array
    log_odds: Array
    reverse_log_odds: Array
    scorable_codes: Array
    information_content: Array
    alphabet: AlphabetPlan = eqx.field(static=True)
    motif_id: str = eqx.field(static=True)
    fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        weights: ArrayLike,
        alphabet: AlphabetPlan = DNA_IUPAC,
        *,
        background: ArrayLike | None = None,
        pseudocount: float = 0.0,
        motif_id: str = "motif",
    ):
        if not isinstance(alphabet, AlphabetPlan):
            raise TypeError("alphabet must be an AlphabetPlan.")
        values = jnp.asarray(weights)
        canonical_count = len(alphabet.canonical_symbols)
        if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] != canonical_count:
            raise ValueError(
                "weights must have shape (positive motif width, canonical symbol count)."
            )
        if not jnp.issubdtype(values.dtype, jnp.floating):
            values = values.astype(jnp.float32)
        concrete = None if isinstance(values, jax_core.Tracer) else np.asarray(values)
        if concrete is not None and (
            np.any(~np.isfinite(concrete)) or np.any(concrete < 0.0)
        ):
            raise ValueError("PWM weights must be finite and non-negative.")
        pseudo = float(pseudocount)
        if not np.isfinite(pseudo) or pseudo < 0.0:
            raise ValueError("pseudocount must be finite and non-negative.")
        adjusted = values + pseudo
        totals = jnp.sum(adjusted, axis=1, keepdims=True)
        if concrete is not None and np.any(np.asarray(totals) <= 0.0):
            raise ValueError("Every PWM position must have positive total weight.")
        probabilities = adjusted / totals

        if background is None:
            background_values = jnp.full(
                (canonical_count,), 1.0 / canonical_count, dtype=probabilities.dtype
            )
        else:
            background_values = jnp.asarray(background, dtype=probabilities.dtype)
            if background_values.shape != (canonical_count,):
                raise ValueError(
                    "background must have one weight per canonical alphabet symbol."
                )
            concrete_background = (
                None
                if isinstance(background_values, jax_core.Tracer)
                else np.asarray(background_values)
            )
            if concrete_background is not None and (
                np.any(~np.isfinite(concrete_background))
                or np.any(concrete_background <= 0.0)
            ):
                raise ValueError(
                    "Background weights must be finite and strictly positive."
                )
            background_values = background_values / jnp.sum(background_values)

        support, scorable = _observation_support(alphabet)
        support = support.astype(probabilities.dtype)
        observed_pwm = probabilities @ support.T
        observed_background = background_values @ support.T
        safe_pwm = jnp.where(observed_pwm > 0.0, observed_pwm, 1.0)
        safe_background = jnp.where(observed_background > 0.0, observed_background, 1.0)
        log_odds = jnp.where(
            scorable[None, :] & (observed_pwm > 0.0),
            jnp.log(safe_pwm) - jnp.log(safe_background)[None, :],
            -jnp.inf,
        )

        if alphabet.complements:
            complement_codes = jnp.asarray(
                [
                    alphabet.code(alphabet.complement_map[symbol])
                    for symbol in alphabet.symbols
                ],
                dtype=jnp.int32,
            )
            reverse_log_odds = log_odds[::-1, :][:, complement_codes]
        else:
            reverse_log_odds = jnp.full_like(log_odds, -jnp.inf)
        tiny = jnp.finfo(probabilities.dtype).tiny
        information = jnp.sum(
            probabilities
            * (
                jnp.log2(jnp.maximum(probabilities, tiny))
                - jnp.log2(background_values)[None, :]
            ),
            axis=1,
        )
        identifier = str(motif_id).strip()
        if not identifier:
            raise ValueError("motif_id must be non-empty.")

        self.probabilities = probabilities
        self.background = background_values
        self.log_odds = log_odds
        self.reverse_log_odds = reverse_log_odds
        self.scorable_codes = scorable
        self.information_content = information
        self.alphabet = alphabet
        self.motif_id = identifier
        self.fingerprint = canonical_fingerprint(
            {
                "kind": "position-weight-matrix",
                "motif_id": identifier,
                "alphabet": alphabet.fingerprint,
                "probabilities": array_tree_fingerprint(probabilities),
                "background": array_tree_fingerprint(background_values),
            }
        )

    @property
    def width(self) -> int:
        return int(self.probabilities.shape[0])

    @classmethod
    def from_counts(
        cls,
        counts: ArrayLike,
        alphabet: AlphabetPlan = DNA_IUPAC,
        *,
        background: ArrayLike | None = None,
        pseudocount: float = 0.5,
        motif_id: str = "motif",
    ) -> PositionWeightMatrix:
        return cls(
            counts,
            alphabet,
            background=background,
            pseudocount=pseudocount,
            motif_id=motif_id,
        )


class MotifScanPlan(StrictModule):
    """Static capacity and strand semantics for a dense PWM scan."""

    maximum_sequence_length: int = eqx.field(static=True)
    reverse_complement: bool = eqx.field(static=True)
    tie_policy: str = eqx.field(static=True)
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)

    def __init__(
        self,
        maximum_sequence_length: int,
        *,
        reverse_complement: bool = True,
    ):
        if (
            isinstance(maximum_sequence_length, bool)
            or not isinstance(maximum_sequence_length, Integral)
            or int(maximum_sequence_length) <= 0
        ):
            raise ValueError("maximum_sequence_length must be a positive integer.")
        self.maximum_sequence_length = int(maximum_sequence_length)
        self.reverse_complement = bool(reverse_complement)
        self.tie_policy = "lowest-position-then-forward-strand"
        self.method_contract = _method_contract(bool(reverse_complement))


class MotifScanEvidence(StrictModule):
    """Observable scan coverage and strand-selection evidence."""

    capacity_sufficient: Array
    evaluated_windows: Array
    valid_windows: Array
    ambiguity_positions: Array
    masked_positions: Array
    forward_selected: Array
    reverse_selected: Array


class MotifScanResult(StrictModule):
    """All window scores plus deterministic per-record best motif calls."""

    forward_scores: Array
    reverse_scores: Array
    scores: Array
    strand: Array
    window_valid: Array
    best_position: Array
    best_score: Array
    best_strand: Array
    valid: Array
    status: Array
    evidence: MotifScanEvidence
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


def scan_motif(
    matrix: PositionWeightMatrix,
    sequences: SequenceBatch,
    plan: MotifScanPlan | None = None,
    /,
) -> MotifScanResult:
    """Evaluate a PWM at every sequence window without truncating reported hits."""
    if not isinstance(matrix, PositionWeightMatrix):
        raise TypeError("matrix must be a PositionWeightMatrix.")
    if not isinstance(sequences, SequenceBatch):
        raise TypeError("sequences must be a SequenceBatch.")
    if sequences.alphabet.fingerprint != matrix.alphabet.fingerprint:
        raise ValueError("PWM and sequence alphabets must match.")
    if plan is None:
        plan = MotifScanPlan(
            max(sequences.capacity, 1),
            reverse_complement=bool(matrix.alphabet.complements),
        )
    if not isinstance(plan, MotifScanPlan):
        raise TypeError("plan must be a MotifScanPlan.")
    if plan.reverse_complement and not matrix.alphabet.complements:
        raise ValueError("Reverse-complement scanning requires a complement alphabet.")

    record_count = sequences.record_capacity
    window_count = max(sequences.sequence_capacity - matrix.width + 1, 0)
    capacity_ok = sequences.sequence_capacity <= plan.maximum_sequence_length
    if window_count == 0 or not capacity_ok:
        unavailable_scores = jnp.full(
            (record_count, window_count),
            -jnp.inf,
            dtype=matrix.probabilities.dtype,
        )
        unavailable_bool = jnp.zeros((record_count, window_count), dtype=bool)
        status = jnp.where(
            sequences.case_mask & ~capacity_ok,
            MOTIF_STATUS_CAPACITY_EXCEEDED,
            MOTIF_STATUS_NO_VALID_WINDOW,
        ).astype(jnp.int32)
        ambiguity_codes = jnp.asarray(
            [matrix.alphabet.code(symbol) for symbol, _ in matrix.alphabet.ambiguities],
            dtype=jnp.int32,
        )
        evidence = MotifScanEvidence(
            jnp.full((record_count,), capacity_ok) & sequences.case_mask,
            jnp.zeros((record_count,), dtype=jnp.int32),
            jnp.zeros((record_count,), dtype=jnp.int32),
            jnp.sum(
                sequences.valid_mask & jnp.isin(sequences.token_codes, ambiguity_codes),
                axis=1,
                dtype=jnp.int32,
            ),
            jnp.sum(sequences.soft_mask, axis=1, dtype=jnp.int32),
            jnp.zeros((record_count,), dtype=jnp.int32),
            jnp.zeros((record_count,), dtype=jnp.int32),
        )
        return MotifScanResult(
            unavailable_scores,
            unavailable_scores,
            unavailable_scores,
            jnp.zeros((record_count, window_count), dtype=jnp.int8),
            unavailable_bool,
            jnp.full((record_count,), -1, dtype=jnp.int32),
            jnp.full((record_count,), -jnp.inf, dtype=matrix.probabilities.dtype),
            jnp.zeros((record_count,), dtype=jnp.int8),
            jnp.zeros((record_count,), dtype=bool),
            status,
            evidence,
            plan.method_contract,
        )

    positions = jnp.arange(window_count, dtype=jnp.int32)[:, None]
    offsets = jnp.arange(matrix.width, dtype=jnp.int32)[None, :]
    window_indices = positions + offsets
    tokens = sequences.token_codes[:, window_indices]
    masks = sequences.valid_mask[:, window_indices]
    soft_masks = sequences.soft_mask[:, window_indices]
    scorable = matrix.scorable_codes[tokens]
    window_valid = sequences.case_mask[:, None] & jnp.all(
        masks & ~soft_masks & scorable, axis=-1
    )
    motif_positions = jnp.arange(matrix.width, dtype=jnp.int32)[None, None, :]
    forward = jnp.sum(matrix.log_odds[motif_positions, tokens], axis=-1)
    forward = jnp.where(window_valid, forward, -jnp.inf)

    if plan.reverse_complement:
        reverse = jnp.sum(matrix.reverse_log_odds[motif_positions, tokens], axis=-1)
        reverse = jnp.where(window_valid, reverse, -jnp.inf)
        reverse_wins = reverse > forward
        scores = jnp.where(reverse_wins, reverse, forward)
        strand = jnp.where(window_valid, jnp.where(reverse_wins, -1, 1), 0).astype(
            jnp.int8
        )
    else:
        reverse = jnp.full_like(forward, -jnp.inf)
        reverse_wins = jnp.zeros_like(window_valid)
        scores = forward
        strand = jnp.where(window_valid, 1, 0).astype(jnp.int8)

    best_position = jnp.argmax(scores, axis=1).astype(jnp.int32)
    best_score = jnp.take_along_axis(scores, best_position[:, None], axis=1)[:, 0]
    any_valid = jnp.any(window_valid, axis=1) & sequences.case_mask & capacity_ok
    best_position = jnp.where(any_valid, best_position, -1)
    best_score = jnp.where(any_valid, best_score, -jnp.inf)
    best_strand = jnp.where(
        any_valid,
        jnp.take_along_axis(strand, jnp.maximum(best_position, 0)[:, None], axis=1)[:, 0],
        0,
    ).astype(jnp.int8)
    status = jnp.where(
        capacity_ok,
        jnp.where(any_valid, MOTIF_STATUS_VALID, MOTIF_STATUS_NO_VALID_WINDOW),
        MOTIF_STATUS_CAPACITY_EXCEEDED,
    ).astype(jnp.int32)
    ambiguity_codes = jnp.asarray(
        [matrix.alphabet.code(symbol) for symbol, _ in matrix.alphabet.ambiguities],
        dtype=jnp.int32,
    )
    evidence = MotifScanEvidence(
        jnp.full((record_count,), capacity_ok),
        jnp.full((record_count,), window_count, dtype=jnp.int32),
        jnp.sum(window_valid, axis=1, dtype=jnp.int32),
        jnp.sum(
            sequences.valid_mask & jnp.isin(sequences.token_codes, ambiguity_codes),
            axis=1,
            dtype=jnp.int32,
        ),
        jnp.sum(sequences.soft_mask, axis=1, dtype=jnp.int32),
        jnp.sum(window_valid & ~reverse_wins, axis=1, dtype=jnp.int32),
        jnp.sum(window_valid & reverse_wins, axis=1, dtype=jnp.int32),
    )
    return MotifScanResult(
        forward,
        reverse,
        scores,
        strand,
        window_valid,
        best_position,
        best_score,
        best_strand,
        any_valid,
        status,
        evidence,
        plan.method_contract,
    )


__all__ = [
    "MOTIF_STATUS_CAPACITY_EXCEEDED",
    "MOTIF_STATUS_NO_VALID_WINDOW",
    "MOTIF_STATUS_VALID",
    "MotifScanEvidence",
    "MotifScanPlan",
    "MotifScanResult",
    "PositionWeightMatrix",
    "scan_motif",
]
