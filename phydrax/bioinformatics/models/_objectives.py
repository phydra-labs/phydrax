#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from phydrax._strict import StrictModule

from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ..sequence import SequenceBatch


class ObjectiveStatus(IntEnum):
    """Array-valued objective status codes safe to return from compiled code."""

    SUCCESS = 0
    EMPTY_SUPPORT = 1
    NONFINITE = 2


_OBJECTIVE_CONTRACT = BioinformaticsMethodContract(
    "biological-supervision-objective",
    MethodKind.EXACT_MODEL,
    ExecutionKind.FLOATING_POINT_DIRECT,
    DifferentiationKind.EXACT_AD,
    OutputKind.SCALAR,
    conditioning_statement="Log-softmax and logistic losses use their stable JAX primitives.",
    truncation_statement="No examples or positions are truncated.",
    capacity_semantics="All supplied dense logits are evaluated; masks define support only.",
    assumptions=("Targets are integer class codes on active positions.",),
    input_dtype="integer targets and real logits",
    compute_dtype="logit dtype",
    output_dtype="logit dtype",
)


class TokenPrediction(StrictModule):
    """Token logits carrying exact alphabet and tokenizer identities."""

    logits: Array
    valid_mask: Array
    valid: Array
    status: Array
    evidence: Array
    alphabet_fingerprint: str = eqx.field(static=True)
    tokenizer_fingerprint: str = eqx.field(static=True)
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)

    def __init__(
        self,
        logits: Array,
        valid_mask: Array,
        *,
        alphabet_fingerprint: str,
        tokenizer_fingerprint: str,
        method_contract: BioinformaticsMethodContract,
    ):
        values = jnp.asarray(logits)
        mask = jnp.asarray(valid_mask, dtype=bool)
        if values.ndim != 3:
            raise ValueError("Token logits must have shape (batch, length, classes).")
        if mask.shape != values.shape[:2]:
            raise ValueError("Token prediction mask must match logits batch and length.")
        if not jnp.issubdtype(values.dtype, jnp.floating):
            raise TypeError("Token logits must have a real floating dtype.")
        self.logits = values
        self.valid_mask = mask
        finite = jnp.all(jnp.isfinite(values) | ~mask[..., None])
        support = jnp.any(mask)
        self.valid = finite & support
        self.status = jnp.where(
            ~support,
            jnp.asarray(ObjectiveStatus.EMPTY_SUPPORT, dtype=jnp.int32),
            jnp.where(
                finite,
                jnp.asarray(ObjectiveStatus.SUCCESS, dtype=jnp.int32),
                jnp.asarray(ObjectiveStatus.NONFINITE, dtype=jnp.int32),
            ),
        )
        self.evidence = jnp.stack(
            (jnp.sum(mask, dtype=jnp.int32), jnp.asarray(mask.size, dtype=jnp.int32))
        )
        self.alphabet_fingerprint = str(alphabet_fingerprint)
        self.tokenizer_fingerprint = str(tokenizer_fingerprint)
        self.method_contract = method_contract


class TokenLabelPrediction(StrictModule):
    """Per-token categorical logits with provenance and padding support."""

    logits: Array
    valid_mask: Array
    valid: Array
    status: Array
    evidence: Array
    label_space_id: str = eqx.field(static=True)
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)

    def __init__(
        self,
        logits: Array,
        valid_mask: Array,
        *,
        label_space_id: str,
        method_contract: BioinformaticsMethodContract,
    ):
        values = jnp.asarray(logits)
        mask = jnp.asarray(valid_mask, dtype=bool)
        if values.ndim != 3 or values.shape[:2] != mask.shape:
            raise ValueError("Token-label logits require shape (batch, length, labels).")
        finite = jnp.all(jnp.isfinite(values) | ~mask[..., None])
        support = jnp.any(mask)
        self.logits = values
        self.valid_mask = mask
        self.valid = finite & support
        self.status = jnp.where(
            ~support,
            jnp.asarray(ObjectiveStatus.EMPTY_SUPPORT, dtype=jnp.int32),
            jnp.where(finite, 0, jnp.asarray(ObjectiveStatus.NONFINITE, jnp.int32)),
        )
        self.evidence = jnp.stack(
            (jnp.sum(mask, dtype=jnp.int32), jnp.asarray(mask.size, dtype=jnp.int32))
        )
        self.label_space_id = str(label_space_id)
        self.method_contract = method_contract


class PairPrediction(StrictModule):
    """Dense ordered-pair logits with explicit pair support."""

    logits: Array
    valid_mask: Array
    valid: Array
    status: Array
    evidence: Array
    pair_space_id: str = eqx.field(static=True)
    symmetric: bool = eqx.field(static=True)
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)

    def __init__(
        self,
        logits: Array,
        valid_mask: Array,
        *,
        pair_space_id: str,
        symmetric: bool,
        method_contract: BioinformaticsMethodContract,
    ):
        values = jnp.asarray(logits)
        mask = jnp.asarray(valid_mask, dtype=bool)
        if values.ndim not in (3, 4):
            raise ValueError(
                "Pair logits must have shape (batch, length, length[, classes])."
            )
        if values.shape[:3] != mask.shape or mask.ndim != 3:
            raise ValueError("Pair support must match the first three logits axes.")
        if int(values.shape[1]) != int(values.shape[2]):
            raise ValueError("Pair prediction sequence axes must be square.")
        if symmetric:
            values = 0.5 * (values + jnp.swapaxes(values, 1, 2))
            mask = mask & jnp.swapaxes(mask, 1, 2)
        finite_values = jnp.isfinite(values)
        finite = jnp.all(
            finite_values | ~mask.reshape(mask.shape + (1,) * (values.ndim - 3))
        )
        support = jnp.any(mask)
        self.logits = values
        self.valid_mask = mask
        self.valid = finite & support
        self.status = jnp.where(
            ~support,
            jnp.asarray(ObjectiveStatus.EMPTY_SUPPORT, dtype=jnp.int32),
            jnp.where(finite, 0, jnp.asarray(ObjectiveStatus.NONFINITE, jnp.int32)),
        )
        self.evidence = jnp.stack(
            (jnp.sum(mask, dtype=jnp.int32), jnp.asarray(mask.size, dtype=jnp.int32))
        )
        self.pair_space_id = str(pair_space_id)
        self.symmetric = bool(symmetric)
        self.method_contract = method_contract


class ObjectiveResult(StrictModule):
    """A masked scalar objective and its full unreduced numerical evidence."""

    loss: Array
    element_loss: Array
    active_mask: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)

    def __init__(self, element_loss: Array, active_mask: Array):
        losses = jnp.asarray(element_loss)
        active = jnp.asarray(active_mask, dtype=bool)
        if losses.shape != active.shape:
            raise ValueError("Element losses and active mask must have identical shapes.")
        count = jnp.sum(active, dtype=jnp.int32)
        finite = jnp.all(jnp.isfinite(losses) | ~active)
        total = jnp.sum(jnp.where(active, losses, 0.0))
        mean = jnp.where(count > 0, total / count.astype(losses.dtype), jnp.nan)
        self.loss = mean
        self.element_loss = jnp.where(active, losses, 0.0)
        self.active_mask = active
        self.valid = (count > 0) & finite
        self.status = jnp.where(
            count == 0,
            jnp.asarray(ObjectiveStatus.EMPTY_SUPPORT, dtype=jnp.int32),
            jnp.where(
                finite,
                jnp.asarray(ObjectiveStatus.SUCCESS, dtype=jnp.int32),
                jnp.asarray(ObjectiveStatus.NONFINITE, dtype=jnp.int32),
            ),
        )
        self.evidence = jnp.stack((count, jnp.asarray(active.size, dtype=jnp.int32)))
        self.method_contract = _OBJECTIVE_CONTRACT


def _check_token_compatibility(
    prediction: TokenPrediction,
    targets: SequenceBatch,
    expected_tokenizer_fingerprint: str,
) -> None:
    if not isinstance(prediction, TokenPrediction):
        raise TypeError("prediction must be a TokenPrediction.")
    if not isinstance(targets, SequenceBatch):
        raise TypeError("targets must be a SequenceBatch.")
    if prediction.alphabet_fingerprint != targets.alphabet.fingerprint:
        raise ValueError("Prediction and target alphabet fingerprints differ.")
    if prediction.tokenizer_fingerprint != str(expected_tokenizer_fingerprint):
        raise ValueError("Prediction tokenizer fingerprint does not match the objective.")
    if int(prediction.logits.shape[-1]) != targets.alphabet.size:
        raise ValueError("Prediction vocabulary size does not match the target alphabet.")


def _categorical_losses(logits: Array, targets: Array, active_mask: Array, /) -> Array:
    values = jnp.asarray(logits)
    codes = jnp.asarray(targets)
    active = jnp.asarray(active_mask, dtype=bool)
    if not jnp.issubdtype(codes.dtype, jnp.integer):
        raise TypeError("Categorical targets must have integer dtype.")
    if codes.shape != values.shape[:-1] or active.shape != codes.shape:
        raise ValueError("Categorical targets and support must match logits.")
    invalid = active & ((codes < 0) | (codes >= int(values.shape[-1])))
    codes = eqx.error_if(
        codes,
        jnp.any(invalid),
        "Active categorical target is outside the prediction class space.",
    )
    safe_codes = jnp.where(active, codes, 0)
    return -jnp.take_along_axis(
        jax.nn.log_softmax(values, axis=-1), safe_codes[..., None], axis=-1
    )[..., 0]


class MaskedTokenObjective(StrictModule):
    """Cross entropy evaluated only at explicitly selected valid tokens."""

    tokenizer_fingerprint: str = eqx.field(static=True)

    def __init__(self, tokenizer_fingerprint: str):
        identifier = str(tokenizer_fingerprint)
        if not identifier:
            raise ValueError("tokenizer_fingerprint must be non-empty.")
        self.tokenizer_fingerprint = identifier

    def __call__(
        self,
        prediction: TokenPrediction,
        targets: SequenceBatch,
        prediction_mask: Array,
        /,
    ) -> ObjectiveResult:
        _check_token_compatibility(prediction, targets, self.tokenizer_fingerprint)
        selected = jnp.asarray(prediction_mask, dtype=bool)
        if selected.shape != targets.valid_mask.shape:
            raise ValueError("prediction_mask must match the target token shape.")
        if prediction.logits.shape[:2] != targets.token_codes.shape:
            raise ValueError("Prediction and target token shapes differ.")
        active = selected & targets.valid_mask & prediction.valid_mask
        return ObjectiveResult(
            _categorical_losses(prediction.logits, targets.token_codes, active),
            active,
        )


class CausalTokenObjective(StrictModule):
    """Next-token cross entropy excluding padding and sequence starts."""

    tokenizer_fingerprint: str = eqx.field(static=True)

    def __init__(self, tokenizer_fingerprint: str):
        identifier = str(tokenizer_fingerprint)
        if not identifier:
            raise ValueError("tokenizer_fingerprint must be non-empty.")
        self.tokenizer_fingerprint = identifier

    def __call__(
        self, prediction: TokenPrediction, targets: SequenceBatch, /
    ) -> ObjectiveResult:
        _check_token_compatibility(prediction, targets, self.tokenizer_fingerprint)
        length = int(targets.token_codes.shape[1])
        if length < 2:
            empty = jnp.zeros(
                (targets.token_codes.shape[0], 0), dtype=prediction.logits.dtype
            )
            return ObjectiveResult(empty, jnp.zeros_like(empty, dtype=bool))
        if prediction.logits.shape[:2] == targets.token_codes.shape:
            logits = prediction.logits[:, :-1]
            predictor_valid = prediction.valid_mask[:, :-1]
        elif prediction.logits.shape[:2] == (targets.token_codes.shape[0], length - 1):
            logits = prediction.logits
            predictor_valid = prediction.valid_mask
        else:
            raise ValueError(
                "Causal logits must have target length or target length minus one."
            )
        active = predictor_valid & targets.valid_mask[:, :-1] & targets.valid_mask[:, 1:]
        return ObjectiveResult(
            _categorical_losses(logits, targets.token_codes[:, 1:], active),
            active,
        )


class TokenLabelObjective(StrictModule):
    """Masked per-token categorical supervision in a named label space."""

    label_space_id: str = eqx.field(static=True)

    def __init__(self, label_space_id: str):
        self.label_space_id = str(label_space_id)

    def __call__(
        self,
        prediction: TokenLabelPrediction,
        labels: Array,
        label_mask: Array | None = None,
        /,
    ) -> ObjectiveResult:
        if prediction.label_space_id != self.label_space_id:
            raise ValueError("Prediction and objective label spaces differ.")
        targets = jnp.asarray(labels)
        if targets.shape != prediction.valid_mask.shape:
            raise ValueError("Token labels must match prediction batch and length.")
        active = prediction.valid_mask
        if label_mask is not None:
            selected = jnp.asarray(label_mask, dtype=bool)
            if selected.shape != active.shape:
                raise ValueError("label_mask must match token labels.")
            active = active & selected
        return ObjectiveResult(
            _categorical_losses(prediction.logits, targets, active), active
        )


class PairObjective(StrictModule):
    """Masked categorical supervision on ordered or symmetric token pairs."""

    pair_space_id: str = eqx.field(static=True)

    def __init__(self, pair_space_id: str):
        self.pair_space_id = str(pair_space_id)

    def __call__(
        self,
        prediction: PairPrediction,
        labels: Array,
        pair_mask: Array | None = None,
        /,
    ) -> ObjectiveResult:
        if prediction.pair_space_id != self.pair_space_id:
            raise ValueError("Prediction and objective pair spaces differ.")
        if prediction.logits.ndim != 4:
            raise ValueError("Categorical pair objectives require a class axis.")
        targets = jnp.asarray(labels)
        if targets.shape != prediction.valid_mask.shape:
            raise ValueError("Pair labels must match pair support.")
        active = prediction.valid_mask
        if pair_mask is not None:
            selected = jnp.asarray(pair_mask, dtype=bool)
            if selected.shape != active.shape:
                raise ValueError("pair_mask must match pair labels.")
            active = active & selected
        return ObjectiveResult(
            _categorical_losses(prediction.logits, targets, active), active
        )


class ContactObjective(StrictModule):
    """Binary contact loss on unique off-diagonal residue pairs."""

    positive_weight: float = eqx.field(static=True)
    pair_space_id: str = eqx.field(static=True)

    def __init__(self, *, positive_weight: float = 1.0, pair_space_id: str = "contact"):
        weight = float(positive_weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("positive_weight must be finite and positive.")
        self.positive_weight = weight
        self.pair_space_id = str(pair_space_id)

    def __call__(
        self,
        prediction: PairPrediction,
        contacts: Array,
        pair_mask: Array | None = None,
        /,
    ) -> ObjectiveResult:
        if prediction.pair_space_id != self.pair_space_id:
            raise ValueError("Prediction and objective contact spaces differ.")
        logits = prediction.logits
        if logits.ndim == 4:
            if int(logits.shape[-1]) != 1:
                raise ValueError("Contact predictions require scalar logits.")
            logits = logits[..., 0]
        targets = jnp.asarray(contacts)
        if targets.shape != prediction.valid_mask.shape:
            raise ValueError("Contact targets must match pair support.")
        targets = targets.astype(logits.dtype)
        length = int(logits.shape[1])
        unique = jnp.triu(jnp.ones((length, length), dtype=bool), k=1)[None, ...]
        active = prediction.valid_mask & unique
        if pair_mask is not None:
            selected = jnp.asarray(pair_mask, dtype=bool)
            if selected.shape != active.shape:
                raise ValueError("pair_mask must match contact targets.")
            active = active & selected
        losses = (
            jnp.maximum(logits, 0.0)
            - logits * targets
            + jnp.log1p(jnp.exp(-jnp.abs(logits)))
        )
        weights = jnp.where(targets > 0.5, self.positive_weight, 1.0)
        return ObjectiveResult(losses * weights, active)


__all__ = [
    "CausalTokenObjective",
    "ContactObjective",
    "MaskedTokenObjective",
    "ObjectiveResult",
    "ObjectiveStatus",
    "PairObjective",
    "PairPrediction",
    "TokenLabelObjective",
    "TokenLabelPrediction",
    "TokenPrediction",
]
