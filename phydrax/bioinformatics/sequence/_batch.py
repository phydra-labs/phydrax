#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.core as jax_core
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax._strict import StrictModule
from phydrax.bioinformatics.sequence._alphabet import AlphabetPlan


def _integer_vector(values: ArrayLike, name: str) -> Array:
    array = jnp.asarray(values)
    if array.ndim != 1 or not jnp.issubdtype(array.dtype, jnp.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array.")
    return array


def _concrete(array: Array) -> np.ndarray | None:
    if isinstance(array, jax_core.Tracer):
        return None
    return np.asarray(array)


def _case_mask(values: ArrayLike, size: int) -> Array:
    mask = jnp.asarray(values)
    if mask.dtype != jnp.bool_ or mask.shape != (size,):
        raise ValueError("case_mask must be boolean with shape (record_capacity,).")
    return mask


class SequenceBatch(StrictModule):
    """Bounded encoded sequences with explicit record, position, and soft masks."""

    record_ids: Array
    token_codes: Array
    valid_mask: Array
    case_mask: Array
    soft_mask: Array
    alphabet: AlphabetPlan = eqx.field(static=True)

    def __init__(
        self,
        record_ids: ArrayLike,
        token_codes: ArrayLike,
        valid_mask: ArrayLike,
        case_mask: ArrayLike,
        soft_mask: ArrayLike,
        alphabet: AlphabetPlan,
    ):
        if not isinstance(alphabet, AlphabetPlan):
            raise TypeError("alphabet must be an AlphabetPlan.")
        ids = _integer_vector(record_ids, "record_ids")
        tokens = jnp.asarray(token_codes)
        if tokens.ndim != 2 or not jnp.issubdtype(tokens.dtype, jnp.integer):
            raise ValueError("token_codes must be a two-dimensional integer array.")
        if tokens.shape[0] != ids.shape[0]:
            raise ValueError("record_ids and token_codes record dimensions must match.")
        cases = _case_mask(case_mask, int(ids.shape[0]))
        valid = jnp.asarray(valid_mask)
        soft = jnp.asarray(soft_mask)
        if valid.dtype != jnp.bool_ or valid.shape != tokens.shape:
            raise ValueError("valid_mask must be boolean with token_codes shape.")
        if soft.dtype != jnp.bool_ or soft.shape != tokens.shape:
            raise ValueError("soft_mask must be boolean with token_codes shape.")

        concrete_tokens = _concrete(tokens)
        concrete_valid = _concrete(valid)
        concrete_cases = _concrete(cases)
        concrete_soft = _concrete(soft)
        if concrete_tokens is not None and (
            np.any(concrete_tokens < 0) or np.any(concrete_tokens >= alphabet.size)
        ):
            raise ValueError("token_codes contains a code outside the alphabet.")
        if concrete_valid is not None:
            if np.any(concrete_valid[:, 1:] & ~concrete_valid[:, :-1]):
                raise ValueError("valid_mask must be a left-aligned prefix per record.")
            if concrete_cases is not None and np.any(
                concrete_valid & ~concrete_cases[:, None]
            ):
                raise ValueError(
                    "Padded records cannot contain valid sequence positions."
                )
            pad_code = alphabet.code(alphabet.pad_symbol)
            if concrete_tokens is not None and np.any(
                concrete_tokens[~concrete_valid] != pad_code
            ):
                raise ValueError("Invalid positions must contain the alphabet pad code.")
            if concrete_soft is not None and np.any(concrete_soft & ~concrete_valid):
                raise ValueError("soft_mask cannot mark an invalid position.")

        self.record_ids = ids
        self.token_codes = tokens
        self.valid_mask = valid
        self.case_mask = cases
        self.soft_mask = soft
        self.alphabet = alphabet

    @property
    def record_capacity(self) -> int:
        return int(self.token_codes.shape[0])

    @property
    def sequence_capacity(self) -> int:
        return int(self.token_codes.shape[1])

    @property
    def capacity(self) -> int:
        """Return the fixed per-record sequence width."""
        return self.sequence_capacity

    @property
    def record_count(self) -> int:
        """Return the fixed number of record slots."""
        return self.record_capacity

    @property
    def case_count(self) -> Array:
        """Return the dynamic number of populated record slots."""
        return jnp.sum(self.case_mask, dtype=jnp.int32)

    @property
    def lengths(self) -> Array:
        return jnp.sum(self.valid_mask, axis=1, dtype=jnp.int32)


class SequenceDistribution(StrictModule):
    """Per-position categorical sequence laws, separate from discrete batches."""

    record_ids: Array
    probabilities: Array
    valid_mask: Array
    case_mask: Array
    alphabet: AlphabetPlan = eqx.field(static=True)

    def __init__(
        self,
        record_ids: ArrayLike,
        probabilities: ArrayLike,
        valid_mask: ArrayLike,
        case_mask: ArrayLike,
        alphabet: AlphabetPlan,
    ):
        if not isinstance(alphabet, AlphabetPlan):
            raise TypeError("alphabet must be an AlphabetPlan.")
        ids = _integer_vector(record_ids, "record_ids")
        values = jnp.asarray(probabilities)
        if values.ndim != 3 or not jnp.issubdtype(values.dtype, jnp.floating):
            raise ValueError("probabilities must be a three-dimensional floating array.")
        if values.shape[0] != ids.shape[0] or values.shape[2] != alphabet.size:
            raise ValueError(
                "probabilities must have shape (record, position, alphabet.size)."
            )
        cases = _case_mask(case_mask, int(ids.shape[0]))
        valid = jnp.asarray(valid_mask)
        if valid.dtype != jnp.bool_ or valid.shape != values.shape[:2]:
            raise ValueError(
                "valid_mask must be boolean with probabilities leading shape."
            )

        concrete_values = _concrete(values)
        concrete_valid = _concrete(valid)
        concrete_cases = _concrete(cases)
        if concrete_valid is not None:
            if np.any(concrete_valid[:, 1:] & ~concrete_valid[:, :-1]):
                raise ValueError("valid_mask must be a left-aligned prefix per record.")
            if concrete_cases is not None and np.any(
                concrete_valid & ~concrete_cases[:, None]
            ):
                raise ValueError("Padded records cannot contain valid distributions.")
        if concrete_values is not None:
            if np.any(~np.isfinite(concrete_values)) or np.any(concrete_values < 0.0):
                raise ValueError("probabilities must be finite and non-negative.")
            sums = concrete_values.sum(axis=-1)
            if concrete_valid is not None:
                if not np.allclose(sums[concrete_valid], 1.0):
                    raise ValueError("Valid categorical distributions must sum to one.")
                if np.any(concrete_values[~concrete_valid] != 0.0):
                    raise ValueError("Invalid distribution positions must be zero.")

        self.record_ids = ids
        self.probabilities = values
        self.valid_mask = valid
        self.case_mask = cases
        self.alphabet = alphabet

    @property
    def lengths(self) -> Array:
        return jnp.sum(self.valid_mask, axis=1, dtype=jnp.int32)


__all__ = ["SequenceBatch", "SequenceDistribution"]
