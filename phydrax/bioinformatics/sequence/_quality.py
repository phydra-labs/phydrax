#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import Enum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax._strict import StrictModule
from phydrax.bioinformatics.sequence._batch import (
    _case_mask,
    _concrete,
    _integer_vector,
)


class PhredEncoding(str, Enum):
    """Printable ASCII offset used to interchange integer Phred scores."""

    PHRED33 = "phred33"
    PHRED64 = "phred64"

    @property
    def offset(self) -> int:
        return 33 if self is PhredEncoding.PHRED33 else 64

    @property
    def maximum_score(self) -> int:
        return 126 - self.offset


class QualityBatch(StrictModule):
    """Bounded Phred scores with explicit record/presence and interchange encoding."""

    record_ids: Array
    phred_scores: Array
    valid_mask: Array
    case_mask: Array
    phred_encoding: PhredEncoding = eqx.field(static=True)

    def __init__(
        self,
        record_ids: ArrayLike,
        phred_scores: ArrayLike,
        valid_mask: ArrayLike,
        case_mask: ArrayLike,
        phred_encoding: PhredEncoding | str,
    ):
        encoding = PhredEncoding(phred_encoding)
        ids = _integer_vector(record_ids, "record_ids")
        scores = jnp.asarray(phred_scores)
        if scores.ndim != 2 or not jnp.issubdtype(scores.dtype, jnp.integer):
            raise ValueError("phred_scores must be a two-dimensional integer array.")
        if scores.shape[0] != ids.shape[0]:
            raise ValueError("record_ids and phred_scores record dimensions must match.")
        cases = _case_mask(case_mask, int(ids.shape[0]))
        valid = jnp.asarray(valid_mask)
        if valid.dtype != jnp.bool_ or valid.shape != scores.shape:
            raise ValueError("valid_mask must be boolean with phred_scores shape.")

        concrete_scores = _concrete(scores)
        concrete_valid = _concrete(valid)
        concrete_cases = _concrete(cases)
        if concrete_valid is not None:
            if np.any(concrete_valid[:, 1:] & ~concrete_valid[:, :-1]):
                raise ValueError("valid_mask must be a left-aligned prefix per record.")
            if concrete_cases is not None and np.any(
                concrete_valid & ~concrete_cases[:, None]
            ):
                raise ValueError("Padded records cannot contain quality scores.")
            if concrete_scores is not None:
                if np.any(concrete_scores[~concrete_valid] != 0):
                    raise ValueError("Absent quality positions must contain zero.")
                present = concrete_scores[concrete_valid]
                if np.any(present < 0) or np.any(present > encoding.maximum_score):
                    raise ValueError(
                        f"Scores for {encoding.value} must be between 0 and "
                        f"{encoding.maximum_score}."
                    )

        self.record_ids = ids
        self.phred_scores = scores
        self.valid_mask = valid
        self.case_mask = cases
        self.phred_encoding = encoding

    @property
    def lengths(self) -> Array:
        return jnp.sum(self.valid_mask, axis=1, dtype=jnp.int32)


__all__ = ["PhredEncoding", "QualityBatch"]
