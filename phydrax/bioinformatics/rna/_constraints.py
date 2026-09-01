#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._energy_model import RNAEnergyModel


class RNAFoldStatus(IntEnum):
    """Observable status shared by exact and explicitly heuristic RNA folds."""

    SUCCESS = 0
    INVALID_SEQUENCE = 1
    INFEASIBLE_CONSTRAINTS = 2
    UNSUPPORTED_PSEUDOKNOT = 3
    CAPACITY_EXCEEDED = 4
    NONFINITE = 5
    HEURISTIC_RESULT = 6


class RNAConstraints(StrictModule, NonTrainableState):
    """Fixed-length hard pairing constraints and additive position energies."""

    required_partner: Array
    prohibited_pairs: Array
    pair_energy_offsets: Array
    unpaired_energy_offsets: Array
    sequence_length: int = eqx.field(static=True)
    contains_required_pseudoknot: bool = eqx.field(static=True)
    constraint_id: str = eqx.field(static=True)

    def __init__(
        self,
        required_partner: ArrayLike,
        /,
        *,
        prohibited_pairs: ArrayLike | None = None,
        pair_energy_offsets: ArrayLike | None = None,
        unpaired_energy_offsets: ArrayLike | None = None,
    ):
        partner = np.asarray(required_partner)
        if partner.ndim != 1 or not np.issubdtype(partner.dtype, np.integer):
            raise TypeError("required_partner must be an integer vector.")
        partner = partner.astype(np.int32, copy=False)
        length = int(partner.size)
        if np.any(partner < -2) or np.any(partner >= length):
            raise ValueError(
                "required_partner values are -2 free, -1 unpaired, or a valid partner index."
            )
        for index, other in enumerate(partner):
            if other >= 0:
                if int(other) == index or int(partner[int(other)]) != index:
                    raise ValueError("Required pairs must be distinct and reciprocal.")
        prohibited = (
            np.zeros((length, length), dtype=bool)
            if prohibited_pairs is None
            else np.asarray(prohibited_pairs, dtype=bool)
        )
        if prohibited.shape != (length, length) or not np.array_equal(
            prohibited, prohibited.T
        ):
            raise ValueError(
                "prohibited_pairs must be a symmetric (length, length) matrix."
            )
        if np.any(np.diag(prohibited)):
            raise ValueError("prohibited_pairs diagonal must be false.")
        for index, other in enumerate(partner):
            if other >= 0 and prohibited[index, int(other)]:
                raise ValueError("A pair cannot be both required and prohibited.")
        pair_offset = (
            np.zeros((length, length), dtype=np.float64)
            if pair_energy_offsets is None
            else np.asarray(pair_energy_offsets)
        )
        if pair_offset.shape != (length, length) or not np.issubdtype(
            pair_offset.dtype, np.inexact
        ):
            raise TypeError(
                "pair_energy_offsets must be an inexact (length, length) matrix."
            )
        if np.any(~np.isfinite(pair_offset)) or not np.allclose(
            pair_offset, pair_offset.T, atol=0.0, rtol=0.0
        ):
            raise ValueError("pair_energy_offsets must be finite and symmetric.")
        unpaired_offset = (
            np.zeros((length,), dtype=pair_offset.dtype)
            if unpaired_energy_offsets is None
            else np.asarray(unpaired_energy_offsets, dtype=pair_offset.dtype)
        )
        if unpaired_offset.shape != (length,) or np.any(~np.isfinite(unpaired_offset)):
            raise ValueError("unpaired_energy_offsets must be a finite length vector.")
        pairs = [
            (index, int(other)) for index, other in enumerate(partner) if other > index
        ]
        crossing = any(
            i < k < j < l or k < i < l < j
            for pair_index, (i, j) in enumerate(pairs)
            for k, l in pairs[pair_index + 1 :]
        )
        self.required_partner = jnp.asarray(partner)
        self.prohibited_pairs = jnp.asarray(prohibited)
        self.pair_energy_offsets = jnp.asarray(pair_offset)
        self.unpaired_energy_offsets = jnp.asarray(unpaired_offset)
        self.sequence_length = length
        self.contains_required_pseudoknot = bool(crossing)
        self.constraint_id = canonical_fingerprint(
            {
                "kind": "rna-fold-constraints",
                "arrays": array_tree_fingerprint(
                    {
                        "required_partner": partner,
                        "prohibited_pairs": prohibited,
                        "pair_energy_offsets": pair_offset,
                        "unpaired_energy_offsets": unpaired_offset,
                    }
                ),
                "contains_required_pseudoknot": bool(crossing),
            }
        )

    @classmethod
    def unconstrained(cls, sequence_length: int) -> "RNAConstraints":
        length = int(sequence_length)
        if length < 0:
            raise ValueError("sequence_length must be non-negative.")
        return cls(np.full((length,), -2, dtype=np.int32))


def validate_sequence_codes(
    sequence_codes: ArrayLike,
    model: RNAEnergyModel,
    constraints: RNAConstraints | None = None,
) -> Array:
    """Return scalar validity without forcing traced sequence data onto the host."""

    sequence = jnp.asarray(sequence_codes)
    if sequence.ndim != 1 or not jnp.issubdtype(sequence.dtype, jnp.integer):
        raise TypeError("sequence_codes must be an integer vector.")
    if constraints is not None and constraints.sequence_length != sequence.shape[0]:
        raise ValueError("constraints length must match sequence_codes.")
    return jnp.all((sequence >= 0) & (sequence < model.alphabet_size))


def allowed_pair_matrix(
    sequence_codes: ArrayLike, model: RNAEnergyModel, constraints: RNAConstraints
) -> Array:
    """Resolve sequence, grammar, hard constraints, and loop separation."""

    sequence = jnp.asarray(sequence_codes, dtype=jnp.int32)
    if sequence.shape != (constraints.sequence_length,):
        raise ValueError("sequence_codes must match the constraints length.")
    safe = jnp.clip(sequence, 0, model.alphabet_size - 1)
    allowed = model.allowed_pairs[safe[:, None], safe[None, :]]
    index = jnp.arange(constraints.sequence_length, dtype=jnp.int32)
    separation = index[None, :] - index[:, None]
    allowed = allowed & (separation > model.minimum_hairpin_length)
    free_or_other = (constraints.required_partner[:, None] == -2) | (
        constraints.required_partner[:, None] == index[None, :]
    )
    partner_accepts = (constraints.required_partner[None, :] == -2) | (
        constraints.required_partner[None, :] == index[:, None]
    )
    allowed = allowed & free_or_other & partner_accepts & ~constraints.prohibited_pairs
    return allowed & (sequence[:, None] >= 0) & (sequence[None, :] >= 0)


def unpaired_allowed(
    sequence_codes: ArrayLike, model: RNAEnergyModel, constraints: RNAConstraints
) -> Array:
    sequence = jnp.asarray(sequence_codes)
    return (
        (constraints.required_partner < 0)
        & (constraints.required_partner != jnp.arange(constraints.sequence_length))
        & (sequence >= 0)
        & (sequence < model.alphabet_size)
    )


__all__ = [
    "RNAConstraints",
    "RNAFoldStatus",
    "allowed_pair_matrix",
    "unpaired_allowed",
    "validate_sequence_codes",
]
