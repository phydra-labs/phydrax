#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._constraints import (
    allowed_pair_matrix,
    RNAConstraints,
    RNAFoldStatus,
    unpaired_allowed,
    validate_sequence_codes,
)
from ._energy_model import RNAEnergyModel


class RNAPartitionResult(StrictModule):
    """Exact log partition, inside/outside tables, and base-pair marginals."""

    log_partition: Array
    pair_marginals: Array
    unpaired_marginals: Array
    inside: Array
    outside: Array
    expected_energy: Array
    entropy: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract
    model_id: str = eqx.field(static=True)
    constraint_id: str = eqx.field(static=True)
    evidence_labels: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        log_partition: Array,
        pair_marginals: Array,
        unpaired_marginals: Array,
        inside: Array,
        outside: Array,
        expected_energy: Array,
        entropy: Array,
        valid: Array,
        status: Array,
        evidence: Array,
        method_contract: BioinformaticsMethodContract,
        model_id: str,
        constraint_id: str,
    ):
        self.log_partition = log_partition
        self.pair_marginals = pair_marginals
        self.unpaired_marginals = unpaired_marginals
        self.inside = inside
        self.outside = outside
        self.expected_energy = expected_energy
        self.entropy = entropy
        self.valid = valid
        self.status = status
        self.evidence = evidence
        self.method_contract = method_contract
        self.model_id = model_id
        self.constraint_id = constraint_id
        self.evidence_labels = (
            "sequence_length",
            "finite_inside_intervals",
            "expected_pair_count",
            "maximum_base_normalization_error",
        )


def _partition_contract(dtype: str) -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "exact-pseudoknot-free-rna-partition",
        MethodKind.EXACT_MODEL,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.EXACT_AD,
        OutputKind.PROBABILISTIC,
        conditioning_statement=(
            "Log-space inside/outside recurrences use log-sum-exp under the declared "
            "positive Boltzmann measure."
        ),
        truncation_statement=(
            "No structures are truncated; the partition sums every noncrossing "
            "partial matching in the declared grammar."
        ),
        capacity_semantics="Inside and outside tables have fixed (L+1, L+1) capacity.",
        assumptions=(
            "Energy is additive over unpaired bases and noncrossing pairs; "
            "temperature and gas constant share declared units.",
        ),
        nondifferentiable_outputs=("status", "valid", "evidence"),
        input_dtype="int32",
        compute_dtype=dtype,
        output_dtype=dtype,
    )


def partition_function(
    sequence_codes: ArrayLike,
    model: RNAEnergyModel,
    constraints: RNAConstraints | None = None,
    /,
) -> RNAPartitionResult:
    """Run exact log-space inside/outside dynamic programming."""

    if not isinstance(model, RNAEnergyModel):
        raise TypeError("model must be an RNAEnergyModel.")
    sequence = jnp.asarray(sequence_codes, dtype=jnp.int32)
    if sequence.ndim != 1:
        raise ValueError("sequence_codes must be a rank-1 vector.")
    length = int(sequence.shape[0])
    constraint = (
        RNAConstraints.unconstrained(length) if constraints is None else constraints
    )
    if not isinstance(constraint, RNAConstraints) or constraint.sequence_length != length:
        raise ValueError("constraints must match sequence length.")
    dtype = model.pair_energies.dtype
    safe = jnp.clip(sequence, 0, model.alphabet_size - 1)
    allowed_pair = allowed_pair_matrix(sequence, model, constraint)
    allowed_single = unpaired_allowed(sequence, model, constraint)
    thermal = model.thermal_energy
    pair_energy = model.pair_energies[
        safe[:, None], safe[None, :]
    ] + constraint.pair_energy_offsets.astype(dtype)
    single_energy = model.unpaired_energies[
        safe
    ] + constraint.unpaired_energy_offsets.astype(dtype)
    negative_infinity = jnp.asarray(-jnp.inf, dtype=dtype)
    log_pair_weight = jnp.where(allowed_pair, -pair_energy / thermal, negative_infinity)
    log_single_weight = jnp.where(
        allowed_single, -single_energy / thermal, negative_infinity
    )

    inside = jnp.full((length + 1, length + 1), negative_infinity, dtype=dtype)
    diagonal = jnp.arange(length + 1, dtype=jnp.int32)
    inside = inside.at[diagonal, diagonal].set(0.0)
    for span in range(1, length + 1):
        for start in range(0, length - span + 1):
            stop = start + span
            terms = [log_single_weight[start] + inside[start + 1, stop]]
            for partner in range(start + 1, stop):
                terms.append(
                    log_pair_weight[start, partner]
                    + inside[start + 1, partner]
                    + inside[partner + 1, stop]
                )
            inside = inside.at[start, stop].set(jax.nn.logsumexp(jnp.stack(terms)))

    outside = jnp.full_like(inside, negative_infinity)
    outside = outside.at[0, length].set(0.0)
    pair_numerator = jnp.full((length, length), negative_infinity, dtype=dtype)
    single_numerator = jnp.full((length,), negative_infinity, dtype=dtype)
    for span in range(length, 0, -1):
        for start in range(0, length - span + 1):
            stop = start + span
            parent = outside[start, stop]
            single_context = parent + log_single_weight[start]
            outside = outside.at[start + 1, stop].set(
                jnp.logaddexp(outside[start + 1, stop], single_context)
            )
            single_numerator = single_numerator.at[start].set(
                jnp.logaddexp(
                    single_numerator[start], single_context + inside[start + 1, stop]
                )
            )
            for partner in range(start + 1, stop):
                context = parent + log_pair_weight[start, partner]
                left_inside = inside[start + 1, partner]
                right_inside = inside[partner + 1, stop]
                pair_numerator = pair_numerator.at[start, partner].set(
                    jnp.logaddexp(
                        pair_numerator[start, partner],
                        context + left_inside + right_inside,
                    )
                )
                outside = outside.at[start + 1, partner].set(
                    jnp.logaddexp(outside[start + 1, partner], context + right_inside)
                )
                outside = outside.at[partner + 1, stop].set(
                    jnp.logaddexp(outside[partner + 1, stop], context + left_inside)
                )

    log_z = inside[0, length]
    sequence_valid = validate_sequence_codes(sequence, model, constraint)
    feasible = jnp.isfinite(log_z)
    status = jnp.where(
        ~sequence_valid,
        int(RNAFoldStatus.INVALID_SEQUENCE),
        jnp.where(
            constraint.contains_required_pseudoknot,
            int(RNAFoldStatus.UNSUPPORTED_PSEUDOKNOT),
            jnp.where(
                feasible,
                int(RNAFoldStatus.SUCCESS),
                int(RNAFoldStatus.INFEASIBLE_CONSTRAINTS),
            ),
        ),
    ).astype(jnp.int32)
    valid = status == int(RNAFoldStatus.SUCCESS)
    upper_pair = jnp.where(
        jnp.isfinite(pair_numerator), jnp.exp(pair_numerator - log_z), 0.0
    )
    pair_marginals = upper_pair + upper_pair.T
    single_marginals = jnp.where(
        jnp.isfinite(single_numerator), jnp.exp(single_numerator - log_z), 0.0
    )
    pair_marginals = jnp.where(valid, pair_marginals, 0.0)
    single_marginals = jnp.where(valid, single_marginals, 0.0)
    expected_pair_energy = jnp.sum(jnp.triu(pair_marginals * pair_energy, 1))
    expected_single_energy = jnp.sum(single_marginals * single_energy)
    expected_energy = jnp.where(
        valid, expected_pair_energy + expected_single_energy, jnp.nan
    )
    entropy = jnp.where(valid, log_z + expected_energy / thermal, jnp.nan)
    normalization = single_marginals + jnp.sum(pair_marginals, axis=1)
    max_error = jnp.max(jnp.abs(normalization - 1.0), initial=0.0)
    evidence = jnp.asarray(
        [
            length,
            jnp.sum(jnp.isfinite(inside)),
            jnp.sum(jnp.triu(pair_marginals, 1)),
            max_error,
        ],
        dtype=dtype,
    )
    return RNAPartitionResult(
        jnp.where(valid, log_z, -jnp.inf),
        pair_marginals,
        single_marginals,
        inside,
        outside,
        expected_energy,
        entropy,
        valid,
        status,
        evidence,
        _partition_contract(np.dtype(dtype).name),
        model.model_id,
        constraint.constraint_id,
    )


def rna_log_partition(
    sequence_codes: ArrayLike,
    model: RNAEnergyModel,
    constraints: RNAConstraints | None = None,
    /,
) -> Array:
    """Return exact log Z; its energy gradients equal negative expected counts/RT."""

    return partition_function(sequence_codes, model, constraints).log_partition


__all__ = ["RNAPartitionResult", "partition_function", "rna_log_partition"]
