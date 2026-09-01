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


class RNAMFEResult(StrictModule):
    """Exact minimum-energy noncrossing partial matching for the declared grammar."""

    energy: Array
    pair_table: Array
    paired_matrix: Array
    dynamic_program: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract
    model_id: str = eqx.field(static=True)
    constraint_id: str = eqx.field(static=True)
    evidence_labels: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        energy: Array,
        pair_table: Array,
        paired_matrix: Array,
        dynamic_program: Array,
        valid: Array,
        status: Array,
        evidence: Array,
        method_contract: BioinformaticsMethodContract,
        model_id: str,
        constraint_id: str,
    ):
        self.energy = energy
        self.pair_table = pair_table
        self.paired_matrix = paired_matrix
        self.dynamic_program = dynamic_program
        self.valid = valid
        self.status = status
        self.evidence = evidence
        self.method_contract = method_contract
        self.model_id = model_id
        self.constraint_id = constraint_id
        self.evidence_labels = (
            "sequence_length",
            "finite_interval_count",
            "pair_count",
            "minimum_energy",
        )


def _mfe_contract(dtype: str) -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "exact-pseudoknot-free-rna-mfe",
        MethodKind.EXACT_MODEL,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.ALMOST_EVERYWHERE,
        OutputKind.STRUCTURED,
        conditioning_statement=(
            "Finite additive energies are minimized by an exact interval dynamic "
            "program; ties use leftmost deterministic order."
        ),
        truncation_statement=(
            "No structures are truncated; every noncrossing partial matching in "
            "the declared grammar is represented."
        ),
        capacity_semantics="The dynamic program has fixed (L+1, L+1) capacity for input length L.",
        assumptions=(
            "Energy is additive over unpaired bases and noncrossing pairs; no loop or stacking term is implied.",
        ),
        nondifferentiable_outputs=("pair_table", "paired_matrix", "status", "evidence"),
        input_dtype="int32",
        compute_dtype=dtype,
        output_dtype=dtype,
    )


def minimum_free_energy(
    sequence_codes: ArrayLike,
    model: RNAEnergyModel,
    constraints: RNAConstraints | None = None,
    /,
) -> RNAMFEResult:
    """Solve the exact pseudoknot-free additive grammar by Nussinov DP."""

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
        raise ValueError(
            "constraints must be RNAConstraints matching the sequence length."
        )
    dtype = model.pair_energies.dtype
    safe = jnp.clip(sequence, 0, model.alphabet_size - 1)
    pair_allowed = allowed_pair_matrix(sequence, model, constraint)
    single_allowed = unpaired_allowed(sequence, model, constraint)
    pair_energy = model.pair_energies[
        safe[:, None], safe[None, :]
    ] + constraint.pair_energy_offsets.astype(dtype)
    single_energy = model.unpaired_energies[
        safe
    ] + constraint.unpaired_energy_offsets.astype(dtype)
    infinity = jnp.asarray(jnp.inf, dtype=dtype)
    table = jnp.full((length + 1, length + 1), infinity, dtype=dtype)
    diagonal = jnp.arange(length + 1, dtype=jnp.int32)
    table = table.at[diagonal, diagonal].set(0.0)
    decision = jnp.full((length + 1, length + 1), -3, dtype=jnp.int32)
    for span in range(1, length + 1):
        for start in range(0, length - span + 1):
            stop = start + span
            candidates = [
                jnp.where(
                    single_allowed[start],
                    single_energy[start] + table[start + 1, stop],
                    infinity,
                )
            ]
            choices = [-1]
            for partner in range(start + 1, stop):
                candidates.append(
                    jnp.where(
                        pair_allowed[start, partner],
                        pair_energy[start, partner]
                        + table[start + 1, partner]
                        + table[partner + 1, stop],
                        infinity,
                    )
                )
                choices.append(partner)
            values = jnp.stack(candidates)
            best = jnp.argmin(values)
            minimum = values[best]
            table = table.at[start, stop].set(minimum)
            choice_array = jnp.asarray(choices, dtype=jnp.int32)
            decision = decision.at[start, stop].set(
                jnp.where(jnp.isfinite(minimum), choice_array[best], -3)
            )

    pair_table = jnp.full((length,), -1, dtype=jnp.int32)
    if length:
        stack_start = jnp.zeros((length + 1,), dtype=jnp.int32).at[0].set(0)
        stack_stop = jnp.zeros((length + 1,), dtype=jnp.int32).at[0].set(length)

        def push_interval(
            state: tuple[Array, Array, Array, Array],
            interval_start: Array,
            interval_stop: Array,
        ) -> tuple[Array, Array, Array, Array]:
            starts, stops, top, pairs = state
            starts = starts.at[top].set(interval_start)
            stops = stops.at[top].set(interval_stop)
            return starts, stops, top + 1, pairs

        def cond(state: tuple[Array, Array, Array, Array, Array]) -> Array:
            _, _, top, _, steps = state
            return (top > 0) & (steps < 2 * length + 1)

        def body(
            state: tuple[Array, Array, Array, Array, Array],
        ) -> tuple[Array, Array, Array, Array, Array]:
            starts, stops, top, pairs, steps = state
            top = top - 1
            start = starts[top]
            stop = stops[top]
            choice = decision[start, stop]
            paired = choice >= 0
            pairs = jax.lax.cond(
                paired,
                lambda value: value.at[start].set(choice).at[choice].set(start),
                lambda value: value,
                pairs,
            )
            base_state = (starts, stops, top, pairs)
            base_state = jax.lax.cond(
                (choice == -1) & (start + 1 < stop),
                lambda value: push_interval(value, start + 1, stop),
                lambda value: value,
                base_state,
            )
            base_state = jax.lax.cond(
                paired & (choice + 1 < stop),
                lambda value: push_interval(value, choice + 1, stop),
                lambda value: value,
                base_state,
            )
            base_state = jax.lax.cond(
                paired & (start + 1 < choice),
                lambda value: push_interval(value, start + 1, choice),
                lambda value: value,
                base_state,
            )
            return (*base_state, steps + 1)

        _, _, _, pair_table, _ = jax.lax.while_loop(
            cond,
            body,
            (
                stack_start,
                stack_stop,
                jnp.asarray(1, dtype=jnp.int32),
                pair_table,
                jnp.asarray(0, dtype=jnp.int32),
            ),
        )
    sequence_valid = validate_sequence_codes(sequence, model, constraint)
    feasible = jnp.isfinite(table[0, length])
    unsupported = constraint.contains_required_pseudoknot
    status = jnp.where(
        ~sequence_valid,
        int(RNAFoldStatus.INVALID_SEQUENCE),
        jnp.where(
            unsupported,
            int(RNAFoldStatus.UNSUPPORTED_PSEUDOKNOT),
            jnp.where(
                feasible,
                int(RNAFoldStatus.SUCCESS),
                int(RNAFoldStatus.INFEASIBLE_CONSTRAINTS),
            ),
        ),
    ).astype(jnp.int32)
    valid = status == int(RNAFoldStatus.SUCCESS)
    pair_table = jnp.where(valid, pair_table, -1)
    paired_matrix = (
        pair_table[:, None] == jnp.arange(length, dtype=jnp.int32)[None, :]
    ) & valid
    energy = jnp.where(valid, table[0, length], jnp.inf)
    evidence = jnp.asarray(
        [length, jnp.sum(jnp.isfinite(table)), jnp.sum(pair_table >= 0) // 2, energy],
        dtype=dtype,
    )
    return RNAMFEResult(
        energy,
        pair_table,
        paired_matrix,
        table,
        valid,
        status,
        evidence,
        _mfe_contract(np.dtype(dtype).name),
        model.model_id,
        constraint.constraint_id,
    )


def mfe_energy(
    sequence_codes: ArrayLike,
    model: RNAEnergyModel,
    constraints: RNAConstraints | None = None,
    /,
) -> Array:
    """Return the audited MFE scalar for differentiable parameter studies."""

    return minimum_free_energy(sequence_codes, model, constraints).energy


__all__ = ["RNAMFEResult", "mfe_energy", "minimum_free_energy"]
