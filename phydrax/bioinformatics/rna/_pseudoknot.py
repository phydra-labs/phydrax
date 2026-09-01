#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._constraints import RNAConstraints, RNAFoldStatus
from ._energy_model import RNAEnergyModel


class RestrictedPseudoknotPlan(StrictModule, NonTrainableState):
    """Capacity-bounded deterministic crossing-pair greedy policy."""

    max_candidates: int = eqx.field(static=True)
    max_pairs: int = eqx.field(static=True)
    max_crossings: int = eqx.field(static=True)
    require_energy_improvement: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_candidates: int,
        max_pairs: int,
        max_crossings: int,
        require_energy_improvement: bool = True,
    ):
        candidates = int(max_candidates)
        pairs = int(max_pairs)
        crossings = int(max_crossings)
        if candidates < 0 or pairs < 0 or crossings < 0:
            raise ValueError("Pseudoknot capacities must be non-negative.")
        self.max_candidates = candidates
        self.max_pairs = pairs
        self.max_crossings = crossings
        self.require_energy_improvement = bool(require_energy_improvement)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "restricted-pseudoknot-plan",
                "max_candidates": candidates,
                "max_pairs": pairs,
                "max_crossings": crossings,
                "require_energy_improvement": self.require_energy_improvement,
            }
        )


class RestrictedPseudoknotResult(StrictModule):
    """Explicitly heuristic crossing structure; never presented as an exact optimum."""

    energy: Array
    pair_table: Array
    paired_matrix: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract
    model_id: str = eqx.field(static=True)
    constraint_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    evidence_labels: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        energy: Array,
        pair_table: Array,
        paired_matrix: Array,
        valid: bool,
        status: RNAFoldStatus,
        evidence: np.ndarray,
        method_contract: BioinformaticsMethodContract,
        model_id: str,
        constraint_id: str,
        plan_id: str,
    ):
        self.energy = jnp.asarray(energy)
        self.pair_table = jnp.asarray(pair_table, dtype=jnp.int32)
        self.paired_matrix = jnp.asarray(paired_matrix, dtype=bool)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(int(status), dtype=jnp.int32)
        self.evidence = jnp.asarray(evidence)
        self.method_contract = method_contract
        self.model_id = model_id
        self.constraint_id = constraint_id
        self.plan_id = plan_id
        self.evidence_labels = (
            "sequence_length",
            "candidate_count",
            "selected_pair_count",
            "crossing_count",
        )


def restricted_pseudoknot_fold(
    sequence_codes: ArrayLike,
    model: RNAEnergyModel,
    plan: RestrictedPseudoknotPlan,
    constraints: RNAConstraints | None = None,
    /,
) -> RestrictedPseudoknotResult:
    """Greedily admit lowest-delta pairs subject to declared crossing capacities."""

    if not isinstance(model, RNAEnergyModel) or not isinstance(
        plan, RestrictedPseudoknotPlan
    ):
        raise TypeError(
            "model and plan must be RNAEnergyModel and RestrictedPseudoknotPlan."
        )
    sequence = np.asarray(sequence_codes)
    if sequence.ndim != 1 or not np.issubdtype(sequence.dtype, np.integer):
        raise TypeError("sequence_codes must be an integer vector.")
    sequence = sequence.astype(np.int32, copy=False)
    length = int(sequence.size)
    constraint = (
        RNAConstraints.unconstrained(length) if constraints is None else constraints
    )
    if constraint.sequence_length != length:
        raise ValueError("constraints must match sequence length.")
    pair_table = np.full((length,), -1, dtype=np.int32)
    paired = np.zeros((length, length), dtype=bool)
    dtype = np.asarray(model.pair_energies).dtype
    evidence = np.asarray([length, 0, 0, 0], dtype=np.int32)
    method = BioinformaticsMethodContract(
        "restricted-greedy-rna-pseudoknot-fold",
        MethodKind.HEURISTIC,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.NONE,
        OutputKind.STRUCTURED,
        conditioning_statement="Deterministic sorting by additive energy delta, then pair indices.",
        truncation_statement="No candidate truncation is permitted; capacity overflow returns failure.",
        capacity_semantics="Candidate, selected-pair, and crossing counts are preflighted against the plan.",
        assumptions=(
            "Greedy selection is not a global pseudoknot optimum and carries no thermodynamic claim.",
        ),
        nondifferentiable_outputs=("all outputs",),
        input_dtype="int32",
        compute_dtype=np.dtype(dtype).name,
        output_dtype=np.dtype(dtype).name,
    )

    def result(
        valid: bool, status: RNAFoldStatus, energy: float
    ) -> RestrictedPseudoknotResult:
        return RestrictedPseudoknotResult(
            np.asarray(energy, dtype=dtype),
            pair_table,
            paired,
            valid,
            status,
            evidence,
            method,
            model.model_id,
            constraint.constraint_id,
            plan.plan_id,
        )

    if np.any(sequence < 0) or np.any(sequence >= model.alphabet_size):
        return result(False, RNAFoldStatus.INVALID_SEQUENCE, np.inf)
    allowed = np.asarray(model.allowed_pairs)[sequence[:, None], sequence[None, :]]
    prohibited = np.asarray(constraint.prohibited_pairs)
    required = np.asarray(constraint.required_partner)
    pair_energy = np.asarray(model.pair_energies)[
        sequence[:, None], sequence[None, :]
    ] + np.asarray(constraint.pair_energy_offsets)
    unpaired_energy = np.asarray(model.unpaired_energies)[sequence] + np.asarray(
        constraint.unpaired_energy_offsets
    )
    candidates: list[tuple[float, int, int]] = []
    for first in range(length):
        for second in range(first + model.minimum_hairpin_length + 1, length):
            if not allowed[first, second] or prohibited[first, second]:
                continue
            if required[first] not in (-2, second) or required[second] not in (-2, first):
                continue
            delta = float(
                pair_energy[first, second]
                - unpaired_energy[first]
                - unpaired_energy[second]
            )
            candidates.append((delta, first, second))
    evidence[1] = len(candidates)
    if len(candidates) > plan.max_candidates:
        return result(False, RNAFoldStatus.CAPACITY_EXCEEDED, np.inf)
    candidate_pairs = {(first, second) for _, first, second in candidates}
    required_pairs = [
        (first, int(second)) for first, second in enumerate(required) if second > first
    ]
    if any(pair not in candidate_pairs for pair in required_pairs):
        return result(False, RNAFoldStatus.INFEASIBLE_CONSTRAINTS, np.inf)

    def crossing_count_with(
        first: int, second: int, selected: list[tuple[int, int]]
    ) -> int:
        return sum(
            int(i < first < j < second or first < i < second < j) for i, j in selected
        )

    selected: list[tuple[int, int]] = []
    for first, second in required_pairs:
        crossings = crossing_count_with(first, second, selected)
        if (
            len(selected) >= plan.max_pairs
            or evidence[3] + crossings > plan.max_crossings
        ):
            return result(False, RNAFoldStatus.CAPACITY_EXCEEDED, np.inf)
        selected.append((first, second))
        pair_table[first] = second
        pair_table[second] = first
        evidence[3] += crossings
    required_set = set(required_pairs)
    for delta, first, second in sorted(candidates):
        if (
            (first, second) in required_set
            or pair_table[first] >= 0
            or pair_table[second] >= 0
        ):
            continue
        if plan.require_energy_improvement and delta >= 0.0:
            continue
        crossings = crossing_count_with(first, second, selected)
        if (
            len(selected) >= plan.max_pairs
            or evidence[3] + crossings > plan.max_crossings
        ):
            continue
        selected.append((first, second))
        pair_table[first] = second
        pair_table[second] = first
        evidence[3] += crossings
    paired[np.arange(length)[pair_table >= 0], pair_table[pair_table >= 0]] = True
    evidence[2] = len(selected)
    energy = float(np.sum(unpaired_energy))
    for first, second in selected:
        energy += float(
            pair_energy[first, second] - unpaired_energy[first] - unpaired_energy[second]
        )
    return result(True, RNAFoldStatus.HEURISTIC_RESULT, energy)


__all__ = [
    "RestrictedPseudoknotPlan",
    "RestrictedPseudoknotResult",
    "restricted_pseudoknot_fold",
]
