#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from enum import IntEnum
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from phydrax._doc import DOC_KEY0
from phydrax._strict import StrictModule

from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ..sequence import SequenceDistribution


class DesignStatus(IntEnum):
    """Array-valued inverse-design status codes."""

    SUCCESS = 0
    NO_HARD_FEASIBLE_CANDIDATE = 1
    NONFINITE_OBJECTIVE = 2


_DESIGN_CONTRACT = BioinformaticsMethodContract(
    "constrained-sequence-inverse-design",
    MethodKind.HEURISTIC,
    ExecutionKind.STOCHASTIC_ESTIMATE,
    DifferentiationKind.NONE,
    OutputKind.SEQUENCE,
    conditioning_statement="Categorical sampling is followed by exact hard repair and full reranking.",
    truncation_statement="Candidate count is capacity-checked and never silently truncated.",
    capacity_semantics="All requested samples are scored; explicit sample capacity is preflighted.",
    assumptions=("The hard objective returns one finite score per sample and record.",),
    nondifferentiable_outputs=("sampled token codes", "hard-constraint status"),
)

_RELAXED_CONTRACT = BioinformaticsMethodContract(
    "relaxed-sequence-design-objective",
    MethodKind.RELAXED_OBJECTIVE,
    ExecutionKind.FLOATING_POINT_DIRECT,
    DifferentiationKind.EXACT_AD,
    OutputKind.SCALAR,
    conditioning_statement="The relaxed objective is evaluated on projected categorical probabilities.",
    truncation_statement="All positions and alphabet states are retained.",
    capacity_semantics="The complete dense sequence distribution is evaluated.",
    assumptions=(
        "Relaxed values are not asserted to equal realizable hard-sequence values.",
    ),
)


class AbstractSequenceConstraint(StrictModule):
    """Native hard constraint with both distribution projection and exact repair."""

    @abstractmethod
    def project(self, probabilities: Array, valid_mask: Array, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def enforce(self, token_codes: Array, valid_mask: Array, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def satisfied(self, token_codes: Array, valid_mask: Array, /) -> Array:
        raise NotImplementedError


class FixedTokenConstraint(AbstractSequenceConstraint):
    """Fix explicit case/position routes to exact alphabet token codes."""

    case_indices: Array
    position_indices: Array
    token_codes: Array
    alphabet_size: int = eqx.field(static=True)
    case_capacity: int = eqx.field(static=True)
    length_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        case_indices: Array,
        position_indices: Array,
        token_codes: Array,
        /,
        *,
        case_capacity: int,
        length_capacity: int,
        alphabet_size: int,
    ):
        cases = jnp.asarray(case_indices)
        positions = jnp.asarray(position_indices)
        tokens = jnp.asarray(token_codes)
        if (
            cases.ndim != 1
            or cases.shape != positions.shape
            or cases.shape != tokens.shape
        ):
            raise ValueError(
                "Fixed-token routes and codes must be equal rank-one arrays."
            )
        if not all(
            jnp.issubdtype(value.dtype, jnp.integer)
            for value in (cases, positions, tokens)
        ):
            raise TypeError("Fixed-token routes and codes must have integer dtype.")
        case_count = int(case_capacity)
        length = int(length_capacity)
        alphabet = int(alphabet_size)
        if case_count <= 0 or length <= 0 or alphabet <= 0:
            raise ValueError("Constraint capacities must be positive.")
        invalid = (
            (cases < 0)
            | (cases >= case_count)
            | (positions < 0)
            | (positions >= length)
            | (tokens < 0)
            | (tokens >= alphabet)
        )
        cases = eqx.error_if(
            cases, jnp.any(invalid), "Fixed-token route exceeds capacity."
        )
        duplicate_route = (cases[:, None] == cases[None, :]) & (
            positions[:, None] == positions[None, :]
        )
        disagreement = duplicate_route & (tokens[:, None] != tokens[None, :])
        tokens = eqx.error_if(
            tokens, jnp.any(disagreement), "Fixed-token constraints disagree."
        )
        self.case_indices = jax.lax.stop_gradient(cases)
        self.position_indices = jax.lax.stop_gradient(positions)
        self.token_codes = jax.lax.stop_gradient(tokens)
        self.alphabet_size = alphabet
        self.case_capacity = case_count
        self.length_capacity = length

    def _check_shape(self, shape: tuple[int, ...]) -> None:
        if shape[-2:] != (self.case_capacity, self.length_capacity):
            raise ValueError(
                "Fixed-token constraint capacity does not match sequence shape."
            )

    def project(self, probabilities: Array, valid_mask: Array, /) -> Array:
        values = jnp.asarray(probabilities)
        self._check_shape(values.shape[:-1])
        if int(values.shape[-1]) != self.alphabet_size:
            raise ValueError("Fixed-token alphabet capacity mismatch.")
        active = jnp.asarray(valid_mask, dtype=bool)
        active = eqx.error_if(
            active,
            jnp.any(~active[self.case_indices, self.position_indices]),
            "Fixed-token constraints cannot target padded positions.",
        )
        one_hot = jax.nn.one_hot(self.token_codes, self.alphabet_size, dtype=values.dtype)
        return values.at[self.case_indices, self.position_indices].set(one_hot)

    def enforce(self, token_codes: Array, valid_mask: Array, /) -> Array:
        values = jnp.asarray(token_codes)
        self._check_shape(values.shape[-2:])
        return values.at[..., self.case_indices, self.position_indices].set(
            self.token_codes
        )

    def satisfied(self, token_codes: Array, valid_mask: Array, /) -> Array:
        values = jnp.asarray(token_codes)
        self._check_shape(values.shape[-2:])
        routed = values[..., self.case_indices, self.position_indices]
        matched = routed == self.token_codes
        result = jnp.ones(values.shape[:-2] + (self.case_capacity,), dtype=bool)
        return result.at[..., self.case_indices].min(matched)


class AllowedTokenConstraint(AbstractSequenceConstraint):
    """Per-case/per-position allowed-token sets with observable empty-set failure."""

    allowed: Array
    fallback_codes: Array
    case_capacity: int = eqx.field(static=True)
    length_capacity: int = eqx.field(static=True)
    alphabet_size: int = eqx.field(static=True)

    def __init__(self, allowed: Array, /):
        support = jnp.asarray(allowed, dtype=bool)
        if support.ndim != 3 or any(int(size) <= 0 for size in support.shape):
            raise ValueError("allowed must have shape (batch, length, alphabet).")
        any_allowed = jnp.any(support, axis=-1)
        support = eqx.error_if(
            support,
            jnp.any(~any_allowed),
            "Every sequence position must allow at least one token.",
        )
        self.allowed = support
        self.fallback_codes = jax.lax.stop_gradient(
            jnp.argmax(support, axis=-1).astype(jnp.int32)
        )
        self.case_capacity = int(support.shape[0])
        self.length_capacity = int(support.shape[1])
        self.alphabet_size = int(support.shape[2])

    def _check(self, values: Array, *, distribution: bool) -> None:
        expected = (
            (self.case_capacity, self.length_capacity, self.alphabet_size)
            if distribution
            else (self.case_capacity, self.length_capacity)
        )
        if values.shape[-len(expected) :] != expected:
            raise ValueError("Allowed-token constraint capacity does not match values.")

    def project(self, probabilities: Array, valid_mask: Array, /) -> Array:
        values = jnp.asarray(probabilities)
        self._check(values, distribution=True)
        masked = jnp.where(self.allowed, values, 0.0)
        total = jnp.sum(masked, axis=-1, keepdims=True)
        fallback = jax.nn.one_hot(
            self.fallback_codes, self.alphabet_size, dtype=values.dtype
        )
        projected = jnp.where(
            total > 0.0, masked / jnp.where(total > 0.0, total, 1.0), fallback
        )
        return jnp.where(jnp.asarray(valid_mask, dtype=bool)[..., None], projected, 0.0)

    def enforce(self, token_codes: Array, valid_mask: Array, /) -> Array:
        values = jnp.asarray(token_codes)
        self._check(values, distribution=False)
        leading = values.shape[:-2]
        allowed_table = jnp.broadcast_to(self.allowed, leading + self.allowed.shape)
        fallback = jnp.broadcast_to(
            self.fallback_codes, leading + self.fallback_codes.shape
        )
        allowed = jnp.take_along_axis(
            allowed_table,
            values[..., None],
            axis=-1,
        )[..., 0]
        return jnp.where(allowed, values, fallback)

    def satisfied(self, token_codes: Array, valid_mask: Array, /) -> Array:
        values = jnp.asarray(token_codes)
        self._check(values, distribution=False)
        leading = values.shape[:-2]
        allowed_table = jnp.broadcast_to(self.allowed, leading + self.allowed.shape)
        allowed = jnp.take_along_axis(allowed_table, values[..., None], axis=-1)[..., 0]
        active = jnp.broadcast_to(jnp.asarray(valid_mask, dtype=bool), values.shape)
        return jnp.all(allowed | ~active, axis=-1)


class SequenceDesignProblem(StrictModule):
    """Finite-capacity native inverse-design problem over a sequence distribution."""

    initial_distribution: SequenceDistribution
    hard_objective: Any
    relaxed_objective: Any
    constraints: tuple[AbstractSequenceConstraint, ...]
    sample_count: int = eqx.field(static=True)
    sample_capacity: int = eqx.field(static=True)
    maximize: bool = eqx.field(static=True)

    def __init__(
        self,
        initial_distribution: SequenceDistribution,
        hard_objective: Any,
        relaxed_objective: Any,
        /,
        *,
        constraints: tuple[AbstractSequenceConstraint, ...] = (),
        sample_count: int,
        sample_capacity: int,
        maximize: bool = True,
    ):
        if not isinstance(initial_distribution, SequenceDistribution):
            raise TypeError("initial_distribution must be a SequenceDistribution.")
        if not callable(hard_objective) or not callable(relaxed_objective):
            raise TypeError("Hard and relaxed objectives must be pure callables.")
        requested = int(sample_count)
        capacity = int(sample_capacity)
        if requested <= 0 or capacity <= 0:
            raise ValueError("sample_count and sample_capacity must be positive.")
        if requested > capacity:
            raise ValueError("Requested samples exceed explicit design capacity.")
        constraints_ = tuple(constraints)
        if any(
            not isinstance(value, AbstractSequenceConstraint) for value in constraints_
        ):
            raise TypeError("constraints must contain native sequence constraints.")
        self.initial_distribution = initial_distribution
        self.hard_objective = hard_objective
        self.relaxed_objective = relaxed_objective
        self.constraints = constraints_
        self.sample_count = requested
        self.sample_capacity = capacity
        self.maximize = bool(maximize)


class SequenceDesignResult(StrictModule):
    """Hard-feasible selected sequences and separate relaxation/diversity evidence."""

    selected_codes: Array
    candidate_codes: Array
    candidate_scores: Array
    hard_objective: Array
    relaxed_objective: Array
    relaxation_gap: Array
    constraint_satisfied: Array
    pairwise_hamming: Array
    unique_fraction: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)
    relaxed_method_contract: BioinformaticsMethodContract = eqx.field(static=True)

    def __init__(
        self,
        selected_codes: Array,
        candidate_codes: Array,
        candidate_scores: Array,
        hard_objective: Array,
        relaxed_objective: Array,
        constraint_satisfied: Array,
        valid_mask: Array,
        *,
        maximize: bool,
    ):
        candidates = jnp.asarray(candidate_codes)
        scores = jnp.asarray(candidate_scores)
        selected = jnp.asarray(selected_codes)
        hard = jnp.asarray(hard_objective)
        relaxed = jnp.asarray(relaxed_objective)
        feasible = jnp.asarray(constraint_satisfied, dtype=bool)
        if candidates.ndim != 3 or scores.shape != candidates.shape[:2]:
            raise ValueError(
                "Candidates and scores require shapes (samples, batch, length)/(samples, batch)."
            )
        if selected.shape != candidates.shape[1:] or hard.shape != scores.shape[1:]:
            raise ValueError(
                "Selected sequences and hard objectives must match the record batch."
            )
        if relaxed.shape != hard.shape or feasible.shape != scores.shape:
            raise ValueError(
                "Relaxed scores and feasibility evidence have incompatible shape."
            )
        mask = jnp.asarray(valid_mask, dtype=bool)
        difference = candidates[:, None, ...] != candidates[None, :, ...]
        distances = jnp.sum(difference & mask[None, None, ...], axis=-1, dtype=jnp.int32)
        length = jnp.sum(mask, axis=-1, dtype=jnp.int32)
        pairwise = distances / jnp.maximum(length, 1)[None, None, :].astype(scores.dtype)
        sample_count = int(candidates.shape[0])
        prior = jnp.tril(jnp.ones((sample_count, sample_count), dtype=bool), k=-1)
        duplicates = jnp.any((distances == 0) & prior[..., None], axis=1)
        unique = jnp.mean((~duplicates).astype(scores.dtype), axis=0)
        finite = jnp.all(jnp.isfinite(scores) | ~feasible) & jnp.all(
            jnp.isfinite(relaxed)
        )
        any_feasible = jnp.all(jnp.any(feasible, axis=0))
        gap = jnp.where(maximize, relaxed - hard, hard - relaxed)
        self.selected_codes = selected
        self.candidate_codes = candidates
        self.candidate_scores = scores
        self.hard_objective = hard
        self.relaxed_objective = relaxed
        self.relaxation_gap = gap
        self.constraint_satisfied = feasible
        self.pairwise_hamming = pairwise
        self.unique_fraction = unique
        self.valid = finite & any_feasible
        self.status = jnp.where(
            ~any_feasible,
            jnp.asarray(DesignStatus.NO_HARD_FEASIBLE_CANDIDATE, dtype=jnp.int32),
            jnp.where(
                finite,
                jnp.asarray(DesignStatus.SUCCESS, dtype=jnp.int32),
                jnp.asarray(DesignStatus.NONFINITE_OBJECTIVE, dtype=jnp.int32),
            ),
        )
        self.evidence = jnp.stack(
            (
                jnp.sum(feasible, dtype=jnp.int32),
                jnp.asarray(feasible.size, dtype=jnp.int32),
            )
        )
        self.method_contract = _DESIGN_CONTRACT
        self.relaxed_method_contract = _RELAXED_CONTRACT


def solve_sequence_design(
    problem: SequenceDesignProblem,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
) -> SequenceDesignResult:
    """Project, sample exactly, hard-repair, fully score, and rerank candidates."""
    if not isinstance(problem, SequenceDesignProblem):
        raise TypeError("problem must be a SequenceDesignProblem.")
    distribution = problem.initial_distribution
    probabilities = distribution.probabilities
    active = distribution.valid_mask & distribution.case_mask[:, None]
    for constraint in problem.constraints:
        probabilities = constraint.project(probabilities, active)
    probabilities = jnp.where(active[..., None], probabilities, 0.0)
    safe_probabilities = jnp.where(
        active[..., None],
        probabilities,
        jax.nn.one_hot(
            jnp.zeros(active.shape, dtype=jnp.int32),
            int(probabilities.shape[-1]),
            dtype=probabilities.dtype,
        ),
    )
    logits = jnp.log(
        jnp.maximum(safe_probabilities, jnp.finfo(safe_probabilities.dtype).tiny)
    )
    candidates = jax.random.categorical(
        key,
        logits,
        axis=-1,
        shape=(problem.sample_count,) + logits.shape[:-1],
    ).astype(jnp.int32)
    for constraint in problem.constraints:
        candidates = constraint.enforce(candidates, active)
    feasibility = jnp.ones(candidates.shape[:2], dtype=bool)
    for constraint in problem.constraints:
        feasibility = feasibility & constraint.satisfied(candidates, active)
    scores = jnp.asarray(problem.hard_objective(candidates, active))
    if scores.shape != candidates.shape[:2]:
        raise ValueError("hard_objective must return shape (samples, batch).")
    invalid_fill = -jnp.inf if problem.maximize else jnp.inf
    ranked = jnp.where(feasibility, scores, invalid_fill)
    best_indices = (
        jnp.argmax(ranked, axis=0) if problem.maximize else jnp.argmin(ranked, axis=0)
    )
    batch_indices = jnp.arange(candidates.shape[1])
    selected = candidates[best_indices, batch_indices]
    hard = scores[best_indices, batch_indices]
    relaxed = jnp.asarray(problem.relaxed_objective(probabilities, active))
    if relaxed.shape != hard.shape:
        raise ValueError("relaxed_objective must return shape (batch,).")
    return SequenceDesignResult(
        selected,
        candidates,
        scores,
        hard,
        relaxed,
        feasibility,
        active,
        maximize=problem.maximize,
    )


__all__ = [
    "AbstractSequenceConstraint",
    "AllowedTokenConstraint",
    "DesignStatus",
    "FixedTokenConstraint",
    "SequenceDesignProblem",
    "SequenceDesignResult",
    "solve_sequence_design",
]
