#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from numbers import Integral

import equinox as eqx
import jax
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
from ._alphabet import AlphabetPlan, PROTEIN_IUPAC
from ._batch import SequenceBatch
from ._motifs import _observation_support


MATCH_STATE = 0
INSERT_STATE = 1
DELETE_STATE = 2
BEGIN_STATE = 3
END_STATE = 4

PROFILE_STATUS_VALID = 0
PROFILE_STATUS_CAPACITY_EXCEEDED = 1
PROFILE_STATUS_INFEASIBLE = 2


def _profile_contract(name: str, probabilistic: bool) -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        name,
        MethodKind.EXACT_MODEL,
        (
            ExecutionKind.FLOATING_POINT_DIRECT
            if probabilistic
            else ExecutionKind.EXACT_DISCRETE
        ),
        (
            DifferentiationKind.ALMOST_EVERYWHERE
            if probabilistic
            else DifferentiationKind.NONE
        ),
        OutputKind.PROBABILISTIC if probabilistic else OutputKind.STRUCTURED,
        conditioning_statement=(
            "Exact dynamic programming for the supplied finite profile-HMM, encoded "
            "sequence, position mask, and floating-point log semiring."
        ),
        truncation_statement="The full profile lattice is evaluated without beam pruning.",
        capacity_semantics="Both sequence and profile lengths must fit the static plan bounds.",
        assumptions=(
            "The profile follows the declared match/insert/delete transition topology.",
            "Ambiguity symbols are marginalized over canonical emission support.",
        ),
        nondifferentiable_outputs=("state_path", "status")
        if not probabilistic
        else ("status",),
    )


def _normalize_rows(values: Array, name: str, /) -> Array:
    concrete = None if isinstance(values, jax_core.Tracer) else np.asarray(values)
    if concrete is not None and (
        np.any(~np.isfinite(concrete))
        or np.any(concrete < 0.0)
        or np.any(np.sum(concrete, axis=-1) <= 0.0)
    ):
        raise ValueError(f"{name} must be finite, non-negative, and nonzero per row.")
    return values / jnp.sum(values, axis=-1, keepdims=True)


def _safe_log(probabilities: Array, /) -> Array:
    return jnp.where(probabilities > 0.0, jnp.log(probabilities), -jnp.inf)


class ProfileHMM(StrictModule):
    """Finite Plan7-style profile HMM with explicit begin, delete, and end routes.

    At profile boundary ``k``, a transition column denotes a move to ``M[k+1]``,
    ``I[k]``, or ``D[k+1]`` respectively. Boundary zero contains begin and I0;
    boundary ``model_length`` contains IL and terminal transitions.
    """

    match_probabilities: Array
    insert_probabilities: Array
    transition_probabilities: Array
    terminal_probabilities: Array
    match_log_emissions: Array
    insert_log_emissions: Array
    transition_log_probabilities: Array
    terminal_log_probabilities: Array
    alphabet: AlphabetPlan = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)
    fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        match_emissions: ArrayLike,
        alphabet: AlphabetPlan = PROTEIN_IUPAC,
        *,
        insert_emissions: ArrayLike | None = None,
        transition_probabilities: ArrayLike | None = None,
        terminal_probabilities: ArrayLike | None = None,
        profile_id: str = "profile-hmm",
    ):
        if not isinstance(alphabet, AlphabetPlan):
            raise TypeError("alphabet must be an AlphabetPlan.")
        canonical_count = len(alphabet.canonical_symbols)
        match = jnp.asarray(match_emissions)
        if match.ndim != 2 or match.shape[0] <= 0 or match.shape[1] != canonical_count:
            raise ValueError(
                "match_emissions must have shape (positive model length, canonical symbol count)."
            )
        if not jnp.issubdtype(match.dtype, jnp.floating):
            match = match.astype(jnp.float32)
        match = _normalize_rows(match, "match_emissions")
        model_length = int(match.shape[0])

        if insert_emissions is None:
            insert = jnp.full(
                (model_length + 1, canonical_count),
                1.0 / canonical_count,
                dtype=match.dtype,
            )
        else:
            insert = jnp.asarray(insert_emissions, dtype=match.dtype)
            if insert.shape != (model_length + 1, canonical_count):
                raise ValueError(
                    "insert_emissions must have shape (model length + 1, canonical symbol count)."
                )
            insert = _normalize_rows(insert, "insert_emissions")

        if transition_probabilities is None:
            transition = np.zeros((model_length + 1, 3, 3), dtype=np.float64)
            transition[0, MATCH_STATE] = (0.90, 0.05, 0.05)
            transition[0, INSERT_STATE] = (0.80, 0.15, 0.05)
            for boundary in range(1, model_length):
                transition[boundary, MATCH_STATE] = (0.90, 0.05, 0.05)
                transition[boundary, INSERT_STATE] = (0.80, 0.15, 0.05)
                transition[boundary, DELETE_STATE] = (0.80, 0.05, 0.15)
            transition[model_length, :, INSERT_STATE] = (0.05, 0.20, 0.05)
            transition_values = jnp.asarray(transition, dtype=match.dtype)
            terminal_values = jnp.asarray((0.95, 0.80, 0.95), dtype=match.dtype)
        else:
            transition_values = jnp.asarray(transition_probabilities, dtype=match.dtype)
            if transition_values.shape != (model_length + 1, 3, 3):
                raise ValueError(
                    "transition_probabilities must have shape (model length + 1, 3, 3)."
                )
            if terminal_probabilities is None:
                raise ValueError(
                    "terminal_probabilities are required with explicit transitions."
                )
            terminal_values = jnp.asarray(terminal_probabilities, dtype=match.dtype)

        if terminal_values.shape != (3,):
            raise ValueError("terminal_probabilities must have shape (3,).")
        concrete_transition = (
            None
            if isinstance(transition_values, jax_core.Tracer)
            else np.asarray(transition_values)
        )
        concrete_terminal = (
            None
            if isinstance(terminal_values, jax_core.Tracer)
            else np.asarray(terminal_values)
        )
        if concrete_transition is not None and (
            np.any(~np.isfinite(concrete_transition)) or np.any(concrete_transition < 0.0)
        ):
            raise ValueError("Transition probabilities must be finite and non-negative.")
        if concrete_terminal is not None and (
            np.any(~np.isfinite(concrete_terminal)) or np.any(concrete_terminal < 0.0)
        ):
            raise ValueError("Terminal probabilities must be finite and non-negative.")
        if concrete_transition is not None:
            invalid = np.zeros_like(concrete_transition, dtype=bool)
            invalid[model_length, :, MATCH_STATE] = True
            invalid[model_length, :, DELETE_STATE] = True
            invalid[0, DELETE_STATE, :] = True
            if np.any(concrete_transition[invalid] != 0.0):
                raise ValueError("Transitions outside the profile topology must be zero.")
            totals = concrete_transition.sum(axis=-1)
            for boundary in range(model_length):
                sources = (MATCH_STATE, INSERT_STATE) if boundary == 0 else range(3)
                if any(
                    not np.isclose(totals[boundary, source], 1.0) for source in sources
                ):
                    raise ValueError(
                        "Every nonterminal reachable transition row must sum to one."
                    )
            if concrete_terminal is not None:
                terminal_totals = (
                    concrete_transition[model_length, :, INSERT_STATE] + concrete_terminal
                )
                if not np.allclose(terminal_totals, 1.0):
                    raise ValueError(
                        "At the final boundary, insert and terminal probabilities must sum to one."
                    )

        support, scorable = _observation_support(alphabet)
        support = support.astype(match.dtype)
        observed_match = match @ support.T
        observed_insert = insert @ support.T
        observed_match = jnp.where(scorable[None, :], observed_match, 0.0)
        observed_insert = jnp.where(scorable[None, :], observed_insert, 0.0)
        identifier = str(profile_id).strip()
        if not identifier:
            raise ValueError("profile_id must be non-empty.")

        self.match_probabilities = match
        self.insert_probabilities = insert
        self.transition_probabilities = transition_values
        self.terminal_probabilities = terminal_values
        self.match_log_emissions = _safe_log(observed_match)
        self.insert_log_emissions = _safe_log(observed_insert)
        self.transition_log_probabilities = _safe_log(transition_values)
        self.terminal_log_probabilities = _safe_log(terminal_values)
        self.alphabet = alphabet
        self.profile_id = identifier
        self.fingerprint = canonical_fingerprint(
            {
                "kind": "profile-hmm",
                "profile_id": identifier,
                "alphabet": alphabet.fingerprint,
                "match": array_tree_fingerprint(match),
                "insert": array_tree_fingerprint(insert),
                "transition": array_tree_fingerprint(transition_values),
                "terminal": array_tree_fingerprint(terminal_values),
            }
        )

    @property
    def model_length(self) -> int:
        return int(self.match_probabilities.shape[0])

    def forward(
        self, sequences: SequenceBatch, plan: ProfileHMMPlan | None = None, /
    ) -> ProfileForwardResult:
        return profile_forward(self, sequences, plan)

    def backward(
        self, sequences: SequenceBatch, plan: ProfileHMMPlan | None = None, /
    ) -> ProfileBackwardResult:
        return profile_backward(self, sequences, plan)

    def forward_backward(
        self, sequences: SequenceBatch, plan: ProfileHMMPlan | None = None, /
    ) -> ProfileMarginalResult:
        return profile_forward_backward(self, sequences, plan)

    def viterbi(
        self, sequences: SequenceBatch, plan: ProfileHMMPlan | None = None, /
    ) -> ProfileViterbiResult:
        return profile_viterbi(self, sequences, plan)


class ProfileHMMPlan(StrictModule):
    """Static bounds and declared exact profile-DP semantics."""

    maximum_sequence_length: int = eqx.field(static=True)
    maximum_model_length: int = eqx.field(static=True)
    tie_policy: str = eqx.field(static=True)
    marginal_contract: BioinformaticsMethodContract = eqx.field(static=True)
    viterbi_contract: BioinformaticsMethodContract = eqx.field(static=True)

    def __init__(self, maximum_sequence_length: int, maximum_model_length: int):
        values = (maximum_sequence_length, maximum_model_length)
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) for value in values
        ):
            raise TypeError("Profile-HMM capacity bounds must be integers.")
        if any(int(value) <= 0 for value in values):
            raise ValueError("Profile-HMM capacity bounds must be positive.")
        self.maximum_sequence_length = int(maximum_sequence_length)
        self.maximum_model_length = int(maximum_model_length)
        self.tie_policy = "match-before-insert-before-delete"
        self.marginal_contract = _profile_contract("profile-HMM forward-backward", True)
        self.viterbi_contract = _profile_contract("profile-HMM Viterbi", False)


class ProfileDPEvidence(StrictModule):
    """Numerical, posterior, and capacity evidence for profile inference."""

    capacity_sufficient: Array
    forward_backward_residual: Array
    emitted_mass_residual: Array
    terminal_mass: Array
    delete_state_reachable: Array


class ProfileForwardResult(StrictModule):
    forward_log_probabilities: Array
    log_likelihood: Array
    valid: Array
    status: Array
    evidence: ProfileDPEvidence
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


class ProfileBackwardResult(StrictModule):
    backward_log_probabilities: Array
    log_likelihood: Array
    valid: Array
    status: Array
    evidence: ProfileDPEvidence
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


class ProfileMarginalResult(StrictModule):
    log_likelihood: Array
    forward_log_probabilities: Array
    backward_log_probabilities: Array
    match_marginals: Array
    insert_marginals: Array
    delete_marginals: Array
    class_posterior_mass: Array
    terminal_marginal: Array
    valid: Array
    status: Array
    evidence: ProfileDPEvidence
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


class ProfileStatePath(StrictModule):
    state_class: Array
    profile_position: Array
    sequence_position: Array
    valid: Array
    length: Array


class ProfileViterbiEvidence(StrictModule):
    capacity_sufficient: Array
    reaches_begin: Array
    reaches_end: Array
    query_positions_consumed: Array
    score_finite: Array


class ProfileViterbiResult(StrictModule):
    score: Array
    path: ProfileStatePath
    valid: Array
    status: Array
    evidence: ProfileViterbiEvidence
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


def _validate_inputs(
    model: ProfileHMM,
    sequences: SequenceBatch,
    plan: ProfileHMMPlan | None,
    /,
) -> tuple[ProfileHMMPlan, bool]:
    if not isinstance(model, ProfileHMM):
        raise TypeError("model must be a ProfileHMM.")
    if not isinstance(sequences, SequenceBatch):
        raise TypeError("sequences must be a SequenceBatch.")
    if sequences.alphabet.fingerprint != model.alphabet.fingerprint:
        raise ValueError("Profile-HMM and sequence alphabets must match.")
    if plan is None:
        plan = ProfileHMMPlan(max(sequences.sequence_capacity, 1), model.model_length)
    if not isinstance(plan, ProfileHMMPlan):
        raise TypeError("plan must be a ProfileHMMPlan.")
    sufficient = (
        sequences.sequence_capacity <= plan.maximum_sequence_length
        and model.model_length <= plan.maximum_model_length
    )
    return plan, sufficient


def _forward_single(
    model: ProfileHMM, tokens: Array, mask: Array, /
) -> tuple[Array, Array]:
    sequence_capacity = int(tokens.shape[0])
    model_length = model.model_length
    dtype = model.match_log_emissions.dtype
    alpha = jnp.full((sequence_capacity + 1, model_length + 1, 3), -jnp.inf, dtype=dtype)
    alpha = alpha.at[0, 0, MATCH_STATE].set(0.0)
    for boundary in range(1, model_length + 1):
        candidates = (
            alpha[0, boundary - 1, :]
            + model.transition_log_probabilities[boundary - 1, :, DELETE_STATE]
        )
        alpha = alpha.at[0, boundary, DELETE_STATE].set(jax.nn.logsumexp(candidates))

    for position in range(1, sequence_capacity + 1):
        token = tokens[position - 1]
        row = jnp.full((model_length + 1, 3), -jnp.inf, dtype=dtype)
        for boundary in range(model_length + 1):
            insert_candidates = (
                alpha[position - 1, boundary, :]
                + model.transition_log_probabilities[boundary, :, INSERT_STATE]
            )
            row = row.at[boundary, INSERT_STATE].set(
                jax.nn.logsumexp(insert_candidates)
                + model.insert_log_emissions[boundary, token]
            )
            if boundary > 0:
                match_candidates = (
                    alpha[position - 1, boundary - 1, :]
                    + model.transition_log_probabilities[boundary - 1, :, MATCH_STATE]
                )
                row = row.at[boundary, MATCH_STATE].set(
                    jax.nn.logsumexp(match_candidates)
                    + model.match_log_emissions[boundary - 1, token]
                )
        for boundary in range(1, model_length + 1):
            delete_candidates = (
                row[boundary - 1, :]
                + model.transition_log_probabilities[boundary - 1, :, DELETE_STATE]
            )
            row = row.at[boundary, DELETE_STATE].set(jax.nn.logsumexp(delete_candidates))
        alpha = alpha.at[position].set(
            jnp.where(mask[position - 1], row, alpha[position - 1])
        )
    likelihood = jax.nn.logsumexp(
        alpha[sequence_capacity, model_length, :] + model.terminal_log_probabilities
    )
    return alpha, likelihood


def _backward_single(
    model: ProfileHMM, tokens: Array, mask: Array, /
) -> tuple[Array, Array]:
    sequence_capacity = int(tokens.shape[0])
    model_length = model.model_length
    dtype = model.match_log_emissions.dtype
    beta = jnp.full((sequence_capacity + 1, model_length + 1, 3), -jnp.inf, dtype=dtype)
    beta = beta.at[sequence_capacity, model_length, :].set(
        model.terminal_log_probabilities
    )
    for boundary in range(model_length - 1, -1, -1):
        beta = beta.at[sequence_capacity, boundary, :].set(
            model.transition_log_probabilities[boundary, :, DELETE_STATE]
            + beta[sequence_capacity, boundary + 1, DELETE_STATE]
        )

    for position in range(sequence_capacity - 1, -1, -1):
        token = tokens[position]
        row = jnp.full((model_length + 1, 3), -jnp.inf, dtype=dtype)
        for boundary in range(model_length, -1, -1):
            candidates = [
                model.transition_log_probabilities[boundary, :, INSERT_STATE]
                + model.insert_log_emissions[boundary, token]
                + beta[position + 1, boundary, INSERT_STATE]
            ]
            if boundary < model_length:
                candidates.extend(
                    [
                        model.transition_log_probabilities[boundary, :, MATCH_STATE]
                        + model.match_log_emissions[boundary, token]
                        + beta[position + 1, boundary + 1, MATCH_STATE],
                        model.transition_log_probabilities[boundary, :, DELETE_STATE]
                        + row[boundary + 1, DELETE_STATE],
                    ]
                )
            row = row.at[boundary, :].set(
                jax.nn.logsumexp(jnp.stack(candidates, axis=0), axis=0)
            )
        beta = beta.at[position].set(jnp.where(mask[position], row, beta[position + 1]))
    return beta, beta[0, 0, MATCH_STATE]


def _case_mask(sequences: SequenceBatch, /) -> Array:
    return jnp.asarray(sequences.case_mask, dtype=bool)


def _base_evidence(
    case_mask: Array,
    capacity_ok: bool,
    forward_score: Array,
    backward_score: Array,
    emitted_residual: Array,
    terminal_mass: Array,
    model: ProfileHMM,
    /,
) -> ProfileDPEvidence:
    delete_reachable = jnp.any(model.transition_probabilities[:, :, DELETE_STATE] > 0.0)
    return ProfileDPEvidence(
        jnp.full(case_mask.shape, capacity_ok) & case_mask,
        jnp.where(case_mask, jnp.abs(forward_score - backward_score), jnp.inf),
        jnp.where(case_mask, emitted_residual, jnp.inf),
        jnp.where(case_mask, terminal_mass, 0.0),
        jnp.full(case_mask.shape, delete_reachable) & case_mask,
    )


def _status(valid: Array, case_mask: Array, capacity_ok: bool, /) -> Array:
    return jnp.where(
        valid,
        PROFILE_STATUS_VALID,
        jnp.where(
            case_mask & jnp.asarray(not capacity_ok),
            PROFILE_STATUS_CAPACITY_EXCEEDED,
            PROFILE_STATUS_INFEASIBLE,
        ),
    ).astype(jnp.int32)


def _capacity_failure_evidence(
    case_mask: Array, model: ProfileHMM, /
) -> ProfileDPEvidence:
    delete_reachable = jnp.any(model.transition_probabilities[:, :, DELETE_STATE] > 0.0)
    return ProfileDPEvidence(
        jnp.zeros(case_mask.shape, dtype=bool),
        jnp.full(case_mask.shape, jnp.inf),
        jnp.full(case_mask.shape, jnp.inf),
        jnp.zeros(case_mask.shape),
        jnp.full(case_mask.shape, delete_reachable) & case_mask,
    )


def _empty_lattice(model: ProfileHMM, sequences: SequenceBatch, /) -> Array:
    return jnp.full(
        (
            sequences.record_capacity,
            sequences.sequence_capacity + 1,
            model.model_length + 1,
            3,
        ),
        -jnp.inf,
        dtype=model.match_log_emissions.dtype,
    )


def profile_forward(
    model: ProfileHMM,
    sequences: SequenceBatch,
    plan: ProfileHMMPlan | None = None,
    /,
) -> ProfileForwardResult:
    plan, capacity_ok = _validate_inputs(model, sequences, plan)
    if not capacity_ok:
        case = _case_mask(sequences)
        valid = jnp.zeros(case.shape, dtype=bool)
        return ProfileForwardResult(
            _empty_lattice(model, sequences),
            jnp.full(case.shape, -jnp.inf, dtype=model.match_log_emissions.dtype),
            valid,
            _status(valid, case, False),
            _capacity_failure_evidence(case, model),
            plan.marginal_contract,
        )
    alpha, likelihood = jax.vmap(lambda token, mask: _forward_single(model, token, mask))(
        sequences.token_codes, sequences.valid_mask
    )
    case = _case_mask(sequences)
    valid = case & capacity_ok & jnp.isfinite(likelihood)
    evidence = _base_evidence(
        case,
        capacity_ok,
        likelihood,
        likelihood,
        jnp.zeros_like(likelihood),
        jnp.where(valid, 1.0, 0.0),
        model,
    )
    return ProfileForwardResult(
        alpha,
        jnp.where(case, likelihood, -jnp.inf),
        valid,
        _status(valid, case, capacity_ok),
        evidence,
        plan.marginal_contract,
    )


def profile_backward(
    model: ProfileHMM,
    sequences: SequenceBatch,
    plan: ProfileHMMPlan | None = None,
    /,
) -> ProfileBackwardResult:
    plan, capacity_ok = _validate_inputs(model, sequences, plan)
    if not capacity_ok:
        case = _case_mask(sequences)
        valid = jnp.zeros(case.shape, dtype=bool)
        return ProfileBackwardResult(
            _empty_lattice(model, sequences),
            jnp.full(case.shape, -jnp.inf, dtype=model.match_log_emissions.dtype),
            valid,
            _status(valid, case, False),
            _capacity_failure_evidence(case, model),
            plan.marginal_contract,
        )
    beta, likelihood = jax.vmap(lambda token, mask: _backward_single(model, token, mask))(
        sequences.token_codes, sequences.valid_mask
    )
    case = _case_mask(sequences)
    valid = case & capacity_ok & jnp.isfinite(likelihood)
    evidence = _base_evidence(
        case,
        capacity_ok,
        likelihood,
        likelihood,
        jnp.zeros_like(likelihood),
        jnp.where(valid, 1.0, 0.0),
        model,
    )
    return ProfileBackwardResult(
        beta,
        jnp.where(case, likelihood, -jnp.inf),
        valid,
        _status(valid, case, capacity_ok),
        evidence,
        plan.marginal_contract,
    )


def profile_forward_backward(
    model: ProfileHMM,
    sequences: SequenceBatch,
    plan: ProfileHMMPlan | None = None,
    /,
) -> ProfileMarginalResult:
    plan, capacity_ok = _validate_inputs(model, sequences, plan)
    if not capacity_ok:
        case = _case_mask(sequences)
        records = sequences.record_capacity
        sequence_capacity = sequences.sequence_capacity
        model_length = model.model_length
        dtype = model.match_log_emissions.dtype
        valid = jnp.zeros(case.shape, dtype=bool)
        lattice = _empty_lattice(model, sequences)
        return ProfileMarginalResult(
            jnp.full(case.shape, -jnp.inf, dtype=dtype),
            lattice,
            lattice,
            jnp.zeros((records, sequence_capacity, model_length), dtype=dtype),
            jnp.zeros((records, sequence_capacity, model_length + 1), dtype=dtype),
            jnp.zeros((records, sequence_capacity + 1, model_length), dtype=dtype),
            jnp.zeros((records, 3), dtype=dtype),
            jnp.zeros(case.shape, dtype=dtype),
            valid,
            _status(valid, case, False),
            _capacity_failure_evidence(case, model),
            plan.marginal_contract,
        )
    alpha, forward_score = jax.vmap(
        lambda token, mask: _forward_single(model, token, mask)
    )(sequences.token_codes, sequences.valid_mask)
    beta, backward_score = jax.vmap(
        lambda token, mask: _backward_single(model, token, mask)
    )(sequences.token_codes, sequences.valid_mask)
    case = _case_mask(sequences)
    finite = jnp.isfinite(forward_score) & jnp.isfinite(backward_score)
    valid = case & capacity_ok & finite
    safe_normalizer = jnp.where(finite, forward_score, 0.0)
    posterior = jnp.where(
        finite[:, None, None, None],
        jnp.exp(alpha + beta - safe_normalizer[:, None, None, None]),
        0.0,
    )
    match = posterior[:, 1:, 1:, MATCH_STATE]
    insert = posterior[:, 1:, :, INSERT_STATE]
    delete = posterior[:, :, 1:, DELETE_STATE]
    active = sequences.valid_mask
    match = jnp.where(active[:, :, None], match, 0.0)
    insert = jnp.where(active[:, :, None], insert, 0.0)
    delete_row_valid = jnp.concatenate(
        (
            jnp.ones((sequences.record_capacity, 1), dtype=bool),
            sequences.valid_mask,
        ),
        axis=1,
    )
    delete = jnp.where(delete_row_valid[:, :, None], delete, 0.0)
    emitted_mass = jnp.sum(match, axis=-1) + jnp.sum(insert, axis=-1)
    emitted_residual = (
        jnp.max(jnp.where(active, jnp.abs(emitted_mass - 1.0), 0.0), axis=1)
        if sequences.sequence_capacity > 0
        else jnp.zeros((sequences.record_capacity,), dtype=emitted_mass.dtype)
    )
    class_mass = jnp.stack(
        (
            jnp.sum(match, axis=(1, 2)),
            jnp.sum(insert, axis=(1, 2)),
            jnp.sum(delete, axis=(1, 2)),
        ),
        axis=1,
    )
    terminal_log_mass = jax.nn.logsumexp(
        alpha[:, -1, -1, :] + model.terminal_log_probabilities[None, :], axis=1
    )
    terminal = jnp.where(finite, jnp.exp(terminal_log_mass - forward_score), 0.0)
    evidence = _base_evidence(
        case,
        capacity_ok,
        forward_score,
        backward_score,
        emitted_residual,
        terminal,
        model,
    )
    return ProfileMarginalResult(
        jnp.where(case, forward_score, -jnp.inf),
        alpha,
        beta,
        match,
        insert,
        delete,
        class_mass,
        terminal,
        valid,
        _status(valid, case, capacity_ok),
        evidence,
        plan.marginal_contract,
    )


def _viterbi_single(
    model: ProfileHMM, tokens: Array, mask: Array, /
) -> tuple[Array, ProfileStatePath, Array, Array, Array]:
    n = int(tokens.shape[0])
    length = model.model_length
    dtype = model.match_log_emissions.dtype
    delta = jnp.full((n + 1, length + 1, 3), -jnp.inf, dtype=dtype)
    predecessor = jnp.full((n + 1, length + 1, 3), -1, dtype=jnp.int8)
    pointer_kind = jnp.full((n + 1, length + 1, 3), -1, dtype=jnp.int8)
    delta = delta.at[0, 0, MATCH_STATE].set(0.0)
    for boundary in range(1, length + 1):
        candidates = (
            delta[0, boundary - 1, :]
            + model.transition_log_probabilities[boundary - 1, :, DELETE_STATE]
        )
        source = jnp.argmax(candidates).astype(jnp.int8)
        delta = delta.at[0, boundary, DELETE_STATE].set(candidates[source])
        predecessor = predecessor.at[0, boundary, DELETE_STATE].set(source)
        pointer_kind = pointer_kind.at[0, boundary, DELETE_STATE].set(DELETE_STATE)

    for position in range(1, n + 1):
        token = tokens[position - 1]
        row = jnp.full((length + 1, 3), -jnp.inf, dtype=dtype)
        row_predecessor = jnp.full((length + 1, 3), -1, dtype=jnp.int8)
        row_kind = jnp.full((length + 1, 3), -1, dtype=jnp.int8)
        for boundary in range(length + 1):
            insert_candidates = (
                delta[position - 1, boundary, :]
                + model.transition_log_probabilities[boundary, :, INSERT_STATE]
            )
            insert_source = jnp.argmax(insert_candidates).astype(jnp.int8)
            row = row.at[boundary, INSERT_STATE].set(
                insert_candidates[insert_source]
                + model.insert_log_emissions[boundary, token]
            )
            row_predecessor = row_predecessor.at[boundary, INSERT_STATE].set(
                insert_source
            )
            row_kind = row_kind.at[boundary, INSERT_STATE].set(INSERT_STATE)
            if boundary > 0:
                match_candidates = (
                    delta[position - 1, boundary - 1, :]
                    + model.transition_log_probabilities[boundary - 1, :, MATCH_STATE]
                )
                match_source = jnp.argmax(match_candidates).astype(jnp.int8)
                row = row.at[boundary, MATCH_STATE].set(
                    match_candidates[match_source]
                    + model.match_log_emissions[boundary - 1, token]
                )
                row_predecessor = row_predecessor.at[boundary, MATCH_STATE].set(
                    match_source
                )
                row_kind = row_kind.at[boundary, MATCH_STATE].set(MATCH_STATE)
        for boundary in range(1, length + 1):
            delete_candidates = (
                row[boundary - 1, :]
                + model.transition_log_probabilities[boundary - 1, :, DELETE_STATE]
            )
            delete_source = jnp.argmax(delete_candidates).astype(jnp.int8)
            row = row.at[boundary, DELETE_STATE].set(delete_candidates[delete_source])
            row_predecessor = row_predecessor.at[boundary, DELETE_STATE].set(
                delete_source
            )
            row_kind = row_kind.at[boundary, DELETE_STATE].set(DELETE_STATE)
        active = mask[position - 1]
        delta = delta.at[position].set(jnp.where(active, row, delta[position - 1]))
        predecessor = predecessor.at[position].set(
            jnp.where(active, row_predecessor, jnp.arange(3, dtype=jnp.int8)[None, :])
        )
        pointer_kind = pointer_kind.at[position].set(
            jnp.where(active, row_kind, jnp.full((length + 1, 3), 3, dtype=jnp.int8))
        )

    terminal = delta[n, length, :] + model.terminal_log_probabilities
    end_class = jnp.argmax(terminal).astype(jnp.int32)
    score = terminal[end_class]
    temporary_capacity = n + length + 1
    classes = jnp.full((temporary_capacity,), -1, dtype=jnp.int8)
    profile_positions = jnp.full((temporary_capacity,), -1, dtype=jnp.int32)
    sequence_positions = jnp.full((temporary_capacity,), -1, dtype=jnp.int32)

    def trace_step(_, state):
        t, k, state_class, alive, count, out_class, out_profile, out_sequence = state
        is_begin = alive & (t == 0) & (k == 0) & (state_class == MATCH_STATE)
        kind = jnp.where(
            is_begin,
            jnp.asarray(-1, dtype=jnp.int8),
            pointer_kind[t, k, state_class],
        )
        is_skip = alive & (kind == 3)
        write_state = alive & ~is_skip
        destination = temporary_capacity - 1 - count
        reported_class = jnp.where(is_begin, BEGIN_STATE, state_class).astype(jnp.int8)
        reported_sequence = jnp.where(
            (state_class == MATCH_STATE) | (state_class == INSERT_STATE),
            t - 1,
            -1,
        ).astype(jnp.int32)
        out_class = out_class.at[jnp.maximum(destination, 0)].set(
            jnp.where(write_state, reported_class, out_class[jnp.maximum(destination, 0)])
        )
        out_profile = out_profile.at[jnp.maximum(destination, 0)].set(
            jnp.where(write_state, k, out_profile[jnp.maximum(destination, 0)])
        )
        out_sequence = out_sequence.at[jnp.maximum(destination, 0)].set(
            jnp.where(
                write_state, reported_sequence, out_sequence[jnp.maximum(destination, 0)]
            )
        )
        count = count + write_state.astype(jnp.int32)
        previous_class = jnp.where(
            is_begin, state_class, predecessor[t, k, state_class].astype(jnp.int32)
        )
        previous_t = jnp.where(
            is_skip | (state_class == MATCH_STATE) | (state_class == INSERT_STATE),
            t - 1,
            t,
        )
        previous_k = jnp.where(
            is_skip,
            k,
            jnp.where(
                (state_class == MATCH_STATE) | (state_class == DELETE_STATE),
                k - 1,
                k,
            ),
        )
        next_alive = alive & ~is_begin & (previous_class >= 0)
        return (
            jnp.where(next_alive, previous_t, t),
            jnp.where(next_alive, previous_k, k),
            jnp.where(next_alive, previous_class, state_class),
            next_alive,
            count,
            out_class,
            out_profile,
            out_sequence,
        )

    initial = (
        jnp.asarray(n, dtype=jnp.int32),
        jnp.asarray(length, dtype=jnp.int32),
        end_class,
        jnp.isfinite(score),
        jnp.asarray(0, dtype=jnp.int32),
        classes,
        profile_positions,
        sequence_positions,
    )
    traced = jax.lax.fori_loop(0, temporary_capacity, trace_step, initial)
    count = traced[4]
    source = jnp.arange(temporary_capacity, dtype=jnp.int32) + (
        temporary_capacity - count
    )
    packed_class = traced[5][jnp.clip(source, 0, temporary_capacity - 1)]
    packed_profile = traced[6][jnp.clip(source, 0, temporary_capacity - 1)]
    packed_sequence = traced[7][jnp.clip(source, 0, temporary_capacity - 1)]
    output_capacity = n + length + 2
    state_class = jnp.full((output_capacity,), -1, dtype=jnp.int8)
    profile_position = jnp.full((output_capacity,), -1, dtype=jnp.int32)
    sequence_position = jnp.full((output_capacity,), -1, dtype=jnp.int32)
    state_class = state_class.at[:temporary_capacity].set(
        jnp.where(jnp.arange(temporary_capacity) < count, packed_class, -1)
    )
    profile_position = profile_position.at[:temporary_capacity].set(
        jnp.where(jnp.arange(temporary_capacity) < count, packed_profile, -1)
    )
    sequence_position = sequence_position.at[:temporary_capacity].set(
        jnp.where(jnp.arange(temporary_capacity) < count, packed_sequence, -1)
    )
    state_class = state_class.at[count].set(END_STATE)
    profile_position = profile_position.at[count].set(length)
    sequence_position = sequence_position.at[count].set(-1)
    path_length = count + 1
    path_valid = jnp.arange(output_capacity) < path_length
    path = ProfileStatePath(
        state_class, profile_position, sequence_position, path_valid, path_length
    )
    reaches_begin = (state_class[0] == BEGIN_STATE) & (profile_position[0] == 0)
    reaches_end = state_class[jnp.maximum(path_length - 1, 0)] == END_STATE
    consumed = jnp.sum(
        path_valid & ((state_class == MATCH_STATE) | (state_class == INSERT_STATE)),
        dtype=jnp.int32,
    )
    return score, path, reaches_begin, reaches_end, consumed


def profile_viterbi(
    model: ProfileHMM,
    sequences: SequenceBatch,
    plan: ProfileHMMPlan | None = None,
    /,
) -> ProfileViterbiResult:
    plan, capacity_ok = _validate_inputs(model, sequences, plan)
    if not capacity_ok:
        case = _case_mask(sequences)
        records = sequences.record_capacity
        path_capacity = sequences.sequence_capacity + model.model_length + 2
        valid = jnp.zeros(case.shape, dtype=bool)
        path = ProfileStatePath(
            jnp.full((records, path_capacity), -1, dtype=jnp.int8),
            jnp.full((records, path_capacity), -1, dtype=jnp.int32),
            jnp.full((records, path_capacity), -1, dtype=jnp.int32),
            jnp.zeros((records, path_capacity), dtype=bool),
            jnp.zeros((records,), dtype=jnp.int32),
        )
        evidence = ProfileViterbiEvidence(
            jnp.zeros(case.shape, dtype=bool),
            jnp.zeros(case.shape, dtype=bool),
            jnp.zeros(case.shape, dtype=bool),
            jnp.zeros((records,), dtype=jnp.int32),
            jnp.zeros(case.shape, dtype=bool),
        )
        return ProfileViterbiResult(
            jnp.full(case.shape, -jnp.inf, dtype=model.match_log_emissions.dtype),
            path,
            valid,
            _status(valid, case, False),
            evidence,
            plan.viterbi_contract,
        )
    score, path, begins, ends, consumed = jax.vmap(
        lambda token, mask: _viterbi_single(model, token, mask)
    )(sequences.token_codes, sequences.valid_mask)
    case = _case_mask(sequences)
    expected = jnp.sum(sequences.valid_mask, axis=1, dtype=jnp.int32)
    finite = jnp.isfinite(score)
    valid = case & capacity_ok & finite & begins & ends & (consumed == expected)
    evidence = ProfileViterbiEvidence(
        jnp.full(case.shape, capacity_ok) & case,
        begins & case,
        ends & case,
        consumed,
        finite & case,
    )
    return ProfileViterbiResult(
        jnp.where(case, score, -jnp.inf),
        path,
        valid,
        _status(valid, case, capacity_ok),
        evidence,
        plan.viterbi_contract,
    )


__all__ = [
    "BEGIN_STATE",
    "DELETE_STATE",
    "END_STATE",
    "INSERT_STATE",
    "MATCH_STATE",
    "PROFILE_STATUS_CAPACITY_EXCEEDED",
    "PROFILE_STATUS_INFEASIBLE",
    "PROFILE_STATUS_VALID",
    "ProfileBackwardResult",
    "ProfileDPEvidence",
    "ProfileForwardResult",
    "ProfileHMM",
    "ProfileHMMPlan",
    "ProfileMarginalResult",
    "ProfileStatePath",
    "ProfileViterbiEvidence",
    "ProfileViterbiResult",
    "profile_backward",
    "profile_forward",
    "profile_forward_backward",
    "profile_viterbi",
]
