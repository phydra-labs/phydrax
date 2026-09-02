#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded quantum experiments with exact branches and addressed shot replay."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from .._doc import DOC_KEY0
from .._fingerprint import canonical_fingerprint
from .._sampling import derive_key, SampleAddress
from .._strict import StrictModule
from ..operators.quantum._operations import QuantumProgram
from ._quantum_measurement import (
    apply_dense_quantum_instrument,
    DenseInstrumentBranchResult,
    QuantumInstrument,
)
from ._quantum_program import (
    DenseQuantumProgramPolicy,
    DenseQuantumProgramResult,
    execute_dense_quantum_program,
    plan_dense_quantum_program,
    prepare_dense_quantum_program,
    PreparedDenseQuantumProgram,
)


_EXPERIMENT_SHOT_ADDRESS = SampleAddress(
    "quantum", "experiment-shot", target="classical-outcome", role="shot"
)


class ClassicalRegisterLayout(StrictModule):
    """Static finite classical registers used by feed-forward tables."""

    register_ids: tuple[str, ...] = eqx.field(static=True)
    bit_widths: tuple[int, ...] = eqx.field(static=True)
    value_capacities: tuple[int, ...] = eqx.field(static=True)
    total_bits: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        register_ids: Sequence[str],
        bit_widths: Sequence[int],
        /,
        *,
        maximum_total_bits: int,
    ):
        identifiers = tuple(str(value) for value in register_ids)
        widths = tuple(int(value) for value in bit_widths)
        maximum = int(maximum_total_bits)
        if not identifiers or len(identifiers) != len(widths):
            raise ValueError(
                "Classical register IDs and widths must be nonempty and aligned."
            )
        if any(not value for value in identifiers) or len(set(identifiers)) != len(
            identifiers
        ):
            raise ValueError("Classical register IDs must be unique and nonempty.")
        if any(value < 1 for value in widths) or maximum < 1:
            raise ValueError("Classical register bit capacities must be positive.")
        total = sum(widths)
        if total > maximum:
            raise MemoryError("Classical register capacity exceeds maximum_total_bits.")
        self.register_ids = identifiers
        self.bit_widths = widths
        self.value_capacities = tuple(1 << width for width in widths)
        self.total_bits = total
        self.layout_id = canonical_fingerprint(
            {
                "kind": "classical-register-layout",
                "register_ids": identifiers,
                "bit_widths": widths,
            }
        )


class QuantumExperimentProgram(StrictModule):
    """One deterministic prefix, one instrument, and a static feed-forward table."""

    prefix: QuantumProgram
    instrument: QuantumInstrument
    branch_programs: tuple[QuantumProgram, ...]
    classical_layout: ClassicalRegisterLayout
    feed_forward_branch_by_outcome: tuple[int, ...] = eqx.field(static=True)
    register_values_by_outcome: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    branch_capacity: int = eqx.field(static=True)
    experiment_id: str = eqx.field(static=True)

    def __init__(
        self,
        prefix: QuantumProgram,
        instrument: QuantumInstrument,
        branch_programs: Sequence[QuantumProgram],
        classical_layout: ClassicalRegisterLayout,
        feed_forward_branch_by_outcome: Sequence[int],
        register_values_by_outcome: Sequence[Sequence[int]],
        /,
        *,
        branch_capacity: int,
    ):
        if not isinstance(prefix, QuantumProgram):
            raise TypeError("prefix must be a QuantumProgram.")
        if not isinstance(instrument, QuantumInstrument):
            raise TypeError("instrument must be a QuantumInstrument.")
        if not isinstance(classical_layout, ClassicalRegisterLayout):
            raise TypeError("classical_layout must be ClassicalRegisterLayout.")
        branches = tuple(branch_programs)
        routes = tuple(int(value) for value in feed_forward_branch_by_outcome)
        register_table = tuple(
            tuple(int(value) for value in row) for row in register_values_by_outcome
        )
        capacity = int(branch_capacity)
        if capacity < 1 or not 1 <= len(branches) <= capacity:
            raise MemoryError(
                "Quantum experiment branch capacity is invalid or exceeded."
            )
        if any(not isinstance(branch, QuantumProgram) for branch in branches):
            raise TypeError("Every feed-forward branch must be a QuantumProgram.")
        if prefix.layout.dimension != instrument.dimension:
            raise ValueError("Prefix Hilbert dimension and instrument dimension differ.")
        if len(routes) != instrument.outcome_count or len(register_table) != len(routes):
            raise ValueError("Feed-forward tables require exactly one row per outcome.")
        if any(not 0 <= route < len(branches) for route in routes):
            raise ValueError("Feed-forward branch index is outside branch_programs.")
        if any(len(row) != len(classical_layout.register_ids) for row in register_table):
            raise ValueError("Every classical table row must fill every register.")
        for row in register_table:
            if any(
                not 0 <= value < limit
                for value, limit in zip(
                    row, classical_layout.value_capacities, strict=True
                )
            ):
                raise ValueError("A feed-forward register value exceeds its bit width.")
        for branch in branches:
            if (
                branch.state_kind != "density-matrix"
                or branch.layout.layout_id != prefix.layout.layout_id
            ):
                raise ValueError(
                    "Feed-forward branches must be density programs on the prefix layout."
                )
        self.prefix = prefix
        self.instrument = instrument
        self.branch_programs = branches
        self.classical_layout = classical_layout
        self.feed_forward_branch_by_outcome = routes
        self.register_values_by_outcome = register_table
        self.branch_capacity = capacity
        self.experiment_id = canonical_fingerprint(
            {
                "kind": "quantum-experiment-program",
                "prefix": prefix.program_id,
                "instrument": instrument.instrument_id,
                "branches": tuple(branch.program_id for branch in branches),
                "classical_layout": classical_layout.layout_id,
                "feed_forward": routes,
                "register_values": register_table,
                "branch_capacity": capacity,
            }
        )


class PreparedQuantumExperiment(StrictModule):
    program: QuantumExperimentProgram
    prefix: PreparedDenseQuantumProgram
    branches: tuple[PreparedDenseQuantumProgram, ...]
    policy: DenseQuantumProgramPolicy
    prepared_id: str = eqx.field(static=True)


def prepare_quantum_experiment(
    program: QuantumExperimentProgram,
    policy: DenseQuantumProgramPolicy,
    /,
) -> PreparedQuantumExperiment:
    if not isinstance(program, QuantumExperimentProgram) or not isinstance(
        policy, DenseQuantumProgramPolicy
    ):
        raise TypeError("program/policy types are invalid.")
    prefix = prepare_dense_quantum_program(
        program.prefix, plan_dense_quantum_program(program.prefix, policy)
    )
    branches = tuple(
        prepare_dense_quantum_program(branch, plan_dense_quantum_program(branch, policy))
        for branch in program.branch_programs
    )
    identifier = canonical_fingerprint(
        {
            "kind": "prepared-quantum-experiment",
            "experiment": program.experiment_id,
            "policy": policy.policy_id,
        }
    )
    return PreparedQuantumExperiment(program, prefix, branches, policy, identifier)


class QuantumExperimentExactResult(StrictModule):
    prefix_result: DenseQuantumProgramResult
    instrument_result: DenseInstrumentBranchResult
    branch_densities: Array
    branch_status: Array
    branch_execution_valid: Array
    zero_probability: Array
    register_values_by_outcome: Array
    feed_forward_branch_by_outcome: Array
    normalization_applied: Array
    valid: Array
    sampling_tolerance: float = eqx.field(static=True)
    experiment_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


def execute_quantum_experiment_exact(
    prepared: PreparedQuantumExperiment,
    initial_state: ArrayLike,
    /,
    *,
    zero_probability_tolerance: float = 1e-12,
) -> QuantumExperimentExactResult:
    """Enumerate the complete fixed outcome table, including zero-mass branches."""
    if not isinstance(prepared, PreparedQuantumExperiment):
        raise TypeError("prepared must be PreparedQuantumExperiment.")
    prefix_result = execute_dense_quantum_program(prepared.prefix, initial_state)
    instrument_result = apply_dense_quantum_instrument(
        prepared.program.instrument,
        prefix_result.final_state,
        zero_probability_tolerance=zero_probability_tolerance,
    )
    densities: list[Array] = []
    statuses: list[Array] = []
    execution_valid: list[Array] = []
    for outcome, branch_index in enumerate(
        prepared.program.feed_forward_branch_by_outcome
    ):
        branch_result = execute_dense_quantum_program(
            prepared.branches[branch_index],
            instrument_result.conditional_densities[outcome],
        )
        densities.append(branch_result.final_state)
        statuses.append(branch_result.diagnostics.status)
        execution_valid.append(branch_result.diagnostics.successful)
    status = jnp.stack(statuses)
    execution_valid_ = jnp.stack(execution_valid)
    zero = instrument_result.zero_probability
    active_valid = zero | execution_valid_
    valid = (
        prefix_result.diagnostics.successful
        & instrument_result.valid
        & jnp.all(active_valid)
        & jnp.all(jnp.isfinite(jnp.stack(densities)))
    )
    return QuantumExperimentExactResult(
        prefix_result,
        instrument_result,
        jnp.stack(densities),
        status,
        execution_valid_,
        zero,
        jnp.asarray(prepared.program.register_values_by_outcome, dtype=jnp.int32),
        jnp.asarray(prepared.program.feed_forward_branch_by_outcome, dtype=jnp.int32),
        instrument_result.normalization_applied,
        valid,
        prepared.program.instrument.tolerance,
        prepared.program.experiment_id,
        prepared.prepared_id,
    )


class QuantumShotBatchResult(StrictModule):
    outcomes: Array
    counts: Array
    branch_indices: Array
    classical_registers: Array
    addressed_shot_indices: Array
    sampling_log_normalizer: Array
    valid: Array
    root_key: Array
    first_shot_address: int = eqx.field(static=True)
    shot_count: int = eqx.field(static=True)
    experiment_id: str = eqx.field(static=True)


def sample_quantum_experiment(
    exact: QuantumExperimentExactResult,
    /,
    *,
    shots: int,
    first_shot_address: int = 0,
    key: Key[Array, ""] = DOC_KEY0,
) -> QuantumShotBatchResult:
    """Sample addressed shots; concatenated batches exactly replay one large batch."""
    if not isinstance(exact, QuantumExperimentExactResult):
        raise TypeError("exact must be QuantumExperimentExactResult.")
    count = int(shots)
    first = int(first_shot_address)
    if count < 0 or first < 0:
        raise ValueError("shots and first_shot_address must be nonnegative.")
    probabilities = exact.instrument_result.probabilities
    logits = jnp.where(probabilities > 0.0, jnp.log(probabilities), -jnp.inf)
    log_normalizer = jax.scipy.special.logsumexp(logits)
    addresses = jnp.arange(first, first + count, dtype=jnp.uint32)
    if count:
        keys = jax.vmap(
            lambda address: derive_key(key, _EXPERIMENT_SHOT_ADDRESS, address, 0)
        )(addresses)
        outcomes = jax.vmap(lambda shot_key: jr.categorical(shot_key, logits))(keys)
        counts = jnp.bincount(outcomes, length=probabilities.shape[0])
        branches = exact.feed_forward_branch_by_outcome[outcomes]
        registers = exact.register_values_by_outcome[outcomes]
    else:
        outcomes = jnp.empty((0,), dtype=jnp.int32)
        counts = jnp.zeros(probabilities.shape, dtype=jnp.int32)
        branches = jnp.empty((0,), dtype=jnp.int32)
        registers = jnp.empty(
            (0, exact.register_values_by_outcome.shape[1]), dtype=jnp.int32
        )
    valid = (
        exact.valid
        & jnp.all(probabilities >= 0.0)
        & jnp.isfinite(log_normalizer)
        & (jnp.abs(jnp.expm1(log_normalizer)) <= exact.sampling_tolerance)
    )
    return QuantumShotBatchResult(
        outcomes,
        counts,
        branches,
        registers,
        addresses,
        log_normalizer,
        valid,
        jnp.asarray(key),
        first,
        count,
        exact.experiment_id,
    )


class StochasticGradientEstimatorEvidence(StrictModule):
    estimate: Array
    exact_gradient: Array
    standard_error: Array
    absolute_error: Array
    finite: Array
    zero_probability_outcomes: Array
    sampled_zero_probability_outcomes: Array
    unsupported_zero_probability_score_terms: Array
    valid: Array
    shot_count: int = eqx.field(static=True)
    estimator: str = eqx.field(static=True)


def estimate_quantum_experiment_gradient(
    exact: QuantumExperimentExactResult,
    shots: QuantumShotBatchResult,
    probability_jacobian: ArrayLike,
    outcome_values: ArrayLike,
    /,
) -> StochasticGradientEstimatorEvidence:
    """Return score-function and exact finite-outcome gradients side by side."""
    if not isinstance(exact, QuantumExperimentExactResult) or not isinstance(
        shots, QuantumShotBatchResult
    ):
        raise TypeError("exact/shots types are invalid.")
    jacobian = jnp.asarray(probability_jacobian)
    values = jnp.asarray(outcome_values)
    probabilities = exact.instrument_result.probabilities
    if jacobian.ndim != 2 or jacobian.shape[0] != probabilities.shape[0]:
        raise ValueError("probability_jacobian requires shape (outcomes, parameters).")
    if values.shape != probabilities.shape:
        raise ValueError("outcome_values requires shape (outcomes,).")
    if shots.shot_count < 1:
        raise ValueError("Stochastic gradient estimation requires at least one shot.")
    outcomes = shots.outcomes
    sampled_probabilities = probabilities[outcomes]
    sampled_zero = sampled_probabilities <= 0.0
    denominators = jnp.where(
        sampled_zero, jnp.ones_like(sampled_probabilities), sampled_probabilities
    )
    contributions = values[outcomes, None] * jacobian[outcomes] / denominators[:, None]
    contributions = jnp.where(sampled_zero[:, None], 0.0, contributions)
    estimate = jnp.mean(contributions, axis=0)
    standard_error = jnp.std(contributions, axis=0, ddof=0) / jnp.sqrt(
        jnp.asarray(shots.shot_count, dtype=contributions.real.dtype)
    )
    exact_gradient = jnp.sum(values[:, None] * jacobian, axis=0)
    finite = (
        jnp.all(jnp.isfinite(estimate))
        & jnp.all(jnp.isfinite(standard_error))
        & jnp.all(jnp.isfinite(exact_gradient))
    )
    sampled_zero_count = jnp.sum(sampled_zero, dtype=jnp.int32)
    unsupported_zero = jnp.sum(
        (probabilities <= 0.0)
        & jnp.any(jnp.abs(values[:, None] * jacobian) > 0.0, axis=1),
        dtype=jnp.int32,
    )
    return StochasticGradientEstimatorEvidence(
        estimate,
        exact_gradient,
        standard_error,
        jnp.abs(estimate - exact_gradient),
        finite,
        jnp.sum(probabilities <= 0.0, dtype=jnp.int32),
        sampled_zero_count,
        unsupported_zero,
        finite & shots.valid & (sampled_zero_count == 0) & (unsupported_zero == 0),
        shots.shot_count,
        "addressed-score-function-with-exact-enumeration-reference",
    )


__all__ = [
    "ClassicalRegisterLayout",
    "PreparedQuantumExperiment",
    "QuantumExperimentExactResult",
    "QuantumExperimentProgram",
    "QuantumShotBatchResult",
    "StochasticGradientEstimatorEvidence",
    "estimate_quantum_experiment_gradient",
    "execute_quantum_experiment_exact",
    "prepare_quantum_experiment",
    "sample_quantum_experiment",
]
