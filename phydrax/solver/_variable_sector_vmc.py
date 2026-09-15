#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Persistent reversible-jump VMC and finite-sector TDVP runtime."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import DenseLinearOperator, LinearSystem, solve
from ..operators.quantum._amplitude import LogAmplitude
from ..operators.quantum.variable_sector import (
    AbstractVariableSectorLocalOperator,
    VariableParticleConfiguration,
    VariableSectorMeasure,
    VariableSectorProposal,
    VariableSectorSpace,
)


VariableSectorVMCStatus: TypeAlias = Literal[0, 1, 2, 3, 4]
VARIABLE_SECTOR_VMC_SUCCESS: VariableSectorVMCStatus = 0
VARIABLE_SECTOR_VMC_INVALID_CHAIN: VariableSectorVMCStatus = 1
VARIABLE_SECTOR_VMC_INVALID_LOCAL_ENERGY: VariableSectorVMCStatus = 2
VARIABLE_SECTOR_VMC_INSUFFICIENT_TAIL_SAMPLES: VariableSectorVMCStatus = 3
VARIABLE_SECTOR_VMC_CUTOFF_TAIL_REFUSED: VariableSectorVMCStatus = 4


def variable_sector_vmc_status_name(status: int | Array, /) -> str:
    names = (
        "success",
        "invalid_chain",
        "invalid_local_energy",
        "insufficient_tail_samples",
        "cutoff_tail_refused",
    )
    code = int(status)
    if code < 0 or code >= len(names):
        raise ValueError(f"Unknown variable-sector VMC status {code}.")
    return names[code]


class VariableSectorVMCPlan(StrictModule, NonTrainableState):
    """Static resource and qualification plan, independent of model parameters."""

    chain_count: int = eqx.field(static=True)
    draw_count: int = eqx.field(static=True)
    steps_per_draw: int = eqx.field(static=True)
    warmup_steps: int = eqx.field(static=True)
    minimum_tail_samples: int = eqx.field(static=True)
    tail_probability_tolerance: float = eqx.field(static=True)
    tail_standard_error_multiplier: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        chain_count: int,
        draw_count: int,
        steps_per_draw: int = 1,
        warmup_steps: int = 0,
        minimum_tail_samples: int = 128,
        tail_probability_tolerance: float = 1e-3,
        tail_standard_error_multiplier: float = 3.0,
    ):
        chains, draws, transitions, warmup, minimum = map(
            int,
            (
                chain_count,
                draw_count,
                steps_per_draw,
                warmup_steps,
                minimum_tail_samples,
            ),
        )
        tolerance, multiplier = (
            float(tail_probability_tolerance),
            float(tail_standard_error_multiplier),
        )
        if chains < 1 or draws < 1 or transitions < 1 or warmup < 0 or minimum < 1:
            raise ValueError("VMC resource counts are invalid.")
        if (
            not np.isfinite(tolerance)
            or tolerance <= 0
            or tolerance >= 1
            or not np.isfinite(multiplier)
            or multiplier <= 0
        ):
            raise ValueError("Tail qualification parameters are invalid.")
        self.chain_count = chains
        self.draw_count = draws
        self.steps_per_draw = transitions
        self.warmup_steps = warmup
        self.minimum_tail_samples = minimum
        self.tail_probability_tolerance = tolerance
        self.tail_standard_error_multiplier = multiplier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "variable-sector-vmc-plan",
                "chain_count": chains,
                "draw_count": draws,
                "steps_per_draw": transitions,
                "warmup_steps": warmup,
                "minimum_tail_samples": minimum,
                "tail_probability_tolerance": tolerance,
                "tail_standard_error_multiplier": multiplier,
            }
        )


class PreparedVariableSectorVMC(StrictModule):
    """A model/operator/proposal binding with immutable initial chains."""

    plan: VariableSectorVMCPlan
    space: VariableSectorSpace
    measure: VariableSectorMeasure
    proposal: VariableSectorProposal
    operator: AbstractVariableSectorLocalOperator
    model: Any
    initial_coordinates: Array
    initial_active_mask: Array
    initial_species: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: VariableSectorVMCPlan,
        model: Any,
        operator: AbstractVariableSectorLocalOperator,
        proposal: VariableSectorProposal,
        measure: VariableSectorMeasure,
        initial_configurations: Sequence[VariableParticleConfiguration],
        /,
    ):
        if not isinstance(plan, VariableSectorVMCPlan):
            raise TypeError("plan must be VariableSectorVMCPlan.")
        if not callable(model):
            raise TypeError("model must be callable.")
        if not isinstance(operator, AbstractVariableSectorLocalOperator):
            raise TypeError(
                "operator must implement AbstractVariableSectorLocalOperator."
            )
        if not isinstance(proposal, VariableSectorProposal) or not isinstance(
            measure, VariableSectorMeasure
        ):
            raise TypeError("proposal/measure must be variable-sector values.")
        space = proposal.space
        if (
            operator.space.space_id != space.space_id
            or measure.space.space_id != space.space_id
        ):
            raise ValueError("operator, proposal, and measure must use one sector space.")
        configurations = tuple(initial_configurations)
        if len(configurations) != plan.chain_count or any(
            not isinstance(value, VariableParticleConfiguration)
            for value in configurations
        ):
            raise ValueError(
                "initial_configurations must contain exactly chain_count states."
            )
        canonical = tuple(value.canonicalized() for value in configurations)
        if any(
            value.coordinates.shape != (space.capacity, space.dimension)
            for value in canonical
        ):
            raise ValueError("Every initial configuration must match the sector space.")
        coordinates = jnp.stack(tuple(value.coordinates for value in canonical))
        active = jnp.stack(tuple(value.active_mask for value in canonical))
        species = jnp.stack(tuple(value.species for value in canonical))
        for value in canonical:
            amplitude = model(value)
            if not isinstance(amplitude, LogAmplitude) or amplitude.log_abs.shape != ():
                raise TypeError("model must return one scalar LogAmplitude.")
            if not bool(
                jnp.asarray(space.valid(value) & amplitude.valid & amplitude.nonzero)
            ):
                raise ValueError(
                    "Every initial chain must have a finite nonzero amplitude."
                )
        self.plan = plan
        self.space = space
        self.measure = measure
        self.proposal = proposal
        self.operator = operator
        self.model = model
        self.initial_coordinates = coordinates
        self.initial_active_mask = active
        self.initial_species = species
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-variable-sector-vmc",
                "plan": plan.plan_id,
                "space": space.space_id,
                "measure": measure.measure_id,
                "proposal": proposal.proposal_id,
                "operator": operator.operator_id,
                "model_arrays": array_tree_fingerprint(eqx.filter(model, eqx.is_array)),
                "initial_configurations": array_tree_fingerprint(
                    (coordinates, active, species)
                ),
            }
        )

    def initial_state(self, *, key: Key[Array, ""] = jr.key(0)) -> VariableSectorVMCState:
        log_target, valid = _batched_log_target(
            self,
            self.initial_coordinates,
            self.initial_active_mask,
            self.initial_species,
        )
        return VariableSectorVMCState(
            coordinates=self.initial_coordinates,
            active_mask=self.initial_active_mask,
            species=self.initial_species,
            log_target=log_target,
            valid=valid,
            transition_index=jnp.asarray(0, dtype=jnp.uint32),
            warmup_remaining=jnp.asarray(self.plan.warmup_steps, dtype=jnp.int32),
            root_key=jnp.asarray(key),
            prepared_id=self.prepared_id,
        )


class VariableSectorVMCState(StrictModule):
    """Persistent reversible-jump chains; rejected moves never mutate the state."""

    coordinates: Array
    active_mask: Array
    species: Array
    log_target: Array
    valid: Array
    transition_index: Array
    warmup_remaining: Array
    root_key: Array
    prepared_id: str = eqx.field(static=True)

    def configuration(self, chain: int, /) -> VariableParticleConfiguration:
        index = int(chain)
        if index < 0 or index >= self.coordinates.shape[0]:
            raise IndexError("chain index is outside the persistent state.")
        return VariableParticleConfiguration(
            self.coordinates[index], self.active_mask[index], self.species[index]
        )


class SectorTailEvidence(StrictModule, NonTrainableState):
    """Observed sector histogram and conservative capacity-tail decision."""

    total_histogram: Array
    species_histogram: Array
    sample_count: Array
    cutoff_count: Array
    cutoff_probability: Array
    cutoff_standard_error: Array
    cutoff_upper_bound: Array
    sufficient_samples: Array
    below_tolerance: Array
    status: Array
    capacity: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    method: str = eqx.field(static=True)

    @property
    def accepted(self) -> Array:
        return (
            self.sufficient_samples
            & self.below_tolerance
            & (self.status == VARIABLE_SECTOR_VMC_SUCCESS)
        )


class VariableSectorVMCResult(StrictModule):
    coordinates: Array
    active_mask: Array
    species: Array
    log_target: Array
    local_energy: Array
    local_energy_valid: Array
    accepted: Array
    log_acceptance_ratio: Array
    proposal_valid: Array
    final_state: VariableSectorVMCState
    tail_evidence: SectorTailEvidence
    energy_mean: Array
    energy_variance: Array
    status: Array
    valid: Array
    prepared_id: str = eqx.field(static=True)

    @property
    def acceptance_rate(self) -> Array:
        return jnp.mean(self.accepted.astype(float))


class _TransitionBatch(StrictModule):
    coordinates: Array
    active_mask: Array
    species: Array
    log_target: Array
    valid: Array
    accepted: Array
    log_acceptance_ratio: Array
    proposal_valid: Array


def prepare_variable_sector_vmc(
    plan: VariableSectorVMCPlan,
    model: Any,
    operator: AbstractVariableSectorLocalOperator,
    proposal: VariableSectorProposal,
    measure: VariableSectorMeasure,
    initial_configurations: Sequence[VariableParticleConfiguration],
    /,
) -> PreparedVariableSectorVMC:
    return PreparedVariableSectorVMC(
        plan, model, operator, proposal, measure, initial_configurations
    )


def _one_log_target(
    prepared: PreparedVariableSectorVMC,
    configuration: VariableParticleConfiguration,
    /,
) -> tuple[Array, Array]:
    amplitude = prepared.model(configuration)
    if not isinstance(amplitude, LogAmplitude):
        raise TypeError("model must return LogAmplitude.")
    valid = prepared.space.valid(configuration) & amplitude.valid & amplitude.nonzero
    log_target = 2.0 * amplitude.log_abs + prepared.measure.log_sector_factor(
        configuration
    )
    return jnp.where(valid, log_target, -jnp.inf), valid & jnp.isfinite(log_target)


def _batched_log_target(
    prepared: PreparedVariableSectorVMC,
    coordinates: Array,
    active_mask: Array,
    species: Array,
    /,
) -> tuple[Array, Array]:
    def evaluate(coordinate, active, labels):
        return _one_log_target(
            prepared, VariableParticleConfiguration(coordinate, active, labels)
        )

    return jax.vmap(evaluate)(coordinates, active_mask, species)


def _transition(
    prepared: PreparedVariableSectorVMC,
    state: VariableSectorVMCState,
    /,
) -> tuple[VariableSectorVMCState, _TransitionBatch]:
    chain_indices = jnp.arange(prepared.plan.chain_count, dtype=jnp.uint32)
    base_index = state.transition_index * jnp.asarray(
        prepared.plan.chain_count, dtype=jnp.uint32
    )
    keys = jax.vmap(lambda index: jr.fold_in(state.root_key, base_index + index))(
        chain_indices
    )

    def one(coordinate, active, labels, current_log_target, current_valid, key):
        proposal_key, acceptance_key = jr.split(key)
        current = VariableParticleConfiguration(coordinate, active, labels)
        proposed = prepared.proposal.sample(proposal_key, current)
        log_forward = prepared.proposal.log_prob(proposed, current)
        log_reverse = prepared.proposal.log_prob(current, proposed)
        proposed_log_target, target_valid = _one_log_target(prepared, proposed)
        proposal_valid = (
            jnp.isfinite(log_forward)
            & jnp.isfinite(log_reverse)
            & prepared.space.valid(proposed)
        )
        log_ratio = proposed_log_target - current_log_target + log_reverse - log_forward
        log_ratio = jnp.where(
            current_valid & target_valid & proposal_valid, log_ratio, -jnp.inf
        )
        accepted = jnp.log(
            jr.uniform(acceptance_key, minval=0.0, maxval=1.0)
        ) < jnp.minimum(log_ratio, 0.0)
        return (
            jnp.where(accepted, proposed.coordinates, current.coordinates),
            jnp.where(accepted, proposed.active_mask, current.active_mask),
            jnp.where(accepted, proposed.species, current.species),
            jnp.where(accepted, proposed_log_target, current_log_target),
            jnp.where(accepted, target_valid, current_valid),
            accepted,
            log_ratio,
            proposal_valid,
        )

    values = jax.vmap(one)(
        state.coordinates,
        state.active_mask,
        state.species,
        state.log_target,
        state.valid,
        keys,
    )
    next_state = VariableSectorVMCState(
        coordinates=values[0],
        active_mask=values[1],
        species=values[2],
        log_target=values[3],
        valid=values[4],
        transition_index=state.transition_index + jnp.asarray(1, dtype=jnp.uint32),
        warmup_remaining=jnp.maximum(state.warmup_remaining - 1, 0),
        root_key=state.root_key,
        prepared_id=state.prepared_id,
    )
    return next_state, _TransitionBatch(
        coordinates=values[0],
        active_mask=values[1],
        species=values[2],
        log_target=values[3],
        valid=values[4],
        accepted=values[5],
        log_acceptance_ratio=values[6],
        proposal_valid=values[7],
    )


def _tail_evidence(
    prepared: PreparedVariableSectorVMC,
    active_mask: Array,
    species: Array,
    /,
) -> SectorTailEvidence:
    total_count = jnp.sum(active_mask, axis=-1, dtype=jnp.int32).reshape((-1,))
    sample_count = jnp.asarray(total_count.size, dtype=jnp.int32)
    total_histogram = jnp.sum(
        total_count[:, None]
        == jnp.arange(prepared.space.capacity + 1, dtype=jnp.int32)[None, :],
        axis=0,
    ).astype(jnp.int32)
    species_counts = []
    for species_index in range(prepared.space.species_count):
        count = jnp.sum(
            active_mask & (species == species_index), axis=-1, dtype=jnp.int32
        ).reshape((-1,))
        species_counts.append(
            jnp.sum(
                count[:, None]
                == jnp.arange(prepared.space.capacity + 1, dtype=jnp.int32)[None, :],
                axis=0,
            ).astype(jnp.int32)
        )
    species_histogram = jnp.stack(species_counts)
    cutoff_count = total_histogram[-1]
    probability = cutoff_count.astype(float) / sample_count.astype(float)
    standard_error = jnp.sqrt(
        probability * (1.0 - probability) / sample_count.astype(float)
    )
    zero_event_bound = prepared.plan.tail_standard_error_multiplier / (
        sample_count.astype(float) + 1.0
    )
    upper = jnp.minimum(
        1.0,
        jnp.where(
            cutoff_count == 0,
            zero_event_bound,
            probability + prepared.plan.tail_standard_error_multiplier * standard_error,
        ),
    )
    sufficient = sample_count >= prepared.plan.minimum_tail_samples
    below = upper <= prepared.plan.tail_probability_tolerance
    status = jnp.where(
        ~sufficient,
        VARIABLE_SECTOR_VMC_INSUFFICIENT_TAIL_SAMPLES,
        jnp.where(
            below, VARIABLE_SECTOR_VMC_SUCCESS, VARIABLE_SECTOR_VMC_CUTOFF_TAIL_REFUSED
        ),
    )
    return SectorTailEvidence(
        total_histogram=total_histogram,
        species_histogram=species_histogram,
        sample_count=sample_count,
        cutoff_count=cutoff_count,
        cutoff_probability=probability,
        cutoff_standard_error=standard_error,
        cutoff_upper_bound=upper,
        sufficient_samples=sufficient,
        below_tolerance=below,
        status=jnp.asarray(status, dtype=jnp.int32),
        capacity=prepared.space.capacity,
        tolerance=prepared.plan.tail_probability_tolerance,
        method="empirical-cutoff-mass-with-conservative-binomial-upper-bound",
    )


def run_variable_sector_vmc(
    prepared: PreparedVariableSectorVMC,
    state: VariableSectorVMCState | None = None,
    /,
) -> VariableSectorVMCResult:
    """Advance persistent chains and refuse finite-cutoff claims with visible tail."""
    if not isinstance(prepared, PreparedVariableSectorVMC):
        raise TypeError("prepared must be PreparedVariableSectorVMC.")
    current = prepared.initial_state() if state is None else state
    if (
        not isinstance(current, VariableSectorVMCState)
        or current.prepared_id != prepared.prepared_id
    ):
        raise ValueError("state must belong to the prepared VMC runtime.")
    for _ in range(prepared.plan.warmup_steps):
        candidate, _ = _transition(prepared, current)
        active = current.warmup_remaining > 0
        current = jax.tree_util.tree_map(
            lambda new, old: jnp.where(active, new, old) if eqx.is_array(new) else new,
            candidate,
            current,
        )

    coordinate_draws = []
    active_draws = []
    species_draws = []
    target_draws = []
    valid_draws = []
    accepted_draws = []
    ratio_draws = []
    proposal_valid_draws = []
    for _ in range(prepared.plan.draw_count):
        step_accepted = []
        step_ratio = []
        step_proposal_valid = []
        batch = None
        for _ in range(prepared.plan.steps_per_draw):
            current, batch = _transition(prepared, current)
            step_accepted.append(batch.accepted)
            step_ratio.append(batch.log_acceptance_ratio)
            step_proposal_valid.append(batch.proposal_valid)
        if batch is None:
            raise RuntimeError("VMC draw executed no transitions.")
        coordinate_draws.append(batch.coordinates)
        active_draws.append(batch.active_mask)
        species_draws.append(batch.species)
        target_draws.append(batch.log_target)
        valid_draws.append(batch.valid)
        accepted_draws.append(jnp.stack(step_accepted, axis=1))
        ratio_draws.append(jnp.stack(step_ratio, axis=1))
        proposal_valid_draws.append(jnp.stack(step_proposal_valid, axis=1))
    coordinates = jnp.stack(coordinate_draws, axis=1)
    active_mask = jnp.stack(active_draws, axis=1)
    species = jnp.stack(species_draws, axis=1)
    log_target = jnp.stack(target_draws, axis=1)
    chain_valid = jnp.stack(valid_draws, axis=1)
    accepted = jnp.stack(accepted_draws, axis=1)
    ratios = jnp.stack(ratio_draws, axis=1)
    proposal_valid = jnp.stack(proposal_valid_draws, axis=1)

    def local_energy(coordinate, active, labels):
        result = prepared.operator.local_value(
            prepared.model, VariableParticleConfiguration(coordinate, active, labels)
        )
        return result.value, result.valid

    energies, energy_valid = jax.vmap(
        jax.vmap(local_energy, in_axes=(0, 0, 0)), in_axes=(0, 0, 0)
    )(coordinates, active_mask, species)
    valid_samples = chain_valid & energy_valid
    safe_energy = jnp.where(valid_samples, energies, 0.0)
    count = jnp.sum(valid_samples)
    energy_mean = jnp.sum(safe_energy) / jnp.maximum(count, 1)
    energy_variance = jnp.sum(
        jnp.where(valid_samples, jnp.abs(energies - energy_mean) ** 2, 0.0)
    ) / jnp.maximum(count - 1, 1)
    tail = _tail_evidence(prepared, active_mask, species)
    status = jnp.where(
        ~jnp.all(chain_valid),
        VARIABLE_SECTOR_VMC_INVALID_CHAIN,
        jnp.where(
            ~jnp.all(energy_valid),
            VARIABLE_SECTOR_VMC_INVALID_LOCAL_ENERGY,
            tail.status,
        ),
    )
    return VariableSectorVMCResult(
        coordinates=coordinates,
        active_mask=active_mask,
        species=species,
        log_target=log_target,
        local_energy=energies,
        local_energy_valid=energy_valid,
        accepted=accepted,
        log_acceptance_ratio=ratios,
        proposal_valid=proposal_valid,
        final_state=current,
        tail_evidence=tail,
        energy_mean=energy_mean,
        energy_variance=energy_variance,
        status=jnp.asarray(status, dtype=jnp.int32),
        valid=(status == VARIABLE_SECTOR_VMC_SUCCESS) & jnp.isfinite(energy_mean),
        prepared_id=prepared.prepared_id,
    )


ParameterMode = Literal["real", "holomorphic"]
EvolutionKind = Literal["imaginary-time", "real-time"]


class StochasticReconfigurationResult(StrictModule):
    metric: Array
    force: Array
    right_hand_side: Array
    update: Array
    energy_mean: Array
    residual_norm: Array
    linear_status: Array
    successful: Array
    sample_count: int = eqx.field(static=True)
    parameter_mode: str = eqx.field(static=True)
    evolution: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)


def solve_stochastic_reconfiguration(
    log_derivatives: ArrayLike,
    local_energies: ArrayLike,
    /,
    *,
    damping: float = 1e-3,
    parameter_mode: ParameterMode = "real",
    evolution: EvolutionKind = "imaginary-time",
) -> StochasticReconfigurationResult:
    """Solve the empirical quantum-geometric system through Phydrax linalg."""
    derivatives = jnp.asarray(log_derivatives)
    energies = jnp.asarray(local_energies)
    damping_ = float(damping)
    if derivatives.ndim != 2 or energies.shape != (derivatives.shape[0],):
        raise ValueError(
            "log_derivatives/local_energies require shapes (sample,param)/(sample,)."
        )
    if derivatives.shape[0] < 2 or derivatives.shape[1] < 1:
        raise ValueError("SR requires at least two samples and one parameter.")
    if not np.isfinite(damping_) or damping_ <= 0:
        raise ValueError("damping must be finite and positive.")
    if parameter_mode not in ("real", "holomorphic"):
        raise ValueError("parameter_mode must be 'real' or 'holomorphic'.")
    if evolution not in ("imaginary-time", "real-time"):
        raise ValueError("evolution must be 'imaginary-time' or 'real-time'.")
    count = derivatives.shape[0]
    centered_derivatives = derivatives - jnp.mean(derivatives, axis=0)
    centered_energy = energies - jnp.mean(energies)
    covariance = (
        contract("np,nq->pq", jnp.conj(centered_derivatives), centered_derivatives)
        / count
    )
    complex_force = (
        contract("np,n->p", jnp.conj(centered_derivatives), centered_energy) / count
    )
    if parameter_mode == "real":
        metric = jnp.real(covariance)
        force = jnp.real(complex_force)
        right_hand_side = (
            -force if evolution == "imaginary-time" else jnp.imag(complex_force)
        )
    else:
        metric = covariance
        force = complex_force
        right_hand_side = -force if evolution == "imaginary-time" else -1j * force
    metric = 0.5 * (metric + jnp.conj(metric.T))
    regularized = metric + damping_ * jnp.eye(metric.shape[0], dtype=metric.dtype)
    linear = solve(LinearSystem(DenseLinearOperator(regularized)), right_hand_side)
    update = linear.value
    residual = jnp.sqrt(jnp.sum(jnp.abs(regularized @ update - right_hand_side) ** 2))
    successful = (
        jnp.all(linear.successful)
        & jnp.all(jnp.isfinite(update))
        & jnp.isfinite(residual)
    )
    return StochasticReconfigurationResult(
        metric=metric,
        force=force,
        right_hand_side=right_hand_side,
        update=update,
        energy_mean=jnp.mean(energies),
        residual_norm=residual,
        linear_status=linear.status,
        successful=successful,
        sample_count=count,
        parameter_mode=parameter_mode,
        evolution=evolution,
        method_id="empirical-quantum-geometric-tensor-native-linear-solve",
    )


def solve_variable_sector_tdvp(
    log_derivatives: ArrayLike,
    local_energies: ArrayLike,
    /,
    *,
    damping: float = 1e-3,
    parameter_mode: ParameterMode = "real",
    evolution: EvolutionKind,
) -> StochasticReconfigurationResult:
    """Return the real- or imaginary-time TDVP tangent in the SR metric."""
    return solve_stochastic_reconfiguration(
        log_derivatives,
        local_energies,
        damping=damping,
        parameter_mode=parameter_mode,
        evolution=evolution,
    )


class VariableSectorTDVPPlan(StrictModule, NonTrainableState):
    time_step: float = eqx.field(static=True)
    step_count: int = eqx.field(static=True)
    evolution: EvolutionKind = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        time_step: float,
        step_count: int,
        /,
        *,
        evolution: EvolutionKind,
    ):
        step, count = float(time_step), int(step_count)
        if not np.isfinite(step) or step <= 0 or count < 1:
            raise ValueError("TDVP time_step/step_count must be finite and positive.")
        if evolution not in ("imaginary-time", "real-time"):
            raise ValueError("evolution must be 'imaginary-time' or 'real-time'.")
        self.time_step = step
        self.step_count = count
        self.evolution = evolution
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-step-variable-sector-tdvp-plan",
                "time_step": step,
                "step_count": count,
                "evolution": evolution,
                "integrator": "rk4",
            }
        )


class VariableSectorTDVPResult(StrictModule):
    final_parameters: Array
    times: Array
    parameter_history: Array
    finite_steps: Array
    tail_evidence: SectorTailEvidence
    status: Array
    valid: Array
    plan_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)


def evolve_variable_sector_tdvp(
    vector_field: Callable[[Array, Array], Array],
    initial_parameters: ArrayLike,
    plan: VariableSectorTDVPPlan,
    tail_evidence: SectorTailEvidence,
    /,
) -> VariableSectorTDVPResult:
    """Integrate a prepared TDVP field only after particle-tail qualification."""
    if not callable(vector_field) or not isinstance(plan, VariableSectorTDVPPlan):
        raise TypeError("vector_field/plan types are invalid.")
    if not isinstance(tail_evidence, SectorTailEvidence):
        raise TypeError("tail_evidence must be SectorTailEvidence.")
    initial = jnp.asarray(initial_parameters)
    if initial.ndim != 1 or initial.size < 1:
        raise ValueError("initial_parameters must be a nonempty vector.")
    parameters = initial
    history = [initial]
    finite_steps = []
    enabled = tail_evidence.accepted
    dt = jnp.asarray(plan.time_step, dtype=jnp.result_type(initial.real.dtype, float))
    for step in range(plan.step_count):
        time = jnp.asarray(step, dtype=dt.dtype) * dt
        k1 = jnp.asarray(vector_field(parameters, time))
        k2 = jnp.asarray(vector_field(parameters + 0.5 * dt * k1, time + 0.5 * dt))
        k3 = jnp.asarray(vector_field(parameters + 0.5 * dt * k2, time + 0.5 * dt))
        k4 = jnp.asarray(vector_field(parameters + dt * k3, time + dt))
        candidate = parameters + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        finite = jnp.all(jnp.isfinite(candidate))
        parameters = jnp.where(enabled & finite, candidate, parameters)
        history.append(parameters)
        finite_steps.append(finite)
    finite_array = jnp.stack(finite_steps)
    status = jnp.where(
        ~tail_evidence.accepted,
        VARIABLE_SECTOR_VMC_CUTOFF_TAIL_REFUSED,
        jnp.where(
            jnp.all(finite_array),
            VARIABLE_SECTOR_VMC_SUCCESS,
            VARIABLE_SECTOR_VMC_INVALID_LOCAL_ENERGY,
        ),
    )
    return VariableSectorTDVPResult(
        final_parameters=parameters,
        times=jnp.arange(plan.step_count + 1, dtype=dt.dtype) * dt,
        parameter_history=jnp.stack(history),
        finite_steps=finite_array,
        tail_evidence=tail_evidence,
        status=jnp.asarray(status, dtype=jnp.int32),
        valid=status == VARIABLE_SECTOR_VMC_SUCCESS,
        plan_id=plan.plan_id,
        method_id=f"fixed-rk4-{plan.evolution}-tdvp",
    )


__all__ = [
    "PreparedVariableSectorVMC",
    "SectorTailEvidence",
    "StochasticReconfigurationResult",
    "VARIABLE_SECTOR_VMC_CUTOFF_TAIL_REFUSED",
    "VARIABLE_SECTOR_VMC_INSUFFICIENT_TAIL_SAMPLES",
    "VARIABLE_SECTOR_VMC_INVALID_CHAIN",
    "VARIABLE_SECTOR_VMC_INVALID_LOCAL_ENERGY",
    "VARIABLE_SECTOR_VMC_SUCCESS",
    "VariableSectorTDVPPlan",
    "VariableSectorTDVPResult",
    "VariableSectorVMCPlan",
    "VariableSectorVMCResult",
    "VariableSectorVMCState",
    "evolve_variable_sector_tdvp",
    "prepare_variable_sector_vmc",
    "run_variable_sector_vmc",
    "solve_stochastic_reconfiguration",
    "solve_variable_sector_tdvp",
    "variable_sector_vmc_status_name",
]
