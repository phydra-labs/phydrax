#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._tree_math import tree_where
from .._barostat import (
    apply_isotropic_monte_carlo_barostat,
    IsotropicMonteCarloBarostatPlan,
)
from .._dynamics import AtomisticDynamicsState, PreparedAtomisticDynamics
from .._thermodynamic import PreparedThermodynamicStateTable
from .._units import AtomisticUnitSystem


_EXCHANGE_STREAM = 0x45584348
_SAMS_STREAM = 0x53414D53
_UINT32_MAX = np.iinfo(np.uint32).max


def _identity_token(identifier: str, /) -> Array:
    value = str(identifier)
    hexadecimal = frozenset("0123456789abcdef")
    if len(value) != 64 or any(character not in hexadecimal for character in value):
        raise ValueError("Continuation identity must be one canonical SHA-256 digest.")
    words = [int(value[start : start + 8], 16) for start in range(0, 64, 8)]
    return jnp.asarray(words, dtype=jnp.uint32)


class AtomisticCanonicalSamplingQualification(StrictModule, NonTrainableState):
    """Authenticated qualification bound to one dynamics and target table."""

    dynamics_id: str = eqx.field(static=True)
    thermodynamic_table_id: str = eqx.field(static=True)
    measure_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    sampling_exact: bool = eqx.field(static=True)
    sampling_bias_bound: float = eqx.field(static=True)
    qualification_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: PreparedAtomisticDynamics,
        thermodynamic: PreparedThermodynamicStateTable,
        evidence_id: str,
        /,
        *,
        sampling_exact: bool,
        sampling_bias_bound: float,
    ):
        if not isinstance(dynamics, PreparedAtomisticDynamics):
            raise TypeError("dynamics must be PreparedAtomisticDynamics.")
        if not isinstance(thermodynamic, PreparedThermodynamicStateTable):
            raise TypeError("thermodynamic must be PreparedThermodynamicStateTable.")
        thermodynamic.validate_dynamics(dynamics)
        evidence = str(evidence_id)
        bound = float(sampling_bias_bound)
        if (
            not evidence
            or evidence != evidence.strip()
            or not isinstance(sampling_exact, bool)
            or not math.isfinite(bound)
            or bound < 0.0
            or (sampling_exact and bound != 0.0)
        ):
            raise ValueError(
                "Canonical-sampling evidence, exactness, or bias bound is invalid."
            )
        self.dynamics_id = dynamics.prepared_id
        self.thermodynamic_table_id = thermodynamic.table_id
        self.measure_id = thermodynamic.phase_space_measure_id
        self.evidence_id = evidence
        self.sampling_exact = sampling_exact
        self.sampling_bias_bound = bound
        self.qualification_id = canonical_fingerprint(
            {
                "kind": "atomistic-canonical-sampling-qualification",
                "dynamics": dynamics.prepared_id,
                "integrator": dynamics.integrator.plan_id,
                "thermodynamic_table": thermodynamic.table_id,
                "measure": thermodynamic.phase_space_measure_id,
                "program": thermodynamic.program_id,
                "evidence": evidence,
                "sampling_exact": sampling_exact,
                "sampling_bias_bound": bound,
            }
        )


class AtomisticReplicaExchangePlan(StrictModule, NonTrainableState):
    """Deterministic odd-even adjacent exchange schedule."""

    exchange_interval: int = eqx.field(static=True)
    realization_id: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, exchange_interval: int = 1, /, *, realization_id: int = 0):
        interval = int(exchange_interval)
        realization = int(realization_id)
        if interval <= 0 or realization < 0:
            raise ValueError("Exchange interval and realization ID are invalid.")
        self.exchange_interval = interval
        self.realization_id = realization
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomistic-replica-exchange",
                "interval": interval,
                "realization_id": realization,
                "proposal": "adjacent-odd-even",
            }
        )

    def scheduled(self, next_iteration: ArrayLike, /) -> Array:
        iteration = jnp.asarray(next_iteration, dtype=jnp.int64)
        return (iteration > 0) & (iteration % self.exchange_interval == 0)


class AtomisticSAMSPlan(StrictModule, NonTrainableState):
    """Finite-adaptation stochastic approximation over thermodynamic labels."""

    target_probabilities: Array
    move_interval: int = eqx.field(static=True)
    adaptation_steps: int = eqx.field(static=True)
    gain_exponent: float = eqx.field(static=True)
    initial_gain: float = eqx.field(static=True)
    realization_id: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        target_probabilities: ArrayLike,
        /,
        *,
        move_interval: int = 1,
        adaptation_steps: int = 1_000,
        gain_exponent: float = 0.6,
        initial_gain: float = 1.0,
        realization_id: int = 0,
    ):
        target = np.asarray(target_probabilities, dtype=np.float64).reshape((-1,))
        interval = int(move_interval)
        steps = int(adaptation_steps)
        exponent = float(gain_exponent)
        gain = float(initial_gain)
        realization = int(realization_id)
        if (
            target.size < 2
            or np.any(~np.isfinite(target))
            or np.any(target <= 0.0)
            or not np.isclose(np.sum(target), 1.0)
            or interval <= 0
            or steps < 0
            or not math.isfinite(exponent)
            or exponent <= 0.5
            or exponent > 1.0
            or not math.isfinite(gain)
            or gain <= 0.0
            or realization < 0
        ):
            raise ValueError("SAMS target, cadence, adaptation, or gain is invalid.")
        normalized = target / np.sum(target)
        self.target_probabilities = jnp.asarray(normalized)
        self.move_interval = interval
        self.adaptation_steps = steps
        self.gain_exponent = exponent
        self.initial_gain = gain
        self.realization_id = realization
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomistic-sams",
                "target": array_tree_fingerprint(normalized),
                "move_interval": interval,
                "adaptation_steps": steps,
                "gain_exponent": exponent,
                "initial_gain": gain,
                "realization_id": realization,
            }
        )

    def scheduled(self, next_iteration: ArrayLike, /) -> Array:
        iteration = jnp.asarray(next_iteration, dtype=jnp.int64)
        return (iteration > 0) & (iteration % self.move_interval == 0)


class AtomisticSAMSState(StrictModule):
    log_weights: Array
    visit_counts: Array
    adaptation_index: Array
    frozen: Array


class AtomisticReducedPotentialEvaluation(StrictModule):
    """Dimensionless dense K-by-R reduced potentials and explicit coverage."""

    values: Array
    successful: Array
    state_indices: Array
    table_id: str = eqx.field(static=True)


class AtomisticMultistateState(StrictModule):
    dynamics: AtomisticDynamicsState
    state_at_replica: Array
    reduced_potential_cache: Array
    reduced_potential_valid: Array
    cache_iteration: Array
    iteration_index: Array
    draw_index: Array
    exchange_action_counter: Array
    exchange_parity: Array
    barostat_action_counter: Array
    sams_action_counter: Array
    sams: AtomisticSAMSState
    root_key: Array
    segment_index: Array
    continuation_token: Array
    plan_id: str = eqx.field(static=True)


class AtomisticMultistateIteration(StrictModule):
    candidate_state: AtomisticMultistateState
    state: AtomisticMultistateState
    reduced_potentials: Array
    coverage: Array
    origin_state: Array
    state_at_replica_after: Array
    pair_indices: Array
    exchange_attempted: Array
    exchange_accepted: Array
    exchange_log_acceptance: Array
    sams_attempted: Array
    sams_changed: Array
    sams_adapting: Array
    dynamics_accepted: Array
    barostat_attempted: Array
    barostat_accepted: Array
    equilibrium_sample: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class AtomisticMultistatePlan(StrictModule, NonTrainableState):
    thermodynamic: PreparedThermodynamicStateTable
    qualification: AtomisticCanonicalSamplingQualification
    replica_ids: Array
    chain_indices: Array
    dependence_group_indices: Array
    exchange: AtomisticReplicaExchangePlan | None
    sams: AtomisticSAMSPlan | None
    barostat: IsotropicMonteCarloBarostatPlan | None
    replica_count: int = eqx.field(static=True)
    state_count: int = eqx.field(static=True)
    barostat_interval: int | None = eqx.field(static=True)
    repeat_index: int = eqx.field(static=True)
    run_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        thermodynamic: PreparedThermodynamicStateTable,
        replica_ids: ArrayLike,
        /,
        *,
        qualification: AtomisticCanonicalSamplingQualification,
        exchange: AtomisticReplicaExchangePlan | None = None,
        sams: AtomisticSAMSPlan | None = None,
        barostat: IsotropicMonteCarloBarostatPlan | None = None,
        barostat_interval: int | None = None,
        chain_indices: ArrayLike | None = None,
        dependence_group_indices: ArrayLike | None = None,
        repeat_index: int = 0,
        run_id: str,
    ):
        if not isinstance(thermodynamic, PreparedThermodynamicStateTable):
            raise TypeError("thermodynamic must be PreparedThermodynamicStateTable.")
        if not isinstance(qualification, AtomisticCanonicalSamplingQualification):
            raise TypeError(
                "qualification must be AtomisticCanonicalSamplingQualification."
            )
        if (
            qualification.dynamics_id != thermodynamic.dynamics_id
            or qualification.thermodynamic_table_id != thermodynamic.table_id
            or qualification.measure_id != thermodynamic.phase_space_measure_id
        ):
            raise ValueError(
                "Canonical-sampling qualification belongs to another dynamics/table target."
            )
        replicas = np.asarray(replica_ids, dtype=np.int64).reshape((-1,))
        replica_count = replicas.size
        if (
            replica_count <= 0
            or len(set(replicas.tolist())) != replica_count
            or np.any(replicas < 0)
        ):
            raise ValueError("Replica IDs must be a nonempty unique non-negative vector.")
        if exchange is not None and not isinstance(
            exchange, AtomisticReplicaExchangePlan
        ):
            raise TypeError("exchange must be AtomisticReplicaExchangePlan or None.")
        if sams is not None and not isinstance(sams, AtomisticSAMSPlan):
            raise TypeError("sams must be AtomisticSAMSPlan or None.")
        if exchange is not None and sams is not None:
            raise ValueError("Replica exchange and SAMS are distinct transition kernels.")
        if exchange is not None and (
            replica_count != thermodynamic.state_count or thermodynamic.state_count < 2
        ):
            raise ValueError("Dense replica exchange requires R equal to K and K >= 2.")
        if sams is not None and sams.target_probabilities.shape != (
            thermodynamic.state_count,
        ):
            raise ValueError("SAMS target probabilities must have one entry per state.")
        if not bool(np.all(np.asarray(thermodynamic.temperature_mask))):
            raise ValueError(
                "Multistate reduced potentials require temperature for every state."
            )
        if barostat is not None and not isinstance(
            barostat, IsotropicMonteCarloBarostatPlan
        ):
            raise TypeError("barostat must be IsotropicMonteCarloBarostatPlan or None.")
        interval = None if barostat_interval is None else int(barostat_interval)
        if (barostat is None) != (interval is None) or (
            interval is not None and interval <= 0
        ):
            raise ValueError(
                "Barostat and positive barostat_interval must be supplied together."
            )
        npt = np.asarray(thermodynamic.pressure_mask)
        if (np.any(npt) and (barostat is None or not np.all(npt))) or (
            not np.any(npt) and barostat is not None
        ):
            raise ValueError(
                "A scheduled barostat is required exactly when every state is NPT."
            )
        chains = (
            np.arange(replica_count, dtype=np.int32)
            if chain_indices is None
            else np.asarray(chain_indices, dtype=np.int32).reshape((-1,))
        )
        groups = (
            np.zeros((replica_count,), dtype=np.int32)
            if dependence_group_indices is None and exchange is not None
            else chains.copy()
            if dependence_group_indices is None
            else np.asarray(dependence_group_indices, dtype=np.int32).reshape((-1,))
        )
        if (
            chains.shape != (replica_count,)
            or groups.shape != (replica_count,)
            or np.any(chains < 0)
            or np.any(groups < 0)
        ):
            raise ValueError(
                "Chain and dependence-group indices must be non-negative vectors of length R."
            )
        if exchange is not None and len(set(groups.tolist())) != 1:
            raise ValueError(
                "An exchange-connected replica ladder must share one dependence group."
            )
        repeat = int(repeat_index)
        identifier = str(run_id)
        if (
            repeat < 0
            or repeat > _UINT32_MAX
            or not identifier
            or identifier != identifier.strip()
        ):
            raise ValueError("repeat_index or run_id is invalid.")
        self.thermodynamic = thermodynamic
        self.qualification = qualification
        self.replica_ids = jnp.asarray(replicas)
        self.chain_indices = jnp.asarray(chains)
        self.dependence_group_indices = jnp.asarray(groups)
        self.exchange = exchange
        self.sams = sams
        self.barostat = barostat
        self.replica_count = replica_count
        self.state_count = thermodynamic.state_count
        self.barostat_interval = interval
        self.repeat_index = repeat
        self.run_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomistic-multistate-plan",
                "qualification": qualification.qualification_id,
                "thermodynamic": thermodynamic.table_id,
                "replica_ids": array_tree_fingerprint(replicas),
                "chain_indices": array_tree_fingerprint(chains),
                "dependence_groups": array_tree_fingerprint(groups),
                "exchange": None if exchange is None else exchange.plan_id,
                "sams": None if sams is None else sams.plan_id,
                "barostat": None if barostat is None else barostat.plan_id,
                "barostat_interval": interval,
                "repeat_index": repeat,
                "run_id": identifier,
            }
        )

    def prepare(
        self, dynamics: PreparedAtomisticDynamics, /
    ) -> "PreparedAtomisticMultistate":
        return PreparedAtomisticMultistate(self, dynamics)


class PreparedAtomisticMultistate(StrictModule):
    plan: AtomisticMultistatePlan
    qualification: AtomisticCanonicalSamplingQualification
    dynamics: PreparedAtomisticDynamics
    thermodynamic: PreparedThermodynamicStateTable
    prepared_id: str = eqx.field(static=True)
    initial_continuation_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: AtomisticMultistatePlan,
        dynamics: PreparedAtomisticDynamics,
        /,
    ):
        if not isinstance(plan, AtomisticMultistatePlan):
            raise TypeError("plan must be AtomisticMultistatePlan.")
        if not isinstance(dynamics, PreparedAtomisticDynamics):
            raise TypeError("dynamics must be PreparedAtomisticDynamics.")
        plan.thermodynamic.validate_dynamics(dynamics)
        if (
            plan.qualification.dynamics_id != dynamics.prepared_id
            or plan.qualification.thermodynamic_table_id != plan.thermodynamic.table_id
        ):
            raise ValueError(
                "Canonical-sampling qualification is not bound to this prepared runtime."
            )
        self.plan = plan
        self.qualification = plan.qualification
        self.dynamics = dynamics
        self.thermodynamic = plan.thermodynamic
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-atomistic-multistate",
                "plan": plan.plan_id,
                "qualification": plan.qualification.qualification_id,
                "dynamics": dynamics.prepared_id,
                "thermodynamic": plan.thermodynamic.table_id,
            }
        )
        self.initial_continuation_id = canonical_fingerprint(
            {"kind": "atomistic-multistate-initial", "runtime": self.prepared_id}
        )

    @staticmethod
    def _semantic_key(root_key: Array, stream: int, counter: Array, identity: Array):
        key = jr.wrap_key_data(root_key)
        key = jr.fold_in(key, jnp.asarray(stream, dtype=jnp.uint32))
        key = jr.fold_in(key, jnp.asarray(counter, dtype=jnp.uint32))
        unsigned = jnp.asarray(identity, dtype=jnp.uint64)
        key = jr.fold_in(key, (unsigned >> jnp.uint64(32)).astype(jnp.uint32))
        return jr.fold_in(key, (unsigned & jnp.uint64(0xFFFFFFFF)).astype(jnp.uint32))

    def initialize(
        self,
        states,
        state_at_replica: ArrayLike,
        key: Key[Array, ""],
        /,
    ) -> AtomisticMultistateState:
        lanes = tuple(states)
        if len(lanes) != self.plan.replica_count or any(
            not isinstance(state, AtomisticDynamicsState) for state in lanes
        ):
            raise ValueError("Initial states must contain exactly R dynamics states.")
        if any(
            state.prepared_dynamics_id != self.dynamics.prepared_id
            or state.thermodynamic_table_id != self.thermodynamic.table_id
            for state in lanes
        ):
            raise ValueError("Initial dynamics states belong to another runtime.")
        labels = np.asarray(state_at_replica, dtype=np.int32).reshape((-1,))
        if (
            labels.shape != (self.plan.replica_count,)
            or np.any(labels < 0)
            or np.any(labels >= self.plan.state_count)
        ):
            raise ValueError("state_at_replica must contain R valid state indices.")
        if self.plan.exchange is not None and sorted(labels.tolist()) != list(
            range(self.plan.state_count)
        ):
            raise ValueError("Dense replica exchange requires a state permutation.")
        batched = jax.tree.map(lambda *values: jnp.stack(values), *lanes)
        root = key
        for identity in (
            canonical_fingerprint(
                {"kind": "atomistic-multistate-run", "run_id": self.plan.run_id}
            ),
            self.prepared_id,
        ):
            token = _identity_token(identity)
            for index in range(token.shape[0]):
                root = jr.fold_in(root, token[index])
        root = jr.fold_in(root, jnp.asarray(self.plan.repeat_index, dtype=jnp.uint32))
        root_data = jr.key_data(root).astype(jnp.uint32)
        physical_keys = jax.vmap(
            lambda replica_id: jr.key_data(
                self._semantic_key(root_data, 0, jnp.uint32(0), replica_id)
            )
        )(self.plan.replica_ids)
        batched = eqx.tree_at(
            lambda value: value.random_key,
            batched,
            physical_keys,
        )
        label_array = jnp.asarray(labels)
        rebased = jax.vmap(
            lambda lane, label: self.dynamics.rebase_thermodynamic_state(
                lane, self.thermodynamic, label
            )
        )(batched, label_array)
        checked_positions = eqx.error_if(
            rebased.accepted_state.kinematics.positions,
            ~jnp.all(rebased.successful),
            "Initial multistate force and ledger rebasing failed.",
        )
        dynamics = eqx.tree_at(
            lambda value: value.kinematics.positions,
            rebased.accepted_state,
            checked_positions,
        )
        state_count = self.plan.state_count
        replica_count = self.plan.replica_count
        dtype = dynamics.kinematics.positions.dtype
        sams = AtomisticSAMSState(
            log_weights=jnp.zeros((state_count,), dtype=dtype),
            visit_counts=jnp.zeros((state_count,), dtype=jnp.int64),
            adaptation_index=jnp.zeros((), dtype=jnp.int64),
            frozen=jnp.asarray(
                self.plan.sams is None or self.plan.sams.adaptation_steps == 0
            ),
        )
        continuation = self.initial_continuation_id
        return AtomisticMultistateState(
            dynamics=dynamics,
            state_at_replica=label_array,
            reduced_potential_cache=jnp.zeros((state_count, replica_count), dtype=dtype),
            reduced_potential_valid=jnp.zeros(
                (state_count, replica_count), dtype=jnp.bool_
            ),
            cache_iteration=jnp.asarray(-1, dtype=jnp.int64),
            iteration_index=jnp.zeros((), dtype=jnp.int64),
            draw_index=jnp.zeros((replica_count,), dtype=jnp.int64),
            exchange_action_counter=jnp.zeros((), dtype=jnp.uint32),
            exchange_parity=jnp.zeros((), dtype=jnp.int32),
            barostat_action_counter=jnp.zeros((replica_count,), dtype=jnp.uint32),
            sams_action_counter=jnp.zeros((replica_count,), dtype=jnp.uint32),
            sams=sams,
            root_key=root_data,
            segment_index=jnp.zeros((), dtype=jnp.int64),
            continuation_token=_identity_token(continuation),
            plan_id=self.prepared_id,
        )

    def reduced_potentials(
        self, dynamics_state: AtomisticDynamicsState, /
    ) -> AtomisticReducedPotentialEvaluation:
        indices = jnp.arange(self.plan.state_count, dtype=jnp.int32)
        unwrapped = jax.vmap(self.dynamics._unwrapped)(
            dynamics_state.kinematics, dynamics_state.cell_vectors
        )

        def evaluate_state(state_index):
            row = self.thermodynamic.state_at_replica(state_index)
            evaluation = jax.vmap(
                lambda lane, whole: self.dynamics._energy_configuration(
                    lane.kinematics.positions,
                    whole,
                    lane.species,
                    lane.cell_vectors,
                    lane.neighborhood,
                    row.controls,
                )
            )(dynamics_state, unwrapped)
            return evaluation.energy, evaluation.successful & row.valid

        energies, successful = jax.vmap(evaluate_state)(indices)
        if self.dynamics.system.cell is None:
            volumes = jnp.zeros((self.plan.replica_count,), dtype=energies.dtype)
        else:
            vectors = dynamics_state.cell_vectors
            volumes = jnp.abs(
                jnp.sum(vectors[:, 0] * jnp.cross(vectors[:, 1], vectors[:, 2]), axis=-1)
            )
        reduced = self.thermodynamic.beta[:, None] * (
            energies + self.thermodynamic.pressure[:, None] * volumes[None, :]
        )
        coverage = (
            successful
            & self.thermodynamic.temperature_mask[:, None]
            & jnp.isfinite(reduced)
        )
        return AtomisticReducedPotentialEvaluation(
            jnp.where(coverage, reduced, 0.0),
            coverage,
            indices,
            self.thermodynamic.table_id,
        )

    def _exchange(
        self,
        state: AtomisticMultistateState,
        reduced: AtomisticReducedPotentialEvaluation,
        scheduled: Array,
    ):
        pair_count = max(self.plan.replica_count - 1, 0)
        starts = jnp.arange(pair_count, dtype=jnp.int32)
        pair_indices = jnp.stack((starts, starts + 1), axis=-1)
        parity = state.exchange_parity & 1
        attempted = scheduled & (starts % 2 == parity)
        labels = state.state_at_replica
        left_label = labels[:-1]
        right_label = labels[1:]
        left_slot = starts
        right_slot = starts + 1
        current = (
            reduced.values[left_label, left_slot]
            + reduced.values[right_label, right_slot]
        )
        proposed = (
            reduced.values[right_label, left_slot]
            + reduced.values[left_label, right_slot]
        )
        log_acceptance = jnp.where(attempted, current - proposed, 0.0)
        exchange = self.plan.exchange
        realization = 0 if exchange is None else exchange.realization_id

        def propose(_):
            keys = jax.vmap(
                lambda start: self._semantic_key(
                    state.root_key,
                    _EXCHANGE_STREAM ^ realization,
                    state.exchange_action_counter,
                    start,
                )
            )(starts)
            uniforms = jax.vmap(lambda value: jr.uniform(value, ()))(keys)
            return attempted & (jnp.log(uniforms) < jnp.minimum(log_acceptance, 0.0))

        accepted = jax.lax.cond(
            scheduled,
            propose,
            lambda _: jnp.zeros((pair_count,), dtype=jnp.bool_),
            operand=None,
        )
        next_labels = labels
        for index in range(pair_count):
            accept = accepted[index]
            left_value = next_labels[index]
            right_value = next_labels[index + 1]
            next_labels = next_labels.at[index].set(
                jnp.where(accept, right_value, left_value)
            )
            next_labels = next_labels.at[index + 1].set(
                jnp.where(accept, left_value, right_value)
            )
        counter_valid = (~scheduled) | (
            state.exchange_action_counter < jnp.uint32(_UINT32_MAX)
        )
        valid = counter_valid & jnp.all((~attempted) | jnp.isfinite(log_acceptance))
        next_counter = state.exchange_action_counter + scheduled.astype(jnp.uint32)
        next_parity = jnp.where(scheduled, 1 - parity, parity)
        return (
            next_labels,
            pair_indices,
            attempted,
            accepted,
            log_acceptance,
            next_counter,
            next_parity,
            valid,
        )

    def _sams(
        self,
        state: AtomisticMultistateState,
        reduced: AtomisticReducedPotentialEvaluation,
        scheduled: Array,
    ):
        sams_plan = self.plan.sams
        replica_count = self.plan.replica_count
        if sams_plan is None:
            return (
                state.state_at_replica,
                jnp.zeros((replica_count,), dtype=jnp.bool_),
                jnp.zeros((replica_count,), dtype=jnp.bool_),
                state.sams_action_counter,
                state.sams,
                jnp.asarray(True),
            )
        attempted = jnp.full((replica_count,), scheduled, dtype=jnp.bool_)
        logits = (
            jnp.log(sams_plan.target_probabilities)[:, None]
            + state.sams.log_weights[:, None]
            - reduced.values
        )

        def propose_labels(_):
            keys = jax.vmap(
                lambda replica_id, counter: self._semantic_key(
                    state.root_key,
                    _SAMS_STREAM ^ sams_plan.realization_id,
                    counter,
                    replica_id,
                )
            )(self.plan.replica_ids, state.sams_action_counter)
            return jax.vmap(
                lambda action_key, lane_logits: jr.categorical(action_key, lane_logits)
            )(keys, logits.T).astype(jnp.int32)

        labels = jax.lax.cond(
            scheduled,
            propose_labels,
            lambda _: state.state_at_replica,
            operand=None,
        )
        changed = attempted & (labels != state.state_at_replica)
        counter_valid = (~scheduled) | jnp.all(
            state.sams_action_counter < jnp.uint32(_UINT32_MAX)
        )
        finite = (~scheduled) | jnp.all(jnp.isfinite(logits))
        next_counters = state.sams_action_counter + attempted.astype(jnp.uint32)
        visits = jnp.bincount(labels, length=self.plan.state_count)
        adapting = scheduled & (state.sams.adaptation_index < sams_plan.adaptation_steps)
        gain = sams_plan.initial_gain * (
            state.sams.adaptation_index.astype(reduced.values.dtype) + 1.0
        ) ** (-sams_plan.gain_exponent)
        observed = visits.astype(reduced.values.dtype) / replica_count
        increment = gain * (sams_plan.target_probabilities - observed)
        weights = jnp.where(
            adapting, state.sams.log_weights + increment, state.sams.log_weights
        )
        weights = weights - jnp.sum(weights * sams_plan.target_probabilities)
        adaptation_index = state.sams.adaptation_index + adapting.astype(jnp.int64)
        sams = AtomisticSAMSState(
            weights,
            state.sams.visit_counts + jnp.where(scheduled, visits, 0),
            adaptation_index,
            adaptation_index >= sams_plan.adaptation_steps,
        )
        return labels, attempted, changed, next_counters, sams, counter_valid & finite

    def iterate(self, state: AtomisticMultistateState, /) -> AtomisticMultistateIteration:
        if not isinstance(state, AtomisticMultistateState):
            raise TypeError("state must be AtomisticMultistateState.")
        if state.plan_id != self.prepared_id:
            raise ValueError("Multistate state belongs to another prepared runtime.")
        propagated = jax.vmap(
            lambda lane: self.dynamics.step_detailed(lane, self.thermodynamic)
        )(state.dynamics)
        dynamics_state = propagated.accepted_state
        propagation_valid = jnp.all(propagated.successful)
        next_iteration = state.iteration_index + 1
        barostat_plan = self.plan.barostat
        if barostat_plan is None:
            barostat_scheduled = jnp.asarray(False)
            barostat_attempted = jnp.zeros((self.plan.replica_count,), dtype=jnp.bool_)
            barostat_accepted = jnp.zeros_like(barostat_attempted)
            barostat_valid = jnp.asarray(True)
            barostat_counter = state.barostat_action_counter
        else:
            barostat_scheduled = next_iteration % self.plan.barostat_interval == 0
            barostat_attempted = jnp.full(
                (self.plan.replica_count,), barostat_scheduled, dtype=jnp.bool_
            )

            def apply_moves(_):
                evaluations = jax.vmap(
                    lambda lane, counter: apply_isotropic_monte_carlo_barostat(
                        self.dynamics,
                        lane,
                        self.thermodynamic,
                        barostat_plan,
                        counter,
                    )
                )(dynamics_state, state.barostat_action_counter)
                return (
                    evaluations.accepted_state,
                    evaluations.accepted,
                    jnp.all(evaluations.successful),
                )

            def skip_moves(_):
                return (
                    dynamics_state,
                    jnp.zeros((self.plan.replica_count,), dtype=jnp.bool_),
                    jnp.asarray(True),
                )

            dynamics_state, moved, moves_valid = jax.lax.cond(
                barostat_scheduled, apply_moves, skip_moves, operand=None
            )
            barostat_accepted = barostat_attempted & moved
            counter_valid = jnp.all(
                state.barostat_action_counter < jnp.uint32(_UINT32_MAX)
            )
            barostat_valid = (~barostat_scheduled) | (counter_valid & moves_valid)
            barostat_counter = state.barostat_action_counter + (
                barostat_attempted.astype(jnp.uint32)
            )
        reduced = self.reduced_potentials(dynamics_state)
        cross_valid = jnp.all(reduced.successful)
        origin = state.state_at_replica
        exchange_plan = self.plan.exchange
        exchange_scheduled = (
            jnp.asarray(False)
            if exchange_plan is None
            else exchange_plan.scheduled(next_iteration)
        )
        (
            exchange_labels,
            pair_indices,
            exchange_attempted,
            exchange_accepted,
            log_acceptance,
            exchange_counter,
            exchange_parity,
            exchange_valid,
        ) = self._exchange(state, reduced, exchange_scheduled)
        sams_plan = self.plan.sams
        sams_adapting = jnp.asarray(False) if sams_plan is None else ~state.sams.frozen
        equilibrium_sample = ~sams_adapting
        sams_scheduled = (
            jnp.asarray(False)
            if sams_plan is None
            else sams_plan.scheduled(next_iteration)
        )
        (
            next_labels,
            sams_attempted,
            sams_changed,
            sams_counter,
            sams_state,
            sams_valid,
        ) = self._sams(
            eqx.tree_at(lambda value: value.state_at_replica, state, exchange_labels),
            reduced,
            sams_scheduled,
        )
        rebased = jax.vmap(
            lambda lane, label: self.dynamics.rebase_thermodynamic_state(
                lane, self.thermodynamic, label
            )
        )(dynamics_state, next_labels)
        rebase_valid = jnp.all(rebased.successful)
        successful = (
            propagation_valid
            & barostat_valid
            & cross_valid
            & exchange_valid
            & sams_valid
            & rebase_valid
        )
        candidate = AtomisticMultistateState(
            dynamics=rebased.accepted_state,
            state_at_replica=next_labels,
            reduced_potential_cache=reduced.values,
            reduced_potential_valid=reduced.successful,
            cache_iteration=next_iteration,
            iteration_index=next_iteration,
            draw_index=state.draw_index + 1,
            exchange_action_counter=exchange_counter,
            exchange_parity=exchange_parity,
            barostat_action_counter=barostat_counter,
            sams_action_counter=sams_counter,
            sams=sams_state,
            root_key=state.root_key,
            segment_index=state.segment_index,
            continuation_token=state.continuation_token,
            plan_id=self.prepared_id,
        )
        committed = tree_where(successful, candidate, state)
        return AtomisticMultistateIteration(
            candidate_state=candidate,
            state=committed,
            reduced_potentials=jnp.where(successful, reduced.values, 0.0),
            coverage=successful & reduced.successful,
            origin_state=jnp.where(successful, origin, -1),
            state_at_replica_after=jnp.where(successful, next_labels, -1),
            pair_indices=pair_indices,
            exchange_attempted=successful & exchange_attempted,
            exchange_accepted=successful & exchange_accepted,
            exchange_log_acceptance=jnp.where(
                successful & exchange_attempted, log_acceptance, 0.0
            ),
            sams_attempted=successful & sams_attempted,
            sams_changed=successful & sams_changed,
            sams_adapting=successful & sams_adapting,
            dynamics_accepted=successful & propagated.successful,
            barostat_attempted=successful & barostat_attempted,
            barostat_accepted=successful & barostat_accepted,
            equilibrium_sample=successful & equilibrium_sample,
            successful=successful,
            plan_id=self.prepared_id,
        )


class AtomisticMultistateSegmentPlan(StrictModule, NonTrainableState):
    runtime: PreparedAtomisticMultistate
    capacity: int = eqx.field(static=True)
    expected_start_iteration: int = eqx.field(static=True)
    segment_index: int = eqx.field(static=True)
    predecessor_id: str = eqx.field(static=True)
    segment_id: str = eqx.field(static=True)

    def __init__(
        self,
        runtime: PreparedAtomisticMultistate,
        capacity: int,
        expected_start_iteration: int,
        segment_index: int,
        predecessor_id: str,
        /,
    ):
        if not isinstance(runtime, PreparedAtomisticMultistate):
            raise TypeError("runtime must be PreparedAtomisticMultistate.")
        capacity_ = int(capacity)
        start = int(expected_start_iteration)
        index = int(segment_index)
        predecessor = str(predecessor_id)
        if capacity_ <= 0 or start < 0 or index < 0 or not predecessor:
            raise ValueError("Segment capacity, watermarks, or predecessor are invalid.")
        _identity_token(predecessor)
        self.runtime = runtime
        self.capacity = capacity_
        self.expected_start_iteration = start
        self.segment_index = index
        self.predecessor_id = predecessor
        self.segment_id = canonical_fingerprint(
            {
                "kind": "atomistic-multistate-segment",
                "runtime": runtime.prepared_id,
                "capacity": capacity_,
                "start_iteration": start,
                "segment_index": index,
                "predecessor": predecessor,
            }
        )

    def run(
        self, state: AtomisticMultistateState, /
    ) -> "AtomisticMultistateSegmentResult":
        if not isinstance(state, AtomisticMultistateState):
            raise TypeError("state must be AtomisticMultistateState.")
        if state.plan_id != self.runtime.prepared_id:
            raise ValueError("Segment state belongs to another multistate runtime.")
        valid_predecessor = (
            jnp.all(
                state.continuation_token
                == jnp.asarray(_identity_token(self.predecessor_id), dtype=jnp.uint32)
            )
            & (state.iteration_index == self.expected_start_iteration)
            & (state.segment_index == self.segment_index)
        )
        checked_iteration = eqx.error_if(
            state.iteration_index,
            ~valid_predecessor,
            "Segment predecessor identity or watermark does not match state.",
        )
        state = eqx.tree_at(
            lambda value: value.iteration_index,
            state,
            checked_iteration,
        )

        def advance(carry, _):
            current, active = carry
            iteration = self.runtime.iterate(current)
            committed = active & iteration.successful
            successor = tree_where(active, iteration.state, current)
            return (successor, committed), (iteration, committed, current.draw_index)

        (successor, successful), (iterations, valid, draw_index) = jax.lax.scan(
            advance,
            (state, jnp.asarray(True)),
            xs=None,
            length=self.capacity,
        )
        retained = valid & iterations.equilibrium_sample
        sample_active = retained[:, None] & jnp.ones(
            (self.capacity, self.runtime.plan.replica_count), dtype=jnp.bool_
        )
        reduced = jnp.where(retained[:, None, None], iterations.reduced_potentials, 0.0)
        coverage = retained[:, None, None] & iterations.coverage
        origin = jnp.where(sample_active, iterations.origin_state, -1)
        assignments = jnp.where(sample_active, iterations.state_at_replica_after, -1)
        chain = jnp.where(
            sample_active,
            self.runtime.plan.chain_indices[None, :],
            -1,
        )
        repeat = jnp.where(
            sample_active,
            jnp.asarray(self.runtime.plan.repeat_index, dtype=jnp.int32),
            jnp.int32(-1),
        )
        dependence = jnp.where(
            sample_active,
            self.runtime.plan.dependence_group_indices[None, :],
            -1,
        )
        draw = jnp.where(sample_active, draw_index, -1)
        count = jnp.sum(valid, dtype=jnp.int64)
        stop = state.iteration_index + count
        successor = AtomisticMultistateState(
            dynamics=successor.dynamics,
            state_at_replica=successor.state_at_replica,
            reduced_potential_cache=successor.reduced_potential_cache,
            reduced_potential_valid=successor.reduced_potential_valid,
            cache_iteration=successor.cache_iteration,
            iteration_index=successor.iteration_index,
            draw_index=successor.draw_index,
            exchange_action_counter=successor.exchange_action_counter,
            exchange_parity=successor.exchange_parity,
            barostat_action_counter=successor.barostat_action_counter,
            sams_action_counter=successor.sams_action_counter,
            sams=successor.sams,
            root_key=successor.root_key,
            segment_index=successor.segment_index + 1,
            continuation_token=_identity_token(self.segment_id),
            plan_id=successor.plan_id,
        )
        return AtomisticMultistateSegmentResult(
            successor_state=successor,
            reduced_potentials=reduced,
            coverage=coverage,
            sample_active=sample_active,
            origin_state=origin,
            state_at_replica=assignments,
            chain_index=chain,
            draw_index=draw,
            repeat_index=repeat,
            dependence_group_index=dependence,
            pair_indices=jnp.where(valid[:, None, None], iterations.pair_indices, -1),
            exchange_attempted=valid[:, None] & iterations.exchange_attempted,
            exchange_accepted=valid[:, None] & iterations.exchange_accepted,
            exchange_log_acceptance=jnp.where(
                valid[:, None], iterations.exchange_log_acceptance, 0.0
            ),
            sams_attempted=valid[:, None] & iterations.sams_attempted,
            sams_changed=valid[:, None] & iterations.sams_changed,
            sams_adapting=valid[:, None]
            & iterations.sams_adapting[:, None]
            & jnp.ones((1, self.runtime.plan.replica_count), dtype=jnp.bool_),
            dynamics_accepted=valid[:, None] & iterations.dynamics_accepted,
            barostat_attempted=valid[:, None] & iterations.barostat_attempted,
            barostat_accepted=valid[:, None] & iterations.barostat_accepted,
            iteration_valid=valid,
            count=count,
            start_watermark=state.iteration_index,
            stop_watermark=stop,
            successful=successful,
            inverse_temperatures=self.runtime.thermodynamic.beta,
            units=self.runtime.dynamics.system.plan.units,
            run_id=self.runtime.plan.run_id,
            measure_id=self.runtime.thermodynamic.phase_space_measure_id,
            state_ids=self.runtime.thermodynamic.state_ids,
            potential_ids=self.runtime.thermodynamic.potential_ids,
            bias_ids=self.runtime.thermodynamic.bias_ids,
            producer_id=self.runtime.prepared_id,
            unit_id="1",
            reduced_convention_id=self.runtime.thermodynamic.reduced_convention_id,
            qualification_id=self.runtime.plan.qualification.qualification_id,
            sampling_exact=self.runtime.plan.qualification.sampling_exact,
            sampling_bias_bound=self.runtime.plan.qualification.sampling_bias_bound,
            segment_id=self.segment_id,
            predecessor_id=self.predecessor_id,
            runtime_id=self.runtime.prepared_id,
        )


class AtomisticMultistateSegmentResult(StrictModule):
    successor_state: AtomisticMultistateState
    reduced_potentials: Array
    coverage: Array
    sample_active: Array
    origin_state: Array
    state_at_replica: Array
    chain_index: Array
    draw_index: Array
    repeat_index: Array
    dependence_group_index: Array
    pair_indices: Array
    exchange_attempted: Array
    exchange_accepted: Array
    exchange_log_acceptance: Array
    sams_attempted: Array
    sams_changed: Array
    sams_adapting: Array
    dynamics_accepted: Array
    barostat_attempted: Array
    barostat_accepted: Array
    iteration_valid: Array
    count: Array
    start_watermark: Array
    stop_watermark: Array
    successful: Array
    inverse_temperatures: Array
    units: AtomisticUnitSystem
    run_id: str = eqx.field(static=True)
    measure_id: str = eqx.field(static=True)
    state_ids: tuple[str, ...] = eqx.field(static=True)
    potential_ids: tuple[str, ...] = eqx.field(static=True)
    bias_ids: tuple[str | None, ...] = eqx.field(static=True)
    producer_id: str = eqx.field(static=True)
    unit_id: str = eqx.field(static=True)
    reduced_convention_id: str = eqx.field(static=True)
    qualification_id: str = eqx.field(static=True)
    sampling_exact: bool = eqx.field(static=True)
    sampling_bias_bound: float = eqx.field(static=True)
    segment_id: str = eqx.field(static=True)
    predecessor_id: str = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)


__all__ = [
    "AtomisticCanonicalSamplingQualification",
    "AtomisticMultistateIteration",
    "AtomisticMultistatePlan",
    "AtomisticMultistateSegmentPlan",
    "AtomisticMultistateSegmentResult",
    "AtomisticMultistateState",
    "AtomisticReducedPotentialEvaluation",
    "AtomisticReplicaExchangePlan",
    "AtomisticSAMSPlan",
    "AtomisticSAMSState",
    "PreparedAtomisticMultistate",
]
