#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Atomic biased-dynamics transactions over the native dynamics runtime.

Biases are terms of the scalar program (and, for force-only ABF, typed
nonconservative force ports); this owner adds no integrator.  One committed
iteration is, in order:

1. propagate one native step with the current Hamiltonian ``H_n``
   (term states fixed);
2. an eligible Monte Carlo cell move, still with ``H_n``;
3. adaptive events at the committed boundary: ABF samples from the explicitly
   requested *physical* force groups, all due hill deposits (heights against
   the pre-event history, one kernel per walker), moving-restraint centers;
4. the event's term states (ABF counts and counters, kernels, centers) are
   committed; if an applied force changed (a term revision advanced), a force
   re-prime at the fixed configuration, otherwise the cache stays valid; the
   energy jump is split into history work and protocol work, both evaluated
   at the committed configuration ``x_{n+1}`` (fixed boundary-state
   convention);
5. publish.

Any failure, including terminal history exhaustion, rejects the whole
iteration: nothing is deposited on rejected, probe, rebase or prime calls.
"""

from __future__ import annotations

from enum import IntFlag
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._array_archive import (
    pack_array_tree,
    read_array_archive,
    unpack_array_tree,
    write_array_archive,
)
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._identity import IdentityWords
from ..._strict import StrictModule
from ..._tree_math import tree_where
from .._barostat import (
    apply_isotropic_monte_carlo_barostat,
    IsotropicMonteCarloBarostatPlan,
)
from .._dynamics import (
    _underlying_program,
    AtomisticDynamicsState,
    PreparedAtomisticDynamics,
)
from .._hybrid import evaluate_force_group
from .._potential_program import AtomisticPotentialProgram
from .._term_state import AtomisticTermStateBindings
from .._thermodynamic import (
    AtomisticThermodynamicStatePlan,
    PreparedThermodynamicStateTable,
)
from .._units import AtomisticUnitSystem
from ._abf import AtomisticABFForcePort, AtomisticABFState
from ._bias import (
    AtomisticBiasTerm,
    AtomisticBiasTermState,
    HillHistoryBiasForm,
    PreparedAtomisticBiasTerm,
    with_bias_controls,
    with_bias_history,
)
from ._bias_history import (
    BiasHistoryOwnership,
    deposit_bias_history,
    FrozenBiasHistory,
    grow_bias_history,
)
from ._cv_geometry import CollectiveVariableGeometry


_CHECKPOINT_FORMAT = "phydrax-biased-atomistic-dynamics-checkpoint"


class BiasedStepReason(IntFlag):
    NONE = 0
    DYNAMICS = 1 << 0
    BAROSTAT = 1 << 1
    PHYSICAL_FORCES = 1 << 2
    HISTORY_EXHAUSTED = 1 << 3
    PRIME = 1 << 4
    COLLECTIVE_VARIABLE = 1 << 5
    RUNTIME_BINDING = 1 << 6


class BiasedDynamicsState(StrictModule):
    """Native dynamics state (term states inside) plus event/work ledgers.

    ``runtime_identity`` is the owning runtime's exact ``prepared_id`` as
    dynamic words (``prepared_id`` decodes it on the host).  A step of a state
    bound to another runtime, or whose history/term states do not belong to
    the runtime's terms, is refused with ``RUNTIME_BINDING``.
    """

    dynamics: AtomisticDynamicsState
    history_work: Array
    protocol_work: Array
    deposition_events: Array
    abf_samples: Array
    last_reasons: Array
    runtime_identity: IdentityWords

    @property
    def prepared_id(self) -> str:
        """Host decoding of the owning biased runtime digest."""
        return self.runtime_identity.fingerprint()


class BiasedStepEvaluation(StrictModule):
    candidate_state: BiasedDynamicsState
    accepted_state: BiasedDynamicsState
    successful: Array
    reasons: Array
    dynamics_reasons: Array
    deposited: Array
    history_work: Array
    protocol_work: Array
    nonconservative_work: Array


def _program_terms(dynamics: PreparedAtomisticDynamics, /) -> tuple[Any, ...]:
    return tuple(_underlying_program(dynamics.potential).terms)


class PreparedBiasedDynamics(StrictModule):
    """Biased single-walker runtime over one native dynamics runtime.

    ``physical_force_groups`` are the force groups of the physical
    Hamiltonian; they must exclude every bias term and, with the bias groups,
    cover the program.  ``walker_index`` is this walker's owner tag for
    walker-owned histories.  ``adaptive=False`` freezes every history and
    control (fixed-Hamiltonian segments for analysis).
    """

    dynamics: PreparedAtomisticDynamics
    thermodynamic: PreparedThermodynamicStateTable
    bias_terms: tuple[PreparedAtomisticBiasTerm, ...]
    abf_ports: tuple[AtomisticABFForcePort, ...]
    barostat: IsotropicMonteCarloBarostatPlan | None
    barostat_interval: int | None = eqx.field(static=True)
    physical_force_groups: tuple[int, ...] = eqx.field(static=True)
    walker_index: Array
    adaptive: bool = eqx.field(static=True)
    prepared_identity: IdentityWords

    def __init__(
        self,
        dynamics: PreparedAtomisticDynamics,
        thermodynamic: PreparedThermodynamicStateTable,
        /,
        *,
        physical_force_groups: tuple[int, ...],
        walker_index: int = 0,
        adaptive: bool = True,
        barostat: IsotropicMonteCarloBarostatPlan | None = None,
        barostat_interval: int | None = None,
    ) -> None:
        if not isinstance(dynamics, PreparedAtomisticDynamics):
            raise TypeError("dynamics must be PreparedAtomisticDynamics.")
        thermodynamic.validate_dynamics(dynamics)
        terms = _program_terms(dynamics)
        bias_terms = tuple(
            term for term in terms if isinstance(term, PreparedAtomisticBiasTerm)
        )
        abf_ports = tuple(
            port
            for port in dynamics.force_ports
            if isinstance(port, AtomisticABFForcePort)
        )
        if len(abf_ports) != len(dynamics.force_ports):
            raise ValueError("Only ABF force ports have a biased-dynamics event owner.")
        if not bias_terms and not abf_ports:
            raise ValueError("The runtime carries no bias term or ABF port.")
        groups = tuple(sorted({int(group) for group in physical_force_groups}))
        bias_groups = {term.force_group for term in bias_terms}
        all_groups = {term.force_group for term in terms}
        if set(groups) & bias_groups or set(groups) | bias_groups != all_groups:
            raise ValueError(
                "Physical force groups must exclude every bias term and together "
                "with the bias groups cover the whole program."
            )
        if abf_ports and not groups:
            raise ValueError("ABF sampling requires explicit physical force groups.")
        temperatures = np.asarray(thermodynamic.temperature)[
            np.asarray(thermodynamic.temperature_mask)
        ]
        for port in abf_ports:
            if temperatures.size == 0 or not np.allclose(
                temperatures, port.abf.plan.temperature, rtol=1.0e-12, atol=0.0
            ):
                raise ValueError(
                    "ABF temperature must equal every thermostat temperature of the table."
                )
        if (barostat is None) != (barostat_interval is None):
            raise ValueError("A barostat and its interval are declared together.")
        if barostat_interval is not None and int(barostat_interval) < 1:
            raise ValueError("barostat_interval must be positive.")
        self.dynamics = dynamics
        self.thermodynamic = thermodynamic
        self.bias_terms = bias_terms
        self.abf_ports = abf_ports
        self.barostat = barostat
        self.barostat_interval = (
            None if barostat_interval is None else int(barostat_interval)
        )
        self.physical_force_groups = groups
        walker = int(walker_index)
        self.walker_index = jnp.asarray(walker, dtype=jnp.int32)
        self.adaptive = bool(adaptive)
        self.prepared_identity = IdentityWords.of(
            {
                "kind": "prepared-biased-dynamics",
                "dynamics": dynamics.prepared_id,
                "thermodynamic": thermodynamic.table_id,
                "physical_groups": list(groups),
                "walker": walker,
                "adaptive": self.adaptive,
                "barostat": None if barostat is None else barostat.plan_id,
                "barostat_interval": self.barostat_interval,
            }
        )

    @property
    def prepared_id(self) -> str:
        """Host decoding of the canonical biased-runtime digest."""
        return self.prepared_identity.fingerprint()

    # ----------------------------------------------------------------- binding

    def owner_key(self, term: PreparedAtomisticBiasTerm, label: Array, /) -> Array:
        history = term.plan.form.history
        if history is None or history.ownership is BiasHistoryOwnership.SHARED:
            return jnp.zeros((), dtype=jnp.int32)
        if history.ownership is BiasHistoryOwnership.WALKER:
            return self.walker_index
        return jnp.asarray(label, dtype=jnp.int32)

    def initialize(self, state: AtomisticDynamicsState, /) -> BiasedDynamicsState:
        """Bind owner keys (walker or thermodynamic label) and the event ledgers."""
        if state.prepared_dynamics_id != self.dynamics.prepared_id:
            raise ValueError("The state belongs to another dynamics runtime.")
        bindings = dict(
            zip(state.term_states.slots, state.term_states.values, strict=True)
        )
        changed = False
        for term in self.bias_terms:
            if term.plan.stateful:
                current = term.require_state(bindings[term.name])
                owner = self.owner_key(term, state.thermodynamic_state_index)
                bindings[term.name] = with_bias_controls(
                    current, current.controls, owner_key=owner
                )
                changed = True
        dynamics_state = state
        if changed:
            dynamics_state = eqx.tree_at(
                lambda value: value.term_states,
                state,
                AtomisticTermStateBindings(bindings),
            )
            primed = self.dynamics.prime_state(dynamics_state, self.thermodynamic)
            if not bool(np.asarray(primed.successful)):
                raise ValueError("The biased initial state could not be primed.")
            dynamics_state = primed.accepted_state
        dtype = state.kinematics.positions.dtype
        zero = jnp.zeros((), dtype=dtype)
        return BiasedDynamicsState(
            dynamics=dynamics_state,
            history_work=zero,
            protocol_work=zero,
            deposition_events=jnp.zeros((), dtype=jnp.int32),
            abf_samples=jnp.zeros((), dtype=jnp.int32),
            last_reasons=jnp.zeros((), dtype=jnp.int32),
            runtime_identity=self.prepared_identity,
        )

    # -------------------------------------------------------------- geometry

    def _kwargs(self, state: AtomisticDynamicsState, /) -> dict[str, Any]:
        row = self.thermodynamic.state_at_replica(state.thermodynamic_state_index)
        return self.dynamics._potential_kwargs(
            self.dynamics._unwrapped(state.kinematics, state.cell_vectors),
            state.kinematics.image_counts,
            state.species,
            state.cell_vectors,
            row.controls,
            state.term_states,
        )

    def geometry(
        self, state: AtomisticDynamicsState, term: PreparedAtomisticBiasTerm, /
    ) -> CollectiveVariableGeometry:
        """The term's program geometry at an accepted state (no strain)."""
        context = self.dynamics.potential.context(
            state.kinematics.positions,
            state.neighborhoods,
            term=term,
            **self._kwargs(state),
        )
        return CollectiveVariableGeometry.from_context(context)

    def _port_geometry(
        self, state: AtomisticDynamicsState, /
    ) -> CollectiveVariableGeometry:
        system = self.dynamics.system
        return CollectiveVariableGeometry(
            state.kinematics.positions,
            unwrapped_positions=self.dynamics._unwrapped(
                state.kinematics, state.cell_vectors
            ),
            cell=system.cell,
            cell_vectors=state.cell_vectors if system.cell is not None else None,
        )

    def physical_forces(self, state: AtomisticDynamicsState, /) -> tuple[Array, Array]:
        """Forces of the declared physical force groups only (no bias)."""
        kwargs = self._kwargs(state)
        forces = jnp.zeros_like(state.kinematics.positions)
        successful = jnp.asarray(True)
        for group in self.physical_force_groups:
            evaluation = evaluate_force_group(
                self.dynamics.potential,
                group,
                state.kinematics.positions,
                state.neighborhoods,
                **kwargs,
            )
            forces = forces + evaluation.forces
            successful = successful & evaluation.successful
        return forces, successful

    def _bias_energy(
        self,
        state: AtomisticDynamicsState,
        terms: tuple[PreparedAtomisticBiasTerm, ...],
        bindings: AtomisticTermStateBindings,
        /,
    ) -> Array:
        energy = jnp.zeros((), dtype=state.kinematics.positions.dtype)
        for term in terms:
            term_state = bindings[term.name] if term.plan.stateful else None
            value, _, _ = term.bias(self.geometry(state, term), term_state)
            energy = energy + value
        return energy

    # ------------------------------------------------------------------ events

    def _events(
        self, state: AtomisticDynamicsState, /
    ) -> tuple[AtomisticTermStateBindings, Array, Array, Array, Array, Array, Array]:
        """Committed-boundary events.

        Returns ``(bindings, deposited, exhausted, ok, samples, changed, bound)``;
        ``bound`` is false when a bound history belongs to another plan.
        """
        bindings = dict(
            zip(state.term_states.slots, state.term_states.values, strict=True)
        )
        deposited = []
        exhausted = jnp.asarray(False)
        successful = jnp.asarray(True)
        bound = jnp.asarray(True)
        samples = jnp.zeros((), dtype=jnp.int32)
        changed = jnp.asarray(False)
        if self.abf_ports:
            forces, force_ok = self.physical_forces(state)
            successful = successful & force_ok
            geometry = self._port_geometry(state)
            for port in self.abf_ports:
                current = bindings[port.name]
                if not isinstance(current, AtomisticABFState):
                    raise TypeError("ABF port state has the wrong type.")
                bound = bound & current.owner_identity.equals(port.owner_identity)
                sample = port.abf.sample(geometry, forces)
                updated = port.abf.accumulate(current, sample)
                bindings[port.name] = updated
                samples = samples + (sample.in_range & sample.regular).astype(jnp.int32)
                changed = changed | (updated.revision != current.revision)
        step = state.step_index
        for term in self.bias_terms:
            if not term.plan.stateful:
                continue
            current = term.require_state(bindings[term.name])
            bound = bound & current.owner_identity.equals(term.term_identity)
            replacement = current
            history_plan = term.plan.form.history
            if (
                isinstance(term.plan.form, HillHistoryBiasForm)
                and history_plan is not None
            ):
                history = current.history
                if history is None:
                    raise ValueError("A hill term lost its history state.")
                evaluation = term.plan.variables.evaluate(self.geometry(state, term))
                due = history_plan.due(step) & evaluation.successful
                successful = successful & (
                    ~history_plan.due(step) | evaluation.successful
                )
                event = deposit_bias_history(
                    history_plan,
                    history,
                    evaluation.values[None, :],
                    self.owner_key(term, state.thermodynamic_state_index)[None],
                    due[None],
                    term.plan.variables.metrics,
                    boltzmann_constant=float(
                        self.dynamics.system.plan.units.boltzmann_constant
                    ),
                )
                bound = bound & event.successful
                exhausted = exhausted | (due & event.successful & ~event.inserted)
                deposited.append(event.inserted)
                replacement = tree_where(
                    event.inserted, with_bias_history(current, event.state), current
                )
                changed = changed | event.inserted
            if term.plan.center_rate is not None:
                center = term.plan.form.initial_controls().astype(state.time.dtype)
                controls = (
                    center + term.plan.center_rate.astype(state.time.dtype) * state.time
                )
                replacement = with_bias_controls(replacement, controls)
                changed = jnp.asarray(True)
            bindings[term.name] = replacement
        return (
            AtomisticTermStateBindings(bindings),
            jnp.stack(deposited) if deposited else jnp.zeros((0,), dtype=jnp.bool_),
            exhausted,
            successful,
            samples,
            changed,
            bound,
        )

    def step(self, state: BiasedDynamicsState, /) -> BiasedStepEvaluation:
        """One atomic biased iteration (see the module order).

        Terminal history exhaustion is latched in ``last_reasons``: a state
        carrying it refuses every further step (no dynamics, deposit, RNG or
        counter is committed) until :func:`grow_bias_history_capacity` rebinds
        a larger history, instead of re-attempting the failed event.
        """
        runtime_ok = state.runtime_identity.equals(self.prepared_identity)
        dynamics, table = self.dynamics, self.thermodynamic
        terminal = (state.last_reasons & int(BiasedStepReason.HISTORY_EXHAUSTED)) != 0
        evaluation = dynamics.step_detailed(state.dynamics, table)
        reasons = jnp.where(evaluation.successful, 0, int(BiasedStepReason.DYNAMICS))
        reasons = reasons | jnp.where(
            terminal, int(BiasedStepReason.HISTORY_EXHAUSTED), 0
        )
        reasons = reasons | jnp.where(
            runtime_ok, 0, int(BiasedStepReason.RUNTIME_BINDING)
        )
        current = evaluation.accepted_state
        if self.barostat is not None and self.barostat_interval is not None:
            due = jnp.mod(current.step_index, self.barostat_interval) == 0
            move = apply_isotropic_monte_carlo_barostat(
                dynamics, current, table, self.barostat
            )
            current = tree_where(due, move.accepted_state, current)
            reasons = reasons | jnp.where(
                due & ~move.successful, int(BiasedStepReason.BAROSTAT), 0
            )
        zero = jnp.zeros((), dtype=current.kinematics.positions.dtype)
        history_work = zero
        protocol_work = zero
        deposited = jnp.zeros((0,), dtype=jnp.bool_)
        samples = jnp.zeros((), dtype=jnp.int32)
        if self.adaptive:
            (
                bindings,
                deposited,
                exhausted,
                events_ok,
                samples,
                changed,
                bound,
            ) = self._events(current)
            reasons = (
                reasons
                | jnp.where(exhausted, int(BiasedStepReason.HISTORY_EXHAUSTED), 0)
                | jnp.where(events_ok, 0, int(BiasedStepReason.PHYSICAL_FORCES))
                | jnp.where(bound, 0, int(BiasedStepReason.RUNTIME_BINDING))
            )
            history_terms = tuple(
                term
                for term in self.bias_terms
                if isinstance(term.plan.form, HillHistoryBiasForm)
            )
            moving_terms = tuple(
                term for term in self.bias_terms if term.plan.center_rate is not None
            )
            history_work = self._bias_energy(
                current, history_terms, bindings
            ) - self._bias_energy(current, history_terms, current.term_states)
            protocol_work = self._bias_energy(
                current, moving_terms, bindings
            ) - self._bias_energy(current, moving_terms, current.term_states)
            # Accumulated statistics and counters (ABF counts below the ramp,
            # out-of-range/irregular tallies) are committed whenever the event
            # succeeds; only a changed applied force field (a term revision
            # advance) re-primes, otherwise the cached forces stay valid.
            rebound = eqx.tree_at(lambda value: value.term_states, current, bindings)
            primed = dynamics.prime_state(rebound, table)
            current = tree_where(changed, primed.accepted_state, rebound)
            reasons = reasons | jnp.where(
                changed & ~primed.successful, int(BiasedStepReason.PRIME), 0
            )
        reasons = jnp.asarray(reasons, dtype=jnp.int32)
        successful = reasons == 0
        candidate = BiasedDynamicsState(
            dynamics=current,
            history_work=state.history_work + history_work,
            protocol_work=state.protocol_work + protocol_work,
            deposition_events=state.deposition_events
            + jnp.sum(deposited, dtype=jnp.int32),
            abf_samples=state.abf_samples + samples,
            last_reasons=reasons,
            runtime_identity=self.prepared_identity,
        )
        rejected = eqx.tree_at(lambda value: value.last_reasons, state, reasons)
        return BiasedStepEvaluation(
            candidate_state=candidate,
            accepted_state=tree_where(successful, candidate, rejected),
            successful=successful,
            reasons=reasons,
            dynamics_reasons=evaluation.rejection_reasons,
            deposited=deposited,
            history_work=history_work,
            protocol_work=protocol_work,
            nonconservative_work=evaluation.nonconservative_work,
        )

    # -------------------------------------------------------------- lifecycle

    def freeze(self, state: BiasedDynamicsState, /) -> dict[str, FrozenBiasHistory]:
        """Authenticated frozen copies of every history (host transaction)."""
        if state.prepared_id != self.prepared_id:
            raise ValueError("The biased state belongs to another runtime.")
        frozen: dict[str, FrozenBiasHistory] = {}
        for term in self.bias_terms:
            plan = term.plan.form.history
            if plan is None:
                continue
            term_state = term.require_state(state.dynamics.term_states[term.name])
            if term_state.owner_identity.fingerprint() != term.term_id:
                raise ValueError(f"Bias term {term.name!r} binds another term's state.")
            if term_state.history is None:
                raise ValueError("A history term lost its history state.")
            frozen[term.name] = FrozenBiasHistory(plan, term_state.history)
        return frozen


def grow_bias_history_capacity(
    runtime: PreparedBiasedDynamics,
    state: BiasedDynamicsState,
    slot: str,
    capacity: int,
    states: tuple[AtomisticThermodynamicStatePlan, ...],
    /,
) -> tuple[PreparedBiasedDynamics, BiasedDynamicsState]:
    """Host capacity-ladder transaction for a terminally exhausted history.

    ``slot`` is the history bias term's declared ``name``.  The bias term
    keeps its scientific identity (capacity is a resource), the Hamiltonian
    is rebound, kernels/owners/counters are copied into the grown buffer and
    the state is re-primed without consuming any event.
    """

    if state.prepared_id != runtime.prepared_id:
        raise ValueError("The biased state belongs to another runtime.")
    dynamics = runtime.dynamics
    program = _underlying_program(dynamics.potential)
    if program is not dynamics.potential:
        raise ValueError("Capacity growth rebinds a plain potential program.")
    plans = []
    grown_state = None
    for term in program.plan.terms:
        if isinstance(term, AtomisticBiasTerm) and term.name == slot:
            history = term.form.history
            if history is None:
                raise ValueError("The selected bias term has no history.")
            current = state.dynamics.term_states[slot]
            if not isinstance(current, AtomisticBiasTermState) or current.history is None:
                raise TypeError("The selected term state carries no history.")
            if current.owner_identity.fingerprint() != term.term_id:
                raise ValueError("The selected term state belongs to another term.")
            grown_plan, grown_history = grow_bias_history(
                history, current.history, capacity
            )
            plans.append(
                AtomisticBiasTerm(
                    term.variables,
                    HillHistoryBiasForm(grown_plan),
                    name=term.name,
                    force_group=term.force_group,
                    center_rate=term.center_rate,
                )
            )
            grown_state = AtomisticBiasTermState(
                controls=current.controls,
                history=grown_history,
                owner_key=current.owner_key,
                coupling=current.coupling,
                revision=current.revision + 1,
                successful=current.successful,
                owner_identity=current.owner_identity,
            )
        else:
            plans.append(term)
    if grown_state is None:
        raise ValueError(f"No bias term in slot {slot!r}.")
    potential = AtomisticPotentialProgram(
        tuple(plans), coefficients=program.plan.coefficients
    ).prepare(
        dynamics.system,
        neighborhood_bundle=program.neighborhood_bundle,
        graph_execution=program.graph_execution,
    )
    rebound_dynamics, table, rebound = dynamics.rebind_hamiltonian(
        potential, state.dynamics, runtime.thermodynamic, states
    )
    bindings = dict(
        zip(rebound.term_states.slots, rebound.term_states.values, strict=True)
    )
    bindings[slot] = grown_state
    rebound = eqx.tree_at(
        lambda value: value.term_states, rebound, AtomisticTermStateBindings(bindings)
    )
    primed = rebound_dynamics.prime_state(rebound, table)
    if not bool(np.asarray(primed.successful)):
        raise RuntimeError("The grown history state could not be re-primed.")
    grown_runtime = PreparedBiasedDynamics(
        rebound_dynamics,
        table,
        physical_force_groups=runtime.physical_force_groups,
        walker_index=int(np.asarray(runtime.walker_index)),
        adaptive=runtime.adaptive,
        barostat=runtime.barostat,
        barostat_interval=runtime.barostat_interval,
    )
    return grown_runtime, BiasedDynamicsState(
        dynamics=primed.accepted_state,
        history_work=state.history_work,
        protocol_work=state.protocol_work,
        deposition_events=state.deposition_events,
        abf_samples=state.abf_samples,
        last_reasons=jnp.zeros((), dtype=jnp.int32),
        runtime_identity=grown_runtime.prepared_identity,
    )


class BiasedDynamicsCheckpoint(StrictModule):
    state: BiasedDynamicsState
    units: AtomisticUnitSystem
    payload_id: str = eqx.field(static=True)


def _identities(runtime: PreparedBiasedDynamics, /) -> dict[str, Any]:
    histories = {
        term.term_id: term.plan.form.history.history_id
        for term in runtime.bias_terms
        if term.plan.form.history is not None
    }
    return {
        "format": _CHECKPOINT_FORMAT,
        "prepared_id": runtime.prepared_id,
        "dynamics_id": runtime.dynamics.prepared_id,
        "thermodynamic_states_id": runtime.thermodynamic.table_id,
        "term_state_slots": list(runtime.dynamics.term_state_slots),
        "histories": histories,
        "abf": [port.abf.prepared_id for port in runtime.abf_ports],
    }


def write_biased_dynamics_checkpoint(
    path: str | Path,
    runtime: PreparedBiasedDynamics,
    state: BiasedDynamicsState,
    /,
) -> BiasedDynamicsCheckpoint:
    """Exact continuation: dynamics, term states (histories/ABF/controls), ledgers."""
    if state.prepared_id != runtime.prepared_id:
        raise ValueError("The biased state belongs to another runtime.")
    arrays: dict[str, object] = {}
    specification = pack_array_tree("runtime", state, arrays)
    units = runtime.dynamics.system.plan.units
    identities = _identities(runtime)
    payload_id = canonical_fingerprint(
        {
            "identities": identities,
            "units": units.unit_system_id,
            "state": specification,
            "arrays": array_tree_fingerprint(arrays),
        }
    )
    write_array_archive(
        path,
        manifest={
            **identities,
            "unit_system": units.to_dict(),
            "state": specification,
            "payload_id": payload_id,
        },
        arrays=arrays,
    )
    return BiasedDynamicsCheckpoint(state, units, payload_id)


def read_biased_dynamics_checkpoint(
    path: str | Path,
    runtime: PreparedBiasedDynamics,
    template: BiasedDynamicsState,
    /,
) -> BiasedDynamicsCheckpoint:
    """Authenticated fresh-process restore against a runtime-built template."""
    if template.prepared_id != runtime.prepared_id:
        raise ValueError("The template belongs to another runtime.")
    manifest, arrays = read_array_archive(path)
    identities = _identities(runtime)
    expected = set(identities) | {"unit_system", "state", "payload_id", "arrays"}
    if set(manifest) != expected:
        raise ValueError("Biased checkpoint manifest is not the canonical format.")
    if any(manifest[name] != value for name, value in identities.items()):
        raise ValueError("Biased checkpoint identity does not match the runtime.")
    units = AtomisticUnitSystem.from_dict(manifest["unit_system"])
    if units.unit_system_id != runtime.dynamics.system.plan.units.unit_system_id:
        raise ValueError("Biased checkpoint unit system is incompatible.")
    state = unpack_array_tree(manifest["state"], arrays, template)
    if not isinstance(state, BiasedDynamicsState):
        raise TypeError("Checkpoint did not reconstruct BiasedDynamicsState.")
    payload_id = canonical_fingerprint(
        {
            "identities": identities,
            "units": units.unit_system_id,
            "state": manifest["state"],
            "arrays": array_tree_fingerprint(arrays),
        }
    )
    if manifest["payload_id"] != payload_id:
        raise ValueError("Biased checkpoint payload identity is corrupt.")
    return BiasedDynamicsCheckpoint(state, units, payload_id)


__all__ = [
    "BiasedDynamicsCheckpoint",
    "BiasedDynamicsState",
    "BiasedStepEvaluation",
    "BiasedStepReason",
    "PreparedBiasedDynamics",
    "grow_bias_history_capacity",
    "read_biased_dynamics_checkpoint",
    "write_biased_dynamics_checkpoint",
]
