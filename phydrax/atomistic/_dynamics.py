#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Transactional atomistic dynamics over one accepted neighborhood bundle.

One ``PreparedAtomisticDynamics`` executes exactly one declared method:
velocity Verlet (NVE), BAOAB Langevin, Nose-Hoover chains, stochastic or
deterministic velocity rescaling, discrete generalized Langevin (NVT, or NPT
with a scheduled Monte Carlo cell move) and integrator-owned MTK NPT/NPH.
Each step refreshes a pure candidate of the accepted
``AtomisticNeighborhoodBundleState`` before every force evaluation, evaluates
energy, forces and (when the method or the periodic measure requires it) the
homogeneous-strain derivative of the same scalar Hamiltonian, and commits
kinematics, cell, bundle, force cache, constraints, typed controllers,
counters and the energy ledger atomically only when every contract passes.

The force cache binds the program/bundle identities (dynamic words), the
configuration revision (advanced by every committed coordinate or cell
change), the Hamiltonian controls, every source epoch and role graph
revision, the revisions of every read-only term state and the realization
key; step, rebase, Monte Carlo moves and priming share the one validity rule
``force_cache_valid``.  Typed nonconservative force ports are evaluated at
every conservative force boundary with the same geometry and enter the
integrator's own kicks before constraint projection; their trajectory work is
booked explicitly and they refuse every energy-based cell or state move.
Capacity-only failures are repaired by growing declared bundle capacities,
rebinding and re-priming the accepted state and replaying the identical
physical/stochastic attempt.

Every runtime, table and state PyTree is content-free: identities are dynamic
``IdentityWords`` compared in compiled code (a foreign or stale binding is a
refused step with ``BINDING``/``STALE_FORCE``/``TERM_STATE`` reasons), static
checks cover only the layout, and host transactions (rebind, transition,
checkpoint) keep exact string refusal.  A parameter or material revision of
one runtime ``layout_id`` therefore reuses the same executable.
"""

from __future__ import annotations

from enum import IntEnum, IntFlag
from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._identity import IdentityWords
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._tree_math import tree_allfinite, tree_where
from ..discretization import SegmentedImageVerletState
from ..typing import checked, PRNGKey
from ._cell_dynamics import (
    AtomisticCellControllerState,
    barostat_force,
    barostat_mass,
    bind_atomistic_bundle_geometry,
    cell_admissible,
    cell_kinetic_energy,
    cell_velocity_gradient,
    cell_volume,
    initial_cell_controller,
    kinetic_tensor,
    mtk_drift,
    mtk_momentum_kick,
    MTKNPHPlan,
    MTKNPTPlan,
    project_cell_force,
)
from ._constraints import PreparedDistanceConstraints
from ._ensemble_advanced import AtomisticSplittingPlan, SplittingOperatorKind
from ._force_port import (
    AbstractAtomisticForcePort,
    AtomisticForcePortEvaluation,
    AtomisticForcePortGeometry,
)
from ._neighborhood_bundle import (
    AtomisticNeighborhoodBundleState,
    AtomisticNeighborhoodView,
    PreparedAtomisticNeighborhoodBundle,
)
from ._potential_program import (
    AbstractPreparedAtomisticHamiltonian,
    AtomisticHamiltonianEvaluation,
)
from ._sites import AtomisticInteractionSiteState, AtomisticSiteDomain
from ._stochastic_runtime import (
    apply_generalized_langevin,
    GeneralizedLangevinMemoryState,
    GeneralizedLangevinPlan,
    initial_generalized_langevin_memory,
)
from ._system import PreparedAtomisticSystem
from ._term_state import AbstractAtomisticTermState, AtomisticTermStateBindings
from ._thermal import (
    apply_baoab_ornstein_uhlenbeck,
    BAOABLangevinPlan,
    CSVR_FIRST_OPERATOR_ID,
    CSVR_SECOND_OPERATOR_ID,
    deterministic_velocity_rescaling_scale,
    initial_nose_hoover_chain_state,
    nose_hoover_chain_energy,
    NoseHooverChainParameters,
    NoseHooverChainPlan,
    NoseHooverChainState,
    propagate_nose_hoover_chain,
    stochastic_velocity_rescaling_scale,
    StochasticVelocityRescalingPlan,
    VelocityRescalingPlan,
)
from ._thermodynamic import (
    AtomisticThermodynamicStatePlan,
    PreparedThermodynamicStateTable,
)


class AtomisticDynamicsStatus(IntEnum):
    SUCCESS = 0
    INVALID_INITIAL_STATE = 1
    REJECTED = 2
    NONFINITE = 3


class AtomisticStepRejectionReason(IntFlag):
    """Separated causes of a rejected transition.

    ``SOURCE_CAPACITY`` (image search storage) and ``ROUTE_CAPACITY`` (role
    route/degree storage) are the only causes a capacity replay may repair.
    ``DOMAIN`` covers source scientific, representation, site realization,
    stale or uninitialized neighborhood failures; ``RELATION`` open reversal
    closures and endpoint identity conflicts; ``IMAGE_AMBIGUITY`` a repeated
    minimum image or self image inside a classical cutoff.  ``FORCE_PORT``
    marks a failed nonconservative force port and ``TERM_STATE`` an
    unsuccessful, missing or foreign read-only term state.  ``BINDING``
    marks a state or thermodynamic table bound to another prepared runtime
    identity (same layout, other material or parameter revision).
    """

    NONE = 0
    SOURCE_CAPACITY = 1 << 0
    ROUTE_CAPACITY = 1 << 1
    DOMAIN = 1 << 2
    RELATION = 1 << 3
    IMAGE_AMBIGUITY = 1 << 4
    POTENTIAL = 1 << 5
    CONSTRAINT = 1 << 6
    NONFINITE = 1 << 7
    STALE_FORCE = 1 << 8
    UNIT_SYSTEM = 1 << 9
    THERMOSTAT = 1 << 10
    THERMODYNAMIC_STATE = 1 << 11
    CELL = 1 << 12
    BAROSTAT = 1 << 13
    STRAIN_UNAVAILABLE = 1 << 14
    FORCE_PORT = 1 << 15
    TERM_STATE = 1 << 16
    BINDING = 1 << 17


CAPACITY_REJECTIONS = int(
    AtomisticStepRejectionReason.SOURCE_CAPACITY
    | AtomisticStepRejectionReason.ROUTE_CAPACITY
)


class AtomisticKinematics(StrictModule):
    positions: Array
    momenta: Array
    image_counts: Array


class AtomisticForceState(StrictModule):
    """Cached scalar-Hamiltonian response bound to its complete evaluation key.

    ``strain_derivative`` is ``dE/d strain`` (row-cell convention, deformation
    ``F = I + strain`` at fixed fractional coordinates and integer images) and
    ``stress`` the tension-positive Cauchy stress ``sym(dE/d strain) / V``;
    both are NaN unless ``strain_available``.  The finite position moment is
    not a periodic virial and is not cached.  ``forces`` are the conservative
    program forces; ``nonconservative_forces`` the summed force-port forces of
    the same geometry (zeros without ports).  ``term_revisions`` are the
    revisions of the term states the evaluation consumed, in the runtime's
    ``term_state_slots`` order.  ``program_identity``/``bundle_identity`` are
    the dynamic words of the producing prepared Hamiltonian and neighborhood
    bundle (host ``program_id``/``bundle_id``).
    """

    forces: Array
    nonconservative_forces: Array
    potential_energy: Array
    term_energies: Array
    atom_energy: Array
    strain_derivative: Array
    stress: Array
    strain_available: Array
    configuration_revision: Array
    hamiltonian_index: Array
    controls: Array
    source_epochs: Array
    graph_epochs: Array
    random_key: Array
    term_revisions: Array
    successful: Array
    program_identity: IdentityWords
    bundle_identity: IdentityWords

    @property
    def program_id(self) -> str:
        return self.program_identity.fingerprint()

    @property
    def bundle_id(self) -> str:
        return self.bundle_identity.fingerprint()


class AtomisticEnergyLedgerState(StrictModule):
    """Energy ledger of the accepted trajectory.

    ``extended_energy`` is ``K + U + K_cell + P_ext V + E_chains``: the
    conserved quantity of the deterministic extended methods (``K + U`` for
    plain NVE).  ``reservoir_heat`` is energy exchanged with stochastic or
    noncanonical reservoirs (BAOAB, CSVR, GLE, deterministic rescaling, Monte
    Carlo acceptance); ``thermostat_heat`` is all kinetic energy changed by
    any thermostat including deterministic chains.  ``barostat_work`` is the
    change of ``K_cell + P_ext V`` (MTK) or ``P_ext dV`` of accepted Monte Carlo
    moves.  ``cumulative_balance_residual`` accumulates
    ``d(extended) - reservoir_heat - constraint_work - external_work`` per
    accepted step, where a step's ``external_work`` is the trajectory work of
    its nonconservative force ports (rebinding events book their jumps as
    external work without balancing).
    """

    initial_kinetic_energy: Array
    initial_potential_energy: Array
    initial_extended_energy: Array
    kinetic_energy: Array
    potential_energy: Array
    total_energy: Array
    cell_kinetic_energy: Array
    pressure_volume: Array
    thermostat_energy: Array
    extended_energy: Array
    thermostat_heat: Array
    reservoir_heat: Array
    barostat_work: Array
    external_work: Array
    constraint_work: Array
    cumulative_balance_residual: Array
    last_relative_energy_change: Array
    accepted_steps: Array


class AtomisticControllerState(StrictModule):
    """Typed continuation of every thermodynamic controller of one method.

    Absent controllers are ``None``; their presence is fixed by the method.
    ``move_counter`` addresses scheduled Monte Carlo cell moves and advances
    once per completed Metropolis decision, whether accepted or rejected;
    ``cell_epoch`` advances on every committed cell change.
    """

    particle_chain: NoseHooverChainState | None
    cell_chain: NoseHooverChainState | None
    gle: GeneralizedLangevinMemoryState | None
    cell: AtomisticCellControllerState | None
    move_counter: Array
    accepted_moves: Array
    cell_epoch: Array
    rescale_events: Array


class AtomisticDynamicsState(StrictModule):
    """Accepted dynamics state; its runtime and table bindings are dynamic words.

    ``prepared_dynamics_identity``/``thermodynamic_table_identity`` (host
    ``prepared_dynamics_id``/``thermodynamic_table_id``) are compared in
    compiled code, so states of every revision of one layout share a treedef.
    """

    time: Array
    step_index: Array
    configuration_revision: Array
    kinematics: AtomisticKinematics
    species: Array
    cell_vectors: Array
    neighborhoods: AtomisticNeighborhoodBundleState
    force: AtomisticForceState
    constraint_lagrange: Array
    constraint_position_residual: Array
    constraint_velocity_residual: Array
    controllers: AtomisticControllerState
    random_key: Array
    energy: AtomisticEnergyLedgerState
    last_status: Array
    last_rejection_reasons: Array
    thermodynamic_state_index: Array
    term_states: AtomisticTermStateBindings
    thermodynamic_table_identity: IdentityWords
    prepared_dynamics_identity: IdentityWords

    @property
    def thermodynamic_table_id(self) -> str:
        return self.thermodynamic_table_identity.fingerprint()

    @property
    def prepared_dynamics_id(self) -> str:
        return self.prepared_dynamics_identity.fingerprint()


class AtomisticDynamicsDiagnostics(StrictModule):
    kinetic_energy: Array
    potential_energy: Array
    total_energy: Array
    extended_energy: Array
    temperature: Array
    pressure: Array
    pressure_tensor: Array
    volume: Array
    total_linear_momentum: Array
    total_angular_momentum: Array
    net_internal_force: Array
    net_internal_torque: Array
    minimum_pair_distance: Array
    cutoff_margin: Array
    neighborhood_rebuilt: Array
    neighborhood_rebuild_count: Array
    neighborhood_certificate_margin: Array
    constraint_residual: Array
    energy: AtomisticEnergyLedgerState
    successful: Array
    rejection_reasons: Array


class AtomisticStepEvaluation(StrictModule):
    candidate_state: AtomisticDynamicsState
    accepted_state: AtomisticDynamicsState
    potential: AtomisticHamiltonianEvaluation
    diagnostics: AtomisticDynamicsDiagnostics
    successful: Array
    residual: Array
    work: Array
    nonconservative_work: Array
    thermodynamic_state_index: Array
    rejection_reasons: Array


class AtomisticThermodynamicRebaseEvaluation(StrictModule):
    candidate_state: AtomisticDynamicsState
    accepted_state: AtomisticDynamicsState
    successful: Array


class AtomisticInitialStateEvaluation(StrictModule):
    """Initial state with its explicit admission status (never a substitute state).

    On failure ``state.last_status`` is ``INVALID_INITIAL_STATE``,
    ``state.last_rejection_reasons`` equals ``rejection_reasons`` and the
    force cache is unsuccessful, so no transition can advance it.
    """

    state: AtomisticDynamicsState
    successful: Array
    rejection_reasons: Array


class AtomisticPrimeEvaluation(StrictModule):
    """Force/strain priming of an accepted state without consuming any event."""

    candidate_state: AtomisticDynamicsState
    accepted_state: AtomisticDynamicsState
    potential: AtomisticHamiltonianEvaluation
    successful: Array
    rejection_reasons: Array


class AtomisticDynamicsHamiltonianTransition(StrictModule):
    """Result of ``transition_hamiltonian``: runtime, table, state and lineage.

    ``energy_jump`` is the external work booked for the scalar change at the
    unchanged accepted state; on failure the source runtime and state return.
    Source/target runtime and lineage digests are dynamic words (host
    ``source_dynamics_id``, ``target_dynamics_id``, ``lineage_id``).
    """

    dynamics: Any
    thermodynamic: PreparedThermodynamicStateTable
    state: AtomisticDynamicsState
    successful: Array
    energy_jump: Array
    rejection_reasons: Array
    source_dynamics_identity: IdentityWords
    target_dynamics_identity: IdentityWords
    lineage_identity: IdentityWords

    @property
    def source_dynamics_id(self) -> str:
        return self.source_dynamics_identity.fingerprint()

    @property
    def target_dynamics_id(self) -> str:
        return self.target_dynamics_identity.fingerprint()

    @property
    def lineage_id(self) -> str:
        return self.lineage_identity.fingerprint()


class AtomisticPotentialEnergyEvaluation(StrictModule):
    energy: Array
    successful: Array


class VelocityVerletPlan(StrictModule, NonTrainableState):
    step_size: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, step_size: float, /) -> None:
        step = float(step_size)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("step_size must be finite and positive.")
        self.step_size = step
        self.plan_id = canonical_fingerprint(
            {"kind": "atomistic-velocity-verlet", "step_size": step}
        )

    @property
    def canonical(self) -> bool:
        return False

    @property
    def stochastic(self) -> bool:
        return False


AtomisticIntegratorPlan: TypeAlias = (
    VelocityVerletPlan
    | BAOABLangevinPlan
    | NoseHooverChainPlan
    | StochasticVelocityRescalingPlan
    | VelocityRescalingPlan
    | GeneralizedLangevinPlan
    | MTKNPTPlan
    | MTKNPHPlan
)
_INTEGRATORS = (
    VelocityVerletPlan,
    BAOABLangevinPlan,
    NoseHooverChainPlan,
    StochasticVelocityRescalingPlan,
    VelocityRescalingPlan,
    GeneralizedLangevinPlan,
    MTKNPTPlan,
    MTKNPHPlan,
)
_TEMPERATURE_METHODS = (
    BAOABLangevinPlan,
    NoseHooverChainPlan,
    StochasticVelocityRescalingPlan,
    VelocityRescalingPlan,
    GeneralizedLangevinPlan,
    MTKNPTPlan,
)
_CELL_METHODS = (MTKNPTPlan, MTKNPHPlan)


def _schedule(
    integrator: AtomisticIntegratorPlan, constrained: bool, /
) -> AtomisticSplittingPlan:
    if isinstance(integrator, VelocityVerletPlan):
        return AtomisticSplittingPlan.velocity_verlet(constrained=constrained)
    if isinstance(integrator, BAOABLangevinPlan):
        return AtomisticSplittingPlan.baoab(constrained=constrained)
    if isinstance(
        integrator,
        (NoseHooverChainPlan, StochasticVelocityRescalingPlan, GeneralizedLangevinPlan),
    ):
        return AtomisticSplittingPlan.symmetric_thermostat(constrained=constrained)
    if isinstance(integrator, VelocityRescalingPlan):
        return AtomisticSplittingPlan.terminal_thermostat(constrained=constrained)
    if isinstance(integrator, MTKNPTPlan):
        return AtomisticSplittingPlan.mtk(thermostatted=True)
    return AtomisticSplittingPlan.mtk(thermostatted=False)


def _underlying_program(potential: AbstractPreparedAtomisticHamiltonian, /) -> Any:
    from ._alchemical import PreparedControlledHamiltonian

    return (
        potential.potential
        if isinstance(potential, PreparedControlledHamiltonian)
        else potential
    )


def _cell_derivative_capable(potential: AbstractPreparedAtomisticHamiltonian, /) -> bool:
    return bool(_underlying_program(potential).plan.capabilities.cell_derivative)


def _stateful_terms(
    potential: AbstractPreparedAtomisticHamiltonian, /
) -> tuple[Any, ...]:
    """Prepared program terms that consume a read-only term state."""
    return tuple(
        term
        for term in _underlying_program(potential).terms
        if term.requirements.state_required
    )


def _stateful_term_slots(
    potential: AbstractPreparedAtomisticHamiltonian, /
) -> tuple[str, ...]:
    """Declared state slots (term ``name``) of the program's stateful terms."""
    return tuple(sorted(term.name for term in _stateful_terms(potential)))


def _chain_layout(parameters: NoseHooverChainParameters, /) -> list[int]:
    return [
        parameters.chain_length,
        parameters.yoshida_suzuki_order,
        parameters.substeps,
    ]


def _integrator_layout(integrator: AtomisticIntegratorPlan, /) -> dict[str, Any]:
    """Static method selectors of an integrator, never its dynamic parameters."""
    layout: dict[str, Any] = {
        "kind": type(integrator).__name__,
        "step_size": integrator.step_size,
    }
    if isinstance(
        integrator,
        (BAOABLangevinPlan, StochasticVelocityRescalingPlan, GeneralizedLangevinPlan),
    ):
        layout["realization_id"] = integrator.realization_id
    if isinstance(integrator, NoseHooverChainPlan):
        layout["chain"] = _chain_layout(integrator.chain)
    if isinstance(integrator, VelocityRescalingPlan):
        layout["interval"] = integrator.interval
    if isinstance(integrator, GeneralizedLangevinPlan):
        layout["channels"] = integrator.transition_matrix.shape[0]
        layout["covariance_tolerance"] = integrator.covariance_tolerance
    if isinstance(integrator, _CELL_METHODS):
        barostat = integrator.barostat
        layout["barostat"] = {
            "chart": barostat.chart.value,
            "time_constant": barostat.time_constant,
            "maximum_condition": barostat.maximum_condition,
            "inertia_temperature": barostat.inertia_temperature is not None,
        }
    if isinstance(integrator, MTKNPTPlan):
        layout["particle_chain"] = _chain_layout(integrator.particle_chain)
        layout["cell_chain"] = _chain_layout(integrator.cell_chain)
    return layout


def neighborhood_route_work(neighborhoods: AtomisticNeighborhoodBundleState, /) -> Array:
    """Stored routes of every role view, charged as neighborhood work."""
    if not neighborhoods.views:
        return jnp.zeros((), dtype=jnp.int32)
    return jnp.sum(
        jnp.stack(
            tuple(
                jnp.sum(view.evidence.stored_routes, dtype=jnp.int32)
                for view in neighborhoods.views
            )
        ),
        dtype=jnp.int32,
    )


def _neighborhood_reasons(neighborhoods: AtomisticNeighborhoodBundleState, /) -> Array:
    """Map per-role bundle evidence of the single case to rejection flags."""
    reasons = jnp.zeros((), dtype=jnp.int32)

    def flag(condition: Array, reason: AtomisticStepRejectionReason) -> Array:
        return jnp.where(jnp.any(condition), int(reason), 0).astype(jnp.int32)

    geometry = neighborhoods.geometry
    geometry_valid = jnp.all(
        ~geometry.active | jnp.all(jnp.isfinite(geometry.positions), axis=-1)
    ) & jnp.all(geometry.site_successful)
    reasons = reasons | flag(~geometry_valid, AtomisticStepRejectionReason.DOMAIN)
    for view in neighborhoods.views:
        evidence = view.evidence
        reasons = reasons | flag(
            evidence.source_capacity_failure,
            AtomisticStepRejectionReason.SOURCE_CAPACITY,
        )
        reasons = reasons | flag(
            evidence.route_overflow | evidence.degree_overflow,
            AtomisticStepRejectionReason.ROUTE_CAPACITY,
        )
        reasons = reasons | flag(
            evidence.source_scientific_failure
            | evidence.representation_overflow
            | evidence.entity_failure
            | evidence.stale
            | ~evidence.initialized,
            AtomisticStepRejectionReason.DOMAIN,
        )
        reasons = reasons | flag(
            evidence.reversal_open | evidence.identity_conflict,
            AtomisticStepRejectionReason.RELATION,
        )
        reasons = reasons | flag(
            evidence.repeated_minimum_image,
            AtomisticStepRejectionReason.IMAGE_AMBIGUITY,
        )
    return reasons


def _flag(condition: Array, reason: AtomisticStepRejectionReason, /) -> Array:
    return jnp.where(condition, int(reason), 0).astype(jnp.int32)


def _strain_reasons(required: bool, available: Array, resources: Array, /) -> Array:
    """Attribute unavailable strain without inventing a second overflow cause.

    Resource refusal invalidates every numerical output, including strain.
    A same-attempt capacity replay re-evaluates that unresolved response;
    independent scientific reasons remain present and still block replay.
    """
    if not required:
        return jnp.zeros((), dtype=jnp.int32)
    overflowed = (resources & CAPACITY_REJECTIONS) != 0
    return _flag(
        ~available & ~overflowed, AtomisticStepRejectionReason.STRAIN_UNAVAILABLE
    )


def initial_atomistic_energy_ledger(
    kinetic: Array,
    potential: Array,
    extended_terms: tuple[Array, Array, Array],
    /,
) -> AtomisticEnergyLedgerState:
    """Ledger of an admitted initial state: no heat, work or residual booked."""
    cell_kinetic, pressure_volume, chains = extended_terms
    total = kinetic + potential
    extended = total + cell_kinetic + pressure_volume + chains
    zero = jnp.zeros((), dtype=jnp.result_type(kinetic))
    return AtomisticEnergyLedgerState(
        initial_kinetic_energy=kinetic,
        initial_potential_energy=potential,
        initial_extended_energy=extended,
        kinetic_energy=kinetic,
        potential_energy=potential,
        total_energy=total,
        cell_kinetic_energy=cell_kinetic,
        pressure_volume=pressure_volume,
        thermostat_energy=chains,
        extended_energy=extended,
        thermostat_heat=zero,
        reservoir_heat=zero,
        barostat_work=zero,
        external_work=zero,
        constraint_work=zero,
        cumulative_balance_residual=zero,
        last_relative_energy_change=zero,
        accepted_steps=jnp.zeros((), dtype=jnp.int32),
    )


def advance_atomistic_energy_ledger(
    previous: AtomisticEnergyLedgerState,
    kinetic: Array,
    potential: Array,
    extended_terms: tuple[Array, Array, Array],
    /,
    *,
    thermostat_heat: Array,
    reservoir_heat: Array,
    constraint_work: Array,
    barostat_work: Array,
    external_work: Array,
    balanced: bool,
    accepted_increment: int,
) -> AtomisticEnergyLedgerState:
    """Book one transition (``balanced``) or rebinding (external work) on a ledger.

    A balanced transition's ``external_work`` is the trajectory work of its
    nonconservative force ports and enters the balance; an unbalanced
    rebinding books its energy jump as external work.
    """
    cell_kinetic, pressure_volume, chains = extended_terms
    total = kinetic + potential
    extended = total + cell_kinetic + pressure_volume + chains
    if balanced:
        balance = (
            extended
            - previous.extended_energy
            - reservoir_heat
            - constraint_work
            - external_work
        )
        scale = jnp.maximum(
            jnp.maximum(jnp.abs(extended), jnp.abs(previous.extended_energy)),
            jnp.finfo(extended.dtype).tiny,
        )
        residual = previous.cumulative_balance_residual + balance
        relative = jnp.abs(balance) / scale
    else:
        residual = previous.cumulative_balance_residual
        relative = jnp.zeros_like(previous.last_relative_energy_change)
    return AtomisticEnergyLedgerState(
        initial_kinetic_energy=previous.initial_kinetic_energy,
        initial_potential_energy=previous.initial_potential_energy,
        initial_extended_energy=previous.initial_extended_energy,
        kinetic_energy=kinetic,
        potential_energy=potential,
        total_energy=total,
        cell_kinetic_energy=cell_kinetic,
        pressure_volume=pressure_volume,
        thermostat_energy=chains,
        extended_energy=extended,
        thermostat_heat=previous.thermostat_heat + thermostat_heat,
        reservoir_heat=previous.reservoir_heat + reservoir_heat,
        barostat_work=previous.barostat_work + barostat_work,
        external_work=previous.external_work + external_work,
        constraint_work=previous.constraint_work + constraint_work,
        cumulative_balance_residual=residual,
        last_relative_energy_change=relative,
        accepted_steps=previous.accepted_steps + accepted_increment,
    )


class AtomisticDynamicsPlan(StrictModule, NonTrainableState):
    """System, bundle-bound Hamiltonian, method and constraints of one runtime.

    The neighborhood resources are those bound to ``potential`` (its
    ``neighborhood_bundle``); there is no second neighborhood argument.
    ``strain_response`` requests ``dE/d strain`` with every force evaluation;
    it defaults to true for a full 3x3 periodic cell whose program owns a cell
    derivative and is mandatory for integrator-owned cell methods.
    ``force_ports`` are typed nonconservative force contributions; a runtime
    with ports has no complete scalar force definition and refuses
    integrator-owned cell methods (and every energy-based move downstream).
    Port state slots (``port.name``) are distinct from every stateful term
    name.  The plan digest binds the material system and Hamiltonian, so it
    is the dynamic ``plan_identity`` (host ``plan_id``, unchanged digest).
    """

    system: PreparedAtomisticSystem
    potential: AbstractPreparedAtomisticHamiltonian
    schedule: AtomisticSplittingPlan
    integrator: AtomisticIntegratorPlan
    constraints: PreparedDistanceConstraints | None
    force_ports: tuple[AbstractAtomisticForcePort, ...]
    plan_identity: IdentityWords
    strain_response: bool = eqx.field(static=True)

    @checked
    def __init__(
        self,
        system: PreparedAtomisticSystem,
        potential: AbstractPreparedAtomisticHamiltonian,
        integrator: AtomisticIntegratorPlan,
        /,
        *,
        constraints: PreparedDistanceConstraints | None = None,
        strain_response: bool | None = None,
        force_ports: tuple[AbstractAtomisticForcePort, ...] = (),
    ) -> None:
        if potential.system.prepared_id != system.prepared_id:
            raise ValueError("Potential program belongs to another atomistic system.")
        bundle = potential.neighborhood_bundle
        if not isinstance(bundle, PreparedAtomisticNeighborhoodBundle):
            raise TypeError("The Hamiltonian must bind a prepared neighborhood bundle.")
        if bundle.system.prepared_id != system.prepared_id:
            raise ValueError("Neighborhood bundle belongs to another atomistic system.")
        if bundle.plan.case_count != 1:
            raise ValueError(
                "Atomistic dynamics binds a single-case neighborhood bundle."
            )
        if not isinstance(integrator, _INTEGRATORS):
            raise TypeError(f"Unsupported dynamics method {type(integrator).__name__}.")
        if constraints is not None:
            if not isinstance(constraints, PreparedDistanceConstraints):
                raise TypeError(
                    "constraints must be PreparedDistanceConstraints or None."
                )
            if constraints.system.prepared_id != system.prepared_id:
                raise ValueError("Constraints belong to another atomistic system.")
        ports = tuple(force_ports)
        if any(not isinstance(port, AbstractAtomisticForcePort) for port in ports):
            raise TypeError("force_ports must contain AbstractAtomisticForcePort values.")
        port_ids = tuple(port.port_id for port in ports)
        port_slots = tuple(port.name for port in ports)
        if (
            len(set(port_ids)) != len(port_ids)
            or len(set(port_slots)) != len(port_slots)
            or set(port_slots) & set(_stateful_term_slots(potential))
        ):
            raise ValueError(
                "Force-port IDs and slots must be unique and slots distinct from "
                "stateful term names."
            )
        if any(port.system_id != system.prepared_id for port in ports):
            raise ValueError("A force port belongs to another atomistic system.")
        if ports and isinstance(integrator, _CELL_METHODS):
            raise ValueError(
                "Integrator-owned cell dynamics needs the complete scalar strain "
                "response; nonconservative force ports have none."
            )
        cell = system.cell
        full_cell = (
            cell is not None
            and cell.fully_periodic
            and cell.rank == 3
            and cell.ambient_dimension == 3
        )
        capable = full_cell and _cell_derivative_capable(potential)
        if isinstance(integrator, _CELL_METHODS):
            if not capable:
                raise ValueError(
                    "Integrator-owned cell dynamics requires a fully periodic 3x3 "
                    "cell and a program that owns its cell derivative."
                )
            if constraints is not None:
                raise ValueError(
                    "Constrained MTK cell dynamics lacks a derived constrained "
                    "measure and projection; refusing."
                )
            if strain_response is False:
                raise ValueError("Cell dynamics requires the strain response.")
        requested = capable if strain_response is None else bool(strain_response)
        if requested and not capable:
            raise ValueError(
                "Strain response requires a fully periodic 3x3 cell and a program "
                "that owns its cell derivative."
            )
        self.system = system
        self.potential = potential
        self.integrator = integrator
        self.schedule = _schedule(integrator, constraints is not None)
        self.constraints = constraints
        self.force_ports = ports
        self.strain_response = requested
        self.plan_identity = IdentityWords.of(
            {
                "kind": "atomistic-dynamics-plan",
                "system": system.prepared_id,
                "potential": potential.prepared_id,
                "integrator": integrator.plan_id,
                "schedule": self.schedule.plan_id,
                "constraints": None if constraints is None else constraints.prepared_id,
                "strain_response": requested,
                "force_ports": list(port_ids),
            }
        )

    @property
    def plan_id(self) -> str:
        """Host decoding of the canonical dynamics-plan digest."""
        return self.plan_identity.fingerprint()

    def prepare(self, /) -> PreparedAtomisticDynamics:
        return PreparedAtomisticDynamics(self)


class _Transition(StrictModule):
    """Mutable-free record of one executed schedule."""

    positions: Array
    unwrapped: Array
    images: Array
    momenta: Array
    cell_vectors: Array
    neighborhoods: AtomisticNeighborhoodBundleState
    evaluation: Any
    ports: AtomisticForcePortEvaluation | None
    nonconservative_work: Array
    controllers: AtomisticControllerState
    constraint_lagrange: Array
    constraint_position_residual: Array
    constraint_velocity_residual: Array
    thermostat_heat: Array
    reservoir_heat: Array
    constraint_work: Array
    constraint_successful: Array
    thermostat_successful: Array
    barostat_successful: Array
    cell_successful: Array


class PreparedAtomisticDynamics(StrictModule):
    """One prepared dynamics runtime with a content-free PyTree.

    Static: declared state slots (``program_term_state_slots``: stateful term
    names; ``term_state_slots``: those plus force-port slots), the strain
    selector and ``layout_id`` (system, Hamiltonian and bundle layouts,
    integrator selectors, schedule, constraint layout, port slots).  The
    prepared digest is the dynamic ``prepared_identity`` (host
    ``prepared_id``, unchanged digest).  Every material or parameter
    revision of one ``layout_id`` shares one executable; compiled transitions
    validate identities with words (``binding_valid``, ``force_cache_valid``,
    ``term_states_successful``) and refuse foreign or stale states by status.
    """

    plan: AtomisticDynamicsPlan
    system: PreparedAtomisticSystem
    potential: AbstractPreparedAtomisticHamiltonian
    bundle: PreparedAtomisticNeighborhoodBundle
    integrator: AtomisticIntegratorPlan
    schedule: AtomisticSplittingPlan
    constraints: PreparedDistanceConstraints | None
    force_ports: tuple[AbstractAtomisticForcePort, ...]
    prepared_identity: IdentityWords
    program_term_state_slots: tuple[str, ...] = eqx.field(static=True)
    term_state_slots: tuple[str, ...] = eqx.field(static=True)
    strain_response: bool = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    @checked
    def __init__(self, plan: AtomisticDynamicsPlan, /) -> None:
        bundle = plan.potential.neighborhood_bundle
        program_slots = _stateful_term_slots(plan.potential)
        slots = tuple(
            sorted(program_slots + tuple(port.name for port in plan.force_ports))
        )
        self.plan = plan
        self.system = plan.system
        self.potential = plan.potential
        self.bundle = bundle
        self.integrator = plan.integrator
        self.schedule = plan.schedule
        self.constraints = plan.constraints
        self.force_ports = plan.force_ports
        self.program_term_state_slots = program_slots
        self.term_state_slots = slots
        self.strain_response = plan.strain_response
        self.prepared_identity = IdentityWords.of(
            {"kind": "prepared-atomistic-dynamics", "plan": plan.plan_id}
        )
        self.layout_id = canonical_fingerprint(
            {
                "kind": "prepared-atomistic-dynamics-layout",
                "system": plan.system.layout_id,
                "potential": plan.potential.layout_id,
                "bundle": bundle.layout_id,
                "integrator": _integrator_layout(plan.integrator),
                "schedule": plan.schedule.plan_id,
                "constraints": None
                if plan.constraints is None
                else plan.constraints.layout_id,
                "strain_response": plan.strain_response,
                "force_ports": [
                    [type(port).__name__, port.name] for port in plan.force_ports
                ],
                "term_state_slots": list(slots),
            }
        )

    @property
    def prepared_id(self) -> str:
        """Host decoding of the canonical prepared-dynamics digest."""
        return self.prepared_identity.fingerprint()

    # ------------------------------------------------------------------ geometry

    @property
    def periodic(self) -> bool:
        return self.system.cell is not None

    def _require_state(
        self,
        state: AtomisticDynamicsState,
        thermodynamic: PreparedThermodynamicStateTable,
        /,
    ) -> None:
        """Static (trace-safe) layout admission of a state and its table.

        Identities are never decoded here; ``binding_valid`` compares them as
        words and every transition folds the result into its status.
        """
        thermodynamic.require_dynamics_layout(self)
        if state.neighborhoods.layout_id != self.bundle.layout_id:
            raise ValueError(
                "State neighborhoods belong to another resource layout; rebind and "
                "prime the state first."
            )
        if tuple(state.term_states.slots) != self.term_state_slots:
            raise ValueError(
                "State term-state slots do not match the runtime's stateful terms "
                f"and force ports {self.term_state_slots}."
            )

    def binding_valid(
        self,
        state: AtomisticDynamicsState,
        thermodynamic: PreparedThermodynamicStateTable,
        /,
    ) -> Array:
        """Traced: state, table and neighborhoods are bound to exactly this runtime."""
        return (
            thermodynamic.bound_to_dynamics(self)
            & state.prepared_dynamics_identity.equals(self.prepared_identity)
            & state.thermodynamic_table_identity.equals(thermodynamic.table_identity)
            & state.neighborhoods.bundle_identity.equals(self.bundle.prepared_identity)
        )

    def _require_state_host(
        self,
        state: AtomisticDynamicsState,
        thermodynamic: PreparedThermodynamicStateTable,
        /,
    ) -> None:
        """Exact host refusal of host lifecycle transactions (rebind/transition)."""
        self._require_state(state, thermodynamic)
        thermodynamic.validate_dynamics(self)
        if (
            state.prepared_dynamics_id != self.prepared_id
            or state.thermodynamic_table_id != thermodynamic.table_id
        ):
            raise ValueError("State and thermodynamic table identities do not match.")
        if (
            state.force.program_id != self.potential.prepared_id
            or state.force.bundle_id != self.bundle.prepared_id
            or state.neighborhoods.bundle_id != self.bundle.prepared_id
        ):
            raise ValueError(
                "State force cache or neighborhoods belong to another Hamiltonian "
                "resource binding; rebind and prime the state first."
            )

    def _unwrapped(
        self,
        kinematics: AtomisticKinematics,
        cell_vectors: Array | None = None,
        /,
    ) -> Array:
        cell = self.system.cell
        if cell is None:
            return kinematics.positions
        vectors = (
            cell.vectors.astype(kinematics.positions.dtype)
            if cell_vectors is None or cell_vectors.size == 0
            else jnp.asarray(cell_vectors, dtype=kinematics.positions.dtype)
        )
        return kinematics.positions + contract(
            "ni,ij->nj",
            kinematics.image_counts.astype(kinematics.positions.dtype),
            vectors,
        )

    def _wrap(
        self, unwrapped: Array, vectors: Array, images: Array, /
    ) -> tuple[Array, Array]:
        if self.system.cell is None:
            return unwrapped, images
        return self.system.cell.wrap_with_vectors(unwrapped, vectors)

    def velocity(self, state: AtomisticDynamicsState, /) -> Array:
        return state.kinematics.momenta * self.system.inverse_masses[:, None]

    def interaction_sites(
        self, state: AtomisticDynamicsState, /
    ) -> AtomisticInteractionSiteState:
        """Realized interaction sites; unsuccessful for a foreign runtime's state."""
        if self.system.cell is None or state.cell_vectors.size == 0:
            sites = self.system.coordinate_map.realize(state.kinematics.positions)
        else:
            fractional = self.system.cell.fractional_with_vectors(
                state.kinematics.positions, state.cell_vectors
            )
            sites = self.system.coordinate_map.realize(
                state.kinematics.positions,
                cell=self.system.cell,
                fractional_positions=fractional,
                cell_vectors=state.cell_vectors,
            )
        return eqx.tree_at(
            lambda value: value.successful,
            sites,
            sites.successful
            & state.prepared_dynamics_identity.equals(self.prepared_identity),
        )

    def kinetic_energy(self, momenta: Array, /) -> Array:
        kinetic_factor = self.system.plan.units.kinetic_to_energy
        per_atom = (
            0.5
            * kinetic_factor
            * jnp.sum(momenta * momenta * self.system.inverse_masses[:, None], axis=-1)
        )
        return jnp.sum(jnp.where(self.system.mobile_mask, per_atom, 0.0))

    # ------------------------------------------------------- neighborhood bundle

    def initialize_neighborhoods(
        self, positions: Array, /, *, cell_vectors: Array, image_counts: Array
    ) -> AtomisticNeighborhoodBundleState:
        return self.bundle.initialize(
            bind_atomistic_bundle_geometry(
                self.bundle,
                self.system,
                positions,
                cell_vectors=cell_vectors if self.periodic else None,
                image_counts=image_counts if self.periodic else None,
            )
        )

    def refresh_neighborhoods(
        self,
        neighborhoods: AtomisticNeighborhoodBundleState,
        positions: Array,
        /,
        *,
        cell_vectors: Array,
        image_counts: Array,
    ) -> AtomisticNeighborhoodBundleState:
        """Pure candidate refresh of the accepted bundle at a proposed geometry."""
        return self.bundle.refresh(
            neighborhoods,
            bind_atomistic_bundle_geometry(
                self.bundle,
                self.system,
                positions,
                cell_vectors=cell_vectors if self.periodic else None,
                image_counts=image_counts if self.periodic else None,
            ),
        )

    # ---------------------------------------------------------------- evaluation

    @property
    def conservative_force_definition(self) -> bool:
        """Whether every applied force is the gradient of the program scalar."""
        return not self.force_ports

    def require_conservative_force_definition(self, consumer: str, /) -> None:
        if self.force_ports:
            raise ValueError(
                f"{consumer} needs the complete scalar Hamiltonian; this runtime "
                "applies nonconservative force ports "
                f"{[port.port_id for port in self.force_ports]} with no energy."
            )

    def initial_term_states(self, dtype: Any, /) -> AtomisticTermStateBindings:
        """Declared initial read-only state of every stateful term and port.

        States are keyed by declared slot: the term ``name`` or the port
        ``name``; each carries its owner's identity words.
        """
        states: dict[str, AbstractAtomisticTermState] = {}
        for term in _stateful_terms(self.potential):
            initial = term.initial_term_state(dtype)
            if initial is None:
                raise ValueError(f"State-required term {term.name!r} has no state.")
            states[term.name] = initial
        for port in self.force_ports:
            states[port.name] = port.initial_term_state(dtype)
        return AtomisticTermStateBindings(states)

    def _term_state_owners(self, /) -> tuple[tuple[str, IdentityWords], ...]:
        """Declared slot and owner identity of every stateful term and port."""
        return tuple(
            (term.name, term.term_identity) for term in _stateful_terms(self.potential)
        ) + tuple((port.name, port.owner_identity) for port in self.force_ports)

    def _term_state_owner_ids(self, /) -> tuple[tuple[str, str], ...]:
        """Host decoding of every slot's owner identity."""
        return tuple(
            (slot, owner.fingerprint()) for slot, owner in self._term_state_owners()
        )

    def require_term_states(
        self, term_states: AtomisticTermStateBindings, /
    ) -> AtomisticTermStateBindings:
        """Static slot-layout admission of term states (trace-safe).

        The owner of each bound state is validated in compiled code by
        ``term_states_successful`` (``TERM_STATE`` refusal).
        """
        if not isinstance(term_states, AtomisticTermStateBindings):
            raise TypeError("term_states must be AtomisticTermStateBindings.")
        if tuple(term_states.slots) != self.term_state_slots:
            raise ValueError(
                "term_states must cover exactly the runtime's stateful terms and "
                f"force ports {self.term_state_slots}."
            )
        return term_states

    def term_revisions(self, term_states: AtomisticTermStateBindings, /) -> Array:
        return jnp.asarray(term_states.revisions, dtype=jnp.int32)

    def term_states_successful(self, term_states: AtomisticTermStateBindings, /) -> Array:
        """Traced: every slot holds a successful state of exactly its owner."""
        valid = jnp.asarray(True)
        for slot, owner in self._term_state_owners():
            valid = valid & term_states.bound_to(slot, owner)
        return valid

    def _program_term_states(
        self, term_states: AtomisticTermStateBindings | None, /
    ) -> AtomisticTermStateBindings | None:
        if not self.program_term_state_slots:
            return None
        if term_states is None:
            raise ValueError(
                "The Hamiltonian has stateful terms; supply their term_states."
            )
        return term_states.subset(self.program_term_state_slots)

    def _potential_kwargs(
        self,
        unwrapped_positions: Array,
        image_counts: Array,
        species: Array,
        cell_vectors: Array,
        controls: Array,
        /,
        term_states: AtomisticTermStateBindings | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "unwrapped_positions": unwrapped_positions,
            "species": species,
            "cell": self.system.cell,
        }
        if self.periodic:
            kwargs["cell_vectors"] = cell_vectors
            kwargs["image_counts"] = image_counts
        if controls.shape[0] > 0:
            kwargs["control_values"] = controls
        program_states = self._program_term_states(term_states)
        if program_states is not None:
            kwargs["term_states"] = program_states
        return kwargs

    def evaluate_configuration(
        self,
        positions: Array,
        neighborhoods: AtomisticNeighborhoodBundleState,
        /,
        *,
        unwrapped_positions: Array,
        image_counts: Array,
        species: Array,
        cell_vectors: Array,
        controls: Array,
        term_states: AtomisticTermStateBindings | None = None,
    ) -> AtomisticHamiltonianEvaluation:
        """Energy, forces and the declared strain response at one configuration."""
        evaluation = self.potential.evaluate(
            positions,
            neighborhoods,
            compute_stress=self.strain_response,
            **self._potential_kwargs(
                unwrapped_positions,
                image_counts,
                species,
                cell_vectors,
                controls,
                term_states,
            ),
        )
        if self.strain_response and evaluation.strain_derivative is None:
            raise RuntimeError("The Hamiltonian omitted a requested strain derivative.")
        return evaluation

    def energy_configuration(
        self,
        positions: Array,
        neighborhoods: AtomisticNeighborhoodBundleState,
        /,
        *,
        unwrapped_positions: Array,
        image_counts: Array,
        species: Array,
        cell_vectors: Array,
        controls: Array,
        term_states: AtomisticTermStateBindings | None = None,
    ) -> AtomisticPotentialEnergyEvaluation:
        """Program scalar energy; never the work of nonconservative ports."""
        energy, auxiliary = self.potential.energy(
            positions,
            neighborhoods,
            **self._potential_kwargs(
                unwrapped_positions,
                image_counts,
                species,
                cell_vectors,
                controls,
                term_states,
            ),
        )
        return AtomisticPotentialEnergyEvaluation(
            energy, auxiliary.successful & jnp.isfinite(energy)
        )

    def evaluate_force_ports(
        self,
        positions: Array,
        /,
        *,
        unwrapped_positions: Array,
        image_counts: Array,
        species: Array,
        cell_vectors: Array,
        term_states: AtomisticTermStateBindings,
    ) -> AtomisticForcePortEvaluation | None:
        """Summed force-port forces at one accepted force-boundary geometry."""
        if not self.force_ports:
            return None
        geometry = AtomisticForcePortGeometry(
            positions=positions,
            unwrapped_positions=unwrapped_positions,
            cell_vectors=cell_vectors if self.periodic else None,
            image_counts=image_counts if self.periodic else None,
            species=species,
        )
        forces = jnp.zeros_like(positions)
        successful = jnp.asarray(True)
        status = jnp.zeros((), dtype=jnp.int32)
        evidence: tuple[StrictModule, ...] = ()
        for port in self.force_ports:
            result = port.forces(geometry, term_states[port.name])
            if result.forces.shape != positions.shape:
                raise ValueError(f"Force port {port.name!r} returned a wrong shape.")
            forces = forces + jnp.where(
                self.system.active_mask[:, None], result.forces, 0.0
            )
            successful = successful & result.successful
            status = status | jnp.asarray(result.status, dtype=jnp.int32)
            evidence = evidence + result.evidence
        successful = successful & jnp.all(jnp.isfinite(forces))
        return AtomisticForcePortEvaluation(
            forces=jnp.where(successful, forces, 0.0),
            successful=successful,
            status=status,
            evidence=evidence,
        )

    def force_state(
        self,
        evaluation: AtomisticHamiltonianEvaluation,
        neighborhoods: AtomisticNeighborhoodBundleState,
        configuration_revision: Array,
        hamiltonian_index: Array,
        controls: Array,
        random_key: Array,
        /,
        *,
        term_states: AtomisticTermStateBindings,
        ports: AtomisticForcePortEvaluation | None,
    ) -> AtomisticForceState:
        if (ports is None) != (not self.force_ports):
            raise ValueError(
                "A force state requires exactly the runtime's force-port evaluation."
            )
        dtype = evaluation.energy.dtype
        nan = jnp.full((3, 3), jnp.nan, dtype=dtype)
        strain = evaluation.strain_derivative
        stress = evaluation.stress
        available = (
            jnp.asarray(False)
            if strain is None
            else evaluation.successful & jnp.all(jnp.isfinite(strain))
        )
        nonconservative = (
            jnp.zeros_like(evaluation.forces) if ports is None else ports.forces
        )
        successful = evaluation.successful & self.term_states_successful(term_states)
        if ports is not None:
            successful = successful & ports.successful
        return AtomisticForceState(
            forces=evaluation.forces,
            nonconservative_forces=nonconservative,
            potential_energy=evaluation.energy,
            term_energies=evaluation.term_energies,
            atom_energy=evaluation.atom_energy,
            strain_derivative=nan if strain is None else strain,
            stress=nan if stress is None else stress,
            strain_available=available,
            configuration_revision=jnp.asarray(configuration_revision, dtype=jnp.int32),
            hamiltonian_index=jnp.asarray(hamiltonian_index, dtype=jnp.int32),
            controls=jnp.asarray(controls),
            source_epochs=neighborhoods.epochs,
            graph_epochs=neighborhoods.graph_epochs,
            random_key=jnp.asarray(random_key, dtype=jnp.uint32),
            term_revisions=self.term_revisions(term_states),
            successful=successful,
            program_identity=evaluation.program_identity,
            bundle_identity=neighborhoods.bundle_identity,
        )

    def _boundary_reasons(
        self,
        term_states: AtomisticTermStateBindings,
        ports: AtomisticForcePortEvaluation | None,
        /,
    ) -> Array:
        reasons = _flag(
            ~self.term_states_successful(term_states),
            AtomisticStepRejectionReason.TERM_STATE,
        )
        if ports is not None:
            reasons = reasons | _flag(
                ~ports.successful, AtomisticStepRejectionReason.FORCE_PORT
            )
        return reasons

    @staticmethod
    def force_cache_finite(force: AtomisticForceState, /) -> Array:
        """Finiteness of the cached fields a transition actually consumes.

        Unrequested strain/stress are NaN by contract (``strain_available``
        false), so they are checked only where available; whether a strain
        response is *required* is a separate ``STRAIN_UNAVAILABLE`` admission.
        """
        required = tree_allfinite(
            (
                force.forces,
                force.nonconservative_forces,
                force.potential_energy,
                force.term_energies,
                force.atom_energy,
            )
        )
        strain_finite = jnp.all(jnp.isfinite(force.strain_derivative)) & jnp.all(
            jnp.isfinite(force.stress)
        )
        return required & (~force.strain_available | strain_finite)

    def force_cache_valid(
        self, state: AtomisticDynamicsState, controls: Array, /
    ) -> Array:
        """The single force-cache validity rule of every transition.

        Includes the traced binding of the cache to this runtime's prepared
        Hamiltonian and neighborhood bundle (identity words), so a cache of
        another revision is stale (``STALE_FORCE``) rather than a host error.
        """
        force = state.force
        valid = (
            force.successful
            & force.program_identity.equals(self.potential.prepared_identity)
            & force.bundle_identity.equals(self.bundle.prepared_identity)
            & (force.configuration_revision == state.configuration_revision)
            & jnp.all(force.controls == controls)
            & jnp.all(force.source_epochs == state.neighborhoods.epochs)
            & jnp.all(force.graph_epochs == state.neighborhoods.graph_epochs)
            & jnp.all(force.random_key == state.random_key)
            & jnp.all(force.term_revisions == self.term_revisions(state.term_states))
            & self.term_states_successful(state.term_states)
        )
        if self.strain_response:
            valid = valid & force.strain_available
        return valid

    # ------------------------------------------------------------ controllers

    def _thermal_energy(self, row: Any, /) -> Array:
        return self.system.plan.units.boltzmann_constant * row.temperature

    def _barostat_mass(self, row: Any, degrees_of_freedom: ArrayLike, /) -> Array:
        integrator = self.integrator
        if not isinstance(integrator, _CELL_METHODS):
            raise RuntimeError("Only MTK methods own a barostat mass.")
        barostat = integrator.barostat
        temperature = (
            row.temperature
            if barostat.inertia_temperature is None
            else jnp.asarray(barostat.inertia_temperature, dtype=row.temperature.dtype)
        )
        return barostat_mass(
            barostat,
            degrees_of_freedom,
            self.system.plan.units.boltzmann_constant * temperature,
        )

    def initial_controllers(
        self, cell_vectors: Array, dtype: Any, /
    ) -> AtomisticControllerState:
        integrator = self.integrator
        particle_chain = None
        cell_chain = None
        gle = None
        cell = None
        if isinstance(integrator, NoseHooverChainPlan):
            particle_chain = initial_nose_hoover_chain_state(integrator.chain, dtype)
        if isinstance(integrator, MTKNPTPlan):
            particle_chain = initial_nose_hoover_chain_state(
                integrator.particle_chain, dtype
            )
            cell_chain = initial_nose_hoover_chain_state(integrator.cell_chain, dtype)
        if isinstance(integrator, _CELL_METHODS):
            cell = initial_cell_controller(integrator.barostat, cell_vectors)
        if isinstance(integrator, GeneralizedLangevinPlan):
            gle = initial_generalized_langevin_memory(
                integrator, self.system.capacity, dtype
            )
        return AtomisticControllerState(
            particle_chain=particle_chain,
            cell_chain=cell_chain,
            gle=gle,
            cell=cell,
            move_counter=jnp.zeros((), dtype=jnp.uint32),
            accepted_moves=jnp.zeros((), dtype=jnp.int32),
            cell_epoch=jnp.zeros((), dtype=jnp.int32),
            rescale_events=jnp.zeros((), dtype=jnp.int32),
        )

    def extended_terms(
        self,
        controllers: AtomisticControllerState,
        cell_vectors: Array,
        thermodynamic: PreparedThermodynamicStateTable,
        row: Any,
        /,
    ) -> tuple[Array, Array, Array]:
        """``(K_cell, P_ext V, E_chains)`` of the method at the given state."""
        dtype = row.temperature.dtype
        zero = jnp.zeros((), dtype=dtype)
        dof = thermodynamic.degrees_of_freedom
        integrator = self.integrator
        chains = zero
        cell_kinetic = zero
        pressure_volume = zero
        if controllers.particle_chain is not None:
            if isinstance(integrator, NoseHooverChainPlan):
                parameters = integrator.chain
            elif isinstance(integrator, MTKNPTPlan):
                parameters = integrator.particle_chain
            else:
                raise RuntimeError("Particle chain requires a Nose-Hoover integrator.")
            chains = chains + nose_hoover_chain_energy(
                parameters, controllers.particle_chain, dof, self._thermal_energy(row)
            )
        if isinstance(integrator, _CELL_METHODS) and controllers.cell is not None:
            mass = self._barostat_mass(row, dof)
            cell_kinetic = cell_kinetic_energy(
                integrator.barostat.chart, controllers.cell.momentum, mass
            )
            pressure_volume = row.pressure * cell_volume(cell_vectors)
            if isinstance(integrator, MTKNPTPlan) and controllers.cell_chain is not None:
                chains = chains + nose_hoover_chain_energy(
                    integrator.cell_chain,
                    controllers.cell_chain,
                    integrator.barostat.chart.dimension,
                    self._thermal_energy(row),
                )
        return cell_kinetic, pressure_volume, chains

    def _ensemble_valid(self, row: Any, /) -> Array:
        valid = row.valid
        if isinstance(self.integrator, _TEMPERATURE_METHODS):
            valid = valid & row.temperature_mask
        if isinstance(self.integrator, _CELL_METHODS):
            valid = valid & row.pressure_mask
        return valid

    # ------------------------------------------------------------ public reads

    @checked
    def evaluate_state(
        self,
        state: AtomisticDynamicsState,
        thermodynamic: PreparedThermodynamicStateTable,
        /,
        *,
        state_index: ArrayLike | None = None,
    ) -> AtomisticHamiltonianEvaluation:
        self._require_state(state, thermodynamic)
        index = (
            state.thermodynamic_state_index
            if state_index is None
            else jnp.asarray(state_index, dtype=jnp.int32)
        )
        row = thermodynamic.state_at_replica(index)
        evaluation = self.evaluate_configuration(
            state.kinematics.positions,
            state.neighborhoods,
            unwrapped_positions=self._unwrapped(state.kinematics, state.cell_vectors),
            image_counts=state.kinematics.image_counts,
            species=state.species,
            cell_vectors=state.cell_vectors,
            controls=row.controls,
            term_states=state.term_states,
        )
        return eqx.tree_at(
            lambda value: value.successful,
            evaluation,
            evaluation.successful & row.valid & self.binding_valid(state, thermodynamic),
        )

    @checked
    def energy_state(
        self,
        state: AtomisticDynamicsState,
        thermodynamic: PreparedThermodynamicStateTable,
        /,
        *,
        state_index: ArrayLike | None = None,
    ) -> AtomisticPotentialEnergyEvaluation:
        self._require_state(state, thermodynamic)
        index = (
            state.thermodynamic_state_index
            if state_index is None
            else jnp.asarray(state_index, dtype=jnp.int32)
        )
        row = thermodynamic.state_at_replica(index)
        evaluation = self.energy_configuration(
            state.kinematics.positions,
            state.neighborhoods,
            unwrapped_positions=self._unwrapped(state.kinematics, state.cell_vectors),
            image_counts=state.kinematics.image_counts,
            species=state.species,
            cell_vectors=state.cell_vectors,
            controls=row.controls,
            term_states=state.term_states,
        )
        return AtomisticPotentialEnergyEvaluation(
            evaluation.energy,
            evaluation.successful & row.valid & self.binding_valid(state, thermodynamic),
        )

    # ------------------------------------------------------------ initialization

    def initialize_state(
        self,
        positions: ArrayLike,
        thermodynamic: PreparedThermodynamicStateTable,
        /,
        *,
        state_index: ArrayLike = 0,
        velocity: ArrayLike | None = None,
        momentum: ArrayLike | None = None,
        time: ArrayLike = 0.0,
        species: ArrayLike | None = None,
        key: PRNGKey,
        term_states: AtomisticTermStateBindings | None = None,
        cell_vectors: ArrayLike | None = None,
    ) -> AtomisticDynamicsState:
        """Form, prime and admit the initial accepted state (raises if invalid).

        The admission-only form of :meth:`initialize_state_detailed`: any
        numerical failure of the initial evaluation raises.
        """
        evaluation = self.initialize_state_detailed(
            positions,
            thermodynamic,
            state_index=state_index,
            velocity=velocity,
            momentum=momentum,
            time=time,
            species=species,
            key=key,
            term_states=term_states,
            cell_vectors=cell_vectors,
        )
        state = evaluation.state
        checked_positions = eqx.error_if(
            state.kinematics.positions,
            ~evaluation.successful,
            "Initial atomistic dynamics state is not admissible.",
        )
        return eqx.tree_at(
            lambda value: value.kinematics.positions, state, checked_positions
        )

    # Initialization is one admission/priming boundary: owner failures must reach
    # the same candidate state before any accepted continuation can be published.
    def initialize_state_detailed(
        self,
        positions: ArrayLike,
        thermodynamic: PreparedThermodynamicStateTable,
        /,
        *,
        state_index: ArrayLike = 0,
        velocity: ArrayLike | None = None,
        momentum: ArrayLike | None = None,
        time: ArrayLike = 0.0,
        species: ArrayLike | None = None,
        key: PRNGKey,
        term_states: AtomisticTermStateBindings | None = None,
        cell_vectors: ArrayLike | None = None,
    ) -> "AtomisticInitialStateEvaluation":
        """Form and prime the initial state; numerical failure is a returned status.

        ``positions`` are unwrapped Cartesian coordinates.  ``cell_vectors``
        is the current periodic cell (for example the last accepted cell of a
        preceding NPT stage); it defaults to the system's reference cell.  The
        configuration is wrapped into that cell with integer image counts, so
        the supplied unwrapped configuration is preserved exactly.
        ``term_states`` default to every stateful term's and force port's
        declared initial state.

        Wrong types, shapes, table layouts and term-state slot layouts raise.
        A nonfinite input, invalid cell, failed constraint projection, failed
        neighborhood, failed potential or force-port evaluation, a term state
        of a foreign owner, a table prepared for another runtime revision
        (``BINDING``) or an inadmissible thermodynamic row is returned as an
        explicitly failed evaluation: the state carries
        ``INVALID_INITIAL_STATE`` and the rejection reasons, its force cache is
        unsuccessful and every transition refuses it as stale.
        """
        if not isinstance(thermodynamic, PreparedThermodynamicStateTable):
            raise TypeError("thermodynamic must be PreparedThermodynamicStateTable.")
        thermodynamic.require_dynamics_layout(self)
        table_bound = thermodynamic.bound_to_dynamics(self)
        thermodynamic_index = jnp.asarray(state_index, dtype=jnp.int32).reshape(())
        time_ = jnp.asarray(time, dtype=self.system.plan.coordinate_dtype).reshape(())
        row = thermodynamic.state_at_replica(thermodynamic_index)
        if (velocity is None) == (momentum is None):
            raise ValueError("Supply exactly one of velocity or momentum.")
        position = jnp.asarray(positions, dtype=self.system.plan.coordinate_dtype)
        expected = (self.system.capacity, 3)
        if position.shape != expected:
            raise ValueError(f"positions must have shape {expected}.")
        cell = self.system.cell
        if cell is None and cell_vectors is not None:
            raise ValueError("A nonperiodic system has no cell_vectors.")
        active_finite = jnp.all(
            jnp.isfinite(jnp.where(self.system.active_mask[:, None], position, 0.0))
        )
        position = jnp.where(self.system.active_mask[:, None], position, 0.0)
        masses = self.system.plan.masses.astype(position.dtype)
        momenta = (
            jnp.asarray(momentum, dtype=position.dtype)
            if momentum is not None
            else masses[:, None] * jnp.asarray(velocity, dtype=position.dtype)
        )
        if momenta.shape != expected:
            raise ValueError(f"momentum or velocity must have shape {expected}.")
        mobile = self.system.mobile_mask
        momenta = jnp.where(mobile[:, None], momenta, 0.0)
        if thermodynamic.momentum_restriction == "center-of-mass":
            total_mass = jnp.sum(jnp.where(mobile, masses, 0.0))
            drift = jnp.sum(momenta, axis=0) / total_mass
            momenta = jnp.where(mobile[:, None], momenta - masses[:, None] * drift, 0.0)
        constraint_lagrange = jnp.zeros(
            (self.system.topology.constraint_count,), dtype=position.dtype
        )
        constraint_position_residual = jnp.zeros((), dtype=position.dtype)
        constraint_velocity_residual = jnp.zeros((), dtype=position.dtype)
        constraint_successful = jnp.asarray(True)
        if self.constraints is not None:
            projection = self.constraints.project_positions(position, position, momenta)
            position = projection.positions
            momenta = projection.momenta
            constraint_lagrange = projection.multipliers
            constraint_position_residual = projection.position_residual
            constraint_velocity_residual = projection.velocity_residual
            constraint_successful = projection.successful
        if cell is None:
            wrapped = position
            image_counts = jnp.zeros(expected, dtype=jnp.int32)
            current_cell = jnp.zeros((0, 0), dtype=position.dtype)
            cell_valid = jnp.asarray(True)
        elif cell_vectors is None:
            wrapped, image_counts = cell.wrap(position)
            current_cell = cell.vectors.astype(position.dtype)
            cell_valid = jnp.asarray(True)
        else:
            current_cell = jnp.asarray(cell_vectors, dtype=position.dtype)
            if current_cell.shape != cell.vectors.shape:
                raise ValueError(f"cell_vectors must have shape {cell.vectors.shape}.")
            cell_valid = jnp.all(jnp.isfinite(current_cell))
            if current_cell.shape == (3, 3):
                cell_valid = cell_valid & (cell_volume(current_cell) > 0.0)
            wrapped, image_counts = cell.wrap_with_vectors(position, current_cell)
        cell_vectors = current_cell
        neighborhoods = self.initialize_neighborhoods(
            wrapped, cell_vectors=cell_vectors, image_counts=image_counts
        )
        species_ = (
            self.system.plan.atom_type_ids
            if species is None
            else jnp.asarray(species, dtype=jnp.int32)
        )
        kinematics = AtomisticKinematics(wrapped, momenta, image_counts)
        bound_states = self.require_term_states(
            self.initial_term_states(position.dtype)
            if term_states is None
            else term_states
        )
        unwrapped = self._unwrapped(kinematics, cell_vectors)
        evaluation = self.evaluate_configuration(
            wrapped,
            neighborhoods,
            unwrapped_positions=unwrapped,
            image_counts=image_counts,
            species=species_,
            cell_vectors=cell_vectors,
            controls=row.controls,
            term_states=bound_states,
        )
        ports = self.evaluate_force_ports(
            wrapped,
            unwrapped_positions=unwrapped,
            image_counts=image_counts,
            species=species_,
            cell_vectors=cell_vectors,
            term_states=bound_states,
        )
        random_key = jr.key_data(key).astype(jnp.uint32)
        revision = jnp.zeros((), dtype=jnp.int32)
        force = self.force_state(
            evaluation,
            neighborhoods,
            revision,
            row.hamiltonian_index,
            row.controls,
            random_key,
            term_states=bound_states,
            ports=ports,
        )
        controllers = self.initial_controllers(cell_vectors, position.dtype)
        kinetic = self.kinetic_energy(momenta)
        terms = self.extended_terms(controllers, cell_vectors, thermodynamic, row)
        ledger = initial_atomistic_energy_ledger(kinetic, evaluation.energy, terms)
        resources = _neighborhood_reasons(neighborhoods)
        reasons = (
            resources
            | self._boundary_reasons(bound_states, ports)
            | _flag(~evaluation.successful, AtomisticStepRejectionReason.POTENTIAL)
            | _flag(~constraint_successful, AtomisticStepRejectionReason.CONSTRAINT)
            | _flag(
                ~self._ensemble_valid(row),
                AtomisticStepRejectionReason.THERMODYNAMIC_STATE,
            )
            | _flag(~cell_valid, AtomisticStepRejectionReason.CELL)
            | _flag(~table_bound, AtomisticStepRejectionReason.BINDING)
            | _flag(
                ~(
                    active_finite
                    & jnp.isfinite(time_)
                    & tree_allfinite((kinematics, ledger, species_, cell_vectors))
                ),
                AtomisticStepRejectionReason.NONFINITE,
            )
            | _strain_reasons(self.strain_response, force.strain_available, resources)
        )
        successful = reasons == 0
        # A failed initialization never leaves a usable force cache behind.
        force = eqx.tree_at(
            lambda value: value.successful, force, force.successful & successful
        )
        state = AtomisticDynamicsState(
            time=time_,
            step_index=jnp.zeros((), dtype=jnp.int32),
            configuration_revision=revision,
            kinematics=kinematics,
            species=species_,
            cell_vectors=cell_vectors,
            neighborhoods=neighborhoods,
            force=force,
            constraint_lagrange=constraint_lagrange,
            constraint_position_residual=constraint_position_residual,
            constraint_velocity_residual=constraint_velocity_residual,
            controllers=controllers,
            random_key=random_key,
            energy=ledger,
            last_status=jnp.where(
                successful,
                int(AtomisticDynamicsStatus.SUCCESS),
                int(AtomisticDynamicsStatus.INVALID_INITIAL_STATE),
            ).astype(jnp.int32),
            last_rejection_reasons=jnp.asarray(reasons, dtype=jnp.int32),
            thermodynamic_state_index=thermodynamic_index,
            term_states=bound_states,
            thermodynamic_table_identity=thermodynamic.table_identity,
            prepared_dynamics_identity=self.prepared_identity,
        )
        return AtomisticInitialStateEvaluation(state, successful, reasons)

    # ----------------------------------------------------------------- priming

    @checked
    def prime_state(
        self,
        state: AtomisticDynamicsState,
        thermodynamic: PreparedThermodynamicStateTable,
        /,
    ) -> AtomisticPrimeEvaluation:
        """Evaluate every required E/F/strain resource at the accepted state.

        The bundle is refreshed (pure) at the accepted geometry and the force
        cache is rebound to the current revision, controls, epochs and key.
        Time, step, configuration revision, RNG key, controllers, counters and
        adaptive histories are untouched; the energy jump relative to the
        previous ledger is booked as external work.
        """
        self._require_state(state, thermodynamic)
        row = thermodynamic.state_at_replica(state.thermodynamic_state_index)
        neighborhoods = self.refresh_neighborhoods(
            state.neighborhoods,
            state.kinematics.positions,
            cell_vectors=state.cell_vectors,
            image_counts=state.kinematics.image_counts,
        )
        unwrapped = self._unwrapped(state.kinematics, state.cell_vectors)
        evaluation = self.evaluate_configuration(
            state.kinematics.positions,
            neighborhoods,
            unwrapped_positions=unwrapped,
            image_counts=state.kinematics.image_counts,
            species=state.species,
            cell_vectors=state.cell_vectors,
            controls=row.controls,
            term_states=state.term_states,
        )
        ports = self.evaluate_force_ports(
            state.kinematics.positions,
            unwrapped_positions=unwrapped,
            image_counts=state.kinematics.image_counts,
            species=state.species,
            cell_vectors=state.cell_vectors,
            term_states=state.term_states,
        )
        force = self.force_state(
            evaluation,
            neighborhoods,
            state.configuration_revision,
            row.hamiltonian_index,
            row.controls,
            state.random_key,
            term_states=state.term_states,
            ports=ports,
        )
        kinetic = self.kinetic_energy(state.kinematics.momenta)
        terms = self.extended_terms(
            state.controllers, state.cell_vectors, thermodynamic, row
        )
        extended = kinetic + evaluation.energy + terms[0] + terms[1] + terms[2]
        ledger = advance_atomistic_energy_ledger(
            state.energy,
            kinetic,
            evaluation.energy,
            terms,
            thermostat_heat=jnp.zeros_like(kinetic),
            reservoir_heat=jnp.zeros_like(kinetic),
            constraint_work=jnp.zeros_like(kinetic),
            barostat_work=jnp.zeros_like(kinetic),
            external_work=extended - state.energy.extended_energy,
            balanced=False,
            accepted_increment=0,
        )
        resources = _neighborhood_reasons(neighborhoods)
        reasons = (
            resources
            | self._boundary_reasons(state.term_states, ports)
            | _flag(~evaluation.successful, AtomisticStepRejectionReason.POTENTIAL)
            | _flag(
                ~self.binding_valid(state, thermodynamic),
                AtomisticStepRejectionReason.BINDING,
            )
            | _flag(
                ~self._ensemble_valid(row),
                AtomisticStepRejectionReason.THERMODYNAMIC_STATE,
            )
            | _flag(
                ~(self.force_cache_finite(force) & tree_allfinite(ledger)),
                AtomisticStepRejectionReason.NONFINITE,
            )
            | _strain_reasons(self.strain_response, force.strain_available, resources)
        )
        successful = reasons == 0
        candidate = eqx.tree_at(
            lambda value: (value.neighborhoods, value.force, value.energy),
            state,
            (neighborhoods, force, ledger),
        )
        return AtomisticPrimeEvaluation(
            candidate,
            tree_where(successful, candidate, state),
            evaluation,
            successful,
            reasons,
        )

    # ------------------------------------------------------------------ rebase

    @checked
    def rebase_thermodynamic_state(
        self,
        state: AtomisticDynamicsState,
        thermodynamic: PreparedThermodynamicStateTable,
        state_index: ArrayLike,
        /,
        *,
        term_states: AtomisticTermStateBindings | None = None,
    ) -> AtomisticThermodynamicRebaseEvaluation:
        """Move an accepted state to another row with warm controller rebinding.

        Canonical temperature rows rescale particle momenta by
        ``sqrt(T'/T)`` and MTK cell momenta by ``T'/T`` (the equilibrium width
        under the temperature-dependent inertia); chain velocities and GLE
        memory are temperature-reduced and retained.  A changed Hamiltonian
        row, or replacement ``term_states`` of the target row (bias controls or
        history owner selected by a label move), re-evaluates the force cache
        at the unchanged configuration.  The extended-energy change is booked
        as external work.  Term states are never advanced here.
        """
        self._require_state(state, thermodynamic)
        previous_row = thermodynamic.state_at_replica(state.thermodynamic_state_index)
        incoming_force_valid = self.force_cache_valid(state, previous_row.controls)
        next_states = (
            state.term_states
            if term_states is None
            else self.require_term_states(term_states)
        )
        next_index = jnp.asarray(state_index, dtype=jnp.int32).reshape(())
        next_row = thermodynamic.state_at_replica(next_index)
        rescale = previous_row.temperature_mask & next_row.temperature_mask
        ratio = jnp.where(
            rescale,
            next_row.temperature
            / jnp.maximum(
                previous_row.temperature, jnp.finfo(next_row.temperature.dtype).tiny
            ),
            1.0,
        )
        momenta = jnp.where(
            self.system.mobile_mask[:, None],
            state.kinematics.momenta * jnp.sqrt(ratio),
            0.0,
        )
        kinematics = AtomisticKinematics(
            state.kinematics.positions, momenta, state.kinematics.image_counts
        )
        controllers = state.controllers
        if (
            controllers.cell is not None
            and isinstance(self.integrator, MTKNPTPlan)
            and self.integrator.barostat.inertia_temperature is None
        ):
            controllers = eqx.tree_at(
                lambda value: value.cell.momentum,
                controllers,
                controllers.cell.momentum * ratio,
            )

        def refresh_force(_: None) -> AtomisticForceState:
            unwrapped = self._unwrapped(kinematics, state.cell_vectors)
            evaluation = self.evaluate_configuration(
                kinematics.positions,
                state.neighborhoods,
                unwrapped_positions=unwrapped,
                image_counts=kinematics.image_counts,
                species=state.species,
                cell_vectors=state.cell_vectors,
                controls=next_row.controls,
                term_states=next_states,
            )
            ports = self.evaluate_force_ports(
                kinematics.positions,
                unwrapped_positions=unwrapped,
                image_counts=kinematics.image_counts,
                species=state.species,
                cell_vectors=state.cell_vectors,
                term_states=next_states,
            )
            return self.force_state(
                evaluation,
                state.neighborhoods,
                state.configuration_revision,
                next_row.hamiltonian_index,
                next_row.controls,
                state.random_key,
                term_states=next_states,
                ports=ports,
            )

        def retain_force(_: None) -> AtomisticForceState:
            return eqx.tree_at(
                lambda value: value.hamiltonian_index,
                state.force,
                next_row.hamiltonian_index,
            )

        # A changed Hamiltonian row or a term state replaced before the rebase
        # (bias controls, history owner) re-evaluates at the fixed configuration.
        force = jax.lax.cond(
            (previous_row.hamiltonian_index != next_row.hamiltonian_index)
            | jnp.any(state.force.term_revisions != self.term_revisions(next_states)),
            refresh_force,
            retain_force,
            operand=None,
        )
        kinetic = self.kinetic_energy(momenta)
        terms = self.extended_terms(
            controllers, state.cell_vectors, thermodynamic, next_row
        )
        extended = kinetic + force.potential_energy + terms[0] + terms[1] + terms[2]
        ledger = advance_atomistic_energy_ledger(
            state.energy,
            kinetic,
            force.potential_energy,
            terms,
            thermostat_heat=jnp.zeros_like(kinetic),
            reservoir_heat=jnp.zeros_like(kinetic),
            constraint_work=jnp.zeros_like(kinetic),
            barostat_work=jnp.zeros_like(kinetic),
            external_work=extended - state.energy.extended_energy,
            balanced=False,
            accepted_increment=0,
        )
        successful = (
            incoming_force_valid
            & self.binding_valid(state, thermodynamic)
            & self.term_states_successful(next_states)
            & previous_row.valid
            & self._ensemble_valid(next_row)
            & force.successful
            & (force.strain_available if self.strain_response else jnp.asarray(True))
            & self.force_cache_finite(force)
            & tree_allfinite((kinematics, ledger, controllers))
        )
        candidate = eqx.tree_at(
            lambda value: (
                value.kinematics,
                value.force,
                value.controllers,
                value.energy,
                value.thermodynamic_state_index,
                value.term_states,
            ),
            state,
            (kinematics, force, controllers, ledger, next_index, next_states),
        )
        return AtomisticThermodynamicRebaseEvaluation(
            candidate, tree_where(successful, candidate, state), successful
        )

    # --------------------------------------------------------- resource rebind

    @checked
    def rebind_hamiltonian(
        self,
        potential: AbstractPreparedAtomisticHamiltonian,
        state: AtomisticDynamicsState,
        thermodynamic: PreparedThermodynamicStateTable,
        states: tuple[AtomisticThermodynamicStatePlan, ...],
        /,
    ) -> tuple[
        PreparedAtomisticDynamics, PreparedThermodynamicStateTable, AtomisticDynamicsState
    ]:
        """Host lifecycle rebinding of an accepted state to a re-resourced Hamiltonian.

        ``potential`` must carry the same scientific plan with another
        resource binding (for example ``with_neighborhood_bundle`` after
        ``grow_capacities``).  The accepted bundle state is re-expressed with
        ``rebase`` (same epochs and route identities); kinematics, cell,
        controllers, counters, RNG, ledger, step and revision are retained.
        The returned state's force cache is stale under the new binding and
        must be refreshed with ``prime_state`` before stepping.
        """
        self._require_state_host(state, thermodynamic)
        if potential.scientific_id != self.potential.scientific_id:
            raise ValueError("Rebinding requires the same scientific Hamiltonian plan.")
        if (
            PreparedThermodynamicStateTable(self, states).table_id
            != thermodynamic.table_id
        ):
            raise ValueError("Thermodynamic states do not reproduce the bound table.")
        dynamics = AtomisticDynamicsPlan(
            self.system,
            potential,
            self.integrator,
            constraints=self.constraints,
            strain_response=self.strain_response,
            force_ports=self.force_ports,
        ).prepare()
        table = PreparedThermodynamicStateTable(dynamics, states)
        neighborhoods = dynamics.bundle.rebase(state.neighborhoods)
        stale = AtomisticForceState(
            forces=state.force.forces,
            nonconservative_forces=state.force.nonconservative_forces,
            potential_energy=state.force.potential_energy,
            term_energies=state.force.term_energies,
            atom_energy=state.force.atom_energy,
            strain_derivative=state.force.strain_derivative,
            stress=state.force.stress,
            strain_available=jnp.zeros_like(state.force.strain_available),
            configuration_revision=state.force.configuration_revision,
            hamiltonian_index=state.force.hamiltonian_index,
            controls=state.force.controls,
            source_epochs=neighborhoods.epochs,
            graph_epochs=neighborhoods.graph_epochs,
            random_key=state.force.random_key,
            term_revisions=state.force.term_revisions,
            successful=jnp.zeros_like(state.force.successful),
            program_identity=potential.prepared_identity,
            bundle_identity=neighborhoods.bundle_identity,
        )
        rebound = AtomisticDynamicsState(
            time=state.time,
            step_index=state.step_index,
            configuration_revision=state.configuration_revision,
            kinematics=state.kinematics,
            species=state.species,
            cell_vectors=state.cell_vectors,
            neighborhoods=neighborhoods,
            force=stale,
            constraint_lagrange=state.constraint_lagrange,
            constraint_position_residual=state.constraint_position_residual,
            constraint_velocity_residual=state.constraint_velocity_residual,
            controllers=state.controllers,
            random_key=state.random_key,
            energy=state.energy,
            last_status=state.last_status,
            last_rejection_reasons=state.last_rejection_reasons,
            thermodynamic_state_index=state.thermodynamic_state_index,
            term_states=dynamics.require_term_states(state.term_states),
            thermodynamic_table_identity=table.table_identity,
            prepared_dynamics_identity=dynamics.prepared_identity,
        )
        return dynamics, table, rebound

    @checked
    def transition_hamiltonian(
        self,
        potential: AbstractPreparedAtomisticHamiltonian,
        state: AtomisticDynamicsState,
        thermodynamic: PreparedThermodynamicStateTable,
        states: tuple[AtomisticThermodynamicStatePlan, ...],
        /,
        *,
        term_states: AtomisticTermStateBindings | None = None,
    ) -> AtomisticDynamicsHamiltonianTransition:
        """Host transition of an accepted state to a *different* scalar Hamiltonian.

        Unlike ``rebind_hamiltonian`` the scientific identity may change (new
        weights, terms or coefficients).  System, units, integrator, constraints,
        strain response, force ports and the controller layout are retained; the
        new table must keep the same state IDs and intensive variables.  The
        accepted configuration, momenta, time, step, configuration revision, RNG,
        controllers and counters are preserved; forces are re-evaluated at the
        unchanged state and the energy jump is booked as external work.  Term
        states carry over only when the stateful slots and their owner
        identities are unchanged; otherwise ``term_states`` must bind the new
        owners explicitly (no history is silently dropped).  On failure the
        result returns this runtime, table and the untouched state.

        The prepared system identity must be unchanged (exact host refusal; a
        different material is never a transition).  A parameter or content
        revision of the same system that keeps the runtime ``layout_id``
        yields a runtime, table and state with identical treedefs, so every
        compiled transition of the source runtime is reused unchanged; only
        the dynamic identity words and numerical leaves differ.
        """
        self._require_state_host(state, thermodynamic)
        if potential.system.prepared_id != self.system.prepared_id:
            raise ValueError("A Hamiltonian transition keeps the atomistic system.")
        dynamics = AtomisticDynamicsPlan(
            self.system,
            potential,
            self.integrator,
            constraints=self.constraints,
            strain_response=self.strain_response,
            force_ports=self.force_ports,
        ).prepare()
        table = PreparedThermodynamicStateTable(dynamics, states)
        for name in ("state_ids", "ensemble_code", "temperature", "pressure"):
            if not np.array_equal(
                np.asarray(getattr(table, name)), np.asarray(getattr(thermodynamic, name))
            ):
                raise ValueError(f"The transition table changes thermodynamic {name}.")
        if term_states is None:
            if (
                dynamics.term_state_slots != self.term_state_slots
                or dynamics._term_state_owner_ids() != self._term_state_owner_ids()
            ):
                raise ValueError(
                    "Stateful term identities changed; bind their states explicitly."
                )
            bound = state.term_states
        else:
            bound = dynamics.require_term_states(term_states)
        if dynamics.bundle.prepared_id == self.bundle.prepared_id:
            neighborhoods = state.neighborhoods
        else:
            neighborhoods = dynamics.initialize_neighborhoods(
                state.kinematics.positions,
                cell_vectors=state.cell_vectors,
                image_counts=state.kinematics.image_counts,
            )
        old = state.force
        stale = AtomisticForceState(
            forces=old.forces,
            nonconservative_forces=old.nonconservative_forces,
            potential_energy=old.potential_energy,
            term_energies=jnp.zeros(
                (len(_underlying_program(potential).terms),), dtype=old.forces.dtype
            ),
            atom_energy=old.atom_energy,
            strain_derivative=old.strain_derivative,
            stress=old.stress,
            strain_available=jnp.zeros_like(old.strain_available),
            configuration_revision=old.configuration_revision,
            hamiltonian_index=old.hamiltonian_index,
            controls=old.controls,
            source_epochs=neighborhoods.epochs,
            graph_epochs=neighborhoods.graph_epochs,
            random_key=old.random_key,
            term_revisions=dynamics.term_revisions(bound),
            successful=jnp.zeros_like(old.successful),
            program_identity=potential.prepared_identity,
            bundle_identity=neighborhoods.bundle_identity,
        )
        candidate = AtomisticDynamicsState(
            time=state.time,
            step_index=state.step_index,
            configuration_revision=state.configuration_revision,
            kinematics=state.kinematics,
            species=state.species,
            cell_vectors=state.cell_vectors,
            neighborhoods=neighborhoods,
            force=stale,
            constraint_lagrange=state.constraint_lagrange,
            constraint_position_residual=state.constraint_position_residual,
            constraint_velocity_residual=state.constraint_velocity_residual,
            controllers=state.controllers,
            random_key=state.random_key,
            energy=state.energy,
            last_status=state.last_status,
            last_rejection_reasons=state.last_rejection_reasons,
            thermodynamic_state_index=state.thermodynamic_state_index,
            term_states=bound,
            thermodynamic_table_identity=table.table_identity,
            prepared_dynamics_identity=dynamics.prepared_identity,
        )
        primed = dynamics.prime_state(candidate, table)
        lineage = IdentityWords.of(
            {
                "kind": "atomistic-hamiltonian-transition",
                "source_hamiltonian": self.potential.scientific_id,
                "target_hamiltonian": potential.scientific_id,
                "source_dynamics": self.prepared_id,
                "target_dynamics": dynamics.prepared_id,
                "source_table": thermodynamic.table_id,
                "target_table": table.table_id,
            }
        )
        jump = primed.accepted_state.energy.external_work - state.energy.external_work
        if not bool(np.asarray(primed.successful)):
            return AtomisticDynamicsHamiltonianTransition(
                self,
                thermodynamic,
                state,
                jnp.asarray(False),
                jnp.zeros_like(jump),
                primed.rejection_reasons,
                source_dynamics_identity=self.prepared_identity,
                target_dynamics_identity=dynamics.prepared_identity,
                lineage_identity=lineage,
            )
        return AtomisticDynamicsHamiltonianTransition(
            dynamics,
            table,
            primed.accepted_state,
            jnp.asarray(True),
            jump,
            primed.rejection_reasons,
            source_dynamics_identity=self.prepared_identity,
            target_dynamics_identity=dynamics.prepared_identity,
            lineage_identity=lineage,
        )

    # -------------------------------------------------------------------- step

    def step(
        self,
        state: AtomisticDynamicsState,
        thermodynamic: PreparedThermodynamicStateTable,
        /,
    ) -> AtomisticDynamicsState:
        return self.step_detailed(state, thermodynamic).accepted_state

    def _evaluate_at(
        self,
        unwrapped: Array,
        vectors: Array,
        images: Array,
        neighborhoods: AtomisticNeighborhoodBundleState,
        species: Array,
        controls: Array,
        term_states: AtomisticTermStateBindings,
        /,
    ) -> tuple[
        Array,
        Array,
        AtomisticNeighborhoodBundleState,
        Any,
        AtomisticForcePortEvaluation | None,
    ]:
        """One force boundary: conservative program evaluation, then every port."""
        position, wrapped_images = self._wrap(unwrapped, vectors, images)
        candidate = self.refresh_neighborhoods(
            neighborhoods, position, cell_vectors=vectors, image_counts=wrapped_images
        )
        evaluation = self.evaluate_configuration(
            position,
            candidate,
            unwrapped_positions=unwrapped,
            image_counts=wrapped_images,
            species=species,
            cell_vectors=vectors,
            controls=controls,
            term_states=term_states,
        )
        ports = self.evaluate_force_ports(
            position,
            unwrapped_positions=unwrapped,
            image_counts=wrapped_images,
            species=species,
            cell_vectors=vectors,
            term_states=term_states,
        )
        return position, wrapped_images, candidate, evaluation, ports

    # Keep the ordered integrator stages in one candidate-state kernel. Force
    # boundaries, controller work, constraint projection and rejection attribution
    # are coupled; metric-only splitting must not alter their numerical order.
    def _execute(
        self,
        state: AtomisticDynamicsState,
        thermodynamic: PreparedThermodynamicStateTable,
        row: Any,
        /,
    ) -> _Transition:
        """Execute the declared schedule on candidate values only."""
        integrator = self.integrator
        units = self.system.plan.units
        dtype = state.kinematics.positions.dtype
        dt = jnp.asarray(integrator.step_size, dtype=dtype)
        force_rate = units.force_to_momentum_rate
        mobile = self.system.mobile_mask
        inverse_mass = self.system.inverse_masses[:, None]
        dof = thermodynamic.degrees_of_freedom
        thermal = self._thermal_energy(row)
        previous_unwrapped = self._unwrapped(state.kinematics, state.cell_vectors)
        unwrapped = previous_unwrapped
        vectors = state.cell_vectors
        momentum = state.kinematics.momenta
        # Kicks apply the complete force: conservative program forces plus the
        # nonconservative port forces of the same boundary, before projection.
        port_forces = state.force.nonconservative_forces
        forces = state.force.forces + port_forces
        ports: AtomisticForcePortEvaluation | None = None
        boundary_unwrapped = previous_unwrapped
        nonconservative_work = jnp.zeros((), dtype=dtype)
        strain = state.force.strain_derivative
        position = state.kinematics.positions
        images = state.kinematics.image_counts
        neighborhoods = state.neighborhoods
        controllers = state.controllers
        evaluation = None
        position_dirty = False
        zero = jnp.zeros((), dtype=dtype)
        thermostat_heat = zero
        reservoir_heat = zero
        constraint_work = zero
        constraint_lagrange = state.constraint_lagrange
        constraint_position_residual = zero
        constraint_velocity_residual = zero
        constraint_successful = jnp.asarray(True)
        thermostat_successful = jnp.asarray(True)
        barostat_successful = jnp.asarray(True)
        cell_successful = jnp.asarray(True)
        thermostat_phase = 0
        cell_method = isinstance(integrator, _CELL_METHODS)
        mass = self._barostat_mass(row, dof) if cell_method else None
        # A failed mid-step force boundary (resource refusal or potential failure)
        # leaves NaN forces/strain by contract; operators applied after it act on
        # those invalid inputs, so their failure is that boundary's consequence
        # (reported as POTENTIAL/NONFINITE), not an independent operator cause.
        # Operators before any failed boundary keep their own attribution.
        boundary_valid = jnp.asarray(True)

        def attributed(successful: Array) -> Array:
            return successful | ~boundary_valid

        def gradient(cell: AtomisticCellControllerState | None) -> Array:
            if not isinstance(integrator, _CELL_METHODS) or cell is None or mass is None:
                raise RuntimeError("Cell schedule lacks an MTK cell controller.")
            return cell_velocity_gradient(integrator.barostat.chart, cell.momentum, mass)

        for operator, coefficient in zip(
            self.schedule.operators, self.schedule.coefficients, strict=True
        ):
            duration = jnp.asarray(coefficient, dtype=dtype) * dt
            if operator is SplittingOperatorKind.FORCE_KICK:
                if position_dirty:
                    position, images, neighborhoods, evaluation, ports = (
                        self._evaluate_at(
                            unwrapped,
                            vectors,
                            images,
                            state.neighborhoods,
                            state.species,
                            row.controls,
                            state.term_states,
                        )
                    )
                    current_port = (
                        jnp.zeros_like(port_forces) if ports is None else ports.forces
                    )
                    # Trapezoidal work of the port forces along the path segment
                    # between consecutive force boundaries.
                    nonconservative_work = nonconservative_work + 0.5 * jnp.sum(
                        (port_forces + current_port) * (unwrapped - boundary_unwrapped)
                    )
                    port_forces = current_port
                    boundary_unwrapped = unwrapped
                    forces = evaluation.forces + current_port
                    boundary_valid = boundary_valid & evaluation.successful
                    if evaluation.strain_derivative is not None:
                        strain = evaluation.strain_derivative
                    position_dirty = False
                if cell_method:
                    momentum, flow_ok = mtk_momentum_kick(
                        gradient(controllers.cell),
                        dof,
                        momentum,
                        force_rate * forces,
                        duration,
                        mobile,
                    )
                    cell_successful = cell_successful & attributed(flow_ok)
                else:
                    momentum = jnp.where(
                        mobile[:, None], momentum + duration * force_rate * forces, 0.0
                    )
            elif operator is SplittingOperatorKind.DRIFT:
                unwrapped = jnp.where(
                    mobile[:, None],
                    unwrapped + duration * momentum * inverse_mass,
                    unwrapped,
                )
                position_dirty = True
            elif operator is SplittingOperatorKind.CELL_UPDATE:
                if self.system.cell is None:
                    raise RuntimeError("Cell update requires a periodic cell.")
                if not isinstance(integrator, _CELL_METHODS):
                    raise RuntimeError("Cell schedule lacks an MTK cell controller.")
                drift = mtk_drift(
                    gradient(controllers.cell),
                    self.system.cell.origin.astype(dtype),
                    unwrapped,
                    momentum * inverse_mass,
                    vectors,
                    duration,
                    mobile,
                )
                unwrapped = drift.unwrapped_positions
                vectors = drift.cell_vectors
                cell_successful = cell_successful & attributed(
                    drift.successful
                    & cell_admissible(vectors, integrator.barostat.maximum_condition)
                )
                position_dirty = True
            elif operator is SplittingOperatorKind.BAROSTAT:
                if not isinstance(integrator, _CELL_METHODS) or controllers.cell is None:
                    raise RuntimeError("Barostat schedule lacks an MTK cell controller.")
                force = barostat_force(
                    kinetic_tensor(
                        momentum,
                        self.system.inverse_masses,
                        mobile,
                        units.kinetic_to_energy,
                    ),
                    strain,
                    cell_volume(vectors),
                    row.pressure,
                    dof,
                )
                updated = controllers.cell.momentum + duration * project_cell_force(
                    integrator.barostat.chart, force
                )
                barostat_successful = barostat_successful & attributed(
                    jnp.all(jnp.isfinite(updated))
                )
                controllers = eqx.tree_at(
                    lambda value: value.cell.momentum, controllers, updated
                )
            elif operator is SplittingOperatorKind.CELL_THERMOSTAT:
                if (
                    not isinstance(integrator, MTKNPTPlan)
                    or controllers.cell is None
                    or controllers.cell_chain is None
                    or mass is None
                ):
                    raise RuntimeError("Cell thermostat requires an MTK NPT cell chain.")
                chart = integrator.barostat.chart
                cell_kinetic = cell_kinetic_energy(chart, controllers.cell.momentum, mass)
                chain = propagate_nose_hoover_chain(
                    integrator.cell_chain,
                    controllers.cell_chain,
                    2.0 * cell_kinetic,
                    chart.dimension,
                    thermal,
                    duration,
                )
                controllers = eqx.tree_at(
                    lambda value: (value.cell_chain, value.cell.momentum),
                    controllers,
                    (chain.state, controllers.cell.momentum * chain.scale),
                )
                thermostat_successful = thermostat_successful & attributed(
                    chain.successful
                )
            elif operator is SplittingOperatorKind.THERMOSTAT:
                kinetic_before = self.kinetic_energy(momentum)
                if isinstance(integrator, BAOABLangevinPlan):
                    ornstein = apply_baoab_ornstein_uhlenbeck(
                        integrator,
                        state.random_key,
                        self.system.plan.particle_ids,
                        momentum,
                        self.system.plan.masses,
                        mobile,
                        state.step_index,
                        row.temperature,
                        boltzmann_constant=units.boltzmann_constant,
                        kinetic_to_energy=units.kinetic_to_energy,
                    )
                    momentum = ornstein.momenta
                    thermostat_successful = thermostat_successful & attributed(
                        ornstein.successful
                    )
                    reservoir_heat = reservoir_heat + ornstein.heat
                elif isinstance(integrator, (NoseHooverChainPlan, MTKNPTPlan)):
                    parameters: NoseHooverChainParameters = (
                        integrator.chain
                        if isinstance(integrator, NoseHooverChainPlan)
                        else integrator.particle_chain
                    )
                    if controllers.particle_chain is None:
                        raise RuntimeError(
                            "Thermostat schedule lacks its particle chain."
                        )
                    chain = propagate_nose_hoover_chain(
                        parameters,
                        controllers.particle_chain,
                        2.0 * kinetic_before,
                        dof,
                        thermal,
                        duration,
                    )
                    momentum = jnp.where(mobile[:, None], momentum * chain.scale, 0.0)
                    controllers = eqx.tree_at(
                        lambda value: value.particle_chain, controllers, chain.state
                    )
                    thermostat_successful = thermostat_successful & attributed(
                        chain.successful
                    )
                elif isinstance(integrator, StochasticVelocityRescalingPlan):
                    scale = stochastic_velocity_rescaling_scale(
                        integrator,
                        state.random_key,
                        state.step_index,
                        kinetic_before,
                        dof,
                        thermal,
                        duration,
                        operator_id=(
                            CSVR_FIRST_OPERATOR_ID
                            if thermostat_phase == 0
                            else CSVR_SECOND_OPERATOR_ID
                        ),
                    )
                    momentum = jnp.where(mobile[:, None], momentum * scale.scale, 0.0)
                    thermostat_successful = thermostat_successful & attributed(
                        scale.successful
                    )
                    reservoir_heat = (
                        reservoir_heat + self.kinetic_energy(momentum) - kinetic_before
                    )
                elif isinstance(integrator, GeneralizedLangevinPlan):
                    if controllers.gle is None:
                        raise RuntimeError("GLE schedule lacks its memory state.")
                    langevin = apply_generalized_langevin(
                        integrator,
                        state.random_key,
                        self.system.plan.particle_ids,
                        momentum,
                        controllers.gle,
                        self.system.plan.masses,
                        mobile,
                        state.step_index,
                        row.temperature,
                        phase=thermostat_phase,
                        boltzmann_constant=units.boltzmann_constant,
                        kinetic_to_energy=units.kinetic_to_energy,
                    )
                    momentum = langevin.momenta
                    controllers = eqx.tree_at(
                        lambda value: value.gle, controllers, langevin.memory
                    )
                    thermostat_successful = thermostat_successful & attributed(
                        langevin.successful
                    )
                    reservoir_heat = reservoir_heat + langevin.heat
                elif isinstance(integrator, VelocityRescalingPlan):
                    due = (state.step_index + 1) % integrator.interval == 0
                    scale = deterministic_velocity_rescaling_scale(
                        kinetic_before, dof, thermal
                    )
                    factor = jnp.where(due, scale.scale, 1.0)
                    momentum = jnp.where(mobile[:, None], momentum * factor, 0.0)
                    thermostat_successful = thermostat_successful & attributed(
                        ~due | scale.successful
                    )
                    controllers = eqx.tree_at(
                        lambda value: value.rescale_events,
                        controllers,
                        controllers.rescale_events + due.astype(jnp.int32),
                    )
                    reservoir_heat = (
                        reservoir_heat + self.kinetic_energy(momentum) - kinetic_before
                    )
                else:
                    raise RuntimeError(
                        f"{type(integrator).__name__} declares no particle thermostat."
                    )
                thermostat_heat = (
                    thermostat_heat + self.kinetic_energy(momentum) - kinetic_before
                )
                thermostat_phase += 1
            elif operator is SplittingOperatorKind.POSITION_CONSTRAINT:
                if self.constraints is None:
                    raise RuntimeError("Position-constraint schedule lacks constraints.")
                kinetic_before = self.kinetic_energy(momentum)
                projection = self.constraints.project_positions(
                    previous_unwrapped, unwrapped, momentum
                )
                unwrapped = projection.positions
                momentum = projection.momenta
                constraint_lagrange = projection.multipliers
                constraint_position_residual = projection.position_residual
                constraint_velocity_residual = projection.velocity_residual
                constraint_successful = constraint_successful & attributed(
                    projection.successful
                )
                constraint_work = (
                    constraint_work + self.kinetic_energy(momentum) - kinetic_before
                )
                position_dirty = True
            elif operator is SplittingOperatorKind.MOMENTUM_CONSTRAINT:
                if self.constraints is None:
                    raise RuntimeError("Momentum-constraint schedule lacks constraints.")
                kinetic_before = self.kinetic_energy(momentum)
                momentum, velocity_residual = self.constraints.project_momenta(
                    unwrapped, momentum
                )
                constraint_work = (
                    constraint_work + self.kinetic_energy(momentum) - kinetic_before
                )
                constraint_velocity_residual = jnp.maximum(
                    constraint_velocity_residual, velocity_residual
                )
                constraint_successful = constraint_successful & attributed(
                    velocity_residual <= self.constraints.plan.tolerance
                )
            else:
                raise RuntimeError(
                    f"Dynamics schedule cannot execute {operator.value!r}."
                )
        if position_dirty or evaluation is None:
            position, images, neighborhoods, evaluation, ports = self._evaluate_at(
                unwrapped,
                vectors,
                images,
                state.neighborhoods,
                state.species,
                row.controls,
                state.term_states,
            )
            current_port = jnp.zeros_like(port_forces) if ports is None else ports.forces
            nonconservative_work = nonconservative_work + 0.5 * jnp.sum(
                (port_forces + current_port) * (unwrapped - boundary_unwrapped)
            )
        return _Transition(
            positions=position,
            unwrapped=unwrapped,
            images=images,
            momenta=momentum,
            cell_vectors=vectors,
            neighborhoods=neighborhoods,
            evaluation=evaluation,
            ports=ports,
            nonconservative_work=nonconservative_work,
            controllers=controllers,
            constraint_lagrange=constraint_lagrange,
            constraint_position_residual=constraint_position_residual,
            constraint_velocity_residual=constraint_velocity_residual,
            thermostat_heat=thermostat_heat,
            reservoir_heat=reservoir_heat,
            constraint_work=constraint_work,
            constraint_successful=constraint_successful,
            thermostat_successful=thermostat_successful,
            barostat_successful=barostat_successful,
            cell_successful=cell_successful,
        )

    @checked
    def step_detailed(
        self,
        state: AtomisticDynamicsState,
        thermodynamic: PreparedThermodynamicStateTable,
        /,
    ) -> AtomisticStepEvaluation:
        """Execute one declared schedule and publish exactly one transaction."""
        self._require_state(state, thermodynamic)
        row = thermodynamic.state_at_replica(state.thermodynamic_state_index)
        force_valid = self.force_cache_valid(state, row.controls)
        transition = self._execute(state, thermodynamic, row)
        evaluation = transition.evaluation
        revision = state.configuration_revision + 1
        cell_changed = isinstance(self.integrator, _CELL_METHODS)
        controllers = transition.controllers
        if cell_changed:
            controllers = eqx.tree_at(
                lambda value: value.cell_epoch, controllers, controllers.cell_epoch + 1
            )
        kinematics = AtomisticKinematics(
            transition.positions, transition.momenta, transition.images
        )
        force = self.force_state(
            evaluation,
            transition.neighborhoods,
            revision,
            row.hamiltonian_index,
            row.controls,
            state.random_key,
            term_states=state.term_states,
            ports=transition.ports,
        )
        kinetic = self.kinetic_energy(transition.momenta)
        terms = self.extended_terms(
            controllers, transition.cell_vectors, thermodynamic, row
        )
        previous_barostat = (
            state.energy.cell_kinetic_energy + state.energy.pressure_volume
        )
        ledger = advance_atomistic_energy_ledger(
            state.energy,
            kinetic,
            evaluation.energy,
            terms,
            thermostat_heat=transition.thermostat_heat,
            reservoir_heat=transition.reservoir_heat,
            constraint_work=transition.constraint_work,
            barostat_work=terms[0] + terms[1] - previous_barostat,
            external_work=transition.nonconservative_work,
            balanced=True,
            accepted_increment=1,
        )
        candidate_finite = tree_allfinite(
            (
                kinematics,
                ledger,
                transition.cell_vectors,
                controllers,
                evaluation.forces,
                force.nonconservative_forces,
            )
        )
        resources = _neighborhood_reasons(transition.neighborhoods)
        reasons = (
            resources
            | self._boundary_reasons(state.term_states, transition.ports)
            | _flag(~evaluation.successful, AtomisticStepRejectionReason.POTENTIAL)
            | _flag(
                ~transition.constraint_successful, AtomisticStepRejectionReason.CONSTRAINT
            )
            | _flag(
                ~transition.thermostat_successful, AtomisticStepRejectionReason.THERMOSTAT
            )
            | _flag(
                ~transition.barostat_successful, AtomisticStepRejectionReason.BAROSTAT
            )
            | _flag(~transition.cell_successful, AtomisticStepRejectionReason.CELL)
            | _flag(
                ~self._ensemble_valid(row),
                AtomisticStepRejectionReason.THERMODYNAMIC_STATE,
            )
            | _flag(~candidate_finite, AtomisticStepRejectionReason.NONFINITE)
            | _flag(~force_valid, AtomisticStepRejectionReason.STALE_FORCE)
            | _flag(
                ~self.binding_valid(state, thermodynamic),
                AtomisticStepRejectionReason.BINDING,
            )
            | _strain_reasons(self.strain_response, force.strain_available, resources)
        )
        successful = reasons == 0
        status = jnp.where(
            successful,
            int(AtomisticDynamicsStatus.SUCCESS),
            jnp.where(
                candidate_finite,
                int(AtomisticDynamicsStatus.REJECTED),
                int(AtomisticDynamicsStatus.NONFINITE),
            ),
        ).astype(jnp.int32)
        candidate = AtomisticDynamicsState(
            time=state.time
            + jnp.asarray(self.integrator.step_size, dtype=state.time.dtype),
            step_index=state.step_index + 1,
            configuration_revision=revision,
            kinematics=kinematics,
            species=state.species,
            cell_vectors=transition.cell_vectors,
            neighborhoods=transition.neighborhoods,
            force=force,
            constraint_lagrange=transition.constraint_lagrange,
            constraint_position_residual=transition.constraint_position_residual,
            constraint_velocity_residual=transition.constraint_velocity_residual,
            controllers=controllers,
            random_key=state.random_key,
            energy=ledger,
            last_status=status,
            last_rejection_reasons=reasons,
            thermodynamic_state_index=state.thermodynamic_state_index,
            term_states=state.term_states,
            thermodynamic_table_identity=state.thermodynamic_table_identity,
            prepared_dynamics_identity=state.prepared_dynamics_identity,
        )
        accepted = tree_where(successful, candidate, state)
        diagnostics = self.diagnostics(candidate, thermodynamic, successful, reasons)
        work = neighborhood_route_work(transition.neighborhoods) + jnp.asarray(
            len(self.potential.terms), dtype=jnp.int32
        )
        return AtomisticStepEvaluation(
            candidate_state=candidate,
            accepted_state=accepted,
            potential=evaluation,
            diagnostics=diagnostics,
            successful=successful,
            residual=ledger.last_relative_energy_change,
            work=work,
            nonconservative_work=transition.nonconservative_work,
            thermodynamic_state_index=state.thermodynamic_state_index,
            rejection_reasons=reasons,
        )

    # -------------------------------------------------------------- diagnostics

    def _view_positions(
        self,
        neighborhoods: AtomisticNeighborhoodBundleState,
        view: AtomisticNeighborhoodView,
        /,
    ) -> Array:
        geometry = neighborhoods.geometry
        if (
            view.entity is AtomisticSiteDomain.INTERACTION_SITES
            and not self.bundle.site_alias
        ):
            if geometry.site_positions is None:
                raise RuntimeError("Site role lacks realized site geometry.")
            return geometry.site_positions
        return geometry.positions

    def pressure_tensor(self, state: AtomisticDynamicsState, /) -> tuple[Array, Array]:
        """Instantaneous internal pressure tensor and volume.

        ``P = (sum_i p_i p_i^T / m_i) / V - sym(dE/d strain) / V``: kinetic
        momentum flux minus the tension-positive configurational stress of the
        cached homogeneous-strain response.  NaN when the system has no full
        periodic cell, the strain response is unavailable or nonconservative
        force ports act (their momentum flux has no strain-derivative form);
        it never falls back to a finite position moment.
        """
        dtype = state.kinematics.momenta.dtype
        if not self.strain_response or self.force_ports:
            nan = jnp.full((3, 3), jnp.nan, dtype=dtype)
            return nan, jnp.asarray(jnp.nan, dtype=dtype)
        volume = cell_volume(state.cell_vectors)
        kinetic = kinetic_tensor(
            state.kinematics.momenta,
            self.system.inverse_masses,
            self.system.mobile_mask,
            self.system.plan.units.kinetic_to_energy,
        )
        strain = state.force.strain_derivative
        tensor = (kinetic - 0.5 * (strain + strain.T)) / volume
        available = state.force.strain_available
        return jnp.where(available, tensor, jnp.nan), volume

    @checked
    def diagnostics(
        self,
        state: AtomisticDynamicsState,
        thermodynamic: PreparedThermodynamicStateTable,
        successful: Array | None = None,
        rejection_reasons: Array | None = None,
        /,
    ) -> AtomisticDynamicsDiagnostics:
        """Instantaneous diagnostics; a foreign state or table reports failure.

        Only the table layout is checked statically; when ``successful`` is
        not supplied, the state's status is combined with the traced runtime
        and table binding (``binding_valid``).
        """
        thermodynamic.require_dynamics_layout(self)
        mass = self.system.plan.masses
        mobile = self.system.mobile_mask
        momentum = jnp.sum(
            jnp.where(mobile[:, None], state.kinematics.momenta, 0.0), axis=0
        )
        unwrapped = self._unwrapped(state.kinematics, state.cell_vectors)
        mass_sum = jnp.sum(jnp.where(mobile, mass, 0.0))
        center = (
            jnp.sum(jnp.where(mobile[:, None], mass[:, None] * unwrapped, 0.0), axis=0)
            / mass_sum
        )
        angular = jnp.sum(jnp.cross(unwrapped - center, state.kinematics.momenta), axis=0)
        force = jnp.where(self.system.active_mask[:, None], state.force.forces, 0.0)
        net_force = jnp.sum(force, axis=0)
        net_torque = jnp.sum(jnp.cross(unwrapped - center, force), axis=0)
        dtype = state.kinematics.positions.dtype
        neighborhoods = state.neighborhoods
        vectors = neighborhoods.geometry.cell_vectors[0]
        minimum_distance = jnp.asarray(jnp.inf, dtype=dtype)
        cutoff_margin = jnp.asarray(jnp.inf, dtype=dtype)
        for view in neighborhoods.views:
            displacement = view.displacement(
                self._view_positions(neighborhoods, view), vectors
            )
            distances = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
            minimum_distance = jnp.minimum(
                minimum_distance, jnp.min(jnp.where(view.mask, distances, jnp.inf))
            )
            if view.cutoff is not None:
                cutoff_margin = jnp.minimum(
                    cutoff_margin,
                    jnp.min(
                        jnp.where(view.mask, jnp.abs(distances - view.cutoff), jnp.inf)
                    ),
                )
        rebuilt = jnp.any(neighborhoods.rebuilt)
        rebuild_count = jnp.zeros((), dtype=jnp.int32)
        certificate = jnp.asarray(jnp.inf, dtype=dtype)
        for source in neighborhoods.sources:
            rebuild_count = rebuild_count + jnp.sum(
                source.rebuild_counts, dtype=jnp.int32
            )
            if isinstance(source, SegmentedImageVerletState):
                certificate = jnp.minimum(
                    certificate, jnp.min(source.certificate_margin).astype(dtype)
                )
        kinetic_energy = state.energy.kinetic_energy
        temperature = (
            2.0
            * kinetic_energy
            / (
                jnp.asarray(thermodynamic.degrees_of_freedom, dtype=kinetic_energy.dtype)
                * self.system.plan.units.boltzmann_constant
            )
        )
        tensor, volume = self.pressure_tensor(state)
        success = (
            (state.last_status == int(AtomisticDynamicsStatus.SUCCESS))
            & self.binding_valid(state, thermodynamic)
            if successful is None
            else jnp.asarray(successful, dtype=jnp.bool_)
        )
        reasons = (
            state.last_rejection_reasons
            if rejection_reasons is None
            else jnp.asarray(rejection_reasons, dtype=jnp.int32)
        )
        return AtomisticDynamicsDiagnostics(
            kinetic_energy=state.energy.kinetic_energy,
            potential_energy=state.energy.potential_energy,
            total_energy=state.energy.total_energy,
            extended_energy=state.energy.extended_energy,
            temperature=temperature,
            pressure=jnp.trace(tensor) / 3.0,
            pressure_tensor=tensor,
            volume=volume,
            total_linear_momentum=momentum,
            total_angular_momentum=angular,
            net_internal_force=net_force,
            net_internal_torque=net_torque,
            minimum_pair_distance=minimum_distance,
            cutoff_margin=cutoff_margin,
            neighborhood_rebuilt=rebuilt,
            neighborhood_rebuild_count=rebuild_count,
            neighborhood_certificate_margin=certificate,
            constraint_residual=jnp.maximum(
                state.constraint_position_residual,
                state.constraint_velocity_residual,
            ),
            energy=state.energy,
            successful=success,
            rejection_reasons=reasons,
        )


_REPAIRABLE = CAPACITY_REJECTIONS | int(
    AtomisticStepRejectionReason.POTENTIAL | AtomisticStepRejectionReason.NONFINITE
)


def retry_atomistic_step_with_capacity(
    dynamics: PreparedAtomisticDynamics,
    state: AtomisticDynamicsState,
    thermodynamic: PreparedThermodynamicStateTable,
    states: tuple[AtomisticThermodynamicStatePlan, ...],
    /,
) -> tuple[
    PreparedAtomisticDynamics, PreparedThermodynamicStateTable, AtomisticStepEvaluation
]:
    """Attempt one step, growing declared bundle capacities only when required.

    A rejection carrying a source/route capacity reason (possibly with the
    potential/nonfinite flags that overflow causes) and no other cause is
    replayed: the bundle selects its next covering declared capacities, the
    Hamiltonian is rebound to them, the accepted state is rebased and
    re-primed (no step, time, revision, RNG or controller event), and the same
    physical attempt is retried at the same step and stochastic address.  Any
    scientific, domain, stale, thermodynamic or constraint cause is returned
    unchanged.  An exhausted declared ladder raises.
    """
    if not isinstance(dynamics, PreparedAtomisticDynamics):
        raise TypeError("dynamics must be PreparedAtomisticDynamics.")
    evaluation = dynamics.step_detailed(state, thermodynamic)
    while not bool(np.asarray(evaluation.successful)):
        reasons = int(np.asarray(evaluation.rejection_reasons))
        if not reasons & CAPACITY_REJECTIONS or reasons & ~_REPAIRABLE:
            return dynamics, thermodynamic, evaluation
        grown = dynamics.bundle.grow_capacities(evaluation.candidate_state.neighborhoods)
        dynamics, thermodynamic, rebound = dynamics.rebind_hamiltonian(
            dynamics.potential.with_neighborhood_bundle(grown),
            state,
            thermodynamic,
            states,
        )
        primed = dynamics.prime_state(rebound, thermodynamic)
        if not bool(np.asarray(primed.successful)):
            raise RuntimeError(
                "The accepted state could not be re-primed under grown capacities; "
                f"rejection reasons {int(np.asarray(primed.rejection_reasons))}."
            )
        state = primed.accepted_state
        evaluation = dynamics.step_detailed(state, thermodynamic)
    return dynamics, thermodynamic, evaluation


__all__ = [
    "AtomisticControllerState",
    "AtomisticDynamicsDiagnostics",
    "AtomisticDynamicsPlan",
    "AtomisticDynamicsState",
    "AtomisticDynamicsStatus",
    "AtomisticEnergyLedgerState",
    "AtomisticForceState",
    "AtomisticDynamicsHamiltonianTransition",
    "AtomisticIntegratorPlan",
    "AtomisticInitialStateEvaluation",
    "AtomisticKinematics",
    "AtomisticPotentialEnergyEvaluation",
    "AtomisticPrimeEvaluation",
    "AtomisticStepEvaluation",
    "AtomisticStepRejectionReason",
    "AtomisticThermodynamicRebaseEvaluation",
    "CAPACITY_REJECTIONS",
    "PreparedAtomisticDynamics",
    "VelocityVerletPlan",
    "advance_atomistic_energy_ledger",
    "initial_atomistic_energy_ledger",
    "neighborhood_route_work",
    "retry_atomistic_step_with_capacity",
]
