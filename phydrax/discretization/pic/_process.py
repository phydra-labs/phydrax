#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Particle processes and recorders composed into a PIC step.

A process acts on the species states of one step at a declared stage:

- ``"momentum"`` processes run after the push and before the drift; they may
  change only proper velocities (collisions, radiation reaction);
- ``"creation"`` processes run after the momentum stage and before the drift;
  they may change proper velocities and create particles at step-start
  positions (strong-field QED emission and pair creation) but must preserve
  the deposited charge pointwise, which the PIC runtime verifies by
  redeposition; created particles drift and deposit current in the same step;
- ``"population"`` processes run after the field advance on the end-of-step
  species; they may create/deactivate particles and change charge numbers
  (ionization) but must preserve the deposited charge pointwise, which the
  PIC runtime verifies by redeposition. A process that declares
  ``redistributes_charge`` (particle merging/splitting) must instead conserve
  total charge; the runtime then Gauss-projects the field onto the
  redeposited charge through the solver's `PICGaussProjection` capability.

A process may own state the runtime carries in the PIC state, checkpoints as
its own restart component, and shifts with a moving window
(`AbstractPICProcess.initialize_state`, `shift_frame`); it may request fields
at positions of its own (`field_probe`), which the runtime gathers at the
step-start field and supplies as `PICProcessContext.probe`.

Processes that dissipate or emit radiation declare a `RadiationOwnership`
claim; a coupled run refuses overlapping claims at construction.
"""

from __future__ import annotations

import abc
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable, TypeAlias

import equinox as eqx
from jax import Array

from ..._physical import RelativityScaleContract
from ..._strict import StrictModule
from ...typing import PRNGKey
from ..particle import (
    ParticleAllocationRequest,
    ParticleAllocationResult,
    ParticlePopulationPlan,
    ParticlePopulationState,
)
from ._charge_state import PICSpeciesPlan, PICSpeciesState
from ._method import RelativisticPushPlan


RadiationOwnership: TypeAlias = Literal[
    "resolved-field", "subgrid-reaction", "diagnostic-only"
]
PICProcessStage: TypeAlias = Literal["momentum", "creation", "population"]


class PICFieldProbe(StrictModule):
    """Positions at which a process needs the step-start fields.

    The runtime gathers through the transfer route of species ``species``;
    the probe count must be a multiple of that species' capacity.
    """

    position: Array
    active: Array
    species: int = eqx.field(static=True)


class PICFieldProbeSample(StrictModule):
    """Total (grid plus external) fields at a process's probe positions."""

    electric: Array
    magnetic: Array
    successful: Array


class PICProcessContext(StrictModule):
    """Step data visible to a process.

    ``electric``/``magnetic`` are the total fields gathered for the push at the
    step-start positions of each species; ``key`` is the process's own
    stateless substream for this step, or ``None`` for deterministic processes.

    For processes declaring ``requires_field_derivatives`` the runtime also
    supplies, per species, the spatial gradients ``∂F_i/∂x_j`` (``[N, 3, 3]``,
    order-one gather plus exact external-field derivatives) and the time
    derivatives ``∂_t F`` (``[N, 3]``) of the same total fields; grid-field
    time derivatives come from the staggered field history (the previous
    accepted field state gathered at the current positions).
    ``grid_cutoff_frequency`` is the highest angular frequency the field
    discretization resolves, ``min(π c / min Δx, π / Δt)``; ``None`` outside a
    field grid. ``state`` is the process's own state (``None`` for stateless
    processes) and ``probe`` the fields at its `field_probe` positions.
    ``pusher`` is the run's `RelativisticPushPlan` and
    ``step_start_proper_velocity`` each species' proper velocities before this
    step's push (the push maps them to ``species`` in the momentum and
    creation stages), so a process can integrate quantities carried with the
    particles (spin) consistently with the push. ``grid_velocity`` is the
    velocity of a Galilean field grid in which positions are expressed (zero
    for lab-fixed grids): a process moving its own particles displaces them by
    ``(v − grid_velocity) Δt``. ``allocator`` is the run's allocation route for
    created particles (`allocate_particles`); ``None`` allocates through the
    population plan.
    """

    __strict_contract__ = True

    species: tuple[PICSpeciesState, ...]
    electric: tuple[Array, ...]
    magnetic: tuple[Array, ...]
    time: Array
    step_size: Array
    step_index: Array
    key: PRNGKey | None
    electric_gradient: tuple[Array, ...] | None = None
    magnetic_gradient: tuple[Array, ...] | None = None
    electric_rate: tuple[Array, ...] | None = None
    magnetic_rate: tuple[Array, ...] | None = None
    grid_cutoff_frequency: Array | None = None
    state: Any = None
    probe: PICFieldProbeSample | None = None
    pusher: RelativisticPushPlan | None = None
    step_start_proper_velocity: tuple[Array, ...] | None = None
    grid_velocity: Array | None = None
    allocator: AbstractPICParticleAllocator | None = None


class PICProcessRadiation(StrictModule):
    """Energy a process hands to radiation the field does not resolve.

    ``radiated_energy`` is this step's total over all bound particles.
    ``maximum_critical_frequency`` is the largest critical angular frequency of
    the emitting particles; ``minimum_scale_separation`` the smallest ratio of
    an emitting particle's critical frequency to the grid cutoff (``inf``
    without a grid). ``scale_separated`` is false when an emitting particle
    radiates at frequencies the grid resolves, so a subgrid claim would
    double-count field-resolved radiation. Processes that create particles or
    close their kinematics against the field also report the rest energy of
    the created particles (``created_rest_energy``) and the energy the field
    supplied (``field_exchange_energy``); ``None`` means zero.
    """

    radiated_energy: Array
    maximum_critical_frequency: Array
    minimum_scale_separation: Array
    scale_separated: Array
    field_exchange_energy: Array | None = None
    created_rest_energy: Array | None = None


class PICProcessLedger(StrictModule):
    """Conservation ledger reported by one process application.

    ``radiation`` is present exactly for processes that emit radiation.
    """

    event_count: Array
    charge_defect: Array
    momentum_defect: Array
    energy_defect: Array
    successful: Array
    process_id: str = eqx.field(static=True)
    radiation: PICProcessRadiation | None = None


class PICProcessResult(StrictModule):
    """Accepted species, conservation ledger, optional evidence and process state."""

    species: tuple[PICSpeciesState, ...]
    ledger: PICProcessLedger
    evidence: Any = None
    state: Any = None


class AbstractPICProcess(StrictModule):
    """One particle process bound to explicit species indices."""

    process_id: eqx.AbstractVar[str]
    stage: eqx.AbstractVar[PICProcessStage]
    stochastic: eqx.AbstractVar[bool]
    radiation_ownership: eqx.AbstractVar[RadiationOwnership | None]
    species_indices: eqx.AbstractVar[tuple[int, ...]]
    # Population processes that move charge between grid locations while
    # conserving its total (resampling) set this; they require a field solver
    # implementing `PICGaussProjection`.
    redistributes_charge: ClassVar[bool] = False
    # Processes owning state carried in the PIC state (`initialize_state`
    # returns non-``None``) set this.
    stateful: ClassVar[bool] = False

    @property
    def requires_field_derivatives(self) -> bool:
        """Whether the force depends on field derivatives (full Landau–Lifshitz).

        The runtime then keeps a staggered field history and supplies field
        gradients and time derivatives in the momentum-stage context.
        """
        return False

    def initialize_state(
        self,
        species: tuple[PICSpeciesPlan, ...],
        states: tuple[PICSpeciesState, ...],
        /,
    ) -> Any:
        """Initial process state carried in the PIC state (``None``: stateless).

        A stateful process returns its updated state as `PICProcessResult.state`;
        the runtime checkpoints it as the restart component ``process/{index}``
        owned by ``process_id``.
        """
        del species, states
        return None

    def field_probe(self, state: Any, /) -> PICFieldProbe | None:
        """Positions where the process needs the step-start fields (default none)."""
        del state
        return None

    def shift_frame(self, state: Any, axis: int, distance: float, /) -> Any:
        """Process state after the particle frame moved by ``distance`` along ``axis``.

        Position-carrying states translate by ``−distance``; others are returned.
        """
        del axis, distance
        return state

    def validate_run(
        self,
        species: tuple[PICSpeciesPlan, ...],
        relativity: RelativityScaleContract,
        /,
    ) -> None:
        """Refuse, at run construction, species or pusher units it cannot act on.

        Processes without such constraints accept every run.
        """
        del species, relativity

    @abc.abstractmethod
    def apply(
        self,
        species: tuple[PICSpeciesPlan, ...],
        context: PICProcessContext,
        /,
    ) -> PICProcessResult:
        """Return accepted species states (unchanged on failure) and the ledger."""
        raise NotImplementedError


class AbstractPICParticleAllocator(StrictModule):
    """Allocation route of the particles a process creates.

    A single-device run allocates through the population plan itself; a
    decomposed run supplies an allocator that keeps created particles in the
    device's slot block and assigns decomposition-independent identities.
    """

    @abc.abstractmethod
    def allocate(
        self,
        plan: ParticlePopulationPlan,
        state: ParticlePopulationState,
        request: ParticleAllocationRequest,
        positions: Array,
        /,
    ) -> ParticleAllocationResult:
        """Allocate ``request``; ``positions[W, d]`` are the grid positions of
        each request row's creating event (its parent's position)."""
        raise NotImplementedError


def allocate_particles(
    context: PICProcessContext,
    plan: ParticlePopulationPlan,
    state: ParticlePopulationState,
    request: ParticleAllocationRequest,
    positions: Array,
    /,
) -> ParticleAllocationResult:
    """Allocate a process's created particles through the run's allocation route.

    ``positions[W, d]`` are the grid positions of each request row's creating
    event: the emitter, decaying photon, merged packet, split particle, or
    ionized ion. Without a context allocator this is ``plan.allocate``.
    """
    allocator = context.allocator
    if allocator is None:
        return plan.allocate(state, request)
    return allocator.allocate(plan, state, request, positions)


class PICProcessBank(StrictModule):
    """A process-owned particle population with positions (a photon bank).

    ``position`` holds the bank slots' grid coordinates and ``leaves`` the
    further per-slot arrays that move with the bank's particles.
    """

    population: ParticlePopulationState
    position: Array
    leaves: tuple[Array, ...]


class PICProcessStatePartition(StrictModule):
    """A process state split by how a slot decomposition distributes it.

    ``companions[k]`` are per-slot leaves aligned with the slots of run species
    ``companion_species[k]`` and move with those particles; ``banks`` are
    process-owned populations decomposed by their own positions; ``totals``
    are additive accumulators (each device adds its own increments, which are
    summed); ``shared`` are values every device holds identically.
    """

    companions: tuple[tuple[Array, ...], ...]
    banks: tuple[PICProcessBank, ...]
    totals: tuple[Array, ...]
    shared: tuple[Array, ...]
    companion_species: tuple[int, ...] = eqx.field(static=True)


@runtime_checkable
class PICDistributedProcess(Protocol):
    """A creation, population, or stateful process executable per device.

    ``localize(species, parts)`` is the process bound to one device's slot
    blocks (``species`` are the block-capacity species plans, every capacity
    divided by ``parts``); it allocates through `allocate_particles`.
    ``partition_state``/``assemble_state`` split and rebuild the process state
    (`PICProcessStatePartition`); ``bank_plans`` are the global population
    plans of ``partition_state(...).banks``. ``combine(ledger, evidence)``
    receives one device's ledger and evidence per leading row (every leaf
    stacked over the ``parts`` devices in slot-block order) and returns the
    run's ledger and evidence.
    """

    def localize(
        self, species: tuple[PICSpeciesPlan, ...], parts: int, /
    ) -> AbstractPICProcess: ...

    def bank_plans(self) -> tuple[ParticlePopulationPlan, ...]: ...

    def partition_state(self, state: Any, /) -> PICProcessStatePartition: ...

    def assemble_state(self, partition: PICProcessStatePartition, /) -> Any: ...

    def combine(
        self, ledger: PICProcessLedger, evidence: Any, /
    ) -> tuple[PICProcessLedger, Any]: ...


class PICParticleExchangeResult(StrictModule):
    """Particles and process states after one particle exchange between devices."""

    species: tuple[PICSpeciesState, ...]
    processes: tuple[Any, ...]
    evidence: Any
    successful: Array


class AbstractPICParticleExecutor(StrictModule):
    """Executes the particle side of a PIC step over a device decomposition.

    ``apply_process`` runs process ``index`` of the run (per device for
    `PICDistributedProcess` processes); ``exchange`` moves particles, with
    their slot-aligned process state, to their owning devices. The runtime
    exchanges before the population stage, so population processes see every
    particle on its owner.
    """

    @abc.abstractmethod
    def apply_process(
        self,
        index: int,
        process: AbstractPICProcess,
        species: tuple[PICSpeciesPlan, ...],
        context: PICProcessContext,
        /,
    ) -> PICProcessResult:
        raise NotImplementedError

    @abc.abstractmethod
    def exchange(
        self, species: tuple[PICSpeciesState, ...], processes: tuple[Any, ...], /
    ) -> PICParticleExchangeResult:
        raise NotImplementedError


class AbstractPICRecorder(StrictModule):
    """Diagnostic recorder updated on every accepted PIC step.

    Recorders observe; they own no radiation energy and never alter the run.
    """

    recorder_id: eqx.AbstractVar[str]

    def validate_run(
        self,
        species: tuple[PICSpeciesPlan, ...],
        relativity: RelativityScaleContract,
        /,
    ) -> None:
        """Refuse, at run construction, species or pusher units it cannot observe.

        Recorders without such constraints accept every run.
        """
        del species, relativity

    def shift_frame(self, state: Any, axis: int, distance: float, /) -> Any:
        """Return ``state`` after the particle frame moved by ``distance`` along ``axis``.

        A moving window translates window-local positions by ``−distance``;
        recorders whose data depend on positions accumulate the offset so that
        they keep observing in the fixed frame. Other recorders return ``state``.
        """
        del axis, distance
        return state

    @abc.abstractmethod
    def initialize(
        self,
        species: tuple[PICSpeciesState, ...],
        time: Array,
        /,
    ) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def record(
        self,
        state: Any,
        species: tuple[PICSpeciesState, ...],
        time: Array,
        step_index: Array,
        /,
    ) -> Any:
        raise NotImplementedError


__all__ = [
    "AbstractPICParticleAllocator",
    "AbstractPICParticleExecutor",
    "AbstractPICProcess",
    "AbstractPICRecorder",
    "allocate_particles",
    "PICDistributedProcess",
    "PICFieldProbe",
    "PICFieldProbeSample",
    "PICParticleExchangeResult",
    "PICProcessBank",
    "PICProcessContext",
    "PICProcessLedger",
    "PICProcessRadiation",
    "PICProcessResult",
    "PICProcessStage",
    "PICProcessStatePartition",
    "RadiationOwnership",
]
