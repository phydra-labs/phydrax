#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit electromagnetic PIC over the minimal PIC field-solver protocol.

`ElectromagneticPICPlan` owns species (runtime populations with persistent
identities), the relativistic pusher, particle processes, particle
boundaries, recorders, field filters, external fields, radiation ownership,
and precision. The field — full 3-D cochain, reduced 1-D/2-D, or tetrahedral —
is any `AbstractPreparedPICFieldSolver`.

One step:

1. gather ``E``/``B`` (plus external fields) at integer-time positions;
2. push half-step proper velocities, then momentum-stage processes;
3. creation-stage processes (strong-field QED emission and pair creation) on
   the pushed species, with pointwise charge preservation verified by
   redeposition at the step-start positions;
4. drift, apply particle boundaries;
5. deposit charge-conserving current along each path and advance the field;
6. population-stage processes on the end-of-step species, with pointwise
   charge preservation verified by redeposition;
7. commit or reject the whole candidate; accepted recorders and process
   states advance.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, assert_never

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from .._fingerprint import canonical_fingerprint
from .._sampling import derive_key, SampleAddress
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.pic import (
    AbstractPICParticleExecutor,
    AbstractPICProcess,
    AbstractPICRecorder,
    ExternalFieldSource,
    PIC_CODE_RELATIVITY,
    PICBoundarySurfaceState,
    PICEnergyLedger,
    PICFieldProbe,
    PICFieldProbeSample,
    PICOpenBoundaryPlan,
    PICParticleState,
    PICProcessContext,
    PICProcessLedger,
    PICProcessResult,
    PICProcessStage,
    PICRejectionReason,
    PICRunStatus,
    PICSpeciesPlan,
    PICSpeciesState,
    RadiationOwnership,
    RelativisticPushPlan,
)
from ..typing import parse, PRNGKey
from ._fixed_step import AbstractFixedStepMethod, FixedStepResult
from ._maxwell_dispersion import CherenkovRegimeEvidence
from ._pic_cherenkov_guard import PICCherenkovGuard
from ._pic_field_solver import (
    AbstractPICFieldFilter,
    AbstractPreparedPICFieldSolver,
    add_deposits,
    chain_deposits,
    deposit_gauss_pairing_defect,
    PICEnergyAccounting,
    PICFieldDeposit,
    PICFieldEnergy,
    PICFilterContinuityReport,
    PICGalileanGrid,
    PICGaussProjection,
    PICGaussProjectionResult,
    PICMultiDeposit,
    PICOpenDomain,
    PICPrecisionPolicy,
    PICRelativisticSelfFields,
    PICRestartComponent,
    PICRestartState,
    PICSelfFieldInitialization,
    restart_component,
    restore_component,
)


class PICFieldHistory(StrictModule):
    """Previous accepted field state and its time.

    Kept only for processes that need field time derivatives; the rate
    ``(F(field) - F(history.field)) / (time - history.time)`` is gathered at the
    current particle positions.
    """

    field: Any
    time: Array


class ElectromagneticPICState(StrictModule):
    """Synchronized PIC state: integer-time positions, half-step proper velocities.

    ``wall_charge`` is the charge of particles absorbed by particle boundaries,
    held immobile on the grid in the field solver's charge layout.
    ``field_history`` is the staggered field history (``None`` unless a process
    requires field derivatives). ``processes`` holds each process's own state
    in process order (``None`` for stateless processes).
    """

    species: tuple[PICSpeciesState, ...]
    field: Any
    boundaries: tuple[PICBoundarySurfaceState, ...]
    wall_charge: Array
    recorders: tuple[Any, ...]
    time: Array
    accepted_step: Array
    status: Array
    field_history: PICFieldHistory | None
    processes: tuple[Any, ...]


class PICExitLedger(StrictModule):
    """Charge, mass, and kinetic energy absorbed particle boundaries took this step."""

    charge: Array
    mass: Array
    kinetic_energy: Array


class PICEnergySnapshot(StrictModule):
    """Particle and field energy synchronized at one integer time.

    ``particle_kinetic`` evaluates ``(γ − 1)mc²`` at the proper velocity after
    the electric half kick ``u + (q/m)EΔt/2`` of the next step, an
    ``O(Δt²)`` integer-time value of the half-step velocities. Field terms are
    those of `PICFieldEnergy` (``material`` is ``None`` for solvers without
    `PICEnergyAccounting`, whose ``electric_field`` is the total field
    energy).
    """

    particle_kinetic: Array
    electric_field: Array
    magnetic_field: Array
    material: Array | None
    total: Array


class ElectromagneticPICDiagnostics(StrictModule):
    """Step evidence.

    When a charge-redistributing population process (resampling) ran,
    ``process_charge_defect`` is the redistributed charge it moved and
    ``gauss_projection`` holds the field's Poisson projection onto the
    redeposited charge; ``electric_constraint`` is then the projected field's
    Gauss residual. ``process_evidence`` holds each process's own evidence in
    ledger order.

    ``particle_field_charge_defect`` compares the field's current-driven Gauss
    charge change with the (filtered) deposited charge change.
    ``continuity_scale`` is the deposit's unsigned charge-rate magnitude
    (`PICFieldDeposit.continuity_scale`). Continuity is certified relative to
    it: ``continuity_defect ≤ r·continuity_scale`` and
    ``particle_field_charge_defect ≤ r·(continuity_scale·Δt + max|ρ_field|)``
    with ``r = max(continuity_tolerance, 64 ε)`` for the field dtype's machine
    epsilon ``ε``, so the gate is unit-free and holds at roundoff on any grid.
    ``medium_charge`` is ``max|ρ_field − ρ_particles − ρ_wall|`` over the
    field's charge layout: the induced wall, conduction, and absorber charge
    the field holds beyond its particle sources (zero in vacuum away from
    conducting walls and absorbers). ``exit`` is the step's particle-boundary
    exit ledger and ``charge_ledger_defect`` the relative particle charge
    balance ``|Q_particles(t+Δt) + Q_exited − Q_particles(t)|``. ``exchange``
    is the evidence of the run executor's particle exchange before the
    population stage (``None`` without one).
    """

    continuity_defect: Array
    particle_field_charge_defect: Array
    continuity_scale: Array
    process_charge_defect: Array
    electric_constraint: Array
    magnetic_constraint: Array
    maximum_displacement_fraction: Array
    energy: PICEnergyLedger
    field: Any
    processes: tuple[PICProcessLedger, ...]
    transfer_successful: Array
    current_successful: Array
    pusher_successful: Array
    field_successful: Array
    process_successful: Array
    finite: Array
    successful: Array
    rejection_reason: Array
    process_evidence: tuple[Any, ...]
    gauss_projection: PICGaussProjectionResult | None
    exit: PICExitLedger
    medium_charge: Array
    charge_ledger_defect: Array
    exchange: Any = None


class ElectromagneticPICStepResult(StrictModule):
    candidate_state: ElectromagneticPICState
    accepted_state: ElectromagneticPICState
    diagnostics: ElectromagneticPICDiagnostics
    current: Any
    successful: Array


class PICRestartCheckpoint(StrictModule):
    """Per-component PIC restart; each component is admitted by its own owner."""

    components: tuple[PICRestartComponent, ...]


class _Gathered(StrictModule):
    electric: Array
    magnetic: Array
    successful: Array


class _FieldDerivatives(StrictModule):
    electric_gradient: tuple[Array, ...]
    magnetic_gradient: tuple[Array, ...]
    electric_rate: tuple[Array, ...]
    magnetic_rate: tuple[Array, ...]
    successful: Array


def _select(predicate: Array, candidate: Any, current: Any, /) -> Any:
    return jax.tree.map(
        lambda proposed, old: jnp.where(predicate, proposed, old), candidate, current
    )


def _validated_ownership(
    ownership: RadiationOwnership,
    processes: tuple[AbstractPICProcess, ...],
    /,
) -> RadiationOwnership:
    """Refuse overlapping static radiation claims.

    The field solver advances the field of the deposited current, so it owns
    the radiation the grid resolves (``"resolved-field"``); a
    ``"diagnostic-only"`` run would leave that self-consistent radiation and
    its back-reaction unowned. ``"subgrid-reaction"`` coexists with the
    resolved field only through the runtime scale-separation evidence of its
    one claiming process.
    """
    declared = parse(ownership, RadiationOwnership, "ownership")
    claims = [value.radiation_ownership for value in processes]
    if "resolved-field" in claims:
        raise ValueError("Only the field solver owns resolved-field radiation.")

    subgrid = claims.count("subgrid-reaction")
    match declared:
        case "subgrid-reaction":
            if subgrid != 1:
                raise ValueError(
                    "subgrid-reaction ownership requires exactly one claiming process."
                )
        case "resolved-field":
            if subgrid:
                raise ValueError(
                    f"A subgrid-reaction process overlaps {declared!r} ownership."
                )
        case "diagnostic-only":
            raise ValueError(
                "The self-consistent PIC field solver claims resolved-field "
                "radiation, which overlaps 'diagnostic-only' ownership."
            )
        case _:
            raise ValueError("ownership is invalid.")
    return declared


def _validate_boundaries(
    solver: AbstractPreparedPICFieldSolver,
    species_count: int,
    boundaries: PICOpenBoundaryPlan,
    /,
) -> None:
    """Particle faces must match the field box and keep stencils on the grid."""
    if not isinstance(boundaries, PICOpenBoundaryPlan):
        raise TypeError("boundaries must be PICOpenBoundaryPlan or None.")
    if boundaries.lower.size != solver.spatial_dimension:
        raise ValueError("Particle boundaries must match the field dimension.")
    if not isinstance(solver, PICOpenDomain):
        if any(boundaries.periodic):
            raise ValueError(
                "PERIODIC particle faces require a field solver implementing "
                "PICOpenDomain."
            )
        return
    periodic = solver.domain_periodic
    if boundaries.periodic != periodic:
        raise ValueError(
            "PERIODIC particle faces must be exactly the field solver's periodic axes."
        )
    lower, upper = solver.domain_bounds
    inset = np.max(
        np.asarray(
            [solver.boundary_inset(index) for index in range(species_count)],
            dtype=np.float64,
        ),
        axis=0,
    )
    planes_lower = np.asarray(boundaries.lower)
    planes_upper = np.asarray(boundaries.upper)
    for axis, wrapped in enumerate(periodic):
        if wrapped:
            continue
        if (
            planes_lower[axis] < lower[axis] + inset[axis]
            or planes_upper[axis] > upper[axis] - inset[axis]
        ):
            raise ValueError(
                f"Particle faces of axis {axis} must lie within the field box inset "
                f"by {inset[axis]:.3e}, where every species' deposit and gather "
                "stencil stays on the grid."
            )


class ElectromagneticPICPlan(StrictModule, NonTrainableState):
    """Explicit electromagnetic PIC over one prepared PIC field solver.

    ``executor`` is ``None`` for single-device execution; a decomposed run
    (`DistributedElectromagneticPICPlan`) installs its particle executor, which
    applies distributed processes per device and exchanges particles between
    devices before the population stage.
    """

    __strict_contract__ = True

    solver: AbstractPreparedPICFieldSolver
    species: tuple[PICSpeciesPlan, ...]
    processes: tuple[AbstractPICProcess, ...]
    boundaries: PICOpenBoundaryPlan | None
    recorders: tuple[AbstractPICRecorder, ...]
    filters: tuple[AbstractPICFieldFilter, ...]
    filter_reports: tuple[PICFilterContinuityReport, ...]
    cherenkov_guards: tuple[PICCherenkovGuard, ...]
    cherenkov_evidence: tuple[CherenkovRegimeEvidence, ...]
    external_fields: tuple[ExternalFieldSource, ...]
    pusher: RelativisticPushPlan
    precision: PICPrecisionPolicy
    random_key: PRNGKey | None
    executor: AbstractPICParticleExecutor | None
    ownership: RadiationOwnership = eqx.field(static=True)
    maximum_displacement_fraction: float = eqx.field(static=True)
    continuity_tolerance: float = eqx.field(static=True)
    constraint_tolerance: float = eqx.field(static=True)
    pairing_defect: float = eqx.field(static=True)
    field_derivatives: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        solver: AbstractPreparedPICFieldSolver,
        /,
        *,
        species: Sequence[PICSpeciesPlan],
        processes: Sequence[AbstractPICProcess] = (),
        boundaries: PICOpenBoundaryPlan | None = None,
        recorders: Sequence[AbstractPICRecorder] = (),
        filters: Sequence[AbstractPICFieldFilter] = (),
        cherenkov_guards: Sequence[PICCherenkovGuard] = (),
        ownership: RadiationOwnership = "resolved-field",
        precision: PICPrecisionPolicy | None = None,
        pusher: RelativisticPushPlan | None = None,
        external_fields: Sequence[ExternalFieldSource] = (),
        key: PRNGKey | None = None,
        maximum_displacement_fraction: float = 0.5,
        continuity_tolerance: float = 1.0e-9,
        constraint_tolerance: float = 1.0e-8,
        pairing_tolerance: float = 1.0e-9,
    ) -> None:
        if not isinstance(solver, AbstractPreparedPICFieldSolver):
            raise TypeError("solver must be an AbstractPreparedPICFieldSolver.")
        species_ = tuple(species)
        if not species_ or any(not isinstance(v, PICSpeciesPlan) for v in species_):
            raise TypeError("species must be a nonempty sequence of PICSpeciesPlan.")
        if len({value.species_id for value in species_}) != len(species_):
            raise ValueError("PIC species identities must be distinct.")
        solver.validate_species(species_)
        processes_ = tuple(processes)
        if any(not isinstance(value, AbstractPICProcess) for value in processes_):
            raise TypeError("processes must be AbstractPICProcess instances.")
        if any(
            index < 0 or index >= len(species_)
            for value in processes_
            for index in value.species_indices
        ):
            raise ValueError("A PIC process references a species outside the run.")
        if any(value.stochastic for value in processes_) and key is None:
            raise ValueError("Stochastic PIC processes require a root key.")
        redistributing = [value for value in processes_ if value.redistributes_charge]
        if any(value.stage != "population" for value in redistributing):
            raise ValueError(
                "Charge-redistributing PIC processes must be population-stage."
            )
        if redistributing and not isinstance(solver, PICGaussProjection):
            raise TypeError(
                "Charge-redistributing PIC processes require a field solver implementing "
                "PICGaussProjection."
            )
        declared = _validated_ownership(ownership, processes_)
        if boundaries is not None:
            _validate_boundaries(solver, len(species_), boundaries)
        recorders_ = tuple(recorders)
        if any(not isinstance(value, AbstractPICRecorder) for value in recorders_):
            raise TypeError("recorders must be AbstractPICRecorder instances.")
        filters_ = tuple(filters)
        if any(not isinstance(value, AbstractPICFieldFilter) for value in filters_):
            raise TypeError("filters must be AbstractPICFieldFilter instances.")
        reports = tuple(value.continuity_report(solver) for value in filters_)
        for report in reports:
            if not report.interior_commutation_defect <= float(pairing_tolerance):
                raise ValueError(
                    f"PIC filter interior commutation defect "
                    f"{report.interior_commutation_defect:.3e} exceeds "
                    f"{float(pairing_tolerance):.1e}: filtered current would not "
                    "conserve filtered charge."
                )
        guards = tuple(cherenkov_guards)
        if any(not isinstance(value, PICCherenkovGuard) for value in guards):
            raise TypeError("cherenkov_guards must be PICCherenkovGuard instances.")
        external = tuple(external_fields)
        if any(not isinstance(value, ExternalFieldSource) for value in external):
            raise TypeError("external_fields must implement ExternalFieldSource.")
        precision_ = PICPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, PICPrecisionPolicy):
            raise TypeError("precision must be PICPrecisionPolicy or None.")
        if precision_.field_dtype != solver.field_dtype:
            raise ValueError("PIC precision field dtype differs from the field solver.")
        pusher_ = (
            RelativisticPushPlan(PIC_CODE_RELATIVITY, method="boris")
            if pusher is None
            else pusher
        )
        if not isinstance(pusher_, RelativisticPushPlan):
            raise TypeError("pusher must be RelativisticPushPlan or None.")
        for value in processes_:
            value.validate_run(species_, pusher_.relativity)
        for value in recorders_:
            value.validate_run(species_, pusher_.relativity)
        cherenkov = tuple(
            value.certify(solver, len(species_), pusher_.speed_of_light)
            for value in guards
        )
        derivatives = any(value.requires_field_derivatives for value in processes_)
        if any(
            value.requires_field_derivatives and value.stage != "momentum"
            for value in processes_
        ):
            raise ValueError("Field-derivative PIC processes must be momentum-stage.")
        maximum = float(maximum_displacement_fraction)
        continuity = float(continuity_tolerance)
        constraint = float(constraint_tolerance)
        if not all(
            np.isfinite(value) and value > 0.0
            for value in (maximum, continuity, constraint, float(pairing_tolerance))
        ):
            raise ValueError("PIC displacement and tolerance limits must be positive.")
        pairing = deposit_gauss_pairing_defect(solver, species_)
        if pairing > float(pairing_tolerance):
            raise ValueError(
                f"PIC deposit↔Gauss pairing defect {pairing:.3e} exceeds "
                f"{float(pairing_tolerance):.1e}: the solver's Gauss charge does not "
                "follow its deposited current."
            )
        self.solver = solver
        self.species = species_
        self.processes = processes_
        self.boundaries = boundaries
        self.recorders = recorders_
        self.filters = filters_
        self.filter_reports = reports
        self.cherenkov_guards = guards
        self.cherenkov_evidence = cherenkov
        self.external_fields = external
        self.pusher = pusher_
        self.precision = precision_
        self.random_key = key
        self.executor = None
        self.ownership = declared
        self.maximum_displacement_fraction = maximum
        self.continuity_tolerance = continuity
        self.constraint_tolerance = constraint
        self.pairing_defect = pairing
        self.field_derivatives = derivatives
        self.plan_id = canonical_fingerprint(
            {
                "kind": "electromagnetic-pic-plan",
                "solver": solver.solver_id,
                "species": [value.plan_id for value in species_],
                "processes": [value.process_id for value in processes_],
                "boundaries": None if boundaries is None else boundaries.plan_id,
                "recorders": [value.recorder_id for value in recorders_],
                "filters": [value.filter_id for value in filters_],
                "cherenkov_guards": [value.guard_id for value in guards],
                "external_fields": [value.source_id for value in external],
                "ownership": declared,
                "precision": precision_.policy_id,
                "pusher": pusher_.plan_id,
                "maximum_displacement_fraction": maximum,
                "continuity_tolerance": continuity,
                "constraint_tolerance": constraint,
            }
        )

    # -- field views -------------------------------------------------------------

    def _grid_velocity(self, dtype: DTypeLike, /) -> Array:
        """Velocity of an admitted Galilean field grid (zero for lab-fixed grids)."""
        if (
            isinstance(self.solver, PICGalileanGrid)
            and self.solver.pic_capability("galilean-grid").admitted
        ):
            return jnp.asarray(self.solver.grid_velocity, dtype=dtype)
        return jnp.zeros((3,), dtype=dtype)

    def _lab_position(self, position: Array, time: Array, /) -> Array:
        """Lab position ``x + v_grid t`` of grid coordinates, padded to three axes."""
        padded = jnp.pad(position, ((0, 0), (0, 3 - position.shape[1])))
        return padded + time * self._grid_velocity(position.dtype)

    def _filtered_charge(self, charge: Array, /) -> Array:
        for value in self.filters:
            charge = value.filter_charge(self.solver, charge)
        return charge

    def _filtered_current(self, current: Any, /) -> Any:
        for value in self.filters:
            current = value.filter_current(self.solver, current)
        return current

    def _filtered_field(self, field: Any, /) -> Any:
        for value in self.filters:
            field = value.filter_field(self.solver, field)
        return field

    def _sample_fields(
        self, index: int, position: Array, active: Array, view: Any, time: Array, /
    ) -> _Gathered:
        """Grid plus external fields at positions through species ``index``'s route."""
        sample = self.solver.gather(index, position, active, view)
        electric, magnetic, successful = (
            sample.electric,
            sample.magnetic,
            sample.successful,
        )
        if self.external_fields:
            position3 = self._lab_position(position, time)
            times = jnp.full((position.shape[0],), time, dtype=position.dtype)
            for source in self.external_fields:
                external = source.external_fields(position3, times)
                electric = electric + jnp.where(active[:, None], external.electric, 0.0)
                magnetic = magnetic + jnp.where(active[:, None], external.magnetic, 0.0)
                successful = successful & jnp.all(external.support | ~active)
        return _Gathered(electric, magnetic, successful)

    def _gather(
        self,
        species: tuple[PICSpeciesState, ...],
        field: Any,
        time: Array,
        /,
    ) -> tuple[_Gathered, ...]:
        view = self._filtered_field(field)
        return tuple(
            self._sample_fields(
                index, state.particles.position, state.population.active, view, time
            )
            for index, state in enumerate(species)
        )

    def _probe(
        self, probe: PICFieldProbe, field: Any, time: Array, /
    ) -> PICFieldProbeSample:
        """Fields at a process's probe positions, in chunks of the route capacity."""
        capacity = self.species[probe.species].capacity
        count, dimension = probe.position.shape
        if count % capacity:
            raise ValueError(
                "A PIC field probe must hold a multiple of its route species capacity."
            )
        view = self._filtered_field(field)
        chunks = count // capacity
        sampled = jax.lax.map(
            lambda chunk: self._sample_fields(
                probe.species, chunk[0], chunk[1], view, time
            ),
            (
                probe.position.reshape(chunks, capacity, dimension),
                probe.active.reshape(chunks, capacity),
            ),
        )
        return PICFieldProbeSample(
            sampled.electric.reshape(count, 3),
            sampled.magnetic.reshape(count, 3),
            jnp.all(sampled.successful),
        )

    def species_charge(
        self, species: tuple[PICSpeciesState, ...], /
    ) -> tuple[Array, Array]:
        """Total deposited particle charge in the solver layout and its success."""
        charges = []
        successful = jnp.asarray(True)
        for index, (plan, state) in enumerate(zip(self.species, species, strict=True)):
            charge, ok = self.solver.deposit_charge(
                index,
                state.particles.position,
                plan.macrocharge(state),
                state.population.active,
            )
            charges.append(charge)
            successful = successful & ok
        total = charges[0]
        for value in charges[1:]:
            total = total + value
        return total, successful

    def _particle_kinetic(self, state: PICSpeciesState, proper: Array, /) -> Array:
        """Per-particle ``(γ − 1)mc²`` of ``proper`` (zero for inactive slots)."""
        dtype = jnp.dtype(self.precision.accumulation_dtype)
        c2 = self.pusher.speed_of_light**2
        u = proper.astype(dtype)
        squared = jnp.sum(u**2, axis=-1) / c2
        # (γ − 1) = u²/c² / (γ + 1) avoids cancellation for slow particles.
        return jnp.where(
            state.population.active,
            state.population.mass.astype(dtype)
            * c2
            * squared
            / (jnp.sqrt(1.0 + squared) + 1.0),
            0.0,
        )

    def _path_kinetic(
        self, previous: PICSpeciesState, pushed: PICSpeciesState, /
    ) -> tuple[Array, Array]:
        """Kinetic energy at the start and end time of each drift path.

        Half-step energies ``K∓`` (before and after the push) give the
        integer-time values ``(K⁻ + K⁺)/2`` and ``(3K⁺ − K⁻)/2`` to second
        order; a slot created this step has no earlier energy and uses ``K⁺``.
        """
        after = self._particle_kinetic(pushed, pushed.particles.proper_velocity)
        before = jnp.where(
            previous.population.active,
            self._particle_kinetic(previous, previous.particles.proper_velocity),
            after,
        )
        return 0.5 * (before + after), 0.5 * (3.0 * after - before)

    def _kinetic(self, species: tuple[PICSpeciesState, ...], /) -> Array:
        dtype = jnp.dtype(self.precision.accumulation_dtype)
        total = jnp.asarray(0.0, dtype=dtype)
        for state in species:
            total = total + jnp.sum(
                self._particle_kinetic(state, state.particles.proper_velocity)
            )
        return total

    def _particle_charge(self, species: tuple[PICSpeciesState, ...], /) -> Array:
        """Total active macrocharge and the unsigned total that scales it."""
        dtype = jnp.dtype(self.precision.accumulation_dtype)
        total = jnp.zeros((2,), dtype=dtype)
        for plan, state in zip(self.species, species, strict=True):
            charge = jnp.where(
                state.population.active, plan.macrocharge(state).astype(dtype), 0.0
            )
            total = total + jnp.stack((jnp.sum(charge), jnp.sum(jnp.abs(charge))))
        return total

    def _field_energy(self, field: Any, step_size: Array, /) -> PICFieldEnergy | None:
        solver = self.solver
        if isinstance(solver, PICEnergyAccounting):
            return solver.energy_components(field, step_size)
        return None

    def synchronized_energy(
        self, state: ElectromagneticPICState, step_size: ArrayLike, /
    ) -> PICEnergySnapshot:
        """Particle and field energy of ``state`` synchronized at its integer time.

        Differences of these snapshots over a run, plus the ledger's dissipated,
        exited, radiated, and created energy, close to the order of the
        leapfrog; per-step ledgers pair half-step kinetic energies with
        integer-time fields and telescope to a first-order endpoint term.
        """
        dt = jnp.asarray(step_size, dtype=state.time.dtype).reshape(())
        gathered = self._gather(state.species, state.field, state.time)
        dtype = jnp.dtype(self.precision.accumulation_dtype)
        kinetic = jnp.asarray(0.0, dtype=dtype)
        for plan, species, sample in zip(
            self.species, state.species, gathered, strict=True
        ):
            kicked = (
                species.particles.proper_velocity
                + 0.5 * dt * plan.specific_charge(species)[:, None] * sample.electric
            )
            kinetic = kinetic + jnp.sum(self._particle_kinetic(species, kicked))
        energy = self._field_energy(state.field, dt)
        if energy is None:
            field = self.solver.field_energy(state.field).astype(dtype)
            return PICEnergySnapshot(
                kinetic, field, jnp.zeros((), dtype=dtype), None, kinetic + field
            )
        electric, magnetic, material = (
            value.astype(dtype)
            for value in (energy.electric, energy.magnetic, energy.material)
        )
        return PICEnergySnapshot(
            kinetic,
            electric,
            magnetic,
            material,
            kinetic + electric + magnetic + material,
        )

    def _process_key(self, process: AbstractPICProcess, step: Array, /) -> PRNGKey | None:
        if not process.stochastic or self.random_key is None:
            return None
        return derive_key(
            self.random_key,
            SampleAddress(
                "phydrax.pic", "process", target=process.process_id, role="event"
            ),
            step,
        )

    def _field_derivatives(
        self,
        species: tuple[PICSpeciesState, ...],
        field: Any,
        history: PICFieldHistory,
        time: Array,
        /,
    ) -> _FieldDerivatives:
        """Gradients and time derivatives of the total gathered fields.

        Grid fields: gradients from the solver's order-one gather, time
        derivatives from the staggered history gathered at the same positions.
        External fields: exact forward-mode derivatives along resolved axes and
        time.
        """
        view = self._filtered_field(field)
        earlier_view = self._filtered_field(history.field)
        interval = time - history.time
        successful = interval > 0.0
        gradients: list[tuple[Array, Array]] = []
        rates: list[tuple[Array, Array]] = []
        for index, state in enumerate(species):
            active = state.population.active
            position = state.particles.position
            sample = self.solver.gather(index, position, active, view, derivative_order=1)
            earlier = self.solver.gather(index, position, active, earlier_view)
            electric_gradient = sample.electric_gradient
            magnetic_gradient = sample.magnetic_gradient
            if electric_gradient is None or magnetic_gradient is None:
                raise ValueError("The PIC field solver returned no order-one gradients.")
            electric_rate = (sample.electric - earlier.electric) / interval
            magnetic_rate = (sample.magnetic - earlier.magnetic) / interval
            successful = successful & sample.successful & earlier.successful
            dimension = position.shape[1]
            position3 = self._lab_position(position, time)
            times = jnp.full((position.shape[0],), time, dtype=position.dtype)
            for source in self.external_fields:

                def fields(
                    x: Array, t: Array, source: ExternalFieldSource = source
                ) -> tuple[Array, Array]:
                    sample = source.external_fields(x, t)
                    return sample.electric, sample.magnetic

                columns = tuple(
                    jax.jvp(
                        fields,
                        (position3, times),
                        (
                            jnp.zeros_like(position3).at[:, axis].set(1.0),
                            jnp.zeros_like(times),
                        ),
                    )[1]
                    for axis in range(dimension)
                )
                padding = ((0, 0), (0, 0), (0, 3 - dimension))
                mask = active[:, None, None]
                electric_gradient = electric_gradient + jnp.where(
                    mask,
                    jnp.pad(jnp.stack(tuple(v[0] for v in columns), axis=-1), padding),
                    0.0,
                )
                magnetic_gradient = magnetic_gradient + jnp.where(
                    mask,
                    jnp.pad(jnp.stack(tuple(v[1] for v in columns), axis=-1), padding),
                    0.0,
                )
                _, (electric_time, magnetic_time) = jax.jvp(
                    fields,
                    (position3, times),
                    (jnp.zeros_like(position3), jnp.ones_like(times)),
                )
                electric_rate = electric_rate + jnp.where(
                    active[:, None], electric_time, 0.0
                )
                magnetic_rate = magnetic_rate + jnp.where(
                    active[:, None], magnetic_time, 0.0
                )
            gradients.append((electric_gradient, magnetic_gradient))
            rates.append((electric_rate, magnetic_rate))
        return _FieldDerivatives(
            tuple(value[0] for value in gradients),
            tuple(value[1] for value in gradients),
            tuple(value[0] for value in rates),
            tuple(value[1] for value in rates),
            successful,
        )

    def _grid_cutoff_frequency(self, dt: Array, /) -> Array:
        """Highest angular frequency the field grid resolves in space and time."""
        widths = self.solver.displacement_widths.astype(dt.dtype)
        spatial = jnp.pi * self.pusher.speed_of_light / jnp.min(widths)
        return jnp.minimum(spatial, jnp.pi / dt)

    def _run_processes(
        self,
        stage: PICProcessStage,
        species: tuple[PICSpeciesState, ...],
        start: tuple[PICSpeciesState, ...],
        gathered: tuple[_Gathered, ...],
        states: tuple[Any, ...],
        field: Any,
        time: Array,
        dt: Array,
        step: Array,
        derivatives: _FieldDerivatives | None,
        /,
    ) -> tuple[
        tuple[PICSpeciesState, ...],
        tuple[PICProcessLedger, ...],
        tuple[Any, ...],
        tuple[Any, ...],
        Array,
    ]:
        """Apply one stage; returns species, ledgers, evidence, states, probe success.

        Probes sample the step-start ``field`` at ``time``; ``start`` are the
        species before this step's push.
        """
        ledgers = []
        evidence = []
        updated = list(states)
        probed = jnp.asarray(True)
        cutoff = self._grid_cutoff_frequency(dt)
        for index, process in enumerate(self.processes):
            if process.stage != stage:
                continue
            supplied = derivatives if process.requires_field_derivatives else None
            probe = process.field_probe(states[index])
            sample = None if probe is None else self._probe(probe, field, time)
            if sample is not None:
                probed = probed & sample.successful
            result = self._apply_process(
                index,
                process,
                self.species,
                PICProcessContext(
                    species,
                    tuple(value.electric for value in gathered),
                    tuple(value.magnetic for value in gathered),
                    time,
                    dt,
                    step,
                    self._process_key(process, step),
                    None if supplied is None else supplied.electric_gradient,
                    None if supplied is None else supplied.magnetic_gradient,
                    None if supplied is None else supplied.electric_rate,
                    None if supplied is None else supplied.magnetic_rate,
                    cutoff,
                    states[index],
                    sample,
                    self.pusher,
                    tuple(value.particles.proper_velocity for value in start),
                    self._grid_velocity(dt.dtype),
                ),
            )
            if stage == "momentum":
                # Momentum-stage processes own proper velocities only.
                species = tuple(
                    PICSpeciesState(
                        PICParticleState(
                            old.particles.position, new.particles.proper_velocity
                        ),
                        old.population,
                        old.charge,
                    )
                    for old, new in zip(species, result.species, strict=True)
                )
            else:
                species = result.species
            ledgers.append(result.ledger)
            evidence.append(result.evidence)
            updated[index] = result.state
        return species, tuple(ledgers), tuple(evidence), tuple(updated), probed

    def _apply_process(
        self,
        index: int,
        process: AbstractPICProcess,
        species: tuple[PICSpeciesPlan, ...],
        context: PICProcessContext,
        /,
    ) -> PICProcessResult:
        executor = self.executor
        if executor is None:
            return process.apply(species, context)
        return executor.apply_process(index, process, species, context)

    def initialize_process_states(
        self, species: tuple[PICSpeciesState, ...], /
    ) -> tuple[Any, ...]:
        """Initial state of every process (``None`` for stateless processes)."""
        return tuple(
            process.initialize_state(self.species, species) for process in self.processes
        )

    # -- lifecycle ---------------------------------------------------------------

    def _relativistic_fields(
        self,
        species: tuple[PICSpeciesState, ...],
        velocities: tuple[ArrayLike, ...],
        masks: tuple[ArrayLike | None, ...],
        drifts: Sequence[ArrayLike | None] | None,
        magnetic: Any,
        /,
    ) -> tuple[Any, Array, Array, Array]:
        """Superposed per-species boosted-Coulomb field, total charge, successes."""
        solver = self.solver
        if not isinstance(solver, PICRelativisticSelfFields):
            raise TypeError(
                "self_fields='relativistic-per-species' requires a field solver "
                "implementing PICRelativisticSelfFields."
            )
        if magnetic is not None:
            raise ValueError(
                "Relativistic self-fields own the initial magnetic field; "
                "magnetic must be None."
            )
        drift_values = (None,) * len(species) if drifts is None else tuple(drifts)
        if len(drift_values) != len(species):
            raise ValueError("drifts needs one entry (or None) per species.")
        light = float(self.pusher.speed_of_light)
        betas = []
        for velocity, mask, drift in zip(velocities, masks, drift_values, strict=True):
            if drift is None:
                values = np.asarray(velocity, dtype=np.float64)
                active = (
                    np.ones((values.shape[0],), dtype=np.bool_)
                    if mask is None
                    else np.asarray(mask, dtype=np.bool_)
                )
                mean = (
                    np.mean(values[active], axis=0)
                    if np.any(active)
                    else np.zeros((3,), dtype=np.float64)
                )
            else:
                mean = np.asarray(drift, dtype=np.float64)
            betas.append(tuple(float(value) / light for value in mean.reshape(-1)))
        charges = []
        successful = jnp.asarray(True)
        for index, (plan, state) in enumerate(zip(self.species, species, strict=True)):
            charge, ok = solver.deposit_charge(
                index,
                state.particles.position,
                plan.macrocharge(state),
                state.population.active,
            )
            charges.append(self._filtered_charge(charge))
            successful = successful & ok
        result = solver.initialize_relativistic_field(tuple(charges), tuple(betas), light)
        total = charges[0]
        for value in charges[1:]:
            total = total + value
        return result.field, total, successful, result.successful

    def initialize(
        self,
        positions: Sequence[ArrayLike],
        velocities: Sequence[ArrayLike],
        step_size: ArrayLike,
        /,
        *,
        active_masks: Sequence[ArrayLike | None] | None = None,
        masses: Sequence[ArrayLike | None] | None = None,
        magnetic: Any = None,
        time: ArrayLike = 0.0,
        self_fields: PICSelfFieldInitialization = "electrostatic",
        drifts: Sequence[ArrayLike | None] | None = None,
    ) -> ElectromagneticPICState:
        """Gauss-consistent initial state from positions and physical velocities.

        ``self_fields="electrostatic"`` solves Poisson for the total charge
        (``B`` from ``magnetic``, else zero). ``"relativistic-per-species"``
        gives every species the lab-frame field of its rest-frame Coulomb field
        (`PICRelativisticSelfFields`): ``drifts[s]`` is its drift velocity
        (physical units, along one grid axis) or ``None`` for the mean velocity
        of its active particles. Drifts are read on the host. A static field
        with ``B = 0`` around a relativistic beam is not a solution of the
        drifting problem and radiates a spurious transient.

        Proper velocities are bootstrapped half a step backward in the initial
        field so the pusher sees leapfrog-staggered momenta.
        """
        mode = parse(self_fields, PICSelfFieldInitialization, "self_fields")
        count = len(self.species)
        position_values = tuple(positions)
        velocity_values = tuple(velocities)
        masks = (None,) * count if active_masks is None else tuple(active_masks)
        mass_values = (None,) * count if masses is None else tuple(masses)
        if not (
            len(position_values)
            == len(velocity_values)
            == len(masks)
            == len(mass_values)
            == count
        ):
            raise ValueError("One position and velocity array is required per species.")
        dtype = jnp.dtype(self.precision.particle_dtype)
        c2 = self.pusher.speed_of_light**2
        species = []
        valid = jnp.asarray(True)
        for plan, position, velocity, mask, mass in zip(
            self.species,
            position_values,
            velocity_values,
            masks,
            mass_values,
            strict=True,
        ):
            position_ = jnp.asarray(position, dtype=dtype)
            velocity_ = jnp.asarray(velocity, dtype=dtype)
            if position_.shape != (plan.capacity, self.solver.spatial_dimension):
                raise ValueError(
                    "PIC positions must have capacity-by-spatial-dimension shape."
                )
            if velocity_.shape != (plan.capacity, 3):
                raise ValueError("PIC velocities must have capacity-by-three shape.")
            speed2 = jnp.sum(velocity_ * velocity_, axis=-1)
            state = plan.initialize(
                position_,
                velocity_ / jnp.sqrt(1.0 - speed2 / c2)[:, None],
                active_mask=mask,
                masses=mass,
            )
            valid = valid & ~jnp.any(
                state.population.active & (~jnp.isfinite(speed2) | (speed2 >= c2))
            )
            species.append(state)
        species_tuple = tuple(species)
        match mode:
            case "electrostatic":
                if drifts is not None:
                    raise ValueError(
                        "drifts require self_fields='relativistic-per-species'."
                    )
                charge, charge_success = self.species_charge(species_tuple)
                field, field_success = self.solver.initialize_field(
                    self._filtered_charge(charge), magnetic=magnetic
                )
            case "relativistic-per-species":
                field, charge, charge_success, field_success = self._relativistic_fields(
                    species_tuple, velocity_values, masks, drifts, magnetic
                )
            case _:
                assert_never(mode)
        t0 = jnp.asarray(time, dtype=dtype).reshape(())
        dt = jnp.asarray(step_size, dtype=dtype).reshape(())
        gathered = self._gather(species_tuple, field, t0)
        bootstrapped = []
        success = charge_success & field_success
        for plan, state, sample in zip(
            self.species, species_tuple, gathered, strict=True
        ):
            backward = self.pusher.push(
                state.particles.proper_velocity,
                sample.electric,
                sample.magnetic,
                plan.specific_charge(state),
                state.population.active,
                -0.5 * dt,
            )
            bootstrapped.append(
                PICSpeciesState(
                    PICParticleState(state.particles.position, backward.proper_velocity),
                    state.population,
                    state.charge,
                )
            )
            success = success & sample.successful & backward.successful
        bootstrapped_tuple = tuple(bootstrapped)
        boundaries = (
            ()
            if self.boundaries is None
            else tuple(self.boundaries.initialize_surface(dtype) for _ in self.species)
        )
        state = ElectromagneticPICState(
            bootstrapped_tuple,
            field,
            boundaries,
            jnp.zeros_like(charge),
            tuple(
                recorder.initialize(bootstrapped_tuple, t0) for recorder in self.recorders
            ),
            t0,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(int(PICRunStatus.SUCCESS), dtype=jnp.int32),
            # The backward half-push bootstrap already treats the initial field
            # as static over the preceding step; the history declares the same.
            PICFieldHistory(field, t0 - dt) if self.field_derivatives else None,
            self.initialize_process_states(bootstrapped_tuple),
        )
        return jax.tree.map(
            lambda value: (
                eqx.error_if(
                    value,
                    ~success | ~valid,
                    "Electromagnetic PIC initialization failed: velocities must be "
                    "finite and subluminal and charge, field, and gather must succeed.",
                )
                if eqx.is_array(value)
                else value
            ),
            state,
        )

    def _advance_species(
        self,
        state: ElectromagneticPICState,
        gathered: tuple[_Gathered, ...],
        dt: Array,
        /,
    ) -> tuple[tuple[PICSpeciesState, ...], tuple[Array, ...], Array]:
        """Push, returning half-step species, their mean velocities, and success."""
        pushed = []
        velocities = []
        successful = jnp.asarray(True)
        for plan, species, sample in zip(
            self.species, state.species, gathered, strict=True
        ):
            result = self.pusher.push(
                species.particles.proper_velocity,
                sample.electric,
                sample.magnetic,
                plan.specific_charge(species),
                species.population.active,
                dt,
            )
            pushed.append(
                PICSpeciesState(
                    PICParticleState(species.particles.position, result.proper_velocity),
                    species.population,
                    species.charge,
                )
            )
            velocities.append(result.velocity)
            successful = successful & result.successful
        return tuple(pushed), tuple(velocities), successful

    def _move(
        self,
        state: ElectromagneticPICState,
        species: tuple[PICSpeciesState, ...],
        velocities: tuple[Array, ...],
        dt: Array,
        /,
    ) -> tuple[
        tuple[PICSpeciesState, ...],
        PICFieldDeposit,
        tuple[PICBoundarySurfaceState, ...],
        Array,
        Array,
        Array,
        PICExitLedger,
    ]:
        """Drift, apply particle boundaries, and deposit every species' current."""
        widths = self.solver.displacement_widths.astype(dt.dtype)
        dimension = self.solver.spatial_dimension
        moved = []
        deposits = []
        surfaces = []
        wall = state.wall_charge
        maximum_fraction = jnp.asarray(0.0, dtype=dt.dtype)
        boundary_success = jnp.asarray(True)
        exited = jnp.zeros((3,), dtype=jnp.dtype(self.precision.accumulation_dtype))
        starts, ends, means, charges, actives = [], [], [], [], []
        for index, (plan, value, velocity) in enumerate(
            zip(self.species, species, velocities, strict=True)
        ):
            active = value.population.active
            start = value.particles.position
            displacement = dt * (
                velocity[:, :dimension] - self._grid_velocity(dt.dtype)[:dimension]
            )
            position = jnp.where(active[:, None], start + displacement, 0.0)
            maximum_fraction = jnp.maximum(
                maximum_fraction,
                jnp.max(
                    jnp.where(active[:, None], jnp.abs(displacement) / widths, 0.0),
                    initial=0.0,
                ),
            )
            macrocharge = plan.macrocharge(value)
            if self.boundaries is None:
                starts.append(start)
                ends.append(position)
                means.append(velocity)
                charges.append(macrocharge)
                actives.append(active)
                moved.append(
                    PICSpeciesState(
                        PICParticleState(position, value.particles.proper_velocity),
                        value.population,
                        value.charge,
                    )
                )
                continue
            boundary = self.boundaries.apply(
                plan.population,
                value.population,
                value.particles,
                position,
                macrocharge,
                state.boundaries[index],
                kinetic_energy=self._path_kinetic(state.species[index], value),
            )
            exited = exited + jnp.stack(
                (
                    jnp.sum(boundary.boundary_charge_flux),
                    jnp.sum(boundary.boundary_mass_flux),
                    jnp.sum(boundary.boundary_energy_flux),
                )
            ).astype(exited.dtype)
            hit = boundary.hit_mask
            waypoint = jnp.where(hit[:, None], boundary.hit_position, position)
            fraction = jnp.where(hit, boundary.hit_fraction, 1.0)[:, None]
            final = boundary.candidate_particles
            final_position = jnp.where(
                boundary.candidate_population.active[:, None],
                final.position,
                waypoint,
            )
            # Reflected paths are two straight segments joined at the wall.
            first = self.solver.deposit(
                index, start, waypoint, fraction * velocity, macrocharge, active, dt
            )
            second = self.solver.deposit(
                index,
                waypoint,
                final_position,
                (1.0 - fraction) * self.pusher.velocity(final.proper_velocity),
                macrocharge,
                active,
                dt,
            )
            deposits.append(chain_deposits(first, second))
            absorbed = active & ~boundary.candidate_population.active
            wall_charge, wall_success = self.solver.deposit_charge(
                index, waypoint, macrocharge, absorbed
            )
            wall = wall + wall_charge
            surfaces.append(boundary.candidate_surface)
            boundary_success = boundary_success & boundary.successful & wall_success
            moved.append(
                PICSpeciesState(
                    PICParticleState(
                        jnp.where(
                            boundary.candidate_population.active[:, None],
                            final.position,
                            0.0,
                        ),
                        final.proper_velocity,
                    ),
                    boundary.candidate_population,
                    value.charge,
                )
            )
        if self.boundaries is None:
            if isinstance(self.solver, PICMultiDeposit):
                total = self.solver.deposit_all(
                    tuple(starts),
                    tuple(ends),
                    tuple(means),
                    tuple(charges),
                    tuple(actives),
                    dt,
                )
            else:
                total = add_deposits(
                    tuple(
                        self.solver.deposit(index, *arrays, dt)
                        for index, arrays in enumerate(
                            zip(starts, ends, means, charges, actives, strict=True)
                        )
                    )
                )
        else:
            total = add_deposits(tuple(deposits))
        return (
            tuple(moved),
            total,
            tuple(surfaces),
            wall,
            maximum_fraction,
            boundary_success,
            PICExitLedger(exited[0], exited[1], exited[2]),
        )

    def _energy_ledger(
        self,
        state: ElectromagneticPICState,
        final: tuple[PICSpeciesState, ...],
        field: Any,
        field_energy: Array,
        advanced_field: Any,
        dt: Array,
        radiated: Array,
        rest_energy: Array,
        exchange: Array,
        exited: Array,
        /,
    ) -> PICEnergyLedger:
        """Step ledger; accounting solvers split field energy and report losses.

        Losses are the solver's source-free ``−dW/dt`` integrated with the
        trapezoidal rule between the step-start and the advanced field.
        """
        dtype = jnp.dtype(self.precision.accumulation_dtype)
        previous_kinetic = self._kinetic(state.species)
        next_kinetic = self._kinetic(final)
        solver = self.solver
        if not isinstance(solver, PICEnergyAccounting):
            previous_total = previous_kinetic + self.solver.field_energy(
                state.field
            ).astype(dtype)
            total = next_kinetic + field_energy.astype(dtype)
            return PICEnergyLedger(
                next_kinetic,
                field_energy.astype(dtype),
                jnp.zeros((), dtype=dtype),
                radiated,
                total,
                previous_total,
                total + radiated + rest_energy + exited - exchange - previous_total,
                rest_energy,
                exchange,
                None,
                None,
                exited,
            )
        before = solver.energy_components(state.field, dt)
        after = solver.energy_components(field, dt)
        dissipated = (
            0.5
            * dt
            * (solver.loss_power(state.field) + solver.loss_power(advanced_field))
        ).astype(dtype)
        electric, magnetic, material = (
            value.astype(dtype)
            for value in (after.electric, after.magnetic, after.material)
        )
        previous_total = previous_kinetic + (
            before.electric + before.magnetic + before.material
        ).astype(dtype)
        total = next_kinetic + electric + magnetic + material
        return PICEnergyLedger(
            next_kinetic,
            electric,
            magnetic,
            radiated,
            total,
            previous_total,
            total
            + radiated
            + rest_energy
            + dissipated
            + exited
            - exchange
            - previous_total,
            rest_energy,
            exchange,
            material,
            dissipated,
            exited,
        )

    def step_detailed(
        self, state: ElectromagneticPICState, step_size: ArrayLike, /
    ) -> ElectromagneticPICStepResult:
        """Execute one atomic PIC transaction in its physical lifecycle order.

        Gather, push, momentum/creation/population processes, deposition, field
        advance, Gauss restoration, ledgers, evidence, and commit/rollback stay
        centralized so no extracted phase can observe or publish a partially
        accepted state. Numerical branch bodies remain in their owning helpers.
        """
        if not isinstance(state, ElectromagneticPICState):
            raise TypeError("state must be ElectromagneticPICState.")
        dt = jnp.asarray(step_size, dtype=state.time.dtype).reshape(())
        step = state.accepted_step
        gathered = self._gather(state.species, state.field, state.time)
        gather_success = jnp.all(jnp.stack(tuple(value.successful for value in gathered)))
        derivatives = None
        if state.field_history is not None:
            derivatives = self._field_derivatives(
                state.species, state.field, state.field_history, state.time
            )
            gather_success = gather_success & derivatives.successful
        pushed, velocities, pusher_success = self._advance_species(state, gathered, dt)
        pushed, momentum_ledgers, momentum_evidence, states, _ = self._run_processes(
            "momentum",
            pushed,
            state.species,
            gathered,
            state.processes,
            state.field,
            state.time,
            dt,
            step,
            derivatives,
        )
        created, creation_ledgers, creation_evidence, states, probed = (
            self._run_processes(
                "creation",
                pushed,
                state.species,
                gathered,
                states,
                state.field,
                state.time,
                dt,
                step,
                None,
            )
        )
        gather_success = gather_success & probed
        creation_charged = jnp.asarray(True)
        creation_charge_defect = jnp.zeros((), dtype=dt.dtype)
        if creation_ledgers:
            # Created particles appear at step-start positions in charge-neutral
            # sets, so the deposited charge must be unchanged pointwise.
            before, _ = self.species_charge(pushed)
            after, redeposit_success = self.species_charge(created)
            creation_charge_defect = jnp.max(
                jnp.abs(after - before), initial=0.0
            ) / jnp.maximum(1.0, jnp.max(jnp.abs(before), initial=0.0))
            creation_charged = redeposit_success & (
                creation_charge_defect <= self.continuity_tolerance
            )
            pushed = created
        if momentum_ledgers or creation_ledgers:
            velocities = tuple(
                self.pusher.velocity(value.particles.proper_velocity) for value in pushed
            )
        moved, deposit, surfaces, wall, fraction, boundary_success, exit_ledger = (
            self._move(state, pushed, velocities, dt)
        )
        current = self._filtered_current(deposit.current)
        advanced = self.solver.advance(state.time, state.field, current, dt)
        # The field's Gauss charge moves by the deposited (filtered) charge
        # change; absorbed particles stay in the end charge at their exit point
        # and join the immobile wall charge from the next step on.
        expected_change = deposit.end_charge - deposit.start_charge
        if self.filters:
            expected_change = self._filtered_charge(expected_change)
        charge_defect = jnp.max(
            jnp.abs(
                advanced.charge - self.solver.field_charge(state.field) - expected_change
            ),
            initial=0.0,
        )
        exchange = None
        exchanged = jnp.asarray(True)
        executor = self.executor
        if executor is not None and any(
            value.stage == "population" for value in self.processes
        ):
            # Population processes group particles by cell and create them at
            # home positions, so every particle first moves to its owner.
            exchange = executor.exchange(moved, states)
            moved, states = exchange.species, exchange.processes
            exchanged = exchange.successful
        final, population_ledgers, population_evidence, states, _ = self._run_processes(
            "population",
            moved,
            state.species,
            gathered,
            states,
            state.field,
            state.time,
            dt,
            step,
            None,
        )
        field = advanced.field
        electric_constraint = advanced.electric_constraint
        field_energy = advanced.energy
        projection = None
        process_charged = jnp.asarray(True)
        particle_layout_charge = deposit.end_charge + state.wall_charge
        if population_ledgers:
            before, _ = self.species_charge(moved)
            after, redeposit_success = self.species_charge(final)
            particle_layout_charge = after + wall
            process_charge_defect = jnp.max(
                jnp.abs(after - before), initial=0.0
            ) / jnp.maximum(1.0, jnp.max(jnp.abs(before), initial=0.0))
            if any(value.redistributes_charge for value in self.processes):
                # Resampling moves charge between grid locations while
                # conserving its total (checked by the process ledger); Gauss is
                # restored by projecting onto the redeposited, filtered charge.
                solver = self.solver
                if not isinstance(solver, PICGaussProjection):
                    raise TypeError("The PIC field solver lost its PICGaussProjection.")
                change = after - before
                if self.filters:
                    change = self._filtered_charge(change)
                projection = solver.project_gauss(
                    field, solver.field_charge(field) + change
                )
                field = projection.field
                electric_constraint = projection.divergence_after
                field_energy = field_energy + projection.energy_change
                process_charged = redeposit_success & projection.successful
            else:
                process_charged = redeposit_success & (
                    process_charge_defect <= self.continuity_tolerance
                )
        else:
            process_charge_defect = jnp.zeros((), dtype=dt.dtype)
        process_charge_defect = jnp.maximum(process_charge_defect, creation_charge_defect)
        ledgers = momentum_ledgers + creation_ledgers + population_ledgers
        process_success = process_charged & creation_charged
        for ledger in ledgers:
            process_success = process_success & ledger.successful
        dtype = jnp.dtype(self.precision.accumulation_dtype)
        radiated = jnp.zeros((), dtype=dtype)
        rest_energy = jnp.zeros((), dtype=dtype)
        exchange = jnp.zeros((), dtype=dtype)
        ownership_ok = jnp.asarray(True)
        for ledger in ledgers:
            radiation = ledger.radiation
            if radiation is not None:
                radiated = radiated + radiation.radiated_energy.astype(dtype)
                ownership_ok = ownership_ok & radiation.scale_separated
                if radiation.created_rest_energy is not None:
                    rest_energy = rest_energy + radiation.created_rest_energy.astype(
                        dtype
                    )
                if radiation.field_exchange_energy is not None:
                    exchange = exchange + radiation.field_exchange_energy.astype(dtype)
        energy = self._energy_ledger(
            state,
            final,
            field,
            field_energy,
            advanced.field,
            dt,
            radiated,
            rest_energy,
            exchange,
            exit_ledger.kinetic_energy,
        )
        total = energy.total
        if self.filters:
            particle_layout_charge = self._filtered_charge(particle_layout_charge)
        medium_charge = jnp.max(
            jnp.abs(self.solver.field_charge(field) - particle_layout_charge),
            initial=0.0,
        )
        charge_before = self._particle_charge(state.species)
        charge_after = self._particle_charge(final)
        charge_ledger_defect = jnp.abs(
            charge_after[0] + exit_ledger.charge - charge_before[0]
        ) / jnp.maximum(charge_before[1], jnp.finfo(charge_before.dtype).tiny)
        stable = (dt <= self.solver.stable_step) & (
            fraction <= self.maximum_displacement_fraction
        )
        cherenkov_ok = jnp.asarray(True)
        for guard in self.cherenkov_guards:
            cherenkov_ok = cherenkov_ok & guard.admissible(
                velocities[guard.species],
                pushed[guard.species].population.active,
                dt,
            )
        finite = (
            jnp.isfinite(dt)
            & (dt > 0.0)
            & jnp.all(
                jnp.stack(
                    tuple(
                        jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(current)
                    )
                )
            )
            & jnp.isfinite(total)
        )
        continuity = deposit.continuity_defect
        # Both residuals are differences of charges of the deposit's size, so
        # their certificate is relative to it and never below dtype roundoff.
        relative = jnp.maximum(
            self.continuity_tolerance, 64.0 * jnp.finfo(continuity.dtype).eps
        )
        charge_scale = (
            deposit.continuity_scale * dt
            + jnp.max(jnp.abs(advanced.charge), initial=0.0)
            + jnp.max(jnp.abs(self.solver.field_charge(state.field)), initial=0.0)
        )
        conserved = (continuity <= relative * deposit.continuity_scale) & (
            charge_defect <= relative * charge_scale
        )
        tolerance = self.constraint_tolerance
        gauss_ok = electric_constraint <= tolerance
        magnetic_ok = advanced.magnetic_constraint <= tolerance
        constraints = conserved & gauss_ok & magnetic_ok
        transfer_success = gather_success & boundary_success
        current_success = deposit.successful
        successful = (
            transfer_success
            & pusher_success
            & current_success
            & advanced.successful
            & process_success
            & ownership_ok
            & cherenkov_ok
            & stable
            & finite
            & constraints
            & exchanged
        )
        flags = (
            (transfer_success, PICRejectionReason.ROUTE),
            (advanced.successful, PICRejectionReason.FIELD),
            (pusher_success, PICRejectionReason.PUSHER),
            (stable, PICRejectionReason.DISPLACEMENT),
            (current_success & conserved, PICRejectionReason.CONTINUITY),
            (gauss_ok, PICRejectionReason.GAUSS),
            (magnetic_ok, PICRejectionReason.MAGNETIC),
            (finite, PICRejectionReason.NONFINITE),
            (process_success, PICRejectionReason.PROCESS),
            (ownership_ok, PICRejectionReason.RADIATION_OWNERSHIP),
            (cherenkov_ok, PICRejectionReason.NUMERICAL_CHERENKOV),
            (exchanged, PICRejectionReason.MIGRATION),
        )
        reason = jnp.asarray(int(PICRejectionReason.NONE), dtype=jnp.int32)
        for passed, flag in flags:
            reason = jnp.where(passed, reason, reason | int(flag))
        next_time = state.time + dt
        next_step = state.accepted_step + jnp.asarray(1, dtype=jnp.int32)
        candidate = ElectromagneticPICState(
            final,
            field,
            surfaces,
            wall,
            tuple(
                recorder.record(value, final, next_time, next_step)
                for recorder, value in zip(self.recorders, state.recorders, strict=True)
            ),
            next_time,
            next_step,
            jnp.where(
                successful, int(PICRunStatus.SUCCESS), int(PICRunStatus.INVALID_STATE)
            ).astype(jnp.int32),
            None
            if state.field_history is None
            else PICFieldHistory(state.field, state.time),
            states,
        )
        accepted = _select(successful, candidate, state)
        diagnostics = ElectromagneticPICDiagnostics(
            continuity,
            charge_defect,
            deposit.continuity_scale,
            process_charge_defect,
            electric_constraint,
            advanced.magnetic_constraint,
            fraction,
            energy,
            advanced.diagnostics,
            ledgers,
            transfer_success,
            current_success,
            pusher_success,
            advanced.successful,
            process_success,
            finite,
            successful,
            reason,
            momentum_evidence + creation_evidence + population_evidence,
            projection,
            exit_ledger,
            medium_charge,
            charge_ledger_defect,
            exchange,
        )
        return ElectromagneticPICStepResult(
            candidate, accepted, diagnostics, current, successful
        )

    # -- restart -----------------------------------------------------------------

    def _boundary_owner(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "pic-particle-boundary-restart",
                "boundaries": None
                if self.boundaries is None
                else self.boundaries.plan_id,
                "solver": self.solver.solver_id,
            }
        )

    def _templates(self) -> dict[str, tuple[str, Any]]:
        dtype = jnp.dtype(self.precision.particle_dtype)
        species = tuple(
            plan.initialize(
                jnp.zeros((plan.capacity, self.solver.spatial_dimension), dtype=dtype),
                jnp.zeros((plan.capacity, 3), dtype=dtype),
            )
            for plan in self.species
        )
        time = jnp.zeros((), dtype=dtype)
        charge, _ = self.species_charge(species)
        templates: dict[str, tuple[str, Any]] = {
            "clock": (
                self.solver.solver_id,
                (time, jnp.zeros((), jnp.int32), jnp.zeros((), jnp.int32)),
            ),
            "boundaries": (
                self._boundary_owner(),
                (
                    ()
                    if self.boundaries is None
                    else tuple(
                        self.boundaries.initialize_surface(dtype) for _ in self.species
                    ),
                    jnp.zeros_like(charge),
                ),
            ),
        }
        for index, (plan, value) in enumerate(zip(self.species, species, strict=True)):
            templates[f"species/{index}"] = (plan.plan_id, value)
        for index, recorder in enumerate(self.recorders):
            templates[f"recorder/{index}"] = (
                recorder.recorder_id,
                recorder.initialize(species, time),
            )
        if self.field_derivatives:
            templates["field-history"] = (
                self.solver.solver_id,
                PICFieldHistory(self.solver.field_with_charge(charge), time),
            )
        for index, template in enumerate(self.initialize_process_states(species)):
            if template is not None:
                templates[f"process/{index}"] = (
                    self.processes[index].process_id,
                    template,
                )
        return templates

    def checkpoint(self, state: ElectromagneticPICState, /) -> PICRestartCheckpoint:
        """Independent restart components; RNG addressing is stateless."""
        solver = self.solver
        if not isinstance(solver, PICRestartState):
            raise TypeError("The PIC field solver does not implement PICRestartState.")
        components = [
            solver.restart_component(state.field),
            restart_component(
                "clock",
                self.solver.solver_id,
                (state.time, state.accepted_step, state.status),
            ),
            restart_component(
                "boundaries",
                self._boundary_owner(),
                (state.boundaries, state.wall_charge),
            ),
        ]
        components.extend(
            restart_component(f"species/{index}", plan.plan_id, value)
            for index, (plan, value) in enumerate(
                zip(self.species, state.species, strict=True)
            )
        )
        components.extend(
            restart_component(f"recorder/{index}", recorder.recorder_id, value)
            for index, (recorder, value) in enumerate(
                zip(self.recorders, state.recorders, strict=True)
            )
        )
        if state.field_history is not None:
            components.append(
                restart_component(
                    "field-history", self.solver.solver_id, state.field_history
                )
            )
        components.extend(
            restart_component(f"process/{index}", process.process_id, value)
            for index, (process, value) in enumerate(
                zip(self.processes, state.processes, strict=True)
            )
            if value is not None
        )
        return PICRestartCheckpoint(tuple(components))

    def restore(self, checkpoint: PICRestartCheckpoint, /) -> ElectromagneticPICState:
        """Admit every component against this plan's owners and rebuild the state."""
        if not isinstance(checkpoint, PICRestartCheckpoint):
            raise TypeError("checkpoint must be PICRestartCheckpoint.")
        solver = self.solver
        if not isinstance(solver, PICRestartState):
            raise TypeError("The PIC field solver does not implement PICRestartState.")
        by_name = {value.name: value for value in checkpoint.components}
        if len(by_name) != len(checkpoint.components):
            raise ValueError("Restart components must have distinct names.")
        templates = self._templates()
        expected = set(templates) | {"field"}
        if set(by_name) != expected:
            raise ValueError(
                "Restart checkpoint components differ from this PIC plan: "
                f"missing {sorted(expected - set(by_name))}, "
                f"unexpected {sorted(set(by_name) - expected)}."
            )
        restored = {
            name: restore_component(by_name[name], name, owner, template)
            for name, (owner, template) in templates.items()
        }
        time, accepted_step, status = restored["clock"]
        boundaries, wall_charge = restored["boundaries"]
        return ElectromagneticPICState(
            tuple(restored[f"species/{index}"] for index in range(len(self.species))),
            solver.restore_component(by_name["field"]),
            boundaries,
            wall_charge,
            tuple(restored[f"recorder/{index}"] for index in range(len(self.recorders))),
            time,
            accepted_step,
            status,
            restored["field-history"] if self.field_derivatives else None,
            tuple(
                restored.get(f"process/{index}") for index in range(len(self.processes))
            ),
        )


class ElectromagneticPICFixedStepMethod(AbstractFixedStepMethod, NonTrainableState):
    plan: ElectromagneticPICPlan
    method_id: str = eqx.field(static=True)

    def __init__(self, plan: ElectromagneticPICPlan, /) -> None:
        if not isinstance(plan, ElectromagneticPICPlan):
            raise TypeError("plan must be ElectromagneticPICPlan.")
        self.plan = plan
        self.method_id = canonical_fingerprint(
            {"kind": "electromagnetic-pic-fixed-step", "plan": plan.plan_id}
        )

    def step(
        self,
        step_index: Array,
        time: Array,
        state: ElectromagneticPICState,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        del step_index, time, args
        result = self.plan.step_detailed(state, step_size)
        residual = jnp.max(
            jnp.stack(
                (
                    result.diagnostics.continuity_defect,
                    result.diagnostics.particle_field_charge_defect,
                    result.diagnostics.electric_constraint,
                    result.diagnostics.magnetic_constraint,
                )
            )
        )
        return FixedStepResult(
            result.candidate_state,
            result.accepted_state,
            result.successful,
            residual,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.zeros((), dtype=state.time.dtype),
        )


__all__ = [
    "ElectromagneticPICDiagnostics",
    "ElectromagneticPICFixedStepMethod",
    "ElectromagneticPICPlan",
    "ElectromagneticPICState",
    "ElectromagneticPICStepResult",
    "PICFieldHistory",
    "PICRestartCheckpoint",
]
