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
3. drift, apply particle boundaries;
4. deposit charge-conserving current along each path and advance the field;
5. population-stage processes on the end-of-step species, with pointwise
   charge preservation verified by redeposition;
6. commit or reject the whole candidate; accepted recorders advance.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._sampling import derive_key, SampleAddress
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.pic import (
    AbstractPICProcess,
    AbstractPICRecorder,
    ExternalFieldSource,
    PIC_CODE_RELATIVITY,
    PICBoundarySurfaceState,
    PICEnergyLedger,
    PICOpenBoundaryPlan,
    PICParticleState,
    PICProcessContext,
    PICProcessLedger,
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
    PICFieldDeposit,
    PICFilterContinuityReport,
    PICGaussProjection,
    PICGaussProjectionResult,
    PICMultiDeposit,
    PICPrecisionPolicy,
    PICRestartComponent,
    PICRestartState,
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
    requires field derivatives).
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


class ElectromagneticPICDiagnostics(StrictModule):
    """Step evidence.

    When a charge-redistributing population process (resampling) ran,
    ``process_charge_defect`` is the redistributed charge it moved and
    ``gauss_projection`` holds the field's Poisson projection onto the
    redeposited charge; ``electric_constraint`` is then the projected field's
    Gauss residual. ``process_evidence`` holds each process's own evidence in
    ledger order.
    """

    continuity_defect: Array
    particle_field_charge_defect: Array
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
        case "resolved-field" | "diagnostic-only":
            if subgrid:
                raise ValueError(
                    f"A subgrid-reaction process overlaps {declared!r} ownership."
                )
        case _:
            raise ValueError("ownership is invalid.")
    return declared


class ElectromagneticPICPlan(StrictModule, NonTrainableState):
    """Explicit electromagnetic PIC over one prepared PIC field solver."""

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
            if not isinstance(boundaries, PICOpenBoundaryPlan):
                raise TypeError("boundaries must be PICOpenBoundaryPlan or None.")
            if boundaries.lower.size != solver.spatial_dimension:
                raise ValueError("Particle boundaries must match the field dimension.")
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

    def _gather(
        self,
        species: tuple[PICSpeciesState, ...],
        field: Any,
        time: Array,
        /,
    ) -> tuple[_Gathered, ...]:
        view = self._filtered_field(field)
        gathered = []
        for index, state in enumerate(species):
            active = state.population.active
            sample = self.solver.gather(index, state.particles.position, active, view)
            electric, magnetic, successful = (
                sample.electric,
                sample.magnetic,
                sample.successful,
            )
            if self.external_fields:
                position = state.particles.position
                position3 = jnp.pad(position, ((0, 0), (0, 3 - position.shape[1])))
                times = jnp.full((position.shape[0],), time, dtype=position.dtype)
                for source in self.external_fields:
                    external = source.external_fields(position3, times)
                    electric = electric + jnp.where(
                        active[:, None], external.electric, 0.0
                    )
                    magnetic = magnetic + jnp.where(
                        active[:, None], external.magnetic, 0.0
                    )
                    successful = successful & jnp.all(external.support | ~active)
            gathered.append(_Gathered(electric, magnetic, successful))
        return tuple(gathered)

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

    def _kinetic(self, species: tuple[PICSpeciesState, ...], /) -> Array:
        dtype = jnp.dtype(self.precision.accumulation_dtype)
        total = jnp.asarray(0.0, dtype=dtype)
        c2 = self.pusher.speed_of_light**2
        for state in species:
            proper = state.particles.proper_velocity.astype(dtype)
            gamma = jnp.sqrt(1.0 + jnp.sum(proper**2, axis=-1) / c2)
            total = total + jnp.sum(
                jnp.where(
                    state.population.active,
                    state.population.mass.astype(dtype) * c2 * (gamma - 1.0),
                    0.0,
                )
            )
        return total

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
            position3 = jnp.pad(position, ((0, 0), (0, 3 - dimension)))
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
        gathered: tuple[_Gathered, ...],
        time: Array,
        dt: Array,
        step: Array,
        derivatives: _FieldDerivatives | None,
        /,
    ) -> tuple[
        tuple[PICSpeciesState, ...], tuple[PICProcessLedger, ...], tuple[Any, ...]
    ]:
        ledgers = []
        evidence = []
        cutoff = self._grid_cutoff_frequency(dt)
        for process in self.processes:
            if process.stage != stage:
                continue
            supplied = derivatives if process.requires_field_derivatives else None
            result = process.apply(
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
        return species, tuple(ledgers), tuple(evidence)

    # -- lifecycle ---------------------------------------------------------------

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
    ) -> ElectromagneticPICState:
        """Gauss-consistent initial state from positions and physical velocities.

        Proper velocities are bootstrapped half a step backward in the initial
        field so the pusher sees leapfrog-staggered momenta.
        """
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
        charge, charge_success = self.species_charge(species_tuple)
        field, field_success = self.solver.initialize_field(
            self._filtered_charge(charge), magnetic=magnetic
        )
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
        starts, ends, means, charges, actives = [], [], [], [], []
        for index, (plan, value, velocity) in enumerate(
            zip(self.species, species, velocities, strict=True)
        ):
            active = value.population.active
            start = value.particles.position
            displacement = dt * velocity[:, :dimension]
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
            )
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
        )

    def step_detailed(
        self, state: ElectromagneticPICState, step_size: ArrayLike, /
    ) -> ElectromagneticPICStepResult:
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
        pushed, momentum_ledgers, momentum_evidence = self._run_processes(
            "momentum", pushed, gathered, state.time, dt, step, derivatives
        )
        if momentum_ledgers:
            velocities = tuple(
                self.pusher.velocity(value.particles.proper_velocity) for value in pushed
            )
        moved, deposit, surfaces, wall, fraction, boundary_success = self._move(
            state, pushed, velocities, dt
        )
        current = self._filtered_current(deposit.current)
        advanced = self.solver.advance(state.time, state.field, current, dt)
        expected_charge = deposit.end_charge + state.wall_charge
        if self.filters:
            expected_charge = self._filtered_charge(expected_charge)
        charge_defect = jnp.max(jnp.abs(advanced.charge - expected_charge), initial=0.0)
        final, population_ledgers, population_evidence = self._run_processes(
            "population", moved, gathered, state.time, dt, step, None
        )
        field = advanced.field
        electric_constraint = advanced.electric_constraint
        field_energy = advanced.energy
        projection = None
        process_charged = jnp.asarray(True)
        if population_ledgers:
            before, _ = self.species_charge(moved)
            after, redeposit_success = self.species_charge(final)
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
        ledgers = momentum_ledgers + population_ledgers
        process_success = process_charged
        for ledger in ledgers:
            process_success = process_success & ledger.successful
        dtype = jnp.dtype(self.precision.accumulation_dtype)
        radiated = jnp.zeros((), dtype=dtype)
        ownership_ok = jnp.asarray(True)
        for ledger in ledgers:
            if ledger.radiation is not None:
                radiated = radiated + ledger.radiation.radiated_energy.astype(dtype)
                ownership_ok = ownership_ok & ledger.radiation.scale_separated
        previous_kinetic = self._kinetic(state.species)
        next_kinetic = self._kinetic(final)
        previous_total = previous_kinetic + self.solver.field_energy(state.field)
        total = next_kinetic + field_energy
        energy = PICEnergyLedger(
            next_kinetic,
            field_energy,
            jnp.zeros((), dtype=total.dtype),
            radiated,
            total,
            previous_total,
            total + radiated - previous_total,
        )
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
        tolerance = self.constraint_tolerance
        gauss_ok = electric_constraint <= tolerance
        magnetic_ok = advanced.magnetic_constraint <= tolerance
        constraints = (
            (continuity <= self.continuity_tolerance)
            & (charge_defect <= self.continuity_tolerance)
            & gauss_ok
            & magnetic_ok
        )
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
        )
        flags = (
            (transfer_success, PICRejectionReason.ROUTE),
            (advanced.successful, PICRejectionReason.FIELD),
            (pusher_success, PICRejectionReason.PUSHER),
            (stable, PICRejectionReason.DISPLACEMENT),
            (
                current_success
                & (continuity <= self.continuity_tolerance)
                & (charge_defect <= self.continuity_tolerance),
                PICRejectionReason.CONTINUITY,
            ),
            (gauss_ok, PICRejectionReason.GAUSS),
            (magnetic_ok, PICRejectionReason.MAGNETIC),
            (finite, PICRejectionReason.NONFINITE),
            (process_success, PICRejectionReason.PROCESS),
            (ownership_ok, PICRejectionReason.RADIATION_OWNERSHIP),
            (cherenkov_ok, PICRejectionReason.NUMERICAL_CHERENKOV),
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
        )
        accepted = _select(successful, candidate, state)
        diagnostics = ElectromagneticPICDiagnostics(
            continuity,
            charge_defect,
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
            momentum_evidence + population_evidence,
            projection,
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
