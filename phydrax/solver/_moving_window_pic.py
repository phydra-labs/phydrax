#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Integer-cell moving window over a `PICWindowShift` field solver."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.particle import ParticleAllocationRequest
from ..discretization.pic import PICChargeState, PICParticleState, PICSpeciesState
from ._electromagnetic_pic import (
    ElectromagneticPICPlan,
    ElectromagneticPICState,
    PICFieldHistory,
)
from ._pic_field_solver import PICWindowShift


class PICMovingWindowState(StrictModule):
    pic: ElectromagneticPICState
    origin: Array
    cumulative_cells: Array
    shift_epoch: Array


class PICWindowInjection(StrictModule):
    """Particles created in the leading cells of one species during a shift.

    ``position[W, d]`` are window-local; ``proper_velocity[W, 3]`` are the
    staggered proper velocities of the created particles.
    """

    species: int = eqx.field(static=True)
    request: ParticleAllocationRequest
    position: Array
    proper_velocity: Array


class PICMovingWindowResult(StrictModule):
    candidate_state: PICMovingWindowState
    accepted_state: PICMovingWindowState
    shifted: Array
    outflow_masks: tuple[Array, ...]
    outflow_mass: Array
    outflow_charge: Array
    particle_field_charge_defect: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class PICMovingWindowPlan(StrictModule, NonTrainableState):
    """Shift field, particles, and window origin by whole cells in one transaction.

    The field translates through the solver's `PICWindowShift` capability;
    particles leaving the trailing face are deactivated and ledgered, and
    optional injections fill the leading cells. Every recorder receives the
    shift through `AbstractPICRecorder.shift_frame`, so position-dependent
    diagnostics stay in the fixed frame, and every process state through
    `AbstractPICProcess.shift_frame` (QED photons translate with the window).
    The particle↔field charge defect after the shift is reported, not repaired.
    """

    pic: ElectromagneticPICPlan
    axis: int = eqx.field(static=True)
    shift_cells: int = eqx.field(static=True)
    interval: float = eqx.field(static=True)
    lower: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        pic: ElectromagneticPICPlan,
        axis: int,
        /,
        *,
        shift_cells: int = 1,
    ) -> None:
        if not isinstance(pic, ElectromagneticPICPlan):
            raise TypeError("pic must be ElectromagneticPICPlan.")
        solver = pic.solver
        if not isinstance(solver, PICWindowShift):
            raise TypeError("The PIC field solver does not implement PICWindowShift.")
        if pic.boundaries is not None:
            raise ValueError(
                "Moving windows own their outflow ledger; particle boundaries are "
                "refused."
            )
        selected = int(axis)
        cells = int(shift_cells)
        if selected < 0 or selected >= pic.solver.spatial_dimension or cells <= 0:
            raise ValueError("Moving-window axis/cell shift is invalid.")
        interval = solver.window_interval(selected)
        lower, upper = solver.window_bounds(selected)
        if cells * interval >= upper - lower:
            raise ValueError("Moving-window shift must be shorter than the domain.")
        self.pic = pic
        self.axis = selected
        self.shift_cells = cells
        self.interval = interval
        self.lower = lower
        self.plan_id = canonical_fingerprint(
            {
                "kind": "pic-moving-window",
                "pic": pic.plan_id,
                "axis": selected,
                "shift_cells": cells,
            }
        )

    def initialize(self, pic: ElectromagneticPICState, /) -> PICMovingWindowState:
        return PICMovingWindowState(
            pic,
            jnp.zeros((), dtype=pic.time.dtype),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )

    def _inject(
        self,
        state: PICSpeciesState,
        injection: PICWindowInjection,
        /,
    ) -> tuple[PICSpeciesState, Array]:
        plan = self.pic.species[injection.species]
        width = injection.request.valid.shape[0]
        position = jnp.asarray(injection.position, dtype=state.particles.position.dtype)
        velocity = jnp.asarray(
            injection.proper_velocity, dtype=state.particles.proper_velocity.dtype
        )
        if position.shape != (width, state.particles.position.shape[1]) or (
            velocity.shape != (width, 3)
        ):
            raise ValueError(
                "Moving-window injection payloads must match request capacity."
            )
        allocation = plan.population.allocate(state.population, injection.request)
        slots = jnp.maximum(allocation.slots, 0)
        use = allocation.allocated
        particles = PICParticleState(
            state.particles.position.at[slots].set(
                jnp.where(use[:, None], position, state.particles.position[slots])
            ),
            state.particles.proper_velocity.at[slots].set(
                jnp.where(use[:, None], velocity, state.particles.proper_velocity[slots])
            ),
        )
        charge = PICChargeState(
            state.charge.charge_number.at[slots].set(
                jnp.where(
                    use,
                    plan.charge_model.initial_charge_number,
                    state.charge.charge_number[slots],
                ).astype(state.charge.charge_number.dtype)
            ),
            state.charge.transition_count,
            state.charge.last_transition_step,
        )
        return (
            PICSpeciesState(particles, allocation.accepted_state, charge),
            allocation.successful,
        )

    def shift(
        self,
        state: PICMovingWindowState,
        /,
        *,
        apply_shift: ArrayLike = True,
        injections: Sequence[PICWindowInjection] = (),
    ) -> PICMovingWindowResult:
        solver = self.pic.solver
        if not isinstance(solver, PICWindowShift):
            raise TypeError("The PIC field solver does not implement PICWindowShift.")
        predicate = jnp.asarray(apply_shift, dtype=jnp.bool_).reshape(())
        distance = self.shift_cells * self.interval
        species = []
        masks = []
        masses = []
        charges = []
        successful = jnp.asarray(True)
        for plan, value in zip(self.pic.species, state.pic.species, strict=True):
            position = value.particles.position.at[:, self.axis].add(-distance)
            outflow = value.population.active & (position[:, self.axis] < self.lower)
            deactivated = plan.population.deactivate(value.population, outflow)
            active = deactivated.accepted_state.active[:, None]
            species.append(
                PICSpeciesState(
                    PICParticleState(
                        jnp.where(active, position, 0.0),
                        jnp.where(active, value.particles.proper_velocity, 0.0),
                    ),
                    deactivated.accepted_state,
                    value.charge,
                )
            )
            masks.append(outflow)
            masses.append(jnp.sum(jnp.where(outflow, value.population.mass, 0.0)))
            charges.append(jnp.sum(jnp.where(outflow, plan.macrocharge(value), 0.0)))
            successful = successful & deactivated.successful
        for injection in injections:
            if not isinstance(injection, PICWindowInjection):
                raise TypeError("injections must be PICWindowInjection values.")
            if not 0 <= injection.species < len(species):
                raise ValueError("Injection references a species outside the run.")
            species[injection.species], injected = self._inject(
                species[injection.species], injection
            )
            successful = successful & injected
        species_tuple = tuple(species)
        field = solver.shift_window(state.pic.field, self.axis, self.shift_cells)
        history = state.pic.field_history
        pic = ElectromagneticPICState(
            species_tuple,
            field,
            state.pic.boundaries,
            state.pic.wall_charge,
            tuple(
                recorder.shift_frame(value, self.axis, distance)
                for recorder, value in zip(
                    self.pic.recorders, state.pic.recorders, strict=True
                )
            ),
            state.pic.time,
            state.pic.accepted_step,
            state.pic.status,
            None
            if history is None
            else PICFieldHistory(
                solver.shift_window(history.field, self.axis, self.shift_cells),
                history.time,
            ),
            tuple(
                process.shift_frame(value, self.axis, distance)
                for process, value in zip(
                    self.pic.processes, state.pic.processes, strict=True
                )
            ),
        )
        deposited, deposit_success = self.pic.species_charge(species_tuple)
        charge_defect = jnp.max(
            jnp.abs(self.pic.solver.field_charge(field) - deposited), initial=0.0
        )
        candidate = PICMovingWindowState(
            pic,
            state.origin + distance,
            state.cumulative_cells + self.shift_cells,
            state.shift_epoch + 1,
        )
        finite = jnp.all(
            jnp.stack(
                tuple(
                    jnp.all(jnp.isfinite(leaf))
                    for leaf in jax.tree.leaves((species_tuple, field))
                    if jnp.issubdtype(jnp.result_type(leaf), jnp.inexact)
                )
            )
        )
        successful = successful & deposit_success & finite
        select = predicate & successful
        accepted = jax.tree.map(
            lambda proposed, old: jnp.where(select, proposed, old), candidate, state
        )
        return PICMovingWindowResult(
            candidate,
            accepted,
            select,
            tuple(masks),
            jnp.stack(masses),
            jnp.stack(charges),
            charge_defect,
            finite,
            successful,
            self.plan_id,
        )


__all__ = [
    "PICMovingWindowPlan",
    "PICMovingWindowResult",
    "PICMovingWindowState",
    "PICWindowInjection",
]
