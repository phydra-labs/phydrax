#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..equations._relativistic_multigroup_radiation import (
    GRMultigroupM1RadiationSystem,
)
from ..metrix._adm_exchange import StressEnergyProjection
from ._gr_m1_finite_volume import (
    FixedGridGRM1SSPRK3Plan,
    GRM1StepResult,
)
from ._relativistic_finite_volume import ValenciaFiniteVolumeStageGeometry


class GRMultigroupM1State(StrictModule):
    densitized_moments: Array
    time: Array
    accepted_steps: Array


class GRMultigroupM1StepResult(StrictModule):
    candidate: GRMultigroupM1State
    state: GRMultigroupM1State
    group_results: tuple[GRM1StepResult, ...]
    stress_energy: StressEnergyProjection
    accepted: Array
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    plan_id: str = eqx.field(static=True)


class FixedGridGRMultigroupM1SSPRK3Plan(StrictModule, NonTrainableState):
    """Independent spatial M1 transport with one retained state per frequency group."""

    system: GRMultigroupM1RadiationSystem
    groups: tuple[FixedGridGRM1SSPRK3Plan, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: GRMultigroupM1RadiationSystem,
        groups: tuple[FixedGridGRM1SSPRK3Plan, ...],
        /,
    ) -> None:
        if not isinstance(system, GRMultigroupM1RadiationSystem):
            raise TypeError("system must be GRMultigroupM1RadiationSystem.")
        plans = tuple(groups)
        if len(plans) != system.group_count or any(
            not isinstance(value, FixedGridGRM1SSPRK3Plan) for value in plans
        ):
            raise TypeError("One fixed-grid M1 plan is required per frequency group.")
        first = plans[0]
        if any(
            value.discretization.prepared_id != first.discretization.prepared_id
            for value in plans[1:]
        ):
            raise ValueError("Multigroup M1 plans must share one discretization.")
        for radiation_system, plan in zip(system.groups, plans, strict=True):
            if plan.system.system_id != radiation_system.system_id:
                raise ValueError("Multigroup equation and transport systems differ.")
        self.system = system
        self.groups = plans
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-grid-gr-multigroup-m1-ssprk3",
                "system": system.system_id,
                "groups": [value.plan_id for value in plans],
            }
        )

    @property
    def cell_shape(self) -> tuple[int, ...]:
        return self.groups[0].cell_shape

    def initialize(
        self,
        moments: ArrayLike,
        geometry: ValenciaFiniteVolumeStageGeometry,
        /,
        *,
        time: ArrayLike = 0.0,
        step_size: ArrayLike | None = None,
    ) -> GRMultigroupM1State:
        grouped = self.system.group_moments(moments)
        states = tuple(
            plan.initialize(
                grouped[..., index, :],
                geometry,
                time=time,
                step_size=step_size,
            )
            for index, plan in enumerate(self.groups)
        )
        densitized = jnp.stack(tuple(value.radiation_state for value in states), axis=-2)
        return GRMultigroupM1State(
            densitized,
            states[0].time,
            jnp.zeros((), dtype=jnp.int32),
        )

    def advance(
        self,
        state: GRMultigroupM1State,
        start_time: ArrayLike,
        end_time: ArrayLike,
        stage_geometries: tuple[
            ValenciaFiniteVolumeStageGeometry,
            ValenciaFiniteVolumeStageGeometry,
            ValenciaFiniteVolumeStageGeometry,
        ],
        /,
        *,
        transport_extinction: ArrayLike = 0.0,
    ) -> GRMultigroupM1StepResult:
        if not isinstance(state, GRMultigroupM1State):
            raise TypeError("state must be GRMultigroupM1State.")
        expected = self.cell_shape + (self.system.group_count, 4)
        if state.densitized_moments.shape != expected:
            raise ValueError(f"Multigroup densitized moments must have shape {expected}.")
        extinction = jnp.asarray(transport_extinction)
        if extinction.shape in ((), (self.system.group_count,)):
            extinction = jnp.broadcast_to(
                extinction, self.cell_shape + (self.system.group_count,)
            )
        elif extinction.shape != self.cell_shape + (self.system.group_count,):
            raise ValueError("Multigroup transport extinction has invalid shape.")
        results = []
        for index, plan in enumerate(self.groups):
            group_state = plan.initialize(
                state.densitized_moments[..., index, :]
                / stage_geometries[0].cell.sqrt_det_spatial_metric[..., None],
                stage_geometries[0],
                time=state.time,
            )
            group_state = eqx.tree_at(
                lambda value: value.radiation_state,
                group_state,
                state.densitized_moments[..., index, :],
            )
            group_state = eqx.tree_at(
                lambda value: value.accepted_step,
                group_state,
                state.accepted_steps,
            )
            results.append(
                plan.advance(
                    group_state,
                    start_time,
                    end_time,
                    stage_geometries,
                    transport_extinction=extinction[..., index],
                )
            )
        group_results = tuple(results)
        candidate_moments = jnp.stack(
            tuple(value.candidate.radiation_state for value in group_results), axis=-2
        )
        accepted = jnp.all(jnp.stack(tuple(value.accepted for value in group_results)))
        accepted_moments = jnp.where(
            accepted, candidate_moments, state.densitized_moments
        )
        end = jnp.asarray(end_time, dtype=state.time.dtype)
        candidate = GRMultigroupM1State(
            candidate_moments,
            end,
            state.accepted_steps + jnp.asarray(1, dtype=jnp.int32),
        )
        accepted_state = GRMultigroupM1State(
            accepted_moments,
            jnp.where(accepted, end, state.time),
            state.accepted_steps + accepted.astype(jnp.int32),
        )
        local = (
            accepted_moments
            / stage_geometries[-1].cell.sqrt_det_spatial_metric[..., None, None]
        )
        projection = self.system.stress_energy_projection(
            self.system.flatten_groups(local), stage_geometries[-1].cell
        )
        finite = jnp.all(jnp.stack(tuple(value.finite for value in group_results)))
        physical = jnp.all(
            jnp.stack(tuple(value.physically_valid for value in group_results))
        )
        qualified = jnp.all(jnp.stack(tuple(value.qualified for value in group_results)))
        derivative = jnp.all(
            jnp.stack(tuple(value.derivative_valid for value in group_results))
        )
        return GRMultigroupM1StepResult(
            candidate,
            accepted_state,
            group_results,
            projection,
            accepted,
            finite,
            physical,
            qualified,
            derivative,
            self.plan_id,
        )


__all__ = [
    "FixedGridGRMultigroupM1SSPRK3Plan",
    "GRMultigroupM1State",
    "GRMultigroupM1StepResult",
]
