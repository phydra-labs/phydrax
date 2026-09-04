#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._array_archive import (
    pack_array_tree,
    read_array_archive,
    unpack_array_tree,
    write_array_archive,
)
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...equations._ksgs import KSGSState, replace_ksgs_kinetic_energy
from ...solver import AbstractFixedStepMethod, FixedStepResult
from ._boussinesq import PreparedCartesianBoussinesqOcean


class OceanBoussinesqContinuationState(StrictModule):
    """Authoritative packed state, KSGS history, and accepted ocean budgets."""

    coordinates: Array
    temperature_boundary_content: Array
    salinity_boundary_content: Array
    coriolis_work: Array
    surface_stress_work: Array
    buoyancy_exchange_defect: Array
    energy_balance_defect: Array
    molecular_potential_energy_mixing: Array
    sgs_potential_energy_mixing: Array
    boundary_potential_energy: Array
    ksgs_state: KSGSState | None

    @classmethod
    def initialize(
        cls,
        coordinates: ArrayLike,
        /,
        *,
        ksgs_state: KSGSState | None = None,
    ) -> "OceanBoussinesqContinuationState":
        value = jnp.asarray(coordinates)
        if ksgs_state is not None and not isinstance(ksgs_state, KSGSState):
            raise TypeError("ksgs_state must be KSGSState or None.")
        zero = jnp.zeros((), dtype=value.dtype)
        return cls(
            value,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            ksgs_state,
        )


class OceanBoussinesqSSPRK33Method(AbstractFixedStepMethod):
    """Fail-closed coupled SSPRK3 with accepted budget quadrature."""

    ocean: PreparedCartesianBoussinesqOcean
    method_id: str = eqx.field(static=True)

    def __init__(self, ocean: PreparedCartesianBoussinesqOcean, /):
        if not isinstance(ocean, PreparedCartesianBoussinesqOcean):
            raise TypeError("ocean must be PreparedCartesianBoussinesqOcean.")
        self.ocean = ocean
        self.method_id = canonical_fingerprint(
            {
                "kind": "ocean-boussinesq-ssprk33",
                "ocean": ocean.prepared_id,
                "stage_evidence": "projection-buoyancy-coriolis-scalar",
            }
        )

    def _rate(
        self,
        time: Array,
        state: OceanBoussinesqContinuationState,
        args: Any,
        /,
        *,
        accept_ksgs_update: ArrayLike = False,
    ) -> tuple[
        OceanBoussinesqContinuationState,
        KSGSState | None,
        Array,
        Array,
    ]:
        stage = self.ocean.dynamics.stage(
            time,
            state.coordinates,
            args,
            ksgs_state=state.ksgs_state,
            accept_ksgs_update=accept_ksgs_update,
        )
        diagnostics = self.ocean.dynamics.diagnostics_from_stage(stage)
        velocity_rate = self.ocean.operators.velocity_space.flatten(stage.velocity_rate)
        scalar_rate = self.ocean.transport.layout.pack(stage.scalar_rates)
        coordinate_rate = jnp.concatenate((velocity_rate, scalar_rate))
        temperature = diagnostics.scalars.fields[
            self.ocean.plan.reference.temperature_name
        ]
        salinity = diagnostics.scalars.fields[self.ocean.plan.reference.salinity_name]
        ocean_forcing = stage.ocean_forcing
        coriolis_power = (
            jnp.asarray(0.0, dtype=coordinate_rate.dtype)
            if ocean_forcing is None
            else ocean_forcing.coriolis_power
        )
        stress_power = (
            jnp.asarray(0.0, dtype=coordinate_rate.dtype)
            if ocean_forcing is None
            else ocean_forcing.surface_stress_power
        )
        rate = OceanBoussinesqContinuationState(
            coordinate_rate,
            temperature.diffusive_content_rate,
            salinity.diffusive_content_rate,
            coriolis_power,
            stress_power,
            stage.buoyancy.exchange_defect,
            diagnostics.energy_balance_defect,
            stage.buoyancy.molecular_potential_energy_mixing,
            stage.buoyancy.sgs_potential_energy_mixing,
            stage.buoyancy.boundary_potential_energy_rate,
            None,
        )
        residual = jnp.maximum(
            diagnostics.divergence_norm,
            jnp.maximum(
                stage.buoyancy.normalized_exchange_defect,
                jnp.abs(diagnostics.energy_balance_defect),
            ),
        )
        success = stage.success & diagnostics.success
        next_ksgs = None if stage.ksgs is None else stage.ksgs.result.state
        return rate, next_ksgs, success, residual

    @staticmethod
    def _axpy(
        base: OceanBoussinesqContinuationState,
        scale: Array,
        rate: OceanBoussinesqContinuationState,
        /,
    ) -> OceanBoussinesqContinuationState:
        base_without_ksgs = eqx.tree_at(lambda value: value.ksgs_state, base, None)
        advanced = jax.tree.map(
            lambda value, slope: value + scale * slope,
            base_without_ksgs,
            rate,
        )
        return eqx.tree_at(
            lambda value: value.ksgs_state,
            advanced,
            base.ksgs_state,
        )

    @staticmethod
    def _combine(
        first_weight: float,
        first: OceanBoussinesqContinuationState,
        second_weight: float,
        second: OceanBoussinesqContinuationState,
        /,
    ) -> OceanBoussinesqContinuationState:
        first_without_ksgs = eqx.tree_at(lambda value: value.ksgs_state, first, None)
        second_without_ksgs = eqx.tree_at(lambda value: value.ksgs_state, second, None)
        combined = jax.tree.map(
            lambda left, right: first_weight * left + second_weight * right,
            first_without_ksgs,
            second_without_ksgs,
        )
        return eqx.tree_at(
            lambda value: value.ksgs_state,
            combined,
            first.ksgs_state,
        )

    def _kinetic_nonnegative(
        self,
        state: OceanBoussinesqContinuationState,
        /,
    ) -> Array:
        field_name = self.ocean.plan.ksgs_field_name
        if field_name is None:
            return jnp.asarray(True)
        _, scalars = self.ocean.dynamics.unpack_state(state.coordinates)
        return jnp.all(scalars[field_name] >= 0.0)

    def _safe_stage_state(
        self,
        proposed: OceanBoussinesqContinuationState,
        current: OceanBoussinesqContinuationState,
        valid: Array,
        /,
    ) -> OceanBoussinesqContinuationState:
        if self.ocean.plan.ksgs_field_name is None:
            return proposed
        return jax.tree.map(
            lambda candidate, accepted: jnp.where(valid, candidate, accepted),
            proposed,
            current,
        )

    def _initialize_ksgs(
        self,
        state: OceanBoussinesqContinuationState,
        /,
    ) -> OceanBoussinesqContinuationState:
        prepared = self.ocean.prepared_ksgs
        if prepared is None or state.ksgs_state is not None:
            return state
        _, scalars = self.ocean.dynamics.unpack_state(state.coordinates)
        initialized = prepared.plan.initialize_state(scalars[prepared.scalar_field_name])
        return eqx.tree_at(lambda value: value.ksgs_state, state, initialized)

    def _candidate_ksgs_state(
        self,
        candidate: OceanBoussinesqContinuationState,
        proposed: KSGSState | None,
        /,
    ) -> KSGSState | None:
        prepared = self.ocean.prepared_ksgs
        if prepared is None:
            return None
        if proposed is None:
            raise ValueError("Ocean KSGS stage did not produce continuation state.")
        _, scalars = self.ocean.dynamics.unpack_state(candidate.coordinates)
        return replace_ksgs_kinetic_energy(
            proposed,
            scalars[prepared.scalar_field_name],
        )

    def step(
        self,
        step_index: Array,
        time: Array,
        state: OceanBoussinesqContinuationState,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        del step_index
        if not isinstance(state, OceanBoussinesqContinuationState):
            raise TypeError("Ocean SSPRK3 requires OceanBoussinesqContinuationState.")
        state = self._initialize_ksgs(state)
        dt = jnp.asarray(step_size, dtype=state.coordinates.dtype)
        first_rate, _, first_success, first_residual = self._rate(time, state, args)
        first = self._axpy(state, dt, first_rate)
        first_positive = self._kinetic_nonnegative(first)
        safe_first = self._safe_stage_state(first, state, first_positive)
        second_rate, _, second_success, second_residual = self._rate(
            time + dt,
            safe_first,
            args,
        )
        second_euler = self._axpy(safe_first, dt, second_rate)
        second = self._combine(0.75, state, 0.25, second_euler)
        second_positive = self._kinetic_nonnegative(second)
        safe_second = self._safe_stage_state(second, state, second_positive)
        third_rate, third_ksgs, third_success, third_residual = self._rate(
            time + 0.5 * dt,
            safe_second,
            args,
            accept_ksgs_update=True,
        )
        third_euler = self._axpy(safe_second, dt, third_rate)
        candidate = self._combine(1.0 / 3.0, state, 2.0 / 3.0, third_euler)
        candidate = eqx.tree_at(
            lambda value: value.ksgs_state,
            candidate,
            self._candidate_ksgs_state(candidate, third_ksgs),
        )
        candidate_positive = self._kinetic_nonnegative(candidate)
        successful = (
            first_success
            & first_positive
            & second_success
            & second_positive
            & third_success
            & candidate_positive
        )
        accepted = jax.tree.map(
            lambda proposed, current: jnp.where(successful, proposed, current),
            candidate,
            state,
        )
        residual = jnp.maximum(
            first_residual,
            jnp.maximum(second_residual, third_residual),
        )
        return FixedStepResult(
            candidate_state=candidate,
            accepted_state=accepted,
            successful=successful,
            residual=residual,
            iterations=jnp.asarray(3, dtype=jnp.int32),
            work=jnp.asarray(3, dtype=jnp.int32),
            transform_applied=jnp.asarray(False),
            transform_correction_norm=jnp.zeros((), dtype=state.coordinates.dtype),
        )


def write_ocean_checkpoint(
    path: str | Path,
    ocean: PreparedCartesianBoussinesqOcean,
    time: ArrayLike,
    accepted_step: ArrayLike,
    state: OceanBoussinesqContinuationState,
    /,
) -> Path:
    if not isinstance(ocean, PreparedCartesianBoussinesqOcean):
        raise TypeError("ocean must be PreparedCartesianBoussinesqOcean.")
    if not isinstance(state, OceanBoussinesqContinuationState):
        raise TypeError("state must be OceanBoussinesqContinuationState.")
    arrays: dict[str, object] = {
        "time": jnp.asarray(time),
        "accepted_step": jnp.asarray(accepted_step),
    }
    state_specification = pack_array_tree("state", state, arrays)
    manifest = {
        "kind": "ocean-boussinesq-checkpoint",
        "ocean_id": ocean.prepared_id,
        "state": state_specification,
    }
    return write_array_archive(path, manifest=manifest, arrays=arrays)


def read_ocean_checkpoint(
    path: str | Path,
    ocean: PreparedCartesianBoussinesqOcean,
    template: OceanBoussinesqContinuationState,
    /,
) -> tuple[Array, Array, OceanBoussinesqContinuationState]:
    manifest, arrays = read_array_archive(path)
    if manifest.get("kind") != "ocean-boussinesq-checkpoint":
        raise ValueError("Archive is not an ocean Boussinesq checkpoint.")
    if manifest.get("ocean_id") != ocean.prepared_id:
        raise ValueError("Ocean checkpoint identity does not match prepared model.")
    restored = unpack_array_tree(manifest["state"], arrays, template)
    return (
        jnp.asarray(arrays["time"]),
        jnp.asarray(arrays["accepted_step"]),
        restored,
    )


__all__ = [
    "OceanBoussinesqContinuationState",
    "OceanBoussinesqSSPRK33Method",
    "read_ocean_checkpoint",
    "write_ocean_checkpoint",
]
