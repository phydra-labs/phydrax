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
from ...solver import AbstractFixedStepMethod, FixedStepResult
from ._boussinesq import PreparedCartesianBoussinesqOcean


class OceanBoussinesqContinuationState(StrictModule):
    """Authoritative packed state plus accepted cumulative ocean budgets."""

    coordinates: Array
    temperature_boundary_content: Array
    salinity_boundary_content: Array
    coriolis_work: Array
    surface_stress_work: Array
    buoyancy_exchange_defect: Array
    energy_balance_defect: Array

    @classmethod
    def initialize(
        cls,
        coordinates: ArrayLike,
        /,
    ) -> "OceanBoussinesqContinuationState":
        value = jnp.asarray(coordinates)
        zero = jnp.zeros((), dtype=value.dtype)
        return cls(value, zero, zero, zero, zero, zero, zero)


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
    ) -> tuple[OceanBoussinesqContinuationState, Array, Array]:
        stage = self.ocean.dynamics.stage(time, state.coordinates, args)
        diagnostics = self.ocean.dynamics.diagnostics(time, state.coordinates, args)
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
        )
        residual = jnp.maximum(
            diagnostics.divergence_norm,
            jnp.maximum(
                stage.buoyancy.normalized_exchange_defect,
                jnp.abs(diagnostics.energy_balance_defect),
            ),
        )
        success = stage.success & diagnostics.success
        return rate, success, residual

    @staticmethod
    def _axpy(
        base: OceanBoussinesqContinuationState,
        scale: Array,
        rate: OceanBoussinesqContinuationState,
        /,
    ) -> OceanBoussinesqContinuationState:
        return jax.tree.map(lambda value, slope: value + scale * slope, base, rate)

    @staticmethod
    def _combine(
        first_weight: float,
        first: OceanBoussinesqContinuationState,
        second_weight: float,
        second: OceanBoussinesqContinuationState,
        /,
    ) -> OceanBoussinesqContinuationState:
        return jax.tree.map(
            lambda left, right: first_weight * left + second_weight * right,
            first,
            second,
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
        dt = jnp.asarray(step_size, dtype=state.coordinates.dtype)
        first_rate, first_success, first_residual = self._rate(time, state, args)
        first = self._axpy(state, dt, first_rate)
        second_rate, second_success, second_residual = self._rate(time + dt, first, args)
        second_euler = self._axpy(first, dt, second_rate)
        second = self._combine(0.75, state, 0.25, second_euler)
        third_rate, third_success, third_residual = self._rate(
            time + 0.5 * dt, second, args
        )
        third_euler = self._axpy(second, dt, third_rate)
        candidate = self._combine(1.0 / 3.0, state, 2.0 / 3.0, third_euler)
        successful = first_success & second_success & third_success
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
