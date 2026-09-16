#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..equations._resistive_grmhd import (
    RelativisticOhmEvaluation,
    ResistiveGRMHDOhmicClosure,
)
from ._grrmhd_runtime import (
    FixedGridGRRMHDIMEXPlan,
    GRRMHDState,
    GRRMHDStepResult,
)
from ._relativistic_finite_volume import ValenciaFiniteVolumeStageGeometry


class ResistiveGRRMHDRunStatus(IntEnum):
    SUCCESS = 0
    BASE_GRRMHD_FAILED = 1
    OHMIC_CLOSURE_FAILED = 2
    MATERIAL_RECOVERY_FAILED = 3
    CHARGE_BALANCE_FAILED = 4
    NONFINITE_STATE = 5


class ResistiveGRRMHDState(StrictModule):
    grrmhd: GRRMHDState
    electric_covector: Array
    densitized_charge: Array
    status: Array


class ResistiveGRRMHDLedger(StrictModule):
    electric_energy_change: Array
    material_energy_change: Array
    energy_balance_residual: Array
    charge_change: Array
    boundary_charge_flux: Array
    charge_balance_residual: Array
    entropy_production: Array
    accepted: Array
    finite: Array
    qualified: Array
    plan_id: str = eqx.field(static=True)


class ResistiveGRRMHDStepResult(StrictModule):
    candidate: ResistiveGRRMHDState
    state: ResistiveGRRMHDState
    base: GRRMHDStepResult
    ohm_before: RelativisticOhmEvaluation
    ohm_after: RelativisticOhmEvaluation
    ledger: ResistiveGRRMHDLedger
    accepted: Array
    status: Array
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    plan_id: str = eqx.field(static=True)


class FixedGridResistiveGRRMHDIMEXPlan(StrictModule, NonTrainableState):
    """Atomic GRRMHD step plus an implicit covariant Ohmic relaxation.

    Charge obeys a conservative finite-volume continuity update. Bounded axes use
    an insulating charge boundary; periodic axes telescope exactly. The stiff
    conductive current is backward-Euler solved analytically, including the
    velocity-parallel projection in the relativistic Ohm law.
    """

    base: FixedGridGRRMHDIMEXPlan
    ohm: ResistiveGRMHDOhmicClosure
    balance_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: FixedGridGRRMHDIMEXPlan,
        ohm: ResistiveGRMHDOhmicClosure,
        /,
        *,
        balance_tolerance: float = 1.0e-9,
    ) -> None:
        if not isinstance(base, FixedGridGRRMHDIMEXPlan):
            raise TypeError("base must be FixedGridGRRMHDIMEXPlan.")
        if not isinstance(ohm, ResistiveGRMHDOhmicClosure):
            raise TypeError("ohm must be ResistiveGRMHDOhmicClosure.")
        material = base.material_transport.system
        if (
            ohm.scale.scale_id != material.scale.scale_id
            or ohm.convention.convention_id != material.convention.convention_id
        ):
            raise ValueError("Ohmic closure and GRRMHD material contracts differ.")
        tolerance = float(balance_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("Resistive GRRMHD balance tolerance is invalid.")
        self.base = base
        self.ohm = ohm
        self.balance_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-grid-resistive-grrmhd-imex",
                "base": base.plan_id,
                "ohm": ohm.closure_id,
                "balance_tolerance": tolerance,
            }
        )

    @property
    def cell_shape(self) -> tuple[int, ...]:
        return self.base.cell_shape

    def _primitive(
        self,
        state: GRRMHDState,
        geometry: ValenciaFiniteVolumeStageGeometry,
        composition: ArrayLike | None,
        /,
    ):
        transport = self.base.material_transport
        full = transport.constrained_transport.full_state(
            state.material_state, state.constrained_transport.magnetic_flux
        )
        return transport.system.recover(full, geometry.cell, composition)

    def initialize(
        self,
        material_conserved: ArrayLike,
        radiation_moments: ArrayLike,
        electric_covector: ArrayLike,
        charge_density: ArrayLike,
        geometry: ValenciaFiniteVolumeStageGeometry,
        /,
        *,
        magnetic_flux: ArrayLike | None = None,
        vector_potential: ArrayLike | None = None,
        gauge_scalar: ArrayLike | None = None,
        composition: ArrayLike | None = None,
        time: ArrayLike = 0.0,
        step_size: ArrayLike | None = None,
    ) -> ResistiveGRRMHDState:
        base = self.base.initialize(
            material_conserved,
            radiation_moments,
            geometry,
            magnetic_flux=magnetic_flux,
            vector_potential=vector_potential,
            gauge_scalar=gauge_scalar,
            composition=composition,
            time=time,
            step_size=step_size,
        )
        electric = jnp.asarray(electric_covector, dtype=base.material_state.dtype)
        charge = jnp.asarray(charge_density, dtype=base.material_state.dtype)
        if electric.shape != self.cell_shape + (3,) or charge.shape != self.cell_shape:
            raise ValueError("Initial resistive fields must match the GRRMHD grid.")
        recovery = self._primitive(base, geometry, composition)
        evaluation = self.ohm.evaluate(
            electric,
            recovery.primitive[..., 5:8],
            recovery.primitive[..., 1:4],
            charge,
            geometry.cell,
        )
        valid = jnp.all(recovery.qualified) & jnp.all(evaluation.qualified)
        electric = eqx.error_if(
            electric, ~valid, "Initial resistive GRRMHD state is unqualified."
        )
        return ResistiveGRRMHDState(
            base,
            electric,
            geometry.cell.sqrt_det_spatial_metric * charge,
            jnp.asarray(int(ResistiveGRRMHDRunStatus.SUCCESS), dtype=jnp.int32),
        )

    def _implicit_electric(
        self,
        electric: Array,
        ideal: Array,
        velocity: Array,
        lorentz: Array,
        lapse: Array,
        step: Array,
        metric: Array,
        /,
    ) -> Array:
        light_speed = jnp.asarray(self.ohm.speed_of_light, dtype=electric.dtype)
        normalized_velocity = velocity / light_speed
        velocity_covector = contract(
            "...ij,...j->...i", metric, normalized_velocity, backend="jax"
        )
        speed_squared = contract(
            "...i,...i->...", normalized_velocity, velocity_covector, backend="jax"
        )
        relaxation = (
            step
            * lapse
            * jnp.asarray(self.ohm.conductivity, dtype=electric.dtype)
            * lorentz
        )
        diagonal = 1.0 + relaxation
        right = electric + relaxation[..., None] * ideal
        velocity_right = contract(
            "...i,...i->...", normalized_velocity, right, backend="jax"
        )
        denominator = diagonal - relaxation * speed_squared
        correction = (
            relaxation
            * velocity_right
            / jnp.maximum(diagonal * denominator, jnp.finfo(electric.dtype).tiny)
        )
        return right / diagonal[..., None] + correction[..., None] * velocity_covector

    def _charge_rate(
        self,
        coordinate_current: Array,
        discretization,
        /,
    ) -> tuple[Array, Array]:
        volumes = discretization.cell_volumes.astype(coordinate_current.dtype)
        rate = jnp.zeros(self.cell_shape, dtype=coordinate_current.dtype)
        boundary_flux = jnp.zeros((), dtype=coordinate_current.dtype)
        for axis, grid_axis in enumerate(discretization.grid.structured_axes):
            current = coordinate_current[..., axis]
            measure = discretization.face_measures[axis].astype(current.dtype)
            if grid_axis.periodic:
                face = 0.5 * (current + jnp.roll(current, -1, axis=axis))
                integrated = face * measure
                rate = rate - (integrated - jnp.roll(integrated, 1, axis=axis)) / volumes
            else:
                lower_index = jnp.arange(current.shape[axis] - 1)
                upper_index = jnp.arange(1, current.shape[axis])
                interior = 0.5 * (
                    jnp.take(current, lower_index, axis=axis)
                    + jnp.take(current, upper_index, axis=axis)
                )
                boundary_shape = list(interior.shape)
                boundary_shape[axis] = 1
                zero = jnp.zeros(tuple(boundary_shape), dtype=current.dtype)
                face = jnp.concatenate((zero, interior, zero), axis=axis)
                integrated = face * measure
                rate = rate - jnp.diff(integrated, axis=axis) / volumes
        return rate, boundary_flux

    def advance(
        self,
        state: ResistiveGRRMHDState,
        start_time: ArrayLike,
        end_time: ArrayLike,
        stage_geometries: tuple[
            ValenciaFiniteVolumeStageGeometry,
            ValenciaFiniteVolumeStageGeometry,
        ],
        composition: ArrayLike | None = None,
        /,
        *,
        transport_extinction: ArrayLike = 0.0,
    ) -> ResistiveGRRMHDStepResult:
        if not isinstance(state, ResistiveGRRMHDState):
            raise TypeError("state must be ResistiveGRRMHDState.")
        base_result = self.base.advance(
            state.grrmhd,
            start_time,
            end_time,
            stage_geometries,
            composition,
            transport_extinction=transport_extinction,
        )
        geometry = stage_geometries[-1]
        step = jnp.asarray(end_time, dtype=state.electric_covector.dtype) - jnp.asarray(
            start_time, dtype=state.electric_covector.dtype
        )
        base_state = base_result.state
        recovery = self._primitive(base_state, geometry, composition)
        volume = geometry.cell.sqrt_det_spatial_metric
        charge = state.densitized_charge / volume
        magnetic = recovery.primitive[..., 5:8]
        velocity = recovery.primitive[..., 1:4]
        ohm_before = self.ohm.evaluate(
            state.electric_covector,
            magnetic,
            velocity,
            charge,
            geometry.cell,
        )
        electric_candidate = self._implicit_electric(
            state.electric_covector,
            ohm_before.ideal_electric_covector,
            velocity,
            ohm_before.lorentz_factor,
            geometry.cell.alpha,
            step,
            geometry.cell.spatial_metric,
        )
        ohm_relaxed = self.ohm.evaluate(
            electric_candidate,
            magnetic,
            velocity,
            charge,
            geometry.cell,
        )
        inverse = geometry.cell.inverse_spatial_metric.astype(electric_candidate.dtype)
        old_energy = 0.5 * contract(
            "...i,...ij,...j->...",
            state.electric_covector,
            inverse,
            state.electric_covector,
            backend="jax",
        )
        new_energy = 0.5 * contract(
            "...i,...ij,...j->...",
            electric_candidate,
            inverse,
            electric_candidate,
            backend="jax",
        )
        material_energy_change = volume * (old_energy - new_energy)
        material_candidate = base_state.material_state.at[..., 4].add(
            material_energy_change
        )
        grrmhd_candidate = GRRMHDState(
            material_candidate,
            base_state.constrained_transport,
            base_state.radiation_state,
            base_state.time,
            base_state.step_size,
            base_state.accepted_step,
            base_state.status,
        )
        corrected_recovery = self._primitive(grrmhd_candidate, geometry, composition)
        densitized_coordinate_current = volume[..., None] * (
            geometry.cell.alpha[..., None] * ohm_relaxed.spatial_current
            - geometry.cell.beta_contravariant * charge[..., None]
        )
        charge_rate, boundary_flux = self._charge_rate(
            densitized_coordinate_current,
            self.base.radiation_transport.discretization,
        )
        charge_candidate = state.densitized_charge + step * charge_rate
        local_charge_candidate = charge_candidate / volume
        ohm_after = self.ohm.evaluate(
            electric_candidate,
            corrected_recovery.primitive[..., 5:8],
            corrected_recovery.primitive[..., 1:4],
            local_charge_candidate,
            geometry.cell,
        )
        discretization = self.base.radiation_transport.discretization
        cell_volumes = discretization.cell_volumes.astype(electric_candidate.dtype)
        charge_change = jnp.sum(
            cell_volumes * (charge_candidate - state.densitized_charge)
        )
        charge_residual = charge_change + step * boundary_flux
        electric_change = volume * (new_energy - old_energy)
        energy_residual = electric_change + material_energy_change
        tolerance = jnp.asarray(self.balance_tolerance, dtype=electric_candidate.dtype)
        charge_scale = jnp.maximum(
            jnp.sum(cell_volumes * jnp.abs(state.densitized_charge)), 1.0
        )
        energy_scale = jnp.maximum(jnp.max(jnp.abs(electric_change), initial=0.0), 1.0)
        finite = (
            base_result.finite
            & jnp.all(ohm_before.finite)
            & jnp.all(ohm_relaxed.finite)
            & jnp.all(ohm_after.finite)
            & jnp.all(corrected_recovery.finite)
            & jnp.all(jnp.isfinite(charge_candidate))
            & jnp.all(jnp.isfinite(energy_residual))
            & jnp.isfinite(charge_residual)
        )
        physically_valid = (
            base_result.physically_valid
            & jnp.all(ohm_before.physically_valid)
            & jnp.all(ohm_relaxed.physically_valid)
            & jnp.all(ohm_after.physically_valid)
            & jnp.all(corrected_recovery.physically_valid)
        )
        balanced = (
            jnp.max(jnp.abs(energy_residual), initial=0.0) <= tolerance * energy_scale
        ) & (jnp.abs(charge_residual) <= tolerance * charge_scale)
        qualified = (
            base_result.accepted
            & finite
            & physically_valid
            & jnp.all(ohm_after.qualified)
            & jnp.all(corrected_recovery.qualified)
            & balanced
        )
        derivative = (
            qualified
            & base_result.derivative_valid
            & jnp.all(ohm_after.derivative_valid)
            & jnp.all(corrected_recovery.derivative_valid)
        )
        status = jnp.where(
            qualified,
            int(ResistiveGRRMHDRunStatus.SUCCESS),
            jnp.where(
                ~base_result.accepted,
                int(ResistiveGRRMHDRunStatus.BASE_GRRMHD_FAILED),
                jnp.where(
                    ~finite,
                    int(ResistiveGRRMHDRunStatus.NONFINITE_STATE),
                    jnp.where(
                        ~jnp.all(corrected_recovery.qualified),
                        int(ResistiveGRRMHDRunStatus.MATERIAL_RECOVERY_FAILED),
                        jnp.where(
                            ~balanced,
                            int(ResistiveGRRMHDRunStatus.CHARGE_BALANCE_FAILED),
                            int(ResistiveGRRMHDRunStatus.OHMIC_CLOSURE_FAILED),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        candidate = ResistiveGRRMHDState(
            grrmhd_candidate,
            electric_candidate,
            charge_candidate,
            status,
        )
        selected_grrmhd = jax.tree.map(
            lambda new, old: jnp.where(qualified, new, old),
            grrmhd_candidate,
            state.grrmhd,
        )
        accepted_state = ResistiveGRRMHDState(
            selected_grrmhd,
            jnp.where(qualified, electric_candidate, state.electric_covector),
            jnp.where(qualified, charge_candidate, state.densitized_charge),
            status,
        )
        zero_if_rejected = lambda value: jnp.where(
            qualified, value, jnp.zeros_like(value)
        )
        ledger = ResistiveGRRMHDLedger(
            zero_if_rejected(electric_change),
            zero_if_rejected(material_energy_change),
            zero_if_rejected(energy_residual),
            zero_if_rejected(charge_change),
            zero_if_rejected(boundary_flux),
            zero_if_rejected(charge_residual),
            zero_if_rejected(step * ohm_after.entropy_production),
            qualified,
            finite,
            qualified,
            self.plan_id,
        )
        return ResistiveGRRMHDStepResult(
            candidate,
            accepted_state,
            base_result,
            ohm_before,
            ohm_after,
            ledger,
            qualified,
            status,
            finite,
            physically_valid,
            qualified,
            derivative,
            self.plan_id,
        )


__all__ = [
    "FixedGridResistiveGRRMHDIMEXPlan",
    "ResistiveGRRMHDLedger",
    "ResistiveGRRMHDRunStatus",
    "ResistiveGRRMHDState",
    "ResistiveGRRMHDStepResult",
]
