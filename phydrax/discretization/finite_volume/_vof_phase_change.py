#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._unstructured_thermal import UnstructuredTwoMaterialThermalDiffusionPlan
from ._unstructured_vof import JAXPLICStageReconstruction, UnstructuredVOFPlan


if TYPE_CHECKING:
    from ...equations._vof_phase_change import (
        TwoMaterialVOFPhaseChangePlan,
        VOFPhaseChangeStepResult,
    )


class StefanHeatFluxReconstruction(StrictModule):
    phase0_normal_heat_flux: Array
    phase1_normal_heat_flux: Array
    phase0_support: Array
    phase1_support: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    plic_plan_id: str = eqx.field(static=True)


class VOFPhaseChangeStageEvaluation(StrictModule):
    result: VOFPhaseChangeStepResult
    interface_area_density: Array
    interface_active: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    plic_plan_id: str = eqx.field(static=True)


class VOFPhaseChangePlan(StrictModule, NonTrainableState):
    phase_change: TwoMaterialVOFPhaseChangePlan
    vof: UnstructuredVOFPlan
    thermal_diffusion: UnstructuredTwoMaterialThermalDiffusionPlan | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        phase_change: TwoMaterialVOFPhaseChangePlan,
        vof: UnstructuredVOFPlan,
        /,
        *,
        thermal_diffusion: UnstructuredTwoMaterialThermalDiffusionPlan | None = None,
    ):
        from ...equations._vof_phase_change import (
            StefanHeatFluxPhaseChangePlan,
            TwoMaterialVOFPhaseChangePlan,
        )

        if not isinstance(phase_change, TwoMaterialVOFPhaseChangePlan):
            raise TypeError("phase_change must be TwoMaterialVOFPhaseChangePlan.")
        if not isinstance(vof, UnstructuredVOFPlan):
            raise TypeError("vof must be UnstructuredVOFPlan.")
        if phase_change.system.dimension != vof.discretization.cell_dimension:
            raise ValueError("Phase-change system and VOF geometry dimensions differ.")
        if thermal_diffusion is not None and (
            not isinstance(thermal_diffusion, UnstructuredTwoMaterialThermalDiffusionPlan)
            or thermal_diffusion.discretization.prepared_id
            != vof.discretization.prepared_id
        ):
            raise ValueError(
                "Thermal phase change requires diffusion on the exact VOF geometry."
            )
        if (
            isinstance(phase_change.rate_law, StefanHeatFluxPhaseChangePlan)
            and thermal_diffusion is None
        ):
            raise ValueError(
                "Stefan heat-flux transfer requires prepared thermal diffusion."
            )
        self.phase_change = phase_change
        self.vof = vof
        self.thermal_diffusion = thermal_diffusion
        self.plan_id = canonical_fingerprint(
            {
                "kind": "vof-phase-change-plan",
                "phase_change": phase_change.plan_id,
                "vof": vof.plan_id,
                "thermal_diffusion": (
                    None if thermal_diffusion is None else thermal_diffusion.plan_id
                ),
            }
        )

    def stefan_heat_fluxes(
        self,
        temperature: ArrayLike,
        stage_plic: JAXPLICStageReconstruction,
        /,
        *,
        cell_centers: ArrayLike | None = None,
    ) -> StefanHeatFluxReconstruction:
        if self.thermal_diffusion is None:
            raise ValueError("Stefan heat fluxes require thermal diffusion.")
        if (
            not isinstance(stage_plic, JAXPLICStageReconstruction)
            or stage_plic.plan_id != self.vof.plan_id
        ):
            raise ValueError("Stefan heat fluxes require current stage PLIC geometry.")
        temperature_ = jnp.asarray(temperature)
        if temperature_.shape != (self.vof.discretization.cell_count,):
            raise ValueError("Stefan temperature must contain one value per VOF cell.")
        centers = (
            self.vof.discretization.cell_centers
            if cell_centers is None
            else jnp.asarray(cell_centers, dtype=temperature_.dtype)
        )
        if centers.shape != self.vof.discretization.cell_centers.shape:
            raise ValueError("Stefan cell centers are incompatible with VOF geometry.")
        stencil = self.vof.gradient.stencil_cells.astype(jnp.int32)
        valid = self.vof.gradient.stencil_valid
        safe_stencil = jnp.clip(stencil, 0, centers.shape[0] - 1)
        displacement = centers[safe_stencil] - centers[:, None, :]
        projection = ein.contract(
            "ckd,cd->ck",
            displacement,
            stage_plic.normals,
            backend="jax",
        )
        distance = jnp.abs(projection)
        minimum_distance = 64.0 * jnp.finfo(temperature_.dtype).eps
        neighbor_alpha = stage_plic.volume_fraction[safe_stencil]
        phase0_mask = valid & (projection < -minimum_distance) & (neighbor_alpha >= 0.5)
        phase1_mask = valid & (projection > minimum_distance) & (neighbor_alpha <= 0.5)
        safe_distance = jnp.maximum(distance, minimum_distance)
        inverse_distance = 1.0 / safe_distance
        phase0_weight = jnp.where(phase0_mask, inverse_distance, 0.0)
        phase1_weight = jnp.where(phase1_mask, inverse_distance, 0.0)
        phase0_support = jnp.sum(phase0_weight, axis=1)
        phase1_support = jnp.sum(phase1_weight, axis=1)
        neighbor_temperature = temperature_[safe_stencil]
        phase0_gradient = jnp.sum(
            phase0_weight
            * (temperature_[:, None] - neighbor_temperature)
            / safe_distance,
            axis=1,
        ) / jnp.where(phase0_support > 0.0, phase0_support, 1.0)
        phase1_gradient = jnp.sum(
            phase1_weight
            * (neighbor_temperature - temperature_[:, None])
            / safe_distance,
            axis=1,
        ) / jnp.where(phase1_support > 0.0, phase1_support, 1.0)
        active = stage_plic.interface_active
        phase0_flux = jnp.where(
            active,
            -self.thermal_diffusion.phase0_conductivity * phase0_gradient,
            0.0,
        )
        phase1_flux = jnp.where(
            active,
            -self.thermal_diffusion.phase1_conductivity * phase1_gradient,
            0.0,
        )
        support = (~active) | ((phase0_support > 0.0) & (phase1_support > 0.0))
        finite = (
            jnp.all(jnp.isfinite(phase0_flux))
            & jnp.all(jnp.isfinite(phase1_flux))
            & jnp.all(jnp.isfinite(phase0_support))
            & jnp.all(jnp.isfinite(phase1_support))
        )
        return StefanHeatFluxReconstruction(
            phase0_flux,
            phase1_flux,
            phase0_support,
            phase1_support,
            finite,
            finite & jnp.all(support),
            self.plan_id,
            stage_plic.plan_id,
        )

    def evaluate_stage(
        self,
        state: ArrayLike,
        step_size: ArrayLike,
        stage_plic: JAXPLICStageReconstruction,
        /,
        *,
        heat_flux0: ArrayLike = 0.0,
        heat_flux1: ArrayLike = 0.0,
    ) -> VOFPhaseChangeStageEvaluation:
        if not isinstance(stage_plic, JAXPLICStageReconstruction):
            raise TypeError("stage_plic must be JAXPLICStageReconstruction.")
        if stage_plic.plan_id != self.vof.plan_id:
            raise ValueError("PLIC stage belongs to another VOF plan.")
        value = self.phase_change.system._state(state)
        alpha = value[..., self.phase_change.system.alpha_index]
        if alpha.shape != stage_plic.volume_fraction.shape:
            raise ValueError("Phase-change state and PLIC alpha shapes differ.")
        alpha_identity = jnp.all(alpha == stage_plic.volume_fraction)
        volumes = self.vof.discretization.cell_volumes.astype(alpha.dtype)
        area_density = stage_plic.interface_measures / volumes
        result = self.phase_change.step(
            value,
            step_size,
            interface_area_density=area_density,
            heat_flux0=heat_flux0,
            heat_flux1=heat_flux1,
        )
        finite = (
            jnp.all(stage_plic.interface_evidence)
            & jnp.all(jnp.isfinite(area_density))
            & jnp.all(result.finite)
        )
        successful = finite & alpha_identity & jnp.all(result.accepted)
        return VOFPhaseChangeStageEvaluation(
            result,
            area_density,
            stage_plic.interface_active,
            finite,
            successful,
            self.plan_id,
            stage_plic.plan_id,
        )


__all__ = [
    "StefanHeatFluxReconstruction",
    "VOFPhaseChangePlan",
    "VOFPhaseChangeStageEvaluation",
]
