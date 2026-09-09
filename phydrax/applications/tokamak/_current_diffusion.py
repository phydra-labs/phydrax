#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Implicit conservative poloidal-flux diffusion on prepared flux surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import solve_tridiagonal_lines
from ._core_transport import (
    PreparedTokamakCoreTransport,
    TokamakCoreState,
    TokamakCoreTransportStepResult,
    TokamakEdgeFlux,
    TokamakTransportCoefficients,
    TokamakTransportSources,
)
from ._flux_surfaces import FluxSurfaceGeometry, PreparedFluxSurfaceGeometry


class CurrentDiffusionState(StrictModule):
    poloidal_flux_wb_per_rad: Array
    time_s: Array

    def __init__(self, poloidal_flux_wb_per_rad: ArrayLike, time_s: ArrayLike = 0.0, /):
        flux = jnp.asarray(poloidal_flux_wb_per_rad)
        time = jnp.asarray(time_s, dtype=flux.dtype)
        if flux.ndim != 1 or time.shape != ():
            raise ValueError(
                "Current-diffusion state requires rank-one flux and scalar time."
            )
        self.poloidal_flux_wb_per_rad = flux
        self.time_s = time


class CurrentDiffusionStepResult(StrictModule):
    candidate_state: CurrentDiffusionState
    accepted_state: CurrentDiffusionState
    integrated_face_flux_rate_wb_s: Array
    balance_residual_wb: Array
    linear_residual_norm: Array
    finite: Array
    domain_valid: Array
    sensitivity_valid: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class CurrentDiffusionPlan:
    geometry: FluxSurfaceGeometry
    pivot_tolerance: float = 1.0e-14
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, FluxSurfaceGeometry):
            raise TypeError("geometry must be FluxSurfaceGeometry.")
        tolerance = float(self.pivot_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("pivot_tolerance must be finite and positive.")
        object.__setattr__(self, "pivot_tolerance", tolerance)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "current-diffusion-plan",
                    "geometry": self.geometry.geometry_id,
                    "pivot_tolerance": tolerance,
                    "equation": "conservative-poloidal-flux-diffusion",
                }
            ),
        )

    def prepare(self) -> PreparedCurrentDiffusion:
        return PreparedCurrentDiffusion(
            self.geometry.prepare(), self.pivot_tolerance, self.plan_id
        )


class PreparedCurrentDiffusion(StrictModule, NonTrainableState):
    geometry: PreparedFluxSurfaceGeometry
    pivot_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @property
    def cell_count(self) -> int:
        return int(self.geometry.cell_volume_m3.shape[0])

    def step(
        self,
        state: CurrentDiffusionState,
        dt_s: ArrayLike,
        face_conductance_m3_s: ArrayLike,
        source_wb_per_rad_m3_s: ArrayLike,
        edge_outward_rate_wb_per_rad_s: ArrayLike = 0.0,
        /,
    ) -> CurrentDiffusionStepResult:
        if not isinstance(state, CurrentDiffusionState):
            raise TypeError("state must be CurrentDiffusionState.")
        flux = state.poloidal_flux_wb_per_rad
        conductance = jnp.asarray(face_conductance_m3_s, dtype=flux.dtype)
        source = jnp.asarray(source_wb_per_rad_m3_s, dtype=flux.dtype)
        edge = jnp.asarray(edge_outward_rate_wb_per_rad_s, dtype=flux.dtype)
        dt = jnp.asarray(dt_s, dtype=flux.dtype)
        if (
            flux.shape != (self.cell_count,)
            or conductance.shape != (self.cell_count + 1,)
            or source.shape != flux.shape
            or edge.shape != ()
            or dt.shape != ()
        ):
            raise ValueError(
                "Current-diffusion state, coefficient, source, and boundary shapes disagree."
            )
        finite_inputs = (
            jnp.all(jnp.isfinite(flux))
            & jnp.all(jnp.isfinite(conductance))
            & jnp.all(jnp.isfinite(source))
            & jnp.isfinite(edge)
            & jnp.isfinite(dt)
        )
        domain_valid = (
            finite_inputs
            & (dt > 0.0)
            & jnp.all(conductance >= 0.0)
            & (conductance[0] == 0.0)
            & (conductance[-1] == 0.0)
        )
        safe_dt = jnp.where(domain_valid, dt, 1.0)
        safe_conductance = jnp.where(
            jnp.isfinite(conductance) & (conductance >= 0.0), conductance, 0.0
        )
        safe_source = jnp.where(jnp.isfinite(source), source, 0.0)
        safe_edge = jnp.where(jnp.isfinite(edge), edge, 0.0)
        volume = self.geometry.cell_volume_m3
        lower = jnp.zeros_like(flux).at[1:].set(-safe_conductance[1:-1])
        upper = jnp.zeros_like(flux).at[:-1].set(-safe_conductance[1:-1])
        diagonal = volume / safe_dt + safe_conductance[:-1] + safe_conductance[1:]
        rhs = volume / safe_dt * flux + volume * safe_source
        rhs = rhs.at[-1].add(-safe_edge)
        linear = solve_tridiagonal_lines(
            lower,
            diagonal,
            upper,
            rhs,
            0,
            pivot_tolerance=self.pivot_tolerance,
        )
        candidate_flux = linear.value
        integrated = jnp.zeros((self.cell_count + 1,), dtype=flux.dtype)
        integrated = integrated.at[1:-1].set(
            safe_conductance[1:-1] * (candidate_flux[:-1] - candidate_flux[1:])
        )
        integrated = integrated.at[-1].set(safe_edge)
        initial_total = self.geometry.metric_line.total_amount(flux)
        candidate_total = self.geometry.metric_line.total_amount(candidate_flux)
        source_total = self.geometry.metric_line.total_amount(safe_source)
        balance = (
            candidate_total - initial_total - safe_dt * source_total + safe_dt * safe_edge
        )
        finite = jnp.all(jnp.isfinite(candidate_flux)) & jnp.isfinite(balance)
        scale = jnp.maximum(
            1.0, jnp.maximum(jnp.abs(initial_total), jnp.abs(candidate_total))
        )
        balance_valid = jnp.abs(balance) <= 1024.0 * jnp.finfo(flux.dtype).eps * scale
        successful = domain_valid & linear.successful & finite & balance_valid
        candidate_state = CurrentDiffusionState(candidate_flux, state.time_s + safe_dt)
        accepted_state = CurrentDiffusionState(
            jnp.where(successful, candidate_flux, flux),
            jnp.where(successful, state.time_s + dt, state.time_s),
        )
        return CurrentDiffusionStepResult(
            candidate_state,
            accepted_state,
            integrated,
            balance,
            linear.residual_norm,
            finite,
            domain_valid,
            successful,
            successful,
        )


class CoupledTokamakTransportState(StrictModule):
    core: TokamakCoreState
    current: CurrentDiffusionState


class CoupledTokamakTransportStepResult(StrictModule):
    core: TokamakCoreTransportStepResult
    current: CurrentDiffusionStepResult
    candidate_state: CoupledTokamakTransportState
    accepted_state: CoupledTokamakTransportState
    successful: Array


class PreparedTokamakTransportCurrentCoupling(StrictModule, NonTrainableState):
    core: PreparedTokamakCoreTransport
    current: PreparedCurrentDiffusion
    coupling_id: str = eqx.field(static=True)

    def __init__(
        self,
        core: PreparedTokamakCoreTransport,
        current: PreparedCurrentDiffusion,
        /,
    ):
        if not isinstance(core, PreparedTokamakCoreTransport) or not isinstance(
            current, PreparedCurrentDiffusion
        ):
            raise TypeError(
                "Coupling requires prepared core and current-diffusion operators."
            )
        if core.geometry.geometry_id != current.geometry.geometry_id:
            raise ValueError(
                "Core and current diffusion must share one flux-surface geometry."
            )
        self.core = core
        self.current = current
        self.coupling_id = canonical_fingerprint(
            {
                "kind": "tokamak-transport-current-coupling",
                "core": core.plan_id,
                "current": current.plan_id,
                "schedule": "synchronous-staggered",
            }
        )

    def step(
        self,
        state: CoupledTokamakTransportState,
        dt_s: ArrayLike,
        transport_coefficients: TokamakTransportCoefficients,
        transport_sources: TokamakTransportSources,
        current_face_conductance_m3_s: ArrayLike,
        current_source_wb_per_rad_m3_s: ArrayLike,
        /,
        *,
        edge_flux: TokamakEdgeFlux | None = None,
        edge_current_rate_wb_per_rad_s: ArrayLike = 0.0,
    ) -> CoupledTokamakTransportStepResult:
        if not isinstance(state, CoupledTokamakTransportState):
            raise TypeError("state must be CoupledTokamakTransportState.")
        core_result = self.core.step(
            state.core,
            dt_s,
            transport_coefficients,
            transport_sources,
            edge_flux,
        )
        current_dt = jnp.where(
            core_result.successful, jnp.asarray(dt_s), -jnp.ones_like(jnp.asarray(dt_s))
        )
        current_result = self.current.step(
            state.current,
            current_dt,
            current_face_conductance_m3_s,
            current_source_wb_per_rad_m3_s,
            edge_current_rate_wb_per_rad_s,
        )
        successful = core_result.successful & current_result.successful
        candidate = CoupledTokamakTransportState(
            core_result.candidate_state, current_result.candidate_state
        )
        accepted = CoupledTokamakTransportState(
            TokamakCoreState(
                jnp.where(
                    successful,
                    core_result.candidate_state.electron_density_m3,
                    state.core.electron_density_m3,
                ),
                jnp.where(
                    successful,
                    core_result.candidate_state.electron_thermal_energy_j,
                    state.core.electron_thermal_energy_j,
                ),
                jnp.where(
                    successful,
                    core_result.candidate_state.ion_thermal_energy_j,
                    state.core.ion_thermal_energy_j,
                ),
                jnp.where(
                    successful, core_result.candidate_state.time_s, state.core.time_s
                ),
            ),
            CurrentDiffusionState(
                jnp.where(
                    successful,
                    current_result.candidate_state.poloidal_flux_wb_per_rad,
                    state.current.poloidal_flux_wb_per_rad,
                ),
                jnp.where(
                    successful,
                    current_result.candidate_state.time_s,
                    state.current.time_s,
                ),
            ),
        )
        return CoupledTokamakTransportStepResult(
            core_result, current_result, candidate, accepted, successful
        )


__all__ = [
    "CoupledTokamakTransportState",
    "CoupledTokamakTransportStepResult",
    "CurrentDiffusionPlan",
    "CurrentDiffusionState",
    "CurrentDiffusionStepResult",
    "PreparedCurrentDiffusion",
    "PreparedTokamakTransportCurrentCoupling",
]
