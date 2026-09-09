#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Implicit conservative radial tokamak particle and thermal transport."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import solve_tridiagonal_lines
from ._flux_surfaces import FluxSurfaceGeometry, PreparedFluxSurfaceGeometry


class TokamakCoreState(StrictModule):
    """Electron density and electron/ion thermal energies per particle."""

    electron_density_m3: Array
    electron_thermal_energy_j: Array
    ion_thermal_energy_j: Array
    time_s: Array

    def __init__(
        self,
        electron_density_m3: ArrayLike,
        electron_thermal_energy_j: ArrayLike,
        ion_thermal_energy_j: ArrayLike,
        time_s: ArrayLike = 0.0,
        /,
    ):
        electron_density = jnp.asarray(electron_density_m3)
        electron_energy = jnp.asarray(
            electron_thermal_energy_j, dtype=electron_density.dtype
        )
        ion_energy = jnp.asarray(ion_thermal_energy_j, dtype=electron_density.dtype)
        if (
            electron_density.ndim != 1
            or electron_energy.shape != electron_density.shape
            or ion_energy.shape != electron_density.shape
        ):
            raise ValueError("Tokamak core profiles must share one rank-one cell shape.")
        time = jnp.asarray(time_s, dtype=electron_density.dtype)
        if time.shape != ():
            raise ValueError("time_s must be scalar.")
        self.electron_density_m3 = electron_density
        self.electron_thermal_energy_j = electron_energy
        self.ion_thermal_energy_j = ion_energy
        self.time_s = time


class TokamakTransportCoefficients(StrictModule):
    """Integrated face conductances for particles and thermal-energy density."""

    particle_conductance_m3_s: Array
    electron_energy_conductance_m3_s: Array
    ion_energy_conductance_m3_s: Array
    valid: Array
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        particle_conductance_m3_s: ArrayLike,
        electron_energy_conductance_m3_s: ArrayLike,
        ion_energy_conductance_m3_s: ArrayLike,
        /,
        *,
        valid: ArrayLike = True,
        model_id: str = "prescribed-integrated-conductance",
    ):
        particle = jnp.asarray(particle_conductance_m3_s)
        electron = jnp.asarray(electron_energy_conductance_m3_s, dtype=particle.dtype)
        ion = jnp.asarray(ion_energy_conductance_m3_s, dtype=particle.dtype)
        if (
            particle.ndim != 1
            or electron.shape != particle.shape
            or ion.shape != particle.shape
        ):
            raise ValueError("Transport conductances must share one rank-one face shape.")
        valid_ = jnp.asarray(valid, dtype=bool)
        if valid_.shape != ():
            raise ValueError("Transport coefficient validity must be scalar.")
        identity = str(model_id).strip()
        if not identity or identity != model_id:
            raise ValueError("model_id must be non-empty canonical text.")
        self.particle_conductance_m3_s = particle
        self.electron_energy_conductance_m3_s = electron
        self.ion_energy_conductance_m3_s = ion
        self.valid = valid_
        self.model_id = identity

    def stacked(self) -> Array:
        return jnp.stack(
            (
                self.particle_conductance_m3_s,
                self.electron_energy_conductance_m3_s,
                self.ion_energy_conductance_m3_s,
            ),
            axis=-1,
        )


class TokamakTransportSources(StrictModule):
    particle_source_m3_s: Array
    electron_heating_w_m3: Array
    ion_heating_w_m3: Array
    exchange_to_electrons_w_m3: Array

    def __init__(
        self,
        particle_source_m3_s: ArrayLike,
        electron_heating_w_m3: ArrayLike,
        ion_heating_w_m3: ArrayLike,
        /,
        *,
        exchange_to_electrons_w_m3: ArrayLike | None = None,
    ):
        particle = jnp.asarray(particle_source_m3_s)
        electron = jnp.asarray(electron_heating_w_m3, dtype=particle.dtype)
        ion = jnp.asarray(ion_heating_w_m3, dtype=particle.dtype)
        exchange = (
            jnp.zeros_like(particle)
            if exchange_to_electrons_w_m3 is None
            else jnp.asarray(exchange_to_electrons_w_m3, dtype=particle.dtype)
        )
        if particle.ndim != 1 or any(
            value.shape != particle.shape for value in (electron, ion, exchange)
        ):
            raise ValueError("Tokamak sources must share one rank-one cell shape.")
        self.particle_source_m3_s = particle
        self.electron_heating_w_m3 = electron
        self.ion_heating_w_m3 = ion
        self.exchange_to_electrons_w_m3 = exchange

    def stacked(self) -> Array:
        return jnp.stack(
            (
                self.particle_source_m3_s,
                self.electron_heating_w_m3 + self.exchange_to_electrons_w_m3,
                self.ion_heating_w_m3 - self.exchange_to_electrons_w_m3,
            ),
            axis=-1,
        )


class TokamakEdgeFlux(StrictModule):
    particle_rate_s: Array
    electron_power_w: Array
    ion_power_w: Array

    def __init__(
        self,
        particle_rate_s: ArrayLike = 0.0,
        electron_power_w: ArrayLike = 0.0,
        ion_power_w: ArrayLike = 0.0,
        /,
    ):
        values = tuple(
            jnp.asarray(value)
            for value in (particle_rate_s, electron_power_w, ion_power_w)
        )
        if any(value.shape != () for value in values):
            raise ValueError("Tokamak edge fluxes must be scalars.")
        self.particle_rate_s, self.electron_power_w, self.ion_power_w = values

    def stacked(self, dtype) -> Array:
        return jnp.asarray(
            (self.particle_rate_s, self.electron_power_w, self.ion_power_w),
            dtype=dtype,
        )


class TokamakTransportLedger(StrictModule):
    initial_totals: Array
    candidate_totals: Array
    integrated_sources: Array
    integrated_edge_losses: Array
    closure_residual: Array
    finite: Array
    successful: Array


class TokamakCoreTransportStepResult(StrictModule):
    candidate_state: TokamakCoreState
    accepted_state: TokamakCoreState
    integrated_face_flux: Array
    amount_rate: Array
    ledger: TokamakTransportLedger
    linear_residual_norm: Array
    minimum_pivot: Array
    finite: Array
    domain_valid: Array
    sensitivity_valid: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class TokamakCoreTransportPlan:
    geometry: FluxSurfaceGeometry
    mean_ion_charge: float
    pivot_tolerance: float = 1.0e-14
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, FluxSurfaceGeometry):
            raise TypeError("geometry must be FluxSurfaceGeometry.")
        charge = float(self.mean_ion_charge)
        tolerance = float(self.pivot_tolerance)
        if not math.isfinite(charge) or charge <= 0.0:
            raise ValueError("mean_ion_charge must be finite and positive.")
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("pivot_tolerance must be finite and positive.")
        object.__setattr__(self, "mean_ion_charge", charge)
        object.__setattr__(self, "pivot_tolerance", tolerance)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "tokamak-core-transport-plan",
                    "geometry": self.geometry.geometry_id,
                    "mean_ion_charge": charge,
                    "pivot_tolerance": tolerance,
                    "equations": [
                        "electron-particle-balance",
                        "electron-thermal-energy-balance",
                        "ion-thermal-energy-balance",
                    ],
                }
            ),
        )

    def prepare(self) -> PreparedTokamakCoreTransport:
        return PreparedTokamakCoreTransport(
            self.geometry.prepare(),
            self.mean_ion_charge,
            self.pivot_tolerance,
            self.plan_id,
        )


class PreparedTokamakCoreTransport(StrictModule, NonTrainableState):
    geometry: PreparedFluxSurfaceGeometry
    mean_ion_charge: float = eqx.field(static=True)
    pivot_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @property
    def cell_count(self) -> int:
        return int(self.geometry.cell_volume_m3.shape[0])

    def conserved_density(self, state: TokamakCoreState, /) -> Array:
        if not isinstance(state, TokamakCoreState):
            raise TypeError("state must be TokamakCoreState.")
        if state.electron_density_m3.shape != (self.cell_count,):
            raise ValueError("Tokamak core state does not match prepared geometry.")
        electron_energy_density = (
            1.5 * state.electron_density_m3 * state.electron_thermal_energy_j
        )
        ion_density = state.electron_density_m3 / self.mean_ion_charge
        ion_energy_density = 1.5 * ion_density * state.ion_thermal_energy_j
        return jnp.stack(
            (state.electron_density_m3, electron_energy_density, ion_energy_density),
            axis=-1,
        )

    def state_from_conserved_density(
        self, conserved: ArrayLike, time_s: ArrayLike, /
    ) -> TokamakCoreState:
        value = jnp.asarray(conserved)
        if value.shape != (self.cell_count, 3):
            raise ValueError("Conserved core state must have shape (cell_count, 3).")
        density = value[:, 0]
        safe_density = jnp.where(density > 0.0, density, 1.0)
        electron_energy = value[:, 1] / (1.5 * safe_density)
        ion_density = safe_density / self.mean_ion_charge
        ion_energy = value[:, 2] / (1.5 * ion_density)
        return TokamakCoreState(density, electron_energy, ion_energy, time_s)

    def step(
        self,
        state: TokamakCoreState,
        dt_s: ArrayLike,
        coefficients: TokamakTransportCoefficients,
        sources: TokamakTransportSources,
        edge_flux: TokamakEdgeFlux | None = None,
        /,
    ) -> TokamakCoreTransportStepResult:
        if not isinstance(coefficients, TokamakTransportCoefficients):
            raise TypeError("coefficients must be TokamakTransportCoefficients.")
        if not isinstance(sources, TokamakTransportSources):
            raise TypeError("sources must be TokamakTransportSources.")
        boundary = TokamakEdgeFlux() if edge_flux is None else edge_flux
        if not isinstance(boundary, TokamakEdgeFlux):
            raise TypeError("edge_flux must be TokamakEdgeFlux or None.")
        conserved = self.conserved_density(state)
        dt = jnp.asarray(dt_s, dtype=conserved.dtype)
        if dt.shape != ():
            raise ValueError("dt_s must be scalar.")
        conductance = coefficients.stacked()
        source = sources.stacked()
        expected_face_shape = (self.cell_count + 1, 3)
        if conductance.shape != expected_face_shape:
            raise ValueError(
                f"Transport conductance must have shape {expected_face_shape}."
            )
        if source.shape != conserved.shape:
            raise ValueError("Transport source shape must match the core state.")
        edge = boundary.stacked(conserved.dtype)
        finite_inputs = (
            jnp.all(jnp.isfinite(conserved))
            & jnp.isfinite(dt)
            & jnp.all(jnp.isfinite(conductance))
            & jnp.all(jnp.isfinite(source))
            & jnp.all(jnp.isfinite(edge))
        )
        positive_state = jnp.all(conserved > 0.0)
        coefficient_valid = (
            coefficients.valid
            & jnp.all(conductance >= 0.0)
            & jnp.all(conductance[0] == 0.0)
            & jnp.all(conductance[-1] == 0.0)
        )
        domain_valid = finite_inputs & positive_state & coefficient_valid & (dt > 0.0)
        safe_dt = jnp.where(domain_valid, dt, 1.0)
        safe_conductance = jnp.where(
            jnp.isfinite(conductance) & (conductance >= 0.0), conductance, 0.0
        )
        safe_source = jnp.where(jnp.isfinite(source), source, 0.0)
        safe_edge = jnp.where(jnp.isfinite(edge), edge, 0.0)
        volume = self.geometry.cell_volume_m3[:, None]
        lower = jnp.zeros_like(conserved).at[1:].set(-safe_conductance[1:-1])
        upper = jnp.zeros_like(conserved).at[:-1].set(-safe_conductance[1:-1])
        diagonal = volume / safe_dt + safe_conductance[:-1] + safe_conductance[1:]
        rhs = volume / safe_dt * conserved + volume * safe_source
        rhs = rhs.at[-1].add(-safe_edge)
        linear = solve_tridiagonal_lines(
            lower,
            diagonal,
            upper,
            rhs,
            0,
            pivot_tolerance=self.pivot_tolerance,
        )
        candidate_conserved = linear.value
        candidate_state = self.state_from_conserved_density(
            candidate_conserved, state.time_s + safe_dt
        )
        integrated_face_flux = jnp.zeros(expected_face_shape, dtype=conserved.dtype)
        integrated_face_flux = integrated_face_flux.at[1:-1].set(
            safe_conductance[1:-1] * (candidate_conserved[:-1] - candidate_conserved[1:])
        )
        integrated_face_flux = integrated_face_flux.at[-1].set(safe_edge)
        amount_rate = self.geometry.metric_line.amount_rate_from_integrated_flux(
            integrated_face_flux, source_density=safe_source
        )
        initial_totals = self.geometry.metric_line.total_amount(conserved)
        candidate_totals = self.geometry.metric_line.total_amount(candidate_conserved)
        integrated_sources = safe_dt * self.geometry.metric_line.total_amount(safe_source)
        integrated_edge = safe_dt * safe_edge
        closure = candidate_totals - initial_totals - integrated_sources + integrated_edge
        finite = (
            linear.finite
            & jnp.all(jnp.isfinite(candidate_conserved))
            & jnp.all(jnp.isfinite(closure))
        )
        positive_candidate = jnp.all(candidate_conserved > 0.0)
        scale = jnp.maximum(
            1.0,
            jnp.maximum(jnp.abs(initial_totals), jnp.abs(candidate_totals)),
        )
        tolerance = 1024.0 * jnp.finfo(conserved.dtype).eps * scale
        ledger_valid = jnp.all(jnp.abs(closure) <= tolerance)
        successful = (
            domain_valid & linear.successful & finite & positive_candidate & ledger_valid
        )
        accepted_conserved = jnp.where(successful, candidate_conserved, conserved)
        accepted_time = jnp.where(successful, state.time_s + dt, state.time_s)
        accepted_state = self.state_from_conserved_density(
            accepted_conserved, accepted_time
        )
        ledger = TokamakTransportLedger(
            initial_totals,
            candidate_totals,
            integrated_sources,
            integrated_edge,
            closure,
            finite,
            ledger_valid,
        )
        return TokamakCoreTransportStepResult(
            candidate_state,
            accepted_state,
            integrated_face_flux,
            amount_rate,
            ledger,
            linear.residual_norm,
            linear.minimum_pivot,
            finite,
            domain_valid,
            successful,
            successful,
        )


__all__ = [
    "PreparedTokamakCoreTransport",
    "TokamakCoreState",
    "TokamakCoreTransportPlan",
    "TokamakCoreTransportStepResult",
    "TokamakEdgeFlux",
    "TokamakTransportCoefficients",
    "TokamakTransportLedger",
    "TokamakTransportSources",
]
