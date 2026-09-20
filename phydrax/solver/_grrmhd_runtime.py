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

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..metrix._adm_exchange import (
    combine_stress_energy_projections,
    StressEnergyProjection,
)
from ._balance_law_composition import AdditiveIMEXTableau
from ._gr_m1_finite_volume import (
    FixedGridGRM1SSPRK3Plan,
    GRM1SpatialRate,
)
from ._grmhd_ct import GRMHDCTDefectLedger, GRMHDCTRate, GRMHDCTState
from ._grmhd_runtime import GRMHDSpatialRate, GRMHDSSPRK3Plan
from ._grrmhd_source import (
    GRRMHDImplicitSourcePlan,
    GRRMHDSourceResult,
)
from ._relativistic_finite_volume import ValenciaFiniteVolumeStageGeometry


class GRRMHDRunStatus(IntEnum):
    SUCCESS = 0
    INVALID_INITIAL_STATE = 1
    GEOMETRY_INVALID = 2
    STABILITY_LIMIT_EXCEEDED = 3
    MATERIAL_RECOVERY_FAILED = 4
    RADIATION_REALIZABILITY_FAILED = 5
    IMPLICIT_SOURCE_FAILED = 6
    MAGNETIC_CONSTRAINT_FAILED = 7
    CONSERVATION_DEFECT = 8
    NONFINITE_STATE = 9


class GRRMHDState(StrictModule):
    material_state: Array
    constrained_transport: GRMHDCTState
    radiation_state: Array
    time: Array
    step_size: Array
    accepted_step: Array
    status: Array


class GRRMHDStageEvidence(StrictModule):
    material_rates: tuple[GRMHDSpatialRate, GRMHDSpatialRate]
    radiation_rates: tuple[GRM1SpatialRate, GRM1SpatialRate]
    source_results: tuple[GRRMHDSourceResult, GRRMHDSourceResult]
    total_stress_energy: tuple[StressEnergyProjection, StressEnergyProjection]
    stable_steps: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array


class GRRMHDStageProposal(StrictModule):
    candidate: GRRMHDState
    source: GRRMHDSourceResult
    material_rate: GRMHDSpatialRate
    radiation_rate: GRM1SpatialRate
    total_stress_energy: StressEnergyProjection
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array


class GRRMHDDefectLedger(StrictModule):
    material_face_flux_integrals: tuple[Array, ...]
    radiation_face_flux_integrals: tuple[Array, ...]
    material_geometric_source_integral: Array
    radiation_geometric_source_integral: Array
    material_exchange_change: Array
    radiation_exchange_change: Array
    combined_energy_defect: Array
    combined_momentum_defect: Array
    source_energy_defect: Array
    source_momentum_defect: Array
    constrained_transport: GRMHDCTDefectLedger
    finite: Array
    qualified: Array
    accepted: Array
    plan_id: str = eqx.field(static=True)


class GRRMHDStepResult(StrictModule):
    candidate: GRRMHDState
    state: GRRMHDState
    accepted: Array
    status: Array
    stages: GRRMHDStageEvidence
    attempted_ledger: GRRMHDDefectLedger
    accepted_ledger: GRRMHDDefectLedger
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array


def _ct_add(base: GRMHDCTState, scale: Array, rate: GRMHDCTRate, /) -> GRMHDCTState:
    return GRMHDCTState(
        base.magnetic_flux + scale * rate.magnetic_rate,
        base.vector_potential + scale * rate.vector_potential_rate,
        base.gauge_scalar + scale * rate.gauge_scalar_rate,
    )


def _ct_weighted(
    base: GRMHDCTState,
    scale: Array,
    first: GRMHDCTRate,
    second: GRMHDCTRate,
    /,
) -> GRMHDCTState:
    return GRMHDCTState(
        base.magnetic_flux + scale * 0.5 * (first.magnetic_rate + second.magnetic_rate),
        base.vector_potential
        + scale * 0.5 * (first.vector_potential_rate + second.vector_potential_rate),
        base.gauge_scalar
        + scale * 0.5 * (first.gauge_scalar_rate + second.gauge_scalar_rate),
    )


class FixedGridGRRMHDIMEXPlan(StrictModule, NonTrainableState):
    """Atomic ideal-GRMHD plus gray-M1 IMEX-SSP2(2,2,2) runtime."""

    material_transport: GRMHDSSPRK3Plan
    radiation_transport: FixedGridGRM1SSPRK3Plan
    source: GRRMHDImplicitSourcePlan
    tableau: AdditiveIMEXTableau
    balance_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        material_transport: GRMHDSSPRK3Plan,
        radiation_transport: FixedGridGRM1SSPRK3Plan,
        source: GRRMHDImplicitSourcePlan,
        /,
        *,
        balance_tolerance: float = 1.0e-8,
    ) -> None:
        if not isinstance(material_transport, GRMHDSSPRK3Plan):
            raise TypeError("material_transport must be GRMHDSSPRK3Plan.")
        if not isinstance(radiation_transport, FixedGridGRM1SSPRK3Plan):
            raise TypeError("radiation_transport must be FixedGridGRM1SSPRK3Plan.")
        if not isinstance(source, GRRMHDImplicitSourcePlan):
            raise TypeError("source must be GRRMHDImplicitSourcePlan.")
        if source.material.system_id != material_transport.system.system_id:
            raise ValueError("GRRMHD source and transport material systems differ.")
        if source.interaction.radiation.system_id != radiation_transport.system.system_id:
            raise ValueError("GRRMHD source and transport radiation systems differ.")
        material_grid = material_transport.constrained_transport.bridge.grid
        radiation_grid = radiation_transport.discretization.grid
        if (
            material_grid.topology.topology_id != radiation_grid.topology.topology_id
            or tuple(material_grid.shape) != tuple(radiation_grid.shape)
        ):
            raise ValueError("GRRMHD material and radiation grids differ.")
        tolerance = float(balance_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("GRRMHD balance_tolerance must be finite and positive.")
        gamma = 1.0 - 1.0 / np.sqrt(2.0)
        tableau = AdditiveIMEXTableau(
            np.asarray(((0.0, 0.0), (1.0, 0.0))),
            np.asarray(((gamma, 0.0), (1.0 - 2.0 * gamma, gamma))),
            np.asarray((0.5, 0.5)),
            np.asarray((gamma, 1.0)),
        )
        self.material_transport = material_transport
        self.radiation_transport = radiation_transport
        self.source = source
        self.tableau = tableau
        self.balance_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-grid-grrmhd-imex-ssp2-222",
                "material_transport": material_transport.plan_id,
                "radiation_transport": radiation_transport.plan_id,
                "source": source.plan_id,
                "tableau": tableau.tableau_id,
                "balance_tolerance": tolerance,
            }
        )

    @property
    def cell_shape(self) -> tuple[int, ...]:
        return self.material_transport.cell_shape

    def initialize(
        self,
        material_conserved: ArrayLike,
        radiation_moments: ArrayLike,
        geometry: ValenciaFiniteVolumeStageGeometry,
        /,
        *,
        magnetic_flux: ArrayLike | None = None,
        vector_potential: ArrayLike | None = None,
        gauge_scalar: ArrayLike | None = None,
        composition: ArrayLike | None = None,
        time: ArrayLike = 0.0,
        step_size: ArrayLike | None = None,
    ) -> GRRMHDState:
        material = self.material_transport.initialize(
            material_conserved,
            geometry.cell,
            magnetic_flux=magnetic_flux,
            vector_potential=vector_potential,
            gauge_scalar=gauge_scalar,
            composition=composition,
            time=time,
            step_size=step_size,
        )
        radiation = self.radiation_transport.initialize(
            radiation_moments,
            geometry,
            time=time,
            step_size=step_size,
        )
        return GRRMHDState(
            material.material_state,
            material.constrained_transport,
            radiation.radiation_state,
            material.time,
            material.step_size,
            material.accepted_step,
            jnp.asarray(int(GRRMHDRunStatus.SUCCESS), dtype=jnp.int32),
        )

    def _source_step(
        self,
        material: Array,
        transport: GRMHDCTState,
        radiation: Array,
        coefficient: Array,
        geometry: ValenciaFiniteVolumeStageGeometry,
        composition: ArrayLike | None,
        /,
    ) -> GRRMHDSourceResult:
        full = self.material_transport.constrained_transport.full_state(
            material, transport.magnetic_flux
        )
        return self.source.advance(
            full,
            radiation,
            coefficient,
            geometry.cell,
            composition,
        )

    def _material_reduced(self, full: Array, /) -> Array:
        return self.material_transport.constrained_transport.layout.reduce_full_state(
            full
        )

    def _ledger(
        self,
        state: GRRMHDState,
        candidate: GRRMHDState,
        step: Array,
        material_rates: tuple[GRMHDSpatialRate, GRMHDSpatialRate],
        radiation_rates: tuple[GRM1SpatialRate, GRM1SpatialRate],
        sources: tuple[GRRMHDSourceResult, GRRMHDSourceResult],
        accepted: Array,
        /,
    ) -> GRRMHDDefectLedger:
        material_first, material_second = material_rates
        radiation_first, radiation_second = radiation_rates
        material_faces = tuple(
            0.5 * step * (first + second)
            for first, second in zip(
                material_first.integrated_face_fluxes,
                material_second.integrated_face_fluxes,
                strict=True,
            )
        )
        radiation_faces = tuple(
            0.5 * step * (first + second)
            for first, second in zip(
                radiation_first.integrated_face_fluxes,
                radiation_second.integrated_face_fluxes,
                strict=True,
            )
        )
        material_geometry = (
            0.5
            * step
            * (material_first.geometric_source + material_second.geometric_source)
        )
        radiation_geometry = (
            0.5
            * step
            * (radiation_first.geometric_source + radiation_second.geometric_source)
        )
        material_explicit = (
            0.5 * step * (material_first.material_rate + material_second.material_rate)
        )
        radiation_explicit = (
            0.5
            * step
            * (radiation_first.radiation_rate + radiation_second.radiation_rate)
        )
        material_exchange = (
            candidate.material_state - state.material_state - material_explicit
        )
        radiation_exchange = (
            candidate.radiation_state - state.radiation_state - radiation_explicit
        )
        volumes = self.radiation_transport.discretization.cell_volumes.astype(
            candidate.material_state.dtype
        )
        axes = tuple(range(len(self.cell_shape)))
        combined_energy = jnp.sum(
            volumes * (material_exchange[..., 4] + radiation_exchange[..., 0]),
            axis=axes,
        )
        combined_momentum = jnp.sum(
            volumes[..., None]
            * (material_exchange[..., 1:4] + radiation_exchange[..., 1:]),
            axis=axes,
        )
        source_energy = sum(jnp.sum(value.ledger.energy_defect) for value in sources)
        source_momentum = sum(
            jnp.sum(value.ledger.momentum_defect, axis=axes) for value in sources
        )
        edge_integral = (
            0.5
            * step
            * (
                material_first.transport_rate.edge_electromotive_circulation
                + material_second.transport_rate.edge_electromotive_circulation
            )
        )
        ct = self.material_transport.constrained_transport.defects(
            state.constrained_transport,
            candidate.constrained_transport,
            edge_integral,
        )
        arrays = (
            combined_energy,
            combined_momentum,
            source_energy,
            source_momentum,
        )
        finite = (
            jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in arrays)))
            & ct.finite
        )
        scale = jnp.maximum(
            jnp.maximum(
                jnp.abs(combined_energy),
                jnp.max(jnp.abs(combined_momentum), initial=0.0),
            ),
            1.0,
        )
        tolerance = jnp.maximum(
            jnp.asarray(self.balance_tolerance, dtype=scale.dtype),
            512.0 * jnp.finfo(scale.dtype).eps * scale,
        )
        qualified = (
            finite
            & ct.qualified
            & (jnp.abs(combined_energy) <= tolerance)
            & (jnp.max(jnp.abs(combined_momentum), initial=0.0) <= tolerance)
            & (jnp.abs(source_energy) <= tolerance)
            & (jnp.max(jnp.abs(source_momentum), initial=0.0) <= tolerance)
        )
        return GRRMHDDefectLedger(
            material_faces,
            radiation_faces,
            material_geometry,
            radiation_geometry,
            material_exchange,
            radiation_exchange,
            combined_energy,
            combined_momentum,
            source_energy,
            source_momentum,
            ct,
            finite,
            qualified,
            jnp.asarray(accepted, dtype=jnp.bool_),
            self.plan_id,
        )

    def _accepted_ledger(
        self, ledger: GRRMHDDefectLedger, accepted: Array, /
    ) -> GRRMHDDefectLedger:
        selected = lambda value: jnp.where(accepted, value, jnp.zeros_like(value))
        ct = ledger.constrained_transport
        accepted_ct = GRMHDCTDefectLedger(
            selected(ct.magnetic_flux_change),
            selected(ct.integrated_edge_electromotive),
            selected(ct.faraday_balance_defect),
            selected(ct.divergence_before),
            selected(ct.divergence_after),
            selected(ct.divergence_change),
            selected(ct.vector_potential_defect),
            selected(ct.gauge_constraint),
            ct.finite,
            ct.physically_valid,
            ct.qualified,
            ct.plan_id,
        )
        return GRRMHDDefectLedger(
            tuple(selected(value) for value in ledger.material_face_flux_integrals),
            tuple(selected(value) for value in ledger.radiation_face_flux_integrals),
            selected(ledger.material_geometric_source_integral),
            selected(ledger.radiation_geometric_source_integral),
            selected(ledger.material_exchange_change),
            selected(ledger.radiation_exchange_change),
            selected(ledger.combined_energy_defect),
            selected(ledger.combined_momentum_defect),
            selected(ledger.source_energy_defect),
            selected(ledger.source_momentum_defect),
            accepted_ct,
            ledger.finite,
            ledger.qualified,
            accepted,
            ledger.plan_id,
        )

    def propose_stage(
        self,
        base: GRRMHDState,
        working: GRRMHDState,
        stage_time: ArrayLike,
        increment: ArrayLike,
        geometry: ValenciaFiniteVolumeStageGeometry,
        composition: ArrayLike | None = None,
        /,
        *,
        transport_extinction: ArrayLike = 0.0,
    ) -> GRRMHDStageProposal:
        """Propose one source-implicit stage for a coupled spacetime coordinator."""

        if not isinstance(base, GRRMHDState) or not isinstance(working, GRRMHDState):
            raise TypeError("base and working must be GRRMHDState values.")
        step = jnp.asarray(increment, dtype=working.time.dtype).reshape(())
        time = jnp.asarray(stage_time, dtype=working.time.dtype).reshape(())
        source = self._source_step(
            working.material_state,
            working.constrained_transport,
            working.radiation_state,
            step,
            geometry,
            composition,
        )
        sourced_material = self._material_reduced(source.material_state)
        sourced_radiation = source.radiation_state
        material_rate = self.material_transport.rate(
            time,
            sourced_material,
            working.constrained_transport,
            geometry.cell,
            composition,
            source_geometry=geometry.source,
        )
        radiation_rate = self.radiation_transport.rate(
            time,
            sourced_radiation,
            geometry,
            transport_extinction=transport_extinction,
        )
        safe_step = jnp.where(step != 0.0, step, 1.0)
        material_implicit = (sourced_material - working.material_state) / safe_step
        radiation_implicit = (sourced_radiation - working.radiation_state) / safe_step
        candidate = GRRMHDState(
            base.material_state
            + step * (material_rate.material_rate + material_implicit),
            _ct_add(base.constrained_transport, step, material_rate.transport_rate),
            base.radiation_state
            + step * (radiation_rate.radiation_rate + radiation_implicit),
            time,
            step,
            working.accepted_step,
            jnp.asarray(int(GRRMHDRunStatus.SUCCESS), dtype=jnp.int32),
        )
        total_stress = combine_stress_energy_projections(
            (material_rate.stress_energy, radiation_rate.stress_energy)
        )
        finite = source.finite & material_rate.finite & radiation_rate.finite
        converged = source.converged & material_rate.converged
        physical = (
            source.physically_valid
            & material_rate.physically_valid
            & radiation_rate.physically_valid
        )
        qualified = source.qualified & material_rate.qualified & radiation_rate.qualified
        derivative = (
            source.derivative_valid
            & material_rate.derivative_valid
            & radiation_rate.derivative_valid
        )
        successful = finite & converged & physical & qualified
        status = jnp.where(
            successful,
            int(GRRMHDRunStatus.SUCCESS),
            jnp.where(
                ~source.accepted,
                int(GRRMHDRunStatus.IMPLICIT_SOURCE_FAILED),
                jnp.where(
                    ~finite,
                    int(GRRMHDRunStatus.NONFINITE_STATE),
                    int(GRRMHDRunStatus.CONSERVATION_DEFECT),
                ),
            ),
        ).astype(jnp.int32)
        candidate = eqx.tree_at(lambda value: value.status, candidate, status)
        return GRRMHDStageProposal(
            candidate,
            source,
            material_rate,
            radiation_rate,
            total_stress,
            finite,
            converged,
            physical,
            qualified,
            derivative,
        )

    def advance(
        self,
        state: GRRMHDState,
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
    ) -> GRRMHDStepResult:
        if not isinstance(state, GRRMHDState):
            raise TypeError("state must be GRRMHDState.")
        geometries = tuple(stage_geometries)
        if len(geometries) != 2:
            raise ValueError("GRRMHD IMEX-SSP2 requires two stage geometries.")
        start = jnp.asarray(start_time, dtype=state.time.dtype).reshape(())
        end = jnp.asarray(end_time, dtype=state.time.dtype).reshape(())
        step = end - start
        tolerance = 32.0 * jnp.finfo(start.dtype).eps * jnp.maximum(jnp.abs(start), 1.0)
        start = eqx.error_if(
            start,
            ~jnp.isfinite(start)
            | ~jnp.isfinite(end)
            | (step <= 0.0)
            | (jnp.abs(start - state.time) > tolerance),
            "GRRMHD interval is invalid or state time is stale.",
        )
        gamma = self.tableau.implicit_matrix[0, 0].astype(step.dtype)
        source_1 = self._source_step(
            state.material_state,
            state.constrained_transport,
            state.radiation_state,
            gamma * step,
            geometries[0],
            composition,
        )
        material_1 = self._material_reduced(source_1.material_state)
        radiation_1 = source_1.radiation_state
        transport_1 = state.constrained_transport
        material_rate_1 = self.material_transport.rate(
            start + gamma * step,
            material_1,
            transport_1,
            geometries[0].cell,
            composition,
            source_geometry=geometries[0].source,
        )
        radiation_rate_1 = self.radiation_transport.rate(
            start + gamma * step,
            radiation_1,
            geometries[0],
            transport_extinction=transport_extinction,
        )
        safe_source_coefficient = jnp.where(gamma * step != 0.0, gamma * step, 1.0)
        material_implicit_1 = (
            material_1 - state.material_state
        ) / safe_source_coefficient
        radiation_implicit_1 = (
            radiation_1 - state.radiation_state
        ) / safe_source_coefficient
        material_provisional = state.material_state + step * (
            material_rate_1.material_rate + (1.0 - 2.0 * gamma) * material_implicit_1
        )
        radiation_provisional = state.radiation_state + step * (
            radiation_rate_1.radiation_rate + (1.0 - 2.0 * gamma) * radiation_implicit_1
        )
        transport_provisional = _ct_add(
            state.constrained_transport,
            step,
            material_rate_1.transport_rate,
        )
        source_2 = self._source_step(
            material_provisional,
            transport_provisional,
            radiation_provisional,
            gamma * step,
            geometries[1],
            composition,
        )
        material_2 = self._material_reduced(source_2.material_state)
        radiation_2 = source_2.radiation_state
        transport_2 = transport_provisional
        material_rate_2 = self.material_transport.rate(
            end,
            material_2,
            transport_2,
            geometries[1].cell,
            composition,
            source_geometry=geometries[1].source,
        )
        radiation_rate_2 = self.radiation_transport.rate(
            end,
            radiation_2,
            geometries[1],
            transport_extinction=transport_extinction,
        )
        material_implicit_2 = (
            material_2 - material_provisional
        ) / safe_source_coefficient
        radiation_implicit_2 = (
            radiation_2 - radiation_provisional
        ) / safe_source_coefficient
        material_final = state.material_state + 0.5 * step * (
            material_rate_1.material_rate
            + material_rate_2.material_rate
            + material_implicit_1
            + material_implicit_2
        )
        radiation_final = state.radiation_state + 0.5 * step * (
            radiation_rate_1.radiation_rate
            + radiation_rate_2.radiation_rate
            + radiation_implicit_1
            + radiation_implicit_2
        )
        transport_final = _ct_weighted(
            state.constrained_transport,
            step,
            material_rate_1.transport_rate,
            material_rate_2.transport_rate,
        )
        candidate = GRRMHDState(
            material_final,
            transport_final,
            radiation_final,
            end,
            step,
            state.accepted_step + jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(int(GRRMHDRunStatus.SUCCESS), dtype=jnp.int32),
        )
        attempted = self._ledger(
            state,
            candidate,
            step,
            (material_rate_1, material_rate_2),
            (radiation_rate_1, radiation_rate_2),
            (source_1, source_2),
            jnp.asarray(False),
        )
        full_final = self.material_transport.constrained_transport.full_state(
            material_final, transport_final.magnetic_flux
        )
        recovery = self.material_transport.system.recover(
            full_final, geometries[1].cell, composition
        )
        final_moments = self.radiation_transport.moments(
            radiation_final, geometries[1].cell
        )
        closure = self.radiation_transport.system.closure(
            final_moments[..., 0], final_moments[..., 1:], geometries[1].cell
        )
        total_stress = (
            combine_stress_energy_projections(
                (material_rate_1.stress_energy, radiation_rate_1.stress_energy)
            ),
            combine_stress_energy_projections(
                (material_rate_2.stress_energy, radiation_rate_2.stress_energy)
            ),
        )
        finite = (
            material_rate_1.finite
            & material_rate_2.finite
            & radiation_rate_1.finite
            & radiation_rate_2.finite
            & source_1.finite
            & source_2.finite
            & attempted.finite
            & jnp.all(recovery.finite | ~geometries[1].cell.active)
            & jnp.all(closure.finite | ~geometries[1].cell.active)
        )
        converged = (
            source_1.converged
            & source_2.converged
            & jnp.all(recovery.converged | ~geometries[1].cell.active)
        )
        physical = (
            material_rate_1.physically_valid
            & material_rate_2.physically_valid
            & radiation_rate_1.physically_valid
            & radiation_rate_2.physically_valid
            & source_1.physically_valid
            & source_2.physically_valid
            & jnp.all(recovery.physically_valid | ~geometries[1].cell.active)
            & jnp.all(closure.physically_valid | ~geometries[1].cell.active)
        )
        qualified = (
            material_rate_1.qualified
            & material_rate_2.qualified
            & radiation_rate_1.qualified
            & radiation_rate_2.qualified
            & source_1.qualified
            & source_2.qualified
            & attempted.qualified
            & jnp.all(recovery.qualified | ~geometries[1].cell.active)
            & jnp.all(closure.qualified | ~geometries[1].cell.active)
        )
        derivative = (
            material_rate_1.derivative_valid
            & material_rate_2.derivative_valid
            & radiation_rate_1.derivative_valid
            & radiation_rate_2.derivative_valid
            & source_1.derivative_valid
            & source_2.derivative_valid
            & jnp.all(recovery.derivative_valid | ~geometries[1].cell.active)
            & jnp.all(closure.derivative_valid | ~geometries[1].cell.active)
        )
        stable_steps = jnp.stack(
            (
                material_rate_1.stable_step,
                material_rate_2.stable_step,
                radiation_rate_1.stable_step,
                radiation_rate_2.stable_step,
            )
        )
        stable = step <= jnp.min(stable_steps) + tolerance
        accepted = finite & converged & physical & qualified & stable
        status = jnp.where(
            accepted,
            int(GRRMHDRunStatus.SUCCESS),
            jnp.where(
                ~finite,
                int(GRRMHDRunStatus.NONFINITE_STATE),
                jnp.where(
                    ~(source_1.accepted & source_2.accepted),
                    int(GRRMHDRunStatus.IMPLICIT_SOURCE_FAILED),
                    jnp.where(
                        ~jnp.all(recovery.qualified | ~geometries[1].cell.active),
                        int(GRRMHDRunStatus.MATERIAL_RECOVERY_FAILED),
                        jnp.where(
                            ~jnp.all(closure.qualified | ~geometries[1].cell.active),
                            int(GRRMHDRunStatus.RADIATION_REALIZABILITY_FAILED),
                            jnp.where(
                                ~stable,
                                int(GRRMHDRunStatus.STABILITY_LIMIT_EXCEEDED),
                                int(GRRMHDRunStatus.CONSERVATION_DEFECT),
                            ),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        candidate = eqx.tree_at(lambda value: value.status, candidate, status)
        rejected = GRRMHDState(
            state.material_state,
            state.constrained_transport,
            state.radiation_state,
            state.time,
            state.step_size,
            state.accepted_step,
            status,
        )
        accepted_state = jax.lax.cond(
            accepted, lambda _: candidate, lambda _: rejected, operand=None
        )
        attempted = eqx.tree_at(lambda value: value.accepted, attempted, accepted)
        accepted_ledger = self._accepted_ledger(attempted, accepted)
        stages = GRRMHDStageEvidence(
            (material_rate_1, material_rate_2),
            (radiation_rate_1, radiation_rate_2),
            (source_1, source_2),
            total_stress,
            stable_steps,
            finite,
            converged,
            physical,
            qualified,
            derivative,
        )
        return GRRMHDStepResult(
            candidate,
            accepted_state,
            accepted,
            status,
            stages,
            attempted,
            accepted_ledger,
            finite,
            converged,
            physical,
            qualified,
            derivative,
        )


__all__ = [
    "FixedGridGRRMHDIMEXPlan",
    "GRRMHDDefectLedger",
    "GRRMHDRunStatus",
    "GRRMHDStageEvidence",
    "GRRMHDStageProposal",
    "GRRMHDState",
    "GRRMHDStepResult",
]
