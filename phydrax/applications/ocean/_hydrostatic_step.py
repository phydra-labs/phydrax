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
from ._external_mode import ExternalModeSubcycleSchedule
from ._hydrostatic import (
    _cell_from_faces,
    _faces_from_cell,
    _safe_divide,
    HydrostaticBoundaryTraces,
    HydrostaticOceanState,
    PreparedHydrostaticOcean,
)


class HydrostaticOceanLedger(StrictModule):
    volume_change: Array
    freshwater_volume: Array
    open_boundary_volume: Array
    tracer_change: dict[str, Array]
    tracer_source: dict[str, Array]
    kinetic_energy_change: Array
    free_surface_energy_change: Array
    coriolis_work: Array
    mixing_dissipation: Array
    limiter_correction: Array
    filter_correction: Array
    reconciliation_correction: Array
    residual: Array

    @classmethod
    def zeros(cls, tracer_names: tuple[str, ...], dtype, /) -> "HydrostaticOceanLedger":
        zero = jnp.zeros((), dtype=dtype)
        return cls(
            zero,
            zero,
            zero,
            {name: zero for name in tracer_names},
            {name: zero for name in tracer_names},
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
        )


class HydrostaticContinuationState(StrictModule):
    state: HydrostaticOceanState
    ledger: HydrostaticOceanLedger
    filtered_eta: Array
    filtered_barotropic_transport: tuple[Array, Array]
    subcycle_phase: Array
    subcycle_schedule: ExternalModeSubcycleSchedule

    @classmethod
    def initialize(
        cls,
        ocean: PreparedHydrostaticOcean,
        state: HydrostaticOceanState,
        /,
    ) -> "HydrostaticContinuationState":
        if not isinstance(ocean, PreparedHydrostaticOcean):
            raise TypeError("ocean must be a PreparedHydrostaticOcean.")
        names = tuple(sorted(state.tracer_inventory))
        zero_x = jnp.zeros(state.transports[0].shape[:-1], dtype=state.eta.dtype)
        zero_y = jnp.zeros(state.transports[1].shape[:-1], dtype=state.eta.dtype)
        return cls(
            state,
            HydrostaticOceanLedger.zeros(names, state.eta.dtype),
            state.eta,
            (zero_x, zero_y),
            jnp.asarray(0, dtype=jnp.int32),
            ocean.plan.subcycle_policy.empty(state.eta.dtype),
        )


class HydrostaticAdvanceEvidence(StrictModule):
    successful: Array
    eos_valid: Array
    eos_finite: Array
    eos_successful: Array
    volume_residual: Array
    tracer_residual: Array
    free_surface_residual: Array
    mixing_residual: Array
    limiter_correction: Array
    filter_correction: Array
    reconciliation_correction: Array

    subcycle_schedule: ExternalModeSubcycleSchedule


def _transport_kinetic_energy(
    ocean: PreparedHydrostaticOcean,
    state: HydrostaticOceanState,
    /,
) -> Array:
    epoch = ocean.geometry.metric_epoch(state.eta)
    velocity = (
        _cell_from_faces(
            _safe_divide(state.transports[0], epoch.x_face_area),
            0,
            ocean.geometry.periodic[0],
        ),
        _cell_from_faces(
            _safe_divide(state.transports[1], epoch.y_face_area),
            1,
            ocean.geometry.periodic[1],
        ),
    )
    speed_squared = ocean.geometry.normal_velocity_inner_product(velocity, velocity)
    return 0.5 * ocean.plan.reference_density * jnp.sum(epoch.cell_volume * speed_squared)


def _surface_energy(ocean: PreparedHydrostaticOcean, eta: Array, /) -> Array:
    return (
        0.5
        * ocean.plan.reference_density
        * ocean.plan.gravity
        * jnp.sum(ocean.geometry.cell_area * eta**2)
    )


def _face_donor_factor(
    cell_factor: Array,
    face_flux: Array,
    axis: int,
    periodic: bool,
    /,
) -> Array:
    moved = jnp.moveaxis(cell_factor, axis, 0)
    moved_flux = jnp.moveaxis(face_flux, axis, 0)
    if periodic:
        left = jnp.roll(moved, 1, axis=0)
        right = moved
    else:
        left = jnp.concatenate((moved[:1], moved), axis=0)
        right = jnp.concatenate((moved, moved[-1:]), axis=0)
    factor = jnp.where(moved_flux >= 0.0, left, right)
    return jnp.moveaxis(factor, 0, axis)


def _cell_outgoing_transport(transport: Array, axis: int, periodic: bool, /) -> Array:
    moved = jnp.moveaxis(transport, axis, 0)
    if periodic:
        lower = moved
        upper = jnp.roll(moved, -1, axis=0)
    else:
        lower = moved[:-1]
        upper = moved[1:]
    outgoing = jnp.maximum(-lower, 0.0) + jnp.maximum(upper, 0.0)
    return jnp.moveaxis(outgoing, 0, axis)


class HydrostaticIMEXMidpointMethod(AbstractFixedStepMethod):
    """Stage-consistent midpoint IMEX method with implicit/split external mode."""

    ocean: PreparedHydrostaticOcean
    method_id: str = eqx.field(static=True)

    def __init__(self, ocean: PreparedHydrostaticOcean, /):
        if not isinstance(ocean, PreparedHydrostaticOcean):
            raise TypeError("ocean must be PreparedHydrostaticOcean.")
        self.ocean = ocean
        self.method_id = canonical_fingerprint(
            {
                "kind": "hydrostatic-imex-midpoint",
                "ocean": ocean.prepared_id,
                "external_mode": ocean.plan.external_mode,
                "subcycle_policy": ocean.plan.subcycle_policy.policy_id,
            }
        )

    def _explicit_transport_tendency(
        self,
        state: HydrostaticOceanState,
        time: Array,
        args: Any,
        /,
        *,
        boundary_traces: HydrostaticBoundaryTraces | None = None,
    ) -> tuple[tuple[Array, Array], Array, Array, Array, Array]:
        del time, args
        epoch = self.ocean.geometry.metric_epoch(state.eta)
        view = self.ocean.view(state)
        hydro_force = self.ocean.geometry.layer_potential_transport_force(
            view.hydrostatic_pressure,
            epoch,
            boundary_values=(
                None if boundary_traces is None else boundary_traces.hydrostatic_pressure
            ),
        )
        coriolis = self.ocean._coriolis_force(state, epoch)
        velocity = (
            _cell_from_faces(view.velocity[0], 0, self.ocean.geometry.periodic[0]),
            _cell_from_faces(view.velocity[1], 1, self.ocean.geometry.periodic[1]),
        )
        velocity_boundary = (
            (None, None) if boundary_traces is None else boundary_traces.velocity
        )
        u_gradient = self.ocean.geometry.layer_gradient(
            velocity[0],
            boundary_values=(None if boundary_traces is None else velocity_boundary[0]),
        )
        v_gradient = self.ocean.geometry.layer_gradient(
            velocity[1],
            boundary_values=(None if boundary_traces is None else velocity_boundary[1]),
        )
        u_gradient_cell = (
            _cell_from_faces(u_gradient[0], 0, self.ocean.geometry.periodic[0]),
            _cell_from_faces(u_gradient[1], 1, self.ocean.geometry.periodic[1]),
        )
        v_gradient_cell = (
            _cell_from_faces(v_gradient[0], 0, self.ocean.geometry.periodic[0]),
            _cell_from_faces(v_gradient[1], 1, self.ocean.geometry.periodic[1]),
        )
        adv_u_cell = -self.ocean.geometry.normal_velocity_inner_product(
            velocity, u_gradient_cell
        )
        adv_v_cell = -self.ocean.geometry.normal_velocity_inner_product(
            velocity, v_gradient_cell
        )
        adv_u = epoch.x_face_area * _faces_from_cell(
            adv_u_cell, 0, self.ocean.geometry.periodic[0]
        )
        adv_v = epoch.y_face_area * _faces_from_cell(
            adv_v_cell, 1, self.ocean.geometry.periodic[1]
        )
        tendency = (
            hydro_force[0] + coriolis[0] + adv_u,
            hydro_force[1] + coriolis[1] + adv_v,
        )
        coriolis_acceleration = (
            _cell_from_faces(
                _safe_divide(coriolis[0], epoch.x_face_area),
                0,
                self.ocean.geometry.periodic[0],
            ),
            _cell_from_faces(
                _safe_divide(coriolis[1], epoch.y_face_area),
                1,
                self.ocean.geometry.periodic[1],
            ),
        )
        coriolis_work = self.ocean.plan.reference_density * jnp.sum(
            epoch.cell_volume
            * self.ocean.geometry.normal_velocity_inner_product(
                velocity, coriolis_acceleration
            )
        )
        return (
            tendency,
            coriolis_work,
            view.eos_valid,
            view.eos_finite,
            view.eos_successful,
        )

    def _apply_boundaries(
        self,
        eta: Array,
        barotropic: tuple[Array, Array],
        time: Array,
        /,
    ) -> tuple[Array, tuple[Array, Array], Array]:
        del time
        eta_ = eta
        transports = [barotropic[0], barotropic[1]]
        boundary_flux = jnp.asarray(0.0, dtype=eta.dtype)
        for boundary in self.ocean.plan.boundaries:
            axis = boundary.axis
            index = 0 if boundary.side == "lower" else -1
            sign = -1.0 if boundary.side == "lower" else 1.0
            if boundary.kind == "prescribed-elevation":
                location = [slice(None)] * 2
                location[axis] = index
                eta_ = eta_.at[tuple(location)].set(boundary.target_eta)
                continue
            location = [slice(None)] * transports[axis].ndim
            location[axis] = index
            target = jnp.asarray(boundary.target_transport, dtype=eta.dtype)
            if boundary.kind == "closed":
                target = jnp.asarray(0.0, dtype=eta.dtype)
            elif boundary.kind in ("flather", "radiation"):
                cell_location = [slice(None)] * 2
                cell_location[axis] = 0 if boundary.side == "lower" else -1
                local_eta = eta_[tuple(cell_location)]
                local_depth = (
                    self.ocean.geometry.rest_depth[tuple(cell_location)] + local_eta
                )
                edge = (
                    self.ocean.geometry.x_edge_length[tuple(location[:2])]
                    if axis == 0
                    else self.ocean.geometry.y_edge_length[tuple(location[:2])]
                )
                target = target + sign * edge * jnp.sqrt(
                    self.ocean.plan.gravity * jnp.maximum(local_depth, 0.0)
                ) * (local_eta - boundary.target_eta)
            transports[axis] = transports[axis].at[tuple(location)].set(target)
            boundary_flux = boundary_flux + sign * jnp.sum(target)
        return eta_, (transports[0], transports[1]), boundary_flux

    def _limit_barotropic_outflow(
        self,
        eta: Array,
        barotropic: tuple[Array, Array],
        dt: Array,
        /,
    ) -> tuple[tuple[Array, Array], Array]:
        if not self.ocean.plan.wetting_and_drying:
            return barotropic, jnp.asarray(0.0, dtype=eta.dtype)
        depth = jnp.maximum(self.ocean.geometry.rest_depth + eta, 0.0)
        available = self.ocean.geometry.cell_area * depth
        x, y = barotropic
        outgoing = _cell_outgoing_transport(
            x, 0, self.ocean.geometry.periodic[0]
        ) + _cell_outgoing_transport(y, 1, self.ocean.geometry.periodic[1])
        factor = jnp.minimum(
            1.0, _safe_divide(available, dt * jnp.maximum(outgoing, 0.0))
        )
        factor = jnp.where(depth > self.ocean.plan.wet_depth, factor, 0.0)
        x_factor = _face_donor_factor(factor, x, 0, self.ocean.geometry.periodic[0])
        y_factor = _face_donor_factor(factor, y, 1, self.ocean.geometry.periodic[1])
        limited = x * x_factor, y * y_factor
        correction = jnp.sum(jnp.abs(x - limited[0])) + jnp.sum(jnp.abs(y - limited[1]))
        return limited, correction

    def _reconcile_layers(
        self,
        transports: tuple[Array, Array],
        target_barotropic: tuple[Array, Array],
        epoch,
        /,
    ) -> tuple[Array, Array]:
        current = self.ocean.geometry.depth_integrate(transports)
        delta = (
            target_barotropic[0] - current[0],
            target_barotropic[1] - current[1],
        )
        x_fraction = _safe_divide(
            epoch.x_face_area,
            jnp.sum(epoch.x_face_area, axis=-1)[..., None],
        )
        y_fraction = _safe_divide(
            epoch.y_face_area,
            jnp.sum(epoch.y_face_area, axis=-1)[..., None],
        )
        return (
            transports[0] + x_fraction * delta[0][..., None],
            transports[1] + y_fraction * delta[1][..., None],
        )

    def _split_external(
        self,
        eta: Array,
        predictor: tuple[Array, Array],
        epoch,
        dt: Array,
        freshwater: Array,
        time: Array,
        /,
        *,
        surface_boundary=None,
    ) -> tuple[
        Array,
        tuple[Array, Array],
        Array,
        Array,
        Array,
        ExternalModeSubcycleSchedule,
    ]:
        barotropic = self.ocean.geometry.depth_integrate(predictor)
        schedule = self.ocean.plan.subcycle_policy.schedule(
            self.ocean.geometry,
            eta,
            dt,
            self.ocean.plan.gravity,
            barotropic_transport=barotropic,
        )
        initial = (
            eta,
            barotropic,
            (jnp.zeros_like(barotropic[0]), jnp.zeros_like(barotropic[1])),
            jnp.asarray(0.0, dtype=eta.dtype),
            jnp.asarray(0.0, dtype=eta.dtype),
            jnp.asarray(0.0, dtype=eta.dtype),
        )

        def subcycle(carry, index):
            (
                eta_fast,
                transport_fast,
                flux_integral,
                limiter_total,
                boundary_total,
                elapsed,
            ) = carry
            active = schedule.active_mask[index]
            fast_dt = schedule.substep_sizes[index]
            current_epoch = self.ocean.geometry.metric_epoch(eta_fast)
            gx, gy = self.ocean.geometry.surface_gradient(
                eta_fast, boundary_values=surface_boundary
            )
            x_depth = jnp.sum(current_epoch.x_face_area, axis=-1)
            y_depth = jnp.sum(current_epoch.y_face_area, axis=-1)
            transport_candidate = (
                transport_fast[0] - self.ocean.plan.gravity * fast_dt * x_depth * gx,
                transport_fast[1] - self.ocean.plan.gravity * fast_dt * y_depth * gy,
            )
            eta_boundary, transport_boundary, boundary_flux = self._apply_boundaries(
                eta_fast, transport_candidate, time + elapsed
            )
            transport_limited, limiter = self._limit_barotropic_outflow(
                eta_boundary, transport_boundary, fast_dt
            )
            net = self.ocean.geometry.surface_net_flux(transport_limited)
            eta_candidate = (
                eta_boundary
                - fast_dt * net / self.ocean.geometry.cell_area
                + fast_dt * freshwater
            )
            candidate = (
                eta_candidate,
                transport_limited,
                (
                    flux_integral[0] + fast_dt * transport_limited[0],
                    flux_integral[1] + fast_dt * transport_limited[1],
                ),
                limiter_total + limiter,
                boundary_total + fast_dt * boundary_flux,
                elapsed + jnp.abs(fast_dt),
            )
            selected = jax.tree.map(
                lambda proposed, current: jnp.where(active, proposed, current),
                candidate,
                carry,
            )
            return selected, None

        final, _ = jax.lax.scan(
            subcycle,
            initial,
            jnp.arange(self.ocean.plan.subcycle_policy.maximum_substeps),
        )
        eta_fast, _, flux_integral, limiter_total, boundary_total, _ = final
        averaged_barotropic = (
            flux_integral[0] / dt,
            flux_integral[1] / dt,
        )
        old_barotropic = self.ocean.geometry.depth_integrate(predictor)
        delta = (
            averaged_barotropic[0] - old_barotropic[0],
            averaged_barotropic[1] - old_barotropic[1],
        )
        x_fraction = _safe_divide(
            epoch.x_face_area,
            jnp.sum(epoch.x_face_area, axis=-1)[..., None],
        )
        y_fraction = _safe_divide(
            epoch.y_face_area,
            jnp.sum(epoch.y_face_area, axis=-1)[..., None],
        )
        reconciled = (
            predictor[0] + x_fraction * delta[0][..., None],
            predictor[1] + y_fraction * delta[1][..., None],
        )
        reconciliation = jnp.sum(
            jnp.abs(
                self.ocean.geometry.depth_integrate(reconciled)[0]
                - averaged_barotropic[0]
            )
        ) + jnp.sum(
            jnp.abs(
                self.ocean.geometry.depth_integrate(reconciled)[1]
                - averaged_barotropic[1]
            )
        )
        return (
            eta_fast,
            reconciled,
            limiter_total,
            reconciliation,
            boundary_total,
            schedule,
        )

    def _advance(
        self,
        base: HydrostaticOceanState,
        evaluation: HydrostaticOceanState,
        time: Array,
        dt: Array,
        args: Any,
        /,
        *,
        boundary_traces: HydrostaticBoundaryTraces | None = None,
    ) -> tuple[HydrostaticOceanState, HydrostaticAdvanceEvidence, HydrostaticOceanLedger]:
        epoch = self.ocean.geometry.metric_epoch(evaluation.eta)
        freshwater = self.ocean.plan.freshwater.evaluate(
            time, self.ocean.geometry.horizontal_shape, args
        )
        (
            tendency,
            coriolis_work,
            evaluation_eos_valid,
            evaluation_eos_finite,
            evaluation_eos_successful,
        ) = self._explicit_transport_tendency(
            evaluation, time, args, boundary_traces=boundary_traces
        )
        predictor = (
            base.transports[0] + dt * tendency[0],
            base.transports[1] + dt * tendency[1],
        )
        if self.ocean.plan.external_mode == "implicit":
            boundary_eta, boundary_barotropic, boundary_before = self._apply_boundaries(
                base.eta,
                self.ocean.geometry.depth_integrate(predictor),
                time,
            )
            predictor = self._reconcile_layers(predictor, boundary_barotropic, epoch)
            surface = self.ocean.free_surface.solve(
                boundary_eta,
                predictor,
                epoch,
                dt,
                freshwater,
                boundary_values=(
                    None if boundary_traces is None else boundary_traces.surface
                ),
            )
            eta_new, corrected_barotropic, boundary_after = self._apply_boundaries(
                surface.eta,
                self.ocean.geometry.depth_integrate(surface.transports),
                time + dt,
            )
            transport_new = self._reconcile_layers(
                surface.transports,
                corrected_barotropic,
                self.ocean.geometry.metric_epoch(eta_new),
            )
            limiter = jnp.asarray(0.0, dtype=dt.dtype)
            reconciliation = jnp.asarray(0.0, dtype=dt.dtype)
            boundary_volume = 0.5 * dt * (boundary_before + boundary_after)
            free_surface_residual = surface.residual_norm
            surface_success = surface.successful
            subcycle_schedule = self.ocean.plan.subcycle_policy.empty(dt.dtype)
        else:
            (
                eta_new,
                transport_new,
                limiter,
                reconciliation,
                boundary_volume,
                subcycle_schedule,
            ) = self._split_external(
                base.eta,
                predictor,
                epoch,
                dt,
                freshwater,
                time,
                surface_boundary=(
                    None if boundary_traces is None else boundary_traces.surface
                ),
            )
            free_surface_residual = jnp.sqrt(
                jnp.real(
                    jnp.vdot(
                        eta_new - base.eta,
                        eta_new - base.eta,
                    )
                )
            )
            surface_success = (
                jnp.all(jnp.isfinite(eta_new)) & subcycle_schedule.successful
            )
        new_epoch = self.ocean.geometry.metric_epoch(eta_new)
        vertical_flux = self.ocean.geometry.diagnose_vertical_flux(transport_new)
        transport_state = HydrostaticOceanState(
            eta_new,
            transport_new,
            evaluation.tracer_inventory,
            evaluation.tke_inventory,
        )
        tracer_rate = self.ocean._tracer_tendency(
            transport_state,
            new_epoch,
            vertical_flux,
            freshwater,
            boundary_traces=boundary_traces,
        )
        inventory = {
            name: base.tracer_inventory[name] + dt * tracer_rate[name]
            for name in base.tracer_inventory
        }
        tke = base.tke_inventory
        if self.ocean.plan.mixing.kind == "tke":
            tke_concentration = _safe_divide(evaluation.tke_inventory, epoch.cell_volume)
            evaluation_view = self.ocean.view(evaluation)
            layer_scale = jnp.maximum(epoch.layer_thickness, 1.0e-12)
            du_dz = (
                jnp.gradient(
                    _cell_from_faces(
                        evaluation_view.velocity[0],
                        0,
                        self.ocean.geometry.periodic[0],
                    ),
                    axis=-1,
                )
                / layer_scale
            )
            dv_dz = (
                jnp.gradient(
                    _cell_from_faces(
                        evaluation_view.velocity[1],
                        1,
                        self.ocean.geometry.periodic[1],
                    ),
                    axis=-1,
                )
                / layer_scale
            )
            production = self.ocean.plan.mixing.tke_coefficient * (du_dz**2 + dv_dz**2)
            mixing_length = jnp.maximum(
                epoch.total_depth[..., None] / self.ocean.geometry.cell_shape[-1],
                1.0e-6,
            )
            dissipation = _safe_divide(
                jnp.maximum(tke_concentration, 0.0) ** 1.5,
                mixing_length,
            )
            tke = jnp.maximum(
                0.0,
                base.tke_inventory + dt * epoch.cell_volume * (production - dissipation),
            )
        candidate = HydrostaticOceanState(
            eta_new,
            transport_new,
            inventory,
            tke,
        )
        mixed, mixing_residual = self.ocean.apply_vertical_mixing(
            candidate,
            new_epoch,
            dt,
            boundary_traces=boundary_traces,
        )
        mixed_view = self.ocean.view(mixed)
        eos_valid = evaluation_eos_valid & mixed_view.eos_valid
        eos_finite = evaluation_eos_finite & mixed_view.eos_finite
        eos_successful = evaluation_eos_successful & mixed_view.eos_successful
        volume_change = jnp.sum(self.ocean.geometry.cell_area * (eta_new - base.eta))
        freshwater_volume = dt * jnp.sum(self.ocean.geometry.cell_area * freshwater)
        tracer_change = {
            name: jnp.sum(mixed.tracer_inventory[name] - base.tracer_inventory[name])
            for name in base.tracer_inventory
        }
        tracer_source = {
            name: dt * jnp.sum(tracer_rate[name]) for name in base.tracer_inventory
        }
        volume_residual = volume_change - freshwater_volume + boundary_volume
        tracer_residuals = {
            name: tracer_change[name] - tracer_source[name] for name in tracer_change
        }
        tracer_residual = jnp.max(
            jnp.stack(tuple(jnp.abs(value) for value in tracer_residuals.values()))
        )
        kinetic_change = _transport_kinetic_energy(
            self.ocean, mixed
        ) - _transport_kinetic_energy(self.ocean, base)
        surface_change = _surface_energy(self.ocean, mixed.eta) - _surface_energy(
            self.ocean, base.eta
        )
        residual = jnp.maximum(
            jnp.abs(volume_residual),
            jnp.maximum(tracer_residual, free_surface_residual),
        )
        finite = (
            new_epoch.finite
            & jnp.all(jnp.isfinite(mixed.eta))
            & jnp.all(jnp.isfinite(mixed.transports[0]))
            & jnp.all(jnp.isfinite(mixed.transports[1]))
            & jnp.all(jnp.isfinite(mixed.tke_inventory))
            & jnp.all(
                jnp.stack(
                    tuple(
                        jnp.all(jnp.isfinite(value))
                        for value in mixed.tracer_inventory.values()
                    )
                )
            )
            & jnp.isfinite(mixing_residual)
            & eos_finite
        )
        successful = (
            surface_success
            & epoch.valid
            & new_epoch.valid
            & finite
            & eos_valid
            & eos_successful
        )
        ledger = HydrostaticOceanLedger(
            volume_change=volume_change,
            freshwater_volume=freshwater_volume,
            open_boundary_volume=-boundary_volume,
            tracer_change=tracer_change,
            tracer_source=tracer_source,
            kinetic_energy_change=kinetic_change,
            free_surface_energy_change=surface_change,
            coriolis_work=dt * coriolis_work,
            mixing_dissipation=jnp.maximum(0.0, mixing_residual),
            limiter_correction=limiter,
            filter_correction=jnp.asarray(0.0, dtype=dt.dtype),
            reconciliation_correction=reconciliation,
            residual=residual,
        )
        evidence = HydrostaticAdvanceEvidence(
            successful=successful,
            eos_valid=eos_valid,
            eos_finite=eos_finite,
            eos_successful=eos_successful,
            volume_residual=volume_residual,
            tracer_residual=tracer_residual,
            free_surface_residual=free_surface_residual,
            mixing_residual=mixing_residual,
            limiter_correction=limiter,
            filter_correction=jnp.asarray(0.0, dtype=dt.dtype),
            reconciliation_correction=reconciliation,
            subcycle_schedule=subcycle_schedule,
        )
        return mixed, evidence, ledger

    @staticmethod
    def _add_ledgers(
        left: HydrostaticOceanLedger,
        right: HydrostaticOceanLedger,
        /,
    ) -> HydrostaticOceanLedger:
        return jax.tree.map(lambda a, b: a + b, left, right)

    def step(
        self,
        step_index: Array,
        time: Array,
        state: HydrostaticContinuationState,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        del step_index
        dt = jnp.asarray(step_size, dtype=state.state.eta.dtype)
        midpoint, first_evidence, first_ledger = self._advance(
            state.state, state.state, time, 0.5 * dt, args
        )
        candidate, second_evidence, second_ledger = self._advance(
            state.state, midpoint, time + 0.5 * dt, dt, args
        )
        successful = first_evidence.successful & second_evidence.successful
        ledger = self._add_ledgers(state.ledger, second_ledger)
        accepted_state = HydrostaticContinuationState(
            candidate,
            ledger,
            0.5 * (state.filtered_eta + candidate.eta),
            self.ocean.geometry.depth_integrate(candidate.transports),
            state.subcycle_phase
            + jnp.where(
                self.ocean.plan.external_mode == "split-explicit",
                second_evidence.subcycle_schedule.count,
                jnp.asarray(1, dtype=jnp.int32),
            ),
            second_evidence.subcycle_schedule,
        )
        accepted = jax.tree.map(
            lambda proposed, current: jnp.where(successful, proposed, current),
            accepted_state,
            state,
        )
        residual = jnp.maximum(
            first_evidence.free_surface_residual,
            second_evidence.free_surface_residual,
        )
        return FixedStepResult(
            candidate_state=accepted_state,
            accepted_state=accepted,
            successful=successful,
            residual=residual,
            iterations=jnp.asarray(2, dtype=jnp.int32),
            work=jnp.asarray(2, dtype=jnp.int32),
            transform_applied=jnp.asarray(False),
            transform_correction_norm=jnp.asarray(0.0, dtype=dt.dtype),
        )


def write_hydrostatic_checkpoint(
    path: str | Path,
    ocean: PreparedHydrostaticOcean,
    method: HydrostaticIMEXMidpointMethod,
    time: ArrayLike,
    accepted_step: ArrayLike,
    state: HydrostaticContinuationState,
    /,
) -> Path:
    arrays: dict[str, object] = {
        "time": jnp.asarray(time),
        "accepted_step": jnp.asarray(accepted_step),
    }
    specification = pack_array_tree("state", state, arrays)
    return write_array_archive(
        path,
        manifest={
            "kind": "hydrostatic-ocean-checkpoint",
            "schema": "canonical",
            "ocean_id": ocean.prepared_id,
            "method_id": method.method_id,
            "state": specification,
        },
        arrays=arrays,
    )


def read_hydrostatic_checkpoint(
    path: str | Path,
    ocean: PreparedHydrostaticOcean,
    method: HydrostaticIMEXMidpointMethod,
    template: HydrostaticContinuationState,
    /,
) -> tuple[Array, Array, HydrostaticContinuationState]:
    manifest, arrays = read_array_archive(path)
    if manifest.get("kind") != "hydrostatic-ocean-checkpoint":
        raise ValueError("Archive is not a hydrostatic ocean checkpoint.")
    if manifest.get("ocean_id") != ocean.prepared_id:
        raise ValueError("Hydrostatic ocean checkpoint model identity mismatch.")
    if manifest.get("method_id") != method.method_id:
        raise ValueError("Hydrostatic ocean checkpoint method identity mismatch.")
    state = unpack_array_tree(manifest["state"], arrays, template)
    return jnp.asarray(arrays["time"]), jnp.asarray(arrays["accepted_step"]), state


__all__ = [
    "HydrostaticAdvanceEvidence",
    "HydrostaticContinuationState",
    "HydrostaticIMEXMidpointMethod",
    "HydrostaticOceanLedger",
    "read_hydrostatic_checkpoint",
    "write_hydrostatic_checkpoint",
]
