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
from ..linalg import (
    DenseLinearOperator,
    DenseSVD,
    LeastSquaresProblem,
    LinearSolvePolicy,
    LinearSolveResult,
    solve as solve_linear,
)
from ..operators.integral.vortex._panel_complete import (
    NativePanelFieldPlan2D,
    NativePanelGeometry2D,
)


class CompletePanelLoad2D(StrictModule):
    pressure_coefficient: Array
    panel_force: Array
    total_force: Array
    moment: Array
    blasius_force: Array
    impulse_force: Array | None
    agreement_defect: Array
    finite: Array


class CompletePanelResult2D(StrictModule):
    source_strength: Array | None
    vortex_strength: Array | None
    doublet_strength: Array | None
    surface_velocity: Array
    normal_residual: Array
    circulation_residual: Array
    kutta_residual: Array
    load: CompletePanelLoad2D
    linear_result: LinearSolveResult
    successful: Array
    solver_id: str = eqx.field(static=True)


class CompletePanelFlowPlan2D(StrictModule, NonTrainableState):
    geometry: NativePanelGeometry2D
    formulation: str = eqx.field(static=True)
    component_ids: Array
    trailing_edge_panels: tuple[int, int] | None = eqx.field(static=True)
    density: float = eqx.field(static=True)
    policy: LinearSolvePolicy
    solver_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: NativePanelGeometry2D,
        formulation: str = "source-vortex",
        /,
        *,
        component_ids: ArrayLike | None = None,
        trailing_edge_panels: tuple[int, int] | None = None,
        density: float = 1.0,
        policy: LinearSolvePolicy | None = None,
    ):
        if (
            not isinstance(geometry, NativePanelGeometry2D)
            or formulation not in ("source", "vortex", "source-vortex", "doublet")
            or float(density) <= 0.0
        ):
            raise ValueError("Complete panel geometry/formulation/density is invalid.")
        count = int(geometry.straight.length.size)
        components = (
            jnp.zeros((count,), dtype=jnp.int32)
            if component_ids is None
            else jnp.asarray(component_ids, dtype=jnp.int32)
        )
        if components.shape != (count,):
            raise ValueError("Panel component_ids must have panel-count shape.")
        trailing = (
            None
            if trailing_edge_panels is None
            else tuple(int(value) for value in trailing_edge_panels)
        )
        if trailing is not None and (
            len(trailing) != 2
            or any(value < 0 or value >= count for value in trailing)
            or trailing[0] == trailing[1]
        ):
            raise ValueError("Trailing-edge panel metadata is invalid.")
        self.geometry, self.formulation, self.component_ids, self.trailing_edge_panels = (
            geometry,
            formulation,
            components,
            trailing,
        )
        self.density, self.policy = (
            float(density),
            LinearSolvePolicy(DenseSVD()) if policy is None else policy,
        )
        self.solver_id = canonical_fingerprint(
            {
                "kind": "complete-panel-flow-2d",
                "geometry": geometry.geometry_id,
                "formulation": formulation,
                "components": tuple(int(value) for value in components),
                "trailing_edge_panels": trailing,
                "density": self.density,
            }
        )

    def solve(
        self,
        incident_velocity: ArrayLike,
        /,
        *,
        body_velocity: ArrayLike | None = None,
        prescribed_circulation: ArrayLike = 0.0,
        reference_point: ArrayLike = (0.0, 0.0),
        previous_impulse: ArrayLike | None = None,
        time_step: ArrayLike | None = None,
    ) -> CompletePanelResult2D:
        geometry = self.geometry.straight
        incident = jnp.asarray(incident_velocity, dtype=geometry.control.dtype)
        if incident.shape == (2,):
            incident = jnp.broadcast_to(incident, geometry.control.shape)
        body = (
            jnp.zeros_like(incident)
            if body_velocity is None
            else jnp.asarray(body_velocity, dtype=incident.dtype)
        )
        if incident.shape != geometry.control.shape or body.shape != incident.shape:
            raise ValueError("Panel incident/body velocities are incompatible.")
        relative = incident - body
        source_normal, source_tangent = NativePanelFieldPlan2D(self.geometry).influence(
            kind="source"
        )
        vortex_normal, vortex_tangent = NativePanelFieldPlan2D(self.geometry).influence(
            kind="vortex"
        )
        normal_rhs = -jnp.sum(relative * geometry.normal, axis=-1)
        count = int(geometry.length.size)
        circulation_target = jnp.asarray(prescribed_circulation, dtype=incident.dtype)
        if circulation_target.shape == ():
            circulation_target = jnp.broadcast_to(
                circulation_target, (int(jnp.max(self.component_ids)) + 1,)
            )
        component_count = int(circulation_target.size)
        circulation_rows = jnp.stack(
            tuple(
                jnp.where(self.component_ids == component, geometry.length, 0.0)
                for component in range(component_count)
            ),
            axis=0,
        )
        if self.formulation == "source":
            matrix, rhs = source_normal, normal_rhs
            source_slice, vortex_slice, doublet_slice = slice(0, count), None, None
        elif self.formulation == "vortex":
            matrix = jnp.concatenate((vortex_normal, circulation_rows), axis=0)
            rhs = jnp.concatenate((normal_rhs, circulation_target))
            source_slice, vortex_slice, doublet_slice = None, slice(0, count), None
        elif self.formulation == "source-vortex":
            matrix = jnp.block(
                [
                    [source_normal, vortex_normal],
                    [source_tangent, vortex_tangent],
                ]
            )
            rhs = jnp.concatenate(
                (normal_rhs, -jnp.sum(relative * geometry.tangent, axis=-1))
            )
            source_slice, vortex_slice, doublet_slice = (
                slice(0, count),
                slice(count, 2 * count),
                None,
            )
        else:
            # In 2-D potential flow a doublet sheet shares the vortex-sheet
            # normal influence; its potential is evaluated separately.
            matrix = jnp.concatenate((vortex_normal, circulation_rows), axis=0)
            rhs = jnp.concatenate((normal_rhs, circulation_target))
            source_slice, vortex_slice, doublet_slice = None, None, slice(0, count)
        if self.trailing_edge_panels is not None and matrix.shape[0] >= matrix.shape[1]:
            first, second = self.trailing_edge_panels
            kutta_row = jnp.zeros((matrix.shape[1],), dtype=matrix.dtype)
            offset = count if self.formulation == "source-vortex" else 0
            kutta_row = kutta_row.at[offset + first].set(1.0).at[offset + second].set(1.0)
            matrix = matrix.at[-1].set(kutta_row)
            rhs = rhs.at[-1].set(0.0)
        linear = solve_linear(
            LeastSquaresProblem(
                DenseLinearOperator(matrix), problem_id=f"{self.solver_id}:sheet"
            ),
            rhs,
            policy=self.policy,
        )
        value = jnp.asarray(linear.value)
        source_strength = None if source_slice is None else value[source_slice]
        vortex_strength = None if vortex_slice is None else value[vortex_slice]
        doublet_strength = None if doublet_slice is None else value[doublet_slice]
        surface_velocity = relative
        if source_strength is not None:
            surface_velocity = (
                surface_velocity
                + NativePanelFieldPlan2D(self.geometry)
                .evaluate(geometry.control, source_strength, kind="source")
                .velocity
            )
        sheet_for_velocity = (
            vortex_strength if vortex_strength is not None else doublet_strength
        )
        if sheet_for_velocity is not None:
            surface_velocity = (
                surface_velocity
                + NativePanelFieldPlan2D(self.geometry)
                .evaluate(geometry.control, sheet_for_velocity, kind="vortex")
                .velocity
            )
        normal_residual = jnp.sum(surface_velocity * geometry.normal, axis=-1)
        sheet = (
            jnp.zeros((count,), dtype=value.dtype)
            if sheet_for_velocity is None
            else sheet_for_velocity
        )
        component_circulation = circulation_rows @ sheet
        circulation_residual = component_circulation - circulation_target
        kutta_residual = (
            jnp.asarray(0.0, dtype=value.dtype)
            if self.trailing_edge_panels is None
            else sheet[self.trailing_edge_panels[0]] + sheet[self.trailing_edge_panels[1]]
        )
        tangent_speed = jnp.sum(surface_velocity * geometry.tangent, axis=-1)
        reference_speed_squared = jnp.maximum(
            jnp.mean(jnp.sum(incident**2, axis=-1)), jnp.finfo(value.dtype).tiny
        )
        pressure = 1.0 - tangent_speed**2 / reference_speed_squared
        dynamic_pressure = 0.5 * self.density * reference_speed_squared
        panel_force = (
            -dynamic_pressure
            * pressure[:, None]
            * geometry.normal
            * geometry.length[:, None]
        )
        total_force = jnp.sum(panel_force, axis=0)
        reference = jnp.asarray(reference_point, dtype=value.dtype)
        moment = jnp.sum(
            jnp.cross(
                jnp.pad(geometry.control - reference, ((0, 0), (0, 1))),
                jnp.pad(panel_force, ((0, 0), (0, 1))),
            )[:, 2]
        )
        complex_velocity = surface_velocity[:, 0] - 1j * surface_velocity[:, 1]
        blasius_complex = (
            0.5j
            * self.density
            * jnp.sum(
                complex_velocity**2
                * (
                    geometry.end[:, 0]
                    - geometry.start[:, 0]
                    + 1j * (geometry.end[:, 1] - geometry.start[:, 1])
                )
            )
        )
        blasius = jnp.asarray((jnp.real(blasius_complex), -jnp.imag(blasius_complex)))
        impulse_force = None
        if previous_impulse is not None and time_step is not None:
            current_impulse = jnp.sum(
                sheet[:, None]
                * jnp.stack((-geometry.control[:, 1], geometry.control[:, 0]), axis=-1)
                * geometry.length[:, None],
                axis=0,
            )
            impulse_force = (
                -self.density
                * (current_impulse - jnp.asarray(previous_impulse, dtype=value.dtype))
                / jnp.asarray(time_step, dtype=value.dtype)
            )
        agreement = jnp.linalg.norm(total_force - blasius)
        if impulse_force is not None:
            agreement = jnp.maximum(
                agreement, jnp.linalg.norm(total_force - impulse_force)
            )
        load = CompletePanelLoad2D(
            pressure,
            panel_force,
            total_force,
            moment,
            blasius,
            impulse_force,
            agreement,
            jnp.all(jnp.isfinite(panel_force)),
        )
        tolerance = 1.0e-8 * jnp.maximum(jnp.linalg.norm(normal_rhs), 1.0)
        successful = (
            linear.successful
            & load.finite
            & (jnp.linalg.norm(normal_residual) <= tolerance)
            & (jnp.linalg.norm(circulation_residual) <= tolerance)
            & (jnp.abs(kutta_residual) <= tolerance)
        )
        return CompletePanelResult2D(
            source_strength,
            vortex_strength,
            doublet_strength,
            surface_velocity,
            normal_residual,
            circulation_residual,
            kutta_residual,
            load,
            linear,
            successful,
            self.solver_id,
        )


__all__ = ["CompletePanelFlowPlan2D", "CompletePanelLoad2D", "CompletePanelResult2D"]
