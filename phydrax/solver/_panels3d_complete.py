#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

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
from ..operators.integral.vortex._panels3d_complete import (
    NativePanelFieldPlan3D,
    NativePanelGeometry3D,
)


class PanelLoadResult3D(StrictModule):
    pressure_coefficient: Array
    panel_force: Array
    total_force: Array
    total_moment: Array
    added_mass: Array | None
    finite: Array


class CompletePanelResult3D(StrictModule):
    source_strength: Array | None
    doublet_strength: Array | None
    surface_velocity: Array
    boundary_residual: Array
    kutta_residual: Array
    load: PanelLoadResult3D
    linear_result: LinearSolveResult
    successful: Array
    solver_id: str = eqx.field(static=True)


class CompletePanelFlowPlan3D(StrictModule, NonTrainableState):
    geometry: NativePanelGeometry3D
    formulation: str = eqx.field(static=True)
    component_ids: Array
    kutta_panel_pairs: Array
    density: float = eqx.field(static=True)
    policy: LinearSolvePolicy
    solver_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: NativePanelGeometry3D,
        formulation: str = "source",
        /,
        *,
        component_ids: ArrayLike | None = None,
        kutta_panel_pairs: ArrayLike | None = None,
        density: float = 1.0,
        policy: LinearSolvePolicy | None = None,
    ):
        if (
            not isinstance(geometry, NativePanelGeometry3D)
            or formulation not in ("source", "doublet", "source-doublet")
            or float(density) <= 0.0
        ):
            raise ValueError("Complete 3-D panel controls are invalid.")
        count = geometry.panel_count
        components = (
            jnp.zeros((count,), dtype=jnp.int32)
            if component_ids is None
            else jnp.asarray(component_ids, dtype=jnp.int32)
        )
        pairs = (
            jnp.empty((0, 2), dtype=jnp.int32)
            if kutta_panel_pairs is None
            else jnp.asarray(kutta_panel_pairs, dtype=jnp.int32)
        )
        if (
            components.shape != (count,)
            or pairs.ndim != 2
            or pairs.shape[1] != 2
            or jnp.any((pairs < 0) | (pairs >= count))
        ):
            raise ValueError("3-D panel component/Kutta metadata are invalid.")
        self.geometry, self.formulation, self.component_ids, self.kutta_panel_pairs = (
            geometry,
            formulation,
            components,
            pairs,
        )
        self.density, self.policy = (
            float(density),
            LinearSolvePolicy(DenseSVD()) if policy is None else policy,
        )
        self.solver_id = canonical_fingerprint(
            {
                "kind": "complete-panel-flow-3d",
                "geometry": geometry.geometry_id,
                "formulation": formulation,
                "components": tuple(int(value) for value in components),
                "kutta_count": int(pairs.shape[0]),
                "density": self.density,
            }
        )

    def solve(
        self,
        incident_velocity: ArrayLike,
        /,
        *,
        body_velocity: ArrayLike | None = None,
        reference_point: ArrayLike = (0.0, 0.0, 0.0),
        potential_rate: ArrayLike | None = None,
        compute_added_mass: bool = False,
    ) -> CompletePanelResult3D:
        incident = jnp.asarray(incident_velocity, dtype=self.geometry.control_point.dtype)
        if incident.shape == (3,):
            incident = jnp.broadcast_to(incident, (self.geometry.panel_count, 3))
        body = (
            jnp.zeros_like(incident)
            if body_velocity is None
            else jnp.asarray(body_velocity, dtype=incident.dtype)
        )
        if (
            incident.shape != (self.geometry.panel_count, 3)
            or body.shape != incident.shape
        ):
            raise ValueError("3-D panel incident/body velocity shapes are invalid.")
        relative = incident - body
        field = NativePanelFieldPlan3D(self.geometry)
        source_velocity, source_potential = field.influence(kind="source")
        doublet_velocity, doublet_potential = field.influence(kind="doublet")
        source_normal = jnp.sum(
            source_velocity * self.geometry.normal[:, None, :], axis=-1
        )
        doublet_normal = jnp.sum(
            doublet_velocity * self.geometry.normal[:, None, :], axis=-1
        )
        normal_rhs = -jnp.sum(relative * self.geometry.normal, axis=-1)
        count = self.geometry.panel_count
        if self.formulation == "source":
            matrix, rhs = source_normal, normal_rhs
            source_slice, doublet_slice = slice(0, count), None
        elif self.formulation == "doublet":
            matrix, rhs = doublet_potential, jnp.zeros((count,), dtype=incident.dtype)
            source_slice, doublet_slice = None, slice(0, count)
        else:
            matrix = jnp.block(
                ((source_normal, doublet_normal), (source_potential, doublet_potential))
            )
            rhs = jnp.concatenate((normal_rhs, jnp.zeros((count,), dtype=incident.dtype)))
            source_slice, doublet_slice = slice(0, count), slice(count, 2 * count)
        offset = count if self.formulation == "source-doublet" else 0
        for pair_index in range(int(self.kutta_panel_pairs.shape[0])):
            upper, lower = self.kutta_panel_pairs[pair_index]
            row = matrix.shape[0] - 1 - pair_index
            kutta = (
                jnp.zeros((matrix.shape[1],), dtype=matrix.dtype)
                .at[offset + upper]
                .set(1.0)
                .at[offset + lower]
                .set(-1.0)
            )
            matrix = matrix.at[row].set(kutta)
            rhs = rhs.at[row].set(0.0)
        linear = solve_linear(
            LeastSquaresProblem(
                DenseLinearOperator(matrix), problem_id=f"{self.solver_id}:surface"
            ),
            rhs,
            policy=self.policy,
        )
        value = jnp.asarray(linear.value)
        source_strength = None if source_slice is None else value[source_slice]
        doublet_strength = None if doublet_slice is None else value[doublet_slice]
        surface_velocity = relative
        if source_strength is not None:
            surface_velocity = surface_velocity + contract(
                "tjc,j->tc",
                source_velocity,
                source_strength,
            )
        if doublet_strength is not None:
            surface_velocity = surface_velocity + contract(
                "tjc,j->tc",
                doublet_velocity,
                doublet_strength,
            )
        boundary_residual = jnp.sum(surface_velocity * self.geometry.normal, axis=-1)
        kutta_residual = jnp.zeros((self.kutta_panel_pairs.shape[0],), dtype=value.dtype)
        if doublet_strength is not None and self.kutta_panel_pairs.size:
            kutta_residual = (
                doublet_strength[self.kutta_panel_pairs[:, 0]]
                - doublet_strength[self.kutta_panel_pairs[:, 1]]
            )
        speed_squared = jnp.sum(surface_velocity**2, axis=-1)
        reference_speed_squared = jnp.maximum(
            jnp.mean(jnp.sum(incident**2, axis=-1)), jnp.finfo(value.dtype).tiny
        )
        rate = (
            jnp.zeros((count,), dtype=value.dtype)
            if potential_rate is None
            else jnp.asarray(potential_rate, dtype=value.dtype)
        )
        pressure = (
            1.0
            - speed_squared / reference_speed_squared
            - 2.0 * rate / reference_speed_squared
        )
        panel_force = (
            -0.5
            * self.density
            * reference_speed_squared
            * pressure[:, None]
            * self.geometry.normal
            * self.geometry.area[:, None]
        )
        reference = jnp.asarray(reference_point, dtype=value.dtype)
        total_force = jnp.sum(panel_force, axis=0)
        total_moment = jnp.sum(
            jnp.cross(self.geometry.control_point - reference, panel_force), axis=0
        )
        added_mass = self.added_mass_matrix(source_normal) if compute_added_mass else None
        load = PanelLoadResult3D(
            pressure,
            panel_force,
            total_force,
            total_moment,
            added_mass,
            jnp.all(jnp.isfinite(panel_force)),
        )
        tolerance = 1.0e-7 * jnp.maximum(jnp.linalg.norm(normal_rhs), 1.0)
        successful = (
            linear.successful
            & load.finite
            & (jnp.linalg.norm(boundary_residual) <= tolerance)
            & (jnp.linalg.norm(kutta_residual) <= tolerance)
        )
        return CompletePanelResult3D(
            source_strength,
            doublet_strength,
            surface_velocity,
            boundary_residual,
            kutta_residual,
            load,
            linear,
            successful,
            self.solver_id,
        )

    def added_mass_matrix(self, source_normal: Array | None = None, /) -> Array:
        field = NativePanelFieldPlan3D(self.geometry)
        source_velocity, source_potential = field.influence(kind="source")
        normal_matrix = (
            jnp.sum(source_velocity * self.geometry.normal[:, None, :], axis=-1)
            if source_normal is None
            else source_normal
        )
        modes = []
        reference = jnp.mean(self.geometry.control_point, axis=0)
        for axis in range(3):
            modes.append(
                jnp.broadcast_to(jnp.eye(3)[axis], self.geometry.control_point.shape)
            )
        for axis in range(3):
            modes.append(
                jnp.cross(
                    jnp.broadcast_to(jnp.eye(3)[axis], self.geometry.control_point.shape),
                    self.geometry.control_point - reference,
                )
            )
        potentials = []
        for mode in modes:
            rhs = jnp.sum(mode * self.geometry.normal, axis=-1)
            linear = solve_linear(
                LeastSquaresProblem(
                    DenseLinearOperator(normal_matrix),
                    problem_id=f"{self.solver_id}:added-mass",
                ),
                rhs,
                policy=self.policy,
            )
            potentials.append(source_potential @ jnp.asarray(linear.value))
        matrix = jnp.stack(tuple(potentials), axis=0)
        return self.density * (
            matrix
            @ (
                self.geometry.area[:, None]
                * jnp.stack(
                    tuple(
                        jnp.sum(mode * self.geometry.normal, axis=-1) for mode in modes
                    ),
                    axis=-1,
                )
            )
        )


__all__ = ["CompletePanelFlowPlan3D", "CompletePanelResult3D", "PanelLoadResult3D"]
