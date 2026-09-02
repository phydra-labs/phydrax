#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._cell_mesh import CellMesh
from ...discretization.contact import (
    assemble_contact_interface_traction,
    ContactInterfacePlan,
    ContactInterfaceResidual,
)


class LubricationContactPlan(StrictModule, NonTrainableState):
    viscosity: float = eqx.field(static=True)
    minimum_film_thickness: float = eqx.field(static=True)
    cavitation_pressure: float = eqx.field(static=True)
    asperity_transition: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        viscosity: float,
        minimum_film_thickness: float,
        cavitation_pressure: float = 0.0,
        asperity_transition: float,
    ):
        viscosity_ = float(viscosity)
        minimum = float(minimum_film_thickness)
        cavitation = float(cavitation_pressure)
        transition = float(asperity_transition)
        if (
            not np.isfinite(viscosity_)
            or viscosity_ <= 0.0
            or not np.isfinite(minimum)
            or minimum <= 0.0
            or not np.isfinite(cavitation)
            or not np.isfinite(transition)
            or transition <= minimum
        ):
            raise ValueError("Lubrication contact parameters are invalid.")
        self.viscosity = viscosity_
        self.minimum_film_thickness = minimum
        self.cavitation_pressure = cavitation
        self.asperity_transition = transition
        self.plan_id = canonical_fingerprint(
            {
                "kind": "lubrication-contact-plan",
                "viscosity": viscosity_.hex(),
                "minimum_film_thickness": minimum.hex(),
                "cavitation_pressure": cavitation.hex(),
                "asperity_transition": transition.hex(),
            }
        )


class LubricationContactResponse(StrictModule):
    film_thickness: Array
    fluid_pressure: Array
    asperity_fraction: Array
    normal_traction: Array
    tangential_traction: Array
    dissipated_power: Array
    cavitated: Array
    finite: Array
    dissipative: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def evaluate_lubrication_contact(
    plan: LubricationContactPlan,
    gap: ArrayLike,
    normal_velocity: ArrayLike,
    tangential_velocity: ArrayLike,
    effective_radius: ArrayLike,
    /,
    *,
    asperity_pressure: ArrayLike = 0.0,
) -> LubricationContactResponse:
    if not isinstance(plan, LubricationContactPlan):
        raise TypeError("plan must be LubricationContactPlan.")
    gap_ = jnp.asarray(gap)
    normal_velocity_ = jnp.asarray(normal_velocity, dtype=gap_.dtype)
    tangential = jnp.asarray(tangential_velocity, dtype=gap_.dtype)
    radius = jnp.asarray(effective_radius, dtype=gap_.dtype)
    asperity = jnp.asarray(asperity_pressure, dtype=gap_.dtype)
    if normal_velocity_.shape != gap_.shape or tangential.shape[:-1] != gap_.shape:
        raise ValueError("Lubrication contact kinematic shapes are invalid.")
    film = jnp.maximum(gap_, plan.minimum_film_thickness)
    closing = jnp.maximum(-normal_velocity_, 0.0)
    squeeze_pressure = 3.0 * plan.viscosity * radius * radius * closing / (2.0 * film**3)
    fluid_pressure = jnp.maximum(plan.cavitation_pressure, squeeze_pressure)
    cavitated = squeeze_pressure < plan.cavitation_pressure
    asperity_fraction = jnp.clip(
        (plan.asperity_transition - film)
        / (plan.asperity_transition - plan.minimum_film_thickness),
        0.0,
        1.0,
    )
    normal_traction = (
        1.0 - asperity_fraction
    ) * fluid_pressure + asperity_fraction * jnp.maximum(asperity, 0.0)
    fluid_shear = -(plan.viscosity / film)[..., None] * tangential
    tangential_traction = (1.0 - asperity_fraction)[..., None] * fluid_shear
    dissipated = (
        -jnp.sum(tangential_traction * tangential, axis=-1) + fluid_pressure * closing
    )
    finite = (
        jnp.all(jnp.isfinite(film))
        & jnp.all(jnp.isfinite(fluid_pressure))
        & jnp.all(jnp.isfinite(normal_traction))
        & jnp.all(jnp.isfinite(tangential_traction))
        & jnp.all(jnp.isfinite(dissipated))
    )
    dissipative = jnp.all(dissipated >= -64.0 * jnp.finfo(gap_.dtype).eps)
    return LubricationContactResponse(
        film,
        fluid_pressure,
        asperity_fraction,
        normal_traction,
        tangential_traction,
        jnp.maximum(dissipated, 0.0),
        cavitated,
        finite,
        dissipative,
        finite & dissipative,
        plan.plan_id,
    )


class ReynoldsPressureBoundaryConditions(StrictModule, NonTrainableState):
    node_indices: Array
    pressure: Array
    boundary_id: str = eqx.field(static=True)

    def __init__(self, node_indices: ArrayLike, pressure: ArrayLike, /):
        indices = np.asarray(node_indices)
        values = np.asarray(pressure, dtype=float)
        if (
            indices.ndim != 1
            or not np.issubdtype(indices.dtype, np.integer)
            or values.shape != indices.shape
            or np.any(indices < 0)
            or np.unique(indices).size != indices.size
        ):
            raise ValueError("Reynolds pressure boundary nodes are invalid.")
        if np.any(~np.isfinite(values)):
            raise ValueError("Reynolds pressure boundary values must be finite.")
        self.node_indices = jnp.asarray(indices, dtype=jnp.int32)
        self.pressure = jnp.asarray(values)
        self.boundary_id = canonical_fingerprint(
            {
                "kind": "reynolds-pressure-boundary",
                "indices": indices.tolist(),
                "pressure": values.tolist(),
            }
        )


class ReynoldsFilmState(StrictModule):
    pressure: Array
    active_set: Array
    accepted_steps: Array


class ReynoldsFilmEvidence(StrictModule):
    pde_residual: Array
    complementarity_residual: Array
    flux_balance: Array
    load: Array
    dissipation: Array
    minimum_film: Array
    active_set_margin: Array
    derivative_valid: Array
    solver_converged: Array
    finite: Array
    successful: Array


class ReynoldsFilmResult(StrictModule):
    pressure: Array
    interface_pressure: Array
    traction: Array
    interface_residual: ContactInterfaceResidual
    candidate_state: ReynoldsFilmState
    accepted_state: ReynoldsFilmState
    evidence: ReynoldsFilmEvidence
    successful: Array


class ReynoldsFilmPlan(StrictModule, NonTrainableState):
    """Connected-patch Reynolds variational inequality on a P1 surface mesh."""

    film_mesh: CellMesh
    interface: ContactInterfacePlan
    boundary_conditions: ReynoldsPressureBoundaryConditions
    viscosity: float = eqx.field(static=True)
    cavitation_pressure: float = eqx.field(static=True)
    minimum_film_thickness: float = eqx.field(static=True)
    active_set_iterations: int = eqx.field(static=True)
    convergence_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        film_mesh: CellMesh,
        interface: ContactInterfacePlan,
        *,
        viscosity: float,
        boundary_conditions: ReynoldsPressureBoundaryConditions,
        cavitation_pressure: float = 0.0,
        minimum_film_thickness: float = 1.0e-9,
        active_set_iterations: int = 32,
        convergence_tolerance: float = 1.0e-8,
    ):
        if not isinstance(film_mesh, CellMesh):
            raise TypeError("film_mesh must be CellMesh.")
        if not isinstance(interface, ContactInterfacePlan):
            raise TypeError("interface must be ContactInterfacePlan.")
        if not isinstance(boundary_conditions, ReynoldsPressureBoundaryConditions):
            raise TypeError(
                "boundary_conditions must be ReynoldsPressureBoundaryConditions."
            )
        if film_mesh.topological_dimension != 2 or any(
            block.cell_kind != "triangle" for block in film_mesh.blocks
        ):
            raise ValueError("Reynolds film requires an affine triangular surface mesh.")
        if interface.plus_node_count != film_mesh.coordinates.shape[0]:
            raise ValueError(
                "Reynolds film/interface binding requires film nodes on the plus trace."
            )
        if np.any(
            np.asarray(boundary_conditions.node_indices) >= film_mesh.coordinates.shape[0]
        ):
            raise ValueError("Reynolds pressure boundary node is out of range.")
        viscosity_ = float(viscosity)
        cavitation = float(cavitation_pressure)
        minimum = float(minimum_film_thickness)
        iterations = int(active_set_iterations)
        tolerance = float(convergence_tolerance)
        if (
            not np.isfinite(viscosity_)
            or viscosity_ <= 0.0
            or not np.isfinite(cavitation)
            or not np.isfinite(minimum)
            or minimum <= 0.0
            or iterations <= 0
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError("Reynolds material/solver parameters are invalid.")
        if np.any(np.asarray(boundary_conditions.pressure) < cavitation):
            raise ValueError(
                "Pressure boundaries must not lie below cavitation pressure."
            )
        _require_reynolds_component_references(
            film_mesh, np.asarray(boundary_conditions.node_indices)
        )
        self.film_mesh = film_mesh
        self.interface = interface
        self.boundary_conditions = boundary_conditions
        self.viscosity = viscosity_
        self.cavitation_pressure = cavitation
        self.minimum_film_thickness = minimum
        self.active_set_iterations = iterations
        self.convergence_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "reynolds-film-plan",
                "mesh": film_mesh.mesh_id,
                "interface": interface.interface_id,
                "viscosity": viscosity_.hex(),
                "boundary": boundary_conditions.boundary_id,
                "cavitation_pressure": cavitation.hex(),
                "minimum_film_thickness": minimum.hex(),
                "active_set_iterations": iterations,
                "convergence_tolerance": tolerance.hex(),
            }
        )

    def prepare(self, /) -> PreparedReynoldsFilm:
        coordinates = np.asarray(self.film_mesh.coordinates)
        cells = np.concatenate(
            tuple(
                np.asarray(block.vertices, dtype=np.int32)
                for block in self.film_mesh.blocks
            ),
            axis=0,
        )
        gradients = np.zeros((cells.shape[0], 3, coordinates.shape[1]), dtype=float)
        areas = np.zeros((cells.shape[0],), dtype=float)
        for cell_index, cell in enumerate(cells):
            vertices = coordinates[cell]
            first = vertices[1] - vertices[0]
            second = vertices[2] - vertices[0]
            gram = np.asarray(
                (
                    (np.dot(first, first), np.dot(first, second)),
                    (np.dot(second, first), np.dot(second, second)),
                )
            )
            determinant = float(np.linalg.det(gram))
            if not np.isfinite(determinant) or determinant <= 0.0:
                raise ValueError("Reynolds film contains a degenerate triangle.")
            inverse = np.linalg.inv(gram)
            reduced_gradients = np.stack((first, second), axis=0)
            physical = inverse @ reduced_gradients
            gradients[cell_index, 1:] = physical
            gradients[cell_index, 0] = -np.sum(physical, axis=0)
            areas[cell_index] = 0.5 * np.sqrt(determinant)
        return PreparedReynoldsFilm(
            self,
            jnp.asarray(cells),
            jnp.asarray(gradients),
            jnp.asarray(areas),
            canonical_fingerprint(
                {
                    "kind": "prepared-reynolds-film",
                    "plan": self.plan_id,
                    "cells": int(cells.shape[0]),
                }
            ),
        )


class PreparedReynoldsFilm(StrictModule, NonTrainableState):
    plan: ReynoldsFilmPlan
    cells: Array
    basis_gradients: Array
    cell_area: Array
    prepared_id: str = eqx.field(static=True)

    def initialize(self, dtype=float, /) -> ReynoldsFilmState:
        pressure = jnp.full(
            (self.plan.film_mesh.coordinates.shape[0],),
            self.plan.cavitation_pressure,
            dtype=dtype,
        )
        pressure = pressure.at[self.plan.boundary_conditions.node_indices].set(
            self.plan.boundary_conditions.pressure.astype(pressure.dtype)
        )
        return ReynoldsFilmState(
            pressure,
            pressure <= self.plan.cavitation_pressure,
            jnp.asarray(0, dtype=jnp.int32),
        )

    def evaluate(
        self,
        state: ReynoldsFilmState,
        film_thickness: ArrayLike,
        squeeze_rate: ArrayLike,
        tangential_velocity: ArrayLike,
        /,
    ) -> ReynoldsFilmResult:
        if not isinstance(state, ReynoldsFilmState):
            raise TypeError("state must be ReynoldsFilmState.")
        thickness = jnp.asarray(film_thickness)
        squeeze = jnp.asarray(squeeze_rate, dtype=thickness.dtype)
        velocity = jnp.asarray(tangential_velocity, dtype=thickness.dtype)
        node_count = int(self.plan.film_mesh.coordinates.shape[0])
        dimension = self.plan.film_mesh.ambient_dimension
        if (
            thickness.shape != (node_count,)
            or squeeze.shape != (node_count,)
            or velocity.shape != (node_count, dimension)
            or state.pressure.shape != (node_count,)
            or state.active_set.shape != (node_count,)
        ):
            raise ValueError("Reynolds film runtime arrays have incompatible shapes.")
        local_thickness = thickness[self.cells]
        cell_thickness = jnp.mean(local_thickness, axis=1)
        coefficient = cell_thickness**3 / (12.0 * self.plan.viscosity)
        local_stiffness = (
            coefficient[:, None, None]
            * self.cell_area[:, None, None]
            * oe.contract(
                "cid,cjd->cij",
                self.basis_gradients,
                self.basis_gradients,
                backend="jax",
            )
        )
        matrix = jnp.zeros((node_count, node_count), dtype=thickness.dtype)
        matrix = matrix.at[self.cells[:, :, None], self.cells[:, None, :]].add(
            local_stiffness
        )
        local_squeeze = -self.cell_area[:, None] * squeeze[self.cells] / 3.0
        cell_velocity = jnp.mean(
            thickness[self.cells][..., None] * velocity[self.cells], axis=1
        )
        local_advection = -self.cell_area[:, None] * oe.contract(
            "cid,cd->ci", self.basis_gradients, cell_velocity, backend="jax"
        )
        right_hand_side = jnp.zeros((node_count,), dtype=thickness.dtype)
        right_hand_side = right_hand_side.at[self.cells].add(
            local_squeeze + local_advection
        )
        boundary_mask = (
            jnp.zeros((node_count,), dtype=bool)
            .at[self.plan.boundary_conditions.node_indices]
            .set(True)
        )
        boundary_pressure = (
            jnp.full((node_count,), self.plan.cavitation_pressure, dtype=thickness.dtype)
            .at[self.plan.boundary_conditions.node_indices]
            .set(self.plan.boundary_conditions.pressure.astype(thickness.dtype))
        )
        diagonal = jnp.diag(matrix)
        safe_diagonal = jnp.where(diagonal > 0.0, diagonal, 1.0)
        pressure = jnp.maximum(
            state.pressure.astype(thickness.dtype), self.plan.cavitation_pressure
        )
        pressure = jnp.where(boundary_mask, boundary_pressure, pressure)
        for _ in range(self.plan.active_set_iterations):
            residual = matrix @ pressure - right_hand_side
            candidate = jnp.maximum(
                self.plan.cavitation_pressure,
                pressure - residual / safe_diagonal,
            )
            pressure = jnp.where(boundary_mask, boundary_pressure, candidate)
        residual = matrix @ pressure - right_hand_side
        active_set = (
            pressure - self.plan.cavitation_pressure <= self.plan.convergence_tolerance
        ) & ~boundary_mask
        free_residual = jnp.where(~active_set & ~boundary_mask, residual, 0.0)
        active_violation = jnp.where(active_set, jnp.minimum(residual, 0.0), 0.0)
        pde_residual = jnp.sqrt(jnp.sum(free_residual * free_residual))
        complementarity = jnp.maximum(
            jnp.max(jnp.abs(active_violation), initial=0.0),
            jnp.max(
                jnp.abs(
                    (pressure - self.plan.cavitation_pressure)
                    * jnp.maximum(residual, 0.0)
                ),
                initial=0.0,
            ),
        )
        flux_balance = jnp.sum(residual)
        safe_plus = jnp.clip(self.plan.interface.plus_indices, 0, node_count - 1)
        interface_pressure = jnp.sum(
            self.plan.interface.plus_weights.astype(pressure.dtype) * pressure[safe_plus],
            axis=-1,
        )
        interface_pressure = jnp.where(self.plan.interface.valid, interface_pressure, 0.0)
        traction = -interface_pressure[
            :, None
        ] * self.plan.interface.reference_normal.astype(pressure.dtype)
        interface_residual = assemble_contact_interface_traction(
            self.plan.interface, traction
        )
        load = jnp.sum(
            jnp.where(
                self.plan.interface.valid,
                interface_pressure
                * self.plan.interface.quadrature_weight.astype(pressure.dtype),
                0.0,
            )
        )
        dissipation = jnp.sum(
            self.cell_area
            * self.plan.viscosity
            * jnp.sum(jnp.mean(velocity[self.cells], axis=1) ** 2, axis=-1)
            / jnp.maximum(cell_thickness, self.plan.minimum_film_thickness)
        )
        active_margin = jnp.min(
            jnp.where(
                boundary_mask,
                jnp.inf,
                jnp.where(
                    active_set,
                    residual,
                    pressure - self.plan.cavitation_pressure,
                ),
            ),
            initial=jnp.inf,
        )
        finite = (
            jnp.all(jnp.isfinite(thickness))
            & jnp.all(jnp.isfinite(squeeze))
            & jnp.all(jnp.isfinite(velocity))
            & jnp.all(jnp.isfinite(pressure))
            & jnp.all(thickness >= self.plan.minimum_film_thickness)
            & interface_residual.finite
        )
        solver_converged = (pde_residual <= self.plan.convergence_tolerance) & (
            complementarity <= self.plan.convergence_tolerance
        )
        derivative_valid = solver_converged & (
            active_margin > self.plan.convergence_tolerance
        )
        successful = (
            finite
            & solver_converged
            & jnp.isfinite(flux_balance)
            & jnp.isfinite(load)
            & jnp.isfinite(dissipation)
            & (dissipation >= 0.0)
        )
        candidate = ReynoldsFilmState(pressure, active_set, state.accepted_steps + 1)
        accepted = ReynoldsFilmState(
            jnp.where(successful, candidate.pressure, state.pressure),
            jnp.where(successful, candidate.active_set, state.active_set),
            jnp.where(successful, candidate.accepted_steps, state.accepted_steps),
        )
        evidence = ReynoldsFilmEvidence(
            pde_residual,
            complementarity,
            flux_balance,
            load,
            dissipation,
            jnp.min(thickness),
            active_margin,
            derivative_valid,
            solver_converged,
            finite,
            successful,
        )
        return ReynoldsFilmResult(
            pressure,
            interface_pressure,
            traction,
            interface_residual,
            candidate,
            accepted,
            evidence,
            successful,
        )


def _require_reynolds_component_references(
    mesh: CellMesh, boundary_nodes: np.ndarray, /
) -> None:
    cells = np.concatenate(
        tuple(np.asarray(block.vertices, dtype=np.int32) for block in mesh.blocks),
        axis=0,
    )
    node_count = int(mesh.coordinates.shape[0])
    adjacency = [set() for _ in range(node_count)]
    for cell in cells:
        for node in cell:
            adjacency[int(node)].update(int(other) for other in cell if other != node)
    unvisited = set(range(node_count))
    boundary = set(int(value) for value in boundary_nodes)
    while unvisited:
        root = min(unvisited)
        stack = [root]
        component = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node] - component)
        unvisited -= component
        if not component.intersection(boundary):
            raise ValueError(
                "Every Reynolds film component requires a pressure reference."
            )


__all__ = [
    "LubricationContactPlan",
    "LubricationContactResponse",
    "PreparedReynoldsFilm",
    "ReynoldsFilmEvidence",
    "ReynoldsFilmPlan",
    "ReynoldsFilmResult",
    "ReynoldsFilmState",
    "ReynoldsPressureBoundaryConditions",
    "evaluate_lubrication_contact",
]
