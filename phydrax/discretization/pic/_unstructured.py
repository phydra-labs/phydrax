#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    ConjugateGradient,
    DenseLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    prepare,
    solve,
    TolerancePolicy,
)
from .._simplicial_locator import CellLocationResult, PreparedSimplicialCellLocator
from ..particle import ParticlePopulationState
from ._charge_state import PICChargeModelPlan, PICChargeState
from ._method import RelativisticBorisPlan
from ._types import PICParticleState


class UnstructuredElectrostaticPICState(StrictModule):
    particles: PICParticleState
    population: ParticlePopulationState
    charge: PICChargeState
    cell_ids: Array
    barycentric: Array
    nodal_charge: Array
    potential: Array
    electric: Array
    time: Array


class UnstructuredElectrostaticPICResult(StrictModule):
    candidate_state: UnstructuredElectrostaticPICState
    accepted_state: UnstructuredElectrostaticPICState
    location: CellLocationResult
    poisson_residual: Array
    charge_balance_defect: Array
    energy: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class UnstructuredElectrostaticPICPlan(StrictModule, NonTrainableState):
    locator: PreparedSimplicialCellLocator
    charge_model: PICChargeModelPlan
    pusher: RelativisticBorisPlan
    gradients: Array
    cell_measures: Array
    stiffness: Array
    dirichlet_mask: Array
    prepared_linear: object
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        locator: PreparedSimplicialCellLocator,
        charge_model: PICChargeModelPlan,
        dirichlet_vertices: ArrayLike,
        /,
        *,
        permittivity: float = 1.0,
        tolerance: float = 1.0e-10,
        maximum_iterations: int = 500,
        pusher: RelativisticBorisPlan | None = None,
    ):
        if not isinstance(locator, PreparedSimplicialCellLocator):
            raise TypeError("locator must be PreparedSimplicialCellLocator.")
        if locator.cell_map.coordinate_element.degree != 1:
            raise ValueError("Whitney electrostatic PIC requires an order-one cell map.")
        if not isinstance(charge_model, PICChargeModelPlan):
            raise TypeError("charge_model must be PICChargeModelPlan.")
        epsilon = float(permittivity)
        if epsilon <= 0.0 or not np.isfinite(epsilon):
            raise ValueError("permittivity must be positive and finite.")
        cells = np.asarray(locator.cells, dtype=np.int32)
        coordinates = np.asarray(locator.coordinates, dtype=float)
        dimension = locator.dimension
        gradients = []
        measures = []
        vertex_count = coordinates.shape[0]
        stiffness = np.zeros((vertex_count, vertex_count), dtype=float)
        for cell in cells:
            vertices = coordinates[cell]
            jacobian = (vertices[1:] - vertices[0]).T
            inverse = np.linalg.solve(
                jacobian,
                np.eye(dimension, dtype=jacobian.dtype),
            )
            local_gradients = np.concatenate(
                (-np.sum(inverse, axis=0, keepdims=True), inverse), axis=0
            )
            measure = abs(np.linalg.det(jacobian)) / math.factorial(dimension)
            local = epsilon * measure * (local_gradients @ local_gradients.T)
            stiffness[np.ix_(cell, cell)] += local
            gradients.append(local_gradients)
            measures.append(measure)
        boundary = np.asarray(dirichlet_vertices, dtype=bool)
        if boundary.shape != (vertex_count,) or not np.any(boundary):
            raise ValueError("At least one Dirichlet vertex is required.")
        interior = ~boundary
        modified = interior[:, None] * stiffness * interior[None, :] + np.diag(
            boundary.astype(float)
        )
        operator = DenseLinearOperator(
            jnp.asarray(modified),
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                },
            ),
            operator_id=canonical_fingerprint(
                {
                    "kind": "unstructured-pic-poisson",
                    "topology": locator.cell_map.topology_id,
                }
            ),
        )
        policy = LinearSolvePolicy(
            ConjugateGradient(),
            tolerance=TolerancePolicy(
                relative=float(tolerance),
                absolute=float(tolerance),
                max_steps=int(maximum_iterations),
            ),
        )
        prepared = prepare(LinearSystem(operator), policy)
        self.locator = locator
        self.charge_model = charge_model
        self.pusher = RelativisticBorisPlan() if pusher is None else pusher
        self.gradients = jnp.asarray(gradients)
        self.cell_measures = jnp.asarray(measures)
        self.stiffness = jnp.asarray(modified)
        self.dirichlet_mask = jnp.asarray(boundary)
        self.prepared_linear = prepared
        self.tolerance = float(tolerance)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "unstructured-electrostatic-pic",
                "locator": locator.locator_id,
                "charge_model": charge_model.plan_id,
                "permittivity": epsilon,
                "linear": prepared.plan.plan_id,
            }
        )

    def deposit(
        self,
        location: CellLocationResult,
        macrocharge: ArrayLike,
        active_mask: ArrayLike,
        /,
    ) -> Array:
        charge = jnp.asarray(macrocharge)
        active = jnp.asarray(active_mask, dtype=bool)
        safe_cell = jnp.maximum(location.cell_ids, 0)
        cell_vertices = self.locator.cells[safe_cell]
        valid = active & location.inside
        nodal = jnp.zeros((self.locator.coordinate_count,), dtype=charge.dtype)
        for local in range(self.locator.cells.shape[1]):
            nodal = nodal.at[cell_vertices[:, local]].add(
                jnp.where(valid, charge * location.barycentric[:, local], 0.0)
            )
        return nodal

    def solve_field(self, nodal_charge: ArrayLike, initial=None):
        rhs = jnp.where(self.dirichlet_mask, 0.0, jnp.asarray(nodal_charge))
        guess = jnp.zeros_like(rhs) if initial is None else jnp.asarray(initial)
        result = solve(self.prepared_linear, rhs, initial_guess=guess)
        potential = jnp.where(self.dirichlet_mask, 0.0, result.value)
        residual = self.stiffness @ potential - rhs
        return potential, residual, result

    def gather_electric(
        self, location: CellLocationResult, potential: ArrayLike, /
    ) -> Array:
        phi = jnp.asarray(potential)
        safe_cell = jnp.maximum(location.cell_ids, 0)
        vertices = self.locator.cells[safe_cell]
        local_phi = phi[vertices]
        gradient = jnp.sum(local_phi[:, :, None] * self.gradients[safe_cell], axis=1)
        value = -gradient
        if self.locator.dimension < 3:
            value = jnp.pad(value, ((0, 0), (0, 3 - self.locator.dimension)))
        return jnp.where(location.inside[:, None], value, 0.0)

    def initialize(
        self,
        particles: PICParticleState,
        population: ParticlePopulationState,
        charge: PICChargeState,
        /,
    ) -> UnstructuredElectrostaticPICState:
        location = self.locator.locate(particles.position)
        macrocharge = self.charge_model.macrocharge(population, charge)
        nodal = self.deposit(location, macrocharge, population.active)
        potential, _, field = self.solve_field(nodal)
        electric = self.gather_electric(location, potential)
        state = UnstructuredElectrostaticPICState(
            particles,
            population,
            charge,
            location.cell_ids,
            location.barycentric,
            nodal,
            potential,
            electric,
            jnp.asarray(0.0, dtype=particles.position.dtype),
        )
        return eqx.error_if(
            state,
            ~location.successful.all() | ~field.successful,
            "Unstructured PIC initialization failed.",
        )

    def step(
        self,
        state: UnstructuredElectrostaticPICState,
        step_size: ArrayLike,
        /,
    ) -> UnstructuredElectrostaticPICResult:
        dt = jnp.asarray(step_size, dtype=state.time.dtype).reshape(())
        specific = (
            self.charge_model.base_specific_charge
            * state.charge.charge_number.astype(state.population.mass.dtype)
        )
        half = self.pusher.push(
            state.particles.proper_velocity,
            state.electric,
            jnp.zeros_like(state.electric),
            specific,
            state.population.active,
            0.5 * dt,
        )
        position = (
            state.particles.position + dt * half.velocity[:, : self.locator.dimension]
        )
        location = self.locator.locate(position)
        macrocharge = self.charge_model.macrocharge(state.population, state.charge)
        nodal = self.deposit(location, macrocharge, state.population.active)
        potential, residual, linear = self.solve_field(nodal, state.potential)
        electric = self.gather_electric(location, potential)
        final = self.pusher.push(
            half.proper_velocity,
            electric,
            jnp.zeros_like(electric),
            specific,
            state.population.active,
            0.5 * dt,
        )
        candidate = UnstructuredElectrostaticPICState(
            PICParticleState(position, final.proper_velocity),
            state.population,
            state.charge,
            location.cell_ids,
            location.barycentric,
            nodal,
            potential,
            electric,
            state.time + dt,
        )
        balance = jnp.abs(jnp.sum(nodal) - jnp.sum(macrocharge))
        residual_norm = jnp.sqrt(jnp.sum(residual**2))
        energy = 0.5 * potential @ (self.stiffness @ potential)
        finite = jnp.all(jnp.isfinite(potential)) & jnp.all(
            jnp.isfinite(final.proper_velocity)
        )
        successful = (
            location.successful.all() & linear.successful & final.successful & finite
        )
        accepted = jax_tree_where(successful, candidate, state)
        return UnstructuredElectrostaticPICResult(
            candidate,
            accepted,
            location,
            residual_norm,
            balance,
            energy,
            finite,
            successful,
            self.plan_id,
        )


def jax_tree_where(predicate, candidate, current):
    import jax

    return jax.tree.map(
        lambda proposed, old: jnp.where(predicate, proposed, old), candidate, current
    )


__all__ = [
    "UnstructuredElectrostaticPICPlan",
    "UnstructuredElectrostaticPICResult",
    "UnstructuredElectrostaticPICState",
]
