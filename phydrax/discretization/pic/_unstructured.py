#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import final, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    ArraySpace,
    ConjugateGradient,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    OperatorProperties,
    prepare,
    PreparedLinearSolve,
    solve,
    TolerancePolicy,
)
from ...typing import AnyDim, Bool, Dim, Float, Int32, Scalar
from .._simplicial_locator import CellLocationResult, PreparedSimplicialCellLocator
from ..particle import ParticlePopulationState
from ._charge_state import PICChargeModelPlan, PICChargeState
from ._method import PIC_CODE_RELATIVITY, RelativisticPushPlan
from ._types import PICParticleState


class ElectrostaticParticleDim(Dim):
    pass


class ElectrostaticVertexDim(Dim):
    pass


class ElectrostaticCellDim(Dim):
    pass


class ElectrostaticAmbientDim(Dim):
    pass


@final
class UnstructuredElectrostaticPICState(StrictModule):
    __strict_contract__ = True

    particles: PICParticleState
    population: ParticlePopulationState
    charge: PICChargeState
    cell_ids: Int32[ElectrostaticParticleDim]
    barycentric: Float[ElectrostaticParticleDim, AnyDim]
    nodal_charge: Float[ElectrostaticVertexDim]
    potential: Float[ElectrostaticVertexDim]
    electric: Float[ElectrostaticParticleDim, Literal[3]]
    time: Float[Scalar]


@final
class UnstructuredElectrostaticPICResult(StrictModule):
    __strict_contract__ = True

    candidate_state: UnstructuredElectrostaticPICState
    accepted_state: UnstructuredElectrostaticPICState
    location: CellLocationResult
    poisson_residual: Float[Scalar]
    charge_balance_defect: Float[Scalar]
    energy: Float[Scalar]
    finite: Bool[Scalar]
    successful: Bool[Scalar]
    plan_id: str = eqx.field(static=True)


@final
class UnstructuredElectrostaticPICPlan(StrictModule, NonTrainableState):
    __strict_contract__ = True

    locator: PreparedSimplicialCellLocator
    charge_model: PICChargeModelPlan
    pusher: RelativisticPushPlan
    gradients: Float[ElectrostaticCellDim, AnyDim, ElectrostaticAmbientDim]
    cell_measures: Float[ElectrostaticCellDim]
    stiffness: FunctionLinearOperator
    dirichlet_mask: Bool[ElectrostaticVertexDim]
    prepared_linear: PreparedLinearSolve
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
        pusher: RelativisticPushPlan | None = None,
    ) -> None:
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
        coordinates = np.asarray(locator.coordinates, dtype=np.float64)
        dimension = locator.dimension
        gradients = locator.affine_gradients()
        vertices = coordinates[cells]
        jacobians = np.swapaxes(vertices[:, 1:] - vertices[:, :1], 1, 2)
        measures = np.abs(np.linalg.det(jacobians)) / math.factorial(dimension)
        vertex_count = coordinates.shape[0]
        cell_measures = jnp.asarray(measures)
        local_stiffness = (
            epsilon
            * cell_measures[:, None, None]
            * jnp.sum(gradients[:, :, None, :] * gradients[:, None, :, :], axis=-1)
        )
        boundary = np.asarray(dirichlet_vertices, dtype=np.bool_)
        if boundary.shape != (vertex_count,) or not np.any(boundary):
            raise ValueError("At least one Dirichlet vertex is required.")
        cell_indices = locator.cells
        boundary_array = jnp.asarray(boundary)

        def stiffness_action(potential: Array) -> Array:
            values = jnp.where(boundary_array, 0.0, potential)[cell_indices]
            local = jnp.sum(local_stiffness * values[:, None, :], axis=-1)
            output = (
                jnp.zeros_like(potential)
                .at[cell_indices.reshape((-1,))]
                .add(local.reshape((-1,)))
            )
            return jnp.where(boundary_array, potential, output)

        space = ArraySpace((vertex_count,), dtype=jnp.float64)
        operator = FunctionLinearOperator(
            stiffness_action,
            source=space,
            target=space,
            transpose_action=stiffness_action,
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
        self.pusher = (
            RelativisticPushPlan(PIC_CODE_RELATIVITY, method="boris")
            if pusher is None
            else pusher
        )
        self.gradients = gradients
        self.cell_measures = cell_measures
        self.stiffness = operator
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
        active = jnp.asarray(active_mask, dtype=jnp.bool_)
        safe_cell = jnp.maximum(location.cell_ids, 0)
        cell_vertices = self.locator.cells[safe_cell]
        valid = active & location.inside
        nodal = jnp.zeros((self.locator.coordinate_count,), dtype=charge.dtype)
        for local in range(self.locator.cells.shape[1]):
            nodal = nodal.at[cell_vertices[:, local]].add(
                jnp.where(valid, charge * location.barycentric[:, local], 0.0)
            )
        return nodal

    def solve_field(
        self, nodal_charge: ArrayLike, initial: ArrayLike | None = None
    ) -> tuple[Array, Array, LinearSolveResult]:
        rhs = jnp.where(self.dirichlet_mask, 0.0, jnp.asarray(nodal_charge))
        guess = jnp.zeros_like(rhs) if initial is None else jnp.asarray(initial)
        result = solve(self.prepared_linear, rhs, initial_guess=guess)
        potential = jnp.where(self.dirichlet_mask, 0.0, result.value)
        residual = self.stiffness.mv(potential) - rhs
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
        energy = 0.5 * potential @ self.stiffness.mv(potential)
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


def jax_tree_where[T](predicate: Array, candidate: T, current: T) -> T:
    import jax

    def select(proposed: Array, old: Array) -> Array:
        return jnp.where(predicate, proposed, old)

    return jax.tree.map(select, candidate, current)


__all__ = [
    "UnstructuredElectrostaticPICPlan",
    "UnstructuredElectrostaticPICResult",
    "UnstructuredElectrostaticPICState",
]
