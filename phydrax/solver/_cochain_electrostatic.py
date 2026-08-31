#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import StructuredCochainBridge
from ..linalg import (
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


ElectrostaticBoundaryKind: TypeAlias = Literal["periodic", "dirichlet"]


class _CochainPoissonAction(StrictModule, NonTrainableState):
    bridge: StructuredCochainBridge
    permittivity: Array
    boundary: ElectrostaticBoundaryKind = eqx.field(static=True)
    active: Array

    def __call__(self, potential: Array, /) -> Array:
        cochain = self.bridge.cochain
        weights = cochain.hodge_stars[0].astype(potential.dtype)
        if self.boundary == "periodic":
            mean = jnp.sum(weights * potential) / jnp.sum(weights)
            value = potential - mean
            electric = cochain.exterior_derivative(0, value)
            core = -cochain.codifferential(1, self.permittivity * electric)
            return core + mean
        value = jnp.where(self.active, potential, 0.0)
        electric = cochain.exterior_derivative(0, value, boundary_policy="relative")
        core = -cochain.codifferential(
            1, self.permittivity * electric, boundary_policy="relative"
        )
        return jnp.where(self.active, core, potential)


class CochainElectrostaticResult(StrictModule):
    charge: Array
    potential: Array
    electric: Array
    physical_electric: tuple[Array, ...]
    poisson_residual: Array
    residual_norm: Array
    compatibility_defect: Array
    gauge_defect: Array
    field_energy: Array
    linear: LinearSolveResult
    finite: Array
    converged: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class CochainElectrostaticPlan(StrictModule, NonTrainableState):
    """Matrix-free compatible electrostatic solve on degree-zero cochains."""

    bridge: StructuredCochainBridge
    permittivity: Array
    boundary: ElectrostaticBoundaryKind = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    compatibility_tolerance: float = eqx.field(static=True)
    active: Array
    linear_policy: LinearSolvePolicy
    linear_problem: LinearSystem
    prepared_linear: PreparedLinearSolve
    operator_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        bridge: StructuredCochainBridge,
        /,
        *,
        permittivity: ArrayLike = 1.0,
        boundary: ElectrostaticBoundaryKind = "periodic",
        tolerance: float = 1.0e-10,
        compatibility_tolerance: float = 1.0e-10,
        maximum_iterations: int = 500,
        linear_policy: LinearSolvePolicy | None = None,
    ):
        if not isinstance(bridge, StructuredCochainBridge):
            raise TypeError("bridge must be StructuredCochainBridge.")
        if boundary not in ("periodic", "dirichlet"):
            raise ValueError("boundary must be 'periodic' or 'dirichlet'.")
        if boundary == "periodic" and any(
            not axis.periodic for axis in bridge.grid.structured_axes
        ):
            raise ValueError("Periodic electrostatics requires every grid axis periodic.")
        if boundary == "dirichlet" and any(
            axis.periodic for axis in bridge.grid.structured_axes
        ):
            raise ValueError("Dirichlet electrostatics requires bounded grid axes.")
        tolerance_ = float(tolerance)
        compatibility_ = float(compatibility_tolerance)
        iterations = int(maximum_iterations)
        if (
            not np.isfinite(tolerance_)
            or tolerance_ <= 0.0
            or not np.isfinite(compatibility_)
            or compatibility_ < 0.0
            or iterations <= 0
        ):
            raise ValueError("Electrostatic tolerances and iterations are invalid.")
        n1 = bridge.cochain.cell_counts[1]
        epsilon = jnp.asarray(permittivity)
        if epsilon.shape not in ((), (1,), (n1,)):
            raise ValueError(f"permittivity must be scalar or have shape ({n1},).")
        epsilon = jnp.broadcast_to(epsilon, (n1,)).astype(
            bridge.cochain.hodge_stars[1].dtype
        )
        epsilon = eqx.error_if(
            epsilon,
            jnp.any(~jnp.isfinite(epsilon) | (epsilon <= 0.0)),
            "permittivity must be positive and finite.",
        )
        active = bridge.cochain.active_mask(
            0, "absolute" if boundary == "periodic" else "relative"
        )
        action = _CochainPoissonAction(bridge, epsilon, boundary, active)
        operator_id = canonical_fingerprint(
            {
                "kind": "cochain-electrostatic-operator",
                "bridge": bridge.bridge_id,
                "boundary": boundary,
            }
        )
        space = bridge.cochain.space(0).vector_space
        operator = FunctionLinearOperator(
            action,
            source=space,
            target=space,
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                },
            ),
            operator_id=operator_id,
        )
        problem = LinearSystem(
            operator,
            problem_id=canonical_fingerprint(
                {"kind": "cochain-electrostatic-system", "operator": operator_id}
            ),
        )
        policy = (
            LinearSolvePolicy(
                ConjugateGradient(),
                tolerance=TolerancePolicy(
                    relative=tolerance_, absolute=tolerance_, max_steps=iterations
                ),
            )
            if linear_policy is None
            else linear_policy
        )
        if not isinstance(policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be LinearSolvePolicy or None.")
        prepared = prepare(problem, policy)
        self.bridge = bridge
        self.permittivity = epsilon
        self.boundary = boundary
        self.tolerance = tolerance_
        self.compatibility_tolerance = compatibility_
        self.active = active
        self.linear_policy = policy
        self.linear_problem = problem
        self.prepared_linear = prepared
        self.operator_id = operator_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cochain-electrostatic-plan",
                "operator": operator_id,
                "linear": prepared.plan.plan_id,
                "tolerance": tolerance_,
                "compatibility": compatibility_,
            }
        )

    def solve(
        self,
        charge: ArrayLike,
        /,
        *,
        initial_potential: ArrayLike | None = None,
    ) -> CochainElectrostaticResult:
        cochain = self.bridge.cochain
        rho = jnp.asarray(charge, dtype=cochain.hodge_stars[0].dtype)
        if rho.shape != (cochain.cell_counts[0],):
            raise ValueError("charge must be a degree-zero cochain.")
        weights = cochain.hodge_stars[0].astype(rho.dtype)
        total_charge = jnp.sum(weights * rho)
        charge_scale = jnp.maximum(jnp.sum(weights * jnp.abs(rho)), 1.0)
        compatibility = jnp.abs(total_charge)
        if self.boundary == "periodic":
            rho = eqx.error_if(
                rho,
                compatibility > self.compatibility_tolerance * charge_scale,
                "Periodic electrostatic charge is incompatible; supply an explicit background.",
            )
            rhs = rho
        else:
            rhs = jnp.where(self.active, rho, 0.0)
        initial = (
            jnp.zeros_like(rho)
            if initial_potential is None
            else jnp.asarray(initial_potential, dtype=rho.dtype)
        )
        if initial.shape != rho.shape:
            raise ValueError("initial_potential must match charge.")
        if self.boundary == "periodic":
            initial = initial - jnp.sum(weights * initial) / jnp.sum(weights)
        else:
            initial = jnp.where(self.active, initial, 0.0)
        linear = solve(self.prepared_linear, rhs, initial_guess=initial)
        potential = linear.value
        if self.boundary == "periodic":
            potential = potential - jnp.sum(weights * potential) / jnp.sum(weights)
        else:
            potential = jnp.where(self.active, potential, 0.0)
        electric = -cochain.exterior_derivative(
            0,
            potential,
            boundary_policy="absolute" if self.boundary == "periodic" else "relative",
        )
        action = _CochainPoissonAction(
            self.bridge, self.permittivity, self.boundary, self.active
        )
        residual = action(potential) - rhs
        residual_norm = jnp.sqrt(jnp.real(cochain.space(0).vector_space.inner(residual, residual)))
        gauge = jnp.abs(jnp.sum(weights * potential) / jnp.sum(weights))
        integrated = self.bridge.unpack(1, electric)
        measures = self.bridge.unpack(1, cochain.primal_measures[1])
        physical = tuple(value / measure for value, measure in zip(integrated, measures, strict=True))
        field_energy = 0.5 * jnp.real(
            cochain.space(1).vector_space.inner(electric, self.permittivity * electric)
        )
        finite = (
            jnp.all(jnp.isfinite(potential))
            & jnp.all(jnp.isfinite(electric))
            & jnp.isfinite(residual_norm)
            & jnp.isfinite(field_energy)
        )
        rhs_norm = jnp.sqrt(
            jnp.real(cochain.space(0).vector_space.inner(rhs, rhs))
        )
        converged = linear.successful & (
            residual_norm <= self.tolerance * jnp.maximum(1.0, rhs_norm)
        )
        return CochainElectrostaticResult(
            rho,
            potential,
            electric,
            physical,
            residual,
            residual_norm,
            compatibility,
            gauge,
            field_energy,
            linear,
            finite,
            converged,
            finite & converged,
            self.plan_id,
        )


__all__ = [
    "CochainElectrostaticPlan",
    "CochainElectrostaticResult",
    "ElectrostaticBoundaryKind",
]
