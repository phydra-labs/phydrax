#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._admissibility import AdmissibilityHeader, AdmissibilityReason
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume._incompressible import (
    FaceVelocity,
    PreparedMACOperators,
)
from ..discretization.finite_volume._mac_electrochemical import mac_cell_to_faces
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


class MACElectrostaticBoundaryKind(StrEnum):
    PERIODIC = "periodic"
    NEUMANN = "neumann"


class MACElectrostaticBoundaryPlan(StrictModule, NonTrainableState):
    operators: PreparedMACOperators
    kind: MACElectrostaticBoundaryKind = eqx.field(static=True)
    neumann_source: Array
    boundary_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        kind: MACElectrostaticBoundaryKind,
        /,
        *,
        neumann_source: ArrayLike = 0.0,
    ) -> None:
        if not isinstance(operators, PreparedMACOperators) or not isinstance(
            kind, MACElectrostaticBoundaryKind
        ):
            raise TypeError("MAC electrostatic boundary inputs are invalid.")
        periodic = all(
            axis.periodic for axis in operators.discretization.grid.structured_axes
        )
        if kind is MACElectrostaticBoundaryKind.PERIODIC and not periodic:
            raise ValueError("Periodic MAC electrostatics requires all axes periodic.")
        if kind is MACElectrostaticBoundaryKind.NEUMANN and periodic:
            raise ValueError("Bounded MAC electrostatics requires a bounded axis.")
        source = jnp.broadcast_to(
            jnp.asarray(neumann_source, dtype=operators.pressure_space.dtype),
            operators.discretization.cell_shape,
        )
        if not bool(jnp.all(jnp.isfinite(source))):
            raise ValueError("MAC electrostatic Neumann source must be finite.")
        self.operators = operators
        self.kind = kind
        self.neumann_source = source
        self.boundary_id = canonical_fingerprint(
            {
                "kind": "mac-electrostatic-boundary",
                "operators": operators.prepared_id,
                "boundary_kind": kind.value,
                "neumann_source": array_tree_fingerprint(source),
            }
        )

    @classmethod
    def periodic(cls, operators: PreparedMACOperators, /):
        return cls(operators, MACElectrostaticBoundaryKind.PERIODIC)

    @classmethod
    def neumann(cls, operators: PreparedMACOperators, neumann_source: ArrayLike = 0.0, /):
        return cls(
            operators,
            MACElectrostaticBoundaryKind.NEUMANN,
            neumann_source=neumann_source,
        )


class _MACPoissonAction(StrictModule, NonTrainableState):
    operators: PreparedMACOperators
    face_permittivity: FaceVelocity

    def __call__(self, potential: Array, /) -> Array:
        value = self.operators.gauge_project(potential)
        electric_gradient = self.operators.gradient(value)
        core = -self.operators.divergence(
            tuple(
                permittivity * gradient
                for permittivity, gradient in zip(
                    self.face_permittivity, electric_gradient, strict=True
                )
            )
        )
        volumes = self.operators.discretization.cell_volumes.astype(value.dtype)
        mean = jnp.sum(volumes * potential) / jnp.sum(volumes)
        return core + mean


class MACElectrostaticResult(StrictModule):
    charge: Array
    potential: Array
    electric_field: FaceVelocity
    poisson_residual: Array
    residual_norm: Array
    compatibility_defect: Array
    gauge_defect: Array
    field_energy: Array
    linear: LinearSolveResult
    header: AdmissibilityHeader
    operators_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class MACElectrostaticPlan(StrictModule, NonTrainableState):
    operators: PreparedMACOperators
    boundary: MACElectrostaticBoundaryPlan
    cell_permittivity: Array
    face_permittivity: FaceVelocity
    tolerance: float = eqx.field(static=True)
    compatibility_tolerance: float = eqx.field(static=True)
    linear_policy: LinearSolvePolicy
    linear_problem: LinearSystem
    prepared_linear: PreparedLinearSolve
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        boundary: MACElectrostaticBoundaryPlan,
        /,
        *,
        permittivity: ArrayLike,
        tolerance: float = 1.0e-10,
        compatibility_tolerance: float = 1.0e-10,
        maximum_iterations: int = 500,
        linear_policy: LinearSolvePolicy | None = None,
    ) -> None:
        epsilon = jnp.broadcast_to(
            jnp.asarray(permittivity, dtype=operators.pressure_space.dtype),
            operators.discretization.cell_shape,
        )
        tolerance_ = float(tolerance)
        compatibility_ = float(compatibility_tolerance)
        iterations = int(maximum_iterations)
        if (
            not isinstance(operators, PreparedMACOperators)
            or not isinstance(boundary, MACElectrostaticBoundaryPlan)
            or boundary.operators.prepared_id != operators.prepared_id
            or not bool(jnp.all(jnp.isfinite(epsilon) & (epsilon > 0.0)))
            or not np.isfinite(tolerance_)
            or tolerance_ <= 0.0
            or not np.isfinite(compatibility_)
            or compatibility_ < 0.0
            or iterations <= 0
        ):
            raise ValueError("MAC electrostatic plan inputs are invalid.")
        face_epsilon = []
        for axis in range(len(operators.discretization.cell_shape)):
            tail, head, _ = mac_cell_to_faces(operators, epsilon, axis)
            face_epsilon.append(0.5 * (tail + head))
        action = _MACPoissonAction(operators, tuple(face_epsilon))
        operator_id = canonical_fingerprint(
            {
                "kind": "mac-electrostatic-operator",
                "operators": operators.prepared_id,
                "boundary": boundary.boundary_id,
                "permittivity": array_tree_fingerprint(epsilon),
            }
        )
        operator = FunctionLinearOperator(
            action,
            source=operators.pressure_space,
            target=operators.pressure_space,
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
                {"kind": "mac-electrostatic-system", "operator": operator_id}
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
        self.operators = operators
        self.boundary = boundary
        self.cell_permittivity = epsilon
        self.face_permittivity = tuple(face_epsilon)
        self.tolerance = tolerance_
        self.compatibility_tolerance = compatibility_
        self.linear_policy = policy
        self.linear_problem = problem
        self.prepared_linear = prepared
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-electrostatic-plan",
                "operator": operator_id,
                "linear": prepared.plan.plan_id,
                "tolerance": tolerance_,
                "compatibility_tolerance": compatibility_,
            }
        )

    def solve(
        self,
        charge: ArrayLike,
        /,
        *,
        initial_potential: ArrayLike | None = None,
    ) -> MACElectrostaticResult:
        rho = self.operators.validate_pressure(charge)
        volumes = self.operators.discretization.cell_volumes.astype(rho.dtype)
        source = rho + self.boundary.neumann_source.astype(rho.dtype)
        total_source = jnp.sum(volumes * source)
        source_scale = jnp.maximum(jnp.sum(volumes * jnp.abs(source)), 1.0)
        compatibility = jnp.abs(total_source)
        rhs = self.operators.compatibility_project(source)
        initial = (
            jnp.zeros_like(rho)
            if initial_potential is None
            else self.operators.validate_pressure(initial_potential)
        )
        initial = self.operators.gauge_project(initial)
        linear = solve(self.prepared_linear, rhs, initial_guess=initial)
        potential = self.operators.gauge_project(linear.value)
        electric = tuple(-value for value in self.operators.gradient(potential))
        action = _MACPoissonAction(self.operators, self.face_permittivity)
        residual = action(potential) - rhs
        residual_norm = jnp.sqrt(jnp.sum(volumes * residual**2))
        rhs_norm = jnp.sqrt(jnp.sum(volumes * rhs**2))
        gauge = jnp.abs(jnp.sum(volumes * potential) / jnp.sum(volumes))
        field_energy = 0.5 * sum(
            jnp.sum(measure * epsilon * field**2)
            for measure, epsilon, field in zip(
                self.operators.face_dual_measures,
                self.face_permittivity,
                electric,
                strict=True,
            )
        )
        compatible = compatibility <= self.compatibility_tolerance * source_scale
        converged = linear.successful & (
            residual_norm <= self.tolerance * jnp.maximum(rhs_norm, 1.0)
        )
        finite = (
            jnp.all(jnp.isfinite(potential))
            & jnp.all(
                jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in electric))
            )
            & jnp.isfinite(field_energy)
            & jnp.isfinite(residual_norm)
        )
        supported = finite & compatible & converged
        reasons = jnp.asarray(0, dtype=jnp.uint32)
        reasons = jnp.where(
            finite,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.NONFINITE), jnp.uint32),
        )
        reasons = jnp.where(
            compatible & converged,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.OUTSIDE_SUPPORT), jnp.uint32),
        )
        header = AdmissibilityHeader(
            jnp.where(
                supported,
                self.tolerance * jnp.maximum(rhs_norm, 1.0) - residual_norm,
                -1.0,
            ),
            reasons,
            self.plan_id,
            canonical_fingerprint(
                {"kind": "mac-electrostatic-evidence", "plan": self.plan_id}
            ),
        )
        return MACElectrostaticResult(
            rho,
            potential,
            electric,
            residual,
            residual_norm,
            compatibility,
            gauge,
            field_energy,
            linear,
            header,
            self.operators.prepared_id,
            self.plan_id,
        )


__all__ = [
    "MACElectrostaticBoundaryKind",
    "MACElectrostaticBoundaryPlan",
    "MACElectrostaticPlan",
    "MACElectrostaticResult",
]
