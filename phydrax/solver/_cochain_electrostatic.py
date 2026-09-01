#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
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


class ElectrostaticBoundaryKind(StrEnum):
    PERIODIC = "periodic"
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    MIXED = "mixed"


class CochainElectrostaticBoundaryPlan(StrictModule, NonTrainableState):
    kind: ElectrostaticBoundaryKind = eqx.field(static=True)
    dirichlet_mask: Array
    dirichlet_values: Array
    neumann_source: Array
    gauge_required: bool = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)

    def __init__(
        self,
        bridge: StructuredCochainBridge,
        kind: ElectrostaticBoundaryKind,
        /,
        *,
        dirichlet_mask: ArrayLike | None = None,
        dirichlet_values: ArrayLike = 0.0,
        neumann_source: ArrayLike = 0.0,
    ):
        if not isinstance(bridge, StructuredCochainBridge):
            raise TypeError("bridge must be StructuredCochainBridge.")
        if not isinstance(kind, ElectrostaticBoundaryKind):
            raise TypeError("kind must be ElectrostaticBoundaryKind.")
        count = bridge.cochain.cell_counts[0]
        periodic_grid = all(axis.periodic for axis in bridge.grid.structured_axes)
        if kind is ElectrostaticBoundaryKind.PERIODIC and not periodic_grid:
            raise ValueError("Periodic electrostatics requires every grid axis periodic.")
        if kind is not ElectrostaticBoundaryKind.PERIODIC and periodic_grid:
            raise ValueError(
                "Bounded electrostatic boundaries require bounded grid axes."
            )
        relative_active = bridge.cochain.active_mask(0, "relative")
        default_mask = ~relative_active
        if kind is ElectrostaticBoundaryKind.PERIODIC:
            mask = jnp.zeros((count,), dtype=bool)
        elif kind is ElectrostaticBoundaryKind.DIRICHLET:
            mask = (
                default_mask
                if dirichlet_mask is None
                else jnp.asarray(dirichlet_mask, dtype=bool)
            )
        elif kind is ElectrostaticBoundaryKind.NEUMANN:
            mask = jnp.zeros((count,), dtype=bool)
        else:
            if dirichlet_mask is None:
                raise ValueError("Mixed boundaries require dirichlet_mask.")
            mask = jnp.asarray(dirichlet_mask, dtype=bool)
        if mask.shape != (count,):
            raise ValueError("dirichlet_mask must be a degree-zero cochain.")
        values = jnp.broadcast_to(jnp.asarray(dirichlet_values), (count,))
        source = jnp.broadcast_to(
            jnp.asarray(neumann_source, dtype=values.dtype), (count,)
        )
        if kind in (
            ElectrostaticBoundaryKind.DIRICHLET,
            ElectrostaticBoundaryKind.MIXED,
        ) and not bool(jnp.any(mask)):
            raise ValueError(
                "Dirichlet or mixed boundaries require at least one fixed node."
            )
        if kind in (
            ElectrostaticBoundaryKind.PERIODIC,
            ElectrostaticBoundaryKind.NEUMANN,
        ) and bool(jnp.any(mask)):
            raise ValueError("Periodic/Neumann boundaries cannot fix potential nodes.")
        if not bool(jnp.all(jnp.isfinite(values))) or not bool(
            jnp.all(jnp.isfinite(source))
        ):
            raise ValueError("Boundary values and sources must be finite.")
        self.kind = kind
        self.dirichlet_mask = mask
        self.dirichlet_values = values
        self.neumann_source = source
        self.gauge_required = kind in (
            ElectrostaticBoundaryKind.PERIODIC,
            ElectrostaticBoundaryKind.NEUMANN,
        )
        self.boundary_id = canonical_fingerprint(
            {
                "kind": "cochain-electrostatic-boundary",
                "boundary_kind": kind.value,
                "mask": array_tree_fingerprint(np.asarray(mask)),
                "values": array_tree_fingerprint(np.asarray(values)),
                "neumann": array_tree_fingerprint(np.asarray(source)),
            }
        )

    @classmethod
    def periodic(cls, bridge: StructuredCochainBridge, /):
        return cls(bridge, ElectrostaticBoundaryKind.PERIODIC)

    @classmethod
    def dirichlet(
        cls,
        bridge: StructuredCochainBridge,
        values: ArrayLike = 0.0,
        /,
        *,
        mask: ArrayLike | None = None,
    ):
        return cls(
            bridge,
            ElectrostaticBoundaryKind.DIRICHLET,
            dirichlet_mask=mask,
            dirichlet_values=values,
        )

    @classmethod
    def neumann(cls, bridge: StructuredCochainBridge, source: ArrayLike = 0.0, /):
        return cls(
            bridge,
            ElectrostaticBoundaryKind.NEUMANN,
            neumann_source=source,
        )


class _CochainPoissonAction(StrictModule, NonTrainableState):
    bridge: StructuredCochainBridge
    permittivity: Array
    boundary: CochainElectrostaticBoundaryPlan
    active: Array

    def __call__(self, potential: Array, /) -> Array:
        cochain = self.bridge.cochain
        weights = cochain.hodge_stars[0].astype(potential.dtype)
        if self.boundary.gauge_required:
            mean = jnp.sum(weights * potential) / jnp.sum(weights)
            value = potential - mean
            electric = cochain.exterior_derivative(0, value)
            core = -cochain.codifferential(1, self.permittivity * electric)
            return core + mean
        value = jnp.where(self.active, potential, 0.0)
        electric = cochain.exterior_derivative(0, value)
        core = -cochain.codifferential(1, self.permittivity * electric)
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
    boundary: CochainElectrostaticBoundaryPlan
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
        boundary: CochainElectrostaticBoundaryPlan,
        /,
        *,
        permittivity: ArrayLike = 1.0,
        tolerance: float = 1.0e-10,
        compatibility_tolerance: float = 1.0e-10,
        maximum_iterations: int = 500,
        linear_policy: LinearSolvePolicy | None = None,
    ):
        if not isinstance(bridge, StructuredCochainBridge):
            raise TypeError("bridge must be StructuredCochainBridge.")
        if not isinstance(boundary, CochainElectrostaticBoundaryPlan):
            raise TypeError("boundary must be CochainElectrostaticBoundaryPlan.")
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
        active = ~boundary.dirichlet_mask
        action = _CochainPoissonAction(bridge, epsilon, boundary, active)
        operator_id = canonical_fingerprint(
            {
                "kind": "cochain-electrostatic-operator",
                "bridge": bridge.bridge_id,
                "boundary": boundary.boundary_id,
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
        source = rho + self.boundary.neumann_source.astype(rho.dtype)
        total_source = jnp.sum(weights * source)
        source_scale = jnp.maximum(jnp.sum(weights * jnp.abs(source)), 1.0)
        compatibility = jnp.abs(total_source)
        if self.boundary.gauge_required:
            source = eqx.error_if(
                source,
                compatibility > self.compatibility_tolerance * source_scale,
                "Electrostatic charge/flux is incompatible with the gauge boundary.",
            )
        lift = self.boundary.dirichlet_values.astype(rho.dtype)
        action = _CochainPoissonAction(
            self.bridge, self.permittivity, self.boundary, self.active
        )
        if self.boundary.gauge_required:
            rhs = source
        else:
            lift_action = -cochain.codifferential(
                1,
                self.permittivity * cochain.exterior_derivative(0, lift),
            )
            rhs = jnp.where(self.active, source - lift_action, 0.0)
        initial = (
            jnp.zeros_like(rho)
            if initial_potential is None
            else jnp.asarray(initial_potential, dtype=rho.dtype)
        )
        if initial.shape != rho.shape:
            raise ValueError("initial_potential must match charge.")
        if self.boundary.gauge_required:
            initial = initial - jnp.sum(weights * initial) / jnp.sum(weights)
        else:
            initial = jnp.where(self.active, initial - lift, 0.0)
        linear = solve(self.prepared_linear, rhs, initial_guess=initial)
        homogeneous = linear.value
        if self.boundary.gauge_required:
            potential = homogeneous - jnp.sum(weights * homogeneous) / jnp.sum(weights)
        else:
            potential = jnp.where(self.active, homogeneous + lift, lift)
        electric = -cochain.exterior_derivative(0, potential)
        homogeneous_residual = action(homogeneous) - rhs
        residual = jnp.where(self.active, homogeneous_residual, potential - lift)
        residual_norm = jnp.sqrt(
            jnp.real(cochain.space(0).vector_space.inner(residual, residual))
        )
        gauge = (
            jnp.abs(jnp.sum(weights * potential) / jnp.sum(weights))
            if self.boundary.gauge_required
            else jnp.max(jnp.abs(jnp.where(~self.active, potential - lift, 0.0)))
        )
        integrated = self.bridge.unpack(1, electric)
        measures = self.bridge.unpack(1, cochain.primal_measures[1])
        physical = tuple(
            value / measure for value, measure in zip(integrated, measures, strict=True)
        )
        field_energy = 0.5 * jnp.real(
            cochain.space(1).vector_space.inner(electric, self.permittivity * electric)
        )
        finite = (
            jnp.all(jnp.isfinite(potential))
            & jnp.all(jnp.isfinite(electric))
            & jnp.isfinite(residual_norm)
            & jnp.isfinite(field_energy)
        )
        rhs_norm = jnp.sqrt(jnp.real(cochain.space(0).vector_space.inner(rhs, rhs)))
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
    "CochainElectrostaticBoundaryPlan",
    "CochainElectrostaticPlan",
    "CochainElectrostaticResult",
    "ElectrostaticBoundaryKind",
]
