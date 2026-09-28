#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Implicit variable-density, variable-viscosity MAC momentum stage.

The stage solves ``(rho_f + dt A_mu) u = rho_f u_star - dt b_mu`` where
``A_mu`` is the positive homogeneous ``-div(2 mu S_d)`` action of
:class:`PreparedMACVariationalViscosityAction`, ``b_mu`` its prescribed-wall
offset and ``rho_f`` a runtime face density. ``A_mu`` is self-adjoint in the
dual-measure face pairing, so the system is assembled in the measure-weighted
form ``M (rho_f + dt A_mu)``, which is symmetric positive definite in plain
coordinates. The unknowns are the free faces that carry mass or viscous
stress; essential wall faces take their stage value and decoupled faces keep
their input value through identity rows. Massless faces coupled only through
viscous stress satisfy the quasi-static traction balance, the natural
free-surface condition; the system is then only semidefinite if a strain-free
combination of such faces exists, and that consistent null component keeps its
input value. One native conjugate-gradient solve is prepared at construction
and refreshed with the runtime coefficients.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume import (
    FaceVelocity,
    MACBoundaryStageData,
    PreparedMACMomentumOperators,
    PreparedMACVariationalViscosityAction,
)
from ..linalg import (
    ArraySpace,
    BlockSpace,
    FunctionLinearOperator,
    LinearSolveControl,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    PCG,
    prepare,
    PreparedLinearSolve,
    refresh,
    solve,
    TolerancePolicy,
)


def _euclidean_norm(values: FaceVelocity, /) -> Array:
    return jnp.sqrt(jnp.sum(jnp.stack(tuple(jnp.sum(value * value) for value in values))))


def _kinetic_energy(
    measure: FaceVelocity, density: FaceVelocity, velocity: FaceVelocity, /
) -> Array:
    return 0.5 * jnp.sum(
        jnp.stack(
            tuple(
                jnp.sum(weight * mass * value * value)
                for weight, mass, value in zip(measure, density, velocity, strict=True)
            )
        )
    )


class _ImplicitViscousAction(StrictModule, NonTrainableState):
    action: PreparedMACVariationalViscosityAction
    measure: FaceVelocity
    face_density: FaceVelocity
    cell_viscosity: Array
    step_size: Array
    active: tuple[Array, ...]

    def __init__(
        self,
        action: PreparedMACVariationalViscosityAction,
        measure: FaceVelocity,
        face_density: FaceVelocity,
        cell_viscosity: Array,
        step_size: Array,
        active: tuple[Array, ...],
        /,
    ) -> None:
        self.action = action
        self.measure = measure
        self.face_density = face_density
        self.cell_viscosity = cell_viscosity
        self.step_size = step_size
        self.active = active

    def __call__(self, velocity: FaceVelocity, /) -> FaceVelocity:
        # Inactive rows and columns of M (rho_f + dt A_mu) vanish except for the
        # mass of essential faces, so identity rows restrict the unknowns
        # without breaking symmetry.
        viscous = self.action.positive_operator_action(velocity, self.cell_viscosity)
        return tuple(
            measure * jnp.where(active, density * value + self.step_size * rate, value)
            for measure, density, value, rate, active in zip(
                self.measure,
                self.face_density,
                velocity,
                viscous,
                self.active,
                strict=True,
            )
        )


class MACVariationalViscosityResult(StrictModule):
    """Implicit viscous stage velocity with energy and solve evidence.

    ``velocity`` is the stage-enforced candidate when ``successful`` and the
    unchanged input otherwise. ``dissipation`` is the rate
    ``integral 2 mu S_d:S_d`` and ``wall_power`` the rate of work done by the
    prescribed boundary velocities, both evaluated at the candidate.
    ``energy_before`` is the face kinetic energy ``1/2 sum M rho_f u^2`` of the
    stage-enforced input and ``energy_after`` that of the candidate. Implicit
    Euler guarantees ``energy_after - energy_before <= dt wall_power``;
    ``energy_increase`` is the nonnegative excess over that bound.
    ``decoupled_face_count`` counts free faces with zero density and no viscous
    coupling, which carry no momentum equation and keep their input value.
    ``linear_status`` is the native :class:`phydrax.linalg.LinearSolveStatus`
    code and ``residual_norm`` its true residual in measure-weighted form.
    """

    velocity: FaceVelocity
    candidate_velocity: FaceVelocity
    dissipation: Array
    wall_power: Array
    energy_before: Array
    energy_after: Array
    energy_increase: Array
    residual_norm: Array
    iterations: Array
    linear_status: Array
    decoupled_face_count: Array
    finite: Array
    converged: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class MACVariationalViscosityPlan(StrictModule, NonTrainableState):
    """Prepared implicit variable-density, variable-viscosity MAC stage.

    The linear solve stops and is accepted on the same true residual in the
    measure-weighted coordinates, with threshold
    ``tolerance * (||b|| + ||H u_0||)`` for right-hand side ``b``, stage matrix
    ``H`` and stage-enforced input ``u_0``: it is dimensionally consistent and
    accepts the exact zero solution of a zero right-hand side.
    """

    action: PreparedMACVariationalViscosityAction
    space: BlockSpace
    measure: FaceVelocity
    free: tuple[Array, ...]
    prepared: PreparedLinearSolve
    tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        momentum: PreparedMACMomentumOperators,
        /,
        *,
        tolerance: float = 1.0e-9,
        maximum_iterations: int = 500,
    ) -> None:
        if not isinstance(momentum, PreparedMACMomentumOperators):
            raise TypeError("momentum must be PreparedMACMomentumOperators.")
        tolerance_ = float(tolerance)
        iterations = int(maximum_iterations)
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0 or iterations <= 0:
            raise ValueError(
                "Viscous tolerance must be positive and finite and "
                "maximum_iterations positive."
            )
        action = PreparedMACVariationalViscosityAction(momentum)
        operators = momentum.operators
        discretization = operators.discretization
        dtype = operators.pressure_space.dtype
        space = BlockSpace(
            tuple(
                ArraySpace(layout.shape, dtype=dtype)
                for layout in discretization.face_layouts
            ),
            names=discretization.grid.axis_names,
        )
        measure = tuple(operators.face_dual_measures)
        unit = tuple(
            jnp.ones(layout.shape, dtype=dtype) for layout in discretization.face_layouts
        )
        free = tuple(value != 0.0 for value in momentum.boundaries.homogeneous_rate(unit))
        operator_id = canonical_fingerprint(
            {
                "kind": "mac-implicit-viscous-operator",
                "action": action.action_id,
                "form": "dual-measure-weighted",
                "unknowns": "free-faces-with-mass-or-viscous-coupling",
            }
        )
        problem_id = canonical_fingerprint(
            {"kind": "mac-implicit-viscous-system", "operator": operator_id}
        )
        problem = LinearSystem(
            self._operator(
                _ImplicitViscousAction(
                    action,
                    measure,
                    unit,
                    jnp.zeros(discretization.cell_shape, dtype=dtype),
                    jnp.ones((), dtype=dtype),
                    free,
                ),
                space,
                operator_id,
            ),
            problem_id=problem_id,
        )
        policy = LinearSolvePolicy(
            PCG(),
            tolerance=TolerancePolicy(
                relative=tolerance_, absolute=0.0, max_steps=iterations
            ),
        )
        prepared = prepare(problem, policy)
        self.action = action
        self.space = space
        self.measure = measure
        self.free = free
        self.prepared = prepared
        self.tolerance = tolerance_
        self.maximum_iterations = iterations
        self.operator_id = operator_id
        self.problem_id = problem_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-variational-viscosity",
                "operator": operator_id,
                "linear_plan": prepared.plan.plan_id,
                "tolerance": tolerance_,
                "maximum_iterations": iterations,
            }
        )

    @staticmethod
    def _operator(
        action: _ImplicitViscousAction,
        space: BlockSpace,
        operator_id: str,
        /,
    ) -> FunctionLinearOperator:
        return FunctionLinearOperator(
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

    def solve(
        self,
        velocity: FaceVelocity,
        face_density: FaceVelocity,
        cell_viscosity: ArrayLike,
        step_size: ArrayLike,
        boundary_stage: MACBoundaryStageData,
        /,
    ) -> MACVariationalViscosityResult:
        """Advance ``velocity`` through one implicit viscous stage."""
        momentum = self.action.momentum
        operators = momentum.operators
        dtype = operators.pressure_space.dtype
        values = operators.validate_velocity(velocity)
        density = operators.validate_velocity(face_density)
        density_valid = jnp.all(
            jnp.stack(
                tuple(jnp.all(jnp.isfinite(value) & (value >= 0.0)) for value in density)
            )
        )
        density = tuple(
            eqx.error_if(
                value.astype(dtype),
                ~density_valid,
                "Viscous face density must be finite and nonnegative.",
            )
            for value in density
        )
        viscosity = jnp.asarray(cell_viscosity, dtype=dtype)
        step = jnp.asarray(step_size, dtype=dtype).reshape(())
        step = eqx.error_if(
            step,
            ~jnp.isfinite(step) | (step <= 0.0),
            "Viscous step_size must be positive and finite.",
        )
        stage = momentum.boundaries.validate_stage(boundary_stage)
        coupled = self.action.coupled_faces(viscosity)
        active = tuple(
            linked | (free & (mass > 0.0))
            for linked, free, mass in zip(coupled, self.free, density, strict=True)
        )
        decoupled_count = jnp.sum(
            jnp.stack(
                tuple(
                    jnp.sum(free & ~selected, dtype=jnp.int32)
                    for free, selected in zip(self.free, active, strict=True)
                )
            ),
            dtype=jnp.int32,
        )
        initial = momentum.boundaries.enforce(values, stage)
        offset = self.action.boundary_affine_action(viscosity, stage)
        system = _ImplicitViscousAction(
            self.action, self.measure, density, viscosity, step, active
        )
        rhs = tuple(
            measure * jnp.where(selected, mass * value - step * wall, value)
            for measure, mass, value, wall, selected in zip(
                self.measure, density, initial, offset, active, strict=True
            )
        )
        absolute = self.tolerance * _euclidean_norm(system(initial))
        prepared = refresh(
            self.prepared,
            LinearSystem(
                self._operator(system, self.space, self.operator_id),
                problem_id=self.problem_id,
            ),
        )
        linear = solve(
            prepared,
            rhs,
            initial_guess=initial,
            control=LinearSolveControl(absolute_tolerance=absolute),
        )
        candidate = momentum.boundaries.enforce(tuple(linear.value), stage)
        evaluation = self.action.evaluate(candidate, viscosity, stage)
        energy_before = _kinetic_energy(self.measure, density, initial)
        energy_after = _kinetic_energy(self.measure, density, candidate)
        wall_work = step * evaluation.boundary_power
        excess = energy_after - energy_before - wall_work
        energy_increase = jnp.maximum(excess, 0.0)
        residual_norm = jnp.asarray(linear.diagnostics.residual_norm).reshape(())
        # The implicit-Euler energy identity holds up to <u, r> for the
        # measure-weighted residual r; the remainder is summation roundoff.
        energy_scale = (
            energy_before
            + energy_after
            + jnp.abs(wall_work)
            + step * jnp.abs(evaluation.integrated_dissipation)
        )
        energy_tolerance = (
            _euclidean_norm(candidate) * residual_norm
            + 4096.0 * jnp.finfo(dtype).eps * energy_scale
        )
        finite = (
            evaluation.finite
            & jnp.all(
                jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in candidate))
            )
            & jnp.isfinite(excess)
        )
        converged = linear.successful
        successful = (
            stage.successful
            & finite
            & converged
            & evaluation.successful
            & (energy_increase <= energy_tolerance)
        )
        accepted = tuple(
            jnp.where(successful, proposed, original)
            for proposed, original in zip(candidate, values, strict=True)
        )
        return MACVariationalViscosityResult(
            velocity=accepted,
            candidate_velocity=candidate,
            dissipation=evaluation.integrated_dissipation,
            wall_power=evaluation.boundary_power,
            energy_before=energy_before,
            energy_after=energy_after,
            energy_increase=energy_increase,
            residual_norm=residual_norm,
            iterations=jnp.asarray(
                linear.diagnostics.iterations, dtype=jnp.int32
            ).reshape(()),
            linear_status=jnp.asarray(linear.status, dtype=jnp.int32).reshape(()),
            decoupled_face_count=decoupled_count,
            finite=finite,
            converged=converged,
            successful=successful,
            plan_id=self.plan_id,
        )


__all__ = ["MACVariationalViscosityPlan", "MACVariationalViscosityResult"]
