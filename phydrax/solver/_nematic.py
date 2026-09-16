#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_difference import (
    diagonalize_fd_laplacian,
    FDLaplacianSolvePlan,
    PreparedFiniteDifferenceDiscretization,
)
from ..discretization.finite_volume import FaceVelocity, PreparedMACOperators
from ..equations._nematic import (
    beris_edwards_constitutive_fields,
    BerisEdwardsConstitutiveFields,
    BerisEdwardsParameters,
    LandauDeGennesClosure,
    LandauDeGennesParameters,
    NematicThermodynamicFields,
)
from ..equations._nematic_anchoring import (
    NematicAnchoringFields,
    NematicAnchoringPlan,
)


class NematicEvaluation(StrictModule):
    thermodynamics: NematicThermodynamicFields
    anchoring: NematicAnchoringFields | None
    compact_gradient: Array
    compact_laplacian: Array
    total_free_energy: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class NematicStepResult(StrictModule):
    compact_q: Array
    evaluation: NematicEvaluation
    successful: Array
    plan_id: str = eqx.field(static=True)


class PreparedNematicDynamics(StrictModule, NonTrainableState):
    finite_difference: PreparedFiniteDifferenceDiscretization
    closure: LandauDeGennesClosure
    thermodynamic_parameters: LandauDeGennesParameters
    dynamics_parameters: BerisEdwardsParameters
    anchoring: NematicAnchoringPlan | None
    energy_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        finite_difference: PreparedFiniteDifferenceDiscretization,
        closure: LandauDeGennesClosure,
        thermodynamic_parameters: LandauDeGennesParameters,
        dynamics_parameters: BerisEdwardsParameters,
        /,
        *,
        anchoring: NematicAnchoringPlan | None = None,
        energy_tolerance: float = 1.0e-10,
    ):
        if not isinstance(finite_difference, PreparedFiniteDifferenceDiscretization):
            raise TypeError(
                "finite_difference must be PreparedFiniteDifferenceDiscretization."
            )
        if not isinstance(closure, LandauDeGennesClosure):
            raise TypeError("closure must be LandauDeGennesClosure.")
        if not isinstance(thermodynamic_parameters, LandauDeGennesParameters):
            raise TypeError("thermodynamic_parameters must be LandauDeGennesParameters.")
        if not isinstance(dynamics_parameters, BerisEdwardsParameters):
            raise TypeError("dynamics_parameters must be BerisEdwardsParameters.")
        if anchoring is not None and (
            not isinstance(anchoring, NematicAnchoringPlan)
            or anchoring.basis.basis_id != closure.basis.basis_id
        ):
            raise TypeError("anchoring must use the closure nematic basis.")
        tolerance = float(energy_tolerance)
        if tolerance < 0.0:
            raise ValueError("energy_tolerance must be nonnegative.")
        self.finite_difference = finite_difference
        self.closure = closure
        self.thermodynamic_parameters = thermodynamic_parameters
        self.dynamics_parameters = dynamics_parameters
        self.anchoring = anchoring
        self.energy_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "prepared-nematic-dynamics",
                "finite_difference": finite_difference.prepared_id,
                "closure": closure.closure_id,
                "anchoring": None if anchoring is None else anchoring.plan_id,
                "energy_tolerance": tolerance,
            }
        )

    def evaluate(
        self,
        compact_q: ArrayLike,
        /,
        *,
        electric_field: ArrayLike | None = None,
    ) -> NematicEvaluation:
        compact = jnp.asarray(compact_q)
        expected = self.finite_difference.grid.shape + (
            self.closure.basis.component_count,
        )
        if compact.shape != expected:
            raise ValueError("compact_q must match finite-difference grid and basis.")
        gradients = []
        second = []
        for axis in self.finite_difference.grid.axis_names:
            gradients.append(
                _apply_components(self.finite_difference.operator(f"d_{axis}_1"), compact)
            )
            second.append(
                _apply_components(self.finite_difference.operator(f"d_{axis}_2"), compact)
            )
        gradient = jnp.stack(gradients, axis=-2)
        laplacian = sum(second)
        thermodynamics = self.closure.evaluate(
            compact,
            gradient,
            laplacian,
            self.thermodynamic_parameters,
            electric_field=electric_field,
        )
        anchoring = None if self.anchoring is None else self.anchoring.evaluate(compact)
        weights = self.finite_difference.grid.quadrature_weights
        total = jnp.sum(weights * thermodynamics.total_energy_density)
        if anchoring is not None:
            total = total + jnp.sum(weights * anchoring.energy_density)
        successful = thermodynamics.successful & jnp.isfinite(total)
        if anchoring is not None:
            successful = successful & anchoring.successful
        return NematicEvaluation(
            thermodynamics,
            anchoring,
            gradient,
            laplacian,
            total,
            successful,
            self.plan_id,
        )

    def rate(
        self,
        compact_q: ArrayLike,
        /,
        *,
        velocity: ArrayLike | None = None,
        velocity_gradient: ArrayLike | None = None,
        electric_field: ArrayLike | None = None,
    ) -> tuple[Array, NematicEvaluation, BerisEdwardsConstitutiveFields | None]:
        compact = jnp.asarray(compact_q)
        evaluation = self.evaluate(compact, electric_field=electric_field)
        molecular = evaluation.thermodynamics.molecular_field
        if evaluation.anchoring is not None:
            molecular = molecular + evaluation.anchoring.molecular_field
        rate = self.dynamics_parameters.rotational_mobility * molecular
        constitutive = None
        if velocity_gradient is not None:
            gradient_value = jnp.asarray(velocity_gradient, dtype=compact.dtype)
            constitutive = beris_edwards_constitutive_fields(
                self.closure.basis,
                compact,
                molecular,
                gradient_value,
                evaluation.thermodynamics.distortion_stress
                + evaluation.thermodynamics.electric_stress,
                self.dynamics_parameters,
            )
            rate = rate + constitutive.alignment_term
        if velocity is not None:
            velocity_value = jnp.asarray(velocity, dtype=compact.dtype)
            spatial_dimension = len(self.finite_difference.grid.axis_names)
            if velocity_value.shape != compact.shape[:-1] + (spatial_dimension,):
                raise ValueError("velocity must match spatial grid and dimension.")
            advection = jnp.sum(
                velocity_value[..., :, None] * evaluation.compact_gradient,
                axis=-2,
            )
            rate = rate - advection
        successful = evaluation.successful & jnp.all(jnp.isfinite(rate))
        if constitutive is not None:
            successful = successful & constitutive.successful
        return jnp.where(successful, rate, jnp.nan), evaluation, constitutive

    def step(
        self,
        compact_q: ArrayLike,
        time_step: ArrayLike,
        /,
        *,
        velocity: ArrayLike | None = None,
        velocity_gradient: ArrayLike | None = None,
        electric_field: ArrayLike | None = None,
    ) -> NematicStepResult:
        incoming = jnp.asarray(compact_q)
        step = jnp.asarray(time_step, dtype=incoming.dtype)
        if step.shape != ():
            raise ValueError("time_step must be scalar.")
        rate, before, _ = self.rate(
            incoming,
            velocity=velocity,
            velocity_gradient=velocity_gradient,
            electric_field=electric_field,
        )
        candidate = incoming + step * rate
        after = self.evaluate(candidate, electric_field=electric_field)
        passive_flow = velocity is None and velocity_gradient is None
        require_energy_decay = (
            self.dynamics_parameters.activity == 0.0
            if passive_flow
            else jnp.asarray(False)
        )
        energy_scale = jnp.maximum(jnp.abs(before.total_free_energy), 1.0)
        energy_ok = (
            after.total_free_energy
            <= before.total_free_energy + self.energy_tolerance * energy_scale
        )
        successful = (
            before.successful
            & after.successful
            & jnp.isfinite(step)
            & (step > 0.0)
            & jnp.all(jnp.isfinite(candidate))
            & jnp.where(require_energy_decay, energy_ok, True)
        )
        accepted = jnp.where(successful, candidate, incoming)
        evaluation = self.evaluate(accepted, electric_field=electric_field)
        return NematicStepResult(
            accepted,
            evaluation,
            successful,
            self.plan_id,
        )


class PreparedNematicSemiImplicitStepPlan(StrictModule, NonTrainableState):
    dynamics: PreparedNematicDynamics
    time_step: Array
    elastic_solve: FDLaplacianSolvePlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: PreparedNematicDynamics,
        time_step: ArrayLike,
        /,
    ):
        if not isinstance(dynamics, PreparedNematicDynamics):
            raise TypeError("dynamics must be PreparedNematicDynamics.")
        step = jnp.asarray(time_step)
        if step.shape != () or not bool(jnp.isfinite(step) & (step > 0.0)):
            raise ValueError("time_step must be one finite positive scalar.")
        if not all(
            axis.periodic for axis in dynamics.finite_difference.grid.structured_axes
        ):
            raise ValueError("Semi-implicit nematic relaxation requires periodic axes.")
        boundaries = tuple(
            ("periodic", "periodic") for _ in dynamics.finite_difference.grid.axis_names
        )
        diagonalization = diagonalize_fd_laplacian(
            dynamics.finite_difference.grid, boundaries
        )
        scale = (
            -step
            * dynamics.dynamics_parameters.rotational_mobility
            * dynamics.thermodynamic_parameters.elastic_constant
        )
        self.dynamics = dynamics
        self.time_step = step
        self.elastic_solve = FDLaplacianSolvePlan(
            diagonalization,
            operator_scale=scale,
            diagonal_shift=1.0,
            compatibility="error",
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "nematic-semi-implicit-step",
                "dynamics": dynamics.plan_id,
                "time_step": float(step),
                "solve": self.elastic_solve.plan_id,
            }
        )

    def step(
        self,
        compact_q: ArrayLike,
        /,
        *,
        electric_field: ArrayLike | None = None,
    ) -> NematicStepResult:
        incoming = jnp.asarray(compact_q)
        before = self.dynamics.evaluate(incoming, electric_field=electric_field)
        elastic_molecular = (
            self.dynamics.thermodynamic_parameters.elastic_constant
            * before.compact_laplacian
        )
        explicit_molecular = before.thermodynamics.molecular_field - elastic_molecular
        if before.anchoring is not None:
            explicit_molecular = explicit_molecular + before.anchoring.molecular_field
        rhs = incoming + (
            self.time_step
            * self.dynamics.dynamics_parameters.rotational_mobility
            * explicit_molecular
        )
        solved_components = []
        successful = before.successful
        for component in range(incoming.shape[-1]):
            solved = self.elastic_solve.solve(rhs[..., component])
            solved_components.append(solved.value)
            successful = successful & solved.converged
        candidate = jnp.stack(solved_components, axis=-1)
        after = self.dynamics.evaluate(candidate, electric_field=electric_field)
        energy_scale = jnp.maximum(jnp.abs(before.total_free_energy), 1.0)
        successful = (
            successful
            & after.successful
            & (
                after.total_free_energy
                <= before.total_free_energy
                + self.dynamics.energy_tolerance * energy_scale
            )
        )
        accepted = jnp.where(successful, candidate, incoming)
        evaluation = self.dynamics.evaluate(accepted, electric_field=electric_field)
        return NematicStepResult(
            accepted,
            evaluation,
            successful,
            self.plan_id,
        )


class MACNematicCouplingEvaluation(StrictModule):
    """Passive nematic rate and exactly dual MAC stress forcing."""

    compact_rate: Array
    face_body_force: FaceVelocity
    cell_body_force: Array
    cell_velocity: Array
    velocity_gradient: Array
    stress: Array
    fluid_work: Array
    nematic_stress_work: Array
    work_residual: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class MACNematicState(StrictModule):
    """One atomically committed Q-tensor and staggered velocity state."""

    compact_q: Array
    face_velocity: FaceVelocity
    accepted_steps: Array
    plan_id: str = eqx.field(static=True)


class MACNematicStepResult(StrictModule):
    candidate_state: MACNematicState
    accepted_state: MACNematicState
    coupling: MACNematicCouplingEvaluation
    energy_before: Array
    candidate_energy: Array
    accepted_energy: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class MACNematicCouplingPlan(StrictModule, NonTrainableState):
    """Periodic passive Beris--Edwards/MAC composition with atomic commit."""

    dynamics: PreparedNematicDynamics
    operators: PreparedMACOperators
    density: float = eqx.field(static=True)
    work_tolerance: float = eqx.field(static=True)
    maximum_cells: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: PreparedNematicDynamics,
        operators: PreparedMACOperators,
        /,
        *,
        density: float = 1.0,
        work_tolerance: float = 1.0e-10,
        maximum_cells: int = 1_000_000,
    ):
        if not isinstance(dynamics, PreparedNematicDynamics):
            raise TypeError("dynamics must be PreparedNematicDynamics.")
        if not isinstance(operators, PreparedMACOperators):
            raise TypeError("operators must be PreparedMACOperators.")
        spatial_dimension = len(dynamics.finite_difference.grid.axis_names)
        if spatial_dimension != dynamics.closure.basis.orientation_dimension:
            raise ValueError(
                "Two-way MAC nematic coupling requires matching spatial/orientation dimensions."
            )
        if operators.discretization.grid.prepared_id != (
            dynamics.finite_difference.grid.prepared_id
        ):
            raise ValueError(
                "MAC and nematic discretizations must share one prepared grid."
            )
        if not all(
            axis.periodic for axis in operators.discretization.grid.structured_axes
        ):
            raise ValueError(
                "The passive MAC nematic profile requires periodic boundary identities."
            )
        if float(dynamics.dynamics_parameters.activity) != 0.0:
            raise ValueError(
                "Atomic MAC nematic coupling is passive and rejects activity."
            )
        density_ = float(density)
        tolerance = float(work_tolerance)
        capacity = int(maximum_cells)
        cell_count = int(np.prod(operators.discretization.cell_shape))
        if (
            not np.isfinite(density_)
            or density_ <= 0.0
            or not np.isfinite(tolerance)
            or tolerance < 0.0
            or capacity <= 0
        ):
            raise ValueError("MAC nematic density, tolerance, and capacity are invalid.")
        if cell_count > capacity:
            raise ValueError("MAC nematic cell count exceeds maximum_cells.")
        self.dynamics = dynamics
        self.operators = operators
        self.density = density_
        self.work_tolerance = tolerance
        self.maximum_cells = capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "passive-mac-nematic-coupling",
                "dynamics": dynamics.plan_id,
                "operators": operators.prepared_id,
                "density": density_,
                "work_tolerance": tolerance,
                "maximum_cells": capacity,
            }
        )

    def _cell_velocity(self, face_velocity: FaceVelocity, /) -> Array:
        values = self.operators.validate_velocity(face_velocity)
        components = tuple(
            0.5 * (value + jnp.roll(value, -1, axis=axis))
            for axis, value in enumerate(values)
        )
        return jnp.stack(components, axis=-1)

    def _velocity_gradient(self, face_velocity: FaceVelocity, /) -> Array:
        velocity = self._cell_velocity(face_velocity)
        rows = []
        for component in range(velocity.shape[-1]):
            rows.append(
                jnp.stack(
                    tuple(
                        self.dynamics.finite_difference.operator(f"d_{axis}_1")(
                            velocity[..., component]
                        )
                        for axis in self.dynamics.finite_difference.grid.axis_names
                    ),
                    axis=-1,
                )
            )
        return jnp.stack(tuple(rows), axis=-2)

    def _dual_stress_force(self, stress: Array, /) -> FaceVelocity:
        zero = tuple(
            jnp.zeros(layout.shape, dtype=stress.dtype)
            for layout in self.operators.discretization.face_layouts
        )

        def gradient_action(velocity):
            return self._velocity_gradient(velocity)

        _, pullback = jax.vjp(gradient_action, zero)
        volumes = self.operators.discretization.cell_volumes.astype(stress.dtype)
        (covector,) = pullback(-volumes[..., None, None] * stress)
        return tuple(
            value / measure.astype(stress.dtype)
            for value, measure in zip(
                covector, self.operators.face_dual_measures, strict=True
            )
        )

    def evaluate(
        self,
        compact_q: ArrayLike,
        face_velocity: FaceVelocity,
        /,
        *,
        electric_field: ArrayLike | None = None,
    ) -> MACNematicCouplingEvaluation:
        velocity = self.operators.validate_velocity(face_velocity)
        cell_velocity = self._cell_velocity(velocity)
        gradient = self._velocity_gradient(velocity)
        rate, evaluation, constitutive = self.dynamics.rate(
            compact_q,
            velocity=cell_velocity,
            velocity_gradient=gradient,
            electric_field=electric_field,
        )
        if constitutive is None:
            raise RuntimeError("Nematic constitutive evaluation was not produced.")
        stress = constitutive.passive_stress
        face_force = self._dual_stress_force(stress)
        cell_force = self._cell_velocity(face_force)
        fluid_work = sum(
            jnp.sum(measure.astype(stress.dtype) * speed * force)
            for measure, speed, force in zip(
                self.operators.face_dual_measures,
                velocity,
                face_force,
                strict=True,
            )
        )
        volumes = self.operators.discretization.cell_volumes.astype(stress.dtype)
        nematic_work = jnp.sum(volumes * constitutive.passive_power)
        work_residual = jnp.abs(fluid_work + nematic_work)
        work_scale = jnp.maximum(
            1.0, jnp.maximum(jnp.abs(fluid_work), jnp.abs(nematic_work))
        )
        successful = (
            evaluation.successful
            & constitutive.successful
            & self.operators.report.passed
            & jnp.all(jnp.isfinite(rate))
            & jnp.all(
                jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in face_force))
            )
            & jnp.all(jnp.isfinite(cell_force))
            & jnp.isfinite(work_residual)
            & (work_residual <= self.work_tolerance * work_scale)
        )
        return MACNematicCouplingEvaluation(
            rate,
            face_force,
            cell_force,
            cell_velocity,
            gradient,
            stress,
            fluid_work,
            nematic_work,
            work_residual,
            successful,
            self.plan_id,
        )

    def _kinetic_energy(self, velocity: FaceVelocity, /) -> Array:
        values = self.operators.validate_velocity(velocity)
        return (
            0.5
            * self.density
            * sum(
                jnp.sum(measure.astype(value.dtype) * value * value)
                for measure, value in zip(
                    self.operators.face_dual_measures, values, strict=True
                )
            )
        )

    def initialize_state(
        self, compact_q: ArrayLike, face_velocity: FaceVelocity, /
    ) -> MACNematicState:
        compact = jnp.asarray(compact_q)
        velocity = self.operators.validate_velocity(face_velocity)
        evaluated = self.evaluate(compact, velocity)
        checked = eqx.error_if(
            compact,
            ~evaluated.successful,
            "Initial MAC nematic state failed passive coupling evidence.",
        )
        return MACNematicState(
            checked,
            velocity,
            jnp.zeros((), dtype=jnp.int32),
            self.plan_id,
        )

    def step(
        self,
        state: MACNematicState,
        time_step: ArrayLike,
        /,
        *,
        electric_field: ArrayLike | None = None,
    ) -> MACNematicStepResult:
        if not isinstance(state, MACNematicState):
            raise TypeError("state must be MACNematicState.")
        if state.plan_id != self.plan_id:
            raise ValueError("MAC nematic state belongs to another coupling plan.")
        step = jnp.asarray(time_step, dtype=state.compact_q.dtype)
        if step.shape != ():
            raise ValueError("time_step must be scalar.")
        step_valid = jnp.isfinite(step) & (step > 0.0)
        safe_step = jnp.where(step_valid, step, 0.0)
        before_q = self.dynamics.evaluate(state.compact_q, electric_field=electric_field)
        coupling = self.evaluate(
            state.compact_q,
            state.face_velocity,
            electric_field=electric_field,
        )
        candidate_q = state.compact_q + safe_step * coupling.compact_rate
        candidate_velocity = tuple(
            velocity + safe_step * force / self.density
            for velocity, force in zip(
                state.face_velocity, coupling.face_body_force, strict=True
            )
        )
        after_q = self.dynamics.evaluate(candidate_q, electric_field=electric_field)
        energy_before = before_q.total_free_energy + self._kinetic_energy(
            state.face_velocity
        )
        candidate_energy = after_q.total_free_energy + self._kinetic_energy(
            candidate_velocity
        )
        energy_scale = jnp.maximum(jnp.abs(energy_before), 1.0)
        energy_ok = candidate_energy <= (
            energy_before + self.dynamics.energy_tolerance * energy_scale
        )
        successful = (
            step_valid
            & coupling.successful
            & after_q.successful
            & jnp.isfinite(candidate_energy)
            & energy_ok
        )
        candidate = MACNematicState(
            candidate_q,
            candidate_velocity,
            state.accepted_steps + 1,
            self.plan_id,
        )
        accepted = MACNematicState(
            jnp.where(successful, candidate_q, state.compact_q),
            tuple(
                jnp.where(successful, trial, incoming)
                for trial, incoming in zip(
                    candidate_velocity, state.face_velocity, strict=True
                )
            ),
            state.accepted_steps + successful.astype(jnp.int32),
            self.plan_id,
        )
        return MACNematicStepResult(
            candidate,
            accepted,
            coupling,
            energy_before,
            candidate_energy,
            jnp.where(successful, candidate_energy, energy_before),
            successful,
            self.plan_id,
        )


def _apply_components(operator, field):
    return jnp.stack(
        [operator(field[..., component]) for component in range(field.shape[-1])],
        axis=-1,
    )


__all__ = [
    "MACNematicCouplingEvaluation",
    "MACNematicCouplingPlan",
    "MACNematicState",
    "MACNematicStepResult",
    "NematicEvaluation",
    "NematicStepResult",
    "PreparedNematicDynamics",
    "PreparedNematicSemiImplicitStepPlan",
]
