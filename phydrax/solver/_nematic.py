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
from ..discretization.finite_difference import (
    diagonalize_fd_laplacian,
    FDLaplacianSolvePlan,
    PreparedFiniteDifferenceDiscretization,
)
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
    cell_body_force: Array
    stress: Array
    passive_power: Array
    active_power: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class MACNematicCouplingPlan(StrictModule, NonTrainableState):
    dynamics: PreparedNematicDynamics
    plan_id: str = eqx.field(static=True)

    def __init__(self, dynamics: PreparedNematicDynamics, /):
        if not isinstance(dynamics, PreparedNematicDynamics):
            raise TypeError("dynamics must be PreparedNematicDynamics.")
        spatial_dimension = len(dynamics.finite_difference.grid.axis_names)
        if spatial_dimension != dynamics.closure.basis.orientation_dimension:
            raise ValueError(
                "Two-way MAC nematic coupling requires matching spatial/orientation dimensions."
            )
        self.dynamics = dynamics
        self.plan_id = canonical_fingerprint(
            {"kind": "mac-nematic-coupling", "dynamics": dynamics.plan_id}
        )

    def evaluate(
        self,
        compact_q: ArrayLike,
        velocity_gradient: ArrayLike,
        /,
        *,
        electric_field: ArrayLike | None = None,
    ) -> MACNematicCouplingEvaluation:
        _, evaluation, constitutive = self.dynamics.rate(
            compact_q,
            velocity_gradient=velocity_gradient,
            electric_field=electric_field,
        )
        if constitutive is None:
            raise RuntimeError("Nematic constitutive evaluation was not produced.")
        stress = constitutive.total_stress
        body_components = []
        for component in range(stress.shape[-2]):
            divergence = jnp.zeros(stress.shape[:-2], dtype=stress.dtype)
            for axis_index, axis in enumerate(
                self.dynamics.finite_difference.grid.axis_names
            ):
                divergence = divergence + self.dynamics.finite_difference.operator(
                    f"d_{axis}_1"
                )(stress[..., component, axis_index])
            body_components.append(divergence)
        body_force = jnp.stack(body_components, axis=-1)
        successful = (
            evaluation.successful
            & constitutive.successful
            & jnp.all(jnp.isfinite(body_force))
        )
        return MACNematicCouplingEvaluation(
            body_force,
            stress,
            constitutive.passive_power,
            constitutive.active_power,
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
    "NematicEvaluation",
    "NematicStepResult",
    "PreparedNematicDynamics",
    "PreparedNematicSemiImplicitStepPlan",
]
