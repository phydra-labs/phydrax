#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import final, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...nonlinear import FixedPointIteration, FixedPointProblem, NonlinearTermination
from ...rheology import GeneralizedNewtonianLaw, MuILaw, MuIParameters
from ..finite_volume import (
    FaceVelocity,
    MACBoundaryStageData,
    MACFreeSurfaceViscousMeasures,
)
from ._method import FLIPMethodPlan
from ._types import FLIPViscousEvidence


if TYPE_CHECKING:
    from ...solver._mac_variational_viscosity import (
        MACVariationalViscosityPlan,
        MACVariationalViscosityResult,
    )


def _norm(values: FaceVelocity) -> Array:
    return jnp.sqrt(sum(jnp.sum(value**2) for value in values))


@final
class _ViscousMap(StrictModule, NonTrainableState):
    plan: MACVariationalViscosityPlan
    law: GeneralizedNewtonianLaw | MuILaw = eqx.field(static=True)
    parameters: MuIParameters | None
    measures: MACFreeSurfaceViscousMeasures
    forced: FaceVelocity
    pressure: Array
    liquid: Array
    step: Array
    stage: MACBoundaryStageData
    scale: Array

    def closure(self, velocity: FaceVelocity) -> tuple[Array, Array, Array, Array, Array]:
        rate = self.plan.action.cell_shear_rate(velocity, self.stage)
        if isinstance(self.law, MuILaw):
            if self.parameters is None:
                raise ValueError("MuILaw requires mu_i_parameters.")
            result = self.law.evaluate(rate, self.pressure, self.parameters)
            viscosity = result.apparent_viscosity_pa_s
            valid = result.successful & result.viscosity_defined & result.finite
            margin = result.hadamard_margin
            applicable = result.hadamard_applicable & self.liquid
        else:
            result = self.law.viscosity(rate)
            viscosity = result.viscosity_pa_s
            valid = result.successful & result.viscosity_defined & result.finite
            margin = jnp.zeros_like(rate)
            applicable = jnp.zeros_like(self.liquid)
        successful = jnp.all(valid | ~self.liquid)
        return (
            rate,
            jnp.where(self.liquid, viscosity, 0.0),
            successful,
            margin,
            applicable,
        )

    def linear(self, viscosity: Array) -> MACVariationalViscosityResult:
        # An invalid closure is rejected by the outer transaction. Finite
        # diagnostic coefficients prevent operator-domain exceptions without
        # admitting the diagnostic solve as physical motion.
        coefficient = jnp.where(
            jnp.isfinite(viscosity) & (viscosity >= 0.0), viscosity, 0.0
        )
        return self.plan.solve(
            self.forced,
            self.measures.face_density,
            coefficient * self.measures.cell_fraction,
            self.step,
            self.stage,
        )

    def __call__(
        self, scaled_velocity: FaceVelocity, args: None, /
    ) -> tuple[FaceVelocity, dict[str, Array]]:
        del args
        velocity = tuple(value * self.scale for value in scaled_velocity)
        _, viscosity, closure_success, _, _ = self.closure(velocity)
        linear = self.linear(viscosity)
        successful = closure_success & linear.successful
        mapped = tuple(
            jnp.where(successful, value / self.scale, jnp.nan)
            for value in linear.candidate_velocity
        )
        return mapped, {
            "linear_iterations": linear.iterations,
            "linear_failures": (~linear.successful).astype(jnp.int32),
            "closure_failures": (~closure_success).astype(jnp.int32),
        }


def viscous_flip_stage(
    plan: MACVariationalViscosityPlan,
    law: GeneralizedNewtonianLaw | MuILaw,
    parameters: MuIParameters | None,
    method: FLIPMethodPlan,
    measures: MACFreeSurfaceViscousMeasures,
    forced: FaceVelocity,
    pressure: Array,
    liquid: Array,
    step: Array,
    stage: MACBoundaryStageData,
    pressure_admitted: Array,
    /,
    *,
    predictor_terms: tuple[FaceVelocity, ...] = (),
) -> tuple[FaceVelocity, FLIPViscousEvidence]:
    """Bounded Picard solve followed by true nonlinear momentum certification.

    The pressure is fixed throughout the iteration. Coordinates and momentum
    evidence retain the norms of uncombined predictor terms when provided.
    This is a backward-error scale: balanced gravity and pressure must not
    normalize roundoff by their nearly-zero difference. No rate or pressure
    is altered. A converged update alone cannot admit a failed closure or solve.
    """
    scale = _norm(forced)
    if predictor_terms:
        scale = scale + sum(_norm(term) for term in predictor_terms)
    scale = jnp.where(scale > 0.0, scale, 1.0)
    mapping = _ViscousMap(
        plan, law, parameters, measures, forced, pressure, liquid, step, stage, scale
    )
    # The inverse viscous operator can hide a substantial momentum defect in
    # a small velocity correction. Reserve three orders of accuracy for the
    # inner update certificate; the true momentum residual still owns admission.
    update_tolerance = 1.0e-3 * method.viscous_tolerance
    nonlinear = FixedPointIteration(damping=method.picard_damping).solve(
        FixedPointProblem(mapping, problem_id=plan.plan_id, evaluation_work=True),
        tuple(value / scale for value in forced),
        termination=NonlinearTermination(
            absolute_residual=update_tolerance,
            relative_residual=update_tolerance,
            absolute_step=0.0,
            relative_step=0.0,
            maximum_steps=method.maximum_picard_iterations,
        ),
    )
    candidate = tuple(value * scale for value in nonlinear.state)
    finite = jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(v)) for v in candidate)))
    diagnostic_velocity = tuple(
        jnp.where(finite, value, original)
        for value, original in zip(candidate, forced, strict=True)
    )
    rate, viscosity, closure_success, margin, applicable = mapping.closure(
        diagnostic_velocity
    )
    linear = mapping.linear(viscosity)
    coefficient = jnp.where(jnp.isfinite(viscosity) & (viscosity >= 0.0), viscosity, 0.0)
    stress = plan.action.affine_positive_action(
        diagnostic_velocity, coefficient * measures.cell_fraction, stage
    )
    residual = tuple(
        measure * jnp.where(free, mass * (value - original) + step * action, 0.0)
        for measure, free, mass, value, original, action in zip(
            plan.measure,
            plan.free,
            measures.face_density,
            diagnostic_velocity,
            forced,
            stress,
            strict=True,
        )
    )
    rhs = tuple(
        measure * jnp.where(free, mass * value, 0.0)
        for measure, free, mass, value in zip(
            plan.measure, plan.free, measures.face_density, forced, strict=True
        )
    )
    momentum = tuple(
        measure * jnp.where(free, mass * value + step * action, 0.0)
        for measure, free, mass, value, action in zip(
            plan.measure,
            plan.free,
            measures.face_density,
            diagnostic_velocity,
            stress,
            strict=True,
        )
    )
    residual_norm = jnp.where(closure_success & finite, _norm(residual), jnp.inf)
    residual_scale = _norm(rhs) + _norm(momentum)
    if predictor_terms:
        for term in predictor_terms:
            weighted = tuple(
                measure * jnp.where(free, mass * value, 0.0)
                for measure, free, mass, value in zip(
                    plan.measure, plan.free, measures.face_density, term, strict=True
                )
            )
            residual_scale = residual_scale + _norm(weighted)
    relative = jnp.where(
        residual_scale > 0.0,
        residual_norm / jnp.where(residual_scale > 0.0, residual_scale, 1.0),
        jnp.where(residual_norm == 0.0, 0.0, jnp.inf),
    )
    converged = nonlinear.successful & (relative <= method.viscous_tolerance)
    successful = (
        finite
        & converged
        & closure_success
        & pressure_admitted
        & linear.successful
        & measures.successful
    )
    accepted = tuple(
        jnp.where(successful, value, original)
        for value, original in zip(candidate, forced, strict=True)
    )
    return accepted, FLIPViscousEvidence(
        rate,
        viscosity,
        pressure,
        margin,
        applicable,
        pressure_admitted,
        closure_success,
        residual_norm,
        relative,
        nonlinear.diagnostics.iterations,
        converged,
        successful,
        nonlinear,
        linear,
        plan.plan_id,
    )
