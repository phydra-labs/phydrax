#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSystem,
    solve as solve_linear,
    svd as svd_api,
)
from ...nonlinear import (
    AndersonAcceleration,
    FixedPointIteration,
    FixedPointProblem,
    implicit_root_result,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
)
from ._closures import evaluate_prism_closure, PRISMClosureEvaluation, PRISMClosurePlan
from ._mixture import (
    SequenceFormFactorPlan,
    SiteMixturePlan,
    SitePairPotentialPlan,
    TabulatedFormFactorPlan,
)
from ._radial import PreparedIsotropicRadialTransform


FormFactorPlan = SequenceFormFactorPlan | TabulatedFormFactorPlan


class PRISMPlan(StrictModule, NonTrainableState):
    closure: PRISMClosurePlan
    damping: float = eqx.field(static=True)
    anderson_history: int = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    maximum_condition: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        closure: PRISMClosurePlan,
        /,
        *,
        damping: float = 0.2,
        anderson_history: int = 8,
        absolute_tolerance: float = 1.0e-8,
        relative_tolerance: float = 1.0e-8,
        maximum_iterations: int = 200,
        maximum_condition: float = 1.0e12,
    ):
        damping_ = float(damping)
        history = int(anderson_history)
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        iterations = int(maximum_iterations)
        condition = float(maximum_condition)
        if not isinstance(closure, PRISMClosurePlan):
            raise TypeError("closure must be PRISMClosurePlan.")
        if (
            not math.isfinite(damping_)
            or not 0.0 < damping_ <= 1.0
            or history <= 0
            or not math.isfinite(absolute)
            or absolute < 0.0
            or not math.isfinite(relative)
            or relative < 0.0
            or iterations <= 0
            or not math.isfinite(condition)
            or condition <= 1.0
        ):
            raise ValueError("PRISM solver controls are invalid.")
        self.closure = closure
        self.damping = damping_
        self.anderson_history = history
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.maximum_iterations = iterations
        self.maximum_condition = condition
        self.plan_id = canonical_fingerprint(
            {
                "kind": "prism-plan",
                "closure": closure.plan_id,
                "damping": damping_,
                "anderson_history": history,
                "absolute_tolerance": absolute,
                "relative_tolerance": relative,
                "maximum_iterations": iterations,
                "maximum_condition": condition,
            }
        )

    def prepare(
        self,
        transform: PreparedIsotropicRadialTransform,
        mixture: SiteMixturePlan,
        form_factor: FormFactorPlan,
        potential: SitePairPotentialPlan,
        /,
    ) -> "PreparedPRISM":
        return PreparedPRISM(self, transform, mixture, form_factor, potential)


class PRISMOZEvaluation(StrictModule):
    total_correlation_wave: Array
    direct_correlation_wave: Array
    structure_factor: Array
    linear_status: Array
    linear_residual: Array
    condition_estimate: Array
    successful: Array


class PRISMEvaluation(StrictModule):
    gamma: Array
    closure: PRISMClosureEvaluation
    total_correlation_radial: Array
    oz: PRISMOZEvaluation
    residual: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class PreparedPRISM(StrictModule, NonTrainableState):
    plan: PRISMPlan
    transform: PreparedIsotropicRadialTransform
    mixture: SiteMixturePlan
    form_factor: FormFactorPlan
    potential: SitePairPotentialPlan
    omega: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: PRISMPlan,
        transform: PreparedIsotropicRadialTransform,
        mixture: SiteMixturePlan,
        form_factor: FormFactorPlan,
        potential: SitePairPotentialPlan,
        /,
    ):
        if not isinstance(plan, PRISMPlan):
            raise TypeError("plan must be PRISMPlan.")
        if not isinstance(transform, PreparedIsotropicRadialTransform):
            raise TypeError("transform must be PreparedIsotropicRadialTransform.")
        if not isinstance(mixture, SiteMixturePlan):
            raise TypeError("mixture must be SiteMixturePlan.")
        if not isinstance(form_factor, (SequenceFormFactorPlan, TabulatedFormFactorPlan)):
            raise TypeError("form_factor must be a supported form-factor plan.")
        if not isinstance(potential, SitePairPotentialPlan):
            raise TypeError("potential must be SitePairPotentialPlan.")
        if (
            mixture.site_count != form_factor.site_count
            or mixture.site_count != potential.site_count
        ):
            raise ValueError(
                "PRISM mixture, form factor, and potential site counts differ."
            )
        if potential.radii.shape != transform.radii.shape or not np.array_equal(
            np.asarray(potential.radii), np.asarray(transform.radii)
        ):
            raise ValueError("PRISM potential requires the exact prepared radial grid.")
        if (
            plan.closure.hard_core_diameters is not None
            and plan.closure.hard_core_diameters.shape
            != (mixture.site_count, mixture.site_count)
        ):
            raise ValueError(
                "PRISM closure hard-core dimensions differ from the mixture."
            )
        omega = form_factor.evaluate(transform.wave_numbers)
        if omega.shape != (
            transform.plan.count,
            mixture.site_count,
            mixture.site_count,
        ):
            raise ValueError("Intramolecular form factors have an invalid shape.")
        self.plan = plan
        self.transform = transform
        self.mixture = mixture
        self.form_factor = form_factor
        self.potential = potential
        self.omega = jnp.asarray(omega, dtype=transform.radii.dtype)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-prism",
                "plan": plan.plan_id,
                "transform": transform.prepared_id,
                "mixture": mixture.plan_id,
                "form_factor": form_factor.plan_id,
                "potential": potential.plan_id,
            }
        )

    def _oz(
        self,
        direct_correlation_radial: Array,
        number_densities: Array,
        /,
    ) -> PRISMOZEvaluation:
        direct_wave = self.transform.forward(direct_correlation_radial)
        direct_by_wave = jnp.moveaxis(direct_wave, -1, 0)
        density = jnp.diag(number_densities).astype(direct_by_wave.dtype)
        identity = jnp.eye(self.mixture.site_count, dtype=direct_by_wave.dtype)
        left = identity[None, :, :] - self.omega @ direct_by_wave @ density
        right = self.omega @ direct_by_wave @ self.omega
        operator = DenseLinearOperator(
            left,
            operator_id=f"{self.prepared_id}:ornstein-zernike",
        )
        linear = solve_linear(
            LinearSystem(operator),
            right,
            policy=LinearSolvePolicy(DenseLU()),
        )

        def condition_one(matrix):
            condition_operator = DenseLinearOperator(
                matrix,
                operator_id=f"{self.prepared_id}:oz-conditioning-operator",
            )
            spectrum = svd_api.svd(
                svd_api.SVDProblem(
                    condition_operator,
                    problem_id=f"{self.prepared_id}:oz-conditioning",
                ),
                policy=svd_api.SVDSolvePolicy(count=self.mixture.site_count),
            )
            singular_values = spectrum.singular_values
            full_rank = spectrum.numerical_rank == self.mixture.site_count
            condition = jnp.where(
                full_rank & (singular_values[-1] > 0.0),
                singular_values[0] / singular_values[-1],
                jnp.asarray(jnp.inf, dtype=singular_values.dtype),
            )
            return condition, spectrum.successful

        condition, spectrum_successful = jax.vmap(condition_one)(left)
        total_wave = jnp.asarray(linear.value)
        square_root_density = jnp.diag(jnp.sqrt(number_densities)).astype(
            total_wave.dtype
        )
        structure = self.omega + square_root_density @ total_wave @ square_root_density
        condition = jnp.asarray(condition)
        successful = (
            jnp.all(linear.successful)
            & jnp.all(jnp.isfinite(total_wave))
            & jnp.all(jnp.isfinite(structure))
            & jnp.all(spectrum_successful)
            & jnp.all(condition <= self.plan.maximum_condition)
            & jnp.all(jnp.isfinite(number_densities))
            & jnp.all(number_densities > 0.0)
        )
        return PRISMOZEvaluation(
            total_wave,
            direct_by_wave,
            structure,
            linear.status,
            linear.diagnostics.relative_residual,
            condition,
            successful,
        )

    def evaluate(
        self,
        gamma: ArrayLike,
        /,
        *,
        number_densities: ArrayLike | None = None,
        beta_potential: ArrayLike | None = None,
    ) -> PRISMEvaluation:
        indirect = jnp.asarray(gamma, dtype=self.transform.radii.dtype)
        expected = (
            self.mixture.site_count,
            self.mixture.site_count,
            self.transform.plan.count,
        )
        if indirect.shape != expected:
            raise ValueError(f"gamma must have shape {expected}.")
        densities = (
            self.mixture.number_densities
            if number_densities is None
            else jnp.asarray(number_densities, dtype=indirect.dtype)
        )
        potential = (
            self.potential.beta_potential
            if beta_potential is None
            else jnp.asarray(beta_potential, dtype=indirect.dtype)
        )
        if densities.shape != (self.mixture.site_count,) or potential.shape != expected:
            raise ValueError("Parameterized PRISM density or potential shape is invalid.")
        closure = evaluate_prism_closure(
            self.plan.closure,
            self.transform.radii,
            indirect,
            potential,
        )
        oz = self._oz(closure.direct_correlation, densities)
        total_radial = self.transform.inverse(
            jnp.moveaxis(oz.total_correlation_wave, 0, -1)
        )
        mapped = total_radial - closure.direct_correlation
        residual = indirect - mapped
        symmetric_structure = 0.5 * (
            oz.structure_factor + jnp.swapaxes(oz.structure_factor, -1, -2)
        )
        minimum_structure_eigenvalue = jnp.min(jnp.linalg.eigvalsh(symmetric_structure))
        successful = (
            closure.successful
            & oz.successful
            & jnp.all(jnp.isfinite(total_radial))
            & jnp.all(jnp.isfinite(residual))
            & jnp.all(closure.radial_distribution >= -self.plan.absolute_tolerance)
            & (minimum_structure_eigenvalue >= -self.plan.absolute_tolerance)
        )
        return PRISMEvaluation(
            indirect,
            closure,
            total_radial,
            oz,
            residual,
            successful,
            self.prepared_id,
        )

    def fixed_point_problem(self, /) -> FixedPointProblem:
        def mapping(gamma, _):
            evaluation = self.evaluate(gamma)
            iteration_successful = (
                evaluation.closure.successful
                & evaluation.oz.successful
                & jnp.all(jnp.isfinite(evaluation.residual))
            )
            mapped = gamma - evaluation.residual
            return jnp.where(iteration_successful, mapped, jnp.nan)

        return FixedPointProblem(mapping, problem_id=f"{self.prepared_id}:fixed-point")

    def root_problem(self, /) -> NonlinearSystemProblem:
        def residual(gamma, _):
            evaluation = self.evaluate(gamma)
            iteration_successful = (
                evaluation.closure.successful
                & evaluation.oz.successful
                & jnp.all(jnp.isfinite(evaluation.residual))
            )
            return jnp.where(iteration_successful, evaluation.residual, jnp.nan)

        return NonlinearSystemProblem(residual, problem_id=f"{self.prepared_id}:root")

    def parameterized_root_problem(self, /) -> NonlinearSystemProblem:
        def residual(gamma, args):
            densities, potential = args
            evaluation = self.evaluate(
                gamma,
                number_densities=densities,
                beta_potential=potential,
            )
            iteration_successful = (
                evaluation.closure.successful
                & evaluation.oz.successful
                & jnp.all(jnp.isfinite(evaluation.residual))
            )
            return jnp.where(iteration_successful, evaluation.residual, jnp.nan)

        return NonlinearSystemProblem(
            residual, problem_id=f"{self.prepared_id}:parameterized-root"
        )


class PRISMResult(StrictModule):
    evaluation: PRISMEvaluation
    nonlinear: NonlinearResult
    successful: Array
    prepared_id: str = eqx.field(static=True)


def solve_prism(
    prepared: PreparedPRISM,
    initial_gamma: ArrayLike | None = None,
    /,
) -> PRISMResult:
    if not isinstance(prepared, PreparedPRISM):
        raise TypeError("prepared must be PreparedPRISM.")
    shape = (
        prepared.mixture.site_count,
        prepared.mixture.site_count,
        prepared.transform.plan.count,
    )
    initial = (
        jnp.zeros(shape, dtype=prepared.transform.radii.dtype)
        if initial_gamma is None
        else jnp.asarray(initial_gamma, dtype=prepared.transform.radii.dtype)
    )
    if initial.shape != shape:
        raise ValueError(f"initial_gamma must have shape {shape}.")
    method = FixedPointIteration(
        damping=prepared.plan.damping,
        acceleration=AndersonAcceleration(history=prepared.plan.anderson_history),
    )
    nonlinear = method.solve(
        prepared.fixed_point_problem(),
        initial,
        termination=NonlinearTermination(
            absolute_residual=prepared.plan.absolute_tolerance,
            relative_residual=prepared.plan.relative_tolerance,
            maximum_steps=prepared.plan.maximum_iterations,
        ),
    )
    evaluation = prepared.evaluate(nonlinear.state)
    successful = nonlinear.successful & evaluation.successful
    return PRISMResult(evaluation, nonlinear, successful, prepared.prepared_id)


def solve_prism_implicit(
    prepared: PreparedPRISM,
    initial_gamma: ArrayLike,
    /,
    *,
    number_densities: ArrayLike | None = None,
    beta_potential: ArrayLike | None = None,
    method: Any = None,
) -> PRISMResult:
    """Solve one certified smooth PRISM branch with implicit root derivatives."""

    if not isinstance(prepared, PreparedPRISM):
        raise TypeError("prepared must be PreparedPRISM.")
    initial = jnp.asarray(initial_gamma, dtype=prepared.transform.radii.dtype)
    densities = (
        prepared.mixture.number_densities
        if number_densities is None
        else jnp.asarray(number_densities, dtype=initial.dtype)
    )
    potential = (
        prepared.potential.beta_potential
        if beta_potential is None
        else jnp.asarray(beta_potential, dtype=initial.dtype)
    )
    nonlinear = implicit_root_result(
        prepared.parameterized_root_problem(),
        initial,
        method=method,
        termination=NonlinearTermination(
            absolute_residual=prepared.plan.absolute_tolerance,
            relative_residual=prepared.plan.relative_tolerance,
            maximum_steps=prepared.plan.maximum_iterations,
        ),
        args=(densities, potential),
    )
    evaluation = prepared.evaluate(
        nonlinear.state,
        number_densities=densities,
        beta_potential=potential,
    )
    successful = nonlinear.successful & evaluation.successful
    return PRISMResult(evaluation, nonlinear, successful, prepared.prepared_id)


__all__ = [
    "PRISMEvaluation",
    "PRISMOZEvaluation",
    "PRISMPlan",
    "PRISMResult",
    "PreparedPRISM",
    "solve_prism",
    "solve_prism_implicit",
]
