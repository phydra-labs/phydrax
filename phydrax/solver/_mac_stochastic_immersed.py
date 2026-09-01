#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    AbstractLinearOperator,
    AbstractVectorSpace,
    FunctionLinearOperator,
    matrix_function_action,
    MatrixFunctionPolicy,
    OperatorProperties,
)


StochasticDifferentiationPolicy = Literal["pathwise", "weak"]
MobilityProvider = Callable[[Array], AbstractLinearOperator]


def _metric_white(
    space: AbstractVectorSpace,
    coordinates: Array,
    policy: MatrixFunctionPolicy,
    /,
):
    pairing = FunctionLinearOperator(
        space.riesz,
        source=space,
        target=space,
        transpose_action=space.riesz,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        ),
        operator_id=f"metric-pairing/{space.space_id}",
    )
    return matrix_function_action(
        pairing,
        space.unflatten(coordinates),
        kind="inverse-sqrt",
        policy=policy,
    )


class StochasticReplayKey(StrictModule, NonTrainableState):
    seed: Array
    accepted_step: Array
    stage: Array
    sample: Array

    def key(self, /) -> Array:
        key = jax.random.PRNGKey(self.seed.astype(jnp.uint32))
        key = jax.random.fold_in(key, self.accepted_step.astype(jnp.uint32))
        key = jax.random.fold_in(key, self.stage.astype(jnp.uint32))
        return jax.random.fold_in(key, self.sample.astype(jnp.uint32))


class FluctuationDissipationReport(StrictModule):
    dissipative_power: Array
    thermal_injection: Array
    balance_residual: Array
    covariance_action_residual: Array
    finite: Array
    passed: Array


class MACFluctuatingHydrodynamicsResult(StrictModule):
    stochastic_momentum: object
    replay_key: StochasticReplayKey
    report: FluctuationDissipationReport
    finite: Array
    differentiation_certified: Array
    plan_id: str = eqx.field(static=True)


class MACDiscreteStochasticStressPlan(StrictModule, NonTrainableState):
    """Discrete stress-divergence factor paired with its viscous dissipation."""

    stress_divergence: AbstractLinearOperator
    dissipation: AbstractLinearOperator
    stress_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        stress_divergence: AbstractLinearOperator,
        dissipation: AbstractLinearOperator,
        /,
        *,
        stress_id: str,
    ):
        if not stress_divergence.target.compatible(dissipation.source) or not (
            dissipation.source.compatible(dissipation.target)
        ):
            raise ValueError(
                "Stochastic stress divergence and dissipation spaces differ."
            )
        identifier = str(stress_id)
        if not identifier:
            raise ValueError("stress_id must be nonempty.")
        self.stress_divergence = stress_divergence
        self.dissipation = dissipation
        self.stress_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-discrete-stochastic-stress",
                "stress_divergence": stress_divergence.operator_id,
                "dissipation": dissipation.operator_id,
                "stress": identifier,
            }
        )

    def thermalize(
        self,
        /,
        *,
        temperature: float,
        boltzmann_constant: float = 1.0,
        tolerance: float = 1.0e-8,
        matrix_function_policy: MatrixFunctionPolicy | None = None,
        differentiation: StochasticDifferentiationPolicy = "pathwise",
    ) -> "MACFluctuatingHydrodynamicsPlan":
        return MACFluctuatingHydrodynamicsPlan(
            self.stress_divergence,
            self.dissipation,
            temperature=temperature,
            boltzmann_constant=boltzmann_constant,
            tolerance=tolerance,
            matrix_function_policy=matrix_function_policy,
            differentiation=differentiation,
        )


class MACFluctuatingHydrodynamicsPlan(StrictModule, NonTrainableState):
    """Thermal momentum forcing from a certified noise factor B with BB*=D."""

    noise_factor: AbstractLinearOperator
    dissipation: AbstractLinearOperator
    temperature: float = eqx.field(static=True)
    boltzmann_constant: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    matrix_function_policy: MatrixFunctionPolicy
    differentiation: StochasticDifferentiationPolicy = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        noise_factor: AbstractLinearOperator,
        dissipation: AbstractLinearOperator,
        /,
        *,
        temperature: float,
        boltzmann_constant: float = 1.0,
        tolerance: float = 1.0e-8,
        matrix_function_policy: MatrixFunctionPolicy | None = None,
        differentiation: StochasticDifferentiationPolicy = "pathwise",
    ):
        if not noise_factor.target.compatible(dissipation.source) or not (
            dissipation.source.compatible(dissipation.target)
        ):
            raise ValueError("Noise factor and dissipation spaces are incompatible.")
        temperature_ = float(temperature)
        boltzmann = float(boltzmann_constant)
        tolerance_ = float(tolerance)
        if (
            not np.isfinite(temperature_)
            or temperature_ < 0.0
            or not np.isfinite(boltzmann)
            or boltzmann <= 0.0
            or tolerance_ <= 0.0
        ):
            raise ValueError("Thermal parameters must be finite and admissible.")
        if differentiation not in ("pathwise", "weak"):
            raise ValueError("Unknown stochastic differentiation policy.")
        self.noise_factor = noise_factor
        self.dissipation = dissipation
        self.matrix_function_policy = (
            MatrixFunctionPolicy()
            if matrix_function_policy is None
            else matrix_function_policy
        )
        self.temperature = temperature_
        self.boltzmann_constant = boltzmann
        self.tolerance = tolerance_
        self.differentiation = differentiation
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-fluctuating-hydrodynamics",
                "noise_factor": noise_factor.operator_id,
                "dissipation": dissipation.operator_id,
                "temperature": temperature_,
                "boltzmann_constant": boltzmann,
                "differentiation": differentiation,
            }
        )

    def sample(
        self,
        step_size: ArrayLike,
        replay_key: StochasticReplayKey,
        /,
    ) -> MACFluctuatingHydrodynamicsResult:
        step = jnp.asarray(step_size)
        step = eqx.error_if(
            step,
            ~jnp.isfinite(step) | (step <= 0.0),
            "Thermal forcing requires a positive finite step.",
        )
        source_coordinates = self.noise_factor.source.flatten(
            self.noise_factor.source.zeros()
        )
        white_coordinates = jax.random.normal(
            replay_key.key(),
            source_coordinates.shape,
            dtype=source_coordinates.dtype,
        )
        white = _metric_white(
            self.noise_factor.source,
            white_coordinates,
            self.matrix_function_policy,
        )
        stochastic = self.noise_factor.mv(white.value)
        scale = jnp.sqrt(2.0 * self.boltzmann_constant * self.temperature / step)
        stochastic = jax.tree.map(lambda value: scale * value, stochastic)
        dissipative = self.dissipation.mv(stochastic)
        dissipative_power = jnp.real(
            self.dissipation.source.inner(stochastic, dissipative)
        )
        thermal_injection = (
            jnp.real(self.dissipation.source.inner(stochastic, stochastic)) / step
        )
        covariance_action = self.noise_factor.mv(self.noise_factor.adjoint_mv(stochastic))
        expected_action = self.dissipation.mv(stochastic)
        covariance_residual = jnp.sqrt(
            jnp.real(
                self.dissipation.source.inner(
                    jax.tree.map(
                        lambda left, right: left - right,
                        covariance_action,
                        expected_action,
                    ),
                    jax.tree.map(
                        lambda left, right: left - right,
                        covariance_action,
                        expected_action,
                    ),
                )
            )
        )
        scale_action = jnp.sqrt(
            jnp.real(self.dissipation.source.inner(expected_action, expected_action))
        )
        finite = (
            white.converged
            & jnp.isfinite(dissipative_power)
            & jnp.isfinite(thermal_injection)
            & jnp.isfinite(covariance_residual)
            & jnp.all(
                jnp.stack(
                    tuple(
                        jnp.all(jnp.isfinite(value))
                        for value in jax.tree.leaves(stochastic)
                    )
                )
            )
        )
        passed = finite & (
            covariance_residual <= self.tolerance * jnp.maximum(scale_action, 1.0)
        )
        report = FluctuationDissipationReport(
            dissipative_power,
            thermal_injection,
            thermal_injection - dissipative_power,
            covariance_residual,
            finite,
            passed,
        )
        return MACFluctuatingHydrodynamicsResult(
            stochastic,
            replay_key,
            report,
            finite,
            jnp.asarray(self.differentiation == "pathwise"),
            self.plan_id,
        )


class MACInertialStochasticStepResult(StrictModule):
    velocity: object
    deterministic_increment: object
    thermal_increment: object
    forcing: MACFluctuatingHydrodynamicsResult
    finite: Array
    accepted: Array
    plan_id: str = eqx.field(static=True)


class MACInertialStochasticStepPlan(StrictModule, NonTrainableState):
    """Inertial velocity update driven by fluctuating hydrodynamic stress."""

    velocity_space: AbstractVectorSpace
    inverse_mass: AbstractLinearOperator
    forcing: MACFluctuatingHydrodynamicsPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        velocity_space: AbstractVectorSpace,
        inverse_mass: AbstractLinearOperator,
        forcing: MACFluctuatingHydrodynamicsPlan,
        /,
    ):
        if not inverse_mass.source.compatible(velocity_space) or not (
            inverse_mass.target.compatible(velocity_space)
        ):
            raise ValueError("Inertial inverse mass must act on velocity space.")
        if not forcing.dissipation.source.compatible(velocity_space):
            raise ValueError("Thermal forcing and inertial velocity spaces differ.")
        self.velocity_space = velocity_space
        self.inverse_mass = inverse_mass
        self.forcing = forcing
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-inertial-stochastic-step",
                "velocity_space": velocity_space.space_id,
                "inverse_mass": inverse_mass.operator_id,
                "forcing": forcing.plan_id,
            }
        )

    def step(
        self,
        velocity,
        deterministic_force,
        step_size: ArrayLike,
        replay_key: StochasticReplayKey,
        /,
    ) -> MACInertialStochasticStepResult:
        velocity_ = self.velocity_space.validate(velocity)
        deterministic_force_ = self.velocity_space.validate(deterministic_force)
        step = jnp.asarray(step_size)
        thermal = self.forcing.sample(step, replay_key)
        deterministic_increment = jax.tree.map(
            lambda value: step * value,
            self.inverse_mass.mv(deterministic_force_),
        )
        thermal_increment = jax.tree.map(
            lambda value: step * value,
            self.inverse_mass.mv(thermal.stochastic_momentum),
        )
        updated = jax.tree.map(
            lambda value, deterministic_, thermal_: value + deterministic_ + thermal_,
            velocity_,
            deterministic_increment,
            thermal_increment,
        )
        finite = thermal.finite & jnp.all(
            jnp.stack(
                tuple(jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(updated))
            )
        )
        return MACInertialStochasticStepResult(
            updated,
            deterministic_increment,
            thermal_increment,
            thermal,
            finite,
            finite,
            self.plan_id,
        )


class FIBOverdampedStepResult(StrictModule):
    position: Array
    deterministic_increment: Array
    brownian_increment: Array
    stochastic_drift: Array
    replay_key: StochasticReplayKey
    mobility_sqrt_converged: Array
    finite: Array
    accepted: Array
    plan_id: str = eqx.field(static=True)


class FIBOverdampedPlan(StrictModule, NonTrainableState):
    """Overdamped fluctuating immersed-boundary step with random-finite-difference drift."""

    marker_space: AbstractVectorSpace
    mobility: MobilityProvider = eqx.field(static=True)
    temperature: float = eqx.field(static=True)
    boltzmann_constant: float = eqx.field(static=True)
    drift_epsilon: float = eqx.field(static=True)
    matrix_function_policy: MatrixFunctionPolicy
    differentiation: StochasticDifferentiationPolicy = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        marker_space: AbstractVectorSpace,
        mobility: MobilityProvider,
        /,
        *,
        temperature: float,
        boltzmann_constant: float = 1.0,
        drift_epsilon: float = 1.0e-6,
        matrix_function_policy: MatrixFunctionPolicy | None = None,
        differentiation: StochasticDifferentiationPolicy = "pathwise",
    ):
        if not callable(mobility):
            raise TypeError("mobility must be callable.")
        temperature_ = float(temperature)
        boltzmann = float(boltzmann_constant)
        epsilon = float(drift_epsilon)
        if temperature_ < 0.0 or boltzmann <= 0.0 or epsilon <= 0.0:
            raise ValueError("FIB thermal parameters must be admissible.")
        if differentiation not in ("pathwise", "weak"):
            raise ValueError("Unknown stochastic differentiation policy.")
        self.marker_space = marker_space
        self.mobility = mobility
        self.temperature = temperature_
        self.boltzmann_constant = boltzmann
        self.drift_epsilon = epsilon
        self.matrix_function_policy = (
            MatrixFunctionPolicy()
            if matrix_function_policy is None
            else matrix_function_policy
        )
        self.differentiation = differentiation
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fib-overdamped",
                "marker_space": marker_space.space_id,
                "temperature": temperature_,
                "boltzmann_constant": boltzmann,
                "drift_epsilon": epsilon,
                "differentiation": differentiation,
            }
        )

    def step(
        self,
        position: ArrayLike,
        force: ArrayLike,
        step_size: ArrayLike,
        replay_key: StochasticReplayKey,
        /,
    ) -> FIBOverdampedStepResult:
        position_ = self.marker_space.validate(jnp.asarray(position))
        force_ = self.marker_space.validate(jnp.asarray(force))
        step = jnp.asarray(step_size, dtype=self.marker_space.flatten(position_).dtype)
        step = eqx.error_if(
            step,
            ~jnp.isfinite(step) | (step <= 0.0),
            "FIB requires a positive finite step.",
        )
        mobility = self.mobility(position_)
        if not mobility.source.compatible(self.marker_space) or not (
            mobility.target.compatible(self.marker_space)
        ):
            raise ValueError("Mobility provider returned an incompatible operator.")
        if not mobility.properties.certifies("self_adjoint") or not (
            mobility.properties.certifies("positive_definite")
        ):
            raise ValueError("FIB mobility must certify self-adjoint positivity.")
        coordinates = self.marker_space.flatten(position_)
        white_coordinates, drift_coordinates = jax.random.normal(
            replay_key.key(),
            (2,) + coordinates.shape,
            dtype=coordinates.dtype,
        )
        white_result = _metric_white(
            self.marker_space, white_coordinates, self.matrix_function_policy
        )
        drift_result = _metric_white(
            self.marker_space, drift_coordinates, self.matrix_function_policy
        )
        white = white_result.value
        drift_probe = drift_result.value
        square_root = matrix_function_action(
            mobility,
            white,
            kind="sqrt",
            policy=self.matrix_function_policy,
        )
        brownian = jax.tree.map(
            lambda value: (
                jnp.sqrt(2.0 * self.boltzmann_constant * self.temperature * step) * value
            ),
            square_root.value,
        )
        epsilon = jnp.asarray(self.drift_epsilon, dtype=coordinates.dtype)
        plus_position = jax.tree.map(
            lambda value, probe: value + 0.5 * epsilon * probe,
            position_,
            drift_probe,
        )
        minus_position = jax.tree.map(
            lambda value, probe: value - 0.5 * epsilon * probe,
            position_,
            drift_probe,
        )
        plus = self.mobility(plus_position).mv(drift_probe)
        minus = self.mobility(minus_position).mv(drift_probe)
        stochastic_drift = jax.tree.map(
            lambda upper, lower: (
                step
                * self.boltzmann_constant
                * self.temperature
                * (upper - lower)
                / epsilon
            ),
            plus,
            minus,
        )
        deterministic = jax.tree.map(lambda value: step * value, mobility.mv(force_))
        updated = jax.tree.map(
            lambda value, deterministic_, brownian_, drift_: (
                value + deterministic_ + brownian_ + drift_
            ),
            position_,
            deterministic,
            brownian,
            stochastic_drift,
        )
        finite = jnp.all(
            jnp.stack(
                tuple(jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(updated))
            )
        )
        accepted = (
            square_root.converged
            & white_result.converged
            & drift_result.converged
            & finite
        )
        return FIBOverdampedStepResult(
            updated,
            deterministic,
            brownian,
            stochastic_drift,
            replay_key,
            square_root.converged,
            finite,
            accepted,
            self.plan_id,
        )


__all__ = [
    "FIBOverdampedPlan",
    "FIBOverdampedStepResult",
    "FluctuationDissipationReport",
    "MACDiscreteStochasticStressPlan",
    "MACFluctuatingHydrodynamicsPlan",
    "MACInertialStochasticStepPlan",
    "MACInertialStochasticStepResult",
    "MACFluctuatingHydrodynamicsResult",
    "MobilityProvider",
    "StochasticDifferentiationPolicy",
    "StochasticReplayKey",
]
