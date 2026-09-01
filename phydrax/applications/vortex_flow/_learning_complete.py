#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...optim import (
    AbstractMinimizationMethod,
    MinimizationResult,
    minimize,
    OptimizationTermination,
)


class VorticityLearningEvidence(StrictModule):
    data_loss: Array
    circulation_residual: Array
    dissipation_violation: Array
    finite: Array
    objective_id: str = eqx.field(static=True)


class VorticityLearningResult(StrictModule):
    model: PyTree[Any]
    minimization_result: MinimizationResult
    evidence: VorticityLearningEvidence
    successful: Array
    plan_id: str = eqx.field(static=True)


class NativeVorticityLearningPlan(StrictModule, NonTrainableState):
    method: AbstractMinimizationMethod
    termination: OptimizationTermination
    circulation_weight: float = eqx.field(static=True)
    dissipation_weight: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: AbstractMinimizationMethod,
        /,
        *,
        termination: OptimizationTermination | None = None,
        circulation_weight: float = 1.0,
        dissipation_weight: float = 1.0,
    ):
        if (
            not isinstance(method, AbstractMinimizationMethod)
            or circulation_weight < 0.0
            or dissipation_weight < 0.0
        ):
            raise ValueError("Native vorticity learning controls are invalid.")
        self.method = method
        self.termination = (
            OptimizationTermination() if termination is None else termination
        )
        self.circulation_weight, self.dissipation_weight = (
            float(circulation_weight),
            float(dissipation_weight),
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "native-vorticity-learning-plan",
                "method": type(method).__name__,
                "circulation_weight": self.circulation_weight,
                "dissipation_weight": self.dissipation_weight,
            }
        )

    def train(
        self,
        model: PyTree[Any],
        sample_position: ArrayLike,
        sample_vorticity: ArrayLike,
        sample_weight: ArrayLike,
        /,
        *,
        target_circulation: ArrayLike,
        previous_prediction: ArrayLike | None = None,
    ) -> VorticityLearningResult:
        position = jnp.asarray(sample_position)
        target = jnp.asarray(sample_vorticity, dtype=position.dtype)
        weight = jnp.asarray(sample_weight, dtype=position.dtype)
        circulation_target = jnp.asarray(target_circulation, dtype=position.dtype)
        if (
            position.ndim != 2
            or target.shape[0] != position.shape[0]
            or weight.shape != (position.shape[0],)
        ):
            raise ValueError("Learning samples/weights have incompatible shapes.")
        previous = (
            None
            if previous_prediction is None
            else jnp.asarray(previous_prediction, dtype=position.dtype)
        )

        def predict(parameters):
            return jax.vmap(parameters)(position)

        def objective(parameters, args):
            del args
            prediction = predict(parameters)
            residual = prediction - target
            data = jnp.sum(
                weight.reshape(weight.shape + (1,) * (prediction.ndim - 1)) * residual**2
            ) / jnp.maximum(jnp.sum(weight), 1.0)
            circulation = jnp.sum(
                weight.reshape(weight.shape + (1,) * (prediction.ndim - 1)) * prediction,
                axis=0,
            )
            circulation_residual = jnp.sum((circulation - circulation_target) ** 2)
            if previous is None:
                dissipation_violation = jnp.asarray(0.0, dtype=data.dtype)
            else:
                current_energy = jnp.sum(
                    weight.reshape(weight.shape + (1,) * (prediction.ndim - 1))
                    * prediction**2
                )
                previous_energy = jnp.sum(
                    weight.reshape(weight.shape + (1,) * (previous.ndim - 1))
                    * previous**2
                )
                dissipation_violation = (
                    jnp.maximum(current_energy - previous_energy, 0.0) ** 2
                )
            return (
                data
                + self.circulation_weight * circulation_residual
                + self.dissipation_weight * dissipation_violation
            )

        result = minimize(
            objective, model, method=self.method, termination=self.termination
        )
        trained = result.parameters
        prediction = predict(trained)
        residual = prediction - target
        data_loss = jnp.sum(
            weight.reshape(weight.shape + (1,) * (prediction.ndim - 1)) * residual**2
        ) / jnp.maximum(jnp.sum(weight), 1.0)
        circulation = jnp.sum(
            weight.reshape(weight.shape + (1,) * (prediction.ndim - 1)) * prediction,
            axis=0,
        )
        circulation_residual = jnp.linalg.norm(circulation - circulation_target)
        dissipation_violation = (
            jnp.asarray(0.0, dtype=data_loss.dtype)
            if previous is None
            else jnp.maximum(jnp.sum(prediction**2) - jnp.sum(previous**2), 0.0)
        )
        finite = jnp.all(jnp.isfinite(prediction)) & jnp.isfinite(data_loss)
        evidence = VorticityLearningEvidence(
            data_loss, circulation_residual, dissipation_violation, finite, self.plan_id
        )
        return VorticityLearningResult(
            trained, result, evidence, result.successful & finite, self.plan_id
        )


class PeriodicVorticityReconstructionResult(StrictModule):
    velocity: Array
    velocity_gradient: Array | None
    divergence_norm: Array
    compatibility_residual: Array
    successful: Array
    reconstruction_id: str = eqx.field(static=True)


class PeriodicVorticityReconstructionPlan(StrictModule, NonTrainableState):
    shape: tuple[int, ...] = eqx.field(static=True)
    periods: Array
    dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, shape: tuple[int, ...], periods: ArrayLike, /):
        shape_ = tuple(int(value) for value in shape)
        periods_ = jnp.asarray(periods, dtype=float)
        if (
            len(shape_) not in (2, 3)
            or periods_.shape != (len(shape_),)
            or any(value < 2 for value in shape_)
            or jnp.any(periods_ <= 0.0)
        ):
            raise ValueError("Periodic reconstruction shape/periods are invalid.")
        self.shape, self.periods, self.dimension = shape_, periods_, len(shape_)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-vorticity-reconstruction",
                "shape": shape_,
                "periods": tuple(float(value) for value in periods_),
            }
        )

    def reconstruct(
        self, vorticity: ArrayLike, /, *, velocity_gradient: bool = False
    ) -> PeriodicVorticityReconstructionResult:
        omega = jnp.asarray(vorticity)
        expected = self.shape if self.dimension == 2 else self.shape + (3,)
        if omega.shape != expected:
            raise ValueError("Periodic vorticity grid shape is incompatible.")
        axes = tuple(
            jnp.fft.fftfreq(count, d=float(self.periods[axis]) / count) * 2.0 * jnp.pi
            for axis, count in enumerate(self.shape)
        )
        mesh = jnp.meshgrid(*axes, indexing="ij")
        squared = sum(component**2 for component in mesh)
        inverse = jnp.where(squared > 0.0, 1.0 / squared, 0.0)
        coefficients = jnp.fft.fftn(omega, axes=tuple(range(self.dimension)))
        total = coefficients[(0,) * self.dimension]
        scale = jnp.maximum(jnp.sum(jnp.abs(coefficients)), 1.0)
        compatibility = jnp.max(jnp.abs(total)) / scale
        if self.dimension == 2:
            velocity_coefficients = jnp.stack(
                (
                    1j * mesh[1] * inverse * coefficients,
                    -1j * mesh[0] * inverse * coefficients,
                ),
                axis=-1,
            )
        else:
            wave = jnp.stack(mesh, axis=-1)
            velocity_coefficients = (
                1j * jnp.cross(wave, coefficients) * inverse[..., None]
            )
        velocity = jnp.fft.ifftn(
            velocity_coefficients, axes=tuple(range(self.dimension))
        ).real
        gradient = None
        if velocity_gradient:
            gradient = jnp.stack(
                tuple(
                    jnp.fft.ifftn(
                        1j * mesh[axis][..., None] * velocity_coefficients,
                        axes=tuple(range(self.dimension)),
                    ).real
                    for axis in range(self.dimension)
                ),
                axis=-1,
            )
        divergence_coefficients = sum(
            1j * mesh[axis] * velocity_coefficients[..., axis]
            for axis in range(self.dimension)
        )
        divergence = jnp.linalg.norm(divergence_coefficients)
        successful = (compatibility <= 1.0e-12) & jnp.all(jnp.isfinite(velocity))
        return PeriodicVorticityReconstructionResult(
            velocity, gradient, divergence, compatibility, successful, self.plan_id
        )


class ConstrainedLearnedClosureResult(StrictModule):
    rate: Array
    circulation_residual: Array
    dissipation: Array
    in_distribution: Array
    successful: Array
    closure_id: str = eqx.field(static=True)


class ConstrainedLearnedClosure(StrictModule, NonTrainableState):
    model: PyTree[Any]
    distribution_center: Array
    distribution_scale: Array
    closure_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: PyTree[Any],
        distribution_center: ArrayLike,
        distribution_scale: ArrayLike,
        /,
        *,
        closure_id: str,
    ):
        center, scale = jnp.asarray(distribution_center), jnp.asarray(distribution_scale)
        if center.shape != scale.shape or jnp.any(scale <= 0.0) or not str(closure_id):
            raise ValueError("Learned closure distribution controls are invalid.")
        self.model, self.distribution_center, self.distribution_scale, self.closure_id = (
            model,
            center,
            scale,
            str(closure_id),
        )

    def evaluate(
        self, features: ArrayLike, strength: ArrayLike, /
    ) -> ConstrainedLearnedClosureResult:
        feature, gamma = jnp.asarray(features), jnp.asarray(strength)
        if (
            feature.shape[-1] != self.distribution_center.size
            or gamma.shape[0] != feature.shape[0]
        ):
            raise ValueError("Learned closure feature/strength shapes are incompatible.")
        raw = jax.vmap(self.model)(feature)
        mean_rate = jnp.mean(raw, axis=0)
        conservative = raw - mean_rate
        energy = jnp.sum(conservative * gamma)
        rate = jnp.where(
            energy > 0.0,
            conservative
            - energy
            * gamma
            / jnp.maximum(jnp.sum(gamma * gamma), jnp.finfo(gamma.dtype).tiny),
            conservative,
        )
        normalized = jnp.abs(
            (feature - self.distribution_center) / self.distribution_scale
        )
        in_distribution = jnp.all(normalized <= 5.0)
        residual = jnp.sum(rate, axis=0)
        dissipation = jnp.sum(rate * gamma)
        successful = (
            in_distribution
            & jnp.all(jnp.isfinite(rate))
            & (jnp.max(jnp.abs(residual)) <= 1.0e-10)
            & (dissipation <= 1.0e-10)
        )
        return ConstrainedLearnedClosureResult(
            rate, residual, dissipation, in_distribution, successful, self.closure_id
        )


__all__ = [
    "ConstrainedLearnedClosure",
    "ConstrainedLearnedClosureResult",
    "NativeVorticityLearningPlan",
    "PeriodicVorticityReconstructionPlan",
    "PeriodicVorticityReconstructionResult",
    "VorticityLearningEvidence",
    "VorticityLearningResult",
]
