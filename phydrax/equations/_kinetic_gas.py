#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve


class MolecularVelocityQuadrature(StrictModule, NonTrainableState):
    """Physical three-dimensional molecular-velocity integration rule."""

    velocities: Array
    weights: Array
    streaming_velocity: Array
    spatial_dimension: int = eqx.field(static=True)
    velocity_count: int = eqx.field(static=True)
    quadrature_id: str = eqx.field(static=True)

    def __init__(
        self,
        velocities: ArrayLike,
        weights: ArrayLike,
        spatial_dimension: int,
        /,
        *,
        streaming_projection: ArrayLike | None = None,
        quadrature_id: str | None = None,
    ) -> None:
        velocity = np.asarray(velocities, dtype=float)
        weight = np.asarray(weights, dtype=float)
        dimension = int(spatial_dimension)
        if (
            velocity.ndim != 2
            or velocity.shape[1] != 3
            or velocity.shape[0] < 5
            or weight.shape != (velocity.shape[0],)
            or np.any(~np.isfinite(velocity))
            or np.any(~np.isfinite(weight))
            or np.any(weight <= 0.0)
            or dimension not in (1, 2, 3)
        ):
            raise ValueError("Molecular velocity quadrature is invalid.")
        projection = (
            velocity[:, :dimension]
            if streaming_projection is None
            else np.asarray(streaming_projection, dtype=float)
        )
        if projection.shape != (velocity.shape[0], dimension) or np.any(
            ~np.isfinite(projection)
        ):
            raise ValueError(
                "streaming_projection must have shape (velocity_count, spatial_dimension)."
            )
        generated = canonical_fingerprint(
            {
                "kind": "molecular-velocity-quadrature",
                "velocities": array_tree_fingerprint(velocity),
                "weights": array_tree_fingerprint(weight),
                "streaming_projection": array_tree_fingerprint(projection),
                "spatial_dimension": dimension,
            }
        )
        self.velocities = jnp.asarray(velocity)
        self.weights = jnp.asarray(weight)
        self.streaming_velocity = jnp.asarray(projection)
        self.spatial_dimension = dimension
        self.velocity_count = velocity.shape[0]
        self.quadrature_id = generated if quadrature_id is None else str(quadrature_id)
        if not self.quadrature_id:
            raise ValueError("quadrature_id must be non-empty.")

    @property
    def moment_features(self) -> Array:
        kinetic_energy = 0.5 * jnp.sum(self.velocities**2, axis=-1, keepdims=True)
        return jnp.concatenate(
            (
                jnp.ones((self.velocity_count, 1), dtype=self.velocities.dtype),
                self.velocities,
                kinetic_energy,
            ),
            axis=-1,
        )

    def moments(self, population: ArrayLike, /) -> Array:
        value = jnp.asarray(population)
        if value.shape[-1] != self.velocity_count:
            raise ValueError("population must end in the molecular velocity axis.")
        return contract(
            "q,...q,qm->...m",
            self.weights.astype(value.dtype),
            value,
            self.moment_features.astype(value.dtype),
            backend="jax",
        )


class DiscreteMaxwellianResult(StrictModule):
    population: Array
    multipliers: Array
    target_moments: Array
    moment_residual: Array
    entropy: Array
    iteration_count: Array
    successful: Array
    quadrature_id: str = eqx.field(static=True)


class PositiveDiscreteMaxwellianPlan(StrictModule):
    quadrature: MolecularVelocityQuadrature
    tolerance: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        quadrature: MolecularVelocityQuadrature,
        /,
        *,
        tolerance: float = 1.0e-10,
        maximum_steps: int = 40,
    ) -> None:
        if not isinstance(quadrature, MolecularVelocityQuadrature):
            raise TypeError("quadrature must be MolecularVelocityQuadrature.")
        tolerance_value = float(tolerance)
        steps = int(maximum_steps)
        if not np.isfinite(tolerance_value) or tolerance_value <= 0.0 or steps <= 0:
            raise ValueError("Discrete Maxwellian solver controls are invalid.")
        self.quadrature = quadrature
        self.tolerance = tolerance_value
        self.maximum_steps = steps
        self.plan_id = canonical_fingerprint(
            {
                "kind": "positive-discrete-maxwellian",
                "quadrature": quadrature.quadrature_id,
                "tolerance": tolerance_value,
                "maximum_steps": steps,
            }
        )

    def solve(
        self,
        target_moments: ArrayLike,
        /,
        *,
        initial_multipliers: ArrayLike | None = None,
    ) -> DiscreteMaxwellianResult:
        target = jnp.asarray(target_moments)
        if target.shape != (5,):
            raise ValueError("target_moments must contain mass, 3-momentum, and energy.")
        features = self.quadrature.moment_features.astype(target.dtype)
        weights = self.quadrature.weights.astype(target.dtype)
        density = target[0]
        velocity = target[1:4] / jnp.maximum(density, jnp.finfo(target.dtype).tiny)
        thermal = (
            2.0 * target[4] / jnp.maximum(density, jnp.finfo(target.dtype).tiny)
            - jnp.sum(velocity**2)
        ) / 3.0
        if initial_multipliers is None:
            energy_multiplier = -1.0 / jnp.maximum(thermal, 1.0e-6)
            momentum_multiplier = velocity / jnp.maximum(thermal, 1.0e-6)
            exponent_without_mass = (
                features[:, 1:4] @ momentum_multiplier
                + features[:, 4] * energy_multiplier
            )
            mass_multiplier = jnp.log(
                jnp.maximum(density, jnp.finfo(target.dtype).tiny)
                / jnp.sum(weights * jnp.exp(exponent_without_mass))
            )
            initial = jnp.concatenate(
                (mass_multiplier[None], momentum_multiplier, energy_multiplier[None])
            )
        else:
            initial = jnp.asarray(initial_multipliers, dtype=target.dtype)
            if initial.shape != (5,):
                raise ValueError("initial_multipliers must have shape (5,).")

        def evaluate(multipliers):
            exponent = features @ multipliers
            population = jnp.exp(jnp.clip(exponent, -700.0, 700.0))
            weighted = weights * population
            moments = contract("q,qm->m", weighted, features, backend="jax")
            jacobian = contract(
                "q,qi,qj->ij", weighted, features, features, backend="jax"
            )
            return population, moments - target, jacobian

        rates = jnp.asarray((1.0, 0.5, 0.25, 0.125, 0.0625), dtype=target.dtype)

        def body(_, carry):
            multipliers, active, linear_success = carry
            _, residual, jacobian = evaluate(multipliers)
            direction = _solve_dense(jacobian, -residual)
            candidates = multipliers[None, :] + rates[:, None] * direction.value[None, :]
            candidate_residual = jax.vmap(lambda value: evaluate(value)[1])(candidates)
            norms = jnp.max(jnp.abs(candidate_residual), axis=-1)
            selected = jnp.argmin(norms)
            updated = candidates[selected]
            next_active = active & (norms[selected] > self.tolerance)
            return updated, next_active, linear_success & direction.successful

        _, initial_residual, _ = evaluate(initial)
        state = jax.lax.fori_loop(
            0,
            self.maximum_steps,
            body,
            (
                initial,
                jnp.max(jnp.abs(initial_residual)) > self.tolerance,
                jnp.asarray(True),
            ),
        )
        multipliers, active, linear_success = state
        population, residual, _ = evaluate(multipliers)
        entropy = jnp.sum(
            weights
            * jnp.where(
                population > 0.0,
                population * (jnp.log(population) - 1.0),
                0.0,
            )
        )
        successful = (
            ~active
            & linear_success
            & (density > 0.0)
            & (thermal > 0.0)
            & jnp.all(jnp.isfinite(population))
            & jnp.all(population > 0.0)
            & (jnp.max(jnp.abs(residual)) <= self.tolerance)
        )
        return DiscreteMaxwellianResult(
            population,
            multipliers,
            target,
            residual,
            entropy,
            jnp.asarray(self.maximum_steps, dtype=jnp.int32),
            successful,
            self.quadrature.quadrature_id,
        )


class PopulationUpwindFluxPlan(StrictModule):
    quadrature: MolecularVelocityQuadrature
    plan_id: str = eqx.field(static=True)

    def __init__(self, quadrature: MolecularVelocityQuadrature, /) -> None:
        if not isinstance(quadrature, MolecularVelocityQuadrature):
            raise TypeError("quadrature must be MolecularVelocityQuadrature.")
        self.quadrature = quadrature
        self.plan_id = canonical_fingerprint(
            {"kind": "population-upwind-flux", "quadrature": quadrature.quadrature_id}
        )

    def physical_flux(self, population: ArrayLike, axis: int, /) -> Array:
        value = jnp.asarray(population)
        axis_value = int(axis)
        if value.shape[-1] != self.quadrature.velocity_count:
            raise ValueError("population must end in the molecular velocity axis.")
        if axis_value < 0 or axis_value >= self.quadrature.spatial_dimension:
            raise ValueError("axis is outside the spatial dimension.")
        return value * self.quadrature.streaming_velocity[:, axis_value]

    def numerical_flux(
        self,
        left: ArrayLike,
        right: ArrayLike,
        normal: ArrayLike,
        /,
    ) -> Array:
        left_value = jnp.asarray(left)
        right_value = jnp.asarray(right, dtype=left_value.dtype)
        normal_value = jnp.asarray(normal, dtype=left_value.dtype)
        if (
            left_value.shape != right_value.shape
            or left_value.shape[-1] != self.quadrature.velocity_count
        ):
            raise ValueError("left and right populations must share the velocity axis.")
        if normal_value.shape != (self.quadrature.spatial_dimension,):
            raise ValueError("normal must have shape (spatial_dimension,).")
        normal_speed = (
            self.quadrature.streaming_velocity.astype(left_value.dtype) @ normal_value
        )
        return (
            jnp.maximum(normal_speed, 0.0) * left_value
            + jnp.minimum(normal_speed, 0.0) * right_value
        )


class KineticCollisionResult(StrictModule):
    population: Array
    equilibrium: Array
    moment_defect: Array
    entropy_change: Array
    relaxation_time: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class MonatomicBGKCollisionPlan(StrictModule):
    quadrature: MolecularVelocityQuadrature
    maxwellian: PositiveDiscreteMaxwellianPlan
    dynamic_viscosity: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        quadrature: MolecularVelocityQuadrature,
        /,
        *,
        dynamic_viscosity: float,
    ) -> None:
        viscosity = float(dynamic_viscosity)
        if not isinstance(quadrature, MolecularVelocityQuadrature):
            raise TypeError("quadrature must be MolecularVelocityQuadrature.")
        if not np.isfinite(viscosity) or viscosity <= 0.0:
            raise ValueError("dynamic_viscosity must be finite and positive.")
        self.quadrature = quadrature
        self.maxwellian = PositiveDiscreteMaxwellianPlan(quadrature)
        self.dynamic_viscosity = viscosity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "monatomic-bgk",
                "quadrature": quadrature.quadrature_id,
                "dynamic_viscosity": viscosity,
            }
        )

    def advance(
        self,
        population: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> KineticCollisionResult:
        value = jnp.asarray(population)
        if value.shape != (self.quadrature.velocity_count,):
            raise ValueError("BGK population must have shape (velocity_count,).")
        moments = self.quadrature.moments(value)
        equilibrium = self.maxwellian.solve(moments)
        density = moments[0]
        velocity = moments[1:4] / density
        thermal = (2.0 * moments[4] / density - jnp.sum(velocity**2)) / 3.0
        pressure = density * thermal
        relaxation = self.dynamic_viscosity / pressure
        decay = jnp.exp(-jnp.asarray(step_size, dtype=value.dtype) / relaxation)
        updated = equilibrium.population + (value - equilibrium.population) * decay
        defect = self.quadrature.moments(updated) - moments
        weights = self.quadrature.weights.astype(value.dtype)
        entropy_before = jnp.sum(weights * value * (jnp.log(value) - 1.0))
        entropy_after = jnp.sum(weights * updated * (jnp.log(updated) - 1.0))
        successful = (
            equilibrium.successful
            & jnp.all(jnp.isfinite(value))
            & jnp.all(value > 0.0)
            & jnp.all(updated > 0.0)
            & jnp.isfinite(relaxation)
            & (relaxation > 0.0)
            & (jnp.max(jnp.abs(defect)) <= 10.0 * self.maxwellian.tolerance)
            & (entropy_after <= entropy_before + 10.0 * self.maxwellian.tolerance)
        )
        return KineticCollisionResult(
            updated,
            equilibrium.population,
            defect,
            entropy_after - entropy_before,
            relaxation,
            successful,
            self.plan_id,
        )


class ShakhovCollisionPlan(StrictModule):
    bgk: MonatomicBGKCollisionPlan
    prandtl_number: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        quadrature: MolecularVelocityQuadrature,
        /,
        *,
        dynamic_viscosity: float,
        prandtl_number: float,
    ) -> None:
        prandtl = float(prandtl_number)
        if not np.isfinite(prandtl) or prandtl <= 0.0:
            raise ValueError("prandtl_number must be finite and positive.")
        self.bgk = MonatomicBGKCollisionPlan(
            quadrature, dynamic_viscosity=dynamic_viscosity
        )
        self.prandtl_number = prandtl
        self.plan_id = canonical_fingerprint(
            {
                "kind": "shakhov-collision",
                "bgk": self.bgk.plan_id,
                "prandtl_number": prandtl,
            }
        )

    def target(self, population: ArrayLike, /) -> tuple[Array, Array]:
        value = jnp.asarray(population)
        moments = self.bgk.quadrature.moments(value)
        maxwellian = self.bgk.maxwellian.solve(moments)
        density = moments[0]
        velocity = moments[1:4] / density
        peculiar = self.bgk.quadrature.velocities.astype(value.dtype) - velocity
        speed_squared = jnp.sum(peculiar**2, axis=-1)
        thermal = (2.0 * moments[4] / density - jnp.sum(velocity**2)) / 3.0
        pressure = density * thermal
        heat_flux = contract(
            "q,q,qi->i",
            self.bgk.quadrature.weights.astype(value.dtype),
            0.5 * speed_squared * value,
            peculiar,
            backend="jax",
        )
        correction = (
            (1.0 - self.prandtl_number)
            * contract("qi,i->q", peculiar, heat_flux, backend="jax")
            / (5.0 * pressure * thermal**2)
            * (speed_squared / thermal - 5.0)
        )
        target = maxwellian.population * (1.0 + correction)
        raw_defect = self.bgk.quadrature.moments(target) - moments
        features = self.bgk.quadrature.moment_features.astype(value.dtype)
        gram = contract(
            "q,qi,qj->ij",
            self.bgk.quadrature.weights.astype(value.dtype),
            features,
            features,
            backend="jax",
        )
        correction_coefficients = _solve_dense(gram, raw_defect)
        target = target - features @ correction_coefficients.value
        defect = self.bgk.quadrature.moments(target) - moments
        return target, defect

    def advance(
        self, population: ArrayLike, step_size: ArrayLike, /
    ) -> KineticCollisionResult:
        value = jnp.asarray(population)
        target, defect = self.target(value)
        moments = self.bgk.quadrature.moments(value)
        density = moments[0]
        velocity = moments[1:4] / density
        thermal = (2.0 * moments[4] / density - jnp.sum(velocity**2)) / 3.0
        relaxation = self.bgk.dynamic_viscosity / (density * thermal)
        decay = jnp.exp(-jnp.asarray(step_size, dtype=value.dtype) / relaxation)
        updated = target + (value - target) * decay
        weights = self.bgk.quadrature.weights.astype(value.dtype)
        entropy_before = jnp.sum(weights * value * (jnp.log(value) - 1.0))
        entropy_after = jnp.sum(
            weights
            * jnp.where(updated > 0.0, updated * (jnp.log(updated) - 1.0), jnp.inf)
        )
        successful = jnp.all(target > 0.0) & jnp.all(updated > 0.0) & jnp.max(
            jnp.abs(defect)
        ) <= 1.0e-8 & jnp.isfinite(entropy_after)
        return KineticCollisionResult(
            updated,
            target,
            defect,
            entropy_after - entropy_before,
            relaxation,
            successful,
            self.plan_id,
        )


class MaxwellGasSurfaceBoundary(StrictModule):
    quadrature: MolecularVelocityQuadrature
    outward_normal: Array
    wall_velocity: Array
    wall_temperature: float = eqx.field(static=True)
    accommodation: float = eqx.field(static=True)
    reflection_indices: Array
    boundary_id: str = eqx.field(static=True)

    def __init__(
        self,
        quadrature: MolecularVelocityQuadrature,
        outward_normal: ArrayLike,
        /,
        *,
        wall_temperature: float,
        wall_velocity: ArrayLike = (0.0, 0.0, 0.0),
        accommodation: float = 1.0,
    ) -> None:
        if not isinstance(quadrature, MolecularVelocityQuadrature):
            raise TypeError("quadrature must be MolecularVelocityQuadrature.")
        normal = np.asarray(outward_normal, dtype=float)
        velocity = np.asarray(wall_velocity, dtype=float)
        temperature = float(wall_temperature)
        alpha = float(accommodation)
        if (
            normal.shape != (3,)
            or not np.all(np.isfinite(normal))
            or np.linalg.norm(normal) <= 0.0
            or velocity.shape != (3,)
            or np.any(~np.isfinite(velocity))
            or not np.isfinite(temperature)
            or temperature <= 0.0
            or not np.isfinite(alpha)
            or not 0.0 <= alpha <= 1.0
        ):
            raise ValueError("Maxwell wall parameters are invalid.")
        unit = normal / np.linalg.norm(normal)
        relative = np.asarray(quadrature.velocities) - velocity
        reflected = relative - 2.0 * (relative @ unit)[:, None] * unit[None, :] + velocity
        distances = np.linalg.norm(
            reflected[:, None, :] - np.asarray(quadrature.velocities)[None, :, :],
            axis=-1,
        )
        indices = np.argmin(distances, axis=-1)
        if np.any(np.min(distances, axis=-1) > 1.0e-10):
            raise ValueError("Velocity quadrature is not closed under wall reflection.")
        self.quadrature = quadrature
        self.outward_normal = jnp.asarray(unit)
        self.wall_velocity = jnp.asarray(velocity)
        self.wall_temperature = temperature
        self.accommodation = alpha
        self.reflection_indices = jnp.asarray(indices, dtype=jnp.int32)
        self.boundary_id = canonical_fingerprint(
            {
                "kind": "maxwell-gas-surface-boundary",
                "quadrature": quadrature.quadrature_id,
                "normal": array_tree_fingerprint(unit),
                "wall_velocity": array_tree_fingerprint(velocity),
                "wall_temperature": temperature,
                "accommodation": alpha,
            }
        )

    def exterior_population(self, interior: ArrayLike, /) -> Array:
        value = jnp.asarray(interior)
        if value.shape != (self.quadrature.velocity_count,):
            raise ValueError("interior population must have shape (velocity_count,).")
        relative = self.quadrature.velocities.astype(value.dtype) - self.wall_velocity
        normal_speed = relative @ self.outward_normal.astype(value.dtype)
        incoming = normal_speed < 0.0
        specular = value[self.reflection_indices]
        unit_maxwellian = jnp.exp(
            -0.5 * jnp.sum(relative**2, axis=-1) / self.wall_temperature
        )
        outgoing_flux = jnp.sum(
            self.quadrature.weights
            * jnp.where(normal_speed > 0.0, normal_speed * value, 0.0)
        )
        incoming_unit_flux = jnp.sum(
            self.quadrature.weights
            * jnp.where(incoming, -normal_speed * unit_maxwellian, 0.0)
        )
        diffuse = (
            unit_maxwellian
            * outgoing_flux
            / jnp.maximum(incoming_unit_flux, jnp.finfo(value.dtype).tiny)
        )
        reflected = (1.0 - self.accommodation) * specular + self.accommodation * diffuse
        return jnp.where(incoming, reflected, value)


class KineticBreakdownEvidence(StrictModule):
    knudsen_number: Array
    distribution_defect: Array
    kinetic_required: Array
    plan_id: str = eqx.field(static=True)


class KineticBreakdownPlan(StrictModule):
    collision: MonatomicBGKCollisionPlan
    knudsen_threshold: float = eqx.field(static=True)
    defect_threshold: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        collision: MonatomicBGKCollisionPlan,
        /,
        *,
        knudsen_threshold: float = 0.01,
        defect_threshold: float = 0.05,
    ) -> None:
        if not isinstance(collision, MonatomicBGKCollisionPlan):
            raise TypeError("collision must be MonatomicBGKCollisionPlan.")
        knudsen = float(knudsen_threshold)
        defect = float(defect_threshold)
        if not 0.0 < knudsen or not 0.0 < defect:
            raise ValueError("Kinetic breakdown thresholds must be positive.")
        self.collision = collision
        self.knudsen_threshold = knudsen
        self.defect_threshold = defect
        self.plan_id = canonical_fingerprint(
            {
                "kind": "kinetic-breakdown",
                "collision": collision.plan_id,
                "knudsen_threshold": knudsen,
                "defect_threshold": defect,
            }
        )

    def evaluate(
        self,
        population: ArrayLike,
        characteristic_length: ArrayLike,
        /,
    ) -> KineticBreakdownEvidence:
        value = jnp.asarray(population)
        moments = self.collision.quadrature.moments(value)
        equilibrium = self.collision.maxwellian.solve(moments)
        density = moments[0]
        velocity = moments[1:4] / density
        thermal = (2.0 * moments[4] / density - jnp.sum(velocity**2)) / 3.0
        pressure = density * thermal
        relaxation = self.collision.dynamic_viscosity / pressure
        knudsen = relaxation * jnp.sqrt(thermal) / jnp.asarray(characteristic_length)
        defect = (
            jnp.sum(
                self.collision.quadrature.weights
                * jnp.abs(value - equilibrium.population)
            )
            / density
        )
        required = (
            ~equilibrium.successful
            | (knudsen >= self.knudsen_threshold)
            | (defect >= self.defect_threshold)
        )
        return KineticBreakdownEvidence(
            knudsen,
            defect,
            required,
            self.plan_id,
        )


def _solve_dense(matrix: Array, right_hand_side: Array, /):
    return solve(
        LinearSystem(DenseLinearOperator(matrix)),
        right_hand_side,
        policy=LinearSolvePolicy(DenseLU()),
    )


__all__ = [
    "DiscreteMaxwellianResult",
    "KineticBreakdownEvidence",
    "KineticBreakdownPlan",
    "KineticCollisionResult",
    "MaxwellGasSurfaceBoundary",
    "MolecularVelocityQuadrature",
    "MonatomicBGKCollisionPlan",
    "PopulationUpwindFluxPlan",
    "PositiveDiscreteMaxwellianPlan",
    "ShakhovCollisionPlan",
]
