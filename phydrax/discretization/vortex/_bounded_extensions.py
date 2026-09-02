#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class EquilibriumWallVortexClosureEvidence(StrictModule):
    root_residual: Array
    y_plus: Array
    mach_margin: Array
    pressure_gradient_margin: Array
    roughness_margin: Array
    dissipation: Array
    finite: Array
    derivative_valid: Array
    successful: Array


class EquilibriumWallVortexClosureResult(StrictModule):
    friction_velocity: Array
    traction: Array
    vortex_strength_increment: Array
    evidence: EquilibriumWallVortexClosureEvidence


class EquilibriumWallVortexClosurePlan(StrictModule, NonTrainableState):
    """Attached equilibrium Reichardt wall law with a finite validity envelope."""

    density: float = eqx.field(static=True)
    kinematic_viscosity: float = eqx.field(static=True)
    wall_sample_height: Array
    wall_measure: Array
    roughness_height: float = eqx.field(static=True)
    root_iterations: int = eqx.field(static=True)
    root_tolerance: float = eqx.field(static=True)
    minimum_y_plus: float = eqx.field(static=True)
    maximum_y_plus: float = eqx.field(static=True)
    maximum_mach: float = eqx.field(static=True)
    maximum_clauser_parameter: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        density: float,
        kinematic_viscosity: float,
        wall_sample_height: ArrayLike,
        wall_measure: ArrayLike,
        /,
        *,
        roughness_height: float = 0.0,
        root_iterations: int = 48,
        root_tolerance: float = 1.0e-8,
        y_plus_envelope: tuple[float, float] = (0.0, 1.0e5),
        maximum_mach: float = 0.3,
        maximum_clauser_parameter: float = 1.0,
    ):
        density_ = float(density)
        viscosity = float(kinematic_viscosity)
        height = np.asarray(wall_sample_height, dtype=float)
        measure = np.asarray(wall_measure, dtype=float)
        roughness = float(roughness_height)
        iterations = int(root_iterations)
        tolerance = float(root_tolerance)
        y_min, y_max = (float(value) for value in y_plus_envelope)
        mach = float(maximum_mach)
        clauser = float(maximum_clauser_parameter)
        if height.ndim != 1 or measure.shape != height.shape or height.size == 0:
            raise ValueError("Wall sample height/measure must be matching vectors.")
        if np.any(~np.isfinite(height)) or np.any(height <= 0.0):
            raise ValueError("Wall sample heights must be finite and positive.")
        if np.any(~np.isfinite(measure)) or np.any(measure <= 0.0):
            raise ValueError("Wall measures must be finite and positive.")
        if (
            not np.isfinite(density_)
            or density_ <= 0.0
            or not np.isfinite(viscosity)
            or viscosity <= 0.0
            or not np.isfinite(roughness)
            or roughness < 0.0
            or iterations <= 0
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
            or y_min < 0.0
            or y_max <= y_min
            or not np.isfinite(mach)
            or mach <= 0.0
            or not np.isfinite(clauser)
            or clauser < 0.0
        ):
            raise ValueError("Equilibrium wall model parameters are invalid.")
        self.density = density_
        self.kinematic_viscosity = viscosity
        self.wall_sample_height = jnp.asarray(height)
        self.wall_measure = jnp.asarray(measure)
        self.roughness_height = roughness
        self.root_iterations = iterations
        self.root_tolerance = tolerance
        self.minimum_y_plus = y_min
        self.maximum_y_plus = y_max
        self.maximum_mach = mach
        self.maximum_clauser_parameter = clauser
        self.plan_id = canonical_fingerprint(
            {
                "kind": "equilibrium-wall-vortex-closure",
                "density": density_.hex(),
                "kinematic_viscosity": viscosity.hex(),
                "height": array_tree_fingerprint(height),
                "measure": array_tree_fingerprint(measure),
                "roughness_height": roughness.hex(),
                "root_iterations": iterations,
                "root_tolerance": tolerance.hex(),
                "y_plus_envelope": (y_min, y_max),
                "maximum_mach": mach,
                "maximum_clauser_parameter": clauser,
            }
        )

    def evaluate(
        self,
        wall_velocity: ArrayLike,
        sample_velocity: ArrayLike,
        normal: ArrayLike,
        step_size: ArrayLike,
        /,
        *,
        mach: ArrayLike = 0.0,
        clauser_pressure_gradient: ArrayLike = 0.0,
    ) -> EquilibriumWallVortexClosureResult:
        wall = jnp.asarray(wall_velocity)
        sample = jnp.asarray(sample_velocity, dtype=wall.dtype)
        normal_ = jnp.asarray(normal, dtype=wall.dtype)
        dt = jnp.asarray(step_size, dtype=wall.dtype)
        if wall.shape != sample.shape or wall.shape != normal_.shape or wall.ndim != 2:
            raise ValueError("Wall/sample/normal arrays must be matching matrices.")
        if wall.shape[0] != self.wall_sample_height.size or wall.shape[1] not in (2, 3):
            raise ValueError("Wall closure arrays do not match prepared wall geometry.")
        normal_norm = jnp.sqrt(jnp.sum(normal_ * normal_, axis=-1))
        unit_normal = normal_ / jnp.where(normal_norm > 0.0, normal_norm, 1.0)[:, None]
        relative = sample - wall
        tangential = (
            relative - jnp.sum(relative * unit_normal, axis=-1)[:, None] * unit_normal
        )
        speed = jnp.sqrt(jnp.sum(tangential * tangential, axis=-1))
        height = self.wall_sample_height.astype(wall.dtype)
        viscosity = jnp.asarray(self.kinematic_viscosity, dtype=wall.dtype)
        lower = jnp.zeros_like(speed)
        viscous_estimate = jnp.sqrt(jnp.maximum(speed * viscosity / height, 0.0))
        upper = jnp.maximum(
            2.0 * jnp.maximum(speed, viscous_estimate), viscosity / height
        )
        for _ in range(self.root_iterations):
            middle = 0.5 * (lower + upper)
            y_plus = height * middle / viscosity
            predicted = middle * _reichardt_velocity_plus(
                y_plus, self.roughness_height / height
            )
            choose_upper = predicted >= speed
            upper = jnp.where(choose_upper, middle, upper)
            lower = jnp.where(choose_upper, lower, middle)
        friction_velocity = jnp.where(speed > 0.0, 0.5 * (lower + upper), 0.0)
        y_plus = height * friction_velocity / viscosity
        prediction = friction_velocity * _reichardt_velocity_plus(
            y_plus, self.roughness_height / height
        )
        residual = jnp.abs(prediction - speed) / jnp.maximum(speed, 1.0)
        direction = tangential / jnp.where(speed > 0.0, speed, 1.0)[:, None]
        traction = -self.density * friction_velocity[:, None] ** 2 * direction
        if wall.shape[1] == 2:
            cross = (
                unit_normal[:, 0] * traction[:, 1] - unit_normal[:, 1] * traction[:, 0]
            )
            strength = (
                self.wall_measure.astype(wall.dtype)
                * dt
                / (self.density * height)
                * cross
            )
        else:
            strength = (
                self.wall_measure.astype(wall.dtype)[:, None]
                * dt
                / (self.density * height[:, None])
                * jnp.cross(unit_normal, traction)
            )
        mach_ = jnp.broadcast_to(jnp.asarray(mach, dtype=wall.dtype), speed.shape)
        clauser = jnp.broadcast_to(
            jnp.asarray(clauser_pressure_gradient, dtype=wall.dtype), speed.shape
        )
        mach_margin = self.maximum_mach - jnp.abs(mach_)
        pressure_margin = self.maximum_clauser_parameter - jnp.abs(clauser)
        roughness_margin = height - self.roughness_height
        dissipation = -jnp.sum(traction * tangential, axis=-1)
        finite = (
            jnp.all(jnp.isfinite(wall), axis=-1)
            & jnp.all(jnp.isfinite(sample), axis=-1)
            & jnp.all(jnp.isfinite(normal_), axis=-1)
            & jnp.isfinite(friction_velocity)
            & jnp.isfinite(residual)
        )
        envelope = (
            (y_plus >= self.minimum_y_plus)
            & (y_plus <= self.maximum_y_plus)
            & (mach_margin >= 0.0)
            & (pressure_margin >= 0.0)
            & (roughness_margin > 0.0)
        )
        successful = (
            finite
            & envelope
            & (normal_norm > 0.0)
            & (residual <= self.root_tolerance)
            & (dissipation >= -self.root_tolerance)
            & (dt > 0.0)
        )
        derivative_valid = successful & (speed > self.root_tolerance)
        evidence = EquilibriumWallVortexClosureEvidence(
            residual,
            y_plus,
            mach_margin,
            pressure_margin,
            roughness_margin,
            dissipation,
            finite,
            derivative_valid,
            successful,
        )
        return EquilibriumWallVortexClosureResult(
            friction_velocity, traction, strength, evidence
        )


def _reichardt_velocity_plus(y_plus: Array, relative_roughness: Array, /) -> Array:
    kappa = 0.41
    smooth = jnp.log1p(kappa * y_plus) / kappa + 7.8 * (
        1.0 - jnp.exp(-y_plus / 11.0) - y_plus / 11.0 * jnp.exp(-y_plus / 3.0)
    )
    roughness_shift = jnp.log1p(jnp.maximum(relative_roughness * y_plus, 0.0)) / kappa
    return jnp.maximum(smooth - roughness_shift, 0.0)


class VortexLoadRecoveryEvidence(StrictModule):
    observed_order: Array
    asymptotic_ratio: Array
    pressure_impulse_discrepancy: Array
    circulation_defect: Array
    impulse_defect: Array
    time_stencil_defect: Array
    topology_correspondence: Array
    panel_residual: Array
    finite: Array
    recoverable: Array


class VortexLoadRecoveryResult(StrictModule):
    estimate: Array
    uncertainty: Array
    lower_bound: Array
    upper_bound: Array
    recoverable: Array
    evidence: VortexLoadRecoveryEvidence


class VortexLoadRecoveryPlan(StrictModule, NonTrainableState):
    resolutions: Array
    formal_order: float = eqx.field(static=True)
    safety_factor: float = eqx.field(static=True)
    consistency_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        resolutions: ArrayLike,
        formal_order: float,
        *,
        safety_factor: float = 1.25,
        consistency_tolerance: float = 0.25,
    ):
        values = np.asarray(resolutions, dtype=float)
        order = float(formal_order)
        safety = float(safety_factor)
        tolerance = float(consistency_tolerance)
        if (
            values.shape not in ((2,), (3,))
            or np.any(~np.isfinite(values))
            or np.any(values <= 0.0)
        ):
            raise ValueError("Load recovery requires two or three positive resolutions.")
        if np.any(np.diff(values) >= 0.0):
            raise ValueError("Load resolutions must be ordered coarse to fine.")
        if not np.isfinite(order) or order <= 0.0 or safety < 1.0 or tolerance <= 0.0:
            raise ValueError("Load recovery order/safety/tolerance is invalid.")
        self.resolutions = jnp.asarray(values)
        self.formal_order = order
        self.safety_factor = safety
        self.consistency_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "vortex-load-recovery",
                "resolutions": array_tree_fingerprint(values),
                "formal_order": order,
                "safety_factor": safety,
                "consistency_tolerance": tolerance,
            }
        )

    def evaluate(
        self,
        pressure_loads: ArrayLike,
        impulse_loads: ArrayLike,
        /,
        *,
        topology_correspondence: ArrayLike,
        circulation_defect: ArrayLike,
        impulse_defect: ArrayLike,
        time_stencil_defect: ArrayLike,
        panel_residual: ArrayLike,
    ) -> VortexLoadRecoveryResult:
        pressure = jnp.asarray(pressure_loads)
        impulse = jnp.asarray(impulse_loads, dtype=pressure.dtype)
        count = int(self.resolutions.size)
        if (
            pressure.ndim != 2
            or pressure.shape != impulse.shape
            or pressure.shape[0] != count
        ):
            raise ValueError(
                "Nested pressure/impulse loads must have resolution by load shape."
            )
        average = 0.5 * (pressure + impulse)
        ratio = self.resolutions[-2] / self.resolutions[-1]
        denominator = ratio**self.formal_order - 1.0
        estimate = average[-1] + (average[-1] - average[-2]) / denominator
        uncertainty = self.safety_factor * jnp.abs(estimate - average[-1])
        if count == 3:
            coarse_difference = jnp.linalg.norm(average[0] - average[1])
            fine_difference = jnp.linalg.norm(average[1] - average[2])
            resolution_ratio = self.resolutions[0] / self.resolutions[1]
            observed_order = jnp.log(
                jnp.maximum(
                    coarse_difference / jnp.maximum(fine_difference, 1.0e-30), 1.0e-30
                )
            ) / jnp.log(resolution_ratio)
            asymptotic_ratio = coarse_difference / jnp.maximum(
                fine_difference * resolution_ratio**self.formal_order, 1.0e-30
            )
        else:
            observed_order = jnp.asarray(self.formal_order, dtype=pressure.dtype)
            asymptotic_ratio = jnp.asarray(1.0, dtype=pressure.dtype)
        discrepancy = jnp.linalg.norm(pressure[-1] - impulse[-1])
        scale = jnp.maximum(jnp.linalg.norm(estimate), 1.0)
        circulation = jnp.asarray(circulation_defect, dtype=pressure.dtype)
        impulse_defect_ = jnp.asarray(impulse_defect, dtype=pressure.dtype)
        time_defect = jnp.asarray(time_stencil_defect, dtype=pressure.dtype)
        panel = jnp.asarray(panel_residual, dtype=pressure.dtype)
        topology = jnp.asarray(topology_correspondence, dtype=bool)
        finite = (
            jnp.all(jnp.isfinite(pressure))
            & jnp.all(jnp.isfinite(impulse))
            & jnp.all(jnp.isfinite(estimate))
            & jnp.all(jnp.isfinite(uncertainty))
            & jnp.isfinite(observed_order)
        )
        order_consistent = jnp.abs(observed_order - self.formal_order) <= max(
            self.consistency_tolerance * self.formal_order,
            self.consistency_tolerance,
        )
        estimator_consistent = (
            jnp.abs(asymptotic_ratio - 1.0) <= self.consistency_tolerance
        )
        defects_valid = (
            jnp.maximum(
                jnp.maximum(jnp.abs(circulation), jnp.abs(impulse_defect_)),
                jnp.maximum(jnp.abs(time_defect), jnp.abs(panel)),
            )
            <= self.consistency_tolerance * scale
        )
        recoverable = (
            finite
            & topology
            & order_consistent
            & estimator_consistent
            & defects_valid
            & (discrepancy <= self.consistency_tolerance * scale)
        )
        evidence = VortexLoadRecoveryEvidence(
            observed_order,
            asymptotic_ratio,
            discrepancy,
            circulation,
            impulse_defect_,
            time_defect,
            topology,
            panel,
            finite,
            recoverable,
        )
        return VortexLoadRecoveryResult(
            estimate,
            uncertainty,
            estimate - uncertainty,
            estimate + uncertainty,
            recoverable,
            evidence,
        )


class CompressibleVortexState(StrictModule):
    density: Array
    momentum: Array
    total_energy: Array
    vortex_velocity: Array
    accepted_coupling_state: Array


class CompressibleVortexAugmentationEvidence(StrictModule):
    divergence_defect: Array
    vorticity_defect: Array
    momentum_defect: Array
    total_energy_defect: Array
    internal_energy_defect: Array
    projection_defect: Array
    transfer_defect: Array
    maximum_mach: Array
    finite: Array
    derivative_valid: Array
    successful: Array


class CompressibleVortexAugmentationResult(StrictModule):
    candidate_state: CompressibleVortexState
    accepted_state: CompressibleVortexState
    solenoidal_velocity: Array
    dilatational_velocity: Array
    baroclinic_source: Array
    evidence: CompressibleVortexAugmentationEvidence
    successful: Array


class CompressibleVortexAugmentationPlan(StrictModule, NonTrainableState):
    solenoidal_projection: Array
    dilatational_projection: Array
    divergence_operator: Array
    curl_operator: Array
    minimum_density: float = eqx.field(static=True)
    minimum_internal_energy: float = eqx.field(static=True)
    maximum_mach: float = eqx.field(static=True)
    projection_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        solenoidal_projection: ArrayLike,
        dilatational_projection: ArrayLike,
        divergence_operator: ArrayLike,
        curl_operator: ArrayLike,
        *,
        minimum_density: float,
        minimum_internal_energy: float,
        maximum_mach: float,
        projection_tolerance: float = 1.0e-8,
    ):
        solenoidal = np.asarray(solenoidal_projection, dtype=float)
        dilatational = np.asarray(dilatational_projection, dtype=float)
        divergence = np.asarray(divergence_operator, dtype=float)
        curl = np.asarray(curl_operator, dtype=float)
        if (
            solenoidal.ndim != 2
            or solenoidal.shape[0] != solenoidal.shape[1]
            or dilatational.shape != solenoidal.shape
            or divergence.ndim != 2
            or divergence.shape[1] != solenoidal.shape[0]
            or curl.ndim != 2
            or curl.shape[1] != solenoidal.shape[0]
        ):
            raise ValueError(
                "Compressible vortex projection/operator shapes are invalid."
            )
        density = float(minimum_density)
        internal = float(minimum_internal_energy)
        mach = float(maximum_mach)
        tolerance = float(projection_tolerance)
        if density <= 0.0 or internal <= 0.0 or mach <= 0.0 or tolerance <= 0.0:
            raise ValueError("Compressible vortex admissibility bounds are invalid.")
        identity = np.eye(solenoidal.shape[0])
        defect = max(
            float(np.linalg.norm(solenoidal @ solenoidal - solenoidal)),
            float(np.linalg.norm(dilatational @ dilatational - dilatational)),
            float(np.linalg.norm(solenoidal + dilatational - identity)),
            float(np.linalg.norm(solenoidal @ dilatational)),
        )
        if not np.isfinite(defect) or defect > tolerance:
            raise ValueError("Helmholtz projections fail complementarity/idempotence.")
        self.solenoidal_projection = jnp.asarray(solenoidal)
        self.dilatational_projection = jnp.asarray(dilatational)
        self.divergence_operator = jnp.asarray(divergence)
        self.curl_operator = jnp.asarray(curl)
        self.minimum_density = density
        self.minimum_internal_energy = internal
        self.maximum_mach = mach
        self.projection_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "compressible-vortex-augmentation",
                "solenoidal": array_tree_fingerprint(solenoidal),
                "dilatational": array_tree_fingerprint(dilatational),
                "divergence": array_tree_fingerprint(divergence),
                "curl": array_tree_fingerprint(curl),
                "minimum_density": density,
                "minimum_internal_energy": internal,
                "maximum_mach": mach,
                "projection_tolerance": tolerance,
            }
        )

    def prepare(self, /) -> PreparedCompressibleVortexAugmentation:
        return PreparedCompressibleVortexAugmentation(self, self.plan_id)


class PreparedCompressibleVortexAugmentation(StrictModule, NonTrainableState):
    plan: CompressibleVortexAugmentationPlan
    prepared_id: str = eqx.field(static=True)

    def evaluate(
        self,
        state: CompressibleVortexState,
        vortex_solenoidal_velocity: ArrayLike,
        sound_speed: ArrayLike,
        /,
        *,
        density_gradient: ArrayLike | None = None,
        pressure_gradient: ArrayLike | None = None,
    ) -> CompressibleVortexAugmentationResult:
        if not isinstance(state, CompressibleVortexState):
            raise TypeError("state must be CompressibleVortexState.")
        density = jnp.asarray(state.density)
        momentum = jnp.asarray(state.momentum, dtype=density.dtype)
        total_energy = jnp.asarray(state.total_energy, dtype=density.dtype)
        vortex = jnp.asarray(vortex_solenoidal_velocity, dtype=density.dtype)
        if (
            momentum.shape != vortex.shape
            or momentum.ndim != 2
            or density.shape != momentum.shape[:1]
            or total_energy.shape != density.shape
        ):
            raise ValueError("Compressible vortex state arrays have incompatible shapes.")
        flattened_size = momentum.size
        if self.plan.solenoidal_projection.shape != (flattened_size, flattened_size):
            raise ValueError(
                "Projection size does not match the compressible velocity grid."
            )
        velocity = momentum / jnp.where(density > 0.0, density, 1.0)[:, None]
        flattened = velocity.reshape((-1,))
        old_solenoidal = (self.plan.solenoidal_projection @ flattened).reshape(
            momentum.shape
        )
        dilatational = (self.plan.dilatational_projection @ flattened).reshape(
            momentum.shape
        )
        vortex_flat = vortex.reshape((-1,))
        projected_vortex = (self.plan.solenoidal_projection @ vortex_flat).reshape(
            momentum.shape
        )
        composed = projected_vortex + dilatational
        total_mass = jnp.sum(density)
        raw_momentum = density[:, None] * composed
        mean_correction = (
            jnp.sum(momentum, axis=0) - jnp.sum(raw_momentum, axis=0)
        ) / jnp.where(total_mass > 0.0, total_mass, 1.0)
        solenoidal_with_mean = projected_vortex + mean_correction
        composed = solenoidal_with_mean + dilatational
        old_kinetic = 0.5 * jnp.sum(momentum * velocity, axis=-1)
        internal_energy = total_energy - old_kinetic
        candidate_momentum = density[:, None] * composed
        new_kinetic = 0.5 * density * jnp.sum(composed * composed, axis=-1)
        candidate_energy = internal_energy + new_kinetic
        sound = jnp.broadcast_to(
            jnp.asarray(sound_speed, dtype=density.dtype), density.shape
        )
        local_mach = jnp.sqrt(jnp.sum(composed * composed, axis=-1)) / jnp.where(
            sound > 0.0, sound, 1.0
        )
        divergence_defect = jnp.linalg.norm(
            self.plan.divergence_operator @ solenoidal_with_mean.reshape((-1,))
        )
        vorticity_defect = jnp.linalg.norm(
            self.plan.curl_operator @ dilatational.reshape((-1,))
        )
        projection_defect = jnp.linalg.norm(vortex - projected_vortex)
        momentum_defect = jnp.sum(candidate_momentum, axis=0) - jnp.sum(momentum, axis=0)
        internal_defect = candidate_energy - new_kinetic - internal_energy
        total_energy_defect = jnp.sum(candidate_energy - total_energy) - jnp.sum(
            new_kinetic - old_kinetic
        )
        transfer_defect = jnp.linalg.norm(old_solenoidal - state.vortex_velocity)
        if density_gradient is None or pressure_gradient is None:
            baroclinic = jnp.zeros_like(vortex)
        else:
            density_gradient_ = jnp.asarray(density_gradient, dtype=density.dtype)
            pressure_gradient_ = jnp.asarray(pressure_gradient, dtype=density.dtype)
            if (
                density_gradient_.shape != momentum.shape
                or pressure_gradient_.shape != momentum.shape
            ):
                raise ValueError("Baroclinic gradients must match the velocity grid.")
            if momentum.shape[1] == 2:
                scalar = (
                    density_gradient_[:, 0] * pressure_gradient_[:, 1]
                    - density_gradient_[:, 1] * pressure_gradient_[:, 0]
                ) / jnp.where(density > 0.0, density**2, 1.0)
                baroclinic = jnp.stack((jnp.zeros_like(scalar), scalar), axis=-1)
            else:
                baroclinic = jnp.cross(density_gradient_, pressure_gradient_) / jnp.where(
                    density[:, None] > 0.0, density[:, None] ** 2, 1.0
                )
        scale = jnp.maximum(jnp.linalg.norm(momentum), 1.0)
        tolerance = self.plan.projection_tolerance
        finite = (
            jnp.all(jnp.isfinite(density))
            & jnp.all(jnp.isfinite(candidate_momentum))
            & jnp.all(jnp.isfinite(candidate_energy))
            & jnp.all(jnp.isfinite(baroclinic))
            & jnp.all(sound > 0.0)
        )
        admissible = (
            jnp.all(density >= self.plan.minimum_density)
            & jnp.all(internal_energy >= self.plan.minimum_internal_energy)
            & jnp.all(local_mach <= self.plan.maximum_mach)
        )
        successful = (
            finite
            & admissible
            & (divergence_defect <= tolerance)
            & (vorticity_defect <= tolerance)
            & (projection_defect <= tolerance * scale)
            & (jnp.linalg.norm(momentum_defect) <= tolerance * scale)
            & (jnp.linalg.norm(internal_defect) <= tolerance * scale)
            & (jnp.abs(total_energy_defect) <= tolerance * scale)
        )
        candidate = CompressibleVortexState(
            density,
            candidate_momentum,
            candidate_energy,
            solenoidal_with_mean,
            state.accepted_coupling_state + 1,
        )
        accepted = CompressibleVortexState(
            jnp.where(successful, candidate.density, state.density),
            jnp.where(successful, candidate.momentum, state.momentum),
            jnp.where(successful, candidate.total_energy, state.total_energy),
            jnp.where(successful, candidate.vortex_velocity, state.vortex_velocity),
            jnp.where(
                successful,
                candidate.accepted_coupling_state,
                state.accepted_coupling_state,
            ),
        )
        derivative_valid = successful & jnp.all(local_mach < self.plan.maximum_mach)
        evidence = CompressibleVortexAugmentationEvidence(
            divergence_defect,
            vorticity_defect,
            momentum_defect,
            total_energy_defect,
            jnp.linalg.norm(internal_defect),
            projection_defect,
            transfer_defect,
            jnp.max(local_mach),
            finite,
            derivative_valid,
            successful,
        )
        return CompressibleVortexAugmentationResult(
            candidate,
            accepted,
            solenoidal_with_mean,
            dilatational,
            baroclinic,
            evidence,
            successful,
        )


__all__ = [
    "CompressibleVortexAugmentationEvidence",
    "CompressibleVortexAugmentationPlan",
    "CompressibleVortexAugmentationResult",
    "CompressibleVortexState",
    "EquilibriumWallVortexClosureEvidence",
    "EquilibriumWallVortexClosurePlan",
    "EquilibriumWallVortexClosureResult",
    "PreparedCompressibleVortexAugmentation",
    "VortexLoadRecoveryEvidence",
    "VortexLoadRecoveryPlan",
    "VortexLoadRecoveryResult",
]
