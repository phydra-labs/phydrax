#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...optim._robust_losses import AbstractRobustLoss, HuberLoss
from ..camera._rig import CameraRig
from ..imaging._photometry import (
    CameraStackRenderResult,
    ParticleImageFormation,
    render_camera_stack,
)
from ..imaging._types import ImageGeometry2D


SHAKE_SUCCESS = 0
SHAKE_NOT_IMPROVED = 1
SHAKE_NONFINITE = 2


class ShakePlan(StrictModule, NonTrainableState):
    """Bounded continuous refinement with frozen particle support."""

    iterations: int = eqx.field(static=True)
    position_step: float = eqx.field(static=True)
    amplitude_step: float = eqx.field(static=True)
    gradient_clip: float = eqx.field(static=True)
    maximum_displacement: float = eqx.field(static=True)
    minimum_amplitude: float = eqx.field(static=True)
    convergence_tolerance: float = eqx.field(static=True)
    loss: AbstractRobustLoss
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        iterations: int = 8,
        position_step: float = 1.0e-3,
        amplitude_step: float = 1.0e-2,
        gradient_clip: float = 100.0,
        maximum_displacement: float = 1.0,
        minimum_amplitude: float = 0.0,
        convergence_tolerance: float = 1.0e-8,
        loss: AbstractRobustLoss | None = None,
    ):
        iterations_ = int(iterations)
        scalars = tuple(
            float(value)
            for value in (
                position_step,
                amplitude_step,
                gradient_clip,
                maximum_displacement,
                convergence_tolerance,
            )
        )
        minimum_amplitude_ = float(minimum_amplitude)
        if iterations_ <= 0:
            raise ValueError("iterations must be positive.")
        if any(not isfinite(value) or value <= 0.0 for value in scalars):
            raise ValueError(
                "Shake steps, bounds, and tolerance must be finite and positive."
            )
        if not isfinite(minimum_amplitude_) or minimum_amplitude_ < 0.0:
            raise ValueError("minimum_amplitude must be finite and non-negative.")
        loss_ = HuberLoss(1.0) if loss is None else loss
        if not isinstance(loss_, AbstractRobustLoss):
            raise TypeError("loss must be an AbstractRobustLoss or None.")
        self.iterations = iterations_
        self.position_step = scalars[0]
        self.amplitude_step = scalars[1]
        self.gradient_clip = scalars[2]
        self.maximum_displacement = scalars[3]
        self.minimum_amplitude = minimum_amplitude_
        self.convergence_tolerance = scalars[4]
        self.loss = loss_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "shake-particle-refinement",
                "iterations": iterations_,
                "position_step": scalars[0],
                "amplitude_step": scalars[1],
                "gradient_clip": scalars[2],
                "maximum_displacement": scalars[3],
                "minimum_amplitude": minimum_amplitude_,
                "convergence_tolerance": scalars[4],
                "loss_id": loss_.loss_id,
            }
        )


class ShakeResult(StrictModule):
    """Refined continuous particle state and robust residual history."""

    positions_xyz: Array
    amplitude: Array
    active: Array
    initial_render: CameraStackRenderResult
    final_render: CameraStackRenderResult
    initial_residual: Array
    residual: Array
    loss_history: Array
    accepted_steps: Array
    converged: Array
    status: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def robust_image_loss(
    predicted: ArrayLike,
    observed: ArrayLike,
    valid_mask: ArrayLike,
    loss: AbstractRobustLoss,
    /,
) -> Array:
    """Mean pixelwise robust loss on an explicit image mask."""
    predicted_ = jnp.asarray(predicted)
    observed_ = jnp.asarray(observed, dtype=predicted_.dtype)
    valid_ = jnp.asarray(valid_mask, dtype=bool)
    if predicted_.shape != observed_.shape or predicted_.shape != valid_.shape:
        raise ValueError("predicted, observed, and valid_mask must have equal shapes.")
    residual = jnp.where(valid_, observed_ - predicted_, 0.0)
    rho = loss.evaluate(residual * residual).rho
    count = jnp.maximum(jnp.sum(valid_, dtype=predicted_.dtype), 1.0)
    return jnp.sum(jnp.where(valid_, rho, 0.0)) / count


def shake_particles(
    plan: ShakePlan,
    formation: ParticleImageFormation,
    rig: CameraRig,
    geometry: ImageGeometry2D,
    observed_images: ArrayLike,
    positions_xyz: ArrayLike,
    amplitude: ArrayLike,
    sigma: ArrayLike,
    active: ArrayLike,
    *,
    valid_mask: ArrayLike | None = None,
) -> ShakeResult:
    """Refine positions and amplitudes without changing particle topology."""
    if not isinstance(plan, ShakePlan):
        raise TypeError("plan must be ShakePlan.")
    if not isinstance(formation, ParticleImageFormation):
        raise TypeError("formation must be ParticleImageFormation.")
    if formation.response.stochastic:
        raise ValueError("Shake requires a deterministic photometric response.")
    if not isinstance(rig, CameraRig):
        raise TypeError("rig must be CameraRig.")
    if not isinstance(geometry, ImageGeometry2D):
        raise TypeError("geometry must be ImageGeometry2D.")
    positions = jnp.asarray(positions_xyz)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_xyz must have shape (particle_capacity, 3).")
    if not jnp.issubdtype(positions.dtype, jnp.inexact):
        positions = positions.astype(float)
    amplitudes = jnp.asarray(amplitude, dtype=positions.dtype)
    active_ = jnp.asarray(active, dtype=bool)
    capacity = int(positions.shape[0])
    if amplitudes.shape != (capacity,) or active_.shape != (capacity,):
        raise ValueError("amplitude and active must match particle capacity.")
    invalid_input = active_ & (
        ~jnp.all(jnp.isfinite(positions), axis=-1)
        | ~jnp.isfinite(amplitudes)
        | (amplitudes < 0.0)
    )
    active_ = active_ & ~invalid_input
    positions = jnp.where(active_[:, None], positions, 0.0)
    amplitudes = jnp.where(active_, amplitudes, 0.0)
    observed = jnp.asarray(observed_images, dtype=positions.dtype)
    expected_shape = (rig.capacity,) + geometry.image_shape
    if observed.shape != expected_shape:
        raise ValueError(
            "observed_images must have shape (camera_capacity, rows, columns)."
        )
    finite_observed = jnp.isfinite(observed)
    valid = (
        finite_observed
        if valid_mask is None
        else finite_observed & jnp.asarray(valid_mask, dtype=bool)
    )
    if valid.shape != observed.shape:
        raise ValueError("valid_mask must have the observed image shape.")
    valid = valid & rig.camera_valid[:, None, None]
    safe_observed = jnp.where(valid, observed, 0.0)

    def objective(position_value: Array, amplitude_value: Array):
        rendered = render_camera_stack(
            formation,
            rig,
            geometry,
            position_value,
            amplitude_value,
            sigma,
            active_,
        )
        return robust_image_loss(rendered.images, safe_observed, valid, plan.loss)

    initial_render = render_camera_stack(
        formation, rig, geometry, positions, amplitudes, sigma, active_
    )
    initial_loss = robust_image_loss(
        initial_render.images, safe_observed, valid, plan.loss
    )
    initial_positions = positions

    def refine_one(carry: tuple[Array, Array, Array], unused):
        current_positions, current_amplitudes, current_loss = carry
        loss_value, gradients = jax.value_and_grad(objective, argnums=(0, 1))(
            current_positions, current_amplitudes
        )
        position_gradient, amplitude_gradient = gradients
        position_gradient = jnp.clip(
            position_gradient, -plan.gradient_clip, plan.gradient_clip
        )
        amplitude_gradient = jnp.clip(
            amplitude_gradient, -plan.gradient_clip, plan.gradient_clip
        )
        proposed_positions = current_positions - plan.position_step * position_gradient
        displacement = proposed_positions - initial_positions
        displacement_norm = jnp.sqrt(
            jnp.sum(displacement * displacement, axis=-1, keepdims=True)
        )
        displacement_scale = jnp.minimum(
            1.0,
            plan.maximum_displacement / jnp.maximum(displacement_norm, 1.0e-30),
        )
        proposed_positions = initial_positions + displacement * displacement_scale
        proposed_amplitudes = jnp.maximum(
            current_amplitudes - plan.amplitude_step * amplitude_gradient,
            plan.minimum_amplitude,
        )
        proposed_positions = jnp.where(
            active_[:, None], proposed_positions, current_positions
        )
        proposed_amplitudes = jnp.where(active_, proposed_amplitudes, current_amplitudes)
        proposed_loss = objective(proposed_positions, proposed_amplitudes)
        finite_proposal = (
            jnp.isfinite(proposed_loss)
            & jnp.all(jnp.isfinite(proposed_positions))
            & jnp.all(jnp.isfinite(proposed_amplitudes))
        )
        accept = finite_proposal & (proposed_loss <= loss_value)
        next_positions = jnp.where(accept, proposed_positions, current_positions)
        next_amplitudes = jnp.where(accept, proposed_amplitudes, current_amplitudes)
        next_loss = jnp.where(accept, proposed_loss, current_loss)
        return (next_positions, next_amplitudes, next_loss), (next_loss, accept)

    (final_positions, final_amplitudes, final_loss), history = jax.lax.scan(
        refine_one,
        (positions, amplitudes, initial_loss),
        xs=None,
        length=plan.iterations,
    )
    step_losses, accepted = history
    loss_history = jnp.concatenate((initial_loss[None], step_losses), axis=0)
    final_render = render_camera_stack(
        formation, rig, geometry, final_positions, final_amplitudes, sigma, active_
    )
    initial_residual = jnp.where(valid, safe_observed - initial_render.images, 0.0)
    residual = jnp.where(valid, safe_observed - final_render.images, 0.0)
    finite_result = (
        jnp.isfinite(final_loss)
        & jnp.all(jnp.isfinite(final_positions))
        & jnp.all(jnp.isfinite(final_amplitudes))
        & ~jnp.any(invalid_input)
    )
    improved = final_loss < initial_loss
    converged = (initial_loss - final_loss) <= plan.convergence_tolerance
    status = jnp.where(
        ~finite_result,
        SHAKE_NONFINITE,
        jnp.where(improved | converged, SHAKE_SUCCESS, SHAKE_NOT_IMPROVED),
    ).astype(jnp.int32)
    return ShakeResult(
        final_positions,
        final_amplitudes,
        active_,
        initial_render,
        final_render,
        initial_residual,
        residual,
        loss_history,
        jnp.sum(accepted, dtype=jnp.int32),
        converged,
        status,
        finite_result & (improved | converged),
        plan.plan_id,
    )


__all__ = [
    "SHAKE_NONFINITE",
    "SHAKE_NOT_IMPROVED",
    "SHAKE_SUCCESS",
    "ShakePlan",
    "ShakeResult",
    "robust_image_loss",
    "shake_particles",
]
