#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


class GLAMMPlan(StrictModule, NonTrainableState):
    contour_nodes: int = eqx.field(static=True)
    time_step: float = eqx.field(static=True)
    plateau_modulus: float = eqx.field(static=True)
    disengagement_time: float = eqx.field(static=True)
    rouse_time: float = eqx.field(static=True)
    contour_diffusivity: float = eqx.field(static=True)
    convective_constraint_release: float = eqx.field(static=True)
    maximum_stability_number: float = eqx.field(static=True)
    incompressibility_tolerance: float = eqx.field(static=True)
    model_variant: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        contour_nodes: int,
        time_step: float,
        plateau_modulus: float,
        disengagement_time: float,
        rouse_time: float,
        /,
        *,
        contour_diffusivity: float = 1.0,
        convective_constraint_release: float = 0.0,
        maximum_stability_number: float = 0.25,
        incompressibility_tolerance: float = 1.0e-10,
    ):
        if int(contour_nodes) < 3:
            raise ValueError("contour_nodes must be at least three.")
        positive = (time_step, plateau_modulus, disengagement_time, rouse_time)
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive
        ):
            raise ValueError(
                "GLaMM time and material scales must be finite and positive."
            )
        if (
            not math.isfinite(float(contour_diffusivity))
            or float(contour_diffusivity) < 0.0
        ):
            raise ValueError("contour_diffusivity must be finite and nonnegative.")
        if (
            not math.isfinite(float(convective_constraint_release))
            or float(convective_constraint_release) < 0.0
        ):
            raise ValueError(
                "convective_constraint_release must be finite and nonnegative."
            )
        if (
            not math.isfinite(float(maximum_stability_number))
            or not 0.0 < float(maximum_stability_number) <= 1.0
        ):
            raise ValueError("maximum_stability_number must lie in (0, 1].")
        if (
            not math.isfinite(float(incompressibility_tolerance))
            or float(incompressibility_tolerance) < 0.0
        ):
            raise ValueError(
                "incompressibility_tolerance must be finite and nonnegative."
            )
        self.contour_nodes = int(contour_nodes)
        self.time_step = float(time_step)
        self.plateau_modulus = float(plateau_modulus)
        self.disengagement_time = float(disengagement_time)
        self.rouse_time = float(rouse_time)
        self.contour_diffusivity = float(contour_diffusivity)
        self.convective_constraint_release = float(convective_constraint_release)
        self.maximum_stability_number = float(maximum_stability_number)
        self.incompressibility_tolerance = float(incompressibility_tolerance)
        self.model_variant = "contour_tensor_glamm"
        self.plan_id = canonical_fingerprint(
            {
                "kind": "glamm",
                "variant": self.model_variant,
                "nodes": self.contour_nodes,
                "dt": self.time_step,
                "plateau": self.plateau_modulus,
                "tau_d": self.disengagement_time,
                "tau_r": self.rouse_time,
                "diffusivity": self.contour_diffusivity,
                "ccr": self.convective_constraint_release,
            }
        )

    def initialize(self, *, dtype=jnp.float64) -> GLAMMState:
        identity = jnp.eye(3, dtype=dtype)
        conformation = jnp.broadcast_to(identity, (self.contour_nodes, 3, 3))
        return GLAMMState(
            conformation,
            jnp.ones((self.contour_nodes,), dtype=dtype),
            jnp.asarray(0.0, dtype=dtype),
            jnp.asarray(0, dtype=jnp.int64),
            jnp.asarray(True),
            self.plan_id,
        )


class GLAMMState(StrictModule):
    conformation: Array
    stretch: Array
    time: Array
    step_index: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class GLAMMStepResult(StrictModule):
    candidate_state: GLAMMState
    accepted_state: GLAMMState
    cauchy_stress: Array
    flow_power_density: Array
    stability_number: Array
    minimum_conformation_eigenvalue: Array
    incompressibility_residual: Array
    accepted: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def _contour_laplacian(values: Array) -> Array:
    padded = jnp.concatenate((values[:1], values, values[-1:]), axis=0)
    return padded[2:] - 2.0 * padded[1:-1] + padded[:-2]


def _rhs(
    plan: GLAMMPlan,
    conformation: Array,
    stretch: Array,
    velocity_gradient: Array,
) -> tuple[Array, Array, Array]:
    identity = jnp.eye(3, dtype=conformation.dtype)
    rate = 0.5 * (velocity_gradient + velocity_gradient.T)
    rate_norm = jnp.sqrt(2.0 * jnp.sum(rate * rate))
    ccr_rate = plan.convective_constraint_release * rate_norm
    upper = contract("ij,njk->nik", velocity_gradient, conformation) + contract(
        "nij,kj->nik", conformation, velocity_gradient
    )
    diffusion_scale = plan.contour_diffusivity * (plan.contour_nodes - 1) ** 2
    conformation_rhs = (
        upper
        - (conformation - identity) / plan.disengagement_time
        + diffusion_scale * _contour_laplacian(conformation)
        - ccr_rate * (conformation - identity)
    )
    alignment = contract("ij,nij->n", rate, conformation) / jnp.maximum(
        jnp.trace(conformation, axis1=-2, axis2=-1),
        jnp.finfo(conformation.dtype).tiny,
    )
    stretch_rhs = (
        stretch * alignment
        - (stretch - 1.0) / plan.rouse_time
        - ccr_rate * (stretch - 1.0)
    )
    stability = plan.time_step * (
        jnp.sqrt(jnp.sum(velocity_gradient * velocity_gradient))
        + 1.0 / plan.disengagement_time
        + 1.0 / plan.rouse_time
        + 4.0 * diffusion_scale
        + ccr_rate
    )
    return conformation_rhs, stretch_rhs, stability


def glamm_step(
    plan: GLAMMPlan,
    state: GLAMMState,
    velocity_gradient: ArrayLike,
    /,
) -> GLAMMStepResult:
    if not isinstance(plan, GLAMMPlan) or not isinstance(state, GLAMMState):
        raise TypeError("plan and state must be GLAMMPlan and GLAMMState.")
    if state.plan_id != plan.plan_id:
        raise ValueError("GLaMM state does not belong to this plan.")
    gradient = jnp.asarray(velocity_gradient, dtype=state.conformation.dtype)
    if gradient.shape != (3, 3):
        raise ValueError("velocity_gradient must have shape (3, 3).")
    conformation_rhs, stretch_rhs, stability = _rhs(
        plan, state.conformation, state.stretch, gradient
    )
    predictor_conformation = state.conformation + plan.time_step * conformation_rhs
    predictor_stretch = state.stretch + plan.time_step * stretch_rhs
    predictor_rhs, predictor_stretch_rhs, _ = _rhs(
        plan, predictor_conformation, predictor_stretch, gradient
    )
    candidate_conformation = state.conformation + 0.5 * plan.time_step * (
        conformation_rhs + predictor_rhs
    )
    candidate_conformation = 0.5 * (
        candidate_conformation + jnp.swapaxes(candidate_conformation, -1, -2)
    )
    candidate_stretch = state.stretch + 0.5 * plan.time_step * (
        stretch_rhs + predictor_stretch_rhs
    )
    candidate = GLAMMState(
        candidate_conformation,
        candidate_stretch,
        state.time + plan.time_step,
        state.step_index + 1,
        state.successful,
        plan.plan_id,
    )
    eigenvalues = jnp.linalg.eigvalsh(candidate_conformation)
    minimum_eigenvalue = jnp.min(eigenvalues)
    incompressibility = jnp.abs(jnp.trace(gradient))
    finite = (
        jnp.all(jnp.isfinite(candidate_conformation))
        & jnp.all(jnp.isfinite(candidate_stretch))
        & jnp.isfinite(stability)
    )
    accepted = (
        state.successful
        & finite
        & (minimum_eigenvalue > 0.0)
        & jnp.all(candidate_stretch > 0.0)
        & (stability <= plan.maximum_stability_number)
        & (incompressibility <= plan.incompressibility_tolerance)
    )
    candidate = GLAMMState(
        candidate.conformation,
        candidate.stretch,
        candidate.time,
        candidate.step_index,
        state.successful & accepted,
        plan.plan_id,
    )
    accepted_state = GLAMMState(
        jnp.where(accepted, candidate.conformation, state.conformation),
        jnp.where(accepted, candidate.stretch, state.stretch),
        jnp.where(accepted, candidate.time, state.time),
        jnp.where(accepted, candidate.step_index, state.step_index),
        state.successful,
        plan.plan_id,
    )
    identity = jnp.eye(3, dtype=state.conformation.dtype)
    stress = plan.plateau_modulus * jnp.mean(
        accepted_state.stretch[:, None, None] ** 2
        * (accepted_state.conformation - identity),
        axis=0,
    )
    rate = 0.5 * (gradient + gradient.T)
    power = jnp.sum(stress * rate)
    successful = accepted & jnp.all(jnp.isfinite(stress))
    return GLAMMStepResult(
        candidate,
        accepted_state,
        stress,
        power,
        stability,
        minimum_eigenvalue,
        incompressibility,
        accepted,
        successful,
        plan.plan_id,
    )


__all__ = ["GLAMMPlan", "GLAMMState", "GLAMMStepResult", "glamm_step"]
