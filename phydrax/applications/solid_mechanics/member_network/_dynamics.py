#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....linalg import DenseLinearOperator, LinearSystem, solve


class MemberDynamicState(StrictModule):
    displacement: Array
    velocity: Array
    acceleration: Array
    kinetic_energy: Array
    strain_energy: Array
    damping_dissipation: Array


class TractionVelocityPortHistory(StrictModule, NonTrainableState):
    """Midpoint traction–velocity samples for one explicit outgoing-work port.

    ``outgoing_traction`` is the traction exerted by the analyzed structure on its
    environment. Its dot product with boundary velocity is therefore positive for
    mechanical energy leaving the structure.
    """

    outgoing_traction: Array
    velocity: Array
    quadrature_weights: Array
    time_steps: Array
    port_id: str = eqx.field(static=True)

    def __init__(
        self,
        outgoing_traction: ArrayLike,
        velocity: ArrayLike,
        quadrature_weights: ArrayLike,
        time_steps: ArrayLike,
        /,
        *,
        port_id: str,
    ):
        traction = jnp.asarray(outgoing_traction)
        velocity_ = jnp.asarray(velocity, dtype=traction.dtype)
        weights = jnp.asarray(quadrature_weights, dtype=traction.dtype)
        steps = jnp.asarray(time_steps, dtype=traction.dtype)
        identifier = str(port_id)
        if traction.ndim != 3 or velocity_.shape != traction.shape:
            raise ValueError(
                "Traction and velocity must have shape (interval, point, dimension)."
            )
        if weights.shape != (traction.shape[1],):
            raise ValueError("quadrature_weights must contain one value per port point.")
        if steps.shape != (traction.shape[0],):
            raise ValueError("time_steps must contain one value per history interval.")
        if not identifier:
            raise ValueError("port_id must be non-empty.")
        if bool(
            jnp.any(~jnp.isfinite(traction))
            | jnp.any(~jnp.isfinite(velocity_))
            | jnp.any(~jnp.isfinite(weights) | (weights <= 0.0))
            | jnp.any(~jnp.isfinite(steps) | (steps <= 0.0))
        ):
            raise ValueError(
                "Traction–velocity port samples must be finite with positive measures."
            )
        self.outgoing_traction = traction
        self.velocity = velocity_
        self.quadrature_weights = weights
        self.time_steps = steps
        self.port_id = identifier

    @property
    def outgoing_work(self) -> Array:
        power = contract(
            "ipd,ipd,p->i",
            self.outgoing_traction,
            self.velocity,
            self.quadrature_weights,
        )
        return self.time_steps * power


class MemberEnergyWorkEvidence(StrictModule):
    """Accepted-history mechanical-energy closure over fixed physical epochs."""

    kinetic_change: Array
    potential_change: Array
    damping_work: Array
    material_contact_work: Array
    external_work: Array
    outgoing_port_work: Array
    algorithmic_defect: Array
    traction_velocity_port_id: str | None = eqx.field(static=True)
    balance_scale: Array
    interval_accepted: Array
    topology_consistent: Array
    contact_consistent: Array
    fracture_consistent: Array
    unilateral_consistent: Array
    mode_consistent: Array
    epoch_consistent: Array
    interval_available: Array
    interval_balanced: Array
    maximum_relative_defect: Array
    available: Array
    finite: Array
    balanced: Array


def member_energy_work_evidence(
    kinetic_energy: ArrayLike,
    potential_energy: ArrayLike,
    damping_work: ArrayLike,
    material_contact_work: ArrayLike,
    external_work: ArrayLike,
    /,
    *,
    accepted: ArrayLike,
    topology_epoch: ArrayLike,
    contact_epoch: ArrayLike,
    fracture_epoch: ArrayLike,
    unilateral_epoch: ArrayLike,
    mode_epoch: ArrayLike,
    traction_velocity_port: TractionVelocityPortHistory | None = None,
    tolerance: float = 1.0e-8,
) -> MemberEnergyWorkEvidence:
    """Audit ΔK + ΔΠ + Wdamp + Wmaterial/contact + Wout − Wext.

    Every state sample must be accepted. Changes to topology, contact/search,
    fracture, unilateral activation, or mode-selection epochs make the complete
    evidence unavailable; they are never interpreted as a balanced physical step.
    """
    kinetic = jnp.asarray(kinetic_energy)
    potential = jnp.asarray(potential_energy, dtype=kinetic.dtype)
    if kinetic.ndim != 1 or kinetic.size < 2 or potential.shape != kinetic.shape:
        raise ValueError(
            "kinetic_energy and potential_energy must be aligned state histories."
        )
    interval_count = kinetic.size - 1
    interval_shape = (interval_count,)
    damping = jnp.asarray(damping_work, dtype=kinetic.dtype)
    material_contact = jnp.asarray(material_contact_work, dtype=kinetic.dtype)
    external = jnp.asarray(external_work, dtype=kinetic.dtype)
    if any(
        value.shape != interval_shape for value in (damping, material_contact, external)
    ):
        raise ValueError("Work histories must contain one value per accepted interval.")
    accepted_ = jnp.asarray(accepted, dtype=bool)
    epoch_values = tuple(
        jnp.asarray(value)
        for value in (
            topology_epoch,
            contact_epoch,
            fracture_epoch,
            unilateral_epoch,
            mode_epoch,
        )
    )
    if accepted_.shape != kinetic.shape or any(
        value.shape != kinetic.shape for value in epoch_values
    ):
        raise ValueError("Acceptance and epoch histories must align with state history.")
    tolerance_ = float(tolerance)
    if not isfinite(tolerance_) or tolerance_ < 0.0:
        raise ValueError("tolerance must be finite and nonnegative.")
    if traction_velocity_port is None:
        outgoing = jnp.zeros(interval_shape, dtype=kinetic.dtype)
        port_id = None
    else:
        if not isinstance(traction_velocity_port, TractionVelocityPortHistory):
            raise TypeError(
                "traction_velocity_port must be TractionVelocityPortHistory or None."
            )
        outgoing = jnp.asarray(
            traction_velocity_port.outgoing_work,
            dtype=kinetic.dtype,
        )
        if outgoing.shape != interval_shape:
            raise ValueError(
                "Traction–velocity port history must match the energy intervals."
            )
        port_id = traction_velocity_port.port_id

    kinetic_change = jnp.diff(kinetic)
    potential_change = jnp.diff(potential)
    defect = (
        kinetic_change
        + potential_change
        + damping
        + material_contact
        + outgoing
        - external
    )
    scale = jnp.maximum(
        1.0,
        jnp.abs(kinetic_change)
        + jnp.abs(potential_change)
        + jnp.abs(damping)
        + jnp.abs(material_contact)
        + jnp.abs(outgoing)
        + jnp.abs(external),
    )
    (
        topology_consistent,
        contact_consistent,
        fracture_consistent,
        unilateral_consistent,
        mode_consistent,
    ) = tuple(epochs[1:] == epochs[:-1] for epochs in epoch_values)
    epoch_consistent = (
        topology_consistent
        & contact_consistent
        & fracture_consistent
        & unilateral_consistent
        & mode_consistent
    )
    interval_accepted = accepted_[1:] & accepted_[:-1]
    finite_intervals = (
        jnp.isfinite(kinetic_change)
        & jnp.isfinite(potential_change)
        & jnp.isfinite(damping)
        & jnp.isfinite(material_contact)
        & jnp.isfinite(external)
        & jnp.isfinite(outgoing)
        & jnp.isfinite(defect)
        & jnp.isfinite(scale)
    )
    interval_available = interval_accepted & epoch_consistent & finite_intervals
    relative_defect = jnp.abs(defect) / scale
    interval_balanced = interval_available & (relative_defect <= tolerance_)
    available = jnp.all(interval_available)
    finite = jnp.all(finite_intervals)
    balanced = available & jnp.all(interval_balanced)
    maximum_relative_defect = jnp.max(
        jnp.where(interval_available, relative_defect, jnp.inf)
    )
    return MemberEnergyWorkEvidence(
        kinetic_change,
        potential_change,
        damping,
        material_contact,
        external,
        outgoing,
        defect,
        port_id,
        scale,
        interval_accepted,
        topology_consistent,
        contact_consistent,
        fracture_consistent,
        unilateral_consistent,
        mode_consistent,
        epoch_consistent,
        interval_available,
        interval_balanced,
        maximum_relative_defect,
        available,
        finite,
        balanced,
    )


class NewmarkPolicy(StrictModule):
    beta: float
    gamma: float

    def __init__(self, *, beta: float = 0.25, gamma: float = 0.5):
        if beta <= 0.0 or gamma <= 0.0:
            raise ValueError("Newmark beta and gamma must be positive.")
        self.beta = float(beta)
        self.gamma = float(gamma)


def newmark_step(
    mass: ArrayLike,
    damping: ArrayLike,
    tangent: ArrayLike,
    force: ArrayLike,
    state: MemberDynamicState,
    time_step: float,
    /,
    *,
    policy: NewmarkPolicy | None = None,
) -> MemberDynamicState:
    """Advance one average-acceleration step with native dense linear solve."""
    policy_ = NewmarkPolicy() if policy is None else policy
    mass_ = jnp.asarray(mass)
    damping_ = jnp.asarray(damping, dtype=mass_.dtype)
    tangent_ = jnp.asarray(tangent, dtype=mass_.dtype)
    force_ = jnp.asarray(force, dtype=mass_.dtype)
    dt = jnp.asarray(time_step, dtype=mass_.dtype)
    shape = mass_.shape
    if mass_.ndim != 2 or mass_.shape[0] != mass_.shape[1]:
        raise ValueError("Dynamic matrices must be square.")
    if damping_.shape != shape or tangent_.shape != shape or force_.shape != (shape[0],):
        raise ValueError("Dynamic matrices and force vector are incompatible.")
    beta, gamma = policy_.beta, policy_.gamma
    predicted_displacement = (
        state.displacement
        + dt * state.velocity
        + dt**2 * (0.5 - beta) * state.acceleration
    )
    predicted_velocity = state.velocity + dt * (1.0 - gamma) * state.acceleration
    effective = mass_ + gamma * dt * damping_ + beta * dt**2 * tangent_
    rhs = force_ - damping_ @ predicted_velocity - tangent_ @ predicted_displacement
    acceleration = solve(LinearSystem(DenseLinearOperator(effective)), rhs).value
    displacement = predicted_displacement + beta * dt**2 * acceleration
    velocity = predicted_velocity + gamma * dt * acceleration
    kinetic = 0.5 * velocity @ mass_ @ velocity
    strain = 0.5 * displacement @ tangent_ @ displacement
    dissipation = dt * velocity @ damping_ @ velocity
    return MemberDynamicState(
        displacement,
        velocity,
        acceleration,
        kinetic,
        strain,
        state.damping_dissipation + dissipation,
    )


__all__ = [
    "MemberDynamicState",
    "MemberEnergyWorkEvidence",
    "NewmarkPolicy",
    "TractionVelocityPortHistory",
    "member_energy_work_evidence",
    "newmark_step",
]
