#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.contact._interface import (
    assemble_contact_interface_traction,
    ContactInterfaceKinematics,
    ContactInterfacePlan,
    ContactInterfaceResidual,
)


class MortarContactPlan(StrictModule, NonTrainableState):
    penalty: float = eqx.field(static=True)
    friction: float = eqx.field(static=True)
    augmentation_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        penalty: float,
        friction: float = 0.0,
        augmentation_tolerance: float = 1.0e-10,
    ):
        penalty_ = float(penalty)
        friction_ = float(friction)
        tolerance = float(augmentation_tolerance)
        if not np.isfinite(penalty_) or penalty_ <= 0.0:
            raise ValueError("Mortar penalty must be finite and positive.")
        if not np.isfinite(friction_) or friction_ < 0.0:
            raise ValueError("Mortar friction must be finite and nonnegative.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("Mortar augmentation tolerance must be positive.")
        self.penalty = penalty_
        self.friction = friction_
        self.augmentation_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mortar-contact-plan",
                "penalty": penalty_.hex(),
                "friction": friction_.hex(),
                "augmentation_tolerance": tolerance.hex(),
            }
        )


class MortarContactState(StrictModule, NonTrainableState):
    route_keys: Array
    normal_multiplier: Array
    tangential_multiplier: Array
    valid: Array
    state_version: Array
    plan_id: str = eqx.field(static=True)

    @classmethod
    def initialize(
        cls,
        interface: ContactInterfacePlan,
        plan: MortarContactPlan,
        /,
        *,
        dtype=jnp.float64,
    ) -> MortarContactState:
        tangent_dimension = interface.ambient_dimension - 1
        return cls(
            interface.route_keys,
            jnp.zeros((interface.capacity,), dtype=dtype),
            jnp.zeros((interface.capacity, tangent_dimension), dtype=dtype),
            interface.valid,
            jnp.asarray(0, dtype=jnp.int32),
            plan.plan_id,
        )


class MortarContactEvidence(StrictModule):
    minimum_gap: Array
    complementarity_defect: Array
    cone_defect: Array
    multiplier_change: Array
    action_reaction_residual: Array
    finite: Array
    converged: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class MortarContactResult(StrictModule):
    traction: Array
    residual: ContactInterfaceResidual
    candidate_state: MortarContactState
    evidence: MortarContactEvidence


def evaluate_mortar_contact(
    plan: MortarContactPlan,
    interface: ContactInterfacePlan,
    kinematics: ContactInterfaceKinematics,
    state: MortarContactState,
    /,
) -> MortarContactResult:
    if not isinstance(plan, MortarContactPlan):
        raise TypeError("plan must be MortarContactPlan.")
    if state.plan_id != plan.plan_id or kinematics.interface_id != interface.interface_id:
        raise ValueError("Mortar plan, state, and interface do not match.")
    if state.route_keys.shape != interface.route_keys.shape or not bool(
        jnp.all(state.route_keys == interface.route_keys)
    ):
        raise ValueError("Mortar multiplier route keys changed.")
    dtype = kinematics.gap.dtype
    penalty = jnp.asarray(plan.penalty, dtype=dtype)
    normal_candidate = jnp.maximum(
        0.0, state.normal_multiplier - penalty * kinematics.gap
    )
    tangent_trial = state.tangential_multiplier - penalty * kinematics.tangential_jump
    tangent_norm = jnp.sqrt(
        jnp.sum(tangent_trial * tangent_trial, axis=-1) + jnp.finfo(dtype).tiny
    )
    tangent_limit = plan.friction * normal_candidate
    tangent_scale = jnp.minimum(
        1.0, tangent_limit / jnp.maximum(tangent_norm, jnp.finfo(dtype).eps)
    )
    tangent_candidate = tangent_scale[:, None] * tangent_trial
    normal_candidate = jnp.where(interface.valid, normal_candidate, 0.0)
    tangent_candidate = jnp.where(interface.valid[:, None], tangent_candidate, 0.0)
    traction = normal_candidate[:, None] * kinematics.normal + jnp.sum(
        kinematics.tangent_basis * tangent_candidate[:, None, :],
        axis=-1,
    )
    residual = assemble_contact_interface_traction(interface, traction)
    normal_change = normal_candidate - state.normal_multiplier
    tangent_change = tangent_candidate - state.tangential_multiplier
    change = jnp.sqrt(
        jnp.sum(normal_change * normal_change) + jnp.sum(tangent_change * tangent_change)
    )
    complementarity = jnp.max(
        jnp.where(
            interface.valid,
            jnp.abs(normal_candidate * jnp.maximum(kinematics.gap, 0.0)),
            0.0,
        ),
        initial=0.0,
    )
    cone_defect = jnp.max(
        jnp.where(
            interface.valid,
            jnp.maximum(
                jnp.sqrt(jnp.sum(tangent_candidate * tangent_candidate, axis=-1))
                - plan.friction * normal_candidate,
                0.0,
            ),
            0.0,
        ),
        initial=0.0,
    )
    finite = (
        jnp.all(jnp.isfinite(traction))
        & jnp.all(jnp.isfinite(normal_candidate))
        & jnp.all(jnp.isfinite(tangent_candidate))
        & residual.finite
    )
    converged = change <= plan.augmentation_tolerance
    candidate = MortarContactState(
        state.route_keys,
        normal_candidate,
        tangent_candidate,
        state.valid,
        state.state_version + 1,
        plan.plan_id,
    )
    evidence = MortarContactEvidence(
        jnp.min(
            jnp.where(interface.valid, kinematics.gap, jnp.inf),
            initial=jnp.inf,
        ),
        complementarity,
        cone_defect,
        change,
        residual.action_reaction_residual,
        finite,
        converged,
        finite & residual.successful,
        plan.plan_id,
    )
    return MortarContactResult(traction, residual, candidate, evidence)


class OneSidedNitscheContactPlan(StrictModule, NonTrainableState):
    stabilization: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, stabilization: float, /):
        value = float(stabilization)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("Nitsche stabilization must be finite and positive.")
        self.stabilization = value
        self.plan_id = canonical_fingerprint(
            {
                "kind": "one-sided-nitsche-contact-plan",
                "stabilization": value.hex(),
            }
        )


class UnbiasedNitscheContactPlan(StrictModule, NonTrainableState):
    stabilization: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, stabilization: float, /):
        value = float(stabilization)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("Nitsche stabilization must be finite and positive.")
        self.stabilization = value
        self.plan_id = canonical_fingerprint(
            {
                "kind": "unbiased-nitsche-contact-plan",
                "stabilization": value.hex(),
            }
        )


class NitscheContactResult(StrictModule):
    traction: Array
    residual: ContactInterfaceResidual
    active: Array
    minimum_gap: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def evaluate_nitsche_contact(
    plan: OneSidedNitscheContactPlan | UnbiasedNitscheContactPlan,
    interface: ContactInterfacePlan,
    kinematics: ContactInterfaceKinematics,
    plus_normal_stress: ArrayLike,
    /,
    *,
    minus_normal_stress: ArrayLike | None = None,
) -> NitscheContactResult:
    plus_stress = jnp.asarray(plus_normal_stress, dtype=kinematics.gap.dtype)
    if plus_stress.shape != (interface.capacity,):
        raise ValueError("Nitsche plus stress must have interface capacity shape.")
    if isinstance(plan, UnbiasedNitscheContactPlan):
        if minus_normal_stress is None:
            raise ValueError("Unbiased Nitsche contact requires both stress traces.")
        minus_stress = jnp.asarray(minus_normal_stress, dtype=plus_stress.dtype)
        if minus_stress.shape != plus_stress.shape:
            raise ValueError("Nitsche stress traces must match.")
        consistency = 0.5 * (plus_stress + minus_stress)
    elif isinstance(plan, OneSidedNitscheContactPlan):
        consistency = plus_stress
    else:
        raise TypeError("plan must be a concrete Nitsche contact plan.")
    pressure = jnp.maximum(
        0.0,
        consistency - plan.stabilization * kinematics.gap,
    )
    active = interface.valid & (pressure > 0.0)
    traction = jnp.where(active[:, None], pressure[:, None] * kinematics.normal, 0.0)
    residual = assemble_contact_interface_traction(interface, traction)
    finite = (
        jnp.all(jnp.isfinite(pressure))
        & jnp.all(jnp.isfinite(traction))
        & residual.finite
    )
    return NitscheContactResult(
        traction,
        residual,
        active,
        jnp.min(
            jnp.where(interface.valid, kinematics.gap, jnp.inf),
            initial=jnp.inf,
        ),
        finite,
        finite & residual.successful,
        plan.plan_id,
    )


class MeshTiePlan(StrictModule, NonTrainableState):
    penalty: float = eqx.field(static=True)
    tension_limit: float | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, penalty: float, /, *, tension_limit: float | None = None):
        penalty_ = float(penalty)
        tension = None if tension_limit is None else float(tension_limit)
        if not np.isfinite(penalty_) or penalty_ <= 0.0:
            raise ValueError("Mesh-tie penalty must be finite and positive.")
        if tension is not None and (not np.isfinite(tension) or tension <= 0.0):
            raise ValueError("Mesh-tie tension limit must be positive or None.")
        self.penalty = penalty_
        self.tension_limit = tension
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mesh-tie-plan",
                "penalty": penalty_.hex(),
                "tension_limit": None if tension is None else tension.hex(),
            }
        )


def evaluate_mesh_tie(
    plan: MeshTiePlan,
    interface: ContactInterfacePlan,
    kinematics: ContactInterfaceKinematics,
    /,
) -> NitscheContactResult:
    traction = -plan.penalty * kinematics.relative_displacement
    if plan.tension_limit is not None:
        normal_traction = jnp.sum(traction * kinematics.normal, axis=-1)
        failed = normal_traction < -plan.tension_limit
        traction = jnp.where(failed[:, None], 0.0, traction)
        active = interface.valid & ~failed
    else:
        active = interface.valid
    traction = jnp.where(active[:, None], traction, 0.0)
    residual = assemble_contact_interface_traction(interface, traction)
    finite = jnp.all(jnp.isfinite(traction)) & residual.finite
    return NitscheContactResult(
        traction,
        residual,
        active,
        jnp.min(jnp.abs(kinematics.gap), initial=0.0),
        finite,
        finite & residual.successful,
        plan.plan_id,
    )


__all__ = [
    "MeshTiePlan",
    "MortarContactEvidence",
    "MortarContactPlan",
    "MortarContactResult",
    "MortarContactState",
    "NitscheContactResult",
    "OneSidedNitscheContactPlan",
    "UnbiasedNitscheContactPlan",
    "evaluate_mesh_tie",
    "evaluate_mortar_contact",
    "evaluate_nitsche_contact",
]
