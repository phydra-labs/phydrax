#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ...discretization.contact._kinematics import ContactKinematicsBatch
from ._barrier import (
    clamped_log_barrier,
    clamped_log_barrier_first_derivative,
    clamped_log_barrier_second_derivative,
    physical_barrier_scale,
)
from ._closure import (
    AbstractNormalContactLaw,
    ContactClosureCapability,
    NormalContactResponse,
)
from ._materials import ContactPairParameters
from ._route_state import ContactRouteState


class BarrierNormalContactLaw(AbstractNormalContactLaw):
    activation_distance: float = eqx.field(static=True)
    _law_id: str = eqx.field(static=True)

    def __init__(self, activation_distance: float, /):
        activation = float(activation_distance)
        if not np.isfinite(activation) or activation <= 0.0:
            raise ValueError("activation_distance must be finite and positive.")
        self.activation_distance = activation
        self._law_id = canonical_fingerprint(
            {
                "kind": "barrier-normal-contact-law",
                "activation_distance": activation.hex(),
            }
        )

    @property
    def law_id(self) -> str:
        return self._law_id

    @property
    def capabilities(self) -> ContactClosureCapability:
        return (
            ContactClosureCapability.POTENTIAL
            | ContactClosureCapability.RESIDUAL
            | ContactClosureCapability.DIFFERENTIABLE
        )

    def evaluate(
        self,
        kinematics: ContactKinematicsBatch,
        parameters: ContactPairParameters,
        state: ContactRouteState,
        /,
    ) -> NormalContactResponse:
        del state
        dtype = kinematics.gap.dtype
        distance = kinematics.distance
        separation = kinematics.minimum_separation
        activation = jnp.asarray(self.activation_distance, dtype=dtype)
        shifted = distance * distance - separation * separation
        threshold = (2.0 * separation + activation) * activation
        safe_shifted = jnp.maximum(shifted, jnp.finfo(dtype).tiny)
        active = kinematics.valid & (kinematics.gap < activation)
        scale = physical_barrier_scale(activation, separation)
        barrier = clamped_log_barrier(safe_shifted, threshold)
        first = clamped_log_barrier_first_derivative(safe_shifted, threshold)
        second = clamped_log_barrier_second_derivative(safe_shifted, threshold)
        stiffness = parameters.normal_stiffness.astype(dtype)
        potential = stiffness * scale * barrier
        traction = -stiffness * scale * first * 2.0 * distance
        tangent = stiffness * scale * (second * (2.0 * distance) ** 2 + 2.0 * first)
        potential = jnp.where(active, potential, 0.0)
        traction = jnp.where(active, jnp.maximum(traction, 0.0), 0.0)
        tangent = jnp.where(active, tangent, 0.0)
        finite = (
            jnp.all((~kinematics.valid) | jnp.isfinite(potential))
            & jnp.all((~kinematics.valid) | jnp.isfinite(traction))
            & jnp.all((~kinematics.valid) | jnp.isfinite(tangent))
        )
        admissible = (~kinematics.valid) | (kinematics.gap > 0.0)
        return NormalContactResponse(
            traction,
            potential,
            tangent,
            active,
            admissible,
            finite,
        )


class GeometricContactNormalLaw(AbstractNormalContactLaw):
    """C2 inverse-power barrier for prefiltered geometric-contact routes."""

    activation_distance: float = eqx.field(static=True)
    _law_id: str = eqx.field(static=True)

    def __init__(self, activation_distance: float, /):
        activation = float(activation_distance)
        if not np.isfinite(activation) or activation <= 0.0:
            raise ValueError("activation_distance must be finite and positive.")
        self.activation_distance = activation
        self._law_id = canonical_fingerprint(
            {
                "kind": "geometric-normal-contact-law",
                "activation_distance": activation.hex(),
            }
        )

    @property
    def law_id(self) -> str:
        return self._law_id

    @property
    def capabilities(self) -> ContactClosureCapability:
        return (
            ContactClosureCapability.POTENTIAL
            | ContactClosureCapability.RESIDUAL
            | ContactClosureCapability.DIFFERENTIABLE
        )

    def evaluate(
        self,
        kinematics: ContactKinematicsBatch,
        parameters: ContactPairParameters,
        state: ContactRouteState,
        /,
    ) -> NormalContactResponse:
        del state
        dtype = kinematics.gap.dtype
        activation = jnp.asarray(self.activation_distance, dtype=dtype)
        safe_gap = jnp.maximum(kinematics.gap, jnp.finfo(dtype).tiny)
        ratio = safe_gap / activation
        one_minus = 1.0 - ratio
        active = kinematics.valid & (kinematics.gap < activation)
        stiffness = parameters.normal_stiffness.astype(dtype)
        potential = stiffness * activation * one_minus**4 / ratio
        first = stiffness * (-4.0 * one_minus**3 / ratio - one_minus**4 / (ratio * ratio))
        traction = -first
        second = (
            stiffness
            / activation
            * (
                12.0 * one_minus**2 / ratio
                + 8.0 * one_minus**3 / (ratio * ratio)
                + 2.0 * one_minus**4 / (ratio**3)
            )
        )
        potential = jnp.where(active, potential, 0.0)
        traction = jnp.where(active, jnp.maximum(traction, 0.0), 0.0)
        second = jnp.where(active, second, 0.0)
        finite = (
            jnp.all((~kinematics.valid) | jnp.isfinite(potential))
            & jnp.all((~kinematics.valid) | jnp.isfinite(traction))
            & jnp.all((~kinematics.valid) | jnp.isfinite(second))
        )
        admissible = (~kinematics.valid) | (kinematics.gap > 0.0)
        return NormalContactResponse(
            traction,
            potential,
            second,
            active,
            admissible,
            finite,
        )


class CompliantNormalContactLaw(AbstractNormalContactLaw):
    damping: float = eqx.field(static=True)
    _law_id: str = eqx.field(static=True)

    def __init__(self, *, damping: float = 0.0):
        damping_ = float(damping)
        if not np.isfinite(damping_) or damping_ < 0.0:
            raise ValueError("Contact damping must be finite and nonnegative.")
        self.damping = damping_
        self._law_id = canonical_fingerprint(
            {
                "kind": "compliant-normal-contact-law",
                "damping": damping_.hex(),
            }
        )

    @property
    def law_id(self) -> str:
        return self._law_id

    @property
    def capabilities(self) -> ContactClosureCapability:
        capability = (
            ContactClosureCapability.RESIDUAL | ContactClosureCapability.DIFFERENTIABLE
        )
        if self.damping == 0.0:
            capability |= ContactClosureCapability.POTENTIAL
        return capability

    def evaluate(
        self,
        kinematics: ContactKinematicsBatch,
        parameters: ContactPairParameters,
        state: ContactRouteState,
        /,
    ) -> NormalContactResponse:
        del state
        penetration = jnp.maximum(-kinematics.gap, 0.0)
        stiffness = parameters.normal_stiffness.astype(kinematics.gap.dtype)
        elastic = stiffness * penetration
        damping_factor = jnp.maximum(
            0.0,
            1.0 - self.damping * kinematics.normal_velocity,
        )
        active = kinematics.valid & (penetration > 0.0)
        traction = jnp.where(active, elastic * damping_factor, 0.0)
        potential = jnp.where(active, 0.5 * stiffness * penetration * penetration, 0.0)
        tangent = jnp.where(active, stiffness * damping_factor, 0.0)
        finite = (
            jnp.all(jnp.isfinite(traction))
            & jnp.all(jnp.isfinite(potential))
            & jnp.all(jnp.isfinite(tangent))
        )
        return NormalContactResponse(
            traction,
            potential,
            tangent,
            active,
            jnp.ones_like(kinematics.valid),
            finite,
        )


class AdhesiveBarrierNormalLaw(AbstractNormalContactLaw):
    repulsion: BarrierNormalContactLaw
    adhesion_range: float = eqx.field(static=True)
    _law_id: str = eqx.field(static=True)

    def __init__(self, activation_distance: float, adhesion_range: float, /):
        adhesion = float(adhesion_range)
        if not np.isfinite(adhesion) or adhesion <= 0.0:
            raise ValueError("adhesion_range must be finite and positive.")
        repulsion = BarrierNormalContactLaw(activation_distance)
        self.repulsion = repulsion
        self.adhesion_range = adhesion
        self._law_id = canonical_fingerprint(
            {
                "kind": "adhesive-barrier-normal-law",
                "repulsion": repulsion.law_id,
                "adhesion_range": adhesion.hex(),
            }
        )

    @property
    def law_id(self) -> str:
        return self._law_id

    @property
    def capabilities(self) -> ContactClosureCapability:
        return self.repulsion.capabilities | ContactClosureCapability.ADHESION

    def evaluate(
        self,
        kinematics: ContactKinematicsBatch,
        parameters: ContactPairParameters,
        state: ContactRouteState,
        /,
    ) -> NormalContactResponse:
        repulsion = self.repulsion.evaluate(kinematics, parameters, state)
        dtype = kinematics.gap.dtype
        range_ = jnp.asarray(self.adhesion_range, dtype=dtype)
        opening = jnp.maximum(kinematics.gap, 0.0)
        active = kinematics.valid & (opening < range_)
        undamaged = 1.0 - jnp.clip(state.adhesion_damage, 0.0, 1.0)
        energy = parameters.adhesion_energy.astype(dtype) * undamaged
        ratio = opening / range_
        cohesive_potential = -energy * (1.0 - ratio) ** 2
        cohesive_traction = -2.0 * energy * (1.0 - ratio) / range_
        cohesive_tangent = -2.0 * energy / (range_ * range_)
        potential = repulsion.potential_density + jnp.where(
            active, cohesive_potential, 0.0
        )
        traction = repulsion.traction + jnp.where(active, cohesive_traction, 0.0)
        tangent = repulsion.tangent_stiffness + jnp.where(active, cohesive_tangent, 0.0)
        finite = (
            repulsion.finite
            & jnp.all(jnp.isfinite(potential))
            & jnp.all(jnp.isfinite(traction))
            & jnp.all(jnp.isfinite(tangent))
        )
        return NormalContactResponse(
            traction,
            potential,
            tangent,
            repulsion.active | active,
            repulsion.admissible,
            finite,
        )


__all__ = [
    "AdhesiveBarrierNormalLaw",
    "BarrierNormalContactLaw",
    "CompliantNormalContactLaw",
    "GeometricContactNormalLaw",
]
