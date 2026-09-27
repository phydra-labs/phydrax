#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import assert_never, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._physical import DimensionalScaleContract, RelativityScaleContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...typing import parse
from ...units import LENGTH, MASS, TIME, UnitDefinition
from ._types import RelativisticPushResult


RelativisticPusher: TypeAlias = Literal["boris", "vay", "higuera-cary"]

_PIC_CODE_REFERENCE_SYSTEM_ID = "phydrax:pic-code"

# Default PIC code units: c = 1 in one declared code reference system. Particle
# kinematics never read G, hbar, or k_B; the quantum constants are marked implicit.
PIC_CODE_RELATIVITY = RelativityScaleContract(
    DimensionalScaleContract(
        UnitDefinition("code_length", LENGTH, _PIC_CODE_REFERENCE_SYSTEM_ID),
        UnitDefinition("code_mass", MASS, _PIC_CODE_REFERENCE_SYSTEM_ID),
        UnitDefinition("code_time", TIME, _PIC_CODE_REFERENCE_SYSTEM_ID),
        length_coordinate_kind="code",
    ),
    1,
    1,
    1,
    1,
    quantum_constants_explicit=False,
)


class PICResourcePolicy(StrictModule, NonTrainableState):
    maximum_state_bytes: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    maximum_segments_per_particle: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_state_bytes: int = 1024**3,
        maximum_workspace_bytes: int = 2 * 1024**3,
        maximum_segments_per_particle: int = 4,
    ) -> None:
        state = int(maximum_state_bytes)
        workspace = int(maximum_workspace_bytes)
        segments = int(maximum_segments_per_particle)
        if state <= 0 or workspace <= 0 or segments <= 0:
            raise ValueError("PIC resource limits must be positive.")
        self.maximum_state_bytes = state
        self.maximum_workspace_bytes = workspace
        self.maximum_segments_per_particle = segments
        self.policy_id = canonical_fingerprint(
            {
                "kind": "pic-resource-policy",
                "state": state,
                "workspace": workspace,
                "segments": segments,
            }
        )

    def admit(self, *, state_bytes: int, workspace_bytes: int) -> None:
        if int(state_bytes) > self.maximum_state_bytes:
            raise ValueError("PIC state exceeds its resource policy.")
        if int(workspace_bytes) > self.maximum_workspace_bytes:
            raise ValueError("PIC workspace exceeds its resource policy.")


def _lorentz_factor(proper_velocity: Array, speed_of_light: float, /) -> Array:
    return jnp.sqrt(
        1.0 + jnp.sum(proper_velocity * proper_velocity, axis=-1) / speed_of_light**2
    )


def _boris_update(
    proper: Array, electric: Array, magnetic: Array, half: Array, light: float, /
) -> Array:
    u_minus = proper + half * electric
    gamma_minus = _lorentz_factor(u_minus, light)
    t = half * magnetic / gamma_minus[:, None]
    s = 2.0 * t / (1.0 + jnp.sum(t * t, axis=-1))[:, None]
    u_prime = u_minus + jnp.cross(u_minus, t)
    u_plus = u_minus + jnp.cross(u_prime, s)
    return u_plus + half * electric


def _implicit_rotation_factor(
    u: Array, gamma: Array, tau: Array, light: float, /
) -> Array:
    # Positive root of gamma_new^4 - sigma gamma_new^2 - (tau^2 + u*^2) = 0 with
    # sigma = gamma^2 - tau^2 and u* = u.tau / c; Vay (2008) evaluates gamma at u',
    # Higuera and Cary (2017) at u^-.
    tau2 = jnp.sum(tau * tau, axis=-1)
    u_star = jnp.sum(u * tau, axis=-1) / light
    sigma = gamma * gamma - tau2
    return jnp.sqrt(0.5 * (sigma + jnp.sqrt(sigma * sigma + 4.0 * (tau2 + u_star**2))))


def _rotate(u: Array, t: Array, /) -> Array:
    # Closed-form solution of u_out - u_out x t = u.
    s = 1.0 / (1.0 + jnp.sum(t * t, axis=-1))
    return s[:, None] * (u + jnp.sum(u * t, axis=-1)[:, None] * t + jnp.cross(u, t))


def _vay_update(
    proper: Array, electric: Array, magnetic: Array, half: Array, light: float, /
) -> Array:
    # Vay, Phys. Plasmas 15, 056701 (2008): exact E x B drift for any gamma.
    gamma = _lorentz_factor(proper, light)
    tau = half * magnetic
    u_prime = proper + 2.0 * half * electric + jnp.cross(proper / gamma[:, None], tau)
    gamma_new = _implicit_rotation_factor(
        u_prime, _lorentz_factor(u_prime, light), tau, light
    )
    return _rotate(u_prime, tau / gamma_new[:, None])


def _higuera_cary_update(
    proper: Array, electric: Array, magnetic: Array, half: Array, light: float, /
) -> Array:
    # Higuera and Cary, Phys. Plasmas 24, 052104 (2017): volume preserving with the
    # magnetic rotation evaluated at the midpoint Lorentz factor.
    u_minus = proper + half * electric
    tau = half * magnetic
    gamma_mid = _implicit_rotation_factor(
        u_minus, _lorentz_factor(u_minus, light), tau, light
    )
    t = tau / gamma_mid[:, None]
    u_plus = _rotate(u_minus, t)
    return u_plus + half * electric + jnp.cross(u_plus, t)


class RelativisticPushPlan(StrictModule, NonTrainableState):
    """Relativistic proper-velocity pusher in one declared relativity scale.

    ``method`` selects the Boris, Vay (2008), or Higuera--Cary (2017) map. The
    speed of light is the exact ``relativity.speed_of_light`` in the scale's own
    velocity unit; fields, specific charge, and step size use that same scale.
    """

    relativity: RelativityScaleContract = eqx.field(static=True)
    method: RelativisticPusher = eqx.field(static=True)
    speed_of_light: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        relativity: RelativityScaleContract,
        /,
        *,
        method: RelativisticPusher,
        tolerance: float = 1.0e-12,
    ) -> None:
        if not isinstance(relativity, RelativityScaleContract):
            raise TypeError("relativity must be a RelativityScaleContract.")
        method_ = parse(method, RelativisticPusher, "method")
        tolerance_ = float(tolerance)
        if not np.isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("tolerance must be finite and nonnegative.")
        self.relativity = relativity
        self.method = method_
        self.speed_of_light = float(relativity.speed_of_light)
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "relativistic-push",
                "relativity": relativity.scale_id,
                "method": method_,
                "tolerance": tolerance_,
            }
        )

    def velocity(self, proper_velocity: ArrayLike, /) -> Array:
        proper = jnp.asarray(proper_velocity)
        return proper / _lorentz_factor(proper, self.speed_of_light)[..., None]

    def push(
        self,
        proper_velocity: ArrayLike,
        electric: ArrayLike,
        magnetic: ArrayLike,
        specific_charge: ArrayLike,
        active_mask: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> RelativisticPushResult:
        proper = jnp.asarray(proper_velocity)
        electric_ = jnp.asarray(electric, dtype=proper.dtype)
        magnetic_ = jnp.asarray(magnetic, dtype=proper.dtype)
        specific = jnp.asarray(specific_charge, dtype=proper.dtype)
        active = jnp.asarray(active_mask, dtype=jnp.bool_)
        step = jnp.asarray(step_size, dtype=proper.dtype).reshape(())
        if proper.ndim != 2 or proper.shape[-1] != 3:
            raise ValueError("proper_velocity must have shape (particles,3).")
        if electric_.shape != proper.shape or magnetic_.shape != proper.shape:
            raise ValueError("electric and magnetic must match proper_velocity.")
        if specific.shape != (proper.shape[0],) or active.shape != specific.shape:
            raise ValueError(
                "specific_charge and active_mask must match particle capacity."
            )
        half = 0.5 * step * specific[:, None]
        light = self.speed_of_light
        match self.method:
            case "boris":
                candidate = _boris_update(proper, electric_, magnetic_, half, light)
            case "vay":
                candidate = _vay_update(proper, electric_, magnetic_, half, light)
            case "higuera-cary":
                candidate = _higuera_cary_update(
                    proper, electric_, magnetic_, half, light
                )
            case _:
                assert_never(self.method)
        candidate = jnp.where(active[:, None], candidate, 0.0)
        velocity = self.velocity(candidate)
        speed = jnp.sqrt(jnp.sum(velocity * velocity, axis=-1))
        finite = jnp.all(
            jnp.where(
                active[:, None],
                jnp.isfinite(candidate) & jnp.isfinite(velocity),
                True,
            )
        )
        subluminal = jnp.all(
            jnp.where(
                active,
                speed <= light * (1.0 + self.tolerance),
                True,
            )
        )
        return RelativisticPushResult(
            candidate,
            velocity,
            jnp.max(jnp.where(active, speed, 0.0), initial=0.0),
            finite,
            subluminal,
            finite & subluminal & jnp.isfinite(step),
        )


__all__ = [
    "PIC_CODE_RELATIVITY",
    "PICResourcePolicy",
    "RelativisticPushPlan",
    "RelativisticPusher",
]
