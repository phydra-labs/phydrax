#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._types import BorisPushResult


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
    ):
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


class RelativisticBorisPlan(StrictModule, NonTrainableState):
    """Relativistic Boris map for proper velocity in one explicit unit system."""

    speed_of_light: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, speed_of_light: float = 1.0, /, *, tolerance: float = 1.0e-12):
        light = float(speed_of_light)
        tolerance_ = float(tolerance)
        if not np.isfinite(light) or light <= 0.0:
            raise ValueError("speed_of_light must be positive and finite.")
        if not np.isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("tolerance must be finite and nonnegative.")
        self.speed_of_light = light
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "relativistic-boris",
                "speed_of_light": light,
                "tolerance": tolerance_,
            }
        )

    def velocity(self, proper_velocity: ArrayLike, /):
        proper = jnp.asarray(proper_velocity)
        gamma = jnp.sqrt(1.0 + jnp.sum(proper * proper, axis=-1) / self.speed_of_light**2)
        return proper / gamma[..., None]

    def push(
        self,
        proper_velocity: ArrayLike,
        electric: ArrayLike,
        magnetic: ArrayLike,
        specific_charge: ArrayLike,
        active_mask: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> BorisPushResult:
        proper = jnp.asarray(proper_velocity)
        electric_ = jnp.asarray(electric, dtype=proper.dtype)
        magnetic_ = jnp.asarray(magnetic, dtype=proper.dtype)
        specific = jnp.asarray(specific_charge, dtype=proper.dtype)
        active = jnp.asarray(active_mask, dtype=bool)
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
        u_minus = proper + half * electric_
        gamma_minus = jnp.sqrt(
            1.0 + jnp.sum(u_minus * u_minus, axis=-1) / self.speed_of_light**2
        )
        t = half * magnetic_ / gamma_minus[:, None]
        s = 2.0 * t / (1.0 + jnp.sum(t * t, axis=-1))[:, None]
        u_prime = u_minus + jnp.cross(u_minus, t)
        u_plus = u_minus + jnp.cross(u_prime, s)
        candidate = u_plus + half * electric_
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
                speed <= self.speed_of_light * (1.0 + self.tolerance),
                True,
            )
        )
        return BorisPushResult(
            candidate,
            velocity,
            jnp.max(jnp.where(active, speed, 0.0), initial=0.0),
            finite,
            subluminal,
            finite & subluminal & jnp.isfinite(step),
        )


__all__ = ["PICResourcePolicy", "RelativisticBorisPlan"]
