#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...dynamics import ContinuousSystem, StateLayout
from ._status import AstrodynamicsStatus


_CR3BP_LAYOUT = StateLayout(
    (6,),
    axes=("phase_component",),
    component_names=("x", "y", "z", "vx", "vy", "vz"),
    layout_id="astrodynamics:cr3bp-state",
)


class CR3BPDiagnostics(StrictModule):
    jacobi_constant: Array
    primary_distances: Array
    valid: Array
    status: Array
    system_id: str = eqx.field(static=True)


class CR3BPLagrangePoints(StrictModule):
    points: Array
    residuals: Array
    valid: Array
    status: Array
    system_id: str = eqx.field(static=True)


class CR3BPSystem(StrictModule, NonTrainableState):
    """Normalized circular restricted three-body problem."""

    mass_ratio: Array
    collision_radius: Array
    system_id: str = eqx.field(static=True)

    def __init__(
        self,
        mass_ratio: ArrayLike,
        /,
        *,
        collision_radius: ArrayLike = 0.0,
    ):
        ratio_host = float(np.asarray(mass_ratio))
        collision_host = float(np.asarray(collision_radius))
        if not np.isfinite(ratio_host) or not 0.0 < ratio_host <= 0.5:
            raise ValueError("CR3BP mass_ratio must lie in (0, 0.5].")
        if not np.isfinite(collision_host) or collision_host < 0.0:
            raise ValueError("collision_radius must be finite and non-negative.")
        self.mass_ratio = jnp.asarray(ratio_host)
        self.collision_radius = jnp.asarray(collision_host)
        self.system_id = canonical_fingerprint(
            {
                "kind": "cr3bp-system",
                "mass_ratio": ratio_host,
                "collision_radius": collision_host,
            }
        )

    def _distances(self, state: Array, /) -> tuple[Array, Array]:
        x, y, z = state[:3]
        first = jnp.sqrt((x + self.mass_ratio) ** 2 + y * y + z * z)
        second = jnp.sqrt((x - (1.0 - self.mass_ratio)) ** 2 + y * y + z * z)
        return first, second

    def vector_field(self, time: Array, state: Array, args=None, /) -> Array:
        del time, args
        first, second = self._distances(state)
        safe_first = jnp.where(first > 0.0, first, 1.0)
        safe_second = jnp.where(second > 0.0, second, 1.0)
        x, y, z, vx, vy, _ = state
        first_mass = 1.0 - self.mass_ratio
        second_mass = self.mass_ratio
        ax = (
            2.0 * vy
            + x
            - first_mass * (x + second_mass) / safe_first**3
            - second_mass * (x - first_mass) / safe_second**3
        )
        ay = (
            -2.0 * vx
            + y
            - first_mass * y / safe_first**3
            - second_mass * y / safe_second**3
        )
        az = -first_mass * z / safe_first**3 - second_mass * z / safe_second**3
        derivative = jnp.concatenate((state[3:], jnp.asarray((ax, ay, az))))
        collision = (first <= self.collision_radius) | (second <= self.collision_radius)
        return jnp.where(collision, jnp.full_like(derivative, jnp.nan), derivative)

    def diagnostics(self, state: ArrayLike, /) -> CR3BPDiagnostics:
        values = jnp.asarray(state)
        if values.shape != (6,):
            raise ValueError("CR3BP state must have shape (6,).")
        first, second = self._distances(values)
        first_mass = 1.0 - self.mass_ratio
        second_mass = self.mass_ratio
        speed_squared = jnp.sum(values[3:] ** 2)
        jacobi = (
            values[0] ** 2
            + values[1] ** 2
            + 2.0 * first_mass / jnp.where(first > 0.0, first, 1.0)
            + 2.0 * second_mass / jnp.where(second > 0.0, second, 1.0)
            - speed_squared
        )
        finite = jnp.all(jnp.isfinite(values))
        collision = (first <= self.collision_radius) | (second <= self.collision_radius)
        valid = finite & ~collision
        status = jnp.where(
            ~finite,
            int(AstrodynamicsStatus.NONFINITE_INPUT),
            jnp.where(
                collision,
                int(AstrodynamicsStatus.COLLISION),
                int(AstrodynamicsStatus.SUCCESS),
            ),
        ).astype(jnp.int32)
        return CR3BPDiagnostics(
            jnp.where(valid, jacobi, jnp.nan),
            jnp.asarray((first, second)),
            valid,
            status,
            self.system_id,
        )

    def continuous_system(self) -> ContinuousSystem:
        return ContinuousSystem(
            self.vector_field,
            state_layout=_CR3BP_LAYOUT,
            system_id=self.system_id,
        )

    def lagrange_points(self, /) -> CR3BPLagrangePoints:
        ratio = self.mass_ratio
        first_mass = 1.0 - ratio

        def equation(x):
            left = x + ratio
            right = x - first_mass
            return (
                x
                - first_mass * left / jnp.abs(left) ** 3
                - ratio * right / jnp.abs(right) ** 3
            )

        seeds = jnp.asarray(
            (
                first_mass - (ratio / 3.0) ** (1.0 / 3.0),
                first_mass + (ratio / 3.0) ** (1.0 / 3.0),
                -1.0 - 5.0 * ratio / 12.0,
            )
        )

        def solve(seed):
            def step(_, value):
                residual = equation(value)
                derivative = jax.grad(equation)(value)
                candidate = value - residual / jnp.where(
                    jnp.abs(derivative) > 0.0, derivative, 1.0
                )
                return jnp.where(jnp.isfinite(candidate), candidate, value)

            root = jax.lax.fori_loop(0, 48, step, seed)
            return root, jnp.abs(equation(root))

        roots, residuals = jax.vmap(solve)(seeds)
        triangular_x = 0.5 - ratio
        triangular_y = jnp.sqrt(3.0) / 2.0
        points = jnp.asarray(
            (
                (roots[0], 0.0, 0.0),
                (roots[1], 0.0, 0.0),
                (roots[2], 0.0, 0.0),
                (triangular_x, triangular_y, 0.0),
                (triangular_x, -triangular_y, 0.0),
            )
        )
        all_residuals = jnp.concatenate((residuals, jnp.zeros((2,))))
        valid = jnp.isfinite(all_residuals) & (all_residuals <= 1.0e-10)
        status = jnp.where(
            valid,
            int(AstrodynamicsStatus.SUCCESS),
            int(AstrodynamicsStatus.NONCONVERGED),
        ).astype(jnp.int32)
        return CR3BPLagrangePoints(points, all_residuals, valid, status, self.system_id)


__all__ = ["CR3BPDiagnostics", "CR3BPLagrangePoints", "CR3BPSystem"]
