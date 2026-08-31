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
from ._context import AstrodynamicsContext
from ._status import AstrodynamicsStatus


def _norm(value: Array, /) -> Array:
    return jnp.sqrt(jnp.sum(value * value))


class LambertPlan(StrictModule, NonTrainableState):
    max_revolutions: int = eqx.field(static=True)
    grid_size: int = eqx.field(static=True)
    bisection_iterations: int = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    maximum_x: float = eqx.field(static=True)
    long_way: bool = eqx.field(static=True)
    plane_normal: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_revolutions: int = 0,
        grid_size: int = 1024,
        bisection_iterations: int = 64,
        relative_tolerance: float = 1.0e-11,
        maximum_x: float = 64.0,
        long_way: bool = False,
        plane_normal: ArrayLike | tuple[float, float, float] = (0.0, 0.0, 1.0),
    ):
        revolutions = int(max_revolutions)
        grid = int(grid_size)
        iterations = int(bisection_iterations)
        tolerance = float(relative_tolerance)
        maximum = float(maximum_x)
        normal = np.asarray(plane_normal, dtype=float)
        if revolutions < 0:
            raise ValueError("max_revolutions must be non-negative.")
        if grid < 64:
            raise ValueError("grid_size must be at least 64.")
        if iterations <= 0:
            raise ValueError("bisection_iterations must be positive.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("relative_tolerance must be finite and positive.")
        if not np.isfinite(maximum) or maximum <= 1.0:
            raise ValueError("maximum_x must be finite and greater than one.")
        if (
            normal.shape != (3,)
            or np.any(~np.isfinite(normal))
            or np.linalg.norm(normal) == 0.0
        ):
            raise ValueError("plane_normal must be a finite nonzero three-vector.")
        if not isinstance(long_way, bool):
            raise TypeError("long_way must be a bool.")
        self.max_revolutions = revolutions
        self.grid_size = grid
        self.bisection_iterations = iterations
        self.relative_tolerance = tolerance
        self.maximum_x = maximum
        self.long_way = long_way
        self.plane_normal = jnp.asarray(normal / np.linalg.norm(normal))
        self.plan_id = canonical_fingerprint(
            {
                "kind": "lambert-plan",
                "max_revolutions": revolutions,
                "grid_size": grid,
                "bisection_iterations": iterations,
                "relative_tolerance": tolerance,
                "maximum_x": maximum,
                "long_way": long_way,
                "plane_normal": normal.tolist(),
            }
        )

    @property
    def capacity(self) -> int:
        return 2 * self.max_revolutions + 1


class LambertResult(StrictModule):
    departure_velocity: Array
    arrival_velocity: Array
    valid: Array
    status: Array
    revolutions: Array
    branch: Array
    root: Array
    residual: Array
    iterations: Array
    transfer_angle: Array
    context: AstrodynamicsContext
    plan_id: str = eqx.field(static=True)


def _tof(x: Array, ll: Array, revolutions: int, /) -> Array:
    one_minus_x2 = 1.0 - x * x
    y = jnp.sqrt(jnp.maximum(1.0 - ll * ll * one_minus_x2, 0.0))
    elliptic = x < 1.0
    elliptic_argument = jnp.clip(x * y + ll * one_minus_x2, -1.0, 1.0)
    hyperbolic_argument = jnp.maximum(x * y - ll * (x * x - 1.0), 1.0)
    psi = jnp.where(
        elliptic,
        jnp.arccos(elliptic_argument),
        jnp.arccosh(hyperbolic_argument),
    )
    denominator_root = jnp.sqrt(jnp.abs(one_minus_x2))
    denominator = jnp.where(jnp.abs(one_minus_x2) > 0.0, one_minus_x2, 1.0)
    numerator = (
        (psi + revolutions * jnp.pi)
        / jnp.where(denominator_root > 0.0, denominator_root, 1.0)
        - x
        + ll * y
    )
    return numerator / denominator


def _root_grid(plan: LambertPlan, dtype, /) -> Array:
    elliptic_count = plan.grid_size - 128
    elliptic = jnp.linspace(-1.0 + 1.0e-8, 1.0 - 1.0e-8, elliptic_count, dtype=dtype)
    hyperbolic = 1.0 + jnp.geomspace(
        jnp.asarray(1.0e-7, dtype=dtype),
        jnp.asarray(plan.maximum_x - 1.0, dtype=dtype),
        128,
    )
    return jnp.concatenate((elliptic, hyperbolic))


def _bisect(
    lower: Array,
    upper: Array,
    ll: Array,
    target: Array,
    revolutions: int,
    iterations: int,
    /,
) -> Array:
    lower_value = _tof(lower, ll, revolutions) - target

    def step(_, carry):
        left, right, left_value = carry
        midpoint = 0.5 * (left + right)
        midpoint_value = _tof(midpoint, ll, revolutions) - target
        same = left_value * midpoint_value > 0.0
        return (
            jnp.where(same, midpoint, left),
            jnp.where(same, right, midpoint),
            jnp.where(same, midpoint_value, left_value),
        )

    left, right, _ = jax.lax.fori_loop(0, iterations, step, (lower, upper, lower_value))
    return 0.5 * (left + right)


def _geometry(r1: Array, r2: Array, mu: Array, long_way: bool, plane_normal: Array, /):
    r1_norm = _norm(r1)
    r2_norm = _norm(r2)
    chord_vector = r2 - r1
    chord = _norm(chord_vector)
    semiperimeter = 0.5 * (r1_norm + r2_norm + chord)
    lambda_squared = jnp.maximum(
        1.0 - chord / jnp.where(semiperimeter > 0.0, semiperimeter, 1.0), 0.0
    )
    ll = jnp.sqrt(lambda_squared)
    ll = jnp.where(long_way, -ll, ll)
    cross = jnp.cross(r1, r2)
    cross_norm = _norm(cross)
    supplied = plane_normal / jnp.where(
        _norm(plane_normal) > 0.0, _norm(plane_normal), 1.0
    )
    angular_normal = jnp.where(cross_norm > 0.0, cross / cross_norm, supplied)
    angular_normal = jnp.where(long_way, -angular_normal, angular_normal)
    target = jnp.sqrt(2.0 * mu / jnp.where(semiperimeter > 0.0, semiperimeter**3, 1.0))
    return r1_norm, r2_norm, chord, semiperimeter, ll, angular_normal, target


def _velocities(
    x: Array,
    r1: Array,
    r2: Array,
    mu: Array,
    ll: Array,
    normal: Array,
    semiperimeter: Array,
    chord: Array,
    /,
) -> tuple[Array, Array]:
    r1_norm = _norm(r1)
    r2_norm = _norm(r2)
    y = jnp.sqrt(jnp.maximum(1.0 - ll * ll * (1.0 - x * x), 0.0))
    gamma = jnp.sqrt(mu * semiperimeter / 2.0)
    rho = (r1_norm - r2_norm) / jnp.where(chord > 0.0, chord, 1.0)
    sigma = jnp.sqrt(jnp.maximum(1.0 - rho * rho, 0.0))
    common_minus = ll * y - x
    common_plus = ll * y + x
    radial_1 = gamma * (common_minus - rho * common_plus) / r1_norm
    radial_2 = -gamma * (common_minus + rho * common_plus) / r2_norm
    tangential_1 = gamma * sigma * (y + ll * x) / r1_norm
    tangential_2 = gamma * sigma * (y + ll * x) / r2_norm
    radial_hat_1 = r1 / r1_norm
    radial_hat_2 = r2 / r2_norm
    transverse_hat_1 = jnp.cross(normal, radial_hat_1)
    transverse_hat_2 = jnp.cross(normal, radial_hat_2)
    return (
        radial_1 * radial_hat_1 + tangential_1 * transverse_hat_1,
        radial_2 * radial_hat_2 + tangential_2 * transverse_hat_2,
    )


def solve_lambert(
    departure_position: ArrayLike,
    arrival_position: ArrayLike,
    time_of_flight: ArrayLike,
    mu: ArrayLike,
    context: AstrodynamicsContext,
    plan: LambertPlan,
    /,
) -> LambertResult:
    """Return every zero- through requested multi-revolution Lambert branch."""

    if not isinstance(context, AstrodynamicsContext):
        raise TypeError("context must be an AstrodynamicsContext.")
    if not isinstance(plan, LambertPlan):
        raise TypeError("plan must be a LambertPlan.")
    r1 = jnp.asarray(departure_position)
    r2 = jnp.asarray(arrival_position, dtype=r1.dtype)
    if r1.shape != (3,) or r2.shape != (3,):
        raise ValueError("Lambert endpoint positions must have shape (3,).")
    tof = jnp.asarray(time_of_flight, dtype=r1.dtype).reshape(())
    coupling = jnp.asarray(mu, dtype=r1.dtype).reshape(())
    (
        r1_norm,
        r2_norm,
        chord,
        semiperimeter,
        ll,
        normal,
        time_scale,
    ) = _geometry(r1, r2, coupling, plan.long_way, plan.plane_normal)
    target = time_scale * tof
    finite = (
        jnp.all(jnp.isfinite(r1))
        & jnp.all(jnp.isfinite(r2))
        & jnp.isfinite(tof)
        & jnp.isfinite(coupling)
    )
    geometry_valid = finite & (r1_norm > 0.0) & (r2_norm > 0.0) & (chord > 0.0)
    domain = geometry_valid & (tof > 0.0) & (coupling > 0.0)
    grid = _root_grid(plan, r1.dtype)
    capacity = plan.capacity
    departure = jnp.zeros((capacity, 3), dtype=r1.dtype)
    arrival = jnp.zeros((capacity, 3), dtype=r1.dtype)
    valid = jnp.zeros((capacity,), dtype=bool)
    status = jnp.full((capacity,), int(AstrodynamicsStatus.NO_SOLUTION), dtype=jnp.int32)
    revolutions = jnp.zeros((capacity,), dtype=jnp.int32)
    branch = jnp.zeros((capacity,), dtype=jnp.int32)
    roots = jnp.zeros((capacity,), dtype=r1.dtype)
    residuals = jnp.full((capacity,), jnp.inf, dtype=r1.dtype)
    iteration_counts = jnp.zeros((capacity,), dtype=jnp.int32)

    for revolution in range(plan.max_revolutions + 1):
        values = _tof(grid, ll, revolution) - target
        changes = (
            jnp.isfinite(values[:-1])
            & jnp.isfinite(values[1:])
            & (values[:-1] * values[1:] <= 0.0)
        )
        indices = jnp.arange(grid.shape[0] - 1, dtype=jnp.int32)
        first = jnp.min(jnp.where(changes, indices, grid.shape[0]))
        last = jnp.max(jnp.where(changes, indices, -1))
        slots = (0,) if revolution == 0 else (2 * revolution - 1, 2 * revolution)
        selected = (first,) if revolution == 0 else (first, last)
        for local_branch, (slot, index) in enumerate(zip(slots, selected, strict=True)):
            found = (index >= 0) & (index < grid.shape[0] - 1)
            safe_index = jnp.clip(index, 0, grid.shape[0] - 2)
            root = _bisect(
                grid[safe_index],
                grid[safe_index + 1],
                ll,
                target,
                revolution,
                plan.bisection_iterations,
            )
            residual = jnp.abs(_tof(root, ll, revolution) - target)
            root_valid = (
                domain
                & found
                & jnp.isfinite(root)
                & jnp.isfinite(residual)
                & (residual <= plan.relative_tolerance * (1.0 + jnp.abs(target)))
            )
            velocity_1, velocity_2 = _velocities(
                root, r1, r2, coupling, ll, normal, semiperimeter, chord
            )
            root_valid = (
                root_valid
                & jnp.all(jnp.isfinite(velocity_1))
                & jnp.all(jnp.isfinite(velocity_2))
            )
            slot_status = jnp.where(
                ~finite,
                int(AstrodynamicsStatus.NONFINITE_INPUT),
                jnp.where(
                    ~geometry_valid,
                    int(AstrodynamicsStatus.SINGULAR_GEOMETRY),
                    jnp.where(
                        (tof <= 0.0) | (coupling <= 0.0),
                        int(AstrodynamicsStatus.INVALID_DOMAIN),
                        jnp.where(
                            ~found,
                            int(AstrodynamicsStatus.NO_SOLUTION),
                            jnp.where(
                                root_valid,
                                int(AstrodynamicsStatus.SUCCESS),
                                int(AstrodynamicsStatus.NONCONVERGED),
                            ),
                        ),
                    ),
                ),
            ).astype(jnp.int32)
            departure = departure.at[slot].set(jnp.where(root_valid, velocity_1, 0.0))
            arrival = arrival.at[slot].set(jnp.where(root_valid, velocity_2, 0.0))
            valid = valid.at[slot].set(root_valid)
            status = status.at[slot].set(slot_status)
            revolutions = revolutions.at[slot].set(revolution)
            branch = branch.at[slot].set(0 if revolution == 0 else 2 * local_branch - 1)
            roots = roots.at[slot].set(root)
            residuals = residuals.at[slot].set(residual)
            iteration_counts = iteration_counts.at[slot].set(plan.bisection_iterations)

    cosine_angle = jnp.clip(
        jnp.sum(r1 * r2) / jnp.where(r1_norm * r2_norm > 0.0, r1_norm * r2_norm, 1.0),
        -1.0,
        1.0,
    )
    angle = jnp.arccos(cosine_angle)
    angle = jnp.where(plan.long_way, 2.0 * jnp.pi - angle, angle)
    return LambertResult(
        departure,
        arrival,
        valid,
        status,
        revolutions,
        branch,
        roots,
        residuals,
        iteration_counts,
        angle,
        context,
        plan.plan_id,
    )


__all__ = ["LambertPlan", "LambertResult", "solve_lambert"]
