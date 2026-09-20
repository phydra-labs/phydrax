#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntFlag

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein
from phydrax.metrix._adm_exchange import ADMGridGeometry

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._surfaces import SphericalSpectralSurface


class EventHorizonStatus(IntFlag):
    """Status bits for offline global generator reconstruction."""

    SUCCESS = 0
    NONFINITE = 1
    NOT_CONVERGED = 2
    OUTSIDE_HISTORY_COVERAGE = 4
    NONNULL_GENERATOR_FLOW = 8
    CAUSTIC_DETECTED = 16
    TERMINAL_SURFACE_UNQUALIFIED = 32
    DERIVATIVE_INVALID = 64
    GEODESIC_TRANSPORT_INVALID = 128


def _integer_capacity(value: int, name: str, /, *, minimum: int = 1) -> int:
    raw = np.asarray(value)
    if raw.shape != () or not np.issubdtype(raw.dtype, np.integer):
        raise TypeError(f"{name} must be one integer.")
    result = int(raw)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


class CompletedSpacetimeHistory(StrictModule, NonTrainableState):
    """Immutable completed ADM history for an offline global horizon trace.

    The entire time history is one fixed-shape ADM batch. Construction rejects
    live or partial histories. Generator directions are not prescribed by this
    storage object: the tracing plan evolves terminal null covectors with the
    time-dependent 3+1 Hamilton equations.
    """

    times: Array
    x_coordinates: Array
    y_coordinates: Array
    z_coordinates: Array
    geometry: ADMGridGeometry
    time_capacity: int = eqx.field(static=True)
    grid_shape: tuple[int, int, int] = eqx.field(static=True)
    completion_id: str = eqx.field(static=True)
    history_id: str = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        x_coordinates: ArrayLike,
        y_coordinates: ArrayLike,
        z_coordinates: ArrayLike,
        geometry: ADMGridGeometry,
        /,
        *,
        completed: bool,
        completion_id: str,
        history_name: str = "completed-spacetime-history",
    ):
        if completed is not True:
            raise ValueError(
                "Event-horizon tracing requires an explicitly completed spacetime history."
            )
        if not isinstance(completion_id, str) or not completion_id:
            raise ValueError(
                "completion_id must identify the immutable completed history."
            )
        if not isinstance(history_name, str) or not history_name:
            raise ValueError("history_name must be a nonempty string.")
        times_host = np.asarray(times, dtype=np.float64)
        axes = tuple(
            np.asarray(axis, dtype=np.float64)
            for axis in (x_coordinates, y_coordinates, z_coordinates)
        )
        if (
            times_host.ndim != 1
            or times_host.size < 2
            or np.any(~np.isfinite(times_host))
            or np.any(np.diff(times_host) <= 0.0)
        ):
            raise ValueError("Completed history times must be finite and increasing.")
        if any(
            axis.ndim != 1
            or axis.size < 2
            or np.any(~np.isfinite(axis))
            or np.any(np.diff(axis) <= 0.0)
            for axis in axes
        ):
            raise ValueError("Cartesian history axes must be finite and increasing.")
        grid_shape = tuple(axis.size for axis in axes)
        leading_shape = (times_host.size,) + grid_shape
        if geometry.leading_shape != leading_shape:
            raise ValueError(
                "ADM geometry leading shape must be (times, x, y, z) for offline tracing."
            )

        self.times = jnp.asarray(times_host, dtype=geometry.alpha.dtype)
        self.x_coordinates = jnp.asarray(axes[0], dtype=geometry.alpha.dtype)
        self.y_coordinates = jnp.asarray(axes[1], dtype=geometry.alpha.dtype)
        self.z_coordinates = jnp.asarray(axes[2], dtype=geometry.alpha.dtype)
        self.geometry = geometry
        self.time_capacity = times_host.size
        self.grid_shape = grid_shape
        self.completion_id = completion_id
        self.history_id = canonical_fingerprint(
            {
                "kind": "completed-offline-spacetime-history",
                "name": history_name,
                "completion_id": completion_id,
                "times": times_host,
                "axes": axes,
                "geometry_lineage_id": geometry.geometry_lineage_id,
                "snapshot_token": geometry.snapshot_token,
                "alpha": geometry.alpha,
                "beta": geometry.beta_contravariant,
                "spatial_metric": geometry.spatial_metric,
                "inverse_spatial_metric": geometry.inverse_spatial_metric,
                "active": geometry.active,
                "valid": geometry.valid,
            }
        )


class OfflineEventHorizonTrace(StrictModule):
    """Hamilton-transported generators from one completed spacetime history.

    This is a global, offline candidate. ``caustic`` records crease/entry
    evidence and invalidates trajectory derivatives, but does not relabel an
    individual spatial slice as an event horizon.
    """

    times: Array
    generator_trajectories: Array
    generator_covectors: Array
    generator_active: Array
    coverage: Array
    caustic: Array
    convergence_error: Array
    covector_convergence_error: Array
    convergence_ratio: Array
    null_residual: Array
    geodesic_residual: Array
    coverage_complete: Array
    caustic_detected: Array
    global_history_complete: Array
    status: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    history_id: str = eqx.field(static=True)
    terminal_surface_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    trace_id: str = eqx.field(static=True)


def _axis_bracket(axis: Array, query: Array, /) -> tuple[Array, Array, Array]:
    index = jnp.searchsorted(axis, query, side="right") - 1
    index = jnp.clip(index, 0, axis.size - 2)
    lower = axis[index]
    upper = axis[index + 1]
    weight = (query - lower) / (upper - lower)
    covered = (query >= axis[0]) & (query <= axis[-1])
    return index, weight, covered


def _coordinate_derivative(values: Array, nodes: Array, /) -> Array:
    previous_width = nodes[1:-1] - nodes[:-2]
    next_width = nodes[2:] - nodes[1:-1]
    previous_weight = -next_width / (previous_width * (previous_width + next_width))
    center_weight = (next_width - previous_width) / (previous_width * next_width)
    next_weight = previous_width / (next_width * (previous_width + next_width))
    extra_axes = (1,) * (values.ndim - 1)
    interior = (
        previous_weight.reshape((-1,) + extra_axes) * values[:-2]
        + center_weight.reshape((-1,) + extra_axes) * values[1:-1]
        + next_weight.reshape((-1,) + extra_axes) * values[2:]
    )
    first = (values[1] - values[0]) / (nodes[1] - nodes[0])
    last = (values[-1] - values[-2]) / (nodes[-1] - nodes[-2])
    return jnp.concatenate((first[None], interior, last[None]), axis=0)


def _grid_derivative(values: Array, nodes: Array, axis: int, /) -> Array:
    moved = jnp.moveaxis(values, axis, 0)
    derivative = _coordinate_derivative(moved, nodes)
    return jnp.moveaxis(derivative, 0, axis)


def _spatial_interpolate(
    values: Array,
    x_axis: Array,
    y_axis: Array,
    z_axis: Array,
    positions: Array,
    /,
) -> tuple[Array, Array]:
    ix, wx, covered_x = _axis_bracket(x_axis, positions[:, 0])
    iy, wy, covered_y = _axis_bracket(y_axis, positions[:, 1])
    iz, wz, covered_z = _axis_bracket(z_axis, positions[:, 2])
    tail_axes = (1,) * (values.ndim - 3)
    result = jnp.zeros(
        (positions.shape[0],) + values.shape[3:],
        dtype=values.dtype,
    )
    for ox in (0, 1):
        x_weight = wx if ox else 1.0 - wx
        for oy in (0, 1):
            y_weight = wy if oy else 1.0 - wy
            for oz in (0, 1):
                z_weight = wz if oz else 1.0 - wz
                weight = (x_weight * y_weight * z_weight).reshape((-1,) + tail_axes)
                result = result + weight * values[ix + ox, iy + oy, iz + oz]
    return result, covered_x & covered_y & covered_z


def _spacetime_interpolate(
    values: Array,
    history: CompletedSpacetimeHistory,
    time: Array,
    positions: Array,
    /,
) -> tuple[Array, Array]:
    time_index = jnp.searchsorted(history.times, time, side="right") - 1
    time_index = jnp.clip(time_index, 0, history.time_capacity - 2)
    lower_time = history.times[time_index]
    upper_time = history.times[time_index + 1]
    time_weight = (time - lower_time) / (upper_time - lower_time)
    lower, lower_covered = _spatial_interpolate(
        values[time_index],
        history.x_coordinates,
        history.y_coordinates,
        history.z_coordinates,
        positions,
    )
    upper, upper_covered = _spatial_interpolate(
        values[time_index + 1],
        history.x_coordinates,
        history.y_coordinates,
        history.z_coordinates,
        positions,
    )
    interpolated = lower + time_weight * (upper - lower)
    time_covered = (time >= history.times[0]) & (time <= history.times[-1])
    return interpolated, lower_covered & upper_covered & time_covered


def _sample_hamilton_flow(
    history: CompletedSpacetimeHistory,
    support_field: Array,
    gradient_alpha: Array,
    gradient_beta: Array,
    gradient_inverse_metric: Array,
    time: Array,
    positions: Array,
    covectors: Array,
    null_tolerance: float,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    alpha, covered = _spacetime_interpolate(
        history.geometry.alpha, history, time, positions
    )
    beta, _ = _spacetime_interpolate(
        history.geometry.beta_contravariant, history, time, positions
    )
    spatial_metric, _ = _spacetime_interpolate(
        history.geometry.spatial_metric, history, time, positions
    )
    inverse_metric, _ = _spacetime_interpolate(
        history.geometry.inverse_spatial_metric, history, time, positions
    )
    alpha_gradient, _ = _spacetime_interpolate(gradient_alpha, history, time, positions)
    beta_gradient, _ = _spacetime_interpolate(gradient_beta, history, time, positions)
    inverse_gradient, _ = _spacetime_interpolate(
        gradient_inverse_metric, history, time, positions
    )
    support, _ = _spacetime_interpolate(support_field, history, time, positions)

    raised_covector = ein.contract("gij,gj->gi", inverse_metric, covectors)
    spatial_momentum_squared = ein.contract("gi,gi->g", covectors, raised_covector)
    tiny = jnp.finfo(alpha.dtype).tiny
    frequency = jnp.sqrt(jnp.maximum(spatial_momentum_squared, tiny))
    relative_velocity = alpha[:, None] * raised_covector / frequency[:, None]
    coordinate_velocity = relative_velocity - beta
    inverse_force = ein.contract("gijk,gj,gk->gi", inverse_gradient, covectors, covectors)
    shift_force = ein.contract("gij,gj->gi", beta_gradient, covectors)
    covector_rate = (
        -frequency[:, None] * alpha_gradient
        - alpha[:, None] * inverse_force / (2.0 * frequency[:, None])
        + shift_force
    )

    spatial_speed_squared = ein.contract(
        "gij,gi,gj->g", spatial_metric, relative_velocity, relative_velocity
    )
    null_residual = jnp.abs(spatial_speed_squared - alpha * alpha) / jnp.maximum(
        alpha * alpha, tiny
    )
    support_valid = support >= 1.0 - 10.0 * jnp.finfo(support.dtype).eps
    null_valid = (
        (null_residual <= null_tolerance)
        & (alpha > 0.0)
        & (spatial_momentum_squared > tiny)
    )
    finite = (
        jnp.all(jnp.isfinite(coordinate_velocity), axis=-1)
        & jnp.all(jnp.isfinite(covector_rate), axis=-1)
        & jnp.isfinite(null_residual)
    )
    return (
        coordinate_velocity,
        covector_rate,
        covered & support_valid,
        null_valid,
        null_residual,
        finite,
    )


def _rk4_step(
    history: CompletedSpacetimeHistory,
    support_field: Array,
    gradient_alpha: Array,
    gradient_beta: Array,
    gradient_inverse_metric: Array,
    positions: Array,
    covectors: Array,
    start_time: Array,
    step: Array,
    null_tolerance: float,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    def sample(time, points, momenta):
        return _sample_hamilton_flow(
            history,
            support_field,
            gradient_alpha,
            gradient_beta,
            gradient_inverse_metric,
            time,
            points,
            momenta,
            null_tolerance,
        )

    k1x, k1p, covered1, null1, residual1, finite1 = sample(
        start_time, positions, covectors
    )
    k2x, k2p, covered2, null2, residual2, finite2 = sample(
        start_time + 0.5 * step,
        positions + 0.5 * step * k1x,
        covectors + 0.5 * step * k1p,
    )
    k3x, k3p, covered3, null3, residual3, finite3 = sample(
        start_time + 0.5 * step,
        positions + 0.5 * step * k2x,
        covectors + 0.5 * step * k2p,
    )
    k4x, k4p, covered4, null4, residual4, finite4 = sample(
        start_time + step,
        positions + step * k3x,
        covectors + step * k3p,
    )
    candidate_positions = positions + step * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
    candidate_covectors = covectors + step * (k1p + 2.0 * k2p + 2.0 * k3p + k4p) / 6.0
    covered = covered1 & covered2 & covered3 & covered4
    null_valid = null1 & null2 & null3 & null4
    finite = finite1 & finite2 & finite3 & finite4
    residual = jnp.maximum(
        jnp.maximum(residual1, residual2), jnp.maximum(residual3, residual4)
    )
    return (
        candidate_positions,
        candidate_covectors,
        covered,
        null_valid,
        residual,
        finite,
    )


class OfflineEventHorizonTracingPlan(StrictModule, NonTrainableState):
    """Fixed-capacity Hamilton/RK4 plan for global backward generators."""

    terminal_surface: SphericalSpectralSurface
    terminal_positions: Array
    terminal_covectors: Array
    terminal_active: Array
    neighbor_indices: Array
    time_capacity: int = eqx.field(static=True)
    grid_shape: tuple[int, int, int] = eqx.field(static=True)
    generator_capacity: int = eqx.field(static=True)
    neighbor_capacity: int = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    covector_absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    null_tolerance: float = eqx.field(static=True)
    caustic_distance: float = eqx.field(static=True)
    terminal_surface_qualified: bool = eqx.field(static=True)
    terminal_surface_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        terminal_surface: SphericalSpectralSurface,
        terminal_positions: ArrayLike,
        terminal_covectors: ArrayLike,
        terminal_active: ArrayLike,
        neighbor_indices: ArrayLike,
        /,
        *,
        time_capacity: int,
        grid_shape: tuple[int, int, int],
        terminal_surface_qualified: bool,
        absolute_tolerance: float = 1.0e-8,
        covector_absolute_tolerance: float = 1.0e-8,
        relative_tolerance: float = 1.0e-6,
        null_tolerance: float = 1.0e-6,
        caustic_distance: float = 1.0e-6,
        plan_name: str = "offline-event-horizon-generator-trace",
    ):
        if not isinstance(terminal_surface, SphericalSpectralSurface):
            raise TypeError("terminal_surface must be a SphericalSpectralSurface.")
        positions = np.asarray(terminal_positions, dtype=np.float64)
        covectors = np.asarray(terminal_covectors, dtype=np.float64)
        active = np.asarray(terminal_active, dtype=np.bool_)
        neighbors = np.asarray(neighbor_indices)
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
            raise ValueError("terminal_positions must have shape (generators, 3).")
        generator_count = positions.shape[0]
        if covectors.shape != positions.shape:
            raise ValueError("terminal_covectors must match terminal_positions.")
        if active.shape != (generator_count,):
            raise ValueError("terminal_active must have one entry per generator.")
        if not np.any(active):
            raise ValueError("At least one terminal generator must be active.")
        if (
            np.any(~np.isfinite(positions))
            or np.any(~np.isfinite(covectors))
            or np.any(active & (np.sum(covectors * covectors, axis=-1) == 0.0))
        ):
            raise ValueError(
                "Active terminal positions/covectors must be finite and nonzero."
            )
        if (
            neighbors.ndim != 2
            or neighbors.shape[0] != generator_count
            or neighbors.shape[1] == 0
            or not np.issubdtype(neighbors.dtype, np.integer)
        ):
            raise TypeError(
                "neighbor_indices must be an integer generator-neighbor table."
            )
        if np.any((neighbors < -1) | (neighbors >= generator_count)):
            raise ValueError("neighbor_indices must use generator indices or -1 padding.")
        if np.any(neighbors == np.arange(generator_count)[:, None]):
            raise ValueError("A generator cannot be its own caustic neighbor.")
        time_count = _integer_capacity(time_capacity, "time_capacity", minimum=2)
        if len(grid_shape) != 3:
            raise ValueError("grid_shape must contain three Cartesian capacities.")
        grid_shape_ = tuple(
            _integer_capacity(value, "grid_shape entry", minimum=2)
            for value in grid_shape
        )
        absolute = float(absolute_tolerance)
        covector_absolute = float(covector_absolute_tolerance)
        relative = float(relative_tolerance)
        null = float(null_tolerance)
        caustic = float(caustic_distance)
        if (
            not np.isfinite(absolute)
            or not np.isfinite(covector_absolute)
            or not np.isfinite(relative)
            or not np.isfinite(null)
            or not np.isfinite(caustic)
            or absolute <= 0.0
            or covector_absolute <= 0.0
            or relative < 0.0
            or null <= 0.0
            or caustic <= 0.0
        ):
            raise ValueError("Tracing tolerances and caustic distance are inadmissible.")
        if not isinstance(terminal_surface_qualified, bool):
            raise TypeError("terminal_surface_qualified must be one host boolean.")
        if not isinstance(plan_name, str) or not plan_name:
            raise ValueError("plan_name must be a nonempty string.")
        terminal_surface_id = canonical_fingerprint(
            {
                "kind": "terminal-marginal-surface",
                "surface_plan": terminal_surface.plan_id,
                "coefficients": terminal_surface.coefficients,
                "center": terminal_surface.center,
            }
        )

        self.terminal_surface = terminal_surface
        self.terminal_positions = jnp.asarray(positions)
        self.terminal_covectors = jnp.asarray(covectors)
        self.terminal_active = jnp.asarray(active)
        self.neighbor_indices = jnp.asarray(neighbors, dtype=jnp.int32)
        self.time_capacity = time_count
        self.grid_shape = grid_shape_
        self.generator_capacity = generator_count
        self.neighbor_capacity = neighbors.shape[1]
        self.absolute_tolerance = absolute
        self.covector_absolute_tolerance = covector_absolute
        self.relative_tolerance = relative
        self.null_tolerance = null
        self.caustic_distance = caustic
        self.terminal_surface_qualified = terminal_surface_qualified
        self.terminal_surface_id = terminal_surface_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-capacity-offline-event-horizon-hamilton-plan",
                "name": plan_name,
                "terminal_surface": terminal_surface_id,
                "terminal_positions": positions,
                "terminal_covectors": covectors,
                "terminal_active": active,
                "neighbor_indices": neighbors,
                "time_capacity": time_count,
                "grid_shape": grid_shape_,
                "terminal_surface_qualified": terminal_surface_qualified,
                "absolute_tolerance": absolute,
                "covector_absolute_tolerance": covector_absolute,
                "relative_tolerance": relative,
                "null_tolerance": null,
                "caustic_distance": caustic,
            }
        )

    def trace(self, history: CompletedSpacetimeHistory, /) -> OfflineEventHorizonTrace:
        if (
            history.time_capacity != self.time_capacity
            or history.grid_shape != self.grid_shape
        ):
            raise ValueError("Completed history does not match tracing capacities.")

        geometry = history.geometry
        support_field = (geometry.active & geometry.physically_valid).astype(
            geometry.alpha.dtype
        )
        axes = (
            history.x_coordinates,
            history.y_coordinates,
            history.z_coordinates,
        )
        gradient_alpha = jnp.stack(
            tuple(
                _grid_derivative(geometry.alpha, axis_nodes, axis_index + 1)
                for axis_index, axis_nodes in enumerate(axes)
            ),
            axis=-1,
        )
        gradient_beta = jnp.stack(
            tuple(
                _grid_derivative(geometry.beta_contravariant, axis_nodes, axis_index + 1)
                for axis_index, axis_nodes in enumerate(axes)
            ),
            axis=-2,
        )
        gradient_inverse_metric = jnp.stack(
            tuple(
                _grid_derivative(
                    geometry.inverse_spatial_metric, axis_nodes, axis_index + 1
                )
                for axis_index, axis_nodes in enumerate(axes)
            ),
            axis=-3,
        )

        terminal_inverse, terminal_interpolation_covered = _spacetime_interpolate(
            geometry.inverse_spatial_metric,
            history,
            history.times[-1],
            self.terminal_positions,
        )
        terminal_norm = jnp.sqrt(
            jnp.maximum(
                ein.contract(
                    "gij,gi,gj->g",
                    terminal_inverse,
                    self.terminal_covectors,
                    self.terminal_covectors,
                ),
                jnp.finfo(geometry.alpha.dtype).tiny,
            )
        )
        terminal_covectors = self.terminal_covectors / terminal_norm[:, None]
        reverse_indices = jnp.arange(self.time_capacity - 1, 0, -1)

        def step(carry, upper_index):
            positions, covectors, active = carry
            upper_time = history.times[upper_index]
            lower_time = history.times[upper_index - 1]
            interval = lower_time - upper_time
            coarse = _rk4_step(
                history,
                support_field,
                gradient_alpha,
                gradient_beta,
                gradient_inverse_metric,
                positions,
                covectors,
                upper_time,
                interval,
                self.null_tolerance,
            )
            half = _rk4_step(
                history,
                support_field,
                gradient_alpha,
                gradient_beta,
                gradient_inverse_metric,
                positions,
                covectors,
                upper_time,
                0.5 * interval,
                self.null_tolerance,
            )
            fine = _rk4_step(
                history,
                support_field,
                gradient_alpha,
                gradient_beta,
                gradient_inverse_metric,
                half[0],
                half[1],
                upper_time + 0.5 * interval,
                0.5 * interval,
                self.null_tolerance,
            )
            covered = coarse[2] & half[2] & fine[2]
            null_valid = coarse[3] & half[3] & fine[3]
            residual = jnp.maximum(coarse[4], jnp.maximum(half[4], fine[4]))
            position_error = (
                jnp.sqrt(
                    ein.contract("gi,gi->g", fine[0] - coarse[0], fine[0] - coarse[0])
                )
                / 15.0
            )
            covector_error = (
                jnp.sqrt(
                    ein.contract("gi,gi->g", fine[1] - coarse[1], fine[1] - coarse[1])
                )
                / 15.0
            )
            finite_candidate = (
                coarse[5]
                & half[5]
                & fine[5]
                & jnp.all(jnp.isfinite(fine[0]), axis=-1)
                & jnp.all(jnp.isfinite(fine[1]), axis=-1)
                & jnp.isfinite(position_error)
                & jnp.isfinite(covector_error)
            )
            accepted = active & covered & null_valid & finite_candidate
            next_positions = jnp.where(accepted[:, None], fine[0], positions)
            next_covectors = jnp.where(accepted[:, None], fine[1], covectors)
            recorded_coverage = (~active) | covered
            recorded_null = (~active) | null_valid
            return (next_positions, next_covectors, accepted), (
                next_positions,
                next_covectors,
                accepted,
                position_error,
                covector_error,
                recorded_coverage,
                recorded_null,
                residual,
            )

        (_, _, _), reverse_output = jax.lax.scan(
            step,
            (self.terminal_positions, terminal_covectors, self.terminal_active),
            reverse_indices,
        )
        (
            reverse_positions,
            reverse_covectors,
            reverse_active,
            reverse_position_error,
            reverse_covector_error,
            reverse_coverage,
            reverse_null,
            reverse_residual,
        ) = reverse_output
        terminal_sample = _sample_hamilton_flow(
            history,
            support_field,
            gradient_alpha,
            gradient_beta,
            gradient_inverse_metric,
            history.times[-1],
            self.terminal_positions,
            terminal_covectors,
            self.null_tolerance,
        )
        terminal_coverage = terminal_interpolation_covered & terminal_sample[2]
        terminal_null = terminal_sample[3]
        terminal_residual = terminal_sample[4]
        trajectories = jnp.concatenate(
            (jnp.flip(reverse_positions, axis=0), self.terminal_positions[None]),
            axis=0,
        )
        covector_trajectories = jnp.concatenate(
            (jnp.flip(reverse_covectors, axis=0), terminal_covectors[None]), axis=0
        )
        generator_active = jnp.concatenate(
            (jnp.flip(reverse_active, axis=0), self.terminal_active[None]), axis=0
        )
        coverage = jnp.concatenate(
            (
                jnp.flip(reverse_coverage, axis=0),
                ((~self.terminal_active) | terminal_coverage)[None],
            ),
            axis=0,
        )
        null_valid = jnp.concatenate(
            (
                jnp.flip(reverse_null, axis=0),
                ((~self.terminal_active) | terminal_null)[None],
            ),
            axis=0,
        )
        convergence_error = jnp.concatenate(
            (
                jnp.flip(reverse_position_error, axis=0),
                jnp.zeros((1, self.generator_capacity), dtype=trajectories.dtype),
            ),
            axis=0,
        )
        covector_convergence_error = jnp.concatenate(
            (
                jnp.flip(reverse_covector_error, axis=0),
                jnp.zeros((1, self.generator_capacity), dtype=trajectories.dtype),
            ),
            axis=0,
        )
        null_residual = jnp.concatenate(
            (jnp.flip(reverse_residual, axis=0), terminal_residual[None]), axis=0
        )
        trajectory_scale = jnp.maximum(
            jnp.sqrt(ein.contract("tgi,tgi->tg", trajectories, trajectories)), 1.0
        )
        covector_scale = jnp.maximum(
            jnp.sqrt(
                ein.contract("tgi,tgi->tg", covector_trajectories, covector_trajectories)
            ),
            1.0,
        )
        position_limit = (
            self.absolute_tolerance + self.relative_tolerance * trajectory_scale
        )
        covector_limit = (
            self.covector_absolute_tolerance + self.relative_tolerance * covector_scale
        )
        convergence_ratio = jnp.maximum(
            convergence_error / position_limit,
            covector_convergence_error / covector_limit,
        )
        geodesic_residual = jnp.sqrt(convergence_error**2 + covector_convergence_error**2)

        safe_neighbors = jnp.clip(self.neighbor_indices, 0, self.generator_capacity - 1)
        neighbor_positions = trajectories[:, safe_neighbors, :]
        separation = trajectories[:, :, None, :] - neighbor_positions
        neighbor_distance = jnp.sqrt(
            ein.contract("tgni,tgni->tgn", separation, separation)
        )
        neighbor_present = self.neighbor_indices >= 0
        neighbor_active = generator_active[:, safe_neighbors]
        caustic = jnp.any(
            neighbor_present[None]
            & generator_active[:, :, None]
            & neighbor_active
            & (neighbor_distance <= self.caustic_distance),
            axis=-1,
        )
        caustic_detected = jnp.any(caustic)
        coverage_complete = jnp.all(coverage)
        finite = (
            jnp.all(jnp.isfinite(trajectories))
            & jnp.all(jnp.isfinite(covector_trajectories))
            & jnp.all(jnp.isfinite(convergence_error))
            & jnp.all(jnp.isfinite(covector_convergence_error))
            & jnp.all(jnp.isfinite(null_residual))
            & jnp.all(geometry.finite)
        )
        geodesic_valid = jnp.all((~generator_active) | (convergence_ratio <= 1.0))
        converged = geodesic_valid
        physically_valid = (
            jnp.all(null_valid)
            & jnp.asarray(self.terminal_surface_qualified)
            & jnp.all(~geometry.active | geometry.physically_valid)
        )
        qualified = finite & converged & physically_valid & coverage_complete
        derivative_valid = qualified & (~caustic_detected)
        status = jnp.asarray(int(EventHorizonStatus.SUCCESS), dtype=jnp.int32)
        status = status | jnp.where(finite, 0, int(EventHorizonStatus.NONFINITE)).astype(
            jnp.int32
        )
        status = status | jnp.where(
            converged, 0, int(EventHorizonStatus.NOT_CONVERGED)
        ).astype(jnp.int32)
        status = status | jnp.where(
            coverage_complete, 0, int(EventHorizonStatus.OUTSIDE_HISTORY_COVERAGE)
        ).astype(jnp.int32)
        status = status | jnp.where(
            jnp.all(null_valid), 0, int(EventHorizonStatus.NONNULL_GENERATOR_FLOW)
        ).astype(jnp.int32)
        status = status | jnp.where(
            geodesic_valid, 0, int(EventHorizonStatus.GEODESIC_TRANSPORT_INVALID)
        ).astype(jnp.int32)
        status = status | jnp.where(
            ~caustic_detected, 0, int(EventHorizonStatus.CAUSTIC_DETECTED)
        ).astype(jnp.int32)
        status = status | jnp.where(
            self.terminal_surface_qualified,
            0,
            int(EventHorizonStatus.TERMINAL_SURFACE_UNQUALIFIED),
        ).astype(jnp.int32)
        status = status | jnp.where(
            derivative_valid, 0, int(EventHorizonStatus.DERIVATIVE_INVALID)
        ).astype(jnp.int32)
        trace_id = canonical_fingerprint(
            {
                "kind": "offline-global-event-horizon-hamilton-trace",
                "history": history.history_id,
                "plan": self.plan_id,
            }
        )
        return OfflineEventHorizonTrace(
            history.times,
            trajectories,
            covector_trajectories,
            generator_active,
            coverage,
            caustic,
            convergence_error,
            covector_convergence_error,
            convergence_ratio,
            null_residual,
            geodesic_residual,
            coverage_complete,
            caustic_detected,
            jnp.asarray(True),
            status,
            finite,
            converged,
            physically_valid,
            qualified,
            derivative_valid,
            history.history_id,
            self.terminal_surface_id,
            self.plan_id,
            trace_id,
        )
