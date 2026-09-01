#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.finite_volume import FiniteVolumeDiscretization
from ...solver._mac_ale import MACALEGeometryPlan, MACALEStageGeometry


FaceTuple = tuple[Array, ...]


def _cell_to_vertices(value: Array, periodic: tuple[bool, bool], /) -> Array:
    result = value
    for axis in (0, 1):
        moved = jnp.moveaxis(result, axis, 0)
        if periodic[axis]:
            unique = 0.5 * (jnp.roll(moved, 1, axis=0) + moved)
            faces = jnp.concatenate((unique, unique[:1]), axis=0)
        else:
            interior = 0.5 * (moved[:-1] + moved[1:])
            faces = jnp.concatenate((moved[:1], interior, moved[-1:]), axis=0)
        result = jnp.moveaxis(faces, 0, axis)
    return result


def _tuple_dot(left: FaceTuple, right: FaceTuple, /) -> Array:
    return sum(jnp.real(jnp.vdot(a, b)) for a, b in zip(left, right, strict=True))


def _tuple_add(left: FaceTuple, scale: Array, right: FaceTuple, /) -> FaceTuple:
    return tuple(a + scale * b for a, b in zip(left, right, strict=True))


def _tuple_scale(scale: Array, value: FaceTuple, /) -> FaceTuple:
    return tuple(scale * component for component in value)


class GraphALEStageArguments(StrictModule):
    eta: Array
    eta_rate: Array
    time_origin: Array
    surface: "PreparedGraphSurfaceALE"
    user_args: Any


class GraphSurfaceGeometryEvidence(StrictModule):
    minimum_height: Array
    maximum_slope: Array
    volume_gcl_residual: Array
    finite: Array
    valid: Array
    surface_id: str = eqx.field(static=True)


class MappedHodgeSolveResult(StrictModule):
    velocity: FaceTuple
    residual_norm: Array
    iterations: Array
    finite: Array
    converged: Array


class SurfaceKinematicResult(StrictModule):
    eta_rate: Array
    target_volume_rate: Array
    reproduced_volume_rate: Array
    residual_norm: Array
    iterations: Array
    converged: Array
    finite: Array


class FreeSurfaceALEState(StrictModule):
    eta: Array
    momentum: FaceTuple
    scalar_content: dict[str, Array]


class FreeSurfaceALEStateView(StrictModule):
    eta: Array
    velocity: FaceTuple
    scalars: dict[str, Array]
    geometry: MACALEStageGeometry
    kinetic_energy: Array
    volume: Array
    view_id: str = eqx.field(static=True)


class GraphSurfaceALEPlan(StrictModule, NonTrainableState):
    """Fixed-topology graph surface mapped onto the native MAC ALE geometry."""

    reference: FiniteVolumeDiscretization
    bottom: Array
    minimum_height: float = eqx.field(static=True)
    maximum_slope: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference: FiniteVolumeDiscretization,
        bottom: ArrayLike,
        /,
        *,
        minimum_height: float = 1.0e-4,
        maximum_slope: float = 1.0,
        tolerance: float = 1.0e-9,
        maximum_iterations: int = 200,
    ):
        if not isinstance(reference, FiniteVolumeDiscretization):
            raise TypeError("reference must be FiniteVolumeDiscretization.")
        if len(reference.cell_shape) != 3:
            raise ValueError("Graph free-surface ALE requires three dimensions.")
        if reference.grid.structured_axes[2].periodic:
            raise ValueError("Graph free-surface vertical axes must be bounded.")
        bottom_ = jnp.asarray(bottom, dtype=reference.cell_volumes.dtype)
        if bottom_.shape != reference.cell_shape[:2]:
            raise ValueError("bottom must match the horizontal cell shape.")
        if bool(jnp.any(~jnp.isfinite(bottom_))):
            raise ValueError("Graph bottom must be finite.")
        values = tuple(
            float(value)
            for value in (
                minimum_height,
                maximum_slope,
                tolerance,
            )
        )
        iterations = int(maximum_iterations)
        if (
            any(not np.isfinite(value) or value <= 0.0 for value in values)
            or iterations <= 0
        ):
            raise ValueError("Invalid graph-surface tolerances or iteration count.")
        self.reference = reference
        self.bottom = bottom_
        self.minimum_height = values[0]
        self.maximum_slope = values[1]
        self.tolerance = values[2]
        self.maximum_iterations = iterations
        self.plan_id = canonical_fingerprint(
            {
                "kind": "graph-surface-ale-plan",
                "reference": reference.prepared_id,
                "minimum_height": values[0],
                "maximum_slope": values[1],
                "tolerance": values[2],
                "maximum_iterations": iterations,
            }
        )

    def prepare(self) -> "PreparedGraphSurfaceALE":
        return PreparedGraphSurfaceALE(self)


class PreparedGraphSurfaceALE(StrictModule):
    """Prepared graph-to-vertex map and compatible kinematic/Hodge operators."""

    plan: GraphSurfaceALEPlan
    ale: MACALEGeometryPlan
    horizontal_area: Array
    bottom_vertices: Array
    surface_id: str = eqx.field(static=True)

    def __init__(self, plan: GraphSurfaceALEPlan, /):
        reference = plan.reference
        x_axis, y_axis, z_axis = reference.grid.structured_axes
        periodic = (x_axis.periodic, y_axis.periodic)
        bottom_vertices = _cell_to_vertices(plan.bottom, periodic)
        horizontal_area = reference.cell_volumes / z_axis.interval_widths[None, None, :]
        horizontal_area = horizontal_area[..., 0]

        def coordinate_map(time, point, args):
            stage = args
            eta = stage.eta + (time - stage.time_origin) * stage.eta_rate
            eta_vertices = _cell_to_vertices(eta, periodic)
            x_points = x_axis.point_coordinates
            y_points = y_axis.point_coordinates
            ix = jnp.argmin(jnp.abs(x_points - point[0]))
            iy = jnp.argmin(jnp.abs(y_points - point[1]))
            surface = eta_vertices[ix, iy]
            bottom = bottom_vertices[ix, iy]
            reference_bottom = z_axis.point_coordinates[0]
            reference_top = z_axis.point_coordinates[-1]
            sigma = (point[2] - reference_bottom) / (reference_top - reference_bottom)
            return jnp.asarray((point[0], point[1], bottom + sigma * (surface - bottom)))

        def grid_velocity(time, point, args):
            del time
            stage = args
            rate_vertices = _cell_to_vertices(stage.eta_rate, periodic)
            x_points = x_axis.point_coordinates
            y_points = y_axis.point_coordinates
            ix = jnp.argmin(jnp.abs(x_points - point[0]))
            iy = jnp.argmin(jnp.abs(y_points - point[1]))
            reference_bottom = z_axis.point_coordinates[0]
            reference_top = z_axis.point_coordinates[-1]
            sigma = (point[2] - reference_bottom) / (reference_top - reference_bottom)
            return jnp.asarray((0.0, 0.0, sigma * rate_vertices[ix, iy]))

        ale = MACALEGeometryPlan(
            reference,
            coordinate_map,
            grid_velocity,
            mapping_id=canonical_fingerprint(
                {"kind": "graph-surface-map", "plan": plan.plan_id}
            ),
            tolerance=plan.tolerance,
            maximum_iterations=plan.maximum_iterations,
        )
        self.plan = plan
        self.ale = ale
        self.horizontal_area = horizontal_area
        self.bottom_vertices = bottom_vertices
        self.surface_id = canonical_fingerprint(
            {"kind": "prepared-graph-surface-ale", "plan": plan.plan_id}
        )

    @property
    def eta_shape(self) -> tuple[int, int]:
        return self.plan.reference.cell_shape[:2]

    def stage_arguments(
        self,
        eta: ArrayLike,
        eta_rate: ArrayLike,
        time_origin: ArrayLike,
        user_args: Any = None,
        /,
    ) -> GraphALEStageArguments:
        eta_ = jnp.asarray(eta, dtype=self.plan.reference.cell_volumes.dtype)
        rate = jnp.asarray(eta_rate, dtype=eta_.dtype)
        if eta_.shape != self.eta_shape or rate.shape != self.eta_shape:
            raise ValueError("Graph eta and eta_rate shapes are invalid.")
        return GraphALEStageArguments(
            eta_, rate, jnp.asarray(time_origin).reshape(()), self, user_args
        )

    def geometry(
        self,
        time: ArrayLike,
        eta: ArrayLike,
        eta_rate: ArrayLike,
        user_args: Any = None,
        /,
    ) -> MACALEStageGeometry:
        arguments = self.stage_arguments(eta, eta_rate, time, user_args)
        return self.ale.evaluate(time, arguments)

    def geometry_evidence(
        self, eta: ArrayLike, eta_rate: ArrayLike, time: ArrayLike = 0.0, /
    ) -> GraphSurfaceGeometryEvidence:
        eta_ = jnp.asarray(eta)
        geometry = self.geometry(time, eta_, eta_rate)
        height = eta_ - self.plan.bottom
        x_axis, y_axis, _ = self.plan.reference.grid.structured_axes
        dx = x_axis.interval_widths[:, None]
        dy = y_axis.interval_widths[None, :]
        slope_x = jnp.gradient(eta_, axis=0) / dx
        slope_y = jnp.gradient(eta_, axis=1) / dy
        slope = jnp.sqrt(slope_x**2 + slope_y**2)
        gcl = jnp.max(jnp.abs(geometry.gcl_residual))
        finite = (
            jnp.all(jnp.isfinite(height))
            & jnp.all(jnp.isfinite(slope))
            & jnp.isfinite(gcl)
            & geometry.finite
        )
        valid = (
            finite
            & geometry.passed
            & jnp.all(height > self.plan.minimum_height)
            & jnp.all(slope <= self.plan.maximum_slope)
        )
        return GraphSurfaceGeometryEvidence(
            minimum_height=jnp.min(height),
            maximum_slope=jnp.max(slope),
            volume_gcl_residual=gcl,
            finite=finite,
            valid=valid,
            surface_id=self.surface_id,
        )

    def kinetic_energy(
        self, geometry: MACALEStageGeometry, velocity: FaceTuple, /
    ) -> Array:
        values = geometry.validate_velocity(velocity)
        diagonal = sum(
            jnp.sum(weight * value**2)
            for weight, value in zip(geometry.face_dual_measures, values, strict=True)
        )
        cells = geometry.reconstruct_cell_velocity(values)
        reconstructed = jnp.sum(geometry.cell_volumes * jnp.sum(cells**2, axis=-1))
        return 0.25 * (diagonal + reconstructed)

    def apply_hodge(
        self, geometry: MACALEStageGeometry, velocity: FaceTuple, /
    ) -> FaceTuple:
        values = geometry.validate_velocity(velocity)
        return jax.grad(lambda candidate: self.kinetic_energy(geometry, candidate))(
            values
        )

    def inverse_hodge(
        self,
        geometry: MACALEStageGeometry,
        momentum: FaceTuple,
        /,
        *,
        free_mask: FaceTuple | None = None,
    ) -> MappedHodgeSolveResult:
        target = tuple(jnp.asarray(value) for value in momentum)
        mask = (
            tuple(jnp.ones_like(value) for value in target)
            if free_mask is None
            else tuple(jnp.asarray(value, dtype=target[0].dtype) for value in free_mask)
        )

        def apply(value):
            masked = tuple(v * m for v, m in zip(value, mask, strict=True))
            image = self.apply_hodge(geometry, masked)
            return tuple(v * m for v, m in zip(image, mask, strict=True))

        value = tuple(jnp.zeros_like(component) for component in target)
        residual = tuple(t * m for t, m in zip(target, mask, strict=True))
        direction = residual
        norm = _tuple_dot(residual, residual)
        threshold = self.plan.tolerance**2 * jnp.maximum(norm, 1.0)
        active = norm > threshold
        failed = jnp.asarray(False)

        def body(_, state):
            current, residual_, direction_, norm_, active_, failed_ = state
            image = apply(direction_)
            denominator = _tuple_dot(direction_, image)
            valid = active_ & jnp.isfinite(denominator) & (denominator > 0.0)
            alpha = jnp.where(valid, norm_ / denominator, 0.0)
            next_value = _tuple_add(current, alpha, direction_)
            next_residual = _tuple_add(residual_, -alpha, image)
            next_norm = _tuple_dot(next_residual, next_residual)
            running = valid & (next_norm > threshold)
            beta = jnp.where(running & (norm_ > 0.0), next_norm / norm_, 0.0)
            next_direction = _tuple_add(next_residual, beta, direction_)
            return (
                next_value,
                next_residual,
                next_direction,
                next_norm,
                running,
                failed_ | (active_ & ~valid),
            )

        value, residual, _, norm, active, failed = jax.lax.fori_loop(
            0,
            self.plan.maximum_iterations,
            body,
            (value, residual, direction, norm, active, failed),
        )
        residual_norm = jnp.sqrt(norm)
        finite = jnp.all(
            jnp.stack(tuple(jnp.all(jnp.isfinite(v)) for v in value))
        ) & jnp.isfinite(residual_norm)
        converged = ~active & ~failed & finite
        return MappedHodgeSolveResult(
            velocity=value,
            residual_norm=residual_norm,
            iterations=jnp.asarray(self.plan.maximum_iterations, dtype=jnp.int32),
            finite=finite,
            converged=converged,
        )

    def top_volume_flux(
        self, geometry: MACALEStageGeometry, velocity: FaceTuple, /
    ) -> Array:
        integrated = velocity[2] * geometry.face_measures[2]
        return jnp.take(integrated, -1, axis=2)

    def _column_volumes(self, eta: Array) -> Array:
        zero = jnp.zeros_like(eta)
        geometry = self.geometry(jnp.asarray(0.0), eta, zero)
        return jnp.sum(geometry.cell_volumes, axis=2)

    def solve_eta_rate(
        self,
        eta: ArrayLike,
        target_volume_rate: ArrayLike,
        /,
    ) -> SurfaceKinematicResult:
        eta_ = jnp.asarray(eta)
        target = jnp.asarray(target_volume_rate, dtype=eta_.dtype)
        if eta_.shape != self.eta_shape or target.shape != self.eta_shape:
            raise ValueError("Surface kinematic shapes are invalid.")

        def action(rate):
            return jax.jvp(self._column_volumes, (eta_,), (rate,))[1]

        value = jnp.zeros_like(eta_)
        residual = target - action(value)
        direction = residual
        norm = jnp.real(jnp.vdot(residual, residual))
        threshold = self.plan.tolerance**2 * jnp.maximum(norm, 1.0)
        active = norm > threshold
        failed = jnp.asarray(False)

        def body(_, state):
            current, residual_, direction_, norm_, active_, failed_ = state
            image = action(direction_)
            denominator = jnp.real(jnp.vdot(direction_, image))
            valid = active_ & jnp.isfinite(denominator) & (denominator > 0.0)
            alpha = jnp.where(valid, norm_ / denominator, 0.0)
            next_value = current + alpha * direction_
            next_residual = residual_ - alpha * image
            next_norm = jnp.real(jnp.vdot(next_residual, next_residual))
            running = valid & (next_norm > threshold)
            beta = jnp.where(running & (norm_ > 0.0), next_norm / norm_, 0.0)
            return (
                next_value,
                next_residual,
                next_residual + beta * direction_,
                next_norm,
                running,
                failed_ | (active_ & ~valid),
            )

        value, residual, _, norm, active, failed = jax.lax.fori_loop(
            0,
            self.plan.maximum_iterations,
            body,
            (value, residual, direction, norm, active, failed),
        )
        reproduced = action(value)
        residual_norm = jnp.sqrt(
            jnp.real(jnp.vdot(reproduced - target, reproduced - target))
        )
        finite = jnp.all(jnp.isfinite(value)) & jnp.isfinite(residual_norm)
        return SurfaceKinematicResult(
            eta_rate=value,
            target_volume_rate=target,
            reproduced_volume_rate=reproduced,
            residual_norm=residual_norm,
            iterations=jnp.asarray(self.plan.maximum_iterations, dtype=jnp.int32),
            converged=~active & ~failed & finite,
            finite=finite,
        )


__all__ = [
    "FreeSurfaceALEState",
    "FreeSurfaceALEStateView",
    "GraphALEStageArguments",
    "GraphSurfaceALEPlan",
    "GraphSurfaceGeometryEvidence",
    "MappedHodgeSolveResult",
    "PreparedGraphSurfaceALE",
    "SurfaceKinematicResult",
]
