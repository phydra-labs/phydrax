#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import product
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._numerics._compensated import two_sum
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import FunctionLinearOperator, OperatorProperties
from .._lagrangian_marker import LagrangianMarkerDiscretization
from ._incompressible import FaceVelocity, PreparedMACOperators


MACMarkerKernelName: TypeAlias = Literal[
    "cubic-bspline",
    "peskin-four-point",
    "roma-three-point",
]
MACMarkerAccumulation: TypeAlias = Literal[
    "fast",
    "deterministic",
    "compensated",
]


class MACMarkerKernelPlan(StrictModule, NonTrainableState):
    """Qualified one-dimensional regularized-delta kernel family."""

    name: MACMarkerKernelName = eqx.field(static=True)
    width: int = eqx.field(static=True)
    regularity: int = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)

    def __init__(self, name: MACMarkerKernelName = "cubic-bspline", /):
        if name not in (
            "cubic-bspline",
            "peskin-four-point",
            "roma-three-point",
        ):
            raise ValueError("Unknown MAC marker kernel.")
        width = 3 if name == "roma-three-point" else 4
        regularity = 1 if name != "cubic-bspline" else 2
        self.name = name
        self.width = width
        self.regularity = regularity
        self.kernel_id = canonical_fingerprint(
            {
                "kind": "mac-marker-kernel",
                "name": name,
                "width": width,
                "regularity": regularity,
            }
        )


def _kernel_value(name: MACMarkerKernelName, coordinate: Array, /) -> Array:
    absolute = jnp.abs(coordinate)
    if name == "cubic-bspline":
        central = 2.0 / 3.0 - absolute**2 + 0.5 * absolute**3
        outer = jnp.maximum(2.0 - absolute, 0.0) ** 3 / 6.0
        return jnp.where(
            absolute < 1.0,
            central,
            jnp.where(absolute < 2.0, outer, 0.0),
        )
    if name == "peskin-four-point":
        inner_root = jnp.sqrt(jnp.maximum(1.0 + 4.0 * absolute - 4.0 * absolute**2, 0.0))
        outer_root = jnp.sqrt(
            jnp.maximum(-7.0 + 12.0 * absolute - 4.0 * absolute**2, 0.0)
        )
        return jnp.where(
            absolute < 1.0,
            (3.0 - 2.0 * absolute + inner_root) / 8.0,
            jnp.where(
                absolute < 2.0,
                (5.0 - 2.0 * absolute - outer_root) / 8.0,
                0.0,
            ),
        )
    inner = (1.0 + jnp.sqrt(jnp.maximum(1.0 - 3.0 * absolute**2, 0.0))) / 3.0
    outer = (
        5.0
        - 3.0 * absolute
        - jnp.sqrt(jnp.maximum(1.0 - 3.0 * (1.0 - absolute) ** 2, 0.0))
    ) / 6.0
    return jnp.where(
        absolute <= 0.5,
        inner,
        jnp.where(absolute < 1.5, outer, 0.0),
    )


def _kernel_basis(name: MACMarkerKernelName, coordinate: Array, /) -> tuple[Array, Array]:
    absolute = jnp.abs(coordinate)
    sign = jnp.sign(coordinate)
    value = _kernel_value(name, coordinate)
    if name == "cubic-bspline":
        central = sign * (-2.0 * absolute + 1.5 * absolute**2)
        outer = -0.5 * sign * jnp.maximum(2.0 - absolute, 0.0) ** 2
        derivative = jnp.where(
            absolute < 1.0,
            central,
            jnp.where(absolute < 2.0, outer, 0.0),
        )
        return value, derivative
    if name == "peskin-four-point":
        inner_root = jnp.sqrt(
            jnp.maximum(
                1.0 + 4.0 * absolute - 4.0 * absolute**2,
                jnp.finfo(coordinate.dtype).tiny,
            )
        )
        outer_root = jnp.sqrt(
            jnp.maximum(
                -7.0 + 12.0 * absolute - 4.0 * absolute**2,
                jnp.finfo(coordinate.dtype).tiny,
            )
        )
        inner = (-2.0 + (4.0 - 8.0 * absolute) / (2.0 * inner_root)) / 8.0
        outer = (-2.0 - (12.0 - 8.0 * absolute) / (2.0 * outer_root)) / 8.0
        derivative = sign * jnp.where(
            absolute < 1.0,
            inner,
            jnp.where(absolute < 2.0, outer, 0.0),
        )
        return value, derivative
    inner_root = jnp.sqrt(
        jnp.maximum(
            1.0 - 3.0 * absolute**2,
            jnp.finfo(coordinate.dtype).tiny,
        )
    )
    outer_root = jnp.sqrt(
        jnp.maximum(
            1.0 - 3.0 * (1.0 - absolute) ** 2,
            jnp.finfo(coordinate.dtype).tiny,
        )
    )
    inner = -absolute / inner_root
    outer = (-3.0 - 3.0 * (1.0 - absolute) / outer_root) / 6.0
    derivative = sign * jnp.where(
        absolute <= 0.5,
        inner,
        jnp.where(absolute < 1.5, outer, 0.0),
    )
    return value, derivative


def _uniform_spacing(coordinates, bounds, periodic, /) -> float | None:
    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 1 or values.size < 4 or np.any(~np.isfinite(values)):
        raise ValueError("Marker assignment requires four finite axis entities.")
    differences = np.diff(values)
    spacing = float(differences[0])
    tolerance = np.finfo(float).eps * max(32.0, abs(spacing) * values.size)
    uniform = spacing > 0.0 and np.allclose(
        differences, spacing, rtol=1.0e-10, atol=tolerance
    )
    if uniform and periodic:
        expected = float(bounds[1] - bounds[0]) / values.size
        uniform = np.isclose(spacing, expected, rtol=1.0e-10, atol=tolerance)
    return spacing if uniform else None


def _nonuniform_affine_weights(
    name: MACMarkerKernelName,
    source: Array,
    targets: Array,
    /,
) -> Array:
    offsets = targets - source
    scale = jnp.maximum(jnp.max(jnp.abs(offsets)), jnp.finfo(source.dtype).eps)
    raw = _kernel_value(name, offsets / scale)
    basis = jnp.stack((jnp.ones_like(offsets), offsets), axis=0)
    gram = (basis * raw[None, :]) @ basis.T
    regularizer = jnp.finfo(source.dtype).eps * jnp.maximum(jnp.trace(gram), 1.0)
    coefficients = jnp.linalg.solve(
        gram + regularizer * jnp.eye(2, dtype=source.dtype),
        jnp.asarray((1.0, 0.0), dtype=source.dtype),
    )
    weights = raw * (basis.T @ coefficients)
    return weights / jnp.sum(weights)


def _axis_stencil(coordinates, bounds, periodic, position, active, kernel, /):
    count = int(coordinates.size)
    width = kernel.width
    if count < width:
        raise ValueError("Marker kernel width exceeds an axis entity count.")
    spacing_value = _uniform_spacing(coordinates, bounds, periodic)
    lower, upper = bounds
    evaluated = jnp.mod(position - lower, upper - lower) + lower if periodic else position
    source_in_domain = active & (
        jnp.ones_like(active) if periodic else (position >= lower) & (position <= upper)
    )
    if spacing_value is not None:
        spacing = jnp.asarray(spacing_value, dtype=position.dtype)
        normalized = (evaluated - coordinates[0]) / spacing
        nearest_integer = jnp.round(normalized)
        snap_tolerance = (
            64.0 * jnp.finfo(position.dtype).eps * jnp.maximum(1.0, jnp.abs(normalized))
        )
        normalized = jnp.where(
            jnp.abs(normalized - nearest_integer) <= snap_tolerance,
            nearest_integer,
            normalized,
        )
        base = jnp.floor(normalized - 0.5) if width == 3 else jnp.floor(normalized - 1.0)
        raw = (
            base.astype(jnp.int32)[:, None] + jnp.arange(width, dtype=jnp.int32)[None, :]
        )
        target_coordinate = coordinates[0] + raw.astype(position.dtype) * spacing
        local = normalized[:, None] - raw.astype(position.dtype)
        weights, derivative = _kernel_basis(kernel.name, local)
        derivative = derivative / spacing
        if periodic:
            indices = jnp.mod(raw, count)
            valid = jnp.broadcast_to(source_in_domain[:, None], raw.shape)
        else:
            valid = source_in_domain[:, None] & (raw >= 0) & (raw < count)
            indices = jnp.clip(raw, 0, count - 1)
        offsets = target_coordinate - evaluated[:, None]
        return indices, weights, derivative, offsets, valid, source_in_domain
    if periodic:
        raise ValueError("Nonuniform periodic marker axes are not supported.")
    insertion = jnp.searchsorted(coordinates, evaluated, side="left")
    base = jnp.clip(insertion - width // 2, 0, count - width)
    indices = base[:, None] + jnp.arange(width, dtype=jnp.int32)[None, :]
    targets = coordinates[indices]
    weights = jax.vmap(_nonuniform_affine_weights, in_axes=(None, 0, 0))(
        kernel.name, evaluated, targets
    )
    derivative = jax.vmap(
        jax.jacfwd(_nonuniform_affine_weights, argnums=1),
        in_axes=(None, 0, 0),
    )(kernel.name, evaluated, targets)
    offsets = targets - evaluated[:, None]
    valid = jnp.broadcast_to(source_in_domain[:, None], indices.shape)
    return indices, weights, derivative, offsets, valid, source_in_domain


def _tensor_routes(layout, axes, bounds, position, active, kernel, /):
    axis_stencils = tuple(
        _axis_stencil(
            coordinates,
            axis_bounds,
            axis.periodic,
            position[:, axis_index],
            active,
            kernel,
        )
        for axis_index, (coordinates, axis_bounds, axis) in enumerate(
            zip(layout.coordinates_by_axis, bounds, axes, strict=True)
        )
    )
    dimension = len(axis_stencils)
    source_count = int(position.shape[0])
    width = kernel.width
    route_indices = []
    route_weights = []
    route_gradients = []
    route_offsets = []
    route_validity = []
    source_in_domain = active.copy()
    for stencil in axis_stencils:
        source_in_domain = source_in_domain & stencil[5]
    for slots in product(range(width), repeat=dimension):
        axis_indices = [
            axis_stencils[axis][0][:, slot] for axis, slot in enumerate(slots)
        ]
        flat_index = axis_indices[0]
        for axis in range(1, dimension):
            flat_index = flat_index * layout.shape[axis] + axis_indices[axis]
        weight = jnp.ones((source_count,), dtype=position.dtype)
        valid = active.copy()
        offsets = []
        for axis, slot in enumerate(slots):
            weight = weight * axis_stencils[axis][1][:, slot]
            valid = valid & axis_stencils[axis][4][:, slot]
            offsets.append(axis_stencils[axis][3][:, slot])
        gradients = []
        for derivative_axis in range(dimension):
            derivative = axis_stencils[derivative_axis][2][:, slots[derivative_axis]]
            for axis, slot in enumerate(slots):
                if axis != derivative_axis:
                    derivative = derivative * axis_stencils[axis][1][:, slot]
            gradients.append(derivative)
        route_indices.append(flat_index)
        route_weights.append(weight)
        route_gradients.append(jnp.stack(gradients, axis=-1))
        route_offsets.append(jnp.stack(offsets, axis=-1))
        route_validity.append(valid)
    indices = jnp.stack(route_indices, axis=-1).astype(jnp.int32)
    weights = jnp.stack(route_weights, axis=-1)
    gradients = jnp.stack(route_gradients, axis=1)
    offsets = jnp.stack(route_offsets, axis=1)
    valid = jnp.stack(route_validity, axis=-1)
    masked_weights = jnp.where(valid, weights, 0.0)
    captured = jnp.sum(masked_weights, axis=-1)
    first = jnp.sum(masked_weights[..., None] * offsets, axis=1)
    gradient_sum = jnp.sum(jnp.where(valid[..., None], gradients, 0.0), axis=1)
    tolerance = jnp.finfo(position.dtype).eps * max(16, width**dimension)
    full_support = active & (jnp.abs(captured - 1.0) <= tolerance)
    return (
        indices,
        weights,
        gradients,
        offsets,
        valid,
        source_in_domain,
        captured,
        full_support,
        first,
        gradient_sum,
    )


class MACMarkerRelation(StrictModule):
    face_indices: tuple[Array, ...]
    weights: tuple[Array, ...]
    weight_gradients: tuple[Array, ...]
    route_offsets: tuple[Array, ...]
    valid: tuple[Array, ...]
    marker_position: Array
    partition_residual: tuple[Array, ...]
    first_moment_residual: tuple[Array, ...]
    gradient_sum_residual: tuple[Array, ...]
    support_truncated: Array
    periodic_image_used: Array
    successful: Array
    relation_id: str = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)


class MACMarkerRouteState(StrictModule, NonTrainableState):
    """Nondifferentiable fixed route indices and validity masks."""

    face_indices: tuple[Array, ...]
    valid: tuple[Array, ...]
    transfer_id: str = eqx.field(static=True)


class MACMarkerTransferDiagnostics(StrictModule):
    marker_resultant: Array
    face_resultant: Array
    force_residual: Array
    marker_torque: Array
    face_torque: Array
    torque_residual: Array
    interpolation_work: Array
    spreading_work: Array
    work_adjoint_residual: Array
    maximum_partition_residual: Array
    maximum_first_moment_residual: Array
    maximum_gradient_sum_residual: Array
    support_truncated: Array
    periodic_image_used: Array
    valid_route_count: Array
    minimum_route_weight: Array
    tolerance: Array
    finite: Array
    successful: Array
    transfer_id: str = eqx.field(static=True)


class MACMarkerTransferPlan(StrictModule, NonTrainableState):
    operators: PreparedMACOperators
    accumulation: MACMarkerAccumulation = eqx.field(static=True)
    markers: LagrangianMarkerDiscretization
    kernel: MACMarkerKernelPlan
    route_width: int = eqx.field(static=True)
    relation_bytes: int = eqx.field(static=True)
    scalar_workspace_bytes: int = eqx.field(static=True)
    maximum_resource_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        markers: LagrangianMarkerDiscretization,
        /,
        *,
        kernel: MACMarkerKernelPlan | None = None,
        accumulation: MACMarkerAccumulation = "deterministic",
        maximum_resource_bytes: int = 1024**3,
    ):
        if not isinstance(operators, PreparedMACOperators):
            raise TypeError("operators must be PreparedMACOperators.")
        if not isinstance(markers, LagrangianMarkerDiscretization):
            raise TypeError("markers must be LagrangianMarkerDiscretization.")
        dimension = len(operators.discretization.cell_shape)
        if markers.ambient_dimension != dimension:
            raise ValueError("Marker and MAC dimensions differ.")
        kernel_ = MACMarkerKernelPlan() if kernel is None else kernel
        if accumulation not in ("fast", "deterministic", "compensated"):
            raise ValueError("Unknown MAC marker accumulation policy.")
        if not isinstance(kernel_, MACMarkerKernelPlan):
            raise TypeError("kernel must be MACMarkerKernelPlan or None.")
        axes = operators.discretization.grid.structured_axes
        for layout in operators.discretization.face_layouts:
            for coordinates, axis in zip(layout.coordinates_by_axis, axes, strict=True):
                _uniform_spacing(
                    coordinates,
                    (float(axis.bounds[0]), float(axis.bounds[1])),
                    axis.periodic,
                )
        width = kernel_.width**dimension
        route_count = dimension * markers.active_count * width
        itemsize = np.dtype(operators.pressure_space.dtype).itemsize
        relation_bytes = route_count * (
            np.dtype(np.int32).itemsize + (2 * dimension + 2) * itemsize + 1
        )
        workspace = sum(
            int(np.prod(layout.shape)) * itemsize
            for layout in operators.discretization.face_layouts
        )
        limit = int(maximum_resource_bytes)
        if limit <= 0 or relation_bytes + workspace > limit:
            raise ValueError("MAC marker transfer exceeds its resource budget.")
        self.operators = operators
        self.markers = markers
        self.accumulation = accumulation
        self.kernel = kernel_
        self.route_width = width
        self.relation_bytes = relation_bytes
        self.scalar_workspace_bytes = workspace
        self.maximum_resource_bytes = limit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-marker-transfer-plan",
                "operators": operators.prepared_id,
                "markers": markers.prepared_id,
                "accumulation": accumulation,
                "assignment": kernel_.kernel_id,
                "route_width": width,
                "resource_limit": limit,
            }
        )

    def prepare(self, /) -> PreparedMACMarkerTransfer:
        return PreparedMACMarkerTransfer(self)


class PreparedMACMarkerTransfer(StrictModule, NonTrainableState):
    operators: PreparedMACOperators
    markers: LagrangianMarkerDiscretization
    kernel: MACMarkerKernelPlan
    route_width: int = eqx.field(static=True)
    target_sizes: tuple[int, ...] = eqx.field(static=True)
    axis_bounds: tuple[tuple[float, float], ...] = eqx.field(static=True)
    flattened_face_centers: tuple[Array, ...]
    flattened_dual_measures: tuple[Array, ...]
    preparation_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    accumulation: MACMarkerAccumulation = eqx.field(static=True)

    def __init__(self, plan: MACMarkerTransferPlan, /):
        if not isinstance(plan, MACMarkerTransferPlan):
            raise TypeError("plan must be MACMarkerTransferPlan.")
        dimension = len(plan.operators.discretization.cell_shape)
        centers = tuple(
            value.reshape((-1, dimension))
            for value in plan.operators.discretization.face_centers
        )
        measures = tuple(
            value.reshape((-1,)) for value in plan.operators.face_dual_measures
        )
        bounds = tuple(
            (float(axis.bounds[0]), float(axis.bounds[1]))
            for axis in plan.operators.discretization.grid.structured_axes
        )
        self.operators = plan.operators
        self.kernel = plan.kernel
        self.markers = plan.markers
        self.route_width = plan.route_width
        self.target_sizes = tuple(int(value.shape[0]) for value in centers)
        self.axis_bounds = bounds
        self.flattened_face_centers = centers
        self.flattened_dual_measures = measures
        self.accumulation = plan.accumulation
        self.preparation_id = canonical_fingerprint(
            {
                "kind": "mac-marker-transfer-resources",
                "relation_bytes": plan.relation_bytes,
                "scalar_workspace_bytes": plan.scalar_workspace_bytes,
                "route_width": plan.route_width,
            }
        )
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-mac-marker-transfer", "plan": plan.plan_id}
        )

    @property
    def dimension(self) -> int:
        return self.markers.ambient_dimension

    def relation(self, marker_positions: ArrayLike, /) -> MACMarkerRelation:
        raw = jnp.asarray(marker_positions, dtype=self.operators.pressure_space.dtype)
        expected = (self.markers.capacity, self.dimension)
        if raw.shape != expected:
            raise ValueError(f"marker_positions must have shape {expected}.")
        active = self.markers.active_mask
        finite = jnp.all(jnp.isfinite(raw), axis=-1)
        fallback = jnp.asarray(
            tuple(bound[0] for bound in self.axis_bounds), dtype=raw.dtype
        )
        safe = jnp.where((active & finite)[:, None], raw, fallback)
        axes = self.operators.discretization.grid.structured_axes
        active_indices = self.markers.active_indices
        active_position = safe[active_indices]
        all_indices = []
        all_weights = []
        all_gradients = []
        all_offsets = []
        all_valid = []
        partitions = []
        first_moments = []
        gradient_sums = []
        truncated = jnp.zeros((self.markers.active_count,), dtype=bool)
        periodic_used = jnp.zeros((self.markers.active_count,), dtype=bool)
        for layout in self.operators.discretization.face_layouts:
            route = _tensor_routes(
                layout,
                axes,
                self.axis_bounds,
                safe,
                active & finite,
                self.kernel,
            )
            indices, weights, gradients, offsets, valid = (
                route[0][active_indices],
                route[1][active_indices],
                route[2][active_indices],
                route[3][active_indices],
                route[4][active_indices],
            )
            full_support = route[7][active_indices]
            captured = route[6][active_indices]
            first = route[8][active_indices]
            gradient = route[9][active_indices]
            all_indices.append(jax.lax.stop_gradient(indices))
            all_weights.append(weights)
            all_gradients.append(gradients)
            all_offsets.append(offsets)
            all_valid.append(jax.lax.stop_gradient(valid))
            partitions.append(jnp.abs(captured - 1.0))
            first_moments.append(jnp.abs(first))
            gradient_sums.append(jnp.abs(gradient))
            truncated = truncated | ~full_support
        spacing = jnp.asarray([axis.interval_widths[0] for axis in axes], dtype=raw.dtype)
        for axis_index, axis in enumerate(axes):
            if axis.periodic:
                lower, upper = self.axis_bounds[axis_index]
                coordinate = active_position[:, axis_index]
                support_radius = 0.5 * self.kernel.width * spacing[axis_index]
                periodic_used = periodic_used | (
                    (coordinate - support_radius < lower)
                    | (coordinate + support_radius > upper)
                )
        tolerance = (
            256.0
            * jnp.finfo(raw.dtype).eps
            * jnp.maximum(1.0, jnp.max(jnp.abs(active_position)))
        )
        successful = (
            jnp.all(finite[active_indices])
            & ~jnp.any(truncated)
            & (jnp.max(jnp.stack(tuple(partitions))) <= tolerance)
            & (jnp.max(jnp.stack(tuple(first_moments))) <= tolerance)
            & (jnp.max(jnp.stack(tuple(gradient_sums))) <= tolerance)
        )
        relation_id = canonical_fingerprint(
            {
                "kind": "mac-marker-relation",
                "transfer": self.prepared_id,
                "route_width": self.route_width,
            }
        )
        return MACMarkerRelation(
            tuple(all_indices),
            tuple(all_weights),
            tuple(all_gradients),
            tuple(all_offsets),
            tuple(all_valid),
            active_position,
            tuple(partitions),
            tuple(first_moments),
            tuple(gradient_sums),
            truncated,
            periodic_used,
            successful,
            relation_id,
            self.prepared_id,
        )

    def relation_on_routes(
        self,
        marker_positions: ArrayLike,
        routes: MACMarkerRouteState,
        /,
    ) -> MACMarkerRelation:
        """Recompute smooth weights while certifying the frozen discrete routes."""
        if not isinstance(routes, MACMarkerRouteState):
            raise TypeError("routes must be MACMarkerRouteState.")
        if routes.transfer_id != self.prepared_id:
            raise ValueError("Marker route state belongs to another transfer.")
        current = self.relation(marker_positions)
        matches = self.routes_match(current, routes)
        return MACMarkerRelation(
            routes.face_indices,
            current.weights,
            current.weight_gradients,
            current.route_offsets,
            routes.valid,
            current.marker_position,
            current.partition_residual,
            current.first_moment_residual,
            current.gradient_sum_residual,
            current.support_truncated,
            current.periodic_image_used,
            current.successful & matches,
            canonical_fingerprint(
                {
                    "kind": "mac-marker-fixed-route-relation",
                    "transfer": self.prepared_id,
                    "route_width": self.route_width,
                }
            ),
            self.prepared_id,
        )

    def route_state(self, relation: MACMarkerRelation, /) -> MACMarkerRouteState:
        self._validate_relation(relation)
        return MACMarkerRouteState(
            tuple(jax.lax.stop_gradient(value) for value in relation.face_indices),
            tuple(jax.lax.stop_gradient(value) for value in relation.valid),
            self.prepared_id,
        )

    def routes_match(
        self, relation: MACMarkerRelation, routes: MACMarkerRouteState, /
    ) -> Array:
        self._validate_relation(relation)
        if not isinstance(routes, MACMarkerRouteState):
            raise TypeError("routes must be MACMarkerRouteState.")
        if routes.transfer_id != self.prepared_id:
            raise ValueError("Marker route state belongs to another transfer.")
        return jnp.all(
            jnp.stack(
                tuple(
                    jnp.all(left_indices == right_indices)
                    & jnp.all(left_valid == right_valid)
                    for left_indices, right_indices, left_valid, right_valid in zip(
                        relation.face_indices,
                        routes.face_indices,
                        relation.valid,
                        routes.valid,
                        strict=True,
                    )
                )
            )
        )

    def gather(
        self, relation: MACMarkerRelation, face_velocity: FaceVelocity, /
    ) -> Array:
        self._validate_relation(relation)
        velocity = self.operators.validate_velocity(face_velocity)
        components = []
        for axis, value in enumerate(velocity):
            sampled = value.reshape((-1,))[relation.face_indices[axis]]
            components.append(
                jnp.sum(
                    jnp.where(
                        relation.valid[axis],
                        relation.weights[axis] * sampled,
                        0.0,
                    ),
                    axis=1,
                )
            )
        return jnp.stack(tuple(components), axis=-1)

    def _raw_transpose(self, relation: MACMarkerRelation, values: ArrayLike, /):
        active_values = self.markers.active_velocity_space.validate(jnp.asarray(values))
        output = []
        order = self.markers.stable_active_order
        for axis, layout in enumerate(self.operators.discretization.face_layouts):
            indices = relation.face_indices[axis]
            valid = relation.valid[axis]
            payload = jnp.where(
                valid,
                relation.weights[axis] * active_values[:, axis, None],
                0.0,
            )
            flat = jnp.zeros((self.target_sizes[axis],), dtype=active_values.dtype)
            if self.accumulation == "fast":
                flat = flat.at[indices.reshape((-1,))].add(payload.reshape((-1,)))
            else:
                width = int(indices.shape[1])

                def add_route(route_index, carry):
                    source_rank = route_index // width
                    slot = route_index - source_rank * width
                    source = order[source_rank]
                    target = indices[source, slot]
                    value = jnp.where(valid[source, slot], payload[source, slot], 0.0)
                    if self.accumulation == "deterministic":
                        return carry.at[target].add(value)
                    total, correction = carry
                    updated, error = two_sum(total[target], value)
                    return (
                        total.at[target].set(updated),
                        correction.at[target].add(error),
                    )

                route_count = int(indices.shape[0]) * width
                if self.accumulation == "deterministic":
                    flat = jax.lax.fori_loop(0, route_count, add_route, flat)
                else:
                    total, correction = jax.lax.fori_loop(
                        0,
                        route_count,
                        add_route,
                        (flat, jnp.zeros_like(flat)),
                    )
                    flat = total + correction
            output.append(flat.reshape(layout.shape))
        return tuple(output)

    def interpolation_operator(self, relation: MACMarkerRelation, /):
        self._validate_relation(relation)
        return FunctionLinearOperator(
            lambda velocity: self.gather(relation, velocity),
            source=self.operators.velocity_space,
            target=self.markers.active_velocity_space,
            transpose_action=lambda values: self._raw_transpose(relation, values),
            properties=OperatorProperties(),
            operator_id=f"mac-marker-interpolation/{relation.relation_id}",
        )

    def spread(self, relation: MACMarkerRelation, marker_force_density: ArrayLike, /):
        values = self.markers.active_velocity_space.validate(
            jnp.asarray(marker_force_density)
        )
        return tuple(self.interpolation_operator(relation).adjoint_mv(values))

    def diagnostics(
        self,
        relation: MACMarkerRelation,
        face_velocity: FaceVelocity,
        marker_force_density: ArrayLike,
        /,
        *,
        torque_origin: ArrayLike | None = None,
    ) -> MACMarkerTransferDiagnostics:
        self._validate_relation(relation)
        velocity = self.operators.validate_velocity(face_velocity)
        values = self.markers.active_velocity_space.validate(
            jnp.asarray(marker_force_density)
        )
        sampled = self.gather(relation, velocity)
        spread = self.spread(relation, values)
        interpolation_work = jnp.real(
            self.markers.active_velocity_space.inner(sampled, values)
        )
        spreading_work = jnp.real(self.operators.velocity_space.inner(velocity, spread))
        work_residual = spreading_work - interpolation_work
        active_weights = self.markers.material_measure.weights[
            self.markers.active_indices
        ].astype(values.dtype)
        integrated = active_weights[:, None] * values
        marker_resultant = jnp.sum(integrated, axis=0)
        face_resultant = jnp.stack(
            tuple(
                jnp.sum(measure * component)
                for measure, component in zip(
                    self.operators.face_dual_measures, spread, strict=True
                )
            )
        )
        force_residual = face_resultant - marker_resultant
        origin = (
            jnp.zeros((self.dimension,), dtype=values.dtype)
            if torque_origin is None
            else jnp.asarray(torque_origin, dtype=values.dtype)
        )
        if origin.shape != (self.dimension,):
            raise ValueError("torque_origin has an incompatible shape.")
        arm = relation.marker_position - origin
        if self.dimension == 2:
            marker_torque = jnp.sum(
                arm[:, 0] * integrated[:, 1] - arm[:, 1] * integrated[:, 0]
            ).reshape((1,))
        else:
            marker_torque = jnp.sum(jnp.cross(arm, integrated), axis=0)
        face_torque = jnp.zeros_like(marker_torque)
        for axis in range(self.dimension):
            route_force = (
                active_weights[:, None] * relation.weights[axis] * values[:, axis, None]
            )
            route_position = (
                relation.marker_position[:, None, :] + relation.route_offsets[axis]
            )
            route_arm = route_position - origin
            if self.dimension == 2:
                torque = (
                    -route_arm[..., 1] * route_force
                    if axis == 0
                    else route_arm[..., 0] * route_force
                )
                face_torque = face_torque + jnp.sum(
                    jnp.where(relation.valid[axis], torque, 0.0)
                ).reshape((1,))
            else:
                direction = jax.nn.one_hot(axis, self.dimension, dtype=values.dtype)
                force = route_force[..., None] * direction
                face_torque = face_torque + jnp.sum(
                    jnp.where(
                        relation.valid[axis][..., None],
                        jnp.cross(route_arm, force),
                        0.0,
                    ),
                    axis=(0, 1),
                )
        torque_residual = face_torque - marker_torque
        valid_route_count = jnp.sum(
            jnp.stack(tuple(jnp.sum(valid) for valid in relation.valid)),
            dtype=jnp.int32,
        )
        minimum_weight = jnp.min(
            jnp.stack(
                tuple(
                    jnp.min(jnp.where(valid, weight, jnp.inf))
                    for valid, weight in zip(
                        relation.valid, relation.weights, strict=True
                    )
                )
            )
        )
        finite = (
            jnp.all(jnp.isfinite(sampled))
            & jnp.all(jnp.isfinite(values))
            & jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(item)) for item in spread)))
            & jnp.all(jnp.isfinite(force_residual))
            & jnp.all(jnp.isfinite(torque_residual))
            & jnp.isfinite(work_residual)
        )
        scale = jnp.maximum(
            1.0,
            jnp.max(
                jnp.stack(
                    (
                        jnp.abs(interpolation_work),
                        jnp.max(jnp.abs(marker_resultant)),
                        jnp.max(jnp.abs(face_resultant)),
                        jnp.max(jnp.abs(marker_torque)),
                        jnp.max(jnp.abs(face_torque)),
                    )
                )
            ),
        )
        tolerance = 4096.0 * jnp.finfo(values.dtype).eps * scale
        successful = (
            relation.successful
            & finite
            & (jnp.abs(work_residual) <= tolerance)
            & (jnp.max(jnp.abs(force_residual)) <= tolerance)
            & (jnp.max(jnp.abs(torque_residual)) <= tolerance)
        )
        return MACMarkerTransferDiagnostics(
            marker_resultant,
            face_resultant,
            force_residual,
            marker_torque,
            face_torque,
            torque_residual,
            interpolation_work,
            spreading_work,
            work_residual,
            jnp.max(jnp.stack(relation.partition_residual)),
            jnp.max(jnp.stack(relation.first_moment_residual)),
            jnp.max(jnp.stack(relation.gradient_sum_residual)),
            relation.support_truncated,
            relation.periodic_image_used,
            valid_route_count,
            minimum_weight,
            tolerance,
            finite,
            successful,
            self.prepared_id,
        )

    def _validate_relation(self, relation: MACMarkerRelation, /) -> None:
        if not isinstance(relation, MACMarkerRelation):
            raise TypeError("relation must be MACMarkerRelation.")
        if relation.transfer_id != self.prepared_id:
            raise ValueError("MAC marker relation belongs to another transfer.")


__all__ = [
    "MACMarkerAccumulation",
    "MACMarkerKernelName",
    "MACMarkerKernelPlan",
    "MACMarkerRelation",
    "MACMarkerRouteState",
    "MACMarkerTransferDiagnostics",
    "MACMarkerTransferPlan",
    "PreparedMACMarkerTransfer",
]
