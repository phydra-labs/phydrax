#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from itertools import product
from typing import Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._sharp_measures import (
    QualifiedSharpGeometry,
    SharpGeometryEvidence,
    SharpGeometryStatus,
    SharpMeasureFidelity,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._incompressible import PreparedMACOperators


class ExactSDFEnclosureCertificate(Protocol):
    """Structural exact-SDF enclosure data owned by the geometry package."""

    field: Any
    evaluation_error: float
    lipschitz_upper_bound: float

    @property
    def certifies_global_enclosure(self) -> bool: ...


SignedDistanceProvider = Callable[[Array, Array, Any], ArrayLike]
WallVelocityProvider = Callable[[Array, Array, Any], ArrayLike]
SweptMeasureRateProvider = Callable[[Array, Any], ArrayLike]


def _box_samples(
    centers: np.ndarray,
    widths: np.ndarray,
    subdivisions: tuple[int, ...],
    full_measure: np.ndarray,
) -> tuple[Array, Array, Array]:
    coordinates = tuple(
        (np.arange(count, dtype=float) + 0.5) / count - 0.5 for count in subdivisions
    )
    normalized = np.asarray(tuple(product(*coordinates)), dtype=float)
    points = centers[..., None, :] + widths[..., None, :] * normalized
    half_width = widths / np.asarray(subdivisions, dtype=float)
    radius = 0.5 * np.sqrt(np.sum(half_width**2, axis=-1))
    capacity = int(np.prod(subdivisions))
    weight = full_measure / capacity
    return (
        jnp.asarray(points),
        jnp.broadcast_to(jnp.asarray(radius)[..., None], points.shape[:-1]),
        jnp.asarray(weight),
    )


def _width_field(operators: PreparedMACOperators) -> np.ndarray:
    discretization = operators.discretization
    values = []
    for axis, structured_axis in enumerate(discretization.grid.structured_axes):
        shape = [1] * len(discretization.cell_shape)
        shape[axis] = structured_axis.interval_widths.size
        values.append(
            np.broadcast_to(
                np.asarray(structured_axis.interval_widths).reshape(shape),
                discretization.cell_shape,
            )
        )
    return np.stack(tuple(values), axis=-1)


class MACSharpGeometryRefreshResult(StrictModule):
    """Candidate and atomically selected accepted sharp geometry."""

    candidate: QualifiedSharpGeometry
    geometry: QualifiedSharpGeometry
    accepted: Array
    refresh_required: Array
    plan_id: str = eqx.field(static=True)


class MACExactSDFMeasurePlan(StrictModule, NonTrainableState):
    """Fixed-capacity Lipschitz enclosure of structured MAC fluid measures.

    The source must be a globally reliable exact signed distance. Each fixed
    sub-box is classified by ``phi(center) ± L radius ± evaluation_error``.
    Unclassified boxes contribute to explicit lower/upper bounds; they are
    never converted into raw sampled fractions.
    """

    operators: PreparedMACOperators
    signed_distance: SignedDistanceProvider = eqx.field(static=True)
    wall_velocity_provider: WallVelocityProvider | None = eqx.field(static=True)
    swept_measure_rate_provider: SweptMeasureRateProvider | None = eqx.field(static=True)
    certificate: ExactSDFEnclosureCertificate = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    subdivisions: tuple[int, ...] = eqx.field(static=True)
    maximum_measure_error_fraction: float = eqx.field(static=True)
    small_cell_fraction: float = eqx.field(static=True)
    gcl_tolerance: float = eqx.field(static=True)
    cell_points: Array
    cell_radii: Array
    cell_sample_measure: Array
    face_points: tuple[Array, ...]
    face_radii: tuple[Array, ...]
    face_sample_measure: tuple[Array, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        signed_distance: SignedDistanceProvider,
        certificate: ExactSDFEnclosureCertificate,
        /,
        *,
        source_id: str,
        subdivisions: int | tuple[int, ...] = 4,
        maximum_measure_error_fraction: float = 1.0,
        small_cell_fraction: float = 1.0e-2,
        wall_velocity: WallVelocityProvider | None = None,
        swept_cell_measure_rate: SweptMeasureRateProvider | None = None,
        gcl_tolerance: float = 1.0e-9,
    ):
        if not isinstance(operators, PreparedMACOperators):
            raise TypeError("operators must be PreparedMACOperators.")
        if not callable(signed_distance):
            raise TypeError("signed_distance must be callable.")
        if not certificate.certifies_global_enclosure:
            raise ValueError("certificate must certify a global exact-SDF enclosure.")
        if wall_velocity is not None and not callable(wall_velocity):
            raise TypeError("wall_velocity must be callable or None.")
        if swept_cell_measure_rate is not None and not callable(swept_cell_measure_rate):
            raise TypeError("swept_cell_measure_rate must be callable or None.")
        identifier = str(source_id)
        dimension = len(operators.discretization.cell_shape)
        counts = (
            (int(subdivisions),) * dimension
            if isinstance(subdivisions, int)
            else tuple(int(value) for value in subdivisions)
        )
        maximum_error = float(maximum_measure_error_fraction)
        small = float(small_cell_fraction)
        tolerance = float(gcl_tolerance)
        if (
            not identifier
            or len(counts) != dimension
            or any(value <= 0 for value in counts)
            or not 0.0 <= maximum_error <= 1.0
            or not 0.0 < small < 1.0
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError("Exact-SDF structured measure policy is invalid.")

        discretization = operators.discretization
        widths = _width_field(operators)
        cell_points, cell_radii, cell_weight = _box_samples(
            np.asarray(discretization.cell_centers),
            widths,
            counts,
            np.asarray(discretization.cell_volumes),
        )
        face_points = []
        face_radii = []
        face_weights = []
        for normal_axis, centers in enumerate(discretization.face_centers):
            tangential = tuple(axis for axis in range(dimension) if axis != normal_axis)
            face_widths = np.zeros(np.asarray(centers).shape, dtype=float)
            for axis in tangential:
                shape = [1] * dimension
                shape[axis] = discretization.grid.structured_axes[
                    axis
                ].interval_widths.size
                face_widths[..., axis] = np.broadcast_to(
                    np.asarray(
                        discretization.grid.structured_axes[axis].interval_widths
                    ).reshape(shape),
                    discretization.face_layouts[normal_axis].shape,
                )
            tangential_counts = tuple(counts[axis] for axis in tangential)
            if tangential_counts:
                points, radii, weights_ = _box_samples(
                    np.asarray(centers),
                    face_widths,
                    tuple(
                        counts[axis] if axis in tangential else 1
                        for axis in range(dimension)
                    ),
                    np.asarray(discretization.face_measures[normal_axis]),
                )
            else:
                points = jnp.asarray(centers)[..., None, :]
                radii = jnp.zeros(points.shape[:-1], dtype=points.dtype)
                weights_ = jnp.asarray(discretization.face_measures[normal_axis])
            face_points.append(points)
            face_radii.append(radii)
            face_weights.append(weights_)

        self.operators = operators
        self.signed_distance = signed_distance
        self.wall_velocity_provider = wall_velocity
        self.swept_measure_rate_provider = swept_cell_measure_rate
        self.certificate = certificate
        self.source_id = identifier
        self.subdivisions = counts
        self.maximum_measure_error_fraction = maximum_error
        self.small_cell_fraction = small
        self.gcl_tolerance = tolerance
        self.cell_points = cell_points
        self.cell_radii = cell_radii
        self.cell_sample_measure = cell_weight
        self.face_points = tuple(face_points)
        self.face_radii = tuple(face_radii)
        self.face_sample_measure = tuple(face_weights)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-exact-sdf-measure-enclosure",
                "operators": operators.prepared_id,
                "source": identifier,
                "source_certificate": {
                    "zero_set": certificate.field.zero_set_accuracy.value,
                    "sign": certificate.field.sign_reliability.value,
                    "distance": certificate.field.distance_semantics.value,
                    "validity": certificate.field.validity_region,
                    "evaluation_error": certificate.evaluation_error,
                    "lipschitz": certificate.lipschitz_upper_bound,
                },
                "subdivisions": list(counts),
                "maximum_measure_error_fraction": maximum_error,
                "small_cell_fraction": small,
                "gcl_tolerance": tolerance,
            }
        )

    def _enclose(
        self,
        points: Array,
        radii: Array,
        sample_measure: Array,
        time: Array,
        args: Any,
        /,
    ) -> tuple[Array, Array, Array, Array]:
        phi = jnp.asarray(self.signed_distance(points, time, args))
        if phi.shape != points.shape[:-1]:
            raise ValueError("Exact SDF must return one value per enclosure point.")
        uncertainty = (
            self.certificate.lipschitz_upper_bound * radii
            + self.certificate.evaluation_error
        )
        fluid = phi - uncertainty >= 0.0
        solid = (~fluid) & (phi + uncertainty <= 0.0)
        weight = sample_measure[..., None]
        lower = jnp.sum(weight * fluid, axis=-1)
        upper = jnp.sum(weight * (~solid), axis=-1)
        nominal = 0.5 * (lower + upper)
        return nominal, lower, upper, jnp.all(jnp.isfinite(phi))

    def evaluate(
        self,
        time: ArrayLike = 0.0,
        /,
        *,
        args: Any = None,
        previous: QualifiedSharpGeometry | None = None,
        step_size: ArrayLike | None = None,
    ) -> QualifiedSharpGeometry:
        time_ = jnp.asarray(time, dtype=self.operators.pressure_space.dtype).reshape(())
        cell, cell_lower, cell_upper, cell_finite = self._enclose(
            self.cell_points,
            self.cell_radii,
            self.cell_sample_measure,
            time_,
            args,
        )
        face_values = []
        face_lower = []
        face_upper = []
        finite_terms = [cell_finite]
        for points, radii, measure in zip(
            self.face_points,
            self.face_radii,
            self.face_sample_measure,
            strict=True,
        ):
            value, lower, upper, finite = self._enclose(
                points, radii, measure, time_, args
            )
            face_values.append(value)
            face_lower.append(lower)
            face_upper.append(upper)
            finite_terms.append(finite)

        discretization = self.operators.discretization
        full_cell = discretization.cell_volumes.astype(cell.dtype)
        full_faces = tuple(
            value.astype(cell.dtype) for value in discretization.face_measures
        )
        cell_width = cell_upper - cell_lower
        face_width = tuple(
            upper - lower for lower, upper in zip(face_lower, face_upper, strict=True)
        )
        bounds_valid = (
            jnp.all((cell_lower >= 0.0) & (cell_lower <= cell) & (cell <= cell_upper))
            & jnp.all(cell_upper <= full_cell)
            & jnp.all(
                jnp.stack(
                    tuple(
                        jnp.all(
                            (lower >= 0.0)
                            & (lower <= value)
                            & (value <= upper)
                            & (upper <= full)
                        )
                        for value, lower, upper, full in zip(
                            face_values,
                            face_lower,
                            face_upper,
                            full_faces,
                            strict=True,
                        )
                    )
                )
            )
        )
        cell_active = cell_lower > 0.0
        face_active = tuple(lower > 0.0 for lower in face_lower)
        topology_resolved = jnp.all((cell_lower > 0.0) | (cell_upper == 0.0)) & jnp.all(
            jnp.stack(
                tuple(
                    jnp.all((lower > 0.0) | (upper == 0.0))
                    for lower, upper in zip(face_lower, face_upper, strict=True)
                )
            )
        )
        error_satisfied = jnp.all(
            cell_width <= self.maximum_measure_error_fraction * full_cell
        ) & jnp.all(
            jnp.stack(
                tuple(
                    jnp.all(width <= self.maximum_measure_error_fraction * full)
                    for width, full in zip(face_width, full_faces, strict=True)
                )
            )
        )
        finite = jnp.all(jnp.stack(tuple(finite_terms))) & jnp.all(jnp.isfinite(cell))

        if previous is not None and (
            not isinstance(previous, QualifiedSharpGeometry)
            or previous.realization_id != self.plan_id
        ):
            raise ValueError("previous geometry belongs to another sharp measure plan.")
        if previous is None:
            swept = jnp.zeros_like(cell)
            gcl_lower = jnp.zeros_like(cell)
            gcl_upper = jnp.zeros_like(cell)
            gcl_satisfied = jnp.asarray(True)
            epoch = jnp.asarray(0, dtype=jnp.int32)
        else:
            if step_size is None:
                raise ValueError("Refreshing sharp geometry requires step_size.")
            step = jnp.asarray(step_size, dtype=cell.dtype).reshape(())
            step = eqx.error_if(
                step,
                ~jnp.isfinite(step) | (step <= 0.0),
                "Sharp geometry refresh requires a positive finite step size.",
            )
            swept = (
                jnp.zeros_like(cell)
                if self.swept_measure_rate_provider is None
                else jnp.asarray(
                    self.swept_measure_rate_provider(time_, args), dtype=cell.dtype
                )
            )
            if swept.shape != cell.shape:
                raise ValueError("Swept measure provider must return one rate per cell.")
            gcl_lower = (cell_lower - previous.cell_fluid_measure_upper) / step - swept
            gcl_upper = (cell_upper - previous.cell_fluid_measure_lower) / step - swept
            scale = jnp.maximum(
                1.0,
                jnp.maximum(
                    jnp.abs(swept), jnp.maximum(jnp.abs(gcl_lower), jnp.abs(gcl_upper))
                ),
            )
            tolerance = self.gcl_tolerance * scale
            gcl_satisfied = jnp.all((gcl_lower <= tolerance) & (gcl_upper >= -tolerance))
            epoch = previous.epoch + jnp.asarray(1, dtype=jnp.int32)

        walls = []
        for axis, centers in enumerate(discretization.face_centers):
            if self.wall_velocity_provider is None:
                component = jnp.zeros_like(face_values[axis])
            else:
                velocity = jnp.asarray(
                    self.wall_velocity_provider(centers, time_, args), dtype=cell.dtype
                )
                if velocity.shape != centers.shape:
                    raise ValueError(
                        "Wall velocity provider must return one vector per MAC face."
                    )
                component = velocity[..., axis]
            walls.append(component)
            finite = finite & jnp.all(jnp.isfinite(component))
        finite = finite & jnp.all(jnp.isfinite(swept))
        source_qualified = jnp.asarray(self.certificate.certifies_global_enclosure)
        accepted = (
            source_qualified
            & bounds_valid
            & topology_resolved
            & error_satisfied
            & gcl_satisfied
            & finite
        )
        refresh_required = ~accepted
        status = jnp.asarray(
            int(SharpGeometryStatus.INTERFACE_MOMENTS_UNQUALIFIED), dtype=jnp.int32
        )
        status = status | jnp.where(
            source_qualified, 0, int(SharpGeometryStatus.UNQUALIFIED_SOURCE)
        ).astype(jnp.int32)
        status = status | jnp.where(
            bounds_valid & error_satisfied,
            0,
            int(SharpGeometryStatus.INVALID_BOUNDS),
        ).astype(jnp.int32)
        status = status | jnp.where(
            topology_resolved, 0, int(SharpGeometryStatus.UNRESOLVED_TOPOLOGY)
        ).astype(jnp.int32)
        status = status | jnp.where(
            gcl_satisfied, 0, int(SharpGeometryStatus.GCL_FAILED)
        ).astype(jnp.int32)
        status = status | jnp.where(finite, 0, int(SharpGeometryStatus.NONFINITE)).astype(
            jnp.int32
        )
        status = status | jnp.where(
            refresh_required, int(SharpGeometryStatus.REFRESH_REQUIRED), 0
        ).astype(jnp.int32)
        evidence = SharpGeometryEvidence(
            cell_width,
            tuple(face_width),
            gcl_lower,
            gcl_upper,
            source_qualified,
            bounds_valid & error_satisfied,
            topology_resolved,
            gcl_satisfied,
            jnp.asarray(False),
            finite,
            accepted,
            refresh_required,
            status,
            canonical_fingerprint(
                {"kind": "mac-exact-sdf-measure-evidence", "plan": self.plan_id}
            ),
        )
        dimension = len(discretization.cell_shape)
        zeros_interface = jnp.zeros_like(cell)
        realization = QualifiedSharpGeometry(
            cell,
            cell_lower,
            cell_upper,
            full_cell,
            tuple(face_values),
            tuple(face_lower),
            tuple(face_upper),
            full_faces,
            zeros_interface,
            zeros_interface,
            zeros_interface,
            jnp.zeros(cell.shape + (dimension,), dtype=cell.dtype),
            jnp.zeros(cell.shape + (dimension,), dtype=cell.dtype),
            jnp.full(cell.shape, -1, dtype=jnp.int32),
            tuple(walls),
            swept,
            cell_active,
            tuple(face_active),
            cell_active & (cell < self.small_cell_fraction * full_cell),
            epoch,
            evidence,
            SharpMeasureFidelity.CERTIFIED_BOUNDED_ERROR,
            "globally_reliable_exact_signed_distance",
            self.source_id,
            discretization.support.support_id,
            discretization.cell_space.field_space_id,
            tuple(space.field_space_id for space in discretization.face_spaces),
            self.operators.prepared_id,
            canonical_fingerprint(
                {
                    "pressure": self.operators.pressure_space.space_id,
                    "velocity": self.operators.velocity_space.space_id,
                }
            ),
            self.plan_id,
        )
        return realization

    def prepare(
        self, time: ArrayLike = 0.0, args: Any = None, /
    ) -> QualifiedSharpGeometry:
        return self.evaluate(time, args=args)

    def refresh(
        self,
        previous: QualifiedSharpGeometry,
        time: ArrayLike,
        step_size: ArrayLike,
        args: Any = None,
        /,
    ) -> MACSharpGeometryRefreshResult:
        candidate = self.evaluate(time, args=args, previous=previous, step_size=step_size)
        accepted = candidate.accepted
        geometry = jax.tree.map(
            lambda proposed, current: jnp.where(accepted, proposed, current),
            candidate,
            previous,
        )
        return MACSharpGeometryRefreshResult(
            candidate,
            geometry,
            accepted,
            candidate.evidence.refresh_required,
            self.plan_id,
        )


__all__ = [
    "MACExactSDFMeasurePlan",
    "MACSharpGeometryRefreshResult",
    "SignedDistanceProvider",
    "SweptMeasureRateProvider",
    "WallVelocityProvider",
]
