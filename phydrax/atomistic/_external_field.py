#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared rectilinear external fields with conservative interpolation."""

from __future__ import annotations

import enum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class ExternalFieldBoundaryPolicy(enum.Enum):
    """Behavior for queries outside the node-aligned grid domain."""

    PERIODIC = "periodic"
    CLAMP = "clamp"
    FAIL = "fail"


class ExternalFieldEvidence(StrictModule):
    """Per-point domain evidence and aggregate evaluation status."""

    out_of_domain: Array
    out_of_domain_count: Array
    finite: Array
    successful: Array


class ExternalFieldEvaluation(StrictModule):
    """Interpolated values and their coordinate Jacobians."""

    values: Array
    jacobian: Array
    evidence: ExternalFieldEvidence
    prepared_id: str = eqx.field(static=True)


class ScalarExternalFieldResult(StrictModule):
    """Total scalar coupling energy, conservative force, and domain evidence."""

    energy: Array
    forces: Array
    per_particle_energy: Array
    evidence: ExternalFieldEvidence
    prepared_id: str = eqx.field(static=True)


class GriddedExternalFieldPlan(StrictModule, NonTrainableState):
    """A scalar or vector field sampled on a regular three-dimensional grid."""

    origin: Array
    spacing: Array
    values: Array
    boundary_policy: ExternalFieldBoundaryPolicy = eqx.field(static=True)
    coordinate_frame: str = eqx.field(static=True)
    coordinate_unit: str = eqx.field(static=True)
    value_unit: str = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        origin: ArrayLike,
        spacing: ArrayLike,
        values: ArrayLike,
        /,
        *,
        boundary_policy: ExternalFieldBoundaryPolicy = ExternalFieldBoundaryPolicy.FAIL,
        coordinate_frame: str,
        coordinate_unit: str,
        value_unit: str,
        plan_id: str | None = None,
    ):
        origin_ = np.asarray(origin)
        spacing_ = np.asarray(spacing)
        values_ = np.asarray(values)
        if origin_.shape != (3,) or spacing_.shape != (3,):
            raise ValueError("External-field origin and spacing must have shape (3,).")
        if origin_.dtype.kind != "f" or spacing_.dtype.kind != "f":
            raise TypeError(
                "External-field origin and spacing must be real floating arrays."
            )
        if values_.dtype.kind != "f" or values_.ndim not in (3, 4):
            raise TypeError(
                "External-field values must be a scalar or vector floating grid."
            )
        if any(size < 2 for size in values_.shape[:3]):
            raise ValueError(
                "Every external-field grid axis requires at least two nodes."
            )
        if values_.ndim == 4 and values_.shape[3] < 1:
            raise ValueError("Vector external fields require at least one component.")
        if np.any(~np.isfinite(origin_)) or np.any(~np.isfinite(spacing_)):
            raise ValueError("External-field coordinates must be finite.")
        if np.any(spacing_ <= 0.0) or np.any(~np.isfinite(values_)):
            raise ValueError("External-field spacing must be positive and values finite.")
        if not isinstance(boundary_policy, ExternalFieldBoundaryPolicy):
            raise TypeError("boundary_policy must be an ExternalFieldBoundaryPolicy.")
        identities = tuple(
            str(value).strip()
            for value in (coordinate_frame, coordinate_unit, value_unit)
        )
        if any(not value for value in identities):
            raise ValueError(
                "External-field coordinate and unit identities must be non-empty."
            )
        dtype = np.result_type(origin_.dtype, spacing_.dtype, values_.dtype)
        origin_ = origin_.astype(dtype, copy=False)
        spacing_ = spacing_.astype(dtype, copy=False)
        values_ = values_.astype(dtype, copy=False)
        generated = canonical_fingerprint(
            {
                "kind": "gridded-external-field",
                "boundary_policy": boundary_policy.value,
                "coordinate_frame": identities[0],
                "coordinate_unit": identities[1],
                "value_unit": identities[2],
                "arrays": array_tree_fingerprint(
                    {"origin": origin_, "spacing": spacing_, "values": values_}
                ),
            }
        )
        identifier = generated if plan_id is None else str(plan_id).strip()
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.origin = jnp.asarray(origin_)
        self.spacing = jnp.asarray(spacing_)
        self.values = jnp.asarray(values_)
        self.boundary_policy = boundary_policy
        self.coordinate_frame, self.coordinate_unit, self.value_unit = identities
        self.component_count = 1 if values_.ndim == 3 else int(values_.shape[3])
        self.plan_id = identifier

    @property
    def scalar(self) -> bool:
        return self.values.ndim == 3

    def prepare(self, /) -> "PreparedGriddedExternalField":
        return PreparedGriddedExternalField(self)


class PreparedGriddedExternalField(StrictModule, NonTrainableState):
    """Grid metadata arranged for fixed-shape multilinear evaluation."""

    plan: GriddedExternalFieldPlan
    grid_shape: tuple[int, int, int] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: GriddedExternalFieldPlan, /):
        if not isinstance(plan, GriddedExternalFieldPlan):
            raise TypeError("plan must be a GriddedExternalFieldPlan.")
        self.plan = plan
        self.grid_shape = tuple(int(size) for size in plan.values.shape[:3])
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-gridded-external-field", "plan": plan.plan_id}
        )

    def _coordinates(self, points: Array, /) -> tuple[Array, Array, Array, Array, Array]:
        shape = jnp.asarray(self.grid_shape, dtype=points.dtype)
        raw_coordinate = (points - self.plan.origin) / self.plan.spacing
        coordinate_finite = jnp.isfinite(raw_coordinate)
        outside = jnp.any(
            ~coordinate_finite | (raw_coordinate < 0.0) | (raw_coordinate > shape - 1.0),
            axis=-1,
        )
        coordinate = jnp.where(coordinate_finite, raw_coordinate, 0.0)
        if self.plan.boundary_policy is ExternalFieldBoundaryPolicy.PERIODIC:
            coordinate = jnp.mod(coordinate, shape)
            lower = jnp.floor(coordinate).astype(jnp.int32)
            upper = jnp.mod(lower + 1, jnp.asarray(self.grid_shape, dtype=jnp.int32))
            fraction = coordinate - lower.astype(coordinate.dtype)
            derivative_gate = jnp.ones_like(coordinate)
        else:
            clipped = jnp.clip(coordinate, 0.0, shape - 1.0)
            lower = jnp.minimum(
                jnp.floor(clipped).astype(jnp.int32),
                jnp.asarray(self.grid_shape, dtype=jnp.int32) - 2,
            )
            upper = lower + 1
            fraction = clipped - lower.astype(clipped.dtype)
            derivative_gate = ((coordinate >= 0.0) & (coordinate <= shape - 1.0)).astype(
                coordinate.dtype
            )
        return lower, upper, fraction, outside, derivative_gate

    def evaluate(self, points: ArrayLike, /) -> ExternalFieldEvaluation:
        coordinate = jnp.asarray(points)
        if coordinate.ndim != 2 or coordinate.shape[1] != 3:
            raise ValueError("External-field query points must have shape (count, 3).")
        if coordinate.dtype.kind != "f":
            raise TypeError("External-field query points must be floating point.")
        lower, upper, fraction, outside, derivative_gate = self._coordinates(coordinate)
        component_shape = () if self.plan.scalar else (self.plan.component_count,)
        interpolated = jnp.zeros(
            (coordinate.shape[0],) + component_shape, coordinate.dtype
        )
        jacobian = jnp.zeros(
            (coordinate.shape[0],) + component_shape + (3,), coordinate.dtype
        )
        for corner in range(8):
            bits = jnp.asarray(
                ((corner >> 0) & 1, (corner >> 1) & 1, (corner >> 2) & 1),
                dtype=jnp.int32,
            )
            indices = jnp.where(bits[None, :] == 1, upper, lower)
            node = self.plan.values[indices[:, 0], indices[:, 1], indices[:, 2]]
            factors = jnp.where(bits[None, :] == 1, fraction, 1.0 - fraction)
            weight = jnp.prod(factors, axis=-1)
            if component_shape:
                interpolated = interpolated + weight[:, None] * node
            else:
                interpolated = interpolated + weight * node
            for axis in range(3):
                sign = jnp.where(bits[axis] == 1, 1.0, -1.0)
                other = jnp.prod(
                    jnp.where(
                        jnp.arange(3)[None, :] == axis,
                        1.0,
                        factors,
                    ),
                    axis=-1,
                )
                derivative = (
                    sign * other * derivative_gate[:, axis] / self.plan.spacing[axis]
                )
                if component_shape:
                    jacobian = jacobian.at[..., axis].add(derivative[:, None] * node)
                else:
                    jacobian = jacobian.at[:, axis].add(derivative * node)
        failed_domain = (
            outside
            if self.plan.boundary_policy is ExternalFieldBoundaryPolicy.FAIL
            else jnp.zeros_like(outside)
        )
        if self.plan.boundary_policy is ExternalFieldBoundaryPolicy.FAIL:
            broadcast = failed_domain.reshape(
                failed_domain.shape + (1,) * len(component_shape)
            )
            interpolated = jnp.where(broadcast, jnp.nan, interpolated)
            jacobian = jnp.where(broadcast[..., None], jnp.nan, jacobian)
        finite = (
            jnp.all(jnp.isfinite(coordinate))
            & jnp.all(jnp.isfinite(interpolated))
            & jnp.all(jnp.isfinite(jacobian))
        )
        evidence = ExternalFieldEvidence(
            outside,
            jnp.sum(outside.astype(jnp.int32)),
            finite,
            finite & ~jnp.any(failed_domain),
        )
        return ExternalFieldEvaluation(interpolated, jacobian, evidence, self.prepared_id)

    def energy_and_forces(
        self, positions: ArrayLike, /, *, coupling: ArrayLike | None = None
    ) -> ScalarExternalFieldResult:
        if not self.plan.scalar:
            raise ValueError("Energy and conservative forces require a scalar field.")
        evaluation = self.evaluate(positions)
        count = evaluation.values.shape[0]
        weights = (
            jnp.ones((count,), dtype=evaluation.values.dtype)
            if coupling is None
            else jnp.asarray(coupling, dtype=evaluation.values.dtype)
        )
        if weights.shape != (count,):
            raise ValueError("Scalar-field coupling must provide one value per particle.")
        per_particle = weights * evaluation.values
        energy = jnp.sum(per_particle)
        forces = -weights[:, None] * evaluation.jacobian
        finite = (
            evaluation.evidence.finite
            & jnp.isfinite(energy)
            & jnp.all(jnp.isfinite(forces))
        )
        evidence = ExternalFieldEvidence(
            evaluation.evidence.out_of_domain,
            evaluation.evidence.out_of_domain_count,
            finite,
            evaluation.evidence.successful & finite,
        )
        return ScalarExternalFieldResult(
            energy, forces, per_particle, evidence, self.prepared_id
        )


__all__ = [
    "ExternalFieldBoundaryPolicy",
    "ExternalFieldEvaluation",
    "ExternalFieldEvidence",
    "GriddedExternalFieldPlan",
    "PreparedGriddedExternalField",
    "ScalarExternalFieldResult",
]
