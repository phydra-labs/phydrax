#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from enum import IntFlag
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._sharp_measures import QualifiedSharpGeometry
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume import (
    FaceVelocity,
    MACBoundaryStageData,
    PreparedMACBoundaryPlan,
    PreparedMACOperators,
)
from ..linalg import (
    ArraySpace,
    ConjugateGradient,
    DiagonalPairing,
    DifferentiationPolicy,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    OperatorProperties,
    solve,
    TolerancePolicy,
)


class MACSharpInterfaceStatus(IntFlag):
    SUCCESS = 0
    LINEAR_SOLVE_FAILED = 1
    DIVERGENCE_FAILED = 2
    GEOMETRY_FAILED = 4
    NONFINITE = 8
    OPERATOR_EVIDENCE_FAILED = 16


class MACSharpOperatorEvidence(StrictModule, NonTrainableState):
    """Numerical witnesses for the declared weighted active pressure space."""

    weighted_adjoint_residual: Array
    symmetry_residual: Array
    component_nullspace_residual: Array
    minimum_probe_rayleigh_quotient: Array
    finite: Array
    passed: Array
    evidence_id: str = eqx.field(static=True)


class MACSharpInterfaceForce(StrictModule):
    force: Array
    torque: Array
    pressure_force: Array
    viscous_force: Array
    finite: Array
    available: Array
    geometry_id: str = eqx.field(static=True)


class MACSharpInterfaceProjectionResult(StrictModule):
    velocity: FaceVelocity
    pressure: Array
    divergence_before: Array
    divergence_after: Array
    divergence_norm: Array
    linear: LinearSolveResult
    force: MACSharpInterfaceForce
    stabilization_defect: Array
    geometry_evidence: Any
    operator_evidence: MACSharpOperatorEvidence
    status: Array
    accepted: Array
    geometry_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def _active_components(
    operators: PreparedMACOperators,
    boundaries: PreparedMACBoundaryPlan,
    geometry: QualifiedSharpGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = operators.discretization.cell_shape
    active = np.asarray(geometry.cell_active, dtype=bool)
    apertures = tuple(np.asarray(value) for value in geometry.face_open_measure_lower)
    flat_active = np.flatnonzero(active.reshape(-1))
    if flat_active.size == 0:
        raise ValueError("Sharp projection requires at least one certainly fluid cell.")
    active_position = {int(flat): index for index, flat in enumerate(flat_active)}
    union = _UnionFind(int(flat_active.size))

    def connect(first: tuple[int, ...], second: tuple[int, ...], opened: bool) -> None:
        first_flat = int(np.ravel_multi_index(first, shape))
        second_flat = int(np.ravel_multi_index(second, shape))
        first_active = first_flat in active_position
        second_active = second_flat in active_position
        if opened and first_active != second_active:
            raise ValueError(
                "A certainly open MAC face cannot connect active and inactive cells."
            )
        if opened and first_active:
            union.union(active_position[first_flat], active_position[second_flat])

    for axis, structured_axis in enumerate(operators.discretization.grid.structured_axes):
        if structured_axis.periodic:
            for face_index in np.ndindex(apertures[axis].shape):
                upper = list(face_index)
                lower = list(face_index)
                lower[axis] = (lower[axis] - 1) % shape[axis]
                connect(tuple(lower), tuple(upper), apertures[axis][face_index] > 0.0)
        else:
            interior_shape = list(apertures[axis].shape)
            interior_shape[axis] -= 2
            for local in np.ndindex(tuple(interior_shape)):
                face = list(local)
                face[axis] += 1
                lower = list(face)
                upper = list(face)
                lower[axis] -= 1
                connect(tuple(lower), tuple(upper), apertures[axis][tuple(face)] > 0.0)

    roots = [union.find(index) for index in range(flat_active.size)]
    ordered_roots: dict[int, int] = {}
    labels = np.empty(flat_active.size, dtype=np.int32)
    for index, root in enumerate(roots):
        if root not in ordered_roots:
            ordered_roots[root] = len(ordered_roots)
        labels[index] = ordered_roots[root]
    anchored = np.zeros(len(ordered_roots), dtype=bool)
    for boundary, axis, side_index in zip(
        boundaries.sides,
        boundaries.side_axes,
        boundaries.side_indices,
        strict=True,
    ):
        if boundary.kind not in ("pressure-outlet", "traction-open"):
            continue
        axis_index = int(axis)
        lower_side = int(side_index) == 0
        face_slice: list[slice | int] = [slice(None)] * len(shape)
        face_slice[axis_index] = (
            0 if lower_side else apertures[axis_index].shape[axis_index] - 1
        )
        open_boundary = apertures[axis_index][tuple(face_slice)] > 0.0
        cell_slice: list[slice | int] = [slice(None)] * len(shape)
        cell_slice[axis_index] = 0 if lower_side else shape[axis_index] - 1
        boundary_active = active[tuple(cell_slice)]
        for tangential in np.argwhere(open_boundary & boundary_active):
            cell_index = list(tangential)
            cell_index.insert(axis_index, 0 if lower_side else shape[axis_index] - 1)
            flat = int(np.ravel_multi_index(tuple(cell_index), shape))
            anchored[labels[active_position[flat]]] = True
    return flat_active.astype(np.int32), labels, ~anchored


class MACSharpInterfaceProjectionPlan(StrictModule, NonTrainableState):
    """Compatible sharp projection on explicit weighted active spaces.

    Cell pressure uses absolute fluid-volume weights. Open-face velocity uses
    aperture times the original face dual measure. Consequently ``G = -D*``;
    aperture occurs exactly once and no denominator floor is hidden in the
    operator. One gauge is added for every unanchored connected component.
    """

    operators: PreparedMACOperators
    boundaries: PreparedMACBoundaryPlan
    geometry: QualifiedSharpGeometry
    active_cell_indices: Array
    component_labels: Array
    unanchored_components: Array
    pressure_space: ArraySpace
    tolerance: float = eqx.field(static=True)
    linear_policy: LinearSolvePolicy
    operator_evidence: MACSharpOperatorEvidence
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        boundaries: PreparedMACBoundaryPlan,
        geometry: QualifiedSharpGeometry,
        /,
        *,
        tolerance: float = 1.0e-9,
        linear_policy: LinearSolvePolicy | None = None,
    ):
        if not isinstance(operators, PreparedMACOperators):
            raise TypeError("operators must be PreparedMACOperators.")
        if not isinstance(boundaries, PreparedMACBoundaryPlan):
            raise TypeError("boundaries must be PreparedMACBoundaryPlan.")
        if boundaries.operators.prepared_id != operators.prepared_id:
            raise ValueError("Sharp-interface boundaries and operators differ.")
        if not isinstance(geometry, QualifiedSharpGeometry):
            raise TypeError("geometry must be QualifiedSharpGeometry.")
        discretization = operators.discretization
        expected_pairing = canonical_fingerprint(
            {
                "pressure": operators.pressure_space.space_id,
                "velocity": operators.velocity_space.space_id,
            }
        )
        if (
            geometry.operator_id != operators.prepared_id
            or geometry.support_id != discretization.support.support_id
            or geometry.cell_field_id != discretization.cell_space.field_space_id
            or geometry.face_field_ids
            != tuple(space.field_space_id for space in discretization.face_spaces)
            or geometry.pairing_id != expected_pairing
        ):
            raise ValueError("Qualified sharp geometry binds another MAC support/space.")
        if not bool(np.asarray(geometry.accepted)) or not bool(
            np.asarray(geometry.qualified)
        ):
            raise ValueError(
                "Sharp projection rejects unaccepted or unqualified geometry."
            )
        tolerance_ = float(tolerance)
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Sharp-interface tolerance must be positive and finite.")
        indices, labels, unanchored = _active_components(operators, boundaries, geometry)
        active_volumes = np.asarray(geometry.cell_fluid_measure).reshape(-1)[indices]
        pressure_space = ArraySpace(
            (int(indices.size),),
            dtype=operators.pressure_space.dtype,
            pairing=DiagonalPairing(
                jnp.asarray(active_volumes),
                pairing_id=canonical_fingerprint(
                    {
                        "kind": "sharp-active-fluid-volume-pairing",
                        "geometry": geometry.realization_id,
                    }
                ),
            ),
        )
        policy = (
            LinearSolvePolicy(
                ConjugateGradient(),
                tolerance=TolerancePolicy(
                    relative=tolerance_, absolute=tolerance_, max_steps=1000
                ),
                differentiation=DifferentiationPolicy("mathematical"),
            )
            if linear_policy is None
            else linear_policy
        )
        if not isinstance(policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be LinearSolvePolicy or None.")
        self.operators = operators
        self.boundaries = boundaries
        self.geometry = geometry
        self.active_cell_indices = jnp.asarray(indices)
        self.component_labels = jnp.asarray(labels)
        self.unanchored_components = jnp.asarray(unanchored)
        self.pressure_space = pressure_space
        self.tolerance = tolerance_
        self.linear_policy = policy
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-qualified-sharp-interface-projection",
                "operators": operators.prepared_id,
                "boundaries": boundaries.prepared_id,
                "geometry": geometry.realization_id,
                "active_cells": int(indices.size),
                "components": int(unanchored.size),
                "unanchored_components": unanchored.tolist(),
                "tolerance": tolerance_,
            }
        )
        evidence = self._build_operator_evidence()
        self.operator_evidence = evidence
        if not bool(np.asarray(evidence.passed)):
            raise RuntimeError(
                "Sharp active operators failed weighted compatibility evidence."
            )

    @property
    def component_count(self) -> int:
        return int(self.unanchored_components.size)

    def _pack(self, value: ArrayLike, /) -> Array:
        full = self.operators.validate_pressure(value)
        return full.reshape(-1)[self.active_cell_indices]

    def _unpack(self, value: ArrayLike, /) -> Array:
        active = self.pressure_space.validate(jnp.asarray(value))
        full = jnp.zeros(
            self.operators.discretization.cell_shape, dtype=active.dtype
        ).reshape(-1)
        return (
            full.at[self.active_cell_indices]
            .set(active)
            .reshape(self.operators.discretization.cell_shape)
        )

    def _component_means(self, value: Array, /) -> Array:
        weights = self.geometry.cell_fluid_measure.reshape(-1)[
            self.active_cell_indices
        ].astype(value.dtype)
        means = []
        for component in range(self.component_count):
            mask = self.component_labels == component
            denominator = jnp.sum(jnp.where(mask, weights, 0.0))
            means.append(jnp.sum(jnp.where(mask, weights * value, 0.0)) / denominator)
        return jnp.stack(tuple(means))

    def gauge(self, value: ArrayLike, /) -> Array:
        active = self.pressure_space.validate(jnp.asarray(value))
        means = self._component_means(active)
        correction = means[self.component_labels]
        remove = self.unanchored_components[self.component_labels]
        return jnp.where(remove, active - correction, active)

    def _gauge_completion(self, value: Array, /) -> Array:
        means = self._component_means(value)
        include = self.unanchored_components[self.component_labels]
        return jnp.where(include, means[self.component_labels], 0.0)

    def divergence(self, velocity: FaceVelocity, /) -> Array:
        values = self.operators.validate_velocity(velocity)
        integrated = tuple(
            aperture * value
            for aperture, value in zip(
                self.geometry.face_open_fraction, values, strict=True
            )
        )
        full_volume_divergence = self.operators.divergence(integrated)
        numerator = (
            self.geometry.cell_full_measure.astype(full_volume_divergence.dtype)
            * full_volume_divergence
        )
        denominator = self.geometry.cell_fluid_measure.astype(
            full_volume_divergence.dtype
        )
        return jnp.where(
            self.geometry.cell_active,
            numerator / jnp.where(self.geometry.cell_active, denominator, 1.0),
            0.0,
        )

    def gradient(
        self,
        pressure: ArrayLike,
        stage: MACBoundaryStageData | None,
        /,
        *,
        homogeneous: bool,
    ) -> FaceVelocity:
        return self.boundaries.pressure_gradient(pressure, stage, homogeneous=homogeneous)

    def _core_action(
        self,
        value: Array,
        inverse_momentum: FaceVelocity,
        stage: MACBoundaryStageData,
        /,
    ) -> Array:
        pressure = self._unpack(value)
        derivative = self.gradient(pressure, stage, homogeneous=True)
        correction = tuple(
            coefficient * gradient
            for coefficient, gradient in zip(inverse_momentum, derivative, strict=True)
        )
        return self._pack(-self.divergence(correction))

    def _gauged_action(
        self,
        value: Array,
        inverse_momentum: FaceVelocity,
        stage: MACBoundaryStageData,
        /,
    ) -> Array:
        projected = self.gauge(value)
        core = self.gauge(self._core_action(projected, inverse_momentum, stage))
        return core + self._gauge_completion(value)

    def _transpose_action(
        self,
        value: Array,
        inverse_momentum: FaceVelocity,
        stage: MACBoundaryStageData,
        /,
    ) -> Array:
        weights = self.geometry.cell_fluid_measure.reshape(-1)[
            self.active_cell_indices
        ].astype(value.dtype)
        return weights * self._gauged_action(value / weights, inverse_momentum, stage)

    def _build_operator_evidence(self) -> MACSharpOperatorEvidence:
        dtype = self.operators.pressure_space.dtype
        count = self.pressure_space.shape[0]
        pressure = jnp.sin(jnp.arange(count, dtype=dtype) + 0.37)
        second = jnp.cos(jnp.arange(count, dtype=dtype) + 0.19)
        velocity = tuple(
            jnp.sin(
                jnp.arange(int(np.prod(layout.shape)), dtype=dtype) + axis + 0.41
            ).reshape(layout.shape)
            for axis, layout in enumerate(self.operators.discretization.face_layouts)
        )
        velocity = self.boundaries.homogeneous_rate(velocity)
        stage = self.boundaries.homogeneous_stage()
        full_pressure = self._unpack(pressure)
        derivative = self.gradient(full_pressure, stage, homogeneous=True)
        cell_pairing = jnp.sum(
            self.geometry.cell_fluid_measure * full_pressure * self.divergence(velocity)
        )
        face_pairing = sum(
            jnp.sum(dual * aperture * component * gradient)
            for dual, aperture, component, gradient in zip(
                self.operators.face_dual_measures,
                self.geometry.face_open_fraction,
                velocity,
                derivative,
                strict=True,
            )
        )
        adjoint_residual = jnp.abs(cell_pairing + face_pairing)
        unit = tuple(
            jnp.ones(layout.shape, dtype=dtype)
            for layout in self.operators.discretization.face_layouts
        )
        first_action = self._gauged_action(pressure, unit, stage)
        second_action = self._gauged_action(second, unit, stage)
        weights = self.geometry.cell_fluid_measure.reshape(-1)[
            self.active_cell_indices
        ].astype(dtype)
        symmetry_residual = jnp.abs(
            jnp.sum(weights * pressure * second_action)
            - jnp.sum(weights * first_action * second)
        )
        nullspace_residual = jnp.asarray(0.0, dtype=dtype)
        for component in range(self.component_count):
            if bool(np.asarray(self.unanchored_components[component])):
                constant = (self.component_labels == component).astype(dtype)
                nullspace_residual = jnp.maximum(
                    nullspace_residual,
                    jnp.max(jnp.abs(self._core_action(constant, unit, stage))),
                )
        probe = pressure + 0.23 * second
        probe_action = self._gauged_action(probe, unit, stage)
        denominator = jnp.sum(weights * probe**2)
        rayleigh = jnp.sum(weights * probe * probe_action) / denominator
        scale = jnp.maximum(
            1.0,
            jnp.maximum(
                jnp.abs(cell_pairing),
                jnp.maximum(
                    jnp.abs(face_pairing),
                    jnp.maximum(
                        jnp.sum(jnp.abs(weights * pressure * second_action)),
                        jnp.sum(jnp.abs(weights * first_action * second)),
                    ),
                ),
            ),
        )
        numeric_tolerance = 1024.0 * jnp.finfo(dtype).eps * scale
        finite = (
            jnp.isfinite(adjoint_residual)
            & jnp.isfinite(symmetry_residual)
            & jnp.isfinite(nullspace_residual)
            & jnp.isfinite(rayleigh)
        )
        passed = (
            finite
            & (adjoint_residual <= numeric_tolerance)
            & (symmetry_residual <= numeric_tolerance)
            & (nullspace_residual <= numeric_tolerance)
            & (rayleigh > 0.0)
        )
        return MACSharpOperatorEvidence(
            adjoint_residual,
            symmetry_residual,
            nullspace_residual,
            rayleigh,
            finite,
            passed,
            canonical_fingerprint(
                {
                    "kind": "mac-sharp-weighted-operator-evidence",
                    "plan": self.plan_id,
                    "probe_size": count,
                }
            ),
        )

    def force(
        self,
        pressure: ArrayLike,
        /,
        *,
        viscous_traction: ArrayLike | None = None,
        reference_point: ArrayLike | None = None,
    ) -> MACSharpInterfaceForce:
        pressure_ = self.operators.validate_pressure(pressure)
        dimension = len(self.operators.discretization.cell_shape)
        available = self.geometry.evidence.interface_moments_qualified
        zero_force = jnp.zeros((dimension,), dtype=pressure_.dtype)
        zero_torque = jnp.zeros((1 if dimension == 2 else 3,), dtype=pressure_.dtype)
        viscous = (
            jnp.zeros(self.geometry.interface_normal.shape, dtype=pressure_.dtype)
            if viscous_traction is None
            else jnp.asarray(viscous_traction, dtype=pressure_.dtype)
        )
        if viscous.shape != self.geometry.interface_normal.shape:
            raise ValueError("viscous_traction must have one vector per cell.")
        pressure_density = -pressure_[..., None] * self.geometry.interface_normal
        axes = tuple(range(dimension))
        pressure_force = jnp.sum(
            self.geometry.interface_measure[..., None] * pressure_density, axis=axes
        )
        viscous_force = jnp.sum(
            self.geometry.interface_measure[..., None] * viscous, axis=axes
        )
        point = (
            jnp.zeros((dimension,), dtype=pressure_.dtype)
            if reference_point is None
            else jnp.asarray(reference_point, dtype=pressure_.dtype)
        )
        arm = self.geometry.interface_centroid - point
        weighted = self.geometry.interface_measure[..., None] * (
            pressure_density + viscous
        )
        torque_density = (
            (arm[..., 0] * weighted[..., 1] - arm[..., 1] * weighted[..., 0])[..., None]
            if dimension == 2
            else jnp.cross(arm, weighted)
        )
        torque = jnp.sum(torque_density, axis=axes)
        pressure_force = jnp.where(available, pressure_force, zero_force)
        viscous_force = jnp.where(available, viscous_force, zero_force)
        torque = jnp.where(available, torque, zero_torque)
        total = pressure_force + viscous_force
        finite = jnp.all(jnp.isfinite(total)) & jnp.all(jnp.isfinite(torque))
        return MACSharpInterfaceForce(
            total,
            torque,
            pressure_force,
            viscous_force,
            finite,
            available,
            self.geometry.realization_id,
        )

    def project(
        self,
        velocity: FaceVelocity,
        inverse_momentum: FaceVelocity,
        boundary_stage: MACBoundaryStageData,
        /,
        *,
        pressure: ArrayLike | None = None,
        jump_source: ArrayLike | None = None,
        wall_velocity: FaceVelocity | None = None,
        viscous_traction: ArrayLike | None = None,
        reference_point: ArrayLike | None = None,
    ) -> MACSharpInterfaceProjectionResult:
        stage = self.boundaries.validate_stage(boundary_stage)
        values = self.boundaries.enforce(
            self.operators.validate_velocity(velocity), stage
        )
        inverse = self.operators.validate_velocity(inverse_momentum)
        inverse_positive = jnp.all(
            jnp.stack(
                tuple(jnp.all(jnp.isfinite(value) & (value > 0.0)) for value in inverse)
            )
        )
        wall = (
            self.geometry.wall_velocity
            if wall_velocity is None
            else self.operators.validate_velocity(wall_velocity)
        )
        values = tuple(
            jnp.where(active, value, prescribed)
            for active, value, prescribed in zip(
                self.geometry.face_active, values, wall, strict=True
            )
        )
        divergence_before = self.divergence(values)
        source = (
            jnp.zeros_like(divergence_before)
            if jump_source is None
            else self.operators.validate_pressure(jump_source)
        )
        swept_source = jnp.where(
            self.geometry.cell_active,
            self.geometry.swept_cell_measure_rate
            / jnp.where(self.geometry.cell_active, self.geometry.cell_fluid_measure, 1.0),
            0.0,
        )
        target = source + swept_source
        boundary_gradient = self.gradient(
            jnp.zeros_like(divergence_before), stage, homogeneous=False
        )
        boundary_corrected = tuple(
            value - coefficient * gradient
            for value, coefficient, gradient in zip(
                values, inverse, boundary_gradient, strict=True
            )
        )

        def action(value):
            return self._gauged_action(value, inverse, stage)

        def transpose_action(value):
            return self._transpose_action(value, inverse, stage)

        operator = FunctionLinearOperator(
            action,
            source=self.pressure_space,
            target=self.pressure_space,
            transpose_action=transpose_action,
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "verified",
                    "positive_definite": "construction",
                },
            ),
            operator_id=f"mac-sharp-pressure/{self.plan_id}",
        )
        right_hand_side = self.gauge(
            self._pack(-self.divergence(boundary_corrected) + target)
        )
        incoming = (
            jnp.zeros_like(divergence_before)
            if pressure is None
            else self.operators.validate_pressure(pressure)
        )
        initial = self.gauge(self._pack(incoming))
        linear = solve(
            LinearSystem(operator, problem_id=f"mac-sharp-pressure/{self.plan_id}"),
            right_hand_side,
            policy=self.linear_policy,
            initial_guess=initial,
        )
        pressure_active = self.gauge(linear.value)
        pressure_value = self._unpack(pressure_active)
        derivative = self.gradient(pressure_value, stage, homogeneous=True)
        total_gradient = tuple(
            homogeneous + boundary
            for homogeneous, boundary in zip(derivative, boundary_gradient, strict=True)
        )
        corrected_open = tuple(
            value - coefficient * gradient
            for value, coefficient, gradient in zip(
                values, inverse, total_gradient, strict=True
            )
        )
        corrected_candidate = tuple(
            jnp.where(active, value, prescribed)
            for active, value, prescribed in zip(
                self.geometry.face_active, corrected_open, wall, strict=True
            )
        )
        corrected_candidate = self.boundaries.enforce(corrected_candidate, stage)
        divergence_after_candidate = self.divergence(corrected_candidate) - target
        volumes = self.geometry.cell_fluid_measure.astype(divergence_before.dtype)
        divergence_norm = jnp.sqrt(jnp.sum(volumes * divergence_after_candidate**2))
        scale = jnp.sqrt(jnp.sum(volumes * divergence_before**2))
        integrated_force = self.force(
            pressure_value,
            viscous_traction=viscous_traction,
            reference_point=reference_point,
        )
        finite = (
            inverse_positive
            & jnp.isfinite(divergence_norm)
            & integrated_force.finite
            & jnp.all(jnp.isfinite(pressure_value))
        )
        divergence_valid = divergence_norm <= self.tolerance * jnp.maximum(scale, 1.0)
        geometry_valid = self.geometry.accepted & self.operator_evidence.passed
        accepted = (
            linear.successful
            & divergence_valid
            & finite
            & geometry_valid
            & stage.successful
        )
        corrected = tuple(
            jnp.where(accepted, candidate, original)
            for candidate, original in zip(corrected_candidate, values, strict=True)
        )
        pressure_output = jnp.where(accepted, pressure_value, incoming)
        divergence_after = self.divergence(corrected) - target
        status = jnp.asarray(int(MACSharpInterfaceStatus.SUCCESS), dtype=jnp.int32)
        status = status | jnp.where(
            linear.successful, 0, int(MACSharpInterfaceStatus.LINEAR_SOLVE_FAILED)
        ).astype(jnp.int32)
        status = status | jnp.where(
            divergence_valid, 0, int(MACSharpInterfaceStatus.DIVERGENCE_FAILED)
        ).astype(jnp.int32)
        status = status | jnp.where(
            self.geometry.accepted, 0, int(MACSharpInterfaceStatus.GEOMETRY_FAILED)
        ).astype(jnp.int32)
        status = status | jnp.where(
            self.operator_evidence.passed,
            0,
            int(MACSharpInterfaceStatus.OPERATOR_EVIDENCE_FAILED),
        ).astype(jnp.int32)
        status = status | jnp.where(
            finite, 0, int(MACSharpInterfaceStatus.NONFINITE)
        ).astype(jnp.int32)
        return MACSharpInterfaceProjectionResult(
            corrected,
            pressure_output,
            divergence_before,
            divergence_after,
            divergence_norm,
            linear,
            integrated_force,
            jnp.asarray(0.0, dtype=divergence_norm.dtype),
            self.geometry.evidence,
            self.operator_evidence,
            status,
            accepted,
            self.geometry.realization_id,
            self.plan_id,
        )


MACSharpGeometryProvider = Callable[[Array, Any], QualifiedSharpGeometry]
MACInterfaceJumpSource = Callable[[Array, QualifiedSharpGeometry, Any], Array]


class MACMovingSharpInterfaceEpochResult(StrictModule):
    candidate_geometry: QualifiedSharpGeometry
    geometry: QualifiedSharpGeometry
    projection: MACSharpInterfaceProjectionPlan
    time: Array
    step_size: Array
    volume_rate: Array
    swept_volume_rate: Array
    gcl_residual: Array
    maximum_gcl_residual: Array
    finite: Array
    accepted: Array
    refresh_required: Array
    epoch_id: str = eqx.field(static=True)


class MACMovingSharpInterfaceEpochPlan(StrictModule, NonTrainableState):
    """Host refresh owner for qualified fixed-capacity moving geometry epochs."""

    operators: PreparedMACOperators
    boundaries: PreparedMACBoundaryPlan
    provider: MACSharpGeometryProvider = eqx.field(static=True)
    geometry_family_id: str = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        boundaries: PreparedMACBoundaryPlan,
        provider: MACSharpGeometryProvider,
        /,
        *,
        geometry_family_id: str,
        tolerance: float = 1.0e-9,
    ):
        if not callable(provider):
            raise TypeError("provider must be callable.")
        identifier = str(geometry_family_id)
        tolerance_ = float(tolerance)
        if not identifier or not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Moving sharp-interface policy is invalid.")
        self.operators = operators
        self.boundaries = boundaries
        self.provider = provider
        self.geometry_family_id = identifier
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-qualified-moving-sharp-interface-epochs",
                "operators": operators.prepared_id,
                "boundaries": boundaries.prepared_id,
                "geometry_family": identifier,
                "tolerance": tolerance_,
            }
        )

    def projection(
        self, time: ArrayLike, args: Any = None, /
    ) -> tuple[QualifiedSharpGeometry, MACSharpInterfaceProjectionPlan]:
        geometry = self.provider(jnp.asarray(time), args)
        if not isinstance(geometry, QualifiedSharpGeometry):
            raise TypeError("Moving sharp provider must return QualifiedSharpGeometry.")
        return geometry, MACSharpInterfaceProjectionPlan(
            self.operators,
            self.boundaries,
            geometry,
            tolerance=self.tolerance,
        )

    def transition(
        self,
        previous_time: ArrayLike,
        previous: QualifiedSharpGeometry,
        time: ArrayLike,
        args: Any = None,
        /,
    ) -> MACMovingSharpInterfaceEpochResult:
        if not isinstance(previous, QualifiedSharpGeometry):
            raise TypeError("previous must be QualifiedSharpGeometry.")
        previous_time_ = jnp.asarray(previous_time)
        time_ = jnp.asarray(time)
        step = time_ - previous_time_
        step = eqx.error_if(
            step,
            ~jnp.isfinite(step) | (step <= 0.0),
            "Moving sharp-interface epoch requires a positive time step.",
        )
        candidate, candidate_projection = self.projection(time_, args)
        volume_rate = (candidate.cell_fluid_measure - previous.cell_fluid_measure) / step
        residual = volume_rate - candidate.swept_cell_measure_rate
        maximum = jnp.max(jnp.abs(residual))
        scale = jnp.maximum(
            1.0,
            jnp.max(jnp.abs(volume_rate))
            + jnp.max(jnp.abs(candidate.swept_cell_measure_rate)),
        )
        finite = jnp.all(jnp.isfinite(residual))
        accepted = candidate.accepted & finite & (maximum <= self.tolerance * scale)
        if bool(np.asarray(accepted)):
            geometry = candidate
            projection = candidate_projection
        else:
            geometry = previous
            projection = MACSharpInterfaceProjectionPlan(
                self.operators,
                self.boundaries,
                previous,
                tolerance=self.tolerance,
            )
        return MACMovingSharpInterfaceEpochResult(
            candidate,
            geometry,
            projection,
            time_,
            step,
            volume_rate,
            candidate.swept_cell_measure_rate,
            residual,
            maximum,
            finite,
            accepted,
            ~accepted,
            canonical_fingerprint(
                {
                    "kind": "mac-moving-sharp-interface-epoch",
                    "plan": self.plan_id,
                    "accepted_geometry": geometry.realization_id,
                }
            ),
        )


class MACImmersedInterfaceProjectionPlan(StrictModule, NonTrainableState):
    """Sharp projection with an explicit pressure/stress-jump cell source."""

    sharp: MACSharpInterfaceProjectionPlan
    jump_source: MACInterfaceJumpSource = eqx.field(static=True)
    jump_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        sharp: MACSharpInterfaceProjectionPlan,
        jump_source: MACInterfaceJumpSource,
        /,
        *,
        jump_id: str,
    ):
        if not isinstance(sharp, MACSharpInterfaceProjectionPlan):
            raise TypeError("sharp must be MACSharpInterfaceProjectionPlan.")
        if not callable(jump_source):
            raise TypeError("jump_source must be callable.")
        identifier = str(jump_id)
        if not identifier:
            raise ValueError("jump_id must be nonempty.")
        self.sharp = sharp
        self.jump_source = jump_source
        self.jump_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-immersed-interface-projection",
                "sharp": sharp.plan_id,
                "jump": identifier,
            }
        )

    def project(
        self,
        time: ArrayLike,
        velocity: FaceVelocity,
        inverse_momentum: FaceVelocity,
        boundary_stage: MACBoundaryStageData,
        /,
        *,
        args: Any = None,
        pressure: ArrayLike | None = None,
        wall_velocity: FaceVelocity | None = None,
        viscous_traction: ArrayLike | None = None,
        reference_point: ArrayLike | None = None,
    ) -> MACSharpInterfaceProjectionResult:
        source = jnp.asarray(
            self.jump_source(jnp.asarray(time), self.sharp.geometry, args),
            dtype=self.sharp.operators.pressure_space.dtype,
        )
        source = self.sharp.operators.validate_pressure(source)
        return self.sharp.project(
            velocity,
            inverse_momentum,
            boundary_stage,
            pressure=pressure,
            jump_source=source,
            wall_velocity=wall_velocity,
            viscous_traction=viscous_traction,
            reference_point=reference_point,
        )


MACInterfaceEnforcement = Literal[
    "regularized-delta", "divergence-free", "sharp", "immersed-interface"
]


class MACInterfaceMethodSelector(StrictModule, NonTrainableState):
    """One explicit enforcement-family selection for configuration and replay."""

    method: MACInterfaceEnforcement = eqx.field(static=True)
    plan: object
    selector_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: MACInterfaceEnforcement,
        plan: object,
        /,
    ):
        if method not in (
            "regularized-delta",
            "divergence-free",
            "sharp",
            "immersed-interface",
        ):
            raise ValueError("Unknown MAC interface enforcement family.")
        from ._mac_dfib import MACDFIBProjectionPlan
        from ._mac_immersed_boundary import MACImmersedBoundaryProjectionPlan

        expected = {
            "regularized-delta": MACImmersedBoundaryProjectionPlan,
            "divergence-free": MACDFIBProjectionPlan,
            "sharp": MACSharpInterfaceProjectionPlan,
            "immersed-interface": MACImmersedInterfaceProjectionPlan,
        }[method]
        if not isinstance(plan, expected):
            raise TypeError("Selected MAC interface family and projection plan differ.")
        self.method = method
        self.plan = plan
        self.selector_id = canonical_fingerprint(
            {
                "kind": "mac-interface-method-selector",
                "method": method,
                "plan": plan.plan_id,
            }
        )


__all__ = [
    "MACImmersedInterfaceProjectionPlan",
    "MACInterfaceEnforcement",
    "MACInterfaceJumpSource",
    "MACInterfaceMethodSelector",
    "MACMovingSharpInterfaceEpochPlan",
    "MACMovingSharpInterfaceEpochResult",
    "MACSharpGeometryProvider",
    "MACSharpInterfaceForce",
    "MACSharpInterfaceProjectionPlan",
    "MACSharpInterfaceProjectionResult",
    "MACSharpInterfaceStatus",
    "MACSharpOperatorEvidence",
]
