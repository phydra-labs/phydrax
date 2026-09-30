#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    ArraySpace,
    BlockSpace,
    DiagonalPairing,
    FunctionLinearOperator,
    HilbertComplex,
)
from ._structured import FiniteVolumeDiscretization


FaceVelocity = tuple[Array, ...]


def _difference(integrated: Array, axis: int, periodic: bool, /) -> Array:
    if periodic:
        return jnp.roll(integrated, -1, axis=axis) - integrated
    lower = [slice(None)] * integrated.ndim
    upper = [slice(None)] * integrated.ndim
    lower[axis] = slice(0, integrated.shape[axis] - 1)
    upper[axis] = slice(1, integrated.shape[axis])
    return integrated[tuple(upper)] - integrated[tuple(lower)]


class MACOperatorReport(StrictModule, NonTrainableState):
    """Weighted adjoint, nullspace, and direct-transform eligibility evidence."""

    weighted_adjoint_residual: Array
    constant_laplacian_residual: Array
    transform_eligible: bool = eqx.field(static=True)
    finite: Array
    passed: Array
    report_id: str = eqx.field(static=True)


class MACOperatorPlan(StrictModule, NonTrainableState):
    """Prepare geometry-only MAC divergence, gradient, gauge, and pressure actions."""

    discretization: FiniteVolumeDiscretization
    plan_id: str = eqx.field(static=True)

    def __init__(self, discretization: FiniteVolumeDiscretization, /) -> None:
        if not isinstance(discretization, FiniteVolumeDiscretization):
            raise TypeError("discretization must be a FiniteVolumeDiscretization.")
        self.discretization = discretization
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-operator-plan",
                "discretization": discretization.prepared_id,
            }
        )

    def prepare(self, /) -> PreparedMACOperators:
        return PreparedMACOperators(self)


@final
class PreparedMACOperators(StrictModule, NonTrainableState):
    """Compatible cell-pressure and normal-face-velocity tensor operators."""

    discretization: FiniteVolumeDiscretization
    pressure_space: ArraySpace
    velocity_space: BlockSpace
    face_dual_measures: FaceVelocity
    report: MACOperatorReport
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: MACOperatorPlan, /) -> None:
        if not isinstance(plan, MACOperatorPlan):
            raise TypeError("plan must be a MACOperatorPlan.")
        discretization = plan.discretization
        volumes = discretization.cell_volumes
        pressure_space = ArraySpace(
            discretization.cell_shape,
            dtype=volumes.dtype,
            pairing=DiagonalPairing(volumes),
        )
        dual_measures = []
        transform_eligible = True
        for axis, structured_axis in enumerate(discretization.grid.structured_axes):
            centers = structured_axis.interval_centers
            widths = np.asarray(structured_axis.interval_widths, dtype=np.float64)
            transform_eligible = transform_eligible and bool(
                np.allclose(widths, widths[0], rtol=1e-10, atol=1e-12)
            )
            if structured_axis.periodic:
                period = structured_axis.bounds[1] - structured_axis.bounds[0]
                previous = jnp.roll(centers, 1).at[0].add(-period)
                distance = centers - previous
            elif centers.size == 1:
                distance = jnp.asarray((0.5 * widths[0], 0.5 * widths[0]))
            else:
                interior = centers[1:] - centers[:-1]
                distance = jnp.concatenate(
                    (0.5 * widths[:1], interior, 0.5 * widths[-1:])
                )
            reshape = [1] * len(discretization.cell_shape)
            reshape[axis] = distance.size
            dual_measures.append(
                discretization.face_measures[axis] * distance.reshape(reshape)
            )
        dual_measures_ = tuple(dual_measures)
        velocity_space = BlockSpace(
            tuple(
                ArraySpace(
                    layout.shape,
                    dtype=volumes.dtype,
                    pairing=DiagonalPairing(measure),
                )
                for layout, measure in zip(
                    discretization.face_layouts,
                    dual_measures_,
                    strict=True,
                )
            ),
            names=discretization.grid.axis_names,
        )
        identifier = canonical_fingerprint(
            {
                "kind": "prepared-mac-operators",
                "plan": plan.plan_id,
                "pressure_space": pressure_space.space_id,
                "velocity_space": velocity_space.space_id,
                "transform_eligible": transform_eligible,
            }
        )
        self.discretization = discretization
        self.pressure_space = pressure_space
        self.velocity_space = velocity_space
        self.face_dual_measures = dual_measures_
        self.prepared_id = identifier

        pressure = jnp.arange(
            int(np.prod(discretization.cell_shape)), dtype=volumes.dtype
        ).reshape(discretization.cell_shape)
        velocity = []
        for axis, layout in enumerate(discretization.face_layouts):
            component = jnp.sin(
                jnp.arange(int(np.prod(layout.shape)), dtype=volumes.dtype)
            ).reshape(layout.shape)
            if not discretization.grid.structured_axes[axis].periodic:
                lower: list[slice | int] = [slice(None)] * component.ndim
                upper: list[slice | int] = [slice(None)] * component.ndim
                lower[axis] = 0
                upper[axis] = component.shape[axis] - 1
                component = component.at[tuple(lower)].set(0.0)
                component = component.at[tuple(upper)].set(0.0)
            velocity.append(component)
        divergence = self.divergence(tuple(velocity))
        gradient = self.gradient(pressure)
        left_pairing = jnp.sum(volumes * pressure * divergence)
        right_pairing = sum(
            jnp.sum(measure * component * derivative)
            for measure, component, derivative in zip(
                self.face_dual_measures, velocity, gradient, strict=True
            )
        )
        adjoint_residual = jnp.abs(left_pairing + right_pairing)
        constant_residual = jnp.max(
            jnp.abs(self.positive_laplacian(jnp.ones(discretization.cell_shape)))
        )
        finite = jnp.isfinite(adjoint_residual) & jnp.isfinite(constant_residual)
        passed = finite & (adjoint_residual <= 5e-10) & (constant_residual <= 5e-12)
        self.report = MACOperatorReport(
            weighted_adjoint_residual=adjoint_residual,
            constant_laplacian_residual=constant_residual,
            transform_eligible=transform_eligible,
            finite=finite,
            passed=passed,
            report_id=canonical_fingerprint(
                {
                    "kind": "mac-operator-report",
                    "operators": identifier,
                    "transform_eligible": transform_eligible,
                }
            ),
        )
        if not bool(passed):
            raise RuntimeError("Prepared MAC operators failed compatibility evidence.")

    def validate_pressure(self, pressure: ArrayLike, /) -> Array:
        return self.pressure_space.validate(jnp.asarray(pressure))

    def validate_velocity(self, velocity: FaceVelocity, /) -> FaceVelocity:
        values = tuple(jnp.asarray(component) for component in velocity)
        if len(values) != len(self.discretization.cell_shape):
            raise ValueError("MAC velocity requires one normal component per axis.")
        return tuple(self.velocity_space.validate(values))

    def gauge_project(self, pressure: ArrayLike, /) -> Array:
        value = self.validate_pressure(pressure)
        volume = self.discretization.cell_volumes.astype(value.dtype)
        return value - jnp.sum(volume * value) / jnp.sum(volume)

    def compatibility_project(self, right_hand_side: ArrayLike, /) -> Array:
        value = self.validate_pressure(right_hand_side)
        volume = self.discretization.cell_volumes.astype(value.dtype)
        return value - jnp.sum(volume * value) / jnp.sum(volume)

    def divergence(self, velocity: FaceVelocity, /) -> Array:
        values = self.validate_velocity(velocity)
        divergence = jnp.zeros(
            self.discretization.cell_shape, dtype=self.pressure_space.dtype
        )
        for axis, component in enumerate(values):
            integrated = component * self.discretization.face_measures[axis]
            divergence = (
                divergence
                + _difference(
                    integrated,
                    axis,
                    self.discretization.grid.structured_axes[axis].periodic,
                )
                / self.discretization.cell_volumes
            )
        return divergence

    def integrated_mass_flux(self, velocity: FaceVelocity, /) -> Array:
        divergence = self.divergence(velocity)
        volumes = self.discretization.cell_volumes.astype(divergence.dtype)
        return jnp.sum(volumes * divergence)

    def gradient(self, pressure: ArrayLike, /) -> FaceVelocity:
        value = self.validate_pressure(pressure)
        output = []
        for axis, structured_axis in enumerate(self.discretization.grid.structured_axes):
            moved = jnp.moveaxis(value, axis, 0)
            centers = structured_axis.interval_centers
            if structured_axis.periodic:
                period = structured_axis.bounds[1] - structured_axis.bounds[0]
                previous = jnp.roll(moved, 1, axis=0)
                previous_centers = jnp.roll(centers, 1).at[0].add(-period)
                distance = centers - previous_centers
                gradient = (moved - previous) / distance.reshape(
                    (distance.size,) + (1,) * (moved.ndim - 1)
                )
            elif moved.shape[0] == 1:
                gradient = jnp.zeros((2,) + moved.shape[1:], dtype=moved.dtype)
            else:
                interior = (moved[1:] - moved[:-1]) / (
                    centers[1:] - centers[:-1]
                ).reshape((-1,) + (1,) * (moved.ndim - 1))
                gradient = jnp.concatenate(
                    (jnp.zeros_like(moved[:1]), interior, jnp.zeros_like(moved[:1])),
                    axis=0,
                )
            output.append(jnp.moveaxis(gradient, 0, axis))
        return tuple(output)

    def hilbert_complex_slice(self, /) -> HilbertComplex:
        """Face-to-cell complex with the homogeneous no-normal-flow boundary.

        Boundary faces remain in the retained layout but are inactive in the
        differential. This is the domain on which the existing Neumann gradient
        is minus the Hilbert adjoint of divergence.
        """

        def divergence(velocity: FaceVelocity) -> Array:
            values = []
            for axis, component in enumerate(velocity):
                if not self.discretization.grid.structured_axes[axis].periodic:
                    lower: list[slice | int] = [slice(None)] * component.ndim
                    upper: list[slice | int] = [slice(None)] * component.ndim
                    lower[axis] = 0
                    upper[axis] = component.shape[axis] - 1
                    component = component.at[tuple(lower)].set(0).at[tuple(upper)].set(0)
                values.append(component)
            return self.divergence(tuple(values))

        def transpose(pressure: Array) -> FaceVelocity:
            coordinate = pressure / self.discretization.cell_volumes
            return tuple(
                -measure * component
                for measure, component in zip(
                    self.face_dual_measures, self.gradient(coordinate), strict=True
                )
            )

        differential = FunctionLinearOperator(
            divergence,
            source=self.velocity_space,
            target=self.pressure_space,
            transpose_action=transpose,
            operator_id=canonical_fingerprint(
                {"kind": "mac-complex-divergence", "operators": self.prepared_id}
            ),
        )
        return HilbertComplex(
            (self.velocity_space, self.pressure_space),
            (differential,),
            complex_id=canonical_fingerprint(
                {"kind": "mac-hilbert-complex-slice", "operators": self.prepared_id}
            ),
        )

    def interpolate_inverse_momentum(
        self, inverse_momentum_diagonal: ArrayLike, /
    ) -> FaceVelocity:
        inverse = self.validate_pressure(inverse_momentum_diagonal)
        inverse = eqx.error_if(
            inverse,
            jnp.any(~jnp.isfinite(inverse) | (inverse <= 0.0)),
            "Inverse momentum diagonal must be positive and finite.",
        )
        output = []
        for axis, structured_axis in enumerate(self.discretization.grid.structured_axes):
            moved = jnp.moveaxis(inverse, axis, 0)
            if structured_axis.periodic:
                face = 0.5 * (moved + jnp.roll(moved, 1, axis=0))
            else:
                interior = 0.5 * (moved[1:] + moved[:-1])
                face = jnp.concatenate((moved[:1], interior, moved[-1:]), axis=0)
            output.append(jnp.moveaxis(face, 0, axis))
        return tuple(output)

    def laplacian(self, pressure: ArrayLike, /) -> Array:
        return self.divergence(self.gradient(pressure))

    def weighted_laplacian(
        self,
        pressure: ArrayLike,
        face_inverse_momentum: FaceVelocity,
        /,
    ) -> Array:
        coefficient = self.validate_velocity(face_inverse_momentum)
        gradient = self.gradient(pressure)
        return self.divergence(
            tuple(
                value * derivative
                for value, derivative in zip(coefficient, gradient, strict=True)
            )
        )

    def positive_laplacian(self, pressure: ArrayLike, /) -> Array:
        return -self.laplacian(pressure)

    def positive_gauged_weighted_laplacian(
        self,
        pressure: ArrayLike,
        face_inverse_momentum: FaceVelocity,
        /,
    ) -> Array:
        value = self.validate_pressure(pressure)
        volume = self.discretization.cell_volumes.astype(value.dtype)
        mean = jnp.sum(volume * value) / jnp.sum(volume)
        projected = value - mean
        return -self.weighted_laplacian(projected, face_inverse_momentum) + mean

    def positive_gauged_weighted_laplacian_diagonal(
        self,
        face_inverse_momentum: FaceVelocity,
        /,
    ) -> Array:
        """Exact diagonal of `positive_gauged_weighted_laplacian`.

        Each gradient-carrying face couples its two cells through the conductance
        ``c_f A_f / d_f``, where ``A_f d_f`` is the face dual measure. Wall faces
        and single-cell periodic axes carry no gradient. The volume-weighted
        gauge row adds ``V_i / sum(V)``.
        """
        coefficient = self.validate_velocity(face_inverse_momentum)
        volumes = self.discretization.cell_volumes.astype(self.pressure_space.dtype)
        diagonal = volumes / jnp.sum(volumes)
        for axis, structured_axis in enumerate(self.discretization.grid.structured_axes):
            measure = self.discretization.face_measures[axis]
            conductance = jnp.moveaxis(
                coefficient[axis] * measure * measure / self.face_dual_measures[axis],
                axis,
                0,
            )
            if structured_axis.periodic:
                if conductance.shape[0] == 1:
                    conductance = jnp.zeros_like(conductance)
                incident = conductance + jnp.roll(conductance, -1, axis=0)
            else:
                conductance = conductance.at[0].set(0.0).at[-1].set(0.0)
                incident = conductance[:-1] + conductance[1:]
            diagonal = diagonal + jnp.moveaxis(incident, 0, axis) / volumes
        return diagonal

    def positive_gauged_laplacian(self, pressure: ArrayLike, /) -> Array:
        value = self.validate_pressure(pressure)
        unit = tuple(
            jnp.ones(layout.shape, dtype=value.dtype)
            for layout in self.discretization.face_layouts
        )
        return self.positive_gauged_weighted_laplacian(value, unit)


__all__ = [
    "FaceVelocity",
    "MACOperatorPlan",
    "MACOperatorReport",
    "PreparedMACOperators",
]
