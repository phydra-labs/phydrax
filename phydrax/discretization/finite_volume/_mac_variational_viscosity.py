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
from ._incompressible import FaceVelocity
from ._mac_boundary import MACBoundaryStageData
from ._mac_momentum import (
    _axis_values,
    _patch_to_component_faces,
    PreparedMACMomentumOperators,
)


def _cell_to_axis_faces(value: Array, axis: int, periodic: bool, /) -> Array:
    moved = jnp.moveaxis(value, axis, 0)
    if periodic:
        faces = 0.5 * (moved + jnp.roll(moved, 1, axis=0))
    else:
        faces = jnp.concatenate(
            (moved[:1], 0.5 * (moved[:-1] + moved[1:]), moved[-1:]), axis=0
        )
    return jnp.moveaxis(faces, 0, axis)


class MACVariationalViscosityResult(StrictModule, NonTrainableState):
    """Variational stress action and its stage-local energy evidence."""

    positive_operator_action: FaceVelocity
    physical_diffusive_rate: FaceVelocity
    boundary_affine_action: FaceVelocity
    integrated_dissipation: Array
    integrated_work: Array
    boundary_power: Array
    positive_work: Array
    variational_defect: Array
    operator_row_sum_bound: Array
    explicit_step_bound: Array
    restriction_supported: bool = eqx.field(static=True)
    finite: Array
    successful: Array
    action_id: str = eqx.field(static=True)
    boundary_stage_id: str = eqx.field(static=True)


class PreparedMACVariationalViscosityAction(StrictModule, NonTrainableState):
    """Prepared MAC deviatoric-strain action with runtime cell viscosity.

    The action is the velocity partial gradient of
    ``∫ ν S_d:S_d dV``. Its positive homogeneous action is therefore the
    Riesz representative of ``-div(2 ν S_d)``; the physical diffusive rate
    has the opposite sign. Cell viscosity is a dynamic argument and may be
    exactly zero. No cell-to-edge stress state is retained.
    """

    momentum: PreparedMACMomentumOperators
    cell_axis_widths: tuple[Array, ...]
    face_axis_widths: tuple[Array, ...]
    homogeneous_boundary_stage: MACBoundaryStageData
    restriction_supported: bool = eqx.field(static=True)
    action_id: str = eqx.field(static=True)

    def __init__(self, momentum: PreparedMACMomentumOperators, /):
        if not isinstance(momentum, PreparedMACMomentumOperators):
            raise TypeError("momentum must be PreparedMACMomentumOperators.")
        axes = momentum.operators.discretization.grid.structured_axes
        cell_widths = tuple(axis.interval_widths for axis in axes)
        face_widths = tuple(momentum.face_dual_widths)
        restriction_supported = all(
            axis.periodic
            and np.allclose(
                np.asarray(axis.interval_widths),
                float(np.asarray(axis.interval_widths)[0]),
                rtol=1.0e-10,
                atol=1.0e-12,
            )
            for axis in axes
        )
        self.momentum = momentum
        self.cell_axis_widths = cell_widths
        self.face_axis_widths = face_widths
        self.homogeneous_boundary_stage = momentum.boundaries.homogeneous_stage()
        self.restriction_supported = restriction_supported
        self.action_id = canonical_fingerprint(
            {
                "kind": "prepared-mac-variational-viscosity-action",
                "momentum": momentum.prepared_id,
                "strain": "deviatoric-symmetric",
                "stress": "tau_d=-2*nu*S_d",
                "factored_axis_widths": True,
                "periodic_uniform_restriction": restriction_supported,
            }
        )

    @property
    def dimension(self) -> int:
        return self.momentum.dimension

    def _viscosity(self, cell_viscosity: ArrayLike, /) -> Array:
        viscosity = self.momentum.operators.validate_pressure(cell_viscosity)
        return eqx.error_if(
            viscosity,
            jnp.any(~jnp.isfinite(viscosity) | (viscosity < 0.0)),
            "Cell viscosity must be finite and nonnegative.",
        )

    def _normal_derivative(self, value: Array, axis: int, /) -> Array:
        moved = jnp.moveaxis(value, axis, 0)
        widths = _axis_values(self.cell_axis_widths[axis], moved.ndim, 0)
        if self.momentum.operators.discretization.grid.structured_axes[axis].periodic:
            derivative = (jnp.roll(moved, -1, axis=0) - moved) / widths
        else:
            derivative = (moved[1:] - moved[:-1]) / widths
        return jnp.moveaxis(derivative, 0, axis)

    def _tangential_derivative(
        self,
        value: Array,
        component_axis: int,
        derivative_axis: int,
        stage: MACBoundaryStageData,
        /,
        *,
        homogeneous: bool,
    ) -> Array:
        grid = self.momentum.operators.discretization.grid
        axis = grid.structured_axes[derivative_axis]
        moved = jnp.moveaxis(value, derivative_axis, 0)
        centers = axis.interval_centers.astype(value.dtype)
        if axis.periodic:
            period = axis.bounds[1] - axis.bounds[0]
            previous_centers = jnp.roll(centers, 1).at[0].add(-period)
            distance = _axis_values(centers - previous_centers, moved.ndim, 0)
            derivative = (moved - jnp.roll(moved, 1, axis=0)) / distance
            return jnp.moveaxis(derivative, 0, derivative_axis)

        component_periodic = grid.structured_axes[component_axis].periodic
        if self.momentum.boundaries.tangential_dirichlet(derivative_axis, "lower"):
            lower_patch = self.momentum.boundaries.tangential_value(
                derivative_axis,
                "lower",
                component_axis,
                stage,
                homogeneous=homogeneous,
            )
            lower = _patch_to_component_faces(
                lower_patch,
                derivative_axis,
                component_axis,
                component_periodic,
            )
            lower = jnp.where(stage.successful, lower, 0.0)
            lower_derivative = (moved[:1] - lower) / (centers[0] - axis.bounds[0])
        else:
            lower_derivative = jnp.zeros_like(moved[:1])
        interior_derivative = (moved[1:] - moved[:-1]) / _axis_values(
            centers[1:] - centers[:-1], moved.ndim, 0
        )
        if self.momentum.boundaries.tangential_dirichlet(derivative_axis, "upper"):
            upper_patch = self.momentum.boundaries.tangential_value(
                derivative_axis,
                "upper",
                component_axis,
                stage,
                homogeneous=homogeneous,
            )
            upper = _patch_to_component_faces(
                upper_patch,
                derivative_axis,
                component_axis,
                component_periodic,
            )
            upper = jnp.where(stage.successful, upper, 0.0)
            upper_derivative = (upper - moved[-1:]) / (axis.bounds[1] - centers[-1])
        else:
            upper_derivative = jnp.zeros_like(moved[-1:])
        derivative = jnp.concatenate(
            (lower_derivative, interior_derivative, upper_derivative), axis=0
        )
        return jnp.moveaxis(derivative, 0, derivative_axis)

    def _edge_viscosity(
        self, viscosity: Array, first_axis: int, second_axis: int, /
    ) -> Array:
        grid = self.momentum.operators.discretization.grid
        result = _cell_to_axis_faces(
            viscosity, first_axis, grid.structured_axes[first_axis].periodic
        )
        return _cell_to_axis_faces(
            result, second_axis, grid.structured_axes[second_axis].periodic
        )

    def _edge_integral(self, value: Array, first_axis: int, second_axis: int, /) -> Array:
        weighted = value
        for axis in range(self.dimension):
            widths = (
                self.face_axis_widths[axis]
                if axis == first_axis or axis == second_axis
                else self.cell_axis_widths[axis]
            )
            weighted = weighted * _axis_values(widths, value.ndim, axis)
        return jnp.sum(weighted)

    def _potential(
        self,
        velocity: FaceVelocity,
        viscosity: Array,
        stage: MACBoundaryStageData,
        homogeneous: bool,
        /,
    ) -> Array:
        boundaries = self.momentum.boundaries
        values = (
            boundaries.homogeneous_rate(velocity)
            if homogeneous
            else boundaries.enforce(velocity, stage)
        )
        normal_derivatives = tuple(
            self._normal_derivative(value, axis) for axis, value in enumerate(values)
        )
        divergence = sum(normal_derivatives[1:], start=normal_derivatives[0])
        diagonal_strain_squared = sum(
            (derivative - divergence / float(self.dimension)) ** 2
            for derivative in normal_derivatives
        )
        volumes = self.momentum.operators.discretization.cell_volumes.astype(
            viscosity.dtype
        )
        potential = jnp.sum(volumes * viscosity * diagonal_strain_squared)
        for first_axis in range(self.dimension):
            for second_axis in range(first_axis + 1, self.dimension):
                first_derivative = self._tangential_derivative(
                    values[first_axis],
                    first_axis,
                    second_axis,
                    stage,
                    homogeneous=homogeneous,
                )
                second_derivative = self._tangential_derivative(
                    values[second_axis],
                    second_axis,
                    first_axis,
                    stage,
                    homogeneous=homogeneous,
                )
                shear = first_derivative + second_derivative
                edge_viscosity = self._edge_viscosity(viscosity, first_axis, second_axis)
                potential = potential + 0.5 * self._edge_integral(
                    edge_viscosity * shear**2, first_axis, second_axis
                )
        return self.momentum.precision.reduction(potential)

    def dissipation_potential(
        self,
        velocity: FaceVelocity,
        cell_viscosity: ArrayLike,
        boundary_stage: MACBoundaryStageData | None = None,
        /,
    ) -> Array:
        """Return half the integrated deviatoric viscous dissipation."""
        values = self.momentum.operators.validate_velocity(velocity)
        viscosity = self._viscosity(cell_viscosity)
        stage = (
            self.homogeneous_boundary_stage
            if boundary_stage is None
            else self.momentum.boundaries.validate_stage(boundary_stage)
        )
        return self._potential(values, viscosity, stage, False)

    def _gradient_action(
        self,
        velocity: FaceVelocity,
        viscosity: Array,
        stage: MACBoundaryStageData,
        /,
        *,
        homogeneous: bool,
    ) -> FaceVelocity:
        covector = jax.grad(PreparedMACVariationalViscosityAction._potential, argnums=1)(
            self, velocity, viscosity, stage, homogeneous
        )
        vector = self.momentum.operators.velocity_space.inverse_riesz(covector)
        return self.momentum.boundaries.homogeneous_rate(tuple(vector))

    def positive_operator_action(
        self, velocity: FaceVelocity, cell_viscosity: ArrayLike, /
    ) -> FaceVelocity:
        """Apply the positive homogeneous ``-div(2 ν S_d)`` operator."""
        values = self.momentum.operators.validate_velocity(velocity)
        viscosity = self._viscosity(cell_viscosity)
        return self._gradient_action(
            values,
            viscosity,
            self.homogeneous_boundary_stage,
            homogeneous=True,
        )

    def boundary_affine_action(
        self,
        cell_viscosity: ArrayLike,
        boundary_stage: MACBoundaryStageData,
        /,
    ) -> FaceVelocity:
        """Return the stage-boundary offset independent of face velocity."""
        viscosity = self._viscosity(cell_viscosity)
        stage = self.momentum.boundaries.validate_stage(boundary_stage)
        zero = tuple(
            jnp.zeros(layout.shape, dtype=self.momentum.operators.pressure_space.dtype)
            for layout in self.momentum.operators.discretization.face_layouts
        )
        return self._gradient_action(zero, viscosity, stage, homogeneous=False)

    def affine_positive_action(
        self,
        velocity: FaceVelocity,
        cell_viscosity: ArrayLike,
        boundary_stage: MACBoundaryStageData,
        /,
    ) -> FaceVelocity:
        """Apply the positive action including the prescribed-boundary offset."""
        values = self.momentum.operators.validate_velocity(velocity)
        viscosity = self._viscosity(cell_viscosity)
        stage = self.momentum.boundaries.validate_stage(boundary_stage)
        return self._gradient_action(values, viscosity, stage, homogeneous=False)

    def physical_diffusive_rate(
        self,
        velocity: FaceVelocity,
        cell_viscosity: ArrayLike,
        boundary_stage: MACBoundaryStageData,
        /,
    ) -> FaceVelocity:
        """Return ``div(2 ν S_d)``, including affine boundary data."""
        action = self.affine_positive_action(velocity, cell_viscosity, boundary_stage)
        return tuple(-value for value in action)

    def operator_row_sum_bound(self, cell_viscosity: ArrayLike, /) -> Array:
        """Return a conservative periodic-uniform row-sum restriction bound.

        Nonperiodic and nonuniform layouts have no certified bound and return
        infinity; ``restriction_supported`` reports that caveat explicitly.
        """
        viscosity = self._viscosity(cell_viscosity)
        if not self.restriction_supported:
            return jnp.asarray(jnp.inf, dtype=viscosity.dtype)
        if self.dimension == 1:
            return jnp.asarray(0.0, dtype=viscosity.dtype)
        inverse_widths = jnp.stack(
            tuple(1.0 / widths[0] for widths in self.cell_axis_widths)
        ).astype(viscosity.dtype)
        reciprocal_sum = jnp.sum(inverse_widths)
        squared_sum = jnp.sum(inverse_widths**2)
        component_bounds = 16.0 * inverse_widths * reciprocal_sum + 4.0 * (
            squared_sum + inverse_widths * reciprocal_sum - 2.0 * inverse_widths**2
        )
        return jnp.max(viscosity) * jnp.max(component_bounds)

    def explicit_step_bound(self, cell_viscosity: ArrayLike, /) -> Array:
        """Return the certified forward-Euler step bound when available."""
        row_sum = self.operator_row_sum_bound(cell_viscosity)
        return jnp.where(row_sum > 0.0, 2.0 / row_sum, jnp.inf)

    def evaluate(
        self,
        velocity: FaceVelocity,
        cell_viscosity: ArrayLike,
        boundary_stage: MACBoundaryStageData,
        /,
    ) -> MACVariationalViscosityResult:
        """Evaluate action, physical rate, work, and fail-closed evidence."""
        values = self.momentum.operators.validate_velocity(velocity)
        viscosity = self._viscosity(cell_viscosity)
        stage = self.momentum.boundaries.validate_stage(boundary_stage)
        positive = self._gradient_action(
            values,
            viscosity,
            self.homogeneous_boundary_stage,
            homogeneous=True,
        )
        zero = tuple(jnp.zeros_like(value) for value in values)
        boundary_affine = self._gradient_action(zero, viscosity, stage, homogeneous=False)
        affine_positive = tuple(
            homogeneous + affine
            for homogeneous, affine in zip(positive, boundary_affine, strict=True)
        )
        physical_rate = tuple(-value for value in affine_positive)
        homogeneous_velocity = self.momentum.boundaries.homogeneous_rate(values)
        enforced_velocity = self.momentum.boundaries.enforce(values, stage)
        space = self.momentum.operators.velocity_space
        positive_work = jnp.real(space.inner(homogeneous_velocity, positive))
        homogeneous_dissipation = 2.0 * self._potential(
            homogeneous_velocity,
            viscosity,
            self.homogeneous_boundary_stage,
            True,
        )
        dissipation = 2.0 * self._potential(values, viscosity, stage, False)
        work = jnp.real(space.inner(enforced_velocity, physical_rate))
        boundary_power = work + dissipation
        defect = jnp.abs(positive_work - homogeneous_dissipation)
        row_sum = self.operator_row_sum_bound(viscosity)
        step_bound = jnp.where(row_sum > 0.0, 2.0 / row_sum, jnp.inf)
        action_finite = jnp.all(
            jnp.stack(
                tuple(
                    jnp.all(jnp.isfinite(component))
                    for block in (positive, physical_rate, boundary_affine)
                    for component in block
                )
            )
        )
        scalars = jnp.stack((dissipation, work, boundary_power, positive_work, defect))
        finite = stage.finite & action_finite & jnp.all(jnp.isfinite(scalars))
        scale = jnp.maximum(1.0, jnp.max(jnp.abs(scalars)))
        tolerance = 4096.0 * jnp.finfo(scalars.dtype).eps * scale
        successful = (
            stage.successful
            & finite
            & (dissipation >= -tolerance)
            & (positive_work >= -tolerance)
            & (defect <= tolerance)
        )
        return MACVariationalViscosityResult(
            positive_operator_action=positive,
            physical_diffusive_rate=physical_rate,
            boundary_affine_action=boundary_affine,
            integrated_dissipation=self.momentum.precision.reduction(dissipation),
            integrated_work=self.momentum.precision.reduction(work),
            boundary_power=self.momentum.precision.reduction(boundary_power),
            positive_work=self.momentum.precision.reduction(positive_work),
            variational_defect=self.momentum.precision.reduction(defect),
            operator_row_sum_bound=self.momentum.precision.reduction(row_sum),
            explicit_step_bound=self.momentum.precision.reduction(step_bound),
            restriction_supported=self.restriction_supported,
            finite=finite,
            successful=successful,
            action_id=self.action_id,
            boundary_stage_id=stage.stage_id,
        )

    def freeze(
        self,
        cell_viscosity: ArrayLike,
        boundary_stage: MACBoundaryStageData,
        /,
    ) -> FrozenMACVariationalViscosityAction:
        """Bind dynamic stage leaves without rebuilding geometric machinery."""
        return FrozenMACVariationalViscosityAction(self, cell_viscosity, boundary_stage)


class FrozenMACVariationalViscosityAction(StrictModule, NonTrainableState):
    """Dynamic coefficient and boundary binding of one prepared action."""

    prepared_action: PreparedMACVariationalViscosityAction
    cell_viscosity: Array
    boundary_stage: MACBoundaryStageData
    frozen_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared_action: PreparedMACVariationalViscosityAction,
        cell_viscosity: ArrayLike,
        boundary_stage: MACBoundaryStageData,
        /,
    ):
        if not isinstance(prepared_action, PreparedMACVariationalViscosityAction):
            raise TypeError(
                "prepared_action must be PreparedMACVariationalViscosityAction."
            )
        self.prepared_action = prepared_action
        self.cell_viscosity = prepared_action._viscosity(cell_viscosity)
        self.boundary_stage = prepared_action.momentum.boundaries.validate_stage(
            boundary_stage
        )
        self.frozen_id = canonical_fingerprint(
            {
                "kind": "frozen-mac-variational-viscosity-action",
                "action": prepared_action.action_id,
                "boundary_stage": boundary_stage.stage_id,
            }
        )

    def positive_operator_action(self, velocity: FaceVelocity, /) -> FaceVelocity:
        return self.prepared_action.positive_operator_action(
            velocity, self.cell_viscosity
        )

    def boundary_affine_action(self, /) -> FaceVelocity:
        return self.prepared_action.boundary_affine_action(
            self.cell_viscosity, self.boundary_stage
        )

    def affine_positive_action(self, velocity: FaceVelocity, /) -> FaceVelocity:
        return self.prepared_action.affine_positive_action(
            velocity, self.cell_viscosity, self.boundary_stage
        )

    def physical_diffusive_rate(self, velocity: FaceVelocity, /) -> FaceVelocity:
        return self.prepared_action.physical_diffusive_rate(
            velocity, self.cell_viscosity, self.boundary_stage
        )

    def evaluate(self, velocity: FaceVelocity, /) -> MACVariationalViscosityResult:
        return self.prepared_action.evaluate(
            velocity, self.cell_viscosity, self.boundary_stage
        )


__all__ = [
    "FrozenMACVariationalViscosityAction",
    "MACVariationalViscosityResult",
    "PreparedMACVariationalViscosityAction",
]
