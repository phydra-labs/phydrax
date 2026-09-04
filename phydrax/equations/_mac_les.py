#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume import (
    FaceVelocity,
    MACBoundaryStageData,
    MACVariationalViscosityResult,
    PreparedMACMomentumOperators,
    PreparedMACVariationalViscosityAction,
)
from ._les_closures import (
    AlgebraicLESInputs,
    AlgebraicLESResult,
    LESFilterScale,
    PreparedAlgebraicLESModel,
)


_MAC_LES_REGIME = "incompressible-unit-density"


def _axis_values(values: Array, dimension: int, axis: int, /) -> Array:
    shape = [1] * dimension
    shape[axis] = int(values.size)
    return values.reshape(tuple(shape))


def _cell_centered_component(
    value: Array, component_axis: int, periodic: bool, /
) -> Array:
    moved = jnp.moveaxis(value, component_axis, 0)
    centered = (
        0.5 * (moved + jnp.roll(moved, -1, axis=0))
        if periodic
        else 0.5 * (moved[:-1] + moved[1:])
    )
    return jnp.moveaxis(centered, 0, component_axis)


def _periodic_center_derivative(
    value: Array,
    coordinates: Array,
    period: Array,
    axis: int,
    /,
) -> Array:
    moved = jnp.moveaxis(value, axis, 0)
    previous = jnp.roll(moved, 1, axis=0)
    following = jnp.roll(moved, -1, axis=0)
    previous_coordinates = jnp.roll(coordinates, 1).at[0].add(-period)
    following_coordinates = jnp.roll(coordinates, -1).at[-1].add(period)
    backward = coordinates - previous_coordinates
    forward = following_coordinates - coordinates
    shape = (coordinates.size,) + (1,) * (moved.ndim - 1)
    backward_ = backward.reshape(shape)
    forward_ = forward.reshape(shape)
    span = backward_ + forward_
    derivative = (
        -forward_ * previous / (backward_ * span)
        + (forward_ - backward_) * moved / (backward_ * forward_)
        + backward_ * following / (forward_ * span)
    )
    return jnp.moveaxis(derivative, 0, axis)


def _wall_center_derivative(
    value: Array,
    coordinates: Array,
    lower_bound: Array,
    upper_bound: Array,
    axis: int,
    /,
) -> Array:
    moved = jnp.moveaxis(value, axis, 0)
    count = int(coordinates.size)
    if count == 1:
        return jnp.zeros_like(value)
    if count == 2:
        lower_distance = coordinates - lower_bound
        upper_distance = upper_bound - coordinates
        lower = (
            2.0
            * lower_distance[0]
            * (moved[1] - moved[0])
            / (lower_distance[1] ** 2 - lower_distance[0] ** 2)
        )
        upper = (
            2.0
            * upper_distance[1]
            * (moved[-1] - moved[-2])
            / (upper_distance[0] ** 2 - upper_distance[1] ** 2)
        )
        return jnp.moveaxis(jnp.stack((lower, upper)), 0, axis)

    previous = moved[:-2]
    center = moved[1:-1]
    following = moved[2:]
    backward = coordinates[1:-1] - coordinates[:-2]
    forward = coordinates[2:] - coordinates[1:-1]
    shape = (count - 2,) + (1,) * (moved.ndim - 1)
    backward_ = backward.reshape(shape)
    forward_ = forward.reshape(shape)
    span = backward_ + forward_
    interior = (
        -forward_ * previous / (backward_ * span)
        + (forward_ - backward_) * center / (backward_ * forward_)
        + backward_ * following / (forward_ * span)
    )
    lower_distance = coordinates[:2] - lower_bound
    lower = (
        2.0
        * lower_distance[0]
        * (moved[1] - moved[0])
        / (lower_distance[1] ** 2 - lower_distance[0] ** 2)
    )
    upper_distance = upper_bound - coordinates[-2:]
    upper = (
        2.0
        * upper_distance[1]
        * (moved[-1] - moved[-2])
        / (upper_distance[0] ** 2 - upper_distance[1] ** 2)
    )
    derivative = jnp.concatenate((lower[None], interior, upper[None]), axis=0)
    return jnp.moveaxis(derivative, 0, axis)


class MACLESStageResult(StrictModule):
    """Pre-projection algebraic LES closure and variational MAC evidence."""

    velocity_gradient: Array
    strain: Array
    filter_scale: LESFilterScale
    model_result: AlgebraicLESResult
    viscosity_result: MACVariationalViscosityResult
    physical_rate: FaceVelocity
    integrated_work: Array
    boundary_power: Array
    finite: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)
    boundary_stage_id: str = eqx.field(static=True)


class MACAlgebraicLESPlan(StrictModule, NonTrainableState):
    """Bind one provenance-complete algebraic model to explicit MAC momentum."""

    prepared_model: PreparedAlgebraicLESModel
    plan_id: str = eqx.field(static=True)

    def __init__(self, prepared_model: PreparedAlgebraicLESModel, /):
        if not isinstance(prepared_model, PreparedAlgebraicLESModel):
            raise TypeError("prepared_model must be PreparedAlgebraicLESModel.")
        self.prepared_model = prepared_model
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-algebraic-les-plan",
                "model": prepared_model.prepared_id,
                "regime": _MAC_LES_REGIME,
                "density": "unit",
                "dimension": 3,
            }
        )

    def prepare(
        self, momentum: PreparedMACMomentumOperators, /
    ) -> PreparedMACAlgebraicLES:
        return PreparedMACAlgebraicLES(self, momentum)


class PreparedMACAlgebraicLES(StrictModule, NonTrainableState):
    """Factored-grid MAC LES preparation with runtime strain and viscosity."""

    plan: MACAlgebraicLESPlan
    momentum: PreparedMACMomentumOperators
    model: PreparedAlgebraicLESModel
    filter_axis_widths: tuple[Array, Array, Array]
    viscosity_action: PreparedMACVariationalViscosityAction
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: MACAlgebraicLESPlan,
        momentum: PreparedMACMomentumOperators,
        /,
    ):
        if not isinstance(plan, MACAlgebraicLESPlan):
            raise TypeError("plan must be MACAlgebraicLESPlan.")
        if not isinstance(momentum, PreparedMACMomentumOperators):
            raise TypeError("momentum must be PreparedMACMomentumOperators.")
        if momentum.dimension != 3:
            raise ValueError("MAC algebraic LES requires a three-dimensional grid.")
        grid = momentum.operators.discretization.grid
        unsupported = tuple(
            side.kind
            for side in momentum.boundaries.sides
            if side.kind not in ("free-slip", "symmetry")
        )
        if unsupported:
            raise ValueError(
                "Active MAC LES supports only periodic, free-slip, and symmetry "
                "impermeable boundaries; no-slip, open, inflow, and other boundary "
                "kinds are unsupported."
            )

        provenance = plan.prepared_model.provenance
        resolved_filter = provenance.resolved_filter
        expected_boundary_class = (
            "periodic"
            if all(axis.periodic for axis in grid.structured_axes)
            else "wall-bounded"
        )
        if (
            resolved_filter.family != "implicit-grid-volume"
            or resolved_filter.axis_names != grid.axis_names
            or resolved_filter.topology != "tensor-product"
            or resolved_filter.boundary_class != expected_boundary_class
            or resolved_filter.scale_rule != "volume-equivalent"
            or resolved_filter.commutation_status != "unmodeled"
            or resolved_filter.repeated_filter_semantics != "unmodeled"
        ):
            raise ValueError(
                "Prepared LES filter semantics do not match the structured MAC "
                "implicit grid-volume filter."
            )
        discretization = momentum.operators.discretization
        if provenance.discretization_id != discretization.prepared_id:
            raise ValueError(
                "Prepared LES provenance does not match the MAC discretization."
            )
        if provenance.regime != _MAC_LES_REGIME:
            raise ValueError(
                "MAC algebraic LES requires the 'incompressible-unit-density' regime."
            )

        widths = tuple(axis.interval_widths for axis in grid.structured_axes)
        action = PreparedMACVariationalViscosityAction(momentum)
        self.plan = plan
        self.momentum = momentum
        self.model = plan.prepared_model
        self.filter_axis_widths = (widths[0], widths[1], widths[2])
        self.viscosity_action = action
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-mac-algebraic-les",
                "plan": plan.plan_id,
                "model": plan.prepared_model.prepared_id,
                "momentum": momentum.prepared_id,
                "discretization": discretization.prepared_id,
                "filter": resolved_filter.filter_id,
                "filter_scale": "factored-cell-axis-widths",
                "viscosity_action": action.action_id,
            }
        )

    def filter_scale(self, /) -> LESFilterScale:
        """Construct the local directional widths from retained one-dimensional factors."""
        cell_shape = self.momentum.operators.discretization.cell_shape
        dimension = len(cell_shape)
        directional = jnp.stack(
            tuple(
                jnp.broadcast_to(_axis_values(width, dimension, axis), cell_shape)
                for axis, width in enumerate(self.filter_axis_widths)
            ),
            axis=-1,
        )
        return LESFilterScale(directional)

    def velocity_gradient(self, velocity: FaceVelocity, /) -> Array:
        """Evaluate the resolved physical velocity gradient at MAC cell centers."""
        return _mac_velocity_gradient(self.momentum, velocity)

    def _evaluate_model(
        self, velocity: FaceVelocity, /
    ) -> tuple[Array, LESFilterScale, AlgebraicLESResult]:
        gradient = self.velocity_gradient(velocity)
        scale = self.filter_scale()
        result = self.model.evaluate(AlgebraicLESInputs(gradient, scale))
        return gradient, scale, result

    def step_restriction(
        self,
        velocity: FaceVelocity,
        boundary_stage: MACBoundaryStageData,
        /,
    ) -> tuple[Array, bool]:
        """Return the current-state SGS explicit bound and its certification status."""
        stage = self.momentum.boundaries.validate_stage(boundary_stage)
        values = self.momentum.boundaries.enforce(
            self.momentum.operators.validate_velocity(velocity), stage
        )
        _, _, result = self._evaluate_model(values)
        return (
            self.viscosity_action.explicit_step_bound(result.kinematic_viscosity),
            self.viscosity_action.restriction_supported,
        )

    def evaluate(
        self,
        velocity: FaceVelocity,
        boundary_stage: MACBoundaryStageData,
        /,
    ) -> MACLESStageResult:
        """Evaluate the current-state closure before pressure projection."""
        stage = self.momentum.boundaries.validate_stage(boundary_stage)
        values = self.momentum.boundaries.enforce(
            self.momentum.operators.validate_velocity(velocity), stage
        )
        gradient, scale, model_result = self._evaluate_model(values)
        return _realize_mac_les_stage(
            self.viscosity_action,
            values,
            stage,
            gradient,
            scale,
            model_result,
            prepared_id=self.prepared_id,
        )


def _mac_velocity_gradient(
    momentum: PreparedMACMomentumOperators,
    velocity: FaceVelocity,
    /,
) -> Array:
    """Evaluate a face velocity gradient at the variational action's cells."""
    values = momentum.operators.validate_velocity(velocity)
    grid = momentum.operators.discretization.grid
    centered = tuple(
        _cell_centered_component(value, axis, grid.structured_axes[axis].periodic)
        for axis, value in enumerate(values)
    )
    rows = []
    for component_axis, (face_value, cell_value) in enumerate(
        zip(values, centered, strict=True)
    ):
        derivatives = []
        for derivative_axis, axis in enumerate(grid.structured_axes):
            if derivative_axis == component_axis:
                moved = jnp.moveaxis(face_value, derivative_axis, 0)
                difference = (
                    jnp.roll(moved, -1, axis=0) - moved
                    if axis.periodic
                    else moved[1:] - moved[:-1]
                )
                derivative = jnp.moveaxis(
                    difference / _axis_values(axis.interval_widths, moved.ndim, 0),
                    0,
                    derivative_axis,
                )
            elif axis.periodic:
                derivative = _periodic_center_derivative(
                    cell_value,
                    axis.interval_centers,
                    axis.bounds[1] - axis.bounds[0],
                    derivative_axis,
                )
            else:
                derivative = _wall_center_derivative(
                    cell_value,
                    axis.interval_centers,
                    axis.bounds[0],
                    axis.bounds[1],
                    derivative_axis,
                )
            derivatives.append(derivative)
        rows.append(jnp.stack(tuple(derivatives), axis=-1))
    return jnp.stack(tuple(rows), axis=-2)


def _periodic_uniform_mac_stress_rate(
    momentum: PreparedMACMomentumOperators,
    specific_stress: Array,
    /,
) -> FaceVelocity:
    """Return the conservative ``-div(tau)`` action on periodic MAC faces.

    The discrete action is the inverse-Riesz adjoint of the backend velocity
    gradient. This makes its face-space work exactly equal to the cell-volume
    stress/gradient contraction without materializing an incompatible
    cell-to-edge stress layout.
    """
    stress = jnp.asarray(specific_stress)
    cell_shape = momentum.operators.discretization.cell_shape
    if stress.shape != cell_shape + (3, 3):
        raise ValueError("MAC learned stress must have cell_shape + (3, 3).")
    zero = tuple(
        jnp.zeros(layout.shape, dtype=stress.dtype)
        for layout in momentum.operators.discretization.face_layouts
    )
    volumes = momentum.operators.discretization.cell_volumes.astype(stress.dtype)

    def stress_work(velocity: FaceVelocity, /) -> Array:
        gradient = _mac_velocity_gradient(momentum, velocity)
        return jnp.sum(volumes[..., None, None] * stress * gradient)

    covector = jax.grad(stress_work)(zero)
    rate = momentum.operators.velocity_space.inverse_riesz(covector)
    return momentum.boundaries.homogeneous_rate(tuple(rate))


def _realize_mac_les_stage(
    viscosity_action: PreparedMACVariationalViscosityAction,
    velocity: FaceVelocity,
    boundary_stage: MACBoundaryStageData,
    gradient: Array,
    scale: LESFilterScale,
    model_result: AlgebraicLESResult,
    /,
    *,
    prepared_id: str,
) -> MACLESStageResult:
    """Realize one already evaluated LES viscosity on the prepared MAC action."""
    strain = 0.5 * (gradient + jnp.swapaxes(gradient, -1, -2))
    viscosity_result = viscosity_action.evaluate(
        velocity, model_result.kinematic_viscosity, boundary_stage
    )
    model_finite = (
        jnp.all(jnp.isfinite(gradient))
        & jnp.all(jnp.isfinite(strain))
        & jnp.all(jnp.isfinite(scale.directional_widths))
        & jnp.all(jnp.isfinite(model_result.kinematic_viscosity))
        & jnp.all(jnp.isfinite(model_result.specific_deviatoric_stress))
        & jnp.all(jnp.isfinite(model_result.energy_transfer))
    )
    finite = boundary_stage.finite & model_finite & viscosity_result.finite
    successful = boundary_stage.successful & finite & viscosity_result.successful
    return MACLESStageResult(
        velocity_gradient=gradient,
        strain=strain,
        filter_scale=scale,
        model_result=model_result,
        viscosity_result=viscosity_result,
        physical_rate=viscosity_result.physical_diffusive_rate,
        integrated_work=viscosity_result.integrated_work,
        boundary_power=viscosity_result.boundary_power,
        finite=finite,
        successful=successful,
        prepared_id=prepared_id,
        boundary_stage_id=boundary_stage.stage_id,
    )


__all__ = [
    "MACAlgebraicLESPlan",
    "MACLESStageResult",
    "PreparedMACAlgebraicLES",
]
