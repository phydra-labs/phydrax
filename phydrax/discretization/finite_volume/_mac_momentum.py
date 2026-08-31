#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._incompressible import FaceVelocity, PreparedMACOperators
from ._mac_boundary import (
    MACBoundaryPlan,
    MACBoundaryStageData,
    PreparedMACBoundaryPlan,
)
from ._precision import FiniteVolumePrecisionPolicy


def _axis_shape(value: Array, axis: int, count: int, /) -> tuple[int, ...]:
    shape = list(value.shape)
    shape[axis] = count
    return tuple(shape)


def _axis_values(values: Array, dimension: int, axis: int, /) -> Array:
    shape = [1] * dimension
    shape[axis] = int(values.size)
    return values.reshape(tuple(shape))


def _face_interpolate(
    value: Array,
    axis: int,
    periodic: bool,
    lower: Array,
    upper: Array,
    /,
) -> Array:
    moved = jnp.moveaxis(value, axis, 0)
    if periodic:
        faces = 0.5 * (moved + jnp.roll(moved, 1, axis=0))
    else:
        lower_face = jnp.expand_dims(jnp.broadcast_to(lower, moved.shape[1:]), axis=0)
        upper_face = jnp.expand_dims(jnp.broadcast_to(upper, moved.shape[1:]), axis=0)
        interior = 0.5 * (moved[:-1] + moved[1:])
        faces = jnp.concatenate((lower_face, interior, upper_face), axis=0)
    return jnp.moveaxis(faces, 0, axis)


def _patch_to_component_faces(
    value: Array,
    boundary_axis: int,
    component_axis: int,
    periodic: bool,
    /,
) -> Array:
    local_axis = component_axis if component_axis < boundary_axis else component_axis - 1
    moved = jnp.moveaxis(value, local_axis, 0)
    if periodic:
        faces = 0.5 * (moved + jnp.roll(moved, 1, axis=0))
    else:
        interior = 0.5 * (moved[:-1] + moved[1:])
        faces = jnp.concatenate((moved[:1], interior, moved[-1:]), axis=0)
    return jnp.moveaxis(faces, 0, local_axis)


def _center_interpolate(value: Array, axis: int, periodic: bool, /) -> Array:
    moved = jnp.moveaxis(value, axis, 0)
    centers = (
        0.5 * (moved + jnp.roll(moved, -1, axis=0))
        if periodic
        else 0.5 * (moved[:-1] + moved[1:])
    )
    return jnp.moveaxis(centers, 0, axis)


def _set_axis_boundary(value: Array, axis: int, index: int, target: Array, /) -> Array:
    location = [slice(None)] * value.ndim
    location[axis] = index
    return value.at[tuple(location)].set(target)


def _axis_boundary(value: Array, axis: int, index: int, /) -> Array:
    location = [slice(None)] * value.ndim
    location[axis] = index
    return value[tuple(location)]


class MACMomentumReport(StrictModule, NonTrainableState):
    """Prepared weighted skew, diffusion symmetry, and dissipation evidence."""

    weighted_skew_residual: Array
    diffusion_symmetry_residual: Array
    homogeneous_diffusion_rate: Array
    finite: Array
    passed: Array
    report_id: str = eqx.field(static=True)


class MACMomentumDiagnostics(StrictModule, NonTrainableState):
    kinetic_energy: Array
    nonlinear_energy_rate: Array
    viscous_energy_rate: Array
    dissipation: Array
    boundary_power: Array
    open_backflow_dissipation: Array
    integrated_mass_flux: Array
    boundary_defect: Array
    finite: Array
    successful: Array
    momentum_id: str = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)


class MACMomentumPlan(StrictModule, NonTrainableState):
    """Prepare symmetry-preserving MAC momentum transport and diffusion."""

    operators: PreparedMACOperators
    boundaries: PreparedMACBoundaryPlan
    precision: FiniteVolumePrecisionPolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        /,
        *,
        boundaries: PreparedMACBoundaryPlan | MACBoundaryPlan | None = None,
        precision: FiniteVolumePrecisionPolicy | None = None,
    ):
        if not isinstance(operators, PreparedMACOperators):
            raise TypeError("operators must be PreparedMACOperators.")
        boundaries_ = (
            MACBoundaryPlan(operators).prepare()
            if boundaries is None
            else boundaries.prepare()
            if isinstance(boundaries, MACBoundaryPlan)
            else boundaries
        )
        if not isinstance(boundaries_, PreparedMACBoundaryPlan):
            raise TypeError(
                "boundaries must be PreparedMACBoundaryPlan, MACBoundaryPlan, or None."
            )
        if boundaries_.operators.prepared_id != operators.prepared_id:
            raise ValueError("MAC momentum boundaries must use the same operators.")
        precision_ = (
            FiniteVolumePrecisionPolicy(np.dtype(operators.pressure_space.dtype).name)
            if precision is None
            else precision
        )
        if not isinstance(precision_, FiniteVolumePrecisionPolicy):
            raise TypeError("precision must be FiniteVolumePrecisionPolicy or None.")
        self.operators = operators
        self.boundaries = boundaries_
        self.precision = precision_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-momentum-plan",
                "operators": operators.prepared_id,
                "boundaries": boundaries_.prepared_id,
                "precision": precision_.policy_id,
            }
        )

    def prepare(self, /) -> PreparedMACMomentumOperators:
        return PreparedMACMomentumOperators(self)


class PreparedMACMomentumOperators(StrictModule, NonTrainableState):
    """Prepared conservative, skew-adjoint transport on MAC face velocities."""

    operators: PreparedMACOperators
    boundaries: PreparedMACBoundaryPlan
    precision: FiniteVolumePrecisionPolicy
    face_dual_widths: tuple[Array, ...]
    report: MACMomentumReport
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: MACMomentumPlan, /):
        if not isinstance(plan, MACMomentumPlan):
            raise TypeError("plan must be MACMomentumPlan.")
        grid = plan.operators.discretization.grid
        dual_widths = []
        for axis in grid.structured_axes:
            centers = axis.interval_centers
            widths = axis.interval_widths
            if axis.periodic:
                period = axis.bounds[1] - axis.bounds[0]
                previous = jnp.roll(centers, 1).at[0].add(-period)
                distance = centers - previous
            elif centers.size == 1:
                distance = jnp.asarray(
                    (0.5 * widths[0], 0.5 * widths[0]), dtype=centers.dtype
                )
            else:
                distance = jnp.concatenate(
                    (0.5 * widths[:1], centers[1:] - centers[:-1], 0.5 * widths[-1:])
                )
            dual_widths.append(distance)
        identifier = canonical_fingerprint(
            {
                "kind": "prepared-mac-momentum",
                "plan": plan.plan_id,
                "dual_width_shapes": [list(value.shape) for value in dual_widths],
            }
        )
        self.operators = plan.operators
        self.boundaries = plan.boundaries
        self.precision = plan.precision
        self.face_dual_widths = tuple(dual_widths)
        self.prepared_id = identifier
        self.report = self._prepare_report(identifier)
        if not bool(self.report.passed):
            raise RuntimeError(
                "Prepared MAC momentum operators failed invariant evidence."
            )

    @property
    def dimension(self) -> int:
        return len(self.operators.discretization.cell_shape)

    def _stage(self, stage: MACBoundaryStageData | None, /) -> MACBoundaryStageData:
        return (
            self.boundaries.evaluate(jnp.asarray(0.0), None)
            if stage is None
            else self.boundaries.validate_stage(stage)
        )

    def _transport_face(
        self,
        transport: FaceVelocity,
        component_axis: int,
        derivative_axis: int,
        stage: MACBoundaryStageData,
        /,
    ) -> Array:
        grid = self.operators.discretization.grid
        if derivative_axis == component_axis:
            return _center_interpolate(
                transport[derivative_axis],
                derivative_axis,
                grid.structured_axes[derivative_axis].periodic,
            )
        component_grid_axis = grid.structured_axes[component_axis]
        if component_grid_axis.periodic:
            zero = jnp.asarray(0.0, dtype=transport[derivative_axis].dtype)
            return _face_interpolate(
                transport[derivative_axis], component_axis, True, zero, zero
            )
        component_periodic = grid.structured_axes[derivative_axis].periodic
        lower = (
            _patch_to_component_faces(
                self.boundaries.tangential_value(
                    component_axis,
                    "lower",
                    derivative_axis,
                    stage,
                    homogeneous=False,
                ),
                component_axis,
                derivative_axis,
                component_periodic,
            )
            if self.boundaries.tangential_dirichlet(component_axis, "lower")
            else _axis_boundary(transport[derivative_axis], component_axis, 0)
        )
        upper = (
            _patch_to_component_faces(
                self.boundaries.tangential_value(
                    component_axis,
                    "upper",
                    derivative_axis,
                    stage,
                    homogeneous=False,
                ),
                component_axis,
                derivative_axis,
                component_periodic,
            )
            if self.boundaries.tangential_dirichlet(component_axis, "upper")
            else _axis_boundary(transport[derivative_axis], component_axis, -1)
        )
        return _face_interpolate(
            transport[derivative_axis],
            component_axis,
            False,
            lower,
            upper,
        )

    def _advected_face(
        self,
        advected: FaceVelocity,
        component_axis: int,
        derivative_axis: int,
        /,
    ) -> Array:
        grid_axis = self.operators.discretization.grid.structured_axes[derivative_axis]
        if derivative_axis == component_axis:
            return _center_interpolate(
                advected[component_axis], derivative_axis, grid_axis.periodic
            )
        zero = jnp.asarray(0.0, dtype=advected[component_axis].dtype)
        return _face_interpolate(
            advected[component_axis],
            derivative_axis,
            grid_axis.periodic,
            zero,
            zero,
        )

    def _dual_divergence(
        self,
        flux: Array,
        component_axis: int,
        derivative_axis: int,
        /,
    ) -> Array:
        grid = self.operators.discretization.grid
        grid_axis = grid.structured_axes[derivative_axis]
        moved = jnp.moveaxis(flux, derivative_axis, 0)
        if derivative_axis == component_axis:
            widths = _axis_values(
                self.face_dual_widths[component_axis], flux.ndim, derivative_axis
            )
            if grid_axis.periodic:
                difference = flux - jnp.roll(flux, 1, axis=derivative_axis)
                return difference / widths
            interior = moved[1:] - moved[:-1]
            result = jnp.zeros(
                _axis_shape(
                    flux, derivative_axis, int(grid_axis.interval_widths.size) + 1
                ),
                dtype=flux.dtype,
            )
            interior_widths = self.face_dual_widths[component_axis][1:-1]
            interior = interior / _axis_values(interior_widths, moved.ndim, 0)
            moved_result = jnp.moveaxis(result, derivative_axis, 0)
            moved_result = moved_result.at[1:-1].set(interior)
            return jnp.moveaxis(moved_result, 0, derivative_axis)
        difference = (
            jnp.roll(flux, -1, axis=derivative_axis) - flux
            if grid_axis.periodic
            else jnp.moveaxis(moved[1:] - moved[:-1], 0, derivative_axis)
        )
        widths = _axis_values(grid_axis.interval_widths, flux.ndim, derivative_axis)
        return difference / widths

    def conservative_transport(
        self,
        transport: FaceVelocity,
        advected: FaceVelocity,
        /,
        stage: MACBoundaryStageData | None = None,
    ) -> FaceVelocity:
        stage_ = self._stage(stage)
        transport_ = self.boundaries.enforce(transport, stage_)
        advected_ = self.boundaries.homogeneous_rate(advected)
        output = []
        for component_axis in range(self.dimension):
            rate = jnp.zeros_like(
                advected_[component_axis],
                dtype=jnp.dtype(self.precision.reduction_dtype),
            )
            for derivative_axis in range(self.dimension):
                transport_face = self.precision.flux(
                    self._transport_face(
                        transport_, component_axis, derivative_axis, stage_
                    )
                )
                advected_face = self.precision.flux(
                    self._advected_face(advected_, component_axis, derivative_axis)
                )
                rate = self.precision.reduction(
                    rate
                    + self._dual_divergence(
                        transport_face * advected_face,
                        component_axis,
                        derivative_axis,
                    )
                )
            output.append(self.precision.storage(rate))
        return self.boundaries.homogeneous_rate(tuple(output))

    def advection(
        self,
        transport: FaceVelocity,
        advected: FaceVelocity,
        /,
        stage: MACBoundaryStageData | None = None,
    ) -> FaceVelocity:
        """Return the weighted skew part of conservative momentum transport."""
        stage_ = self._stage(stage)
        transport_ = self.boundaries.enforce(transport, stage_)
        advected_ = self.boundaries.homogeneous_rate(advected)

        def action(value):
            return self.conservative_transport(transport_, value, stage=stage_)

        conservative = action(advected_)
        covector = self.operators.velocity_space.riesz(advected_)
        zero = tuple(jnp.zeros_like(value) for value in advected_)
        transpose = jax.linear_transpose(action, zero)
        euclidean_adjoint = transpose(covector)[0]
        hilbert_adjoint = self.operators.velocity_space.inverse_riesz(euclidean_adjoint)
        skew = tuple(
            self.precision.storage(0.5 * (direct - adjoint))
            for direct, adjoint in zip(conservative, hilbert_adjoint, strict=True)
        )
        return self.boundaries.homogeneous_rate(skew)

    def _open_backflow(
        self,
        velocity: FaceVelocity,
        stage: MACBoundaryStageData,
        /,
    ) -> tuple[FaceVelocity, Array]:
        values = self.operators.validate_velocity(velocity)
        stabilization = [jnp.zeros_like(value) for value in values]
        for boundary, axis, side_index in zip(
            self.boundaries.sides,
            self.boundaries.side_axes,
            self.boundaries.side_indices,
            strict=True,
        ):
            if boundary.kind != "traction-open" or boundary.backflow_coefficient == 0.0:
                continue
            trace = _axis_boundary(values[axis], axis, side_index)
            outward_sign = -1.0 if side_index == 0 else 1.0
            backflow_speed = jnp.maximum(-outward_sign * trace, 0.0)
            dual_width = self.face_dual_widths[axis][side_index]
            contribution = (
                boundary.backflow_coefficient * backflow_speed * trace / dual_width
            )
            current = _axis_boundary(stabilization[axis], axis, side_index)
            stabilization[axis] = _set_axis_boundary(
                stabilization[axis],
                axis,
                side_index,
                current + contribution,
            )
        stabilized = self.boundaries.homogeneous_rate(tuple(stabilization))
        dissipation = jnp.real(self.operators.velocity_space.inner(values, stabilized))
        return stabilized, self.precision.reduction(jnp.maximum(dissipation, 0.0))

    def convection(
        self,
        velocity: FaceVelocity,
        /,
        stage: MACBoundaryStageData | None = None,
    ) -> FaceVelocity:
        stage_ = self._stage(stage)
        value = self.boundaries.enforce(velocity, stage_)
        skew = self.advection(value, value, stage=stage_)
        stabilization, _ = self._open_backflow(value, stage_)
        return tuple(
            direct + stabilized
            for direct, stabilized in zip(skew, stabilization, strict=True)
        )

    def _laplacian_component(
        self,
        value: Array,
        component_axis: int,
        stage: MACBoundaryStageData,
        /,
        *,
        homogeneous: bool,
    ) -> Array:
        grid = self.operators.discretization.grid
        result = jnp.zeros_like(value, dtype=jnp.dtype(self.precision.reduction_dtype))
        for derivative_axis, axis in enumerate(grid.structured_axes):
            moved = jnp.moveaxis(self.precision.reconstruction(value), derivative_axis, 0)
            if derivative_axis == component_axis:
                widths = axis.interval_widths
                if axis.periodic:
                    gradient = (jnp.roll(moved, -1, axis=0) - moved) / _axis_values(
                        widths, moved.ndim, 0
                    )
                    contribution = (
                        gradient - jnp.roll(gradient, 1, axis=0)
                    ) / _axis_values(self.face_dual_widths[component_axis], moved.ndim, 0)
                else:
                    gradient = (moved[1:] - moved[:-1]) / _axis_values(
                        widths, moved.ndim, 0
                    )
                    interior = (gradient[1:] - gradient[:-1]) / _axis_values(
                        self.face_dual_widths[component_axis][1:-1],
                        moved.ndim,
                        0,
                    )
                    contribution = jnp.zeros_like(moved)
                    contribution = contribution.at[1:-1].set(interior)
            else:
                centers = axis.interval_centers
                widths = axis.interval_widths
                if axis.periodic:
                    period = axis.bounds[1] - axis.bounds[0]
                    previous_centers = jnp.roll(centers, 1).at[0].add(-period)
                    distance = centers - previous_centers
                    gradient = (moved - jnp.roll(moved, 1, axis=0)) / _axis_values(
                        distance, moved.ndim, 0
                    )
                    contribution = (
                        jnp.roll(gradient, -1, axis=0) - gradient
                    ) / _axis_values(widths, moved.ndim, 0)
                else:
                    component_periodic = grid.structured_axes[component_axis].periodic
                    if self.boundaries.tangential_dirichlet(derivative_axis, "lower"):
                        lower_patch = self.boundaries.tangential_value(
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
                        lower_gradient = (moved[:1] - lower) / (
                            centers[0] - axis.bounds[0]
                        )
                    else:
                        lower_gradient = jnp.zeros_like(moved[:1])
                    interior_gradient = (moved[1:] - moved[:-1]) / _axis_values(
                        centers[1:] - centers[:-1], moved.ndim, 0
                    )
                    if self.boundaries.tangential_dirichlet(derivative_axis, "upper"):
                        upper_patch = self.boundaries.tangential_value(
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
                        upper_gradient = (upper - moved[-1:]) / (
                            axis.bounds[1] - centers[-1]
                        )
                    else:
                        upper_gradient = jnp.zeros_like(moved[-1:])
                    gradient = jnp.concatenate(
                        (lower_gradient, interior_gradient, upper_gradient), axis=0
                    )
                    contribution = (gradient[1:] - gradient[:-1]) / _axis_values(
                        widths, moved.ndim, 0
                    )
            result = self.precision.reduction(
                result + jnp.moveaxis(contribution, 0, derivative_axis)
            )
        return self.precision.storage(result)

    def homogeneous_laplacian(
        self,
        velocity: FaceVelocity,
        /,
        stage: MACBoundaryStageData | None = None,
    ) -> FaceVelocity:
        stage_ = self._stage(stage)
        values = self.boundaries.homogeneous_rate(velocity)
        result = tuple(
            self._laplacian_component(value, component_axis, stage_, homogeneous=True)
            for component_axis, value in enumerate(values)
        )
        return self.boundaries.homogeneous_rate(result)

    def laplacian(
        self,
        velocity: FaceVelocity,
        /,
        stage: MACBoundaryStageData | None = None,
    ) -> FaceVelocity:
        stage_ = self._stage(stage)
        values = self.boundaries.enforce(velocity, stage_)
        result = tuple(
            self._laplacian_component(value, component_axis, stage_, homogeneous=False)
            for component_axis, value in enumerate(values)
        )
        return self.boundaries.homogeneous_rate(result)

    def _boundary_traction_power(
        self,
        velocity: FaceVelocity,
        stage: MACBoundaryStageData,
        /,
    ) -> Array:
        grid = self.operators.discretization.grid
        power = jnp.asarray(0.0, dtype=self.operators.pressure_space.dtype)
        for position, (boundary, axis, side_index) in enumerate(
            zip(
                self.boundaries.sides,
                self.boundaries.side_axes,
                self.boundaries.side_indices,
                strict=True,
            )
        ):
            if boundary.kind not in ("pressure-outlet", "traction-open"):
                continue
            measure = _axis_boundary(
                self.operators.discretization.face_measures[axis], axis, side_index
            )
            outward_sign = -1.0 if side_index == 0 else 1.0
            if boundary.kind == "pressure-outlet":
                pressure = stage.values[position]
                normal_velocity = _axis_boundary(velocity[axis], axis, side_index)
                power = power - jnp.sum(
                    measure * pressure * outward_sign * normal_velocity
                )
                continue
            traction = stage.values[position]
            local_power = jnp.zeros_like(measure)
            for component_axis in range(self.dimension):
                if component_axis == axis:
                    trace = _axis_boundary(velocity[component_axis], axis, side_index)
                else:
                    centered = _center_interpolate(
                        velocity[component_axis],
                        component_axis,
                        grid.structured_axes[component_axis].periodic,
                    )
                    trace = _axis_boundary(centered, axis, side_index)
                local_power = local_power + traction[component_axis] * trace
            power = power + jnp.sum(measure * local_power)
        return self.precision.reduction(power)

    def diagnostics(
        self,
        velocity: FaceVelocity,
        /,
        stage: MACBoundaryStageData | None = None,
    ) -> MACMomentumDiagnostics:
        stage_ = self._stage(stage)
        value = self.boundaries.enforce(velocity, stage_)
        convection = self.convection(value, stage=stage_)
        diffusion = self.laplacian(value, stage=stage_)
        homogeneous = self.homogeneous_laplacian(value, stage=stage_)
        _, backflow_dissipation = self._open_backflow(value, stage_)
        space = self.operators.velocity_space
        kinetic_energy = 0.5 * jnp.real(space.inner(value, value))
        nonlinear_rate = -jnp.real(space.inner(value, convection))
        viscous_rate = jnp.real(space.inner(value, diffusion))
        homogeneous_rate = jnp.real(space.inner(value, homogeneous))
        boundary_power = (
            viscous_rate - homogeneous_rate + self._boundary_traction_power(value, stage_)
        )
        integrated_mass_flux = self.boundaries.integrated_mass_flux(value)
        boundary_defect = self.boundaries.defect(value, stage_)
        evidence = jnp.stack(
            (
                kinetic_energy,
                nonlinear_rate,
                viscous_rate,
                homogeneous_rate,
                boundary_power,
                backflow_dissipation,
                integrated_mass_flux,
                boundary_defect,
            )
        )
        finite = stage_.finite & jnp.all(jnp.isfinite(evidence))
        successful = finite & stage_.successful & (backflow_dissipation >= 0.0)
        return MACMomentumDiagnostics(
            kinetic_energy=self.precision.reduction(kinetic_energy),
            nonlinear_energy_rate=self.precision.reduction(nonlinear_rate),
            viscous_energy_rate=self.precision.reduction(viscous_rate),
            dissipation=self.precision.reduction(-homogeneous_rate),
            boundary_power=self.precision.reduction(boundary_power),
            open_backflow_dissipation=self.precision.reduction(backflow_dissipation),
            integrated_mass_flux=self.precision.reduction(integrated_mass_flux),
            boundary_defect=self.precision.reduction(boundary_defect),
            finite=finite,
            successful=successful,
            momentum_id=self.prepared_id,
            boundary_id=self.boundaries.prepared_id,
        )

    def _probe(self, phase: float, /) -> FaceVelocity:
        values = []
        dtype = self.operators.pressure_space.dtype
        for axis, layout in enumerate(self.operators.discretization.face_layouts):
            count = int(np.prod(layout.shape))
            coordinates = jnp.arange(count, dtype=dtype).reshape(layout.shape)
            values.append(jnp.sin(0.37 * coordinates + phase + axis))
        return self.boundaries.homogeneous_rate(tuple(values))

    def _prepare_report(self, identifier: str, /) -> MACMomentumReport:
        stage = self.boundaries.homogeneous_stage()
        transport = self._probe(0.2)
        left = self._probe(0.7)
        right = self._probe(1.3)
        left_action = self.advection(transport, left, stage=stage)
        right_action = self.advection(transport, right, stage=stage)
        space = self.operators.velocity_space
        skew = jnp.abs(space.inner(right, left_action) + space.inner(right_action, left))
        left_diffusion = self.homogeneous_laplacian(left, stage=stage)
        right_diffusion = self.homogeneous_laplacian(right, stage=stage)
        diffusion_symmetry = jnp.abs(
            space.inner(right, left_diffusion) - space.inner(right_diffusion, left)
        )
        homogeneous_rate = jnp.real(space.inner(left, left_diffusion))
        scale = jnp.maximum(
            1.0,
            jnp.maximum(
                jnp.abs(space.inner(right, left_action)),
                jnp.abs(space.inner(right, left_diffusion)),
            ),
        )
        epsilon = jnp.finfo(jnp.dtype(self.precision.reduction_dtype)).eps
        tolerance = 4096.0 * epsilon * scale
        finite = (
            jnp.isfinite(skew)
            & jnp.isfinite(diffusion_symmetry)
            & jnp.isfinite(homogeneous_rate)
        )
        passed = (
            finite
            & (skew <= tolerance)
            & (diffusion_symmetry <= tolerance)
            & (homogeneous_rate <= tolerance)
        )
        return MACMomentumReport(
            weighted_skew_residual=self.precision.reduction(skew),
            diffusion_symmetry_residual=self.precision.reduction(diffusion_symmetry),
            homogeneous_diffusion_rate=self.precision.reduction(homogeneous_rate),
            finite=finite,
            passed=passed,
            report_id=canonical_fingerprint(
                {"kind": "mac-momentum-report", "momentum": identifier}
            ),
        )


__all__ = [
    "MACMomentumDiagnostics",
    "MACMomentumPlan",
    "MACMomentumReport",
    "PreparedMACMomentumOperators",
]
