#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._incompressible import FaceVelocity, PreparedMACOperators
from ._mac_boundary import MACBoundaryStageData
from ._mac_momentum import PreparedMACMomentumOperators


FaceMomentumFlux = tuple[tuple[Array, ...], ...]


def _axis_boundary(value: Array, axis: int, index: int, /) -> Array:
    location = [slice(None)] * value.ndim
    location[axis] = index
    return value[tuple(location)]


def _center_interpolate(value: Array, axis: int, periodic: bool, /) -> Array:
    moved = jnp.moveaxis(value, axis, 0)
    centered = (
        0.5 * (moved + jnp.roll(moved, -1, axis=0))
        if periodic
        else 0.5 * (moved[:-1] + moved[1:])
    )
    return jnp.moveaxis(centered, 0, axis)


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
        lower_face = jnp.broadcast_to(lower, moved[:1].shape)
        upper_face = jnp.broadcast_to(upper, moved[:1].shape)
        faces = jnp.concatenate(
            (lower_face, 0.5 * (moved[:-1] + moved[1:]), upper_face), axis=0
        )
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
        faces = jnp.concatenate(
            (moved[:1], 0.5 * (moved[:-1] + moved[1:]), moved[-1:]), axis=0
        )
    return jnp.moveaxis(faces, 0, local_axis)


def _maximum_abs(values: tuple[Array, ...], dtype: jnp.dtype, /) -> Array:
    if not values:
        return jnp.asarray(0.0, dtype=dtype)
    return jnp.max(jnp.stack(tuple(jnp.max(jnp.abs(value)) for value in values)))


class MACVariableDensityReport(StrictModule, NonTrainableState):
    """Prepared constant-density and constitutive identity evidence."""

    constant_face_density_residual: Array
    constant_velocity_residual: Array
    authoritative_mass_flux_residual: Array
    conservative_density_residual: Array
    positive_face_density: Array
    passed: Array
    report_id: str = eqx.field(static=True)


class MACDensityUpdateResult(StrictModule):
    """Fail-closed conservative donor-cell density update."""

    density: Array
    candidate_density: Array
    density_rate: Array
    mass_before: Array
    candidate_mass: Array
    accepted_mass: Array
    boundary_mass_flux: Array
    mass_balance_residual: Array
    minimum_candidate_density: Array
    positive: Array
    finite: Array
    successful: Array
    operators_id: str = eqx.field(static=True)


class MACVariableDensityTransportResult(StrictModule):
    """One conservative mass and staggered momentum transport evaluation."""

    density: Array
    face_momentum: FaceVelocity
    face_density: FaceVelocity
    face_inverse_density: FaceVelocity
    velocity: FaceVelocity
    mass_flux: FaceVelocity
    momentum_flux: FaceMomentumFlux
    density_rate: Array
    momentum_advection: FaceVelocity
    momentum_rate: FaceVelocity
    mass: Array
    mass_rate: Array
    boundary_mass_flux: Array
    mass_balance_residual: Array
    total_momentum: Array
    advective_momentum_rate: Array
    kinetic_energy: Array
    advective_kinetic_energy_rate: Array
    minimum_density: Array
    minimum_face_density: Array
    authoritative_mass_flux_residual: Array
    positive: Array
    finite: Array
    successful: Array
    operators_id: str = eqx.field(static=True)


class MACVariableDensityPlan(StrictModule, NonTrainableState):
    """Bind a donor-cell variable-density policy to prepared MAC momentum."""

    momentum: PreparedMACMomentumOperators
    plan_id: str = eqx.field(static=True)

    def __init__(self, momentum: PreparedMACMomentumOperators, /):
        if not isinstance(momentum, PreparedMACMomentumOperators):
            raise TypeError("momentum must be PreparedMACMomentumOperators.")
        unsupported = tuple(
            boundary.kind
            for boundary in momentum.boundaries.sides
            if boundary.kind not in ("no-slip", "free-slip", "symmetry")
        )
        if unsupported:
            raise ValueError(
                "Variable-density MAC currently requires impermeable boundaries; "
                f"got {unsupported!r}."
            )
        self.momentum = momentum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-variable-density-plan",
                "momentum": momentum.prepared_id,
                "density_flux": "donor-cell",
                "momentum_flux": "mass-flux-donor-velocity",
            }
        )

    def prepare(self, /) -> PreparedMACVariableDensityOperators:
        return PreparedMACVariableDensityOperators(self)


class PreparedMACVariableDensityOperators(StrictModule, NonTrainableState):
    """Positive density closure and shared conservative MAC mass/momentum fluxes."""

    operators: PreparedMACOperators
    momentum: PreparedMACMomentumOperators
    report: MACVariableDensityReport
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: MACVariableDensityPlan, /):
        if not isinstance(plan, MACVariableDensityPlan):
            raise TypeError("plan must be MACVariableDensityPlan.")
        identifier = canonical_fingerprint(
            {
                "kind": "prepared-mac-variable-density",
                "plan": plan.plan_id,
                "operators": plan.momentum.operators.prepared_id,
            }
        )
        self.operators = plan.momentum.operators
        self.momentum = plan.momentum
        self.prepared_id = identifier
        self.report = self._prepare_report(identifier)
        if not bool(self.report.passed):
            raise RuntimeError(
                "Prepared variable-density MAC operators failed identity evidence."
            )

    @property
    def dimension(self) -> int:
        return self.momentum.dimension

    def validate_density(self, density: ArrayLike, /) -> Array:
        value = self.operators.validate_pressure(density)
        return eqx.error_if(
            value,
            jnp.any(~jnp.isfinite(value) | (value <= 0.0)),
            "MAC cell density must be positive and finite.",
        )

    def validate_face_momentum(self, face_momentum: FaceVelocity, /) -> FaceVelocity:
        values = self.operators.validate_velocity(face_momentum)
        finite = jnp.all(
            jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in values))
        )
        boundary_defect = self.momentum.boundaries.defect(
            values, self.momentum.boundaries.homogeneous_stage()
        )
        dtype = self.operators.pressure_space.dtype
        tolerance = 64.0 * jnp.finfo(dtype).eps
        invalid = ~finite | (boundary_defect > tolerance)
        return tuple(
            eqx.error_if(
                value,
                invalid,
                "MAC face momentum must be finite with zero impermeable-wall normal flux.",
            )
            for value in values
        )

    def face_density(
        self,
        density: ArrayLike,
        face_momentum: FaceVelocity,
        /,
    ) -> FaceVelocity:
        """Return the strictly positive donor density selected by mass-flux sign."""
        cell = self.validate_density(density)
        momentum = self.validate_face_momentum(face_momentum)
        output = []
        for axis, grid_axis in enumerate(
            self.operators.discretization.grid.structured_axes
        ):
            moved = jnp.moveaxis(cell, axis, 0)
            moved_momentum = jnp.moveaxis(momentum[axis], axis, 0)
            if grid_axis.periodic:
                lower = jnp.roll(moved, 1, axis=0)
                upper = moved
            else:
                lower = jnp.concatenate((moved[:1], moved), axis=0)
                upper = jnp.concatenate((moved, moved[-1:]), axis=0)
            donor = jnp.where(moved_momentum >= 0.0, lower, upper)
            output.append(jnp.moveaxis(donor, 0, axis))
        return tuple(output)

    def face_density_rate(
        self,
        density_rate: ArrayLike,
        face_momentum: FaceVelocity,
        /,
    ) -> FaceVelocity:
        """Differentiate the active donor branch without changing its stage policy."""
        cell_rate = self.operators.validate_pressure(density_rate)
        momentum = self.validate_face_momentum(face_momentum)
        output = []
        for axis, grid_axis in enumerate(
            self.operators.discretization.grid.structured_axes
        ):
            moved = jnp.moveaxis(cell_rate, axis, 0)
            moved_momentum = jnp.moveaxis(momentum[axis], axis, 0)
            if grid_axis.periodic:
                lower = jnp.roll(moved, 1, axis=0)
                upper = moved
            else:
                lower = jnp.concatenate((moved[:1], moved), axis=0)
                upper = jnp.concatenate((moved, moved[-1:]), axis=0)
            donor = jnp.where(moved_momentum >= 0.0, lower, upper)
            output.append(jnp.moveaxis(donor, 0, axis))
        return tuple(output)

    def velocity(
        self,
        density: ArrayLike,
        face_momentum: FaceVelocity,
        /,
    ) -> FaceVelocity:
        momentum = self.validate_face_momentum(face_momentum)
        face_density = self.face_density(density, momentum)
        return tuple(
            value / density_value
            for value, density_value in zip(momentum, face_density, strict=True)
        )

    def mass_flux(
        self,
        density: ArrayLike,
        face_momentum: FaceVelocity,
        /,
    ) -> FaceVelocity:
        """Return the single authoritative mass flux carried by the state momentum."""
        self.validate_density(density)
        return self.validate_face_momentum(face_momentum)

    def density_rate(
        self,
        density: ArrayLike,
        face_momentum: FaceVelocity,
        /,
    ) -> Array:
        return -self.operators.divergence(self.mass_flux(density, face_momentum))

    def _stage(self, stage: MACBoundaryStageData | None, /) -> MACBoundaryStageData:
        return (
            self.momentum.boundaries.evaluate(jnp.asarray(0.0), None)
            if stage is None
            else self.momentum.boundaries.validate_stage(stage)
        )

    def _transverse_boundary_mass_flux(
        self,
        density: Array,
        mass_flux: FaceVelocity,
        component_axis: int,
        derivative_axis: int,
        side: str,
        stage: MACBoundaryStageData,
        /,
    ) -> Array:
        index = 0 if side == "lower" else -1
        if not self.momentum.boundaries.tangential_dirichlet(component_axis, side):
            return _axis_boundary(mass_flux[derivative_axis], component_axis, index)
        derivative_periodic = self.operators.discretization.grid.structured_axes[
            derivative_axis
        ].periodic
        boundary_density = _patch_to_component_faces(
            _axis_boundary(density, component_axis, index),
            component_axis,
            derivative_axis,
            derivative_periodic,
        )
        boundary_velocity = _patch_to_component_faces(
            self.momentum.boundaries.tangential_value(
                component_axis,
                side,
                derivative_axis,
                stage,
                homogeneous=False,
            ),
            component_axis,
            derivative_axis,
            derivative_periodic,
        )
        return boundary_density * boundary_velocity

    def _dual_mass_flux(
        self,
        density: Array,
        mass_flux: FaceVelocity,
        component_axis: int,
        derivative_axis: int,
        stage: MACBoundaryStageData,
        /,
    ) -> Array:
        grid_axis = self.operators.discretization.grid.structured_axes[component_axis]
        if derivative_axis == component_axis:
            return _center_interpolate(
                mass_flux[derivative_axis], component_axis, grid_axis.periodic
            )
        if grid_axis.periodic:
            zero = jnp.asarray(0.0, dtype=density.dtype)
            return _face_interpolate(
                mass_flux[derivative_axis],
                component_axis,
                True,
                zero,
                zero,
            )
        return _face_interpolate(
            mass_flux[derivative_axis],
            component_axis,
            False,
            self._transverse_boundary_mass_flux(
                density,
                mass_flux,
                component_axis,
                derivative_axis,
                "lower",
                stage,
            ),
            self._transverse_boundary_mass_flux(
                density,
                mass_flux,
                component_axis,
                derivative_axis,
                "upper",
                stage,
            ),
        )

    def _donor_velocity(
        self,
        velocity: FaceVelocity,
        transport_mass_flux: Array,
        component_axis: int,
        derivative_axis: int,
        stage: MACBoundaryStageData,
        /,
    ) -> Array:
        moved = jnp.moveaxis(velocity[component_axis], derivative_axis, 0)
        grid_axis = self.operators.discretization.grid.structured_axes[derivative_axis]
        if derivative_axis == component_axis:
            if grid_axis.periodic:
                lower = moved
                upper = jnp.roll(moved, -1, axis=0)
            else:
                lower = moved[:-1]
                upper = moved[1:]
        elif grid_axis.periodic:
            lower = jnp.roll(moved, 1, axis=0)
            upper = moved
        else:
            component_periodic = self.operators.discretization.grid.structured_axes[
                component_axis
            ].periodic
            lower_wall = jnp.expand_dims(
                _patch_to_component_faces(
                    self.momentum.boundaries.tangential_value(
                        derivative_axis,
                        "lower",
                        component_axis,
                        stage,
                        homogeneous=False,
                    ),
                    derivative_axis,
                    component_axis,
                    component_periodic,
                ),
                axis=0,
            )
            upper_wall = jnp.expand_dims(
                _patch_to_component_faces(
                    self.momentum.boundaries.tangential_value(
                        derivative_axis,
                        "upper",
                        component_axis,
                        stage,
                        homogeneous=False,
                    ),
                    derivative_axis,
                    component_axis,
                    component_periodic,
                ),
                axis=0,
            )
            lower = jnp.concatenate((lower_wall, moved), axis=0)
            upper = jnp.concatenate((moved, upper_wall), axis=0)
        donor = jnp.where(
            jnp.moveaxis(transport_mass_flux, derivative_axis, 0) >= 0.0,
            lower,
            upper,
        )
        return jnp.moveaxis(donor, 0, derivative_axis)

    def momentum_fluxes(
        self,
        density: ArrayLike,
        face_momentum: FaceVelocity,
        /,
        *,
        stage: MACBoundaryStageData | None = None,
    ) -> FaceMomentumFlux:
        stage_ = self._stage(stage)
        cell = self.validate_density(density)
        mass = self.mass_flux(cell, face_momentum)
        velocity = self.velocity(cell, mass)
        fluxes = []
        for component_axis in range(self.dimension):
            component_fluxes = []
            for derivative_axis in range(self.dimension):
                transport = self.momentum.precision.flux(
                    self._dual_mass_flux(
                        cell, mass, component_axis, derivative_axis, stage_
                    )
                )
                donor = self.momentum.precision.flux(
                    self._donor_velocity(
                        velocity,
                        transport,
                        component_axis,
                        derivative_axis,
                        stage_,
                    )
                )
                component_fluxes.append(
                    self.momentum.precision.storage(transport * donor)
                )
            fluxes.append(tuple(component_fluxes))
        return tuple(fluxes)

    def momentum_advection(
        self,
        density: ArrayLike,
        face_momentum: FaceVelocity,
        /,
        *,
        stage: MACBoundaryStageData | None = None,
    ) -> FaceVelocity:
        fluxes = self.momentum_fluxes(density, face_momentum, stage=self._stage(stage))
        output = []
        for component_axis, component_fluxes in enumerate(fluxes):
            rate = jnp.zeros_like(
                face_momentum[component_axis],
                dtype=jnp.dtype(self.momentum.precision.reduction_dtype),
            )
            for derivative_axis, flux in enumerate(component_fluxes):
                rate = self.momentum.precision.reduction(
                    rate
                    + self.momentum._dual_divergence(
                        flux, component_axis, derivative_axis
                    )
                )
            output.append(self.momentum.precision.storage(rate))
        return self.momentum.boundaries.homogeneous_rate(tuple(output))

    def boundary_mass_flux(self, mass_flux: FaceVelocity, /) -> Array:
        mass = self.validate_face_momentum(mass_flux)
        net = jnp.asarray(0.0, dtype=self.operators.pressure_space.dtype)
        for axis, grid_axis in enumerate(
            self.operators.discretization.grid.structured_axes
        ):
            if grid_axis.periodic:
                continue
            measure = self.operators.discretization.face_measures[axis]
            net = net + jnp.sum(
                _axis_boundary(measure * mass[axis], axis, -1)
                - _axis_boundary(measure * mass[axis], axis, 0)
            )
        return net

    def update_density(
        self,
        density: ArrayLike,
        face_momentum: FaceVelocity,
        step_size: ArrayLike,
        /,
    ) -> MACDensityUpdateResult:
        cell = self.validate_density(density)
        mass_flux = self.mass_flux(cell, face_momentum)
        dtype = self.operators.pressure_space.dtype
        step = jnp.asarray(step_size, dtype=dtype).reshape(())
        step = eqx.error_if(
            step,
            ~jnp.isfinite(step) | (step <= 0.0),
            "Density update step_size must be positive and finite.",
        )
        rate = -self.operators.divergence(mass_flux)
        candidate = cell + step * rate
        volumes = self.operators.discretization.cell_volumes.astype(dtype)
        mass_before = jnp.sum(volumes * cell)
        candidate_mass = jnp.sum(volumes * candidate)
        boundary_flux = self.boundary_mass_flux(mass_flux)
        balance = candidate_mass - mass_before + step * boundary_flux
        minimum = jnp.min(candidate)
        finite = jnp.all(jnp.isfinite(candidate))
        positive = minimum > 0.0
        scale = jnp.maximum(jnp.abs(mass_before), 1.0)
        tolerance = 256.0 * jnp.finfo(dtype).eps * scale
        successful = (
            finite & positive & jnp.isfinite(balance) & (jnp.abs(balance) <= tolerance)
        )
        accepted = jnp.where(successful, candidate, cell)
        return MACDensityUpdateResult(
            density=accepted,
            candidate_density=candidate,
            density_rate=rate,
            mass_before=mass_before,
            candidate_mass=candidate_mass,
            accepted_mass=jnp.sum(volumes * accepted),
            boundary_mass_flux=boundary_flux,
            mass_balance_residual=balance,
            minimum_candidate_density=minimum,
            positive=positive,
            finite=finite,
            successful=successful,
            operators_id=self.prepared_id,
        )

    def transport(
        self,
        density: ArrayLike,
        face_momentum: FaceVelocity,
        /,
        *,
        stage: MACBoundaryStageData | None = None,
    ) -> MACVariableDensityTransportResult:
        stage_ = self._stage(stage)
        cell = self.validate_density(density)
        momentum = self.mass_flux(cell, face_momentum)
        face_density = self.face_density(cell, momentum)
        inverse_density = tuple(1.0 / value for value in face_density)
        velocity = tuple(
            value * inverse
            for value, inverse in zip(momentum, inverse_density, strict=True)
        )
        momentum_flux = self.momentum_fluxes(cell, momentum, stage=stage_)
        density_rate = -self.operators.divergence(momentum)
        momentum_advection = self.momentum_advection(cell, momentum, stage=stage_)
        momentum_rate = tuple(-value for value in momentum_advection)
        volumes = self.operators.discretization.cell_volumes.astype(cell.dtype)
        mass = jnp.sum(volumes * cell)
        mass_rate = jnp.sum(volumes * density_rate)
        boundary_flux = self.boundary_mass_flux(momentum)
        total_momentum = jnp.stack(
            tuple(
                jnp.sum(measure * component)
                for measure, component in zip(
                    self.operators.face_dual_measures, momentum, strict=True
                )
            )
        )
        advective_momentum_rate = jnp.stack(
            tuple(
                jnp.sum(measure * component_rate)
                for measure, component_rate in zip(
                    self.operators.face_dual_measures, momentum_rate, strict=True
                )
            )
        )
        kinetic = 0.5 * sum(
            jnp.sum(measure * component * speed)
            for measure, component, speed in zip(
                self.operators.face_dual_measures,
                momentum,
                velocity,
                strict=True,
            )
        )
        advective_kinetic_rate = sum(
            jnp.sum(measure * speed * component_rate)
            for measure, speed, component_rate in zip(
                self.operators.face_dual_measures,
                velocity,
                momentum_rate,
                strict=True,
            )
        )
        identity = _maximum_abs(
            tuple(
                density_value * speed - component
                for density_value, speed, component in zip(
                    face_density, velocity, momentum, strict=True
                )
            ),
            cell.dtype,
        )
        finite = (
            jnp.all(jnp.isfinite(cell))
            & jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(v)) for v in momentum)))
            & jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(v)) for v in velocity)))
            & jnp.all(jnp.isfinite(total_momentum))
            & jnp.isfinite(kinetic)
        )
        positive = (jnp.min(cell) > 0.0) & (
            jnp.min(jnp.stack(tuple(jnp.min(value) for value in face_density))) > 0.0
        )
        scale = jnp.maximum(jnp.abs(mass_rate), 1.0)
        tolerance = 256.0 * jnp.finfo(cell.dtype).eps * scale
        balance = mass_rate + boundary_flux
        successful = (
            finite & positive & (identity <= tolerance) & (jnp.abs(balance) <= tolerance)
        )
        return MACVariableDensityTransportResult(
            density=cell,
            face_momentum=momentum,
            face_density=face_density,
            face_inverse_density=inverse_density,
            velocity=velocity,
            mass_flux=momentum,
            momentum_flux=momentum_flux,
            density_rate=density_rate,
            momentum_advection=momentum_advection,
            momentum_rate=momentum_rate,
            mass=mass,
            mass_rate=mass_rate,
            boundary_mass_flux=boundary_flux,
            mass_balance_residual=balance,
            total_momentum=total_momentum,
            advective_momentum_rate=advective_momentum_rate,
            kinetic_energy=kinetic,
            advective_kinetic_energy_rate=advective_kinetic_rate,
            minimum_density=jnp.min(cell),
            minimum_face_density=jnp.min(
                jnp.stack(tuple(jnp.min(value) for value in face_density))
            ),
            authoritative_mass_flux_residual=identity,
            positive=positive,
            finite=finite,
            successful=successful,
            operators_id=self.prepared_id,
        )

    def _prepare_report(self, identifier: str, /) -> MACVariableDensityReport:
        dtype = self.operators.pressure_space.dtype
        constant = jnp.full(self.operators.discretization.cell_shape, 2.0, dtype=dtype)
        probe = []
        for axis, layout in enumerate(self.operators.discretization.face_layouts):
            values = jnp.sin(
                0.31 * jnp.arange(int(np.prod(layout.shape)), dtype=dtype) + axis
            ).reshape(layout.shape)
            probe.append(values)
        momentum = self.momentum.boundaries.homogeneous_rate(tuple(probe))
        face_density = self.face_density(constant, momentum)
        velocity = self.velocity(constant, momentum)
        mass = self.mass_flux(constant, momentum)
        density_rate = self.density_rate(constant, momentum)
        face_residual = _maximum_abs(tuple(value - 2.0 for value in face_density), dtype)
        velocity_residual = _maximum_abs(
            tuple(
                speed - component / 2.0
                for speed, component in zip(velocity, momentum, strict=True)
            ),
            dtype,
        )
        mass_residual = _maximum_abs(
            tuple(
                density_value * speed - component
                for density_value, speed, component in zip(
                    face_density, velocity, mass, strict=True
                )
            ),
            dtype,
        )
        conservative_residual = jnp.max(
            jnp.abs(density_rate + self.operators.divergence(mass))
        )
        positive = jnp.all(
            jnp.stack(tuple(jnp.all(value > 0.0) for value in face_density))
        )
        epsilon = jnp.finfo(dtype).eps
        tolerance = 256.0 * epsilon
        passed = (
            positive
            & jnp.isfinite(face_residual)
            & jnp.isfinite(velocity_residual)
            & jnp.isfinite(mass_residual)
            & jnp.isfinite(conservative_residual)
            & (face_residual <= tolerance)
            & (velocity_residual <= tolerance)
            & (mass_residual <= tolerance)
            & (conservative_residual <= tolerance)
        )
        return MACVariableDensityReport(
            constant_face_density_residual=face_residual,
            constant_velocity_residual=velocity_residual,
            authoritative_mass_flux_residual=mass_residual,
            conservative_density_residual=conservative_residual,
            positive_face_density=positive,
            passed=passed,
            report_id=canonical_fingerprint(
                {"kind": "mac-variable-density-report", "operators": identifier}
            ),
        )


__all__ = [
    "FaceMomentumFlux",
    "MACDensityUpdateResult",
    "MACVariableDensityPlan",
    "MACVariableDensityReport",
    "MACVariableDensityTransportResult",
    "PreparedMACVariableDensityOperators",
]
