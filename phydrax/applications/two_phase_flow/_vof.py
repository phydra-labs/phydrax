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
from ..._sharp_measures import QualifiedSharpGeometry
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.finite_volume import (
    FiniteVolumeDiscretization,
    MACBoundaryPlan,
    MACOperatorPlan,
)
from ...solver import MACVariableDensityProjectionPlan
from ...solver._mac_sharp_interface import MACSharpInterfaceProjectionPlan


FaceTuple = tuple[Array, ...]


class PLICGeometry(StrictModule):
    normal: Array
    plane_offset: Array
    mixed_cell: Array
    reconstruction_residual: Array
    finite: Array
    valid: Array
    solid_interface_conflict: Array


class TwoPhaseTopologyEvidence(StrictModule):
    liquid_volume: Array
    gas_volume: Array
    mixed_cell_count: Array
    interface_measure: Array
    component_proxy: Array
    changed_cell_mask: Array
    finite: Array
    valid: Array


class TwoPhaseVOFState(StrictModule):
    liquid_content: Array
    momentum: FaceTuple
    phase_scalar_content: dict[str, Array]
    level_set: Array
    geometry_epoch: Array
    geometry_id: str = eqx.field(static=True)


class TwoPhaseVOFView(StrictModule):
    alpha: Array
    density: Array
    viscosity: Array
    velocity: FaceTuple
    pressure: Array
    plic: PLICGeometry
    topology: TwoPhaseTopologyEvidence
    view_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)


class TwoPhaseMaterialPlan(StrictModule, NonTrainableState):
    liquid_density: float = eqx.field(static=True)
    gas_density: float = eqx.field(static=True)
    liquid_viscosity: float = eqx.field(static=True)
    gas_viscosity: float = eqx.field(static=True)
    surface_tension: float = eqx.field(static=True)
    contact_angle: float = eqx.field(static=True)
    material_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        liquid_density: float = 1000.0,
        gas_density: float = 1.2,
        liquid_viscosity: float = 1.0e-3,
        gas_viscosity: float = 1.8e-5,
        surface_tension: float = 0.0,
        contact_angle: float = 0.5 * np.pi,
    ):
        values = tuple(
            float(v)
            for v in (
                liquid_density,
                gas_density,
                liquid_viscosity,
                gas_viscosity,
                surface_tension,
                contact_angle,
            )
        )
        if (
            any(not np.isfinite(v) for v in values)
            or values[0] <= 0.0
            or values[1] <= 0.0
            or values[2] < 0.0
            or values[3] < 0.0
            or values[4] < 0.0
            or not 0.0 < values[5] < np.pi
        ):
            raise ValueError("Invalid two-phase material parameters.")
        (
            self.liquid_density,
            self.gas_density,
            self.liquid_viscosity,
            self.gas_viscosity,
            self.surface_tension,
            self.contact_angle,
        ) = values
        self.material_id = canonical_fingerprint(
            {"kind": "two-phase-material-plan", "values": list(values)}
        )


class IncompressibleTwoPhaseVOFPlan(StrictModule, NonTrainableState):
    """Compile fixed-grid conservative incompressible two-phase VOF flow."""

    discretization: FiniteVolumeDiscretization
    material: TwoPhaseMaterialPlan
    geometry: QualifiedSharpGeometry | None
    tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: FiniteVolumeDiscretization,
        material: TwoPhaseMaterialPlan | None = None,
        /,
        *,
        tolerance: float = 1.0e-9,
        maximum_iterations: int = 500,
        geometry: QualifiedSharpGeometry | None = None,
    ):
        if not isinstance(discretization, FiniteVolumeDiscretization):
            raise TypeError("discretization must be FiniteVolumeDiscretization.")
        if len(discretization.cell_shape) not in (2, 3):
            raise ValueError("Two-phase VOF supports two or three dimensions.")
        material_ = TwoPhaseMaterialPlan() if material is None else material
        tolerance_ = float(tolerance)
        iterations = int(maximum_iterations)
        if tolerance_ <= 0.0 or iterations <= 0:
            raise ValueError("Invalid two-phase solve policy.")
        if geometry is not None:
            if not isinstance(geometry, QualifiedSharpGeometry):
                raise TypeError("geometry must be QualifiedSharpGeometry or None.")
            if (
                geometry.support_id != discretization.support.support_id
                or geometry.cell_field_id != discretization.cell_space.field_space_id
                or geometry.face_field_ids
                != tuple(space.field_space_id for space in discretization.face_spaces)
            ):
                raise ValueError("VOF solid geometry binds another finite-volume grid.")
            if not bool(np.asarray(geometry.accepted)):
                raise ValueError("VOF preparation rejects failed sharp geometry.")
            if np.any(np.asarray(geometry.swept_cell_measure_rate) != 0.0):
                raise ValueError(
                    "Structured VOF sharp composition currently requires static geometry."
                )
        self.discretization = discretization
        self.material = material_
        self.tolerance = tolerance_
        self.maximum_iterations = iterations
        self.geometry = geometry
        self.plan_id = canonical_fingerprint(
            {
                "kind": "incompressible-two-phase-vof-plan",
                "discretization": discretization.prepared_id,
                "material": material_.material_id,
                "tolerance": tolerance_,
                "maximum_iterations": iterations,
                "geometry": None if geometry is None else geometry.realization_id,
            }
        )

    def prepare(self) -> "PreparedIncompressibleTwoPhaseVOF":
        operators = MACOperatorPlan(self.discretization).prepare()
        boundaries = MACBoundaryPlan(operators).prepare()
        projection = MACVariableDensityProjectionPlan(
            operators,
            tolerance=self.tolerance,
            maximum_iterations=self.maximum_iterations,
        )
        sharp_projection = (
            None
            if self.geometry is None
            else MACSharpInterfaceProjectionPlan(
                operators,
                boundaries,
                self.geometry,
                tolerance=self.tolerance,
            )
        )
        if sharp_projection is not None and sharp_projection.component_count != 1:
            raise ValueError(
                "Static VOF sharp composition currently requires one connected "
                "fluid component."
            )
        return PreparedIncompressibleTwoPhaseVOF(
            self, operators, boundaries, projection, sharp_projection
        )


class PreparedIncompressibleTwoPhaseVOF(StrictModule):
    plan: IncompressibleTwoPhaseVOFPlan
    operators: Any
    boundaries: Any
    projection: MACVariableDensityProjectionPlan
    sharp_projection: MACSharpInterfaceProjectionPlan | None
    geometry: QualifiedSharpGeometry | None
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan, operators, boundaries, projection, sharp_projection, /):
        self.plan = plan
        self.operators = operators
        self.boundaries = boundaries
        self.projection = projection
        self.sharp_projection = sharp_projection
        self.geometry = plan.geometry
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-incompressible-two-phase-vof",
                "plan": plan.plan_id,
                "operators": operators.prepared_id,
                "boundaries": boundaries.prepared_id,
                "projection": projection.plan_id,
                "sharp_projection": (
                    None if sharp_projection is None else sharp_projection.plan_id
                ),
                "geometry": (
                    None if self.geometry is None else self.geometry.realization_id
                ),
            }
        )

    @property
    def cell_fluid_measure(self) -> Array:
        return (
            self.plan.discretization.cell_volumes
            if self.geometry is None
            else self.geometry.cell_fluid_measure
        )

    @property
    def face_open_measure(self) -> FaceTuple:
        return (
            self.plan.discretization.face_measures
            if self.geometry is None
            else self.geometry.face_open_measure
        )

    @property
    def face_open_dual_measure(self) -> FaceTuple:
        if self.geometry is None:
            return self.operators.face_dual_measures
        return tuple(
            dual * opened / full
            for dual, opened, full in zip(
                self.operators.face_dual_measures,
                self.geometry.face_open_measure,
                self.geometry.face_full_measure,
                strict=True,
            )
        )

    def initial_state(
        self,
        alpha: ArrayLike,
        velocity: FaceTuple | None = None,
        phase_scalars: dict[str, ArrayLike] | None = None,
        /,
    ) -> TwoPhaseVOFState:
        alpha_ = jnp.asarray(alpha, dtype=self.plan.discretization.cell_volumes.dtype)
        if alpha_.shape != self.plan.discretization.cell_shape:
            raise ValueError("Initial VOF alpha shape is invalid.")
        if bool(jnp.any(~jnp.isfinite(alpha_))) or bool(
            jnp.any((alpha_ < 0.0) | (alpha_ > 1.0))
        ):
            raise ValueError("Initial VOF alpha must lie in [0, 1].")
        fluid_volume = self.cell_fluid_measure
        fluid_active = fluid_volume > 0.0
        if bool(jnp.any((~fluid_active) & (alpha_ != 0.0))):
            raise ValueError("Initial VOF alpha must be zero in solid cells.")
        solid_cut = (
            jnp.zeros_like(alpha_, dtype=bool)
            if self.geometry is None
            else self.geometry.cell_fluid_measure < self.geometry.cell_full_measure
        )
        liquid_cut = (alpha_ > 0.0) & (alpha_ < 1.0)
        if bool(jnp.any(solid_cut & liquid_cut)):
            raise ValueError(
                "Initial VOF rejects cells cut by both solid and liquid-gas PLIC."
            )
        velocity_ = (
            tuple(
                jnp.zeros(layout.shape, dtype=alpha_.dtype)
                for layout in self.plan.discretization.face_layouts
            )
            if velocity is None
            else self.operators.validate_velocity(velocity)
        )
        density = self.mixture_density(alpha_)
        face_density = self.face_density(density)
        momentum = tuple(
            rho * measure * component
            for rho, measure, component in zip(
                face_density,
                self.face_open_dual_measure,
                velocity_,
                strict=True,
            )
        )
        supplied = {} if phase_scalars is None else dict(phase_scalars)
        scalar_content = {}
        for name, value in supplied.items():
            concentration = jnp.asarray(value, dtype=alpha_.dtype)
            if concentration.shape == ():
                concentration = jnp.broadcast_to(concentration, alpha_.shape)
            if concentration.shape != alpha_.shape:
                raise ValueError(f"Two-phase scalar {name!r} shape is invalid.")
            scalar_content[name] = fluid_volume * alpha_ * concentration
        level_set = self.level_set_from_alpha(alpha_)
        geometry_epoch = (
            jnp.asarray(-1, dtype=jnp.int32)
            if self.geometry is None
            else self.geometry.epoch
        )
        geometry_id = "" if self.geometry is None else self.geometry.realization_id
        return TwoPhaseVOFState(
            liquid_content=fluid_volume * alpha_,
            momentum=momentum,
            phase_scalar_content=scalar_content,
            level_set=level_set,
            geometry_epoch=geometry_epoch,
            geometry_id=geometry_id,
        )

    def alpha(self, state: TwoPhaseVOFState, /) -> Array:
        expected = "" if self.geometry is None else self.geometry.realization_id
        if state.geometry_id != expected:
            raise ValueError("VOF state belongs to another solid geometry identity.")
        active = self.cell_fluid_measure > 0.0
        return jnp.where(
            active,
            state.liquid_content / jnp.where(active, self.cell_fluid_measure, 1.0),
            0.0,
        )

    def mixture_density(self, alpha: ArrayLike, /) -> Array:
        alpha_ = jnp.asarray(alpha)
        return (
            self.plan.material.gas_density
            + (self.plan.material.liquid_density - self.plan.material.gas_density)
            * alpha_
        )

    def mixture_viscosity(self, alpha: ArrayLike, /) -> Array:
        alpha_ = jnp.asarray(alpha)
        return (
            self.plan.material.gas_viscosity
            + (self.plan.material.liquid_viscosity - self.plan.material.gas_viscosity)
            * alpha_
        )

    def face_density(self, density: ArrayLike, /) -> FaceTuple:
        density_ = jnp.asarray(density)
        return tuple(
            _harmonic_faces(density_, axis, grid_axis.periodic)
            for axis, grid_axis in enumerate(
                self.plan.discretization.grid.structured_axes
            )
        )

    def velocity(self, state: TwoPhaseVOFState, /) -> FaceTuple:
        density = self.mixture_density(self.alpha(state))
        face_density = self.face_density(density)
        return tuple(
            jnp.where(
                rho * measure > 0.0,
                momentum / jnp.where(rho * measure > 0.0, rho * measure, 1.0),
                0.0,
            )
            for rho, measure, momentum in zip(
                face_density,
                self.face_open_dual_measure,
                state.momentum,
                strict=True,
            )
        )

    def _contact_angle_normal(self, normal: Array, mixed: Array, /) -> Array:
        output = normal
        angle = self.plan.material.contact_angle
        dimension = normal.shape[-1]
        for axis, grid_axis in enumerate(self.plan.discretization.grid.structured_axes):
            if grid_axis.periodic:
                continue
            for index, sign in ((0, -1.0), (-1, 1.0)):
                location = [slice(None)] * mixed.ndim
                location[axis] = index
                boundary = output[tuple(location)]
                boundary_mixed = mixed[tuple(location)]
                tangent = boundary.at[..., axis].set(0.0)
                tangent_norm = jnp.linalg.norm(tangent, axis=-1)
                basis = (
                    jnp.zeros((dimension,), dtype=normal.dtype)
                    .at[(axis + 1) % dimension]
                    .set(1.0)
                )
                direction = jnp.where(
                    tangent_norm[..., None] > 1.0e-12,
                    tangent / tangent_norm[..., None],
                    basis,
                )
                adjusted = jnp.sin(angle) * direction
                adjusted = adjusted.at[..., axis].set(sign * jnp.cos(angle))
                output = output.at[tuple(location)].set(
                    jnp.where(
                        boundary_mixed[..., None],
                        adjusted,
                        boundary,
                    )
                )
        return output

    def plic(self, alpha: ArrayLike, /) -> PLICGeometry:
        alpha_ = jnp.asarray(alpha)
        gradients = jnp.stack(
            tuple(jnp.gradient(alpha_, axis=axis) for axis in range(alpha_.ndim)),
            axis=-1,
        )
        norm = jnp.linalg.norm(gradients, axis=-1)
        normal = jnp.where(
            norm[..., None] > 1.0e-12,
            gradients / norm[..., None],
            jnp.zeros_like(gradients),
        )
        mixed = (alpha_ > 1.0e-12) & (alpha_ < 1.0 - 1.0e-12)
        solid_cut = (
            jnp.zeros_like(mixed)
            if self.geometry is None
            else self.geometry.cell_fluid_measure < self.geometry.cell_full_measure
        )
        conflict = mixed & solid_cut
        normal = self._contact_angle_normal(normal, mixed)
        offset = _plic_offset(normal, alpha_)
        reconstructed = _plic_fraction(normal, offset)
        residual = jnp.where(mixed, reconstructed - alpha_, 0.0)
        finite = (
            jnp.all(jnp.isfinite(normal))
            & jnp.all(jnp.isfinite(offset))
            & jnp.all(jnp.isfinite(residual))
        )
        return PLICGeometry(
            normal=normal,
            plane_offset=offset,
            mixed_cell=mixed,
            reconstruction_residual=residual,
            finite=finite,
            valid=finite & ~jnp.any(conflict) & (jnp.max(jnp.abs(residual)) <= 5.0e-4),
            solid_interface_conflict=conflict,
        )

    def level_set_from_alpha(self, alpha: ArrayLike, /) -> Array:
        alpha_ = jnp.asarray(alpha)
        gradients = jnp.stack(
            tuple(jnp.gradient(alpha_, axis=axis) for axis in range(alpha_.ndim)),
            axis=-1,
        )
        scale = jnp.maximum(jnp.linalg.norm(gradients, axis=-1), 1.0e-6)
        value = (alpha_ - 0.5) / scale
        return jnp.where(self.cell_fluid_measure > 0.0, value, 0.0)

    def topology_evidence(
        self,
        state: TwoPhaseVOFState,
        previous_alpha: ArrayLike | None = None,
        /,
    ) -> TwoPhaseTopologyEvidence:
        alpha = self.alpha(state)
        mixed = (alpha > 1.0e-12) & (alpha < 1.0 - 1.0e-12)
        previous = alpha if previous_alpha is None else jnp.asarray(previous_alpha)
        changed = (alpha >= 0.5) != (previous >= 0.5)
        liquid_volume = jnp.sum(state.liquid_content)
        total_volume = jnp.sum(self.cell_fluid_measure)
        plic = self.plic(alpha)
        interface_measure = jnp.sum(
            self.cell_fluid_measure * jnp.linalg.norm(plic.normal, axis=-1) * mixed
        )
        finite = (
            jnp.all(jnp.isfinite(alpha))
            & jnp.isfinite(liquid_volume)
            & jnp.isfinite(interface_measure)
        )
        return TwoPhaseTopologyEvidence(
            liquid_volume=liquid_volume,
            gas_volume=total_volume - liquid_volume,
            mixed_cell_count=jnp.sum(mixed),
            interface_measure=interface_measure,
            component_proxy=jnp.sum(changed),
            changed_cell_mask=changed,
            finite=finite,
            valid=finite
            & plic.valid
            & jnp.all(
                (alpha >= -64.0 * jnp.finfo(alpha.dtype).eps)
                & (alpha <= 1.0 + 64.0 * jnp.finfo(alpha.dtype).eps)
            ),
        )

    def view(
        self, state: TwoPhaseVOFState, pressure: ArrayLike | None = None, /
    ) -> TwoPhaseVOFView:
        alpha = self.alpha(state)
        density = self.mixture_density(alpha)
        viscosity = self.mixture_viscosity(alpha)
        pressure_ = (
            jnp.zeros_like(alpha)
            if pressure is None
            else jnp.asarray(pressure, dtype=alpha.dtype)
        )
        return TwoPhaseVOFView(
            alpha=alpha,
            density=density,
            viscosity=viscosity,
            velocity=self.velocity(state),
            pressure=pressure_,
            plic=self.plic(alpha),
            topology=self.topology_evidence(state),
            view_id=self.prepared_id,
            geometry_id="" if self.geometry is None else self.geometry.realization_id,
        )


def _harmonic_faces(value: Array, axis: int, periodic: bool, /) -> Array:
    moved = jnp.moveaxis(value, axis, 0)
    if periodic:
        left = jnp.roll(moved, 1, axis=0)
        right = moved
    else:
        left = jnp.concatenate((moved[:1], moved), axis=0)
        right = jnp.concatenate((moved, moved[-1:]), axis=0)
    harmonic = jnp.where(
        left + right > 0.0,
        2.0 * left * right / (left + right),
        0.0,
    )
    return jnp.moveaxis(harmonic, 0, axis)


def _plic_fraction(normal: Array, offset: Array) -> Array:
    dimension = normal.shape[-1]
    corners = jnp.asarray(
        np.asarray(
            [
                [(mask >> axis) & 1 for axis in range(dimension)]
                for mask in range(1 << dimension)
            ],
            dtype=float,
        ),
        dtype=normal.dtype,
    )
    signed = jnp.sum(normal[..., None, :] * corners, axis=-1) - offset[..., None]
    smooth = jax.nn.sigmoid(-64.0 * signed)
    return jnp.mean(smooth, axis=-1)


def _plic_offset(normal: Array, fraction: Array) -> Array:
    lower = jnp.sum(jnp.minimum(normal, 0.0), axis=-1) - 1.0
    upper = jnp.sum(jnp.maximum(normal, 0.0), axis=-1) + 1.0
    for _ in range(40):
        midpoint = 0.5 * (lower + upper)
        reconstructed = _plic_fraction(normal, midpoint)
        lower = jnp.where(reconstructed < fraction, midpoint, lower)
        upper = jnp.where(reconstructed >= fraction, midpoint, upper)
    return 0.5 * (lower + upper)


__all__ = [
    "IncompressibleTwoPhaseVOFPlan",
    "PLICGeometry",
    "PreparedIncompressibleTwoPhaseVOF",
    "TwoPhaseMaterialPlan",
    "TwoPhaseTopologyEvidence",
    "TwoPhaseVOFState",
    "TwoPhaseVOFView",
]
