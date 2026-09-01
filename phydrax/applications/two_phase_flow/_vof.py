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
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.finite_volume import (
    FiniteVolumeDiscretization,
    MACBoundaryPlan,
    MACOperatorPlan,
)
from ...solver import MACVariableDensityProjectionPlan


FaceTuple = tuple[Array, ...]


class PLICGeometry(StrictModule):
    normal: Array
    plane_offset: Array
    mixed_cell: Array
    reconstruction_residual: Array
    finite: Array
    valid: Array


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


class TwoPhaseVOFView(StrictModule):
    alpha: Array
    density: Array
    viscosity: Array
    velocity: FaceTuple
    pressure: Array
    plic: PLICGeometry
    topology: TwoPhaseTopologyEvidence
    view_id: str = eqx.field(static=True)


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
        self.discretization = discretization
        self.material = material_
        self.tolerance = tolerance_
        self.maximum_iterations = iterations
        self.plan_id = canonical_fingerprint(
            {
                "kind": "incompressible-two-phase-vof-plan",
                "discretization": discretization.prepared_id,
                "material": material_.material_id,
                "tolerance": tolerance_,
                "maximum_iterations": iterations,
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
        return PreparedIncompressibleTwoPhaseVOF(self, operators, boundaries, projection)


class PreparedIncompressibleTwoPhaseVOF(StrictModule):
    plan: IncompressibleTwoPhaseVOFPlan
    operators: Any
    boundaries: Any
    projection: MACVariableDensityProjectionPlan
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan, operators, boundaries, projection, /):
        self.plan = plan
        self.operators = operators
        self.boundaries = boundaries
        self.projection = projection
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-incompressible-two-phase-vof",
                "plan": plan.plan_id,
                "operators": operators.prepared_id,
                "boundaries": boundaries.prepared_id,
                "projection": projection.plan_id,
            }
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
                self.operators.face_dual_measures,
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
            scalar_content[name] = (
                self.plan.discretization.cell_volumes * alpha_ * concentration
            )
        level_set = self.level_set_from_alpha(alpha_)
        return TwoPhaseVOFState(
            liquid_content=self.plan.discretization.cell_volumes * alpha_,
            momentum=momentum,
            phase_scalar_content=scalar_content,
            level_set=level_set,
        )

    def alpha(self, state: TwoPhaseVOFState, /) -> Array:
        return state.liquid_content / self.plan.discretization.cell_volumes

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
                momentum / (rho * measure),
                0.0,
            )
            for rho, measure, momentum in zip(
                face_density,
                self.operators.face_dual_measures,
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
            valid=finite & (jnp.max(jnp.abs(residual)) <= 5.0e-4),
        )

    def level_set_from_alpha(self, alpha: ArrayLike, /) -> Array:
        alpha_ = jnp.asarray(alpha)
        gradients = jnp.stack(
            tuple(jnp.gradient(alpha_, axis=axis) for axis in range(alpha_.ndim)),
            axis=-1,
        )
        scale = jnp.maximum(jnp.linalg.norm(gradients, axis=-1), 1.0e-6)
        return (alpha_ - 0.5) / scale

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
        total_volume = jnp.sum(self.plan.discretization.cell_volumes)
        plic = self.plic(alpha)
        interface_measure = jnp.sum(
            self.plan.discretization.cell_volumes
            * jnp.linalg.norm(plic.normal, axis=-1)
            * mixed
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
