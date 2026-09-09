#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.finite_volume._dynamics import (
    FiniteVolumeBoundaryTrace,
    PreparedFiniteVolumeDynamics,
)


class CompressibleAerodynamicReference(StrictModule, NonTrainableState):
    """Dimensional reference state and aerodynamic force basis."""

    moment_origin: Array
    aerodynamic_basis: Array
    reference_pressure: float = eqx.field(static=True)
    reference_density: float = eqx.field(static=True)
    reference_velocity: float = eqx.field(static=True)
    reference_area: float = eqx.field(static=True)
    reference_length: float = eqx.field(static=True)
    reference_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_pressure: float,
        reference_density: float,
        reference_velocity: float,
        reference_area: float,
        reference_length: float,
        moment_origin: ArrayLike,
        aerodynamic_basis: ArrayLike,
        /,
    ):
        pressure = float(reference_pressure)
        density = float(reference_density)
        velocity = float(reference_velocity)
        area = float(reference_area)
        length = float(reference_length)
        origin = jnp.asarray(moment_origin)
        basis = jnp.asarray(aerodynamic_basis)
        dimension = origin.size
        if (
            not np.isfinite(pressure)
            or any(
                not np.isfinite(value) or value <= 0.0
                for value in (density, velocity, area, length)
            )
            or origin.ndim != 1
            or dimension not in (2, 3)
            or basis.shape != (dimension, dimension)
            or not np.all(np.isfinite(np.asarray(origin)))
            or not np.all(np.isfinite(np.asarray(basis)))
            or not np.allclose(
                np.asarray(basis) @ np.asarray(basis).T,
                np.eye(dimension),
                rtol=0.0,
                atol=1.0e-10,
            )
        ):
            raise ValueError("Aerodynamic reference values or basis are invalid.")
        self.reference_pressure = pressure
        self.reference_density = density
        self.reference_velocity = velocity
        self.reference_area = area
        self.reference_length = length
        self.moment_origin = origin
        self.aerodynamic_basis = basis
        self.reference_id = canonical_fingerprint(
            {
                "kind": "compressible-aerodynamic-reference",
                "pressure": pressure,
                "density": density,
                "velocity": velocity,
                "area": area,
                "length": length,
                "origin": array_tree_fingerprint(origin),
                "basis": array_tree_fingerprint(basis),
            }
        )

    @property
    def dynamic_pressure(self) -> float:
        return 0.5 * self.reference_density * self.reference_velocity**2


class CompressibleSurfacePatchPlan(StrictModule, NonTrainableState):
    """One axis-aligned structured or mapped boundary surface selection."""

    mask: Array | None
    axis: int = eqx.field(static=True)
    side: Literal["lower", "upper"] = eqx.field(static=True)
    name: str = eqx.field(static=True)
    patch_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis: int,
        side: Literal["lower", "upper"],
        /,
        *,
        name: str,
        mask: ArrayLike | None = None,
    ):
        axis_ = int(axis)
        name_ = str(name)
        mask_ = None if mask is None else jnp.asarray(mask, dtype=bool)
        if axis_ < 0 or side not in ("lower", "upper") or not name_:
            raise ValueError("Surface patch axis, side, and name are required.")
        if mask_ is not None and mask_.ndim == 0:
            raise ValueError("Surface patch mask must describe boundary faces.")
        self.axis = axis_
        self.side = side
        self.name = name_
        self.mask = mask_
        self.patch_id = canonical_fingerprint(
            {
                "kind": "compressible-surface-patch",
                "axis": axis_,
                "side": side,
                "name": name_,
                "mask": None
                if mask_ is None
                else array_tree_fingerprint(mask_.astype(jnp.int8)),
            }
        )


class CompressibleSurfacePatchObservation(StrictModule):
    trace: FiniteVolumeBoundaryTrace
    mask: Array
    pressure: Array
    temperature: Array
    pressure_coefficient: Array
    pressure_force_density: Array
    viscous_force_density: Array
    total_force_density: Array
    wall_heat_flux_to_body: Array
    integrated_pressure_force: Array
    integrated_viscous_force: Array
    integrated_total_force: Array
    integrated_moment: Array
    force_balance_defect: Array
    finite: Array
    successful: Array
    name: str = eqx.field(static=True)


class CompressibleSurfaceObservation(StrictModule):
    patches: tuple[CompressibleSurfacePatchObservation, ...]
    integrated_pressure_force: Array
    integrated_viscous_force: Array
    integrated_total_force: Array
    integrated_moment: Array
    force_coefficients: Array
    moment_coefficients: Array
    force_balance_defect: Array
    finite: Array
    successful: Array


class CompressibleSurfaceObservationPlan(StrictModule, NonTrainableState):
    """Discrete aerodynamic loads from the same face traces used by FV."""

    dynamics: PreparedFiniteVolumeDynamics
    patches: tuple[CompressibleSurfacePatchPlan, ...]
    reference: CompressibleAerodynamicReference
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: PreparedFiniteVolumeDynamics,
        patches: tuple[CompressibleSurfacePatchPlan, ...],
        reference: CompressibleAerodynamicReference,
        /,
    ):
        if (
            not isinstance(dynamics, PreparedFiniteVolumeDynamics)
            or not patches
            or any(
                not isinstance(patch, CompressibleSurfacePatchPlan) for patch in patches
            )
            or len({patch.patch_id for patch in patches}) != len(patches)
            or not isinstance(reference, CompressibleAerodynamicReference)
            or reference.moment_origin.shape != (dynamics.system.dimension,)
        ):
            raise ValueError("Compressible surface observation inputs are invalid.")
        for patch in patches:
            if (
                patch.axis >= dynamics.system.dimension
                or dynamics.discretization.grid.structured_axes[patch.axis].periodic
            ):
                raise ValueError("Surface patches must identify nonperiodic boundaries.")
        self.dynamics = dynamics
        self.patches = tuple(patches)
        self.reference = reference
        self.plan_id = canonical_fingerprint(
            {
                "kind": "compressible-surface-observation",
                "dynamics": dynamics.dynamics_id,
                "patches": tuple(patch.patch_id for patch in patches),
                "reference": reference.reference_id,
            }
        )

    def _patch_observation(
        self,
        patch: CompressibleSurfacePatchPlan,
        time: Array,
        state: Array,
        args: Any,
        /,
    ) -> CompressibleSurfacePatchObservation:
        trace = self.dynamics.boundary_trace(time, state, patch.axis, patch.side, args)
        mask = (
            jnp.ones(trace.face_measure.shape, dtype=bool)
            if patch.mask is None
            else jnp.broadcast_to(patch.mask, trace.face_measure.shape)
        )
        system = self.dynamics.system
        pressure = system.pressure(trace.interior_state)
        temperature = system.temperature(trace.interior_state)
        pressure_coefficient = (
            pressure - self.reference.reference_pressure
        ) / self.reference.dynamic_pressure
        pressure_force_density = pressure[..., None] * trace.outward_normal
        viscous_force_density = -trace.outward_diffusive_flux[..., system.momentum_slice]
        total_force_density = trace.outward_total_flux[..., system.momentum_slice]
        weight = jnp.where(mask, trace.face_measure, 0.0)
        pressure_force = jnp.sum(
            pressure_force_density * weight[..., None],
            axis=tuple(range(weight.ndim)),
        )
        viscous_force = jnp.sum(
            viscous_force_density * weight[..., None],
            axis=tuple(range(weight.ndim)),
        )
        total_force = jnp.sum(
            total_force_density * weight[..., None],
            axis=tuple(range(weight.ndim)),
        )
        force = total_force_density * weight[..., None]
        arm = trace.face_coordinates - self.reference.moment_origin
        if system.dimension == 2:
            moment_density = arm[..., 0] * force[..., 1] - arm[..., 1] * force[..., 0]
            moment = jnp.sum(moment_density)
        else:
            moment = jnp.sum(jnp.cross(arm, force), axis=tuple(range(weight.ndim)))
        wall_heat = -trace.outward_diffusive_flux[..., system.energy_index]
        defect = total_force - pressure_force - viscous_force
        finite = jnp.all(
            jnp.stack(
                (
                    jnp.all(jnp.isfinite(pressure)),
                    jnp.all(jnp.isfinite(temperature)),
                    jnp.all(jnp.isfinite(total_force)),
                    jnp.all(jnp.isfinite(moment)),
                )
            )
        )
        successful = finite & jnp.all(system.admissible(trace.interior_state))
        return CompressibleSurfacePatchObservation(
            trace,
            mask,
            pressure,
            temperature,
            pressure_coefficient,
            pressure_force_density,
            viscous_force_density,
            total_force_density,
            wall_heat,
            pressure_force,
            viscous_force,
            total_force,
            moment,
            defect,
            finite,
            successful,
            patch.name,
        )

    def evaluate(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> CompressibleSurfaceObservation:
        value = jnp.asarray(state)
        observations = tuple(
            self._patch_observation(patch, jnp.asarray(time), value, args)
            for patch in self.patches
        )
        pressure_force = jnp.sum(
            jnp.stack(tuple(item.integrated_pressure_force for item in observations)),
            axis=0,
        )
        viscous_force = jnp.sum(
            jnp.stack(tuple(item.integrated_viscous_force for item in observations)),
            axis=0,
        )
        total_force = jnp.sum(
            jnp.stack(tuple(item.integrated_total_force for item in observations)),
            axis=0,
        )
        moment = jnp.sum(
            jnp.stack(tuple(item.integrated_moment for item in observations)), axis=0
        )
        denominator = self.reference.dynamic_pressure * self.reference.reference_area
        force_coefficients = (
            self.reference.aerodynamic_basis @ total_force
        ) / denominator
        moment_coefficients = moment / (denominator * self.reference.reference_length)
        defect = total_force - pressure_force - viscous_force
        finite = jnp.all(
            jnp.stack(tuple(item.finite for item in observations))
        ) & jnp.all(jnp.isfinite(force_coefficients))
        successful = finite & jnp.all(
            jnp.stack(tuple(item.successful for item in observations))
        )
        return CompressibleSurfaceObservation(
            observations,
            pressure_force,
            viscous_force,
            total_force,
            moment,
            force_coefficients,
            moment_coefficients,
            defect,
            finite,
            successful,
        )


__all__ = [
    "CompressibleAerodynamicReference",
    "CompressibleSurfaceObservation",
    "CompressibleSurfaceObservationPlan",
    "CompressibleSurfacePatchObservation",
    "CompressibleSurfacePatchPlan",
]
