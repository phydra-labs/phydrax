#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._incompressible import FaceVelocity, PreparedMACOperators


def _axis_boundary_mask(shape: tuple[int, ...], axis: int, periodic: bool, /) -> Array:
    mask = jnp.ones(shape)
    if periodic:
        return mask
    lower = [slice(None)] * len(shape)
    upper = [slice(None)] * len(shape)
    lower[axis] = 0
    upper[axis] = shape[axis] - 1
    return mask.at[tuple(lower)].set(0.0).at[tuple(upper)].set(0.0)


def _to_cell_centers(value: Array, axis: int, periodic: bool, /) -> Array:
    moved = jnp.moveaxis(value, axis, 0)
    centered = (
        0.5 * (moved + jnp.roll(moved, -1, axis=0))
        if periodic
        else 0.5 * (moved[:-1] + moved[1:])
    )
    return jnp.moveaxis(centered, 0, axis)


def _to_component_faces(value: Array, axis: int, periodic: bool, /) -> Array:
    moved = jnp.moveaxis(value, axis, 0)
    if periodic:
        faces = 0.5 * (moved + jnp.roll(moved, 1, axis=0))
    else:
        zero = jnp.zeros_like(moved[:1])
        interior = 0.5 * (moved[:-1] + moved[1:])
        faces = jnp.concatenate((zero, interior, zero), axis=0)
    return jnp.moveaxis(faces, 0, axis)


class MACOceanForcingEvidence(StrictModule):
    """Weighted Coriolis and rigid-lid surface-stress evidence."""

    force: FaceVelocity
    coriolis_force: FaceVelocity
    surface_stress_force: FaceVelocity
    coriolis_power: Array
    surface_stress_power: Array
    coriolis_work_scale: Array
    normalized_coriolis_work_defect: Array
    finite: Array
    success: Array
    plan_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class PreparedMACOceanForcing(StrictModule, NonTrainableState):
    """Energy-neutral f-plane rotation and impermeable top-surface stress."""

    operators: PreparedMACOperators
    coriolis_parameter: float = eqx.field(static=True)
    surface_index: int = eqx.field(static=True)
    horizontal_axes: tuple[int, int] = eqx.field(static=True)
    vertical_axis: int = eqx.field(static=True)
    reference_density: float = eqx.field(static=True)
    surface_stress: Array
    surface_stress_function: Any = eqx.field(static=True)
    surface_stress_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        coriolis_parameter: float,
        /,
        *,
        surface_at_upper: bool = True,
        vertical_axis: int,
        reference_density: float,
        surface_stress: Sequence[float] | Any | None = None,
        surface_stress_id: str | None = None,
    ):
        if not isinstance(operators, PreparedMACOperators):
            raise TypeError("operators must be PreparedMACOperators.")
        dimension = len(operators.discretization.cell_shape)
        vertical = int(vertical_axis)
        if dimension != 3 or vertical not in range(dimension):
            raise ValueError("MAC ocean forcing requires one vertical axis in 3D.")
        horizontal = tuple(axis for axis in range(dimension) if axis != vertical)
        f = float(coriolis_parameter)
        density = float(reference_density)
        if not np.isfinite(f) or not np.isfinite(density) or density <= 0.0:
            raise ValueError("Coriolis parameter and reference density must be finite.")
        vertical_grid_axis = operators.discretization.grid.structured_axes[vertical]
        if vertical_grid_axis.periodic:
            raise ValueError("MAC ocean forcing requires a bounded vertical axis.")
        if callable(surface_stress):
            identifier = "" if surface_stress_id is None else str(surface_stress_id)
            if not identifier:
                raise ValueError("Dynamic surface stress requires surface_stress_id.")
            stress = jnp.zeros((dimension,), dtype=operators.pressure_space.dtype)
            stress_function = surface_stress
        else:
            if surface_stress_id is not None:
                raise ValueError("surface_stress_id is only valid for dynamic stress.")
            raw = (
                jnp.zeros((dimension,), dtype=operators.pressure_space.dtype)
                if surface_stress is None
                else jnp.asarray(surface_stress, dtype=operators.pressure_space.dtype)
            )
            if raw.shape != (dimension,) or bool(jnp.any(~jnp.isfinite(raw))):
                raise ValueError("Surface stress must be one finite value per axis.")
            if float(raw[vertical]) != 0.0:
                raise ValueError(
                    "Rigid-lid surface stress cannot have a normal component."
                )
            stress = raw
            stress_function = None
            identifier = canonical_fingerprint(np.asarray(raw).tolist())
        self.operators = operators
        self.coriolis_parameter = f
        self.horizontal_axes = (horizontal[0], horizontal[1])
        self.vertical_axis = vertical
        self.surface_index = -1 if bool(surface_at_upper) else 0
        self.reference_density = density
        self.surface_stress = stress
        self.surface_stress_function = stress_function
        self.surface_stress_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "prepared-mac-ocean-forcing",
                "operators": operators.prepared_id,
                "f": f,
                "horizontal_axes": list(horizontal),
                "vertical_axis": vertical,
                "surface_index": self.surface_index,
                "reference_density": density,
                "surface_stress": identifier,
            }
        )

    def _mask(self, component_axis: int, /) -> Array:
        grid_axis = self.operators.discretization.grid.structured_axes[component_axis]
        layout = self.operators.discretization.face_layouts[component_axis]
        return _axis_boundary_mask(layout.shape, component_axis, grid_axis.periodic)

    def _cross_interpolate(
        self,
        source: Array,
        source_axis: int,
        target_axis: int,
        /,
    ) -> Array:
        grid = self.operators.discretization.grid
        source_mask = self._mask(source_axis).astype(source.dtype)
        centered = _to_cell_centers(
            source_mask * source,
            source_axis,
            grid.structured_axes[source_axis].periodic,
        )
        target = _to_component_faces(
            centered,
            target_axis,
            grid.structured_axes[target_axis].periodic,
        )
        return self._mask(target_axis).astype(target.dtype) * target

    def _coriolis_force(self, velocity: FaceVelocity, /) -> FaceVelocity:
        values = self.operators.validate_velocity(velocity)
        first_axis, second_axis = self.horizontal_axes
        first_weight = self.operators.face_dual_measures[first_axis]
        second_weight = self.operators.face_dual_measures[second_axis]

        def first_from_second(component):
            return self._cross_interpolate(component, second_axis, first_axis)

        first_force = self.coriolis_parameter * first_from_second(values[second_axis])
        cotangent = self.coriolis_parameter * first_weight * values[first_axis]
        transpose = jax.linear_transpose(first_from_second, values[second_axis])(
            cotangent
        )[0]
        second_force = -transpose / second_weight
        output = [jnp.zeros_like(component) for component in values]
        output[first_axis] = first_force
        output[second_axis] = second_force
        return tuple(output)

    def _surface_stress_values(
        self,
        time: Array,
        axis: int,
        args: Any,
        /,
    ) -> Array:
        coordinates = jnp.take(
            self.operators.discretization.face_centers[axis],
            self.surface_index,
            axis=self.vertical_axis,
        )
        target_shape = coordinates.shape[:-1] + (3,)
        if self.surface_stress_function is None:
            return jnp.broadcast_to(self.surface_stress, target_shape)
        value = jnp.asarray(
            self.surface_stress_function(time, coordinates, args),
            dtype=coordinates.dtype,
        )
        if value.shape == (3,):
            value = jnp.broadcast_to(value, target_shape)
        elif value.shape != target_shape:
            raise ValueError(
                "Dynamic surface stress must return a three-component vector "
                f"over boundary shape {coordinates.shape[:-1]}."
            )
        return eqx.error_if(
            value,
            jnp.any(~jnp.isfinite(value)),
            "Dynamic surface stress must be finite.",
        )

    def _surface_stress_force(
        self,
        time: Array,
        velocity: FaceVelocity,
        args: Any,
        /,
    ) -> FaceVelocity:
        values = self.operators.validate_velocity(velocity)
        vertical_width = self.operators.discretization.grid.structured_axes[
            self.vertical_axis
        ].interval_widths[self.surface_index]
        output = [jnp.zeros_like(component) for component in values]
        for axis in self.horizontal_axes:
            stresses = self._surface_stress_values(time, axis, args)[..., axis]
            location = [slice(None)] * output[axis].ndim
            location[self.vertical_axis] = self.surface_index
            output[axis] = (
                output[axis]
                .at[tuple(location)]
                .set(stresses / (self.reference_density * vertical_width))
            )
        return tuple(output)

    def evaluate(
        self,
        time: ArrayLike,
        velocity: FaceVelocity,
        args: Any = None,
        /,
    ) -> MACOceanForcingEvidence:
        time_ = jnp.asarray(time)
        if time_.shape != ():
            raise ValueError("MAC ocean forcing time must be scalar.")
        values = self.operators.validate_velocity(velocity)
        coriolis = self._coriolis_force(values)
        stress = self._surface_stress_force(time_, values, args)
        force = tuple(
            rotation + traction
            for rotation, traction in zip(coriolis, stress, strict=True)
        )
        space = self.operators.velocity_space
        coriolis_power = jnp.real(space.inner(values, coriolis))
        stress_power = jnp.real(space.inner(values, stress))
        scale = jnp.maximum(
            jnp.abs(self.coriolis_parameter) * jnp.real(space.inner(values, values)),
            jnp.finfo(values[0].dtype).tiny,
        )
        normalized = jnp.abs(coriolis_power) / scale
        tolerance = 128.0 * jnp.finfo(values[0].dtype).eps
        finite = (
            jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(v)) for v in force)))
            & jnp.isfinite(coriolis_power)
            & jnp.isfinite(stress_power)
            & jnp.isfinite(normalized)
        )
        success = finite & (normalized <= tolerance)
        return MACOceanForcingEvidence(
            force=force,
            coriolis_force=coriolis,
            surface_stress_force=stress,
            coriolis_power=coriolis_power,
            surface_stress_power=stress_power,
            coriolis_work_scale=scale,
            normalized_coriolis_work_defect=normalized,
            finite=finite,
            success=success,
            plan_id=self.plan_id,
            evidence_id=canonical_fingerprint(
                {"kind": "mac-ocean-forcing-evidence", "plan": self.plan_id}
            ),
        )

    def step_restriction(self, /) -> Array:
        frequency = abs(self.coriolis_parameter)
        return jnp.asarray(
            math.inf if frequency == 0.0 else math.sqrt(3.0) / frequency,
            dtype=self.operators.pressure_space.dtype,
        )


__all__ = ["MACOceanForcingEvidence", "PreparedMACOceanForcing"]
