#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..discretization import (
    DerivativeRequest,
    DiscretizationBundle,
    DiscretizationKey,
    DiscretizationRecord,
    DiscretizationRole,
    FiniteDifferencePlan,
    GridLocation,
    PreparedFiniteDifferenceDiscretization,
    PreparedTensorGrid,
)


def _pml_profile(
    count: int,
    cell_count: int,
    width: int,
    maximum_attenuation: float,
    polynomial_order: int,
    /,
    *,
    offset: float,
) -> Array:
    if width == 0:
        return jnp.zeros((count,))
    coordinate = jnp.arange(count, dtype=float) + offset
    lower_depth = jnp.clip((width - coordinate) / width, 0.0, 1.0)
    upper_depth = jnp.clip(
        (coordinate - (cell_count - width)) / width,
        0.0,
        1.0,
    )
    return (
        maximum_attenuation
        * jnp.maximum(
            lower_depth,
            upper_depth,
        )
        ** polynomial_order
    )


class SplitFieldPMLPlan(StrictModule):
    """Per-axis polynomial damping profiles for a split-field acoustic PML."""

    widths: tuple[int, ...] = eqx.field(static=True)
    maximum_attenuation: float = eqx.field(static=True)
    polynomial_order: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        widths: int | Sequence[int],
        /,
        *,
        maximum_attenuation: float,
        polynomial_order: int = 2,
    ):
        values = (
            (int(widths),)
            if isinstance(widths, int)
            else tuple(int(value) for value in widths)
        )
        attenuation = float(maximum_attenuation)
        order = int(polynomial_order)
        if not values or any(value < 0 for value in values):
            raise ValueError("PML widths must be non-negative.")
        if not np.isfinite(attenuation) or attenuation < 0.0 or order <= 0:
            raise ValueError("PML attenuation/order must be valid.")
        self.widths = values
        self.maximum_attenuation = attenuation
        self.polynomial_order = order
        self.plan_id = canonical_fingerprint(
            {
                "kind": "split-field-pml-plan",
                "widths": list(values),
                "maximum_attenuation": attenuation,
                "polynomial_order": order,
            }
        )

    def prepare(self, grid: PreparedTensorGrid, /) -> "PreparedSplitFieldPML":
        return PreparedSplitFieldPML(self, grid)


class PreparedSplitFieldPML(StrictModule):
    """Cell and normal-face damping for every split acoustic direction."""

    cell_damping: tuple[Array, ...]
    face_damping: tuple[Array, ...]
    widths: tuple[int, ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: SplitFieldPMLPlan,
        grid: PreparedTensorGrid,
        /,
    ):
        widths = plan.widths * len(grid.shape) if len(plan.widths) == 1 else plan.widths
        if len(widths) != len(grid.shape):
            raise ValueError("PML requires one width per tensor axis.")
        cell_profiles = []
        face_profiles = []
        for axis, (axis_name, size, width) in enumerate(
            zip(grid.axis_names, grid.shape, widths, strict=True)
        ):
            if width and grid.structured_axes[axis].periodic:
                raise ValueError("A periodic axis cannot also carry a physical PML.")
            if 2 * width >= size:
                raise ValueError("PML leaves no undamped interior.")
            cell_profile = _pml_profile(
                size,
                size,
                width,
                plan.maximum_attenuation,
                plan.polynomial_order,
                offset=0.5,
            )
            cell_reshape = [1] * len(grid.shape)
            cell_reshape[axis] = size
            cell_profiles.append(
                jnp.broadcast_to(cell_profile.reshape(cell_reshape), grid.cells().shape)
            )
            face_shape = grid.faces(axis_name).shape
            face_count = face_shape[axis]
            face_profile = _pml_profile(
                face_count,
                size,
                width,
                plan.maximum_attenuation,
                plan.polynomial_order,
                offset=0.0,
            )
            face_reshape = [1] * len(face_shape)
            face_reshape[axis] = face_count
            face_profiles.append(
                jnp.broadcast_to(face_profile.reshape(face_reshape), face_shape)
            )
        self.cell_damping = tuple(cell_profiles)
        self.face_damping = tuple(face_profiles)
        self.widths = widths
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-split-field-pml",
                "plan": plan.plan_id,
                "grid": grid.prepared_id,
            }
        )


class StaggeredAcousticState(StrictModule):
    """Directional pressure splits and independently shaped face velocities."""

    pressure_components: tuple[Array, ...]
    velocity: tuple[Array, ...]

    def __init__(
        self,
        pressure_components: Sequence[ArrayLike],
        velocity: Sequence[ArrayLike],
        /,
    ):
        components = tuple(jnp.asarray(value) for value in pressure_components)
        if not components:
            raise ValueError("Acoustic state requires at least one pressure component.")
        self.pressure_components = components
        self.velocity = tuple(jnp.asarray(value) for value in velocity)

    @property
    def pressure(self) -> Array:
        total = self.pressure_components[0]
        for component in self.pressure_components[1:]:
            total = total + component
        return total


class StaggeredAcousticPlan(StrictModule):
    """Pressure-cell/velocity-face acoustics with optional split-field PML."""

    grid: PreparedTensorGrid
    bulk_modulus: Array
    density: Array
    accuracy_order: int = eqx.field(static=True)
    pml: SplitFieldPMLPlan | None
    source: Callable[[Array, Array, Any], ArrayLike] | None = eqx.field(static=True)
    sensor_indices: Array | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedTensorGrid,
        /,
        *,
        bulk_modulus: ArrayLike,
        density: ArrayLike,
        accuracy_order: int = 2,
        pml: SplitFieldPMLPlan | None = None,
        source: Callable[[Array, Array, Any], ArrayLike] | None = None,
        sensor_indices: ArrayLike | None = None,
        plan_id: str | None = None,
    ):
        if not isinstance(grid, PreparedTensorGrid):
            raise TypeError("grid must be a PreparedTensorGrid.")
        if grid.primary_entity_layout.layout_id != grid.cells().layout_id:
            raise ValueError(
                "Staggered acoustics requires an interval-primary tensor grid."
            )
        bulk = jnp.broadcast_to(jnp.asarray(bulk_modulus), grid.shape)
        density_ = jnp.broadcast_to(jnp.asarray(density), grid.shape)
        bulk = eqx.error_if(
            bulk,
            jnp.any(~jnp.isfinite(bulk)) | jnp.any(bulk <= 0.0),
            "Acoustic bulk modulus must be finite and positive.",
        )
        density_ = eqx.error_if(
            density_,
            jnp.any(~jnp.isfinite(density_)) | jnp.any(density_ <= 0.0),
            "Acoustic density must be finite and positive.",
        )
        order = int(accuracy_order)
        if order <= 0:
            raise ValueError("accuracy_order must be positive.")
        if pml is not None and not isinstance(pml, SplitFieldPMLPlan):
            raise TypeError("pml must be SplitFieldPMLPlan or None.")
        if source is not None and not callable(source):
            raise TypeError("source must be callable or None.")
        sensors = None
        if sensor_indices is not None:
            sensors_host = np.asarray(sensor_indices, dtype=np.int32)
            if sensors_host.ndim != 2 or sensors_host.shape[1] != len(grid.shape):
                raise ValueError("sensor_indices must have shape (sensors, dimension).")
            for axis, size in enumerate(grid.shape):
                if np.any(sensors_host[:, axis] < 0) or np.any(
                    sensors_host[:, axis] >= size
                ):
                    raise ValueError("sensor_indices contain out-of-range entries.")
            sensors = jnp.asarray(sensors_host)
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "staggered-acoustic-plan",
                    "grid": grid.prepared_id,
                    "accuracy_order": order,
                    "pml": None if pml is None else pml.plan_id,
                    "source": None if source is None else repr(source),
                    "sensor_shape": None if sensors is None else list(sensors.shape),
                }
            )
            if plan_id is None
            else str(plan_id)
        )
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.grid = grid
        self.bulk_modulus = bulk
        self.density = density_
        self.accuracy_order = order
        self.pml = pml
        self.source = source
        self.sensor_indices = sensors
        self.plan_id = identifier

    def prepare(self, /) -> "PreparedStaggeredAcoustics":
        requests = []
        cell_layout = self.grid.cells()
        center = GridLocation(
            self.grid.axis_names,
            cell_layout.offsets,
            location_id=cell_layout.location_id,
        )
        for axis in self.grid.axis_names:
            face_layout = self.grid.faces(axis)
            face = GridLocation(
                self.grid.axis_names,
                face_layout.offsets,
                location_id=face_layout.location_id,
            )
            requests.extend(
                (
                    DerivativeRequest(
                        f"grad_{axis}",
                        self.grid,
                        axis,
                        accuracy_order=self.accuracy_order,
                        source_location=center,
                        target_location=face,
                    ),
                    DerivativeRequest(
                        f"div_{axis}",
                        self.grid,
                        axis,
                        accuracy_order=self.accuracy_order,
                        source_location=face,
                        target_location=center,
                    ),
                )
            )
        discretization = FiniteDifferencePlan(
            self.grid,
            requests,
            field_name="acoustic",
            key=DiscretizationKey(
                "staggered_acoustics",
                DiscretizationRole.PHYSICAL,
                domain_labels=self.grid.axis_names,
            ),
        ).prepare()
        return PreparedStaggeredAcoustics(self, discretization)


def _integrating_factor_step(
    value: Array,
    forcing: Array,
    damping: Array,
    step_size: Array,
    /,
) -> Array:
    decay = jnp.exp(-damping * step_size)
    safe_damping = jnp.where(damping > 0.0, damping, 1.0)
    gain = jnp.where(
        damping > 0.0,
        (1.0 - decay) / safe_damping,
        step_size,
    )
    return decay * value + gain * forcing


class PreparedStaggeredAcoustics(StrictModule):
    """Split-pressure PML acoustics on exact cell and face entity layouts."""

    plan: StaggeredAcousticPlan
    discretization: PreparedFiniteDifferenceDiscretization
    pml: PreparedSplitFieldPML | None
    face_density: tuple[Array, ...]
    split_pressure_damping: tuple[Array, ...]
    velocity_damping: tuple[Array, ...]
    stable_dt: Array
    pressure_shape: tuple[int, ...] = eqx.field(static=True)
    velocity_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    discretization_bundle: DiscretizationBundle
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: StaggeredAcousticPlan,
        discretization: PreparedFiniteDifferenceDiscretization,
        /,
    ):
        if not isinstance(plan, StaggeredAcousticPlan) or not isinstance(
            discretization, PreparedFiniteDifferenceDiscretization
        ):
            raise TypeError("Invalid staggered acoustic preparation inputs.")
        prepared_pml = None if plan.pml is None else plan.pml.prepare(plan.grid)
        split_pressure_damping = (
            tuple(jnp.zeros(plan.grid.cells().shape) for _ in plan.grid.axis_names)
            if prepared_pml is None
            else prepared_pml.cell_damping
        )
        velocity_damping = (
            tuple(jnp.zeros(plan.grid.faces(axis).shape) for axis in plan.grid.axis_names)
            if prepared_pml is None
            else prepared_pml.face_damping
        )
        densities = []
        spacings = []
        velocity_shapes = []
        for axis, axis_name in enumerate(plan.grid.axis_names):
            face_layout = plan.grid.faces(axis_name)
            velocity_shapes.append(face_layout.shape)
            density = plan.density
            if plan.grid.structured_axes[axis].periodic:
                face_density = 0.5 * (density + jnp.roll(density, 1, axis=axis))
            else:
                lower_density = jnp.take(density, jnp.asarray([0]), axis=axis)
                upper_density = jnp.take(density, jnp.asarray([-1]), axis=axis)
                left_density = jnp.take(
                    density,
                    jnp.arange(plan.grid.shape[axis] - 1),
                    axis=axis,
                )
                right_density = jnp.take(
                    density,
                    jnp.arange(1, plan.grid.shape[axis]),
                    axis=axis,
                )
                face_density = jnp.concatenate(
                    (
                        lower_density,
                        0.5 * (left_density + right_density),
                        upper_density,
                    ),
                    axis=axis,
                )
            if face_density.shape != face_layout.shape:
                raise RuntimeError(
                    "Prepared acoustic face density has wrong entity shape."
                )
            densities.append(face_density)
            spacings.append(jnp.min(plan.grid.structured_axes[axis].interval_widths))
        wave_speed = jnp.sqrt(plan.bulk_modulus / plan.density)
        inverse_spacing_norm = jnp.sqrt(jnp.sum(1.0 / jnp.asarray(spacings) ** 2))
        stable_dt = 0.45 / (jnp.max(wave_speed) * inverse_spacing_norm)
        key = discretization.key
        acoustic_key = DiscretizationKey(
            "acoustic_system",
            DiscretizationRole.RESIDUAL,
            domain_labels=plan.grid.axis_names,
        )
        self.plan = plan
        self.discretization = discretization
        self.pml = prepared_pml
        self.face_density = tuple(densities)
        self.split_pressure_damping = split_pressure_damping
        self.velocity_damping = velocity_damping
        self.stable_dt = stable_dt
        self.pressure_shape = plan.grid.cells().shape
        self.velocity_shapes = tuple(velocity_shapes)
        self.discretization_bundle = DiscretizationBundle(
            (
                DiscretizationRecord(
                    key,
                    type(discretization).__name__,
                    discretization.prepared_id,
                    numeric_version=discretization.numeric_version,
                ),
                DiscretizationRecord(
                    acoustic_key,
                    "split-field-pml-acoustic-system",
                    plan.plan_id,
                    dependency_key_ids=(key.key_id,),
                ),
            )
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-staggered-acoustics",
                "plan": plan.plan_id,
                "discretization": discretization.prepared_id,
                "pml": None if prepared_pml is None else prepared_pml.prepared_id,
            }
        )

    @property
    def damping(self) -> Array:
        total = self.split_pressure_damping[0]
        for profile in self.split_pressure_damping[1:]:
            total = total + profile
        return total

    def pack(
        self,
        pressure: ArrayLike,
        velocity: Sequence[ArrayLike],
        /,
    ) -> StaggeredAcousticState:
        pressure_ = jnp.asarray(pressure)
        component = pressure_ / float(len(self.plan.grid.axis_names))
        return self.pack_split(
            tuple(component for _ in self.plan.grid.axis_names),
            velocity,
        )

    def pack_split(
        self,
        pressure_components: Sequence[ArrayLike],
        velocity: Sequence[ArrayLike],
        /,
    ) -> StaggeredAcousticState:
        state = StaggeredAcousticState(pressure_components, velocity)
        self.unpack_split(state)
        return state

    def unpack_split(
        self,
        state: StaggeredAcousticState,
        /,
    ) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
        if not isinstance(state, StaggeredAcousticState):
            raise TypeError("Acoustic state must be StaggeredAcousticState.")
        dimension = len(self.plan.grid.axis_names)
        if (
            len(state.pressure_components) != dimension
            or any(
                component.shape != self.pressure_shape
                for component in state.pressure_components
            )
            or len(state.velocity) != len(self.velocity_shapes)
        ):
            raise ValueError("Pressure/velocity fields do not match acoustic spaces.")
        if any(
            value.shape != shape
            for value, shape in zip(state.velocity, self.velocity_shapes, strict=True)
        ):
            raise ValueError("Velocity field does not match its face entity layout.")
        return state.pressure_components, state.velocity

    def unpack(
        self,
        state: StaggeredAcousticState,
        /,
    ) -> tuple[Array, tuple[Array, ...]]:
        pressure_components, velocity = self.unpack_split(state)
        pressure = pressure_components[0]
        for component in pressure_components[1:]:
            pressure = pressure + component
        return pressure, velocity

    def _source(self, time: Array, args: Any, /) -> Array:
        if self.plan.source is None:
            return jnp.zeros(self.pressure_shape)
        source = jnp.asarray(
            self.plan.source(jnp.asarray(time), self.plan.grid.points, args)
        )
        if source.shape == (self.plan.grid.size,):
            source = source.reshape(self.pressure_shape)
        if source.shape != self.pressure_shape:
            raise ValueError("Acoustic source must return one value per cell.")
        return source

    def drift(
        self,
        time: Array,
        state: StaggeredAcousticState,
        args: Any,
    ) -> StaggeredAcousticState:
        pressure_components, velocity = self.unpack_split(state)
        pressure = state.pressure
        source = self._source(jnp.asarray(time), args) / float(len(pressure_components))
        component_derivative = []
        velocity_derivative = []
        for index, axis in enumerate(self.plan.grid.axis_names):
            divergence = self.discretization.operator(f"div_{axis}").mv(velocity[index])
            component_derivative.append(
                -self.plan.bulk_modulus * divergence
                - self.split_pressure_damping[index] * pressure_components[index]
                + source
            )
            gradient = self.discretization.operator(f"grad_{axis}").mv(pressure)
            velocity_derivative.append(
                -gradient / self.face_density[index]
                - self.velocity_damping[index] * velocity[index]
            )
        return self.pack_split(component_derivative, velocity_derivative)

    def leapfrog_step(
        self,
        time: Array,
        state: StaggeredAcousticState,
        step_size: ArrayLike,
        args: Any = None,
    ) -> StaggeredAcousticState:
        dt = jnp.asarray(step_size)
        dt = eqx.error_if(
            dt,
            ~jnp.isfinite(dt) | (dt <= 0.0),
            "Acoustic leapfrog step_size must be finite and positive.",
        )
        pressure_components, velocity = self.unpack_split(state)
        pressure = state.pressure
        half_step = 0.5 * dt
        velocity_half = []
        for index, axis in enumerate(self.plan.grid.axis_names):
            gradient = self.discretization.operator(f"grad_{axis}").mv(pressure)
            velocity_half.append(
                _integrating_factor_step(
                    velocity[index],
                    -gradient / self.face_density[index],
                    self.velocity_damping[index],
                    half_step,
                )
            )
        source = self._source(jnp.asarray(time) + half_step, args) / float(
            len(pressure_components)
        )
        pressure_new_components = []
        for index, axis in enumerate(self.plan.grid.axis_names):
            divergence = self.discretization.operator(f"div_{axis}").mv(
                velocity_half[index]
            )
            pressure_new_components.append(
                _integrating_factor_step(
                    pressure_components[index],
                    -self.plan.bulk_modulus * divergence + source,
                    self.split_pressure_damping[index],
                    dt,
                )
            )
        pressure_new = pressure_new_components[0]
        for component in pressure_new_components[1:]:
            pressure_new = pressure_new + component
        velocity_new = []
        for index, axis in enumerate(self.plan.grid.axis_names):
            gradient = self.discretization.operator(f"grad_{axis}").mv(pressure_new)
            velocity_new.append(
                _integrating_factor_step(
                    velocity_half[index],
                    -gradient / self.face_density[index],
                    self.velocity_damping[index],
                    half_step,
                )
            )
        return self.pack_split(pressure_new_components, velocity_new)

    def energy(self, state: StaggeredAcousticState, /) -> Array:
        pressure, velocity = self.unpack(state)
        total = 0.5 * jnp.sum(
            self.plan.grid.cells().measure
            * jnp.real(pressure * jnp.conj(pressure))
            / self.plan.bulk_modulus
        )
        for index, axis in enumerate(self.plan.grid.axis_names):
            total = total + 0.5 * jnp.sum(
                self.plan.grid.faces(axis).measure
                * self.face_density[index]
                * jnp.real(velocity[index] * jnp.conj(velocity[index]))
            )
        return total

    def observe(self, state: StaggeredAcousticState, /) -> Array:
        if self.plan.sensor_indices is None:
            raise ValueError("Acoustic plan has no sensor indices.")
        pressure, _ = self.unpack(state)
        indices = tuple(
            self.plan.sensor_indices[:, axis] for axis in range(pressure.ndim)
        )
        return pressure[indices]


__all__ = [
    "PreparedSplitFieldPML",
    "PreparedStaggeredAcoustics",
    "SplitFieldPMLPlan",
    "StaggeredAcousticPlan",
    "StaggeredAcousticState",
]
