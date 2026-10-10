"""Device-array views of validated fabrication and free-volume records.

The host records hold static Python floats; these views lift the same values to
float64 device leaves so one compiled fuel step can serve rods that differ only
in fabrication data. Views neither revalidate inputs nor change physics: the
host record remains the scientific identity and the validation boundary.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ..._strict import StrictModule
from ...discretization.fem import PreparedFiniteElementPointInterpolation
from ...linalg import ArraySpace
from ._free_volume import FreeVolumeModel
from ._identity import CladdingAlloy, CladdingHeatTreatment, FuelForm
from ._specification import (
    CladdingSpecification,
    FuelRodSpecification,
    PelletSpecification,
    RodSpecification,
)


def _device_float(value: float | tuple[float, ...], /) -> Array:
    return jnp.asarray(value, dtype=jnp.float64)


class NumericalPelletSpecification(StrictModule):
    """Device view of `PelletSpecification` (same fields and SI units)."""

    form: FuelForm = eqx.field(static=True)
    radius: Array
    height: Array
    dish_volume: Array
    chamfer_volume: Array
    density_fraction: Array
    open_porosity_fraction: Array
    grain_radius: Array
    roughness: Array
    uranium_235_weight_fraction: Array
    oxygen_to_metal_ratio: Array
    resintering_density_change: Array

    def __init__(self, source: PelletSpecification, /) -> None:
        self.form = source.form
        self.radius = _device_float(source.radius)
        self.height = _device_float(source.height)
        self.dish_volume = _device_float(source.dish_volume)
        self.chamfer_volume = _device_float(source.chamfer_volume)
        self.density_fraction = _device_float(source.density_fraction)
        self.open_porosity_fraction = _device_float(source.open_porosity_fraction)
        self.grain_radius = _device_float(source.grain_radius)
        self.roughness = _device_float(source.roughness)
        self.uranium_235_weight_fraction = _device_float(
            source.uranium_235_weight_fraction
        )
        self.oxygen_to_metal_ratio = _device_float(source.oxygen_to_metal_ratio)
        self.resintering_density_change = _device_float(source.resintering_density_change)

    @property
    def removed_volume_fraction(self) -> Array:
        return (self.dish_volume + self.chamfer_volume) / (
            jnp.pi * self.radius**2 * self.height
        )


class NumericalCladdingSpecification(StrictModule):
    """Device view of `CladdingSpecification` (same fields and SI units)."""

    alloy: CladdingAlloy = eqx.field(static=True)
    heat_treatment: CladdingHeatTreatment = eqx.field(static=True)
    inner_radius: Array
    outer_radius: Array
    cold_work: Array
    roughness: Array
    initial_oxide_thickness: Array
    initial_hydrogen_ppm: Array

    def __init__(self, source: CladdingSpecification, /) -> None:
        self.alloy, self.heat_treatment = source.alloy, source.heat_treatment
        self.inner_radius = _device_float(source.inner_radius)
        self.outer_radius = _device_float(source.outer_radius)
        self.cold_work = _device_float(source.cold_work)
        self.roughness = _device_float(source.roughness)
        self.initial_oxide_thickness = _device_float(source.initial_oxide_thickness)
        self.initial_hydrogen_ppm = _device_float(source.initial_hydrogen_ppm)


class NumericalRodSpecification(StrictModule):
    """Device view of `RodSpecification` (same fields and SI units)."""

    stack_length: Array
    plenum_length: Array
    spring_volume: Array
    spring_preload: Array
    spring_stiffness: Array
    fill_pressure: Array
    fill_temperature: Array
    fill_gas_mole_fractions: Array

    def __init__(self, source: RodSpecification, /) -> None:
        self.stack_length = _device_float(source.stack_length)
        self.plenum_length = _device_float(source.plenum_length)
        self.spring_volume = _device_float(source.spring_volume)
        self.spring_preload = _device_float(source.spring_preload)
        self.spring_stiffness = _device_float(source.spring_stiffness)
        self.fill_pressure = _device_float(source.fill_pressure)
        self.fill_temperature = _device_float(source.fill_temperature)
        self.fill_gas_mole_fractions = _device_float(source.fill_gas_mole_fractions)


class FuelNumericalSpecification(StrictModule):
    """Device view of `FuelRodSpecification`; keeps the host specification ID."""

    pellet: NumericalPelletSpecification
    cladding: NumericalCladdingSpecification
    rod: NumericalRodSpecification
    specification_id: str = eqx.field(static=True)

    def __init__(self, source: FuelRodSpecification, /) -> None:
        self.pellet = NumericalPelletSpecification(source.pellet)
        self.cladding = NumericalCladdingSpecification(source.cladding)
        self.rod = NumericalRodSpecification(source.rod)
        self.specification_id = source.specification_id

    @property
    def cold_gap(self) -> Array:
        return self.cladding.inner_radius - self.pellet.radius


class NumericalFreeVolumeModel(StrictModule):
    """Device view of `FreeVolumeModel` (same fields and SI units)."""

    plenum_reference_length: Array
    spring_volume: Array
    dish_chamfer_volume_per_length: Array
    open_porosity_fraction: Array
    fuel_radius: Array

    def __init__(self, source: FreeVolumeModel, /) -> None:
        self.plenum_reference_length = _device_float(source.plenum_reference_length)
        self.spring_volume = _device_float(source.spring_volume)
        self.dish_chamfer_volume_per_length = _device_float(
            source.dish_chamfer_volume_per_length
        )
        self.open_porosity_fraction = _device_float(source.open_porosity_fraction)
        self.fuel_radius = _device_float(source.fuel_radius)


class NumericalPointInterpolation(StrictModule):
    """Device view of a prepared FE point route consumed by the route kernels."""

    weights: Array
    dof_routes: Array
    reference_positions: Array
    dof_reference_positions: Array
    field_space: ArraySpace
    field_name: str = eqx.field(static=True)
    derivative_axis: int | None = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    role_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self, source: PreparedFiniteElementPointInterpolation, role_id: str, /
    ) -> None:
        if not role_id:
            raise ValueError(
                "A numerical FE route requires an explicit structural role identity."
            )
        self.weights, self.dof_routes = source.weights, source.dof_routes
        self.reference_positions = source.reference_positions
        self.dof_reference_positions = source.dof_reference_positions
        self.field_space = source.field_space
        self.field_name, self.derivative_axis = source.field_name, source.derivative_axis
        self.tolerance, self.role_id = source.tolerance, role_id
        self.prepared_id = source.prepared_id

    @property
    def attachment_count(self) -> int:
        return self.dof_routes.shape[0]

    @property
    def value_shape(self) -> tuple[int, ...]:
        return self.field_space.shape[1:]

    def interpolate(self, coefficients: ArrayLike, /) -> Array:
        return PreparedFiniteElementPointInterpolation.interpolate(self, coefficients)

    def transpose_scatter(self, point_dual: ArrayLike, /) -> Array:
        return PreparedFiniteElementPointInterpolation.transpose_scatter(self, point_dual)
