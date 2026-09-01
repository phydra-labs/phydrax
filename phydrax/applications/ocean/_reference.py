#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


if TYPE_CHECKING:
    from ...discretization.finite_volume import FiniteVolumeDiscretization
    from ...equations import MACBuoyancyLaw


class OceanAxisConvention(StrictModule, NonTrainableState):
    """Three-dimensional Cartesian ocean axis and gravity convention."""

    vertical_axis: int = eqx.field(static=True)
    positive_up: bool = eqx.field(static=True)
    coordinate_units: str = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertical_axis: int = 2,
        /,
        *,
        positive_up: bool = True,
        coordinate_units: str = "m",
    ):
        vertical = int(vertical_axis)
        units = str(coordinate_units)
        if vertical not in (0, 1, 2):
            raise ValueError("Ocean vertical_axis must be zero, one, or two.")
        if not units:
            raise ValueError("Ocean coordinate_units must be non-empty.")
        self.vertical_axis = vertical
        self.positive_up = bool(positive_up)
        self.coordinate_units = units
        self.convention_id = canonical_fingerprint(
            {
                "kind": "ocean-axis-convention",
                "dimension": 3,
                "vertical_axis": vertical,
                "positive_up": bool(positive_up),
                "coordinate_units": units,
            }
        )

    @property
    def horizontal_axes(self) -> tuple[int, int]:
        axes = tuple(axis for axis in range(3) if axis != self.vertical_axis)
        return axes[0], axes[1]

    @property
    def surface_index(self) -> int:
        return -1 if self.positive_up else 0

    def gravity(self, magnitude: ArrayLike, /) -> Array:
        value = float(jnp.asarray(magnitude))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("Ocean gravity magnitude must be finite and positive.")
        output = jnp.zeros((3,), dtype=float)
        sign = -1.0 if self.positive_up else 1.0
        return output.at[self.vertical_axis].set(sign * value)

    def validate_discretization(
        self,
        discretization: FiniteVolumeDiscretization,
        /,
    ) -> None:
        if len(discretization.cell_shape) != 3:
            raise ValueError(
                "Cartesian ocean modeling requires a three-dimensional grid."
            )
        grid = discretization.grid
        vertical = grid.structured_axes[self.vertical_axis]
        if vertical.periodic:
            raise ValueError("Ocean vertical axes must be bounded, not periodic.")


class LinearSeawaterReference(StrictModule, NonTrainableState):
    """Linear temperature-salinity Boussinesq reference state."""

    reference_density: float = eqx.field(static=True)
    heat_capacity: float = eqx.field(static=True)
    gravity_magnitude: float = eqx.field(static=True)
    reference_temperature: float = eqx.field(static=True)
    reference_salinity: float = eqx.field(static=True)
    thermal_expansion: float = eqx.field(static=True)
    haline_contraction: float = eqx.field(static=True)
    temperature_name: str = eqx.field(static=True)
    salinity_name: str = eqx.field(static=True)
    reference_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        reference_density: float = 1027.0,
        heat_capacity: float = 3990.0,
        gravity_magnitude: float = 9.81,
        reference_temperature: float = 10.0,
        reference_salinity: float = 35.0,
        thermal_expansion: float = 2.0e-4,
        haline_contraction: float = 7.6e-4,
        temperature_name: str = "temperature",
        salinity_name: str = "salinity",
    ):
        values = tuple(
            float(value)
            for value in (
                reference_density,
                heat_capacity,
                gravity_magnitude,
                reference_temperature,
                reference_salinity,
                thermal_expansion,
                haline_contraction,
            )
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("Linear seawater reference parameters must be finite.")
        density, capacity, gravity, temperature, salinity, alpha, beta = values
        if density <= 0.0 or capacity <= 0.0 or gravity <= 0.0:
            raise ValueError(
                "Reference density, heat capacity, and gravity must be positive."
            )
        if alpha < 0.0 or beta < 0.0:
            raise ValueError(
                "Thermal expansion and haline contraction must be nonnegative."
            )
        temperature_field = str(temperature_name)
        salinity_field = str(salinity_name)
        if (
            not temperature_field
            or not salinity_field
            or temperature_field == salinity_field
        ):
            raise ValueError("Ocean temperature and salinity names must be distinct.")
        self.reference_density = density
        self.heat_capacity = capacity
        self.gravity_magnitude = gravity
        self.reference_temperature = temperature
        self.reference_salinity = salinity
        self.thermal_expansion = alpha
        self.haline_contraction = beta
        self.temperature_name = temperature_field
        self.salinity_name = salinity_field
        self.reference_id = canonical_fingerprint(
            {
                "kind": "linear-seawater-reference",
                "rho0": density,
                "cp": capacity,
                "gravity": gravity,
                "temperature_reference": temperature,
                "salinity_reference": salinity,
                "thermal_expansion": alpha,
                "haline_contraction": beta,
                "temperature_name": temperature_field,
                "salinity_name": salinity_field,
            }
        )

    @property
    def field_names(self) -> tuple[str, str]:
        return tuple(sorted((self.temperature_name, self.salinity_name)))

    def density_anomaly(
        self,
        temperature: ArrayLike,
        salinity: ArrayLike,
        /,
    ) -> Array:
        temperature_ = jnp.asarray(temperature)
        salinity_ = jnp.asarray(salinity, dtype=temperature_.dtype)
        if salinity_.shape != temperature_.shape:
            raise ValueError(
                "Temperature and salinity arrays must have identical shapes."
            )
        return self.reference_density * (
            -self.thermal_expansion * (temperature_ - self.reference_temperature)
            + self.haline_contraction * (salinity_ - self.reference_salinity)
        )

    def temperature_flux_from_heat_flux(self, heat_flux: ArrayLike, /) -> Array:
        return jnp.asarray(heat_flux) / (self.reference_density * self.heat_capacity)

    def buoyancy_law(
        self,
        axes: OceanAxisConvention,
        /,
    ) -> MACBuoyancyLaw:
        from ...equations import MACBuoyancyLaw

        if not isinstance(axes, OceanAxisConvention):
            raise TypeError("axes must be OceanAxisConvention.")
        return MACBuoyancyLaw(
            axes.gravity(self.gravity_magnitude),
            {
                self.temperature_name: -self.thermal_expansion,
                self.salinity_name: self.haline_contraction,
            },
            references={
                self.temperature_name: self.reference_temperature,
                self.salinity_name: self.reference_salinity,
            },
            enforce_exchange=True,
        )


__all__ = ["LinearSeawaterReference", "OceanAxisConvention"]
