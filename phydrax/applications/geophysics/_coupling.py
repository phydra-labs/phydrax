#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._array_archive import (
    pack_array_tree,
    read_array_archive,
    unpack_array_tree,
    write_array_archive,
)
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...discretization import DiscreteFieldSpace, DiscreteMeasure, EntityDofLayout
from ...linalg import ArraySpace
from ...solver._partitioned_coupling_types import (
    AbstractCouplingSubsystem,
    CouplingPort,
    CouplingQuantity,
    CouplingSubsystemCapabilities,
    CouplingSubsystemResult,
)
from ...units import derived_unit, JOULE, KELVIN, KILOGRAM, METER, SECOND
from ..ocean._boussinesq import PreparedCartesianBoussinesqOcean
from ..ocean._hydrostatic import PreparedHydrostaticOcean
from ..ocean._hydrostatic_step import (
    HydrostaticContinuationState,
    HydrostaticIMEXMidpointMethod,
)
from ..ocean._step import OceanBoussinesqContinuationState, OceanBoussinesqSSPRK33Method
from ._quantities import GeophysicalQuantity


_HEAT_REFERENCE = "constant-cp-relative-to-273.15K"
_HEAT = CouplingQuantity(
    "enthalpy_per_area",
    derived_unit("J/m²", ((JOULE, 1), (METER, -2))),
    reference_configuration=_HEAT_REFERENCE,
)
_WATER = CouplingQuantity(
    "water_mass_per_area",
    derived_unit("kg/m²", ((KILOGRAM, 1), (METER, -2))),
)
_TEMPERATURE = CouplingQuantity("temperature", KELVIN)
_AREA_UNIT = derived_unit("m²", ((METER, 2),))
_CAPABILITIES = CouplingSubsystemCapabilities(
    jit=True,
    differentiable=True,
    deterministic_replay=True,
    fixed_topology=True,
    supports_endpoint=True,
    supports_waveform=False,
)


def geophysical_coupling_quantity(quantity: GeophysicalQuantity, /) -> CouplingQuantity:
    """Bind physical semantics without importing geophysics into the solver."""
    if not isinstance(quantity, GeophysicalQuantity):
        raise TypeError("quantity must be GeophysicalQuantity.")
    return CouplingQuantity(
        quantity.quantity_kind,
        quantity.unit,
        reference_configuration=quantity.reference_configuration,
        sign_convention=quantity.sign_convention,
    )


def coupling_surface_field(cell_area, support_id: str, /):
    """Flatten a physical surface into native cell-average coordinates and measure."""
    area = np.asarray(cell_area)
    if not area.size or np.any(~np.isfinite(area)) or np.any(area <= 0):
        raise ValueError("Coupled surface areas must be finite and strictly positive.")
    measure = DiscreteMeasure(
        "surface-area", support_id, f"{support_id}/cells", area.reshape(-1)
    )
    layout = EntityDofLayout(measure.entity_set_id, area.size, area.size)
    space = ArraySpace(
        (area.size,), dtype=area.dtype, space_id=f"{measure.measure_id}/values"
    )
    field = DiscreteFieldSpace(
        "surface-cell-average",
        support_id,
        layout,
        space,
        representation="cell_average",
    )
    return field, measure


def _port(name, direction, field, measure, quantity, *, integrated=False):
    return CouplingPort(
        name,
        direction,
        field.vector_space,
        field_space=field,
        measure=measure,
        measure_unit=_AREA_UNIT,
        quantity=quantity,
        temporal_kind="interval_integral" if integrated else "instantaneous",
        reference_scale=1.0,
    )


def _positive(value, name):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return number


class SlabReservoirState(StrictModule):
    """Physical enthalpy (J/m², relative to 273.15 K) and water mass (kg/m²)."""

    enthalpy: Array
    water_mass: Array


class SlabReservoir(AbstractCouplingSubsystem):
    """Finite heat/water reservoir proposing the sole outgoing interface flux.

    Surface heat capacity is dry_heat_capacity + water_heat_capacity * water_mass.
    Heat includes both sensible exchange and enthalpy carried by outgoing water.
    No clipping hides depletion: an exhausted or nonphysical reservoir rejects
    the complete window. The law is first-order frozen at the window start.
    """

    field: DiscreteFieldSpace
    measure: DiscreteMeasure
    input_ports: tuple[CouplingPort, ...]
    output_ports: tuple[CouplingPort, ...]
    capabilities: CouplingSubsystemCapabilities
    dry_heat_capacity: float = eqx.field(static=True)
    water_heat_capacity: float = eqx.field(static=True)
    conductance: float = eqx.field(static=True)
    water_rate: float = eqx.field(static=True)
    subsystem_id: str = eqx.field(static=True)
    discretization_bundle_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        field,
        measure,
        /,
        *,
        dry_heat_capacity=1.0e6,
        water_heat_capacity=4184.0,
        conductance=10.0,
        water_rate=0.0,
        name="slab",
    ):
        capacity = _positive(dry_heat_capacity, "dry_heat_capacity")
        cp = _positive(water_heat_capacity, "water_heat_capacity")
        conductance_ = float(conductance)
        rate = float(water_rate)
        if (
            not np.isfinite(conductance_)
            or conductance_ < 0
            or not np.isfinite(rate)
            or rate < 0
        ):
            raise ValueError(
                "Slab conductance and outgoing water_rate must be finite and nonnegative."
            )
        identity = canonical_fingerprint(
            {
                "kind": "slab-heat-water-reservoir",
                "name": name,
                "field": field.field_space_id,
                "measure": measure.measure_id,
                "dry_heat_capacity": capacity,
                "water_heat_capacity": cp,
                "conductance": conductance_,
                "water_rate": rate,
                "heat_reference": _HEAT_REFERENCE,
            }
        )
        self.field, self.measure = field, measure
        self.dry_heat_capacity, self.water_heat_capacity = capacity, cp
        self.conductance, self.water_rate = conductance_, rate
        self.subsystem_id = f"{name}:{identity}"
        self.discretization_bundle_id = field.support_id
        self.capabilities = _CAPABILITIES
        self.input_ports = (
            _port(f"{identity}/temperature", "input", field, measure, _TEMPERATURE),
        )
        self.output_ports = (
            _port(f"{identity}/heat", "output", field, measure, _HEAT, integrated=True),
            _port(f"{identity}/water", "output", field, measure, _WATER, integrated=True),
        )

    def initialize(self, temperature, water_mass, /):
        shape = (self.field.vector_space.size,)
        dtype = self.measure.weights.dtype
        temperature_ = jnp.broadcast_to(jnp.asarray(temperature, dtype=dtype), shape)
        water = jnp.broadcast_to(jnp.asarray(water_mass, dtype=dtype), shape)
        if np.any(~np.isfinite(np.asarray(temperature_))) or np.any(
            np.asarray(temperature_) <= 0
        ):
            raise ValueError("Slab absolute temperature must be finite and positive.")
        if np.any(~np.isfinite(np.asarray(water))) or np.any(np.asarray(water) < 0):
            raise ValueError("Slab water mass must be finite and nonnegative.")
        return SlabReservoirState(
            (self.dry_heat_capacity + self.water_heat_capacity * water)
            * (temperature_ - 273.15),
            water,
        )

    def temperature(self, state, /):
        return 273.15 + state.enthalpy / (
            self.dry_heat_capacity + self.water_heat_capacity * state.water_mass
        )

    def advance_window(self, window, start_state, inputs, args, /):
        del args
        temperature = self.temperature(start_state)
        water = jnp.full_like(start_state.water_mass, self.water_rate) * window.size
        heat = self.conductance * (temperature - inputs[0]) * window.size
        heat = heat + self.water_heat_capacity * water * (temperature - 273.15)
        candidate = SlabReservoirState(
            start_state.enthalpy - heat, start_state.water_mass - water
        )
        successful = jnp.all(candidate.water_mass >= 0) & jnp.all(
            self.temperature(candidate) > 0
        )
        return CouplingSubsystemResult(
            candidate,
            (heat, water),
            successful=successful,
            status=jnp.where(successful, 0, 1),
            work=1,
        )


class HydrostaticOceanCouplingSubsystem(AbstractCouplingSubsystem):
    """Heat/freshwater boundary adapter retaining the native ocean continuation.

    Heat is constant-cp potential enthalpy, rho0 * cp * CT-volume inventory,
    not full in-situ seawater enthalpy. Signed freshwater changes real free-surface
    volume without transporting salt; its complete heat exchange is supplied
    separately. Native mosaic traces and lateral boundaries are not intercepted.
    This adapter accepts one prepared ocean, not a mosaic.
    """

    method: HydrostaticIMEXMidpointMethod
    field: DiscreteFieldSpace
    measure: DiscreteMeasure
    input_ports: tuple[CouplingPort, ...]
    output_ports: tuple[CouplingPort, ...]
    capabilities: CouplingSubsystemCapabilities
    heat_capacity: float = eqx.field(static=True)
    freshwater_density: float = eqx.field(static=True)
    subsystem_id: str = eqx.field(static=True)
    discretization_bundle_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        ocean: PreparedHydrostaticOcean,
        /,
        *,
        heat_capacity=3990.0,
        freshwater_density=1000.0,
        name="hydrostatic-ocean",
    ):
        if not isinstance(ocean, PreparedHydrostaticOcean):
            raise TypeError(
                "Hydrostatic coupling requires a native PreparedHydrostaticOcean, not a mosaic."
            )
        if np.any(np.asarray(ocean.plan.freshwater.rate) != 0):
            raise ValueError(
                "Coupling must be the authoritative freshwater source; prepare zero background rate."
            )
        if ocean.plan.freshwater.absolute_salinity != 0:
            raise ValueError(
                "Freshwater coupling requires zero incoming absolute salinity."
            )
        self.heat_capacity = _positive(heat_capacity, "heat_capacity")
        self.freshwater_density = _positive(freshwater_density, "freshwater_density")
        self.method = HydrostaticIMEXMidpointMethod(ocean)
        self.field, self.measure = coupling_surface_field(
            ocean.geometry.cell_area, ocean.geometry.geometry_id
        )
        identity = canonical_fingerprint(
            {
                "kind": "hydrostatic-surface-coupling",
                "name": name,
                "ocean": ocean.prepared_id,
                "method": self.method.method_id,
                "cp": self.heat_capacity,
                "freshwater_density": self.freshwater_density,
                "heat_reference": _HEAT_REFERENCE,
            }
        )
        self.subsystem_id = f"{name}:{identity}"
        self.discretization_bundle_id = ocean.geometry.geometry_id
        self.capabilities = _CAPABILITIES
        self.input_ports = (
            _port(
                f"{identity}/heat",
                "input",
                self.field,
                self.measure,
                _HEAT,
                integrated=True,
            ),
            _port(
                f"{identity}/water",
                "input",
                self.field,
                self.measure,
                _WATER,
                integrated=True,
            ),
        )
        self.output_ports = (
            _port(
                f"{identity}/temperature",
                "output",
                self.field,
                self.measure,
                _TEMPERATURE,
            ),
        )

    def temperature(self, state: HydrostaticContinuationState, /):
        ocean = self.method.ocean
        volume = ocean.geometry.metric_epoch(state.state.eta).cell_volume[..., -1]
        inventory = state.state.tracer_inventory["conservative_temperature"][..., -1]
        return (273.15 + inventory / volume).reshape(-1)

    def advance_window(self, window, start_state, inputs, args, /):
        ocean = self.method.ocean
        shape = ocean.geometry.horizontal_shape
        heat, water = (value.reshape(shape) for value in inputs)
        freshwater_volume = water / self.freshwater_density
        rate = freshwater_volume / window.size
        method = eqx.tree_at(
            lambda value: value.ocean.plan.freshwater.rate, self.method, rate
        )
        result = method.step(window.index, window.start, start_state, window.size, args)
        candidate = result.candidate_state
        # The native freshwater step already advected its explicitly declared
        # incoming CT. Correct only the remaining potential-enthalpy contribution.
        coefficient = ocean.plan.reference_density * self.heat_capacity
        correction = ocean.geometry.cell_area * (
            heat / coefficient
            - freshwater_volume * ocean.plan.freshwater.conservative_temperature
        )
        name = "conservative_temperature"
        inventory = candidate.state.tracer_inventory[name].at[..., -1].add(correction)
        total = jnp.sum(correction)
        candidate = eqx.tree_at(
            lambda value: (
                value.state.tracer_inventory[name],
                value.ledger.tracer_change[name],
                value.ledger.tracer_source[name],
            ),
            candidate,
            (
                inventory,
                candidate.ledger.tracer_change[name] + total,
                candidate.ledger.tracer_source[name] + total,
            ),
        )
        view = ocean.view(candidate.state)
        successful = (
            result.successful & view.eos_valid & view.eos_successful & view.eos_finite
        )
        successful = successful & jnp.all(self.temperature(candidate) > 0)
        return CouplingSubsystemResult(
            candidate,
            (self.temperature(candidate),),
            successful=successful,
            status=jnp.where(successful, 0, 1),
            residual_norm=result.residual,
            iterations=result.iterations,
            work=result.work,
        )


class BoussinesqOceanCouplingSubsystem(AbstractCouplingSubsystem):
    """Rigid-lid heat and optional spatially uniform Cartesian stress adapter.

    No freshwater port exists: virtual salt flux is not freshwater physics.
    Heat enters the native conservative scalar boundary law and its accepted
    quadrature; optional stress impulses enter native MAC surface forcing.
    """

    method: OceanBoussinesqSSPRK33Method
    field: DiscreteFieldSpace
    measure: DiscreteMeasure
    input_ports: tuple[CouplingPort, ...]
    output_ports: tuple[CouplingPort, ...]
    capabilities: CouplingSubsystemCapabilities
    temperature_index: int = eqx.field(static=True)
    stress: bool = eqx.field(static=True)
    subsystem_id: str = eqx.field(static=True)
    discretization_bundle_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        ocean: PreparedCartesianBoussinesqOcean,
        /,
        *,
        stress=False,
        name="boussinesq-ocean",
    ):
        if not isinstance(ocean, PreparedCartesianBoussinesqOcean):
            raise TypeError(
                "Boussinesq coupling requires a native prepared rigid-lid ocean."
            )
        if ocean.plan.temperature_surface_flux.function is not None:
            raise ValueError(
                "Coupling cannot replace a separately owned dynamic heat boundary."
            )
        if np.any(np.asarray(ocean.plan.temperature_surface_flux.value) != 0):
            raise ValueError("Coupling requires an initially zero heat boundary.")
        if ocean.plan.salinity_surface_flux.function is not None or np.any(
            np.asarray(ocean.plan.salinity_surface_flux.value) != 0
        ):
            raise ValueError(
                "Virtual salinity flux cannot masquerade as coupled freshwater."
            )
        if stress and (
            ocean.plan.surface_stress_function is not None
            or np.any(np.asarray(ocean.plan.surface_stress) != 0)
        ):
            raise ValueError("Coupling must be the sole owner of prescribed stress.")
        axes = ocean.plan.axes
        grid = ocean.operators.discretization.grid
        horizontal_shape = tuple(grid.shape[axis] for axis in axes.horizontal_axes)
        area = np.ones(horizontal_shape)
        for local_axis, axis in enumerate(axes.horizontal_axes):
            reshape = [1, 1]
            reshape[local_axis] = horizontal_shape[local_axis]
            area = area * np.asarray(grid.structured_axes[axis].interval_widths).reshape(
                reshape
            )
        self.field, self.measure = coupling_surface_field(area, ocean.prepared_id)
        self.method = OceanBoussinesqSSPRK33Method(ocean)
        self.temperature_index = ocean.transport.layout.field_names.index(
            ocean.plan.reference.temperature_name
        )
        self.stress = bool(stress)
        identity = canonical_fingerprint(
            {
                "kind": "rigid-lid-ocean-coupling",
                "name": name,
                "method": self.method.method_id,
                "stress": self.stress,
                "heat_reference": _HEAT_REFERENCE,
            }
        )
        self.subsystem_id = f"{name}:{identity}"
        self.discretization_bundle_id = ocean.prepared_id
        self.capabilities = _CAPABILITIES
        ports = [
            _port(
                f"{identity}/heat",
                "input",
                self.field,
                self.measure,
                _HEAT,
                integrated=True,
            )
        ]
        if self.stress:
            impulse_unit = derived_unit(
                "kg/(m*s)", ((KILOGRAM, 1), (METER, -1), (SECOND, -1))
            )
            for axis in axes.horizontal_axes:
                quantity = CouplingQuantity(
                    "impulse_per_area",
                    impulse_unit,
                    reference_configuration=f"Cartesian-component-{axis}",
                )
                ports.append(
                    _port(
                        f"{identity}/impulse-{axis}",
                        "input",
                        self.field,
                        self.measure,
                        quantity,
                        integrated=True,
                    )
                )
        self.input_ports = tuple(ports)
        self.output_ports = (
            _port(
                f"{identity}/temperature",
                "output",
                self.field,
                self.measure,
                _TEMPERATURE,
            ),
        )

    def temperature(self, state: OceanBoussinesqContinuationState, /):
        ocean = self.method.ocean
        _, scalars = ocean.dynamics.unpack_state(state.coordinates)
        temperature = scalars[ocean.plan.reference.temperature_name]
        return (
            273.15
            + jnp.take(
                temperature,
                ocean.plan.axes.surface_index,
                axis=ocean.plan.axes.vertical_axis,
            )
        ).reshape(-1)

    def advance_window(self, window, start_state, inputs, args, /):
        ocean = self.method.ocean
        axes = ocean.plan.axes
        vertical, side = axes.vertical_axis, 1 if axes.surface_index == -1 else 0
        shape = tuple(
            ocean.operators.discretization.grid.shape[axis]
            for axis in axes.horizontal_axes
        )
        coefficient = (
            ocean.plan.reference.reference_density * ocean.plan.reference.heat_capacity
        )
        # Native scalar flux is outward loss; physical coupling heat is inward gain.
        flux = -inputs[0].reshape(shape) / (coefficient * window.size)
        index = self.temperature_index
        transport = eqx.tree_at(
            lambda value: value.boundaries.conditions[index][vertical][side].value,
            ocean.transport,
            flux,
        )
        method = eqx.tree_at(
            lambda value: (value.ocean.transport, value.ocean.dynamics.transport),
            self.method,
            (transport, transport),
        )
        uniform = jnp.asarray(True)
        if self.stress:
            vector = jnp.zeros((3,), dtype=inputs[0].dtype)
            for local, axis in enumerate(axes.horizontal_axes):
                impulse = inputs[local + 1]
                uniform = uniform & jnp.all(impulse == impulse[0])
                vector = vector.at[axis].set(impulse[0] / window.size)
            method = eqx.tree_at(
                lambda value: value.ocean.dynamics.ocean_forcing.surface_stress,
                method,
                vector,
            )
        result = method.step(window.index, window.start, start_state, window.size, args)
        successful = result.successful & uniform
        return CouplingSubsystemResult(
            result.candidate_state,
            (self.temperature(result.candidate_state),),
            successful=successful,
            status=jnp.where(successful, 0, 1),
            residual_norm=result.residual,
            iterations=result.iterations,
            work=result.work,
        )


def write_geophysical_coupling_checkpoint(path: str | Path, prepared, state, /):
    """Archive an accepted coupling boundary including every native continuation."""
    if state.graph_id != prepared.graph_id:
        raise ValueError("Coupling checkpoint state and graph identities differ.")
    arrays = {}
    specification = pack_array_tree("state", state, arrays)
    return write_array_archive(
        path,
        manifest={
            "kind": "geophysical-coupling-checkpoint",
            "graph_id": prepared.graph_id,
            "plan_id": prepared.plan_id,
            "state": specification,
        },
        arrays=arrays,
    )


def read_geophysical_coupling_checkpoint(path: str | Path, prepared, /):
    manifest, arrays = read_array_archive(path)
    if manifest.get("kind") != "geophysical-coupling-checkpoint":
        raise ValueError("Not a geophysical coupling checkpoint.")
    if (
        manifest.get("graph_id") != prepared.graph_id
        or manifest.get("plan_id") != prepared.plan_id
    ):
        raise ValueError(
            "Coupling checkpoint physical graph or native plan identity mismatch."
        )
    return unpack_array_tree(manifest["state"], arrays, prepared.reference_state)


__all__ = [
    "SlabReservoir",
    "SlabReservoirState",
    "HydrostaticOceanCouplingSubsystem",
    "BoussinesqOceanCouplingSubsystem",
    "coupling_surface_field",
    "geophysical_coupling_quantity",
    "write_geophysical_coupling_checkpoint",
    "read_geophysical_coupling_checkpoint",
]
