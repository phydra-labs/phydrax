#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._interpolation import apply_gather_stencil, rectilinear_stencil
from ..._strict import StrictModule
from ...equations._chemical_thermodynamics import UNIVERSAL_GAS_CONSTANT
from ...equations._homogeneous_thermodynamics import (
    HomogeneousHelmholtzPlan,
    ZeroResidualHelmholtzTerm,
)
from ...equations._peng_robinson import (
    peng_robinson_roots,
    PengRobinsonResidualHelmholtzTerm,
)


class GasStation(StrictModule):
    total_temperature: Array
    total_pressure: Array
    mass_flow: Array
    mole_fraction: Array
    mass_specific_enthalpy: Array
    mass_specific_entropy: Array
    successful: Array
    thermodynamics_id: str = eqx.field(static=True)


class CompressorMapEvaluation(StrictModule):
    corrected_flow: Array
    pressure_ratio: Array
    isentropic_efficiency: Array
    supported: Array
    map_id: str = eqx.field(static=True)


class CompressorMapPlan(StrictModule):
    corrected_speed_axis: Array
    operating_line_axis: Array
    corrected_flow: Array
    pressure_ratio: Array
    isentropic_efficiency: Array
    reference_temperature: float = eqx.field(static=True)
    reference_pressure: float = eqx.field(static=True)
    provenance: str = eqx.field(static=True)
    map_id: str = eqx.field(static=True)

    def __init__(
        self,
        corrected_speed_axis: ArrayLike,
        operating_line_axis: ArrayLike,
        corrected_flow: ArrayLike,
        pressure_ratio: ArrayLike,
        isentropic_efficiency: ArrayLike,
        /,
        *,
        reference_temperature: float,
        reference_pressure: float,
        provenance: str,
    ) -> None:
        speed = np.asarray(corrected_speed_axis, dtype=float)
        line = np.asarray(operating_line_axis, dtype=float)
        flow = np.asarray(corrected_flow, dtype=float)
        ratio = np.asarray(pressure_ratio, dtype=float)
        efficiency = np.asarray(isentropic_efficiency, dtype=float)
        shape = (speed.size, line.size)
        reference_temperature_value = float(reference_temperature)
        reference_pressure_value = float(reference_pressure)
        source = str(provenance)
        if speed.ndim != 1 or line.ndim != 1 or speed.size < 2 or line.size < 2:
            raise ValueError(
                "Compressor map axes must be one-dimensional with two nodes."
            )
        if np.any(~np.isfinite(speed)) or np.any(np.diff(speed) <= 0.0):
            raise ValueError("corrected_speed_axis must be finite and increasing.")
        if np.any(~np.isfinite(line)) or np.any(np.diff(line) <= 0.0):
            raise ValueError("operating_line_axis must be finite and increasing.")
        if flow.shape != shape or ratio.shape != shape or efficiency.shape != shape:
            raise ValueError("Compressor map tables must match the axis tensor shape.")
        if (
            np.any(~np.isfinite(flow))
            or np.any(flow <= 0.0)
            or np.any(~np.isfinite(ratio))
            or np.any(ratio < 1.0)
            or np.any(~np.isfinite(efficiency))
            or np.any(efficiency <= 0.0)
            or np.any(efficiency > 1.0)
        ):
            raise ValueError("Compressor map values are outside physical bounds.")
        if (
            not np.isfinite(reference_temperature_value)
            or reference_temperature_value <= 0.0
            or not np.isfinite(reference_pressure_value)
            or reference_pressure_value <= 0.0
            or not source
        ):
            raise ValueError(
                "Compressor reference conditions and provenance are required."
            )
        self.corrected_speed_axis = jnp.asarray(speed)
        self.operating_line_axis = jnp.asarray(line)
        self.corrected_flow = jnp.asarray(flow)
        self.pressure_ratio = jnp.asarray(ratio)
        self.isentropic_efficiency = jnp.asarray(efficiency)
        self.reference_temperature = reference_temperature_value
        self.reference_pressure = reference_pressure_value
        self.provenance = source
        self.map_id = canonical_fingerprint(
            {
                "kind": "compressor-map",
                "speed": array_tree_fingerprint(speed),
                "line": array_tree_fingerprint(line),
                "flow": array_tree_fingerprint(flow),
                "pressure_ratio": array_tree_fingerprint(ratio),
                "efficiency": array_tree_fingerprint(efficiency),
                "reference_temperature": reference_temperature_value,
                "reference_pressure": reference_pressure_value,
                "provenance": source,
            }
        )

    def evaluate(
        self,
        corrected_speed: ArrayLike,
        operating_line: ArrayLike,
        /,
    ) -> CompressorMapEvaluation:
        speed = jnp.asarray(corrected_speed)
        line = jnp.asarray(operating_line)
        shape = jnp.broadcast_shapes(speed.shape, line.shape)
        query = jnp.stack(
            (jnp.broadcast_to(speed, shape), jnp.broadcast_to(line, shape)), axis=-1
        )
        stencil = rectilinear_stencil(
            (self.corrected_speed_axis, self.operating_line_axis),
            query,
            boundary=("constant", "constant"),
        )
        flow = apply_gather_stencil(self.corrected_flow.reshape((-1,)), stencil)
        ratio = apply_gather_stencil(self.pressure_ratio.reshape((-1,)), stencil)
        efficiency = apply_gather_stencil(
            self.isentropic_efficiency.reshape((-1,)), stencil
        )
        supported = flow.support & ratio.support & efficiency.support
        return CompressorMapEvaluation(
            flow.values,
            ratio.values,
            efficiency.values,
            supported,
            self.map_id,
        )


class CompressorDesignArtifact(StrictModule):
    corrected_flow_scale: Array
    pressure_ratio_increment_scale: Array
    efficiency_scale: Array
    parent_map_id: str = eqx.field(static=True)
    design_id: str = eqx.field(static=True)


class CompressorEvaluation(StrictModule):
    outlet: GasStation
    shaft_power: Array
    map_evaluation: CompressorMapEvaluation
    successful: Array
    compressor_id: str = eqx.field(static=True)


class CompressorPlan(StrictModule):
    thermodynamics: HomogeneousHelmholtzPlan
    performance_map: CompressorMapPlan
    compressor_id: str = eqx.field(static=True)

    def __init__(
        self,
        thermodynamics: HomogeneousHelmholtzPlan,
        performance_map: CompressorMapPlan,
        /,
    ) -> None:
        if not isinstance(thermodynamics, HomogeneousHelmholtzPlan):
            raise TypeError("thermodynamics must be HomogeneousHelmholtzPlan.")
        if not isinstance(performance_map, CompressorMapPlan):
            raise TypeError("performance_map must be CompressorMapPlan.")
        self.thermodynamics = thermodynamics
        self.performance_map = performance_map
        self.compressor_id = canonical_fingerprint(
            {
                "kind": "compressor-plan",
                "thermodynamics": thermodynamics.model_id,
                "map": performance_map.map_id,
            }
        )

    def station(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        mass_flow: ArrayLike,
        mole_fraction: ArrayLike,
        /,
    ) -> GasStation:
        state = _gas_state_tp(
            self.thermodynamics,
            jnp.asarray(temperature),
            jnp.asarray(pressure),
            jnp.asarray(mole_fraction),
        )
        return GasStation(
            state.temperature,
            state.pressure,
            jnp.asarray(mass_flow),
            state.mole_fraction,
            state.molar_enthalpy / state.molar_mass,
            state.molar_entropy / state.molar_mass,
            state.evidence.successful,
            self.thermodynamics.model_id,
        )

    def design(
        self,
        corrected_speed: ArrayLike,
        operating_line: ArrayLike,
        /,
        *,
        corrected_flow: float,
        pressure_ratio: float,
        isentropic_efficiency: float,
    ) -> CompressorDesignArtifact:
        raw = self.performance_map.evaluate(corrected_speed, operating_line)
        targets = tuple(
            float(value)
            for value in (
                corrected_flow,
                pressure_ratio,
                isentropic_efficiency,
            )
        )
        if (
            any(not np.isfinite(value) for value in targets)
            or targets[0] <= 0.0
            or targets[1] < 1.0
            or not 0.0 < targets[2] <= 1.0
            or not bool(np.asarray(raw.supported))
        ):
            raise ValueError("Compressor design target or map point is invalid.")
        flow_scale = jnp.asarray(targets[0]) / raw.corrected_flow
        ratio_scale = (jnp.asarray(targets[1]) - 1.0) / jnp.maximum(
            raw.pressure_ratio - 1.0, jnp.finfo(raw.pressure_ratio.dtype).eps
        )
        efficiency_scale = jnp.asarray(targets[2]) / raw.isentropic_efficiency
        design_id = canonical_fingerprint(
            {
                "kind": "compressor-design",
                "compressor": self.compressor_id,
                "map": self.performance_map.map_id,
                "corrected_speed": float(corrected_speed),
                "operating_line": float(operating_line),
                "targets": list(targets),
            }
        )
        return CompressorDesignArtifact(
            flow_scale,
            ratio_scale,
            efficiency_scale,
            self.performance_map.map_id,
            design_id,
        )

    def evaluate(
        self,
        inlet: GasStation,
        corrected_speed: ArrayLike,
        operating_line: ArrayLike,
        design: CompressorDesignArtifact,
        /,
    ) -> CompressorEvaluation:
        if inlet.thermodynamics_id != self.thermodynamics.model_id:
            raise ValueError("Inlet station thermodynamics do not match compressor.")
        if design.parent_map_id != self.performance_map.map_id:
            raise ValueError("Compressor design artifact belongs to another map.")
        raw = self.performance_map.evaluate(corrected_speed, operating_line)
        corrected_flow = raw.corrected_flow * design.corrected_flow_scale
        pressure_ratio = (
            1.0 + (raw.pressure_ratio - 1.0) * design.pressure_ratio_increment_scale
        )
        efficiency = raw.isentropic_efficiency * design.efficiency_scale
        mass_flow = (
            corrected_flow
            * (inlet.total_pressure / self.performance_map.reference_pressure)
            / jnp.sqrt(
                inlet.total_temperature / self.performance_map.reference_temperature
            )
        )
        outlet_pressure = inlet.total_pressure * pressure_ratio
        isentropic = _solve_temperature_for_property(
            self.thermodynamics,
            outlet_pressure,
            inlet.mole_fraction,
            inlet.mass_specific_entropy,
            property_name="entropy",
        )
        isentropic_enthalpy = isentropic.molar_enthalpy / isentropic.molar_mass
        outlet_enthalpy = (
            inlet.mass_specific_enthalpy
            + (isentropic_enthalpy - inlet.mass_specific_enthalpy) / efficiency
        )
        outlet_state = _solve_temperature_for_property(
            self.thermodynamics,
            outlet_pressure,
            inlet.mole_fraction,
            outlet_enthalpy,
            property_name="enthalpy",
        )
        outlet = GasStation(
            outlet_state.temperature,
            outlet_pressure,
            mass_flow,
            inlet.mole_fraction,
            outlet_state.molar_enthalpy / outlet_state.molar_mass,
            outlet_state.molar_entropy / outlet_state.molar_mass,
            outlet_state.evidence.successful,
            self.thermodynamics.model_id,
        )
        power = mass_flow * (outlet.mass_specific_enthalpy - inlet.mass_specific_enthalpy)
        successful = (
            inlet.successful
            & raw.supported
            & (efficiency > 0.0)
            & (efficiency <= 1.0)
            & isentropic.evidence.successful
            & outlet.successful
            & jnp.isfinite(power)
            & (power >= 0.0)
        )
        return CompressorEvaluation(
            outlet,
            power,
            CompressorMapEvaluation(
                corrected_flow,
                pressure_ratio,
                efficiency,
                raw.supported,
                raw.map_id,
            ),
            successful,
            self.compressor_id,
        )


def _gas_state_tp(thermodynamics, temperature, pressure, composition):
    if isinstance(thermodynamics.residual, ZeroResidualHelmholtzTerm):
        density = pressure / (UNIVERSAL_GAS_CONSTANT * temperature)
    elif isinstance(thermodynamics.residual, PengRobinsonResidualHelmholtzTerm):
        roots = peng_robinson_roots(thermodynamics, temperature, pressure, composition)
        index = roots.stable.shape[0] - 1 - jnp.argmax(roots.stable[::-1])
        density = roots.molar_density[index]
    else:
        raise TypeError("Unsupported gas thermodynamic residual model.")
    return thermodynamics.evaluate(temperature, density, composition)


def _solve_temperature_for_property(
    thermodynamics,
    pressure,
    composition,
    target,
    *,
    property_name: str,
):
    lower = jnp.asarray(
        thermodynamics.thermodynamics.minimum_temperature,
        dtype=jnp.result_type(pressure, composition, target),
    )
    upper = jnp.asarray(
        thermodynamics.thermodynamics.maximum_temperature,
        dtype=lower.dtype,
    )

    def property_value(temperature):
        state = _gas_state_tp(thermodynamics, temperature, pressure, composition)
        if property_name == "entropy":
            return state.molar_entropy / state.molar_mass
        if property_name == "enthalpy":
            return state.molar_enthalpy / state.molar_mass
        raise ValueError("Unknown compressor property solve.")

    def body(_, bounds):
        low, high = bounds
        midpoint = 0.5 * (low + high)
        value = property_value(midpoint)
        return (
            jnp.where(value < target, midpoint, low),
            jnp.where(value < target, high, midpoint),
        )

    low, high = jax.lax.fori_loop(0, 80, body, (lower, upper))
    return _gas_state_tp(thermodynamics, 0.5 * (low + high), pressure, composition)


__all__ = [
    "CompressorDesignArtifact",
    "CompressorEvaluation",
    "CompressorMapEvaluation",
    "CompressorMapPlan",
    "CompressorPlan",
    "GasStation",
]
