#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntFlag
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._admissibility import (
    AdmissibilityHeader,
    AdmissibilityReason,
)
from ..._differentiation import (
    branch_policy_contract,
    BranchDifferentiationPolicy,
    DerivativeContract,
    DerivativeSurface,
)
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._conservation_boundary import AbstractConservationBoundary, ALEBoundaryContext


class RarefiedWallReason(IntFlag):
    KNUDSEN_LIMIT_EXCEEDED = 1 << 8
    WALL_RESOLUTION_EXCEEDED = 1 << 9
    UNSUPPORTED_GAS_MODEL = 1 << 10
    INVALID_WALL_NORMAL = 1 << 11


class MaxwellSmoluchowskiWallCoefficients(StrictModule, NonTrainableState):
    slip_prefactor: float = eqx.field(static=True)
    temperature_jump_prefactor: float = eqx.field(static=True)
    thermal_creep_prefactor: float = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)
    coefficient_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        slip_prefactor: float,
        temperature_jump_prefactor: float,
        thermal_creep_prefactor: float,
        convention_id: str,
    ) -> None:
        values = tuple(
            float(value)
            for value in (
                slip_prefactor,
                temperature_jump_prefactor,
                thermal_creep_prefactor,
            )
        )
        convention = str(convention_id)
        if (
            any(not np.isfinite(value) or value < 0.0 for value in values)
            or not convention
        ):
            raise ValueError("Rarefied wall coefficients or convention are invalid.")
        self.slip_prefactor = values[0]
        self.temperature_jump_prefactor = values[1]
        self.thermal_creep_prefactor = values[2]
        self.convention_id = convention
        self.coefficient_id = canonical_fingerprint(
            {
                "kind": "maxwell-smoluchowski-wall-coefficients",
                "values": values,
                "convention": convention,
            }
        )


class ContinuumGasWallMaterial(StrictModule):
    wall_velocity: Array
    wall_temperature: Array | None
    outward_heat_flux: Array | None
    tangential_momentum_accommodation: float = eqx.field(static=True)
    thermal_accommodation: float = eqx.field(static=True)
    material_id: str = eqx.field(static=True)

    def __init__(
        self,
        wall_velocity: ArrayLike,
        /,
        *,
        tangential_momentum_accommodation: float,
        thermal_accommodation: float,
        wall_temperature: ArrayLike | None = None,
        outward_heat_flux: ArrayLike | None = None,
    ) -> None:
        velocity = jnp.asarray(wall_velocity)
        momentum = float(tangential_momentum_accommodation)
        thermal = float(thermal_accommodation)
        if (
            velocity.ndim != 1
            or velocity.size == 0
            or not 0.0 < momentum <= 1.0
            or not 0.0 < thermal <= 1.0
            or (wall_temperature is None) == (outward_heat_flux is None)
        ):
            raise ValueError("Rarefied gas wall material is invalid.")
        temperature = (
            None
            if wall_temperature is None
            else jnp.asarray(wall_temperature).reshape(())
        )
        heat = (
            None
            if outward_heat_flux is None
            else jnp.asarray(outward_heat_flux).reshape(())
        )
        if temperature is not None and (
            not bool(jnp.isfinite(temperature)) or not bool(temperature > 0.0)
        ):
            raise ValueError("Rarefied wall temperature must be finite and positive.")
        if heat is not None and not bool(jnp.isfinite(heat)):
            raise ValueError("Rarefied wall heat flux must be finite.")
        self.wall_velocity = velocity
        self.wall_temperature = temperature
        self.outward_heat_flux = heat
        self.tangential_momentum_accommodation = momentum
        self.thermal_accommodation = thermal
        self.material_id = canonical_fingerprint(
            {
                "kind": "continuum-gas-wall-material",
                "wall_velocity": array_tree_fingerprint(velocity),
                "wall_temperature": (
                    None if temperature is None else array_tree_fingerprint(temperature)
                ),
                "outward_heat_flux": None
                if heat is None
                else array_tree_fingerprint(heat),
                "tmac": momentum,
                "thermal_accommodation": thermal,
            }
        )


class WallRegimePolicy(StrictModule, NonTrainableState):
    characteristic_length: float = eqx.field(static=True)
    maximum_knudsen_number: float = eqx.field(static=True)
    maximum_lambda_to_wall_distance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        characteristic_length: float,
        maximum_knudsen_number: float,
        maximum_lambda_to_wall_distance: float,
    ) -> None:
        length = float(characteristic_length)
        knudsen = float(maximum_knudsen_number)
        resolution = float(maximum_lambda_to_wall_distance)
        if any(
            not np.isfinite(value) or value <= 0.0
            for value in (length, knudsen, resolution)
        ):
            raise ValueError("Rarefied wall regime policy is invalid.")
        self.characteristic_length = length
        self.maximum_knudsen_number = knudsen
        self.maximum_lambda_to_wall_distance = resolution
        self.policy_id = canonical_fingerprint(
            {
                "kind": "rarefied-wall-regime",
                "characteristic_length": length,
                "maximum_knudsen_number": knudsen,
                "maximum_lambda_to_wall_distance": resolution,
            }
        )


# Derivatives of the executed algorithm with model and regime decisions frozen.
_DERIVATIVE_CONTRACT = branch_policy_contract(
    BranchDifferentiationPolicy.FROZEN_DECISION,
    surfaces=(DerivativeSurface.PRIMAL_STATE, DerivativeSurface.PHYSICAL_PARAMETER),
)


class RarefiedWallEvaluation(StrictModule):
    normal_diffusive_flux: Array
    gas_velocity_trace: Array
    gas_temperature_trace: Array
    slip_length: Array
    temperature_jump_length: Array
    mean_free_path: Array
    knudsen_number: Array
    lambda_to_wall_distance: Array
    wall_mechanical_power: Array
    outward_thermal_flux: Array
    header: AdmissibilityHeader
    derivative_contract: DerivativeContract
    plan_id: str = eqx.field(static=True)


class MaxwellSmoluchowskiContinuumWallPlan(AbstractConservationBoundary):
    """First-order slip, temperature jump, and thermal-creep wall closure."""

    coefficients: MaxwellSmoluchowskiWallCoefficients
    material: ContinuumGasWallMaterial
    regime: WallRegimePolicy
    boundary_id: str = eqx.field(static=True)

    def __init__(
        self,
        coefficients: MaxwellSmoluchowskiWallCoefficients,
        material: ContinuumGasWallMaterial,
        regime: WallRegimePolicy,
        /,
    ) -> None:
        if (
            not isinstance(coefficients, MaxwellSmoluchowskiWallCoefficients)
            or not isinstance(material, ContinuumGasWallMaterial)
            or not isinstance(regime, WallRegimePolicy)
        ):
            raise TypeError("Rarefied wall requires coefficients, material, and regime.")
        self.coefficients = coefficients
        self.material = material
        self.regime = regime
        self.boundary_id = canonical_fingerprint(
            {
                "kind": "maxwell-smoluchowski-continuum-wall",
                "coefficients": coefficients.coefficient_id,
                "material": material.material_id,
                "regime": regime.policy_id,
            }
        )

    def exterior_state(
        self,
        system: Any,
        time: Array,
        interior: Array,
        coordinates: Array,
        outward_normal: Array,
        axis: int,
        args: Any,
        /,
    ) -> Array:
        del time, coordinates, axis, args
        primitive = system.conserved_to_primitive(interior)
        velocity = system.primitive_velocity(primitive)
        if self.material.wall_velocity.shape != (system.dimension,):
            raise ValueError("Rarefied wall velocity must match the gas dimension.")
        wall_velocity = self.material.wall_velocity.astype(velocity.dtype)
        relative = velocity - wall_velocity
        reflected = (
            velocity
            - 2.0
            * contract("...i,...i->...", relative, outward_normal, backend="jax")[
                ..., None
            ]
            * outward_normal
        )
        return system.primitive_to_conserved(
            system.with_primitive_velocity(primitive, reflected)
        )

    def ale_exterior_state(
        self,
        system: Any,
        interior: Array,
        context: ALEBoundaryContext,
        axis: int,
        /,
    ) -> Array:
        del system, interior, context, axis
        raise ValueError("Rarefied continuum walls do not support ALE geometry.")

    def evaluate_normal_flux(
        self,
        system: Any,
        interior: ArrayLike,
        conserved_gradient: ArrayLike,
        wall_distance: ArrayLike,
        outward_normal: ArrayLike,
        args: Any = None,
        /,
    ) -> RarefiedWallEvaluation:
        from ...equations._gas_dynamics import (
            HomogeneousMixtureCompressibleNavierStokesSystem,
        )

        if (
            not isinstance(system, HomogeneousMixtureCompressibleNavierStokesSystem)
            or system.favre_les is not None
            or system.species_diffusivities is not None
        ):
            raise TypeError(
                "Rarefied continuum walls require the single-temperature viscous "
                "ideal-mixture route without LES or species diffusion."
            )
        value = jnp.asarray(interior)
        gradient = jnp.asarray(conserved_gradient, dtype=value.dtype)
        distance = jnp.asarray(wall_distance, dtype=value.dtype)
        normal = jnp.asarray(outward_normal, dtype=value.dtype)
        leading = value.shape[:-1]
        if (
            value.shape[-1] != system.component_count
            or gradient.shape != value.shape + (system.dimension,)
            or distance.shape != leading
            or normal.shape not in ((system.dimension,), leading + (system.dimension,))
            or self.material.wall_velocity.shape != (system.dimension,)
        ):
            raise ValueError(
                "Rarefied wall state, gradient, distance, or normal is invalid."
            )
        normal = jnp.broadcast_to(normal, leading + (system.dimension,))
        flat_value = value.reshape((-1, system.component_count))
        flat_gradient = gradient.reshape((-1, system.component_count, system.dimension))

        def velocity_from_state(state):
            primitive_state = system.conserved_to_primitive(state)
            return system.primitive_velocity(primitive_state)

        velocity_jacobian = jax.vmap(jax.jacfwd(velocity_from_state))(flat_value)
        temperature_jacobian = jax.vmap(jax.jacfwd(system.temperature))(flat_value)
        velocity_gradient = contract(
            "pic,pcj->pij", velocity_jacobian, flat_gradient, backend="jax"
        ).reshape(leading + (system.dimension, system.dimension))
        temperature_gradient = contract(
            "pc,pcj->pj", temperature_jacobian, flat_gradient, backend="jax"
        ).reshape(leading + (system.dimension,))
        primitive = system.conserved_to_primitive(value)
        velocity = system.primitive_velocity(primitive)
        temperature = system.temperature(value)
        density = system.density(value)
        transport = system.transport_properties(value, args)
        mean_free_path = system.mean_free_path(value, args)
        inward_normal = -normal
        wall_velocity = jnp.broadcast_to(
            self.material.wall_velocity.astype(value.dtype), velocity.shape
        )
        wall_normal_velocity = contract(
            "...i,...i->...", wall_velocity, normal, backend="jax"
        )
        tangential_temperature_gradient = (
            temperature_gradient
            - contract(
                "...i,...i->...", temperature_gradient, inward_normal, backend="jax"
            )[..., None]
            * inward_normal
        )
        thermal_creep = (
            self.coefficients.thermal_creep_prefactor
            * transport.dynamic_viscosity[..., None]
            / jnp.maximum(density * temperature, jnp.finfo(value.dtype).tiny)[..., None]
            * tangential_temperature_gradient
        )
        slip_length = (
            self.coefficients.slip_prefactor
            * (2.0 - self.material.tangential_momentum_accommodation)
            / self.material.tangential_momentum_accommodation
            * mean_free_path
        )
        jump_length = (
            self.coefficients.temperature_jump_prefactor
            * (2.0 - self.material.thermal_accommodation)
            / self.material.thermal_accommodation
            * mean_free_path
        )
        slip_ratio = slip_length / distance
        gas_velocity_trace = (
            wall_velocity + thermal_creep + slip_ratio[..., None] * velocity
        ) / (1.0 + slip_ratio)[..., None]
        gas_velocity_trace = (
            gas_velocity_trace
            + (
                wall_normal_velocity
                - contract("...i,...i->...", gas_velocity_trace, normal, backend="jax")
            )[..., None]
            * normal
        )
        inward_velocity_derivative = (velocity - gas_velocity_trace) / distance[..., None]
        current_inward_velocity_derivative = contract(
            "...ij,...j->...i", velocity_gradient, inward_normal, backend="jax"
        )
        face_velocity_gradient = (
            velocity_gradient
            + (inward_velocity_derivative - current_inward_velocity_derivative)[
                ..., :, None
            ]
            * inward_normal[..., None, :]
        )

        if self.material.wall_temperature is not None:
            wall_temperature = self.material.wall_temperature.astype(value.dtype)
            jump_ratio = jump_length / distance
            gas_temperature_trace = (wall_temperature + jump_ratio * temperature) / (
                1.0 + jump_ratio
            )
            inward_temperature_derivative = (
                temperature - gas_temperature_trace
            ) / distance
        else:
            outward_heat = self.material.outward_heat_flux.astype(value.dtype)
            inward_temperature_derivative = -outward_heat / transport.thermal_conductivity
            gas_temperature_trace = temperature - distance * inward_temperature_derivative
        current_inward_temperature_derivative = contract(
            "...i,...i->...", temperature_gradient, inward_normal, backend="jax"
        )
        face_temperature_gradient = (
            temperature_gradient
            + (inward_temperature_derivative - current_inward_temperature_derivative)[
                ..., None
            ]
            * inward_normal
        )
        viscous = system.viscous_flux_from_primitive_gradients(
            gas_velocity_trace,
            face_velocity_gradient,
            face_temperature_gradient,
            transport.dynamic_viscosity,
            transport.bulk_viscosity,
            transport.thermal_conductivity,
        )
        normal_flux = contract("...ci,...i->...c", viscous, normal, backend="jax")
        traction = normal_flux[..., system.momentum_slice]
        mechanical_power = contract(
            "...i,...i->...", wall_velocity, traction, backend="jax"
        )
        outward_thermal = transport.thermal_conductivity * contract(
            "...i,...i->...", face_temperature_gradient, normal, backend="jax"
        )
        knudsen = mean_free_path / self.regime.characteristic_length
        resolution = mean_free_path / distance
        finite = (
            jnp.all(jnp.isfinite(normal_flux), axis=-1)
            & jnp.isfinite(mean_free_path)
            & jnp.isfinite(distance)
            & jnp.isfinite(gas_temperature_trace)
            & jnp.isfinite(wall_normal_velocity)
        )
        supported = (
            finite
            & (distance > 0.0)
            & (gas_temperature_trace > 0.0)
            & (jnp.abs(wall_normal_velocity) <= 1.0e-12)
            & (knudsen <= self.regime.maximum_knudsen_number)
            & (resolution <= self.regime.maximum_lambda_to_wall_distance)
        )
        reasons = jnp.zeros(leading, dtype=jnp.uint32)
        reasons = jnp.where(
            finite,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.NONFINITE), jnp.uint32),
        )
        reasons = jnp.where(
            knudsen <= self.regime.maximum_knudsen_number,
            reasons,
            reasons
            | jnp.asarray(int(RarefiedWallReason.KNUDSEN_LIMIT_EXCEEDED), jnp.uint32),
        )
        reasons = jnp.where(
            resolution <= self.regime.maximum_lambda_to_wall_distance,
            reasons,
            reasons
            | jnp.asarray(int(RarefiedWallReason.WALL_RESOLUTION_EXCEEDED), jnp.uint32),
        )
        reasons = jnp.where(
            (distance > 0.0)
            & (gas_temperature_trace > 0.0)
            & (jnp.abs(wall_normal_velocity) <= 1.0e-12),
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.OUTSIDE_SUPPORT), jnp.uint32),
        )
        margin = jnp.minimum(
            self.regime.maximum_knudsen_number - knudsen,
            self.regime.maximum_lambda_to_wall_distance - resolution,
        )
        header = AdmissibilityHeader(
            jnp.where(supported, margin, jnp.minimum(margin, -1.0)),
            reasons,
            self.boundary_id,
            canonical_fingerprint(
                {"kind": "rarefied-wall-evidence", "wall": self.boundary_id}
            ),
        )
        return RarefiedWallEvaluation(
            normal_flux,
            gas_velocity_trace,
            gas_temperature_trace,
            slip_length,
            jump_length,
            mean_free_path,
            knudsen,
            resolution,
            mechanical_power,
            outward_thermal,
            header,
            _DERIVATIVE_CONTRACT,
            self.boundary_id,
        )


__all__ = [
    "ContinuumGasWallMaterial",
    "MaxwellSmoluchowskiContinuumWallPlan",
    "MaxwellSmoluchowskiWallCoefficients",
    "RarefiedWallEvaluation",
    "RarefiedWallReason",
    "WallRegimePolicy",
]
