#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class TransportProperties(StrictModule):
    dynamic_viscosity: Array
    bulk_viscosity: Array
    thermal_conductivity: Array


class AbstractTransportClosure(StrictModule, NonTrainableState):
    """Temperature-dependent viscous and thermal transport closure."""

    closure_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def properties(
        self,
        temperature: Array,
        state: Array,
        args: Any = None,
        /,
    ) -> TransportProperties:
        raise NotImplementedError


class ConstantTransport(AbstractTransportClosure):
    dynamic_viscosity: float = eqx.field(static=True)
    bulk_viscosity: float = eqx.field(static=True)
    thermal_conductivity: float = eqx.field(static=True)

    def __init__(
        self,
        dynamic_viscosity: float,
        thermal_conductivity: float,
        /,
        *,
        bulk_viscosity: float = 0.0,
    ):
        viscosity = float(dynamic_viscosity)
        conductivity = float(thermal_conductivity)
        bulk = float(bulk_viscosity)
        if (
            not np.isfinite(viscosity)
            or not np.isfinite(conductivity)
            or not np.isfinite(bulk)
            or viscosity < 0.0
            or conductivity < 0.0
        ):
            raise ValueError(
                "Constant transport coefficients must be finite and non-negative."
            )
        self.dynamic_viscosity = viscosity
        self.bulk_viscosity = bulk
        self.thermal_conductivity = conductivity
        self.closure_id = canonical_fingerprint(
            {
                "kind": "constant-transport",
                "dynamic_viscosity": viscosity,
                "bulk_viscosity": bulk,
                "thermal_conductivity": conductivity,
            }
        )

    def properties(
        self,
        temperature: Array,
        state: Array,
        args: Any = None,
        /,
    ) -> TransportProperties:
        del state, args
        shape = jnp.asarray(temperature).shape
        dtype = jnp.asarray(temperature).dtype
        return TransportProperties(
            dynamic_viscosity=jnp.full(shape, self.dynamic_viscosity, dtype=dtype),
            bulk_viscosity=jnp.full(shape, self.bulk_viscosity, dtype=dtype),
            thermal_conductivity=jnp.full(shape, self.thermal_conductivity, dtype=dtype),
        )


class SutherlandTransport(AbstractTransportClosure):
    """Sutherland viscosity with constant Prandtl heat conduction."""

    reference_viscosity: float = eqx.field(static=True)
    reference_temperature: float = eqx.field(static=True)
    sutherland_temperature: float = eqx.field(static=True)
    specific_heat_cp: float = eqx.field(static=True)
    prandtl_number: float = eqx.field(static=True)
    bulk_viscosity: float = eqx.field(static=True)

    def __init__(
        self,
        reference_viscosity: float,
        reference_temperature: float,
        sutherland_temperature: float,
        specific_heat_cp: float,
        prandtl_number: float,
        /,
        *,
        bulk_viscosity: float = 0.0,
    ):
        values = tuple(
            float(value)
            for value in (
                reference_viscosity,
                reference_temperature,
                sutherland_temperature,
                specific_heat_cp,
                prandtl_number,
                bulk_viscosity,
            )
        )
        mu_ref, temperature_ref, sutherland, cp, prandtl, bulk = values
        if (
            any(not np.isfinite(value) for value in values)
            or mu_ref < 0.0
            or temperature_ref <= 0.0
            or sutherland < 0.0
            or cp <= 0.0
            or prandtl <= 0.0
        ):
            raise ValueError("Sutherland transport parameters are invalid.")
        self.reference_viscosity = mu_ref
        self.reference_temperature = temperature_ref
        self.sutherland_temperature = sutherland
        self.specific_heat_cp = cp
        self.prandtl_number = prandtl
        self.bulk_viscosity = bulk
        self.closure_id = canonical_fingerprint(
            {
                "kind": "sutherland-transport",
                "reference_viscosity": mu_ref,
                "reference_temperature": temperature_ref,
                "sutherland_temperature": sutherland,
                "specific_heat_cp": cp,
                "prandtl_number": prandtl,
                "bulk_viscosity": bulk,
            }
        )

    def properties(
        self,
        temperature: Array,
        state: Array,
        args: Any = None,
        /,
    ) -> TransportProperties:
        del state, args
        value = jnp.asarray(temperature)
        safe = eqx.error_if(
            value,
            jnp.any(~jnp.isfinite(value) | (value <= 0.0)),
            "Sutherland transport requires finite positive temperature.",
        )
        ratio = safe / self.reference_temperature
        viscosity = (
            self.reference_viscosity
            * ratio**1.5
            * (self.reference_temperature + self.sutherland_temperature)
            / (safe + self.sutherland_temperature)
        )
        conductivity = viscosity * self.specific_heat_cp / self.prandtl_number
        return TransportProperties(
            dynamic_viscosity=viscosity,
            bulk_viscosity=jnp.full_like(viscosity, self.bulk_viscosity),
            thermal_conductivity=conductivity,
        )


class PrandtlTransport(AbstractTransportClosure):
    """Wrap a viscosity closure and derive conductivity from constant cp/Pr."""

    viscosity: AbstractTransportClosure
    specific_heat_cp: float = eqx.field(static=True)
    prandtl_number: float = eqx.field(static=True)

    def __init__(
        self,
        viscosity: AbstractTransportClosure,
        specific_heat_cp: float,
        prandtl_number: float,
        /,
    ):
        cp = float(specific_heat_cp)
        prandtl = float(prandtl_number)
        if not isinstance(viscosity, AbstractTransportClosure):
            raise TypeError("viscosity must be an AbstractTransportClosure.")
        if not np.isfinite(cp) or not np.isfinite(prandtl) or cp <= 0.0 or prandtl <= 0.0:
            raise ValueError("Prandtl transport requires positive finite cp and Pr.")
        self.viscosity = viscosity
        self.specific_heat_cp = cp
        self.prandtl_number = prandtl
        self.closure_id = canonical_fingerprint(
            {
                "kind": "prandtl-transport",
                "viscosity": viscosity.closure_id,
                "specific_heat_cp": cp,
                "prandtl_number": prandtl,
            }
        )

    def properties(
        self,
        temperature: Array,
        state: Array,
        args: Any = None,
        /,
    ) -> TransportProperties:
        base = self.viscosity.properties(temperature, state, args)
        return TransportProperties(
            dynamic_viscosity=base.dynamic_viscosity,
            bulk_viscosity=base.bulk_viscosity,
            thermal_conductivity=(
                base.dynamic_viscosity * self.specific_heat_cp / self.prandtl_number
            ),
        )


__all__ = [
    "AbstractTransportClosure",
    "ConstantTransport",
    "PrandtlTransport",
    "SutherlandTransport",
    "TransportProperties",
]
