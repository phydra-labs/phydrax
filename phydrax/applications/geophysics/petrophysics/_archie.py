#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._strict import StrictModule
from ....discretization._transfer import FieldTransfer
from ....interchange._geospatial import GeospatialContract
from ..electrical._finite_patch import PreparedDC


class ArchieSaturationConductivity(StrictModule):
    """Calibration variables; conductivity is S/m, temperature is K.

    All entries remain differentiable leaves suitable for native priors and
    bijectors. The temperature coefficient is an exponential log-conductivity
    slope in inverse kelvin, not an unqualified salinity model.
    """

    water_conductivity_S_m: Array
    cementation_exponent: Array
    saturation_exponent: Array
    tortuosity_factor: Array
    temperature_coefficient_K_inverse: Array
    reference_temperature_K: Array

    def __init__(
        self,
        water_conductivity_S_m: ArrayLike,
        *,
        cementation_exponent: ArrayLike = 2.0,
        saturation_exponent: ArrayLike = 2.0,
        tortuosity_factor: ArrayLike = 1.0,
        temperature_coefficient_K_inverse: ArrayLike = 0.0,
        reference_temperature_K: ArrayLike = 293.15,
    ):
        self.water_conductivity_S_m = jnp.asarray(water_conductivity_S_m)
        self.cementation_exponent = jnp.asarray(cementation_exponent)
        self.saturation_exponent = jnp.asarray(saturation_exponent)
        self.tortuosity_factor = jnp.asarray(tortuosity_factor)
        self.temperature_coefficient_K_inverse = jnp.asarray(
            temperature_coefficient_K_inverse
        )
        self.reference_temperature_K = jnp.asarray(reference_temperature_K)

    def conductivity(
        self,
        porosity: ArrayLike,
        saturation: ArrayLike,
        temperature_K: ArrayLike,
        *,
        log_discrepancy: ArrayLike = 0.0,
    ) -> Array:
        phi, saturation, temperature, discrepancy = jnp.broadcast_arrays(
            jnp.asarray(porosity),
            jnp.asarray(saturation),
            jnp.asarray(temperature_K),
            jnp.asarray(log_discrepancy),
        )
        invalid = (
            jnp.any(~jnp.isfinite(phi))
            | jnp.any((phi <= 0) | (phi > 1))
            | jnp.any(~jnp.isfinite(saturation))
            | jnp.any((saturation <= 0) | (saturation > 1))
            | jnp.any(~jnp.isfinite(temperature))
            | jnp.any(temperature <= 0)
            | jnp.any(~jnp.isfinite(discrepancy))
        )
        for value in (
            self.water_conductivity_S_m,
            self.cementation_exponent,
            self.saturation_exponent,
            self.tortuosity_factor,
            self.reference_temperature_K,
        ):
            invalid = invalid | jnp.any(~jnp.isfinite(value)) | jnp.any(value <= 0)
        invalid = invalid | jnp.any(~jnp.isfinite(self.temperature_coefficient_K_inverse))
        phi = eqx.error_if(
            phi,
            invalid,
            "Archie model requires positive calibrated parameters, 0<porosity,saturation<=1 and positive K.",
        )
        log_sigma = (
            jnp.log(self.water_conductivity_S_m)
            - jnp.log(self.tortuosity_factor)
            + self.cementation_exponent * jnp.log(phi)
            + self.saturation_exponent * jnp.log(saturation)
            + self.temperature_coefficient_K_inverse
            * (temperature - self.reference_temperature_K)
            + discrepancy
        )
        conductivity = jnp.exp(log_sigma)
        return eqx.error_if(
            conductivity,
            jnp.any(~jnp.isfinite(conductivity)) | jnp.any(conductivity <= 0),
            "Archie conductivity must remain finite and strictly positive for DC diffusion.",
        )


class HydrogeophysicalPrediction(StrictModule):
    response: Array
    conductivity_S_m: Array
    target_porosity: Array
    target_saturation: Array
    target_temperature_K: Array
    water_volume_residual_m3: Array


class HydrogeophysicalPlan(StrictModule):
    """Transfer conserved water fractions, then evaluate uncertain petrophysics.

    A supplied native cell-average FieldTransfer owns fixed overlap geometry.
    Its claims are checked algebraically without materializing a dense matrix.
    Conductivity itself is neither treated nor advertised as conserved.
    """

    electrical: PreparedDC
    transfer: FieldTransfer
    source_volumes_m3: Array
    target_volumes_m3: Array
    source_geometry_id: str = eqx.field(static=True)
    target_geometry_id: str = eqx.field(static=True)

    def __init__(
        self,
        electrical: PreparedDC,
        transfer: FieldTransfer,
        source_volumes_m3: ArrayLike,
        target_volumes_m3: ArrayLike,
        *,
        source_geometry_id: str,
        target_geometry_id: str,
        source_coordinates: GeospatialContract,
        target_coordinates: GeospatialContract,
        tolerance: float = 1e-9,
    ):
        if not isinstance(electrical, PreparedDC) or not isinstance(
            transfer, FieldTransfer
        ):
            raise TypeError(
                "Hydrogeophysics requires prepared DC and a native FieldTransfer."
            )
        source_coordinates.require_compatible(target_coordinates, dimensions=3)
        if not source_geometry_id or not target_geometry_id:
            raise ValueError("Both fixed mesh geometry identities are required.")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Transfer tolerance must be positive and finite.")
        properties = transfer.properties
        if not (
            properties.constant_preserving
            and properties.conservative
            and properties.positivity_preserving
            and properties.adjoint_paired
        ):
            raise ValueError(
                "Water transfer requires constant, conservative, positive and adjoint-paired contracts."
            )
        source = np.asarray(source_volumes_m3, dtype=float).reshape(-1)
        target = np.asarray(target_volumes_m3, dtype=float).reshape(-1)
        source_space = transfer.primal_operator.source
        target_space = transfer.primal_operator.target
        if source.size != source_space.size or target.size != target_space.size:
            raise ValueError("Cell-volume counts must match scalar transfer spaces.")
        if (
            np.any(~np.isfinite(source))
            or np.any(source <= 0)
            or np.any(~np.isfinite(target))
            or np.any(target <= 0)
        ):
            raise ValueError("Transfer cell volumes must be positive and finite.")
        constant = target_space.flatten(
            transfer.primal_operator.mv(source_space.unflatten(jnp.ones(source.size)))
        )
        dual_pullback = transfer.dual_pullback_operator
        if dual_pullback is None:
            raise RuntimeError("Conservative transfer lost its required dual pullback.")
        transpose_volume = source_space.flatten(
            dual_pullback.mv(target_space.unflatten(jnp.asarray(target)))
        )
        if not np.allclose(np.asarray(constant), 1, atol=tolerance, rtol=tolerance):
            raise ValueError("Declared transfer does not preserve constants.")
        if not np.allclose(
            np.asarray(transpose_volume),
            source,
            atol=tolerance * np.max(source),
            rtol=tolerance,
        ):
            raise ValueError(
                "Declared transfer does not conserve volume-weighted cell averages."
            )
        self.electrical = electrical
        self.transfer = transfer
        self.source_volumes_m3 = jnp.asarray(source)
        self.target_volumes_m3 = jnp.asarray(target)
        self.source_geometry_id = str(source_geometry_id)
        self.target_geometry_id = str(target_geometry_id)

    def require_geometry(self, source_geometry_id: str, target_geometry_id: str):
        if (
            source_geometry_id != self.source_geometry_id
            or target_geometry_id != self.target_geometry_id
        ):
            raise ValueError(
                "Hydrogeophysical transfer cannot be reused after an unqualified geometry change."
            )

    def _transfer(self, value):
        value = jnp.asarray(value).reshape(-1)
        source = self.transfer.primal_operator.source
        target = self.transfer.primal_operator.target
        if value.size != source.size:
            raise ValueError("Physical field does not match source cell count.")
        return target.flatten(self.transfer.primal_operator.mv(source.unflatten(value)))

    def predict(
        self,
        porosity: ArrayLike,
        water_volume_m3: ArrayLike,
        temperature_K: ArrayLike,
        calibration: ArchieSaturationConductivity,
        *,
        log_discrepancy: ArrayLike = 0.0,
    ) -> HydrogeophysicalPrediction:
        if not isinstance(calibration, ArchieSaturationConductivity):
            raise TypeError(
                "Petrophysical calibration must be explicit ArchieSaturationConductivity."
            )
        porosity = jnp.broadcast_to(jnp.asarray(porosity), self.source_volumes_m3.shape)
        water = jnp.asarray(water_volume_m3)
        temperature = jnp.broadcast_to(jnp.asarray(temperature_K), porosity.shape)
        if water.shape != porosity.shape:
            raise ValueError("Water volume must match the source mesh cells.")
        water = eqx.error_if(
            water,
            jnp.any(~jnp.isfinite(water))
            | jnp.any(water <= 0)
            | jnp.any(water > porosity * self.source_volumes_m3)
            | jnp.any(~jnp.isfinite(temperature))
            | jnp.any(temperature <= 0)
            | jnp.any(~jnp.isfinite(porosity))
            | jnp.any((porosity <= 0) | (porosity > 1)),
            "Source water, pore volume and temperature are outside Archie applicability.",
        )
        content = water / self.source_volumes_m3
        target_phi = self._transfer(porosity)
        target_content = self._transfer(content)
        target_temperature = self._transfer(content * temperature) / target_content
        saturation = target_content / target_phi
        conductivity = calibration.conductivity(
            target_phi, saturation, target_temperature, log_discrepancy=log_discrepancy
        )
        response = self.electrical.predict(conductivity)
        balance = jnp.sum(target_content * self.target_volumes_m3) - jnp.sum(water)
        return HydrogeophysicalPrediction(
            response, conductivity, target_phi, saturation, target_temperature, balance
        )


__all__ = [
    "ArchieSaturationConductivity",
    "HydrogeophysicalPlan",
    "HydrogeophysicalPrediction",
]
