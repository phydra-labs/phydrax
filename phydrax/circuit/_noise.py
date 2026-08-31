#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..control._descriptor_frequency import descriptor_frequency_response
from ..dynamics._linear_descriptor import LinearDescriptorSystem


_BOLTZMANN = 1.380649e-23


class NoiseSpectralFactor(StrictModule):
    factor: Array
    source_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        factor: ArrayLike,
        /,
        *,
        source_ids: tuple[str, ...] | None = None,
    ):
        value = jnp.asarray(factor, dtype=jnp.complex128)
        if value.ndim < 2 or bool(jnp.any(~jnp.isfinite(value))):
            raise ValueError("Noise spectral factor must be a finite matrix or batch.")
        count = int(value.shape[-1])
        ids = (
            tuple(f"noise-{index}" for index in range(count))
            if source_ids is None
            else tuple(str(item) for item in source_ids)
        )
        if len(ids) != count or len(set(ids)) != count or any(not item for item in ids):
            raise ValueError(
                "source_ids must uniquely identify every noise factor column."
            )
        self.factor, self.source_ids = value, ids

    @property
    def covariance(self) -> Array:
        return self.factor @ jnp.swapaxes(jnp.conj(self.factor), -1, -2)


class CircuitNoiseDiagnostics(StrictModule):
    hermitian_defect: Array
    minimum_eigenvalue: Array
    positive_semidefinite: Array
    finite: Array
    linear_success: Array


class CircuitNoiseResult(StrictModule):
    output_factor: Array
    output_covariance: Array
    transfer: Array
    diagnostics: CircuitNoiseDiagnostics
    source_ids: tuple[str, ...] = eqx.field(static=True)
    system_id: str = eqx.field(static=True)


def thermal_resistor_noise_factor(
    resistance: ArrayLike,
    temperature: ArrayLike,
    /,
    *,
    source_id: str = "resistor-thermal-noise",
) -> NoiseSpectralFactor:
    value = jnp.asarray(resistance, dtype=float)
    kelvin = jnp.asarray(temperature, dtype=float)
    if (
        value.shape != ()
        or kelvin.shape != ()
        or bool(~jnp.isfinite(value))
        or bool(~jnp.isfinite(kelvin))
        or bool(value <= 0.0)
        or bool(kelvin < 0.0)
    ):
        raise ValueError("Resistance and temperature must be finite physical scalars.")
    amplitude = jnp.sqrt(4.0 * _BOLTZMANN * kelvin / value)
    return NoiseSpectralFactor(
        jnp.asarray([[amplitude], [-amplitude]]), source_ids=(source_id,)
    )


def propagate_descriptor_noise(
    system: LinearDescriptorSystem,
    angular_frequency: ArrayLike,
    source_factor: NoiseSpectralFactor,
    /,
) -> CircuitNoiseResult:
    if not isinstance(system, LinearDescriptorSystem):
        raise TypeError("system must be LinearDescriptorSystem.")
    if not isinstance(source_factor, NoiseSpectralFactor):
        raise TypeError("source_factor must be NoiseSpectralFactor.")
    if source_factor.factor.shape[-2] != system.input_size:
        raise ValueError("Noise factor rows must match descriptor input size.")
    response = descriptor_frequency_response(system, angular_frequency)
    factor = response.response @ source_factor.factor
    covariance = factor @ jnp.swapaxes(jnp.conj(factor), -1, -2)
    hermitian = covariance - jnp.swapaxes(jnp.conj(covariance), -1, -2)
    scale = jnp.maximum(jnp.linalg.norm(covariance, axis=(-2, -1)), 1.0)
    defect = jnp.linalg.norm(hermitian, axis=(-2, -1)) / scale
    eigenvalues = jnp.linalg.eigvalsh(
        0.5 * (covariance + jnp.swapaxes(jnp.conj(covariance), -1, -2))
    )
    minimum = jnp.min(eigenvalues, axis=-1)
    tolerance = 100 * jnp.finfo(covariance.real.dtype).eps * scale
    finite = jnp.all(jnp.isfinite(covariance), axis=(-2, -1))
    diagnostics = CircuitNoiseDiagnostics(
        defect,
        minimum,
        minimum >= -tolerance,
        finite,
        response.successful,
    )
    return CircuitNoiseResult(
        factor,
        covariance,
        response.response,
        diagnostics,
        source_factor.source_ids,
        system.system_id,
    )


__all__ = [
    "CircuitNoiseDiagnostics",
    "CircuitNoiseResult",
    "NoiseSpectralFactor",
    "propagate_descriptor_noise",
    "thermal_resistor_noise_factor",
]
