#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class NeutrinoMassOrdering(StrEnum):
    NORMAL = "normal"
    INVERTED = "inverted"


class NeutrinoOscillationParameters(StrictModule, NonTrainableState):
    theta12: Array
    theta13: Array
    theta23: Array
    delta_cp: Array
    delta_m21_squared: Array
    delta_m31_squared: Array
    ordering: NeutrinoMassOrdering = eqx.field(static=True)
    parameter_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        theta12: float,
        theta13: float,
        theta23: float,
        delta_cp: float,
        delta_m21_squared: float,
        delta_m31_squared: float,
        ordering: NeutrinoMassOrdering,
    ):
        angles = tuple(map(float, (theta12, theta13, theta23, delta_cp)))
        splittings = tuple(map(float, (delta_m21_squared, delta_m31_squared)))
        if (
            any(not math.isfinite(value) for value in angles + splittings)
            or any(not 0.0 <= value <= 0.5 * math.pi for value in angles[:3])
            or splittings[0] <= 0.0
            or not isinstance(ordering, NeutrinoMassOrdering)
        ):
            raise ValueError(
                "Neutrino mixing angles, splittings, or ordering are invalid."
            )
        if ordering is NeutrinoMassOrdering.NORMAL and splittings[1] <= 0.0:
            raise ValueError("Normal ordering requires positive delta_m31_squared.")
        if ordering is NeutrinoMassOrdering.INVERTED and splittings[1] >= 0.0:
            raise ValueError("Inverted ordering requires negative delta_m31_squared.")
        dtype = jnp.float64
        self.theta12, self.theta13, self.theta23, self.delta_cp = (
            jnp.asarray(value, dtype=dtype) for value in angles
        )
        self.delta_m21_squared, self.delta_m31_squared = (
            jnp.asarray(value, dtype=dtype) for value in splittings
        )
        self.ordering = ordering
        self.parameter_id = canonical_fingerprint(
            {
                "kind": "neutrino-oscillation-parameters",
                "angles": list(angles),
                "splittings": list(splittings),
                "ordering": ordering.value,
            }
        )


def _pmns(parameters: NeutrinoOscillationParameters, antineutrino: bool):
    s12, c12 = jnp.sin(parameters.theta12), jnp.cos(parameters.theta12)
    s13, c13 = jnp.sin(parameters.theta13), jnp.cos(parameters.theta13)
    s23, c23 = jnp.sin(parameters.theta23), jnp.cos(parameters.theta23)
    phase = jnp.exp(-1j * parameters.delta_cp)
    matrix = jnp.asarray(
        [
            [c12 * c13, s12 * c13, s13 * phase],
            [
                -s12 * c23 - c12 * s23 * s13 / phase,
                c12 * c23 - s12 * s23 * s13 / phase,
                s23 * c13,
            ],
            [
                s12 * s23 - c12 * c23 * s13 / phase,
                -c12 * s23 - s12 * c23 * s13 / phase,
                c23 * c13,
            ],
        ]
    )
    return jnp.conj(matrix) if antineutrino else matrix


class OscillationProbabilityResult(StrictModule, NonTrainableState):
    probabilities: Array
    row_unitarity_residual: Array
    column_unitarity_residual: Array
    finite: Array
    valid: Array
    parameter_id: str = eqx.field(static=True)


def oscillation_probabilities(
    parameters: NeutrinoOscillationParameters,
    energies_gev: ArrayLike,
    baseline_km: ArrayLike,
    /,
    *,
    matter_density_g_cm3: ArrayLike = 0.0,
    electron_fraction: float = 0.5,
    antineutrino: bool = False,
) -> OscillationProbabilityResult:
    """Evaluate three-flavor oscillation probabilities in constant-density matter."""
    if not isinstance(parameters, NeutrinoOscillationParameters):
        raise TypeError("parameters must be NeutrinoOscillationParameters.")
    energies = jnp.asarray(energies_gev, dtype=jnp.float64)
    baseline = jnp.asarray(baseline_km, dtype=energies.dtype)
    density = jnp.asarray(matter_density_g_cm3, dtype=energies.dtype)
    energies, baseline, density = jnp.broadcast_arrays(energies, baseline, density)
    if not math.isfinite(float(electron_fraction)) or not 0.0 <= electron_fraction <= 1.0:
        raise ValueError("electron_fraction must lie in [0, 1].")
    mixing = _pmns(parameters, antineutrino)
    masses_squared = jnp.asarray(
        [0.0, parameters.delta_m21_squared, parameters.delta_m31_squared]
    )

    def one(energy, distance, matter_density):
        vacuum_mass = mixing @ jnp.diag(masses_squared) @ jnp.conj(mixing.T)
        matter_potential = 7.56e-5 * matter_density * float(electron_fraction) * energy
        matter_sign = -1.0 if antineutrino else 1.0
        effective = vacuum_mass.at[0, 0].add(matter_sign * matter_potential)
        eigenvalues, eigenvectors = jnp.linalg.eigh(effective)
        phases = jnp.exp(-1j * 2.533865 * eigenvalues * distance / energy)
        amplitude = eigenvectors @ jnp.diag(phases) @ jnp.conj(eigenvectors.T)
        return jnp.abs(amplitude.T) ** 2

    safe_energy = jnp.maximum(energies, jnp.finfo(energies.dtype).tiny)
    probabilities = jax.vmap(one)(
        safe_energy.reshape((-1,)), baseline.reshape((-1,)), density.reshape((-1,))
    ).reshape(energies.shape + (3, 3))
    row_residual = jnp.max(jnp.abs(jnp.sum(probabilities, axis=-1) - 1.0), axis=-1)
    column_residual = jnp.max(jnp.abs(jnp.sum(probabilities, axis=-2) - 1.0), axis=-1)
    finite = jnp.all(jnp.isfinite(probabilities), axis=(-1, -2))
    valid = (
        (energies > 0.0)
        & (baseline >= 0.0)
        & (density >= 0.0)
        & finite
        & (row_residual <= 1.0e-10)
        & (column_residual <= 1.0e-10)
    )
    return OscillationProbabilityResult(
        probabilities,
        row_residual,
        column_residual,
        finite,
        valid,
        parameters.parameter_id,
    )


__all__ = [
    "NeutrinoMassOrdering",
    "NeutrinoOscillationParameters",
    "OscillationProbabilityResult",
    "oscillation_probabilities",
]
