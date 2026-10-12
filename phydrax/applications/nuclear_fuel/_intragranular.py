#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Intragranular fission gas diffusion in the Booth (AECL-496, 1957) equivalent sphere.

Speight (Nucl. Sci. Eng. 37 (1969) 180) effective diffusivity ``D b / (b + g)``:
single-atom diffusivity of Turnbull et al. (J. Nucl. Mater. 107 (1982) 168),
trapping ``g = 4 pi D R N`` and re-solution ``b = 3.03 pi mu_f F (R + Z_0)^2``
with the White and Tucker (J. Nucl. Mater. 118 (1983) 1) intragranular bubble
radius and density. ``modal_constants`` returns the sine-mode basis
``psi_m(x) = (-1)^(m+1) sin(m pi x) / (m pi x)`` of the equivalent sphere.
"""

from __future__ import annotations

from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ._constants import BOLTZMANN_J_PER_K
from ._gas_math import (
    host_parameters,
    positive_square_root,
)


@final
class TurnbullDiffusivity(StrictModule):
    """Single gas atom diffusivity ``D1 + D2 + D3`` [m^2/s] in UO2.

    Turnbull et al., J. Nucl. Mater. 107 (1982) 168: ``D1`` intrinsic,
    ``D2`` irradiation-enhanced cation-vacancy term proportional to
    ``sqrt(F)``, ``D3 = 2e-40 F`` athermal. ``D1`` and ``D3`` coefficients
    from NUREG/CR-7003 (Turnbull and Beyer, 2010) Eqs. (6) and (8); the ``D2``
    prefactor ``5.64e-25`` (four times the ``1.41e-25`` of the revised ANS-5.4
    expression) as tabulated by Barani (PhD thesis, Politecnico di Milano,
    2020, Table 3.4) and implemented in BISON Sifgrs (``D = D1 + 4 D2``).
    """

    intrinsic_prefactor: Array
    intrinsic_activation_energy: Array
    irradiation_prefactor: Array
    irradiation_activation_energy: Array
    athermal_coefficient: Array
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        intrinsic_prefactor: ArrayLike = 7.6e-10,
        intrinsic_activation_energy: ArrayLike = 4.86e-19,
        irradiation_prefactor: ArrayLike = 5.64e-25,
        irradiation_activation_energy: ArrayLike = 1.91e-19,
        athermal_coefficient: ArrayLike = 2.0e-40,
    ) -> None:
        parameters = host_parameters(
            {
                "intrinsic_prefactor": intrinsic_prefactor,
                "intrinsic_activation_energy": intrinsic_activation_energy,
                "irradiation_prefactor": irradiation_prefactor,
                "irradiation_activation_energy": irradiation_activation_energy,
                "athermal_coefficient": athermal_coefficient,
            }
        )
        self.intrinsic_prefactor = jnp.asarray(parameters["intrinsic_prefactor"])
        self.intrinsic_activation_energy = jnp.asarray(
            parameters["intrinsic_activation_energy"]
        )
        self.irradiation_prefactor = jnp.asarray(parameters["irradiation_prefactor"])
        self.irradiation_activation_energy = jnp.asarray(
            parameters["irradiation_activation_energy"]
        )
        self.athermal_coefficient = jnp.asarray(parameters["athermal_coefficient"])
        self.source_id = "turnbull-1982-jnm-107-168:d1+d2+d3"
        self.law_id = canonical_fingerprint(
            {"kind": "turnbull-single-atom-diffusivity", "parameters": parameters}
        )

    def diffusivity(
        self, temperature: ArrayLike, fission_rate_density: ArrayLike, /
    ) -> Array:
        """``D`` [m^2/s] at temperature [K] and fission rate density [1/m^3/s]."""

        thermal_energy = BOLTZMANN_J_PER_K * jnp.asarray(temperature)
        fission_rate = jnp.asarray(fission_rate_density, dtype=jnp.float64)
        return (
            self.intrinsic_prefactor
            * jnp.exp(-self.intrinsic_activation_energy / thermal_energy)
            + self.irradiation_prefactor
            * positive_square_root(fission_rate)
            * jnp.exp(-self.irradiation_activation_energy / thermal_energy)
            + self.athermal_coefficient * fission_rate
        )


@final
class SpeightTrapping(StrictModule):
    """Intragranular trapping and irradiation re-solution for Speight's diffusivity.

    Pastore et al. (INL/CON-13-28389, 2013, Eqs. (2)-(4) and Table I; the
    model of Nucl. Eng. Des. 256 (2013) 75): ``g = 4 pi D R N`` (Ham, 1958),
    ``b = 3.03 pi mu_f F (R + Z_0)^2`` with the White and Tucker (1983)
    correlations ``R = 5e-10 (1 + 106 exp(-1.20e-19 / (k T)))`` m and
    ``N = 1.52e27 / T - 3.3e23`` m^-3. ``bubble_atomic_volume`` is the volume
    per gas atom in intragranular bubbles (Barani thesis 2020, Eq. (3.5) and
    Table 3.4), so the bubble swelling is ``Omega g / (b + g) C``.
    """

    bubble_radius_scale: Array
    bubble_radius_factor: Array
    bubble_radius_activation_energy: Array
    bubble_density_coefficient: Array
    bubble_density_offset: Array
    resolution_coefficient: Array
    fission_fragment_range: Array
    fission_fragment_influence_radius: Array
    bubble_atomic_volume: Array
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        bubble_radius_scale: ArrayLike = 5.0e-10,
        bubble_radius_factor: ArrayLike = 106.0,
        bubble_radius_activation_energy: ArrayLike = 1.20e-19,
        bubble_density_coefficient: ArrayLike = 1.52e27,
        bubble_density_offset: ArrayLike = 3.3e23,
        resolution_coefficient: ArrayLike = 3.03,
        fission_fragment_range: ArrayLike = 6.0e-6,
        fission_fragment_influence_radius: ArrayLike = 1.0e-9,
        bubble_atomic_volume: ArrayLike = 4.09e-29,
    ) -> None:
        parameters = host_parameters(
            {
                "bubble_radius_scale": bubble_radius_scale,
                "bubble_radius_factor": bubble_radius_factor,
                "bubble_radius_activation_energy": bubble_radius_activation_energy,
                "bubble_density_coefficient": bubble_density_coefficient,
                "bubble_density_offset": bubble_density_offset,
                "resolution_coefficient": resolution_coefficient,
                "fission_fragment_range": fission_fragment_range,
                "fission_fragment_influence_radius": fission_fragment_influence_radius,
                "bubble_atomic_volume": bubble_atomic_volume,
            }
        )
        self.bubble_radius_scale = jnp.asarray(parameters["bubble_radius_scale"])
        self.bubble_radius_factor = jnp.asarray(parameters["bubble_radius_factor"])
        self.bubble_radius_activation_energy = jnp.asarray(
            parameters["bubble_radius_activation_energy"]
        )
        self.bubble_density_coefficient = jnp.asarray(
            parameters["bubble_density_coefficient"]
        )
        self.bubble_density_offset = jnp.asarray(parameters["bubble_density_offset"])
        self.resolution_coefficient = jnp.asarray(parameters["resolution_coefficient"])
        self.fission_fragment_range = jnp.asarray(parameters["fission_fragment_range"])
        self.fission_fragment_influence_radius = jnp.asarray(
            parameters["fission_fragment_influence_radius"]
        )
        self.bubble_atomic_volume = jnp.asarray(parameters["bubble_atomic_volume"])
        self.source_id = "pastore-inl-con-13-28389:eq2-4;white-tucker-1983-jnm-118-1"
        self.law_id = canonical_fingerprint(
            {"kind": "speight-trapping-resolution", "parameters": parameters}
        )

    def bubble_radius(self, temperature: ArrayLike, /) -> Array:
        """White-Tucker mean intragranular bubble radius [m]."""

        return self.bubble_radius_scale * (
            1.0
            + self.bubble_radius_factor
            * jnp.exp(
                -self.bubble_radius_activation_energy
                / (BOLTZMANN_J_PER_K * jnp.asarray(temperature))
            )
        )

    def bubble_density(self, temperature: ArrayLike, /) -> Array:
        """White-Tucker intragranular bubble number density [1/m^3]."""

        return (
            self.bubble_density_coefficient / jnp.asarray(temperature)
            - self.bubble_density_offset
        )

    def rates(
        self,
        temperature: ArrayLike,
        fission_rate_density: ArrayLike,
        diffusivity: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        """Trapping ``g`` and re-solution ``b`` rates [1/s]."""

        radius = self.bubble_radius(temperature)
        trapping = (
            4.0
            * jnp.pi
            * jnp.asarray(diffusivity)
            * radius
            * self.bubble_density(temperature)
        )
        resolution = (
            self.resolution_coefficient
            * jnp.pi
            * self.fission_fragment_range
            * jnp.asarray(fission_rate_density)
            * (radius + self.fission_fragment_influence_radius) ** 2
        )
        return trapping, resolution


def modal_constants(
    mode_count: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Mode numbers, mean weights, reconstruction basis and ``1 - Z_N``."""

    modes = np.arange(1, mode_count + 1, dtype=np.float64)
    weights = 3.0 / (modes * np.pi) ** 2
    # Points r/a = j / (2N), j = 0 .. 2N - 1: center included, sink excluded.
    points = np.arange(2 * mode_count, dtype=np.float64) / (2.0 * mode_count)
    signs = np.where(modes % 2.0 == 1.0, 1.0, -1.0)
    basis = signs[None, :] * np.sinc(points[:, None] * modes[None, :])
    unresolved = 1.0 - float(np.sum(2.0 * weights))
    return modes, weights, basis, unresolved


__all__ = [
    "SpeightTrapping",
    "TurnbullDiffusivity",
    "modal_constants",
]
