#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Continuum, compressibility and curvature validity evidence for bubbles."""

from __future__ import annotations

from typing import assert_never

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._radial import RadialBubbleEquation


BOLTZMANN_CONSTANT = 1.380649e-23
"""Exact SI Boltzmann constant in J K⁻¹."""


def _positive_or_none(value: float | None, name: str, /) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite or None.")
    return number


def _positive(value: float, name: str, /) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return number


class BubbleValidityPolicy(StrictModule, NonTrainableState):
    """Thresholds that decide whether a solution stays inside model support.

    `molecular_diameter` (kinetic gas diameter) enables Knudsen-number evidence
    `Kn = k_B T/(√2 π d² p R)`; `tolman_length` enables the curvature ratio
    `2|δ|/R`. Both are evidence only: they never modify the dynamics (the Tolman
    correction of the surface tension is a separate, explicit policy).
    """

    mach_limit: float = eqx.field(static=True)
    knudsen_limit: float = eqx.field(static=True)
    tolman_ratio_limit: float = eqx.field(static=True)
    molecular_diameter: float | None = eqx.field(static=True)
    tolman_length: float | None = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        mach_limit: float = 1.0,
        knudsen_limit: float = 0.1,
        tolman_ratio_limit: float = 0.1,
        molecular_diameter: float | None = None,
        tolman_length: float | None = None,
    ) -> None:
        mach = _positive(mach_limit, "mach_limit")
        knudsen = _positive(knudsen_limit, "knudsen_limit")
        tolman_limit = _positive(tolman_ratio_limit, "tolman_ratio_limit")
        diameter = _positive_or_none(molecular_diameter, "molecular_diameter")
        tolman = None if tolman_length is None else float(tolman_length)
        if tolman is not None and not np.isfinite(tolman):
            raise ValueError("tolman_length must be finite or None.")
        self.mach_limit = mach
        self.knudsen_limit = knudsen
        self.tolman_ratio_limit = tolman_limit
        self.molecular_diameter = diameter
        self.tolman_length = tolman
        self.policy_id = canonical_fingerprint(
            {
                "kind": "bubble-validity-policy",
                "mach_limit": mach,
                "knudsen_limit": knudsen,
                "tolman_ratio_limit": tolman_limit,
                "molecular_diameter": diameter,
                "tolman_length": tolman,
            }
        )

    def knudsen_number(self, temperature: Array, pressure: Array, radius: Array, /) -> Array | None:
        """Gas Knudsen number based on the bubble radius, when a diameter is declared."""
        if self.molecular_diameter is None:
            return None
        mean_free_path = BOLTZMANN_CONSTANT * temperature / (
            jnp.sqrt(2.0) * jnp.pi * self.molecular_diameter**2 * pressure
        )
        return mean_free_path / radius

    def tolman_ratio(self, radius: Array, /) -> Array | None:
        """Curvature ratio `2|δ|/R`, when a Tolman length is declared."""
        if self.tolman_length is None:
            return None
        return 2.0 * abs(self.tolman_length) / radius


class BubbleValidityEvidence(StrictModule):
    """Extremes over the accepted trajectory and the resulting support decision.

    `max_laplace_ratio` is the capillary pressure divided by the ambient pressure
    (infinite for a zero ambient pressure). Optional entries are `None` when the
    policy does not declare the required molecular property.
    """

    max_wall_mach: Array
    max_wall_pressure: Array
    min_radius_ratio: Array
    max_radius_ratio: Array
    max_gas_temperature: Array
    min_hard_core_margin: Array
    max_laplace_ratio: Array
    max_knudsen: Array | None
    max_tolman_ratio: Array | None
    within_support: Array
    neglected_terms: tuple[str, ...] = eqx.field(static=True)


def neglected_terms(equation: RadialBubbleEquation, /) -> tuple[str, ...]:
    """Physics omitted by each radial equation, reported with every solution."""
    common = ("nonspherical-modes", "translation", "gas-liquid-mass-transfer")
    match equation:
        case "rayleigh_plesset":
            return ("liquid-compressibility", "acoustic-radiation") + common
        case "rayleigh_plesset_radiation":
            return ("second-order-compressibility",) + common
        case "rayleigh_plesset_gas_radiation":
            return (
                "second-order-compressibility",
                "non-gas-pressure-radiation",
            ) + common
        case "keller_miksis":
            return ("second-order-compressibility", "nonlinear-liquid-equation-of-state") + common
        case "gilmore":
            return ("liquid-shock-formation", "liquid-thermal-effects") + common
        case _:
            assert_never(equation)


__all__ = [
    "BOLTZMANN_CONSTANT",
    "BubbleValidityEvidence",
    "BubbleValidityPolicy",
    "neglected_terms",
]
