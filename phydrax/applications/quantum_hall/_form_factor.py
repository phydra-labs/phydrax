#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-subband Coulomb form factors for quasi-two-dimensional Hall systems."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from scipy.special import roots_genlaguerre

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class SubbandCoulombFormFactorPlan(StrictModule, NonTrainableState):
    positions_meter: Array
    probability_density_per_meter: Array
    normalization_residual: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        positions_meter: ArrayLike,
        probability_density_per_meter: ArrayLike,
        /,
        *,
        normalization_tolerance: float = 1.0e-8,
    ):
        positions = np.asarray(positions_meter, dtype=np.float64)
        density = np.asarray(probability_density_per_meter, dtype=np.float64)
        tolerance = float(normalization_tolerance)
        if (
            positions.ndim != 1
            or positions.size < 2
            or density.shape != positions.shape
            or np.any(~np.isfinite(positions))
            or np.any(~np.isfinite(density))
            or np.any(density < 0.0)
            or np.any(np.diff(positions) <= 0.0)
            or not isfinite(tolerance)
            or tolerance < 0.0
        ):
            raise ValueError("Subband positions, density, or tolerance is invalid.")
        normalization = float(np.trapezoid(density, positions))
        residual = abs(normalization - 1.0)
        if residual > tolerance:
            raise ValueError("Subband probability density is not normalized in meters.")
        self.positions_meter = jnp.asarray(positions)
        self.probability_density_per_meter = jnp.asarray(density)
        self.normalization_residual = residual
        self.plan_id = canonical_fingerprint(
            {
                "kind": "subband-coulomb-form-factor-plan",
                "arrays": array_tree_fingerprint(
                    {"positions_meter": positions, "density": density}
                ),
                "normalization_tolerance": tolerance,
            }
        )


class SubbandCoulombFormFactorResult(StrictModule, NonTrainableState):
    wave_numbers_per_meter: Array
    form_factors: Array
    normalization_residual: Array
    zero_momentum_residual: Array
    bounded: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


def evaluate_subband_coulomb_form_factor(
    plan: SubbandCoulombFormFactorPlan,
    wave_numbers_per_meter: ArrayLike,
    /,
) -> SubbandCoulombFormFactorResult:
    if not isinstance(plan, SubbandCoulombFormFactorPlan):
        raise TypeError("plan must be SubbandCoulombFormFactorPlan.")
    wave_numbers = np.asarray(wave_numbers_per_meter, dtype=np.float64)
    if (
        wave_numbers.ndim != 1
        or wave_numbers.size < 1
        or np.any(~np.isfinite(wave_numbers))
        or np.any(wave_numbers < 0.0)
    ):
        raise ValueError("Wave numbers must be a nonempty finite non-negative vector.")
    positions = np.asarray(plan.positions_meter)
    density = np.asarray(plan.probability_density_per_meter)
    intervals = np.diff(positions)
    weights = np.empty_like(positions)
    weights[0] = 0.5 * intervals[0]
    weights[-1] = 0.5 * intervals[-1]
    weights[1:-1] = 0.5 * (intervals[:-1] + intervals[1:])
    probability_weights = weights * density
    separation = np.abs(positions[:, None] - positions[None, :])
    factors = np.asarray(
        tuple(
            np.sum(
                probability_weights[:, None]
                * probability_weights[None, :]
                * np.exp(-wave_number * separation)
            )
            for wave_number in wave_numbers
        )
    )
    zero_residual = (
        abs(float(factors[np.argmin(wave_numbers)]) - 1.0)
        if float(np.min(wave_numbers)) == 0.0
        else np.nan
    )
    bounded = bool(np.all((factors >= -1.0e-12) & (factors <= 1.0 + 1.0e-12)))
    successful = bool(bounded and np.all(np.isfinite(factors)))
    return SubbandCoulombFormFactorResult(
        jnp.asarray(wave_numbers),
        jnp.asarray(factors),
        jnp.asarray(plan.normalization_residual),
        jnp.asarray(zero_residual),
        jnp.asarray(bounded),
        jnp.asarray(successful),
        plan.plan_id,
        canonical_fingerprint(
            {
                "kind": "subband-coulomb-form-factor-result",
                "plan": plan.plan_id,
                "arrays": array_tree_fingerprint(
                    {"wave_numbers": wave_numbers, "form_factors": factors}
                ),
            }
        ),
    )


def apply_subband_form_factor(
    momentum_interaction: ArrayLike,
    form_factor: SubbandCoulombFormFactorResult,
    /,
) -> Array:
    if not isinstance(form_factor, SubbandCoulombFormFactorResult):
        raise TypeError("form_factor must be SubbandCoulombFormFactorResult.")
    interaction = jnp.asarray(momentum_interaction)
    if interaction.shape != form_factor.form_factors.shape:
        raise ValueError("Momentum interaction must align with form-factor samples.")
    return interaction * form_factor.form_factors


class PlanarCoulombPseudopotentialResult(StrictModule, NonTrainableState):
    relative_channels: tuple[tuple[int, float], ...] = eqx.field(static=True)
    landau_level: int = eqx.field(static=True)
    quadrature_order: int = eqx.field(static=True)
    refinement_residual: Array
    successful: Array
    result_id: str = eqx.field(static=True)


def _planar_channels(level: int, maximum_relative: int, order: int, /):
    nodes, weights = roots_genlaguerre(order, -0.5)
    level_coefficients = np.zeros((level + 1,), dtype=np.float64)
    level_coefficients[-1] = 1.0
    level_form = np.polynomial.laguerre.lagval(0.5 * nodes, level_coefficients)
    base = 0.5 * weights * level_form**2
    channels = []
    for relative in range(maximum_relative + 1):
        coefficients = np.zeros((relative + 1,), dtype=np.float64)
        coefficients[-1] = 1.0
        channels.append(
            (
                relative,
                float(np.sum(base * np.polynomial.laguerre.lagval(nodes, coefficients))),
            )
        )
    return tuple(channels)


def planar_coulomb_pseudopotentials(
    landau_level: int,
    maximum_relative: int,
    /,
    *,
    quadrature_order: int = 128,
) -> PlanarCoulombPseudopotentialResult:
    """Return planar Coulomb V_m^n in units of e²/(epsilon magnetic_length)."""

    level = int(landau_level)
    maximum = int(maximum_relative)
    order = int(quadrature_order)
    if level < 0 or maximum < 0 or order < 16:
        raise ValueError(
            "Landau level, relative channel, or quadrature order is invalid."
        )
    channels = _planar_channels(level, maximum, order)
    refined = _planar_channels(level, maximum, 2 * order)
    residual = max(
        abs(left[1] - right[1]) for left, right in zip(channels, refined, strict=True)
    )
    successful = bool(np.isfinite(residual) and residual <= 1.0e-10)
    return PlanarCoulombPseudopotentialResult(
        refined,
        level,
        order,
        jnp.asarray(residual),
        jnp.asarray(successful),
        canonical_fingerprint(
            {
                "kind": "planar-coulomb-pseudopotentials",
                "landau_level": level,
                "maximum_relative": maximum,
                "quadrature_order": order,
                "channels": refined,
            }
        ),
    )


__all__ = [
    "SubbandCoulombFormFactorPlan",
    "SubbandCoulombFormFactorResult",
    "PlanarCoulombPseudopotentialResult",
    "apply_subband_form_factor",
    "evaluate_subband_coulomb_form_factor",
    "planar_coulomb_pseudopotentials",
]
