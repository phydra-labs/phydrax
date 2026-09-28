#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Planck and Rosseland means of a spectral absorption coefficient.

Means are taken over frequency ``ν`` (cycles per time unit):

- ``κ_P(T) = ∫ α_ν B_ν(T) dν / ∫ B_ν(T) dν`` (emission, at the matter temperature);
- ``κ_a = ∫ α_ν B_ν(T_r) dν / ∫ B_ν(T_r) dν`` (absorption of a blackbody field at the
  radiation temperature by matter at ``T``);
- ``1/κ_R = ∫ α_ν⁻¹ ∂B_ν/∂T dν / ∫ ∂B_ν/∂T dν`` (Rosseland, at ``T``).

The denominators are the exact closed forms ``σT⁴/π`` and ``4σT³/π``. Numerators
are integrated in ``ln ν`` over the spectral model's declared support window with
a composite embedded Gauss–Kronrod rule; a mean is supported only when its
integrand has decayed at both window edges and the Kronrod–Gauss difference is
within tolerance, so an unconverged or out-of-support mean is reported, never
silently truncated.
"""

from __future__ import annotations

from collections.abc import Callable
from math import pi

import jax.numpy as jnp
from jax import Array

from .._numerics._quadrature_rules import gauss_kronrod_data
from .._strict import StrictModule


class GrayMeanOpacities(StrictModule):
    """Planck, radiation-temperature Planck, and Rosseland means per length.

    ``edge_fraction[..., k]`` bounds the omitted tail of mean ``k`` (emission,
    absorption, Rosseland) by the integrand at the window edges times the log
    width of the window, relative to the integral; ``quadrature_error`` is the
    relative Kronrod–Gauss difference. ``*_supported`` holds where both are within
    the declared tolerance and the spectral model qualified every node.
    """

    planck_emission: Array
    planck_absorption: Array
    rosseland: Array
    edge_fraction: Array
    quadrature_error: Array
    emission_supported: Array
    absorption_supported: Array
    rosseland_supported: Array


def _log_planck(
    frequency: Array, temperature: Array, h: float, k: float, c: float
) -> Array:
    x = h * frequency / (k * temperature)
    log_denominator = jnp.where(x > 50.0, x, jnp.log(jnp.expm1(jnp.minimum(x, 50.0))))
    return jnp.log(2.0 * h / (c * c)) + 3.0 * jnp.log(frequency) - log_denominator


def gray_mean_opacities(
    absorption: Callable[[Array], tuple[Array, Array]],
    temperature: Array,
    radiation_temperature: Array,
    lower_frequency: Array,
    upper_frequency: Array,
    /,
    *,
    planck_constant: float,
    boltzmann_constant: float,
    speed_of_light: float,
    panels: int,
    order: int,
    tolerance: float,
) -> GrayMeanOpacities:
    """Means of ``absorption(ν) -> (α_ν, node_qualified)`` over a log-ν window.

    ``absorption`` receives frequencies of shape ``batch + (nodes,)`` and must
    broadcast its own state along the trailing node axis.
    """
    rule = gauss_kronrod_data(order)
    if rule.embedded_weights is None:
        raise RuntimeError("The Gauss–Kronrod rule carries no embedded Gauss weights.")
    unit = 0.5 * (rule.nodes + 1.0)
    kronrod = 0.5 * rule.weights
    gauss = 0.5 * rule.embedded_weights
    lower = jnp.log(lower_frequency)[..., None]
    width = (jnp.log(upper_frequency) - jnp.log(lower_frequency))[..., None]
    panel_width = width / panels
    offsets = jnp.arange(panels, dtype=jnp.float64)
    local = (offsets[:, None] + unit[None, :]).reshape((-1,))
    log_frequency = lower + panel_width * local
    kronrod_weights = jnp.tile(kronrod, panels) * panel_width
    gauss_weights = jnp.tile(gauss, panels) * panel_width
    frequency = jnp.exp(log_frequency)
    h, k, c = planck_constant, boltzmann_constant, speed_of_light
    matter = temperature[..., None]
    alpha, node_qualified = absorption(frequency)
    log_matter = _log_planck(frequency, matter, h, k, c)
    log_radiation = _log_planck(frequency, radiation_temperature[..., None], h, k, c)
    x = h * frequency / (k * matter)
    # ∂B/∂T = B · x eˣ/(eˣ − 1) / T = B · x / (−expm1(−x)) / T.
    log_derivative = log_matter + jnp.log(x / -jnp.expm1(-x)) - jnp.log(matter)
    integrands = jnp.stack(
        (
            alpha * jnp.exp(log_matter + log_frequency),
            alpha * jnp.exp(log_radiation + log_frequency),
            jnp.exp(log_derivative + log_frequency) / alpha,
        ),
        axis=0,
    )
    integral = jnp.sum(integrands * kronrod_weights, axis=-1)
    embedded = jnp.sum(integrands * gauss_weights, axis=-1)
    edge = jnp.maximum(integrands[..., 0], integrands[..., -1]) * width[..., 0]
    edge_fraction = jnp.moveaxis(edge / integral, 0, -1)
    quadrature_error = jnp.moveaxis(jnp.abs(integral - embedded) / integral, 0, -1)
    stefan = 2.0 * pi**4 * k**4 / (15.0 * h**3 * c * c)
    planck_norm = stefan * temperature**4
    radiation_norm = stefan * radiation_temperature**4
    rosseland_norm = 4.0 * stefan * temperature**3
    converged = (edge_fraction <= tolerance) & (quadrature_error <= tolerance)
    qualified = jnp.all(node_qualified, axis=-1)
    return GrayMeanOpacities(
        planck_emission=integral[0] / planck_norm,
        planck_absorption=integral[1] / radiation_norm,
        rosseland=rosseland_norm / integral[2],
        edge_fraction=edge_fraction,
        quadrature_error=quadrature_error,
        emission_supported=qualified & converged[..., 0],
        absorption_supported=qualified & converged[..., 1],
        rosseland_supported=qualified & converged[..., 2],
    )


__all__ = ["GrayMeanOpacities", "gray_mean_opacities"]
