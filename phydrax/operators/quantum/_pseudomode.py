#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


class BathCorrelationExpansion(StrictModule):
    coefficients: Array
    exponents: Array
    fit_residual: Array
    valid: Array
    expansion_id: str = eqx.field(static=True)

    def __init__(
        self,
        coefficients: ArrayLike,
        exponents: ArrayLike,
        /,
        *,
        expansion_id: str,
        fit_residual: ArrayLike = 0.0,
    ):
        coefficients_ = jnp.asarray(coefficients)
        exponents_ = jnp.asarray(exponents, dtype=coefficients_.dtype)
        if coefficients_.ndim != 1 or exponents_.shape != coefficients_.shape:
            raise ValueError("Bath expansion coefficients/exponents must be vectors.")
        self.coefficients = coefficients_
        self.exponents = exponents_
        self.fit_residual = jnp.asarray(fit_residual)
        self.valid = (
            jnp.all(jnp.isfinite(coefficients_))
            & jnp.all(jnp.isfinite(exponents_))
            & jnp.all(jnp.real(exponents_) > 0.0)
            & jnp.isfinite(self.fit_residual)
        )
        self.expansion_id = str(expansion_id)

    @property
    def rank(self) -> int:
        return int(self.coefficients.shape[0])

    def __call__(self, time: ArrayLike, /) -> Array:
        value = jnp.asarray(time)
        return jnp.sum(
            self.coefficients * jnp.exp(-value[..., None] * self.exponents), axis=-1
        )

    def residual_against(
        self,
        target: Callable[[Array], Array],
        times: ArrayLike,
        /,
    ) -> Array:
        values = jnp.asarray(times)
        reference = jnp.asarray(target(values))
        if reference.shape != values.shape:
            raise ValueError("Bath target must preserve time shape.")
        return jnp.sqrt(jnp.mean(jnp.abs(self(values) - reference) ** 2))


class Pseudomode(StrictModule):
    frequency: float = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    coupling: complex = eqx.field(static=True)
    cutoff: int = eqx.field(static=True)
    mode_id: str = eqx.field(static=True)

    def __init__(
        self,
        frequency: float,
        damping: float,
        coupling: complex,
        /,
        *,
        cutoff: int,
        mode_id: str,
    ):
        if damping < 0.0 or int(cutoff) < 2:
            raise ValueError("Pseudomode damping/cutoff are invalid.")
        self.frequency = float(frequency)
        self.damping = float(damping)
        self.coupling = complex(coupling)
        self.cutoff = int(cutoff)
        self.mode_id = str(mode_id)


class ReactionCoordinateMapping(StrictModule):
    frequency: float = eqx.field(static=True)
    coupling: float = eqx.field(static=True)
    residual_damping: float = eqx.field(static=True)
    mapping_id: str = eqx.field(static=True)

    def __init__(
        self,
        frequency: float,
        coupling: float,
        residual_damping: float,
        /,
        *,
        mapping_id: str,
    ):
        if coupling < 0.0 or residual_damping < 0.0:
            raise ValueError("Reaction-coordinate parameters must be non-negative.")
        self.frequency = float(frequency)
        self.coupling = float(coupling)
        self.residual_damping = float(residual_damping)
        self.mapping_id = str(mapping_id)


def lorentzian_pseudomode(
    center_frequency: float,
    linewidth: float,
    coupling: float,
    /,
    *,
    cutoff: int = 8,
) -> tuple[BathCorrelationExpansion, Pseudomode, ReactionCoordinateMapping]:
    coefficient = float(coupling) ** 2
    exponent = 0.5 * float(linewidth) + 1j * float(center_frequency)
    expansion = BathCorrelationExpansion(
        jnp.asarray([coefficient], dtype=complex),
        jnp.asarray([exponent], dtype=complex),
        expansion_id="lorentzian-one-pole",
    )
    mode = Pseudomode(
        center_frequency,
        linewidth,
        coupling,
        cutoff=cutoff,
        mode_id="lorentzian-pseudomode",
    )
    mapping = ReactionCoordinateMapping(
        center_frequency,
        coupling,
        linewidth,
        mapping_id="lorentzian-reaction-coordinate",
    )
    return expansion, mode, mapping


__all__ = [
    "BathCorrelationExpansion",
    "Pseudomode",
    "ReactionCoordinateMapping",
    "lorentzian_pseudomode",
]
