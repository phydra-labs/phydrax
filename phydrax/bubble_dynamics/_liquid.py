#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Liquid rheology around a spherical bubble.

Each law returns the wall stress integral `S = 2 ∫_R^∞ (τ_rr − τ_θθ)/r dr` of
the incompressible radial flow `u = R²Ṙ/r²`; it enters every radial equation as
an additive wall-pressure contribution. Relaxation variables are ODE state.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._trainable import fixed_field, parameter_field
from .._validation import positive_integer
from ._contracts import (
    AbstractBubbleLiquidLaw,
    BubbleEnvironment,
    BubbleLiquidEvaluation,
    BubbleScales,
    scalar_parameter,
)


def _viscous_evaluation(stress: Array, radius: Array, wall_velocity: Array, /) -> BubbleLiquidEvaluation:
    power = -4.0 * jnp.pi * radius**2 * wall_velocity * stress
    return BubbleLiquidEvaluation(
        stress,
        jnp.zeros((0,), dtype=stress.dtype),
        power,
        jnp.isfinite(stress),
    )


class NewtonianBubbleLiquidLaw(AbstractBubbleLiquidLaw):
    """Newtonian liquid, `S = −4 μ Ṙ/R`."""

    viscosity: Array = parameter_field()
    law_id: str = eqx.field(static=True)

    def __init__(self, viscosity: ArrayLike, /) -> None:
        mu = scalar_parameter(viscosity, "viscosity", lower=0.0, inclusive=True)
        self.viscosity = mu
        self.law_id = canonical_fingerprint({"kind": "bubble-liquid-newtonian"})

    @property
    def stiff(self) -> bool:
        return False

    def initialize(self, reference_radius: Array, environment: BubbleEnvironment, /) -> Array:
        del reference_radius, environment
        return jnp.zeros((0,), dtype=jnp.float64)

    def evaluate(
        self, radius: Array, wall_velocity: Array, internal: Array, /
    ) -> BubbleLiquidEvaluation:
        del internal
        return _viscous_evaluation(
            -4.0 * self.viscosity * wall_velocity / radius, radius, wall_velocity
        )

    def internal_scale(self, scales: BubbleScales, /) -> Array:
        del scales
        return jnp.zeros((0,), dtype=jnp.float64)


class PowerLawBubbleLiquidLaw(AbstractBubbleLiquidLaw):
    """Truncated Ostwald–de Waele liquid `μ = K max(γ̇, γ̇_min)^{n−1}`.

    The shear-rate magnitude of the radial flow is `γ̇ = 2√3 R²|Ṙ|/r³`. The
    stress integral is exact: with `q = min(γ̇_min/γ̇_w, 1)` at the wall,
    `S = −4 (Ṙ/R) [K γ̇_w^{n−1} (1 − qⁿ)/n + μ₀ q]` and `μ₀ = K γ̇_min^{n−1}`.
    The truncation removes the singular zero-shear viscosity of `n < 1`.
    """

    consistency: Array = parameter_field()
    flow_index: Array = parameter_field()
    minimum_shear_rate: Array = parameter_field()
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        consistency: ArrayLike,
        flow_index: ArrayLike,
        minimum_shear_rate: ArrayLike,
        /,
    ) -> None:
        k = scalar_parameter(consistency, "consistency", lower=0.0)
        n = scalar_parameter(flow_index, "flow_index", lower=0.0)
        rate = scalar_parameter(minimum_shear_rate, "minimum_shear_rate", lower=0.0)
        self.consistency = k
        self.flow_index = n
        self.minimum_shear_rate = rate
        self.law_id = canonical_fingerprint({"kind": "bubble-liquid-power-law"})

    @property
    def stiff(self) -> bool:
        return False

    def initialize(self, reference_radius: Array, environment: BubbleEnvironment, /) -> Array:
        del reference_radius, environment
        return jnp.zeros((0,), dtype=jnp.float64)

    def evaluate(
        self, radius: Array, wall_velocity: Array, internal: Array, /
    ) -> BubbleLiquidEvaluation:
        del internal
        n = self.flow_index
        wall_rate = 2.0 * jnp.sqrt(3.0) * jnp.abs(wall_velocity) / radius
        bounded_rate = jnp.maximum(wall_rate, self.minimum_shear_rate)
        ratio = self.minimum_shear_rate / bounded_rate
        plateau = self.consistency * self.minimum_shear_rate ** (n - 1.0)
        effective = (
            self.consistency * bounded_rate ** (n - 1.0) * (1.0 - ratio**n) / n
            + plateau * ratio
        )
        return _viscous_evaluation(
            -4.0 * effective * wall_velocity / radius, radius, wall_velocity
        )

    def internal_scale(self, scales: BubbleScales, /) -> Array:
        del scales
        return jnp.zeros((0,), dtype=jnp.float64)


def _elastic_stress(shear_modulus: Array, radius: Array, reference_radius: Array, /) -> Array:
    # Yang & Church (2005) small-strain linear elasticity of the surrounding medium.
    return -4.0 * shear_modulus / 3.0 * (1.0 - (reference_radius / radius) ** 3)


class KelvinVoigtBubbleLiquidLaw(AbstractBubbleLiquidLaw):
    """Kelvin–Voigt viscoelastic medium (Yang & Church 2005).

    `S = −4 μ Ṙ/R − (4G/3)(1 − R₀³/R³)` with the stress-free radius `R₀` equal
    to the equilibrium radius.
    """

    viscosity: Array = parameter_field()
    shear_modulus: Array = parameter_field()
    law_id: str = eqx.field(static=True)

    def __init__(self, viscosity: ArrayLike, shear_modulus: ArrayLike, /) -> None:
        mu = scalar_parameter(viscosity, "viscosity", lower=0.0, inclusive=True)
        g = scalar_parameter(shear_modulus, "shear_modulus", lower=0.0, inclusive=True)
        self.viscosity = mu
        self.shear_modulus = g
        self.law_id = canonical_fingerprint({"kind": "bubble-liquid-kelvin-voigt"})

    @property
    def stiff(self) -> bool:
        return False

    def initialize(self, reference_radius: Array, environment: BubbleEnvironment, /) -> Array:
        del environment
        return reference_radius[None]

    def evaluate(
        self, radius: Array, wall_velocity: Array, internal: Array, /
    ) -> BubbleLiquidEvaluation:
        viscous = -4.0 * self.viscosity * wall_velocity / radius
        stress = viscous + _elastic_stress(self.shear_modulus, radius, internal[0])
        return BubbleLiquidEvaluation(
            stress,
            jnp.zeros_like(internal),
            16.0 * jnp.pi * self.viscosity * radius * wall_velocity**2,
            jnp.isfinite(stress),
        )

    def internal_scale(self, scales: BubbleScales, /) -> Array:
        return scales.radius[None]


class ZenerBubbleLiquidLaw(AbstractBubbleLiquidLaw):
    """Small-strain Zener (standard linear solid) medium.

    Constitutive law `τ + λ τ̇ = 2 G ε + 2 μ ε̇`, realized as an equilibrium
    spring `G` in parallel with a Maxwell arm of viscosity `η = μ − G λ ≥ 0`
    and relaxation time `λ`. The Maxwell stress integral `s` is ODE state:
    `λ ṡ + s = −4 η Ṙ/R`, and `S = −(4G/3)(1 − R₀³/R³) + s`. `λ → 0` recovers
    Kelvin–Voigt and `G → 0` the linear Maxwell liquid.
    """

    viscosity: Array = parameter_field()
    shear_modulus: Array = parameter_field()
    relaxation_time: Array = parameter_field()
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        viscosity: ArrayLike,
        shear_modulus: ArrayLike,
        relaxation_time: ArrayLike,
        /,
    ) -> None:
        mu = scalar_parameter(viscosity, "viscosity", lower=0.0, inclusive=True)
        g = scalar_parameter(shear_modulus, "shear_modulus", lower=0.0, inclusive=True)
        tau = scalar_parameter(relaxation_time, "relaxation_time", lower=0.0)
        if float(mu) < float(g) * float(tau):
            raise ValueError(
                "Zener admissibility requires viscosity >= shear_modulus * relaxation_time."
            )
        self.viscosity = mu
        self.shear_modulus = g
        self.relaxation_time = tau
        self.law_id = canonical_fingerprint({"kind": "bubble-liquid-zener"})

    @property
    def stiff(self) -> bool:
        return False

    def initialize(self, reference_radius: Array, environment: BubbleEnvironment, /) -> Array:
        del environment
        return jnp.stack((reference_radius, jnp.zeros_like(reference_radius)))

    def evaluate(
        self, radius: Array, wall_velocity: Array, internal: Array, /
    ) -> BubbleLiquidEvaluation:
        reference_radius, maxwell = internal
        arm_viscosity = self.viscosity - self.shear_modulus * self.relaxation_time
        maxwell_rate = (
            -4.0 * arm_viscosity * wall_velocity / radius - maxwell
        ) / self.relaxation_time
        stress = _elastic_stress(self.shear_modulus, radius, reference_radius) + maxwell
        positive_arm = arm_viscosity > 0.0
        safe_arm = jnp.where(positive_arm, arm_viscosity, 1.0)
        dissipation = jnp.where(
            positive_arm, jnp.pi * maxwell**2 * radius**3 / safe_arm, 0.0
        )
        return BubbleLiquidEvaluation(
            stress,
            jnp.stack((jnp.zeros_like(maxwell), maxwell_rate)),
            dissipation,
            jnp.isfinite(stress) & jnp.isfinite(maxwell_rate),
        )

    def internal_scale(self, scales: BubbleScales, /) -> Array:
        return jnp.stack((scales.radius, scales.pressure))


class OldroydBBubbleLiquidLaw(AbstractBubbleLiquidLaw):
    """Oldroyd-B liquid: Newtonian solvent plus upper-convected Maxwell polymer.

    The polymer stresses are advected exactly along material shells labelled by
    the Lagrangian volume coordinate `x = r³ − R³`; on each shell
    `τ̇_rr = −(τ_rr + 4 μ_p g)/λ − 4 g τ_rr` and
    `τ̇_θθ = −(τ_θθ − 2 μ_p g)/λ + 2 g τ_θθ` with `g = R²Ṙ/r³`. The shells sit at
    `quadrature_nodes` Gauss–Legendre points of `x = R₀³(1 + ξ)/(1 − ξ)`, which
    integrates the linear-response stress integral exactly. Dissipation uses the
    conformation tensor `A = I + τ λ/μ_p`, which must stay positive definite.
    """

    solvent_viscosity: Array = parameter_field()
    polymer_viscosity: Array = parameter_field()
    relaxation_time: Array = parameter_field()
    shell_coordinates: Array = fixed_field()
    shell_weights: Array = fixed_field()
    quadrature_nodes: int = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        solvent_viscosity: ArrayLike,
        polymer_viscosity: ArrayLike,
        relaxation_time: ArrayLike,
        /,
        *,
        quadrature_nodes: int = 16,
    ) -> None:
        solvent = scalar_parameter(
            solvent_viscosity, "solvent_viscosity", lower=0.0, inclusive=True
        )
        polymer = scalar_parameter(polymer_viscosity, "polymer_viscosity", lower=0.0)
        tau = scalar_parameter(relaxation_time, "relaxation_time", lower=0.0)
        count = positive_integer(quadrature_nodes, "quadrature_nodes")
        nodes, weights = np.polynomial.legendre.leggauss(count)
        self.solvent_viscosity = solvent
        self.polymer_viscosity = polymer
        self.relaxation_time = tau
        # Unit-reference shell coordinate x/R₀³ and weight ∫ dx/R₀³ per node.
        self.shell_coordinates = jnp.asarray((1.0 + nodes) / (1.0 - nodes), dtype=jnp.float64)
        self.shell_weights = jnp.asarray(2.0 * weights / (1.0 - nodes) ** 2, dtype=jnp.float64)
        self.quadrature_nodes = count
        self.law_id = canonical_fingerprint(
            {"kind": "bubble-liquid-oldroyd-b", "quadrature_nodes": count}
        )

    @property
    def stiff(self) -> bool:
        return False

    def initialize(self, reference_radius: Array, environment: BubbleEnvironment, /) -> Array:
        del environment
        return jnp.concatenate(
            (reference_radius[None], jnp.zeros((2 * self.quadrature_nodes,), dtype=jnp.float64))
        )

    def evaluate(
        self, radius: Array, wall_velocity: Array, internal: Array, /
    ) -> BubbleLiquidEvaluation:
        count = self.quadrature_nodes
        reference_cube = internal[0] ** 3
        radial = internal[1 : 1 + count]
        hoop = internal[1 + count :]
        coordinate = reference_cube * self.shell_coordinates
        weight = reference_cube * self.shell_weights
        cube = coordinate + radius**3
        strain_rate = radius**2 * wall_velocity / cube
        polymer = self.polymer_viscosity
        tau = self.relaxation_time
        radial_rate = -(radial + 4.0 * polymer * strain_rate) / tau - 4.0 * strain_rate * radial
        hoop_rate = -(hoop - 2.0 * polymer * strain_rate) / tau + 2.0 * strain_rate * hoop
        stress = (
            -4.0 * self.solvent_viscosity * wall_velocity / radius
            + 2.0 / 3.0 * jnp.sum(weight * (radial - hoop) / cube)
        )
        modulus = polymer / tau
        conformation_radial = 1.0 + radial / modulus
        conformation_hoop = 1.0 + hoop / modulus
        density = modulus / (2.0 * tau) * (
            conformation_radial
            + 1.0 / conformation_radial
            - 2.0
            + 2.0 * (conformation_hoop + 1.0 / conformation_hoop - 2.0)
        )
        dissipation = 16.0 * jnp.pi * self.solvent_viscosity * radius * wall_velocity**2 + (
            4.0 * jnp.pi / 3.0 * jnp.sum(weight * density)
        )
        rates = jnp.concatenate((jnp.zeros((1,), dtype=radial.dtype), radial_rate, hoop_rate))
        admissible = (
            jnp.isfinite(stress)
            & jnp.all(conformation_radial > 0.0)
            & jnp.all(conformation_hoop > 0.0)
        )
        return BubbleLiquidEvaluation(stress, rates, dissipation, admissible)

    def internal_scale(self, scales: BubbleScales, /) -> Array:
        return jnp.concatenate(
            (
                scales.radius[None],
                jnp.full((2 * self.quadrature_nodes,), scales.pressure),
            )
        )


__all__ = [
    "KelvinVoigtBubbleLiquidLaw",
    "NewtonianBubbleLiquidLaw",
    "OldroydBBubbleLiquidLaw",
    "PowerLawBubbleLiquidLaw",
    "ZenerBubbleLiquidLaw",
]
