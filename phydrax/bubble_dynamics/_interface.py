#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Clean interfaces and encapsulating shells.

A law returns the capillary, shell-elastic and shell-viscous pressures that
reduce the liquid pressure at the wall. Piecewise laws (Marmottant) expose a
closed regime set and signed guards so that the single-bubble solver localizes
every regime change with a native event before switching the smooth branch.
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import parameter_field
from ._contracts import (
    AbstractBubbleInterfaceLaw,
    BubbleEnvironment,
    BubbleInterfaceEvaluation,
    BubbleScales,
    scalar_parameter,
)


class TolmanCorrectionPolicy(StrictModule):
    """Curvature-dependent surface tension `σ(R) = σ∞/(1 + 2δ/R)`.

    The Tolman length `δ` of real liquids is disputed; bubble laws therefore use
    no Tolman correction unless this policy is supplied explicitly.
    """

    tolman_length: Array = parameter_field()

    def __init__(self, tolman_length: ArrayLike, /) -> None:
        self.tolman_length = scalar_parameter(tolman_length, "tolman_length")

    def surface_tension(self, planar_surface_tension: Array, radius: Array, /) -> Array:
        """Surface tension of a sphere of radius `radius`."""
        return planar_surface_tension / (1.0 + 2.0 * self.tolman_length / radius)


def _interface_evaluation(
    radius: Array,
    wall_velocity: Array,
    surface_tension: Array,
    elastic_pressure: Array,
    viscous_pressure: Array,
    internal_rate: Array,
    dissipation: Array,
    /,
) -> BubbleInterfaceEvaluation:
    capillary = 2.0 * surface_tension / radius
    admissible = (
        jnp.isfinite(capillary)
        & jnp.isfinite(elastic_pressure)
        & jnp.isfinite(viscous_pressure)
        & (radius > 0.0)
    )
    return BubbleInterfaceEvaluation(
        surface_tension,
        capillary,
        elastic_pressure,
        viscous_pressure,
        radius,
        wall_velocity,
        internal_rate,
        dissipation,
        admissible,
    )


class AbstractSmoothBubbleInterfaceLaw(AbstractBubbleInterfaceLaw):
    """Interface law with one smooth regime and no guards."""

    @property
    def regime_names(self) -> tuple[str, ...]:
        return ("smooth",)

    @property
    def guard_count(self) -> int:
        return 0

    @abc.abstractmethod
    def evaluate_smooth(
        self, radius: Array, wall_velocity: Array, internal: Array, /
    ) -> BubbleInterfaceEvaluation:
        """Interface stresses of the single smooth regime."""
        raise NotImplementedError

    def evaluate(
        self,
        radius: Array,
        wall_velocity: Array,
        internal: Array,
        regime: Array,
        /,
    ) -> BubbleInterfaceEvaluation:
        del regime
        return self.evaluate_smooth(radius, wall_velocity, internal)

    def initial_regime(self, radius: Array, internal: Array, /) -> Array:
        del radius, internal
        return jnp.asarray(0, dtype=jnp.int32)

    def regime_guards(self, radius: Array, internal: Array, regime: Array, /) -> Array:
        del internal, regime
        return jnp.zeros((0,), dtype=radius.dtype)

    def regime_after_crossing(
        self,
        radius: Array,
        wall_velocity: Array,
        internal: Array,
        regime: Array,
        guard: Array,
        /,
    ) -> Array:
        del radius, wall_velocity, internal, guard
        return regime


def _reference_radius_state(reference_radius: Array, /) -> Array:
    return reference_radius[None]


class CleanBubbleInterfaceLaw(AbstractSmoothBubbleInterfaceLaw):
    """Clean gas–liquid interface with constant (or Tolman-corrected) tension."""

    surface_tension: Array = parameter_field()
    tolman: TolmanCorrectionPolicy | None
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        surface_tension: ArrayLike,
        /,
        *,
        tolman: TolmanCorrectionPolicy | None = None,
    ) -> None:
        sigma = scalar_parameter(surface_tension, "surface_tension", lower=0.0, inclusive=True)
        if tolman is not None and not isinstance(tolman, TolmanCorrectionPolicy):
            raise TypeError("tolman must be a TolmanCorrectionPolicy or None.")
        self.surface_tension = sigma
        self.tolman = tolman
        self.law_id = canonical_fingerprint(
            {"kind": "bubble-interface-clean", "tolman": tolman is not None}
        )

    @property
    def stiff(self) -> bool:
        return False

    def initialize(self, reference_radius: Array, environment: BubbleEnvironment, /) -> Array:
        del reference_radius, environment
        return jnp.zeros((0,), dtype=jnp.float64)

    def tension_at(self, radius: Array, /) -> Array:
        """Surface tension at `radius`, including the optional Tolman correction."""
        if self.tolman is None:
            return jnp.broadcast_to(self.surface_tension, jnp.shape(radius))
        return self.tolman.surface_tension(self.surface_tension, radius)

    def evaluate_smooth(
        self, radius: Array, wall_velocity: Array, internal: Array, /
    ) -> BubbleInterfaceEvaluation:
        zero = jnp.zeros_like(radius)
        return _interface_evaluation(
            radius,
            wall_velocity,
            self.tension_at(radius),
            zero,
            zero,
            jnp.zeros_like(internal),
            zero,
        )

    def internal_scale(self, scales: BubbleScales, /) -> Array:
        del scales
        return jnp.zeros((0,), dtype=jnp.float64)


def _buckling_radius(
    reference_radius: Array, initial_surface_tension: Array, elasticity: Array, /
) -> Array:
    return reference_radius / jnp.sqrt(1.0 + initial_surface_tension / elasticity)


class MarmottantShell(AbstractBubbleInterfaceLaw):
    """Marmottant et al. (2005) lipid shell with buckling and rupture.

    `σ = 0` for `R ≤ R_b` (buckled), `σ = χ(R²/R_b² − 1)` in the elastic branch
    and `σ = σ_l` above the upper radius (ruptured); `R_b = R₀/√(1 + σ₀/χ)` so
    that `σ(R₀) = σ₀`. The shell viscous pressure is `4 κ_s Ṙ/R²`.

    Without `rupture_surface_tension` the three branches are reversible and the
    upper radius is `R_r = R_b √(1 + σ_l/χ)`. With it, the intact elastic branch
    extends to `R_break = R_b √(1 + σ_break/χ)`, where the shell breaks
    irreversibly; the broken shell then follows the three-branch law with `R_r`.
    Regime index is `branch + 3·broken` with branches buckled/elastic/ruptured.
    """

    shell_elasticity: Array = parameter_field()
    initial_surface_tension: Array = parameter_field()
    liquid_surface_tension: Array = parameter_field()
    shell_viscosity: Array = parameter_field()
    rupture_surface_tension: Array | None = parameter_field()
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        shell_elasticity: ArrayLike,
        initial_surface_tension: ArrayLike,
        liquid_surface_tension: ArrayLike,
        shell_viscosity: ArrayLike,
        /,
        *,
        rupture_surface_tension: ArrayLike | None = None,
    ) -> None:
        chi = scalar_parameter(shell_elasticity, "shell_elasticity", lower=0.0)
        liquid = scalar_parameter(liquid_surface_tension, "liquid_surface_tension", lower=0.0)
        initial = scalar_parameter(
            initial_surface_tension, "initial_surface_tension", lower=0.0, inclusive=True
        )
        viscosity = scalar_parameter(shell_viscosity, "shell_viscosity", lower=0.0, inclusive=True)
        rupture = (
            None
            if rupture_surface_tension is None
            else scalar_parameter(rupture_surface_tension, "rupture_surface_tension", lower=0.0)
        )
        upper = float(liquid) if rupture is None else float(rupture)
        if rupture is not None and float(rupture) < float(liquid):
            raise ValueError("rupture_surface_tension must be >= liquid_surface_tension.")
        if float(initial) >= upper:
            raise ValueError(
                "initial_surface_tension must lie below the rupture surface tension."
            )
        self.shell_elasticity = chi
        self.initial_surface_tension = initial
        self.liquid_surface_tension = liquid
        self.shell_viscosity = viscosity
        self.rupture_surface_tension = rupture
        self.law_id = canonical_fingerprint(
            {"kind": "bubble-interface-marmottant", "irreversible_rupture": rupture is not None}
        )

    @property
    def irreversible_rupture(self) -> bool:
        """Whether break-up above `R_break` is irreversible."""
        return self.rupture_surface_tension is not None

    @property
    def regime_names(self) -> tuple[str, ...]:
        intact = ("buckled", "elastic", "ruptured")
        if not self.irreversible_rupture:
            return intact
        return intact + ("broken-buckled", "broken-elastic", "broken-ruptured")

    @property
    def guard_count(self) -> int:
        return 2

    @property
    def stiff(self) -> bool:
        return False

    def initialize(self, reference_radius: Array, environment: BubbleEnvironment, /) -> Array:
        del environment
        return _reference_radius_state(reference_radius)

    def characteristic_radii(self, internal: Array, /) -> tuple[Array, Array, Array]:
        """Buckling, rupture (`σ = σ_l`) and break-up radii."""
        buckling = _buckling_radius(
            internal[0], self.initial_surface_tension, self.shell_elasticity
        )
        rupture = buckling * jnp.sqrt(1.0 + self.liquid_surface_tension / self.shell_elasticity)
        breakup = (
            rupture
            if self.rupture_surface_tension is None
            else buckling
            * jnp.sqrt(1.0 + self.rupture_surface_tension / self.shell_elasticity)
        )
        return buckling, rupture, breakup

    def _upper_radius(self, internal: Array, broken: Array, /) -> Array:
        _, rupture, breakup = self.characteristic_radii(internal)
        return jnp.where(broken, rupture, breakup)

    def evaluate(
        self,
        radius: Array,
        wall_velocity: Array,
        internal: Array,
        regime: Array,
        /,
    ) -> BubbleInterfaceEvaluation:
        branch = jnp.remainder(regime, 3)
        buckling, _, _ = self.characteristic_radii(internal)
        elastic = self.shell_elasticity * ((radius / buckling) ** 2 - 1.0)
        sigma = jnp.where(
            branch == 0,
            jnp.zeros_like(radius),
            jnp.where(branch == 1, elastic, self.liquid_surface_tension),
        )
        viscous = 4.0 * self.shell_viscosity * wall_velocity / radius**2
        return _interface_evaluation(
            radius,
            wall_velocity,
            sigma,
            jnp.zeros_like(radius),
            viscous,
            jnp.zeros_like(internal),
            16.0 * jnp.pi * self.shell_viscosity * wall_velocity**2,
        )

    def initial_regime(self, radius: Array, internal: Array, /) -> Array:
        buckling = self.characteristic_radii(internal)[0]
        upper = self._upper_radius(internal, jnp.asarray(False))
        branch = jnp.where(radius < buckling, 0, jnp.where(radius < upper, 1, 2))
        return branch.astype(jnp.int32)

    def regime_guards(self, radius: Array, internal: Array, regime: Array, /) -> Array:
        branch = jnp.remainder(regime, 3)
        broken = regime >= 3
        buckling = self.characteristic_radii(internal)[0]
        upper = self._upper_radius(internal, broken)
        lower_guard = jnp.where(branch == 0, buckling - radius, radius - buckling)
        upper_guard = jnp.where(branch == 2, radius - upper, upper - radius)
        one = jnp.ones_like(radius)
        return jnp.stack(
            (
                jnp.where(branch == 2, one, lower_guard / buckling),
                jnp.where(branch == 0, one, upper_guard / buckling),
            )
        )

    def regime_after_crossing(
        self,
        radius: Array,
        wall_velocity: Array,
        internal: Array,
        regime: Array,
        guard: Array,
        /,
    ) -> Array:
        del radius, wall_velocity, internal
        branch = jnp.remainder(regime, 3)
        broken = (regime >= 3).astype(jnp.int32)
        new_branch = jnp.where(
            guard == 0,
            jnp.where(branch == 0, 1, 0),
            jnp.where(branch == 1, 2, 1),
        )
        breaking = (guard == 1) & (branch == 1) & self.irreversible_rupture
        new_broken = jnp.maximum(broken, breaking.astype(jnp.int32))
        return (new_branch + 3 * new_broken).astype(jnp.int32)

    def internal_scale(self, scales: BubbleScales, /) -> Array:
        return scales.radius[None]


class GompertzMarmottantShell(AbstractSmoothBubbleInterfaceLaw):
    """Smooth Marmottant–Gompertz shell for gradient-based inference.

    `σ = σ_l exp(−b exp(c (1 − R/R_b)))` with the Marmottant buckling radius,
    `c = (2χe/σ_l) √(1 + σ_l/(2χ))` (maximum slope equal to the Marmottant slope
    at `R_b √(1 + σ_l/(2χ))`) and `b` fixed by `σ(R₀) = σ₀`
    (arXiv:2106.12004, Eqs. 13–15). Requires `σ₀ > 0`.
    """

    shell_elasticity: Array = parameter_field()
    initial_surface_tension: Array = parameter_field()
    liquid_surface_tension: Array = parameter_field()
    shell_viscosity: Array = parameter_field()
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        shell_elasticity: ArrayLike,
        initial_surface_tension: ArrayLike,
        liquid_surface_tension: ArrayLike,
        shell_viscosity: ArrayLike,
        /,
    ) -> None:
        chi = scalar_parameter(shell_elasticity, "shell_elasticity", lower=0.0)
        initial = scalar_parameter(initial_surface_tension, "initial_surface_tension", lower=0.0)
        liquid = scalar_parameter(liquid_surface_tension, "liquid_surface_tension", lower=0.0)
        viscosity = scalar_parameter(shell_viscosity, "shell_viscosity", lower=0.0, inclusive=True)
        if float(initial) >= float(liquid):
            raise ValueError("initial_surface_tension must be below liquid_surface_tension.")
        self.shell_elasticity = chi
        self.initial_surface_tension = initial
        self.liquid_surface_tension = liquid
        self.shell_viscosity = viscosity
        self.law_id = canonical_fingerprint({"kind": "bubble-interface-gompertz-marmottant"})

    @property
    def stiff(self) -> bool:
        return False

    def initialize(self, reference_radius: Array, environment: BubbleEnvironment, /) -> Array:
        del environment
        return _reference_radius_state(reference_radius)

    def tension_at(self, radius: Array, internal: Array, /) -> Array:
        """Gompertz surface tension at `radius`."""
        chi = self.shell_elasticity
        liquid = self.liquid_surface_tension
        reference = internal[0]
        buckling = _buckling_radius(reference, self.initial_surface_tension, chi)
        steepness = 2.0 * chi * jnp.e / liquid * jnp.sqrt(1.0 + liquid / (2.0 * chi))
        offset = -jnp.log(self.initial_surface_tension / liquid) / jnp.exp(
            steepness * (1.0 - reference / buckling)
        )
        return liquid * jnp.exp(-offset * jnp.exp(steepness * (1.0 - radius / buckling)))

    def evaluate_smooth(
        self, radius: Array, wall_velocity: Array, internal: Array, /
    ) -> BubbleInterfaceEvaluation:
        return _interface_evaluation(
            radius,
            wall_velocity,
            self.tension_at(radius, internal),
            jnp.zeros_like(radius),
            4.0 * self.shell_viscosity * wall_velocity / radius**2,
            jnp.zeros_like(internal),
            16.0 * jnp.pi * self.shell_viscosity * wall_velocity**2,
        )

    def internal_scale(self, scales: BubbleScales, /) -> Array:
        return scales.radius[None]


class HoffShell(AbstractSmoothBubbleInterfaceLaw):
    """Thin incompressible Kelvin–Voigt shell (Hoff, Sontum & Hovem 2000).

    Elastic `12 G_s d_s R₀²/R³ (1 − R₀/R)`, viscous `12 μ_s d_s R₀² Ṙ/R⁴` and
    capillary `2σ/R`, with shell thickness `d_s` at the equilibrium radius `R₀`.
    """

    shell_shear_modulus: Array = parameter_field()
    shell_viscosity: Array = parameter_field()
    shell_thickness: Array = parameter_field()
    surface_tension: Array = parameter_field()
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        shell_shear_modulus: ArrayLike,
        shell_viscosity: ArrayLike,
        shell_thickness: ArrayLike,
        /,
        *,
        surface_tension: ArrayLike = 0.0,
    ) -> None:
        modulus = scalar_parameter(
            shell_shear_modulus, "shell_shear_modulus", lower=0.0, inclusive=True
        )
        viscosity = scalar_parameter(shell_viscosity, "shell_viscosity", lower=0.0, inclusive=True)
        thickness = scalar_parameter(shell_thickness, "shell_thickness", lower=0.0)
        sigma = scalar_parameter(surface_tension, "surface_tension", lower=0.0, inclusive=True)
        self.shell_shear_modulus = modulus
        self.shell_viscosity = viscosity
        self.shell_thickness = thickness
        self.surface_tension = sigma
        self.law_id = canonical_fingerprint({"kind": "bubble-interface-hoff"})

    @property
    def stiff(self) -> bool:
        return False

    def initialize(self, reference_radius: Array, environment: BubbleEnvironment, /) -> Array:
        del environment
        return _reference_radius_state(reference_radius)

    def evaluate_smooth(
        self, radius: Array, wall_velocity: Array, internal: Array, /
    ) -> BubbleInterfaceEvaluation:
        reference = internal[0]
        factor = 12.0 * self.shell_thickness * reference**2
        elastic = factor * self.shell_shear_modulus / radius**3 * (1.0 - reference / radius)
        viscous = factor * self.shell_viscosity * wall_velocity / radius**4
        return _interface_evaluation(
            radius,
            wall_velocity,
            jnp.broadcast_to(self.surface_tension, jnp.shape(radius)),
            elastic,
            viscous,
            jnp.zeros_like(internal),
            4.0 * jnp.pi * radius**2 * wall_velocity * viscous,
        )

    def internal_scale(self, scales: BubbleScales, /) -> Array:
        return scales.radius[None]


class ChurchShell(AbstractSmoothBubbleInterfaceLaw):
    """Incompressible Kelvin–Voigt shell of finite thickness (Church 1995).

    With shell volume `V_S = (R₀ + d)³ − R₀³` and outer radius
    `R₂ = (R³ + V_S)^{1/3}`, the shell contributes elastic
    `4 G_s V_S (1 − R₀/R)/R₂³`, viscous `4 μ_s V_S Ṙ/(R R₂³)` and capillary
    `2σ₁/R + 2σ₂/R₂`; the liquid law acts at `R₂`. The shell density equals the
    liquid density: Church's inertial density-contrast terms are not modeled.
    """

    shell_shear_modulus: Array = parameter_field()
    shell_viscosity: Array = parameter_field()
    shell_thickness: Array = parameter_field()
    inner_surface_tension: Array = parameter_field()
    outer_surface_tension: Array = parameter_field()
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        shell_shear_modulus: ArrayLike,
        shell_viscosity: ArrayLike,
        shell_thickness: ArrayLike,
        /,
        *,
        inner_surface_tension: ArrayLike = 0.0,
        outer_surface_tension: ArrayLike = 0.0,
    ) -> None:
        modulus = scalar_parameter(
            shell_shear_modulus, "shell_shear_modulus", lower=0.0, inclusive=True
        )
        viscosity = scalar_parameter(shell_viscosity, "shell_viscosity", lower=0.0, inclusive=True)
        thickness = scalar_parameter(shell_thickness, "shell_thickness", lower=0.0)
        inner = scalar_parameter(
            inner_surface_tension, "inner_surface_tension", lower=0.0, inclusive=True
        )
        outer = scalar_parameter(
            outer_surface_tension, "outer_surface_tension", lower=0.0, inclusive=True
        )
        self.shell_shear_modulus = modulus
        self.shell_viscosity = viscosity
        self.shell_thickness = thickness
        self.inner_surface_tension = inner
        self.outer_surface_tension = outer
        self.law_id = canonical_fingerprint({"kind": "bubble-interface-church"})

    @property
    def stiff(self) -> bool:
        return False

    def initialize(self, reference_radius: Array, environment: BubbleEnvironment, /) -> Array:
        del environment
        return _reference_radius_state(reference_radius)

    def evaluate_smooth(
        self, radius: Array, wall_velocity: Array, internal: Array, /
    ) -> BubbleInterfaceEvaluation:
        reference = internal[0]
        shell_volume = (reference + self.shell_thickness) ** 3 - reference**3
        outer_cube = radius**3 + shell_volume
        outer = jnp.cbrt(outer_cube)
        outer_velocity = radius**2 * wall_velocity / outer**2
        elastic = 4.0 * self.shell_shear_modulus * shell_volume * (1.0 - reference / radius) / outer_cube
        viscous = 4.0 * self.shell_viscosity * shell_volume * wall_velocity / (radius * outer_cube)
        capillary = 2.0 * self.inner_surface_tension / radius + 2.0 * self.outer_surface_tension / outer
        admissible = (
            jnp.isfinite(capillary) & jnp.isfinite(elastic) & jnp.isfinite(viscous) & (radius > 0.0)
        )
        return BubbleInterfaceEvaluation(
            capillary * radius / 2.0,
            capillary,
            elastic,
            viscous,
            outer,
            outer_velocity,
            jnp.zeros_like(internal),
            4.0 * jnp.pi * radius**2 * wall_velocity * viscous,
            admissible,
        )

    def internal_scale(self, scales: BubbleScales, /) -> Array:
        return scales.radius[None]


class SarkarShell(AbstractSmoothBubbleInterfaceLaw):
    """Viscoelastic interface of Sarkar et al. (2005), optionally strain-softening.

    `σ(R) = γ₀ + E_s β exp(−α β)` with area strain `β = R²/R₀² − 1` and interface
    dilatational viscosity `κ_s` (`4 κ_s Ṙ/R²`). `α = 0` is the linear Sarkar
    model; `α > 0` is the exponential elasticity model of Paul et al. (2010).
    """

    reference_surface_tension: Array = parameter_field()
    dilatational_elasticity: Array = parameter_field()
    dilatational_viscosity: Array = parameter_field()
    elasticity_decay: Array = parameter_field()
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_surface_tension: ArrayLike,
        dilatational_elasticity: ArrayLike,
        dilatational_viscosity: ArrayLike,
        /,
        *,
        elasticity_decay: ArrayLike = 0.0,
    ) -> None:
        sigma = scalar_parameter(
            reference_surface_tension, "reference_surface_tension", lower=0.0, inclusive=True
        )
        elasticity = scalar_parameter(
            dilatational_elasticity, "dilatational_elasticity", lower=0.0, inclusive=True
        )
        viscosity = scalar_parameter(
            dilatational_viscosity, "dilatational_viscosity", lower=0.0, inclusive=True
        )
        decay = scalar_parameter(elasticity_decay, "elasticity_decay", lower=0.0, inclusive=True)
        self.reference_surface_tension = sigma
        self.dilatational_elasticity = elasticity
        self.dilatational_viscosity = viscosity
        self.elasticity_decay = decay
        self.law_id = canonical_fingerprint({"kind": "bubble-interface-sarkar"})

    @property
    def stiff(self) -> bool:
        return False

    def initialize(self, reference_radius: Array, environment: BubbleEnvironment, /) -> Array:
        del environment
        return _reference_radius_state(reference_radius)

    def evaluate_smooth(
        self, radius: Array, wall_velocity: Array, internal: Array, /
    ) -> BubbleInterfaceEvaluation:
        strain = (radius / internal[0]) ** 2 - 1.0
        sigma = self.reference_surface_tension + self.dilatational_elasticity * strain * jnp.exp(
            -self.elasticity_decay * strain
        )
        return _interface_evaluation(
            radius,
            wall_velocity,
            sigma,
            jnp.zeros_like(radius),
            4.0 * self.dilatational_viscosity * wall_velocity / radius**2,
            jnp.zeros_like(internal),
            16.0 * jnp.pi * self.dilatational_viscosity * wall_velocity**2,
        )

    def internal_scale(self, scales: BubbleScales, /) -> Array:
        return scales.radius[None]


class DoinikovShearThinningShell(AbstractSmoothBubbleInterfaceLaw):
    """Zero-thickness viscoelastic shell with Cross-law shear thinning.

    Doinikov, Haac & Dayton (2009): elastic `4χ(1/R₀ − 1/R)`, capillary `2σ/R`
    and viscous `4 κ_s(Ṙ/R) Ṙ/R²` with `κ_s = κ₀/(1 + α |Ṙ|/R)`. `α = 0`
    recovers the Kelvin–Voigt zero-thickness shell.
    """

    shell_elasticity: Array = parameter_field()
    surface_tension: Array = parameter_field()
    shell_viscosity: Array = parameter_field()
    characteristic_time: Array = parameter_field()
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        shell_elasticity: ArrayLike,
        surface_tension: ArrayLike,
        shell_viscosity: ArrayLike,
        characteristic_time: ArrayLike,
        /,
    ) -> None:
        chi = scalar_parameter(shell_elasticity, "shell_elasticity", lower=0.0, inclusive=True)
        sigma = scalar_parameter(surface_tension, "surface_tension", lower=0.0, inclusive=True)
        viscosity = scalar_parameter(shell_viscosity, "shell_viscosity", lower=0.0, inclusive=True)
        time = scalar_parameter(
            characteristic_time, "characteristic_time", lower=0.0, inclusive=True
        )
        self.shell_elasticity = chi
        self.surface_tension = sigma
        self.shell_viscosity = viscosity
        self.characteristic_time = time
        self.law_id = canonical_fingerprint({"kind": "bubble-interface-doinikov-shear-thinning"})

    @property
    def stiff(self) -> bool:
        return False

    def initialize(self, reference_radius: Array, environment: BubbleEnvironment, /) -> Array:
        del environment
        return _reference_radius_state(reference_radius)

    def evaluate_smooth(
        self, radius: Array, wall_velocity: Array, internal: Array, /
    ) -> BubbleInterfaceEvaluation:
        viscosity = self.shell_viscosity / (
            1.0 + self.characteristic_time * jnp.abs(wall_velocity) / radius
        )
        elastic = 4.0 * self.shell_elasticity * (1.0 / internal[0] - 1.0 / radius)
        return _interface_evaluation(
            radius,
            wall_velocity,
            jnp.broadcast_to(self.surface_tension, jnp.shape(radius)),
            elastic,
            4.0 * viscosity * wall_velocity / radius**2,
            jnp.zeros_like(internal),
            16.0 * jnp.pi * viscosity * wall_velocity**2,
        )

    def internal_scale(self, scales: BubbleScales, /) -> Array:
        return scales.radius[None]


class MaxwellShell(AbstractSmoothBubbleInterfaceLaw):
    """Thin linear-Maxwell shell (thin-shell reduction of Doinikov & Dayton 2007).

    The shell stress pressure `s` relaxes as
    `λ ṡ + s = 12 η_s d_s R₀² Ṙ/R⁴`: `λ → 0` is Hoff's viscous shell and fast
    motion (`ωλ ≫ 1`) is Hookean with shear modulus `η_s/λ`. Capillary `2σ/R`.
    """

    shell_viscosity: Array = parameter_field()
    shell_thickness: Array = parameter_field()
    relaxation_time: Array = parameter_field()
    surface_tension: Array = parameter_field()
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        shell_viscosity: ArrayLike,
        shell_thickness: ArrayLike,
        relaxation_time: ArrayLike,
        /,
        *,
        surface_tension: ArrayLike = 0.0,
    ) -> None:
        viscosity = scalar_parameter(shell_viscosity, "shell_viscosity", lower=0.0)
        thickness = scalar_parameter(shell_thickness, "shell_thickness", lower=0.0)
        tau = scalar_parameter(relaxation_time, "relaxation_time", lower=0.0)
        sigma = scalar_parameter(surface_tension, "surface_tension", lower=0.0, inclusive=True)
        self.shell_viscosity = viscosity
        self.shell_thickness = thickness
        self.relaxation_time = tau
        self.surface_tension = sigma
        self.law_id = canonical_fingerprint({"kind": "bubble-interface-maxwell"})

    @property
    def stiff(self) -> bool:
        return False

    def initialize(self, reference_radius: Array, environment: BubbleEnvironment, /) -> Array:
        del environment
        return jnp.stack((reference_radius, jnp.zeros_like(reference_radius)))

    def evaluate_smooth(
        self, radius: Array, wall_velocity: Array, internal: Array, /
    ) -> BubbleInterfaceEvaluation:
        reference, stress = internal
        target = (
            12.0 * self.shell_viscosity * self.shell_thickness * reference**2 * wall_velocity
            / radius**4
        )
        stress_rate = (target - stress) / self.relaxation_time
        dissipation = (
            jnp.pi * stress**2 * radius**6
            / (3.0 * self.shell_thickness * reference**2 * self.shell_viscosity)
        )
        return _interface_evaluation(
            radius,
            wall_velocity,
            jnp.broadcast_to(self.surface_tension, jnp.shape(radius)),
            jnp.zeros_like(radius),
            stress,
            jnp.stack((jnp.zeros_like(stress), stress_rate)),
            dissipation,
        )

    def internal_scale(self, scales: BubbleScales, /) -> Array:
        return jnp.stack((scales.radius, scales.pressure))


__all__ = [
    "AbstractSmoothBubbleInterfaceLaw",
    "ChurchShell",
    "CleanBubbleInterfaceLaw",
    "DoinikovShearThinningShell",
    "GompertzMarmottantShell",
    "HoffShell",
    "MarmottantShell",
    "MaxwellShell",
    "SarkarShell",
    "TolmanCorrectionPolicy",
]
