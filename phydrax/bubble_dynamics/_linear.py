#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Small-amplitude response from the Jacobian of the composed radial equation.

The nondimensional state `x` of a bubble at rest obeys `ẋ = f(x, p_d)`. The
bounded dense Jacobian `J = ∂f/∂x` (the state has tens of entries) and the drive
column `b = ∂f/∂p_d` give the harmonic response `(iω I − J) x̂ = b`, solved with
the native dense LU solve. The linearized wall pressure of each law then splits
the equivalent-oscillator stiffness and damping:
`β_k = −Im(p̂_k/R̂)/(2ωρR₀)`, stiffness `−Re(p̂_k/R̂)`, and radiation damping
`Im(I)/(2ωρR₀)` from the inertial impedance `I = (p̂_L − 1)/R̂`.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree
from jax.typing import ArrayLike

from .._strict import StrictModule
from ..linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve
from ..nonlinear import LocalRootPlan
from ._contracts import AbstractBubblePressureDrive, PressureDriveEvaluation
from ._radial import BubbleState, RadialBubbleModel


class LinearBubbleResponseEvidence(StrictModule):
    """Jacobian, linear-solve and resonance-root evidence."""

    jacobian_finite: Array
    solve_successful: Array
    relative_residual: Array
    resonance_converged: Array
    resonance_residual: Array
    equilibrium_admissible: Array
    state_dimension: int = eqx.field(static=True)


class LinearBubbleResponse(StrictModule):
    """Harmonic response at each requested angular frequency.

    `radius_response` is `R̂` per unit excess far-field pressure (m Pa⁻¹).
    Damping rates are in s⁻¹ for the equivalent oscillator
    `R̈ + 2β Ṙ + ω₀² (R − R₀) ∝ −p_d`; `natural_frequency` is
    `ω₀(ω) = √(stiffness/(ρR₀))` (NaN when the stiffness is negative) and
    `resonance_frequency` solves `ω = ω₀(ω)`.
    """

    angular_frequency: Array
    radius_response: Array
    gas_stiffness: Array
    interface_stiffness: Array
    liquid_stiffness: Array
    natural_frequency: Array
    effective_polytropic_index: Array
    thermal_damping: Array
    viscous_damping: Array
    shell_damping: Array
    radiation_damping: Array
    total_damping: Array
    resonance_frequency: Array
    equilibrium_radius: Array
    equilibrium_gas_pressure: Array
    evidence: LinearBubbleResponseEvidence

    @property
    def successful(self) -> Array:
        """Whether the Jacobian, every harmonic solve and the resonance root succeeded."""
        evidence = self.evidence
        return (
            evidence.equilibrium_admissible
            & evidence.jacobian_finite
            & jnp.all(evidence.solve_successful)
            & evidence.resonance_converged
        )


class _AffineDrive(AbstractBubblePressureDrive):
    """Excess pressure `level + slope·t`; its derivatives give the drive columns."""

    level: Array
    slope: Array
    drive_id: str = eqx.field(static=True)

    def __init__(self, level: Array, slope: Array, /) -> None:
        self.level = level
        self.slope = slope
        self.drive_id = "bubble-drive-affine-linearization"

    def evaluate(self, time: Array, /) -> PressureDriveEvaluation:
        return PressureDriveEvaluation(
            self.level + self.slope * time,
            jnp.broadcast_to(self.slope, jnp.shape(time)),
            jnp.ones(jnp.shape(time), dtype=jnp.bool_),
        )

    def characteristic_pressure(self) -> Array:
        return jnp.abs(self.level)


class _Linearization(StrictModule):
    model: RadialBubbleModel
    regime: Array
    template: BubbleState
    scale: BubbleState
    jacobian: Array
    pressure_column: Array
    rate_column: Array
    point: Array

    def physical(self, flat: Array, /) -> BubbleState:
        _, unravel = ravel_pytree(self.template)
        return jax.tree.map(jnp.multiply, unravel(flat), self.scale)

    def pressure_terms(self, flat: Array, /) -> Array:
        """Gas, (negated) interface and liquid wall-pressure contributions."""
        state = self.physical(flat)
        wall = self.model.wall_pressure(
            state, self.regime, jnp.zeros_like(state.radius), None
        )
        interface = (
            wall.interface.capillary_pressure
            + wall.interface.elastic_pressure
            + wall.interface.viscous_pressure
        )
        return jnp.stack((wall.gas.pressure, -interface, wall.liquid.stress))

    def prescribed_motion(self, angular_frequency: Array, /) -> tuple[Array, Array, Array, Array]:
        """Response to a prescribed unit wall displacement `R̂ = R_scale e^{iωt}`.

        The unknowns are the drive amplitude `p̂_d` that sustains the motion and
        the internal-state response `ẑ`; they solve the velocity and internal
        rows `[u_U  J_Uz; u_z  J_zz − iωI] [p̂_d; ẑ] = r`, where a drive enters
        through the far-field pressure and its rate, `u = b₀ + iω b₁`. The
        system stays regular at an undamped resonance, where `p̂_d → 0`.
        Returns the scaled state tangent, `p̂_d`, the success flag and the
        relative residual.
        """
        jacobian = self.jacobian.astype(jnp.complex128)
        size = self.point.shape[0]
        velocity = 1j * angular_frequency * self.scale.radius / self.scale.wall_velocity
        known = jnp.zeros((size,), dtype=jnp.complex128).at[0].set(1.0).at[1].set(velocity)
        drive = self.pressure_column + 1j * angular_frequency * self.rate_column
        shifted = jacobian - 1j * angular_frequency * jnp.eye(size, dtype=jnp.complex128)
        matrix = jnp.concatenate((drive[1:, None], shifted[1:, 2:]), axis=1)
        right = 1j * angular_frequency * known[1:] - jacobian[1:, :2] @ known[:2]
        result = solve(
            LinearSystem(DenseLinearOperator(matrix)),
            right,
            policy=LinearSolvePolicy(DenseLU()),
        )
        residual = matrix @ result.value - right
        relative = jnp.linalg.norm(residual) / jnp.maximum(jnp.linalg.norm(right), 1.0e-300)
        tangent = known.at[2:].set(result.value[1:])
        return tangent, result.value[0], result.successful, relative

    def contributions(self, tangent: Array, /) -> Array:
        """Complex linearized pressure contributions along the scaled tangent."""

        def directional(direction: Array) -> Array:
            return jax.jvp(self.pressure_terms, (self.point,), (direction,))[1]

        return directional(jnp.real(tangent)) + 1j * directional(jnp.imag(tangent))


def _linearize(model: RadialBubbleModel, equilibrium_radius: ArrayLike, /) -> tuple[_Linearization, Array, Array]:
    equilibrium = model.equilibrium(equilibrium_radius)
    scales = model.characteristic_scales(equilibrium, 0.0, 0.0)
    scale = model.state_scale(equilibrium.state.gas, scales)
    scaled = jax.tree.map(jnp.divide, equilibrium.state, scale)
    point, unravel = ravel_pytree(scaled)
    regime = equilibrium.regime

    def field(flat: Array, level: Array, slope: Array) -> Array:
        state = jax.tree.map(jnp.multiply, unravel(flat), scale)
        rates = model.rates(state, regime, jnp.zeros_like(level), _AffineDrive(level, slope))
        return ravel_pytree(jax.tree.map(jnp.divide, rates.derivative, scale))[0]

    zero = jnp.zeros((), dtype=jnp.float64)
    jacobian = jax.jacfwd(field, argnums=0)(point, zero, zero)
    pressure_column = jax.jacfwd(field, argnums=1)(point, zero, zero)
    rate_column = jax.jacfwd(field, argnums=2)(point, zero, zero)
    linearization = _Linearization(
        model, regime, scaled, scale, jacobian, pressure_column, rate_column, point
    )
    return linearization, equilibrium.gas_pressure, equilibrium.admissible


def linear_bubble_response(
    model: RadialBubbleModel,
    equilibrium_radius: ArrayLike,
    angular_frequencies: ArrayLike,
    /,
) -> LinearBubbleResponse:
    """Linear harmonic response of `model` about its equilibrium at `equilibrium_radius`.

    The interface regime is the one occupied at rest (for example the elastic
    Marmottant branch). The response is exact for the linearized composed
    equation: no law supplies a separate linear formula.
    """
    if not isinstance(model, RadialBubbleModel):
        raise TypeError("model must be a RadialBubbleModel.")
    frequencies = jnp.atleast_1d(jnp.asarray(angular_frequencies, dtype=jnp.float64))
    if frequencies.ndim != 1:
        raise ValueError("angular_frequencies must be a scalar or a rank-1 array.")
    linearization, gas_pressure, admissible = _linearize(model, equilibrium_radius)
    radius = linearization.template.radius * linearization.scale.radius
    radius_scale = linearization.scale.radius
    density = model.far_field_density()
    mass = density * radius

    def dynamic_stiffness(frequency: Array) -> tuple[Array, Array, Array, Array, Array]:
        tangent, drive, successful, residual = linearization.prescribed_motion(frequency)
        contributions = linearization.contributions(tangent) / radius_scale
        impedance = jnp.sum(contributions) - drive / radius_scale
        return radius_scale / drive, contributions, impedance, successful, residual

    displacement, contributions, impedance, successful, residual = jax.vmap(
        dynamic_stiffness
    )(frequencies)
    stiffness = -jnp.real(contributions)
    damping = -jnp.imag(contributions) / (2.0 * frequencies[:, None] * mass)
    radiation = jnp.imag(impedance) / (2.0 * frequencies * mass)
    total_stiffness = jnp.sum(stiffness, axis=1)
    squared = total_stiffness / mass
    natural = jnp.where(squared >= 0.0, jnp.sqrt(jnp.abs(squared)), jnp.nan)
    polytropic = stiffness[:, 0] * radius / (3.0 * gas_pressure)

    def balance(log_frequency: Array) -> Array:
        frequency = jnp.exp(log_frequency)
        tangent, _, _, _ = linearization.prescribed_motion(frequency)
        local = linearization.contributions(tangent) / radius_scale
        return jnp.log(frequency**2 * mass / (-jnp.sum(jnp.real(local))))

    initial = jnp.log(jnp.sqrt(jnp.abs(total_stiffness[0]) / mass))
    root = LocalRootPlan(maximum_steps=40, tolerance=1.0e-12, plan_id="bubble-resonance")
    log_resonance, diagnostics = root.solve_with_diagnostics(balance, initial)
    evidence = LinearBubbleResponseEvidence(
        jnp.all(jnp.isfinite(linearization.jacobian))
        & jnp.all(jnp.isfinite(linearization.pressure_column))
        & jnp.all(jnp.isfinite(linearization.rate_column)),
        successful,
        residual,
        diagnostics.converged,
        diagnostics.residual,
        admissible,
        state_dimension=linearization.point.shape[0],
    )
    return LinearBubbleResponse(
        frequencies,
        displacement,
        stiffness[:, 0],
        stiffness[:, 1],
        stiffness[:, 2],
        natural,
        polytropic,
        damping[:, 0],
        damping[:, 2],
        damping[:, 1],
        radiation,
        jnp.sum(damping, axis=1) + radiation,
        jnp.exp(log_resonance),
        radius,
        gas_pressure,
        evidence,
    )


def minnaert_angular_frequency(
    equilibrium_radius: ArrayLike,
    ambient_pressure: ArrayLike,
    density: ArrayLike,
    polytropic_index: ArrayLike,
    /,
    *,
    surface_tension: ArrayLike = 0.0,
) -> Array:
    """Undamped natural frequency `√((3κp₀ + (3κ − 1) 2σ/R₀)/(ρR₀²))`."""
    radius = jnp.asarray(equilibrium_radius, dtype=jnp.float64)
    kappa = jnp.asarray(polytropic_index, dtype=jnp.float64)
    sigma = jnp.asarray(surface_tension, dtype=jnp.float64)
    stiffness = 3.0 * kappa * jnp.asarray(ambient_pressure) + (3.0 * kappa - 1.0) * 2.0 * sigma / radius
    return jnp.sqrt(stiffness / (jnp.asarray(density) * radius**2))


def prosperetti_polytropic_index(heat_capacity_ratio: ArrayLike, peclet: ArrayLike, /) -> Array:
    """Exact linear complex polytropic index of a homobaric ideal gas.

    Prosperetti (1977, 1991) with an isothermal wall:
    `κ = γ/(1 + 3(γ − 1)(z coth z − 1)/z²)`, `z² = i Pe`, `Pe = ω R₀²/χ`. The gas
    pressure responds as `p̂/p₀ = −3κ R̂/R₀`; `Re κ` is the effective polytropic
    index and `3 p₀ Im κ/(2ωρR₀²)` the thermal damping rate.
    """
    gamma = jnp.asarray(heat_capacity_ratio, dtype=jnp.float64)
    peclet_ = jnp.asarray(peclet, dtype=jnp.float64)
    root = jnp.sqrt(peclet_ / 2.0) * (1.0 + 1j)
    decay = jnp.exp(-2.0 * root)
    shifted = root * (1.0 + decay) / (1.0 - decay) - 1.0
    return gamma / (1.0 + 3.0 * (gamma - 1.0) * shifted / (1j * peclet_))



__all__ = [
    "LinearBubbleResponse",
    "LinearBubbleResponseEvidence",
    "linear_bubble_response",
    "minnaert_angular_frequency",
    "prosperetti_polytropic_index",
]
