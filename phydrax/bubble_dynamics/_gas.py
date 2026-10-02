#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Volume-based gas interiors: polytropic, thermal and compartment closures."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._trainable import fixed_field, parameter_field
from .._validation import positive_integer
from ..equations import AbstractThermodynamicMaterial
from ..nonlinear import LocalRootPlan
from ..typing import checked
from ._contracts import (
    AbstractBubbleCompartmentGasLaw,
    AbstractBubbleGasLaw,
    BubbleEnvironment,
    BubbleGasCapabilities,
    BubbleGasEvaluation,
    BubbleGasMergeResult,
    BubbleGasSplitResult,
    BubbleGasState,
    BubbleScales,
    MOLAR_GAS_CONSTANT,
    scalar_parameter,
)


def sphere_volume(radius: ArrayLike, /) -> Array:
    """Volume `4πR³/3` of a sphere."""
    return 4.0 * jnp.pi * jnp.asarray(radius) ** 3 / 3.0


def sphere_radius(volume: ArrayLike, /) -> Array:
    """Radius of the sphere with the given volume."""
    return jnp.cbrt(3.0 * jnp.asarray(volume) / (4.0 * jnp.pi))


def _polytropic_reference(
    reference_volume: Array,
    reference_pressure: Array,
    environment: BubbleEnvironment,
    /,
) -> BubbleGasState:
    temperature = environment.ambient_temperature
    amount = reference_pressure * reference_volume / (MOLAR_GAS_CONSTANT * temperature)
    return BubbleGasState(
        amount,
        None,
        jnp.stack((reference_volume, reference_pressure, temperature)),
    )


def _polytropic_scale(scales: BubbleScales, /) -> Array:
    return jnp.stack(
        (sphere_volume(scales.radius), scales.pressure, scales.temperature)
    )


_CLOSED_POLYTROPIC = BubbleGasCapabilities(
    caloric=False, heat_transfer=False, mass_transfer=False, mixing=False, stiff=False
)


class PolytropicBubbleGasLaw(AbstractBubbleGasLaw):
    """Ideal gas following `p V^κ = const` from its reference state.

    `κ = 1` is isothermal and `κ = γ` adiabatic. A zero reference pressure is the
    exact empty-cavity limit used by the Rayleigh collapse problem.
    """

    polytropic_index: Array = parameter_field()
    law_id: str = eqx.field(static=True)

    def __init__(self, polytropic_index: ArrayLike, /) -> None:
        index = scalar_parameter(polytropic_index, "polytropic_index", lower=0.0)
        self.polytropic_index = index
        self.law_id = canonical_fingerprint({"kind": "bubble-gas-polytropic"})

    @property
    def capabilities(self) -> BubbleGasCapabilities:
        return _CLOSED_POLYTROPIC

    def initialize(
        self,
        reference_volume: Array,
        reference_pressure: Array,
        environment: BubbleEnvironment,
        /,
    ) -> BubbleGasState:
        return _polytropic_reference(reference_volume, reference_pressure, environment)

    def evaluate(
        self,
        volume: Array,
        volume_rate: Array,
        state: BubbleGasState,
        environment: BubbleEnvironment,
        /,
    ) -> BubbleGasEvaluation:
        del volume_rate, environment
        reference_volume, reference_pressure, reference_temperature = state.internal
        ratio = reference_volume / volume
        pressure = reference_pressure * ratio**self.polytropic_index
        temperature = reference_temperature * ratio ** (self.polytropic_index - 1.0)
        admissible = (
            jnp.isfinite(pressure) & jnp.isfinite(temperature) & (volume > 0.0)
        ) & (pressure >= 0.0)
        return BubbleGasEvaluation(
            pressure,
            temperature,
            jnp.zeros_like(state.internal),
            jnp.zeros_like(state.amount),
            None,
            None,
            jnp.ones_like(volume),
            admissible,
        )

    def internal_scale(self, scales: BubbleScales, /) -> Array:
        return _polytropic_scale(scales)


class HardCorePolytropicBubbleGasLaw(AbstractBubbleGasLaw):
    """Polytropic van der Waals hard-core gas `p (V − V_c)^κ = const`.

    The excluded core volume is `V_c = (h/R_ref)³ V_ref` with
    `core_radius_ratio = h/R_ref`; `R_ref/8.86` is the classical air value.
    """

    polytropic_index: Array = parameter_field()
    core_radius_ratio: Array = parameter_field()
    law_id: str = eqx.field(static=True)

    def __init__(self, polytropic_index: ArrayLike, core_radius_ratio: ArrayLike, /) -> None:
        index = scalar_parameter(polytropic_index, "polytropic_index", lower=0.0)
        ratio = scalar_parameter(core_radius_ratio, "core_radius_ratio", lower=0.0, upper=1.0)
        self.polytropic_index = index
        self.core_radius_ratio = ratio
        self.law_id = canonical_fingerprint({"kind": "bubble-gas-hard-core-polytropic"})

    @property
    def capabilities(self) -> BubbleGasCapabilities:
        return _CLOSED_POLYTROPIC

    def initialize(
        self,
        reference_volume: Array,
        reference_pressure: Array,
        environment: BubbleEnvironment,
        /,
    ) -> BubbleGasState:
        return _polytropic_reference(reference_volume, reference_pressure, environment)

    def evaluate(
        self,
        volume: Array,
        volume_rate: Array,
        state: BubbleGasState,
        environment: BubbleEnvironment,
        /,
    ) -> BubbleGasEvaluation:
        del volume_rate, environment
        reference_volume, reference_pressure, reference_temperature = state.internal
        core = self.core_radius_ratio**3 * reference_volume
        free = volume - core
        ratio = (reference_volume - core) / free
        pressure = reference_pressure * ratio**self.polytropic_index
        temperature = reference_temperature * ratio ** (self.polytropic_index - 1.0)
        margin = free / volume
        admissible = (
            jnp.isfinite(pressure) & jnp.isfinite(temperature) & (margin > 0.0)
        ) & (pressure >= 0.0)
        return BubbleGasEvaluation(
            pressure,
            temperature,
            jnp.zeros_like(state.internal),
            jnp.zeros_like(state.amount),
            None,
            None,
            margin,
            admissible,
        )

    def internal_scale(self, scales: BubbleScales, /) -> Array:
        return _polytropic_scale(scales)


def _molar_heat_capacity(heat_capacity_ratio: Array, /) -> Array:
    return MOLAR_GAS_CONSTANT / (heat_capacity_ratio - 1.0)


def _thermal_diffusivity(
    heat_capacity_ratio: Array,
    thermal_conductivity: Array,
    volume: Array,
    amount: Array,
    /,
) -> Array:
    # χ = λ/(ρ c_p) with ρ c_p = γ R n/((γ−1) V); the molar mass cancels.
    return (
        thermal_conductivity
        * (heat_capacity_ratio - 1.0)
        * volume
        / (heat_capacity_ratio * MOLAR_GAS_CONSTANT * amount)
    )


def _caloric_reference(
    reference_volume: Array,
    reference_pressure: Array,
    environment: BubbleEnvironment,
    heat_capacity_ratio: Array,
    core_volume: Array,
    internal: Array,
    /,
) -> BubbleGasState:
    temperature = environment.ambient_temperature
    amount = (
        reference_pressure * (reference_volume - core_volume)
        / (MOLAR_GAS_CONSTANT * temperature)
    )
    energy = amount * _molar_heat_capacity(heat_capacity_ratio) * temperature
    return BubbleGasState(amount, energy, internal)


def _transfer_evaluation(
    volume: Array,
    volume_rate: Array,
    state: BubbleGasState,
    environment: BubbleEnvironment,
    heat_capacity_ratio: Array,
    core_volume: Array,
    conductance: Array,
    /,
) -> BubbleGasEvaluation:
    """Uniform-temperature gas exchanging heat `G (T∞ − T)` with the liquid."""
    if state.internal_energy is None:
        raise ValueError("Caloric gas state requires internal energy.")
    capacity = state.amount * _molar_heat_capacity(heat_capacity_ratio)
    temperature = state.internal_energy / capacity
    free = volume - core_volume
    pressure = state.amount * MOLAR_GAS_CONSTANT * temperature / free
    heat = conductance * (environment.ambient_temperature - temperature)
    energy_rate = -pressure * volume_rate + heat
    margin = free / volume
    admissible = (
        jnp.isfinite(pressure)
        & jnp.isfinite(heat)
        & (temperature > 0.0)
        & (margin > 0.0)
    )
    return BubbleGasEvaluation(
        pressure,
        temperature,
        jnp.zeros_like(state.internal),
        jnp.zeros_like(state.amount),
        energy_rate,
        heat,
        margin,
        admissible,
    )


class BoundaryLayerThermalBubbleGasLaw(AbstractBubbleGasLaw):
    """Uniform-temperature gas with a thermal boundary-layer heat flux.

    Toegel et al. (2000) and Stricker et al. (2011): the heat flux through the
    wall is `λ (T∞ − T)/ℓ` with `ℓ = min(√(R χ/|Ṙ|), R/π)`, the gas internal
    energy obeys `U̇ = −p V̇ + Q̇` and the pressure is the hard-core ideal gas
    `p = n R T/(V − V_c)`.
    """

    heat_capacity_ratio: Array = parameter_field()
    thermal_conductivity: Array = parameter_field()
    core_radius_ratio: Array = parameter_field()
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        heat_capacity_ratio: ArrayLike,
        thermal_conductivity: ArrayLike,
        /,
        *,
        core_radius_ratio: ArrayLike = 0.0,
    ) -> None:
        gamma = scalar_parameter(heat_capacity_ratio, "heat_capacity_ratio", lower=1.0)
        conductivity = scalar_parameter(thermal_conductivity, "thermal_conductivity", lower=0.0)
        ratio = scalar_parameter(
            core_radius_ratio, "core_radius_ratio", lower=0.0, inclusive=True, upper=1.0
        )
        self.heat_capacity_ratio = gamma
        self.thermal_conductivity = conductivity
        self.core_radius_ratio = ratio
        self.law_id = canonical_fingerprint({"kind": "bubble-gas-boundary-layer-thermal"})

    @property
    def capabilities(self) -> BubbleGasCapabilities:
        return BubbleGasCapabilities(
            caloric=True, heat_transfer=True, mass_transfer=False, mixing=False, stiff=False
        )

    def initialize(
        self,
        reference_volume: Array,
        reference_pressure: Array,
        environment: BubbleEnvironment,
        /,
    ) -> BubbleGasState:
        core = self.core_radius_ratio**3 * reference_volume
        return _caloric_reference(
            reference_volume,
            reference_pressure,
            environment,
            self.heat_capacity_ratio,
            core,
            reference_volume[None],
        )

    def thermal_diffusivity(self, volume: Array, state: BubbleGasState, /) -> Array:
        """Gas thermal diffusivity at the instantaneous density."""
        return _thermal_diffusivity(
            self.heat_capacity_ratio, self.thermal_conductivity, volume, state.amount
        )

    def evaluate(
        self,
        volume: Array,
        volume_rate: Array,
        state: BubbleGasState,
        environment: BubbleEnvironment,
        /,
    ) -> BubbleGasEvaluation:
        radius = sphere_radius(volume)
        wall_velocity = volume_rate / (4.0 * jnp.pi * radius**2)
        diffusivity = self.thermal_diffusivity(volume, state)
        # R/ℓ = max(√(R|Ṙ|/χ), π) written as √max(·, π²): the diffusive length
        # never exceeds R/π and the derivative stays finite at rest.
        inverse_length_ratio = jnp.sqrt(
            jnp.maximum(radius * jnp.abs(wall_velocity) / diffusivity, jnp.pi**2)
        )
        conductance = 4.0 * jnp.pi * radius * self.thermal_conductivity * inverse_length_ratio
        core = self.core_radius_ratio**3 * state.internal[0]
        return _transfer_evaluation(
            volume,
            volume_rate,
            state,
            environment,
            self.heat_capacity_ratio,
            core,
            conductance,
        )

    def internal_scale(self, scales: BubbleScales, /) -> Array:
        return sphere_volume(scales.radius)[None]


# Coefficients a_n = 2^{2n} B_{2n}/(2n)! of z coth z = Σ a_n z^{2n}, n = 2…8.
_COTH_SERIES = tuple(
    2.0 ** (2 * order) * bernoulli / math.factorial(2 * order)
    for order, bernoulli in (
        (2, -1.0 / 30.0),
        (3, 1.0 / 42.0),
        (4, -1.0 / 30.0),
        (5, 5.0 / 66.0),
        (6, -691.0 / 2730.0),
        (7, 7.0 / 6.0),
        (8, -3617.0 / 510.0),
    )
)


def preston_transfer_coefficient(peclet: ArrayLike, /) -> Array:
    """Real part of Preston's thermal transfer function `Ψ(Pe)`.

    `Ψ = z²(z coth z − 1)/(z² − 3(z coth z − 1))` with `z² = i Pe`; `Ψ → 5` as
    `Pe → 0`. Below `Pe = 0.1` the closed form loses digits to cancellation, so
    `Ψ = −(1/3 + z² U)/(3U)` is used with the convergent series
    `U = Σ_{n≥2} a_n z^{2(n−2)}` of `(z coth z − 1 − z²/3)/z⁴`.
    """
    peclet_ = jnp.asarray(peclet)
    small = peclet_ < 0.1
    safe_peclet = jnp.where(small, 1.0, peclet_)
    square = 1j * safe_peclet
    root = jnp.sqrt(safe_peclet / 2.0) * (1.0 + 1j)
    decay = jnp.exp(-2.0 * root)
    coth = (1.0 + decay) / (1.0 - decay)
    shifted = root * coth - 1.0
    closed = square * shifted / (square - 3.0 * shifted)
    small_square = 1j * jnp.where(small, peclet_, 0.0)
    remainder = jnp.zeros_like(small_square)
    for coefficient in reversed(_COTH_SERIES):
        remainder = remainder * small_square + coefficient
    series = -(1.0 / 3.0 + small_square * remainder) / (3.0 * remainder)
    return jnp.real(jnp.where(small, series, closed))


class ReducedTransferBubbleGasLaw(AbstractBubbleGasLaw):
    """Preston–Colonius–Brennen (2007) constant heat-transfer reduced model.

    The wall heat flux is `λ β_T (T∞ − T̄)/R` with `β_T = Re Ψ(Pe)` evaluated at
    `Pe = ω_T R_ref²/χ_ref`; the imaginary part of the linear transfer function
    is neglected as in the reference model. Low Péclet numbers give `β_T → 5`.
    """

    heat_capacity_ratio: Array = parameter_field()
    thermal_conductivity: Array = parameter_field()
    transfer_angular_frequency: Array = parameter_field()
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        heat_capacity_ratio: ArrayLike,
        thermal_conductivity: ArrayLike,
        transfer_angular_frequency: ArrayLike,
        /,
    ) -> None:
        gamma = scalar_parameter(heat_capacity_ratio, "heat_capacity_ratio", lower=1.0)
        conductivity = scalar_parameter(thermal_conductivity, "thermal_conductivity", lower=0.0)
        frequency = scalar_parameter(
            transfer_angular_frequency, "transfer_angular_frequency", lower=0.0
        )
        self.heat_capacity_ratio = gamma
        self.thermal_conductivity = conductivity
        self.transfer_angular_frequency = frequency
        self.law_id = canonical_fingerprint({"kind": "bubble-gas-reduced-transfer"})

    @property
    def capabilities(self) -> BubbleGasCapabilities:
        return BubbleGasCapabilities(
            caloric=True, heat_transfer=True, mass_transfer=False, mixing=False, stiff=False
        )

    def initialize(
        self,
        reference_volume: Array,
        reference_pressure: Array,
        environment: BubbleEnvironment,
        /,
    ) -> BubbleGasState:
        return _caloric_reference(
            reference_volume,
            reference_pressure,
            environment,
            self.heat_capacity_ratio,
            jnp.zeros_like(reference_volume),
            reference_volume[None],
        )

    def transfer_coefficient(self, state: BubbleGasState, /) -> Array:
        """Constant `β_T` implied by the reference state of `state`."""
        reference_volume = state.internal[0]
        diffusivity = _thermal_diffusivity(
            self.heat_capacity_ratio,
            self.thermal_conductivity,
            reference_volume,
            state.amount,
        )
        radius = sphere_radius(reference_volume)
        return preston_transfer_coefficient(
            self.transfer_angular_frequency * radius**2 / diffusivity
        )

    def evaluate(
        self,
        volume: Array,
        volume_rate: Array,
        state: BubbleGasState,
        environment: BubbleEnvironment,
        /,
    ) -> BubbleGasEvaluation:
        radius = sphere_radius(volume)
        conductance = (
            4.0 * jnp.pi * radius * self.thermal_conductivity * self.transfer_coefficient(state)
        )
        return _transfer_evaluation(
            volume,
            volume_rate,
            state,
            environment,
            self.heat_capacity_ratio,
            jnp.zeros_like(volume),
            conductance,
        )

    def internal_scale(self, scales: BubbleScales, /) -> Array:
        return sphere_volume(scales.radius)[None]


def _chebyshev_lobatto_unit(count: int, /) -> tuple[np.ndarray, np.ndarray]:
    """Nodes `s_k = (1 − cos(πk/N))/2` on [0, 1] and their differentiation matrix."""
    index = np.arange(count + 1)
    reference = np.cos(np.pi * index / count)
    weights = np.where((index == 0) | (index == count), 2.0, 1.0) * (-1.0) ** index
    difference = reference[:, None] - reference[None, :]
    off = np.outer(weights, 1.0 / weights) / (difference + np.eye(count + 1))
    np.fill_diagonal(off, 0.0)
    matrix = off - np.diag(np.sum(off, axis=1))
    # s = (1 − x)/2 reverses orientation and halves the interval.
    return (1.0 - reference) / 2.0, -2.0 * matrix


class SpectralThermalBubbleGasLaw(AbstractBubbleGasLaw):
    """Homobaric ideal gas with a resolved radial temperature field.

    Prosperetti (1991): uniform pressure, gas velocity
    `u = [(γ−1) K ∂_r T − r ṗ/3]/(γ p)` and energy equation
    `ρ c_p DT/Dt − ṗ = ∇·(K ∇T)` with an isothermal liquid wall. The field is a
    polynomial in `s = (r/R)²` collocated at `node_count + 1` Chebyshev–Lobatto
    nodes (fixed capacity); regularity at the center is exact. The conductivity
    is `K(T) = K∞ + dK/dT (T − T∞)`. The internal state holds the temperatures
    at the interior and center nodes; the wall node equals the liquid
    temperature. The collocated conduction operator is stiff.
    """

    heat_capacity_ratio: Array = parameter_field()
    thermal_conductivity: Array = parameter_field()
    conductivity_temperature_slope: Array = parameter_field()
    nodes: Array = fixed_field()
    differentiation: Array = fixed_field()
    node_count: int = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        heat_capacity_ratio: ArrayLike,
        thermal_conductivity: ArrayLike,
        /,
        *,
        node_count: int = 12,
        conductivity_temperature_slope: ArrayLike = 0.0,
    ) -> None:
        gamma = scalar_parameter(heat_capacity_ratio, "heat_capacity_ratio", lower=1.0)
        conductivity = scalar_parameter(thermal_conductivity, "thermal_conductivity", lower=0.0)
        slope = scalar_parameter(conductivity_temperature_slope, "conductivity_temperature_slope")
        count = positive_integer(node_count, "node_count")
        if count < 2:
            raise ValueError("node_count must be at least 2.")
        nodes, differentiation = _chebyshev_lobatto_unit(count)
        self.heat_capacity_ratio = gamma
        self.thermal_conductivity = conductivity
        self.conductivity_temperature_slope = slope
        self.nodes = jnp.asarray(nodes, dtype=jnp.float64)
        self.differentiation = jnp.asarray(differentiation, dtype=jnp.float64)
        self.node_count = count
        self.law_id = canonical_fingerprint(
            {"kind": "bubble-gas-spectral-thermal", "node_count": count}
        )

    @property
    def capabilities(self) -> BubbleGasCapabilities:
        return BubbleGasCapabilities(
            caloric=True, heat_transfer=True, mass_transfer=False, mixing=False, stiff=True
        )

    def initialize(
        self,
        reference_volume: Array,
        reference_pressure: Array,
        environment: BubbleEnvironment,
        /,
    ) -> BubbleGasState:
        profile = jnp.full(
            (self.node_count,), environment.ambient_temperature, dtype=jnp.float64
        )
        return _caloric_reference(
            reference_volume,
            reference_pressure,
            environment,
            self.heat_capacity_ratio,
            jnp.zeros_like(reference_volume),
            profile,
        )

    def temperature_profile(
        self, state: BubbleGasState, environment: BubbleEnvironment, /
    ) -> tuple[Array, Array]:
        """Collocation nodes `s = (r/R)²` and temperatures including the wall node."""
        profile = jnp.concatenate(
            (state.internal, environment.ambient_temperature[None])
        )
        return self.nodes, profile

    def evaluate(
        self,
        volume: Array,
        volume_rate: Array,
        state: BubbleGasState,
        environment: BubbleEnvironment,
        /,
    ) -> BubbleGasEvaluation:
        if state.internal_energy is None:
            raise ValueError("Spectral thermal gas state requires internal energy.")
        gamma = self.heat_capacity_ratio
        radius = sphere_radius(volume)
        wall_velocity = volume_rate / (4.0 * jnp.pi * radius**2)
        _, temperature = self.temperature_profile(state, environment)
        conductivity = self.thermal_conductivity + self.conductivity_temperature_slope * (
            temperature - environment.ambient_temperature
        )
        slope = self.differentiation @ temperature
        flux = conductivity * slope
        # (1/y²) ∂_y(y² K ∂_y T) = 6F + 4 s ∂_s F with F = K ∂_s T and s = y².
        divergence = (6.0 * flux + 4.0 * self.nodes * (self.differentiation @ flux)) / radius**2
        wall_gradient = 2.0 * slope[-1] / radius
        heat = 4.0 * jnp.pi * radius**2 * conductivity[-1] * wall_gradient
        pressure = (gamma - 1.0) * state.internal_energy / volume
        energy_rate = -pressure * volume_rate + heat
        pressure_rate = -gamma * pressure * volume_rate / volume + (gamma - 1.0) * heat / volume
        # y u = s [(γ−1) K (2/R) ∂_s T − R ṗ/3]/(γ p)
        radial_flux = self.nodes * (
            (gamma - 1.0) * conductivity * 2.0 * slope / radius - radius * pressure_rate / 3.0
        ) / (gamma * pressure)
        advection = 2.0 * (self.nodes * wall_velocity - radial_flux) * slope / radius
        source = (gamma - 1.0) * temperature / (gamma * pressure) * (pressure_rate + divergence)
        temperature_rate = (advection + source)[:-1]
        mean_temperature = pressure * volume / (state.amount * MOLAR_GAS_CONSTANT)
        admissible = (
            jnp.all(jnp.isfinite(temperature_rate))
            & jnp.all(temperature > 0.0)
            & (pressure > 0.0)
            & jnp.isfinite(heat)
        )
        return BubbleGasEvaluation(
            pressure,
            mean_temperature,
            temperature_rate,
            jnp.zeros_like(state.amount),
            energy_rate,
            heat,
            jnp.ones_like(volume),
            admissible,
        )

    def internal_scale(self, scales: BubbleScales, /) -> Array:
        return jnp.full((self.node_count,), scales.temperature)


def _stack_total(values: Array, /) -> Array:
    return jnp.sum(values, axis=0)


def _uniform_split(
    law: AbstractBubbleCompartmentGasLaw,
    state: BubbleGasState,
    volume: Array,
    child_volumes: Array,
    environment: BubbleEnvironment,
    /,
) -> BubbleGasSplitResult:
    if state.internal_energy is None:
        raise ValueError("Compartment gas states require internal energy.")
    child_volumes_ = jnp.asarray(child_volumes, dtype=jnp.float64)
    if child_volumes_.ndim != 1 or child_volumes_.shape[0] < 2:
        raise ValueError("child_volumes must list at least two children.")
    fractions = child_volumes_ / jnp.sum(child_volumes_)
    amounts = fractions * state.amount
    energies = fractions * state.internal_energy
    internal = jnp.broadcast_to(
        state.internal, child_volumes_.shape + state.internal.shape
    )
    children = BubbleGasState(amounts, energies, internal)
    child_entropy = jax.vmap(
        lambda child_volume, amount, energy, child_internal: law.entropy(
            child_volume,
            BubbleGasState(amount, energy, child_internal),
            environment,
        )
    )(child_volumes_, amounts, energies, internal)
    production = jnp.sum(child_entropy) - law.entropy(volume, state, environment)
    admissible = (
        jnp.all(child_volumes_ > 0.0)
        & jnp.all(jnp.isfinite(amounts))
        & jnp.all(jnp.isfinite(energies))
    )
    return BubbleGasSplitResult(
        children,
        production,
        jnp.sum(amounts) - state.amount,
        jnp.sum(energies) - state.internal_energy,
        jnp.sum(child_volumes_) - volume,
        admissible,
        policy="uniform-intensive",
    )


def _additive_merge(
    law: AbstractBubbleCompartmentGasLaw,
    states: BubbleGasState,
    volumes: Array,
    merged_volume: Array,
    environment: BubbleEnvironment,
    /,
) -> BubbleGasMergeResult:
    if states.internal_energy is None:
        raise ValueError("Compartment gas states require internal energy.")
    volumes_ = jnp.asarray(volumes, dtype=jnp.float64)
    if volumes_.ndim != 1 or volumes_.shape[0] < 2:
        raise ValueError("merge requires at least two stacked compartments.")
    if states.amount.shape != volumes_.shape:
        raise ValueError("Stacked gas states must align with volumes.")
    amount = _stack_total(states.amount)
    energy = _stack_total(states.internal_energy)
    merged = BubbleGasState(amount, energy, states.internal[0])
    parts = jax.vmap(
        lambda volume, part_amount, part_energy, part_internal: law.entropy(
            volume,
            BubbleGasState(part_amount, part_energy, part_internal),
            environment,
        )
    )(volumes_, states.amount, states.internal_energy, states.internal)
    production = law.entropy(merged_volume, merged, environment) - jnp.sum(parts)
    evaluation = law.evaluate(merged_volume, jnp.zeros_like(merged_volume), merged, environment)
    return BubbleGasMergeResult(
        merged,
        evaluation.pressure,
        evaluation.temperature,
        production,
        amount - jnp.sum(states.amount),
        energy - jnp.sum(states.internal_energy),
        merged_volume - jnp.sum(volumes_),
        evaluation.admissible & (merged_volume > 0.0),
    )


class IsothermalIdealBubbleGasLaw(AbstractBubbleCompartmentGasLaw):
    """Ideal gas held at the ambient liquid temperature.

    `p = n R T∞/V`; the internal energy `n c_v T∞` is constant and the heat
    supplied by the liquid is `p V̇`. This is the only closure for which the
    `p V` invariant is additive under merge.
    """

    heat_capacity_ratio: Array = parameter_field()
    law_id: str = eqx.field(static=True)

    def __init__(self, heat_capacity_ratio: ArrayLike, /) -> None:
        gamma = scalar_parameter(heat_capacity_ratio, "heat_capacity_ratio", lower=1.0)
        self.heat_capacity_ratio = gamma
        self.law_id = canonical_fingerprint({"kind": "bubble-gas-isothermal-ideal"})

    @property
    def capabilities(self) -> BubbleGasCapabilities:
        return BubbleGasCapabilities(
            caloric=True, heat_transfer=True, mass_transfer=False, mixing=True, stiff=False
        )

    def initialize(
        self,
        reference_volume: Array,
        reference_pressure: Array,
        environment: BubbleEnvironment,
        /,
    ) -> BubbleGasState:
        return _caloric_reference(
            reference_volume,
            reference_pressure,
            environment,
            self.heat_capacity_ratio,
            jnp.zeros_like(reference_volume),
            jnp.zeros((0,), dtype=jnp.float64),
        )

    def evaluate(
        self,
        volume: Array,
        volume_rate: Array,
        state: BubbleGasState,
        environment: BubbleEnvironment,
        /,
    ) -> BubbleGasEvaluation:
        temperature = environment.ambient_temperature
        pressure = state.amount * MOLAR_GAS_CONSTANT * temperature / volume
        admissible = jnp.isfinite(pressure) & (volume > 0.0) & (state.amount >= 0.0)
        return BubbleGasEvaluation(
            pressure,
            temperature,
            jnp.zeros_like(state.internal),
            jnp.zeros_like(state.amount),
            jnp.zeros_like(pressure),
            pressure * volume_rate,
            jnp.ones_like(volume),
            admissible,
        )

    def internal_scale(self, scales: BubbleScales, /) -> Array:
        del scales
        return jnp.zeros((0,), dtype=jnp.float64)

    def entropy(
        self,
        volume: Array,
        state: BubbleGasState,
        environment: BubbleEnvironment,
        /,
    ) -> Array:
        del environment
        return state.amount * MOLAR_GAS_CONSTANT * jnp.log(volume / state.amount)

    def merge(
        self,
        states: BubbleGasState,
        volumes: Array,
        merged_volume: Array,
        environment: BubbleEnvironment,
        /,
    ) -> BubbleGasMergeResult:
        return _additive_merge(self, states, volumes, merged_volume, environment)

    def split(
        self,
        state: BubbleGasState,
        volume: Array,
        child_volumes: Array,
        environment: BubbleEnvironment,
        /,
    ) -> BubbleGasSplitResult:
        return _uniform_split(self, state, volume, child_volumes, environment)


class CaloricIdealBubbleGasLaw(AbstractBubbleCompartmentGasLaw):
    """Adiabatic calorically perfect ideal gas `p = (γ − 1) U/V`, `U̇ = −p V̇`."""

    heat_capacity_ratio: Array = parameter_field()
    law_id: str = eqx.field(static=True)

    def __init__(self, heat_capacity_ratio: ArrayLike, /) -> None:
        gamma = scalar_parameter(heat_capacity_ratio, "heat_capacity_ratio", lower=1.0)
        self.heat_capacity_ratio = gamma
        self.law_id = canonical_fingerprint({"kind": "bubble-gas-caloric-ideal"})

    @property
    def capabilities(self) -> BubbleGasCapabilities:
        return BubbleGasCapabilities(
            caloric=True, heat_transfer=False, mass_transfer=False, mixing=True, stiff=False
        )

    def initialize(
        self,
        reference_volume: Array,
        reference_pressure: Array,
        environment: BubbleEnvironment,
        /,
    ) -> BubbleGasState:
        return _caloric_reference(
            reference_volume,
            reference_pressure,
            environment,
            self.heat_capacity_ratio,
            jnp.zeros_like(reference_volume),
            jnp.zeros((0,), dtype=jnp.float64),
        )

    def evaluate(
        self,
        volume: Array,
        volume_rate: Array,
        state: BubbleGasState,
        environment: BubbleEnvironment,
        /,
    ) -> BubbleGasEvaluation:
        del environment
        if state.internal_energy is None:
            raise ValueError("Caloric gas state requires internal energy.")
        pressure = (self.heat_capacity_ratio - 1.0) * state.internal_energy / volume
        temperature = state.internal_energy / (
            state.amount * _molar_heat_capacity(self.heat_capacity_ratio)
        )
        admissible = jnp.isfinite(pressure) & (volume > 0.0) & (temperature > 0.0)
        return BubbleGasEvaluation(
            pressure,
            temperature,
            jnp.zeros_like(state.internal),
            jnp.zeros_like(state.amount),
            -pressure * volume_rate,
            jnp.zeros_like(pressure),
            jnp.ones_like(volume),
            admissible,
        )

    def internal_scale(self, scales: BubbleScales, /) -> Array:
        del scales
        return jnp.zeros((0,), dtype=jnp.float64)

    def entropy(
        self,
        volume: Array,
        state: BubbleGasState,
        environment: BubbleEnvironment,
        /,
    ) -> Array:
        del environment
        if state.internal_energy is None:
            raise ValueError("Caloric gas state requires internal energy.")
        capacity = _molar_heat_capacity(self.heat_capacity_ratio)
        return state.amount * (
            capacity * jnp.log(state.internal_energy / state.amount)
            + MOLAR_GAS_CONSTANT * jnp.log(volume / state.amount)
        )

    def merge(
        self,
        states: BubbleGasState,
        volumes: Array,
        merged_volume: Array,
        environment: BubbleEnvironment,
        /,
    ) -> BubbleGasMergeResult:
        return _additive_merge(self, states, volumes, merged_volume, environment)

    def split(
        self,
        state: BubbleGasState,
        volume: Array,
        child_volumes: Array,
        environment: BubbleEnvironment,
        /,
    ) -> BubbleGasSplitResult:
        return _uniform_split(self, state, volume, child_volumes, environment)


class MaterialBubbleGasLaw(AbstractBubbleGasLaw):
    """Adiabatic homogeneous gas described by a thermodynamic material.

    Composes any `equations.AbstractThermodynamicMaterial` (ideal, stiffened or
    Noble–Abel stiffened gas): `ρ = n M/V`, `e = U/(n M)`, `p = p(ρ, e)`,
    `U̇ = −p V̇`. The reference density at the ambient temperature is the root
    of `T(ρ, p_ref) = T∞`; a failed root raises at initialization.
    """

    material: AbstractThermodynamicMaterial
    molar_mass: Array = parameter_field()
    law_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self, material: AbstractThermodynamicMaterial, molar_mass: ArrayLike, /
    ) -> None:
        mass = scalar_parameter(molar_mass, "molar_mass", lower=0.0)
        self.material = material
        self.molar_mass = mass
        self.law_id = canonical_fingerprint(
            {"kind": "bubble-gas-material", "material": material.material_id}
        )

    @property
    def capabilities(self) -> BubbleGasCapabilities:
        return BubbleGasCapabilities(
            caloric=True, heat_transfer=False, mass_transfer=False, mixing=False, stiff=False
        )

    def initialize(
        self,
        reference_volume: Array,
        reference_pressure: Array,
        environment: BubbleEnvironment,
        /,
    ) -> BubbleGasState:
        temperature = environment.ambient_temperature
        guess = reference_pressure * self.molar_mass / (MOLAR_GAS_CONSTANT * temperature)
        root = LocalRootPlan(maximum_steps=60, tolerance=1.0e-12, plan_id="bubble-gas-density")

        def residual(log_ratio: Array) -> Array:
            density = guess * jnp.exp(log_ratio)
            return self.material.temperature(density, reference_pressure) / temperature - 1.0

        log_ratio, diagnostics = root.solve_with_diagnostics(
            residual, jnp.zeros_like(reference_pressure)
        )
        density = guess * jnp.exp(log_ratio)
        density = eqx.error_if(
            density,
            ~diagnostics.converged
            | ~self.material.admissible(density, reference_pressure),
            "Material gas density at the ambient temperature did not converge.",
        )
        mass = density * reference_volume
        energy = mass * self.material.specific_internal_energy(density, reference_pressure)
        return BubbleGasState(
            mass / self.molar_mass, energy, jnp.zeros((0,), dtype=jnp.float64)
        )

    def evaluate(
        self,
        volume: Array,
        volume_rate: Array,
        state: BubbleGasState,
        environment: BubbleEnvironment,
        /,
    ) -> BubbleGasEvaluation:
        del environment
        if state.internal_energy is None:
            raise ValueError("Material gas state requires internal energy.")
        mass = state.amount * self.molar_mass
        density = mass / volume
        pressure = self.material.pressure(density, state.internal_energy / mass)
        temperature = self.material.temperature(density, pressure)
        admissible = self.material.admissible(density, pressure) & (temperature > 0.0)
        return BubbleGasEvaluation(
            pressure,
            temperature,
            jnp.zeros_like(state.internal),
            jnp.zeros_like(state.amount),
            -pressure * volume_rate,
            jnp.zeros_like(pressure),
            jnp.ones_like(volume),
            admissible,
        )

    def internal_scale(self, scales: BubbleScales, /) -> Array:
        del scales
        return jnp.zeros((0,), dtype=jnp.float64)


__all__ = [
    "BoundaryLayerThermalBubbleGasLaw",
    "CaloricIdealBubbleGasLaw",
    "HardCorePolytropicBubbleGasLaw",
    "IsothermalIdealBubbleGasLaw",
    "MaterialBubbleGasLaw",
    "PolytropicBubbleGasLaw",
    "ReducedTransferBubbleGasLaw",
    "SpectralThermalBubbleGasLaw",
    "preston_transfer_coefficient",
    "sphere_radius",
    "sphere_volume",
]
