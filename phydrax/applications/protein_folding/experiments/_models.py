# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Named equilibrium and isothermal kinetic models, independently derived.

All free-energy differences are G_destination - G_source. A positive unfolding
free energy therefore favors the folded state. Runtime values use the declared
energy and concentration units, never an implicit molar/single-system conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

from ....units import (
    conversion_factor,
    derived_unit,
    JOULE,
    KELVIN,
    KILOJOULE_PER_MOLE,
    MOLE,
    MOLE_PER_CUBIC_METER,
    ONE,
    UnitDefinition,
)


BOLTZMANN_CONSTANT = 1.380649e-23
GAS_CONSTANT = BOLTZMANN_CONSTANT * 6.02214076e23
JOULE_PER_MOLE = derived_unit("J/mol", ((JOULE, 1), (MOLE, -1)))


@dataclass(frozen=True)
class ThermodynamicConvention:
    """Explicit energy basis, concentration scale, and reference temperature.

    ``basis='molar'`` requires energy/amount units. ``basis='single-system'``
    requires energy units. The two bases cannot be mixed by unit conversion.
    Concentrations, including standard_concentration, use concentration_unit.
    """

    energy_unit: UnitDefinition = KILOJOULE_PER_MOLE
    concentration_unit: UnitDefinition = MOLE_PER_CUBIC_METER
    basis: str = "molar"
    reference_temperature: float = 298.15
    standard_concentration: float = 1000.0

    def __post_init__(self):
        if self.basis not in ("molar", "single-system"):
            raise ValueError("basis must be molar or single-system.")
        reference = JOULE_PER_MOLE if self.basis == "molar" else JOULE
        conversion_factor(self.energy_unit, reference)
        conversion_factor(self.concentration_unit, MOLE_PER_CUBIC_METER)
        if not isfinite(self.reference_temperature) or self.reference_temperature <= 0:
            raise ValueError("reference_temperature must be positive Kelvin.")
        if not isfinite(self.standard_concentration) or self.standard_concentration <= 0:
            raise ValueError("standard_concentration must be finite and positive.")

    @property
    def thermal_constant(self) -> float:
        reference = JOULE_PER_MOLE if self.basis == "molar" else JOULE
        constant = GAS_CONSTANT if self.basis == "molar" else BOLTZMANN_CONSTANT
        return constant / float(conversion_factor(self.energy_unit, reference))

    @property
    def thermal_units(self) -> tuple[UnitDefinition, ...]:
        e, c = self.energy_unit, self.concentration_unit
        return (
            e,
            e,
            derived_unit("energy/K", ((e, 1), (KELVIN, -1))),
            derived_unit("energy/concentration", ((e, 1), (c, -1))),
            derived_unit("energy/(concentration K)", ((e, 1), (c, -1), (KELVIN, -1))),
        )


def celsius_to_kelvin(temperature: ArrayLike) -> Array:
    """Declared offset adapter, not a multiplicative UnitDefinition conversion."""
    return jnp.asarray(temperature, dtype=float) + 273.15


def thermal_unfolding_free_energy(
    parameters, temperature, denaturant, reference_temperature
):
    """Constant-heat-capacity law with linear denaturant and thermal m-value.

    parameters = (dG_ref, dH_ref, dCp, m_ref, dm_dT). The thermal reference
    is at zero denaturant. dG = dH - T dS, with dS_ref=(dH_ref-dG_ref)/T_ref.
    """
    g, h, cp, m, dm = jnp.moveaxis(jnp.asarray(parameters), -1, 0)
    t, d = jnp.asarray(temperature), jnp.asarray(denaturant)
    tr = reference_temperature
    return (
        h * (1 - t / tr)
        + g * t / tr
        + cp * (t - tr - t * jnp.log(t / tr))
        - (m + dm * (t - tr)) * d
    )


def two_state_log_populations(delta_g, thermal_energy):
    """Log fractions ordered (folded, unfolded), with dG=G_U-G_F."""
    reduced = jnp.asarray(delta_g) / thermal_energy
    return jnp.stack((-jax.nn.softplus(-reduced), -jax.nn.softplus(reduced)), axis=-1)


def dimer_log_populations(
    delta_g, thermal_energy, total_concentration, standard_concentration
):
    """N2 <-> 2U monomer-equivalent fractions, ordered (N2, U).

    Kd=[U]^2/[N2]=c_standard exp(-dG/RT), C=[U]+2[N2]. The stable
    quadratic root is f_U=2/(1+sqrt(1+8C/Kd)). C=0 returns its infinite-
    dilution limit; a normalized fluorescence experiment must have C>0.
    """
    concentration = jnp.asarray(total_concentration)
    safe_concentration = jnp.where(concentration > 0, concentration, 1.0)
    log_ratio = (
        jnp.log(8.0)
        + jnp.log(safe_concentration / standard_concentration)
        + delta_g / thermal_energy
    )
    log_u = jnp.log(2.0) - jnp.logaddexp(0.0, 0.5 * jnp.logaddexp(0.0, log_ratio))
    log_u = jnp.where(concentration == 0, 0.0, log_u)
    log_n = jnp.where(
        concentration == 0, -jnp.inf, log_ratio - jnp.log(4.0) + 2.0 * log_u
    )
    result = jnp.stack((log_n, log_u), axis=-1)
    return jnp.where((concentration >= 0)[..., None], result, jnp.nan)


_THERMAL_NAMES = ("dg_ref", "dh_ref", "dcp", "m_ref", "dm_dt")


def _thermal_slots(prefix, convention):
    return tuple(
        (f"{prefix}.{name}", unit)
        for name, unit in zip(_THERMAL_NAMES, convention.thermal_units, strict=True)
    )


@dataclass(frozen=True)
class TwoStateUnfolding:
    """Reversible monomer F <-> U equilibrium, not a thermal-ramp model."""

    convention: ThermodynamicConvention = ThermodynamicConvention()
    prefix: str = "unfolding"
    state_names = ("folded", "unfolded")

    def parameter_slots(self):
        return _thermal_slots(self.prefix, self.convention)

    def populations(self, parameters, temperature, denaturant, concentration):
        dg = thermal_unfolding_free_energy(
            parameters, temperature, denaturant, self.convention.reference_temperature
        )
        return jnp.exp(
            two_state_log_populations(dg, self.convention.thermal_constant * temperature)
        )


@dataclass(frozen=True)
class ThreeStateUnfolding:
    """Reversible F <-> I <-> U; parameters are consecutive state differences."""

    convention: ThermodynamicConvention = ThermodynamicConvention()
    prefix: str = "unfolding"
    state_names = ("folded", "intermediate", "unfolded")

    def parameter_slots(self):
        return _thermal_slots(f"{self.prefix}.fi", self.convention) + _thermal_slots(
            f"{self.prefix}.iu", self.convention
        )

    def populations(self, parameters, temperature, denaturant, concentration):
        fi = thermal_unfolding_free_energy(
            parameters[:5], temperature, denaturant, self.convention.reference_temperature
        )
        iu = thermal_unfolding_free_energy(
            parameters[5:], temperature, denaturant, self.convention.reference_temperature
        )
        energies = jnp.stack((jnp.zeros_like(fi), fi, fi + iu), axis=-1)
        return jax.nn.softmax(
            -energies / (self.convention.thermal_constant * temperature[..., None]),
            axis=-1,
        )


@dataclass(frozen=True)
class DimerTwoStateUnfolding(TwoStateUnfolding):
    """Reversible N2 <-> 2U with monomer-equivalent fluorescence baselines."""

    def populations(self, parameters, temperature, denaturant, concentration):
        dg = thermal_unfolding_free_energy(
            parameters, temperature, denaturant, self.convention.reference_temperature
        )
        return jnp.exp(
            dimer_log_populations(
                dg,
                self.convention.thermal_constant * temperature,
                concentration,
                self.convention.standard_concentration,
            )
        )


@dataclass(frozen=True)
class DimerThreeStateUnfolding(ThreeStateUnfolding):
    """Reversible N2 <-> 2I <-> 2U; IU energy is per monomer."""

    def populations(self, parameters, temperature, denaturant, concentration):
        ni = thermal_unfolding_free_energy(
            parameters[:5], temperature, denaturant, self.convention.reference_temperature
        )
        iu = thermal_unfolding_free_energy(
            parameters[5:], temperature, denaturant, self.convention.reference_temperature
        )
        kt = self.convention.thermal_constant * temperature
        log_q = -iu / kt
        log_mon_partition = jnp.logaddexp(0.0, log_q)
        # [I]+[U] behaves as one monomer species with Kd_eff=Kd*(1+q)^2.
        log_dimer = dimer_log_populations(
            ni - 2 * kt * log_mon_partition,
            kt,
            concentration,
            self.convention.standard_concentration,
        )
        return jnp.exp(
            jnp.stack(
                (
                    log_dimer[..., 0],
                    log_dimer[..., 1] - log_mon_partition,
                    log_dimer[..., 1] + log_q - log_mon_partition,
                ),
                axis=-1,
            )
        )


def repeat_transfer_statistics(
    folding_free_energy, interface_free_energy, thermal_energy
):
    """Open-chain binary-repeat partition and folded marginal probabilities.

    E(x)=sum_i g_i*x_i + sum_i J_i*x_i*x_(i+1), x_i=1 folded. Energies
    are formation energies (negative favors folding/contact). No periodic bond
    is implied. Log-space forward/backward messages cost O(number of repeats).
    """
    g = jnp.asarray(folding_free_energy)
    bonds = jnp.asarray(interface_free_energy)
    local = jnp.stack((jnp.zeros_like(g), -g / thermal_energy), axis=-1)
    pair = jnp.array([[0.0, 0.0], [0.0, 1.0]])
    transitions = -bonds[:, None, None] / thermal_energy * pair

    def forward(previous, inputs):
        node, edge = inputs
        current = node + jsp.special.logsumexp(previous[:, None] + edge, axis=0)
        return current, current

    _, rest = jax.lax.scan(forward, local[0], (local[1:], transitions))
    alpha = jnp.concatenate((local[:1], rest), axis=0)

    def backward(following, inputs):
        node, edge = inputs
        current = jsp.special.logsumexp(edge + node[None, :] + following[None, :], axis=1)
        return current, current

    _, before = jax.lax.scan(
        backward, jnp.zeros(2, dtype=g.dtype), (local[1:], transitions), reverse=True
    )
    beta = jnp.concatenate((before, jnp.zeros((1, 2), dtype=g.dtype)), axis=0)
    log_z = jsp.special.logsumexp(alpha[-1])
    folded = jnp.exp(alpha[:, 1] + beta[:, 1] - log_z)
    return log_z, folded


@dataclass(frozen=True)
class RepeatTransferUnfolding:
    """Heterogeneous open repeat chain; fluorescence tracks mean folded fraction."""

    repeat_count: int
    convention: ThermodynamicConvention = ThermodynamicConvention()
    prefix: str = "repeat"
    state_names = ("folded", "unfolded")

    def __post_init__(self):
        if (
            isinstance(self.repeat_count, bool)
            or not isinstance(self.repeat_count, int)
            or self.repeat_count < 1
        ):
            raise ValueError("repeat_count must be a positive integer.")

    def parameter_slots(self):
        nodes = tuple(
            slot
            for i in range(self.repeat_count)
            for slot in _thermal_slots(f"{self.prefix}.{i}", self.convention)
        )
        return nodes + tuple(
            (f"{self.prefix}.interface.{i}", self.convention.energy_unit)
            for i in range(self.repeat_count - 1)
        )

    def populations(self, parameters, temperature, denaturant, concentration):
        nodes = parameters[: 5 * self.repeat_count].reshape((self.repeat_count, 5))
        bonds = parameters[5 * self.repeat_count :]

        def at_condition(t, d):
            dg = thermal_unfolding_free_energy(
                nodes, t, d, self.convention.reference_temperature
            )
            _, folded = repeat_transfer_statistics(
                -dg, bonds, self.convention.thermal_constant * t
            )
            mean = jnp.mean(folded)
            return jnp.stack((mean, 1.0 - mean))

        return jax.vmap(at_condition)(temperature, denaturant)


@dataclass(frozen=True)
class ChevronKinetics:
    """Isothermal two-state relaxation k_obs=k_f+k_u; log rates use 1/s."""

    convention: ThermodynamicConvention = ThermodynamicConvention()
    prefix: str = "chevron"

    def parameter_slots(self):
        slope = self.convention.thermal_units[3]
        return (
            (f"{self.prefix}.log_kf", ONE),
            (f"{self.prefix}.log_ku", ONE),
            (f"{self.prefix}.mf", slope),
            (f"{self.prefix}.mu", slope),
        )

    def log_rates(self, parameters, temperature, denaturant):
        kf, ku, mf, mu = parameters
        kt = self.convention.thermal_constant * temperature
        return jnp.stack((kf - mf * denaturant / kt, ku + mu * denaturant / kt), axis=-1)

    def predict_log_rate(self, parameters, temperature, denaturant):
        return jsp.special.logsumexp(
            self.log_rates(parameters, temperature, denaturant), axis=-1
        )


@dataclass(frozen=True)
class ParallelPathKinetics:
    """Two parallel F<->U barriers sharing one equilibrium free energy.

    Each pathway obeys k_f/k_u=exp(dG/RT). Only sums of pathway rates are
    measured by relaxation; interchangeable/indistinguishable paths remain
    non-identifiable. No arbitrary mechanism or intermediate lifetime is fitted.
    """

    convention: ThermodynamicConvention = ThermodynamicConvention()
    prefix: str = "parallel"

    def parameter_slots(self):
        e, slope = self.convention.energy_unit, self.convention.thermal_units[3]
        return (
            (f"{self.prefix}.dg_ref", e),
            (f"{self.prefix}.m_ref", slope),
            (f"{self.prefix}.log_kf1", ONE),
            (f"{self.prefix}.mf1", slope),
            (f"{self.prefix}.log_kf2", ONE),
            (f"{self.prefix}.mf2", slope),
        )

    def log_rates(self, parameters, temperature, denaturant):
        dg, m, k1, m1, k2, m2 = parameters
        kt = self.convention.thermal_constant * temperature
        log_f = jnp.stack((k1 - m1 * denaturant / kt, k2 - m2 * denaturant / kt), axis=-1)
        log_u = log_f - ((dg - m * denaturant) / kt)[..., None]
        return jnp.stack(
            (
                jsp.special.logsumexp(log_f, axis=-1),
                jsp.special.logsumexp(log_u, axis=-1),
            ),
            axis=-1,
        )

    def predict_log_rate(self, parameters, temperature, denaturant):
        return jsp.special.logsumexp(
            self.log_rates(parameters, temperature, denaturant), axis=-1
        )


EquilibriumModel = (
    TwoStateUnfolding
    | ThreeStateUnfolding
    | DimerTwoStateUnfolding
    | DimerThreeStateUnfolding
    | RepeatTransferUnfolding
)
KineticModel = ChevronKinetics | ParallelPathKinetics
