#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""SI thermodynamics of two three-dimensional, parabolic semiconductor bands.

The normalized Fermi integrals are those of NIST DLMF 25.12.14,
https://dlmf.nist.gov/25.12.E14. The DOS, compressibility, inverse statistics,
pressure and kinetic energy all use the *same* occupation quadrature, rather
than unrelated fitted approximations. The generalized Einstein relation is
D/mu = density / (q * compressibility); see Kantner and Koprucki,
https://arxiv.org/abs/2002.10133. The selected gap-temperature law is Varshni,
https://doi.org/10.1016/0031-8914(67)90062-6.

These constitutive laws and numerical domains are not empirical qualification.
Every material requires explicit DOS, alignment and parameter provenance.
Constructors and ``admit_*`` methods are host admission boundaries. All other
methods accept broadcastable physical SI arrays and support JAX transforms;
invalid numeric states raise through ``eqx.error_if``, never clipping a
population, extrapolating a material, or returning an unsuccessful root.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._strict import StrictModule
from ...ein import contract
from ...integration import GaussLegendreRule
from ...nonlinear import NonlinearTermination, scalar_root, ScalarRootProblem
from ...units import derived_unit, JOULE, KELVIN, ONE, UnitDefinition
from ._quantities import (
    _positive_scalar,
    _si,
    _text,
    BOLTZMANN_CONSTANT_SI as _KB,
    ELEMENTARY_CHARGE_SI as _Q,
    PER_CUBIC_METER,
)


_ENERGY_PER_TEMPERATURE = derived_unit("J/K", ((JOULE, 1), (KELVIN, -1)))
_MAXIMUM_FD_ETA = 80.0


def _finite_scalar(value, unit, reference, name):
    array = _si(value, unit, reference)
    host = np.asarray(array)
    if host.shape != () or not np.isrealobj(host) or not np.isfinite(host):
        raise ValueError(f"{name} must be one finite real scalar.")
    return array


def _checked_finite(value, name, *, nonnegative=False, positive=False):
    array = jnp.asarray(value)
    array = array.astype(jnp.result_type(array, 1.0))
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise ValueError(f"{name} must be real-valued.")
    invalid = ~jnp.isfinite(array)
    if nonnegative:
        invalid = invalid | (array < 0)
    if positive:
        invalid = invalid | (array <= 0)
    return eqx.error_if(array, invalid, f"{name} is outside its finite physical domain.")


def _log_nonnegative(value):
    # A zero concentration is a genuinely absent species, not a density floor.
    return jnp.where(value > 0, jnp.log(jnp.where(value > 0, value, 1.0)), -jnp.inf)


def _implicit_scalar_root(function, lower, upper):
    """Native bracketed solve, differentiated only through its accepted equation."""
    initial = 0.5 * (lower + upper)
    tolerance = max(2e-12, 16 * jnp.finfo(initial.dtype).eps)
    termination = NonlinearTermination(
        absolute_residual=tolerance,
        relative_residual=0.0,
        absolute_step=0.0,
        relative_step=0.0,
        maximum_steps=160,
    )

    def solve(equation, guess):
        del guess
        result = scalar_root(
            ScalarRootProblem(
                lambda state, args: equation(state), bracket=(lower, upper)
            ),
            termination=termination,
        )
        return eqx.error_if(
            result.root,
            ~result.successful
            | ~jnp.isfinite(result.root)
            | (jnp.abs(result.value) > tolerance),
            "Band thermodynamic root did not satisfy its admitted equation.",
        )

    # This is a scalar linear equation, not a dense inverse or an iterative AD.
    return jax.lax.custom_root(
        function,
        initial,
        solve,
        lambda linearized, rhs: rhs / linearized(jnp.ones_like(rhs)),
    )


class BandThermodynamics(StrictModule):
    """Explicit aligned bands with Boltzmann or normalized Fermi--Dirac statistics.

    Parameters are scalar material data, not node-major solver coordinates.
    At psi=0 and Tref the band edges are the supplied energies, measured from
    ``energy_reference``. At general psi, both edges shift by -q*psi. Electronic
    quasi-Fermi energies use that same reference, including the *hole* EFp.

    The selected constant-effective-mass DOS law is Nc,v(T)=Nc,v(Tref)*(T/Tref)^1.5.
    Nc and Nv must be independently supplied; neither is inferred from ni.
    Ec(0,T)=Ec_ref+a_c*(T-Tref), and
    Eg(T)=Eg_ref-alpha*[T^2/(T+beta)-Tref^2/(Tref+beta)]. Ev=Ec-Eg.
    Coefficients have units J/K by default; beta and all temperatures are kelvin.
    Setting alpha=a_c=0 selects constant bands without another closure type.

    ``statistics`` is exactly ``"boltzmann"`` or ``"fermi-dirac"``. FD evaluation
    admits all finite negative reduced Fermi energies and eta <= 80. The latter
    is a declared numerical domain, not an asymptotic extrapolation. A composite
    native 32-point Gauss--Legendre rule integrates sqrt(energy/kT) from 0 to 12;
    the omitted occupation tail starts at least 64 kT above the Fermi energy.
    Log-scaled occupations preserve the nondegenerate tail without density floors.

    ``electron_energy_density`` and ``hole_energy_density`` return *kinetic*
    internal energy relative to the local carrier edge (J/m3):
    u=(3/2) Nc,v*kT*F_3/2(eta), with pressure 2u/3 and enthalpy 5u/3.
    The separate ``*_material_free_energy_density`` methods include Ec(0,T)*n
    or -Ev(0,T)*p. Their ``*_material_internal_energy_density`` counterparts
    evaluate f-T*partial_T(f) at fixed physical density, including band entropy.
    Neither material nor kinetic energies include -q*psi per electron, +q*psi
    per hole, or field energy. The total-energy model owns the field energy
    exactly once, together with ionic/lattice/filled-valence reference ledgers.
    """

    conduction_band_edge: Array
    valence_band_edge: Array
    conduction_density_of_states: Array
    valence_density_of_states: Array
    reference_temperature: Array
    conduction_temperature_coefficient: Array
    gap_varshni_alpha: Array
    gap_varshni_beta: Array
    _quadrature_energy: Array
    _density_weights: Array
    name: str = eqx.field(static=True)
    energy_reference: str = eqx.field(static=True)
    provenance: str = eqx.field(static=True)
    statistics: str = eqx.field(static=True)
    temperature_range: tuple[float, float] = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        /,
        *,
        conduction_band_edge,
        valence_band_edge,
        conduction_density_of_states,
        valence_density_of_states,
        reference_temperature,
        temperature_range,
        energy_reference: str,
        provenance: str,
        statistics: str = "boltzmann",
        conduction_temperature_coefficient=0.0,
        gap_varshni_alpha=0.0,
        gap_varshni_beta=1.0,
        energy_unit: UnitDefinition = JOULE,
        density_unit: UnitDefinition = PER_CUBIC_METER,
        temperature_unit: UnitDefinition = KELVIN,
        energy_temperature_unit: UnitDefinition = _ENERGY_PER_TEMPERATURE,
    ):
        self.name = _text(name, "band material name")
        self.energy_reference = _text(energy_reference, "electronic energy reference")
        self.provenance = _text(provenance, "band parameter provenance")
        if statistics not in ("boltzmann", "fermi-dirac"):
            raise ValueError("Statistics must be boltzmann or fermi-dirac.")
        self.statistics = statistics
        self.conduction_band_edge = _finite_scalar(
            conduction_band_edge, energy_unit, JOULE, "conduction band edge"
        )
        self.valence_band_edge = _finite_scalar(
            valence_band_edge, energy_unit, JOULE, "valence band edge"
        )
        self.conduction_density_of_states = _positive_scalar(
            conduction_density_of_states,
            density_unit,
            PER_CUBIC_METER,
            "Nc reference DOS",
        )
        self.valence_density_of_states = _positive_scalar(
            valence_density_of_states, density_unit, PER_CUBIC_METER, "Nv reference DOS"
        )
        self.reference_temperature = _positive_scalar(
            reference_temperature, temperature_unit, KELVIN, "reference temperature"
        )
        self.conduction_temperature_coefficient = _finite_scalar(
            conduction_temperature_coefficient,
            energy_temperature_unit,
            _ENERGY_PER_TEMPERATURE,
            "conduction edge temperature coefficient",
        )
        self.gap_varshni_alpha = _positive_scalar(
            gap_varshni_alpha,
            energy_temperature_unit,
            _ENERGY_PER_TEMPERATURE,
            "Varshni alpha",
            nonnegative=True,
        )
        self.gap_varshni_beta = _positive_scalar(
            gap_varshni_beta, temperature_unit, KELVIN, "Varshni beta"
        )
        bounds = np.asarray(_si(temperature_range, temperature_unit, KELVIN))
        if (
            bounds.shape != (2,)
            or not np.all(np.isfinite(bounds))
            or bounds[0] <= 0
            or bounds[1] < bounds[0]
            or not bounds[0] <= float(self.reference_temperature) <= bounds[1]
        ):
            raise ValueError("Temperature domain must be positive and include Tref.")
        self.temperature_range = (float(bounds[0]), float(bounds[1]))
        # Alpha >= 0 makes the minimum gap occur at the upper endpoint.
        minimum_gap = float(self._band_gap(jnp.asarray(bounds[1])))
        if not math.isfinite(minimum_gap) or minimum_gap <= 0:
            raise ValueError(
                "Band gap must remain finite and positive throughout the temperature domain."
            )
        if statistics == "fermi-dirac":
            rule = GaussLegendreRule(32).data()
            points = (jnp.arange(12)[:, None] + 0.5 * (rule.nodes + 1)).reshape(-1)
            weights = jnp.broadcast_to(0.5 * rule.weights, (12, 32)).reshape(-1)
            self._quadrature_energy = points**2
            self._density_weights = (4 / math.sqrt(math.pi)) * weights * points**2
        else:
            self._quadrature_energy = jnp.empty((0,))
            self._density_weights = jnp.empty((0,))

    def _band_gap(self, temperature):
        tref, beta = self.reference_temperature, self.gap_varshni_beta
        return (
            self.conduction_band_edge
            - self.valence_band_edge
            - self.gap_varshni_alpha
            * (temperature**2 / (temperature + beta) - tref**2 / (tref + beta))
        )

    def temperature_valid(self, temperature):
        """Elementwise JAX validity predicate; no host conversion or exception."""
        temperature = jnp.asarray(temperature)
        lower, upper = self.temperature_range
        gap = self._band_gap(temperature)
        return (
            jnp.isfinite(temperature)
            & (temperature >= lower)
            & (temperature <= upper)
            & jnp.isfinite(gap)
            & (gap > 0)
        )

    def admit_temperature(self, temperature):
        """Host-only admission of SI temperatures before topology/solver preparation."""
        if not np.all(np.asarray(self.temperature_valid(temperature))):
            raise ValueError("Temperature is outside the admitted band material domain.")

    def _temperature(self, temperature):
        temperature = _checked_finite(temperature, "temperature", positive=True)
        return eqx.error_if(
            temperature,
            ~self.temperature_valid(temperature),
            "Temperature is outside the admitted band material domain.",
        )

    def density_of_states(self, T):
        """Return (Nc(T), Nv(T)) in m^-3 for the declared parabolic DOS law."""
        temperature = self._temperature(T)
        factor = (temperature / self.reference_temperature) ** 1.5
        return (
            _checked_finite(
                self.conduction_density_of_states * factor,
                "conduction DOS",
                positive=True,
            ),
            _checked_finite(
                self.valence_density_of_states * factor, "valence DOS", positive=True
            ),
        )

    def band_edges(self, psi, T):
        """Return (Ec, Ev) in J, both shifted by -q*psi relative to the reference."""
        temperature = self._temperature(T)
        potential = _checked_finite(psi, "electrostatic potential")
        ec = (
            self.conduction_band_edge
            + self.conduction_temperature_coefficient
            * (temperature - self.reference_temperature)
            - _Q * potential
        )
        return ec, ec - self._band_gap(temperature)

    def material_band_temperature_derivatives(self, T):
        """Return (dEc(0,T)/dT, dEv(0,T)/dT) in J/K, excluding electrostatics.

        These are the band-entropy terms needed when converting carrier
        Helmholtz free energy into internal energy at fixed number density.
        """
        temperature = self._temperature(T)
        beta = self.gap_varshni_beta
        conduction_derivative = jnp.broadcast_to(
            self.conduction_temperature_coefficient, temperature.shape
        )
        gap_derivative = (
            -self.gap_varshni_alpha
            * (temperature / (temperature + beta))
            * ((temperature + 2 * beta) / (temperature + beta))
        )
        return conduction_derivative, conduction_derivative - gap_derivative

    def _eta(self, eta):
        eta = _checked_finite(eta, "reduced Fermi energy")
        if self.statistics == "fermi-dirac":
            eta = eqx.error_if(
                eta,
                eta > _MAXIMUM_FD_ETA,
                "Fermi-Dirac reduced energy exceeds the admitted maximum of 80.",
            )
        return eta

    def _fd_occupation(self, eta):
        argument = eta[..., None] - self._quadrature_energy
        shift = jnp.minimum(eta, 0.0)
        scaled = jnp.exp(
            (eta - shift)[..., None] - self._quadrature_energy - jax.nn.softplus(argument)
        )
        return shift, scaled, argument

    def _log_statistics(self, eta):
        if self.statistics == "boltzmann":
            return eta
        shift, scaled, _ = self._fd_occupation(eta)
        return shift + jnp.log(
            contract("...q,q->...", scaled, self._density_weights, backend="jax")
        )

    def statistics_value(self, eta):
        """Normalized F_1/2(eta), or exp(eta) for the selected MB reduction."""
        value = jnp.exp(self._log_statistics(self._eta(eta)))
        return _checked_finite(value, "normalized carrier population", positive=True)

    def statistics_derivative(self, eta):
        """Derivative of the selected F_1/2, using its own occupation quadrature."""
        eta = self._eta(eta)
        if self.statistics == "boltzmann":
            return self.statistics_value(eta)
        shift, scaled, argument = self._fd_occupation(eta)
        value = jnp.exp(shift) * contract(
            "...q,q->...",
            scaled * jax.nn.sigmoid(-argument),
            self._density_weights,
            backend="jax",
        )
        return _checked_finite(value, "population compressibility", positive=True)

    def _inverse_log_statistics(self, log_population):
        if self.statistics == "boltzmann":
            return log_population
        maximum = self._log_statistics(jnp.asarray(_MAXIMUM_FD_ETA))
        log_population = eqx.error_if(
            log_population,
            log_population > maximum,
            "Carrier density exceeds the admitted Fermi-Dirac domain.",
        )

        def solve_one(target):
            # F_1/2(eta) <= exp(eta); the negative lower endpoint safely brackets
            # even the most dilute representable physical density.
            lower = jnp.minimum(target - 2.0, -2.0)
            upper = jnp.asarray(_MAXIMUM_FD_ETA, dtype=target.dtype)
            return _implicit_scalar_root(
                lambda eta: self._log_statistics(eta) - target, lower, upper
            )

        flat = jnp.reshape(log_population, (-1,))
        return jax.vmap(solve_one)(flat).reshape(jnp.shape(log_population))

    def inverse_statistics(self, population):
        """Return eta for a strictly positive normalized density, implicitly differentiated."""
        population = _checked_finite(
            population, "normalized carrier density", positive=True
        )
        return self._inverse_log_statistics(jnp.log(population))

    def electron_density(self, psi, efn, T):
        temperature = self._temperature(T)
        ec, _ = self.band_edges(psi, temperature)
        nc, _ = self.density_of_states(temperature)
        eta = self._eta(
            (_checked_finite(efn, "electron Fermi energy") - ec) / (_KB * temperature)
        )
        return _checked_finite(
            jnp.exp(jnp.log(nc) + self._log_statistics(eta)),
            "electron density",
            positive=True,
        )

    def hole_density(self, psi, efp, T):
        temperature = self._temperature(T)
        _, ev = self.band_edges(psi, temperature)
        _, nv = self.density_of_states(temperature)
        eta = self._eta(
            (ev - _checked_finite(efp, "hole Fermi energy")) / (_KB * temperature)
        )
        return _checked_finite(
            jnp.exp(jnp.log(nv) + self._log_statistics(eta)),
            "hole density",
            positive=True,
        )

    def electron_fermi_energy(self, psi, n, T):
        temperature = self._temperature(T)
        density = _checked_finite(n, "electron density", positive=True)
        ec, _ = self.band_edges(psi, temperature)
        nc, _ = self.density_of_states(temperature)
        return ec + _KB * temperature * self._inverse_log_statistics(
            jnp.log(density) - jnp.log(nc)
        )

    def hole_fermi_energy(self, psi, p, T):
        temperature = self._temperature(T)
        density = _checked_finite(p, "hole density", positive=True)
        _, ev = self.band_edges(psi, temperature)
        _, nv = self.density_of_states(temperature)
        return ev - _KB * temperature * self._inverse_log_statistics(
            jnp.log(density) - jnp.log(nv)
        )

    def _logarithmic_derivative(self, eta):
        if self.statistics == "boltzmann":
            return jnp.ones_like(eta)
        _, scaled, argument = self._fd_occupation(eta)
        return contract(
            "...q,q->...",
            scaled * jax.nn.sigmoid(-argument),
            self._density_weights,
            backend="jax",
        ) / contract("...q,q->...", scaled, self._density_weights, backend="jax")

    def electron_compressibility(self, psi, efn, T):
        """Positive dn/dEFn at fixed psi,T, in m^-3/J."""
        temperature = self._temperature(T)
        ec, _ = self.band_edges(psi, temperature)
        eta = self._eta((jnp.asarray(efn) - ec) / (_KB * temperature))
        value = (
            self.electron_density(psi, efn, temperature)
            * self._logarithmic_derivative(eta)
            / (_KB * temperature)
        )
        return _checked_finite(value, "electron compressibility", positive=True)

    def hole_compressibility(self, psi, efp, T):
        """Positive -dp/dEFp at fixed psi,T, in m^-3/J (hole chemical potential is -EFp)."""
        temperature = self._temperature(T)
        _, ev = self.band_edges(psi, temperature)
        eta = self._eta((ev - jnp.asarray(efp)) / (_KB * temperature))
        value = (
            self.hole_density(psi, efp, temperature)
            * self._logarithmic_derivative(eta)
            / (_KB * temperature)
        )
        return _checked_finite(value, "hole compressibility", positive=True)

    def _einstein_ratio(self, density, temperature, dos):
        density = _checked_finite(density, "carrier density", positive=True)
        eta = self._inverse_log_statistics(jnp.log(density) - jnp.log(dos))
        return (_KB * temperature / _Q) / self._logarithmic_derivative(eta)

    def electron_einstein_ratio(self, n, T):
        """Electron D/mu in V, reducing to kT/q in the MB limit."""
        temperature = self._temperature(T)
        nc, _ = self.density_of_states(temperature)
        return self._einstein_ratio(n, temperature, nc)

    def hole_einstein_ratio(self, p, T):
        """Hole D/mu in V, with positive hole compressibility."""
        temperature = self._temperature(T)
        _, nv = self.density_of_states(temperature)
        return self._einstein_ratio(p, temperature, nv)

    def _kinetic_energy(self, density, temperature, dos):
        density = _checked_finite(density, "carrier density", nonnegative=True)
        if self.statistics == "boltzmann":
            return _checked_finite(
                density * (1.5 * _KB * temperature),
                "carrier kinetic energy",
                nonnegative=True,
            )
        return self._fd_kinetic_state(density, temperature, dos)[1]

    def _fd_kinetic_state(self, density, temperature, dos):
        """Reuse one FD inverse and occupation evaluation for eta and kinetic u."""
        # Exactly empty populations have zero energy and the dilute right
        # derivative. The dummy log-density only keeps the unselected root finite.
        logarithm = jnp.log(jnp.where(density > 0, density, dos)) - jnp.log(dos)
        eta = self._inverse_log_statistics(logarithm)
        _, scaled, _ = self._fd_occupation(eta)
        mean_energy = contract(
            "...q,q->...",
            scaled,
            self._density_weights * self._quadrature_energy,
            backend="jax",
        ) / contract("...q,q->...", scaled, self._density_weights, backend="jax")
        energy = _checked_finite(
            density * (_KB * temperature * jnp.where(density > 0, mean_energy, 1.5)),
            "carrier kinetic energy",
            nonnegative=True,
        )
        return eta, energy

    def electron_energy_density(self, n, T):
        """Electron kinetic internal energy above Ec in J/m3; n=0 returns zero."""
        temperature = self._temperature(T)
        nc, _ = self.density_of_states(temperature)
        return self._kinetic_energy(n, temperature, nc)

    def hole_energy_density(self, p, T):
        """Hole kinetic internal energy below Ev in J/m3; p=0 returns zero."""
        temperature = self._temperature(T)
        _, nv = self.density_of_states(temperature)
        return self._kinetic_energy(p, temperature, nv)

    def _kinetic_free_energy(self, density, temperature, dos):
        density = _checked_finite(density, "carrier density", positive=True)
        if self.statistics == "boltzmann":
            eta = jnp.log(density) - jnp.log(dos)
            return density * (_KB * temperature * (eta - 1.0))
        eta, energy = self._fd_kinetic_state(density, temperature, dos)
        return density * (_KB * temperature * eta) - (2 / 3) * energy

    def electron_material_free_energy_density(self, n, T):
        """Electron material Helmholtz density in J/m3, excluding electrostatics.

        f=n*Ec(0,T)+n*kT*eta-2*u_kinetic/3. Its density derivative is
        EFn+q*psi. As for finite chemical energy, n must be strictly positive;
        the chemical derivative is singular at an exactly empty population.
        """
        temperature = self._temperature(T)
        density = _checked_finite(n, "electron density", positive=True)
        nc, _ = self.density_of_states(temperature)
        ec, _ = self.band_edges(0.0, temperature)
        return _checked_finite(
            density * ec + self._kinetic_free_energy(density, temperature, nc),
            "electron material free energy",
        )

    def hole_material_free_energy_density(self, p, T):
        """Hole material Helmholtz density in J/m3, excluding electrostatics.

        f=-p*Ev(0,T)+p*kT*eta_h-2*u_kinetic/3. Its density derivative is
        -EFp-q*psi, not EFp. The filled-valence reference energy is excluded.
        Like finite chemical energy, this method requires p > 0.
        """
        temperature = self._temperature(T)
        density = _checked_finite(p, "hole density", positive=True)
        _, nv = self.density_of_states(temperature)
        _, ev = self.band_edges(0.0, temperature)
        return _checked_finite(
            -density * ev + self._kinetic_free_energy(density, temperature, nv),
            "hole material free energy",
        )

    def electron_material_internal_energy_density(self, n, T):
        """Electron u=f-T*partial_T(f)|n in J/m3, with no electrostatic energy.

        u=u_kinetic+n*(Ec(0,T)-T*dEc(0,T)/dT), not u_kinetic+n*Ec(psi,T).
        Zero density returns zero. Negative values merely reflect the declared
        electronic energy reference and are not negative kinetic energies.
        """
        temperature = self._temperature(T)
        density = _checked_finite(n, "electron density", nonnegative=True)
        ec, _ = self.band_edges(0.0, temperature)
        dec, _ = self.material_band_temperature_derivatives(temperature)
        return _checked_finite(
            self.electron_energy_density(density, temperature)
            + density * (ec - temperature * dec),
            "electron material internal energy",
        )

    def hole_material_internal_energy_density(self, p, T):
        """Hole u=f-T*partial_T(f)|p in J/m3, with no electrostatic energy.

        u=u_kinetic+p*(-Ev(0,T)+T*dEv(0,T)/dT). The filled-valence, ionic,
        lattice and field-energy reference ledgers remain separate.
        Zero density returns zero.
        """
        temperature = self._temperature(T)
        density = _checked_finite(p, "hole density", nonnegative=True)
        _, ev = self.band_edges(0.0, temperature)
        _, dev = self.material_band_temperature_derivatives(temperature)
        return _checked_finite(
            self.hole_energy_density(density, temperature)
            + density * (-ev + temperature * dev),
            "hole material internal energy",
        )

    def equilibrium_fermi_energy(
        self, psi, T, donors=0.0, acceptors=0.0, *, ionization=None
    ):
        """Unique neutral common EF in J; ``ionization=None`` means full ionization.

        Neutrality is p+Nd_plus=n+Na_minus, at one common temperature and one
        electronic chemical energy. It is solved in dimensionless energies with
        logarithmic charge balance, so freeze-out and compensation do not lose
        the minority population or silently become fully ionized. Material and
        dopant parameters retain implicit forward/reverse JAX derivatives.
        """
        if ionization is not None and not isinstance(ionization, IncompleteIonization):
            raise TypeError("ionization must be IncompleteIonization or None.")
        temperature = self._temperature(T)
        potential = _checked_finite(psi, "electrostatic potential")
        donors = _checked_finite(donors, "donor concentration", nonnegative=True)
        acceptors = _checked_finite(acceptors, "acceptor concentration", nonnegative=True)
        arrays = jnp.broadcast_arrays(potential, temperature, donors, acceptors)

        def solve_one(potential, temperature, nd, na):
            ec, ev = self.band_edges(potential, temperature)
            nc, nv = self.density_of_states(temperature)
            kt = _KB * temperature
            half_gap = (ec - ev) / (2 * kt)
            center = 0.5 * (ec + ev)
            log_nd, log_na = _log_nonnegative(nd), _log_nonnegative(na)

            def residual(coordinate):
                log_n = jnp.log(nc) + self._log_statistics(coordinate - half_gap)
                log_p = jnp.log(nv) + self._log_statistics(-coordinate - half_gap)
                if ionization is None:
                    log_nd_plus, log_na_minus = log_nd, log_na
                else:
                    log_donor_fraction, log_acceptor_fraction = ionization._log_fractions(
                        ec, ev, center + kt * coordinate, temperature
                    )
                    log_nd_plus = log_nd + log_donor_fraction
                    log_na_minus = log_na + log_acceptor_fraction
                return jnp.logaddexp(log_p, log_nd_plus) - jnp.logaddexp(
                    log_n, log_na_minus
                )

            if self.statistics == "fermi-dirac":
                bound = half_gap + _MAXIMUM_FD_ETA
            else:
                bound = (
                    half_gap
                    + 2
                    + jnp.maximum(
                        jnp.logaddexp(log_nd, log_na)
                        - jnp.minimum(jnp.log(nc), jnp.log(nv)),
                        0.5 * jnp.abs(jnp.log(nc) - jnp.log(nv)),
                    )
                )
            coordinate = _implicit_scalar_root(residual, -bound, bound)
            return center + kt * coordinate

        shape = arrays[0].shape
        flat = tuple(array.reshape(-1) for array in arrays)
        return jax.vmap(solve_one)(*flat).reshape(shape)


class IncompleteIonization(StrictModule):
    """Single donor and acceptor levels in local equilibrium with common EF.

    Ed=Ec-donor_binding_energy and Ea=Ev+acceptor_binding_energy. The explicit
    degeneracy convention is Nd_plus=Nd/[1+gD*exp((EF-Ed)/kT)] and
    Na_minus=Na/[1+gA*exp((Ea-EF)/kT)]. Thus the usual noninteracting shallow-level
    values are gD=2, gA=4; these are not silently imposed on a different defect.
    See Sze and Ng, *Physics of Semiconductor Devices*, third edition, ch. 1,
    https://doi.org/10.1002/0470068329. Binding energies and degeneracies require
    their own declared provenance and are not process-calibrated by this model.

    This is an equilibrium impurity closure, not trap kinetics or a prescription
    to use an arbitrary average of split quasi-Fermi levels out of equilibrium.
    """

    donor_binding_energy: Array
    acceptor_binding_energy: Array
    donor_degeneracy: Array
    acceptor_degeneracy: Array
    provenance: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        donor_binding_energy,
        acceptor_binding_energy,
        donor_degeneracy,
        acceptor_degeneracy,
        provenance: str,
        energy_unit: UnitDefinition = JOULE,
    ):
        self.donor_binding_energy = _positive_scalar(
            donor_binding_energy, energy_unit, JOULE, "donor binding energy"
        )
        self.acceptor_binding_energy = _positive_scalar(
            acceptor_binding_energy, energy_unit, JOULE, "acceptor binding energy"
        )
        self.donor_degeneracy = _positive_scalar(
            donor_degeneracy, ONE, ONE, "donor degeneracy"
        )
        self.acceptor_degeneracy = _positive_scalar(
            acceptor_degeneracy, ONE, ONE, "acceptor degeneracy"
        )
        self.provenance = _text(provenance, "ionization parameter provenance")

    def admit(self, bands: BandThermodynamics):
        """Host-only check that both impurity levels stay strictly inside the gap."""
        if not isinstance(bands, BandThermodynamics):
            raise TypeError("Impurity admission requires BandThermodynamics.")
        gap = float(bands._band_gap(jnp.asarray(bands.temperature_range[1])))
        if (
            max(float(self.donor_binding_energy), float(self.acceptor_binding_energy))
            >= gap
        ):
            raise ValueError(
                "Impurity binding energies must remain strictly inside the gap."
            )

    def _log_fractions(self, ec, ev, ef, temperature):
        return self._log_fractions_split(ec, ev, ef, ef, temperature)

    def _log_fractions_split(self, ec, ev, electron_fermi, hole_fermi, temperature):
        """Ionized donor/acceptor fractions with explicit carrier reservoirs."""
        gap = ec - ev
        electron_fermi = eqx.error_if(
            electron_fermi,
            (self.donor_binding_energy >= gap) | (self.acceptor_binding_energy >= gap),
            "Impurity levels are outside the material band gap.",
        )
        donor_argument = (electron_fermi - ec + self.donor_binding_energy) / (
            _KB * temperature
        )
        acceptor_argument = (ev + self.acceptor_binding_energy - hole_fermi) / (
            _KB * temperature
        )
        return (
            -jax.nn.softplus(jnp.log(self.donor_degeneracy) + donor_argument),
            -jax.nn.softplus(jnp.log(self.acceptor_degeneracy) + acceptor_argument),
        )

    def ionized_densities(self, bands: BandThermodynamics, psi, ef, T, donors, acceptors):
        """Return (Nd_plus, Na_minus), each nonnegative and bounded by its SI total."""
        temperature = bands._temperature(T)
        ec, ev = bands.band_edges(psi, temperature)
        ef = _checked_finite(ef, "common impurity Fermi energy")
        donors = _checked_finite(donors, "donor concentration", nonnegative=True)
        acceptors = _checked_finite(acceptors, "acceptor concentration", nonnegative=True)
        log_donor, log_acceptor = self._log_fractions(ec, ev, ef, temperature)
        return donors * jnp.exp(log_donor), acceptors * jnp.exp(log_acceptor)

    def ionized_densities_split(
        self,
        bands: BandThermodynamics,
        psi,
        electron_fermi,
        hole_fermi,
        T,
        donors,
        acceptors,
    ):
        """Out-of-equilibrium donor/electron and acceptor/hole reservoir closure."""
        temperature = bands._temperature(T)
        ec, ev = bands.band_edges(psi, temperature)
        electron_fermi = _checked_finite(electron_fermi, "electron impurity Fermi energy")
        hole_fermi = _checked_finite(hole_fermi, "hole impurity Fermi energy")
        donors = _checked_finite(donors, "donor concentration", nonnegative=True)
        acceptors = _checked_finite(acceptors, "acceptor concentration", nonnegative=True)
        log_donor, log_acceptor = self._log_fractions_split(
            ec, ev, electron_fermi, hole_fermi, temperature
        )
        return donors * jnp.exp(log_donor), acceptors * jnp.exp(log_acceptor)

    def bound_energy_density(
        self, bands: BandThermodynamics, psi, ef, T, donors, acceptors
    ):
        """Electronic impurity energy in J/m3 relative to empty impurity levels.

        Occupied donors contribute Ed, occupied acceptors contribute Ea. Host ion
        energies and electrostatic field energy are excluded. This *electronic*
        inventory shifts with the declared reference, unlike kinetic carrier
        energy; it must be combined with ionic/reference ledgers for total energy.
        """
        temperature = bands._temperature(T)
        ec, ev = bands.band_edges(psi, temperature)
        ef = _checked_finite(ef, "common impurity Fermi energy")
        donors = _checked_finite(donors, "donor concentration", nonnegative=True)
        acceptors = _checked_finite(acceptors, "acceptor concentration", nonnegative=True)
        log_donor, log_acceptor = self._log_fractions(ec, ev, ef, temperature)
        donor_occupied = donors * (-jnp.expm1(log_donor))
        acceptor_occupied = acceptors * jnp.exp(log_acceptor)
        return (ec - self.donor_binding_energy) * donor_occupied + (
            ev + self.acceptor_binding_energy
        ) * acceptor_occupied

    def bound_energy_density_split(
        self,
        bands: BandThermodynamics,
        psi,
        electron_fermi,
        hole_fermi,
        T,
        donors,
        acceptors,
    ):
        """Electronic bound-impurity energy using the declared split reservoirs."""
        temperature = bands._temperature(T)
        ec, ev = bands.band_edges(psi, temperature)
        donors = _checked_finite(donors, "donor concentration", nonnegative=True)
        acceptors = _checked_finite(acceptors, "acceptor concentration", nonnegative=True)
        log_donor, log_acceptor = self._log_fractions_split(
            ec,
            ev,
            _checked_finite(electron_fermi, "electron impurity Fermi energy"),
            _checked_finite(hole_fermi, "hole impurity Fermi energy"),
            temperature,
        )
        donor_occupied = donors * (-jnp.expm1(log_donor))
        acceptor_occupied = acceptors * jnp.exp(log_acceptor)
        return (ec - self.donor_binding_energy) * donor_occupied + (
            ev + self.acceptor_binding_energy
        ) * acceptor_occupied


__all__ = ["BandThermodynamics", "IncompleteIonization"]
