#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Disjoining-pressure laws and black-film equilibria for thin liquid films.

Sign convention: ``Pi(h) > 0`` is repulsive (it thickens the film). The film
pressure is ``p = p_capillary - Pi(h)`` and the interaction energy per unit
area is ``W(h) = int_h^inf Pi(s) ds`` so that ``W'(h) = -Pi(h)``. A flat film
at equilibrium with a meniscus of capillary suction ``P_c`` satisfies
``Pi(h) = P_c``; it is mechanically stable only where ``dPi/dh < 0``.
"""

from __future__ import annotations

import abc
from enum import IntEnum
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import parameter_field, ParameterOwner
from ..nonlinear import Brent, NonlinearTermination, scalar_root, ScalarRootProblem


# Exact SI defining constants (2019 SI) and CODATA 2018 vacuum permittivity.
_BOLTZMANN_J_K = 1.380649e-23
_ELEMENTARY_CHARGE_C = 1.602176634e-19
_AVOGADRO_PER_MOL = 6.02214076e23
_VACUUM_PERMITTIVITY_F_M = 8.8541878128e-12


def _positive_scalar(value: ArrayLike, name: str, /) -> Array:
    host = np.asarray(value, dtype=np.float64)
    if host.shape != () or not isfinite(float(host)) or float(host) <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar.")
    return jnp.asarray(host)


def _thickness(value: ArrayLike, /) -> Array:
    return jnp.asarray(value, dtype=jnp.float64)


class AbstractDisjoiningPressure(StrictModule):
    """Disjoining pressure ``Pi(h)`` with its derivative and energy ``W(h)``."""

    @abc.abstractmethod
    def pressure(self, thickness_m: ArrayLike, /) -> Array:
        """Return ``Pi(h)`` in Pa."""

    @abc.abstractmethod
    def derivative(self, thickness_m: ArrayLike, /) -> Array:
        """Return ``dPi/dh`` in Pa/m."""

    @abc.abstractmethod
    def energy(self, thickness_m: ArrayLike, /) -> Array:
        """Return ``W(h) = int_h^inf Pi`` in J/m^2."""

    @abc.abstractmethod
    def monotone_decreasing(self) -> Array:
        """Return whether ``Pi`` is non-increasing for every ``h > 0``.

        This makes ``W`` convex, the premise of unconditional energy decay of
        backward-Euler film drainage.
        """

    @property
    @abc.abstractmethod
    def law_id(self) -> str:
        """Return the static identity of the law form."""


class VanDerWaalsDisjoiningPressure(AbstractDisjoiningPressure, ParameterOwner):
    """Non-retarded van der Waals ``Pi = -A / (6 pi h^3)``.

    A positive Hamaker constant ``A`` is attractive (destabilizing) for a
    symmetric film in air.
    """

    hamaker_constant_j: Array = parameter_field()

    def __init__(self, hamaker_constant_j: ArrayLike, /) -> None:
        value = np.asarray(hamaker_constant_j, dtype=np.float64)
        if value.shape != () or not isfinite(float(value)):
            raise ValueError("hamaker_constant_j must be a finite scalar.")
        self.hamaker_constant_j = jnp.asarray(value)

    def pressure(self, thickness_m: ArrayLike, /) -> Array:
        h = _thickness(thickness_m)
        return -self.hamaker_constant_j / (6.0 * jnp.pi * h**3)

    def derivative(self, thickness_m: ArrayLike, /) -> Array:
        h = _thickness(thickness_m)
        return self.hamaker_constant_j / (2.0 * jnp.pi * h**4)

    def energy(self, thickness_m: ArrayLike, /) -> Array:
        h = _thickness(thickness_m)
        return -self.hamaker_constant_j / (12.0 * jnp.pi * h**2)

    def monotone_decreasing(self) -> Array:
        return self.hamaker_constant_j <= 0.0

    @property
    def law_id(self) -> str:
        return "van-der-waals-nonretarded"


class DoubleLayerDisjoiningPressure(AbstractDisjoiningPressure, ParameterOwner):
    """Weak-overlap electrostatic double layer between two equal interfaces.

    ``Pi = 64 n k T gamma^2 exp(-kappa h)`` with ``n = c N_A``,
    ``gamma = tanh(z e psi / 4 k T)`` and Debye parameter
    ``kappa^2 = 2 n z^2 e^2 / (eps0 eps_r k T)`` for a symmetric ``z:z``
    electrolyte (Israelachvili, *Intermolecular and Surface Forces*, 3rd ed.,
    eq. 14.36). Valid for ``kappa h`` of order one or larger.
    """

    ionic_concentration_mol_m3: Array = parameter_field()
    surface_potential_v: Array = parameter_field()
    temperature_k: Array = parameter_field()
    relative_permittivity: Array = parameter_field()
    valence: int = eqx.field(static=True)

    def __init__(
        self,
        ionic_concentration_mol_m3: ArrayLike,
        surface_potential_v: ArrayLike,
        temperature_k: ArrayLike,
        relative_permittivity: ArrayLike,
        /,
        *,
        valence: int = 1,
    ) -> None:
        concentration = _positive_scalar(
            ionic_concentration_mol_m3, "ionic_concentration_mol_m3"
        )
        potential = np.asarray(surface_potential_v, dtype=np.float64)
        if potential.shape != () or not isfinite(float(potential)):
            raise ValueError("surface_potential_v must be a finite scalar.")
        temperature = _positive_scalar(temperature_k, "temperature_k")
        permittivity = _positive_scalar(relative_permittivity, "relative_permittivity")
        if isinstance(valence, bool) or not isinstance(valence, int) or valence < 1:
            raise ValueError("valence must be a positive integer.")
        self.ionic_concentration_mol_m3 = concentration
        self.surface_potential_v = jnp.asarray(potential)
        self.temperature_k = temperature
        self.relative_permittivity = permittivity
        self.valence = valence

    @property
    def debye_parameter_m_inv(self) -> Array:
        number = self.ionic_concentration_mol_m3 * _AVOGADRO_PER_MOL
        charge = self.valence * _ELEMENTARY_CHARGE_C
        return jnp.sqrt(
            2.0
            * number
            * charge**2
            / (
                _VACUUM_PERMITTIVITY_F_M
                * self.relative_permittivity
                * _BOLTZMANN_J_K
                * self.temperature_k
            )
        )

    @property
    def contact_pressure_pa(self) -> Array:
        thermal = _BOLTZMANN_J_K * self.temperature_k
        number = self.ionic_concentration_mol_m3 * _AVOGADRO_PER_MOL
        gamma = jnp.tanh(
            self.valence
            * _ELEMENTARY_CHARGE_C
            * self.surface_potential_v
            / (4.0 * thermal)
        )
        return 64.0 * number * thermal * gamma**2

    def pressure(self, thickness_m: ArrayLike, /) -> Array:
        h = _thickness(thickness_m)
        return self.contact_pressure_pa * jnp.exp(-self.debye_parameter_m_inv * h)

    def derivative(self, thickness_m: ArrayLike, /) -> Array:
        return -self.debye_parameter_m_inv * self.pressure(thickness_m)

    def energy(self, thickness_m: ArrayLike, /) -> Array:
        return self.pressure(thickness_m) / self.debye_parameter_m_inv

    def monotone_decreasing(self) -> Array:
        return jnp.asarray(True)

    @property
    def law_id(self) -> str:
        return f"double-layer-weak-overlap-z{self.valence}"


class ShortRangeRepulsionPressure(AbstractDisjoiningPressure, ParameterOwner):
    """Short-range (Born/steric) repulsion ``Pi = P0 (l / h)^n`` with ``n > 1``."""

    pressure_scale_pa: Array = parameter_field()
    length_scale_m: Array = parameter_field()
    exponent: int = eqx.field(static=True)

    def __init__(
        self,
        pressure_scale_pa: ArrayLike,
        length_scale_m: ArrayLike,
        /,
        *,
        exponent: int = 9,
    ) -> None:
        pressure = _positive_scalar(pressure_scale_pa, "pressure_scale_pa")
        length = _positive_scalar(length_scale_m, "length_scale_m")
        if isinstance(exponent, bool) or not isinstance(exponent, int) or exponent < 2:
            raise ValueError("exponent must be an integer of at least two.")
        self.pressure_scale_pa = pressure
        self.length_scale_m = length
        self.exponent = exponent

    def pressure(self, thickness_m: ArrayLike, /) -> Array:
        h = _thickness(thickness_m)
        return self.pressure_scale_pa * (self.length_scale_m / h) ** self.exponent

    def derivative(self, thickness_m: ArrayLike, /) -> Array:
        h = _thickness(thickness_m)
        return -self.exponent * self.pressure(h) / h

    def energy(self, thickness_m: ArrayLike, /) -> Array:
        h = _thickness(thickness_m)
        return self.pressure(h) * h / (self.exponent - 1)

    def monotone_decreasing(self) -> Array:
        return jnp.asarray(True)

    @property
    def law_id(self) -> str:
        return f"short-range-power-{self.exponent}"


class CompositeDisjoiningPressure(AbstractDisjoiningPressure):
    """Additive superposition of disjoining contributions, e.g. DLVO."""

    components: tuple[AbstractDisjoiningPressure, ...]

    def __init__(self, components: tuple[AbstractDisjoiningPressure, ...], /) -> None:
        if not isinstance(components, tuple) or not components:
            raise ValueError("components must be a non-empty tuple.")
        if not all(isinstance(item, AbstractDisjoiningPressure) for item in components):
            raise TypeError("Every component must be an AbstractDisjoiningPressure.")
        self.components = components

    def pressure(self, thickness_m: ArrayLike, /) -> Array:
        return sum(
            (item.pressure(thickness_m) for item in self.components[1:]),
            self.components[0].pressure(thickness_m),
        )

    def derivative(self, thickness_m: ArrayLike, /) -> Array:
        return sum(
            (item.derivative(thickness_m) for item in self.components[1:]),
            self.components[0].derivative(thickness_m),
        )

    def energy(self, thickness_m: ArrayLike, /) -> Array:
        return sum(
            (item.energy(thickness_m) for item in self.components[1:]),
            self.components[0].energy(thickness_m),
        )

    def monotone_decreasing(self) -> Array:
        flags = jnp.stack([item.monotone_decreasing() for item in self.components])
        return jnp.all(flags)

    @property
    def law_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "composite-disjoining-pressure",
                "components": [item.law_id for item in self.components],
            }
        )


class BlackFilmStatus(IntEnum):
    """Terminal status of a black-film equilibrium search."""

    CONVERGED = 0
    NO_ROOT_IN_BRACKET = 1
    NOT_CONVERGED = 2


class BlackFilmEquilibriumResult(StrictModule):
    """Equilibrium ``Pi(h) = P_c`` inside one declared thickness bracket.

    ``stable`` reports ``dPi/dh < 0`` at the root; ``status`` is a
    ``BlackFilmStatus`` and ``nonlinear_status`` the native scalar-root status.
    """

    thickness_m: Array
    disjoining_slope_pa_m: Array
    relative_residual: Array
    stable: Array
    status: Array
    nonlinear_status: Array
    iterations: Array
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        thickness_m: Array,
        disjoining_slope_pa_m: Array,
        relative_residual: Array,
        stable: Array,
        status: Array,
        nonlinear_status: Array,
        iterations: Array,
        law_id: str,
    ) -> None:
        self.thickness_m = jnp.asarray(thickness_m)
        self.disjoining_slope_pa_m = jnp.asarray(disjoining_slope_pa_m)
        self.relative_residual = jnp.asarray(relative_residual)
        self.stable = jnp.asarray(stable, dtype=jnp.bool_)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.nonlinear_status = jnp.asarray(nonlinear_status, dtype=jnp.int32)
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32)
        self.law_id = law_id

    @property
    def successful(self) -> Array:
        return self.status == BlackFilmStatus.CONVERGED


def black_film_equilibrium(
    law: AbstractDisjoiningPressure,
    capillary_pressure_pa: ArrayLike,
    thickness_bracket_m: tuple[float, float],
    /,
    *,
    termination: NonlinearTermination | None = None,
) -> BlackFilmEquilibriumResult:
    """Solve ``Pi(h) = P_c`` for one equilibrium branch with native Brent.

    The equation is solved in ``s = ln h`` with the dimensionless residual
    ``Pi(e^s) / P_c - 1``. The bracket selects the branch (for example the
    common black film on the double-layer branch or the Newton black film on
    the short-range branch). A bracket without a sign change reports
    ``NO_ROOT_IN_BRACKET``; it is never replaced by a nearby thickness.
    """
    if not isinstance(law, AbstractDisjoiningPressure):
        raise TypeError("law must be an AbstractDisjoiningPressure.")
    if not isinstance(thickness_bracket_m, tuple) or len(thickness_bracket_m) != 2:
        raise ValueError("thickness_bracket_m must be a (lower, upper) tuple.")
    lower, upper = (float(value) for value in thickness_bracket_m)
    if not (isfinite(lower) and isfinite(upper) and 0.0 < lower < upper):
        raise ValueError("thickness_bracket_m must satisfy 0 < lower < upper.")
    pressure = jnp.asarray(capillary_pressure_pa, dtype=jnp.float64)
    if pressure.shape != ():
        raise ValueError("capillary_pressure_pa must be a scalar.")
    pressure = eqx.error_if(
        pressure,
        ~jnp.isfinite(pressure) | (pressure <= 0.0),
        "capillary_pressure_pa must be finite and positive.",
    )
    termination_ = (
        NonlinearTermination(
            absolute_residual=1e-12,
            relative_residual=0.0,
            absolute_step=0.0,
            relative_step=1e-14,
            maximum_steps=200,
        )
        if termination is None
        else termination
    )
    if not isinstance(termination_, NonlinearTermination):
        raise TypeError("termination must be a NonlinearTermination or None.")

    def residual(log_thickness: Array, suction: Array) -> Array:
        return law.pressure(jnp.exp(log_thickness)) / suction - 1.0

    problem = ScalarRootProblem(
        residual,
        bracket=(np.log(lower), np.log(upper)),
        problem_id="black-film-equilibrium",
    )
    result = scalar_root(problem, method=Brent(), termination=termination_, args=pressure)
    thickness = jnp.exp(result.root)
    status = jnp.where(
        ~result.bracket_valid,
        BlackFilmStatus.NO_ROOT_IN_BRACKET,
        jnp.where(
            result.successful, BlackFilmStatus.CONVERGED, BlackFilmStatus.NOT_CONVERGED
        ),
    )
    slope = law.derivative(thickness)
    return BlackFilmEquilibriumResult(
        thickness_m=thickness,
        disjoining_slope_pa_m=slope,
        relative_residual=result.value,
        stable=slope < 0.0,
        status=status,
        nonlinear_status=result.status,
        iterations=result.nonlinear_result.diagnostics.iterations,
        law_id=law.law_id,
    )


__all__ = [
    "AbstractDisjoiningPressure",
    "BlackFilmEquilibriumResult",
    "BlackFilmStatus",
    "CompositeDisjoiningPressure",
    "DoubleLayerDisjoiningPressure",
    "ShortRangeRepulsionPressure",
    "VanDerWaalsDisjoiningPressure",
    "black_film_equilibrium",
]
