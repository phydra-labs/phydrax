#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Tree-level QED scattering amplitudes and analytic references."""

from __future__ import annotations

import abc
import math
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._kinematics import mandelstam, minkowski_dot, Particle
from ._wavefunctions import (
    dirac_adjoint,
    dirac_u,
    dirac_v,
    slash,
    vector_current,
)


class ScatteringProcess(StrictModule, NonTrainableState):
    """Ordered external species and normalization contract for a process."""

    incoming: tuple[Particle, ...]
    outgoing: tuple[Particle, ...]
    symmetry_factor: Array
    name: str = eqx.field(static=True)
    perturbative_order: str = eqx.field(static=True)
    process_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        incoming: Sequence[Particle],
        outgoing: Sequence[Particle],
        /,
        *,
        symmetry_factor: float = 1.0,
        perturbative_order: str = "tree",
    ):
        incoming_ = tuple(incoming)
        outgoing_ = tuple(outgoing)
        factor = float(symmetry_factor)
        if not name or not perturbative_order or not incoming_ or not outgoing_:
            raise ValueError("Scattering process declarations must be nonempty.")
        if not all(isinstance(particle, Particle) for particle in incoming_ + outgoing_):
            raise TypeError("Scattering process legs must be Particle instances.")
        if not math.isfinite(factor) or factor <= 0.0 or factor > 1.0:
            raise ValueError("symmetry_factor must lie in (0, 1].")
        self.incoming = incoming_
        self.outgoing = outgoing_
        self.symmetry_factor = jnp.asarray(factor)
        self.name = str(name)
        self.perturbative_order = str(perturbative_order)
        self.process_id = canonical_fingerprint(
            {
                "kind": "scattering-process",
                "name": name,
                "incoming": [particle.particle_id for particle in incoming_],
                "outgoing": [particle.particle_id for particle in outgoing_],
                "symmetry_factor": factor,
                "perturbative_order": perturbative_order,
            }
        )


class AbstractScatteringAmplitude(StrictModule):
    """Callable contract binding an amplitude implementation to one process."""

    __strict_abstract__ = True

    @property
    @abc.abstractmethod
    def process(self) -> ScatteringProcess:
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(
        self,
        momenta: Sequence[ArrayLike],
        spins: Sequence[int],
        polarizations: Sequence[ArrayLike],
        /,
    ) -> Array:
        raise NotImplementedError


class AmplitudeEvaluation(StrictModule):
    """Complex amplitude plus finite and momentum-conservation evidence."""

    value: Array
    finite: Array
    conservation_residual: Array
    process_id: str = eqx.field(static=True)


def evaluate_amplitude(
    process: ScatteringProcess,
    value: ArrayLike,
    incoming_momenta: Sequence[ArrayLike],
    outgoing_momenta: Sequence[ArrayLike],
    /,
) -> AmplitudeEvaluation:
    """Bind a computed amplitude to process-level conservation evidence."""
    if len(incoming_momenta) != len(process.incoming) or len(outgoing_momenta) != len(
        process.outgoing
    ):
        raise ValueError("Momentum lists must align with process legs.")
    amplitude = jnp.asarray(value)
    incoming_total = sum(
        (jnp.asarray(momentum) for momentum in incoming_momenta), jnp.zeros((4,))
    )
    outgoing_total = sum(
        (jnp.asarray(momentum) for momentum in outgoing_momenta), jnp.zeros((4,))
    )
    residual = jnp.max(jnp.abs(incoming_total - outgoing_total))
    return AmplitudeEvaluation(
        amplitude,
        jnp.all(jnp.isfinite(amplitude)),
        residual,
        process.process_id,
    )


def qed_coupling(alpha: ArrayLike, /) -> Array:
    """Return the positive electric coupling ``sqrt(4 pi alpha)``."""
    alpha_ = jnp.asarray(alpha)
    return jnp.sqrt(4.0 * jnp.pi * alpha_)


def _current_contraction(left: Array, right: Array, /) -> Array:
    return minkowski_dot(left, right)


def electron_muon_annihilation_amplitude(
    electron: ArrayLike,
    positron: ArrayLike,
    muon: ArrayLike,
    antimuon: ArrayLike,
    spins: tuple[int, int, int, int],
    /,
    *,
    electron_mass: float,
    muon_mass: float,
    alpha: float = 1.0 / 137.035999084,
) -> Array:
    """Tree amplitude for ``e- e+ -> mu- mu+`` in a spin-z basis."""
    p1, p2, p3, p4 = map(jnp.asarray, (electron, positron, muon, antimuon))
    u1 = dirac_u(p1, electron_mass, spins[0])
    v2 = dirac_v(p2, electron_mass, spins[1])
    u3 = dirac_u(p3, muon_mass, spins[2])
    v4 = dirac_v(p4, muon_mass, spins[3])
    s = minkowski_dot(p1 + p2, p1 + p2)
    incoming_current = vector_current(v2, u1)
    outgoing_current = vector_current(u3, v4)
    return (
        qed_coupling(alpha) ** 2
        * _current_contraction(incoming_current, outgoing_current)
        / s
    )


def compton_amplitude(
    electron_in: ArrayLike,
    photon_in: ArrayLike,
    electron_out: ArrayLike,
    photon_out: ArrayLike,
    electron_spins: tuple[int, int],
    polarization_in: ArrayLike,
    polarization_out: ArrayLike,
    /,
    *,
    electron_mass: float,
    alpha: float = 1.0 / 137.035999084,
) -> Array:
    """Gauge-invariant sum of the s- and u-channel Compton diagrams."""
    p, k, p_prime, k_prime = map(
        jnp.asarray, (electron_in, photon_in, electron_out, photon_out)
    )
    epsilon = jnp.asarray(polarization_in)
    epsilon_prime = jnp.conj(jnp.asarray(polarization_out))
    incoming = dirac_u(p, electron_mass, electron_spins[0])
    outgoing = dirac_u(p_prime, electron_mass, electron_spins[1])
    s_momentum = p + k
    u_momentum = p - k_prime
    s_denominator = minkowski_dot(s_momentum, s_momentum) - electron_mass**2
    u_denominator = minkowski_dot(u_momentum, u_momentum) - electron_mass**2
    matrix = (
        slash(epsilon_prime)
        @ (slash(s_momentum) + electron_mass * jnp.eye(4))
        @ slash(epsilon)
        / s_denominator
        + slash(epsilon)
        @ (slash(u_momentum) + electron_mass * jnp.eye(4))
        @ slash(epsilon_prime)
        / u_denominator
    )
    return -(qed_coupling(alpha) ** 2) * (dirac_adjoint(outgoing) @ matrix @ incoming)


def moller_amplitude(
    electron_one_in: ArrayLike,
    electron_two_in: ArrayLike,
    electron_one_out: ArrayLike,
    electron_two_out: ArrayLike,
    spins: tuple[int, int, int, int],
    /,
    *,
    electron_mass: float,
    alpha: float = 1.0 / 137.035999084,
) -> Array:
    """Antisymmetrized t/u-channel ``e- e- -> e- e-`` amplitude."""
    p1, p2, p3, p4 = map(
        jnp.asarray,
        (electron_one_in, electron_two_in, electron_one_out, electron_two_out),
    )
    u1, u2 = dirac_u(p1, electron_mass, spins[0]), dirac_u(p2, electron_mass, spins[1])
    u3, u4 = dirac_u(p3, electron_mass, spins[2]), dirac_u(p4, electron_mass, spins[3])
    _, t, u = mandelstam(p1, p2, p3, p4)
    direct = _current_contraction(vector_current(u3, u1), vector_current(u4, u2)) / t
    exchange = _current_contraction(vector_current(u4, u1), vector_current(u3, u2)) / u
    return qed_coupling(alpha) ** 2 * (direct - exchange)


def bhabha_amplitude(
    electron_in: ArrayLike,
    positron_in: ArrayLike,
    electron_out: ArrayLike,
    positron_out: ArrayLike,
    spins: tuple[int, int, int, int],
    /,
    *,
    electron_mass: float,
    alpha: float = 1.0 / 137.035999084,
) -> Array:
    """Crossing-consistent s/t-channel ``e- e+ -> e- e+`` amplitude."""
    p1, p2, p3, p4 = map(
        jnp.asarray, (electron_in, positron_in, electron_out, positron_out)
    )
    u1, v2 = dirac_u(p1, electron_mass, spins[0]), dirac_v(p2, electron_mass, spins[1])
    u3, v4 = dirac_u(p3, electron_mass, spins[2]), dirac_v(p4, electron_mass, spins[3])
    s, t, _ = mandelstam(p1, p2, p3, p4)
    t_channel = _current_contraction(vector_current(u3, u1), vector_current(v2, v4)) / t
    s_channel = _current_contraction(vector_current(v2, u1), vector_current(u3, v4)) / s
    return qed_coupling(alpha) ** 2 * (t_channel - s_channel)


def two_photon_annihilation_amplitude(
    electron: ArrayLike,
    positron: ArrayLike,
    photon_one: ArrayLike,
    photon_two: ArrayLike,
    electron_spins: tuple[int, int],
    polarization_one: ArrayLike,
    polarization_two: ArrayLike,
    /,
    *,
    electron_mass: float,
    alpha: float = 1.0 / 137.035999084,
) -> Array:
    """Tree amplitude for ``e- e+ -> gamma gamma`` including both orderings."""
    p, q, k1, k2 = map(jnp.asarray, (electron, positron, photon_one, photon_two))
    epsilon_one = jnp.conj(jnp.asarray(polarization_one))
    epsilon_two = jnp.conj(jnp.asarray(polarization_two))
    incoming = dirac_u(p, electron_mass, electron_spins[0])
    anti = dirac_v(q, electron_mass, electron_spins[1])
    intermediate_one = p - k1
    intermediate_two = p - k2
    denominator_one = minkowski_dot(intermediate_one, intermediate_one) - electron_mass**2
    denominator_two = minkowski_dot(intermediate_two, intermediate_two) - electron_mass**2
    matrix = (
        slash(epsilon_two)
        @ (slash(intermediate_one) + electron_mass * jnp.eye(4))
        @ slash(epsilon_one)
        / denominator_one
        + slash(epsilon_one)
        @ (slash(intermediate_two) + electron_mass * jnp.eye(4))
        @ slash(epsilon_two)
        / denominator_two
    )
    return qed_coupling(alpha) ** 2 * (dirac_adjoint(anti) @ matrix @ incoming)


def breit_wheeler_amplitude(
    photon_one: ArrayLike,
    photon_two: ArrayLike,
    electron: ArrayLike,
    positron: ArrayLike,
    electron_spins: tuple[int, int],
    polarization_one: ArrayLike,
    polarization_two: ArrayLike,
    /,
    *,
    electron_mass: float,
    alpha: float = 1.0 / 137.035999084,
) -> Array:
    """Tree amplitude for Breit-Wheeler pair production ``gamma gamma -> e- e+``."""
    k1, k2, p, q = map(jnp.asarray, (photon_one, photon_two, electron, positron))
    epsilon_one = jnp.asarray(polarization_one)
    epsilon_two = jnp.asarray(polarization_two)
    outgoing = dirac_u(p, electron_mass, electron_spins[0])
    anti = dirac_v(q, electron_mass, electron_spins[1])
    intermediate_one = p - k1
    intermediate_two = p - k2
    denominator_one = minkowski_dot(intermediate_one, intermediate_one) - electron_mass**2
    denominator_two = minkowski_dot(intermediate_two, intermediate_two) - electron_mass**2
    matrix = (
        slash(epsilon_one)
        @ (slash(intermediate_one) + electron_mass * jnp.eye(4))
        @ slash(epsilon_two)
        / denominator_one
        + slash(epsilon_two)
        @ (slash(intermediate_two) + electron_mass * jnp.eye(4))
        @ slash(epsilon_one)
        / denominator_two
    )
    return qed_coupling(alpha) ** 2 * (dirac_adjoint(outgoing) @ matrix @ anti)


def electron_muon_differential_cross_section(
    s: ArrayLike,
    cosine: ArrayLike,
    /,
    *,
    alpha: float = 1.0 / 137.035999084,
) -> Array:
    """Massless unpolarized ``d sigma / d Omega`` for lepton annihilation."""
    return alpha**2 * (1.0 + jnp.asarray(cosine) ** 2) / (4.0 * jnp.asarray(s))


def electron_muon_total_cross_section(
    s: ArrayLike,
    /,
    *,
    muon_mass: float = 0.0,
    alpha: float = 1.0 / 137.035999084,
) -> Array:
    """Unpolarized total cross section with the exact final muon mass."""
    s_ = jnp.asarray(s)
    beta = jnp.sqrt(jnp.maximum(1.0 - 4.0 * muon_mass**2 / s_, 0.0))
    return 4.0 * jnp.pi * alpha**2 * beta * (1.0 + 2.0 * muon_mass**2 / s_) / (3.0 * s_)


def klein_nishina_differential_cross_section(
    photon_energy: ArrayLike,
    cosine: ArrayLike,
    /,
    *,
    electron_mass: float,
    alpha: float = 1.0 / 137.035999084,
) -> Array:
    """Electron-rest-frame Compton ``d sigma / d Omega``."""
    energy = jnp.asarray(photon_energy)
    cosine_ = jnp.asarray(cosine)
    ratio = 1.0 / (1.0 + energy * (1.0 - cosine_) / electron_mass)
    return (
        alpha**2
        / (2.0 * electron_mass**2)
        * ratio**2
        * (ratio + 1.0 / ratio - (1.0 - cosine_**2))
    )


def bhabha_differential_cross_section(
    s: ArrayLike,
    cosine: ArrayLike,
    /,
    *,
    alpha: float = 1.0 / 137.035999084,
) -> Array:
    """Ultrarelativistic unpolarized Bhabha ``d sigma / d Omega``."""
    cosine_ = jnp.asarray(cosine)
    angular = (
        1.0
        + cosine_**2
        + (1.0 + cosine_) ** 2
        + 4.0 * (1.0 + cosine_) ** 2 / (1.0 - cosine_) ** 2
    )
    return alpha**2 * angular / (4.0 * jnp.asarray(s))


def moller_differential_cross_section(
    s: ArrayLike,
    cosine: ArrayLike,
    /,
    *,
    alpha: float = 1.0 / 137.035999084,
) -> Array:
    """Ultrarelativistic unpolarized Møller ``d sigma / d Omega``."""
    cosine_ = jnp.asarray(cosine)
    return (
        alpha**2
        * (3.0 + cosine_**2) ** 2
        / (2.0 * jnp.asarray(s) * (1.0 - cosine_**2) ** 2)
    )


def two_photon_annihilation_differential_cross_section(
    s: ArrayLike,
    cosine: ArrayLike,
    /,
    *,
    alpha: float = 1.0 / 137.035999084,
) -> Array:
    """Massless ``e+ e- -> gamma gamma`` differential reference."""
    cosine_ = jnp.asarray(cosine)
    return alpha**2 * (1.0 + cosine_**2) / (jnp.asarray(s) * (1.0 - cosine_**2))


def breit_wheeler_total_cross_section(
    s: ArrayLike,
    /,
    *,
    electron_mass: float,
    alpha: float = 1.0 / 137.035999084,
) -> Array:
    """Exact unpolarized Breit-Wheeler total cross section above threshold."""
    s_ = jnp.asarray(s)
    beta = jnp.sqrt(jnp.maximum(1.0 - 4.0 * electron_mass**2 / s_, 0.0))
    logarithm = jnp.log(
        (1.0 + beta) / jnp.maximum(1.0 - beta, jnp.finfo(beta.dtype).tiny)
    )
    return (
        jnp.pi
        * alpha**2
        / electron_mass**2
        * (1.0 - beta**2)
        * (0.5 * (3.0 - beta**4) * logarithm - beta * (2.0 - beta**2))
    )


__all__ = [
    "AbstractScatteringAmplitude",
    "AmplitudeEvaluation",
    "ScatteringProcess",
    "bhabha_differential_cross_section",
    "bhabha_amplitude",
    "breit_wheeler_amplitude",
    "breit_wheeler_total_cross_section",
    "compton_amplitude",
    "electron_muon_annihilation_amplitude",
    "electron_muon_differential_cross_section",
    "electron_muon_total_cross_section",
    "evaluate_amplitude",
    "klein_nishina_differential_cross_section",
    "moller_amplitude",
    "moller_differential_cross_section",
    "qed_coupling",
    "two_photon_annihilation_amplitude",
    "two_photon_annihilation_differential_cross_section",
]
