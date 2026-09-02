#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Energy-derived advanced terms for fixed-capacity polarizable force fields."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._polarization import (
    PermanentMultipoleSiteData,
    PolarizationPlan,
    PolarizationPredictorState,
    PolarizationScaleData,
    PolarizationSolveResult,
    prepared_polarization_energy,
    PreparedPolarizationSolver,
)


def _name(value, /) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError("Term name must be non-empty.")
    return result


def _positive(value, name, /) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _site_vector(value, name, /, *, nonnegative=False, positive=False):
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or result.size == 0 or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be a finite non-empty rank-one array.")
    if nonnegative and np.any(result < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    if positive and np.any(result <= 0.0):
        raise ValueError(f"{name} must be positive.")
    return result


def _pair_scale(value, capacity, /):
    if value is None:
        result = np.ones((capacity, capacity), dtype=float) - np.eye(capacity)
    else:
        result = np.asarray(value, dtype=float)
    if (
        result.shape != (capacity, capacity)
        or np.any(~np.isfinite(result))
        or np.any(result < 0.0)
        or np.any(result > 1.0)
        or not np.allclose(result, result.T)
        or not np.allclose(np.diag(result), 0.0)
    ):
        raise ValueError(
            "pair_scale must be finite, symmetric, in [0,1], and zero-diagonal."
        )
    return result


def _routes(value, width, capacity, name, /):
    result = np.asarray(value)
    if result.ndim != 2 or result.shape[1] != width or result.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer array with shape (R,{width}).")
    result = result.astype(np.int32, copy=False)
    if np.any(result < 0) or np.any(result >= capacity):
        raise ValueError(f"{name} contains an out-of-capacity site index.")
    if result.shape[0] and np.any(
        np.apply_along_axis(lambda route: np.unique(route).size != width, 1, result)
    ):
        raise ValueError(f"Every {name} route must contain distinct sites.")
    return result


def _positions(value, capacity, /):
    positions = jnp.asarray(value)
    if positions.shape != (capacity, 3):
        raise ValueError("positions must have fixed shape (N,3).")
    if not jnp.issubdtype(positions.dtype, jnp.floating):
        positions = positions.astype(jnp.float32)
    return positions


def _pair_geometry(positions, pair_scale, /):
    displacement = positions[:, None, :] - positions[None, :, :]
    squared = jnp.sum(displacement * displacement, axis=-1)
    upper = jnp.triu(pair_scale, k=1)
    participating = upper > 0.0
    positive = squared > 0.0
    distance = jnp.sqrt(jnp.where(participating & positive, squared, 1.0))
    valid = jnp.all(jnp.isfinite(positions)) & jnp.all(
        jnp.where(participating, positive, True)
    )
    return distance, upper, participating, valid


def _angle(a, b, /):
    norm_a = jnp.sqrt(jnp.sum(a * a, axis=-1))
    norm_b = jnp.sqrt(jnp.sum(b * b, axis=-1))
    valid = (norm_a > 0.0) & (norm_b > 0.0)
    denominator = jnp.where(valid, norm_a * norm_b, 1.0)
    cosine = jnp.clip(jnp.sum(a * b, axis=-1) / denominator, -1.0, 1.0)
    return jnp.arccos(cosine), valid


def _route_id(kind, arrays, parameters, /):
    return canonical_fingerprint(
        {
            "kind": kind,
            "arrays": array_tree_fingerprint(arrays),
            "parameters": parameters,
        }
    )


class Buffered147Potential(StrictModule, NonTrainableState):
    """Buffered 14-7 van der Waals energy with arithmetic radii mixing."""

    radii: Array
    epsilon: Array
    pair_scale: Array
    delta: float = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    site_capacity: int = eqx.field(static=True)
    name: str = eqx.field(static=True)
    term_id: str = eqx.field(static=True)

    def __init__(
        self,
        radii: ArrayLike,
        epsilon: ArrayLike,
        /,
        *,
        pair_scale: ArrayLike | None = None,
        delta: float = 0.07,
        gamma: float = 0.12,
        name: str = "buffered-14-7",
    ):
        radius = _site_vector(radii, "radii", positive=True)
        depth = _site_vector(epsilon, "epsilon", nonnegative=True)
        if depth.shape != radius.shape:
            raise ValueError("radii and epsilon must have equal capacity.")
        scale = _pair_scale(pair_scale, radius.size)
        delta_, gamma_ = _positive(delta, "delta"), _positive(gamma, "gamma")
        self.radii, self.epsilon, self.pair_scale = (
            jnp.asarray(radius),
            jnp.asarray(depth),
            jnp.asarray(scale),
        )
        self.delta, self.gamma = delta_, gamma_
        self.site_capacity, self.name = radius.size, _name(name)
        self.term_id = _route_id(
            "buffered-14-7",
            {"radii": radius, "epsilon": depth, "scale": scale},
            {"delta": delta_.hex(), "gamma": gamma_.hex()},
        )

    def energy(self, positions: ArrayLike, /):
        coordinate = _positions(positions, self.site_capacity)
        distance, scale, participating, valid = _pair_geometry(
            coordinate, self.pair_scale
        )
        equilibrium = self.radii[:, None] + self.radii[None, :]
        rho = distance / equilibrium
        depth = jnp.sqrt(self.epsilon[:, None] * self.epsilon[None, :])
        buffered = ((1.0 + self.delta) / (rho + self.delta)) ** 7
        attraction = (1.0 + self.gamma) / (rho**7 + self.gamma) - 2.0
        pair = scale * depth * buffered * attraction
        energy = jnp.sum(jnp.where(participating, pair, 0.0))
        successful = valid & jnp.isfinite(energy)
        return jnp.where(successful, energy, jnp.nan)


class ChargePenetrationPotential(StrictModule, NonTrainableState):
    """Core/valence electrostatics with finite-size valence damping."""

    core_charges: Array
    valence_charges: Array
    exponents: Array
    pair_scale: Array
    coulomb_constant: float = eqx.field(static=True)
    site_capacity: int = eqx.field(static=True)
    name: str = eqx.field(static=True)
    term_id: str = eqx.field(static=True)

    def __init__(
        self,
        core_charges: ArrayLike,
        valence_charges: ArrayLike,
        exponents: ArrayLike,
        /,
        *,
        coulomb_constant: float = 1.0,
        pair_scale: ArrayLike | None = None,
        name: str = "charge-penetration",
    ):
        core = _site_vector(core_charges, "core_charges")
        valence = _site_vector(valence_charges, "valence_charges")
        exponent = _site_vector(exponents, "exponents", positive=True)
        if valence.shape != core.shape or exponent.shape != core.shape:
            raise ValueError("Charge-penetration arrays must have equal capacity.")
        scale = _pair_scale(pair_scale, core.size)
        constant = _positive(coulomb_constant, "coulomb_constant")
        self.core_charges, self.valence_charges, self.exponents, self.pair_scale = (
            jnp.asarray(core),
            jnp.asarray(valence),
            jnp.asarray(exponent),
            jnp.asarray(scale),
        )
        self.coulomb_constant, self.site_capacity, self.name = (
            constant,
            core.size,
            _name(name),
        )
        self.term_id = _route_id(
            "charge-penetration",
            {"core": core, "valence": valence, "exponents": exponent, "scale": scale},
            {"coulomb_constant": constant.hex()},
        )

    def energy(self, positions: ArrayLike, /):
        coordinate = _positions(positions, self.site_capacity)
        distance, scale, participating, valid = _pair_geometry(
            coordinate, self.pair_scale
        )
        damping = -jnp.expm1(-self.exponents[:, None] * distance)
        core_i, core_j = self.core_charges[:, None], self.core_charges[None, :]
        valence_i, valence_j = (
            self.valence_charges[:, None],
            self.valence_charges[None, :],
        )
        damping_i, damping_j = damping, damping.T
        numerator = (
            core_i * core_j
            + valence_i * damping_i * core_j
            + core_i * valence_j * damping_j
            + valence_i * valence_j * damping_i * damping_j
        )
        pair = self.coulomb_constant * scale * numerator / distance
        energy = jnp.sum(jnp.where(participating, pair, 0.0))
        successful = valid & jnp.isfinite(energy)
        return jnp.where(successful, energy, jnp.nan)


class ChargeTransferPotential(StrictModule, NonTrainableState):
    """Short-range attractive charge-transfer energy."""

    amplitudes: Array
    exponents: Array
    pair_scale: Array
    site_capacity: int = eqx.field(static=True)
    name: str = eqx.field(static=True)
    term_id: str = eqx.field(static=True)

    def __init__(
        self,
        amplitudes: ArrayLike,
        exponents: ArrayLike,
        /,
        *,
        pair_scale: ArrayLike | None = None,
        name: str = "charge-transfer",
    ):
        amplitude = _site_vector(amplitudes, "amplitudes", nonnegative=True)
        exponent = _site_vector(exponents, "exponents", positive=True)
        if exponent.shape != amplitude.shape:
            raise ValueError("Charge-transfer arrays must have equal capacity.")
        scale = _pair_scale(pair_scale, amplitude.size)
        self.amplitudes, self.exponents, self.pair_scale = (
            jnp.asarray(amplitude),
            jnp.asarray(exponent),
            jnp.asarray(scale),
        )
        self.site_capacity, self.name = amplitude.size, _name(name)
        self.term_id = _route_id(
            "charge-transfer",
            {"amplitudes": amplitude, "exponents": exponent, "scale": scale},
            {},
        )

    def energy(self, positions: ArrayLike, /):
        coordinate = _positions(positions, self.site_capacity)
        distance, scale, participating, valid = _pair_geometry(
            coordinate, self.pair_scale
        )
        amplitude = jnp.sqrt(self.amplitudes[:, None] * self.amplitudes[None, :])
        exponent = 0.5 * (self.exponents[:, None] + self.exponents[None, :])
        pair = -scale * amplitude * jnp.exp(-exponent * distance)
        energy = jnp.sum(jnp.where(participating, pair, 0.0))
        successful = valid & jnp.isfinite(energy)
        return jnp.where(successful, energy, jnp.nan)


def _tang_toennies(order, argument, /):
    return jsp.gammainc(jnp.asarray(float(order + 1), dtype=argument.dtype), argument)


def _damped_inverse_power(order, argument, distance, /):
    inverse_length = argument / distance
    term = jnp.ones_like(argument) / float(order + 1)
    series = term
    for index in range(1, 13):
        term = (
            term
            * (-argument)
            / float(index)
            * float(order + index)
            / float(order + index + 1)
        )
        series = series + term
    small = (
        inverse_length ** (order + 1) * distance / float(math.factorial(order)) * series
    )
    regular = _tang_toennies(order, argument) / distance**order
    return jnp.where(argument < 0.5, small, regular)


class DampedDispersionPotential(StrictModule, NonTrainableState):
    """Tang--Toennies-damped C6/C8/C10 dispersion energy."""

    c6: Array
    c8: Array
    c10: Array
    damping: Array
    pair_scale: Array
    site_capacity: int = eqx.field(static=True)
    name: str = eqx.field(static=True)
    term_id: str = eqx.field(static=True)

    def __init__(
        self,
        c6: ArrayLike,
        damping: ArrayLike,
        /,
        *,
        c8: ArrayLike | None = None,
        c10: ArrayLike | None = None,
        pair_scale: ArrayLike | None = None,
        name: str = "damped-dispersion",
    ):
        c6_ = _site_vector(c6, "c6", nonnegative=True)
        c8_ = (
            np.zeros_like(c6_) if c8 is None else _site_vector(c8, "c8", nonnegative=True)
        )
        c10_ = (
            np.zeros_like(c6_)
            if c10 is None
            else _site_vector(c10, "c10", nonnegative=True)
        )
        damping_ = _site_vector(damping, "damping", positive=True)
        if (
            c8_.shape != c6_.shape
            or c10_.shape != c6_.shape
            or damping_.shape != c6_.shape
        ):
            raise ValueError("Dispersion arrays must have equal capacity.")
        if not np.any(c6_ > 0.0) and not np.any(c8_ > 0.0) and not np.any(c10_ > 0.0):
            raise ValueError("At least one dispersion coefficient must be positive.")
        scale = _pair_scale(pair_scale, c6_.size)
        self.c6, self.c8, self.c10, self.damping, self.pair_scale = (
            jnp.asarray(c6_),
            jnp.asarray(c8_),
            jnp.asarray(c10_),
            jnp.asarray(damping_),
            jnp.asarray(scale),
        )
        self.site_capacity, self.name = c6_.size, _name(name)
        self.term_id = _route_id(
            "damped-dispersion",
            {"c6": c6_, "c8": c8_, "c10": c10_, "damping": damping_, "scale": scale},
            {},
        )

    def energy(self, positions: ArrayLike, /):
        coordinate = _positions(positions, self.site_capacity)
        distance, scale, participating, valid = _pair_geometry(
            coordinate, self.pair_scale
        )
        argument = 0.5 * (self.damping[:, None] + self.damping[None, :]) * distance
        c6 = jnp.sqrt(self.c6[:, None] * self.c6[None, :])
        c8 = jnp.sqrt(self.c8[:, None] * self.c8[None, :])
        c10 = jnp.sqrt(self.c10[:, None] * self.c10[None, :])
        pair = -scale * (
            c6 * _damped_inverse_power(6, argument, distance)
            + c8 * _damped_inverse_power(8, argument, distance)
            + c10 * _damped_inverse_power(10, argument, distance)
        )
        energy = jnp.sum(jnp.where(participating, pair, 0.0))
        successful = valid & jnp.isfinite(energy)
        return jnp.where(successful, energy, jnp.nan)


class PauliRepulsionPotential(StrictModule, NonTrainableState):
    """Isotropic Born--Mayer Pauli exchange repulsion."""

    amplitudes: Array
    exponents: Array
    pair_scale: Array
    site_capacity: int = eqx.field(static=True)
    name: str = eqx.field(static=True)
    term_id: str = eqx.field(static=True)

    def __init__(
        self,
        amplitudes: ArrayLike,
        exponents: ArrayLike,
        /,
        *,
        pair_scale: ArrayLike | None = None,
        name: str = "pauli-repulsion",
    ):
        amplitude = _site_vector(amplitudes, "amplitudes", nonnegative=True)
        exponent = _site_vector(exponents, "exponents", positive=True)
        if exponent.shape != amplitude.shape:
            raise ValueError("Pauli arrays must have equal capacity.")
        scale = _pair_scale(pair_scale, amplitude.size)
        self.amplitudes, self.exponents, self.pair_scale = (
            jnp.asarray(amplitude),
            jnp.asarray(exponent),
            jnp.asarray(scale),
        )
        self.site_capacity, self.name = amplitude.size, _name(name)
        self.term_id = _route_id(
            "pauli-repulsion",
            {"amplitudes": amplitude, "exponents": exponent, "scale": scale},
            {},
        )

    def energy(self, positions: ArrayLike, /):
        coordinate = _positions(positions, self.site_capacity)
        distance, scale, participating, valid = _pair_geometry(
            coordinate, self.pair_scale
        )
        amplitude = jnp.sqrt(self.amplitudes[:, None] * self.amplitudes[None, :])
        exponent = 0.5 * (self.exponents[:, None] + self.exponents[None, :])
        pair = scale * amplitude * jnp.exp(-exponent * distance)
        energy = jnp.sum(jnp.where(participating, pair, 0.0))
        successful = valid & jnp.isfinite(energy)
        return jnp.where(successful, energy, jnp.nan)


class ChargeFluxPotential(StrictModule, NonTrainableState):
    """Conservative bond/angle charge flux coupled to Coulomb energy."""

    reference_charges: Array
    bond_routes: Array
    bond_coefficients: Array
    equilibrium_lengths: Array
    angle_routes: Array
    angle_coefficients: Array
    equilibrium_angles: Array
    pair_scale: Array
    coulomb_constant: float = eqx.field(static=True)
    site_capacity: int = eqx.field(static=True)
    name: str = eqx.field(static=True)
    term_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_charges: ArrayLike,
        bond_routes: ArrayLike,
        bond_coefficients: ArrayLike,
        equilibrium_lengths: ArrayLike,
        angle_routes: ArrayLike,
        angle_coefficients: ArrayLike,
        equilibrium_angles: ArrayLike,
        /,
        *,
        coulomb_constant: float = 1.0,
        pair_scale: ArrayLike | None = None,
        name: str = "charge-flux",
    ):
        charges = _site_vector(reference_charges, "reference_charges")
        capacity = charges.size
        bonds = _routes(bond_routes, 2, capacity, "bond_routes")
        bond_coefficient = np.asarray(bond_coefficients, dtype=float)
        bond_length = np.asarray(equilibrium_lengths, dtype=float)
        angles = _routes(angle_routes, 3, capacity, "angle_routes")
        angle_coefficient = np.asarray(angle_coefficients, dtype=float)
        angle_equilibrium = np.asarray(equilibrium_angles, dtype=float)
        if (
            bond_coefficient.shape != (bonds.shape[0],)
            or bond_length.shape != (bonds.shape[0],)
            or angle_coefficient.shape != (angles.shape[0], 2)
            or angle_equilibrium.shape != (angles.shape[0],)
            or np.any(~np.isfinite(bond_coefficient))
            or np.any(~np.isfinite(bond_length))
            or np.any(~np.isfinite(angle_coefficient))
            or np.any(~np.isfinite(angle_equilibrium))
            or np.any(bond_length <= 0.0)
            or np.any(angle_equilibrium <= 0.0)
            or np.any(angle_equilibrium >= np.pi)
        ):
            raise ValueError("Charge-flux route parameters are invalid.")
        if bonds.shape[0] + angles.shape[0] == 0:
            raise ValueError("Charge flux requires at least one bond or angle route.")
        scale = _pair_scale(pair_scale, capacity)
        constant = _positive(coulomb_constant, "coulomb_constant")
        (
            self.reference_charges,
            self.bond_routes,
            self.bond_coefficients,
            self.equilibrium_lengths,
            self.angle_routes,
            self.angle_coefficients,
            self.equilibrium_angles,
            self.pair_scale,
        ) = (
            jnp.asarray(charges),
            jnp.asarray(bonds),
            jnp.asarray(bond_coefficient),
            jnp.asarray(bond_length),
            jnp.asarray(angles),
            jnp.asarray(angle_coefficient),
            jnp.asarray(angle_equilibrium),
            jnp.asarray(scale),
        )
        self.coulomb_constant, self.site_capacity, self.name = (
            constant,
            capacity,
            _name(name),
        )
        self.term_id = _route_id(
            "charge-flux",
            {
                "charges": charges,
                "bond_routes": bonds,
                "bond_coefficients": bond_coefficient,
                "equilibrium_lengths": bond_length,
                "angle_routes": angles,
                "angle_coefficients": angle_coefficient,
                "equilibrium_angles": angle_equilibrium,
                "scale": scale,
            },
            {"coulomb_constant": constant.hex()},
        )

    def _charges_and_validity(self, positions, /):
        coordinate = _positions(positions, self.site_capacity)
        charges = self.reference_charges
        bond_vector = (
            coordinate[self.bond_routes[:, 1]] - coordinate[self.bond_routes[:, 0]]
        )
        bond_length = jnp.sqrt(jnp.sum(bond_vector * bond_vector, axis=-1))
        bond_delta = self.bond_coefficients * (bond_length - self.equilibrium_lengths)
        charges = charges.at[self.bond_routes[:, 0]].add(-bond_delta)
        charges = charges.at[self.bond_routes[:, 1]].add(bond_delta)
        left = coordinate[self.angle_routes[:, 0]] - coordinate[self.angle_routes[:, 1]]
        right = coordinate[self.angle_routes[:, 2]] - coordinate[self.angle_routes[:, 1]]
        angle, angle_valid = _angle(left, right)
        angle_delta = angle - self.equilibrium_angles
        left_delta = self.angle_coefficients[:, 0] * angle_delta
        right_delta = self.angle_coefficients[:, 1] * angle_delta
        charges = charges.at[self.angle_routes[:, 0]].add(left_delta)
        charges = charges.at[self.angle_routes[:, 2]].add(right_delta)
        charges = charges.at[self.angle_routes[:, 1]].add(-left_delta - right_delta)
        valid = (
            jnp.all(bond_length > 0.0)
            & jnp.all(angle_valid)
            & jnp.all(jnp.isfinite(charges))
        )
        return charges, valid

    def charges(self, positions: ArrayLike, /):
        coordinate = _positions(positions, self.site_capacity)
        charges, _ = self._charges_and_validity(coordinate)
        return charges

    def energy(self, positions: ArrayLike, /):
        coordinate = _positions(positions, self.site_capacity)
        distance, scale, participating, valid = _pair_geometry(
            coordinate, self.pair_scale
        )
        charges, flux_valid = self._charges_and_validity(coordinate)
        pair = (
            self.coulomb_constant * scale * charges[:, None] * charges[None, :] / distance
        )
        energy = jnp.sum(jnp.where(participating, pair, 0.0))
        charge_conserved = jnp.isclose(
            jnp.sum(charges), jnp.sum(self.reference_charges), rtol=1.0e-6, atol=1.0e-7
        )
        successful = (
            valid
            & flux_valid
            & charge_conserved
            & jnp.all(jnp.isfinite(charges))
            & jnp.isfinite(energy)
        )
        return jnp.where(successful, energy, jnp.nan)


class StretchBendPotential(StrictModule, NonTrainableState):
    """Three-site stretch--bend cross energy."""

    routes: Array
    stiffness: Array
    equilibrium_lengths: Array
    equilibrium_angles: Array
    site_capacity: int = eqx.field(static=True)
    name: str = eqx.field(static=True)
    term_id: str = eqx.field(static=True)

    def __init__(
        self,
        site_capacity: int,
        routes: ArrayLike,
        stiffness: ArrayLike,
        equilibrium_lengths: ArrayLike,
        equilibrium_angles: ArrayLike,
        /,
        *,
        name: str = "stretch-bend",
    ):
        capacity = int(site_capacity)
        if capacity <= 0:
            raise ValueError("site_capacity must be positive.")
        route = _routes(routes, 3, capacity, "routes")
        stiffness_ = np.asarray(stiffness, dtype=float)
        lengths = np.asarray(equilibrium_lengths, dtype=float)
        angles = np.asarray(equilibrium_angles, dtype=float)
        if (
            route.shape[0] == 0
            or stiffness_.shape != (route.shape[0], 2)
            or lengths.shape != (route.shape[0], 2)
            or angles.shape != (route.shape[0],)
            or np.any(~np.isfinite(stiffness_))
            or np.any(~np.isfinite(lengths))
            or np.any(~np.isfinite(angles))
            or np.any(lengths <= 0.0)
            or np.any(angles <= 0.0)
            or np.any(angles >= np.pi)
        ):
            raise ValueError("Stretch--bend route parameters are invalid.")
        (
            self.routes,
            self.stiffness,
            self.equilibrium_lengths,
            self.equilibrium_angles,
        ) = (
            jnp.asarray(route),
            jnp.asarray(stiffness_),
            jnp.asarray(lengths),
            jnp.asarray(angles),
        )
        self.site_capacity, self.name = capacity, _name(name)
        self.term_id = _route_id(
            "stretch-bend",
            {
                "routes": route,
                "stiffness": stiffness_,
                "lengths": lengths,
                "angles": angles,
            },
            {},
        )

    def energy(self, positions: ArrayLike, /):
        coordinate = _positions(positions, self.site_capacity)
        left = coordinate[self.routes[:, 0]] - coordinate[self.routes[:, 1]]
        right = coordinate[self.routes[:, 2]] - coordinate[self.routes[:, 1]]
        left_length = jnp.sqrt(jnp.sum(left * left, axis=-1))
        right_length = jnp.sqrt(jnp.sum(right * right, axis=-1))
        angle, angle_valid = _angle(left, right)
        stretch = jnp.stack(
            (
                left_length - self.equilibrium_lengths[:, 0],
                right_length - self.equilibrium_lengths[:, 1],
            ),
            axis=-1,
        )
        energy = jnp.sum(
            (angle - self.equilibrium_angles) * jnp.sum(self.stiffness * stretch, axis=-1)
        )
        successful = (
            jnp.all(angle_valid)
            & jnp.all(jnp.isfinite(coordinate))
            & jnp.isfinite(energy)
        )
        return jnp.where(successful, energy, jnp.nan)


class AngleAnglePotential(StrictModule, NonTrainableState):
    """Four-site angle--angle cross energy about one central site."""

    routes: Array
    stiffness: Array
    equilibrium_angles: Array
    site_capacity: int = eqx.field(static=True)
    name: str = eqx.field(static=True)
    term_id: str = eqx.field(static=True)

    def __init__(
        self,
        site_capacity: int,
        routes: ArrayLike,
        stiffness: ArrayLike,
        equilibrium_angles: ArrayLike,
        /,
        *,
        name: str = "angle-angle",
    ):
        capacity = int(site_capacity)
        if capacity <= 0:
            raise ValueError("site_capacity must be positive.")
        route = _routes(routes, 4, capacity, "routes")
        stiffness_ = np.asarray(stiffness, dtype=float)
        angles = np.asarray(equilibrium_angles, dtype=float)
        if (
            route.shape[0] == 0
            or stiffness_.shape != (route.shape[0],)
            or angles.shape != (route.shape[0], 2)
            or np.any(~np.isfinite(stiffness_))
            or np.any(~np.isfinite(angles))
            or np.any(angles <= 0.0)
            or np.any(angles >= np.pi)
        ):
            raise ValueError("Angle--angle route parameters are invalid.")
        self.routes, self.stiffness, self.equilibrium_angles = (
            jnp.asarray(route),
            jnp.asarray(stiffness_),
            jnp.asarray(angles),
        )
        self.site_capacity, self.name = capacity, _name(name)
        self.term_id = _route_id(
            "angle-angle",
            {"routes": route, "stiffness": stiffness_, "angles": angles},
            {},
        )

    def energy(self, positions: ArrayLike, /):
        coordinate = _positions(positions, self.site_capacity)
        center = coordinate[self.routes[:, 1]]
        first, first_valid = _angle(
            coordinate[self.routes[:, 0]] - center,
            coordinate[self.routes[:, 2]] - center,
        )
        second, second_valid = _angle(
            coordinate[self.routes[:, 0]] - center,
            coordinate[self.routes[:, 3]] - center,
        )
        energy = jnp.sum(
            self.stiffness
            * (first - self.equilibrium_angles[:, 0])
            * (second - self.equilibrium_angles[:, 1])
        )
        successful = (
            jnp.all(first_valid)
            & jnp.all(second_valid)
            & jnp.all(jnp.isfinite(coordinate))
            & jnp.isfinite(energy)
        )
        return jnp.where(successful, energy, jnp.nan)


class OutOfPlaneBendPotential(StrictModule, NonTrainableState):
    """Four-site signed out-of-plane bending energy."""

    routes: Array
    stiffness: Array
    target_angles: Array
    site_capacity: int = eqx.field(static=True)
    name: str = eqx.field(static=True)
    term_id: str = eqx.field(static=True)

    def __init__(
        self,
        site_capacity: int,
        routes: ArrayLike,
        stiffness: ArrayLike,
        /,
        *,
        target_angles: ArrayLike | None = None,
        name: str = "out-of-plane-bend",
    ):
        capacity = int(site_capacity)
        if capacity <= 0:
            raise ValueError("site_capacity must be positive.")
        route = _routes(routes, 4, capacity, "routes")
        stiffness_ = np.asarray(stiffness, dtype=float)
        targets = (
            np.zeros((route.shape[0],), dtype=float)
            if target_angles is None
            else np.asarray(target_angles, dtype=float)
        )
        if (
            route.shape[0] == 0
            or stiffness_.shape != (route.shape[0],)
            or targets.shape != (route.shape[0],)
            or np.any(~np.isfinite(stiffness_))
            or np.any(stiffness_ < 0.0)
            or np.any(~np.isfinite(targets))
            or np.any(np.abs(targets) >= 0.5 * np.pi)
        ):
            raise ValueError("Out-of-plane route parameters are invalid.")
        self.routes, self.stiffness, self.target_angles = (
            jnp.asarray(route),
            jnp.asarray(stiffness_),
            jnp.asarray(targets),
        )
        self.site_capacity, self.name = capacity, _name(name)
        self.term_id = _route_id(
            "out-of-plane-bend",
            {"routes": route, "stiffness": stiffness_, "targets": targets},
            {},
        )

    def energy(self, positions: ArrayLike, /):
        coordinate = _positions(positions, self.site_capacity)
        center = coordinate[self.routes[:, 1]]
        out = coordinate[self.routes[:, 0]] - center
        plane_left = coordinate[self.routes[:, 2]] - center
        plane_right = coordinate[self.routes[:, 3]] - center
        normal = jnp.cross(plane_left, plane_right)
        out_norm = jnp.sqrt(jnp.sum(out * out, axis=-1))
        normal_norm = jnp.sqrt(jnp.sum(normal * normal, axis=-1))
        valid = (out_norm > 0.0) & (normal_norm > 0.0)
        denominator = jnp.where(valid, out_norm * normal_norm, 1.0)
        sine = jnp.clip(jnp.sum(out * normal, axis=-1) / denominator, -1.0, 1.0)
        angle = jnp.arcsin(sine)
        energy = 0.5 * jnp.sum(self.stiffness * (angle - self.target_angles) ** 2)
        successful = (
            jnp.all(valid) & jnp.all(jnp.isfinite(coordinate)) & jnp.isfinite(energy)
        )
        return jnp.where(successful, energy, jnp.nan)


_POLARIZABLE_TERM_TYPES = (
    Buffered147Potential,
    ChargePenetrationPotential,
    ChargeTransferPotential,
    ChargeFluxPotential,
    DampedDispersionPotential,
    PauliRepulsionPotential,
    StretchBendPotential,
    AngleAnglePotential,
    OutOfPlaneBendPotential,
)


class PolarizableTermEvaluation(StrictModule):
    """Energy-derived force and affine-coordinate virial for one term."""

    energy: Array
    forces: Array
    virial: Array
    finite: Array
    successful: Array


def evaluate_polarizable_term(term, positions: ArrayLike, /) -> PolarizableTermEvaluation:
    """Differentiate any advanced scalar term and fail closed on nonfinite output."""
    if not isinstance(term, _POLARIZABLE_TERM_TYPES):
        raise TypeError("term is not a supported polarizable force-field term.")
    coordinate = _positions(positions, term.site_capacity)
    energy, gradient = jax.value_and_grad(term.energy)(coordinate)
    forces = -gradient
    virial = contract("ni,nj->ij", coordinate, forces)
    finite = (
        jnp.isfinite(energy)
        & jnp.all(jnp.isfinite(forces))
        & jnp.all(jnp.isfinite(virial))
    )
    return PolarizableTermEvaluation(
        jnp.where(finite, energy, jnp.nan),
        jnp.where(finite, forces, jnp.nan),
        jnp.where(finite, virial, jnp.nan),
        finite,
        finite,
    )


class PolarizableForceQualification(StrictModule):
    """Observable qualification gates for energy-derived forces and virials."""

    derivative_mode: str = eqx.field(static=True)
    virial_mode: str = eqx.field(static=True)
    energy_finite: Array
    forces_finite: Array
    virial_finite: Array
    term_energies_finite: Array
    force_balance_residual: Array
    polarization_force_valid: Array
    successful: Array


class PolarizableForceFieldPlan(StrictModule, NonTrainableState):
    """Composable plan for advanced scalar terms and optional polarization."""

    terms: tuple
    polarization: PolarizationPlan | None
    force_balance_tolerance: float = eqx.field(static=True)
    site_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        terms,
        /,
        *,
        polarization: PolarizationPlan | None = None,
        force_balance_tolerance: float = 1.0e-5,
    ):
        terms_ = tuple(terms)
        if any(not isinstance(term, _POLARIZABLE_TERM_TYPES) for term in terms_):
            raise TypeError("Every term must be an advanced polarizable energy term.")
        if polarization is not None and not isinstance(polarization, PolarizationPlan):
            raise TypeError("polarization must be PolarizationPlan or None.")
        if not terms_ and polarization is None:
            raise ValueError("A force-field plan requires a term or polarization.")
        capacities = tuple(term.site_capacity for term in terms_)
        if capacities and any(capacity != capacities[0] for capacity in capacities):
            raise ValueError("All force-field terms must have equal site capacity.")
        tolerance = _positive(force_balance_tolerance, "force_balance_tolerance")
        self.terms, self.polarization = terms_, polarization
        self.force_balance_tolerance = tolerance
        self.site_capacity = -1 if not capacities else capacities[0]
        self.plan_id = canonical_fingerprint(
            {
                "kind": "polarizable-force-field-plan",
                "terms": [term.term_id for term in terms_],
                "polarization": None if polarization is None else polarization.plan_id,
                "force_balance_tolerance": tolerance.hex(),
            }
        )

    def prepare(
        self,
        /,
        *,
        multipoles: PermanentMultipoleSiteData | None = None,
        scaling: PolarizationScaleData | None = None,
    ) -> PreparedPolarizableForceField:
        if self.polarization is None:
            if multipoles is not None or scaling is not None:
                raise ValueError("multipoles and scaling require a polarization plan.")
            polarization = None
            capacity = self.site_capacity
        else:
            if not isinstance(multipoles, PermanentMultipoleSiteData):
                raise TypeError(
                    "multipoles must be PermanentMultipoleSiteData when "
                    "polarization is enabled."
                )
            if self.site_capacity >= 0 and multipoles.site_capacity != self.site_capacity:
                raise ValueError("Term and multipole capacities differ.")
            polarization = self.polarization.prepare(multipoles, scaling=scaling)
            capacity = multipoles.site_capacity
        return PreparedPolarizableForceField(self, polarization, capacity)


class PreparedPolarizableForceField(StrictModule, NonTrainableState):
    """Prepared fixed-capacity advanced polarizable force-field runtime."""

    plan: PolarizableForceFieldPlan
    polarization: PreparedPolarizationSolver | None
    site_capacity: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan, polarization, site_capacity, /):
        capacity = int(site_capacity)
        if capacity <= 0:
            raise ValueError("Prepared force fields require positive site capacity.")
        self.plan, self.polarization, self.site_capacity = (
            plan,
            polarization,
            capacity,
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-polarizable-force-field",
                "plan": plan.plan_id,
                "polarization": (
                    None if polarization is None else polarization.prepared_id
                ),
                "site_capacity": capacity,
            }
        )

    def evaluate(
        self,
        positions: ArrayLike,
        /,
        *,
        predictor_state: PolarizationPredictorState | None = None,
        cell_vectors: ArrayLike | None = None,
    ) -> PolarizableForceFieldEvaluation:
        coordinate = _positions(positions, self.site_capacity)
        if self.polarization is None and predictor_state is not None:
            raise ValueError("predictor_state requires polarization.")
        if self.polarization is None and cell_vectors is not None:
            raise ValueError("cell_vectors require periodic polarization.")

        def total_energy(value):
            term_energies = (
                jnp.stack(tuple(term.energy(value) for term in self.plan.terms))
                if self.plan.terms
                else jnp.zeros((0,), dtype=value.dtype)
            )
            if self.polarization is None:
                polarization_energy = jnp.zeros((), dtype=value.dtype)
                result = None
            else:
                polarization_energy, result = prepared_polarization_energy(
                    self.polarization,
                    value,
                    predictor_state=predictor_state,
                    cell_vectors=cell_vectors,
                )
            return jnp.sum(term_energies) + polarization_energy, (
                term_energies,
                polarization_energy,
                result,
            )

        (energy, auxiliary), gradient = jax.value_and_grad(total_energy, has_aux=True)(
            coordinate
        )
        term_energies, polarization_energy, polarization_result = auxiliary
        forces = -gradient
        virial = contract("ni,nj->ij", coordinate, forces)
        energy_finite = jnp.isfinite(energy)
        forces_finite = jnp.all(jnp.isfinite(forces))
        virial_finite = jnp.all(jnp.isfinite(virial))
        terms_finite = jnp.all(jnp.isfinite(term_energies))
        force_balance = jnp.sqrt(jnp.sum(jnp.sum(forces, axis=0) ** 2))
        force_scale = jnp.maximum(
            jnp.sqrt(jnp.sum(forces * forces)), jnp.asarray(1.0, dtype=forces.dtype)
        )
        balance_valid = force_balance <= self.plan.force_balance_tolerance * force_scale
        polarization_valid = (
            jnp.asarray(True)
            if polarization_result is None
            else (
                polarization_result.state.successful
                & polarization_result.state.force_valid
                & polarization_result.operator.successful
            )
        )
        successful = (
            energy_finite
            & forces_finite
            & virial_finite
            & terms_finite
            & balance_valid
            & polarization_valid
        )
        qualification = PolarizableForceQualification(
            "energy-gradient",
            "affine-coordinate",
            energy_finite,
            forces_finite,
            virial_finite,
            terms_finite,
            force_balance,
            polarization_valid,
            successful,
        )
        return PolarizableForceFieldEvaluation(
            jnp.where(successful, energy, jnp.nan),
            jnp.where(successful, forces, jnp.nan),
            jnp.where(successful, virial, jnp.nan),
            jnp.where(successful, term_energies, jnp.nan),
            jnp.where(successful, polarization_energy, jnp.nan),
            polarization_result,
            (
                None
                if polarization_result is None
                else polarization_result.predictor_state
            ),
            qualification,
            successful,
        )


class PolarizableForceFieldEvaluation(StrictModule):
    """Fail-closed force-field energy, derivatives, state, and qualification."""

    energy: Array
    forces: Array
    virial: Array
    term_energies: Array
    polarization_energy: Array
    polarization_result: PolarizationSolveResult | None
    predictor_state: PolarizationPredictorState | None
    qualification: PolarizableForceQualification
    successful: Array


__all__ = [
    "AngleAnglePotential",
    "Buffered147Potential",
    "ChargeFluxPotential",
    "ChargePenetrationPotential",
    "ChargeTransferPotential",
    "DampedDispersionPotential",
    "OutOfPlaneBendPotential",
    "PauliRepulsionPotential",
    "PolarizableForceFieldEvaluation",
    "PolarizableForceFieldPlan",
    "PolarizableForceQualification",
    "PolarizableTermEvaluation",
    "PreparedPolarizableForceField",
    "StretchBendPotential",
    "evaluate_polarizable_term",
]
