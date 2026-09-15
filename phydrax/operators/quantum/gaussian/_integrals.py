#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Differentiable Cartesian Gaussian molecular integrals through arbitrary shells."""

from __future__ import annotations

from math import comb

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._basis import PreparedGaussianBasis
from ._boys import boys0, boys_values


def _hermite_coefficients(
    left_order: int,
    right_order: int,
    left_exponent: Array,
    right_exponent: Array,
    displacement: Array,
    /,
) -> tuple[Array, ...]:
    power = left_exponent + right_exponent
    reduced = left_exponent * right_exponent / power
    cache: dict[tuple[int, int, int], Array] = {}

    def coefficient(left: int, right: int, order: int) -> Array:
        key = (left, right, order)
        if key in cache:
            return cache[key]
        if order < 0 or order > left + right:
            result = jnp.asarray(0.0, dtype=power.dtype)
        elif left == 0 and right == 0 and order == 0:
            result = jnp.exp(-reduced * displacement * displacement)
        elif left > 0:
            result = (
                coefficient(left - 1, right, order - 1) / (2.0 * power)
                - reduced
                * displacement
                / left_exponent
                * coefficient(left - 1, right, order)
                + (order + 1) * coefficient(left - 1, right, order + 1)
            )
        else:
            result = (
                coefficient(left, right - 1, order - 1) / (2.0 * power)
                + reduced
                * displacement
                / right_exponent
                * coefficient(left, right - 1, order)
                + (order + 1) * coefficient(left, right - 1, order + 1)
            )
        cache[key] = result
        return result

    return tuple(
        coefficient(left_order, right_order, order)
        for order in range(left_order + right_order + 1)
    )


def _primitive_overlap(
    left_exponent: Array,
    right_exponent: Array,
    left_center: Array,
    right_center: Array,
    left_angular: tuple[int, int, int],
    right_angular: tuple[int, int, int],
    /,
) -> Array:
    power = left_exponent + right_exponent
    values = tuple(
        _hermite_coefficients(
            left_angular[axis],
            right_angular[axis],
            left_exponent,
            right_exponent,
            left_center[axis] - right_center[axis],
        )[0]
        for axis in range(3)
    )
    return values[0] * values[1] * values[2] * (jnp.pi / power) ** 1.5


def _primitive_multipole(
    left_exponent: Array,
    right_exponent: Array,
    left_center: Array,
    right_center: Array,
    left_angular: tuple[int, int, int],
    right_angular: tuple[int, int, int],
    powers: tuple[int, int, int],
    /,
) -> Array:
    power = left_exponent + right_exponent
    result = jnp.asarray(1.0, dtype=power.dtype)
    for axis in range(3):
        axis_value = jnp.asarray(0.0, dtype=power.dtype)
        for added in range(powers[axis] + 1):
            coefficient = comb(powers[axis], added) * right_center[axis] ** (
                powers[axis] - added
            )
            hermite = _hermite_coefficients(
                left_angular[axis],
                right_angular[axis] + added,
                left_exponent,
                right_exponent,
                left_center[axis] - right_center[axis],
            )[0]
            axis_value = axis_value + coefficient * hermite * jnp.sqrt(jnp.pi / power)
        result = result * axis_value
    return result


def _primitive_kinetic(
    left_exponent: Array,
    right_exponent: Array,
    left_center: Array,
    right_center: Array,
    left_angular: tuple[int, int, int],
    right_angular: tuple[int, int, int],
    /,
) -> Array:
    overlap = _primitive_overlap(
        left_exponent,
        right_exponent,
        left_center,
        right_center,
        left_angular,
        right_angular,
    )
    value = right_exponent * (2 * sum(right_angular) + 3) * overlap
    for axis in range(3):
        raised = list(right_angular)
        raised[axis] += 2
        value = value - 2.0 * right_exponent**2 * _primitive_overlap(
            left_exponent,
            right_exponent,
            left_center,
            right_center,
            left_angular,
            (raised[0], raised[1], raised[2]),
        )
        if right_angular[axis] >= 2:
            lowered = list(right_angular)
            lowered[axis] -= 2
            value = value - 0.5 * right_angular[axis] * (
                right_angular[axis] - 1
            ) * _primitive_overlap(
                left_exponent,
                right_exponent,
                left_center,
                right_center,
                left_angular,
                (lowered[0], lowered[1], lowered[2]),
            )
    return value


def _coulomb_auxiliary(
    exponent: Array,
    displacement: Array,
    maximum_orders: tuple[int, int, int],
    /,
):
    maximum_boys = sum(maximum_orders)
    argument = exponent * jnp.sum(displacement * displacement)
    boys = boys_values(maximum_boys, argument)
    cache: dict[tuple[int, int, int, int], Array] = {}

    def auxiliary(x: int, y: int, z: int, order: int) -> Array:
        key = (x, y, z, order)
        if key in cache:
            return cache[key]
        if x == 0 and y == 0 and z == 0:
            result = (-2.0 * exponent) ** order * boys[order]
        elif x > 0:
            result = displacement[0] * auxiliary(x - 1, y, z, order + 1)
            if x > 1:
                result = result + (x - 1) * auxiliary(x - 2, y, z, order + 1)
        elif y > 0:
            result = displacement[1] * auxiliary(x, y - 1, z, order + 1)
            if y > 1:
                result = result + (y - 1) * auxiliary(x, y - 2, z, order + 1)
        else:
            result = displacement[2] * auxiliary(x, y, z - 1, order + 1)
            if z > 1:
                result = result + (z - 1) * auxiliary(x, y, z - 2, order + 1)
        cache[key] = result
        return result

    return auxiliary


def _primitive_nuclear_attraction(
    left_exponent: Array,
    right_exponent: Array,
    left_center: Array,
    right_center: Array,
    left_angular: tuple[int, int, int],
    right_angular: tuple[int, int, int],
    source_position: Array,
    source_charge: Array,
    /,
) -> Array:
    power = left_exponent + right_exponent
    product_center = (left_exponent * left_center + right_exponent * right_center) / power
    hermite = tuple(
        _hermite_coefficients(
            left_angular[axis],
            right_angular[axis],
            left_exponent,
            right_exponent,
            left_center[axis] - right_center[axis],
        )
        for axis in range(3)
    )
    maxima = (
        len(hermite[0]) - 1,
        len(hermite[1]) - 1,
        len(hermite[2]) - 1,
    )
    auxiliary = _coulomb_auxiliary(power, product_center - source_position, maxima)
    value = jnp.asarray(0.0, dtype=power.dtype)
    for x, coefficient_x in enumerate(hermite[0]):
        for y, coefficient_y in enumerate(hermite[1]):
            for z, coefficient_z in enumerate(hermite[2]):
                value = value + (
                    coefficient_x * coefficient_y * coefficient_z * auxiliary(x, y, z, 0)
                )
    return -source_charge * 2.0 * jnp.pi / power * value


def _primitive_electron_repulsion(
    exponent_a: Array,
    exponent_b: Array,
    exponent_c: Array,
    exponent_d: Array,
    center_a: Array,
    center_b: Array,
    center_c: Array,
    center_d: Array,
    angular_a: tuple[int, int, int],
    angular_b: tuple[int, int, int],
    angular_c: tuple[int, int, int],
    angular_d: tuple[int, int, int],
    /,
    *,
    range_separation: float | None = None,
    short_range: bool = False,
) -> Array:
    power_ab = exponent_a + exponent_b
    power_cd = exponent_c + exponent_d
    product_ab = (exponent_a * center_a + exponent_b * center_b) / power_ab
    product_cd = (exponent_c * center_c + exponent_d * center_d) / power_cd
    hermite_ab = tuple(
        _hermite_coefficients(
            angular_a[axis],
            angular_b[axis],
            exponent_a,
            exponent_b,
            center_a[axis] - center_b[axis],
        )
        for axis in range(3)
    )
    hermite_cd = tuple(
        _hermite_coefficients(
            angular_c[axis],
            angular_d[axis],
            exponent_c,
            exponent_d,
            center_c[axis] - center_d[axis],
        )
        for axis in range(3)
    )
    combined = (
        len(hermite_ab[0]) + len(hermite_cd[0]) - 2,
        len(hermite_ab[1]) + len(hermite_cd[1]) - 2,
        len(hermite_ab[2]) + len(hermite_cd[2]) - 2,
    )
    reduced = power_ab * power_cd / (power_ab + power_cd)
    prefactor = 2.0 * jnp.pi**2.5 / (power_ab * power_cd * jnp.sqrt(power_ab + power_cd))

    def evaluate(effective_exponent):
        auxiliary = _coulomb_auxiliary(
            effective_exponent, product_ab - product_cd, combined
        )
        value = jnp.asarray(0.0, dtype=power_ab.dtype)
        for x, coefficient_x in enumerate(hermite_ab[0]):
            for y, coefficient_y in enumerate(hermite_ab[1]):
                for z, coefficient_z in enumerate(hermite_ab[2]):
                    left = coefficient_x * coefficient_y * coefficient_z
                    for xx, coefficient_xx in enumerate(hermite_cd[0]):
                        for yy, coefficient_yy in enumerate(hermite_cd[1]):
                            for zz, coefficient_zz in enumerate(hermite_cd[2]):
                                parity = -1.0 if (xx + yy + zz) % 2 else 1.0
                                value = value + (
                                    left
                                    * coefficient_xx
                                    * coefficient_yy
                                    * coefficient_zz
                                    * parity
                                    * auxiliary(x + xx, y + yy, z + zz, 0)
                                )
        return prefactor * jnp.sqrt(effective_exponent / reduced) * value

    full = evaluate(reduced)
    if range_separation is None:
        if short_range:
            raise ValueError("short_range requires a range-separation parameter.")
        return full
    omega = jnp.asarray(range_separation, dtype=power_ab.dtype)
    omega = eqx.error_if(
        omega,
        (~jnp.isfinite(omega)) | (omega <= 0.0),
        "Range-separation parameter must be finite and positive.",
    )
    long_exponent = reduced * omega**2 / (reduced + omega**2)
    long_range = evaluate(long_exponent)
    return full - long_range if short_range else long_range


def _contracted_one_body(
    basis: PreparedGaussianBasis,
    positions: Array,
    evaluator,
    /,
) -> Array:
    count = basis.cartesian_basis_function_count
    primitive_count = int(basis.exponents.shape[1])
    matrix = jnp.zeros((count, count), dtype=positions.dtype)
    for left in range(count):
        center_left = positions[basis.center_indices[left]]
        for right in range(left + 1):
            center_right = positions[basis.center_indices[right]]
            value = jnp.asarray(0.0, dtype=positions.dtype)
            for primitive_left in range(primitive_count):
                for primitive_right in range(primitive_count):
                    active = (
                        basis.primitive_mask[left, primitive_left]
                        & basis.primitive_mask[right, primitive_right]
                    )
                    term = evaluator(
                        basis.exponents[left, primitive_left],
                        basis.exponents[right, primitive_right],
                        center_left,
                        center_right,
                        basis.angular_tuples[left],
                        basis.angular_tuples[right],
                    )
                    value = value + jnp.where(
                        active,
                        basis.normalized_coefficients[left, primitive_left]
                        * basis.normalized_coefficients[right, primitive_right]
                        * term,
                        0.0,
                    )
            matrix = matrix.at[left, right].set(value)
            matrix = matrix.at[right, left].set(value)
    return basis.transform_one_body(matrix)


def overlap_matrix(basis: PreparedGaussianBasis, positions: ArrayLike, /) -> Array:
    return _contracted_one_body(basis, jnp.asarray(positions), _primitive_overlap)


def kinetic_matrix(basis: PreparedGaussianBasis, positions: ArrayLike, /) -> Array:
    return _contracted_one_body(basis, jnp.asarray(positions), _primitive_kinetic)


def multipole_integrals(
    basis: PreparedGaussianBasis,
    positions: ArrayLike,
    powers: tuple[int, int, int],
    /,
) -> Array:
    powers_ = (int(powers[0]), int(powers[1]), int(powers[2]))
    if len(powers_) != 3 or any(value < 0 for value in powers_):
        raise ValueError("Multipole powers must be three non-negative integers.")
    return _contracted_one_body(
        basis,
        jnp.asarray(positions),
        lambda a, b, left, right, la, lb: _primitive_multipole(
            a, b, left, right, la, lb, powers_
        ),
    )


def dipole_integrals(basis: PreparedGaussianBasis, positions: ArrayLike, /) -> Array:
    return jnp.stack(
        tuple(
            multipole_integrals(
                basis,
                positions,
                (
                    1 if component == 0 else 0,
                    1 if component == 1 else 0,
                    1 if component == 2 else 0,
                ),
            )
            for component in range(3)
        )
    )


def nuclear_attraction_matrix(
    basis: PreparedGaussianBasis,
    positions: ArrayLike,
    nuclear_charges: ArrayLike,
    /,
) -> Array:
    coordinate = jnp.asarray(positions)
    charges = jnp.asarray(nuclear_charges, dtype=coordinate.dtype)
    if charges.shape != (coordinate.shape[0],):
        raise ValueError("nuclear_charges must align with positions.")

    def evaluator(a, b, left, right, la, lb):
        value = jnp.asarray(0.0, dtype=coordinate.dtype)
        for index in range(int(charges.size)):
            active = charges[index] != 0.0
            source = jnp.where(
                active, coordinate[index], jnp.zeros((3,), dtype=coordinate.dtype)
            )
            value = value + jnp.where(
                active,
                _primitive_nuclear_attraction(
                    a, b, left, right, la, lb, source, charges[index]
                ),
                0.0,
            )
        return value

    return _contracted_one_body(basis, coordinate, evaluator)


def point_charge_potential_matrix(
    basis: PreparedGaussianBasis,
    nuclear_positions: ArrayLike,
    point_positions: ArrayLike,
    point_charges: ArrayLike,
    /,
) -> Array:
    nuclei = jnp.asarray(nuclear_positions)
    points = jnp.asarray(point_positions, dtype=nuclei.dtype)
    charges = jnp.asarray(point_charges, dtype=nuclei.dtype)
    if points.ndim != 2 or points.shape[1] != 3 or charges.shape != (points.shape[0],):
        raise ValueError("Point positions and charges must have shapes (P, 3) and (P,).")

    def evaluator(a, b, left, right, la, lb):
        value = jnp.asarray(0.0, dtype=nuclei.dtype)
        for index in range(int(charges.size)):
            value = value + _primitive_nuclear_attraction(
                a, b, left, right, la, lb, points[index], charges[index]
            )
        return value

    return _contracted_one_body(basis, nuclei, evaluator)


def contracted_electron_repulsion_element(
    basis: PreparedGaussianBasis,
    positions: ArrayLike,
    a: int,
    b: int,
    c: int,
    d: int,
    /,
    *,
    range_separation: float | None = None,
    short_range: bool = False,
) -> Array:
    coordinate = jnp.asarray(positions)
    indices = (int(a), int(b), int(c), int(d))
    count = basis.cartesian_basis_function_count
    if any(index < 0 or index >= count for index in indices):
        raise IndexError("Cartesian Gaussian function index is out of range.")
    center_a, center_b, center_c, center_d = (
        coordinate[basis.center_indices[index]] for index in indices
    )
    primitive_count = int(basis.exponents.shape[1])
    value = jnp.asarray(0.0, dtype=coordinate.dtype)
    for pa in range(primitive_count):
        for pb in range(primitive_count):
            for pc in range(primitive_count):
                for pd in range(primitive_count):
                    active = (
                        basis.primitive_mask[a, pa]
                        & basis.primitive_mask[b, pb]
                        & basis.primitive_mask[c, pc]
                        & basis.primitive_mask[d, pd]
                    )
                    primitive = _primitive_electron_repulsion(
                        basis.exponents[a, pa],
                        basis.exponents[b, pb],
                        basis.exponents[c, pc],
                        basis.exponents[d, pd],
                        center_a,
                        center_b,
                        center_c,
                        center_d,
                        basis.angular_tuples[a],
                        basis.angular_tuples[b],
                        basis.angular_tuples[c],
                        basis.angular_tuples[d],
                        range_separation=range_separation,
                        short_range=short_range,
                    )
                    coefficient = (
                        basis.normalized_coefficients[a, pa]
                        * basis.normalized_coefficients[b, pb]
                        * basis.normalized_coefficients[c, pc]
                        * basis.normalized_coefficients[d, pd]
                    )
                    value = value + jnp.where(active, coefficient * primitive, 0.0)
    return value


def electron_repulsion_tensor(
    basis: PreparedGaussianBasis, positions: ArrayLike, /
) -> Array:
    coordinate = jnp.asarray(positions)
    count = basis.cartesian_basis_function_count
    if count > 64:
        raise ValueError(
            "Dense ERIs are limited to 64 Cartesian functions; use direct or factorized routes."
        )
    tensor = jnp.zeros((count, count, count, count), dtype=coordinate.dtype)
    for a in range(count):
        for b in range(count):
            for c in range(count):
                for d in range(count):
                    tensor = tensor.at[a, b, c, d].set(
                        contracted_electron_repulsion_element(
                            basis, coordinate, a, b, c, d
                        )
                    )
    return basis.transform_two_body(tensor)


def range_separated_electron_repulsion_tensor(
    basis: PreparedGaussianBasis,
    positions: ArrayLike,
    range_separation: float,
    /,
    *,
    short_range: bool = False,
) -> Array:
    coordinate = jnp.asarray(positions)
    count = basis.cartesian_basis_function_count
    if count > 64:
        raise ValueError(
            "Dense range-separated ERIs are limited to 64 Cartesian functions."
        )
    tensor = jnp.zeros((count, count, count, count), dtype=coordinate.dtype)
    for a in range(count):
        for b in range(count):
            for c in range(count):
                for d in range(count):
                    tensor = tensor.at[a, b, c, d].set(
                        contracted_electron_repulsion_element(
                            basis,
                            coordinate,
                            a,
                            b,
                            c,
                            d,
                            range_separation=range_separation,
                            short_range=short_range,
                        )
                    )
    return basis.transform_two_body(tensor)


def nuclear_repulsion_energy(
    positions: ArrayLike, nuclear_charges: ArrayLike, /
) -> Array:
    coordinate = jnp.asarray(positions)
    charges = jnp.asarray(nuclear_charges, dtype=coordinate.dtype)
    energy = jnp.asarray(0.0, dtype=coordinate.dtype)
    for left in range(int(charges.size)):
        for right in range(left):
            active = (charges[left] != 0.0) & (charges[right] != 0.0)
            safe_left = jnp.where(
                active, coordinate[left], jnp.zeros((3,), dtype=coordinate.dtype)
            )
            safe_right = jnp.where(
                active, coordinate[right], jnp.ones((3,), dtype=coordinate.dtype)
            )
            distance = jnp.sqrt(jnp.sum((safe_left - safe_right) ** 2))
            energy = energy + jnp.where(
                active, charges[left] * charges[right] / distance, 0.0
            )
    return energy


def nuclear_point_charge_energy(
    nuclear_positions: ArrayLike,
    nuclear_charges: ArrayLike,
    point_positions: ArrayLike,
    point_charges: ArrayLike,
    /,
) -> Array:
    nuclei = jnp.asarray(nuclear_positions)
    nuclear_charge = jnp.asarray(nuclear_charges, dtype=nuclei.dtype)
    points = jnp.asarray(point_positions, dtype=nuclei.dtype)
    point_charge = jnp.asarray(point_charges, dtype=nuclei.dtype)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or point_charge.shape != (points.shape[0],)
    ):
        raise ValueError("Point positions and charges must have shapes (P, 3) and (P,).")
    displacement = nuclei[:, None, :] - points[None, :, :]
    distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=2))
    return jnp.sum(nuclear_charge[:, None] * point_charge[None, :] / distance)


class MolecularIntegralKernel(eqx.Module):
    overlap: Array
    kinetic: Array
    nuclear_attraction: Array
    core_hamiltonian: Array
    electron_repulsion: Array
    dipole: Array
    nuclear_repulsion: Array


def molecular_integrals(
    basis: PreparedGaussianBasis,
    positions: ArrayLike,
    nuclear_charges: ArrayLike,
    /,
) -> MolecularIntegralKernel:
    overlap = overlap_matrix(basis, positions)
    kinetic = kinetic_matrix(basis, positions)
    attraction = nuclear_attraction_matrix(basis, positions, nuclear_charges)
    return MolecularIntegralKernel(
        overlap,
        kinetic,
        attraction,
        kinetic + attraction,
        electron_repulsion_tensor(basis, positions),
        dipole_integrals(basis, positions),
        nuclear_repulsion_energy(positions, nuclear_charges),
    )


__all__ = [
    "MolecularIntegralKernel",
    "boys0",
    "boys_values",
    "contracted_electron_repulsion_element",
    "dipole_integrals",
    "electron_repulsion_tensor",
    "kinetic_matrix",
    "molecular_integrals",
    "multipole_integrals",
    "nuclear_attraction_matrix",
    "nuclear_point_charge_energy",
    "nuclear_repulsion_energy",
    "overlap_matrix",
    "range_separated_electron_repulsion_tensor",
    "point_charge_potential_matrix",
]
