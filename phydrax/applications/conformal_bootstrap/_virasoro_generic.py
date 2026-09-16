#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Generic finite-level Virasoro blocks and bounded Liouville correlators."""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import mpmath as mp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ._virasoro import elliptic_nome


def _complex_payload(value: complex, /) -> tuple[float, float]:
    return float(value.real), float(value.imag)


@functools.cache
def _partitions(level: int, maximum: int | None = None) -> tuple[tuple[int, ...], ...]:
    if level == 0:
        return ((),)
    upper = level if maximum is None else min(level, maximum)
    result: list[tuple[int, ...]] = []
    for first in range(upper, 0, -1):
        result.extend((first,) + rest for rest in _partitions(level - first, first))
    return tuple(result)


@functools.cache
def _canonical_negative_modes(
    sequence: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], complex], ...]:
    for index in range(len(sequence) - 1):
        left = sequence[index]
        right = sequence[index + 1]
        if left < right:
            swapped = sequence[:index] + (right, left) + sequence[index + 2 :]
            merged = sequence[:index] + (left + right,) + sequence[index + 2 :]
            values: dict[tuple[int, ...], complex] = {}
            for partition, coefficient in _canonical_negative_modes(swapped):
                values[partition] = values.get(partition, 0.0j) + coefficient
            for partition, coefficient in _canonical_negative_modes(merged):
                values[partition] = (
                    values.get(partition, 0.0j) + (right - left) * coefficient
                )
            return tuple(sorted(values.items(), reverse=True))
    return ((sequence, 1.0 + 0.0j),)


def _prepend_mode(
    mode: int, state: dict[tuple[int, ...], complex], /
) -> dict[tuple[int, ...], complex]:
    result: dict[tuple[int, ...], complex] = {}
    for partition, coefficient in state.items():
        for canonical, factor in _canonical_negative_modes((mode,) + partition):
            result[canonical] = result.get(canonical, 0.0j) + coefficient * factor
    return result


def _positive_mode_action(
    mode: int,
    partition: tuple[int, ...],
    central_charge: complex,
    highest_weight: complex,
    /,
) -> dict[tuple[int, ...], complex]:
    if mode <= 0:
        raise ValueError("Positive Virasoro mode action requires mode > 0.")
    if not partition:
        return {}
    first = partition[0]
    rest = partition[1:]
    result = _prepend_mode(
        first,
        _positive_mode_action(mode, rest, central_charge, highest_weight),
    )
    difference = mode - first
    factor = mode + first
    if difference > 0:
        commutator = _positive_mode_action(
            difference,
            rest,
            central_charge,
            highest_weight,
        )
        for target, coefficient in commutator.items():
            result[target] = result.get(target, 0.0j) + factor * coefficient
    elif difference == 0:
        level = sum(rest)
        result[rest] = result.get(rest, 0.0j) + factor * (highest_weight + level)
        central = central_charge * mode * (mode * mode - 1) / 12.0
        result[rest] = result.get(rest, 0.0j) + central
    else:
        commutator = _prepend_mode(first - mode, {rest: 1.0 + 0.0j})
        for target, coefficient in commutator.items():
            result[target] = result.get(target, 0.0j) + factor * coefficient
    return {key: value for key, value in result.items() if abs(value) > 1e-30}


def _gram_entry(
    left: tuple[int, ...],
    right: tuple[int, ...],
    central_charge: complex,
    highest_weight: complex,
    /,
) -> complex:
    state = {right: 1.0 + 0.0j}
    for mode in left:
        next_state: dict[tuple[int, ...], complex] = {}
        for partition, coefficient in state.items():
            for target, value in _positive_mode_action(
                mode,
                partition,
                central_charge,
                highest_weight,
            ).items():
                next_state[target] = next_state.get(target, 0.0j) + coefficient * value
        state = next_state
    return state.get((), 0.0 + 0.0j)


def _three_point_descendant(
    internal_weight: complex,
    first_external: complex,
    second_external: complex,
    partition: tuple[int, ...],
    /,
) -> complex:
    value = 1.0 + 0.0j
    previous_level = 0
    for mode in partition:
        value *= (
            internal_weight + mode * first_external - second_external + previous_level
        )
        previous_level += mode
    return value


class VirasoroEllipticBlockPlan(StrictModule):
    """Generic finite Verma-module recursion in elliptic normalization."""

    central_charge: complex = eqx.field(static=True)
    external_weights: tuple[complex, complex, complex, complex] = eqx.field(static=True)
    internal_weight: complex = eqx.field(static=True)
    maximum_level: int = eqx.field(static=True)
    singular_value_tolerance: float = eqx.field(static=True)
    maximum_partition_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        central_charge: complex,
        external_weights: Sequence[complex],
        internal_weight: complex,
        /,
        *,
        maximum_level: int = 8,
        singular_value_tolerance: float = 1e-12,
        maximum_partition_count: int = 4096,
    ):
        central = complex(central_charge)
        external = tuple(complex(value) for value in external_weights)
        internal = complex(internal_weight)
        level = int(maximum_level)
        tolerance = float(singular_value_tolerance)
        capacity = int(maximum_partition_count)
        values = np.asarray((central, internal, *external), dtype=np.complex128)
        if len(external) != 4 or not np.all(np.isfinite(values)):
            raise ValueError("Virasoro weights and central charge must be finite.")
        if level < 0 or tolerance <= 0.0 or capacity < 1:
            raise ValueError("Virasoro recursion limits must be positive.")
        partition_count = sum(len(_partitions(index)) for index in range(level + 1))
        if partition_count > capacity:
            raise ValueError("Virasoro descendant count exceeds maximum_partition_count.")
        content = {
            "kind": "virasoro-elliptic-block-plan",
            "central_charge": _complex_payload(central),
            "external_weights": [_complex_payload(value) for value in external],
            "internal_weight": _complex_payload(internal),
            "maximum_level": level,
            "singular_value_tolerance": tolerance,
            "maximum_partition_count": capacity,
        }
        self.central_charge = central
        self.external_weights = external  # type: ignore[assignment]
        self.internal_weight = internal
        self.maximum_level = level
        self.singular_value_tolerance = tolerance
        self.maximum_partition_count = capacity
        self.plan_id = canonical_fingerprint(content)


class VirasoroEllipticBlockEvidence(StrictModule):
    coefficients: Array
    gram_condition_numbers: Array
    values: Array
    elliptic_remainders: Array
    truncation_proxy: Array
    finite: Array
    plan_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class PreparedVirasoroEllipticBlock(StrictModule):
    plan: VirasoroEllipticBlockPlan
    coefficients: Array
    gram_condition_numbers: Array
    prepared_id: str = eqx.field(static=True)

    def values(self, cross_ratios: ArrayLike, /) -> Array:
        points = jnp.asarray(cross_ratios, dtype=jnp.complex128)
        if points.ndim < 1:
            raise ValueError(
                "Virasoro block evaluation requires an array of cross ratios."
            )
        if bool(np.any(np.asarray(~jnp.isfinite(points)))):
            raise ValueError("Virasoro cross ratios must be finite.")
        powers = jnp.arange(self.coefficients.shape[0])
        reduced = jnp.sum(self.coefficients * points[..., None] ** powers, axis=-1)
        leading = (
            self.plan.internal_weight
            - self.plan.external_weights[0]
            - self.plan.external_weights[1]
        )
        return points**leading * reduced

    def evidence(self, cross_ratios: ArrayLike, /) -> VirasoroEllipticBlockEvidence:
        points = jnp.asarray(cross_ratios, dtype=jnp.complex128)
        values = self.values(points)
        q = elliptic_nome(jnp.real(points))
        theta = 1.0 + 2.0 * sum(
            (q ** (index * index) for index in range(1, self.plan.maximum_level + 2)),
            jnp.zeros_like(q),
        )
        shifted = (self.plan.central_charge - 1.0) / 24.0
        h1, h2, h3, _h4 = self.plan.external_weights
        prefactor = (
            (16.0 * q) ** (self.plan.internal_weight - shifted)
            * points ** (shifted - h1 - h2)
            * (1.0 - points) ** (shifted - h2 - h3)
            * theta
            ** (
                (self.plan.central_charge - 1.0) / 2.0
                - 4.0 * sum(self.plan.external_weights)
            )
        )
        remainder = values / prefactor
        final_term = self.coefficients[-1] * points ** (self.coefficients.shape[0] - 1)
        truncation = jnp.abs(final_term) / jnp.maximum(jnp.abs(values), 1e-300)
        finite = jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(remainder))
        evidence_id = canonical_fingerprint(
            {
                "kind": "virasoro-elliptic-block-evidence",
                "plan": self.plan.plan_id,
                "points": array_tree_fingerprint(np.asarray(points)),
            }
        )
        return VirasoroEllipticBlockEvidence(
            coefficients=self.coefficients,
            gram_condition_numbers=self.gram_condition_numbers,
            values=values,
            elliptic_remainders=remainder,
            truncation_proxy=truncation,
            finite=finite,
            plan_id=self.plan.plan_id,
            evidence_id=evidence_id,
        )


def prepare_virasoro_elliptic_block(
    plan: VirasoroEllipticBlockPlan,
    /,
) -> PreparedVirasoroEllipticBlock:
    """Prepare generic chiral Virasoro coefficients from exact algebra relations."""

    if not isinstance(plan, VirasoroEllipticBlockPlan):
        raise TypeError("plan must be VirasoroEllipticBlockPlan.")
    coefficients: list[complex] = []
    conditions: list[float] = []
    h1, h2, h3, h4 = plan.external_weights
    for level in range(plan.maximum_level + 1):
        partitions = _partitions(level)
        gram = np.asarray(
            [
                [
                    _gram_entry(
                        left,
                        right,
                        plan.central_charge,
                        plan.internal_weight,
                    )
                    for right in partitions
                ]
                for left in partitions
            ],
            dtype=np.complex128,
        )
        singular_values = np.linalg.svd(gram, compute_uv=False)
        condition = float(
            singular_values[0] / singular_values[-1]
            if singular_values[-1] > 0.0
            else math.inf
        )
        conditions.append(condition)
        if singular_values[-1] <= plan.singular_value_tolerance:
            raise ValueError(
                f"Virasoro Gram matrix is singular at descendant level {level}."
            )
        right = np.asarray(
            [
                _three_point_descendant(plan.internal_weight, h1, h2, partition)
                for partition in partitions
            ]
        )
        left = np.asarray(
            [
                _three_point_descendant(plan.internal_weight, h3, h4, partition)
                for partition in partitions
            ]
        )
        coefficients.append(complex(left @ np.linalg.solve(gram, right)))
    coefficient_table = np.asarray(coefficients, dtype=np.complex128)
    condition_table = np.asarray(conditions, dtype=float)
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-virasoro-elliptic-block",
            "plan": plan.plan_id,
            "coefficients": array_tree_fingerprint(coefficient_table),
            "gram_condition_numbers": array_tree_fingerprint(condition_table),
        }
    )
    return PreparedVirasoroEllipticBlock(
        plan=plan,
        coefficients=jnp.asarray(coefficient_table),
        gram_condition_numbers=jnp.asarray(condition_table),
        prepared_id=prepared_id,
    )


def _upsilon_strip(b: mp.mpf, value: mp.mpc, /) -> mp.mpc:
    charge = b + 1 / b
    displacement = charge / 2 - value

    def integrand(parameter: mp.mpf) -> mp.mpc:
        if abs(parameter) < mp.mpf("1e-30"):
            return -(displacement**2)
        numerator = mp.sinh(displacement * parameter / 2) ** 2
        denominator = mp.sinh(b * parameter / 2) * mp.sinh(parameter / (2 * b))
        return (
            displacement**2 * mp.e ** (-parameter) - numerator / denominator
        ) / parameter

    return mp.e ** mp.quad(integrand, [0, 1, mp.inf])


def liouville_upsilon(
    coupling: float,
    value: complex,
    /,
    *,
    precision_digits: int = 50,
) -> complex:
    """Evaluate Upsilon_b by its strip integral and exact shift relations."""

    b_value = float(coupling)
    digits = int(precision_digits)
    if not math.isfinite(b_value) or b_value <= 0.0 or digits < 24:
        raise ValueError("Liouville coupling and precision are invalid.")
    with mp.workdps(digits):
        b = mp.mpf(str(b_value))
        charge = b + 1 / b
        argument = mp.mpc(value)
        factor = mp.mpc(1)
        steps = 0
        while mp.re(argument) <= 0:
            factor /= mp.gamma(b * argument) / mp.gamma(1 - b * argument)
            factor /= b ** (1 - 2 * b * argument)
            argument += b
            steps += 1
            if steps > 256:
                raise ValueError("Liouville Upsilon shift count exceeds admission.")
        while mp.re(argument) >= charge:
            argument -= b
            factor *= mp.gamma(b * argument) / mp.gamma(1 - b * argument)
            factor *= b ** (1 - 2 * b * argument)
            steps += 1
            if steps > 256:
                raise ValueError("Liouville Upsilon shift count exceeds admission.")
        return complex(factor * _upsilon_strip(b, argument))


def dozz_structure_constant(
    coupling: float,
    cosmological_constant: float,
    alphas: Sequence[complex],
    /,
    *,
    precision_digits: int = 50,
) -> complex:
    """Evaluate one convention-pinned DOZZ three-point structure constant."""

    values = tuple(complex(value) for value in alphas)
    b = float(coupling)
    mu = float(cosmological_constant)
    if len(values) != 3 or b <= 0.0 or mu <= 0.0:
        raise ValueError("DOZZ requires three momenta and positive b and mu.")
    charge = b + 1.0 / b
    total = sum(values)
    epsilon = 10.0 ** (-(precision_digits // 3))
    derivative_zero = (
        liouville_upsilon(
            b,
            epsilon,
            precision_digits=precision_digits,
        )
        / epsilon
    )
    gamma_b2 = complex(mp.gamma(b * b) / mp.gamma(1.0 - b * b))
    normalization = (math.pi * mu * gamma_b2 * b ** (2.0 - 2.0 * b * b)) ** (
        (charge - total) / b
    )
    numerator = derivative_zero
    for alpha in values:
        numerator *= liouville_upsilon(
            b,
            2.0 * alpha,
            precision_digits=precision_digits,
        )
    denominator = liouville_upsilon(
        b,
        total - charge,
        precision_digits=precision_digits,
    )
    for index in range(3):
        denominator *= liouville_upsilon(
            b,
            total - 2.0 * values[index],
            precision_digits=precision_digits,
        )
    return normalization * numerator / denominator


class LiouvilleFourPointPlan(StrictModule):
    """Bounded principal-series Liouville four-point quadrature."""

    coupling: float = eqx.field(static=True)
    cosmological_constant: float = eqx.field(static=True)
    external_alphas: tuple[complex, complex, complex, complex] = eqx.field(static=True)
    cross_ratios: Array
    maximum_momentum: float = eqx.field(static=True)
    momentum_nodes: int = eqx.field(static=True)
    block_level: int = eqx.field(static=True)
    precision_digits: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        coupling: float,
        cosmological_constant: float,
        external_alphas: Sequence[complex],
        cross_ratios: ArrayLike,
        /,
        *,
        maximum_momentum: float,
        momentum_nodes: int,
        block_level: int = 6,
        precision_digits: int = 50,
    ):
        b = float(coupling)
        mu = float(cosmological_constant)
        alphas = tuple(complex(value) for value in external_alphas)
        points = np.asarray(cross_ratios, dtype=float)
        maximum = float(maximum_momentum)
        nodes = int(momentum_nodes)
        level = int(block_level)
        digits = int(precision_digits)
        if b <= 0.0 or mu <= 0.0 or len(alphas) != 4:
            raise ValueError("Liouville plan parameters are invalid.")
        if (
            points.ndim != 1
            or not points.size
            or np.any((points <= 0.0) | (points >= 1.0))
        ):
            raise ValueError("Liouville cross ratios must be a non-empty array in (0,1).")
        if maximum <= 0.0 or nodes < 3 or nodes % 2 == 0 or level < 0 or digits < 24:
            raise ValueError("Liouville quadrature and precision limits are invalid.")
        content = {
            "kind": "liouville-four-point-plan",
            "coupling": b,
            "cosmological_constant": mu,
            "external_alphas": [_complex_payload(value) for value in alphas],
            "cross_ratios": array_tree_fingerprint(points),
            "maximum_momentum": maximum,
            "momentum_nodes": nodes,
            "block_level": level,
            "precision_digits": digits,
        }
        self.coupling = b
        self.cosmological_constant = mu
        self.external_alphas = alphas  # type: ignore[assignment]
        self.cross_ratios = jnp.asarray(points)
        self.maximum_momentum = maximum
        self.momentum_nodes = nodes
        self.block_level = level
        self.precision_digits = digits
        self.plan_id = canonical_fingerprint(content)


class LiouvilleFourPointEvidence(StrictModule):
    values: Array
    reduced_cutoff_values: Array
    truncation_proxy: Array
    finite: Array
    plan_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def _liouville_integrand(
    plan: LiouvilleFourPointPlan,
    momentum: float,
    cross_ratios: np.ndarray,
    /,
) -> np.ndarray:
    b = plan.coupling
    charge = b + 1.0 / b
    internal_alpha = charge / 2.0 + 1j * momentum
    internal_weight = internal_alpha * (charge - internal_alpha)
    weights = tuple(alpha * (charge - alpha) for alpha in plan.external_alphas)
    central_charge = 1.0 + 6.0 * charge * charge
    block = prepare_virasoro_elliptic_block(
        VirasoroEllipticBlockPlan(
            central_charge,
            weights,
            internal_weight,
            maximum_level=plan.block_level,
        )
    )
    block_values = np.asarray(block.values(cross_ratios), dtype=np.complex128)
    left = dozz_structure_constant(
        b,
        plan.cosmological_constant,
        (*plan.external_alphas[:2], internal_alpha),
        precision_digits=plan.precision_digits,
    )
    right = dozz_structure_constant(
        b,
        plan.cosmological_constant,
        (charge - internal_alpha, *plan.external_alphas[2:]),
        precision_digits=plan.precision_digits,
    )
    return np.real(left * right * np.abs(block_values) ** 2)


def prepare_liouville_four_point(
    plan: LiouvilleFourPointPlan,
    /,
) -> LiouvilleFourPointEvidence:
    """Integrate the bounded Liouville principal series and audit cutoff sensitivity."""

    if not isinstance(plan, LiouvilleFourPointPlan):
        raise TypeError("plan must be LiouvilleFourPointPlan.")
    momenta = np.linspace(0.0, plan.maximum_momentum, plan.momentum_nodes)
    points = np.asarray(plan.cross_ratios)
    integrand = np.stack(
        [_liouville_integrand(plan, value, points) for value in momenta],
        axis=0,
    )
    values = np.trapezoid(integrand, momenta, axis=0)
    reduced_nodes = max(3, plan.momentum_nodes // 2 * 2 - 1)
    reduced_momenta = np.linspace(0.0, 0.75 * plan.maximum_momentum, reduced_nodes)
    reduced_integrand = np.stack(
        [_liouville_integrand(plan, value, points) for value in reduced_momenta],
        axis=0,
    )
    reduced = np.trapezoid(reduced_integrand, reduced_momenta, axis=0)
    truncation = np.abs(values - reduced) / np.maximum(np.abs(values), 1e-300)
    finite = bool(np.all(np.isfinite(values)) and np.all(np.isfinite(truncation)))
    evidence_id = canonical_fingerprint(
        {
            "kind": "liouville-four-point-evidence",
            "plan": plan.plan_id,
            "values": array_tree_fingerprint(values),
            "reduced": array_tree_fingerprint(reduced),
        }
    )
    return LiouvilleFourPointEvidence(
        values=jnp.asarray(values),
        reduced_cutoff_values=jnp.asarray(reduced),
        truncation_proxy=jnp.asarray(truncation),
        finite=jnp.asarray(finite),
        plan_id=plan.plan_id,
        evidence_id=evidence_id,
    )


__all__ = [
    "LiouvilleFourPointEvidence",
    "LiouvilleFourPointPlan",
    "PreparedVirasoroEllipticBlock",
    "VirasoroEllipticBlockEvidence",
    "VirasoroEllipticBlockPlan",
    "dozz_structure_constant",
    "liouville_upsilon",
    "prepare_liouville_four_point",
    "prepare_virasoro_elliptic_block",
]
