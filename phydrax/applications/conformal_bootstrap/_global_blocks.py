#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite meromorphic radial recursion for global scalar conformal blocks."""

from __future__ import annotations

from collections.abc import Sequence
from math import factorial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ._data import ConformalDataPlan


def _rising_factorial(value: float, count: int, /) -> float:
    result = 1.0
    for index in range(int(count)):
        result *= value + index
    return result


def _delta_pole(nu: float, k: int, spin: int, series: int, /) -> float:
    if series == 1:
        return 1.0 - spin - k
    if series == 2:
        return 1.0 + nu - k
    if series == 3:
        return 1.0 + spin + 2.0 * nu - k
    raise ValueError("Unknown meromorphic pole series.")


def _delta_residue(
    nu: float,
    k: int,
    spin: int,
    delta_12: float,
    delta_34: float,
    series: int,
    /,
) -> float:
    if series != 2 and k % 2 and delta_12 == 0.0 and delta_34 == 0.0:
        return 0.0
    k_factorial = float(factorial(k))
    if series == 1:
        result = -(
            k
            * (-4.0) ** k
            / k_factorial**2
            * _rising_factorial((1.0 - k + delta_12) / 2.0, k)
            * _rising_factorial((1.0 - k + delta_34) / 2.0, k)
        )
        denominator = _rising_factorial(spin + nu, k)
        if denominator == 0.0:
            return np.nan
        return result * _rising_factorial(spin + 2.0 * nu, k) / denominator
    if series == 2:
        denominator = (
            _rising_factorial((spin + nu - k + 1.0) / 2.0, k)
            * _rising_factorial((spin + nu - k) / 2.0, k)
        ) ** 2
        spin_denominator = spin + nu + k
        if denominator == 0.0 or spin_denominator == 0.0:
            return np.nan
        result = (
            k
            * _rising_factorial(nu + 1.0, k - 1)
            / k_factorial**2
            * (spin + nu - k)
            / spin_denominator
            * _rising_factorial(-nu, k + 1)
            / denominator
        )
        for factor in (
            spin + nu + 1.0 - delta_12,
            spin + nu + 1.0 + delta_12,
            spin + nu + 1.0 - delta_34,
            spin + nu + 1.0 + delta_34,
        ):
            result *= _rising_factorial((factor - k) / 2.0, k)
        return result
    if series == 3:
        denominator = _rising_factorial(1.0 + nu + spin - k, k)
        if denominator == 0.0:
            return np.nan
        return -(
            k
            * (-4.0) ** k
            / k_factorial**2
            * _rising_factorial(1.0 + spin - k, k)
            * _rising_factorial((1.0 - k + delta_12) / 2.0, k)
            * _rising_factorial((1.0 - k + delta_34) / 2.0, k)
            / denominator
        )
    raise ValueError("Unknown meromorphic residue series.")


def _pole_data(
    spacetime_dimension: float,
    spin: int,
    maximum_level: int,
    delta_12: float,
    delta_34: float,
    /,
) -> tuple[tuple[int, float, float, int], ...]:
    nu = spacetime_dimension / 2.0 - 1.0
    values: list[tuple[int, float, float, int]] = []
    for k in range(1, maximum_level + 1):
        residue = _delta_residue(nu, k, spin, delta_12, delta_34, 1)
        if np.isfinite(residue) and residue != 0.0:
            values.append((k, _delta_pole(nu, k, spin, 1), residue, spin + k))
    for k in range(1, maximum_level // 2 + 1):
        residue = _delta_residue(nu, k, spin, delta_12, delta_34, 2)
        if np.isfinite(residue) and residue != 0.0:
            values.append((2 * k, _delta_pole(nu, k, spin, 2), residue, spin))
    for k in range(1, min(spin, maximum_level) + 1):
        residue = _delta_residue(nu, k, spin, delta_12, delta_34, 3)
        if np.isfinite(residue) and residue != 0.0:
            values.append((k, _delta_pole(nu, k, spin, 3), residue, spin - k))
    return tuple(sorted(values, key=lambda value: (value[0], value[3], value[1])))


def _gegenbauer(spin: int, nu: float, eta: Array, /) -> Array:
    if spin == 0:
        return jnp.ones_like(eta)
    if abs(nu) <= 1e-14:
        previous = jnp.ones_like(eta)
        current = eta
        for _ in range(2, spin + 1):
            previous, current = current, 2.0 * eta * current - previous
        return current
    previous = jnp.ones_like(eta)
    current = 2.0 * nu * eta
    for degree in range(2, spin + 1):
        following = (
            2.0 * (degree + nu - 1.0) * eta * current
            - (degree + 2.0 * nu - 2.0) * previous
        ) / degree
        previous, current = current, following
    normalization = factorial(spin) / _rising_factorial(2.0 * nu, spin)
    return normalization * current


def _leading_block(
    spacetime_dimension: float,
    spin: int,
    delta_12: float,
    delta_34: float,
    radius: Array,
    eta: Array,
    /,
) -> Array:
    nu = spacetime_dimension / 2.0 - 1.0
    angular = (-1.0) ** spin * _gegenbauer(spin, nu, eta)
    radial = (1.0 - radius**2) ** nu
    plus = (1.0 + radius**2 + 2.0 * radius * eta) ** ((1.0 + delta_12 - delta_34) / 2.0)
    minus = (1.0 + radius**2 - 2.0 * radius * eta) ** ((1.0 - delta_12 + delta_34) / 2.0)
    return angular / (radial * plus * minus)


def _radial_block(
    spacetime_dimension: float,
    scaling_dimension: Array,
    spin: int,
    delta_12: float,
    delta_34: float,
    radius: Array,
    eta: Array,
    recursion_order: int,
    pole_tolerance: float,
    /,
) -> Array:
    def recurse(delta: Array, ell: int, remaining: int) -> Array:
        value = _leading_block(spacetime_dimension, ell, delta_12, delta_34, radius, eta)
        if remaining == 0:
            return value
        for level, pole, residue, descendant_spin in _pole_data(
            spacetime_dimension,
            ell,
            remaining,
            delta_12,
            delta_34,
        ):
            denominator = delta - pole
            denominator = eqx.error_if(
                denominator,
                jnp.abs(denominator) <= pole_tolerance,
                "Scaling dimension lies on a retained meromorphic block pole.",
            )
            value = value + (
                residue
                * radius**level
                / denominator
                * recurse(
                    jnp.asarray(pole + level, dtype=delta.dtype),
                    descendant_spin,
                    remaining - level,
                )
            )
        return value

    return (4.0 * radius) ** scaling_dimension * recurse(
        scaling_dimension, spin, recursion_order
    )


def _rho(cross_ratio: Array, /) -> Array:
    return cross_ratio / (1.0 + jnp.sqrt(1.0 - cross_ratio)) ** 2


def _radial_coordinates(z: Array, zbar: Array, /) -> tuple[Array, Array]:
    rho = _rho(z)
    rho_bar = _rho(zbar)
    radius = jnp.sqrt(rho * rho_bar)
    eta = (rho + rho_bar) / (2.0 * radius)
    return radius, eta


def _hyp2f1_series(
    first: Array,
    second: Array,
    third: Array,
    argument: Array,
    order: int,
    pole_tolerance: float,
    /,
) -> Array:
    initial = (jnp.ones_like(argument), jnp.ones_like(argument))

    def body(index, carry):
        term, value = carry
        denominator = (third + index) * (index + 1.0)
        denominator = eqx.error_if(
            denominator,
            jnp.abs(denominator) <= pole_tolerance,
            "Hypergeometric block parameter lies on a retained pole.",
        )
        next_term = term * (first + index) * (second + index) / denominator * argument
        return next_term, value + next_term

    _, value = jax.lax.fori_loop(0, order - 1, body, initial)
    return value


def _k_beta(
    beta: Array,
    argument: Array,
    delta_12: float,
    delta_34: float,
    order: int,
    pole_tolerance: float,
    /,
) -> Array:
    return argument ** (beta / 2.0) * _hyp2f1_series(
        (beta - delta_12) / 2.0,
        (beta + delta_34) / 2.0,
        beta,
        argument,
        order,
        pole_tolerance,
    )


def _two_dimensional_block(
    scaling_dimension: Array,
    spin: int,
    z: Array,
    zbar: Array,
    delta_12: float,
    delta_34: float,
    order: int,
    pole_tolerance: float,
    /,
) -> Array:
    high = scaling_dimension + spin
    low = scaling_dimension - spin
    direct = _k_beta(high, z, delta_12, delta_34, order, pole_tolerance) * _k_beta(
        low, zbar, delta_12, delta_34, order, pole_tolerance
    )
    reflected = _k_beta(high, zbar, delta_12, delta_34, order, pole_tolerance) * _k_beta(
        low, z, delta_12, delta_34, order, pole_tolerance
    )
    normalization = 0.5 if spin == 0 else 1.0
    return normalization * (direct + reflected)


def _four_dimensional_block(
    scaling_dimension: Array,
    spin: int,
    z: Array,
    zbar: Array,
    delta_12: float,
    delta_34: float,
    order: int,
    pole_tolerance: float,
    /,
) -> Array:
    high = scaling_dimension + spin
    low = scaling_dimension - spin - 2.0

    def high_block(value):
        return _k_beta(high, value, delta_12, delta_34, order, pole_tolerance)

    def low_block(value):
        return _k_beta(low, value, delta_12, delta_34, order, pole_tolerance)

    high_z = high_block(z)
    low_z = low_block(z)
    high_zbar = high_block(zbar)
    low_zbar = low_block(zbar)
    close = jnp.abs(z - zbar) <= 32.0 * jnp.finfo(z.dtype).eps
    safe_difference = jnp.where(close, 1.0, z - zbar)
    separated = z * zbar / safe_difference * (high_z * low_zbar - high_zbar * low_z)
    high_derivative = jax.grad(high_block)(z)
    low_derivative = jax.grad(low_block)(z)
    diagonal = z**2 * (high_derivative * low_z - high_z * low_derivative)
    return ((-1.0) ** spin / 2.0**spin) * jnp.where(close, diagonal, separated)


def _mixed_derivative(function, z: Array, zbar: Array, first: int, second: int):
    differentiated = function
    for _ in range(first):
        previous = differentiated
        differentiated = lambda left, right, previous=previous: jax.grad(
            previous, argnums=0
        )(left, right)
    for _ in range(second):
        previous = differentiated
        differentiated = lambda left, right, previous=previous: jax.grad(
            previous, argnums=1
        )(left, right)
    return differentiated(z, zbar)


class GlobalScalarBlockPlan(StrictModule):
    """Finite global scalar-block recursion and derivative plan."""

    data: ConformalDataPlan
    evaluation_points: Array
    spins: tuple[int, ...] = eqx.field(static=True)
    derivative_orders: tuple[tuple[int, int], ...] = eqx.field(static=True)
    recursion_order: int = eqx.field(static=True)
    hypergeometric_order: int = eqx.field(static=True)
    integer_dimension_offset: float = eqx.field(static=True)
    pole_tolerance: float = eqx.field(static=True)
    maximum_evaluations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        data: ConformalDataPlan,
        evaluation_points: ArrayLike,
        spins: Sequence[int],
        derivative_orders: Sequence[tuple[int, int]],
        /,
        *,
        recursion_order: int = 6,
        hypergeometric_order: int = 128,
        integer_dimension_offset: float = 1e-6,
        pole_tolerance: float = 1e-10,
        maximum_evaluations: int = 2_000_000,
    ):
        if not isinstance(data, ConformalDataPlan):
            raise TypeError("data must be ConformalDataPlan.")
        points = np.asarray(evaluation_points, dtype=np.float64)
        spin_values = tuple(spins)
        derivatives = tuple(
            (int(first), int(second)) for first, second in derivative_orders
        )
        recursion = int(recursion_order)
        hypergeometric = int(hypergeometric_order)
        offset = float(integer_dimension_offset)
        pole = float(pole_tolerance)
        maximum = int(maximum_evaluations)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 1:
            raise ValueError("evaluation_points must have shape (points, 2).")
        if not np.all(np.isfinite(points)) or np.any((points <= 0.0) | (points >= 1.0)):
            raise ValueError("Real Euclidean z and zbar points must lie in (0, 1).")
        if not spin_values or any(value < 0 for value in spin_values):
            raise ValueError("spins must be nonempty and non-negative.")
        if (
            len(set(spin_values)) != len(spin_values)
            or tuple(sorted(spin_values)) != spin_values
        ):
            raise ValueError("spins must be unique and increasing.")
        declared_spins = {
            spin for sector in data.exchanged_sectors for spin in sector.spins
        }
        if not set(spin_values) <= declared_spins:
            raise ValueError("A requested spin is outside the conformal-data sectors.")
        if not derivatives or any(
            first < 0 or second < 0 for first, second in derivatives
        ):
            raise ValueError("derivative_orders must be nonempty and non-negative.")
        if len(set(derivatives)) != len(derivatives):
            raise ValueError("derivative_orders must be unique.")
        if recursion < 0 or hypergeometric < 2 or maximum < 1:
            raise ValueError("Block truncation/resource limits are invalid.")
        if (
            not np.isfinite(offset)
            or offset <= 0.0
            or not np.isfinite(pole)
            or pole <= 0.0
        ):
            raise ValueError("Dimension-limit and pole tolerances must be positive.")
        forecast = (
            points.shape[0]
            * len(spin_values)
            * len(derivatives)
            * max(1, (recursion + 1) ** 3)
        )
        if forecast > maximum:
            raise ValueError("Global block recursion exceeds maximum_evaluations.")
        self.data = data
        self.evaluation_points = jnp.asarray(points)
        self.spins = spin_values
        self.derivative_orders = derivatives
        self.recursion_order = recursion
        self.hypergeometric_order = hypergeometric
        self.integer_dimension_offset = offset
        self.pole_tolerance = pole
        self.maximum_evaluations = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "global-scalar-block-plan",
                "data": data.plan_id,
                "evaluation_points": array_tree_fingerprint(points),
                "spins": spin_values,
                "derivative_orders": derivatives,
                "recursion_order": recursion,
                "hypergeometric_order": hypergeometric,
                "integer_dimension_offset": offset,
                "pole_tolerance": pole,
                "maximum_evaluations": maximum,
            }
        )


class GlobalBlockEvidence(StrictModule):
    values: Array
    derivatives: Array
    truncation_proxy: Array
    casimir_residuals: Array
    finite: Array
    scaling_dimension: Array
    spin: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    method: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class PreparedGlobalScalarBlocks(StrictModule):
    """Prepared general-dimensional scalar blocks at fixed Euclidean points."""

    plan: GlobalScalarBlockPlan
    delta_12: float = eqx.field(static=True)
    delta_34: float = eqx.field(static=True)
    method: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def _unitarity_bound(self, spin: int, /) -> float:
        dimension = self.plan.data.spacetime_dimension
        if spin == 0:
            return max(0.0, (dimension - 2.0) / 2.0)
        return spin + dimension - 2.0

    def _value_at(
        self,
        scaling_dimension: Array,
        spin: int,
        z: Array,
        zbar: Array,
        /,
        *,
        recursion_order: int | None = None,
        hypergeometric_order: int | None = None,
    ) -> Array:
        dimension = self.plan.data.spacetime_dimension
        recursion = (
            self.plan.recursion_order if recursion_order is None else recursion_order
        )
        hypergeometric = (
            self.plan.hypergeometric_order
            if hypergeometric_order is None
            else hypergeometric_order
        )
        if dimension == 2.0:
            return _two_dimensional_block(
                scaling_dimension,
                spin,
                z,
                zbar,
                self.delta_12,
                self.delta_34,
                hypergeometric,
                self.plan.pole_tolerance,
            )
        if dimension == 4.0:
            return _four_dimensional_block(
                scaling_dimension,
                spin,
                z,
                zbar,
                self.delta_12,
                self.delta_34,
                hypergeometric,
                self.plan.pole_tolerance,
            )
        radius, eta = _radial_coordinates(z, zbar)
        nu = dimension / 2.0 - 1.0
        if abs(nu - round(nu)) <= 1e-12:
            offset = self.plan.integer_dimension_offset
            lower = _radial_block(
                dimension - 2.0 * offset,
                scaling_dimension,
                spin,
                self.delta_12,
                self.delta_34,
                radius,
                eta,
                recursion,
                self.plan.pole_tolerance,
            )
            upper = _radial_block(
                dimension + 2.0 * offset,
                scaling_dimension,
                spin,
                self.delta_12,
                self.delta_34,
                radius,
                eta,
                recursion,
                self.plan.pole_tolerance,
            )
            return 0.5 * (lower + upper)
        return _radial_block(
            dimension,
            scaling_dimension,
            spin,
            self.delta_12,
            self.delta_34,
            radius,
            eta,
            recursion,
            self.plan.pole_tolerance,
        )

    def values(self, scaling_dimension: ArrayLike, spin: int, /) -> Array:
        delta = jnp.asarray(scaling_dimension, dtype=self.plan.evaluation_points.dtype)
        ell = int(spin)
        if delta.ndim != 0 or ell not in self.plan.spins:
            raise ValueError("A scalar dimension and prepared spin are required.")
        delta = eqx.error_if(
            delta,
            ~jnp.isfinite(delta) | (delta <= self._unitarity_bound(ell)),
            "Scaling dimension must lie strictly above the declared unitarity bound.",
        )
        return jax.vmap(lambda point: self._value_at(delta, ell, point[0], point[1]))(
            self.plan.evaluation_points
        )

    def derivative_table(self, scaling_dimension: ArrayLike, spin: int, /) -> Array:
        delta = jnp.asarray(scaling_dimension, dtype=self.plan.evaluation_points.dtype)
        ell = int(spin)
        if delta.ndim != 0 or ell not in self.plan.spins:
            raise ValueError("A scalar dimension and prepared spin are required.")
        delta = eqx.error_if(
            delta,
            ~jnp.isfinite(delta) | (delta <= self._unitarity_bound(ell)),
            "Scaling dimension must lie strictly above the declared unitarity bound.",
        )

        def at_point(point):
            function = lambda z, zbar: self._value_at(delta, ell, z, zbar)
            return jnp.stack(
                tuple(
                    _mixed_derivative(function, point[0], point[1], first, second)
                    for first, second in self.plan.derivative_orders
                )
            )

        return jax.vmap(at_point)(self.plan.evaluation_points).T

    def _casimir_residual_at(
        self, scaling_dimension: Array, spin: int, point: Array, /
    ) -> Array:
        z, zbar = point[0], point[1]
        function = lambda left, right: self._value_at(
            scaling_dimension, spin, left, right
        )
        value = function(z, zbar)
        dz = jax.grad(function, argnums=0)(z, zbar)
        dzbar = jax.grad(function, argnums=1)(z, zbar)
        dzz = jax.grad(jax.grad(function, argnums=0), argnums=0)(z, zbar)
        dzbar_zbar = jax.grad(jax.grad(function, argnums=1), argnums=1)(z, zbar)
        dz_zbar = jax.grad(jax.grad(function, argnums=0), argnums=1)(z, zbar)
        a = -0.5 * self.delta_12
        b = 0.5 * self.delta_34
        direct = z**2 * (1.0 - z) * dzz - (a + b + 1.0) * z**2 * dz - a * b * z * value
        reflected = (
            zbar**2 * (1.0 - zbar) * dzbar_zbar
            - (a + b + 1.0) * zbar**2 * dzbar
            - a * b * zbar * value
        )
        nu = self.plan.data.spacetime_dimension / 2.0 - 1.0
        close = jnp.abs(z - zbar) <= 32.0 * jnp.finfo(z.dtype).eps
        safe_denominator = jnp.where(close, 1.0, z - zbar)
        separated = (
            2.0
            * nu
            * z
            * zbar
            / safe_denominator
            * ((1.0 - z) * dz - (1.0 - zbar) * dzbar)
        )
        diagonal_ratio = -(1.0 - z) * dz_zbar - dzbar + (1.0 - z) * dzbar_zbar
        coupled = jnp.where(close, 2.0 * nu * z**2 * diagonal_ratio, separated)
        eigenvalue = 0.5 * (
            scaling_dimension * (scaling_dimension - self.plan.data.spacetime_dimension)
            + spin * (spin + self.plan.data.spacetime_dimension - 2.0)
        )
        residual = direct + reflected + coupled - eigenvalue * value
        return jnp.abs(residual) / jnp.maximum(1.0, jnp.abs(eigenvalue * value))

    def evidence(self, scaling_dimension: ArrayLike, spin: int, /) -> GlobalBlockEvidence:
        delta = jnp.asarray(scaling_dimension, dtype=self.plan.evaluation_points.dtype)
        values = self.values(delta, spin)
        derivatives = self.derivative_table(delta, spin)
        reduced_recursion = max(0, self.plan.recursion_order - 2)
        reduced_hypergeometric = max(2, self.plan.hypergeometric_order // 2)
        reduced = jax.vmap(
            lambda point: self._value_at(
                delta,
                int(spin),
                point[0],
                point[1],
                recursion_order=reduced_recursion,
                hypergeometric_order=reduced_hypergeometric,
            )
        )(self.plan.evaluation_points)
        truncation = jnp.abs(values - reduced) / jnp.maximum(1.0, jnp.abs(values))
        casimir = jax.vmap(
            lambda point: self._casimir_residual_at(delta, int(spin), point)
        )(self.plan.evaluation_points)
        finite = (
            jnp.all(jnp.isfinite(values))
            & jnp.all(jnp.isfinite(derivatives))
            & jnp.all(jnp.isfinite(truncation))
            & jnp.all(jnp.isfinite(casimir))
        )
        return GlobalBlockEvidence(
            values=values,
            derivatives=derivatives,
            truncation_proxy=truncation,
            casimir_residuals=casimir,
            finite=finite,
            scaling_dimension=delta,
            spin=int(spin),
            plan_id=self.plan.plan_id,
            method=self.method,
            claim="finite-global-block-recursion-and-casimir-evidence-only",
        )


def prepare_global_scalar_blocks(
    plan: GlobalScalarBlockPlan, /
) -> PreparedGlobalScalarBlocks:
    if not isinstance(plan, GlobalScalarBlockPlan):
        raise TypeError("plan must be GlobalScalarBlockPlan.")
    delta_12, delta_34 = plan.data.external_dimension_differences
    dimension = plan.data.spacetime_dimension
    if dimension == 2.0:
        method = "two-dimensional-factorized-hypergeometric-series"
    elif dimension == 4.0:
        method = "four-dimensional-dolan-osborn-hypergeometric-series"
    elif abs((dimension / 2.0 - 1.0) - round(dimension / 2.0 - 1.0)) <= 1e-12:
        method = "symmetric-dimension-limit-meromorphic-radial-recursion"
    else:
        method = "meromorphic-radial-recursion"
    return PreparedGlobalScalarBlocks(
        plan=plan,
        delta_12=delta_12,
        delta_34=delta_34,
        method=method,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-global-scalar-blocks",
                "plan": plan.plan_id,
                "delta_12": delta_12,
                "delta_34": delta_34,
                "method": method,
            }
        ),
    )


__all__ = [
    "GlobalBlockEvidence",
    "GlobalScalarBlockPlan",
    "PreparedGlobalScalarBlocks",
    "prepare_global_scalar_blocks",
]
