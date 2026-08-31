#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import comb

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._lattice import LatticeBoltzmannVelocitySet
from ._precision import LatticeBoltzmannPrecisionPolicy


_D2Q9_EXPONENTS = (
    (0, 0),
    (1, 0),
    (0, 1),
    (2, 0),
    (0, 2),
    (1, 1),
    (2, 1),
    (1, 2),
    (2, 2),
)
_D3Q19_EXPONENTS = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (2, 0, 0),
    (0, 2, 0),
    (0, 0, 2),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (2, 1, 0),
    (2, 0, 1),
    (1, 2, 0),
    (0, 2, 1),
    (1, 0, 2),
    (0, 1, 2),
    (2, 2, 0),
    (2, 0, 2),
    (0, 2, 2),
)
_D3Q27_EXPONENTS = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (2, 0, 0),
    (0, 2, 0),
    (0, 0, 2),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (2, 1, 0),
    (2, 0, 1),
    (1, 2, 0),
    (0, 2, 1),
    (1, 0, 2),
    (0, 1, 2),
    (2, 2, 0),
    (2, 0, 2),
    (0, 2, 2),
    (1, 1, 1),
    (2, 2, 1),
    (2, 1, 2),
    (1, 2, 2),
    (2, 1, 1),
    (1, 2, 1),
    (1, 1, 2),
    (2, 2, 2),
)


def _default_exponents(velocity_set: LatticeBoltzmannVelocitySet, /):
    key = (velocity_set.dimension, velocity_set.population_count)
    if key == (2, 9):
        return _D2Q9_EXPONENTS
    if key == (3, 19):
        return _D3Q19_EXPONENTS
    if key == (3, 27):
        return _D3Q27_EXPONENTS
    raise ValueError(
        "A custom velocity set requires an explicit complete monomial exponent basis."
    )


class PreparedMomentBasis(StrictModule, NonTrainableState):
    """One certified population-to-moment transform prepared outside cell kernels."""

    transform: Array
    inverse_transform: Array
    exponents_array: Array
    shift_coefficients: Array
    shift_exponents: Array
    exponents: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    conserved_indices: tuple[int, ...] = eqx.field(static=True)
    second_order_indices: tuple[int, ...] = eqx.field(static=True)
    cumulant_order: tuple[int, ...] = eqx.field(static=True)
    cumulant_terms: tuple[tuple[tuple[int, int, int], ...], ...] = eqx.field(static=True)
    lattice_id: str = eqx.field(static=True)
    basis_name: str = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    @property
    def population_count(self) -> int:
        return len(self.exponents)

    @property
    def dimension(self) -> int:
        return len(self.exponents[0])

    def require_lattice(self, velocity_set: LatticeBoltzmannVelocitySet, /) -> None:
        if velocity_set.lattice_id != self.lattice_id:
            raise ValueError("Moment basis and velocity set do not match.")


class MomentBasisPlan(StrictModule, NonTrainableState):
    """Plan for a raw monomial basis, certified for invertibility at preparation."""

    basis_name: str = eqx.field(static=True)
    exponents: tuple[tuple[int, ...], ...] | None = eqx.field(static=True)
    maximum_condition_number: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        exponents: Sequence[Sequence[int]] | None = None,
        /,
        *,
        basis_name: str = "raw-monomial",
        maximum_condition_number: float = 1.0e12,
    ):
        name = str(basis_name)
        condition_limit = float(maximum_condition_number)
        if not name:
            raise ValueError("Moment basis_name must be non-empty.")
        if not np.isfinite(condition_limit) or condition_limit <= 1.0:
            raise ValueError(
                "maximum_condition_number must be finite and greater than one."
            )
        exponents_ = None
        if exponents is not None:
            exponents_ = tuple(tuple(int(value) for value in row) for row in exponents)
            if not exponents_ or any(value < 0 for row in exponents_ for value in row):
                raise ValueError(
                    "Moment exponents must be a non-empty nonnegative table."
                )
            if len(set(exponents_)) != len(exponents_):
                raise ValueError("Moment exponents must be unique.")
        self.basis_name = name
        self.exponents = exponents_
        self.maximum_condition_number = condition_limit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-moment-basis-plan",
                "basis_name": name,
                "exponents": exponents_,
                "maximum_condition_number": condition_limit,
            }
        )

    def prepare(
        self,
        velocity_set: LatticeBoltzmannVelocitySet,
        precision: LatticeBoltzmannPrecisionPolicy,
        /,
    ) -> PreparedMomentBasis:
        if not isinstance(velocity_set, LatticeBoltzmannVelocitySet):
            raise TypeError("velocity_set must be LatticeBoltzmannVelocitySet.")
        if not isinstance(precision, LatticeBoltzmannPrecisionPolicy):
            raise TypeError("precision must be LatticeBoltzmannPrecisionPolicy.")
        exponents = (
            _default_exponents(velocity_set) if self.exponents is None else self.exponents
        )
        q = velocity_set.population_count
        dimension = velocity_set.dimension
        if len(exponents) != q or any(len(row) != dimension for row in exponents):
            raise ValueError("Moment exponents must have shape (Q, dimension).")
        exponent_lookup = {row: index for index, row in enumerate(exponents)}
        zero = (0,) * dimension
        units = tuple(
            tuple(1 if axis == component else 0 for axis in range(dimension))
            for component in range(dimension)
        )
        if zero not in exponent_lookup or any(
            unit not in exponent_lookup for unit in units
        ):
            raise ValueError(
                "Moment basis must explicitly contain density and momentum modes."
            )

        velocities = np.asarray(velocity_set.velocities, dtype=np.float64)
        transform = np.prod(
            velocities[None, :, :] ** np.asarray(exponents)[:, None, :], axis=-1
        )
        rank = int(np.linalg.matrix_rank(transform))
        condition_number = float(np.linalg.cond(transform))
        if rank != q or not np.isfinite(condition_number):
            raise ValueError("Moment transform must be finite and exactly invertible.")
        if condition_number > self.maximum_condition_number:
            raise ValueError(
                "Moment transform exceeds the certified condition-number limit."
            )
        inverse = np.linalg.inv(transform)
        identity_residual = np.max(np.abs(transform @ inverse - np.eye(q)))
        tolerance = 1.0e-11 if precision.certification_dtype == "float64" else 2.0e-5
        if not np.isfinite(identity_residual) or identity_residual > tolerance:
            raise ValueError(
                "Moment inverse failed certification in the requested precision."
            )

        shift_coefficients = np.zeros((q, q), dtype=np.float64)
        shift_exponents = np.zeros((q, q, dimension), dtype=np.int32)
        cumulant_terms: list[tuple[tuple[int, int, int], ...]] = []
        for alpha in exponents:
            alpha_terms: list[tuple[int, int, int]] = []
            for beta, beta_index in exponent_lookup.items():
                if any(beta[axis] > alpha[axis] for axis in range(dimension)):
                    continue
                coefficient = int(
                    np.prod([comb(alpha[a], beta[a]) for a in range(dimension)])
                )
                alpha_index = exponent_lookup[alpha]
                shift_coefficients[alpha_index, beta_index] = coefficient
                shift_exponents[alpha_index, beta_index] = np.subtract(alpha, beta)
                if beta == zero or beta == alpha:
                    continue
                active_axis = next(axis for axis, value in enumerate(alpha) if value)
                if beta[active_axis] == 0:
                    continue
                remainder = tuple(alpha[a] - beta[a] for a in range(dimension))
                remainder_index = exponent_lookup.get(remainder)
                if remainder_index is None:
                    raise ValueError(
                        "Cumulant-capable moment exponents must be downward closed."
                    )
                recurrence_coefficient = int(
                    np.prod(
                        [
                            comb(
                                alpha[a] - (1 if a == active_axis else 0),
                                beta[a] - (1 if a == active_axis else 0),
                            )
                            for a in range(dimension)
                        ]
                    )
                )
                alpha_terms.append((beta_index, remainder_index, recurrence_coefficient))
            cumulant_terms.append(tuple(alpha_terms))

        conserved = (exponent_lookup[zero],) + tuple(
            exponent_lookup[unit] for unit in units
        )
        second_order = tuple(
            i for i, exponent in enumerate(exponents) if sum(exponent) == 2
        )
        cumulant_order = tuple(
            sorted(
                (i for i, exponent in enumerate(exponents) if exponent != zero),
                key=lambda i: (sum(exponents[i]), exponents[i]),
            )
        )
        basis_id = canonical_fingerprint(
            {
                "kind": "prepared-lattice-boltzmann-moment-basis",
                "plan": self.plan_id,
                "lattice": velocity_set.lattice_id,
                "exponents": exponents,
                "transform": transform.tolist(),
            }
        )
        coefficient_dtype = jnp.dtype(precision.compute_dtype)
        return PreparedMomentBasis(
            transform=jnp.asarray(transform, dtype=coefficient_dtype),
            inverse_transform=jnp.asarray(inverse, dtype=coefficient_dtype),
            exponents_array=jnp.asarray(exponents, dtype=jnp.int32),
            shift_coefficients=jnp.asarray(shift_coefficients, dtype=coefficient_dtype),
            shift_exponents=jnp.asarray(shift_exponents, dtype=jnp.int32),
            exponents=exponents,
            conserved_indices=conserved,
            second_order_indices=second_order,
            cumulant_order=cumulant_order,
            cumulant_terms=tuple(cumulant_terms),
            lattice_id=velocity_set.lattice_id,
            basis_name=self.basis_name,
            basis_id=basis_id,
        )


class PreparedRelaxationSpectrum(StrictModule, NonTrainableState):
    """Certified MRT spectrum with runtime viscosity modes represented by a mask."""

    base_rates: Array
    conserved_mask: Array
    shear_mask: Array
    basis_id: str = eqx.field(static=True)
    spectrum_id: str = eqx.field(static=True)

    def relaxation_rates(self, shear_rate: ArrayLike, /) -> Array:
        shear = jnp.asarray(shear_rate, dtype=self.base_rates.dtype)
        rates = jnp.where(self.shear_mask, shear[..., None], self.base_rates)
        return jnp.where(self.conserved_mask, 0.0, rates)


class RelaxationSpectrumPlan(StrictModule, NonTrainableState):
    """Plan a diagonal moment spectrum; all admissibility is checked at prepare."""

    default_rate: float = eqx.field(static=True)
    rates: tuple[float, ...] | None = eqx.field(static=True)
    shear_rate_indices: tuple[int, ...] | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        rates: Sequence[float] | None = None,
        /,
        *,
        default_rate: float = 1.0,
        shear_rate_indices: Sequence[int] | None = None,
    ):
        default = float(default_rate)
        if not np.isfinite(default) or default <= 0.0 or default >= 2.0:
            raise ValueError("default_rate must lie strictly between zero and two.")
        rates_ = None if rates is None else tuple(float(value) for value in rates)
        if rates_ is not None and any(
            not np.isfinite(value) or value < 0.0 or value >= 2.0 for value in rates_
        ):
            raise ValueError("Every prepared relaxation rate must lie in [0, 2).")
        shear = (
            None
            if shear_rate_indices is None
            else tuple(int(i) for i in shear_rate_indices)
        )
        if shear is not None and (
            len(set(shear)) != len(shear) or any(i < 0 for i in shear)
        ):
            raise ValueError("shear_rate_indices must be unique nonnegative indices.")
        self.default_rate = default
        self.rates = rates_
        self.shear_rate_indices = shear
        self.plan_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-relaxation-spectrum-plan",
                "default_rate": default,
                "rates": rates_,
                "shear_rate_indices": shear,
            }
        )

    def prepare(self, basis: PreparedMomentBasis, /) -> PreparedRelaxationSpectrum:
        if not isinstance(basis, PreparedMomentBasis):
            raise TypeError("basis must be PreparedMomentBasis.")
        q = basis.population_count
        conserved = set(basis.conserved_indices)
        shear = set(
            basis.second_order_indices
            if self.shear_rate_indices is None
            else self.shear_rate_indices
        )
        if any(index >= q for index in shear):
            raise ValueError("A shear-rate index lies outside the moment basis.")
        if conserved & shear:
            raise ValueError("Conserved moments cannot be viscosity-controlled modes.")
        if not shear:
            raise ValueError("At least one viscosity-controlled mode is required.")
        if self.rates is None:
            rates = np.full((q,), self.default_rate, dtype=np.float64)
            rates[list(conserved)] = 0.0
        else:
            if len(self.rates) != q:
                raise ValueError("Explicit relaxation spectrum must have length Q.")
            rates = np.asarray(self.rates, dtype=np.float64)
            if np.any(rates[list(conserved)] != 0.0):
                raise ValueError("Explicit conserved relaxation rates must be zero.")
        conserved_mask = np.zeros((q,), dtype=bool)
        conserved_mask[list(conserved)] = True
        shear_mask = np.zeros((q,), dtype=bool)
        shear_mask[list(shear)] = True
        spectrum_id = canonical_fingerprint(
            {
                "kind": "prepared-lattice-boltzmann-relaxation-spectrum",
                "plan": self.plan_id,
                "basis": basis.basis_id,
                "base_rates": rates.tolist(),
                "conserved": sorted(conserved),
                "shear": sorted(shear),
            }
        )
        return PreparedRelaxationSpectrum(
            base_rates=jnp.asarray(rates, dtype=basis.transform.dtype),
            conserved_mask=jnp.asarray(conserved_mask),
            shear_mask=jnp.asarray(shear_mask),
            basis_id=basis.basis_id,
            spectrum_id=spectrum_id,
        )


def raw_moments(
    populations: Array,
    basis: PreparedMomentBasis,
    precision: LatticeBoltzmannPrecisionPolicy,
    /,
) -> Array:
    values = precision.accumulation(populations)
    return precision.compute(
        oe.contract(
            "...q,mq->...m", values, jnp.asarray(basis.transform, dtype=values.dtype)
        )
    )


def populations_from_raw_moments(
    moments: Array,
    basis: PreparedMomentBasis,
    precision: LatticeBoltzmannPrecisionPolicy,
    /,
) -> Array:
    return precision.compute(
        oe.contract(
            "qm,...m->...q",
            precision.coefficient(basis.inverse_transform),
            precision.compute(moments),
        )
    )


def central_moments(
    populations: Array,
    velocity: Array,
    velocity_set: LatticeBoltzmannVelocitySet,
    basis: PreparedMomentBasis,
    precision: LatticeBoltzmannPrecisionPolicy,
    /,
) -> Array:
    basis.require_lattice(velocity_set)
    values = precision.accumulation(populations)
    u = precision.accumulation(velocity)
    c = precision.accumulation(velocity_set.velocities)
    leading = (1,) * (u.ndim - 1)
    centered = c.reshape(leading + c.shape) - u[..., None, :]
    exponents = basis.exponents_array.reshape(
        leading + (basis.population_count, 1, basis.dimension)
    )
    monomials = jnp.prod(centered[..., None, :, :] ** exponents, axis=-1)
    return precision.compute(oe.contract("...q,...mq->...m", values, monomials))


def _shift_moments(
    moments: Array, velocity: Array, basis: PreparedMomentBasis, sign: float, /
) -> Array:
    values = jnp.asarray(moments)
    u = jnp.asarray(velocity, dtype=values.dtype) * sign
    differences = basis.shift_exponents
    powers = jnp.prod(
        u[..., None, None, :]
        ** differences.reshape((1,) * (u.ndim - 1) + differences.shape),
        axis=-1,
    )
    return oe.contract(
        "ab,...ab,...b->...a",
        jnp.asarray(basis.shift_coefficients, dtype=values.dtype),
        powers,
        values,
    )


def populations_from_central_moments(
    moments: Array,
    velocity: Array,
    basis: PreparedMomentBasis,
    precision: LatticeBoltzmannPrecisionPolicy,
    /,
) -> Array:
    return populations_from_raw_moments(
        _shift_moments(precision.compute(moments), velocity, basis, 1.0), basis, precision
    )


def cumulants_from_central_moments(
    moments: Array, basis: PreparedMomentBasis, /
) -> Array:
    values = jnp.asarray(moments)
    density_index = basis.conserved_indices[0]
    density = values[..., density_index]
    normalized = values / jnp.where(density > 0.0, density, 1.0)[..., None]
    cumulants = jnp.zeros_like(normalized).at[..., density_index].set(density)
    for index in basis.cumulant_order:
        correction = jnp.zeros_like(density)
        for beta, remainder, coefficient in basis.cumulant_terms[index]:
            correction = (
                correction
                + coefficient * cumulants[..., beta] * normalized[..., remainder]
            )
        cumulants = cumulants.at[..., index].set(normalized[..., index] - correction)
    return cumulants


def central_moments_from_cumulants(
    cumulants: Array, basis: PreparedMomentBasis, /
) -> Array:
    values = jnp.asarray(cumulants)
    density_index = basis.conserved_indices[0]
    density = values[..., density_index]
    normalized = jnp.zeros_like(values).at[..., density_index].set(1.0)
    for index in basis.cumulant_order:
        correction = jnp.zeros_like(density)
        for beta, remainder, coefficient in basis.cumulant_terms[index]:
            correction = (
                correction + coefficient * values[..., beta] * normalized[..., remainder]
            )
        normalized = normalized.at[..., index].set(values[..., index] + correction)
    return normalized * density[..., None]


__all__ = [
    "MomentBasisPlan",
    "PreparedMomentBasis",
    "PreparedRelaxationSpectrum",
    "RelaxationSpectrumPlan",
    "central_moments",
    "central_moments_from_cumulants",
    "cumulants_from_central_moments",
    "populations_from_central_moments",
    "populations_from_raw_moments",
    "raw_moments",
]
