#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from math import prod
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jax.typing import DTypeLike
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


VelocityTransportKind: TypeAlias = Literal["integer_lattice", "off_lattice"]


def _double_factorial(value: int, /) -> int:
    if value <= 0:
        return 1
    return prod(range(value, 0, -2))


def _gaussian_moment(exponents: tuple[int, ...], temperature: float, /) -> float:
    if any(exponent % 2 for exponent in exponents):
        return 0.0
    return float(
        prod(
            _double_factorial(exponent - 1) * temperature ** (exponent // 2)
            for exponent in exponents
        )
    )


def _monomial_exponents(
    dimension: int, maximum_degree: int, /
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        exponents
        for exponents in product(range(maximum_degree + 1), repeat=dimension)
        if sum(exponents) <= maximum_degree
    )


class QuadratureMomentCertification(StrictModule, NonTrainableState):
    """Numerical evidence for centered-Maxwellian polynomial exactness."""

    exponents: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    expected_moments: Array
    measured_moments: Array
    absolute_residuals: Array
    maximum_degree: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    maximum_residual: float = eqx.field(static=True)
    normalized: bool = eqx.field(static=True)
    positive_weights: bool = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    certification_id: str = eqx.field(static=True)

    def __init__(
        self,
        exponents: Sequence[Sequence[int]],
        expected_moments: ArrayLike,
        measured_moments: ArrayLike,
        /,
        *,
        maximum_degree: int,
        tolerance: float,
        positive_weights: bool,
    ):
        exponents_ = tuple(tuple(int(value) for value in row) for row in exponents)
        expected_host = np.asarray(expected_moments)
        measured_host = np.asarray(measured_moments, dtype=expected_host.dtype)
        residuals_host = np.abs(measured_host - expected_host)
        expected = jnp.asarray(expected_host)
        measured = jnp.asarray(measured_host)
        residuals = jnp.asarray(residuals_host)
        maximum_residual = float(np.max(residuals_host))
        tolerance_ = float(tolerance)
        normalized = bool(abs(float(measured_host[0]) - 1.0) <= tolerance_)
        passed = bool(positive_weights and maximum_residual <= tolerance_)
        self.exponents = exponents_
        self.expected_moments = expected
        self.measured_moments = measured
        self.absolute_residuals = residuals
        self.maximum_degree = int(maximum_degree)
        self.tolerance = tolerance_
        self.maximum_residual = maximum_residual
        self.normalized = normalized
        self.positive_weights = bool(positive_weights)
        self.passed = passed
        self.certification_id = canonical_fingerprint(
            {
                "kind": "centered-maxwellian-quadrature-certification-v1",
                "exponents": [list(row) for row in exponents_],
                "expected": array_tree_fingerprint(expected_host),
                "measured": array_tree_fingerprint(measured_host),
                "maximum_degree": int(maximum_degree),
                "tolerance": tolerance_,
                "positive_weights": bool(positive_weights),
                "passed": passed,
            }
        )


class CertifiedDiscreteVelocityQuadrature(StrictModule, NonTrainableState):
    """Prepared velocity quadrature with trailing-population conventions.

    Velocities have shape ``(Q, D)`` and every population field has a trailing
    ``Q`` axis. ``transport_kind`` is certified separately from moment
    exactness so an off-lattice rule can never be presented as exact streaming.
    """

    velocities: Array
    weights: Array
    reference_temperature: float = eqx.field(static=True)
    transport_kind: VelocityTransportKind = eqx.field(static=True)
    name: str = eqx.field(static=True)
    certification: QuadratureMomentCertification
    dimension: int = eqx.field(static=True)
    population_count: int = eqx.field(static=True)
    quadrature_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        velocities: ArrayLike,
        weights: ArrayLike,
        /,
        *,
        reference_temperature: float,
        certified_degree: int,
        transport_kind: VelocityTransportKind,
        tolerance: float = 5e-12,
    ):
        name_ = str(name)
        velocity_values = np.asarray(velocities)
        weight_values = np.asarray(weights)
        temperature = float(reference_temperature)
        degree = int(certified_degree)
        tolerance_ = float(tolerance)
        if not name_:
            raise ValueError("Discrete-velocity quadrature name must be non-empty.")
        if (
            velocity_values.ndim != 2
            or velocity_values.shape[0] == 0
            or velocity_values.shape[1] == 0
        ):
            raise ValueError("Discrete velocities must have non-empty shape (Q, D).")
        if weight_values.shape != (velocity_values.shape[0],):
            raise ValueError("Quadrature weights must have shape (Q,).")
        if not np.issubdtype(velocity_values.dtype, np.number) or not np.issubdtype(
            weight_values.dtype, np.number
        ):
            raise TypeError("Quadrature velocities and weights must be numeric.")
        dtype = np.result_type(velocity_values.dtype, weight_values.dtype, np.float64)
        velocity_values = velocity_values.astype(dtype, copy=False)
        weight_values = weight_values.astype(dtype, copy=False)
        if (
            np.any(~np.isfinite(velocity_values))
            or np.any(~np.isfinite(weight_values))
            or np.any(weight_values <= 0.0)
            or not np.isfinite(temperature)
            or temperature <= 0.0
            or degree < 0
            or not np.isfinite(tolerance_)
            or tolerance_ <= 0.0
        ):
            raise ValueError(
                "Quadrature data, temperature, degree, or tolerance is invalid."
            )
        if transport_kind not in ("integer_lattice", "off_lattice"):
            raise ValueError("Unknown discrete-velocity transport kind.")
        integer_residual = float(
            np.max(np.abs(velocity_values - np.rint(velocity_values)))
        )
        if transport_kind == "integer_lattice" and integer_residual > tolerance_:
            raise ValueError(
                "integer_lattice transport requires integer velocities; use off_lattice explicitly."
            )
        exponents = _monomial_exponents(velocity_values.shape[1], degree)
        measured = np.asarray(
            [
                np.sum(
                    weight_values
                    * np.prod(velocity_values ** np.asarray(row)[None, :], axis=1)
                )
                for row in exponents
            ],
            dtype=dtype,
        )
        expected = np.asarray(
            [_gaussian_moment(row, temperature) for row in exponents], dtype=dtype
        )
        certification = QuadratureMomentCertification(
            exponents,
            expected,
            measured,
            maximum_degree=degree,
            tolerance=tolerance_,
            positive_weights=bool(np.all(weight_values > 0.0)),
        )
        if not certification.passed:
            raise ValueError(
                "Discrete-velocity quadrature failed centered-Maxwellian moment "
                f"certification: maximum residual {certification.maximum_residual:.6e} "
                f"exceeds {tolerance_:.6e}."
            )
        self.velocities = jnp.asarray(velocity_values)
        self.weights = jnp.asarray(weight_values)
        self.reference_temperature = temperature
        self.transport_kind = transport_kind
        self.name = name_
        self.certification = certification
        self.dimension = int(velocity_values.shape[1])
        self.population_count = int(velocity_values.shape[0])
        self.quadrature_id = canonical_fingerprint(
            {
                "kind": "certified-discrete-velocity-quadrature-v1",
                "name": name_,
                "velocities": array_tree_fingerprint(velocity_values),
                "weights": array_tree_fingerprint(weight_values),
                "reference_temperature": temperature,
                "transport_kind": transport_kind,
                "certification": certification.certification_id,
            }
        )

    def validate_populations(self, populations: ArrayLike, /) -> Array:
        values = jnp.asarray(populations)
        if values.ndim == 0 or values.shape[-1] != self.population_count:
            raise ValueError(
                "Discrete-velocity populations must have trailing shape "
                f"({self.population_count},)."
            )
        return values

    def raw_moment(self, populations: ArrayLike, exponents: Sequence[int], /) -> Array:
        values = self.validate_populations(populations)
        powers = tuple(int(value) for value in exponents)
        if len(powers) != self.dimension or any(value < 0 for value in powers):
            raise ValueError("Moment exponents must be non-negative with length D.")
        monomial = jnp.prod(
            self.velocities ** jnp.asarray(powers, dtype=self.velocities.dtype)[None, :],
            axis=-1,
        )
        return oe.contract("...q,q->...", values, monomial)

    def hydrodynamic_moment_matrix(
        self, /, *, include_total_energy: bool = True
    ) -> Array:
        rows = [jnp.ones((self.population_count,), dtype=self.velocities.dtype)]
        rows.extend(self.velocities[:, axis] for axis in range(self.dimension))
        if include_total_energy:
            rows.append(0.5 * oe.contract("qd,qd->q", self.velocities, self.velocities))
        return jnp.stack(rows, axis=0)


def d2v17_quadrature(
    *, dtype: DTypeLike = jnp.float64
) -> CertifiedDiscreteVelocityQuadrature:
    """Return the fourth-degree integer-lattice D2V17 reference rule."""

    shells = (
        ((0, 0),),
        ((1, 0), (-1, 0), (0, 1), (0, -1)),
        ((1, 1), (1, -1), (-1, 1), (-1, -1)),
        ((2, 0), (-2, 0), (0, 2), (0, -2)),
        ((2, 2), (2, -2), (-2, 2), (-2, -2)),
    )
    shell_weights = (43.0 / 128.0, 5.0 / 48.0, 5.0 / 96.0, 7.0 / 768.0, 1.0 / 1536.0)
    velocities = np.asarray(
        tuple(point for shell in shells for point in shell), dtype=dtype
    )
    weights = np.asarray(
        tuple(
            weight
            for shell, weight in zip(shells, shell_weights, strict=True)
            for _ in shell
        ),
        dtype=dtype,
    )
    return CertifiedDiscreteVelocityQuadrature(
        "D2V17",
        velocities,
        weights,
        reference_temperature=0.5,
        certified_degree=4,
        transport_kind="integer_lattice",
    )


def d2v37_off_lattice_quadrature(
    *, dtype: DTypeLike = jnp.float64
) -> CertifiedDiscreteVelocityQuadrature:
    """Return the fourth-degree D2V37 rule with explicit off-lattice scaling."""

    scale = 1.1969797703930744
    shells = (
        ((0, 0),),
        ((1, 0), (-1, 0), (0, 1), (0, -1)),
        ((1, 1), (1, -1), (-1, 1), (-1, -1)),
        ((2, 0), (-2, 0), (0, 2), (0, -2)),
        ((2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)),
        ((2, 2), (2, -2), (-2, 2), (-2, -2)),
        ((3, 0), (-3, 0), (0, 3), (0, -3)),
        ((3, 1), (3, -1), (-3, 1), (-3, -1), (1, 3), (1, -3), (-1, 3), (-1, -3)),
    )
    shell_weights = (
        0.2331506691323525,
        0.1073060915422190,
        0.05766785988879488,
        0.01420821615845075,
        0.005353049000513775,
        0.001011937592673576,
        0.0002453010277577173,
        0.0002834142529941982,
    )
    velocities = scale * np.asarray(
        tuple(point for shell in shells for point in shell), dtype=dtype
    )
    weights = np.asarray(
        tuple(
            weight
            for shell, weight in zip(shells, shell_weights, strict=True)
            for _ in shell
        ),
        dtype=dtype,
    )
    return CertifiedDiscreteVelocityQuadrature(
        "D2V37-off-lattice",
        velocities,
        weights,
        reference_temperature=1.0,
        certified_degree=4,
        transport_kind="off_lattice",
        tolerance=2e-12,
    )


__all__ = [
    "CertifiedDiscreteVelocityQuadrature",
    "QuadratureMomentCertification",
    "VelocityTransportKind",
    "d2v17_quadrature",
    "d2v37_off_lattice_quadrature",
]
