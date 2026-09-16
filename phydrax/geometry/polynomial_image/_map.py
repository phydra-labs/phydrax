#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from fractions import Fraction
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...algebraic._exact import ExactSparsePolynomialSystem, QQ
from ...algebraic._system import SparsePolynomialSystem


class SparsePolynomialMap(StrictModule):
    """A sparse polynomial system interpreted as an affine coordinate map.

    ``exact_system`` is optional QQ provenance. When present it must describe the
    same support and numeric coefficients, so exact composition cannot silently
    prove a statement about a different map.
    """

    system: SparsePolynomialSystem
    exact_system: ExactSparsePolynomialSystem | None
    map_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: SparsePolynomialSystem,
        /,
        *,
        exact_system: ExactSparsePolynomialSystem | None = None,
    ):
        if not isinstance(system, SparsePolynomialSystem):
            raise TypeError("system must be a SparsePolynomialSystem.")
        if exact_system is not None:
            if not isinstance(exact_system, ExactSparsePolynomialSystem):
                raise TypeError(
                    "exact_system must be an ExactSparsePolynomialSystem or None."
                )
            if exact_system.domain.domain_id != QQ.domain_id:
                raise ValueError("Polynomial-map exact provenance must use QQ.")
            if exact_system.support.support_id != system.support.support_id:
                raise ValueError("Numeric and exact polynomial-map supports differ.")
            expected = jnp.asarray(
                [float(Fraction(value)) for value in exact_system.coefficients],
                dtype=system.coefficients.dtype,
            )
            if not np.array_equal(
                np.asarray(expected),
                np.asarray(system.coefficients),
            ):
                raise ValueError(
                    "Numeric polynomial-map coefficients disagree with exact QQ data."
                )
        self.system = system
        self.exact_system = exact_system
        self.map_id = canonical_fingerprint(
            {
                "kind": "sparse-polynomial-map-v1",
                "numeric_system": system.system_id,
                "exact_system": (
                    None if exact_system is None else exact_system.system_id
                ),
            }
        )

    @classmethod
    def from_exact(
        cls,
        system: ExactSparsePolynomialSystem,
        /,
        *,
        dtype: Any = float,
    ) -> SparsePolynomialMap:
        """Construct matching numerical evaluation data from one exact QQ system."""

        if not isinstance(system, ExactSparsePolynomialSystem):
            raise TypeError("system must be an ExactSparsePolynomialSystem.")
        if system.domain.domain_id != QQ.domain_id:
            raise ValueError("SparsePolynomialMap.from_exact requires QQ coefficients.")
        dtype_ = jnp.dtype(dtype)
        if not jnp.issubdtype(dtype_, jnp.inexact):
            raise TypeError("SparsePolynomialMap numeric dtype must be real or complex.")
        coefficients = jnp.asarray(
            [float(Fraction(value)) for value in system.coefficients],
            dtype=dtype_,
        )
        numeric = SparsePolynomialSystem(system.support, coefficients)
        return cls(numeric, exact_system=system)

    @property
    def source_dimension(self) -> int:
        return self.system.support.variable_count

    @property
    def target_dimension(self) -> int:
        return self.system.support.equation_count

    @property
    def source_labels(self) -> tuple[str, ...]:
        return self.system.support.variable_labels

    @property
    def target_labels(self) -> tuple[str, ...]:
        return self.system.support.equation_labels

    def evaluate(self, points: ArrayLike, /) -> Array:
        """Evaluate map coordinates at points with a trailing source axis."""

        return self.system.evaluate(points)

    def jacobian(self, points: ArrayLike, /) -> Array:
        """Evaluate map Jacobians with trailing ``(target, source)`` axes."""

        return self.system.jacobian(points)


__all__ = ["SparsePolynomialMap"]
