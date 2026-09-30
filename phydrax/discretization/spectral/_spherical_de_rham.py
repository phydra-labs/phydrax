#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ...exterior._complex import AbstractDeRhamComplex, ComplexBoundary
from ...exterior._form_type import FormTwist
from ...linalg._complexes import (
    HarmonicSubspace,
    HilbertComplex,
    HodgeDecomposition,
    HodgeDecompositionPolicy,
)
from ...linalg._operators import FunctionLinearOperator
from ...linalg._pairings import DiagonalPairing
from ...linalg._spaces import ArraySpace
from ...typing import Bool, Complex, Dim, Float, Int32, Size
from ._complex_support import (
    analytic_decomposition_policy,
    closed_boundary,
    decomposition_result,
    harmonic_projection,
    riesz_operator,
)
from ._spherical import SphericalSpectralDiscretization
from ._spherical_vector import PreparedSphericalVectorOperators


class SphericalScalarCoordinateDim(Dim):
    """Independent real scalar harmonic coordinates."""


class SphericalVectorModeDim(Dim):
    """Positive-degree scalar harmonics supporting tangent modes."""


@final
class SphericalDeRhamComplex(AbstractDeRhamComplex):
    """Real spin realization of all form degrees on an oriented round sphere.

    Scalars/densities use independent real orthonormal harmonic coordinates.
    One-forms use interleaved normalized poloidal/toroidal coordinates for
    ell>=1. Physical tangent synthesis and analysis compose the existing
    prepared spin-one vector owner. Invalid padded transform modes and the
    absent ell=0 vector modes are not coordinates of the Hilbert complex.
    """

    __strict_contract__ = True

    space: SphericalSpectralDiscretization
    vector_operators: PreparedSphericalVectorOperators
    scalar_indices: Int32[SphericalScalarCoordinateDim]
    conjugate_indices: Int32[SphericalScalarCoordinateDim]
    scalar_factors: Complex[SphericalScalarCoordinateDim]
    conjugate_factors: Complex[SphericalScalarCoordinateDim]
    imaginary_coordinates: Bool[SphericalScalarCoordinateDim]
    derivative_scale: Float[SphericalVectorModeDim]
    _complex: HilbertComplex
    _realization_id: str = eqx.field(static=True)
    scalar_count: Size[SphericalScalarCoordinateDim] = eqx.field(static=True)

    def __init__(self, space: SphericalSpectralDiscretization, /) -> None:
        if not isinstance(space, SphericalSpectralDiscretization):
            raise TypeError("space must be a prepared spherical spectral discretization.")
        vector = PreparedSphericalVectorOperators(space)
        limit = space.layout.bandlimit
        width = space.coefficient_shape[1]
        indices, conjugates, factors, conjugate_factors, imaginary, degrees = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for ell in range(limit):
            for order in range(ell + 1):
                for sine in range(1 if order == 0 else 2):
                    indices.append(ell * width + limit - 1 + order)
                    conjugates.append(ell * width + limit - 1 - order)
                    factor = 1.0 if order == 0 else (1j if sine else 1.0) / np.sqrt(2.0)
                    factors.append(factor)
                    conjugate_factors.append(
                        0.0 if order == 0 else (-1) ** order * np.conj(factor)
                    )
                    imaginary.append(bool(sine))
                    degrees.append(ell)
        dtype = jnp.dtype(space.plan.precision.physical_dtype)
        coefficient_dtype = jnp.dtype(space.plan.precision.coefficient_dtype)
        scale = (
            jnp.sqrt(
                jnp.asarray(degrees[1:], dtype=dtype)
                * (jnp.asarray(degrees[1:], dtype=dtype) + 1.0)
            )
            / space.radius
        )
        identifier = canonical_fingerprint(
            {
                "kind": "spherical-de-rham",
                "space": space.prepared_id,
                "vector": vector.operator_id,
            }
        )
        count = limit**2
        sizes = (count, 2 * (count - 1), count)
        spaces = tuple(
            ArraySpace(
                (size,),
                dtype=dtype,
                pairing=DiagonalPairing(
                    jnp.full((size,), space.radius**2, dtype=dtype),
                    pairing_id=f"{identifier}:parseval:{degree}",
                ),
                space_id=f"{identifier}:degree:{degree}",
            )
            for degree, size in enumerate(sizes)
        )

        def gradient(value: Array) -> Array:
            return jnp.stack(
                (scale * value[1:], jnp.zeros_like(value[1:])), axis=-1
            ).reshape((-1,))

        def gradient_transpose(value: Array) -> Array:
            return jnp.concatenate(
                (jnp.zeros((1,), dtype=value.dtype), scale * value.reshape((-1, 2))[:, 0])
            )

        def curl(value: Array) -> Array:
            return jnp.concatenate(
                (
                    jnp.zeros((1,), dtype=value.dtype),
                    -scale * value.reshape((-1, 2))[:, 1],
                )
            )

        def curl_transpose(value: Array) -> Array:
            return jnp.stack(
                (jnp.zeros_like(value[1:]), -scale * value[1:]), axis=-1
            ).reshape((-1,))

        differentials = (
            FunctionLinearOperator(
                gradient,
                source=spaces[0],
                target=spaces[1],
                transpose_action=gradient_transpose,
                operator_id=f"{identifier}:d:0",
            ),
            FunctionLinearOperator(
                curl,
                source=spaces[1],
                target=spaces[2],
                transpose_action=curl_transpose,
                operator_id=f"{identifier}:d:1",
            ),
        )
        complex_ = HilbertComplex(spaces, differentials, complex_id=identifier)
        self.space = space
        self.vector_operators = vector
        self.scalar_indices = jnp.asarray(indices, dtype=jnp.int32)
        self.conjugate_indices = jnp.asarray(conjugates, dtype=jnp.int32)
        self.scalar_factors = jnp.asarray(factors, dtype=coefficient_dtype)
        self.conjugate_factors = jnp.asarray(conjugate_factors, dtype=coefficient_dtype)
        self.imaginary_coordinates = jnp.asarray(imaginary, dtype=jnp.bool_)
        self.derivative_scale = scale
        self._complex = complex_
        self._realization_id = identifier
        self.scalar_count = count

    @property
    def dimension(self) -> int:
        return 2

    @property
    def primal_twist(self) -> FormTwist:
        return "untwisted"

    @property
    def realization_id(self) -> str:
        return self._realization_id

    @property
    def betti_numbers(self) -> tuple[int, int, int]:
        return (1, 0, 1)

    def hilbert_complex(
        self, *, boundary: ComplexBoundary = "absolute"
    ) -> HilbertComplex:
        closed_boundary(boundary)
        return self._complex

    def scalar_to_modal(self, values: ArrayLike, /) -> Array:
        value = self._complex.space(0).validate(values)
        modal = jnp.zeros(
            (prod(self.space.coefficient_shape),), dtype=self.scalar_factors.dtype
        )
        modal = modal.at[self.scalar_indices].add(self.scalar_factors * value)
        modal = modal.at[self.conjugate_indices].add(self.conjugate_factors * value)
        return modal.reshape(self.space.coefficient_shape)

    def scalar_from_modal(self, coefficients: ArrayLike, /) -> Array:
        modal = self.vector_operators._scalar(coefficients)
        if modal.shape != self.space.coefficient_shape:
            raise ValueError(
                "Scalar modal coordinates must have the prepared coefficient shape."
            )
        selected = modal.reshape((-1,))[self.scalar_indices]
        multiplier = jnp.where(
            self.scalar_indices == self.conjugate_indices, 1.0, jnp.sqrt(2.0)
        )
        return (
            multiplier
            * jnp.where(
                self.imaginary_coordinates, jnp.imag(selected), jnp.real(selected)
            )
        ).astype(self.derivative_scale.dtype)

    def to_physical(self, degree: int, values: ArrayLike, /) -> Array:
        value = self._complex.space(degree).validate(values)
        match degree:
            case 0 | 2:
                return self.space.reconstruct(self.scalar_to_modal(value))
            case 1:
                components = value.reshape((-1, 2))
                zero = jnp.zeros((1,), dtype=value.dtype)
                chi = jnp.concatenate((zero, components[:, 0] / self.derivative_scale))
                psi = jnp.concatenate((zero, components[:, 1] / self.derivative_scale))
                east, north = self.vector_operators.gradient(self.scalar_to_modal(chi))
                stream_east, stream_north = self.vector_operators.gradient(
                    self.scalar_to_modal(psi)
                )
                return jnp.stack((east - stream_north, north + stream_east), axis=-1)
            case _:
                raise ValueError("Spherical form degree must lie in [0, 2].")

    def from_physical(self, degree: int, values: ArrayLike, /) -> Array:
        self.form_type(degree)
        value = jnp.asarray(values)
        match degree:
            case 0 | 2:
                return self.scalar_from_modal(self.space.project(value))
            case 1:
                if value.shape != self.space.sample_shape + (2,):
                    raise ValueError(
                        "Tangent values must have the spherical sample shape plus east/north components."
                    )
                divergence = self.scalar_from_modal(
                    self.vector_operators.divergence(value[..., 0], value[..., 1])
                )
                curl = self.scalar_from_modal(
                    self.vector_operators.curl(value[..., 0], value[..., 1])
                )
                return jnp.stack(
                    (
                        -divergence[1:] / self.derivative_scale,
                        -curl[1:] / self.derivative_scale,
                    ),
                    axis=-1,
                ).reshape((-1,))
            case _:
                raise ValueError("Spherical form degree must lie in [0, 2].")

    def hodge_operator(self, degree: int, /) -> FunctionLinearOperator:
        space = self._complex.space(degree)
        if not isinstance(space, ArraySpace):
            raise TypeError("Spherical forms require an array coordinate space.")
        return riesz_operator(space, f"{self.realization_id}:hodge:{degree}")

    def hodge_diagonal(self, degree: int, /) -> Array:
        return jnp.full(
            (self._complex.space(degree).size,),
            self.space.radius**2,
            dtype=self.derivative_scale.dtype,
        )

    def hodge_star(self, degree: int, values: ArrayLike, /) -> Array:
        return self._complex.space(degree).riesz(values)

    def inverse_hodge_star(self, degree: int, values: ArrayLike, /) -> Array:
        return self._complex.space(degree).inverse_riesz(values)

    def metric_star(self, degree: int, values: ArrayLike, /) -> Array:
        """Metric star in harmonic coordinates using outward sphere orientation."""
        value = self._complex.space(degree).validate(values)
        if degree == 1:
            components = value.reshape((-1, 2))
            return jnp.stack((-components[:, 1], components[:, 0]), axis=-1).reshape(
                (-1,)
            )
        return value

    def harmonic_basis(self, degree: int, /) -> Array:
        space = self._complex.space(degree)
        if degree == 1:
            return jnp.zeros((space.size, 0), dtype=self.derivative_scale.dtype)
        return (
            jnp.zeros((space.size, 1), dtype=self.derivative_scale.dtype)
            .at[0, 0]
            .set(1.0 / self.space.radius)
        )

    def hodge_decomposition(
        self,
        degree: int,
        values: ArrayLike,
        /,
        *,
        boundary: ComplexBoundary = "absolute",
        harmonic: HarmonicSubspace | None = None,
        lower_harmonic: HarmonicSubspace | None = None,
        policy: HodgeDecompositionPolicy | None = None,
    ) -> HodgeDecomposition:
        complex_ = self.hilbert_complex(boundary=boundary)
        analytic_decomposition_policy(policy)
        if degree == 0 and lower_harmonic is not None:
            raise ValueError("Degree-zero decomposition has no lower harmonic space.")
        space = complex_.space(degree)
        if not isinstance(space, ArraySpace):
            raise TypeError("Spherical forms require an array coordinate space.")
        value = space.validate(values)
        harmonic_part = jnp.zeros_like(value)
        if degree != 1:
            harmonic_part = harmonic_part.at[0].set(value[0])
        if harmonic is not None:
            value, harmonic_part = harmonic_projection(
                space,
                value,
                self.harmonic_basis(degree),
                harmonic,
                complex_.complex_id,
                degree,
            )
        exact, coexact = jnp.zeros_like(value), jnp.zeros_like(value)
        potential = None
        match degree:
            case 0:
                coexact = value - harmonic_part
            case 1:
                components = value.reshape((-1, 2))
                exact = jnp.stack(
                    (components[:, 0], jnp.zeros_like(components[:, 0])), axis=-1
                ).reshape((-1,))
                coexact = value - exact
                potential = jnp.concatenate(
                    (
                        jnp.zeros((1,), dtype=value.dtype),
                        components[:, 0] / self.derivative_scale,
                    )
                )
            case 2:
                exact = value - harmonic_part
                potential = jnp.stack(
                    (jnp.zeros_like(value[1:]), -value[1:] / self.derivative_scale),
                    axis=-1,
                ).reshape((-1,))
            case _:
                raise ValueError("Spherical form degree must lie in [0, 2].")
        if lower_harmonic is not None and potential is not None:
            lower_space = complex_.space(degree - 1)
            if not isinstance(lower_space, ArraySpace):
                raise TypeError("Spherical forms require an array coordinate space.")
            potential, gauge = harmonic_projection(
                lower_space,
                potential,
                self.harmonic_basis(degree - 1),
                lower_harmonic,
                complex_.complex_id,
                degree - 1,
            )
            potential = potential - gauge
        return decomposition_result(
            space, value, potential, exact, coexact, harmonic_part
        )


__all__ = ["SphericalDeRhamComplex"]
