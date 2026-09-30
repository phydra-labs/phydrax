#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import combinations
from math import comb, prod
from typing import final, Literal

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
from ...typing import Dim, Float, Int32, parse, Size
from ._complex_support import (
    analytic_decomposition_policy,
    closed_boundary,
    decomposition_result,
    harmonic_projection,
    riesz_operator,
)
from ._incompressible import PeriodicLerayProjector
from ._space import TensorSpectralDiscretization


type FourierNyquistPolicy = Literal["zero-self-conjugate"]


class FourierModeDim(Dim):
    """Admissible Fourier modes."""


class FourierAxisDim(Dim):
    """Cartesian derivative axes."""


@final
class FourierDeRhamComplex(AbstractDeRhamComplex):
    """Orthonormal Fourier forms on a periodic Cartesian torus.

    Coordinates are flattened ``(active_modes, binomial(n, k))`` arrays. The
    active-mode map removes every Nyquist hyperplane, as does the owning Leray
    projector. Eliminating those coordinates, rather than annihilating their
    derivatives in a larger space, preserves the torus's true harmonic counts.
    Modal admission explicitly projects forbidden modes away. No distributed
    runtime is replaced; modal arrays retain the spectral owner's ordering.
    """

    __strict_contract__ = True

    space: TensorSpectralDiscretization
    active_modes: Int32[FourierModeDim]
    wavenumbers: Float[FourierModeDim, FourierAxisDim]
    inverse_squared: Float[FourierModeDim]
    _complex: HilbertComplex
    leray: PeriodicLerayProjector | None
    nyquist_policy: FourierNyquistPolicy = eqx.field(static=True)
    _dimension: Size[FourierAxisDim] = eqx.field(static=True)
    _realization_id: str = eqx.field(static=True)
    _bases: tuple[tuple[tuple[int, ...], ...], ...] = eqx.field(static=True)

    def __init__(
        self,
        space: TensorSpectralDiscretization,
        /,
        *,
        nyquist_policy: FourierNyquistPolicy,
    ) -> None:
        if not isinstance(space, TensorSpectralDiscretization):
            raise TypeError("space must be a prepared tensor spectral discretization.")
        if not space.axes or any(axis.family != "fourier" for axis in space.axes):
            raise ValueError("Fourier de Rham complexes require periodic Fourier axes.")
        policy = parse(nyquist_policy, FourierNyquistPolicy, "nyquist_policy")
        dimension = len(space.axes)
        dtype = jnp.dtype(space.plan.precision.coefficient_dtype)
        real_dtype = jnp.empty((), dtype=dtype).real.dtype
        active = np.ones(space.modal_shape, dtype=np.bool_)
        waves = []
        for index, axis in enumerate(space.axes):
            shape = [1] * dimension
            shape[index] = axis.mode_count
            active &= np.broadcast_to(
                ~np.asarray(axis.modes.nyquist_mask).reshape(tuple(shape)),
                space.modal_shape,
            )
            wave = (2.0 * jnp.pi * axis.modes.mode_numbers / axis.length).astype(
                real_dtype
            )
            waves.append(jnp.broadcast_to(wave.reshape(tuple(shape)), space.modal_shape))
        indices = jnp.asarray(np.flatnonzero(active.reshape((-1,))), dtype=jnp.int32)
        wave_array = jnp.stack(waves, axis=-1).reshape((-1, dimension))[indices]
        squared = jnp.sum(wave_array**2, axis=-1)
        identifier = canonical_fingerprint(
            {
                "kind": "fourier-de-rham",
                "space": space.prepared_id,
                "nyquist_policy": policy,
            }
        )
        bases = tuple(
            tuple(combinations(range(dimension), k)) for k in range(dimension + 1)
        )
        spaces = tuple(
            ArraySpace(
                (indices.size * comb(dimension, degree),),
                dtype=dtype,
                pairing=DiagonalPairing(
                    jnp.ones(
                        (indices.size * comb(dimension, degree),), dtype=wave_array.dtype
                    ),
                    pairing_id=f"{identifier}:parseval:{degree}",
                ),
                space_id=f"{identifier}:degree:{degree}",
            )
            for degree in range(dimension + 1)
        )
        differentials = []
        for degree in range(dimension):
            source_basis, target_basis = bases[degree], bases[degree + 1]
            routes = tuple(
                tuple(
                    (
                        position,
                        source_basis.index(blade[:position] + blade[position + 1 :]),
                        axis,
                    )
                    for position, axis in enumerate(blade)
                )
                for blade in target_basis
            )

            def differential(
                value: Array,
                routes: tuple[tuple[tuple[int, int, int], ...], ...] = routes,
                components: int = len(source_basis),
            ) -> Array:
                coefficients = value.reshape((indices.size, components))
                outputs = []
                for row in routes:
                    result = jnp.zeros((indices.size,), dtype=coefficients.dtype)
                    for position, component, axis in row:
                        result = (
                            result
                            + (-1) ** position
                            * 1j
                            * wave_array[:, axis]
                            * coefficients[:, component]
                        )
                    outputs.append(result)
                return jnp.stack(outputs, axis=-1).reshape((-1,))

            differentials.append(
                FunctionLinearOperator(
                    differential,
                    source=spaces[degree],
                    target=spaces[degree + 1],
                    operator_id=f"{identifier}:d:{degree}",
                )
            )
        complex_ = HilbertComplex(spaces, tuple(differentials), complex_id=identifier)
        leray = PeriodicLerayProjector(space) if dimension in (2, 3) else None
        self.space = space
        self.active_modes = indices
        self.wavenumbers = wave_array
        self.inverse_squared = jnp.where(
            squared > 0.0, 1.0 / jnp.where(squared > 0.0, squared, 1.0), 0.0
        )
        self._complex = complex_
        self.leray = leray
        self.nyquist_policy = policy
        self._dimension = dimension
        self._realization_id = identifier
        self._bases = bases

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def primal_twist(self) -> FormTwist:
        return "untwisted"

    @property
    def realization_id(self) -> str:
        return self._realization_id

    @property
    def betti_numbers(self) -> tuple[int, ...]:
        return tuple(comb(self.dimension, degree) for degree in range(self.dimension + 1))

    def hilbert_complex(
        self, *, boundary: ComplexBoundary = "absolute"
    ) -> HilbertComplex:
        closed_boundary(boundary)
        return self._complex

    def from_modal(self, degree: int, coefficients: ArrayLike, /) -> Array:
        """Admit modal component coefficients, projecting forbidden modes away."""
        components = self.form_type(degree).component_count
        value = jnp.asarray(coefficients)
        expected = self.space.modal_shape + (components,)
        if value.shape != expected:
            raise ValueError(
                f"Modal forms must have shape {expected}; got {value.shape}."
            )
        compact = value.reshape((prod(self.space.modal_shape), components))[
            self.active_modes
        ].reshape((-1,))
        return self._complex.space(degree).validate(compact)

    def to_modal(self, degree: int, values: ArrayLike, /) -> Array:
        components = self.form_type(degree).component_count
        value = self._complex.space(degree).validate(values)
        modal = jnp.zeros((prod(self.space.modal_shape), components), dtype=value.dtype)
        return (
            modal.at[self.active_modes]
            .set(value.reshape((-1, components)))
            .reshape(self.space.modal_shape + (components,))
        )

    def hodge_operator(self, degree: int, /) -> FunctionLinearOperator:
        space = self._complex.space(degree)
        if not isinstance(space, ArraySpace):
            raise TypeError("Fourier forms require an array coordinate space.")
        return riesz_operator(space, f"{self.realization_id}:hodge:{degree}")

    def hodge_diagonal(self, degree: int, /) -> Array:
        return jnp.ones((self._complex.space(degree).size,), dtype=self.wavenumbers.dtype)

    def hodge_star(self, degree: int, values: ArrayLike, /) -> Array:
        return self._complex.space(degree).riesz(values)

    def inverse_hodge_star(self, degree: int, values: ArrayLike, /) -> Array:
        return self._complex.space(degree).inverse_riesz(values)

    def metric_star(self, degree: int, values: ArrayLike, /) -> Array:
        """Smooth metric star in the global Cartesian frame (twist flips)."""
        value = (
            self._complex.space(degree)
            .validate(values)
            .reshape((-1, comb(self.dimension, degree)))
        )
        source, target = self._bases[degree], self._bases[self.dimension - degree]
        outputs = []
        for complement in target:
            blade = tuple(
                axis for axis in range(self.dimension) if axis not in complement
            )
            inversions = sum(left > right for left in blade for right in complement)
            outputs.append((-1) ** inversions * value[:, source.index(blade)])
        return jnp.stack(outputs, axis=-1).reshape((-1,))

    def harmonic_basis(self, degree: int, /) -> Array:
        """Parseval-orthonormal constant forms in flattened coordinates."""
        components = self.form_type(degree).component_count
        zero = jnp.all(self.wavenumbers == 0.0, axis=-1).astype(self.wavenumbers.dtype)
        return (
            zero[:, None, None]
            * jnp.eye(components, dtype=self._complex.space(degree).structure().dtype)[
                None, :, :
            ]
        ).reshape((-1, components))

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
            raise TypeError("Fourier forms require an array coordinate space.")
        value = space.validate(values)
        components = self.form_type(degree).component_count
        inverse = jnp.repeat(self.inverse_squared, components)
        harmonic_part = value * jnp.repeat(self.inverse_squared == 0.0, components)
        if harmonic is not None:
            value, harmonic_part = harmonic_projection(
                space,
                value,
                self.harmonic_basis(degree),
                harmonic,
                complex_.complex_id,
                degree,
            )
        potential = None
        exact = jnp.zeros_like(value)
        if degree > 0:
            potential = complex_.differential(degree - 1).adjoint_mv(value * inverse)
            if lower_harmonic is not None:
                lower_space = complex_.space(degree - 1)
                if not isinstance(lower_space, ArraySpace):
                    raise TypeError("Fourier forms require an array coordinate space.")
                potential, gauge = harmonic_projection(
                    lower_space,
                    potential,
                    self.harmonic_basis(degree - 1),
                    lower_harmonic,
                    complex_.complex_id,
                    degree - 1,
                )
                potential = potential - gauge
            exact = complex_.differential(degree - 1).mv(potential)
        if degree == 1 and self.leray is not None:
            transverse = self.from_modal(1, self.leray.project(self.to_modal(1, value)))
            exact = value - transverse
            coexact = transverse - harmonic_part
        else:
            coexact = value - exact - harmonic_part
        return decomposition_result(
            space, value, potential, exact, coexact, harmonic_part
        )


__all__ = ["FourierDeRhamComplex", "FourierNyquistPolicy"]
