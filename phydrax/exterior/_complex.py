#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Metric de Rham realizations without a dependency on discretization owners."""

from __future__ import annotations

import abc
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier
from ..linalg import AbstractLinearOperator, apply_real_map_componentwise, ArraySpace
from ..linalg._complexes import (
    codifferential as complex_codifferential,
    HarmonicSubspace,
    HilbertComplex,
    hodge_decomposition as complex_hodge_decomposition,
    hodge_laplacian as complex_hodge_laplacian,
    HodgeDecomposition,
    HodgeDecompositionPolicy,
    HodgeLaplacianPart,
)
from ..typing import parse
from ._form_type import FormTwist, FormType


ComplexBoundary: TypeAlias = Literal["absolute", "relative"]


def _apply_map(
    operator: AbstractLinearOperator,
    values: ArrayLike,
    /,
    *,
    transpose: bool = False,
) -> Array:
    input_space = operator.target if transpose else operator.source
    action = operator.transpose_mv if transpose else operator.mv
    vector = jnp.asarray(values)
    if isinstance(input_space, ArraySpace) and not jnp.issubdtype(
        input_space.dtype, jnp.complexfloating
    ):
        return jnp.asarray(apply_real_map_componentwise(action, vector))
    return jnp.asarray(action(vector))


class AbstractDeRhamComplex(StrictModule, NonTrainableState):
    """A typed de Rham realization and its boundary-conditioned Hilbert complex.

    The protocol uses compact Hilbert coordinates. Cell realizations may expose
    full coordinate methods which restrict and zero extend at their boundary.
    It deliberately does not define ``space`` or ``capabilities``: prepared
    discretizations own those names and their existing contracts.
    """

    dimension: eqx.AbstractVar[int]
    primal_twist: eqx.AbstractVar[FormTwist]
    realization_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def hilbert_complex(
        self, /, *, boundary: ComplexBoundary = "absolute"
    ) -> HilbertComplex:
        """Return the prepared metric complex in the requested coordinates."""
        raise NotImplementedError

    @abc.abstractmethod
    def hodge_star(self, degree: int, values: ArrayLike, /) -> Array:
        """Apply the metric Riesz map into dual-complex coordinates."""
        raise NotImplementedError

    @abc.abstractmethod
    def inverse_hodge_star(self, degree: int, values: ArrayLike, /) -> Array:
        """Apply the metric inverse Riesz map from dual-complex coordinates."""
        raise NotImplementedError

    def form_type(self, degree: int, /, *, dual: bool = False) -> FormType:
        primal = FormType(self.dimension, degree, twist=self.primal_twist)
        return primal.hodge_dual() if dual else primal

    @property
    def cell_counts(self) -> tuple[int, ...]:
        return tuple(space.size for space in self.hilbert_complex().spaces)

    def exterior_derivative(
        self,
        degree: int,
        values: ArrayLike,
        /,
        *,
        boundary: ComplexBoundary = "absolute",
    ) -> Array:
        complex_ = self.hilbert_complex(
            boundary=parse(boundary, ComplexBoundary, "boundary")
        )
        return _apply_map(complex_.differential(degree), values)

    def codifferential(
        self,
        degree: int,
        values: ArrayLike,
        /,
        *,
        boundary: ComplexBoundary = "absolute",
    ) -> Array:
        complex_ = self.hilbert_complex(
            boundary=parse(boundary, ComplexBoundary, "boundary")
        )
        return _apply_map(complex_codifferential(complex_, degree), values)

    def hodge_laplacian(
        self,
        degree: int,
        values: ArrayLike,
        /,
        *,
        boundary: ComplexBoundary = "absolute",
        part: HodgeLaplacianPart = "complete",
    ) -> Array:
        complex_ = self.hilbert_complex(
            boundary=parse(boundary, ComplexBoundary, "boundary")
        )
        return _apply_map(complex_hodge_laplacian(complex_, degree, part=part), values)

    def dual_exterior_derivative(self, degree: int, values: ArrayLike, /) -> Array:
        """Apply the Hirani dual derivative to the dual of primal degree k."""
        if degree <= 0 or degree > self.dimension:
            raise ValueError("Dual exterior derivative requires 0 < degree <= dimension.")
        differential = self.hilbert_complex().differential(degree - 1)
        return (-1) ** degree * jnp.conj(
            _apply_map(differential, jnp.conj(jnp.asarray(values)), transpose=True)
        )

    def hodge_decomposition(
        self,
        degree: int,
        values: ArrayLike,
        /,
        *,
        boundary: ComplexBoundary = "absolute",
        harmonic: HarmonicSubspace,
        lower_harmonic: HarmonicSubspace | None = None,
        policy: HodgeDecompositionPolicy | None = None,
    ) -> HodgeDecomposition:
        complex_ = self.hilbert_complex(
            boundary=parse(boundary, ComplexBoundary, "boundary")
        )
        return complex_hodge_decomposition(
            complex_,
            degree,
            values,
            harmonic=harmonic,
            lower_harmonic=lower_harmonic,
            policy=policy,
        )


@final
class DiscreteForm(StrictModule, NonTrainableState):
    """One form whose values belong to an explicitly identified realization.

    Dual placement is determined relative to the realization's primal twist;
    dual degree n-k has the same coordinates as its primal Riesz source k.
    """

    realization_id: str = eqx.field(static=True)
    form_type: FormType = eqx.field(static=True)
    values: Array

    def __init__(
        self, realization_id: str, form_type: FormType, values: ArrayLike, /
    ) -> None:
        identifier = canonical_identifier(realization_id, "realization_id")
        if not isinstance(form_type, FormType):
            raise TypeError("form_type must be a FormType.")
        coefficients = jnp.asarray(values)
        if coefficients.ndim != 1:
            raise ValueError(
                "Discrete form values must be one coordinate vector; batch with vmap."
            )
        if not jnp.issubdtype(coefficients.dtype, jnp.inexact):
            coefficients = coefficients.astype(jnp.float64)
        self.realization_id = identifier
        self.form_type = form_type
        self.values = coefficients


__all__ = ["AbstractDeRhamComplex", "ComplexBoundary", "DiscreteForm"]
