#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg import ArraySpace, ComplexCartesianCoordinates, DenseLinearOperator
from ..nonlinear import NonlinearSystemProblem
from ._system import SparsePolynomialSystem


class _RealPolynomialResidual(StrictModule):
    system: SparsePolynomialSystem
    variable_coordinates: ComplexCartesianCoordinates
    equation_coordinates: ComplexCartesianCoordinates
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: SparsePolynomialSystem,
        variable_coordinates: ComplexCartesianCoordinates,
        equation_coordinates: ComplexCartesianCoordinates,
        /,
    ):
        self.system = system
        self.variable_coordinates = variable_coordinates
        self.equation_coordinates = equation_coordinates
        self.operator_id = canonical_fingerprint(
            {
                "kind": "real-polynomial-residual",
                "system": system.system_id,
                "variables": variable_coordinates.coordinate_id,
                "equations": equation_coordinates.coordinate_id,
            }
        )

    def __call__(self, coordinates: ArrayLike, args: Any = None, /) -> Array:
        del args
        root = self.variable_coordinates.from_real_coordinates(coordinates)
        residual = self.system.evaluate(root)
        return self.equation_coordinates.to_real_coordinates(residual)

    def linear_setup(
        self,
        coordinates: ArrayLike,
        args: Any = None,
        /,
    ) -> DenseLinearOperator:
        del args
        root = self.variable_coordinates.from_real_coordinates(coordinates)
        jacobian = self.system.jacobian(root)
        real = jnp.real(jacobian)
        imaginary = jnp.imag(jacobian)
        block = jnp.concatenate(
            (
                jnp.concatenate((real, -imaginary), axis=-1),
                jnp.concatenate((imaginary, real), axis=-1),
            ),
            axis=-2,
        )
        return DenseLinearOperator(
            block,
            source=self.variable_coordinates.coordinate_space,
            target=self.equation_coordinates.coordinate_space,
            operator_id=f"real-polynomial-jacobian:{self.operator_id}",
        )


class ComplexPolynomialRootLowering(StrictModule):
    """One unbatched complex polynomial root problem in real Cartesian coordinates."""

    problem: NonlinearSystemProblem
    variable_coordinates: ComplexCartesianCoordinates
    equation_coordinates: ComplexCartesianCoordinates
    lowering_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: SparsePolynomialSystem,
        /,
        *,
        complex_dtype: Any | None = None,
    ):
        if not isinstance(system, SparsePolynomialSystem):
            raise TypeError("system must be SparsePolynomialSystem.")
        if system.support.equation_count != system.support.variable_count:
            raise ValueError("Complex polynomial root lowering requires a square system.")
        dtype = (
            np.result_type(np.dtype(system.coefficients.dtype), np.complex64)
            if complex_dtype is None
            else np.dtype(complex_dtype)
        )
        if not np.issubdtype(dtype, np.complexfloating):
            raise TypeError("Complex polynomial root lowering requires a complex dtype.")
        root_space_id = (
            f"complex-polynomial-root-space:{system.support.support_id}:{dtype.str}"
        )
        variable_space = ArraySpace(
            (system.support.variable_count,),
            dtype=dtype,
            space_id=root_space_id,
        )
        equation_space = ArraySpace(
            (system.support.equation_count,),
            dtype=dtype,
            space_id=root_space_id,
        )
        variable_coordinates = ComplexCartesianCoordinates(variable_space, pair_axis=0)
        equation_coordinates = ComplexCartesianCoordinates(equation_space, pair_axis=0)
        residual = _RealPolynomialResidual(
            system,
            variable_coordinates,
            equation_coordinates,
        )
        identifier = canonical_fingerprint(
            {
                "kind": "complex-polynomial-root-lowering",
                "system": system.system_id,
                "variable_coordinates": variable_coordinates.coordinate_id,
                "equation_coordinates": equation_coordinates.coordinate_id,
            }
        )
        problem = NonlinearSystemProblem(
            residual,
            state_space=variable_coordinates.coordinate_space,
            residual_space=equation_coordinates.coordinate_space,
            linear_setup=residual.linear_setup,
            problem_id=f"complex-polynomial-root:{identifier}",
        )
        self.problem = problem
        self.variable_coordinates = variable_coordinates
        self.equation_coordinates = equation_coordinates
        self.lowering_id = identifier

    def to_real_coordinates(self, root: ArrayLike, /) -> Array:
        return self.variable_coordinates.to_real_coordinates(root)

    def from_real_coordinates(self, coordinates: ArrayLike, /) -> Array:
        return self.variable_coordinates.from_real_coordinates(coordinates)

    def real_block_jacobian(self, coordinates: ArrayLike, /) -> Array:
        operator = self.problem.linear_setup(coordinates)
        if not isinstance(operator, DenseLinearOperator):
            raise RuntimeError(
                "Polynomial root lowering did not produce a dense Jacobian."
            )
        return operator.matrix


def lower_complex_polynomial_root(
    system: SparsePolynomialSystem,
    /,
    *,
    complex_dtype: Any | None = None,
) -> ComplexPolynomialRootLowering:
    """Lower one selected complex root state to the native nonlinear contracts."""
    return ComplexPolynomialRootLowering(system, complex_dtype=complex_dtype)


__all__ = [
    "ComplexPolynomialRootLowering",
    "lower_complex_polynomial_root",
]
