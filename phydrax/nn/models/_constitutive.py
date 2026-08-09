#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from ..._doc import DOC_KEY0
from ..._strict import StrictModule
from .._keys import EvalKey
from ._input_convex import ConvexActivation, InputConvexNetwork


class DeformationGradientMinors(StrictModule):
    """Polynomial first-order minors, cofactors, and determinant for 2D or 3D."""

    dimension: int = eqx.field(static=True)
    lifted_size: int = eqx.field(static=True)

    def __init__(self, dimension: int, /):
        self.dimension = int(dimension)
        if self.dimension not in (2, 3):
            raise ValueError("Deformation-gradient minors support dimension 2 or 3.")
        self.lifted_size = 2 * self.dimension**2 + 1

    def _validate(self, deformation_gradient: Array, /) -> Array:
        gradient = jnp.asarray(deformation_gradient)
        expected = (self.dimension, self.dimension)
        if gradient.ndim < 2 or gradient.shape[-2:] != expected:
            raise ValueError(f"deformation_gradient must end in shape {expected}.")
        if not (
            jnp.issubdtype(gradient.dtype, jnp.floating)
            or jnp.issubdtype(gradient.dtype, jnp.complexfloating)
        ):
            raise TypeError("deformation_gradient must have an inexact dtype.")
        return gradient

    def cofactor(self, deformation_gradient: Array, /) -> Array:
        """Return the cofactor matrix without inversion or determinant division."""
        gradient = self._validate(deformation_gradient)
        if self.dimension == 2:
            first = jnp.stack((gradient[..., 1, 1], -gradient[..., 1, 0]), axis=-1)
            second = jnp.stack((-gradient[..., 0, 1], gradient[..., 0, 0]), axis=-1)
            return jnp.stack((first, second), axis=-2)
        first_column = jnp.cross(gradient[..., :, 1], gradient[..., :, 2])
        second_column = jnp.cross(gradient[..., :, 2], gradient[..., :, 0])
        third_column = jnp.cross(gradient[..., :, 0], gradient[..., :, 1])
        return jnp.stack((first_column, second_column, third_column), axis=-1)

    def determinant(self, deformation_gradient: Array, /) -> Array:
        """Return the determinant from polynomial minors."""
        gradient = self._validate(deformation_gradient)
        cofactor = self.cofactor(gradient)
        if self.dimension == 2:
            return (
                gradient[..., 0, 0] * gradient[..., 1, 1]
                - gradient[..., 0, 1] * gradient[..., 1, 0]
            )
        return jnp.sum(gradient[..., :, 0] * cofactor[..., :, 0], axis=-1)

    def __call__(self, deformation_gradient: Array, /) -> Array:
        """Pack ``(F, cof(F), det(F))`` into the lifted polyconvex coordinates."""
        gradient = self._validate(deformation_gradient)
        batch_shape = gradient.shape[:-2]
        cofactor = self.cofactor(gradient)
        determinant = self.determinant(gradient)
        return jnp.concatenate(
            (
                gradient.reshape(batch_shape + (self.dimension**2,)),
                cofactor.reshape(batch_shape + (self.dimension**2,)),
                determinant[..., None],
            ),
            axis=-1,
        )


class PolyconvexPotential(StrictModule):
    """Hyperelastic energy convex in ``(F, cof(F), det(F))`` by construction.

    Polyconvexity follows from the owned :class:`InputConvexNetwork`. Objectivity,
    isotropy, coercivity, orientation preservation, and a stress-free reference are
    separate constitutive constraints and are deliberately not implied here.
    """

    potential: InputConvexNetwork
    minors: DeformationGradientMinors

    def __init__(
        self,
        dimension: int,
        /,
        *,
        potential: InputConvexNetwork | None = None,
        width_size: int = 64,
        depth: int = 3,
        activation: ConvexActivation = "softplus",
        use_bias: bool = True,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.minors = DeformationGradientMinors(dimension)
        convex_potential = (
            InputConvexNetwork(
                in_size=self.minors.lifted_size,
                width_size=width_size,
                depth=depth,
                activation=activation,
                use_bias=use_bias,
                key=key,
            )
            if potential is None
            else potential
        )
        if not isinstance(convex_potential, InputConvexNetwork):
            raise TypeError("potential must be an InputConvexNetwork.")
        if convex_potential.in_size != self.minors.lifted_size:
            raise ValueError(
                f"potential input size must be {self.minors.lifted_size} in "
                f"dimension {self.minors.dimension}."
            )
        self.potential = convex_potential

    @property
    def dimension(self) -> int:
        return self.minors.dimension

    def lifted_energy(
        self,
        lifted_minors: Array,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        """Evaluate the convex outer potential directly in lifted minor coordinates."""
        coordinates = jnp.asarray(lifted_minors)
        if coordinates.ndim < 1 or int(coordinates.shape[-1]) != self.minors.lifted_size:
            raise ValueError(
                f"lifted_minors must end in width {self.minors.lifted_size}."
            )
        return self.potential(coordinates, key=key)

    def __call__(
        self,
        deformation_gradient: Array,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        """Return the stored energy density at one or more deformation gradients."""
        return self.lifted_energy(self.minors(deformation_gradient), key=key)

    def first_piola_stress(
        self,
        deformation_gradient: Array,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        """Return the first Piola stress ``dW/dF`` with the same leading batch shape."""
        gradient = self.minors._validate(deformation_gradient)
        batch_shape = gradient.shape[:-2]
        flattened = gradient.reshape((-1, self.dimension, self.dimension))
        derivative = jax.grad(lambda value: self(value, key=key))
        stresses = jax.vmap(derivative)(flattened)
        return stresses.reshape(batch_shape + (self.dimension, self.dimension))

    def material_tangent(
        self,
        deformation_gradient: Array,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        """Return the consistent material tangent ``d²W/dF²`` pointwise."""
        gradient = self.minors._validate(deformation_gradient)
        batch_shape = gradient.shape[:-2]
        flattened = gradient.reshape((-1, self.dimension, self.dimension))
        derivative = jax.hessian(lambda value: self(value, key=key))
        tangents = jax.vmap(derivative)(flattened)
        return tangents.reshape(
            batch_shape
            + (
                self.dimension,
                self.dimension,
                self.dimension,
                self.dimension,
            )
        )


__all__ = ["DeformationGradientMinors", "PolyconvexPotential"]
