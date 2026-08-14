#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ._operators import AbstractLinearOperator
from ._spaces import AbstractVectorSpace
from ._subspaces import LinearSubspace


class RecyclingSubspace(StrictModule):
    """Image-orthonormal coarse space reusable across related linear solves."""

    source: AbstractVectorSpace
    target: AbstractVectorSpace
    source_basis: Array
    image_basis: Array
    operator_id: str = eqx.field(static=True)
    recycling_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        source: AbstractVectorSpace,
        target: AbstractVectorSpace,
        source_basis: ArrayLike,
        image_basis: ArrayLike,
        operator_id: str,
        recycling_id: str,
    ):
        if not isinstance(source, AbstractVectorSpace) or not isinstance(
            target, AbstractVectorSpace
        ):
            raise TypeError("source and target must be AbstractVectorSpace values.")
        source_basis_ = jnp.asarray(source_basis)
        image_basis_ = jnp.asarray(image_basis)
        if (
            source_basis_.ndim != 2
            or image_basis_.ndim != 2
            or source_basis_.shape[0] != source.size
            or image_basis_.shape[0] != target.size
            or source_basis_.shape[1] != image_basis_.shape[1]
            or source_basis_.shape[1] < 1
        ):
            raise ValueError(
                "Recycling bases must have shapes (source.size, k) and "
                "(target.size, k) for the same positive k."
            )
        if not jnp.issubdtype(source_basis_.dtype, jnp.inexact) or not jnp.issubdtype(
            image_basis_.dtype, jnp.inexact
        ):
            raise TypeError("Recycling bases must use inexact dtypes.")
        identifiers = (str(operator_id), str(recycling_id))
        if any(not value for value in identifiers):
            raise ValueError("operator_id and recycling_id must be non-empty.")
        self.source = source
        self.target = target
        self.source_basis = source_basis_
        self.image_basis = image_basis_
        self.operator_id, self.recycling_id = identifiers

    @property
    def dimension(self) -> int:
        return int(self.source_basis.shape[1])

    def coefficients(self, residual: PyTree[Any], /) -> Array:
        """Return coarse coefficients under the target-space pairing."""
        value = self.target.validate(residual)
        return jax.vmap(
            lambda column: self.target.inner(self.target.unflatten(column), value),
            in_axes=1,
        )(self.image_basis)

    def correction(self, residual: PyTree[Any], /) -> PyTree[Array]:
        """Map a residual to its coarse source-space correction."""
        return self.source.unflatten(self.source_basis @ self.coefficients(residual))

    def project_residual(self, residual: PyTree[Any], /) -> PyTree[Array]:
        """Remove the represented image component from a residual."""
        value = self.target.validate(residual)
        coordinates = self.target.flatten(value)
        projected = coordinates - self.image_basis @ self.coefficients(value)
        return self.target.unflatten(projected)

    def augment(
        self,
        initial: PyTree[Any],
        residual: PyTree[Any],
        /,
    ) -> PyTree[Array]:
        """Add the coarse correction for ``residual`` to an initial solution."""
        value = self.source.validate(initial)
        correction = self.correction(residual)
        return jax.tree.map(lambda left, right: left + right, value, correction)


def prepare_recycling_subspace(
    operator: AbstractLinearOperator,
    subspace: LinearSubspace | ArrayLike,
    /,
    *,
    recycling_id: str | None = None,
) -> RecyclingSubspace:
    """Prepare an image-orthonormal coarse space for repeated solves with ``operator``."""
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("operator must be an AbstractLinearOperator.")
    if operator.batch_shape:
        raise ValueError("Recycling preparation requires an unbatched operator.")
    if isinstance(subspace, LinearSubspace):
        if not subspace.space.compatible(operator.source):
            raise ValueError("The recycling subspace must use the operator source space.")
        dimension = int(subspace.dimension)
        if dimension < 1:
            raise ValueError("The recycling subspace must have positive dimension.")
        source_basis = subspace.basis[:, :dimension]
        source_id = subspace.subspace_id
    else:
        source_basis = jnp.asarray(subspace)
        if source_basis.ndim != 2 or source_basis.shape[0] != operator.source.size:
            raise ValueError("basis must have shape (operator.source.size, k).")
        if source_basis.shape[1] < 1:
            raise ValueError("basis must contain at least one vector.")
        source_id = canonical_fingerprint(
            {
                "kind": "recycling-source",
                "source": operator.source.space_id,
                "basis": array_tree_fingerprint(source_basis),
            }
        )
    if not jnp.issubdtype(source_basis.dtype, jnp.inexact):
        raise TypeError("The recycling basis must use an inexact dtype.")

    image_basis = jax.vmap(
        lambda coordinates: operator.target.flatten(
            operator.mv(operator.source.unflatten(coordinates))
        ),
        in_axes=1,
        out_axes=1,
    )(source_basis)
    gram = _coordinate_gram(operator.target, image_basis)
    gram = 0.5 * (gram + jnp.conj(gram.T))
    eigenvalues, eigenvectors = jnp.linalg.eigh(gram)
    eigenvalues_host = np.asarray(eigenvalues)
    scale = max(float(np.max(np.abs(eigenvalues_host))), 1.0)
    tolerance = np.finfo(eigenvalues_host.dtype).eps * source_basis.shape[1] * scale
    if np.any(eigenvalues_host <= tolerance):
        raise ValueError(
            "The operator images of the recycling basis must be linearly independent."
        )
    inverse_square_root = (
        eigenvectors
        * jax.lax.rsqrt(jnp.asarray(eigenvalues, dtype=eigenvalues.dtype))[None, :]
    ) @ jnp.conj(eigenvectors.T)
    normalized_source = source_basis @ inverse_square_root
    normalized_image = image_basis @ inverse_square_root
    identifier = (
        canonical_fingerprint(
            {
                "kind": "recycling-subspace",
                "operator": operator.operator_id,
                "source_subspace": source_id,
                "dimension": int(source_basis.shape[1]),
            }
        )
        if recycling_id is None
        else str(recycling_id)
    )
    if not identifier:
        raise ValueError("recycling_id must be non-empty.")
    return RecyclingSubspace(
        source=operator.source,
        target=operator.target,
        source_basis=normalized_source,
        image_basis=normalized_image,
        operator_id=operator.operator_id,
        recycling_id=identifier,
    )


def _coordinate_gram(space: AbstractVectorSpace, basis: Array, /) -> Array:
    return jax.vmap(
        lambda left: jax.vmap(
            lambda right: space.inner(space.unflatten(left), space.unflatten(right)),
            in_axes=1,
        )(basis),
        in_axes=1,
    )(basis)


__all__ = ["RecyclingSubspace", "prepare_recycling_subspace"]
