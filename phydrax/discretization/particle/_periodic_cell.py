#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import product
from math import ceil, sqrt

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class ParticleCell(StrictModule, NonTrainableState):
    """Certified fixed parallelepiped cell with explicit periodic axes.

    Lattice vectors are rows, so a fractional row vector ``s`` maps to physical
    coordinates as ``origin + s @ vectors``.  Minimum-image evaluation first
    centers fractional displacements and then searches a finite stencil whose
    extent is bounded from the cell condition number.
    """

    origin: Array
    vectors: Array
    inverse_vectors: Array
    periodic_mask: Array
    image_shifts: Array
    periodic_axes: tuple[bool, ...] = eqx.field(static=True)
    volume: float = eqx.field(static=True)
    unique_image_radius: float = eqx.field(static=True)
    condition_number: float = eqx.field(static=True)
    certified_condition_number: float = eqx.field(static=True)
    image_extent: int = eqx.field(static=True)
    cell_id: str = eqx.field(static=True)

    def __init__(
        self,
        vectors: ArrayLike,
        /,
        *,
        origin: ArrayLike | None = None,
        periodic_axes: tuple[bool, ...] | None = None,
        maximum_condition_number: float | None = None,
        maximum_image_count: int = 4096,
    ):
        matrix = np.asarray(vectors)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
            raise ValueError("ParticleCell vectors must form a non-empty square matrix.")
        if matrix.shape[0] > 3:
            raise ValueError("ParticleCell supports dimensions one through three.")
        dtype = np.result_type(matrix.dtype, np.float32)
        matrix = matrix.astype(dtype, copy=False)
        if np.any(~np.isfinite(matrix)):
            raise ValueError("ParticleCell vectors must be finite.")
        determinant = float(np.linalg.det(matrix))
        if not np.isfinite(determinant) or abs(determinant) <= np.finfo(dtype).eps:
            raise ValueError("ParticleCell vectors must be nonsingular.")
        origin_host = (
            np.zeros((matrix.shape[0],), dtype=dtype)
            if origin is None
            else np.asarray(origin, dtype=dtype)
        )
        if origin_host.shape != (matrix.shape[0],) or np.any(~np.isfinite(origin_host)):
            raise ValueError("ParticleCell origin must be a finite dimension vector.")
        axes = (
            (True,) * matrix.shape[0]
            if periodic_axes is None
            else tuple(bool(value) for value in periodic_axes)
        )
        if len(axes) != matrix.shape[0]:
            raise ValueError("periodic_axes must align with ParticleCell vectors.")
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        condition = float(singular_values[0] / singular_values[-1])
        certified_condition = (
            condition
            if maximum_condition_number is None
            else float(maximum_condition_number)
        )
        if not np.isfinite(certified_condition) or certified_condition < condition:
            raise ValueError(
                "maximum_condition_number must cover the initial cell condition."
            )
        extent = max(1, ceil(0.5 + certified_condition * sqrt(matrix.shape[0])))
        choices = [range(-extent, extent + 1) if axis else (0,) for axis in axes]
        shifts = np.asarray(tuple(product(*choices)), dtype=np.int32)
        limit = int(maximum_image_count)
        if limit <= 0:
            raise ValueError("maximum_image_count must be positive.")
        if shifts.shape[0] > limit:
            raise ValueError(
                f"ParticleCell requires {shifts.shape[0]} image candidates, exceeding "
                f"maximum_image_count={limit}."
            )
        nonzero = np.any(shifts != 0, axis=1)
        if np.any(nonzero):
            lattice_vectors = shifts[nonzero] @ matrix
            shortest = float(
                np.min(np.sqrt(np.sum(lattice_vectors * lattice_vectors, axis=1)))
            )
            unique_radius = 0.5 * shortest
        else:
            unique_radius = float("inf")
        inverse = np.linalg.solve(
            matrix,
            np.eye(matrix.shape[0], dtype=matrix.dtype),
        )
        self.origin = jnp.asarray(origin_host)
        self.vectors = jnp.asarray(matrix)
        self.inverse_vectors = jnp.asarray(inverse, dtype=dtype)
        self.periodic_mask = jnp.asarray(axes, dtype=bool)
        self.image_shifts = jnp.asarray(shifts, dtype=jnp.int32)
        self.periodic_axes = axes
        self.volume = abs(determinant)
        self.unique_image_radius = unique_radius
        self.condition_number = condition
        self.image_extent = extent
        self.certified_condition_number = certified_condition
        self.cell_id = canonical_fingerprint(
            {
                "kind": "particle-cell",
                "arrays": array_tree_fingerprint(
                    {
                        "origin": origin_host,
                        "vectors": matrix,
                        "periodic_axes": np.asarray(axes),
                    }
                ),
                "image_extent": extent,
                "maximum_image_count": limit,
                "certified_condition_number": certified_condition,
            }
        )

    @property
    def ambient_dimension(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def fully_periodic(self) -> bool:
        return all(self.periodic_axes)

    @property
    def box_id(self) -> str:
        return self.cell_id

    def fractional(self, position: ArrayLike, /) -> Array:
        value = jnp.asarray(position)
        if not value.shape or value.shape[-1] != self.ambient_dimension:
            raise ValueError("ParticleCell positions must end in the cell dimension.")
        return contract(
            "...i,ij->...j",
            value - self.origin.astype(value.dtype),
            self.inverse_vectors.astype(value.dtype),
        )

    def cartesian(self, fractional: ArrayLike, /) -> Array:
        value = jnp.asarray(fractional)
        if not value.shape or value.shape[-1] != self.ambient_dimension:
            raise ValueError("Fractional positions must end in the cell dimension.")
        return self.origin.astype(value.dtype) + contract(
            "...i,ij->...j", value, self.vectors.astype(value.dtype)
        )

    def wrap(self, position: ArrayLike, /) -> tuple[Array, Array]:
        value = jnp.asarray(position)
        fractional = self.fractional(value)
        raw_images = jax.lax.stop_gradient(jnp.floor(fractional).astype(jnp.int32))
        images = jnp.where(self.periodic_mask, raw_images, 0)
        wrapped_fractional = fractional - images.astype(fractional.dtype)
        return self.cartesian(wrapped_fractional), images

    def minimum_image(self, displacement: ArrayLike, /) -> Array:
        value = jnp.asarray(displacement)
        if not value.shape or value.shape[-1] != self.ambient_dimension:
            raise ValueError("ParticleCell displacements must end in the cell dimension.")
        fractional = contract(
            "...i,ij->...j", value, self.inverse_vectors.astype(value.dtype)
        )
        central_shift = jax.lax.stop_gradient(jnp.round(fractional).astype(jnp.int32))
        central_shift = jnp.where(self.periodic_mask, central_shift, 0)
        centered = fractional - central_shift.astype(fractional.dtype)
        candidate_fractional = centered[..., None, :] - self.image_shifts.astype(
            value.dtype
        )
        candidate = contract(
            "...si,ij->...sj", candidate_fractional, self.vectors.astype(value.dtype)
        )
        distance_squared = jnp.sum(candidate * candidate, axis=-1)
        selected = jax.lax.stop_gradient(jnp.argmin(distance_squared, axis=-1))
        return jnp.take_along_axis(
            candidate,
            selected[..., None, None],
            axis=-2,
        )[..., 0, :]

    def inverse_for_vectors(self, vectors: ArrayLike, /) -> Array:
        matrix = jnp.asarray(vectors, dtype=self.vectors.dtype)
        if matrix.shape != self.vectors.shape:
            raise ValueError("Dynamic cell vectors must match the prepared cell shape.")
        if self.ambient_dimension == 1:
            return 1.0 / matrix
        if self.ambient_dimension == 2:
            determinant = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
            return (
                jnp.asarray(
                    [[matrix[1, 1], -matrix[0, 1]], [-matrix[1, 0], matrix[0, 0]]]
                )
                / determinant
            )
        first, second, third = matrix
        determinant = jnp.sum(first * jnp.cross(second, third))
        columns = jnp.stack(
            (
                jnp.cross(second, third),
                jnp.cross(third, first),
                jnp.cross(first, second),
            ),
            axis=1,
        )
        return columns / determinant

    def fractional_with_vectors(
        self, position: ArrayLike, vectors: ArrayLike, /
    ) -> Array:
        value = jnp.asarray(position)
        inverse = self.inverse_for_vectors(vectors).astype(value.dtype)
        return contract("...i,ij->...j", value - self.origin.astype(value.dtype), inverse)

    def cartesian_with_vectors(
        self, fractional: ArrayLike, vectors: ArrayLike, /
    ) -> Array:
        value = jnp.asarray(fractional)
        matrix = jnp.asarray(vectors, dtype=value.dtype)
        return self.origin.astype(value.dtype) + contract("...i,ij->...j", value, matrix)

    def wrap_with_vectors(
        self, position: ArrayLike, vectors: ArrayLike, /
    ) -> tuple[Array, Array]:
        fractional = self.fractional_with_vectors(position, vectors)
        raw_images = jax.lax.stop_gradient(jnp.floor(fractional).astype(jnp.int32))
        images = jnp.where(self.periodic_mask, raw_images, 0)
        return (
            self.cartesian_with_vectors(
                fractional - images.astype(fractional.dtype), vectors
            ),
            images,
        )

    def minimum_image_with_vectors(
        self, displacement: ArrayLike, vectors: ArrayLike, /
    ) -> Array:
        value = jnp.asarray(displacement)
        matrix = jnp.asarray(vectors, dtype=value.dtype)
        inverse = self.inverse_for_vectors(matrix).astype(value.dtype)
        fractional = contract("...i,ij->...j", value, inverse)
        central_shift = jax.lax.stop_gradient(jnp.round(fractional).astype(jnp.int32))
        central_shift = jnp.where(self.periodic_mask, central_shift, 0)
        centered = fractional - central_shift.astype(value.dtype)
        candidates_fractional = centered[..., None, :] - self.image_shifts.astype(
            value.dtype
        )
        candidates = contract("...si,ij->...sj", candidates_fractional, matrix)
        selected = jax.lax.stop_gradient(
            jnp.argmin(jnp.sum(candidates * candidates, axis=-1), axis=-1)
        )
        return jnp.take_along_axis(candidates, selected[..., None, None], axis=-2)[
            ..., 0, :
        ]

    def require_unique_image(self, radius: float, /) -> None:
        value = float(radius)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("Interaction radius must be finite and positive.")
        if value >= self.unique_image_radius:
            raise ValueError(
                "Interaction radius violates the ParticleCell unique-image certificate."
            )


__all__ = ["ParticleCell"]
