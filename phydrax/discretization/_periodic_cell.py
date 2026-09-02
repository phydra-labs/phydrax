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

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


_TWO_PI = 2.0 * np.pi


class PeriodicCell(StrictModule, NonTrainableState):
    """Certified rank-r affine periodic lattice in ambient dimension d.

    Direct lattice vectors are rows.  Fractional row coordinates ``s`` map to
    the lattice span as ``origin + s @ vectors``.  ``inverse_vectors`` is the
    right coordinate inverse with shape ``(d, r)`` and ``reciprocal_vectors``
    contains row reciprocal vectors satisfying
    ``vectors @ reciprocal_vectors.T == 2 pi I``.

    For a lower-rank lattice, wrapping and minimum-image evaluation preserve
    the component orthogonal to the lattice span.  Image enumeration is fixed
    at preparation, bounded by ``maximum_image_count``, and certified against
    ``maximum_condition_number``.
    """

    origin: Array
    vectors: Array
    inverse_vectors: Array
    reciprocal_vectors: Array
    periodic_mask: Array
    image_shifts: Array
    periodic_axes: tuple[bool, ...] = eqx.field(static=True)
    cell_measure: float = eqx.field(static=True)
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
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("PeriodicCell vectors must have shape (rank > 0, d > 0).")
        rank, ambient_dimension = map(int, matrix.shape)
        if rank > ambient_dimension:
            raise ValueError("PeriodicCell rank cannot exceed its ambient dimension.")
        dtype = np.result_type(matrix.dtype, np.float32)
        matrix = matrix.astype(dtype, copy=False)
        if np.any(~np.isfinite(matrix)):
            raise ValueError("PeriodicCell vectors must be finite.")
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        threshold = np.finfo(dtype).eps * max(matrix.shape) * singular_values[0]
        if singular_values[-1] <= threshold:
            raise ValueError("PeriodicCell vectors must have full row rank.")

        origin_host = (
            np.zeros((ambient_dimension,), dtype=dtype)
            if origin is None
            else np.asarray(origin, dtype=dtype)
        )
        if origin_host.shape != (ambient_dimension,) or np.any(~np.isfinite(origin_host)):
            raise ValueError(
                "PeriodicCell origin must be a finite ambient-dimension vector."
            )
        axes = (
            (True,) * rank
            if periodic_axes is None
            else tuple(bool(value) for value in periodic_axes)
        )
        if len(axes) != rank:
            raise ValueError(
                "periodic_axes must align with PeriodicCell lattice vectors."
            )

        condition = float(singular_values[0] / singular_values[-1])
        certified_condition = (
            condition
            if maximum_condition_number is None
            else float(maximum_condition_number)
        )
        if not np.isfinite(certified_condition) or certified_condition < condition:
            raise ValueError(
                "maximum_condition_number must cover the initial lattice condition."
            )
        extent = max(1, ceil(0.5 + certified_condition * sqrt(rank)))
        choices = [range(-extent, extent + 1) if axis else (0,) for axis in axes]
        shifts = np.asarray(tuple(product(*choices)), dtype=np.int32)
        limit = int(maximum_image_count)
        if limit <= 0:
            raise ValueError("maximum_image_count must be positive.")
        if shifts.shape[0] > limit:
            raise ValueError(
                f"PeriodicCell requires {shifts.shape[0]} image candidates, "
                f"exceeding maximum_image_count={limit}."
            )

        gram = matrix @ matrix.T
        gram_inverse = np.linalg.inv(gram)
        inverse = matrix.T @ gram_inverse
        reciprocal = _TWO_PI * inverse.T
        measure = float(np.sqrt(np.linalg.det(gram)))
        if not np.isfinite(measure) or measure <= np.finfo(dtype).eps:
            raise ValueError("PeriodicCell lattice measure must be finite and positive.")
        nonzero = np.any(shifts != 0, axis=1)
        if np.any(nonzero):
            translations = shifts[nonzero] @ matrix
            shortest = float(np.min(np.linalg.norm(translations, axis=1)))
            unique_radius = 0.5 * shortest
        else:
            unique_radius = float("inf")

        self.origin = jnp.asarray(origin_host)
        self.vectors = jnp.asarray(matrix)
        self.inverse_vectors = jnp.asarray(inverse, dtype=dtype)
        self.reciprocal_vectors = jnp.asarray(reciprocal, dtype=dtype)
        self.periodic_mask = jnp.asarray(axes, dtype=bool)
        self.image_shifts = jnp.asarray(shifts, dtype=jnp.int32)
        self.periodic_axes = axes
        self.cell_measure = measure
        self.volume = measure
        self.unique_image_radius = unique_radius
        self.condition_number = condition
        self.certified_condition_number = certified_condition
        self.image_extent = extent
        self.cell_id = canonical_fingerprint(
            {
                "kind": "periodic-cell",
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
    def rank(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def ambient_dimension(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def fully_periodic(self) -> bool:
        return all(self.periodic_axes)

    @property
    def box_id(self) -> str:
        return self.cell_id

    def _require_ambient(self, value: Array, noun: str, /) -> None:
        if not value.shape or value.shape[-1] != self.ambient_dimension:
            raise ValueError(
                f"PeriodicCell {noun} must end in ambient dimension "
                f"{self.ambient_dimension}."
            )

    def _require_fractional(self, value: Array, /) -> None:
        if not value.shape or value.shape[-1] != self.rank:
            raise ValueError(
                f"Fractional coordinates must end in lattice rank {self.rank}."
            )

    def fractional(self, position: ArrayLike, /) -> Array:
        value = jnp.asarray(position)
        self._require_ambient(value, "positions")
        return contract(
            "...i,ij->...j",
            value - self.origin.astype(value.dtype),
            self.inverse_vectors.astype(value.dtype),
            backend="jax",
        )

    def cartesian(self, fractional: ArrayLike, /) -> Array:
        value = jnp.asarray(fractional)
        self._require_fractional(value)
        return self.origin.astype(value.dtype) + contract(
            "...i,ij->...j", value, self.vectors.astype(value.dtype), backend="jax"
        )

    def wrap(self, position: ArrayLike, /) -> tuple[Array, Array]:
        value = jnp.asarray(position)
        self._require_ambient(value, "positions")
        fractional = self.fractional(value)
        raw_images = jax.lax.stop_gradient(jnp.floor(fractional).astype(jnp.int32))
        images = jnp.where(self.periodic_mask, raw_images, 0)
        translation = contract(
            "...i,ij->...j",
            images.astype(value.dtype),
            self.vectors.astype(value.dtype),
            backend="jax",
        )
        return value - translation, images

    def minimum_image(self, displacement: ArrayLike, /) -> Array:
        value = jnp.asarray(displacement)
        self._require_ambient(value, "displacements")
        fractional = contract(
            "...i,ij->...j",
            value,
            self.inverse_vectors.astype(value.dtype),
            backend="jax",
        )
        central = jax.lax.stop_gradient(jnp.round(fractional).astype(jnp.int32))
        central = jnp.where(self.periodic_mask, central, 0)
        shifts = central[..., None, :] + self.image_shifts
        translations = contract(
            "...si,ij->...sj",
            shifts.astype(value.dtype),
            self.vectors.astype(value.dtype),
            backend="jax",
        )
        candidates = value[..., None, :] - translations
        selected = jax.lax.stop_gradient(
            jnp.argmin(jnp.sum(candidates * candidates, axis=-1), axis=-1)
        )
        return jnp.take_along_axis(candidates, selected[..., None, None], axis=-2)[
            ..., 0, :
        ]

    def inverse_for_vectors(self, vectors: ArrayLike, /) -> Array:
        matrix = jnp.asarray(vectors, dtype=self.vectors.dtype)
        if matrix.shape != self.vectors.shape:
            raise ValueError("Dynamic lattice vectors must match the prepared shape.")
        gram = contract("ik,jk->ij", matrix, matrix, backend="jax")
        return contract("ij,ik->jk", matrix, jnp.linalg.inv(gram), backend="jax")

    def fractional_with_vectors(
        self, position: ArrayLike, vectors: ArrayLike, /
    ) -> Array:
        value = jnp.asarray(position)
        self._require_ambient(value, "positions")
        inverse = self.inverse_for_vectors(vectors).astype(value.dtype)
        return contract(
            "...i,ij->...j",
            value - self.origin.astype(value.dtype),
            inverse,
            backend="jax",
        )

    def cartesian_with_vectors(
        self, fractional: ArrayLike, vectors: ArrayLike, /
    ) -> Array:
        value = jnp.asarray(fractional)
        self._require_fractional(value)
        matrix = jnp.asarray(vectors, dtype=value.dtype)
        if matrix.shape != self.vectors.shape:
            raise ValueError("Dynamic lattice vectors must match the prepared shape.")
        return self.origin.astype(value.dtype) + contract(
            "...i,ij->...j", value, matrix, backend="jax"
        )

    def wrap_with_vectors(
        self, position: ArrayLike, vectors: ArrayLike, /
    ) -> tuple[Array, Array]:
        value = jnp.asarray(position)
        self._require_ambient(value, "positions")
        matrix = jnp.asarray(vectors, dtype=value.dtype)
        fractional = self.fractional_with_vectors(value, matrix)
        raw_images = jax.lax.stop_gradient(jnp.floor(fractional).astype(jnp.int32))
        images = jnp.where(self.periodic_mask, raw_images, 0)
        translation = contract(
            "...i,ij->...j", images.astype(value.dtype), matrix, backend="jax"
        )
        return value - translation, images

    def minimum_image_with_vectors(
        self, displacement: ArrayLike, vectors: ArrayLike, /
    ) -> Array:
        value = jnp.asarray(displacement)
        self._require_ambient(value, "displacements")
        matrix = jnp.asarray(vectors, dtype=value.dtype)
        inverse = self.inverse_for_vectors(matrix).astype(value.dtype)
        fractional = contract("...i,ij->...j", value, inverse, backend="jax")
        central = jax.lax.stop_gradient(jnp.round(fractional).astype(jnp.int32))
        central = jnp.where(self.periodic_mask, central, 0)
        shifts = central[..., None, :] + self.image_shifts
        translations = contract(
            "...si,ij->...sj", shifts.astype(value.dtype), matrix, backend="jax"
        )
        candidates = value[..., None, :] - translations
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
                "Interaction radius violates the PeriodicCell unique-image certificate."
            )


__all__ = ["PeriodicCell"]
