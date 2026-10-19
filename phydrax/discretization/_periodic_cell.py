#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import product
from math import ceil, prod, sqrt

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import positive_integer
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    determinant_small_linear,
    DifferentiationPolicy,
    LinearSolvePolicy,
    LinearSystem,
    SmallLinearSolvePlan,
    solve,
    solve_small_linear,
)


_TWO_PI = 2.0 * np.pi


def lattice_right_inverse_with_status(vectors: ArrayLike, /) -> tuple[Array, Array]:
    """Return the right coordinate inverse and the native solve status.

    ``inverse = vectors.T @ inv(vectors @ vectors.T)``.  Lattices of rank at
    most four solve their Gram system through the batched closed-form/pivoted
    small-matrix owner (no LAPACK custom calls, so frozen exports compile);
    higher ranks use the dense solve owner.  Leading case axes are batched and
    ``successful`` has the leading shape.  Failed solves never raise here:
    numerical evaluation routes fold the status into their own evidence.
    """
    matrix = jnp.asarray(vectors)
    if matrix.ndim < 2:
        raise ValueError(
            "lattice_right_inverse_with_status.vectors must have shape (..., rank, d); got fewer than two axes"
        )
    rank = matrix.shape[-2]
    gram = contract("...ik,...jk->...ij", matrix, matrix, backend="jax")
    if rank <= 4:
        small = solve_small_linear(SmallLinearSolvePlan(rank), gram, matrix)
        inverse = jnp.swapaxes(small.value, -1, -2)
        successful = small.successful & jnp.all(jnp.isfinite(inverse), axis=(-2, -1))
        return inverse, successful
    if matrix.ndim > 2:
        flat = matrix.reshape((-1,) + matrix.shape[-2:])
        inverse, successful = jax.vmap(lattice_right_inverse_with_status)(flat)
        return (
            inverse.reshape(matrix.shape[:-2] + inverse.shape[-2:]),
            successful.reshape(matrix.shape[:-2]),
        )
    result = solve(LinearSystem(DenseLinearOperator(gram)), matrix)
    inverse = jnp.swapaxes(result.value, -1, -2)
    successful = jnp.all(result.successful) & jnp.all(jnp.isfinite(inverse))
    return inverse, successful


def lattice_right_inverse(vectors: ArrayLike, /) -> Array:
    """Return the right coordinate inverse, refusing a failed solve.

    Raise-oriented host/API form of :func:`lattice_right_inverse_with_status`.
    """
    inverse, successful = lattice_right_inverse_with_status(vectors)
    return eqx.error_if(
        inverse,
        ~jnp.all(successful),
        "lattice_right_inverse.vectors must admit a successful coordinate solve; got a failed native solve",
    )


def lattice_measure(vectors: ArrayLike, /) -> tuple[Array, Array]:
    """Return the dynamic lattice measure ``sqrt(det(H @ H.T))`` and its status.

    For a full-rank ``(3, 3)`` row cell this is the cell volume ``|det H|``; for
    lower rank it is the r-dimensional lattice measure.  The Gram determinant
    uses the native scaled small-matrix owner, leading case axes are batched,
    and ``successful`` is false for nonfinite or numerically singular lattices
    (their measure is reported as ``nan``, never a finite substitute).
    """
    matrix = jnp.asarray(vectors)
    if matrix.ndim < 2 or matrix.shape[-2] > 4:
        raise ValueError(
            "lattice_measure.vectors must have shape (..., rank <= 4, d); got an unsupported rank or shape"
        )
    if not jnp.issubdtype(matrix.dtype, jnp.inexact):
        matrix = matrix.astype(jnp.float64)
    gram = contract("...ik,...jk->...ij", matrix, matrix, backend="jax")
    determinant = determinant_small_linear(SmallLinearSolvePlan(matrix.shape[-2]), gram)
    scale = jnp.max(jnp.abs(gram), axis=(-2, -1)) ** matrix.shape[-2]
    tolerance = jnp.finfo(matrix.dtype).eps * matrix.shape[-2] * scale
    successful = (
        jnp.all(jnp.isfinite(matrix), axis=(-2, -1))
        & jnp.isfinite(determinant)
        & (determinant > tolerance)
    )
    safe = jnp.where(successful, determinant, 1.0)
    return jnp.where(successful, jnp.sqrt(safe), jnp.nan), successful


def complete_image_shifts(extents: tuple[int, ...], /) -> np.ndarray:
    """Every integer translation ``|n_i| <= extents[i]`` in lexicographic order."""
    grids = np.meshgrid(
        *(np.arange(-extent, extent + 1, dtype=np.int32) for extent in extents),
        indexing="ij",
    )
    return np.stack([grid.reshape((-1,)) for grid in grids], axis=-1).reshape(
        (-1, len(extents))
    )


class PeriodicImageStencil(StrictModule, NonTrainableState):
    """Enumerate bounded integer translations for one radius.

    ``shifts`` are every row ``n`` with ``|n_i| <= extents[i]`` in
    lexicographic order; they are derived from ``extents`` and never supplied,
    so the identity binds every translation row. They cover every
    image ``d = x_receiver - x_source + n @ vectors`` with ``|d| < radius``
    while endpoint fractional separations stay below ``fractional_excursion``.
    Nonperiodic axes of the enumerating cell have extent zero.
    ``axis_reach`` is ``||inverse[:, i]||`` at enumeration; rank and condition
    evidence come from the host SVD.
    """

    shifts: Array
    extents: tuple[int, ...] = eqx.field(static=True)
    radius: float = eqx.field(static=True)
    fractional_excursion: tuple[float, ...] = eqx.field(static=True)
    axis_reach: tuple[float, ...] = eqx.field(static=True)
    minimum_singular_value: float = eqx.field(static=True)
    condition_number: float = eqx.field(static=True)
    image_count: int = eqx.field(static=True)
    maximum_image_count: int = eqx.field(static=True)
    cell_id: str = eqx.field(static=True)
    stencil_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        extents: tuple[int, ...],
        radius: float,
        fractional_excursion: tuple[float, ...],
        axis_reach: tuple[float, ...],
        minimum_singular_value: float,
        condition_number: float,
        maximum_image_count: int,
        cell_id: str,
    ) -> None:
        bounds = tuple(extents)
        if not bounds or any(
            isinstance(item, (bool, np.bool_))
            or not isinstance(item, (int, np.integer))
            or item < 0
            for item in bounds
        ):
            raise ValueError(
                "PeriodicImageStencil.extents must contain nonnegative integers per lattice axis; got an empty or invalid extent bank"
            )
        if len(fractional_excursion) != len(bounds) or len(axis_reach) != len(bounds):
            raise ValueError(
                "PeriodicImageStencil.fractional_excursion must align with extents and axis_reach; got incompatible bank lengths"
            )
        limit = int(maximum_image_count)
        count = prod(2 * int(item) + 1 for item in bounds)
        if limit <= 0 or count > limit:
            raise ValueError(
                "PeriodicImageStencil.maximum_image_count must cover the image stencil; got a nonpositive limit or too many images"
            )
        self.extents = tuple(int(item) for item in bounds)
        self.shifts = jnp.asarray(complete_image_shifts(self.extents))
        self.radius = float(radius)
        self.fractional_excursion = tuple(float(item) for item in fractional_excursion)
        self.axis_reach = tuple(float(item) for item in axis_reach)
        self.minimum_singular_value = float(minimum_singular_value)
        self.condition_number = float(condition_number)
        self.image_count = count
        self.maximum_image_count = limit
        self.cell_id = str(cell_id)
        self.stencil_id = canonical_fingerprint(
            {
                "kind": "periodic-image-stencil",
                "cell": self.cell_id,
                "extents": list(self.extents),
                "radius": self.radius,
                "fractional_excursion": list(self.fractional_excursion),
                "maximum_image_count": self.maximum_image_count,
                "axis_reach": list(self.axis_reach),
            }
        )


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
    maximum_condition_number: float | None = eqx.field(static=True)
    maximum_image_count: int = eqx.field(static=True)
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
    ) -> None:
        matrix = np.asarray(vectors)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError(
                "PeriodicCell.vectors must have shape (rank > 0, d > 0); got an empty or nonmatrix bank"
            )
        rank, ambient_dimension = matrix.shape
        if rank > ambient_dimension:
            raise ValueError(
                "PeriodicCell.vectors must have rank at most the ambient dimension; got too many lattice rows"
            )
        dtype = np.result_type(matrix.dtype, np.float32)
        matrix = matrix.astype(dtype, copy=False)
        if np.any(~np.isfinite(matrix)):
            raise ValueError("PeriodicCell.vectors must be finite; got a nonfinite entry")
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        threshold = np.finfo(dtype).eps * max(matrix.shape) * singular_values[0]
        if singular_values[-1] <= threshold:
            raise ValueError(
                "PeriodicCell.vectors must have full row rank; got a deficient lattice"
            )

        origin_host = (
            np.zeros((ambient_dimension,), dtype=dtype)
            if origin is None
            else np.asarray(origin, dtype=dtype)
        )
        if origin_host.shape != (ambient_dimension,) or np.any(~np.isfinite(origin_host)):
            raise ValueError(
                "PeriodicCell.origin must be a finite ambient-dimension vector; got another shape or a nonfinite value"
            )
        axes = (
            (True,) * rank
            if periodic_axes is None
            else tuple(bool(value) for value in periodic_axes)
        )
        if len(axes) != rank:
            raise ValueError(
                "PeriodicCell.periodic_axes must align with lattice rows; got another axis-bank length"
            )

        condition = float(singular_values[0] / singular_values[-1])
        certified_condition = (
            condition
            if maximum_condition_number is None
            else float(maximum_condition_number)
        )
        if not np.isfinite(certified_condition) or certified_condition < condition:
            raise ValueError(
                "PeriodicCell.maximum_condition_number must be finite and cover the lattice condition; got an invalid condition bound"
            )
        extent = max(1, ceil(0.5 + certified_condition * sqrt(rank)))
        limit = positive_integer(maximum_image_count, "maximum_image_count")
        # Admit the scalar stencil size before enumerating condition-sized shifts.
        image_count = (2 * extent + 1) ** sum(axes)
        if image_count > limit:
            raise ValueError(
                f"PeriodicCell.maximum_image_count must cover the lattice image candidates; got {image_count} > {limit}"
            )
        choices = [range(-extent, extent + 1) if axis else (0,) for axis in axes]
        shifts = np.asarray(tuple(product(*choices)), dtype=np.int32)

        gram = matrix @ matrix.T
        # These coordinates become immutable host preparation, so tracing a
        # mathematical derivative would charge unused Krylov work.
        coordinate_solve = solve(
            LinearSystem(DenseLinearOperator(gram)),
            matrix,
            policy=LinearSolvePolicy(
                DenseLU(), differentiation=DifferentiationPolicy("none")
            ),
        )
        inverse = np.asarray(coordinate_solve.value).T
        reciprocal = _TWO_PI * inverse.T
        measure = float(np.sqrt(np.linalg.det(gram)))
        if not np.isfinite(measure) or measure <= np.finfo(dtype).eps:
            raise ValueError(
                "PeriodicCell.cell_measure must be finite and positive above the dtype threshold; got a singular or nonfinite measure"
            )
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
        self.periodic_mask = jnp.asarray(axes, dtype=jnp.bool_)
        self.image_shifts = jnp.asarray(shifts, dtype=jnp.int32)
        self.periodic_axes = axes
        self.cell_measure = measure
        self.volume = measure
        self.unique_image_radius = unique_radius
        self.condition_number = condition
        self.certified_condition_number = certified_condition
        self.image_extent = extent
        self.maximum_condition_number = (
            None if maximum_condition_number is None else float(maximum_condition_number)
        )
        self.maximum_image_count = limit
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
                "maximum_condition_number": self.maximum_condition_number,
            }
        )

    def validate_restored(self) -> None:
        """Validate retained controls and every constructor-derived lattice field."""
        replay = PeriodicCell(
            self.vectors,
            origin=self.origin,
            periodic_axes=self.periodic_axes,
            maximum_condition_number=self.maximum_condition_number,
            maximum_image_count=self.maximum_image_count,
        )
        for name, value, expected in (
            ("origin", self.origin, replay.origin),
            ("vectors", self.vectors, replay.vectors),
            ("inverse_vectors", self.inverse_vectors, replay.inverse_vectors),
            ("reciprocal_vectors", self.reciprocal_vectors, replay.reciprocal_vectors),
            ("periodic_mask", self.periodic_mask, replay.periodic_mask),
            ("image_shifts", self.image_shifts, replay.image_shifts),
        ):
            if array_tree_fingerprint(value) != array_tree_fingerprint(expected):
                raise ValueError(
                    f"PeriodicCell.{name} must match constructor-derived lattice arrays; got a changed stored value"
                )
        for name, value, expected in (
            ("periodic_axes", self.periodic_axes, replay.periodic_axes),
            ("cell_measure", self.cell_measure, replay.cell_measure),
            ("volume", self.volume, replay.volume),
            ("unique_image_radius", self.unique_image_radius, replay.unique_image_radius),
            ("condition_number", self.condition_number, replay.condition_number),
            (
                "certified_condition_number",
                self.certified_condition_number,
                replay.certified_condition_number,
            ),
            ("image_extent", self.image_extent, replay.image_extent),
            (
                "maximum_condition_number",
                self.maximum_condition_number,
                replay.maximum_condition_number,
            ),
            ("maximum_image_count", self.maximum_image_count, replay.maximum_image_count),
            ("cell_id", self.cell_id, replay.cell_id),
        ):
            if value != expected:
                raise ValueError(
                    f"PeriodicCell.{name} must match constructor-derived metadata; got a changed stored value"
                )

    @property
    def rank(self) -> int:
        return self.vectors.shape[0]

    @property
    def ambient_dimension(self) -> int:
        return self.vectors.shape[1]

    @property
    def fully_periodic(self) -> bool:
        return all(self.periodic_axes)

    @property
    def box_id(self) -> str:
        return self.cell_id

    def _require_ambient(self, value: Array, noun: str, /) -> None:
        if not value.shape or value.shape[-1] != self.ambient_dimension:
            raise ValueError(
                f"PeriodicCell.{noun} must end in ambient dimension {self.ambient_dimension}; got {value.shape}"
            )

    def _require_fractional(self, value: Array, /) -> None:
        if not value.shape or value.shape[-1] != self.rank:
            raise ValueError(
                f"PeriodicCell.fractional must end in lattice rank {self.rank}; got {value.shape}"
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

    def inverse_for_vectors_with_status(
        self, vectors: ArrayLike, /
    ) -> tuple[Array, Array]:
        """Return the dynamic right inverse and its solve status without raising."""
        matrix = jnp.asarray(vectors, dtype=self.vectors.dtype)
        if matrix.shape != self.vectors.shape:
            raise ValueError(
                "PeriodicCell.vectors must match the prepared lattice shape; got incompatible dynamic vectors"
            )
        return lattice_right_inverse_with_status(matrix)

    def fractional_with_vectors_with_status(
        self, position: ArrayLike, vectors: ArrayLike, /
    ) -> tuple[Array, Array]:
        """Return fractional coordinates under ``vectors`` and the solve status."""
        value = jnp.asarray(position)
        self._require_ambient(value, "positions")
        inverse, successful = self.inverse_for_vectors_with_status(vectors)
        fractional = contract(
            "...i,ij->...j",
            value - self.origin.astype(value.dtype),
            inverse.astype(value.dtype),
            backend="jax",
        )
        return fractional, successful

    def inverse_for_vectors(self, vectors: ArrayLike, /) -> Array:
        matrix = jnp.asarray(vectors, dtype=self.vectors.dtype)
        if matrix.shape != self.vectors.shape:
            raise ValueError(
                "PeriodicCell.vectors must match the prepared lattice shape; got incompatible dynamic vectors"
            )
        return lattice_right_inverse(matrix)

    def measure_with_vectors(self, vectors: ArrayLike, /) -> tuple[Array, Array]:
        """Return the dynamic cell measure (volume for full rank) and its status."""
        matrix = jnp.asarray(vectors)
        if matrix.shape[-2:] != self.vectors.shape:
            raise ValueError(
                "PeriodicCell.vectors must match the prepared lattice shape; got incompatible dynamic vectors"
            )
        return lattice_measure(matrix)

    def axis_reach_with_vectors(self, vectors: ArrayLike, /) -> Array:
        """Return ``||inverse[:, i]||``, the reciprocal perpendicular lattice heights.

        Any displacement ``d = (delta_s + n) @ vectors + d_perp`` satisfies
        ``|delta_s_i + n_i| <= |d| * reach_i``; these per-axis
        integer bounds drive image enumeration and its certificates.
        """
        inverse = self.inverse_for_vectors(vectors)
        return jnp.sqrt(jnp.sum(inverse * inverse, axis=0))

    def image_stencil(
        self,
        radius: float,
        /,
        *,
        maximum_image_count: int,
        cell_vectors: ArrayLike | None = None,
        fractional_excursion: float | tuple[float, ...] = 1.0,
    ) -> PeriodicImageStencil:
        """Enumerate every integer translation that can reach within ``radius``.

        Unlike the nearest-image ``image_shifts`` stencil, this set covers every
        ``d = x_receiver - x_source + n @ vectors`` with ``|d| < radius``
        whenever the endpoint fractional separation on each periodic axis is
        below ``fractional_excursion`` (``1`` for wrapped coordinates).  The
        bound uses the host right inverse and refuses conditioning outside the
        cell certificate or a stencil larger than ``maximum_image_count``.
        """
        value = float(radius)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                "PeriodicCell.radius must be finite and positive for image enumeration; got an invalid radius"
            )
        limit = int(maximum_image_count)
        if limit <= 0:
            raise ValueError(
                "PeriodicCell.maximum_image_count must be positive; got a nonpositive image limit"
            )
        matrix = np.asarray(
            self.vectors if cell_vectors is None else cell_vectors, dtype=np.float64
        )
        if matrix.shape != tuple(self.vectors.shape) or np.any(~np.isfinite(matrix)):
            raise ValueError(
                "PeriodicCell.cell_vectors must be finite and match the prepared lattice shape; got incompatible enumeration vectors"
            )
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        if singular_values[-1] <= np.finfo(np.float64).eps * singular_values[0]:
            raise ValueError(
                "PeriodicCell.cell_vectors must have full row rank for image enumeration; got a deficient lattice"
            )
        condition = float(singular_values[0] / singular_values[-1])
        if condition > self.certified_condition_number:
            raise ValueError(
                "PeriodicCell.cell_vectors must satisfy the prepared condition certificate; got a larger condition number"
            )
        excursion = (
            (float(fractional_excursion),) * self.rank
            if isinstance(fractional_excursion, (int, float))
            else tuple(float(item) for item in fractional_excursion)
        )
        if len(excursion) != self.rank or any(
            not np.isfinite(item) or item < 0.0 for item in excursion
        ):
            raise ValueError(
                "PeriodicCell.fractional_excursion must be finite, nonnegative, and lattice-rank aligned; got invalid excursion controls"
            )
        # Image budget derivation is host-only, not a differentiable solve.
        coordinate_solve = solve(
            LinearSystem(DenseLinearOperator(matrix @ matrix.T)),
            matrix,
            policy=LinearSolvePolicy(
                DenseLU(), differentiation=DifferentiationPolicy("none")
            ),
        )
        inverse = np.asarray(coordinate_solve.value).T
        reach = np.sqrt(np.sum(inverse * inverse, axis=0))
        extents = tuple(
            int(np.floor(excursion[axis] + value * reach[axis])) if periodic else 0
            for axis, periodic in enumerate(self.periodic_axes)
        )
        count = 1
        for extent in extents:
            count *= 2 * extent + 1
        if count > limit:
            raise ValueError(
                f"PeriodicCell.maximum_image_count must cover enumerated translations; got {count} > {limit}"
            )
        return PeriodicImageStencil(
            extents=extents,
            radius=value,
            fractional_excursion=excursion,
            axis_reach=tuple(float(item) for item in reach),
            minimum_singular_value=float(singular_values[-1]),
            condition_number=condition,
            maximum_image_count=limit,
            cell_id=self.cell_id,
        )

    def image_stencil_margin(
        self,
        stencil: PeriodicImageStencil,
        vectors: ArrayLike,
        radius: ArrayLike,
        fractional_excursion: ArrayLike,
        /,
    ) -> Array:
        """Return the per-axis completeness margin of ``stencil`` under ``vectors``.

        Every image with ``|d| < radius`` lies in the stencil when all periodic
        margins ``extent_i + 1 - excursion_i - radius * reach_i`` are positive.
        Nonperiodic axes report ``+inf``.  The reach uses the native dynamic
        lattice solve, so the result is a traced certificate.
        """
        if stencil.cell_id != self.cell_id:
            raise ValueError(
                "PeriodicCell.stencil must bind this cell_id; got another lattice identity"
            )
        reach = self.axis_reach_with_vectors(vectors)
        excursion = jnp.asarray(fractional_excursion, dtype=reach.dtype)
        extents = jnp.asarray(stencil.extents, dtype=reach.dtype)
        margin = (
            extents + 1.0 - excursion - jnp.asarray(radius, dtype=reach.dtype) * reach
        )
        return jnp.where(self.periodic_mask, margin, jnp.inf)

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
            raise ValueError(
                "PeriodicCell.vectors must match the prepared lattice shape; got incompatible dynamic vectors"
            )
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
            raise ValueError(
                "PeriodicCell.radius must be finite and positive; got an invalid interaction radius"
            )
        if value >= self.unique_image_radius:
            raise ValueError(
                "PeriodicCell.radius must remain below the unique-image certificate; got an unsupported interaction radius"
            )


__all__ = [
    "PeriodicCell",
    "PeriodicImageStencil",
    "lattice_measure",
    "lattice_right_inverse_with_status",
]
