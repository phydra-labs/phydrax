#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..metrix import SphereLaplacianLevels
from ._base import AbstractPositiveDefiniteKernel
from ._spectral import AbstractSpectralMultiplier


def _spectral_coefficients(
    multiplier: AbstractSpectralMultiplier,
    eigenvalues: Array,
    spectral_dimension: float,
    /,
    *,
    level_multiplicities: Array | None,
    normalize: bool,
) -> Array:
    log_weights = multiplier.log_weights(eigenvalues, spectral_dimension)
    if log_weights.shape != eigenvalues.shape:
        raise ValueError("Spectral multiplier output must match the compact levels.")
    if level_multiplicities is not None:
        log_weights = log_weights + jnp.log(level_multiplicities)
    log_weights = eqx.error_if(
        log_weights,
        jnp.any(jnp.isnan(log_weights)) | jnp.any(log_weights == jnp.inf),
        "Compact spectral log weights must be finite or negative infinity.",
    )
    if normalize:
        maximum = jnp.max(log_weights)
        log_weights = eqx.error_if(
            log_weights,
            ~jnp.isfinite(maximum),
            "Normalized compact spectral weights cannot all vanish.",
        )
        log_weights = log_weights - (
            maximum + jnp.log(jnp.sum(jnp.exp(log_weights - maximum)))
        )
    return jnp.exp(log_weights)


def _sphere_points(
    points: ArrayLike, ambient_dimension: int, tolerance: float, /
) -> Array:
    array = jnp.asarray(points, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or int(array.shape[1]) != ambient_dimension:
        raise ValueError(
            f"Sphere points must have shape ({ambient_dimension},) or "
            f"(point, {ambient_dimension})."
        )
    norms = jnp.sum(array * array, axis=-1)
    array = eqx.error_if(
        array,
        jnp.any(~jnp.isfinite(array)) | jnp.any(jnp.abs(norms - 1.0) > tolerance),
        "Sphere points must be finite unit vectors.",
    )
    return array / jnp.sqrt(norms)[:, None]


def _sphere_series(
    similarity: Array,
    coefficients: Array,
    dimension: int,
    max_level: int,
    /,
) -> Array:
    value = coefficients[0] * jnp.ones_like(similarity)
    if max_level == 0:
        return value
    previous = jnp.ones_like(similarity)
    current = similarity
    value = value + coefficients[1] * current
    for level in range(1, max_level):
        if dimension == 1:
            following = 2.0 * similarity * current - previous
        else:
            alpha = 0.5 * (dimension - 1)
            denominator = level + 2.0 * alpha
            following = (
                2.0 * (level + alpha) / denominator * similarity * current
                - level / denominator * previous
            )
        value = value + coefficients[level + 1] * following
        previous, current = current, following
    return value


class SphereSpectralKernel(AbstractPositiveDefiniteKernel):
    """Isotropic heat or Matérn kernel on a unit sphere via addition theorems."""

    spectrum: SphereLaplacianLevels
    multiplier: AbstractSpectralMultiplier
    normalize: bool = eqx.field(static=True)
    membership_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        max_level: int,
        multiplier: AbstractSpectralMultiplier,
        /,
        *,
        normalize: bool = True,
        membership_tolerance: float = 1e-6,
    ):
        if not isinstance(multiplier, AbstractSpectralMultiplier):
            raise TypeError("multiplier must be an AbstractSpectralMultiplier.")
        if not 0.0 < float(membership_tolerance) < 1.0:
            raise ValueError(
                "membership_tolerance must lie strictly between zero and one."
            )
        self.spectrum = SphereLaplacianLevels(dimension, max_level)
        self.multiplier = multiplier
        self.normalize = bool(normalize)
        self.membership_tolerance = float(membership_tolerance)

    def _coefficients(self) -> Array:
        return _spectral_coefficients(
            self.multiplier,
            self.spectrum.eigenvalues,
            float(self.spectrum.dimension),
            level_multiplicities=jnp.asarray(self.spectrum.multiplicities, dtype=float),
            normalize=self.normalize,
        )

    def pairwise(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_points = _sphere_points(
            left, self.spectrum.dimension + 1, self.membership_tolerance
        )
        right_points = _sphere_points(
            right, self.spectrum.dimension + 1, self.membership_tolerance
        )
        if left_points.shape[0] != 1 or right_points.shape[0] != 1:
            raise ValueError("pairwise requires one sphere point per argument.")
        similarity = jnp.clip(jnp.dot(left_points[0], right_points[0]), -1.0, 1.0)
        return _sphere_series(
            similarity,
            self._coefficients(),
            self.spectrum.dimension,
            self.spectrum.max_level,
        )

    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_points = _sphere_points(
            left, self.spectrum.dimension + 1, self.membership_tolerance
        )
        right_points = _sphere_points(
            right, self.spectrum.dimension + 1, self.membership_tolerance
        )
        similarities = jnp.clip(left_points @ right_points.T, -1.0, 1.0)
        return _sphere_series(
            similarities,
            self._coefficients(),
            self.spectrum.dimension,
            self.spectrum.max_level,
        )

    def diagonal(self, points: ArrayLike, /) -> Array:
        point_design = _sphere_points(
            points, self.spectrum.dimension + 1, self.membership_tolerance
        )
        return jnp.full(
            (point_design.shape[0],),
            jnp.sum(self._coefficients()),
            dtype=point_design.dtype,
        )

    @property
    def max_derivative_order(self) -> None:
        return None

    @property
    def is_unit_diagonal(self) -> bool:
        return self.normalize

    @property
    def kernel_id(self) -> str:
        return (
            f"SphereSpectralKernel[S{self.spectrum.dimension};"
            f"levels={self.spectrum.max_level};{self.multiplier.multiplier_id};"
            f"normalize={int(self.normalize)}]"
        )


def _real_matrix_points(
    points: ArrayLike,
    rows: int,
    columns: int,
    tolerance: float,
    /,
    *,
    special: bool,
) -> Array:
    array = jnp.asarray(points, dtype=float)
    if array.shape == (rows, columns):
        array = array[None, :, :]
    elif array.ndim == 1 and int(array.size) == rows * columns:
        array = array.reshape((1, rows, columns))
    elif array.ndim == 2 and int(array.shape[1]) == rows * columns:
        array = array.reshape((array.shape[0], rows, columns))
    if array.ndim != 3 or tuple(array.shape[1:]) != (rows, columns):
        raise ValueError(
            f"Matrix points must have trailing shape ({rows}, {columns}) or be flattened."
        )
    gram = jnp.swapaxes(array, -1, -2) @ array
    identity = jnp.eye(columns, dtype=array.dtype)
    invalid = jnp.any(~jnp.isfinite(array)) | jnp.any(
        jnp.abs(gram - identity) > tolerance
    )
    if special:
        invalid = invalid | jnp.any(jnp.abs(jnp.linalg.det(array) - 1.0) > tolerance)
    return eqx.error_if(
        array,
        invalid,
        "Matrix points do not satisfy the declared compact-manifold constraints.",
    )


def _complex_matrix_points(
    points: ArrayLike,
    dimension: int,
    tolerance: float,
    /,
) -> Array:
    array = jnp.asarray(points)
    if array.shape == (dimension, dimension):
        array = array[None, :, :]
    elif array.ndim == 1 and int(array.size) == dimension * dimension:
        array = array.reshape((1, dimension, dimension))
    elif array.ndim == 2 and int(array.shape[1]) == dimension * dimension:
        array = array.reshape((array.shape[0], dimension, dimension))
    if array.ndim != 3 or tuple(array.shape[1:]) != (dimension, dimension):
        raise ValueError(
            "Special-unitary points must be square matrices or flattened square matrices."
        )
    gram = jnp.swapaxes(jnp.conj(array), -1, -2) @ array
    identity = jnp.eye(dimension, dtype=array.dtype)
    invalid = (
        jnp.any(~jnp.isfinite(jnp.real(array)))
        | jnp.any(~jnp.isfinite(jnp.imag(array)))
        | jnp.any(jnp.abs(gram - identity) > tolerance)
        | jnp.any(jnp.abs(jnp.linalg.det(array) - 1.0) > tolerance)
    )
    return eqx.error_if(
        array,
        invalid,
        "Matrix points do not satisfy the special-unitary constraints.",
    )


class AbstractHomogeneousPolynomialKernel(AbstractPositiveDefiniteKernel):
    """Finite nonnegative expansion of normalized homogeneous-space features."""

    multiplier: AbstractSpectralMultiplier
    max_level: int = eqx.field(static=True)
    spectral_dimension: float = eqx.field(static=True)
    casimir_shift: float = eqx.field(static=True)
    normalize: bool = eqx.field(static=True)
    membership_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        multiplier: AbstractSpectralMultiplier,
        max_level: int,
        /,
        *,
        spectral_dimension: float,
        casimir_shift: float,
        normalize: bool,
        membership_tolerance: float,
    ):
        if not isinstance(multiplier, AbstractSpectralMultiplier):
            raise TypeError("multiplier must be an AbstractSpectralMultiplier.")
        if int(max_level) < 0:
            raise ValueError("max_level must be nonnegative.")
        if float(spectral_dimension) <= 0.0 or float(casimir_shift) < 0.0:
            raise ValueError("Homogeneous-space spectral dimensions are invalid.")
        if float(membership_tolerance) <= 0.0:
            raise ValueError("membership_tolerance must be positive.")
        self.multiplier = multiplier
        self.max_level = int(max_level)
        self.spectral_dimension = float(spectral_dimension)
        self.casimir_shift = float(casimir_shift)
        self.normalize = bool(normalize)
        self.membership_tolerance = float(membership_tolerance)

    @abstractmethod
    def _similarities(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        raise NotImplementedError

    def _coefficients(self) -> Array:
        levels = jnp.arange(self.max_level + 1, dtype=float)
        eigenvalues = levels * (levels + self.casimir_shift)
        return _spectral_coefficients(
            self.multiplier,
            eigenvalues,
            self.spectral_dimension,
            level_multiplicities=None,
            normalize=self.normalize,
        )

    def _series(self, similarity: Array, /) -> Array:
        coefficients = self._coefficients()
        value = jnp.zeros_like(similarity, dtype=jnp.result_type(similarity, float))
        for coefficient in coefficients[::-1]:
            value = coefficient + similarity * value
        return value

    def pairwise(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        similarities = self._similarities(left, right)
        if similarities.shape != (1, 1):
            raise ValueError(
                "pairwise requires one homogeneous-space point per argument."
            )
        return jnp.real(self._series(similarities)[0, 0])

    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        return jnp.real(self._series(self._similarities(left, right)))

    def diagonal(self, points: ArrayLike, /) -> Array:
        similarities = self._similarities(points, points)
        return jnp.real(jnp.diag(self._series(similarities)))

    @property
    def max_derivative_order(self) -> None:
        return None

    @property
    def is_unit_diagonal(self) -> bool:
        return False


class SpecialOrthogonalCharacterKernel(AbstractHomogeneousPolynomialKernel):
    """Bi-invariant SO(n) kernel from standard-representation character powers."""

    dimension: int = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        max_level: int,
        multiplier: AbstractSpectralMultiplier,
        /,
        *,
        normalize: bool = True,
        membership_tolerance: float = 1e-6,
    ):
        resolved = int(dimension)
        if resolved < 2:
            raise ValueError("SO(n) dimension must be at least two.")
        self.dimension = resolved
        super().__init__(
            multiplier,
            max_level,
            spectral_dimension=0.5 * resolved * (resolved - 1),
            casimir_shift=float(resolved - 2),
            normalize=normalize,
            membership_tolerance=membership_tolerance,
        )

    def _similarities(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_matrices = _real_matrix_points(
            left,
            self.dimension,
            self.dimension,
            self.membership_tolerance,
            special=True,
        )
        right_matrices = _real_matrix_points(
            right,
            self.dimension,
            self.dimension,
            self.membership_tolerance,
            special=True,
        )
        return jnp.einsum("aij,bij->ab", left_matrices, right_matrices) / self.dimension

    @property
    def kernel_id(self) -> str:
        return (
            f"SpecialOrthogonalCharacterKernel[SO({self.dimension});"
            f"levels={self.max_level};{self.multiplier.multiplier_id}]"
        )


class SpecialUnitaryCharacterKernel(AbstractHomogeneousPolynomialKernel):
    """Bi-invariant real SU(n) kernel from tensor-character powers."""

    dimension: int = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        max_level: int,
        multiplier: AbstractSpectralMultiplier,
        /,
        *,
        normalize: bool = True,
        membership_tolerance: float = 1e-6,
    ):
        resolved = int(dimension)
        if resolved < 2:
            raise ValueError("SU(n) dimension must be at least two.")
        self.dimension = resolved
        super().__init__(
            multiplier,
            max_level,
            spectral_dimension=float(resolved * resolved - 1),
            casimir_shift=float(resolved - 1),
            normalize=normalize,
            membership_tolerance=membership_tolerance,
        )

    def _similarities(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_matrices = _complex_matrix_points(
            left, self.dimension, self.membership_tolerance
        )
        right_matrices = _complex_matrix_points(
            right, self.dimension, self.membership_tolerance
        )
        return (
            jnp.einsum("aij,bij->ab", jnp.conj(left_matrices), right_matrices)
            / self.dimension
        )

    @property
    def kernel_id(self) -> str:
        return (
            f"SpecialUnitaryCharacterKernel[SU({self.dimension});"
            f"levels={self.max_level};{self.multiplier.multiplier_id}]"
        )


class StiefelSpectralKernel(AbstractHomogeneousPolynomialKernel):
    """O(n)-invariant polynomial spectral kernel on Stiefel frames V_p(R^n)."""

    ambient_dimension: int = eqx.field(static=True)
    frame_dimension: int = eqx.field(static=True)

    def __init__(
        self,
        ambient_dimension: int,
        frame_dimension: int,
        max_level: int,
        multiplier: AbstractSpectralMultiplier,
        /,
        *,
        normalize: bool = True,
        membership_tolerance: float = 1e-6,
    ):
        ambient = int(ambient_dimension)
        frame = int(frame_dimension)
        if ambient < 2 or frame <= 0 or frame > ambient:
            raise ValueError("Stiefel dimensions require 0 < frame_dimension <= ambient.")
        self.ambient_dimension = ambient
        self.frame_dimension = frame
        super().__init__(
            multiplier,
            max_level,
            spectral_dimension=float(ambient * frame - frame * (frame + 1) / 2),
            casimir_shift=float(ambient - 2),
            normalize=normalize,
            membership_tolerance=membership_tolerance,
        )

    def _similarities(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_frames = _real_matrix_points(
            left,
            self.ambient_dimension,
            self.frame_dimension,
            self.membership_tolerance,
            special=False,
        )
        right_frames = _real_matrix_points(
            right,
            self.ambient_dimension,
            self.frame_dimension,
            self.membership_tolerance,
            special=False,
        )
        return jnp.einsum("aij,bij->ab", left_frames, right_frames) / self.frame_dimension

    @property
    def kernel_id(self) -> str:
        return (
            f"StiefelSpectralKernel[V_{self.frame_dimension}(R^{self.ambient_dimension});"
            f"levels={self.max_level};{self.multiplier.multiplier_id}]"
        )


class GrassmannSpectralKernel(AbstractHomogeneousPolynomialKernel):
    """Quotient-invariant polynomial spectral kernel on Gr(p, n)."""

    ambient_dimension: int = eqx.field(static=True)
    subspace_dimension: int = eqx.field(static=True)

    def __init__(
        self,
        ambient_dimension: int,
        subspace_dimension: int,
        max_level: int,
        multiplier: AbstractSpectralMultiplier,
        /,
        *,
        normalize: bool = True,
        membership_tolerance: float = 1e-6,
    ):
        ambient = int(ambient_dimension)
        subspace = int(subspace_dimension)
        if ambient < 2 or subspace <= 0 or subspace >= ambient:
            raise ValueError("Grassmann dimensions require 0 < subspace < ambient.")
        self.ambient_dimension = ambient
        self.subspace_dimension = subspace
        super().__init__(
            multiplier,
            max_level,
            spectral_dimension=float(subspace * (ambient - subspace)),
            casimir_shift=float(ambient - 2),
            normalize=normalize,
            membership_tolerance=membership_tolerance,
        )

    def _similarities(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_frames = _real_matrix_points(
            left,
            self.ambient_dimension,
            self.subspace_dimension,
            self.membership_tolerance,
            special=False,
        )
        right_frames = _real_matrix_points(
            right,
            self.ambient_dimension,
            self.subspace_dimension,
            self.membership_tolerance,
            special=False,
        )
        overlap = jnp.einsum("anp,bnq->abpq", left_frames, right_frames)
        return jnp.sum(overlap * overlap, axis=(-1, -2)) / self.subspace_dimension

    @property
    def kernel_id(self) -> str:
        return (
            f"GrassmannSpectralKernel[Gr({self.subspace_dimension},"
            f"{self.ambient_dimension});levels={self.max_level};"
            f"{self.multiplier.multiplier_id}]"
        )


__all__ = [
    "GrassmannSpectralKernel",
    "SpecialOrthogonalCharacterKernel",
    "SpecialUnitaryCharacterKernel",
    "SphereSpectralKernel",
    "StiefelSpectralKernel",
]
