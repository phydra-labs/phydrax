#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from math import prod
from typing import cast, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike
from matfree.low_rank import cholesky_partial_pivot
from matfree.stochtrace import nystrom_eigh

from .._strict import StrictModule
from ..discretization._tensor import AbstractStrongFormDiscretization


def _canonicalize_signs(modes: np.ndarray, /) -> np.ndarray:
    out = np.array(modes, dtype=float, copy=True)
    for column in range(out.shape[1]):
        pivot = int(np.argmax(np.abs(out[:, column])))
        if out[pivot, column] < 0.0:
            out[:, column] *= -1.0
    return out


def _basis_digest(
    *,
    state_shape: tuple[int, ...],
    modes: np.ndarray,
    eigenvalues: np.ndarray,
    weights: np.ndarray,
    mode_ids: tuple[str, ...],
    field_space_id: str | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"spatial-noise-basis-v1\0")
    digest.update(repr(state_shape).encode("ascii"))
    digest.update((field_space_id or "").encode("utf-8"))
    for mode_id in mode_ids:
        digest.update(mode_id.encode("utf-8"))
        digest.update(b"\0")
    for array in (modes, eigenvalues, weights):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(repr(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


_ApproximationMethod = Literal[
    "dense_eigh",
    "pivoted_cholesky",
    "randomized_nystrom",
]
_ResidualKind = Literal["relative_frobenius", "relative_trace"]


class SpatialNoiseApproximation(StrictModule):
    """Auditable diagnostics for an approximate spatial covariance factor."""

    method: _ApproximationMethod = eqx.field(static=True)
    matrix_size: int = eqx.field(static=True)
    requested_rank: int = eqx.field(static=True)
    retained_rank: int = eqx.field(static=True)
    residual_kind: _ResidualKind = eqx.field(static=True)
    residual_estimate: float = eqx.field(static=True)
    absolute_residual_estimate: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    seed: tuple[int, ...] | None = eqx.field(static=True)
    sketch_size: int | None = eqx.field(static=True)
    converged: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        method: _ApproximationMethod,
        matrix_size: int,
        requested_rank: int,
        retained_rank: int,
        residual_kind: _ResidualKind,
        residual_estimate: float,
        absolute_residual_estimate: float,
        tolerance: float,
        seed: Sequence[int] | None = None,
        sketch_size: int | None = None,
    ):
        if method not in (
            "dense_eigh",
            "pivoted_cholesky",
            "randomized_nystrom",
        ):
            raise ValueError(f"Unknown spatial-noise approximation method {method!r}.")
        size = int(matrix_size)
        requested = int(requested_rank)
        retained = int(retained_rank)
        if size <= 0:
            raise ValueError("matrix_size must be positive.")
        if requested <= 0 or requested > size:
            raise ValueError("requested_rank must lie within the matrix size.")
        if retained <= 0 or retained > requested:
            raise ValueError("retained_rank must lie in [1, requested_rank].")
        if residual_kind not in ("relative_frobenius", "relative_trace"):
            raise ValueError(f"Unknown residual kind {residual_kind!r}.")
        residual = float(residual_estimate)
        absolute = float(absolute_residual_estimate)
        threshold = float(tolerance)
        if not np.isfinite(residual) or residual < 0.0:
            raise ValueError("residual_estimate must be finite and non-negative.")
        if not np.isfinite(absolute) or absolute < 0.0:
            raise ValueError(
                "absolute_residual_estimate must be finite and non-negative."
            )
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("tolerance must be finite and non-negative.")
        resolved_sketch = None if sketch_size is None else int(sketch_size)
        if resolved_sketch is not None and resolved_sketch < retained:
            raise ValueError("sketch_size must be at least retained_rank.")

        self.method = method
        self.matrix_size = size
        self.requested_rank = requested
        self.retained_rank = retained
        self.residual_kind = residual_kind
        self.residual_estimate = residual
        self.absolute_residual_estimate = absolute
        self.tolerance = threshold
        self.seed = None if seed is None else tuple(int(value) for value in seed)
        self.sketch_size = resolved_sketch
        self.converged = bool(residual <= threshold)

    @property
    def rank(self) -> int:
        return self.retained_rank


def _factor_eigenpairs(
    factor: ArrayLike,
    weights: np.ndarray,
    state_shape: tuple[int, ...],
    /,
    *,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a weighted covariance factor into weighted KL eigenpairs."""
    factor_host = np.asarray(factor, dtype=float)
    count = int(prod(state_shape))
    if factor_host.ndim != 2 or factor_host.shape[0] != count:
        raise ValueError(f"Covariance factor must have {count} rows.")
    if np.any(~np.isfinite(factor_host)):
        raise ValueError("Covariance factorization produced non-finite values.")
    left, singular_values, _ = np.linalg.svd(factor_host, full_matrices=False)
    retained = int(rank)
    if retained > singular_values.size:
        raise ValueError("Covariance factor has fewer columns than the requested rank.")
    root = np.sqrt(weights.reshape((-1,)))
    eigenvalues = singular_values[:retained] ** 2
    modes = left[:, :retained] / root[:, None]
    modes = _canonicalize_signs(modes)
    weighted_factor = (root[:, None] * modes) * np.sqrt(eigenvalues)[None, :]
    return (
        eigenvalues,
        modes.reshape(state_shape + (retained,)),
        weighted_factor,
    )


def _key_seed(key: ArrayLike, /) -> tuple[int, ...]:
    return tuple(int(value) for value in np.asarray(jr.key_data(key)).reshape((-1,)))


class SpatialNoiseBasis(StrictModule):
    r"""Finite-rank spatial basis for a discrete :math:`Q`-Wiener process.

    Modes satisfy :math:`\Phi^\mathsf{T}M\Phi=I`. The diffusion factor is
    :math:`B=\Phi\operatorname{diag}(\sqrt{q_1},\ldots,\sqrt{q_r})` and has shape
    ``state_shape + (rank,)``.
    """

    modes: Array
    eigenvalues: Array
    quadrature_weights: Array
    state_shape: tuple[int, ...] = eqx.field(static=True)
    mode_ids: tuple[str, ...] = eqx.field(static=True)
    field_space_id: str | None = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)
    approximation: SpatialNoiseApproximation | None = eqx.field(static=True)

    def __init__(
        self,
        modes: ArrayLike,
        eigenvalues: ArrayLike,
        /,
        *,
        quadrature_weights: ArrayLike,
        state_shape: Sequence[int] | None = None,
        mode_ids: Sequence[str] | None = None,
        field_space_id: str | None = None,
        approximation: SpatialNoiseApproximation | None = None,
        orthonormal_rtol: float = 1e-6,
        orthonormal_atol: float = 1e-7,
    ):
        raw_modes = jnp.asarray(modes)
        if jnp.iscomplexobj(raw_modes):
            raise ValueError("SpatialNoiseBasis modes must be real-valued.")
        modes_array = jnp.asarray(raw_modes, dtype=float)
        if modes_array.ndim < 2:
            raise ValueError("modes must have shape state_shape + (rank,).")
        inferred_shape = tuple(int(size) for size in modes_array.shape[:-1])
        resolved_shape = (
            inferred_shape
            if state_shape is None
            else tuple(int(size) for size in state_shape)
        )
        if not resolved_shape or any(size <= 0 for size in resolved_shape):
            raise ValueError("state_shape must contain positive dimensions.")
        if inferred_shape != resolved_shape:
            raise ValueError(
                f"modes state shape must be {resolved_shape}; got {inferred_shape}."
            )
        rank = int(modes_array.shape[-1])
        if rank <= 0 or rank > int(prod(resolved_shape)):
            raise ValueError("Noise rank must lie between one and the state size.")
        eigenvalue_array = jnp.asarray(eigenvalues, dtype=float).reshape((-1,))
        if eigenvalue_array.shape != (rank,):
            raise ValueError("eigenvalues must contain one value per noise mode.")
        weights = jnp.asarray(quadrature_weights, dtype=float)
        if tuple(weights.shape) != resolved_shape:
            raise ValueError(
                "quadrature_weights must have exact state_shape "
                f"{resolved_shape}; got {weights.shape}."
            )
        modes_host = np.asarray(modes_array, dtype=float)
        eigenvalues_host = np.asarray(eigenvalue_array, dtype=float)
        weights_host = np.asarray(weights, dtype=float)
        if np.any(~np.isfinite(modes_host)):
            raise ValueError("Noise modes must be finite.")
        if np.any(~np.isfinite(eigenvalues_host)) or np.any(eigenvalues_host < 0.0):
            raise ValueError("Noise eigenvalues must be finite and non-negative.")
        if np.any(~np.isfinite(weights_host)) or np.any(weights_host <= 0.0):
            raise ValueError("Quadrature weights must be finite and positive.")
        flat_modes = modes_host.reshape((-1, rank))
        gram = flat_modes.T @ (weights_host.reshape((-1, 1)) * flat_modes)
        if not np.allclose(
            gram,
            np.eye(rank),
            rtol=float(orthonormal_rtol),
            atol=float(orthonormal_atol),
        ):
            raise ValueError(
                "Noise modes must satisfy the weighted Gram contract Phi.T M Phi = I."
            )
        if mode_ids is None:
            identifiers = tuple(f"mode:{index}" for index in range(rank))
        else:
            identifiers = tuple(str(value) for value in mode_ids)
            if len(identifiers) != rank or any(not value for value in identifiers):
                raise ValueError("mode_ids must contain one non-empty ID per mode.")
            if len(set(identifiers)) != rank:
                raise ValueError("mode_ids must be unique.")
        if field_space_id is not None and not str(field_space_id):
            raise ValueError("field_space_id must be non-empty or None.")
        if approximation is not None:
            if not isinstance(approximation, SpatialNoiseApproximation):
                raise TypeError(
                    "approximation must be a SpatialNoiseApproximation or None."
                )
            if approximation.retained_rank != rank:
                raise ValueError(
                    "approximation.retained_rank must match the noise basis rank."
                )
        self.modes = modes_array
        self.eigenvalues = eigenvalue_array
        self.quadrature_weights = weights
        self.state_shape = resolved_shape
        self.mode_ids = identifiers
        self.field_space_id = (
            None if field_space_id is None else str(field_space_id)
        )
        self.approximation = approximation
        self.basis_id = _basis_digest(
            state_shape=resolved_shape,
            modes=modes_host,
            eigenvalues=eigenvalues_host,
            weights=weights_host,
            mode_ids=identifiers,
            field_space_id=self.field_space_id,
        )

    @property
    def rank(self) -> int:
        return int(self.eigenvalues.size)

    @property
    def noise_shape(self) -> tuple[int, ...]:
        return (self.rank,)

    @property
    def diffusion(self) -> Array:
        scale = jnp.sqrt(self.eigenvalues).reshape(
            (1,) * len(self.state_shape) + (self.rank,)
        )
        return self.modes * scale

    @property
    def diffusion_matrix(self) -> Array:
        return self.diffusion.reshape((-1, self.rank))

    def reconstructed_covariance(self) -> Array:
        factor = self.diffusion_matrix
        return factor @ factor.T

    @classmethod
    def from_modes(
        cls,
        modes: ArrayLike,
        eigenvalues: ArrayLike,
        /,
        *,
        quadrature_weights: ArrayLike,
        state_shape: Sequence[int] | None = None,
        mode_ids: Sequence[str] | None = None,
        field_space_id: str | None = None,
    ) -> "SpatialNoiseBasis":
        """Validate explicit weighted-orthonormal modes and eigenvalues."""
        return cls(
            modes,
            eigenvalues,
            quadrature_weights=quadrature_weights,
            state_shape=state_shape,
            mode_ids=mode_ids,
            field_space_id=field_space_id,
        )

    @classmethod
    def from_spectrum(
        cls,
        discretization: AbstractStrongFormDiscretization,
        spectrum: Callable[[Array], ArrayLike] | ArrayLike,
        /,
        *,
        rank: int,
    ) -> "SpatialNoiseBasis":
        r"""Select low Laplacian modes and evaluate a spectral covariance law.

        ``spectrum`` receives the retained non-negative eigenvalues of
        ``-laplacian`` and returns the corresponding :math:`Q` eigenvalues.
        """
        if not isinstance(discretization, AbstractStrongFormDiscretization):
            raise TypeError(
                "discretization must implement AbstractStrongFormDiscretization."
            )
        laplacian_eigenvalues, modes = discretization.eigenpairs(rank=int(rank))
        if callable(spectrum):
            spectrum_function = cast(Callable[[Array], ArrayLike], spectrum)
            values = spectrum_function(laplacian_eigenvalues)
        else:
            values = spectrum
        covariance_eigenvalues = jnp.asarray(values, dtype=float)
        if covariance_eigenvalues.shape == ():
            covariance_eigenvalues = jnp.full(
                (int(rank),), covariance_eigenvalues, dtype=float
            )
        covariance_eigenvalues = covariance_eigenvalues.reshape((-1,))
        mode_ids = tuple(
            f"laplacian:{index}:{float(value):.17g}"
            for index, value in enumerate(np.asarray(laplacian_eigenvalues))
        )
        return cls(
            modes,
            covariance_eigenvalues,
            quadrature_weights=discretization.quadrature_weights,
            state_shape=discretization.state_shape,
            mode_ids=mode_ids,
            field_space_id=discretization.field_spaces[0].field_space_id,
        )

    @classmethod
    def from_discrete_covariance(
        cls,
        covariance: ArrayLike,
        /,
        *,
        state_shape: Sequence[int],
        quadrature_weights: ArrayLike,
        rank: int,
        field_space_id: str | None = None,
        psd_tolerance: float = 1e-10,
    ) -> "SpatialNoiseBasis":
        """Factor a nodal covariance matrix using its quadrature-weighted KL basis."""
        shape = tuple(int(size) for size in state_shape)
        count = int(prod(shape))
        retained = int(rank)
        if retained <= 0 or retained > count:
            raise ValueError(f"rank must lie in [1, {count}].")
        tolerance = float(psd_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("psd_tolerance must be finite and non-negative.")
        covariance_host = np.asarray(covariance, dtype=float)
        if covariance_host.shape != (count, count):
            raise ValueError(
                f"covariance must have shape {(count, count)}; got {covariance_host.shape}."
            )
        if np.any(~np.isfinite(covariance_host)):
            raise ValueError("covariance must be finite.")
        if not np.allclose(covariance_host, covariance_host.T, rtol=1e-8, atol=1e-10):
            raise ValueError("covariance must be symmetric.")
        weights_host = np.asarray(quadrature_weights, dtype=float)
        if weights_host.shape != shape:
            raise ValueError(
                f"quadrature_weights must have shape {shape}; got {weights_host.shape}."
            )
        if np.any(~np.isfinite(weights_host)) or np.any(weights_host <= 0.0):
            raise ValueError("Quadrature weights must be finite and positive.")
        flat_weights = weights_host.reshape((-1,))
        root = np.sqrt(flat_weights)
        transformed = root[:, None] * covariance_host * root[None, :]
        transformed = 0.5 * (transformed + transformed.T)
        eigenvalues, weighted_modes = np.linalg.eigh(transformed)
        scale = max(1.0, float(np.max(np.abs(eigenvalues))))
        if float(np.min(eigenvalues)) < -tolerance * scale:
            raise ValueError("covariance must be positive semidefinite.")
        order = np.argsort(eigenvalues, kind="stable")[::-1]
        ordered = np.maximum(eigenvalues[order], 0.0)
        selected = ordered[:retained]
        modes = weighted_modes[:, order[:retained]] / root[:, None]
        modes = _canonicalize_signs(modes)
        absolute_residual = float(np.linalg.vector_norm(ordered[retained:]))
        total_norm = float(np.linalg.vector_norm(ordered))
        relative_residual = 0.0 if total_norm == 0.0 else absolute_residual / total_norm
        approximation = SpatialNoiseApproximation(
            method="dense_eigh",
            matrix_size=count,
            requested_rank=retained,
            retained_rank=retained,
            residual_kind="relative_frobenius",
            residual_estimate=relative_residual,
            absolute_residual_estimate=absolute_residual,
            tolerance=tolerance,
        )
        return cls(
            modes.reshape(shape + (retained,)),
            selected,
            quadrature_weights=weights_host,
            state_shape=shape,
            mode_ids=tuple(f"covariance:{index}" for index in range(retained)),
            field_space_id=field_space_id,
            approximation=approximation,
        )

    @classmethod
    def from_kernel_covariance(
        cls,
        kernel: Callable[[Array, Array], ArrayLike],
        discretization: AbstractStrongFormDiscretization,
        /,
        *,
        rank: int,
        points: ArrayLike | None = None,
        tolerance: float = 1e-6,
    ) -> "SpatialNoiseBasis":
        r"""Factor a kernel covariance with Matfree pivoted Cholesky.

        The kernel is queried only at scalar point pairs. The construction stores
        :math:`O(nr)` values and never materializes the :math:`n\times n` covariance.
        """
        if not callable(kernel):
            raise TypeError("kernel must be callable.")
        if not isinstance(discretization, AbstractStrongFormDiscretization):
            raise TypeError(
                "discretization must implement AbstractStrongFormDiscretization."
            )
        count = discretization.num_points
        retained = int(rank)
        if retained <= 0 or retained > count:
            raise ValueError(f"rank must lie in [1, {count}].")
        threshold = float(tolerance)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("tolerance must be finite and non-negative.")

        resolved_points = discretization.points if points is None else points
        if resolved_points is None:
            raise ValueError(
                "points are required when the spatial discretization has no coordinates."
            )
        point_array = jnp.asarray(resolved_points, dtype=float)
        if point_array.ndim != 2 or int(point_array.shape[0]) != count:
            raise ValueError(
                "points must have shape (discretization.num_points, coordinate_dim)."
            )
        if np.any(~np.isfinite(np.asarray(point_array))):
            raise ValueError("points must be finite.")

        weights_host = np.asarray(discretization.quadrature_weights, dtype=float)
        if weights_host.shape != discretization.state_shape:
            raise ValueError("discretization quadrature weights have an invalid shape.")
        if np.any(~np.isfinite(weights_host)) or np.any(weights_host <= 0.0):
            raise ValueError("Quadrature weights must be finite and positive.")
        root = jnp.sqrt(jnp.asarray(weights_host).reshape((-1,)))

        sample = jnp.asarray(kernel(point_array[0], point_array[0]), dtype=float)
        if sample.shape != ():
            raise ValueError("kernel must return one scalar for a pair of points.")
        if not bool(jnp.isfinite(sample)):
            raise ValueError("kernel must return finite covariance values.")

        def matrix_element(left_index, right_index):
            value = jnp.asarray(
                kernel(point_array[left_index], point_array[right_index]),
                dtype=float,
            )
            if value.shape != ():
                raise ValueError("kernel must return one scalar for a pair of points.")
            return root[left_index] * value * root[right_index]

        factorize = cholesky_partial_pivot(
            matrix_element,
            nrows=count,
            rank=retained,
        )
        weighted_cholesky, info = factorize()
        success = bool(np.asarray(info["success"]))
        if not success or np.any(~np.isfinite(np.asarray(weighted_cholesky))):
            raise ValueError(
                "Matfree pivoted Cholesky could not reach the requested rank; "
                "lower rank or use a strictly positive-definite kernel."
            )

        eigenvalues, modes, weighted_factor = _factor_eigenpairs(
            weighted_cholesky,
            weights_host,
            discretization.state_shape,
            rank=retained,
        )
        diagonal = np.asarray(
            jax.vmap(lambda index: matrix_element(index, index))(jnp.arange(count)),
            dtype=float,
        )
        residual_diagonal = diagonal - np.sum(weighted_factor**2, axis=1)
        diagonal_scale = max(1.0, float(np.max(np.abs(diagonal))))
        roundoff = 100.0 * np.finfo(float).eps * diagonal_scale
        if float(np.min(residual_diagonal)) < -max(roundoff, threshold * diagonal_scale):
            raise ValueError(
                "Pivoted Cholesky produced an invalid negative residual diagonal."
            )
        absolute_residual = float(np.sum(np.maximum(residual_diagonal, 0.0)))
        total_trace = float(np.sum(np.maximum(diagonal, 0.0)))
        relative_residual = 0.0 if total_trace == 0.0 else absolute_residual / total_trace
        approximation = SpatialNoiseApproximation(
            method="pivoted_cholesky",
            matrix_size=count,
            requested_rank=retained,
            retained_rank=retained,
            residual_kind="relative_trace",
            residual_estimate=relative_residual,
            absolute_residual_estimate=absolute_residual,
            tolerance=threshold,
        )
        return cls(
            modes,
            eigenvalues,
            quadrature_weights=weights_host,
            state_shape=discretization.state_shape,
            mode_ids=tuple(f"kernel-pivot:{index}" for index in range(retained)),
            field_space_id=discretization.field_spaces[0].field_space_id,
            approximation=approximation,
        )

    @classmethod
    def from_covariance_operator(
        cls,
        covariance_operator: Callable[[Array], ArrayLike],
        discretization: AbstractStrongFormDiscretization,
        /,
        *,
        rank: int,
        key: ArrayLike,
        oversampling: int = 8,
        tolerance: float = 1e-6,
        diagnostic_probes: int = 8,
    ) -> "SpatialNoiseBasis":
        r"""Randomize a matrix-free covariance with Matfree Nyström.

        ``covariance_operator`` receives and returns arrays with
        ``discretization.state_shape``. Internally the operator is transformed by
        the quadrature mass matrix before randomized factorization.
        """
        if not callable(covariance_operator):
            raise TypeError("covariance_operator must be callable.")
        if not isinstance(discretization, AbstractStrongFormDiscretization):
            raise TypeError(
                "discretization must implement AbstractStrongFormDiscretization."
            )
        count = discretization.num_points
        retained = int(rank)
        if retained <= 0 or retained > count:
            raise ValueError(f"rank must lie in [1, {count}].")
        extra = int(oversampling)
        if extra < 0:
            raise ValueError("oversampling must be non-negative.")
        sketch_size = min(count, retained + extra)
        probes_count = int(diagnostic_probes)
        if probes_count <= 0:
            raise ValueError("diagnostic_probes must be positive.")
        probes_count = min(count, probes_count)
        threshold = float(tolerance)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("tolerance must be finite and non-negative.")
        seed = _key_seed(key)

        weights_host = np.asarray(discretization.quadrature_weights, dtype=float)
        if weights_host.shape != discretization.state_shape:
            raise ValueError("discretization quadrature weights have an invalid shape.")
        if np.any(~np.isfinite(weights_host)) or np.any(weights_host <= 0.0):
            raise ValueError("Quadrature weights must be finite and positive.")
        root = jnp.sqrt(jnp.asarray(weights_host).reshape((-1,)))

        def weighted_matvec(vector):
            state = discretization.unflatten(root * vector)
            result = jnp.asarray(covariance_operator(state), dtype=float)
            if tuple(result.shape) != discretization.state_shape:
                raise ValueError(
                    "covariance_operator must preserve discretization.state_shape."
                )
            flattened = discretization.flatten(result)
            if flattened.shape != (count,):
                raise ValueError(
                    "covariance_operator must return a scalar spatial field."
                )
            return root * flattened

        preflight = np.asarray(weighted_matvec(jnp.zeros((count,), dtype=float)))
        if np.any(~np.isfinite(preflight)):
            raise ValueError("covariance_operator must return finite values.")

        sketch_key = jr.fold_in(key, 0)
        omega = jr.normal(sketch_key, (count, sketch_size), dtype=float)
        omega, _ = jnp.linalg.qr(omega, mode="reduced")
        nystrom = nystrom_eigh(
            eigenvalues_rtol=max(threshold, np.finfo(float).eps),
        )
        raw_factor, _, _ = nystrom(weighted_matvec, omega)
        eigenvalues, modes, weighted_factor = _factor_eigenpairs(
            raw_factor,
            weights_host,
            discretization.state_shape,
            rank=retained,
        )

        diagnostic_key = jr.fold_in(key, 1)
        probes = (
            2.0
            * jr.bernoulli(
                diagnostic_key,
                0.5,
                (count, probes_count),
            ).astype(float)
            - 1.0
        )
        operator_images = jax.vmap(
            weighted_matvec,
            in_axes=1,
            out_axes=1,
        )(probes)
        approximation_images = weighted_factor @ (weighted_factor.T @ probes)
        residual_images = np.asarray(operator_images - approximation_images)
        operator_images_host = np.asarray(operator_images)
        absolute_residual = float(
            np.linalg.vector_norm(residual_images) / np.sqrt(float(probes_count))
        )
        operator_norm = float(
            np.linalg.vector_norm(operator_images_host) / np.sqrt(float(probes_count))
        )
        relative_residual = (
            0.0 if operator_norm == 0.0 else absolute_residual / operator_norm
        )
        approximation = SpatialNoiseApproximation(
            method="randomized_nystrom",
            matrix_size=count,
            requested_rank=retained,
            retained_rank=retained,
            residual_kind="relative_frobenius",
            residual_estimate=relative_residual,
            absolute_residual_estimate=absolute_residual,
            tolerance=threshold,
            seed=seed,
            sketch_size=sketch_size,
        )
        seed_label = "-".join(str(value) for value in seed)
        return cls(
            modes,
            eigenvalues,
            quadrature_weights=weights_host,
            state_shape=discretization.state_shape,
            mode_ids=tuple(f"nystrom:{seed_label}:{index}" for index in range(retained)),
            field_space_id=discretization.field_spaces[0].field_space_id,
            approximation=approximation,
        )


__all__ = ["SpatialNoiseApproximation", "SpatialNoiseBasis"]
