#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from math import prod
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._spatial import AbstractSpatialDiscretization


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
    discretization_id: str | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"spatial-noise-basis-v1\0")
    digest.update(repr(state_shape).encode("ascii"))
    digest.update((discretization_id or "").encode("utf-8"))
    for mode_id in mode_ids:
        digest.update(mode_id.encode("utf-8"))
        digest.update(b"\0")
    for array in (modes, eigenvalues, weights):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(repr(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


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
    discretization_id: str | None = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    def __init__(
        self,
        modes: ArrayLike,
        eigenvalues: ArrayLike,
        /,
        *,
        quadrature_weights: ArrayLike,
        state_shape: Sequence[int] | None = None,
        mode_ids: Sequence[str] | None = None,
        discretization_id: str | None = None,
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
        if discretization_id is not None and not str(discretization_id):
            raise ValueError("discretization_id must be non-empty or None.")
        self.modes = modes_array
        self.eigenvalues = eigenvalue_array
        self.quadrature_weights = weights
        self.state_shape = resolved_shape
        self.mode_ids = identifiers
        self.discretization_id = (
            None if discretization_id is None else str(discretization_id)
        )
        self.basis_id = _basis_digest(
            state_shape=resolved_shape,
            modes=modes_host,
            eigenvalues=eigenvalues_host,
            weights=weights_host,
            mode_ids=identifiers,
            discretization_id=self.discretization_id,
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
        discretization_id: str | None = None,
    ) -> "SpatialNoiseBasis":
        """Validate explicit weighted-orthonormal modes and eigenvalues."""
        return cls(
            modes,
            eigenvalues,
            quadrature_weights=quadrature_weights,
            state_shape=state_shape,
            mode_ids=mode_ids,
            discretization_id=discretization_id,
        )

    @classmethod
    def from_spectrum(
        cls,
        discretization: AbstractSpatialDiscretization,
        spectrum: Callable[[Array], ArrayLike] | ArrayLike,
        /,
        *,
        rank: int,
    ) -> "SpatialNoiseBasis":
        r"""Select low Laplacian modes and evaluate a spectral covariance law.

        ``spectrum`` receives the retained non-negative eigenvalues of
        ``-laplacian`` and returns the corresponding :math:`Q` eigenvalues.
        """
        if not isinstance(discretization, AbstractSpatialDiscretization):
            raise TypeError(
                "discretization must implement AbstractSpatialDiscretization."
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
            discretization_id=discretization.discretization_id,
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
        discretization_id: str | None = None,
        psd_tolerance: float = 1e-10,
    ) -> "SpatialNoiseBasis":
        """Factor a nodal covariance matrix using its quadrature-weighted KL basis."""
        shape = tuple(int(size) for size in state_shape)
        count = int(prod(shape))
        retained = int(rank)
        if retained <= 0 or retained > count:
            raise ValueError(f"rank must lie in [1, {count}].")
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
        if float(np.min(eigenvalues)) < -float(psd_tolerance) * scale:
            raise ValueError("covariance must be positive semidefinite.")
        order = np.argsort(eigenvalues, kind="stable")[::-1][:retained]
        selected = np.maximum(eigenvalues[order], 0.0)
        modes = weighted_modes[:, order] / root[:, None]
        modes = _canonicalize_signs(modes)
        return cls(
            modes.reshape(shape + (retained,)),
            selected,
            quadrature_weights=weights_host,
            state_shape=shape,
            mode_ids=tuple(f"covariance:{index}" for index in range(retained)),
            discretization_id=discretization_id,
        )

    @classmethod
    def from_kernel_covariance(
        cls,
        kernel: Callable[[Array, Array], ArrayLike],
        discretization: AbstractSpatialDiscretization,
        /,
        *,
        rank: int,
        points: ArrayLike | None = None,
    ) -> "SpatialNoiseBasis":
        """Discretize a continuous covariance kernel, then factor the nodal matrix."""
        if not callable(kernel):
            raise TypeError("kernel must be callable.")
        if not isinstance(discretization, AbstractSpatialDiscretization):
            raise TypeError(
                "discretization must implement AbstractSpatialDiscretization."
            )
        resolved_points = points
        if resolved_points is None:
            resolved_points = getattr(discretization, "points", None)
        if resolved_points is None:
            raise ValueError(
                "points are required when the spatial discretization has no coordinates."
            )
        point_array = jnp.asarray(resolved_points, dtype=float)
        if (
            point_array.ndim != 2
            or int(point_array.shape[0]) != discretization.num_points
        ):
            raise ValueError(
                "points must have shape (discretization.num_points, coordinate_dim)."
            )
        try:
            covariance = jnp.asarray(
                kernel(point_array[:, None, :], point_array[None, :, :]),
                dtype=float,
            )
        except (TypeError, ValueError):
            covariance = jnp.asarray(())
        expected = (discretization.num_points, discretization.num_points)
        if covariance.shape != expected:
            covariance = jax.vmap(
                lambda left: jax.vmap(lambda right: kernel(left, right))(point_array)
            )(point_array)
        return cls.from_discrete_covariance(
            covariance,
            state_shape=discretization.state_shape,
            quadrature_weights=discretization.quadrature_weights,
            rank=int(rank),
            discretization_id=discretization.discretization_id,
        )


__all__ = ["SpatialNoiseBasis"]
