#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Callable
from math import prod
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import opt_einsum as oe
import scipy.linalg as scipy_linalg
import scipy.sparse as scipy_sparse
import scipy.sparse.linalg as scipy_sparse_linalg
from jaxtyping import Array, Key

from ...._doc import DOC_KEY0
from ...._trainable import NonTrainableState
from ..._utils import _get_size
from ..core._base import _AbstractOperatorModel
from ..core._keys import EvalKey, fold_in_eval_key
from ..core._operator import FunctionSamples, OperatorBatch
from ..layers._linear import Linear


def _eigenspace_groups(eigenvalues: np.ndarray, tolerance: float, /) -> np.ndarray:
    groups = np.zeros(eigenvalues.size, dtype=np.int32)
    group = 0
    for index in range(1, eigenvalues.size):
        scale = max(
            1.0, abs(float(eigenvalues[index - 1])), abs(float(eigenvalues[index]))
        )
        if abs(float(eigenvalues[index] - eigenvalues[index - 1])) > tolerance * scale:
            group += 1
        groups[index] = group
    return groups


def _canonicalize_eigenvector_signs(vectors: np.ndarray, /) -> np.ndarray:
    result = np.array(vectors, dtype=float, copy=True)
    for mode in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, mode])))
        if result[pivot, mode] < 0.0:
            result[:, mode] *= -1.0
    return result


_DENSE_GENERALIZED_EIGH_THRESHOLD = 256


def _finite_symmetric_matrix(value: Any, /, *, name: str):
    if scipy_sparse.issparse(value):
        matrix = value.astype(float).tocsr(copy=True)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name} must be a square matrix.")
        matrix.sum_duplicates()
        matrix.sort_indices()
        if np.any(~np.isfinite(matrix.data)):
            raise ValueError(f"{name} must be finite.")
        scale = max(
            1.0,
            float(np.max(np.abs(matrix.data))) if matrix.data.size else 0.0,
        )
        difference = matrix - matrix.T
        error = float(np.max(np.abs(difference.data))) if difference.data.size else 0.0
        if error > 1e-10 * scale:
            raise ValueError(f"{name} must be symmetric.")
        return ((matrix + matrix.T) * 0.5).tocsr()

    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    if np.any(~np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite.")
    scale = max(1.0, float(np.max(np.abs(matrix))) if matrix.size else 0.0)
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-10 * scale):
        raise ValueError(f"{name} must be symmetric.")
    return 0.5 * (matrix + matrix.T)


def _update_matrix_digest(digest: Any, matrix: Any, /) -> None:
    if scipy_sparse.issparse(matrix):
        csr = matrix.tocsr(copy=True)
        csr.sum_duplicates()
        csr.sort_indices()
        digest.update(b"csr")
        digest.update(np.asarray(csr.shape, dtype=np.int64).tobytes())
        digest.update(np.asarray(csr.indptr, dtype=np.int64).tobytes())
        digest.update(np.asarray(csr.indices, dtype=np.int64).tobytes())
        digest.update(np.asarray(csr.data, dtype=float).tobytes())
        return
    digest.update(b"dense")
    digest.update(np.ascontiguousarray(np.asarray(matrix, dtype=float)).tobytes())


class SpectralDiscretization(eqx.Module, NonTrainableState):
    """Fixed analysis/synthesis maps and eigenspace metadata for one manifold."""

    analysis: Array
    synthesis: Array
    eigenvalues: Array
    group_ids: Array
    quadrature_weights: Array
    basis_id: str

    def __init__(
        self,
        *,
        analysis: Any,
        synthesis: Any,
        eigenvalues: Any,
        group_ids: Any,
        quadrature_weights: Any,
        basis_id: str,
    ):
        analysis_ = jnp.asarray(analysis, dtype=float)
        synthesis_ = jnp.asarray(synthesis, dtype=float)
        eigenvalues_ = jnp.asarray(eigenvalues, dtype=float).reshape((-1,))
        groups = jnp.asarray(group_ids, dtype=jnp.int32).reshape((-1,))
        quadrature = jnp.asarray(quadrature_weights, dtype=float).reshape((-1,))
        modes = int(eigenvalues_.size)
        points = int(quadrature.size)
        if modes <= 0 or points <= 0:
            raise ValueError("Spectral discretization must have points and modes.")
        if analysis_.shape != (modes, points):
            raise ValueError(
                f"Analysis matrix must have shape {(modes, points)}; "
                f"got {analysis_.shape}."
            )
        if synthesis_.shape != (points, modes):
            raise ValueError(
                f"Synthesis matrix must have shape {(points, modes)}; "
                f"got {synthesis_.shape}."
            )
        if groups.shape != (modes,):
            raise ValueError("group_ids must have one entry per spectral mode.")
        if bool(jnp.any(groups < 0)):
            raise ValueError("Spectral eigenspace group IDs must be non-negative.")
        unique = tuple(int(value) for value in np.unique(np.asarray(groups)))
        if unique != tuple(range(len(unique))):
            raise ValueError("Spectral eigenspace group IDs must be contiguous.")
        if bool(jnp.any(~jnp.isfinite(analysis_))) or bool(
            jnp.any(~jnp.isfinite(synthesis_))
        ):
            raise ValueError("Spectral transforms must be finite.")
        if bool(jnp.any(~jnp.isfinite(eigenvalues_))):
            raise ValueError("Spectral eigenvalues must be finite.")
        if bool(jnp.any(~jnp.isfinite(quadrature))) or bool(jnp.any(quadrature <= 0.0)):
            raise ValueError("Spectral quadrature must be finite and positive.")
        if not str(basis_id):
            raise ValueError("Spectral basis_id must not be empty.")
        self.analysis = analysis_
        self.synthesis = synthesis_
        self.eigenvalues = eigenvalues_
        self.group_ids = groups
        self.quadrature_weights = quadrature
        self.basis_id = str(basis_id)

    @property
    def num_points(self) -> int:
        return int(self.synthesis.shape[0])

    @property
    def num_modes(self) -> int:
        return int(self.synthesis.shape[1])

    @property
    def num_groups(self) -> int:
        return int(jnp.max(self.group_ids)) + 1

    @classmethod
    def from_eigenpairs(
        cls,
        eigenvalues: Any,
        eigenvectors: Any,
        measure: Any,
        /,
        *,
        group_tolerance: float = 1e-7,
        basis_id: str | None = None,
    ) -> "SpectralDiscretization":
        values = np.asarray(eigenvalues, dtype=float).reshape((-1,))
        vectors = np.asarray(eigenvectors, dtype=float)
        weights = np.asarray(measure, dtype=float).reshape((-1,))
        if vectors.shape != (weights.size, values.size):
            raise ValueError(
                "Eigenvectors must have shape (points, modes) aligned with measure."
            )
        if np.any(weights <= 0.0) or np.any(~np.isfinite(weights)):
            raise ValueError("Eigenpair measure must be finite and positive.")
        order = np.argsort(values, kind="stable")
        values = values[order]
        vectors = _canonicalize_eigenvector_signs(vectors[:, order])
        norms = np.sqrt(np.sum(weights[:, None] * vectors**2, axis=0))
        if np.any(norms <= 0.0) or np.any(~np.isfinite(norms)):
            raise ValueError("Eigenvectors must have positive weighted norm.")
        vectors = vectors / norms[None, :]
        groups = _eigenspace_groups(values, float(group_tolerance))
        if basis_id is None:
            digest = hashlib.sha256()
            digest.update(np.ascontiguousarray(values).tobytes())
            digest.update(np.ascontiguousarray(vectors).tobytes())
            digest.update(np.ascontiguousarray(weights).tobytes())
            identifier = digest.hexdigest()
        else:
            identifier = str(basis_id)
        return cls(
            analysis=vectors.T * weights[None, :],
            synthesis=vectors,
            eigenvalues=values,
            group_ids=groups,
            quadrature_weights=weights,
            basis_id=identifier,
        )

    @classmethod
    def from_stiffness(
        cls,
        stiffness: Any,
        mass: Any,
        /,
        *,
        n_modes: int,
        group_tolerance: float = 1e-7,
        basis_id: str | None = None,
    ) -> "SpectralDiscretization":
        r"""Construct low modes of a positive-semidefinite stiffness operator.

        Solves the generalized eigenproblem $K v = \lambda M v$, where $K$
        represents the positive-semidefinite operator $-\Delta$ and $M$ is a
        positive mass matrix. Sparse inputs use a sparse partial eigensolve.
        """
        stiffness_matrix = _finite_symmetric_matrix(stiffness, name="Stiffness")
        count = int(stiffness_matrix.shape[0])
        if count == 0:
            raise ValueError("Stiffness must not be empty.")

        diagonal_mass = False
        if scipy_sparse.issparse(mass):
            mass_matrix = _finite_symmetric_matrix(mass, name="Mass")
            if mass_matrix.shape != (count, count):
                raise ValueError("Mass matrix shape must match stiffness.")
            measure = np.asarray(mass_matrix.diagonal(), dtype=float)
        else:
            mass_array = np.asarray(mass, dtype=float)
            if mass_array.ndim == 1:
                if mass_array.shape != (count,):
                    raise ValueError("Diagonal mass must have one entry per point.")
                if np.any(~np.isfinite(mass_array)):
                    raise ValueError("Mass must be finite.")
                measure = np.array(mass_array, dtype=float, copy=True)
                mass_matrix = scipy_sparse.diags(measure, format="csr")
                diagonal_mass = True
            elif mass_array.shape == (count, count):
                mass_matrix = _finite_symmetric_matrix(mass_array, name="Mass")
                measure = np.asarray(np.diag(mass_matrix), dtype=float)
            else:
                raise ValueError("Mass must be diagonal entries or a square matrix.")

        if np.any(~np.isfinite(measure)) or np.any(measure <= 0.0):
            raise ValueError("Mass diagonal must be finite and strictly positive.")

        modes = int(n_modes)
        if modes <= 0 or modes > count:
            raise ValueError("n_modes must lie between one and the matrix size.")

        if not diagonal_mass:
            if scipy_sparse.issparse(mass_matrix):
                if count == 1:
                    smallest_mass = float(mass_matrix[0, 0])
                else:
                    smallest_mass = float(
                        scipy_sparse_linalg.eigsh(
                            mass_matrix,
                            k=1,
                            which="SA",
                            return_eigenvectors=False,
                        )[0]
                    )
                if not np.isfinite(smallest_mass) or smallest_mass <= 0.0:
                    raise ValueError("Mass matrix must be positive definite.")
            else:
                try:
                    scipy_linalg.cholesky(
                        mass_matrix,
                        lower=True,
                        check_finite=False,
                    )
                except np.linalg.LinAlgError as exc:
                    raise ValueError("Mass matrix must be positive definite.") from exc

        use_dense = (
            count <= _DENSE_GENERALIZED_EIGH_THRESHOLD
            or 5 * modes >= count
            or (
                not scipy_sparse.issparse(stiffness_matrix)
                and not scipy_sparse.issparse(mass_matrix)
            )
        )
        if use_dense:
            stiffness_dense = (
                stiffness_matrix.toarray()
                if scipy_sparse.issparse(stiffness_matrix)
                else stiffness_matrix
            )
            mass_dense = (
                mass_matrix.toarray()
                if scipy_sparse.issparse(mass_matrix)
                else mass_matrix
            )
            try:
                values, physical = scipy_linalg.eigh(
                    stiffness_dense,
                    mass_dense,
                    subset_by_index=(0, modes - 1),
                    check_finite=False,
                )
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "Generalized stiffness eigendecomposition failed; "
                    "mass must be positive definite."
                ) from exc
        else:
            stiffness_sparse = (
                stiffness_matrix.tocsr()
                if scipy_sparse.issparse(stiffness_matrix)
                else scipy_sparse.csr_matrix(stiffness_matrix)
            )
            mass_sparse = (
                mass_matrix.tocsr()
                if scipy_sparse.issparse(mass_matrix)
                else scipy_sparse.csr_matrix(mass_matrix)
            )
            rng = np.random.default_rng(0)
            initial = rng.standard_normal((count, modes))
            initial[:, 0] = 1.0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                values, physical = scipy_sparse_linalg.lobpcg(
                    stiffness_sparse,
                    initial,
                    B=mass_sparse,
                    largest=False,
                    tol=1e-9,
                    maxiter=500,
                )

            stiffness_times_physical = np.asarray(
                stiffness_sparse @ physical,
                dtype=float,
            )
            mass_times_physical = np.asarray(mass_sparse @ physical, dtype=float)
            residual = (
                stiffness_times_physical
                - mass_times_physical * np.asarray(values)[None, :]
            )
            stiffness_scale = float(np.max(np.asarray(abs(stiffness_sparse).sum(axis=1))))
            mass_scale = float(np.max(np.asarray(abs(mass_sparse).sum(axis=1))))
            vector_norm = np.linalg.norm(physical, axis=0)
            denominator = (
                stiffness_scale + np.abs(np.asarray(values)) * mass_scale
            ) * vector_norm
            relative_residual = np.linalg.norm(residual, axis=0) / np.maximum(
                denominator,
                np.finfo(float).tiny,
            )
            if np.any(~np.isfinite(relative_residual)) or np.any(
                relative_residual > 1e-7
            ):
                raise RuntimeError(
                    "Sparse stiffness eigendecomposition failed to converge; "
                    f"maximum relative residual is "
                    f"{float(np.max(relative_residual))}."
                )

        order = np.argsort(values, kind="stable")
        values = np.asarray(values[order], dtype=float)
        physical = np.asarray(physical[:, order], dtype=float)
        spectral_scale = max(
            1.0,
            float(np.max(np.abs(values))) if values.size else 0.0,
        )
        negative_tolerance = 1e-8 * spectral_scale
        if np.any(values < -negative_tolerance):
            raise ValueError(
                "Stiffness must be positive semidefinite; "
                f"smallest generalized eigenvalue is {float(values[0])}."
            )
        values = np.maximum(values, 0.0)

        mass_times_physical = np.asarray(mass_matrix @ physical, dtype=float)
        norms = np.sqrt(np.sum(physical * mass_times_physical, axis=0))
        if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
            raise ValueError("Stiffness eigenvectors must have positive mass norm.")
        physical = physical / norms[None, :]
        physical = _canonicalize_eigenvector_signs(physical)
        mass_times_physical = np.asarray(mass_matrix @ physical, dtype=float)
        groups = _eigenspace_groups(values, float(group_tolerance))

        identifier = basis_id
        if identifier is None:
            digest = hashlib.sha256()
            _update_matrix_digest(digest, stiffness_matrix)
            _update_matrix_digest(digest, mass_matrix)
            digest.update(str(modes).encode("utf-8"))
            identifier = digest.hexdigest()
        return cls(
            analysis=mass_times_physical.T,
            synthesis=physical,
            eigenvalues=values,
            group_ids=groups,
            quadrature_weights=measure,
            basis_id=identifier,
        )

    @classmethod
    def from_triangle_mesh(
        cls,
        vertices: Any,
        faces: Any,
        /,
        *,
        n_modes: int,
        group_tolerance: float = 1e-7,
        basis_id: str | None = None,
    ) -> "SpectralDiscretization":
        from ....graph._mesh import (
            mesh_cotangent_weights,
            mesh_lumped_vertex_areas,
        )

        points = np.asarray(vertices, dtype=float)
        sender, receiver, edge_weight = mesh_cotangent_weights(vertices, faces)
        sender_ = np.asarray(sender, dtype=np.int32)
        receiver_ = np.asarray(receiver, dtype=np.int32)
        weights = np.asarray(edge_weight, dtype=float)
        rows = np.concatenate((sender_, sender_))
        columns = np.concatenate((sender_, receiver_))
        data = np.concatenate((weights, -weights))
        stiffness = scipy_sparse.coo_matrix(
            (data, (rows, columns)),
            shape=(points.shape[0], points.shape[0]),
            dtype=float,
        ).tocsr()
        stiffness.sum_duplicates()
        mass = np.asarray(mesh_lumped_vertex_areas(vertices, faces), dtype=float)
        return cls.from_stiffness(
            stiffness,
            mass,
            n_modes=n_modes,
            group_tolerance=group_tolerance,
            basis_id=basis_id,
        )


class ManifoldSpectralConv(eqx.Module):
    """Gauge-safe spectral convolution with one channel map per eigenspace."""

    weight: Array
    source: SpectralDiscretization
    target: SpectralDiscretization
    in_channels: int
    out_channels: int

    def __init__(
        self,
        source: SpectralDiscretization,
        /,
        *,
        in_channels: int,
        out_channels: int,
        target: SpectralDiscretization | None = None,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        target_ = source if target is None else target
        if source.num_modes != target_.num_modes:
            raise ValueError("Source and target spectral mode counts must match.")
        if not bool(jnp.array_equal(source.group_ids, target_.group_ids)):
            raise ValueError("Source and target eigenspace groups must match.")
        if source.basis_id != target_.basis_id:
            raise ValueError(
                "Source and target spectral plans require one aligned basis_id."
            )
        self.source = source
        self.target = target_
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("Spectral convolution channels must be positive.")
        scale = 1.0 / jnp.sqrt(float(self.in_channels))
        self.weight = scale * jr.normal(
            key,
            (source.num_groups, self.out_channels, self.in_channels),
        )

    def __call__(self, values: Array, /) -> Array:
        array = jnp.asarray(values)
        if array.shape[-2:] != (self.source.num_points, self.in_channels):
            raise ValueError(
                "Manifold spectral values must end in source points/channels "
                f"{(self.source.num_points, self.in_channels)}; got {array.shape}."
            )
        coefficients = oe.contract("mp,...pc->...mc", self.source.analysis, array)
        mode_weight = self.weight[self.source.group_ids]
        transformed = oe.contract("moc,...mc->...mo", mode_weight, coefficients)
        return oe.contract("pm,...mo->...po", self.target.synthesis, transformed)


class ManifoldSpectralOperator(_AbstractOperatorModel):
    """Intrinsic Laplace-eigenbasis neural operator on a fixed/aligned manifold."""

    lift: Linear
    spectral: tuple[ManifoldSpectralConv, ...]
    pointwise: tuple[Linear, ...]
    projection: Linear
    source_plan: SpectralDiscretization
    target_plan: SpectralDiscretization
    source_key: str | None
    activation: Callable[[Array], Array]
    residual: bool
    cross_discretization: bool
    in_channels: int
    out_channels: int
    width: int
    depth: int
    in_size: int | Literal["scalar"]
    out_size: int | Literal["scalar"]

    def __init__(
        self,
        source_plan: SpectralDiscretization,
        /,
        *,
        in_channels: int | Literal["scalar"] = "scalar",
        out_channels: int | Literal["scalar"] = "scalar",
        width: int = 64,
        depth: int = 4,
        target_plan: SpectralDiscretization | None = None,
        source_key: str | None = None,
        activation: Callable[[Array], Array] = jax.nn.gelu,
        residual: bool = True,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.source_plan = source_plan
        self.target_plan = source_plan if target_plan is None else target_plan
        self.cross_discretization = (
            target_plan is not None and target_plan is not source_plan
        )
        self.source_key = source_key
        self.activation = activation
        self.residual = bool(residual)
        self.in_channels = _get_size(in_channels)
        self.out_channels = _get_size(out_channels)
        self.width = int(width)
        self.depth = int(depth)
        self.in_size = in_channels
        self.out_size = out_channels
        if min(self.in_channels, self.out_channels, self.width, self.depth) <= 0:
            raise ValueError("Channels, width, and depth must be positive.")
        keys = jr.split(key, 2 * self.depth + 2)
        self.lift = Linear(
            in_size=self.in_channels,
            out_size=self.width,
            activation=None,
            rwf=False,
            key=keys[0],
        )
        self.spectral = tuple(
            ManifoldSpectralConv(
                self.source_plan if index == 0 else self.target_plan,
                in_channels=self.width,
                out_channels=self.width,
                target=self.target_plan,
                key=keys[1 + index],
            )
            for index in range(self.depth)
        )
        self.pointwise = tuple(
            Linear(
                in_size=self.width,
                out_size=self.width,
                activation=None,
                rwf=False,
                key=keys[1 + self.depth + index],
            )
            for index in range(self.depth)
        )
        self.projection = Linear(
            in_size=self.width,
            out_size=self.out_channels,
            activation=None,
            rwf=False,
            key=keys[-1],
        )

    def _source(self, batch: OperatorBatch, /) -> FunctionSamples:
        if self.source_key is not None:
            return batch.input(self.source_key)
        if len(batch.inputs) != 1:
            raise ValueError(
                "ManifoldSpectralOperator requires source_key for multiple inputs."
            )
        return next(iter(batch.inputs.values()))

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        source = self._source(batch)
        if source.values is None:
            raise ValueError("Manifold spectral source values cannot be None.")
        if prod(source.sample_shape) != self.source_plan.num_points:
            raise ValueError("Source sample count does not match the spectral plan.")
        if prod(batch.require_single_query().sample_shape) != self.target_plan.num_points:
            raise ValueError(
                "Query sample count does not match the target spectral plan."
            )
        values = jnp.asarray(source.values)
        sample_ndim = len(source.sample_shape)
        trailing = values.shape[len(batch.case_shape) + sample_ndim :]
        if not trailing:
            if self.in_channels != 1:
                raise ValueError("Scalar source values require one input channel.")
            values = values[..., None]
        elif tuple(int(size) for size in trailing) != (self.in_channels,):
            raise ValueError("Manifold source channel shape is incompatible.")
        values = values.reshape(
            batch.case_shape + (self.source_plan.num_points, self.in_channels)
        )
        source_mask = source.mask_array(case_shape=batch.case_shape).reshape(
            batch.case_shape + (self.source_plan.num_points, 1)
        )
        hidden = self.lift(values * source_mask, key=fold_in_eval_key(key, 0))
        for index, (spectral, pointwise) in enumerate(
            zip(self.spectral, self.pointwise, strict=True)
        ):
            spectral_update = spectral(hidden)
            if index == 0 and self.cross_discretization:
                hidden = self.activation(spectral_update)
                continue
            update = spectral_update + pointwise(
                hidden,
                key=fold_in_eval_key(key, 2 * index + 1),
            )
            hidden = self.activation(hidden + update if self.residual else update)
        output = self.projection(
            hidden,
            key=fold_in_eval_key(key, 2 * self.depth + 1),
        )
        query_mask = (
            batch.require_single_query()
            .mask_array(case_shape=batch.case_shape)
            .reshape(batch.case_shape + (self.target_plan.num_points, 1))
        )
        output = output * query_mask
        output = output.reshape(
            batch.case_shape
            + batch.require_single_query().sample_shape
            + (self.out_channels,)
        )
        return output[..., 0] if self.out_size == "scalar" else output

    def __call__(
        self,
        x: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        if not isinstance(x, OperatorBatch):
            raise TypeError("ManifoldSpectralOperator requires an OperatorBatch.")
        return self.__call_operator_batch__(x, key=key)


__all__ = [
    "ManifoldSpectralConv",
    "ManifoldSpectralOperator",
    "SpectralDiscretization",
]
