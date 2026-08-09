#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import scipy.linalg as scipy_linalg
import scipy.sparse as scipy_sparse
import scipy.sparse.linalg as scipy_sparse_linalg
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


SpectralBasis: TypeAlias = Literal["fourier", "sine", "cosine", "legendre"]
_DENSE_GENERALIZED_EIGH_THRESHOLD = 256
_DEFAULT_CONSTRUCTION_BYTES = 512 * 1024**2


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


def _byte_limit(value: int, /, *, estimate: int, context: str) -> int:
    limit = int(value)
    if limit <= 0:
        raise ValueError("max_construction_bytes must be positive.")
    if estimate > limit:
        raise ValueError(
            f"{context} exceeds max_construction_bytes; estimated {estimate} bytes."
        )
    return limit


class SpectralDiscretization(StrictModule, NonTrainableState):
    """Fixed weighted eigenbasis analysis, synthesis, and degeneracy metadata."""

    analysis: Array
    synthesis: Array
    eigenvalues: Array
    group_ids: Array
    quadrature_weights: Array
    basis_id: str = eqx.field(static=True)
    _num_groups: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        analysis: Any,
        synthesis: Any,
        eigenvalues: Any,
        group_ids: Any,
        quadrature_weights: Any,
        basis_id: str,
        max_construction_bytes: int = _DEFAULT_CONSTRUCTION_BYTES,
    ):
        analysis_host = np.asarray(analysis, dtype=float)
        synthesis_host = np.asarray(synthesis, dtype=float)
        eigenvalues_host = np.asarray(eigenvalues, dtype=float).reshape((-1,))
        groups_host = np.asarray(group_ids, dtype=np.int32).reshape((-1,))
        quadrature_host = np.asarray(quadrature_weights, dtype=float).reshape((-1,))
        estimate = sum(
            value.nbytes
            for value in (
                analysis_host,
                synthesis_host,
                eigenvalues_host,
                groups_host,
                quadrature_host,
            )
        )
        _byte_limit(
            max_construction_bytes,
            estimate=estimate,
            context="Spectral discretization",
        )
        modes = int(eigenvalues_host.size)
        points = int(quadrature_host.size)
        if modes <= 0 or points <= 0:
            raise ValueError("Spectral discretization must have points and modes.")
        if analysis_host.shape != (modes, points):
            raise ValueError(
                f"Analysis matrix must have shape {(modes, points)}; "
                f"got {analysis_host.shape}."
            )
        if synthesis_host.shape != (points, modes):
            raise ValueError(
                f"Synthesis matrix must have shape {(points, modes)}; "
                f"got {synthesis_host.shape}."
            )
        if groups_host.shape != (modes,):
            raise ValueError("group_ids must have one entry per spectral mode.")
        if np.any(groups_host < 0):
            raise ValueError("Spectral eigenspace group IDs must be non-negative.")
        unique = tuple(int(value) for value in np.unique(groups_host))
        if unique != tuple(range(len(unique))):
            raise ValueError("Spectral eigenspace group IDs must be contiguous.")
        if np.any(~np.isfinite(analysis_host)) or np.any(~np.isfinite(synthesis_host)):
            raise ValueError("Spectral transforms must be finite.")
        if np.any(~np.isfinite(eigenvalues_host)):
            raise ValueError("Spectral eigenvalues must be finite.")
        if np.any(~np.isfinite(quadrature_host)) or np.any(quadrature_host <= 0.0):
            raise ValueError("Spectral quadrature must be finite and positive.")
        identifier = str(basis_id)
        if not identifier:
            raise ValueError("Spectral basis_id must not be empty.")
        self.analysis = jnp.asarray(analysis_host)
        self.synthesis = jnp.asarray(synthesis_host)
        self.eigenvalues = jnp.asarray(eigenvalues_host)
        self.group_ids = jnp.asarray(groups_host)
        self.quadrature_weights = jnp.asarray(quadrature_host)
        self.basis_id = identifier
        self._num_groups = len(unique)

    @property
    def num_points(self) -> int:
        return int(self.synthesis.shape[0])

    @property
    def num_modes(self) -> int:
        return int(self.synthesis.shape[1])

    @property
    def num_groups(self) -> int:
        return self._num_groups

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
        max_construction_bytes: int = _DEFAULT_CONSTRUCTION_BYTES,
    ) -> "SpectralDiscretization":
        values = np.asarray(eigenvalues, dtype=float).reshape((-1,))
        vectors = np.asarray(eigenvectors, dtype=float)
        weights = np.asarray(measure, dtype=float).reshape((-1,))
        estimate = values.nbytes + 3 * vectors.nbytes + weights.nbytes
        _byte_limit(
            max_construction_bytes,
            estimate=estimate,
            context="Eigenbasis construction",
        )
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
            max_construction_bytes=max_construction_bytes,
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
        max_construction_bytes: int = _DEFAULT_CONSTRUCTION_BYTES,
    ) -> "SpectralDiscretization":
        """Construct low modes of the generalized problem K v = λ M v."""
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
        estimate = (
            4 * count * count * np.dtype(float).itemsize
            if use_dense
            else 8 * count * modes * np.dtype(float).itemsize
        )
        _byte_limit(
            max_construction_bytes,
            estimate=estimate,
            context="Generalized eigensolve",
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
            initial = np.random.default_rng(0).standard_normal((count, modes))
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
                stiffness_sparse @ physical, dtype=float
            )
            mass_times_physical = np.asarray(mass_sparse @ physical, dtype=float)
            residual = (
                stiffness_times_physical
                - mass_times_physical * np.asarray(values)[None, :]
            )
            stiffness_scale = float(
                np.max(np.asarray(abs(stiffness_sparse).sum(axis=1)))
            )
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
                    f"maximum relative residual is {float(np.max(relative_residual))}."
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
            max_construction_bytes=max_construction_bytes,
        )


def _trapezoid_weights(nodes: Array, /) -> Array:
    if int(nodes.shape[0]) == 1:
        return jnp.ones_like(nodes)
    interior = 0.5 * (nodes[2:] - nodes[:-2])
    return jnp.concatenate(
        (
            (0.5 * (nodes[1] - nodes[0]))[None],
            interior,
            (0.5 * (nodes[-1] - nodes[-2]))[None],
        )
    )


def _normalized_nodes(
    nodes: Array,
    quadrature_weights: Array | None,
    periodic: bool,
    /,
) -> Array:
    span = nodes[-1] - nodes[0]
    if periodic and int(nodes.shape[0]) > 1:
        span = (
            span + jnp.mean(jnp.diff(nodes))
            if quadrature_weights is None
            else jnp.sum(quadrature_weights)
        )
    nodes = eqx.error_if(
        nodes,
        jnp.isclose(span, 0.0),
        "Spectral basis nodes must span a nonzero interval.",
    )
    return (nodes - nodes[0]) / span


def _basis_matrix(
    nodes: Array,
    quadrature_weights: Array | None,
    periodic: bool,
    basis: SpectralBasis,
    modes: int,
    /,
) -> Array:
    coordinate = _normalized_nodes(nodes, quadrature_weights, periodic)
    columns: list[Array] = []
    if basis == "fourier":
        columns.append(jnp.ones_like(coordinate))
        frequency = 1
        while len(columns) < modes:
            columns.append(
                jnp.sqrt(2.0) * jnp.cos(2.0 * jnp.pi * frequency * coordinate)
            )
            if len(columns) < modes:
                columns.append(
                    jnp.sqrt(2.0)
                    * jnp.sin(2.0 * jnp.pi * frequency * coordinate)
                )
            frequency += 1
    elif basis == "sine":
        columns.extend(
            jnp.sqrt(2.0) * jnp.sin(jnp.pi * (index + 1) * coordinate)
            for index in range(modes)
        )
    elif basis == "cosine":
        columns.append(jnp.ones_like(coordinate))
        columns.extend(
            jnp.sqrt(2.0) * jnp.cos(jnp.pi * index * coordinate)
            for index in range(1, modes)
        )
    elif basis == "legendre":
        z = 2.0 * coordinate - 1.0
        columns.append(jnp.ones_like(z))
        if modes > 1:
            columns.append(z)
        for degree in range(2, modes):
            columns.append(
                (
                    (2.0 * degree - 1.0) * z * columns[-1]
                    - (degree - 1.0) * columns[-2]
                )
                / float(degree)
            )
    else:
        raise ValueError("basis must be 'fourier', 'sine', 'cosine', or 'legendre'.")
    return jnp.stack(columns[:modes], axis=-1)


class BasisTransformPlan(StrictModule, NonTrainableState):
    """Reusable weighted analysis/synthesis maps for separable primitive axes."""

    analysis_matrices: tuple[Array, ...]
    synthesis_matrices: tuple[Array, ...]
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    bases: tuple[SpectralBasis, ...] = eqx.field(static=True)
    n_modes: tuple[int, ...] = eqx.field(static=True)
    fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        nodes: Sequence[ArrayLike],
        quadrature_weights: Sequence[ArrayLike | None],
        periodic: Sequence[bool],
        bases: Sequence[SpectralBasis],
        n_modes: Sequence[int],
        /,
        *,
        max_construction_bytes: int = _DEFAULT_CONSTRUCTION_BYTES,
    ):
        nodes_value = tuple(jnp.asarray(value, dtype=float).reshape((-1,)) for value in nodes)
        quadrature_value = tuple(
            None
            if value is None
            else jnp.asarray(value, dtype=float).reshape((-1,))
            for value in quadrature_weights
        )
        periodic_value = tuple(bool(value) for value in periodic)
        bases_value = tuple(bases)
        modes_value = tuple(int(mode) for mode in n_modes)
        count = len(nodes_value)
        if not count or any(
            len(values) != count
            for values in (
                quadrature_value,
                periodic_value,
                bases_value,
                modes_value,
            )
        ):
            raise ValueError("Transform plan axis metadata must align and be non-empty.")
        for axis_nodes, weights, mode in zip(
            nodes_value, quadrature_value, modes_value, strict=True
        ):
            if mode <= 0 or mode > int(axis_nodes.size):
                raise ValueError("Basis mode counts must lie within available nodes.")
            if weights is not None and weights.shape != axis_nodes.shape:
                raise ValueError("Quadrature weights must align with axis nodes.")
        estimate = sum(
            2 * int(axis.size) * mode * np.dtype(float).itemsize
            for axis, mode in zip(nodes_value, modes_value, strict=True)
        )
        _byte_limit(
            max_construction_bytes,
            estimate=estimate,
            context="Separable basis plan",
        )
        synthesis = tuple(
            _basis_matrix(axis, weights, is_periodic, basis, mode)
            for axis, weights, is_periodic, basis, mode in zip(
                nodes_value,
                quadrature_value,
                periodic_value,
                bases_value,
                modes_value,
                strict=True,
            )
        )
        analysis = []
        for axis, weights, basis_matrix in zip(
            nodes_value, quadrature_value, synthesis, strict=True
        ):
            integration_weights = (
                _trapezoid_weights(axis) if weights is None else weights
            )
            weighted_basis = integration_weights[:, None] * basis_matrix
            gram = basis_matrix.T @ weighted_basis
            regularizer = jnp.finfo(basis_matrix.dtype).eps * jnp.trace(gram)
            analysis.append(
                jnp.linalg.solve(
                    gram
                    + regularizer * jnp.eye(gram.shape[0], dtype=gram.dtype),
                    weighted_basis.T,
                )
            )
        digest = array_tree_fingerprint((tuple(analysis), synthesis))["sha256"]
        self.analysis_matrices = tuple(analysis)
        self.synthesis_matrices = synthesis
        self.sample_shape = tuple(int(axis.size) for axis in nodes_value)
        self.bases = bases_value
        self.n_modes = modes_value
        self.fingerprint = canonical_fingerprint(
            {
                "kind": "basis-transform-plan-v1",
                "sample_shape": self.sample_shape,
                "bases": bases_value,
                "n_modes": modes_value,
                "periodic": periodic_value,
                "matrices": digest,
            }
        )


__all__ = ["BasisTransformPlan", "SpectralBasis", "SpectralDiscretization"]
