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
import scipy.fft as scipy_fft
import scipy.linalg as scipy_linalg
import scipy.sparse as scipy_sparse
import scipy.sparse.linalg as scipy_sparse_linalg
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


ModalTransformKind: TypeAlias = Literal["fourier", "sine", "cosine", "legendre"]
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


class ModalTransform(StrictModule, NonTrainableState):
    """Weighted analysis/synthesis transform independent of any operator."""

    analysis: Array
    synthesis: Array
    quadrature_weights: Array
    active_mask: Array
    mode_ids: tuple[str, ...] = eqx.field(static=True)
    orthonormality_residual: float = eqx.field(static=True)
    transform_id: str = eqx.field(static=True)

    def __init__(
        self,
        analysis: ArrayLike,
        synthesis: ArrayLike,
        quadrature_weights: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
        mode_ids: Sequence[str] | None = None,
        transform_id: str | None = None,
    ):
        analysis_host = np.asarray(analysis)
        synthesis_host = np.asarray(synthesis)
        weights_host = np.asarray(quadrature_weights, dtype=float).reshape((-1,))
        if analysis_host.ndim != 2 or synthesis_host.ndim != 2:
            raise ValueError("Modal transforms require rank-2 analysis and synthesis.")
        mode_count, point_count = analysis_host.shape
        if synthesis_host.shape != (point_count, mode_count):
            raise ValueError("Analysis and synthesis shapes must be exact transposes.")
        if weights_host.shape != (point_count,):
            raise ValueError("quadrature_weights must contain one value per point.")
        active = (
            np.ones((point_count,), dtype=bool)
            if active_mask is None
            else np.asarray(active_mask, dtype=bool)
        )
        if active.shape != (point_count,):
            raise ValueError("active_mask must contain one value per point.")
        if (
            np.any(~np.isfinite(analysis_host))
            or np.any(~np.isfinite(synthesis_host))
            or np.any(~np.isfinite(weights_host[active]))
            or np.any(weights_host[active] <= 0.0)
        ):
            raise ValueError("Active modal transforms and weights must be finite.")
        if np.any(weights_host[~active] != 0.0):
            raise ValueError("Inactive modal weights must be zero.")
        if np.any(synthesis_host[~active] != 0.0) or np.any(
            analysis_host[:, ~active] != 0.0
        ):
            raise ValueError("Inactive modal transform entries must be zero.")
        modes = (
            tuple(f"mode:{index}" for index in range(mode_count))
            if mode_ids is None
            else tuple(str(value) for value in mode_ids)
        )
        if (
            len(modes) != mode_count
            or any(not value for value in modes)
            or len(set(modes)) != mode_count
        ):
            raise ValueError("mode_ids must contain one unique non-empty ID per mode.")
        gram = synthesis_host.conj().T @ (weights_host[:, None] * synthesis_host)
        residual = float(np.max(np.abs(gram - np.eye(mode_count))))
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "modal-transform",
                    "analysis": array_tree_fingerprint(analysis_host),
                    "synthesis": array_tree_fingerprint(synthesis_host),
                    "weights": array_tree_fingerprint(weights_host),
                    "active": array_tree_fingerprint(active),
                    "mode_ids": list(modes),
                }
            )
            if transform_id is None
            else str(transform_id)
        )
        if not identifier:
            raise ValueError("transform_id must be non-empty.")
        self.analysis = jnp.asarray(analysis_host)
        self.synthesis = jnp.asarray(synthesis_host)
        self.quadrature_weights = jnp.asarray(weights_host)
        self.active_mask = jnp.asarray(active)
        self.mode_ids = modes
        self.orthonormality_residual = residual
        self.transform_id = identifier

    @property
    def num_points(self) -> int:
        return int(self.synthesis.shape[0])

    @property
    def num_modes(self) -> int:
        return int(self.synthesis.shape[1])

    def analyze(self, values: ArrayLike, /) -> Array:
        array = jnp.asarray(values)
        if array.shape[0] != self.num_points:
            raise ValueError("Modal analysis leading axis must match point count.")
        return jnp.tensordot(self.analysis, array, axes=((1,), (0,)))

    def synthesize(self, coefficients: ArrayLike, /) -> Array:
        array = jnp.asarray(coefficients)
        if array.shape[0] != self.num_modes:
            raise ValueError("Modal synthesis leading axis must match mode count.")
        return jnp.tensordot(self.synthesis, array, axes=((1,), (0,)))


class LaplacianEigenbasisReport(StrictModule, NonTrainableState):
    """Immutable provenance and numerical diagnostics for one eigendecomposition."""

    method_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    requested_modes: int | None = eqx.field(static=True)
    retained_modes: int = eqx.field(static=True)
    active_dimension: int = eqx.field(static=True)
    zero_mode_count: int = eqx.field(static=True)
    canonicalized_zero_count: int = eqx.field(static=True)
    exact: bool = eqx.field(static=True)
    tail_certified: bool = eqx.field(static=True)
    next_eigenvalue: float = eqx.field(static=True)
    boundary_gap: float = eqx.field(static=True)
    orthonormality_residual: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        method_id: str,
        source_id: str,
        requested_modes: int | None,
        retained_modes: int,
        active_dimension: int,
        zero_mode_count: int,
        canonicalized_zero_count: int,
        exact: bool,
        tail_certified: bool,
        next_eigenvalue: float,
        boundary_gap: float,
        orthonormality_residual: float,
    ):
        method = str(method_id)
        source = str(source_id)
        requested = None if requested_modes is None else int(requested_modes)
        retained = int(retained_modes)
        active = int(active_dimension)
        zeros = int(zero_mode_count)
        canonicalized = int(canonicalized_zero_count)
        next_value = float(next_eigenvalue)
        gap = float(boundary_gap)
        residual = float(orthonormality_residual)
        if not method or not source:
            raise ValueError("method_id and source_id must be non-empty.")
        if requested is not None and requested <= 0:
            raise ValueError("requested_modes must be positive or None.")
        if retained <= 0 or active < retained:
            raise ValueError("retained_modes must lie within the active dimension.")
        if not 0 <= zeros <= retained or not 0 <= canonicalized <= zeros:
            raise ValueError("Zero-mode counts must lie within the retained basis.")
        if np.isnan(next_value) or next_value < 0.0:
            raise ValueError("next_eigenvalue must be non-negative or positive infinity.")
        if np.isnan(gap) or gap < 0.0 or not np.isfinite(residual) or residual < 0.0:
            raise ValueError("Spectrum diagnostics must be non-negative and non-NaN.")
        certified = bool(tail_certified)
        exact_ = bool(exact)
        if exact_ and not certified:
            raise ValueError("An exact spectrum must certify its omitted tail.")
        if certified and not exact_ and not np.isfinite(next_value):
            raise ValueError(
                "A certified truncated spectrum requires a finite next_eigenvalue."
            )
        self.method_id = method
        self.source_id = source
        self.requested_modes = requested
        self.retained_modes = retained
        self.active_dimension = active
        self.zero_mode_count = zeros
        self.canonicalized_zero_count = canonicalized
        self.exact = exact_
        self.tail_certified = certified
        self.next_eigenvalue = next_value
        self.boundary_gap = gap
        self.orthonormality_residual = residual


SpectrumClassification: TypeAlias = Literal[
    "discrete",
    "pseudospectral",
    "eigendecomposition",
    "custom",
]


class OperatorSpectrum(StrictModule, NonTrainableState):
    """Modal values of one operator in one exact transform."""

    modal_values: Array
    group_ids: Array
    nullspace_mask: Array
    transform_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    classification: SpectrumClassification = eqx.field(static=True)
    spectral_dimension: float | None = eqx.field(static=True)
    index_offset: int = eqx.field(static=True)
    report: LaplacianEigenbasisReport | None
    spectrum_id: str = eqx.field(static=True)
    _num_groups: int = eqx.field(static=True)

    def __init__(
        self,
        transform: ModalTransform,
        operator_id: str,
        modal_values: ArrayLike,
        /,
        *,
        group_ids: ArrayLike | None = None,
        nullspace_mask: ArrayLike | None = None,
        classification: SpectrumClassification = "custom",
        spectral_dimension: float | None = None,
        index_offset: int = 0,
        report: LaplacianEigenbasisReport | None = None,
        spectrum_id: str | None = None,
        zero_tolerance: float = 1e-10,
    ):
        if not isinstance(transform, ModalTransform):
            raise TypeError("transform must be a ModalTransform.")
        operator = str(operator_id)
        if not operator:
            raise ValueError("operator_id must be non-empty.")
        values = np.asarray(modal_values).reshape((-1,))
        if values.shape != (transform.num_modes,) or np.any(~np.isfinite(values)):
            raise ValueError("modal_values must be finite with one value per mode.")
        groups = (
            np.arange(transform.num_modes, dtype=np.int32)
            if group_ids is None
            else np.asarray(group_ids, dtype=np.int32).reshape((-1,))
        )
        if groups.shape != values.shape or np.any(groups < 0):
            raise ValueError("group_ids must contain one non-negative ID per mode.")
        unique = tuple(int(value) for value in np.unique(groups))
        if unique != tuple(range(len(unique))):
            raise ValueError("Operator spectrum group IDs must be contiguous.")
        tolerance = float(zero_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("zero_tolerance must be finite and non-negative.")
        nullspace = (
            np.abs(values) <= tolerance
            if nullspace_mask is None
            else np.asarray(nullspace_mask, dtype=bool)
        )
        if nullspace.shape != values.shape:
            raise ValueError("nullspace_mask must contain one value per mode.")
        if classification not in (
            "discrete",
            "pseudospectral",
            "eigendecomposition",
            "custom",
        ):
            raise ValueError("Unknown operator spectrum classification.")
        dimension = None if spectral_dimension is None else float(spectral_dimension)
        if dimension is not None and (not np.isfinite(dimension) or dimension <= 0.0):
            raise ValueError("spectral_dimension must be finite and positive.")
        offset = int(index_offset)
        if offset < 0:
            raise ValueError("index_offset must be non-negative.")
        if report is not None:
            if not isinstance(report, LaplacianEigenbasisReport):
                raise TypeError("report must be LaplacianEigenbasisReport or None.")
            if report.retained_modes != transform.num_modes:
                raise ValueError("report.retained_modes must match transform rank.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "operator-spectrum",
                    "transform": transform.transform_id,
                    "operator": operator,
                    "modal_values": array_tree_fingerprint(values),
                    "groups": array_tree_fingerprint(groups),
                    "nullspace": array_tree_fingerprint(nullspace),
                    "classification": classification,
                    "spectral_dimension": dimension,
                    "index_offset": offset,
                    "report": None if report is None else repr(report),
                }
            )
            if spectrum_id is None
            else str(spectrum_id)
        )
        if not identifier:
            raise ValueError("spectrum_id must be non-empty.")
        self.modal_values = jnp.asarray(values)
        self.group_ids = jnp.asarray(groups)
        self.nullspace_mask = jnp.asarray(nullspace)
        self.transform_id = transform.transform_id
        self.operator_id = operator
        self.classification = classification
        self.spectral_dimension = dimension
        self.index_offset = offset
        self.report = report
        self.spectrum_id = identifier
        self._num_groups = len(unique)

    @property
    def num_modes(self) -> int:
        return int(self.modal_values.shape[0])

    @property
    def num_groups(self) -> int:
        return self._num_groups


def trigonometric_modal_transform(
    kind: Literal["dct", "dst"],
    transform_type: Literal[1, 2, 3, 4],
    count: int,
    /,
) -> ModalTransform:
    """Prepare one orthonormal DCT/DST transform without coupling an operator."""
    size = int(count)
    type_ = int(transform_type)
    if kind not in ("dct", "dst") or type_ not in (1, 2, 3, 4) or size < 2:
        raise ValueError("Trigonometric transform kind/type/count is invalid.")
    identity = np.eye(size)
    analysis = (
        scipy_fft.dct(identity, type=type_, norm="ortho", axis=0)
        if kind == "dct"
        else scipy_fft.dst(identity, type=type_, norm="ortho", axis=0)
    )
    synthesis = analysis.T
    return ModalTransform(
        analysis,
        synthesis,
        np.ones((size,)),
        mode_ids=tuple(f"{kind}{type_}:{index}" for index in range(size)),
        transform_id=canonical_fingerprint(
            {
                "kind": "trigonometric-modal-transform",
                "family": kind,
                "type": type_,
                "count": size,
            }
        ),
    )




class TensorModalTransform(StrictModule, NonTrainableState):
    """Separable composition of independent one-dimensional modal transforms."""

    transforms: tuple[ModalTransform, ...]
    physical_shape: tuple[int, ...] = eqx.field(static=True)
    modal_shape: tuple[int, ...] = eqx.field(static=True)
    transform_id: str = eqx.field(static=True)

    def __init__(self, transforms: Sequence[ModalTransform], /):
        values = tuple(transforms)
        if not values or not all(isinstance(value, ModalTransform) for value in values):
            raise TypeError("transforms must contain one or more ModalTransform values.")
        self.transforms = values
        self.physical_shape = tuple(value.num_points for value in values)
        self.modal_shape = tuple(value.num_modes for value in values)
        self.transform_id = canonical_fingerprint(
            {
                "kind": "tensor-modal-transform",
                "transforms": [value.transform_id for value in values],
            }
        )

    def analyze(self, values: ArrayLike, /) -> Array:
        result = jnp.asarray(values)
        if result.shape[: len(self.physical_shape)] != self.physical_shape:
            raise ValueError("Tensor modal input must begin with physical_shape.")
        for axis, transform in enumerate(self.transforms):
            moved = jnp.moveaxis(result, axis, 0)
            moved = transform.analyze(moved)
            result = jnp.moveaxis(moved, 0, axis)
        return result

    def synthesize(self, coefficients: ArrayLike, /) -> Array:
        result = jnp.asarray(coefficients)
        if result.shape[: len(self.modal_shape)] != self.modal_shape:
            raise ValueError("Tensor modal coefficients must begin with modal_shape.")
        for axis in reversed(range(len(self.transforms))):
            moved = jnp.moveaxis(result, axis, 0)
            moved = self.transforms[axis].synthesize(moved)
            result = jnp.moveaxis(moved, 0, axis)
        return result


class SpectralDecomposition(StrictModule, NonTrainableState):
    """Convenience pairing of an independent modal transform and operator spectrum."""

    transform: ModalTransform
    spectrum: OperatorSpectrum
    analysis: Array
    synthesis: Array
    eigenvalues: Array
    group_ids: Array
    quadrature_weights: Array
    active_mask: Array
    report: LaplacianEigenbasisReport | None
    spectral_dimension: float | None = eqx.field(static=True)
    index_offset: int = eqx.field(static=True)
    decomposition_id: str = eqx.field(static=True)
    mode_ids: tuple[str, ...] = eqx.field(static=True)
    _num_groups: int = eqx.field(static=True)

    def __init__(
        self,
        *eigenbasis: Any,
        analysis: Any | None = None,
        synthesis: Any | None = None,
        eigenvalues: Any | None = None,
        group_ids: Any | None = None,
        quadrature_weights: Any | None = None,
        decomposition_id: str,
        active_mask: Any | None = None,
        spectral_dimension: float | None = None,
        index_offset: int = 0,
        report: LaplacianEigenbasisReport | None = None,
        mode_ids: Sequence[str] | None = None,
        negative_eigenvalue_tolerance: float = 1e-10,
        orthonormality_tolerance: float = 1e-8,
        max_construction_bytes: int = _DEFAULT_CONSTRUCTION_BYTES,
    ):
        legacy = bool(eigenbasis)
        if legacy:
            if len(eigenbasis) != 3:
                raise TypeError(
                    "Positional SpectralDecomposition construction requires "
                    "(eigenvalues, eigenfunctions, probability_measure)."
                )
            if any(
                value is not None
                for value in (
                    analysis,
                    synthesis,
                    eigenvalues,
                    group_ids,
                    quadrature_weights,
                )
            ):
                raise TypeError(
                    "Positional eigenbasis construction cannot mix transform keywords."
                )
            eigenvalues_host = np.asarray(eigenbasis[0], dtype=float).reshape((-1,))
            synthesis_host = np.asarray(eigenbasis[1], dtype=float)
            quadrature_host = np.asarray(eigenbasis[2], dtype=float).reshape((-1,))
            if synthesis_host.ndim != 2:
                raise ValueError("eigenfunctions must have shape (entity, mode).")
            groups_host = _eigenspace_groups(eigenvalues_host, 1e-8)
            analysis_host = synthesis_host.T * quadrature_host[None, :]
        else:
            if any(
                value is None
                for value in (
                    analysis,
                    synthesis,
                    eigenvalues,
                    group_ids,
                    quadrature_weights,
                )
            ):
                raise TypeError(
                    "Transform construction requires analysis, synthesis, eigenvalues, "
                    "group_ids, and quadrature_weights."
                )
            analysis_host = np.asarray(analysis, dtype=float)
            synthesis_host = np.asarray(synthesis, dtype=float)
            eigenvalues_host = np.asarray(eigenvalues, dtype=float).reshape((-1,))
            groups_host = np.asarray(group_ids, dtype=np.int32).reshape((-1,))
            quadrature_host = np.asarray(quadrature_weights, dtype=float).reshape((-1,))
        modes = int(eigenvalues_host.size)
        points = int(quadrature_host.size)
        active = (
            np.ones((points,), dtype=bool)
            if active_mask is None
            else np.asarray(active_mask, dtype=bool)
        )
        if active.shape != (points,):
            raise ValueError("active_mask must have one entry per spectral point.")
        estimate = sum(
            value.nbytes
            for value in (
                analysis_host,
                synthesis_host,
                eigenvalues_host,
                groups_host,
                quadrature_host,
                active,
            )
        )
        _byte_limit(
            max_construction_bytes,
            estimate=estimate,
            context="Spectral basis",
        )
        if modes <= 0 or points <= 0:
            raise ValueError("Spectral bases must have points and modes.")
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
        if groups_host.shape != (modes,) or np.any(groups_host < 0):
            raise ValueError("group_ids must contain one non-negative ID per mode.")
        unique = tuple(int(value) for value in np.unique(groups_host))
        if unique != tuple(range(len(unique))):
            raise ValueError("Spectral eigenspace group IDs must be contiguous.")
        if (
            np.any(~np.isfinite(analysis_host))
            or np.any(~np.isfinite(synthesis_host))
            or np.any(~np.isfinite(eigenvalues_host))
            or np.any(~np.isfinite(quadrature_host))
        ):
            raise ValueError(
                "Spectral transforms, eigenvalues, and weights must be finite."
            )
        negative_tolerance = float(negative_eigenvalue_tolerance)
        orthogonality_tolerance = float(orthonormality_tolerance)
        if negative_tolerance < 0.0 or orthogonality_tolerance <= 0.0:
            raise ValueError("Spectrum tolerances must be positive where required.")
        if np.any(eigenvalues_host < -negative_tolerance):
            raise ValueError("eigenvalues contain a materially negative value.")
        if np.any(np.diff(eigenvalues_host) < 0.0):
            raise ValueError("eigenvalues must be sorted nondecreasingly.")
        if np.any(quadrature_host[active] <= 0.0):
            raise ValueError("Active spectral points require positive measure.")
        if np.any(quadrature_host[~active] != 0.0):
            raise ValueError("Inactive spectral points must have zero measure.")
        if np.any(synthesis_host[~active] != 0.0):
            raise ValueError("Inactive eigenfunction rows must be zero.")
        if np.any(analysis_host[:, ~active] != 0.0):
            raise ValueError("Inactive spectral analysis entries must be zero.")
        canonical_values = np.where(
            np.abs(eigenvalues_host) <= negative_tolerance,
            0.0,
            eigenvalues_host,
        )
        gram = synthesis_host.T @ (quadrature_host[:, None] * synthesis_host)
        residual = float(np.max(np.abs(gram - np.eye(modes))))
        if legacy:
            if not np.isclose(
                np.sum(quadrature_host),
                1.0,
                rtol=0.0,
                atol=orthogonality_tolerance,
            ):
                raise ValueError("probability_measure must sum to one.")
            if residual > orthogonality_tolerance:
                raise ValueError("eigenfunctions are not orthonormal under the measure.")
        dimension = None if spectral_dimension is None else float(spectral_dimension)
        if legacy and dimension is None:
            raise ValueError(
                "Positional Laplacian-basis construction requires spectral_dimension."
            )
        if dimension is not None and (not np.isfinite(dimension) or dimension <= 0.0):
            raise ValueError("spectral_dimension must be finite and positive.")
        offset = int(index_offset)
        if offset < 0:
            raise ValueError("index_offset must be non-negative.")
        identifier = str(decomposition_id)
        if not identifier:
            raise ValueError("Spectral decomposition_id must not be empty.")
        zero_count = int(np.count_nonzero(canonical_values == 0.0))
        report_ = report
        if legacy and report_ is None:
            report_ = LaplacianEigenbasisReport(
                method_id="provided",
                source_id=identifier,
                requested_modes=modes,
                retained_modes=modes,
                active_dimension=int(np.count_nonzero(active)),
                zero_mode_count=zero_count,
                canonicalized_zero_count=int(
                    np.count_nonzero(eigenvalues_host != canonical_values)
                ),
                exact=modes == int(np.count_nonzero(active)),
                tail_certified=modes == int(np.count_nonzero(active)),
                next_eigenvalue=float("inf"),
                boundary_gap=float("inf"),
                orthonormality_residual=residual,
            )
        if report_ is not None:
            if not isinstance(report_, LaplacianEigenbasisReport):
                raise TypeError("report must be LaplacianEigenbasisReport or None.")
            if report_.retained_modes != modes:
                raise ValueError("report.retained_modes must match the basis rank.")
            if report_.active_dimension != int(np.count_nonzero(active)):
                raise ValueError("report.active_dimension must match active_mask.")
            if report_.zero_mode_count != zero_count:
                raise ValueError("report.zero_mode_count must match the spectrum.")
            if not np.isclose(
                report_.orthonormality_residual,
                residual,
                rtol=1e-6,
                atol=orthogonality_tolerance,
            ):
                raise ValueError(
                    "report.orthonormality_residual must match the measured residual."
                )
            if np.isfinite(report_.next_eigenvalue):
                expected_gap = report_.next_eigenvalue - float(canonical_values[-1])
                if expected_gap < -negative_tolerance or not np.isclose(
                    report_.boundary_gap,
                    max(0.0, expected_gap),
                    rtol=1e-6,
                    atol=max(negative_tolerance, orthogonality_tolerance),
                ):
                    raise ValueError(
                        "Finite next-eigenvalue provenance must follow the spectrum."
                    )
        modes_ = (
            tuple(f"mode:{offset + index}" for index in range(modes))
            if mode_ids is None
            else tuple(str(value) for value in mode_ids)
        )
        if (
            len(modes_) != modes
            or any(not value for value in modes_)
            or len(set(modes_)) != modes
        ):
            raise ValueError("mode_ids must contain one unique non-empty ID per mode.")
        transform = ModalTransform(
            analysis_host,
            synthesis_host,
            quadrature_host,
            active_mask=active,
            mode_ids=modes_,
        )
        spectrum = OperatorSpectrum(
            transform,
            "laplacian",
            canonical_values,
            group_ids=groups_host,
            classification="eigendecomposition",
            spectral_dimension=dimension,
            index_offset=offset,
            report=report_,
        )
        self.transform = transform
        self.spectrum = spectrum
        self.analysis = transform.analysis
        self.synthesis = transform.synthesis
        self.eigenvalues = spectrum.modal_values
        self.group_ids = spectrum.group_ids
        self.quadrature_weights = transform.quadrature_weights
        self.active_mask = transform.active_mask
        self.report = spectrum.report
        self.spectral_dimension = spectrum.spectral_dimension
        self.index_offset = spectrum.index_offset
        self.decomposition_id = identifier
        self.mode_ids = transform.mode_ids
        self._num_groups = spectrum.num_groups

    @property
    def num_points(self) -> int:
        return int(self.synthesis.shape[0])

    @property
    def num_modes(self) -> int:
        return int(self.synthesis.shape[1])

    @property
    def num_groups(self) -> int:
        return self._num_groups

    @property
    def transform_id(self) -> str:
        return self.transform.transform_id

    @property
    def spectrum_id(self) -> str:
        return self.spectrum.spectrum_id

    def diagonal_representation(
        self,
        operator: Any,
        /,
        *,
        spectrum: OperatorSpectrum | None = None,
    ):
        from ..linalg import DenseLinearTransform, TransformDiagonalRepresentation

        spectrum_ = self.spectrum if spectrum is None else spectrum
        if not isinstance(spectrum_, OperatorSpectrum):
            raise TypeError("spectrum must be an OperatorSpectrum.")
        if spectrum_.transform_id != self.transform_id:
            raise ValueError("Operator spectrum must belong to this modal transform.")
        transform = DenseLinearTransform(
            self.transform.analysis,
            self.transform.synthesis,
            transform_id=self.transform_id,
        )
        return TransformDiagonalRepresentation.from_transform(
            operator,
            spectrum_.modal_values,
            transform,
            representation_id=canonical_fingerprint(
                {
                    "kind": "modal-operator-representation",
                    "transform": self.transform_id,
                    "spectrum": spectrum_.spectrum_id,
                    "operator": operator.operator_id,
                }
            ),
        )

    @property
    def eigenfunctions(self) -> Array:
        return self.synthesis

    @property
    def probability_measure(self) -> Array:
        return self.quadrature_weights

    @property
    def mode_count(self) -> int:
        return self.num_modes

    @property
    def entity_count(self) -> int:
        return self.num_points

    @property
    def zero_mode_count(self) -> int:
        return (
            self.report.zero_mode_count
            if self.report is not None
            else int(np.count_nonzero(np.asarray(self.eigenvalues) == 0.0))
        )

    @classmethod
    def from_eigenpairs(
        cls,
        eigenvalues: Any,
        eigenvectors: Any,
        measure: Any,
        /,
        *,
        group_tolerance: float = 1e-7,
        decomposition_id: str | None = None,
        max_construction_bytes: int = _DEFAULT_CONSTRUCTION_BYTES,
    ) -> "SpectralDecomposition":
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
        if decomposition_id is None:
            digest = hashlib.sha256()
            digest.update(np.ascontiguousarray(values).tobytes())
            digest.update(np.ascontiguousarray(vectors).tobytes())
            digest.update(np.ascontiguousarray(weights).tobytes())
            identifier = digest.hexdigest()
        else:
            identifier = str(decomposition_id)
        return cls(
            analysis=vectors.T * weights[None, :],
            synthesis=vectors,
            eigenvalues=values,
            group_ids=groups,
            quadrature_weights=weights,
            decomposition_id=identifier,
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
        decomposition_id: str | None = None,
        max_construction_bytes: int = _DEFAULT_CONSTRUCTION_BYTES,
    ) -> "SpectralDecomposition":
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
        identifier = decomposition_id
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
            decomposition_id=identifier,
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
    basis: ModalTransformKind,
    modes: int,
    /,
) -> Array:
    coordinate = _normalized_nodes(nodes, quadrature_weights, periodic)
    columns: list[Array] = []
    if basis == "fourier":
        columns.append(jnp.ones_like(coordinate))
        frequency = 1
        while len(columns) < modes:
            columns.append(jnp.sqrt(2.0) * jnp.cos(2.0 * jnp.pi * frequency * coordinate))
            if len(columns) < modes:
                columns.append(
                    jnp.sqrt(2.0) * jnp.sin(2.0 * jnp.pi * frequency * coordinate)
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
                ((2.0 * degree - 1.0) * z * columns[-1] - (degree - 1.0) * columns[-2])
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
    bases: tuple[ModalTransformKind, ...] = eqx.field(static=True)
    n_modes: tuple[int, ...] = eqx.field(static=True)
    fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        nodes: Sequence[ArrayLike],
        quadrature_weights: Sequence[ArrayLike | None],
        periodic: Sequence[bool],
        bases: Sequence[ModalTransformKind],
        n_modes: Sequence[int],
        /,
        *,
        max_construction_bytes: int = _DEFAULT_CONSTRUCTION_BYTES,
    ):
        nodes_value = tuple(
            jnp.asarray(value, dtype=float).reshape((-1,)) for value in nodes
        )
        quadrature_value = tuple(
            None if value is None else jnp.asarray(value, dtype=float).reshape((-1,))
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
            integration_weights = _trapezoid_weights(axis) if weights is None else weights
            weighted_basis = integration_weights[:, None] * basis_matrix
            gram = basis_matrix.T @ weighted_basis
            regularizer = jnp.finfo(basis_matrix.dtype).eps * jnp.trace(gram)
            analysis.append(
                jnp.linalg.solve(
                    gram + regularizer * jnp.eye(gram.shape[0], dtype=gram.dtype),
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


__all__ = [
    "BasisTransformPlan",
    "LaplacianEigenbasisReport",
    "ModalTransform",
    "ModalTransformKind",
    "OperatorSpectrum",
    "SpectralDecomposition",
    "TensorModalTransform",
    "trigonometric_modal_transform",
    "SpectrumClassification",
]
