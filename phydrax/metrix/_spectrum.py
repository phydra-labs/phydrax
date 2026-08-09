#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from .._trainable import NonTrainableState


class LaplacianEigenbasisReport(StrictModule):
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
        if not isinstance(method_id, str) or not method_id:
            raise ValueError("method_id must be a nonempty string.")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source_id must be a nonempty string.")
        if requested_modes is not None and int(requested_modes) <= 0:
            raise ValueError("requested_modes must be positive or None.")
        if int(retained_modes) <= 0 or int(active_dimension) < int(retained_modes):
            raise ValueError("retained_modes must lie within the active dimension.")
        if not 0 <= int(zero_mode_count) <= int(retained_modes):
            raise ValueError("zero_mode_count must lie within the retained basis.")
        if not 0 <= int(canonicalized_zero_count) <= int(zero_mode_count):
            raise ValueError(
                "canonicalized_zero_count must lie within the reported zero modes."
            )
        next_value = float(next_eigenvalue)
        gap = float(boundary_gap)
        residual = float(orthonormality_residual)
        if np.isnan(next_value) or next_value < 0.0:
            raise ValueError("next_eigenvalue must be nonnegative or positive infinity.")
        if np.isnan(gap) or gap < 0.0 or not np.isfinite(residual) or residual < 0.0:
            raise ValueError("Spectrum diagnostics must be nonnegative and non-NaN.")
        certified = bool(tail_certified)
        if bool(exact) and not certified:
            raise ValueError("An exact spectrum must certify its empty omitted tail.")
        if certified and not bool(exact) and not np.isfinite(next_value):
            raise ValueError(
                "A certified truncated spectrum requires a finite next_eigenvalue."
            )
        self.method_id = method_id
        self.source_id = source_id
        self.requested_modes = None if requested_modes is None else int(requested_modes)
        self.retained_modes = int(retained_modes)
        self.active_dimension = int(active_dimension)
        self.zero_mode_count = int(zero_mode_count)
        self.canonicalized_zero_count = int(canonicalized_zero_count)
        self.exact = bool(exact)
        self.tail_certified = certified
        self.next_eigenvalue = next_value
        self.boundary_gap = gap
        self.orthonormality_residual = residual


class DiscreteLaplacianEigenbasis(StrictModule, NonTrainableState):
    """Finite self-adjoint Laplacian basis orthonormal under a probability measure."""

    eigenvalues: Array
    eigenfunctions: Array
    probability_measure: Array
    active_mask: Array
    report: LaplacianEigenbasisReport
    spectral_dimension: float = eqx.field(static=True)
    index_offset: int = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    def __init__(
        self,
        eigenvalues: ArrayLike,
        eigenfunctions: ArrayLike,
        probability_measure: ArrayLike,
        /,
        *,
        spectral_dimension: float,
        basis_id: str,
        active_mask: ArrayLike | None = None,
        index_offset: int = 0,
        report: LaplacianEigenbasisReport | None = None,
        negative_eigenvalue_tolerance: float = 1e-10,
        orthonormality_tolerance: float = 1e-8,
    ):
        values = np.asarray(eigenvalues, dtype=float)
        functions = np.asarray(eigenfunctions, dtype=float)
        measure = np.asarray(probability_measure, dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("eigenvalues must be a nonempty rank-one array.")
        if functions.ndim != 2 or functions.shape[1] != values.size:
            raise ValueError("eigenfunctions must have shape (entity, mode).")
        if measure.shape != (functions.shape[0],):
            raise ValueError("probability_measure must have one entry per entity.")
        if active_mask is None:
            active = np.ones((functions.shape[0],), dtype=bool)
        else:
            active = np.asarray(active_mask, dtype=bool)
        if active.shape != measure.shape:
            raise ValueError("active_mask must have one entry per entity.")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(functions)):
            raise ValueError("Eigenpairs must contain only finite real values.")
        if not np.all(np.isfinite(measure)):
            raise ValueError("probability_measure must contain only finite values.")
        negative_tolerance = float(negative_eigenvalue_tolerance)
        orthonormality_tolerance = float(orthonormality_tolerance)
        if negative_tolerance < 0.0 or orthonormality_tolerance <= 0.0:
            raise ValueError("Spectrum tolerances must be positive where required.")
        if np.any(values < -negative_tolerance):
            raise ValueError("eigenvalues contain a materially negative value.")
        if np.any(np.diff(values) < 0.0):
            raise ValueError("eigenvalues must be sorted nondecreasingly.")
        if np.any(measure[active] <= 0.0):
            raise ValueError("Active entities require strictly positive measure.")
        if np.any(measure[~active] != 0.0):
            raise ValueError("Inactive entities must have zero measure.")
        if not np.isclose(np.sum(measure), 1.0, rtol=0.0, atol=orthonormality_tolerance):
            raise ValueError("probability_measure must sum to one.")
        if np.any(functions[~active] != 0.0):
            raise ValueError("Inactive eigenfunction rows must be zero.")
        gram = functions.T @ (measure[:, None] * functions)
        residual = float(np.max(np.abs(gram - np.eye(values.size))))
        if residual > orthonormality_tolerance:
            raise ValueError("eigenfunctions are not orthonormal under the measure.")
        dimension = float(spectral_dimension)
        if not np.isfinite(dimension) or dimension <= 0.0:
            raise ValueError("spectral_dimension must be finite and positive.")
        if not isinstance(basis_id, str) or not basis_id:
            raise ValueError("basis_id must be a nonempty string.")
        if int(index_offset) < 0:
            raise ValueError("index_offset must be nonnegative.")
        canonical_values = np.where(np.abs(values) <= negative_tolerance, 0.0, values)
        zero_count = int(np.count_nonzero(canonical_values == 0.0))
        if report is None:
            report = LaplacianEigenbasisReport(
                method_id="provided",
                source_id=basis_id,
                requested_modes=int(values.size),
                retained_modes=int(values.size),
                active_dimension=int(np.count_nonzero(active)),
                zero_mode_count=zero_count,
                canonicalized_zero_count=int(
                    np.count_nonzero(values != canonical_values)
                ),
                exact=int(values.size) == int(np.count_nonzero(active)),
                tail_certified=int(values.size) == int(np.count_nonzero(active)),
                next_eigenvalue=float("inf"),
                boundary_gap=float("inf"),
                orthonormality_residual=residual,
            )
        if report.retained_modes != int(values.size):
            raise ValueError("report.retained_modes must match the basis rank.")
        if report.active_dimension != int(np.count_nonzero(active)):
            raise ValueError("report.active_dimension must match active_mask.")
        if report.zero_mode_count != zero_count:
            raise ValueError("report.zero_mode_count must match the canonical spectrum.")
        if not np.isclose(
            report.orthonormality_residual,
            residual,
            rtol=1e-6,
            atol=orthonormality_tolerance,
        ):
            raise ValueError(
                "report.orthonormality_residual must match the measured residual."
            )
        if np.isfinite(report.next_eigenvalue):
            expected_gap = report.next_eigenvalue - float(canonical_values[-1])
            if expected_gap < -negative_tolerance or not np.isclose(
                report.boundary_gap,
                max(0.0, expected_gap),
                rtol=1e-6,
                atol=max(negative_tolerance, orthonormality_tolerance),
            ):
                raise ValueError(
                    "Finite next-eigenvalue provenance must follow the retained spectrum."
                )
        self.eigenvalues = jnp.asarray(canonical_values)
        self.eigenfunctions = jnp.asarray(functions)
        self.probability_measure = jnp.asarray(measure)
        self.active_mask = jnp.asarray(active)
        self.report = report
        self.spectral_dimension = dimension
        self.index_offset = int(index_offset)
        self.basis_id = basis_id

    @property
    def mode_count(self) -> int:
        return int(self.eigenvalues.shape[0])

    @property
    def entity_count(self) -> int:
        return int(self.eigenfunctions.shape[0])

    @property
    def zero_mode_count(self) -> int:
        return self.report.zero_mode_count


__all__ = ["DiscreteLaplacianEigenbasis", "LaplacianEigenbasisReport"]
