#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..metrix import DiscreteLaplacianEigenbasis, LaplacianEigenbasisReport
from ._cochain import (
    CochainBoundaryKind,
    CochainBoundaryPolicy,
    CochainComplexIR,
    HarmonicSubspace,
)


CochainLaplacianComponent: TypeAlias = Literal["complete", "lower", "upper"]


def _restricted_boundary_matrix(
    complex_ir: CochainComplexIR,
    degree: int,
    policy: CochainBoundaryPolicy,
    /,
) -> sp.csr_matrix:
    boundary = complex_ir.incidences[degree - 1].scipy_matrix()
    if policy.kind == "absolute":
        return boundary
    lower_active = np.asarray(complex_ir.active_mask(degree - 1, policy), dtype=bool)
    upper_active = np.asarray(complex_ir.active_mask(degree, policy), dtype=bool)
    return boundary[lower_active][:, upper_active].tocsr()


def _assemble_symmetric_hodge_laplacian(
    complex_ir: CochainComplexIR,
    degree: int,
    component: CochainLaplacianComponent,
    boundary_policy: CochainBoundaryKind,
    /,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Assemble the metric-symmetric active-cell Hodge Laplacian."""
    if not isinstance(complex_ir, CochainComplexIR):
        raise TypeError("Hodge assembly requires a CochainComplexIR.")
    resolved_degree = int(degree)
    if resolved_degree < 0 or resolved_degree > complex_ir.max_degree:
        raise ValueError(f"degree must lie in [0, {complex_ir.max_degree}].")
    if component not in ("complete", "lower", "upper"):
        raise ValueError("component must be 'complete', 'lower', or 'upper'.")
    policy = CochainBoundaryPolicy(boundary_policy)
    active = np.asarray(complex_ir.active_mask(resolved_degree, policy), dtype=bool)
    metric = np.asarray(complex_ir.hodge_stars[resolved_degree], dtype=float)[active]
    active_count = int(np.count_nonzero(active))
    laplacian = sp.csr_matrix((active_count, active_count), dtype=float)
    inverse_sqrt_metric = sp.diags(1.0 / np.sqrt(metric))
    sqrt_metric = sp.diags(np.sqrt(metric))

    if component in ("complete", "lower") and resolved_degree > 0:
        boundary = _restricted_boundary_matrix(complex_ir, resolved_degree, policy)
        lower_active = np.asarray(
            complex_ir.active_mask(resolved_degree - 1, policy), dtype=bool
        )
        lower_metric = np.asarray(
            complex_ir.hodge_stars[resolved_degree - 1], dtype=float
        )[lower_active]
        transformed = sqrt_metric @ boundary.T @ sp.diags(1.0 / np.sqrt(lower_metric))
        laplacian = laplacian + transformed @ transformed.T

    if component in ("complete", "upper") and resolved_degree < complex_ir.max_degree:
        boundary = _restricted_boundary_matrix(complex_ir, resolved_degree + 1, policy)
        upper_active = np.asarray(
            complex_ir.active_mask(resolved_degree + 1, policy), dtype=bool
        )
        upper_metric = np.asarray(
            complex_ir.hodge_stars[resolved_degree + 1], dtype=float
        )[upper_active]
        transformed = sp.diags(np.sqrt(upper_metric)) @ boundary.T @ inverse_sqrt_metric
        laplacian = laplacian + transformed.T @ transformed

    laplacian = 0.5 * (laplacian + laplacian.T)
    return laplacian.tocsr(), active, metric


def _ordered_eigenpairs(
    laplacian: sp.csr_matrix,
    requested_modes: int,
    /,
    *,
    dense_threshold: int,
    solver_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    dimension = int(laplacian.shape[0])
    compute_full = (
        requested_modes == dimension
        or dimension <= int(dense_threshold)
        or requested_modes >= dimension - 1
    )
    if compute_full:
        values, vectors = np.linalg.eigh(laplacian.toarray())
        return values, vectors, True
    values, vectors = eigsh(
        laplacian,
        k=requested_modes + 1,
        which="SA",
        tol=float(solver_tolerance),
    )
    order = np.argsort(values, kind="stable")
    return values[order], vectors[:, order], False


def _symmetric_operator_scale(laplacian: sp.csr_matrix, /) -> float:
    absolute_laplacian = laplacian.copy()
    absolute_laplacian.data = np.abs(absolute_laplacian.data)
    row_sums = np.asarray(absolute_laplacian.sum(axis=1)).reshape(-1)
    return max(1.0, float(np.max(row_sums, initial=0.0)))


def cochain_laplacian_eigenbasis(
    complex_ir: CochainComplexIR,
    degree: int,
    /,
    *,
    num_modes: int | None,
    component: CochainLaplacianComponent = "complete",
    boundary_policy: CochainBoundaryKind = "absolute",
    spectral_dimension: float = 1.0,
    dense_threshold: int = 256,
    solver_tolerance: float = 1e-10,
    eigenvalue_tolerance: float = 1e-10,
    degeneracy_tolerance: float = 1e-8,
) -> DiscreteLaplacianEigenbasis:
    """Compute a probability-normalized cochain basis with explicit tail provenance."""
    if int(dense_threshold) <= 0:
        raise ValueError("dense_threshold must be positive.")
    if float(solver_tolerance) <= 0.0 or float(eigenvalue_tolerance) < 0.0:
        raise ValueError("Eigensolver tolerances are invalid.")
    if float(degeneracy_tolerance) < 0.0:
        raise ValueError("degeneracy_tolerance must be nonnegative.")
    laplacian, active, metric = _assemble_symmetric_hodge_laplacian(
        complex_ir,
        degree,
        component,
        boundary_policy,
    )
    active_dimension = int(laplacian.shape[0])
    if active_dimension == 0:
        raise ValueError("The selected boundary policy has no active cells.")
    requested = active_dimension if num_modes is None else int(num_modes)
    if requested <= 0 or requested > active_dimension:
        raise ValueError("num_modes must lie within the active spectral dimension.")
    values, vectors, used_dense_solver = _ordered_eigenpairs(
        laplacian,
        requested,
        dense_threshold=int(dense_threshold),
        solver_tolerance=float(solver_tolerance),
    )
    operator_scale = _symmetric_operator_scale(laplacian)
    zero_threshold = float(eigenvalue_tolerance) * operator_scale
    materially_negative = values < -zero_threshold
    if np.any(materially_negative):
        raise ValueError("The assembled self-adjoint Laplacian has a negative mode.")
    canonical_values = np.where(np.abs(values) <= zero_threshold, 0.0, values)
    if requested < active_dimension:
        if values.size <= requested:
            raise ValueError("The truncated solve did not return a lookahead mode.")
        boundary_scale = max(
            1.0,
            abs(float(canonical_values[requested - 1])),
            abs(float(canonical_values[requested])),
        )
        boundary_gap = float(
            canonical_values[requested] - canonical_values[requested - 1]
        )
        if boundary_gap <= float(degeneracy_tolerance) * boundary_scale:
            raise ValueError("num_modes splits a numerically degenerate eigenspace.")
        next_eigenvalue = float(canonical_values[requested])
    else:
        boundary_gap = float("inf")
        next_eigenvalue = float("inf")
    retained_values = canonical_values[:requested]
    retained_vectors = vectors[:, :requested]
    total_metric_mass = float(np.sum(metric))
    physical_active = (
        retained_vectors / np.sqrt(metric)[:, None] * np.sqrt(total_metric_mass)
    )
    functions = np.zeros((complex_ir.cell_counts[int(degree)], requested), dtype=float)
    functions[np.flatnonzero(active), :] = physical_active
    probability_measure = np.zeros((functions.shape[0],), dtype=float)
    probability_measure[active] = metric / total_metric_mass
    gram = physical_active.T @ (probability_measure[active, None] * physical_active)
    residual = float(np.max(np.abs(gram - np.eye(requested))))
    exact = requested == active_dimension
    source_id = (
        f"cochain:{complex_ir.fingerprint}:degree={int(degree)}:"
        f"component={component}:boundary={boundary_policy}"
    )
    method_id = "dense-eigh" if used_dense_solver else "sparse-eigsh"
    report = LaplacianEigenbasisReport(
        method_id=method_id,
        source_id=source_id,
        requested_modes=num_modes,
        retained_modes=requested,
        active_dimension=active_dimension,
        zero_mode_count=int(np.count_nonzero(retained_values == 0.0)),
        canonicalized_zero_count=int(
            np.count_nonzero(values[:requested] != retained_values)
        ),
        exact=exact,
        tail_certified=used_dense_solver or exact,
        next_eigenvalue=next_eigenvalue,
        boundary_gap=boundary_gap,
        orthonormality_residual=residual,
    )
    basis_id = (
        f"{source_id}:rank={requested}:exact={int(exact)}:"
        f"tail-certified={int(used_dense_solver or exact)}"
    )
    return DiscreteLaplacianEigenbasis(
        retained_values,
        functions,
        probability_measure,
        spectral_dimension=float(spectral_dimension),
        basis_id=basis_id,
        active_mask=active,
        index_offset=complex_ir.cell_offsets[int(degree)],
        report=report,
        negative_eigenvalue_tolerance=zero_threshold,
        orthonormality_tolerance=max(1e-8, 10.0 * float(solver_tolerance)),
    )


def compute_harmonic_subspace(
    complex_ir: CochainComplexIR,
    /,
    *,
    boundary_policy: CochainBoundaryKind = "absolute",
    max_modes: int = 8,
    tolerance: float = 1e-9,
    dense_threshold: int = 256,
) -> HarmonicSubspace:
    """Precompute metric harmonic bases without target-dependent information."""
    if not isinstance(complex_ir, CochainComplexIR):
        raise TypeError("compute_harmonic_subspace requires a CochainComplexIR.")
    policy = CochainBoundaryPolicy(boundary_policy)
    if int(max_modes) < 0:
        raise ValueError("max_modes must be nonnegative.")
    if float(tolerance) <= 0.0:
        raise ValueError("tolerance must be positive.")
    bases = []
    eigenvalues = []
    ranks = []
    for degree, count in enumerate(complex_ir.cell_counts):
        laplacian, active, metric = _assemble_symmetric_hodge_laplacian(
            complex_ir,
            degree,
            "complete",
            policy.kind,
        )
        active_count = int(laplacian.shape[0])
        if active_count == 0:
            values = np.zeros((0,), dtype=float)
            vectors = np.zeros((0, 0), dtype=float)
        else:
            requested = min(active_count, int(max_modes) + 1)
            values, vectors, _ = _ordered_eigenpairs(
                laplacian,
                requested,
                dense_threshold=int(dense_threshold),
                solver_tolerance=float(tolerance),
            )
        threshold = float(tolerance) * _symmetric_operator_scale(laplacian)
        rank = int(np.count_nonzero(np.abs(values) <= threshold))
        if rank > int(max_modes) or (
            values.size and rank == values.size and values.size < active_count
        ):
            raise ValueError(
                "Harmonic nullspace exceeds max_modes or is not separated from nonzero modes."
            )
        physical_active = vectors[:, :rank] / np.sqrt(metric)[:, None]
        physical = np.zeros((count, int(max_modes)), dtype=float)
        if rank:
            physical[np.flatnonzero(active), :rank] = physical_active
            gram = physical_active.T @ (metric[:, None] * physical_active)
            if not np.allclose(gram, np.eye(rank), rtol=1e-7, atol=1e-9):
                raise ValueError("Computed harmonic basis is not metric orthonormal.")
        stored_values = np.full((int(max_modes),), np.inf, dtype=float)
        stored_values[: min(values.size, int(max_modes))] = values[: int(max_modes)]
        bases.append(jnp.asarray(physical))
        eigenvalues.append(jnp.asarray(stored_values))
        ranks.append(rank)
    return HarmonicSubspace(
        bases,
        eigenvalues,
        ranks,
        max_modes=int(max_modes),
        boundary_policy=policy.kind,
        complex_fingerprint=complex_ir.fingerprint,
    )


class CochainHodgeSectorSpectra(StrictModule, NonTrainableState):
    """Complete harmonic, exact, and coexact spectral sectors for one degree."""

    harmonic: DiscreteLaplacianEigenbasis | None
    exact: DiscreteLaplacianEigenbasis | None
    coexact: DiscreteLaplacianEigenbasis | None
    degree: int = eqx.field(static=True)
    boundary_policy: CochainBoundaryKind = eqx.field(static=True)
    complex_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        harmonic: DiscreteLaplacianEigenbasis | None,
        exact: DiscreteLaplacianEigenbasis | None,
        coexact: DiscreteLaplacianEigenbasis | None,
        degree: int,
        boundary_policy: CochainBoundaryKind,
        complex_fingerprint: str,
    ):
        for sector in (harmonic, exact, coexact):
            if sector is not None and not isinstance(sector, DiscreteLaplacianEigenbasis):
                raise TypeError("Hodge sectors must be Laplacian eigenbases or None.")
        if all(sector is None for sector in (harmonic, exact, coexact)):
            raise ValueError("At least one Hodge sector must be nonempty.")
        if boundary_policy not in ("absolute", "relative"):
            raise ValueError("Unknown Hodge-sector boundary policy.")
        self.harmonic = harmonic
        self.exact = exact
        self.coexact = coexact
        self.degree = int(degree)
        self.boundary_policy = boundary_policy
        self.complex_fingerprint = str(complex_fingerprint)

    @property
    def total_rank(self) -> int:
        return sum(
            sector.mode_count
            for sector in (self.harmonic, self.exact, self.coexact)
            if sector is not None
        )


def _hodge_sector_basis(
    complex_ir: CochainComplexIR,
    degree: int,
    active: np.ndarray,
    metric: np.ndarray,
    values: np.ndarray,
    vectors: np.ndarray,
    sector: Literal["harmonic", "exact", "coexact"],
    boundary_policy: CochainBoundaryKind,
    /,
) -> DiscreteLaplacianEigenbasis | None:
    rank = int(values.size)
    if rank == 0:
        return None
    total_mass = float(np.sum(metric))
    physical_active = vectors / np.sqrt(metric)[:, None] * np.sqrt(total_mass)
    functions = np.zeros((complex_ir.cell_counts[int(degree)], rank), dtype=float)
    functions[np.flatnonzero(active), :] = physical_active
    measure = np.zeros((functions.shape[0],), dtype=float)
    measure[active] = metric / total_mass
    gram = physical_active.T @ (measure[active, None] * physical_active)
    residual = float(np.max(np.abs(gram - np.eye(rank))))
    source_id = (
        f"cochain:{complex_ir.fingerprint}:degree={int(degree)}:"
        f"sector={sector}:boundary={boundary_policy}"
    )
    report = LaplacianEigenbasisReport(
        method_id="dense-hodge-sector",
        source_id=source_id,
        requested_modes=None,
        retained_modes=rank,
        active_dimension=int(np.count_nonzero(active)),
        zero_mode_count=rank if sector == "harmonic" else 0,
        canonicalized_zero_count=0,
        exact=True,
        tail_certified=True,
        next_eigenvalue=float("inf"),
        boundary_gap=float("inf"),
        orthonormality_residual=residual,
    )
    return DiscreteLaplacianEigenbasis(
        values,
        functions,
        measure,
        spectral_dimension=float(max(1, complex_ir.max_degree)),
        basis_id=f"{source_id}:rank={rank}",
        active_mask=active,
        index_offset=complex_ir.cell_offsets[int(degree)],
        report=report,
        orthonormality_tolerance=1e-7,
    )


def cochain_hodge_sector_spectra(
    complex_ir: CochainComplexIR,
    degree: int,
    /,
    *,
    boundary_policy: CochainBoundaryKind = "absolute",
    eigenvalue_tolerance: float = 1e-9,
) -> CochainHodgeSectorSpectra:
    """Resolve complete Hodge sectors using positive lower and upper spectra."""
    if float(eigenvalue_tolerance) <= 0.0:
        raise ValueError("eigenvalue_tolerance must be positive.")
    complete, active, metric = _assemble_symmetric_hodge_laplacian(
        complex_ir,
        degree,
        "complete",
        boundary_policy,
    )
    lower, lower_active, lower_metric = _assemble_symmetric_hodge_laplacian(
        complex_ir,
        degree,
        "lower",
        boundary_policy,
    )
    upper, upper_active, upper_metric = _assemble_symmetric_hodge_laplacian(
        complex_ir,
        degree,
        "upper",
        boundary_policy,
    )
    if (
        not np.array_equal(active, lower_active)
        or not np.array_equal(active, upper_active)
        or not np.array_equal(metric, lower_metric)
        or not np.array_equal(metric, upper_metric)
    ):
        raise ValueError("Hodge component assemblies disagree on their active metric.")
    if int(complete.shape[0]) == 0:
        raise ValueError("The selected boundary policy has no active cells.")
    complete_values, complete_vectors = np.linalg.eigh(complete.toarray())
    lower_values, lower_vectors = np.linalg.eigh(lower.toarray())
    upper_values, upper_vectors = np.linalg.eigh(upper.toarray())
    scale = max(
        1.0,
        float(np.max(np.abs(complete_values), initial=0.0)),
        float(np.max(np.abs(lower_values), initial=0.0)),
        float(np.max(np.abs(upper_values), initial=0.0)),
    )
    threshold = float(eigenvalue_tolerance) * scale
    for values in (complete_values, lower_values, upper_values):
        if np.any(values < -threshold):
            raise ValueError("A Hodge component has a materially negative eigenvalue.")
    harmonic_indices = np.flatnonzero(np.abs(complete_values) <= threshold)
    exact_indices = np.flatnonzero(lower_values > threshold)
    coexact_indices = np.flatnonzero(upper_values > threshold)
    if harmonic_indices.size + exact_indices.size + coexact_indices.size != int(
        complete.shape[0]
    ):
        raise ValueError("Hodge sector ranks do not span the active cochain space.")
    harmonic = _hodge_sector_basis(
        complex_ir,
        degree,
        active,
        metric,
        np.zeros((harmonic_indices.size,), dtype=float),
        complete_vectors[:, harmonic_indices],
        "harmonic",
        boundary_policy,
    )
    exact = _hodge_sector_basis(
        complex_ir,
        degree,
        active,
        metric,
        lower_values[exact_indices],
        lower_vectors[:, exact_indices],
        "exact",
        boundary_policy,
    )
    coexact = _hodge_sector_basis(
        complex_ir,
        degree,
        active,
        metric,
        upper_values[coexact_indices],
        upper_vectors[:, coexact_indices],
        "coexact",
        boundary_policy,
    )
    return CochainHodgeSectorSpectra(
        harmonic=harmonic,
        exact=exact,
        coexact=coexact,
        degree=int(degree),
        boundary_policy=boundary_policy,
        complex_fingerprint=complex_ir.fingerprint,
    )


__all__ = [
    "CochainLaplacianComponent",
    "CochainHodgeSectorSpectra",
    "cochain_laplacian_eigenbasis",
    "compute_harmonic_subspace",
    "cochain_hodge_sector_spectra",
]
