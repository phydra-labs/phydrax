#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import final, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    AbstractLinearOperator,
    ArraySpace,
    assemble_diagonal,
    DiagonalLinearOperator,
    FunctionLinearOperator,
    hodge_laplacian_form,
    HodgeLaplacianPart,
    IdentityLinearOperator,
    mass_form,
    MaterializationPolicy,
    transpose,
)
from ..linalg.eigen import (
    eigensolve,
    EigenSolvePolicy,
    EigenSolveResult,
    EigenSolveStatus,
    GeneralizedEigenproblem,
)
from ..typing import parse
from ._complex import AbstractDeRhamComplex, ComplexBoundary


if TYPE_CHECKING:
    from ..discretization import SpectralDecomposition


@final
class HodgeSpectrumPolicy(StrictModule):
    """Native eigen policy plus zero and eigenspace-boundary tolerances."""

    eigen_policy: EigenSolvePolicy
    eigenvalue_tolerance: float = eqx.field(static=True)
    degeneracy_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        eigen_policy: EigenSolvePolicy | None = None,
        /,
        *,
        eigenvalue_tolerance: float = 1e-9,
        degeneracy_tolerance: float = 1e-8,
    ) -> None:
        tolerances = (float(eigenvalue_tolerance), float(degeneracy_tolerance))
        if any(not np.isfinite(value) or value < 0 for value in tolerances):
            raise ValueError("Hodge spectrum tolerances must be finite and nonnegative.")
        resolved = EigenSolvePolicy() if eigen_policy is None else eigen_policy
        if not isinstance(resolved, EigenSolvePolicy):
            raise TypeError("eigen_policy must be an EigenSolvePolicy.")
        if resolved.which != "smallest-algebraic":
            raise ValueError("Hodge spectra require smallest-algebraic eigen selection.")
        self.eigen_policy = resolved
        self.eigenvalue_tolerance, self.degeneracy_tolerance = tolerances


def _hodge_eigensolve(
    complex: AbstractDeRhamComplex,
    k: int,
    /,
    *,
    boundary: ComplexBoundary,
    part: HodgeLaplacianPart,
    count: int | None,
    policy: HodgeSpectrumPolicy,
) -> tuple[EigenSolveResult, Array, AbstractLinearOperator, int]:
    hilbert = complex.hilbert_complex(boundary=boundary)
    space = hilbert.space(k)
    dimension = space.size
    requested = dimension if count is None else int(count)
    if requested < 1 or requested > dimension:
        raise ValueError("count must lie within the active spectral dimension.")
    weak = hodge_laplacian_form(hilbert, k, part=part)
    mass = mass_form(hilbert, k)
    eigen_policy = policy.eigen_policy
    native_policy = EigenSolvePolicy(
        eigen_policy.method,
        count=min(dimension, requested + 1),
        which=eigen_policy.which,
        max_steps=eigen_policy.max_steps,
        tolerance=eigen_policy.tolerance,
        resources=eigen_policy.resources,
        materialization=eigen_policy.materialization,
        initial_basis=eigen_policy.initial_basis,
        key=eigen_policy.key,
        preconditioning=eigen_policy.preconditioning,
        differentiation=eigen_policy.differentiation,
        failure=eigen_policy.failure,
    )
    result = eigensolve(GeneralizedEigenproblem(weak, mass), policy=native_policy)
    if int(result.status) != int(EigenSolveStatus.SUCCESS):
        raise ValueError(
            f"Native Hodge eigensolve failed with status {int(result.status)}."
        )
    if not isinstance(result.eigenvectors, Array):
        raise TypeError("Coordinate eigensolve must return array eigenvectors.")
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(result.eigenvalues)))
    threshold = policy.eigenvalue_tolerance * scale
    values = result.eigenvalues
    if bool(jnp.any(values < -threshold)):
        raise ValueError("A Hodge component has a materially negative eigenvalue.")
    values = jnp.where(jnp.abs(values) <= threshold, 0.0, values)
    if requested < dimension:
        gap = values[requested] - values[requested - 1]
        gap_scale = jnp.maximum(
            1.0, jnp.maximum(jnp.abs(values[requested]), jnp.abs(values[requested - 1]))
        )
        if bool(gap <= policy.degeneracy_tolerance * gap_scale):
            raise ValueError("count splits a numerically degenerate eigenspace.")
    return result, values, mass, requested


def _spectral_basis(
    complex: AbstractDeRhamComplex,
    k: int,
    boundary: ComplexBoundary,
    values: Array,
    vectors: Array,
    metric: AbstractLinearOperator,
    result: EigenSolveResult,
    /,
    *,
    source_id: str,
    exact: bool,
    materialization: MaterializationPolicy,
    next_eigenvalue: float = float("inf"),
    boundary_gap: float = float("inf"),
    canonicalized_zero_count: int = 0,
    degeneracy_tolerance: float = 1e-8,
) -> SpectralDecomposition:
    from ..discretization import LaplacianEigenbasisReport, SpectralDecomposition
    from ..discretization._cell_de_rham import AbstractCellDeRhamComplex

    rank = values.shape[0]
    active_count = metric.source.size
    if isinstance(complex, AbstractCellDeRhamComplex):
        indices = complex.active_indices(k, boundary=boundary)
    else:
        indices = jnp.arange(active_count, dtype=jnp.int32)
    full_count = (
        complex.cell_counts[k]
        if isinstance(complex, AbstractCellDeRhamComplex)
        else active_count
    )
    diagonal = jnp.real(assemble_diagonal(metric, materialization=materialization))
    total_mass = jnp.sum(diagonal)
    metric_vectors = jax.vmap(metric.mv, in_axes=1, out_axes=1)(vectors)
    physical = vectors * jnp.sqrt(total_mass)
    synthesis = (
        jnp.zeros((full_count, rank), dtype=physical.dtype).at[indices].set(physical)
    )
    active_analysis = jnp.conj(metric_vectors.T) / jnp.sqrt(total_mass)
    analysis = (
        jnp.zeros((rank, full_count), dtype=physical.dtype)
        .at[:, indices]
        .set(active_analysis)
    )
    measure = None
    analysis_metric = None
    if isinstance(metric, (DiagonalLinearOperator, IdentityLinearOperator)):
        measure = (
            jnp.zeros((full_count,), dtype=values.dtype)
            .at[indices]
            .set(diagonal / total_mass)
        )
    else:
        coordinates = ArraySpace(
            (full_count,),
            dtype=vectors.dtype,
            space_id=f"{source_id}:coordinates",
        )
        transpose_metric = transpose(metric)

        def lifted_metric(values: Array) -> Array:
            action = metric.mv(values[indices]) / total_mass
            return jnp.zeros((full_count,), dtype=values.dtype).at[indices].set(action)

        def lifted_transpose(values: Array) -> Array:
            action = transpose_metric.mv(values[indices]) / total_mass
            return jnp.zeros((full_count,), dtype=values.dtype).at[indices].set(action)

        analysis_metric = FunctionLinearOperator(
            lifted_metric,
            source=coordinates,
            target=coordinates,
            transpose_action=lifted_transpose,
            operator_id=f"{source_id}:analysis-metric:{metric.operator_id}",
        )
    gram = jnp.conj(vectors.T) @ metric_vectors
    residual = float(jnp.max(jnp.abs(gram - jnp.eye(rank)), initial=0.0))
    host_values = np.asarray(values)
    groups = np.zeros((rank,), dtype=np.int32)
    if rank > 1:
        groups[1:] = np.cumsum(
            np.diff(host_values)
            > degeneracy_tolerance
            * np.maximum(
                1.0, np.maximum(np.abs(host_values[1:]), np.abs(host_values[:-1]))
            )
        )
    report = LaplacianEigenbasisReport(
        method_id=result.provenance.method,
        source_id=source_id,
        requested_modes=None if exact else rank,
        retained_modes=rank,
        active_dimension=active_count,
        zero_mode_count=int(np.count_nonzero(host_values == 0)),
        canonicalized_zero_count=canonicalized_zero_count,
        exact=exact,
        tail_certified=exact or result.provenance.method == "dense-eigh",
        next_eigenvalue=next_eigenvalue,
        boundary_gap=boundary_gap,
        orthonormality_residual=residual,
    )
    return SpectralDecomposition(
        analysis=analysis,
        synthesis=synthesis,
        eigenvalues=values,
        group_ids=groups,
        quadrature_weights=measure,
        analysis_metric=analysis_metric,
        decomposition_id=source_id,
        active_mask=np.isin(np.arange(full_count), indices),
        spectral_dimension=float(max(1, complex.dimension)),
        index_offset=sum(complex.cell_counts[:k])
        if isinstance(complex, AbstractCellDeRhamComplex)
        else 0,
        report=report,
        eigen_solve=result,
    )


def hodge_laplacian_eigenbasis(
    complex: AbstractDeRhamComplex,
    k: int,
    /,
    *,
    boundary: ComplexBoundary = "absolute",
    part: HodgeLaplacianPart = "complete",
    count: int | None = None,
    policy: HodgeSpectrumPolicy | None = None,
) -> SpectralDecomposition:
    """Solve M L u = lambda M u with the realization's true Riesz inverse."""
    if not isinstance(complex, AbstractDeRhamComplex):
        raise TypeError("complex must be an AbstractDeRhamComplex.")
    boundary = parse(boundary, ComplexBoundary, "boundary")
    resolved = HodgeSpectrumPolicy() if policy is None else policy
    if not isinstance(resolved, HodgeSpectrumPolicy):
        raise TypeError("policy must be a HodgeSpectrumPolicy.")
    result, values, metric, requested = _hodge_eigensolve(
        complex, k, boundary=boundary, part=part, count=count, policy=resolved
    )
    if not isinstance(result.eigenvectors, Array):
        raise TypeError("Coordinate eigensolve must return array eigenvectors.")
    exact = requested == metric.source.size
    return _spectral_basis(
        complex,
        k,
        boundary,
        values[:requested],
        result.eigenvectors[:, :requested],
        metric,
        result,
        source_id=f"hodge:{complex.realization_id}:{k}:{part}:{boundary}:rank={requested}",
        exact=exact,
        materialization=resolved.eigen_policy.materialization,
        canonicalized_zero_count=int(
            jnp.count_nonzero(result.eigenvalues[:requested] != values[:requested])
        ),
        degeneracy_tolerance=resolved.degeneracy_tolerance,
        next_eigenvalue=float("inf") if exact else float(values[requested]),
        boundary_gap=float("inf")
        if exact
        else float(values[requested] - values[requested - 1]),
    )


@final
class HodgeSectorSpectra(StrictModule, NonTrainableState):
    """Harmonic, exact and coexact sectors with native solve evidence."""

    harmonic: SpectralDecomposition | None
    exact: SpectralDecomposition | None
    coexact: SpectralDecomposition | None
    solve_results: tuple[EigenSolveResult, ...]
    degree: int = eqx.field(static=True)
    boundary: ComplexBoundary = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        harmonic: SpectralDecomposition | None,
        exact: SpectralDecomposition | None,
        coexact: SpectralDecomposition | None,
        solve_results: tuple[EigenSolveResult, ...],
        degree: int,
        boundary: ComplexBoundary,
        realization_id: str,
    ) -> None:
        from ..discretization import SpectralDecomposition

        if all(value is None for value in (harmonic, exact, coexact)):
            raise ValueError("At least one Hodge sector must be nonempty.")
        if any(
            value is not None and not isinstance(value, SpectralDecomposition)
            for value in (harmonic, exact, coexact)
        ):
            raise TypeError("Hodge sectors must be SpectralDecomposition values.")
        boundary = parse(boundary, ComplexBoundary, "boundary")
        self.harmonic, self.exact, self.coexact = harmonic, exact, coexact
        self.solve_results = solve_results
        self.degree, self.boundary, self.realization_id = (
            int(degree),
            boundary,
            realization_id,
        )

    @property
    def total_rank(self) -> int:
        return sum(
            sector.mode_count
            for sector in (self.harmonic, self.exact, self.coexact)
            if sector is not None
        )


def hodge_sector_spectra(
    complex: AbstractDeRhamComplex,
    k: int,
    /,
    *,
    boundary: ComplexBoundary = "absolute",
    policy: HodgeSpectrumPolicy | None = None,
) -> HodgeSectorSpectra:
    """Resolve all active coordinates into orthogonal native Hodge sectors."""
    boundary = parse(boundary, ComplexBoundary, "boundary")
    resolved = HodgeSpectrumPolicy() if policy is None else policy
    sectors: list[SpectralDecomposition | None] = []
    results: list[EigenSolveResult] = []
    for part in ("complete", "lower", "upper"):
        if (part == "lower" and k == 0) or (part == "upper" and k == complex.dimension):
            sectors.append(None)
            continue
        result, values, metric, _ = _hodge_eigensolve(
            complex, k, boundary=boundary, part=part, count=None, policy=resolved
        )
        results.append(result)
        if not isinstance(result.eigenvectors, Array):
            raise TypeError("Coordinate eigensolve must return array eigenvectors.")
        selected = np.flatnonzero(
            np.asarray(values == 0 if part == "complete" else values > 0)
        )
        sectors.append(
            None
            if selected.size == 0
            else _spectral_basis(
                complex,
                k,
                boundary,
                values[selected],
                result.eigenvectors[:, selected],
                metric,
                result,
                source_id=f"hodge-sector:{complex.realization_id}:{k}:{part}:{boundary}",
                exact=True,
                materialization=resolved.eigen_policy.materialization,
                canonicalized_zero_count=int(
                    jnp.count_nonzero(result.eigenvalues[selected] != values[selected])
                ),
                degeneracy_tolerance=resolved.degeneracy_tolerance,
            )
        )
    harmonic, exact, coexact = sectors
    total = sum(value.mode_count for value in sectors if value is not None)
    if total != complex.hilbert_complex(boundary=boundary).space(k).size:
        raise ValueError("Hodge sectors do not span the active coordinate space.")
    return HodgeSectorSpectra(
        harmonic=harmonic,
        exact=exact,
        coexact=coexact,
        solve_results=tuple(results),
        degree=k,
        boundary=boundary,
        realization_id=complex.realization_id,
    )


__all__ = [
    "HodgeSpectrumPolicy",
    "hodge_laplacian_eigenbasis",
    "HodgeSectorSpectra",
    "hodge_sector_spectra",
]
