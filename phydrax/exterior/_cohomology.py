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
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    harmonic_subspace,
    HarmonicSubspace,
    HarmonicSubspacePolicy,
    hodge_laplacian,
    KernelCertificate,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSolveStatus,
    LinearSubspace,
    LinearSystem,
    mass_form,
    prepare,
    PreparedLinearSolve,
    solve,
)
from ..linalg.eigen import (
    DenseEigh,
    Eigenproblem,
    eigensolve,
    EigenSolvePolicy,
    EigenSolveResult,
    EigenSolveStatus,
)
from ..linalg.svd import svd, SVDProblem, SVDSolvePolicy, SVDSolveResult, SVDSolveStatus
from ..typing import parse
from ._complex import AbstractDeRhamComplex, ComplexBoundary


if TYPE_CHECKING:
    from ..discretization._cell_de_rham import AbstractCellDeRhamComplex
    from ..topology import (
        CellComplexPair,
        CellSubcomplex,
        RationalClassBasis,
        TopologyResourcePolicy,
    )


def _analysis_complex(
    complex: AbstractCellDeRhamComplex, boundary: ComplexBoundary, /
) -> CellSubcomplex | CellComplexPair:
    from ..topology import CellComplexPair, CellSubcomplex

    ambient = CellSubcomplex.full(complex.topology)
    if boundary == "absolute":
        return ambient
    masks = tuple(
        np.asarray(selected) & np.asarray(boundary_mask)
        for selected, boundary_mask in zip(
            ambient.masks, complex.boundary_masks, strict=True
        )
    )
    return CellComplexPair(ambient, CellSubcomplex(complex.topology, masks))


@final
class HodgeCohomologyReport(StrictModule, NonTrainableState):
    """Exact Betti dimension and independently measured numerical kernel evidence."""

    kernel_residuals: Array
    orthonormality_residual: Array
    next_eigenvalue: Array
    ranks_match: Array
    complete: Array
    degree: int = eqx.field(static=True)
    exact_dimension: int = eqx.field(static=True)
    harmonic_rank: int = eqx.field(static=True)
    boundary: ComplexBoundary = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        degree: int,
        exact_dimension: int,
        harmonic_rank: int,
        boundary: ComplexBoundary,
        topology_id: str,
        source_id: str,
        kernel_residuals: Array,
        orthonormality_residual: Array,
        next_eigenvalue: Array,
        ranks_match: Array,
        complete: Array,
    ) -> None:
        boundary = parse(boundary, ComplexBoundary, "boundary")
        if degree < 0 or exact_dimension < 0 or harmonic_rank < 0:
            raise ValueError("Cohomology dimensions and degree must be nonnegative.")
        self.kernel_residuals = kernel_residuals
        self.orthonormality_residual = orthonormality_residual
        self.next_eigenvalue = next_eigenvalue
        self.ranks_match, self.complete = ranks_match, complete
        self.degree, self.exact_dimension, self.harmonic_rank = (
            degree,
            exact_dimension,
            harmonic_rank,
        )
        self.boundary, self.topology_id, self.source_id = boundary, topology_id, source_id
        self.report_id = canonical_fingerprint(
            {
                "kind": "hodge-cohomology",
                "degree": degree,
                "exact_dimension": exact_dimension,
                "harmonic_rank": harmonic_rank,
                "boundary": boundary,
                "topology": topology_id,
                "source": source_id,
            }
        )


def validate_harmonic_cohomology(
    complex: AbstractDeRhamComplex,
    k: int,
    /,
    *,
    boundary: ComplexBoundary = "absolute",
    harmonic: HarmonicSubspace | None = None,
    tolerance: float = 1e-9,
    resources: TopologyResourcePolicy | None = None,
) -> tuple[HarmonicSubspace, HodgeCohomologyReport]:
    """Compare the active metric kernel against exact rational topology."""
    from ..discretization._cell_de_rham import AbstractCellDeRhamComplex
    from ..topology import CellSubcomplex, compute_betti_dimensions, RationalField

    if not isinstance(complex, AbstractCellDeRhamComplex):
        raise TypeError("Exact harmonic cohomology requires a cell de Rham realization.")
    boundary = parse(boundary, ComplexBoundary, "boundary")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")
    analysis = _analysis_complex(complex, boundary)
    exact = compute_betti_dimensions(
        analysis, coefficients=RationalField(), degrees=(k,), resources=resources
    )
    expected = exact.dimension(k)
    hilbert = complex.hilbert_complex(boundary=boundary)
    space = hilbert.space(k)
    resolved = (
        harmonic_subspace(
            hilbert,
            k,
            expected_dimension=expected,
            policy=HarmonicSubspacePolicy(tolerance=tolerance),
        )
        if harmonic is None
        else harmonic
    )
    if not isinstance(resolved, HarmonicSubspace):
        raise TypeError("harmonic must be a HarmonicSubspace or None.")
    if resolved.complex_id != hilbert.complex_id or resolved.degree != k:
        raise ValueError(
            "Harmonic subspace belongs to a different metric complex, degree, or boundary."
        )
    layout = (
        analysis.layout
        if isinstance(analysis, CellSubcomplex)
        else analysis.quotient_layout
    )
    if space.size != layout.counts[k]:
        raise ValueError(
            "Numerical active coordinates differ from exact topology coordinates."
        )
    rank = resolved.dimension
    basis = resolved.basis
    operator = hodge_laplacian(hilbert, k)
    images = jax.vmap(
        lambda value: space.flatten(operator.mv(space.unflatten(value))),
        in_axes=1,
        out_axes=1,
    )(basis)
    norms = jnp.linalg.norm(basis, axis=0)
    residuals = jnp.linalg.norm(images, axis=0) / jnp.maximum(
        norms, jnp.finfo(basis.real.dtype).tiny
    )
    mass = mass_form(hilbert, k)
    metric_basis = jax.vmap(mass.mv, in_axes=1, out_axes=1)(basis)
    gram = jnp.conj(basis.T) @ metric_basis
    orthonormality = jnp.max(jnp.abs(gram - jnp.eye(rank)), initial=0.0)
    ranks_match = jnp.asarray(rank == expected)
    complete = (
        ranks_match
        & resolved.valid
        & jnp.all(jnp.isfinite(residuals))
        & jnp.all(residuals <= tolerance)
        & (orthonormality <= 10 * tolerance)
    )
    report = HodgeCohomologyReport(
        degree=k,
        exact_dimension=expected,
        harmonic_rank=rank,
        boundary=boundary,
        topology_id=complex.topology.topology_id,
        source_id=exact.result_id,
        kernel_residuals=residuals,
        orthonormality_residual=orthonormality,
        next_eigenvalue=resolved.gap,
        ranks_match=ranks_match,
        complete=complete,
    )
    return resolved, report


def harmonic_kernel_certificate(
    complex: AbstractDeRhamComplex,
    k: int,
    /,
    *,
    boundary: ComplexBoundary = "absolute",
    harmonic: HarmonicSubspace | None = None,
    tolerance: float = 1e-9,
    resources: TopologyResourcePolicy | None = None,
) -> tuple[LinearSubspace, KernelCertificate, HodgeCohomologyReport]:
    """Certify the actual restricted Hodge kernel without a solver gauge choice."""
    resolved, report = validate_harmonic_cohomology(
        complex,
        k,
        boundary=boundary,
        harmonic=harmonic,
        tolerance=tolerance,
        resources=resources,
    )
    hilbert = complex.hilbert_complex(boundary=boundary)
    subspace = LinearSubspace(
        hilbert.space(k),
        resolved.basis,
        orthonormal=True,
        subspace_id=f"harmonic:{hilbert.complex_id}:{k}",
    )
    certificate = KernelCertificate(
        hodge_laplacian(hilbert, k),
        subspace,
        complete=bool(report.complete),
        tolerance=tolerance,
    )
    return subspace, certificate, report


@final
class HarmonicClassFrame(StrictModule, NonTrainableState):
    """Exact cycle labels and a prepared native period solve (never an inverse)."""

    exact_basis: RationalClassBasis
    harmonic_subspace: HarmonicSubspace
    harmonic_basis: Array
    cycles: Array
    period_matrix: Array
    period_defect: Array
    period_solve: PreparedLinearSolve | None
    solve_evidence: LinearSolveResult | None
    kernel_certificate: KernelCertificate
    cohomology_report: HodgeCohomologyReport
    degree: int = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)

    def __init__(
        self,
        exact_basis: RationalClassBasis,
        harmonic_subspace: HarmonicSubspace,
        harmonic_basis: Array,
        cycles: Array,
        period_matrix: Array,
        period_defect: Array,
        period_solve: PreparedLinearSolve | None,
        solve_evidence: LinearSolveResult | None,
        kernel_certificate: KernelCertificate,
        cohomology_report: HodgeCohomologyReport,
        /,
        *,
        realization_id: str,
    ) -> None:
        rank = exact_basis.generator_count
        if (
            harmonic_basis.ndim != 2
            or cycles.shape != harmonic_basis.shape
            or period_matrix.shape != (rank, rank)
        ):
            raise ValueError(
                "Harmonic frame cycle, basis and period dimensions disagree."
            )
        if (rank > 0) != (period_solve is not None):
            raise ValueError(
                "A nonempty harmonic frame requires a prepared period solve."
            )
        self.exact_basis, self.harmonic_subspace = exact_basis, harmonic_subspace
        self.harmonic_basis, self.cycles = harmonic_basis, cycles
        self.period_matrix, self.period_defect = period_matrix, period_defect
        self.period_solve, self.solve_evidence = period_solve, solve_evidence
        self.kernel_certificate, self.cohomology_report = (
            kernel_certificate,
            cohomology_report,
        )
        self.degree, self.realization_id = exact_basis.degree, realization_id
        self.frame_id = canonical_fingerprint(
            {
                "kind": "harmonic-class-frame",
                "basis": exact_basis.basis_id,
                "realization": realization_id,
                "complex": harmonic_subspace.complex_id,
                "degree": self.degree,
            }
        )

    def periods(self, cochain: ArrayLike, /) -> Array:
        values = jnp.asarray(cochain)
        if values.shape != (self.cycles.shape[0],):
            raise ValueError("Cochain coordinates do not match the harmonic frame.")
        return self.cycles.T @ values

    def with_periods(self, cochain: ArrayLike, target: ArrayLike, /) -> Array:
        values = jnp.asarray(cochain)
        desired = jnp.asarray(target)
        current = self.periods(values)
        if desired.shape != current.shape:
            raise ValueError("Target periods do not match the exact class count.")
        if self.period_solve is None:
            return values
        result = solve(self.period_solve, desired - current)
        correction = eqx.error_if(
            result.value,
            jnp.any(result.status != int(LinearSolveStatus.SUCCESS)),
            "Native harmonic period solve failed.",
        )
        return values + self.harmonic_basis @ correction


def prepare_harmonic_class_frame(
    complex: AbstractDeRhamComplex,
    exact_basis: RationalClassBasis,
    /,
    *,
    boundary: ComplexBoundary = "absolute",
    harmonic: HarmonicSubspace | None = None,
    tolerance: float = 1e-9,
    resources: TopologyResourcePolicy | None = None,
    policy: LinearSolvePolicy | None = None,
) -> HarmonicClassFrame:
    """Measure exact-generator periods and bind a reusable native LU solve."""
    from ..discretization._cell_de_rham import AbstractCellDeRhamComplex
    from ..topology import CellSubcomplex, RationalClassBasis

    if not isinstance(complex, AbstractCellDeRhamComplex) or not isinstance(
        exact_basis, RationalClassBasis
    ):
        raise TypeError(
            "A harmonic class frame requires a cell realization and rational class basis."
        )
    analysis = _analysis_complex(complex, parse(boundary, ComplexBoundary, "boundary"))
    source_id = (
        analysis.subcomplex_id
        if isinstance(analysis, CellSubcomplex)
        else analysis.pair_id
    )
    if exact_basis.source_id != source_id:
        raise ValueError(
            "Rational class basis belongs to a different cell complex or boundary."
        )
    k = exact_basis.degree
    resolved, report = validate_harmonic_cohomology(
        complex,
        k,
        boundary=boundary,
        harmonic=harmonic,
        tolerance=tolerance,
        resources=resources,
    )
    if not bool(report.complete) or resolved.dimension != exact_basis.generator_count:
        raise ValueError(
            "Harmonic frame requires complete numerical and exact class evidence."
        )
    subspace, certificate, _ = harmonic_kernel_certificate(
        complex,
        k,
        boundary=boundary,
        harmonic=resolved,
        tolerance=tolerance,
        resources=resources,
    )
    if not bool(certificate.valid):
        raise ValueError("Harmonic frame requires a valid measured kernel certificate.")
    layout = (
        analysis.layout
        if isinstance(analysis, CellSubcomplex)
        else analysis.quotient_layout
    )
    indices = np.asarray(layout.compact_to_ambient[k])
    cycles = jnp.asarray(
        np.asarray(exact_basis.dense(complex.cell_counts[k]), dtype=np.float64),
        dtype=resolved.basis.dtype,
    )
    full_basis = (
        jnp.zeros(cycles.shape, dtype=resolved.basis.dtype)
        .at[indices]
        .set(subspace.basis)
    )
    periods = cycles.T @ full_basis
    rank = exact_basis.generator_count
    prepared: PreparedLinearSolve | None = None
    evidence: LinearSolveResult | None = None
    if rank:
        operator = DenseLinearOperator(
            periods,
            operator_id=f"harmonic-periods:{complex.realization_id}:{exact_basis.basis_id}",
        )
        prepared = prepare(
            LinearSystem(operator),
            LinearSolvePolicy(DenseLU()) if policy is None else policy,
        )
        probe = jnp.arange(1, rank + 1, dtype=periods.dtype)
        evidence = solve(prepared, probe)
        defect = jnp.max(jnp.abs(periods @ evidence.value - probe))
        if not bool(
            jnp.all(evidence.status == int(LinearSolveStatus.SUCCESS))
        ) or not bool(jnp.isfinite(defect) & (defect <= tolerance)):
            raise ValueError("Harmonic period solve failed its numerical certificate.")
    else:
        defect = jnp.asarray(0.0, dtype=periods.real.dtype)
    return HarmonicClassFrame(
        exact_basis,
        resolved,
        full_basis,
        cycles,
        periods,
        defect,
        prepared,
        evidence,
        certificate,
        report,
        realization_id=complex.realization_id,
    )


def _inverse_sqrt(matrix: Array, /) -> tuple[Array, EigenSolveResult | None]:
    if matrix.shape[0] == 0:
        return matrix, None
    from ..linalg import OperatorProperties

    operator = DenseLinearOperator(
        matrix,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={"self_adjoint": "verified", "positive_definite": "verified"},
        ),
    )
    result = eigensolve(
        Eigenproblem(operator),
        policy=EigenSolvePolicy(DenseEigh(), count=matrix.shape[0]),
    )
    values = eqx.error_if(
        result.eigenvalues,
        (result.status != int(EigenSolveStatus.SUCCESS))
        | jnp.any(result.eigenvalues <= 0),
        "Native tracking Gram eigensolve must converge to a positive definite Gram.",
    )
    if not isinstance(result.eigenvectors, Array):
        raise TypeError("Native Gram eigensolve must return array eigenvectors.")
    return (result.eigenvectors / jnp.sqrt(values)[None, :]) @ jnp.conj(
        result.eigenvectors.T
    ), result


@final
class HodgeSubspaceTracking(StrictModule, NonTrainableState):
    """Principal angles and projector defects with native SVD evidence."""

    principal_angles: Array
    projector_residual: Array
    rank_changed: Array
    svd_evidence: SVDSolveResult | None
    gram_evidence: tuple[EigenSolveResult | None, EigenSolveResult | None]
    source_dimension: int = eqx.field(static=True)
    target_dimension: int = eqx.field(static=True)
    tracking_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_basis: ArrayLike,
        target_basis: ArrayLike,
        metric: ArrayLike,
        /,
        *,
        source_id: str,
        target_id: str,
    ) -> None:
        source, target, pairing = (
            jnp.asarray(source_basis),
            jnp.asarray(target_basis),
            jnp.asarray(metric),
        )
        if (
            source.ndim != 2
            or target.ndim != 2
            or pairing.shape != (source.shape[0], source.shape[0])
            or source.shape[0] != target.shape[0]
        ):
            raise ValueError(
                "Hodge tracking bases and metric must share one ambient space."
            )
        source_scale, source_evidence = _inverse_sqrt(
            jnp.conj(source.T) @ pairing @ source
        )
        target_scale, target_evidence = _inverse_sqrt(
            jnp.conj(target.T) @ pairing @ target
        )
        left, right = source @ source_scale, target @ target_scale
        overlap = jnp.conj(left.T) @ pairing @ right
        rank = min(source.shape[1], target.shape[1])
        evidence = (
            None
            if rank == 0
            else svd(
                SVDProblem(DenseLinearOperator(overlap)),
                policy=SVDSolvePolicy(count=rank),
            )
        )
        singular = (
            jnp.zeros((0,), dtype=source.real.dtype)
            if evidence is None
            else eqx.error_if(
                evidence.singular_values,
                evidence.status != int(SVDSolveStatus.SUCCESS),
                "Native Hodge tracking SVD failed.",
            )
        )
        self.principal_angles = jnp.arccos(jnp.clip(jnp.real(singular), 0.0, 1.0))
        self.projector_residual = jnp.linalg.norm(
            left @ jnp.conj(left.T) @ pairing - right @ jnp.conj(right.T) @ pairing
        )
        self.rank_changed = jnp.asarray(source.shape[1] != target.shape[1])
        self.svd_evidence = evidence
        self.gram_evidence = source_evidence, target_evidence
        self.source_dimension, self.target_dimension = source.shape[1], target.shape[1]
        self.tracking_id = canonical_fingerprint(
            {
                "kind": "hodge-subspace-tracking",
                "source": source_id,
                "target": target_id,
                "source_dimension": self.source_dimension,
                "target_dimension": self.target_dimension,
            }
        )


__all__ = [
    "HodgeCohomologyReport",
    "validate_harmonic_cohomology",
    "harmonic_kernel_certificate",
    "HarmonicClassFrame",
    "prepare_harmonic_class_frame",
    "HodgeSubspaceTracking",
]
