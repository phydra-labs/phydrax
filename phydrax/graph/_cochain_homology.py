#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CochainBoundaryKind
from ..linalg import (
    ArraySpace,
    FunctionLinearOperator,
    KernelCertificate,
    LinearSubspace,
)
from ..topology import (
    CellComplexPair,
    CellSubcomplex,
    compute_betti_dimensions,
    RationalField,
    TopologyResourcePolicy,
)
from ._cochain import CochainComplexIR, HarmonicSubspace
from ._cochain_spectrum import compute_harmonic_subspace


class HodgeHomologyReport(StrictModule, NonTrainableState):
    """Exact-nullity and numerical harmonic-kernel comparison evidence."""

    kernel_residuals: Array
    orthonormality_residual: Array
    next_eigenvalue: Array
    ranks_match: Array
    complete: Array
    degree: int = eqx.field(static=True)
    exact_dimension: int = eqx.field(static=True)
    harmonic_rank: int = eqx.field(static=True)
    boundary_policy: CochainBoundaryKind = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        degree: int,
        exact_dimension: int,
        harmonic_rank: int,
        boundary_policy: CochainBoundaryKind,
        topology_id: str,
        source_id: str,
        kernel_residuals: Array,
        orthonormality_residual: Array,
        next_eigenvalue: Array,
        ranks_match: Array,
        complete: Array,
    ):
        self.kernel_residuals = jnp.asarray(kernel_residuals)
        self.orthonormality_residual = jnp.asarray(orthonormality_residual)
        self.next_eigenvalue = jnp.asarray(next_eigenvalue)
        self.ranks_match = jnp.asarray(ranks_match)
        self.complete = jnp.asarray(complete)
        self.degree = int(degree)
        self.exact_dimension = int(exact_dimension)
        self.harmonic_rank = int(harmonic_rank)
        self.boundary_policy = boundary_policy
        self.topology_id = str(topology_id)
        self.source_id = str(source_id)
        self.report_id = canonical_fingerprint(
            {
                "kind": "hodge-homology-report",
                "degree": int(degree),
                "exact_dimension": int(exact_dimension),
                "harmonic_rank": int(harmonic_rank),
                "boundary_policy": boundary_policy,
                "topology": self.topology_id,
                "source": self.source_id,
            }
        )


def _analysis_complex(
    complex_ir: CochainComplexIR,
    boundary_policy: CochainBoundaryKind,
    /,
) -> CellSubcomplex | CellComplexPair:
    topology = complex_ir.discretization.topology
    ambient = CellSubcomplex.full(topology)
    if boundary_policy == "absolute":
        return ambient
    if boundary_policy != "relative":
        raise ValueError("Hodge homology boundary policy must be absolute or relative.")
    relative = CellSubcomplex(topology, complex_ir.boundary_masks)
    return CellComplexPair(ambient, relative)


def _layout(value: CellSubcomplex | CellComplexPair):
    return value.layout if isinstance(value, CellSubcomplex) else value.quotient_layout


def _compact_operator(
    complex_ir: CochainComplexIR,
    degree: int,
    boundary_policy: CochainBoundaryKind,
    compact_to_ambient: Array,
    /,
):
    count = complex_ir.cell_counts[int(degree)]

    def apply(values):
        ambient = jnp.zeros((count,), dtype=values.dtype)
        ambient = ambient.at[compact_to_ambient].set(values)
        image = complex_ir.discretization.laplace_de_rham(
            int(degree),
            ambient,
            boundary_policy=boundary_policy,
        )
        return image[compact_to_ambient]

    return apply


def validate_hodge_homology(
    complex_ir: CochainComplexIR,
    degree: int,
    /,
    *,
    boundary_policy: CochainBoundaryKind = "absolute",
    harmonic_subspace: HarmonicSubspace | None = None,
    tolerance: float = 1e-9,
    resources: TopologyResourcePolicy | None = None,
) -> tuple[HarmonicSubspace, HodgeHomologyReport]:
    """Compare exact rational Betti dimension with a metric harmonic basis."""
    if not isinstance(complex_ir, CochainComplexIR):
        raise TypeError("Hodge homology validation requires a CochainComplexIR.")
    degree_ = int(degree)
    if degree_ < 0 or degree_ > complex_ir.max_degree:
        raise ValueError(f"degree must lie in [0, {complex_ir.max_degree}].")
    tolerance_ = float(tolerance)
    if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
        raise ValueError("Hodge homology tolerance must be finite and positive.")
    analysis = _analysis_complex(complex_ir, boundary_policy)
    layout = _layout(analysis)
    exact = compute_betti_dimensions(
        analysis,
        coefficients=RationalField(),
        degrees=(degree_,),
        resources=resources,
    )
    exact_dimension = exact.dimension(degree_)
    resolved = (
        compute_harmonic_subspace(
            complex_ir,
            boundary_policy=boundary_policy,
            max_modes=max(1, exact_dimension + 1),
            tolerance=tolerance_,
        )
        if harmonic_subspace is None
        else harmonic_subspace
    )
    if not isinstance(resolved, HarmonicSubspace):
        raise TypeError("harmonic_subspace must be a HarmonicSubspace or None.")
    if resolved.complex_fingerprint != complex_ir.fingerprint:
        raise ValueError("Harmonic subspace belongs to a different metric complex.")
    if resolved.boundary_policy != boundary_policy:
        raise ValueError("Harmonic subspace uses a different boundary policy.")
    harmonic_rank = resolved.ranks[degree_]
    compact = layout.compact_to_ambient[degree_]
    basis = resolved.bases[degree_][compact, :harmonic_rank]
    apply = _compact_operator(
        complex_ir,
        degree_,
        boundary_policy,
        compact,
    )
    if harmonic_rank:
        images = jnp.stack(
            tuple(apply(basis[:, index]) for index in range(harmonic_rank)),
            axis=1,
        )
        norms = jnp.linalg.norm(basis, axis=0)
        residuals = jnp.linalg.norm(images, axis=0) / jnp.maximum(
            norms,
            jnp.finfo(basis.real.dtype).tiny,
        )
        hodge_matrix = complex_ir.discretization.hodge_matrices[degree_]
        metric = (
            jnp.diag(complex_ir.discretization.hodge_stars[degree_][compact])
            if hodge_matrix is None
            else hodge_matrix[jnp.ix_(compact, compact)]
        )
        gram = jnp.conj(basis.T) @ metric @ basis
        orthonormality = jnp.max(jnp.abs(gram - jnp.eye(harmonic_rank)))
    else:
        residuals = jnp.zeros((0,), dtype=complex_ir.hodge_stars[degree_].dtype)
        orthonormality = jnp.asarray(0.0, dtype=residuals.dtype)
    stored_values = resolved.eigenvalues[degree_]
    next_eigenvalue = (
        stored_values[harmonic_rank]
        if harmonic_rank < resolved.max_modes
        else jnp.asarray(jnp.inf, dtype=stored_values.dtype)
    )
    ranks_match = jnp.asarray(harmonic_rank == exact_dimension)
    residual_valid = jnp.all(jnp.isfinite(residuals)) & jnp.all(residuals <= tolerance_)
    gap_valid = jnp.asarray(exact_dimension == layout.counts[degree_]) | (
        jnp.isfinite(next_eigenvalue) & (next_eigenvalue > tolerance_)
    )
    complete = (
        ranks_match & residual_valid & (orthonormality <= 10.0 * tolerance_) & gap_valid
    )
    report = HodgeHomologyReport(
        degree=degree_,
        exact_dimension=exact_dimension,
        harmonic_rank=harmonic_rank,
        boundary_policy=boundary_policy,
        topology_id=complex_ir.discretization.topology.topology_id,
        source_id=exact.result_id,
        kernel_residuals=residuals,
        orthonormality_residual=orthonormality,
        next_eigenvalue=next_eigenvalue,
        ranks_match=ranks_match,
        complete=complete,
    )
    return resolved, report


def cochain_harmonic_kernel_certificate(
    complex_ir: CochainComplexIR,
    degree: int,
    /,
    *,
    boundary_policy: CochainBoundaryKind = "absolute",
    harmonic_subspace: HarmonicSubspace | None = None,
    tolerance: float = 1e-9,
    resources: TopologyResourcePolicy | None = None,
) -> tuple[LinearSubspace, KernelCertificate, HodgeHomologyReport]:
    """Build compact verified Hodge-kernel evidence without choosing a solver gauge."""
    resolved, report = validate_hodge_homology(
        complex_ir,
        degree,
        boundary_policy=boundary_policy,
        harmonic_subspace=harmonic_subspace,
        tolerance=tolerance,
        resources=resources,
    )
    analysis = _analysis_complex(complex_ir, boundary_policy)
    layout = _layout(analysis)
    degree_ = int(degree)
    compact = layout.compact_to_ambient[degree_]
    basis = resolved.bases[degree_][compact, : resolved.ranks[degree_]]
    dtype = complex_ir.hodge_stars[degree_].dtype
    space = ArraySpace(
        (layout.counts[degree_],),
        dtype=dtype,
        space_id=f"cochain-harmonic:{report.report_id}:space",
    )
    apply = _compact_operator(
        complex_ir,
        degree_,
        boundary_policy,
        compact,
    )
    operator = FunctionLinearOperator(
        apply,
        source=space,
        target=space,
        operator_id=f"cochain-harmonic:{report.report_id}:laplacian",
    )
    subspace = LinearSubspace(
        space,
        basis.astype(dtype),
        dimension=resolved.ranks[degree_],
        orthonormal=False,
        subspace_id=f"cochain-harmonic:{report.report_id}:kernel",
    )
    certificate = KernelCertificate(
        operator,
        subspace,
        evidence="verified",
        scope="numerical",
        complete=bool(report.complete),
        tolerance=float(tolerance),
    )
    return subspace, certificate, report


__all__ = [
    "HodgeHomologyReport",
    "cochain_harmonic_kernel_certificate",
    "validate_hodge_homology",
]
