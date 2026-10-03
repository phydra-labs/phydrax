#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Auxiliary-space preconditioning of collocated meshfree systems.

A collocated strong-form operator ``A`` on a cloud is preconditioned through
the symmetric positive definite linear-simplex stiffness ``K`` of the same
coefficient on a Delaunay triangulation of the same points. Interior rows are
mapped to weak-form rows by the lumped mass and corrected with an approximate
inverse of ``K``; collocated flux/traction rows are eliminated exactly. The
composition is the native multiplicative subspace correction

``z1 = E M_K E (s * r)``,   ``z = z1 + S^T A_TT^{-1} S (r - A z1)``,

where ``E`` zeroes trace rows and ``S`` selects them. ``[K^{-1}]_II`` is the
inverse of ``K``'s Schur complement on the interior, i.e. the same eliminated
boundary-value problem the collocated interior block solves, so ``K`` needs no
counterpart of the collocated boundary rows.
"""

from __future__ import annotations

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry._triangulation import (
    DelaunayTriangulation,
    SimplexQualityEvidence,
    SimplexQualitySubcomplex,
    TriangulationEvidence,
)
from ...linalg import (
    AbstractPreconditioner,
    ArraySpace,
    assemble_sparse,
    BlockLinearOperator,
    BlockSpace,
    DiagonalLinearOperator,
    EuclideanPairing,
    JacobiPreconditionerBuilder,
    MaterializationPolicy,
    MultiplicativeSubspaceCorrectionBuilder,
    OperatorProperties,
    PreconditionerSource,
    SparseAssemblyPolicy,
    SparseFactorizationPolicy,
    SparseFactorizationPreconditionerBuilder,
    SubspaceCorrectionTerm,
)
from ...sparse import EdgeRelation, SparseCoordinateOperator
from ...typing import checked
from .._cell_mesh import CellMesh
from ..fem import FiniteElementFieldSpec, FiniteElementPlan
from ..fem._reference import lagrange_element
from ._multilevel import (
    meshfree_multigrid_builder,
    MeshfreeComponentSpace,
    MeshfreeHierarchyPlan,
    MeshfreeNearNullspace,
)


# Normalized volume-length ratio below which a Delaunay simplex is excluded
# from the auxiliary stiffness. Measured on jittered cube clouds (3-D
# traction elasticity, default ghost route): GMRES iterations 71 -> 46 at
# N=1000 and 166 -> 57 at N=8000, excluding about 5% of the simplices that
# carry under 1% of the measure.
_MINIMUM_SIMPLEX_QUALITY = 0.1
# Damped Jacobi relaxation of the auxiliary hierarchy: 0.45 * 4.45 < 2 for the
# measured level spectra lambda_max(D^-1 K) (see meshfree_auxiliary_builder).
_JACOBI_RELAXATION = 0.45


@final
class MeshfreeAuxiliaryStiffness(StrictModule, NonTrainableState):
    """Linear-simplex stiffness of a coupled elliptic coefficient on cloud nodes.

    The canonical Delaunay triangulation of the points (provider ``"qhull"``:
    a preconditioner-only auxiliary mesh needs no exact predicates and must not
    require the optional exact provider) is screened for quality
    (`SimplexQualitySubcomplex`): flat simplices carry no measure, and
    simplices of normalized volume-length ratio below the threshold (3-D
    slivers) are excluded unless needed to cover a point or keep the complex
    facet-connected. A sliver's P1 gradient is ill-conditioned, so its exact
    element energy grossly overestimates the continuum energy of non-affine
    fields; omitting its negligible measure keeps the auxiliary energy
    spectrally close to the continuum one (``quality`` records the threshold,
    counts, and excluded measure fraction). The retained simplices carry
    continuous piecewise-linear fields; each block ``K_ab`` is the native
    finite-element cell assembly of ``int d_i phi_p C_abij d_j phi_q`` with the
    element coefficient the vertex mean, composed in component-major
    coordinates and restricted to the non-eliminated coordinates (eliminated
    rows and columns are identity). It is symmetric by construction and
    positive definite when the eliminated coordinates anchor every kernel mode;
    it approximates the same continuum energy as a consistent collocated
    operator, which it preconditions only. ``lumped_mass`` holds the native
    mass-matrix row sums. The triangulation covers the convex hull: on
    nonconvex clouds it also couples points across gaps, which weakens but
    never invalidates the preconditioner (the outer Krylov solve keeps the
    original equation).
    """

    operator: SparseCoordinateOperator
    lumped_mass: Array
    points: Array
    eliminated: Array
    components: MeshfreeComponentSpace
    triangulation: TriangulationEvidence
    quality: SimplexQualityEvidence
    stiffness_id: str = eqx.field(static=True)


@checked
def meshfree_auxiliary_stiffness(
    points: ArrayLike,
    coefficients: ArrayLike,
    space: ArraySpace,
    /,
    *,
    eliminated: ArrayLike | None = None,
    assembly: SparseAssemblyPolicy | None = None,
    minimum_quality: float = _MINIMUM_SIMPLEX_QUALITY,
) -> MeshfreeAuxiliaryStiffness:
    """Assemble the Delaunay P1 stiffness ``int grad v_a : C_ab : grad u_b``.

    ``coefficients`` has shape ``(m, m, d, d)`` or ``(points, m, m, d, d)``
    with ``C[a, b, i, j]`` coupling ``d_i v_a`` and ``d_j u_b`` (the block
    coupling convention of meshfree block systems). ``space`` is the flat
    Euclidean float64 solve space of ``points * m`` component-major
    coordinates and ``eliminated`` marks decoupled identity coordinates;
    ``assembly`` bounds the block composition. ``minimum_quality`` is the
    `SimplexQualitySubcomplex` threshold (0 keeps every simplex of positive
    measure).
    """
    host_points = np.asarray(jax.device_get(jnp.asarray(points)), dtype=np.float64)
    if host_points.ndim != 2 or host_points.shape[1] not in (2, 3):
        raise ValueError("Auxiliary stiffness requires 2-D or 3-D points.")
    count, dimension = host_points.shape
    coefficient = np.asarray(jax.device_get(jnp.asarray(coefficients)), dtype=np.float64)
    if coefficient.ndim not in (4, 5) or coefficient.shape[-2:] != (dimension, dimension):
        raise ValueError(
            "coefficients must have shape (m, m, d, d) or (points, m, m, d, d)."
        )
    m = coefficient.shape[-4]
    shape = (m, m, dimension, dimension)
    if coefficient.shape == shape:
        coefficient = np.broadcast_to(coefficient, (count, *shape))
    if coefficient.shape != (count, *shape) or not np.all(np.isfinite(coefficient)):
        raise ValueError(
            "coefficients must be finite with shape (m, m, d, d) or (points, m, m, d, d)."
        )
    if (
        space.shape != (count * m,)
        or not isinstance(space.pairing, EuclideanPairing)
        or space.dtype != np.dtype(np.float64)
    ):
        raise ValueError(
            "space must be the flat Euclidean float64 space of points * components coordinates."
        )
    mask = (
        np.zeros(count * m, dtype=np.bool_)
        if eliminated is None
        else np.asarray(jax.device_get(jnp.asarray(eliminated)))
    )
    if mask.shape != (count * m,) or mask.dtype != np.bool_:
        raise ValueError(
            "eliminated must be a Boolean mask with one entry per coordinate."
        )
    triangulation = DelaunayTriangulation(host_points, provider="qhull")
    if triangulation.evidence.duplicate_count or triangulation.evidence.redundant_count:
        raise ValueError("Auxiliary stiffness requires every point to be a vertex.")
    screened = SimplexQualitySubcomplex(
        host_points, triangulation.simplices, minimum_quality=minimum_quality
    )
    simplices = screened.simplices
    if np.unique(simplices).size != count:
        raise ValueError("Every point must belong to a simplex of positive measure.")
    mesh = (
        CellMesh.from_triangles(host_points, simplices)
        if dimension == 2
        else CellMesh.from_tetrahedra(host_points, simplices)
    )
    element = lagrange_element("triangle" if dimension == 2 else "tetrahedron", 1)
    discretization = FiniteElementPlan(
        mesh, FiniteElementFieldSpec("auxiliary", element)
    ).prepare()
    dof_map = discretization.dof_maps[0]
    if not np.array_equal(np.asarray(dof_map.dof_coordinates), host_points):
        raise RuntimeError("Linear Lagrange dofs must coincide with the cloud points.")
    (geometry,) = discretization.evaluate_geometry(
        "auxiliary", discretization.default_runtime.coordinates
    )
    cells = np.asarray(dof_map.cell_dofs[0])
    local = ein.contract(
        "cq,cqpi,cabij,cqrj->abcpr",
        geometry.physical_weights,
        geometry.physical_gradients,
        jnp.asarray(np.mean(coefficient[cells], axis=1)),
        geometry.physical_gradients,
    )
    identifier = canonical_fingerprint(
        {
            "kind": "meshfree-auxiliary-stiffness",
            "points": array_tree_fingerprint(host_points),
            "coefficients": array_tree_fingerprint(coefficient),
            "eliminated": array_tree_fingerprint(mask),
            "components": m,
            "simplices": array_tree_fingerprint(simplices),
            "minimum_quality": screened.evidence.minimum_quality,
        }
    )
    blocks = tuple(
        tuple(
            discretization.assemble_cell_operator(
                "auxiliary",
                (local[a, b],),
                operator_id=f"{identifier}:block:{a}:{b}",
            )
            for b in range(m)
        )
        for a in range(m)
    )
    scalar = blocks[0][0].source
    block_space = BlockSpace((scalar,) * m, names=tuple(f"c{a}" for a in range(m)))
    assembled = assemble_sparse(
        BlockLinearOperator(blocks, source=block_space, target=block_space), assembly
    )
    if not isinstance(assembled, SparseCoordinateOperator):
        raise TypeError("Auxiliary block assembly requires canonical coordinate storage.")
    # The block space flattens component by component: the assembled relation
    # is already in the component-major solve coordinates.
    flat = SparseCoordinateOperator(
        assembled.relation, assembled.coefficients, source=space, target=space
    )
    keep = DiagonalLinearOperator(jnp.asarray((~mask).astype(np.float64)), space=space)
    identity = DiagonalLinearOperator(jnp.asarray(mask.astype(np.float64)), space=space)
    restricted = assemble_sparse(keep @ flat @ keep + identity, assembly)
    if not isinstance(restricted, SparseCoordinateOperator):
        raise TypeError(
            "Auxiliary restriction assembly requires canonical coordinate storage."
        )
    operator = SparseCoordinateOperator(
        restricted.relation,
        restricted.coefficients,
        source=space,
        target=space,
        properties=OperatorProperties(
            self_adjoint=True, evidence={"self_adjoint": "construction"}
        ),
        operator_id=identifier,
    )
    lumped = discretization.mass.mv(jnp.ones((count,), dtype=jnp.float64))
    return MeshfreeAuxiliaryStiffness(
        operator=operator,
        lumped_mass=lumped,
        points=jnp.asarray(host_points),
        eliminated=jnp.asarray(mask),
        components=MeshfreeComponentSpace(m, layout="block"),
        triangulation=triangulation.evidence,
        quality=screened.evidence,
        stiffness_id=identifier,
    )


def _selection(
    rows: np.ndarray, space: ArraySpace, /
) -> tuple[SparseCoordinateOperator, SparseCoordinateOperator]:
    """Exact row selection ``S`` onto ``rows`` and its sparse transpose."""
    local = ArraySpace(
        (rows.size,),
        dtype=space.dtype,
        space_id=canonical_fingerprint(
            {
                "kind": "meshfree-auxiliary-trace-space",
                "space": space.space_id,
                "rows": array_tree_fingerprint(rows),
            }
        ),
    )
    positions = np.arange(rows.size, dtype=np.int32)
    ones = jnp.ones((rows.size,), dtype=space.dtype)
    restriction = SparseCoordinateOperator(
        EdgeRelation(rows, positions, source_size=space.size, target_size=rows.size),
        ones,
        source=space,
        target=local,
    )
    prolongation = SparseCoordinateOperator(
        EdgeRelation(positions, rows, source_size=rows.size, target_size=space.size),
        ones,
        source=local,
        target=space,
    )
    return restriction, prolongation


@checked
def meshfree_auxiliary_builder(
    stiffness: MeshfreeAuxiliaryStiffness,
    /,
    *,
    trace: ArrayLike | None = None,
    row_measure: ArrayLike | None = None,
    near_nullspace: MeshfreeNearNullspace | None = None,
    interior: PreconditionerSource | None = None,
    assembly: SparseAssemblyPolicy | None = None,
) -> MultiplicativeSubspaceCorrectionBuilder:
    """Auxiliary-space preconditioner for a collocated system on the same cloud.

    Interior residual rows are scaled by the lumped mass (divided by
    ``row_measure`` when the system rows are already quadrature-weighted) and
    corrected by an approximate inverse of the auxiliary stiffness; ``trace``
    rows (flux, traction, Robin) are then eliminated by an exact sparse
    factorization of their Galerkin block ``A_TT``. The auxiliary action is
    prepared here, once: by default a geometric meshfree hierarchy of the
    auxiliary stiffness (``near_nullspace`` transfer candidates, constants per
    component by default and ``"rigid-body"`` for elasticity; eliminated
    coordinates excluded) with damped Jacobi smoothing (relaxation 0.45) and
    an exact sparse coarse solve. On the SPD auxiliary stiffness the measured
    ``lambda_max(D^-1 K)`` of every level stays below 4.45 (2-D/3-D elasticity,
    lambda/mu = 2 and 200), so the relaxation keeps the smoother convergent,
    unlike stationary smoothing of the collocated operator. The returned
    native multiplicative subspace correction is linear and fixed but not
    self-adjoint: use it as a right preconditioner of GMRES. Numeric refresh
    reuses the auxiliary action and refreshes the trace factorization.
    """
    space = stiffness.operator.source
    if not isinstance(space, ArraySpace):
        raise RuntimeError("Auxiliary stiffness always lives on an ArraySpace.")
    count = stiffness.points.shape[0]
    m = stiffness.components.components
    measure = (
        np.ones((count,))
        if row_measure is None
        else np.asarray(jax.device_get(jnp.asarray(row_measure)), dtype=np.float64)
    )
    if (
        measure.shape != (count,)
        or not np.all(np.isfinite(measure))
        or np.any(measure <= 0)
    ):
        raise ValueError("row_measure must hold one finite positive value per point.")
    eliminated = np.asarray(jax.device_get(stiffness.eliminated))
    trace_mask = (
        np.zeros(count * m, dtype=np.bool_)
        if trace is None
        else np.asarray(jax.device_get(jnp.asarray(trace)))
    )
    if trace_mask.shape != eliminated.shape or trace_mask.dtype != np.bool_:
        raise ValueError("trace must be a Boolean mask with one entry per coordinate.")
    if np.any(trace_mask & eliminated):
        raise ValueError("A coordinate cannot be both a trace row and eliminated.")
    # Component-major coordinates repeat the per-point scale per component.
    scale = np.tile(np.asarray(jax.device_get(stiffness.lumped_mass)) / measure, m)
    interior_mask = ~trace_mask
    scale = np.where(eliminated, 1.0, scale) * interior_mask
    if interior is None:
        # Geometric meshfree hierarchy of the SPD auxiliary stiffness with
        # damped Jacobi and an exact sparse coarse solve. Measured on the
        # default traction elasticity route (quality-screened stiffness):
        # symmetric Gauss--Seidel needs 58/46 GMRES iterations (2-D N=4225 /
        # 3-D N=1000) at 79/66 ms per iteration, because its sweep is
        # row-sequential; Jacobi needs 75/49 at 23/18 ms. Native smoothed
        # aggregation of the screened 3-D stiffness is refused (level-1
        # Galerkin product of 4.5e7 symbolic contributions at N=1000), whereas
        # width-limited meshfree transfers stay sparse.
        hierarchy = MeshfreeHierarchyPlan(
            stiffness.points,
            eliminated=jnp.asarray(eliminated),
            components=stiffness.components,
            near_nullspace=(
                MeshfreeNearNullspace() if near_nullspace is None else near_nullspace
            ),
        ).prepare(space)
        interior = meshfree_multigrid_builder(
            hierarchy,
            smoothers=tuple(
                JacobiPreconditionerBuilder(relaxation=_JACOBI_RELAXATION)
                for _ in hierarchy.transfers
            )
            or None,
            assembly=assembly,
        )
    elif near_nullspace is not None:
        raise ValueError("near_nullspace configures only the default interior source.")
    action: AbstractPreconditioner = (
        interior
        if isinstance(interior, AbstractPreconditioner)
        else interior.prepare(stiffness.operator, materialization=MaterializationPolicy())
    )
    terms = [
        SubspaceCorrectionTerm(
            DiagonalLinearOperator(jnp.asarray(scale), space=space),
            DiagonalLinearOperator(
                jnp.asarray(interior_mask.astype(np.float64)), space=space
            ),
            action,
        )
    ]
    rows = np.flatnonzero(trace_mask).astype(np.int32)
    if rows.size:
        restriction, prolongation = _selection(rows, space)
        # The trace rows of a 3-D face couple as a surface graph: in natural
        # order their LU fills almost densely (measured 3.1e8 symbolic updates,
        # 437 s, for 972 trace rows at 8000 points); a minimum-degree ordering
        # keeps the factor sparse.
        terms.append(
            SubspaceCorrectionTerm(
                restriction,
                prolongation,
                SparseFactorizationPreconditionerBuilder(
                    SparseFactorizationPolicy("lu", ordering="approximate-minimum-degree")
                ),
                assembly=assembly,
            )
        )
    return MultiplicativeSubspaceCorrectionBuilder(tuple(terms), sweep="forward")


__all__ = [
    "MeshfreeAuxiliaryStiffness",
    "meshfree_auxiliary_builder",
    "meshfree_auxiliary_stiffness",
]
