#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Coupled second-order block equations on prepared point clouds.

The system ``-sum_b div(C_ab grad u_b) + g_a(u, x) = f_a`` couples named
components through a fourth-order coefficient ``C[a, b, i, j]``. Each component
owns its boundary rows (Dirichlet values, conormal ``sum_b n_i C_abij d_j u_b``
such as elastic traction, Robin, or periodic identification). Blocks are
assembled through native ``BlockSpace`` sparse assembly; this is not a
channel-wise application of the scalar Poisson operator.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import canonical_identifier
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    BlockLinearOperator,
    BlockSpace,
    DensePropertyVerificationPolicy,
    DiagonalLinearOperator,
    FailurePolicy,
    GMRES,
    ILUPreconditionerBuilder,
    LinearSolveControl,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    plan_sparse_assembly,
    PreconditioningPolicy,
    prepare,
    prepare_sparse_assembly,
    PreparedLinearSolve,
    PreparedSparseAssembly,
    refresh,
    refresh_sparse_assembly,
    solve,
    SolveResourcePolicy,
    SparseAssemblyPolicy,
    TolerancePolicy,
    transpose,
    verify_dense_properties,
)
from ...linalg.eigen import GeneralEigenSolvePolicy
from ...nonlinear import (
    NewtonKrylov,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
    root,
)
from ...sparse import SparseCoordinateOperator
from ...typing import parse
from .._point_cloud import PreparedPointCloudDiscretization
from .._point_cloud_pde import (
    _capacity_assembly_policy,
    _components_and_gauges,
    _image_relation,
    _module_identity,
    _row_layout,
    _RowLayout,
    _selection,
    _unit,
    assess_square_collocation,
    collocation_stability_assessment,
    PointBoundaryPlan,
    PointCollocationStability,
    PointDiffusionForm,
    PointStabilityPolicy,
    PreparedCollocationAssessment,
)
from ._auxiliary import (
    meshfree_auxiliary_builder,
    meshfree_auxiliary_stiffness,
    MeshfreeAuxiliaryStiffness,
)
from ._boundary import (
    discretization_family,
    PointDerivativeFamily,
    PreparedPointGhostLayer,
)
from ._multilevel import (
    meshfree_multigrid_builder,
    MeshfreeCoarseningPolicy,
    MeshfreeComponentSpace,
    MeshfreeHierarchyPlan,
    MeshfreeNearNullspace,
    PreparedMeshfreeHierarchy,
)


def isotropic_elasticity_coefficients(
    lame_lambda: ArrayLike, shear_modulus: ArrayLike, dimension: int, /
) -> Array:
    """``C[a, b, i, j] = lambda d_ai d_bj + mu (d_ab d_ij + d_aj d_bi)``.

    Point-varying moduli of shape ``(points,)`` give ``(points, d, d, d, d)``.
    The divergence form then reads ``-div sigma(u)`` with Hooke's law.
    """
    if dimension not in (1, 2, 3):
        raise ValueError("Elasticity coefficients support dimensions one to three.")
    lam = jnp.asarray(lame_lambda, dtype=jnp.float64)
    mu = jnp.asarray(shear_modulus, dtype=jnp.float64)
    eye = jnp.eye(dimension, dtype=jnp.float64)
    volumetric = eye[:, None, :, None] * eye[None, :, None, :]
    shear = (
        eye[:, :, None, None] * eye[None, None, :, :]
        + eye[:, None, None, :] * eye[None, :, :, None]
    )
    return (
        lam[..., None, None, None, None] * volumetric
        + mu[..., None, None, None, None] * shear
    )


@final
class PointCouplingEvidence(StrictModule, NonTrainableState):
    """Native evidence that ``C`` has major symmetry and is positive semidefinite.

    The flattened ``(a i), (b j)`` matrix is verified per point. Strong
    ellipticity is not implied; the solve's original residual decides.
    """

    minimum_eigenvalue: Array
    maximum_eigenvalue: Array
    symmetry_defect: Array
    finite: Array
    positive_semidefinite: Array

    @property
    def successful(self) -> Array:
        return self.finite & self.positive_semidefinite


@final
class PointBlockSystemResult(StrictModule):
    values: Array
    residual_norm: Array
    residual_tolerance: Array
    boundary_residual_norm: Array
    linear_result: LinearSolveResult
    coupling_evidence: PointCouplingEvidence
    ghost_values: Array | None
    ghost_extension_defect: Array | None
    components: tuple[str, ...] = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return (
            self.linear_result.successful
            & self.coupling_evidence.successful
            & jnp.isfinite(self.residual_norm)
            & (self.residual_norm <= self.residual_tolerance)
        )


@final
class PointBlockNonlinearResult(StrictModule):
    values: Array
    residual_norm: Array
    nonlinear_result: NonlinearResult
    coupling_evidence: PointCouplingEvidence
    components: tuple[str, ...] = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.nonlinear_result.successful & self.coupling_evidence.successful


@final
class PointBlockSystemPlan(StrictModule):
    """Coupled block equations with component-specific boundary ownership.

    Components must each be anchored by Dirichlet or positive-Robin rows on
    every connected component; floating systems (rigid or constant modes)
    are refused rather than gauged by an implicit choice.

    Without ``linear_policy`` the solve is GMRES (restart up to 200) right
    preconditioned by a numerically refreshed ILU of the assembled block
    operator. ``auxiliary`` selects the auxiliary-space route: the interior
    rows are preconditioned by a symmetric Gauss--Seidel meshfree hierarchy of the symmetric
    positive definite Delaunay P1 stiffness of the same coefficient on the
    same points (``meshfree_auxiliary_stiffness``) with the given candidates
    (``MeshfreeNearNullspace("rigid-body")`` for elasticity), and the
    flux/traction rows are eliminated exactly by a sparse factorization of
    their block. It needs Dirichlet rows in every component (anchoring the
    auxiliary stiffness), no periodic rows, and 2-D or 3-D points; the
    auxiliary hierarchy is prepared with the coefficients and rebuilt on
    coefficient refresh. ``multigrid`` instead builds a block meshfree
    hierarchy of the collocated operator itself on component-major
    (``"block"``) coordinates with eliminated Dirichlet coordinates; no
    stationary smoother contracts collocated block operators, so its
    iteration count grows with resolution. ``auxiliary``, ``multigrid`` and
    ``coarsening`` configure only the default policy; an explicit
    ``linear_policy`` owns its own preconditioning.

    Preparation runs the native square-collocation spectral assessment
    (``assess_square_collocation``) on the flat solve operator restricted to
    the non-Dirichlet coordinates. Under ``stability="require-assessment"``
    (default) an operator with converged eigenvalue estimates of nonpositive
    real part, or an unfinished assessment, is refused with
    ``PointCollocationStabilityRefusal``; ``"diagnostic"`` records the
    evidence on ``PreparedPointBlockSystem.stability`` and proceeds.
    """

    discretization: PreparedPointCloudDiscretization
    boundary: PointBoundaryPlan
    layouts: tuple[_RowLayout, ...]
    linear_policy: LinearSolvePolicy
    assembly_policy: SparseAssemblyPolicy
    solve_space: ArraySpace
    hierarchy: PreparedMeshfreeHierarchy | None
    auxiliary: MeshfreeNearNullspace | None
    ghosts: PreparedPointGhostLayer | None
    stability_assessment: GeneralEigenSolvePolicy
    components: tuple[str, ...] = eqx.field(static=True)
    form: PointDiffusionForm = eqx.field(static=True)
    stability: PointStabilityPolicy = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: PreparedPointCloudDiscretization,
        boundary: PointBoundaryPlan,
        /,
        *,
        components: Sequence[str],
        form: PointDiffusionForm = "collocated",
        linear_policy: LinearSolvePolicy | None = None,
        assembly_policy: SparseAssemblyPolicy | None = None,
        multigrid: MeshfreeNearNullspace | None = None,
        coarsening: MeshfreeCoarseningPolicy | None = None,
        auxiliary: MeshfreeNearNullspace | None = None,
        stability: PointStabilityPolicy = "require-assessment",
        stability_assessment: GeneralEigenSolvePolicy | None = None,
        ghosts: PreparedPointGhostLayer | None = None,
    ) -> None:
        if not isinstance(discretization, PreparedPointCloudDiscretization) or not (
            isinstance(boundary, PointBoundaryPlan)
        ):
            raise TypeError("A prepared point cloud and PointBoundaryPlan are required.")
        names = tuple(canonical_identifier(name, "component") for name in components)
        if len(names) < 1 or len(set(names)) != len(names):
            raise ValueError("components must be unique identifiers.")
        form_ = parse(form, PointDiffusionForm, "form")
        stability_ = parse(stability, PointStabilityPolicy, "stability")
        count = discretization.state_shape[0]
        if boundary.row_count != count or boundary.components != len(names):
            raise ValueError("Boundary plan must declare every component on every point.")
        declared = np.asarray(discretization.plan.boundary_mask)
        family = discretization_family(discretization)
        layouts: list[_RowLayout] = []
        for component, name in enumerate(names):
            if not np.array_equal(boundary.owned(component), declared):
                raise ValueError(
                    f"Component {name!r} must own exactly the declared boundary rows."
                )
            if any(c.side is not None for c in boundary.conditions):
                raise ValueError("Block systems use the cloud's single-sided support.")
            layout = _row_layout(
                boundary,
                (),
                ("cloud",),
                np.ones((count, 1), dtype=np.bool_),
                discretization.spatial_dimension,
                row_points=np.asarray(discretization.plan.points),
                address=discretization.plan.address,
                component=component,
            )
            _, gauges = _components_and_gauges(
                (family,), layout, count, "square-collocation", None
            )
            if gauges:
                raise ValueError(
                    f"Component {name!r} floats on a connected component; declare Dirichlet or Robin rows."
                )
            layouts.append(layout)
        if ghosts is not None:
            _validate_ghosts(ghosts, discretization, tuple(layouts), form_)
        unknowns = count if ghosts is None else ghosts.row_count
        solve_space = ArraySpace(
            (unknowns * len(names),),
            dtype=jnp.float64,
            space_id=canonical_fingerprint(
                {
                    "kind": "point-block-solve-space",
                    "discretization": discretization.prepared_id,
                    "components": names,
                    "ghosts": None if ghosts is None else ghosts.prepared_id,
                }
            ),
        )
        hierarchy: PreparedMeshfreeHierarchy | None = None
        tolerance = TolerancePolicy(relative=1e-10, absolute=1e-11, max_steps=2000)
        if linear_policy is not None and (
            multigrid is not None or coarsening is not None or auxiliary is not None
        ):
            raise ValueError(
                "auxiliary, multigrid and coarsening configure only the default linear policy."
            )
        if multigrid is None and coarsening is not None:
            raise ValueError("coarsening requires a multigrid declaration.")
        if multigrid is not None and auxiliary is not None:
            raise ValueError("Choose either the multigrid or the auxiliary route.")
        if auxiliary is not None:
            _validate_auxiliary_route(layouts, discretization.spatial_dimension)
        # A block row couples every component's stencil: N c rows of width
        # c w bound each product by N c (c w)^2 = (N c^3) w^2 entries, the
        # scalar capacity law at N c^3 rows (recorded through plan_id).
        assembly = (
            _capacity_assembly_policy(count * len(names) ** 3, (family,))
            if assembly_policy is None
            else assembly_policy
        )
        if not isinstance(assembly, SparseAssemblyPolicy):
            raise TypeError("assembly_policy must be SparseAssemblyPolicy.")
        if linear_policy is None and auxiliary is not None:
            # The auxiliary hierarchy depends on the coefficients; it is
            # prepared with them (PreparedPointBlockSystem).
            policy = _block_solve_policy(
                solve_space.size, tolerance, assembly, None, hierarchy=True
            )
        elif linear_policy is None and multigrid is not None:
            # Dirichlet rows are decoupled identity equations of the solve
            # operator (values are lifted into the right-hand side); in the
            # component-major block layout they are eliminated coordinates.
            eliminated = _eliminated_coordinates(tuple(layouts), ghosts)
            hierarchy = MeshfreeHierarchyPlan(
                discretization.points if ghosts is None else ghosts.points,
                eliminated=eliminated,
                stable_ids=discretization.stable_ids
                if ghosts is None
                else ghosts.point_ids,
                components=MeshfreeComponentSpace(len(names), layout="block"),
                near_nullspace=multigrid,
                policy=coarsening,
            ).prepare(solve_space)
            # One declared assembly policy budgets the fine assembly and the
            # Galerkin coarse products of the hierarchy.
            policy = _block_solve_policy(
                solve_space.size,
                tolerance,
                assembly,
                PreconditioningPolicy(
                    meshfree_multigrid_builder(hierarchy, assembly=assembly)
                ),
                hierarchy=True,
            )
        elif linear_policy is None:
            policy = _block_solve_policy(
                solve_space.size,
                tolerance,
                assembly,
                PreconditioningPolicy(ILUPreconditionerBuilder(), refresh="numeric"),
                hierarchy=False,
            )
        elif not isinstance(linear_policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be a native LinearSolvePolicy.")
        else:
            policy = linear_policy
        self.discretization = discretization
        self.boundary = boundary
        self.layouts = tuple(layouts)
        self.linear_policy = policy
        self.assembly_policy = assembly
        self.solve_space = solve_space
        self.hierarchy = hierarchy
        self.auxiliary = auxiliary
        self.ghosts = ghosts
        # A coupled component-major row spans every component's stencil.
        assessment = (
            collocation_stability_assessment(
                solve_space.size,
                len(names)
                * (family if ghosts is None else ghosts.family).relation.route_shape[-1],
                discretization.spatial_dimension,
                # The auxiliary route's policy is completed per coefficient
                # field; it always carries a preconditioner.
                preconditioned=auxiliary is not None
                or policy.preconditioning is not None,
                # The reused preconditioner is charged against the solve's
                # budgets; the auxiliary route completes its policy per
                # coefficient field with the hierarchy budgets.
                resources=(
                    _block_solve_resources(solve_space.size, assembly, hierarchy=True)
                    if auxiliary is not None
                    else policy.resources
                ),
                # The default solve policies budget Krylov memory for their
                # restart (_block_solve_resources); the transform reuses it.
                restart=min(200, solve_space.size) if linear_policy is None else 40,
            )
            if stability_assessment is None
            else stability_assessment
        )
        self.stability_assessment = assessment
        self.components = names
        self.form = form_
        self.stability = stability_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "point-block-system-plan",
                "discretization": discretization.prepared_id,
                "boundary": boundary.plan_id,
                "components": names,
                "form": form_,
                "auxiliary": None if auxiliary is None else auxiliary.kind,
                "ghosts": None if ghosts is None else ghosts.prepared_id,
                "stability": stability_,
                "stability_assessment": _module_identity(assessment),
                "linear_policy": _module_identity(policy),
                "assembly_policy": _module_identity(assembly),
                "precision": np.dtype(np.float64).name,
            }
        )

    @property
    def space(self) -> BlockSpace:
        d = self.discretization
        return BlockSpace(
            tuple(
                ArraySpace(
                    d.state_shape,
                    dtype=jnp.float64,
                    space_id=f"{self.plan_id}:component:{name}",
                )
                for name in self.components
            ),
            names=self.components,
            space_id=f"{self.plan_id}:block-space",
        )

    @property
    def equation_space(self) -> BlockSpace:
        """Block space of the solved equations: cloud values, then ghost values."""
        if self.ghosts is None:
            return self.space
        return BlockSpace(
            tuple(
                ArraySpace(
                    (self.ghosts.row_count,),
                    dtype=jnp.float64,
                    space_id=f"{self.plan_id}:ghost-component:{name}",
                )
                for name in self.components
            ),
            names=self.components,
            space_id=f"{self.plan_id}:ghost-block-space",
        )

    def prepare(self, coefficients: ArrayLike, /) -> PreparedPointBlockSystem:
        return PreparedPointBlockSystem(self, coefficients)

    def physical_operator(
        self, coefficients: ArrayLike, /
    ) -> tuple[BlockLinearOperator, PointCouplingEvidence]:
        """Verified coefficients' physical blocks for composition into mixed systems.

        Rows are bulk divergence (or dissipative pairing), conormal flux, and
        identity Dirichlet/periodic rows; Dirichlet columns are not eliminated.
        """
        d = self.discretization
        coefficient, evidence = _coupling(
            coefficients, d.state_shape[0], len(self.components), d.spatial_dimension
        )
        physical, _ = _system_operators(self, coefficient)
        return physical, evidence

    def flat_operator(
        self, assembled: AbstractLinearOperator, /
    ) -> SparseCoordinateOperator:
        """An assembled block-space operator on component-major ``solve_space``.

        The block space flattens component by component, so the assembled
        relation is already in the ``"block"`` layout of the multigrid
        hierarchy; only the coordinate space changes, never the entries.
        """
        if not isinstance(assembled, SparseCoordinateOperator) or not (
            assembled.source.compatible(self.equation_space)
            and assembled.target.compatible(self.equation_space)
        ):
            raise TypeError(
                "flat_operator needs this plan's assembled equation operator."
            )
        return SparseCoordinateOperator(
            assembled.relation,
            assembled.coefficients,
            source=self.solve_space,
            target=self.solve_space,
            properties=assembled.properties,
            operator_id=f"{assembled.operator_id}:flat",
        )


def _validate_ghosts(
    ghosts: PreparedPointGhostLayer,
    discretization: PreparedPointCloudDiscretization,
    layouts: tuple[_RowLayout, ...],
    form: PointDiffusionForm,
    /,
) -> None:
    """The ghost layer must extend exactly the flux rows of these components."""
    if not isinstance(ghosts, PreparedPointGhostLayer):
        raise TypeError("ghosts must be PreparedPointGhostLayer.")
    if ghosts.discretization_id != discretization.prepared_id:
        raise ValueError("The ghost layer was prepared for another cloud.")
    if form != "collocated":
        raise ValueError("Boundary ghosts extend collocated block equations only.")
    rows = np.asarray(ghosts.plan.rows)
    direction = np.asarray(ghosts.plan.normals)
    flux = np.zeros(discretization.state_shape[0], dtype=np.bool_)
    for layout in layouts:
        if np.any(np.asarray(layout.replica) | np.asarray(layout.partner)):
            raise ValueError("Boundary ghosts do not serve periodic rows.")
        owned = np.asarray(layout.flux)
        flux |= owned
        # Each component's condition uses its own declared normal; the shared
        # ghost must lie outside along every one of them.
        normals = np.asarray(layout.flux_normals[0])[rows]
        if np.any(np.sum(normals * direction, axis=1)[owned[rows]] <= 0.0):
            raise ValueError(
                "Each ghost must lie outward of every component's flux normal."
            )
    if not np.array_equal(np.sort(rows), np.flatnonzero(flux)):
        raise ValueError(
            "The ghost layer must extend exactly the rows that any component owns with a flux condition."
        )


def _eliminated_coordinates(
    layouts: tuple[_RowLayout, ...], ghosts: PreparedPointGhostLayer | None, /
) -> np.ndarray:
    """Component-major Dirichlet identity coordinates of the solved equations."""
    padding = np.zeros(0 if ghosts is None else ghosts.ghost_count, dtype=np.bool_)
    return np.concatenate(
        [np.concatenate((np.asarray(layout.dirichlet), padding)) for layout in layouts]
    )


def _equation_bulk(
    layouts: tuple[_RowLayout, ...], ghosts: PreparedPointGhostLayer | None, /
) -> np.ndarray:
    """Component-major PDE rows of the solved equations.

    On the ghost route every non-Dirichlet cloud row (flux boundary points
    included) is a PDE row and the ghost condition rows are not.
    """
    if ghosts is None:
        return np.concatenate([np.asarray(layout.bulk) for layout in layouts])
    padding = np.zeros(ghosts.ghost_count, dtype=np.bool_)
    return np.concatenate(
        [np.concatenate((~np.asarray(layout.dirichlet), padding)) for layout in layouts]
    )


def _ghost_block(
    plan: PointBlockSystemPlan,
    ghosts: PreparedPointGhostLayer,
    cloud: PointDerivativeFamily,
    coefficient: Array,
    row: int,
    column: int,
    /,
) -> AbstractLinearOperator:
    """Ghost-extended block ``(row, column)`` over cloud-then-ghost unknowns.

    Non-Dirichlet cloud row ``i`` collocates the PDE of component ``row``
    with ghost-extended stencils (coefficient gradients from the cloud's own
    stencils), including flux boundary points. Ghost row ``N + g`` at boundary
    row ``b`` carries component ``row``'s own condition there: its conormal
    ``sum_ij n_i C_ij D_j u`` (plus Robin ``k u_b``) divided by the ghost offset
    ``δ_b`` when the component is flux-owned at ``b``, else its PDE at ``x_b``.
    """
    space = plan.equation_space
    source, target = space.spaces[column], space.spaces[row]
    family = ghosts.family
    dimension = plan.discretization.spatial_dimension
    count, total = ghosts.cloud_count, ghosts.row_count
    rows = np.asarray(ghosts.plan.rows)
    ghost_rows = count + np.arange(ghosts.ghost_count)
    layout = plan.layouts[row]
    flux = np.asarray(layout.flux)[rows]
    dirichlet = np.asarray(layout.dirichlet)
    if np.any(~flux & ~dirichlet[rows]):
        raise RuntimeError("Every ghost row is flux- or Dirichlet-owned per component.")
    padding = jnp.zeros((ghosts.ghost_count,), dtype=jnp.float64)
    # Component ``row``'s condition uses its own declared normal (at a corner
    # it differs from the shared ghost direction).
    normals = (
        jnp.zeros((total, dimension), dtype=jnp.float64)
        .at[rows]
        .set(layout.flux_normals[0][rows])
    )
    pde = jnp.zeros_like(family.weights[0])
    conormal = jnp.zeros_like(family.weights[0])
    for i in range(dimension):
        for j in range(dimension):
            c = coefficient[:, row, column, i, j]
            k = jnp.concatenate((c, padding))
            gradient = jnp.concatenate((cloud.apply(c, _unit(dimension, i)), padding))
            first = family.weights_for(_unit(dimension, j))
            pde = (
                pde
                - k[:, None] * family.weights_for(_unit(dimension, i, j))
                - gradient[:, None] * first
            )
            conormal = conormal + (normals[:, i] * k)[:, None] * first
    bulk = jnp.asarray(
        np.concatenate((~dirichlet, np.zeros(ghosts.ghost_count, dtype=np.bool_)))
    )
    offsets = jnp.asarray(ghosts.offsets)
    condition = jnp.where(
        jnp.asarray(flux)[:, None], conormal[rows] / offsets[:, None], pde[rows]
    )
    routed = jnp.zeros_like(pde).at[rows].set(condition)
    relation, image = _image_relation(family.relation, routed, ghost_rows, rows)
    operator: AbstractLinearOperator = SparseCoordinateOperator(
        family.relation,
        jnp.where(bulk[:, None], pde, 0.0),
        source=source,
        target=target,
        operator_id=f"{plan.plan_id}:ghost-block:{row}:{column}",
    ) + SparseCoordinateOperator(
        relation,
        image,
        source=source,
        target=target,
        operator_id=f"{plan.plan_id}:ghost-condition:{row}:{column}",
    )
    if row == column:
        robin = np.asarray(layout.robin)[rows] * flux
        if np.any(robin > 0.0):
            coupling = (
                jnp.zeros((total, 1), dtype=jnp.float64)
                .at[ghost_rows, 0]
                .set(jnp.asarray(robin) / offsets)
            )
            operator = operator + SparseCoordinateOperator(
                _selection(total, total, ghost_rows, rows),
                coupling,
                source=source,
                target=target,
                operator_id=f"{plan.plan_id}:ghost-robin:{row}",
            )
        operator = operator + DiagonalLinearOperator(
            jnp.asarray(
                np.concatenate((dirichlet, np.zeros(ghosts.ghost_count))).astype(
                    np.float64
                )
            ),
            space=target,
        )
    return operator


def _equation_operators(
    plan: PointBlockSystemPlan, coefficient: Array, /
) -> tuple[BlockLinearOperator, BlockLinearOperator]:
    """Physical and Dirichlet-eliminated operators of the solved equations."""
    ghosts = plan.ghosts
    if ghosts is None:
        return _system_operators(plan, coefficient)
    cloud = discretization_family(plan.discretization)
    size = len(plan.components)
    space = plan.equation_space
    padding = np.zeros(ghosts.ghost_count, dtype=np.bool_)
    physical = tuple(
        tuple(
            _ghost_block(plan, ghosts, cloud, coefficient, row, column)
            for column in range(size)
        )
        for row in range(size)
    )
    eliminated: list[tuple[AbstractLinearOperator, ...]] = []
    for row in range(size):
        blocks: list[AbstractLinearOperator] = []
        for column in range(size):
            dirichlet = np.concatenate(
                (np.asarray(plan.layouts[column].dirichlet), padding)
            )
            block = physical[row][column]
            if np.any(dirichlet):
                block = block @ DiagonalLinearOperator(
                    jnp.asarray((~dirichlet).astype(np.float64)),
                    space=space.spaces[column],
                )
            row_dirichlet = np.concatenate(
                (np.asarray(plan.layouts[row].dirichlet), padding)
            )
            if row == column and np.any(row_dirichlet):
                block = block + DiagonalLinearOperator(
                    jnp.asarray(row_dirichlet.astype(np.float64)), space=space.spaces[row]
                )
            blocks.append(block)
        eliminated.append(tuple(blocks))
    return (
        BlockLinearOperator(physical, source=space, target=space),
        BlockLinearOperator(tuple(eliminated), source=space, target=space),
    )


def _validate_auxiliary_route(layouts: Sequence[_RowLayout], dimension: int, /) -> None:
    if dimension not in (2, 3):
        raise ValueError("The auxiliary route triangulates 2-D or 3-D clouds only.")
    for component, layout in enumerate(layouts):
        if np.any(np.asarray(layout.replica)):
            raise ValueError(
                "The auxiliary route has no periodic identification; select the multigrid route."
            )
        if not np.any(np.asarray(layout.dirichlet)):
            raise ValueError(
                f"The auxiliary route needs Dirichlet rows in component {component} "
                "to anchor its auxiliary stiffness; select the multigrid route."
            )


def _block_solve_policy(
    size: int,
    tolerance: TolerancePolicy,
    assembly: SparseAssemblyPolicy,
    preconditioning: PreconditioningPolicy | None,
    /,
    *,
    hierarchy: bool,
) -> LinearSolvePolicy:
    """Default right-preconditioned GMRES with capacity-declared budgets."""
    return LinearSolvePolicy(
        GMRES(restart=min(200, size)),
        tolerance=tolerance,
        preconditioning=preconditioning,
        failure=FailurePolicy("status"),
        resources=_block_solve_resources(size, assembly, hierarchy=hierarchy),
    )


def _block_solve_resources(
    size: int, assembly: SparseAssemblyPolicy, /, *, hierarchy: bool
) -> SolveResourcePolicy:
    """Budgets of the default GMRES, declared from capacity (part of plan_id)."""
    restart = min(200, size)
    defaults = SolveResourcePolicy()
    # Right-preconditioned GMRES stores 2 restart + 1 coordinate vectors; the
    # Krylov budget is declared from that capacity, so it is part of the
    # policy identity (plan_id) rather than inflated after a refusal.
    krylov = (2 * restart + 2) * size * np.dtype(np.float64).itemsize
    # A hierarchy keeps its coarse operators and product recipes, bounded by
    # the same declared assembly workspace as the fine assembly.
    stored = assembly.max_workspace_bytes if hierarchy else 0
    return SolveResourcePolicy(
        workspace_bytes=max(defaults.workspace_bytes, stored),
        krylov_basis_bytes=max(defaults.krylov_basis_bytes, krylov),
        preconditioner_bytes=max(defaults.preconditioner_bytes, stored),
    )


def _auxiliary_solve_policy(
    plan: PointBlockSystemPlan, coefficient: Array, /
) -> tuple[LinearSolvePolicy, MeshfreeAuxiliaryStiffness | None]:
    """The plan's policy, completed by the coefficient-dependent auxiliary route."""
    if plan.auxiliary is None:
        return plan.linear_policy, None
    d = plan.discretization
    ghosts = plan.ghosts
    eliminated = _eliminated_coordinates(plan.layouts, ghosts)
    # Trace rows are the collocated flux rows; on the ghost route they are
    # the ghost condition rows (flux boundary points carry PDE rows). The
    # auxiliary triangulation then includes the ghost points, whose
    # coefficient is that of their boundary row.
    trace = ~eliminated & ~_equation_bulk(plan.layouts, ghosts)
    points = d.points if ghosts is None else ghosts.points
    if ghosts is not None:
        coefficient = jnp.concatenate(
            (coefficient, coefficient[jnp.asarray(ghosts.plan.rows)]), axis=0
        )
    stiffness = meshfree_auxiliary_stiffness(
        points,
        coefficient,
        plan.solve_space,
        eliminated=eliminated,
        assembly=plan.assembly_policy,
    )
    builder = meshfree_auxiliary_builder(
        stiffness,
        trace=trace,
        # Dissipative bulk rows are already quadrature-weighted.
        row_measure=d.quadrature_weights if plan.form == "dissipative" else None,
        near_nullspace=plan.auxiliary,
        assembly=plan.assembly_policy,
    )
    return (
        _block_solve_policy(
            plan.solve_space.size,
            plan.linear_policy.tolerance,
            plan.assembly_policy,
            PreconditioningPolicy(builder),
            hierarchy=True,
        ),
        stiffness,
    )


def _assess(
    plan: PointBlockSystemPlan,
    physical: PreparedSparseAssembly,
    solved: PreparedSparseAssembly,
    linear_solve: PreparedLinearSolve,
    prior: PreparedCollocationAssessment | None,
    /,
) -> tuple[PreparedCollocationAssessment | None, PointCollocationStability]:
    """Native spectral admission of the flat solve operator (refuses when required).

    An unpreconditioned iterative shift-invert transform reuses the solve's
    own prepared preconditioner; a declared one is used as declared.
    """
    transform_solve = plan.stability_assessment.transform_solve
    state = linear_solve.preconditioning_state
    reused = (
        state.action
        if state is not None
        and isinstance(transform_solve, LinearSolvePolicy)
        and transform_solve.preconditioning is None
        else None
    )
    return assess_square_collocation(
        plan.flat_operator(physical.operator),
        plan.flat_operator(solved.operator),
        space=plan.solve_space,
        eliminated=_eliminated_coordinates(plan.layouts, plan.ghosts),
        bulk=_equation_bulk(plan.layouts, plan.ghosts),
        policy=plan.stability,
        assessment=plan.stability_assessment,
        assembly_policy=plan.assembly_policy,
        problem_id=f"{plan.plan_id}:stability",
        prior=prior,
        transform_preconditioner=reused,
    )


def _coupling(
    coefficients: ArrayLike, count: int, components: int, dimension: int, /
) -> tuple[Array, PointCouplingEvidence]:
    shape = (components, components, dimension, dimension)
    value = jnp.asarray(coefficients, dtype=jnp.float64)
    if value.shape not in (shape, (count, *shape)):
        raise ValueError(
            "Coupling coefficients must have shape (m, m, d, d) or (points, m, m, d, d)."
        )
    value = jnp.broadcast_to(value, (count, *shape))
    flattened = jnp.transpose(value, (0, 1, 3, 2, 4)).reshape(
        (count, components * dimension, components * dimension)
    )
    evidence = verify_dense_properties(
        flattened,
        policy=DensePropertyVerificationPolicy(require_positive_semidefinite=True),
    )
    result = PointCouplingEvidence(
        minimum_eigenvalue=jnp.min(evidence.eigenvalues),
        maximum_eigenvalue=jnp.max(evidence.eigenvalues),
        symmetry_defect=jnp.max(evidence.hermitian_defect),
        finite=jnp.all(evidence.finite),
        positive_semidefinite=jnp.all(evidence.positive_semidefinite),
    )
    value = eqx.error_if(
        value,
        ~result.successful,
        "Coupling coefficients must be finite, major-symmetric, and positive semidefinite.",
    )
    return value, result


def _block(
    plan: PointBlockSystemPlan,
    family: PointDerivativeFamily,
    coefficient: Array,
    row: int,
    column: int,
    /,
) -> AbstractLinearOperator:
    """Physical block ``(row, column)``: bulk divergence, conormal, and diagonal rows."""
    d = plan.discretization
    space = plan.space
    source, target = space.spaces[column], space.spaces[row]
    if not isinstance(source, ArraySpace) or not isinstance(target, ArraySpace):
        raise TypeError("Point system component spaces must be ArraySpace instances.")
    dimension = d.spatial_dimension
    layout = plan.layouts[row]
    bulk = layout.bulk.astype(jnp.float64)
    normals = layout.flux_normals[0]
    image_normals = layout.image_normals[0]
    conormal = jnp.zeros_like(family.weights[0])
    image = jnp.zeros_like(family.weights[0])
    bulk_weights = jnp.zeros_like(family.weights[0])
    for i in range(dimension):
        for j in range(dimension):
            c = coefficient[:, row, column, i, j]
            first = family.weights_for(_unit(dimension, j))
            conormal = conormal + (normals[:, i] * c)[:, None] * first
            image = image + (image_normals[:, i] * c)[:, None] * first
            if plan.form == "collocated":
                gradient = family.apply(c, _unit(dimension, i))
                bulk_weights = (
                    bulk_weights
                    - (bulk * c)[:, None] * family.weights_for(_unit(dimension, i, j))
                    - (bulk * gradient)[:, None] * first
                )
    operator: AbstractLinearOperator = SparseCoordinateOperator(
        family.relation,
        bulk_weights + conormal,
        source=source,
        target=target,
        operator_id=f"{plan.plan_id}:block:{row}:{column}",
    )
    if plan.form == "dissipative":
        # Full quadrature pairing; non-bulk rows are replaced by their conditions.
        rows = DiagonalLinearOperator(bulk, space=target)
        for i in range(dimension):
            for j in range(dimension):
                operator = operator + rows @ transpose(
                    family.operator(_unit(dimension, i), source=target, target=target)
                ) @ family.operator(
                    _unit(dimension, j),
                    source=source,
                    target=target,
                    coefficients=d.quadrature_weights * coefficient[:, row, column, i, j],
                )
    replica = np.flatnonzero(np.asarray(layout.replica))
    if replica.size:
        images = np.asarray(layout.replica_images)[replica]
        relation, weights = _image_relation(family.relation, image, images, replica)
        operator = operator + SparseCoordinateOperator(
            relation, weights, source=source, target=target
        )
    if row == column:
        count = d.state_shape[0]
        diagonal = (
            layout.dirichlet.astype(jnp.float64)
            + layout.robin
            + layout.replica.astype(jnp.float64)
        )
        operator = operator + DiagonalLinearOperator(diagonal, space=target)
        if replica.size:
            operator = operator - SparseCoordinateOperator(
                _selection(
                    count, count, replica, np.asarray(layout.replica_images)[replica]
                ),
                jnp.ones((count, 1), dtype=jnp.float64),
                source=source,
                target=target,
            )
    return operator


def _system_operators(
    plan: PointBlockSystemPlan, coefficient: Array, /
) -> tuple[BlockLinearOperator, BlockLinearOperator]:
    family = discretization_family(plan.discretization)
    space = plan.space
    size = len(plan.components)
    physical = tuple(
        tuple(_block(plan, family, coefficient, row, column) for column in range(size))
        for row in range(size)
    )
    eliminated: list[tuple[AbstractLinearOperator, ...]] = []
    for row in range(size):
        blocks: list[AbstractLinearOperator] = []
        for column in range(size):
            block = physical[row][column]
            free = ~plan.layouts[column].dirichlet
            if not bool(np.all(np.asarray(free))):
                block = block @ DiagonalLinearOperator(
                    free.astype(jnp.float64), space=space.spaces[column]
                )
            if row == column and not bool(np.all(np.asarray(free))):
                block = block + DiagonalLinearOperator(
                    plan.layouts[row].dirichlet.astype(jnp.float64),
                    space=space.spaces[row],
                )
            blocks.append(block)
        eliminated.append(tuple(blocks))
    return (
        BlockLinearOperator(physical, source=space, target=space),
        BlockLinearOperator(tuple(eliminated), source=space, target=space),
    )


@final
class PreparedPointBlockSystem(StrictModule, NonTrainableState):
    """Reusable block sparse assembly and native solve for one coefficient field.

    ``physical_assembly`` holds the square physical rows on cloud values
    (``apply``/``physical_rhs``). On the ghost route ``equation_assembly``
    holds the solved ghost-extended physical equations and ``assembly`` their
    Dirichlet elimination; otherwise the square rows are the solved equations.
    """

    plan: PointBlockSystemPlan
    coefficients: Array
    evidence: PointCouplingEvidence
    physical_assembly: PreparedSparseAssembly
    equation_assembly: PreparedSparseAssembly | None
    assembly: PreparedSparseAssembly
    linear_solve: PreparedLinearSolve
    auxiliary_stiffness: MeshfreeAuxiliaryStiffness | None
    stability: PointCollocationStability
    stability_state: PreparedCollocationAssessment | None

    def __init__(self, plan: PointBlockSystemPlan, coefficients: ArrayLike, /) -> None:
        if not isinstance(plan, PointBlockSystemPlan):
            raise TypeError("plan must be PointBlockSystemPlan.")
        d = plan.discretization
        coefficient, evidence = _coupling(
            coefficients, d.state_shape[0], len(plan.components), d.spatial_dimension
        )
        physical, square_eliminated = _system_operators(plan, coefficient)
        physical_assembly = prepare_sparse_assembly(
            plan_sparse_assembly(physical, plan.assembly_policy), physical
        )
        equation_assembly = None
        if plan.ghosts is None:
            eliminated = square_eliminated
        else:
            equations, eliminated = _equation_operators(plan, coefficient)
            equation_assembly = prepare_sparse_assembly(
                plan_sparse_assembly(equations, plan.assembly_policy), equations
            )
        assembly = prepare_sparse_assembly(
            plan_sparse_assembly(eliminated, plan.assembly_policy), eliminated
        )
        self.plan = plan
        self.coefficients = coefficient
        self.evidence = evidence
        self.physical_assembly = physical_assembly
        self.equation_assembly = equation_assembly
        self.assembly = assembly
        policy, stiffness = _auxiliary_solve_policy(plan, coefficient)
        self.auxiliary_stiffness = stiffness
        self.linear_solve = prepare(_flat_system(plan, assembly), policy)
        self.stability_state, self.stability = _assess(
            plan,
            physical_assembly if equation_assembly is None else equation_assembly,
            assembly,
            self.linear_solve,
            None,
        )

    def refresh(self, coefficients: ArrayLike, /) -> PreparedPointBlockSystem:
        d = self.plan.discretization
        coefficient, evidence = _coupling(
            coefficients, d.state_shape[0], len(self.plan.components), d.spatial_dimension
        )
        physical, square_eliminated = _system_operators(self.plan, coefficient)
        physical_assembly = refresh_sparse_assembly(self.physical_assembly, physical)
        equation_assembly = None
        if self.equation_assembly is None:
            eliminated = square_eliminated
        else:
            equations, eliminated = _equation_operators(self.plan, coefficient)
            equation_assembly = refresh_sparse_assembly(self.equation_assembly, equations)
        assembly = refresh_sparse_assembly(self.assembly, eliminated)
        if self.plan.auxiliary is None:
            linear_solve = refresh(self.linear_solve, _flat_system(self.plan, assembly))
            stiffness = None
        else:
            # The auxiliary stiffness carries the coefficients: a declared
            # rebuild of the auxiliary hierarchy, not a numeric refresh.
            policy, stiffness = _auxiliary_solve_policy(self.plan, coefficient)
            linear_solve = prepare(_flat_system(self.plan, assembly), policy)
        stability_state, stability = _assess(
            self.plan,
            physical_assembly if equation_assembly is None else equation_assembly,
            assembly,
            linear_solve,
            self.stability_state,
        )
        return eqx.tree_at(
            lambda p: (
                p.coefficients,
                p.evidence,
                p.physical_assembly,
                p.equation_assembly,
                p.assembly,
                p.linear_solve,
                p.auxiliary_stiffness,
                p.stability,
                p.stability_state,
            ),
            self,
            (
                coefficient,
                evidence,
                physical_assembly,
                equation_assembly,
                assembly,
                linear_solve,
                stiffness,
                stability,
                stability_state,
            ),
            is_leaf=lambda value: value is None,
        )

    def apply(self, values: ArrayLike, /) -> Array:
        """Physical block action on ``(points, components)`` values."""
        return _stack(self.physical_assembly.operator.mv(_blocks(values, self.plan)))

    def physical_rhs(
        self,
        source: ArrayLike,
        /,
        *,
        boundary_values: Mapping[str, ArrayLike] | None = None,
    ) -> Array:
        plan = self.plan
        d = plan.discretization
        source_ = jnp.asarray(source, dtype=jnp.float64)
        if source_.shape != (d.state_shape[0], len(plan.components)):
            raise ValueError("Block source must have shape (points, components).")
        source_ = eqx.error_if(
            source_, jnp.any(~jnp.isfinite(source_)), "Block source must be finite."
        )
        scale = (
            d.quadrature_weights[:, None]
            if plan.form == "dissipative"
            else jnp.ones_like(source_)
        )
        columns = []
        for component, layout in enumerate(plan.layouts):
            values = plan.boundary.row_values(component, overrides=boundary_values)
            columns.append(
                jnp.where(layout.bulk, (scale * source_)[:, component], values)
            )
        return jnp.stack(columns, axis=1)

    def equation_rhs(
        self,
        source: ArrayLike,
        /,
        *,
        boundary_values: Mapping[str, ArrayLike] | None = None,
    ) -> Array:
        """Right-hand side of the solved equations, ``(unknowns, components)``.

        On the ghost route, cloud rows carry the source (or Dirichlet data) and
        ghost row ``g`` carries its component's condition data divided by
        ``δ_b`` (or the source at ``x_b`` where that component is Dirichlet).
        """
        rhs = self.physical_rhs(source, boundary_values=boundary_values)
        ghosts = self.plan.ghosts
        if ghosts is None:
            return rhs
        source_ = jnp.asarray(source, dtype=jnp.float64)
        rows = np.asarray(ghosts.plan.rows)
        columns = []
        for component, layout in enumerate(self.plan.layouts):
            values = self.plan.boundary.row_values(component, overrides=boundary_values)
            cloud = jnp.where(layout.dirichlet, values, source_[:, component])
            ghost = jnp.where(
                layout.flux[rows], values[rows] / ghosts.offsets, source_[rows, component]
            )
            columns.append(jnp.concatenate((cloud, ghost)))
        return jnp.stack(columns, axis=1)

    def solve(
        self,
        source: ArrayLike,
        /,
        *,
        boundary_values: Mapping[str, ArrayLike] | None = None,
    ) -> PointBlockSystemResult:
        plan = self.plan
        ghosts = plan.ghosts
        operator = (
            self.physical_assembly
            if self.equation_assembly is None
            else self.equation_assembly
        ).operator
        rhs = self.equation_rhs(source, boundary_values=boundary_values)
        size = len(plan.components)
        dirichlet = _eliminated_coordinates(plan.layouts, ghosts).reshape((size, -1)).T
        lift = jnp.where(dirichlet, rhs, 0.0)
        eliminated = jnp.where(dirichlet, rhs, rhs - _stack(operator.mv(_columns(lift))))
        tolerance = plan.linear_policy.tolerance
        threshold = tolerance.absolute + tolerance.relative * jnp.linalg.norm(rhs)
        # Lifting can amplify the algebraic RHS. Stop against the requested
        # original-equation tolerance, not the artificially enlarged norm.
        control = (
            LinearSolveControl(relative_tolerance=0.0, absolute_tolerance=threshold)
            if self.linear_solve.plan.backend == "native-krylov"
            else None
        )
        linear_result = solve(
            self.linear_solve, eliminated.T.reshape((-1,)), control=control
        )
        # Dirichlet coordinates are decoupled identity equations of the
        # eliminated system, exactly solved by their data. A Krylov iterate
        # carries rounding there that the eliminated residual weighs by one,
        # but the physical bulk rows weigh it by their (unscaled) Dirichlet
        # columns; take the exact solution of those decoupled equations.
        unknowns = jnp.where(dirichlet, rhs, linear_result.value.reshape((size, -1)).T)
        residual = _stack(operator.mv(_columns(unknowns))) - rhs
        norm = jnp.linalg.norm(residual)
        bulk = _equation_bulk(plan.layouts, ghosts).reshape((size, -1)).T
        count = plan.discretization.state_shape[0]
        values = unknowns[:count]
        # "error" refuses by raising; "status" publishes successful=False.
        if plan.linear_policy.failure.mode == "error":
            values = eqx.error_if(
                values,
                ~jnp.isfinite(norm) | (norm > threshold),
                "Point block system refused: unresolved physical or boundary equations.",
            )
        ghost_values = None if ghosts is None else unknowns[count:]
        defect = (
            None
            if ghosts is None
            else jnp.max(
                jnp.stack(
                    [
                        ghosts.extension_defect(unknowns[:, component])
                        for component in range(size)
                    ]
                )
            )
        )
        return PointBlockSystemResult(
            values=values,
            residual_norm=norm,
            residual_tolerance=threshold,
            boundary_residual_norm=jnp.linalg.norm(jnp.where(bulk, 0.0, residual)),
            linear_result=linear_result,
            coupling_evidence=self.evidence,
            ghost_values=ghost_values,
            ghost_extension_defect=defect,
            components=plan.components,
        )

    def solve_nonlinear(
        self,
        reaction: Callable[[Array, Array], Array],
        source: ArrayLike,
        /,
        *,
        initial: ArrayLike | None = None,
        boundary_values: Mapping[str, ArrayLike] | None = None,
        termination: NonlinearTermination | None = None,
        method: NewtonKrylov | None = None,
    ) -> PointBlockNonlinearResult:
        """Solve ``A u + g(u, x) = f`` on bulk rows with native Newton–Krylov.

        ``reaction(values, points)`` maps ``(points, components)`` values to the
        pointwise nonlinear term. Jacobian actions come from the native prepared
        linearization (matrix-free); the selected ``method`` owns the step solve.
        """
        plan = self.plan
        if plan.ghosts is not None:
            raise ValueError(
                "Pointwise nonlinear reactions are solved on the square route; "
                "the ghost route solves linear block equations."
            )
        d = plan.discretization
        count, size = d.state_shape[0], len(plan.components)
        rhs = self.physical_rhs(source, boundary_values=boundary_values)
        bulk = jnp.stack([layout.bulk for layout in plan.layouts], axis=1)
        operator = self.physical_assembly.operator
        points = d.points

        def residual(state: Array, args: None) -> Array:
            del args
            values = state.reshape((size, count)).T
            pointwise = jnp.asarray(reaction(values, points))
            if pointwise.shape != values.shape:
                raise ValueError("reaction must return (points, components) values.")
            linear = _stack(operator.mv(_blocks(values, plan)))
            return (linear + jnp.where(bulk, pointwise, 0.0) - rhs).T.reshape((-1,))

        space = plan.solve_space
        problem = NonlinearSystemProblem(
            residual,
            state_space=space,
            residual_space=space,
            problem_id=f"{plan.plan_id}:nonlinear",
        )
        start = (
            jnp.zeros((count, size), dtype=jnp.float64)
            if initial is None
            else jnp.asarray(initial, dtype=jnp.float64)
        )
        if start.shape != (count, size):
            raise ValueError("initial must have shape (points, components).")
        newton = (
            NewtonKrylov(
                linear_policy=LinearSolvePolicy(
                    GMRES(restart=min(60, count * size)),
                    tolerance=TolerancePolicy(
                        relative=1e-10, absolute=1e-12, max_steps=2000
                    ),
                    failure=FailurePolicy("status"),
                )
            )
            if method is None
            else method
        )
        result = root(
            problem,
            start.T.reshape((-1,)),
            method=newton,
            termination=NonlinearTermination(
                absolute_residual=1e-9, relative_residual=1e-10, maximum_steps=50
            )
            if termination is None
            else termination,
        )
        values = result.state.reshape((size, count)).T
        return PointBlockNonlinearResult(
            values=values,
            residual_norm=jnp.linalg.norm(residual(result.state, None)),
            nonlinear_result=result,
            coupling_evidence=self.evidence,
            components=plan.components,
        )


def _flat_system(
    plan: PointBlockSystemPlan, assembly: PreparedSparseAssembly, /
) -> LinearSystem:
    return LinearSystem(plan.flat_operator(assembly.operator), problem_id=plan.plan_id)


def _blocks(values: ArrayLike, plan: PointBlockSystemPlan, /) -> tuple[Array, ...]:
    value = jnp.asarray(values, dtype=jnp.float64)
    if value.shape != (plan.discretization.state_shape[0], len(plan.components)):
        raise ValueError("Block values must have shape (points, components).")
    return tuple(value[:, component] for component in range(len(plan.components)))


def _columns(values: Array, /) -> tuple[Array, ...]:
    """Per-component columns of ``(rows, components)`` values."""
    return tuple(values[:, component] for component in range(values.shape[1]))


def _stack(blocks: object, /) -> Array:
    if not isinstance(blocks, (tuple, list)):
        raise TypeError("Block operators must return one array per component.")
    return jnp.stack([jnp.asarray(block) for block in blocks], axis=1)


__all__ = [
    "PointBlockNonlinearResult",
    "PointBlockSystemPlan",
    "PointBlockSystemResult",
    "PointCouplingEvidence",
    "PreparedPointBlockSystem",
    "isotropic_elasticity_coefficients",
]
