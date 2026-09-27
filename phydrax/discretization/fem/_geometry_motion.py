#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-topology mesh motion routes for affine vertex-coordinate meshes.

A :class:`FiniteElementMotionExtension` maps prescribed boundary vertex
displacements to interior displacements through one explicit
:class:`FiniteElementMeshMotionRoute`. Linear routes stay matrix-free on
element tensors (:class:`phydrax.sparse.ElementTensorOperator`) and solve once
per call through a prepared :mod:`phydrax.linalg` Krylov solve; nonlinear routes
solve their stationarity equations through :mod:`phydrax.nonlinear` with
implicit root derivatives, so every route is differentiable with respect to the
boundary displacement and, for MMPDE, the monitor parameters. Acceptance of the
moved coordinates is owned by :class:`MotionValidityPlan`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from enum import IntFlag, StrEnum
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import positive_finite_float
from ...linalg import (
    ConjugateGradient,
    determinant_small_linear,
    DifferentiationPolicy,
    FailurePolicy,
    GMRES,
    inverse_small_linear,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    prepare,
    PreparedLinearSolve,
    RHSLayout,
    SmallLinearSolvePlan,
    solve,
    solve_small_linear,
    TolerancePolicy,
)
from ...nonlinear import (
    implicit_root_result,
    ImplicitRootDerivativePolicy,
    NewtonKrylov,
    NonlinearSystemProblem,
    NonlinearTermination,
    PseudoTransient,
)
from ...sparse import ElementTensorOperator
from .._cell_geometry_validity import (
    CellValidityCertificate,
    CellValidityPolicy,
    certify_cell_geometry_validity,
)
from .._motion_validity import (
    corner_frame_matrices,
    motion_cell_dimension,
    motion_corner_routes,
    MotionValidityEvidence,
    MotionValidityPlan,
    MotionValidityPolicy,
    MotionValidityStatus,
)
from .._reference_cell import reference_cell_topology
from ._generic import (
    _degree_aware_reference_rule,
    FiniteElementDiscretization,
    FiniteElementRuntimeData,
)
from ._reference import lagrange_element


MeshMonitor = Callable[[Array], Array]
# Time-dependent monitors of ALE consumers: ``monitor(time, points, args)``.
TimeMeshMonitor = Callable[[Array, Array, Any], ArrayLike]

# Huang (2001) meshing functional: theta balances alignment against
# equidistribution and p > 1 strengthens equidistribution.
_MMPDE_THETA = 1.0 / 3.0
_MMPDE_EXPONENT = 1.5


class FiniteElementMeshMotionRoute(StrEnum):
    """Closed set of fixed-topology interior motion routes.

    ``HARMONIC``: graph-Laplacian extension with inverse-edge-length weights
    (valid on convex 2-D domains by Tutte's theorem). ``LINEAR_ELASTICITY``:
    finite-element linear elasticity on the reference mesh with Jacobian
    stiffening ``E = (J_max / J)**chi``. ``WINSLOW``: inverse harmonic map,
    the minimizer of ``sum m det(F) |F^-1|**2`` whose barrier keeps maps onto a
    convex reference domain valid. ``MMPDE``: steady state of the Huang–Russell
    moving-mesh PDE ``dx/dt = -(P / tau) dI_h/dx`` for a monitor ``M(x)``.
    ``PRESCRIBED``: every vertex follows the provider (direct map).
    """

    HARMONIC = "harmonic"
    LINEAR_ELASTICITY = "linear_elasticity"
    WINSLOW = "winslow"
    MMPDE = "mmpde"
    PRESCRIBED = "prescribed"


class FiniteElementMeshMotionStatus(IntFlag):
    """Runtime failures of fixed-topology finite-element mesh motion."""

    SUCCESS = 0
    BOUNDARY_REJECTED = 1
    EXTENSION_FAILED = 2
    NONFINITE_COORDINATES = 4
    EXCESSIVE_DISPLACEMENT = 8
    JACOBIAN_TOO_SMALL = 16
    ORIENTATION_CHANGED = 32


_GEOMETRY_STATUS = (
    (
        MotionValidityStatus.NONFINITE_COORDINATES,
        FiniteElementMeshMotionStatus.NONFINITE_COORDINATES,
    ),
    (
        MotionValidityStatus.EXCESSIVE_DISPLACEMENT,
        FiniteElementMeshMotionStatus.EXCESSIVE_DISPLACEMENT,
    ),
    (
        MotionValidityStatus.JACOBIAN_TOO_SMALL,
        FiniteElementMeshMotionStatus.JACOBIAN_TOO_SMALL,
    ),
    (
        MotionValidityStatus.ORIENTATION_CHANGED,
        FiniteElementMeshMotionStatus.ORIENTATION_CHANGED,
    ),
)


class FiniteElementMeshMotionPolicy(StrictModule, NonTrainableState):
    """Route selection, route controls, solve tolerances, and acceptance policy.

    ``stiffening_exponent`` (chi) and ``poisson_ratio`` configure
    ``LINEAR_ELASTICITY``; the nonlinear tolerances configure ``WINSLOW`` and
    ``MMPDE``; ``relaxation_time`` (tau) and ``mmpde_initial_time_step`` set the
    MMPDE time scale and the first pseudo-transient step.
    """

    route: FiniteElementMeshMotionRoute = eqx.field(static=True)
    validity: MotionValidityPolicy
    solve_relative_tolerance: float = eqx.field(static=True)
    solve_absolute_tolerance: float = eqx.field(static=True)
    maximum_solve_steps: int = eqx.field(static=True)
    stiffening_exponent: float = eqx.field(static=True)
    poisson_ratio: float = eqx.field(static=True)
    nonlinear_relative_tolerance: float = eqx.field(static=True)
    nonlinear_absolute_tolerance: float = eqx.field(static=True)
    maximum_nonlinear_steps: int = eqx.field(static=True)
    relaxation_time: float = eqx.field(static=True)
    mmpde_initial_time_step: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        route: FiniteElementMeshMotionRoute = FiniteElementMeshMotionRoute.HARMONIC,
        validity: MotionValidityPolicy | None = None,
        solve_relative_tolerance: float = 1.0e-10,
        solve_absolute_tolerance: float = 1.0e-12,
        maximum_solve_steps: int = 500,
        stiffening_exponent: float = 1.0,
        poisson_ratio: float = 0.3,
        nonlinear_relative_tolerance: float = 1.0e-8,
        nonlinear_absolute_tolerance: float = 1.0e-12,
        maximum_nonlinear_steps: int = 100,
        relaxation_time: float = 1.0e-2,
        mmpde_initial_time_step: float = 1.0e-3,
    ) -> None:
        if not isinstance(route, FiniteElementMeshMotionRoute):
            raise TypeError("route must be FiniteElementMeshMotionRoute.")
        validity_ = MotionValidityPolicy() if validity is None else validity
        if not isinstance(validity_, MotionValidityPolicy):
            raise TypeError("validity must be MotionValidityPolicy or None.")
        relative = positive_finite_float(
            solve_relative_tolerance, "solve_relative_tolerance"
        )
        absolute = positive_finite_float(
            solve_absolute_tolerance, "solve_absolute_tolerance"
        )
        nonlinear_relative = positive_finite_float(
            nonlinear_relative_tolerance, "nonlinear_relative_tolerance"
        )
        nonlinear_absolute = positive_finite_float(
            nonlinear_absolute_tolerance, "nonlinear_absolute_tolerance"
        )
        relaxation = positive_finite_float(relaxation_time, "relaxation_time")
        initial_step = positive_finite_float(
            mmpde_initial_time_step, "mmpde_initial_time_step"
        )
        chi = float(stiffening_exponent)
        if not math.isfinite(chi) or chi < 0.0:
            raise ValueError("stiffening_exponent must be finite and non-negative.")
        nu = float(poisson_ratio)
        if not math.isfinite(nu) or not 0.0 <= nu < 0.5:
            raise ValueError("poisson_ratio must lie in [0, 0.5).")
        for name, value in (
            ("maximum_solve_steps", maximum_solve_steps),
            ("maximum_nonlinear_steps", maximum_nonlinear_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        self.route = route
        self.validity = validity_
        self.solve_relative_tolerance = relative
        self.solve_absolute_tolerance = absolute
        self.maximum_solve_steps = int(maximum_solve_steps)
        self.stiffening_exponent = chi
        self.poisson_ratio = nu
        self.nonlinear_relative_tolerance = nonlinear_relative
        self.nonlinear_absolute_tolerance = nonlinear_absolute
        self.maximum_nonlinear_steps = int(maximum_nonlinear_steps)
        self.relaxation_time = relaxation
        self.mmpde_initial_time_step = initial_step
        self.policy_id = canonical_fingerprint(
            {
                "kind": "finite-element-mesh-motion-policy",
                "route": route.value,
                "validity": validity_.policy_id,
                "solve": [relative, absolute, int(maximum_solve_steps)],
                "elasticity": [chi, nu],
                "nonlinear": [
                    nonlinear_relative,
                    nonlinear_absolute,
                    int(maximum_nonlinear_steps),
                ],
                "mmpde": [relaxation, initial_step, _MMPDE_THETA, _MMPDE_EXPONENT],
            }
        )

    def linear_policy(self, /) -> LinearSolvePolicy:
        return LinearSolvePolicy(
            GMRES(restart=32),
            tolerance=TolerancePolicy(
                relative=self.solve_relative_tolerance,
                absolute=self.solve_absolute_tolerance,
                max_steps=self.maximum_solve_steps,
            ),
        )

    def nonlinear_termination(self, /) -> NonlinearTermination:
        return NonlinearTermination(
            absolute_residual=self.nonlinear_absolute_tolerance,
            relative_residual=self.nonlinear_relative_tolerance,
            maximum_steps=self.maximum_nonlinear_steps,
        )


@runtime_checkable
class FiniteElementBoundaryProvider(Protocol):
    """Structural provider of fixed-route boundary coordinates."""

    @property
    def mapping_id(self) -> str: ...

    @property
    def reference_points(self) -> Array: ...

    def realize(self, design: Any, /) -> Any: ...


class FiniteElementBoundaryRealization(StrictModule):
    """Normalized boundary coordinates and provider acceptance evidence."""

    proposed_points: Array
    points: Array
    accepted: Array
    refresh_required: Array
    status: Array
    mapping_id: str = eqx.field(static=True)

    def __init__(
        self,
        proposed_points: Any,
        points: Any,
        /,
        *,
        accepted: Any,
        refresh_required: Any,
        status: Any,
        mapping_id: str,
    ) -> None:
        proposed = jnp.asarray(proposed_points, dtype=jnp.float64)
        safe = jnp.asarray(points, dtype=proposed.dtype)
        if proposed.ndim != 2 or safe.shape != proposed.shape:
            raise ValueError(
                "Boundary coordinates must have matching shape (points, dim)."
            )
        if not mapping_id:
            raise ValueError("mapping_id must be non-empty.")
        self.proposed_points = proposed
        self.points = safe
        self.accepted = jnp.asarray(accepted, dtype=jnp.bool_).reshape(())
        self.refresh_required = jnp.asarray(refresh_required, dtype=jnp.bool_).reshape(())
        self.status = jnp.asarray(status, dtype=jnp.int32).reshape(())
        self.mapping_id = str(mapping_id)


class FiniteElementMotionExtensionResult(StrictModule):
    """Vertex displacement of one route plus its native solver evidence.

    ``status`` is the linear-solve status (one entry per coordinate column for
    ``HARMONIC``) or the nonlinear status of ``WINSLOW``/``MMPDE``; ``PRESCRIBED``
    reports success.
    """

    displacement: Array
    successful: Array
    status: Array
    iterations: Array
    residual_norm: Array
    route: FiniteElementMeshMotionRoute = eqx.field(static=True)


class FiniteElementMeshMotionEvidence(StrictModule):
    """Complete acceptance evidence for one mesh-motion proposal."""

    boundary: FiniteElementBoundaryRealization
    geometry: MotionValidityEvidence
    extension: FiniteElementMotionExtensionResult
    status: Array
    plan_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        boundary: FiniteElementBoundaryRealization,
        geometry: MotionValidityEvidence,
        extension: FiniteElementMotionExtensionResult,
        status: Any,
        plan_id: str,
        topology_id: str,
        geometry_layout_id: str,
    ) -> None:
        self.boundary = boundary
        self.geometry = geometry
        self.extension = extension
        self.status = jnp.asarray(status, dtype=jnp.int32).reshape(())
        self.plan_id = str(plan_id)
        self.topology_id = str(topology_id)
        self.geometry_layout_id = str(geometry_layout_id)

    @property
    def accepted(self) -> Array:
        return self.status == int(FiniteElementMeshMotionStatus.SUCCESS)

    @property
    def refresh_required(self) -> Array:
        return self.boundary.refresh_required


class FiniteElementMeshRealization(StrictModule):
    """Proposed and safe FE coordinates plus the safe execution runtime."""

    proposed_coordinates: Array
    coordinates: Array
    runtime: FiniteElementRuntimeData
    evidence: FiniteElementMeshMotionEvidence

    def __init__(
        self,
        proposed_coordinates: Any,
        coordinates: Any,
        runtime: FiniteElementRuntimeData,
        evidence: FiniteElementMeshMotionEvidence,
        /,
    ) -> None:
        proposed = jnp.asarray(proposed_coordinates, dtype=jnp.float64)
        safe = jnp.asarray(coordinates, dtype=proposed.dtype)
        if proposed.ndim != 2 or safe.shape != proposed.shape:
            raise ValueError("FE coordinates must have matching shape (points, dim).")
        if runtime.coordinates.shape != safe.shape:
            raise ValueError("FE runtime coordinates must match the realization.")
        self.proposed_coordinates = proposed
        self.coordinates = safe
        self.runtime = runtime
        self.evidence = evidence

    @property
    def accepted(self) -> Array:
        return self.evidence.accepted

    @property
    def refresh_required(self) -> Array:
        return self.evidence.refresh_required


# Host preparation ------------------------------------------------------------------


def _validated_blocks(
    reference: np.ndarray, cell_blocks: Sequence[tuple[str, ArrayLike]], /
) -> tuple[tuple[str, np.ndarray], ...]:
    blocks = []
    for kind, cells in cell_blocks:
        kind_ = str(kind)
        if motion_cell_dimension(kind_) != reference.shape[1]:
            raise ValueError("Mesh motion requires full-dimensional cells.")
        cells_ = np.asarray(cells, dtype=np.int64)
        arity = len(reference_cell_topology(kind_).vertices)
        if cells_.ndim != 2 or cells_.shape[1] != arity or cells_.shape[0] == 0:
            raise ValueError(f"{kind_} cells must have shape (cells > 0, {arity}).")
        if np.any(cells_ < 0) or np.any(cells_ >= reference.shape[0]):
            raise ValueError("Cell vertices must index the reference coordinates.")
        blocks.append((kind_, cells_))
    if not blocks:
        raise ValueError("Mesh motion requires at least one cell block.")
    return tuple(blocks)


def _mesh_edges(blocks: tuple[tuple[str, np.ndarray], ...], /) -> np.ndarray:
    pairs = []
    for kind, cells in blocks:
        local = np.asarray(reference_cell_topology(kind).entities[1], dtype=np.int64)
        pairs.append(cells[:, local].reshape(-1, 2))
    edges = np.sort(np.concatenate(pairs), axis=1)
    return np.unique(edges, axis=0)


def _require_boundary_reachable(
    edges: np.ndarray, boundary: np.ndarray, vertex_count: int, /
) -> None:
    graph = coo_matrix(
        (np.ones((edges.shape[0],)), (edges[:, 0], edges[:, 1])),
        shape=(vertex_count, vertex_count),
    )
    _, labels = connected_components(graph, directed=False)
    anchored = np.zeros((labels.max() + 1,), dtype=np.bool_)
    anchored[labels[boundary]] = True
    if not np.all(anchored[labels]):
        raise ValueError("Every interior mesh component must connect to the boundary.")


def _padded_element_tensors(
    local_matrices: Sequence[np.ndarray], vertex_routes: Sequence[np.ndarray], /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate element tensors of different vertex widths with inert padding."""

    width = max(route.shape[1] for route in vertex_routes)
    matrices = []
    routes = []
    padding = []
    for matrix, route in zip(local_matrices, vertex_routes, strict=True):
        components = matrix.shape[1] // route.shape[1]
        extra = (width - route.shape[1]) * components
        matrices.append(np.pad(matrix, ((0, 0), (0, extra), (0, extra))))
        routes.append(np.pad(route, ((0, 0), (0, width - route.shape[1]))))
        padding.append(
            np.broadcast_to(np.arange(width) >= route.shape[1], (route.shape[0], width))
        )
    return np.concatenate(matrices), np.concatenate(routes), np.concatenate(padding)


def _harmonic_tensors(
    reference: np.ndarray, edges: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = np.linalg.norm(reference[edges[:, 1]] - reference[edges[:, 0]], axis=-1)
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= 0.0):
        raise ValueError("Mesh-motion graph edges must have positive finite length.")
    stencil = np.asarray(((1.0, -1.0), (-1.0, 1.0)))
    local = stencil[None] / lengths[:, None, None]
    return local, edges, np.zeros(edges.shape, dtype=np.bool_)


def _reference_gradients(
    reference: np.ndarray, kind: str, cells: np.ndarray, degree: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Physical P1/Q1 gradients, |det J|, and weights at reference quadrature."""

    points, weights = _degree_aware_reference_rule(kind, degree)
    _, gradients = lagrange_element(kind, 1).tabulate(points)
    dimension = reference.shape[1]
    jacobian = contract("cla,qlb->cqab", jnp.asarray(reference[cells]), gradients)
    # grad_x N = J^-T grad_xi N, solved per quadrature point.
    result = solve_small_linear(
        SmallLinearSolvePlan(dimension),
        jnp.swapaxes(jacobian, -1, -2),
        jnp.broadcast_to(
            jnp.swapaxes(gradients, -1, -2)[None],
            jacobian.shape[:2] + (dimension, gradients.shape[1]),
        ),
    )
    if not np.all(np.asarray(result.successful)):
        raise ValueError("Reference mesh has a singular cell Jacobian.")
    return (
        np.asarray(jnp.swapaxes(result.value, -1, -2)),
        np.abs(np.asarray(result.determinant)),
        np.asarray(weights),
    )


def _elasticity_tensors(
    reference: np.ndarray,
    blocks: tuple[tuple[str, np.ndarray], ...],
    policy: FiniteElementMeshMotionPolicy,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Jacobian-stiffened linear-elasticity element tensors (Stein et al. 2003)."""

    dimension = reference.shape[1]
    samples = [_reference_gradients(reference, kind, cells, 2) for kind, cells in blocks]
    largest = max(float(np.max(determinant)) for _, determinant, _ in samples)
    nu = policy.poisson_ratio
    lame_lambda = nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    lame_mu = 0.5 / (1.0 + nu)
    identity = np.eye(dimension)
    matrices = []
    for (gradients, determinant, weights), (_, cells) in zip(
        samples, blocks, strict=True
    ):
        stiffness = (largest / determinant) ** policy.stiffening_exponent
        measure = jnp.asarray(weights[None, :] * determinant * stiffness)
        grad = jnp.asarray(gradients)
        laplace = contract("cq,cqld,cqmd->clm", measure, grad, grad)
        transposed = contract("cq,cqlb,cqma->clamb", measure, grad, grad)
        divergence = contract("cq,cqla,cqmb->clamb", measure, grad, grad)
        local = lame_mu * (
            np.asarray(laplace)[:, :, None, :, None] * identity[None, None, :, None, :]
            + np.asarray(transposed)
        ) + lame_lambda * np.asarray(divergence)
        arity = cells.shape[1]
        matrices.append(local.reshape(cells.shape[0], arity * dimension, -1))
    return _padded_element_tensors(matrices, [cells for _, cells in blocks])


class _DirichletExtension(StrictModule, NonTrainableState):
    """Matrix-free Dirichlet elimination ``K_II u_I = -K_IB u_B`` on element tensors.

    Element rows are vertex-major ``(vertex, component)`` blocks. Scalar
    (``components == 1``) operators solve every coordinate column as one
    multi-right-hand-side solve; vector operators couple the components.
    """

    interior_operator: ElementTensorOperator
    coupling_operator: ElementTensorOperator
    prepared: PreparedLinearSolve
    components: int = eqx.field(static=True)

    def __init__(
        self,
        tensors: tuple[np.ndarray, np.ndarray, np.ndarray],
        boundary_vertices: np.ndarray,
        interior_vertices: np.ndarray,
        dimension: int,
        policy: FiniteElementMeshMotionPolicy,
        problem_id: str,
        /,
    ) -> None:
        local_matrices, vertex_routes, padding = tensors
        components = local_matrices.shape[1] // vertex_routes.shape[1]
        vertex_count = boundary_vertices.size + interior_vertices.size
        interior_local = np.full((vertex_count,), -1, dtype=np.int64)
        boundary_local = np.full((vertex_count,), -1, dtype=np.int64)
        interior_local[interior_vertices] = np.arange(interior_vertices.size)
        boundary_local[boundary_vertices] = np.arange(boundary_vertices.size)
        component = np.arange(components)
        active = ~np.repeat(padding, components, axis=1)
        is_interior = np.repeat(interior_local[vertex_routes] >= 0, components, axis=1)
        is_boundary = np.repeat(boundary_local[vertex_routes] >= 0, components, axis=1)
        is_interior &= active
        is_boundary &= active

        def dofs(local: np.ndarray) -> np.ndarray:
            return (
                np.maximum(local[vertex_routes], 0)[..., None] * components + component
            ).reshape(vertex_routes.shape[0], -1)

        interior_dofs = dofs(interior_local)
        interior_size = interior_vertices.size * components
        self.interior_operator = ElementTensorOperator(
            np.where(
                is_interior[:, :, None] & is_interior[:, None, :], local_matrices, 0.0
            ),
            interior_dofs,
            interior_dofs,
            interior_size,
            interior_size,
            valid=np.any(is_interior, axis=1),
            accumulation="deterministic",
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={"positive_definite": "construction"},
            ),
        )
        self.coupling_operator = ElementTensorOperator(
            np.where(
                is_interior[:, :, None] & is_boundary[:, None, :], local_matrices, 0.0
            ),
            dofs(boundary_local),
            interior_dofs,
            boundary_vertices.size * components,
            interior_size,
            valid=np.any(is_interior, axis=1) & np.any(is_boundary, axis=1),
            accumulation="deterministic",
        )
        self.prepared = prepare(
            LinearSystem(
                self.interior_operator.as_linear_operator(), problem_id=problem_id
            ),
            LinearSolvePolicy(
                ConjugateGradient(),
                tolerance=TolerancePolicy(
                    relative=policy.solve_relative_tolerance,
                    absolute=policy.solve_absolute_tolerance,
                    max_steps=policy.maximum_solve_steps,
                ),
                differentiation=DifferentiationPolicy("rhs-only"),
                failure=FailurePolicy("status"),
            ),
            rhs_layout=(
                RHSLayout((dimension,), names=("coordinate",))
                if components == 1
                else None
            ),
        )
        self.components = components

    def solve(self, boundary_displacement: Array, /) -> Any:
        """Interior displacement ``(interior, dim)`` and the native solve result."""

        if self.components == 1:
            coupling = jax.vmap(self.coupling_operator.mv, in_axes=1, out_axes=1)(
                boundary_displacement
            )
            result = solve(self.prepared, -coupling)
            return result.value, result
        coupling = self.coupling_operator.mv(boundary_displacement.reshape(-1))
        result = solve(self.prepared, -coupling)
        return result.value.reshape(-1, boundary_displacement.shape[1]), result


def _reference_cell_measure(kind: str, /) -> float:
    return float(np.sum(np.asarray(_degree_aware_reference_rule(kind, 0)[1])))


class _CornerEnergy(StrictModule, NonTrainableState):
    """Corner Jacobians ``F = A Ahat^-1`` against the reference mesh.

    ``measures`` split each reference cell measure over its corners so that the
    corner sum integrates affine maps exactly.
    """

    corner_vertices: Array
    corner_neighbors: Array
    reference_inverse: Array
    measures: Array
    inverse_plan: SmallLinearSolvePlan
    determinant_plan: SmallLinearSolvePlan

    def __init__(
        self, reference: np.ndarray, blocks: tuple[tuple[str, np.ndarray], ...], /
    ) -> None:
        dimension = reference.shape[1]
        determinant_plan = SmallLinearSolvePlan(dimension)
        vertices = []
        neighbors = []
        scale = []
        for kind, cells in blocks:
            table_vertices, table_neighbors = motion_corner_routes(kind)
            unit = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)
            unit_frames = np.swapaxes(
                unit[table_neighbors] - unit[table_vertices][:, None, :], -1, -2
            )
            unit_total = float(
                np.sum(
                    np.asarray(
                        determinant_small_linear(
                            determinant_plan, jnp.asarray(unit_frames)
                        )
                    )
                )
            )
            vertices.append(cells[:, table_vertices].reshape(-1))
            neighbors.append(cells[:, table_neighbors].reshape(-1, dimension))
            scale.append(
                np.full(
                    (cells.shape[0] * table_vertices.size,),
                    _reference_cell_measure(kind) / unit_total,
                )
            )
        corner_vertices = jnp.asarray(np.concatenate(vertices), dtype=jnp.int32)
        corner_neighbors = jnp.asarray(np.concatenate(neighbors), dtype=jnp.int32)
        frames = corner_frame_matrices(
            jnp.asarray(reference), corner_vertices, corner_neighbors
        )
        inverse = inverse_small_linear(determinant_plan, frames)
        if not np.all(np.asarray(inverse.successful)) or np.any(
            np.asarray(inverse.determinant) <= 0.0
        ):
            raise ValueError("Reference corner frames must be positively oriented.")
        self.corner_vertices = corner_vertices
        self.corner_neighbors = corner_neighbors
        self.reference_inverse = inverse.value
        self.measures = jnp.asarray(
            np.asarray(inverse.determinant) * np.concatenate(scale)
        )
        self.inverse_plan = SmallLinearSolvePlan(dimension, refinement_iterations=0)
        self.determinant_plan = determinant_plan

    def jacobians(self, points: Array, /) -> Array:
        frames = corner_frame_matrices(
            points, self.corner_vertices, self.corner_neighbors
        )
        return frames @ self.reference_inverse

    def admissible(self, points: Array, /) -> Array:
        determinants = determinant_small_linear(
            self.determinant_plan, self.jacobians(points)
        )
        return jnp.all(jnp.isfinite(determinants) & (determinants > 0.0))

    def winslow(self, points: Array, /) -> Array:
        """Dirichlet energy of the inverse map, ``sum m det(F) |F^-1|**2``."""

        inverse = inverse_small_linear(self.inverse_plan, self.jacobians(points))
        return jnp.sum(
            self.measures
            * inverse.determinant
            * jnp.sum(inverse.value * inverse.value, axis=(-2, -1))
        )

    def huang(self, points: Array, monitor: Array, /) -> Array:
        """Huang meshing functional ``sum |K| G(J, det J, M)`` with ``J = F^-1``."""

        dimension = points.shape[1]
        corner_monitor = jnp.mean(
            jnp.concatenate(
                (
                    monitor[self.corner_vertices][:, None],
                    monitor[self.corner_neighbors],
                ),
                axis=1,
            ),
            axis=1,
        )
        inverse = inverse_small_linear(self.inverse_plan, self.jacobians(points))
        monitor_inverse = inverse_small_linear(self.inverse_plan, corner_monitor)
        root = jnp.sqrt(monitor_inverse.determinant)
        alignment = jnp.sum(
            (inverse.value @ monitor_inverse.value) * inverse.value, axis=(-2, -1)
        )
        power = 0.5 * dimension * _MMPDE_EXPONENT
        density = _MMPDE_THETA * root * alignment**power + (
            1.0 - 2.0 * _MMPDE_THETA
        ) * dimension**power * root * (1.0 / (inverse.determinant * root)) ** (
            _MMPDE_EXPONENT
        )
        return jnp.sum(self.measures * inverse.determinant * density)


def _monitor_tensors(values: Array, count: int, dimension: int, /) -> Array:
    monitor = jnp.asarray(values)
    if monitor.shape == (count,):
        return monitor[:, None, None] * jnp.eye(dimension, dtype=monitor.dtype)
    if monitor.shape == (count, dimension, dimension):
        return monitor
    raise ValueError(
        "A mesh monitor must return (vertices,) scalars or (vertices, dim, dim) tensors."
    )


class _RouteArguments(StrictModule):
    boundary_points: Array
    monitor: Any


class FiniteElementMotionExtension(StrictModule, NonTrainableState):
    """Prepared fixed-topology route from boundary to interior vertex displacement.

    ``cell_blocks`` lists ``(cell_kind, cells)`` in canonical reference vertex
    order. ``boundary_vertices`` are prescribed; for ``PRESCRIBED`` they must be
    every vertex. Preparation (graph/element tensors, Krylov solve, corner
    frames) happens once; :meth:`extend` is traceable and differentiable in the
    boundary displacement and the MMPDE monitor parameters.
    """

    reference_coordinates: Array
    boundary_indices: Array
    interior_indices: Array
    linear: _DirichletExtension | None
    energy: _CornerEnergy | None
    policy: FiniteElementMeshMotionPolicy
    extension_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_coordinates: ArrayLike,
        cell_blocks: Sequence[tuple[str, ArrayLike]],
        boundary_vertices: ArrayLike,
        /,
        *,
        policy: FiniteElementMeshMotionPolicy | None = None,
    ) -> None:
        policy_ = FiniteElementMeshMotionPolicy() if policy is None else policy
        if not isinstance(policy_, FiniteElementMeshMotionPolicy):
            raise TypeError("policy must be FiniteElementMeshMotionPolicy or None.")
        reference = np.asarray(reference_coordinates, dtype=np.float64)
        if reference.ndim != 2 or reference.shape[1] not in (1, 2, 3):
            raise ValueError("Reference coordinates must have shape (points, 1-3).")
        if not np.all(np.isfinite(reference)):
            raise ValueError("Reference coordinates must be finite.")
        blocks = _validated_blocks(reference, cell_blocks)
        boundary = np.asarray(boundary_vertices, dtype=np.int64)
        count, dimension = reference.shape
        if (
            boundary.ndim != 1
            or boundary.size == 0
            or np.any(np.diff(boundary) <= 0)
            or boundary[0] < 0
            or boundary[-1] >= count
        ):
            raise ValueError(
                "boundary_vertices must be sorted unique non-empty vertex indices."
            )
        interior_mask = np.ones((count,), dtype=np.bool_)
        interior_mask[boundary] = False
        interior = np.flatnonzero(interior_mask)
        route = policy_.route
        if route is FiniteElementMeshMotionRoute.PRESCRIBED and interior.size:
            raise ValueError("The PRESCRIBED route must prescribe every vertex.")
        linear = None
        energy = None
        if interior.size:
            edges = _mesh_edges(blocks)
            _require_boundary_reachable(edges, boundary, count)
            problem_id = f"{array_tree_fingerprint(reference)}:{route.value}"
            match route:
                case FiniteElementMeshMotionRoute.HARMONIC:
                    linear = _DirichletExtension(
                        _harmonic_tensors(reference, edges),
                        boundary,
                        interior,
                        dimension,
                        policy_,
                        problem_id,
                    )
                case FiniteElementMeshMotionRoute.LINEAR_ELASTICITY:
                    linear = _DirichletExtension(
                        _elasticity_tensors(reference, blocks, policy_),
                        boundary,
                        interior,
                        dimension,
                        policy_,
                        problem_id,
                    )
                case (
                    FiniteElementMeshMotionRoute.WINSLOW
                    | FiniteElementMeshMotionRoute.MMPDE
                ):
                    # The harmonic extension is the admissible initial state.
                    linear = _DirichletExtension(
                        _harmonic_tensors(reference, edges),
                        boundary,
                        interior,
                        dimension,
                        policy_,
                        problem_id,
                    )
                    energy = _CornerEnergy(reference, blocks)
                case _:
                    raise ValueError(f"Unsupported mesh motion route {route!r}.")
        self.reference_coordinates = jnp.asarray(reference)
        self.boundary_indices = jnp.asarray(boundary, dtype=jnp.int32)
        self.interior_indices = jnp.asarray(interior, dtype=jnp.int32)
        self.linear = linear
        self.energy = energy
        self.policy = policy_
        self.extension_id = canonical_fingerprint(
            {
                "kind": "finite-element-motion-extension",
                "reference": array_tree_fingerprint(reference),
                "blocks": [
                    [kind, array_tree_fingerprint(cells)] for kind, cells in blocks
                ],
                "boundary": array_tree_fingerprint(boundary),
                "policy": policy_.policy_id,
            }
        )

    def _points(self, interior: Array, boundary_points: Array, /) -> Array:
        return (
            self.reference_coordinates.at[self.boundary_indices]
            .set(boundary_points)
            .at[self.interior_indices]
            .set(interior)
        )

    def _stationary_route(
        self,
        boundary_points: Array,
        initial: Array,
        monitor: MeshMonitor | None,
        /,
    ) -> Any:
        energy = self.energy
        policy = self.policy
        count, dimension = self.reference_coordinates.shape
        interior_indices = self.interior_indices

        def admissible(state: Any, arguments: Any) -> Any:
            # ty: ignore[unresolved-attribute]
            return energy.admissible(self._points(state, arguments.boundary_points))

        match policy.route:
            case FiniteElementMeshMotionRoute.WINSLOW:

                def residual(state: Any, arguments: Any) -> Any:
                    return jax.grad(
                        # ty: ignore[unresolved-attribute]
                        lambda interior: energy.winslow(
                            self._points(interior, arguments.boundary_points)
                        )
                    )(state)

                method = NewtonKrylov(linear_policy=policy.linear_policy())
            case FiniteElementMeshMotionRoute.MMPDE:

                def residual(state: Any, arguments: Any) -> Any:
                    def functional(interior: Any) -> Any:
                        points = self._points(interior, arguments.boundary_points)
                        values = _monitor_tensors(
                            arguments.monitor(points), count, dimension
                        )
                        # ty: ignore[unresolved-attribute]
                        return energy.huang(points, values)

                    points = self._points(state, arguments.boundary_points)
                    values = _monitor_tensors(arguments.monitor(points), count, dimension)
                    # Huang–Kamenski balancing P = det(M)**((p - 1) / 2).
                    balance = determinant_small_linear(
                        # ty: ignore[unresolved-attribute]
                        energy.determinant_plan,
                        values[interior_indices],
                    ) ** (0.5 * (_MMPDE_EXPONENT - 1.0))
                    return (balance[:, None] / policy.relaxation_time) * jax.grad(
                        functional
                    )(state)

                method = PseudoTransient(
                    linear=policy.linear_policy(),
                    initial_step=policy.mmpde_initial_time_step,
                )
            case route:
                raise ValueError(f"Route {route!r} has no stationarity equation.")
        problem = NonlinearSystemProblem(
            residual,
            trial_validity=admissible,
            trial_validity_id="positive-corner-jacobians",
            problem_id=f"{self.extension_id}:{policy.route.value}",
        )
        return implicit_root_result(
            problem,
            initial,
            method=method,
            termination=policy.nonlinear_termination(),
            derivative_policy=ImplicitRootDerivativePolicy(
                tangent_linear_policy=policy.linear_policy()
            ),
            args=_RouteArguments(boundary_points, monitor),
        )

    def extend(
        self,
        boundary_displacement: ArrayLike,
        /,
        *,
        monitor: MeshMonitor | None = None,
    ) -> FiniteElementMotionExtensionResult:
        """Interior displacement for one boundary displacement ``(boundary, dim)``."""

        route = self.policy.route
        if (monitor is None) == (route is FiniteElementMeshMotionRoute.MMPDE):
            raise ValueError("A monitor is required exactly for the MMPDE route.")
        if monitor is not None and not callable(monitor):
            raise TypeError("monitor must be callable.")
        reference = self.reference_coordinates
        boundary = jnp.asarray(boundary_displacement, dtype=reference.dtype)
        if boundary.shape != (self.boundary_indices.shape[0], reference.shape[1]):
            raise ValueError("boundary_displacement must have shape (boundary, dim).")
        boundary_points = reference[self.boundary_indices] + boundary

        def result(
            interior: Any,
            successful: Any,
            status: Any,
            iterations: Any,
            residual_norm: Any,
        ) -> Any:
            displacement = (
                jnp.zeros_like(reference)
                .at[self.boundary_indices]
                .set(boundary)
                .at[self.interior_indices]
                .set(interior)
            )
            return FiniteElementMotionExtensionResult(
                displacement=displacement,
                successful=jnp.asarray(successful, dtype=jnp.bool_),
                status=jnp.asarray(status, dtype=jnp.int32),
                iterations=jnp.asarray(iterations, dtype=jnp.int32),
                residual_norm=jnp.asarray(residual_norm, dtype=reference.dtype),
                route=route,
            )

        if self.interior_indices.shape[0] == 0:
            return result(
                jnp.zeros((0, reference.shape[1]), dtype=reference.dtype),
                True,
                0,
                0,
                0.0,
            )
        match route:
            case (
                FiniteElementMeshMotionRoute.HARMONIC
                | FiniteElementMeshMotionRoute.LINEAR_ELASTICITY
            ):
                # ty: ignore[unresolved-attribute]
                interior, solved = self.linear.solve(boundary)
                return result(
                    interior,
                    jnp.all(solved.successful),
                    solved.status,
                    jnp.max(solved.diagnostics.iterations),
                    jnp.max(solved.diagnostics.residual_norm),
                )
            case (
                FiniteElementMeshMotionRoute.WINSLOW | FiniteElementMeshMotionRoute.MMPDE
            ):
                # ty: ignore[unresolved-attribute]
                harmonic, _ = self.linear.solve(boundary)
                initial = reference[self.interior_indices] + harmonic
                root = self._stationary_route(boundary_points, initial, monitor)
                return result(
                    root.state - reference[self.interior_indices],
                    root.successful,
                    root.status,
                    root.diagnostics.iterations,
                    root.diagnostics.final_residual_norm,
                )
            case _:
                raise ValueError(f"Unsupported mesh motion route {route!r}.")


@eqx.filter_jit
def _propose_coordinates(
    extension: FiniteElementMotionExtension,
    validity: MotionValidityPlan,
    boundary_displacement: Array,
    monitor: MeshMonitor | None,
    /,
) -> tuple[Array, FiniteElementMotionExtensionResult, MotionValidityEvidence]:
    extended = extension.extend(boundary_displacement, monitor=monitor)
    proposed = extension.reference_coordinates + extended.displacement
    return proposed, extended, validity.evaluate(proposed)


def _normalized_boundary(
    provider: FiniteElementBoundaryProvider,
    design: Any,
) -> FiniteElementBoundaryRealization:
    result = provider.realize(design)
    return FiniteElementBoundaryRealization(
        result.proposed_points,
        result.points,
        accepted=result.accepted,
        refresh_required=result.refresh_required,
        status=result.status,
        mapping_id=provider.mapping_id,
    )


class FiniteElementMeshMotionPlan(StrictModule):
    """Fixed-topology coordinate realization of affine FE meshes along one route.

    The boundary provider realizes every boundary vertex (every vertex for the
    ``PRESCRIBED`` route); the policy route extends the displacement to the
    interior, and :class:`MotionValidityPlan` accepts or rejects the result
    inside the traced step. :meth:`certify` is the host-epoch Bernstein proof.
    """

    discretization: FiniteElementDiscretization
    boundary_provider: FiniteElementBoundaryProvider
    extension: FiniteElementMotionExtension
    validity: MotionValidityPlan
    policy: FiniteElementMeshMotionPolicy
    plan_id: str = eqx.field(static=True)
    mapping_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: FiniteElementDiscretization,
        boundary_provider: FiniteElementBoundaryProvider,
        /,
        *,
        policy: FiniteElementMeshMotionPolicy | None = None,
    ) -> None:
        if not isinstance(discretization, FiniteElementDiscretization):
            raise TypeError("discretization must be FiniteElementDiscretization.")
        if not isinstance(boundary_provider, FiniteElementBoundaryProvider):
            raise TypeError(
                "boundary_provider must satisfy FiniteElementBoundaryProvider."
            )
        policy_ = FiniteElementMeshMotionPolicy() if policy is None else policy
        if not isinstance(policy_, FiniteElementMeshMotionPolicy):
            raise TypeError("policy must be FiniteElementMeshMotionPolicy or None.")
        mesh = discretization.mesh
        if mesh.ambient_dimension != mesh.topological_dimension or (
            mesh.ambient_dimension not in (2, 3)
        ):
            raise ValueError("Mesh motion requires a full-dimensional 2-D or 3-D mesh.")
        if discretization.default_runtime.coordinates.shape != mesh.coordinates.shape:
            raise ValueError("Mesh motion requires a vertex-coordinate layout.")
        for block, element, dofs in zip(
            mesh.blocks,
            discretization.coordinate_elements,
            discretization.coordinate_dofs,
            strict=True,
        ):
            # ty: ignore[unresolved-attribute]
            if element.degree != 1 or element.local_dof_count != block.arity:
                raise ValueError("Mesh motion requires affine P1/Q1 coordinates.")
            if not np.array_equal(np.asarray(dofs), np.asarray(block.vertices)):
                raise ValueError("Coordinate DOFs must coincide with mesh vertices.")
        reference = np.asarray(mesh.coordinates, dtype=np.float64)
        if policy_.route is FiniteElementMeshMotionRoute.PRESCRIBED:
            boundary = np.arange(reference.shape[0], dtype=np.int64)
        else:
            boundary = np.flatnonzero(
                np.asarray(mesh.topology.entities(0).subset("boundary").mask)
            )
        provider_reference = np.asarray(
            boundary_provider.reference_points, dtype=np.float64
        )
        if provider_reference.shape != (boundary.size, mesh.ambient_dimension):
            raise ValueError("Boundary provider routes must match every moved vertex.")
        if not np.allclose(
            provider_reference,
            reference[boundary],
            rtol=0.0,
            atol=max(policy_.validity.minimum_absolute_jacobian, 1.0e-12),
        ):
            raise ValueError(
                "Boundary provider reference points do not match the FE mesh."
            )
        blocks = tuple((block.cell_kind, block.vertices) for block in mesh.blocks)
        extension = FiniteElementMotionExtension(
            reference, blocks, boundary, policy=policy_
        )
        self.discretization = discretization
        self.boundary_provider = boundary_provider
        self.extension = extension
        self.validity = MotionValidityPlan(reference, blocks, policy=policy_.validity)
        self.policy = policy_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-element-mesh-motion",
                "prepared": discretization.prepared_id,
                "mapping": boundary_provider.mapping_id,
                "extension": extension.extension_id,
            }
        )
        self.mapping_id = boundary_provider.mapping_id
        self.topology_id = mesh.topology_id
        self.geometry_layout_id = discretization.default_runtime.geometry_layout_id

    @property
    def reference_coordinates(self) -> Array:
        return self.extension.reference_coordinates

    def realize(
        self,
        design: Any,
        /,
        *,
        numeric_version: str,
        monitor: MeshMonitor | None = None,
    ) -> FiniteElementMeshRealization:
        """Realize one design; ``monitor`` drives the MMPDE route."""

        version = str(numeric_version)
        if not version:
            raise ValueError("numeric_version must be non-empty.")
        boundary = _normalized_boundary(self.boundary_provider, design)
        indices = self.extension.boundary_indices
        if boundary.proposed_points.shape != (
            indices.shape[0],
            self.reference_coordinates.shape[1],
        ):
            raise ValueError("Boundary provider changed its fixed coordinate shape.")
        proposed, extended, geometry = _propose_coordinates(
            self.extension,
            self.validity,
            boundary.proposed_points - self.reference_coordinates[indices],
            monitor,
        )
        status = jnp.where(
            boundary.accepted, 0, int(FiniteElementMeshMotionStatus.BOUNDARY_REJECTED)
        ) | jnp.where(
            extended.successful, 0, int(FiniteElementMeshMotionStatus.EXTENSION_FAILED)
        )
        for reason, flag in _GEOMETRY_STATUS:
            status = status | jnp.where(
                (geometry.status & int(reason)) != 0, int(flag), 0
            )
        evidence = FiniteElementMeshMotionEvidence(
            boundary=boundary,
            geometry=geometry,
            extension=extended,
            status=status.astype(jnp.int32),
            plan_id=self.plan_id,
            topology_id=self.topology_id,
            geometry_layout_id=self.geometry_layout_id,
        )
        safe = jnp.where(evidence.accepted, proposed, self.reference_coordinates)
        runtime = self.discretization.prepare_runtime(
            safe,
            numeric_version=canonical_fingerprint(
                {
                    "kind": "finite-element-mesh-motion-runtime",
                    "plan": self.plan_id,
                    "numeric_version": version,
                }
            ),
        )
        return FiniteElementMeshRealization(proposed, safe, runtime, evidence)

    def certify(
        self,
        coordinates: ArrayLike,
        /,
        *,
        policy: CellValidityPolicy | None = None,
    ) -> CellValidityCertificate:
        """Host-epoch Bernstein certificate of moved vertex coordinates."""

        mesh = self.discretization.mesh
        return certify_cell_geometry_validity(
            mesh.with_coordinates(
                np.asarray(coordinates, dtype=np.float64),
                numeric_version=f"{self.plan_id}:certification",
            ),
            policy=policy,
        )


__all__ = [
    "FiniteElementBoundaryProvider",
    "FiniteElementBoundaryRealization",
    "FiniteElementMeshMotionEvidence",
    "FiniteElementMeshMotionPlan",
    "FiniteElementMeshMotionPolicy",
    "FiniteElementMeshMotionRoute",
    "FiniteElementMeshMotionStatus",
    "FiniteElementMeshRealization",
    "FiniteElementMotionExtension",
    "FiniteElementMotionExtensionResult",
]
