#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-topology mesh coordinate optimization through :mod:`phydrax.optim`.

Target-matrix optimization (Knupp 2012) measures every corner Jacobian ``A``
against a target ``W`` through ``T = A W**-1``. Corner frames of triangles,
tetrahedra, quadrilaterals, hexahedra, prisms, pyramids, and polygons, and the
face-centroid star simplices of polyhedra share one energy. Fixed nodes are
eliminated: only free rows are optimization parameters and fixed rows are
scattered unchanged, so they remain bit-identical. Every solve executes the same
stable module-level compiled route through :func:`phydrax.optim.minimize`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from enum import StrEnum
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import finite_real_scalar
from ..discretization import CellGeometrySpec, CellMesh
from ..discretization._cell_geometry_validity import polyhedral_star_tables
from ..linalg import (
    determinant_small_linear,
    hermitian_inverse_sqrt,
    inverse_small_linear,
    SmallLinearSolvePlan,
)
from ..optim import (
    Bounds,
    MinimizationResult,
    minimize,
    NewtonTrustRegion,
    OptimizationStatus,
    OptimizationTermination,
    ProjectedLBFGS,
)
from ..sparse import EdgeRelation, SparseLinearMap
from ._audit import audit_cell_mesh, CellMeshAuditPolicy
from ._canonical import certify_cell_mesh
from ._metric import interpolate_mesh_metric, MeshMetricField
from ._quality import _corner_table, _ideal_vertices, evaluate_cell_quality
from ._result import CellMeshingResult


class MeshQualityObjective(StrEnum):
    """Per-corner target-matrix quality term.

    ``SHAPE``: generalized Knupp ``mu_2 = |T|**d / (d**(d/2) det T) - 1``
    (scale invariant, infinite at inversion). ``SHAPE_SIZE``: ``mu_2`` plus the
    size term ``mu_77 = (det T - 1/det T)**2 / 2``. ``METRIC_ALIGNMENT``:
    ``SHAPE_SIZE`` against targets ``W = M**(-1/2) W_ideal`` from a Riemannian
    metric (unit metric edges). ``GRAM_DETERMINANT``: the Frobenius energy
    ``|T^T T - I|**2 + (det T - 1)**2``. Every objective rejects inverted corners.
    """

    SHAPE = "shape"
    SHAPE_SIZE = "shape_size"
    METRIC_ALIGNMENT = "metric_alignment"
    GRAM_DETERMINANT = "gram_determinant"


class MeshOptimizationStatus(StrEnum):
    """Outcome of one fixed-topology mesh optimization.

    ``OPTIMIZED``: the native minimization converged and the inversion-free,
    audited mesh is certified. ``VALID_NONCONVERGED``: the minimization stopped
    without converging (step or evaluation budget, stagnation, line-search
    failure, ...), its iterate is inversion free and passes the audit, and the plan
    explicitly accepts valid non-convergence; the iterate is certified.
    ``NONCONVERGED``: the same valid but non-converged iterate under a plan that
    does not accept it; nothing is committed. ``INVERTED_INPUT``: the input is
    inverted and the plan has no untangling policy. ``UNTANGLING_FAILED``: no
    untangling stage was accepted. ``AUDIT_FAILED``: the optimized iterate is
    inverted or fails the mesh audit.
    """

    OPTIMIZED = "optimized"
    VALID_NONCONVERGED = "valid_nonconverged"
    NONCONVERGED = "nonconverged"
    INVERTED_INPUT = "inverted_input"
    UNTANGLING_FAILED = "untangling_failed"
    AUDIT_FAILED = "audit_failed"


_ACCEPTED_STATUSES = frozenset(
    (MeshOptimizationStatus.OPTIMIZED, MeshOptimizationStatus.VALID_NONCONVERGED)
)


_SIMPLEX_KINDS = frozenset(("triangle", "tetrahedron"))


def _corner_frames(kind: str, arity: int, /) -> tuple[np.ndarray, np.ndarray]:
    """Corner routes ``(vertex, neighbors...)`` and their ideal-cell frames.

    Simplex corners share one affine Jacobian, so the first corner represents
    the cell; every other kind keeps each positively oriented vertex frame.
    """

    table = _corner_table(kind, arity)
    count = 1 if kind in _SIMPLEX_KINDS else table.vertices.size
    vertices = table.vertices[:count]
    neighbors = table.neighbors[:count]
    ideal = _ideal_vertices(kind, arity)
    frames = np.swapaxes(ideal[neighbors] - ideal[vertices][:, None, :], -1, -2)
    return np.concatenate((vertices[:, None], neighbors), axis=1), frames


def _corner_matrices(points: Array, corners: Array, /) -> Array:
    """Columns ``x[c_k] - x[c_0]`` for every corner simplex."""
    values = points[corners]
    return jnp.swapaxes(values[:, 1:] - values[:, :1], -1, -2)


class _StarCentroids(StrictModule):
    """Appends polyhedral cell and face centroids after the mesh vertices.

    Star simplices ``(cell centroid, face centroid, edge)`` of polyhedra index
    these appended rows, so their corner Jacobians are linear in the vertices.
    """

    averages: SparseLinearMap

    def __call__(self, coordinates: Array, /) -> Array:
        return jnp.concatenate((coordinates, self.averages.mv(coordinates)), axis=0)


def _augmented(centroids: _StarCentroids | None, coordinates: Array, /) -> Array:
    return coordinates if centroids is None else centroids(coordinates)


class _TargetMatrixEnergy(StrictModule):
    """Target-matrix energy with an inversion guard or Escobar regularization."""

    corners: Array
    centroids: _StarCentroids | None
    target_inverses: Array
    weights: Array
    reference: Array
    displacement_scale: Array
    regularization: Array
    solve_plan: SmallLinearSolvePlan
    objective: MeshQualityObjective = eqx.field(static=True)
    untangling: bool = eqx.field(static=True)

    def regularized(self, delta: float, /) -> _TargetMatrixEnergy:
        """Escobar untangling energy with regularization ``delta``."""
        return _TargetMatrixEnergy(
            corners=self.corners,
            centroids=self.centroids,
            target_inverses=self.target_inverses,
            weights=self.weights,
            reference=self.reference,
            displacement_scale=self.displacement_scale,
            regularization=jnp.asarray(delta, dtype=self.reference.dtype),
            solve_plan=self.solve_plan,
            objective=self.objective,
            untangling=True,
        )

    def _transform(self, coordinates: Array, /) -> Array:
        points = _augmented(self.centroids, coordinates)
        return _corner_matrices(points, self.corners) @ self.target_inverses

    def corner_determinants(self, coordinates: Array, /) -> Array:
        return determinant_small_linear(self.solve_plan, self._transform(coordinates))

    def _corner_quality(self, transform: Array, determinant: Array, /) -> Array:
        dimension = transform.shape[-1]
        frobenius = jnp.sum(transform**2, axis=(-2, -1))
        if self.untangling:
            # Escobar et al. (2003): h(tau) = (tau + sqrt(tau**2 + 4 delta**2)) / 2 is
            # positive for every tau, so inverted corners carry finite energy.
            regularized = 0.5 * (
                determinant + jnp.sqrt(determinant**2 + 4.0 * self.regularization**2)
            )
            return frobenius / (dimension * regularized ** (2.0 / dimension))
        safe = jnp.where(determinant > 0.0, determinant, 1.0)
        match self.objective:
            case MeshQualityObjective.SHAPE:
                return (
                    frobenius ** (0.5 * dimension)
                    / (dimension ** (0.5 * dimension) * safe)
                    - 1.0
                )
            case MeshQualityObjective.SHAPE_SIZE | MeshQualityObjective.METRIC_ALIGNMENT:
                shape = (
                    frobenius ** (0.5 * dimension)
                    / (dimension ** (0.5 * dimension) * safe)
                    - 1.0
                )
                return shape + 0.5 * (safe - 1.0 / safe) ** 2
            case MeshQualityObjective.GRAM_DETERMINANT:
                gram = jnp.swapaxes(transform, -1, -2) @ transform
                identity = jnp.eye(dimension, dtype=transform.dtype)
                return jnp.sum((gram - identity) ** 2, axis=(-2, -1)) + (safe - 1.0) ** 2
            case _:
                raise ValueError(
                    f"Unsupported mesh quality objective {self.objective!r}."
                )

    def __call__(self, coordinates: Array, /) -> Array:
        transform = self._transform(coordinates)
        determinant = determinant_small_linear(self.solve_plan, transform)
        quality = jnp.sum(self.weights * self._corner_quality(transform, determinant))
        displacement = self.displacement_scale * jnp.sum(
            (coordinates - self.reference) ** 2
        )
        if self.untangling:
            return quality
        return jnp.where(jnp.all(determinant > 0.0), quality + displacement, jnp.inf)


class _FreeCoordinateArguments(StrictModule):
    energy: Callable[[Array], Array]
    base: Array
    free_rows: Array


def _free_objective(free: Array, arguments: _FreeCoordinateArguments, /) -> Array:
    coordinates = arguments.base.at[arguments.free_rows].set(free)
    return arguments.energy(coordinates)


@eqx.filter_jit
def _minimize_free_coordinates(
    energy: Callable[[Array], Array],
    method: ProjectedLBFGS | NewtonTrustRegion,
    termination: OptimizationTermination,
    base: Array,
    free_rows: Array,
    lower: Array,
    upper: Array,
    /,
) -> tuple[Array, MinimizationResult]:
    arguments = _FreeCoordinateArguments(energy, base, free_rows)
    initial = base[free_rows]
    match method:
        case ProjectedLBFGS():
            # Bounds are built from dynamic leaves so bound values never enter
            # the compiled cache key.
            result = minimize(
                _free_objective,
                initial,
                method=method,
                termination=termination,
                args=arguments,
                bounds=Bounds(lower, upper),
            )
        case NewtonTrustRegion():
            result = minimize(
                _free_objective,
                initial,
                method=method,
                termination=termination,
                args=arguments,
            )
        case _:
            raise TypeError("method must be ProjectedLBFGS or NewtonTrustRegion.")
    return base.at[free_rows].set(result.parameters), result


@eqx.filter_jit
def _evaluate_energy(energy: _TargetMatrixEnergy, coordinates: Array, /) -> Any:
    return energy(coordinates), energy.corner_determinants(coordinates)


def _method(method: ProjectedLBFGS | NewtonTrustRegion | None, /) -> Any:
    resolved = ProjectedLBFGS() if method is None else method
    if not isinstance(resolved, (ProjectedLBFGS, NewtonTrustRegion)):
        raise TypeError("method must be ProjectedLBFGS, NewtonTrustRegion, or None.")
    return resolved


def _termination(termination: OptimizationTermination | None, /) -> Any:
    resolved = (
        OptimizationTermination(maximum_steps=100) if termination is None else termination
    )
    if not isinstance(resolved, OptimizationTermination):
        raise TypeError("termination must be OptimizationTermination or None.")
    return resolved


def _fixed_mask(fixed: ArrayLike | None, rows: int, name: str, /) -> np.ndarray:
    mask = (
        np.zeros((rows,), dtype=np.bool_)
        if fixed is None
        else np.asarray(fixed, dtype=np.bool_)
    )
    if mask.shape != (rows,):
        raise ValueError(f"{name} must match the coordinate row count.")
    if np.all(mask):
        raise ValueError(f"{name} leaves no free coordinates to optimize.")
    return mask


class MeshUntanglingPolicy(StrictModule, NonTrainableState):
    """Explicit Escobar untangling stages run only when inversions are present.

    Each stage minimizes the regularized shape energy with
    ``delta = sqrt(epsilon (epsilon - tau_min))`` from the current smallest
    corner determinant ``tau_min``. A stage is accepted only when the inversion
    count reaches zero, the mesh audit passes, and its minimization converged or
    the plan accepts valid non-convergence; otherwise the next stage continues
    from its iterate.
    """

    maximum_stages: int = eqx.field(static=True)
    regularization_threshold: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self, *, maximum_stages: int = 4, regularization_threshold: float = 1.0e-3
    ) -> None:
        if not isinstance(maximum_stages, (int, np.integer)) or maximum_stages <= 0:
            raise ValueError("maximum_stages must be a positive integer.")
        threshold = finite_real_scalar(
            regularization_threshold, "regularization_threshold"
        )
        if threshold <= 0.0:
            raise ValueError("regularization_threshold must be positive.")
        self.maximum_stages = int(maximum_stages)
        self.regularization_threshold = threshold
        self.policy_id = canonical_fingerprint(
            {
                "kind": "mesh-untangling-policy",
                "maximum_stages": int(maximum_stages),
                "regularization_threshold": threshold,
            }
        )


class MeshUntanglingEvidence(StrictModule, NonTrainableState):
    """Native evidence of every untangling stage and the final-stage verdict.

    ``optimizer_statuses`` are the native termination statuses of the stages in
    order. ``final_inverted_count`` and ``audit_passed`` describe the last stage;
    ``succeeded`` requires that stage to be inversion free, audited, and
    converged, or valid under a plan that accepts valid non-convergence.
    """

    minimizations: tuple[MinimizationResult, ...]
    regularizations: tuple[float, ...] = eqx.field(static=True)
    optimizer_statuses: tuple[OptimizationStatus, ...] = eqx.field(static=True)
    initial_inverted_count: int = eqx.field(static=True)
    final_inverted_count: int = eqx.field(static=True)
    audit_passed: bool = eqx.field(static=True)
    accept_valid_nonconverged: bool = eqx.field(static=True)

    def __init__(
        self,
        minimizations: tuple[MinimizationResult, ...],
        regularizations: tuple[float, ...],
        /,
        *,
        initial_inverted_count: int,
        final_inverted_count: int,
        audit_passed: bool,
        accept_valid_nonconverged: bool,
    ) -> None:
        if not minimizations or len(minimizations) != len(regularizations):
            raise ValueError("Every untangling stage needs one minimization and delta.")
        self.minimizations = tuple(minimizations)
        self.regularizations = tuple(float(value) for value in regularizations)
        self.optimizer_statuses = tuple(
            OptimizationStatus(int(np.asarray(value.status))) for value in minimizations
        )
        self.initial_inverted_count = int(initial_inverted_count)
        self.final_inverted_count = int(final_inverted_count)
        self.audit_passed = bool(audit_passed)
        self.accept_valid_nonconverged = bool(accept_valid_nonconverged)

    @property
    def converged(self) -> bool:
        """Whether the final stage's native minimization converged."""
        return self.optimizer_statuses[-1] is OptimizationStatus.SUCCESS

    @property
    def succeeded(self) -> bool:
        return (
            self.final_inverted_count == 0
            and self.audit_passed
            and (self.converged or self.accept_valid_nonconverged)
        )


class TargetMatrixOptimizationPlan(StrictModule, NonTrainableState):
    """Prepared corner routes, targets, fixed rows, bounds, and solver controls.

    Targets come from ``target_coordinates`` (default: the mesh itself) or, for
    ``METRIC_ALIGNMENT``, from the log-Euclidean cell mean of a vertex
    :class:`MeshMetricField`. ``coordinate_bounds`` is a broadcastable
    ``(lower, upper)`` box enforced by projection; fixed vertices are eliminated
    from the parameters. ``displacement_weight`` scales
    ``sum |x - x_target|**2 / L**2`` with ``L`` the mean target corner size.
    ``accept_valid_nonconverged`` explicitly admits an inversion-free, audited
    iterate whose native minimization stopped without converging (status
    ``VALID_NONCONVERGED``, also for untangling stages); by default such an
    iterate is ``NONCONVERGED`` and is not committed.
    """

    mesh: CellMesh
    energy: _TargetMatrixEnergy
    fixed_vertices: Array
    lower: Array
    upper: Array
    method: ProjectedLBFGS | NewtonTrustRegion
    termination: OptimizationTermination
    untangling: MeshUntanglingPolicy | None
    audit_policy: CellMeshAuditPolicy
    objective: MeshQualityObjective = eqx.field(static=True)
    accept_valid_nonconverged: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        /,
        *,
        objective: MeshQualityObjective = MeshQualityObjective.SHAPE_SIZE,
        target_coordinates: ArrayLike | None = None,
        metric: MeshMetricField | None = None,
        fixed_vertices: ArrayLike | None = None,
        coordinate_bounds: tuple[ArrayLike, ArrayLike] | None = None,
        displacement_weight: float = 0.0,
        method: ProjectedLBFGS | NewtonTrustRegion | None = None,
        termination: OptimizationTermination | None = None,
        untangling: MeshUntanglingPolicy | None = MeshUntanglingPolicy(),
        audit_policy: CellMeshAuditPolicy | None = None,
        accept_valid_nonconverged: bool = False,
    ) -> None:
        if not isinstance(mesh, CellMesh):
            raise TypeError("mesh must be CellMesh.")
        if not isinstance(objective, MeshQualityObjective):
            raise TypeError("objective must be MeshQualityObjective.")
        if untangling is not None and not isinstance(untangling, MeshUntanglingPolicy):
            raise TypeError("untangling must be MeshUntanglingPolicy or None.")
        audit = CellMeshAuditPolicy() if audit_policy is None else audit_policy
        if not isinstance(audit, CellMeshAuditPolicy):
            raise TypeError("audit_policy must be CellMeshAuditPolicy or None.")
        if not isinstance(accept_valid_nonconverged, bool):
            raise TypeError("accept_valid_nonconverged must be bool.")
        method_ = _method(method)
        termination_ = _termination(termination)
        rows, dimension = mesh.coordinates.shape
        target = np.asarray(
            mesh.coordinates if target_coordinates is None else target_coordinates,
            dtype=np.float64,
        )
        if target.shape != (rows, dimension) or not np.all(np.isfinite(target)):
            raise ValueError("target_coordinates must match the finite mesh coordinates.")
        if (metric is None) != (objective is not MeshQualityObjective.METRIC_ALIGNMENT):
            raise ValueError("A metric is required exactly for METRIC_ALIGNMENT.")
        weight = finite_real_scalar(displacement_weight, "displacement_weight")
        if weight < 0.0:
            raise ValueError("displacement_weight must be non-negative.")
        fixed = _fixed_mask(fixed_vertices, rows, "fixed_vertices")
        lower, upper = _coordinate_box(coordinate_bounds, np.asarray(mesh.coordinates))
        if isinstance(method_, NewtonTrustRegion) and (
            np.any(np.isfinite(lower[~fixed])) or np.any(np.isfinite(upper[~fixed]))
        ):
            raise ValueError("NewtonTrustRegion cannot enforce coordinate bounds.")
        solve_plan = SmallLinearSolvePlan(dimension)
        corners, targets, centroids = _corner_targets(mesh, target, metric, solve_plan)
        inverse = inverse_small_linear(solve_plan, jnp.asarray(targets))
        if not np.all(np.asarray(inverse.successful)):
            raise ValueError("Target corner matrices must be well conditioned.")
        measures = np.asarray(inverse.determinant)
        scale = float(np.mean(measures ** (1.0 / dimension)))
        self.mesh = mesh
        self.energy = _TargetMatrixEnergy(
            corners=jnp.asarray(corners, dtype=jnp.int32),
            centroids=centroids,
            target_inverses=inverse.value,
            weights=jnp.asarray(measures / np.sum(measures)),
            reference=jnp.asarray(target),
            displacement_scale=jnp.asarray(weight / scale**2),
            regularization=jnp.asarray(0.0),
            solve_plan=solve_plan,
            objective=objective,
            untangling=False,
        )
        self.fixed_vertices = jnp.asarray(fixed)
        self.lower = jnp.asarray(lower)
        self.upper = jnp.asarray(upper)
        self.method = method_
        self.termination = termination_
        self.untangling = untangling
        self.audit_policy = audit
        self.objective = objective
        self.accept_valid_nonconverged = accept_valid_nonconverged
        self.plan_id = canonical_fingerprint(
            {
                "kind": "target-matrix-optimization-plan",
                "mesh": mesh.mesh_id,
                "objective": objective.value,
                "target_coordinates": array_tree_fingerprint(target),
                "metric": None if metric is None else metric.metric_id,
                "fixed_vertices": array_tree_fingerprint(fixed),
                "lower": array_tree_fingerprint(lower),
                "upper": array_tree_fingerprint(upper),
                "displacement_weight": weight,
                "method": method_.method_id,
                "termination": {
                    "absolute_optimality": termination_.absolute_optimality,
                    "relative_optimality": termination_.relative_optimality,
                    "absolute_step": termination_.absolute_step,
                    "relative_step": termination_.relative_step,
                    "maximum_steps": termination_.maximum_steps,
                    "maximum_evaluations": termination_.maximum_evaluations,
                },
                "untangling": None if untangling is None else untangling.policy_id,
                "audit": audit.policy_id,
                "accept_valid_nonconverged": accept_valid_nonconverged,
            }
        )

    def evaluate(self, coordinates: ArrayLike, /) -> Array:
        """Objective value; infinite when any corner is inverted."""
        points = jnp.asarray(coordinates, dtype=self.energy.reference.dtype)
        if points.shape != self.energy.reference.shape:
            raise ValueError("coordinates must match the mesh coordinate shape.")
        return _evaluate_energy(self.energy, points)[0]


def _coordinate_box(
    bounds: tuple[ArrayLike, ArrayLike] | None, coordinates: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    if bounds is None:
        infinite = np.full(coordinates.shape, np.inf)
        return -infinite, infinite
    if len(bounds) != 2:
        raise ValueError("coordinate_bounds must be one (lower, upper) pair.")
    lower = np.broadcast_to(
        np.asarray(bounds[0], dtype=np.float64), coordinates.shape
    ).copy()
    upper = np.broadcast_to(
        np.asarray(bounds[1], dtype=np.float64), coordinates.shape
    ).copy()
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)) or np.any(lower > upper):
        raise ValueError("coordinate_bounds must be ordered and not NaN.")
    return lower, upper


def _metric_rows(mesh: CellMesh, metric: MeshMetricField, /) -> np.ndarray:
    scope = metric.scope
    vertices = mesh.entity_set(0)
    if scope.entity_dimension != 0 or scope.entity_set_id != vertices.entity_set_id:
        raise ValueError("The alignment metric must be bound to the mesh vertices.")
    identifiers = np.asarray(scope.entity_ids, dtype=np.int64)
    order = np.argsort(identifiers)
    wanted = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    positions = np.searchsorted(identifiers[order], wanted)
    positions = np.minimum(positions, identifiers.size - 1)
    if not np.array_equal(identifiers[order[positions]], wanted):
        raise ValueError("The alignment metric must cover every mesh vertex.")
    return order[positions]


def _cell_inverse_roots(metric_values: Array, vertices: np.ndarray, /) -> Array:
    """``M**(-1/2)`` of the log-Euclidean mean metric of every cell."""

    cell_metric = interpolate_mesh_metric(
        metric_values[vertices],
        jnp.full(vertices.shape, 1.0 / vertices.shape[1]),
    )
    inverse_root = hermitian_inverse_sqrt(cell_metric)
    if not np.all(np.asarray(inverse_root.valid)):
        raise ValueError("Cell metrics must be symmetric positive definite.")
    return inverse_root.value


def _polyhedral_stars(
    mesh: CellMesh, cells: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, _StarCentroids]:
    """Star-simplex routes of polyhedral ``cells`` into appended centroid rows."""

    # ty: ignore[invalid-argument-type]
    tables = polyhedral_star_tables(mesh.connectivity)
    selected = np.isin(tables.star_cell, cells)
    star_cell = tables.star_cell[selected]
    star_face = tables.star_face[selected]
    cell_rows = np.unique(star_cell)
    face_rows = np.unique(star_face)
    cell_entries = np.isin(tables.cell_vertex_cell, cell_rows)
    face_entries = np.isin(tables.face_corner_face, face_rows)
    member_cells = tables.cell_vertex_cell[cell_entries]
    member_faces = tables.face_corner_face[face_entries]
    count = mesh.coordinates.shape[0]
    averages = SparseLinearMap(
        EdgeRelation(
            np.concatenate(
                (
                    tables.cell_vertex_values[cell_entries],
                    tables.face_corner_vertex[face_entries],
                )
            ),
            np.concatenate(
                (
                    np.searchsorted(cell_rows, member_cells),
                    cell_rows.size + np.searchsorted(face_rows, member_faces),
                )
            ),
            source_size=count,
            target_size=cell_rows.size + face_rows.size,
        ),
        np.concatenate(
            (
                1.0 / tables.cell_vertex_counts[member_cells],
                1.0 / tables.face_sizes[member_faces],
            )
        ),
    )
    corners = np.stack(
        (
            count + np.searchsorted(cell_rows, star_cell),
            count + cell_rows.size + np.searchsorted(face_rows, star_face),
            tables.star_first[selected],
            tables.star_second[selected],
        ),
        axis=1,
    )
    return corners, star_cell, _StarCentroids(averages)


def _corner_targets(
    mesh: CellMesh,
    target: np.ndarray,
    metric: MeshMetricField | None,
    solve_plan: SmallLinearSolvePlan,
    /,
) -> tuple[np.ndarray, np.ndarray, _StarCentroids | None]:
    dimension = mesh.coordinates.shape[1]
    metric_values = (
        None if metric is None else jnp.asarray(metric.values)[_metric_rows(mesh, metric)]
    )
    points = jnp.asarray(target)
    corners = []
    targets = []
    polyhedral_cells = []
    polyhedral_roots = []
    offset = 0
    for block in mesh.blocks:
        if block.topological_dimension != dimension:
            raise ValueError(
                "Target-matrix optimization requires full-dimensional cells."
            )
        vertices = np.asarray(block.vertices, dtype=np.int64)
        cell_count = vertices.shape[0]
        if block.cell_kind == "polyhedron":
            polyhedral_cells.append(np.arange(offset, offset + cell_count))
            if metric_values is not None:
                polyhedral_roots.append(_cell_inverse_roots(metric_values, vertices))
            offset += cell_count
            continue
        routes, ideal = _corner_frames(block.cell_kind, block.arity)
        block_corners = vertices[:, routes].reshape(-1, routes.shape[1])
        corners.append(block_corners)
        if metric_values is None:
            # ty: ignore[invalid-argument-type]
            targets.append(np.asarray(_corner_matrices(points, block_corners)))
        else:
            # Log-Euclidean cell metric; every corner of a cell shares it.
            roots = _cell_inverse_roots(metric_values, vertices)
            targets.append(
                np.asarray(roots[:, None] @ jnp.asarray(ideal)[None]).reshape(
                    -1, dimension, dimension
                )
            )
        offset += cell_count
    centroids = None
    if polyhedral_cells:
        cells = np.concatenate(polyhedral_cells)
        star_corners, star_cells, centroids = _polyhedral_stars(mesh, cells)
        corners.append(star_corners)
        star_targets = np.asarray(
            _corner_matrices(centroids(points), jnp.asarray(star_corners))
        )
        if metric_values is not None:
            # Metric size and alignment with the target star shape.
            measure = np.asarray(
                determinant_small_linear(solve_plan, jnp.asarray(star_targets))
            )
            shape = star_targets / np.cbrt(measure)[:, None, None]
            roots = jnp.concatenate(polyhedral_roots)[np.searchsorted(cells, star_cells)]
            star_targets = np.asarray(roots @ jnp.asarray(shape))
        targets.append(star_targets)
    targets_ = np.concatenate(targets, axis=0)
    determinants = np.asarray(determinant_small_linear(solve_plan, jnp.asarray(targets_)))
    if not np.all(np.isfinite(determinants)) or np.any(determinants <= 0.0):
        raise ValueError("Target corner matrices must be positively oriented.")
    return np.concatenate(corners, axis=0), targets_, centroids


class MeshOptimizationResult(StrictModule, NonTrainableState):
    """Certified optimized mesh or an explicit failure that leaves the mesh intact.

    ``result`` is the certified mesh exactly for the accepted statuses
    (``OPTIMIZED`` and ``VALID_NONCONVERGED``) and ``coordinates`` are then the
    optimized coordinates; every other status carries no result and the
    unmodified input coordinates. ``minimization`` carries the native termination
    evidence (including the rejected iterate of a ``NONCONVERGED`` optimization)
    and ``optimizer_status`` its portable status.
    """

    status: MeshOptimizationStatus = eqx.field(static=True)
    result: CellMeshingResult | None
    coordinates: Array
    minimization: MinimizationResult | None
    untangling: MeshUntanglingEvidence | None
    initial_objective: float = eqx.field(static=True)
    final_objective: float = eqx.field(static=True)
    iterations: int = eqx.field(static=True)
    accepted_steps: int = eqx.field(static=True)
    optimizer_status: OptimizationStatus | None = eqx.field(static=True)
    inverted_count: int = eqx.field(static=True)
    optimization_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: TargetMatrixOptimizationPlan,
        status: MeshOptimizationStatus,
        coordinates: ArrayLike,
        /,
        *,
        result: CellMeshingResult | None,
        minimization: MinimizationResult | None,
        untangling: MeshUntanglingEvidence | None,
        initial_objective: float,
        final_objective: float,
        inverted_count: int,
    ) -> None:
        if not isinstance(status, MeshOptimizationStatus):
            raise TypeError("status must be MeshOptimizationStatus.")
        if (status in _ACCEPTED_STATUSES) != (result is not None):
            raise ValueError("Exactly the accepted statuses carry a certified result.")
        optimizer_status = (
            None
            if minimization is None
            else OptimizationStatus(int(np.asarray(minimization.status)))
        )
        converged = optimizer_status is OptimizationStatus.SUCCESS
        match status:
            case MeshOptimizationStatus.OPTIMIZED:
                if not converged:
                    raise ValueError("An optimized status requires a converged solve.")
            case (
                MeshOptimizationStatus.VALID_NONCONVERGED
                | MeshOptimizationStatus.NONCONVERGED
            ):
                if optimizer_status is None or converged:
                    raise ValueError(
                        "A non-converged status requires a non-converged solve."
                    )
                if (
                    status is MeshOptimizationStatus.VALID_NONCONVERGED
                ) != plan.accept_valid_nonconverged:
                    raise ValueError(
                        "Valid non-convergence is accepted exactly when the plan "
                        "permits it."
                    )
            case (
                MeshOptimizationStatus.INVERTED_INPUT
                | MeshOptimizationStatus.UNTANGLING_FAILED
                | MeshOptimizationStatus.AUDIT_FAILED
            ):
                pass
            case _:
                raise ValueError(f"Unsupported mesh optimization status {status!r}.")
        values = np.asarray(coordinates, dtype=np.float64)
        self.status = status
        self.result = result
        self.coordinates = jnp.asarray(values)
        self.minimization = minimization
        self.untangling = untangling
        self.initial_objective = float(initial_objective)
        self.final_objective = float(final_objective)
        if minimization is None:
            self.iterations = 0
            self.accepted_steps = 0
        else:
            diagnostics = minimization.diagnostics
            self.iterations = int(np.asarray(diagnostics.iterations))
            self.accepted_steps = int(np.asarray(diagnostics.accepted_steps))
        self.optimizer_status = optimizer_status
        self.inverted_count = int(inverted_count)
        self.optimization_id = canonical_fingerprint(
            {
                "kind": "mesh-optimization-result",
                "plan": plan.plan_id,
                "status": status.value,
                "result": None if result is None else result.result_id,
                "coordinates": array_tree_fingerprint(values),
                "initial_objective": repr(self.initial_objective),
                "final_objective": repr(self.final_objective),
                "iterations": self.iterations,
                "accepted_steps": self.accepted_steps,
                "optimizer_status": None
                if self.optimizer_status is None
                else int(self.optimizer_status),
                "inverted_count": self.inverted_count,
            }
        )

    @property
    def accepted(self) -> bool:
        """Whether ``result`` is a certified mesh the caller may commit."""
        return self.status in _ACCEPTED_STATUSES


def _inverted_count(
    plan: TargetMatrixOptimizationPlan, coordinates: Array, determinants: Array, /
) -> int:
    corners = int(np.count_nonzero(np.asarray(determinants) <= 0.0))
    cells = int(
        np.count_nonzero(
            ~np.asarray(evaluate_cell_quality(plan.mesh, coordinates).sampled_valid)
        )
    )
    return max(corners, cells)


def _audit_passed(
    plan: TargetMatrixOptimizationPlan, coordinates: Array, numeric_version: str, /
) -> tuple[bool, CellMesh]:
    candidate = plan.mesh.with_coordinates(coordinates, numeric_version=numeric_version)
    audit = audit_cell_mesh(
        candidate,
        CellGeometrySpec.affine(candidate),
        evaluate_cell_quality(candidate),
        policy=plan.audit_policy,
    )
    return audit.passed, candidate


def _untangle(
    plan: TargetMatrixOptimizationPlan,
    policy: MeshUntanglingPolicy,
    coordinates: Array,
    determinants: Array,
    initial_count: int,
    numeric_version: str,
    /,
) -> tuple[Array, MeshUntanglingEvidence]:
    free = jnp.asarray(np.flatnonzero(~np.asarray(plan.fixed_vertices)), dtype=jnp.int32)
    epsilon = policy.regularization_threshold
    minimizations = []
    regularizations = []
    count = initial_count
    audit_passed = False
    for _ in range(policy.maximum_stages):
        minimum = float(np.min(np.asarray(determinants)))
        delta = math.sqrt(epsilon * (epsilon - minimum)) if minimum < epsilon else 0.0
        coordinates, minimization = _minimize_free_coordinates(
            plan.energy.regularized(delta),
            plan.method,
            plan.termination,
            coordinates,
            free,
            plan.lower[free],
            plan.upper[free],
        )
        minimizations.append(minimization)
        regularizations.append(delta)
        _, determinants = _evaluate_energy(plan.energy, coordinates)
        count = _inverted_count(plan, coordinates, determinants)
        if count:
            continue
        audit_passed = _audit_passed(plan, coordinates, numeric_version)[0]
        # A valid stage that did not converge is not accepted unless the plan
        # permits it; an unaudited or refused iterate continues to the next
        # regularization stage.
        if audit_passed and (
            plan.accept_valid_nonconverged or bool(np.asarray(minimization.successful))
        ):
            break
    return coordinates, MeshUntanglingEvidence(
        tuple(minimizations),
        tuple(regularizations),
        initial_inverted_count=initial_count,
        final_inverted_count=count,
        audit_passed=count == 0 and audit_passed,
        accept_valid_nonconverged=plan.accept_valid_nonconverged,
    )


def optimize_cell_mesh(
    plan: TargetMatrixOptimizationPlan,
    coordinate_contract: SpatialCoordinateContract,
    /,
    *,
    numeric_version: str = "mesh-optimized",
) -> MeshOptimizationResult:
    """Untangle when needed, optimize free coordinates, audit, and certify.

    Inverted input fails as ``INVERTED_INPUT`` without an untangling policy and
    as ``UNTANGLING_FAILED`` when no stage is accepted. The optimized iterate is
    then checked for inversions and audited independently of the native
    termination status: an inverted or unaudited iterate returns
    ``AUDIT_FAILED``; a valid iterate is ``OPTIMIZED`` when the minimization
    converged, ``VALID_NONCONVERGED`` when it did not and the plan accepts valid
    non-convergence, and ``NONCONVERGED`` otherwise. Every non-accepted status
    returns the unmodified input coordinates.
    """
    if not isinstance(plan, TargetMatrixOptimizationPlan):
        raise TypeError("plan must be TargetMatrixOptimizationPlan.")
    if not isinstance(coordinate_contract, SpatialCoordinateContract):
        raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
    original = jnp.asarray(plan.mesh.coordinates, dtype=plan.energy.reference.dtype)
    initial_value, determinants = _evaluate_energy(plan.energy, original)
    initial = float(np.asarray(initial_value))
    inverted = _inverted_count(plan, original, determinants)

    def failure(status: Any, count: Any, minimization: Any, untangling: Any) -> Any:
        return MeshOptimizationResult(
            plan,
            status,
            original,
            result=None,
            minimization=minimization,
            untangling=untangling,
            initial_objective=initial,
            final_objective=initial,
            inverted_count=count,
        )

    coordinates = original
    untangling = None
    if inverted:
        if plan.untangling is None:
            return failure(MeshOptimizationStatus.INVERTED_INPUT, inverted, None, None)
        coordinates, untangling = _untangle(
            plan,
            plan.untangling,
            original,
            determinants,
            inverted,
            numeric_version,
        )
        if not untangling.succeeded:
            return failure(
                MeshOptimizationStatus.UNTANGLING_FAILED,
                untangling.final_inverted_count,
                None,
                untangling,
            )
    free = jnp.asarray(np.flatnonzero(~np.asarray(plan.fixed_vertices)), dtype=jnp.int32)
    optimized, minimization = _minimize_free_coordinates(
        plan.energy,
        plan.method,
        plan.termination,
        coordinates,
        free,
        plan.lower[free],
        plan.upper[free],
    )
    final_value, final_determinants = _evaluate_energy(plan.energy, optimized)
    converged = bool(np.asarray(minimization.successful))
    count = _inverted_count(plan, optimized, final_determinants)
    passed, candidate = _audit_passed(plan, optimized, numeric_version)
    if count or not passed:
        return failure(
            MeshOptimizationStatus.AUDIT_FAILED, count, minimization, untangling
        )
    if not (converged or plan.accept_valid_nonconverged):
        return failure(MeshOptimizationStatus.NONCONVERGED, 0, minimization, untangling)
    result = certify_cell_mesh(
        candidate, coordinate_contract, audit_policy=plan.audit_policy
    )
    return MeshOptimizationResult(
        plan,
        MeshOptimizationStatus.OPTIMIZED
        if converged
        else MeshOptimizationStatus.VALID_NONCONVERGED,
        optimized,
        result=result,
        minimization=minimization,
        untangling=untangling,
        initial_objective=initial,
        final_objective=float(np.asarray(final_value)),
        inverted_count=0,
    )


class CellGeometryOptimizationResult(StrictModule):
    """Optimized geometry coordinates with native termination evidence.

    ``coordinates`` are the final iterate whether or not the minimization
    converged; ``optimizer_status`` and ``converged`` report the native
    termination, and consumers decide explicitly whether to accept a
    non-converged iterate.
    """

    coordinates: Array
    minimization: MinimizationResult
    optimizer_status: OptimizationStatus = eqx.field(static=True)

    def __init__(self, coordinates: Array, minimization: MinimizationResult, /) -> None:
        if not isinstance(minimization, MinimizationResult):
            raise TypeError("minimization must be MinimizationResult.")
        self.coordinates = coordinates
        self.minimization = minimization
        self.optimizer_status = OptimizationStatus(int(np.asarray(minimization.status)))

    @property
    def converged(self) -> bool:
        """Whether the native minimization terminated successfully."""
        return self.optimizer_status is OptimizationStatus.SUCCESS


def optimize_cell_geometry_coordinates(
    geometry: CellGeometrySpec,
    objective: Callable[[Array], Array],
    /,
    *,
    fixed_coordinates: ArrayLike | None = None,
    coordinate_bounds: tuple[ArrayLike, ArrayLike] | None = None,
    method: ProjectedLBFGS | NewtonTrustRegion | None = None,
    termination: OptimizationTermination | None = None,
) -> CellGeometryOptimizationResult:
    """Optimize arbitrary (high-order) geometry coordinates under fixed topology.

    ``objective`` maps the full coordinate array to a scalar and must encode the
    required curved-element validity (return ``inf`` for inadmissible states).
    Fixed rows are eliminated and stay bit-identical.
    """
    if not isinstance(geometry, CellGeometrySpec):
        raise TypeError("geometry must be CellGeometrySpec.")
    if not callable(objective):
        raise TypeError("objective must be callable.")
    coordinates = jnp.asarray(geometry.coordinates)
    if not jnp.issubdtype(coordinates.dtype, jnp.floating):
        coordinates = coordinates.astype(jnp.float64)
    fixed = _fixed_mask(fixed_coordinates, coordinates.shape[0], "fixed_coordinates")
    lower, upper = _coordinate_box(coordinate_bounds, np.asarray(coordinates))
    method_ = _method(method)
    if isinstance(method_, NewtonTrustRegion) and (
        np.any(np.isfinite(lower[~fixed])) or np.any(np.isfinite(upper[~fixed]))
    ):
        raise ValueError("NewtonTrustRegion cannot enforce coordinate bounds.")
    free = np.flatnonzero(~fixed)
    optimized, minimization = _minimize_free_coordinates(
        objective,
        method_,
        _termination(termination),
        coordinates,
        jnp.asarray(free, dtype=jnp.int32),
        jnp.asarray(lower[free], dtype=coordinates.dtype),
        jnp.asarray(upper[free], dtype=coordinates.dtype),
    )
    return CellGeometryOptimizationResult(optimized, minimization)


__all__ = [
    "CellGeometryOptimizationResult",
    "MeshOptimizationResult",
    "MeshOptimizationStatus",
    "MeshQualityObjective",
    "MeshUntanglingEvidence",
    "MeshUntanglingPolicy",
    "TargetMatrixOptimizationPlan",
    "optimize_cell_geometry_coordinates",
    "optimize_cell_mesh",
]
