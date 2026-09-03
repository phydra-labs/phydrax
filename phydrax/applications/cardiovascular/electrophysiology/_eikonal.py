#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-topology anisotropic cardiac eikonal routes.

The graph and finite-element routes are deliberately distinct types. The graph
route uses immutable min-plus edge relaxation. The finite-element route adds
causal affine-simplex roots of ``grad(T)^T C grad(T) = 1`` with edge updates as
the obtuse-simplex fallback. Differentiation is only qualified while the
prepared topology and selected update paths remain fixed.
"""

from __future__ import annotations

from enum import IntFlag
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....linalg import SmallLinearSolvePlan, solve_small_linear


class EikonalSolveStatus(IntFlag):
    """Fail-closed bit field for one arrival-time solve."""

    SUCCESS = 0
    NONFINITE = 1
    NOT_CONVERGED = 2
    UNREACHABLE = 4
    PATH_TIE = 8


class GraphEikonalRoute(StrictModule, NonTrainableState):
    """An immutable undirected graph route with stable node and edge IDs."""

    node_ids: Array
    edge_ids: Array
    edge_nodes: Array
    maximum_sweeps: int = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def __init__(
        self,
        node_ids: ArrayLike,
        edge_ids: ArrayLike,
        edge_nodes: ArrayLike,
        /,
        *,
        maximum_sweeps: int | None = None,
    ):
        nodes = np.asarray(node_ids, dtype=np.int64)
        edges = np.asarray(edge_ids, dtype=np.int64)
        incidence = np.asarray(edge_nodes, dtype=np.int32)
        if nodes.ndim != 1 or nodes.size < 2:
            raise ValueError("node_ids must contain at least two stable IDs.")
        if np.any(nodes < 0) or np.unique(nodes).size != nodes.size:
            raise ValueError("node_ids must be unique nonnegative integers.")
        if incidence.ndim != 2 or incidence.shape[1] != 2 or incidence.shape[0] == 0:
            raise ValueError("edge_nodes must have non-empty shape [edge, 2].")
        if edges.shape != (incidence.shape[0],):
            raise ValueError("edge_ids must have one entry per graph edge.")
        if np.any(edges < 0) or np.unique(edges).size != edges.size:
            raise ValueError("edge_ids must be unique nonnegative integers.")
        if np.any(incidence < 0) or np.any(incidence >= nodes.size):
            raise ValueError("edge_nodes contains an index outside node capacity.")
        if np.any(incidence[:, 0] == incidence[:, 1]):
            raise ValueError("Eikonal graph edges may not be self loops.")
        canonical_edges = np.sort(incidence, axis=1)
        if np.unique(canonical_edges, axis=0).shape[0] != incidence.shape[0]:
            raise ValueError("Eikonal graph edges must be unique as undirected pairs.")
        sweeps = nodes.size if maximum_sweeps is None else int(maximum_sweeps)
        if sweeps < nodes.size:
            raise ValueError("maximum_sweeps must be at least the graph node count.")
        self.node_ids = jnp.asarray(nodes)
        self.edge_ids = jnp.asarray(edges)
        self.edge_nodes = jnp.asarray(incidence)
        self.maximum_sweeps = sweeps
        self.route_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-graph-eikonal-route",
                "arrays": array_tree_fingerprint((nodes, edges, incidence)),
                "maximum_sweeps": sweeps,
            }
        )


class FiniteElementEikonalRoute(StrictModule, NonTrainableState):
    """A fixed simplex mesh route whose complete element edge stencil is prepared."""

    node_ids: Array
    element_ids: Array
    elements: Array
    maximum_sweeps: int = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def __init__(
        self,
        node_ids: ArrayLike,
        element_ids: ArrayLike,
        elements: ArrayLike,
        /,
        *,
        maximum_sweeps: int | None = None,
    ):
        nodes = np.asarray(node_ids, dtype=np.int64)
        element_id_values = np.asarray(element_ids, dtype=np.int64)
        cells = np.asarray(elements, dtype=np.int32)
        if nodes.ndim != 1 or nodes.size < 2:
            raise ValueError("node_ids must contain at least two stable IDs.")
        if np.any(nodes < 0) or np.unique(nodes).size != nodes.size:
            raise ValueError("node_ids must be unique nonnegative integers.")
        if cells.ndim != 2 or cells.shape[0] == 0 or cells.shape[1] not in (2, 3, 4):
            raise ValueError(
                "elements must be non-empty line, triangle, or tetrahedron incidence."
            )
        if element_id_values.shape != (cells.shape[0],):
            raise ValueError("element_ids must have one entry per element.")
        if (
            np.any(element_id_values < 0)
            or np.unique(element_id_values).size != cells.shape[0]
        ):
            raise ValueError("element_ids must be unique nonnegative integers.")
        if np.any(cells < 0) or np.any(cells >= nodes.size):
            raise ValueError("elements contains an index outside node capacity.")
        if any(np.unique(cell).size != cell.size for cell in cells):
            raise ValueError("Every finite element must contain distinct node indices.")
        sweeps = nodes.size if maximum_sweeps is None else int(maximum_sweeps)
        if sweeps < nodes.size:
            raise ValueError("maximum_sweeps must be at least the mesh node count.")
        self.node_ids = jnp.asarray(nodes)
        self.element_ids = jnp.asarray(element_id_values)
        self.elements = jnp.asarray(cells)
        self.maximum_sweeps = sweeps
        self.route_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-fem-eikonal-route",
                "arrays": array_tree_fingerprint((nodes, element_id_values, cells)),
                "maximum_sweeps": sweeps,
            }
        )


EikonalRoute = GraphEikonalRoute | FiniteElementEikonalRoute


class AnisotropicEikonalPlan(StrictModule, NonTrainableState):
    """Geometry and squared-velocity tensors in the mm/ms kernel convention."""

    route: EikonalRoute
    node_positions_mm: Array
    velocity_tensors_mm2_per_ms2: Array
    residual_tolerance_ms: float = eqx.field(static=True)
    path_tie_tolerance_ms: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        route: EikonalRoute,
        node_positions_mm: ArrayLike,
        velocity_tensors_mm2_per_ms2: ArrayLike,
        /,
        *,
        residual_tolerance_ms: float = 1.0e-7,
        path_tie_tolerance_ms: float = 1.0e-7,
    ):
        if not isinstance(route, (GraphEikonalRoute, FiniteElementEikonalRoute)):
            raise TypeError(
                "route must be GraphEikonalRoute or FiniteElementEikonalRoute."
            )
        positions = np.asarray(node_positions_mm, dtype=float)
        node_count = int(route.node_ids.shape[0])
        if positions.ndim != 2 or positions.shape[0] != node_count:
            raise ValueError("node_positions_mm must have shape [route node, dimension].")
        dimension = positions.shape[1]
        if dimension not in (1, 2, 3) or not np.all(np.isfinite(positions)):
            raise ValueError(
                "Eikonal positions must be finite and one-, two-, or three-dimensional."
            )
        tensors = np.asarray(velocity_tensors_mm2_per_ms2, dtype=float)
        if tensors.shape == (dimension, dimension):
            tensors = np.broadcast_to(tensors, (node_count, dimension, dimension)).copy()
        if tensors.shape != (node_count, dimension, dimension):
            raise ValueError(
                "velocity_tensors_mm2_per_ms2 must have shape [node, dimension, dimension]."
            )
        if not np.all(np.isfinite(tensors)) or not np.allclose(
            tensors, np.swapaxes(tensors, -1, -2), rtol=1.0e-10, atol=1.0e-12
        ):
            raise ValueError(
                "Every squared-velocity tensor must be finite and symmetric."
            )
        if np.any(np.linalg.eigvalsh(tensors) <= 0.0):
            raise ValueError("Every squared-velocity tensor must be positive definite.")
        residual_tolerance = float(residual_tolerance_ms)
        path_tolerance = float(path_tie_tolerance_ms)
        if (
            not isfinite(residual_tolerance)
            or residual_tolerance <= 0.0
            or not isfinite(path_tolerance)
            or path_tolerance < 0.0
        ):
            raise ValueError(
                "Eikonal tolerances must be finite; residual tolerance is positive."
            )
        self.route = route
        self.node_positions_mm = jnp.asarray(positions)
        self.velocity_tensors_mm2_per_ms2 = jnp.asarray(tensors)
        self.residual_tolerance_ms = residual_tolerance
        self.path_tie_tolerance_ms = path_tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-anisotropic-eikonal-plan",
                "route": route.route_id,
                "arrays": array_tree_fingerprint((positions, tensors)),
                "residual_tolerance_ms": residual_tolerance,
                "path_tie_tolerance_ms": path_tolerance,
            }
        )

    def prepare(self, /) -> "PreparedAnisotropicEikonal":
        return prepare_anisotropic_eikonal(self)


class PreparedAnisotropicEikonal(StrictModule, NonTrainableState):
    """Fixed edge fallback plus exact simplex-local anisotropic update data."""

    plan: AnisotropicEikonalPlan
    edge_nodes: Array
    edge_delay_ms: Array
    fem_target_nodes: Array
    fem_other_nodes: Array
    fem_inverse_displacement: Array
    fem_velocity_tensors_mm2_per_ms2: Array
    prepared_id: str = eqx.field(static=True)

    @property
    def node_count(self) -> int:
        return int(self.plan.route.node_ids.shape[0])


class EikonalSolveEvidence(StrictModule):
    """Convergence, reachability, and fixed-path derivative evidence."""

    maximum_bellman_residual_ms: Array
    convergence_sweep: Array
    unreachable_node_count: Array
    shortest_path_margin_ms: Array
    finite: Array
    converged: Array
    unique_shortest_paths: Array
    fixed_topology_derivative_valid: Array
    status: Array
    successful: Array


class EikonalSolveResult(StrictModule):
    """Arrival times and the deterministic selected predecessor at every node."""

    arrival_time_ms: Array
    predecessor_index: Array
    evidence: EikonalSolveEvidence
    prepared_id: str = eqx.field(static=True)


def _fem_edges(elements: np.ndarray, /) -> np.ndarray:
    pairs: set[tuple[int, int]] = set()
    for cell in elements:
        for left_position in range(cell.size):
            for right_position in range(left_position + 1, cell.size):
                left = int(cell[left_position])
                right = int(cell[right_position])
                pairs.add((min(left, right), max(left, right)))
    return np.asarray(sorted(pairs), dtype=np.int32)


def prepare_anisotropic_eikonal(
    plan: AnisotropicEikonalPlan, /
) -> PreparedAnisotropicEikonal:
    """Compile graph incidence or exact affine-simplex anisotropic updates."""

    if not isinstance(plan, AnisotropicEikonalPlan):
        raise TypeError("plan must be an AnisotropicEikonalPlan.")
    positions = np.asarray(plan.node_positions_mm)
    tensors = np.asarray(plan.velocity_tensors_mm2_per_ms2)
    dimension = positions.shape[1]
    if isinstance(plan.route, GraphEikonalRoute):
        edges = np.asarray(plan.route.edge_nodes, dtype=np.int32)
        target_nodes = np.empty((0,), dtype=np.int32)
        other_nodes = np.empty((0, dimension), dtype=np.int32)
        inverse_displacement = np.empty((0, dimension, dimension), dtype=positions.dtype)
        fem_tensors = np.empty((0, dimension, dimension), dtype=tensors.dtype)
    else:
        elements = np.asarray(plan.route.elements, dtype=np.int32)
        if elements.shape[1] != dimension + 1:
            raise ValueError(
                "Finite-element eikonal simplices must match the geometry dimension."
            )
        edges = _fem_edges(elements)
        target_items: list[int] = []
        other_items: list[list[int]] = []
        displacement_items: list[np.ndarray] = []
        tensor_items: list[np.ndarray] = []
        stable_ids = np.asarray(plan.route.node_ids)
        for cell in elements:
            element_tensor = np.mean(tensors[cell], axis=0)
            for target in cell:
                others = [int(node) for node in cell if node != target]
                others.sort(key=lambda node: int(stable_ids[node]))
                displacement = positions[others] - positions[target]
                target_items.append(int(target))
                other_items.append(others)
                displacement_items.append(displacement)
                tensor_items.append(element_tensor)
        target_nodes = np.asarray(target_items, dtype=np.int32)
        other_nodes = np.asarray(other_items, dtype=np.int32)
        displacement_batch = np.asarray(displacement_items, dtype=positions.dtype)
        identity_batch = np.broadcast_to(
            np.eye(dimension, dtype=positions.dtype), displacement_batch.shape
        )
        inverse_result = solve_small_linear(
            SmallLinearSolvePlan(dimension),
            jnp.asarray(displacement_batch),
            jnp.asarray(identity_batch),
        )
        if not bool(jnp.all(inverse_result.successful)):
            raise ValueError(
                "Every finite-element eikonal simplex must be nondegenerate."
            )
        inverse_displacement = np.asarray(inverse_result.value)
        fem_tensors = np.asarray(tensor_items, dtype=tensors.dtype)
    edge_displacement = positions[edges[:, 1]] - positions[edges[:, 0]]
    if np.any(np.all(edge_displacement == 0.0, axis=1)):
        raise ValueError(
            "Every prepared eikonal edge must have positive geometric length."
        )
    edge_tensors = 0.5 * (tensors[edges[:, 0]] + tensors[edges[:, 1]])
    metric_result = solve_small_linear(
        SmallLinearSolvePlan(dimension),
        jnp.asarray(edge_tensors),
        jnp.asarray(edge_displacement),
    )
    if not bool(jnp.all(metric_result.successful)):
        raise ValueError("Every anisotropic edge metric must be nonsingular.")
    delays = np.sqrt(np.sum(edge_displacement * np.asarray(metric_result.value), axis=1))
    if not np.all(np.isfinite(delays)) or np.any(delays <= 0.0):
        raise ValueError("Prepared anisotropic edge delays must be finite and positive.")
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-cardiovascular-anisotropic-eikonal",
            "plan": plan.plan_id,
            "stencil": array_tree_fingerprint(
                (
                    edges,
                    delays,
                    target_nodes,
                    other_nodes,
                    inverse_displacement,
                    fem_tensors,
                )
            ),
        }
    )
    return PreparedAnisotropicEikonal(
        plan,
        jnp.asarray(edges),
        jnp.asarray(delays),
        jnp.asarray(target_nodes),
        jnp.asarray(other_nodes),
        jnp.asarray(inverse_displacement),
        jnp.asarray(fem_tensors),
        prepared_id,
    )


def _fem_local_candidates(
    prepared: PreparedAnisotropicEikonal,
    arrival_time_ms: Array,
    infinity: Array,
    /,
) -> tuple[Array, Array]:
    """Return causal simplex updates and one deterministic diagnostic parent."""

    if isinstance(prepared.plan.route, GraphEikonalRoute):
        return (
            jnp.empty((0,), dtype=arrival_time_ms.dtype),
            jnp.empty((0,), dtype=jnp.int32),
        )
    other_times = arrival_time_ms[prepared.fem_other_nodes]
    known = jnp.all(jnp.isfinite(other_times), axis=1)
    safe_times = jnp.where(known[:, None], other_times, 0.0)
    constant_gradient = contract(
        "mij,mj->mi", prepared.fem_inverse_displacement, safe_times
    )
    time_gradient = jnp.sum(prepared.fem_inverse_displacement, axis=2)
    tensor = prepared.fem_velocity_tensors_mm2_per_ms2
    quadratic = contract("mi,mij,mj->m", time_gradient, tensor, time_gradient)
    linear = contract("mi,mij,mj->m", time_gradient, tensor, constant_gradient)
    constant = (
        contract("mi,mij,mj->m", constant_gradient, tensor, constant_gradient) - 1.0
    )
    discriminant = linear * linear - quadratic * constant
    root = (linear + jnp.sqrt(jnp.maximum(discriminant, 0.0))) / quadratic
    causal = root + prepared.plan.residual_tolerance_ms >= jnp.max(safe_times, axis=1)
    valid = known & (discriminant >= 0.0) & (quadratic > 0.0) & causal
    candidates = jnp.where(valid, root, infinity)
    parent_position = jnp.argmax(safe_times, axis=1)
    parent = jnp.take_along_axis(
        prepared.fem_other_nodes, parent_position[:, None], axis=1
    )[:, 0]
    return candidates, parent


def solve_anisotropic_eikonal(
    prepared: PreparedAnisotropicEikonal,
    source_node_indices: ArrayLike,
    source_times_ms: ArrayLike,
    /,
) -> EikonalSolveResult:
    """Solve earliest arrivals by fixed-count edge and affine-simplex sweeps.

    The route incidence and sweep count are static.  Gradients with respect to
    squared velocities or source times are meaningful only when
    ``fixed_topology_derivative_valid`` is true; topology changes, causal-root
    switches, and shortest-path ties are intentional derivative boundaries.
    """

    if not isinstance(prepared, PreparedAnisotropicEikonal):
        raise TypeError("prepared must be PreparedAnisotropicEikonal.")
    sources_host = np.asarray(source_node_indices, dtype=np.int32)
    source_times_array = jnp.asarray(source_times_ms)
    if sources_host.ndim != 1 or sources_host.size == 0:
        raise ValueError("source_node_indices must be a non-empty vector.")
    if source_times_array.shape != sources_host.shape:
        raise ValueError("source_times_ms must match source_node_indices.")
    if np.any(sources_host < 0) or np.any(sources_host >= prepared.node_count):
        raise ValueError("An eikonal source node lies outside the prepared route.")

    dtype = jnp.result_type(prepared.edge_delay_ms, source_times_array)
    sources = jnp.asarray(sources_host)
    source_times = source_times_array.astype(dtype)
    source_times_finite = jnp.all(jnp.isfinite(source_times))
    left = prepared.edge_nodes[:, 0]
    right = prepared.edge_nodes[:, 1]
    origins = jnp.concatenate((left, right))
    destinations = jnp.concatenate((right, left))
    weights = jnp.concatenate((prepared.edge_delay_ms, prepared.edge_delay_ms)).astype(
        dtype
    )
    infinity = jnp.asarray(jnp.inf, dtype=dtype)
    arrivals = jnp.full((prepared.node_count,), infinity, dtype=dtype)
    arrivals = arrivals.at[sources].min(source_times)

    def relax(sweep, carry):
        values, first_converged = carry
        edge_proposals = values[origins] + weights
        best = jnp.full_like(values, infinity).at[destinations].min(edge_proposals)
        fem_proposals, _ = _fem_local_candidates(prepared, values, infinity)
        if isinstance(prepared.plan.route, FiniteElementEikonalRoute):
            best = best.at[prepared.fem_target_nodes].min(fem_proposals)
        updated = jnp.minimum(values, best)
        changed = jnp.any(updated < values)
        first_converged = jnp.where(
            (first_converged == 0) & (~changed),
            jnp.asarray(sweep + 1, dtype=jnp.int32),
            first_converged,
        )
        return updated, first_converged

    arrivals, convergence_sweep = jax.lax.fori_loop(
        0,
        prepared.plan.route.maximum_sweeps,
        relax,
        (arrivals, jnp.asarray(0, dtype=jnp.int32)),
    )
    edge_proposals = arrivals[origins] + weights
    fem_proposals, fem_parents = _fem_local_candidates(prepared, arrivals, infinity)
    all_proposals = jnp.concatenate((edge_proposals, fem_proposals))
    all_destinations = jnp.concatenate((destinations, prepared.fem_target_nodes))
    all_parents = jnp.concatenate((origins, fem_parents))
    relaxation_defect = arrivals[all_destinations] - all_proposals
    finite_defect = jnp.where(jnp.isfinite(relaxation_defect), relaxation_defect, 0.0)
    maximum_residual = jnp.maximum(jnp.max(finite_defect), 0.0)
    reachable = jnp.isfinite(arrivals)
    unreachable_count = jnp.sum(~reachable, dtype=jnp.int32)

    if isinstance(prepared.plan.route, GraphEikonalRoute):
        selected_proposals = edge_proposals
        selected_destinations = destinations
        selected_parents = origins
    else:
        edge_best = jnp.full_like(arrivals, infinity).at[destinations].min(edge_proposals)
        improving_fem = fem_proposals < (
            edge_best[prepared.fem_target_nodes] - prepared.plan.path_tie_tolerance_ms
        )
        selected_proposals = jnp.concatenate(
            (edge_proposals, jnp.where(improving_fem, fem_proposals, infinity))
        )
        selected_destinations = jnp.concatenate((destinations, prepared.fem_target_nodes))
        selected_parents = jnp.concatenate((origins, fem_parents))
    candidate_count = selected_proposals.shape[0]
    candidate_matrix = jnp.full(
        (candidate_count, prepared.node_count), infinity, dtype=dtype
    )
    candidate_matrix = candidate_matrix.at[
        jnp.arange(candidate_count), selected_destinations
    ].set(selected_proposals)
    selected_candidate = jnp.argmin(candidate_matrix, axis=0).astype(jnp.int32)
    predecessor = selected_parents[selected_candidate]
    source_mask = jnp.zeros((prepared.node_count,), dtype=bool).at[sources].set(True)
    predecessor = jnp.where(reachable & (~source_mask), predecessor, -1)
    sorted_candidates = jnp.sort(candidate_matrix, axis=0)
    node_margin = sorted_candidates[1] - sorted_candidates[0]
    relevant = reachable & (~source_mask)
    path_margin = jnp.min(jnp.where(relevant, node_margin, jnp.inf))
    unique_paths = path_margin > prepared.plan.path_tie_tolerance_ms
    finite = source_times_finite & jnp.all(reachable) & jnp.all(jnp.isfinite(weights))
    converged = (convergence_sweep > 0) & (
        maximum_residual <= prepared.plan.residual_tolerance_ms
    )
    derivative_valid = finite & converged & unique_paths
    status = jnp.asarray(int(EikonalSolveStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        finite,
        status,
        jnp.bitwise_or(status, int(EikonalSolveStatus.NONFINITE)),
    )
    status = jnp.where(
        converged,
        status,
        jnp.bitwise_or(status, int(EikonalSolveStatus.NOT_CONVERGED)),
    )
    status = jnp.where(
        unreachable_count == 0,
        status,
        jnp.bitwise_or(status, int(EikonalSolveStatus.UNREACHABLE)),
    )
    status = jnp.where(
        unique_paths,
        status,
        jnp.bitwise_or(status, int(EikonalSolveStatus.PATH_TIE)),
    )
    successful = status == int(EikonalSolveStatus.SUCCESS)
    evidence = EikonalSolveEvidence(
        maximum_residual,
        convergence_sweep,
        unreachable_count,
        path_margin,
        finite,
        converged,
        unique_paths,
        derivative_valid,
        status,
        successful,
    )
    return EikonalSolveResult(arrivals, predecessor, evidence, prepared.prepared_id)


__all__ = [
    "AnisotropicEikonalPlan",
    "EikonalRoute",
    "EikonalSolveEvidence",
    "EikonalSolveResult",
    "EikonalSolveStatus",
    "FiniteElementEikonalRoute",
    "GraphEikonalRoute",
    "PreparedAnisotropicEikonal",
    "prepare_anisotropic_eikonal",
    "solve_anisotropic_eikonal",
]
