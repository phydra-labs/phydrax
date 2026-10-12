# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from enum import IntEnum
from functools import partial
from typing import assert_never, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, core
from jax.typing import ArrayLike
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..sparse import EdgeRelation, RelationExecutionPlan, RelationExecutionState
from ..typing import Bool, checked, Dim, Float64, Int32, parse, Scalar


type ShortestPathFieldMethod = Literal["host-dijkstra", "bounded-relaxation-reference"]


class _VertexDim(Dim):
    pass


class ShortestPathFieldStatus(IntEnum):
    SUCCESS = 0
    MAXIMUM_SWEEPS_REACHED = 1
    NEGATIVE_COST = 2
    NONFINITE_INPUT = 3
    CERTIFICATION_FAILED = 4
    WORK_BUDGET_EXCEEDED = 5
    INVALID_SOURCE = 6
    HOST_BYTE_BUDGET_EXCEEDED = 7


class ShortestPathFieldResult(StrictModule):
    """Certified distance field; nonregular derivatives select one forest subgradient."""

    __strict_contract__ = True
    distance: Float64[_VertexDim]
    reachable: Bool[_VertexDim]
    predecessor_edge: Int32[_VertexDim]
    hops: Int32[_VertexDim]
    margin: Float64[_VertexDim]
    sweeps: Int32[Scalar]
    work: Array
    projected_work: Array
    bellman_residual: Float64[Scalar]
    dual_residual: Float64[Scalar]
    derivative_regular: Bool[Scalar]
    status: Int32[Scalar]
    plan_id: str = eqx.field(static=True)
    method: ShortestPathFieldMethod = eqx.field(static=True)
    host_synchronization: bool = eqx.field(static=True)


def _host_dijkstra(
    costs: np.ndarray,
    sources: np.ndarray,
    offsets: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    *,
    vertices: int,
) -> tuple[np.ndarray, ...]:
    """Visible provider boundary: SciPy distances and minimum-hop tight forest."""
    source = source[valid]
    target = target[valid]
    edge_ids = np.flatnonzero(valid)
    weight = costs[valid]
    base = np.min(offsets)
    rows = np.concatenate((source, np.full(sources.size, vertices, np.int32)))
    cols = np.concatenate((target, sources))
    values = np.concatenate((weight, offsets - base))
    # CSR construction sums duplicates; shortest paths require their minimum.
    keys, inverse = np.unique(
        rows.astype(np.int64) * (vertices + 1) + cols, return_inverse=True
    )
    minimum = np.full(keys.size, np.inf, dtype=np.float64)
    np.minimum.at(minimum, inverse, values)
    graph = csr_matrix(
        (minimum, (keys // (vertices + 1), keys % (vertices + 1))),
        shape=(vertices + 1, vertices + 1),
    )
    distance = (
        np.asarray(dijkstra(graph, indices=vertices), dtype=np.float64)[:vertices] + base
    )
    tolerance = 64 * np.finfo(np.float64).eps * np.maximum(1.0, np.abs(distance[target]))
    tight = np.isfinite(distance[source]) & (
        np.abs(distance[source] + weight - distance[target]) <= tolerance
    )
    roots = np.abs(offsets - distance[sources]) <= 64 * np.finfo(
        np.float64
    ).eps * np.maximum(1.0, np.abs(offsets))
    tight_rows = np.concatenate(
        (source[tight], np.full(np.count_nonzero(roots), vertices, np.int32))
    )
    tight_cols = np.concatenate((target[tight], sources[roots]))
    tight_graph = csr_matrix(
        (np.ones(tight_rows.size, dtype=np.float64), (tight_rows, tight_cols)),
        shape=(vertices + 1, vertices + 1),
    )
    # Use unweighted mode: parallel tight routes must not become longer hops.
    hop_value = (
        np.asarray(dijkstra(tight_graph, indices=vertices, unweighted=True))[:vertices]
        - 1
    )
    hops = np.where(np.isfinite(hop_value), hop_value, -1).astype(np.int32)
    chosen = tight & (hops[source] + 1 == hops[target]) & (hops[target] > 0)
    predecessor = np.full(vertices, costs.size, dtype=np.int32)
    np.minimum.at(predecessor, target[chosen], edge_ids[chosen])
    predecessor[predecessor == costs.size] = -1
    root = np.full(vertices, sources.size, dtype=np.int32)
    root_candidate = roots & (hops[sources] == 0)
    np.minimum.at(
        root, sources[root_candidate], np.flatnonzero(root_candidate).astype(np.int32)
    )
    root[root == sources.size] = -1
    order = np.argsort(np.where(hops >= 0, hops, vertices), kind="stable").astype(
        np.int32
    )
    return (
        distance,
        predecessor,
        hops,
        order,
        root,
        np.asarray(1, np.int32),
        np.asarray(edge_ids.size, np.int64),
    )


class ShortestPathFieldPlan(StrictModule, NonTrainableState):
    """Multi-source nonnegative-cost fields over a directed relation.

    ``host-dijkstra`` synchronizes through ``jax.pure_callback`` to SciPy on the
    host; certificates, margins, and the forest VJP execute on the device.
    ``bounded-relaxation-reference`` is an explicitly budgeted device reference,
    not a scalable solver. Gradients at ties are deterministic selected Clarke
    subgradients; inspect ``derivative_regular`` before assuming differentiability.
    """

    relation: EdgeRelation
    execution: RelationExecutionState
    target_has_routes: Array
    source_count: int = eqx.field(static=True)
    method: ShortestPathFieldMethod = eqx.field(static=True)
    maximum_sweeps: int = eqx.field(static=True)
    maximum_work: int = eqx.field(static=True)
    projected_work: int = eqx.field(static=True)
    host_projected_work: int = eqx.field(static=True)
    maximum_host_bytes: int = eqx.field(static=True)
    projected_host_bytes: int = eqx.field(static=True)
    tie_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        relation: EdgeRelation,
        /,
        *,
        source_count: int = 1,
        method: ShortestPathFieldMethod = "host-dijkstra",
        maximum_sweeps: int | None = None,
        tie_tolerance: float = 0.0,
        maximum_work: int = 10_000_000,
        maximum_host_bytes: int = 268_435_456,
    ) -> None:
        method = parse(method, ShortestPathFieldMethod, "method")
        if relation.source_size != relation.target_size or relation.source_size < 1:
            raise ValueError("Shortest-path fields require one nonempty vertex space.")
        if source_count < 1 or source_count > relation.source_size:
            raise ValueError("source_count must lie between one and the vertex count.")
        if maximum_work < 1 or not np.isfinite(tie_tolerance) or tie_tolerance < 0:
            raise ValueError(
                "Work budget must be positive and tie tolerance finite and nonnegative."
            )
        if maximum_host_bytes < 1:
            raise ValueError("Host byte budget must be positive.")
        sweeps = relation.source_size if maximum_sweeps is None else maximum_sweeps
        if sweeps < 1:
            raise ValueError("maximum_sweeps must be positive.")
        match method:
            case "host-dijkstra":
                if maximum_sweeps is not None:
                    raise ValueError(
                        "maximum_sweeps applies only to the bounded reference method."
                    )
            case "bounded-relaxation-reference":
                if sweeps > relation.source_size:
                    raise ValueError(
                        "Reference maximum_sweeps cannot exceed the V*E work bound."
                    )
            case _:
                assert_never(method)
        self.relation = relation
        self.execution = RelationExecutionPlan().prepare(relation)
        target_counts, _ = self.execution.reduce(
            jnp.ones(relation.capacity, dtype=jnp.int32), reduction="sum"
        )
        self.target_has_routes = target_counts > 0
        self.source_count = source_count
        self.method = method
        self.maximum_sweeps = sweeps
        self.maximum_work = maximum_work
        vertices, edges = relation.source_size, relation.capacity
        augmented_edges = edges + source_count
        # Two Dijkstra traversals, duplicate sorting, and forest ordering. Capacity
        # (not live support) bounds masked copies before the host sees the inputs.
        self.host_projected_work = (
            8
            * (vertices + 1 + augmented_edges)
            * ((vertices + 1).bit_length() + max(1, augmented_edges).bit_length())
        )
        self.maximum_host_bytes = maximum_host_bytes
        # Callback inputs; masked edge/source arrays and certificate temporaries;
        # COO/unique scratch, both retained CSR graphs and provider heap/workspace;
        # vertex outputs/transfer copies; scalars and fixed Python/provider overhead.
        self.projected_host_bytes = (
            17 * edges
            + 12 * source_count
            + 88 * edges
            + 32 * source_count
            + 256 * augmented_edges
            + 160 * (vertices + 1)
            + 16
            + 65_536
        )
        self.projected_work = (
            self.host_projected_work if method == "host-dijkstra" else vertices * edges
        )
        self.tie_tolerance = tie_tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "shortest-path-field",
                "relation": array_tree_fingerprint(
                    (relation.source_indices, relation.target_indices, relation.valid)
                ),
                "vertices": relation.source_size,
                "sources": source_count,
                "method": method,
                "maximum_sweeps": sweeps,
                "maximum_work": maximum_work,
                "maximum_host_bytes": maximum_host_bytes,
                "tie_tolerance": tie_tolerance,
            }
        )

    def solve(
        self,
        costs: ArrayLike,
        sources: ArrayLike,
        /,
        *,
        source_offsets: ArrayLike | None = None,
    ) -> ShortestPathFieldResult:
        c = jnp.asarray(costs, dtype=jnp.float64)
        if isinstance(sources, (Array, core.Tracer)):
            original = jnp.asarray(sources)
            if original.shape != (self.source_count,):
                raise ValueError("Sources must match the planned source count.")
            if not jnp.issubdtype(original.dtype, jnp.integer):
                raise TypeError("Sources must be integer vertex indices.")
            valid_sources = original >= 0
            if self.relation.source_size <= jnp.iinfo(original.dtype).max:
                valid_sources = valid_sources & (original < self.relation.source_size)
            source_valid = jnp.all(valid_sources)
            s = jnp.where(valid_sources, original, 0).astype(jnp.int32)
        else:
            original_host = np.asarray(sources)
            if original_host.shape != (self.source_count,):
                raise ValueError("Sources must match the planned source count.")
            if not np.issubdtype(original_host.dtype, np.integer):
                raise TypeError("Sources must be integer vertex indices.")
            valid_host = (original_host >= 0) & (
                original_host < self.relation.source_size
            )
            source_valid = jnp.asarray(np.all(valid_host))
            s = jnp.asarray(np.where(valid_host, original_host, 0), dtype=jnp.int32)
        if c.shape != (self.relation.capacity,):
            raise ValueError("Costs must match the planned route count.")
        o = (
            jnp.zeros(self.source_count, jnp.float64)
            if source_offsets is None
            else jnp.asarray(source_offsets, dtype=jnp.float64)
        )
        if o.shape != (self.source_count,):
            raise ValueError("Source offsets must match source_count.")
        field = _distance_field(self, c, s, o, source_valid)
        return _certificate(self, c, s, o, field, source_valid)


def _minimum(plan: ShortestPathFieldPlan, values: Array) -> Array:
    result, _ = plan.execution.reduce(values, reduction="min")
    # Native dense relation reduction zero-fills targets without stored routes.
    # The distance-field owner restores the min identity at those vertices.
    identity = (
        jnp.iinfo(values.dtype).max
        if jnp.issubdtype(values.dtype, jnp.integer)
        else jnp.inf
    )
    return jnp.where(plan.target_has_routes, result, jnp.asarray(identity, values.dtype))


def _reference(
    plan: ShortestPathFieldPlan, costs: Array, sources: Array, offsets: Array
) -> tuple[Array, ...]:
    n = plan.relation.source_size
    u, v = plan.relation.source_indices, plan.relation.target_indices
    valid = plan.relation.valid
    source_value = jnp.full(n, jnp.inf).at[sources].min(offsets)
    initial_hops = jnp.where(jnp.isfinite(source_value), 0, n)

    def condition(state: tuple[Array, Array, Array, Array]) -> Array:
        return state[2] & (state[3] < plan.maximum_sweeps)

    def sweep(
        state: tuple[Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array]:
        d, h, _, count = state
        candidates = jnp.where(valid, d[u] + costs, jnp.inf)
        best = jnp.minimum(source_value, _minimum(plan, candidates))
        candidate_hops = jnp.where(valid & (candidates == best[v]), h[u] + 1, n)
        best_hops = jnp.minimum(
            jnp.where(source_value == best, 0, n), _minimum(plan, candidate_hops)
        )
        changed = jnp.any(best != d) | jnp.any(best_hops != h)
        return best, best_hops, changed, count + 1

    d, h, _, sweeps = jax.lax.while_loop(
        condition,
        sweep,
        (source_value, initial_hops, jnp.asarray(True), jnp.asarray(0, jnp.int32)),
    )
    ids = jnp.arange(costs.size, dtype=jnp.int32)
    eligible = (
        valid
        & jnp.isfinite(d[u])
        & (d[u] + costs == d[v])
        & (h[u] + 1 == h[v])
        & (h[v] > 0)
    )
    pred = _minimum(plan, jnp.where(eligible, ids, costs.size))
    pred = jnp.where((pred < costs.size) & jnp.isfinite(d), pred, -1)
    roots = (
        jnp.full(n, plan.source_count, jnp.int32)
        .at[sources]
        .min(
            jnp.where(
                (offsets == d[sources]) & (h[sources] == 0),
                jnp.arange(plan.source_count, dtype=jnp.int32),
                plan.source_count,
            )
        )
    )
    roots = jnp.where(roots < plan.source_count, roots, -1)
    hops = jnp.where(jnp.isfinite(d), h, -1)
    order = jnp.argsort(jnp.where(hops >= 0, hops, n), stable=True).astype(jnp.int32)
    return (
        d,
        pred.astype(jnp.int32),
        hops.astype(jnp.int32),
        order,
        roots,
        sweeps,
        sweeps.astype(jnp.int64) * costs.size,
    )


def _raw_field(
    plan: ShortestPathFieldPlan,
    costs: Array,
    sources: Array,
    offsets: Array,
    source_valid: Array,
) -> tuple[Array, ...]:
    n = plan.relation.source_size

    def invalid() -> tuple[Array, ...]:
        return (
            jnp.full(n, jnp.nan, jnp.float64),
            jnp.full(n, -1, jnp.int32),
            jnp.full(n, -1, jnp.int32),
            jnp.arange(n, dtype=jnp.int32),
            jnp.full(n, -1, jnp.int32),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int64),
        )

    if plan.projected_work > plan.maximum_work or (
        plan.method == "host-dijkstra"
        and plan.projected_host_bytes > plan.maximum_host_bytes
    ):
        return invalid()

    def execute() -> tuple[Array, ...]:
        match plan.method:
            case "host-dijkstra":
                signature = (
                    jax.ShapeDtypeStruct((n,), jnp.float64),
                    *(jax.ShapeDtypeStruct((n,), jnp.int32) for _ in range(4)),
                    jax.ShapeDtypeStruct((), jnp.int32),
                    jax.ShapeDtypeStruct((), jnp.int64),
                )
                return jax.pure_callback(
                    partial(_host_dijkstra, vertices=n),
                    signature,
                    costs,
                    sources,
                    offsets,
                    plan.relation.source_indices,
                    plan.relation.target_indices,
                    plan.relation.valid,
                    vmap_method="sequential",
                )
            case "bounded-relaxation-reference":
                return _reference(plan, costs, sources, offsets)
            case _:
                assert_never(plan.method)

    admissible = (
        jnp.all(jnp.where(plan.relation.valid, jnp.isfinite(costs) & (costs >= 0), True))
        & jnp.all(jnp.isfinite(offsets))
        & source_valid
    )
    return jax.lax.cond(admissible, execute, invalid)


@jax.custom_vjp
def _distance_field(
    plan: ShortestPathFieldPlan,
    costs: Array,
    sources: Array,
    offsets: Array,
    source_valid: Array,
) -> tuple[Array, ...]:
    return _raw_field(plan, costs, sources, offsets, source_valid)


def _distance_forward(
    plan: ShortestPathFieldPlan,
    costs: Array,
    sources: Array,
    offsets: Array,
    source_valid: Array,
) -> tuple[tuple[Array, ...], tuple[ShortestPathFieldPlan, tuple[Array, ...]]]:
    field = _raw_field(plan, costs, sources, offsets, source_valid)
    return field, (plan, field)


def _distance_backward(
    residual: tuple[ShortestPathFieldPlan, tuple[Array, ...]],
    cotangent: tuple[Array, ...],
) -> tuple[None, Array, None, Array, None]:
    plan, field = residual
    distance, predecessor, hops, order, roots, _, _ = field
    n, e = plan.relation.source_size, plan.relation.capacity
    safe_edges = jnp.maximum(predecessor, 0)
    parent = jnp.concatenate((plan.relation.source_indices, jnp.zeros(1, jnp.int32)))[
        jnp.where(predecessor >= 0, safe_edges, e)
    ]
    accum = jnp.where(jnp.isfinite(distance), cotangent[0], 0.0)

    def accumulate(
        i: int, state: tuple[Array, Array, Array]
    ) -> tuple[Array, Array, Array]:
        total, dc, do = state
        vertex = order[n - 1 - i]
        amount = total[vertex]
        has_parent = predecessor[vertex] >= 0
        if e:
            dc = dc.at[jnp.maximum(predecessor[vertex], 0)].add(
                jnp.where(has_parent, amount, 0.0)
            )
        total = total.at[parent[vertex]].add(jnp.where(has_parent, amount, 0.0))
        do = do.at[jnp.maximum(roots[vertex], 0)].add(
            jnp.where(roots[vertex] >= 0, amount, 0.0)
        )
        return total, dc, do

    _, dc, do = jax.lax.fori_loop(
        0,
        n,
        accumulate,
        (accum, jnp.zeros(e, jnp.float64), jnp.zeros(plan.source_count, jnp.float64)),
    )
    forest_valid = jnp.all(
        ~jnp.isfinite(distance)
        | ((predecessor >= 0) & (hops[parent] + 1 == hops) | (roots >= 0))
    ) & jnp.all(~jnp.isnan(distance))
    return (
        None,
        jnp.where(forest_valid, dc, jnp.nan),
        None,
        jnp.where(forest_valid, do, jnp.nan),
        None,
    )


_distance_field.defvjp(_distance_forward, _distance_backward)


def _certificate(
    plan: ShortestPathFieldPlan,
    costs: Array,
    sources: Array,
    offsets: Array,
    field: tuple[Array, ...],
    source_valid: Array,
) -> ShortestPathFieldResult:
    d, pred, hops, order, roots, sweeps, work = field
    n, e = plan.relation.source_size, plan.relation.capacity
    u, v, valid = (
        plan.relation.source_indices,
        plan.relation.target_indices,
        plan.relation.valid,
    )
    reachable = jnp.isfinite(d)
    candidate = jnp.where(valid & reachable[u], d[u] + costs, jnp.inf)
    source_best = jnp.full(n, jnp.inf).at[sources].min(offsets, mode="drop")
    best = jnp.minimum(source_best, _minimum(plan, candidate))
    bellman = jnp.max(
        jnp.where(
            reachable, jnp.abs(best - d), jnp.where(jnp.isfinite(best), jnp.inf, 0.0)
        ),
        initial=0.0,
    )
    slack = candidate - d[v]
    dual = jnp.max(
        jnp.where(valid & reachable[u], jnp.maximum(-slack, 0.0), 0.0), initial=0.0
    )
    safe_pred = jnp.maximum(pred, 0)
    parent = jnp.concatenate((u, jnp.zeros(1, jnp.int32)))[
        jnp.where(pred >= 0, safe_pred, e)
    ]
    edge_cost = jnp.concatenate((costs, jnp.zeros(1, jnp.float64)))[
        jnp.where(pred >= 0, safe_pred, e)
    ]
    tightness = jnp.max(
        jnp.where(pred >= 0, jnp.abs(d[parent] + edge_cost - d), 0.0), initial=0.0
    )
    acyclic = jnp.all(
        ~reachable
        | jnp.where(
            pred >= 0, (hops[parent] + 1 == hops) & (hops > 0), (roots >= 0) & (hops == 0)
        )
    )
    tolerance = (
        64
        * jnp.finfo(jnp.float64).eps
        * jnp.maximum(1.0, jnp.max(jnp.where(reachable, jnp.abs(d), 0.0), initial=0.0))
    )
    certified = (
        acyclic & (bellman <= tolerance) & (dual <= tolerance) & (tightness <= tolerance)
    )
    gap = _minimum(
        plan,
        jnp.where(
            valid & reachable[u] & (jnp.arange(e, dtype=jnp.int32) != pred[v]),
            jnp.maximum(slack, 0.0),
            jnp.inf,
        ),
    )
    source_gap = (
        jnp.full(n, jnp.inf)
        .at[sources]
        .min(
            jnp.where(
                jnp.arange(plan.source_count, dtype=jnp.int32) != roots[sources],
                jnp.maximum(offsets - d[sources], 0.0),
                jnp.inf,
            ),
            mode="drop",
        )
    )
    local = jnp.minimum(gap, source_gap)
    if plan.source_count == 1:
        # A nonnegative cycle cannot compete with the fixed single-source value.
        # Remove its gap before inheritance, not merely in the returned root.
        local = local.at[sources[0]].set(jnp.inf)

    def inherit(i: int, margin: Array) -> Array:
        vertex = order[i]
        value = jnp.where(
            pred[vertex] >= 0,
            jnp.minimum(margin[vertex], margin[parent[vertex]]),
            margin[vertex],
        )
        return margin.at[vertex].set(value)

    margin = jax.lax.fori_loop(0, n, inherit, local)
    margin = jnp.where(reachable, margin, jnp.inf)
    status = jnp.where(
        certified,
        int(ShortestPathFieldStatus.SUCCESS),
        int(ShortestPathFieldStatus.CERTIFICATION_FAILED),
    )
    if plan.method == "bounded-relaxation-reference":
        status = jnp.where(
            ~certified & (sweeps >= plan.maximum_sweeps),
            int(ShortestPathFieldStatus.MAXIMUM_SWEEPS_REACHED),
            status,
        )
    if plan.projected_work > plan.maximum_work:
        status = jnp.asarray(int(ShortestPathFieldStatus.WORK_BUDGET_EXCEEDED), jnp.int32)
    if (
        plan.method == "host-dijkstra"
        and plan.projected_host_bytes > plan.maximum_host_bytes
    ):
        status = jnp.asarray(
            int(ShortestPathFieldStatus.HOST_BYTE_BUDGET_EXCEEDED), jnp.int32
        )
    status = jnp.where(
        jnp.any(jnp.where(valid, costs < 0, False)),
        int(ShortestPathFieldStatus.NEGATIVE_COST),
        status,
    )
    status = jnp.where(
        jnp.any(jnp.where(valid, ~jnp.isfinite(costs), False))
        | jnp.any(~jnp.isfinite(offsets)),
        int(ShortestPathFieldStatus.NONFINITE_INPUT),
        status,
    )
    status = jnp.where(
        ~source_valid,
        int(ShortestPathFieldStatus.INVALID_SOURCE),
        status,
    ).astype(jnp.int32)
    regular = (
        (status == int(ShortestPathFieldStatus.SUCCESS))
        & certified
        & jnp.all(~reachable | (margin > plan.tie_tolerance))
    )
    return ShortestPathFieldResult(
        d,
        reachable,
        pred,
        hops,
        margin,
        sweeps,
        work,
        jnp.asarray(plan.projected_work, jnp.int64),
        bellman,
        dual,
        regular,
        status,
        plan.plan_id,
        plan.method,
        plan.method == "host-dijkstra",
    )
