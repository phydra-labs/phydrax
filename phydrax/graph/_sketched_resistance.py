"""Row-addressed Rademacher resistance embeddings."""

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    LinearSolvePolicy,
    LinearSystem,
    prepare,
    PreparedLinearSolve,
    refresh,
    RHSLayout,
    solve_many,
    SolveResourcePolicy,
)
from ..sparse import EdgeRelation
from ..typing import Bool, checked, Dim, Float, Int32, parse, PRNGKey, Scalar, Size
from ._effective_resistance import _default_policy, EffectiveResistanceStatus
from ._resistance_laplacian import (
    _block_resources,
    _GroundedLaplacian,
    _NodeDim,
    _solver_block_size,
)


class _SketchDim(Dim):
    """Caller-selected sketch rows."""


@final
class SketchedResistanceEmbedding(StrictModule):
    """Ground-zero embedding and solve evidence for a caller-chosen dimension."""

    __strict_contract__ = True

    embedding: Float[_SketchDim, _NodeDim]
    component_label: Int32[_NodeDim]
    status: Int32[Scalar]
    solve_status: Int32[_SketchDim]
    residual_norm: Float[_SketchDim]
    iterations: Int32[_SketchDim]
    derivative_regular: Bool[_SketchDim]
    sketch_dimension: Size[_SketchDim] = eqx.field(static=True)

    def resistance(self, sources: ArrayLike, targets: ArrayLike, /) -> Array:
        """Return squared embedding distances, infinity across components."""
        source, target = jnp.asarray(sources), jnp.asarray(targets)
        if not jnp.issubdtype(source.dtype, jnp.integer) or not jnp.issubdtype(
            target.dtype, jnp.integer
        ):
            raise TypeError("Embedding query indices must be integers.")
        n = self.embedding.shape[1]
        in_range = (source >= 0) & (source < n) & (target >= 0) & (target < n)
        safe_source, safe_target = jnp.clip(source, 0, n - 1), jnp.clip(target, 0, n - 1)
        difference = self.embedding[:, safe_source] - self.embedding[:, safe_target]
        distance = jnp.sum(difference * difference, axis=0)
        connected = self.component_label[safe_source] == self.component_label[safe_target]
        return jnp.where(
            (self.status == 0) & in_range,
            jnp.where(connected, distance, jnp.inf),
            jnp.nan,
        )


@final
class SketchedEffectiveResistancePlan(NonTrainableState, StrictModule):
    """Prepare a grounded JL embedding; the caller explicitly chooses k.

    Rademacher rows are addressed by ``fold_in(key, row)``. Their weighted
    incidence RHS uses sqrt(w), not w, and is invariant to execution blocking.
    ``workspace_bytes`` includes retained embedding and logical primal/implicit
    derivative outer arrays, not compiler temporaries or native solver internals.
    """

    __strict_contract__ = True

    laplacian: _GroundedLaplacian
    prepared: PreparedLinearSolve | None
    sketch_dimension: Size[_SketchDim] = eqx.field(static=True)
    block_size: int = eqx.field(static=True)
    resource_items: tuple[tuple[str, int], ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        relation: EdgeRelation,
        /,
        *,
        sketch_dimension: int,
        policy: LinearSolvePolicy | None = None,
        dtype: DTypeLike = jnp.float64,
        resources: SolveResourcePolicy | None = None,
        workspace_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        if (
            isinstance(sketch_dimension, bool)
            or not isinstance(sketch_dimension, int)
            or sketch_dimension < 1
        ):
            raise ValueError("sketch_dimension must be a positive integer.")
        lap = _GroundedLaplacian(relation, dtype=dtype)
        k = sketch_dimension
        # Signs, weighted routes, two endpoint routes, and their node reduction.
        row_bytes = (relation.capacity * 4 + lap.node_count) * lap.dtype.itemsize
        selected = _default_policy(resources) if policy is None else policy
        if policy is not None and resources is not None:
            selected = eqx.tree_at(lambda p: p.resources, selected, resources)
        unit_operator = (
            None
            if lap.reduced_count == 0
            else lap.operator(jnp.ones(relation.route_shape, dtype=lap.dtype))
        )
        block, estimate = _block_resources(
            k,
            lap.reduced_count,
            k * lap.node_count * lap.dtype.itemsize,
            lap.dtype,
            workspace_bytes,
            maximum_block_size=_solver_block_size(unit_operator, selected, k),
            query_count=lap.node_count,
            edge_count=relation.capacity,
            row_assembly_bytes=row_bytes,
            sketch=True,
        )
        plan_id = canonical_fingerprint(
            {
                "kind": "sketched-resistance-plan",
                "structure": lap.structure_id,
                "sketch_dimension": k,
                "policy": {
                    "structure": str(jax.tree_util.tree_structure(selected)),
                    "leaves": jax.tree_util.tree_leaves(selected),
                },
                "block_size": block,
            }
        )
        self.laplacian = lap
        self.sketch_dimension = k
        self.block_size = block
        self.resource_items = tuple(sorted(estimate.items()))
        self.plan_id = plan_id
        self.prepared = (
            None
            if unit_operator is None
            else prepare(
                LinearSystem(
                    unit_operator,
                    problem_id=plan_id,
                ),
                selected,
                rhs_layout=RHSLayout((block,)),
            )
        )

    @property
    def resource_estimate(self) -> dict[str, int]:
        """Logical outer primal/derivative arrays, including retained embedding."""
        return dict(self.resource_items)

    @checked
    def embed(
        self, conductances: ArrayLike, key: PRNGKey, /
    ) -> SketchedResistanceEmbedding:
        """Embed conductances with a typed JAX random key, preserving evidence."""
        key = parse(key, PRNGKey, "key")
        w = self.laplacian.conductances(conductances)
        lap, k, b = self.laplacian, self.sketch_dimension, self.block_size

        def invalid(_: Array) -> SketchedResistanceEmbedding:
            return SketchedResistanceEmbedding(
                jnp.full((k, lap.node_count), jnp.nan, dtype=lap.dtype),
                lap.component_label,
                jnp.asarray(
                    EffectiveResistanceStatus.INVALID_CONDUCTANCE, dtype=jnp.int32
                ),
                jnp.full(k, -1, dtype=jnp.int32),
                jnp.full(k, jnp.nan, dtype=lap.dtype),
                jnp.zeros(k, dtype=jnp.int32),
                jnp.zeros(k, dtype=jnp.bool_),
                k,
            )

        def valid(values: Array) -> SketchedResistanceEmbedding:
            if self.prepared is None:
                return SketchedResistanceEmbedding(
                    jnp.zeros((k, lap.node_count), dtype=lap.dtype),
                    lap.component_label,
                    jnp.asarray(0, dtype=jnp.int32),
                    jnp.zeros(k, dtype=jnp.int32),
                    jnp.zeros(k, dtype=lap.dtype),
                    jnp.zeros(k, dtype=jnp.int32),
                    jnp.ones(k, dtype=jnp.bool_),
                    k,
                )
            bound = refresh(
                self.prepared, LinearSystem(lap.operator(values), problem_id=self.plan_id)
            )
            weight = jnp.sqrt(jnp.where(lap.relation.valid, values, 0)) / jnp.sqrt(
                jnp.asarray(k, dtype=lap.dtype)
            )

            def rhs_row(row: Array) -> Array:
                signs = jax.random.rademacher(
                    jax.random.fold_in(key, row),
                    (lap.relation.capacity,),
                    dtype=lap.dtype,
                )
                route = signs * weight
                rhs, _ = lap.incident.reduce(
                    jnp.concatenate((route, -route)),
                    reduction="sum",
                    accumulation="deterministic",
                )
                if not isinstance(rhs, Array):
                    raise TypeError("Sketch endpoint reduction must return an array.")
                return jnp.where(row < k, rhs[lap.retained_nodes], 0)

            def columns(start: Array) -> tuple[Array, Array, Array, Array, Array]:
                rows = start * b + jnp.arange(b, dtype=jnp.int32)
                rhs = jax.lax.map(rhs_row, rows).T
                result = solve_many(bound, rhs)
                if not isinstance(result.value, Array):
                    raise TypeError("Grounded solves must return array potentials.")
                embedding = jnp.where(
                    lap.reduced_index[:, None] >= 0,
                    result.value[jnp.maximum(lap.reduced_index, 0)],
                    0,
                ).T
                # Per-column validity of the solve's derivative contract
                # (route, convergence, and any regularity evidence).
                regular = result.derivative_valid
                return (
                    embedding,
                    result.status,
                    result.diagnostics.residual_norm,
                    result.diagnostics.iterations,
                    regular,
                )

            embedding, statuses, residuals, iterations, regular = jax.lax.map(
                columns, jnp.arange((k + b - 1) // b, dtype=jnp.int32)
            )
            statuses = statuses.reshape(-1)[:k]
            ok = jnp.all(statuses == 0)
            return SketchedResistanceEmbedding(
                jnp.where(ok, embedding.reshape((-1, lap.node_count))[:k], jnp.nan),
                lap.component_label,
                jnp.where(ok, 0, int(EffectiveResistanceStatus.SOLVE_FAILED)).astype(
                    jnp.int32
                ),
                statuses,
                residuals.reshape(-1)[:k],
                iterations.reshape(-1)[:k],
                regular.reshape(-1)[:k],
                k,
            )

        return jax.lax.cond(lap.admissible(w), valid, invalid, w)
