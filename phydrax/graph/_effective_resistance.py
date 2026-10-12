"""Block-budgeted grounded-solve effective resistance."""

from enum import IntEnum
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    JacobiPreconditionerBuilder,
    LinearSolvePolicy,
    LinearSystem,
    PCG,
    PreconditioningPolicy,
    prepare,
    PreparedLinearSolve,
    refresh,
    RHSLayout,
    solve_many,
    SolveResourcePolicy,
)
from ..sparse import EdgeRelation
from ..typing import Bool, checked, Dim, Float, Int32, Scalar, Size
from ._resistance_laplacian import (
    _block_resources,
    _GroundedLaplacian,
    _solver_block_size,
)


class _QueryDim(Dim):
    """Static resistance query nodes."""


class EffectiveResistanceStatus(IntEnum):
    """Portable outcome of grounded resistance execution."""

    SUCCESS = 0
    INVALID_CONDUCTANCE = 1
    SOLVE_FAILED = 2


@final
class EffectiveResistanceResult(StrictModule):
    """Pairwise resistances and per-query numerical solve evidence."""

    __strict_contract__ = True

    resistance: Float[_QueryDim, _QueryDim]
    connected: Bool[_QueryDim, _QueryDim]
    status: Int32[Scalar]
    solve_status: Int32[_QueryDim]
    residual_norm: Float[_QueryDim]
    iterations: Int32[_QueryDim]
    derivative_regular: Bool[_QueryDim]


def _default_policy(resources: SolveResourcePolicy | None, /) -> LinearSolvePolicy:
    return LinearSolvePolicy(
        PCG(),
        preconditioning=PreconditioningPolicy(JacobiPreconditionerBuilder()),
        resources=resources,
    )


@final
class EffectiveResistancePlan(NonTrainableState, StrictModule):
    """Prepare one grounded solve template and stream query RHS blocks.

    ``workspace_bytes`` budgets retained output and logical primal/implicit
    derivative outer arrays; ``resources`` independently budgets native solver
    internals. Compiler temporaries are outside both logical outer estimates.
    Conductances never participate in the plan fingerprint.
    """

    __strict_contract__ = True

    laplacian: _GroundedLaplacian
    nodes: Int32[_QueryDim]
    query_index: Int32[_QueryDim]
    connected: Bool[_QueryDim, _QueryDim]
    prepared: PreparedLinearSolve | None
    block_size: int = eqx.field(static=True)
    query_count: Size[_QueryDim] = eqx.field(static=True)
    resource_items: tuple[tuple[str, int], ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        relation: EdgeRelation,
        nodes: ArrayLike,
        /,
        *,
        policy: LinearSolvePolicy | None = None,
        dtype: DTypeLike = jnp.float64,
        resources: SolveResourcePolicy | None = None,
        workspace_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        lap = _GroundedLaplacian(relation, dtype=dtype)
        host = np.asarray(nodes)
        if host.ndim != 1 or host.size < 1 or not np.issubdtype(host.dtype, np.integer):
            raise ValueError(
                "nodes must be a nonempty one-dimensional host integer array."
            )
        if np.any((host < 0) | (host >= lap.node_count)):
            raise ValueError("Query nodes are outside the relation node space.")
        q = host.size
        selected = _default_policy(resources) if policy is None else policy
        if policy is not None and resources is not None:
            selected = eqx.tree_at(lambda p: p.resources, selected, resources)
        unit_operator = (
            None
            if lap.reduced_count == 0
            else lap.operator(jnp.ones(relation.route_shape, dtype=lap.dtype))
        )
        block, estimate = _block_resources(
            q,
            lap.reduced_count,
            q * q * lap.dtype.itemsize,
            lap.dtype,
            workspace_bytes,
            query_count=q,
            edge_count=relation.capacity,
            maximum_block_size=_solver_block_size(unit_operator, selected, q),
        )
        plan_id = canonical_fingerprint(
            {
                "kind": "effective-resistance-plan",
                "structure": lap.structure_id,
                "nodes": host,
                "policy": {
                    "structure": str(jax.tree_util.tree_structure(selected)),
                    "leaves": jax.tree_util.tree_leaves(selected),
                },
                "block_size": block,
            }
        )
        self.laplacian = lap
        self.nodes = jnp.asarray(host, dtype=jnp.int32)
        self.query_index = lap.reduced_index[self.nodes]
        self.connected = (
            lap.component_label[self.nodes, None]
            == lap.component_label[self.nodes][None, :]
        )
        self.block_size = block
        self.query_count = q
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
        """Logical outer primal/derivative arrays, including retained output."""
        return dict(self.resource_items)

    def solve(self, conductances: ArrayLike, /) -> EffectiveResistanceResult:
        """Solve with positive finite conductances; invalid values never enter a solve."""
        w = self.laplacian.conductances(conductances)
        q, b = self.query_count, self.block_size

        def invalid(_: Array) -> EffectiveResistanceResult:
            return EffectiveResistanceResult(
                jnp.full((q, q), jnp.nan, dtype=self.laplacian.dtype),
                self.connected,
                jnp.asarray(
                    EffectiveResistanceStatus.INVALID_CONDUCTANCE, dtype=jnp.int32
                ),
                jnp.full(q, -1, dtype=jnp.int32),
                jnp.full(q, jnp.nan, dtype=self.laplacian.dtype),
                jnp.zeros(q, dtype=jnp.int32),
                jnp.zeros(q, dtype=jnp.bool_),
            )

        def valid(values: Array) -> EffectiveResistanceResult:
            if self.prepared is None:
                return EffectiveResistanceResult(
                    jnp.where(
                        self.connected,
                        jnp.zeros((q, q), dtype=self.laplacian.dtype),
                        jnp.inf,
                    ),
                    self.connected,
                    jnp.asarray(0, dtype=jnp.int32),
                    jnp.zeros(q, dtype=jnp.int32),
                    jnp.zeros(q, dtype=self.laplacian.dtype),
                    jnp.zeros(q, dtype=jnp.int32),
                    jnp.ones(q, dtype=jnp.bool_),
                )
            bound = refresh(
                self.prepared,
                LinearSystem(self.laplacian.operator(values), problem_id=self.plan_id),
            )

            def columns(start: Array) -> tuple[Array, Array, Array, Array, Array]:
                ids = start * b + jnp.arange(b, dtype=jnp.int32)
                indices = self.query_index[jnp.minimum(ids, q - 1)]
                rhs = (
                    jax.nn.one_hot(
                        indices, self.laplacian.reduced_count, dtype=self.laplacian.dtype
                    ).T
                    * (ids < q).astype(self.laplacian.dtype)[None, :]
                )
                result = solve_many(bound, rhs)
                if not isinstance(result.value, Array):
                    raise TypeError("Grounded solves must return array potentials.")
                potentials = jnp.where(
                    self.query_index[:, None] >= 0,
                    result.value[jnp.maximum(self.query_index, 0), :],
                    0,
                )
                # Per-column validity of the solve's derivative contract
                # (route, convergence, and any regularity evidence).
                regular = result.derivative_valid
                return (
                    potentials.T,
                    result.status,
                    result.diagnostics.residual_norm,
                    result.diagnostics.iterations,
                    regular,
                )

            potentials, statuses, residuals, iterations, regular = jax.lax.map(
                columns, jnp.arange((q + b - 1) // b, dtype=jnp.int32)
            )
            p = potentials.reshape((-1, q))[:q, :].T
            statuses = statuses.reshape(-1)[:q]
            ok = jnp.all(statuses == 0)
            diagonal = jnp.diag(p)
            resistance = diagonal[:, None] + diagonal[None, :] - p - p.T
            resistance = jnp.where(self.connected, resistance, jnp.inf)
            resistance = resistance.at[jnp.diag_indices(q)].set(0)
            resistance = jnp.where(ok, resistance, jnp.nan)
            return EffectiveResistanceResult(
                resistance,
                self.connected,
                jnp.where(ok, 0, int(EffectiveResistanceStatus.SOLVE_FAILED)).astype(
                    jnp.int32
                ),
                statuses,
                residuals.reshape(-1)[:q],
                iterations.reshape(-1)[:q],
                regular.reshape(-1)[:q],
            )

        return jax.lax.cond(self.laplacian.admissible(w), valid, invalid, w)
