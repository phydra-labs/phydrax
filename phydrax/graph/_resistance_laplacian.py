"""Immutable topology and deterministic assembly of component-grounded Laplacians."""

from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    ArraySpace,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    plan as plan_linear_solve,
    RHSLayout,
)
from ..sparse import (
    EdgeRelation,
    RelationExecutionPlan,
    RelationExecutionState,
    SparseCoordinateOperator,
)
from ..typing import checked, Dim, Int32, Size


class _NodeDim(Dim):
    """Original graph nodes."""


class _ReducedDim(Dim):
    """Non-ground coordinates."""


class _OffEdgeDim(Dim):
    """Edges with two non-ground endpoints."""


@final
class _GroundedLaplacian(NonTrainableState, StrictModule):
    """Ground one minimum-index node per component.

    For positive conductances, x^T L_g x = sum_e w_e (x_u-x_v)^2.
    Every component has a fixed zero ground, so a zero quadratic form forces
    every reduced coordinate to zero: the reduced operator is SPD.
    """

    __strict_contract__ = True

    relation: EdgeRelation
    reduced_relation: EdgeRelation
    incident: RelationExecutionState
    component_label: Int32[_NodeDim]
    roots: Int32[_NodeDim]
    reduced_index: Int32[_NodeDim]
    retained_nodes: Int32[_ReducedDim]
    off_edge: Int32[_OffEdgeDim]
    node_count: Size[_NodeDim] = eqx.field(static=True)
    reduced_count: Size[_ReducedDim] = eqx.field(static=True)
    dtype: np.dtype = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    space: ArraySpace

    @checked
    def __init__(self, relation: EdgeRelation, /, *, dtype: DTypeLike) -> None:
        from ..topology import ComponentLabelingStatus, ConnectedComponentPlan

        n = relation.source_size
        if n < 1 or n != relation.target_size:
            raise ValueError(
                "Resistance requires a square relation with at least one node."
            )
        u = np.asarray(relation.source_indices)
        v = np.asarray(relation.target_indices)
        valid = np.asarray(relation.valid)
        if np.any(valid & (u == v)):
            raise ValueError("Resistance relations must not contain self loops.")
        if np.any(valid & (u > v)):
            raise ValueError(
                "Use canonical source < target orientation, as in grid_adjacency_relation."
            )
        dtype_ = np.dtype(dtype)
        if dtype_ not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("Resistance requires float32 or float64 dtype.")
        components = ConnectedComponentPlan(
            relation, component_capacity=n, maximum_rounds=n
        ).label(jnp.ones(n, dtype=jnp.bool_))
        if int(components.status) != int(ComponentLabelingStatus.CONVERGED):
            raise RuntimeError("Connected-component preparation did not converge.")
        roots = np.asarray(components.root)
        retained = np.flatnonzero(roots != np.arange(n)).astype(np.int32)
        reduced = np.full(n, -1, dtype=np.int32)
        reduced[retained] = np.arange(retained.size, dtype=np.int32)
        # A fully grounded graph has no unknowns; its execution path skips solves.
        r = retained.size
        safe_u, safe_v = np.where(valid, u, 0), np.where(valid, v, 0)
        edge = np.flatnonzero(
            valid & (reduced[safe_u] >= 0) & (reduced[safe_v] >= 0)
        ).astype(np.int32)
        ru, rv = reduced[u[edge]], reduced[v[edge]]
        diagonal = np.arange(r, dtype=np.int32)
        reduced_relation = EdgeRelation(
            np.concatenate((ru, rv, diagonal)),
            np.concatenate((rv, ru, diagonal)),
            source_size=r,
            target_size=r,
        )
        incident_relation = EdgeRelation(
            np.concatenate(
                (
                    np.arange(relation.capacity, dtype=np.int32),
                    np.arange(relation.capacity, dtype=np.int32),
                )
            ),
            np.concatenate((safe_u, safe_v)),
            valid=np.concatenate((valid, valid)),
            source_size=relation.capacity,
            target_size=n,
        )
        self.relation = relation
        self.reduced_relation = reduced_relation
        self.incident = RelationExecutionPlan().prepare(incident_relation)
        self.component_label = components.label
        self.roots = components.root
        self.reduced_index = jnp.asarray(reduced)
        self.retained_nodes = jnp.asarray(retained)
        self.off_edge = jnp.asarray(edge)
        self.node_count = n
        self.reduced_count = r
        self.dtype = dtype_
        self.structure_id = canonical_fingerprint(
            {
                "kind": "grounded-laplacian",
                "source": u,
                "target": v,
                "valid": valid,
                "roots": roots,
                "dtype": dtype_.str,
            }
        )
        self.space = ArraySpace((r,), dtype=dtype_)

    def conductances(self, values: ArrayLike, /) -> Array:
        w = jnp.asarray(values)
        if w.shape != self.relation.route_shape:
            raise ValueError(f"Conductances must have shape {self.relation.route_shape}.")
        if w.dtype != self.dtype:
            raise TypeError(f"Conductances must have dtype {self.dtype}.")
        return w

    def admissible(self, w: Array, /) -> Array:
        return jnp.all(~self.relation.valid | (jnp.isfinite(w) & (w > 0)))

    def operator(self, w: Array, /) -> SparseCoordinateOperator:
        clean = jnp.where(self.relation.valid, w, 0)
        diagonal, _ = self.incident.reduce(
            jnp.concatenate((clean, clean)), reduction="sum", accumulation="deterministic"
        )
        if not isinstance(diagonal, Array):
            raise TypeError("Incident reduction must return an array.")
        coefficients = jnp.concatenate(
            (-clean[self.off_edge], -clean[self.off_edge], diagonal[self.retained_nodes])
        )
        return SparseCoordinateOperator(
            self.reduced_relation,
            coefficients,
            source=self.space,
            target=self.space,
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                },
            ),
            operator_id=self.structure_id,
        )


def _block_resources(
    count: int,
    reduced: int,
    output_bytes: int,
    dtype: np.dtype,
    budget: int,
    /,
    *,
    query_count: int,
    edge_count: int,
    row_assembly_bytes: int = 0,
    sketch: bool = False,
    maximum_block_size: int | None = None,
) -> tuple[int, dict[str, int]]:
    """Budget logical outer arrays, not compiler or native solver temporaries.

    The implicit derivative retains the reduced solutions for all mapped
    columns, not just the current block. Sketch RHS differentiation also
    retains the row-addressed signs. Count padded map outputs explicitly.
    """
    if budget < 0:
        raise ValueError("workspace_bytes must be nonnegative.")
    size = dtype.itemsize

    def estimate(block: int) -> dict[str, int]:
        padded = ((count + block - 1) // block) * block if reduced else 0
        active = block if reduced else 0
        items = {
            "output_bytes": output_bytes,
            "retained_query_bytes": query_count * padded * size,
            "retained_evidence_bytes": padded
            * (size + 2 * np.dtype(np.int32).itemsize + 1),
            "rhs_bytes": reduced * active * size,
            "potential_bytes": reduced * active * size,
            "query_block_bytes": query_count * active * size,
            "row_assembly_bytes": row_assembly_bytes if reduced else 0,
            "weight_bytes": edge_count * size if sketch and reduced else 0,
            "output_cotangent_bytes": output_bytes,
            "retained_potential_bytes": reduced * padded * size,
            "retained_sign_bytes": edge_count * padded * size if sketch else 0,
            "potential_cotangent_bytes": reduced * active * size,
            "adjoint_potential_bytes": reduced * active * size,
            "query_cotangent_bytes": query_count * active * size,
            "conductance_gradient_bytes": edge_count * size,
            "edge_gradient_block_bytes": edge_count * active * size,
            "rhs_cotangent_bytes": reduced * active * size if sketch else 0,
        }
        primal = sum(
            items[key]
            for key in (
                "output_bytes",
                "retained_query_bytes",
                "retained_evidence_bytes",
                "rhs_bytes",
                "potential_bytes",
                "query_block_bytes",
                "row_assembly_bytes",
                "weight_bytes",
            )
        )
        derivative = primal + sum(
            items[key]
            for key in (
                "output_cotangent_bytes",
                "retained_potential_bytes",
                "retained_sign_bytes",
                "potential_cotangent_bytes",
                "adjoint_potential_bytes",
                "query_cotangent_bytes",
                "conductance_gradient_bytes",
                "edge_gradient_block_bytes",
                "rhs_cotangent_bytes",
            )
        )
        items["primal_workspace_bytes"] = primal
        items["derivative_workspace_bytes"] = derivative
        items["budgeted_workspace_bytes"] = max(primal, derivative)
        return items

    minimum = estimate(1)
    if minimum["budgeted_workspace_bytes"] > budget:
        raise ValueError("Workspace byte budget cannot fit even one RHS column.")
    # Retained padded arrays are at least as large as at block size one.
    # This upper bound avoids scanning block sizes that cannot possibly fit.
    per_column = (
        (4 * reduced + 2 * query_count + edge_count + (reduced if sketch else 0)) * size
        if reduced
        else 0
    )
    block = min(
        count,
        1 + (budget - minimum["budgeted_workspace_bytes"]) // max(per_column, 1),
    )
    if maximum_block_size is not None:
        block = min(block, maximum_block_size)
    while block >= 1:
        items = estimate(block)
        if items["budgeted_workspace_bytes"] <= budget:
            return block, items
        block -= 1
    raise ValueError("Workspace byte budget cannot fit even one RHS column.")


def _solver_block_size(
    operator: SparseCoordinateOperator | None, policy: LinearSolvePolicy, count: int, /
) -> int:
    """Limit outer blocking by native per-column workspace and Krylov budgets."""
    if operator is None:
        return count
    symbolic = plan_linear_solve(
        LinearSystem(operator, problem_id=operator.operator_id),
        policy,
        rhs_layout=RHSLayout((1,)),
    )
    estimate = symbolic.candidates[-1]
    workspace = (
        estimate.solve_workspace_bytes_per_rhs
        + estimate.operator_apply_workspace_bytes_per_rhs
        + estimate.preconditioner_apply_workspace_bytes_per_rhs
    )
    block = count
    for available, required in (
        (policy.resources.workspace_bytes, workspace),
        (policy.resources.krylov_basis_bytes, estimate.krylov_basis_bytes_per_rhs),
    ):
        if required > 0:
            block = min(block, available // required)
    return block
