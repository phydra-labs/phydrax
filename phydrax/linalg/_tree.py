#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Linear-storage rooted-tree operators and exact leaf elimination."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._operators import (
    _AbstractCostedLinearOperator,
    _generic_adjoint,
    _id,
    _validate_action_dtype,
)
from ._properties import OperatorCapabilities, OperatorProperties
from ._spaces import AbstractVectorSpace, ArraySpace


class TreeTopology(StrictModule, NonTrainableState):
    """Host-validated rooted tree, independent of numerical coefficients.

    Parents use arbitrary vertex ordering and exactly one ``-1`` root. Topology
    construction, storage, and its postorder schedule require linear work/space.
    Construct once outside transformed numerical code and reuse the result.
    """

    parent_index: Array
    elimination_order: Array
    root_index: int = eqx.field(static=True)
    size: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)

    def __init__(self, parent_index: Sequence[int], /):
        parents = tuple(parent_index)
        count = len(parents)
        if not count:
            raise ValueError("A tree must contain at least one vertex.")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in parents
        ):
            raise TypeError("Tree parent indices must be host integers.")
        parents = tuple(int(value) for value in parents)
        roots = [index for index, parent in enumerate(parents) if parent == -1]
        if len(roots) != 1:
            raise ValueError("A tree requires exactly one -1 root.")
        children: list[list[int]] = [[] for _ in parents]
        for child, parent in enumerate(parents):
            if parent < -1 or parent >= count or parent == child:
                raise ValueError("Tree parent indices must refer to other vertices.")
            if parent >= 0:
                children[parent].append(child)
        root = roots[0]
        preorder: list[int] = []
        stack = [root]
        while stack:
            vertex = stack.pop()
            preorder.append(vertex)
            stack.extend(children[vertex])
        if len(preorder) != count:
            raise ValueError("Tree vertices must be connected and acyclic.")
        self.parent_index = jnp.asarray(parents, dtype=jnp.int32)
        self.elimination_order = jnp.asarray(preorder[:0:-1], dtype=jnp.int32)
        self.root_index = root
        self.size = count
        self.topology_id = canonical_fingerprint(
            {"kind": "tree-topology", "parents": parents}
        )


class TreeLinearOperator(_AbstractCostedLinearOperator):
    """Tree matrix stored in three length-n coefficient vectors.

    ``lower[child] = A[child, parent]`` and
    ``upper[child] = A[parent, child]``; root entries of these two vectors are
    unused. Distinct lower/upper values support exact Dirichlet rows and general
    nonsymmetric tree matrices. StructuredDirect uses no-pivot leaf elimination;
    unusable reduced pivots fail closed instead of densifying or perturbing A.
    """

    diagonal: Array
    lower: Array
    upper: Array
    topology: TreeTopology

    def __init__(
        self,
        diagonal: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        topology: TreeTopology,
        /,
        *,
        space: AbstractVectorSpace | None = None,
        operator_id: str | None = None,
    ):
        if not isinstance(topology, TreeTopology):
            raise TypeError("topology must be a prepared TreeTopology.")
        arrays = tuple(map(jnp.asarray, (diagonal, lower, upper)))
        if any(value.shape != (topology.size,) for value in arrays):
            raise ValueError(
                "Tree coefficient vectors must each have length topology.size."
            )
        dtype = jnp.result_type(*arrays, float)
        self.diagonal, self.lower, self.upper = (value.astype(dtype) for value in arrays)
        space_ = ArraySpace((topology.size,), dtype=dtype) if space is None else space
        if not isinstance(space_, AbstractVectorSpace) or space_.size != topology.size:
            raise ValueError("space must have the tree coordinate dimension.")
        _validate_action_dtype(dtype, space_, space_, "Tree operator")
        self.source = space_
        self.target = space_
        self.topology = topology
        self.batch_shape = ()
        self.properties = OperatorProperties()
        self.capabilities = OperatorCapabilities(
            transpose=True, adjoint=True, materialize=True, diagonal_assembly=True
        )
        self.operator_id = _id(
            operator_id,
            {"kind": "tree", "topology": topology.topology_id, "space": space_.space_id},
        )

    def _action(self, value: Array, lower: Array, upper: Array, /) -> Array:
        children = self.topology.elimination_order
        parents = self.topology.parent_index[children]
        result = self.diagonal * value
        result = result.at[children].add(lower[children] * value[parents])
        return result.at[parents].add(upper[children] * value[children])

    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        value = self.source.flatten(vector)
        return self.target.unflatten(self._action(value, self.lower, self.upper))

    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        value = self.target.flatten(vector)
        return self.source.unflatten(self._action(value, self.upper, self.lower))

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _generic_adjoint(self, vector)

    def _materialize(self, /) -> Array:
        children = self.topology.elimination_order
        parents = self.topology.parent_index[children]
        matrix = jnp.diag(self.diagonal)
        matrix = matrix.at[children, parents].set(self.lower[children])
        return matrix.at[parents, children].set(self.upper[children])

    def _assemble_diagonal(self, /) -> Array:
        return self.diagonal

    def _action_workspace_cost(self, /) -> tuple[int, str]:
        return 3 * self.diagonal.size * self.diagonal.dtype.itemsize, "tree-action"


class _TreeFactorization(StrictModule):
    diagonal: Array
    singular: Array


def _prepare_tree(operator: TreeLinearOperator, /) -> _TreeFactorization:
    children = operator.topology.elimination_order
    parents = operator.topology.parent_index[children]
    row_scale = jnp.abs(operator.diagonal)
    row_scale = row_scale.at[children].add(jnp.abs(operator.lower[children]))
    row_scale = row_scale.at[parents].add(jnp.abs(operator.upper[children]))
    threshold = jnp.finfo(operator.diagonal.real.dtype).eps * row_scale
    invalid = (
        jnp.any(~jnp.isfinite(operator.diagonal))
        | jnp.any(~jnp.isfinite(operator.lower[children]))
        | jnp.any(~jnp.isfinite(operator.upper[children]))
    )

    def eliminate(position, carry):
        diagonal, failed = carry
        child = children[position]
        parent = parents[position]
        pivot = diagonal[child]
        unusable = ~jnp.isfinite(pivot) | (jnp.abs(pivot) <= threshold[child])
        safe_pivot = jnp.where(unusable, jnp.ones_like(pivot), pivot)
        reduced = operator.upper[child] * (operator.lower[child] / safe_pivot)
        return diagonal.at[parent].add(-reduced), failed | unusable

    diagonal, invalid = (
        jax.lax.fori_loop(
            0, operator.topology.size - 1, eliminate, (operator.diagonal, invalid)
        )
        if operator.topology.size > 1
        else (operator.diagonal, invalid)
    )
    root = operator.topology.root_index
    invalid = (
        invalid
        | ~jnp.isfinite(diagonal[root])
        | (jnp.abs(diagonal[root]) <= threshold[root])
    )
    return _TreeFactorization(diagonal, invalid)


def _solve_tree(
    operator: TreeLinearOperator,
    factor: _TreeFactorization,
    rhs: Array,
    /,
    *,
    transposed: bool = False,
) -> tuple[Array, Array]:
    """Solve canonical [vertex, rhs] columns using one reusable factorization."""
    order = operator.topology.elimination_order
    lower, upper = (
        (operator.upper, operator.lower)
        if transposed
        else (operator.lower, operator.upper)
    )
    safe_diagonal = jnp.where(
        factor.singular, jnp.ones_like(factor.diagonal), factor.diagonal
    )

    def reduce_rhs(position, reduced):
        child = order[position]
        parent = operator.topology.parent_index[child]
        return reduced.at[parent].add(
            -(upper[child] / safe_diagonal[child]) * reduced[child]
        )

    count = operator.topology.size
    reduced = jax.lax.fori_loop(0, count - 1, reduce_rhs, rhs) if count > 1 else rhs
    root = operator.topology.root_index
    value = jnp.zeros_like(rhs).at[root].set(reduced[root] / safe_diagonal[root])

    def substitute(position, solution):
        child = order[count - 2 - position]
        parent = operator.topology.parent_index[child]
        return solution.at[child].set(
            (reduced[child] - lower[child] * solution[parent]) / safe_diagonal[child]
        )

    value = jax.lax.fori_loop(0, count - 1, substitute, value) if count > 1 else value
    failed = factor.singular | jnp.any(~jnp.isfinite(value))
    return jnp.where(failed, jnp.full_like(value, jnp.nan), value), failed


def _implicit_tree_value(
    operator: TreeLinearOperator,
    rhs: Array,
    initial: Array,
    factor: _TreeFactorization,
    /,
) -> Array:
    """Mathematical root derivative with linear-storage primal/transpose actions."""
    fixed_operator = jax.tree.map(jax.lax.stop_gradient, operator)
    fixed_factor = jax.tree.map(jax.lax.stop_gradient, factor)

    def residual(value):
        return operator.mv_block(value) - rhs

    def tangent_solve(linearized, target):
        return jax.lax.custom_linear_solve(
            linearized,
            target,
            solve=lambda _, right: _solve_tree(fixed_operator, fixed_factor, right)[0],
            transpose_solve=lambda _, right: _solve_tree(
                fixed_operator, fixed_factor, right, transposed=True
            )[0],
        )

    return jax.lax.custom_root(
        residual,
        jax.lax.stop_gradient(initial),
        solve=lambda _, value: value,
        tangent_solve=tangent_solve,
    )


__all__ = ["TreeLinearOperator", "TreeTopology"]
