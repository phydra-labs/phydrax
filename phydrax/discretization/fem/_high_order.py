#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import product
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._reference import FiniteElementSpec


NodeSet = Literal["equispaced", "gauss-lobatto"]


def _nodes(order: int, node_set: NodeSet) -> np.ndarray:
    if node_set == "equispaced":
        return np.linspace(0.0, 1.0, order + 1)
    if node_set == "gauss-lobatto":
        if order == 1:
            return np.asarray([0.0, 1.0])
        roots = np.polynomial.legendre.Legendre.basis(order).deriv().roots()
        return 0.5 * (np.concatenate(([-1.0], roots, [1.0])) + 1.0)
    raise ValueError("Unknown nodal point set.")


def lagrange_1d_tabulation(nodes: ArrayLike, points: ArrayLike, /) -> tuple[Array, Array]:
    nodes_ = jnp.asarray(nodes)
    points_ = jnp.asarray(points).reshape((-1,))
    count = nodes_.shape[0]
    values = []
    gradients = []
    for i in range(count):
        factors = []
        for j in range(count):
            if i != j:
                factors.append((points_ - nodes_[j]) / (nodes_[i] - nodes_[j]))
        value = jnp.ones_like(points_)
        for factor in factors:
            value = value * factor
        gradient = jnp.zeros_like(points_)
        for omitted in range(len(factors)):
            product = jnp.ones_like(points_)
            for index, factor in enumerate(factors):
                if index != omitted:
                    product = product * factor
            denominator_index = omitted if omitted < i else omitted + 1
            gradient = gradient + product / (nodes_[i] - nodes_[denominator_index])
        values.append(value)
        gradients.append(gradient)
    return jnp.stack(tuple(values), axis=-1), jnp.stack(tuple(gradients), axis=-1)


class ReferenceNodalFamily(StrictModule, NonTrainableState):
    cell_kind: str = eqx.field(static=True)
    order: int = eqx.field(static=True)
    node_set: NodeSet = eqx.field(static=True)
    axis_nodes: Array
    family_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_kind: Literal["quadrilateral"],
        order: int,
        /,
        *,
        node_set: NodeSet = "gauss-lobatto",
    ):
        cell = str(cell_kind)
        order_ = int(order)
        if cell != "quadrilateral" or order_ < 1:
            raise ValueError(
                "Initial arbitrary-order family supports quadrilaterals, p >= 1."
            )
        points = _nodes(order_, node_set)
        self.cell_kind = cell
        self.order = order_
        self.node_set = node_set
        self.axis_nodes = jnp.asarray(points)
        self.family_id = canonical_fingerprint(
            {
                "kind": "reference-nodal-family",
                "cell_kind": cell,
                "order": order_,
                "node_set": node_set,
                "axis_nodes": array_tree_fingerprint(points),
            }
        )

    def tabulate(self, points: ArrayLike, /) -> tuple[Array, Array]:
        points_ = jnp.asarray(points)
        if points_.ndim != 2 or points_.shape[1] != 2:
            raise ValueError("Quadrilateral points must have shape (count, 2).")
        bx, gx = lagrange_1d_tabulation(self.axis_nodes, points_[:, 0])
        by, gy = lagrange_1d_tabulation(self.axis_nodes, points_[:, 1])
        values = oe.contract("qi,qj->qij", bx, by).reshape((points_.shape[0], -1))
        grad_x = oe.contract("qi,qj->qij", gx, by).reshape((points_.shape[0], -1))
        grad_y = oe.contract("qi,qj->qij", bx, gy).reshape((points_.shape[0], -1))
        return values, jnp.stack((grad_x, grad_y), axis=-1)

    def finite_element(self, /) -> FiniteElementSpec:
        p = self.order
        grid = np.stack(
            np.meshgrid(
                np.asarray(self.axis_nodes), np.asarray(self.axis_nodes), indexing="ij"
            ),
            axis=-1,
        ).reshape((-1, 2))
        index = np.arange((p + 1) ** 2, dtype=np.int32).reshape((p + 1, p + 1))
        vertices = (
            (int(index[0, 0]),),
            (int(index[p, 0]),),
            (int(index[p, p]),),
            (int(index[0, p]),),
        )
        edges = (
            tuple(int(value) for value in index[1:p, 0]),
            tuple(int(value) for value in index[p, 1:p]),
            tuple(int(value) for value in index[p - 1 : 0 : -1, p]),
            tuple(int(value) for value in index[0, p - 1 : 0 : -1]),
        )
        interior = tuple(int(value) for value in index[1:p, 1:p].reshape((-1,)))
        return FiniteElementSpec(
            "TensorProductLagrange",
            "quadrilateral",
            p,
            grid,
            (vertices, edges, (interior,)),
            conformity="H1",
            tabulator=self.tabulate,
            tabulator_id=self.family_id,
        )


class TensorProductTabulation(StrictModule, NonTrainableState):
    basis_x: Array
    basis_y: Array
    gradient_x: Array
    gradient_y: Array
    tabulation_id: str = eqx.field(static=True)

    def __init__(
        self,
        family: ReferenceNodalFamily,
        points_x: ArrayLike,
        points_y: ArrayLike,
        /,
    ):
        if not isinstance(family, ReferenceNodalFamily):
            raise TypeError("family must be ReferenceNodalFamily.")
        bx, gx = lagrange_1d_tabulation(family.axis_nodes, points_x)
        by, gy = lagrange_1d_tabulation(family.axis_nodes, points_y)
        self.basis_x = bx
        self.basis_y = by
        self.gradient_x = gx
        self.gradient_y = gy
        self.tabulation_id = canonical_fingerprint(
            {
                "kind": "tensor-product-tabulation",
                "family": family.family_id,
                "points_x": array_tree_fingerprint(np.asarray(points_x)),
                "points_y": array_tree_fingerprint(np.asarray(points_y)),
            }
        )


class SumFactorizationPlan(StrictModule, NonTrainableState):
    tabulation: TensorProductTabulation
    plan_id: str = eqx.field(static=True)

    def __init__(self, tabulation: TensorProductTabulation, /):
        if not isinstance(tabulation, TensorProductTabulation):
            raise TypeError("tabulation must be TensorProductTabulation.")
        self.tabulation = tabulation
        self.plan_id = canonical_fingerprint(
            {"kind": "sum-factorization-plan", "tabulation": tabulation.tabulation_id}
        )

    def interpolate(self, coefficients: ArrayLike, /) -> Array:
        values = jnp.asarray(coefficients)
        expected = (
            self.tabulation.basis_x.shape[1],
            self.tabulation.basis_y.shape[1],
        )
        if values.shape[-2:] != expected:
            raise ValueError("Tensor coefficients have incompatible nodal axes.")
        first = oe.contract("...ij,qj->...iq", values, self.tabulation.basis_y)
        return oe.contract("pi,...iq->...pq", self.tabulation.basis_x, first)

    def interpolate_transpose(self, values: ArrayLike, /) -> Array:
        quadrature = jnp.asarray(values)
        expected = (
            self.tabulation.basis_x.shape[0],
            self.tabulation.basis_y.shape[0],
        )
        if quadrature.shape[-2:] != expected:
            raise ValueError("Tensor quadrature values have incompatible axes.")
        first = oe.contract("pi,...pq->...iq", self.tabulation.basis_x, quadrature)
        return oe.contract("...iq,qj->...ij", first, self.tabulation.basis_y)

    def gradient(self, coefficients: ArrayLike, /) -> Array:
        values = jnp.asarray(coefficients)
        dx = oe.contract(
            "pi,...ij,qj->...pq",
            self.tabulation.gradient_x,
            values,
            self.tabulation.basis_y,
        )
        dy = oe.contract(
            "pi,...ij,qj->...pq",
            self.tabulation.basis_x,
            values,
            self.tabulation.gradient_y,
        )
        return jnp.stack((dx, dy), axis=-1)

    def gradient_transpose(self, values: ArrayLike, /) -> Array:
        gradient = jnp.asarray(values)
        expected = (
            self.tabulation.basis_x.shape[0],
            self.tabulation.basis_y.shape[0],
            2,
        )
        if gradient.shape[-3:] != expected:
            raise ValueError("Tensor quadrature gradients have incompatible axes.")
        dx = oe.contract(
            "pi,...pq,qj->...ij",
            self.tabulation.gradient_x,
            gradient[..., 0],
            self.tabulation.basis_y,
        )
        dy = oe.contract(
            "pi,...pq,qj->...ij",
            self.tabulation.basis_x,
            gradient[..., 1],
            self.tabulation.gradient_y,
        )
        return dx + dy


class QuadratureChunkPolicy(StrictModule, NonTrainableState):
    chunk_size: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(self, chunk_size: int, /):
        size = int(chunk_size)
        if size <= 0:
            raise ValueError("Quadrature chunk size must be positive.")
        self.chunk_size = size
        self.policy_id = canonical_fingerprint(
            {"kind": "quadrature-chunk-policy", "chunk_size": size}
        )

    def chunks(self, count: int, /) -> tuple[tuple[int, int], ...]:
        count_ = int(count)
        if count_ < 0:
            raise ValueError("Quadrature count must be non-negative.")
        return tuple(
            (start, min(start + self.chunk_size, count_))
            for start in range(0, count_, self.chunk_size)
        )


def local_diagonal(
    local_matrices: ArrayLike, local_dofs: ArrayLike, size: int, /
) -> Array:
    matrices = jnp.asarray(local_matrices)
    dofs = jnp.asarray(local_dofs, dtype=jnp.int32)
    if matrices.shape[:2] != dofs.shape or matrices.shape[-1] != dofs.shape[-1]:
        raise ValueError("Local matrices/DOF routes have incompatible shapes.")
    diagonal = jnp.diagonal(matrices, axis1=-2, axis2=-1)
    return jnp.zeros((int(size),), dtype=matrices.dtype).at[dofs].add(diagonal)


class SimplexNodalFamily(StrictModule, NonTrainableState):
    cell_kind: str = eqx.field(static=True)
    order: int = eqx.field(static=True)
    nodes: Array
    multiindices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    monomial_powers: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    coefficients: Array
    family_id: str = eqx.field(static=True)

    def __init__(self, cell_kind: str, order: int, /):
        cell = str(cell_kind)
        p = int(order)
        dimension = {"triangle": 2, "tetrahedron": 3}.get(cell)
        if dimension is None or p < 1:
            raise ValueError(
                "Simplex nodal families require triangle/tetrahedron and p>=1."
            )
        barycentric = tuple(
            index
            for index in product(range(p + 1), repeat=dimension + 1)
            if sum(index) == p
        )
        nodes = np.asarray([index[1:] for index in barycentric], dtype=float) / p
        powers = tuple(
            index for index in product(range(p + 1), repeat=dimension) if sum(index) <= p
        )
        vandermonde = np.stack(
            [np.prod(nodes ** np.asarray(power), axis=1) for power in powers],
            axis=1,
        )
        self.cell_kind = cell
        self.order = p
        self.nodes = jnp.asarray(nodes)
        self.multiindices = barycentric
        self.monomial_powers = powers
        self.coefficients = jnp.asarray(np.linalg.inv(vandermonde))
        self.family_id = canonical_fingerprint(
            {"kind": "simplex-nodal-family", "cell": cell, "order": p}
        )

    def tabulate(self, points: ArrayLike, /) -> tuple[Array, Array]:
        points_ = jnp.asarray(points)
        dimension = points_.shape[-1]
        monomials = jnp.stack(
            tuple(
                jnp.prod(points_ ** jnp.asarray(power), axis=-1)
                for power in self.monomial_powers
            ),
            axis=-1,
        )
        values = monomials @ self.coefficients
        derivatives = []
        for axis in range(dimension):
            terms = []
            for power in self.monomial_powers:
                exponent = power[axis]
                reduced = list(power)
                reduced[axis] = max(exponent - 1, 0)
                terms.append(
                    exponent * jnp.prod(points_ ** jnp.asarray(reduced), axis=-1)
                )
            derivatives.append(jnp.stack(tuple(terms), axis=-1) @ self.coefficients)
        return values, jnp.stack(tuple(derivatives), axis=-1)

    def finite_element(self) -> FiniteElementSpec:
        from ._reference_topology import reference_cell_topology

        topology = reference_cell_topology(self.cell_kind)
        entity_dofs = [[[] for _ in entities] for entities in topology.entities]
        entity_sets = [
            tuple(frozenset(entity) for entity in entities)
            for entities in topology.entities
        ]
        for dof, alpha in enumerate(self.multiindices):
            support = frozenset(index for index, value in enumerate(alpha) if value > 0)
            dimension = len(support) - 1
            entity = entity_sets[dimension].index(support)
            entity_dofs[dimension][entity].append(dof)
        return FiniteElementSpec(
            "SimplexLagrange",
            self.cell_kind,
            self.order,
            self.nodes,
            tuple(
                tuple(tuple(values) for values in dimension) for dimension in entity_dofs
            ),
            conformity="H1",
            tabulator=self.tabulate,
            tabulator_id=self.family_id,
        )


__all__ = [
    "NodeSet",
    "QuadratureChunkPolicy",
    "ReferenceNodalFamily",
    "SumFactorizationPlan",
    "SimplexNodalFamily",
    "TensorProductTabulation",
    "lagrange_1d_tabulation",
    "local_diagonal",
]
