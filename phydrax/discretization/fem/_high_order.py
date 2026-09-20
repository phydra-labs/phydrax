#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import comb, factorial, sqrt
from numbers import Integral
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._interpolation import (
    barycentric_basis,
    barycentric_differentiation_matrix,
)
from ..._polynomial._orthogonal import legendre_rule_data
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._reference import FiniteElementSpec


NodeSet = Literal["equispaced", "gauss-lobatto"]
TensorOrder = int | tuple[int, ...]


def _normalize_orders(
    cell_kind: str, order: TensorOrder, /
) -> tuple[str, tuple[int, ...]]:
    cell = str(cell_kind)
    dimensions = {"interval": 1, "quadrilateral": 2, "hexahedron": 3}
    if cell not in dimensions:
        raise ValueError(
            "Tensor nodal families require interval, quadrilateral, or hexahedron."
        )
    if isinstance(order, bool):
        raise TypeError("Tensor polynomial orders must be integers.")
    if isinstance(order, Integral):
        orders = (int(order),) * dimensions[cell]
    elif isinstance(order, tuple):
        if len(order) != dimensions[cell] or any(
            isinstance(value, bool) or not isinstance(value, Integral) for value in order
        ):
            raise ValueError(
                "Anisotropic polynomial orders must match the cell dimension."
            )
        orders = tuple(order)
    else:
        raise TypeError("Tensor polynomial order must be an integer or integer tuple.")
    if any(value < 0 for value in orders):
        raise ValueError("Tensor polynomial orders must be nonnegative.")
    return cell, orders


def _axis_data(
    order: int, node_set: NodeSet, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    count = order + 1
    if node_set == "equispaced":
        nodes = np.linspace(0.0, 1.0, count)
        barycentric = np.asarray(
            [(-1.0) ** index * comb(order, index) for index in range(count)],
            dtype=np.float64,
        )
        barycentric /= np.max(np.abs(barycentric))
        return nodes, barycentric, None
    if node_set == "gauss-lobatto":
        rule = legendre_rule_data(count, "lobatto")
        nodes = 0.5 * (np.asarray(rule.nodes, dtype=np.float64) + 1.0)
        quadrature = 0.5 * np.asarray(rule.weights, dtype=np.float64)
        barycentric = (-1.0) ** np.arange(count) * np.sqrt(
            np.asarray(rule.weights, dtype=np.float64)
        )
        barycentric /= np.max(np.abs(barycentric))
        return nodes, barycentric, quadrature
    raise ValueError("Unknown nodal point set.")


def _default_barycentric_weights(nodes: Array, /) -> Array:
    count = nodes.shape[0]
    differences = nodes[:, None] - nodes[None, :]
    safe = differences + jnp.eye(count, dtype=nodes.dtype)
    weights = jnp.reciprocal(jnp.prod(safe, axis=1))
    return weights / jnp.max(jnp.abs(weights))


def lagrange_1d_tabulation(
    nodes: ArrayLike,
    points: ArrayLike,
    /,
    *,
    barycentric_weights: ArrayLike | None = None,
) -> tuple[Array, Array]:
    """Evaluate a nodal basis and its derivative through barycentric data."""

    nodes_ = jnp.asarray(nodes)
    points_ = jnp.asarray(points).reshape((-1,))
    if nodes_.ndim != 1 or nodes_.shape[0] == 0:
        raise ValueError("Lagrange nodes must be a nonempty vector.")
    dtype = jnp.result_type(nodes_, points_, jnp.float64)
    nodes_ = nodes_.astype(dtype)
    points_ = points_.astype(dtype)
    weights = (
        _default_barycentric_weights(nodes_)
        if barycentric_weights is None
        else jnp.asarray(barycentric_weights, dtype=dtype)
    )
    if weights.shape != nodes_.shape:
        raise ValueError("Barycentric weights must match the Lagrange nodes.")
    values = jax.vmap(barycentric_basis, in_axes=(0, None, None))(
        points_, nodes_, weights
    )
    differentiation = barycentric_differentiation_matrix(nodes_, weights=weights)
    gradients = ein.contract("qi,ij->qj", values, differentiation)
    return values, gradients


def _dense_tensor_tabulation(
    basis: tuple[Array, ...], gradients: tuple[Array, ...], /
) -> tuple[Array, Array]:
    if len(basis) == 1:
        values = basis[0]
        components = (gradients[0],)
    elif len(basis) == 2:
        values = ein.contract("qi,qj->qij", basis[0], basis[1])
        components = (
            ein.contract("qi,qj->qij", gradients[0], basis[1]),
            ein.contract("qi,qj->qij", basis[0], gradients[1]),
        )
    elif len(basis) == 3:
        values = ein.contract("qi,qj,qk->qijk", basis[0], basis[1], basis[2])
        components = (
            ein.contract("qi,qj,qk->qijk", gradients[0], basis[1], basis[2]),
            ein.contract("qi,qj,qk->qijk", basis[0], gradients[1], basis[2]),
            ein.contract("qi,qj,qk->qijk", basis[0], basis[1], gradients[2]),
        )
    else:
        raise ValueError("Tensor tabulation requires one, two, or three axes.")
    point_count = values.shape[0]
    return values.reshape((point_count, -1)), jnp.stack(
        tuple(component.reshape((point_count, -1)) for component in components),
        axis=-1,
    )


def _quad_entity_dofs(
    index: np.ndarray, orders: tuple[int, ...], /
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    px, py = orders
    vertices = (
        (int(index[0, 0]),),
        (int(index[px, 0]),),
        (int(index[px, py]),),
        (int(index[0, py]),),
    )
    edges = (
        tuple(index[1:px, 0]),
        tuple(index[px, 1:py]),
        tuple(index[px - 1 : 0 : -1, py]),
        tuple(index[0, py - 1 : 0 : -1]),
    )
    interior = tuple(index[1:px, 1:py].reshape((-1,)))
    return vertices, edges, (interior,)


def _hex_entity_dofs(
    index: np.ndarray, orders: tuple[int, ...], /
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    px, py, pz = orders
    vertices = (
        (int(index[0, 0, 0]),),
        (int(index[px, 0, 0]),),
        (int(index[px, py, 0]),),
        (int(index[0, py, 0]),),
        (int(index[0, 0, pz]),),
        (int(index[px, 0, pz]),),
        (int(index[px, py, pz]),),
        (int(index[0, py, pz]),),
    )
    edges = (
        tuple(index[1:px, 0, 0]),
        tuple(index[px, 1:py, 0]),
        tuple(index[px - 1 : 0 : -1, py, 0]),
        tuple(index[0, py - 1 : 0 : -1, 0]),
        tuple(index[1:px, 0, pz]),
        tuple(index[px, 1:py, pz]),
        tuple(index[px - 1 : 0 : -1, py, pz]),
        tuple(index[0, py - 1 : 0 : -1, pz]),
        tuple(index[0, 0, 1:pz]),
        tuple(index[px, 0, 1:pz]),
        tuple(index[px, py, 1:pz]),
        tuple(index[0, py, 1:pz]),
    )
    faces = (
        tuple(int(index[x, y, 0]) for y in range(1, py) for x in range(1, px)),
        tuple(int(index[x, y, pz]) for x in range(1, px) for y in range(1, py)),
        tuple(int(index[x, 0, z]) for x in range(1, px) for z in range(1, pz)),
        tuple(int(index[px, y, z]) for y in range(1, py) for z in range(1, pz)),
        tuple(int(index[x, py, z]) for x in range(px - 1, 0, -1) for z in range(1, pz)),
        tuple(int(index[0, y, z]) for y in range(py - 1, 0, -1) for z in range(1, pz)),
    )
    interior = tuple(index[1:px, 1:py, 1:pz].reshape((-1,)))
    return vertices, edges, faces, (interior,)


def _interval_entity_dofs(
    index: np.ndarray, orders: tuple[int, ...], /
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    order = orders[0]
    if order == 0:
        return (((), ()), ((int(index[0]),),))
    return (
        ((int(index[0]),), (int(index[-1]),)),
        (tuple(index[1:-1]),),
    )


class ReferenceNodalFamily(StrictModule, NonTrainableState):
    """An anisotropic tensor-product nodal family on interval, quad, or hex."""

    cell_kind: str = eqx.field(static=True)
    orders: tuple[int, ...] = eqx.field(static=True)
    node_set: NodeSet = eqx.field(static=True)
    nodes_by_axis: tuple[Array, ...]
    barycentric_weights_by_axis: tuple[Array, ...]
    quadrature_weights_by_axis: tuple[Array | None, ...]
    family_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_kind: Literal["interval", "quadrilateral", "hexahedron"],
        order: TensorOrder,
        /,
        *,
        node_set: NodeSet = "gauss-lobatto",
    ):
        cell, orders = _normalize_orders(cell_kind, order)
        axis_data = tuple(_axis_data(value, node_set) for value in orders)
        nodes = tuple(jnp.asarray(data[0]) for data in axis_data)
        barycentric = tuple(jnp.asarray(data[1]) for data in axis_data)
        quadrature = tuple(
            None if data[2] is None else jnp.asarray(data[2]) for data in axis_data
        )
        self.cell_kind = cell
        self.orders = orders
        self.node_set = node_set
        self.nodes_by_axis = nodes
        self.barycentric_weights_by_axis = barycentric
        self.quadrature_weights_by_axis = quadrature
        self.family_id = canonical_fingerprint(
            {
                "kind": "reference-nodal-family",
                "cell_kind": cell,
                "orders": orders,
                "node_set": node_set,
                "nodes_by_axis": tuple(
                    array_tree_fingerprint(data[0]) for data in axis_data
                ),
                "barycentric_weights_by_axis": tuple(
                    array_tree_fingerprint(data[1]) for data in axis_data
                ),
                "quadrature_weights_by_axis": tuple(
                    None if data[2] is None else array_tree_fingerprint(data[2])
                    for data in axis_data
                ),
            }
        )

    @property
    def dimension(self) -> int:
        return len(self.orders)

    @property
    def nodal_shape(self) -> tuple[int, ...]:
        return tuple(order + 1 for order in self.orders)

    def tabulate(self, points: ArrayLike, /) -> tuple[Array, Array]:
        points_ = jnp.asarray(points)
        if points_.ndim != 2 or points_.shape[1] != self.dimension:
            raise ValueError(
                "Tensor reference points must have shape (count, cell_dimension)."
            )
        factors = tuple(
            lagrange_1d_tabulation(
                nodes,
                points_[:, axis],
                barycentric_weights=weights,
            )
            for axis, (nodes, weights) in enumerate(
                zip(self.nodes_by_axis, self.barycentric_weights_by_axis)
            )
        )
        return _dense_tensor_tabulation(
            tuple(factor[0] for factor in factors),
            tuple(factor[1] for factor in factors),
        )

    def finite_element(self, /) -> FiniteElementSpec:
        grid = np.stack(
            np.meshgrid(
                *(np.asarray(nodes) for nodes in self.nodes_by_axis),
                indexing="ij",
            ),
            axis=-1,
        ).reshape((-1, self.dimension))
        index = np.arange(np.prod(self.nodal_shape), dtype=np.int32).reshape(
            self.nodal_shape
        )
        if self.cell_kind == "interval":
            entity_dofs = _interval_entity_dofs(index, self.orders)
        elif self.cell_kind == "quadrilateral":
            entity_dofs = _quad_entity_dofs(index, self.orders)
        else:
            entity_dofs = _hex_entity_dofs(index, self.orders)
        return FiniteElementSpec(
            "TensorProductLagrange",
            self.cell_kind,
            max(self.orders),
            grid,
            entity_dofs,
            conformity="H1",
            representation="point_value",
            tabulator=self.tabulate,
            tabulator_id=self.family_id,
        )


class TensorProductTabulation(StrictModule, NonTrainableState):
    """One-dimensional factors for tensor-product basis actions."""

    basis_factors: tuple[Array, ...]
    gradient_factors: tuple[Array, ...]
    tabulation_id: str = eqx.field(static=True)

    def __init__(
        self,
        family: ReferenceNodalFamily,
        points_by_axis: tuple[ArrayLike, ...],
        /,
    ):
        if not isinstance(family, ReferenceNodalFamily):
            raise TypeError("family must be ReferenceNodalFamily.")
        if (
            not isinstance(points_by_axis, tuple)
            or len(points_by_axis) != family.dimension
        ):
            raise ValueError("Tensor points must be one tuple entry per cell axis.")
        point_arrays = tuple(jnp.asarray(points) for points in points_by_axis)
        if any(points.ndim != 1 or points.shape[0] == 0 for points in point_arrays):
            raise ValueError("Tensor point factors must be nonempty vectors.")
        factors = tuple(
            lagrange_1d_tabulation(
                nodes,
                points,
                barycentric_weights=weights,
            )
            for nodes, weights, points in zip(
                family.nodes_by_axis,
                family.barycentric_weights_by_axis,
                point_arrays,
            )
        )
        self.basis_factors = tuple(factor[0] for factor in factors)
        self.gradient_factors = tuple(factor[1] for factor in factors)
        self.tabulation_id = canonical_fingerprint(
            {
                "kind": "tensor-product-tabulation",
                "family": family.family_id,
                "points_by_axis": tuple(
                    array_tree_fingerprint(np.asarray(points)) for points in point_arrays
                ),
            }
        )

    @property
    def dimension(self) -> int:
        return len(self.basis_factors)

    @property
    def nodal_shape(self) -> tuple[int, ...]:
        return tuple(factor.shape[1] for factor in self.basis_factors)

    @property
    def evaluation_shape(self) -> tuple[int, ...]:
        return tuple(factor.shape[0] for factor in self.basis_factors)


def _factorized_forward(factors: tuple[Array, ...], values: Array, /) -> Array:
    if len(factors) == 1:
        return ein.contract("ai,...i->...a", factors[0], values)
    if len(factors) == 2:
        return ein.contract("ai,...ij,bj->...ab", factors[0], values, factors[1])
    if len(factors) == 3:
        return ein.contract(
            "ai,...ijk,bj,ck->...abc",
            factors[0],
            values,
            factors[1],
            factors[2],
        )
    raise ValueError("Factorized actions require one, two, or three tensor axes.")


def _factorized_transpose(factors: tuple[Array, ...], values: Array, /) -> Array:
    if len(factors) == 1:
        return ein.contract("ai,...a->...i", factors[0], values)
    if len(factors) == 2:
        return ein.contract("ai,...ab,bj->...ij", factors[0], values, factors[1])
    if len(factors) == 3:
        return ein.contract(
            "ai,...abc,bj,ck->...ijk",
            factors[0],
            values,
            factors[1],
            factors[2],
        )
    raise ValueError("Factorized actions require one, two, or three tensor axes.")


class SumFactorizationPlan(StrictModule, NonTrainableState):
    """Dense-equivalent tensor interpolation and reference-gradient actions."""

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
        if values.shape[-self.tabulation.dimension :] != self.tabulation.nodal_shape:
            raise ValueError("Tensor coefficients have incompatible nodal axes.")
        return _factorized_forward(self.tabulation.basis_factors, values)

    def interpolate_transpose(self, values: ArrayLike, /) -> Array:
        quadrature = jnp.asarray(values)
        if (
            quadrature.shape[-self.tabulation.dimension :]
            != self.tabulation.evaluation_shape
        ):
            raise ValueError("Tensor quadrature values have incompatible axes.")
        return _factorized_transpose(self.tabulation.basis_factors, quadrature)

    def gradient(self, coefficients: ArrayLike, /) -> Array:
        values = jnp.asarray(coefficients)
        if values.shape[-self.tabulation.dimension :] != self.tabulation.nodal_shape:
            raise ValueError("Tensor coefficients have incompatible nodal axes.")
        components = []
        for axis in range(self.tabulation.dimension):
            factors = list(self.tabulation.basis_factors)
            factors[axis] = self.tabulation.gradient_factors[axis]
            components.append(_factorized_forward(tuple(factors), values))
        return jnp.stack(tuple(components), axis=-1)

    def gradient_transpose(self, values: ArrayLike, /) -> Array:
        gradient = jnp.asarray(values)
        expected = self.tabulation.evaluation_shape + (self.tabulation.dimension,)
        if gradient.shape[-len(expected) :] != expected:
            raise ValueError("Tensor quadrature gradients have incompatible axes.")
        terms = []
        for axis in range(self.tabulation.dimension):
            factors = list(self.tabulation.basis_factors)
            factors[axis] = self.tabulation.gradient_factors[axis]
            terms.append(_factorized_transpose(tuple(factors), gradient[..., axis]))
        return sum(terms[1:], start=terms[0])


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


_ALPHA_OPT_2D = (
    0.0,
    0.0,
    1.4152,
    0.1001,
    0.2751,
    0.9800,
    1.0999,
    1.2832,
    1.3648,
    1.4773,
    1.4959,
    1.5743,
    1.5770,
    1.6223,
    1.6258,
)
_ALPHA_OPT_3D = (
    0.0,
    0.0,
    0.0,
    0.1002,
    1.1332,
    1.5608,
    1.3413,
    1.2577,
    1.1603,
    1.10153,
    0.6080,
    0.4523,
    0.8856,
    0.8717,
    0.9655,
)
_EQUILATERAL_VERTICES = {
    2: np.asarray(
        ((-1.0, -1.0 / sqrt(3.0)), (1.0, -1.0 / sqrt(3.0)), (0.0, 2.0 / sqrt(3.0)))
    ),
    3: np.asarray(
        (
            (-1.0, -1.0 / sqrt(3.0), -1.0 / sqrt(6.0)),
            (1.0, -1.0 / sqrt(3.0), -1.0 / sqrt(6.0)),
            (0.0, 2.0 / sqrt(3.0), -1.0 / sqrt(6.0)),
            (0.0, 0.0, 3.0 / sqrt(6.0)),
        )
    ),
}


def _simplex_node_tuples(order: int, dimension: int, /) -> tuple[tuple[int, ...], ...]:
    def generate(maximum: int, length: int):
        if length == 0:
            yield ()
            return
        for value in range(maximum + 1):
            for remainder in generate(maximum - value, length - 1):
                yield (*remainder, value)

    return tuple(generate(order, dimension))


def _unit_to_barycentric(unit: np.ndarray, /) -> np.ndarray:
    tail = 0.5 * (unit + 1.0)
    return np.vstack((1.0 - np.sum(tail, axis=0), tail))


def _barycentric_to_equilateral(barycentric: np.ndarray, /) -> np.ndarray:
    return _EQUILATERAL_VERTICES[len(barycentric) - 1].T @ barycentric


def _equilateral_to_unit(equilateral: np.ndarray, /) -> np.ndarray:
    dimension = len(equilateral)
    if dimension == 2:
        matrix = np.asarray(((1.0, -1.0 / sqrt(3.0)), (0.0, 2.0 / sqrt(3.0))))
        offset = np.asarray((-1.0 / 3.0, -1.0 / 3.0))
    elif dimension == 3:
        matrix = np.asarray(
            (
                (1.0, -1.0 / sqrt(3.0), -1.0 / sqrt(6.0)),
                (0.0, 2.0 / sqrt(3.0), -1.0 / sqrt(6.0)),
                (0.0, 0.0, sqrt(6.0) / 2.0),
            )
        )
        offset = np.asarray((-0.5, -0.5, -0.5))
    else:
        raise ValueError("Warp-and-blend nodes require dimension two or three.")
    return matrix @ equilateral + offset[:, None]


def _warp_factor(order: int, output_nodes: np.ndarray, /) -> np.ndarray:
    lobatto = np.asarray(legendre_rule_data(order + 1, "lobatto").nodes)
    equispaced = np.linspace(-1.0, 1.0, order + 1)
    equispaced_vandermonde = np.polynomial.legendre.legvander(equispaced, order)
    output_vandermonde = np.polynomial.legendre.legvander(output_nodes, order)
    interpolation = np.linalg.solve(equispaced_vandermonde.T, output_vandermonde.T).T
    warp = interpolation @ (lobatto - equispaced)
    interior = (np.abs(output_nodes) < 1.0 - 1.0e-10).astype(np.float64)
    scale = 1.0 - (interior * output_nodes) ** 2
    return warp / scale + warp * (interior - 1.0)


def _equilateral_shift_2d(
    order: int, barycentric: np.ndarray, alpha: float, /
) -> np.ndarray:
    vertices = _EQUILATERAL_VERTICES[2]
    result = np.zeros((2, barycentric.shape[1]), dtype=np.float64)
    for first in range(3):
        second, third = tuple(index for index in range(3) if index != first)
        blend = 4.0 * barycentric[second] * barycentric[third]
        warp_factor = _warp_factor(order, barycentric[second] - barycentric[third])
        warp = blend * warp_factor * (1.0 + (alpha * barycentric[first]) ** 2)
        tangent = vertices[second] - vertices[third]
        tangent /= np.linalg.norm(tangent)
        result += tangent[:, None] * warp[None, :]
    return result


def _warp_and_blend_nodes_2d(
    order: int, node_tuples: tuple[tuple[int, ...], ...], /
) -> np.ndarray:
    alpha = _ALPHA_OPT_2D[order - 1] if order <= len(_ALPHA_OPT_2D) else 5.0 / 3.0
    unit_nodes = (np.asarray(node_tuples, dtype=np.float64) / order * 2.0 - 1.0).T
    barycentric = _unit_to_barycentric(unit_nodes)
    equilateral = _barycentric_to_equilateral(barycentric)
    return _equilateral_to_unit(
        equilateral + _equilateral_shift_2d(order, barycentric, alpha)
    )


def _warp_and_blend_nodes_3d(
    order: int, node_tuples: tuple[tuple[int, ...], ...], /
) -> np.ndarray:
    alpha = _ALPHA_OPT_3D[order - 1] if order <= len(_ALPHA_OPT_3D) else 1.0
    unit_nodes = (np.asarray(node_tuples, dtype=np.float64) / order * 2.0 - 1.0).T
    barycentric = _unit_to_barycentric(unit_nodes)
    equilateral = _barycentric_to_equilateral(barycentric)
    vertices = _EQUILATERAL_VERTICES[3]
    shift = np.zeros_like(equilateral)
    tolerance = 1.0e-8
    for first, second, third, fourth, vertex_step in (
        (0, 1, 2, 3, -1),
        (1, 2, 3, 0, -1),
        (2, 3, 0, 1, -1),
        (3, 0, 1, 2, -1),
    ):
        vertex_two, vertex_three, vertex_four = (
            (first + vertex_step * index) % 4 for index in range(1, 4)
        )
        tangent_one = vertices[vertex_three] - vertices[vertex_four]
        tangent_one /= np.linalg.norm(tangent_one)
        tangent_two = vertices[vertex_two] - vertices[vertex_three]
        tangent_two -= np.dot(tangent_one, tangent_two) * tangent_one
        tangent_two /= np.linalg.norm(tangent_two)

        face_barycentric = barycentric[[second, third, fourth]]
        warp_one, warp_two = _equilateral_shift_2d(order, face_barycentric, alpha)
        first_barycentric = barycentric[first]
        second_barycentric, third_barycentric, fourth_barycentric = face_barycentric
        blend = second_barycentric * third_barycentric * fourth_barycentric
        denominator = (
            (second_barycentric + 0.5 * first_barycentric)
            * (third_barycentric + 0.5 * first_barycentric)
            * (fourth_barycentric + 0.5 * first_barycentric)
        )
        valid_denominator = denominator > tolerance
        blend[valid_denominator] = (
            (1.0 + (alpha * first_barycentric[valid_denominator]) ** 2)
            * blend[valid_denominator]
            / denominator[valid_denominator]
        )
        shift += (blend * warp_one)[None, :] * tangent_one[:, None]
        shift += (blend * warp_two)[None, :] * tangent_two[:, None]

        on_face = (first_barycentric < tolerance) & (
            (second_barycentric > tolerance)
            | (third_barycentric > tolerance)
            | (fourth_barycentric > tolerance)
        )
        shift[:, on_face] = (
            warp_one[on_face][None, :] * tangent_one[:, None]
            + warp_two[on_face][None, :] * tangent_two[:, None]
        )
    return _equilateral_to_unit(equilateral + shift)


def _warp_and_blend_nodes(
    dimension: int, order: int, node_tuples: tuple[tuple[int, ...], ...], /
) -> np.ndarray:
    if dimension == 2:
        return _warp_and_blend_nodes_2d(order, node_tuples)
    if dimension == 3:
        return _warp_and_blend_nodes_3d(order, node_tuples)
    raise ValueError("Warp-and-blend nodes require dimension two or three.")


class SimplexNodalFamily(StrictModule, NonTrainableState):
    """Warp-and-blend simplex nodes with an orthonormal modal tabulation."""

    cell_kind: str = eqx.field(static=True)
    order: int = eqx.field(static=True)
    nodes: Array
    multiindices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    coefficients: Array
    condition_number: float = eqx.field(static=True)
    family_id: str = eqx.field(static=True)

    def __init__(self, cell_kind: str, order: int, /):
        cell = str(cell_kind)
        p = int(order)
        dimension = {"triangle": 2, "tetrahedron": 3}.get(cell)
        if dimension is None or p < 0:
            raise ValueError(
                "Simplex nodal families require triangle/tetrahedron and p>=0."
            )
        node_tuples = _simplex_node_tuples(p, dimension)
        space_dimension = comb(p + dimension, dimension)
        if p == 0:
            nodes = np.full((1, dimension), 1.0 / (dimension + 1.0))
        else:
            unit_nodes = _warp_and_blend_nodes(dimension, p, node_tuples)
            nodes = 0.5 * (unit_nodes.T + 1.0)
        barycentric_nodes = np.concatenate(
            (1.0 - np.sum(nodes, axis=1, keepdims=True), nodes),
            axis=1,
        )
        exponents = np.asarray(
            tuple((p - sum(index),) + tuple(index) for index in node_tuples),
            dtype=np.int64,
        )
        multinomial = np.asarray(
            tuple(
                factorial(p) / np.prod(tuple(factorial(int(value)) for value in exponent))
                for exponent in exponents
            ),
            dtype=np.float64,
        )
        modal = multinomial[None, :] * np.prod(
            barycentric_nodes[:, None, :] ** exponents[None, :, :],
            axis=-1,
        )
        coefficients = np.linalg.solve(modal, np.eye(space_dimension))
        condition = float(np.linalg.cond(modal))
        barycentric = tuple((p - sum(index),) + tuple(index) for index in node_tuples)
        self.cell_kind = cell
        self.order = p
        self.nodes = jnp.asarray(nodes)
        self.multiindices = barycentric
        self.coefficients = jnp.asarray(coefficients)
        self.condition_number = condition
        self.family_id = canonical_fingerprint(
            {
                "kind": "simplex-warp-blend-nodal-family",
                "cell": cell,
                "order": p,
                "node_source": "phydrax:warburton-warp-blend",
                "nodes": array_tree_fingerprint(nodes),
                "condition_number": condition,
            }
        )

    def tabulate(self, points: ArrayLike, /) -> tuple[Array, Array]:
        points_ = jnp.asarray(points)
        dimension = {"triangle": 2, "tetrahedron": 3}[self.cell_kind]
        if points_.ndim != 2 or points_.shape[-1] != dimension:
            raise ValueError("Simplex tabulation points have incompatible shape.")
        if not jnp.issubdtype(points_.dtype, jnp.inexact):
            points_ = points_.astype("float64")
        barycentric = jnp.concatenate(
            (1.0 - jnp.sum(points_, axis=-1, keepdims=True), points_),
            axis=-1,
        )
        exponents = jnp.asarray(self.multiindices, dtype=jnp.int32)
        multinomial = jnp.asarray(
            tuple(
                factorial(self.order)
                / np.prod(tuple(factorial(int(value)) for value in exponent))
                for exponent in self.multiindices
            ),
            dtype=points_.dtype,
        )
        modal = multinomial[None, :] * jnp.prod(
            barycentric[:, None, :] ** exponents[None, :, :],
            axis=-1,
        )
        coefficients = jnp.asarray(self.coefficients, dtype=points_.dtype)
        values = modal @ coefficients
        gradient_components = []
        for axis in range(dimension):
            zero_exponents = exponents.at[:, 0].add(-1)
            axis_exponents = exponents.at[:, axis + 1].add(-1)
            zero_term = (
                multinomial[None, :]
                * exponents[None, :, 0]
                * jnp.prod(
                    barycentric[:, None, :] ** jnp.maximum(zero_exponents, 0)[None, :, :],
                    axis=-1,
                )
            )
            axis_term = (
                multinomial[None, :]
                * exponents[None, :, axis + 1]
                * jnp.prod(
                    barycentric[:, None, :] ** jnp.maximum(axis_exponents, 0)[None, :, :],
                    axis=-1,
                )
            )
            modal_gradient = jnp.where(
                exponents[None, :, axis + 1] > 0,
                axis_term,
                0.0,
            ) - jnp.where(exponents[None, :, 0] > 0, zero_term, 0.0)
            gradient_components.append(modal_gradient @ coefficients)
        gradients = jnp.stack(tuple(gradient_components), axis=-1)
        return values, gradients

    def finite_element(self) -> FiniteElementSpec:
        from .._reference_cell import reference_cell_topology

        topology = reference_cell_topology(self.cell_kind)
        entity_dofs = [[[] for _ in entities] for entities in topology.entities]
        entity_sets = [
            tuple(frozenset(entity) for entity in entities)
            for entities in topology.entities
        ]
        if self.order == 0:
            entity_dofs[-1][0].append(0)
            return FiniteElementSpec(
                "SimplexLagrange",
                self.cell_kind,
                self.order,
                self.nodes,
                tuple(
                    tuple(tuple(values) for values in dimension)
                    for dimension in entity_dofs
                ),
                conformity="H1",
                representation="point_value",
                tabulator=self.tabulate,
                tabulator_id=self.family_id,
            )
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
            representation="point_value",
            tabulator=self.tabulate,
            tabulator_id=self.family_id,
        )


__all__ = [
    "NodeSet",
    "TensorOrder",
    "QuadratureChunkPolicy",
    "ReferenceNodalFamily",
    "SumFactorizationPlan",
    "SimplexNodalFamily",
    "TensorProductTabulation",
    "lagrange_1d_tabulation",
    "local_diagonal",
]
