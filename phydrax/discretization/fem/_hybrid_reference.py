#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Hybrid source reference families and nodal trace preparation."""

from __future__ import annotations

from fractions import Fraction
from typing import final, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from scipy.special import eval_jacobi

import phydrax.ein as ein
from phydrax.discretization.fem._reference_types import (
    AbstractFiniteElementTabulationSource,
    FiniteElementSpec,
    FiniteElementTabulator,
)

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._polynomial._orthogonal import legendre_rule_data
from ...exterior import FormType, FormValueSpec
from ...geometry._rational import rational_vector
from .._reference_cell import reference_cell_topology
from . import _jacobi as jacobi_owner
from ._high_order import lagrange_1d_tabulation, simplex_node_tuples, SimplexNodalFamily


# Height-major rational lattice order and vertex support identify the constructor's trace rows.
def _pyramid_reference_lattice(
    p: int, faces: tuple[tuple[int, ...], ...]
) -> tuple[list[tuple[Fraction, ...]], dict[frozenset[int], list[int]]]:
    labels: list[tuple[Fraction, ...]] = []
    face_rows: dict[frozenset[int], list[int]] = {
        frozenset(face): [] for face in faces if len(face) == 3
    }
    for height in range(p + 1):
        cross = p - height
        for first in range(cross + 1):
            for second in range(cross + 1):
                row = len(labels)
                labels.append(
                    (
                        Fraction(2 * first + height, 2 * p),
                        Fraction(2 * second + height, 2 * p),
                        Fraction(height, p),
                    )
                )
                if 0 < height < p:
                    support = frozenset(
                        vertex
                        for vertex, active in enumerate(
                            (
                                first < cross and second < cross,
                                first > 0 and second < cross,
                                first > 0 and second > 0,
                                first < cross and second > 0,
                                True,
                            )
                        )
                        if active
                    )
                    if support in face_rows:
                        face_rows[support].append(row)
    return labels, face_rows


def _shifted_jacobi(
    degree: int, alpha: float, beta: float, values: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    polynomial = eval_jacobi(degree, alpha, beta, 2.0 * values - 1.0)
    derivative = (
        np.zeros_like(values)
        if degree == 0
        else (degree + alpha + beta + 1.0)
        * eval_jacobi(degree - 1, alpha + 1.0, beta + 1.0, 2.0 * values - 1.0)
    )
    return np.asarray(polynomial), np.asarray(derivative)


def pyramid_modal_tabulation(
    points: np.ndarray,
    indices: tuple[tuple[int, int, int], ...],
    /,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(points, dtype=np.float64)
    height = values[:, 2]
    scale = 1.0 - height
    safe = np.where(scale > 1.0e-12, scale, 1.0)
    first = np.where(scale > 1.0e-12, (values[:, 0] - 0.5 * height) / safe, 0.5)
    second = np.where(scale > 1.0e-12, (values[:, 1] - 0.5 * height) / safe, 0.5)
    modal_values = []
    modal_gradients = []
    for first_degree, second_degree, height_degree in indices:
        maximum = max(first_degree, second_degree)
        first_value, first_derivative = _shifted_jacobi(first_degree, 0.0, 0.0, first)
        second_value, second_derivative = _shifted_jacobi(second_degree, 0.0, 0.0, second)
        height_value, height_derivative = _shifted_jacobi(
            height_degree, 2.0 * maximum + 2.0, 0.0, height
        )
        scale_power = scale**maximum
        mode = first_value * second_value * scale_power * height_value
        x_gradient = (
            first_derivative
            * second_value
            * np.where(maximum > 0, scale ** max(maximum - 1, 0), 1.0)
            * height_value
        )
        y_gradient = (
            first_value
            * second_derivative
            * np.where(maximum > 0, scale ** max(maximum - 1, 0), 1.0)
            * height_value
        )
        first_height_derivative = np.where(scale > 1.0e-12, (first - 0.5) / safe, 0.0)
        second_height_derivative = np.where(scale > 1.0e-12, (second - 0.5) / safe, 0.0)
        scale_derivative = (
            np.zeros_like(scale) if maximum == 0 else -maximum * scale ** (maximum - 1)
        )
        z_gradient = (
            first_derivative
            * first_height_derivative
            * second_value
            * scale_power
            * height_value
            + first_value
            * second_derivative
            * second_height_derivative
            * scale_power
            * height_value
            + first_value * second_value * scale_derivative * height_value
            + first_value * second_value * scale_power * height_derivative
        )
        modal_values.append(mode)
        modal_gradients.append(np.stack((x_gradient, y_gradient, z_gradient), axis=-1))
    return np.stack(tuple(modal_values), axis=-1), np.stack(
        tuple(modal_gradients), axis=1
    )


def _pyramid_modal_tabulation_jax(
    points: Array,
    indices: tuple[tuple[int, int, int], ...],
) -> tuple[Array, Array]:
    """Execute the same prepared Bergot modes without host conversion."""

    height = points[:, 2]
    collapse = 1.0 - height
    interior = collapse != 0.0
    denominator = jnp.where(interior, collapse, 1.0)
    first = jnp.where(interior, (points[:, 0] - 0.5 * height) / denominator, 0.5)
    second = jnp.where(interior, (points[:, 1] - 0.5 * height) / denominator, 0.5)
    first_height = jnp.where(interior, (first - 0.5) / denominator, 0.0)
    second_height = jnp.where(interior, (second - 0.5) / denominator, 0.0)
    values, gradients = [], []
    for i, j, k in indices:
        maximum = max(i, j)
        a, da = (
            jacobi_owner.jacobi_jax(i, 0, 0, first),
            jacobi_owner.jacobi_derivative_jax(i, 0, 0, first),
        )
        b, db = (
            jacobi_owner.jacobi_jax(j, 0, 0, second),
            jacobi_owner.jacobi_derivative_jax(j, 0, 0, second),
        )
        c = jacobi_owner.jacobi_jax(k, 2 * maximum + 2, 0, height)
        dc = jacobi_owner.jacobi_derivative_jax(k, 2 * maximum + 2, 0, height)
        power = collapse**maximum
        horizontal = collapse ** max(maximum - 1, 0)
        derivative = jnp.zeros_like(collapse) if maximum == 0 else -maximum * horizontal
        values.append(a * b * power * c)
        gradients.append(
            jnp.stack(
                (
                    da * b * horizontal * c,
                    a * db * horizontal * c,
                    (da * first_height * b + a * db * second_height) * power * c
                    + a * b * derivative * c
                    + a * b * power * dc,
                ),
                axis=-1,
            )
        )
    return jnp.stack(values, axis=-1), jnp.stack(gradients, axis=1)


def _conforming_pyramid_boundary_nodes(nodes: np.ndarray, degree: int, /) -> np.ndarray:
    """Use the same triangular trace nodes as tetrahedra and prisms.

    Base quadrilateral and edge Lobatto nodes, and interior collapsed nodes,
    remain unchanged. Only triangular face-interior dual points are relocated;
    the rational modal space is preserved by the nodal solve.
    """
    if degree < 3:
        return nodes
    triangle = np.asarray(SimplexNodalFamily("triangle", degree).nodes, dtype=np.float64)
    barycentric = np.column_stack((1.0 - np.sum(triangle, axis=1), triangle))
    interior = barycentric[np.all(barycentric > 2.0e-10, axis=1)]
    height = nodes[:, 2]
    collapse = 1.0 - height
    regular = collapse > 2.0e-10
    first = np.full(nodes.shape[0], 0.5, dtype=np.float64)
    second = first.copy()
    first[regular] = (nodes[regular, 0] - 0.5 * height[regular]) / collapse[regular]
    second[regular] = (nodes[regular, 1] - 0.5 * height[regular]) / collapse[regular]
    weights = np.column_stack(
        (
            collapse * (1.0 - first) * (1.0 - second),
            collapse * first * (1.0 - second),
            collapse * first * second,
            collapse * (1.0 - first) * second,
            height,
        )
    )
    support = weights > 2.0e-10
    topology = reference_cell_topology("pyramid")
    vertices = np.asarray(topology.vertices, dtype=np.float64)
    result = nodes.copy()
    for face in topology.entities[2]:
        if len(face) != 3:
            continue
        rows = np.flatnonzero(
            np.all(support[:, list(face)], axis=1) & (np.sum(support, axis=1) == 3)
        )
        if rows.size != interior.shape[0]:
            raise ValueError(
                "HybridReferenceFamily.triangular_trace must match the simplex trace dimension; got incompatible trace rows"
            )
        result[rows] = interior @ vertices[list(face)]
    return result


@final
class HybridReferenceFamily(AbstractFiniteElementTabulationSource):
    """Anisotropic prism/rational pyramid with common simplex/tensor nodal traces."""

    cell_kind: Literal["prism", "pyramid"] = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    orders: tuple[int, int] = eqx.field(static=True)
    nodes: Array
    basis_permutation: tuple[int, ...] = eqx.field(static=True)
    modal_indices: tuple[tuple[int, int, int], ...] = eqx.field(static=True)
    coefficients: Array
    condition_number: float = eqx.field(static=True)
    triangle_family: SimplexNodalFamily | None
    axial_nodes: Array
    family_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_kind: Literal["prism", "pyramid"],
        degree: int | tuple[int, int],
        /,
    ) -> None:
        kind = str(cell_kind)
        match degree:
            case tuple():
                if kind != "prism":
                    raise TypeError(
                        "HybridReferenceFamily.degree must be scalar for a pyramid; got a degree tuple"
                    )
                if len(degree) != 2:
                    raise ValueError(
                        "HybridReferenceFamily.degree must contain (triangle, axial) orders for a prism; got an incompatible tuple"
                    )
                triangle_degree, axial_degree = (int(value) for value in degree)
            case _:
                triangle_degree = axial_degree = int(degree)
        p = triangle_degree
        q = axial_degree
        if kind not in ("prism", "pyramid") or min(p, q) < 1:
            raise ValueError(
                "HybridReferenceFamily.cell_kind_degree must select prism or pyramid with degree at least one; got an unsupported cell or degree"
            )
        if kind == "pyramid" and p != q:
            raise ValueError(
                "HybridReferenceFamily.degree must be isotropic for a pyramid; got unequal orders"
            )
        triangle = None
        z_nodes = np.zeros(0, dtype=np.float64)

        if kind == "prism":
            triangle = SimplexNodalFamily("triangle", p)
            rule = legendre_rule_data(q + 1, "lobatto")
            z_nodes = 0.5 * (np.asarray(rule.nodes) + 1.0)
            generated_nodes = np.asarray(
                [
                    (float(point[0]), float(point[1]), float(z))
                    for point in np.asarray(triangle.nodes)
                    for z in z_nodes
                ]
            )
            if (p, q) == (1, 1):
                nodes = np.asarray(reference_cell_topology("prism").vertices)
                permutation = tuple(
                    int(
                        np.flatnonzero(
                            np.max(np.abs(generated_nodes - point), axis=1) <= 2.0e-12
                        )[0]
                    )
                    for point in nodes
                )
            else:
                nodes = generated_nodes
                permutation = tuple(range(nodes.shape[0]))
            modal_indices: tuple[tuple[int, int, int], ...] = ()
            coefficients = np.zeros((0, 0))
            condition_number = 1.0
        else:
            height_rule = legendre_rule_data(p + 1, "lobatto")
            height_nodes = 0.5 * (np.asarray(height_rule.nodes) + 1.0)
            pyramid_nodes = []
            for layer, height in enumerate(height_nodes):
                cross_degree = p - layer
                if cross_degree == 0:
                    cross_nodes = np.asarray((0.5,))
                else:
                    cross_rule = legendre_rule_data(cross_degree + 1, "lobatto")
                    cross_nodes = 0.5 * (np.asarray(cross_rule.nodes) + 1.0)
                scale = 1.0 - height
                pyramid_nodes.extend(
                    (
                        scale * first + 0.5 * height,
                        scale * second + 0.5 * height,
                        height,
                    )
                    for first in cross_nodes
                    for second in cross_nodes
                )
            nodes = (
                np.asarray(reference_cell_topology("pyramid").vertices)
                if p == 1
                else np.asarray(pyramid_nodes)
            )
            nodes = _conforming_pyramid_boundary_nodes(nodes, p)
            permutation = tuple(range(nodes.shape[0]))
            modal_indices = tuple(
                (first, second, height)
                for first in range(p + 1)
                for second in range(p + 1)
                for height in range(p - max(first, second) + 1)
            )
            modal, _modal_gradients = pyramid_modal_tabulation(nodes, modal_indices)
            if modal.shape[0] != modal.shape[1]:
                raise ValueError(
                    "HybridReferenceFamily.nodes must match the modal dimension; got incompatible node count"
                )
            coefficients = np.linalg.solve(modal, np.eye(modal.shape[0]))
            condition_number = float(np.linalg.cond(modal))

        self.cell_kind = kind
        self.degree = max(p, q)
        self.orders = (p, q)
        self.nodes = jnp.asarray(nodes)
        self.basis_permutation = permutation
        self.modal_indices = modal_indices
        self.coefficients = jnp.asarray(coefficients)
        self.condition_number = condition_number
        self.triangle_family = triangle
        self.axial_nodes = jnp.asarray(z_nodes)
        self.family_id = canonical_fingerprint(
            {
                "kind": "hybrid-reference-family",
                "cell_kind": kind,
                "degree": max(p, q),
                "orders": (p, q),
                "basis": (
                    "simplex-times-legendre"
                    if kind == "prism"
                    else "bergot-cohen-durufle-rational-pyramid"
                ),
                "nodes": array_tree_fingerprint(nodes),
                "condition_number": condition_number,
            }
        )

    # Constructor trace relocation and the final basis permutation define nodal-label identity.
    def nodal_reference_labels(self) -> tuple[tuple[Fraction, ...], ...]:
        """Return logical dual labels in basis and node order.

        Lobatto and warp-and-blend coordinates remain dynamic numerical leaves;
        labels identify integer reference nodes, not rounded points.
        """
        p, q = self.orders
        if self.cell_kind == "prism":
            triangle = self.triangle_family
            if triangle is None:
                raise RuntimeError(
                    "HybridReferenceFamily.triangle_family must be prepared for a prism; got None"
                )
            labels = tuple(
                (Fraction(alpha[1], p), Fraction(alpha[2], p), Fraction(axial, q))
                for alpha in triangle.multiindices
                for axial in range(q + 1)
            )
            return tuple(labels[index] for index in self.basis_permutation)
        topology = reference_cell_topology("pyramid")
        vertices = tuple(rational_vector(point) for point in topology.vertices)
        if p == 1:
            return tuple(vertices[index] for index in self.basis_permutation)
        labels, face_rows = _pyramid_reference_lattice(p, topology.entities[2])
        if p >= 3:
            interior = tuple(
                (p - sum(index), *index)
                for index in simplex_node_tuples(p, 2)
                if all(value > 0 for value in (p - sum(index), *index))
            )
            # The constructor relocates each triangular interior in precisely
            # this simplex-owner order, without changing the rational space.
            for face in topology.entities[2]:
                if len(face) != 3:
                    continue
                rows = face_rows[frozenset(face)]
                if len(rows) != len(interior):
                    raise RuntimeError(
                        "HybridReferenceFamily.labels must cover the simplex trace; "
                        f"got {len(rows)} rows, expected {len(interior)}"
                    )
                for row, alpha in zip(rows, interior, strict=True):
                    labels[row] = tuple(
                        sum(
                            (
                                Fraction(weight, p) * vertices[vertex][axis]
                                for weight, vertex in zip(alpha, face, strict=True)
                            ),
                            Fraction(0),
                        )
                        for axis in range(3)
                    )
        return tuple(labels[index] for index in self.basis_permutation)

    def tabulate(self, points: ArrayLike, /) -> tuple[Array, Array]:
        points_ = jnp.asarray(points)
        if points_.ndim != 2 or points_.shape[-1] != 3:
            raise ValueError(
                "HybridReferenceFamily.points must have shape (n, 3); got incompatible shape"
            )
        if self.cell_kind == "pyramid":
            modal, modal_gradients = _pyramid_modal_tabulation_jax(
                points_, self.modal_indices
            )
            values = modal @ self.coefficients
            gradients = ein.contract("qim,ij->qjm", modal_gradients, self.coefficients)
            return values, gradients

        triangle = self.triangle_family
        if triangle is None:
            raise RuntimeError(
                "HybridReferenceFamily.triangle_family must be prepared for a prism; got None"
            )
        triangle_values, triangle_gradients = triangle.tabulate(points_[..., :2])
        z_values, z_gradients = lagrange_1d_tabulation(self.axial_nodes, points_[..., 2])
        values = ein.contract(
            "qi,qj->qij", triangle_values, z_values, backend="jax"
        ).reshape((points_.shape[0], -1))
        horizontal = ein.contract(
            "qid,qj->qijd", triangle_gradients, z_values, backend="jax"
        )
        vertical = ein.contract(
            "qi,qj->qij", triangle_values, z_gradients, backend="jax"
        )[..., None]
        gradients = jnp.concatenate((horizontal, vertical), axis=-1).reshape(
            (points_.shape[0], -1, 3)
        )
        permutation = jnp.asarray(self.basis_permutation, dtype=jnp.int32)
        return values[:, permutation], gradients[:, permutation]

    def _entity_support(self, point: np.ndarray, /) -> frozenset[int]:
        tolerance = 2.0e-10
        if self.cell_kind == "prism":
            height = float(point[2])
            triangle = np.asarray(point[:2])
            barycentric = np.asarray(
                (1.0 - triangle[0] - triangle[1], triangle[0], triangle[1])
            )
            triangle_support = tuple(
                index for index, value in enumerate(barycentric) if value > tolerance
            )
            if height <= tolerance:
                return frozenset(triangle_support)
            if height >= 1.0 - tolerance:
                return frozenset(index + 3 for index in triangle_support)
            return frozenset(
                value for index in triangle_support for value in (index, index + 3)
            )

        height = float(point[2])
        if height >= 1.0 - tolerance:
            return frozenset((4,))
        scale = 1.0 - height
        first = (float(point[0]) - 0.5 * height) / scale
        second = (float(point[1]) - 0.5 * height) / scale
        base_weights = np.asarray(
            (
                (1.0 - first) * (1.0 - second),
                first * (1.0 - second),
                first * second,
                (1.0 - first) * second,
            )
        )
        base_support = frozenset(
            index for index, value in enumerate(base_weights) if value > tolerance
        )
        return base_support if height <= tolerance else frozenset((*base_support, 4))

    def finite_element(self, /, *, conformity: str = "H1") -> FiniteElementSpec:
        if conformity not in ("H1", "L2"):
            raise ValueError(
                "HybridReferenceFamily.conformity must select H1 or L2; got an unsupported conformity"
            )
        topology = reference_cell_topology(self.cell_kind)
        entities = [[[] for _entity in dimension] for dimension in topology.entities]
        if conformity == "H1":
            entity_sets = [
                tuple(frozenset(entity) for entity in dimension)
                for dimension in topology.entities
            ]
            for dof, point in enumerate(np.asarray(self.nodes)):
                support = self._entity_support(point)
                matches = tuple(
                    (dimension, entities_.index(support))
                    for dimension, entities_ in enumerate(entity_sets)
                    if support in entities_
                )
                if len(matches) != 1:
                    raise ValueError(
                        "HybridReferenceFamily.nodal_support must identify one entity; got ambiguous entity membership"
                    )
                dimension, entity = matches[0]
                entities[dimension][entity].append(dof)
        else:
            entities[-1][0].extend(range(self.nodes.shape[0]))
        return FiniteElementSpec(
            "HybridLagrange",
            self.cell_kind,
            self.degree,
            self.nodes,
            tuple(tuple(tuple(values) for values in dimension) for dimension in entities),
            value_spec=FormValueSpec(
                FormType(topology.dimension, 0, twist="untwisted"), proxy="scalar"
            ),
            continuity="conforming" if conformity == "H1" else "discontinuous",
            representation="point_value",
            tabulator=FiniteElementTabulator(self),
            tabulator_id=self.family_id,
        )
