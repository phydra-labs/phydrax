#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Native tetrahedral Raviart--Thomas and BDM reference families."""

from __future__ import annotations

from functools import lru_cache

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...ein import contract
from ._reference import FiniteElementSpec


_VERTICES = np.asarray(
    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
)
_FACE_VERTICES = ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3))
_OPPOSITE_VERTICES = np.asarray((3, 2, 0, 1), dtype=np.int32)
_FACE_NORMALS = np.asarray(
    (
        (0.0, 0.0, -1.0),
        (0.0, -1.0, 0.0),
        (1.0, 1.0, 1.0),
        (-1.0, 0.0, 0.0),
    )
)
_FACE_NORMALS /= np.linalg.norm(_FACE_NORMALS, axis=1)[:, None]


def _entity_dofs(
    face_width: int, cell_width: int = 0
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    face_count = 4 * face_width
    return (
        ((), (), (), ()),
        ((), (), (), (), (), ()),
        tuple(
            tuple(range(face * face_width, (face + 1) * face_width)) for face in range(4)
        ),
        (tuple(range(face_count, face_count + cell_width)),),
    )


def _rt0_tabulate(points: ArrayLike, /) -> tuple[Array, Array]:
    locations = jnp.asarray(points)
    vertices = jnp.asarray(_VERTICES[_OPPOSITE_VERTICES], dtype=locations.dtype)
    values = 2.0 * (locations[:, None, :] - vertices[None, :, :])
    identity = 2.0 * jnp.eye(3, dtype=locations.dtype)
    gradients = jnp.broadcast_to(identity, (len(locations), 4, 3, 3))
    return values, gradients


@lru_cache(maxsize=1)
def _bdm1_coefficients() -> np.ndarray:
    # psi_(vertex,component) = lambda_vertex e_component. Face moments use
    # lambda at each oriented reference-face vertex.
    matrix = np.zeros((12, 12), dtype=float)
    row = 0
    for face, vertices in enumerate(_FACE_VERTICES):
        normal = _FACE_NORMALS[face]
        first = _VERTICES[vertices[1]] - _VERTICES[vertices[0]]
        second = _VERTICES[vertices[2]] - _VERTICES[vertices[0]]
        area = 0.5 * np.linalg.norm(np.cross(first, second))
        for moment_vertex in vertices:
            for basis_vertex in range(4):
                integral = 0.0
                if basis_vertex in vertices:
                    integral = area * (
                        1.0 / 6.0 if basis_vertex == moment_vertex else 1.0 / 12.0
                    )
                for component in range(3):
                    matrix[row, 3 * basis_vertex + component] = (
                        normal[component] * integral
                    )
            row += 1
    return np.linalg.inv(matrix)


def _barycentric(points: Array, /) -> Array:
    return jnp.concatenate(
        ((1.0 - jnp.sum(points, axis=1, keepdims=True)), points), axis=1
    )


def _bdm1_tabulate(points: ArrayLike, /) -> tuple[Array, Array]:
    locations = jnp.asarray(points)
    barycentric = _barycentric(locations)
    coefficients = jnp.asarray(_bdm1_coefficients(), dtype=locations.dtype)
    polynomial = jnp.zeros((len(locations), 12, 3), dtype=locations.dtype)
    for vertex in range(4):
        for component in range(3):
            polynomial = polynomial.at[:, 3 * vertex + component, component].set(
                barycentric[:, vertex]
            )
    values = contract("pbc,bd->pdc", polynomial, coefficients)
    barycentric_gradients = jnp.asarray(
        (
            (-1.0, -1.0, -1.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=locations.dtype,
    )
    polynomial_gradients = jnp.zeros((12, 3, 3), dtype=locations.dtype)
    for vertex in range(4):
        for component in range(3):
            polynomial_gradients = polynomial_gradients.at[
                3 * vertex + component, component, :
            ].set(barycentric_gradients[vertex])
    gradients = contract("bcg,bd->dcg", polynomial_gradients, coefficients)
    gradients = jnp.broadcast_to(gradients, (len(locations),) + gradients.shape)
    return values, gradients


def _gauss_unit(order: int) -> tuple[np.ndarray, np.ndarray]:
    points, weights = np.polynomial.legendre.leggauss(order)
    return 0.5 * (points + 1.0), 0.5 * weights


def _p2_monomials_numpy(points: np.ndarray) -> np.ndarray:
    x, y, z = points.T
    return np.column_stack(
        (np.ones_like(x), x, y, z, x * x, x * y, x * z, y * y, y * z, z * z)
    )


def _face_quadrature(
    face: tuple[int, int, int], /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes, weights = _gauss_unit(4)
    points = []
    physical_weights = []
    barycentrics = []
    vertices = _VERTICES[np.asarray(face)]
    surface_jacobian = np.linalg.norm(
        np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    )
    for first, first_weight in zip(nodes, weights, strict=True):
        for second, second_weight in zip(nodes, weights, strict=True):
            barycentric = np.asarray(
                (1.0 - first - (1.0 - first) * second, first, (1.0 - first) * second)
            )
            points.append(barycentric @ vertices)
            barycentrics.append(barycentric)
            physical_weights.append(
                first_weight * second_weight * (1.0 - first) * surface_jacobian
            )
    return np.asarray(points), np.asarray(physical_weights), np.asarray(barycentrics)


def _tetrahedron_quadrature() -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = _gauss_unit(5)
    points = []
    physical_weights = []
    for first, first_weight in zip(nodes, weights, strict=True):
        for second, second_weight in zip(nodes, weights, strict=True):
            for third, third_weight in zip(nodes, weights, strict=True):
                points.append(
                    (
                        first,
                        (1.0 - first) * second,
                        (1.0 - first) * (1.0 - second) * third,
                    )
                )
                physical_weights.append(
                    first_weight
                    * second_weight
                    * third_weight
                    * (1.0 - first) ** 2
                    * (1.0 - second)
                )
    return np.asarray(points), np.asarray(physical_weights)


def _bernstein_p2(barycentric: np.ndarray) -> np.ndarray:
    first, second, third = barycentric.T
    return np.column_stack(
        (
            first * first,
            second * second,
            third * third,
            2.0 * first * second,
            2.0 * second * third,
            2.0 * third * first,
        )
    )


@lru_cache(maxsize=1)
def _bdm2_coefficients() -> np.ndarray:
    # [P2]^3 has 30 coefficients. Twenty-four face moments use the degree-two
    # Bernstein basis, whose permutation follows face vertices/edges. Six
    # interior moments use the lowest Nedelec space: constants and rotations.
    matrix = np.zeros((30, 30), dtype=float)
    for face_index, face in enumerate(_FACE_VERTICES):
        points, weights, barycentric = _face_quadrature(face)
        monomials = _p2_monomials_numpy(points)
        moments = _bernstein_p2(barycentric)
        normal = _FACE_NORMALS[face_index]
        for moment in range(6):
            row = 6 * face_index + moment
            for scalar in range(10):
                integral = np.sum(weights * moments[:, moment] * monomials[:, scalar])
                for component in range(3):
                    matrix[row, 3 * scalar + component] = normal[component] * integral
    points, weights = _tetrahedron_quadrature()
    monomials = _p2_monomials_numpy(points)
    x, y, z = points.T
    tests = np.stack(
        (
            np.column_stack((np.ones_like(x), np.zeros_like(x), np.zeros_like(x))),
            np.column_stack((np.zeros_like(x), np.ones_like(x), np.zeros_like(x))),
            np.column_stack((np.zeros_like(x), np.zeros_like(x), np.ones_like(x))),
            np.column_stack((np.zeros_like(x), -z, y)),
            np.column_stack((z, np.zeros_like(x), -x)),
            np.column_stack((-y, x, np.zeros_like(x))),
        )
    )
    for moment in range(6):
        row = 24 + moment
        for scalar in range(10):
            for component in range(3):
                matrix[row, 3 * scalar + component] = np.sum(
                    weights * tests[moment, :, component] * monomials[:, scalar]
                )
    if np.linalg.matrix_rank(matrix) != 30:
        raise RuntimeError("Native tetrahedral BDM2 moments are not unisolvent.")
    return np.linalg.inv(matrix)


def _p2_monomials(points: Array, /) -> tuple[Array, Array]:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    values = jnp.stack(
        (
            jnp.ones_like(x),
            x,
            y,
            z,
            x * x,
            x * y,
            x * z,
            y * y,
            y * z,
            z * z,
        ),
        axis=1,
    )
    zero = jnp.zeros_like(x)
    one = jnp.ones_like(x)
    gradients = jnp.stack(
        (
            jnp.stack((zero, zero, zero), axis=1),
            jnp.stack((one, zero, zero), axis=1),
            jnp.stack((zero, one, zero), axis=1),
            jnp.stack((zero, zero, one), axis=1),
            jnp.stack((2.0 * x, zero, zero), axis=1),
            jnp.stack((y, x, zero), axis=1),
            jnp.stack((z, zero, x), axis=1),
            jnp.stack((zero, 2.0 * y, zero), axis=1),
            jnp.stack((zero, z, y), axis=1),
            jnp.stack((zero, zero, 2.0 * z), axis=1),
        ),
        axis=1,
    )
    return values, gradients


def _bdm2_tabulate(points: ArrayLike, /) -> tuple[Array, Array]:
    locations = jnp.asarray(points)
    monomials, monomial_gradients = _p2_monomials(locations)
    coefficients = jnp.asarray(_bdm2_coefficients(), dtype=locations.dtype)
    polynomial = jnp.zeros((len(locations), 30, 3), dtype=locations.dtype)
    polynomial_gradients = jnp.zeros((len(locations), 30, 3, 3), dtype=locations.dtype)
    for scalar in range(10):
        for component in range(3):
            basis = 3 * scalar + component
            polynomial = polynomial.at[:, basis, component].set(monomials[:, scalar])
            polynomial_gradients = polynomial_gradients.at[:, basis, component, :].set(
                monomial_gradients[:, scalar]
            )
    values = contract("pbc,bd->pdc", polynomial, coefficients)
    gradients = contract("pbcg,bd->pdcg", polynomial_gradients, coefficients)
    return values, gradients


def tetrahedral_rt_element(degree: int = 0, /) -> FiniteElementSpec:
    if degree != 0:
        raise ValueError(
            "The native tetrahedral RT family currently supports degree zero."
        )
    face_centers = np.asarray(
        [np.mean(_VERTICES[list(face)], axis=0) for face in _FACE_VERTICES]
    )
    return FiniteElementSpec(
        "RaviartThomas",
        "tetrahedron",
        0,
        face_centers,
        _entity_dofs(1),
        conformity="Hdiv",
        representation="flux_moment",
        mapping="contravariant_piola",
        value_shape=(3,),
        tabulator=_rt0_tabulate,
        tabulator_id="tetrahedral-rt0-analytic",
    )


def tetrahedral_bdm_element(degree: int = 1, /) -> FiniteElementSpec:
    if degree == 1:
        nodes = np.concatenate([_VERTICES[list(face)] for face in _FACE_VERTICES])
        return FiniteElementSpec(
            "BrezziDouglasMarini",
            "tetrahedron",
            1,
            nodes,
            _entity_dofs(3),
            conformity="Hdiv",
            representation="flux_moment",
            mapping="contravariant_piola",
            value_shape=(3,),
            tabulator=_bdm1_tabulate,
            tabulator_id="tetrahedral-bdm1-moment-dual",
        )
    if degree == 2:
        nodes = []
        for face in _FACE_VERTICES:
            vertices = _VERTICES[np.asarray(face)]
            nodes.extend(
                (
                    vertices[0],
                    vertices[1],
                    vertices[2],
                    0.5 * (vertices[0] + vertices[1]),
                    0.5 * (vertices[1] + vertices[2]),
                    0.5 * (vertices[2] + vertices[0]),
                )
            )
        nodes.extend(
            (
                (0.25, 0.25, 0.25),
                (0.2, 0.2, 0.2),
                (0.4, 0.2, 0.2),
                (0.2, 0.4, 0.2),
                (0.2, 0.2, 0.4),
                (0.3, 0.3, 0.2),
            )
        )
        return FiniteElementSpec(
            "BrezziDouglasMarini",
            "tetrahedron",
            2,
            np.asarray(nodes),
            _entity_dofs(6, 6),
            conformity="Hdiv",
            representation="flux_moment",
            mapping="contravariant_piola",
            value_shape=(3,),
            tabulator=_bdm2_tabulate,
            tabulator_id="tetrahedral-bdm2-moment-dual",
        )
    raise ValueError("The native tetrahedral BDM family supports degrees one and two.")


__all__ = ["tetrahedral_bdm_element", "tetrahedral_rt_element"]
