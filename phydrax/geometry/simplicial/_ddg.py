#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from ..._strict import StrictModule
from ._mesh import TriangleMesh


class DDGOperators(StrictModule):
    """Matrix-free cotangent and finite-element operators on one triangle mesh."""

    vertices: Array
    faces: Array
    edges: Array
    edge_weights: Array
    vertex_mass: Array
    face_area: Array
    face_normal: Array
    basis_gradients: Array
    boundary_vertices: Array

    def __init__(self, mesh: TriangleMesh, *, vertices: ArrayLike | None = None) -> None:
        """Assemble operators on ``mesh``.

        ``vertices`` refreshes the embedding for identical topology: the
        connectivity of ``mesh`` is reused and only numerical coefficients are
        recomputed, so moving-surface consumers never rebuild topology.
        """
        if not isinstance(mesh, TriangleMesh):
            raise TypeError("DDGOperators requires a TriangleMesh.")
        if vertices is None:
            vertices = mesh.vertices
        else:
            vertices = jnp.asarray(vertices, dtype=mesh.vertices.dtype)
            if vertices.shape != mesh.vertices.shape:
                raise ValueError(
                    "Refreshed vertices must match the mesh vertex array shape."
                )
        faces = mesh.faces
        triangles = vertices[faces]
        cross = jnp.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        doubled_area = jnp.linalg.norm(cross, axis=-1)
        face_area = 0.5 * doubled_area
        face_normal = cross / doubled_area[:, None]

        # Half-edge ``3 * face + k`` runs from corner ``k`` to corner ``k + 1``;
        # its cotangent is taken at the opposite corner ``k + 2``.
        squared = jnp.sum(
            (triangles - jnp.roll(triangles, -1, axis=1)) ** 2,
            axis=-1,
        )
        halfedge_cotangent = (
            jnp.roll(squared, -1, axis=1) + jnp.roll(squared, 1, axis=1) - squared
        ) / (2.0 * doubled_area[:, None])
        edge_lookup = mesh.topology.edges
        edge_weights = jnp.zeros((edge_lookup.shape[0],), dtype=vertices.dtype)
        edge_weights = edge_weights.at[mesh.topology.halfedge_edge].add(
            0.5 * halfedge_cotangent.reshape((-1,))
        )

        vertex_mass = jnp.zeros((vertices.shape[0],), dtype=vertices.dtype)
        vertex_mass = vertex_mass.at[faces.reshape((-1,))].add(
            jnp.repeat(face_area / 3.0, 3)
        )

        basis_gradients = (
            jnp.stack(
                (
                    jnp.cross(face_normal, triangles[:, 2] - triangles[:, 1]),
                    jnp.cross(face_normal, triangles[:, 0] - triangles[:, 2]),
                    jnp.cross(face_normal, triangles[:, 1] - triangles[:, 0]),
                ),
                axis=1,
            )
            / doubled_area[:, None, None]
        )
        boundary_vertices = jnp.zeros((vertices.shape[0],), dtype=jnp.bool_)
        if mesh.topology.boundary_halfedges.shape[0]:
            boundary_vertices = boundary_vertices.at[
                mesh.topology.halfedge_origin[mesh.topology.boundary_halfedges]
            ].set(True)

        self.vertices = vertices
        self.faces = faces
        self.edges = edge_lookup
        self.edge_weights = edge_weights
        self.vertex_mass = vertex_mass
        self.face_area = face_area
        self.face_normal = face_normal
        self.basis_gradients = basis_gradients
        self.boundary_vertices = boundary_vertices

    def apply_stiffness(self, values: Array, /) -> Array:
        """Apply the positive cotangent stiffness matrix without materializing it."""
        return self._edge_action(values, self.edge_weights)

    def apply_weighted_stiffness(
        self, values: Array, edge_coefficients: Array, /
    ) -> Array:
        """Apply ``sum_e c_e w_e (u_i - u_j)`` with per-edge coefficients ``c_e``.

        This is the matrix-free action of a variable-coefficient (for example
        mobility-weighted) cotangent stiffness. Coefficients are aligned with
        ``edges``; the action stays conservative for any coefficients.
        """
        coefficients = jnp.asarray(edge_coefficients)
        if coefficients.shape != self.edge_weights.shape:
            raise ValueError("edge_coefficients must have shape (num_edges,).")
        return self._edge_action(values, self.edge_weights * coefficients)

    def _edge_action(self, values: Array, weights: Array, /) -> Array:
        values_ = jnp.asarray(values)
        first = self.edges[:, 0]
        second = self.edges[:, 1]
        difference = values_[first] - values_[second]
        weighted = (
            weights.reshape((weights.shape[0],) + (1,) * (values_.ndim - 1)) * difference
        )
        result = jnp.zeros_like(values_)
        result = result.at[first].add(weighted)
        return result.at[second].add(-weighted)

    def apply_laplacian(self, values: Array, /) -> Array:
        """Apply the negative-semidefinite mass-inverted Laplace--Beltrami operator."""
        stiffness = self.apply_stiffness(values)
        mass = self.vertex_mass.reshape(
            (self.vertex_mass.shape[0],) + (1,) * (stiffness.ndim - 1)
        )
        return -stiffness / mass

    def gradient(self, vertex_values: Array, /) -> Array:
        """Map scalar or vector vertex values to piecewise-constant face gradients."""
        values = jnp.asarray(vertex_values)[self.faces]
        return ein.contract("fka,fk...->fa...", self.basis_gradients, values)

    def divergence(self, face_vectors: Array, /) -> Array:
        """Return the mass-adjoint divergence of piecewise-constant face vectors."""
        vectors = jnp.asarray(face_vectors)
        if vectors.shape[:2] != (self.faces.shape[0], 3):
            raise ValueError("face_vectors must have shape (num_faces, 3, ...).")
        contractions = ein.contract(
            "fa...,fka->fk...",
            vectors,
            self.basis_gradients,
        )
        weighted = contractions * self.face_area.reshape(
            (self.face_area.shape[0], 1) + (1,) * (contractions.ndim - 2)
        )
        result = jnp.zeros(
            (self.vertices.shape[0], *contractions.shape[2:]),
            dtype=vectors.dtype,
        )
        result = result.at[self.faces.reshape((-1,))].add(
            -weighted.reshape((-1, *contractions.shape[2:]))
        )
        mass = self.vertex_mass.reshape(
            (self.vertex_mass.shape[0],) + (1,) * (result.ndim - 1)
        )
        return result / mass

    @property
    def vertex_normals(self) -> Array:
        weighted = self.face_normal * self.face_area[:, None]
        normals = jnp.zeros_like(self.vertices)
        normals = normals.at[self.faces.reshape((-1,))].add(
            jnp.repeat(weighted, 3, axis=0)
        )
        return normals / jnp.linalg.norm(normals, axis=-1, keepdims=True)

    @property
    def mean_curvature_normal(self) -> Array:
        return 0.5 * self.apply_laplacian(self.vertices)

    @property
    def gaussian_curvature(self) -> Array:
        triangles = self.vertices[self.faces]
        previous = jnp.roll(triangles, 1, axis=1) - triangles
        following = jnp.roll(triangles, -1, axis=1) - triangles
        cosine = jnp.sum(previous * following, axis=-1) / (
            jnp.linalg.norm(previous, axis=-1) * jnp.linalg.norm(following, axis=-1)
        )
        angles = jnp.arccos(jnp.clip(cosine, -1.0, 1.0))
        angle_sum = jnp.zeros((self.vertices.shape[0],), dtype=self.vertices.dtype)
        angle_sum = angle_sum.at[self.faces.reshape((-1,))].add(angles.reshape((-1,)))
        target = jnp.where(self.boundary_vertices, jnp.pi, 2.0 * jnp.pi)
        return (target - angle_sum) / self.vertex_mass


def discrete_operators(mesh: TriangleMesh, /) -> DDGOperators:
    return DDGOperators(mesh)


__all__ = ["DDGOperators", "discrete_operators"]
