#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein
import phydrax.linalg as la
from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry.simplicial._ddg import DDGOperators, discrete_operators
from ...geometry.simplicial._mesh import TriangleMesh


_VACUUM_PERMEABILITY = 1.25663706212e-6


class ThinFilmLondonEvidence(StrictModule):
    linear_residual: Array
    constraint_residual: Array
    current_divergence: Array
    reciprocity_residual: Array
    minimum_energy_curvature: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ThinFilmLondonResult(StrictModule):
    stream_function: Array
    face_sheet_current: Array
    face_normal_field: Array
    constraint_reaction: Array
    magnetic_energy: Array
    kinetic_energy: Array
    applied_work: Array
    total_energy: Array
    evidence: ThinFilmLondonEvidence
    plan_id: str = eqx.field(static=True)


class ThinFilmLondonPlan(StrictModule, NonTrainableState):
    mesh: TriangleMesh
    operators: DDGOperators
    pearl_length: float = eqx.field(static=True)
    plane_normal: Array
    constraint_matrix: Array
    constraint_labels: tuple[str, ...] = eqx.field(static=True)
    hessian: Array
    moment_load: Array
    factorization: object
    softening_length: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: TriangleMesh,
        /,
        *,
        pearl_length: float,
        constraint_matrix: ArrayLike | None = None,
        constraint_labels: tuple[str, ...] = (),
        softening_length: float | None = None,
        tolerance: float = 1.0e-9,
    ):
        if not isinstance(mesh, TriangleMesh):
            raise TypeError("mesh must be TriangleMesh.")
        pearl = float(pearl_length)
        tolerance_ = float(tolerance)
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        triangles = vertices[faces]
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
        normal = np.mean(normals, axis=0)
        normal = normal / np.linalg.norm(normal)
        centered = vertices - np.mean(vertices, axis=0)
        planar_residual = np.max(np.abs(centered @ normal))
        normal_residual = np.max(np.linalg.norm(normals - normal, axis=-1))
        edge_length = np.linalg.norm(
            vertices[mesh.topology.edges[:, 0]] - vertices[mesh.topology.edges[:, 1]],
            axis=-1,
        )
        softening = (
            0.25 * float(np.mean(edge_length))
            if softening_length is None
            else float(softening_length)
        )
        if (
            not isfinite(pearl)
            or pearl < 0.0
            or not isfinite(tolerance_)
            or tolerance_ <= 0.0
            or not isfinite(softening)
            or softening <= 0.0
            or planar_residual > tolerance_ * max(float(np.max(edge_length)), 1.0)
            or normal_residual > tolerance_
        ):
            raise ValueError("Thin-film London geometry or material support is invalid.")
        vertex_count = vertices.shape[0]
        supplied = (
            np.zeros((0, vertex_count), dtype=np.float64)
            if constraint_matrix is None
            else np.asarray(constraint_matrix, dtype=np.float64)
        )
        labels = tuple(str(value).strip() for value in constraint_labels)
        if (
            supplied.ndim != 2
            or supplied.shape[1] != vertex_count
            or supplied.shape[0] != len(labels)
            or len(set(labels)) != len(labels)
            or any(not value for value in labels)
            or np.any(~np.isfinite(supplied))
        ):
            raise ValueError("London constraint matrix or labels are invalid.")
        ddg = discrete_operators(mesh)
        basis_gradients = np.asarray(ddg.basis_gradients)
        face_count = faces.shape[0]
        current_basis = np.zeros((face_count, 3, vertex_count), dtype=np.float64)
        for face in range(face_count):
            for local in range(3):
                vertex = faces[face, local]
                current_basis[face, :, vertex] += np.cross(
                    normals[face], basis_gradients[face, local]
                )
        face_area = np.asarray(ddg.face_area)
        centers = np.mean(triangles, axis=1)
        separation = centers[:, None, :] - centers[None, :, :]
        radius = np.sqrt(np.sum(separation**2, axis=-1) + softening**2)
        kernel = (
            _VACUUM_PERMEABILITY
            / (4.0 * np.pi)
            * face_area[:, None]
            * face_area[None, :]
            / radius
        )
        basis_matrix = np.moveaxis(current_basis, 2, 0).reshape((vertex_count, -1)).T
        magnetic_metric = np.kron(kernel, np.eye(3))
        magnetic_hessian = basis_matrix.T @ magnetic_metric @ basis_matrix
        kinetic_metric = np.diag(np.repeat(face_area, 3))
        kinetic_hessian = (
            _VACUUM_PERMEABILITY
            * pearl
            * (basis_matrix.T @ kinetic_metric @ basis_matrix)
        )
        hessian = 0.5 * (
            magnetic_hessian + kinetic_hessian + (magnetic_hessian + kinetic_hessian).T
        )
        gauge = np.asarray(ddg.vertex_mass)
        gauge = gauge / np.sum(gauge)
        constraints = np.concatenate((gauge[None, :], supplied), axis=0)
        rank = np.linalg.matrix_rank(constraints, tol=tolerance_)
        if rank != constraints.shape[0]:
            raise ValueError("London constraints are linearly dependent.")
        kkt = np.block(
            [
                [hessian, constraints.T],
                [constraints, np.zeros((constraints.shape[0], constraints.shape[0]))],
            ]
        )
        factorization = la.factorize(
            la.DenseLinearOperator(jnp.asarray(kkt)), la.FactorizationPolicy("lu")
        )
        centered_faces = centers - np.mean(vertices, axis=0)
        moment_load = np.zeros((vertex_count,), dtype=np.float64)
        for vertex in range(vertex_count):
            moment_load[vertex] = 0.5 * np.sum(
                face_area
                * ein.contract(
                    "i,fi->f",
                    normal,
                    np.cross(centered_faces, current_basis[:, :, vertex]),
                )
            )
        self.mesh = mesh
        self.operators = ddg
        self.pearl_length = pearl
        self.plane_normal = jnp.asarray(normal)
        self.constraint_matrix = jnp.asarray(constraints)
        self.constraint_labels = ("gauge-zero-mean", *labels)
        self.hessian = jnp.asarray(hessian)
        self.moment_load = jnp.asarray(moment_load)
        self.factorization = factorization
        self.softening_length = softening
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "planar-thin-film-london",
                "mesh": mesh.source_id,
                "vertices": array_tree_fingerprint(vertices),
                "faces": array_tree_fingerprint(faces),
                "pearl_length": pearl,
                "constraints": array_tree_fingerprint(constraints),
                "labels": self.constraint_labels,
                "softening_length": softening,
                "tolerance": tolerance_,
            }
        )

    @property
    def physical_constraint_count(self) -> int:
        return len(self.constraint_labels) - 1

    def solve(
        self,
        applied_normal_field: ArrayLike,
        constraint_values: ArrayLike | None = None,
        /,
    ) -> ThinFilmLondonResult:
        applied = jnp.asarray(applied_normal_field, dtype=self.hessian.dtype)
        if applied.shape != ():
            raise ValueError(
                "Applied London field must be scalar and normal to the film."
            )
        values = (
            jnp.zeros((self.physical_constraint_count,), dtype=self.hessian.dtype)
            if constraint_values is None
            else jnp.asarray(constraint_values, dtype=self.hessian.dtype)
        )
        if values.shape != (self.physical_constraint_count,):
            raise ValueError("London constraint values do not match the plan.")
        right = jnp.concatenate(
            (
                applied * self.moment_load,
                jnp.concatenate((jnp.zeros((1,), dtype=values.dtype), values)),
            )
        )
        linear = self.factorization.solve(right)
        vertex_count = self.mesh.vertices.shape[0]
        stream = linear.value[:vertex_count]
        reaction = linear.value[vertex_count:]
        gradient = self.operators.gradient(stream)
        current = jnp.cross(self.plane_normal, gradient)
        centers = jnp.mean(self.mesh.vertices[self.mesh.faces], axis=1)
        separation = centers[:, None, :] - centers[None, :, :]
        radius = (jnp.sum(separation**2, axis=-1) + self.softening_length**2) ** 1.5
        induced = (
            _VACUUM_PERMEABILITY
            / (4.0 * jnp.pi)
            * jnp.sum(
                self.operators.face_area[None, :]
                * contract(
                    "i,fhi->fh",
                    self.plane_normal,
                    jnp.cross(current[None, :, :], separation),
                    backend="jax",
                )
                / radius,
                axis=1,
            )
        )
        field = applied + induced
        magnetic_energy = (
            0.5
            * stream
            @ (
                self.hessian
                - _VACUUM_PERMEABILITY
                * self.pearl_length
                * _kinetic_matrix(self.operators)
            )
            @ stream
        )
        kinetic_energy = (
            0.5
            * _VACUUM_PERMEABILITY
            * self.pearl_length
            * jnp.sum(self.operators.face_area * jnp.sum(current**2, axis=-1))
        )
        applied_work = applied * (self.moment_load @ stream)
        total_energy = magnetic_energy + kinetic_energy - applied_work
        residual = (
            self.hessian @ stream
            + self.constraint_matrix.T @ reaction
            - applied * self.moment_load
        )
        constraint_residual = self.constraint_matrix @ stream - jnp.concatenate(
            (jnp.zeros((1,), dtype=values.dtype), values)
        )
        divergence = self.operators.divergence(current)
        interior_divergence = jnp.where(self.operators.boundary_vertices, 0.0, divergence)
        eigenvalues = jnp.linalg.eigvalsh(
            self.hessian + self.constraint_matrix.T @ self.constraint_matrix
        )
        curvature = jnp.min(eigenvalues)
        finite = (
            linear.successful
            & jnp.all(jnp.isfinite(stream))
            & jnp.all(jnp.isfinite(current))
            & jnp.isfinite(total_energy)
        )
        scale = jnp.maximum(jnp.linalg.norm(applied * self.moment_load), 1.0)
        successful = (
            finite
            & (jnp.linalg.norm(residual) <= self.tolerance * scale)
            & (jnp.linalg.norm(constraint_residual) <= self.tolerance)
            & (
                jnp.linalg.norm(interior_divergence)
                <= self.tolerance * jnp.maximum(jnp.linalg.norm(current), 1.0)
            )
            & (curvature >= -self.tolerance)
        )
        evidence = ThinFilmLondonEvidence(
            jnp.linalg.norm(residual),
            jnp.linalg.norm(constraint_residual),
            divergence,
            jnp.asarray(0.0, dtype=stream.dtype),
            curvature,
            finite,
            successful,
            self.plan_id,
        )
        return ThinFilmLondonResult(
            stream,
            current,
            field,
            reaction,
            magnetic_energy,
            kinetic_energy,
            applied_work,
            total_energy,
            evidence,
            self.plan_id,
        )

    def inductance_matrix(self, /) -> tuple[Array, Array]:
        count = self.physical_constraint_count
        if count == 0:
            empty = jnp.zeros((0, 0), dtype=self.hessian.dtype)
            return empty, jnp.asarray(0.0, dtype=self.hessian.dtype)
        vertex_count = self.mesh.vertices.shape[0]
        right = jnp.concatenate(
            (
                jnp.zeros((vertex_count, count), dtype=self.hessian.dtype),
                jnp.concatenate(
                    (
                        jnp.zeros((1, count), dtype=self.hessian.dtype),
                        jnp.eye(count, dtype=self.hessian.dtype),
                    ),
                    axis=0,
                ),
            ),
            axis=0,
        )
        linear = self.factorization.solve(right)
        reactions = linear.value[vertex_count + 1 :, :]
        matrix = -reactions
        reciprocity = jnp.max(jnp.abs(matrix - matrix.T), initial=0.0)
        return 0.5 * (matrix + matrix.T), reciprocity


def _kinetic_matrix(operators: DDGOperators) -> Array:
    vertex_count = operators.vertices.shape[0]
    basis = jnp.eye(vertex_count, dtype=operators.vertices.dtype)
    gradients = jax.vmap(operators.gradient)(basis).swapaxes(0, 1)
    currents = jnp.cross(operators.face_normal[:, None, :], gradients)
    return contract(
        "f,fia,fja->ij",
        operators.face_area,
        currents,
        currents,
        backend="jax",
    )


__all__ = ["ThinFilmLondonEvidence", "ThinFilmLondonPlan", "ThinFilmLondonResult"]
