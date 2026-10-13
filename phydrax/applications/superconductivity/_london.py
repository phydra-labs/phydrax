#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.linalg as la
import phydrax.linalg.eigen as le
from phydrax.sparse import EdgeRelation, route_reduce

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._physical import ElectromagneticScaleContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import positive_integer
from ...geometry.simplicial._ddg import DDGOperators, discrete_operators
from ...geometry.simplicial._mesh import TriangleMesh
from ...typing import checked, parse


_VACUUM_PERMEABILITY = float(ElectromagneticScaleContract.si().vacuum_permeability)


type LondonSolveRoute = Literal["auto", "dense", "minres", "gmres"]


class LondonSolvePolicy(StrictModule, NonTrainableState):
    """Fail-closed dense admission and bounded matrix-free solve/eigen resources."""

    route: LondonSolveRoute = eqx.field(static=True)
    dense_maximum_bytes: int = eqx.field(static=True)
    dense_maximum_dofs: int = eqx.field(static=True)
    face_tile_size: int = eqx.field(static=True)
    krylov_restart: int = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    eigen_subspace_dimension: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        route: LondonSolveRoute = "auto",
        dense_maximum_bytes: int = 16 * 1024 * 1024,
        dense_maximum_dofs: int = 512,
        face_tile_size: int = 8,
        krylov_restart: int = 32,
        maximum_steps: int = 2000,
        workspace_bytes: int = 64 * 1024 * 1024,
        eigen_subspace_dimension: int = 32,
    ) -> None:
        route = parse(route, LondonSolveRoute, "route")
        values = tuple(
            positive_integer(value, name)
            for value, name in (
                (dense_maximum_bytes, "dense_maximum_bytes"),
                (dense_maximum_dofs, "dense_maximum_dofs"),
                (face_tile_size, "face_tile_size"),
                (krylov_restart, "krylov_restart"),
                (maximum_steps, "maximum_steps"),
                (workspace_bytes, "workspace_bytes"),
                (eigen_subspace_dimension, "eigen_subspace_dimension"),
            )
        )
        if values[-1] < 2:
            raise ValueError("eigen_subspace_dimension must be at least two.")
        self.route = route
        (
            self.dense_maximum_bytes,
            self.dense_maximum_dofs,
            self.face_tile_size,
            self.krylov_restart,
            self.maximum_steps,
            self.workspace_bytes,
            self.eigen_subspace_dimension,
        ) = values
        self.policy_id = canonical_fingerprint(
            {"kind": "london-solve-policy", "route": route, "capacities": values}
        )


class _LondonAction(StrictModule):
    faces: Array
    local_current: Array
    local_kinetic: Array
    centers: Array
    area: Array
    relation: EdgeRelation
    softening: float = eqx.field(static=True)
    tile: int = eqx.field(static=True)

    def current(self, stream: Array) -> Array:
        return jnp.sum(self.local_current * stream[self.faces, None], axis=1)

    def transpose_current(self, current: Array) -> Array:
        local = jnp.sum(self.local_current * current[:, None, :], axis=-1)
        return route_reduce(self.relation, local.reshape((-1,)))

    def magnetic_current(self, current: Array) -> Array:
        def row(center: Array) -> Array:
            radius = jnp.sqrt(
                jnp.sum((center - self.centers) ** 2, axis=-1) + self.softening**2
            )
            weights = _VACUUM_PERMEABILITY / (4.0 * jnp.pi) * self.area / radius
            return jnp.sum(weights[:, None] * current, axis=0)

        return self.area[:, None] * jax.lax.map(row, self.centers, batch_size=self.tile)

    def magnetic(self, stream: Array) -> Array:
        return self.transpose_current(self.magnetic_current(self.current(stream)))

    def kinetic(self, stream: Array) -> Array:
        kinetic = jnp.sum(self.local_kinetic * stream[self.faces][:, None, :], axis=-1)
        return route_reduce(self.relation, kinetic.reshape((-1,)))

    def __call__(self, stream: Array) -> Array:
        return self.magnetic(stream) + self.kinetic(stream)

    def normal_field(self, current: Array, normal: Array) -> Array:
        def row(center: Array) -> Array:
            separation = center - self.centers
            radius = (jnp.sum(separation**2, axis=-1) + self.softening**2) ** 1.5
            numerator = jnp.sum(jnp.cross(current, separation) * normal, axis=-1)
            return (
                _VACUUM_PERMEABILITY
                / (4.0 * jnp.pi)
                * jnp.sum(self.area * numerator / radius)
            )

        return jax.lax.map(row, self.centers, batch_size=self.tile)


class _LondonKKTAction(StrictModule):
    energy: _LondonAction
    constraints: Array

    def __call__(self, vector: Array) -> Array:
        count = self.constraints.shape[1]
        stream, reaction = vector[:count], vector[count:]
        return jnp.concatenate(
            (
                self.energy(stream) / _VACUUM_PERMEABILITY
                + self.constraints.T @ reaction,
                self.constraints @ stream,
            )
        )


class _LondonCurvatureAction(StrictModule):
    energy: _LondonAction
    constraints: Array

    def __call__(self, vector: Array) -> Array:
        return self.energy(vector) + self.constraints.T @ (self.constraints @ vector)


def _prepare_london_solve(
    action: _LondonAction, constraints: Array, policy: LondonSolvePolicy, tolerance: float
) -> tuple[LondonSolveRoute, la.PreparedLinearSolve]:
    vertex_count = constraints.shape[1]
    dofs = vertex_count + constraints.shape[0]
    # Admit the KKT, its factors, and dense curvature work before allocation.
    dense_bytes = 8 * (3 * dofs**2 + 3 * vertex_count**2)
    admitted = (
        dofs <= policy.dense_maximum_dofs and dense_bytes <= policy.dense_maximum_bytes
    )
    match policy.route:
        case "auto":
            route: LondonSolveRoute = "dense" if admitted else "minres"
        case "dense":
            if not admitted:
                raise ValueError(
                    "London dense KKT exceeds its declared byte/dof bound; select minres or gmres."
                )
            route = "dense"
        case "minres" | "gmres":
            route = policy.route
        case _:
            raise ValueError("Unsupported London solve route.")
    tile_bytes = 8 * action.tile * action.area.size * 8
    basis_bytes = 8 * vertex_count * min(vertex_count, policy.eigen_subspace_dimension)
    if tile_bytes + basis_bytes > policy.workspace_bytes:
        raise ValueError(
            "London face-kernel/eigen workspace exceeds its declared byte bound."
        )
    kkt_action = _LondonKKTAction(action, constraints)
    space = la.ArraySpace((dofs,), dtype=np.float64)
    properties = la.OperatorProperties(
        self_adjoint=True, evidence={"self_adjoint": "construction"}
    )
    operator: la.AbstractLinearOperator = la.FunctionLinearOperator(
        kkt_action,
        source=space,
        target=space,
        transpose_action=kkt_action,
        properties=properties,
        closure_convert=False,
    )
    materialization = la.MaterializationPolicy(
        max_entries=dofs**2, max_bytes=policy.dense_maximum_bytes
    )
    match route:
        case "dense":
            matrix = jax.lax.map(
                lambda index: kkt_action(jax.nn.one_hot(index, dofs, dtype=jnp.float64)),
                jnp.arange(dofs),
                batch_size=1,
            ).T
            operator = la.DenseLinearOperator(matrix, properties=properties)
            method: la.AbstractLinearMethod = la.DenseLU()
        case "minres":
            method = la.MINRES()
        case "gmres":
            method = la.GMRES(restart=min(policy.krylov_restart, dofs))
        case _:
            raise ValueError("Unsupported prepared London solve route.")
    prepared = la.prepare(
        la.LinearSystem(operator),
        la.LinearSolvePolicy(
            method,
            tolerance=la.TolerancePolicy(
                relative=min(tolerance, 1.0e-11),
                absolute=min(tolerance, 1.0e-12),
                max_steps=policy.maximum_steps,
            ),
            resources=la.SolveResourcePolicy(workspace_bytes=policy.workspace_bytes),
            materialization=materialization,
            failure=la.FailurePolicy("status"),
        ),
    )
    return route, prepared


def _prepare_london_curvature(
    action: _LondonAction,
    constraints: Array,
    policy: LondonSolvePolicy,
    route: LondonSolveRoute,
    tolerance: float,
) -> le.EigenSolveResult:
    vertex_count = constraints.shape[1]
    curvature_action = _LondonCurvatureAction(action, constraints)
    space = la.ArraySpace((vertex_count,), dtype=np.float64)
    operator: la.AbstractLinearOperator = la.FunctionLinearOperator(
        curvature_action,
        source=space,
        target=space,
        transpose_action=curvature_action,
        properties=la.OperatorProperties(
            self_adjoint=True, evidence={"self_adjoint": "construction"}
        ),
        closure_convert=False,
    )
    if route == "dense":
        # Native dense eigen preparation receives only the admitted vertex matrix,
        # never a vmapped face-by-vertex current workspace.
        matrix = jax.lax.map(
            lambda index: curvature_action(
                jax.nn.one_hot(index, vertex_count, dtype=jnp.float64)
            ),
            jnp.arange(vertex_count),
            batch_size=1,
        ).T
        operator = la.DenseLinearOperator(matrix, properties=operator.properties)
        method: le.DenseEigh | le.RestartedLanczos = le.DenseEigh()
    else:
        method = le.RestartedLanczos(
            subspace_dimension=min(vertex_count, policy.eigen_subspace_dimension)
        )
    return le.eigensolve(
        le.Eigenproblem(operator),
        policy=le.EigenSolvePolicy(
            method,
            count=1,
            max_steps=policy.maximum_steps,
            key=jax.random.key(0),
            tolerance=le.EigenTolerancePolicy(
                relative=1.0e-10, absolute=min(tolerance, 1.0e-12)
            ),
            resources=le.EigenResourcePolicy(
                workspace_bytes=policy.workspace_bytes,
                krylov_basis_bytes=policy.workspace_bytes,
            ),
            materialization=la.MaterializationPolicy(
                max_entries=vertex_count**2,
                max_bytes=policy.dense_maximum_bytes,
            ),
            failure=la.FailurePolicy("status"),
        ),
    )


class ThinFilmLondonEvidence(StrictModule):
    linear_residual: Array
    constraint_residual: Array
    current_divergence: Array
    reciprocity_residual: Array
    minimum_energy_curvature: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    solve_route: LondonSolveRoute = eqx.field(static=True)
    linear_status: Array
    curvature_successful: Array


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
    energy_action: _LondonAction
    minimum_energy_curvature: Array
    curvature_successful: Array
    solve_policy: LondonSolvePolicy
    solve_route: LondonSolveRoute = eqx.field(static=True)
    moment_load: Array
    prepared_solve: la.PreparedLinearSolve
    softening_length: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
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
        solve_policy: LondonSolvePolicy | None = None,
    ) -> None:
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
        local_current = np.cross(normals[:, None, :], basis_gradients)
        face_area = np.asarray(ddg.face_area)
        centers = np.mean(triangles, axis=1)
        policy = LondonSolvePolicy() if solve_policy is None else solve_policy
        if not isinstance(policy, LondonSolvePolicy):
            raise TypeError("solve_policy must be LondonSolvePolicy or None.")
        relation = EdgeRelation(
            np.repeat(np.arange(face_count, dtype=np.int32), 3),
            faces.reshape((-1,)),
            source_size=face_count,
            target_size=vertex_count,
        )
        local_kinetic = (
            _VACUUM_PERMEABILITY
            * pearl
            * face_area[:, None, None]
            * (local_current @ local_current.swapaxes(1, 2))
        )
        action = _LondonAction(
            jnp.asarray(faces),
            jnp.asarray(local_current),
            jnp.asarray(local_kinetic),
            jnp.asarray(centers),
            jnp.asarray(face_area),
            relation,
            softening,
            min(policy.face_tile_size, face_count),
        )
        gauge = np.asarray(ddg.vertex_mass)
        gauge = gauge / np.sum(gauge)
        constraints = np.concatenate((gauge[None, :], supplied), axis=0)
        rank = np.linalg.matrix_rank(constraints, tol=tolerance_)
        if rank != constraints.shape[0]:
            raise ValueError("London constraints are linearly dependent.")
        constraint_array = jnp.asarray(constraints)
        route, prepared = _prepare_london_solve(
            action, constraint_array, policy, tolerance_
        )
        curvature_result = _prepare_london_curvature(
            action, constraint_array, policy, route, tolerance_
        )
        centered_faces = centers - np.mean(vertices, axis=0)
        local_moment = (
            0.5
            * face_area[:, None]
            * np.sum(
                normal * np.cross(centered_faces[:, None, :], local_current), axis=-1
            )
        )
        moment_load = route_reduce(relation, jnp.asarray(local_moment.reshape((-1,))))
        self.mesh = mesh
        self.operators = ddg
        self.pearl_length = pearl
        self.plane_normal = jnp.asarray(normal)
        self.constraint_matrix = jnp.asarray(constraints)
        self.constraint_labels = ("gauge-zero-mean", *labels)
        self.energy_action = action
        self.minimum_energy_curvature = curvature_result.eigenvalues[0]
        self.curvature_successful = curvature_result.successful
        self.solve_policy = policy
        self.solve_route = route
        self.moment_load = moment_load
        self.prepared_solve = prepared
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
                "solve_policy": policy.policy_id,
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
        applied = jnp.asarray(applied_normal_field, dtype=self.moment_load.dtype)
        if applied.shape != ():
            raise ValueError(
                "Applied London field must be scalar and normal to the film."
            )
        values = (
            jnp.zeros((self.physical_constraint_count,), dtype=self.moment_load.dtype)
            if constraint_values is None
            else jnp.asarray(constraint_values, dtype=self.moment_load.dtype)
        )
        if values.shape != (self.physical_constraint_count,):
            raise ValueError("London constraint values do not match the plan.")
        right = jnp.concatenate(
            (
                applied * self.moment_load / _VACUUM_PERMEABILITY,
                jnp.concatenate((jnp.zeros((1,), dtype=values.dtype), values)),
            )
        )
        linear = la.solve(self.prepared_solve, right)
        vertex_count = self.mesh.vertices.shape[0]
        stream = linear.value[:vertex_count]
        reaction = _VACUUM_PERMEABILITY * linear.value[vertex_count:]
        current = self.energy_action.current(stream)
        field = applied + self.energy_action.normal_field(current, self.plane_normal)
        magnetic_action = self.energy_action.transpose_current(
            self.energy_action.magnetic_current(current)
        )
        magnetic_energy = 0.5 * stream @ magnetic_action
        kinetic_energy = (
            0.5
            * _VACUUM_PERMEABILITY
            * self.pearl_length
            * jnp.sum(self.operators.face_area * jnp.sum(current**2, axis=-1))
        )
        applied_work = applied * (self.moment_load @ stream)
        total_energy = magnetic_energy + kinetic_energy - applied_work
        residual = (
            magnetic_action
            + self.energy_action.kinetic(stream)
            + self.constraint_matrix.T @ reaction
            - applied * self.moment_load
        )
        constraint_residual = self.constraint_matrix @ stream - jnp.concatenate(
            (jnp.zeros((1,), dtype=values.dtype), values)
        )
        divergence = self.operators.divergence(current)
        interior_divergence = jnp.where(self.operators.boundary_vertices, 0.0, divergence)
        curvature = self.minimum_energy_curvature
        finite = (
            linear.successful
            & jnp.all(jnp.isfinite(stream))
            & jnp.all(jnp.isfinite(current))
            & jnp.isfinite(total_energy)
        )
        scale = jnp.maximum(jnp.linalg.norm(applied * self.moment_load), 1.0)
        successful = (
            finite
            & self.curvature_successful
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
            self.solve_route,
            linear.status,
            self.curvature_successful,
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
        """Return inductance and reciprocity, refusing failed native solves on host."""
        count = self.physical_constraint_count
        if count == 0:
            empty = jnp.zeros((0, 0), dtype=self.moment_load.dtype)
            return empty, jnp.asarray(0.0, dtype=self.moment_load.dtype)
        vertex_count = self.mesh.vertices.shape[0]
        right = jnp.concatenate(
            (
                jnp.zeros((vertex_count, count), dtype=self.moment_load.dtype),
                jnp.concatenate(
                    (
                        jnp.zeros((1, count), dtype=self.moment_load.dtype),
                        jnp.eye(count, dtype=self.moment_load.dtype),
                    ),
                    axis=0,
                ),
            ),
            axis=0,
        )
        linear = la.solve(self.prepared_solve, right)
        if not bool(jnp.all(linear.successful)):
            statuses = np.asarray(linear.status)
            messages = tuple(
                la.linear_status_message(int(status))
                for status in statuses.reshape((-1,))
            )
            raise RuntimeError(
                f"London inductance multi-RHS solve failed on route {self.solve_route!r}: "
                f"status={statuses!r}, messages={messages!r}, "
                f"residual_norm={np.asarray(linear.diagnostics.residual_norm)!r}."
            )
        reactions = _VACUUM_PERMEABILITY * linear.value[vertex_count + 1 :, :]
        matrix = -reactions
        reciprocity = jnp.max(jnp.abs(matrix - matrix.T), initial=0.0)
        return 0.5 * (matrix + matrix.T), reciprocity


__all__ = [
    "LondonSolvePolicy",
    "LondonSolveRoute",
    "ThinFilmLondonEvidence",
    "ThinFilmLondonPlan",
    "ThinFilmLondonResult",
]
