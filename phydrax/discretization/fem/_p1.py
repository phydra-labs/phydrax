#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import (
    ArraySpace,
    DiagonalPairing,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    OperatorProperties,
    prepare,
    PreparedLinearSolve,
    solve,
)
from ...sparse import EdgeRelation, SparseLinearMap
from .._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
)
from .._lifecycle import (
    AbstractDiscretizationPlan,
    AbstractPreparedDiscretization,
    validate_prepared_metadata,
)
from .._measure import DiscreteMeasure
from .._spaces import DiscreteFieldSpace, EntityDofLayout
from .._support import DiscreteSupport
from .._topology import CellComplexTopology
from .._triangular import triangle_cell_complex


def p1_local_matrices(
    vertices: ArrayLike,
    faces: ArrayLike,
    /,
) -> tuple[Array, Array, Array, Array]:
    """Return area, ambient gradients, local mass, and local stiffness per triangle."""
    points = jnp.asarray(vertices)
    cells = jnp.asarray(faces, dtype=jnp.int32)
    triangles = points[cells]
    edges = jnp.stack(
        (triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=-1,
    )
    gram = oe.contract("nai,naj->nij", edges, edges)
    determinant = gram[:, 0, 0] * gram[:, 1, 1] - gram[:, 0, 1] ** 2
    determinant = eqx.error_if(
        determinant,
        jnp.any(~jnp.isfinite(determinant)) | jnp.any(determinant <= 0.0),
        "P1 triangles require finite positive metric determinant.",
    )
    area = 0.5 * jnp.sqrt(determinant)
    reference_gradients = jnp.asarray(
        [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=points.dtype,
    )
    inverse_gram = jnp.linalg.inv(gram)
    gradients = oe.contract(
        "nai,nij,kj->nak",
        edges,
        inverse_gram,
        reference_gradients,
    ).swapaxes(1, 2)
    local_stiffness = area[:, None, None] * oe.contract(
        "nia,nja->nij", gradients, gradients
    )
    mass_template = jnp.asarray(
        [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]],
        dtype=points.dtype,
    )
    local_mass = area[:, None, None] * mass_template[None, :, :] / 12.0
    return area, gradients, local_mass, local_stiffness


def _assembled_operator(
    faces: np.ndarray,
    local_values: Array,
    vertex_count: int,
    operator_id: str,
    /,
    *,
    positive_definite: bool,
) -> SparseLinearMap:
    source = np.broadcast_to(faces[:, None, :], (faces.shape[0], 3, 3)).reshape((-1,))
    target = np.broadcast_to(faces[:, :, None], (faces.shape[0], 3, 3)).reshape((-1,))
    relation = EdgeRelation(
        source,
        target,
        source_size=vertex_count,
        target_size=vertex_count,
    )
    return SparseLinearMap(
        relation,
        local_values.reshape((-1,)),
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=positive_definite,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
                **({"positive_definite": "construction"} if positive_definite else {}),
            },
        ),
        operator_id=operator_id,
    )


class P1FiniteElementPlan(AbstractDiscretizationPlan):
    """Validated affine triangular P1 discretization plan."""

    vertices: Array
    faces: Array
    field_name: str = eqx.field(static=True)
    key: DiscretizationKey
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertices: ArrayLike,
        faces: ArrayLike,
        /,
        *,
        field_name: str = "state",
        key: DiscretizationKey | None = None,
        plan_id: str | None = None,
    ):
        points = np.asarray(vertices, dtype=float)
        cells = np.asarray(faces, dtype=np.int32)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 2:
            raise ValueError("vertices must have shape (num_vertices > 0, ambient >= 2).")
        if cells.ndim != 2 or cells.shape[0] == 0 or cells.shape[1] != 3:
            raise ValueError("faces must have shape (num_faces > 0, 3).")
        if np.any(~np.isfinite(points)):
            raise ValueError("vertices must be finite.")
        if np.any(cells < 0) or np.any(cells >= points.shape[0]):
            raise ValueError("faces contain out-of-range vertex indices.")
        if np.any(
            (cells[:, 0] == cells[:, 1])
            | (cells[:, 1] == cells[:, 2])
            | (cells[:, 2] == cells[:, 0])
        ):
            raise ValueError("Every triangle must reference three distinct vertices.")
        if np.unique(np.sort(cells, axis=1), axis=0).shape[0] != cells.shape[0]:
            raise ValueError("P1 meshes cannot contain duplicate faces.")
        triangles = points[cells]
        first = triangles[:, 1] - triangles[:, 0]
        second = triangles[:, 2] - triangles[:, 0]
        gram00 = np.sum(first * first, axis=-1)
        gram11 = np.sum(second * second, axis=-1)
        gram01 = np.sum(first * second, axis=-1)
        determinant = gram00 * gram11 - gram01**2
        scale = np.maximum(gram00, gram11) ** 2
        if np.any(determinant <= 64.0 * np.finfo(float).eps * scale):
            raise ValueError("P1 mesh contains a degenerate triangle.")
        if np.unique(cells).size != points.shape[0]:
            raise ValueError("Every P1 vertex must belong to at least one triangle.")
        triangle_cell_complex(cells, points.shape[0])
        field = str(field_name)
        if not field:
            raise ValueError("field_name must be non-empty.")
        key_ = (
            DiscretizationKey(
                "finite_element",
                DiscretizationRole.PHYSICAL,
                domain_labels=("space",),
            )
            if key is None
            else key
        )
        if not isinstance(key_, DiscretizationKey):
            raise TypeError("key must be a DiscretizationKey.")
        capabilities = (
            DiscretizationCapability.PROJECTION,
            DiscretizationCapability.RECONSTRUCTION,
            DiscretizationCapability.TRACE,
            DiscretizationCapability.VARIATIONAL_ASSEMBLY,
            DiscretizationCapability.SPARSE_ASSEMBLY,
            DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
        )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "p1-finite-element-plan",
                    "vertices": array_tree_fingerprint(points),
                    "faces": array_tree_fingerprint(cells),
                    "field": field,
                    "key": key_.key_id,
                }
            )
            if plan_id is None
            else str(plan_id)
        )
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.vertices = jnp.asarray(points)
        self.faces = jnp.asarray(cells, dtype=jnp.int32)
        self.field_name = field
        self.key = key_
        self.capabilities = capabilities
        self.plan_id = identifier

    def prepare(
        self, /, *, numeric_version: str = "0"
    ) -> "P1FiniteElementDiscretization":
        return P1FiniteElementDiscretization(self, numeric_version=numeric_version)


class P1DirichletElimination(eqx.Module):
    """Free-coordinate elimination with an explicit full-space lift."""

    boundary_mask: Array
    free_indices: Array
    lift: Array
    stiffness: SparseLinearMap
    reduced_stiffness: SparseLinearMap

    def reduce_rhs(self, right_hand_side: ArrayLike, /) -> Array:
        rhs = jnp.asarray(right_hand_side)
        if rhs.shape != self.lift.shape:
            raise ValueError(
                "Dirichlet right-hand side must match the full vertex space."
            )
        return (rhs - self.stiffness.mv(self.lift))[self.free_indices]

    def expand(self, free_values: ArrayLike, /) -> Array:
        values = jnp.asarray(free_values)
        if values.shape != self.free_indices.shape:
            raise ValueError("Free values must match the free-coordinate count.")
        return self.lift.at[self.free_indices].set(values)


class P1FiniteElementDiscretization(AbstractPreparedDiscretization):
    """Prepared affine-triangle scalar H¹ P1 finite element method."""

    vertices: Array
    faces: Array
    topology: CellComplexTopology
    areas: Array
    basis_gradients: Array
    mass: SparseLinearMap
    stiffness: SparseLinearMap
    lumped_mass: Array
    key: DiscretizationKey
    support: DiscreteSupport
    field_spaces: tuple[DiscreteFieldSpace, ...]
    measures: tuple[DiscreteMeasure, ...]
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    preparation: PreparationReport

    def __init__(
        self,
        plan: P1FiniteElementPlan,
        /,
        *,
        numeric_version: str = "0",
    ):
        if not isinstance(plan, P1FiniteElementPlan):
            raise TypeError("plan must be a P1FiniteElementPlan.")
        points = np.asarray(plan.vertices, dtype=float)
        cells = np.asarray(plan.faces, dtype=np.int32)
        topology = triangle_cell_complex(cells, points.shape[0])
        area, gradients, local_mass, local_stiffness = p1_local_matrices(
            plan.vertices, plan.faces
        )
        area_host = np.asarray(area)
        if np.any(~np.isfinite(area_host)) or np.any(area_host <= 0.0):
            raise ValueError("P1 mesh contains degenerate triangles.")
        embedding_id = canonical_fingerprint(
            {
                "kind": "p1-embedding",
                "vertices": array_tree_fingerprint(points),
            }
        )
        support = DiscreteSupport(topology, points.shape[1], embedding_id)
        mass = _assembled_operator(
            cells,
            local_mass,
            points.shape[0],
            canonical_fingerprint({"kind": "p1-mass", "plan": plan.plan_id}),
            positive_definite=True,
        )
        stiffness = _assembled_operator(
            cells,
            local_stiffness,
            points.shape[0],
            canonical_fingerprint({"kind": "p1-stiffness", "plan": plan.plan_id}),
            positive_definite=False,
        )
        lumped_mass = (
            jnp.zeros((points.shape[0],), dtype=area.dtype)
            .at[plan.faces.reshape((-1,))]
            .add(jnp.repeat(area / 3.0, 3))
        )
        vertex_entities = topology.entity_sets[0]
        face_entities = topology.entity_sets[2]
        field_space = DiscreteFieldSpace(
            plan.field_name,
            support.support_id,
            EntityDofLayout(
                vertex_entities.entity_set_id,
                vertex_entities.count,
                vertex_entities.count,
            ),
            ArraySpace(
                (points.shape[0],),
                pairing=DiagonalPairing(lumped_mass),
            ),
            representation="point_value",
            conformity="H1",
            projection_id=canonical_fingerprint(
                {"kind": "p1-nodal-projection", "plan": plan.plan_id}
            ),
            reconstruction_id=canonical_fingerprint(
                {"kind": "p1-barycentric-reconstruction", "plan": plan.plan_id}
            ),
        )
        measures = (
            DiscreteMeasure(
                "vertex_lumped",
                support.support_id,
                vertex_entities.entity_set_id,
                lumped_mass,
                normalization="physical",
            ),
            DiscreteMeasure(
                "cell_area",
                support.support_id,
                face_entities.entity_set_id,
                area,
                normalization="physical",
            ),
        )
        preparation = PreparationReport(
            capabilities=plan.capabilities,
            resource_counts={
                "vertices": points.shape[0],
                "edges": topology.entity_sets[1].count,
                "cells": cells.shape[0],
                "mass_routes": mass.relation.route_shape[0],
                "stiffness_routes": stiffness.relation.route_shape[0],
            },
        )
        spaces, measures_, capabilities = validate_prepared_metadata(
            key=plan.key,
            support=support,
            field_spaces=(field_space,),
            measures=measures,
            capabilities=plan.capabilities,
            preparation=preparation,
        )
        version = str(numeric_version)
        if not version:
            raise ValueError("numeric_version must be non-empty.")
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-p1-finite-element",
                "plan": plan.plan_id,
                "embedding": embedding_id,
                "numeric_version": version,
            }
        )
        self.vertices = plan.vertices
        self.faces = plan.faces
        self.topology = topology
        self.areas = area
        self.basis_gradients = gradients
        self.mass = mass
        self.stiffness = stiffness
        self.lumped_mass = lumped_mass
        self.key = plan.key
        self.support = support
        self.field_spaces = spaces
        self.measures = measures_
        self.capabilities = capabilities
        self.plan_id = plan.plan_id
        self.prepared_id = prepared_id
        self.numeric_version = version
        self.preparation = preparation

    @property
    def boundary_vertex_mask(self) -> Array:
        return self.topology.entity_sets[0].subset("boundary").mask

    @property
    def boundary_edges(self) -> Array:
        mask = np.asarray(
            self.topology.entity_sets[1].subset("boundary").mask,
            dtype=bool,
        )
        incidence = self.topology.incidences[0]
        matrix = incidence.scipy_boundary()
        edges = []
        for edge_index in np.flatnonzero(mask):
            edges.append(np.sort(matrix[:, edge_index].nonzero()[0]))
        return jnp.asarray(np.asarray(edges, dtype=np.int32))

    @property
    def quadrature_points(self) -> Array:
        barycentric = jnp.asarray(
            [
                [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
                [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
            ],
            dtype=self.vertices.dtype,
        )
        return oe.contract(
            "qi,cia->cqa",
            barycentric,
            self.vertices[self.faces],
        )

    def assemble_load(self, values: ArrayLike, /) -> Array:
        """Assemble volume loading from degree-two three-point triangle quadrature."""
        samples = jnp.asarray(values)
        expected = (int(self.faces.shape[0]), 3)
        if samples.shape != expected:
            raise ValueError(f"Volume load samples must have shape {expected}.")
        barycentric = jnp.asarray(
            [
                [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
                [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
            ],
            dtype=samples.dtype,
        )
        local = self.areas[:, None] / 3.0 * oe.contract("qi,cq->ci", barycentric, samples)
        return (
            jnp.zeros((self.vertices.shape[0],), dtype=samples.dtype)
            .at[self.faces.reshape((-1,))]
            .add(local.reshape((-1,)))
        )

    def assemble_boundary_load(self, values: ArrayLike, /) -> Array:
        """Assemble midpoint Neumann loading on every boundary edge."""
        boundary_edges = self.boundary_edges
        samples = jnp.asarray(values)
        if samples.shape != (int(boundary_edges.shape[0]),):
            raise ValueError("Boundary load must provide one midpoint value per edge.")
        lengths = jnp.linalg.norm(
            self.vertices[boundary_edges[:, 1]] - self.vertices[boundary_edges[:, 0]],
            axis=-1,
        )
        contributions = samples * lengths / 2.0
        return (
            jnp.zeros((self.vertices.shape[0],), dtype=samples.dtype)
            .at[boundary_edges.reshape((-1,))]
            .add(jnp.repeat(contributions, 2))
        )

    def reconstruct(self, values: ArrayLike, barycentric: ArrayLike, /) -> Array:
        """Reconstruct one nodal field at per-cell barycentric query points."""
        field = jnp.asarray(values)
        coordinates = jnp.asarray(barycentric)
        if field.shape != (int(self.vertices.shape[0]),):
            raise ValueError("P1 nodal values must match vertex count.")
        if coordinates.shape[-1] != 3 or coordinates.shape[0] != self.faces.shape[0]:
            raise ValueError(
                "Barycentric coordinates must align with cells and have width 3."
            )
        coordinates = eqx.error_if(
            coordinates,
            jnp.any(~jnp.isfinite(coordinates))
            | jnp.any(coordinates < 0.0)
            | jnp.any(jnp.abs(jnp.sum(coordinates, axis=-1) - 1.0) > 1e-8),
            "Barycentric coordinates must be finite, non-negative, and sum to one.",
        )
        return jnp.sum(coordinates * field[self.faces], axis=-1)

    def dirichlet(
        self,
        values: ArrayLike = 0.0,
        /,
        *,
        boundary_mask: ArrayLike | None = None,
    ) -> P1DirichletElimination:
        mask = (
            np.asarray(self.boundary_vertex_mask, dtype=bool)
            if boundary_mask is None
            else np.asarray(boundary_mask, dtype=bool)
        )
        vertex_count = int(self.vertices.shape[0])
        if mask.shape != (vertex_count,) or not np.any(mask) or np.all(mask):
            raise ValueError(
                "Dirichlet mask must select a non-empty proper subset of vertices."
            )
        adjacency = [set() for _ in range(vertex_count)]
        for face in np.asarray(self.faces, dtype=np.int32):
            for left, right in (
                (int(face[0]), int(face[1])),
                (int(face[1]), int(face[2])),
                (int(face[2]), int(face[0])),
            ):
                adjacency[left].add(right)
                adjacency[right].add(left)
        remaining = set(range(vertex_count))
        while remaining:
            root = remaining.pop()
            component = {root}
            frontier = [root]
            while frontier:
                vertex = frontier.pop()
                for neighbor in adjacency[vertex]:
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        frontier.append(neighbor)
            component_indices = np.fromiter(component, dtype=np.int32)
            if not np.any(mask[component_indices]):
                raise ValueError(
                    "Every connected mesh component requires a Dirichlet vertex."
                )
        supplied = jnp.asarray(values)
        lift = (
            jnp.full((vertex_count,), supplied, dtype=self.vertices.dtype)
            if supplied.shape == ()
            else supplied
        )
        if lift.shape != (vertex_count,):
            raise ValueError("Dirichlet values must be scalar or one value per vertex.")
        lift = jnp.where(jnp.asarray(mask), lift, jnp.zeros_like(lift))
        free = np.flatnonzero(~mask).astype(np.int32)
        global_to_free = np.full((vertex_count,), -1, dtype=np.int32)
        global_to_free[free] = np.arange(free.size, dtype=np.int32)
        relation = self.stiffness.relation
        if not isinstance(relation, EdgeRelation):
            raise RuntimeError("P1 stiffness unexpectedly lost its edge relation.")
        source = np.asarray(relation.source_indices)
        target = np.asarray(relation.target_indices)
        valid = np.asarray(relation.valid, dtype=bool)
        keep = valid & ~mask[source] & ~mask[target]
        reduced_relation = EdgeRelation(
            global_to_free[source[keep]],
            global_to_free[target[keep]],
            source_size=free.size,
            target_size=free.size,
        )
        reduced = SparseLinearMap(
            reduced_relation,
            self.stiffness.coefficients[keep],
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                positive_semidefinite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                    "positive_semidefinite": "transformed",
                },
            ),
            operator_id=canonical_fingerprint(
                {
                    "kind": "p1-dirichlet-stiffness",
                    "stiffness": self.stiffness.operator_id,
                    "mask": array_tree_fingerprint(mask),
                }
            ),
        )
        return P1DirichletElimination(
            boundary_mask=jnp.asarray(mask),
            free_indices=jnp.asarray(free),
            lift=lift,
            stiffness=self.stiffness,
            reduced_stiffness=reduced,
        )

    def solve_poisson(
        self,
        right_hand_side: ArrayLike,
        /,
        *,
        dirichlet_values: ArrayLike = 0.0,
        boundary_mask: ArrayLike | None = None,
        policy: LinearSolvePolicy | None = None,
    ) -> tuple[Array, LinearSolveResult]:
        elimination = self.dirichlet(
            dirichlet_values,
            boundary_mask=boundary_mask,
        )
        reduced_rhs = elimination.reduce_rhs(right_hand_side)
        result = solve(
            LinearSystem(elimination.reduced_stiffness),
            reduced_rhs,
            policy=policy,
        )
        return elimination.expand(result.value), result

    def heat_dynamics(
        self,
        diffusivity: ArrayLike,
        /,
        *,
        source: Callable[[Array, Array, Any], ArrayLike] | None = None,
        mass_policy: LinearSolvePolicy | None = None,
    ) -> "P1HeatDynamics":
        return P1HeatDynamics(
            self,
            diffusivity,
            source=source,
            mass_policy=mass_policy,
        )


class P1HeatDynamics(StrictModule):
    """Mass-matrix-consistent semidiscrete heat equation."""

    discretization: P1FiniteElementDiscretization
    diffusivity: Array
    source: Callable[[Array, Array, Any], ArrayLike] | None = eqx.field(static=True)
    mass_solve: PreparedLinearSolve

    def __init__(
        self,
        discretization: P1FiniteElementDiscretization,
        diffusivity: ArrayLike,
        /,
        *,
        source: Callable[[Array, Array, Any], ArrayLike] | None = None,
        mass_policy: LinearSolvePolicy | None = None,
    ):
        if not isinstance(discretization, P1FiniteElementDiscretization):
            raise TypeError("discretization must be a P1FiniteElementDiscretization.")
        coefficient = jnp.asarray(diffusivity)
        if coefficient.shape != ():
            raise ValueError("diffusivity must be scalar.")
        coefficient = eqx.error_if(
            coefficient,
            ~jnp.isfinite(coefficient) | (coefficient < 0.0),
            "diffusivity must be finite and non-negative.",
        )
        if source is not None and not callable(source):
            raise TypeError("source must be callable or None.")
        self.discretization = discretization
        self.diffusivity = coefficient
        self.source = source
        self.mass_solve = prepare(
            LinearSystem(discretization.mass),
            mass_policy,
        )

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        value = jnp.asarray(state)
        expected = (int(self.discretization.vertices.shape[0]),)
        if value.shape != expected:
            raise ValueError(f"Heat state must have shape {expected}.")
        load = (
            jnp.zeros_like(value)
            if self.source is None
            else jnp.asarray(self.source(jnp.asarray(time), value, args))
        )
        if load.shape != value.shape:
            raise ValueError("Heat source must match the nodal state shape.")
        right_hand_side = load - self.diffusivity * self.discretization.stiffness.mv(
            value
        )
        return solve(self.mass_solve, right_hand_side).value


__all__ = [
    "P1DirichletElimination",
    "P1FiniteElementDiscretization",
    "P1FiniteElementPlan",
    "P1HeatDynamics",
    "p1_local_matrices",
]
