#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Tetrahedral BDM-DG Stokes with symmetric tangential Nitsche coupling."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...discretization import (
    CellMesh,
    discontinuous_element,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    PressureGaugePolicy,
    tetrahedral_bdm_element,
)
from ...ein import contract
from ...linalg import AbstractVectorSpace, BlockSpace, OperatorProperties
from ...sparse import EdgeRelation, SparseLinearMap
from .._finite_element_variational import (
    CellResidualAction,
    compile_finite_element_problem,
    CompiledFiniteElementProblem,
    FiniteElementForm,
)


_REFERENCE_VERTICES = np.asarray(
    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
)
_REFERENCE_FACES = ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3))


def _triangle_quadrature(order: int = 4) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    barycentric = []
    triangle_weights = []
    for first, first_weight in zip(nodes, weights, strict=True):
        for second, second_weight in zip(nodes, weights, strict=True):
            barycentric.append(
                (
                    1.0 - first - (1.0 - first) * second,
                    first,
                    (1.0 - first) * second,
                )
            )
            triangle_weights.append(first_weight * second_weight * (1.0 - first))
    return np.asarray(barycentric), np.asarray(triangle_weights)


_TRIANGLE_BARYCENTRIC, _TRIANGLE_WEIGHTS = _triangle_quadrature()


def hdiv_stokes_form(
    velocity_field: str,
    pressure_field: str,
    viscosity: ArrayLike = 1.0,
    /,
    *,
    form_id: str = "hdiv-stokes",
) -> FiniteElementForm:
    """BDM/DG Stokes cell form; tangential Nitsche terms are prepared separately."""

    viscosity_ = jnp.asarray(viscosity)

    def momentum(values, gradients, points, weights, test_basis, test_gradients, context):
        del points, test_basis, context
        velocity_gradient, _ = gradients
        _, pressure = values
        strain = 0.5 * (velocity_gradient + jnp.swapaxes(velocity_gradient, -1, -2))
        test_strain = 0.5 * (test_gradients + jnp.swapaxes(test_gradients, -1, -2))
        viscous = (
            2.0 * viscosity_ * contract("cq,cqkab,cqab->ck", weights, test_strain, strain)
        )
        test_divergence = jnp.trace(test_gradients, axis1=-2, axis2=-1)
        pressure_term = contract("cq,cqk,cq->ck", weights, test_divergence, pressure)
        return viscous - pressure_term

    def incompressibility(
        values, gradients, points, weights, test_basis, test_gradients, context
    ):
        del values, points, test_gradients, context
        divergence = jnp.trace(gradients[0], axis1=-2, axis2=-1)
        if test_basis.ndim == 2:
            return contract("cq,qk,cq->ck", weights, test_basis, divergence)
        return contract("cq,cqk,cq->ck", weights, test_basis, divergence)

    return FiniteElementForm(
        form_id,
        (velocity_field, pressure_field),
        (
            CellResidualAction(
                velocity_field,
                (velocity_field, pressure_field),
                momentum,
                action_id="hdiv-stokes-momentum",
            ),
            CellResidualAction(
                pressure_field,
                (velocity_field,),
                incompressibility,
                action_id="hdiv-stokes-incompressibility",
            ),
        ),
        properties=OperatorProperties(
            self_adjoint=True, evidence={"self_adjoint": "construction"}
        ),
    )


class HDivNormalBoundaryCondition(StrictModule):
    """Persistent-face normal-flow and hydraulic-resistance data."""

    face_global_ids: Array
    resistance: Array
    prescribed_flux: Array | None
    condition_id: str = eqx.field(static=True)

    def __init__(
        self,
        face_global_ids: ArrayLike,
        /,
        *,
        resistance: ArrayLike = 0.0,
        prescribed_flux: ArrayLike | None = None,
    ):
        identifiers = np.asarray(face_global_ids)
        if identifiers.ndim != 1 or not np.issubdtype(identifiers.dtype, np.integer):
            raise TypeError("face_global_ids must be one rank-1 integer array.")
        identifiers = identifiers.astype(np.int64, copy=False)
        if identifiers.size == 0:
            raise ValueError("A normal boundary condition requires at least one face.")
        if np.any(identifiers < 0) or np.unique(identifiers).size != identifiers.size:
            raise ValueError("face_global_ids must be unique non-negative integers.")
        order = np.argsort(identifiers, kind="stable")
        identifiers = identifiers[order]

        resistance_ = np.asarray(resistance)
        if resistance_.ndim == 0:
            resistance_ = np.broadcast_to(resistance_, identifiers.shape)
        if resistance_.shape != identifiers.shape:
            raise ValueError(
                f"resistance must be scalar or have shape {identifiers.shape}."
            )
        resistance_ = np.asarray(resistance_)[order]
        if np.any(~np.isfinite(resistance_)) or np.any(resistance_ < 0.0):
            raise ValueError("resistance must be finite and non-negative.")

        flux_ = None
        if prescribed_flux is not None:
            flux_ = np.asarray(prescribed_flux)
            if flux_.ndim == 0:
                flux_ = np.broadcast_to(flux_, identifiers.shape)
            if flux_.shape != identifiers.shape:
                raise ValueError(
                    f"prescribed_flux must be scalar or have shape {identifiers.shape}."
                )
            flux_ = np.asarray(flux_)[order]
            if np.any(~np.isfinite(flux_)):
                raise ValueError("prescribed_flux must be finite.")

        self.face_global_ids = jnp.asarray(identifiers)
        self.resistance = jnp.asarray(resistance_)
        self.prescribed_flux = None if flux_ is None else jnp.asarray(flux_)
        self.condition_id = canonical_fingerprint(
            {
                "kind": "hdiv-normal-boundary-condition",
                "face_global_ids": array_tree_fingerprint(identifiers),
                "resistance": array_tree_fingerprint(resistance_),
                "prescribed_flux": (
                    None if flux_ is None else array_tree_fingerprint(flux_)
                ),
            }
        )


class HDivStokesEvidence(StrictModule):
    hdiv_conforming: Array
    discontinuous_pressure: Array
    pressure_gauge_explicit: Array
    tangential_nitsche_symmetric: Array
    finite_viscosity: Array
    finite_penalty: Array
    normal_boundary_faces_valid: Array
    normal_resistance_nonnegative: Array
    normal_constraints_finite: Array
    successful: Array


class PreparedHDivStokes(StrictModule):
    problem: CompiledFiniteElementProblem
    gauge: PressureGaugePolicy
    viscosity: Array
    tangential_operator: SparseLinearMap
    normal_resistance_operator: SparseLinearMap
    normal_flux_operator: SparseLinearMap
    normal_flux_face_ids: Array
    normal_flux_target: Array
    state_space: AbstractVectorSpace
    evidence: HDivStokesEvidence
    prepared_id: str = eqx.field(static=True)

    def gauge_pressure(self, pressure: ArrayLike, /) -> Array:
        values = jnp.asarray(pressure)
        if self.gauge.mode == "mean-zero":
            if self.gauge.weights is None:
                return values - jnp.mean(values)
            weights = self.gauge.weights / jnp.sum(self.gauge.weights)
            return values - jnp.vdot(weights, values)
        if self.gauge.mode == "pinned":
            if self.gauge.pinned_dof is None:
                raise RuntimeError("Pinned pressure gauge has no pinned DOF.")
            return values - values[self.gauge.pinned_dof]
        raise ValueError("H(div) Stokes requires mean-zero or pinned pressure gauge.")

    def normal_flux(self, velocity: ArrayLike, /) -> Array:
        return self.normal_flux_operator.mv(jnp.asarray(velocity))

    def normal_flux_residual(self, velocity: ArrayLike, /) -> Array:
        return self.normal_flux(velocity) - self.normal_flux_target

    def _fluid_residual(
        self, velocity: Array, pressure: ArrayLike, /
    ) -> tuple[Array, Array]:
        gauged_pressure = self.gauge_pressure(pressure)
        cell_velocity, incompressibility = self.problem.residual(
            (velocity, gauged_pressure)
        )
        return (
            cell_velocity
            + self.tangential_operator.mv(velocity)
            + self.normal_resistance_operator.mv(velocity),
            incompressibility,
        )

    def constrained_residual(
        self,
        state: tuple[ArrayLike, ArrayLike, ArrayLike],
        /,
    ) -> tuple[Array, Array, Array]:
        if not isinstance(state, tuple) or len(state) != 3:
            raise TypeError(
                "Constrained H(div) Stokes state must be "
                "(velocity, pressure, normal_flux_multiplier)."
            )
        velocity = jnp.asarray(state[0])
        multiplier = jnp.asarray(state[2])
        if multiplier.shape != self.normal_flux_target.shape:
            raise ValueError(
                "normal_flux_multiplier must match the prescribed normal-flow count."
            )
        momentum, incompressibility = self._fluid_residual(velocity, state[1])
        return (
            momentum + self.normal_flux_operator.adjoint_mv(multiplier),
            incompressibility,
            self.normal_flux_residual(velocity),
        )

    def residual(
        self,
        state: tuple[ArrayLike, ...],
        /,
    ) -> tuple[Array, ...]:
        if int(self.normal_flux_target.size) > 0:
            if not isinstance(state, tuple) or len(state) != 3:
                raise TypeError(
                    "Prescribed normal flow requires state "
                    "(velocity, pressure, normal_flux_multiplier)."
                )
            return self.constrained_residual(state)
        if not isinstance(state, tuple) or len(state) != 2:
            raise TypeError("H(div) Stokes state must be (velocity, pressure).")
        return self._fluid_residual(jnp.asarray(state[0]), state[1])


def _physical_basis(
    element,
    cell_points: np.ndarray,
    local_face: int,
    orientation: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    face = _REFERENCE_VERTICES[np.asarray(_REFERENCE_FACES[local_face])]
    reference_points = _TRIANGLE_BARYCENTRIC @ face
    reference_values, reference_gradients = element.tabulate(reference_points)
    jacobian = (cell_points[1:] - cell_points[0]).T
    determinant = float(np.linalg.det(jacobian))
    if determinant <= 0.0:
        raise ValueError("H(div) Stokes requires positively oriented tetrahedra.")
    inverse = np.linalg.inv(jacobian)
    values = contract("ab,qkb->qka", jacobian, np.asarray(reference_values)) / determinant
    gradients = (
        contract(
            "ab,qkbd,dc->qkac",
            jacobian,
            np.asarray(reference_gradients),
            inverse,
        )
        / determinant
    )
    values *= orientation[None, :, None]
    gradients *= orientation[None, :, None, None]
    return values, gradients


def _tangential_nitsche_entries(
    discretization,
    viscosity: float,
    penalty: float,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = discretization.mesh
    if len(mesh.blocks) != 1 or mesh.blocks[0].cell_kind != "tetrahedron":
        raise ValueError("H(div) Nitsche preparation requires one tetrahedral block.")
    block = mesh.blocks[0]
    connectivity = mesh.connectivity
    cells = np.asarray(block.vertices, dtype=np.int32)
    coordinates = np.asarray(mesh.coordinates)
    face_vertices = np.asarray(connectivity.faces, dtype=np.int32)
    face_points = coordinates[face_vertices]
    cross = np.cross(
        face_points[:, 1] - face_points[:, 0],
        face_points[:, 2] - face_points[:, 0],
    )
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    if np.any(areas <= 0.0):
        raise ValueError("H(div) Nitsche faces must have positive area.")
    normals = cross / (2.0 * areas[:, None])
    centers = np.mean(coordinates[cells], axis=1)
    volumes = (
        np.abs(np.linalg.det(coordinates[cells[:, 1:]] - coordinates[cells[:, :1]])) / 6.0
    )
    incidents: list[list[tuple[int, int]]] = [[] for _ in face_vertices]
    face_by_vertices = {
        tuple(int(vertex) for vertex in vertices): face
        for face, vertices in enumerate(face_vertices)
    }
    for cell, vertices in enumerate(cells):
        for local_face, reference_face in enumerate(_REFERENCE_FACES):
            key = tuple(sorted(int(vertices[local]) for local in reference_face))
            incidents[face_by_vertices[key]].append((cell, local_face))
    dof_map = discretization.dof_maps[0]
    element = discretization.elements[0][0]
    source_indices = []
    target_indices = []
    coefficients = []
    identity = np.eye(3)
    for face, adjacent in enumerate(incidents):
        owner, owner_local = adjacent[0]
        normal = normals[face]
        midpoint = np.mean(face_points[face], axis=0)
        if np.dot(normal, midpoint - centers[owner]) < 0.0:
            normal = -normal
        projector = identity - np.outer(normal, normal)
        owner_routes = np.asarray(dof_map.cell_dofs[0][owner], dtype=np.int32)
        owner_orientation = np.asarray(dof_map.orientations[0][owner])
        owner_values, owner_gradients = _physical_basis(
            element,
            coordinates[cells[owner]],
            owner_local,
            owner_orientation,
        )
        owner_jump = contract("ab,qkb->qka", projector, owner_values)
        owner_strain = 0.5 * (owner_gradients + np.swapaxes(owner_gradients, -1, -2))
        owner_stress = (
            2.0 * viscosity * contract("ab,qkbc,c->qka", projector, owner_strain, normal)
        )
        height = 3.0 * volumes[owner] / areas[face]
        if len(adjacent) == 2:
            neighbour, neighbour_local = adjacent[1]
            neighbour_routes = np.asarray(dof_map.cell_dofs[0][neighbour], dtype=np.int32)
            neighbour_orientation = np.asarray(dof_map.orientations[0][neighbour])
            neighbour_values, neighbour_gradients = _physical_basis(
                element,
                coordinates[cells[neighbour]],
                neighbour_local,
                neighbour_orientation,
            )
            neighbour_jump = -contract("ab,qkb->qka", projector, neighbour_values)
            neighbour_strain = 0.5 * (
                neighbour_gradients + np.swapaxes(neighbour_gradients, -1, -2)
            )
            neighbour_stress = viscosity * contract(
                "ab,qkbc,c->qka", projector, neighbour_strain, normal
            )
            stress = np.concatenate((0.5 * owner_stress, neighbour_stress), axis=1)
            jump = np.concatenate((owner_jump, neighbour_jump), axis=1)
            routes = np.concatenate((owner_routes, neighbour_routes))
            height = min(height, 3.0 * volumes[neighbour] / areas[face])
        else:
            stress = owner_stress
            jump = owner_jump
            routes = owner_routes
        local = np.zeros((len(routes), len(routes)))
        for point, reference_weight in enumerate(_TRIANGLE_WEIGHTS):
            jump_gram = jump[point] @ jump[point].T
            stress_jump = stress[point] @ jump[point].T
            local += (
                2.0
                * areas[face]
                * reference_weight
                * (
                    -stress_jump
                    - stress_jump.T
                    + penalty * viscosity / height * jump_gram
                )
            )
        symmetry = np.max(np.abs(local - local.T), initial=0.0)
        tolerance = (
            1024.0
            * np.finfo(local.dtype).eps
            * max(1.0, float(np.max(np.abs(local), initial=0.0)))
        )
        if symmetry > tolerance:
            raise RuntimeError("Tangential Nitsche local operator lost symmetry.")
        target_indices.append(np.repeat(routes, len(routes)))
        source_indices.append(np.tile(routes, len(routes)))
        coefficients.append(local.reshape((-1,)))
    return (
        np.concatenate(tuple(source_indices)).astype(np.int32),
        np.concatenate(tuple(target_indices)).astype(np.int32),
        np.concatenate(tuple(coefficients)),
    )


def _normal_boundary_operators(
    discretization,
    boundaries: tuple[HDivNormalBoundaryCondition, ...],
    operator_prefix: str,
    /,
) -> tuple[SparseLinearMap, SparseLinearMap, Array, Array]:
    mesh = discretization.mesh
    connectivity = mesh.connectivity
    face_entities = mesh.topology.entity_sets[2]
    face_ids = np.asarray(face_entities.entity_ids, dtype=np.int64)
    boundary_faces = np.asarray(connectivity.boundary_faces, dtype=bool)
    if face_ids.shape != boundary_faces.shape:
        raise RuntimeError("Tetrahedral face identity and connectivity disagree.")
    face_by_id = {int(identifier): index for index, identifier in enumerate(face_ids)}

    records = []
    for condition in boundaries:
        for local, identifier in enumerate(
            np.asarray(condition.face_global_ids, dtype=np.int64)
        ):
            face = face_by_id.get(int(identifier))
            if face is None:
                raise ValueError(
                    f"Unknown normal-boundary face global ID {int(identifier)}."
                )
            if not boundary_faces[face]:
                raise ValueError(
                    f"Normal-boundary face global ID {int(identifier)} is not exterior."
                )
            records.append((int(identifier), face, condition, local))
    records.sort(key=lambda record: record[0])
    record_ids = [record[0] for record in records]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("Normal-boundary face selections must not overlap.")

    velocity_map = discretization.dof_maps[0]
    face_width = velocity_map.entity_dofs_per_entity[2]
    if face_width != 6:
        raise RuntimeError(
            "Normal-flow constraints require tetrahedral BDM2 face moments."
        )
    velocity_size = velocity_map.global_dof_count
    outward_sign_by_face = {}
    for _, face, _, _ in records:
        if face in outward_sign_by_face:
            continue
        start = face * face_width
        stop = start + face_width
        selected_signs = []
        for routes, orientations in zip(
            velocity_map.cell_dofs,
            velocity_map.orientations,
            strict=True,
        ):
            routes_ = np.asarray(routes)
            orientations_ = np.asarray(orientations)
            selected_signs.extend(
                orientations_[(routes_ >= start) & (routes_ < stop)].tolist()
            )
        signs = np.asarray(selected_signs)
        if signs.shape != (face_width,) or np.unique(signs).size != 1:
            raise RuntimeError("Boundary-face H(div) orientation is inconsistent.")
        outward_sign_by_face[face] = float(signs[0])

    resistance_sources = []
    resistance_targets = []
    resistance_coefficients = []
    for _, face, condition, local in records:
        dofs = face * face_width + np.arange(face_width, dtype=np.int32)
        resistance_sources.append(np.tile(dofs, face_width))
        resistance_targets.append(np.repeat(dofs, face_width))
        resistance_coefficients.append(
            jnp.broadcast_to(condition.resistance[local], (face_width * face_width,))
        )
    resistance_source = (
        np.concatenate(tuple(resistance_sources))
        if resistance_sources
        else np.empty((0,), dtype=np.int32)
    )
    resistance_target = (
        np.concatenate(tuple(resistance_targets))
        if resistance_targets
        else np.empty((0,), dtype=np.int32)
    )
    resistance_values = (
        jnp.concatenate(tuple(resistance_coefficients))
        if resistance_coefficients
        else jnp.empty((0,), dtype=jnp.asarray(0.0).dtype)
    )
    resistance_relation = EdgeRelation(
        resistance_source,
        resistance_target,
        source_size=velocity_size,
        target_size=velocity_size,
    )
    resistance_operator = SparseLinearMap(
        resistance_relation,
        resistance_values,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        ),
        operator_id=f"{operator_prefix}:normal-resistance",
    )

    constrained = []
    constrained_targets_ = []
    for record in records:
        prescribed = record[2].prescribed_flux
        if prescribed is not None:
            constrained.append(record)
            constrained_targets_.append(prescribed[record[3]])
    flux_source = (
        np.concatenate(
            tuple(
                face * face_width + np.arange(face_width, dtype=np.int32)
                for _, face, _, _ in constrained
            )
        )
        if constrained
        else np.empty((0,), dtype=np.int32)
    )
    flux_target = np.repeat(np.arange(len(constrained), dtype=np.int32), face_width)
    flux_relation = EdgeRelation(
        flux_source,
        flux_target,
        source_size=velocity_size,
        target_size=len(constrained),
    )
    flux_coefficients = (
        jnp.asarray(
            np.concatenate(
                tuple(
                    np.full((face_width,), outward_sign_by_face[face])
                    for _, face, _, _ in constrained
                )
            ),
            dtype=resistance_values.dtype,
        )
        if constrained
        else jnp.empty((0,), dtype=resistance_values.dtype)
    )
    flux_operator = SparseLinearMap(
        flux_relation,
        flux_coefficients,
        operator_id=f"{operator_prefix}:normal-flux",
    )
    constrained_face_ids = jnp.asarray(
        [identifier for identifier, _, _, _ in constrained],
        dtype=face_entities.entity_ids.dtype,
    )
    constrained_targets = (
        jnp.stack(tuple(constrained_targets_))
        if constrained_targets_
        else jnp.empty((0,), dtype=resistance_values.dtype)
    )
    return (
        resistance_operator,
        flux_operator,
        constrained_face_ids,
        constrained_targets,
    )


class HDivStokesPlan(StrictModule):
    mesh: CellMesh
    gauge: PressureGaugePolicy
    viscosity: Array
    normal_boundaries: tuple[HDivNormalBoundaryCondition, ...]
    penalty: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        gauge: PressureGaugePolicy,
        viscosity: ArrayLike = 1.0,
        /,
        *,
        penalty: float = 20.0,
        normal_boundaries: Sequence[HDivNormalBoundaryCondition] = (),
    ):
        if not isinstance(mesh, CellMesh) or any(
            block.cell_kind != "tetrahedron" for block in mesh.blocks
        ):
            raise TypeError("H(div) Stokes requires a tetrahedral CellMesh.")
        if not isinstance(gauge, PressureGaugePolicy) or gauge.mode == "none":
            raise ValueError("H(div) Stokes requires an explicit pressure gauge.")
        viscosity_ = jnp.asarray(viscosity)
        viscosity_ = eqx.error_if(
            viscosity_,
            jnp.any(~jnp.isfinite(viscosity_)) | jnp.any(viscosity_ <= 0.0),
            "viscosity must be finite and positive.",
        )
        penalty_ = float(penalty)
        if not np.isfinite(penalty_) or penalty_ <= 0.0:
            raise ValueError("penalty must be finite and positive.")
        boundaries = tuple(normal_boundaries)
        if not all(
            isinstance(boundary, HDivNormalBoundaryCondition) for boundary in boundaries
        ):
            raise TypeError(
                "normal_boundaries must contain HDivNormalBoundaryCondition values."
            )
        self.mesh = mesh
        self.gauge = gauge
        self.viscosity = viscosity_
        self.normal_boundaries = boundaries
        self.penalty = penalty_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "hdiv-stokes-plan",
                "mesh": mesh.mesh_id,
                "gauge": gauge.gauge_id,
                "velocity_element": tetrahedral_bdm_element(2).element_id,
                "pressure_element": discontinuous_element("tetrahedron", 1).element_id,
                "viscosity": array_tree_fingerprint(np.asarray(viscosity_)),
                "penalty": penalty_.hex(),
                "normal_boundaries": [boundary.condition_id for boundary in boundaries],
            }
        )

    def prepare(self) -> PreparedHDivStokes:
        velocity = tetrahedral_bdm_element(2)
        pressure = discontinuous_element("tetrahedron", 1)
        discretization = FiniteElementPlan(
            self.mesh,
            (
                FiniteElementFieldSpec("velocity", velocity),
                FiniteElementFieldSpec("pressure", pressure),
            ),
        ).prepare()
        problem = compile_finite_element_problem(
            hdiv_stokes_form(
                "velocity",
                "pressure",
                self.viscosity,
                form_id=f"{self.plan_id}:stokes",
            ),
            discretization,
        )
        viscosity = float(np.asarray(self.viscosity))
        source_indices, target_indices, coefficients = _tangential_nitsche_entries(
            discretization, viscosity, self.penalty
        )
        velocity_space = discretization.field_spaces[0].vector_space
        relation = EdgeRelation(
            source_indices,
            target_indices,
            source_size=velocity_space.size,
            target_size=velocity_space.size,
        )
        tangential = SparseLinearMap(
            relation,
            jnp.asarray(coefficients),
            properties=OperatorProperties(
                self_adjoint=True,
                evidence={"self_adjoint": "construction"},
            ),
            operator_id=f"{self.plan_id}:tangential-nitsche",
        )
        (
            normal_resistance,
            normal_flux,
            normal_flux_face_ids,
            normal_flux_target,
        ) = _normal_boundary_operators(
            discretization,
            self.normal_boundaries,
            self.plan_id,
        )
        normal_resistance_relation = normal_resistance.relation
        if not isinstance(normal_resistance_relation, EdgeRelation):
            raise RuntimeError("Normal resistance lost its edge-list relation.")
        state_space = (
            BlockSpace(
                (*problem.state_space.spaces, normal_flux.target),
                names=("velocity", "pressure", "normal_flux_multiplier"),
            )
            if int(normal_flux_target.size) > 0
            else problem.state_space
        )
        probe = jnp.arange(velocity_space.size, dtype=coefficients.dtype) + 1.0
        finite = (
            jnp.all(jnp.isfinite(self.viscosity))
            & jnp.all(jnp.isfinite(tangential.coefficients))
            & jnp.all(jnp.isfinite(normal_resistance.coefficients))
            & jnp.all(jnp.isfinite(normal_flux_target))
        )
        symmetry = jnp.allclose(tangential.mv(probe), tangential.adjoint_mv(probe))
        resistance_nonnegative = jnp.all(normal_resistance.coefficients >= 0.0)
        constraints_finite = jnp.all(jnp.isfinite(normal_flux_target))
        evidence = HDivStokesEvidence(
            jnp.asarray(velocity.conformity == "Hdiv"),
            jnp.asarray(pressure.conformity == "L2"),
            jnp.asarray(self.gauge.mode != "none"),
            symmetry,
            jnp.all(jnp.isfinite(self.viscosity)),
            jnp.asarray(np.isfinite(self.penalty) and self.penalty > 0.0),
            jnp.asarray(True),
            resistance_nonnegative,
            constraints_finite,
            finite & symmetry & resistance_nonnegative & constraints_finite,
        )
        return PreparedHDivStokes(
            problem,
            self.gauge,
            self.viscosity,
            tangential,
            normal_resistance,
            normal_flux,
            normal_flux_face_ids,
            normal_flux_target,
            state_space,
            evidence,
            canonical_fingerprint(
                {
                    "kind": "prepared-hdiv-stokes",
                    "plan": self.plan_id,
                    "problem": problem.compilation_id,
                    "tangential": array_tree_fingerprint(
                        {
                            "source": source_indices,
                            "target": target_indices,
                            "coefficients": coefficients,
                        }
                    ),
                    "normal_resistance": array_tree_fingerprint(
                        {
                            "source": np.asarray(
                                normal_resistance_relation.source_indices
                            ),
                            "target": np.asarray(
                                normal_resistance_relation.target_indices
                            ),
                            "coefficients": np.asarray(normal_resistance.coefficients),
                        }
                    ),
                    "normal_flux": array_tree_fingerprint(
                        {
                            "face_ids": np.asarray(normal_flux_face_ids),
                            "target": np.asarray(normal_flux_target),
                        }
                    ),
                }
            ),
        )


__all__ = [
    "HDivNormalBoundaryCondition",
    "HDivStokesEvidence",
    "HDivStokesPlan",
    "PreparedHDivStokes",
    "hdiv_stokes_form",
]
