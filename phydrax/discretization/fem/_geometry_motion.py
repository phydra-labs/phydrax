#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntFlag
from itertools import product
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import (
    ArraySpace,
    ConjugateGradient,
    DifferentiationPolicy,
    FailurePolicy,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    prepare,
    PreparedLinearSolve,
    RHSLayout,
    solve,
    TolerancePolicy,
)
from ._generic import (
    FiniteElementDiscretization,
    FiniteElementRuntimeData,
)


class FiniteElementMeshMotionStatus(IntFlag):
    """Runtime failures of fixed-topology finite-element mesh motion."""

    SUCCESS = 0
    BOUNDARY_REJECTED = 1
    EXTENSION_FAILED = 2
    NONFINITE_COORDINATES = 4
    EXCESSIVE_DISPLACEMENT = 8
    JACOBIAN_TOO_SMALL = 16
    ORIENTATION_CHANGED = 32


@dataclass(frozen=True, slots=True)
class FiniteElementMeshMotionPolicy:
    """Static acceptance and solve policy for fixed-topology mesh motion."""

    minimum_absolute_jacobian: float = 1.0e-10
    minimum_relative_jacobian: float = 0.05
    maximum_displacement_fraction: float = 0.5
    solve_relative_tolerance: float = 1.0e-10
    solve_absolute_tolerance: float = 1.0e-12
    maximum_solve_steps: int = 500

    def __post_init__(self):
        for name, value in (
            ("minimum_absolute_jacobian", self.minimum_absolute_jacobian),
            ("minimum_relative_jacobian", self.minimum_relative_jacobian),
            ("maximum_displacement_fraction", self.maximum_displacement_fraction),
            ("solve_relative_tolerance", self.solve_relative_tolerance),
            ("solve_absolute_tolerance", self.solve_absolute_tolerance),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.maximum_solve_steps <= 0:
            raise ValueError("maximum_solve_steps must be positive.")


_DEFAULT_MESH_MOTION_POLICY = FiniteElementMeshMotionPolicy()


@runtime_checkable
class FiniteElementBoundaryProvider(Protocol):
    """Structural provider of fixed-route boundary coordinates."""

    @property
    def mapping_id(self) -> str: ...

    @property
    def reference_points(self) -> Array: ...

    def realize(self, design: Any, /) -> Any: ...


class FiniteElementBoundaryRealization(StrictModule):
    """Normalized boundary coordinates and provider acceptance evidence."""

    proposed_points: Array
    points: Array
    accepted: Array
    refresh_required: Array
    status: Array
    mapping_id: str = eqx.field(static=True)

    def __init__(
        self,
        proposed_points: Any,
        points: Any,
        /,
        *,
        accepted: Any,
        refresh_required: Any,
        status: Any,
        mapping_id: str,
    ):
        proposed = jnp.asarray(proposed_points, dtype=float)
        safe = jnp.asarray(points, dtype=proposed.dtype)
        if proposed.ndim != 2 or safe.shape != proposed.shape:
            raise ValueError(
                "Boundary coordinates must have matching shape (points, dim)."
            )
        if not mapping_id:
            raise ValueError("mapping_id must be non-empty.")
        self.proposed_points = proposed
        self.points = safe
        self.accepted = jnp.asarray(accepted, dtype=bool).reshape(())
        self.refresh_required = jnp.asarray(refresh_required, dtype=bool).reshape(())
        self.status = jnp.asarray(status, dtype=jnp.int32).reshape(())
        self.mapping_id = str(mapping_id)


class FiniteElementGeometryEvidence(StrictModule):
    """Signed-Jacobian and displacement evidence for a coordinate proposal."""

    finite: Array
    orientation_preserved: Array
    minimum_absolute_jacobian: Array
    minimum_relative_jacobian: Array
    maximum_displacement_ratio: Array

    def __init__(
        self,
        *,
        finite: Any,
        orientation_preserved: Any,
        minimum_absolute_jacobian: Any,
        minimum_relative_jacobian: Any,
        maximum_displacement_ratio: Any,
    ):
        self.finite = jnp.asarray(finite, dtype=bool).reshape(())
        self.orientation_preserved = jnp.asarray(
            orientation_preserved, dtype=bool
        ).reshape(())
        self.minimum_absolute_jacobian = jnp.asarray(
            minimum_absolute_jacobian, dtype=float
        ).reshape(())
        self.minimum_relative_jacobian = jnp.asarray(
            minimum_relative_jacobian, dtype=float
        ).reshape(())
        self.maximum_displacement_ratio = jnp.asarray(
            maximum_displacement_ratio, dtype=float
        ).reshape(())


class FiniteElementMeshMotionEvidence(StrictModule):
    """Complete acceptance evidence for one mesh-motion proposal."""

    boundary: FiniteElementBoundaryRealization
    geometry: FiniteElementGeometryEvidence
    extension_status: Array
    status: Array
    plan_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        boundary: FiniteElementBoundaryRealization,
        geometry: FiniteElementGeometryEvidence,
        extension_status: Any,
        status: Any,
        plan_id: str,
        topology_id: str,
        geometry_layout_id: str,
    ):
        self.boundary = boundary
        self.geometry = geometry
        self.extension_status = jnp.asarray(extension_status, dtype=jnp.int32)
        self.status = jnp.asarray(status, dtype=jnp.int32).reshape(())
        self.plan_id = str(plan_id)
        self.topology_id = str(topology_id)
        self.geometry_layout_id = str(geometry_layout_id)

    @property
    def accepted(self) -> Array:
        return self.status == int(FiniteElementMeshMotionStatus.SUCCESS)

    @property
    def refresh_required(self) -> Array:
        return self.boundary.refresh_required


class FiniteElementMeshRealization(StrictModule):
    """Proposed and safe FE coordinates plus the safe execution runtime."""

    proposed_coordinates: Array
    coordinates: Array
    runtime: FiniteElementRuntimeData
    evidence: FiniteElementMeshMotionEvidence

    def __init__(
        self,
        proposed_coordinates: Any,
        coordinates: Any,
        runtime: FiniteElementRuntimeData,
        evidence: FiniteElementMeshMotionEvidence,
        /,
    ):
        proposed = jnp.asarray(proposed_coordinates, dtype=float)
        safe = jnp.asarray(coordinates, dtype=proposed.dtype)
        if proposed.ndim != 2 or safe.shape != proposed.shape:
            raise ValueError("FE coordinates must have matching shape (points, dim).")
        if runtime.coordinates.shape != safe.shape:
            raise ValueError("FE runtime coordinates must match the realization.")
        self.proposed_coordinates = proposed
        self.coordinates = safe
        self.runtime = runtime
        self.evidence = evidence

    @property
    def accepted(self) -> Array:
        return self.evidence.accepted

    @property
    def refresh_required(self) -> Array:
        return self.evidence.refresh_required


def _cell_edges(cell_kind: str) -> tuple[tuple[int, int], ...]:
    if cell_kind == "triangle":
        return ((0, 1), (1, 2), (2, 0))
    if cell_kind == "quadrilateral":
        return ((0, 1), (1, 2), (2, 3), (3, 0))
    if cell_kind == "tetrahedron":
        return ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    if cell_kind == "hexahedron":
        return (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        )
    raise ValueError(f"Unsupported cell kind {cell_kind!r}.")


def _reference_probes(cell_kind: str) -> Array:
    if cell_kind == "triangle":
        return jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1 / 3, 1 / 3)))
    if cell_kind == "tetrahedron":
        return jnp.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.25, 0.25, 0.25),
            )
        )
    dimension = 2 if cell_kind == "quadrilateral" else 3
    return jnp.asarray(tuple(product((0.0, 0.5, 1.0), repeat=dimension)))


def _normalized_boundary(
    provider: FiniteElementBoundaryProvider,
    design: Any,
) -> FiniteElementBoundaryRealization:
    result = provider.realize(design)
    return FiniteElementBoundaryRealization(
        result.proposed_points,
        result.points,
        accepted=result.accepted,
        refresh_required=result.refresh_required,
        status=result.status,
        mapping_id=provider.mapping_id,
    )


class FiniteElementMeshMotionPlan(StrictModule):
    """Graph-harmonic, fixed-topology coordinate realization for affine FE meshes."""

    discretization: FiniteElementDiscretization
    boundary_provider: FiniteElementBoundaryProvider
    reference_coordinates: Array
    boundary_indices: Array
    interior_indices: Array
    interior_boundary_interior: Array
    interior_boundary_boundary: Array
    interior_boundary_weights: Array
    prepared_extension: PreparedLinearSolve | None
    reference_determinants: tuple[Array, ...]
    minimum_edge_length: Array
    policy: FiniteElementMeshMotionPolicy = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    mapping_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: FiniteElementDiscretization,
        boundary_provider: FiniteElementBoundaryProvider,
        /,
        *,
        policy: FiniteElementMeshMotionPolicy = _DEFAULT_MESH_MOTION_POLICY,
    ):
        if not isinstance(discretization, FiniteElementDiscretization):
            raise TypeError("discretization must be FiniteElementDiscretization.")
        if not isinstance(boundary_provider, FiniteElementBoundaryProvider):
            raise TypeError(
                "boundary_provider must satisfy FiniteElementBoundaryProvider."
            )
        if not isinstance(policy, FiniteElementMeshMotionPolicy):
            raise TypeError("policy must be FiniteElementMeshMotionPolicy.")
        mesh = discretization.mesh
        if (
            mesh.ambient_dimension != mesh.topological_dimension
            or mesh.ambient_dimension
            not in (
                2,
                3,
            )
        ):
            raise ValueError("Mesh motion requires a full-dimensional 2-D or 3-D mesh.")
        if discretization.default_runtime.coordinates.shape != mesh.coordinates.shape:
            raise ValueError(
                "Mesh motion initially supports vertex-coordinate layouts only."
            )
        for block, element, dofs in zip(
            mesh.blocks,
            discretization.coordinate_elements,
            discretization.coordinate_dofs,
            strict=True,
        ):
            if element.degree != 1 or element.local_dof_count != block.arity:
                raise ValueError(
                    "Mesh motion initially supports affine P1/Q1 coordinates."
                )
            if not np.array_equal(np.asarray(dofs), np.asarray(block.vertices)):
                raise ValueError("Coordinate DOFs must coincide with mesh vertices.")
        boundary_mask = np.asarray(
            mesh.topology.entities(0).subset("boundary").mask,
            dtype=bool,
        )
        boundary = np.flatnonzero(boundary_mask).astype(np.int32)
        interior = np.flatnonzero(~boundary_mask).astype(np.int32)
        reference = np.asarray(mesh.coordinates, dtype=float)
        provider_reference = np.asarray(boundary_provider.reference_points, dtype=float)
        if provider_reference.shape != (boundary.size, mesh.ambient_dimension):
            raise ValueError("Boundary provider routes must match every boundary vertex.")
        if not np.allclose(
            provider_reference,
            reference[boundary],
            rtol=0.0,
            atol=policy.minimum_absolute_jacobian,
        ):
            raise ValueError(
                "Boundary provider reference points do not match the FE mesh."
            )

        edge_set: set[tuple[int, int]] = set()
        for block in mesh.blocks:
            cells = np.asarray(block.vertices, dtype=np.int32)
            for cell in cells:
                for first, second in _cell_edges(block.cell_kind):
                    edge_set.add(tuple(sorted((int(cell[first]), int(cell[second])))))
        edges = np.asarray(sorted(edge_set), dtype=np.int32)
        edge_vectors = reference[edges[:, 1]] - reference[edges[:, 0]]
        edge_lengths = np.linalg.norm(edge_vectors, axis=-1)
        if np.any(~np.isfinite(edge_lengths)) or np.any(edge_lengths <= 0.0):
            raise ValueError("Mesh-motion graph edges must have positive finite length.")
        weights = 1.0 / edge_lengths
        interior_local = np.full((reference.shape[0],), -1, dtype=np.int32)
        boundary_local = np.full((reference.shape[0],), -1, dtype=np.int32)
        interior_local[interior] = np.arange(interior.size, dtype=np.int32)
        boundary_local[boundary] = np.arange(boundary.size, dtype=np.int32)
        diagonal = np.zeros((interior.size,), dtype=float)
        ii_first: list[int] = []
        ii_second: list[int] = []
        ii_weight: list[float] = []
        ib_interior: list[int] = []
        ib_boundary: list[int] = []
        ib_weight: list[float] = []
        for (first, second), weight in zip(edges, weights, strict=True):
            first_interior = interior_local[first]
            second_interior = interior_local[second]
            if first_interior >= 0:
                diagonal[first_interior] += weight
            if second_interior >= 0:
                diagonal[second_interior] += weight
            if first_interior >= 0 and second_interior >= 0:
                ii_first.append(int(first_interior))
                ii_second.append(int(second_interior))
                ii_weight.append(float(weight))
            elif first_interior >= 0:
                ib_interior.append(int(first_interior))
                ib_boundary.append(int(boundary_local[second]))
                ib_weight.append(float(weight))
            elif second_interior >= 0:
                ib_interior.append(int(second_interior))
                ib_boundary.append(int(boundary_local[first]))
                ib_weight.append(float(weight))
        if interior.size:
            adjacency = [set() for _ in range(interior.size)]
            for first, second in zip(ii_first, ii_second, strict=True):
                adjacency[first].add(second)
                adjacency[second].add(first)
            touches_boundary = set(ib_interior)
            remaining = set(range(interior.size))
            while remaining:
                start = min(remaining)
                pending = [start]
                component = {start}
                remaining.remove(start)
                while pending:
                    current = pending.pop()
                    for neighbour in adjacency[current]:
                        if neighbour in remaining:
                            remaining.remove(neighbour)
                            component.add(neighbour)
                            pending.append(neighbour)
                if component.isdisjoint(touches_boundary):
                    raise ValueError(
                        "Every interior mesh component must connect to the boundary."
                    )
            if np.any(diagonal <= 0.0):
                raise ValueError("Interior harmonic extension has a zero graph diagonal.")

        prepared_extension = None
        if interior.size:
            diagonal_array = jnp.asarray(diagonal)
            ii_first_array = jnp.asarray(ii_first, dtype=jnp.int32)
            ii_second_array = jnp.asarray(ii_second, dtype=jnp.int32)
            ii_weight_array = jnp.asarray(ii_weight, dtype=float)

            def action(value):
                result = diagonal_array * value
                result = result.at[ii_first_array].add(
                    -ii_weight_array * value[ii_second_array]
                )
                result = result.at[ii_second_array].add(
                    -ii_weight_array * value[ii_first_array]
                )
                return result

            space = ArraySpace((int(interior.size),), dtype=jnp.asarray(reference).dtype)
            operator = FunctionLinearOperator(
                action,
                source=space,
                target=space,
                transpose_action=action,
                properties=OperatorProperties(
                    self_adjoint=True,
                    positive_definite=True,
                    evidence={"positive_definite": "construction"},
                ),
                operator_id=f"{mesh.topology_id}:harmonic-extension",
            )
            solve_policy = LinearSolvePolicy(
                ConjugateGradient(),
                tolerance=TolerancePolicy(
                    relative=policy.solve_relative_tolerance,
                    absolute=policy.solve_absolute_tolerance,
                    max_steps=policy.maximum_solve_steps,
                ),
                differentiation=DifferentiationPolicy("rhs-only"),
                failure=FailurePolicy("status"),
            )
            prepared_extension = prepare(
                LinearSystem(
                    operator,
                    problem_id=f"{mesh.topology_id}:harmonic-extension-system",
                ),
                solve_policy,
                rhs_layout=RHSLayout((mesh.ambient_dimension,), names=("coordinate",)),
            )

        reference_determinants = self._determinants_for(
            discretization,
            jnp.asarray(reference),
        )
        for determinant in reference_determinants:
            determinant_host = np.asarray(determinant)
            if np.any(~np.isfinite(determinant_host)) or np.any(
                np.abs(determinant_host) <= policy.minimum_absolute_jacobian
            ):
                raise ValueError("Reference FE geometry has a singular coordinate map.")
        identifier = canonical_fingerprint(
            {
                "kind": "finite-element-mesh-motion",
                "prepared": discretization.prepared_id,
                "mapping": boundary_provider.mapping_id,
                "boundary": boundary.tolist(),
                "edges": edges.tolist(),
                "policy": repr(policy),
            }
        )
        self.discretization = discretization
        self.boundary_provider = boundary_provider
        self.reference_coordinates = jnp.asarray(reference)
        self.boundary_indices = jnp.asarray(boundary)
        self.interior_indices = jnp.asarray(interior)
        self.interior_boundary_interior = jnp.asarray(ib_interior, dtype=jnp.int32)
        self.interior_boundary_boundary = jnp.asarray(ib_boundary, dtype=jnp.int32)
        self.interior_boundary_weights = jnp.asarray(ib_weight, dtype=float)
        self.prepared_extension = prepared_extension
        self.reference_determinants = reference_determinants
        self.minimum_edge_length = jnp.asarray(np.min(edge_lengths))
        self.policy = policy
        self.plan_id = identifier
        self.mapping_id = boundary_provider.mapping_id
        self.topology_id = mesh.topology_id
        self.geometry_layout_id = discretization.default_runtime.geometry_layout_id

    @staticmethod
    def _determinants_for(
        discretization: FiniteElementDiscretization,
        coordinates: Array,
    ) -> tuple[Array, ...]:
        determinants = []
        for block, element, dofs in zip(
            discretization.mesh.blocks,
            discretization.coordinate_elements,
            discretization.coordinate_dofs,
            strict=True,
        ):
            probes = _reference_probes(block.cell_kind)
            _, gradients = element.tabulate(probes)
            cell_coordinates = coordinates[dofs]
            jacobian = contract("cla,qld->cqad", cell_coordinates, gradients)
            determinants.append(jnp.linalg.det(jacobian))
        return tuple(determinants)

    def realize(self, design: Any, /) -> FiniteElementMeshRealization:
        boundary = _normalized_boundary(self.boundary_provider, design)
        if boundary.proposed_points.shape != (
            self.boundary_indices.shape[0],
            self.reference_coordinates.shape[1],
        ):
            raise ValueError("Boundary provider changed its fixed coordinate shape.")
        boundary_displacement = (
            boundary.proposed_points - self.reference_coordinates[self.boundary_indices]
        )
        if self.interior_indices.shape[0]:
            right_hand_side = jnp.zeros(
                (
                    self.interior_indices.shape[0],
                    self.reference_coordinates.shape[1],
                ),
                dtype=self.reference_coordinates.dtype,
            )
            right_hand_side = right_hand_side.at[self.interior_boundary_interior].add(
                self.interior_boundary_weights[:, None]
                * boundary_displacement[self.interior_boundary_boundary]
            )
            extension = solve(self.prepared_extension, right_hand_side)
            interior_displacement = extension.value
            extension_success = jnp.all(extension.successful)
            extension_status = extension.status
        else:
            interior_displacement = jnp.empty(
                (0, self.reference_coordinates.shape[1]),
                dtype=self.reference_coordinates.dtype,
            )
            extension_success = jnp.asarray(True)
            extension_status = jnp.asarray(0, dtype=jnp.int32)
        displacement = jnp.zeros_like(self.reference_coordinates)
        displacement = displacement.at[self.boundary_indices].set(boundary_displacement)
        displacement = displacement.at[self.interior_indices].set(interior_displacement)
        proposed = self.reference_coordinates + displacement
        determinants = self._determinants_for(self.discretization, proposed)
        finite = jnp.all(jnp.isfinite(proposed)) & jnp.all(
            jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in determinants))
        )
        absolute_values = jnp.concatenate(
            tuple(jnp.reshape(jnp.abs(value), (-1,)) for value in determinants)
        )
        relative_values = jnp.concatenate(
            tuple(
                jnp.reshape(jnp.abs(value / reference), (-1,))
                for value, reference in zip(
                    determinants,
                    self.reference_determinants,
                    strict=True,
                )
            )
        )
        orientation_preserved = jnp.all(
            jnp.stack(
                tuple(
                    jnp.all(value * reference > 0.0)
                    for value, reference in zip(
                        determinants,
                        self.reference_determinants,
                        strict=True,
                    )
                )
            )
        )
        displacement_norm = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
        maximum_displacement_ratio = jnp.max(displacement_norm) / self.minimum_edge_length
        geometry = FiniteElementGeometryEvidence(
            finite=finite,
            orientation_preserved=orientation_preserved,
            minimum_absolute_jacobian=jnp.min(absolute_values),
            minimum_relative_jacobian=jnp.min(relative_values),
            maximum_displacement_ratio=maximum_displacement_ratio,
        )
        status = jnp.asarray(int(FiniteElementMeshMotionStatus.SUCCESS), dtype=jnp.int32)
        status = status | jnp.where(
            boundary.accepted,
            0,
            int(FiniteElementMeshMotionStatus.BOUNDARY_REJECTED),
        ).astype(jnp.int32)
        status = status | jnp.where(
            extension_success,
            0,
            int(FiniteElementMeshMotionStatus.EXTENSION_FAILED),
        ).astype(jnp.int32)
        status = status | jnp.where(
            finite,
            0,
            int(FiniteElementMeshMotionStatus.NONFINITE_COORDINATES),
        ).astype(jnp.int32)
        status = status | jnp.where(
            maximum_displacement_ratio <= self.policy.maximum_displacement_fraction,
            0,
            int(FiniteElementMeshMotionStatus.EXCESSIVE_DISPLACEMENT),
        ).astype(jnp.int32)
        status = status | jnp.where(
            geometry.minimum_absolute_jacobian >= self.policy.minimum_absolute_jacobian,
            0,
            int(FiniteElementMeshMotionStatus.JACOBIAN_TOO_SMALL),
        ).astype(jnp.int32)
        status = status | jnp.where(
            geometry.minimum_relative_jacobian >= self.policy.minimum_relative_jacobian,
            0,
            int(FiniteElementMeshMotionStatus.JACOBIAN_TOO_SMALL),
        ).astype(jnp.int32)
        status = status | jnp.where(
            orientation_preserved,
            0,
            int(FiniteElementMeshMotionStatus.ORIENTATION_CHANGED),
        ).astype(jnp.int32)
        evidence = FiniteElementMeshMotionEvidence(
            boundary=boundary,
            geometry=geometry,
            extension_status=extension_status,
            status=status,
            plan_id=self.plan_id,
            topology_id=self.topology_id,
            geometry_layout_id=self.geometry_layout_id,
        )
        safe = jnp.where(evidence.accepted, proposed, self.reference_coordinates)
        runtime = self.discretization.prepare_runtime(
            safe,
            numeric_version=self.plan_id,
        )
        return FiniteElementMeshRealization(proposed, safe, runtime, evidence)


__all__ = [
    "FiniteElementBoundaryProvider",
    "FiniteElementBoundaryRealization",
    "FiniteElementGeometryEvidence",
    "FiniteElementMeshMotionEvidence",
    "FiniteElementMeshMotionPlan",
    "FiniteElementMeshMotionPolicy",
    "FiniteElementMeshMotionStatus",
    "FiniteElementMeshRealization",
]
