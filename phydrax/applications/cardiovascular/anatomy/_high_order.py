#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization import CellGeometrySpec, CellMesh
from ....discretization.fem import FiniteElementSpec
from ._roles import CardiacBoundaryProfile


class HighOrderGeometryEpoch(StrictModule, NonTrainableState):
    """Geometry and reference-configuration epochs for a curved cardiac mesh."""

    geometry: Array
    reference: Array

    def __init__(self, geometry: int | ArrayLike, reference: int | ArrayLike, /):
        geometry_host = np.asarray(geometry)
        reference_host = np.asarray(reference)
        if geometry_host.shape != () or reference_host.shape != ():
            raise ValueError("High-order geometry epochs must be scalar values.")
        if not np.issubdtype(geometry_host.dtype, np.integer) or not np.issubdtype(
            reference_host.dtype, np.integer
        ):
            raise TypeError("High-order geometry epochs must be integers.")
        if int(geometry_host) < 0 or int(reference_host) < 0:
            raise ValueError("High-order geometry epochs must be non-negative.")
        self.geometry = jnp.asarray(geometry_host, dtype=jnp.int32)
        self.reference = jnp.asarray(reference_host, dtype=jnp.int32)


class HighOrderCardiacGeometryEvidence(StrictModule, NonTrainableState):
    """Qualification and lifecycle evidence for one curved geometry candidate."""

    minimum_jacobian_determinants: Array
    minimum_cell_measures_mm3: Array
    finite: Array
    orientation_valid: Array
    measure_valid: Array
    boundary_role_matches: Array
    boundary_profile_matches: Array
    geometry_epoch_matches: Array
    reference_epoch_matches: Array
    transfer_required: Array
    rebuild_required: Array
    accepted: Array
    fixed_topology: bool = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class HighOrderCardiacGeometryCandidate(StrictModule, NonTrainableState):
    """Fixed-topology curved coordinates, cell measures, and acceptance evidence."""

    coordinates_mm: Array
    block_cell_measures_mm3: tuple[Array, ...]
    evidence: HighOrderCardiacGeometryEvidence


class HighOrderCardiacGeometryPlan(StrictModule, NonTrainableState):
    """Qualification plan for existing quadratic tetrahedral or hexahedral geometry."""

    mesh: CellMesh
    coordinate_spec: CellGeometrySpec
    boundary_profile: CardiacBoundaryProfile
    prepared_epoch: HighOrderGeometryEpoch
    boundary_role_id: str = eqx.field(static=True)
    minimum_jacobian_determinant: float = eqx.field(static=True)
    minimum_cell_measure_mm3: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        coordinate_spec: CellGeometrySpec,
        /,
        *,
        boundary_role_id: str,
        boundary_profile: CardiacBoundaryProfile,
        prepared_epoch: HighOrderGeometryEpoch,
        minimum_jacobian_determinant: float = 1.0e-10,
        minimum_cell_measure_mm3: float = 1.0e-10,
        plan_id: str | None = None,
    ):
        if not isinstance(mesh, CellMesh):
            raise TypeError("mesh must be a CellMesh.")
        if not isinstance(coordinate_spec, CellGeometrySpec):
            raise TypeError("coordinate_spec must be a CellGeometrySpec.")
        if not isinstance(boundary_profile, CardiacBoundaryProfile):
            raise TypeError("boundary_profile must be a CardiacBoundaryProfile.")
        if not isinstance(prepared_epoch, HighOrderGeometryEpoch):
            raise TypeError("prepared_epoch must be a HighOrderGeometryEpoch.")
        role_id = str(boundary_role_id)
        if not role_id:
            raise ValueError("boundary_role_id must be non-empty.")
        if mesh.topological_dimension != 3 or mesh.ambient_dimension != 3:
            raise ValueError(
                "High-order cardiac geometry requires a three-dimensional mesh."
            )
        if any(
            block.cell_kind not in ("tetrahedron", "hexahedron") for block in mesh.blocks
        ):
            raise ValueError(
                "High-order cardiac geometry supports P2 tetrahedra and Q2 "
                "hexahedra only."
            )
        elements, routes, coordinates = coordinate_spec.resolve(mesh)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("High-order cardiac coordinates must have shape (count, 3).")
        for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
            _validate_quadratic_coordinate_element(block.cell_kind, element)
            if route.shape != (block.cell_count, element.local_dof_count):
                raise ValueError(
                    "High-order coordinate routes must have one complete row per cell."
                )
        jacobian_floor = float(minimum_jacobian_determinant)
        measure_floor = float(minimum_cell_measure_mm3)
        if (
            not np.isfinite(jacobian_floor)
            or jacobian_floor <= 0.0
            or not np.isfinite(measure_floor)
            or measure_floor <= 0.0
        ):
            raise ValueError("High-order Jacobian and measure floors must be positive.")
        payload = {
            "kind": "high-order-cardiac-geometry-plan",
            "mesh": mesh.mesh_id,
            "coordinates": coordinate_spec.geometry_layout_id,
            "coordinate_values": array_tree_fingerprint(coordinate_spec.coordinates),
            "boundary_role": role_id,
            "boundary_profile": boundary_profile.profile_id,
            "geometry_epoch": int(np.asarray(prepared_epoch.geometry)),
            "reference_epoch": int(np.asarray(prepared_epoch.reference)),
            "minimum_jacobian_determinant": jacobian_floor,
            "minimum_cell_measure_mm3": measure_floor,
            "admitted_cells": [block.cell_kind for block in mesh.blocks],
        }
        self.mesh = mesh
        self.coordinate_spec = coordinate_spec
        self.boundary_profile = boundary_profile
        self.prepared_epoch = prepared_epoch
        self.boundary_role_id = role_id
        self.minimum_jacobian_determinant = jacobian_floor
        self.minimum_cell_measure_mm3 = measure_floor
        self.plan_id = _resolved_id("plan_id", plan_id, payload)

    def prepare(self, /) -> PreparedHighOrderCardiacGeometry:
        elements, routes, coordinates = self.coordinate_spec.resolve(self.mesh)
        quadrature_points: list[Array] = []
        quadrature_weights: list[Array] = []
        quadrature_gradients: list[Array] = []
        qualification_gradients: list[Array] = []
        quadrature_orders: list[int] = []
        for element in elements:
            order = 5 if element.cell_kind == "tetrahedron" else 4
            points, weights = _reference_quadrature(element.cell_kind, order)
            _, gradients = element.tabulate(points)
            qualification_points = jnp.concatenate(
                (points, element.reference_nodes), axis=0
            )
            _, qualification = element.tabulate(qualification_points)
            quadrature_points.append(points)
            quadrature_weights.append(weights)
            quadrature_gradients.append(gradients)
            qualification_gradients.append(qualification)
            quadrature_orders.append(order)
        payload = {
            "kind": "prepared-high-order-cardiac-geometry",
            "plan": self.plan_id,
            "geometry_dofs": [array_tree_fingerprint(value) for value in routes],
            "quadrature_orders": quadrature_orders,
            "quadrature_points": [
                array_tree_fingerprint(value) for value in quadrature_points
            ],
            "quadrature_weights": [
                array_tree_fingerprint(value) for value in quadrature_weights
            ],
            "quadrature_gradients": [
                array_tree_fingerprint(value) for value in quadrature_gradients
            ],
            "qualification_gradients": [
                array_tree_fingerprint(value) for value in qualification_gradients
            ],
        }
        prepared = PreparedHighOrderCardiacGeometry(
            plan=self,
            coordinate_elements=elements,
            geometry_dofs=routes,
            quadrature_points=tuple(quadrature_points),
            quadrature_weights=tuple(quadrature_weights),
            quadrature_gradients=tuple(quadrature_gradients),
            qualification_gradients=tuple(qualification_gradients),
            quadrature_orders=tuple(quadrature_orders),
            prepared_id=canonical_fingerprint(payload),
        )
        initial = prepared.evaluate(
            coordinates,
            self.prepared_epoch,
            boundary_role_id=self.boundary_role_id,
            boundary_profile_id=self.boundary_profile.profile_id,
        )
        if not bool(np.asarray(initial.evidence.accepted)):
            raise ValueError(
                "Default high-order cardiac coordinates failed Jacobian or measure "
                "qualification."
            )
        return prepared


class PreparedHighOrderCardiacGeometry(StrictModule, NonTrainableState):
    """Prepared quadrature and fixed coordinate routes for curved cardiac geometry."""

    plan: HighOrderCardiacGeometryPlan
    coordinate_elements: tuple[FiniteElementSpec, ...]
    geometry_dofs: tuple[Array, ...]
    quadrature_points: tuple[Array, ...]
    quadrature_weights: tuple[Array, ...]
    quadrature_gradients: tuple[Array, ...]
    qualification_gradients: tuple[Array, ...]
    quadrature_orders: tuple[int, ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        plan: HighOrderCardiacGeometryPlan,
        coordinate_elements: tuple[FiniteElementSpec, ...],
        geometry_dofs: tuple[Array, ...],
        quadrature_points: tuple[Array, ...],
        quadrature_weights: tuple[Array, ...],
        quadrature_gradients: tuple[Array, ...],
        qualification_gradients: tuple[Array, ...],
        quadrature_orders: tuple[int, ...],
        prepared_id: str,
    ):
        if not isinstance(plan, HighOrderCardiacGeometryPlan):
            raise TypeError("plan must be a HighOrderCardiacGeometryPlan.")
        block_count = len(plan.mesh.blocks)
        collections = (
            coordinate_elements,
            geometry_dofs,
            quadrature_points,
            quadrature_weights,
            quadrature_gradients,
            qualification_gradients,
            quadrature_orders,
        )
        if any(len(values) != block_count for values in collections):
            raise ValueError("Prepared high-order data must contain one item per block.")
        if not all(
            isinstance(element, FiniteElementSpec) for element in coordinate_elements
        ):
            raise TypeError("coordinate_elements must contain FiniteElementSpec values.")
        resolved_elements, resolved_routes, coordinates = plan.coordinate_spec.resolve(
            plan.mesh
        )
        normalized_routes: list[Array] = []
        normalized_points: list[Array] = []
        normalized_weights: list[Array] = []
        normalized_gradients: list[Array] = []
        normalized_qualification: list[Array] = []
        normalized_orders: list[int] = []
        for (
            block,
            element,
            expected_element,
            routes,
            expected_routes,
            points,
            weights,
            gradients,
            qualification,
            order,
        ) in zip(
            plan.mesh.blocks,
            coordinate_elements,
            resolved_elements,
            geometry_dofs,
            resolved_routes,
            quadrature_points,
            quadrature_weights,
            quadrature_gradients,
            qualification_gradients,
            quadrature_orders,
            strict=True,
        ):
            if element.element_id != expected_element.element_id:
                raise ValueError("Prepared coordinate element does not match the plan.")
            routes_host = np.asarray(routes)
            expected_routes_host = np.asarray(expected_routes)
            points_host = np.asarray(points)
            weights_host = np.asarray(weights)
            gradients_host = np.asarray(gradients)
            qualification_host = np.asarray(qualification)
            order_ = int(order)
            expected_order = 5 if block.cell_kind == "tetrahedron" else 4
            if (
                routes_host.shape != (block.cell_count, element.local_dof_count)
                or not np.issubdtype(routes_host.dtype, np.integer)
                or not np.array_equal(routes_host, expected_routes_host)
                or np.any(routes_host < 0)
                or np.any(routes_host >= coordinates.shape[0])
            ):
                raise ValueError(
                    "Prepared high-order coordinate routes do not match the plan."
                )
            expected_quadrature_count = expected_order**3
            if (
                order_ != expected_order
                or points_host.shape != (expected_quadrature_count, 3)
                or weights_host.shape != (expected_quadrature_count,)
                or gradients_host.shape
                != (expected_quadrature_count, element.local_dof_count, 3)
                or qualification_host.shape
                != (
                    expected_quadrature_count + element.local_dof_count,
                    element.local_dof_count,
                    3,
                )
            ):
                raise ValueError(
                    "Prepared high-order quadrature has incompatible fixed shapes."
                )
            if (
                not np.issubdtype(points_host.dtype, np.inexact)
                or not np.issubdtype(weights_host.dtype, np.inexact)
                or not np.issubdtype(gradients_host.dtype, np.inexact)
                or not np.issubdtype(qualification_host.dtype, np.inexact)
                or not np.all(np.isfinite(points_host))
                or not np.all(np.isfinite(weights_host))
                or not np.all(weights_host > 0.0)
                or not np.all(np.isfinite(gradients_host))
                or not np.all(np.isfinite(qualification_host))
            ):
                raise ValueError(
                    "Prepared high-order quadrature data must be finite and positive."
                )
            reference_measure = 1.0 / 6.0 if block.cell_kind == "tetrahedron" else 1.0
            accumulation_tolerance = max(1.0e-12, 64.0 * np.finfo(weights_host.dtype).eps)
            if not np.isclose(
                np.sum(weights_host),
                reference_measure,
                rtol=accumulation_tolerance,
                atol=accumulation_tolerance,
            ):
                raise ValueError(
                    "Prepared quadrature does not integrate reference measure."
                )
            normalized_routes.append(jnp.asarray(routes_host, dtype=jnp.int32))
            normalized_points.append(jnp.asarray(points_host))
            normalized_weights.append(jnp.asarray(weights_host))
            normalized_gradients.append(jnp.asarray(gradients_host))
            normalized_qualification.append(jnp.asarray(qualification_host))
            normalized_orders.append(order_)
        identifier = str(prepared_id)
        if not identifier:
            raise ValueError("prepared_id must be non-empty.")
        self.plan = plan
        self.coordinate_elements = tuple(coordinate_elements)
        self.geometry_dofs = tuple(normalized_routes)
        self.quadrature_points = tuple(normalized_points)
        self.quadrature_weights = tuple(normalized_weights)
        self.quadrature_gradients = tuple(normalized_gradients)
        self.qualification_gradients = tuple(normalized_qualification)
        self.quadrature_orders = tuple(normalized_orders)
        self.prepared_id = identifier

    def evaluate(
        self,
        coordinates_mm: ArrayLike,
        current_epoch: HighOrderGeometryEpoch,
        /,
        *,
        boundary_role_id: str,
        boundary_profile_id: str,
    ) -> HighOrderCardiacGeometryCandidate:
        if not isinstance(current_epoch, HighOrderGeometryEpoch):
            raise TypeError("current_epoch must be a HighOrderGeometryEpoch.")
        coordinates = jnp.asarray(coordinates_mm)
        expected_shape = self.plan.coordinate_spec.coordinates.shape
        if coordinates.shape != expected_shape:
            raise ValueError(
                "High-order coordinates changed from the prepared fixed shape."
            )
        if not jnp.issubdtype(coordinates.dtype, jnp.inexact):
            raise TypeError("High-order coordinates must be floating-point values.")
        minimum_determinants: list[Array] = []
        minimum_measures: list[Array] = []
        cell_measures: list[Array] = []
        all_determinants: list[Array] = []
        for routes, weights, gradients, qualification in zip(
            self.geometry_dofs,
            self.quadrature_weights,
            self.quadrature_gradients,
            self.qualification_gradients,
            strict=True,
        ):
            cell_coordinates = coordinates[routes]
            quadrature_jacobian = oe.contract(
                "qir,cid->cqdr", gradients, cell_coordinates
            )
            qualification_jacobian = oe.contract(
                "qir,cid->cqdr", qualification, cell_coordinates
            )
            quadrature_determinant = _determinant_3x3(quadrature_jacobian)
            qualification_determinant = _determinant_3x3(qualification_jacobian)
            measures = oe.contract("q,cq->c", weights, jnp.abs(quadrature_determinant))
            minimum_determinants.append(jnp.min(qualification_determinant))
            minimum_measures.append(jnp.min(measures))
            cell_measures.append(measures)
            all_determinants.append(qualification_determinant)
        minimum_determinant_array = jnp.stack(minimum_determinants)
        minimum_measure_array = jnp.stack(minimum_measures)
        determinant_finite = jnp.all(
            jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in all_determinants))
        )
        measure_finite = jnp.all(
            jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in cell_measures))
        )
        finite = jnp.all(jnp.isfinite(coordinates)) & determinant_finite & measure_finite
        orientation_valid = jnp.all(
            minimum_determinant_array >= self.plan.minimum_jacobian_determinant
        )
        measure_valid = jnp.all(
            minimum_measure_array >= self.plan.minimum_cell_measure_mm3
        )
        role_matches = jnp.asarray(str(boundary_role_id) == self.plan.boundary_role_id)
        profile_matches = jnp.asarray(
            str(boundary_profile_id) == self.plan.boundary_profile.profile_id
        )
        geometry_epoch_matches = (
            current_epoch.geometry == self.plan.prepared_epoch.geometry
        )
        reference_epoch_matches = (
            current_epoch.reference == self.plan.prepared_epoch.reference
        )
        transfer_required = (
            (~geometry_epoch_matches)
            | (~reference_epoch_matches)
            | (~role_matches)
            | (~profile_matches)
        )
        rebuild_required = (
            (~geometry_epoch_matches) | (~role_matches) | (~profile_matches)
        )
        accepted = (
            finite
            & orientation_valid
            & measure_valid
            & role_matches
            & profile_matches
            & geometry_epoch_matches
            & reference_epoch_matches
        )
        evidence = HighOrderCardiacGeometryEvidence(
            minimum_jacobian_determinants=minimum_determinant_array,
            minimum_cell_measures_mm3=minimum_measure_array,
            finite=finite,
            orientation_valid=orientation_valid,
            measure_valid=measure_valid,
            boundary_role_matches=role_matches,
            boundary_profile_matches=profile_matches,
            geometry_epoch_matches=geometry_epoch_matches,
            reference_epoch_matches=reference_epoch_matches,
            transfer_required=transfer_required,
            rebuild_required=rebuild_required,
            accepted=accepted,
            fixed_topology=True,
            prepared_id=self.prepared_id,
        )
        return HighOrderCardiacGeometryCandidate(
            coordinates_mm=coordinates,
            block_cell_measures_mm3=tuple(cell_measures),
            evidence=evidence,
        )

    def commit_epoch(
        self,
        candidate: HighOrderCardiacGeometryCandidate,
        target_epoch: HighOrderGeometryEpoch,
        /,
    ) -> PreparedHighOrderCardiacGeometry:
        """Commit accepted coordinates into a rebuilt fixed-topology geometry epoch."""
        if not isinstance(candidate, HighOrderCardiacGeometryCandidate):
            raise TypeError("candidate must be a HighOrderCardiacGeometryCandidate.")
        if not isinstance(target_epoch, HighOrderGeometryEpoch):
            raise TypeError("target_epoch must be a HighOrderGeometryEpoch.")
        if candidate.evidence.prepared_id != self.prepared_id:
            raise ValueError(
                "High-order geometry candidate belongs to a different preparation."
            )
        if not bool(np.asarray(candidate.evidence.accepted)):
            raise ValueError("Only an accepted high-order geometry candidate can commit.")
        source_geometry = int(np.asarray(self.plan.prepared_epoch.geometry))
        source_reference = int(np.asarray(self.plan.prepared_epoch.reference))
        target_geometry = int(np.asarray(target_epoch.geometry))
        target_reference = int(np.asarray(target_epoch.reference))
        if (
            target_geometry < source_geometry
            or target_reference < source_reference
            or (
                target_geometry == source_geometry
                and target_reference == source_reference
            )
        ):
            raise ValueError("Committed geometry epochs must advance without regression.")
        coordinates_changed = not np.array_equal(
            np.asarray(candidate.coordinates_mm),
            np.asarray(self.plan.coordinate_spec.coordinates),
        )
        if coordinates_changed and target_geometry == source_geometry:
            raise ValueError(
                "Changed high-order coordinates must advance the geometry epoch."
            )
        coordinate_spec = CellGeometrySpec(
            dict(
                zip(
                    self.plan.coordinate_spec.block_names,
                    self.plan.coordinate_spec.elements,
                    strict=True,
                )
            ),
            dict(
                zip(
                    self.plan.coordinate_spec.block_names,
                    self.geometry_dofs,
                    strict=True,
                )
            ),
            candidate.coordinates_mm,
        )
        committed_plan = HighOrderCardiacGeometryPlan(
            self.plan.mesh,
            coordinate_spec,
            boundary_role_id=self.plan.boundary_role_id,
            boundary_profile=self.plan.boundary_profile,
            prepared_epoch=target_epoch,
            minimum_jacobian_determinant=self.plan.minimum_jacobian_determinant,
            minimum_cell_measure_mm3=self.plan.minimum_cell_measure_mm3,
        )
        return committed_plan.prepare()


def _validate_quadratic_coordinate_element(
    cell_kind: str, element: FiniteElementSpec, /
) -> None:
    expected_dofs = 10 if cell_kind == "tetrahedron" else 27
    expected_family = (
        "SimplexLagrange" if cell_kind == "tetrahedron" else "TensorProductLagrange"
    )
    if (
        element.cell_kind != cell_kind
        or element.degree != 2
        or element.family != expected_family
        or element.conformity != "H1"
        or element.representation != "point_value"
        or element.mapping != "identity"
        or element.value_shape
        or element.local_dof_count != expected_dofs
    ):
        name = "P2 tetrahedral" if cell_kind == "tetrahedron" else "Q2 hexahedral"
        raise ValueError(
            f"High-order cardiac geometry requires a qualified {name} coordinate element."
        )


def _reference_quadrature(cell_kind: str, order: int, /) -> tuple[Array, Array]:
    axis, weights = np.polynomial.legendre.leggauss(order)
    axis = 0.5 * (axis + 1.0)
    weights = 0.5 * weights
    first, second, third = np.meshgrid(axis, axis, axis, indexing="ij")
    if cell_kind == "hexahedron":
        points = np.stack((first, second, third), axis=-1)
        combined = (
            weights[:, None, None] * weights[None, :, None] * weights[None, None, :]
        )
    elif cell_kind == "tetrahedron":
        one_minus_first = 1.0 - first
        one_minus_second = 1.0 - second
        points = np.stack(
            (
                first,
                one_minus_first * second,
                one_minus_first * one_minus_second * third,
            ),
            axis=-1,
        )
        combined = (
            weights[:, None, None]
            * weights[None, :, None]
            * weights[None, None, :]
            * one_minus_first**2
            * one_minus_second
        )
    else:
        raise ValueError("Only tetrahedral and hexahedral quadrature is qualified.")
    return (
        jnp.asarray(points.reshape((-1, 3))),
        jnp.asarray(combined.reshape((-1,))),
    )


def _determinant_3x3(matrix: Array, /) -> Array:
    return (
        matrix[..., 0, 0]
        * (matrix[..., 1, 1] * matrix[..., 2, 2] - matrix[..., 1, 2] * matrix[..., 2, 1])
        - matrix[..., 0, 1]
        * (matrix[..., 1, 0] * matrix[..., 2, 2] - matrix[..., 1, 2] * matrix[..., 2, 0])
        + matrix[..., 0, 2]
        * (matrix[..., 1, 0] * matrix[..., 2, 1] - matrix[..., 1, 1] * matrix[..., 2, 0])
    )


def _resolved_id(name: str, value: str | None, payload: dict[str, object], /) -> str:
    if value is None:
        return canonical_fingerprint(payload)
    identifier = str(value)
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


__all__ = [
    "HighOrderCardiacGeometryCandidate",
    "HighOrderCardiacGeometryEvidence",
    "HighOrderCardiacGeometryPlan",
    "HighOrderGeometryEpoch",
    "PreparedHighOrderCardiacGeometry",
]
