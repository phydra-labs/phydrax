#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""High-order CAD curving of Pk simplex and Qk tensor cell meshes.

Geometry nodes are created in the reference ordering of the discretization's
Lagrange elements and shared between cells by entity identity, classified on
B-Rep entities by the association rules of :mod:`._association`, and nodes on
B-Rep vertices, edges, or faces of lower dimension than the ambient space are
projected onto their entity. Interior nodes are relaxed with
:func:`optimize_cell_geometry_coordinates`; the objective combines the inverse
mean-ratio distortion of the Jacobian relative to the straight-sided element
(sampled at the Bernstein control-point lattice of the Jacobian determinant),
a displacement term, and the squared distance of constrained nodes from the
tangent space at their current foot points. Constrained nodes are re-projected
between accepted rounds, periodic target nodes follow their source nodes
through the declared isometry, and a candidate is accepted only when the
Bernstein certificate proves every cell valid and every constrained node lies
within the residual tolerance; otherwise the result rolls back.
"""

from __future__ import annotations

from enum import StrEnum
from itertools import product
from typing import final, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    CellGeometrySpec,
    CellMesh,
    CellValidityCertificate,
    CellValidityPolicy,
    certify_cell_geometry_validity,
    reference_cell_topology,
)
from ..discretization.fem import lagrange_element
from ..ein import contract
from ..geometry.brep._projection import BRepProjectionStatus, PreparedBRepProjection
from ..linalg import determinant_small_linear, inverse_small_linear, SmallLinearSolvePlan
from ..optim import MinimizationResult, OptimizationTermination
from ._association import (
    _entity_rows,
    _incidence_pairs,
    _mesh_entity_classes,
    _RESOLVED_STATUS,
    GeometryAssociation,
)
from ._coupling import PeriodicCoupling
from ._optimization import optimize_cell_geometry_coordinates


_CURVED_KINDS = ("triangle", "tetrahedron", "quadrilateral", "hexahedron")


class HighOrderCurvingStatus(StrEnum):
    """Outcome of one curving.

    ``CURVED``: the returned geometry is certified valid with every constrained
    node within the residual tolerance. ``ROLLED_BACK_INVALID``: no candidate was
    certified valid; the straight geometry is returned. ``ROLLED_BACK_RESIDUAL``:
    valid candidates missed the residual tolerance. ``UNRESOLVED_ASSOCIATION``:
    some geometry node has an ambiguous or unclassified B-Rep class, so no
    curving was attempted.
    """

    CURVED = "curved"
    ROLLED_BACK_INVALID = "rolled_back_invalid"
    ROLLED_BACK_RESIDUAL = "rolled_back_residual"
    UNRESOLVED_ASSOCIATION = "unresolved_association"


def _positive(value: float, name: str, /) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _non_negative(value: float, name: str, /) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


@final
class HighOrderCurvingPolicy(StrictModule, NonTrainableState):
    """Degree, objective weights, acceptance tolerances, and solver controls.

    ``degree`` is the geometry order (2 or 3). ``relaxation_rounds`` bounds the
    optimize/re-project/certify rounds (0 accepts or rejects the projected
    configuration directly). Weights scale the mean sampled distortion, the
    squared tangent-space residual of constrained nodes, and the squared nodal
    displacement (the latter two relative to the mean straight edge length).
    ``regularization`` is the Escobar determinant regularization of the
    distortion, which keeps inverted starting configurations finite.
    ``residual_tolerance`` is the absolute acceptance bound of constrained-node
    CAD residuals.
    """

    degree: int = eqx.field(static=True)
    relaxation_rounds: int = eqx.field(static=True)
    distortion_weight: float = eqx.field(static=True)
    tangent_weight: float = eqx.field(static=True)
    displacement_weight: float = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    validity: CellValidityPolicy
    termination: OptimizationTermination
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        degree: int = 2,
        relaxation_rounds: int = 3,
        distortion_weight: float = 1.0,
        tangent_weight: float = 1.0e3,
        displacement_weight: float = 1.0e-3,
        regularization: float = 5.0e-2,
        residual_tolerance: float = 1.0e-7,
        validity: CellValidityPolicy | None = None,
        termination: OptimizationTermination | None = None,
    ):
        if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)):
            raise TypeError("degree must be an integer.")
        if degree not in (2, 3):
            raise ValueError("High-order curving supports geometry degrees 2 and 3.")
        if isinstance(relaxation_rounds, bool) or not isinstance(
            relaxation_rounds, (int, np.integer)
        ):
            raise TypeError("relaxation_rounds must be an integer.")
        if relaxation_rounds < 0:
            raise ValueError("relaxation_rounds must be non-negative.")
        validity_ = CellValidityPolicy() if validity is None else validity
        termination_ = (
            OptimizationTermination(maximum_steps=64)
            if termination is None
            else termination
        )
        if not isinstance(validity_, CellValidityPolicy):
            raise TypeError("validity must be CellValidityPolicy or None.")
        if not isinstance(termination_, OptimizationTermination):
            raise TypeError("termination must be OptimizationTermination or None.")
        self.degree = int(degree)
        self.relaxation_rounds = int(relaxation_rounds)
        self.distortion_weight = _positive(distortion_weight, "distortion_weight")
        self.tangent_weight = _non_negative(tangent_weight, "tangent_weight")
        self.displacement_weight = _non_negative(
            displacement_weight, "displacement_weight"
        )
        self.regularization = _positive(regularization, "regularization")
        self.residual_tolerance = _positive(residual_tolerance, "residual_tolerance")
        self.validity = validity_
        self.termination = termination_
        self.policy_id = canonical_fingerprint(
            {
                "kind": "high-order-curving-policy",
                "degree": self.degree,
                "relaxation_rounds": self.relaxation_rounds,
                "distortion_weight": self.distortion_weight,
                "tangent_weight": self.tangent_weight,
                "displacement_weight": self.displacement_weight,
                "regularization": self.regularization,
                "residual_tolerance": self.residual_tolerance,
                "validity": validity_.policy_id,
                "termination": [
                    termination_.absolute_optimality,
                    termination_.relative_optimality,
                    termination_.absolute_step,
                    termination_.relative_step,
                    termination_.maximum_steps,
                    termination_.maximum_evaluations,
                ],
            }
        )


@final
class CurvedGeometryEvidence(StrictModule, NonTrainableState):
    """Validity certificate and CAD residuals of one high-order geometry.

    Node rows follow ``geometry.coordinates``: ``node_dimensions``/``node_indices``
    are the B-Rep class of each node's owning mesh entity, ``constrained`` marks
    nodes on B-Rep entities of lower dimension than the ambient space, and
    ``node_residuals`` their distance to that entity (zero for free nodes).
    ``minimum_scaled_jacobian`` and ``maximum_distortion`` are sampled at the
    Bernstein control-point lattice; ``accepted`` requires every cell certified
    valid and every constrained residual within the tolerance.
    """

    certificate: CellValidityCertificate
    node_dimensions: Array
    node_indices: Array
    constrained: Array
    node_residuals: Array
    maximum_residual: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    minimum_scaled_jacobian: float = eqx.field(static=True)
    maximum_distortion: float = eqx.field(static=True)
    accepted: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        certificate: CellValidityCertificate,
        node_dimensions: np.ndarray,
        node_indices: np.ndarray,
        constrained: np.ndarray,
        node_residuals: np.ndarray,
        residual_tolerance: float,
        minimum_scaled_jacobian: float,
        maximum_distortion: float,
        /,
    ):
        if not isinstance(certificate, CellValidityCertificate):
            raise TypeError("certificate must be CellValidityCertificate.")
        dims = np.asarray(node_dimensions, dtype=np.int8)
        indices = np.asarray(node_indices, dtype=np.int32)
        mask = np.asarray(constrained, dtype=np.bool_)
        residuals = np.asarray(node_residuals, dtype=np.float64)
        if (
            dims.ndim != 1
            or indices.shape != dims.shape
            or mask.shape != dims.shape
            or residuals.shape != dims.shape
        ):
            raise ValueError("Curved geometry evidence rows must align with nodes.")
        maximum = float(np.max(residuals)) if residuals.size else 0.0
        self.certificate = certificate
        self.node_dimensions = jnp.asarray(dims)
        self.node_indices = jnp.asarray(indices)
        self.constrained = jnp.asarray(mask)
        self.node_residuals = jnp.asarray(residuals)
        self.maximum_residual = maximum
        self.residual_tolerance = float(residual_tolerance)
        self.minimum_scaled_jacobian = float(minimum_scaled_jacobian)
        self.maximum_distortion = float(maximum_distortion)
        self.accepted = bool(certificate.all_certified and maximum <= residual_tolerance)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "curved-geometry-evidence",
                "certificate": certificate.certificate_id,
                "node_dimensions": array_tree_fingerprint(dims),
                "node_indices": array_tree_fingerprint(indices),
                "node_residuals": array_tree_fingerprint(residuals),
                "residual_tolerance": self.residual_tolerance,
            }
        )


@final
class HighOrderCurvingResult(StrictModule, NonTrainableState):
    """Accepted curved geometry, or the straight geometry after a rollback.

    ``evidence`` certifies the returned ``geometry``; ``candidate`` is the
    evidence of the last evaluated curved candidate (the rejection reason after
    a rollback). ``minimizations`` carry the native optimizer evidence of every
    relaxation round.
    """

    status: HighOrderCurvingStatus = eqx.field(static=True)
    geometry: CellGeometrySpec
    straight: CellGeometrySpec
    evidence: CurvedGeometryEvidence
    candidate: CurvedGeometryEvidence | None
    minimizations: tuple[MinimizationResult, ...]
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        status: HighOrderCurvingStatus,
        geometry: CellGeometrySpec,
        straight: CellGeometrySpec,
        evidence: CurvedGeometryEvidence,
        candidate: CurvedGeometryEvidence | None,
        minimizations: tuple[MinimizationResult, ...],
        /,
    ):
        if not isinstance(status, HighOrderCurvingStatus):
            raise TypeError("status must be HighOrderCurvingStatus.")
        if status is HighOrderCurvingStatus.CURVED and not evidence.accepted:
            raise ValueError("A curved result requires accepted evidence.")
        self.status = status
        self.geometry = geometry
        self.straight = straight
        self.evidence = evidence
        self.candidate = candidate
        self.minimizations = tuple(minimizations)
        self.result_id = canonical_fingerprint(
            {
                "kind": "high-order-curving-result",
                "status": status.value,
                "geometry": geometry.geometry_layout_id,
                "coordinates": array_tree_fingerprint(np.asarray(geometry.coordinates)),
                "evidence": evidence.evidence_id,
                "candidate": None if candidate is None else candidate.evidence_id,
            }
        )


# -- geometry nodes ---------------------------------------------------------------


_WEIGHT_QUANTUM = float(2**40)


def _straight_geometry(mesh: CellMesh, degree: int, /) -> CellGeometrySpec:
    """Straight-sided Lagrange geometry with shared nodes in canonical order.

    Local nodes follow the reference ordering of the discretization's Lagrange
    element. A node is identified by its owning mesh entity (sorted vertex global
    IDs) and its affine or multilinear weights on those vertices, so neighboring
    cells share it independently of their local edge and face orientation. Vertex
    nodes are the mesh vertex rows; the other nodes follow in (dimension, entity,
    weights) order.
    """
    kinds = {block.cell_kind for block in mesh.blocks}
    if not kinds <= set(_CURVED_KINDS):
        raise ValueError("Curving supports triangle, tetrahedron, quad, and hex blocks.")
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    elements, routes, keys, positions, owners = {}, {}, [], [], []
    for block in mesh.blocks:
        element = lagrange_element(block.cell_kind, degree)
        topology = reference_cell_topology(block.cell_kind)
        cells = np.asarray(block.vertices, dtype=np.int64)
        weights, _ = lagrange_element(block.cell_kind, 1).tabulate(
            element.reference_nodes
        )
        weights = np.asarray(weights, dtype=np.float64)
        route = np.full((cells.shape[0], element.local_dof_count), -1, dtype=np.int64)
        for dimension, entities in enumerate(element.entity_dofs):
            for local, dofs in enumerate(entities):
                if not dofs:
                    continue
                vertices = list(topology.entities[dimension][local])
                if dimension == 0:
                    route[:, dofs[0]] = cells[:, vertices[0]]
                    continue
                ids = vertex_ids[cells[:, vertices]]
                order = np.argsort(ids, axis=1, kind="stable")
                entity_weights = np.broadcast_to(
                    weights[np.asarray(dofs)][:, vertices],
                    (cells.shape[0], len(dofs), len(vertices)),
                )
                sorted_weights = np.take_along_axis(
                    entity_weights, order[:, None, :], axis=2
                )
                key = np.full((cells.shape[0], len(dofs), 17), -1, dtype=np.int64)
                key[..., 0] = dimension
                key[..., 1 : 1 + len(vertices)] = np.take_along_axis(ids, order, axis=1)[
                    :, None, :
                ]
                key[..., 9 : 9 + len(vertices)] = np.rint(
                    sorted_weights * _WEIGHT_QUANTUM
                )
                keys.append(key.reshape(-1, 17))
                positions.append(
                    np.asarray(
                        contract("dv,cva->cda", weights[np.asarray(dofs)], points[cells])
                    ).reshape(-1, points.shape[1])
                )
                owners.append((block.name, dofs, cells.shape[0]))
        elements[block.name] = element
        routes[block.name] = route
    table, inverse = np.unique(np.concatenate(keys), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    coordinates = np.empty((points.shape[0] + table.shape[0], points.shape[1]))
    coordinates[: points.shape[0]] = points
    coordinates[points.shape[0] + inverse] = np.concatenate(positions)
    cursor = 0
    for name, dofs, count in owners:
        size = count * len(dofs)
        routes[name][:, list(dofs)] = points.shape[0] + inverse[
            cursor : cursor + size
        ].reshape(count, len(dofs))
        cursor += size
    return CellGeometrySpec(elements, routes, coordinates)


def _entity_key_table(mesh: CellMesh, dimension: int, /) -> np.ndarray:
    """Sorted vertex-global-ID key of every entity row of ``dimension``."""
    pairs = _incidence_pairs(mesh, 0, dimension)
    counts = np.bincount(pairs[:, 1], minlength=mesh.entity_set(dimension).count)
    if np.unique(counts).size != 1:
        raise ValueError("Geometry node ownership requires uniform entity arity.")
    identifiers = np.asarray(mesh.vertex_global_ids, dtype=np.int64)[pairs[:, 0]]
    order = np.lexsort((identifiers, pairs[:, 1]))
    return identifiers[order].reshape(-1, int(counts[0]))


def _node_owners(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> tuple[np.ndarray, np.ndarray]:
    """Owning mesh entity ``(dimension, row)`` of every geometry node."""
    from ._topology_edit import key_rows

    elements, routes, coordinates = geometry.resolve(mesh)
    count = coordinates.shape[0]
    top = mesh.topological_dimension
    dims = np.full((count,), -1, dtype=np.int64)
    rows = np.full((count,), -1, dtype=np.int64)
    tables = {
        dimension: _entity_key_table(mesh, dimension) for dimension in range(1, top)
    }
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        route_ = np.asarray(route, dtype=np.int64)
        cells = np.asarray(block.vertices, dtype=np.int64)
        topology = reference_cell_topology(block.cell_kind)
        cell_rows = _entity_rows(mesh, top, np.asarray(block.global_ids, dtype=np.int64))
        for dimension, entities in enumerate(element.entity_dofs):
            for local, dofs in enumerate(entities):
                if not dofs:
                    continue
                vertices = topology.entities[dimension][local]
                if dimension == 0:
                    owners = cells[:, vertices[0]]
                elif dimension == top:
                    owners = cell_rows
                else:
                    keys = np.sort(vertex_ids[cells[:, list(vertices)]], axis=1)
                    owners = key_rows(tables[dimension], keys)
                nodes = route_[:, list(dofs)]
                previous = rows[nodes]
                if np.any((previous >= 0) & (previous != owners[:, None])) or np.any(
                    (dims[nodes] >= 0) & (dims[nodes] != dimension)
                ):
                    raise ValueError("Geometry nodes are shared inconsistently.")
                dims[nodes] = dimension
                rows[nodes] = owners[:, None]
    if np.any(rows < 0):
        raise ValueError("Every geometry node must belong to one mesh entity.")
    return dims, rows


class _NodeClasses(NamedTuple):
    dimensions: np.ndarray
    indices: np.ndarray
    resolved: np.ndarray
    constrained: np.ndarray
    fixed: np.ndarray


def _node_classes(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    association: GeometryAssociation,
    projection: PreparedBRepProjection,
    /,
) -> _NodeClasses:
    owner_dims, owner_rows = _node_owners(mesh, geometry)
    levels = _mesh_entity_classes(mesh, association, projection)
    count = owner_dims.size
    dims = np.full((count,), -1, dtype=np.int64)
    indices = np.full((count,), -1, dtype=np.int64)
    status = np.full((count,), BRepProjectionStatus.FAILED, dtype=np.int8)
    for level, classes in enumerate(levels):
        selected = owner_dims == level
        dims[selected] = classes.dimensions[owner_rows[selected]]
        indices[selected] = classes.indices[owner_rows[selected]]
        status[selected] = classes.status[owner_rows[selected]]
    constrained = (dims >= 0) & (dims <= 2) & (dims < mesh.ambient_dimension)
    return _NodeClasses(
        dims,
        indices,
        np.isin(status, _RESOLVED_STATUS),
        constrained,
        owner_dims == 0,
    )


# -- distortion -------------------------------------------------------------------


def _determinant_degree(kind: str, degree: int, embedded: bool, /) -> int:
    match kind:
        case "triangle":
            return (4 if embedded else 2) * (degree - 1)
        case "tetrahedron":
            return 3 * (degree - 1)
        case "quadrilateral":
            return (2 if embedded else 1) * (2 * degree - 1)
        case "hexahedron":
            return 3 * degree - 1
        case _:
            raise ValueError(f"No curving sample lattice for {kind!r}.")


def _control_lattice(kind: str, degree: int, embedded: bool, /) -> np.ndarray:
    """Bernstein control-point (Greville) lattice of the Jacobian determinant."""
    order = max(_determinant_degree(kind, degree, embedded), 1)
    dimension = reference_cell_topology(kind).dimension
    grid = np.asarray(tuple(product(range(order + 1), repeat=dimension)), np.float64)
    if kind in ("triangle", "tetrahedron"):
        grid = grid[np.sum(grid, axis=1) <= order]
    return grid / order


class _BlockDistortion(StrictModule):
    """Sampled Jacobians of one block relative to its straight-sided element."""

    routes: Array
    gradients: Array
    target_inverses: Array
    plan: SmallLinearSolvePlan
    embedded: bool = eqx.field(static=True)

    def ratios(self, coordinates: Array, /) -> tuple[Array, Array, Array]:
        """Target-normalized ``(frobenius, determinant, dimension)`` per sample."""
        jacobian = contract("mnk,cna->cmak", self.gradients, coordinates[self.routes])
        if self.embedded:
            gram = jnp.swapaxes(jacobian, -1, -2) @ jacobian
            normalized = gram @ self.target_inverses
            frobenius = jnp.trace(normalized, axis1=-2, axis2=-1)
            determinant = jnp.sqrt(
                jnp.maximum(determinant_small_linear(self.plan, normalized), 0.0)
            )
        else:
            normalized = jacobian @ self.target_inverses
            frobenius = jnp.sum(normalized**2, axis=(-2, -1))
            determinant = determinant_small_linear(self.plan, normalized)
        return frobenius, determinant, jacobian

    def distortion(self, coordinates: Array, regularization: Array, /) -> Array:
        frobenius, determinant, _ = self.ratios(coordinates)
        dimension = self.gradients.shape[-1]
        # Escobar regularization keeps inverted samples finite and penalized.
        regularized = 0.5 * (
            determinant + jnp.sqrt(determinant**2 + 4 * regularization**2)
        )
        return frobenius / (dimension * regularized ** (2.0 / dimension)) - 1.0


class _CurvingEnergy(StrictModule):
    """Distortion, tangent-space residual, and displacement of one relaxation."""

    blocks: tuple[_BlockDistortion, ...]
    constrained_rows: Array
    feet: Array
    normal_projectors: Array
    reference: Array
    weights: Array
    regularization: Array

    def __call__(self, coordinates: Array, /) -> Array:
        distortion = sum(
            jnp.mean(block.distortion(coordinates, self.regularization))
            for block in self.blocks
        )
        offset = coordinates[self.constrained_rows] - self.feet
        normal = contract("bij,bj->bi", self.normal_projectors, offset)
        tangent = jnp.sum(normal**2)
        displacement = jnp.sum((coordinates - self.reference) ** 2)
        return (
            self.weights[0] * distortion
            + self.weights[1] * tangent
            + self.weights[2] * displacement
        )


def _distortion_blocks(
    mesh: CellMesh, geometry: CellGeometrySpec, degree: int, /
) -> tuple[_BlockDistortion, ...]:
    """Sample tables and straight-sided target inverses of every block."""
    elements, routes, _ = geometry.resolve(mesh)
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    ambient = mesh.ambient_dimension
    blocks = []
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        dimension = reference_cell_topology(block.cell_kind).dimension
        embedded = ambient > dimension
        samples = _control_lattice(block.cell_kind, degree, embedded)
        _, gradients = element.tabulate(samples)
        _, linear = lagrange_element(block.cell_kind, 1).tabulate(samples)
        straight = contract(
            "mnk,cna->cmak",
            np.asarray(linear),
            points[np.asarray(block.vertices, dtype=np.int64)],
        )
        plan = SmallLinearSolvePlan(dimension)
        target = (
            jnp.swapaxes(straight, -1, -2) @ straight
            if embedded
            else jnp.asarray(straight)
        )
        inverse = inverse_small_linear(plan, target)
        if not bool(jnp.all(inverse.successful)):
            raise ValueError("Straight-sided reference elements must be nondegenerate.")
        blocks.append(
            _BlockDistortion(
                routes=jnp.asarray(route, dtype=jnp.int32),
                gradients=jnp.asarray(gradients, dtype=jnp.float64),
                target_inverses=inverse.value,
                plan=plan,
                embedded=embedded,
            )
        )
    return tuple(blocks)


def _sampled_quality(
    blocks: tuple[_BlockDistortion, ...], coordinates: np.ndarray, /
) -> tuple[float, float]:
    """Minimum sampled scaled Jacobian and maximum sampled distortion."""
    values = jnp.asarray(coordinates)
    scaled, distortion = [], []
    for block in blocks:
        frobenius, determinant, jacobian = block.ratios(values)
        dimension = block.gradients.shape[-1]
        columns = jnp.prod(jnp.linalg.norm(jacobian, axis=-2), axis=-1)
        if block.embedded:
            gram = jnp.swapaxes(jacobian, -1, -2) @ jacobian
            measure = jnp.sqrt(
                jnp.maximum(determinant_small_linear(block.plan, gram), 0.0)
            )
        else:
            measure = determinant_small_linear(block.plan, jacobian)
        scaled.append(
            jnp.min(measure / jnp.maximum(columns, jnp.finfo(values.dtype).tiny))
        )
        positive = jnp.where(determinant > 0.0, determinant, jnp.nan)
        distortion.append(
            jnp.nanmax(frobenius / (dimension * positive ** (2.0 / dimension)))
        )
    return float(min(scaled)), float(max(distortion))


# -- evidence ---------------------------------------------------------------------


def _node_residuals(
    projection: PreparedBRepProjection,
    coordinates: np.ndarray,
    classes: _NodeClasses,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Closest points, residuals, unit normals-space projectors of constrained nodes."""
    rows = np.flatnonzero(classes.constrained)
    ambient = coordinates.shape[1]
    residuals = np.zeros((coordinates.shape[0],))
    result = projection.project(
        coordinates[rows], classes.dimensions[rows], classes.indices[rows]
    )
    failed = ~np.isin(np.asarray(result.status), _RESOLVED_STATUS)
    feet = np.asarray(result.points)
    residuals[rows] = np.where(failed, np.inf, np.asarray(result.residuals))
    identity = np.eye(ambient)
    tangents = np.asarray(result.tangents)
    normals = np.asarray(result.normals)
    projectors = np.zeros((rows.size, ambient, ambient))
    edge = classes.dimensions[rows] == 1
    face = classes.dimensions[rows] == 2
    projectors[edge] = identity - tangents[edge, 0, :, None] * tangents[edge, 0, None, :]
    projectors[face] = normals[face, :, None] * normals[face, None, :]
    vertex = classes.dimensions[rows] == 0
    projectors[vertex] = identity
    projectors = np.where(np.isfinite(projectors), projectors, 0.0)
    feet = np.where(np.isfinite(feet), feet, coordinates[rows])
    return rows, feet, residuals, projectors


def _evidence(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    classes: _NodeClasses,
    projection: PreparedBRepProjection,
    blocks: tuple[_BlockDistortion, ...],
    policy: HighOrderCurvingPolicy,
    /,
) -> CurvedGeometryEvidence:
    coordinates = np.asarray(geometry.coordinates, dtype=np.float64)
    _, _, residuals, _ = _node_residuals(projection, coordinates, classes)
    scaled, distortion = _sampled_quality(blocks, coordinates)
    return CurvedGeometryEvidence(
        certify_cell_geometry_validity(geometry, mesh=mesh, policy=policy.validity),
        classes.dimensions,
        classes.indices,
        classes.constrained,
        residuals,
        policy.residual_tolerance,
        scaled,
        distortion,
    )


def _require_inputs(
    mesh: CellMesh,
    association: GeometryAssociation,
    projection: PreparedBRepProjection,
    policy: HighOrderCurvingPolicy,
    /,
) -> None:
    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if not isinstance(association, GeometryAssociation):
        raise TypeError("association must be GeometryAssociation.")
    if not isinstance(projection, PreparedBRepProjection):
        raise TypeError("projection must be PreparedBRepProjection.")
    if not isinstance(policy, HighOrderCurvingPolicy):
        raise TypeError("policy must be HighOrderCurvingPolicy.")


def verify_curved_geometry(
    geometry: CellGeometrySpec,
    mesh: CellMesh,
    association: GeometryAssociation,
    projection: PreparedBRepProjection,
    /,
    *,
    policy: HighOrderCurvingPolicy,
) -> CurvedGeometryEvidence:
    """Certify an existing high-order geometry (e.g. a Gmsh high-order output).

    Nodes are classified through their owning mesh entities and the vertex
    ``association``; the evidence carries the Bernstein validity certificate and
    the CAD residual of every constrained node. Nothing is moved.
    """
    _require_inputs(mesh, association, projection, policy)
    if not isinstance(geometry, CellGeometrySpec):
        raise TypeError("geometry must be CellGeometrySpec.")
    classes = _node_classes(mesh, geometry, association, projection)
    degree = max(int(element.degree) for element in geometry.elements)
    blocks = _distortion_blocks(mesh, geometry, degree)
    return _evidence(mesh, geometry, classes, projection, blocks, policy)


# -- curving ----------------------------------------------------------------------


class _PeriodicNodes(NamedTuple):
    """Target nodes and the isometry mapping their paired source nodes onto them."""

    sources: np.ndarray
    targets: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray


def _periodic_nodes(
    mesh: CellMesh,
    straight: CellGeometrySpec,
    periodic: tuple[PeriodicCoupling, ...],
    /,
) -> _PeriodicNodes:
    """Pair the geometry nodes of every periodic coupling through its isometry.

    A node belongs to a side when every vertex of its owning mesh entity is in
    the side's vertex scope; sides are paired on the straight positions.
    """
    owner_dims, owner_rows = _node_owners(mesh, straight)
    coordinates = np.asarray(straight.coordinates, dtype=np.float64)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    ambient = mesh.ambient_dimension
    incidences = tuple(
        _incidence_pairs(mesh, 0, dimension)
        for dimension in range(mesh.topological_dimension + 1)
    )
    sources, targets, rotations, translations = [], [], [], []
    for coupling in periodic:
        if not isinstance(coupling, PeriodicCoupling):
            raise TypeError("periodic must contain PeriodicCoupling values.")
        scopes = (coupling.source_scope, coupling.target_scope)
        if any(
            scope.entity_dimension != 0
            or scope.entity_set_id != mesh.entity_set(0).entity_set_id
            for scope in scopes
        ):
            raise ValueError("Periodic couplings must pair vertices of the curved mesh.")
        sides = []
        for scope in scopes:
            inside = np.isin(vertex_ids, np.asarray(scope.entity_ids, dtype=np.int64))
            member = np.zeros((owner_dims.size,), dtype=np.bool_)
            for dimension, pairs in enumerate(incidences):
                count = mesh.entity_set(dimension).count
                total = np.bincount(pairs[:, 1], minlength=count)
                hits = np.bincount(pairs[inside[pairs[:, 0]], 1], minlength=count)
                selected = owner_dims == dimension
                member[selected] = (hits == total)[owner_rows[selected]]
            sides.append(np.flatnonzero(member))
        source_nodes, target_nodes = sides
        source_rows, target_rows = coupling.match_points(
            coordinates[source_nodes], coordinates[target_nodes]
        )
        sources.append(source_nodes[source_rows])
        targets.append(target_nodes[target_rows])
        rotations.append(
            np.broadcast_to(
                np.asarray(coupling.rotation), (target_rows.size, ambient, ambient)
            )
        )
        translations.append(
            np.broadcast_to(np.asarray(coupling.translation), (target_rows.size, ambient))
        )
    if not sources:
        empty = np.zeros((0,), dtype=np.int64)
        return _PeriodicNodes(
            empty, empty, np.zeros((0, ambient, ambient)), np.zeros((0, ambient))
        )
    target = np.concatenate(targets)
    if np.unique(target).size != target.size:
        raise ValueError("Periodic couplings must not share target nodes.")
    return _PeriodicNodes(
        np.concatenate(sources),
        target,
        np.concatenate(rotations),
        np.concatenate(translations),
    )


def _apply_periodic(coordinates: np.ndarray, periodic: _PeriodicNodes, /) -> np.ndarray:
    result = coordinates.copy()
    result[periodic.targets] = (
        np.asarray(
            contract("pij,pj->pi", periodic.rotations, coordinates[periodic.sources])
        )
        + periodic.translations
    )
    return result


def _project_constrained(
    projection: PreparedBRepProjection,
    coordinates: np.ndarray,
    classes: _NodeClasses,
    periodic: _PeriodicNodes,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-project movable constrained nodes; periodic targets follow their sources."""
    rows, feet, _, projectors = _node_residuals(projection, coordinates, classes)
    result = coordinates.copy()
    movable = ~classes.fixed[rows]
    result[rows[movable]] = feet[movable]
    return _apply_periodic(result, periodic), rows, projectors


def curve_cell_mesh(
    mesh: CellMesh,
    association: GeometryAssociation,
    projection: PreparedBRepProjection,
    /,
    *,
    policy: HighOrderCurvingPolicy,
    periodic: tuple[PeriodicCoupling, ...] = (),
) -> HighOrderCurvingResult:
    """Curve an affine mesh to its B-Rep geometry with certified high-order nodes.

    ``association`` is the complete B-Rep vertex association of ``mesh``;
    ``periodic`` couplings pair vertices of ``mesh`` so the target-side high-order
    nodes are the declared isometry of the source-side nodes.
    """
    _require_inputs(mesh, association, projection, policy)
    straight = _straight_geometry(mesh, policy.degree)
    classes = _node_classes(mesh, straight, association, projection)
    blocks = _distortion_blocks(mesh, straight, policy.degree)
    straight_evidence = _evidence(mesh, straight, classes, projection, blocks, policy)
    if not np.all(classes.resolved):
        return HighOrderCurvingResult(
            HighOrderCurvingStatus.UNRESOLVED_ASSOCIATION,
            straight,
            straight,
            straight_evidence,
            None,
            (),
        )
    pairs = _periodic_nodes(mesh, straight, tuple(periodic))
    fixed = classes.fixed.copy()
    fixed[pairs.targets] = True
    elements, routes, _ = straight.resolve(mesh)
    layout = (
        {
            block.name: element
            for block, element in zip(mesh.blocks, elements, strict=True)
        },
        {block.name: route for block, route in zip(mesh.blocks, routes, strict=True)},
    )
    vertices = np.asarray(mesh.coordinates, dtype=np.float64)
    edges = _incidence_pairs(mesh, 0, 1)
    edge_vertices = edges[np.argsort(edges[:, 1], kind="stable"), 0].reshape(-1, 2)
    length = float(
        np.mean(
            np.linalg.norm(
                vertices[edge_vertices[:, 0]] - vertices[edge_vertices[:, 1]], axis=1
            )
        )
    )
    coordinates, constrained_rows, projectors = _project_constrained(
        projection, np.asarray(straight.coordinates, dtype=np.float64), classes, pairs
    )
    candidate = _evidence(
        mesh, CellGeometrySpec(*layout, coordinates), classes, projection, blocks, policy
    )
    accepted = (coordinates, candidate) if candidate.accepted else None
    minimizations = []
    for _ in range(policy.relaxation_rounds):
        energy = _CurvingEnergy(
            blocks=blocks,
            constrained_rows=jnp.asarray(constrained_rows, dtype=jnp.int32),
            feet=jnp.asarray(coordinates[constrained_rows]),
            normal_projectors=jnp.asarray(projectors),
            reference=jnp.asarray(coordinates),
            weights=jnp.asarray(
                (
                    policy.distortion_weight,
                    policy.tangent_weight / length**2,
                    policy.displacement_weight / length**2,
                ),
                dtype=jnp.float64,
            ),
            regularization=jnp.asarray(policy.regularization, dtype=jnp.float64),
        )
        relaxed = optimize_cell_geometry_coordinates(
            CellGeometrySpec(*layout, coordinates),
            energy,
            fixed_coordinates=fixed,
            termination=policy.termination,
        )
        minimizations.append(relaxed.minimization)
        moved = np.asarray(relaxed.coordinates, dtype=np.float64)
        if not np.all(np.isfinite(moved)):
            break
        coordinates, constrained_rows, projectors = _project_constrained(
            projection, moved, classes, pairs
        )
        candidate = _evidence(
            mesh,
            CellGeometrySpec(*layout, coordinates),
            classes,
            projection,
            blocks,
            policy,
        )
        if candidate.accepted:
            accepted = (coordinates, candidate)
    if accepted is not None:
        return HighOrderCurvingResult(
            HighOrderCurvingStatus.CURVED,
            CellGeometrySpec(*layout, accepted[0]),
            straight,
            accepted[1],
            candidate,
            tuple(minimizations),
        )
    status = (
        HighOrderCurvingStatus.ROLLED_BACK_RESIDUAL
        if candidate.certificate.all_certified
        else HighOrderCurvingStatus.ROLLED_BACK_INVALID
    )
    return HighOrderCurvingResult(
        status, straight, straight, straight_evidence, candidate, tuple(minimizations)
    )


__all__ = [
    "CurvedGeometryEvidence",
    "HighOrderCurvingPolicy",
    "HighOrderCurvingResult",
    "HighOrderCurvingStatus",
    "curve_cell_mesh",
    "verify_curved_geometry",
]
