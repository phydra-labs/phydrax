#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""High-order native CAD curving of mixed polynomial and rational cell maps.

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
between rounds, periodic target nodes follow their source nodes through the
declared isometry. Acceptance requires local validity, continuous source fidelity,
global embedding and, for volume meshes, domain/interface coverage. Nodal
residuals and sampled quality never substitute for these certificates. Relaxation
convergence status is retained independently; otherwise the result rolls back.
"""

from __future__ import annotations

from contextlib import nullcontext
from enum import StrEnum
from itertools import product
from typing import final, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax.discretization._cell_geometry as _dependency_phydrax_discretization__cell_geometry
import phydrax.discretization._cell_geometry_validity as _dependency_phydrax_discretization__cell_geometry_validity
import phydrax.geometry._approximate_source_coverage as approximate_source_coverage_owner
import phydrax.geometry._domain_boundary as domain_boundary_owner
import phydrax.geometry._domain_coverage as domain_coverage_owner
import phydrax.geometry._domain_planar_boundary as domain_planar_boundary_owner
import phydrax.geometry._embedding_state as _dependency_phydrax_geometry__embedding_state
import phydrax.geometry._global_embedding as global_embedding_owner
import phydrax.geometry._mapped_reference_domain as _dependency_phydrax_geometry__mapped_reference_domain
import phydrax.geometry._meshing_domain as _owner_phydrax_geometry__meshing_domain
import phydrax.geometry._source_boundary_records as source_boundary_records_owner
import phydrax.geometry._source_fidelity as source_fidelity_owner
import phydrax.geometry.brep._projection_contracts as _dependency_phydrax_geometry_brep__projection_contracts
import phydrax.geometry.brep._query_projection as query_projection_owner
import phydrax.meshing._association as _dependency_phydrax_meshing__association
import phydrax.meshing._association_routes as association_routes_owner
import phydrax.meshing._association_state as association_state_owner
import phydrax.meshing._boundary_layer as _dependency_phydrax_meshing__boundary_layer
import phydrax.meshing._coupling as _dependency_phydrax_meshing__coupling
import phydrax.meshing._entity_keys as entity_keys_owner
import phydrax.meshing._layer_curving as _dependency_phydrax_meshing__layer_curving
import phydrax.meshing._optimization as _dependency_phydrax_meshing__optimization

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import (
    nonnegative_finite_float,
    nonnegative_integer,
    positive_finite_float,
    positive_integer,
)
from ..discretization import (
    CellGeometrySpec,
    CellMesh,
    CellValidityCertificate,
    CellValidityPolicy,
    certify_cell_geometry_validity,
    reference_cell_topology,
)
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..ein import contract
from ..linalg import (
    determinant_small_linear,
    inverse_small_linear,
    SmallLinearSolvePlan,
)
from ..optim import MinimizationResult, OptimizationStatus, OptimizationTermination
from ..typing import checked, parse


_CURVING_DEGREES = (2, 3, 4, 6, 10)


import phydrax.discretization.fem as _owner_phydrax_discretization_fem
import phydrax.geometry._planar_embedding as _owner_phydrax_geometry__planar_embedding


class HighOrderCurvingStatus(StrEnum):
    """Outcome of one curving.

    ``CURVED``: the returned geometry is certified valid with every constrained
    node within the residual tolerance, and came from the CAD projection or a
    converged (or explicitly admitted non-converged) relaxation.
    ``ROLLED_BACK_INVALID``: no candidate was certified valid; the straight
    geometry is returned. ``ROLLED_BACK_RESIDUAL``: valid candidates missed the
    residual tolerance. ``ROLLED_BACK_NONCONVERGED``: the last candidate was valid
    within the tolerance but its relaxation did not converge and the policy does
    not accept valid non-converged relaxations. ``ROLLED_BACK_CERTIFICATION``:
    local validity and nodal residuals pass but global, continuous source or
    domain/interface certification fails or is unresolved.
    ``UNRESOLVED_ASSOCIATION``: a node has no unique B-Rep class; no curving runs.
    """

    CURVED = "curved"
    ROLLED_BACK_INVALID = "rolled_back_invalid"
    ROLLED_BACK_RESIDUAL = "rolled_back_residual"
    ROLLED_BACK_NONCONVERGED = "rolled_back_nonconverged"
    ROLLED_BACK_CERTIFICATION = "rolled_back_certification"
    UNRESOLVED_ASSOCIATION = "unresolved_association"


@final
class HighOrderCurvingPolicy(StrictModule, NonTrainableState):
    """Degree, objective weights, acceptance tolerances, and solver controls.

    ``degree`` is the geometry order (2, 3, 4, 6 or 10). ``relaxation_rounds`` bounds the
    optimize/re-project/certify rounds (0 accepts or rejects the projected
    configuration directly). Weights scale the mean sampled distortion, the
    squared tangent-space residual of constrained nodes, and the squared nodal
    displacement (the latter two relative to the mean straight edge length).
    ``regularization`` is the Escobar determinant regularization of the
    distortion, which keeps inverted starting configurations finite.
    ``residual_tolerance`` bounds constrained-node CAD residuals;
    ``fidelity_tolerance`` is required: it bounds continuous two-sided source
    approximation error in the source coordinate units, independently of nodal
    solver accuracy. Choose it from the declared source-deviation budget or a
    resolution-dependent approximation bound. A relaxation whose native
    minimization did not converge cannot replace the accepted geometry unless
    ``accept_valid_nonconverged_relaxation`` explicitly permits it; the next
    round still continues from its iterate.
    """

    degree: int = eqx.field(static=True)
    relaxation_rounds: int = eqx.field(static=True)
    distortion_weight: float = eqx.field(static=True)
    tangent_weight: float = eqx.field(static=True)
    displacement_weight: float = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    fidelity_tolerance: float = eqx.field(static=True)
    accept_valid_nonconverged_relaxation: bool = eqx.field(static=True)
    maximum_geometry_nodes: int = eqx.field(static=True)
    maximum_tabulation_entries: int = eqx.field(static=True)
    validity: CellValidityPolicy
    termination: OptimizationTermination
    policy_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        *,
        fidelity_tolerance: float,
        degree: int = 2,
        relaxation_rounds: int = 3,
        distortion_weight: float = 1.0,
        tangent_weight: float = 1.0e3,
        displacement_weight: float = 1.0e-3,
        regularization: float = 5.0e-2,
        residual_tolerance: float = 1.0e-7,
        validity: CellValidityPolicy | None = None,
        termination: OptimizationTermination | None = None,
        accept_valid_nonconverged_relaxation: bool = False,
        maximum_geometry_nodes: int = 1 << 20,
        maximum_tabulation_entries: int = 1 << 27,
    ) -> None:
        degree = positive_integer(degree, "degree")
        if degree not in _CURVING_DEGREES:
            raise ValueError(
                f"HighOrderCurvingPolicy.degree must be one of geometry degrees 2, 3, 4, 6 and 10; got {degree!r}"
            )
        relaxation_rounds = nonnegative_integer(relaxation_rounds, "relaxation_rounds")
        validity_ = (
            CellValidityPolicy(maximum_bernstein_nodes=1 << 15)
            if validity is None
            else validity
        )
        termination_ = (
            OptimizationTermination(maximum_steps=64)
            if termination is None
            else termination
        )
        if not isinstance(accept_valid_nonconverged_relaxation, bool):
            raise TypeError(
                f"HighOrderCurvingPolicy.accept_valid_nonconverged_relaxation must be bool; got {type(accept_valid_nonconverged_relaxation).__name__}"
            )
        for value, name in (
            (maximum_geometry_nodes, "maximum_geometry_nodes"),
            (maximum_tabulation_entries, "maximum_tabulation_entries"),
        ):
            value = positive_integer(value, name)
        self.maximum_geometry_nodes = int(maximum_geometry_nodes)
        self.maximum_tabulation_entries = int(maximum_tabulation_entries)
        self.degree = int(degree)
        self.relaxation_rounds = int(relaxation_rounds)
        self.distortion_weight = positive_finite_float(
            distortion_weight, "distortion_weight"
        )
        self.tangent_weight = nonnegative_finite_float(tangent_weight, "tangent_weight")
        self.displacement_weight = nonnegative_finite_float(
            displacement_weight, "displacement_weight"
        )
        self.regularization = positive_finite_float(regularization, "regularization")
        self.residual_tolerance = positive_finite_float(
            residual_tolerance, "residual_tolerance"
        )
        self.fidelity_tolerance = positive_finite_float(
            fidelity_tolerance, "fidelity_tolerance"
        )
        self.accept_valid_nonconverged_relaxation = accept_valid_nonconverged_relaxation
        self.validity = validity_
        self.termination = termination_
        self.policy_id = canonical_fingerprint(
            {
                "kind": "high-order-curving-policy",
                "degree": self.degree,
                "relaxation_rounds": self.relaxation_rounds,
                "maximum_geometry_nodes": self.maximum_geometry_nodes,
                "maximum_tabulation_entries": self.maximum_tabulation_entries,
                "distortion_weight": self.distortion_weight,
                "tangent_weight": self.tangent_weight,
                "displacement_weight": self.displacement_weight,
                "regularization": self.regularization,
                "residual_tolerance": self.residual_tolerance,
                "fidelity_tolerance": self.fidelity_tolerance,
                "accept_valid_nonconverged_relaxation": (
                    accept_valid_nonconverged_relaxation
                ),
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
    are the B-Rep class of each node's mesh entity, ``constrained`` marks
    nodes on B-Rep entities of lower dimension than the ambient space, and
    ``node_residuals`` their distance to that entity (zero for free nodes).
    ``minimum_scaled_jacobian`` and ``maximum_distortion`` are sampled at the
    Bernstein control-point lattice. ``accepted`` also requires global embedding,
    continuous source fidelity, applicable domain/interface coverage and every
    constrained nodal residual within tolerance.
    Piecewise-linear and mapped-reference domains retain region-measure and
    interface agreement. ``CadSourceDomain`` instead reports a certified
    region-approximation bound.
    """

    certificate: CellValidityCertificate
    embedding: global_embedding_owner.GlobalEmbeddingCertificate
    fidelity: source_fidelity_owner.SourceFidelityCertificate | None
    coverage: (
        domain_coverage_owner.DomainCoverageCertificate
        | approximate_source_coverage_owner.ApproximateSourceCoverageCertificate
        | None
    )
    certification_failures: tuple[str, ...] = eqx.field(static=True)
    node_dimensions: Array
    node_indices: Array
    node_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    source_projection_id: str = eqx.field(static=True)
    constrained: Array
    node_residuals: Array
    maximum_residual: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    minimum_scaled_jacobian: float = eqx.field(static=True)
    maximum_distortion: float = eqx.field(static=True)
    accepted: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    @checked
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
        *,
        embedding: global_embedding_owner.GlobalEmbeddingCertificate,
        fidelity: source_fidelity_owner.SourceFidelityCertificate | None,
        coverage: domain_coverage_owner.DomainCoverageCertificate
        | approximate_source_coverage_owner.ApproximateSourceCoverageCertificate
        | None,
        certification_failures: tuple[str, ...],
        node_occurrence_paths: tuple[tuple[str, ...], ...],
        source_revision: str,
        source_projection_id: str,
    ) -> None:
        if not isinstance(source_revision, str) or not isinstance(
            source_projection_id, str
        ):
            raise TypeError(
                f"CurvedGeometryEvidence.source must have string revision and projection identities; got kinds={(type(source_revision).__name__, type(source_projection_id).__name__)!r}"
            )
        if not source_revision or not source_projection_id:
            raise ValueError(
                f"CurvedGeometryEvidence.source must bind a non-empty revision and projection identity; got {(source_revision, source_projection_id)!r}"
            )
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
            raise ValueError(
                f"CurvedGeometryEvidence.nodes must have aligned dimension, index, constraint and residual vectors; got shapes={(dims.shape, indices.shape, mask.shape, residuals.shape)!r}"
            )
        if (
            not isinstance(node_occurrence_paths, tuple)
            or len(node_occurrence_paths) != dims.size
            or any(
                not isinstance(path, tuple)
                or any(not isinstance(name, str) or not name for name in path)
                for path in node_occurrence_paths
            )
        ):
            raise ValueError(
                f"CurvedGeometryEvidence.node_occurrence_paths must supply one valid occurrence path per node; got kind={type(node_occurrence_paths).__name__} for {dims.size} nodes"
            )
        maximum = float(np.max(residuals)) if residuals.size else 0.0
        self.certificate = certificate
        self.embedding = embedding
        self.fidelity = fidelity
        self.coverage = coverage
        self.certification_failures = certification_failures
        _host_node_dimensions = np.asarray(dims)
        _host_node_indices = np.asarray(indices)
        self.node_occurrence_paths = node_occurrence_paths
        self.source_revision = source_revision
        self.source_projection_id = source_projection_id
        _host_constrained = np.asarray(mask)
        _host_node_residuals = np.asarray(residuals)
        (
            self.node_dimensions,
            self.node_indices,
            self.constrained,
            self.node_residuals,
        ) = jax.device_put(
            (
                _host_node_dimensions,
                _host_node_indices,
                _host_constrained,
                _host_node_residuals,
            )
        )
        self.maximum_residual = maximum
        self.residual_tolerance = float(residual_tolerance)
        self.minimum_scaled_jacobian = float(minimum_scaled_jacobian)
        self.maximum_distortion = float(maximum_distortion)
        self.accepted = bool(
            certificate.all_certified
            and maximum <= residual_tolerance
            and embedding.status == "certified"
            and fidelity is not None
            and fidelity.status == "certified"
            and (coverage is None or coverage.status == "certified")
            and not certification_failures
        )
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "curved-geometry-evidence",
                "certificate": certificate.certificate_id,
                "embedding": embedding.certificate_id,
                "fidelity": None if fidelity is None else fidelity.certificate_id,
                "coverage": None if coverage is None else coverage.certificate_id,
                "certification_failures": certification_failures,
                "node_dimensions": array_tree_fingerprint(dims),
                "node_indices": array_tree_fingerprint(indices),
                "node_occurrence_paths": node_occurrence_paths,
                "source_revision": source_revision,
                "source_projection": source_projection_id,
                "node_residuals": array_tree_fingerprint(residuals),
                "residual_tolerance": self.residual_tolerance,
            }
        )


@final
class HighOrderCurvingResult(StrictModule, NonTrainableState):
    """Accepted curved geometry, or the straight geometry after a rollback.

    ``evidence`` certifies the returned ``geometry``; ``candidate_geometry`` and
    ``candidate`` retain the last evaluated curved map and its evidence even
    after rollback. ``minimizations`` carry the native optimizer evidence of every
    relaxation round and ``relaxation_statuses`` their termination statuses.
    ``accepted_round`` is ``0`` when the geometry is the CAD projection, ``k``
    when it is the candidate of relaxation round ``k``, and ``None`` unless the
    status is ``CURVED``.
    """

    status: HighOrderCurvingStatus = eqx.field(static=True)
    geometry: CellGeometrySpec
    straight: CellGeometrySpec
    evidence: CurvedGeometryEvidence
    candidate: CurvedGeometryEvidence | None
    candidate_geometry: CellGeometrySpec | None
    minimizations: tuple[MinimizationResult, ...]
    relaxation_statuses: tuple[OptimizationStatus, ...] = eqx.field(static=True)
    accepted_round: int | None = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        status: HighOrderCurvingStatus,
        geometry: CellGeometrySpec,
        straight: CellGeometrySpec,
        evidence: CurvedGeometryEvidence,
        candidate: CurvedGeometryEvidence | None,
        candidate_geometry: CellGeometrySpec | None,
        minimizations: tuple[MinimizationResult, ...],
        accepted_round: int | None,
        /,
    ) -> None:
        status = parse(status, HighOrderCurvingStatus, "HighOrderCurvingResult.status")
        if status is HighOrderCurvingStatus.CURVED and not evidence.accepted:
            raise ValueError(
                f"HighOrderCurvingResult.evidence must be accepted for a curved result; got accepted={evidence.accepted!r}"
            )
        if (status is HighOrderCurvingStatus.CURVED) != (accepted_round is not None):
            raise ValueError(
                f"HighOrderCurvingResult.accepted_round must be present exactly for a curved result; got status={status!r}, round={accepted_round!r}"
            )
        if accepted_round is not None and not 0 <= accepted_round <= len(minimizations):
            raise ValueError(
                f"HighOrderCurvingResult.accepted_round must name the projection or a relaxation round; got {accepted_round!r} for {len(minimizations)} rounds"
            )
        if (candidate is None) != (candidate_geometry is None):
            raise ValueError(
                "HighOrderCurvingResult.candidate_geometry must accompany candidate evidence; "
                f"got geometry={candidate_geometry is not None}, evidence={candidate is not None}."
            )
        evidence.certificate.require_bound(geometry)
        if candidate is not None and candidate_geometry is not None:
            candidate.certificate.require_bound(candidate_geometry)
        self.status = status
        self.geometry = geometry
        self.straight = straight
        self.evidence = evidence
        self.candidate = candidate
        self.candidate_geometry = candidate_geometry
        self.minimizations = tuple(minimizations)
        self.relaxation_statuses = tuple(
            OptimizationStatus(int(np.asarray(value.status)))
            for value in self.minimizations
        )
        self.accepted_round = accepted_round
        self.result_id = canonical_fingerprint(
            {
                "kind": "high-order-curving-result",
                "status": status.value,
                "geometry": geometry.geometry_layout_id,
                "coordinates": array_tree_fingerprint(np.asarray(geometry.coordinates)),
                "evidence": evidence.evidence_id,
                "candidate": None if candidate is None else candidate.evidence_id,
                "candidate_geometry": (
                    None
                    if candidate_geometry is None
                    else array_tree_fingerprint(
                        np.asarray(candidate_geometry.coordinates)
                    )
                ),
                "relaxation_statuses": [int(value) for value in self.relaxation_statuses],
                "accepted_round": accepted_round,
            }
        )


# -- geometry nodes ---------------------------------------------------------------


_WEIGHT_QUANTUM = float(2**40)


@checked
def admit_geometry_resources(
    mesh: CellMesh,
    degree: int,
    maximum_nodes: int,
    maximum_entries: int,
    maximum_bernstein_nodes: int,
    /,
    *,
    include_distortion: bool = True,
    include_layout_nodes: bool = True,
) -> None:
    """Refuse layout/reference/distortion work before allocating any new nodes."""
    if degree not in _CURVING_DEGREES:
        raise ValueError(
            f"admit_geometry_resources.degree must be a supported high-order coordinate degree; got {degree!r}"
        )
    nodes = mesh.coordinates.shape[0]
    retained_entries = 0
    reference_entries = 0
    for block in mesh.blocks:
        p = degree
        match block.cell_kind:
            case "triangle":
                count = (p + 1) * (p + 2) // 2
            case "tetrahedron":
                count = (p + 1) * (p + 2) * (p + 3) // 6
            case "quadrilateral":
                count = (p + 1) ** 2
            case "hexahedron":
                count = (p + 1) ** 3
            case "prism":
                count = (p + 1) ** 2 * (p + 2) // 2
            case "pyramid":
                count = (p + 1) * (p + 2) * (2 * p + 3) // 6
            case _:
                raise ValueError(
                    f"admit_geometry_resources.cell_kind must be a supported high-order coordinate family; got {block.cell_kind!r}"
                )
        if include_layout_nodes:
            nodes += block.cell_count * count
            if nodes > maximum_nodes:
                raise ValueError(
                    f"admit_geometry_resources.maximum_nodes must cover the upper-bound node count; got node resource limit exceeded with {nodes} nodes above {maximum_nodes}"
                )
        topology = reference_cell_topology(block.cell_kind)
        dimension = topology.dimension
        linear_count = len(topology.vertices)
        domain, degrees = (
            _dependency_phydrax_discretization__cell_geometry_validity.determinant_route(
                block.cell_kind, p, mesh.ambient_dimension > dimension
            )
        )
        certificate_nodes = _dependency_phydrax_discretization__cell_geometry_validity.bernstein_node_count(
            domain, degrees
        )
        if certificate_nodes > maximum_bernstein_nodes:
            raise ValueError(
                f"admit_geometry_resources.maximum_bernstein_nodes must cover the certificate node count; got certificate resource limit exceeded with {certificate_nodes} nodes above {maximum_bernstein_nodes} for {block.cell_kind} degree {p}"
            )
        order = max(
            _determinant_degree(block.cell_kind, p, mesh.ambient_dimension > dimension), 1
        )
        match block.cell_kind:
            case "triangle":
                samples = (order + 1) * (order + 2) // 2
            case "tetrahedron":
                samples = (order + 1) * (order + 2) * (order + 3) // 6
            case "prism":
                samples = (order + 1) ** 2 * (order + 2) // 2
            case "pyramid":
                samples = order * (order + 1) ** 2
            case _:
                samples = (order + 1) ** dimension
        reference_entries += count * count
        entries = reference_entries + max(
            count * count,
            count * linear_count * (dimension + 1),
        )
        if include_distortion:
            gradients = samples * count * dimension
            values = samples * count
            linear = samples * linear_count * (dimension + 1)
            jacobians = block.cell_count * samples * mesh.ambient_dimension * dimension
            inverse_entries = block.cell_count * samples * dimension * dimension
            entries = max(
                entries,
                reference_entries
                + retained_entries
                + gradients
                + max(
                    values + linear,
                    linear + jacobians + 2 * inverse_entries,
                ),
            )
            retained_entries += gradients + inverse_entries
        if entries > maximum_entries:
            raise ValueError(
                f"admit_geometry_resources.maximum_entries must cover aggregate tabulation entries; got tabulation resource limit exceeded with {entries} entries above {maximum_entries} for {block.cell_kind} degree {p}"
            )


@checked
def straight_geometry(
    mesh: CellMesh,
    degree: int,
    /,
    *,
    maximum_nodes: int = 1 << 20,
    maximum_entries: int = 1 << 27,
    maximum_bernstein_nodes: int = 1 << 15,
    source_geometry: CellGeometrySpec | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> CellGeometrySpec:
    """Return straight-sided Lagrange geometry with shared, deterministically ordered nodes.

    Local nodes follow the reference ordering of the discretization's Lagrange
    element. A node is identified by its mesh entity (sorted vertex global
    IDs) and its affine or multilinear weights on those vertices, so neighboring
    cells share it independently of their local edge and face orientation. Vertex
    nodes are the mesh vertex rows; the other nodes follow in (dimension, entity,
    weights) order.
    """
    admit_geometry_resources(
        mesh,
        degree,
        maximum_nodes,
        maximum_entries,
        maximum_bernstein_nodes,
        include_distortion=False,
    )
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    elements, routes, keys, positions, owners = {}, {}, [], [], []
    for block in mesh.blocks:
        element = (
            _dependency_phydrax_discretization__cell_geometry.coordinate_lagrange_element(
                block.cell_kind, degree
            )
        )
        topology = reference_cell_topology(block.cell_kind)
        cells = np.asarray(block.vertices, dtype=np.int64)
        weights, linear_gradients = (
            _dependency_phydrax_discretization__cell_geometry.coordinate_lagrange_element(
                block.cell_kind, 1
            ).tabulate(element.reference_nodes)
        )
        weights = np.asarray(weights, dtype=np.float64)
        del linear_gradients
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
                if source_geometry is None:
                    positions.append(
                        np.asarray(
                            contract(
                                "dv,cva->cda", weights[np.asarray(dofs)], points[cells]
                            )
                        ).reshape(-1, points.shape[1])
                    )
                owners.append((block.name, dofs, cells.shape[0]))
        elements[block.name] = element
        routes[block.name] = route
        del weights
    table, inverse = np.unique(np.concatenate(keys), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    coordinate_count = points.shape[0] + table.shape[0]
    cursor = 0
    for name, dofs, count in owners:
        size = count * len(dofs)
        routes[name][:, list(dofs)] = points.shape[0] + inverse[
            cursor : cursor + size
        ].reshape(count, len(dofs))
        cursor += size
    if source_geometry is None:
        coordinates = np.empty((coordinate_count, points.shape[1]), dtype=np.float64)
        coordinates[: points.shape[0]] = points
        coordinates[points.shape[0] + inverse] = np.concatenate(positions)
        geometry = CellGeometrySpec(elements, routes, coordinates)
    else:
        from ..discretization._exact_rational_geometry import (
            rational_geometry_lift,
            rational_geometry_lift_preparation_bounds,
        )

        # Resolve against the carrier before deriving interpolation actions, so
        # a foreign mesh or cell order cannot acquire the parent source binding.
        work, memory = (
            rational_geometry_lift_preparation_bounds(
                source_geometry, elements, coordinate_count
            )
            if budget is None
            else (0, 0)
        )
        with (
            nullcontext(budget)
            if budget is not None
            else _dependency_phydrax_discretization__cell_geometry.owned_coordinate_preparation(
                work,
                memory,
                maximum_geometry_queries=_dependency_phydrax_discretization__cell_geometry.cell_geometry_preparation_query_upper(
                    source_geometry
                ),
            )
        ) as admission:
            source_geometry.resolve(mesh, budget=admission)
            source = rational_geometry_lift(
                source_geometry,
                elements,
                routes,
                coordinate_count,
                budget=admission,
            )
            coordinates = source.representatives(budget=admission)
            geometry = CellGeometrySpec(
                elements, routes, coordinates, exact_source=source, budget=admission
            )
    return (
        geometry.with_periodic_source(mesh)
        if mesh.periodic_topology is not None
        else geometry
    )


def _entity_key_table(mesh: CellMesh, dimension: int, /) -> np.ndarray:
    """Sorted vertex-global-ID key of every entity row of ``dimension``."""
    pairs = association_routes_owner.incidence_pairs(mesh, 0, dimension)
    table = np.full((mesh.entity_set(dimension).count, 8), -1, dtype=np.int64)
    identifiers = np.asarray(mesh.vertex_global_ids, dtype=np.int64)[pairs[:, 0]]
    order = np.lexsort((identifiers, pairs[:, 1]))
    counts = np.bincount(pairs[:, 1], minlength=table.shape[0])
    if np.any(counts > table.shape[1]):
        raise ValueError(
            f"straight_geometry.entities must not exceed the supported arity; got maximum_arity={np.max(counts)!r}, supported_arity={table.shape[1]}"
        )
    offsets = np.cumsum(counts) - counts
    slots = np.arange(order.size) - np.repeat(offsets, counts)
    table[pairs[order, 1], slots] = identifiers[order]
    return table


@checked
def node_owners(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> tuple[np.ndarray, np.ndarray]:
    """Return the mesh entity ``(dimension, row)`` containing each geometry node."""

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
        cell_rows = association_state_owner.entity_rows(
            mesh, top, np.asarray(block.global_ids, dtype=np.int64)
        )
        # ty: ignore[unresolved-attribute]
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
                    keys = np.full((cells.shape[0], 8), -1, dtype=np.int64)
                    keys[:, : len(vertices)] = np.sort(
                        vertex_ids[cells[:, list(vertices)]], axis=1
                    )
                    owners = entity_keys_owner.key_rows(tables[dimension], keys)
                nodes = route_[:, list(dofs)]
                previous = rows[nodes]
                if np.any((previous >= 0) & (previous != owners[:, None])) or np.any(
                    (dims[nodes] >= 0) & (dims[nodes] != dimension)
                ):
                    raise ValueError(
                        f"node_owners.geometry must share nodes with consistent entity identities; got inconsistent sharing at dimension={dimension}"
                    )
                dims[nodes] = dimension
                rows[nodes] = owners[:, None]
    if np.any(rows < 0):
        raise ValueError(
            "node_owners.geometry must bind every node to one mesh entity; got unassigned nodes"
        )
    return dims, rows


class _NodeClasses(NamedTuple):
    dimensions: np.ndarray
    indices: np.ndarray
    occurrence_paths: tuple[tuple[str, ...], ...]
    resolved: np.ndarray
    constrained: np.ndarray
    fixed: np.ndarray


def _node_classes(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    association: association_state_owner.GeometryAssociation,
    projection: _dependency_phydrax_geometry_brep__projection_contracts.AbstractBRepProjection,
    /,
) -> _NodeClasses:
    owner_dims, owner_rows = node_owners(mesh, geometry)
    levels = _dependency_phydrax_meshing__association.mesh_entity_classes(
        mesh, association, projection
    )
    count = owner_dims.size
    dims = np.full((count,), -1, dtype=np.int64)
    indices = np.full((count,), -1, dtype=np.int64)
    status = np.full(
        (count,),
        _dependency_phydrax_geometry_brep__projection_contracts.BRepProjectionStatus.FAILED,
        dtype=np.int8,
    )
    paths: list[tuple[str, ...]] = [()] * count
    for level, classes in enumerate(levels):
        selected = owner_dims == level
        dims[selected] = classes.dimensions[owner_rows[selected]]
        indices[selected] = classes.indices[owner_rows[selected]]
        status[selected] = classes.status[owner_rows[selected]]
        for node in np.flatnonzero(selected):
            paths[node] = classes.paths[owner_rows[node]]
    constrained = (dims >= 0) & (dims <= 2) & (dims < mesh.ambient_dimension)
    return _NodeClasses(
        dims,
        indices,
        tuple(paths),
        np.isin(status, _dependency_phydrax_meshing__association.RESOLVED_STATUS),
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
        case "prism":
            return 3 * degree - 1
        case "pyramid":
            return 3 * degree - 1
        case _:
            raise ValueError(
                f"curve_cell_mesh.sample_lattice.cell_kind must have a supported curving lattice; got {kind!r}"
            )


def _control_lattice(kind: str, degree: int, embedded: bool, /) -> np.ndarray:
    """Bernstein control-point (Greville) lattice of the Jacobian determinant."""
    order = max(_determinant_degree(kind, degree, embedded), 1)
    dimension = reference_cell_topology(kind).dimension
    grid = np.asarray(tuple(product(range(order + 1), repeat=dimension)), np.float64)
    if kind in ("triangle", "tetrahedron"):
        grid = grid[np.sum(grid, axis=1) <= order]
    elif kind == "prism":
        grid = grid[np.sum(grid[:, :2], axis=1) <= order]
    samples = grid / order
    if kind == "pyramid":
        # The objective is sampled, not a certificate. Avoid the collapsed
        # apex where the rational reference gradient has directional limits.
        samples = samples[samples[:, 2] < 1.0]
        height = samples[:, 2:3]
        samples[:, :2] = samples[:, :2] * (1.0 - height) + 0.5 * height
    return samples


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
        # ty: ignore[unresolved-attribute]
        _, gradients = element.tabulate(samples)
        _, linear = (
            _dependency_phydrax_discretization__cell_geometry.coordinate_lagrange_element(
                block.cell_kind, 1
            ).tabulate(samples)
        )
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
            raise ValueError(
                "curve_cell_mesh.reference_geometry must have nondegenerate reference elements; got a degenerate element"
            )
        blocks.append(
            _BlockDistortion(
                routes=jnp.asarray(route, dtype=jnp.int32),
                gradients=jnp.asarray(gradients, dtype=jnp.float64),
                target_inverses=inverse.value,
                plan=plan,
                embedded=embedded,
            )
        )
        del samples, gradients, linear, straight, target, inverse, plan, _
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
    projection: _dependency_phydrax_geometry_brep__projection_contracts.AbstractBRepProjection,
    coordinates: np.ndarray,
    classes: _NodeClasses,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Closest points, residuals, unit normals-space projectors of constrained nodes."""
    rows = np.flatnonzero(classes.constrained)
    ambient = coordinates.shape[1]
    residuals = np.zeros((coordinates.shape[0],))
    result = projection.project(
        coordinates[rows],
        classes.dimensions[rows],
        classes.indices[rows],
        occurrence_paths=tuple(classes.occurrence_paths[row] for row in rows),
    )
    failed = ~np.isin(
        np.asarray(result.status),
        _dependency_phydrax_meshing__association.RESOLVED_STATUS,
    )
    failed |= np.asarray(
        [
            path != classes.occurrence_paths[row]
            for row, path in zip(rows, result.source_occurrence_paths, strict=True)
        ],
        dtype=np.bool_,
    )
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
    projection: _dependency_phydrax_geometry_brep__projection_contracts.AbstractBRepProjection,
    blocks: tuple[_BlockDistortion, ...],
    policy: HighOrderCurvingPolicy,
    /,
    *,
    source: source_boundary_records_owner.SourceBoundaryQuery | None = None,
    domain: domain_coverage_owner.PiecewiseLinearDomain
    | _dependency_phydrax_geometry__mapped_reference_domain.MappedReferenceDomain
    | domain_planar_boundary_owner.CadSourceDomain
    | None = None,
    cell_regions: np.ndarray | None = None,
    certificate_limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits
    | None = None,
) -> CurvedGeometryEvidence:
    coordinates = np.asarray(geometry.coordinates, dtype=np.float64)
    _, _, residuals, _ = _node_residuals(projection, coordinates, classes)
    scaled, distortion = _sampled_quality(blocks, coordinates)
    local = certify_cell_geometry_validity(geometry, mesh=mesh, policy=policy.validity)
    embedding = global_embedding_owner.certify_global_embedding(
        mesh, geometry, local, limits=certificate_limits
    )
    failures = [] if embedding.status == "certified" else ["global_embedding"]
    fidelity = None
    if source is None:
        failures.append("source_fidelity")
    else:
        if source.source_revision != projection.source_revision:
            raise ValueError(
                f"curve_cell_mesh.source must use the projection revision; got source_revision={source.source_revision!r}, projection_revision={projection.source_revision!r}"
            )
        fidelity = source_fidelity_owner.certify_source_fidelity(
            mesh,
            geometry,
            source,
            tolerance=policy.fidelity_tolerance,
            limits=certificate_limits,
        )
        if fidelity.status != "certified":
            failures.append("source_fidelity")
    coverage = None
    if domain is None and isinstance(
        source, source_fidelity_owner.MappedDomainBoundarySource
    ):
        declared = source.domain
        if not isinstance(
            declared,
            _dependency_phydrax_geometry__mapped_reference_domain.MappedReferenceDomain,
        ):
            raise TypeError(
                f"curve_cell_mesh.source.domain must be MappedReferenceDomain; got {type(declared).__name__}"
            )
        domain = declared
        cell_regions = np.asarray(source.cell_regions, dtype=np.int64)
    if mesh.topological_dimension == mesh.ambient_dimension:
        if cell_regions is None:
            failures.append("domain_interface_coverage")
        else:
            match domain:
                case domain_planar_boundary_owner.CadSourceDomain():
                    if fidelity is not None:
                        coverage = approximate_source_coverage_owner.certify_approximate_source_coverage(
                            mesh,
                            geometry,
                            domain,
                            cell_regions,
                            embedding=embedding,
                            fidelity=fidelity,
                            limits=certificate_limits,
                        )
                case (
                    domain_coverage_owner.PiecewiseLinearDomain()
                    | _dependency_phydrax_geometry__mapped_reference_domain.MappedReferenceDomain()
                ):
                    coverage = domain_coverage_owner.certify_domain_coverage(
                        mesh,
                        geometry,
                        domain,
                        cell_regions,
                        embedding=embedding,
                        limits=certificate_limits,
                    )
                case None:
                    pass
                case _:
                    raise TypeError(
                        "Curving.domain must be a declared CAD, piecewise-linear or mapped "
                        f"domain; got {type(domain).__name__}."
                    )
            if coverage is None or coverage.status != "certified":
                failures.append("domain_interface_coverage")
    return CurvedGeometryEvidence(
        local,
        classes.dimensions,
        classes.indices,
        classes.constrained,
        residuals,
        policy.residual_tolerance,
        scaled,
        distortion,
        embedding=embedding,
        fidelity=fidelity,
        coverage=coverage,
        certification_failures=tuple(failures),
        node_occurrence_paths=classes.occurrence_paths,
        source_revision=projection.source_revision,
        source_projection_id=projection.projection_id,
    )


def _continuous_source(
    projection: _dependency_phydrax_geometry_brep__projection_contracts.AbstractBRepProjection,
    source: source_boundary_records_owner.SourceBoundaryQuery | None,
    /,
) -> source_boundary_records_owner.SourceBoundaryQuery | None:
    """Admit a native projection's revision-bound continuous source."""

    if source is not None:
        return source
    match projection:
        case query_projection_owner.NativeBRepProjection(embedding=None):
            domain = _owner_phydrax_geometry__meshing_domain.MeshingDomain.from_brep(
                projection.query.model
            )
            return domain_boundary_owner.MeshingDomainBoundarySource(
                domain, tuple(range(len(domain.patches)))
            )
        case query_projection_owner.NativeBRepProjection(
            embedding=_owner_phydrax_geometry__planar_embedding.PlanarEmbedding() as embedding
        ):
            return domain_planar_boundary_owner.EmbeddedPlanarBoundarySource(
                projection.query.model, embedding
            )
        case _dependency_phydrax_geometry_brep__projection_contracts.AbstractBRepProjection():
            return None
        case _:
            raise TypeError(
                "Curving.projection must be an AbstractBRepProjection; "
                f"got {type(projection).__name__}."
            )


@checked
def verify_curved_geometry(
    geometry: CellGeometrySpec,
    mesh: CellMesh,
    association: association_state_owner.GeometryAssociation,
    projection: _dependency_phydrax_geometry_brep__projection_contracts.AbstractBRepProjection,
    /,
    *,
    policy: HighOrderCurvingPolicy,
    source: source_boundary_records_owner.SourceBoundaryQuery | None = None,
    domain: domain_coverage_owner.PiecewiseLinearDomain
    | _dependency_phydrax_geometry__mapped_reference_domain.MappedReferenceDomain
    | domain_planar_boundary_owner.CadSourceDomain
    | None = None,
    cell_regions: np.ndarray | None = None,
    certificate_limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits
    | None = None,
) -> CurvedGeometryEvidence:
    """Certify an existing high-order geometry (e.g. a Gmsh high-order output).

    Nodes are classified through their mesh entities and the vertex
    ``association``; the evidence carries the Bernstein validity certificate and
    the CAD residual of every constrained node. Nothing is moved.
    Native world-space and embedded-planar projections supply their revision-bound
    carrier when ``source`` is omitted; acceptance requires a continuous proof.
    """
    if geometry.coordinates.shape[0] > policy.maximum_geometry_nodes:
        raise ValueError(
            f"verify_curved_geometry.geometry must fit maximum_geometry_nodes; got node resource limit exceeded with {geometry.coordinates.shape[0]} nodes above {policy.maximum_geometry_nodes}"
        )
    # ty: ignore[unresolved-attribute]
    degree = max(int(element.degree) for element in geometry.elements)
    admit_geometry_resources(
        mesh,
        degree,
        policy.maximum_geometry_nodes,
        policy.maximum_tabulation_entries,
        policy.validity.maximum_bernstein_nodes,
        include_layout_nodes=False,
    )
    source = _continuous_source(projection, source)
    classes = _node_classes(mesh, geometry, association, projection)
    blocks = _distortion_blocks(mesh, geometry, degree)
    return _evidence(
        mesh,
        geometry,
        classes,
        projection,
        blocks,
        policy,
        source=source,
        domain=domain,
        cell_regions=cell_regions,
        certificate_limits=certificate_limits,
    )


# -- curving ----------------------------------------------------------------------


class _PeriodicNodes(NamedTuple):
    """Target nodes and the isometry mapping their paired source nodes onto them."""

    sources: np.ndarray
    targets: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray


def _quotient_geometry_nodes(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> _PeriodicNodes:
    """Lower the FE owner's quotient nodal numbering to geometry-node isometries."""

    topology = mesh.periodic_topology
    if topology is None:
        raise ValueError(
            "periodic_nodes.mesh.periodic_topology must be supplied for quotient geometry nodes; got None"
        )
    elements, routes, _ = geometry.resolve(mesh)
    if any(
        not isinstance(element, _owner_phydrax_discretization_fem.FiniteElementSpec)
        for element in elements
    ):
        raise ValueError(
            f"periodic_nodes.geometry.elements must be nodal finite elements; got kinds={tuple(type(element).__name__ for element in elements)!r}"
        )
    fields = tuple(
        element
        for element in elements
        if isinstance(element, _owner_phydrax_discretization_fem.FiniteElementSpec)
    )
    dofs = _owner_phydrax_discretization_fem.FiniteElementDofMap(
        mesh, fields, coordinate_spec=geometry
    )
    if dofs.coordinate_gather is None:
        raise RuntimeError(
            "periodic_nodes.coordinate_gather must contain representative nodes; got None"
        )
    flattened = np.concatenate([np.asarray(route).reshape(-1) for route in routes])
    representatives = flattened[np.asarray(dofs.coordinate_gather)]
    quotient = np.full((geometry.coordinates.shape[0],), -1, dtype=np.int64)
    for route, field_route in zip(routes, dofs.cell_dofs, strict=True):
        nodes = np.asarray(route)
        indices = np.asarray(field_route)
        if nodes.shape != indices.shape:
            raise ValueError(
                f"periodic_nodes.geometry must align coordinate and field nodal layouts; got coordinate_shape={nodes.shape}, field_shape={indices.shape}"
            )
        previous = quotient[nodes]
        if np.any((previous >= 0) & (previous != indices)):
            raise ValueError(
                "periodic_nodes.geometry must share nodes with one quotient identity; got inconsistent identities"
            )
        quotient[nodes] = indices
    if np.any(quotient < 0):
        raise ValueError(
            "periodic_nodes.geometry must bind every coordinate node to a quotient identity; got unassigned nodes"
        )
    sources = representatives[quotient]
    targets = np.flatnonzero(sources != np.arange(sources.size))
    dimensions, owners = node_owners(mesh, geometry)
    matrices = np.empty(
        (targets.size, mesh.ambient_dimension + 1, mesh.ambient_dimension + 1),
        dtype=np.float64,
    )
    for degree in range(mesh.topological_dimension + 1):
        selected = dimensions[targets] == degree
        if np.any(selected):
            matrices[selected] = topology.orbit_isometries(degree)[
                owners[targets[selected]]
            ]
    return _PeriodicNodes(
        sources[targets], targets, matrices[:, :-1, :-1], matrices[:, :-1, -1]
    )


@checked
def periodic_nodes(
    mesh: CellMesh,
    straight: CellGeometrySpec,
    periodic: tuple[_dependency_phydrax_meshing__coupling.PeriodicCoupling, ...],
    /,
) -> _PeriodicNodes:
    """Pair the geometry nodes of every periodic coupling through its isometry.

    A node belongs to a side when every vertex of its mesh entity is in
    the side's vertex scope; sides are paired on the straight positions.
    """
    owner_dims, owner_rows = node_owners(mesh, straight)
    coordinates = np.asarray(straight.coordinates, dtype=np.float64)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    ambient = mesh.ambient_dimension
    incidences = tuple(
        association_routes_owner.incidence_pairs(mesh, 0, dimension)
        for dimension in range(mesh.topological_dimension + 1)
    )
    native = (
        _quotient_geometry_nodes(mesh, straight)
        if mesh.periodic_topology is not None
        else None
    )
    sources, targets, rotations, translations = [], [], [], []
    if native is not None:
        sources.append(native.sources)
        targets.append(native.targets)
        rotations.append(native.rotations)
        translations.append(native.translations)
    for coupling in periodic:
        if not isinstance(
            coupling, _dependency_phydrax_meshing__coupling.PeriodicCoupling
        ):
            raise TypeError(
                f"periodic_nodes.periodic must contain PeriodicCoupling values; got {type(coupling).__name__}"
            )
        scopes = (coupling.source_scope, coupling.target_scope)
        if any(
            scope.entity_dimension != 0
            or scope.entity_set_id != mesh.entity_set(0).entity_set_id
            for scope in scopes
        ):
            raise ValueError(
                "periodic_nodes.periodic must pair vertices of the curved mesh; got coupling scopes from another entity set or dimension"
            )
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
        raise ValueError(
            f"periodic_nodes.periodic must not share target nodes; got {target.size} targets including duplicates"
        )
    return _PeriodicNodes(
        np.concatenate(sources),
        target,
        np.concatenate(rotations),
        np.concatenate(translations),
    )


@checked
def apply_periodic(coordinates: np.ndarray, periodic: _PeriodicNodes, /) -> np.ndarray:
    result = coordinates.copy()
    result[periodic.targets] = (
        np.asarray(
            contract("pij,pj->pi", periodic.rotations, coordinates[periodic.sources])
        )
        + periodic.translations
    )
    return result


def _project_constrained(
    projection: _dependency_phydrax_geometry_brep__projection_contracts.AbstractBRepProjection,
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
    return apply_periodic(result, periodic), rows, projectors


@final
class _ProjectedCurvingState(StrictModule, NonTrainableState):
    pairs: _PeriodicNodes
    layer_nodes: _dependency_phydrax_meshing__layer_curving.LayerCurvingNodes | None
    fixed: np.ndarray
    length: float = eqx.field(static=True)
    reference_coordinates: np.ndarray
    coordinates: np.ndarray
    constrained_rows: np.ndarray
    projectors: np.ndarray
    geometry: CellGeometrySpec


def _prepare_projected_curving(
    mesh: CellMesh,
    straight: CellGeometrySpec,
    classes: _NodeClasses,
    projection: _dependency_phydrax_geometry_brep__projection_contracts.AbstractBRepProjection,
    periodic: tuple[_dependency_phydrax_meshing__coupling.PeriodicCoupling, ...],
    layers: _dependency_phydrax_meshing__boundary_layer.BoundaryLayerMesh | None,
) -> _ProjectedCurvingState:
    pairs = periodic_nodes(mesh, straight, tuple(periodic))
    layer_nodes = (
        None
        if layers is None
        else _dependency_phydrax_meshing__layer_curving.prepare_layer_curving_nodes(
            mesh, straight, layers
        )
    )
    fixed = classes.fixed.copy()
    fixed[pairs.targets] = True
    if layer_nodes is not None:
        fixed[layer_nodes.nodes[layer_nodes.nodes != layer_nodes.owners]] = True
    vertices = np.asarray(mesh.coordinates, dtype=np.float64)
    edges = association_routes_owner.incidence_pairs(mesh, 0, 1)
    edge_vertices = edges[np.argsort(edges[:, 1], kind="stable"), 0].reshape(-1, 2)
    length = float(
        np.mean(
            np.linalg.norm(
                vertices[edge_vertices[:, 0]] - vertices[edge_vertices[:, 1]], axis=1
            )
        )
    )
    reference_coordinates = np.asarray(straight.coordinates, dtype=np.float64)
    coordinates, constrained_rows, projectors = _project_constrained(
        projection, reference_coordinates, classes, pairs
    )
    if layer_nodes is not None:
        _dependency_phydrax_meshing__layer_curving.transport_layer_curvature(
            coordinates, reference_coordinates, layer_nodes
        )
        coordinates = apply_periodic(coordinates, pairs)
    candidate_geometry = straight.with_coordinates(coordinates)
    return _ProjectedCurvingState(
        pairs,
        layer_nodes,
        fixed,
        length,
        reference_coordinates,
        coordinates,
        constrained_rows,
        projectors,
        candidate_geometry,
    )


def _curving_round_energy(
    blocks: tuple[_BlockDistortion, ...],
    coordinates: np.ndarray,
    constrained_rows: np.ndarray,
    projectors: np.ndarray,
    length: float,
    policy: HighOrderCurvingPolicy,
) -> _CurvingEnergy:
    """Freeze the current foot points and tangent penalties for one relaxation."""
    return _CurvingEnergy(
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


# Keep projection, sequential relaxation acceptance, and rollback evidence in order.
@checked
def curve_cell_mesh(
    mesh: CellMesh,
    association: association_state_owner.GeometryAssociation,
    projection: _dependency_phydrax_geometry_brep__projection_contracts.AbstractBRepProjection,
    /,
    *,
    policy: HighOrderCurvingPolicy,
    periodic: tuple[_dependency_phydrax_meshing__coupling.PeriodicCoupling, ...] = (),
    source: source_boundary_records_owner.SourceBoundaryQuery | None = None,
    domain: domain_coverage_owner.PiecewiseLinearDomain
    | _dependency_phydrax_geometry__mapped_reference_domain.MappedReferenceDomain
    | domain_planar_boundary_owner.CadSourceDomain
    | None = None,
    cell_regions: np.ndarray | None = None,
    certificate_limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits
    | None = None,
    layers: _dependency_phydrax_meshing__boundary_layer.BoundaryLayerMesh | None = None,
) -> HighOrderCurvingResult:
    """Curve an affine mesh to its B-Rep geometry with certified high-order nodes.

    ``association`` binds every B-Rep vertex of ``mesh``;
    ``periodic`` couplings pair vertices of ``mesh`` so the target-side high-order
    nodes are the declared isometry of the source-side nodes. A mesh-bound
    ``PeriodicMeshTopology`` instead supplies quotient node identity through the
    finite-element nodal numbering and declared entity isometries;
    winding-distinct entities are never merged by a positional search.
    Native world-space and embedded-planar projections supply their revision-bound
    continuous carrier when ``source`` is omitted. Nodal projection alone never
    supplies the source-fidelity certificate. Full-dimensional meshes require
    explicit ``domain`` and ``cell_regions``: domain coverage for piecewise-linear
    or mapped-reference domains, approximate region coverage for ``CadSourceDomain``.
    ``layers`` retains the prepared prism columns. Their wall curvature
    displacement is transported identically through every axial level, so the
    prepared physical interval vectors survive both projection and relaxation.
    Source fidelity, validity, global embedding, and declared domain coverage
    still certify the resulting map independently.
    """
    admit_geometry_resources(
        mesh,
        policy.degree,
        policy.maximum_geometry_nodes,
        policy.maximum_tabulation_entries,
        policy.validity.maximum_bernstein_nodes,
    )
    source = _continuous_source(projection, source)
    straight = straight_geometry(
        mesh,
        policy.degree,
        maximum_nodes=policy.maximum_geometry_nodes,
        maximum_entries=policy.maximum_tabulation_entries,
        maximum_bernstein_nodes=policy.validity.maximum_bernstein_nodes,
    )
    classes = _node_classes(mesh, straight, association, projection)
    blocks = _distortion_blocks(mesh, straight, policy.degree)
    if not np.all(classes.resolved):
        straight_evidence = _evidence(
            mesh,
            straight,
            classes,
            projection,
            blocks,
            policy,
            source=source,
            domain=domain,
            cell_regions=cell_regions,
            certificate_limits=certificate_limits,
        )
        return HighOrderCurvingResult(
            HighOrderCurvingStatus.UNRESOLVED_ASSOCIATION,
            straight,
            straight,
            straight_evidence,
            None,
            None,
            (),
            None,
        )
    projected = _prepare_projected_curving(
        mesh, straight, classes, projection, periodic, layers
    )
    pairs, layer_nodes, fixed = projected.pairs, projected.layer_nodes, projected.fixed
    length, reference_coordinates = projected.length, projected.reference_coordinates
    coordinates, constrained_rows, projectors = (
        projected.coordinates,
        projected.constrained_rows,
        projected.projectors,
    )
    candidate_geometry = projected.geometry
    candidate = _evidence(
        mesh,
        candidate_geometry,
        classes,
        projection,
        blocks,
        policy,
        source=source,
        domain=domain,
        cell_regions=cell_regions,
        certificate_limits=certificate_limits,
    )
    # The CAD projection needs no relaxation; it is accepted on its own when valid.
    accepted = (candidate_geometry, candidate, 0) if candidate.accepted else None
    minimizations = []
    for round_ in range(1, policy.relaxation_rounds + 1):
        energy = _curving_round_energy(
            blocks, coordinates, constrained_rows, projectors, length, policy
        )
        relaxed = (
            _dependency_phydrax_meshing__optimization.optimize_cell_geometry_coordinates(
                straight.with_coordinates(coordinates),
                energy,
                fixed_coordinates=fixed,
                termination=policy.termination,
            )
        )
        minimizations.append(relaxed.minimization)
        moved = np.asarray(relaxed.coordinates, dtype=np.float64)
        if not np.all(np.isfinite(moved)):
            break
        coordinates, constrained_rows, projectors = _project_constrained(
            projection, moved, classes, pairs
        )
        if layer_nodes is not None:
            _dependency_phydrax_meshing__layer_curving.transport_layer_curvature(
                coordinates, reference_coordinates, layer_nodes
            )
            coordinates = apply_periodic(coordinates, pairs)
        candidate_geometry = straight.with_coordinates(coordinates)
        candidate = _evidence(
            mesh,
            candidate_geometry,
            classes,
            projection,
            blocks,
            policy,
            source=source,
            domain=domain,
            cell_regions=cell_regions,
            certificate_limits=certificate_limits,
        )
        if candidate.accepted and (
            relaxed.converged or policy.accept_valid_nonconverged_relaxation
        ):
            accepted = (candidate_geometry, candidate, round_)
    if accepted is not None:
        return HighOrderCurvingResult(
            HighOrderCurvingStatus.CURVED,
            accepted[0],
            straight,
            accepted[1],
            candidate,
            candidate_geometry,
            tuple(minimizations),
            accepted[2],
        )
    # Without an accepted geometry, a valid last candidate within the residual
    # tolerance was refused only because its relaxation did not converge.
    if candidate.accepted:
        status = HighOrderCurvingStatus.ROLLED_BACK_NONCONVERGED
    elif candidate.certificate.all_certified and (
        candidate.maximum_residual <= policy.residual_tolerance
    ):
        status = HighOrderCurvingStatus.ROLLED_BACK_CERTIFICATION
    elif candidate.certificate.all_certified:
        status = HighOrderCurvingStatus.ROLLED_BACK_RESIDUAL
    else:
        status = HighOrderCurvingStatus.ROLLED_BACK_INVALID
    straight_evidence = _evidence(
        mesh,
        straight,
        classes,
        projection,
        blocks,
        policy,
        source=source,
        domain=domain,
        cell_regions=cell_regions,
        certificate_limits=certificate_limits,
    )
    return HighOrderCurvingResult(
        status,
        straight,
        straight,
        straight_evidence,
        candidate,
        candidate_geometry,
        tuple(minimizations),
        None,
    )


__all__ = [
    "CurvedGeometryEvidence",
    "HighOrderCurvingPolicy",
    "HighOrderCurvingResult",
    "HighOrderCurvingStatus",
    "curve_cell_mesh",
    "verify_curved_geometry",
]
