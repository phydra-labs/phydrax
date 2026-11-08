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
tangent space at their current foot points. Each round relaxes only the coupled
star of the cells the latest certified candidate does not prove valid, solved by
the native Newton trust region on displacements in units of the mean straight
edge; a candidate proving every cell valid ends relaxation. Constrained nodes
are re-projected between rounds, periodic target nodes follow their source
nodes through the declared isometry, and a candidate is accepted only when the
Bernstein certificate proves every cell valid, global embedding is certified,
every constrained node lies within the residual tolerance, and any supplied
retained source certification passes; relaxations must also converge unless the
policy explicitly accepts valid non-converged relaxations. Otherwise the result
rolls back.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from fractions import Fraction
from itertools import product
from typing import cast, final, NamedTuple, Protocol, runtime_checkable, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import positive_finite_float
from ..discretization import (
    CellGeometrySpec,
    CellMesh,
    CellValidityCertificate,
    CellValidityPolicy,
    CellValidityStatus,
    reference_cell_topology,
)
from ..discretization._cell_geometry import (
    BarycentricCellGeometryElement,
    CellGeometryElement,
    PerCellRestrictedCellGeometryElement,
    PolynomialComposedCellGeometryElement,
    RationalComposedCellGeometryElement,
    RestrictedCellGeometryElement,
    ScalarCoordinateElement,
)
from ..discretization._cell_geometry_transition_types import CellGeometryTransition
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    ComputedCellValidity,
)
from ..discretization._coordinate_enclosure import CoordinateSourceBank
from ..discretization._exact_algebraic_geometry import ExactScalar, RadicalElement
from ..discretization._exact_coordinate_bank import ExactCoordinateBank
from ..discretization.fem import (
    canonical_nodal_entity_identity,
    lagrange_element,
    lagrange_reference_node_data,
)
from ..ein import contract
from ..geometry._global_embedding import (
    certify_global_embedding,
    GlobalEmbeddingCertificate,
)
from ..geometry.brep._projection_contracts import (
    AbstractBRepProjection,
    BRepProjectionStatus,
)
from ..linalg import (
    determinant_small_linear,
    inverse_small_linear,
    SmallLinearSolvePlan,
)
from ..optim import (
    MinimizationResult,
    NewtonTrustRegion,
    OptimizationStatus,
    OptimizationTermination,
)
from ..typing import checked
from ._adaptation import (
    _extend_prior_volume_material_chain,
    _identity_lineage,
    _identity_reference_maps_for_transition,
    _renew_transition_certification_inputs,
    _retarget_source_boundary_chart_witness,
    _transition_geometry_record,
)
from ._association import (
    _entity_rows,
    _incidence_pairs,
    _mesh_entity_classes,
    _RESOLVED_STATUS,
    GeometryAssociation,
)
from ._canonical import certify_cell_mesh
from ._certification import MeshCertificationReport
from ._contracts import MeshingFailure
from ._coupling import PeriodicCoupling
from ._lineage import CellMeshTransition, MeshTransitionKind
from ._native_construction import NativeMeshOrganization
from ._optimization import optimize_cell_geometry_coordinates
from ._result import CellMeshingResult


if TYPE_CHECKING:
    from ..geometry.brep._projection import _ProjectionPreparation


@runtime_checkable
class _NodalTabulatingGeometryElement(Protocol):
    """Reference element carrying nodal ownership and point tabulation."""

    reference_nodes: Array
    entity_dofs: tuple[tuple[tuple[int, ...], ...], ...]

    def tabulate(self, points: ArrayLike, /) -> tuple[Array, Array]: ...


@runtime_checkable
class _NodalReferenceElement(Protocol):
    """Underlying nodal reference basis used to classify coordinate rows."""

    reference_nodes: Array
    cell_kind: str
    entity_dofs: tuple[tuple[tuple[int, ...], ...], ...]


def _nodal_reference_element(element: CellGeometryElement, /) -> _NodalReferenceElement:
    if not isinstance(element, ScalarCoordinateElement):
        raise TypeError("Curving requires a canonical scalar coordinate element.")
    while not isinstance(element, _NodalReferenceElement):
        if isinstance(element, PerCellRestrictedCellGeometryElement):
            element = element.source_element
        elif isinstance(element, BarycentricCellGeometryElement):
            element = element.source_element
        elif isinstance(element, RestrictedCellGeometryElement):
            element = element.source_element
        elif isinstance(element, PolynomialComposedCellGeometryElement):
            element = element.source_element
        elif isinstance(element, RationalComposedCellGeometryElement):
            element = element.source_element
        else:
            raise ValueError(
                f"Curving cannot classify non-nodal coordinate element {type(element).__name__}."
            )
    return element


_CURVED_KINDS = (
    "triangle",
    "quadrilateral",
    "tetrahedron",
    "simplex:3",
    "prism",
    "pyramid",
    "hexahedron",
)


class HighOrderCurvingStatus(StrEnum):
    """Outcome of one curving.

    ``CURVED``: the returned geometry is certified valid with every constrained
    node within the residual tolerance, and came from the CAD projection or a
    converged (or explicitly admitted non-converged) relaxation.
    A rollback returns the straight geometry and names the first refusal of the
    last candidate, in this order. ROLLED_BACK_INVALID: some cell is not
    certified valid. ROLLED_BACK_EMBEDDING: cells are locally valid but do not
    have a certified global embedding. ROLLED_BACK_RESIDUAL: a constrained node
    misses the residual tolerance. ROLLED_BACK_SOURCE_ACCEPTANCE: the
    source-bound acceptance of the candidate was not renewed.
    ROLLED_BACK_NONCONVERGED: the last candidate passed every check but its
    relaxation did not converge.
    """

    CURVED = "curved"
    ROLLED_BACK_INVALID = "rolled_back_invalid"
    ROLLED_BACK_RESIDUAL = "rolled_back_residual"
    ROLLED_BACK_NONCONVERGED = "rolled_back_nonconverged"
    ROLLED_BACK_EMBEDDING = "rolled_back_embedding"
    ROLLED_BACK_SOURCE_ACCEPTANCE = "rolled_back_source_acceptance"
    UNRESOLVED_ASSOCIATION = "unresolved_association"


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
    configuration directly); round ``k`` relaxes the cells the latest certified
    candidate does not prove valid together with ``k`` rings of node-sharing
    cells, and no round runs once a candidate proves every cell valid. Weights
    scale the mean sampled distortion of the relaxed cells, the squared
    tangent-space residual of constrained nodes, and the squared nodal
    displacement (the latter two relative to the mean straight edge length).
    ``termination`` controls the native Newton trust region, whose parameters are
    nodal displacements in units of the mean straight edge length.
    ``regularization`` is the Escobar determinant regularization of the
    distortion, which keeps inverted starting configurations finite.
    ``residual_tolerance`` is the absolute acceptance bound of constrained-node
    CAD residuals. A relaxation whose native minimization did not converge
    cannot replace the accepted geometry unless
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
    accept_valid_nonconverged_relaxation: bool = eqx.field(static=True)
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
        accept_valid_nonconverged_relaxation: bool = False,
    ) -> None:
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
        if not isinstance(accept_valid_nonconverged_relaxation, bool):
            raise TypeError("accept_valid_nonconverged_relaxation must be bool.")
        self.degree = int(degree)
        self.relaxation_rounds = int(relaxation_rounds)
        self.distortion_weight = positive_finite_float(
            distortion_weight, "distortion_weight"
        )
        self.tangent_weight = _non_negative(tangent_weight, "tangent_weight")
        self.displacement_weight = _non_negative(
            displacement_weight, "displacement_weight"
        )
        self.regularization = positive_finite_float(regularization, "regularization")
        self.residual_tolerance = positive_finite_float(
            residual_tolerance, "residual_tolerance"
        )
        self.accept_valid_nonconverged_relaxation = accept_valid_nonconverged_relaxation
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
    are the B-Rep class of each node's owning mesh entity, ``constrained`` marks
    nodes on B-Rep entities of lower dimension than the ambient space, and
    ``node_residuals`` their distance to that entity (zero for resolved free
    nodes). Positive infinity denotes an unavailable association or projection
    bound, not a measured distance; it always prevents acceptance.
    ``minimum_scaled_jacobian`` and ``maximum_distortion`` are sampled at the
    Bernstein control-point lattice; acceptance additionally requires a
    certified global embedding and source fidelity when source claims are supplied.
    Source claims are renewed only for a locally accepted map (certified cells
    and embedding, residuals within tolerance); an unavailable residual is
    recorded as a source failure. An explicit material-bearing
    `prior_transition` also yields a fresh source-to-curved `material_transition`
    in this evidence.
    """

    certificate: CellValidityCertificate
    embedding: GlobalEmbeddingCertificate
    node_dimensions: Array
    node_indices: Array
    constrained: Array
    node_residuals: Array
    maximum_residual: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    minimum_scaled_jacobian: float = eqx.field(static=True)
    maximum_distortion: float = eqx.field(static=True)
    accepted: bool = eqx.field(static=True)
    source_certification: MeshCertificationReport | None
    source_failure: str | None = eqx.field(static=True)
    coordinate_transition: CellGeometryTransition | None
    material_transition: CellMeshTransition | None
    target_result: CellMeshingResult | None
    evidence_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        certificate: CellValidityCertificate,
        embedding: GlobalEmbeddingCertificate,
        node_dimensions: np.ndarray,
        node_indices: np.ndarray,
        constrained: np.ndarray,
        node_residuals: np.ndarray,
        residual_tolerance: float,
        minimum_scaled_jacobian: float,
        maximum_distortion: float,
        /,
        *,
        source_certification: MeshCertificationReport | None = None,
        source_failure: str | None = None,
        target_result: CellMeshingResult | None = None,
        coordinate_transition: CellGeometryTransition | None = None,
        material_transition: CellMeshTransition | None = None,
    ) -> None:
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
        # An unresolved projection has no finite residual bound. Retain infinity
        # as refusal evidence rather than making the rollback itself fail.
        if np.any(np.isnan(residuals)) or np.any(residuals < 0.0):
            raise ValueError("Curved geometry residuals must be nonnegative and not NaN.")
        maximum = float(np.max(residuals, initial=0.0))
        if target_result is not None and not isinstance(target_result, CellMeshingResult):
            raise TypeError("target_result must be CellMeshingResult or None.")
        if coordinate_transition is not None and not isinstance(
            coordinate_transition, CellGeometryTransition
        ):
            raise TypeError(
                "coordinate_transition must be CellGeometryTransition or None."
            )
        if target_result is not None:
            transition = coordinate_transition
            if transition is None:
                raise ValueError(
                    "A target curved result must retain its coordinate transition."
                )
            target_mesh = target_result.mesh
            if (
                transition.target_mesh.mesh_id != target_mesh.mesh_id
                or transition.target_topology_id != target_mesh.topology_id
                or transition.target_geometry_id
                != cell_geometry_id(target_result.geometry)
            ):
                raise ValueError(
                    "Curving coordinate transition does not bind its target result."
                )
        if material_transition is not None:
            if not isinstance(material_transition, CellMeshTransition):
                raise TypeError("material_transition must be CellMeshTransition or None.")
            if (
                target_result is None
                or coordinate_transition is None
                or material_transition.volume_material_chain is None
                or material_transition.target.result_id != target_result.result_id
                or material_transition.geometry_transition is None
                or material_transition.geometry_transition.transition_id
                != coordinate_transition.transition_id
            ):
                raise ValueError(
                    "Curving material transition does not bind its source/target endpoints."
                )
        self.embedding = embedding
        self.certificate = certificate
        self.node_dimensions = jnp.asarray(dims)
        self.node_indices = jnp.asarray(indices)
        self.constrained = jnp.asarray(mask)
        self.node_residuals = jnp.asarray(residuals)
        self.maximum_residual = maximum
        self.residual_tolerance = float(residual_tolerance)
        self.minimum_scaled_jacobian = float(minimum_scaled_jacobian)
        self.maximum_distortion = float(maximum_distortion)
        self.source_certification = source_certification
        self.source_failure = source_failure
        self.target_result = target_result
        self.coordinate_transition = coordinate_transition
        self.material_transition = material_transition
        self.accepted = bool(
            certificate.all_certified
            and embedding.status == "certified"
            and np.isfinite(maximum)
            and maximum <= residual_tolerance
            and (source_certification is None or source_certification.passed)
            and source_failure is None
        )
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "curved-geometry-evidence",
                "certificate": certificate.certificate_id,
                "source_certification": None
                if source_certification is None
                else source_certification.report_id,
                "source_failure": source_failure,
                "target_result": None
                if target_result is None
                else target_result.result_id,
                "coordinate_transition": None
                if coordinate_transition is None
                else coordinate_transition.transition_id,
                "material_transition": (
                    None
                    if material_transition is None
                    else material_transition.transition_id
                ),
                "embedding": embedding.certificate_id,
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
    evidence of the last fully certified curved candidate: the accepted one, or
    after a rollback the last round's (the rejection reason). ``minimizations``
    carry the native optimizer evidence of every
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
    minimizations: tuple[MinimizationResult, ...]
    relaxation_statuses: tuple[OptimizationStatus, ...] = eqx.field(static=True)
    accepted_round: int | None = eqx.field(static=True)
    cell_result: CellMeshingResult | None
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        status: HighOrderCurvingStatus,
        geometry: CellGeometrySpec,
        straight: CellGeometrySpec,
        evidence: CurvedGeometryEvidence,
        candidate: CurvedGeometryEvidence | None,
        minimizations: tuple[MinimizationResult, ...],
        accepted_round: int | None,
        /,
        *,
        cell_result: CellMeshingResult | None = None,
    ) -> None:
        if not isinstance(status, HighOrderCurvingStatus):
            raise TypeError("status must be HighOrderCurvingStatus.")
        if status is HighOrderCurvingStatus.CURVED and not evidence.accepted:
            raise ValueError("A curved result requires accepted evidence.")
        if (status is HighOrderCurvingStatus.CURVED) != (accepted_round is not None):
            raise ValueError("Exactly a curved result names its accepted round.")
        if accepted_round is not None and not 0 <= accepted_round <= len(minimizations):
            raise ValueError("accepted_round must name the projection or a round.")
        if cell_result is not None:
            if not isinstance(cell_result, CellMeshingResult):
                raise TypeError("cell_result must be CellMeshingResult or None.")
            if cell_result.geometry.geometry_layout_id != geometry.geometry_layout_id:
                raise ValueError(
                    "cell_result geometry must match the returned curved geometry."
                )
        self.status = status
        self.geometry = geometry
        self.straight = straight
        self.evidence = evidence
        self.candidate = candidate
        self.minimizations = tuple(minimizations)
        self.relaxation_statuses = tuple(
            OptimizationStatus(int(np.asarray(value.status)))
            for value in self.minimizations
        )
        self.accepted_round = accepted_round
        self.cell_result = cell_result
        self.result_id = canonical_fingerprint(
            {
                "kind": "high-order-curving-result",
                "status": status.value,
                "geometry": geometry.geometry_layout_id,
                "coordinates": array_tree_fingerprint(np.asarray(geometry.coordinates)),
                "evidence": evidence.evidence_id,
                "candidate": None if candidate is None else candidate.evidence_id,
                "relaxation_statuses": [int(value) for value in self.relaxation_statuses],
                "accepted_round": accepted_round,
                "cell_result": None if cell_result is None else cell_result.result_id,
            }
        )


# -- geometry nodes ---------------------------------------------------------------


def _entity_key_table(mesh: CellMesh, dimension: int, /) -> dict[tuple[int, ...], int]:
    """Map exact vertex-global-ID closures to entity rows; never inspect coordinates."""
    pairs = np.asarray(_incidence_pairs(mesh, 0, dimension), dtype=np.int64)
    members: list[list[int]] = [[] for _ in range(mesh.entity_set(dimension).count)]
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    for vertex, entity in pairs:
        members[int(entity)].append(int(vertex_ids[int(vertex)]))
    table: dict[tuple[int, ...], int] = {}
    for row, identifiers in enumerate(members):
        if not identifiers:
            raise ValueError("Every mesh entity must have a vertex closure.")
        key = tuple(sorted(identifiers))
        if key in table:
            raise ValueError(
                "Curving cannot identify distinct periodic entities from vertex representatives alone."
            )
        table[key] = row
    return table


def _entity_rows_for_cells(
    mesh: CellMesh,
    dimension: int,
    cells: np.ndarray,
    local_vertices: tuple[int, ...],
    table: dict[tuple[int, ...], int],
    vertex_ids: np.ndarray,
    /,
) -> np.ndarray:
    identifiers = vertex_ids[cells[:, list(local_vertices)]]
    result = np.empty((cells.shape[0],), dtype=np.int64)
    for row, corners in enumerate(identifiers):
        key = tuple(sorted(int(value) for value in corners))
        try:
            result[row] = table[key]
        except KeyError as error:
            raise ValueError(
                f"Curving has no topological source entity of dimension {dimension} for a cell closure."
            ) from error
    return result


@final
class _ExactCoordinateBankMap(StrictModule, NonTrainableState):
    """Exact coefficient bank with a separate target-mesh execution view."""

    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    source_map_id: str = eqx.field(static=True)
    identity_context: tuple[str, ...] = eqx.field(static=True)
    rounded_vertices: np.ndarray
    exact_coordinates: ExactCoordinateBank = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        source_id: str,
        source_revision: str,
        identity_context: tuple[str, ...],
        exact_coordinates: CoordinateSourceBank,
        execution_coordinates: np.ndarray,
        /,
    ) -> None:
        execution = np.asarray(execution_coordinates, dtype=np.float64)
        try:
            bank = ExactCoordinateBank.from_rows(exact_coordinates)
        except TypeError as error:
            raise ValueError(
                "Exact coordinate-bank rows must round to their execution coordinates."
            ) from error
        except ValueError as error:
            raise ValueError(
                "Exact coordinate-bank values require finite binary64 shadows."
            ) from error
        try:
            rounded_exact = bank.binary64()
        except OverflowError as error:
            raise ValueError(
                "Exact coordinate-bank values require finite binary64 shadows."
            ) from error
        if (
            not source_id
            or not source_revision
            or execution.ndim != 2
            or rounded_exact.shape != execution.shape
            or not np.all(np.isfinite(execution))
            or not np.array_equal(rounded_exact, execution)
        ):
            raise ValueError(
                "Exact coordinate-bank rows must round to their execution coordinates."
            )
        vertices = np.array(mesh.coordinates, dtype=np.float64, copy=True)
        vertices.setflags(write=False)
        self.source_id = str(source_id)
        self.source_revision = str(source_revision)
        self.identity_context = tuple(identity_context)
        self.source_map_id = canonical_fingerprint(
            {
                "kind": "exact-coordinate-bank-map",
                "mesh_id": mesh.mesh_id,
                "topology_id": mesh.topology_id,
                "numeric_version": mesh.numeric_version,
                "source_id": self.source_id,
                "source_revision": self.source_revision,
                "identity_context": self.identity_context,
                "exact_coordinate_bank": bank.coefficient_payloads(),
            }
        )
        self.rounded_vertices = vertices
        self.exact_coordinates = bank

    @classmethod
    def from_stored_fields(
        cls, value: _ExactCoordinateBankMap, /, *, mesh: CellMesh
    ) -> _ExactCoordinateBankMap:
        """Renew an archived bank on its target mesh and require its identity."""
        if type(value) is not cls:
            raise TypeError("value must be an exact coordinate-bank map.")
        exact = value.exact_coordinates
        rebuilt = cls(
            mesh,
            value.source_id,
            value.source_revision,
            value.identity_context,
            exact,
            exact.binary64(),
        )
        if rebuilt.source_map_id != value.source_map_id or not np.array_equal(
            rebuilt.rounded_vertices, np.asarray(value.rounded_vertices)
        ):
            raise ValueError(
                "Stored exact coordinate bank differs from its renewed target mesh."
            )
        return rebuilt

    def source_coordinates(
        self, indices: ArrayLike | None = None, /
    ) -> CoordinateSourceBank:
        if indices is None:
            return self.exact_coordinates
        selected = np.asarray(indices)
        if selected.ndim != 1 or selected.dtype.kind not in "iu":
            raise TypeError("Exact-coordinate-bank indices must be an integer vector.")
        rows = selected.astype(np.int64)
        if np.any(rows < 0) or np.any(rows >= len(self.exact_coordinates)):
            raise IndexError("Exact-coordinate-bank index is out of range.")
        return self.exact_coordinates.rows(rows)


def _exact_coordinate_bank_source_map(
    mesh: CellMesh,
    source_id: str,
    source_revision: str,
    identity_context: tuple[str, ...],
    exact_coordinates: CoordinateSourceBank,
    execution_coordinates: np.ndarray,
    /,
) -> _ExactCoordinateBankMap:
    return _ExactCoordinateBankMap(
        mesh,
        source_id,
        source_revision,
        identity_context,
        exact_coordinates,
        execution_coordinates,
    )


def _geometry_with_exact_coordinate_bank(
    mesh: CellMesh,
    elements: dict[str, CellGeometryElement],
    routes: Mapping[str, ArrayLike],
    straight: CellGeometrySpec,
    execution_coordinates: np.ndarray,
    /,
    *,
    identity_context: tuple[str, ...],
    exact_overrides: dict[int, tuple[ExactScalar, ...]] | None = None,
) -> CellGeometrySpec:
    execution = np.asarray(execution_coordinates, dtype=np.float64)
    straight_bank = straight.exact_coordinate_bank
    if (
        straight_bank is None
        or execution.shape != (len(straight_bank), mesh.ambient_dimension)
        or not np.all(np.isfinite(execution))
    ):
        raise ValueError("A candidate exact bank must align with the straight nodal map.")
    if exact_overrides is not None and any(
        row < 0 or row >= len(straight_bank) for row in exact_overrides
    ):
        raise IndexError("Exact candidate-coordinate row is outside the geometry bank.")
    try:
        override_rows = np.asarray(sorted(exact_overrides or ()), dtype=np.int64)
        overrides = ExactCoordinateBank.from_rows(
            tuple(
                tuple(cast(dict[int, tuple[ExactScalar, ...]], exact_overrides)[row])
                for row in override_rows.tolist()
            )
        )
    except TypeError as error:
        raise ValueError(
            "Exact coordinate-bank rows must round to their execution coordinates."
        ) from error
    if override_rows.size and (
        overrides.shape != (override_rows.size, execution.shape[1])
        or not np.array_equal(overrides.binary64(), execution[override_rows])
    ):
        raise ValueError(
            "Exact candidate coordinates do not round to their execution row."
        )
    # Unmoved rows keep their exact straight value; moved rows are exact binary64.
    fresh = ~np.all(straight_bank.binary64() == execution, axis=1)
    fresh[override_rows] = False
    fresh_rows = np.flatnonzero(fresh)
    candidate_bank = straight_bank
    if fresh_rows.size:
        candidate_bank = candidate_bank.replace_rows(
            fresh_rows, ExactCoordinateBank.from_binary64(execution[fresh_rows])
        )
    if override_rows.size:
        candidate_bank = candidate_bank.replace_rows(override_rows, overrides)
    straight_map_id = straight.source_map_id
    if not isinstance(straight_map_id, str) or not straight_map_id:
        raise ValueError(
            "Straight geometry has no current exact coordinate-map identity."
        )
    source_map = _exact_coordinate_bank_source_map(
        mesh,
        mesh.mesh_id,
        mesh.numeric_version,
        ("curving-candidate", straight_map_id, *identity_context),
        candidate_bank,
        execution,
    )
    return CellGeometrySpec(
        elements,
        routes,
        execution,
        source_map=source_map,
        source_mesh=mesh,
    )


def _straight_geometry(
    mesh: CellMesh,
    degree: int,
    /,
    *,
    source_geometry: CellGeometrySpec | None = None,
) -> CellGeometrySpec:
    """Evaluate the actual source map at canonical nodes, or the P1 mesh map.

    Interpolation at these nodes does not itself certify equality of the full
    map when the requested degree is below the source's degree.
    """
    kinds = {block.cell_kind for block in mesh.blocks}
    if not kinds <= set(_CURVED_KINDS):
        raise ValueError(
            "Curving supports triangle, quadrilateral, tetrahedron/simplex:3, prism, pyramid, and hexahedron blocks."
        )
    mesh_coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    linear_geometry = (
        CellGeometrySpec(
            {block.name: lagrange_element(block.cell_kind, 1) for block in mesh.blocks},
            {
                block.name: np.asarray(block.vertices, dtype=np.int32)
                for block in mesh.blocks
            },
            mesh_coordinates,
        )
        if source_geometry is None
        else source_geometry
    )
    linear_geometry.resolve(mesh)
    elements: dict[str, CellGeometryElement] = {}
    routes: dict[str, np.ndarray] = {}
    local_keys: dict[str, tuple[tuple[tuple[object, ...], ...], ...]] = {}
    exact_nodes: dict[tuple[object, ...], tuple[ExactScalar, ...]] = {}
    vertex_node_rows: dict[tuple[object, ...], int] = {}
    vertex_coordinates: list[tuple[ExactScalar, ...] | None] = [
        None for _ in range(mesh_coordinates.shape[0])
    ]
    node_identity_payload: list[str] = []

    for block in mesh.blocks:
        element = lagrange_element(block.cell_kind, degree)
        nodal_data = lagrange_reference_node_data(element)
        if len(nodal_data) != element.local_dof_count or any(
            node.local_dof != local_dof for local_dof, node in enumerate(nodal_data)
        ):
            raise ValueError("Exact nodal metadata does not align with local DOF order.")
        execution_nodes = np.asarray(element.reference_nodes, dtype=np.float64)
        nodal_shadows = np.asarray(
            tuple(node.execution_shadow for node in nodal_data), dtype=np.float64
        )
        if not np.array_equal(execution_nodes, nodal_shadows):
            raise ValueError(
                "Exact nodal metadata shadows differ from the Lagrange element."
            )
        if any(node.exact_point is None for node in nodal_data):
            missing = tuple(
                node.local_dof for node in nodal_data if node.exact_point is None
            )
            raise ValueError(
                f"Exact straight-map evaluation lacks representable reference nodes {missing} "
                f"for {block.cell_kind!r} degree {degree}."
            )
        reference_points = tuple(
            cast(tuple[ExactScalar, ...], node.exact_point) for node in nodal_data
        )
        cells = np.asarray(block.vertices, dtype=np.int64)
        cell_ids = np.asarray(block.global_ids, dtype=np.int64)
        images = linear_geometry.evaluate_exact(
            mesh,
            cell_ids,
            tuple(reference_points for _ in range(block.cell_count)),
        )
        route = np.full((block.cell_count, element.local_dof_count), -1, dtype=np.int32)
        rows_for_cells: list[tuple[tuple[object, ...], ...]] = []
        for cell_row in range(block.cell_count):
            cell_vertices = tuple(
                int(cells[cell_row, local_vertex])
                for local_vertex in range(cells.shape[1])
            )
            cell_images = images[cell_row]
            cell_vertex_ids = tuple(
                int(vertex_ids[int(index)]) for index in cell_vertices
            )
            identity_row: list[tuple[object, ...]] = []
            for local_dof, (node, image) in enumerate(
                zip(nodal_data, cell_images, strict=True)
            ):
                identity, orientation = canonical_nodal_entity_identity(
                    node,
                    cell_vertex_ids,
                    cell_global_id=int(cell_ids[cell_row]),
                )
                if len(orientation) != len(node.entity_vertices):
                    raise ValueError(
                        "Canonical nodal orientation does not match its owning entity."
                    )
                exact_image = tuple(image)
                if any(
                    not isinstance(value, (Fraction, RadicalElement))
                    for value in exact_image
                ):
                    raise ValueError(
                        "Straight source-map evaluation did not return exact scalar coordinates."
                    )
                previous = exact_nodes.setdefault(identity, exact_image)
                if previous != exact_image:
                    raise ValueError(
                        "Adjacent cell maps disagree on an exact canonical nodal identity."
                    )
                if node.entity_dimension == 0:
                    if len(node.entity_vertices) != 1:
                        raise ValueError(
                            "A vertex nodal functional needs one local vertex."
                        )
                    vertex_row = int(cell_vertices[node.entity_vertices[0]])
                    if (
                        int(vertex_ids[vertex_row])
                        != cell_vertex_ids[node.entity_vertices[0]]
                    ):
                        raise ValueError(
                            "Vertex nodal identity disagrees with mesh connectivity."
                        )
                    owner_row = vertex_node_rows.setdefault(identity, vertex_row)
                    if owner_row != vertex_row:
                        raise ValueError(
                            "One canonical vertex identity resolves to multiple mesh rows."
                        )
                    vertex_image = vertex_coordinates[vertex_row]
                    if vertex_image is not None and vertex_image != exact_image:
                        raise ValueError(
                            "Shared mesh vertices have different exact straight-map images."
                        )
                    vertex_coordinates[vertex_row] = exact_image
                identity_row.append(identity)
                node_identity_payload.append(
                    canonical_fingerprint(
                        {
                            "block": block.name,
                            "cell_id": int(cell_ids[cell_row]),
                            "local_dof": local_dof,
                            "identity": identity,
                            "orientation": orientation,
                        }
                    )
                )
            rows_for_cells.append(tuple(identity_row))
        elements[block.name] = element
        routes[block.name] = route
        local_keys[block.name] = tuple(rows_for_cells)

    for row, exact_coordinate in enumerate(vertex_coordinates):
        if exact_coordinate is None:
            if source_geometry is not None:
                raise ValueError(
                    "Source geometry cannot prove an unreferenced mesh vertex."
                )
            vertex_coordinates[row] = tuple(
                Fraction(float(value)) for value in mesh_coordinates[row]
            )
            continue
        shadow = np.asarray(tuple(float(value) for value in exact_coordinate))
        if not np.array_equal(shadow, mesh_coordinates[row]):
            raise ValueError(
                "Exact straight-map vertex coordinates do not round to the mesh vertices."
            )
    exact_vertices = tuple(
        cast(tuple[ExactScalar, ...], value) for value in vertex_coordinates
    )
    high_order_keys = tuple(
        sorted(
            (key for key in exact_nodes if key not in vertex_node_rows),
            key=lambda key: canonical_fingerprint({"identity": key}),
        )
    )
    geometry_row_by_key = {
        **vertex_node_rows,
        **{key: len(exact_vertices) + index for index, key in enumerate(high_order_keys)},
    }
    exact_coordinates = (
        *exact_vertices,
        *(exact_nodes[key] for key in high_order_keys),
    )
    for block in mesh.blocks:
        block_routes = routes[block.name]
        for cell_row, identities in enumerate(local_keys[block.name]):
            block_routes[cell_row] = np.asarray(
                [geometry_row_by_key[identity] for identity in identities],
                dtype=np.int32,
            )
    coordinates = np.asarray(
        tuple(tuple(float(value) for value in row) for row in exact_coordinates),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(coordinates)) or not np.array_equal(
        coordinates[: mesh_coordinates.shape[0]], mesh_coordinates
    ):
        raise ValueError("Exact straight-map coordinates need finite mesh-bound shadows.")
    source_map = _exact_coordinate_bank_source_map(
        mesh,
        mesh.mesh_id,
        mesh.numeric_version,
        (
            "straight-p1-mesh-map"
            if source_geometry is None
            else "source-map-nodal-evaluation",
            str(degree),
            cell_geometry_id(linear_geometry),
            *node_identity_payload,
        ),
        exact_coordinates,
        coordinates,
    )
    return CellGeometrySpec(
        elements,
        routes,
        coordinates,
        source_map=source_map,
        source_mesh=mesh,
    )


def _node_owners(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> tuple[np.ndarray, np.ndarray]:
    """Owning mesh entity ``(dimension, row)`` of every geometry node."""
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
        ownership_element = _nodal_reference_element(element)
        if ownership_element.cell_kind != block.cell_kind:
            raise ValueError(
                "Curving nodal ownership requires target-local reference entities."
            )
        route_ = np.asarray(route, dtype=np.int64)
        cells = np.asarray(block.vertices, dtype=np.int64)
        topology = reference_cell_topology(block.cell_kind)
        cell_rows = _entity_rows(mesh, top, np.asarray(block.global_ids, dtype=np.int64))
        for dimension, entities in enumerate(ownership_element.entity_dofs):
            for local, dofs in enumerate(entities):
                if not dofs:
                    continue
                vertices = tuple(topology.entities[dimension][local])
                if dimension == 0:
                    owners = cells[:, vertices[0]]
                elif dimension == top:
                    owners = cell_rows
                else:
                    owners = _entity_rows_for_cells(
                        mesh,
                        dimension,
                        cells,
                        vertices,
                        tables[dimension],
                        vertex_ids,
                    )
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
    projection: AbstractBRepProjection,
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
        case "tetrahedron" | "simplex:3":
            return 3 * (degree - 1)
        case "quadrilateral":
            return (2 if embedded else 1) * (2 * degree - 1)
        case "hexahedron" | "prism" | "pyramid":
            return 3 * degree - 1
        case _:
            raise ValueError(f"No curving sample lattice for {kind!r}.")


def _control_lattice(kind: str, degree: int, embedded: bool, /) -> np.ndarray:
    """Polynomial control lattice on the reference simplex, prism, or cube."""
    order = max(_determinant_degree(kind, degree, embedded), 1)
    dimension = reference_cell_topology(kind).dimension
    grid = np.asarray(tuple(product(range(order + 1), repeat=dimension)), np.float64)
    grid = grid / order
    if kind in ("triangle", "tetrahedron", "prism") or kind.startswith("simplex:"):
        grid = grid[np.sum(grid[:, : 2 if kind == "prism" else dimension], axis=1) <= 1.0]
    if kind == "pyramid":
        scale = 1.0 - grid[:, 2]
        grid = np.stack(
            (
                scale * grid[:, 0] + 0.5 * grid[:, 2],
                scale * grid[:, 1] + 0.5 * grid[:, 2],
                grid[:, 2],
            ),
            axis=1,
        )
    return grid


class _BlockDistortion(StrictModule):
    """Sampled Jacobians of one block relative to its straight-sided element."""

    routes: Array
    gradients: Array
    target_inverses: Array
    plan: SmallLinearSolvePlan
    embedded: bool = eqx.field(static=True)
    per_cell_gradients: bool = eqx.field(static=True)

    def ratios(self, coordinates: Array, /) -> tuple[Array, Array, Array]:
        """Target-normalized ``(frobenius, determinant, dimension)`` per sample."""
        jacobian = (
            contract("cmnk,cna->cmak", self.gradients, coordinates[self.routes])
            if self.per_cell_gradients
            else contract("mnk,cna->cmak", self.gradients, coordinates[self.routes])
        )
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
        if not isinstance(element, ScalarCoordinateElement):
            raise TypeError("Curving requires a canonical scalar cell-geometry element.")
        dimension = reference_cell_topology(block.cell_kind).dimension
        embedded = ambient > dimension
        samples = _control_lattice(block.cell_kind, degree, embedded)
        if isinstance(element, PerCellRestrictedCellGeometryElement):
            if not block.cell_count:
                raise ValueError(
                    "Curving requires at least one cell per restricted block."
                )
            # Cells with one exact chart share one tabulation; the result is
            # the per-cell tabulation of each cell's own chart.
            gradients = np.asarray(
                element.tabulate_cells(np.arange(block.cell_count), samples)[1]
            )
            per_cell_gradients = True
        else:
            if not isinstance(element, _NodalTabulatingGeometryElement):
                raise TypeError("Curving requires a tabulating nodal coordinate element.")
            _, gradients = element.tabulate(samples)
            per_cell_gradients = False
        _, linear = lagrange_element(block.cell_kind, 1).tabulate(samples)
        linear = np.asarray(linear, dtype=np.float64)
        linear_bits = linear.view(np.uint64)
        if np.array_equal(
            linear_bits, np.broadcast_to(linear_bits[:1], linear_bits.shape)
        ):
            # Identical straight-map Jacobians share one solve per cell.
            # Broadcast the retained inverse over the unchanged sample lattice.
            linear = linear[:1]
        straight = contract(
            "mnk,cna->cmak",
            linear,
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
                per_cell_gradients=per_cell_gradients,
            )
        )
    return tuple(blocks)


def _relaxation_star(
    blocks: tuple[_BlockDistortion, ...],
    certificate: CellValidityCertificate,
    fixed: np.ndarray,
    rings: int,
    /,
) -> tuple[np.ndarray, tuple[_BlockDistortion, ...]]:
    """Movable nodes and distortion blocks of one relaxation round.

    The cells ``certificate`` does not prove valid, grown by ``rings`` rings of
    node-sharing cells, free their non-fixed nodes; every cell sharing a movable
    node carries the distortion, so the round solves that coupled star alone and
    every other node keeps its position.
    """
    routes = tuple(np.asarray(block.routes, dtype=np.int64) for block in blocks)
    unproven = np.split(
        np.asarray(certificate.status) != CellValidityStatus.CERTIFIED_VALID,
        certificate.block_offsets[1:-1],
    )
    star = np.zeros(fixed.shape, dtype=np.bool_)
    for route, cells in zip(routes, unproven, strict=True):
        star[route[cells]] = True
    for _ in range(rings):
        grown = star.copy()
        for route in routes:
            grown[route[np.any(star[route], axis=1)]] = True
        star = grown
    movable = star & ~fixed
    restricted = []
    for block, route in zip(blocks, routes, strict=True):
        cells = np.flatnonzero(np.any(movable[route], axis=1))
        if not cells.size:
            continue
        rows = jnp.asarray(cells, dtype=jnp.int32)
        restricted.append(
            _BlockDistortion(
                routes=block.routes[rows],
                gradients=(
                    block.gradients[rows] if block.per_cell_gradients else block.gradients
                ),
                target_inverses=block.target_inverses[rows],
                plan=block.plan,
                embedded=block.embedded,
                per_cell_gradients=block.per_cell_gradients,
            )
        )
    return movable, tuple(restricted)


def _sampled_block_quality(
    block: _BlockDistortion, values: Array, /
) -> tuple[Array, Array]:
    """Reduce sampled diagnostics in their original eager evaluation order."""
    frobenius, determinant, jacobian = block.ratios(values)
    dimension = block.gradients.shape[-1]
    columns = jnp.prod(jnp.linalg.norm(jacobian, axis=-2), axis=-1)
    if block.embedded:
        gram = jnp.swapaxes(jacobian, -1, -2) @ jacobian
        measure = jnp.sqrt(jnp.maximum(determinant_small_linear(block.plan, gram), 0.0))
    else:
        measure = determinant_small_linear(block.plan, jacobian)
    scaled = jnp.min(measure / jnp.maximum(columns, jnp.finfo(values.dtype).tiny))
    positive = jnp.where(determinant > 0.0, determinant, jnp.nan)
    distortion = jnp.nanmax(frobenius / (dimension * positive ** (2.0 / dimension)))
    return scaled, distortion


# Host certificate preparation must retain the eager rounding of each numeric
# kernel. Bound live Jacobian work rather than fusing their arithmetic.
_QUALITY_WORKSET_ENTRIES = 1 << 18


def _quality_workset(
    block: _BlockDistortion,
    cell_start: int,
    cell_count: int,
    /,
) -> _BlockDistortion:
    cell_offset = jnp.asarray(cell_start, dtype=jnp.int32)
    routes = lax.dynamic_slice_in_dim(block.routes, cell_offset, cell_count, axis=0)
    gradients = block.gradients
    if block.per_cell_gradients:
        gradients = lax.dynamic_slice_in_dim(gradients, cell_offset, cell_count, axis=0)
    inverses = lax.dynamic_slice_in_dim(
        block.target_inverses, cell_offset, cell_count, axis=0
    )
    return _BlockDistortion(
        routes,
        gradients,
        inverses,
        block.plan,
        block.embedded,
        block.per_cell_gradients,
    )


def _bounded_sampled_block_quality(
    block: _BlockDistortion, values: Array, /
) -> tuple[Array, Array]:
    cells = block.routes.shape[0]
    samples = block.gradients.shape[1 if block.per_cell_gradients else 0]
    width = values.shape[1] * block.gradients.shape[-1]
    if cells * samples * width <= _QUALITY_WORKSET_ENTRIES:
        return _sampled_block_quality(block, values)
    cell_count = max(1, _QUALITY_WORKSET_ENTRIES // (samples * width))
    scaled: list[np.ndarray] = []
    distortion: list[np.ndarray] = []
    for cell_start in range(0, cells, cell_count):
        workset = _quality_workset(block, cell_start, min(cell_count, cells - cell_start))
        scale, shape = _sampled_block_quality(workset, values)
        # This is an explicit host-evidence boundary, not an optimizer
        # iteration: finish one workset before admitting its successor.
        scaled.append(np.asarray(scale))
        distortion.append(np.asarray(shape))
    return (
        jnp.min(jnp.asarray(np.stack(scaled))),
        jnp.nanmax(jnp.asarray(np.stack(distortion))),
    )


def _sampled_quality(
    blocks: tuple[_BlockDistortion, ...], coordinates: np.ndarray, /
) -> tuple[float, float]:
    """Minimum sampled scaled Jacobian and maximum sampled distortion."""
    values = jnp.asarray(coordinates)
    scaled, distortion = [], []
    for block in blocks:
        block_scaled, block_distortion = _bounded_sampled_block_quality(block, values)
        scaled.append(block_scaled)
        distortion.append(block_distortion)
    return float(min(scaled)), float(max(distortion))


# -- evidence ---------------------------------------------------------------------


@dataclass(slots=True)
class _NodeProjectionCache:
    """Previous row-aligned projections, scoped to one curving invocation."""

    projection: AbstractBRepProjection
    classes: _NodeClasses
    rows: np.ndarray | None = None
    dimensions: np.ndarray | None = None
    indices: np.ndarray | None = None
    queries: np.ndarray | None = None
    feet: np.ndarray | None = None
    residuals: np.ndarray | None = None
    projectors: np.ndarray | None = None
    preparation: _ProjectionPreparation | None = None
    hits: int = 0
    misses: int = 0


def _node_residuals(
    projection: AbstractBRepProjection,
    coordinates: np.ndarray,
    classes: _NodeClasses,
    /,
    *,
    cache: _NodeProjectionCache | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Closest points, residuals, unit normals-space projectors of constrained nodes."""
    rows = np.flatnonzero(classes.constrained)
    ambient = coordinates.shape[1]
    residuals = np.zeros((coordinates.shape[0],))
    query = np.ascontiguousarray(coordinates[rows], dtype=np.float64)
    same = np.zeros(rows.size, dtype=np.bool_)
    if (
        cache is not None
        and cache.projection is projection
        and cache.classes is classes
        and cache.rows is not None
        and np.array_equal(cache.rows, rows)
        and cache.dimensions is not None
        and cache.indices is not None
        and np.array_equal(cache.dimensions, classes.dimensions[rows])
        and np.array_equal(cache.indices, classes.indices[rows])
        and cache.queries is not None
        and cache.queries.shape == query.shape
    ):
        # Bitwise identity preserves even signed zero and never treats a
        # tolerance-near query as the same scientific input.
        same = np.all(query.view(np.uint64) == cache.queries.view(np.uint64), axis=1)
    feet = np.empty_like(query)
    projected_residuals = np.empty(rows.size, dtype=np.float64)
    projectors = np.empty((rows.size, ambient, ambient), dtype=np.float64)
    if np.any(same):
        assert cache is not None and cache.feet is not None
        assert cache.residuals is not None and cache.projectors is not None
        feet[same] = cache.feet[same]
        projected_residuals[same] = cache.residuals[same]
        projectors[same] = cache.projectors[same]
    missing = ~same
    if np.any(missing):
        selected = rows[missing]
        from ..geometry.brep._projection import NativeBRepProjection

        if (
            cache is not None
            and cache.projection is projection
            and cache.classes is classes
            and isinstance(projection, NativeBRepProjection)
        ):
            if cache.preparation is None:
                cache.preparation = projection._prepare_project()
            result = projection._project_prepared(
                query[missing],
                classes.dimensions[selected],
                classes.indices[selected],
                preparation=cache.preparation,
            )
        else:
            result = projection.project(
                query[missing], classes.dimensions[selected], classes.indices[selected]
            )
        failed = ~np.isin(np.asarray(result.status), _RESOLVED_STATUS)
        projected_residuals[missing] = np.where(
            failed, np.inf, np.asarray(result.residuals)
        )
        identity = np.eye(ambient)
        tangents = np.asarray(result.tangents)
        normals = np.asarray(result.normals)
        fresh_projectors = np.zeros((selected.size, ambient, ambient))
        edge = classes.dimensions[selected] == 1
        face = classes.dimensions[selected] == 2
        fresh_projectors[edge] = (
            identity - tangents[edge, 0, :, None] * tangents[edge, 0, None, :]
        )
        fresh_projectors[face] = normals[face, :, None] * normals[face, None, :]
        fresh_projectors[classes.dimensions[selected] == 0] = identity
        unframed = ~np.all(np.isfinite(fresh_projectors), axis=(1, 2))
        fresh_projectors[unframed] = identity
        projectors[missing] = fresh_projectors
        fresh_feet = np.asarray(result.points)
        feet[missing] = np.where(np.isfinite(fresh_feet), fresh_feet, query[missing])
    residuals[rows] = projected_residuals
    residuals[~classes.resolved] = np.inf
    if cache is not None and cache.projection is projection and cache.classes is classes:
        cache.rows = rows.copy()
        cache.dimensions = classes.dimensions[rows].copy()
        cache.indices = classes.indices[rows].copy()
        cache.queries = query
        cache.feet = feet.copy()
        cache.residuals = projected_residuals.copy()
        cache.projectors = projectors.copy()
        cache.hits += int(np.count_nonzero(same))
        cache.misses += int(np.count_nonzero(missing))
    return rows, feet, residuals, projectors


def _curving_target_result(
    source_result: CellMeshingResult | None,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    projection: AbstractBRepProjection,
    /,
    *,
    prior_transition: CellMeshTransition | None = None,
    embedding: GlobalEmbeddingCertificate,
) -> tuple[
    CellMeshingResult | None,
    MeshCertificationReport | None,
    str | None,
    CellGeometryTransition | None,
    CellMeshTransition | None,
]:
    if source_result is None:
        return None, None, None, None, None
    inputs = None
    validated_witnesses = ()
    audit_policy = None
    layer_curving_evidence = None
    coordinate_transition = None
    material_transition = None
    organization = NativeMeshOrganization(
        boundary=source_result.boundary,
        patches=source_result.patches,
        zones=source_result.zones,
        labels=source_result.labels,
        attributes=source_result.attributes,
        associations=source_result.associations,
        surface_source=source_result.surface_source,
        region_boundary_evidence=source_result.region_boundary_evidence,
    )
    try:
        if (
            source_result.mesh.mesh_id != mesh.mesh_id
            or source_result.mesh.topology_id != mesh.topology_id
        ):
            raise ValueError("source_result must retain the exact mesh being curved.")
        lineage = _identity_lineage(mesh, mesh)
        identity_maps = _identity_reference_maps_for_transition(mesh, mesh)
        # Curving relocates geometry nodes on a fixed topology: the identity cell
        # maps carry lineage, never a coordinate-map equality claim.
        coordinate_transition = _transition_geometry_record(source_result, mesh, geometry)
        if source_result.layer_curving_evidence is not None:
            from ._layer_curving import renew_layer_curving_source_evidence

            source_layer_evidence = source_result.layer_curving_evidence
            target_chart_witness = _retarget_source_boundary_chart_witness(
                source_result,
                mesh,
                geometry,
                lineage,
                source_layer_evidence.chart_witness,
                reference_maps=identity_maps,
            )
            layer_curving_evidence = renew_layer_curving_source_evidence(
                source_layer_evidence,
                source_result.mesh,
                source_result.geometry,
                mesh,
                geometry,
                target_chart_witness,
                lineage,
                identity_maps,
            )
        if source_result.certification is not None:
            original = source_result.certification.inputs
            if original.source is not None and (
                original.source_id != projection.source_id
                or original.source_revision != projection.source_revision
            ):
                raise ValueError(
                    "Curving projection and retained source revision disagree."
                )
            renewed_inputs = _renew_transition_certification_inputs(
                source_result,
                mesh,
                geometry,
                lineage,
                reference_maps=identity_maps,
                transition=coordinate_transition,
            )
            if renewed_inputs is not None:
                inputs, validated_witnesses = renewed_inputs
            audit_policy = source_result.certification.audit_policy
        # The in-hand proofs are reused only when they answer the same premises.
        reuse = (
            audit_policy is not None
            and audit_policy.validity_policy.policy_id
            == embedding.validity.policy.policy_id
        )
        result = certify_cell_mesh(
            mesh,
            source_result.coordinate_contract,
            geometry=geometry,
            audit_policy=audit_policy,
            certification_inputs=inputs,
            validated_witnesses=validated_witnesses,
            boundary=source_result.boundary,
            patches=source_result.patches,
            zones=source_result.zones,
            labels=source_result.labels,
            attributes=source_result.attributes,
            associations=source_result.associations,
            surface_source=source_result.surface_source,
            region_boundary_evidence=source_result.region_boundary_evidence,
            planar_embedding=source_result.planar_embedding,
            layer_curving_evidence=layer_curving_evidence,
            embedding=embedding if reuse else None,
            validity=embedding.validity if reuse else None,
        )
        if prior_transition is not None:
            material_chain = _extend_prior_volume_material_chain(
                prior_transition,
                source_result,
                result,
                lineage,
                reference_maps=identity_maps,
                coordinate_transition=coordinate_transition,
            )
            if material_chain is not None:
                material_transition = CellMeshTransition(
                    source_result.mesh.mesh_id,
                    source_result.mesh.topology_id,
                    result,
                    lineage,
                    MeshTransitionKind.REPAIR,
                    geometry_transition=coordinate_transition,
                    volume_material_chain=material_chain,
                )
    except (MeshingFailure, ValueError) as error:
        report = None
        if inputs is not None:
            try:
                report = MeshCertificationReport(
                    inputs,
                    audit_policy=audit_policy,
                    organization=organization,
                )
            except (MeshingFailure, ValueError):
                pass
        return None, report, str(error), coordinate_transition, None
    return result, result.certification, None, coordinate_transition, material_transition


class _LocalEvidence(NamedTuple):
    """Per-map evidence: CAD residuals, sampled quality and cell validity."""

    residuals: np.ndarray
    scaled: float
    distortion: float
    computed_validity: ComputedCellValidity

    def passes(self, policy: HighOrderCurvingPolicy, /) -> bool:
        """Every cell certified valid and every constrained node on the source."""
        return bool(
            self.computed_validity.certificate.all_certified
            and float(np.max(self.residuals, initial=0.0)) <= policy.residual_tolerance
        )


def _local_evidence(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    classes: _NodeClasses,
    projection: AbstractBRepProjection,
    blocks: tuple[_BlockDistortion, ...],
    policy: HighOrderCurvingPolicy,
    /,
    *,
    parent: ComputedCellValidity | None = None,
    cache: _NodeProjectionCache | None = None,
) -> _LocalEvidence:
    coordinates = np.asarray(geometry.coordinates, dtype=np.float64)
    _, _, residuals, _ = _node_residuals(projection, coordinates, classes, cache=cache)
    scaled, distortion = _sampled_quality(blocks, coordinates)
    computed_validity = ComputedCellValidity(geometry, mesh, policy.validity, parent)
    return _LocalEvidence(residuals, scaled, distortion, computed_validity)


def _evidence(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    classes: _NodeClasses,
    projection: AbstractBRepProjection,
    blocks: tuple[_BlockDistortion, ...],
    policy: HighOrderCurvingPolicy,
    /,
    *,
    source_result: CellMeshingResult | None = None,
    prior_transition: CellMeshTransition | None = None,
    local: _LocalEvidence | None = None,
    cache: _NodeProjectionCache | None = None,
    embedding: GlobalEmbeddingCertificate | None = None,
) -> CurvedGeometryEvidence:
    if local is None:
        local = _local_evidence(
            mesh, geometry, classes, projection, blocks, policy, cache=cache
        )
    residuals, scaled, distortion, computed_validity = local
    validity = computed_validity.certificate
    computed_validity.require(geometry, mesh, validity)
    certification = None if source_result is None else source_result.certification
    junction_vertices = (
        None if certification is None else certification.inputs.junction_vertices
    )
    if embedding is None:
        embedding = certify_global_embedding(
            mesh,
            geometry,
            validity,
            junction_vertices=junction_vertices,
            validated_validity=computed_validity,
        )
    else:
        embedding.require_inputs(
            mesh, geometry, validity, junction_vertices=junction_vertices
        )
    # Source claims are renewed only for a locally accepted map: a rejected
    # candidate's refusal is decided before any source acceptance is consulted.
    locally_accepted = local.passes(policy) and embedding.status == "certified"
    (
        target_result,
        source_certification,
        source_failure,
        coordinate_transition,
        material_transition,
    ) = (
        _curving_target_result(
            source_result,
            mesh,
            geometry,
            projection,
            prior_transition=prior_transition,
            embedding=embedding,
        )
        if locally_accepted
        else (None, None, None, None, None)
    )
    if np.any(~np.isfinite(residuals)):
        source_failure = (
            "CAD projection or node association has no finite residual bound."
        )
    return CurvedGeometryEvidence(
        validity,
        embedding,
        classes.dimensions,
        classes.indices,
        classes.constrained,
        residuals,
        policy.residual_tolerance,
        scaled,
        distortion,
        source_certification=source_certification,
        source_failure=source_failure,
        target_result=target_result,
        coordinate_transition=coordinate_transition,
        material_transition=material_transition,
    )


def _require_inputs(
    mesh: CellMesh,
    association: GeometryAssociation,
    projection: AbstractBRepProjection,
    policy: HighOrderCurvingPolicy,
    /,
) -> None:
    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if not isinstance(association, GeometryAssociation):
        raise TypeError("association must be GeometryAssociation.")
    if not isinstance(projection, AbstractBRepProjection):
        raise TypeError("projection must be AbstractBRepProjection.")
    if not isinstance(policy, HighOrderCurvingPolicy):
        raise TypeError("policy must be HighOrderCurvingPolicy.")


def _curving_source_result(
    mesh: CellMesh,
    source_result: CellMeshingResult | None,
    prior_transition: CellMeshTransition | None,
    /,
) -> CellMeshingResult | None:
    if prior_transition is None:
        return source_result
    if not isinstance(prior_transition, CellMeshTransition):
        raise TypeError("prior_transition must be CellMeshTransition or None.")
    source = prior_transition.target if source_result is None else source_result
    if not isinstance(source, CellMeshingResult):
        raise TypeError("source_result must be CellMeshingResult or None.")
    if (
        prior_transition.target.result_id != source.result_id
        or source.mesh.mesh_id != mesh.mesh_id
        or source.mesh.topology_id != mesh.topology_id
    ):
        raise ValueError(
            "prior_transition must target the exact mesh/result being curved."
        )
    return source


def verify_curved_geometry(
    geometry: CellGeometrySpec,
    mesh: CellMesh,
    association: GeometryAssociation,
    projection: AbstractBRepProjection,
    /,
    *,
    policy: HighOrderCurvingPolicy,
    source_result: CellMeshingResult | None = None,
    prior_transition: CellMeshTransition | None = None,
) -> CurvedGeometryEvidence:
    """Certify an existing high-order map and retained source/material claims.

    Nodes are classified through their owning mesh entities and vertex association.
    ``source_result`` renews its target-bound acceptance; ``prior_transition`` must
    target that exact result and may carry explicit material-reference authority.
    Nothing is moved.
    """
    _require_inputs(mesh, association, projection, policy)
    source_result = _curving_source_result(mesh, source_result, prior_transition)
    if not isinstance(geometry, CellGeometrySpec):
        raise TypeError("geometry must be CellGeometrySpec.")
    classes = _node_classes(mesh, geometry, association, projection)
    # ty: ignore[unresolved-attribute]
    degree = max(int(element.degree) for element in geometry.elements)
    blocks = _distortion_blocks(mesh, geometry, degree)
    return _evidence(
        mesh,
        geometry,
        classes,
        projection,
        blocks,
        policy,
        source_result=source_result,
        prior_transition=prior_transition,
    )


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
    ambient = mesh.ambient_dimension
    if not periodic:
        empty = np.zeros((0,), dtype=np.int64)
        return _PeriodicNodes(
            empty,
            empty,
            np.zeros((0, ambient, ambient), dtype=np.float64),
            np.zeros((0, ambient), dtype=np.float64),
        )
    owner_dims, owner_rows = _node_owners(mesh, straight)
    coordinates = np.asarray(straight.coordinates, dtype=np.float64)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
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
    projection: AbstractBRepProjection,
    coordinates: np.ndarray,
    classes: _NodeClasses,
    periodic: _PeriodicNodes,
    /,
    *,
    cache: _NodeProjectionCache | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-project movable constrained nodes; periodic targets follow their sources."""
    rows, feet, _, projectors = _node_residuals(
        projection, coordinates, classes, cache=cache
    )
    result = coordinates.copy()
    movable = ~classes.fixed[rows]
    result[rows[movable]] = feet[movable]
    return _apply_periodic(result, periodic), rows, projectors


def curve_cell_mesh(
    mesh: CellMesh,
    association: GeometryAssociation,
    projection: AbstractBRepProjection,
    /,
    *,
    policy: HighOrderCurvingPolicy,
    periodic: tuple[PeriodicCoupling, ...] = (),
    source_result: CellMeshingResult | None = None,
    prior_transition: CellMeshTransition | None = None,
) -> HighOrderCurvingResult:
    """Curve a mesh with local, global, source, and optional material acceptance.

    ``source_result`` retains target-bound certification. ``prior_transition`` must
    target that exact source result and is required to continue any material
    declaration/chain; a bare result never authors material identity. When
    retained, its exact coordinate map supplies every initial nodal image.

    This coordinator keeps candidate selection and rollback-owned caches in one
    invocation; numerical preparation, projection and proof phases stay in their
    owning helpers so their ordering and accepted-state lifetime remain explicit.
    """
    _require_inputs(mesh, association, projection, policy)
    source_result = _curving_source_result(mesh, source_result, prior_transition)
    straight = _straight_geometry(
        mesh,
        policy.degree,
        source_geometry=None if source_result is None else source_result.geometry,
    )
    classes = _node_classes(mesh, straight, association, projection)
    blocks = _distortion_blocks(mesh, straight, policy.degree)
    projection_cache = _NodeProjectionCache(projection, classes)
    straight_projection_cache = projection_cache

    def rollback(
        status: HighOrderCurvingStatus,
        candidate: CurvedGeometryEvidence | None,
        minimizations: tuple[MinimizationResult, ...],
        /,
    ) -> HighOrderCurvingResult:
        # The straight map's evidence is needed only when it is the returned geometry.
        evidence = _evidence(
            mesh,
            straight,
            classes,
            projection,
            blocks,
            policy,
            source_result=source_result,
            prior_transition=prior_transition,
            cache=straight_projection_cache,
        )
        return HighOrderCurvingResult(
            status, straight, straight, evidence, candidate, minimizations, None
        )

    if not np.all(classes.resolved):
        return rollback(HighOrderCurvingStatus.UNRESOLVED_ASSOCIATION, None, ())
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
        projection,
        np.asarray(straight.coordinates, dtype=np.float64),
        classes,
        pairs,
        cache=projection_cache,
    )
    # Cache updates replace their arrays. Retain the initial exact-query rows
    # without copying them so rollback does not repeat source projections.
    straight_projection_cache = replace(projection_cache)

    def candidate_geometry(values: np.ndarray, /) -> CellGeometrySpec:
        return _geometry_with_exact_coordinate_bank(
            mesh,
            layout[0],
            layout[1],
            straight,
            values,
            identity_context=("curving-candidate",),
        )

    # Candidate maps with their local evidence: the CAD projection (round 0, which
    # needs no relaxation), every admissible relaxation round and the last round.
    candidates: list[tuple[int, np.ndarray, CellGeometrySpec, _LocalEvidence, bool]] = []

    def record(round_: int, values: np.ndarray, admissible: bool, /) -> None:
        geometry = candidate_geometry(values)
        # Cells the relaxation left exactly unchanged keep the previous proof.
        local = _local_evidence(
            mesh,
            geometry,
            classes,
            projection,
            blocks,
            policy,
            parent=candidates[-1][3].computed_validity if candidates else None,
            cache=projection_cache,
        )
        candidates.append((round_, values, geometry, local, admissible))

    record(0, coordinates, True)
    minimizations = []
    for round_ in range(1, policy.relaxation_rounds + 1):
        # Round k relaxes the k-ring star of the cells the latest certified
        # candidate leaves unproven; relaxation ends once every cell is proven.
        certificate = candidates[-1][3].computed_validity.certificate
        if certificate.all_certified:
            break
        movable, star_blocks = _relaxation_star(blocks, certificate, fixed, round_)
        if not np.any(movable):
            break
        relaxed_constraints = movable[constrained_rows]
        relaxed_rows = constrained_rows[relaxed_constraints]
        energy = _CurvingEnergy(
            blocks=star_blocks,
            constrained_rows=jnp.asarray(relaxed_rows, dtype=jnp.int32),
            feet=jnp.asarray(coordinates[relaxed_rows]),
            normal_projectors=jnp.asarray(projectors[relaxed_constraints]),
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
        # Displacements in mean-edge units make the trust radius and the step
        # tolerances independent of the model's length unit.
        relaxed = optimize_cell_geometry_coordinates(
            candidate_geometry(coordinates),
            energy,
            fixed_coordinates=~movable,
            method=NewtonTrustRegion(),
            termination=policy.termination,
            length_scale=length,
        )
        minimizations.append(relaxed.minimization)
        moved = np.asarray(relaxed.coordinates, dtype=np.float64)
        if not np.all(np.isfinite(moved)):
            break
        coordinates, constrained_rows, projectors = _project_constrained(
            projection,
            moved,
            classes,
            pairs,
            cache=projection_cache,
        )
        admissible = relaxed.converged or policy.accept_valid_nonconverged_relaxation
        if not admissible and round_ < policy.relaxation_rounds:
            # An intermediate candidate that may not replace the geometry needs
            # no evidence; the last round's candidate is always certified.
            continue
        record(round_, coordinates, admissible)

    def full_evidence(
        geometry: CellGeometrySpec, local: _LocalEvidence, /
    ) -> CurvedGeometryEvidence:
        return _evidence(
            mesh,
            geometry,
            classes,
            projection,
            blocks,
            policy,
            source_result=source_result,
            prior_transition=prior_transition,
            local=local,
        )

    # The global embedding and source acceptance dominate the cost, so they run
    # on locally passing admissible candidates latest round first and stop at the
    # first accepted map: the map that certifying every round and keeping the
    # latest accepted one would select.
    accepted: tuple[CellGeometrySpec, CurvedGeometryEvidence, int] | None = None
    final: CurvedGeometryEvidence | None = None
    for index in range(len(candidates) - 1, -1, -1):
        round_, _, geometry, local, admissible = candidates[index]
        if not (admissible and local.passes(policy)):
            continue
        evidence = full_evidence(geometry, local)
        if index == len(candidates) - 1:
            final = evidence
        if evidence.accepted:
            accepted = (geometry, evidence, round_)
            break
    if accepted is not None:
        # The accepted map is the last fully certified candidate.
        return HighOrderCurvingResult(
            HighOrderCurvingStatus.CURVED,
            accepted[0],
            straight,
            accepted[1],
            accepted[1],
            tuple(minimizations),
            accepted[2],
            cell_result=accepted[1].target_result,
        )
    # After a rollback the last round's candidate is fully certified: it names
    # the refusal below.
    candidate = (
        full_evidence(candidates[-1][2], candidates[-1][3]) if final is None else final
    )
    # Without an accepted geometry, the last candidate names the first refusal in
    # the order: local cell validity, global embedding, CAD residual, retained
    # source acceptance; a candidate passing all of them was refused only
    # because its relaxation did not converge.
    if not candidate.certificate.all_certified:
        status = HighOrderCurvingStatus.ROLLED_BACK_INVALID
    elif candidate.embedding.status != "certified":
        status = HighOrderCurvingStatus.ROLLED_BACK_EMBEDDING
    elif candidate.maximum_residual > candidate.residual_tolerance:
        status = HighOrderCurvingStatus.ROLLED_BACK_RESIDUAL
    elif (
        candidate.source_failure is not None
        or (
            candidate.source_certification is not None
            and not candidate.source_certification.passed
        )
        or (source_result is not None and candidate.target_result is None)
    ):
        status = HighOrderCurvingStatus.ROLLED_BACK_SOURCE_ACCEPTANCE
    else:
        status = HighOrderCurvingStatus.ROLLED_BACK_NONCONVERGED
    return rollback(status, candidate, tuple(minimizations))


__all__ = [
    "CurvedGeometryEvidence",
    "HighOrderCurvingPolicy",
    "HighOrderCurvingResult",
    "HighOrderCurvingStatus",
    "curve_cell_mesh",
    "verify_curved_geometry",
]
