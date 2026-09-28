#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Quasi-static equilibrium of explicit soap films and dry foams.

The equilibrium problem is

``minimize E(x) = sum_f gamma_f A_f(x)  subject to  V_r(x) = V_r^0``

over the free vertex coordinates of a fixed-topology multiregion surface, with
wire frames prescribing selected coordinates. The native constrained
optimizers use ``L = E + lambda^T c``, so the pressure of region ``r`` relative
to the boundary labels is ``p_r = -lambda_r`` (``dE = sum_r p_r dV_r``); a
spherical soap bubble recovers the Laplace law ``p = 2 gamma / R = 4 sigma / R``.

**Independent constraints and pressure gauge.** When the free coordinates
cannot change the total volume of a set of finite regions (a partition of a
rigid container), their Jacobian rows are linearly dependent and the region
pressures are only defined up to a common constant. Preparation selects a
maximal independent row set in table order with a native rank decision; each
dropped region becomes the zero-pressure reference of its dependent set and
its volume target is certified after the solve. Regions touching a free
boundary label are referenced to that label (the ambient) directly.

**Geometric gauge.** A free foam (no wires) is invariant under rigid motions,
which makes its KKT matrix singular. Six linear gauge rows fix the mean
displacement and the linearized rotation about the reference centroid; their
multipliers vanish at equilibrium (force and torque balance) and are reported.
An interior vertex of an open film has two tangential reparameterization modes.
The same modes become exact at the flat limit for a film separating two finite
cells. Preparation fixes those modes with a deterministic orthonormal tangent
basis. KKT rank and inertia are therefore evaluated on the physical quotient
rather than treating mesh-parameterization modes as failed physics.

**Derivatives.** `PreparedFoamEquilibrium.implicit_equilibrium` returns
fixed-topology equilibrium positions, pressures and energy differentiable with
respect to tensions, volume targets and wire positions through the native
`implicit_constrained_minimize`, and is refused unless the solve converged and
the quotient KKT evidence certifies full rank, inertia ``(n, m, 0)`` and bounded
conditioning.
"""

from __future__ import annotations

from enum import IntEnum
from typing import assert_never, final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState, parameter_field, ParameterOwner
from ...geometry.multiregion_surface import (
    MultiRegionSurfaceState,
    MultiRegionSurfaceTopology,
    PreparedMultiRegionSurface,
)
from ...linalg import (
    DenseLinearOperator,
    DenseLU,
    DensePropertyVerificationPolicy,
    FunctionLinearOperator,
    JacobianLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    MaterializationPolicy,
    materialize,
    prepare as prepare_linear,
    prepare_linearization,
    solve as solve_linear,
    verify_dense_properties,
)
from ...optim import (
    AbstractMinimizationMethod,
    AbstractScalarIterativeMethod,
    AugmentedLagrangian,
    implicit_constrained_minimize,
    implicit_minimize,
    MinimizationProblem,
    minimize,
    NewtonTrustRegion,
    NonlinearConstraint,
    OptimizationStatus,
    OptimizationTermination,
    SQP,
)
from ...typing import Bool, Dim, Float, Float64, Identifier, Int32, Scalar
from ._contracts import (
    FoamDerivativeUnavailableError,
    FoamEquilibriumEvidence,
    FoamEquilibriumPlan,
    FoamEquilibriumResult,
    FoamEquilibriumRoute,
    FoamEquilibriumStatus,
    FoamKKTStatus,
    FoamMaterialPlan,
)


_GAUGE_ROWS = 6


class _TensionLabelDim(Dim, minimum=2):
    """Tension labels."""


class _TargetDim(Dim):
    """Finite regions (volume targets)."""


class _WireValueDim(Dim):
    """Wire vertices."""


class _FreeCoordinateDim(Dim, minimum=1):
    """Free vertex coordinates."""


class _TangentialGaugeDim(Dim):
    """Open- and flat-film tangential gauge rows."""


class _FlatCoordinateDim(Dim, minimum=3):
    """Flattened capacity coordinates."""


class _FaceSlotDim(Dim, minimum=1):
    """Face slots."""


class _EdgeSlotDim(Dim, minimum=1):
    """Edge slots."""


class _FixedCoordinateDim(Dim):
    """Wire-fixed coordinates."""


class _RegionSlotDim(Dim, minimum=2):
    """Region slots."""


class _VertexSlotDim(Dim, minimum=3):
    """Vertex slots."""


@final
class FoamEquilibriumParameters(StrictModule, ParameterOwner):
    """Dynamic physical inputs of one equilibrium: the differentiable leaves.

    ``tension_values`` has the structure of the material tension matrix,
    ``volume_targets`` follows the surface's finite regions in table order, and
    ``wire_positions`` follows the material wire vertices (zero rows without
    wires).
    """

    __strict_contract__ = True

    tension_values: Float64[_TensionLabelDim, _TensionLabelDim] | Float64[Scalar] = (
        parameter_field()
    )
    volume_targets: Float64[_TargetDim] = parameter_field()
    wire_positions: Float64[_WireValueDim, Literal[3]] = parameter_field()

    def __init__(
        self,
        tension_values: ArrayLike,
        volume_targets: ArrayLike,
        wire_positions: ArrayLike,
        /,
    ) -> None:
        self.tension_values = jnp.asarray(tension_values, dtype=jnp.float64)
        self.volume_targets = jnp.asarray(volume_targets, dtype=jnp.float64)
        self.wire_positions = jnp.asarray(wire_positions, dtype=jnp.float64)


@final
class FoamImplicitEquilibrium(StrictModule):
    """Differentiable fixed-topology equilibrium quantities."""

    __strict_contract__ = True

    positions: Float[_VertexSlotDim, Literal[3]]
    pressures: Float[_RegionSlotDim]
    region_volumes: Float[_RegionSlotDim]
    energy: Float[Scalar]


def _host_flat(values: Array, /) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _independent_rows(jacobian: np.ndarray, /) -> tuple[int, ...]:
    """Greedy maximal independent row set in table order (native rank decisions)."""
    kept: list[int] = []
    for row in range(jacobian.shape[0]):
        candidate = jacobian[kept + [row]]
        gram = candidate @ candidate.T
        rank = int(verify_dense_properties(jnp.asarray(gram)).numerical_rank)
        if rank == len(kept) + 1:
            kept.append(row)
    return tuple(kept)


class FoamConstraintBasisStatus(IntEnum):
    """Outcome of bounded finite-region constraint-basis preparation."""

    ACCEPTED = 0
    CONSTRAINT_ENTRIES_EXCEEDED = 1
    PREPARATION_BYTES_EXCEEDED = 2
    RANK_CHECK_ACTIONS_EXCEEDED = 3
    NUMERIC_RANK_DEFICIENT = 4
    NUMERIC_RANK_INVALID = 5
    NOT_APPLICABLE = 6


@final
class FoamConstraintBasisEvidence(StrictModule, NonTrainableState):
    """Topology, numerical-rank, and resource evidence for one volume basis."""

    status: FoamConstraintBasisStatus = eqx.field(static=True)
    accepted: bool = eqx.field(static=True)
    finite_region_count: int = eqx.field(static=True)
    partition_count: int = eqx.field(static=True)
    closed_partition_count: int = eqx.field(static=True)
    constrained_count: int = eqx.field(static=True)
    dependent_count: int = eqx.field(static=True)
    numerical_rank: int = eqx.field(static=True)
    rank_condition: float | None = eqx.field(static=True)
    required_constraint_entries: int = eqx.field(static=True)
    maximum_constraint_entries: int = eqx.field(static=True)
    rank_check_actions: int = eqx.field(static=True)
    maximum_rank_check_actions: int = eqx.field(static=True)
    preparation_bytes: int = eqx.field(static=True)
    maximum_preparation_bytes: int = eqx.field(static=True)
    logical_retained_bytes: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class FoamConstraintBasisPreparationError(ValueError):
    """A finite-region constraint basis failed bounded host preparation."""

    def __init__(self, evidence: FoamConstraintBasisEvidence, /) -> None:
        self.evidence = evidence
        super().__init__(
            "Foam volume constraint-basis preparation failed with status "
            f"{evidence.status.name}: entries "
            f"{evidence.required_constraint_entries}/"
            f"{evidence.maximum_constraint_entries}, actions "
            f"{evidence.rank_check_actions}/{evidence.maximum_rank_check_actions}, "
            f"bytes {evidence.preparation_bytes}/"
            f"{evidence.maximum_preparation_bytes}."
        )


@final
class FoamVolumeConstraintBasis(StrictModule, NonTrainableState):
    """Canonical independent finite-region rows and pressure references."""

    evidence: FoamConstraintBasisEvidence
    finite_region_slots: tuple[int, ...] = eqx.field(static=True)
    constrained_rows: tuple[int, ...] = eqx.field(static=True)
    dependent_rows: tuple[int, ...] = eqx.field(static=True)
    closed_partitions: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)


def _finite_region_partitions(
    topology: MultiRegionSurfaceTopology, /
) -> tuple[tuple[int, ...], ...]:
    """Connected finite-region row partitions from sparse face incidence."""
    finite = topology.finite_region_indices
    if not finite:
        return ()
    row_of = {slot: row for row, slot in enumerate(finite)}
    parent = list(range(len(finite)))

    def root(row: int, /) -> int:
        while parent[row] != row:
            parent[row] = parent[parent[row]]
            row = parent[row]
        return row

    def unite(first: int, second: int, /) -> None:
        first_root = root(first)
        second_root = root(second)
        if first_root == second_root:
            return
        lower, upper = sorted((first_root, second_root))
        parent[upper] = lower

    for first, second in topology.host_face_labels().tolist():
        if first in row_of and second in row_of:
            unite(row_of[first], row_of[second])
    groups: dict[int, list[int]] = {}
    for row in range(len(finite)):
        groups.setdefault(root(row), []).append(row)
    return tuple(tuple(groups[key]) for key in sorted(groups))


def _closed_finite_partitions(
    surface: PreparedMultiRegionSurface,
    positions: np.ndarray,
    free: np.ndarray,
    partitions: tuple[tuple[int, ...], ...],
    /,
) -> tuple[tuple[int, ...], ...]:
    """Partitions whose ambient boundary has zero free-coordinate volume action."""
    if not partitions:
        return ()
    topology = surface.topology
    finite = topology.finite_region_indices
    component_of_slot = np.full((topology.region_count,), -1, dtype=np.int64)
    for component, rows in enumerate(partitions):
        for row in rows:
            component_of_slot[finite[row]] = component
    free_mask = np.zeros((positions.size,), dtype=np.bool_)
    free_mask[free] = True
    faces = topology.host_faces()
    labels = topology.host_face_labels()
    reference = np.asarray(surface.volume_reference, dtype=np.float64)
    gradients: list[dict[int, np.ndarray]] = [{} for _ in partitions]
    for face, (first, second) in zip(faces, labels, strict=True):
        first_component = component_of_slot[first]
        second_component = component_of_slot[second]
        if first_component == second_component:
            continue
        component = first_component if first_component >= 0 else second_component
        if component < 0:
            continue
        sign = 1.0 if first_component >= 0 else -1.0
        corners = positions[face] - reference
        local = (
            sign
            * np.asarray(
                (
                    np.cross(corners[1], corners[2]),
                    np.cross(corners[2], corners[0]),
                    np.cross(corners[0], corners[1]),
                )
            )
            / 6.0
        )
        for vertex, gradient in zip(face.tolist(), local, strict=True):
            current = gradients[component].get(vertex)
            gradients[component][vertex] = (
                gradient.copy() if current is None else current + gradient
            )
    closed: list[tuple[int, ...]] = []
    epsilon = np.finfo(np.float64).eps
    for rows, component_gradients in zip(partitions, gradients, strict=True):
        scale = max(
            (float(np.max(np.abs(value))) for value in component_gradients.values()),
            default=0.0,
        )
        free_scale = max(
            (
                float(np.max(np.abs(value[free_mask[3 * vertex : 3 * vertex + 3]])))
                for vertex, value in component_gradients.items()
                if np.any(free_mask[3 * vertex : 3 * vertex + 3])
            ),
            default=0.0,
        )
        tolerance = 128.0 * epsilon * max(scale, np.finfo(np.float64).tiny)
        if free_scale <= tolerance:
            closed.append(rows)
    return tuple(closed)


def _constraint_basis_evidence(
    *,
    status: FoamConstraintBasisStatus,
    topology: MultiRegionSurfaceTopology,
    partitions: tuple[tuple[int, ...], ...],
    closed: tuple[tuple[int, ...], ...],
    constrained: tuple[int, ...],
    dependent: tuple[int, ...],
    numerical_rank: int,
    rank_condition: float | None,
    maximum_constraint_entries: int,
    maximum_rank_check_actions: int,
    maximum_preparation_bytes: int,
    free_coordinate_count: int,
) -> FoamConstraintBasisEvidence:
    finite_count = len(topology.finite_region_indices)
    constrained_count = len(constrained)
    applicable = status != FoamConstraintBasisStatus.NOT_APPLICABLE
    entries = constrained_count**2 if applicable else 0
    actions = 2 * constrained_count if applicable else 0
    scalar_bytes = np.dtype(np.float64).itemsize
    preparation_bytes = (
        scalar_bytes * (entries + 2 * free_coordinate_count + 4 * constrained_count)
        if applicable
        else 0
    )
    retained_bytes = (
        np.dtype(np.int32).itemsize * (finite_count + len(constrained) + len(dependent))
        if applicable
        else 0
    )
    accepted = status in (
        FoamConstraintBasisStatus.ACCEPTED,
        FoamConstraintBasisStatus.NOT_APPLICABLE,
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "foam-constraint-basis-evidence",
            "status": int(status),
            "topology": topology.topology_id,
            "partitions": [list(rows) for rows in partitions],
            "closed": [list(rows) for rows in closed],
            "constrained": list(constrained),
            "dependent": list(dependent),
            "numerical_rank": numerical_rank,
            "rank_condition": (
                None if rank_condition is None else float(rank_condition).hex()
            ),
            "required_constraint_entries": entries,
            "maximum_constraint_entries": maximum_constraint_entries,
            "rank_check_actions": actions,
            "maximum_rank_check_actions": maximum_rank_check_actions,
            "preparation_bytes": preparation_bytes,
            "maximum_preparation_bytes": maximum_preparation_bytes,
            "logical_retained_bytes": retained_bytes,
        }
    )
    return FoamConstraintBasisEvidence(
        status=status,
        accepted=accepted,
        finite_region_count=finite_count,
        partition_count=len(partitions),
        closed_partition_count=len(closed),
        constrained_count=constrained_count,
        dependent_count=len(dependent),
        numerical_rank=numerical_rank,
        rank_condition=rank_condition,
        required_constraint_entries=entries,
        maximum_constraint_entries=maximum_constraint_entries,
        rank_check_actions=actions,
        maximum_rank_check_actions=maximum_rank_check_actions,
        preparation_bytes=preparation_bytes,
        maximum_preparation_bytes=maximum_preparation_bytes,
        logical_retained_bytes=retained_bytes,
        topology_id=topology.topology_id,
        evidence_id=evidence_id,
    )


def _not_applicable_foam_volume_constraint_basis(
    topology: MultiRegionSurfaceTopology,
    /,
    *,
    maximum_constraint_entries: int,
    maximum_rank_check_actions: int,
    maximum_preparation_bytes: int,
) -> FoamVolumeConstraintBasis:
    """Record that the selected dynamics route has no equality-volume basis."""
    evidence = _constraint_basis_evidence(
        status=FoamConstraintBasisStatus.NOT_APPLICABLE,
        topology=topology,
        partitions=(),
        closed=(),
        constrained=(),
        dependent=(),
        numerical_rank=0,
        rank_condition=None,
        maximum_constraint_entries=maximum_constraint_entries,
        maximum_rank_check_actions=maximum_rank_check_actions,
        maximum_preparation_bytes=maximum_preparation_bytes,
        free_coordinate_count=0,
    )
    return FoamVolumeConstraintBasis(
        evidence=evidence,
        finite_region_slots=(),
        constrained_rows=(),
        dependent_rows=(),
        closed_partitions=(),
        basis_id=evidence.evidence_id,
    )


def prepare_foam_volume_constraint_basis(
    surface: PreparedMultiRegionSurface,
    positions: np.ndarray,
    free: np.ndarray,
    /,
    *,
    maximum_constraint_entries: int,
    maximum_rank_check_actions: int,
    maximum_preparation_bytes: int,
    rank_relative_tolerance: float,
) -> FoamVolumeConstraintBasis:
    """Prepare a topology-first basis with one bounded matrix-free Gram check."""
    topology = surface.topology
    finite = topology.finite_region_indices
    partitions = _finite_region_partitions(topology)
    closed = _closed_finite_partitions(surface, positions, free, partitions)
    dependent = tuple(rows[-1] for rows in closed)
    dependent_set = set(dependent)
    constrained = tuple(row for row in range(len(finite)) if row not in dependent_set)

    status = FoamConstraintBasisStatus.ACCEPTED
    entries = len(constrained) ** 2
    scalar_bytes = np.dtype(np.float64).itemsize
    preparation_bytes = scalar_bytes * (entries + 2 * free.size + 4 * len(constrained))
    if entries > maximum_constraint_entries:
        status = FoamConstraintBasisStatus.CONSTRAINT_ENTRIES_EXCEEDED
    elif preparation_bytes > maximum_preparation_bytes:
        status = FoamConstraintBasisStatus.PREPARATION_BYTES_EXCEEDED
    elif 2 * len(constrained) > maximum_rank_check_actions:
        status = FoamConstraintBasisStatus.RANK_CHECK_ACTIONS_EXCEEDED
    elif len(constrained) > free.size:
        status = FoamConstraintBasisStatus.NUMERIC_RANK_DEFICIENT
    if status != FoamConstraintBasisStatus.ACCEPTED:
        evidence = _constraint_basis_evidence(
            status=status,
            topology=topology,
            partitions=partitions,
            closed=closed,
            constrained=constrained,
            dependent=dependent,
            numerical_rank=-1,
            rank_condition=None,
            maximum_constraint_entries=maximum_constraint_entries,
            maximum_rank_check_actions=maximum_rank_check_actions,
            maximum_preparation_bytes=maximum_preparation_bytes,
            free_coordinate_count=free.size,
        )
        raise FoamConstraintBasisPreparationError(evidence)

    numerical_rank = 0
    rank_condition = 1.0
    if constrained:
        slots = jnp.asarray([finite[row] for row in constrained], dtype=jnp.int32)
        base = jnp.asarray(positions.reshape(-1), dtype=jnp.float64)
        indices = jnp.asarray(free, dtype=jnp.int32)

        def selected_volumes(values: Array, /) -> Array:
            flat = base.at[indices].set(values)
            return surface.region_volumes(flat.reshape(positions.shape))[slots]

        linearization = prepare_linearization(
            selected_volumes,
            base[indices],
            linearization_id=(
                f"{surface.prepared_id}:constraint-basis-volume-linearization"
            ),
        )
        operator = JacobianLinearOperator(
            linearization,
            operator_id=f"{surface.prepared_id}:constraint-basis-volume-jacobian",
        )

        def gram_action(weights: Array, /) -> Array:
            return operator.mv(operator.transpose_mv(weights))

        gram_operator = FunctionLinearOperator(
            gram_action,
            source=operator.target,
            target=operator.target,
            transpose_action=gram_action,
            operator_id=f"{surface.prepared_id}:constraint-basis-volume-gram",
        )
        gram = materialize(
            gram_operator,
            MaterializationPolicy(
                max_entries=maximum_constraint_entries,
                max_bytes=maximum_preparation_bytes,
            ),
        )
        properties = verify_dense_properties(
            gram,
            policy=DensePropertyVerificationPolicy(
                require_positive_semidefinite=True,
                relative_tolerance=(rank_relative_tolerance / np.finfo(np.float64).eps),
            ),
        )
        numerical_rank = int(np.asarray(properties.numerical_rank))
        rank_condition = float(np.asarray(properties.condition_estimate))
        if not bool(np.asarray(properties.successful)):
            status = FoamConstraintBasisStatus.NUMERIC_RANK_INVALID
        elif numerical_rank != len(constrained):
            status = FoamConstraintBasisStatus.NUMERIC_RANK_DEFICIENT
    evidence = _constraint_basis_evidence(
        status=status,
        topology=topology,
        partitions=partitions,
        closed=closed,
        constrained=constrained,
        dependent=dependent,
        numerical_rank=numerical_rank,
        rank_condition=rank_condition,
        maximum_constraint_entries=maximum_constraint_entries,
        maximum_rank_check_actions=maximum_rank_check_actions,
        maximum_preparation_bytes=maximum_preparation_bytes,
        free_coordinate_count=free.size,
    )
    if not evidence.accepted:
        raise FoamConstraintBasisPreparationError(evidence)
    return FoamVolumeConstraintBasis(
        evidence=evidence,
        finite_region_slots=finite,
        constrained_rows=constrained,
        dependent_rows=dependent,
        closed_partitions=closed,
        basis_id=evidence.evidence_id,
    )


def _tangential_gauge_basis(
    topology: MultiRegionSurfaceTopology,
    positions: np.ndarray,
    free: np.ndarray,
    relative_tolerance: float,
    /,
) -> np.ndarray:
    """Deterministic tangent rows for open and flat finite-cell sheets."""
    vertices = topology.vertex_count
    faces = topology.host_faces()
    labels = np.sort(topology.host_face_labels(), axis=1)
    pair_keys = labels[:, 0] * topology.region_count + labels[:, 1]
    tangential_pairs: set[int] = set()
    finite = np.asarray(topology.region_finite)
    for pair in np.unique(pair_keys).tolist():
        pair_faces = np.flatnonzero(pair_keys == pair)
        pair_labels = labels[pair_faces[0]]
        pair_finite = finite[pair_labels]
        if not np.any(pair_finite):
            tangential_pairs.add(int(pair))
            continue
        if not np.all(pair_finite):
            continue
        pair_vertices = np.unique(faces[pair_faces])
        pair_points = positions[pair_vertices]
        pair_corners = positions[faces[pair_faces]]
        pair_normals = np.cross(
            pair_corners[:, 1] - pair_corners[:, 0],
            pair_corners[:, 2] - pair_corners[:, 0],
        )
        pair_normal = np.sum(pair_normals, axis=0)
        pair_norm = float(np.linalg.norm(pair_normal))
        pair_scale = float(np.linalg.norm(np.ptp(pair_points, axis=0)))
        if pair_norm <= 0.0 or pair_scale <= 0.0:
            continue
        pair_origin = np.mean(pair_points, axis=0)
        pair_distance = np.abs((pair_points - pair_origin) @ (pair_normal / pair_norm))
        if float(np.max(pair_distance)) <= relative_tolerance * pair_scale:
            tangential_pairs.add(int(pair))
    edges = np.asarray(topology.edges[: topology.edge_count], dtype=np.int64)
    edge_faces = np.asarray(topology.edge_faces[: topology.edge_count], dtype=np.int64)
    free_rank = np.full((3 * vertices,), -1, dtype=np.int64)
    free_rank[free] = np.arange(free.size)
    vertex_faces: list[list[int]] = [[] for _ in range(vertices)]
    for face, corners in enumerate(faces.tolist()):
        for vertex in corners:
            vertex_faces[vertex].append(face)
    vertex_edges: list[list[int]] = [[] for _ in range(vertices)]
    for edge, ends in enumerate(edges.tolist()):
        vertex_edges[ends[0]].append(edge)
        vertex_edges[ends[1]].append(edge)
    if not 0.0 < relative_tolerance < 0.25:
        raise ValueError("The tangential gauge tolerance must lie in (0, 0.25).")
    rows: list[np.ndarray] = []
    for vertex in range(vertices):
        coordinate_slots = 3 * vertex + np.arange(3)
        ranks = free_rank[coordinate_slots]
        incident_faces = np.asarray(vertex_faces[vertex], dtype=np.int64)
        incident_edges = np.asarray(vertex_edges[vertex], dtype=np.int64)
        if (
            np.any(ranks < 0)
            or incident_faces.size < 3
            or incident_edges.size < 3
            or np.unique(labels[incident_faces], axis=0).shape[0] != 1
            or int(pair_keys[incident_faces[0]]) not in tangential_pairs
            or np.any(np.sum(edge_faces[incident_edges] >= 0, axis=1) != 2)
        ):
            continue
        neighbors = np.unique(edges[incident_edges].reshape(-1))
        neighbors = neighbors[neighbors != vertex]
        corners = positions[faces[incident_faces]]
        normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        normal = np.sum(normals, axis=0)
        norm = float(np.linalg.norm(normal))
        scale = float(
            np.max(np.linalg.norm(positions[neighbors] - positions[vertex], axis=1))
        )
        if norm <= 0.0 or scale <= 0.0:
            continue
        unit_normal = normal / norm
        axis = np.zeros((3,), dtype=np.float64)
        axis[int(np.argmin(np.abs(unit_normal)))] = 1.0
        first = np.cross(unit_normal, axis)
        first /= np.linalg.norm(first)
        second = np.cross(unit_normal, first)
        for tangent in (first, second):
            row = np.zeros((free.size,), dtype=np.float64)
            row[ranks] = tangent
            rows.append(row)
    return (
        np.asarray(rows, dtype=np.float64)
        if rows
        else np.zeros((0, free.size), dtype=np.float64)
    )


@final
class PreparedFoamEquilibrium(StrictModule):
    """Prepared equilibrium problem of one foam topology epoch."""

    __strict_contract__ = True

    surface: PreparedMultiRegionSurface
    material: FoamMaterialPlan
    plan: FoamEquilibriumPlan
    base_positions: Float64[_FlatCoordinateDim]
    free_indices: Int32[_FreeCoordinateDim]
    free_center: Float64[_FreeCoordinateDim]
    reference_free: Float64[_FreeCoordinateDim]
    tangential_gauge_basis: Float64[_TangentialGaugeDim, _FreeCoordinateDim]
    wire_flat_indices: Int32[_FixedCoordinateDim]
    wire_value_indices: Int32[_FixedCoordinateDim]
    face_first: Int32[_FaceSlotDim]
    face_second: Int32[_FaceSlotDim]
    junction_edges: Bool[_EdgeSlotDim]
    finite_region_slots: tuple[int, ...] = eqx.field(static=True)
    constrained_rows: tuple[int, ...] = eqx.field(static=True)
    dependent_rows: tuple[int, ...] = eqx.field(static=True)
    rigid_motion_gauge: bool = eqx.field(static=True)
    tangential_gauge_dimension: int = eqx.field(static=True)
    route: FoamEquilibriumRoute = eqx.field(static=True)
    length_scale: float = eqx.field(static=True)
    tension_scale: float = eqx.field(static=True)
    prepared_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        plan: FoamEquilibriumPlan,
        surface: PreparedMultiRegionSurface,
        material: FoamMaterialPlan,
        state: MultiRegionSurfaceState,
        /,
    ) -> None:
        if not isinstance(plan, FoamEquilibriumPlan):
            raise TypeError("plan must be a FoamEquilibriumPlan.")
        if not isinstance(surface, PreparedMultiRegionSurface):
            raise TypeError("surface must be a PreparedMultiRegionSurface.")
        if not isinstance(material, FoamMaterialPlan):
            raise TypeError("material must be a FoamMaterialPlan.")
        topology = surface.topology
        state.require_topology(topology)
        face_first, face_second = material.face_tension_indices(topology)
        vertices = topology.vertex_count
        positions = np.asarray(state.positions, dtype=np.float64)
        wire_flat, wire_values = _wire_coordinates(material, topology)
        active = np.arange(3 * vertices)
        free = np.setdiff1d(active, wire_flat)
        finite = topology.finite_region_indices
        rigid = material.wires is None
        if not finite and rigid:
            raise ValueError(
                "A foam without finite regions needs wire constraints; otherwise "
                "surface-energy minimization collapses the films."
            )
        center = np.mean(positions[:vertices], axis=0)
        tension_scale = float(np.max(np.asarray(material.tensions.values)))
        volumes = _host_flat(surface.region_volumes(state.positions))[list(finite)]
        length = (
            float(np.mean(np.abs(volumes))) ** (1.0 / 3.0)
            if finite
            else 0.5 * float(np.linalg.norm(np.ptp(positions[:vertices], axis=0)))
        )
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError("The foam geometry has no positive length scale.")
        free_center = np.tile(center, vertices)[free]
        reference_free = (positions.reshape(-1)[free] - free_center) / length
        rows = _volume_jacobian(surface, positions, free, finite)
        constrained = _independent_rows(rows) if finite else ()
        dependent = tuple(row for row in range(len(finite)) if row not in constrained)
        tangential_gauge = _tangential_gauge_basis(
            topology, positions, free, plan.tangential_gauge_tolerance
        )
        constraint_count = (
            len(constrained) + (_GAUGE_ROWS if rigid else 0) + tangential_gauge.shape[0]
        )
        route = _route(plan, constraint_count)
        if route == "sqp-exact-hessian" and free.size + constraint_count > (
            plan.maximum_dense_dimension
        ):
            raise ValueError(
                f"Dense SQP needs {free.size + constraint_count} KKT rows, above "
                f"maximum_dense_dimension={plan.maximum_dense_dimension}; use "
                "method='augmented_lagrangian'."
            )
        edge_faces = np.asarray(topology.edge_faces)
        edges = np.maximum(np.asarray(topology.edges, dtype=np.int64), 0)
        wired = np.zeros((topology.vertex_capacity,), dtype=np.bool_)
        wired[np.unique(wire_flat // 3)] = True
        self.surface = surface
        self.material = material
        self.plan = plan
        self.base_positions = jnp.asarray(positions.reshape(-1))
        self.free_indices = jnp.asarray(free, dtype=jnp.int32)
        self.free_center = jnp.asarray(free_center)
        self.reference_free = jnp.asarray(reference_free)
        self.tangential_gauge_basis = jnp.asarray(tangential_gauge)
        self.wire_flat_indices = jnp.asarray(wire_flat, dtype=jnp.int32)
        self.wire_value_indices = jnp.asarray(wire_values, dtype=jnp.int32)
        self.face_first = jnp.asarray(face_first, dtype=jnp.int32)
        self.face_second = jnp.asarray(face_second, dtype=jnp.int32)
        self.junction_edges = jnp.asarray(
            np.asarray(topology.edge_active)
            & (np.sum(edge_faces >= 0, axis=1) == 3)
            & ~(wired[edges[:, 0]] & wired[edges[:, 1]])
        )
        self.finite_region_slots = finite
        self.constrained_rows = constrained
        self.dependent_rows = dependent
        self.rigid_motion_gauge = rigid
        self.tangential_gauge_dimension = tangential_gauge.shape[0]
        self.route = route
        self.length_scale = length
        self.tension_scale = tension_scale
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-foam-equilibrium",
                "plan": plan.plan_id,
                "surface": surface.prepared_id,
                "material": material.material_id,
                "reference": array_tree_fingerprint(positions),
                "constrained_rows": list(constrained),
                "tangential_gauge": array_tree_fingerprint(tangential_gauge),
                "route": route,
            }
        )

    # --------------------------------------------------------------- parameters

    def parameters(
        self,
        volume_targets: ArrayLike,
        /,
        *,
        tension_values: ArrayLike | None = None,
        wire_positions: ArrayLike | None = None,
    ) -> FoamEquilibriumParameters:
        """Dynamic inputs; tensions and wire positions default to the material."""
        targets = jnp.asarray(volume_targets, dtype=jnp.float64)
        if targets.shape != (len(self.finite_region_slots),):
            raise ValueError(
                "volume_targets must list every finite region in table order."
            )
        tensions = (
            self.material.tensions.values
            if tension_values is None
            else jnp.asarray(tension_values, dtype=jnp.float64)
        )
        if tensions.shape != self.material.tensions.values.shape:
            raise ValueError("tension_values must match the material tension structure.")
        wires = self.material.wires
        default_wires = (
            jnp.zeros((0, 3), dtype=jnp.float64) if wires is None else wires.positions
        )
        positions = (
            default_wires
            if wire_positions is None
            else jnp.asarray(wire_positions, dtype=jnp.float64)
        )
        if positions.shape != default_wires.shape:
            raise ValueError("wire_positions must match the material wire vertices.")
        return FoamEquilibriumParameters(tensions, targets, positions)

    @property
    def constrained_region_ids(self) -> tuple[str, ...]:
        topology = self.surface.topology
        return tuple(
            topology.region_ids[self.finite_region_slots[row]]
            for row in self.constrained_rows
        )

    @property
    def pressure_reference_region_ids(self) -> tuple[str, ...]:
        topology = self.surface.topology
        return tuple(
            topology.region_ids[self.finite_region_slots[row]]
            for row in self.dependent_rows
        )

    # ------------------------------------------------------------ problem terms

    def positions(self, free: Array, parameters: FoamEquilibriumParameters, /) -> Array:
        """Capacity positions from nondimensional free coordinates and wires."""
        flat = self.base_positions
        if self.material.wires is not None:
            flat = flat.at[self.wire_flat_indices].set(
                parameters.wire_positions.reshape(-1)[self.wire_value_indices]
            )
        flat = flat.at[self.free_indices].set(self.free_center + self.length_scale * free)
        return flat.reshape((-1, 3))

    def face_tension(self, parameters: FoamEquilibriumParameters, /) -> Array:
        tensions = eqx.tree_at(
            lambda matrix: matrix.values,
            self.material.tensions,
            parameters.tension_values,
        )
        values = tensions.pair_values(self.face_first, self.face_second)
        return jnp.where(self.surface.topology.face_active, values, 0.0)

    def _energy(self, free: Array, parameters: FoamEquilibriumParameters, /) -> Array:
        energy = self.surface.surface_energy(
            self.positions(free, parameters), self.face_tension(parameters)
        )
        return energy / (self.tension_scale * self.length_scale**2)

    def _finite_volumes(
        self, free: Array, parameters: FoamEquilibriumParameters, /
    ) -> Array:
        volumes = self.surface.region_volumes(self.positions(free, parameters))
        return volumes[jnp.asarray(self.finite_region_slots, dtype=jnp.int32)]

    def _volume_residuals(
        self, free: Array, parameters: FoamEquilibriumParameters, /
    ) -> Array:
        rows = jnp.asarray(self.constrained_rows, dtype=jnp.int32)
        volumes = self._finite_volumes(free, parameters)
        return (volumes[rows] - parameters.volume_targets[rows]) / self.length_scale**3

    def _gauge(self, free: Array, parameters: FoamEquilibriumParameters, /) -> Array:
        del parameters
        parts = []
        if self.rigid_motion_gauge:
            current = free.reshape((-1, 3))
            reference = self.reference_free.reshape((-1, 3))
            translation = jnp.mean(current - reference, axis=0)
            rotation = jnp.mean(jnp.cross(reference, current), axis=0)
            parts.append(jnp.concatenate((translation, rotation)))
        if self.tangential_gauge_dimension:
            parts.append(self.tangential_gauge_basis @ (free - self.reference_free))
        return jnp.concatenate(parts)

    def _constraints(
        self, free: Array, parameters: FoamEquilibriumParameters, /
    ) -> Array:
        parts = []
        if self.constrained_rows:
            parts.append(self._volume_residuals(free, parameters))
        if self.gauge_count:
            parts.append(self._gauge(free, parameters))
        return jnp.concatenate(parts)

    @property
    def gauge_count(self) -> int:
        return (_GAUGE_ROWS if self.rigid_motion_gauge else 0) + (
            self.tangential_gauge_dimension
        )

    @property
    def constraint_count(self) -> int:
        return len(self.constrained_rows) + self.gauge_count

    def problem(self) -> MinimizationProblem:
        """Native minimization problem over nondimensional free coordinates."""
        constraints = (
            (
                NonlinearConstraint(
                    self._constraints,
                    lower=0.0,
                    upper=0.0,
                    constraint_id="foam-volume-and-gauge",
                ),
            )
            if self.constraint_count
            else ()
        )
        return MinimizationProblem(
            self._energy,
            constraints=constraints,
            problem_id=f"foam-equilibrium/{self.prepared_id}",
        )

    def method(self) -> AbstractMinimizationMethod:
        """Native optimizer of the prepared route."""
        match self.route:
            case "sqp-exact-hessian":
                return SQP(
                    hessian_update="exact",
                    max_dense_dimension=self.plan.maximum_dense_dimension,
                )
            case "augmented-lagrangian-trust-region":
                return AugmentedLagrangian(
                    inner_method=NewtonTrustRegion(),
                    maximum_outer_steps=self.plan.maximum_steps,
                    inner_maximum_steps=self.plan.maximum_steps,
                )
            case "unconstrained-trust-region":
                return NewtonTrustRegion()
            case _:
                assert_never(self.route)

    def termination(self) -> OptimizationTermination:
        # Nondimensional absolute KKT tolerance only: a relative term would let
        # the accepted volume residual grow with the initial infeasibility.
        return OptimizationTermination(
            absolute_optimality=self.plan.optimality_tolerance,
            relative_optimality=0.0,
            maximum_steps=self.plan.maximum_steps,
        )

    def free_coordinates(self, state: MultiRegionSurfaceState, /) -> Array:
        """Nondimensional free coordinates of one state."""
        state.require_topology(self.surface.topology)
        flat = state.positions.reshape(-1)
        return (flat[self.free_indices] - self.free_center) / self.length_scale

    # --------------------------------------------------------------- execution

    def solve(
        self, state: MultiRegionSurfaceState, parameters: FoamEquilibriumParameters, /
    ) -> FoamEquilibriumResult:
        """Solve for the equilibrium reachable from ``state`` at fixed topology."""
        if not isinstance(parameters, FoamEquilibriumParameters):
            raise TypeError("parameters must be FoamEquilibriumParameters.")
        free, multipliers, status, iterations = _minimize_foam(
            self, self.free_coordinates(state), parameters
        )
        evidence, pressures, volumes, energy = _equilibrium_evidence(
            self, free, multipliers, status, iterations, parameters
        )
        return FoamEquilibriumResult(
            state=state.with_positions(self.positions(free, parameters)),
            pressures=pressures,
            region_volumes=volumes,
            energy=energy,
            free_coordinates=free,
            multipliers=multipliers,
            evidence=evidence,
        )

    def implicit_equilibrium(
        self, result: FoamEquilibriumResult, parameters: FoamEquilibriumParameters, /
    ) -> FoamImplicitEquilibrium:
        """Fixed-topology equilibrium differentiable in ``parameters``.

        Refused with `FoamDerivativeUnavailableError` unless ``result``
        converged with regular KKT evidence (full rank, inertia ``(n, m, 0)``,
        bounded condition). The primal point is re-solved by the native method
        from ``result`` and differentiated through the KKT system by
        `implicit_constrained_minimize` (`implicit_minimize` without
        constraints); pressures are the stationarity multipliers
        ``lambda = -(J J^T)^{-1} J grad E`` at that point.
        """
        if not bool(result.evidence.derivative_available):
            raise FoamDerivativeUnavailableError(
                "Equilibrium derivatives need a converged solve with regular KKT "
                f"evidence (kkt_status={int(result.evidence.kkt_status)})."
            )
        start = jax.lax.stop_gradient(result.free_coordinates)
        if self.constraint_count:
            free = implicit_constrained_minimize(
                self.problem(),
                start,
                method=self.method(),
                termination=self.termination(),
                args=parameters,
            )
        else:
            method = self.method()
            if not isinstance(method, AbstractScalarIterativeMethod):
                raise RuntimeError(
                    "An unconstrained foam route requires a scalar iterative method."
                )
            free = implicit_minimize(
                self.problem(),
                start,
                method=method,
                termination=self.termination(),
                args=parameters,
            )
        multipliers = self._least_squares_multipliers(free, parameters)
        positions = self.positions(free, parameters)
        return FoamImplicitEquilibrium(
            positions=positions,
            pressures=self._pressures(multipliers),
            region_volumes=self.surface.region_volumes(positions),
            energy=self.surface.surface_energy(positions, self.face_tension(parameters)),
        )

    def _least_squares_multipliers(
        self, free: Array, parameters: FoamEquilibriumParameters, /
    ) -> Array:
        if not self.constraint_count:
            return jnp.zeros((0,), dtype=free.dtype)
        gradient = jax.grad(self._energy)(free, parameters)
        jacobian = jax.jacrev(self._constraints)(free, parameters)
        system = LinearSystem(DenseLinearOperator(jacobian @ jacobian.T))
        prepared = prepare_linear(system, LinearSolvePolicy(DenseLU()))
        return solve_linear(prepared, -(jacobian @ gradient)).value

    def _pressures(self, multipliers: Array, /) -> Array:
        """Region pressures ``-lambda gamma_ref / ell`` scattered to region slots."""
        topology = self.surface.topology
        pressures = jnp.zeros((topology.region_capacity,), dtype=multipliers.dtype)
        if not self.constrained_rows:
            return pressures
        slots = jnp.asarray(
            [self.finite_region_slots[row] for row in self.constrained_rows],
            dtype=jnp.int32,
        )
        scale = self.tension_scale / self.length_scale
        return pressures.at[slots].set(-scale * multipliers[: len(self.constrained_rows)])


def _wire_coordinates(
    material: FoamMaterialPlan, topology: MultiRegionSurfaceTopology, /
) -> tuple[np.ndarray, np.ndarray]:
    wires = material.wires
    if wires is None:
        empty = np.zeros((0,), dtype=np.int64)
        return empty, empty.copy()
    global_ids = np.asarray(topology.vertex_global_ids[: topology.vertex_count])
    lookup = {int(value): index for index, value in enumerate(global_ids)}
    unknown = [value for value in wires.vertex_global_ids if value not in lookup]
    if unknown:
        raise ValueError(f"Wire vertices {unknown[:4]} are not surface vertices.")
    flat, values = [], []
    for row, (vertex_id, mask) in enumerate(
        zip(wires.vertex_global_ids, wires.fixed_components, strict=True)
    ):
        for component in range(3):
            if mask[component]:
                flat.append(3 * lookup[vertex_id] + component)
                values.append(3 * row + component)
    order = np.argsort(np.asarray(flat), kind="stable")
    return np.asarray(flat, dtype=np.int64)[order], np.asarray(values, dtype=np.int64)[
        order
    ]


def _volume_jacobian(
    surface: PreparedMultiRegionSurface,
    positions: np.ndarray,
    free: np.ndarray,
    finite: tuple[int, ...],
    /,
) -> np.ndarray:
    if not finite:
        return np.zeros((0, free.size))
    slots = jnp.asarray(finite, dtype=jnp.int32)
    base = jnp.asarray(positions.reshape(-1))
    indices = jnp.asarray(free, dtype=jnp.int32)

    def volumes(values: Array, /) -> Array:
        flat = base.at[indices].set(values)
        return surface.region_volumes(flat.reshape((-1, 3)))[slots]

    return np.asarray(jax.jacrev(volumes)(base[indices]), dtype=np.float64)


def _route(plan: FoamEquilibriumPlan, constraint_count: int, /) -> FoamEquilibriumRoute:
    if constraint_count == 0:
        return "unconstrained-trust-region"
    match plan.method:
        case "sqp":
            return "sqp-exact-hessian"
        case "augmented_lagrangian":
            return "augmented-lagrangian-trust-region"
        case _:
            assert_never(plan.method)


def _minimize_foam_impl(
    prepared: PreparedFoamEquilibrium,
    start: Array,
    parameters: FoamEquilibriumParameters,
    /,
) -> tuple[Array, Array, Array, Array]:
    result = minimize(
        prepared.problem(),
        start,
        method=prepared.method(),
        termination=prepared.termination(),
        args=parameters,
    )
    certificate = result.certificate
    multipliers = (
        jnp.zeros((0,), dtype=start.dtype)
        if certificate is None
        else certificate.equality_multipliers
    )
    return result.parameters, multipliers, result.status, result.diagnostics.iterations


_minimize_foam = eqx.filter_jit(_minimize_foam_impl)


def _kkt_matrix(
    prepared: PreparedFoamEquilibrium,
    free: Array,
    multipliers: Array,
    parameters: FoamEquilibriumParameters,
    /,
) -> Array:
    def lagrangian(values: Array, /) -> Array:
        value = prepared._energy(values, parameters)
        if prepared.constraint_count:
            value = value + jnp.vdot(
                multipliers, prepared._constraints(values, parameters)
            )
        return value

    hessian = jax.hessian(lagrangian)(free)
    if not prepared.constraint_count:
        return hessian
    jacobian = jax.jacrev(prepared._constraints)(free, parameters)
    zeros = jnp.zeros((jacobian.shape[0], jacobian.shape[0]), dtype=hessian.dtype)
    return jnp.block([[hessian, jacobian.T], [jacobian, zeros]])


def _kkt_evidence(
    prepared: PreparedFoamEquilibrium,
    free: Array,
    multipliers: Array,
    parameters: FoamEquilibriumParameters,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    primal = free.size
    constraints = prepared.constraint_count
    dimension = primal + constraints
    if dimension > prepared.plan.maximum_dense_dimension:
        missing = jnp.asarray(-1, dtype=jnp.int32)
        nan = jnp.asarray(jnp.nan, dtype=free.dtype)
        status = jnp.asarray(int(FoamKKTStatus.NOT_EVALUATED), dtype=jnp.int32)
        return status, missing, missing, missing, missing, nan, nan
    matrix = _kkt_matrix(prepared, free, multipliers, parameters)
    relative = prepared.plan.kkt_relative_tolerance / float(np.finfo(np.float64).eps)
    spectrum = verify_dense_properties(
        matrix, policy=DensePropertyVerificationPolicy(relative_tolerance=relative)
    )
    eigenvalues = spectrum.eigenvalues
    positive = jnp.sum(eigenvalues > spectrum.tolerance, dtype=jnp.int32)
    negative = jnp.sum(eigenvalues < -spectrum.tolerance, dtype=jnp.int32)
    zero = jnp.asarray(dimension, dtype=jnp.int32) - positive - negative
    rank = spectrum.numerical_rank.astype(jnp.int32)
    condition = jnp.where(rank == dimension, spectrum.condition_estimate, jnp.inf)
    status = jnp.where(
        rank < dimension,
        int(FoamKKTStatus.RANK_DEFICIENT),
        jnp.where(
            (positive != primal) | (negative != constraints),
            int(FoamKKTStatus.INDEFINITE),
            jnp.where(
                condition > prepared.plan.maximum_kkt_condition,
                int(FoamKKTStatus.ILL_CONDITIONED),
                int(FoamKKTStatus.REGULAR),
            ),
        ),
    ).astype(jnp.int32)
    return (
        status,
        rank,
        positive,
        negative,
        zero,
        condition,
        jnp.min(jnp.abs(eigenvalues)),
    )


def _equilibrium_status(
    finite: Array,
    successful: Array,
    volume_residual: Array,
    dependent_residual: Array,
    tolerance: float,
    /,
) -> Array:
    return jnp.where(
        ~finite,
        int(FoamEquilibriumStatus.NONFINITE),
        jnp.where(
            ~successful,
            int(FoamEquilibriumStatus.OPTIMIZER_FAILED),
            jnp.where(
                volume_residual > tolerance,
                int(FoamEquilibriumStatus.VOLUME_RESIDUAL_EXCEEDED),
                jnp.where(
                    dependent_residual > tolerance,
                    int(FoamEquilibriumStatus.DEPENDENT_TARGETS_INCONSISTENT),
                    int(FoamEquilibriumStatus.CONVERGED),
                ),
            ),
        ),
    ).astype(jnp.int32)


def _junction_angles(
    prepared: PreparedFoamEquilibrium, positions: Array, /
) -> tuple[Array, Array]:
    wedges = prepared.surface.junction_wedges(positions)
    mask = prepared.junction_edges[:, None] & wedges.valid
    minimum = jnp.min(jnp.where(mask, wedges.angles, jnp.inf))
    maximum = jnp.max(jnp.where(mask, wedges.angles, -jnp.inf))
    has = jnp.any(mask)
    return jnp.where(has, minimum, jnp.nan), jnp.where(has, maximum, jnp.nan)


def _equilibrium_evidence_impl(
    prepared: PreparedFoamEquilibrium,
    free: Array,
    multipliers: Array,
    optimizer_status: Array,
    iterations: Array,
    parameters: FoamEquilibriumParameters,
    /,
) -> tuple[FoamEquilibriumEvidence, Array, Array, Array]:
    positions = prepared.positions(free, parameters)
    tension = prepared.face_tension(parameters)
    energy = prepared.surface.surface_energy(positions, tension)
    volumes = prepared.surface.region_volumes(positions)
    pressures = prepared._pressures(multipliers)
    gradient = jax.grad(prepared._energy)(free, parameters)
    if prepared.constraint_count:
        jacobian = jax.jacrev(prepared._constraints)(free, parameters)
        gradient = gradient + jacobian.T @ multipliers
    stationarity = jnp.max(jnp.abs(gradient))
    scale = prepared.length_scale**3
    finite_slots = jnp.asarray(prepared.finite_region_slots, dtype=jnp.int32)
    finite_volumes = (
        volumes[finite_slots] if prepared.finite_region_slots else volumes[:0]
    )
    defects = jnp.abs(finite_volumes - parameters.volume_targets) / scale
    constrained = jnp.asarray(prepared.constrained_rows, dtype=jnp.int32)
    dependent = jnp.asarray(prepared.dependent_rows, dtype=jnp.int32)
    volume_residual = jnp.max(defects[constrained], initial=0.0)
    dependent_residual = jnp.max(defects[dependent], initial=0.0)
    gauge = multipliers[len(prepared.constrained_rows) :]
    gauge_norm = jnp.linalg.norm(gauge) if prepared.gauge_count else jnp.asarray(0.0)
    virial = (
        (3.0 * jnp.sum(pressures * volumes) - 2.0 * energy) / (2.0 * energy)
        if prepared.rigid_motion_gauge
        else jnp.asarray(jnp.nan)
    )
    kkt_status, rank, positive, negative, zero, condition, smallest = _kkt_evidence(
        prepared, free, multipliers, parameters
    )
    finite = (
        jnp.all(jnp.isfinite(positions))
        & jnp.isfinite(energy)
        & jnp.all(jnp.isfinite(multipliers))
    )
    successful = optimizer_status == int(OptimizationStatus.SUCCESS)
    status = _equilibrium_status(
        finite,
        successful,
        volume_residual,
        dependent_residual,
        prepared.plan.volume_tolerance,
    )
    tensions = eqx.tree_at(
        lambda matrix: matrix.values,
        prepared.material.tensions,
        parameters.tension_values,
    )
    junction_minimum, junction_maximum = _junction_angles(prepared, positions)
    second_order = kkt_status == int(FoamKKTStatus.REGULAR)
    evidence = FoamEquilibriumEvidence(
        status=status,
        optimizer_status=optimizer_status.astype(jnp.int32),
        optimizer_successful=successful,
        iterations=iterations.astype(jnp.int32),
        stationarity_residual=stationarity,
        volume_residual=volume_residual,
        dependent_volume_residual=dependent_residual,
        gauge_multiplier_norm=gauge_norm,
        virial_residual=virial,
        kkt_status=kkt_status,
        kkt_rank=rank,
        kkt_positive=positive,
        kkt_negative=negative,
        kkt_zero=zero,
        kkt_condition=condition,
        kkt_minimum_absolute_eigenvalue=smallest,
        second_order_sufficient=second_order,
        derivative_available=second_order
        & (status == int(FoamEquilibriumStatus.CONVERGED)),
        tension_admissible=tensions.admissibility().admissible,
        junction_minimum_angle=junction_minimum,
        junction_maximum_angle=junction_maximum,
        finite=finite,
        route=prepared.route,
        primal_dimension=free.size,
        constraint_dimension=prepared.constraint_count,
        constrained_region_ids=prepared.constrained_region_ids,
        pressure_reference_region_ids=prepared.pressure_reference_region_ids,
        rigid_motion_gauge=prepared.rigid_motion_gauge,
        tangential_gauge_dimension=prepared.tangential_gauge_dimension,
        prepared_id=prepared.prepared_id,
    )
    return evidence, pressures, volumes, energy


_equilibrium_evidence = eqx.filter_jit(_equilibrium_evidence_impl)


__all__ = [
    "FoamConstraintBasisEvidence",
    "FoamConstraintBasisPreparationError",
    "FoamConstraintBasisStatus",
    "FoamEquilibriumParameters",
    "FoamImplicitEquilibrium",
    "FoamVolumeConstraintBasis",
    "PreparedFoamEquilibrium",
]
