#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Overlapping Dirichlet coupling of a point cloud and a finite-volume grid.

One original elliptic problem is discretized twice on two overlapping
subdomains: a point-cloud collocation owner (typically near the true boundary)
and a cell-centered finite-volume owner (typically in the interior). Each owner
closes its artificial boundary with Dirichlet data equal to the other owner's
value reconstruction at the declared artificial sites, which is the composite
(overlapping Schwarz) discretization of the original problem:

* the point-cloud owner's artificial rows are ordinary identity rows
  ``u_i - 0`` of its native equations; the law adds ``-(T_c v)_i``, where
  ``T_c`` are native value stencils from the finite-volume cell centers to the
  artificial nodes;
* the finite-volume owner's artificial faces are native Dirichlet faces with
  zero data; the law adds ``-V D(0; alpha T_p u)``, where ``T_p`` are native
  value stencils from the cloud to the face centers. The native diffusion
  action ``D`` is affine in its boundary data, so this is exactly the change of
  the owner's cell balances under the transferred data.

Sites come from the owners' own geometry (cloud coordinates, grid cell and
face centers), never from coincident shapes. Every transfer reports its row
admission, conditioning, and constant/linear reproduction; the identity rows
and zero data the law relies on are measured at preparation. The law's
certificate gates the artificial-node transfer relation and reports the
discretization mismatch of the two owners on the declared overlap cells.

:func:`prepare_overlap_schwarz` prepares one native subspace-correction term
per subdomain whose local solver is the sparse factorization of exactly that
subdomain's diagonal block of the coupled operator; additive (parallel) and
multiplicative (alternating) Schwarz are the native linalg builders over
those terms.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import canonical_identifier, positive_finite_float, positive_integer
from ...discretization import (
    FiniteVolumeDiscretization,
    PreparedPointCloudDiscretization,
)
from ...discretization.meshfree import (
    LocalStencilPolicy,
    MeshfreeFunctional,
    MeshfreeNeighborhoodPlan,
    MeshfreeOperator,
    prepare_local_stencils,
)
from ...linalg import (
    AbstractLinearOperator,
    AbstractSparseLinearOperator,
    ArraySpace,
    BlockSpace,
    DualSpace,
    FunctionLinearOperator,
    MaterializationPolicy,
    SparseFactorizationPolicy,
    SparseFactorizationPreconditionerBuilder,
    SubspaceCorrectionTerm,
)
from ...sparse import compile_sparse_jacobian, EdgeRelation, verify_sparse_derivative
from ...typing import Dim, Int32, parse
from ._components import AbstractSpatialComponent
from ._contributions import (
    Contribution,
    ContributionEndpoint,
    LawImposition,
    LinearContribution,
)
from ._finite_volume_components import FiniteVolumeComponent
from ._interfaces import InterfaceBinding, InterfaceOwner
from ._laws import (
    AbstractCouplingLaw,
    AbstractLawCertificate,
    FieldStates,
    InterfaceDefectReport,
    PreparedLaw,
)
from ._meshfree_components import MeshfreeComponent
from ._prepared_problem import PreparedCoupledProblem


OverlapFaceSide: TypeAlias = Literal["lower", "upper"]
OverlapTransferRoute: TypeAlias = Literal[
    "nodes-to-faces", "cells-to-nodes", "nodes-to-overlap-cells"
]


class _ArtificialNodeDim(Dim):
    """Artificial point-cloud rows closed by finite-volume data."""


class _OverlapCellDim(Dim):
    """Finite-volume cells inside the declared overlap region."""


# --- Transfers -----------------------------------------------------------------------


@final
class OverlapTransferPolicy(StrictModule, NonTrainableState):
    """Native value stencils of the two overlap transfers.

    ``point_stencil``/``point_neighbors`` fit cloud values at finite-volume face
    and overlap-cell centers; ``cell_stencil``/``cell_neighbors`` fit cell
    averages, treated as point values at the cell centers, at the artificial
    cloud nodes. ``reproduction_tolerance`` bounds the measured constant and
    linear reproduction defects every transfer must meet.
    """

    point_stencil: LocalStencilPolicy
    cell_stencil: LocalStencilPolicy
    point_neighbors: int = eqx.field(static=True)
    cell_neighbors: int = eqx.field(static=True)
    reproduction_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        point_stencil: LocalStencilPolicy,
        cell_stencil: LocalStencilPolicy,
        point_neighbors: int,
        cell_neighbors: int,
        reproduction_tolerance: float = 1.0e-9,
    ) -> None:
        for policy, name in (
            (point_stencil, "point_stencil"),
            (cell_stencil, "cell_stencil"),
        ):
            if not isinstance(policy, LocalStencilPolicy):
                raise TypeError(f"{name} must be a LocalStencilPolicy.")
            if policy.acceptance != "refuse":
                raise ValueError(
                    f"{name} must refuse failed rows; a masked transfer row would "
                    "silently drop artificial boundary data."
                )
            if policy.polynomial_degree < 1:
                raise ValueError(f"{name} must reproduce at least linear fields.")
        self.point_stencil = point_stencil
        self.cell_stencil = cell_stencil
        self.point_neighbors = positive_integer(point_neighbors, "point_neighbors")
        self.cell_neighbors = positive_integer(cell_neighbors, "cell_neighbors")
        self.reproduction_tolerance = positive_finite_float(
            reproduction_tolerance, "reproduction_tolerance"
        )


@final
class OverlapTransferEvidence(StrictModule, NonTrainableState):
    """Admission and reproduction evidence of one overlap value transfer.

    ``constant_defect`` is the measured ``max |T 1 - 1|`` and
    ``linear_defect`` the measured ``max |T x_k - x_k|`` over the coordinates;
    ``maximum_condition`` and ``maximum_amplification`` are the native stencil
    report of the admitted rows, of which none were refused.
    """

    route: OverlapTransferRoute = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)
    targets: int = eqx.field(static=True)
    refused_rows: int = eqx.field(static=True)
    maximum_condition: float = eqx.field(static=True)
    maximum_amplification: float = eqx.field(static=True)
    constant_defect: float = eqx.field(static=True)
    linear_defect: float = eqx.field(static=True)

    def __init__(
        self,
        route: OverlapTransferRoute,
        transfer_id: str,
        /,
        *,
        targets: int,
        refused_rows: int,
        maximum_condition: float,
        maximum_amplification: float,
        constant_defect: float,
        linear_defect: float,
    ) -> None:
        self.route = parse(route, OverlapTransferRoute, "route")
        self.transfer_id = canonical_identifier(transfer_id, "transfer_id")
        self.targets = int(targets)
        self.refused_rows = int(refused_rows)
        self.maximum_condition = float(maximum_condition)
        self.maximum_amplification = float(maximum_amplification)
        self.constant_defect = float(constant_defect)
        self.linear_defect = float(linear_defect)


def _value_transfer(
    route: OverlapTransferRoute,
    sources: np.ndarray,
    targets: np.ndarray,
    neighbors: int,
    stencil: LocalStencilPolicy,
    tolerance: float,
    /,
) -> tuple[MeshfreeOperator, OverlapTransferEvidence]:
    """Native value stencils ``sources -> targets`` with measured reproduction."""
    if neighbors > sources.shape[0]:
        raise ValueError(
            f"The {route} transfer requests {neighbors} neighbors from "
            f"{sources.shape[0]} sources."
        )
    neighborhood = MeshfreeNeighborhoodPlan(sources, neighbors, targets=targets).prepare()
    zero = (0,) * sources.shape[1]
    stencils = prepare_local_stencils(
        neighborhood,
        sources,
        targets,
        (MeshfreeFunctional((zero,), (1.0,), name="value"),),
        stencil,
    )
    operator = MeshfreeOperator(stencils, 0)
    report = stencils.report
    if report.refused_rows:
        raise ValueError(
            f"The {route} transfer refused {report.refused_rows} rows; the overlap "
            "sites lack unisolvent support."
        )
    constant = float(
        np.max(
            np.abs(
                np.asarray(operator.apply(np.ones((sources.shape[0],), dtype=np.float64)))
                - 1.0
            )
        )
    )
    linear = max(
        float(
            np.max(
                np.abs(np.asarray(operator.apply(sources[:, axis])) - targets[:, axis])
            )
        )
        for axis in range(sources.shape[1])
    )
    if constant > tolerance or linear > tolerance:
        raise ValueError(
            f"The {route} transfer moves constants by {constant:.3e} and linear "
            f"fields by {linear:.3e}; it is no consistent Dirichlet data transfer."
        )
    evidence = OverlapTransferEvidence(
        route,
        operator.operator_id,
        targets=targets.shape[0],
        refused_rows=report.refused_rows,
        maximum_condition=report.maximum_condition_number,
        maximum_amplification=report.maximum_amplification,
        constant_defect=constant,
        linear_defect=linear,
    )
    return operator, evidence


def _unique_rows(value: ArrayLike, size: int, name: str, /) -> np.ndarray:
    rows = np.asarray(value)
    if (
        rows.ndim != 1
        or rows.size == 0
        or not np.issubdtype(rows.dtype, np.integer)
        or np.any(rows < 0)
        or np.any(rows >= size)
        or np.unique(rows).size != rows.size
    ):
        raise ValueError(f"{name} must be unique in-range integer indices.")
    return rows.astype(np.int32)


def _side_index(side: OverlapFaceSide, /) -> int:
    """Native boundary position of one face side (lower 0, upper 1)."""
    match side:
        case "lower":
            return 0
        case "upper":
            return 1
        case _:
            assert_never(side)


def _faces(
    discretization: FiniteVolumeDiscretization,
    faces: Sequence[tuple[str, OverlapFaceSide]],
    /,
) -> tuple[tuple[str, OverlapFaceSide], ...]:
    """Canonically ordered artificial faces (axis order, lower before upper)."""
    grid = discretization.grid
    declared: list[tuple[int, int, str, OverlapFaceSide]] = []
    for entry in faces:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise TypeError("Artificial faces are (axis name, side) pairs.")
        axis, side = entry
        if axis not in grid.axis_names:
            raise ValueError(f"Artificial face names unknown axis {axis!r}.")
        side_ = parse(side, OverlapFaceSide, "side")
        index = grid.axis_names.index(axis)
        if grid.structured_axes[index].periodic:
            raise ValueError(f"Periodic axis {axis!r} has no boundary faces.")
        declared.append((index, _side_index(side_), axis, side_))
    if not declared or len(set(declared)) != len(declared):
        raise ValueError("Declare each artificial face set once.")
    return tuple((axis, side) for _, _, axis, side in sorted(declared))


def _face_centers(
    discretization: FiniteVolumeDiscretization,
    faces: tuple[tuple[str, OverlapFaceSide], ...],
    /,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Host face centers per declared face set, on its transverse cell layout."""
    grid = discretization.grid
    centers = [
        np.asarray(axis.interval_centers, dtype=np.float64)
        for axis in grid.structured_axes
    ]
    blocks: list[np.ndarray] = []
    shapes: list[tuple[int, ...]] = []
    for name, side in faces:
        axis = grid.axis_names.index(name)
        bounds = np.asarray(grid.structured_axes[axis].bounds, dtype=np.float64)
        others = [values for index, values in enumerate(centers) if index != axis]
        mesh = np.meshgrid(*others, indexing="ij")
        transverse = mesh[0].shape
        columns = [part.reshape(-1) for part in mesh]
        constant = np.full(columns[0].shape, bounds[_side_index(side)])
        columns.insert(axis, constant)
        blocks.append(np.stack(columns, axis=1))
        shapes.append(transverse)
    return np.concatenate(blocks, axis=0), tuple(shapes)


@final
class PreparedOverlapTransfers(StrictModule, NonTrainableState):
    """Prepared value transfers between the two owners of an overlap law.

    ``rows`` are the artificial cloud rows, ``faces`` the artificial
    finite-volume face sets (``transverse_shapes`` their transverse cell
    layouts, concatenated in that order), and ``overlap_cells`` the flat cell
    indices of the declared overlap region on which the owners' mismatch is
    measured. ``overlap_gap`` is the smallest distance from an artificial face
    center to an artificial cloud node: the two artificial boundaries are
    separated, so the overlap is genuine.
    """

    __strict_contract__ = True
    meshfree_owner_id: str = eqx.field(static=True)
    finite_volume_owner_id: str = eqx.field(static=True)
    rows: Int32[_ArtificialNodeDim]
    faces: tuple[tuple[str, OverlapFaceSide], ...] = eqx.field(static=True)
    transverse_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    overlap_cells: Int32[_OverlapCellDim]
    nodes_to_faces: MeshfreeOperator
    cells_to_nodes: MeshfreeOperator
    nodes_to_overlap: MeshfreeOperator
    evidence: tuple[OverlapTransferEvidence, ...]
    overlap_gap: float = eqx.field(static=True)
    transfers_id: str = eqx.field(static=True)

    def __init__(
        self,
        meshfree: PreparedPointCloudDiscretization,
        rows: ArrayLike,
        finite_volume: FiniteVolumeDiscretization,
        faces: Sequence[tuple[str, OverlapFaceSide]],
        overlap_cells: ArrayLike,
        /,
        *,
        policy: OverlapTransferPolicy,
    ) -> None:
        if not isinstance(meshfree, PreparedPointCloudDiscretization):
            raise TypeError("meshfree must be a PreparedPointCloudDiscretization.")
        if not isinstance(finite_volume, FiniteVolumeDiscretization):
            raise TypeError("finite_volume must be a FiniteVolumeDiscretization.")
        if not isinstance(policy, OverlapTransferPolicy):
            raise TypeError("policy must be an OverlapTransferPolicy.")
        points = np.asarray(meshfree.points, dtype=np.float64)
        cell_centers = np.asarray(finite_volume.cell_centers, dtype=np.float64)
        dimension = points.shape[1]
        if cell_centers.shape[-1] != dimension:
            raise ValueError("The cloud and the grid must share one spatial dimension.")
        cells = cell_centers.reshape((-1, dimension))
        rows_ = _unique_rows(rows, points.shape[0], "rows")
        overlap_ = _unique_rows(overlap_cells, cells.shape[0], "overlap_cells")
        faces_ = _faces(finite_volume, faces)
        face_centers, shapes = _face_centers(finite_volume, faces_)
        nodes = points[rows_]
        # Cell values are interpolated, never extrapolated, at artificial nodes.
        lower = cells.min(axis=0)
        upper = cells.max(axis=0)
        if np.any(nodes <= lower) or np.any(nodes >= upper):
            raise ValueError(
                "Artificial cloud nodes must lie strictly inside the hull of the "
                "finite-volume cell centers."
            )
        nearest = MeshfreeNeighborhoodPlan(nodes, 1, targets=face_centers).prepare()
        gap = float(np.min(np.asarray(nearest.distances)))
        if gap <= 0.0:
            raise ValueError(
                "An artificial face center coincides with an artificial cloud node; "
                "the two artificial boundaries must be separated by the overlap."
            )
        tolerance = policy.reproduction_tolerance
        nodes_to_faces, faces_evidence = _value_transfer(
            "nodes-to-faces",
            points,
            face_centers,
            policy.point_neighbors,
            policy.point_stencil,
            tolerance,
        )
        cells_to_nodes, nodes_evidence = _value_transfer(
            "cells-to-nodes",
            cells,
            nodes,
            policy.cell_neighbors,
            policy.cell_stencil,
            tolerance,
        )
        nodes_to_overlap, overlap_evidence = _value_transfer(
            "nodes-to-overlap-cells",
            points,
            cells[overlap_],
            policy.point_neighbors,
            policy.point_stencil,
            tolerance,
        )
        self.meshfree_owner_id = meshfree.prepared_id
        self.finite_volume_owner_id = finite_volume.prepared_id
        self.rows = jnp.asarray(rows_)
        self.faces = faces_
        self.transverse_shapes = shapes
        self.overlap_cells = jnp.asarray(overlap_)
        self.nodes_to_faces = nodes_to_faces
        self.cells_to_nodes = cells_to_nodes
        self.nodes_to_overlap = nodes_to_overlap
        self.evidence = (faces_evidence, nodes_evidence, overlap_evidence)
        self.overlap_gap = gap
        self.transfers_id = canonical_fingerprint(
            {
                "kind": "overlap-transfers",
                "meshfree": meshfree.prepared_id,
                "finite_volume": finite_volume.prepared_id,
                "rows": array_tree_fingerprint(rows_),
                "faces": [list(face) for face in faces_],
                "overlap_cells": array_tree_fingerprint(overlap_),
                "transfers": [item.transfer_id for item in self.evidence],
            }
        )


# --- Law -----------------------------------------------------------------------------


@final
class OverlapDirichletEvidence(StrictModule, NonTrainableState):
    """Transfers, measured owner preconditions, and overlap extent of one law.

    ``node_identity_defect`` is the measured relative defect of the declared
    cloud rows being identity rows of the owner's native equations and
    ``node_load_defect`` the magnitude of the owner's own data on them (both
    must vanish: the law supplies the complete artificial data).
    """

    transfers: tuple[OverlapTransferEvidence, ...]
    artificial_nodes: int = eqx.field(static=True)
    artificial_faces: int = eqx.field(static=True)
    overlap_cells: int = eqx.field(static=True)
    overlap_gap: float = eqx.field(static=True)
    node_identity_defect: float = eqx.field(static=True)
    node_load_defect: float = eqx.field(static=True)

    def __init__(
        self,
        transfers: PreparedOverlapTransfers,
        /,
        *,
        artificial_faces: int,
        node_identity_defect: float,
        node_load_defect: float,
    ) -> None:
        self.transfers = transfers.evidence
        self.artificial_nodes = transfers.rows.shape[0]
        self.artificial_faces = int(artificial_faces)
        self.overlap_cells = transfers.overlap_cells.shape[0]
        self.overlap_gap = transfers.overlap_gap
        self.node_identity_defect = float(node_identity_defect)
        self.node_load_defect = float(node_load_defect)


@final
class _OverlapCertificate(AbstractLawCertificate):
    """Artificial-node transfer relation and the owners' overlap mismatch."""

    law_id: str = eqx.field(static=True)
    meshfree: ContributionEndpoint
    finite_volume: ContributionEndpoint
    transfers: PreparedOverlapTransfers
    overlap_volumes: Array

    def defects(
        self,
        fields: FieldStates,
        law_state: tuple[Array, ...],
        args: Mapping[str, object],
        /,
    ) -> InterfaceDefectReport:
        del law_state, args
        transfers = self.transfers
        nodal = fields[(self.meshfree.owner, self.meshfree.block)]
        cells = fields[(self.finite_volume.owner, self.finite_volume.block)].reshape(
            (-1,)
        )
        imposed = nodal[transfers.rows]
        transferred = transfers.cells_to_nodes.apply(cells)
        relation = jnp.max(jnp.abs(imposed - transferred))
        relation_scale = jnp.max(jnp.abs(imposed)) + jnp.max(jnp.abs(transferred))
        cloud = transfers.nodes_to_overlap.apply(nodal)
        grid = cells[transfers.overlap_cells]
        difference = cloud - grid
        mismatch = jnp.max(jnp.abs(difference))
        mismatch_scale = jnp.max(jnp.abs(cloud)) + jnp.max(jnp.abs(grid))
        volumes = self.overlap_volumes
        l2 = jnp.sqrt(jnp.sum(volumes * difference**2))
        l2_scale = jnp.sqrt(jnp.sum(volumes * cloud**2)) + jnp.sqrt(
            jnp.sum(volumes * grid**2)
        )
        return InterfaceDefectReport(
            self.law_id,
            ("artificial-node-transfer", "overlap-mismatch-max", "overlap-mismatch-l2"),
            (True, False, False),
            jnp.stack((relation, mismatch, l2)),
            jnp.stack((relation_scale, mismatch_scale, l2_scale)),
        )


def _probes(space: ArraySpace, /) -> tuple[Array, Array]:
    """Deterministic sign-varying non-polynomial coordinates of one field space."""
    index = np.arange(space.size, dtype=np.float64)
    return (
        jnp.asarray(np.cos(1.3 * index + 0.2), dtype=space.dtype).reshape(space.shape),
        jnp.asarray(
            np.sin(0.7 * index + 1.0) * (1.0 + index / space.size), dtype=space.dtype
        ).reshape(space.shape),
    )


def _identity_rows(
    component: AbstractSpatialComponent, field: str, rows: Array, /
) -> tuple[float, float]:
    """Measured identity-row and zero-data defects of the artificial cloud rows."""
    record = component.field(field)
    if record.constraint is not None or len(component.state_blocks) != 1:
        raise ValueError(
            "Artificial cloud rows must be free full rows of a scalar unconstrained "
            "owner; constrained rows are eliminated and cannot carry transferred data."
        )
    zero = record.full_space.zeros()
    offset = component.residual((zero,), None)[0][rows]
    load = float(jnp.max(jnp.abs(offset)))
    defect = 0.0
    for probe in _probes(record.full_space):
        image = component.residual((probe,), None)[0][rows] - offset
        reference = probe[rows]
        defect = max(
            defect,
            float(
                jnp.max(jnp.abs(image - reference))
                / jnp.maximum(jnp.max(jnp.abs(reference)), 1.0e-300)
            ),
        )
    return defect, load


def _require_dirichlet_faces(
    component: FiniteVolumeComponent,
    faces: tuple[tuple[str, OverlapFaceSide], ...],
    /,
) -> tuple[float, ...]:
    """Each artificial face is a native zero-data Dirichlet face; its ``alpha``."""
    grid = component.discretization.grid
    alphas: list[float] = []
    for name, side in faces:
        axis = grid.axis_names.index(name)
        position = _side_index(side)
        condition = component.diffusion.plan.boundaries[axis][position]
        match condition.kind:
            case "dirichlet":
                pass
            case "neumann" | "robin" | "periodic":
                raise ValueError(
                    f"Artificial face {name}:{side} carries a native {condition.kind} "
                    "condition; declare it Dirichlet with zero data so the law "
                    "supplies its complete value."
                )
            case _:
                assert_never(condition.kind)
        if np.any(np.asarray(component.boundary_targets[axis][position]) != 0.0):
            raise ValueError(
                f"Artificial face {name}:{side} carries nonzero native Dirichlet data; "
                "the law supplies the complete artificial value."
            )
        alphas.append(condition.alpha)
    return tuple(alphas)


def _face_facets(
    component: FiniteVolumeComponent,
    faces: tuple[tuple[str, OverlapFaceSide], ...],
    /,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Exterior facets of the artificial faces and their owner cells."""
    discretization = component.discretization
    exterior = discretization.integration_domain("exterior_facet")
    local = np.asarray(exterior.owner_local_entities, dtype=np.int64)
    entities = np.asarray(exterior.entity_indices, dtype=np.int64)
    owners = np.asarray(exterior.owner_cells, dtype=np.int64)
    selected = np.zeros(local.shape, dtype=np.bool_)
    for name, side in faces:
        axis = discretization.grid.axis_names.index(name)
        selected |= local == 2 * axis + _side_index(side)
    return exterior.entity_set_id, entities[selected], owners[selected]


@final
class _OverlapActions(StrictModule):
    """Row actions of the two transferred Dirichlet data and their transposes."""

    transfers: PreparedOverlapTransfers
    component: FiniteVolumeComponent
    alphas: tuple[float, ...] = eqx.field(static=True)
    meshfree_space: ArraySpace
    cell_space: ArraySpace

    def node_rows(self, cells: Array, /) -> Array:
        """Cloud rows ``-(T_c v)`` on the artificial nodes, zero elsewhere."""
        data = self.transfers.cells_to_nodes.apply(cells.reshape((-1,)))
        rows = jnp.zeros(self.meshfree_space.shape, dtype=self.meshfree_space.dtype)
        return rows.at[self.transfers.rows].set(-data)

    def node_rows_transpose(self, rows: Array, /) -> Array:
        cotangent = -rows[self.transfers.rows]
        return self.transfers.cells_to_nodes.transpose_apply(cotangent).reshape(
            self.cell_space.shape
        )

    def face_rows(self, nodal: Array, /) -> Array:
        """Cell rows ``-V D(0; alpha T_p u)`` of the transferred face data."""
        grid = self.component.discretization.grid
        values = self.transfers.nodes_to_faces.apply(nodal)
        data: dict[str, list[Array]] = {}
        offset = 0
        for (name, side), shape, alpha in zip(
            self.transfers.faces,
            self.transfers.transverse_shapes,
            self.alphas,
            strict=True,
        ):
            size = int(np.prod(shape))
            block = alpha * values[offset : offset + size].reshape(shape)
            offset += size
            pair = data.setdefault(
                name,
                [
                    jnp.zeros(shape, dtype=block.dtype),
                    jnp.zeros(shape, dtype=block.dtype),
                ],
            )
            pair[_side_index(side)] = block
        cells = jnp.zeros(grid.shape, dtype=self.cell_space.dtype)
        action = self.component.diffusion.apply(
            cells,
            boundary_values={name: (pair[0], pair[1]) for name, pair in data.items()},
        )
        return -self.component.volumes * action

    def face_rows_transpose(self, rows: Array, /) -> Array:
        return jax.linear_transpose(self.face_rows, self.meshfree_space.zeros())(rows)[0]


def _overlap_contributions(
    law_id: str,
    meshfree: ContributionEndpoint,
    finite_volume: ContributionEndpoint,
    actions: _OverlapActions,
    /,
) -> tuple[Contribution, ...]:
    imposition = canonical_fingerprint({"kind": "overlap-dirichlet", "law": law_id})
    return (
        LinearContribution(
            meshfree,
            finite_volume,
            FunctionLinearOperator(
                actions.node_rows,
                source=actions.cell_space,
                target=DualSpace(actions.meshfree_space),
                transpose_action=actions.node_rows_transpose,
                operator_id=f"{law_id}:cells-to-artificial-nodes",
            ),
            law_id=law_id,
            imposition_id=imposition,
        ),
        LinearContribution(
            finite_volume,
            meshfree,
            FunctionLinearOperator(
                actions.face_rows,
                source=actions.meshfree_space,
                target=DualSpace(actions.cell_space),
                transpose_action=actions.face_rows_transpose,
                operator_id=f"{law_id}:nodes-to-artificial-faces",
            ),
            law_id=law_id,
            imposition_id=imposition,
        ),
    )


@final
class OverlapDirichletLaw(AbstractCouplingLaw, NonTrainableState):
    """Overlapping Dirichlet (composite Schwarz) coupling of a cloud and a grid.

    ``meshfree`` names the point-cloud component field and ``finite_volume``
    the finite-volume component field. ``transfers`` are the overlap
    transfers prepared on exactly those owners: the declared artificial cloud
    rows must be identity rows of the cloud owner's native equations with zero
    data, and the declared artificial faces native zero-data Dirichlet faces.
    The cloud rows then read ``u_i - (T_c v)_i`` and the cell balances carry
    the face data ``T_p u``, so the coupled solution is the composite
    discretization of the original problem; the law owns no unknowns.
    ``tolerance`` bounds the measured identity-row and zero-data defects.
    """

    law_id: str = eqx.field(static=True)
    meshfree: ContributionEndpoint
    finite_volume: ContributionEndpoint
    transfers: PreparedOverlapTransfers
    tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        law_id: str,
        meshfree: ContributionEndpoint,
        finite_volume: ContributionEndpoint,
        transfers: PreparedOverlapTransfers,
        /,
        *,
        tolerance: float = 1.0e-10,
    ) -> None:
        for endpoint, name in ((meshfree, "meshfree"), (finite_volume, "finite_volume")):
            if not isinstance(endpoint, ContributionEndpoint):
                raise TypeError(f"{name} must be a ContributionEndpoint.")
            if endpoint.space != "full":
                raise ValueError(f"The {name} endpoint must name a component field.")
        if meshfree.owner == finite_volume.owner:
            raise ValueError("An overlap law couples two different components.")
        if not isinstance(transfers, PreparedOverlapTransfers):
            raise TypeError("transfers must be PreparedOverlapTransfers.")
        self.law_id = canonical_identifier(law_id, "law_id")
        self.meshfree = meshfree
        self.finite_volume = finite_volume
        self.transfers = transfers
        self.tolerance = positive_finite_float(tolerance, "tolerance")

    @property
    def bindings(self) -> tuple[InterfaceBinding, ...]:
        # The overlap sites are the owners' own geometry carried by the transfers.
        return ()

    def _owners(
        self, components: Mapping[str, AbstractSpatialComponent], /
    ) -> tuple[MeshfreeComponent, FiniteVolumeComponent]:
        for endpoint in (self.meshfree, self.finite_volume):
            if endpoint.owner not in components:
                raise ValueError(
                    f"Law endpoint names unknown component {endpoint.owner!r}."
                )
        cloud = components[self.meshfree.owner]
        grid = components[self.finite_volume.owner]
        if not isinstance(cloud, MeshfreeComponent) or not isinstance(
            cloud.owner, PreparedPointCloudDiscretization
        ):
            raise TypeError(
                "The meshfree endpoint must be a point-cloud MeshfreeComponent."
            )
        if not isinstance(grid, FiniteVolumeComponent):
            raise TypeError("The finite-volume endpoint must be a FiniteVolumeComponent.")
        if cloud.owner.prepared_id != self.transfers.meshfree_owner_id:
            raise ValueError("The transfers were prepared on another point-cloud owner.")
        if grid.discretization.prepared_id != self.transfers.finite_volume_owner_id:
            raise ValueError(
                "The transfers were prepared on another finite-volume owner."
            )
        return cloud, grid

    def prepare(
        self,
        components: Mapping[str, AbstractSpatialComponent],
        interface_owners: tuple[InterfaceOwner, ...],
        /,
    ) -> PreparedLaw:
        del interface_owners  # The overlap acts on no interface binding.
        cloud, grid = self._owners(components)
        owner = cloud.owner
        if not isinstance(owner, PreparedPointCloudDiscretization):
            raise TypeError("Overlap requires a prepared point-cloud owner.")
        transfers = self.transfers
        identity, load = _identity_rows(cloud, self.meshfree.block, transfers.rows)
        if identity > self.tolerance or load > self.tolerance:
            raise ValueError(
                f"The artificial cloud rows are not zero-data identity rows of the "
                f"owner (identity defect {identity:.3e}, data {load:.3e}); declare them "
                "Dirichlet rows with zero data."
            )
        alphas = _require_dirichlet_faces(grid, transfers.faces)
        cloud_field = cloud.field(self.meshfree.block)
        grid_field = grid.field(self.finite_volume.block)
        meshfree_space = cloud_field.full_space
        cell_space = grid_field.full_space
        if transfers.nodes_to_faces.operator.source.size != meshfree_space.size:
            raise ValueError("The transfers do not act on the cloud field coordinates.")
        actions = _OverlapActions(transfers, grid, alphas, meshfree_space, cell_space)
        entity_set, facets, owner_cells = _face_facets(grid, transfers.faces)
        imposition = canonical_fingerprint(
            {"kind": "overlap-dirichlet", "law": self.law_id}
        )
        impositions = (
            LawImposition(
                cloud.name,
                self.meshfree.block,
                field_space_id=cloud.field_space_id(self.meshfree.block),
                entity_set_id=owner.support.support_id,
                facets=np.asarray(transfers.rows),
                rows=np.asarray(transfers.rows),
                imposition_id=imposition,
            ),
            LawImposition(
                grid.name,
                self.finite_volume.block,
                field_space_id=grid.field_space_id(self.finite_volume.block),
                entity_set_id=entity_set,
                facets=facets,
                rows=owner_cells,
                imposition_id=imposition,
            ),
        )
        volumes = grid.volumes.reshape((-1,))[transfers.overlap_cells]
        return PreparedLaw(
            self.law_id,
            binding_id=None,
            state_blocks=(),
            row_blocks=(),
            contributions=_overlap_contributions(
                self.law_id, self.meshfree, self.finite_volume, actions
            ),
            impositions=impositions,
            certificate=_OverlapCertificate(
                self.law_id, self.meshfree, self.finite_volume, transfers, volumes
            ),
            evidence=OverlapDirichletEvidence(
                transfers,
                artificial_faces=facets.size,
                node_identity_defect=identity,
                node_load_defect=load,
            ),
        )


# --- Native subspace correction over the two subdomains ------------------------------------


@final
class OverlapSubdomainEvidence(StrictModule, NonTrainableState):
    """Exact local block and factorization evidence of one Schwarz subdomain.

    ``block_relative_error`` is the measured probe error of the assembled
    sparse block against the coupled operator's own block action
    (``block_verified`` when it meets the native verification tolerance);
    ``factor_status`` the native sparse factorization status and
    ``minimum_pivot`` its smallest pivot magnitude.
    """

    component: str = eqx.field(static=True)
    field: str = eqx.field(static=True)
    size: int = eqx.field(static=True)
    pattern_entries: int = eqx.field(static=True)
    colors: int = eqx.field(static=True)
    block_relative_error: float = eqx.field(static=True)
    block_verified: bool = eqx.field(static=True)
    factor_entries: int = eqx.field(static=True)
    factor_status: int = eqx.field(static=True)
    minimum_pivot: float = eqx.field(static=True)

    def __init__(
        self,
        component: str,
        field: str,
        /,
        *,
        size: int,
        pattern_entries: int,
        colors: int,
        block_relative_error: float,
        block_verified: bool,
        factor_entries: int,
        factor_status: int,
        minimum_pivot: float,
    ) -> None:
        self.component = canonical_identifier(component, "component")
        self.field = canonical_identifier(field, "field")
        self.size = int(size)
        self.pattern_entries = int(pattern_entries)
        self.colors = int(colors)
        self.block_relative_error = float(block_relative_error)
        self.block_verified = bool(block_verified)
        self.factor_entries = int(factor_entries)
        self.factor_status = int(factor_status)
        self.minimum_pivot = float(minimum_pivot)


@final
class OverlapSchwarz(StrictModule):
    """One native subspace-correction term per overlap subdomain and its evidence.

    ``terms`` are in (point cloud, finite volume) order; pass them to
    ``AdditiveSubspaceCorrectionBuilder`` (parallel Schwarz) or
    ``MultiplicativeSubspaceCorrectionBuilder`` (alternating Schwarz).
    """

    terms: tuple[SubspaceCorrectionTerm, ...]
    evidence: tuple[OverlapSubdomainEvidence, ...]

    def __init__(
        self,
        terms: tuple[SubspaceCorrectionTerm, ...],
        evidence: tuple[OverlapSubdomainEvidence, ...],
        /,
    ) -> None:
        if len(terms) != len(evidence) or not terms:
            raise ValueError("Every Schwarz term carries its subdomain evidence.")
        self.terms = terms
        self.evidence = evidence


def _block_transfers(
    space: BlockSpace, owner: str, block: str, /
) -> tuple[FunctionLinearOperator, FunctionLinearOperator, ArraySpace]:
    """Exact selection of one solve block and its zero-filled embedding."""
    if owner not in space.names:
        raise ValueError(f"Component {owner!r} owns no solve block.")
    owner_index = space.names.index(owner)
    member = space.spaces[owner_index]
    if not isinstance(member, BlockSpace) or block not in member.names:
        raise ValueError(f"Component {owner!r} has no solve block {block!r}.")
    block_index = member.names.index(block)
    local = member.spaces[block_index]
    if not isinstance(local, ArraySpace):
        raise TypeError("Overlap subdomain blocks are array spaces.")
    owners: list[BlockSpace] = []
    for nested in space.spaces:
        if not isinstance(nested, BlockSpace):
            raise TypeError("Solve owners are block spaces.")
        owners.append(nested)

    def restrict(value: tuple[tuple[Array, ...], ...]) -> Array:
        return value[owner_index][block_index]

    def prolong(value: Array) -> tuple[tuple[Array, ...], ...]:
        return tuple(
            tuple(
                value
                if (index, position) == (owner_index, block_index)
                else entry.zeros()
                for position, entry in enumerate(nested.spaces)
            )
            for index, nested in enumerate(owners)
        )

    restriction = FunctionLinearOperator(
        restrict,
        source=space,
        target=local,
        transpose_action=prolong,
        operator_id=f"{space.space_id}:restrict:{owner}:{block}",
    )
    prolongation = FunctionLinearOperator(
        prolong,
        source=local,
        target=space,
        transpose_action=restrict,
        operator_id=f"{space.space_id}:prolong:{owner}:{block}",
    )
    return restriction, prolongation, local


def _block_pattern(component: AbstractSpatialComponent, /) -> EdgeRelation:
    """Declared sparsity of one subdomain's own coupled diagonal block."""
    if isinstance(component, MeshfreeComponent):
        operator = component.native_operator
        if not isinstance(operator, AbstractSparseLinearOperator):
            raise TypeError(
                "The point-cloud subdomain needs native sparse equations to factor."
            )
        storage = operator.sparse_storage()
        size = storage.shape[0]
        indptr = np.asarray(storage.indptr, dtype=np.int64)
        targets = np.repeat(np.arange(size, dtype=np.int64), np.diff(indptr))
        sources = np.asarray(storage.indices, dtype=np.int64)
        return EdgeRelation(sources, targets, source_size=size, target_size=size)
    if isinstance(component, FiniteVolumeComponent):
        # Conservative cell diffusion couples a cell to its 3^d neighborhood
        # (normal fluxes plus tangential gradients of tensor cross terms).
        shape = component.discretization.grid.shape
        size = int(np.prod(shape))
        index = np.arange(size, dtype=np.int64).reshape(shape)
        offsets = np.stack(
            np.meshgrid(*([np.arange(-1, 2)] * len(shape)), indexing="ij"), axis=-1
        ).reshape((-1, len(shape)))
        coordinates = np.stack(np.unravel_index(index.reshape(-1), shape), axis=1)
        sources_: list[np.ndarray] = []
        targets_: list[np.ndarray] = []
        for offset in offsets:
            neighbor = coordinates + offset
            valid = np.all((neighbor >= 0) & (neighbor < np.asarray(shape)), axis=1)
            targets_.append(index.reshape(-1)[valid])
            sources_.append(np.ravel_multi_index(tuple(neighbor[valid].T), shape))
        return EdgeRelation(
            np.concatenate(sources_),
            np.concatenate(targets_),
            source_size=size,
            target_size=size,
        )
    raise TypeError("Overlap subdomains are point-cloud or finite-volume components.")


def _subdomain_term(
    operator: AbstractLinearOperator,
    space: BlockSpace,
    component: AbstractSpatialComponent,
    endpoint: ContributionEndpoint,
    policy: SparseFactorizationPolicy,
    plan_id: str,
    /,
) -> tuple[SubspaceCorrectionTerm, OverlapSubdomainEvidence]:
    """Verified sparse factorization of one subdomain's coupled diagonal block."""
    record = component.field(endpoint.block)
    if record.constraint is not None:
        raise ValueError("Overlap subdomain fields are unconstrained full fields.")
    restriction, prolongation, local = _block_transfers(
        space, endpoint.owner, record.state_block
    )

    def block(value: Array, arguments: object) -> Array:
        del arguments
        return restriction.mv(operator.mv(prolongation.mv(value)))

    zero = local.zeros()
    plan = compile_sparse_jacobian(
        block,
        zero,
        source=local,
        target=local,
        structure=_block_pattern(component),
        compiler="native",
        plan_id=plan_id,
    )
    verification = verify_sparse_derivative(plan, zero, key=jax.random.key(0))
    verified = bool(verification.passed)
    relative = float(verification.maximum_relative_error)
    if not verified:
        raise ValueError(
            f"The declared pattern of subdomain {endpoint.owner!r} misses entries of "
            f"its coupled block (relative probe error {relative:.3e})."
        )
    solver = SparseFactorizationPreconditionerBuilder(policy).prepare(
        plan.operator(zero), materialization=MaterializationPolicy()
    )
    factorization = solver.factorization
    status = int(factorization.status)
    if status != 0:
        raise ValueError(
            f"The sparse factorization of subdomain {endpoint.owner!r} failed with "
            f"status {status}."
        )
    evidence = OverlapSubdomainEvidence(
        endpoint.owner,
        endpoint.block,
        size=local.size,
        pattern_entries=plan.nnz,
        colors=plan.num_colors,
        block_relative_error=relative,
        block_verified=verified,
        factor_entries=factorization.plan.factor_nnz,
        factor_status=status,
        minimum_pivot=float(factorization.diagnostics.minimum_pivot),
    )
    return SubspaceCorrectionTerm(restriction, prolongation, solver), evidence


def prepare_overlap_schwarz(
    prepared: PreparedCoupledProblem,
    law_id: str,
    /,
    *,
    arguments: Mapping[str, object] | None = None,
    factorization: SparseFactorizationPolicy | None = None,
) -> OverlapSchwarz:
    """Native per-subdomain correction terms of one prepared overlap problem.

    The overlap law ``law_id`` names the two subdomains. For each, the exact
    diagonal block ``R A R^T`` of the coupled linear operator is assembled by
    the native sparse Jacobian with the owner's declared pattern, verified
    against the block action by probes (refused when it disagrees), and
    factored completely by the native sparse factorization (refused unless the
    factorization succeeds). The restriction selects the subdomain's solve
    block and the prolongation embeds it with zeros elsewhere.
    """
    if not isinstance(prepared, PreparedCoupledProblem):
        raise TypeError("prepared must be a PreparedCoupledProblem.")
    law = next((item for item in prepared.laws if item.law_id == law_id), None)
    if law is None or not isinstance(law.evidence, OverlapDirichletEvidence):
        raise ValueError(f"The prepared problem has no overlap law {law_id!r}.")
    certificate = law.certificate
    if not isinstance(certificate, _OverlapCertificate):
        raise TypeError("The overlap law carries a foreign certificate.")
    system, _ = prepared.linear_system(arguments)
    components = {component.name: component for component in prepared.components}
    policy = (
        SparseFactorizationPolicy("lu", ordering="approximate-minimum-degree")
        if factorization is None
        else factorization
    )
    pairs = tuple(
        _subdomain_term(
            system.operator,
            prepared.state_space,
            components[endpoint.owner],
            endpoint,
            policy,
            f"{law_id}:{endpoint.owner}:subdomain-block",
        )
        for endpoint in (certificate.meshfree, certificate.finite_volume)
    )
    return OverlapSchwarz(
        tuple(term for term, _ in pairs), tuple(evidence for _, evidence in pairs)
    )


__all__ = [
    "OverlapDirichletEvidence",
    "OverlapDirichletLaw",
    "OverlapFaceSide",
    "OverlapSchwarz",
    "OverlapSubdomainEvidence",
    "OverlapTransferEvidence",
    "OverlapTransferPolicy",
    "OverlapTransferRoute",
    "PreparedOverlapTransfers",
    "prepare_overlap_schwarz",
]
