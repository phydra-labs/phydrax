#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization._cell_complex import PolygonalConnectivity
from ....discretization._cell_mesh import CellMesh
from ....discretization.fem._adaptivity import (
    dorfler_mark,
    FiniteElementAdaptationMap,
    maximum_mark,
    refine_triangles_local,
)
from ....geometry.surface._contracts import (
    SurfaceInterface,
    SurfaceMetadata,
    SurfaceSelection,
)
from ....geometry.surface._model import SurfaceModel
from ._fast_provider import BEMExecutionEnvelope


BoundaryMarkingStrategy: TypeAlias = Literal["dorfler", "maximum"]


class BoundaryEpochError(ValueError):
    """Raised when prepared boundary state is used with a stale mesh epoch."""


class BoundaryRefinementPolicy(StrictModule, NonTrainableState):
    """Deterministic marking and host-refinement resource limits."""

    strategy: BoundaryMarkingStrategy = eqx.field(static=True)
    fraction: float = eqx.field(static=True)
    max_marked_faces: int = eqx.field(static=True)
    max_target_faces: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        strategy: BoundaryMarkingStrategy = "dorfler",
        fraction: float = 0.5,
        max_marked_faces: int = 100_000,
        max_target_faces: int = 1_000_000,
    ):
        if strategy not in ("dorfler", "maximum"):
            raise ValueError("Boundary marking strategy must be 'dorfler' or 'maximum'.")
        fraction_ = float(fraction)
        marked_limit = int(max_marked_faces)
        target_limit = int(max_target_faces)
        if not math.isfinite(fraction_) or not 0.0 < fraction_ <= 1.0:
            raise ValueError("Boundary marking fraction must lie in (0, 1].")
        if marked_limit <= 0 or target_limit <= 0:
            raise ValueError("Boundary refinement resource limits must be positive.")
        self.strategy = strategy
        self.fraction = fraction_
        self.max_marked_faces = marked_limit
        self.max_target_faces = target_limit
        self.policy_id = canonical_fingerprint(
            {
                "kind": "boundary-h-refinement-policy",
                "strategy": strategy,
                "fraction": fraction_,
                "max_marked_faces": marked_limit,
                "max_target_faces": target_limit,
            }
        )


def _validated_closed_surface_mesh(mesh: CellMesh, /) -> None:
    if not isinstance(mesh, CellMesh):
        raise TypeError("Boundary adaptation requires a CellMesh.")
    if (
        mesh.topological_dimension != 2
        or mesh.ambient_dimension != 3
        or len(mesh.blocks) != 1
        or mesh.blocks[0].cell_kind != "triangle"
    ):
        raise ValueError(
            "Boundary adaptation requires one triangular 2-manifold embedded in 3D."
        )
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolygonalConnectivity):
        raise TypeError("Triangular boundary adaptation requires polygonal connectivity.")
    boundary_edges = np.asarray(connectivity.boundary_edges, dtype=bool)
    edge_counts = np.asarray(connectivity.edge_cell_counts, dtype=np.int32)
    if np.any(boundary_edges) or np.any(edge_counts != 2):
        raise ValueError("Boundary adaptation requires a closed two-manifold surface.")
    cell_edges = np.asarray(connectivity.cell_edges, dtype=np.int32)
    cell_valid = np.asarray(connectivity.cell_edge_valid, dtype=bool)
    cell_signs = np.asarray(connectivity.cell_edge_signs, dtype=float)
    orientation_balance = np.bincount(
        cell_edges[cell_valid],
        weights=cell_signs[cell_valid],
        minlength=edge_counts.size,
    )
    if np.any(orientation_balance != 0.0):
        raise ValueError("Boundary surface faces must have a consistent orientation.")
    faces = np.asarray(mesh.blocks[0].vertices, dtype=np.int32)
    triangles = np.asarray(mesh.coordinates, dtype=float)[faces]
    edges = np.stack(
        (
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 1],
            triangles[:, 0] - triangles[:, 2],
        ),
        axis=1,
    )
    doubled_areas = np.linalg.norm(np.cross(edges[:, 0], -edges[:, 2]), axis=1)
    edge_scale_squared = np.max(np.sum(edges**2, axis=2), axis=1)
    tolerance = np.finfo(float).eps * edge_scale_squared * 64.0
    if np.any(~np.isfinite(doubled_areas)) or np.any(doubled_areas <= tolerance):
        raise ValueError("Boundary surface contains a degenerate triangle.")


def _face_areas(mesh: CellMesh, /) -> np.ndarray:
    faces = np.asarray(mesh.blocks[0].vertices, dtype=np.int32)
    triangles = np.asarray(mesh.coordinates, dtype=float)[faces]
    return 0.5 * np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )


def _boundary_envelope(
    mesh: CellMesh,
    /,
    *,
    formulation: str,
    provider: str,
    resource_evidence: tuple[str, ...],
    error_evidence: tuple[str, ...],
    non_goals: tuple[str, ...],
) -> BEMExecutionEnvelope:
    return BEMExecutionEnvelope(
        ambient_dimension=3,
        pde="laplace",
        geometry="closed-oriented-triangular-surface",
        formulation=formulation,
        provider=provider,
        precision=np.dtype(mesh.coordinates.dtype).name,
        resource_evidence=resource_evidence,
        error_evidence=error_evidence,
        non_goals=non_goals,
        accelerated=False,
    )


class BoundaryMeshEpoch(StrictModule, NonTrainableState):
    """Immutable triangular surface snapshot for one Laplace DP0 BEM epoch.

    The epoch binds exact topology and geometry fingerprints. Any refinement,
    coordinate change, or numeric-version change invalidates prepared state.
    """

    mesh: CellMesh
    surface_model: SurfaceModel | None
    generation: int = eqx.field(static=True)
    parent_epoch_id: str | None = eqx.field(static=True)
    envelope: BEMExecutionEnvelope
    epoch_id: str = eqx.field(static=True)

    def __init__(
        self,
        surface: SurfaceModel | CellMesh,
        /,
        *,
        generation: int = 0,
        parent_epoch_id: str | None = None,
    ):
        if isinstance(surface, SurfaceModel):
            mesh = surface.mesh
            surface_model = surface
        elif isinstance(surface, CellMesh):
            mesh = surface
            surface_model = None
        else:
            raise TypeError("Boundary epoch requires SurfaceModel or CellMesh.")
        _validated_closed_surface_mesh(mesh)
        generation_ = int(generation)
        parent = None if parent_epoch_id is None else str(parent_epoch_id).strip()
        if generation_ < 0:
            raise ValueError("Boundary epoch generation must be nonnegative.")
        if parent_epoch_id is not None and not parent:
            raise ValueError("parent_epoch_id must be non-empty when supplied.")
        if generation_ == 0 and parent is not None:
            raise ValueError("An initial boundary epoch cannot declare a parent.")
        if generation_ > 0 and parent is None:
            raise ValueError("A successor boundary epoch must declare its parent.")
        face_count = mesh.blocks[0].cell_count
        envelope = _boundary_envelope(
            mesh,
            formulation="dp0-galerkin-boundary-epoch",
            provider=(
                "authoritative-surface-model-lineage"
                if surface_model is not None
                else "cell-mesh-triangle-lineage"
            ),
            resource_evidence=(
                f"face-count={face_count}",
                f"vertex-count={mesh.coordinates.shape[0]}",
                f"generation={generation_}",
            ),
            error_evidence=(
                "exact-topology-and-geometry-fingerprints",
                "closed-oriented-connectivity-validation",
            ),
            non_goals=(
                "continuum-discretization-certification",
                "curved-or-high-order-surface-refinement",
                "geometry-repair-or-cad",
            ),
        )
        self.mesh = mesh
        self.surface_model = surface_model
        self.generation = generation_
        self.parent_epoch_id = parent
        self.envelope = envelope
        self.epoch_id = canonical_fingerprint(
            {
                "kind": "boundary-mesh-epoch",
                "mesh": mesh.mesh_id,
                "numeric_version": mesh.numeric_version,
                "surface_model": (
                    None if surface_model is None else surface_model.model_id
                ),
                "generation": generation_,
                "parent": parent,
                "envelope": envelope.envelope_id,
            }
        )

    def validate_mesh(self, surface: SurfaceModel | CellMesh, /) -> None:
        """Reject any surface not bound to this exact authoritative epoch."""
        if self.surface_model is not None:
            if (
                not isinstance(surface, SurfaceModel)
                or surface.model_id != self.surface_model.model_id
            ):
                raise BoundaryEpochError(
                    "Boundary mesh belongs to a stale or foreign epoch."
                )
            mesh = surface.mesh
        elif isinstance(surface, SurfaceModel):
            mesh = surface.mesh
        elif isinstance(surface, CellMesh):
            mesh = surface
        else:
            raise BoundaryEpochError("Boundary mesh belongs to a stale or foreign epoch.")
        if mesh.mesh_id != self.mesh.mesh_id:
            raise BoundaryEpochError("Boundary mesh belongs to a stale or foreign epoch.")


class BoundaryRefinementMarking(StrictModule, NonTrainableState):
    """Deterministically selected global face IDs and indicator evidence."""

    marked_face_global_ids: Array
    total_indicator_energy: Array
    captured_indicator_energy: Array
    envelope: BEMExecutionEnvelope
    policy_id: str = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)
    marking_id: str = eqx.field(static=True)


class DP0BoundaryTransfer(StrictModule, NonTrainableState):
    """Constant-preserving DP0 density prolongation and exact dual transpose.

    The transfer is represented by one target-to-parent route, avoiding a dense
    target-by-source matrix. For planar h-refinement it preserves integrated
    density because each parent's child areas sum to the parent area.
    """

    target_parent_local_indices: Array
    source_face_areas: Array
    target_face_areas: Array
    maximum_area_defect: Array
    source_epoch_id: str = eqx.field(static=True)
    target_epoch_id: str = eqx.field(static=True)
    source_face_count: int = eqx.field(static=True)
    target_face_count: int = eqx.field(static=True)
    envelope: BEMExecutionEnvelope
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: BoundaryMeshEpoch,
        target: BoundaryMeshEpoch,
        parent_local_indices: ArrayLike,
        /,
    ):
        if not isinstance(source, BoundaryMeshEpoch) or not isinstance(
            target, BoundaryMeshEpoch
        ):
            raise TypeError("DP0 transfer requires source and target boundary epochs.")
        if target.parent_epoch_id != source.epoch_id:
            raise BoundaryEpochError(
                "Target boundary epoch is not a child of the source."
            )
        route = np.asarray(parent_local_indices, dtype=np.int32)
        source_count = source.mesh.blocks[0].cell_count
        target_count = target.mesh.blocks[0].cell_count
        if (
            route.shape != (target_count,)
            or np.any(route < 0)
            or np.any(route >= source_count)
        ):
            raise ValueError("DP0 target-to-parent route is invalid.")
        source_areas = _face_areas(source.mesh)
        target_areas = _face_areas(target.mesh)
        accumulated = np.zeros((source_count,), dtype=target_areas.dtype)
        np.add.at(accumulated, route, target_areas)
        defects = np.abs(accumulated - source_areas)
        maximum_defect = float(np.max(defects, initial=0.0))
        scale = float(max(np.max(source_areas, initial=0.0), 1.0))
        tolerance = np.finfo(source_areas.dtype).eps * scale * 256.0
        if maximum_defect > tolerance:
            raise ValueError(
                "DP0 h-transfer failed its parent/child area-conservation check."
            )
        envelope = _boundary_envelope(
            target.mesh,
            formulation="dp0-density-h-prolongation-and-dual-transpose",
            provider="deterministic-triangle-parent-route",
            resource_evidence=(
                f"source-face-count={source_count}",
                f"target-face-count={target_count}",
                f"route-storage-bytes={route.nbytes}",
            ),
            error_evidence=(
                f"maximum-parent-area-defect={maximum_defect:.17g}",
                "constant-preserving-row-routes",
            ),
            non_goals=(
                "higher-order-projection",
                "continuum-solution-error-estimation",
                "non-nested-or-curved-surface-transfer",
            ),
        )
        self.target_parent_local_indices = jnp.asarray(route)
        self.source_face_areas = jnp.asarray(source_areas)
        self.target_face_areas = jnp.asarray(target_areas)
        self.maximum_area_defect = jnp.asarray(maximum_defect)
        self.source_epoch_id = source.epoch_id
        self.target_epoch_id = target.epoch_id
        self.source_face_count = source_count
        self.target_face_count = target_count
        self.envelope = envelope
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "dp0-boundary-h-transfer",
                "source_epoch": source.epoch_id,
                "target_epoch": target.epoch_id,
                "parent_routes": array_tree_fingerprint(route),
                "envelope": envelope.envelope_id,
            }
        )

    def apply(
        self,
        source_coefficients: ArrayLike,
        source_epoch: BoundaryMeshEpoch,
        /,
    ) -> Array:
        """Prolong DP0 densities after enforcing the exact source epoch."""
        if (
            not isinstance(source_epoch, BoundaryMeshEpoch)
            or source_epoch.epoch_id != self.source_epoch_id
        ):
            raise BoundaryEpochError("DP0 prolongation received a stale source epoch.")
        values = jnp.asarray(source_coefficients)
        if values.ndim == 0 or values.shape[0] != self.source_face_count:
            raise ValueError(
                "DP0 source coefficients must begin with the source face axis."
            )
        return values[self.target_parent_local_indices]

    def transpose_apply(
        self,
        target_dual_values: ArrayLike,
        target_epoch: BoundaryMeshEpoch,
        /,
    ) -> Array:
        """Apply the exact algebraic transpose after target-epoch validation."""
        if (
            not isinstance(target_epoch, BoundaryMeshEpoch)
            or target_epoch.epoch_id != self.target_epoch_id
        ):
            raise BoundaryEpochError("DP0 transpose received a stale target epoch.")
        values = jnp.asarray(target_dual_values)
        if values.ndim == 0 or values.shape[0] != self.target_face_count:
            raise ValueError("DP0 target values must begin with the target face axis.")
        output = jnp.zeros(
            (self.source_face_count, *values.shape[1:]), dtype=values.dtype
        )
        return output.at[self.target_parent_local_indices].add(values)


class BoundaryRefinementResult(StrictModule, NonTrainableState):
    """Accepted local h-refinement, lineage, transfer, and bounded evidence."""

    source_epoch: BoundaryMeshEpoch
    target_epoch: BoundaryMeshEpoch
    marking: BoundaryRefinementMarking
    adaptation: FiniteElementAdaptationMap
    transfer: DP0BoundaryTransfer
    envelope: BEMExecutionEnvelope
    result_id: str = eqx.field(static=True)


def mark_boundary_faces(
    epoch: BoundaryMeshEpoch,
    indicators: ArrayLike,
    policy: BoundaryRefinementPolicy,
    /,
) -> BoundaryRefinementMarking:
    """Select global face IDs with stable indicator/ID tie breaking."""
    if not isinstance(epoch, BoundaryMeshEpoch):
        raise TypeError("epoch must be BoundaryMeshEpoch.")
    if not isinstance(policy, BoundaryRefinementPolicy):
        raise TypeError("policy must be BoundaryRefinementPolicy.")
    values = np.asarray(indicators, dtype=float)
    cell_ids = np.asarray(epoch.mesh.blocks[0].global_ids, dtype=np.int64)
    if (
        values.shape != cell_ids.shape
        or np.any(~np.isfinite(values))
        or np.any(values < 0.0)
    ):
        raise ValueError("Boundary indicators must be finite nonnegative face values.")
    squared = values**2
    total = float(np.sum(squared))
    if total == 0.0:
        marked = jnp.asarray(np.empty((0,), dtype=np.int64))
    elif policy.strategy == "dorfler":
        marked = dorfler_mark(values, policy.fraction, cell_global_ids=cell_ids)
    else:
        marked = maximum_mark(values, policy.fraction, cell_global_ids=cell_ids)
    marked_host = np.asarray(marked, dtype=np.int64)
    if marked_host.size > policy.max_marked_faces:
        raise ValueError("Boundary marking exceeds max_marked_faces.")
    local_by_id = {int(value): index for index, value in enumerate(cell_ids)}
    selected_local = np.asarray(
        [local_by_id[int(value)] for value in marked_host], dtype=np.int32
    )
    captured = float(np.sum(squared[selected_local])) if selected_local.size else 0.0
    envelope = _boundary_envelope(
        epoch.mesh,
        formulation="dp0-galerkin-h-refinement-marking",
        provider=f"deterministic-{policy.strategy}-marking",
        resource_evidence=(
            f"candidate-face-count={cell_ids.size}",
            f"marked-face-count={marked_host.size}",
            f"max-marked-faces={policy.max_marked_faces}",
        ),
        error_evidence=(
            f"total-indicator-energy={total:.17g}",
            f"captured-indicator-energy={captured:.17g}",
        ),
        non_goals=(
            "reliability-or-efficiency-proof-for-input-indicators",
            "continuum-error-certification",
            "anisotropic-or-p-refinement",
        ),
    )
    marking_id = canonical_fingerprint(
        {
            "kind": "boundary-refinement-marking",
            "epoch": epoch.epoch_id,
            "policy": policy.policy_id,
            "indicators": array_tree_fingerprint(values),
            "marked": array_tree_fingerprint(marked_host),
        }
    )
    return BoundaryRefinementMarking(
        marked_face_global_ids=jnp.asarray(marked_host),
        total_indicator_energy=jnp.asarray(total),
        captured_indicator_energy=jnp.asarray(captured),
        envelope=envelope,
        policy_id=policy.policy_id,
        epoch_id=epoch.epoch_id,
        marking_id=marking_id,
    )


def _dp0_parent_routes(
    source: BoundaryMeshEpoch,
    target_mesh: CellMesh,
    adaptation: FiniteElementAdaptationMap,
    /,
) -> np.ndarray:
    source_ids = np.asarray(source.mesh.blocks[0].global_ids, dtype=np.int64)
    target_ids = np.asarray(target_mesh.blocks[0].global_ids, dtype=np.int64)
    source_local = {int(value): index for index, value in enumerate(source_ids)}
    target_local = {int(value): index for index, value in enumerate(target_ids)}
    routes = np.full((target_ids.size,), -1, dtype=np.int32)
    for cell_id, local in source_local.items():
        if cell_id in target_local:
            routes[target_local[cell_id]] = local
    parent_ids = np.asarray(adaptation.parent_cell_ids, dtype=np.int64)
    child_ids = np.asarray(adaptation.child_cell_ids, dtype=np.int64)
    child_valid = np.asarray(adaptation.child_valid, dtype=bool)
    for parent_id, children, valid in zip(
        parent_ids, child_ids, child_valid, strict=True
    ):
        parent = source_local[int(parent_id)]
        for child_id in children[valid]:
            routes[target_local[int(child_id)]] = parent
    if np.any(routes < 0):
        raise ValueError("Local refinement did not provide complete DP0 parent lineage.")
    return routes


def refine_boundary_h(
    epoch: BoundaryMeshEpoch,
    indicators: ArrayLike,
    policy: BoundaryRefinementPolicy,
    /,
) -> BoundaryRefinementResult:
    """Mark, conformingly bisect, advance the epoch, and prepare DP0 transfer."""
    marking = mark_boundary_faces(epoch, indicators, policy)
    marked = np.asarray(marking.marked_face_global_ids, dtype=np.int64)
    if marked.size == 0:
        raise ValueError("Boundary refinement requires at least one positive indicator.")
    generation = epoch.generation + 1
    target_mesh, adaptation, _ = refine_triangles_local(
        epoch.mesh,
        marked,
        numeric_version=f"boundary-h-epoch-{generation}",
    )
    target_count = target_mesh.blocks[0].cell_count
    if target_count > policy.max_target_faces:
        raise ValueError(
            f"Refined boundary has {target_count} faces, exceeding the declared "
            f"limit {policy.max_target_faces}."
        )
    routes = _dp0_parent_routes(epoch, target_mesh, adaptation)
    if epoch.surface_model is None:
        target_surface = target_mesh
    else:
        source_model = epoch.surface_model
        source_metadata = source_model.metadata
        target_cell_ids = np.asarray(target_mesh.blocks[0].global_ids, dtype=np.int64)
        source_cell_ids = np.asarray(epoch.mesh.blocks[0].global_ids, dtype=np.int64)
        target_cell_set = target_mesh.entity_set(2)
        target_tags = (
            ()
            if not source_metadata.cell_tags
            else tuple(source_metadata.cell_tags[index] for index in routes)
        )
        target_metadata = SurfaceMetadata(
            source_id=source_metadata.source_id,
            source_revision=(
                f"{source_metadata.source_revision}:boundary-h-{generation}"
            ),
            length_unit=source_metadata.length_unit,
            coordinate_system=source_metadata.coordinate_system,
            provenance=(
                *source_metadata.provenance,
                f"deterministic local h-refinement from epoch {epoch.epoch_id}",
            ),
            cell_tags=target_tags,
        )
        selection_by_source_id: dict[str, SurfaceSelection] = {}
        original_selection_ids = {
            selection.selection_id for selection in source_model.selections
        }
        supports = list(source_model.selections)
        for interface in source_model.interfaces:
            if interface.support.selection_id not in {
                selection.selection_id for selection in supports
            }:
                supports.append(interface.support)
        target_selections = []
        for selection in supports:
            selected_parent = np.isin(
                source_cell_ids[routes],
                np.asarray(selection.cell_global_ids, dtype=np.int64),
            )
            target_selection = SurfaceSelection(
                selection.name,
                target_cell_ids[selected_parent],
                cell_entity_set_id=target_cell_set.entity_set_id,
                role=selection.role,
            )
            if selection.selection_id in original_selection_ids:
                target_selections.append(target_selection)
            selection_by_source_id[selection.selection_id] = target_selection
        target_interfaces = tuple(
            SurfaceInterface(
                interface.name,
                selection_by_source_id[interface.support.selection_id],
                minus_region=interface.minus_region,
                plus_region=interface.plus_region,
            )
            for interface in source_model.interfaces
        )
        target_surface = SurfaceModel(
            target_mesh,
            target_metadata,
            selections=tuple(target_selections),
            interfaces=target_interfaces,
        )
    target_epoch = BoundaryMeshEpoch(
        target_surface,
        generation=generation,
        parent_epoch_id=epoch.epoch_id,
    )
    transfer = DP0BoundaryTransfer(epoch, target_epoch, routes)
    envelope = _boundary_envelope(
        target_mesh,
        formulation="dp0-galerkin-local-h-refinement",
        provider="longest-edge-conforming-triangle-bisection",
        resource_evidence=(
            f"source-face-count={epoch.mesh.blocks[0].cell_count}",
            f"target-face-count={target_count}",
            f"marked-face-count={marked.size}",
        ),
        error_evidence=(
            "exact-parent-child-lineage",
            f"maximum-parent-area-defect={float(transfer.maximum_area_defect):.17g}",
        ),
        non_goals=(
            "continuum-error-certification",
            "automatic-indicator-construction",
            "coarsening-or-high-order-geometry",
        ),
    )
    return BoundaryRefinementResult(
        source_epoch=epoch,
        target_epoch=target_epoch,
        marking=marking,
        adaptation=adaptation,
        transfer=transfer,
        envelope=envelope,
        result_id=canonical_fingerprint(
            {
                "kind": "boundary-h-refinement-result",
                "source_epoch": epoch.epoch_id,
                "target_epoch": target_epoch.epoch_id,
                "marking": marking.marking_id,
                "adaptation": adaptation.adaptation_id,
                "transfer": transfer.transfer_id,
                "envelope": envelope.envelope_id,
            }
        ),
    )


__all__ = [
    "BoundaryEpochError",
    "BoundaryMarkingStrategy",
    "BoundaryMeshEpoch",
    "BoundaryRefinementMarking",
    "BoundaryRefinementPolicy",
    "BoundaryRefinementResult",
    "DP0BoundaryTransfer",
    "mark_boundary_faces",
    "refine_boundary_h",
]
