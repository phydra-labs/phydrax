#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Opt-in certified surface mesh repair backed by native meshcore kernels.

Every geometry-changing operation is explicitly gated on ``SurfaceRepairPolicy``
and defaults to disabled. Destructive edits record exact displacement bounds,
entity counts, closure evidence, and embedding verification in
``SurfaceRepairEvidence``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never, Literal, TypeAlias

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._meshcore import (
    arrange_soup,
    arrange_soup_extract_solid,
    ArrangementFailure,
    mesh_certify,
    mesh_remesh,
    MeshcoreError,
    SolidWindingCode,
    surface_fill_holes,
    surface_orient_and_prune,
    surface_weld_and_clean,
    SurfaceFindingKind,
    verify_surface_embedding,
)
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...typing import parse
from ._boolean import SurfaceClosureEvidence
from ._contracts import SurfaceMetadata
from ._model import _surface_connectivity, SurfaceModel


SolidWindingRule: TypeAlias = Literal[
    "nonzero",
    "positive",
    "even_odd",
    "generalized_half",
]

SurfaceRepairStatus: TypeAlias = Literal[
    "invalid_input",
    "weld_fold",
    "cap_self_intersection",
    "open_boundary",
    "nonmanifold_output",
    "embedding_failed",
    "empty_output",
    "remesh_failed",
]


class SurfaceRepairError(ValueError):
    """Fail-closed surface repair refusal carrying canonical status and counts."""

    def __init__(
        self,
        status: SurfaceRepairStatus,
        message: str,
        /,
        *,
        finding_count: int = 0,
    ) -> None:
        status = parse(status, SurfaceRepairStatus, "status")
        self.status = status
        self.finding_count = int(finding_count)
        super().__init__(f"{message} (status {status})")


def _nonnegative_float(value: float, name: str, /) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return parsed


def _unit_fraction(value: float, name: str, /) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must lie in [0, 1].")
    return parsed


def _nonnegative_int(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return parsed


def _strict_bool(value: bool, name: str, /) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool.")
    return value


def _winding_code(rule: SolidWindingRule, /) -> SolidWindingCode:
    match rule:
        case "nonzero":
            return SolidWindingCode.NONZERO
        case "positive":
            return SolidWindingCode.POSITIVE
        case "even_odd":
            return SolidWindingCode.EVEN_ODD
        case "generalized_half":
            return SolidWindingCode.GENERALIZED_HALF
        case _:
            assert_never(rule)


class SurfaceRepairPolicy(StrictModule, NonTrainableState):
    """Explicit opt-in policy for certified surface mesh repair.

    All geometry-changing switches default to ``False`` / ``0.0`` / ``None`` so
    a default policy performs zero-displacement audit and exact bitwise welding
    only. Use :meth:`watertight_solid` for the canonical solid-repair preset.
    """

    weld_tolerance: float = eqx.field(static=True)
    t_junction_tolerance: float = eqx.field(static=True)
    drop_degenerate_triangles: bool = eqx.field(static=True)
    min_edge_length: float = eqx.field(static=True)
    min_triangle_area: float = eqx.field(static=True)
    resolve_self_intersections: bool = eqx.field(static=True)
    fill_holes: bool = eqx.field(static=True)
    max_hole_edges: int = eqx.field(static=True)
    max_hole_diameter: float = eqx.field(static=True)
    fair_hole_caps: bool = eqx.field(static=True)
    refine_hole_caps: bool = eqx.field(static=True)
    extract_solid: bool = eqx.field(static=True)
    winding_rule: SolidWindingRule = eqx.field(static=True)
    split_nonmanifold_pinches: bool = eqx.field(static=True)
    orient_shells: bool = eqx.field(static=True)
    prune_small_components: bool = eqx.field(static=True)
    min_component_faces: int = eqx.field(static=True)
    min_component_area_fraction: float = eqx.field(static=True)
    min_component_volume_fraction: float = eqx.field(static=True)
    polish_edge_length: float | None = eqx.field(static=True)
    polish_tolerance: float | None = eqx.field(static=True)
    polish_minimum_angle_degrees: float = eqx.field(static=True)
    require_watertight: bool = eqx.field(static=True)
    require_manifold: bool = eqx.field(static=True)
    verify_embedding: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        weld_tolerance: float = 0.0,
        t_junction_tolerance: float = 0.0,
        drop_degenerate_triangles: bool = False,
        min_edge_length: float = 0.0,
        min_triangle_area: float = 0.0,
        resolve_self_intersections: bool = False,
        fill_holes: bool = False,
        max_hole_edges: int = 0,
        max_hole_diameter: float = 0.0,
        fair_hole_caps: bool = False,
        refine_hole_caps: bool = False,
        extract_solid: bool = False,
        winding_rule: SolidWindingRule = "nonzero",
        split_nonmanifold_pinches: bool = False,
        orient_shells: bool = False,
        prune_small_components: bool = False,
        min_component_faces: int = 0,
        min_component_area_fraction: float = 0.0,
        min_component_volume_fraction: float = 0.0,
        polish_edge_length: float | None = None,
        polish_tolerance: float | None = None,
        polish_minimum_angle_degrees: float = 30.0,
        require_watertight: bool = False,
        require_manifold: bool = False,
        verify_embedding: bool = True,
    ) -> None:
        weld_tol = _nonnegative_float(weld_tolerance, "weld_tolerance")
        tj_tol = _nonnegative_float(t_junction_tolerance, "t_junction_tolerance")
        drop_deg = _strict_bool(drop_degenerate_triangles, "drop_degenerate_triangles")
        min_edge = _nonnegative_float(min_edge_length, "min_edge_length")
        min_area = _nonnegative_float(min_triangle_area, "min_triangle_area")
        resolve_si = _strict_bool(
            resolve_self_intersections, "resolve_self_intersections"
        )
        do_fill = _strict_bool(fill_holes, "fill_holes")
        max_edges = _nonnegative_int(max_hole_edges, "max_hole_edges")
        max_diam = _nonnegative_float(max_hole_diameter, "max_hole_diameter")
        fair_caps = _strict_bool(fair_hole_caps, "fair_hole_caps")
        refine_caps = _strict_bool(refine_hole_caps, "refine_hole_caps")
        do_solid = _strict_bool(extract_solid, "extract_solid")
        rule = parse(winding_rule, SolidWindingRule, "winding_rule")
        split_pinches = _strict_bool(
            split_nonmanifold_pinches, "split_nonmanifold_pinches"
        )
        do_orient = _strict_bool(orient_shells, "orient_shells")
        do_prune = _strict_bool(prune_small_components, "prune_small_components")
        min_faces = _nonnegative_int(min_component_faces, "min_component_faces")
        min_area_frac = _unit_fraction(
            min_component_area_fraction, "min_component_area_fraction"
        )
        min_vol_frac = _unit_fraction(
            min_component_volume_fraction, "min_component_volume_fraction"
        )
        req_watertight = _strict_bool(require_watertight, "require_watertight")
        req_manifold = _strict_bool(require_manifold, "require_manifold")
        do_verify = _strict_bool(verify_embedding, "verify_embedding")
        if (polish_edge_length is None) != (polish_tolerance is None):
            raise ValueError(
                "polish_edge_length and polish_tolerance must either both be set or both None."
            )
        p_edge: float | None = None
        p_tol: float | None = None
        if polish_edge_length is not None and polish_tolerance is not None:
            p_edge = float(polish_edge_length)
            p_tol = float(polish_tolerance)
            if not (np.isfinite(p_edge) and p_edge > 0.0):
                raise ValueError("polish_edge_length must be positive and finite.")
            if not (np.isfinite(p_tol) and p_tol > 0.0):
                raise ValueError("polish_tolerance must be positive and finite.")
        p_angle = float(polish_minimum_angle_degrees)
        if not (np.isfinite(p_angle) and 0.0 < p_angle < 60.0):
            raise ValueError("polish_minimum_angle_degrees must lie in (0, 60).")

        self.weld_tolerance = weld_tol
        self.t_junction_tolerance = tj_tol
        self.drop_degenerate_triangles = drop_deg
        self.min_edge_length = min_edge
        self.min_triangle_area = min_area
        self.resolve_self_intersections = resolve_si
        self.fill_holes = do_fill
        self.max_hole_edges = max_edges
        self.max_hole_diameter = max_diam
        self.fair_hole_caps = fair_caps
        self.refine_hole_caps = refine_caps
        self.extract_solid = do_solid
        self.winding_rule = rule
        self.split_nonmanifold_pinches = split_pinches
        self.orient_shells = do_orient
        self.prune_small_components = do_prune
        self.min_component_faces = min_faces
        self.min_component_area_fraction = min_area_frac
        self.min_component_volume_fraction = min_vol_frac
        self.polish_edge_length = p_edge
        self.polish_tolerance = p_tol
        self.polish_minimum_angle_degrees = p_angle
        self.require_watertight = req_watertight
        self.require_manifold = req_manifold
        self.verify_embedding = do_verify
        self.policy_id = canonical_fingerprint(
            {
                "kind": "surface-repair-policy",
                "weld_tolerance": weld_tol,
                "t_junction_tolerance": tj_tol,
                "drop_degenerate_triangles": drop_deg,
                "min_edge_length": min_edge,
                "min_triangle_area": min_area,
                "resolve_self_intersections": resolve_si,
                "fill_holes": do_fill,
                "max_hole_edges": max_edges,
                "max_hole_diameter": max_diam,
                "fair_hole_caps": fair_caps,
                "refine_hole_caps": refine_caps,
                "extract_solid": do_solid,
                "winding_rule": rule,
                "split_nonmanifold_pinches": split_pinches,
                "orient_shells": do_orient,
                "prune_small_components": do_prune,
                "min_component_faces": min_faces,
                "min_component_area_fraction": min_area_frac,
                "min_component_volume_fraction": min_vol_frac,
                "polish_edge_length": p_edge,
                "polish_tolerance": p_tol,
                "polish_minimum_angle_degrees": p_angle,
                "require_watertight": req_watertight,
                "require_manifold": req_manifold,
                "verify_embedding": do_verify,
            }
        )

    @classmethod
    def watertight_solid(
        cls,
        *,
        weld_tolerance: float = 0.0,
        t_junction_tolerance: float = 0.0,
        min_edge_length: float = 0.0,
        min_triangle_area: float = 0.0,
        max_hole_edges: int = 0,
        max_hole_diameter: float = 0.0,
        fair_hole_caps: bool = True,
        refine_hole_caps: bool = False,
        winding_rule: SolidWindingRule = "nonzero",
        split_nonmanifold_pinches: bool = False,
        min_component_faces: int = 0,
        min_component_area_fraction: float = 0.0,
        polish_edge_length: float | None = None,
        polish_tolerance: float | None = None,
        polish_minimum_angle_degrees: float = 30.0,
    ) -> SurfaceRepairPolicy:
        """Explicit opt-in preset repairing a dirty soup into a certified watertight solid."""

        return cls(
            weld_tolerance=weld_tolerance,
            t_junction_tolerance=t_junction_tolerance,
            drop_degenerate_triangles=True,
            min_edge_length=min_edge_length,
            min_triangle_area=min_triangle_area,
            resolve_self_intersections=True,
            fill_holes=True,
            max_hole_edges=max_hole_edges,
            max_hole_diameter=max_hole_diameter,
            fair_hole_caps=fair_hole_caps,
            refine_hole_caps=refine_hole_caps,
            extract_solid=True,
            winding_rule=winding_rule,
            split_nonmanifold_pinches=split_nonmanifold_pinches,
            orient_shells=True,
            prune_small_components=(
                min_component_faces > 0 or min_component_area_fraction > 0.0
            ),
            min_component_faces=min_component_faces,
            min_component_area_fraction=min_component_area_fraction,
            polish_edge_length=polish_edge_length,
            polish_tolerance=polish_tolerance,
            polish_minimum_angle_degrees=polish_minimum_angle_degrees,
            require_watertight=True,
            require_manifold=True,
            verify_embedding=True,
        )


class SurfaceRepairEvidence(StrictModule, NonTrainableState):
    """Exact counts, bounds, topology closure, and certification evidence for a repair."""

    input_vertex_count: int = eqx.field(static=True)
    input_triangle_count: int = eqx.field(static=True)
    output_vertex_count: int = eqx.field(static=True)
    output_triangle_count: int = eqx.field(static=True)
    max_vertex_displacement: float = eqx.field(static=True)
    welded_vertex_count: int = eqx.field(static=True)
    t_junction_splits: int = eqx.field(static=True)
    collapsed_sliver_edges: int = eqx.field(static=True)
    flipped_cap_edges: int = eqx.field(static=True)
    dropped_degenerate_faces: int = eqx.field(static=True)
    dropped_duplicate_faces: int = eqx.field(static=True)
    rejected_fold_welds: int = eqx.field(static=True)
    arrangement_fragments: int = eqx.field(static=True)
    canceled_coplanar_fragments: int = eqx.field(static=True)
    filled_hole_count: int = eqx.field(static=True)
    skipped_large_holes: int = eqx.field(static=True)
    added_cap_triangles: int = eqx.field(static=True)
    filled_hole_area: float = eqx.field(static=True)
    flipped_face_count: int = eqx.field(static=True)
    cavity_component_count: int = eqx.field(static=True)
    pruned_component_count: int = eqx.field(static=True)
    pruned_face_count: int = eqx.field(static=True)
    split_pinch_vertices: int = eqx.field(static=True)
    used_generalized_winding: bool = eqx.field(static=True)
    closure: SurfaceClosureEvidence
    embedding_verified: bool = eqx.field(static=True)
    hausdorff_bound: float | None = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        input_vertex_count: int,
        input_triangle_count: int,
        output_vertex_count: int,
        output_triangle_count: int,
        max_vertex_displacement: float,
        welded_vertex_count: int,
        t_junction_splits: int,
        collapsed_sliver_edges: int,
        flipped_cap_edges: int,
        dropped_degenerate_faces: int,
        dropped_duplicate_faces: int,
        rejected_fold_welds: int,
        arrangement_fragments: int,
        canceled_coplanar_fragments: int,
        filled_hole_count: int,
        skipped_large_holes: int,
        added_cap_triangles: int,
        filled_hole_area: float,
        flipped_face_count: int,
        cavity_component_count: int,
        pruned_component_count: int,
        pruned_face_count: int,
        split_pinch_vertices: int,
        used_generalized_winding: bool,
        closure: SurfaceClosureEvidence,
        embedding_verified: bool,
        hausdorff_bound: float | None,
    ) -> None:
        if not isinstance(closure, SurfaceClosureEvidence):
            raise TypeError("closure must be SurfaceClosureEvidence.")
        self.input_vertex_count = _nonnegative_int(
            input_vertex_count, "input_vertex_count"
        )
        self.input_triangle_count = _nonnegative_int(
            input_triangle_count, "input_triangle_count"
        )
        self.output_vertex_count = _nonnegative_int(
            output_vertex_count, "output_vertex_count"
        )
        self.output_triangle_count = _nonnegative_int(
            output_triangle_count, "output_triangle_count"
        )
        self.max_vertex_displacement = _nonnegative_float(
            max_vertex_displacement, "max_vertex_displacement"
        )
        self.welded_vertex_count = _nonnegative_int(
            welded_vertex_count, "welded_vertex_count"
        )
        self.t_junction_splits = _nonnegative_int(t_junction_splits, "t_junction_splits")
        self.collapsed_sliver_edges = _nonnegative_int(
            collapsed_sliver_edges, "collapsed_sliver_edges"
        )
        self.flipped_cap_edges = _nonnegative_int(flipped_cap_edges, "flipped_cap_edges")
        self.dropped_degenerate_faces = _nonnegative_int(
            dropped_degenerate_faces, "dropped_degenerate_faces"
        )
        self.dropped_duplicate_faces = _nonnegative_int(
            dropped_duplicate_faces, "dropped_duplicate_faces"
        )
        self.rejected_fold_welds = _nonnegative_int(
            rejected_fold_welds, "rejected_fold_welds"
        )
        self.arrangement_fragments = _nonnegative_int(
            arrangement_fragments, "arrangement_fragments"
        )
        self.canceled_coplanar_fragments = _nonnegative_int(
            canceled_coplanar_fragments, "canceled_coplanar_fragments"
        )
        self.filled_hole_count = _nonnegative_int(filled_hole_count, "filled_hole_count")
        self.skipped_large_holes = _nonnegative_int(
            skipped_large_holes, "skipped_large_holes"
        )
        self.added_cap_triangles = _nonnegative_int(
            added_cap_triangles, "added_cap_triangles"
        )
        self.filled_hole_area = _nonnegative_float(filled_hole_area, "filled_hole_area")
        self.flipped_face_count = _nonnegative_int(
            flipped_face_count, "flipped_face_count"
        )
        self.cavity_component_count = _nonnegative_int(
            cavity_component_count, "cavity_component_count"
        )
        self.pruned_component_count = _nonnegative_int(
            pruned_component_count, "pruned_component_count"
        )
        self.pruned_face_count = _nonnegative_int(pruned_face_count, "pruned_face_count")
        self.split_pinch_vertices = _nonnegative_int(
            split_pinch_vertices, "split_pinch_vertices"
        )
        self.used_generalized_winding = _strict_bool(
            used_generalized_winding, "used_generalized_winding"
        )
        self.closure = closure
        self.embedding_verified = _strict_bool(embedding_verified, "embedding_verified")
        self.hausdorff_bound = (
            None
            if hausdorff_bound is None
            else _nonnegative_float(hausdorff_bound, "hausdorff_bound")
        )
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "surface-repair-evidence",
                "input_vertex_count": self.input_vertex_count,
                "input_triangle_count": self.input_triangle_count,
                "output_vertex_count": self.output_vertex_count,
                "output_triangle_count": self.output_triangle_count,
                "max_vertex_displacement": self.max_vertex_displacement,
                "welded_vertex_count": self.welded_vertex_count,
                "t_junction_splits": self.t_junction_splits,
                "collapsed_sliver_edges": self.collapsed_sliver_edges,
                "flipped_cap_edges": self.flipped_cap_edges,
                "dropped_degenerate_faces": self.dropped_degenerate_faces,
                "dropped_duplicate_faces": self.dropped_duplicate_faces,
                "rejected_fold_welds": self.rejected_fold_welds,
                "arrangement_fragments": self.arrangement_fragments,
                "canceled_coplanar_fragments": self.canceled_coplanar_fragments,
                "filled_hole_count": self.filled_hole_count,
                "skipped_large_holes": self.skipped_large_holes,
                "added_cap_triangles": self.added_cap_triangles,
                "filled_hole_area": self.filled_hole_area,
                "flipped_face_count": self.flipped_face_count,
                "cavity_component_count": self.cavity_component_count,
                "pruned_component_count": self.pruned_component_count,
                "pruned_face_count": self.pruned_face_count,
                "split_pinch_vertices": self.split_pinch_vertices,
                "used_generalized_winding": self.used_generalized_winding,
                "closure_id": closure.evidence_id,
                "embedding_verified": self.embedding_verified,
                "hausdorff_bound": self.hausdorff_bound,
            }
        )


class SurfaceRepairResult(StrictModule, NonTrainableState):
    """Repaired authoritative surface model and complete repair provenance."""

    surface: SurfaceModel
    evidence: SurfaceRepairEvidence
    policy: SurfaceRepairPolicy
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        surface: SurfaceModel,
        evidence: SurfaceRepairEvidence,
        policy: SurfaceRepairPolicy,
    ) -> None:
        if not isinstance(surface, SurfaceModel):
            raise TypeError("surface must be SurfaceModel.")
        if not isinstance(evidence, SurfaceRepairEvidence):
            raise TypeError("evidence must be SurfaceRepairEvidence.")
        if not isinstance(policy, SurfaceRepairPolicy):
            raise TypeError("policy must be SurfaceRepairPolicy.")
        self.surface = surface
        self.evidence = evidence
        self.policy = policy
        self.result_id = canonical_fingerprint(
            {
                "kind": "surface-repair-result",
                "model_id": surface.model_id,
                "evidence_id": evidence.evidence_id,
                "policy_id": policy.policy_id,
            }
        )


@dataclass
class _RepairCounters:
    max_vertex_displacement: float = 0.0
    welded_vertex_count: int = 0
    t_junction_splits: int = 0
    collapsed_sliver_edges: int = 0
    flipped_cap_edges: int = 0
    dropped_degenerate_faces: int = 0
    dropped_duplicate_faces: int = 0
    rejected_fold_welds: int = 0
    arrangement_fragments: int = 0
    canceled_coplanar_fragments: int = 0
    filled_hole_count: int = 0
    skipped_large_holes: int = 0
    added_cap_triangles: int = 0
    filled_hole_area: float = 0.0
    flipped_face_count: int = 0
    cavity_component_count: int = 0
    pruned_component_count: int = 0
    pruned_face_count: int = 0
    split_pinch_vertices: int = 0
    used_generalized_winding: bool = False
    hausdorff_bound: float | None = None


def _compute_closure_evidence(
    vertices: np.ndarray, triangles: np.ndarray, /
) -> SurfaceClosureEvidence:
    tri_count = triangles.shape[0]
    if tri_count == 0:
        return SurfaceClosureEvidence(
            triangle_count=0,
            component_count=0,
            boundary_edges=0,
            nonmanifold_edges=0,
            nonmanifold_vertices=0,
            signed_volume=0.0,
        )
    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]
    p0 = vertices[a]
    p1 = vertices[b]
    p2 = vertices[c]
    signed_volume = float(np.sum(p0 * np.cross(p1, p2)) / 6.0)

    e0 = np.stack((a, b), axis=1)
    e1 = np.stack((b, c), axis=1)
    e2 = np.stack((c, a), axis=1)
    directed = np.concatenate((e0, e1, e2), axis=0)
    undirected = np.sort(directed, axis=1)
    forward = directed[:, 0] < directed[:, 1]
    corners = np.arange(3 * tri_count, dtype=np.int64)
    # Corner index per undirected edge entry: edge k of triangle t has start corner 3*t + k.
    tri_idx = np.tile(np.arange(tri_count, dtype=np.int64), 3)
    local_k = np.repeat(np.arange(3, dtype=np.int64), tri_count)
    start_corners = 3 * tri_idx + local_k
    order = np.lexsort((corners, undirected[:, 1], undirected[:, 0]))
    sorted_u = undirected[order]
    sorted_fwd = forward[order]
    sorted_tris = tri_idx[order]
    sorted_sc = start_corners[order]

    change = np.empty((sorted_u.shape[0] + 1,), dtype=np.bool_)
    change[0] = True
    change[-1] = True
    change[1:-1] = np.any(sorted_u[1:] != sorted_u[:-1], axis=1)
    starts = np.flatnonzero(change)
    counts = np.diff(starts)

    boundary_edges = int(np.sum(counts == 1))
    nonmanifold_edges = int(np.sum(counts > 2))

    # Build DSU over triangle components and vertex-corner fans.
    face_parent = list(range(tri_count))

    def _find(parent: list[int], x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(parent: list[int], x: int, y: int) -> None:
        rx = _find(parent, x)
        ry = _find(parent, y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    corner_parent = list(range(3 * tri_count))
    for group_idx, count in enumerate(counts):
        s = int(starts[group_idx])
        c_int = int(count)
        first_face = int(sorted_tris[s])
        for offset in range(1, c_int):
            _union(face_parent, first_face, int(sorted_tris[s + offset]))
        if c_int == 2:
            if bool(sorted_fwd[s]) == bool(sorted_fwd[s + 1]):
                nonmanifold_edges += 1
            else:
                c0 = int(sorted_sc[s])
                c0n = c0 - 2 if c0 % 3 == 2 else c0 + 1
                c1 = int(sorted_sc[s + 1])
                c1n = c1 - 2 if c1 % 3 == 2 else c1 + 1
                if bool(sorted_fwd[s]):
                    _union(corner_parent, c0, c1n)
                    _union(corner_parent, c0n, c1)
                else:
                    _union(corner_parent, c0n, c1)
                    _union(corner_parent, c0, c1n)

    components = len({_find(face_parent, i) for i in range(tri_count)})
    first_root: dict[int, int] = {}
    nonmanifold_vertices = 0
    for t in range(tri_count):
        for k in range(3):
            v = int(triangles[t, k])
            r = _find(corner_parent, 3 * t + k)
            prev = first_root.get(v)
            if prev is None:
                first_root[v] = r
            elif prev >= 0 and prev != r:
                nonmanifold_vertices += 1
                first_root[v] = -2

    return SurfaceClosureEvidence(
        triangle_count=tri_count,
        component_count=components,
        boundary_edges=boundary_edges,
        nonmanifold_edges=nonmanifold_edges,
        nonmanifold_vertices=nonmanifold_vertices,
        signed_volume=signed_volume,
    )


def _extract_input_arrays(
    surface: SurfaceModel | tuple[ArrayLike, ArrayLike],
    metadata: SurfaceMetadata | None,
    /,
) -> tuple[np.ndarray, np.ndarray, SurfaceMetadata]:
    if isinstance(surface, SurfaceModel):
        pts = np.asarray(surface.mesh.coordinates, dtype=np.float64)
        conn = _surface_connectivity(surface.mesh)
        tris = np.asarray(conn.cell_vertices, dtype=np.int64)[:, :3]
        base_meta = surface.metadata if metadata is None else metadata
        return np.ascontiguousarray(pts), np.ascontiguousarray(tris), base_meta
    if not isinstance(surface, tuple) or len(surface) != 2:
        raise TypeError(
            "surface must be a SurfaceModel or a (vertices, triangles) tuple."
        )
    pts = np.ascontiguousarray(np.asarray(surface[0], dtype=np.float64))
    tris = np.ascontiguousarray(np.asarray(surface[1], dtype=np.int64))
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise SurfaceRepairError(
            "invalid_input", "Input vertices must have shape (V, 3)."
        )
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise SurfaceRepairError(
            "invalid_input", "Input triangles must have shape (T, 3)."
        )
    if not np.all(np.isfinite(pts)):
        raise SurfaceRepairError(
            "invalid_input", "Input vertices must contain finite coordinates."
        )
    if tris.shape[0] > 0 and (np.any(tris < 0) or np.any(tris >= pts.shape[0])):
        raise SurfaceRepairError(
            "invalid_input", "Input triangle vertex indices are out of range."
        )
    base_meta = (
        metadata
        if metadata is not None
        else SurfaceMetadata(
            source_id="repaired-triangle-soup",
            source_revision="raw-soup",
            coordinate_contract=SpatialCoordinateContract.si(),
            provenance=("raw-triangle-soup",),
        )
    )
    return pts, tris, base_meta


def _run_arrangement_stage(
    pts: np.ndarray,
    tris: np.ndarray,
    policy: SurfaceRepairPolicy,
    counters: _RepairCounters,
    *,
    extract_solid_now: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if tris.shape[0] == 0:
        return pts, tris
    try:
        if extract_solid_now:
            ext = arrange_soup_extract_solid(
                pts,
                tris,
                winding_rule=_winding_code(policy.winding_rule),
                split_nonmanifold_pinches=policy.split_nonmanifold_pinches,
            )
            counters.arrangement_fragments += ext.fragment_count
            counters.canceled_coplanar_fragments += ext.canceled_coplanar_fragments
            counters.split_pinch_vertices += ext.split_pinch_vertices
            counters.used_generalized_winding = (
                counters.used_generalized_winding or ext.used_generalized_winding
            )
            return ext.vertices, ext.triangles
        arr = arrange_soup(pts, tris)
        counters.arrangement_fragments += arr.fragments.shape[0]
        return arr.vertices, arr.fragments
    except (ArrangementFailure, MeshcoreError) as error:
        raise SurfaceRepairError(
            "embedding_failed",
            f"Exact self-arrangement refused input soup: {error}",
        ) from error


def _run_hole_fill_stage(
    pts: np.ndarray,
    tris: np.ndarray,
    policy: SurfaceRepairPolicy,
    counters: _RepairCounters,
) -> tuple[np.ndarray, np.ndarray, bool]:
    filled = surface_fill_holes(
        pts,
        tris,
        max_hole_edges=policy.max_hole_edges,
        max_hole_diameter=policy.max_hole_diameter,
        fair_caps=policy.fair_hole_caps,
        refine_caps=policy.refine_hole_caps,
        check_embedding=True,
    )
    counters.filled_hole_count += filled.filled_hole_count
    counters.skipped_large_holes += filled.skipped_large_holes
    counters.added_cap_triangles += filled.added_cap_triangles
    counters.filled_hole_area += filled.filled_hole_area
    had_cap_collision = filled.colliding_cap_triangles > 0
    if had_cap_collision and not policy.resolve_self_intersections:
        raise SurfaceRepairError(
            "cap_self_intersection",
            (
                f"Hole filling produced {filled.colliding_cap_triangles} cap triangle(s) "
                "colliding with existing geometry; enable resolve_self_intersections to "
                "resolve cap collisions."
            ),
            finding_count=filled.colliding_cap_triangles,
        )
    return filled.vertices, filled.triangles, had_cap_collision


def _run_polish_stage(
    pts: np.ndarray,
    tris: np.ndarray,
    policy: SurfaceRepairPolicy,
    counters: _RepairCounters,
) -> tuple[np.ndarray, np.ndarray]:
    if policy.polish_edge_length is None or policy.polish_tolerance is None:
        return pts, tris
    try:
        remeshed = mesh_remesh(
            pts,
            tris,
            policy.polish_tolerance,
            policy.polish_edge_length,
            policy.polish_minimum_angle_degrees,
        )
        cert = mesh_certify(
            pts,
            tris,
            remeshed.vertices,
            remeshed.triangles,
            policy.polish_tolerance,
            policy.polish_edge_length,
            policy.polish_minimum_angle_degrees,
        )
    except (MeshcoreError, ValueError) as error:
        raise SurfaceRepairError(
            "remesh_failed",
            f"Certified surface polish failed: {error}",
        ) from error
    counters.hausdorff_bound = float(np.max(cert.distances))
    return cert.vertices, cert.triangles


def _verify_repaired_surface(
    pts: np.ndarray,
    tris: np.ndarray,
    policy: SurfaceRepairPolicy,
    counters: _RepairCounters,
    /,
) -> tuple[SurfaceClosureEvidence, bool]:
    closure = _compute_closure_evidence(pts, tris)
    if (policy.require_watertight or policy.extract_solid) and closure.boundary_edges > 0:
        raise SurfaceRepairError(
            "open_boundary",
            (
                f"Repaired surface retains {closure.boundary_edges} open boundary edge(s); "
                "enable fill_holes, widen max_hole_diameter, or wrap with SurfaceEnvelopeSource."
            ),
            finding_count=closure.boundary_edges,
        )
    if policy.require_manifold and (
        closure.nonmanifold_edges > 0 or closure.nonmanifold_vertices > 0
    ):
        raise SurfaceRepairError(
            "nonmanifold_output",
            (
                f"Repaired surface has {closure.nonmanifold_edges} non-manifold edge(s) and "
                f"{closure.nonmanifold_vertices} non-manifold vertex/vertices; enable "
                "split_nonmanifold_pinches for strict combinatorial 2-manifoldness."
            ),
            finding_count=closure.nonmanifold_edges + closure.nonmanifold_vertices,
        )
    if not policy.verify_embedding:
        return closure, False
    embedding = verify_surface_embedding(pts, tris)
    finding_count = int(embedding.findings.shape[0])
    if closure.boundary_edges == 0 and counters.split_pinch_vertices == 0:
        if finding_count > 0:
            raise SurfaceRepairError(
                "embedding_failed",
                f"Post-repair exact embedding audit reported {finding_count} defect(s).",
                finding_count=finding_count,
            )
        return closure, True
    illegal_kinds = (
        int(SurfaceFindingKind.OUTSIDE_DOMAIN),
        int(SurfaceFindingKind.DEGENERATE),
        int(SurfaceFindingKind.CONTACT),
    )
    kinds = (
        embedding.findings["kind"]
        if finding_count > 0
        else np.empty((0,), dtype=np.int32)
    )
    illegal = int(np.sum(np.isin(kinds, illegal_kinds)))
    if illegal > 0 and counters.split_pinch_vertices == 0:
        raise SurfaceRepairError(
            "embedding_failed",
            f"Post-repair exact contact audit reported {illegal} illegal contact(s).",
            finding_count=illegal,
        )
    return closure, finding_count == 0


def repair_surface_mesh(
    surface: SurfaceModel | tuple[ArrayLike, ArrayLike],
    /,
    *,
    policy: SurfaceRepairPolicy | None = None,
    metadata: SurfaceMetadata | None = None,
) -> SurfaceRepairResult:
    """Repair a triangle surface mesh or raw triangle soup under explicit opt-in policy."""

    active_policy = SurfaceRepairPolicy() if policy is None else policy
    if not isinstance(active_policy, SurfaceRepairPolicy):
        raise TypeError("policy must be SurfaceRepairPolicy or None.")
    pts, tris, base_meta = _extract_input_arrays(surface, metadata)
    input_v = pts.shape[0]
    input_t = tris.shape[0]
    counters = _RepairCounters()

    # Stage 0/1/2: exact bitwise dedup + opt-in displacement-bounded welding, T-junctions, cleanup.
    cleaned = surface_weld_and_clean(
        pts,
        tris,
        weld_tolerance=active_policy.weld_tolerance,
        t_junction_tolerance=active_policy.t_junction_tolerance,
        min_edge_length=active_policy.min_edge_length,
        min_triangle_area=active_policy.min_triangle_area,
        drop_degenerate=active_policy.drop_degenerate_triangles,
        preserve_embedding=not active_policy.resolve_self_intersections,
    )
    pts, tris = cleaned.vertices, cleaned.triangles
    counters.welded_vertex_count = cleaned.welded_vertices
    counters.t_junction_splits = cleaned.t_junction_splits
    counters.collapsed_sliver_edges = cleaned.collapsed_edges
    counters.flipped_cap_edges = cleaned.flipped_caps
    counters.dropped_degenerate_faces = cleaned.dropped_degenerate_faces
    counters.dropped_duplicate_faces = cleaned.dropped_duplicate_faces
    counters.rejected_fold_welds = cleaned.rejected_fold_welds
    counters.max_vertex_displacement = cleaned.max_vertex_displacement
    if tris.shape[0] == 0:
        raise SurfaceRepairError(
            "empty_output", "No valid triangles remained after surface cleaning."
        )

    # Stage 3: exact single-soup self-arrangement (before hole filling to expose true loops).
    extracted_solid = False
    if active_policy.resolve_self_intersections:
        direct_solid = active_policy.extract_solid and not active_policy.fill_holes
        pts, tris = _run_arrangement_stage(
            pts, tris, active_policy, counters, extract_solid_now=direct_solid
        )
        extracted_solid = direct_solid

    # Stage 4: certified Liepa DP hole filling (with exact cap-collision check).
    had_cap_collision = False
    if active_policy.fill_holes and tris.shape[0] > 0:
        pts, tris, had_cap_collision = _run_hole_fill_stage(
            pts, tris, active_policy, counters
        )

    # Stage 5a: exact 3D arrangement cell-complex solid extraction.
    if not extracted_solid and (active_policy.extract_solid or had_cap_collision):
        pts, tris = _run_arrangement_stage(
            pts, tris, active_policy, counters, extract_solid_now=True
        )
        extracted_solid = True
    if tris.shape[0] == 0:
        raise SurfaceRepairError(
            "empty_output", "Surface repair produced zero surviving triangles."
        )

    # Stage 5b / 6a: shell orientation & debris pruning.
    if active_policy.orient_shells or active_policy.prune_small_components:
        oriented = surface_orient_and_prune(
            pts,
            tris,
            orient_shells=active_policy.orient_shells and not extracted_solid,
            prune_small=active_policy.prune_small_components,
            min_component_faces=active_policy.min_component_faces,
            min_component_area_fraction=active_policy.min_component_area_fraction,
            min_component_volume_fraction=active_policy.min_component_volume_fraction,
        )
        pts, tris = oriented.vertices, oriented.triangles
        counters.flipped_face_count += oriented.flipped_face_count
        counters.cavity_component_count += oriented.cavity_component_count
        counters.pruned_component_count += oriented.pruned_component_count
        counters.pruned_face_count += oriented.pruned_face_count
    if tris.shape[0] == 0:
        raise SurfaceRepairError(
            "empty_output", "All surface components were pruned by size filters."
        )

    # Stage 6b: optional certified surface remesh & Hausdorff bound.
    pts, tris = _run_polish_stage(pts, tris, active_policy, counters)
    closure, embedding_verified = _verify_repaired_surface(
        pts, tris, active_policy, counters
    )

    evidence = SurfaceRepairEvidence(
        input_vertex_count=input_v,
        input_triangle_count=input_t,
        output_vertex_count=pts.shape[0],
        output_triangle_count=tris.shape[0],
        max_vertex_displacement=counters.max_vertex_displacement,
        welded_vertex_count=counters.welded_vertex_count,
        t_junction_splits=counters.t_junction_splits,
        collapsed_sliver_edges=counters.collapsed_sliver_edges,
        flipped_cap_edges=counters.flipped_cap_edges,
        dropped_degenerate_faces=counters.dropped_degenerate_faces,
        dropped_duplicate_faces=counters.dropped_duplicate_faces,
        rejected_fold_welds=counters.rejected_fold_welds,
        arrangement_fragments=counters.arrangement_fragments,
        canceled_coplanar_fragments=counters.canceled_coplanar_fragments,
        filled_hole_count=counters.filled_hole_count,
        skipped_large_holes=counters.skipped_large_holes,
        added_cap_triangles=counters.added_cap_triangles,
        filled_hole_area=counters.filled_hole_area,
        flipped_face_count=counters.flipped_face_count,
        cavity_component_count=counters.cavity_component_count,
        pruned_component_count=counters.pruned_component_count,
        pruned_face_count=counters.pruned_face_count,
        split_pinch_vertices=counters.split_pinch_vertices,
        used_generalized_winding=counters.used_generalized_winding,
        closure=closure,
        embedding_verified=embedding_verified,
        hausdorff_bound=counters.hausdorff_bound,
    )
    repaired_metadata = SurfaceMetadata(
        source_id=base_meta.source_id,
        source_revision=f"{base_meta.source_revision}:repair-{evidence.evidence_id[:16]}",
        coordinate_contract=base_meta.coordinate_contract,
        provenance=(
            *base_meta.provenance,
            f"surface-repair:{active_policy.policy_id[:16]}",
            f"surface-repair-evidence:{evidence.evidence_id[:16]}",
        ),
        cell_tags=(),
    )
    repaired_surface = SurfaceModel.from_triangles(
        pts,
        tris.astype(np.int32, copy=False),
        repaired_metadata,
        numeric_version=f"repair-{evidence.evidence_id[:16]}",
    )
    return SurfaceRepairResult(
        surface=repaired_surface,
        evidence=evidence,
        policy=active_policy,
    )
