#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Compile surface-meshing requests against the strata of a meshing domain.

`compile_surface_domain` resolves the scopes of one `SurfaceMeshingSpec`
(size controls, protected features) against the corner/curve/surface strata of
a `MeshingDomain` by exact identity: scope source, revision, entity set and
entity indices. It never infers identity from coordinates or names. The result
is the protected constraint complex (every corner and every boundary curve of
the selected patches, shared by all patches that use it), per-stratum size and
source-fidelity worksets, and the curve request under which the shared feature
curves are discretized once.
"""

from __future__ import annotations

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .._bvh import bvh_nearest_items, bvh_overlap_pair_blocks, PackedBVH, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import (
    charge_native_geometry_queries,
    MeshcoreStatus,
    point_triangle_locations,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..geometry._meshing_domain import (
    _cross_interval,
    MeshingDomain,
    MeshingDomainBoundarySource,
    PatchCurveUse,
    PatchPoleUse,
)
from ..geometry.brep._intersection_curve import (
    original_trim_intersection_preparation,
)
from ..geometry.brep._patches import PlanePatch, surface_differential
from ..geometry.brep._placed import PlacedSurface
from ..linalg import HermitianSpectrum, SmallLinearSolvePlan, solve_small_linear
from ._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    CurveEnd,
    CurveJunction,
    CurveMeshingSpec,
    MeshingFailure,
    MeshingFailureCategory,
    SurfaceMeshingSpec,
)
from ._controls import (
    BackgroundMetricControl,
    BackgroundMetricMode,
    FeatureKind,
    ProtectedFeature,
)
from ._metric import interpolate_mesh_metric
from ._scope import MeshingEntityKind, MeshingScope
from ._sizing import (
    CurvatureSizeControl,
    ProximitySizeControl,
    resolve_size_controls,
    SizeControl,
    SizeControlStrength,
    SizeFieldDomain,
    SourceProximityGapEvidence,
    UniformSizeControl,
)
from ._tetra_metric import _locate_stencil
from ._trace import MeshingStageKind


def _binds(domain: MeshingDomain, scope: MeshingScope, /) -> bool:
    return (
        scope.source_id == domain.source_id
        and scope.source_revision == domain.source_revision
        and scope.entity_kind is MeshingEntityKind.GEOMETRY
        and scope.entity_dimension in (0, 1, 2, 3)
        and scope.entity_set_id == domain.entity_set_id(scope.entity_dimension)
    )


def _resolved(domain: MeshingDomain, scope: MeshingScope, /) -> np.ndarray:
    identifiers = np.asarray(scope.global_entity_ids, dtype=np.int64)
    if np.setdiff1d(identifiers, domain.scope_indices(scope.entity_dimension)).size:
        raise MeshingFailure(
            MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
            "Scope names entities absent from the authoritative source stratum.",
            stage=MeshingStageKind.SOURCE_INSPECTION.value,
            provider_code="unknown_source_entity",
        )
    return domain.resolve_indices(scope.entity_dimension, identifiers)


def _feature_issues(
    domain: MeshingDomain, specification: SurfaceMeshingSpec, patches: np.ndarray, /
) -> list[str]:
    """Require protected source entities to survive the selected construction."""
    curves, corners = _selected_strata(domain, patches)
    selected = {0: corners, 1: curves, 2: patches}
    dimensions = {FeatureKind.CORNER: 0, FeatureKind.CURVE: 1, FeatureKind.SURFACE: 2}
    issues: list[str] = []
    for feature in specification.protected_features:
        scope = feature.scope
        dimension = dimensions.get(feature.feature_kind)
        if dimension is None or scope.entity_dimension != dimension:
            issues.append(
                "protected feature kinds matching their source stratum dimensions"
            )
            continue
        if not _binds(domain, scope):
            issues.append("protected features bound to authoritative source strata")
            continue
        identifiers = np.asarray(scope.global_entity_ids, dtype=np.int64)
        if np.setdiff1d(identifiers, domain.scope_indices(dimension)).size:
            issues.append("protected features that exist in the meshing domain")
            continue
        if (
            feature.hard
            and np.setdiff1d(_resolved(domain, scope), selected[dimension]).size
        ):
            issues.append(
                "hard protected features contained in the selected surface closure"
            )
    return issues


def surface_domain_issues(
    domain: MeshingDomain, specification: SurfaceMeshingSpec, /
) -> list[str]:
    """Requests of one surface specification the compiled domain cannot resolve."""

    unsupported: list[str] = []
    scope = specification.scope
    if scope.entity_dimension != 2 or not _binds(domain, scope):
        unsupported.append("a surface scope over the meshing domain's surface strata")
    elif np.setdiff1d(
        np.asarray(scope.global_entity_ids, dtype=np.int64), domain.scope_indices(2)
    ).size:
        unsupported.append("surfaces that exist in the meshing domain")
    else:
        unsupported.extend(
            _feature_issues(domain, specification, _resolved(domain, scope))
        )
    for control in specification.size_controls:
        scopes = (
            (control.source_scope, control.target_scope)
            if isinstance(control, ProximitySizeControl)
            else (control.scope,)
        )
        if any(not _binds(domain, bound) for bound in scopes):
            unsupported.append("size controls bound to authoritative source strata")
        elif any(
            np.setdiff1d(
                np.asarray(bound.global_entity_ids, dtype=np.int64),
                domain.scope_indices(bound.entity_dimension),
            ).size
            for bound in scopes
        ):
            unsupported.append("size controls on existing authoritative source entities")
        if isinstance(control, UniformSizeControl):
            if control.scope.entity_dimension not in (1, 2, 3):
                unsupported.append("uniform sizing on curves, patches or regions")
        elif any(bound.entity_dimension != 2 for bound in scopes):
            unsupported.append("curvature/proximity sizing on source patches")
    for feature in specification.protected_features:
        if feature.feature_kind is FeatureKind.MATERIAL_INTERFACE:
            unsupported.append("material-interface protected features")
        elif not _binds(domain, feature.scope):
            unsupported.append("protected features on the domain's strata")
    for control in specification.region_controls:
        if control.scope.entity_dimension not in (2, 3) or not _binds(
            domain, control.scope
        ):
            unsupported.append("region controls bound to authoritative source regions")
        elif np.setdiff1d(
            np.asarray(control.scope.global_entity_ids, dtype=np.int64),
            domain.scope_indices(control.scope.entity_dimension),
        ).size:
            unsupported.append(
                "region controls on existing authoritative source entities"
            )
    for control in specification.patch_controls:
        if control.scope.entity_dimension not in (1, 2) or not _binds(
            domain, control.scope
        ):
            unsupported.append("patch controls bound to authoritative source patches")
        elif np.setdiff1d(
            np.asarray(control.scope.global_entity_ids, dtype=np.int64),
            domain.scope_indices(control.scope.entity_dimension),
        ).size:
            unsupported.append("patch controls on existing authoritative source entities")
    if specification.periodic_constraints:
        unsupported.append("periodic constraints")
    if specification.layer_controls:
        unsupported.append("boundary-layer controls")
    return unsupported


def _surface_scope(domain: MeshingDomain, patches: np.ndarray, /) -> MeshingScope:
    identifiers = np.asarray(domain.scope_indices(2), dtype=np.int64)[patches]
    return MeshingScope(
        domain.source_id,
        domain.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        identifiers,
    )


def _surface_controls(
    domain: MeshingDomain, specification: SurfaceMeshingSpec, /
) -> tuple[SizeControl, ...]:
    controls: list[SizeControl] = []
    for control in specification.size_controls:
        scopes = (
            (control.source_scope, control.target_scope)
            if isinstance(control, ProximitySizeControl)
            else (control.scope,)
        )
        for scope in scopes:
            if not _binds(domain, scope):
                raise _stale(domain, scope)
            _resolved(domain, scope)
        if (
            isinstance(control, UniformSizeControl)
            and control.scope.entity_dimension == 3
        ):
            patches = np.unique(
                np.asarray(
                    [
                        patch
                        for region in _resolved(domain, control.scope).tolist()
                        for patch, _ in domain.regions[region].boundary
                    ],
                    dtype=np.int64,
                )
            )
            controls.append(
                UniformSizeControl(
                    _surface_scope(domain, patches),
                    control.target_size,
                    minimum_size=control.minimum_size,
                    maximum_size=control.maximum_size,
                    maximum_growth_rate=control.maximum_growth_rate,
                    strength=control.strength,
                    priority=control.priority,
                )
            )
        elif all(scope.entity_dimension == 2 for scope in scopes):
            controls.append(control)
    return tuple(controls)


def _curvature_samples(
    domain: MeshingDomain, patches: np.ndarray, charts: np.ndarray, /
) -> np.ndarray:
    """Principal source curvature with native Gram whitening and solve evidence."""
    result = np.empty((patches.size,), dtype=np.float64)
    for patch in np.unique(patches).tolist():
        selected = patches == patch
        surface = domain.patches[patch].surface
        parameters = jnp.asarray(charts[selected], dtype=jnp.float64)
        charge_native_geometry_queries(parameters.shape[0])
        first = surface_differential(surface, parameters)
        charge_native_geometry_queries(parameters.shape[0])
        second = jax.vmap(jax.jacfwd(jax.jacfwd(surface.evaluate)))(parameters)
        normal = jnp.cross(first[..., :, 0], first[..., :, 1])
        normal /= jnp.linalg.norm(normal, axis=-1, keepdims=True)
        gram = HermitianSpectrum(jnp.swapaxes(first, -1, -2) @ first)
        form = HermitianSpectrum(jnp.sum(second * normal[..., :, None, None], axis=-3))
        valid = np.asarray(gram.valid & form.valid & (gram.minimum_eigenvalue > 0))
        if not np.all(valid):
            raise ValueError("Curvature sizing needs regular source samples.")
        root = (
            gram.eigenvectors * jnp.sqrt(gram.eigenvalues)[..., None, :]
        ) @ jnp.swapaxes(gram.eigenvectors, -1, -2)
        plan = SmallLinearSolvePlan(2)
        left = solve_small_linear(plan, root, form.matrix)
        right = solve_small_linear(plan, root, jnp.swapaxes(left.value, -1, -2))
        if not np.all(np.asarray(left.successful & right.successful)):
            raise ValueError("Source curvature Gram whitening did not converge.")
        shape = right.value
        spectrum = HermitianSpectrum(0.5 * (shape + jnp.swapaxes(shape, -1, -2)))
        if not np.all(np.asarray(spectrum.valid)):
            raise ValueError("Source principal-curvature spectrum is unresolved.")
        result[selected] = np.asarray(jnp.max(jnp.abs(spectrum.eigenvalues), axis=-1))
    return result


@final
class SourceSurfaceMetric(StrictModule, NonTrainableState):
    """One exact background revision and prepared native affine simplex locator."""

    control: BackgroundMetricControl
    tree: PackedBVH
    values: np.ndarray
    vertex_ids: np.ndarray
    cells: np.ndarray
    cell_ids: np.ndarray
    corners: np.ndarray
    maximum_pairs: int = eqx.field(static=True)
    minimum_directional_size: float = eqx.field(static=True)
    maximum_directional_size: float = eqx.field(static=True)

    def __init__(
        self, control: BackgroundMetricControl, maximum_pairs: int, scratch: int, /
    ) -> None:
        mesh = control.mesh
        if mesh.topological_dimension not in (2, 3) or mesh.ambient_dimension != 3:
            raise ValueError(
                "Surface background metrics require an ambient-3 affine simplex cover."
            )
        count = sum(block.vertices.shape[0] for block in mesh.blocks)
        required = 4096 * count + 1024 * mesh.coordinates.shape[0]
        if required > scratch:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "Prepared source metric locator exceeds scratch capacity.",
                stage=MeshingStageKind.SOURCE_INSPECTION.value,
                requested=(("maximum_scratch_bytes", scratch),),
                achieved=(("scratch_bytes", required),),
            )
        cells = np.concatenate(
            [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
        )
        corners = np.asarray(mesh.coordinates, dtype=np.float64)[cells]
        values = control.vertex_metrics
        evidence = HermitianSpectrum(jnp.asarray(values, dtype=jnp.float64))
        if not np.all(np.asarray(evidence.valid & (evidence.minimum_eigenvalue > 0))):
            raise ValueError("Prepared source metrics need native SPD evidence.")
        eigenvalues = np.asarray(evidence.eigenvalues)
        match control.mode:
            case BackgroundMetricMode.ANISOTROPIC:
                pass
            case BackgroundMetricMode.ISOTROPIC:
                values = (
                    eigenvalues[:, -1, None, None] * np.eye(3, dtype=np.float64)[None]
                )
            case _:
                raise ValueError("Unknown source background metric mode.")
        self.control = control
        self.tree = prepare_bvh(
            np.min(corners, axis=1), np.max(corners, axis=1), dtype=jnp.float64
        )
        self.values = values
        self.vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        self.cells = cells
        self.corners = corners
        self.cell_ids = np.concatenate(
            [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
        )
        self.maximum_pairs = maximum_pairs
        self.minimum_directional_size = float(1 / np.sqrt(np.max(eigenvalues)))
        self.maximum_directional_size = float(1 / np.sqrt(np.min(eigenvalues)))

    def sample(self, points: np.ndarray, maximum_pairs: int, /) -> tuple[np.ndarray, int]:
        if points.shape[0] == 0:
            return np.empty((0, 3, 3), dtype=np.float64), 0
        work: list[int] = [0]
        if self.control.mesh.topological_dimension == 3:
            identifiers, weights, valid = _locate_stencil(
                self.control.mesh,
                points,
                min(maximum_pairs, self.maximum_pairs),
                prepared_bvh=self.tree,
                work_counter=work,
            )
        else:
            identifiers, weights, valid = self._triangle_stencil(
                points, min(maximum_pairs, self.maximum_pairs), work
            )
        if not np.all(valid):
            raise ValueError(
                "Source metric query lacks complete native source-cell coverage."
            )
        order = np.argsort(self.vertex_ids)
        rows = order[np.searchsorted(self.vertex_ids[order], identifiers)]
        values = np.asarray(
            interpolate_mesh_metric(self.values[rows], weights), dtype=np.float64
        )
        return values, sum(work)

    def _triangle_stencil(
        self, points: np.ndarray, maximum_pairs: int, work: list[int], /
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        queries = prepare_bvh(points, points, dtype=jnp.float64)
        selected = np.full((points.shape[0],), -1, dtype=np.int64)
        selected_ids = np.full((points.shape[0],), np.iinfo(np.int64).max, dtype=np.int64)
        weights = np.zeros((points.shape[0], 3), dtype=np.float64)
        for cells, query in bvh_overlap_pair_blocks(
            self.tree, queries, include_touching=True
        ):
            work[0] += cells.size
            if work[0] > maximum_pairs:
                raise MeshingFailure(
                    MeshingFailureCategory.RESOURCE_EXHAUSTED,
                    "Source surface metric point location exceeds its pair budget.",
                    stage=MeshingStageKind.SURFACE_MESHING.value,
                    requested=(("maximum_location_pairs", maximum_pairs),),
                    achieved=(("location_pairs", work[0]),),
                )
            corners = self.corners[cells]
            target = points[query]
            side, feature, status = point_triangle_locations(target, corners)
            basis = corners[:, 1:] - corners[:, :1]
            gram = basis @ np.swapaxes(basis, -1, -2)
            right = (basis @ (target - corners[:, 0])[..., None])[..., 0]
            solve = solve_small_linear(SmallLinearSolvePlan(2), gram, right)
            reference = np.asarray(solve.value, dtype=np.float64)
            coefficients = np.concatenate(
                (1 - np.sum(reference, axis=1, keepdims=True), reference), axis=1
            )
            valid = (
                (side == 0)
                & (feature >= 0)
                & (status == MeshcoreStatus.OK)
                & np.asarray(solve.successful)
            )
            for slot in range(3):
                coefficients[feature == slot + 3, slot] = 0
                vertices = feature == slot
                coefficients[vertices] = 0
                coefficients[vertices, slot] = 1
            accepted = np.flatnonzero(
                valid & (self.cell_ids[cells] < selected_ids[query])
            )
            ordered = accepted[
                np.lexsort((self.cell_ids[cells[accepted]], query[accepted]))
            ]
            first = (
                np.concatenate(
                    (np.asarray((True,), dtype=np.bool_), np.diff(query[ordered]) != 0)
                )
                if ordered.size
                else np.empty((0,), dtype=np.bool_)
            )
            chosen = ordered[first]
            selected[query[chosen]] = cells[chosen]
            selected_ids[query[chosen]] = self.cell_ids[cells[chosen]]
            weights[query[chosen]] = coefficients[chosen]
        covered = selected >= 0
        if not np.all(covered):
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "Source background triangles do not exactly cover the requested physical points.",
                stage=MeshingStageKind.SURFACE_MESHING.value,
                achieved=(("uncovered_points", int(np.sum(~covered))),),
            )
        weights = np.maximum(weights, 0)
        weights /= np.sum(weights, axis=1, keepdims=True)
        identifiers = self.vertex_ids[self.cells[selected]]
        return identifiers, weights, np.broadcast_to(covered[:, None], weights.shape)


@final
class SourcePatchNeighborhood(StrictModule, NonTrainableState):
    """Prepared source interval cover; boxes are only outward source enclosures."""

    tree: PackedBVH
    plane_point_bounds: np.ndarray
    normal_lower: np.ndarray
    normal_upper: np.ndarray
    patch: int = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    planar: bool = eqx.field(static=True)
    geometry_queries: int = eqx.field(static=True)

    def __init__(self, domain: MeshingDomain, patch: int, /) -> None:
        surface = domain.patches[patch].surface
        bounds = []
        curve_bound_queries = 0
        for loop in domain.patches[patch].loops:
            for use in loop:
                match use:
                    case PatchCurveUse():
                        first, last = sorted((use.first, use.last))
                        charge_native_geometry_queries(1)
                        curve_bound_queries += 1
                        bounds.append(use.pcurve.bounding_box(first, last))
                    case PatchPoleUse():
                        coordinates = np.asarray((use.start, use.end), dtype=np.float64)
                        bounds.append(
                            np.stack(
                                (np.min(coordinates, axis=0), np.max(coordinates, axis=0))
                            )
                        )
        box = np.stack(
            (
                np.min(np.asarray(bounds)[:, 0], axis=0),
                np.max(np.asarray(bounds)[:, 1], axis=0),
            )
        )
        axes = [np.linspace(box[0, axis], box[1, axis], 9) for axis in range(2)]
        boxes = np.asarray(
            [
                [
                    [axes[0][first], axes[1][second]],
                    [axes[0][first + 1], axes[1][second + 1]],
                ]
                for first in range(8)
                for second in range(8)
            ],
            dtype=np.float64,
        )
        charge_native_geometry_queries(boxes.shape[0])
        spatial = np.asarray([surface.bounding_box(cell) for cell in boxes])
        normal_boxes = []
        for cell in boxes:
            charge_native_geometry_queries(1)
            low, high = surface.derivative_bounds(cell, order=1)
            normal_low, normal_high = _cross_interval(
                low[:, 0], high[:, 0], low[:, 1], high[:, 1]
            )
            normal_boxes.append(
                (normal_low, normal_high)
                if domain.patches[patch].orientation > 0
                else (-normal_high, -normal_low)
            )
        centers = np.mean(box, axis=0)[None]
        normals, regular = domain.oriented_normals(
            np.asarray((patch,), dtype=np.int64), centers
        )
        definition = surface.definition if isinstance(surface, PlacedSurface) else surface
        planar = isinstance(definition, PlanePatch)
        if planar and not regular[0]:
            raise ValueError("A proximity source plane must be regular.")
        self.tree = prepare_bvh(spatial[:, 0], spatial[:, 1], dtype=jnp.float64)
        charge_native_geometry_queries(1)
        self.plane_point_bounds = surface.bounding_box(np.concatenate((centers, centers)))
        self.normal_lower = np.min(np.asarray(normal_boxes)[:, 0], axis=0)
        self.normal_upper = np.max(np.asarray(normal_boxes)[:, 1], axis=0)
        self.patch, self.domain_id, self.planar = patch, domain.domain_id, planar
        self.geometry_queries = 130 + curve_bound_queries

    def lower_gaps(
        self, points: np.ndarray, normals: np.ndarray, opposite_only: bool, /
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.planar:
            offset_low = np.nextafter(points - self.plane_point_bounds[1], -np.inf)
            offset_high = np.nextafter(points - self.plane_point_bounds[0], np.inf)
            products = np.stack(
                (
                    offset_low * self.normal_lower,
                    offset_low * self.normal_upper,
                    offset_high * self.normal_lower,
                    offset_high * self.normal_upper,
                )
            )
            roundoff = (
                64
                * np.finfo(np.float64).eps
                * np.sum(np.max(np.abs(products), axis=0), axis=1)
            )
            signed_low = np.sum(np.min(products, axis=0), axis=1) - roundoff
            signed_high = np.sum(np.max(products, axis=0), axis=1) + roundoff
            separation = np.maximum(np.maximum(signed_low, -signed_high), 0)
            normal_upper = np.linalg.norm(
                np.maximum(np.abs(self.normal_lower), np.abs(self.normal_upper))
            ) * (1 + 64 * np.finfo(np.float64).eps)
            lower = separation / normal_upper
            products_low = normals * self.normal_lower
            products_high = normals * self.normal_upper
            upper = np.sum(np.maximum(products_low, products_high), axis=1)
            bottom = np.sum(np.minimum(products_low, products_high), axis=1)
            facing = upper < 0
            complete = (
                (upper < 0) | (bottom >= 0) if opposite_only else np.ones_like(facing)
            )
        else:
            result = bvh_nearest_items(self.tree, points, query_batch_capacity=64)
            lower = np.sqrt(np.asarray(result.distance_squared)[:, 0])
            first = normals * self.normal_lower
            second = normals * self.normal_upper
            high = np.sum(np.maximum(first, second), axis=1)
            low = np.sum(np.minimum(first, second), axis=1)
            facing = high < 0
            complete = (high < 0) | (low >= 0) if opposite_only else np.ones_like(facing)
        lower = np.maximum(
            np.nextafter(lower * (1 - 128 * np.finfo(np.float64).eps), 0), 0
        )
        if opposite_only:
            lower = np.where(facing, lower, np.inf)
        complete &= lower > 0
        return lower, complete


def _source_gap_evidence(
    domain: MeshingDomain,
    controls: tuple[SizeControl, ...],
    neighborhoods: tuple[SourcePatchNeighborhood, ...],
    identifiers: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    /,
) -> dict[str, SourceProximityGapEvidence]:
    evidence = {}
    lookup = {plan.patch: plan for plan in neighborhoods}
    for control in controls:
        if not isinstance(control, ProximitySizeControl):
            continue
        source = np.asarray(control.source_scope.global_entity_ids, dtype=np.int64)
        target = np.asarray(control.target_scope.global_entity_ids, dtype=np.int64)
        source_only, target_only = (
            np.setdiff1d(source, target),
            np.setdiff1d(target, source),
        )
        if not source_only.size or not target_only.size:
            raise ValueError(
                "Proximity source scopes must contain distinct authoritative patches."
            )
        lower = np.full((points.shape[0],), np.inf)
        complete = np.ones((points.shape[0],), dtype=np.bool_)
        for query_ids, target_ids in ((source, target_only), (target, source_only)):
            rows = np.flatnonzero(np.isin(identifiers, query_ids))
            for patch in domain.resolve_indices(2, target_ids).tolist():
                plan = lookup[patch]
                if plan.domain_id != domain.domain_id:
                    raise ValueError(
                        "Prepared proximity neighborhood binds another source revision."
                    )
                gap, decided = plan.lower_gaps(
                    points[rows], normals[rows], control.opposite_normals_only
                )
                lower[rows] = np.minimum(lower[rows], gap)
                complete[rows] &= decided
        evidence[control.control_id] = SourceProximityGapEvidence(
            control, identifiers, points, lower, complete
        )
    return evidence


@final
class SourceSurfaceSizing(StrictModule, NonTrainableState):
    """Revision-bound source samples for the canonical sizing resolver."""

    controls: tuple[SizeControl, ...]
    patches: np.ndarray
    charts: np.ndarray
    points: np.ndarray
    normals: np.ndarray
    curvature: np.ndarray
    sizes: np.ndarray
    geometry_queries: int = eqx.field(static=True)
    metric: SourceSurfaceMetric | None
    neighborhoods: tuple[SourcePatchNeighborhood, ...]

    def __init__(
        self, domain: MeshingDomain, specification: SurfaceMeshingSpec, /
    ) -> None:
        controls = _surface_controls(domain, specification)
        background = specification.background_metric
        metric = (
            None
            if background is None
            else SourceSurfaceMetric(
                background,
                specification.limits.maximum_work_units,
                specification.limits.maximum_scratch_bytes,
            )
        )
        if not controls and metric is not None:
            controls = (
                UniformSizeControl(
                    specification.scope,
                    metric.maximum_directional_size,
                    strength=SizeControlStrength.SOFT,
                ),
            )
        selected = (
            np.unique(
                np.concatenate(
                    [
                        _resolved(domain, scope)
                        for control in controls
                        for scope in (
                            (control.source_scope, control.target_scope)
                            if isinstance(control, ProximitySizeControl)
                            else (control.scope,)
                        )
                    ]
                )
            )
            if controls
            else np.empty((0,), dtype=np.int64)
        )
        if not selected.size:
            raise ValueError("Every meshed surface needs a source size control.")
        boundary_source = MeshingDomainBoundarySource(
            domain,
            tuple(selected.tolist()),
            resolution=7,
        )
        sample_capacity = boundary_source._sampling_capacity()
        boundary_intervals = max(3, boundary_source.resolution - 1)
        scratch = sample_capacity * 2048 + sum(
            len(loop) * boundary_intervals * 64
            for patch in selected.tolist()
            for loop in domain.patches[patch].loops
        )
        limits = specification.limits
        proximity_patches = np.unique(
            np.concatenate(
                [
                    _resolved(domain, scope)
                    for control in controls
                    if isinstance(control, ProximitySizeControl)
                    for scope in (control.source_scope, control.target_scope)
                ]
                + [np.empty((0,), dtype=np.int64)]
            )
        )
        scratch += proximity_patches.size * 64 * 4096
        queries = (
            sample_capacity * 5
            + proximity_patches.size * 130
            + sum(
                len(loop)
                for patch in proximity_patches.tolist()
                for loop in domain.patches[patch].loops
            )
        )
        if (
            scratch > limits.maximum_scratch_bytes
            or queries > limits.maximum_geometry_queries
        ):
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "Source sizing preparation exceeds its scratch or query capacity.",
                stage=MeshingStageKind.SOURCE_INSPECTION.value,
                requested=(
                    ("maximum_scratch_bytes", limits.maximum_scratch_bytes),
                    ("maximum_geometry_queries", limits.maximum_geometry_queries),
                ),
                achieved=(("scratch_bytes", scratch), ("geometry_queries", queries)),
            )
        neighborhoods = tuple(
            SourcePatchNeighborhood(domain, patch) for patch in proximity_patches.tolist()
        )
        # Candidates include authored boundary chains; interior points are
        # retained only after exact classification against the original trims.
        trim_curves = tuple(
            trim
            for patch in selected.tolist()
            for loop in domain.patches[patch].loops
            for use in loop
            for trim in (use.trim_curve,)
            if trim is not None
        )
        with original_trim_intersection_preparation(trim_curves):
            patches, charts, points, _, _ = boundary_source._grid()
        normals, regular = domain.oriented_normals(patches, charts)
        evaluated_samples = patches.size
        patches, charts, points, normals = (
            patches[regular],
            charts[regular],
            points[regular],
            normals[regular],
        )
        curvature = (
            _curvature_samples(domain, patches, charts)
            if any(isinstance(control, CurvatureSizeControl) for control in controls)
            else np.zeros((patches.size,), dtype=np.float64)
        )
        identifiers = np.asarray(domain.scope_indices(2), dtype=np.int64)[patches]
        field, _ = resolve_size_controls(
            controls,
            points,
            identifiers,
            SizeFieldDomain.SURFACE_GEODESIC,
            curvature=curvature,
            normals=normals,
            proximity_gaps=_source_gap_evidence(
                domain, controls, neighborhoods, identifiers, points, normals
            ),
        )
        self.controls = controls
        self.patches, self.charts, self.points = patches, charts, points
        self.normals, self.curvature = normals, curvature
        self.sizes = np.asarray(field.values, dtype=np.float64)
        self.geometry_queries = (
            sample_capacity
            + evaluated_samples
            + (
                2 * patches.size
                if any(isinstance(control, CurvatureSizeControl) for control in controls)
                else 0
            )
        )
        self.geometry_queries += sum(plan.geometry_queries for plan in neighborhoods)
        self.metric = metric
        self.neighborhoods = neighborhoods

    def evaluate(
        self, domain: MeshingDomain, patch: int, charts: np.ndarray, points: np.ndarray, /
    ) -> np.ndarray:
        count = points.shape[0]
        if count == 0:
            return np.empty((0,), dtype=np.float64)
        if all(isinstance(control, UniformSizeControl) for control in self.controls):
            return np.full(
                (count,), np.min(self.sizes[self.patches == patch]), dtype=np.float64
            )
        patches = np.concatenate((self.patches, np.full((count,), patch, dtype=np.int64)))
        identifiers = np.asarray(domain.scope_indices(2), dtype=np.int64)[patches]
        normals, regular = domain.oriented_normals(np.full((count,), patch), charts)
        if not np.all(regular):
            raise ValueError("Local source sizing requires regular chart points.")
        curvature = (
            _curvature_samples(domain, np.full((count,), patch), charts)
            if any(isinstance(control, CurvatureSizeControl) for control in self.controls)
            else np.zeros((count,), dtype=np.float64)
        )
        field, _ = resolve_size_controls(
            self.controls,
            np.concatenate((self.points, points)),
            identifiers,
            SizeFieldDomain.SURFACE_GEODESIC,
            curvature=np.concatenate((self.curvature, curvature)),
            normals=np.concatenate((self.normals, normals)),
            proximity_gaps=_source_gap_evidence(
                domain,
                self.controls,
                self.neighborhoods,
                identifiers,
                np.concatenate((self.points, points)),
                np.concatenate((self.normals, normals)),
            ),
        )
        return np.asarray(field.values, dtype=np.float64)[-count:]


@final
class CompiledSurfaceDomain(StrictModule, NonTrainableState):
    """Protected constraints and size/fidelity worksets of one surface request.

    ``patches``, ``curves`` and ``corners`` are the selected surfaces and the
    curves/corners bounding them, sorted. ``patch_sizes``/``curve_sizes`` are
    the hard local edge-length bounds (the smallest applicable target or
    maximum size); ``patch_deviations``/``curve_deviations`` the source
    fidelity bounds (infinite when unrequested). ``curve_request`` discretizes
    the shared curves once for every patch that uses them.
    """

    domain: MeshingDomain
    patches: np.ndarray
    curves: np.ndarray
    corners: np.ndarray
    patch_sizes: np.ndarray
    curve_sizes: np.ndarray
    patch_deviations: np.ndarray
    curve_deviations: np.ndarray
    curve_request: CurveMeshingSpec
    source_sizing: SourceSurfaceSizing
    patch_normal_angles: np.ndarray
    specification_id: str = eqx.field(static=True)
    compiled_id: str = eqx.field(static=True)

    def __init__(
        self,
        domain: MeshingDomain,
        patches: np.ndarray,
        curves: np.ndarray,
        corners: np.ndarray,
        patch_sizes: np.ndarray,
        curve_sizes: np.ndarray,
        patch_deviations: np.ndarray,
        curve_deviations: np.ndarray,
        curve_request: CurveMeshingSpec,
        source_sizing: SourceSurfaceSizing,
        patch_normal_angles: np.ndarray,
        specification_id: str,
        /,
    ) -> None:
        if not isinstance(domain, MeshingDomain):
            raise TypeError("domain must be MeshingDomain.")
        if not isinstance(curve_request, CurveMeshingSpec):
            raise TypeError("curve_request must be CurveMeshingSpec.")
        self.domain = domain
        self.patches = np.asarray(patches, dtype=np.int64)
        self.curves = np.asarray(curves, dtype=np.int64)
        self.corners = np.asarray(corners, dtype=np.int64)
        self.patch_sizes = np.asarray(patch_sizes, dtype=np.float64)
        self.curve_sizes = np.asarray(curve_sizes, dtype=np.float64)
        self.patch_deviations = np.asarray(patch_deviations, dtype=np.float64)
        self.curve_deviations = np.asarray(curve_deviations, dtype=np.float64)
        self.curve_request = curve_request
        self.source_sizing = source_sizing
        self.patch_normal_angles = np.asarray(patch_normal_angles, dtype=np.float64)
        self.specification_id = specification_id
        self.compiled_id = canonical_fingerprint(
            {
                "kind": "compiled-surface-domain",
                "domain": domain.domain_id,
                "specification": specification_id,
                "curve_request": curve_request.specification_id,
                "background_metric": None
                if source_sizing.metric is None
                else source_sizing.metric.control.control_id,
                "source_controls": tuple(
                    control.control_id for control in source_sizing.controls
                ),
                "worksets": array_tree_fingerprint(
                    (
                        self.patches,
                        self.curves,
                        self.corners,
                        self.patch_sizes,
                        self.curve_sizes,
                        self.patch_deviations,
                        self.curve_deviations,
                    )
                ),
            }
        )


def _stale(domain: MeshingDomain, scope: MeshingScope, /) -> MeshingFailure:
    return MeshingFailure(
        MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
        f"Scope {scope.scope_id} does not bind meshing domain {domain.source_id!r} "
        f"at revision {domain.source_revision!r}.",
        stage=MeshingStageKind.SOURCE_INSPECTION.value,
        provider_code="stale_scope",
    )


def _selected_strata(
    domain: MeshingDomain, patches: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    curves = np.unique(
        np.concatenate(
            [domain.patch_curves(int(patch)) for patch in patches.tolist()]
            + [np.zeros((0,), dtype=np.int64)]
        )
    )
    ends = [
        corner
        for curve in curves.tolist()
        for corner in (domain.curves[curve].start, domain.curves[curve].end)
    ]
    poles = [
        use.corner
        for patch in patches.tolist()
        for loop in domain.patches[patch].loops
        for use in loop
        if isinstance(use, PatchPoleUse)
    ]
    return curves, np.unique(np.asarray(ends + poles, dtype=np.int64))


def _size_worksets(
    domain: MeshingDomain,
    specification: SurfaceMeshingSpec,
    patches: np.ndarray,
    curves: np.ndarray,
    sizing: SourceSurfaceSizing,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    patch_sizes = np.full((len(domain.patches),), np.inf)
    curve_sizes = np.full((len(domain.curves),), np.inf)
    for patch in patches.tolist():
        values = sizing.sizes[sizing.patches == patch]
        if not values.size:
            raise ValueError("Every selected patch needs resolved source sizing.")
        patch_sizes[patch] = np.max(values)
        bounded = domain.patch_curves(patch)
        curve_sizes[bounded] = np.minimum(curve_sizes[bounded], np.min(values))
        if sizing.metric is not None:
            curve_sizes[bounded] = np.minimum(
                curve_sizes[bounded],
                sizing.metric.minimum_directional_size
                * (sizing.metric.control.maximum_metric_edge_length or 1.0)
                * (1 - 256 * np.finfo(np.float64).eps),
            )
    for control in specification.size_controls:
        if (
            isinstance(control, UniformSizeControl)
            and control.scope.entity_dimension == 1
        ):
            if not _binds(domain, control.scope):
                raise _stale(domain, control.scope)
            entities = _resolved(domain, control.scope)
            curve_sizes[entities] = np.minimum(curve_sizes[entities], control.target_size)
    for control in specification.size_controls:
        if (
            isinstance(control, UniformSizeControl)
            and control.strength is SizeControlStrength.HARD
        ):
            if control.minimum_size is None:
                continue
            entities = _resolved(domain, control.scope)
            affected = (
                entities
                if control.scope.entity_dimension == 1
                else np.unique(
                    np.concatenate(
                        [
                            domain.patch_curves(patch)
                            for patch in (
                                entities.tolist()
                                if control.scope.entity_dimension == 2
                                else [
                                    patch
                                    for region in entities.tolist()
                                    for patch, _ in domain.regions[region].boundary
                                ]
                            )
                        ]
                    )
                )
            )
            if np.any(curve_sizes[affected] < control.minimum_size):
                raise ValueError(
                    "Shared source curves contradict an active hard minimum size."
                )
    return patch_sizes[patches], curve_sizes[curves]


def _semantic_controls(
    domain: MeshingDomain, specification: SurfaceMeshingSpec, patches: np.ndarray, /
) -> None:
    names = {region: source.name for region, source in enumerate(domain.regions)}
    bindings: dict[int, str] = {}
    surface_bindings: dict[int, str] = {}
    surface_names: dict[int, str] = {}
    for control in specification.region_controls:
        if control.scope.entity_dimension not in (2, 3) or not _binds(
            domain, control.scope
        ):
            raise _stale(domain, control.scope)
        if control.scope.entity_dimension == 2:
            for patch in _resolved(domain, control.scope).tolist():
                if (
                    patch in surface_bindings
                    and surface_bindings[patch] != control.control_id
                ):
                    raise ValueError("Exclusive surface-region controls overlap.")
                surface_bindings[patch] = control.control_id
                surface_names[patch] = control.region_name
                if not control.meshing_enabled and patch in patches:
                    raise ValueError("A selected surface region is disabled for meshing.")
            continue
        for region in _resolved(domain, control.scope).tolist():
            if region in bindings and bindings[region] != control.control_id:
                raise ValueError(
                    "One authoritative region cannot carry contradictory controls."
                )
            bindings[region] = control.control_id
            names[region] = control.region_name
            if not control.meshing_enabled and any(
                patch in patches for patch, _ in domain.regions[region].boundary
            ):
                raise ValueError("Selected surfaces bound a region disabled for meshing.")
    for control in specification.patch_controls:
        if control.scope.entity_dimension not in (1, 2) or not _binds(
            domain, control.scope
        ):
            raise _stale(domain, control.scope)
        selected = _resolved(domain, control.scope)
        if control.scope.entity_dimension == 1:
            selected_curves, _ = _selected_strata(domain, patches)
            if control.required and np.setdiff1d(selected, selected_curves).size:
                raise ValueError(
                    "A required curve patch lies outside the selected surfaces."
                )
            for curve in selected.tolist():
                adjacent_patches = {
                    patch for patch, _ in domain.curve_uses(curve) if patch in patches
                }
                if not adjacent_patches.issubset(surface_names):
                    raise ValueError(
                        "Curve patch controls require authored adjacent surface-region controls."
                    )
                adjacent = tuple(
                    sorted({surface_names[patch] for patch in adjacent_patches})
                )
                if adjacent != control.adjacent_region_names:
                    raise ValueError(
                        "A curve patch contradicts authoritative adjacent surface regions."
                    )
            continue
        if control.required and np.setdiff1d(selected, patches).size:
            raise ValueError(
                "A required source patch control lies outside the surface selection."
            )
        for patch in selected.tolist():
            adjacent = tuple(
                sorted(
                    names[int(region)]
                    for region in domain.patch_regions[patch]
                    if region >= 0
                )
            )
            if adjacent != control.adjacent_region_names:
                raise ValueError(
                    "Patch controls must match authoritative source-region incidence."
                )


def _deviation_worksets(
    domain: MeshingDomain,
    specification: SurfaceMeshingSpec,
    patches: np.ndarray,
    curves: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    patch_bounds = np.full((len(domain.patches),), np.inf)
    curve_bounds = np.full((len(domain.curves),), np.inf)
    for feature in specification.protected_features:
        if not _binds(domain, feature.scope):
            raise _stale(domain, feature.scope)
        entities = _resolved(domain, feature.scope)
        bound = feature.maximum_deviation
        match feature.scope.entity_dimension:
            case 2:
                patch_bounds[entities] = np.minimum(patch_bounds[entities], bound)
            case 1:
                curve_bounds[entities] = np.minimum(curve_bounds[entities], bound)
            case 0:
                # Corners are exact mesh vertices of every construction.
                pass
            case _:
                raise ValueError("Protected features scope corners, curves or surfaces.")
    for patch in patches.tolist():
        bounded = domain.patch_curves(patch)
        curve_bounds[bounded] = np.minimum(curve_bounds[bounded], patch_bounds[patch])
    return patch_bounds[patches], curve_bounds[curves]


def _curve_request(
    domain: MeshingDomain,
    specification: SurfaceMeshingSpec,
    curves: np.ndarray,
    curve_sizes: np.ndarray,
    curve_deviations: np.ndarray,
    /,
) -> CurveMeshingSpec:
    """Curve request whose junctions are the domain corners joining curve ends."""

    def scope(selected: np.ndarray, /) -> MeshingScope:
        return MeshingScope(
            domain.source_id,
            domain.source_revision,
            MeshingEntityKind.GEOMETRY,
            1,
            domain.entity_set_id(1),
            np.asarray(selected, dtype=np.int64),
        )

    controls = tuple(
        UniformSizeControl(
            scope(curves[curve_sizes == size]),
            float(size),
            strength=SizeControlStrength.SOFT,
        )
        for size in np.unique(curve_sizes).tolist()
    )
    features = tuple(
        ProtectedFeature(
            scope(curves[curve_deviations == bound]),
            FeatureKind.CURVE,
            maximum_deviation=float(bound),
        )
        for bound in np.unique(curve_deviations[np.isfinite(curve_deviations)]).tolist()
    )
    ends: dict[int, list[tuple[int, CurveEnd]]] = {}
    for curve in curves.tolist():
        ends.setdefault(domain.curves[curve].start, []).append((curve, "start"))
        ends.setdefault(domain.curves[curve].end, []).append((curve, "end"))
    junctions = tuple(
        CurveJunction(f"corner:{corner}", tuple(members))
        for corner, members in sorted(ends.items())
        if len(members) >= 2
    )
    return CurveMeshingSpec(
        CellMeshingTarget(1, 3, CellFamilyPolicy(required=("interval",))),
        scope(curves),
        size_controls=controls,
        protected_features=features,
        junctions=junctions,
        limits=specification.limits,
    )


def compile_surface_domain(
    domain: MeshingDomain, specification: SurfaceMeshingSpec, /
) -> CompiledSurfaceDomain:
    """Resolve one admitted surface request against the domain's strata.

    A scope bound to another source, a stale revision or another entity set is
    refused with ``SCOPE_RESOLUTION_FAILED`` before any work.
    """

    if not isinstance(domain, MeshingDomain):
        raise TypeError("domain must be MeshingDomain.")
    if not isinstance(specification, SurfaceMeshingSpec):
        raise TypeError("specification must be SurfaceMeshingSpec.")
    scope = specification.scope
    if scope.entity_dimension != 2 or not _binds(domain, scope):
        raise _stale(domain, scope)
    patches = np.unique(_resolved(domain, scope))
    _semantic_controls(domain, specification, patches)
    curves, corners = _selected_strata(domain, patches)
    for feature in specification.protected_features:
        if not _binds(domain, feature.scope):
            raise _stale(domain, feature.scope)
    feature_issues = _feature_issues(domain, specification, patches)
    if feature_issues:
        raise MeshingFailure(
            MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
            "; ".join(feature_issues),
            stage=MeshingStageKind.SOURCE_INSPECTION.value,
            provider_code="protected_source_scope",
        )
    sizing = SourceSurfaceSizing(domain, specification)
    patch_sizes, curve_sizes = _size_worksets(
        domain, specification, patches, curves, sizing
    )
    normal_angles = np.full((patches.size,), np.inf)
    for control in sizing.controls:
        if isinstance(control, CurvatureSizeControl):
            selected = np.isin(patches, _resolved(domain, control.scope))
            normal_angles[selected] = np.minimum(
                normal_angles[selected], control.normal_angle
            )
    patch_deviations, curve_deviations = _deviation_worksets(
        domain, specification, patches, curves
    )
    request = _curve_request(domain, specification, curves, curve_sizes, curve_deviations)
    return CompiledSurfaceDomain(
        domain,
        patches,
        curves,
        corners,
        patch_sizes,
        curve_sizes,
        patch_deviations,
        curve_deviations,
        request,
        sizing,
        normal_angles,
        specification.specification_id,
    )


__all__ = [
    "CompiledSurfaceDomain",
    "compile_surface_domain",
    "surface_domain_issues",
]
