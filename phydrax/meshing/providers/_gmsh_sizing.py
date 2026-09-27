#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Gmsh size fields (uniform, proximity, background metric) and size evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...discretization import CellMesh
from ...discretization._reference_cell import reference_cell_topology
from ...geometry.brep import BRepModel
from .._contracts import (
    MeshingFailure,
    MeshingFailureCategory,
    SurfaceMeshingSpec,
    VolumeMeshingSpec,
)
from .._controls import BackgroundMetricControl, BackgroundMetricMode
from .._metric import _host_spectrum
from .._sizing import (
    CurvatureSizeControl,
    ProximitySizeControl,
    resolve_size_controls,
    SizeCombinationPolicy,
    SizeCompliancePolicy,
    SizeControlStrength,
    SizeFieldDomain,
    UniformSizeControl,
)
from .._trace import MeshingStageKind
from ._gmsh_evidence import _EvidenceSection
from ._gmsh_import import _CadEntityMap, _resolve_entities, _source_scale


_LIST_NAMES = {
    0: "PointsList",
    1: "CurvesList",
    2: "SurfacesList",
    3: "VolumesList",
}


def _size_values(specification: SurfaceMeshingSpec | VolumeMeshingSpec, /) -> Any:
    top_scope = (
        specification.scope
        if isinstance(specification, SurfaceMeshingSpec)
        else specification.boundary_scope
    )
    whole = tuple(
        control
        for control in specification.size_controls
        if not isinstance(control, ProximitySizeControl)
        and control.scope.scope_id == top_scope.scope_id
    )
    uniform = tuple(
        control for control in whole if isinstance(control, UniformSizeControl)
    )
    hard = tuple(
        control for control in whole if control.strength is SizeControlStrength.HARD
    )
    minimum = max(
        (control.minimum_size for control in hard if control.minimum_size is not None),
        default=None,
    )
    maximum = min(
        (control.maximum_size for control in hard if control.maximum_size is not None),
        default=None,
    )
    if not uniform:
        target = None
    elif specification.size_combination is SizeCombinationPolicy.REJECT_HARD_CONFLICTS:
        target = min(control.target_size for control in uniform)
    else:
        hard_uniform = tuple(
            control for control in uniform if control.strength is SizeControlStrength.HARD
        )
        candidates = hard_uniform if hard_uniform else uniform
        priority = max(control.priority for control in candidates)
        target = next(
            control.target_size for control in candidates if control.priority == priority
        )
    if target is not None:
        if minimum is not None:
            target = max(target, minimum)
        if maximum is not None:
            target = min(target, maximum)
    curvature_angles = tuple(
        control.normal_angle
        for control in whole
        if isinstance(control, CurvatureSizeControl)
    )
    curvature_points = (
        0 if not curvature_angles else int(np.ceil(2.0 * np.pi / min(curvature_angles)))
    )
    return minimum, target, maximum, curvature_points


def _uniform_size_fields(
    gmsh: Any,
    source: BRepModel,
    shape: Any,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    cad_entities: _CadEntityMap | None,
    outside_size: float,
    /,
) -> list[int]:
    controls = tuple(
        control
        for control in specification.size_controls
        if isinstance(control, UniformSizeControl)
    )
    if not controls:
        return []
    semantic_volume = isinstance(specification, VolumeMeshingSpec) and bool(
        specification.region_controls
    )
    if semantic_volume:
        if cad_entities is None:
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "Semantic solid size fields require resolved CAD volume identities.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
        solid_ids = np.arange(source.topology.num_solids, dtype=np.int64)
        resolved, _ = resolve_size_controls(
            controls,
            np.zeros((solid_ids.size, 1), dtype=np.float64),
            solid_ids,
            SizeFieldDomain.EUCLIDEAN_VOLUME,
            combination=specification.size_combination,
        )
        values = np.asarray(resolved.values, dtype=np.float64)
        groups = tuple(
            (
                float(value),
                tuple(
                    cad_entities.solid_to_volume[int(index)]
                    for index in np.flatnonzero(values == value)
                ),
            )
            for value in np.unique(values)
        )
        dimension = 3
    else:
        top_scope = (
            specification.scope
            if isinstance(specification, SurfaceMeshingSpec)
            else specification.boundary_scope
        )
        if specification.size_combination is SizeCombinationPolicy.EXPLICIT_PRIORITY:
            groups = [
                (
                    outside_size,
                    tuple(
                        tag
                        for _, tag in gmsh.model.getEntities(top_scope.entity_dimension)
                    ),
                    top_scope.entity_dimension,
                )
            ]
        else:
            groups = []
            for control in controls:
                dimension = control.scope.entity_dimension
                tags = (
                    tuple(tag for _, tag in gmsh.model.getEntities(dimension))
                    if control.scope.scope_id == top_scope.scope_id
                    else _resolve_entities(gmsh, source, shape, control.scope)
                )
                groups.append((control.target_size, tags, dimension))
    fields = []
    if semantic_volume:
        # ty: ignore[invalid-assignment]
        grouped = tuple((value, tags, dimension) for value, tags in groups)
    else:
        grouped = tuple(groups)
    # ty: ignore[invalid-assignment]
    for value, tags, entity_dimension in grouped:
        field = gmsh.model.mesh.field.add("Constant")
        gmsh.model.mesh.field.setNumber(field, "VIn", value)
        gmsh.model.mesh.field.setNumber(field, "VOut", outside_size)
        gmsh.model.mesh.field.setNumbers(field, _LIST_NAMES[entity_dimension], list(tags))
        gmsh.model.mesh.field.setNumber(field, "IncludeBoundary", 1)
        fields.append(field)
    return fields


@dataclass(frozen=True, slots=True)
class _ProximityField:
    control: ProximitySizeControl
    source_tags: tuple[int, ...]
    target_tags: tuple[int, ...]
    gap: float
    field: int


def _chord_is_normal(
    gmsh: Any, dimension: int, tag: int, point: np.ndarray, chord: np.ndarray, /
) -> bool:
    parameters = gmsh.model.getParametrization(dimension, tag, point.tolist())
    if dimension == 1:
        tangent = np.asarray(
            gmsh.model.getDerivative(1, tag, parameters), dtype=np.float64
        )[:3]
        return abs(float(tangent @ chord)) <= 1.0e-6 * float(np.linalg.norm(tangent))
    normal = np.asarray(gmsh.model.getNormal(tag, parameters), dtype=np.float64)[:3]
    return abs(float(normal @ chord)) >= (1.0 - 1.0e-6) * float(np.linalg.norm(normal))


def _proximity_gap(
    gmsh: Any,
    dimension: int,
    sources: tuple[int, ...],
    targets: tuple[int, ...],
    facing_required: bool,
    tolerance: float,
    /,
) -> float:
    """Closest CAD gap; facing entities meet it along a chord normal to both."""
    best = None
    for source in sources:
        for target in targets:
            distance, *coordinates = gmsh.model.occ.getDistance(
                dimension, source, dimension, target
            )
            if best is None or float(distance) < best[0]:
                best = (
                    float(distance),
                    source,
                    target,
                    np.asarray(coordinates, dtype=np.float64).reshape((2, 3)),
                )
    # ty: ignore[not-iterable]
    gap, source, target, points = best
    if not np.isfinite(gap) or gap <= tolerance:
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SPECIFICATION,
            "ProximitySizeControl scopes touch or overlap; no positive gap exists.",
            stage=MeshingStageKind.SIZE_FIELD_RESOLUTION.value,
        )
    chord = (points[1] - points[0]) / gap
    if facing_required and not (
        _chord_is_normal(gmsh, dimension, source, points[0], chord)
        and _chord_is_normal(gmsh, dimension, target, points[1], chord)
    ):
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SPECIFICATION,
            "ProximitySizeControl scopes do not face each other across their closest gap.",
            stage=MeshingStageKind.SIZE_FIELD_RESOLUTION.value,
        )
    return gap


def _proximity_field(
    gmsh: Any, source: BRepModel, shape: Any, control: ProximitySizeControl, /
) -> _ProximityField:
    """Local gap width is the sum of distances to both walls; size is gap / count."""
    dimension = control.source_scope.entity_dimension
    scale = _source_scale(source)
    sources = _resolve_entities(gmsh, source, shape, control.source_scope)
    targets = _resolve_entities(gmsh, source, shape, control.target_scope)
    gap = _proximity_gap(
        gmsh,
        dimension,
        sources,
        targets,
        control.opposite_normals_only,
        1.0e-9 * scale,
    )
    # Distance fields sample walls; resolve each wall finer than the gap size.
    resolution = int(np.ceil(4.0 * scale * control.elements_per_gap / gap))
    sampling = int(np.clip(resolution, 20, 2000 if dimension == 1 else 300))
    distances = []
    for tags in (sources, targets):
        field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(field, _LIST_NAMES[dimension], list(tags))
        gmsh.model.mesh.field.setNumber(field, "Sampling", sampling)
        distances.append(field)
    expression = f"(F{distances[0]} + F{distances[1]}) / {control.elements_per_gap}"
    if control.maximum_size is not None:
        expression = f"Min({control.maximum_size:.17g}, {expression})"
    if control.minimum_size is not None:
        expression = f"Max({control.minimum_size:.17g}, {expression})"
    field = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(field, "F", expression)
    return _ProximityField(control, sources, targets, gap, field)


@dataclass(frozen=True, slots=True)
class _BackgroundView:
    control: BackgroundMetricControl
    view: int
    field: int


def _background_view(gmsh: Any, control: BackgroundMetricControl, /) -> _BackgroundView:
    """Lower a vertex metric to a Gmsh list-data post-processing view."""
    mesh = control.mesh
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    cells = np.concatenate(
        tuple(np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks)
    )
    metrics = control.vertex_metrics
    match control.mode:
        case BackgroundMetricMode.ISOTROPIC:
            eigenvalues, _ = _host_spectrum(metrics)
            values = (1.0 / np.sqrt(eigenvalues[:, -1]))[:, None]
            prefix = "S"
        case BackgroundMetricMode.ANISOTROPIC:
            values = metrics.reshape((metrics.shape[0], -1))
            prefix = "T"
        case _:
            raise ValueError("Unsupported background metric mode.")
    suffix = {"triangle": "T", "tetrahedron": "S"}[mesh.blocks[0].cell_kind]
    element_coordinates = np.swapaxes(coordinates[cells], 1, 2).reshape(
        (cells.shape[0], -1)
    )
    element_values = values[cells].reshape((cells.shape[0], -1))
    data = np.concatenate((element_coordinates, element_values), axis=1)
    view = gmsh.view.add(f"phydrax-background-{control.control_id[:12]}")
    gmsh.view.addListData(
        view, prefix + suffix, cells.shape[0], data.reshape(-1).tolist()
    )
    field = gmsh.model.mesh.field.add("PostView")
    gmsh.model.mesh.field.setNumber(field, "ViewTag", view)
    return _BackgroundView(control, view, field)


@dataclass(frozen=True, slots=True)
class _SizeFields:
    field_ids: tuple[int, ...]
    proximity: tuple[_ProximityField, ...]
    background: _BackgroundView | None


def _apply_size_fields(
    gmsh: Any,
    source: BRepModel,
    shape: Any,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    cad_entities: _CadEntityMap | None,
    outside_size: float,
    background: BackgroundMetricControl | None,
    /,
) -> _SizeFields:
    """Install one background field: the minimum of every scalar size request.

    An anisotropic tensor metric cannot enter a scalar minimum; preflight admits
    it only as the sole field, with every uniform target dominated by it.
    """
    view = None if background is None else _background_view(gmsh, background)
    # ty: ignore[unresolved-attribute]
    if view is not None and background.mode is BackgroundMetricMode.ANISOTROPIC:
        gmsh.model.mesh.field.setAsBackgroundMesh(view.field)
        return _SizeFields((view.field,), (), view)
    fields = _uniform_size_fields(
        gmsh, source, shape, specification, cad_entities, outside_size
    )
    proximity = tuple(
        _proximity_field(gmsh, source, shape, control)
        for control in specification.size_controls
        if isinstance(control, ProximitySizeControl)
    )
    fields.extend(value.field for value in proximity)
    if view is not None:
        fields.append(view.field)
    if not fields:
        return _SizeFields((), (), None)
    active = fields[0]
    if len(fields) > 1:
        active = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(active, "FieldsList", fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(active)
    return _SizeFields(tuple(fields), proximity, view)


def _edge_size_evidence(
    control: UniformSizeControl,
    edges: np.ndarray,
    points: np.ndarray,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    /,
) -> tuple[list[str], tuple[tuple[str, float], ...], tuple[tuple[str, float], ...]]:
    unique_edges = np.unique(np.sort(edges, axis=1), axis=0)
    lengths = np.linalg.norm(
        points[unique_edges[:, 1]] - points[unique_edges[:, 0]], axis=1
    )
    if not lengths.size:
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "A size control has no generated mesh edges.",
            stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
        )
    vertex_minimum = np.full((points.shape[0],), np.inf)
    vertex_maximum = np.zeros((points.shape[0],), dtype=np.float64)
    np.minimum.at(vertex_minimum, unique_edges[:, 0], lengths)
    np.minimum.at(vertex_minimum, unique_edges[:, 1], lengths)
    np.maximum.at(vertex_maximum, unique_edges[:, 0], lengths)
    np.maximum.at(vertex_maximum, unique_edges[:, 1], lengths)
    active = np.isfinite(vertex_minimum) & (vertex_minimum > 0.0)
    growth = float(np.max(vertex_maximum[active] / vertex_minimum[active], initial=1.0))
    local_minimum = float(np.min(lengths))
    local_maximum = float(np.max(lengths))
    key = f"size:{control.control_id}"
    requested = [(f"{key}:target_size", control.target_size)]
    achieved = [
        (f"{key}:minimum_edge", local_minimum),
        (f"{key}:maximum_edge", local_maximum),
        (f"{key}:maximum_local_edge_ratio", growth),
    ]
    for statistic in specification.size_compliance.target_statistics:
        quantile = {"p50": 0.5, "p95": 0.95}[statistic]
        achieved.append(
            (f"{key}:{statistic}_edge", float(np.quantile(lengths, quantile)))
        )
    optional = (
        ("minimum_size", control.minimum_size),
        ("maximum_size", control.maximum_size),
        ("maximum_growth_rate", control.maximum_growth_rate),
    )
    requested.extend(
        (f"{key}:{name}", value) for name, value in optional if value is not None
    )
    issues = []
    if control.strength is SizeControlStrength.HARD:
        policy = specification.size_compliance
        if control.minimum_size is not None:
            tolerance = policy.absolute_tolerance + (
                policy.relative_tolerance * abs(control.minimum_size)
            )
            if local_minimum < control.minimum_size - tolerance:
                issues.append(f"minimum_size:{control.control_id}")
        if control.maximum_size is not None:
            tolerance = policy.absolute_tolerance + (
                policy.relative_tolerance * abs(control.maximum_size)
            )
            if local_maximum > control.maximum_size + tolerance:
                issues.append(f"maximum_size:{control.control_id}")
        if control.maximum_growth_rate is not None:
            tolerance = policy.absolute_tolerance + (
                policy.relative_tolerance * abs(control.maximum_growth_rate)
            )
            if growth > control.maximum_growth_rate + tolerance:
                issues.append(f"maximum_growth_rate:{control.control_id}")
    return issues, tuple(requested), tuple(achieved)


def _semantic_size_compliance(
    mesh: CellMesh,
    specification: VolumeMeshingSpec,
    cell_solid_ids: np.ndarray,
    /,
) -> tuple[list[str], tuple[tuple[str, float], ...], tuple[tuple[str, float], ...]]:
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    issues = []
    requested = []
    achieved = []
    for control in specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            continue
        selected_edges = []
        cursor = 0
        solid_ids = np.asarray(control.scope.entity_ids, dtype=np.int64)
        for block in mesh.blocks:
            stop = cursor + block.cell_count
            selected = np.isin(cell_solid_ids[cursor:stop], solid_ids)
            if np.any(selected):
                cells = np.asarray(block.vertices, dtype=np.int32)[selected]
                pairs = np.asarray(
                    reference_cell_topology(block.cell_kind).entities[1],
                    dtype=np.int32,
                )
                selected_edges.append(cells[:, pairs].reshape((-1, 2)))
            cursor = stop
        if not selected_edges:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "A solid-scoped size control has no generated cells.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        local_edges = np.concatenate(selected_edges)
        local_issues, local_requested, local_achieved = _edge_size_evidence(
            control, local_edges, points, specification
        )
        issues.extend(local_issues)
        requested.extend(local_requested)
        achieved.extend(local_achieved)
    return issues, tuple(requested), tuple(achieved)


def _unique_edges(edges: np.ndarray, /) -> np.ndarray:
    return np.unique(np.sort(np.asarray(edges, dtype=np.int64), axis=1), axis=0)


def _wall_distance(
    gmsh: Any, dimension: int, tags: Any, points: np.ndarray, /
) -> np.ndarray:
    distance = np.full((points.shape[0],), np.inf)
    for tag in tags:
        closest, _ = gmsh.model.getClosestPoint(dimension, tag, points.reshape(-1))
        distance = np.minimum(
            distance,
            np.linalg.norm(
                np.asarray(closest, dtype=np.float64).reshape((-1, 3)) - points, axis=1
            ),
        )
    return distance


def _proximity_evidence(
    gmsh: Any,
    fields: tuple[_ProximityField, ...],
    edges: np.ndarray,
    points: np.ndarray,
    policy: SizeCompliancePolicy,
    /,
) -> _EvidenceSection:
    """Check every edge against the exact CAD local gap at its midpoint."""
    if not fields:
        return _EvidenceSection()
    unique = _unique_edges(edges)
    lengths = np.linalg.norm(points[unique[:, 1]] - points[unique[:, 0]], axis=1)
    midpoints = 0.5 * (points[unique[:, 0]] + points[unique[:, 1]])
    requested = []
    achieved = []
    issues = []
    for value in fields:
        control = value.control
        dimension = control.source_scope.entity_dimension
        local_gap = _wall_distance(
            gmsh, dimension, value.source_tags, midpoints
        ) + _wall_distance(gmsh, dimension, value.target_tags, midpoints)
        required = local_gap / control.elements_per_gap
        required = np.clip(
            required,
            -np.inf if control.minimum_size is None else control.minimum_size,
            np.inf if control.maximum_size is None else control.maximum_size,
        )
        allowed = required * (1.0 + policy.relative_tolerance) + policy.absolute_tolerance
        ratio = float(np.max(lengths / required))
        key = f"proximity:{control.control_id}"
        requested.extend(
            (
                (f"{key}:elements_per_gap", float(control.elements_per_gap)),
                *(
                    (f"{key}:{name}", bound)
                    for name, bound in (
                        ("minimum_size", control.minimum_size),
                        ("maximum_size", control.maximum_size),
                    )
                    if bound is not None
                ),
            )
        )
        achieved.extend(
            (
                (f"{key}:gap", value.gap),
                (f"{key}:maximum_edge_ratio", ratio),
                (f"{key}:minimum_elements_per_gap", control.elements_per_gap / ratio),
            )
        )
        if control.strength is SizeControlStrength.HARD and np.any(lengths > allowed):
            issues.append(f"proximity:{control.control_id}")
    return _EvidenceSection(tuple(requested), tuple(achieved), tuple(issues))


def _background_metric_evidence(
    gmsh: Any,
    view: _BackgroundView | None,
    edges: np.ndarray,
    points: np.ndarray,
    policy: SizeCompliancePolicy,
    tolerance: float,
    /,
) -> _EvidenceSection:
    """Measure every generated edge in the metric Gmsh interpolates from the view."""
    if view is None:
        return _EvidenceSection()
    unique = _unique_edges(edges)
    vectors = points[unique[:, 1]] - points[unique[:, 0]]
    midpoints = 0.5 * (points[unique[:, 0]] + points[unique[:, 1]])
    anisotropic = view.control.mode is BackgroundMetricMode.ANISOTROPIC
    width = 9 if anisotropic else 1
    samples = np.empty((midpoints.shape[0], width), dtype=np.float64)
    # Gmsh probes one point per call; this host evidence loop is provider-bound.
    for index, point in enumerate(midpoints):
        values, distance = gmsh.view.probe(view.view, *point.tolist())
        if len(values) != width or float(distance) > tolerance:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "The background metric does not cover a generated mesh edge.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        samples[index] = values
    if anisotropic:
        tensors = samples.reshape((-1, 3, 3))
        lengths = np.sqrt(
            np.sum(vectors * (tensors @ vectors[:, :, None])[:, :, 0], axis=1)
        )
    else:
        lengths = np.linalg.norm(vectors, axis=1) / samples[:, 0]
    control = view.control
    key = f"background_metric:{control.control_id}"
    unit = (lengths >= 1.0 / np.sqrt(2.0)) & (lengths <= np.sqrt(2.0))
    achieved = (
        (f"{key}:minimum_edge_length", float(np.min(lengths))),
        (f"{key}:mean_edge_length", float(np.mean(lengths))),
        (f"{key}:maximum_edge_length", float(np.max(lengths))),
        (f"{key}:unit_edge_fraction", float(np.mean(unit))),
    )
    bound = control.maximum_metric_edge_length
    if bound is None:
        return _EvidenceSection((), achieved)
    issues = ()
    if float(np.max(lengths)) > bound * (1.0 + policy.relative_tolerance) + (
        policy.absolute_tolerance
    ):
        issues = (f"background_metric:{control.control_id}",)
    return _EvidenceSection(((f"{key}:maximum_edge_length", bound),), achieved, issues)
