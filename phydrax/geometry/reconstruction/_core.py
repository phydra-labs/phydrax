#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import assert_never, Protocol, runtime_checkable, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._mass import Mass
from ..._strict import StrictModule
from ..._validation import positive_finite_float, positive_integer
from ...measurement.lidar import LidarPointProduct
from ...typing import checked, parse, PRNGKey
from .._capabilities import (
    ClosestPointProvider,
    ContactCurvatureProvider,
    GeometryCapability,
    SeamDiagnosticsProvider,
    SupportMapProvider,
)
from .._certificate import FieldCertificate
from .._contracts import (
    ClosestPointResult,
    ContactCurvatureResult,
    GeometryKernel,
    GeometryKind,
    GeometrySource,
)
from .._cubature import CubatureAtlas, CubatureComponent
from .._triangulation import DelaunayTriangulation
from .._validity import GeometryValidityEvidence, representation_validity
from ..design._schema import _ParameterCollector, DesignState
from ..simplicial._io import (
    _canonical_triangle_arrays,
    planar_region_from_triangles,
)
from ..simplicial._regions import MeshRegion, PlanarMeshRegion, TriangleSurface
from ..simplicial._topology import TriangleTopology
from ._normals import (
    estimate_point_normals,
    NormalEstimationEvidence,
    NormalOrientation,
)
from ._poisson import (
    extract_indicator_surface,
    PoissonDiscretization,
    PoissonIndicator,
    PoissonSolveEvidence,
    sample_areas,
    sampled_surface_deviation,
    SampledSurfaceDeviation,
    solve_regular_poisson,
)
from ._poisson_octree import solve_octree_poisson
from ._robustness import (
    component_fit,
    ComponentFitEvidence,
    OutlierRemovalEvidence,
    ReconstructionRobustness,
    remove_statistical_outliers,
    sampling_coverage,
    SamplingCoverageEvidence,
    thin_features,
    ThinFeatureEvidence,
)


# The default Poisson grid spacing is twice the typical sample spacing: coarse
# enough that every cell near the surface holds samples, fine enough to resolve
# features at the sampling scale. Four padding cells keep the level set off the
# natural boundary. Coverage and thin-feature distances default to two finest
# cells, the width over which the smoothed indicator crosses its unit jump.
_SPACING_PER_SAMPLE = 2.0
_PADDING_CELLS = 4
_RESOLUTION_CELLS = 2.0
_DEFAULT_ROBUSTNESS = ReconstructionRobustness()


if TYPE_CHECKING:
    from .._atlas import BoundaryAtlas
    from .._sampling import RejectionSamplingPlan, SamplingResult


@dataclass(frozen=True, slots=True)
class ReconstructionReport:
    """Provenance, filtering, topology, and approximation facts for reconstruction.

    Point-cloud surface routes also carry their normal estimation/orientation
    evidence, native screened Poisson solve evidence, the output Euler
    characteristic, the sampled two-sided deviation of the accepted mesh, and
    the outlier-removal, thin-feature, component-fit, and sampling-coverage
    evidence of their declared :class:`ReconstructionRobustness` policy.
    """

    source_kind: str
    algorithm: str
    input_digest: str
    input_points: int
    retained_points: int
    output_vertices: int
    output_cells: int
    connected_components: int
    watertight: bool
    winding_consistent: bool
    recenter_offset: tuple[float, ...]
    parameters: tuple[tuple[str, str], ...]
    warnings: tuple[str, ...] = ()
    source_product_id: str | None = None
    euler_characteristic: int | None = None
    normal_evidence: NormalEstimationEvidence | None = None
    poisson_evidence: PoissonSolveEvidence | None = None
    deviation_evidence: SampledSurfaceDeviation | None = None
    outlier_evidence: OutlierRemovalEvidence | None = None
    thin_feature_evidence: ThinFeatureEvidence | None = None
    component_evidence: ComponentFitEvidence | None = None
    coverage_evidence: SamplingCoverageEvidence | None = None


@runtime_checkable
class ReconstructionReportProvider(Protocol):
    report: ReconstructionReport


class ReconstructionFailure(ValueError):
    """Reconstruction failure retaining the diagnostics produced before rejection."""

    report: ReconstructionReport

    def __init__(self, message: str, report: ReconstructionReport) -> None:
        super().__init__(message)
        self.report = report


class ReconstructedGeometrySource(GeometrySource):
    """Geometry source carrying an immutable reconstruction report."""

    source: GeometrySource
    report: ReconstructionReport = eqx.field(static=True)

    @checked
    def __init__(self, source: GeometrySource, report: ReconstructionReport) -> None:
        self.source = source
        self.report = report

    def _compile(self, context: _ParameterCollector, /) -> GeometryKernel:
        return _ReconstructedGeometryKernel(self.source._compile(context), self.report)


class _ReconstructedGeometryKernel(GeometryKernel):
    child: GeometryKernel
    report: ReconstructionReport = eqx.field(static=True)

    def __init__(self, child: GeometryKernel, report: ReconstructionReport) -> None:
        self.child = child
        self.report = report

    @property
    def ambient_dimension(self) -> int:
        return self.child.ambient_dimension

    @property
    def intrinsic_dimension(self) -> int:
        return self.child.intrinsic_dimension

    @property
    def kind(self) -> GeometryKind:
        return self.child.kind

    @property
    def capabilities(self) -> frozenset[GeometryCapability]:
        return self.child.capabilities

    @property
    def field_certificate(self) -> FieldCertificate:
        return self.child.field_certificate

    def geometry_validity(self, state: DesignState, /) -> GeometryValidityEvidence:
        return representation_validity(self.child, state)

    def boundary_field(self, state: DesignState, points: Array, /) -> Array:
        return self.child.boundary_field(state, points)

    def contains(self, state: DesignState, points: Array, /) -> Array:
        return self.child.contains(state, points)

    def boundary_normal(self, state: DesignState, points: Array, /) -> Array:
        return self.child.boundary_normal(state, points)

    def closest_point(self, state: DesignState, points: Array, /) -> ClosestPointResult:
        if not isinstance(self.child, ClosestPointProvider):
            raise TypeError("Reconstructed child lacks a closest-point provider.")
        result = self.child.closest_point(state, points)
        if not isinstance(result, ClosestPointResult):
            raise TypeError("Child closest-point query returned an invalid result.")
        return result

    def contact_curvature(
        self, state: DesignState, points: Array, /
    ) -> ContactCurvatureResult:
        if not isinstance(self.child, ContactCurvatureProvider):
            raise TypeError("Reconstructed child lacks a contact-curvature provider.")
        result = self.child.contact_curvature(state, points)
        if not isinstance(result, ContactCurvatureResult):
            raise TypeError("Child contact-curvature query returned an invalid result.")
        return result

    def support_map(self, state: DesignState, directions: Array, /) -> Array:
        if not isinstance(self.child, SupportMapProvider):
            raise TypeError("Reconstructed child lacks a support-map provider.")
        return self.child.support_map(state, directions)

    def bounds(self, state: DesignState, /) -> Array:
        return self.child.bounds(state)

    def measure(self, state: DesignState, /) -> Array:
        return self.child.measure(state)

    def boundary_measure(self, state: DesignState, /) -> Array:
        return self.child.boundary_measure(state)

    def interior_mass(self, state: DesignState, /) -> Mass:
        return self.child.interior_mass(state)

    def boundary_mass(self, state: DesignState, /) -> Mass:
        return self.child.boundary_mass(state)

    def sample_interior(
        self,
        state: DesignState,
        num_points: int,
        /,
        *,
        key: PRNGKey,
        plan: RejectionSamplingPlan | None = None,
    ) -> SamplingResult:
        return self.child.sample_interior(
            state,
            num_points,
            key=key,
            plan=plan,
        )

    def sample_boundary(
        self, state: DesignState, num_points: int, /, *, key: PRNGKey
    ) -> SamplingResult:
        return self.child.sample_boundary(state, num_points, key=key)

    def boundary_atlas(self, state: DesignState, /) -> BoundaryAtlas:
        return self.child.boundary_atlas(state)

    def cubature_atlas(
        self, state: DesignState, component: CubatureComponent, /
    ) -> CubatureAtlas:
        return self.child.cubature_atlas(state, component)

    def seam_residual(self, state: DesignState, /) -> Array:
        if not isinstance(self.child, SeamDiagnosticsProvider):
            raise TypeError("Reconstructed child lacks a seam-diagnostics provider.")
        return self.child.seam_residual(state)


def _point_digest(points: np.ndarray) -> str:
    canonical = np.ascontiguousarray(points, dtype=np.float64)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _validated_points(points: ArrayLike, dimension: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < dimension:
        raise ValueError(f"points must have shape (num_points, >= {dimension}).")
    values = values[:, :dimension]
    if values.shape[0] < dimension + 1:
        raise ValueError("The point cloud has too few points for reconstruction.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Reconstruction points must all be finite.")
    return values


def _recenter(points: np.ndarray, enabled: bool) -> tuple[np.ndarray, np.ndarray]:
    offset = (
        0.5 * (np.min(points, axis=0) + np.max(points, axis=0))
        if enabled
        else np.zeros((points.shape[1],), dtype=np.float64)
    )
    return points - offset, offset


def _planar_triangulation(
    points: np.ndarray,
    /,
    *,
    tolerance: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = max(float(np.max(np.ptp(points, axis=0))), 1.0)
    if tolerance > 0.0:
        quantized = np.rint(points / (tolerance * scale)).astype(np.int64)
        _, retained = np.unique(quantized, axis=0, return_index=True)
    else:
        _, retained = np.unique(points, axis=0, return_index=True)
    retained = np.sort(retained)
    vertices = points[retained]
    if vertices.shape[0] < 3:
        raise ValueError("Planar reconstruction retained fewer than three points.")
    # Exact native Delaunay: counterclockwise triangles, cocircular ties resolved
    # by index-ordered symbolic perturbation of the retained (distinct) points.
    faces = np.array(DelaunayTriangulation(vertices).simplices, dtype=np.int32)
    if alpha > 0.0:
        triangles = vertices[faces]
        doubled_area = (triangles[:, 1, 0] - triangles[:, 0, 0]) * (
            triangles[:, 2, 1] - triangles[:, 0, 1]
        ) - (triangles[:, 1, 1] - triangles[:, 0, 1]) * (
            triangles[:, 2, 0] - triangles[:, 0, 0]
        )
        first = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
        second = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
        third = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)
        radius = first * second * third / np.maximum(2.0 * doubled_area, 1.0e-300)
        faces = faces[radius <= alpha]
    if faces.shape[0] == 0:
        raise ValueError("Planar reconstruction produced no retained triangles.")
    return vertices, faces, retained


def _clean_surface_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, TriangleTopology]:
    vertices_, faces_ = _canonical_triangle_arrays(vertices, faces)
    # ty: ignore[invalid-argument-type]
    topology = TriangleTopology(faces_, num_vertices=vertices_.shape[0])
    return vertices_, faces_, topology


def _parameter_records(**parameters: object) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((name, repr(value)) for name, value in parameters.items()))


def reconstruct_planar_region(
    points: ArrayLike,
    *,
    recenter: bool = True,
    alpha: float = 0.0,
    tolerance: float = 1e-5,
    feature_id: str | None = None,
) -> ReconstructedGeometrySource:
    """Reconstruct one planar region and report every host-side approximation.

    Points merged by ``tolerance`` (relative to the point extent) are
    triangulated by the exact native Delaunay kernel; ``alpha > 0`` removes
    triangles whose circumradius exceeds ``alpha`` before the oriented boundary
    loops are recovered.
    """

    points_ = _validated_points(points, 2)
    if alpha < 0.0 or tolerance < 0.0:
        raise ValueError("alpha and tolerance must be non-negative.")
    vertices_2d, faces, _ = _planar_triangulation(
        points_,
        tolerance=float(tolerance),
        alpha=float(alpha),
    )
    planar = planar_region_from_triangles(
        vertices_2d,
        faces,
        recenter=False,
        feature_id=feature_id,
    )
    planar_vertices = np.asarray(planar.vertices, dtype=np.float64)
    vertices, center = _recenter(planar_vertices, recenter)
    offsets = np.asarray(planar.loop_offsets, dtype=np.int32)
    loops = tuple(
        np.arange(offsets[index], offsets[index + 1], dtype=np.int32)
        for index in range(offsets.shape[0] - 1)
    )
    source = PlanarMeshRegion(vertices, loops, feature_id=feature_id)
    report = ReconstructionReport(
        source_kind="planar_point_cloud",
        algorithm="native_delaunay_2d_native_boundary",
        input_digest=_point_digest(points_),
        input_points=points_.shape[0],
        retained_points=points_.shape[0],
        output_vertices=vertices.shape[0],
        output_cells=faces.shape[0],
        connected_components=1,
        watertight=True,
        winding_consistent=True,
        recenter_offset=tuple(float(value) for value in center),
        parameters=_parameter_records(alpha=float(alpha), tolerance=float(tolerance)),
    )
    return ReconstructedGeometrySource(source, report)


@dataclass(frozen=True, slots=True)
class _SurfaceRequest:
    """Provenance and accumulated evidence shared by every surface report."""

    source_kind: str
    algorithm: str
    input_digest: str
    recenter: bool
    parameters: tuple[tuple[str, str], ...]
    input_points: int
    warnings: tuple[str, ...] = ()
    feature_id: str | None = None
    source_product_id: str | None = None
    normal_evidence: NormalEstimationEvidence | None = None
    poisson_evidence: PoissonSolveEvidence | None = None
    outlier_evidence: OutlierRemovalEvidence | None = None
    thin_feature_evidence: ThinFeatureEvidence | None = None
    component_evidence: ComponentFitEvidence | None = None
    coverage_evidence: SamplingCoverageEvidence | None = None


def _report(
    points: np.ndarray,
    request: _SurfaceRequest,
    /,
    *,
    vertices: int,
    cells: int,
    topology: TriangleTopology | None,
    recenter_offset: tuple[float, ...],
    warnings: tuple[str, ...],
    deviation: SampledSurfaceDeviation | None,
) -> ReconstructionReport:
    return ReconstructionReport(
        source_kind=request.source_kind,
        algorithm=request.algorithm,
        input_digest=request.input_digest,
        input_points=request.input_points,
        retained_points=points.shape[0],
        output_vertices=vertices,
        output_cells=cells,
        connected_components=0 if topology is None else topology.num_face_components,
        watertight=False if topology is None else topology.watertight,
        winding_consistent=topology is not None,
        recenter_offset=recenter_offset,
        parameters=request.parameters,
        warnings=warnings,
        source_product_id=request.source_product_id,
        euler_characteristic=None if topology is None else topology.euler_characteristic,
        normal_evidence=request.normal_evidence,
        poisson_evidence=request.poisson_evidence,
        deviation_evidence=deviation,
        outlier_evidence=request.outlier_evidence,
        thin_feature_evidence=request.thin_feature_evidence,
        component_evidence=request.component_evidence,
        coverage_evidence=request.coverage_evidence,
    )


def _failure_report(
    points: np.ndarray,
    request: _SurfaceRequest,
    vertices: int,
    cells: int,
    reason: str,
    /,
) -> ReconstructionReport:
    return _report(
        points,
        request,
        vertices=vertices,
        cells=cells,
        topology=None,
        recenter_offset=(0.0, 0.0, 0.0),
        warnings=(*request.warnings, reason),
        deviation=None,
    )


def _clean_or_fail(
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    request: _SurfaceRequest,
    /,
) -> tuple[np.ndarray, np.ndarray, TriangleTopology]:
    try:
        return _clean_surface_mesh(vertices, faces)
    except ValueError as error:
        report = _failure_report(
            points, request, vertices.shape[0], faces.shape[0], str(error)
        )
        raise ReconstructionFailure(
            "Surface reconstruction produced invalid triangle topology.", report
        ) from error


def _surface_source(
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    request: _SurfaceRequest,
    /,
    *,
    measure_deviation: bool,
) -> ReconstructedGeometrySource:
    vertices_clean, faces_clean, topology = _clean_or_fail(
        points, vertices, faces, request
    )
    deviation = (
        sampled_surface_deviation(points, vertices_clean, faces_clean)
        if measure_deviation and topology.watertight
        else None
    )
    vertices_clean, center = _recenter(vertices_clean, request.recenter)
    report = _report(
        points,
        request,
        vertices=vertices_clean.shape[0],
        cells=faces_clean.shape[0],
        topology=topology,
        recenter_offset=tuple(float(value) for value in center),
        warnings=request.warnings,
        deviation=deviation,
    )
    if not report.watertight:
        raise ReconstructionFailure(
            "Surface reconstruction did not produce a watertight consistently wound solid.",
            report,
        )
    # ty: ignore[invalid-argument-type]
    source = MeshRegion(vertices_clean, faces_clean, feature_id=request.feature_id)
    return ReconstructedGeometrySource(source, report)


def _normal_warnings(evidence: NormalEstimationEvidence, /) -> tuple[str, ...]:
    warnings: list[str] = []
    if evidence.ambiguous_direction_count:
        warnings.append(
            f"{evidence.ambiguous_direction_count} neighborhoods have no dominant "
            "least-variance direction."
        )
    if evidence.ambiguous_orientation_edges:
        warnings.append(
            f"{evidence.ambiguous_orientation_edges} orientation-propagation edges join "
            "nearly perpendicular normals."
        )
    if evidence.conflicting_neighbor_edges:
        warnings.append(
            f"{evidence.conflicting_neighbor_edges} neighbor pairs keep opposing "
            "normals after orientation."
        )
    return tuple(warnings)


@dataclass(frozen=True, slots=True)
class _PoissonRequest:
    """Validated numerical options of one point-cloud Poisson reconstruction."""

    normals: ArrayLike | None
    normal_orientation: NormalOrientation
    neighborhood_size: int
    sample_spacing: float | None
    screening: float
    maximum_grid_nodes: int
    discretization: PoissonDiscretization
    robustness: ReconstructionRobustness


@dataclass(frozen=True, slots=True)
class _PoissonSurface:
    """Extracted level set with the evidence accumulated up to extraction."""

    points: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray
    unsupported: np.ndarray
    coverage: SamplingCoverageEvidence
    request: _SurfaceRequest


def _poisson_request(
    *,
    normals: ArrayLike | None,
    normal_orientation: NormalOrientation,
    neighborhood_size: int,
    sample_spacing: float | None,
    screening: float,
    maximum_grid_nodes: int,
    discretization: PoissonDiscretization,
    robustness: ReconstructionRobustness,
) -> _PoissonRequest:
    if not isinstance(robustness, ReconstructionRobustness):
        raise TypeError("robustness must be ReconstructionRobustness.")
    return _PoissonRequest(
        normals=normals,
        normal_orientation=parse(
            normal_orientation, NormalOrientation, "normal_orientation"
        ),
        neighborhood_size=positive_integer(neighborhood_size, "neighborhood_size"),
        sample_spacing=None
        if sample_spacing is None
        else positive_finite_float(sample_spacing, "sample_spacing"),
        screening=positive_finite_float(screening, "screening"),
        maximum_grid_nodes=positive_integer(maximum_grid_nodes, "maximum_grid_nodes"),
        discretization=parse(discretization, PoissonDiscretization, "discretization"),
        robustness=robustness,
    )


def _poisson_parameters(options: _PoissonRequest, /) -> dict[str, object]:
    policy = options.robustness
    return {
        "neighborhood_size": options.neighborhood_size,
        "sample_spacing": options.sample_spacing,
        "screening": options.screening,
        "maximum_grid_nodes": options.maximum_grid_nodes,
        "discretization": options.discretization,
        "outlier_std_ratio": policy.outlier_std_ratio,
        "coverage_distance": policy.coverage_distance,
        "incomplete_sampling": policy.incomplete_sampling,
        "thin_feature_distance": policy.thin_feature_distance,
        "component_fit_tolerance": policy.component_fit_tolerance,
    }


def _without_outliers(
    points: np.ndarray, options: _PoissonRequest, request: _SurfaceRequest, /
) -> tuple[np.ndarray, ArrayLike | None, _SurfaceRequest]:
    """Apply declared statistical outlier removal to samples and supplied normals."""

    ratio = options.robustness.outlier_std_ratio
    if ratio is None:
        return points, options.normals, request
    retained, evidence = remove_statistical_outliers(
        points, options.neighborhood_size, ratio
    )
    normals = options.normals
    if normals is not None:
        supplied = np.asarray(normals, dtype=np.float64)
        if supplied.shape != points.shape:
            raise ValueError("normals must have shape (num_points, 3).")
        normals = supplied[retained]
    warnings = request.warnings
    if evidence.removed_indices:
        warnings = (
            *warnings,
            f"Statistical outlier removal dropped {len(evidence.removed_indices)} "
            f"samples above mean kNN distance {evidence.threshold:.6g}.",
        )
    if np.count_nonzero(retained) <= options.neighborhood_size:
        raise ReconstructionFailure(
            "Statistical outlier removal retained too few samples.",
            _failure_report(
                points[retained],
                replace(request, outlier_evidence=evidence),
                0,
                0,
                "fewer samples than neighborhood_size + 1 remain",
            ),
        )
    return (
        points[retained],
        normals,
        replace(request, warnings=warnings, outlier_evidence=evidence),
    )


def _solve(
    points: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    spacing: float,
    options: _PoissonRequest,
    /,
) -> PoissonIndicator:
    match options.discretization:
        case "octree":
            solver = solve_octree_poisson
        case "regular":
            solver = solve_regular_poisson
        case _:
            assert_never(options.discretization)
    return solver(
        points,
        normals,
        areas,
        spacing=spacing,
        screening=options.screening,
        padding_cells=_PADDING_CELLS,
        maximum_grid_nodes=options.maximum_grid_nodes,
    )


def _poisson_surface(
    points: np.ndarray, request: _SurfaceRequest, options: _PoissonRequest, /
) -> _PoissonSurface:
    """Filter, orient, solve, check, extract, and measure the Poisson surface."""

    points, supplied, request = _without_outliers(points, options, request)
    oriented = estimate_point_normals(
        points,
        normals=supplied,
        orientation=options.normal_orientation,
        neighborhood_size=options.neighborhood_size,
    )
    areas = sample_areas(oriented.neighbor_distances)
    spacing = (
        _SPACING_PER_SAMPLE * float(np.sqrt(np.median(areas)))
        if options.sample_spacing is None
        else options.sample_spacing
    )
    policy = options.robustness
    thin = thin_features(
        points,
        oriented.normals,
        options.neighborhood_size,
        _RESOLUTION_CELLS * spacing
        if policy.thin_feature_distance is None
        else policy.thin_feature_distance,
    )
    warnings = (*request.warnings, *_normal_warnings(oriented.evidence))
    if thin.thin_samples:
        warnings = (
            *warnings,
            f"{thin.thin_samples} samples have a stacked sheet closer than "
            f"{thin.thin_feature_distance:.6g}; Poisson smoothing cannot separate "
            "features that thin.",
        )
    indicator = _solve(points, oriented.normals, areas, spacing, options)
    fit = component_fit(
        indicator.sample_values,
        areas,
        oriented.components,
        policy.component_fit_tolerance,
    )
    request = replace(
        request,
        warnings=warnings,
        normal_evidence=oriented.evidence,
        poisson_evidence=indicator.evidence,
        thin_feature_evidence=thin,
        component_evidence=fit,
    )
    if not indicator.evidence.solver_converged:
        raise ReconstructionFailure(
            "Screened Poisson solve did not converge.",
            _failure_report(points, request, 0, 0, indicator.evidence.solver_message),
        )
    if fit.unresolved_components:
        raise ReconstructionFailure(
            "Sample components are not fit by the reconstructed level set.",
            _failure_report(
                points,
                request,
                0,
                0,
                f"components {fit.unresolved_components} exceed the sampled-indicator "
                f"spread tolerance {fit.tolerance:.6g}",
            ),
        )
    vertices, faces = extract_indicator_surface(indicator)
    unsupported, coverage = sampling_coverage(
        points,
        vertices,
        faces,
        _RESOLUTION_CELLS * spacing
        if policy.coverage_distance is None
        else policy.coverage_distance,
    )
    if coverage.unsupported_triangles:
        request = replace(
            request,
            warnings=(
                *request.warnings,
                f"{coverage.unsupported_patches} surface patches of area "
                f"{coverage.unsupported_area:.6g} lie farther than "
                f"{coverage.coverage_distance:.6g} from every sample.",
            ),
        )
    request = replace(request, coverage_evidence=coverage)
    return _PoissonSurface(points, vertices, faces, unsupported, coverage, request)


def _poisson_source(
    points: np.ndarray, request: _SurfaceRequest, options: _PoissonRequest, /
) -> ReconstructedGeometrySource:
    """Reconstruct the closed Poisson solid under the incomplete-sampling policy."""

    surface = _poisson_surface(points, request, options)
    match options.robustness.incomplete_sampling:
        case "close":
            pass
        case "refuse":
            if surface.coverage.unsupported_triangles:
                raise ReconstructionFailure(
                    "Incomplete sampling leaves surface unsupported by samples.",
                    _failure_report(
                        surface.points,
                        surface.request,
                        surface.vertices.shape[0],
                        surface.faces.shape[0],
                        "incomplete_sampling='refuse'",
                    ),
                )
        case _:
            assert_never(options.robustness.incomplete_sampling)
    return _surface_source(
        surface.points,
        surface.vertices,
        surface.faces,
        surface.request,
        measure_deviation=True,
    )


def reconstruct_surface_region(
    points: ArrayLike,
    *,
    normals: ArrayLike | None = None,
    normal_orientation: NormalOrientation = "propagate",
    recenter: bool = True,
    neighborhood_size: int = 16,
    sample_spacing: float | None = None,
    screening: float = 4.0,
    maximum_grid_nodes: int = 1 << 19,
    discretization: PoissonDiscretization = "octree",
    robustness: ReconstructionRobustness = _DEFAULT_ROBUSTNESS,
    feature_id: str | None = None,
) -> ReconstructedGeometrySource:
    """Reconstruct a closed surface from samples by native screened Poisson.

    Normals are estimated by bounded-neighborhood PCA when omitted and oriented
    by minimum-spanning-forest propagation unless ``normal_orientation`` is
    ``"supplied"`` (see :func:`estimate_point_normals`). The indicator is solved
    with finest width ``sample_spacing`` (default: twice the typical sample
    spacing) on an adaptive octree refined only around the samples
    (``discretization="octree"``) or on the full regular grid
    (``"regular"``), padded by four finest cells, with screening weight
    ``screening``; more than ``maximum_grid_nodes`` unknowns are refused before
    assembly. ``robustness`` declares outlier removal, the thin-feature and
    coverage distances, the incomplete-sampling policy, and the component-fit
    tolerance above which a sample component is refused. The report carries
    normal, solve, robustness, Euler-characteristic, and sampled two-sided
    deviation evidence. Screened Poisson smoothing approximates the sampled
    surface at the finest resolution and does not preserve features below it.
    """

    points_ = _validated_points(points, 3)
    options = _poisson_request(
        normals=normals,
        normal_orientation=normal_orientation,
        neighborhood_size=neighborhood_size,
        sample_spacing=sample_spacing,
        screening=screening,
        maximum_grid_nodes=maximum_grid_nodes,
        discretization=discretization,
        robustness=robustness,
    )
    request = _SurfaceRequest(
        source_kind="surface_point_cloud",
        algorithm="native_screened_poisson",
        input_digest=_point_digest(points_),
        recenter=recenter,
        parameters=_parameter_records(
            normals="supplied" if normals is not None else "estimated",
            normal_orientation=options.normal_orientation,
            **_poisson_parameters(options),
        ),
        input_points=points_.shape[0],
        feature_id=feature_id,
    )
    return _poisson_source(points_, request, options)


class TrimmedSurfaceReconstruction(StrictModule):
    """Open Poisson surface trimmed to its sample support, with its report.

    ``surface`` keeps the triangles within ``coverage_distance`` of a sample;
    the report's topology fields describe the trimmed surface and its coverage
    evidence the removed triangles.
    """

    surface: TriangleSurface
    report: ReconstructionReport = eqx.field(static=True)

    def __init__(self, surface: TriangleSurface, report: ReconstructionReport) -> None:
        if not isinstance(surface, TriangleSurface):
            raise TypeError("surface must be TriangleSurface.")
        if not isinstance(report, ReconstructionReport):
            raise TypeError("report must be ReconstructionReport.")
        self.surface = surface
        self.report = report


def reconstruct_trimmed_surface(
    points: ArrayLike,
    *,
    normals: ArrayLike | None = None,
    normal_orientation: NormalOrientation = "propagate",
    recenter: bool = True,
    neighborhood_size: int = 16,
    sample_spacing: float | None = None,
    screening: float = 4.0,
    maximum_grid_nodes: int = 1 << 19,
    discretization: PoissonDiscretization = "octree",
    robustness: ReconstructionRobustness = _DEFAULT_ROBUSTNESS,
) -> TrimmedSurfaceReconstruction:
    """Reconstruct an open sampled surface by Poisson extraction and density trimming.

    The pipeline and options follow :func:`reconstruct_surface_region`; the
    closed level set is then trimmed to the triangles whose centroid lies
    within the robustness ``coverage_distance`` of a sample, so incompletely
    sampled regions become open boundaries instead of interpolated closures.
    ``incomplete_sampling`` does not apply. A surface with no supported
    triangle is refused.
    """

    points_ = _validated_points(points, 3)
    options = _poisson_request(
        normals=normals,
        normal_orientation=normal_orientation,
        neighborhood_size=neighborhood_size,
        sample_spacing=sample_spacing,
        screening=screening,
        maximum_grid_nodes=maximum_grid_nodes,
        discretization=discretization,
        robustness=robustness,
    )
    request = _SurfaceRequest(
        source_kind="surface_point_cloud",
        algorithm="native_screened_poisson_density_trimmed",
        input_digest=_point_digest(points_),
        recenter=recenter,
        parameters=_parameter_records(
            normals="supplied" if normals is not None else "estimated",
            normal_orientation=options.normal_orientation,
            **_poisson_parameters(options),
        ),
        input_points=points_.shape[0],
    )
    surface = _poisson_surface(points_, request, options)
    kept = surface.faces[~surface.unsupported]
    if kept.shape[0] == 0:
        raise ReconstructionFailure(
            "Density trimming removed every reconstructed triangle.",
            _failure_report(
                surface.points,
                surface.request,
                surface.vertices.shape[0],
                surface.faces.shape[0],
                "no triangle lies within coverage_distance of a sample",
            ),
        )
    vertices, faces, topology = _clean_or_fail(
        surface.points, surface.vertices, kept, surface.request
    )
    vertices, center = _recenter(vertices, recenter)
    report = _report(
        surface.points,
        surface.request,
        vertices=vertices.shape[0],
        cells=faces.shape[0],
        topology=topology,
        recenter_offset=tuple(float(value) for value in center),
        warnings=surface.request.warnings,
        deviation=None,
    )
    return TrimmedSurfaceReconstruction(
        TriangleSurface(
            jnp.asarray(vertices, dtype=jnp.float64), jnp.asarray(faces, dtype=jnp.int32)
        ),
        report,
    )


def _terrain_points(
    points_or_grid: ArrayLike,
    *,
    x: ArrayLike | None,
    y: ArrayLike | None,
) -> np.ndarray:
    values = np.asarray(points_or_grid, dtype=np.float64)
    if values.ndim == 2 and values.shape[1] != 3:
        rows, columns = values.shape
        x_values = (
            np.arange(columns, dtype=np.float64)
            if x is None
            else np.asarray(x, dtype=np.float64)
        )
        y_values = (
            np.arange(rows, dtype=np.float64)
            if y is None
            else np.asarray(y, dtype=np.float64)
        )
        if x_values.shape != (columns,) or y_values.shape != (rows,):
            raise ValueError("x and y coordinate vectors must match the height grid.")
        x_grid, y_grid = np.meshgrid(x_values, y_values)
        points = np.column_stack((x_grid.ravel(), y_grid.ravel(), values.ravel()))
    else:
        points = _validated_points(values, 3)
    if not np.all(np.isfinite(points)):
        raise ValueError("Terrain samples must all be finite.")
    return points


def reconstruct_dem_region(
    points_or_grid: ArrayLike,
    *,
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
    recenter: bool = True,
    alpha: float = 0.0,
    tolerance: float = 1e-5,
    extrude_depth: float = 1.0,
    feature_id: str | None = None,
) -> ReconstructedGeometrySource:
    """Triangulate a terrain by native Delaunay and cap a downward extrusion.

    The retained plan-view samples are triangulated exactly as in
    :func:`reconstruct_planar_region`; the terrain sheet, its copy lowered by
    ``extrude_depth`` and the side walls along the boundary form the solid.
    """

    points = _terrain_points(points_or_grid, x=x, y=y)
    if alpha < 0.0 or tolerance < 0.0 or extrude_depth <= 0.0:
        raise ValueError(
            "alpha/tolerance must be non-negative and extrude_depth positive."
        )
    planar_vertices, top_faces, retained_indices = _planar_triangulation(
        points[:, :2],
        tolerance=float(tolerance),
        alpha=float(alpha),
    )
    top_vertices = points[retained_indices].copy()
    top_vertices[:, :2] = planar_vertices
    # ty: ignore[invalid-argument-type]
    topology = TriangleTopology(top_faces, num_vertices=top_vertices.shape[0])
    boundary = np.asarray(topology.boundary_halfedges, dtype=np.int32)
    origins = np.asarray(topology.halfedge_origin, dtype=np.int32)[boundary]
    destinations = np.asarray(topology.halfedge_destination, dtype=np.int32)[boundary]
    count = top_vertices.shape[0]
    bottom_vertices = top_vertices.copy()
    bottom_vertices[:, 2] -= float(extrude_depth)
    side_first = np.stack((origins + count, destinations + count, destinations), axis=1)
    side_second = np.stack((origins + count, destinations, origins), axis=1)
    solid_vertices = np.concatenate((top_vertices, bottom_vertices), axis=0)
    solid_faces = np.concatenate(
        (
            top_faces,
            top_faces[:, [0, 2, 1]] + count,
            side_first,
            side_second,
        ),
        axis=0,
    )
    request = _SurfaceRequest(
        source_kind="digital_elevation_model",
        algorithm="native_delaunay_2d_capped_extrusion",
        input_digest=_point_digest(points),
        recenter=recenter,
        parameters=_parameter_records(
            alpha=float(alpha),
            tolerance=float(tolerance),
            extrude_depth=float(extrude_depth),
        ),
        input_points=points.shape[0],
        feature_id=feature_id,
    )
    return _surface_source(
        points, solid_vertices, solid_faces, request, measure_deviation=False
    )


def reconstruct_point_region(
    points: ArrayLike,
    *,
    recenter: bool = True,
    roi: tuple[float, float, float, float, float, float] | None = None,
    voxel_size: float | None = None,
    neighborhood_size: int = 16,
    sample_spacing: float | None = None,
    screening: float = 4.0,
    maximum_grid_nodes: int = 1 << 19,
    discretization: PoissonDiscretization = "octree",
    robustness: ReconstructionRobustness = _DEFAULT_ROBUSTNESS,
    feature_id: str | None = None,
    source_product_id: str | None = None,
) -> ReconstructedGeometrySource:
    """Crop and voxel-downsample Cartesian points, then reconstruct natively.

    Filtering keeps points inside ``roi`` and the first point of every
    ``voxel_size`` voxel in input order; the retained samples follow
    :func:`reconstruct_surface_region` with PCA normals oriented by
    minimum-spanning-forest propagation, the declared ``discretization``, and
    the ``robustness`` policy.
    """

    original = _validated_points(points, 3)
    options = _poisson_request(
        normals=None,
        normal_orientation="propagate",
        neighborhood_size=neighborhood_size,
        sample_spacing=sample_spacing,
        screening=screening,
        maximum_grid_nodes=maximum_grid_nodes,
        discretization=discretization,
        robustness=robustness,
    )
    retained = original
    warnings: list[str] = []
    if roi is not None:
        x_min, x_max, y_min, y_max, z_min, z_max = map(float, roi)
        if not (x_min < x_max and y_min < y_max and z_min < z_max):
            raise ValueError("roi minima must be strictly below maxima.")
        mask = (
            (retained[:, 0] >= x_min)
            & (retained[:, 0] <= x_max)
            & (retained[:, 1] >= y_min)
            & (retained[:, 1] <= y_max)
            & (retained[:, 2] >= z_min)
            & (retained[:, 2] <= z_max)
        )
        retained = retained[mask]
    if voxel_size is not None:
        if voxel_size <= 0.0:
            raise ValueError("voxel_size must be positive when provided.")
        voxel = np.floor(retained / float(voxel_size)).astype(np.int64)
        _, indices = np.unique(voxel, axis=0, return_index=True)
        retained = retained[np.sort(indices)]
    if retained.shape[0] < 4:
        raise ValueError("LiDAR filtering retained too few points for reconstruction.")
    if retained.shape[0] < original.shape[0] / 10:
        warnings.append("Filtering retained fewer than ten percent of input points.")
    request = _SurfaceRequest(
        source_kind="point_cloud",
        algorithm="voxel_filter_then_native_screened_poisson",
        input_digest=_point_digest(original),
        recenter=recenter,
        parameters=_parameter_records(
            roi=roi, voxel_size=voxel_size, **_poisson_parameters(options)
        ),
        input_points=original.shape[0],
        warnings=tuple(warnings),
        feature_id=feature_id,
        source_product_id=source_product_id,
    )
    return _poisson_source(retained, request, options)


def reconstruct_lidar_region(
    product: LidarPointProduct,
    *,
    recenter: bool = True,
    roi: tuple[float, float, float, float, float, float] | None = None,
    voxel_size: float | None = None,
    neighborhood_size: int = 16,
    sample_spacing: float | None = None,
    screening: float = 4.0,
    maximum_grid_nodes: int = 1 << 19,
    discretization: PoissonDiscretization = "octree",
    robustness: ReconstructionRobustness = _DEFAULT_ROBUSTNESS,
    feature_id: str | None = None,
) -> ReconstructedGeometrySource:
    """Reconstruct valid derived LiDAR points while retaining acquisition lineage."""
    if not isinstance(product, LidarPointProduct):
        raise TypeError("product must be LidarPointProduct.")
    active = np.asarray(product.support.active_mask, dtype=np.bool_)
    return reconstruct_point_region(
        np.asarray(product.support.points)[active],
        recenter=recenter,
        roi=roi,
        voxel_size=voxel_size,
        neighborhood_size=neighborhood_size,
        sample_spacing=sample_spacing,
        screening=screening,
        maximum_grid_nodes=maximum_grid_nodes,
        discretization=discretization,
        robustness=robustness,
        feature_id=feature_id,
        source_product_id=product.point_product_id,
    )


__all__ = [
    "ReconstructedGeometrySource",
    "ReconstructionReportProvider",
    "ReconstructionFailure",
    "ReconstructionReport",
    "reconstruct_dem_region",
    "reconstruct_lidar_region",
    "reconstruct_point_region",
    "reconstruct_planar_region",
    "reconstruct_surface_region",
    "reconstruct_trimmed_surface",
    "TrimmedSurfaceReconstruction",
]
