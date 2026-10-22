# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from fractions import Fraction
from time import perf_counter
from typing import Any, TypedDict

import numpy as np

import phydrax as phx
from phydrax import array_tree_fingerprint, canonical_fingerprint
from phydrax.geometry.reconstruction import ReconstructionFailure
from phydrax.geometry.simplicial import TriangleBVH, TriangleMesh
from phydrax.meshing import (
    declared_plc_domain,
    NativeMeshingPhaseMeasurement,
    NativeVolumeSchedule,
)
from tools.meshing_qualification._controls import (
    capacity_meshing_limits,
    QualificationControls,
)
from tools.meshing_qualification._records import (
    _phase_recordings,
    _result_identity,
    record_failure,
)
from tools.meshing_qualification._runtime import (
    _measured,
    _runtime_identity,
    Deadline,
    peak_resident_bytes,
    PEAK_RESIDENT_SCOPE,
    qualification_failure,
    StageLedger,
)
from tools.meshing_qualification.scenarios.envelope import _envelope_qualification_epochs
from tools.meshing_qualification.scenarios.triangulation import _simplex_measures


# Underspecified-defective sampled-surface acceptance domain. The input is an
# unoriented, noisy point sample of one unknown smooth closed surface that also
# has isolated outliers, a twofold density drop over one band, and one
# unsampled cap. Such data does not determine a unique solid; the declared
# native repair policy chooses one, and acceptance proves only what the data
# supports. Every defect is defined in the source's own frame, so one rule set
# applies to every registered source of the family.
_NEIGHBORHOOD_SIZE = 16
_OUTLIER_STD_RATIO = 2.0
_NOISE_STD = 0.0008
# The unsampled cap is every sample within ``_CAP_RADIUS`` of the source point
# farthest along ``_CAP_DIRECTION``; it stays clear of the thinned band.
_CAP_DIRECTION = np.asarray((0.6, 0.0, 0.8), dtype=np.float64)
_CAP_RADIUS = 0.2
# Every second sample is dropped where x < -_BAND_FRACTION * max|x|.
_BAND_FRACTION = 0.5
# Outliers lie on a shell this far beyond the source's bounding radius.
_OUTLIER_OFFSETS = (0.35, 0.55)
# Uniform sample density: the lobed sphere's 16 r^2 samples over area 4 pi 0.2^2.
_SAMPLES_PER_AREA = 16.0 / (4.0 * np.pi * 0.2**2)
_MAXIMUM_DISCARDED_GENUINE_FRACTION = 0.05
_MAXIMUM_REPAIR_EXCURSION_FRACTION = 0.4
_MINIMUM_DENSITY_RATIO = 1.2
_VOLUME_CLOSURE = 1.0e-12
_CENTER = np.asarray((0.07, -0.03, 0.02), dtype=np.float64)
_TORUS_RADII = (0.32, 0.17)
_ELLIPSOID_AXES = np.asarray((0.32, 0.24, 0.18), dtype=np.float64)


def _surface_sample_count(resolution: int, area: float) -> int:
    return int(round(_SAMPLES_PER_AREA * resolution * resolution * area))


def _lobed_sphere_samples(resolution: int, rng: np.random.Generator) -> np.ndarray:
    """Fibonacci samples of a three-lobed genus-zero perturbation of a sphere."""
    del rng
    count = 16 * resolution * resolution
    index = np.arange(count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * index / count
    angle = np.pi * (3.0 - np.sqrt(5.0)) * index
    radial = np.sqrt(1.0 - z * z)
    direction = np.stack((radial * np.cos(angle), radial * np.sin(angle), z), axis=1)
    radius = 0.2 * (1.0 + 0.04 * np.sin(3.0 * angle) * radial * radial)
    return direction * radius[:, None]


def _torus_samples(resolution: int, rng: np.random.Generator) -> np.ndarray:
    """Area-uniform random samples of a genus-one torus."""
    major, minor = _TORUS_RADII
    count = _surface_sample_count(resolution, 4.0 * np.pi**2 * major * minor)
    accepted: list[np.ndarray] = []
    total = 0
    while total < count:
        u, v, keep = rng.uniform(0.0, 1.0, size=(3, 4 * count))
        u, v = 2.0 * np.pi * u, 2.0 * np.pi * v
        ring = major + minor * np.cos(v)
        chosen = keep < ring / (major + minor)
        block = np.stack(
            (ring * np.cos(u), ring * np.sin(u), minor * np.sin(v)), axis=1
        )[chosen]
        accepted.append(block)
        total += block.shape[0]
    return np.concatenate(accepted)[:count]


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    unit = axis / np.linalg.norm(axis)
    cross = np.asarray(
        ((0.0, -unit[2], unit[1]), (unit[2], 0.0, -unit[0]), (-unit[1], unit[0], 0.0))
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * cross @ cross


def _rotated_ellipsoid_samples(resolution: int, rng: np.random.Generator) -> np.ndarray:
    """Area-uniform random samples of an obliquely rotated triaxial ellipsoid."""
    a, b, c = _ELLIPSOID_AXES
    # Thomsen's area formula only sets the sample count, not any acceptance.
    p = 1.6075
    area = 4.0 * np.pi * (((a * b) ** p + (a * c) ** p + (b * c) ** p) / 3.0) ** (1 / p)
    count = _surface_sample_count(resolution, area)
    rotation = _rotation(np.ones(3), np.deg2rad(40.0))
    accepted: list[np.ndarray] = []
    total = 0
    while total < count:
        direction = rng.normal(size=(2 * count, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        # The map d -> A d scales sphere area by det(A) |A^{-1} d|.
        weight = np.linalg.norm(direction / _ELLIPSOID_AXES, axis=1) * np.min(
            _ELLIPSOID_AXES
        )
        chosen = rng.uniform(size=weight.shape) < weight
        block = (direction[chosen] * _ELLIPSOID_AXES) @ rotation.T
        accepted.append(block)
        total += block.shape[0]
    return np.concatenate(accepted)[:count]


_RECONSTRUCTION_SOURCES: dict[
    str, tuple[Callable[[int, np.random.Generator], np.ndarray], int, str]
] = {
    "lobed-sphere": (_lobed_sphere_samples, 1709, "genus-0 three-lobed sphere"),
    "torus": (_torus_samples, 1711, "genus-1 torus R=0.32 r=0.17"),
    "rotated-ellipsoid": (
        _rotated_ellipsoid_samples,
        1713,
        "genus-0 triaxial ellipsoid (0.32, 0.24, 0.18) rotated 40 deg about (1,1,1)",
    ),
}


class _RepairPolicy(TypedDict):
    outlier_removal: str
    neighborhood_size: int
    outlier_std_ratio: float
    normal_orientation: str
    closure: str
    incomplete_sampling: str
    sample_spacing: float
    coverage_distance: float
    thin_feature_distance: float


class _ReconstructionDomain(TypedDict):
    name: str
    input: str
    registered_sources: dict[str, str]
    defect_frame_origin: list[float]
    admitted_defects: dict[str, str]
    repair_policy: _RepairPolicy
    acceptance: dict[str, str]
    preregistered_bounds: dict[str, float]
    refused: list[str]
    nonclaims: list[str]


class _ReconstructionGate(TypedDict):
    status: str
    domain: str
    case: object
    obligations: dict[str, bool]


def _reconstruction_qualification_domain(resolution: int) -> _ReconstructionDomain:
    """Declare the admitted input class, repair policy, claims, and nonclaims."""

    return {
        "name": "underspecified-defective-sampled-closed-surface",
        "input": (
            "unoriented 3D point samples of one unknown smooth closed surface of any "
            "genus whose thickness and handle openings exceed twice the coverage "
            "distance; no normals or connectivity"
        ),
        "registered_sources": {
            name: description
            for name, (_, _, description) in _RECONSTRUCTION_SOURCES.items()
        },
        "defect_frame_origin": _CENTER.tolist(),
        "admitted_defects": {
            "missing-normals": "no normals supplied",
            "additive-noise": f"isotropic Gaussian, std {_NOISE_STD}",
            "density-variation": (
                f"every second sample dropped where x < -{_BAND_FRACTION} max|x|"
            ),
            "isolated-outliers": (
                f"max(8, n/32) samples on a shell {_OUTLIER_OFFSETS} beyond the bounding radius"
            ),
            "unsampled-surface-region": (
                f"all samples within {_CAP_RADIUS} of the extreme point along {_CAP_DIRECTION.tolist()}"
            ),
        },
        "repair_policy": {
            "outlier_removal": "statistical-kNN-mean-distance",
            "neighborhood_size": _NEIGHBORHOOD_SIZE,
            "outlier_std_ratio": _OUTLIER_STD_RATIO,
            "normal_orientation": "propagate",
            "closure": "native-screened-Poisson-indicator-isosurface",
            "incomplete_sampling": "close",
            "sample_spacing": 0.44 / resolution,
            "coverage_distance": 0.44 / resolution,
            "thin_feature_distance": 0.22 / resolution,
        },
        "acceptance": {
            "evidence": (
                "observable data and this declared domain only; generating geometry, "
                "injected-outlier labels and source genus are reported diagnostics, never acceptance"
            ),
            "topology": (
                "one vertex-connected component, every edge used by exactly two faces in "
                "opposite directions; Euler characteristic V - E + F of the repaired "
                "triangles reported"
            ),
            "sample_representation": "every retained sample lies within coverage_distance of the repaired surface",
            "supported_surface": "triangles whose centroid lies within coverage_distance of a retained sample",
            "repair_region": (
                "unsupported triangles form the declared repair: one patch; area, patches, and "
                "excursion (largest repaired-vertex or centroid distance to a retained sample) "
                "are reported; excursion must not exceed "
                f"{_MAXIMUM_REPAIR_EXCURSION_FRACTION} of the retained-sample bounding diagonal"
            ),
            "outliers": (
                "removed set equals the independently recomputed statistical rule; some removed "
                "samples lie beyond coverage_distance of the repair; removed samples within "
                "coverage_distance of the repair are discarded supported data, at most "
                f"{_MAXIMUM_DISCARDED_GENUINE_FRACTION} of all supported samples"
            ),
            "density_variation": (
                "spacing ratio sqrt(rho_regular / rho_band) >= "
                f"{_MINIMUM_DENSITY_RATIO}, rho = retained samples per supported repaired "
                "area measured by exact halfspace clipping of supported triangles, band x - origin_x < "
                f"-{_BAND_FRACTION} max|x - origin_x| over retained samples"
            ),
            "consumer": "native tetrahedral fill with exact signed-volume closure and manufactured transferred-state PDE",
        },
        "preregistered_bounds": {
            "maximum_discarded_genuine_fraction": _MAXIMUM_DISCARDED_GENUINE_FRACTION,
            "maximum_repair_excursion_fraction": _MAXIMUM_REPAIR_EXCURSION_FRACTION,
            "minimum_density_ratio": _MINIMUM_DENSITY_RATIO,
            "volume_closure_absolute": _VOLUME_CLOSURE,
        },
        "refused": [
            "nonconverged Poisson solve",
            "unresolved sample components",
            "nonwatertight, inconsistently wound, or disconnected repair",
            "repair excursion or genuine-sample loss beyond the declared bounds",
        ],
        "nonclaims": [
            "The unsampled region's geometry is not recovered; its surface is the declared policy's choice.",
            "The repaired solid's topology is not claimed to equal the unknown original's.",
            "Sampled distances are not continuous Hausdorff certificates.",
            "Screened Poisson smoothing does not preserve features below its finest width.",
            "Generating source geometry is never used for acceptance.",
        ],
    }


@dataclass(frozen=True, slots=True)
class _DefectiveSamples:
    """One generated instance of the underspecified-defective acceptance domain.

    ``points`` lists the ``genuine_count`` retained surface samples first and
    the injected outliers after them; ``thinned`` marks genuine samples in the
    reduced-density band.
    """

    case: str
    points: np.ndarray
    genuine_count: int
    thinned: np.ndarray
    removed_cap_samples: int
    removed_band_samples: int
    bounding_radius: float


def _reconstruction_qualification_samples(case: str, resolution: int) -> _DefectiveSamples:
    """Sample one registered source, then apply every declared defect in its frame."""

    if case not in _RECONSTRUCTION_SOURCES:
        raise ValueError(
            f"qualification.reconstruction_case must be one of {sorted(_RECONSTRUCTION_SOURCES)}; got {case!r}"
        )
    factory, seed, _ = _RECONSTRUCTION_SOURCES[case]
    rng = np.random.default_rng(seed)
    surface = factory(resolution, rng)
    count = surface.shape[0]
    samples = surface + rng.normal(0.0, _NOISE_STD, size=surface.shape) + _CENTER
    bound = float(np.max(np.linalg.norm(surface, axis=1)))
    anchor = surface[int(np.argmax(surface @ _CAP_DIRECTION))]
    uncapped = np.linalg.norm(surface - anchor, axis=1) > _CAP_RADIUS
    band = surface[:, 0] < -_BAND_FRACTION * float(np.max(np.abs(surface[:, 0])))
    kept = uncapped & (~band | (np.arange(count) % 2 == 0))
    genuine = samples[kept]
    outlier_count = max(8, count // 32)
    outlier_direction = rng.normal(size=(outlier_count, 3))
    outlier_direction /= np.linalg.norm(outlier_direction, axis=1, keepdims=True)
    outlier_radius = rng.uniform(
        bound + _OUTLIER_OFFSETS[0], bound + _OUTLIER_OFFSETS[1], size=(outlier_count, 1)
    )
    outliers = outlier_direction * outlier_radius + _CENTER
    return _DefectiveSamples(
        case=case,
        points=np.concatenate((genuine, outliers), axis=0),
        genuine_count=genuine.shape[0],
        thinned=band[kept],
        removed_cap_samples=int(np.count_nonzero(~uncapped)),
        removed_band_samples=int(np.count_nonzero(uncapped & ~kept)),
        bounding_radius=bound,
    )


def _reconstruction_qualification_source(
    case: str,
    resolution: int,
    capacity: int,
) -> tuple[
    phx.geometry.ReconstructedGeometrySource, phx.meshing.NativePlcSource, _DefectiveSamples
]:
    """Repair an underspecified defective cloud with native numerical owners."""

    samples = _reconstruction_qualification_samples(case, resolution)
    reconstructed = phx.geometry.reconstruct_surface_region(
        samples.points,
        recenter=False,
        neighborhood_size=_NEIGHBORHOOD_SIZE,
        sample_spacing=0.44 / resolution,
        maximum_grid_nodes=5 * capacity,
        discretization="octree",
        robustness=phx.geometry.ReconstructionRobustness(
            outlier_std_ratio=_OUTLIER_STD_RATIO,
            incomplete_sampling="close",
            coverage_distance=0.44 / resolution,
            thin_feature_distance=0.22 / resolution,
        ),
        feature_id="qualification-native-underspecified-defective-samples",
    )
    child = reconstructed.source
    if not isinstance(child, phx.geometry.MeshRegion):
        raise TypeError(
            "qualification.reconstruction.source must be a MeshRegion; "
            f"got {type(child).__name__}"
        )
    vertices = np.asarray(child.vertices, dtype=np.float64)
    faces = np.asarray(child.faces, dtype=np.int64)
    revision = canonical_fingerprint(
        {
            "kind": "native-reconstructed-plc-source",
            "report": asdict(reconstructed.report),
            "vertices": array_tree_fingerprint(vertices),
            "triangles": array_tree_fingerprint(faces),
        }
    )
    complex_ = phx.meshing.PiecewiseLinearComplex(
        vertices,
        tuple(faces),
        np.arange(faces.shape[0], dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (faces.shape[0], 1)),
        ("reconstructed-solid",),
        boundary="conforming",
    )
    return (
        reconstructed,
        phx.meshing.NativePlcSource(complex_, child.feature_id, revision),
        samples,
    )


def _reconstruction_qualification_request(
    source: phx.meshing.NativePlcSource,
    capacity: int,
    timeout: float,
    adaptation_rounds: int,
) -> tuple[phx.meshing.VolumeMeshingSpec, phx.meshing.NativeMeshingOptions]:
    """Freeze the extracted PLC controls and explicit requested-adaptation schedule."""

    scope = phx.meshing.MeshingScope(
        source.source_id,
        source.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        "reconstructed-facets",
        np.arange(source.complex.facet_count, dtype=np.int64),
    )
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3, 3, phx.meshing.CellFamilyPolicy(required=("tetrahedron",))
        ),
        scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope, 2.0, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
        limits=capacity_meshing_limits(
            capacity, timeout, cells=5, work_units=1000, cavity_cells=None
        ),
    )
    return specification, phx.meshing.NativeMeshingOptions(
        "plc_tetrahedral",
        volume_schedule=NativeVolumeSchedule(
            refinement_rounds=1,
            improvement_passes=1,
            minimum_dihedral_degrees=0.0,
            prepare_fixed_boundary_adaptation=(
                source.complex.boundary == "fixed" and adaptation_rounds > 0
            ),
        ),
    )


def _nearest_distances(queries: np.ndarray, points: np.ndarray, /) -> np.ndarray:
    """Brute-force Euclidean distance from each query to its nearest point."""

    nearest = np.empty((queries.shape[0],), dtype=np.float64)
    for start in range(0, queries.shape[0], 256):
        block = queries[start : start + 256]
        squared = np.sum((block[:, None, :] - points[None, :, :]) ** 2, axis=-1)
        nearest[start : start + block.shape[0]] = np.sqrt(np.min(squared, axis=1))
    return nearest


def _statistical_outliers(
    points: np.ndarray, neighborhood_size: int, std_ratio: float, /
) -> tuple[np.ndarray, float]:
    """Brute-force reference of the Rusu mean-kNN-distance outlier rule."""

    means = np.empty((points.shape[0],), dtype=np.float64)
    for start in range(0, points.shape[0], 256):
        block = points[start : start + 256]
        distance = np.sqrt(np.sum((block[:, None, :] - points[None, :, :]) ** 2, axis=-1))
        rows = np.arange(block.shape[0])
        distance[rows, start + rows] = np.inf
        nearest = np.partition(distance, neighborhood_size - 1, axis=1)
        means[start : start + block.shape[0]] = np.mean(
            nearest[:, :neighborhood_size], axis=1
        )
    threshold = float(np.mean(means) + std_ratio * np.std(means))
    return np.flatnonzero(means > threshold), threshold


def _supported_band_areas(
    triangles: np.ndarray,
    areas: np.ndarray,
    supported: np.ndarray,
    threshold: Fraction,
) -> tuple[float, float]:
    """Measure strict x<threshold by exact barycentric clipping, not centroids.

    Support remains the independently declared whole-triangle classification.
    Zero-area triangles contribute zero. A triangle lying in the cut plane is
    entirely in the regular side because the band is strict.
    """
    if (
        triangles.ndim != 3 or triangles.shape[1:] != (3, 3)
        or areas.shape != (triangles.shape[0],) or supported.shape != areas.shape
    ):
        raise ValueError("Band areas require aligned physical triangles, areas and support.")
    band = regular = Fraction(0)
    for triangle, area, active in zip(triangles, areas, supported, strict=True):
        if not active:
            continue
        if not np.isfinite(area) or area < 0:
            raise ValueError("Supported triangle areas must be finite and nonnegative.")
        if area == 0:
            continue
        distances = tuple(Fraction(float(point[0])) - threshold for point in triangle)
        inside = tuple(index for index, value in enumerate(distances) if value < 0)
        if len(inside) in (0, 3):
            fraction = Fraction(len(inside) // 3)
        else:
            isolated = (
                inside[0] if len(inside) == 1
                else next(index for index in range(3) if index not in inside)
            )
            fraction = Fraction(1)
            for other in range(3):
                if other != isolated:
                    parameter = distances[isolated] / (
                        distances[isolated] - distances[other]
                    )
                    if not 0 <= parameter <= 1:
                        raise ValueError("Exact band intersection must lie on its triangle edge.")
                    fraction *= parameter
            if len(inside) == 2:
                fraction = 1 - fraction
        weight = Fraction(float(area))
        band += weight * fraction
        regular += weight * (1 - fraction)
    return float(band), float(regular)


def _vertex_components(faces: np.ndarray, vertex_count: int, /) -> int:
    """Count vertex-connected components of a triangle subset by union-find."""

    parent = np.arange(vertex_count, dtype=np.int64)

    def root(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    for a, b, c in faces.tolist():
        for first, second in ((a, b), (b, c)):
            left, right = root(first), root(second)
            if left != right:
                parent[max(left, right)] = min(left, right)
    return len({root(int(vertex)) for vertex in np.unique(faces)})


def _surface_topology(faces: np.ndarray, /) -> dict[str, int]:
    """Count the repaired triangles' incidences and Euler characteristic V - E + F."""

    directed = faces[:, ((0, 1), (1, 2), (2, 0))].reshape((-1, 2))
    _, directed_uses = np.unique(directed, axis=0, return_counts=True)
    _, uses = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
    vertices = np.unique(faces)
    return {
        "vertices": int(vertices.size),
        "edges": int(uses.size),
        "faces": int(faces.shape[0]),
        "boundary_edges": int(np.count_nonzero(uses == 1)),
        "nonmanifold_edges": int(np.count_nonzero(uses > 2)),
        "repeated_directed_edges": int(np.count_nonzero(directed_uses > 1)),
        "components": _vertex_components(faces, int(np.max(faces)) + 1)
        if faces.size
        else 0,
        "euler_characteristic": int(vertices.size - uses.size + faces.shape[0]),
    }


def _index_digest(indices: np.ndarray, /) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(indices, dtype=np.int64).tobytes()
    ).hexdigest()


def _spacing_ratio(
    band_samples: float, regular_samples: float, band_area: float, regular_area: float
) -> float:
    """Spacing ratio sqrt(rho_regular / rho_band) of area-normalized densities."""

    if min(band_samples, regular_samples, band_area, regular_area) <= 0.0:
        return 0.0
    return float(
        np.sqrt((regular_samples / regular_area) / (band_samples / band_area))
    )


def _reconstruction_qualification_repair(
    reconstructed: phx.geometry.ReconstructedGeometrySource,
    source: phx.meshing.NativePlcSource,
    samples: _DefectiveSamples,
) -> dict[str, object]:
    """Prove the declared repair from observable data and the declared domain alone.

    Only ``samples.points`` and ``samples.case`` enter acceptance. Generator
    labels (genuine count, injected outliers, thinned band, removed cap and
    band counts) appear solely in ``generator_diagnostics``.
    """

    def refuse(message: str, evidence: dict[str, object]) -> RuntimeError:
        return qualification_failure(
            "_reconstruction_qualification_repair", message, evidence
        )

    report = reconstructed.report
    points = samples.points
    if (
        report.input_digest
        != hashlib.sha256(
            np.ascontiguousarray(points, dtype=np.float64).tobytes()
        ).hexdigest()
        or report.input_points != points.shape[0]
        or dict(report.parameters).get("normals") != repr("estimated")
    ):
        raise refuse(
            "Reconstruction report binds a different or oriented sample cloud.",
            {"points": points, "report": report},
        )
    poisson, coverage, deviation = (
        report.poisson_evidence,
        report.coverage_evidence,
        report.deviation_evidence,
    )
    outliers, normals = report.outlier_evidence, report.normal_evidence
    if (
        poisson is None
        or not poisson.solver_converged
        or report.component_evidence is None
        or report.component_evidence.unresolved_components
        or coverage is None
        or deviation is None
        or outliers is None
        or normals is None
        or normals.orientation != "propagate"
    ):
        raise refuse(
            "Declared repair lacks a converged solve or its declared policy evidence.",
            {"report": report},
        )
    reference_removed, reference_threshold = _statistical_outliers(
        points, _NEIGHBORHOOD_SIZE, _OUTLIER_STD_RATIO
    )
    removed = np.asarray(outliers.removed_indices, dtype=np.int64)
    if (
        not np.array_equal(reference_removed, removed)
        or abs(reference_threshold - outliers.threshold)
        > 1.0e-12 * max(1.0, abs(outliers.threshold))
        or removed.size == 0
    ):
        raise refuse(
            "Outlier removal disagrees with its independent rule or removes nothing.",
            {
                "outliers": outliers,
                "reference_removed": reference_removed,
                "reference_threshold": reference_threshold,
            },
        )
    retained = np.delete(points, removed, axis=0)
    if retained.shape[0] != report.retained_points:
        raise refuse(
            "Reconstruction retained-sample count disagrees with its outlier evidence.",
            {"report": report, "retained": retained.shape[0]},
        )
    vertices = np.asarray(source.complex.vertices, dtype=np.float64)
    faces = np.asarray(source.complex.polygon_vertices, dtype=np.int64).reshape((-1, 3))
    topology = _surface_topology(faces)
    if (
        topology["boundary_edges"]
        or topology["nonmanifold_edges"]
        or topology["repeated_directed_edges"]
        or topology["components"] != 1
        or topology["euler_characteristic"] != report.euler_characteristic
        or topology["components"] != report.connected_components
        or not report.watertight
        or not report.winding_consistent
    ):
        raise refuse(
            "Repaired triangles are not one closed consistently wound component, or the report disagrees.",
            {"report": report, "topology": topology},
        )
    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    centroids = np.mean(triangles, axis=1)
    centroid_distance = _nearest_distances(centroids, retained)
    vertex_distance = _nearest_distances(vertices, retained)
    bvh = TriangleBVH(TriangleMesh(vertices, faces))
    sample_distance = np.asarray(bvh.query(retained).distance, dtype=np.float64)
    removed_distance = np.asarray(bvh.query(points[removed]).distance, dtype=np.float64)
    unsupported = centroid_distance > coverage.coverage_distance
    unsupported_area = float(np.sum(areas[unsupported]))
    surface_area = float(np.sum(areas))
    patches = (
        _vertex_components(faces[unsupported], vertices.shape[0])
        if np.any(unsupported)
        else 0
    )
    lower = np.min(retained, axis=0)
    diagonal = float(np.linalg.norm(np.max(retained, axis=0) - lower))
    excursion_bound = _MAXIMUM_REPAIR_EXCURSION_FRACTION * diagonal
    excursion = float(max(np.max(centroid_distance), np.max(vertex_distance)))
    area_scale = 1.0e-9 * surface_area
    if (
        abs(float(np.max(sample_distance)) - deviation.sample_to_surface_max) > 1.0e-12
        or abs(float(np.max(vertex_distance)) - deviation.surface_to_sample_max)
        > 1.0e-12
        or abs(surface_area - coverage.surface_area) > area_scale
        or abs(unsupported_area - coverage.unsupported_area) > area_scale
    ):
        raise refuse(
            "Reconstruction report disagrees with independent sample/surface distance queries.",
            {
                "coverage": coverage,
                "deviation": deviation,
                "sample_distance": sample_distance,
                "unsupported_area": unsupported_area,
                "vertex_distance": vertex_distance,
            },
        )
    if (
        float(np.max(sample_distance)) > coverage.coverage_distance
        or not np.any(unsupported)
        or patches != 1
        or excursion > excursion_bound
    ):
        raise refuse(
            "Repair leaves samples unrepresented, exercises no declared closure, or exceeds its excursion bound.",
            {
                "excursion": excursion,
                "excursion_bound": excursion_bound,
                "patches": patches,
                "sample_distance": sample_distance,
                "unsupported": unsupported,
            },
        )
    # Removed samples the repaired surface represents are discarded supported
    # data; removed samples off the surface are the observed outlier defect.
    removed_within = int(np.count_nonzero(removed_distance <= coverage.coverage_distance))
    removed_beyond = int(removed.size - removed_within)
    supported_total = int(retained.shape[0] + removed_within)
    if (
        removed_beyond == 0
        or removed_within > _MAXIMUM_DISCARDED_GENUINE_FRACTION * supported_total
    ):
        raise refuse(
            "Outlier removal exercises no off-surface sample or discards supported data beyond the declared bound.",
            {
                "removed_beyond": removed_beyond,
                "removed_within": removed_within,
                "supported_total": supported_total,
            },
        )
    # Area-normalized density on the supported repaired surface. Integrate
    # triangles crossing the declared band plane instead of assigning their
    # entire area from a centroid, which biases this metric at the boundary.
    relative = retained[:, 0] - _CENTER[0]
    cut = -_BAND_FRACTION * float(np.max(np.abs(relative)))
    band_samples = int(np.count_nonzero(relative < cut))
    regular_samples = int(retained.shape[0] - band_samples)
    band_area, regular_area = _supported_band_areas(
        triangles, areas, ~unsupported, Fraction(float(_CENTER[0])) + Fraction(cut)
    )
    spacing_ratio = _spacing_ratio(band_samples, regular_samples, band_area, regular_area)
    if spacing_ratio < _MINIMUM_DENSITY_RATIO:
        raise refuse(
            "Declared density-variation defect is absent from the admitted samples.",
            {
                "band_area": band_area,
                "band_samples": band_samples,
                "regular_area": regular_area,
                "regular_samples": regular_samples,
                "spacing_ratio": spacing_ratio,
            },
        )
    injected = np.arange(samples.genuine_count, points.shape[0], dtype=np.int64)
    return {
        "passed": True,
        "case": samples.case,
        "input_digest": report.input_digest,
        "input_points": int(points.shape[0]),
        "reconstruction_report": asdict(report),
        "outliers": {
            "removed": int(removed.size),
            "independent_removed": int(reference_removed.size),
            "removed_digest": _index_digest(removed),
            "independent_removed_digest": _index_digest(reference_removed),
            "threshold": outliers.threshold,
            "independent_threshold": reference_threshold,
            "retained": int(retained.shape[0]),
            "removed_within_coverage": removed_within,
            "removed_beyond_coverage": removed_beyond,
            "supported_discard_fraction": removed_within / supported_total,
        },
        "repaired_topology": {
            **topology,
            "claim": "topology of the declared repair, not of the unknown original",
        },
        "supported_fidelity": {
            "retained_sample_to_surface_max": float(np.max(sample_distance)),
            "coverage_distance": coverage.coverage_distance,
            "grid_spacing": poisson.grid_spacing,
        },
        "repair_region": {
            "unsupported_triangles": int(np.count_nonzero(unsupported)),
            "unsupported_patches": patches,
            "unsupported_area": unsupported_area,
            "surface_area": surface_area,
            "unsupported_area_fraction": unsupported_area / surface_area,
            "excursion_from_data": excursion,
            "retained_bounding_diagonal": diagonal,
            "semantics": "geometry chosen by the declared closure; ambiguity is bounded only by the reported excursion",
        },
        "density_variation": {
            "band_cut": cut,
            "band_samples": band_samples,
            "regular_samples": regular_samples,
            "band_supported_area": band_area,
            "regular_supported_area": regular_area,
            "spacing_ratio_thinned_to_regular": spacing_ratio,
        },
        "generator_diagnostics": {
            "used_for_acceptance": False,
            "source": _RECONSTRUCTION_SOURCES[samples.case][2],
            "source_bounding_radius": samples.bounding_radius,
            "genuine_samples": samples.genuine_count,
            "injected_outliers": int(injected.size),
            "injected_removed": int(np.count_nonzero(np.isin(injected, removed))),
            "genuine_discarded": int(np.count_nonzero(removed < samples.genuine_count)),
            "unsampled_cap_samples_removed": samples.removed_cap_samples,
            "density_band_samples_removed": samples.removed_band_samples,
        },
        "fidelity_semantics": "sampled-two-sided-deviation-not-continuous-Hausdorff",
    }


def _reconstruction_qualification_acceptance(
    source: phx.meshing.NativePlcSource,
    result: phx.meshing.CellMeshingResult,
) -> dict[str, object]:
    """Accept one native fill of the repaired PLC by exact domain evidence."""

    certificate = result.certification
    if (
        certificate is None
        or not certificate.passed
        or not result.audit.passed
        or not result.compliance.passed
    ):
        raise qualification_failure(
            "_reconstruction_qualification_acceptance",
            ("Reconstructed volume requires strict native acceptance."),
            {"certificate": certificate, "result": result},
        )
    domain = declared_plc_domain(source.complex, source.source_id)
    coverage, embedding, fidelity = (
        certificate.coverage,
        certificate.embedding,
        certificate.fidelity,
    )
    if (
        not isinstance(coverage, phx.geometry.DomainCoverageCertificate)
        or embedding is None
        or coverage.status != "certified"
        or embedding.status != "certified"
        or certificate.request.domain is None
        or certificate.request.domain.domain_id != domain.domain_id
        or coverage.covered_source_facet_count != coverage.source_facet_count
        or coverage.region_ids != ("reconstructed-solid",)
        or {zone.name for zone in result.zones} != {"reconstructed-solid"}
    ):
        raise qualification_failure(
            "_reconstruction_qualification_acceptance",
            ("Native fill lost reconstructed source-domain coverage or region identity."),
            {
                "certificate": certificate,
                "coverage": coverage,
                "domain": domain,
                "embedding": embedding,
                "result": result,
            },
        )
    faces = np.asarray(source.complex.polygon_vertices, dtype=np.int64).reshape((-1, 3))
    skin = source.complex.vertices[faces]
    surface_volume = float(np.sum(skin[:, 0] * np.cross(skin[:, 1], skin[:, 2])) / 6.0)
    mesh_points = np.asarray(result.mesh.coordinates, dtype=np.float64)
    cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in result.mesh.blocks]
    )
    volumes = _simplex_measures(mesh_points, cells)
    volume = float(np.sum(volumes))
    if (
        np.any(volumes <= 0.0)
        or surface_volume <= 0.0
        or abs(volume - surface_volume) > _VOLUME_CLOSURE
    ):
        raise qualification_failure(
            "_reconstruction_qualification_acceptance",
            ("Native reconstructed fill fails independent signed-volume closure."),
            {"surface_volume": surface_volume, "volume": volume, "volumes": volumes},
        )
    return {
        "passed": True,
        "source_id": source.source_id,
        "source_revision": source.source_revision,
        "source_domain_id": domain.domain_id,
        "embedding_status": embedding.status,
        "coverage_status": coverage.status,
        "fidelity_status": "exact-domain-coverage",
        "coverage_certificate_id": coverage.certificate_id,
        "additional_fidelity_certificate_id": None
        if fidelity is None
        else fidelity.certificate_id,
        "source_boundary_volume": surface_volume,
        "mesh_volume": volume,
        "volume_defect": volume - surface_volume,
        "minimum_cell_volume": float(np.min(volumes)),
        "volume_fidelity_semantics": "exact-represented-repaired-PLC-domain",
    }


_RESEARCH_GATE = "underspecified-defective-surface-reconstruction"


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if np.isfinite(value) else None


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _repair_obligation(
    repair: object, domain: _ReconstructionDomain, case: str, input_points: object
) -> bool:
    """Recompute every repair verdict from recorded measurements.

    Bounds come from the freshly derived domain, never from the record; the
    generator diagnostics are not read.
    """

    if not isinstance(repair, dict):
        return False
    bounds, policy = domain["preregistered_bounds"], domain["repair_policy"]
    try:
        outliers = repair["outliers"]
        topology = repair["repaired_topology"]
        fidelity = repair["supported_fidelity"]
        region = repair["repair_region"]
        density = repair["density_variation"]
        counts = {
            name: _count(outliers[name])
            for name in (
                "removed",
                "independent_removed",
                "retained",
                "removed_within_coverage",
                "removed_beyond_coverage",
            )
        }
        incidences = {
            name: _count(topology[name])
            for name in (
                "vertices",
                "edges",
                "faces",
                "boundary_edges",
                "nonmanifold_edges",
                "repeated_directed_edges",
                "components",
                "euler_characteristic",
            )
        }
        unsupported = _count(region["unsupported_triangles"])
        patches = _count(region["unsupported_patches"])
        band = _count(density["band_samples"])
        regular = _count(density["regular_samples"])
        values = [
            _number(item)
            for item in (
                outliers["threshold"],
                outliers["independent_threshold"],
                fidelity["coverage_distance"],
                fidelity["retained_sample_to_surface_max"],
                region["unsupported_area"],
                region["surface_area"],
                region["excursion_from_data"],
                region["retained_bounding_diagonal"],
                density["band_supported_area"],
                density["regular_supported_area"],
            )
        ]
    except (KeyError, TypeError):
        return False
    integers = [*counts.values(), *incidences.values(), unsupported, patches, band, regular]
    if (
        any(value is None for value in values)
        or any(value is None for value in integers)
        or _count(input_points) is None
    ):
        return False
    if unsupported is None or patches is None or band is None or regular is None:
        return False
    numeric_values = [value for value in values if value is not None]
    counts = {name: value for name, value in counts.items() if value is not None}
    incidences = {name: value for name, value in incidences.items() if value is not None}
    (
        threshold,
        reference,
        coverage,
        sample_max,
        unsupported_area,
        surface_area,
        excursion,
        diagonal,
        band_area,
        regular_area,
    ) = numeric_values
    within, beyond = counts["removed_within_coverage"], counts["removed_beyond_coverage"]
    supported = counts["retained"] + within
    return bool(
        repair.get("case") == case
        # Outliers: the native removal equals the independent rule, some
        # removed samples are off the repair, and supported data loss is bounded.
        and outliers.get("removed_digest") == outliers.get("independent_removed_digest")
        and counts["removed"] == counts["independent_removed"]
        and abs(threshold - reference) <= 1.0e-12 * max(1.0, abs(threshold))
        and counts["retained"] + counts["removed"] == input_points
        and within + beyond == counts["removed"]
        and beyond > 0
        and within <= bounds["maximum_discarded_genuine_fraction"] * supported
        # Topology of the repaired triangles: closed, consistently wound, one
        # component; the Euler characteristic is V - E + F with 2E = 3F.
        and incidences["faces"] > 0
        and incidences["boundary_edges"] == 0
        and incidences["nonmanifold_edges"] == 0
        and incidences["repeated_directed_edges"] == 0
        and incidences["components"] == 1
        and 2 * incidences["edges"] == 3 * incidences["faces"]
        and incidences["euler_characteristic"]
        == incidences["vertices"] - incidences["edges"] + incidences["faces"]
        # Retained samples are represented within the declared coverage.
        and coverage == policy["coverage_distance"]
        and 0.0 <= sample_max <= coverage
        # One declared repair patch whose excursion meets the preregistered bound.
        and unsupported > 0
        and patches == 1
        and 0.0 < unsupported_area < surface_area
        and diagonal > 0.0
        and 0.0 <= excursion <= bounds["maximum_repair_excursion_fraction"] * diagonal
        # Density variation measured on the supported repaired surface.
        and band + regular == counts["retained"]
        and band_area + regular_area <= surface_area - unsupported_area + 1.0e-9 * surface_area
        and _spacing_ratio(band, regular, band_area, regular_area)
        >= bounds["minimum_density_ratio"]
    )


def _volume_obligation(acceptance: object) -> bool:
    """Recompute the signed-volume closure from the two recorded volumes."""

    if not isinstance(acceptance, dict):
        return False
    mesh = _number(acceptance.get("mesh_volume"))
    boundary = _number(acceptance.get("source_boundary_volume"))
    smallest = _number(acceptance.get("minimum_cell_volume"))
    return (
        acceptance.get("embedding_status") == "certified"
        and acceptance.get("coverage_status") == "certified"
        and mesh is not None
        and boundary is not None
        and boundary > 0.0
        and abs(mesh - boundary) <= _VOLUME_CLOSURE
        and smallest is not None
        and smallest > 0.0
    )


def reconstruction_research_gate(
    record: dict[str, Any], args: QualificationControls
) -> _ReconstructionGate:
    """Recompute the research-gate verdict from the record's numeric evidence.

    The domain and its preregistered bounds are rederived from the controls
    and must equal the recorded declaration; every obligation is rechecked
    from measured quantities, so no nested pass flag or copied bound closes it.
    """

    domain = _reconstruction_qualification_domain(args.resolution)
    reconstruction = record.get("reconstruction")
    input_points = (
        reconstruction.get("input_points") if isinstance(reconstruction, dict) else None
    )
    epochs = record.get("epochs", [])
    adapted = [epoch.get("target_acceptance") for epoch in epochs[:-1]]
    transfers = [epoch.get("transfer") for epoch in epochs[:-1]]
    errors = [_number(epoch.get("l2_error")) for epoch in epochs]
    elapsed = _number(record.get("end_to_end_seconds"))
    checkpoint = record.get("checkpoint")
    if not isinstance(checkpoint, dict):
        checkpoint = {}
    cold_error = _number(checkpoint.get("continued_physical_L2_error"))
    density_defect = _number(checkpoint.get("density_max_defect"))
    history_defect = _number(checkpoint.get("history_max_defect"))
    obligations = {
        "preregistered_domain_and_case": record.get("acceptance_domain") == domain
        and record.get("reconstruction_case") == args.reconstruction_case
        and args.reconstruction_case in domain["registered_sources"],
        "native_repair_with_independent_geometry": _repair_obligation(
            record.get("independent_repair"),
            domain,
            args.reconstruction_case,
            input_points,
        ),
        "native_volume_exact_domain_every_epoch": _volume_obligation(
            record.get("independent_acceptance")
        )
        and len(adapted) == args.adaptation_rounds
        and all(_volume_obligation(item) for item in adapted),
        "consumer_transferred_state_physics": args.adaptation_rounds >= 1
        and len(epochs) == args.adaptation_rounds + 1
        and all(isinstance(item, dict) and item for item in transfers)
        and all(value is not None for value in errors)
        and errors[-1] is not None
        and errors[-1] <= args.target_error,
        "independent_cold_allfield_continuation": checkpoint.get("status") == "passed"
        and checkpoint.get("accepted_fields_bitwise_preserved") is True
        and checkpoint.get("field_names") == ["u", "density", "history"]
        and checkpoint.get("source_authority_stripped") is False
        and _volume_obligation(checkpoint.get("source_acceptance"))
        and cold_error is not None
        and cold_error <= args.target_error
        and density_defect is not None
        and 0.0 <= density_defect <= 1.0e-12
        and history_defect is not None
        and 0.0 <= history_defect <= 1.0e-12,
        "runtime_within_deadline": elapsed is not None and elapsed <= args.timeout,
    }
    complete = record.get("status") == "passed" and all(obligations.values())
    return {
        "status": "complete" if complete else "incomplete",
        "domain": domain["name"],
        "case": record.get("reconstruction_case"),
        "obligations": obligations,
    }


def _reconstruction_qualification_lifecycle(
    args: QualificationControls,
) -> dict[str, object]:
    """Repair underspecified defective samples, mesh natively, and solve on the result."""

    if (
        args.resolution < 4
        or args.capacity < 1
        or args.repeats < 1
        or args.adaptation_rounds < 1
        or not np.isfinite(args.timeout)
        or args.timeout <= 0.0
        or not np.isfinite(args.target_error)
        or args.target_error <= 0.0
    ):
        raise ValueError(
            "qualification.reconstruction.controls must have resolution >= 4, capacity >= 1, "
            "repeats >= 1, adaptation_rounds >= 1, and finite positive timeout and target_error; "
            f"got resolution={args.resolution!r}, capacity={args.capacity!r}, "
            f"repeats={args.repeats!r}, adaptation_rounds={args.adaptation_rounds!r}, "
            f"timeout={args.timeout!r}, target_error={args.target_error!r}"
        )
    started = perf_counter()
    deadline = Deadline(started + args.timeout)
    phases: list[NativeMeshingPhaseMeasurement] = []
    record: dict[str, Any] = {
        "scenario": "native-reconstruction-lifecycle",
        "reconstruction_case": args.reconstruction_case,
        "engine": "native",
        "status": "failed",
        "runtime": _runtime_identity(),
        "expected_admission": "positive",
        "targets": {"l2_error": args.target_error, "wall_seconds": args.timeout},
        "control_capacities": {
            "resolution": args.resolution,
            "entity_capacity": args.capacity,
            "adaptation_rounds": args.adaptation_rounds,
            "warm_repeats": args.repeats,
        },
        "acceptance_domain": _reconstruction_qualification_domain(args.resolution),
        "qualification_scale": {
            "resolution": args.resolution,
            "routine_resolution": 4,
            "role": "routine" if args.resolution == 4 else "larger",
            "combined_defects": "all admitted defects in every case",
            "samples": None,
            "cells": None,
            "dofs": None,
        },
        "epochs": [],
        "stages_seconds": {},
        "time_to_target_error_seconds": None,
        "mandatory_workflow_completion": "incomplete",
        "case_contract_completion": "incomplete",
        "independent_scientific_certification": "unassessed",
        "active_phase": "native_reconstruction",
        "nonclaims": [
            "Sampled reconstruction bounds are not continuous Hausdorff certificates.",
            "No analytic surface replaces the actual reconstructed triangles.",
            "The unsampled region is a declared repair, not recovered original geometry.",
        ],
    }
    _ledger = StageLedger(record, deadline=deadline)
    stages = record["stages_seconds"]
    try:
        fixture, measured = _ledger.run(
            record.get("active_phase", "_reconstruction_qualification_source"),
            lambda: _measured(
                lambda: _reconstruction_qualification_source(
                    args.reconstruction_case, args.resolution, args.capacity
                )
            ),
        )
        reconstructed, source, samples = fixture
        stages[record["active_phase"]] = measured["wall_seconds"]
        record["reconstruction"] = asdict(reconstructed.report)
        record["source_binding_id"] = source.binding_id
        record["source_id"] = source.source_id
        record["source_revision"] = source.source_revision
        record["qualification_scale"]["samples"] = int(samples.points.shape[0])
        record["independent_repair"], measured = _ledger.run(
            "independent_repair_acceptance",
            lambda: _measured(
                lambda: _reconstruction_qualification_repair(
                    reconstructed, source, samples
                )
            ),
        )
        stages[record["active_phase"]] = measured["wall_seconds"]

        specification, options = _ledger.run(
            "reconstructed_request_preparation",
            lambda: _reconstruction_qualification_request(
                source, args.capacity, args.timeout, args.adaptation_rounds
            ),
        )
        record["request_id"] = specification.specification_id
        record["native_options_id"] = options.options_id

        plan, measured = _ledger.run(
            "geometry_constraint_preparation",
            lambda: _measured(
                lambda: phx.meshing.NativeMeshingProvider(options).plan(
                    source,
                    specification,
                    coordinate_contract=phx.SpatialCoordinateContract.si(),
                    record_phase=phases.append,
                )
            ),
        )
        stages[record["active_phase"]] = measured["wall_seconds"]

        result, measured = _ledger.run(
            "generation_publication_certification",
            lambda: _measured(lambda: plan.execute(record_phase=phases.append)),
        )
        stages[record["active_phase"]] = measured["wall_seconds"]
        record["generation"] = _result_identity(result)

        def accept_result(candidate: phx.meshing.CellMeshingResult) -> dict[str, object]:
            return _reconstruction_qualification_acceptance(source, candidate)

        record["independent_acceptance"] = _ledger.run(
            "independent_acceptance", lambda: accept_result(result)
        )
        _envelope_qualification_epochs(
            source,
            result,
            specification,
            args,
            record,
            started,
            deadline=deadline,
            generation_options=options,
            accept_result=accept_result,
        )
        final_error = record["epochs"][-1]["l2_error"]
        passed = final_error <= args.target_error
        record.update(
            {
                "status": "passed" if passed else "failed",
                "physical_error": final_error,
                "target_error_reached": passed,
                "case_contract_completion": "complete" if passed else "incomplete",
                "independent_scientific_certification": "passed"
                if passed
                else "failed",
                "implemented_admitted_workflow": "native-underspecified-repair-volume-CompositionRebind-scalar-solve",
            }
        )
    except ReconstructionFailure as failure:
        record_failure(record, failure, record["active_phase"])
        record.update({"reconstruction": asdict(failure.report)})
    except phx.meshing.MeshingFailure as failure:
        record_failure(record, failure, record["active_phase"])
    except (RuntimeError, ValueError, TypeError, TimeoutError) as failure:
        record_failure(record, failure, record["active_phase"])
    finally:
        record["native_phases"] = _phase_recordings(phases)
        record["end_to_end_seconds"] = perf_counter() - started
        if record["end_to_end_seconds"] > args.timeout:
            record["status"] = "timeout"
        if record["status"] != "passed":
            record["case_contract_completion"] = "incomplete"
            if record["independent_scientific_certification"] == "passed":
                record["independent_scientific_certification"] = "unassessed"
        if record["epochs"]:
            final = record["epochs"][-1]
            record["qualification_scale"]["cells"] = final["result"]["cells"]
            record["qualification_scale"]["dofs"] = final["dofs"]
        gate = reconstruction_research_gate(record, args)
        record["research_gates"] = {_RESEARCH_GATE: gate}
        record["mandatory_workflow_completion"] = gate["status"]
        record["peak_process_resident_bytes"] = peak_resident_bytes()
        record["process_peak_scope"] = PEAK_RESIDENT_SCOPE
        record.pop("active_phase")
    return record


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return _reconstruction_qualification_lifecycle(args)
