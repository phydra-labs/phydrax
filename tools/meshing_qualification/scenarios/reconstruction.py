from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from pathlib import Path
from time import perf_counter
from typing import Literal

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._array_archive import ArrayArchiveLimits
from phydrax._execution_resources import MemoryBudgetError
from phydrax._execution_sampling import process_peak_memory
from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.reconstruction import (
    AuthoredTriangleSheet,
    reconstruct_surface_region,
    ReconstructionRobustness,
)
from phydrax.geometry.reconstruction._core import ReconstructionReport
from phydrax.geometry.simplicial import MeshRegion, TriangleBVH, TriangleMesh
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    SurfaceMeshingSpec,
    UniformSizeControl,
)
from phydrax.meshing._scope import MeshingScope
from phydrax.meshing._sizing import SizeControlStrength
from phydrax.meshing._volume_generation import NativeVolumeSchedule
from phydrax.meshing.providers import NativeMeshingProvider, NativePlcSource

from .._runtime import (
    accepted_result_record,
    require_meshcore,
    runtime_identity,
    single_cell_kind,
)


_CENTER = np.asarray((0.07, -0.03, 0.02), dtype=np.float64)
_CAP_DIRECTION = np.asarray((0.6, 0.0, 0.8), dtype=np.float64)
_CAP_DIRECTION /= np.linalg.norm(_CAP_DIRECTION)
_TORUS_RADII = (0.32, 0.17)
_ELLIPSOID_AXES = np.asarray((0.32, 0.24, 0.18), dtype=np.float64)


def _rotation() -> np.ndarray:
    axis = np.ones((3,), dtype=np.float64) / np.sqrt(3.0)
    angle = np.deg2rad(40.0)
    cross = np.asarray(
        ((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0)),
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def _directions(count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = np.arange(count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * index / count
    angle = np.pi * (3.0 - np.sqrt(5.0)) * index
    radial = np.sqrt(1.0 - z * z)
    return (
        np.stack((radial * np.cos(angle), radial * np.sin(angle), z), axis=1),
        angle,
        radial,
    )


class CorrectRoundingUndecided(ArithmeticError):
    """Ziv's strategy did not decide a correctly rounded value by its cap."""


# Ziv precisions (bits). mpmath's interval sin/cos of the exact argument at
# precision p is outward-rounded, so its enclosure is at most two p-bit ulps
# wide. Round to nearest is decided once that width falls below the result's
# distance to the nearest binary64 midpoint.
# Proven binary64 bounds on that distance (the Table Maker's Dilemma for
# sin/cos is solved):
# - |x| >= 2**10: V. Lefevre, T. Ly, P. Zimmermann, "Computing hard-to-round
#   cases of sin, cos, tan in double precision", ARITH 2026, Tables V-VI. The
#   hardest cases have m <= 68 (sin) and m <= 66 (cos) identical bits after
#   the round bit, so a midpoint is at least 2**-69 ulp away.
# - |x| < 2**11: CORE-MATH src/binary64/{sin,cos}/{sin,cos}.wc (complete sets
#   with >= 43 identical bits; sin.wc includes the three Koizumi cases;
#   sha256 8fb40c3d... / 40660955...). Measured over every listed argument,
#   the closest midpoint is 2**-60.06 ulp (sin 0x1.598bae9e632f6p-7) and
#   2**-59.78 ulp (cos 0x1.34ec2f9fc9c00p+1).
# So 53 + 1 + 68 + 2 (width) = 124 bits always decide. The 128-bit ceiling
# adds a 4-bit guard, and the typed refusal is unreachable for binary64 input.
_ZIV_PRECISIONS = (64, 96, 128)


def _correctly_rounded_sin_cos(
    function: Literal["sin", "cos"], values: np.ndarray, /
) -> np.ndarray:
    """Correctly rounded binary64 ``sin``/``cos`` (round to nearest, ties to even).

    The exact binary64 arguments first go through the host enclosure entry
    points ``phydrax.geometry._interval_enclosure.interval_sin`` /
    ``interval_cos``. They decide only the exact cases: under round to
    nearest only a width-0 enclosure decides directly,
    while a 1-ulp enclosure decides directed rounding but not RN. So there is
    no enclosure fast path. Every other argument is converted exactly to an
    mpmath float and evaluated in mpmath interval arithmetic at increasing
    precision (Ziv's strategy) until both endpoints round to the same binary64,
    independent of the platform libm.
    """
    from mpmath.libmp import from_float, round_nearest, to_float
    from mpmath.libmp.libmpi import mpi_cos, mpi_sin

    from phydrax.geometry._interval_enclosure import interval_cos, interval_sin

    arguments = np.asarray(values, dtype=np.float64)
    enclose = interval_sin if function == "sin" else interval_cos
    lower, upper = (
        np.asarray(endpoint, dtype=np.float64).ravel()
        for endpoint in enclose((arguments, arguments))
    )
    rounded = np.where(lower == upper, lower, np.nan)
    evaluate = mpi_sin if function == "sin" else mpi_cos
    flat = arguments.ravel()
    for index in np.flatnonzero(lower != upper).tolist():
        value = float(flat[index])
        if value == 0.0:
            # Exact: sin(+-0) = +-0 keeps its sign, cos(+-0) = 1.
            rounded[index] = value if function == "sin" else 1.0
            continue
        exact = from_float(value)
        for precision in _ZIV_PRECISIONS:
            low, high = evaluate((exact, exact), precision)
            candidate = to_float(low, rnd=round_nearest)
            if candidate == to_float(high, rnd=round_nearest):
                rounded[index] = candidate
                break
        else:
            raise CorrectRoundingUndecided(
                f"{function}({value!r}) is undecided at {_ZIV_PRECISIONS[-1]} bits."
            )
    return rounded.reshape(arguments.shape)


def _surface_samples(kind: str, resolution: int, rng: np.random.Generator) -> np.ndarray:
    samples_per_area = 16.0 / (4.0 * np.pi * 0.2**2)
    if kind == "lobed-sphere":
        count = 16 * resolution * resolution
    elif kind == "torus":
        major, minor = _TORUS_RADII
        area = 4.0 * np.pi**2 * major * minor
        count = int(round(samples_per_area * resolution * resolution * area))
    elif kind == "rotated-ellipsoid":
        a, b, c = _ELLIPSOID_AXES
        p = 1.6075
        area = (
            4.0
            * np.pi
            * (((a * b) ** p + (a * c) ** p + (b * c) ** p) / 3.0) ** (1.0 / p)
        )
        count = int(round(samples_per_area * resolution * resolution * area))
    else:
        raise ValueError(f"Unknown reconstruction source {kind!r}.")
    if kind == "lobed-sphere":
        direction, angle, radial = _directions(count)
        radius = 0.2 * (1.0 + 0.04 * np.sin(3.0 * angle) * radial * radial)
        return direction * radius[:, None]
    if kind == "torus":
        major, minor = _TORUS_RADII
        accepted: list[np.ndarray] = []
        total = 0
        while total < count:
            u, v, keep = rng.random(size=(3, 4 * count))
            u, v = 2.0 * np.pi * u, 2.0 * np.pi * v
            # Platform libm sin/cos differ in the last ulp across OSes; the
            # sample points use the correctly rounded values instead.
            cos_v = _correctly_rounded_sin_cos("cos", v)
            ring = major + minor * cos_v
            selected = keep < ring / (major + minor)
            block = np.stack(
                (
                    ring * _correctly_rounded_sin_cos("cos", u),
                    ring * _correctly_rounded_sin_cos("sin", u),
                    minor * _correctly_rounded_sin_cos("sin", v),
                ),
                axis=1,
            )[selected]
            accepted.append(block)
            total += block.shape[0]
        return np.concatenate(accepted)[:count]
    if kind == "rotated-ellipsoid":
        rotation = _rotation()
        accepted = []
        total = 0
        while total < count:
            direction = rng.standard_normal(size=(2 * count, 3))
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            weight = np.min(_ELLIPSOID_AXES) * np.linalg.norm(
                direction / _ELLIPSOID_AXES, axis=1
            )
            chosen = rng.random(size=weight.shape) < weight
            block = (direction[chosen] * _ELLIPSOID_AXES) @ rotation.T
            accepted.append(block)
            total += block.shape[0]
        return np.concatenate(accepted)[:count]
    raise ValueError(f"Unknown reconstruction source {kind!r}.")


def _defective_samples(
    kind: str, resolution: int, seed: int
) -> tuple[np.ndarray, dict[str, object]]:
    # Draws use random/standard_normal with separate NumPy arithmetic:
    # Generator.uniform/normal fuse their affine map on macOS, which changes
    # samples by an ulp across operating systems.
    rng = np.random.default_rng(seed)
    local = _surface_samples(kind, resolution, rng)
    count = local.shape[0]
    noisy = local + 0.0008 * rng.standard_normal(size=local.shape) + _CENTER
    cap_center = local[int(np.argmax(local @ _CAP_DIRECTION))]
    cap = np.linalg.norm(local - cap_center, axis=1) <= 0.2
    kept = ~cap
    x_extent = float(np.max(np.abs(local[:, 0])))
    low_density = local[:, 0] < -0.5 * x_extent
    alternating = (np.arange(count) % 2) == 0
    kept &= ~low_density | alternating
    observed = noisy[kept]
    outlier_count = max(8, count // 32)
    directions = rng.standard_normal(size=(outlier_count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    maximum_radius = float(np.max(np.linalg.norm(local, axis=1)))
    low, high = maximum_radius + 0.35, maximum_radius + 0.55
    radii = low + (high - low) * rng.random(size=(outlier_count, 1))
    outliers = _CENTER + directions * radii
    return np.concatenate((observed, outliers)), {
        "seed": seed,
        "source_samples": count,
        "missing_cap_samples": int(np.count_nonzero(cap)),
        "density_band_samples": int(np.count_nonzero(low_density)),
        "retained_before_noise": int(observed.shape[0]),
        "injected_outliers": outlier_count,
        "noise_standard_deviation": 0.0008,
        "cap_direction": _CAP_DIRECTION.tolist(),
        "cap_radius": 0.2,
        "density_drop": "every second sample in x < -0.5 max(|x|)",
    }


def _source_deviation(kind: str, coordinates: np.ndarray) -> np.ndarray:
    local = coordinates - _CENTER
    if kind == "lobed-sphere":
        radius = np.linalg.norm(local, axis=1)
        angle = np.arctan2(local[:, 1], local[:, 0])
        planar_radius_squared = local[:, 0] ** 2 + local[:, 1] ** 2
        polar_factor = np.divide(
            planar_radius_squared,
            radius**2,
            out=np.zeros_like(radius),
            where=radius > 0,
        )
        expected = 0.2 * (1.0 + 0.04 * np.sin(3.0 * angle) * polar_factor)
        return np.abs(radius - expected)
    if kind == "torus":
        major, minor = _TORUS_RADII
        ring_distance = np.sqrt(local[:, 0] ** 2 + local[:, 1] ** 2) - major
        return np.abs(np.sqrt(ring_distance**2 + local[:, 2] ** 2) - minor)
    if kind == "rotated-ellipsoid":
        local = local @ _rotation()
        scaled_radius = np.linalg.norm(local / _ELLIPSOID_AXES, axis=1)
        return np.abs(scaled_radius - 1.0) * float(np.max(_ELLIPSOID_AXES))
    raise ValueError(f"Unknown reconstruction source {kind!r}.")


def _independent_reconstruction_evidence(
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    report: ReconstructionReport,
) -> dict[str, float | int]:
    """Measure the declared outlier and sample support contracts independently."""
    means = np.empty(points.shape[0], dtype=np.float64)
    for start in range(0, len(points), 256):
        block = points[start : start + 256]
        distances = np.sqrt(
            np.sum((block[:, None, :] - points[None, :, :]) ** 2, axis=-1)
        )
        distances[np.arange(len(block)), start + np.arange(len(block))] = np.inf
        means[start : start + len(block)] = np.mean(
            np.partition(distances, 15, axis=1)[:, :16], axis=1
        )
    threshold = float(np.mean(means) + 2.0 * np.std(means))
    removed = np.flatnonzero(means > threshold)
    evidence = report.outlier_evidence
    if (
        evidence is None
        or report.coverage_evidence is None
        or report.deviation_evidence is None
    ):
        raise RuntimeError(
            "Reconstruction omitted required independent acceptance operands."
        )
    if not np.array_equal(removed, np.asarray(evidence.removed_indices)) or abs(
        threshold - evidence.threshold
    ) > 1.0e-12 * max(1.0, abs(threshold)):
        raise RuntimeError(
            "Independent statistical outlier rule disagrees with reconstruction."
        )
    retained = np.delete(points, removed, axis=0)
    bvh = TriangleBVH(TriangleMesh(vertices, faces))
    sample_distance = np.asarray(bvh.query(retained).distance, dtype=np.float64)
    removed_distance = np.asarray(bvh.query(points[removed]).distance, dtype=np.float64)
    coverage = report.coverage_evidence
    within = int(np.count_nonzero(removed_distance <= coverage.coverage_distance))
    beyond = int(removed.size - within)
    supported = len(retained) + within
    if (
        beyond == 0
        or within > 0.05 * supported
        or np.max(sample_distance) > coverage.coverage_distance
    ):
        raise RuntimeError("Reconstruction violates supported-sample loss or coverage.")
    if (
        abs(
            float(np.max(sample_distance))
            - report.deviation_evidence.sample_to_surface_max
        )
        > 1.0e-12
    ):
        raise RuntimeError(
            "Independent sample-to-surface deviation disagrees with report."
        )

    def nearest(queries: np.ndarray) -> np.ndarray:
        result = np.empty(len(queries), dtype=np.float64)
        for start in range(0, len(queries), 256):
            delta = queries[start : start + 256, None, :] - retained[None, :, :]
            result[start : start + len(delta)] = np.sqrt(
                np.min(np.sum(delta * delta, axis=-1), axis=1)
            )
        return result

    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    centroid_distance = nearest(triangles.mean(axis=1))
    vertex_distance = nearest(vertices)
    unsupported = (centroid_distance > coverage.coverage_distance) & np.any(
        vertex_distance[faces] > coverage.coverage_distance, axis=1
    )

    def components(rows: np.ndarray) -> int:
        parent = np.arange(len(vertices), dtype=np.int64)

        def root(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = int(parent[index])
            return index

        for a, b, c in rows.tolist():
            for first, second in ((a, b), (b, c)):
                left, right = root(first), root(second)
                if left != right:
                    parent[max(left, right)] = min(left, right)
        return len({root(int(index)) for index in np.unique(rows)})

    directed = faces[:, ((0, 1), (1, 2), (2, 0))].reshape((-1, 2))
    _, directed_counts = np.unique(directed, axis=0, return_counts=True)
    _, edge_counts = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
    euler = len(np.unique(faces)) - len(edge_counts) + len(faces)
    if (
        np.any(directed_counts != 1)
        or np.any(edge_counts != 2)
        or components(faces) != 1
        or euler != report.euler_characteristic
    ):
        raise RuntimeError(
            "Independent reconstructed-surface incidence disagrees with the topology report."
        )
    unsupported_area = float(np.sum(areas[unsupported]))
    surface_area = float(np.sum(areas))
    excursion = float(max(np.max(centroid_distance), np.max(vertex_distance)))
    diagonal = float(np.linalg.norm(np.max(retained, axis=0) - np.min(retained, axis=0)))
    if (
        coverage.unsupported_patches != 1
        or not np.any(unsupported)
        or components(faces[unsupported]) != 1
        or excursion > 0.4 * diagonal
    ):
        raise RuntimeError("Reconstruction repair violates patch or excursion contract.")
    if (
        abs(unsupported_area - coverage.unsupported_area) > 1.0e-9 * surface_area
        or abs(
            float(np.max(vertex_distance))
            - report.deviation_evidence.surface_to_sample_max
        )
        > 1.0e-12
    ):
        raise RuntimeError(
            "Independent surface support or deviation disagrees with report."
        )
    relative = retained[:, 0] - _CENTER[0]
    cut = -0.5 * float(np.max(np.abs(relative)))
    threshold_x = Fraction(float(_CENTER[0])) + Fraction(cut)
    band_area = Fraction(0)
    regular_area = Fraction(0)
    for triangle, area, active in zip(triangles, areas, ~unsupported, strict=True):
        if not active or area == 0.0:
            continue
        distances = tuple(Fraction(float(point[0])) - threshold_x for point in triangle)
        inside = tuple(i for i, distance in enumerate(distances) if distance < 0)
        if len(inside) in (0, 3):
            fraction = Fraction(len(inside) // 3)
        else:
            isolated = (
                inside[0]
                if len(inside) == 1
                else next(i for i in range(3) if i not in inside)
            )
            fraction = Fraction(1)
            for other in range(3):
                if other != isolated:
                    fraction *= distances[isolated] / (
                        distances[isolated] - distances[other]
                    )
            if len(inside) == 2:
                fraction = 1 - fraction
        band_area += Fraction(float(area)) * fraction
        regular_area += Fraction(float(area)) * (1 - fraction)
    band_count = int(np.count_nonzero(relative < cut))
    regular_count = len(retained) - band_count
    if min(band_count, regular_count, band_area, regular_area) <= 0:
        raise RuntimeError("Reconstruction does not exercise both density regions.")
    density_ratio = float(
        np.sqrt((regular_count / float(regular_area)) / (band_count / float(band_area)))
    )
    if density_ratio < 1.2:
        raise RuntimeError("Reconstruction does not meet the density variation contract.")
    return {
        "independent_outlier_threshold": threshold,
        "removed_within_coverage": within,
        "removed_beyond_coverage": beyond,
        "supported_discard_fraction": within / supported,
        "retained_sample_to_surface_max": float(np.max(sample_distance)),
        "model_implied_repair_area": unsupported_area,
        "sample_supported_model_area": surface_area - unsupported_area,
        "repair_excursion": excursion,
        "retained_sample_diagonal": diagonal,
        "density_spacing_ratio": density_ratio,
    }


# Donor c56614a9f research gate: one complete marked refinement (all cells),
# the manufactured |x|^2 Poisson problem, and the 0.05 L2 target, unchanged.
_TARGET_L2_ERROR = 0.05
_ADAPTATION_ROUNDS = 1
_FIELD_NAMES = ("u", "density", "history")
_DENSITY = 2.0
_VOLUME_CLOSURE = 1.0e-12
_INVENTORY_TOLERANCE = 1.0e-10
_RESTORED_FIELD_TOLERANCE = 1.0e-12
_PROCESS_STARTED = perf_counter()
_STAGE_LOG = "PHYDRAX_RECONSTRUCTION_STAGE_LOG"
# Declared symbolic work (ordering, elimination tree, counts) of every
# manufactured Cholesky analysis: the largest measured reconstruction
# operator (refined lobed-sphere-2, 18,665 DOFs) needs 4,607,569 under minimum
# degree; 2**27 leaves ~29x for the larger unmeasured rows.
_SYMBOLIC_WORK_BUDGET = 1 << 27


def _stage(profile: str, stage: str, /) -> None:
    """Durably append one stage record (name, wall, peak RSS) before it starts.

    Entry points set the log path; writer and reader processes inherit it, so
    a failure or guard kill names its stage and the operands captured so far.
    """
    path = os.environ.get(_STAGE_LOG)
    if not path:
        raise RuntimeError(f"Reconstruction workflows require {_STAGE_LOG}.")
    peak = process_peak_memory()
    record = {
        "profile": profile,
        "stage": stage,
        "pid": os.getpid(),
        "elapsed_seconds": perf_counter() - _PROCESS_STARTED,
        "peak_rss_bytes": peak.value_bytes,
        "peak_rss_unavailable": peak.unavailable_reason,
    }
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _stage_log(directory: str, /) -> str:
    """Bind the durable stage log of one workflow run under its directory."""
    path = os.environ.get(_STAGE_LOG)
    if not path:
        path = str(Path(directory) / "stages.jsonl")
        os.environ[_STAGE_LOG] = path
    return path


def _manufactured_exact(points: np.ndarray, /) -> np.ndarray:
    return np.sum(np.square(points), axis=-1)


def _unit_tetrahedron_rule() -> tuple[np.ndarray, np.ndarray]:
    """Degree-four Duffy rule on the unit tetrahedron, normalized to unit mass."""
    from phydrax._polynomial._cubature import duffy_simplex_rule_data

    rule = duffy_simplex_rule_data("tetrahedron", 4)
    points = np.asarray(rule.points, dtype=np.float64)
    weights = np.asarray(rule.weights, dtype=np.float64)
    if (
        rule.exact_degree < 4
        or points.ndim != 2
        or points.shape[1] != 3
        or np.any(points < 0.0)
        or np.any(np.sum(points, axis=1) > 1.0)
        or np.any(weights <= 0.0)
    ):
        raise RuntimeError("The L2 oracle requires an exact positive tetrahedral rule.")
    return points, weights / np.sum(weights)


def _tetrahedra(mesh: phx.discretization.CellMesh, /) -> np.ndarray:
    single_cell_kind(mesh, "tetrahedron")
    return np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
    )


def _cell_volumes(mesh: phx.discretization.CellMesh, /) -> np.ndarray:
    corners = np.asarray(mesh.coordinates, dtype=np.float64)[_tetrahedra(mesh)]
    return np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0


def _vertex_values(
    space: phx.discretization.FiniteElementDiscretization,
    mesh: phx.discretization.CellMesh,
    values: np.ndarray,
    /,
) -> np.ndarray:
    """P1 coefficients in mesh-vertex order, refusing any other DOF layout."""
    dof_map = space.dof_maps[0]
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    if (
        dof_map.association != "vertex"
        or dof_map.global_dof_count != coordinates.shape[0]
        or not np.array_equal(np.asarray(dof_map.dof_coordinates), coordinates)
        or np.asarray(values).shape != (coordinates.shape[0],)
    ):
        raise ValueError("The manufactured P1 oracle requires the vertex DOF layout.")
    return np.asarray(values, dtype=np.float64)


def _p1_integral(mesh: phx.discretization.CellMesh, values: np.ndarray, /) -> float:
    cells = _tetrahedra(mesh)
    return float(np.sum(_cell_volumes(mesh) * np.mean(values[cells], axis=1)))


def _quadratic_l2_error(
    mesh: phx.discretization.CellMesh, values: np.ndarray, /
) -> float:
    """Exact (degree-four quadrature) L2 distance of a P1 field to |x|^2."""
    points, weights = _unit_tetrahedron_rule()
    barycentric = np.column_stack((1.0 - np.sum(points, axis=1), points))
    cells = _tetrahedra(mesh)
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    total = 0.0
    for start in range(0, cells.shape[0], 8192):
        block = cells[start : start + 8192]
        corners = coordinates[block]
        physical = np.einsum("qk,ckd->cqd", barycentric, corners)
        approximate = np.einsum("qk,ck->cq", barycentric, values[block])
        volume = np.abs(np.linalg.det(corners[:, 1:] - corners[:, :1])) / 6.0
        total += float(
            np.sum(
                volume[:, None]
                * weights[None, :]
                * np.square(approximate - _manufactured_exact(physical))
            )
        )
    return math.sqrt(total)


def _manufactured_problem(
    result: phx.meshing.CellMeshingResult, form_id: str, /
) -> tuple[
    phx.discretization.FiniteElementDiscretization,
    phx.equations.CompiledFiniteElementProblem,
    phx.equations.FiniteElementForm,
]:
    """-div grad u = -6 with Dirichlet u = |x|^2: the donor manufactured problem."""
    field = phx.discretization.FiniteElementFieldSpec(
        "u",
        phx.discretization.lagrange_element(
            single_cell_kind(result.mesh, "tetrahedron"), 1
        ),
    )
    space = phx.discretization.FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry
    ).prepare()
    form = phx.equations.FiniteElementForm(
        form_id,
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction("u", -6.0),
        ),
        # The Dirichlet-constrained P1 Laplacian is symmetric positive definite.
        properties=phx.linalg.OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={"positive_definite": "asserted"},
        ),
    )
    problem = phx.equations.compile_finite_element_problem(
        form,
        space,
        constraint=phx.discretization.dirichlet_constraint(space, "u"),
        dirichlet_values=lambda values: jnp.sum(values * values, axis=-1),
        execution_policy=phx.equations.FiniteElementExecutionPolicy(realization="sparse"),
    )
    return space, problem, form


@dataclass(frozen=True, slots=True)
class _ManufacturedSolve:
    """Operation-local fresh solve; never persisted."""

    space: phx.discretization.FiniteElementDiscretization
    problem: phx.equations.CompiledFiniteElementProblem
    form: phx.equations.FiniteElementForm
    solution: phx.linalg.LinearSolveResult
    check: phx.linalg.LinearSolveCheckEvidence
    values: np.ndarray
    l2_error: float
    record: dict[str, object]


def _manufactured_policy(budget: int, /) -> phx.linalg.LinearSolvePolicy:
    """PCG on the exact supernodal Cholesky factor under one declared byte budget.

    Every solve and factorization budget is the profile's declared execution
    budget: all fields are static, so they enter plan_id, and the writer and
    the cold reader prepare under the same declaration. The factor's
    structural nonzeros are bounded by the same bytes (one float64 each).
    Minimum degree is the ordering: on the refined lobed-sphere-2 operator
    (18,665 DOFs) it needs 2,346,552 factor nonzeros and 0.95 GFLOP, nested
    dissection 6,647,037 and 8.9 GFLOP.
    """
    return phx.linalg.LinearSolvePolicy(
        phx.linalg.PCG(),
        tolerance=phx.linalg.TolerancePolicy(relative=1.0e-14, absolute=0.0),
        preconditioning=phx.linalg.PreconditioningPolicy(
            phx.linalg.SparseFactorizationPreconditionerBuilder(
                phx.linalg.SparseFactorizationPolicy(
                    "cholesky",
                    ordering="approximate-minimum-degree",
                    max_factor_nnz=budget // np.dtype(np.float64).itemsize,
                    max_factor_bytes=budget,
                    max_symbolic_work=_SYMBOLIC_WORK_BUDGET,
                )
            )
        ),
        resources=phx.linalg.SolveResourcePolicy(
            factorization_bytes=budget,
            workspace_bytes=budget,
            krylov_basis_bytes=budget,
            preconditioner_bytes=budget,
            recycling_state_bytes=budget,
        ),
    )


def _manufactured_solve(
    result: phx.meshing.CellMeshingResult, form_id: str, budget: int, /
) -> _ManufacturedSolve:
    """Fresh P1 solve with an independent exact L2 error against |x|^2."""
    space, problem, form = _manufactured_problem(result, form_id)
    operator, rhs = problem.linear_system(storage="sparse")
    prepared = phx.linalg.prepare(operator, _manufactured_policy(budget))
    state = prepared.preconditioning_state
    if state is None or not isinstance(
        state.action, phx.linalg.SparseFactorizationPreconditioner
    ):
        raise RuntimeError("The SPD solve did not retain its Cholesky factorization.")
    factor = state.action.factorization.plan
    solved, check = phx.linalg.solve_checked(prepared, rhs)
    values = _vertex_values(space, result.mesh, np.asarray(problem.expand(solved.value)))
    residual = float(jnp.linalg.norm(operator.operator.mv(solved.value) - rhs))
    rhs_norm = float(jnp.linalg.norm(rhs))
    if (
        not bool(np.all(np.asarray(check.valid)))
        or not bool(jnp.all(solved.successful))
        or residual > 2.0e-8 * max(1.0, rhs_norm)
    ):
        raise RuntimeError(
            f"{form_id} manufactured solve failed: {residual=}, {rhs_norm=}."
        )
    l2_error = _quadratic_l2_error(result.mesh, values)
    return _ManufacturedSolve(
        space,
        problem,
        form,
        solved,
        check,
        values,
        l2_error,
        {
            "form": form.form_id,
            "dofs": int(values.size),
            "l2_error": l2_error,
            "normalized_residual": residual / max(1.0, rhs_norm),
            "solver": {
                "backend": str(prepared.plan.backend),
                "method": str(prepared.plan.method),
                "status": int(np.asarray(solved.status)),
                "iterations": int(np.asarray(solved.diagnostics.iterations)),
                "plan_id": prepared.plan.plan_id,
                "memory_budget_bytes": budget,
                "factorization": "supernodal-cholesky",
                "factor_nnz": factor.factor_nnz,
                "factor_bytes": factor.factor_bytes,
                "symbolic_work": factor.symbolic_work,
            },
        },
    )


def _volume_acceptance(
    plc: NativePlcSource, result: phx.meshing.CellMeshingResult, /
) -> dict[str, object]:
    """Strict native acceptance, exact PLC strata and independent volume closure."""
    certificate = result.certification
    if (
        certificate is None
        or not certificate.passed
        or not result.audit.passed
        or not result.compliance.passed
    ):
        raise RuntimeError("The reconstructed volume requires strict native acceptance.")
    phx.meshing.PlcAssociationTransfer(plc).validate_source(result)
    faces = np.asarray(plc.complex.polygon_vertices, dtype=np.int64).reshape((-1, 3))
    skin = np.asarray(plc.complex.vertices, dtype=np.float64)[faces]
    boundary = float(np.sum(skin[:, 0] * np.cross(skin[:, 1], skin[:, 2])) / 6.0)
    volumes = _cell_volumes(result.mesh)
    volume = float(np.sum(volumes))
    if (
        np.any(volumes <= 0.0)
        or boundary <= 0.0
        or abs(volume - boundary) > _VOLUME_CLOSURE
    ):
        raise RuntimeError(
            f"Reconstructed PLC fill violates signed-volume closure: {boundary=}, {volume=}."
        )
    return {
        "result_id": result.result_id,
        "certification_id": certificate.report_id,
        "certified": True,
        "audit_passed": True,
        "compliance_passed": True,
        "plc_strata_verified": True,
        "source_boundary_volume": boundary,
        "mesh_volume": volume,
        "volume_defect": volume - boundary,
        "minimum_cell_volume": float(np.min(volumes)),
        "cells": int(volumes.size),
    }


def _refinement_admission(mesh: phx.discretization.CellMesh, budget: int, /) -> int:
    """Admission estimate of one all-cell refinement, refused above ``budget``.

    Every tetrahedron red-refines into eight children; the estimate charges the
    stage's measured resident growth per child (_REFINEMENT_BYTES_PER_TARGET_CELL).
    The refusal is typed and comes before any adaptation work is allocated.
    """
    marked = sum(block.cell_count for block in mesh.blocks)
    single_cell_kind(mesh, "tetrahedron")
    required = (
        marked * _REFINED_CHILDREN_PER_TETRAHEDRON * _REFINEMENT_BYTES_PER_TARGET_CELL
    )
    if required > budget:
        raise MemoryBudgetError("refinement_memory_bytes", required, budget)
    return required


def _refine_every_cell(
    plc: NativePlcSource, result: phx.meshing.CellMeshingResult, budget: int, /
) -> tuple[phx.meshing.MeshAdaptationResult, int]:
    admitted = _refinement_admission(result.mesh, budget)
    marks = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in result.mesh.blocks]
    )
    prepared = phx.meshing.prepare_mesh_adaptation(
        result,
        phx.meshing.MarkedMeshAdaptation(marks, np.empty((0,), dtype=np.int64)),
        policy=phx.meshing.MeshAdaptationPolicy(
            phx.meshing.MeshAdaptationRoute.NATIVE_MIXED,
            association_transfer=phx.meshing.PlcAssociationTransfer(plc),
            audit_policy=phx.meshing.CellMeshAuditPolicy(
                require_complete_association=True,
                watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT,
            ),
        ),
    )
    adaptation = phx.meshing.execute_mesh_adaptation(prepared)
    if (
        adaptation.status is not phx.meshing.MeshAdaptationStatus.COMPLETE
        or adaptation.transition is None
    ):
        raise RuntimeError(
            f"Marked reconstructed-volume refinement was not accepted: {adaptation.status}."
        )
    return adaptation, admitted


def _all_field_layouts(
    mesh: phx.discretization.CellMesh, /
) -> tuple[phx.solver.FiniteElementAcceptedFieldLayout, ...]:
    # Scalar P1 Lagrange fields carry no form basis; nested refinement keeps the
    # coarse space inside the fine one, so nodal interpolation is the exact
    # embedding and the inventory checks below measure it independently.
    element = phx.discretization.lagrange_element(
        single_cell_kind(mesh, "tetrahedron"), 1
    )
    semantics = (
        phx.solver.FiniteElementFieldTransferSemantics.NONCONSERVATIVE_INTERPOLATION
    )
    return tuple(
        phx.solver.FiniteElementAcceptedFieldLayout(
            phx.discretization.FiniteElementFieldSpec(name, element),
            transfer_semantics=semantics,
        )
        for name in _FIELD_NAMES
    )


def _all_field_space(
    result: phx.meshing.CellMeshingResult, /
) -> phx.discretization.FiniteElementDiscretization:
    return phx.discretization.FiniteElementPlan(
        result.mesh,
        tuple(layout.field_spec for layout in _all_field_layouts(result.mesh)),
        coordinate_spec=result.geometry,
    ).prepare()


def _auxiliary_defects(
    mesh: phx.discretization.CellMesh, density: np.ndarray, history: np.ndarray, /
) -> tuple[float, float]:
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    return (
        float(np.max(np.abs(np.asarray(density) - _DENSITY), initial=0.0)),
        float(
            np.max(np.abs(np.asarray(history) - (coordinates[:, 0] + 0.5)), initial=0.0)
        ),
    )


@dataclass(slots=True)
class _CarriedState:
    """Operation-local measurements of the transferred fields before refresh."""

    u_inventory: float | None = None
    density_inventory: float | None = None
    transferred_l2_error: float | None = None
    fresh: _ManufacturedSolve | None = None


def _transfer_accepted_state(
    source: phx.meshing.CellMeshingResult,
    source_values: np.ndarray,
    source_form: phx.equations.FiniteElementForm,
    adaptation: phx.meshing.MeshAdaptationResult,
    label: str,
    budget: int,
    /,
) -> tuple[dict[str, object], phx.solver.FiniteElementTopologyResult, _ManufacturedSolve]:
    """Carry actual source fields without retaining a completed solve's workspace."""
    target = adaptation.target
    layouts = _all_field_layouts(source.mesh)
    source_space = _all_field_space(source)
    target_space = _all_field_space(target)
    state_form = phx.equations.FiniteElementForm(
        f"{label}-state", _FIELD_NAMES, source_form.actions
    )
    source_state = phx.equations.compile_finite_element_problem(state_form, source_space)
    target_state = phx.equations.compile_finite_element_problem(state_form, target_space)
    coordinates = np.asarray(source.mesh.coordinates, dtype=np.float64)
    accepted = phx.solver.FiniteElementAcceptedState(
        (
            jnp.asarray(source_values),
            jnp.full((coordinates.shape[0],), _DENSITY),
            jnp.asarray(coordinates[:, 0] + 0.5),
        ),
        0.0,
        0,
        source.mesh.topology_id,
        source_space.prepared_id,
        source_state.compilation_id,
        field_layouts=layouts,
    )
    source_inventory = _p1_integral(source.mesh, source_values)
    source_density = _p1_integral(source.mesh, np.full(coordinates.shape[0], _DENSITY))
    carried = _CarriedState()

    def reanalyze(
        candidate_mesh: phx.discretization.CellMesh,
        fields: tuple[object, ...],
        materials: object,
        candidate: phx.meshing.MeshAdaptationResult,
        args: object,
    ) -> phx.solver.FiniteElementReanalysisResult:
        del materials, args
        if (
            candidate_mesh.mesh_id != target.mesh.mesh_id
            or candidate.target.result_id != target.result_id
        ):
            raise ValueError("Reanalysis must bind the accepted refined result.")
        u = np.asarray(fields[0], dtype=np.float64)
        carried.u_inventory = _p1_integral(candidate_mesh, u)
        carried.density_inventory = _p1_integral(
            candidate_mesh, np.asarray(fields[1], dtype=np.float64)
        )
        carried.transferred_l2_error = _quadratic_l2_error(candidate_mesh, u)
        fresh = _manufactured_solve(target, f"{label}-reanalysis", budget)
        carried.fresh = fresh
        return phx.solver.FiniteElementReanalysisResult(
            fresh.problem,
            fresh.solution,
            fresh.check,
            "u",
            state_problem=target_state,
            check_policy=phx.linalg.LinearSolveCheckPolicy(),
        )

    def certify(
        candidate_mesh: phx.discretization.CellMesh,
        fields: tuple[object, ...],
        materials: object,
        lineage: object,
        args: object,
    ) -> bool:
        del materials, lineage, args
        fresh = carried.fresh
        if (
            candidate_mesh.mesh_id != target.mesh.mesh_id
            or fresh is None
            or carried.u_inventory is None
            or carried.density_inventory is None
            or len(fields) != len(_FIELD_NAMES)
        ):
            return False
        u, density, history = (np.asarray(value, dtype=np.float64) for value in fields)
        density_defect, history_defect = _auxiliary_defects(
            candidate_mesh, density, history
        )
        return bool(
            np.array_equal(u, fresh.values)
            and density_defect <= _RESTORED_FIELD_TOLERANCE
            and history_defect <= _RESTORED_FIELD_TOLERANCE
            and abs(carried.u_inventory - source_inventory)
            <= _INVENTORY_TOLERANCE * max(1.0, abs(source_inventory))
            and abs(carried.density_inventory - source_density)
            <= _INVENTORY_TOLERANCE * max(1.0, abs(source_density))
        )

    transaction = phx.solver.FiniteElementTopologyTransaction(certify)
    unrefreshed = transaction.execute(accepted, source.mesh, adaptation)
    if bool(np.asarray(unrefreshed.committed)) or unrefreshed.state is not accepted:
        raise RuntimeError(
            "A transferred state without physical reanalysis was accepted."
        )
    staged = transaction.execute(
        accepted, source.mesh, adaptation, reanalysis_action=reanalyze
    )
    fresh = carried.fresh
    if (
        not bool(np.asarray(staged.committed))
        or staged.reanalysis is None
        or fresh is None
        or carried.u_inventory is None
        or carried.density_inventory is None
    ):
        raise RuntimeError(
            f"All-field transfer and reanalysis refused: {staged.diagnostics}"
        )
    density_defect, history_defect = _auxiliary_defects(
        target.mesh,
        np.asarray(staged.state.fields[1]),
        np.asarray(staged.state.fields[2]),
    )
    record = {
        "field_names": list(_FIELD_NAMES),
        "transfer_semantics": [layout.transfer_semantics.value for layout in layouts],
        "source_accepted_id": accepted.accepted_id,
        "target_accepted_id": staged.state.accepted_id,
        "field_transfer_ids": [item.transfer_id for item in staged.field_transfers],
        "unrefreshed_refusal": str(unrefreshed.diagnostics),
        "reanalysis_evidence_id": staged.reanalysis.evidence_id,
        "state_inventory": {
            "source": source_inventory,
            "target": carried.u_inventory,
            "defect": carried.u_inventory - source_inventory,
        },
        "density_inventory_defect": carried.density_inventory - source_density,
        "transferred_state_l2_error": carried.transferred_l2_error,
        "density_max_defect": density_defect,
        "history_max_defect": history_defect,
    }
    return record, staged, fresh


def _reconstruct_case(
    kind: str, scale: int, archive_directory: str, /
) -> dict[str, object]:
    """Writer process of one original profile; returns before the cold reader starts."""
    _stage(_profile_id(kind, scale), "reconstruction")
    resolution = 4 * scale
    spacing = 0.44 / resolution
    coverage_distance = spacing
    cap_radius = 0.2
    seed = {"lobed-sphere": 1709, "torus": 1711, "rotated-ellipsoid": 1713}[kind]
    points, input_record = _defective_samples(kind, resolution, seed)
    capture_directory = os.environ.get("PHYDRAX_RECONSTRUCTION_CAPTURE_DIR")
    if capture_directory:
        np.savez_compressed(
            Path(capture_directory) / f"{kind}-resolution-{resolution}-input.npz",
            points=points,
            seed=np.asarray(seed),
            resolution=np.asarray(resolution),
            sample_spacing=np.asarray(spacing),
            coverage_distance=np.asarray(coverage_distance),
            thin_feature_distance=np.asarray(0.22 / resolution),
        )
    robustness = ReconstructionRobustness(
        outlier_std_ratio=2.0,
        coverage_distance=coverage_distance,
        incomplete_sampling="close",
        thin_feature_distance=0.22 / resolution,
    )
    source = reconstruct_surface_region(
        points,
        normal_orientation="propagate",
        recenter=False,
        neighborhood_size=16,
        sample_spacing=spacing,
        screening=4.0,
        discretization="octree",
        robustness=robustness,
        feature_id="qualification-native-underspecified-defective-samples",
    )
    report = source.report
    region = source.source
    if not isinstance(region, MeshRegion):
        raise TypeError("Reconstruction source lost its triangular MeshRegion.")
    if capture_directory:
        np.savez_compressed(
            Path(capture_directory) / f"{kind}-resolution-{resolution}.npz",
            points=points,
            vertices=np.asarray(region.vertices, dtype=np.float64),
            faces=np.asarray(region.faces, dtype=np.int64),
            seed=np.asarray(seed),
            resolution=np.asarray(resolution),
            sample_spacing=np.asarray(spacing),
            coverage_distance=np.asarray(coverage_distance),
            thin_feature_distance=np.asarray(0.22 / resolution),
            recenter_offset=np.asarray(report.recenter_offset, dtype=np.float64),
            source_id=np.asarray(source.source_id),
            source_revision=np.asarray(source.source_revision),
        )
    expected_euler = 0 if kind == "torus" else 2
    if (
        report.connected_components != 1
        or not report.watertight
        or not report.winding_consistent
        or report.euler_characteristic != expected_euler
        or report.normal_evidence is None
        or report.component_evidence is None
        or bool(report.component_evidence.unresolved_components)
        or report.poisson_evidence is None
        or not report.poisson_evidence.solver_converged
        or report.poisson_evidence.solver_relative_residual > 1.0e-8
        or report.outlier_evidence is None
        or not report.outlier_evidence.removed_indices
        or report.coverage_evidence is None
        or report.coverage_evidence.unsupported_patches < 1
        or report.deviation_evidence is None
    ):
        raise RuntimeError(
            f"{kind} reconstruction failed topology, native-solve, outlier, coverage, or deviation acceptance: {report!r}."
        )
    source_vertices = np.asarray(region.vertices, dtype=np.float64) + np.asarray(
        report.recenter_offset, dtype=np.float64
    )
    independent_repair = _independent_reconstruction_evidence(
        points, source_vertices, np.asarray(region.faces, dtype=np.int64), report
    )
    source_deviation = float(
        np.max(_source_deviation(kind, source_vertices), initial=0.0)
    )
    source_deviation_tolerance = 0.14 / scale

    contract = SpatialCoordinateContract.si()
    _stage(_profile_id(kind, scale), "native-surface")
    scope = MeshingScope.whole(source, 2)
    # The reconstructed route conforms to the source triangles and can only
    # red-refine them. They are already finer than 2 * spacing, so a hard
    # target there would demand coarsening; the size is a preference.
    size_control = UniformSizeControl(
        scope, 2.0 * spacing, strength=SizeControlStrength.SOFT
    )
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(size_control,),
        maximum_deviation=source_deviation_tolerance,
    )
    plan = NativeMeshingProvider().plan(
        source, specification, coordinate_contract=contract
    )
    result = plan.execute()
    accepted = accepted_result_record(result)
    # The reconstructed route certifies fidelity against the reconstructed
    # triangle source itself; it publishes no MeshingDomain chart bank.
    if result.certification is None or result.certification.fidelity is None:
        raise RuntimeError(
            f"{kind} native surface consumption omitted source fidelity evidence."
        )
    # The soft size preference stays unmet, as declared: the result keeps the
    # source triangles unrefined and reports a median edge finer than requested.
    achieved_size = dict(result.compliance.achieved)
    median_edge = achieved_size.get(f"size:{size_control.control_id}:p50_edge")
    if (
        median_edge is None
        or not median_edge < size_control.target_size
        or accepted["cells"] != report.output_cells
    ):
        raise RuntimeError(
            f"{kind} reconstructed surface size evidence changed: median edge {median_edge}, "
            f"target {size_control.target_size}, cells {accepted['cells']} vs source {report.output_cells}."
        )
    mesh_vertices = np.asarray(result.mesh.coordinates, dtype=np.float64) + np.asarray(
        report.recenter_offset, dtype=np.float64
    )
    mesh_deviation = float(np.max(_source_deviation(kind, mesh_vertices), initial=0.0))
    faces = np.asarray(region.faces, dtype=np.int64)
    domain = source.meshing_domain
    if domain is None:
        raise ValueError(
            "Reconstructed source lacks its persistent meshing-domain entities."
        )
    complex_ = phx.meshing.PiecewiseLinearComplex(
        source_vertices,
        tuple(faces),
        np.arange(faces.shape[0], dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (faces.shape[0], 1)),
        ("reconstructed-solid",),
        vertex_ids=domain.stratum_entity_ids(0),
        vertex_occurrence_paths=((),) * len(source_vertices),
        facet_ids=domain.stratum_entity_ids(2),
        facet_occurrence_paths=((),) * len(faces),
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((),),
        boundary="conforming",
    )
    plc = NativePlcSource(complex_, source.source_id, source.source_revision)
    volume_scope = MeshingScope.whole(plc, 2)
    volume_specification = phx.meshing.VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        volume_scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            UniformSizeControl(volume_scope, 2.0, strength=SizeControlStrength.SOFT),
        ),
    )
    # Donor c56614a9f reconstruction.py:431-440 authors the default radius-edge
    # 2.0 and a zero dihedral as AIMS (donor _volume_generation.py:453-455 and
    # 5117-5139 publish unmet aim rows as evidence, never refusal), so they are
    # declared soft. Validity, size, fidelity and coverage stay hard.
    volume_schedule = NativeVolumeSchedule(
        radius_edge_bound=2.0,
        minimum_dihedral_degrees=0.0,
        shape_strength=SizeControlStrength.SOFT,
    )
    volume_options = phx.meshing.NativeMeshingOptions(
        "plc_tetrahedral", volume_schedule=volume_schedule
    )
    volume_plan = NativeMeshingProvider(volume_options).plan(
        plc, volume_specification, coordinate_contract=contract
    )
    _stage(_profile_id(kind, scale), "plc-volume-fill")
    volume_result = volume_plan.execute()
    lifecycle = _volume_lifecycle(
        plc,
        plc,
        volume_result,
        volume_specification,
        _profile_id(kind, scale),
        f"native-reconstruction-{kind}-resolution-{resolution}",
        Path(archive_directory),
    )
    return {
        "status": "writer-complete",
        "profile": _profile_id(kind, scale),
        "source_id": source.source_id,
        "source_revision": source.source_revision,
        "source_input": input_record,
        "provenance": {
            "kind": "deterministic-analytic-geometry-with-declared-noise-density-gap-and-outliers",
            "donor": "c56614a9f tools/meshing_qualification/scenarios/reconstruction.py",
            "seed": input_record["seed"],
            "source_family": kind,
        },
        "diagnostics": {
            "used_for_acceptance": False,
            "topological_components": 1,
            "euler_characteristic": expected_euler,
            "watertight": report.watertight,
            "source_deviation_reference_m": source_deviation_tolerance,
            "reconstruction_maximum_source_deviation_m": source_deviation,
            "native_surface_mesh_maximum_source_deviation_m": mesh_deviation,
            "poisson_relative_residual": report.poisson_evidence.solver_relative_residual,
            "sample_to_surface_max_m": report.deviation_evidence.sample_to_surface_max,
            "surface_to_sample_max_m": report.deviation_evidence.surface_to_sample_max,
            "unsupported_repair_patches": report.coverage_evidence.unsupported_patches,
            "missing_cap_radius_m": cap_radius,
            "surface_size_preference_m": size_control.target_size,
            "surface_median_edge_m": median_edge,
        },
        "native_surface_result": accepted,
        "reconstruction": {
            "algorithm": report.algorithm,
            "input_points": report.input_points,
            "retained_points": report.retained_points,
            "vertices": report.output_vertices,
            "faces": report.output_cells,
            "normal_estimated": report.normal_evidence is not None,
            "outliers_removed": len(report.outlier_evidence.removed_indices),
            "repair_interpretation": report.repair_interpretation,
        },
        "independent_repair": independent_repair,
        "native_volume": {
            "route": volume_plan.route,
            "plc_source_id": plc.source_id,
            "plc_source_revision": plc.source_revision,
            "radius_edge_bound": volume_schedule.radius_edge_bound,
            "minimum_dihedral_degrees": volume_schedule.minimum_dihedral_degrees,
            "shape_strength": volume_schedule.shape_strength.value,
            "shape_strength_citation": "c56614a9f tools/meshing_qualification/scenarios/"
            "reconstruction.py:431-440; phydrax/meshing/_volume_generation.py:453-455,5117-5139",
            "accepted": accepted_result_record(volume_result),
        },
        "lifecycle": lifecycle,
        "runtime": runtime_identity(),
        "process_id": os.getpid(),
    }


def _profile_id(kind: str, scale: int, /) -> str:
    return f"native-reconstruction-lifecycle:{kind}" + (":larger" if scale == 2 else "")


def _volume_lifecycle(
    plc: NativePlcSource,
    archived_source: NativePlcSource | phx.meshing.SurfaceEnvelopeSource,
    result: phx.meshing.CellMeshingResult,
    specification: phx.meshing.VolumeMeshingSpec,
    profile: str,
    label: str,
    archive_directory: Path,
    /,
) -> dict[str, object]:
    """Accept, solve, refine every cell, transfer all fields, reanalyse and archive."""
    _capture_accepted_volume(plc, archived_source, result, profile, label)
    _stage(profile, "volume-acceptance")
    initial_acceptance = _volume_acceptance(plc, result)
    _stage(profile, "manufactured-solve-epoch-0")
    budget = _SOLVE_MEMORY_BUDGET_BYTES[profile]
    initial = _manufactured_solve(result, f"{label}-epoch-0", budget)
    _stage(profile, "marked-refinement")
    refinement_budget = _REFINEMENT_MEMORY_BUDGET_BYTES[profile]
    adaptation, refinement_admitted = _refine_every_cell(plc, result, refinement_budget)
    target = adaptation.target
    _stage(profile, "target-acceptance")
    target_acceptance = _volume_acceptance(plc, target)
    _stage(profile, "all-field-transfer-reanalysis")
    transfer, staged, refreshed = _transfer_accepted_state(
        result, initial.values, initial.form, adaptation, label, budget
    )
    path = archive_directory / f"{label}.phx"
    _stage(profile, "archive-write")
    content_id, expected = _write_accepted_archive(
        path,
        archived_source,
        plc,
        specification,
        target,
        staged.state.fields,
        profile,
        label,
    )
    return {
        "independent_acceptance": initial_acceptance,
        "epochs": [
            {
                "epoch": 0,
                "result_id": result.result_id,
                "seed_origin": "fresh",
                **initial.record,
            },
            {
                "epoch": 1,
                "result_id": target.result_id,
                "seed_origin": "actual-transferred-accepted-state-refreshed-by-reanalysis",
                "target_acceptance": target_acceptance,
                "adaptation": {
                    "route": phx.meshing.MeshAdaptationRoute.NATIVE_MIXED.value,
                    "status": adaptation.status.value,
                    "marked_cells": initial_acceptance["cells"],
                    "source_result_id": result.result_id,
                    "target_result_id": target.result_id,
                    "association_transfer": "PlcAssociationTransfer",
                    "memory_budget_bytes": refinement_budget,
                    "admission_estimate_bytes": refinement_admitted,
                },
                "transfer": transfer,
                **refreshed.record,
            },
        ],
        "archive": {"path": str(path), "content_id": content_id, "expected": expected},
    }


def _capture_accepted_volume(
    plc: NativePlcSource,
    archived_source: NativePlcSource | phx.meshing.SurfaceEnvelopeSource,
    result: phx.meshing.CellMeshingResult,
    profile: str,
    label: str,
    /,
) -> None:
    """Durable model archive of the accepted volume before acceptance and refinement.

    Written only into a declared capture directory, so a failing downstream
    stage replays from these genuine operands instead of a whole producer run.
    """
    directory = os.environ.get("PHYDRAX_RECONSTRUCTION_CAPTURE_DIR")
    if not directory:
        return
    from phydrax.lifecycle import Composition, CompositionEntry
    from phydrax.meshing._model_codec import (
        model_archive_content_id,
        write_model_artifact,
    )

    _stage(profile, "pre-refinement-capture")
    _, source_revision_id, source_semantics = _archived_source_identity(archived_source)
    source_entry = CompositionEntry(
        archived_source,
        entry_id="source",
        role="topology",
        owner_id="phydrax.meshing",
        structure_id=plc.complex.complex_id,
        revision_id=source_revision_id,
        semantics_id=source_semantics,
    )
    result_entry = CompositionEntry(
        result,
        entry_id="result/accepted",
        role="topology",
        owner_id="phydrax.meshing",
        structure_id=result.mesh.topology_id,
        revision_id=result.result_id,
        semantics_id="accepted-volume-before-refinement",
        dependencies=(source_entry.binding("revision"),),
    )
    path = Path(directory) / f"{label}-pre-refinement.phx"
    limits = _archive_limits(profile)
    write_model_artifact(
        path,
        Composition((source_entry, result_entry), boundary_id=f"{label}-pre-refinement"),
        limits=limits,
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "path": str(path),
                "content_id": model_archive_content_id(path, limits=limits),
                "profile": profile,
                "label": label,
                "source_type": type(archived_source).__name__,
                "plc_binding_id": plc.binding_id,
                "result_id": result.result_id,
            },
            indent=1,
        )
    )


def _archived_source_identity(
    source: NativePlcSource | phx.meshing.SurfaceEnvelopeSource, /
) -> tuple[NativePlcSource, str, str]:
    """Derived PLC carrier, revision ID and semantics of one archived source."""
    if isinstance(source, phx.meshing.SurfaceEnvelopeSource):
        return (
            source.plc_source,
            source.envelope_id,
            "original-imperfect-sheet-surface-envelope-source",
        )
    if isinstance(source, NativePlcSource):
        return source, source.binding_id, "original-reconstructed-plc-source"
    raise TypeError("Archived sources are reconstructed PLCs or surface envelopes.")


def _write_accepted_archive(
    path: Path,
    source: NativePlcSource | phx.meshing.SurfaceEnvelopeSource,
    plc: NativePlcSource,
    specification: phx.meshing.VolumeMeshingSpec,
    result: phx.meshing.CellMeshingResult,
    fields: tuple[object, ...],
    profile: str,
    label: str,
    /,
) -> tuple[str, dict[str, object]]:
    """Persist the source, controls, accepted target and all accepted field banks."""
    from phydrax.lifecycle import Composition, CompositionEntry
    from phydrax.meshing._model_codec import (
        finite_element_field_banks_recipe,
        model_archive_content_id,
        write_model_artifact,
    )

    derived, source_revision_id, source_semantics = _archived_source_identity(source)
    if derived.binding_id != plc.binding_id:
        raise ValueError("The archived source must own the lifecycle PLC carrier.")
    banks = finite_element_field_banks_recipe(
        _all_field_space(result), _all_field_layouts(result.mesh), tuple(fields)
    )
    fingerprint = canonical_fingerprint(array_tree_fingerprint(banks.fields))
    source_entry = CompositionEntry(
        source,
        entry_id="source",
        role="topology",
        owner_id="phydrax.meshing",
        structure_id=plc.complex.complex_id,
        revision_id=source_revision_id,
        semantics_id=source_semantics,
    )
    specification_entry = CompositionEntry(
        specification,
        entry_id="specification",
        role="model-state",
        owner_id="phydrax.meshing",
        structure_id=plc.binding_id,
        revision_id=specification.specification_id,
        semantics_id="original-volume-controls",
        dependencies=(source_entry.binding("revision"),),
    )
    result_entry = CompositionEntry(
        result,
        entry_id="result/accepted",
        role="topology",
        owner_id="phydrax.meshing",
        structure_id=result.mesh.topology_id,
        revision_id=result.result_id,
        semantics_id="actual-accepted-refined-volume",
        dependencies=(
            source_entry.binding("revision"),
            specification_entry.binding("revision"),
        ),
    )
    fields_entry = CompositionEntry(
        banks,
        entry_id="fields/accepted",
        role="physical-state",
        owner_id="phydrax.solver",
        structure_id=banks.topology_id,
        revision_id=fingerprint,
        semantics_id="u-density-history-P1-banks",
        dependencies=(result_entry.binding("revision"),),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    limits = _archive_limits(profile)
    write_model_artifact(
        path,
        Composition(
            (source_entry, specification_entry, result_entry, fields_entry),
            boundary_id=f"{label}-accepted-state",
        ),
        limits=limits,
    )
    return model_archive_content_id(path, limits=limits), {
        "source_type": type(source).__name__,
        "source_revision_id": source_revision_id,
        "source_binding_id": plc.binding_id,
        "specification_id": specification.specification_id,
        "result_id": result.result_id,
        "fields_fingerprint": fingerprint,
        "profile": profile,
        "label": label,
        "solve_memory_budget_bytes": _SOLVE_MEMORY_BUDGET_BYTES[profile],
        "archive_budget_bytes": _ARCHIVE_BUDGET_BYTES[profile],
        "archive_member_limit": _ARCHIVE_MEMBER_LIMITS[profile],
        "resource_plan_id": _resource_plan_id(profile, limits),
    }


def _resume_accepted_archive(
    path: str, content_id: str, expected: dict[str, object], /
) -> dict[str, object]:
    """Independent fresh-process restore and continued physics of all fields."""
    from phydrax.lifecycle import Composition
    from phydrax.meshing._model_codec import (
        read_model_artifact,
        restore_finite_element_field_banks,
        restore_model_archive_value,
    )

    # The cold reader admits the archive under its profile's own declaration,
    # which must be the archive budget the writer declared.
    profile = str(expected["profile"])
    if _ARCHIVE_BUDGET_BYTES[profile] != expected["archive_budget_bytes"]:
        raise RuntimeError(
            "Cold reader's declared archive budget differs from the writer's."
        )
    if _ARCHIVE_MEMBER_LIMITS[profile] != expected["archive_member_limit"]:
        raise RuntimeError(
            "Cold reader's declared archive member limit differs from the writer's."
        )
    limits = _archive_limits(profile)
    if _resource_plan_id(profile, limits) != expected["resource_plan_id"]:
        raise RuntimeError("Cold reader's resource plan differs from the writer's.")
    archive = read_model_artifact(
        path,
        expected_type=Composition,
        restore=restore_model_archive_value,
        limits=limits,
        expected_content_id=content_id,
    )
    _stage(str(expected["profile"]), "cold-archive-restore")
    source = archive.value("source")
    specification = archive.value("specification")
    result = archive.value("result/accepted")
    banks = archive.value("fields/accepted")
    plc, source_revision_id, _ = _archived_source_identity(source)
    if (
        type(source).__name__ != expected["source_type"]
        or source_revision_id != expected["source_revision_id"]
        or not isinstance(result, phx.meshing.CellMeshingResult)
        or plc.binding_id != expected["source_binding_id"]
        or specification.specification_id != expected["specification_id"]
        or result.result_id != expected["result_id"]
    ):
        raise RuntimeError(
            "Cold archive changed its source, controls or accepted result."
        )
    _stage(str(expected["profile"]), "cold-acceptance")
    acceptance = _volume_acceptance(plc, result)
    layouts, fields = restore_finite_element_field_banks(banks, _all_field_space(result))
    fingerprint = canonical_fingerprint(array_tree_fingerprint(fields))
    if (
        tuple(layout.name for layout in layouts) != _FIELD_NAMES
        or fingerprint != expected["fields_fingerprint"]
    ):
        raise RuntimeError("Cold archive changed the accepted field layouts or values.")
    u, density, history = (np.asarray(value, dtype=np.float64) for value in fields)
    density_defect, history_defect = _auxiliary_defects(result.mesh, density, history)
    _stage(str(expected["profile"]), "cold-continued-solve")
    # The cold reader prepares under the profile's own declaration, which must
    # be the budget the writer solved under (it is part of plan identity).
    budget = _SOLVE_MEMORY_BUDGET_BYTES[str(expected["profile"])]
    if budget != expected["solve_memory_budget_bytes"]:
        raise RuntimeError(
            "Cold reader's declared solve budget differs from the writer's."
        )
    continued = _manufactured_solve(result, f"{expected['label']}-cold", budget)
    continued_error = continued.l2_error
    if (
        continued_error > _TARGET_L2_ERROR
        or density_defect > _RESTORED_FIELD_TOLERANCE
        or history_defect > _RESTORED_FIELD_TOLERANCE
    ):
        raise RuntimeError(
            "Cold continued PDE or restored density/history oracle failed: "
            f"{continued_error=}, {density_defect=}, {history_defect=}."
        )
    return {
        "status": "passed",
        "process_id": os.getpid(),
        "source_acceptance": acceptance,
        "field_names": list(_FIELD_NAMES),
        "accepted_fields_bitwise_preserved": True,
        "source_authority_stripped": False,
        "restored_u_l2_error": _quadratic_l2_error(result.mesh, u),
        "restored_u_continued_max_difference": float(
            np.max(np.abs(u - continued.values), initial=0.0)
        ),
        "continued_physical_L2_error": continued_error,
        "density_max_defect": density_defect,
        "history_max_defect": history_defect,
        "continued_solve": continued.record,
        "runtime": runtime_identity(),
    }


def _run_process(function: str, *arguments: str) -> dict[str, object]:
    import subprocess
    import sys

    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            "import json,sys; "
            "from tools.meshing_qualification.scenarios import reconstruction as owner; "
            f"print(json.dumps(owner.{function}(*sys.argv[1:])))",
            *arguments,
        ),
        cwd=str(Path(__file__).resolve().parents[3]),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"{function} process failed ({completed.returncode}): {completed.stderr[-4000:]}"
        )
    return json.loads(completed.stdout.strip().splitlines()[-1])


_GATE_NONCLAIMS = {
    "underspecified-defective-surface-reconstruction": (
        "The donor 120-second wall target is an administrative envelope, not a scientific obligation.",
        "Generating source geometry and injected-outlier labels are diagnostics, never acceptance.",
    ),
    "imperfect-open-sheet-envelope": (
        "The donor whole-workflow wall target is an administrative envelope, not a scientific obligation.",
        "No intended dirty-shell interior is inferred.",
        "Topology-change permission is not a homeomorphism theorem.",
    ),
}


def _research_gate(record: dict[str, object], gate: str, /) -> dict[str, object]:
    """Recompute the donor research-gate obligations from recorded numbers only."""

    def volume(value: object) -> bool:
        return (
            isinstance(value, dict)
            and value.get("certified") is True
            and value.get("plc_strata_verified") is True
            and isinstance(value.get("volume_defect"), float)
            and abs(value["volume_defect"]) <= _VOLUME_CLOSURE
            and isinstance(value.get("minimum_cell_volume"), float)
            and value["minimum_cell_volume"] > 0.0
        )

    lifecycle = record.get("lifecycle")
    cold = record.get("checkpoint")
    epochs = lifecycle.get("epochs", []) if isinstance(lifecycle, dict) else []
    errors = [epoch.get("l2_error") for epoch in epochs]
    transfers = [epoch.get("transfer") for epoch in epochs[1:]]
    obligations = {
        "native_repair_with_independent_geometry": isinstance(
            record.get("independent_repair"), dict
        ),
        "native_volume_exact_domain_every_epoch": isinstance(lifecycle, dict)
        and volume(lifecycle.get("independent_acceptance"))
        and len(epochs) == _ADAPTATION_ROUNDS + 1
        and all(volume(epoch.get("target_acceptance")) for epoch in epochs[1:]),
        "consumer_transferred_state_physics": len(transfers) == _ADAPTATION_ROUNDS
        and all(
            isinstance(item, dict)
            and abs(item["state_inventory"]["defect"])
            <= _INVENTORY_TOLERANCE * max(1.0, abs(item["state_inventory"]["source"]))
            and item["density_max_defect"] <= _RESTORED_FIELD_TOLERANCE
            and item["history_max_defect"] <= _RESTORED_FIELD_TOLERANCE
            for item in transfers
        )
        and all(isinstance(value, float) for value in errors)
        and bool(errors)
        and errors[-1] <= _TARGET_L2_ERROR,
        "independent_cold_allfield_continuation": isinstance(cold, dict)
        and cold.get("status") == "passed"
        and cold.get("process_id") != record.get("process_id")
        and cold.get("accepted_fields_bitwise_preserved") is True
        and cold.get("field_names") == list(_FIELD_NAMES)
        and cold.get("source_authority_stripped") is False
        and volume(cold.get("source_acceptance"))
        and cold["continued_physical_L2_error"] <= _TARGET_L2_ERROR
        and 0.0 <= cold["density_max_defect"] <= _RESTORED_FIELD_TOLERANCE
        and 0.0 <= cold["history_max_defect"] <= _RESTORED_FIELD_TOLERANCE,
    }
    return {
        "gate": gate,
        "status": "complete" if all(obligations.values()) else "incomplete",
        "obligations": obligations,
        "nonclaims": list(_GATE_NONCLAIMS[gate]),
    }


def _writer_process(kind: str, scale: str, directory: str, /) -> dict[str, object]:
    return _reconstruct_case(kind, int(scale), directory)


def _envelope_writer_process(directory: str, /) -> dict[str, object]:
    return _envelope_case(directory)


def _reader_process(path: str, content_id: str, expected: str, /) -> dict[str, object]:
    return _resume_accepted_archive(path, content_id, json.loads(expected))


def _cold_reader(record: dict[str, object], gate: str, /) -> dict[str, object]:
    """Restore the writer's archive in a fresh process and recompute the gate."""
    lifecycle = record["lifecycle"]
    archive = lifecycle.get("archive") if isinstance(lifecycle, dict) else None
    if not isinstance(archive, dict):
        raise RuntimeError("The writer process retained no accepted-state archive.")
    record["checkpoint"] = _run_process(
        "_reader_process",
        str(archive["path"]),
        str(archive["content_id"]),
        json.dumps(archive["expected"]),
    )
    research_gate = _research_gate(record, gate)
    record["research_gate"] = research_gate
    record["status"] = "passed" if research_gate["status"] == "complete" else "failed"
    return record


def _qualify_profile(kind: str, scale: int, directory: str, /) -> dict[str, object]:
    _stage_log(directory)
    return _cold_reader(
        _run_process("_writer_process", kind, str(scale), directory),
        "underspecified-defective-surface-reconstruction",
    )


def native_reconstruction() -> dict[str, object]:
    meshcore = require_meshcore()
    directory = os.environ.get("PHYDRAX_RECONSTRUCTION_CAPTURE_DIR")
    if not directory:
        import tempfile

        directory = tempfile.mkdtemp(prefix="phydrax-native-reconstruction-")
    cases: list[dict[str, object]] = []
    failures: list[str] = []
    for kind in ("lobed-sphere", "torus", "rotated-ellipsoid"):
        for scale in (1, 2):
            try:
                case = _qualify_profile(kind, scale, directory)
            except Exception as error:
                failures.append(
                    f"{_profile_id(kind, scale)}: {type(error).__name__}: {error}"
                )
                continue
            cases.append(case)
            if case["status"] != "passed":
                failures.append(f"{_profile_id(kind, scale)}: research gate incomplete")
    return {
        "status": "failed" if failures else "passed",
        "workflow": "native defective point cloud -> reconstruction -> PLC volume -> "
        "marked refinement -> all-field transfer/reanalysis -> independent cold continuation",
        "source_registry": "tools/meshing_qualification/native_case_migration.json#imperfect-surface",
        "cases": cases,
        "failures": failures,
        "meshcore_identity": meshcore,
        "runtime": runtime_identity(),
    }


# Donor c56614a9f scenarios/envelope.py open sheet and controls, unchanged:
# one 0.2 m triangle, offset 0.263 m, maximum deviation 0.55 m, resolution 4.
_ENVELOPE_SHEET_VERTICES = ((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.2, 0.0))
_ENVELOPE_SHEET_TRIANGLES = ((0, 1, 2),)
_ENVELOPE_OFFSET = 0.263
_ENVELOPE_MAXIMUM_DEVIATION = 0.55
_ENVELOPE_RESOLUTION = 4
_ENVELOPE_SPACING = 0.44 / _ENVELOPE_RESOLUTION
_ENVELOPE_CERTIFICATE_SPACING = 0.2 / _ENVELOPE_RESOLUTION
# The donor authored its radius-edge and dihedral targets as aims whose unmet
# cells are evidence, never refusals.
_ENVELOPE_SHAPE_AIM_CITATION = (
    "c56614a9f phydrax/meshing/qualification/_envelope.py:127-131 (envelope "
    "schedule: default radius-edge 2.0, dihedral 0, one refinement round, one "
    "improvement pass), :539 ('Native radius-edge and dihedral aims retain "
    "explicit unmet evidence.'); phydrax/meshing/_volume_generation.py:453-455 "
    "('radius_edge_bound is the circumradius-to-shortest-edge aim ... unmet aims "
    "are published as evidence, never hidden'); "
    "phydrax/meshing/providers/_native_volume.py:425 ('Enforce physical size "
    "and metric floors; retain soft shape AIM evidence.')"
)


def _flat_convex_offset_volume(area: float, perimeter: float, radius: float, /) -> float:
    """Steiner volume of the closed ``radius`` neighbourhood of a planar convex set."""
    return (
        2.0 * area * radius
        + 0.5 * np.pi * perimeter * radius**2
        + 4.0 / 3.0 * np.pi * radius**3
    )


def _point_triangle_distances(points: np.ndarray, triangles: np.ndarray, /) -> np.ndarray:
    """Brute-force distance from points to a triangle soup by Voronoi-region tests."""
    p = points[:, None, :]
    a, b, c = (triangles[None, :, k, :] for k in range(3))
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.sum(ab * ap, axis=-1), np.sum(ac * ap, axis=-1)
    bp, cp = p - b, p - c
    d3, d4 = np.sum(ab * bp, axis=-1), np.sum(ac * bp, axis=-1)
    d5, d6 = np.sum(ab * cp, axis=-1), np.sum(ac * cp, axis=-1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = va + vb + vc
        closest = (
            a + (vb / denominator)[..., None] * ab + (vc / denominator)[..., None] * ac
        )
        on_ab = a + (d1 / (d1 - d3))[..., None] * ab
        on_ac = a + (d2 / (d2 - d6))[..., None] * ac
        on_bc = b + ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[..., None] * (c - b)
    conditions = (
        (d1 <= 0.0) & (d2 <= 0.0),
        (d3 >= 0.0) & (d4 <= d3),
        (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0),
        (d6 >= 0.0) & (d5 <= d6),
        (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0),
        (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0),
    )
    choices = (a, b, on_ab, c, on_ac, on_bc)
    for condition, choice in zip(reversed(conditions), reversed(choices), strict=True):
        closest = np.where(
            condition[..., None], np.broadcast_to(choice, closest.shape), closest
        )
    return np.min(np.linalg.norm(p - closest, axis=-1), axis=1)


def _input_to_envelope_upper_bound(
    source_triangles: np.ndarray,
    target_triangles: np.ndarray,
    maximum_deviation: float,
    /,
) -> float:
    """Enclose every source triangle using an actual target vertex as witness.

    A triangle is the convex hull of its vertices, so its distance to any
    fixed target vertex is bounded by the greatest vertex-to-witness distance.
    The witness selection is only a heuristic; all distances and the threshold
    comparison are evaluated exactly for the represented binary coordinates.
    A failed enclosure is inconclusive, not proof of a deviation violation.
    """
    source = np.asarray(source_triangles, dtype=np.float64)
    target = np.asarray(target_triangles, dtype=np.float64)
    if (
        source.ndim != 3
        or source.shape[1:] != (3, 3)
        or not len(source)
        or target.ndim != 3
        or target.shape[1:] != (3, 3)
        or not len(target)
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(target))
        or not math.isfinite(maximum_deviation)
        or maximum_deviation < 0.0
    ):
        raise ValueError(
            "The envelope enclosure requires finite nonempty triangle soups and a nonnegative tolerance."
        )
    target_vertices = target.reshape((-1, 3))
    upper_squared = Fraction(0)
    for triangle in source:
        # Minimize a floating approximation to the maximum witness distance.
        # Rounding here can only select a less effective witness, never weaken
        # the exact proof below.
        with np.errstate(over="ignore", invalid="ignore"):
            distances = np.max(
                np.sum((target_vertices[:, None, :] - triangle[None, :, :]) ** 2, axis=2),
                axis=1,
            )
        witness = target_vertices[int(np.argmin(distances))]
        exact_witness = tuple(Fraction.from_float(float(x)) for x in witness)
        for vertex in triangle:
            squared = sum(
                (Fraction.from_float(float(x)) - y) ** 2
                for x, y in zip(vertex, exact_witness, strict=True)
            )
            upper_squared = max(upper_squared, squared)
    tolerance = Fraction.from_float(float(maximum_deviation))
    if upper_squared > tolerance * tolerance:
        raise RuntimeError(
            "Open-sheet input-to-envelope continuous enclosure is inconclusive at the declared maximum deviation."
        )
    try:
        rounded_squared = float(upper_squared)
    except OverflowError:
        return float(maximum_deviation)
    if Fraction.from_float(rounded_squared) < upper_squared:
        rounded_squared = math.nextafter(rounded_squared, math.inf)
    if not math.isfinite(rounded_squared):
        return float(maximum_deviation)
    upper = math.sqrt(rounded_squared)
    if Fraction.from_float(upper) ** 2 < upper_squared:
        upper = math.nextafter(upper, math.inf)
    return upper


def _actual_target_triangles(
    mesh: phx.discretization.CellMesh, geometry: phx.discretization.CellGeometrySpec, /
) -> np.ndarray:
    """Extract actual affine target facets without rounding their vertices."""
    from phydrax.geometry._source_fidelity import _target_boundary_triangles

    _, exact_triangles = _target_boundary_triangles(mesh, geometry)
    triangles = np.asarray(exact_triangles, dtype=np.float64)
    coordinates = (
        coordinate
        for triangle in exact_triangles
        for point in triangle
        for coordinate in point
    )
    for exact, value in zip(coordinates, triangles.flat, strict=True):
        if not math.isfinite(float(value)) or Fraction.from_float(float(value)) != exact:
            raise RuntimeError(
                "An actual target facet vertex is not exactly representable in binary64."
            )
    return triangles


_ENVELOPE_PROFILE = "native-envelope-lifecycle"

# MAC-20261107T1649Z: every profile declares the execution-memory budget of
# all its manufactured solves and factorizations (epoch 0, reanalysis, cold).
# Run envelope: 16 GiB per row. The largest measured process-tree peak is
# 10.78 GB (lobed-sphere-2 through all-field transfer, reanalysis and archive
# packing; the 6.1 rows that reached no reanalysis peaked at 5.24 GB under an
# 8 GiB guard, which the larger row exceeds). A Cholesky solve holds its factor
# and its preparation workspace, each within the budget, so 1.5 GiB keeps the
# worst case (10.78 GB + 2 x 1.61 GB = 14.0 GB) inside the 17.18 GB envelope.
# Measured need: the refined lobed-sphere-2 factor is 40,886,178 bytes.
_SOLVE_MEMORY_BUDGET_BYTES = {
    **{
        _profile_id(kind, scale): 3 << 29
        for kind in ("lobed-sphere", "torus", "rotated-ellipsoid")
        for scale in (1, 2)
    },
    _ENVELOPE_PROFILE: 3 << 29,
}

# MAC-20261108 (MarkedRefinementBudget): every profile declares the memory
# budget of its all-cell marked refinement, checked before any adaptation work
# against the admission estimate of _refinement_admission. The estimate is a
# sampled rate, not an allocation bound: NATIVE_MIXED keeps each child's exact
# restricted geometry live (torus resolution 3: 3.15 M Fractions, 0.53 M
# rational-polynomial mapping proxies after the stage) and its transient
# Python objects stay allocator-resident. Measured on that row (10,095
# tetrahedra -> 80,760 children) after the organization membership went
# sparse: 3,189,260,288 and 3,356,000,256 bytes of stage growth in two runs,
# at most 41,555 per child. The estimate charges 48 KiB per child (18% over
# that rate): 8.84 GB for torus-1 (179,944 children). R009 A1 measured
# 33,174,061,056 bytes required for torus-2, and A2 observed 15.169 GB peak
# during its 674,928-child refinement. That profile declares 32 GiB; all
# other profiles retain 10 GiB. The estimate and admission remain unchanged.
_REFINED_CHILDREN_PER_TETRAHEDRON = 8
_REFINEMENT_BYTES_PER_TARGET_CELL = 48 << 10
_REFINEMENT_MEMORY_BUDGET_BYTES = {
    **{
        _profile_id(kind, scale): 10 << 30
        for kind in ("lobed-sphere", "torus", "rotated-ellipsoid")
        for scale in (1, 2)
    },
    _profile_id("torus", 2): 32 << 30,
    _ENVELOPE_PROFILE: 10 << 30,
}

# MAC-20261107T1759Z: every profile declares the byte budget of its model
# archives (pre-refinement capture, accepted state), shared by the writer and
# the cold reader and recorded in the receipt. Measured on 6.1 for the largest
# row, the lobed-sphere-2 accepted state: a 256,132,493-byte container,
# 238,085,115 stored array bytes, a 17,660,054-byte manifest and 3,521
# members. The admission check charges the recipe's logical expansion, and
# that expansion exceeded the 1 GiB default. 4 GiB keeps 16x headroom over
# the stored bytes. It also stays within the 16 GiB run envelope, whose
# measured process peak is 10.78 GB. Re-measure on wave-7, where array
# deduplication changes the logical/stored ratio.
_ARCHIVE_BUDGET_BYTES = {
    **{
        _profile_id(kind, scale): 4 << 30
        for kind in ("lobed-sphere", "torus", "rotated-ellipsoid")
        for scale in (1, 2)
    },
    _ENVELOPE_PROFILE: 4 << 30,
}
# The larger rotated ellipsoid has 9,964 stored array owners plus its manifest
# (LIN 20261108). The larger torus pre-refinement source capture has 30,503
# distinct stored arrays plus its manifest (LIN 20261109); this is not a
# measurement of its refined accepted-state archive. Other profiles retain
# their prior 8,192-array envelope.
_ARCHIVE_MEMBER_LIMITS = {
    **{profile: 8193 for profile in _ARCHIVE_BUDGET_BYTES},
    _profile_id("rotated-ellipsoid", 2): 16385,
    _profile_id("torus", 2): 32769,
}


def _archive_limits(profile: str, /) -> ArrayArchiveLimits:
    """Model-archive admission limits of one profile from its declared budget.

    Container, aggregate and member bytes are the budget. Element and axis
    counts use the same number, since each element takes at least one byte.
    Manifest bytes are budget / 64: 64 MiB, 3.8x the measured 17.7 MB.
    Central-directory bytes are budget / 1024: 4 MiB, 19x the measured
    0.22 MB. Nesting stays at the codec's model limit.
    """
    from phydrax.meshing._model_codec import MODEL_ARCHIVE_LIMITS

    budget = _ARCHIVE_BUDGET_BYTES[profile]
    return replace(
        MODEL_ARCHIVE_LIMITS,
        max_container_bytes=budget,
        max_aggregate_bytes=budget,
        max_member_bytes=budget,
        max_manifest_bytes=budget // 64,
        max_members=_ARCHIVE_MEMBER_LIMITS[profile],
        max_central_directory_bytes=budget // 1024,
        max_axis_length=budget,
        max_array_elements=budget,
        max_total_array_elements=budget,
    )


def _resource_plan_id(profile: str, limits: ArrayArchiveLimits, /) -> str:
    """Fingerprint the complete archive policy and lifecycle resource budgets."""
    archive = asdict(limits)
    archive["allowed_dtype_kinds"] = sorted(limits.allowed_dtype_kinds)
    return canonical_fingerprint(
        {
            "profile": profile,
            "archive": archive,
            "solve_memory_budget_bytes": _SOLVE_MEMORY_BUDGET_BYTES[profile],
            "refinement_memory_budget_bytes": _REFINEMENT_MEMORY_BUDGET_BYTES[profile],
        }
    )


def _envelope_case(directory: str, /) -> dict[str, object]:
    """Writer process of the donor open-sheet envelope lifecycle."""
    _stage(_ENVELOPE_PROFILE, "open-sheet-repair")
    vertices = np.asarray(_ENVELOPE_SHEET_VERTICES, dtype=np.float64)
    triangles = np.asarray(_ENVELOPE_SHEET_TRIANGLES, dtype=np.int64)
    np.savez_compressed(
        Path(directory) / "open-sheet-input.npz",
        vertices=vertices,
        triangles=triangles,
        offset=np.asarray(_ENVELOPE_OFFSET),
        spacing=np.asarray(_ENVELOPE_SPACING),
        certificate_spacing=np.asarray(_ENVELOPE_CERTIFICATE_SPACING),
        maximum_deviation=np.asarray(_ENVELOPE_MAXIMUM_DEVIATION),
    )
    sheet = AuthoredTriangleSheet(
        vertices, triangles, feature_id="qualification-open-envelope-sheet"
    )
    repair = sheet.repair
    if (
        sheet.report.watertight
        or not sheet.report.winding_consistent
        or sheet.report.connected_components != 1
        or sheet.report.euler_characteristic != 1
        or repair.boundary_edges != 3
        or repair.repaired_triangles != 1
    ):
        raise RuntimeError(
            f"Open sheet repair disagrees with its open one-triangle topology: {repair!r}, {sheet.report!r}."
        )
    contract = SpatialCoordinateContract.si()
    _stage(_ENVELOPE_PROFILE, "surface-envelope")
    source = phx.meshing.SurfaceEnvelopeSource(
        sheet,
        source_id="native-qualification:open-sheet-envelope",
        coordinate_contract=contract,
        offset=_ENVELOPE_OFFSET,
        spacing=_ENVELOPE_SPACING,
        certificate_spacing=_ENVELOPE_CERTIFICATE_SPACING,
        maximum_deviation=_ENVELOPE_MAXIMUM_DEVIATION,
        # Donor c56614 kept the kernel's marching-tetrahedra volume as its
        # carrier; the band-remeshed surface was never part of this profile.
        envelope_mode="kernel",
    )
    np.savez_compressed(
        Path(directory) / "open-sheet-envelope.npz",
        authored_vertices=vertices,
        authored_triangles=triangles,
        envelope_vertices=np.asarray(source.domain.vertices),
        envelope_facets=np.asarray(source.domain.facets),
        bounds=np.asarray(source.result.bounds),
        source_id=np.asarray(source.source_id),
        source_revision=np.asarray(source.source_revision),
    )
    # bounds[0]: input-to-envelope upper; bounds[2] <= envelope-to-input <= bounds[1].
    certified_bounds = np.asarray(source.result.bounds, dtype=np.float64)
    input_to_envelope_bound = float(certified_bounds[0])
    envelope_upper_bound = float(certified_bounds[1])
    envelope_lower_bound = float(certified_bounds[2])
    authored_triangles = vertices[triangles]
    if (
        max(input_to_envelope_bound, envelope_upper_bound) > _ENVELOPE_MAXIMUM_DEVIATION
        or envelope_lower_bound <= 0.0
    ):
        raise RuntimeError(
            "Open-sheet envelope violates its native deviation bounds: "
            f"certified=({input_to_envelope_bound}, {envelope_upper_bound}, {envelope_lower_bound})."
        )
    # Donor volume controls: a soft 2.0 size on the envelope facets, radius-edge
    # aim 2.0 and no dihedral aim. The donor authored the shape target as an
    # aim (_ENVELOPE_SHAPE_AIM_CITATION), so it is declared soft: refined
    # toward, unmet cells published as evidence.
    scope = MeshingScope.whole(source, 2)
    specification = phx.meshing.VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            UniformSizeControl(scope, 2.0, strength=SizeControlStrength.SOFT),
        ),
    )
    plan = NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions(
            "surface_envelope_tetrahedral",
            volume_schedule=NativeVolumeSchedule(
                radius_edge_bound=2.0,
                minimum_dihedral_degrees=0.0,
                shape_strength=SizeControlStrength.SOFT,
            ),
        )
    ).plan(source, specification, coordinate_contract=contract)
    _stage(_ENVELOPE_PROFILE, "envelope-volume-fill")
    result = plan.execute()
    _stage(_ENVELOPE_PROFILE, "independent-envelope-deviation")
    accepted = accepted_result_record(result)
    target_triangles = _actual_target_triangles(result.mesh, result.geometry)
    input_to_envelope_upper = _input_to_envelope_upper_bound(
        authored_triangles, target_triangles, _ENVELOPE_MAXIMUM_DEVIATION
    )
    # Reverse-direction samples are diagnostics, not a continuous proof.
    envelope_to_input = _point_triangle_distances(
        np.concatenate(
            (target_triangles.reshape((-1, 3)), target_triangles.mean(axis=1))
        ),
        authored_triangles,
    )
    roundoff = 1.0e-12
    if (
        float(np.min(envelope_to_input)) < envelope_lower_bound - roundoff
        or float(np.max(envelope_to_input)) > envelope_upper_bound + roundoff
    ):
        raise RuntimeError(
            "Open-sheet target violates its sampled reverse-direction certificate diagnostics: "
            f"target_to_input=[{np.min(envelope_to_input)}, {np.max(envelope_to_input)}], "
            f"certified=({input_to_envelope_bound}, {envelope_upper_bound}, {envelope_lower_bound})."
        )
    achieved = dict(result.compliance.achieved)
    repair_quantities = {
        name: achieved.get(f"source_topology_repair:{name}")
        for name, _ in repair.quantities
    }
    if any(repair_quantities[name] != float(value) for name, value in repair.quantities):
        raise RuntimeError(
            f"Envelope volume result lost the authored topology-repair interpretation: {repair_quantities}."
        )
    measures = np.asarray(result.quality.evaluation.measures, dtype=np.float64)
    volume = float(np.sum(measures))
    corners = authored_triangles[0]
    area = 0.5 * float(
        np.linalg.norm(np.cross(corners[1] - corners[0], corners[2] - corners[0]))
    )
    perimeter = float(
        np.sum(np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1))
    )
    # The meshed region contains {d < lower} and lies inside {d <= upper}.
    volume_lower = _flat_convex_offset_volume(area, perimeter, envelope_lower_bound)
    volume_upper = _flat_convex_offset_volume(area, perimeter, envelope_upper_bound)
    if np.any(measures <= 0.0) or not volume_lower <= volume <= volume_upper:
        raise RuntimeError(
            f"Envelope tetrahedra integrate {volume} m^3 outside the Steiner offset bracket [{volume_lower}, {volume_upper}] m^3."
        )
    lifecycle = _volume_lifecycle(
        source.plc_source,
        source,
        result,
        specification,
        _ENVELOPE_PROFILE,
        "native-open-sheet-envelope",
        Path(directory),
    )
    return {
        "status": "writer-complete",
        "profile": _ENVELOPE_PROFILE,
        "source_input": {
            "vertices": vertices.tolist(),
            "triangles": triangles.tolist(),
            "fingerprint": array_tree_fingerprint((vertices, triangles)),
            "interpretation": "sampled unsigned-distance sublevel; feature rounding and topology change permitted",
        },
        "provenance": {
            "source": "c56614a9f scenarios/envelope.py _envelope_qualification_fixture "
            "(PHYDRA synthetic open sheet; test_surface_envelope positive fixture)",
            "controls": "donor offset 0.263 m, maximum deviation 0.55 m, spacing 0.44/4 m, "
            "certificate spacing 0.2/4 m, soft size 2.0, radius-edge 2.0, dihedral 0, "
            "one all-cell refinement, absolute L2 target 0.05",
            "donor_feature_edges": "edge (0, 1) under allow-rounding permission; no feature is enforced",
            "donor_volume": "c56614a9f phydrax/meshing/_surface_envelope.py:1414-1540 "
            "_generate_surface_envelope_volume: the marching-tetrahedra carrier "
            "(volume_points, cells) is kept and refined in place (envelope_mode='kernel')",
            "shape_strength": "soft",
            "shape_strength_citation": _ENVELOPE_SHAPE_AIM_CITATION,
        },
        "source_id": source.source_id,
        "source_revision": source.source_revision,
        "original_source_id": sheet.source_id,
        "original_source_revision": sheet.source_revision,
        "repair_interpretation": sheet.report.repair_interpretation,
        "envelope": {
            "envelope_id": source.envelope_id,
            "plc_binding_id": source.plc_source.binding_id,
            "envelope_mode": source.envelope_mode,
            "kept_carrier_vertices": int(source.result.volume_vertices.shape[0]),
            "kept_carrier_tetrahedra": int(source.result.tetrahedra.shape[0]),
            "offset_m": source.offset,
            "spacing_m": source.spacing,
            "certificate_spacing_m": source.certificate_spacing,
            "maximum_deviation_m": source.maximum_deviation,
            "certified_input_to_envelope_upper_m": input_to_envelope_bound,
            "certified_envelope_to_input_bounds_m": [
                envelope_lower_bound,
                envelope_upper_bound,
            ],
        },
        "independent_repair": {
            "topology_repair": dict(repair.quantities),
            "result_topology_repair": repair_quantities,
            "independent_input_to_envelope_upper_m": input_to_envelope_upper,
            "sampled_envelope_to_input_min_m": float(np.min(envelope_to_input)),
            "sampled_envelope_to_input_max_m": float(np.max(envelope_to_input)),
            "measured_volume_m3": volume,
            "steiner_volume_bracket_m3": [volume_lower, volume_upper],
        },
        "native_volume_result": accepted,
        "lifecycle": lifecycle,
        "runtime": runtime_identity(),
        "process_id": os.getpid(),
    }


def native_envelope() -> dict[str, object]:
    meshcore = require_meshcore()
    directory = os.environ.get("PHYDRAX_RECONSTRUCTION_CAPTURE_DIR")
    if not directory:
        import tempfile

        directory = tempfile.mkdtemp(prefix="phydrax-native-envelope-")
    _stage_log(directory)
    record = _cold_reader(
        _run_process("_envelope_writer_process", directory),
        "imperfect-open-sheet-envelope",
    )
    record["workflow"] = (
        "native imperfect open sheet -> topology repair -> unsigned-distance envelope -> "
        "tetrahedral volume -> marked refinement -> all-field transfer/reanalysis -> "
        "independent cold continuation"
    )
    record["source_registry"] = (
        "tools/meshing_qualification/native_case_migration.json#imperfect-surface"
    )
    record["meshcore_identity"] = meshcore
    return record
