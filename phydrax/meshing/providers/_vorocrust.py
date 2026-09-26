#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import io
import os
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from ..._external_runtime import NativeWorkerIdentity
from ..._identity import SemanticProvenance
from ...discretization import CellMesh
from ...geometry.surface import SurfaceModel
from ...logging import emit
from .._canonical import certify_cell_mesh
from .._contracts import (
    MeshingCapability,
    MeshingDerivativeMode,
    MeshingExecutionMode,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingLimits,
    MeshingOperation,
    MeshingProviderInfo,
    MeshingSourceKind,
)
from .._result import CellMeshingResult, MeshingComplianceReport, MeshingRuntimeInfo
from .._trace import (
    MeshingStageKind,
    MeshingStageReport,
    MeshingStageStatus,
    MeshingTrace,
)
from ._worker import ProviderWorker


_SEED_HEADER = b"x1coord, x2coord, x3coord, radius"
_DIGESTS = ("source_sha256", "config_sha256", "library_sha256")
_OUTPUT_ARRAYS = frozenset(
    (
        "vertices",
        "face_offsets",
        "face_vertices",
        "face_seeds",
        "seed_points",
        "seed_regions",
    )
)


@dataclass(frozen=True)
class VoroCrustOptions:
    """VoroCrust sampling controls and extraction acceptance tolerances.

    ``relative_merge_tolerance`` (times the output bounding-box diagonal) merges
    backend vertex aliases; VoroCrust leaves some shared cell corners unwelded
    about 1e-11 relative apart, which the 1e-9 default closes. Merging never
    moves retained coordinates and transitive alias chains are refused.
    """

    maximum_radius: float
    lipschitz_constant: float = 0.25
    feature_angle_degrees: float = 60.0
    relative_volume_tolerance: float = 1e-6
    relative_merge_tolerance: float = 1e-9
    require_native_output_preallocation: bool = False

    def __post_init__(self):
        if not np.isfinite(self.maximum_radius) or self.maximum_radius <= 0:
            raise ValueError("maximum_radius must be positive and finite.")
        if (
            not np.isfinite(self.lipschitz_constant)
            or not 0 <= self.lipschitz_constant < 1
        ):
            raise ValueError("lipschitz_constant must lie in [0, 1).")
        if (
            not np.isfinite(self.feature_angle_degrees)
            or not 0 < self.feature_angle_degrees < 180
        ):
            raise ValueError("feature_angle_degrees must lie in (0, 180).")
        if (
            not np.isfinite(self.relative_volume_tolerance)
            or not 0 < self.relative_volume_tolerance < 1
        ):
            raise ValueError("relative_volume_tolerance must lie in (0, 1).")
        if (
            not np.isfinite(self.relative_merge_tolerance)
            or not 0 <= self.relative_merge_tolerance <= 1e-8
        ):
            raise ValueError("relative_merge_tolerance must lie in [0, 1e-8].")
        if type(self.require_native_output_preallocation) is not bool:
            raise TypeError("require_native_output_preallocation must be a Boolean.")


def _conversion(message: str, /) -> MeshingFailure:
    return MeshingFailure(MeshingFailureCategory.CONVERSION_FAILED, message)


def _resource(message: str, /) -> MeshingFailure:
    return MeshingFailure(MeshingFailureCategory.RESOURCE_EXHAUSTED, message)


def _executable(value: str | Path) -> str:
    path = shutil.which(str(value))
    if path is None:
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_UNAVAILABLE,
            f"Executable is unavailable: {value}",
        )
    return str(Path(path).resolve())


def _run(command: list[str], directory: Path, deadline: float) -> None:
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise MeshingFailure(
            MeshingFailureCategory.TIMED_OUT, "VoroCrust deadline expired."
        )
    started = monotonic()
    emit(
        "DEBUG",
        "provider.process.started",
        "VoroCrust process started",
        executable=Path(command[0]).name,
        provider="vorocrust",
    )
    with (directory / "provider.log").open("ab") as log:
        try:
            result = subprocess.run(
                command,
                cwd=directory,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=remaining,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            emit(
                "ERROR",
                "provider.process.failed",
                "VoroCrust process timed out",
                elapsed_seconds=monotonic() - started,
                failure_category="timed_out",
                provider="vorocrust",
            )
            raise MeshingFailure(
                MeshingFailureCategory.TIMED_OUT, "VoroCrust execution timed out."
            ) from error
        except OSError as error:
            emit(
                "ERROR",
                "provider.process.failed",
                "VoroCrust process was unavailable",
                elapsed_seconds=monotonic() - started,
                failure_category="provider_unavailable",
                provider="vorocrust",
            )
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_UNAVAILABLE,
                "VoroCrust process could not be launched.",
            ) from error
    if result.returncode:
        emit(
            "ERROR",
            "provider.process.failed",
            "VoroCrust process failed",
            elapsed_seconds=monotonic() - started,
            failure_category="nonzero_exit",
            provider="vorocrust",
            return_code=result.returncode,
        )
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            f"VoroCrust process exited with code {result.returncode}.",
        )
    emit(
        "DEBUG",
        "provider.process.completed",
        "VoroCrust process completed",
        elapsed_seconds=monotonic() - started,
        provider="vorocrust",
        return_code=result.returncode,
    )


def _provider_info(
    identity: NativeWorkerIdentity, /
) -> tuple[MeshingProviderInfo, dict[str, str]]:
    """Provider identity from the worker's hello record, probed once per session."""
    reported = identity.reported
    revision = reported.get("revision")
    digests = {name: reported.get(name) for name in _DIGESTS}
    if (
        reported.get("provider") != "vorocrust"
        or not isinstance(revision, str)
        or not revision
        or any(character.isspace() for character in revision)
        or not isinstance(reported.get("operations"), list)
        or "extract" not in reported["operations"]
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in digests.values()
        )
    ):
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_UNAVAILABLE,
            "Unsupported phydrax-vorocrust-worker identity; it must report the exact "
            "revision and source/configuration/library SHA-256 identities.",
            stage="startup",
        )
    return (
        MeshingProviderInfo(
            "vorocrust",
            revision,
            "BSD-3-Clause",
            operations=(MeshingOperation.MESH_VOLUME,),
            source_kinds=(MeshingSourceKind.SURFACE,),
            capabilities=(MeshingCapability.POLYHEDRAL,),
            cell_kinds=("polyhedron",),
            dimensions=(3,),
            execution_modes=(MeshingExecutionMode.SUBPROCESS,),
        ),
        {"revision": revision, **digests},
    )


def _read_seeds(
    path: Path, limits: MeshingLimits, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse vc_mesh seeds.csv into coordinates, sizing radii, and region IDs."""
    if path.stat().st_size > limits.maximum_data_bytes:
        raise _resource("VoroCrust seeds exceed transfer budget.")
    header, _, body = path.read_bytes().partition(b"\n")
    if header.rstrip(b"\r") != _SEED_HEADER:
        raise _conversion("Unexpected VoroCrust seed CSV header.")
    maximum = min(2 * limits.maximum_cells, 40_000_000)
    if body.count(b"\n") > maximum:
        raise _resource("VoroCrust seed count exceeds its bound.")
    if not body.strip():
        raise _conversion("VoroCrust produced no seeds.")
    # Ragged or non-numeric rows are external-format violations.
    try:
        table = np.loadtxt(
            io.BytesIO(body), delimiter=",", dtype=np.float64, ndmin=2, encoding=None
        )
    except ValueError as error:
        raise _conversion(f"Invalid VoroCrust seed rows: {error}") from error
    if table.shape[0] == 0:
        raise _conversion("VoroCrust produced no seeds.")
    if table.shape[0] > maximum:
        raise _resource("VoroCrust seed count exceeds its bound.")
    if table.shape[1] != 5:
        raise _conversion("VoroCrust seed rows require five columns.")
    points, radii, regions = table[:, :3], table[:, 3], table[:, 4]
    if (
        not np.all(np.isfinite(table))
        or np.any(radii <= 0)
        or np.any(regions != np.floor(regions))
        or np.any((regions < 0) | (regions > table.shape[0]))
    ):
        raise _conversion("VoroCrust seeds require finite points, radii, and regions.")
    return (
        np.ascontiguousarray(points),
        np.ascontiguousarray(radii),
        regions.astype(np.int64),
    )


def _preflight_native_output(
    seed_count: int, interior_count: int, limits: MeshingLimits, /
) -> None:
    """Refuse before extraction whenever VoroCrust's output could exceed the limits.

    VoroCrust emits one convex Voronoi polytope per interior (nonzero-region)
    seed, and each polytope has at most F = n - 1 facets for n seeds. By Euler's
    formula a convex polytope with F facets has at most 2F - 4 vertices and
    3F - 6 edges, so its face loops hold at most 6F - 12 vertex entries. Every
    face is emitted by one cell, and connectivity is counted from both sides.
    """
    facets = max(seed_count - 1, 0)
    if (
        interior_count > limits.maximum_cells
        or interior_count * max(2 * facets - 4, 0) > limits.maximum_vertices
        or interior_count * facets > limits.maximum_faces
        or 2 * interior_count * max(6 * facets - 12, 0)
        > limits.maximum_connectivity_entries
    ):
        raise _resource(
            "Conservative VoroCrust output bounds exceed configured limits before native extraction."
        )


def _vertex_aliases(points: np.ndarray, relative_tolerance: float) -> np.ndarray:
    """Normalize backend vertex aliases without changing retained coordinates.

    Vertices within the tolerance are connected; every connected component is
    represented by its lowest vertex index, and no member may lie farther than
    the tolerance from that representative (transitive chains are refused).
    """
    count = points.shape[0]
    tolerance = relative_tolerance * float(np.linalg.norm(np.ptp(points, axis=0)))
    pairs = cKDTree(points).query_pairs(tolerance, output_type="ndarray")
    graph = coo_matrix(
        (np.ones(pairs.shape[0], dtype=np.int8), (pairs[:, 0], pairs[:, 1])),
        shape=(count, count),
    )
    components, labels = connected_components(graph, directed=False)
    representatives = np.full(components, count, dtype=np.int64)
    np.minimum.at(representatives, labels, np.arange(count, dtype=np.int64))
    aliases = representatives[labels]
    if np.any(np.linalg.norm(points - points[aliases], axis=1) > tolerance):
        raise _conversion(
            "Transitive vertex aliases exceed the requested merge tolerance."
        )
    return aliases


def _extraction_arrays(
    arrays: Mapping[str, np.ndarray], limits: MeshingLimits, /
) -> tuple[np.ndarray, ...]:
    """Validate the extraction worker's packed output before any assembly."""
    if set(arrays) != _OUTPUT_ARRAYS:
        raise _conversion("VoroCrust extraction output arrays are incomplete.")
    points = np.asarray(arrays["vertices"], dtype=np.float64)
    offsets = np.asarray(arrays["face_offsets"], dtype=np.int64)
    loops = np.asarray(arrays["face_vertices"], dtype=np.int64)
    pairs = np.asarray(arrays["face_seeds"], dtype=np.int64)
    seeds = np.asarray(arrays["seed_points"], dtype=np.float64)
    regions = np.asarray(arrays["seed_regions"], dtype=np.int64)
    vertex_count, face_count, seed_count = points.shape[0], pairs.shape[0], seeds.shape[0]
    if (
        vertex_count > limits.maximum_vertices
        or face_count > limits.maximum_faces
        or seed_count > 2 * limits.maximum_cells
    ):
        raise _resource("VoroCrust output exceeds entity budgets.")
    if 2 * loops.size > limits.maximum_connectivity_entries:
        raise _resource("VoroCrust connectivity exceeds budget.")
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or vertex_count == 0
        or pairs.ndim != 2
        or pairs.shape[1] != 2
        or face_count == 0
        or offsets.shape != (face_count + 1,)
        or seeds.ndim != 2
        or seeds.shape[1] != 3
        or regions.shape != (seed_count,)
        or not np.all(np.isfinite(points))
        or not np.all(np.isfinite(seeds))
        or np.any(regions < 0)
    ):
        raise _conversion("Invalid VoroCrust point or seed arrays.")
    if (
        offsets[0] != 0
        or offsets[-1] != loops.size
        or np.any(np.diff(offsets) < 3)
        or np.any((loops < 0) | (loops >= vertex_count))
        or np.any((pairs < 0) | (pairs >= seed_count))
    ):
        raise _conversion("VoroCrust connectivity index is out of bounds.")
    return points, offsets, loops, pairs, seeds, regions


def _polyhedra(
    arrays: Mapping[str, np.ndarray],
    limits: MeshingLimits,
    relative_merge_tolerance: float,
    /,
) -> tuple[CellMesh, int, int]:
    """Assemble outward-oriented polyhedra from packed VoroCrust output.

    Face loops are alias-normalized; consecutive repeated vertices are removed
    and faces left with fewer than three vertices are counted as collapsed.
    Every face bounds each of its interior seeds, oriented away from that seed.
    Returns the mesh, the merged alias count, and the collapsed face count.
    """
    points, offsets, loops, pairs, seeds, regions = _extraction_arrays(arrays, limits)
    interior = np.flatnonzero(regions > 0)
    if interior.size == 0 or interior.size > limits.maximum_cells:
        raise _resource("Invalid VoroCrust interior cell count.")
    aliases = _vertex_aliases(points, relative_merge_tolerance)
    widths = np.diff(offsets)
    entry_faces = np.repeat(np.arange(widths.size), widths)
    previous = np.arange(loops.size) - 1
    previous[offsets[:-1]] = offsets[1:] - 1
    loops = aliases[loops]
    keep = loops != loops[previous]
    kept = np.bincount(entry_faces[keep], minlength=widths.size)
    retained = kept >= 3
    collapsed_faces = np.count_nonzero(~retained)
    keep &= retained[entry_faces]
    if not np.any(retained):
        raise _conversion("VoroCrust interior cells must be bounded by faces.")
    faces, sizes = np.flatnonzero(retained), kept[retained]
    values, loop_faces = loops[keep], np.repeat(np.arange(faces.size), sizes)
    starts = np.concatenate(([0], np.cumsum(sizes)))
    following = np.arange(values.size) + 1
    following[starts[1:] - 1] = starts[:-1]
    corners = points[values]
    anchors = corners[starts[:-1]][loop_faces]
    areas = np.add.reduceat(
        np.cross(corners - anchors, corners[following] - anchors), starts[:-1], axis=0
    )
    centroids = np.add.reduceat(corners, starts[:-1], axis=0) / sizes[:, None]
    # One incidence per (face, interior side seed), ordered by cell then face.
    sides = pairs[faces]
    incident = regions[sides] > 0
    incidence_faces = np.broadcast_to(np.arange(faces.size)[:, None], sides.shape)[
        incident
    ]
    incidence_cells = sides[incident]
    order = np.lexsort((np.nonzero(incident)[1], incidence_faces, incidence_cells))
    incidence_faces, incidence_cells = incidence_faces[order], incidence_cells[order]
    outward = (
        np.sum(
            areas[incidence_faces]
            * (centroids[incidence_faces] - seeds[incidence_cells]),
            axis=1,
        )
        > 0
    )
    cell_faces = np.searchsorted(
        incidence_cells, interior, side="right"
    ) - np.searchsorted(incidence_cells, interior, side="left")
    if np.any(cell_faces == 0):
        raise _conversion("VoroCrust interior cells must be bounded by faces.")
    lengths = sizes[incidence_faces]
    incidence_starts = np.concatenate(([0], np.cumsum(lengths)))
    local = np.arange(incidence_starts[-1]) - np.repeat(incidence_starts[:-1], lengths)
    local = np.where(
        np.repeat(outward, lengths), local, np.repeat(lengths, lengths) - 1 - local
    )
    entries = values[np.repeat(starts[:-1][incidence_faces], lengths) + local]
    used = np.unique(entries)
    mapping = np.full(points.shape[0], -1, dtype=np.int64)
    mapping[used] = np.arange(used.size)
    # CellMesh.from_polyhedra consumes nested face sequences; splitting the
    # packed loops is its input boundary. Backend cells that do not close into
    # oriented polytopes (for example corners VoroCrust left unwelded) are a
    # conversion failure of the provider output, not an invalid caller input.
    face_loops = np.split(mapping[entries], incidence_starts[1:-1])
    cell_starts = np.concatenate(([0], np.cumsum(cell_faces)))
    try:
        mesh = CellMesh.from_polyhedra(
            points[used],
            tuple(
                tuple(face_loops[start:stop])
                for start, stop in zip(cell_starts[:-1], cell_starts[1:], strict=True)
            ),
            vertex_global_ids=used,
            cell_global_ids=interior,
        )
    except ValueError as error:
        raise _conversion(f"VoroCrust cells are not closed polytopes: {error}") from error
    if mesh.entity_set(1).count > limits.maximum_edges:
        raise _resource("VoroCrust edges exceed the entity budget.")
    return mesh, points.shape[0] - np.unique(aliases).size, collapsed_faces


def _remaining_limits(limits: MeshingLimits, deadline: float, /) -> MeshingLimits:
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise MeshingFailure(
            MeshingFailureCategory.TIMED_OUT, "VoroCrust deadline expired."
        )
    return MeshingLimits(
        maximum_vertices=limits.maximum_vertices,
        maximum_edges=limits.maximum_edges,
        maximum_faces=limits.maximum_faces,
        maximum_cells=limits.maximum_cells,
        maximum_connectivity_entries=limits.maximum_connectivity_entries,
        maximum_data_bytes=limits.maximum_data_bytes,
        maximum_wall_seconds=remaining,
    )


class VoroCrustProvider:
    """Real VoroCrust sampling and public-API packed polyhedron extraction.

    ``executable`` is the upstream ``vc_mesh`` CLI, run once per call as a
    bounded subprocess. Extraction runs in one persistent
    ``phydrax-vorocrust-worker`` session (explicit ``worker`` path, else
    PHYDRAX_VOROCRUST_WORKER, else PATH) whose identity is probed once.
    Radius is the backend sphere-sizing bound, not a guaranteed cell-edge size.
    Material/selection transfer is not inferred from provisional seed colors.
    """

    def __init__(
        self,
        executable: str | Path = "vc_mesh",
        worker: str | os.PathLike[str] | None = None,
    ):
        self.executable = _executable(executable)
        self.worker = ProviderWorker(
            "vorocrust",
            executable=worker,
            environment_variable="PHYDRAX_VOROCRUST_WORKER",
            default_executable="phydrax-vorocrust-worker",
            build_hint=(
                "Build native/providers/vorocrust and set PHYDRAX_VOROCRUST_WORKER "
                "or pass worker."
            ),
        )

    def info(self) -> MeshingProviderInfo:
        return _provider_info(self.worker.identity)[0]

    def close(self) -> None:
        self.worker.close()

    def __enter__(self) -> VoroCrustProvider:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def execute(
        self,
        surface: SurfaceModel,
        options: VoroCrustOptions,
        /,
        *,
        limits: MeshingLimits | None = None,
    ) -> CellMeshingResult:
        if not isinstance(surface, SurfaceModel) or not isinstance(
            options, VoroCrustOptions
        ):
            raise TypeError("Expected SurfaceModel and VoroCrustOptions.")
        if surface.selections or surface.interfaces or surface.metadata.cell_tags:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
                "VoroCrust does not transfer source selections, interfaces, or material tags.",
            )
        limits = MeshingLimits() if limits is None else limits
        if not isinstance(limits, MeshingLimits):
            raise TypeError("limits must be MeshingLimits.")
        if options.require_native_output_preallocation:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
                "VoroCrust exposes no bounded native output allocator or callback.",
            )
        if (
            limits.maximum_vertices > 10_000_000
            or limits.maximum_faces > 50_000_000
            or limits.maximum_connectivity_entries > 500_000_000
            or limits.maximum_data_bytes > 4_000_000_000
            or 2 * limits.maximum_cells > 40_000_000
        ):
            raise ValueError("limits exceed the VoroCrust worker hard bounds.")
        point_count = surface.mesh.coordinates.shape[0]
        face_count = sum(block.cell_count for block in surface.mesh.blocks)
        connectivity_count = sum(block.vertices.size for block in surface.mesh.blocks)
        estimated_input_bytes = (
            point_count * 80 + face_count * 24 + connectivity_count * 12 + 4_096
        )
        if (
            point_count > limits.maximum_vertices
            or face_count > limits.maximum_faces
            or connectivity_count > limits.maximum_connectivity_entries
            or estimated_input_bytes > limits.maximum_data_bytes
        ):
            raise _resource("VoroCrust source exceeds configured entity or byte limits.")
        faces = np.concatenate(
            [np.asarray(block.vertices) for block in surface.mesh.blocks]
        )
        points = np.asarray(surface.mesh.coordinates, dtype=np.float64)
        edges = np.sort(
            np.concatenate((faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)])), axis=1
        )
        _, counts = np.unique(edges, axis=0, return_counts=True)
        if np.any(counts != 2):
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SOURCE,
                "VoroCrust requires a closed manifold surface.",
            )
        triangles = points[faces]
        source_volume = float(
            np.sum(
                np.sum(
                    triangles[:, 0] * np.cross(triangles[:, 1], triangles[:, 2]), axis=1
                )
            )
            / 6
        )
        if not np.isfinite(source_volume) or source_volume <= 0:
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SOURCE,
                "VoroCrust requires outward-oriented positive-volume input.",
            )
        provider, extractor_identity = _provider_info(self.worker.identity)
        deadline = monotonic() + limits.maximum_wall_seconds
        with TemporaryDirectory(prefix="phydrax-vorocrust-") as temporary:
            directory = Path(temporary)
            with (directory / "surface.obj").open("w") as stream:
                np.savetxt(stream, points, fmt="v %.17g %.17g %.17g")
                np.savetxt(stream, faces.astype(np.int64) + 1, fmt="f %d %d %d")
            (directory / "vc.in").write_text(
                f"INPUT_MESH_FILE = surface.obj\nR_MAX = {options.maximum_radius:.17g}\n"
                f"LIP_CONST = {options.lipschitz_constant:.17g}\n"
                f"VC_ANGLE = {options.feature_angle_degrees:.17g}\nNUM_THREADS = 1\n"
            )
            _run([self.executable, "-vc", "vc.in"], directory, deadline)
            seeds = directory / "seeds.csv"
            if not seeds.is_file():
                raise MeshingFailure(
                    MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                    "VoroCrust produced no seeds.",
                )
            seed_points, seed_radii, seed_regions = _read_seeds(seeds, limits)
        _preflight_native_output(
            seed_regions.size, np.count_nonzero(seed_regions), limits
        )
        call = self.worker.call(
            "extract",
            {
                "maximum_vertices": limits.maximum_vertices,
                "maximum_faces": limits.maximum_faces,
                "maximum_connectivity_entries": limits.maximum_connectivity_entries,
            },
            {
                "seeds": seed_points,
                "seed_radii": seed_radii,
                "seed_regions": seed_regions,
            },
            limits=_remaining_limits(limits, deadline),
        )
        mesh, merged_vertices, collapsed_faces = _polyhedra(
            call.arrays, limits, options.relative_merge_tolerance
        )
        native = certify_cell_mesh(mesh, surface.metadata.coordinate_contract)
        volume = float(np.sum(np.asarray(native.quality.evaluation.measures)))
        error = abs(volume - source_volume) / source_volume
        if error > options.relative_volume_tolerance:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "VoroCrust output does not preserve source enclosed volume.",
            )
        identity = self.worker.identity
        provenance = SemanticProvenance(
            {
                "kind": "vorocrust-volume",
                "source": surface.mesh.mesh_id,
                "extractor_identity": tuple(
                    extractor_identity[name] for name in _DIGESTS
                ),
                "options": (
                    options.maximum_radius,
                    options.lipschitz_constant,
                    options.feature_angle_degrees,
                    options.relative_volume_tolerance,
                    options.relative_merge_tolerance,
                    options.require_native_output_preallocation,
                ),
                "input_manifest_sha256": call.evidence["input_manifest_sha256"],
                "output_manifest_sha256": call.evidence["output_manifest_sha256"],
                "worker_sequence": call.sequence,
                "worker_peak_rss_bytes": call.evidence["peak_rss_bytes"],
            },
            resource_ids={
                "worker_identity": identity.identity_id,
                "worker_session": identity.session_id,
            },
        )
        compliance = MeshingComplianceReport(
            provenance.semantic_id,
            achieved=(
                ("relative_volume_error", error),
                ("merged_vertex_aliases", float(merged_vertices)),
                ("collapsed_zero_area_faces", float(collapsed_faces)),
            ),
        )
        trace = MeshingTrace(
            (
                MeshingStageReport(
                    MeshingStageKind.VOLUME_FILL,
                    MeshingStageStatus.PASSED,
                    input_ids=(surface.mesh.mesh_id,),
                    output_ids=(native.mesh.mesh_id,),
                ),
                *native.trace.stages,
            )
        )
        return CellMeshingResult(
            native.mesh,
            native.geometry,
            native.coordinate_contract,
            native.audit,
            native.quality,
            compliance,
            trace,
            provider,
            MeshingRuntimeInfo(
                provider.provider_id,
                provider.version,
                MeshingExecutionMode.SUBPROCESS,
                deterministic=False,
                enforced_limits=(
                    "wall_seconds",
                    "input_entities",
                    "seed_rows",
                    "input_bytes",
                    "serialized_output_bytes",
                    self.worker.memory_limit_evidence(),
                ),
                unenforced_limits=(
                    "provider_internal_workspace",
                    "native_output_entities_preallocation",
                    "native_output_connectivity_preallocation",
                ),
            ),
            MeshingDerivativeMode.NONDIFFERENTIABLE,
            provenance,
        )


__all__ = ["VoroCrustProvider", "VoroCrustOptions"]
