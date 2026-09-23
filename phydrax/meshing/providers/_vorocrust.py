#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import os
import shutil
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from time import monotonic

import numpy as np
from scipy.spatial import cKDTree

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


@dataclass(frozen=True)
class VoroCrustOptions:
    maximum_radius: float
    lipschitz_constant: float = 0.25
    feature_angle_degrees: float = 60.0
    relative_volume_tolerance: float = 1e-6
    relative_merge_tolerance: float = 1e-12
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


def _extractor_identity(extractor: str, /) -> tuple[MeshingProviderInfo, dict[str, str]]:
    maximum_bytes = 16 * 1024
    deadline = monotonic() + 10.0
    with TemporaryFile(mode="w+b") as output:
        try:
            process = subprocess.Popen(
                [extractor, "--version"],
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
        except OSError as error:
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_UNAVAILABLE,
                "VoroCrust extractor version probe could not be launched.",
            ) from error

        def terminate() -> None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            process.wait()

        overflow = False
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                terminate()
                raise MeshingFailure(
                    MeshingFailureCategory.TIMED_OUT,
                    "VoroCrust extractor version probe timed out.",
                )
            try:
                process.wait(timeout=min(remaining, 0.05))
                break
            except subprocess.TimeoutExpired:
                if os.fstat(output.fileno()).st_size > maximum_bytes:
                    overflow = True
                    terminate()
                    break
        size = os.fstat(output.fileno()).st_size
        if overflow or size > maximum_bytes:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "VoroCrust extractor version output exceeds its byte bound.",
            )
        if process.returncode:
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                f"VoroCrust extractor version probe exited with code {process.returncode}.",
            )
        output.seek(0)
        try:
            version_output = output.read(maximum_bytes + 1).decode(
                "utf-8", errors="strict"
            )
        except UnicodeDecodeError as error:
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_UNAVAILABLE,
                "VoroCrust extractor version output is not UTF-8.",
            ) from error
    fields = version_output.strip().split()
    prefixes = (
        "phydrax-vorocrust/1",
        "vorocrust/",
        "source-sha256/",
        "config-sha256/",
        "library-sha256/",
    )
    if (
        len(fields) != len(prefixes)
        or fields[0] != prefixes[0]
        or any(
            not field.startswith(prefix) or len(field) == len(prefix)
            for field, prefix in zip(fields[1:], prefixes[1:], strict=True)
        )
    ):
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_UNAVAILABLE,
            "Unsupported phydrax-vorocrust bridge protocol/version.",
        )
    identity = {
        "revision": fields[1][len(prefixes[1]) :],
        "source_sha256": fields[2][len(prefixes[2]) :],
        "config_sha256": fields[3][len(prefixes[3]) :],
        "library_sha256": fields[4][len(prefixes[4]) :],
    }
    if any(
        len(identity[name]) != 64
        or any(character not in "0123456789abcdef" for character in identity[name])
        for name in ("source_sha256", "config_sha256", "library_sha256")
    ):
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_UNAVAILABLE,
            "VoroCrust extractor returned invalid source/build/library identities.",
        )
    return (
        MeshingProviderInfo(
            "vorocrust",
            identity["revision"],
            "BSD-3-Clause",
            operations=(MeshingOperation.MESH_VOLUME,),
            source_kinds=(MeshingSourceKind.SURFACE,),
            capabilities=(MeshingCapability.POLYHEDRAL,),
            cell_kinds=("polyhedron",),
            dimensions=(3,),
            execution_modes=(MeshingExecutionMode.SUBPROCESS,),
        ),
        identity,
    )


def _seed_count(path: Path, limits: MeshingLimits, /) -> int:
    if path.stat().st_size > limits.maximum_data_bytes:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "VoroCrust seeds exceed transfer budget.",
        )
    maximum = min(2 * limits.maximum_cells, 40_000_000)
    with path.open(encoding="utf-8", newline="") as stream:
        if stream.readline().rstrip("\r\n") != "x1coord, x2coord, x3coord, radius":
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                "Unexpected VoroCrust seed CSV header.",
            )
        count = 0
        for line in stream:
            count += 1
            if count > maximum:
                raise MeshingFailure(
                    MeshingFailureCategory.RESOURCE_EXHAUSTED,
                    "VoroCrust seed count exceeds its bound.",
                )
            if len(line.rstrip("\r\n").split(",")) != 5:
                raise MeshingFailure(
                    MeshingFailureCategory.CONVERSION_FAILED,
                    "VoroCrust seed rows require five columns.",
                )
    if count == 0:
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "VoroCrust produced no seeds.",
        )
    return count


def _bounded_combination(count: int, choose: int, maximum: int, /) -> int:
    if count < choose:
        return 0
    result = 1
    for factor in range(1, choose + 1):
        result = result * (count - choose + factor) // factor
        if result > maximum:
            return maximum + 1
    return result


def _preflight_native_output(seed_count: int, limits: MeshingLimits, /) -> None:
    vertex_bound = _bounded_combination(
        seed_count,
        4,
        limits.maximum_vertices,
    )
    face_bound = _bounded_combination(
        seed_count,
        2,
        limits.maximum_faces,
    )
    face_width = max(seed_count - 2, 3)
    connectivity_exceeded = face_bound > limits.maximum_connectivity_entries // face_width
    if (
        seed_count > limits.maximum_cells
        or vertex_bound > limits.maximum_vertices
        or face_bound > limits.maximum_faces
        or connectivity_exceeded
    ):
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Conservative VoroCrust output bounds exceed configured limits before native extraction.",
        )


def _vertex_aliases(points: np.ndarray, relative_tolerance: float) -> np.ndarray:
    """Normalize backend vertex aliases without changing retained coordinates."""
    tolerance = relative_tolerance * float(np.linalg.norm(np.ptp(points, axis=0)))
    tree = cKDTree(points)
    parent = np.arange(len(points), dtype=np.int64)

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for index, point in enumerate(points):
        for neighbor in tree.query_ball_point(point, tolerance):
            first, second = root(index), root(neighbor)
            if first != second:
                parent[max(first, second)] = min(first, second)
    aliases = np.asarray([root(index) for index in range(len(points))], dtype=np.int64)
    if np.any(np.linalg.norm(points - points[aliases], axis=1) > tolerance):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Transitive vertex aliases exceed the requested merge tolerance.",
        )
    return aliases


def _read_polyhedra(
    path: Path, limits: MeshingLimits, relative_merge_tolerance: float
) -> tuple[CellMesh, int, int]:
    if path.stat().st_size > limits.maximum_data_bytes:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "VoroCrust output exceeds data budget.",
        )
    with path.open() as stream:
        counts = tuple(map(int, stream.readline().split()))
        if len(counts) != 3 or any(value <= 0 for value in counts):
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                "Invalid VoroCrust output counts.",
            )
        vertex_count, face_count, seed_count = counts
        if (
            vertex_count > limits.maximum_vertices
            or face_count > limits.maximum_faces
            or seed_count > 2 * limits.maximum_cells
        ):
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "VoroCrust output exceeds entity budgets.",
            )
        points = np.asarray(
            [tuple(map(float, stream.readline().split())) for _ in range(vertex_count)]
        )
        seed_data = np.asarray(
            [tuple(map(float, stream.readline().split())) for _ in range(seed_count)]
        )
        if (
            points.shape != (vertex_count, 3)
            or seed_data.shape != (seed_count, 4)
            or not np.all(np.isfinite(points))
            or not np.all(np.isfinite(seed_data))
        ):
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                "Invalid VoroCrust point or seed arrays.",
            )
        interior = np.flatnonzero(seed_data[:, 0] > 0)
        if interior.size == 0 or interior.size > limits.maximum_cells:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "Invalid VoroCrust interior cell count.",
            )
        aliases = _vertex_aliases(points, relative_merge_tolerance)
        collapsed_faces = 0
        cells = {int(index): [] for index in interior}
        entries = 0
        for _ in range(face_count):
            row = np.fromstring(stream.readline(), dtype=np.int64, sep=" ")
            if row.size < 6 or row[0] < 3 or row.size != row[0] + 3:
                raise MeshingFailure(
                    MeshingFailureCategory.CONVERSION_FAILED,
                    "Invalid VoroCrust polygon record.",
                )
            loop = row[1:-2]
            pair = row[-2:]
            entries += len(loop) * 2
            if entries > limits.maximum_connectivity_entries:
                raise MeshingFailure(
                    MeshingFailureCategory.RESOURCE_EXHAUSTED,
                    "VoroCrust connectivity exceeds budget.",
                )
            if (
                np.any(loop < 0)
                or np.any(loop >= vertex_count)
                or np.any(pair < 0)
                or np.any(pair >= seed_count)
            ):
                raise MeshingFailure(
                    MeshingFailureCategory.CONVERSION_FAILED,
                    "VoroCrust connectivity index is out of bounds.",
                )
            loop = aliases[loop]
            loop = loop[loop != np.roll(loop, 1)]
            if len(loop) < 3:
                collapsed_faces += 1
                continue
            vertices = points[loop]
            area = np.sum(
                np.cross(
                    vertices - vertices[0], np.roll(vertices, -1, axis=0) - vertices[0]
                ),
                axis=0,
            )
            for seed in pair:
                if int(seed) in cells:
                    outward = (
                        float(
                            np.dot(area, np.mean(vertices, axis=0) - seed_data[seed, 1:])
                        )
                        > 0
                    )
                    cells[int(seed)].append(loop if outward else loop[::-1])
        if stream.read().strip():
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                "Unexpected trailing VoroCrust output.",
            )
    used = np.unique(np.concatenate([face for cell in cells.values() for face in cell]))
    mapping = np.full(vertex_count, -1, dtype=np.int64)
    mapping[used] = np.arange(len(used))
    mesh = CellMesh.from_polyhedra(
        points[used],
        tuple(tuple(mapping[face] for face in cell) for cell in cells.values()),
        vertex_global_ids=used,
        cell_global_ids=interior,
    )
    if mesh.entity_set(1).count > limits.maximum_edges:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "VoroCrust edges exceed the entity budget.",
        )
    return mesh, vertex_count - len(np.unique(aliases)), collapsed_faces


class VoroCrustProvider:
    """Real VoroCrust sampling and public-API packed polyhedron extraction.

    Radius is the backend sphere-sizing bound, not a guaranteed cell-edge size.
    Material/selection transfer is not inferred from provisional seed colors.
    """

    def __init__(
        self,
        executable: str | Path = "vc_mesh",
        extractor: str | Path = "phydrax-vorocrust",
    ):
        self.executable = _executable(executable)
        self.extractor = _executable(extractor)

    def info(self) -> MeshingProviderInfo:
        return _extractor_identity(self.extractor)[0]

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
            raise ValueError("limits exceed the VoroCrust bridge hard bounds.")
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
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "VoroCrust source exceeds configured entity or byte limits.",
            )
        faces = np.concatenate(
            [np.asarray(block.vertices) for block in surface.mesh.blocks]
        )
        points = np.asarray(surface.mesh.coordinates)
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
        provider, extractor_identity = _extractor_identity(self.extractor)
        deadline = monotonic() + limits.maximum_wall_seconds
        with TemporaryDirectory(prefix="phydrax-vorocrust-") as temporary:
            directory = Path(temporary)
            with (directory / "surface.obj").open("w") as stream:
                for point in points:
                    stream.write(
                        "v "
                        + " ".join(format(float(value), ".17g") for value in point)
                        + "\n"
                    )
                for face in faces:
                    stream.write(
                        "f " + " ".join(str(int(value) + 1) for value in face) + "\n"
                    )
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
            maximum_seed_count = _seed_count(seeds, limits)
            _preflight_native_output(maximum_seed_count, limits)
            _run(
                [
                    self.extractor,
                    "seeds.csv",
                    "mesh.raw",
                    str(maximum_seed_count),
                    str(limits.maximum_vertices),
                    str(limits.maximum_faces),
                    str(limits.maximum_connectivity_entries),
                    str(limits.maximum_data_bytes),
                ],
                directory,
                deadline,
            )
            mesh, merged_vertices, collapsed_faces = _read_polyhedra(
                directory / "mesh.raw", limits, options.relative_merge_tolerance
            )
        native = certify_cell_mesh(mesh, surface.metadata.coordinate_contract)
        volume = float(np.sum(np.asarray(native.quality.evaluation.measures)))
        error = abs(volume - source_volume) / source_volume
        if error > options.relative_volume_tolerance:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "VoroCrust output does not preserve source enclosed volume.",
            )
        provenance = SemanticProvenance(
            {
                "kind": "vorocrust-volume",
                "source": surface.mesh.mesh_id,
                "extractor_identity": (
                    extractor_identity["source_sha256"],
                    extractor_identity["config_sha256"],
                    extractor_identity["library_sha256"],
                ),
                "options": (
                    options.maximum_radius,
                    options.lipschitz_constant,
                    options.feature_angle_degrees,
                    options.relative_volume_tolerance,
                    options.relative_merge_tolerance,
                    options.require_native_output_preallocation,
                ),
            }
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
