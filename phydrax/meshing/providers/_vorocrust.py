#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np
from scipy.spatial import cKDTree

from ..._identity import SemanticProvenance
from ...discretization import CellMesh
from ...geometry.surface import SurfaceModel
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
            raise MeshingFailure(
                MeshingFailureCategory.TIMED_OUT, "VoroCrust execution timed out."
            ) from error
    if result.returncode:
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            f"VoroCrust process exited with code {result.returncode}.",
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
        completed = subprocess.run(
            [self.extractor, "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return MeshingProviderInfo(
            "vorocrust",
            completed.stdout.strip(),
            "BSD-3-Clause",
            operations=(MeshingOperation.MESH_VOLUME,),
            source_kinds=(MeshingSourceKind.SURFACE,),
            capabilities=(MeshingCapability.POLYHEDRAL,),
            cell_kinds=("polyhedron",),
            dimensions=(3,),
            execution_modes=(MeshingExecutionMode.SUBPROCESS,),
        )

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
            if seeds.stat().st_size > limits.maximum_data_bytes:
                raise MeshingFailure(
                    MeshingFailureCategory.RESOURCE_EXHAUSTED,
                    "VoroCrust seeds exceed transfer budget.",
                )
            _run([self.extractor, "seeds.csv", "mesh.raw"], directory, deadline)
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
        provider = self.info()
        provenance = SemanticProvenance(
            {
                "kind": "vorocrust-volume",
                "source": surface.mesh.mesh_id,
                "options": (
                    options.maximum_radius,
                    options.lipschitz_constant,
                    options.feature_angle_degrees,
                    options.relative_volume_tolerance,
                    options.relative_merge_tolerance,
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
                enforced_limits=("wall_time", "output_entities", "output_bytes"),
                unenforced_limits=("provider_workspace",),
            ),
            MeshingDerivativeMode.NONDIFFERENTIABLE,
            provenance,
        )


__all__ = ["VoroCrustProvider", "VoroCrustOptions"]
