#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping, Sequence

import numpy as np

from ..._identity import SemanticProvenance
from ..._physical import SpatialCoordinateContract
from ...discretization import CellMesh
from ...discretization._partition import CellPartition
from .._assembly import MeshPart
from .._canonical import certify_cell_mesh
from .._contracts import (
    MeshingCapability,
    MeshingDerivativeMode,
    MeshingExecutionMode,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingOperation,
    MeshingProviderInfo,
    MeshingSourceKind,
)
from .._distribution import MeshDistribution
from .._result import CellMeshingResult, MeshingRuntimeInfo
from .._scope import MeshingEntityKind, MeshingScope
from .._sizing import MeshMetricField
from .._trace import (
    MeshingStageKind,
    MeshingStageReport,
    MeshingStageStatus,
    MeshingTrace,
)


@dataclass(frozen=True, slots=True)
class OmegaHPartition:
    """Native local ordering, including ghosts; owners are (rank, local index).

    Membership in a rank's vertex_ids/cell_ids is its residence. These records
    preserve Omega_h's vertex ownership, which cannot be inferred from cell
    ownership. IDs are authoritative for the output revision, not lineage IDs.
    """

    rank: int
    vertex_ids: tuple[int, ...]
    vertex_owner_ranks: tuple[int, ...]
    vertex_owner_indices: tuple[int, ...]
    cell_ids: tuple[int, ...]
    cell_owner_ranks: tuple[int, ...]
    cell_owner_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class OmegaHAdaptationResult:
    target: CellMeshingResult
    distribution: MeshDistribution
    metric: MeshMetricField
    partitions: tuple[OmegaHPartition, ...]
    source_mesh_id: str
    iterations: int
    # Omega_h renumbers globals during adaptation. Coincident numeric IDs must
    # never be treated as preserved entities or used to fabricate a transfer.
    lineage_status: str = "unknown"


def _run(command: Sequence[str], timeout: float, environment: Mapping[str, str]) -> str:
    try:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, **environment},
            start_new_session=os.name == "posix",
        )
    except OSError as error:
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_UNAVAILABLE, str(error)
        ) from error
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        output, _ = process.communicate()
        raise MeshingFailure(
            MeshingFailureCategory.TIMED_OUT,
            f"Omega_h exceeded {timeout:g} seconds. {output[-4000:]}",
        ) from error
    if process.returncode:
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            f"Omega_h exited with status {process.returncode}: {output[-4000:]}",
        )
    return output


def _unpack_metric(values: np.ndarray, dimension: int) -> np.ndarray:
    # INRIA lower-triangular row-major: xx,xy,yy[,xz,yz,zz].
    rows, columns = np.tril_indices(dimension)
    result = np.zeros((len(values), dimension, dimension), dtype=np.float64)
    result[:, rows, columns] = values
    result[:, columns, rows] = values
    return result


def _merge_partitions(records: list[dict], dimension: int):
    """Reject contradictory copies, missing owners and incomplete global covers."""
    partitions = []
    vertices, cells = {}, {}
    metric_width = dimension * (dimension + 1) // 2
    fields = (
        "vertex_ids",
        "vertex_owner_ranks",
        "vertex_owner_indices",
        "cell_ids",
        "cell_owner_ranks",
        "cell_owner_indices",
    )
    for rank, record in enumerate(records):
        if (record["protocol"], record["rank"], record["size"], record["dimension"]) != (
            1,
            rank,
            len(records),
            dimension,
        ):
            raise ValueError("Omega_h rank output header is inconsistent.")
        if any(
            any(type(value) is not int for value in record[field]) for field in fields
        ):
            raise ValueError("Omega_h IDs and ownership must be integer arrays.")
        partition = OmegaHPartition(rank, *(tuple(record[field]) for field in fields))
        partitions.append(partition)
        coordinates = np.asarray(record["coordinates"], dtype=np.float64).reshape(
            (-1, dimension)
        )
        metric = np.asarray(record["metric"], dtype=np.float64).reshape(
            (-1, metric_width)
        )
        connectivity = np.asarray(record["cells"], dtype=np.int64).reshape(
            (-1, dimension + 1)
        )
        if len(coordinates) != len(partition.vertex_ids) or len(metric) != len(
            coordinates
        ):
            raise ValueError("Omega_h vertex arrays do not align.")
        if len(connectivity) != len(partition.cell_ids):
            raise ValueError("Omega_h cell arrays do not align.")
        if np.any(connectivity < 0) or np.any(connectivity >= len(coordinates)):
            raise ValueError("Omega_h connectivity references unavailable vertices.")
        if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(metric)):
            raise ValueError("Omega_h returned nonfinite coordinates or metric.")
        for kind, table, identifiers, owners, indices in (
            (
                "vertex",
                vertices,
                partition.vertex_ids,
                partition.vertex_owner_ranks,
                partition.vertex_owner_indices,
            ),
            (
                "cell",
                cells,
                partition.cell_ids,
                partition.cell_owner_ranks,
                partition.cell_owner_indices,
            ),
        ):
            if len(set(identifiers)) != len(identifiers) or any(
                value < 0 for value in identifiers
            ):
                raise ValueError(
                    "Omega_h local global IDs must be unique and nonnegative."
                )
            if len(owners) != len(identifiers) or len(indices) != len(identifiers):
                raise ValueError("Omega_h ownership arrays do not align.")
            for local, (identifier, owner, index) in enumerate(
                zip(identifiers, owners, indices, strict=True)
            ):
                if not 0 <= owner < len(records) or index < 0:
                    raise ValueError("Omega_h returned an invalid owner rank/index.")
                owner_record = records[owner]
                owner_ids = owner_record[f"{kind}_ids"]
                if (
                    index >= len(owner_ids)
                    or owner_ids[index] != identifier
                    or owner_record[f"{kind}_owner_ranks"][index] != owner
                    or owner_record[f"{kind}_owner_indices"][index] != index
                ):
                    raise ValueError(
                        "Omega_h owner reference does not resolve to its authoritative copy."
                    )
                payload = (
                    (coordinates[local], metric[local])
                    if kind == "vertex"
                    else (
                        tuple(
                            partition.vertex_ids[value] for value in connectivity[local]
                        ),
                    )
                )
                previous = table.get(identifier)
                if previous is not None:
                    if previous[:2] != (owner, index) or any(
                        not np.array_equal(left, right)
                        for left, right in zip(previous[2], payload, strict=True)
                    ):
                        raise ValueError("Omega_h resident copies disagree.")
                else:
                    table[identifier] = (owner, index, payload)
    for record in records:
        if record["global_vertices"] != len(vertices) or record["global_cells"] != len(
            cells
        ):
            raise ValueError(
                "Omega_h partitions do not cover the declared global carrier."
            )
        if record["iterations"] != records[0]["iterations"]:
            raise ValueError("Omega_h ranks disagree on collective adaptation progress.")
    vertex_ids = sorted(vertices)
    cell_ids = sorted(cells)
    vertex_lookup = {identifier: index for index, identifier in enumerate(vertex_ids)}
    constructor = CellMesh.from_triangles if dimension == 2 else CellMesh.from_tetrahedra
    mesh = constructor(
        np.asarray([vertices[value][2][0] for value in vertex_ids]),
        np.asarray(
            [
                [vertex_lookup[value] for value in cells[identifier][2][0]]
                for identifier in cell_ids
            ]
        ),
        vertex_global_ids=np.asarray(vertex_ids, dtype=np.int64),
        cell_global_ids=np.asarray(cell_ids, dtype=np.int64),
        numeric_version="omega_h-adapted",
    )
    metric = _unpack_metric(
        np.asarray([vertices[value][2][1] for value in vertex_ids]), dimension
    )
    owners = np.asarray([cells[value][0] for value in cell_ids], dtype=np.int32)
    return mesh, metric, owners, tuple(partitions)


class OmegaHProvider:
    """Adapt affine planar triangles or volume tetrahedra using real Omega_h.

    Build ``native/omega_h`` with CMake against an installed Omega_h CMake
    package, then set PHYDRAX_OMEGA_H_EXECUTABLE or pass the executable path.
    The official Python bindings omit owner/build/export APIs; the small bridge
    uses public C++ APIs and the official rank-zero import/balance pattern.

    Input is a global carrier. Adaptation runs collectively in the requested MPI
    world, returning an audited global carrier and exact native rank residence.
    Import and result aggregation are host-side, not a distributed-memory I/O
    claim. Boundary features are classified geometrically by sharp angle; CAD,
    semantic label transfer and differentiation are deliberately not claimed.
    ``maximum_gradation`` is passed as Omega_h's metric-gradation rate (not a
    neighboring-edge size ratio); grading can modify the requested metric.

    Omega_h 9.34.13 can abort in coarsening on very small MPI carriers (observed
    with two triangles on two ranks). Native failures propagate; this adapter
    does not silently switch to serial, disable coarsening, or change the mesh.
    """

    def __init__(
        self,
        executable: str | os.PathLike[str] | None = None,
        /,
        *,
        mpi_launcher: Sequence[str] = ("mpiexec",),
        environment: Mapping[str, str] | None = None,
        timeout: float = 300.0,
    ):
        self.executable = str(executable) if executable is not None else None
        self.mpi_launcher = tuple(str(value) for value in mpi_launcher)
        self.environment = dict(environment or {})
        self.timeout = float(timeout)
        if not np.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be finite and positive.")

    def _runtime(self) -> tuple[str, dict, MeshingProviderInfo]:
        requested = self.executable or os.environ.get(
            "PHYDRAX_OMEGA_H_EXECUTABLE", "phydrax_omega_h"
        )
        executable = shutil.which(requested)
        if executable is None:
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_UNAVAILABLE,
                "Build phydrax/meshing/providers/native/omega_h against Omega_h and set "
                "PHYDRAX_OMEGA_H_EXECUTABLE to the installed phydrax_omega_h executable.",
            )
        try:
            details = json.loads(
                _run((executable, "--version"), self.timeout, self.environment)
            )
            if (
                details["protocol"] != 1
                or type(details["mpi"]) is not bool
                or not details["version"]
            ):
                raise ValueError("Unsupported Omega_h bridge protocol.")
        except (ValueError, KeyError, TypeError) as error:
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_UNAVAILABLE, str(error)
            ) from error
        capabilities = (
            MeshingCapability.ANISOTROPIC_METRIC,
            MeshingCapability.DETERMINISTIC,
        )
        if details["mpi"]:
            capabilities += (MeshingCapability.PARALLEL, MeshingCapability.DISTRIBUTED)
        info = MeshingProviderInfo(
            "omega_h",
            details["version"],
            "BSD-2-Clause",
            operations=(MeshingOperation.REMESH_SURFACE, MeshingOperation.ADAPT_VOLUME),
            source_kinds=(MeshingSourceKind.CELL_MESH,),
            capabilities=capabilities,
            cell_kinds=("triangle", "tetrahedron"),
            dimensions=(2, 3),
            execution_modes=(MeshingExecutionMode.SUBPROCESS,),
        )
        return executable, details, info

    def info(self) -> MeshingProviderInfo:
        return self._runtime()[2]

    def execute(
        self,
        mesh: CellMesh,
        metric: MeshMetricField,
        coordinate_contract: SpatialCoordinateContract,
        /,
        *,
        ranks: int = 1,
        feature_angle: float = np.pi / 4,
        maximum_iterations: int = 100,
    ) -> OmegaHAdaptationResult:
        if not isinstance(mesh, CellMesh) or not isinstance(metric, MeshMetricField):
            raise TypeError("Omega_h requires CellMesh and MeshMetricField values.")
        dimension = mesh.topological_dimension
        kind = "triangle" if dimension == 2 else "tetrahedron"
        if (
            dimension not in (2, 3)
            or mesh.ambient_dimension != dimension
            or any(block.cell_kind != kind for block in mesh.blocks)
        ):
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
                "Omega_h supports affine planar triangles and volume tetrahedra only.",
            )
        if (
            type(ranks) is not int
            or ranks < 1
            or type(maximum_iterations) is not int
            or maximum_iterations < 1
        ):
            raise ValueError("ranks and maximum_iterations must be positive integers.")
        angle = float(feature_angle)
        if not np.isfinite(angle) or not 0 < angle < np.pi:
            raise ValueError("feature_angle must be in (0, pi) radians.")
        vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        scope = metric.scope
        if (
            scope.source_id != mesh.mesh_id
            or scope.source_revision != mesh.numeric_version
            or scope.entity_kind is not MeshingEntityKind.MESH
            or scope.entity_dimension != 0
            or scope.entity_set_id != mesh.entity_set(0).entity_set_id
            or not np.array_equal(scope.entity_ids, np.sort(vertex_ids))
        ):
            raise ValueError(
                "Metric scope must cover exactly this mesh revision's vertices."
            )
        values = np.asarray(metric.values, dtype=np.float64)
        if values.shape != (len(vertex_ids), dimension, dimension):
            raise ValueError(
                "Metric matrix dimension must match the Omega_h mesh dimension."
            )
        eigenvalues = np.linalg.eigvalsh(values)
        if (
            np.any(eigenvalues < metric.maximum_size**-2 * (1 - 1e-12))
            or np.any(eigenvalues > metric.minimum_size**-2 * (1 + 1e-12))
            or np.any(
                np.sqrt(eigenvalues[:, -1] / eigenvalues[:, 0])
                > metric.maximum_anisotropy * (1 + 1e-12)
            )
        ):
            raise ValueError(
                "Input metrics exceed their declared size/anisotropy bounds."
            )
        source = certify_cell_mesh(mesh, coordinate_contract)
        executable, details, provider = self._runtime()
        if ranks > 1 and (not details["mpi"] or not self.mpi_launcher):
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
                "Distributed adaptation requires an MPI-enabled bridge and launcher.",
            )
        order = np.argsort(vertex_ids)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        connectivity = np.concatenate(
            [np.asarray(block.vertices) for block in source.mesh.blocks]
        )
        cell_ids = np.concatenate(
            [np.asarray(block.global_ids) for block in source.mesh.blocks]
        )
        packed = values[:, *np.tril_indices(dimension)]
        with TemporaryDirectory(prefix="phydrax-omega-h-") as directory:
            input_path = Path(directory) / "input.txt"
            with input_path.open("w", encoding="utf-8") as stream:
                stream.write(
                    f"PHYDRAX_OMEGA_H_1\n{dimension} {len(order)} {len(connectivity)} "
                    f"{metric.maximum_gradation:.17g} {angle:.17g} {maximum_iterations}\n"
                )
                for array, fmt in (
                    (vertex_ids[order], "%d"),
                    (np.asarray(mesh.coordinates)[order], "%.17g"),
                    (inverse[connectivity], "%d"),
                    (cell_ids, "%d"),
                    (packed, "%.17g"),
                ):
                    np.savetxt(stream, array, fmt=fmt)
            command = (executable, str(input_path), directory)
            if ranks > 1:
                command = (*self.mpi_launcher, "-n", str(ranks), *command)
            _run(command, self.timeout, self.environment)
            try:
                records = [
                    json.loads((Path(directory) / f"rank-{rank}.json").read_text())
                    for rank in range(ranks)
                ]
                target_mesh, target_values, owners, partitions = _merge_partitions(
                    records, dimension
                )
            except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
                raise MeshingFailure(
                    MeshingFailureCategory.CONVERSION_FAILED, str(error)
                ) from error
        certified = certify_cell_mesh(target_mesh, coordinate_contract)
        provenance = SemanticProvenance(
            {
                "kind": "omega-h-metric-adaptation",
                "source_mesh": mesh.mesh_id,
                "input_metric": metric.metric_id,
                "omega_h_commit": details["commit"],
                "ranks": ranks,
                "feature_angle": angle,
                "lineage": "unknown",
                "global_ids": "authoritative-output-revision-only",
                "target_metric": "Omega_h-gradation-limited",
                "iterations": records[0]["iterations"],
            }
        )
        target = CellMeshingResult(
            certified.mesh,
            certified.geometry,
            coordinate_contract,
            certified.audit,
            certified.quality,
            certified.compliance,
            MeshingTrace(
                (
                    MeshingStageReport(
                        MeshingStageKind.OPTIMIZATION,
                        MeshingStageStatus.PASSED,
                        input_ids=(mesh.mesh_id, metric.metric_id),
                        output_ids=(certified.mesh.mesh_id,),
                    ),
                    *certified.trace.stages,
                )
            ),
            provider,
            MeshingRuntimeInfo(
                provider.provider_id,
                provider.version,
                MeshingExecutionMode.SUBPROCESS,
                deterministic=True,
            ),
            MeshingDerivativeMode.NONDIFFERENTIABLE,
            provenance,
        )
        distribution = MeshDistribution(
            MeshPart("omega_h", target),
            CellPartition(owners, ranks),
            halo_global_ids=tuple(
                np.asarray(
                    [
                        identifier
                        for identifier, owner in zip(
                            part.cell_ids, part.cell_owner_ranks, strict=True
                        )
                        if owner != part.rank
                    ],
                    dtype=np.int64,
                )
                for part in partitions
            ),
        )
        target_scope = MeshingScope(
            target.mesh.mesh_id,
            target.mesh.numeric_version,
            MeshingEntityKind.MESH,
            0,
            target.mesh.entity_set(0).entity_set_id,
            target.mesh.vertex_global_ids,
        )
        target_metric = MeshMetricField(
            target_scope,
            target_values,
            minimum_size=metric.minimum_size,
            maximum_size=metric.maximum_size,
            maximum_anisotropy=metric.maximum_anisotropy,
            maximum_gradation=metric.maximum_gradation,
        )
        return OmegaHAdaptationResult(
            target,
            distribution,
            target_metric,
            partitions,
            mesh.mesh_id,
            records[0]["iterations"],
        )


__all__ = ["OmegaHAdaptationResult", "OmegaHPartition", "OmegaHProvider"]
