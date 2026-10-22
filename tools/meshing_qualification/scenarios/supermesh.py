# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from typing import TypedDict

import numpy as np

import phydrax as phx
from phydrax.meshing import meshcore_available
from tools.meshing_qualification._controls import QualificationControls
from tools.meshing_qualification._fixtures import (
    _HEXAHEDRON_CORNERS,
    _jittered_unit_square_triangles,
    _kuhn_tetrahedra,
)
from tools.meshing_qualification._records import _quality_summary, _result_identity
from tools.meshing_qualification._runtime import (
    _contract,
    _measured,
    _missing_dependency,
    _runtime_identity,
    qualification_failure,
    StageMeasurement,
)


_MESHCORE_DETAIL = "Set PHYDRAX_MESHCORE_LIBRARY or install phydrax[meshcore]."


class _SupermeshCaseRecord(TypedDict):
    source: dict[str, object]
    target: dict[str, object]
    identities: dict[str, str]
    quality: dict[str, dict[str, object]]
    coverage: dict[str, str | float | int]
    conservation: dict[str, float | dict[str, float]]
    pairs: dict[str, int]
    predicates: dict[str, str | int]
    resources: dict[str, int | StageMeasurement]


def _unit_lattice_cells(
    cells_per_axis: int, dimension: int, /
) -> tuple[np.ndarray, np.ndarray]:
    """Unit square/cube lattice points and quadrilateral/hexahedron corners."""

    count = cells_per_axis
    axis = np.linspace(0.0, 1.0, count + 1)
    points = np.stack(np.meshgrid(*(axis,) * dimension, indexing="ij"), axis=-1).reshape(
        (-1, dimension)
    )
    index = np.arange((count + 1) ** dimension, dtype=np.int32).reshape(
        (count + 1,) * dimension
    )
    offsets = np.unique(_HEXAHEDRON_CORNERS[:, :dimension], axis=0, return_index=True)[1]
    corners = _HEXAHEDRON_CORNERS[np.sort(offsets), :dimension]
    cells = np.stack(
        [
            index[tuple(slice(start, start + count) for start in corner)].reshape((-1,))
            for corner in corners
        ],
        axis=1,
    )
    return points, cells


def _jittered_cube_tetrahedra(
    cells_per_axis: int, seed: int, /
) -> phx.discretization.CellMesh:
    """Kuhn tetrahedra of the unit cube with interior vertices jittered in place."""

    points, tetrahedra = _kuhn_tetrahedra(cells_per_axis)
    interior = np.all((points > 0.0) & (points < 1.0), axis=1)
    jitter = np.random.default_rng(seed).uniform(-0.1, 0.1, points.shape)
    points[interior] += jitter[interior] / cells_per_axis
    return phx.discretization.CellMesh.from_tetrahedra(points, tetrahedra)


def _supermesh_case(
    source: phx.discretization.CellMesh, target: phx.discretization.CellMesh, /
) -> _SupermeshCaseRecord:
    """Certify one common refinement and recount its measure and moment ledgers."""

    contract = _contract()
    source_result, source_certify = _measured(
        lambda: phx.meshing.certify_cell_mesh(source, contract)
    )
    target_result, target_certify = _measured(
        lambda: phx.meshing.certify_cell_mesh(target, contract)
    )
    policy = phx.geometry.CommonRefinementPolicy(overlap_simplices=True)
    refinement, refinement_record = _measured(
        lambda: phx.geometry.prepare_common_refinement(source, target, policy=policy)
    )

    evidence = refinement.evidence

    if not refinement.succeeded:
        raise qualification_failure(
            "_supermesh_case",
            (f"Common refinement failed: {refinement.status.name}: {evidence.reason}."),
            {"refinement": refinement},
        )

    dimension = refinement.dimension

    volumes = np.asarray(refinement.volumes, dtype=np.float64)

    moments = np.asarray(refinement.first_moments, dtype=np.float64)
    ledgers: dict[str, dict[str, float]] = {}
    for role, cells, measures, first in (
        (
            "source",
            refinement.source_cells,
            refinement.source_measures,
            refinement.source_first_moments,
        ),
        (
            "target",
            refinement.target_cells,
            refinement.target_measures,
            refinement.target_first_moments,
        ),
    ):
        cells = np.asarray(cells, dtype=np.int64)
        measures = np.asarray(measures, dtype=np.float64)
        first = np.asarray(first, dtype=np.float64)
        covered = np.zeros_like(measures)
        np.add.at(covered, cells, volumes)
        covered_moments = np.zeros_like(first)
        np.add.at(covered_moments, cells, moments)
        total = float(np.sum(measures))
        ledgers[role] = {
            "measure": total,
            "total_measure_relative_defect": abs(float(np.sum(volumes)) - total) / total,
            "maximum_cell_measure_relative_defect": float(
                np.max(np.abs(covered - measures) / measures)
            ),
            # Cell centroid integrals: the overlap first moments of each cell sum
            # to the cell's own first moment.
            "maximum_cell_first_moment_relative_defect": float(
                np.max(np.abs(covered_moments - first)) / np.max(np.abs(first))
            ),
        }
    # The unit square/cube has measure one and centroid (1/2, ..., 1/2).
    domain = {
        "measure_defect": abs(float(np.sum(volumes)) - 1.0),
        "first_moment_defect": float(np.max(np.abs(np.sum(moments, axis=0) - 0.5))),
    }

    offsets = np.asarray(refinement.simplex_offsets, dtype=np.int64)

    simplices = np.asarray(refinement.simplices, dtype=np.float64)
    signed = np.linalg.det(simplices[:, 1:] - simplices[:, :1]) / np.prod(
        np.arange(1, dimension + 1)
    )
    partition = np.zeros_like(volumes)
    np.add.at(partition, np.repeat(np.arange(volumes.size), np.diff(offsets)), signed)
    partition_defect = float(np.max(np.abs(partition - volumes) / volumes))
    defects = (
        *(
            value
            for ledger in ledgers.values()
            for key, value in ledger.items()
            if key != "measure"
        ),
        *domain.values(),
        partition_defect,
        evidence.maximum_relative_source_defect,
        evidence.maximum_relative_target_defect,
    )
    if (
        max(defects) > 1.0e-12
        or evidence.source_gap_count
        or evidence.target_gap_count
        or evidence.source_double_count
        or evidence.target_double_count
        or evidence.uncertain_predicate_count
        or evidence.invalid_cell_count
        or evidence.intersection_failure_count
        or not (source_result.audit.passed and target_result.audit.passed)
    ):
        raise qualification_failure(
            "_supermesh_case",
            (
                f"Common refinement ledgers failed: {ledgers}, {domain}, "
                f"partition {partition_defect}, evidence {evidence.reason}."
            ),
            {
                "defects": defects,
                "evidence": evidence,
                "source_result": source_result,
                "target_result": target_result,
            },
        )
    return {
        "source": {**_result_identity(source_result), "mesh_id": source.mesh_id},
        "target": {**_result_identity(target_result), "mesh_id": target.mesh_id},
        "identities": {
            "refinement_id": refinement.refinement_id,
            "evidence_id": evidence.evidence_id,
            "policy_id": policy.policy_id,
            "source_topology_id": refinement.source_topology_id,
            "target_topology_id": refinement.target_topology_id,
        },
        "quality": {
            "source": _quality_summary(source_result.quality),
            "target": _quality_summary(target_result.quality),
        },
        "coverage": {
            "status": refinement.status.name,
            "reason": evidence.reason,
            "maximum_relative_source_defect": evidence.maximum_relative_source_defect,
            "maximum_relative_target_defect": evidence.maximum_relative_target_defect,
            "source_gap_count": evidence.source_gap_count,
            "target_gap_count": evidence.target_gap_count,
            "source_double_count": evidence.source_double_count,
            "target_double_count": evidence.target_double_count,
        },
        "conservation": {
            **ledgers,
            "unit_domain": domain,
            "simplex_partition_maximum_relative_defect": partition_defect,
        },
        "pairs": {
            "candidate": evidence.candidate_pair_count,
            "accepted": evidence.accepted_pair_count,
            "piece_pairs": evidence.piece_pair_count,
            "entries": refinement.entry_count,
            "overlap_simplices": simplices.shape[0],
        },
        "predicates": {
            "mode": policy.predicate_mode.value,
            "uncertain": evidence.uncertain_predicate_count,
            "invalid_cells": evidence.invalid_cell_count,
            "intersection_failures": evidence.intersection_failure_count,
        },
        "resources": {
            "source_certify": source_certify,
            "target_certify": target_certify,
            "common_refinement": refinement_record,
            "retained_bytes": evidence.retained_bytes,
            "working_bytes": evidence.working_bytes,
        },
    }


def qualify_supermesh() -> dict[str, object]:

    if not meshcore_available():
        return _missing_dependency("phydrax-meshcore", _MESHCORE_DETAIL)

    square_points, quadrilaterals = _unit_lattice_cells(5, 2)
    cube_points, hexahedra = _unit_lattice_cells(2, 3)
    meshes = {
        "triangles-quadrilaterals-2d": (
            _jittered_unit_square_triangles(6, 1),
            phx.discretization.CellMesh(
                square_points,
                (phx.discretization.CellBlock("quads", "quadrilateral", quadrilaterals),),
            ),
        ),
        "triangles-triangles-2d": (
            _jittered_unit_square_triangles(6, 1),
            _jittered_unit_square_triangles(5, 2),
        ),
        "tetrahedra-tetrahedra-3d": (
            _jittered_cube_tetrahedra(3, 3),
            _jittered_cube_tetrahedra(2, 4),
        ),
        "hexahedra-tetrahedra-3d": (
            phx.discretization.CellMesh(
                cube_points,
                (phx.discretization.CellBlock("hexahedra", "hexahedron", hexahedra),),
            ),
            _jittered_cube_tetrahedra(3, 3),
        ),
    }
    cases = {name: _supermesh_case(*pair) for name, pair in meshes.items()}
    sections = ("source", "target", "quality", "conservation", "resources")
    return {
        "status": "passed",
        "source": {name: case["source"] for name, case in cases.items()},
        "target": {name: case["target"] for name, case in cases.items()},
        "runtime": _runtime_identity(),
        "quality": {
            "before": {name: case["quality"]["source"] for name, case in cases.items()},
            "after": {name: case["quality"]["target"] for name, case in cases.items()},
        },
        "conservation": {name: case["conservation"] for name, case in cases.items()},
        "resources": {name: case["resources"] for name, case in cases.items()},
        "cases": {
            name: {key: value for key, value in case.items() if key not in sections}
            for name, case in cases.items()
        },
    }


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return qualify_supermesh()
