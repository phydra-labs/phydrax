from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

import phydrax as phx
from phydrax._execution_resources import MemoryBudgetError
from phydrax._meshcore import meshcore_available
from phydrax.discretization import CellBlock, CellMesh
from tools.meshing_qualification.scenarios import reconstruction


def _tetrahedra(count: int) -> CellMesh:
    """``count`` disjoint positively oriented unit tetrahedra."""
    corners = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    coordinates = np.concatenate(
        [corners + (2.0 * index, 0.0, 0.0) for index in range(count)]
    )
    cells = np.arange(4 * count, dtype=np.int64).reshape((count, 4))
    return CellMesh(
        coordinates,
        (CellBlock("tets", "tetrahedron", cells, global_ids=np.arange(count)),),
    )


def test_refinement_admission_charges_every_declared_child() -> None:
    mesh = _tetrahedra(3)
    required = (
        3
        * reconstruction._REFINED_CHILDREN_PER_TETRAHEDRON
        * reconstruction._REFINEMENT_BYTES_PER_TARGET_CELL
    )
    assert reconstruction._refinement_admission(mesh, required) == required
    with pytest.raises(MemoryBudgetError) as refused:
        reconstruction._refinement_admission(mesh, required - 1)
    assert refused.value.budget == "refinement_memory_bytes"
    assert refused.value.required_bytes == required
    assert refused.value.limit_bytes == required - 1


def test_refinement_refuses_before_any_adaptation_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Result:
        mesh = _tetrahedra(2)

    def unreachable(*args: object, **kwargs: object) -> None:
        raise AssertionError("A refused refinement must not prepare an adaptation.")

    monkeypatch.setattr(
        reconstruction.phx.meshing, "prepare_mesh_adaptation", unreachable
    )
    with pytest.raises(MemoryBudgetError):
        reconstruction._refine_every_cell(None, _Result(), 1)  # ty: ignore[invalid-argument-type]


def test_larger_torus_admits_measured_refinement_and_changes_plan_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = reconstruction._profile_id("torus", 2)
    budget = reconstruction._REFINEMENT_MEMORY_BUDGET_BYTES[profile]
    mesh = _tetrahedra(84_366)
    required = reconstruction._refinement_admission(mesh, budget)
    assert required == 33_174_061_056
    limits = reconstruction._archive_limits(profile)
    current = reconstruction._resource_plan_id(profile, limits)
    monkeypatch.setitem(reconstruction._REFINEMENT_MEMORY_BUDGET_BYTES, profile, 10 << 30)
    with pytest.raises(MemoryBudgetError) as refused:
        reconstruction._refinement_admission(
            mesh, reconstruction._REFINEMENT_MEMORY_BUDGET_BYTES[profile]
        )
    assert refused.value.required_bytes == required
    assert refused.value.limit_bytes == 10 << 30
    assert reconstruction._resource_plan_id(profile, limits) != current


@pytest.mark.skipif(not meshcore_available(), reason="native meshcore unavailable")
def test_checkpoint_failure_retains_actual_precommit_fields_and_solve() -> None:
    points = np.asarray(tuple(np.ndindex(2, 2, 2)), dtype=np.float64)
    cells = []
    for order in permutations(range(3)):
        corner = np.zeros(3, dtype=np.int64)
        vertices = [0]
        for direction in order:
            corner[direction] += 1
            vertices.append(corner @ np.asarray((4, 2, 1), dtype=np.int64))
        cells.append(vertices)
    connectivity = np.asarray(cells, dtype=np.int32)
    corners = points[connectivity]
    negative = np.linalg.det(corners[:, 1:] - corners[:, :1]) < 0.0
    connectivity[negative] = connectivity[negative][:, (0, 2, 1, 3)]
    source = phx.meshing.certify_cell_mesh(
        CellMesh.from_tetrahedra(points, connectivity), phx.SpatialCoordinateContract.si()
    )
    adaptation = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source,
            phx.meshing.MarkedMeshAdaptation(
                np.concatenate(
                    [np.asarray(block.global_ids) for block in source.mesh.blocks]
                )
            ),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_MIXED
            ),
        )
    )
    source_values = np.sum(points * points, axis=1)
    form = phx.equations.FiniteElementForm(
        "checkpoint-source",
        "u",
        (phx.equations.DiffusionAction("u", 1.0), phx.equations.SourceAction("u", -6.0)),
    )
    retained: list[reconstruction._ReanalysisCheckpoint] = []

    def refuse(checkpoint: reconstruction._ReanalysisCheckpoint) -> None:
        retained.append(checkpoint)
        raise OSError("checkpoint writer refused")

    with pytest.raises(OSError, match="checkpoint writer refused"):
        reconstruction._transfer_accepted_state(
            source,
            source_values,
            form,
            adaptation,
            "checkpoint-failure",
            256 << 20,
            on_reanalysis=refuse,
        )

    checkpoint = retained[0]
    fresh = checkpoint.fresh
    target_points = np.asarray(adaptation.target.mesh.coordinates)
    np.testing.assert_allclose(
        np.asarray(checkpoint.fields[0]), target_points.sum(axis=1)
    )
    np.testing.assert_allclose(np.asarray(checkpoint.fields[1]), reconstruction._DENSITY)
    np.testing.assert_allclose(
        np.asarray(checkpoint.fields[2]), target_points[:, 0] + 0.5
    )
    assert np.all(np.asarray(fresh.solution.successful))
    assert np.all(np.asarray(fresh.check.valid))
    np.testing.assert_array_equal(
        fresh.values,
        reconstruction._vertex_values(
            fresh.space,
            adaptation.target.mesh,
            np.asarray(fresh.problem.expand(fresh.solution.value)),
        ),
    )
    assert np.max(np.abs(fresh.values - np.asarray(checkpoint.fields[0]))) > 0.01
    np.testing.assert_array_equal(source_values, points.sum(axis=1))
