from __future__ import annotations

import numpy as np
import pytest

from phydrax._execution_resources import MemoryBudgetError
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
