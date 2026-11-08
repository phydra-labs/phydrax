"""Installed-wheel native mesh execution and exact CAD archive contracts."""

from pathlib import Path
from typing import assert_type

from phydrax.execution import ExecutionGroup
from phydrax.geometry.brep import BRepModel
from phydrax.interchange import load_brep_archive, save_brep_archive
from phydrax.meshing import (
    CellMeshingResult,
    MeshAssembly,
    NativeVolumeSchedule,
    PLC_SEARCH_BUDGET,
    PlcSearchBudget,
    SizeControlStrength,
    VolumeMeshingSpec,
)
from phydrax.meshing.providers import (
    NativeMeshingOptions,
    NativeMeshingPlan,
    NativeMeshingProvider,
    NativePlcSource,
)


def native_volume(
    source: NativePlcSource, specification: VolumeMeshingSpec, group: ExecutionGroup
) -> None:
    plan = NativeMeshingProvider().plan(source, specification)
    assert_type(plan, NativeMeshingPlan)
    assert_type(plan.execute(), CellMeshingResult)
    assert_type(plan.execute(execution_group=group), MeshAssembly)


def fixed_boundary_volume(
    source: NativePlcSource, specification: VolumeMeshingSpec
) -> None:
    schedule = NativeVolumeSchedule(radius_edge_bound=None, minimum_dihedral_degrees=0.0)
    assert_type(schedule.radius_edge_bound, float | None)
    options = NativeMeshingOptions("plc_tetrahedral", volume_schedule=schedule)
    assert_type(
        NativeMeshingProvider(options).plan(source, specification), NativeMeshingPlan
    )


def soft_shape_volume_schedule() -> None:
    schedule = NativeVolumeSchedule(shape_strength=SizeControlStrength.SOFT)
    assert_type(schedule.shape_strength, SizeControlStrength)


def bounded_native_search() -> None:
    budget = PlcSearchBudget(nodes=1 << 20, bytes=512 << 20)
    schedule = NativeVolumeSchedule(plc_search_budget=budget)
    assert_type(schedule.plc_search_budget, PlcSearchBudget)
    assert_type(PLC_SEARCH_BUDGET, PlcSearchBudget)


def exact_archive(model: BRepModel, path: Path) -> None:
    written = save_brep_archive(model, path)
    assert_type(written, Path)
    restored = load_brep_archive(written)
    assert_type(restored, BRepModel)
