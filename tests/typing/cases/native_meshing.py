"""Native source preparation, execution, and CAD persistence consumer types."""

from pathlib import Path
from typing import assert_type

import phydrax as phx
from phydrax.meshing.providers import (
    NativeImplicitSource,
    NativeMeshingOptions,
    NativeMeshingPlan,
    NativeMeshingProvider,
    NativePlcSource,
)


def native_volume(
    source: NativePlcSource,
    specification: phx.meshing.VolumeMeshingSpec,
    group: phx.execution.ExecutionGroup,
) -> None:
    plan = NativeMeshingProvider().plan(source, specification)
    assert_type(plan, NativeMeshingPlan)
    assert_type(plan.execute(), phx.meshing.CellMeshingResult)
    assert_type(plan.execute(execution_group=group), phx.meshing.MeshAssembly)


def fixed_surface(
    source: NativeImplicitSource,
    specification: phx.meshing.SurfaceMeshingSpec,
    prior: NativeMeshingPlan,
) -> None:
    plan = NativeMeshingProvider().plan(source, specification, fixed_plan=prior)
    assert_type(plan, NativeMeshingPlan)
    assert_type(plan.execute(), phx.meshing.CellMeshingResult)


def fixed_boundary_volume(
    source: NativePlcSource, specification: phx.meshing.VolumeMeshingSpec
) -> None:
    schedule = phx.meshing.NativeVolumeSchedule(
        radius_edge_bound=None, minimum_dihedral_degrees=0.0
    )
    assert_type(schedule.radius_edge_bound, float | None)
    options = NativeMeshingOptions("plc_tetrahedral", volume_schedule=schedule)
    assert_type(
        NativeMeshingProvider(options).plan(source, specification), NativeMeshingPlan
    )


def soft_shape_volume_schedule() -> None:
    schedule = phx.meshing.NativeVolumeSchedule(
        shape_strength=phx.meshing.SizeControlStrength.SOFT
    )
    assert_type(schedule.shape_strength, phx.meshing.SizeControlStrength)
    phx.meshing.NativeVolumeSchedule(shape_strength="soft")  # ty: ignore[invalid-argument-type]


def bounded_native_search() -> None:
    budget = phx.meshing.PlcSearchBudget(nodes=1 << 20, bytes=512 << 20)
    schedule = phx.meshing.NativeVolumeSchedule(plc_search_budget=budget)
    assert_type(schedule.plc_search_budget, phx.meshing.PlcSearchBudget)
    assert_type(phx.meshing.PLC_SEARCH_BUDGET, phx.meshing.PlcSearchBudget)


def native_cad_archive(model: phx.geometry.brep.BRepModel, path: Path) -> None:
    assert_type(phx.interchange.save_brep_archive(model, path), Path)
    assert_type(phx.interchange.load_brep_archive(path), phx.geometry.brep.BRepModel)
