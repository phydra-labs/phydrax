# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

import phydrax as phx
from tools.meshing_qualification._runtime import _contract, qualification_failure


def _layout_bytes() -> bytes:
    # The optional dependency is admitted when this workflow executes.
    import gdstk  # ty: ignore[unresolved-import]

    with TemporaryDirectory(prefix="phydrax-layout-fixture-") as temporary:
        path = Path(temporary) / "qualification.gds"
        library = gdstk.Library(unit=1.0e-3, precision=1.0e-9)
        top = library.new_cell("TOP")
        top.add(gdstk.rectangle((0.0, 0.0), (2.0, 1.0), layer=1, datatype=0))
        library.write_gds(path)
        return path.read_bytes()


def _decode_layout() -> Any:

    contract = _contract()
    limits = phx.interchange.ResourceLimits(
        max_bytes=1_000_000,
        max_depth=4,
        max_nodes=1_000,
        max_attributes=1_000,
        max_losses=0,
    )
    policy = phx.interchange.LayoutImportPolicy(
        phx.interchange.LayoutFormat.GDSII,
        contract,
        limits,
        top_cell="TOP",
    )
    return phx.interchange.decode_layout_bytes(
        _layout_bytes(), policy, source_path="qualification.gds"
    )


def _hybrid_face_indices(source: Any, axis: int, coordinate: float) -> Any:
    points = np.asarray(source.mesh_vertices)
    triangles = points[np.asarray(source.mesh_faces)]
    face_ids = np.asarray(source.triangle_face_ids)
    return tuple(
        np.unique(
            face_ids[
                np.all(
                    np.isclose(triangles[:, :, axis], coordinate),
                    axis=1,
                )
            ]
        )
    )


def _hybrid_gmsh_result(destination: Path, *, schedule: Any = None) -> Any:

    # Optional provider imports are admitted only for this fixture workflow.
    from OCP.BOPAlgo import BOPAlgo_Splitter
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    lower = BRepPrimAPI_MakeBox(gp_Pnt(0.0, 0.0, 0.0), 1.0, 1.0, 0.3).Shape()
    upper = BRepPrimAPI_MakeBox(gp_Pnt(0.0, 0.0, 0.3), 1.0, 1.0, 0.7).Shape()
    splitter = BOPAlgo_Splitter()
    splitter.AddArgument(lower)
    splitter.AddArgument(upper)
    splitter.Perform()
    if splitter.HasErrors():
        raise qualification_failure(
            "_hybrid_gmsh_result",
            ("OCCT failed to build the hybrid qualification fixture."),
            {"splitter": splitter},
        )
    source = phx.interchange.persist_occt_shape(
        splitter.Shape(),
        destination,
        coordinate_contract=_contract(),
        linear_deflection=0.03,
        angular_deflection=0.15,
    )
    source_face = _hybrid_face_indices(source, 2, 0.0)
    target_face = _hybrid_face_indices(source, 2, 0.3)
    if len(source_face) != 1 or len(target_face) != 1:
        raise qualification_failure(
            "_hybrid_gmsh_result",
            ("Hybrid qualification could not resolve its CAD caps."),
            {"source_face": source_face, "target_face": target_face},
        )
    swept = source.topology.face_solids[source_face[0]][0]
    core = next(index for index in range(source.topology.num_solids) if index != swept)
    provider = phx.meshing.GmshProvider()
    swept_scope = provider.entity_scope(source, source.solid_ids[swept])
    core_scope = provider.entity_scope(source, source.solid_ids[core])
    layer_schedule = (
        phx.meshing.LayerSchedule(np.asarray((0.08, 0.10, 0.12), dtype=np.float64))
        if schedule is None
        else schedule
    )
    if not isinstance(layer_schedule, phx.meshing.LayerSchedule):
        raise TypeError(
            "layout.schedule must be LayerSchedule or None; "
            f"got {type(schedule).__name__}"
        )
    control = phx.meshing.BoundaryLayerControl(
        provider.entity_scope(source, source.face_ids[source_face[0]]),
        layer_schedule,
        route=phx.meshing.BoundaryLayerRoute.EXACT_SWEEP,
        volume_scope=swept_scope,
        cap_scope=provider.entity_scope(source, source.face_ids[target_face[0]]),
    )
    whole = provider.whole_scope(source, 3)
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(
                required=("prism", "tetrahedron"),
                allow_mixed=True,
            ),
        ),
        whole,
        phx.meshing.VolumeFillStrategy.SWEEP,
        size_controls=(phx.meshing.UniformSizeControl(whole, 0.24),),
        region_controls=(
            phx.meshing.RegionControl(
                swept_scope,
                "swept-slab",
                "swept-material",
                phx.meshing.RegionRole.SOLID,
            ),
            phx.meshing.RegionControl(
                core_scope,
                "tetra-core",
                "core-material",
                phx.meshing.RegionRole.SOLID,
            ),
        ),
        patch_controls=(
            phx.meshing.PatchControl(
                "slab-core",
                provider.entity_scope(source, source.face_ids[target_face[0]]),
                ("swept-slab", "tetra-core"),
            ),
        ),
        layer_controls=(control,),
    )
    return provider.plan(source, specification).execute(), control
