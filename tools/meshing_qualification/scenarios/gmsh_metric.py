# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypedDict

import numpy as np

import phydrax as phx
from tools.meshing_qualification._controls import QualificationControls
from tools.meshing_qualification._records import _quality_summary, _result_identity
from tools.meshing_qualification._runtime import (
    _contract,
    _measured,
    _missing_dependency,
    _runtime_identity,
    qualification_failure,
)
from tools.meshing_qualification.scenarios.metric_adaptation import (
    _layer_metric,
    _layer_metric_tensors,
    _mesh_triangles,
    _metric_lattice,
    _triangle_areas,
)


class _GmshMetricRunRecord(TypedDict):
    mode: str
    control_id: str
    plan_id: str
    algorithm_2d: str
    compliance_metric_edge_lengths: dict[str, float]
    analytic_metric_edge_lengths: dict[str, float]
    mean_edge_extent_y_over_x: float
    area: float
    resources: dict[str, dict[str, float]]


def _gmsh_metric_run(
    square: Any, control: phx.meshing.BackgroundMetricControl, options: Any, /
) -> tuple[phx.meshing.CellMeshingResult, _GmshMetricRunRecord]:

    provider = phx.meshing.GmshProvider(options)
    scope = provider.whole_scope(square, 2)
    specification = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 3, phx.meshing.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
        size_controls=(phx.meshing.UniformSizeControl(scope, 0.5),),
        background_metric=control,
    )
    plan, plan_record = _measured(lambda: provider.plan(square, specification))

    result, execute_record = _measured(plan.execute)

    if not result.audit.passed or not result.compliance.passed:
        raise qualification_failure(
            "_gmsh_metric_run",
            (f"Gmsh {control.mode.value} metric mesh failed its audit."),
            {"result": result},
        )
    prefix = f"background_metric:{control.control_id}:"
    compliance = {
        key.removeprefix(prefix): value
        for key, value in result.compliance.achieved
        if key.startswith(prefix)
    }
    if sorted(compliance) != [
        "maximum_edge_length",
        "mean_edge_length",
        "minimum_edge_length",
        "unit_edge_fraction",
    ]:
        raise qualification_failure(
            "_gmsh_metric_run",
            ("Gmsh compliance lacks the background metric edge lengths."),
            {"compliance": compliance},
        )

    points = np.asarray(result.mesh.coordinates, dtype=np.float64)

    connectivity = result.mesh.connectivity
    if not isinstance(connectivity, phx.discretization.PolygonalConnectivity):
        raise TypeError(
            f"_gmsh_metric_run.connectivity must be PolygonalConnectivity for triangle metric qualification; got {type(connectivity).__name__!r}."
        )
    edges = np.asarray(connectivity.edges)
    vectors = points[edges[:, 1]] - points[edges[:, 0]]
    midpoints = 0.5 * (points[edges[:, 0]] + points[edges[:, 1]])
    diagonal = np.diagonal(_layer_metric_tensors(midpoints), axis1=1, axis2=2)
    analytic = np.sqrt(np.sum(vectors**2 * diagonal, axis=1))

    area = float(np.sum(_triangle_areas(points, _mesh_triangles(result.mesh))))
    extent = np.mean(np.abs(vectors), axis=0)
    return result, {
        "mode": control.mode.value,
        "control_id": control.control_id,
        "plan_id": plan.plan_id,
        "algorithm_2d": options.algorithm_2d.value,
        "compliance_metric_edge_lengths": compliance,
        "analytic_metric_edge_lengths": {
            "minimum": float(np.min(analytic)),
            "mean": float(np.mean(analytic)),
            "maximum": float(np.max(analytic)),
            "unit_fraction": float(
                np.mean((analytic >= 1.0 / np.sqrt(2.0)) & (analytic <= np.sqrt(2.0)))
            ),
        },
        "mean_edge_extent_y_over_x": float(extent[1] / extent[0]),
        "area": area,
        "resources": {"plan": plan_record, "execute": execute_record},
    }


def qualify_gmsh_metric() -> dict[str, object]:

    # Optional provider imports are admitted only for this fixture workflow.
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
    from OCP.gp import gp_Pnt

    if find_spec("gmsh") is None:
        return _missing_dependency(
            "gmsh",
            "Install the optional Gmsh Python package: "
            "pip install 'phydrax[meshing-gmsh]'.",
        )

    contract = _contract()
    grid, triangles = _metric_lattice(8, -0.01, 1.01)
    background = phx.discretization.CellMesh(
        np.concatenate((grid, np.zeros((grid.shape[0], 1), dtype=np.float64)), axis=1),
        (phx.discretization.CellBlock("triangles", "triangle", triangles),),
    )
    metric = _layer_metric(background)
    certified_background, background_record = _measured(
        lambda: phx.meshing.certify_cell_mesh(background, contract)
    )
    controls = {
        mode: phx.meshing.BackgroundMetricControl(
            background, metric, contract, mode=mode, maximum_metric_edge_length=2.0
        )
        for mode in phx.meshing.BackgroundMetricMode
    }
    polygon = BRepBuilderAPI_MakePolygon()
    for x, y in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)):
        polygon.Add(gp_Pnt(x, y, 0.0))
    polygon.Close()
    face = BRepBuilderAPI_MakeFace(polygon.Wire()).Face()
    with TemporaryDirectory(prefix="phydrax-gmsh-metric-") as temporary:
        square = phx.interchange.persist_occt_shape(
            face,
            Path(temporary) / "square.brep",
            coordinate_contract=contract,
            linear_deflection=0.01,
            angular_deflection=0.2,
        )
        isotropic, isotropic_record = _gmsh_metric_run(
            square,
            controls[phx.meshing.BackgroundMetricMode.ISOTROPIC],
            phx.meshing.GmshOptions(),
        )
        anisotropic, anisotropic_record = _gmsh_metric_run(
            square,
            controls[phx.meshing.BackgroundMetricMode.ANISOTROPIC],
            phx.meshing.GmshOptions(algorithm_2d=phx.meshing.GmshSurfaceAlgorithm.BAMG),
        )
    results = {"isotropic": isotropic, "anisotropic": anisotropic}
    records = {"isotropic": isotropic_record, "anisotropic": anisotropic_record}
    edge_record = anisotropic_record["compliance_metric_edge_lengths"]
    unit = edge_record["unit_edge_fraction"]
    checks = {
        "background certified": certified_background.audit.passed,
        "isotropic mode refines every direction": (
            isotropic.mesh.entity_set(2).count > anisotropic.mesh.entity_set(2).count
        ),
        "anisotropic cells stretch along the layer": (
            anisotropic_record["mean_edge_extent_y_over_x"] > 1.5
        ),
        "anisotropic mesh is near unit": unit >= 0.8,
    }
    for name, record in records.items():
        checks[f"{name} mesh covers the face"] = abs(record["area"] - 1.0) <= 1.0e-12
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise qualification_failure(
            "qualify_gmsh_metric",
            (f"Gmsh metric qualification failed: {failed}."),
            {"failed": failed},
        )
    return {
        "status": "passed",
        "checks": sorted(checks),
        "source": {
            "brep_source_revision": square.source_revision,
            "background": _result_identity(certified_background),
            "metric_id": metric.metric_id,
        },
        "target": {name: _result_identity(result) for name, result in results.items()},
        "runtime": {
            **_runtime_identity(),
            "provider": anisotropic.provider.name,
            "provider_version": anisotropic.runtime.actual_version,
        },
        "quality": {
            "before": _quality_summary(certified_background.quality),
            "after": {
                name: _quality_summary(result.quality) for name, result in results.items()
            },
        },
        "conservation": {
            name: {
                "area": record["area"],
                "area_residual": abs(record["area"] - 1.0),
            }
            for name, record in records.items()
        },
        "routes": {
            name: {key: value for key, value in record.items() if key != "resources"}
            for name, record in records.items()
        },
        "resources": {
            "background_certify": background_record,
            **{name: record["resources"] for name, record in records.items()},
        },
    }


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return qualify_gmsh_metric()
