from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from phydrax import SpatialCoordinateContract
from phydrax.geometry import prepare_brep_projection
from phydrax.geometry.brep import (
    brep_extrusion,
    BRepModel,
    PlanarProfile,
    ProfileLoop,
    ProfilePlane,
)
from phydrax.interchange import CadImportPolicy, read_step, write_step
from phydrax.meshing import (
    CellFamilyPolicy,
    CellMeshingTarget,
    MeshingScope,
    NativeMeshingOptions,
    NativeMeshingPlan,
    NativeMeshingProvider,
    UniformSizeControl,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing.providers import NativeSurfaceSchedule


DEFAULT_VOLUME_SIZE = 0.35
DEFAULT_MAXIMUM_DEVIATION = 0.02
DEFAULT_SURFACE_SPACING_FRACTION = 0.9


def prepare_workflow(
    *,
    size: float = DEFAULT_VOLUME_SIZE,
    maximum_deviation: float = DEFAULT_MAXIMUM_DEVIATION,
    surface_spacing_fraction: float = DEFAULT_SURFACE_SPACING_FRACTION,
) -> tuple[BRepModel, VolumeMeshingSpec, NativeMeshingPlan]:
    """Prepare the authored source, volume request, and native plan."""
    contract = SpatialCoordinateContract.si()
    profile = PlanarProfile(
        ProfilePlane((0.0, 0.0, 0.0)),
        ProfileLoop.polygon(((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0))),
    )
    model = brep_extrusion(
        profile,
        (0.0, 0.0, 1.0),
        coordinate_contract=contract,
        source_id="example:native-brep-profile-extrusion",
    )
    boundary = MeshingScope.whole(model, 2)
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        boundary,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(UniformSizeControl(boundary, size),),
        maximum_deviation=maximum_deviation,
    )
    options = (
        None
        if surface_spacing_fraction == DEFAULT_SURFACE_SPACING_FRACTION
        else NativeMeshingOptions(
            surface_schedule=NativeSurfaceSchedule(
                spacing_fraction=surface_spacing_fraction
            )
        )
    )
    plan = NativeMeshingProvider(options).plan(
        model, specification, coordinate_contract=contract
    )
    return model, specification, plan


def run_workflow() -> dict[str, object]:
    """Construct a source-defined profile extrusion, mesh it, and audit the solid."""
    model, _specification, plan = prepare_workflow()
    if model.geometry is None:
        raise RuntimeError("The constructed BRep has no native topology.")
    projection = prepare_brep_projection(model)
    vertex = np.asarray(model.geometry.vertex_points, dtype=np.float64)[0]
    query = projection.project(
        vertex[None, :],
        np.asarray((0,), dtype=np.int64),
        np.asarray((0,), dtype=np.int64),
    )
    if int(query.status[0]) != 0 or float(query.residuals[0]) > 1.0e-12:
        raise RuntimeError("Native BRep vertex query did not resolve exactly.")
    with TemporaryDirectory(prefix="phydrax-native-brep-") as directory:
        path = Path(directory) / "profile.step"
        exported = write_step(model, path)
        imported = read_step(
            path, CadImportPolicy(SpatialCoordinateContract.si()), trusted_root=directory
        )
        if not exported.report.valid or not imported.report.valid:
            raise RuntimeError("Native STEP exchange did not validate.")
        if imported.model.topology.num_solids != model.topology.num_solids:
            raise RuntimeError("Native STEP exchange lost the source solid.")
    result = plan.execute()
    result.audit.require_passed()
    certification = result.certification
    if certification is None:
        raise RuntimeError("Accepted native BRep mesh omitted computed certification.")
    certification.require_passed()
    if not result.compliance.passed:
        raise RuntimeError(
            "Native BRep mesh did not meet its requested family/fidelity controls."
        )
    if not result.region_boundary_evidence:
        raise RuntimeError("Native BRep mesh omitted source-region boundary evidence.")

    measured_volume = float(np.sum(np.asarray(result.quality.evaluation.measures)))
    exact_volume = 2.0
    relative_volume_error = abs(measured_volume - exact_volume) / exact_volume
    if relative_volume_error > 1.0e-10:
        raise RuntimeError(
            f"Native mesh volume {measured_volume} differs from exact profile-extrusion volume {exact_volume}."
        )
    return {
        "status": "passed",
        "workflow": "native planar profile extrusion -> tetrahedral volume meshing",
        "source_id": model.source_id,
        "source_revision": model.source_revision,
        "source_geometry_id": model.geometry.geometry_id,
        "source_solids": model.topology.num_solids,
        "query_status": int(query.status[0]),
        "query_residual": float(query.residuals[0]),
        "step_export_valid": exported.report.valid,
        "step_import_valid": imported.report.valid,
        "source_faces": model.topology.num_faces,
        "plan_id": plan.plan_id,
        "result_id": result.result_id,
        "topology_id": result.mesh.topology_id,
        "provider": result.provider.name,
        "actual_backend_version": result.runtime.actual_version,
        "cells": result.mesh.entity_set(3).count,
        "measured_volume": measured_volume,
        "exact_volume": exact_volume,
        "relative_volume_error": relative_volume_error,
        "source_region_boundary_count": len(result.region_boundary_evidence),
        "audit_passed": result.audit.passed,
        "certification_id": certification.report_id,
        "certification_passed": certification.passed,
        "compliance_passed": result.compliance.passed,
    }


def main() -> None:
    print(json.dumps(run_workflow(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
