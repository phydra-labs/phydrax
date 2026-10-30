from __future__ import annotations

import phydrax as phx
import numpy as np

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.brep._constructors import brep_box
from phydrax.geometry.implicit import AnalyticImplicitProfile
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._scope import MeshingScope
from phydrax.meshing._sizing import SizeControlStrength, UniformSizeControl
from phydrax.meshing.providers import NativeMeshingProvider

from .._runtime import accepted_result_record, require_meshcore, runtime_identity


def native_brep_workflow() -> dict[str, object]:
    meshcore = require_meshcore()
    contract = SpatialCoordinateContract.si()
    model = brep_box(
        np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((2.0, 1.0, 1.0), dtype=np.float64),
        coordinate_contract=contract,
        source_id="qualification:native-profile-extrusion-box",
    )
    boundary = MeshingScope.whole(model, 2)
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        boundary,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(UniformSizeControl(boundary, 0.35),),
        maximum_deviation=0.02,
    )
    provider = NativeMeshingProvider()
    plan = provider.plan(model, specification, coordinate_contract=contract)
    result = plan.execute()
    accepted = accepted_result_record(result)
    measured_volume = float(np.sum(np.asarray(result.quality.evaluation.measures)))
    exact_volume = 2.0
    relative_error = abs(measured_volume - exact_volume) / exact_volume
    if relative_error > 1.0e-10:
        raise RuntimeError(
            f"Native profile-extrusion volume {measured_volume} differs from exact volume {exact_volume}."
        )
    if not result.region_boundary_evidence:
        raise RuntimeError("Native BRep volume result omitted source-region boundary evidence.")
    return {
        "status": "passed",
        "workflow": "native planar profile -> extrusion B-Rep -> tetrahedral volume mesh",
        "route": plan.route,
        "plan_id": plan.plan_id,
        "source_id": model.source_id,
        "source_revision": model.source_revision,
        "source_geometry_id": model.geometry.geometry_id,
        "source_faces": model.topology.num_faces,
        "source_solids": model.topology.num_solids,
        "measured_volume": measured_volume,
        "exact_volume": exact_volume,
        "relative_volume_error": relative_error,
        "source_region_boundary_count": len(result.region_boundary_evidence),
        "accepted": accepted,
        "runtime": runtime_identity(),
        "meshcore_identity": meshcore,
    }


def native_volume() -> dict[str, object]:
    meshcore = require_meshcore()
    contract = SpatialCoordinateContract.si()
    compiled = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 0.75, feature_id="qualification:native-volume-sphere"
    ).compile()
    source = AnalyticImplicitProfile(
        compiled, contract, source_id="qualification:native-volume-sphere"
    )
    boundary = MeshingScope.whole(source, 2)
    maximum_deviation = 0.15
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        boundary,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(
            UniformSizeControl(
                boundary, 0.35, strength=SizeControlStrength.SOFT
            ),
        ),
        maximum_deviation=maximum_deviation,
    )
    plan = NativeMeshingProvider().plan(
        source, specification, coordinate_contract=contract
    )
    result = plan.execute()
    accepted = accepted_result_record(result)
    measured_volume = float(np.sum(np.asarray(result.quality.evaluation.measures)))
    exact_volume = (4.0 / 3.0) * np.pi * 0.75**3
    relative_error = abs(measured_volume - exact_volume) / exact_volume
    if relative_error > 0.12:
        raise RuntimeError(
            f"Native implicit volume differs from the analytic sphere volume: {relative_error:.6g}."
        )
    return {
        "status": "passed",
        "route": plan.route,
        "source_id": source.source_id,
        "source_revision": source.source_revision,
        "measured_volume": measured_volume,
        "analytic_volume": exact_volume,
        "relative_volume_error": relative_error,
        "maximum_deviation_request": maximum_deviation,
        "accepted": accepted,
        "runtime": runtime_identity(),
        "meshcore_identity": meshcore,
    }
