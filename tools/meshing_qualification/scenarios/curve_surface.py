from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry._meshing_domain import MeshingDomain
from phydrax.geometry.brep._constructors import brep_sphere
from phydrax.geometry.simplicial import SegmentMesh
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    CurveMeshingSpec,
    SurfaceMeshingSpec,
)
from phydrax.meshing._scope import MeshingScope
from phydrax.meshing._sizing import SizeCompliancePolicy, UniformSizeControl
from phydrax.meshing.providers import NativeMeshingOptions, NativeMeshingProvider
from phydrax.meshing.providers._native_sources import NativeCurveSource

from .._runtime import accepted_result_record, require_meshcore, runtime_identity


def native_curve() -> dict[str, object]:
    meshcore = require_meshcore()
    contract = SpatialCoordinateContract.si()
    segments = SegmentMesh(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0)), dtype=jnp.float64),
        jnp.asarray(((0, 1),), dtype=jnp.int32),
        source_id="qualification:native-curve",
    )
    source = NativeCurveSource(
        segments,
        "revision-1",
        curve_entity_ids=np.asarray((1,), dtype=np.int64),
        vertex_entity_ids=np.asarray((1, 2), dtype=np.int64),
    )
    scope = MeshingScope.whole(source, 1)
    specification = CurveMeshingSpec(
        CellMeshingTarget(1, 2, CellFamilyPolicy(required=("interval",))),
        scope,
        size_controls=(UniformSizeControl(scope, 0.2),),
        # 0.2 m is not binary-representable, so a zero-tolerance hard target is
        # unsatisfiable by any floating-point partition; the request accepts a
        # 1 ppm size deviation while the length/straightness oracles stay at 1e-12.
        size_compliance=SizeCompliancePolicy(relative_tolerance=1.0e-6),
        maximum_deviation=1.0e-8,
    )
    plan = NativeMeshingProvider().plan(
        source, specification, coordinate_contract=contract
    )
    result = plan.execute()
    accepted = accepted_result_record(result)
    cells = np.asarray(result.mesh.blocks[0].vertices, dtype=np.int64)
    points = np.asarray(result.mesh.coordinates, dtype=np.float64)
    edge_lengths = np.linalg.norm(points[cells[:, 1]] - points[cells[:, 0]], axis=1)
    length = float(np.sum(edge_lengths))
    if not np.isclose(length, 1.0, rtol=0.0, atol=1.0e-12):
        raise RuntimeError(
            f"Native curve length {length} differs from the exact unit segment."
        )
    if np.any(np.abs(points[:, 1]) > 1.0e-12):
        raise RuntimeError("Native curve departed from its exact source line.")
    if not result.associations:
        raise RuntimeError("Native curve result omitted source associations.")
    return {
        "status": "passed",
        "route": plan.route,
        "plan_id": plan.plan_id,
        "length": length,
        "exact_length": 1.0,
        "maximum_line_residual": float(np.max(np.abs(points[:, 1]))),
        "association_count": len(result.associations),
        "accepted": accepted,
        "runtime": runtime_identity(),
        "meshcore_identity": meshcore,
    }


def native_surface() -> dict[str, object]:
    meshcore = require_meshcore()
    contract = SpatialCoordinateContract.si()
    model = brep_sphere(
        1.0, coordinate_contract=contract, source_id="qualification:native-sphere"
    )
    domain = MeshingDomain.from_brep(model)
    scope = MeshingScope.whole(domain, 2)
    deviation_tolerance = 0.05
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(UniformSizeControl(scope, 0.35),),
        maximum_deviation=deviation_tolerance,
    )
    plan = NativeMeshingProvider(NativeMeshingOptions("parametric_surface")).plan(
        domain, specification, coordinate_contract=contract
    )
    result = plan.execute()
    accepted = accepted_result_record(result)
    source = result.surface_source
    certification = result.certification
    if source is None or certification is None or certification.fidelity is None:
        raise RuntimeError(
            "Native BRep surface result omitted its source-chart fidelity proof."
        )
    witness = certification.inputs.fidelity_witness
    if witness is None or witness.witness_id != source.chart_witness.witness_id:
        raise RuntimeError(
            "Surface fidelity certification is not bound to retained chart ancestry."
        )
    points = np.asarray(result.mesh.coordinates, dtype=np.float64)
    radial_error = float(np.max(np.abs(np.linalg.norm(points, axis=1) - 1.0)))
    if radial_error > deviation_tolerance:
        raise RuntimeError(
            "Native BRep sphere vertices exceed the exact radial deviation request."
        )
    approximate_area = float(np.sum(np.asarray(result.quality.evaluation.measures)))
    if not np.isfinite(approximate_area) or approximate_area <= 0.0:
        raise RuntimeError("Native sphere surface has invalid measured area.")
    return {
        "status": "passed",
        "route": plan.route,
        "source_id": model.source_id,
        "source_revision": model.source_revision,
        "source_chart_witness_id": source.chart_witness.witness_id,
        "source_fidelity_certificate_id": certification.fidelity.certificate_id,
        "radial_deviation_tolerance": deviation_tolerance,
        "maximum_radial_residual": radial_error,
        "approximate_surface_area": approximate_area,
        "analytic_surface_area": 4.0 * np.pi,
        "accepted": accepted,
        "runtime": runtime_identity(),
        "meshcore_identity": meshcore,
    }
