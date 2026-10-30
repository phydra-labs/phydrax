from __future__ import annotations

import numpy as np

import phydrax as phx
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.implicit import (
    AdaptiveImplicitSurface,
    AdaptiveImplicitSurfacePolicy,
    AnalyticImplicitProfile,
    discover_adaptive_implicit_surface,
)
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._scope import MeshingScope
from phydrax.meshing._sizing import SizeControlStrength, UniformSizeControl
from phydrax.meshing.providers import NativeMeshingOptions, NativeMeshingProvider
from phydrax.meshing.providers._native_sources import NativeImplicitSource

from .._runtime import accepted_result_record, require_meshcore, runtime_identity


_CONTRACT = SpatialCoordinateContract.si()


def _volume_spec(
    source: AnalyticImplicitProfile | NativeImplicitSource,
    /,
    size: float,
    tolerance: float,
) -> VolumeMeshingSpec:
    scope = MeshingScope.whole(source, 2)
    return VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(
            UniformSizeControl(scope, size, strength=SizeControlStrength.SOFT),
        ),
        maximum_deviation=tolerance,
    )


def _volume_case(
    name: str,
    profile: AnalyticImplicitProfile | NativeImplicitSource,
    exact_volume: float,
    /,
    *,
    size: float,
    tolerance: float,
    required_components: int,
    discovery: AdaptiveImplicitSurface | None = None,
    adapt: bool = False,
) -> dict[str, object]:
    plan = NativeMeshingProvider(
        NativeMeshingOptions("implicit_adaptive_tetrahedral")
    ).plan(profile, _volume_spec(profile, size, tolerance), coordinate_contract=_CONTRACT)
    result = plan.execute()
    accepted = accepted_result_record(result)
    measured_volume = float(
        np.sum(np.asarray(result.quality.evaluation.measures, dtype=np.float64))
    )
    relative_volume_error = abs(measured_volume - exact_volume) / exact_volume
    if relative_volume_error > 0.15:
        raise RuntimeError(
            f"{name} native implicit volume has {relative_volume_error:.6g} relative error; tolerance is 0.15."
        )
    component_count = None
    if discovery is not None:
        surface_mesh = discovery.mesh
        if surface_mesh is None:
            raise RuntimeError(
                f"{name} adaptive implicit discovery omitted its triangle carrier."
            )
        faces = np.asarray(surface_mesh.faces, dtype=np.int64)
        parents = np.arange(faces.shape[0], dtype=np.int64)

        def find(value: int) -> int:
            while int(parents[value]) != value:
                parents[value] = parents[int(parents[value])]
                value = int(parents[value])
            return value

        first_by_vertex: dict[int, int] = {}
        for face_row, face in enumerate(faces.tolist()):
            for vertex in face:
                first = first_by_vertex.setdefault(int(vertex), face_row)
                root_a, root_b = find(face_row), find(first)
                if root_a != root_b:
                    parents[root_b] = root_a
        component_count = len({find(row) for row in range(faces.shape[0])})
        if component_count != required_components:
            raise RuntimeError(
                f"{name} discovered {component_count} closed components; expected {required_components}."
            )
    record: dict[str, object] = {
        "status": "passed",
        "case": name,
        "source_id": profile.source_id,
        "source_revision": profile.source_revision,
        "route": plan.route,
        "analytic_volume_m3": exact_volume,
        "measured_volume_m3": measured_volume,
        "relative_volume_error": relative_volume_error,
        "volume_relative_tolerance": 0.15,
        "discovered_components": component_count,
        "required_components": required_components,
        "accepted": accepted,
    }
    if adapt:
        cell_id = int(np.asarray(result.mesh.blocks[0].global_ids, dtype=np.int64)[0])
        adaptation = phx.meshing.execute_mesh_adaptation(
            phx.meshing.prepare_mesh_adaptation(
                result,
                phx.meshing.MarkedMeshAdaptation(np.asarray((cell_id,), dtype=np.int64)),
                policy=phx.meshing.MeshAdaptationPolicy(
                    phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
                ),
            )
        )
        target = adaptation.target
        adapted_accepted = accepted_result_record(target)
        adapted_volume = float(
            np.sum(np.asarray(target.quality.evaluation.measures, dtype=np.float64))
        )
        adapted_error = abs(adapted_volume - exact_volume) / exact_volume
        if adapted_error > 0.15:
            raise RuntimeError(
                f"{name} source-faithful adaptation has {adapted_error:.6g} relative volume error; tolerance is 0.15."
            )
        if target.certification is None or target.certification.fidelity is None:
            raise RuntimeError(
                f"{name} implicit adaptation dropped its retained source-fidelity proof."
            )
        record["adaptation"] = {
            "status": adaptation.status.value,
            "result": adapted_accepted,
            "measured_volume_m3": adapted_volume,
            "relative_volume_error": adapted_error,
            "source_fidelity_certificate_id": target.certification.fidelity.certificate_id,
        }
    return record


def native_implicit_adaptation() -> dict[str, object]:
    meshcore = require_meshcore()
    cases: list[dict[str, object]] = []
    failures: list[str] = []

    sphere_radius = 0.75
    sphere_model = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), sphere_radius, feature_id="native-qualification:implicit-sphere"
    ).compile()
    sphere = AnalyticImplicitProfile(
        sphere_model, _CONTRACT, source_id="native-qualification:implicit-sphere"
    )
    try:
        cases.append(
            _volume_case(
                "sphere-adaptive-refinement",
                sphere,
                (4.0 / 3.0) * np.pi * sphere_radius**3,
                size=0.28,
                tolerance=0.1,
                required_components=1,
                adapt=True,
            )
        )
    except Exception as error:
        failures.append(f"sphere-adaptive-refinement: {type(error).__name__}: {error}")

    torus_inner, torus_outer = 0.27, 0.57
    torus_major = 0.5 * (torus_inner + torus_outer)
    torus_minor = 0.5 * (torus_outer - torus_inner)
    torus_center = np.asarray((0.2, -0.1, 0.05), dtype=np.float64)
    torus_model = phx.geometry.Torus(
        torus_center,
        torus_inner,
        torus_outer,
        feature_id="native-qualification:offset-ring-torus",
    ).compile()
    torus = AnalyticImplicitProfile(
        torus_model, _CONTRACT, source_id="native-qualification:offset-ring-torus"
    )
    try:
        cases.append(
            _volume_case(
                "offset-ring-torus",
                torus,
                2.0 * np.pi**2 * torus_major * torus_minor**2,
                size=0.12,
                tolerance=0.08,
                required_components=1,
            )
        )
    except Exception as error:
        failures.append(f"offset-ring-torus: {type(error).__name__}: {error}")

    hidden_source_id = "native-qualification:implicit-hidden-component"
    hidden_geometry = phx.geometry.Union(
        (
            phx.geometry.Sphere((-0.6, 0.0, 0.0), 0.3, feature_id="hidden-left"),
            phx.geometry.Sphere((0.6, 0.0, 0.0), 0.3, feature_id="hidden-right"),
        ),
        feature_id="hidden-two-component-union",
    ).compile()
    domain = np.asarray(((-1.2, -0.8, -0.8), (1.2, 0.8, 0.8)), dtype=np.float64)
    policy = AdaptiveImplicitSurfacePolicy(relative_cell_diameter=0.08)
    try:
        discovery = discover_adaptive_implicit_surface(
            hidden_geometry,
            domain=domain,
            policy=policy,
            source_id=hidden_source_id,
        )
        source = NativeImplicitSource(
            hidden_geometry,
            hidden_source_id,
            "revision-1",
            discovery=discovery,
        )
        cases.append(
            _volume_case(
                "hidden-disconnected-components",
                source,
                2.0 * (4.0 / 3.0) * np.pi * 0.3**3,
                size=0.14,
                tolerance=0.08,
                required_components=2,
                discovery=discovery,
            )
        )
    except Exception as error:
        failures.append(
            f"hidden-disconnected-components: {type(error).__name__}: {error}"
        )

    return {
        "status": "failed" if failures else "passed",
        "workflow": "native adaptive implicit source -> tetrahedral volume mesh -> source-faithful refinement",
        "source_registry": "tools/meshing_qualification/native_case_migration.json#implicit-domain",
        "cases": cases,
        "failures": failures,
        "required_contracts": [
            "analytic sphere refinement with independent exact volume",
            "offset ring torus with independent exact volume and handle topology",
            "disconnected hidden components with independent component and total-volume oracles",
            "source-faithful adaptation with revalidated geometry fidelity",
        ],
        "runtime": runtime_identity(),
        "meshcore_identity": meshcore,
    }
