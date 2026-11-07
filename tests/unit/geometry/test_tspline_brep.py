import dataclasses
import json
import os
import platform
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._interpolation import TMesh2D
from phydrax._model._structure import _restore_dataclass
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry._meshing_domain import (
    MeshingDomain,
    MeshingDomainCurve,
    MeshingDomainRegion,
    MeshingSurfacePatch,
    PatchCurveUse,
)
from phydrax.geometry._source_boundary_records import SourceProjectionStatus
from phydrax.geometry.brep._constructors import (
    assemble_brep_model,
    brep_planar_face,
    PlanarProfile,
    ProfileLoop,
    ProfilePlane,
)
from phydrax.geometry.brep._intersection import intersect_surface_regions
from phydrax.geometry.brep._intersection_curve import IntersectionCurve, SurfaceRegion
from phydrax.geometry.brep._model import BRepModel
from phydrax.geometry.brep._patches import LineCurve, PlanePatch
from phydrax.geometry.brep._projection import prepare_brep_projection
from phydrax.geometry.brep._projection_contracts import brep_entity_id
from phydrax.geometry.brep._tspline import TSplineSurfacePatch
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    SurfaceMeshingSpec,
)
from phydrax.meshing._result import CellMeshingResult
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._sizing import SizeControlStrength, UniformSizeControl
from phydrax.meshing.providers import NativeMeshingProvider
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.meshing.providers._native_schedules import NativeSurfaceSchedule


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
# OpenBLAS kernels without the native micro-architecture's blocking and FMA order.
_GENERIC_OPENBLAS_CORES = {
    "aarch64": "ARMV8",
    "arm64": "ARMV8",
    "x86_64": "PRESCOTT",
    "AMD64": "PRESCOTT",
}
_IDENTITY_SCRIPT = """
import json

from tests.unit.geometry.test_tspline_brep import _tspline_surface_model

model = _tspline_surface_model()
patch = model.patches[0]
print(
    json.dumps(
        {
            "model_id": model.model_id,
            "source_revision": model.source_revision,
            "basis_certificate_id": patch.realization.basis_certificate.certificate_id,
            "source_basis_certificate_id": patch.source_basis_certificate.certificate_id,
            "realization_id": patch.realization.realization_id,
        }
    )
)
"""

_CONTRACT = SpatialCoordinateContract.si()


def _tspline_surface_model() -> BRepModel:
    base = brep_planar_face(
        PlanarProfile(
            ProfilePlane(),
            ProfileLoop.polygon(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        ),
        coordinate_contract=_CONTRACT,
    )
    mesh = TMesh2D.tensor_product(
        tuple(Fraction(index, 4) for index in range(5)),
        tuple(Fraction(index, 4) for index in range(5)),
    )

    def bump_blossom(knots: tuple[Fraction, ...]) -> Fraction:
        # Cubic blossom of t(1 - t) at the interior local knots; these controls
        # reproduce the quadratic exactly.
        first, second, third = knots[1:4]
        return (first + second + third) / 3 - (
            first * second + first * third + second * third
        ) / 3

    exact_controls = tuple(
        (
            anchor.local_knots.greville_parameter[0],
            anchor.local_knots.greville_parameter[1],
            bump_blossom(anchor.local_knots.u) * bump_blossom(anchor.local_knots.v) / 4,
        )
        for anchor in mesh.anchors
    )
    patch = TSplineSurfacePatch(
        mesh,
        np.asarray([[float(value) for value in point] for point in exact_controls]),
        np.ones((mesh.coefficient_count,), dtype=np.float64),
        exact_control_points=exact_controls,
        exact_weights=tuple(Fraction(1) for _ in exact_controls),
    )
    return assemble_brep_model(
        base.geometry,
        (patch,),
        np.asarray(base.parameter_bounds),
        np.asarray(base.orientation),
        base.physical_tags,
        coordinate_contract=_CONTRACT,
        source_id="native-tspline-brep-query",
        trim_domains=base.trim_domains,
    )


def _surface_specification(domain: MeshingDomain) -> SurfaceMeshingSpec:
    scope = MeshingScope(
        domain.source_id,
        domain.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        domain.scope_entity_set_id(2),
        domain.scope_entity_ids(2),
    )
    return SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(
            UniformSizeControl(scope, 0.4, strength=SizeControlStrength.HARD),
        ),
        maximum_deviation=0.02,
    )


def _certified_surface_mesh(domain: MeshingDomain) -> CellMeshingResult:
    """Mesh through the public native plan, which certifies the candidate."""
    return (
        NativeMeshingProvider(
            NativeMeshingOptions(
                "parametric_surface",
                surface_schedule=NativeSurfaceSchedule(spacing_fraction=0.45),
            )
        )
        .plan(domain, _surface_specification(domain), coordinate_contract=_CONTRACT)
        .execute()
    )


def test_t_spline_projection_and_native_meshing_retain_source_geometry() -> None:
    model = _tspline_surface_model()
    projection = prepare_brep_projection(model)
    entity_id = brep_entity_id(model.source_id, model.source_revision, 2, 0)
    projected = projection.project_boundary(
        np.asarray(((0.5, 0.5, 0.25),), dtype=np.float64),
        entity_ids=(entity_id,),
    )
    expected = (0.5, 0.5, 1.0 / 64.0)

    assert projected.entity_ids() == (entity_id,)
    assert int(np.asarray(projected.status)[0]) == SourceProjectionStatus.UNIQUE
    np.testing.assert_allclose(projected.parameters, ((0.5, 0.5),), atol=1.0e-12)
    np.testing.assert_allclose(projected.points, (expected,), atol=1.0e-12)

    domain = MeshingDomain.from_brep(model)
    result = _certified_surface_mesh(domain)
    mesh_coordinates = np.asarray(result.mesh.coordinates, dtype=np.float64)
    chart_samples = np.asarray(
        ((0.125, 0.25), (0.5, 0.5), (0.875, 0.75)), dtype=np.float64
    )
    sampled_surface = domain.evaluate(np.zeros((3,), dtype=np.int64), chart_samples)
    expected_height = (
        chart_samples[:, 0]
        * (1.0 - chart_samples[:, 0])
        * chart_samples[:, 1]
        * (1.0 - chart_samples[:, 1])
        / 4.0
    )

    assert result.mesh.topological_dimension == 2
    assert result.certification is not None and result.certification.passed
    assert result.compliance.passed
    np.testing.assert_allclose(sampled_surface[:, :2], chart_samples, atol=1.0e-12)
    np.testing.assert_allclose(sampled_surface[:, 2], expected_height, atol=1.0e-12)
    assert np.all(np.isfinite(mesh_coordinates))
    assert np.all(mesh_coordinates[:, 2] >= -1.0e-12)
    assert np.max(mesh_coordinates[:, 2]) <= 1.0 / 64.0 + 1.0e-12


def _identities_under_openblas_core(core: str | None, /) -> dict[str, str]:
    environment = {
        name: value for name, value in os.environ.items() if name != "OPENBLAS_CORETYPE"
    }
    if core is not None:
        environment["OPENBLAS_CORETYPE"] = core
    completed = subprocess.run(
        [sys.executable, "-c", _IDENTITY_SCRIPT],
        cwd=_REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.splitlines()[-1])


def test_t_spline_model_identities_do_not_depend_on_the_blas_kernel() -> None:
    native = _identities_under_openblas_core(None)
    generic = _identities_under_openblas_core(_GENERIC_OPENBLAS_CORES[platform.machine()])

    assert generic == native


def _nonuniform_transverse_patch() -> TSplineSurfacePatch:
    mesh = TMesh2D.tensor_product((Fraction(0), Fraction(1)), (Fraction(0), Fraction(1)))
    exact_weights = tuple(
        Fraction(1) + u / 100 + v / 200
        for u, v in (anchor.local_knots.greville_parameter for anchor in mesh.anchors)
    )
    exact_controls = tuple(
        (
            u,
            v,
            v + Fraction(1, 100) * u * (1 - u) * v * (1 - v),
        )
        for u, v in (anchor.local_knots.greville_parameter for anchor in mesh.anchors)
    )
    return TSplineSurfacePatch(
        mesh,
        np.asarray([[float(value) for value in point] for point in exact_controls]),
        np.asarray([float(value) for value in exact_weights], dtype=np.float64),
        exact_control_points=exact_controls,
        exact_weights=exact_weights,
    )


def _intersection_trim_domain(
    patch: TSplineSurfacePatch, branch: IntersectionCurve, /
) -> MeshingDomain:
    branch_pcurve = branch.p_curve("first")
    parameter_range = (0.0, float(branch.num_charts))
    endpoints = np.asarray(
        branch_pcurve.evaluate(jnp.asarray(parameter_range, dtype=jnp.float64)),
        dtype=np.float64,
    )
    if endpoints[0, 0] <= endpoints[1, 0]:
        left, right = endpoints
        branch_sense = -1
        branch_start, branch_end = 3, 2
    else:
        left, right = endpoints[1], endpoints[0]
        branch_sense = 1
        branch_start, branch_end = 2, 3
    bottom_left = np.asarray((0.0, 0.0), dtype=np.float64)
    bottom_right = np.asarray((1.0, 0.0), dtype=np.float64)
    uses = (
        PatchCurveUse(
            0,
            LineCurve(bottom_left, bottom_right - bottom_left),
            0.0,
            1.0,
        ),
        PatchCurveUse(
            1,
            LineCurve(bottom_right, right - bottom_right),
            0.0,
            1.0,
        ),
        PatchCurveUse(
            2,
            branch_pcurve,
            parameter_range[0],
            parameter_range[1],
            sense=branch_sense,
        ),
        PatchCurveUse(
            3,
            LineCurve(left, bottom_left - left),
            0.0,
            1.0,
        ),
    )
    return MeshingDomain(
        (MeshingSurfacePatch(patch, (uses,)),),
        (
            MeshingDomainCurve(0, 1),
            MeshingDomainCurve(1, 2),
            MeshingDomainCurve(branch_start, branch_end),
            MeshingDomainCurve(3, 0),
        ),
        4,
        source_id="tspline-intersection-trim",
        source_revision=branch.branch_id,
        regions=(MeshingDomainRegion("below-plane", ((0, 1),)),),
        tolerance=1.0e-9,
    )


def _plane_intersection_branch(patch: TSplineSurfacePatch, /) -> IntersectionCurve:
    result = intersect_surface_regions(
        SurfaceRegion(patch, np.asarray(((0.0, 0.0), (1.0, 1.0)), dtype=np.float64)),
        SurfaceRegion(
            PlanePatch(
                (0.0, 0.0, 0.5),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            ),
            np.asarray(((-0.1, -0.1), (1.1, 1.1)), dtype=np.float64),
        ),
    )
    assert result.complete and len(result.curves) == 1
    branch = result.curves[0]
    assert branch.fully_certified
    return branch


def test_t_spline_intersection_pcurve_endpoint_boxes_stay_in_patch_domain() -> None:
    patch = _nonuniform_transverse_patch()
    branch = _plane_intersection_branch(patch)
    pcurve = branch.p_curve("first")
    domain = np.asarray(patch.parameter_domain, dtype=np.float64)
    for parameter in (0.0, float(branch.num_charts)):
        box = np.asarray(pcurve.bounding_box(parameter, parameter), dtype=np.float64)
        endpoint = np.asarray(
            pcurve.evaluate(jnp.asarray(parameter, dtype=jnp.float64)),
            dtype=np.float64,
        )

        # The branch ends on the T-spline domain boundary; its rooted endpoint
        # enclosure must not leave that domain by an outward subnormal.
        assert np.all(box[0] <= endpoint) and np.all(endpoint <= box[1])
        assert np.all(box[0] >= domain[0]) and np.all(box[1] <= domain[1])
        assert np.all(box[0] < box[1])
        assert np.all(np.isfinite(patch.bounding_box(box)))


def test_certified_t_spline_intersection_pcurve_drives_native_trimmed_mesh() -> None:
    patch = _nonuniform_transverse_patch()
    branch = _plane_intersection_branch(patch)
    domain = _intersection_trim_domain(patch, branch)
    result = _certified_surface_mesh(domain)
    mesh_coordinates = np.asarray(result.mesh.coordinates, dtype=np.float64)
    curve_use = domain.patches[0].loops[0][2]
    boundary_parameters = np.linspace(
        curve_use.first, curve_use.last, 33, dtype=np.float64
    )
    boundary_uv = curve_use.pcurve.evaluate(jnp.asarray(boundary_parameters))
    boundary_points = patch.evaluate(boundary_uv)

    assert result.mesh.topological_dimension == 2
    assert result.certification is not None and result.certification.passed
    assert result.compliance.passed
    assert result.surface_source is not None
    assert result.surface_source.domain.domain_id == domain.domain_id
    assert float(np.max(mesh_coordinates[:, 2])) <= 0.5 + 2.0e-2
    np.testing.assert_allclose(np.asarray(boundary_points)[:, 2], 0.5, atol=2.0e-8)


def test_t_spline_transfer_chain_validation_reproves_a_restored_patch() -> None:
    patch = _nonuniform_transverse_patch()
    patch.validate_exact_transfer_chain()
    fields = {
        field.name: getattr(patch, field.name) for field in dataclasses.fields(patch)
    }
    restored = _restore_dataclass(TSplineSurfacePatch, fields)
    restored.validate_exact_transfer_chain()
    first, *rest = patch.exact_target_homogeneous
    edited_row = (first[0] + Fraction(1, 7), *first[1:])
    tampered = _restore_dataclass(
        TSplineSurfacePatch,
        fields | {"exact_target_homogeneous": (edited_row, *rest)},
    )
    with pytest.raises(ValueError, match="target bank differs from its transfer chain"):
        tampered.validate_exact_transfer_chain()
