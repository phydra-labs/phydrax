import math
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization._cell_geometry import (
    elevate_spline_extraction,
    SplineCellGeometryElement,
)
from phydrax.discretization._cell_geometry_validity import (
    cell_geometry_id,
    certify_cell_geometry_validity,
)
from phydrax.discretization._coordinate_enclosure import (
    coordinate_map_expressions,
    expression_equal,
    RationalPolynomial,
)
from phydrax.geometry._embedding_cells import prepare_embedding_cells
from phydrax.geometry._global_embedding import _retained_root_charts, _root_chart_contact
from phydrax.geometry._mapped_embedding import (
    _polynomial_scaled_jacobian,
    prove_mapped_cell_injective,
)
from phydrax.geometry._meshing_domain import (
    MeshingDomain,
    MeshingSurfacePatch,
    PatchCurveUse,
)
from phydrax.geometry._surface_source_support import (
    _SplineSurfaceCoordinateMap,
    prepare_surface_source_root_atlas,
)
from phydrax.geometry.brep import BSplineSurfacePatch, LineCurve
from phydrax.meshing._contracts import (
    MeshingFailure,
    MeshingFailureCategory,
    MeshQualityTarget,
    SurfaceMeshingSpec,
)
from phydrax.meshing._quality import (
    _with_certified_mapped_measures,
    evaluate_cell_quality,
    summarize_cell_quality,
)
from phydrax.meshing._sizing import (
    SizeCompliancePolicy,
    SizeControlStrength,
    UniformSizeControl,
)
from phydrax.meshing.providers import NativeMeshingProvider
from phydrax.meshing.providers._native_surface import prepare_native_surface
from phydrax.meshing.providers._native_surface_quad import (
    _mapped_corner_minimum_angle,
    _mapped_size_compliance,
    _require_mapped_quality,
)
from tools.meshing_qualification.scenarios.native_family_matrix import (
    _CONTRACT,
    _quarter_extrusion,
    _quarter_surface_case,
    _run_profile,
    _surface_specification,
)


def test_exact_tensor_extraction_elevation_retains_source_map() -> None:
    domain = _quarter_extrusion()
    atlas = prepare_surface_source_root_atlas(
        domain, (0,), SpatialCoordinateContract.si()
    )
    source_map = atlas.geometry.source_map
    assert isinstance(source_map, _SplineSurfaceCoordinateMap)
    expected_owner = (
        domain.source_id,
        domain.source_revision,
        domain.stratum_entity_ids(2)[0],
        domain.source_occurrences[2][0],
    )
    assert source_map.coefficient_authorities == tuple(
        (*expected_owner, "tensor-lattice", u, v) for u in range(2) for v in range(3)
    )
    elements, routes, _ = atlas.geometry.resolve(atlas.mesh)
    root = elements[0]
    assert isinstance(root, SplineCellGeometryElement)
    elevated = elevate_spline_extraction(root, (3, 3))
    controls = atlas.geometry.source_coordinates()
    local = tuple(controls[int(index)] for index in np.asarray(routes[0])[0])
    original_map = coordinate_map_expressions(root, local, cell_row=0)
    elevated_map = coordinate_map_expressions(elevated, local, cell_row=0)
    assert elevated.degrees == (3, 3)
    assert elevated.anchor_ids == root.anchor_ids
    assert elevated.exact_weights == root.exact_weights
    assert all(
        expression_equal(a, b) for a, b in zip(original_map, elevated_map, strict=True)
    )
    assert all(
        isinstance(value, Fraction)
        for row in elevated.exact_extraction_operator
        for value in row
    )


def test_exact_rational_root_has_whole_cell_injectivity_proof() -> None:
    domain = _quarter_extrusion()
    atlas = prepare_surface_source_root_atlas(
        domain, (0,), SpatialCoordinateContract.si()
    )
    (cell,) = prepare_embedding_cells(atlas.mesh, atlas.geometry)
    proof = prove_mapped_cell_injective(cell)
    assert proof is not None
    assert proof.denominator_controls


def test_rational_jacobian_clearing_requires_nonvanishing_denominator() -> None:
    zero = (0, 0)
    x = (1, 0)
    good = RationalPolynomial({zero: Fraction(1)}, {zero: Fraction(1), x: Fraction(1)})
    bad = RationalPolynomial({zero: Fraction(1)}, {zero: Fraction(-1, 2), x: Fraction(1)})
    proven = _polynomial_scaled_jacobian(((good,),), "box", 2)
    assert proven is not None
    assert proven[0] == (({zero: Fraction(1)},),)
    assert proven[1]
    assert _polynomial_scaled_jacobian(((bad,),), "box", 2) is None


def test_quad_cells_retain_injective_common_root_charts() -> None:
    case = _quarter_surface_case(4)
    assert isinstance(case.specification, SurfaceMeshingSpec)
    prepared = prepare_native_surface(
        case.source, case.specification, case.options, _CONTRACT
    )
    construction = prepared.execute()
    cells = prepare_embedding_cells(construction.mesh, construction.geometry)
    roots = _retained_root_charts(construction.mesh, construction.geometry, cells)
    retained = tuple(root for root in roots if root is not None)
    assert len(retained) == len(roots)
    assert all(root.injective for root in retained)
    for i, first in enumerate(cells):
        for j, second in enumerate(cells[i + 1 :], i + 1):
            if set(first.vertex_ids) & set(second.vertex_ids):
                assert _root_chart_contact(retained[i], retained[j]) == "declared", (i, j)


def test_mapped_quality_reports_certified_physical_area() -> None:
    domain = _quarter_extrusion()
    atlas = prepare_surface_source_root_atlas(
        domain, (0,), SpatialCoordinateContract.si()
    )
    validity = certify_cell_geometry_validity(atlas.geometry, mesh=atlas.mesh)
    quality = _with_certified_mapped_measures(
        evaluate_cell_quality(atlas.mesh), atlas.mesh, atlas.geometry, validity
    )
    assert quality.measure_geometry_id == cell_geometry_id(atlas.geometry)
    assert quality.measure_absolute_errors is not None
    assert quality.measure_exact is False
    error = float(np.asarray(quality.measure_absolute_errors)[0])
    value = float(np.asarray(quality.measures)[0])
    assert abs(value - math.pi / 2) <= error + 1e-15
    assert (
        summarize_cell_quality(atlas.mesh, quality).report_id
        != summarize_cell_quality(
            atlas.mesh, replace(quality, measure_exact=True)
        ).report_id
    )


def test_original_exact_quarter_quad_family_gate() -> None:
    record = _run_profile("native-family-lifecycle:curved-quarter-extrusion-quad")
    assert record["status"] == "passed", record.get("failure")


def test_exact_quarter_source_recipe_replays_and_rejects_forgery(
    tmp_path: Path,
) -> None:
    from phydrax._model._structure import model_archive_projection
    from phydrax.geometry._surface_source_support import (
        SurfaceNativeRestrictionBoundarySource,
    )
    from phydrax.meshing._model_codec import (
        read_model_artifact,
        restore_model_archive_value,
        write_model_artifact,
    )
    from phydrax.meshing._result import CellMeshingResult

    case = _quarter_surface_case(4)
    result = (
        NativeMeshingProvider(case.options)
        .plan(case.source, case.specification, coordinate_contract=_CONTRACT)
        .execute()
    )
    assert result.certification is not None
    source = result.certification.inputs.source
    assert isinstance(source, SurfaceNativeRestrictionBoundarySource)
    recipe = model_archive_projection(source)
    restored = restore_model_archive_value(recipe)
    assert isinstance(restored, SurfaceNativeRestrictionBoundarySource)
    restored.require_current()
    assert restored.source_scope_id == source.source_scope_id
    with pytest.raises(ValueError):
        restore_model_archive_value(
            recipe._replace(support=recipe.support._replace(expected_support_id="forged"))
        )
    with pytest.raises(ValueError):
        restore_model_archive_value(recipe._replace(source_revision="forged"))
    path = tmp_path / "quarter-result.phx"
    write_model_artifact(path, result, limits=None)
    replayed = read_model_artifact(
        path,
        expected_type=CellMeshingResult,
        restore=restore_model_archive_value,
        limits=None,
    )
    assert replayed.result_id == result.result_id


def test_tensor_quad_refuses_partial_rectangle_trim() -> None:
    case = _quarter_surface_case(4)
    uv = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 0.5)))
    loop = tuple(
        PatchCurveUse(i, LineCurve(uv[i], uv[(i + 1) % 4] - uv[i]), 0.0, 1.0)
        for i in range(4)
    )
    domain = MeshingDomain(
        (MeshingSurfacePatch(case.source.patches[0].surface, (loop,)),),
        case.source.curves,
        4,
        source_id="partial-tensor-trim",
        source_revision="r1",
    )
    specification = _surface_specification(
        domain, 4, "quadrilateral", ambient_dimension=3, geometry_order=3
    )
    prepared = prepare_native_surface(domain, specification, case.options, _CONTRACT)
    with pytest.raises(MeshingFailure) as error:
        prepared.execute()
    assert error.value.category is MeshingFailureCategory.UNSUPPORTED_COMBINATION


def test_hard_scoped_minimum_size_is_enforced() -> None:
    case = _quarter_surface_case(4)
    assert isinstance(case.specification, SurfaceMeshingSpec)
    control = UniformSizeControl(
        case.specification.scope,
        0.7,
        minimum_size=0.65,
        maximum_size=1.0,
        strength=SizeControlStrength.HARD,
    )
    specification = SurfaceMeshingSpec(
        case.specification.target,
        case.specification.scope,
        size_controls=(control,),
        protected_features=case.specification.protected_features,
        size_compliance=SizeCompliancePolicy(
            relative_tolerance=0.05, target_statistics=("p50",)
        ),
        quality_target=case.specification.quality_target,
    )
    construction = prepare_native_surface(
        case.source, specification, case.options, _CONTRACT
    ).execute()
    assert f"{control.control_id}:minimum_size" in construction.compliance.issues
    assert not construction.compliance.passed
    assert f"{control.control_id}:maximum_size" not in construction.compliance.issues
    allowed = UniformSizeControl(
        case.specification.scope,
        0.7,
        minimum_size=0.01,
        maximum_size=1.0,
        strength=SizeControlStrength.HARD,
    )
    allowed_specification = SurfaceMeshingSpec(
        case.specification.target,
        case.specification.scope,
        size_controls=(allowed,),
        size_compliance=SizeCompliancePolicy(
            relative_tolerance=0.95, target_statistics=("p50",)
        ),
    )
    allowed_prepared = prepare_native_surface(
        case.source, allowed_specification, case.options, _CONTRACT
    )
    _, _, allowed_issues = _mapped_size_compliance(
        allowed_prepared, construction.mesh, construction.geometry
    )
    assert not allowed_issues
    with pytest.raises(MeshingFailure):
        NativeMeshingProvider(case.options).plan(
            case.source, specification, coordinate_contract=_CONTRACT
        ).execute()


def test_hard_quad_quality_uses_mapped_tangents_not_chords() -> None:
    original = _quarter_extrusion()
    surface = original.patches[0].surface
    assert isinstance(surface, BSplineSurfacePatch)
    controls = np.array(surface.control_points, dtype=np.float64, copy=True)
    controls[:, :, 0] += np.asarray((0.0, 0.0, 4.0))
    warped = BSplineSurfacePatch(
        controls,
        surface.weights,
        surface.u_knots,
        surface.v_knots,
        surface.u_degree,
        surface.v_degree,
    )
    domain = MeshingDomain(
        (MeshingSurfacePatch(warped, original.patches[0].loops),),
        original.curves,
        4,
        source_id="warped-quality-source",
        source_revision="r1",
    )
    atlas = prepare_surface_source_root_atlas(
        domain, (0,), SpatialCoordinateContract.si()
    )
    tangent_angle = _mapped_corner_minimum_angle(atlas.mesh, atlas.geometry)
    chord_angle = float(
        np.min(np.asarray(evaluate_cell_quality(atlas.mesh).minimum_angle))
    )
    assert tangent_angle < chord_angle
    threshold = (tangent_angle + chord_angle) / 2.0
    assert 0.0 < threshold < math.pi / 3.0
    with pytest.raises(MeshingFailure) as error:
        _require_mapped_quality(
            atlas.mesh,
            atlas.geometry,
            MeshQualityTarget(minimum_angle=threshold, hard=True),
        )
    assert error.value.category is MeshingFailureCategory.COMPLIANCE_FAILED
