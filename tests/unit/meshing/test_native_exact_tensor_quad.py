import math
from fractions import Fraction

import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization._cell_geometry import elevate_spline_extraction
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
from phydrax.geometry._surface_source_support import prepare_surface_source_root_atlas
from phydrax.geometry.brep import LineCurve
from phydrax.meshing._contracts import MeshingFailure, MeshingFailureCategory
from phydrax.meshing._quality import (
    _with_certified_mapped_measures,
    evaluate_cell_quality,
)
from phydrax.meshing.providers._native_surface import prepare_native_surface
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
    prepared = prepare_native_surface(
        case.source, case.specification, case.options, _CONTRACT
    )
    construction = prepared.execute()
    cells = prepare_embedding_cells(construction.mesh, construction.geometry)
    roots = _retained_root_charts(construction.mesh, construction.geometry, cells)
    assert all(root is not None for root in roots), sum(root is None for root in roots)
    assert all(root.injective for root in roots)
    for i, first in enumerate(cells):
        for j, second in enumerate(cells[i + 1 :], i + 1):
            if set(first.vertex_ids) & set(second.vertex_ids):
                assert _root_chart_contact(roots[i], roots[j]) == "declared", (i, j)


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
    error = float(np.asarray(quality.measure_absolute_errors)[0])
    value = float(np.asarray(quality.measures)[0])
    assert abs(value - math.pi / 2) <= error + 1e-15


def test_original_exact_quarter_quad_family_gate() -> None:
    record = _run_profile("native-family-lifecycle:curved-quarter-extrusion-quad")
    assert record["status"] == "passed", record.get("failure")


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
