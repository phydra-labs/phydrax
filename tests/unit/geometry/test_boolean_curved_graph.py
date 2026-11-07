import math
from fractions import Fraction

import numpy as np
import pytest

from phydrax import SpatialCoordinateContract
from phydrax.geometry.brep import boolean_brep, brep_box

from phydrax.geometry.brep._boolean_arrangement import _spatial_root_vertex
from phydrax.geometry.brep._intersection import (
    _cardinal_conic_angle_bounds,
    BranchRootEndpoint,
    CurveSurfaceIntersectionRoot,
    intersect_surface_regions,
)
from phydrax.geometry.brep._intersection_curve import SurfaceRegion
from phydrax.geometry.brep._model import _certify_edge_vertex_gap_bounds
from phydrax.geometry.brep._patches import (
    CircleCurve,
    ConicParameterRoot,
    LineCurve,
    PlanePatch,
)
from phydrax.geometry.brep._projection import prepare_brep_projection
from phydrax.geometry.brep._projection_contracts import BRepProjectionStatus
from phydrax.geometry.brep._root_bindings import BRepVertexRoot
from tools.meshing_qualification.scenarios.native_operation_sequences import (
    _blend_cut_sequence,
    _pi_bounds,
)


def test_cardinal_bounds_preserve_distinct_node_and_signed_gauges() -> None:
    conic = CircleCurve((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1.0)
    root = ConicParameterRoot(conic, Fraction(-1), Fraction(0))
    bounds = _cardinal_conic_angle_bounds(root)
    assert bounds is not None
    lower, upper = bounds
    pi_lower, pi_upper = _pi_bounds()
    assert lower <= pi_lower < pi_upper <= upper
    assert Fraction(math.nextafter(math.nextafter(math.pi, -math.inf), -math.inf)) < lower
    shifted = _cardinal_conic_angle_bounds(root, Fraction(1, 3), -1)
    assert shifted is not None
    low, high = shifted
    assert low <= Fraction(1, 3) - pi_upper < Fraction(1, 3) - pi_lower <= high
    noncardinal = ConicParameterRoot(conic, Fraction(3, 5), Fraction(4, 5))
    assert _cardinal_conic_angle_bounds(noncardinal) is None


def test_spatial_root_vertex_retains_source_ownership() -> None:
    line = LineCurve((0, 0, 0), (3, 0, 0))
    plane = SurfaceRegion(
        PlanePatch((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        np.asarray(((-1.0, -1.0), (1.0, 1.0)), dtype=np.float64),
    )
    root = CurveSurfaceIntersectionRoot(
        line,
        plane,
        parameter_lower=np.asarray((0.3, -0.1, -0.1), dtype=np.float64),
        parameter_upper=np.asarray((0.4, 0.1, 0.1), dtype=np.float64),
    )
    original = _spatial_root_vertex((0, 1), 2, (1, 3), root)
    assert original == _spatial_root_vertex((0, 7), 2, (1, 3), root)
    assert original != _spatial_root_vertex((0, 1), 9, (1, 3), root)
    assert original != _spatial_root_vertex((1, 1), 2, (1, 3), root)
    assert original != _spatial_root_vertex((0, 1), 2, (1, 4), root)


def test_branch_endpoint_gap_requires_real_root_incidence() -> None:
    first = SurfaceRegion(
        PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        np.asarray(((0.0, 0.0), (1.0, 1.0))),
    )
    second = SurfaceRegion(
        PlanePatch((0.5, 0, -1), (0, 1, 0), (0, 0, 1)),
        np.asarray(((0.0, 0.0), (1.0, 2.0))),
    )
    intersections = intersect_surface_regions(first, second)
    assert intersections.complete and len(intersections.curves) == 1
    branch = intersections.curves[0]
    line = LineCurve((0.0, 0.0, 0.0), (1.0, 0.5, 0.0))
    root = CurveSurfaceIntersectionRoot(
        line,
        second,
        parameter_lower=np.asarray((0.5 - 1e-14, 0.25 - 1e-14, 1.0 - 1e-14)),
        parameter_upper=np.asarray((0.5 + 1e-14, 0.25 + 1e-14, 1.0 + 1e-14)),
    )
    endpoint = BranchRootEndpoint(
        root,
        branch,
        0,
        source_pcurve=LineCurve((0.0, 0.0), (1.0, 0.5)),
        source_side="first",
        source_first=0.0,
        source_last=1.0,
    )
    lower, upper = endpoint.parameter_enclosure()
    assert lower < upper
    representative = lower + 0.5 * (upper - lower)
    binding = BRepVertexRoot(root)
    assert binding.supports_endpoint(endpoint)
    points = np.asarray(((0.5 + 1e-10, 0.25, 0.0),))
    ranges = np.asarray(((representative, representative),))

    def gap(vertex: BRepVertexRoot) -> tuple[tuple[float, float], ...] | None:
        return _certify_edge_vertex_gap_bounds(
            points, (None,), (vertex,), (branch,), (0,), ranges, ((0, 0),),
            ((endpoint, endpoint),),
        )

    certified_gap = gap(binding)
    assert certified_gap is not None
    assert 1e-10 <= certified_gap[0][0] < 1e-8
    wrong = BRepVertexRoot.from_curve_point(branch, 0.0)
    assert not wrong.supports_endpoint(endpoint)
    wrong_gap = gap(wrong)
    assert wrong_gap is not None and wrong_gap[0][0] > 0.0


def test_solid_index_scope_distinguishes_shared_definition_path() -> None:
    contract = SpatialCoordinateContract.si()
    first = brep_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), coordinate_contract=contract)
    second = brep_box((2.0, 0.0, 0.0), (3.0, 1.0, 1.0), coordinate_contract=contract)
    model = boolean_brep(first, second, "union").model
    assert model.topology.num_solids == 2
    projection = prepare_brep_projection(model)
    points = np.asarray(((1.0, 0.5, 0.5), (0.5, 0.5, 0.5)), dtype=np.float64)
    unscoped = projection.locate_solids(points)
    assert int(np.asarray(unscoped.status)[0]) == BRepProjectionStatus.AMBIGUOUS
    assert int(np.asarray(unscoped.status)[1]) == BRepProjectionStatus.UNIQUE
    containing = int(np.asarray(unscoped.indices)[1])
    other = 1 - containing
    scoped = projection.locate_solids(
        points,
        occurrence_paths=((), ()),
        solid_indices=np.asarray((other, containing), dtype=np.int32),
    )
    assert np.asarray(scoped.status).tolist() == [
        BRepProjectionStatus.FAILED,
        BRepProjectionStatus.UNIQUE,
    ]
    assert int(np.asarray(scoped.indices)[1]) == containing
    with pytest.raises(ValueError, match="solid_indices"):
        projection.locate_solids(points, solid_indices=np.asarray((0, 2)))
    with pytest.raises(ValueError, match="solid_indices"):
        projection.locate_solids(points, solid_indices=np.asarray((0.0, 1.0)))
    with pytest.raises(ValueError, match="solid_indices"):
        projection.locate_solids(points, solid_indices=np.asarray((0,)))
    with pytest.raises(ValueError, match="solid_indices"):
        projection.locate_solids(points, solid_indices=np.asarray((-1, 0)))


def test_blended_corner_split_retains_two_certified_solids() -> None:
    result = _blend_cut_sequence()
    cut = result["mixed_plane_cylinder_cut"]
    assert isinstance(cut, dict)
    assert cut["regions"] == 2
