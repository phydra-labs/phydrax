"""Tessellation refusal evidence and white-box source-bound fault checks.

The phase tests alter one bound in a native-generated artifact. Each refusal
must identify the violated obligation and retain its measured value.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

import equinox as eqx
import numpy as np
import pytest

from phydrax.geometry.brep import (
    BRepEntityId,
    BRepModel,
    BRepTessellationFailure,
    BRepTessellationPolicy,
)
from phydrax.geometry.brep._tessellation import (
    _assemble_template,
    _certify_execution,
    _certify_placements,
    _execute_request,
    _place_occurrences,
    _prepare_request,
    _TessellationExecution,
    _TessellationRequest,
    BRepTessellation,
    tessellate_brep,
)
from phydrax.meshing import MeshingFailure, MeshingFailureCategory
from phydrax.meshing._surface_construction import SurfaceConstruction
from tests._support.cad_models import native_repeated_assembly, native_solid


# These are declared fidelity limits, not oracle tolerances. Faults double a
# limit so that comparisons are independent of platform-level rounding.
_LINEAR_DEFLECTION = 0.05
_ANGULAR_DEFLECTION = 0.4
_POLICY = BRepTessellationPolicy(
    linear_deflection=_LINEAR_DEFLECTION,
    angular_deflection=_ANGULAR_DEFLECTION,
)


@dataclass(frozen=True)
class _ExecutedBox:
    model: BRepModel
    request: _TessellationRequest
    execution: _TessellationExecution
    template: BRepTessellation


@pytest.fixture(scope="module")
def executed_box() -> _ExecutedBox:
    model = native_solid("box", tessellation=BRepTessellationPolicy(realize=False))
    assert model.geometry is not None
    request = _prepare_request(
        model.geometry,
        model.patches,
        np.asarray(model.orientation, dtype=np.float64),
        _POLICY,
        source_id=model.source_id,
        source_revision=model.source_revision,
    )
    execution = _execute_request(request, _POLICY)
    certified = _certify_execution(request, execution, _POLICY)
    template = _assemble_template(request, certified)
    return _ExecutedBox(model, request, execution, template)


def _triangle_deviation(construction: SurfaceConstruction, /) -> np.ndarray:
    return construction.triangle_deviation_bounds


def _curve_deviation(construction: SurfaceConstruction, /) -> np.ndarray:
    return construction.curve_edge_deviation_bounds


def _normal_angle(construction: SurfaceConstruction, /) -> np.ndarray:
    return construction.triangle_normal_bounds


def _triangle_indices(construction: SurfaceConstruction, /) -> np.ndarray:
    return construction.triangle_patches


def _curve_indices(construction: SurfaceConstruction, /) -> np.ndarray:
    return construction.curve_edge_curves


@pytest.mark.parametrize(
    "selector,index_selector,field,limit,dimension,kind",
    (
        (
            _triangle_deviation,
            _triangle_indices,
            "triangle_deviation",
            _LINEAR_DEFLECTION,
            2,
            "face",
        ),
        (
            _curve_deviation,
            _curve_indices,
            "curve_deviation",
            _LINEAR_DEFLECTION,
            1,
            "edge",
        ),
        (
            _normal_angle,
            _triangle_indices,
            "normal_angle",
            _ANGULAR_DEFLECTION,
            2,
            "face",
        ),
    ),
    ids=("triangle-deviation", "edge-deviation", "normal-turn"),
)
def test_source_bound_refusal_carries_qualified_entity(
    executed_box: _ExecutedBox,
    selector: Callable[[SurfaceConstruction], np.ndarray],
    index_selector: Callable[[SurfaceConstruction], np.ndarray],
    field: str,
    limit: float,
    dimension: int,
    kind: str,
) -> None:
    construction = executed_box.execution.construction
    bounds = selector(construction).copy()
    bounds[0] = 2.0 * limit
    changed = eqx.tree_at(selector, construction, replace=bounds)
    execution = _TessellationExecution(
        construction=changed, cover=executed_box.execution.cover
    )
    with pytest.raises(BRepTessellationFailure, match=field) as raised:
        _certify_execution(executed_box.request, execution, _POLICY)
    failure = raised.value
    indices = index_selector(construction)
    entity = BRepEntityId(
        executed_box.model.source_revision,
        kind,
        int(executed_box.request.domain.source_indices[dimension][indices[0]]),
    )
    assert failure.category is MeshingFailureCategory.COMPLIANCE_FAILED
    assert failure.measured_bound == 2.0 * limit
    assert failure.limit == limit
    assert failure.entity == entity
    assert failure.entity_ids == (entity.index,)
    assert failure.requested == ((field, limit),)
    assert failure.achieved == ((field, 2.0 * limit),)
    assert failure.stage == "brep-tessellation"


def test_occurrence_triangle_budget_carries_measured_count(
    executed_box: _ExecutedBox,
) -> None:
    geometry = native_repeated_assembly().geometry
    assert geometry is not None
    maximum = executed_box.template.triangles.shape[0]
    with pytest.raises(BRepTessellationFailure, match="maximum_triangles") as raised:
        _place_occurrences(executed_box.template, geometry, maximum)
    failure = raised.value
    assert failure.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert failure.measured_bound == 2.0 * maximum
    assert failure.limit == maximum
    assert failure.entity is None
    assert failure.entity_ids == ()
    assert failure.requested == (("maximum_triangles", float(maximum)),)
    assert failure.achieved == (("maximum_triangles", 2.0 * maximum),)


def test_native_triangle_budget_retains_resource_failure_evidence() -> None:
    with pytest.raises(MeshingFailure) as raised:
        native_solid("box", tessellation=BRepTessellationPolicy(maximum_triangles=1))
    assert raised.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert raised.value.evidence.evidence_id


def _placed_deviation_fault(
    placed: BRepTessellation, row: int, value: float, /
) -> BRepTessellation:
    bounds = placed.deviation_bounds.copy()
    bounds[row] = value
    return replace(placed, deviation_bounds=bounds)


def _placed_normal_fault(
    placed: BRepTessellation, row: int, value: float, /
) -> BRepTessellation:
    bounds = placed.normal_bounds.copy()
    bounds[row] = value
    return replace(placed, normal_bounds=bounds)


@pytest.mark.parametrize(
    "fault,field,limit",
    (
        (_placed_deviation_fault, "placed_deviation", _LINEAR_DEFLECTION),
        (_placed_normal_fault, "placed_normal_angle", _ANGULAR_DEFLECTION),
    ),
    ids=("placed-deviation", "placed-normal-turn"),
)
def test_placed_bound_refusal_retains_occurrence_path(
    executed_box: _ExecutedBox,
    fault: Callable[[BRepTessellation, int, float], BRepTessellation],
    field: str,
    limit: float,
) -> None:
    geometry = native_repeated_assembly().geometry
    assert geometry is not None
    placed = _place_occurrences(
        executed_box.template, geometry, _POLICY.maximum_triangles
    )
    row = int(np.flatnonzero(placed.triangle_occurrence_ids == 1)[0])
    changed = fault(placed, row, 2.0 * limit)
    with pytest.raises(BRepTessellationFailure, match=field) as raised:
        _certify_placements(
            changed, geometry, _POLICY, executed_box.model.source_revision
        )
    failure = raised.value
    assert failure.category is MeshingFailureCategory.COMPLIANCE_FAILED
    assert failure.measured_bound == 2.0 * limit
    assert failure.limit == limit
    assert failure.entity == BRepEntityId(
        executed_box.model.source_revision,
        "face",
        int(placed.face_ids[row]),
        geometry.occurrences[1].path,
    )


def test_occurrence_selection_preserves_template_and_disjoint_connectivity(
    executed_box: _ExecutedBox,
) -> None:
    geometry = native_repeated_assembly().geometry
    assert geometry is not None
    template = executed_box.template
    triangles, parameters = template.triangles.copy(), template.parameters.copy()
    placed = _place_occurrences(template, geometry, _POLICY.maximum_triangles)
    np.testing.assert_array_equal(template.triangles, triangles)
    np.testing.assert_array_equal(template.parameters, parameters)
    assert placed.triangles.shape[0] == 2 * template.triangles.shape[0]
    first = placed.triangles[placed.triangle_occurrence_ids == 0]
    second = placed.triangles[placed.triangle_occurrence_ids == 1]
    assert np.intersect1d(first, second).size == 0
    np.testing.assert_array_equal(
        placed.vertex_source_indices[: template.vertices.shape[0]],
        template.vertex_source_indices,
    )


def test_unrealized_tessellation_retains_empty_host_contract(
    executed_box: _ExecutedBox,
) -> None:
    model = executed_box.model
    assert model.geometry is not None
    result = tessellate_brep(
        model.geometry,
        model.patches,
        np.asarray(model.orientation, dtype=np.float64),
        BRepTessellationPolicy(realize=False),
        source_id=model.source_id,
        source_revision=model.source_revision,
    )
    assert result.vertices.shape == (0, 3)
    assert result.vertices.dtype == np.float64
    assert result.triangles.shape == (0, 3)
    assert result.triangles.dtype == np.int32
    assert result.parameters.shape == (0, 3, 2)
    assert result.chart_restriction_edges.shape == (0, 2)
    assert result.chart_restriction_endpoint_parameters.shape == (0, 2, 2)
    assert result.triangle_occurrence_ids.shape == (0,)


def test_source_cover_deviation_refusal_is_model_wide(
    executed_box: _ExecutedBox,
) -> None:
    bound = float(np.max(executed_box.execution.cover.deviation_bounds))
    assert bound > 0.0
    policy = BRepTessellationPolicy(
        linear_deflection=bound / 2.0, angular_deflection=_ANGULAR_DEFLECTION
    )
    with pytest.raises(BRepTessellationFailure, match="cover_deviation") as raised:
        _certify_execution(executed_box.request, executed_box.execution, policy)
    failure = raised.value
    assert failure.category is MeshingFailureCategory.COMPLIANCE_FAILED
    assert failure.measured_bound == bound
    assert failure.limit == bound / 2.0
    assert failure.entity is None


def test_unresolved_topology_refuses_before_fidelity(
    executed_box: _ExecutedBox,
) -> None:
    construction = executed_box.execution.construction
    unresolved = construction.unresolved.copy()
    unresolved[0] = True
    changed = eqx.tree_at(
        lambda result: result.unresolved, construction, replace=unresolved
    )
    execution = _TessellationExecution(
        construction=changed, cover=executed_box.execution.cover
    )
    bound = float(np.max(executed_box.execution.cover.deviation_bounds))
    policy = BRepTessellationPolicy(
        linear_deflection=bound / 2.0, angular_deflection=_ANGULAR_DEFLECTION
    )
    with pytest.raises(BRepTessellationFailure, match="unresolved") as raised:
        _certify_execution(executed_box.request, execution, policy)
    failure = raised.value
    assert failure.category is MeshingFailureCategory.AUDIT_FAILED
    assert failure.measured_bound == 1.0
    assert failure.limit == 0.0
    assert failure.entity == BRepEntityId(
        executed_box.model.source_revision,
        "face",
        int(
            executed_box.request.domain.source_indices[2][
                construction.triangle_patches[0]
            ]
        ),
    )


def test_checked_geometry_refuses_before_unrealized_shortcut() -> None:
    with pytest.raises(TypeError, match="geometry"):
        tessellate_brep(
            None,  # ty: ignore[invalid-argument-type]
            (),
            np.empty((0,), dtype=np.float64),
            BRepTessellationPolicy(realize=False),
            source_id="invalid-geometry",
            source_revision="invalid-revision",
        )
