"""Constructor and identity contracts for native layer/core source bindings."""

from dataclasses import replace

import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import meshcore_available
from phydrax.meshing.providers._native_layer import _lower_core_specification
from tests._support.native_layer_sources import _Case, _case, _physical_layer_source
from tests._support.source_authority import _cube_domain


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="native layer preparation requires meshcore"
)
M = phx.meshing


def _source(case: _Case) -> M.NativeLayerCoreSource:
    return M.NativeLayerCoreSource(
        case.layers,
        case.core,
        "layered-square",
        "reference",
        vertex_layer_ids=case.vertex_layer_ids,
        cap_polygon_ids=case.cap_polygon_ids,
        layer_regions=case.layer_regions,
        region_ids=case.region_ids,
        core_region_map=case.core_region_map,
    )


@pytest.mark.parametrize(
    "field",
    ("vertex_layer_ids", "cap_polygon_ids", "layer_regions"),
    ids=("vertex-map", "cap-incidence", "cell-materials"),
)
def test_layer_identity_arrays_reject_noninteger_dtype(field: str) -> None:
    case = _case()
    values = {
        "vertex_layer_ids": case.vertex_layer_ids,
        "cap_polygon_ids": case.cap_polygon_ids,
        "layer_regions": case.layer_regions,
    }
    with pytest.raises(TypeError, match=rf"NativeLayerCoreSource\.{field}"):
        _source(replace(case, **{field: values[field].astype(np.float64)}))


def test_layer_source_freezes_identity_without_aliasing_inputs() -> None:
    case = _case(material_interface=True)
    source = _source(case)
    same = _source(
        replace(
            case,
            vertex_layer_ids=case.vertex_layer_ids.astype(np.int32),
            cap_polygon_ids=case.cap_polygon_ids.astype(np.int32),
            layer_regions=case.layer_regions.astype(np.int32),
        )
    )
    assert same.binding_id == source.binding_id
    vertices = source.vertex_layer_ids.copy()
    case.vertex_layer_ids[:] = -1
    np.testing.assert_array_equal(source.vertex_layer_ids, vertices)
    assert not source.vertex_layer_ids.flags.writeable
    assert not source.cap_polygon_ids.flags.writeable
    assert not source.layer_regions.flags.writeable
    assert source.region_ids == ("wall-material", "core-material")
    np.testing.assert_array_equal(source.core_region_map, (1,))


def test_layer_source_validates_maps_before_periodic_ancestry() -> None:
    case = _case()
    with pytest.raises(TypeError, match="NativeLayerCoreSource.vertex_layer_ids"):
        M.NativeLayerCoreSource(
            case.layers,
            case.core,
            "layered-square",
            "reference",
            vertex_layer_ids=case.vertex_layer_ids.astype(np.float64),
            cap_polygon_ids=case.cap_polygon_ids,
            layer_regions=case.layer_regions,
            core_vertex_representatives=np.zeros(1, dtype=np.int64),
        )


def test_layer_source_requires_all_periodic_ancestry_fields() -> None:
    case = _case()
    with pytest.raises(ValueError, match="NativeLayerCoreSource.core_periodic_ancestry"):
        M.NativeLayerCoreSource(
            case.layers,
            case.core,
            "layered-square",
            "reference",
            vertex_layer_ids=case.vertex_layer_ids,
            cap_polygon_ids=case.cap_polygon_ids,
            layer_regions=case.layer_regions,
            core_vertex_representatives=np.zeros(
                case.core.vertices.shape[0], dtype=np.int64
            ),
        )


def test_layer_source_reference_domain_is_jointly_required() -> None:
    case = _case()
    reference = _source(case)
    with pytest.raises(ValueError, match="NativeLayerCoreSource.reference_source"):
        M.NativeLayerCoreSource(
            case.layers,
            case.core,
            "layered-square",
            "reference",
            vertex_layer_ids=case.vertex_layer_ids,
            cap_polygon_ids=case.cap_polygon_ids,
            layer_regions=case.layer_regions,
            reference_source=reference,
        )


def test_core_specification_evolution_preserves_canonical_request_identity() -> None:
    source = _source(_case())
    boundary = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        "layered-core-facets",
        np.arange(source.complex.facet_count, dtype=np.int64),
    )
    target = M.CellMeshingTarget(
        3, 3, M.CellFamilyPolicy(required=("prism", "tetrahedron"), allow_mixed=True)
    )
    size = M.UniformSizeControl(boundary, 2.0, strength=M.SizeControlStrength.SOFT)
    request = M.VolumeMeshingSpec(
        target,
        boundary,
        M.VolumeFillStrategy.SIMPLEX,
        size_controls=(size,),
        region_controls=(
            M.RegionControl(
                M.MeshingScope(
                    source.source_id,
                    source.source_revision,
                    M.MeshingEntityKind.GEOMETRY,
                    3,
                    f"{source.complex.complex_id}:regions",
                    np.asarray((0,), dtype=np.int64),
                ),
                "fluid",
                "fluid-material",
                M.RegionRole.FLUID,
            ),
        ),
        patch_controls=(M.PatchControl("exterior", boundary, ("fluid",)),),
    )
    expected = M.VolumeMeshingSpec(
        target,
        boundary,
        M.VolumeFillStrategy.SIMPLEX,
        size_controls=request.size_controls,
        protected_features=request.protected_features,
        region_controls=request.region_controls,
        region_seeds=request.region_seeds,
        hole_seeds=request.hole_seeds,
        size_combination=request.size_combination,
        size_compliance=request.size_compliance,
        limits=request.limits,
        deterministic=request.deterministic,
    )
    lowered = _lower_core_specification(source, request)
    assert lowered.specification_id == expected.specification_id
    assert lowered.patch_controls == ()
    assert lowered.layer_controls == ()
    assert lowered.periodic_constraints == ()
    assert request.patch_controls != ()
    assert lowered.boundary_scope.scope_id == request.boundary_scope.scope_id


def test_layer_lowering_identifies_mismatching_source_domain_field() -> None:
    source, specification = _physical_layer_source(
        source_domain=_cube_domain(parameter_scale=2.0)
    )
    with pytest.raises(ValueError, match="NativeLayerCoreSource.source_domain.domain_id"):
        _lower_core_specification(source, specification)


def test_layer_lowering_identifies_mismatching_boundary_scope_field() -> None:
    source, specification = _physical_layer_source()
    boundary = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        "foreign-boundary",
        specification.boundary_scope.entity_ids,
    )
    changed = replace(
        specification,
        boundary_scope=boundary,
        size_controls=(
            M.UniformSizeControl(boundary, 2.0, strength=M.SizeControlStrength.SOFT),
        ),
        layer_controls=(),
    )
    with pytest.raises(ValueError, match="VolumeMeshingSpec.boundary_scope.scope_id"):
        _lower_core_specification(source, changed)


def test_layer_source_checked_boundary_rejects_wrong_layer_kind() -> None:
    case = _case()
    with pytest.raises(TypeError, match="layers"):
        M.NativeLayerCoreSource(
            case.core,  # ty: ignore[invalid-argument-type]
            case.core,
            "layered-square",
            "reference",
            vertex_layer_ids=case.vertex_layer_ids,
            cap_polygon_ids=case.cap_polygon_ids,
            layer_regions=case.layer_regions,
        )
