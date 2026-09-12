import numpy as np
import pytest

import phydrax as phx


def _scope(dimension=2, ids=(10, 20), *, revision="revision-1"):
    return phx.meshing.MeshingScope(
        "shape",
        revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        dimension,
        f"entities-{dimension}",
        np.asarray(ids, dtype=np.int64),
    )


def test_surface_and_volume_specs_share_generic_semantic_and_sizing_controls():
    surface_scope = _scope()
    surface_target = phx.meshing.CellMeshingTarget(
        2,
        3,
        phx.meshing.CellFamilyPolicy(required=("triangle",)),
    )
    surface_regions = (
        phx.meshing.RegionControl(
            _scope(2, (10,)),
            "left",
            "material-left",
            phx.meshing.RegionRole.SOLID,
        ),
        phx.meshing.RegionControl(
            _scope(2, (20,)),
            "right",
            "material-right",
            phx.meshing.RegionRole.SOLID,
        ),
    )
    surface_patch = phx.meshing.PatchControl(
        "left-boundary",
        _scope(1, (30,)),
        ("left",),
    )
    compliance = phx.meshing.SizeCompliancePolicy(
        absolute_tolerance=1.0e-8,
        relative_tolerance=1.0e-3,
    )
    surface = phx.meshing.SurfaceMeshingSpec(
        surface_target,
        surface_scope,
        size_controls=(phx.meshing.UniformSizeControl(surface_scope, 0.1),),
        region_controls=surface_regions,
        patch_controls=(surface_patch,),
        size_combination=phx.meshing.SizeCombinationPolicy.EXPLICIT_PRIORITY,
        size_compliance=compliance,
    )

    volume_scope = _scope(3)
    volume_target = phx.meshing.CellMeshingTarget(
        3,
        3,
        phx.meshing.CellFamilyPolicy(required=("tetrahedron",)),
    )
    volume_regions = (
        phx.meshing.RegionControl(
            _scope(3, (10,)),
            "fluid",
            "water",
            phx.meshing.RegionRole.FLUID,
        ),
        phx.meshing.RegionControl(
            _scope(3, (20,)),
            "solid",
            "steel",
            phx.meshing.RegionRole.SOLID,
        ),
    )
    interface = phx.meshing.PatchControl(
        "fluid-solid",
        _scope(2, (40,)),
        ("solid", "fluid"),
    )
    reversed_interface = phx.meshing.PatchControl(
        "fluid-solid",
        _scope(2, (40,)),
        ("fluid", "solid"),
    )
    optional_interface = phx.meshing.PatchControl(
        "fluid-solid",
        _scope(2, (40,)),
        ("fluid", "solid"),
        required=False,
    )
    repeated_pair = phx.meshing.PatchControl(
        "second-fluid-solid",
        _scope(2, (50,)),
        ("fluid", "solid"),
        required=False,
    )
    volume = phx.meshing.VolumeMeshingSpec(
        volume_target,
        volume_scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(phx.meshing.UniformSizeControl(volume_scope, 0.2),),
        region_controls=volume_regions,
        patch_controls=(interface, repeated_pair),
        size_combination=phx.meshing.SizeCombinationPolicy.EXPLICIT_PRIORITY,
        size_compliance=compliance,
    )

    assert surface.region_controls == surface_regions
    assert surface.patch_controls == (surface_patch,)
    assert surface.size_combination is volume.size_combination
    assert surface.size_compliance.policy_id == volume.size_compliance.policy_id
    assert interface.adjacent_region_names == ("fluid", "solid")
    assert interface.control_id == reversed_interface.control_id
    assert interface.required
    assert not optional_interface.required
    assert interface.control_id != optional_interface.control_id
    assert volume.patch_controls == (interface, repeated_pair)
    assert surface.specification_id != volume.specification_id


def test_generic_semantic_controls_reject_wrong_scope_overlap_and_unknown_region():
    scope = _scope(3)
    target = phx.meshing.CellMeshingTarget(
        3,
        3,
        phx.meshing.CellFamilyPolicy(required=("tetrahedron",)),
    )
    size = (phx.meshing.UniformSizeControl(scope, 0.2),)
    regions = (
        phx.meshing.RegionControl(
            _scope(3, (10,)),
            "first",
            "material-first",
            phx.meshing.RegionRole.SOLID,
        ),
        phx.meshing.RegionControl(
            _scope(3, (20,)),
            "second",
            "material-second",
            phx.meshing.RegionRole.SOLID,
        ),
    )
    with pytest.raises(ValueError, match="one or two"):
        phx.meshing.PatchControl("invalid", _scope(2, (30,)), ())
    with pytest.raises(ValueError, match="codimension-one"):
        phx.meshing.VolumeMeshingSpec(
            target,
            scope,
            phx.meshing.VolumeFillStrategy.SIMPLEX,
            size_controls=size,
            region_controls=regions,
            patch_controls=(
                phx.meshing.PatchControl("wrong-dimension", _scope(1, (30,)), ("first",)),
            ),
        )
    with pytest.raises(ValueError, match="must be disjoint"):
        phx.meshing.VolumeMeshingSpec(
            target,
            scope,
            phx.meshing.VolumeFillStrategy.SIMPLEX,
            size_controls=size,
            region_controls=(
                regions[0],
                phx.meshing.RegionControl(
                    _scope(3, (10, 20)),
                    "overlap",
                    "material-overlap",
                    phx.meshing.RegionRole.SOLID,
                ),
            ),
        )
    with pytest.raises(ValueError, match="declared region names"):
        phx.meshing.VolumeMeshingSpec(
            target,
            scope,
            phx.meshing.VolumeFillStrategy.SIMPLEX,
            size_controls=size,
            region_controls=regions,
            patch_controls=(
                phx.meshing.PatchControl("unknown", _scope(2, (30,)), ("absent",)),
            ),
        )
    with pytest.raises(ValueError, match="top-level source binding"):
        phx.meshing.VolumeMeshingSpec(
            target,
            scope,
            phx.meshing.VolumeFillStrategy.SIMPLEX,
            size_controls=size,
            region_controls=(
                phx.meshing.RegionControl(
                    _scope(3, (10,), revision="foreign"),
                    "foreign",
                    "material-foreign",
                    phx.meshing.RegionRole.SOLID,
                ),
            ),
        )
    with pytest.raises(ValueError, match="Patch control scopes must be disjoint"):
        phx.meshing.VolumeMeshingSpec(
            target,
            scope,
            phx.meshing.VolumeFillStrategy.SIMPLEX,
            size_controls=size,
            region_controls=regions,
            patch_controls=(
                phx.meshing.PatchControl("first-patch", _scope(2, (30,)), ("first",)),
                phx.meshing.PatchControl(
                    "second-patch", _scope(2, (30, 40)), ("second",)
                ),
            ),
        )


def test_provider_support_report_fails_before_execution():
    scope = _scope()
    target = phx.meshing.CellMeshingTarget(
        2,
        3,
        phx.meshing.CellFamilyPolicy(required=("triangle",)),
    )
    specification = phx.meshing.SurfaceMeshingSpec(
        target,
        scope,
        size_controls=(phx.meshing.UniformSizeControl(scope, 0.1),),
    )
    provider = phx.meshing.MeshingProviderInfo(
        "unit-provider",
        "1",
        "MIT",
        operations=(phx.meshing.MeshingOperation.MESH_SURFACE,),
        source_kinds=(phx.meshing.MeshingSourceKind.BREP,),
        capabilities=(),
        cell_kinds=("triangle",),
        dimensions=(2,),
        execution_modes=(phx.meshing.MeshingExecutionMode.IN_PROCESS,),
    )
    source = phx.meshing.MeshingSourceDescriptor(
        "shape",
        "revision-1",
        phx.meshing.MeshingSourceKind.BREP,
        2,
        3,
        closed=False,
    )
    report = phx.meshing.ProviderSupportReport(
        provider,
        source,
        specification,
        unsupported=("periodic surface meshing",),
    )

    with pytest.raises(phx.meshing.MeshingFailure) as captured:
        report.require_supported()
    assert (
        captured.value.category
        is phx.meshing.MeshingFailureCategory.UNSUPPORTED_COMBINATION
    )


def test_layer_schedule_geometric_constructor_has_explicit_schedule_identity():
    explicit = phx.meshing.LayerSchedule((0.125, 0.25, 0.5))
    geometric = phx.meshing.LayerSchedule.geometric(
        3,
        0.125,
        growth_rate=2.0,
    )
    shrinking = phx.meshing.LayerSchedule.geometric(
        3,
        0.4,
        growth_rate=0.5,
    )

    assert geometric.thicknesses == explicit.thicknesses
    assert geometric.schedule_id == explicit.schedule_id
    assert geometric.layer_count == 3
    assert geometric.total_thickness == pytest.approx(0.875)
    assert shrinking.thicknesses == (0.4, 0.2, 0.1)


@pytest.mark.parametrize(
    "thicknesses",
    (
        (),
        (0.1, 0.0),
        (0.1, -0.2),
        (0.1, np.inf),
        (0.1, np.nan),
        ("0.1", "0.2"),
    ),
)
def test_layer_schedule_rejects_non_positive_non_finite_or_non_numeric_values(
    thicknesses,
):
    with pytest.raises((TypeError, ValueError)):
        phx.meshing.LayerSchedule(thicknesses)


def test_geometric_layer_schedule_rejects_non_integral_count_and_invalid_parameters():
    with pytest.raises(TypeError, match="integer"):
        phx.meshing.LayerSchedule.geometric(2.5, 0.1)
    with pytest.raises(ValueError, match="positive"):
        phx.meshing.LayerSchedule.geometric(0, 0.1)
    with pytest.raises(ValueError, match="positive and finite"):
        phx.meshing.LayerSchedule.geometric(2, 0.0)
    with pytest.raises(ValueError, match="positive and finite"):
        phx.meshing.LayerSchedule.geometric(2, 0.1, growth_rate=0.0)


def test_swept_layer_control_is_face_to_face_and_volume_bound():
    source = _scope(2, (10,))
    target = _scope(2, (20,))
    volume_scope = _scope(3, (30,))
    schedule = phx.meshing.LayerSchedule((0.1, 0.15, 0.2))
    control = phx.meshing.SweptLayerControl(
        source,
        target,
        volume_scope,
        schedule,
    )
    equivalent = phx.meshing.SweptLayerControl(
        source,
        target,
        volume_scope,
        phx.meshing.LayerSchedule((0.1, 0.15, 0.2)),
    )
    target_spec = phx.meshing.CellMeshingTarget(
        3,
        3,
        phx.meshing.CellFamilyPolicy(
            required=("prism", "tetrahedron"),
            allow_mixed=True,
        ),
    )
    specification = phx.meshing.VolumeMeshingSpec(
        target_spec,
        volume_scope,
        phx.meshing.VolumeFillStrategy.SWEEP,
        size_controls=(phx.meshing.UniformSizeControl(volume_scope, 0.2),),
        layer_controls=(control,),
    )

    assert control.schedule is schedule
    assert control.termination is phx.meshing.LayerTerminationPolicy.REJECT
    assert control.control_id == equivalent.control_id
    assert specification.layer_controls == (control,)


def test_swept_layer_control_rejects_wrong_dimensions_binding_and_termination():
    source = _scope(2, (10,))
    target = _scope(2, (20,))
    volume_scope = _scope(3, (30,))
    schedule = phx.meshing.LayerSchedule((0.1,))

    with pytest.raises(ValueError, match="face scopes"):
        phx.meshing.SweptLayerControl(
            _scope(1, (10,)),
            target,
            volume_scope,
            schedule,
        )
    with pytest.raises(ValueError, match="source binding"):
        phx.meshing.SweptLayerControl(
            source,
            _scope(2, (20,), revision="foreign"),
            volume_scope,
            schedule,
        )
    with pytest.raises(ValueError, match="disjoint"):
        phx.meshing.SweptLayerControl(
            source,
            source,
            volume_scope,
            schedule,
        )
    with pytest.raises(ValueError, match="only REJECT"):
        phx.meshing.SweptLayerControl(
            source,
            target,
            volume_scope,
            schedule,
            termination=phx.meshing.LayerTerminationPolicy.COLLAPSE,
        )


def test_volume_spec_rejects_swept_layer_outside_top_level_volume_scope():
    boundary_scope = _scope(3, (30,))
    target_spec = phx.meshing.CellMeshingTarget(
        3,
        3,
        phx.meshing.CellFamilyPolicy(
            required=("prism", "tetrahedron"),
            allow_mixed=True,
        ),
    )
    control = phx.meshing.SweptLayerControl(
        _scope(2, (10,)),
        _scope(2, (20,)),
        _scope(3, (40,)),
        phx.meshing.LayerSchedule((0.1,)),
    )

    with pytest.raises(ValueError, match="contained"):
        phx.meshing.VolumeMeshingSpec(
            target_spec,
            boundary_scope,
            phx.meshing.VolumeFillStrategy.SWEEP,
            size_controls=(phx.meshing.UniformSizeControl(boundary_scope, 0.2),),
            layer_controls=(control,),
        )
