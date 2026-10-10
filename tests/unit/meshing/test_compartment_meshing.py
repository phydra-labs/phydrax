from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

import phydrax as phx
from phydrax.meshing._compartments import revalidate_region_evidence
from tests._support.image_sources import _execute, _region_measure_array, _source


def test_native_tiny_material_retains_exact_volume_interface_and_identity() -> None:
    values = np.ones((3, 3, 3), dtype=np.int16)
    values[1, 1, 1] = 2
    source = _source(values)
    result = _execute(source)
    evidence = result.region_evidence
    assert evidence is not None
    evidence.require_source(source.compartments)
    evidence.require_current(
        result.mesh, result.zones, result.patches, geometry=result.geometry
    )
    assert evidence.adjacency_pairs == (("region:1", "region:2"),)
    measured = dict(
        zip(
            evidence.coverage.region_ids,
            _region_measure_array(evidence).tolist(),
            strict=True,
        )
    )
    np.testing.assert_allclose(measured["region:2"], 1.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(measured["region:1"], 26.0, rtol=0.0, atol=0.0)
    assert {row[0] for row in evidence.interface_facets} == {
        "interface:region:1:region:2"
    }


def test_native_triple_junction_has_one_shared_complex_and_three_adjacencies() -> None:
    values = np.asarray((((1,), (2,)), ((3,), (3,))), dtype=np.int16)
    source = _source(values)
    result = _execute(source)
    evidence = result.region_evidence
    assert evidence is not None
    assert evidence.adjacency_pairs == (
        ("region:1", "region:2"),
        ("region:1", "region:3"),
        ("region:2", "region:3"),
    )
    evidence.require_source(source.compartments)
    face_ids = np.asarray(result.mesh.entity_set(2).entity_ids, dtype=np.int64)
    face_rows = {value: row for row, value in enumerate(face_ids.tolist())}
    connectivity = result.mesh.connectivity
    if not isinstance(connectivity, phx.discretization.TetrahedralConnectivity):
        raise TypeError("The compartment fixture must have tetrahedral incidence.")
    faces = np.asarray(connectivity.faces, dtype=np.int64)
    region_vertices: dict[str, set[int]] = {}
    for name, facet, _, _, _ in evidence.interface_facets:
        region_vertices.setdefault(name, set()).update(faces[face_rows[facet]].tolist())
    common = set.intersection(*region_vertices.values())
    coordinates = np.asarray(result.mesh.coordinates)[sorted(common)]
    np.testing.assert_allclose(coordinates[:, :2], 0.5, rtol=0.0, atol=0.0)
    assert np.ptp(coordinates[:, 2]) == 1.0


def test_source_rejects_unannounced_image_interpretation_and_stale_complex() -> None:
    source = _source(np.ones((1, 1, 1), dtype=np.int16))
    with pytest.raises((ValueError, TypeError)):
        replace(source, interpretation="categorical-samples")
    changed = replace(source.compartments, source_revision="stale")
    with pytest.raises(ValueError, match="revisions"):
        replace(source, compartments=changed)


def test_compartment_renewal_recertifies_fixed_topology_and_refuses_stale_motion() -> (
    None
):
    source = _source(np.ones((2, 2, 2), dtype=np.int16))
    result = _execute(source)
    renewal = revalidate_region_evidence(
        result, result.mesh, result.geometry, result.zones, result.patches
    )
    renewal.region_evidence.require_source(source.compartments)
    coordinates = np.asarray(result.mesh.coordinates) + np.asarray(
        (0.125, 0.0, 0.0), dtype=np.float64
    )
    moved = result.mesh.with_coordinates(coordinates, numeric_version="translated-image")
    assert result.region_evidence is not None
    with pytest.raises(ValueError, match="stale"):
        result.region_evidence.require_current(moved, result.zones, result.patches)
    with pytest.raises(phx.meshing.MeshingFailure, match="coverage"):
        revalidate_region_evidence(
            result, moved, phx.discretization.CellGeometrySpec.affine(moved), (), ()
        )


def test_material_refinement_and_coarsening_renew_exact_region_evidence() -> None:
    source = _source(np.asarray((((1,),), ((2,),)), dtype=np.int16))
    result = _execute(source)
    assert result.region_evidence is not None
    from phydrax.meshing._compartments import prepare_compartment_complex
    from phydrax.meshing._volume_generation import prepare_plc_source

    prepared_source = prepare_plc_source(
        prepare_compartment_complex(source),
        source.source_id,
        source.source_revision,
        source.coordinate_contract,
        limits=phx.meshing.MeshingLimits(),
    )
    policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION,
        association_transfer=prepared_source.association_transfer,
        compatibility=phx.meshing.BisectionCompatibility.UNIFORM_REFINEMENT,
        audit_policy=phx.meshing.CellMeshAuditPolicy(
            watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT,
            require_complete_association=True,
        ),
    )
    refined = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            result,
            phx.meshing.MarkedMeshAdaptation(
                np.asarray(result.mesh.entity_set(3).entity_ids, dtype=np.int64)
            ),
            policy=policy,
        )
    )
    target = refined.target
    assert target.region_evidence is not None
    target.region_evidence.require_source(source.compartments)
    target.region_evidence.require_current(
        target.mesh, target.zones, target.patches, geometry=target.geometry
    )
    assert target.region_evidence.adjacency_pairs == (("region:1", "region:2"),)
    np.testing.assert_allclose(
        _region_measure_array(target.region_evidence),
        np.asarray((1.0, 1.0), dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
    from phydrax.discretization._cell_geometry import (
        PolynomialComposedCellGeometryElement,
    )

    elements, _, _ = target.geometry.resolve(target.mesh)
    exact_third = False
    for element in elements:
        if not isinstance(element, PolynomialComposedCellGeometryElement):
            raise TypeError(
                "Uniform compartment refinement requires exact rational charts."
            )
        for row in element.chart_coefficients:
            references = tuple(
                Fraction(numerator, denominator) for numerator, denominator in row
            )
            barycentric = (1 - sum(references, Fraction(0)), *references)
            assert sum(barycentric, Fraction(0)) == 1
            assert min(barycentric) >= 0
            exact_third |= any(value.denominator == 3 for value in barycentric)
    assert exact_third
    assert target.region_evidence.coverage.findings == ()
    assert (
        target.region_evidence.coverage.covered_source_facet_count
        == target.region_evidence.coverage.source_facet_count
    )
    restored = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            target,
            phx.meshing.MarkedMeshAdaptation(
                np.empty((0,), dtype=np.int64),
                np.asarray(target.mesh.entity_set(3).entity_ids, dtype=np.int64),
                hierarchy=refined.hierarchy,
            ),
            policy=policy,
        )
    ).target
    assert restored.region_evidence is not None
    restored.region_evidence.require_source(source.compartments)
    restored.region_evidence.require_current(
        restored.mesh,
        restored.zones,
        restored.patches,
        geometry=restored.geometry,
    )
    assert restored.region_evidence.adjacency_pairs == (("region:1", "region:2"),)
    np.testing.assert_allclose(
        _region_measure_array(restored.region_evidence),
        np.asarray((1.0, 1.0), dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert restored.region_evidence.coverage.findings == ()
    assert (
        restored.region_evidence.coverage.covered_source_facet_count
        == restored.region_evidence.coverage.source_facet_count
    )


def test_material_controls_bind_physical_roles_and_renamed_interface_patch() -> None:
    source = _source(np.asarray((((1,),), ((2,),)), dtype=np.int16))

    def scope(dimension: int, entity_set: str, index: int) -> phx.meshing.MeshingScope:
        return phx.meshing.MeshingScope(
            source.source_id,
            source.source_revision,
            phx.meshing.MeshingEntityKind.GEOMETRY,
            dimension,
            f"{source.source_id}:{entity_set}",
            np.asarray((index,), dtype=np.int64),
        )

    boundary = scope(2, "boundary", 0)
    interface = scope(2, "interfaces", 0)
    request = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3, 3, phx.meshing.CellFamilyPolicy(required=("tetrahedron",))
        ),
        boundary,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                boundary, 3.0, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
        region_controls=(
            phx.meshing.RegionControl(
                scope(3, "regions", 0),
                "region:1",
                "tissue",
                phx.meshing.RegionRole.POROUS,
            ),
            phx.meshing.RegionControl(
                scope(3, "regions", 1), "region:2", "csf", phx.meshing.RegionRole.FLUID
            ),
        ),
        patch_controls=(
            phx.meshing.PatchControl(
                "exchange-facet", interface, ("region:1", "region:2")
            ),
        ),
        protected_features=(
            phx.meshing.ProtectedFeature(
                interface, phx.meshing.FeatureKind.MATERIAL_INTERFACE
            ),
        ),
    )
    result = (
        phx.meshing.NativeMeshingProvider(
            phx.meshing.NativeMeshingOptions("image_material_tetrahedral")
        )
        .plan(source, request, coordinate_contract=source.coordinate_contract)
        .execute()
    )
    evidence = result.region_evidence
    assert evidence is not None
    zone_by_id = {zone.zone_id: zone for zone in result.zones}
    roles = {
        region: (zone_by_id[zone_id].material_id, zone_by_id[zone_id].region_role)
        for region, zone_id in evidence.region_zone_ids
    }
    assert roles == {
        "region:1": ("tissue", phx.meshing.RegionRole.POROUS),
        "region:2": ("csf", phx.meshing.RegionRole.FLUID),
    }
    interface_patch = next(
        patch
        for patch in result.patches
        if patch.patch_id == evidence.interface_patch_ids[0][1]
    )
    assert interface_patch.name == "exchange-facet"
    renewal = revalidate_region_evidence(
        result, result.mesh, result.geometry, result.zones, result.patches
    )
    assert renewal.region_evidence.interface_patch_ids == evidence.interface_patch_ids
    assert renewal.region_evidence.region_zone_ids == evidence.region_zone_ids


def test_optional_compartment_interface_requires_declared_endpoints() -> None:
    compartments = (
        phx.geometry.CompartmentDefinition("inside", ("inside",), "material"),
        phx.geometry.CompartmentDefinition("outside", ("outside",), "material"),
    )
    interface = phx.geometry.CompartmentInterfaceDefinition(
        "invalid-optional", "inside", "undeclared", "diagnostic", required=False
    )
    with pytest.raises(
        ValueError, match="Interface endpoints must identify declared compartments"
    ):
        phx.geometry.CompartmentComplex(
            "revision",
            compartments,
            (interface,),
            (("inside", 1.0), ("outside", 1.0)),
            (),
        )
