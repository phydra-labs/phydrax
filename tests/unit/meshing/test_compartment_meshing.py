from copy import copy
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import NoReturn

import numpy as np
import pytest

import phydrax as phx
from phydrax.geometry._compartments import CompartmentMeshingSource
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
        raise TypeError(
            "fixture.connectivity must have tetrahedral incidence; got incompatible connectivity"
        )
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
    with pytest.raises(
        ValueError, match=r"CompartmentMeshingSource\.compartments\.source_revision"
    ):
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
    from phydrax.meshing._volume_generation import (
        native_volume_execution_budget,
        prepare_plc_source,
    )

    limits = phx.meshing.MeshingLimits()
    with native_volume_execution_budget(limits) as scope:
        prepared_source = prepare_plc_source(
            prepare_compartment_complex(source, limits=limits, scope=scope),
            source.source_id,
            source.source_revision,
            source.coordinate_contract,
            limits=limits,
            scope=scope,
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
                "fixture.element must contain rational charts; got incompatible element"
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
    with pytest.raises(ValueError, match="CompartmentComplex.interfaces"):
        phx.geometry.CompartmentComplex(
            "revision",
            compartments,
            (interface,),
            (("inside", 1.0), ("outside", 1.0)),
            (),
        )


def _certified_cell_request(
    source: CompartmentMeshingSource,
    target: float = 1.0,
    *,
    strength: phx.meshing.SizeControlStrength | None = None,
    protected: tuple[phx.meshing.ProtectedFeature, ...] = (),
) -> phx.meshing.VolumeMeshingSpec:
    from tests._support.image_sources import image_specification

    base = image_specification(source)
    return phx.meshing.VolumeMeshingSpec(
        base.target,
        base.boundary_scope,
        base.fill_strategy,
        size_controls=(
            phx.meshing.UniformSizeControl(
                base.boundary_scope,
                target,
                strength=phx.meshing.SizeControlStrength.HARD
                if strength is None
                else strength,
            ),
        ),
        protected_features=protected,
    )


def _independent_affine_source_edges(
    coordinates: tuple[tuple[Fraction, ...], ...], cells: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.asarray(
        sorted(
            {
                tuple(sorted(edge))
                for cell in cells.tolist()
                for edge in combinations(cell, 2)
            }
        ),
        dtype=np.int64,
    )
    lengths = []
    # Fixture squared lengths have fewer than 128 dyadic coefficient bits;
    # 100 decimal digits separate their binary64 rounding midpoints.
    with localcontext() as context:
        context.prec = 100
        for first, last in edges:
            squared = sum(
                (
                    (coordinates[last][axis] - coordinates[first][axis]) ** 2
                    for axis in range(3)
                ),
                Fraction(0),
            )
            lengths.append(
                float((Decimal(squared.numerator) / Decimal(squared.denominator)).sqrt())
            )
    return edges, np.asarray(lengths, dtype=np.float64)


@pytest.mark.parametrize(
    ("linear", "translation", "scale"),
    (
        (np.eye(3), (0.0, 0.0, 0.0), 1.0),
        (np.eye(3), (4.0, -2.0, 8.0), 0.5),
        (np.asarray(((0, -1, 0), (0, 0, 1), (1, 0, 0))), (-4.0, 2.0, 1.0), 2.0),
    ),
    ids=("authored-unit", "translated-dyadic", "signed-permuted-dyadic"),
)
def test_certified_unit_cells_finish_against_unchanged_original_source(
    linear: np.ndarray, translation: tuple[float, float, float], scale: float
) -> None:
    from phydrax._meshcore import plc_source_constraints
    from phydrax.meshing._compartments import _prepare_compartment_complex_bound
    from phydrax.meshing._volume_generation import (
        generate_plc_volume,
        native_volume_coordinate_budget,
        NativeVolumeSchedule,
    )
    from tests._support.coordinate_references import native_meshing_request

    matrix = np.eye(4)
    matrix[:3, :3] = scale * linear
    matrix[:3, 3] = translation
    values = np.ones((2, 2, 2), dtype=np.int16)
    values[1] = 2
    source = _source(values, matrix)
    request = _certified_cell_request(source, scale)
    with native_meshing_request(request.limits) as (scope, _, _):
        plc, budget = _prepare_compartment_complex_bound(
            source, request.limits, None, scope=scope
        )
        certificate = budget.certified_lattice
        assert certificate is not None
        original = plc_source_constraints(
            plc.vertices,
            plc.polygon_offsets,
            plc.polygon_vertices,
            plc.polygon_facets,
            plc.facet_regions,
            segments=plc.segments,
            work_limit=request.limits.maximum_work_units,
        )
        before = plc.vertices.copy()
        construction = generate_plc_volume(
            plc,
            request,
            NativeVolumeSchedule(),
            validity_policy=phx.discretization.CellValidityPolicy(),
            source_id=source.source_id,
            source_revision=source.source_revision,
            input_id=source.source_id,
            certified_lattice=certificate,
            source_work_units=budget.work_units,
            scope=scope,
        )
        geometry = construction.geometry
        assert geometry is not None
        with native_volume_coordinate_budget(request.limits, scope=scope) as ledger:
            source_coordinates = geometry.rational_source_coordinates(budget=ledger)
            execution_error = geometry.source_execution_error(
                construction.mesh, budget=ledger
            )
    counters = dict(construction.construction_counters)
    assert counters["certified-unit-cell-complex"] == 1
    assert counters["certified_unit_cell_count"] == 8
    assert counters["certified_unit_cell_orientation_predicates"] == 256
    assert counters["initial_goal_shape_certified"] == 1
    assert construction.refinement is None and construction.improvement is None
    assert construction.metric_optimization is not None
    assert construction.metric_optimization.converged
    assert construction.mesh.coordinates.shape[0] == 71
    assert construction.mesh.blocks[0].vertices.shape[0] == 256
    fidelity = construction.source_fidelity
    assert fidelity is not None and fidelity.declared_source is not None
    declared = fidelity.declared_source
    np.testing.assert_array_equal(plc.vertices, before)
    np.testing.assert_array_equal(declared.points, before)
    np.testing.assert_array_equal(declared.faces, original.input_triangles)
    np.testing.assert_array_equal(declared.segments, original.plc_edges)
    np.testing.assert_array_equal(
        declared.face_ids[declared.face_group_rows],
        plc.polygon_facets[original.input_polygons],
    )
    assert fidelity.achieved_bound == 0.0
    assert fidelity.refused_carriers == 0
    cells = np.asarray(construction.mesh.blocks[0].vertices)
    edges, lengths = _independent_affine_source_edges(source_coordinates, cells)
    np.testing.assert_array_equal(np.quantile(lengths, (0.5, 0.95)), (scale, scale))
    metric = construction.metric_optimization
    assert metric.metric_arc_bounds is not None
    assert metric.metric_arc_edge_ids is not None
    np.testing.assert_array_equal(
        metric.metric_arc_edge_ids,
        np.asarray(construction.mesh.vertex_global_ids)[edges],
    )
    bounds = np.asarray(metric.metric_arc_bounds)
    assert np.all(bounds[:, 0] <= lengths / scale)
    assert np.all(bounds[:, 1] >= lengths / scale)
    if any(translation):
        assert execution_error > 0.0
    else:
        assert execution_error == 0.0
    assert set(construction.cell_regions.tolist()) == {0, 1}


@pytest.mark.parametrize(
    "linear",
    (
        np.diag((1.0, 2.0, 1.0)),
        np.diag((1.5, 1.5, 1.5)),
        np.asarray(((1.0, 0.25, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    ),
    ids=("anisotropic", "non-dyadic-scale", "sheared"),
)
def test_ineligible_affine_does_not_acquire_source_lattice_certificate(
    linear: np.ndarray,
) -> None:
    from phydrax.meshing._compartments import _prepare_compartment_complex_bound

    matrix = np.eye(4)
    matrix[:3, :3] = linear
    source = _source(np.ones((1, 1, 1), dtype=np.int16), matrix)
    plc, budget = _prepare_compartment_complex_bound(
        source, phx.meshing.MeshingLimits(), None, scope=None
    )
    assert budget.certified_lattice is None
    assert plc.vertices.shape[0] == 8


@pytest.mark.parametrize(
    "profile", ("soft", "wrong-target", "positive-feature-bound", "no-certificate")
)
def test_ineligible_cell_profile_preserves_general_recovery_route(
    profile: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import phydrax.meshing._volume_generation as generation
    from phydrax.meshing._compartments import _prepare_compartment_complex_bound
    from tests._support.coordinate_references import native_meshing_request

    source = _source(np.ones((1, 1, 1), dtype=np.int16))
    request = _certified_cell_request(source)
    if profile == "soft":
        request = _certified_cell_request(
            source, strength=phx.meshing.SizeControlStrength.SOFT
        )
    elif profile == "wrong-target":
        request = _certified_cell_request(source, 2.0)
    elif profile == "positive-feature-bound":
        request = _certified_cell_request(
            source,
            protected=(
                phx.meshing.ProtectedFeature(
                    request.boundary_scope,
                    phx.meshing.FeatureKind.SURFACE,
                    maximum_deviation=0.01,
                ),
            ),
        )
    plc, budget = _prepare_compartment_complex_bound(
        source, request.limits, None, scope=None
    )

    class GeneralRecoveryReached(Exception):
        pass

    def recover(*args: object, **kwargs: object) -> NoReturn:
        raise GeneralRecoveryReached

    monkeypatch.setattr(generation, "recover", recover)
    with native_meshing_request(request.limits) as (scope, _, _):
        with pytest.raises(GeneralRecoveryReached):
            generation._construct_plc_volume(
                plc,
                request,
                generation.NativeVolumeSchedule(),
                validity_policy=phx.discretization.CellValidityPolicy(),
                source_id=source.source_id,
                source_revision=source.source_revision,
                input_id=source.source_id,
                certified_lattice=None
                if profile == "no-certificate"
                else budget.certified_lattice,
                scope=scope,
            )


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("source", "source"),
        ("gap", "boundary"),
        ("overlap", "overlap"),
        ("material", "material"),
    ),
)
def test_certified_cell_identity_refuses_wrong_source_gap_overlap_or_material(
    corruption: str, message: str
) -> None:
    from phydrax._meshcore import plc_source_constraints
    from phydrax.meshing._certified_unit_cells import construct_unit_cells
    from phydrax.meshing._compartments import _prepare_compartment_complex_bound
    from tests._support.coordinate_references import native_meshing_request

    values = np.ones((2, 1, 1), dtype=np.int16)
    values[1] = 2
    source = _source(values)
    limits = phx.meshing.MeshingLimits()
    plc, budget = _prepare_compartment_complex_bound(source, limits, None, scope=None)
    certificate = budget.certified_lattice
    assert certificate is not None
    if corruption == "source":
        certificate = replace(certificate, complex_id="wrong-original-source")
    elif corruption == "gap":
        certificate = replace(certificate, cells=certificate.cells[:-1])
    elif corruption == "overlap":
        certificate = replace(
            certificate, cells=certificate.cells + certificate.cells[:1]
        )
    else:
        first = certificate.cells[0]
        certificate = replace(
            certificate, cells=((*first[:3], 7), *certificate.cells[1:])
        )
    with native_meshing_request(limits) as (scope, _, _):
        constraints = plc_source_constraints(
            plc.vertices,
            plc.polygon_offsets,
            plc.polygon_vertices,
            plc.polygon_facets,
            plc.facet_regions,
            segments=plc.segments,
            work_limit=limits.maximum_work_units,
        )
        with pytest.raises(ValueError, match=message):
            construct_unit_cells(certificate, plc, constraints)


@pytest.mark.parametrize(
    ("linear", "translation", "scale"),
    (
        (np.eye(3), (0.0, 0.0, 0.0), 1.0),
        (np.eye(3), (4.0, -2.0, 8.0), 0.5),
        (np.asarray(((0, -1, 0), (0, 0, 1), (1, 0, 0))), (-4.0, 2.0, 1.0), 2.0),
    ),
    ids=("authored-unit", "translated-dyadic", "signed-permuted-dyadic"),
)
def test_certified_unit_cell_publication_keeps_exact_material_and_outer_coverage(
    linear: np.ndarray, translation: tuple[float, float, float], scale: float
) -> None:
    from phydrax._model._structure import (
        model_from_array_recipe,
        model_recipe_array_values,
        model_structure_recipe,
    )
    from phydrax.discretization._cell_geometry import (
        cell_geometry_preparation_bounds,
        owned_coordinate_preparation,
    )
    from phydrax.lifecycle._meshing_sources import (
        register_meshing_source_artifacts,
        validate_meshing_source_closure,
    )
    from phydrax.meshing.qualification import declared_geometry_edge_lengths

    values = np.ones((2, 2, 2), dtype=np.int16)
    values[1] = 2
    matrix = np.eye(4)
    matrix[:3, :3] = scale * linear
    matrix[:3, 3] = translation
    source = _source(values, matrix)
    result = (
        phx.meshing.NativeMeshingProvider(
            phx.meshing.NativeMeshingOptions("image_material_tetrahedral")
        )
        .plan(
            source,
            _certified_cell_request(source, scale),
            coordinate_contract=source.coordinate_contract,
        )
        .execute()
    )
    assert result.audit.passed
    assert result.geometry.exact_source is not None
    evidence = result.region_evidence
    assert evidence is not None
    evidence.require_source(source.compartments)
    evidence.require_current(
        result.mesh, result.zones, result.patches, geometry=result.geometry
    )
    assert evidence.coverage.findings == ()
    assert evidence.adjacency_pairs == (("region:1", "region:2"),)
    np.testing.assert_allclose(
        _region_measure_array(evidence),
        (4.0 * scale**3, 4.0 * scale**3),
        rtol=0.0,
        atol=0.0,
    )
    register_meshing_source_artifacts()
    recipe = model_structure_recipe((result.mesh, result.geometry))
    arrays = model_recipe_array_values(
        (result.mesh, result.geometry), recipe, prefix="unit-cell-source"
    )
    restored_mesh, restored_geometry = model_from_array_recipe(
        recipe, arrays, prefix="unit-cell-source"
    )
    validate_meshing_source_closure(restored_geometry)
    assert restored_geometry.exact_source is not None
    assert (
        restored_geometry.exact_source.source_id == result.geometry.exact_source.source_id
    )
    assert restored_geometry.geometry_layout_id == result.geometry.geometry_layout_id
    np.testing.assert_array_equal(
        restored_geometry.coordinates, result.geometry.coordinates
    )
    work, memory = cell_geometry_preparation_bounds(result.geometry)
    with owned_coordinate_preparation(2 * work, 2 * memory) as budget:
        source_coordinates = result.geometry.rational_source_coordinates(budget=budget)
        assert restored_geometry.source_coordinates(budget=budget) == source_coordinates
        evidence.require_current(
            restored_mesh,
            result.zones,
            result.patches,
            geometry=restored_geometry,
        )
    edges, reference_lengths = _independent_affine_source_edges(
        source_coordinates, np.asarray(result.mesh.blocks[0].vertices)
    )
    oracle_lengths = declared_geometry_edge_lengths(
        result.mesh, result.geometry, edges, failure_factory=RuntimeError
    )
    np.testing.assert_array_equal(oracle_lengths, reference_lengths)
    np.testing.assert_array_equal(
        np.quantile(oracle_lengths, (0.5, 0.95)), (scale, scale)
    )
    world_geometry = phx.discretization.CellGeometrySpec.affine(result.mesh)
    np.testing.assert_array_equal(
        declared_geometry_edge_lengths(
            result.mesh, world_geometry, edges, failure_factory=RuntimeError
        ),
        np.linalg.norm(
            np.asarray(result.mesh.coordinates)[edges[:, 1]]
            - np.asarray(result.mesh.coordinates)[edges[:, 0]],
            axis=1,
        ),
    )


@pytest.mark.parametrize("control_count", (0, 2))
def test_certified_unit_cells_require_exactly_one_global_hard_control(
    control_count: int,
) -> None:
    from phydrax.meshing._compartments import _prepare_compartment_complex_bound
    from phydrax.meshing._volume_generation import (
        _certified_unit_cell_eligible,
        NativeVolumeSchedule,
    )

    source = _source(np.ones((1, 1, 1), dtype=np.int16))
    base = _certified_cell_request(source)
    if control_count == 0:
        # The public source contract owns the empty-profile refusal.
        with pytest.raises(ValueError, match="size_controls"):
            phx.meshing.VolumeMeshingSpec(
                base.target,
                base.boundary_scope,
                base.fill_strategy,
                size_controls=(),
            )
        return
    request = phx.meshing.VolumeMeshingSpec(
        base.target,
        base.boundary_scope,
        base.fill_strategy,
        size_controls=base.size_controls * control_count,
    )
    plc, budget = _prepare_compartment_complex_bound(
        source, request.limits, None, scope=None
    )
    assert not _certified_unit_cell_eligible(
        plc,
        request,
        NativeVolumeSchedule(),
        phx.discretization.CellValidityPolicy(),
        budget.certified_lattice,
    )


def test_certified_unit_cells_retain_protected_feature_source_id_refusal() -> None:
    from phydrax.meshing._compartments import _prepare_compartment_complex_bound
    from phydrax.meshing._volume_generation import (
        generate_plc_volume,
        NativeVolumeSchedule,
    )
    from tests._support.coordinate_references import native_meshing_request

    source = _source(np.ones((1, 1, 1), dtype=np.int16))
    base = _certified_cell_request(source)
    plc, budget = _prepare_compartment_complex_bound(
        source, base.limits, None, scope=None
    )
    scope = phx.meshing.MeshingScope(
        source.source_id,
        source.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        base.boundary_scope.entity_set_id,
        np.asarray((plc.facet_count,), dtype=np.int64),
    )
    request = _certified_cell_request(
        source,
        protected=(phx.meshing.ProtectedFeature(scope, phx.meshing.FeatureKind.SURFACE),),
    )
    before = plc.vertices.copy()
    with native_meshing_request(request.limits) as (execution, _, _):
        with pytest.raises(ValueError, match="protected_features"):
            generate_plc_volume(
                plc,
                request,
                NativeVolumeSchedule(),
                validity_policy=phx.discretization.CellValidityPolicy(),
                source_id=source.source_id,
                source_revision=source.source_revision,
                input_id=source.source_id,
                certified_lattice=budget.certified_lattice,
                scope=execution,
            )
    np.testing.assert_array_equal(plc.vertices, before)


@pytest.mark.parametrize("resource", ("vertices", "cells"))
def test_certified_unit_cell_construction_refuses_one_short_capacity(
    resource: str,
) -> None:
    from phydrax.meshing._compartments import _prepare_compartment_complex_bound
    from phydrax.meshing._volume_generation import (
        generate_plc_volume,
        NativeVolumeSchedule,
    )
    from tests._support.coordinate_references import native_meshing_request

    values = np.ones((2, 2, 2), dtype=np.int16)
    values[1] = 2
    source = _source(values)
    base = _certified_cell_request(source)
    limits = (
        phx.meshing.MeshingLimits(maximum_vertices=70)
        if resource == "vertices"
        else phx.meshing.MeshingLimits(maximum_cells=255)
    )
    request = phx.meshing.VolumeMeshingSpec(
        base.target,
        base.boundary_scope,
        base.fill_strategy,
        size_controls=base.size_controls,
        limits=limits,
    )
    with native_meshing_request(limits) as (scope, _, _):
        plc, budget = _prepare_compartment_complex_bound(
            source, limits, None, scope=scope
        )
        before = plc.vertices.copy()
        with pytest.raises(phx.meshing.MeshingFailure) as refusal:
            generate_plc_volume(
                plc,
                request,
                NativeVolumeSchedule(),
                validity_policy=phx.discretization.CellValidityPolicy(),
                source_id=source.source_id,
                source_revision=source.source_revision,
                input_id=source.source_id,
                certified_lattice=budget.certified_lattice,
                source_work_units=budget.work_units,
                scope=scope,
            )
        assert (
            refusal.value.evidence.category
            is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
        )
        np.testing.assert_array_equal(plc.vertices, before)


@pytest.mark.parametrize("defect", ("wrong-geometry", "curved-map"))
def test_independent_size_oracle_refuses_wrong_or_unsupported_coordinate_map(
    defect: str,
) -> None:
    from phydrax.discretization._cell_geometry import coordinate_lagrange_element
    from phydrax.discretization._exact_rational_geometry import (
        ExactRationalCellGeometrySource,
        rational_source_layouts,
    )
    from phydrax.meshing._curving import straight_geometry
    from phydrax.meshing.qualification import declared_geometry_edge_lengths

    points = np.asarray(
        ((4, -2, 8), (4.5, -2, 8), (4, -1.5, 8), (4, -2, 8.5)),
        dtype=np.float64,
    )
    cells = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    mesh = phx.discretization.CellMesh.from_tetrahedra(points, cells)
    parent = phx.discretization.CellGeometrySpec.affine(mesh)
    if defect == "wrong-geometry":
        name = mesh.blocks[0].name
        elements = {name: coordinate_lagrange_element("tetrahedron", 1)}
        routes = {name: cells[:, (1, 0, 2, 3)]}
        source = ExactRationalCellGeometrySource(
            parent,
            tuple(((index, 1, 1),) for index in range(4)),
            rational_source_layouts(elements, routes),
            maximum_work=100_000,
            maximum_bits=4096,
        )
        geometry = phx.discretization.CellGeometrySpec(
            elements, routes, points, exact_source=source
        )
        message = "different geometry"
    else:
        lifted = straight_geometry(mesh, 2, source_geometry=parent)
        coefficients = np.asarray(lifted.coordinates).copy()
        coefficients[-1, 2] += 0.125
        np.testing.assert_array_equal(coefficients[:4], points)
        geometry = phx.discretization.CellGeometrySpec(
            dict(zip(lifted.block_names, lifted.elements, strict=True)),
            dict(zip(lifted.block_names, lifted.geometry_dofs, strict=True)),
            coefficients,
        )
        message = "unsupported coordinate map"
    with pytest.raises(RuntimeError, match=message):
        declared_geometry_edge_lengths(
            mesh,
            geometry,
            np.asarray(((0, 1),), dtype=np.int64),
            failure_factory=RuntimeError,
        )


def test_hard_unit_cell_region_cold_renewal_retains_controls_and_patch_order(
    tmp_path: Path,
) -> None:
    from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        source_content_equal,
        write_meshing_source_closure,
    )

    values = np.ones((2, 2, 2), dtype=np.int16)
    values[1] = 2
    source = _source(values)
    request = _certified_cell_request(source)
    result = (
        phx.meshing.NativeMeshingProvider(
            phx.meshing.NativeMeshingOptions("image_material_tetrahedral")
        )
        .plan(source, request, coordinate_contract=source.coordinate_contract)
        .execute()
    )
    evidence = result.region_evidence
    assert evidence is not None and result.certification is not None
    material_limits = evidence.certificate_limits
    final_limits = result.certification.request.limits
    assert material_limits.maximum_work_units == 200_000_000
    assert material_limits.maximum_scratch_bytes == 536_870_912
    assert (
        final_limits.maximum_work_units
        == request.limits.maximum_work_units
        == 1_000_000_000
    )
    assert (
        final_limits.maximum_scratch_bytes
        == request.limits.maximum_scratch_bytes
        == 8_000_000_000
    )
    assert material_limits.limits_id == evidence.coverage.binding.limits_id
    assert material_limits.limits_id != final_limits.limits_id
    assert tuple(patch.patch_id for patch in result.patches) == tuple(
        sorted(patch.patch_id for patch in result.patches)
    )
    path = tmp_path / "hard-material-source.zip"
    receipt = write_meshing_source_closure(path, (evidence, result))
    restored, owner = read_meshing_source_closure(
        path, expected_content_id=receipt.content_id
    )
    assert source_content_equal(
        restored.certificate_limits,
        material_limits,
        limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
    )
    renewal = revalidate_region_evidence(
        owner,
        owner.mesh,
        owner.geometry,
        owner.zones,
        owner.patches,
        compartment_source=source,
        limits=request.limits,
    )
    assert source_content_equal(
        restored, renewal.region_evidence, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS
    )
    assert source_content_equal(
        owner.zones, renewal.zones, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS
    )
    assert source_content_equal(
        owner.patches, renewal.patches, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS
    )


@pytest.mark.parametrize("tamper", ("foreign", "stale-id"))
def test_region_evidence_refuses_foreign_or_tampered_material_controls(
    tamper: str,
) -> None:
    from phydrax.lifecycle._meshing_source_records import reconstruct_stored_fields
    from phydrax.lifecycle._meshing_sources import register_meshing_source_artifacts

    result = _execute(_source(np.asarray((((1,),), ((2,),)), dtype=np.int16)))
    evidence = result.region_evidence
    assert evidence is not None
    if tamper == "foreign":
        controls = replace(
            evidence.certificate_limits,
            maximum_work_units=evidence.certificate_limits.maximum_work_units - 1,
        )
    else:
        controls = copy(evidence.certificate_limits)
        object.__setattr__(
            controls,
            "maximum_work_units",
            controls.maximum_work_units - 1,
        )
    changed = copy(evidence)
    object.__setattr__(changed, "certificate_limits", controls)
    with pytest.raises(ValueError, match="certificate_limits"):
        changed.require_current(
            result.mesh, result.zones, result.patches, geometry=result.geometry
        )
    register_meshing_source_artifacts()
    with pytest.raises(ValueError, match="certificate_limits"):
        reconstruct_stored_fields(
            changed,
            bindings={"mesh": result.mesh, "geometry": result.geometry},
        )
