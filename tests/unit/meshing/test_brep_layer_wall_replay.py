from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import equinox as eqx
import numpy as np
import pytest
from jax.typing import ArrayLike

import phydrax as phx
from phydrax._meshcore import meshcore_available
from phydrax.geometry._meshing_domain import MeshingDomain
from phydrax.geometry.brep import brep_box
from phydrax.geometry.brep._projection import NativeBRepProjection
from phydrax.geometry.brep._projection_contracts import brep_entity_id
from phydrax.meshing._association import GeometryAssociation, GeometryAssociationKind
from phydrax.meshing._association_routes import target_dimension
from phydrax.meshing._layer_core_association_transfer import (
    NativeLayerCoreAssociationTransfer,
)
from phydrax.meshing._layer_core_brep import brep_layer_core_source, fixed_core
from phydrax.meshing._layer_core_controls import wall_layer_control
from phydrax.meshing._layer_wall_source_replay import validate_wall_source
from phydrax.meshing.providers._native_sources import NativeLayerCoreSource
from tests.unit.meshing.test_source_association_transfer import (
    _layer_association_semantics,
)
from tools.meshing_qualification.scenarios.native_hybrid_profiles import (
    _accepted_layer_columns,
    _face_vertex_rows,
    _periodic_mixed_refinement,
)


type LayerCase = tuple[
    NativeLayerCoreSource,
    phx.meshing.VolumeMeshingSpec,
    phx.meshing.CellMeshingResult,
    NativeLayerCoreAssociationTransfer,
]


@pytest.fixture(scope="module", params=("brep", "surface", "projected"))
def brep_layer_core(request: pytest.FixtureRequest) -> LayerCase:
    if not meshcore_available():
        pytest.skip("native layer/core generation requires meshcore")
    coordinates = phx.SpatialCoordinateContract.si()
    model = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=coordinates,
        source_id="wall-replay-box",
    )
    domain = MeshingDomain.from_brep(model)
    if request.param == "surface":
        # The same exact six-patch solid is independently authored as a
        # parametric domain, without BRep authority or projected associations.
        domain = MeshingDomain(
            domain.patches,
            domain.curves,
            domain.corner_count,
            source_id="wall-replay-parametric-box",
            source_revision="wall-replay-parametric-r1",
            regions=domain.regions,
            tolerance=domain.tolerance,
            accuracy=domain.accuracy,
            source_indices=domain.source_indices,
            source_occurrences=domain.source_occurrences,
            region_source_indices=domain.region_source_indices,
            region_source_occurrences=domain.region_source_occurrences,
            corner_points=domain.corner_points,
        )
    whole = phx.meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        domain.scope_entity_set_id(2),
        np.asarray(domain.scope_entity_ids(2), dtype=np.int64),
    )
    surface_spec = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 3, phx.meshing.CellFamilyPolicy(required=("triangle",))
        ),
        whole,
        size_controls=(
            phx.meshing.UniformSizeControl(
                whole,
                0.125,
                strength=phx.meshing.SizeControlStrength.SOFT,
            ),
        ),
    )
    surface_plan = phx.meshing.NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions("parametric_surface")
    ).plan(domain, surface_spec, coordinate_contract=coordinates)
    boundary = surface_plan.execute()
    assert boundary.surface_source is not None
    bottom = next(
        row
        for row, face in enumerate(domain.source_indices[2])
        if np.all(
            np.asarray(model.geometry.vertex_points)[
                _face_vertex_rows(model, int(face)), 2
            ]
            == 0.0
        )
    )
    bottom_key = (
        domain.entity_id(2, bottom),
        tuple(domain.source_occurrences[2][bottom]),
    )
    bottom_scope_id = next(
        int(identifier)
        for identifier in domain.scope_entity_ids(2)
        if domain.source_entity_key_for_scope(2, int(identifier)) == bottom_key
    )
    wall_scope = phx.meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        domain.scope_entity_set_id(2),
        np.asarray((bottom_scope_id,), dtype=np.int64),
    )
    volume_scope = phx.meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        3,
        domain.scope_entity_set_id(3),
        np.asarray(domain.scope_entity_ids(3), dtype=np.int64),
    )
    control = phx.meshing.BoundaryLayerControl(
        wall_scope,
        phx.meshing.LayerSchedule.geometric(2, 0.1, growth_rate=1.0),
        route=phx.meshing.BoundaryLayerRoute.ADVANCING,
        volume_scope=volume_scope,
        collision=phx.meshing.BoundaryLayerCollisionPolicy.FAIL,
    )
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(
                required=("prism", "tetrahedron"),
                preferred=("prism", "tetrahedron"),
                allow_mixed=True,
            ),
            geometry_order=2,
        ),
        whole,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                whole,
                0.125,
                strength=phx.meshing.SizeControlStrength.SOFT,
            ),
        ),
        region_seeds=(
            phx.meshing.RegionSeed(
                np.asarray((0.5, 0.5, 0.5)),
                domain.regions[0].name,
                "unit-box-fluid",
                phx.meshing.RegionRole.SOLID,
            ),
        ),
        layer_controls=(control,),
    )
    assert surface_plan.route_options.surface_schedule is not None
    source = brep_layer_core_source(
        domain,
        boundary,
        specification,
        surface_plan.route_options.surface_schedule,
    )
    assert source.layers.wall_association is not None
    assert source.layers.wall_association.association_kind is (
        GeometryAssociationKind.SURFACE
        if request.param == "surface"
        else GeometryAssociationKind.BREP
    )
    if request.param == "projected":
        projection = phx.geometry.prepare_brep_projection(model)
        policy = phx.meshing.AssociationPropagationPolicy(classification_tolerance=1.0e-8)
        association = phx.meshing.associate_mesh_vertices(
            source.layers.source_wall,
            projection,
            policy=policy,
        )
        source = _projected_source(source, control, association, projection, policy)
    result = (
        phx.meshing.NativeMeshingProvider(phx.meshing.NativeMeshingOptions("layer_core"))
        .plan(source, specification, coordinate_contract=coordinates)
        .execute()
    )
    assert result.certification is not None and result.certification.passed
    transfer = NativeLayerCoreAssociationTransfer(source, specification)
    return source, specification, result, transfer


def _projected_source(
    base: NativeLayerCoreSource,
    control: phx.meshing.BoundaryLayerControl,
    association: phx.meshing.GeometryAssociation,
    projection: NativeBRepProjection,
    policy: phx.meshing.AssociationPropagationPolicy,
) -> NativeLayerCoreSource:
    charts = base.source_wall_charts
    if charts is None or base.layer_policy is None:
        raise ValueError(
            "Projected wall replay requires its retained charts and layer policy."
        )
    assert isinstance(base.source_domain, MeshingDomain)
    realized_control = wall_layer_control(control, charts)
    layers = phx.meshing.prepare_boundary_layers(
        charts.mesh,
        realized_control,
        wall_association=association,
        policy=base.layer_policy,
    )
    core = fixed_core(base.source_domain, layers, charts)
    return NativeLayerCoreSource(
        layers,
        core.complex_,
        base.source_id,
        base.source_revision,
        vertex_layer_ids=core.vertex_layer_ids,
        cap_polygon_ids=np.arange(core.cap_count, dtype=np.int64),
        layer_regions=np.zeros(
            sum(block.cell_count for block in layers.mesh.blocks), dtype=np.int64
        ),
        region_entity_ids=base.region_entity_ids,
        region_occurrence_paths=base.region_occurrence_paths,
        region_ids=base.region_ids,
        core_region_map=base.core_region_map,
        source_boundary_scope=base.source_boundary_scope,
        source_domain=base.source_domain,
        fidelity_source=base.fidelity_source,
        source_wall_charts=charts,
        source_boundary_charts=base.source_boundary_charts,
        source_boundary_schedule=base.source_boundary_schedule,
        source_wall_projection=projection,
        source_wall_association_policy=policy,
        noncap_refinement=base.noncap_refinement,
        layer_control=realized_control,
        layer_policy=base.layer_policy,
        core_facet_source_ids=base.core_facet_source_ids,
        core_facet_source_occurrence_paths=base.core_facet_source_occurrence_paths,
    )


def _wall_table(
    result: phx.meshing.CellMeshingResult,
) -> tuple[int, phx.meshing.GeometryAssociation]:
    return next(
        (index, table)
        for index, table in enumerate(result.associations)
        if table.association_kind
        in (GeometryAssociationKind.BREP, GeometryAssociationKind.SURFACE)
        and target_dimension(result.mesh, table) == 0
    )


def _replace_table(
    result: phx.meshing.CellMeshingResult,
    index: int,
    table: phx.meshing.GeometryAssociation,
) -> phx.meshing.CellMeshingResult:
    tables = list(result.associations)
    tables[index] = table
    return eqx.tree_at(lambda value: value.associations, result, tuple(tables))


def _forged_occurrence(association: GeometryAssociation) -> GeometryAssociation:
    """Re-key one classified row to a forged occurrence through the constructor.

    Occurrence paths are static metadata, so the forgery is a coherent new table:
    the row's B-Rep entity key is re-derived for the forged path, as a forger
    who also controls the identities would emit it.
    """
    dimensions = np.asarray(association.source_dimensions)
    indices = np.asarray(association.source_indices)
    row = int(np.flatnonzero(dimensions >= 0)[0])
    paths = list(association.source_occurrence_paths)
    paths[row] = (*paths[row], "forged-occurrence")
    entities = list(association.source_entity_ids)
    if association.association_kind is GeometryAssociationKind.BREP:
        entities[row] = brep_entity_id(
            association.source_id,
            association.source_revision,
            int(dimensions[row]),
            int(indices[row]),
            occurrence_path=paths[row],
        )
    return GeometryAssociation(
        association.association_kind,
        association.source_id,
        association.source_revision,
        association.target_entity_set_id,
        association.target_global_ids,
        tuple(entities),
        association.residuals,
        resolved=association.resolved,
        ambiguous=association.ambiguous,
        exact=association.exact,
        source_dimensions=dimensions,
        source_indices=indices,
        source_roles=association.source_roles,
        source_occurrence_paths=tuple(paths),
        source_region_sides=association.source_region_sides,
        source_parameter_intervals=association.source_parameter_intervals,
        target_parameter_intervals=association.target_parameter_intervals,
        target_reference_pieces=association.target_reference_pieces,
        source_boundary_witness_id=association.source_boundary_witness_id,
        source_parameter_points=association.source_parameter_points,
        parameters=association.parameters,
        orientations=association.orientations,
        parent_dimensions=association.parent_dimensions,
        parent_ids=association.parent_ids,
        parent_association_id=association.parent_association_id,
        provenance=association.provenance,
    )


def test_brep_wall_refinement_and_complete_coarsening(brep_layer_core: LayerCase) -> None:
    _, _, original, transfer = brep_layer_core
    transfer.validate_source(original)
    _, refinement, policy = _periodic_mixed_refinement(
        original,
        _accepted_layer_columns(original),
        transfer,
        None,
        mark_layer=True,
    )
    refined = refinement.target
    transfer.validate_source(refined)
    _, old_wall = _wall_table(original)
    _, new_wall = _wall_table(refined)
    old_ids = set(np.asarray(old_wall.target_global_ids, dtype=np.int64))
    new_ids = set(np.asarray(new_wall.target_global_ids, dtype=np.int64)) - old_ids
    assert new_ids
    assert np.all(np.asarray(new_wall.resolved))
    assert not np.any(np.asarray(new_wall.ambiguous))
    assert all(
        new_wall.source_entity_ids[row] is not None
        for row, identifier in enumerate(new_wall.target_global_ids)
        if int(identifier) in new_ids
    )
    original_rows = {
        row[4]: row
        for row in _layer_association_semantics(original)
        if row[2] is old_wall.association_kind and row[3] == 0
    }
    refined_rows = {
        row[4]: row
        for row in _layer_association_semantics(refined)
        if row[2] is old_wall.association_kind and row[3] == 0
    }
    assert all(
        refined_rows[identifier] == row for identifier, row in original_rows.items()
    )
    new_rows = [row for row in refined_rows.values() if row[4] in new_ids]
    # Created wall vertices lie on source curves and faces. Their vertex rows
    # carry no target pieces or exact points; each created curve vertex keeps
    # its exact normalized curve parameter in the wall's CURVE vertex table
    # (wall-record contract of 396c7e272, ``_layer_wall_source_records``).
    assert {row[15] for row in new_rows} == {1, 2}
    curve_key = (
        old_wall.source_id,
        old_wall.source_revision,
        GeometryAssociationKind.CURVE,
        0,
    )
    curve_points = {
        row[4]: row[10]
        for row in _layer_association_semantics(refined)
        if row[:4] == curve_key and row[4] in new_ids
    }
    assert set(curve_points) == {row[4] for row in new_rows if row[15] == 1}
    assert all(
        isinstance(value, Fraction) and 0 < value < 1 for value in curve_points.values()
    )
    active = {
        int(identifier)
        for block in refined.mesh.blocks
        for identifier in block.global_ids
    }
    children = np.asarray(
        sorted(
            child
            for sibling in refinement.hierarchy.siblings
            for child in sibling.child_ids
            if child in active
        ),
        dtype=np.int64,
    )
    assert children.size
    restored = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            refined,
            phx.meshing.MarkedMeshAdaptation(
                (), children, hierarchy=refinement.hierarchy
            ),
            policy=policy,
        )
    ).target
    transfer.validate_source(restored)
    assert restored.mesh.topology_id == original.mesh.topology_id
    # An adapted target carries its own ``adaptation:<prepared_id>`` numeric
    # version (ff7c77f82, ``_finalize_native``), so its mesh/geometry ids and the
    # face witness ids that bind them differ from the original's. Every other
    # association field must be restored exactly, grouped under the witnesses
    # exactly as the original strata were.
    assert restored.mesh.mesh_id != original.mesh.mesh_id

    def strata(
        result: phx.meshing.CellMeshingResult,
    ) -> list[tuple[tuple[object, ...], ...]]:
        groups: dict[object, list[tuple[object, ...]]] = {}
        for row in _layer_association_semantics(result):
            groups.setdefault(row[13], []).append(row[:13] + row[14:])
        return sorted(
            (tuple(sorted(rows, key=repr)) for rows in groups.values()), key=repr
        )

    assert strata(restored) == strata(original)


@pytest.mark.parametrize(
    "field", ("parameters", "source_indices", "source_occurrence_paths")
)
def test_brep_wall_rejects_forged_source_rows(
    brep_layer_core: LayerCase, field: str
) -> None:
    _, _, result, transfer = brep_layer_core
    index, table = _wall_table(result)
    if field == "source_occurrence_paths":
        forged = _forged_occurrence(table)
    else:
        if field == "parameters":
            values = np.asarray(table.parameters).copy()
            row = int(np.flatnonzero(np.isfinite(values[:, 0]))[0])
            values[row, 0] += 0.125
        else:
            values = np.asarray(table.source_indices).copy()
            values[0] += 1
        forged = eqx.tree_at(lambda value: getattr(value, field), table, values)
    with pytest.raises(ValueError):
        transfer.validate_source(_replace_table(result, index, forged))


def test_brep_wall_rejects_stale_authority(brep_layer_core: LayerCase) -> None:
    source, specification, _, _ = brep_layer_core
    regions = np.asarray(source.layer_regions).copy()
    regions[0] += 1
    stale = eqx.tree_at(lambda value: value.layer_regions, source, regions)
    with pytest.raises(ValueError):
        NativeLayerCoreAssociationTransfer(stale, specification)


@pytest.mark.parametrize(
    "field", ("parameters", "source_indices", "source_occurrence_paths")
)
def test_projected_original_wall_rejects_forged_rows(
    brep_layer_core: LayerCase, field: str
) -> None:
    source, specification, _, _ = brep_layer_core
    projection = source.source_wall_projection
    policy = source.source_wall_association_policy
    if projection is None:
        pytest.skip("requires the original projected wall fixture")
    association = source.layers.wall_association
    assert association is not None and policy is not None
    if field == "source_occurrence_paths":
        forged = _forged_occurrence(association)
    else:
        if field == "parameters":
            values = np.asarray(association.parameters).copy()
            row = int(np.flatnonzero(np.isfinite(values[:, 0]))[0])
            values[row, 0] += 0.125
        else:
            values = np.asarray(association.source_indices).copy()
            values[0] += 1
        forged = eqx.tree_at(lambda value: getattr(value, field), association, values)
    with pytest.raises(ValueError):
        rebuilt = _projected_source(
            source,
            specification.layer_controls[0],
            forged,
            projection,
            policy,
        )
        validate_wall_source(rebuilt, 1.0e-8)


def test_projected_original_wall_rejects_stale_policy(brep_layer_core: LayerCase) -> None:
    source, _, _, _ = brep_layer_core
    if source.source_wall_projection is None:
        pytest.skip("requires the original projected wall fixture")
    altered = phx.meshing.AssociationPropagationPolicy()
    stale = eqx.tree_at(
        lambda value: value.source_wall_association_policy, source, altered
    )
    with pytest.raises(ValueError):
        validate_wall_source(stale, 1.0e-8)


@pytest.fixture(scope="module")
def unit_box_staged_source() -> NativeLayerCoreSource:
    if not meshcore_available():
        pytest.skip("native layer/core generation requires meshcore")
    from tools.meshing_qualification.scenarios.native_hybrid_profiles import (
        _unit_box_source,
    )

    source, _, _, _ = _unit_box_source()
    return source


@pytest.mark.parametrize("change", ("face", "occurrence", "cap"))
def test_staged_brep_boundary_rejects_forged_source_identity(
    unit_box_staged_source: NativeLayerCoreSource, change: str
) -> None:
    from phydrax.meshing._layer_core_staged_domain import (
        validate_brep_staged_layer_domain,
    )

    source = unit_box_staged_source
    domain = source.source_domain
    assert isinstance(domain, MeshingDomain)
    validate_brep_staged_layer_domain(source)
    identities = list(source.core_facet_source_ids)
    occurrences = list(source.core_facet_source_occurrence_paths)
    exterior = next(
        row for row, identity in enumerate(identities) if identity is not None
    )
    cap_polygon = int(np.asarray(source.cap_polygon_ids, dtype=np.int64)[0])
    cap_facet = int(
        np.asarray(source.complex.polygon_facets, dtype=np.int64)[cap_polygon]
    )
    if change == "face":
        wrong = next(
            domain.entity_id(2, row)
            for row in range(len(domain.patches))
            if domain.entity_id(2, row) != identities[exterior]
        )
        identities[exterior] = wrong
    elif change == "occurrence":
        occurrence = occurrences[exterior]
        assert occurrence is not None
        occurrences[exterior] = (*occurrence, "wrong-occurrence")
    else:
        identities[cap_facet] = identities[exterior]
        occurrences[cap_facet] = occurrences[exterior]
    forged = NativeLayerCoreSource(
        source.layers,
        source.complex,
        source.source_id,
        source.source_revision,
        vertex_layer_ids=source.vertex_layer_ids,
        cap_polygon_ids=source.cap_polygon_ids,
        layer_regions=source.layer_regions,
        region_entity_ids=source.region_entity_ids,
        region_occurrence_paths=source.region_occurrence_paths,
        region_ids=source.region_ids,
        core_region_map=source.core_region_map,
        source_boundary_scope=source.source_boundary_scope,
        source_domain=source.source_domain,
        fidelity_source=source.fidelity_source,
        source_wall_charts=source.source_wall_charts,
        source_boundary_charts=source.source_boundary_charts,
        source_boundary_schedule=source.source_boundary_schedule,
        source_wall_projection=source.source_wall_projection,
        source_wall_association_policy=source.source_wall_association_policy,
        cap_refinement=source.cap_refinement,
        noncap_refinement=source.noncap_refinement,
        layer_control=source.layer_control,
        layer_policy=source.layer_policy,
        maximum_deviation=source.maximum_deviation,
        core_facet_source_ids=tuple(identities),
        core_facet_source_occurrence_paths=tuple(occurrences),
    )
    with pytest.raises(ValueError):
        validate_brep_staged_layer_domain(forged)


def test_unit_box_execution_groups_preserve_caps_and_exact_cad_curves(
    unit_box_staged_source: NativeLayerCoreSource,
) -> None:
    from fractions import Fraction

    from phydrax.discretization._reference_cell import reference_cell_topology
    from phydrax.geometry._surface_source_support import (
        _cad_root_edges,
        validate_cad_curve_ancestry,
    )
    from phydrax.meshing._layer_core_edge_policy import (
        conforming_physical_edge_rows,
        noncap_source_edge_classification,
    )
    from phydrax.meshing._layer_core_execution import build_layer_core_execution
    from phydrax.meshing._layer_core_subdivision import subdivide_nested_prism_columns
    from phydrax.meshing.providers._native_layer_overlap import (
        exact_collinear_segment_partitions,
    )

    source = unit_box_staged_source
    classified = noncap_source_edge_classification(source)
    assert classified is not None
    removable, cad_edges = classified
    assert not set(removable) & set(cad_edges)
    polygons = np.asarray(source.complex.polygon_vertices, dtype=np.int64).reshape(-1, 3)
    original_points = np.asarray(source.complex.vertices, dtype=np.float64)
    virtual = next(
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for row, edge in enumerate(source.complex.plc_edge_vertices)
        if len(source.complex.plc_edge_facets[row]) == 2
        and all(original_points[int(vertex), 2] == 1.0 for vertex in edge)
        and all(
            0.0 < original_points[int(vertex), axis] < 1.0
            for vertex in edge
            for axis in (0, 1)
        )
        and len(
            {
                (
                    source.core_facet_source_ids[int(facet)],
                    source.core_facet_source_occurrence_paths[int(facet)],
                )
                for facet in source.complex.plc_edge_facets[row]
            }
        )
        == 1
    )
    assert virtual in removable
    virtual_polygons = {
        row
        for row, polygon in enumerate(polygons)
        if set(virtual) <= {int(vertex) for vertex in polygon}
    }
    assert len(virtual_polygons) == 2
    cap_edges = {
        tuple(sorted((int(polygon[index]), int(polygon[(index + 1) % 3]))))
        for polygon in polygons[np.asarray(source.cap_polygon_ids, dtype=np.int64)]
        for index in range(3)
    }
    assert not set(removable) & cap_edges
    execution = build_layer_core_execution(
        source, subdivide_nested_prism_columns(source.layers.mesh, 1)
    )
    output_facets = np.asarray(execution.complex.polygon_facets, dtype=np.int64)
    assert np.all(execution.protected_facets[output_facets[execution.cap_polygon_ids]])
    assert all(
        len(execution.source_polygon_candidates[int(output_facets[row])]) == 1
        for row in execution.cap_polygon_ids
    )
    original_facets = np.asarray(source.complex.polygon_facets, dtype=np.int64)
    for facet, rows in enumerate(execution.source_polygon_candidates):
        if execution.protected_facets[facet]:
            continue
        keys = {
            (
                source.core_facet_source_ids[int(original_facets[row])],
                source.core_facet_source_occurrence_paths[int(original_facets[row])],
            )
            for row in rows
        }
        assert len(keys) == 1
        assert next(iter(keys))[0] is not None
    assert any(
        virtual_polygons <= set(rows) for rows in execution.source_polygon_candidates
    )
    execution_polygons = np.asarray(
        execution.complex.polygon_vertices, dtype=np.int64
    ).reshape(-1, 3)
    uses: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for facet, polygon in zip(output_facets, execution_polygons, strict=True):
        for index, first in enumerate(polygon):
            last = int(polygon[(index + 1) % 3])
            first = int(first)
            edge = (min(first, last), max(first, last))
            uses.setdefault(edge, []).append((int(facet), 1 if first < last else -1))
    explicit = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in np.asarray(execution.complex.segments, dtype=np.int64)
    }
    internal = {
        edge
        for edge, rows in uses.items()
        if edge not in explicit
        and len(rows) == 2
        and rows[0][0] == rows[1][0]
        and rows[0][1] == -rows[1][1]
    }
    physical = {
        tuple(sorted(execution.complex.plc_edge_vertices[row]))
        for row in conforming_physical_edge_rows(execution.complex)
    }
    execution_vertices = {
        int(stencil[0][0]): row
        for row, stencil in enumerate(execution.vertex_stencils)
        if len(stencil) == 1 and stencil[0][1] == Fraction(1)
    }
    virtual_target = tuple(sorted(execution_vertices[vertex] for vertex in virtual))
    assert virtual_target not in physical
    assert virtual_target not in explicit
    assert not internal & physical
    assert explicit <= physical

    atlas = source.source_boundary_charts
    assert atlas is not None
    bindings = validate_cad_curve_ancestry(atlas, 0.0)
    root_edges, adjacent = _cad_root_edges(atlas)
    cells = {
        int(identifier): (block, row)
        for block in atlas.mesh.blocks
        for row, identifier in enumerate(np.asarray(block.global_ids, dtype=np.int64))
    }
    curve_segments = []
    for edge_id in bindings:
        cell_id = adjacent[edge_id][0][0]
        block, row = cells[cell_id]
        vertices = tuple(int(value) for value in np.asarray(block.vertices[row]))
        reference = reference_cell_topology(block.cell_kind).vertices
        points = tuple(
            tuple(
                Fraction.from_float(float(value))
                for value in reference[vertices.index(vertex)]
            )
            for vertex in root_edges[edge_id]
        )
        exact = atlas.geometry.evaluate_exact(
            atlas.mesh, np.asarray((cell_id,), dtype=np.int64), (points,)
        )[0]
        curve_segments.append(exact)
    candidates = tuple(
        tuple(
            tuple(Fraction.from_float(float(value)) for value in original_points[vertex])
            for vertex in edge
        )
        for edge in sorted(removable)
    )
    assert not exact_collinear_segment_partitions(candidates, tuple(curve_segments))


def test_layer_face_batch_reproduces_each_exact_cell_map(
    unit_box_staged_source: NativeLayerCoreSource,
) -> None:
    from fractions import Fraction

    from phydrax.discretization._cell_geometry import CellGeometrySpec
    from phydrax.discretization._reference_cell import reference_cell_topology
    from phydrax.meshing.providers._native_layer import _layer_output_boundary_faces
    from phydrax.meshing.providers._native_layer_overlap import (
        _evaluate_boundary_face_images,
    )

    mesh = unit_box_staged_source.layers.mesh
    geometry = CellGeometrySpec.affine(mesh)
    faces = tuple(matches[0] for matches in _layer_output_boundary_faces(mesh).values())[
        :8
    ]
    images, centers = _evaluate_boundary_face_images(mesh, geometry, faces)
    for face, image in zip(faces, images, strict=True):
        _, cell, _, kind, order, references = face
        points = tuple(references[int(vertex)] for vertex in order)
        expected = geometry.evaluate_exact(
            mesh, np.asarray((cell,), dtype=np.int64), (points,)
        )[0]
        assert image == expected
        vertices = reference_cell_topology(kind).vertices
        center = tuple(
            sum(
                (Fraction.from_float(float(vertex[axis])) for vertex in vertices),
                Fraction(0),
            )
            / Fraction(len(vertices))
            for axis in range(3)
        )
        expected_center = geometry.evaluate_exact(
            mesh, np.asarray((cell,), dtype=np.int64), ((center,),)
        )[0][0]
        assert centers[cell] == expected_center


def test_original_unit_box_grown_wall_query_is_complete_and_source_bound(
    unit_box_staged_source: NativeLayerCoreSource,
) -> None:
    from phydrax.meshing._layer_core_brep import grown_wall_cells
    from phydrax.meshing._layer_curving_records import parametric_wall_query

    source = unit_box_staged_source
    charts = source.source_boundary_charts
    assert charts is not None
    grown = tuple(int(value) for value in grown_wall_cells(source.layers))
    assert len(grown) > 1
    query = parametric_wall_query(charts, grown)
    assert query.domain.domain_id == charts.domain.domain_id
    assert query.source_id == charts.source_id
    assert query.source_revision == charts.source_revision
    assert len(query.patches) == 1
    assert set(query.patches) < set(charts.boundary_source.patches)
    with pytest.raises(ValueError, match="cover each original source patch"):
        parametric_wall_query(charts, grown[:-1])
    with pytest.raises(ValueError, match="do not bind"):
        parametric_wall_query(charts, (*grown, max(charts.cell_global_ids) + 1))


@pytest.mark.parametrize("ulps", (1, 7))
def test_full_atlas_replay_keeps_sampled_residuals_under_other_platform_rounding(
    unit_box_staged_source: NativeLayerCoreSource,
    monkeypatch: pytest.MonkeyPatch,
    ulps: int,
) -> None:
    """Sampled chord residuals are stored evidence, not re-derived bit for bit.

    Another platform's surface evaluation (XLA, FMA contraction, einsum order)
    lands a few ulps away; the original atlas still restores unchanged.
    """
    from phydrax.meshing._layer_wall_source_replay import validate_full_boundary_charts

    source = unit_box_staged_source
    charts = source.source_boundary_charts
    assert charts is not None and charts.mesh.blocks[0].cell_kind == "triangle"
    evaluate = MeshingDomain.evaluate

    def other_platform(
        self: MeshingDomain, patches: ArrayLike, charts_: ArrayLike, /
    ) -> np.ndarray:
        values = np.asarray(evaluate(self, patches, charts_))
        for _ in range(ulps):
            values = np.nextafter(values, np.inf)
        return values

    monkeypatch.setattr(MeshingDomain, "evaluate", other_platform)
    binding = source.binding_id
    validate_full_boundary_charts(source, 0.0)
    assert source.binding_id == binding


@pytest.mark.parametrize("residual", (-1.0, float("nan"), float("inf")))
def test_full_atlas_replay_refuses_an_invalid_stored_residual(
    unit_box_staged_source: NativeLayerCoreSource, residual: float
) -> None:
    from phydrax.meshing._layer_wall_source_replay import (
        _stored_residual,
        validate_full_boundary_charts,
    )

    assert not _stored_residual(residual, None)
    assert not _stored_residual(np.nextafter(0.0, 1.0), 0.0)
    assert _stored_residual(0.0, 0.0) and _stored_residual(1.0e-16, None)
    source = unit_box_staged_source
    charts = source.source_boundary_charts
    assert charts is not None
    faces = charts.mesh.entity_set(2).entity_set_id
    tables = []
    for table in charts.associations:
        if table.target_entity_set_id == faces:
            residuals = np.array(table.residuals, dtype=np.float64, copy=True)
            residuals[0] = residual
            table = eqx.tree_at(lambda value: value.residuals, table, residuals)
        tables.append(table)
    forged_charts = eqx.tree_at(lambda value: value.associations, charts, tuple(tables))
    forged = eqx.tree_at(
        lambda value: value.source_boundary_charts, source, forged_charts
    )
    with pytest.raises(ValueError):
        validate_full_boundary_charts(forged, 0.0)


def test_real_refined_prism_chart_requires_actual_parent_face(
    unit_box_staged_source: NativeLayerCoreSource,
) -> None:
    from fractions import Fraction

    from phydrax.discretization._nested_reference import nested_cell_reference_maps
    from phydrax.discretization._reference_cell import reference_cell_topology
    from phydrax.meshing._layer_core_subdivision import (
        nested_prism_reference_partition,
        subdivide_nested_prism_columns,
    )
    from phydrax.meshing._layer_curving import _mapped_child_chart_on_parent_face

    layer_mesh = unit_box_staged_source.layers.mesh
    partition = nested_prism_reference_partition(
        subdivide_nested_prism_columns(layer_mesh, 2)
    )
    maps = tuple(
        mapping
        for group in partition.maps
        for mapping in nested_cell_reference_maps(group)
    )
    topology = reference_cell_topology("prism")
    vertices = tuple(
        tuple(Fraction.from_float(float(value)) for value in point)
        for point in topology.vertices
    )
    boundary = None
    interior = None
    for mapping in maps:
        for face in topology.entities[2]:
            corners = tuple(vertices[int(index)] for index in face)
            actual = tuple(mapping.evaluate(point) for point in corners)
            center = tuple(
                sum((point[axis] for point in actual), Fraction(0)) / len(actual)
                for axis in range(3)
            )
            if len(face) == 3 and all(point[2] == 0 for point in actual):
                boundary = (mapping, corners)
            if (
                len(face) == 4
                and 0 < center[0]
                and 0 < center[1]
                and center[0] + center[1] < 1
                and 0 < center[2] < 1
            ):
                interior = (mapping, corners[:3])
            if boundary is not None and interior is not None:
                break
        if boundary is not None and interior is not None:
            break
    assert boundary is not None and interior is not None
    _mapped_child_chart_on_parent_face(boundary[0], "prism", "prism", boundary[1])
    with pytest.raises(ValueError, match="parent boundary face"):
        _mapped_child_chart_on_parent_face(interior[0], "prism", "prism", interior[1])


def _assert_selected_wall_query_and_replay(
    source: NativeLayerCoreSource,
    result: phx.meshing.CellMeshingResult,
    transfer: NativeLayerCoreAssociationTransfer,
) -> None:
    from phydrax.geometry._domain_boundary import MeshingDomainBoundarySource
    from phydrax.geometry._source_fidelity import validate_source_fidelity_witness

    charts = source.source_wall_charts
    evidence = result.layer_curving_evidence
    assert charts is not None and evidence is not None
    query = evidence.source_query
    assert isinstance(query, MeshingDomainBoundarySource)
    assert set(query.patches) < set(charts.boundary_source.patches)
    rebuilt_domain = eqx.tree_at(
        lambda value: value.corner_points,
        query.domain,
        np.array(query.domain.corner_points, copy=True),
    )
    rebuilt = MeshingDomainBoundarySource(
        rebuilt_domain,
        query.patches,
        resolution=query.resolution,
        boundary_cell_ids=query.boundary_cell_ids,
        boundary_patch_ids=query.boundary_patch_ids,
        boundary_occurrences=query.boundary_occurrences,
    )
    token = validate_source_fidelity_witness(
        result.mesh, result.geometry, rebuilt, evidence.chart_witness
    )
    assert token.binds(result.mesh, result.geometry, rebuilt)
    with pytest.raises(ValueError):
        validate_source_fidelity_witness(
            result.mesh, result.geometry, charts.boundary_source, evidence.chart_witness
        )
    foreign_patch = next(
        patch for patch in charts.boundary_source.patches if patch not in query.patches
    )
    foreign_query = MeshingDomainBoundarySource(
        rebuilt_domain, (foreign_patch,), resolution=query.resolution
    )
    with pytest.raises(ValueError):
        validate_source_fidelity_witness(
            result.mesh, result.geometry, foreign_query, evidence.chart_witness
        )
    transfer.validate_source(result)


def test_full_wall_selected_query_reproves_after_independent_domain_rebuild(
    brep_layer_core: LayerCase,
) -> None:
    source, _, result, transfer = brep_layer_core
    _assert_selected_wall_query_and_replay(source, result, transfer)


def _small_real_box_selected_wall_case() -> tuple[
    NativeLayerCoreSource, phx.meshing.VolumeMeshingSpec, phx.meshing.CellMeshingResult
]:
    if not meshcore_available():
        pytest.skip("native layer/core generation requires meshcore")
    coordinates = phx.SpatialCoordinateContract.si()
    model = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=coordinates,
        source_id="selected-wall-small-box",
    )
    domain = MeshingDomain.from_brep(model)
    whole = phx.meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        domain.scope_entity_set_id(2),
        np.asarray(domain.scope_entity_ids(2), dtype=np.int64),
    )
    surface_spec = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 3, phx.meshing.CellFamilyPolicy(required=("triangle",))
        ),
        whole,
        size_controls=(
            phx.meshing.UniformSizeControl(
                whole,
                0.25,
                strength=phx.meshing.SizeControlStrength.SOFT,
            ),
        ),
    )
    surface_plan = phx.meshing.NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions("parametric_surface")
    ).plan(domain, surface_spec, coordinate_contract=coordinates)
    boundary = surface_plan.execute()
    bottom = next(
        row
        for row, face in enumerate(domain.source_indices[2])
        if np.all(
            np.asarray(model.geometry.vertex_points)[
                _face_vertex_rows(model, int(face)), 2
            ]
            == 0.0
        )
    )
    bottom_key = (
        domain.entity_id(2, bottom),
        tuple(domain.source_occurrences[2][bottom]),
    )
    bottom_id = next(
        int(identifier)
        for identifier in domain.scope_entity_ids(2)
        if domain.source_entity_key_for_scope(2, int(identifier)) == bottom_key
    )
    wall = phx.meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        domain.scope_entity_set_id(2),
        np.asarray((bottom_id,), dtype=np.int64),
    )
    volume = phx.meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        3,
        domain.scope_entity_set_id(3),
        np.asarray(domain.scope_entity_ids(3), dtype=np.int64),
    )
    control = phx.meshing.BoundaryLayerControl(
        wall,
        phx.meshing.LayerSchedule.geometric(2, 0.1, growth_rate=1.0),
        route=phx.meshing.BoundaryLayerRoute.ADVANCING,
        volume_scope=volume,
        collision=phx.meshing.BoundaryLayerCollisionPolicy.FAIL,
    )
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(
                required=("prism", "tetrahedron"),
                preferred=("prism", "tetrahedron"),
                allow_mixed=True,
            ),
            geometry_order=2,
        ),
        whole,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                whole,
                0.25,
                strength=phx.meshing.SizeControlStrength.SOFT,
            ),
        ),
        region_seeds=(
            phx.meshing.RegionSeed(
                np.asarray((0.5, 0.5, 0.5)),
                domain.regions[0].name,
                "unit-box-fluid",
                phx.meshing.RegionRole.SOLID,
            ),
        ),
        layer_controls=(control,),
    )
    assert surface_plan.route_options.surface_schedule is not None
    source = brep_layer_core_source(
        domain, boundary, specification, surface_plan.route_options.surface_schedule
    )
    result = (
        phx.meshing.NativeMeshingProvider(phx.meshing.NativeMeshingOptions("layer_core"))
        .plan(source, specification, coordinate_contract=coordinates)
        .execute()
    )
    assert result.certification is not None and result.certification.passed
    return source, specification, result


def test_small_real_box_selected_wall_query_and_association_replay() -> None:
    source, specification, result = _small_real_box_selected_wall_case()
    _assert_selected_wall_query_and_replay(
        source, result, NativeLayerCoreAssociationTransfer(source, specification)
    )


def test_small_real_box_selected_wall_actual_result_cold(tmp_path: Path) -> None:
    """Persist the actual accepted P2 result, not a reconstructed stripped result."""
    import json
    import subprocess
    import sys

    if not meshcore_available():
        pytest.skip("native layer/core generation requires meshcore")
    path = tmp_path / "selected-wall-actual.phx"
    writer = """
import json, os, sys
from phydrax.lifecycle import Composition, CompositionEntry
from phydrax.meshing._model_codec import (MODEL_ARCHIVE_LIMITS, layer_core_authoring_recipe,
    layer_core_accepted_result_recipe, model_archive_content_id, write_model_artifact)
from tests.unit.meshing.test_brep_layer_wall_replay import _small_real_box_selected_wall_case
s, spec, result = _small_real_box_selected_wall_case()
assert s.layer_control is not None and s.layer_policy is not None
assert result.layer_curving_evidence is not None
a = CompositionEntry(layer_core_authoring_recipe(s, s.layer_control, s.layer_policy),
    entry_id="source", role="topology", owner_id="phydrax.meshing",
    structure_id=s.layers.mesh.topology_id, revision_id=s.binding_id, semantics_id="actual-source")
b = CompositionEntry(spec, entry_id="specification", role="model-state", owner_id="phydrax.meshing",
    structure_id=s.binding_id, revision_id=spec.specification_id, semantics_id="actual-request",
    dependencies=(a.binding("revision"),))
c = CompositionEntry(layer_core_accepted_result_recipe(result, s, spec),
    entry_id="result", role="topology", owner_id="phydrax.meshing",
    structure_id=result.mesh.topology_id, revision_id=result.result_id, semantics_id="actual-accepted-result",
    dependencies=(a.binding("revision"), b.binding("revision")))
write_model_artifact(sys.argv[1], Composition((a,b,c), boundary_id="selected-wall-actual-cold"),
    limits=MODEL_ARCHIVE_LIMITS)
print(json.dumps({"pid":os.getpid(), "source":s.binding_id, "result":result.result_id,
    "mesh":result.mesh.mesh_id, "curving":result.layer_curving_evidence.evidence_id,
    "content":model_archive_content_id(sys.argv[1], limits=MODEL_ARCHIVE_LIMITS)}))
"""
    reader = """
import json, os, sys
from phydrax.lifecycle import Composition
from phydrax.meshing._contracts import VolumeMeshingSpec
from phydrax.meshing._model_codec import (MODEL_ARCHIVE_LIMITS, read_model_artifact,
    restore_layer_core_authoring, restore_layer_core_accepted_result, restore_model_archive_value)
expected = json.loads(sys.argv[2])
archive = read_model_artifact(sys.argv[1], expected_type=Composition, restore=restore_model_archive_value,
    expected_content_id=expected["content"], limits=MODEL_ARCHIVE_LIMITS)
spec = VolumeMeshingSpec.from_stored_fields(archive.value("specification"))
source = restore_layer_core_authoring(archive.value("source"), specification=spec)
result = restore_layer_core_accepted_result(archive.value("result"), source, spec)
assert result.layer_curving_evidence is not None
assert (source.binding_id, result.result_id, result.mesh.mesh_id, result.layer_curving_evidence.evidence_id) == (
    expected["source"], expected["result"], expected["mesh"], expected["curving"])
assert os.getpid() != expected["pid"]
print(json.dumps({"pid":os.getpid(), "result":result.result_id}))
"""
    produced = subprocess.run(
        [sys.executable, "-c", writer, str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=900,
    )
    expected = json.loads(produced.stdout.strip().splitlines()[-1])
    consumed = subprocess.run(
        [sys.executable, "-c", reader, str(path), json.dumps(expected)],
        check=True,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert json.loads(consumed.stdout.strip().splitlines()[-1])["pid"] != expected["pid"]
