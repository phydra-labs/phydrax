#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import gc
import weakref
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import replace
from fractions import Fraction
from typing import Any, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import jvp, vjp

import phydrax as phx
import phydrax.meshing._surface_metric as owner
from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
from phydrax._fingerprint import tree_content_equal
from phydrax._meshcore import (
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionLimits,
    NativeExecutionScope,
)
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization import CellBlock, CellMesh
from phydrax.discretization._cell_geometry import (
    PolynomialComposedCellGeometryElement,
    SplineCellGeometryElement,
)
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.discretization._coordinate_enclosure import (
    coordinate_expressions,
    Polynomial,
)
from phydrax.discretization._sphere_chart_deformation import (
    cone_candidates,
    sphere_full_map_displacement,
)
from phydrax.discretization._surface_chart_records import (
    PreparedSurfaceChartDeformation,
    SurfaceChartWitness,
)
from phydrax.discretization.fem import FiniteElementDiscretization
from phydrax.discretization.finite_volume._remap_evidence import (
    MappedSurfaceChartRemapEvidence,
)
from phydrax.geometry import BSplineSurfacePatch
from phydrax.geometry._sphere_material_atlas import sphere_material_strata
from phydrax.geometry._surface_source_support import SurfaceSourceRootAtlas
from phydrax.lifecycle._meshing_field_records import prepare_meshing_field_owner
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    source_content_equal,
    validate_meshing_source_closure,
    write_meshing_source_closure,
)
from phydrax.meshing import (
    execute_mesh_adaptation,
    MeshAdaptationStatus,
    MeshingEntityKind,
    MeshingScope,
    MeshMetricField,
    MetricMeshAdaptation,
    prepare_mesh_adaptation,
)
from phydrax.meshing._adaptation import MeshAdaptationPolicy, MeshAdaptationRoute
from phydrax.meshing._contracts import MeshingLimits
from phydrax.meshing._entity_keys import entity_keys, key_rows
from phydrax.meshing._lineage import inherit_mesh_organization
from phydrax.meshing._mesh_part import MeshPart
from phydrax.meshing._periodic import metric_target_lineage
from phydrax.meshing._result import CellMeshingResult
from phydrax.meshing._surface_association_transfer import prepare_surface_curve_witness
from phydrax.meshing._surface_boundary_sources import SurfaceChartBoundarySource
from phydrax.meshing._surface_metric import (
    execute_sphere_metric_adaptation,
    execute_surface_metric_adaptation,
)
from phydrax.meshing._tetra_metric import MetricRemeshingEvidence, MetricRemeshingStatus
from phydrax.meshing._topology_edit import assemble_topology_edit, TopologyEditBlock
from phydrax.meshing.qualification import AnalyticSurface, sphere_surface, torus_surface
from phydrax.solver import FiniteElementAcceptedState, FiniteElementTopologyTransaction
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.surface_metric import (
    COORDINATE_RECONSTRUCTION_RTOL,
    GEOMETRY_EVALUATION_ATOL,
    host_coordinate_phase,
    MATERIAL_FIELD_ATOL,
    MATERIAL_FIELD_RTOL,
    material_p1_reference_matrix,
    METRIC_ARC_RTOL,
    native_coordinate_phase,
    native_coordinate_resources,
    prepare_rational_metric_archive,
    prepare_rational_metric_fields,
    prepare_rational_metric_scenario,
    prepare_rational_trim_metric_source,
    prepare_sphere_metric_source,
    rational_patch_point,
    rational_surface_reconstruction_reference,
    rational_trim_point,
    RationalMaterialHistory,
    RationalMetricArchive,
    RationalMetricFieldState,
    RationalMetricScenario,
    require_exact_rational_patch_map,
    solve_surface_reaction,
    SOURCE_RESIDUAL_ATOL,
    SOURCE_RESIDUAL_RTOL,
    sphere_native_scope,
    SphereMetricSource,
    surface_patch,
    surface_resources,
    UVMetricSource,
)


def test_rational_metric_source_reuses_certified_coordinate_restrictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.meshing import _surface_association_transfer as associations

    prepared = prepare_rational_trim_metric_source()
    with coordinate_reference_budget() as budget:
        proof = associations.source_proof(
            prepared.transfer.support,
            prepared.source,
            prepared.transfer.receipts,
            budget=budget,
        )
        original_work = budget.work_units

    def forbid_reproof(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("retained source-fidelity proof must not be repeated")

    monkeypatch.setattr(associations, "source_proof", forbid_reproof)
    with coordinate_reference_budget() as budget:
        witness = owner.prepare_surface_metric_source(
            prepared.source, prepared.transfer, budget=budget
        )
        reused_work = budget.work_units
    rows = key_rows(proof.cell_ids[:, None], np.asarray(witness.cell_global_ids)[:, None])
    np.testing.assert_array_equal(witness.patches, proof.patches[rows])
    np.testing.assert_array_equal(
        witness.charts, np.asarray(proof.source_corners, dtype=np.float64)[rows]
    )
    assert reused_work < original_work


@pytest.mark.parametrize(
    "builder", [sphere_surface, torus_surface], ids=["sphere", "torus"]
)
def test_curved_patch_metric_preserves_source_chart_geometry(
    builder: Callable[[], AnalyticSurface],
) -> None:
    surface = builder()
    source, patches, charts = surface_patch(surface)
    # Independent analytic tangent scales at (u,v)=(0,0): sphere (1,1), torus (2.6,.6).
    if surface.domain.source_id == "sphere":
        tensor = np.eye(3, dtype=np.float64) * (0.95 / 0.2) ** 2
    else:
        tensor = np.diag((1.0, (0.95 / 0.52) ** 2, (0.95 / 0.12) ** 2))
    metric = np.broadcast_to(tensor, (4, 3, 3))
    before = np.asarray(source.coordinates).copy()
    with surface_resources() as (native_scope, ledger):
        outcome = execute_surface_metric_adaptation(
            source,
            metric,
            surface.domain,
            source_id=surface.domain.source_id,
            source_revision=surface.domain.source_revision,
            cell_patches=patches,
            cell_charts=charts,
            maximum_fidelity=0.1,
            maximum_passes=4,
            topology_operations=False,
            relocation=False,
            budget=ledger,
            scope=native_scope,
        )
    assert outcome.evidence.status is MetricRemeshingStatus.COMPLETE
    assert outcome.geometry.source_id == surface.domain.source_id
    assert outcome.geometry.source_revision == surface.domain.source_revision
    cell_blocks = []
    for block in outcome.geometry.edit.blocks:
        assert isinstance(block, TopologyEditBlock)
        cell_blocks.append(block.cells)
    target_cells = np.concatenate(cell_blocks)
    represented = surface.domain.evaluate(
        np.repeat(outcome.geometry.cell_patches, 3),
        outcome.geometry.cell_charts.reshape((-1, 2)),
    ).reshape((-1, 3, 3))
    np.testing.assert_allclose(
        represented,
        outcome.geometry.edit.coordinates[target_cells],
        atol=GEOMETRY_EVALUATION_ATOL,
    )
    assert (
        np.max(surface.distance(outcome.geometry.edit.coordinates))
        < GEOMETRY_EVALUATION_ATOL
    )
    assert outcome.evidence.maximum_fidelity_bound <= 0.1
    np.testing.assert_array_equal(source.coordinates, before)


def test_metric_target_lineage_binds_regrouped_scientific_cells() -> None:

    surface = torus_surface()
    source, patches, charts = surface_patch(surface)
    with surface_resources() as (native_scope, ledger):
        outcome = execute_surface_metric_adaptation(
            source,
            np.broadcast_to(np.eye(3), (4, 3, 3)),
            surface.domain,
            source_id=surface.domain.source_id,
            source_revision=surface.domain.source_revision,
            cell_patches=patches,
            cell_charts=charts,
            maximum_fidelity=0.1,
            maximum_passes=1,
            topology_operations=False,
            relocation=False,
            budget=ledger,
            scope=native_scope,
        )
    with coordinate_reference_budget() as budget:
        staged, _, _ = assemble_topology_edit(
            source, outcome.geometry.edit, numeric_version="regroup", budget=budget
        )
    cells = np.concatenate([np.asarray(block.vertices) for block in staged.blocks])
    ids = np.concatenate([np.asarray(block.global_ids) for block in staged.blocks])
    blocks = tuple(
        CellBlock(
            f"chart/{row}",
            "triangle",
            cells[row : row + 1],
            global_ids=ids[row : row + 1],
        )
        for row in reversed(range(ids.size))
    )
    regrouped = CellMesh(
        staged.coordinates,
        blocks,
        vertex_global_ids=staged.vertex_global_ids,
        numeric_version=staged.numeric_version,
    )
    edge_rows = key_rows(entity_keys(staged, 1), entity_keys(regrouped, 1))
    assert np.all(edge_rows >= 0)
    target = CellMesh(
        staged.coordinates,
        blocks,
        vertex_global_ids=staged.vertex_global_ids,
        entity_global_ids={1: np.asarray(staged.entity_set(1).entity_ids)[edge_rows]},
        numeric_version=staged.numeric_version,
    )
    with surface_resources() as (native_scope, ledger):
        lineage = metric_target_lineage(
            source, outcome.geometry.edit, target, scope=native_scope, budget=ledger
        )
    assert lineage.target_topology_id == target.topology_id
    assert lineage.source_topology_id == source.topology_id
    mismatched = CellMesh(
        staged.coordinates,
        blocks,
        vertex_global_ids=staged.vertex_global_ids,
        entity_global_ids={1: staged.entity_set(1).entity_ids},
        numeric_version=staged.numeric_version,
    )
    with (
        pytest.raises(ValueError, match="scientific carrier incidence"),
        surface_resources() as (native_scope, ledger),
    ):
        metric_target_lineage(
            source, outcome.geometry.edit, mismatched, scope=native_scope, budget=ledger
        )


def test_metric_target_lineage_refuses_replaced_scientific_cell() -> None:

    surface = torus_surface()
    source, patches, charts = surface_patch(surface)
    with surface_resources() as (native_scope, ledger):
        outcome = execute_surface_metric_adaptation(
            source,
            np.broadcast_to(np.eye(3), (4, 3, 3)),
            surface.domain,
            source_id=surface.domain.source_id,
            source_revision=surface.domain.source_revision,
            cell_patches=patches,
            cell_charts=charts,
            maximum_fidelity=0.1,
            maximum_passes=1,
            topology_operations=False,
            relocation=False,
            budget=ledger,
            scope=native_scope,
        )
    with coordinate_reference_budget() as budget:
        staged, _, _ = assemble_topology_edit(
            source, outcome.geometry.edit, numeric_version="replacement", budget=budget
        )
    target = CellMesh(
        staged.coordinates,
        tuple(
            CellBlock(
                block.name,
                block.cell_kind,
                block.vertices,
                global_ids=np.asarray(block.global_ids) + 1000,
            )
            for block in staged.blocks
        ),
        vertex_global_ids=staged.vertex_global_ids,
        entity_global_ids={
            degree: staged.entity_set(degree).entity_ids for degree in range(2)
        },
        numeric_version=staged.numeric_version,
    )
    with (
        pytest.raises(ValueError, match="scientific carrier entity identities"),
        surface_resources() as (native_scope, ledger),
    ):
        metric_target_lineage(
            source, outcome.geometry.edit, target, scope=native_scope, budget=ledger
        )


def test_complete_sphere_quality_bounds_the_independent_curved_differential() -> None:
    surface = sphere_surface()
    mesh, patches, charts = surface_patch(surface)
    metric = np.broadcast_to(np.eye(3) / 0.2**2, (4, 3, 3))
    with surface_resources() as (native_scope, ledger):
        outcome = execute_surface_metric_adaptation(
            mesh,
            metric,
            surface.domain,
            source_id=surface.domain.source_id,
            source_revision=surface.domain.source_revision,
            cell_patches=patches,
            cell_charts=charts,
            maximum_fidelity=0.1,
            minimum_metric_quality=0.2,
            maximum_passes=1,
            topology_operations=False,
            relocation=False,
            budget=ledger,
            scope=native_scope,
        )
    assert outcome.evidence.status is MetricRemeshingStatus.COMPLETE
    assert outcome.evidence.minimum_metric_quality >= 0.2
    assert outcome.evidence.converged
    for triangle in charts:
        frame = (triangle[1:] - triangle[0]).T
        sampled = []
        for first in np.linspace(0.0, 1.0, 9):
            for second in np.linspace(0.0, 1.0 - first, 9):
                u, v = (
                    (1 - first - second) * triangle[0]
                    + first * triangle[1]
                    + second * triangle[2]
                )
                differential = np.asarray(
                    (
                        (-np.sin(u) * np.cos(v), -np.cos(u) * np.sin(v)),
                        (np.cos(u) * np.cos(v), -np.sin(u) * np.sin(v)),
                        (0.0, np.cos(v)),
                    )
                )
                tangents = differential @ frame
                gram = tangents.T @ tangents
                sampled.append(
                    np.sqrt(3 * np.linalg.det(gram)) / (np.trace(gram) - gram[0, 1])
                )
        assert outcome.evidence.minimum_metric_quality <= min(sampled)


def test_spherical_edge_metric_uses_source_arc_not_straight_chord() -> None:
    surface = sphere_surface()
    source, patches, charts = surface_patch(surface)
    metric = np.broadcast_to(np.eye(3, dtype=np.float64) / 0.2**2, (4, 3, 3))
    with surface_resources() as (native_scope, ledger):
        outcome = execute_surface_metric_adaptation(
            source,
            metric,
            surface.domain,
            source_id=surface.domain.source_id,
            source_revision=surface.domain.source_revision,
            cell_patches=patches,
            cell_charts=charts,
            maximum_fidelity=0.1,
            maximum_passes=0,
            budget=ledger,
            scope=native_scope,
        )
    # Latitude v=.2 is shorter than a meridian by cos(.2); Gauss integrates the
    # analytic arc, whereas chord length is 2 sin(.1) cos(.2)/.2.
    assert outcome.evidence.minimum_metric_length == pytest.approx(
        np.cos(0.2), rel=METRIC_ARC_RTOL
    )
    chord = 2.0 * np.sin(0.1) * np.cos(0.2) / 0.2
    assert outcome.evidence.minimum_metric_length > chord + 1.0e-3


def test_stale_surface_revision_is_refused_without_mutation() -> None:
    surface = sphere_surface()
    source, patches, charts = surface_patch(surface)
    before = np.asarray(source.coordinates).copy()
    with pytest.raises(ValueError, match="revision"):
        with surface_resources() as (native_scope, ledger):
            execute_surface_metric_adaptation(
                source,
                np.broadcast_to(np.eye(3), (4, 3, 3)),
                surface.domain,
                source_id=surface.domain.source_id,
                source_revision="stale",
                cell_patches=patches,
                cell_charts=charts,
                maximum_fidelity=0.1,
                budget=ledger,
                scope=native_scope,
            )
    np.testing.assert_array_equal(source.coordinates, before)


def test_surface_budget_exhaustion_keeps_source_and_unmet_metric_evidence() -> None:
    surface = sphere_surface()
    source, patches, charts = surface_patch(surface)
    before = np.asarray(source.coordinates).copy()
    with surface_resources() as (native_scope, ledger):
        outcome = execute_surface_metric_adaptation(
            source,
            np.broadcast_to(np.eye(3) / 0.05**2, (4, 3, 3)),
            surface.domain,
            source_id=surface.domain.source_id,
            source_revision=surface.domain.source_revision,
            cell_patches=patches,
            cell_charts=charts,
            maximum_fidelity=0.1,
            maximum_operations=0,
            maximum_passes=3,
            budget=ledger,
            scope=native_scope,
        )
    assert outcome.evidence.status is MetricRemeshingStatus.RESOURCE_LIMIT
    assert outcome.evidence.out_of_range_edges == 5
    assert outcome.evidence.resource_message
    assert dict(outcome.evidence.resource_requested)["maximum_operations"] == 0
    assert dict(outcome.evidence.resource_achieved)["operation_attempts"] == 0
    np.testing.assert_array_equal(outcome.geometry.edit.coordinates, before)
    np.testing.assert_array_equal(source.coordinates, before)


def test_spherical_refinement_improves_continuous_fidelity_without_flattening() -> None:
    surface = sphere_surface()
    source, patches, charts = surface_patch(surface)
    before = np.asarray(source.coordinates).copy()
    with surface_resources() as (native_scope, ledger):
        outcome = execute_surface_metric_adaptation(
            source,
            np.broadcast_to(np.eye(3) / 0.1**2, (4, 3, 3)),
            surface.domain,
            source_id=surface.domain.source_id,
            source_revision=surface.domain.source_revision,
            cell_patches=patches,
            cell_charts=charts,
            maximum_fidelity=0.01,
            maximum_passes=12,
            relocation=False,
            budget=ledger,
            scope=native_scope,
        )
    assert outcome.evidence.status is MetricRemeshingStatus.COMPLETE
    assert outcome.evidence.maximum_fidelity_bound <= 0.01
    assert (
        np.max(surface.distance(outcome.geometry.edit.coordinates))
        < GEOMETRY_EVALUATION_ATOL
    )
    cell_blocks = []
    for block in outcome.geometry.edit.blocks:
        assert isinstance(block, TopologyEditBlock)
        cell_blocks.append(block.cells)
    target_cells = np.concatenate(cell_blocks)
    corner = outcome.geometry.edit.coordinates[target_cells]
    normal = np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0])
    assert np.all(np.sum(normal * np.mean(corner, axis=1), axis=1) > 0.0)
    represented = surface.domain.evaluate(
        np.repeat(outcome.geometry.cell_patches, 3),
        outcome.geometry.cell_charts.reshape((-1, 2)),
    ).reshape((-1, 3, 3))
    np.testing.assert_allclose(represented, corner, atol=GEOMETRY_EVALUATION_ATOL)
    np.testing.assert_array_equal(source.coordinates, before)


@pytest.fixture(scope="module")
def sphere_metric_source() -> SphereMetricSource:
    return prepare_sphere_metric_source(0.8)


@pytest.fixture(scope="module")
def fine_sphere_metric_source() -> SphereMetricSource:
    return prepare_sphere_metric_source(0.4)


def test_sphere_cone_filter_resolves_zero_area_contact_and_retains_source_cells(
    sphere_metric_source: SphereMetricSource,
) -> None:

    _, source, atlas, _, policy = sphere_metric_source
    cells = np.concatenate([np.asarray(block.vertices) for block in source.mesh.blocks])
    physical = np.asarray(atlas.physical_rows)
    neighbors = np.flatnonzero(
        np.sum(np.isin(cells[physical], cells[physical[0]]), axis=1) == 2
    )
    assert neighbors.size
    ledger = CoordinateEnclosureBudget(
        HostBatchTicket(
            policy.limits.maximum_work_units,
            bytes_upper=policy.limits.maximum_scratch_bytes,
        ),
        maximum_work_units=policy.limits.maximum_work_units,
        maximum_memory_bytes=policy.limits.maximum_scratch_bytes,
    )
    with (
        sphere_native_scope(policy.limits) as native_scope,
        native_coordinate_phase(native_scope, ledger),
    ):
        kept_source, kept_target, _ = cone_candidates(
            atlas,
            atlas,
            np.asarray((0, 0), dtype=np.int64),
            np.asarray((0, neighbors[0]), dtype=np.int64),
            policy.limits.maximum_scratch_bytes,
            budget=ledger,
        )
    np.testing.assert_array_equal(kept_source, (0,))
    np.testing.assert_array_equal(kept_target, (0,))


def test_batched_sphere_location_preserves_serial_cone_order_and_boundary_signs(
    sphere_metric_source: SphereMetricSource,
) -> None:
    from phydrax._meshcore import exact_orient3d

    domain, source, atlas, transfer, policy = sphere_metric_source
    metric = np.broadcast_to(np.eye(3), (source.mesh.coordinates.shape[0], 3, 3))
    with (
        sphere_native_scope(policy.limits) as native_scope,
        native_coordinate_resources(native_scope, policy.limits) as ledger,
    ):
        _, background = owner._sphere_initial_state(
            source, metric, atlas, transfer, policy, 0.16, budget=ledger
        )
        rows = np.arange(atlas.num_charts, dtype=np.int64)[::-1]
        for direction in (
            background.cone_directions[0, 0],
            np.mean(background.cone_directions[0], axis=0),
        ):
            expected = []
            for row in rows:
                corners = background.cone_directions[row]
                signs = exact_orient3d(
                    np.zeros(3),
                    np.stack((direction, corners[0], corners[0])),
                    np.stack((corners[1], direction, corners[1])),
                    np.stack((corners[2], corners[2], direction)),
                )
                if np.all(signs * background.cone_orientations[row] >= 0):
                    expected.append((int(row), tuple(map(int, signs))))
            actual = [
                (row, tuple(map(int, signs)))
                for row, signs in owner._sphere_location_cones(
                    background, rows, direction, budget=ledger
                )
            ]
            assert actual == expected


def test_sphere_displacement_zero_requires_exact_coefficient_cancellation() -> None:

    vertices = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    zero = CoordinateEnclosureBudget(
        HostBatchTicket(3, bytes_upper=0), maximum_work_units=3, maximum_memory_bytes=0
    )
    with host_coordinate_phase(zero):
        assert sphere_full_map_displacement(({}, {}, {}), vertices, budget=zero) == 0.0
    nonzero = CoordinateEnclosureBudget(
        HostBatchTicket(3, bytes_upper=0), maximum_work_units=3, maximum_memory_bytes=0
    )
    with pytest.raises(CoordinateEnclosureResourceError), host_coordinate_phase(nonzero):
        sphere_full_map_displacement(
            ({(0, 0): Fraction(1, 2**100)}, {}, {}), vertices, budget=nonzero
        )


@pytest.mark.parametrize(
    "minimum_metric_quality", [0.05, 0.1], ids=["default-floor", "authored-floor"]
)
def test_native_sphere_metric_changes_cells_with_radial_correspondence(
    sphere_metric_source: SphereMetricSource,
    minimum_metric_quality: float,
) -> None:
    domain, source, atlas, transfer, policy = sphere_metric_source
    before = np.asarray(source.geometry.coordinates).copy()
    ledger = CoordinateEnclosureBudget(
        HostBatchTicket(
            policy.limits.maximum_work_units,
            bytes_upper=policy.limits.maximum_scratch_bytes,
        ),
        maximum_work_units=policy.limits.maximum_work_units,
        maximum_memory_bytes=policy.limits.maximum_scratch_bytes,
    )
    metric = np.broadcast_to(
        np.eye(3, dtype=np.float64) / 0.8**2, (source.mesh.coordinates.shape[0], 3, 3)
    )
    native = sphere_native_scope(policy.limits)
    with native as native_scope, native_coordinate_phase(native_scope, ledger):
        outcome = execute_sphere_metric_adaptation(
            source,
            metric,
            domain,
            atlas,
            transfer,
            policy=policy,
            coordinate_contract=source.coordinate_contract,
            maximum_fidelity=0.16,
            minimum_metric_quality=minimum_metric_quality,
            topology_operations=True,
            relocation=False,
            budget=ledger,
            scope=native_scope,
        )
    assert outcome.evidence.metric_quality_floor == minimum_metric_quality
    assert (
        not outcome.evidence.converged
        or outcome.evidence.minimum_metric_quality >= minimum_metric_quality
    )
    assert (
        sum((outcome.evidence.splits, outcome.evidence.collapses, outcome.evidence.flips))
        > 0
    )
    assert outcome.target_mesh.topology_id != source.mesh.topology_id
    assert outcome.reconstruction.target_validity.all_certified
    assert outcome.reconstruction.target_embedding.status == "certified"
    outcome.reconstruction.require_bound(atlas, outcome.target_mesh)
    outcome.deformation.require_bound(
        source.mesh, source.geometry, outcome.target_mesh, outcome.reconstruction.geometry
    )
    with coordinate_reference_budget() as budget:
        target, lineage, _ = assemble_topology_edit(
            source.mesh,
            outcome.edit,
            numeric_version=outcome.target_mesh.numeric_version,
            budget=budget,
        )
    patches, zones, labels = inherit_mesh_organization(source, target, lineage)
    assert tuple(value.name for value in patches) == tuple(
        value.name for value in source.patches
    )
    assert tuple(value.name for value in zones) == tuple(
        value.name for value in source.zones
    )
    assert tuple(value.name for value in labels) == tuple(
        value.name for value in source.labels
    )
    assert all(
        value == Fraction(1, 2) for value in outcome.deformation.source_reference_coverage
    )
    assert all(
        value == Fraction(1, 2) for value in outcome.deformation.target_reference_coverage
    )
    assert outcome.evidence.maximum_fidelity_bound <= 0.16
    bounds = np.asarray(outcome.evidence.metric_arc_bounds)
    within_goal = np.all(
        (bounds[:, 0] >= 1 / np.sqrt(2.0)) & (bounds[:, 1] <= np.sqrt(2.0))
    )
    within_goal &= outcome.evidence.minimum_metric_quality >= 0.05
    assert (outcome.evidence.status is MetricRemeshingStatus.COMPLETE) == bool(
        within_goal
    )
    original_ids = set(map(int, np.asarray(source.mesh.vertex_global_ids)))
    assert set(outcome.exact_stencils) == set(
        map(int, np.asarray(outcome.target_mesh.vertex_global_ids))
    )
    assert all(
        sum((weight for _, weight in key), Fraction(0)) == 1
        and all(identifier in original_ids and weight > 0 for identifier, weight in key)
        for key in outcome.exact_stencils.values()
    )
    # Independent affine endpoint maps and homogeneous point evaluation check
    # the full-map bound, not the radial source image shared by both atlases.
    old_corners = np.concatenate(
        [
            np.asarray(source.mesh.coordinates)[np.asarray(block.vertices)]
            for block in source.mesh.blocks
        ]
    )
    new_corners = np.concatenate(
        [
            np.asarray(outcome.target_mesh.coordinates)[np.asarray(block.vertices)]
            for block in outcome.target_mesh.blocks
        ]
    )
    for piece in outcome.deformation.pieces:
        vertices = np.asarray(piece.source_reference_vertices)
        references = np.concatenate(
            (
                vertices,
                np.mean(vertices, axis=0)[None],
                (np.asarray((0.2, 0.3, 0.5), dtype=np.float64) @ vertices)[None],
            )
        )
        homogeneous = (
            np.column_stack((np.ones(references.shape[0], dtype=np.float64), references))
            @ np.asarray(piece.projective_map.coefficients).T
        )
        target_references = homogeneous[:, 1:] / np.sum(homogeneous, axis=1)[:, None]
        old_weights = np.column_stack((1.0 - np.sum(references, axis=1), references))
        new_weights = np.column_stack(
            (1.0 - np.sum(target_references, axis=1), target_references)
        )
        displacement = np.linalg.norm(
            old_weights @ old_corners[piece.source_row]
            - new_weights @ new_corners[piece.target_row],
            axis=1,
        )
        assert (
            np.max(displacement)
            <= piece.displacement_bound + 128 * np.finfo(np.float64).eps
        )
    assert ledger.native_charged_work_units == ledger.work_units
    assert native.evidence is not None
    # Native clipping and other FFI stages have separate work; only the
    # coordinate attribution counter must equal this source host ledger.
    assert (
        ledger.work_units
        <= native.evidence.externally_charged_work
        <= policy.limits.maximum_work_units
    )
    np.testing.assert_array_equal(np.asarray(source.geometry.coordinates), before)


def test_sphere_interior_split_collapse_preserves_current_organization(
    sphere_metric_source: SphereMetricSource,
) -> None:
    from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
    from phydrax.meshing._host_banks import selected_rows

    domain, source, atlas, transfer, policy = sphere_metric_source
    metric = np.broadcast_to(np.eye(3), (source.mesh.coordinates.shape[0], 3, 3))
    with (
        sphere_native_scope(policy.limits) as native_scope,
        native_coordinate_resources(native_scope, policy.limits) as ledger,
    ):
        state, background = owner._sphere_initial_state(
            source, metric, atlas, transfer, policy, 0.16, budget=ledger
        )
        original_cells = {tuple(sorted(map(int, cell))) for cell in state.cells}
        original_features = dict(state.features)
        born = state.points.shape[0]
        for pair in owner._sphere_edges(state)[0]:
            edge = tuple(map(int, pair))
            if edge in state.features or edge in state.blocked:
                continue
            closures = [
                background.vertex_closure_memberships[int(state.vertex_ids[vertex])]
                for vertex in edge
            ]
            if closures[0] == closures[1]:
                continue
            interior = [
                vertex
                for vertex in edge
                if vertex not in state.fixed
                and not any(vertex in feature for feature in state.features)
                and np.unique(state.classes[np.any(state.cells == vertex, axis=1)]).size
                == 1
                and np.unique(state.patches[np.any(state.cells == vertex, axis=1)]).size
                == 1
            ]
            if not interior:
                continue
            if owner._sphere_apply_operation(
                state,
                "split",
                edge,
                domain,
                background,
                transfer,
                budget=ledger,
                scope=native_scope,
            ):
                kept = interior[0]
                break
        else:
            pytest.fail(
                "native sphere must expose an interior edge next to a labeled trace"
            )
        assert (
            len(
                {
                    background.vertex_closure_memberships[parent]
                    for parent in state.vertex_sources[born]
                }
            )
            > 1
        )
        assert owner._sphere_collapse_organization(state, born, kept, background)
        trace_end = edge[1] if edge[0] == kept else edge[0]
        assert any(trace_end in feature for feature in state.features)
        assert not owner._sphere_collapse_organization(state, born, trace_end, background)
        assert not owner._sphere_collapse(
            state,
            born,
            trace_end,
            domain,
            background,
            state.minimum_metric_quality,
            budget=ledger,
        )
        assert owner._sphere_apply_operation(
            state,
            "collapse",
            (born, kept),
            domain,
            background,
            transfer,
            budget=ledger,
            scope=native_scope,
        )
        assert state.features == original_features
        assert {tuple(sorted(map(int, cell))) for cell in state.cells} == original_cells
        edit, _, _, _, _ = owner._sphere_assemble(source, state)
        target, lineage, _ = assemble_topology_edit(
            source.mesh,
            edit,
            numeric_version=canonical_fingerprint(
                array_tree_fingerprint(edit.coordinates)
            ),
            budget=ledger,
        )
        patches, zones, labels = inherit_mesh_organization(source, target, lineage)
    for original, inherited in zip(
        (*source.patches, *source.zones, *source.labels),
        (*patches, *zones, *labels),
        strict=True,
    ):
        assert original.name == inherited.name
        old_keys = entity_keys(source.mesh, original.scope.entity_dimension)[
            selected_rows(source.mesh, original.scope)
        ]
        new_keys = entity_keys(target, inherited.scope.entity_dimension)[
            selected_rows(target, inherited.scope)
        ]
        if original.scope.entity_dimension < 2:
            assert set(map(tuple, old_keys.tolist())) == set(
                map(tuple, new_keys.tolist())
            )
        else:

            def physical_cell_keys(
                mesh: CellMesh,
                identifiers: np.ndarray,
            ) -> set[tuple[int, ...]]:
                selected = set(map(int, identifiers[:, 0]))
                vertex_ids = np.asarray(mesh.vertex_global_ids)
                return {
                    tuple(sorted(map(int, vertex_ids[corners])))
                    for block in mesh.blocks
                    for identifier, corners in zip(
                        np.asarray(block.global_ids),
                        np.asarray(block.vertices),
                        strict=True,
                    )
                    if int(identifier) in selected
                }

            assert physical_cell_keys(source.mesh, old_keys) == physical_cell_keys(
                target, new_keys
            )


def test_sphere_midpoint_collapse_relocates_interior_kept_vertex_with_lineage(
    sphere_metric_source: SphereMetricSource,
) -> None:
    from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
    from phydrax.meshing._host_banks import selected_rows

    domain, source, atlas, transfer, policy = sphere_metric_source
    # Coarse metric: interior edges are short, so a merged star stays below
    # the upper length obligation.
    metric = np.broadcast_to(np.eye(3) / 4.0, (source.mesh.coordinates.shape[0], 3, 3))
    with (
        sphere_native_scope(policy.limits) as native_scope,
        native_coordinate_resources(native_scope, policy.limits) as ledger,
    ):
        state, background = owner._sphere_initial_state(
            source, metric, atlas, transfer, policy, 0.16, budget=ledger
        )
        original_features = dict(state.features)
        cell_count = state.cells.shape[0]
        for pair in owner._sphere_edges(state)[0]:
            removed, kept = map(int, pair)
            before = state.points[kept].copy()
            if owner._sphere_midpoint_collapse(
                state,
                removed,
                kept,
                domain,
                background,
                state.minimum_metric_quality,
                budget=ledger,
            ):
                break
            # A refused trial restores the carrier exactly.
            np.testing.assert_array_equal(state.points[kept], before)
        else:
            pytest.fail("native sphere must admit one interior midpoint collapse")
        assert state.dimensions[removed] == state.dimensions[kept] == 2
        assert not any(
            vertex in feature for feature in state.features for vertex in (removed, kept)
        )
        assert state.features == original_features
        assert state.successors[removed] == kept
        assert kept in state.relocated
        assert state.vertex_sources[kept] == {
            parent for parent, _ in state.stencils[kept]
        }
        assert not np.any(state.cells == removed)
        assert state.cells.shape[0] == cell_count - 2
        assert not np.array_equal(state.points[kept], before)
        edit, _, _, _, _ = owner._sphere_assemble(source, state)
        target, lineage, _ = assemble_topology_edit(
            source.mesh,
            edit,
            numeric_version=canonical_fingerprint(
                array_tree_fingerprint(edit.coordinates)
            ),
            budget=ledger,
        )
        patches, zones, labels = inherit_mesh_organization(source, target, lineage)
    for original, inherited in zip(
        (*source.patches, *source.zones, *source.labels),
        (*patches, *zones, *labels),
        strict=True,
    ):
        assert original.name == inherited.name
        if original.scope.entity_dimension < 2:
            old_keys = entity_keys(source.mesh, original.scope.entity_dimension)[
                selected_rows(source.mesh, original.scope)
            ]
            new_keys = entity_keys(target, inherited.scope.entity_dimension)[
                selected_rows(target, inherited.scope)
            ]
            assert set(map(tuple, old_keys.tolist())) == set(
                map(tuple, new_keys.tolist())
            )


def test_sphere_invalid_cavity_is_rejected_before_rational_cone_construction(
    sphere_metric_source: SphereMetricSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.geometry._sphere_material_atlas as material_owner

    domain, source, atlas, transfer, policy = sphere_metric_source
    metric = np.broadcast_to(np.eye(3), (source.mesh.coordinates.shape[0], 3, 3))
    with (
        sphere_native_scope(policy.limits) as native_scope,
        native_coordinate_resources(native_scope, policy.limits) as ledger,
    ):
        state, background = owner._sphere_initial_state(
            source, metric, atlas, transfer, policy, 0.16, budget=ledger
        )
        cells = state.cells[:2].copy()
        cells[1] = cells[1, [0, 2, 1]]

        def forbid_bound(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("an inverted cavity must not construct cone bounds")

        monkeypatch.setattr(material_owner, "sphere_radial_triangle_bounds", forbid_bound)
        before = ledger.work_units
        assert not owner._sphere_legal(
            domain,
            cells,
            state.patches[:2],
            state.directions,
            points=state.points,
            background=background,
            maximum_fidelity=0.16,
            budget=ledger,
        )
        assert ledger.work_units == before


@pytest.mark.parametrize("orientation", [1, -1])
def test_sphere_interval_normalization_encloses_the_exact_cone_bound(
    orientation: int,
) -> None:
    from phydrax.geometry._sphere_material_atlas import sphere_radial_triangle_bounds

    rays = np.asarray(((1.0, 0.0, 0.0), (0.9, 0.3, 0.0), (0.9, 0.0, 0.3)))
    if orientation < 0:
        rays = rays[[0, 2, 1]]
    source_norm = 1.3
    with coordinate_reference_budget() as budget:
        filtered = owner._sphere_radial_normalization_upper(
            rays, orientation, source_norm, budget=budget
        )
        assert budget.work_units == 1
        exact = sphere_radial_triangle_bounds(rays, budget=budget)
    assert exact is not None and filtered is not None
    determinant, floor, ceiling, unit_error = exact
    assert determinant * orientation > 0
    normalization = Fraction(source_norm) * (
        max(1 - Fraction(floor), Fraction(ceiling) - 1, Fraction(0)) + unit_error
    )
    assert Fraction(filtered) >= normalization


def test_sphere_interval_normalization_preserves_exact_uncertainty_fallback() -> None:
    rays = np.asarray(
        ((1.0, 1.0, 1.0), (1.0, 2.0, 3.0), (2.0, 3.0, np.nextafter(4.0, np.inf)))
    )
    with coordinate_reference_budget() as budget:
        assert (
            owner._sphere_radial_normalization_upper(rays, 1, 1.0, budget=budget) is None
        )


def test_sphere_filtered_legality_preserves_exact_candidate_decision(
    sphere_metric_source: SphereMetricSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domain, source, atlas, transfer, policy = sphere_metric_source
    metric = np.broadcast_to(np.eye(3), (source.mesh.coordinates.shape[0], 3, 3))
    with (
        sphere_native_scope(policy.limits) as native_scope,
        native_coordinate_resources(native_scope, policy.limits) as ledger,
    ):
        state, background = owner._sphere_initial_state(
            source, metric, atlas, transfer, policy, 0.16, budget=ledger
        )
        before = ledger.work_units
        assert owner._sphere_legal(
            domain,
            state.cells,
            state.patches,
            state.directions,
            points=state.points,
            background=background,
            maximum_fidelity=0.16,
            budget=ledger,
        )
        filtered_work = ledger.work_units - before
        monkeypatch.setattr(
            owner, "_sphere_radial_normalization_upper", lambda *a, **kw: None
        )
        before = ledger.work_units
        assert owner._sphere_legal(
            domain,
            state.cells,
            state.patches,
            state.directions,
            points=state.points,
            background=background,
            maximum_fidelity=0.16,
            budget=ledger,
        )
        assert filtered_work < ledger.work_units - before


def test_sphere_relocation_preserves_an_already_conforming_live_star(
    sphere_metric_source: SphereMetricSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domain, source, atlas, transfer, policy = sphere_metric_source
    metric = np.broadcast_to(np.eye(3), (source.mesh.coordinates.shape[0], 3, 3))
    with (
        sphere_native_scope(policy.limits) as native_scope,
        native_coordinate_resources(native_scope, policy.limits) as ledger,
    ):
        state, background = owner._sphere_initial_state(
            source, metric, atlas, transfer, policy, 0.16, budget=ledger
        )
        for candidate in np.unique(state.cells):
            vertex = int(candidate)
            if (
                vertex in state.fixed
                or state.dimensions[vertex] != 2
                or any(vertex in edge for edge in state.features)
            ):
                continue
            rows = np.flatnonzero(np.any(state.cells == vertex, axis=1))
            if (
                np.unique(state.patches[rows]).size != 1
                or np.unique(state.classes[rows]).size != 1
            ):
                continue
            neighbors = np.unique(state.cells[rows])
            neighbors = neighbors[neighbors != vertex]
            lengths = np.linalg.norm(
                state.points[neighbors] - state.points[vertex], axis=1
            )
            lower = owner._LOWER / np.min(lengths)
            upper = owner._UPPER / np.max(lengths)
            if lower >= upper:
                continue
            state.metric[:] = np.eye(3) * ((lower + upper) / 2) ** 2
            if np.all(
                owner._sphere_quality(
                    state, state.cells[rows], state.patches[rows], background
                )
                >= state.minimum_metric_quality
            ):
                break
        else:
            pytest.fail("native sphere must contain a conforming interior star")

        def forbid_proposal(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("a conforming star needs no source relocation query")

        monkeypatch.setattr(owner, "_sphere_point", forbid_proposal)
        before = state.points.copy()
        work = ledger.work_units
        assert not owner._sphere_relocate(
            state, vertex, domain, background, transfer, budget=ledger
        )
        np.testing.assert_array_equal(state.points, before)
        assert ledger.work_units == work
        # Admission is recomputed from live metric values, not remembered as a
        # permanent exemption for this source vertex.
        state.metric *= 4
        with pytest.raises(AssertionError, match="source relocation query"):
            owner._sphere_relocate(
                state, vertex, domain, background, transfer, budget=ledger
            )


@pytest.mark.parametrize(
    "source_dimension", [1, 2], ids=["source-curve", "source-surface"]
)
def test_sphere_relocation_certification_refusal_restores_trial_carrier(
    sphere_metric_source: SphereMetricSource,
    monkeypatch: pytest.MonkeyPatch,
    source_dimension: int,
) -> None:
    domain, source, atlas, transfer, policy = sphere_metric_source
    metric = np.broadcast_to(
        np.eye(3, dtype=np.float64), (source.mesh.coordinates.shape[0], 3, 3)
    )
    with (
        sphere_native_scope(policy.limits) as native_scope,
        native_coordinate_resources(native_scope, policy.limits) as ledger,
    ):
        state, background = owner._sphere_initial_state(
            source, metric, atlas, transfer, policy, 0.16, budget=ledger
        )
        if source_dimension == 1:
            # Uniform authored curve intervals have no improving relocation.
            # Earn a nonuniform carrier with a real certified feature split.
            edge = next(
                edge
                for edge, curve in sorted(state.features.items())
                if curve >= 0
                and edge not in state.blocked
                and all(
                    vertex not in state.fixed
                    and state.dimensions[vertex] == 1
                    and state.indices[vertex] == curve
                    for vertex in edge
                )
                and np.unique(
                    state.patches[np.sum(np.isin(state.cells, edge), axis=1) == 2]
                ).size
                == 1
            )
            vertices_before, cells_before = state.points.shape[0], state.cells.shape[0]
            assert owner._sphere_apply_operation(
                state,
                "split",
                edge,
                domain,
                background,
                transfer,
                budget=ledger,
                scope=native_scope,
            )
            assert state.points.shape[0] == vertices_before + 1
            assert state.cells.shape[0] == cells_before + 2
            assert state.dimensions[vertices_before] == 1
        points, directions, tensors = (
            state.points.copy(),
            state.directions.copy(),
            state.metric.copy(),
        )
        parameters = tuple(value.copy() for value in state.parameters)
        stencils = tuple(state.stencils)
        ancestry = tuple(frozenset(value) for value in state.vertex_sources)
        kinds = dict(state.kinds)
        relocated = set(state.relocated)
        work_before = ledger.work_units
        refused_vertices: list[int] = []

        def refuse(*args: Any, **kwargs: Any) -> bool:
            # The production merit gate, not this test, admitted the real trial.
            # Prove that the exception occurs after numeric installation.
            assert any(
                not np.array_equal(actual, original)
                for actual, original in (
                    (state.points, points),
                    (state.directions, directions),
                    (state.metric, tensors),
                )
            )
            refused_vertices.append(vertex)
            raise ValueError(
                "surface_metric.certification must accept the trial carrier; got deliberate certification refusal"
            )

        monkeypatch.setattr(owner, "_sphere_legal", refuse)
        with pytest.raises(ValueError, match="surface_metric.certification"):
            for candidate in np.unique(state.cells):
                vertex = int(candidate)
                if vertex in state.fixed or state.dimensions[vertex] != source_dimension:
                    continue
                rows = np.flatnonzero(np.any(state.cells == vertex, axis=1))
                if (
                    np.unique(state.patches[rows]).size != 1
                    or np.unique(state.classes[rows]).size != 1
                ):
                    continue
                incident = [edge for edge in state.features if vertex in edge]
                if source_dimension == 1:
                    if len(incident) != 2 or any(
                        edge in state.blocked
                        or state.features[edge] != state.indices[vertex]
                        for edge in incident
                    ):
                        continue
                elif incident:
                    continue
                # Reuse the production proposal/merit implementation.
                # Nonimproving proposals are allowed to reject, never forced in.
                try:
                    changed = owner._sphere_relocate(
                        state, vertex, domain, background, transfer, budget=ledger
                    )
                finally:
                    np.testing.assert_array_equal(state.points, points)
                    np.testing.assert_array_equal(state.directions, directions)
                    np.testing.assert_array_equal(state.metric, tensors)
                    for actual, original in zip(
                        state.parameters, parameters, strict=True
                    ):
                        np.testing.assert_array_equal(actual, original)
                    assert tuple(state.stencils) == stencils
                    assert (
                        tuple(frozenset(value) for value in state.vertex_sources)
                        == ancestry
                    )
                    assert state.kinds == kinds
                    assert state.relocated == relocated
                    assert vertex not in state.relocated
                assert not changed
            pytest.fail(
                "surface_metric.fixture.relocation must contain a naturally merit-admitted "
                f"dimension-{source_dimension} trial under the original sphere/.8, "
                "identity metric and fidelity .16 controls; got no qualifying proposal"
            )
        assert len(refused_vertices) == 1
        assert state.dimensions[refused_vertices[0]] == source_dimension
        assert ledger.work_units > work_before


def test_sphere_carrier_operation_admits_its_actual_cavity_before_mutation(
    sphere_metric_source: SphereMetricSource,
) -> None:
    domain, source, atlas, transfer, policy = sphere_metric_source
    ledger = CoordinateEnclosureBudget(
        HostBatchTicket(
            policy.limits.maximum_work_units,
            bytes_upper=policy.limits.maximum_scratch_bytes,
        ),
        maximum_work_units=policy.limits.maximum_work_units,
        maximum_memory_bytes=policy.limits.maximum_scratch_bytes,
    )
    metric = np.broadcast_to(
        np.eye(3, dtype=np.float64), (source.mesh.coordinates.shape[0], 3, 3)
    )
    with (
        sphere_native_scope(policy.limits) as native_scope,
        native_coordinate_phase(native_scope, ledger),
    ):
        state, background = owner._sphere_initial_state(
            source, metric, atlas, transfer, policy, 0.16, budget=ledger
        )
    vertex = int(state.cells[0, 0])
    before = state.points.copy()
    cells = state.cells.copy()
    stencils = tuple(state.stencils)
    work_before = ledger.work_units
    native = NativeExecutionScope(
        NativeExecutionLimits(
            policy.limits.maximum_work_units,
            policy.limits.maximum_geometry_queries,
            0,
            policy.limits.maximum_scratch_bytes,
            policy.limits.maximum_wall_seconds,
        )
    )
    with (
        pytest.raises(MeshcoreError) as refused,
        native as native_scope,
        native_coordinate_phase(native_scope, ledger),
    ):
        owner._sphere_apply_operation(
            state,
            "relocate",
            (vertex,),
            domain,
            background,
            transfer,
            budget=ledger,
            scope=native_scope,
        )
    assert refused.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
    np.testing.assert_array_equal(state.points, before)
    np.testing.assert_array_equal(state.cells, cells)
    assert tuple(state.stencils) == stencils
    assert ledger.work_units == work_before


@pytest.mark.parametrize(
    "source_dimension", [1, 2], ids=["source-curve", "source-surface"]
)
def test_public_sphere_relocation_refuses_uncertified_source_without_mutation(
    sphere_metric_source: SphereMetricSource,
    source_dimension: int,
) -> None:
    _, source, _, _, policy = sphere_metric_source
    before = np.asarray(source.geometry.coordinates).copy()
    vertex_association = next(
        value
        for value in source.associations
        if value.target_entity_set_id == source.mesh.entity_set(0).entity_set_id
    )
    rows = np.flatnonzero(
        np.asarray(vertex_association.source_dimensions) == source_dimension
    )
    assert rows.size
    parameters = np.asarray(vertex_association.parameters)[rows].copy()
    uncertified = eqx.tree_at(lambda value: value.certification, source, None)
    vertices = source.mesh.entity_set(0)
    scope = MeshingScope(
        source.mesh.mesh_id,
        source.mesh.numeric_version,
        MeshingEntityKind.MESH,
        0,
        vertices.entity_set_id,
        vertices.entity_ids,
    )
    metric = MeshMetricField(
        scope,
        np.broadcast_to(
            np.eye(3, dtype=np.float64), (source.mesh.coordinates.shape[0], 3, 3)
        ),
        minimum_size=1.0,
        maximum_size=1.0,
    )
    with pytest.raises(ValueError, match="MeshAdaptation.source.certification"):
        prepare_mesh_adaptation(
            uncertified,
            phx.meshing.RelocationMeshAdaptation(
                metric, coordinate_contract=source.coordinate_contract
            ),
            policy=policy,
        )
    np.testing.assert_array_equal(source.geometry.coordinates, before)
    np.testing.assert_array_equal(vertex_association.parameters[rows], parameters)
    assert uncertified.mesh.mesh_id == source.mesh.mesh_id
    assert uncertified.geometry.geometry_layout_id == source.geometry.geometry_layout_id
    assert tree_content_equal(uncertified.associations, source.associations)


def test_public_sphere_cavity_refusal_preserves_source_and_reports_allowance(
    sphere_metric_source: SphereMetricSource,
) -> None:
    _, source, _, transfer, policy = sphere_metric_source
    before = np.asarray(source.geometry.coordinates).copy()
    limits = MeshingLimits(
        maximum_vertices=20_000,
        maximum_edges=160_000,
        maximum_faces=160_000,
        maximum_cells=160_000,
        maximum_work_units=640_000,
        maximum_geometry_queries=1_280_000,
        maximum_cavity_cells=1,
        maximum_scratch_bytes=81_920_000,
    )
    bounded = MeshAdaptationPolicy(
        MeshAdaptationRoute.NATIVE_SURFACE_METRIC,
        association_transfer=transfer,
        maximum_passes=policy.maximum_passes,
        relocation=True,
        limits=limits,
        geometry_transition=policy.geometry_transition,
    )
    vertices = source.mesh.entity_set(0)
    scope = MeshingScope(
        source.mesh.mesh_id,
        source.mesh.numeric_version,
        MeshingEntityKind.MESH,
        0,
        vertices.entity_set_id,
        vertices.entity_ids,
    )
    metric = MeshMetricField(
        scope,
        np.broadcast_to(
            np.eye(3, dtype=np.float64) / 0.07**2,
            (source.mesh.coordinates.shape[0], 3, 3),
        ),
        minimum_size=0.07,
        maximum_size=0.07,
    )
    result = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            source,
            phx.meshing.RelocationMeshAdaptation(
                metric, coordinate_contract=source.coordinate_contract
            ),
            policy=bounded,
        )
    )
    assert result.target.result_id == source.result_id
    assert result.transition is None and result.lineage is None and result.stencil is None
    assert isinstance(result.evidence, MetricRemeshingEvidence)
    assert result.evidence.status is MetricRemeshingStatus.STALLED
    assert result.evidence.rejected_operations == source.mesh.coordinates.shape[0]
    assert result.evidence.operation_attempts == 0
    assert (
        result.evidence.splits
        == result.evidence.collapses
        == result.evidence.flips
        == result.evidence.relocations
        == 0
    )
    assert result.evidence.execution_evidence is not None
    assert result.evidence.execution_evidence.total_work_units <= 640_000
    assert result.policy.limits.maximum_cavity_cells == 1
    np.testing.assert_array_equal(source.geometry.coordinates, before)


def test_native_sphere_metric_zero_ledger_refuses_without_source_mutation(
    sphere_metric_source: SphereMetricSource,
) -> None:
    domain, source, atlas, transfer, policy = sphere_metric_source
    before = np.asarray(source.geometry.coordinates).copy()
    ledger = CoordinateEnclosureBudget(
        HostBatchTicket(0, bytes_upper=0), maximum_work_units=0, maximum_memory_bytes=0
    )
    with (
        sphere_native_scope(policy.limits) as native_scope,
        pytest.raises(CoordinateEnclosureResourceError),
        native_coordinate_phase(native_scope, ledger),
    ):
        execute_sphere_metric_adaptation(
            source,
            np.broadcast_to(
                np.eye(3, dtype=np.float64), (source.mesh.coordinates.shape[0], 3, 3)
            ),
            domain,
            atlas,
            transfer,
            policy=policy,
            coordinate_contract=source.coordinate_contract,
            maximum_fidelity=0.16,
            topology_operations=True,
            relocation=False,
            budget=ledger,
            scope=native_scope,
        )
    assert ledger.work_units == ledger.native_charged_work_units == 0
    np.testing.assert_array_equal(np.asarray(source.geometry.coordinates), before)


def test_native_sphere_metric_stale_source_revision_refuses_without_mutation(
    sphere_metric_source: SphereMetricSource,
) -> None:
    _, source, atlas, transfer, policy = sphere_metric_source
    before = np.asarray(source.geometry.coordinates).copy()
    ledger = CoordinateEnclosureBudget(
        HostBatchTicket(
            policy.limits.maximum_work_units,
            bytes_upper=policy.limits.maximum_scratch_bytes,
        ),
        maximum_work_units=policy.limits.maximum_work_units,
        maximum_memory_bytes=policy.limits.maximum_scratch_bytes,
    )
    with (
        pytest.raises(ValueError, match="stale"),
        sphere_native_scope(policy.limits) as native_scope,
        native_coordinate_phase(native_scope, ledger),
    ):
        execute_sphere_metric_adaptation(
            source,
            np.broadcast_to(
                np.eye(3, dtype=np.float64), (source.mesh.coordinates.shape[0], 3, 3)
            ),
            sphere_surface(revision="changed").domain,
            atlas,
            transfer,
            policy=policy,
            coordinate_contract=source.coordinate_contract,
            maximum_fidelity=0.16,
            topology_operations=True,
            relocation=False,
            budget=ledger,
            scope=native_scope,
        )
    assert ledger.work_units == 0
    np.testing.assert_array_equal(np.asarray(source.geometry.coordinates), before)


@pytest.mark.parametrize("nested", (False, True), ids=("root", "nested"))
@pytest.mark.parametrize("work_limit", (1_000, 640_000), ids=("refusal", "stalled"))
def test_public_sphere_metric_handoff_preserves_refusal_or_stall(
    sphere_metric_source: SphereMetricSource,
    monkeypatch: pytest.MonkeyPatch,
    nested: bool,
    work_limit: int,
) -> None:

    _, source, _, transfer, policy = sphere_metric_source
    before = np.asarray(source.geometry.coordinates).copy()
    limits = MeshingLimits(
        maximum_vertices=20_000,
        maximum_edges=160_000,
        maximum_faces=160_000,
        maximum_cells=160_000,
        maximum_work_units=work_limit,
        maximum_geometry_queries=1_280_000,
        maximum_scratch_bytes=81_920_000,
    )
    starved = MeshAdaptationPolicy(
        MeshAdaptationRoute.NATIVE_SURFACE_METRIC,
        association_transfer=transfer,
        maximum_passes=policy.maximum_passes,
        relocation=False,
        limits=limits,
        geometry_transition=policy.geometry_transition,
    )
    vertices = source.mesh.entity_set(0)
    scope = MeshingScope(
        source.mesh.mesh_id,
        source.mesh.numeric_version,
        MeshingEntityKind.MESH,
        0,
        vertices.entity_set_id,
        vertices.entity_ids,
    )
    metric = MeshMetricField(
        scope,
        np.broadcast_to(
            np.eye(3, dtype=np.float64) / 0.8**2, (source.mesh.coordinates.shape[0], 3, 3)
        ),
        minimum_size=0.8,
        maximum_size=0.8,
    )
    from phydrax.meshing import (
        _adaptation as adaptation_owner,
        _native_preparation as preparation_owner,
    )

    ended = []
    retained = []
    execute = adaptation_owner._execute_mesh_routes
    remember = preparation_owner.NativePreparationLedger.remember_receipt

    def capture_outcome(*args: Any, **kwargs: Any) -> Any:
        outcome = execute(*args, **kwargs)
        ended.append(outcome)
        return outcome

    def capture_receipt(self: Any, parent: Any, record: Any, **kwargs: Any) -> None:
        if ended:
            retained.append(record)
        remember(self, parent, record, **kwargs)

    monkeypatch.setattr(adaptation_owner, "_execute_mesh_routes", capture_outcome)
    monkeypatch.setattr(
        preparation_owner.NativePreparationLedger, "remember_receipt", capture_receipt
    )
    parent = NativeExecutionScope(limits.native_limits()) if nested else None
    with parent if parent is not None else nullcontext():
        result = execute_mesh_adaptation(
            prepare_mesh_adaptation(
                source,
                MetricMeshAdaptation(
                    metric, coordinate_contract=source.coordinate_contract
                ),
                policy=starved,
            )
        )

    assert len(ended) == 1
    evidence = result.evidence
    assert isinstance(evidence, MetricRemeshingEvidence)
    assert evidence.execution_evidence is not None
    assert result.target.execution_evidence is not None
    assert tree_content_equal(
        result.target.execution_evidence, evidence.execution_evidence
    )
    if work_limit == 1_000:
        assert result.status is MeshAdaptationStatus.RESOURCE_LIMIT
        assert result.target.result_id == source.result_id
        assert evidence.status is MetricRemeshingStatus.RESOURCE_LIMIT
        assert evidence.resource_message == ended[0].evidence.resource_message
        assert evidence.resource_requested == ended[0].evidence.resource_requested
        assert evidence.resource_achieved == ended[0].evidence.resource_achieved
        assert evidence.resource_message
        assert dict(evidence.resource_requested)["maximum_work_units"] == work_limit
        assert not retained
    else:
        assert result.status is MeshAdaptationStatus.STALLED
        assert not retained
    np.testing.assert_array_equal(np.asarray(source.geometry.coordinates), before)


def test_public_native_sphere_query_refusal_retains_ended_status_and_source(
    sphere_metric_source: SphereMetricSource,
) -> None:

    _, source, _, transfer, policy = sphere_metric_source
    before = np.asarray(source.geometry.coordinates).copy()
    limits = MeshingLimits(
        maximum_vertices=20_000,
        maximum_edges=160_000,
        maximum_faces=160_000,
        maximum_cells=160_000,
        maximum_work_units=640_000,
        maximum_geometry_queries=1,
        maximum_scratch_bytes=81_920_000,
    )
    starved = MeshAdaptationPolicy(
        MeshAdaptationRoute.NATIVE_SURFACE_METRIC,
        association_transfer=transfer,
        maximum_passes=policy.maximum_passes,
        relocation=False,
        limits=limits,
        geometry_transition=policy.geometry_transition,
    )
    vertices = source.mesh.entity_set(0)
    scope = phx.meshing.MeshingScope(
        source.mesh.mesh_id,
        source.mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        0,
        vertices.entity_set_id,
        vertices.entity_ids,
    )
    metric = phx.meshing.MeshMetricField(
        scope,
        np.broadcast_to(
            np.eye(3, dtype=np.float64) / 0.8**2, (source.mesh.coordinates.shape[0], 3, 3)
        ),
        minimum_size=0.8,
        maximum_size=0.8,
    )
    result = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source,
            phx.meshing.MetricMeshAdaptation(
                metric, coordinate_contract=source.coordinate_contract
            ),
            policy=starved,
        )
    )
    assert result.status is phx.meshing.MeshAdaptationStatus.RESOURCE_LIMIT
    assert result.target.result_id == source.result_id
    assert isinstance(result.evidence, MetricRemeshingEvidence)
    assert result.evidence.execution_evidence is not None
    assert (
        int(result.evidence.execution_evidence.status) == MeshcoreStatus.CAPACITY_EXCEEDED
    )

    requested = dict(result.evidence.resource_requested)
    achieved = dict(result.evidence.resource_achieved)
    queries_upper = sphere_material_strata(
        sphere_metric_source.domain, source
    ).queries_upper()
    assert requested["native_batch:requested_geometry_queries"] == queries_upper
    assert queries_upper > limits.maximum_geometry_queries
    assert requested["maximum_geometry_queries"] == limits.maximum_geometry_queries
    assert int(result.evidence.execution_evidence.total_geometry_queries) == 0
    assert achieved["native_batch:refusal:5"] == 0
    np.testing.assert_array_equal(source.geometry.coordinates, before)


def test_public_fine_sphere_metric_preserves_original_all_phase_capacity(
    fine_sphere_metric_source: SphereMetricSource,
) -> None:

    _, source, _, _, policy = fine_sphere_metric_source
    before = np.asarray(source.geometry.coordinates).copy()
    vertices = source.mesh.entity_set(0)
    scope = phx.meshing.MeshingScope(
        source.mesh.mesh_id,
        source.mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        0,
        vertices.entity_set_id,
        vertices.entity_ids,
    )
    metric = phx.meshing.MeshMetricField(
        scope,
        np.broadcast_to(
            np.eye(3, dtype=np.float64), (source.mesh.coordinates.shape[0], 3, 3)
        ),
        minimum_size=1.0,
        maximum_size=1.0,
    )
    adaptation = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source,
            phx.meshing.MetricMeshAdaptation(
                metric, coordinate_contract=source.coordinate_contract
            ),
            policy=policy,
        )
    )
    evidence = adaptation.evidence
    if not isinstance(evidence, MetricRemeshingEvidence):
        raise TypeError(
            "surface_metric.test.result must retain its declared scientific contract; got the fine sphere route requires its metric evidence"
        )
    assert adaptation.status is not phx.meshing.MeshAdaptationStatus.RESOURCE_LIMIT, (
        dict(evidence.resource_requested),
        dict(evidence.resource_achieved),
        evidence.resource_message,
        evidence.execution_evidence,
    )
    assert adaptation.target.mesh.topology_id != source.mesh.topology_id
    assert adaptation.geometry_transition is not None
    assert adaptation.geometry_transition.chart_deformation is not None
    assert (
        adaptation.target.certification is not None
        and adaptation.target.certification.passed
    )
    assert adaptation.target.audit.passed
    assert evidence.status.value == adaptation.status.value
    assert evidence.execution_evidence is not None
    total_work = int(evidence.execution_evidence.total_work_units)
    coordinate_charge = int(evidence.execution_evidence.externally_charged_work)
    native_work = total_work - coordinate_charge
    assert evidence.work_units + native_work <= 640_000
    assert total_work <= 640_000
    np.testing.assert_array_equal(source.geometry.coordinates, before)


def test_live_coefficient_bank_cannot_evade_memory_limit_with_recycled_wrappers() -> None:
    # All source polynomials stay live simultaneously; short-lived argument
    # tuples must not make later scientific coefficients invisible to the ledger.
    polynomials = [{(index,): Fraction(index + 1)} for index in range(100)]
    ledger = CoordinateEnclosureBudget(
        HostBatchTicket(10000, bytes_upper=10000),
        maximum_work_units=10000,
        maximum_memory_bytes=10000,
    )
    with pytest.raises(CoordinateEnclosureResourceError), host_coordinate_phase(ledger):
        for polynomial in polynomials:
            ledger.retain_basis((polynomial,))


def test_rational_derivative_batch_reuses_only_exact_boxes_under_original_budget() -> (
    None
):

    patch = BSplineSurfacePatch(
        np.asarray(
            (((0.0, 0.0, 0.0), (0.0, 1.0, 0.05)), ((1.0, 0.0, 0.1), (1.0, 1.0, 0.2))),
            dtype=np.float64,
        ),
        np.asarray(((1.0, 0.9), (1.1, 1.0)), dtype=np.float64),
        (0.0, 0.0, 1.0, 1.0),
        (0.0, 0.0, 1.0, 1.0),
        1,
        1,
    )
    box = np.asarray(((0.25, 0.25), (0.75, 0.75)), dtype=np.float64)
    neighboring = box.copy()
    neighboring[1, 0] = np.nextafter(neighboring[1, 0], np.inf)
    boxes = np.stack((box, box, neighboring))
    prepared = CoordinateEnclosureBudget(
        HostBatchTicket(640000, bytes_upper=81920000),
        maximum_work_units=640000,
        maximum_memory_bytes=81920000,
    )
    with host_coordinate_phase(prepared):
        lower, upper = patch.derivative_bounds_batch(boxes, order=2, budget=prepared)
    cold_work = 0
    for row, query in enumerate(boxes):
        cold = CoordinateEnclosureBudget(
            HostBatchTicket(640000, bytes_upper=81920000),
            maximum_work_units=640000,
            maximum_memory_bytes=81920000,
        )
        with host_coordinate_phase(cold):
            expected_lower, expected_upper = patch.derivative_bounds_batch(
                query[None], order=2, budget=cold
            )
        np.testing.assert_array_equal(lower[row], expected_lower[0])
        np.testing.assert_array_equal(upper[row], expected_upper[0])
        cold_work += cold.work_units
    assert 0 < prepared.work_units < cold_work < 640_000
    first_work = prepared.work_units
    first_retained = prepared.retained_basis_bytes
    with host_coordinate_phase(prepared):
        repeated_lower, repeated_upper = patch.derivative_bounds_batch(
            boxes, order=2, budget=prepared
        )
    np.testing.assert_array_equal(repeated_lower, lower)
    np.testing.assert_array_equal(repeated_upper, upper)
    assert 0 < prepared.work_units - first_work < first_work
    assert prepared.retained_basis_bytes == first_retained
    before_first_jet = prepared.work_units
    validation_keys = set(prepared.source_derivative_box_validation_cache)
    basis_ids = set(prepared.retained_objects)
    owner_ids = set(prepared.retained_owner_objects)
    with host_coordinate_phase(prepared):
        prepared_first_lower, prepared_first_upper = patch.derivative_bounds_batch(
            boxes, order=1, budget=prepared
        )
        assert prepared._storage_owner is not None
        assert (
            prepared._storage_owner.bound + prepared._work_storage.bound
            == prepared.retained_basis_bytes + prepared.temporary_bytes_upper
        )
    cold_first = CoordinateEnclosureBudget(
        HostBatchTicket(640000, bytes_upper=81920000),
        maximum_work_units=640000,
        maximum_memory_bytes=81920000,
    )
    with host_coordinate_phase(cold_first):
        cold_first_lower, cold_first_upper = patch.derivative_bounds_batch(
            boxes, order=1, budget=cold_first
        )
    np.testing.assert_array_equal(prepared_first_lower, cold_first_lower)
    np.testing.assert_array_equal(prepared_first_upper, cold_first_upper)
    assert 0 < prepared.work_units - before_first_jet < cold_first.work_units
    # A lower derivative order reuses the jets but owns its distinct C0 mask.
    (validation_key,) = (
        prepared.source_derivative_box_validation_cache.keys() - validation_keys
    )
    assert validation_key[1:] == (1, boxes.tobytes())
    validation = prepared.source_derivative_box_validation_cache[validation_key]
    assert validation.nbytes == boxes.shape[0]
    assert not validation.flags.writeable and validation.flags.owndata
    assert set(prepared.retained_owner_objects) - owner_ids == {id(validation)}
    new_basis = {
        key: value
        for key, value in prepared.retained_objects.items()
        if key not in basis_ids
    }
    (wrapper,) = (
        value
        for value in new_basis.values()
        if isinstance(value, tuple) and len(value) == 1 and value[0] is validation_key
    )
    assert (
        set(new_basis)
        == ({id(wrapper), id(validation_key)} | {id(value) for value in validation_key})
        - basis_ids
    )
    first_jet_retained = prepared.retained_basis_bytes
    assert first_jet_retained > first_retained
    with host_coordinate_phase(prepared):
        repeated_first = patch.derivative_bounds_batch(boxes, order=1, budget=prepared)
    np.testing.assert_array_equal(repeated_first[0], prepared_first_lower)
    np.testing.assert_array_equal(repeated_first[1], prepared_first_upper)
    assert prepared.retained_basis_bytes == first_jet_retained
    validation_reference = weakref.ref(validation)
    starved = CoordinateEnclosureBudget(
        HostBatchTicket(first_work - 1, bytes_upper=81920000),
        maximum_work_units=first_work - 1,
        maximum_memory_bytes=81920000,
    )
    with host_coordinate_phase(starved), pytest.raises(CoordinateEnclosureResourceError):
        patch.derivative_bounds_batch(boxes, order=2, budget=starved)
    altered_points = np.asarray(patch.control_points).copy()
    altered_points[1, 1, 2] += 0.1
    altered = BSplineSurfacePatch(
        altered_points, patch.weights, patch.u_knots, patch.v_knots, 1, 1
    )
    with host_coordinate_phase(prepared):
        changed_lower, changed_upper = altered.derivative_bounds_batch(
            boxes, order=2, budget=prepared
        )
    independent = CoordinateEnclosureBudget(
        HostBatchTicket(640000, bytes_upper=81920000),
        maximum_work_units=640000,
        maximum_memory_bytes=81920000,
    )
    with host_coordinate_phase(independent):
        independent_lower, independent_upper = altered.derivative_bounds_batch(
            boxes, order=2, budget=independent
        )
    np.testing.assert_array_equal(changed_lower, independent_lower)
    np.testing.assert_array_equal(changed_upper, independent_upper)
    assert not np.array_equal(changed_lower, lower)
    del prepared, validation
    gc.collect()
    assert validation_reference() is None


@pytest.fixture(scope="module")
def rational_trim_metric_source() -> UVMetricSource:
    return prepare_rational_trim_metric_source()


def test_rational_source_required_growth_preserves_hard_density_and_geometry(
    rational_trim_metric_source: UVMetricSource,
) -> None:
    _, source, _, _ = rational_trim_metric_source
    requested = dict(source.compliance.requested)
    achieved = dict(source.compliance.achieved)
    assert (
        next(value for name, value in requested.items() if name.endswith(":target_size"))
        == 0.16
    )
    assert (
        next(value for name, value in requested.items() if name.endswith(":maximum_size"))
        == 0.24
    )
    assert (
        0.08
        <= next(value for name, value in achieved.items() if name.endswith(":p50_edge"))
        <= 0.24
    )
    assert (
        next(value for name, value in achieved.items() if name.endswith(":maximum_edge"))
        <= 0.24
    )
    assert (
        next(
            value
            for name, value in requested.items()
            if name.endswith(":maximum_deviation")
        )
        == 0.01
    )
    assert (
        next(
            value
            for name, value in achieved.items()
            if name.endswith(":maximum_deviation")
        )
        <= 0.01
    )
    assert source.compliance.passed and source.audit.passed
    assert source.certification is not None and source.certification.passed
    assert source.quality.sampled_invalid_count == 0
    assert (
        source.quality.minimum_measure > 0.0 and source.quality.minimum_mean_ratio > 0.0
    )
    assert achieved["work_units"] <= 640_000


@pytest.mark.parametrize(
    "component", ["curve-index", "curve-parameter", "entity", "occurrence"]
)
def test_rational_curve_witness_refuses_changed_source_strata(
    rational_trim_metric_source: UVMetricSource,
    component: str,
) -> None:

    domain, source, transfer, policy = rational_trim_metric_source
    with (
        sphere_native_scope(policy.limits) as native_scope,
        native_coordinate_resources(native_scope, policy.limits) as ledger,
    ):
        witness = prepare_surface_curve_witness(transfer.support, source, budget=ledger)
        witness.validate_restored()
        row = int(np.flatnonzero(np.asarray(witness.source_dimensions) == 1)[0])
        if component == "curve-index":
            indices = np.asarray(witness.source_indices).copy()
            indices[row] = (int(indices[row]) + 1) % len(domain.curves)
            changed = eqx.tree_at(
                lambda value: value.source_indices, witness, jnp.asarray(indices)
            )
        elif component == "curve-parameter":
            parameters = np.asarray(witness.source_parameters).copy()
            parameters[row, 0] = np.nextafter(parameters[row, 0], np.inf)
            changed = eqx.tree_at(
                lambda value: value.source_parameters, witness, jnp.asarray(parameters)
            )
        elif component == "entity":
            entities = list(witness.source_entity_ids)
            entities[row] = domain.entity_id(
                1, (int(witness.source_indices[row]) + 1) % len(domain.curves)
            )
            changed = replace(witness, source_entity_ids=tuple(entities))
        else:
            occurrences = list(witness.source_occurrence_paths)
            occurrences[row] = (*occurrences[row], "foreign-source-occurrence")
            changed = replace(witness, source_occurrence_paths=tuple(occurrences))
        with pytest.raises(ValueError, match="SurfaceCurveWitness"):
            changed.validate_restored()
        witness.validate_restored()


def test_public_rational_curve_split_keeps_independent_exact_material_anchor(
    rational_metric_scenario: RationalMetricScenario,
) -> None:
    scenario = rational_metric_scenario
    domain, source, _, _ = scenario.source_fixture
    target, chart = scenario.target, scenario.chart
    source_ids = set(map(int, np.asarray(source.mesh.vertex_global_ids)))
    association = next(
        value
        for value in target.associations
        if value.target_entity_set_id == target.mesh.entity_set(0).entity_set_id
    )
    association_rows = association.target_rows(np.asarray(target.mesh.vertex_global_ids))
    new_features = [
        row
        for row, identifier in enumerate(np.asarray(target.mesh.vertex_global_ids))
        if int(identifier) not in source_ids
        and int(association.source_dimensions[association_rows[row]]) == 1
    ]
    assert new_features
    target_cells = np.concatenate(
        [np.asarray(block.vertices) for block in target.mesh.blocks]
    )
    material = np.asarray(chart.material_charts(target=True), dtype=object)
    physical_charts = np.asarray(chart.target_witness.charts)
    nonlinear_anchors = 0
    source_cells = np.concatenate(
        [np.asarray(block.vertices) for block in source.mesh.blocks]
    )
    source_material = chart.material_charts(target=False)
    source_vertices = np.asarray(source.mesh.vertex_global_ids)
    midpoints = {
        tuple((source_material[cell, left] + source_material[cell, right]) / 2): {
            int(source_vertices[corners[left]]),
            int(source_vertices[corners[right]]),
        }
        for cell, corners in enumerate(source_cells)
        for left, right in ((0, 1), (1, 2), (2, 0))
    }
    stencil = scenario.adaptation.stencil
    assert stencil is not None
    exact_midpoints = 0
    for row in new_features:
        token = association_rows[row]
        curve = next(
            index
            for index in range(len(domain.curves))
            if domain.entity_id(1, index) == association.source_entities.entity_id(token)
        )
        expected = rational_trim_point(
            domain, curve, float(association.parameters[token, 0])
        )
        np.testing.assert_allclose(
            target.mesh.coordinates[row],
            expected,
            rtol=0.0,
            atol=domain.tolerance * domain.scale,
        )
        incident = np.flatnonzero(np.any(target_cells == row, axis=1))
        anchors = [
            tuple(material[cell, np.flatnonzero(target_cells[cell] == row)[0]])
            for cell in incident
        ]
        assert all(anchor == anchors[0] for anchor in anchors)
        assert all(isinstance(value, Fraction) for value in anchors[0])
        if anchors[0] in midpoints:
            exact_midpoints += 1
            stencil_row = int(
                np.flatnonzero(
                    np.asarray(stencil.target_global_ids)
                    == np.asarray(target.mesh.vertex_global_ids)[row]
                )[0]
            )
            positive = np.asarray(stencil.valid)[stencil_row] & (
                np.asarray(stencil.weights)[stencil_row] > 0
            )
            assert (
                set(
                    map(int, np.asarray(stencil.source_global_ids)[stencil_row, positive])
                )
                == midpoints[anchors[0]]
            )
            np.testing.assert_array_equal(
                np.asarray(stencil.weights)[stencil_row, positive], (0.5, 0.5)
            )
        nonlinear_anchors += any(
            np.linalg.norm(
                np.asarray(anchors[0], dtype=np.float64)
                - physical_charts[cell, np.flatnonzero(target_cells[cell] == row)[0]]
            )
            > domain.tolerance
            for cell in incident
        )
    assert nonlinear_anchors > 0
    assert exact_midpoints > 0
    assert scenario.adaptation.lineage is not None
    np.testing.assert_array_equal(
        np.asarray(source.geometry.coordinates).view(np.uint64),
        scenario.before.view(np.uint64),
    )


def test_rational_surface_reconstruction_preserves_original_complete_map(
    rational_trim_metric_source: UVMetricSource,
) -> None:
    with rational_surface_reconstruction_reference(
        rational_trim_metric_source
    ) as reference_state:
        source = reference_state.source
        atlas = reference_state.atlas
        original_controls = reference_state.original_controls
        source_coordinates = reference_state.source_coordinates
        original = reference_state.original
        reconstruction = reference_state.reconstruction
        mesh = reference_state.mesh
        ledger = reference_state.ledger
        before_restore = ledger.work_units
        restored = type(reconstruction).from_stored_fields(reconstruction, budget=ledger)
        assert (
            ledger.work_units >= before_restore + 8 * reconstruction.cell_global_ids.size
        )
        assert restored is reconstruction
        assert restored.reconstruction_id == reconstruction.reconstruction_id
        assert restored.source_geometry_id == reconstruction.source_geometry_id
        stale = replace(reconstruction, target_topology_id="stale")
        with pytest.raises(ValueError, match=r"surface_reconstruction\.binding"):
            type(reconstruction).from_stored_fields(stale, budget=ledger)
        invalid = replace(
            reconstruction,
            fidelity_bounds=reconstruction.fidelity_bounds.at[0].set(-1.0),
        )
        with pytest.raises(ValueError, match=r"surface_reconstruction\.bounds"):
            type(reconstruction).from_stored_fields(invalid, budget=ledger)
        changed = replace(
            reconstruction,
            vertex_coordinates=reconstruction.vertex_coordinates.at[0, 0].add(0.01),
        )
        with pytest.raises(ValueError, match="changed restored representatives"):
            type(reconstruction).from_stored_fields(changed, budget=ledger)
        assert isinstance(reference_state.atlas, SurfaceSourceRootAtlas)
        assert isinstance(reference_state.original, SplineCellGeometryElement)
        np.testing.assert_array_equal(
            np.sort(
                np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks])
            ),
            np.sort(
                np.concatenate(
                    [np.asarray(block.global_ids) for block in source.mesh.blocks]
                )
            ),
        )
        np.testing.assert_array_equal(
            np.asarray(mesh.vertex_global_ids), np.asarray(source.mesh.vertex_global_ids)
        )
        np.testing.assert_array_equal(
            np.asarray(reconstruction.geometry.coordinates).view(np.uint64),
            original_controls.view(np.uint64),
        )
        elements, routes, _ = reconstruction.geometry.resolve(mesh, budget=ledger)
        chart_rows = {
            int(identifier): row
            for row, identifier in enumerate(np.asarray(reconstruction.cell_global_ids))
        }
        reference = jnp.asarray(((1 / 7, 2 / 9),), dtype=jnp.float64)
        barycentric = np.asarray((1 - 1 / 7 - 2 / 9, 1 / 7, 2 / 9))
        for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
            assert isinstance(element, PolynomialComposedCellGeometryElement)
            retained = element.source_element
            assert isinstance(retained, SplineCellGeometryElement)
            for actual, authored in (
                (retained.u_knots, original.u_knots),
                (retained.v_knots, original.v_knots),
                (retained.weights, original.weights),
            ):
                np.testing.assert_array_equal(
                    np.asarray(actual).view(np.uint64),
                    np.asarray(authored).view(np.uint64),
                )
            for identifier, dofs in zip(
                np.asarray(block.global_ids), np.asarray(route), strict=True
            ):
                charts = np.asarray(reconstruction.cell_charts)[
                    chart_rows[int(identifier)]
                ]
                actual_maps = coordinate_expressions(
                    element, original_controls[dofs], budget=ledger
                )
                assert actual_maps is not None
                require_exact_rational_patch_map(
                    actual_maps,
                    charts,
                    original_controls.reshape((2, 2, 3)),
                    np.asarray(original.weights),
                )
                # The degree-sufficient rational proof above covers the whole map;
                # this nonnodal probe also checks numerical element tabulation.
                u, v = barycentric @ charts
                basis = np.asarray(
                    (((1 - u) * (1 - v), (1 - u) * v), (u * (1 - v), u * v))
                )
                weighted = basis * np.asarray(original.weights)
                physical = np.sum(
                    weighted[..., None] * original_controls.reshape((2, 2, 3)),
                    axis=(0, 1),
                ) / np.sum(weighted)
                values, _ = element.tabulate(reference)
                represented = np.asarray(values)[0] @ original_controls[dofs]
                np.testing.assert_allclose(
                    represented,
                    physical,
                    rtol=COORDINATE_RECONSTRUCTION_RTOL,
                    atol=SOURCE_RESIDUAL_ATOL,
                )
    np.testing.assert_array_equal(
        np.asarray(source.geometry.coordinates).view(np.uint64),
        source_coordinates.view(np.uint64),
    )
    np.testing.assert_array_equal(
        np.asarray(atlas.geometry.coordinates).view(np.uint64),
        original_controls.view(np.uint64),
    )


@pytest.fixture(scope="module")
def rational_metric_scenario(
    rational_trim_metric_source: UVMetricSource,
) -> RationalMetricScenario:
    return prepare_rational_metric_scenario(rational_trim_metric_source)


@pytest.fixture(scope="module")
def rational_metric_fields(
    rational_metric_scenario: RationalMetricScenario,
) -> RationalMetricFieldState:
    return prepare_rational_metric_fields(rational_metric_scenario)


@pytest.fixture(scope="module")
def rational_metric_archive(
    rational_metric_fields: RationalMetricFieldState,
) -> RationalMetricArchive:
    return prepare_rational_metric_archive(rational_metric_fields)


def test_rational_metric_success_preserves_geometry_and_request_budget(
    rational_metric_scenario: RationalMetricScenario,
) -> None:
    adaptation = rational_metric_scenario.adaptation
    target = rational_metric_scenario.target
    chart = rational_metric_scenario.chart
    domain = rational_metric_scenario.source_fixture.domain
    source = rational_metric_scenario.source_fixture.source
    assert adaptation.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    assert target.mesh.topology_id != source.mesh.topology_id
    assert adaptation.geometry_transition is not None
    chart = adaptation.geometry_transition.chart_deformation
    if not isinstance(chart, PreparedSurfaceChartDeformation):
        raise TypeError(
            "surface_metric.test.result must retain its declared scientific contract; got the rational trim route requires its UV deformation, not a sphere map"
        )
    assert chart.source_domain_coverage == chart.target_domain_coverage == "certified"
    assert target.certification is not None and target.certification.passed
    if not isinstance(adaptation.evidence, MetricRemeshingEvidence):
        raise TypeError(
            "surface_metric.test.result must retain its declared scientific contract; got the surface route must expose its metric evidence"
        )
    assert adaptation.evidence.execution_evidence is not None
    assert int(adaptation.evidence.execution_evidence.total_work_units) <= 640_000
    assert all(
        value.source_id == domain.source_id
        and value.source_revision == domain.source_revision
        for value in target.associations
    )


@pytest.mark.parametrize("partial_index", [0, 1], ids=["occurrence", "piece"])
def test_rational_material_closure_rejects_partial_deformation(
    rational_metric_scenario: RationalMetricScenario,
    partial_index: int,
) -> None:
    chart = rational_metric_scenario.chart
    validate_meshing_source_closure(chart)
    partial = (chart.occurrences[0], chart.occurrences[0].pieces[0])[partial_index]
    with pytest.raises(ValueError, match="material deformation"):
        validate_meshing_source_closure(partial)


def test_rational_material_closure_rejects_foreign_root_witness(
    rational_metric_scenario: RationalMetricScenario,
) -> None:
    chart = rational_metric_scenario.chart
    root = chart.material_root_witness
    malformed_root = SurfaceChartWitness(
        geometry_id=root.geometry_id,
        topology_id=root.topology_id,
        domain_id="foreign-material-source-domain",
        cell_global_ids=root.cell_global_ids,
        patches=root.patches,
        charts=root.charts,
        geometry_entity_ids=root.geometry_entity_ids,
        occurrence_paths=root.occurrence_paths,
    )
    malformed_chart = replace(chart, material_root_witness=malformed_root)
    with pytest.raises(ValueError, match="surface_chart.require_bound.binding"):
        malformed_chart.require_bound(
            chart.source_mesh,
            chart.source_geometry,
            chart.target_mesh,
            chart.target_geometry,
        )


def test_rational_curve_associations_bound_independent_source_residuals(
    rational_metric_scenario: RationalMetricScenario,
) -> None:
    target = rational_metric_scenario.target
    domain = rational_metric_scenario.source_fixture.domain
    source = rational_metric_scenario.source_fixture.source
    vertex_association = next(
        value
        for value in target.associations
        if value.target_entity_set_id == target.mesh.entity_set(0).entity_set_id
    )
    feature_rows = np.flatnonzero(np.asarray(vertex_association.source_dimensions) == 1)
    assert feature_rows.size > 0 and not vertex_association.exact
    actual_residuals = []
    vertex_rows = vertex_association.target_rows(
        np.asarray(target.mesh.vertex_global_ids)
    )
    for row in feature_rows:
        curve = next(
            index
            for index in range(len(domain.curves))
            if domain.entity_id(1, index)
            == vertex_association.source_entities.entity_id(row)
        )
        parameter = float(vertex_association.parameters[row, 0])
        analytic = rational_trim_point(domain, curve, parameter)
        target_row = int(np.flatnonzero(vertex_rows == row)[0])
        residual = np.linalg.norm(
            np.asarray(analytic) - np.asarray(target.mesh.coordinates)[target_row]
        )
        np.testing.assert_allclose(
            np.asarray(vertex_association.residuals[row]),
            np.asarray(residual),
            rtol=SOURCE_RESIDUAL_RTOL,
            atol=SOURCE_RESIDUAL_ATOL,
        )
        actual_residuals.append(residual)
    assert max(actual_residuals) <= domain.tolerance * domain.scale
    assert (
        source.certification is not None
        and source.certification.request.fidelity_tolerance is not None
    )
    assert max(actual_residuals) <= source.certification.request.fidelity_tolerance


def test_rational_fv_material_history_preserves_conservation_and_source(
    rational_metric_fields: RationalMetricFieldState,
) -> None:
    before = rational_metric_fields.scenario.before
    source = rational_metric_fields.scenario.source_fixture.source
    old_fv = rational_metric_fields.old_fv
    remap = rational_metric_fields.remap
    history = rational_metric_fields.history
    source_material = rational_metric_fields.source_material
    carried_history = rational_metric_fields.carried_history
    bound = rational_metric_fields.bound
    if not isinstance(remap.evidence, MappedSurfaceChartRemapEvidence):
        raise TypeError(
            f"surface_metric.test.remap.evidence must describe material correspondence; got {remap.evidence!r}"
        )
    assert remap.succeeded and remap.plan is not None and remap.evidence.passed
    defect = np.max(
        np.abs(
            np.asarray(old_fv.cell_volumes) @ np.asarray(history)
            - np.asarray(rational_metric_fields.new_fv.cell_volumes)
            @ np.asarray(carried_history)
        )
    )
    bound = float(remap.plan.report.total_source_coverage_error_bound) * float(
        np.max(np.abs(history))
    )
    bound += (
        np.finfo(np.float64).eps
        * 32
        * float(np.sum(np.abs(np.asarray(old_fv.cell_volumes)[:, None] * history)))
    )
    assert defect <= bound
    np.testing.assert_array_equal(
        source_material.state("surface-temperature-history").committed, history
    )
    np.testing.assert_array_equal(source.geometry.coordinates, before)


def test_rational_fe_material_transfer_preserves_primal_jvp_and_transpose(
    rational_metric_fields: RationalMetricFieldState,
) -> None:
    source_temperature = rational_metric_fields.scenario.source_temperature
    transfer = rational_metric_fields.transfer
    carried_temperature = rational_metric_fields.carried_temperature
    assert transfer.evidence.passed
    reference = material_p1_reference_matrix(
        rational_metric_fields.scenario.source_space,
        rational_metric_fields.prepared,
        rational_metric_fields.scenario.chart,
    )
    np.testing.assert_allclose(
        carried_temperature,
        reference @ np.asarray(source_temperature),
        rtol=MATERIAL_FIELD_RTOL,
        atol=MATERIAL_FIELD_ATOL,
    )
    direction = jnp.cos(jnp.arange(source_temperature.size, dtype=jnp.float64)).reshape(
        source_temperature.shape
    )
    primal, tangent = jvp(transfer.transfer.apply, (source_temperature,), (direction,))
    np.testing.assert_allclose(
        primal, carried_temperature, rtol=MATERIAL_FIELD_RTOL, atol=MATERIAL_FIELD_ATOL
    )
    np.testing.assert_allclose(
        tangent,
        reference @ np.asarray(direction),
        rtol=MATERIAL_FIELD_RTOL,
        atol=MATERIAL_FIELD_ATOL,
    )
    cotangent = jnp.sin(jnp.arange(primal.size, dtype=jnp.float64) + 0.5).reshape(
        primal.shape
    )
    _, pullback = vjp(transfer.transfer.apply, source_temperature)
    (source_cotangent,) = pullback(cotangent)
    np.testing.assert_allclose(
        source_cotangent,
        reference.T @ np.asarray(cotangent),
        rtol=MATERIAL_FIELD_RTOL,
        atol=MATERIAL_FIELD_ATOL,
    )
    np.testing.assert_allclose(
        jnp.vdot(tangent, cotangent),
        jnp.vdot(direction, source_cotangent),
        rtol=MATERIAL_FIELD_RTOL,
        atol=MATERIAL_FIELD_ATOL,
    )


def test_rational_topology_transaction_publishes_fields_and_material_history(
    rational_metric_fields: RationalMetricFieldState,
) -> None:
    field = rational_metric_fields.scenario.field
    source_space = rational_metric_fields.scenario.source_space
    source_problem = rational_metric_fields.scenario.source_problem
    source_temperature = rational_metric_fields.scenario.source_temperature
    adaptation = rational_metric_fields.scenario.adaptation
    chart = rational_metric_fields.scenario.chart
    source = rational_metric_fields.scenario.source_fixture.source
    old_fv = rational_metric_fields.old_fv
    new_fv = rational_metric_fields.new_fv
    remap = rational_metric_fields.remap
    remap_plan = rational_metric_fields.remap_plan
    history = rational_metric_fields.history
    source_material = rational_metric_fields.source_material
    carried_history = rational_metric_fields.carried_history
    bound = rational_metric_fields.bound
    prepared = rational_metric_fields.prepared
    carried_temperature = rational_metric_fields.carried_temperature
    problem = rational_metric_fields.problem
    temperature = rational_metric_fields.temperature
    accepted = FiniteElementAcceptedState(
        (source_temperature,),
        jnp.asarray(0.0, dtype=jnp.float64),
        1,
        source.mesh.topology_id,
        source_space.prepared_id,
        source_problem.compilation_id,
        materials=source_material,
        schedule_cursor=1,
        state_version=1,
    )
    material_history = RationalMaterialHistory(
        adaptation,
        old_fv,
        new_fv,
        chart,
        remap_plan,
        history,
        carried_temperature,
        bound,
        rational_metric_fields.scenario.target,
    )
    executor = FiniteElementTopologyTransaction(
        material_history.certify_material,
        fields=(field,),
        material_transfer=material_history.move_material,
        surface_chart_semantics={"temperature": "intensive"},
        projection_policy=phx.geometry.CommonRefinementPolicy(
            overlap_simplices=True,
            maximum_exact_work=640_000,
            maximum_memory_bytes=81_920_000,
        ),
        callback_owners=(
            old_fv,
            new_fv,
            source_space,
            prepared,
            remap,
            history,
            carried_temperature,
        ),
    )
    published = executor.execute(
        accepted, source.mesh, adaptation, compiled_layout_id=problem.compilation_id
    )
    assert (
        bool(published.committed)
        and published.receipt is not None
        and published.receipt.published
    )
    assert published.execution_evidence is not None
    assert int(published.execution_evidence.total_work_units) <= 640_000
    assert int(published.execution_evidence.total_geometry_queries) <= 1_280_000
    assert (
        published.material_remaps
        and published.material_remaps[0].execution_evidence is not None
    )
    assert published.state.materials is not None
    np.testing.assert_allclose(
        published.state.fields[0],
        carried_temperature,
        rtol=MATERIAL_FIELD_RTOL,
        atol=MATERIAL_FIELD_ATOL,
    )
    np.testing.assert_allclose(
        published.state.materials.state("surface-temperature-history").committed,
        carried_history,
        rtol=MATERIAL_FIELD_RTOL,
        atol=MATERIAL_FIELD_ATOL,
    )
    _, continued_after_rebind = solve_surface_reaction(
        prepared, published.state.fields[0]
    )
    np.testing.assert_allclose(
        continued_after_rebind,
        temperature,
        rtol=MATERIAL_FIELD_RTOL,
        atol=MATERIAL_FIELD_ATOL,
    )
    assert accepted.state_version == 1 and published.state.state_version == 2
    assert published.state.schedule_cursor == accepted.schedule_cursor
    assert published.state.transition_id == published.receipt.receipt_id


def test_rational_material_acceptance_refuses_corrupted_history(
    rational_metric_fields: RationalMetricFieldState,
) -> None:
    adaptation = rational_metric_fields.scenario.adaptation
    target = rational_metric_fields.scenario.target
    chart = rational_metric_fields.scenario.chart
    old_fv = rational_metric_fields.old_fv
    new_fv = rational_metric_fields.new_fv
    remap_plan = rational_metric_fields.remap_plan
    history = rational_metric_fields.history
    carried_history = rational_metric_fields.carried_history
    bound = rational_metric_fields.bound
    carried_temperature = rational_metric_fields.carried_temperature
    material_history = RationalMaterialHistory(
        adaptation,
        old_fv,
        new_fv,
        chart,
        remap_plan,
        history,
        carried_temperature,
        bound,
        target,
    )
    corrupted = phx.equations.MaterialTransaction(
        (
            phx.equations.MaterialState(
                phx.equations.MaterialSiteId("surface-temperature-history"),
                "surface-temperature-history",
                carried_history + 1.0,
            ),
        )
    )
    assert not material_history.certify_material(
        target.mesh, (carried_temperature,), corrupted, None, None
    )


def test_rational_archive_trust_binds_native_part_identity(
    rational_metric_archive: RationalMetricArchive,
) -> None:
    part = rational_metric_archive.part
    carrier = rational_metric_archive.carrier
    fresh_part = MeshPart(
        part.name,
        carrier,
        coordinate_contract=part.coordinate_contract,
    )
    assert source_content_equal(
        part,
        fresh_part,
        limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
        freshly_rebuilt=True,
    )
    name_fault = MeshPart(
        f"{part.name}-authority-fault",
        carrier,
        coordinate_contract=part.coordinate_contract,
    )
    assert not source_content_equal(
        name_fault,
        fresh_part,
        limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
        freshly_rebuilt=True,
    )


def test_rational_archive_part_refuses_uncertified_carrier_kind(
    rational_metric_archive: RationalMetricArchive,
) -> None:
    part, carrier = rational_metric_archive.part, rational_metric_archive.carrier
    # A bare mesh is not a certified archive carrier.
    with pytest.raises(TypeError, match="MeshPart.carrier"):
        MeshPart(
            part.name,
            carrier.mesh,  # ty: ignore[invalid-argument-type] -- deliberate public wrong-carrier refusal
            coordinate_contract=part.coordinate_contract,
        )


def test_rational_archive_part_refuses_foreign_coordinate_frame(
    rational_metric_archive: RationalMetricArchive,
) -> None:
    part, carrier = rational_metric_archive.part, rational_metric_archive.carrier
    altered_contract = phx.SpatialCoordinateContract(
        part.coordinate_contract.length_unit, reference_frame="foreign-material-frame"
    )
    with pytest.raises(ValueError, match="MeshPart.coordinate_contract"):
        MeshPart(part.name, carrier, coordinate_contract=altered_contract)


def test_rational_archive_refuses_changed_native_provider(
    rational_metric_archive: RationalMetricArchive,
) -> None:
    carrier = rational_metric_archive.carrier
    archive_records = rational_metric_archive.archive_records
    provider = carrier.provider
    altered_provider = type(provider)(
        f"{provider.name}-authority-fault",
        provider.version,
        provider.license_spdx,
        operations=provider.operations,
        source_kinds=provider.source_kinds,
        capabilities=provider.capabilities,
        cell_kinds=provider.cell_kinds,
        dimensions=provider.dimensions,
        execution_modes=provider.execution_modes,
    )
    malformed_carrier = eqx.tree_at(
        lambda value: value.provider, carrier, altered_provider
    )
    runtime = carrier.runtime
    altered_runtime = type(runtime)(
        altered_provider.provider_id,
        runtime.actual_version,
        runtime.execution_mode,
        deterministic=runtime.deterministic,
        enforced_limits=runtime.enforced_limits,
        unenforced_limits=runtime.unenforced_limits,
    )
    malformed_carrier = eqx.tree_at(
        lambda value: value.runtime, malformed_carrier, altered_runtime
    )
    malformed_records = dict(archive_records)
    malformed_records["accepted_target"] = malformed_carrier
    with pytest.raises(
        ValueError,
        match="stored_fields",
    ):
        validate_meshing_source_closure(malformed_records)


def test_rational_archive_refuses_missing_certification(
    rational_metric_archive: RationalMetricArchive,
) -> None:
    carrier = rational_metric_archive.carrier
    archive_records = rational_metric_archive.archive_records
    missing_certification = eqx.tree_at(lambda value: value.certification, carrier, None)
    missing_certification_records = dict(archive_records)
    missing_certification_records["accepted_target"] = missing_certification
    with pytest.raises(ValueError, match="source_closure.accepted_target"):
        validate_meshing_source_closure(missing_certification_records)


def test_rational_archive_refuses_missing_lineage(
    rational_metric_archive: RationalMetricArchive,
) -> None:
    adaptation_event = rational_metric_archive.adaptation_event
    archive_records = rational_metric_archive.archive_records
    missing_lineage_records = dict(archive_records)
    missing_lineage_event = dict(adaptation_event)
    missing_lineage_event["lineage"] = None
    missing_lineage_records["accepted_data"] = dict(
        archive_records["accepted_data"],
        adaptation_event=missing_lineage_event,
    )
    with pytest.raises((TypeError, ValueError), match="source_closure.accepted_target"):
        validate_meshing_source_closure(missing_lineage_records)


def test_rational_archive_refuses_duplicate_field_owner(
    rational_metric_archive: RationalMetricArchive,
) -> None:
    part = rational_metric_archive.part
    declaration = rational_metric_archive.declaration
    archive_records = rational_metric_archive.archive_records
    duplicate_owner_records = dict(archive_records)
    duplicate_owner_records["field_declarations"] = {
        part.name: declaration,
        f"{part.name}-duplicate": declaration,
    }
    with pytest.raises(ValueError, match="field_declarations"):
        validate_meshing_source_closure(duplicate_owner_records)


def test_rational_archive_refuses_mislabeled_generation(
    rational_metric_archive: RationalMetricArchive,
) -> None:
    part = rational_metric_archive.part
    archive_records = rational_metric_archive.archive_records
    mislabeled_generation_records: dict[str, Any] = dict(archive_records)
    mislabeled_generation_records["generation_part"] = part
    with pytest.raises(ValueError, match="accepted_target|roles"):
        validate_meshing_source_closure(mislabeled_generation_records)


class ReopenedRationalMetricArchive(NamedTuple):
    records: Mapping[str, Any]
    carrier: CellMeshingResult
    chart: PreparedSurfaceChartDeformation
    space: FiniteElementDiscretization


@pytest.fixture(scope="module")
def reopened_rational_metric_archive(
    rational_metric_archive: RationalMetricArchive,
    tmp_path_factory: pytest.TempPathFactory,
) -> ReopenedRationalMetricArchive:
    receipt = write_meshing_source_closure(
        tmp_path_factory.mktemp("rational-metric") / "temperature.npz",
        rational_metric_archive.archive_records,
    )
    reopened = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    cold = reopened["accepted_target"]
    if not isinstance(cold, CellMeshingResult) or cold.certification is None:
        raise TypeError(
            "rational_archive.carrier must retain certified cell geometry; got an incompatible carrier"
        )
    boundary = cold.certification.request.source
    if not isinstance(boundary, SurfaceChartBoundarySource):
        raise TypeError(
            "rational_archive.boundary must retain surface charts; got an incompatible boundary owner"
        )
    chart = boundary.deformation
    if not isinstance(chart, PreparedSurfaceChartDeformation):
        raise TypeError(
            "rational_archive.deformation must retain material correspondence; got an incompatible chart"
        )
    space, _ = prepare_meshing_field_owner(
        reopened["field_declarations"][rational_metric_archive.part.name], cold
    )
    if not isinstance(space, FiniteElementDiscretization):
        raise TypeError(
            "rational_archive.space must retain the finite-element temperature owner; got an incompatible space"
        )
    return ReopenedRationalMetricArchive(reopened, cold, chart, space)


def test_rational_archive_reopens_bound_material_deformation(
    rational_metric_archive: RationalMetricArchive,
    reopened_rational_metric_archive: ReopenedRationalMetricArchive,
) -> None:
    cold = reopened_rational_metric_archive
    original = rational_metric_archive.fields.scenario.chart
    validate_meshing_source_closure(cold.chart)
    cold.chart.require_bound(
        cold.chart.source_mesh,
        cold.chart.source_geometry,
        cold.carrier.mesh,
        cold.carrier.geometry,
    )
    assert cold.chart.deformation_id == original.deformation_id
    assert (
        cold.chart.material_root_witness.witness_id
        == original.material_root_witness.witness_id
    )


@pytest.mark.parametrize("target", [False, True], ids=["source", "target"])
def test_rational_archive_reopens_exact_material_charts(
    rational_metric_archive: RationalMetricArchive,
    reopened_rational_metric_archive: ReopenedRationalMetricArchive,
    target: bool,
) -> None:
    np.testing.assert_array_equal(
        reopened_rational_metric_archive.chart.material_charts(target=target),
        rational_metric_archive.fields.scenario.chart.material_charts(target=target),
    )


def test_rational_archive_reopens_exact_material_partition(
    rational_metric_archive: RationalMetricArchive,
    reopened_rational_metric_archive: ReopenedRationalMetricArchive,
) -> None:
    original_chart = rational_metric_archive.fields.scenario.chart
    for restored, original in zip(
        reopened_rational_metric_archive.chart.occurrences,
        original_chart.occurrences,
        strict=True,
    ):
        assert restored.material_root_witness_id == original.material_root_witness_id
        for restored_piece, original_piece in zip(
            restored.pieces, original.pieces, strict=True
        ):
            assert restored_piece.piece_id == original_piece.piece_id
            assert (
                restored_piece.exact_source_reference_vertices
                == original_piece.exact_source_reference_vertices
            )
            assert (
                restored_piece.exact_target_reference_vertices
                == original_piece.exact_target_reference_vertices
            )


def test_rational_archive_reopens_field_owner_and_continues_reaction(
    rational_metric_archive: RationalMetricArchive,
    reopened_rational_metric_archive: ReopenedRationalMetricArchive,
) -> None:
    cold = reopened_rational_metric_archive
    fields = rational_metric_archive.fields
    assert cold.space.prepared_id == fields.prepared.prepared_id
    problem, continued = solve_surface_reaction(
        cold.space, jnp.asarray(cold.records["accepted_data"]["fields"]["temperature"])
    )
    assert problem.compilation_id == fields.problem.compilation_id
    np.testing.assert_allclose(
        continued, fields.temperature, rtol=MATERIAL_FIELD_RTOL, atol=MATERIAL_FIELD_ATOL
    )


def test_rational_archive_reopens_material_history_without_source_mutation(
    rational_metric_archive: RationalMetricArchive,
    reopened_rational_metric_archive: ReopenedRationalMetricArchive,
) -> None:
    fields = rational_metric_archive.fields
    restored_material = phx.equations.MaterialTransaction(
        (
            phx.equations.MaterialState(
                phx.equations.MaterialSiteId("surface-temperature-history"),
                "surface-temperature-history",
                reopened_rational_metric_archive.records["accepted_data"]["fields"][
                    "material-temperature"
                ],
            ),
        )
    )
    assert restored_material.transaction_id == fields.material.transaction_id
    np.testing.assert_array_equal(
        fields.source_material.state("surface-temperature-history").committed,
        fields.history,
    )
    np.testing.assert_array_equal(
        fields.scenario.source_fixture.source.geometry.coordinates,
        fields.scenario.before,
    )


@pytest.mark.parametrize("fault", ["numerator", "chart", "control", "weight"])
def test_exact_rational_map_reference_detects_authored_fault(fault: str) -> None:
    controls = np.asarray(
        (((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0))),
        dtype=np.float64,
    )
    weights = np.ones((2, 2), dtype=np.float64)
    charts = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
    expressions: tuple[Polynomial, ...] = (
        {(1, 0): Fraction(1)},
        {(0, 1): Fraction(1)},
        {},
    )
    require_exact_rational_patch_map(expressions, charts, controls, weights)
    if fault == "numerator":
        expressions = ({(1, 0): Fraction(2)}, expressions[1], expressions[2])
    elif fault == "chart":
        charts[0, 0] = 0.125
    elif fault == "control":
        controls[1, 1, 2] = 0.125
    else:
        weights[1, 1] = 2.0
    with pytest.raises(AssertionError):
        require_exact_rational_patch_map(expressions, charts, controls, weights)


def test_rational_patch_reference_keeps_projective_weights() -> None:
    controls = np.asarray(
        (((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0))),
        dtype=np.float64,
    )
    weights = np.asarray(((1.0, 1.0), (1.0, 3.0)), dtype=np.float64)
    expected = (Fraction(2, 3), Fraction(2, 3), Fraction(0))
    assert (
        rational_patch_point(controls, weights, Fraction(1, 2), Fraction(1, 2))
        == expected
    )
    # Deliberately omit homogeneous weights, then exercise the same rational
    # point invariant against the corrupted reference output.
    faulty = rational_patch_point(
        controls, np.ones_like(weights), Fraction(1, 2), Fraction(1, 2)
    )
    with pytest.raises(AssertionError):
        assert faulty == expected


def test_material_p1_reference_refuses_uncovered_authored_target(
    rational_metric_fields: RationalMetricFieldState,
) -> None:
    fields = rational_metric_fields
    chart = fields.scenario.chart
    altered_occurrences = tuple(
        replace(
            occurrence,
            exact_target_material_charts=tuple(
                tuple((u + 10, v + 10) for u, v in triangle)
                for triangle in occurrence.exact_target_material_charts
            ),
        )
        for occurrence in chart.occurrences
    )
    altered = replace(chart, occurrences=altered_occurrences)
    with pytest.raises(ValueError, match=r"material_reference\.coverage"):
        material_p1_reference_matrix(
            fields.scenario.source_space, fields.prepared, altered
        )


def test_material_p1_reference_detects_corrupted_operator_output(
    rational_metric_fields: RationalMetricFieldState,
) -> None:
    fields = rational_metric_fields
    matrix = material_p1_reference_matrix(
        fields.scenario.source_space, fields.prepared, fields.scenario.chart
    )
    direction = np.ones(matrix.shape[1], dtype=np.float64)
    expected = np.asarray(fields.transfer.transfer.apply(jnp.asarray(direction)))
    np.testing.assert_allclose(
        matrix @ direction,
        expected,
        rtol=MATERIAL_FIELD_RTOL,
        atol=MATERIAL_FIELD_ATOL,
    )
    faulty = matrix.copy()
    faulty[0, 0] += 1.0
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            faulty @ direction,
            expected,
            rtol=MATERIAL_FIELD_RTOL,
            atol=MATERIAL_FIELD_ATOL,
        )


def test_rational_trim_reference_detects_changed_authored_curve(
    rational_trim_metric_source: UVMetricSource,
) -> None:
    domain = rational_trim_metric_source.domain
    # The authored first Bezier has weights (1, 3/4, 1). At t=1/2 its
    # UV point is exactly (19/28, 19/28), unlike its affine endpoint midpoint.
    surface = domain.patches[0].surface
    assert isinstance(surface, BSplineSurfacePatch)
    expected = np.asarray(
        rational_patch_point(
            np.asarray(surface.control_points),
            np.asarray(surface.weights),
            Fraction(19, 28),
            Fraction(19, 28),
        ),
        dtype=np.float64,
    )
    np.testing.assert_array_equal(rational_trim_point(domain, 0, 0.5), expected)
    faulty = rational_trim_point(domain, 0, 0.25)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(faulty, expected)


def test_public_rational_uv_resource_refusal_retains_source(
    rational_trim_metric_source: UVMetricSource,
) -> None:

    _, source, transfer, policy = rational_trim_metric_source
    before = np.asarray(source.geometry.coordinates).copy()
    limits = MeshingLimits(
        maximum_vertices=20_000,
        maximum_edges=160_000,
        maximum_faces=160_000,
        maximum_cells=160_000,
        maximum_work_units=1_000,
        maximum_geometry_queries=1_280_000,
        maximum_scratch_bytes=81_920_000,
    )
    starved = MeshAdaptationPolicy(
        MeshAdaptationRoute.NATIVE_SURFACE_METRIC,
        association_transfer=transfer,
        maximum_passes=policy.maximum_passes,
        relocation=False,
        limits=limits,
        geometry_transition=policy.geometry_transition,
    )
    vertices = source.mesh.entity_set(0)
    scope = phx.meshing.MeshingScope(
        source.mesh.mesh_id,
        source.mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        0,
        vertices.entity_set_id,
        vertices.entity_ids,
    )
    metric = phx.meshing.MeshMetricField(
        scope,
        np.broadcast_to(
            np.eye(3, dtype=np.float64) / 0.14**2,
            (source.mesh.coordinates.shape[0], 3, 3),
        ),
        minimum_size=0.14,
        maximum_size=0.14,
    )
    result = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source,
            phx.meshing.MetricMeshAdaptation(
                metric, coordinate_contract=source.coordinate_contract
            ),
            policy=starved,
        )
    )
    assert result.status is phx.meshing.MeshAdaptationStatus.RESOURCE_LIMIT
    assert result.target.result_id == source.result_id
    assert isinstance(result.evidence, MetricRemeshingEvidence)
    assert not result.evidence.converged and result.evidence.resource_message
    assert result.evidence.execution_evidence is not None
    assert dict(result.evidence.resource_requested)["maximum_work_units"] == 1_000
    np.testing.assert_array_equal(source.geometry.coordinates, before)


def test_rational_uv_capacity_refusal_rolls_back_partial_edit(
    rational_trim_metric_source: UVMetricSource,
) -> None:

    _, source, transfer, policy = rational_trim_metric_source
    before = np.asarray(source.geometry.coordinates).copy()
    maximum_vertices = source.mesh.coordinates.shape[0] + 1
    limits = MeshingLimits(
        maximum_vertices=maximum_vertices,
        maximum_edges=160_000,
        maximum_faces=160_000,
        maximum_cells=160_000,
        maximum_work_units=640_000,
        maximum_geometry_queries=1_280_000,
        maximum_scratch_bytes=81_920_000,
    )
    bounded = MeshAdaptationPolicy(
        MeshAdaptationRoute.NATIVE_SURFACE_METRIC,
        association_transfer=transfer,
        maximum_passes=policy.maximum_passes,
        relocation=False,
        limits=limits,
        geometry_transition=policy.geometry_transition,
    )
    vertices = source.mesh.entity_set(0)
    scope = phx.meshing.MeshingScope(
        source.mesh.mesh_id,
        source.mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        0,
        vertices.entity_set_id,
        vertices.entity_ids,
    )
    metric = phx.meshing.MeshMetricField(
        scope,
        np.broadcast_to(
            np.eye(3, dtype=np.float64) / 0.07**2,
            (source.mesh.coordinates.shape[0], 3, 3),
        ),
        minimum_size=0.07,
        maximum_size=0.07,
    )
    result = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source,
            phx.meshing.MetricMeshAdaptation(
                metric, coordinate_contract=source.coordinate_contract
            ),
            policy=bounded,
        )
    )
    assert result.status is phx.meshing.MeshAdaptationStatus.RESOURCE_LIMIT
    assert result.target.result_id == source.result_id
    assert isinstance(result.evidence, MetricRemeshingEvidence)
    assert result.evidence.splits > 0
    assert result.evidence.resource_message and not result.evidence.converged
    assert (
        dict(result.evidence.resource_requested)["maximum_vertices"] == maximum_vertices
    )
    assert dict(result.evidence.resource_achieved)["vertices"] == maximum_vertices
    assert result.evidence.execution_evidence is not None
    np.testing.assert_array_equal(source.geometry.coordinates, before)


@pytest.mark.parametrize(
    "minimum_quality",
    [-0.01, 1.01, np.nan, np.inf],
    ids=["below-zero", "above-one", "nan", "infinite"],
)
def test_sphere_quality_floor_refuses_invalid_value_without_source_mutation(
    sphere_metric_source: SphereMetricSource,
    minimum_quality: float,
) -> None:
    domain, source, atlas, transfer, policy = sphere_metric_source
    before = np.asarray(source.geometry.coordinates).copy()
    metric = np.broadcast_to(
        np.eye(3, dtype=np.float64), (source.mesh.coordinates.shape[0], 3, 3)
    )
    with sphere_native_scope(policy.limits) as native_scope:
        with native_coordinate_resources(native_scope, policy.limits) as ledger:
            with pytest.raises(ValueError, match="sphere_metric.minimum_metric_quality"):
                execute_sphere_metric_adaptation(
                    source,
                    metric,
                    domain,
                    atlas,
                    transfer,
                    policy=policy,
                    coordinate_contract=source.coordinate_contract,
                    maximum_fidelity=0.16,
                    minimum_metric_quality=minimum_quality,
                    topology_operations=True,
                    relocation=False,
                    budget=ledger,
                    scope=native_scope,
                )
    np.testing.assert_array_equal(source.geometry.coordinates, before)


@pytest.mark.parametrize("orientation", (-1, 1))
def test_nominal_coedge_split_preserves_coincident_source_sides(orientation: int) -> None:
    cells = np.asarray(((0, 1, 2), (0, 1, 3)), dtype=np.int32)
    bank = np.full((2, 3, 3), -1, dtype=np.int32)
    bank[0, 0] = (0, 4, orientation)
    bank[1, 0] = (2, 9, -orientation)
    state = owner.SurfaceState(
        points=np.zeros((4, 3)),
        metric=np.tile(np.eye(3), (4, 1, 1)),
        cells=cells,
        patches=np.asarray((7, 7), dtype=np.int32),
        charts=np.zeros((2, 3, 2)),
        classes=np.zeros(2, dtype=np.int32),
        cell_ids=np.asarray((901, 17), dtype=np.int64),
        vertex_ids=np.asarray((100, 42, 905, 8), dtype=np.int64),
        parents=[{901}, {17}],
        vertex_sources=[{100}, {42}, {905}, {8}],
        next_cell_id=902,
        next_vertex_id=906,
        cell_coedges=bank,
    )
    result = owner._transport_cell_coedges(
        state,
        np.asarray((1, 0, 1, 0), dtype=np.int64),
        np.asarray(((4, 1, 3), (0, 4, 2), (0, 4, 3), (4, 1, 2))),
        split_edge=np.asarray((0, 1)),
        split_vertex=4,
    )
    assert result is not None
    np.testing.assert_array_equal(result[:, 0], bank[[1, 0, 1, 0], 0])
    np.testing.assert_array_equal(result[:, 1:], -np.ones((4, 2, 3), dtype=np.int32))
    moved, preserved = owner._cavity_cell_coedges(
        state, np.asarray((0,)), np.asarray(((0, 1, 3),))
    )
    assert preserved and moved is not None
    np.testing.assert_array_equal(moved[0, 0], bank[0, 0])
    merged, preserved = owner._cavity_cell_coedges(
        state, np.asarray((0, 1)), np.asarray(((0, 1, 3),))
    )
    assert not preserved and merged is None
