#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from examples._native_surface_sources import AnalyticSurface, sphere, torus
from phydrax._meshcore import NativeExecutionBudget
from phydrax.discretization import CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry_transfer import CellGeometryTransitionPolicy
from phydrax.discretization._coordinate_enclosure import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.discretization._surface_chart_deformation import (
    PreparedSurfaceChartDeformation,
)
from phydrax.discretization.fem import FiniteElementDiscretization
from phydrax.discretization.finite_volume._remap_evidence import (
    MappedSurfaceChartRemapEvidence,
)
from phydrax.equations import CompiledFiniteElementProblem
from phydrax.geometry import MeshingDomain
from phydrax.geometry._sphere_material_atlas import SphereMaterialCellAtlas
from phydrax.geometry._surface_source_support import (
    prepare_surface_source_root_atlas,
    PreparedSurfaceSourceSupport,
    SurfaceSourceCharts,
)
from phydrax.linalg._spaces import ArraySpace
from phydrax.meshing._adaptation import MeshAdaptationPolicy, MeshAdaptationRoute
from phydrax.meshing._association import SurfaceAssociationTransfer
from phydrax.meshing._contracts import MeshingLimits
from phydrax.meshing._lineage import inherit_mesh_organization
from phydrax.meshing._result import CellMeshingResult
from phydrax.meshing._surface_metric import (
    execute_sphere_metric_adaptation,
    execute_surface_metric_adaptation,
)
from phydrax.meshing._tetra_metric import MetricRemeshingEvidence, MetricRemeshingStatus
from phydrax.meshing._topology_edit import assemble_topology_edit, TopologyEditBlock
from tests.unit.geometry.test_sphere_material_atlas import _accepted


def _patch(surface: AnalyticSurface) -> tuple[CellMesh, np.ndarray, np.ndarray]:
    chart_points = np.asarray(
        ((0.0, 0.0), (0.2, 0.0), (0.2, 0.2), (0.0, 0.2)), dtype=np.float64
    )
    cells = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
    points = surface.domain.evaluate(np.zeros((4,), dtype=np.int32), chart_points)
    return (
        CellMesh.from_triangles(points, cells),
        np.zeros((2,), dtype=np.int32),
        chart_points[cells],
    )


@pytest.mark.parametrize("builder", [sphere, torus], ids=["sphere", "torus"])
def test_curved_patch_metric_keeps_authoritative_chart_geometry(
    builder: Callable[[], AnalyticSurface],
) -> None:
    surface = builder()
    source, patches, charts = _patch(surface)
    # Independent analytic tangent scales at (u,v)=(0,0): sphere (1,1), torus (2.6,.6).
    if surface.domain.source_id == "sphere":
        tensor = np.eye(3, dtype=np.float64) * (0.95 / 0.2) ** 2
    else:
        tensor = np.diag((1.0, (0.95 / 0.52) ** 2, (0.95 / 0.12) ** 2))
    metric = np.broadcast_to(tensor, (4, 3, 3))
    before = np.asarray(source.coordinates).copy()
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
    )
    assert outcome.evidence.status is MetricRemeshingStatus.COMPLETE
    assert outcome.source_id == surface.domain.source_id
    assert outcome.source_revision == surface.domain.source_revision
    cell_blocks = []
    for block in outcome.edit.blocks:
        assert isinstance(block, TopologyEditBlock)
        cell_blocks.append(block.cells)
    target_cells = np.concatenate(cell_blocks)
    represented = surface.domain.evaluate(
        np.repeat(outcome.cell_patches, 3), outcome.cell_charts.reshape((-1, 2))
    ).reshape((-1, 3, 3))
    np.testing.assert_allclose(
        represented, outcome.edit.coordinates[target_cells], atol=1.0e-12
    )
    assert np.max(surface.distance(outcome.edit.coordinates)) < 1.0e-12
    assert outcome.evidence.maximum_fidelity_bound <= 0.1
    np.testing.assert_array_equal(source.coordinates, before)


def test_metric_target_lineage_binds_regrouped_scientific_cells() -> None:
    from phydrax.discretization import CellBlock
    from phydrax.meshing._periodic import metric_target_lineage
    from phydrax.meshing._topology_edit import entity_keys, key_rows

    surface = torus()
    source, patches, charts = _patch(surface)
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
    )
    staged, _, _ = assemble_topology_edit(source, outcome.edit, numeric_version="regroup")
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
    lineage = metric_target_lineage(source, outcome.edit, target)
    assert lineage.target_topology_id == target.topology_id
    assert lineage.source_topology_id == source.topology_id
    mismatched = CellMesh(
        staged.coordinates,
        blocks,
        vertex_global_ids=staged.vertex_global_ids,
        entity_global_ids={1: staged.entity_set(1).entity_ids},
        numeric_version=staged.numeric_version,
    )
    with pytest.raises(ValueError, match="scientific carrier incidence"):
        metric_target_lineage(source, outcome.edit, mismatched)


def test_metric_target_lineage_refuses_replaced_scientific_cell() -> None:
    from phydrax.discretization import CellBlock
    from phydrax.meshing._periodic import metric_target_lineage

    surface = torus()
    source, patches, charts = _patch(surface)
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
    )
    staged, _, _ = assemble_topology_edit(
        source, outcome.edit, numeric_version="replacement"
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
    with pytest.raises(ValueError, match="scientific carrier entity identities"):
        metric_target_lineage(source, outcome.edit, target)


def test_complete_sphere_quality_bounds_the_independent_curved_differential() -> None:
    surface = sphere()
    mesh, patches, charts = _patch(surface)
    metric = np.broadcast_to(np.eye(3) / 0.2**2, (4, 3, 3))
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
    surface = sphere()
    source, patches, charts = _patch(surface)
    metric = np.broadcast_to(np.eye(3, dtype=np.float64) / 0.2**2, (4, 3, 3))
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
    )
    # Latitude v=.2 is shorter than a meridian by cos(.2); Gauss integrates the
    # exact analytic arc, whereas chord length is 2 sin(.1) cos(.2)/.2.
    assert outcome.evidence.minimum_metric_length == pytest.approx(
        np.cos(0.2), rel=1.0e-10
    )
    chord = 2.0 * np.sin(0.1) * np.cos(0.2) / 0.2
    assert outcome.evidence.minimum_metric_length > chord + 1.0e-3


def test_stale_surface_revision_is_refused_without_mutation() -> None:
    surface = sphere()
    source, patches, charts = _patch(surface)
    before = np.asarray(source.coordinates).copy()
    with pytest.raises(ValueError, match="revision"):
        execute_surface_metric_adaptation(
            source,
            np.broadcast_to(np.eye(3), (4, 3, 3)),
            surface.domain,
            source_id=surface.domain.source_id,
            source_revision="stale",
            cell_patches=patches,
            cell_charts=charts,
            maximum_fidelity=0.1,
        )
    np.testing.assert_array_equal(source.coordinates, before)


def test_surface_budget_exhaustion_keeps_source_and_unmet_metric_evidence() -> None:
    surface = sphere()
    source, patches, charts = _patch(surface)
    before = np.asarray(source.coordinates).copy()
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
    )
    assert outcome.evidence.status is MetricRemeshingStatus.RESOURCE_LIMIT
    assert outcome.evidence.out_of_range_edges == 5
    assert outcome.evidence.resource_message
    assert dict(outcome.evidence.resource_requested)["maximum_operations"] == 0
    assert dict(outcome.evidence.resource_achieved)["operation_attempts"] == 0
    np.testing.assert_array_equal(outcome.edit.coordinates, before)
    np.testing.assert_array_equal(source.coordinates, before)


def test_spherical_refinement_improves_continuous_fidelity_without_flattening() -> None:
    surface = sphere()
    source, patches, charts = _patch(surface)
    before = np.asarray(source.coordinates).copy()
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
    )
    assert outcome.evidence.status is MetricRemeshingStatus.COMPLETE
    assert outcome.evidence.maximum_fidelity_bound <= 0.01
    assert np.max(surface.distance(outcome.edit.coordinates)) < 1.0e-12
    cell_blocks = []
    for block in outcome.edit.blocks:
        assert isinstance(block, TopologyEditBlock)
        cell_blocks.append(block.cells)
    target_cells = np.concatenate(cell_blocks)
    corner = outcome.edit.coordinates[target_cells]
    normal = np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0])
    assert np.all(np.sum(normal * np.mean(corner, axis=1), axis=1) > 0.0)
    represented = surface.domain.evaluate(
        np.repeat(outcome.cell_patches, 3), outcome.cell_charts.reshape((-1, 2))
    ).reshape((-1, 3, 3))
    np.testing.assert_allclose(represented, corner, atol=1.0e-12)
    np.testing.assert_array_equal(source.coordinates, before)


class _SphereProducerFixture(NamedTuple):
    domain: MeshingDomain
    source: CellMeshingResult
    atlas: SphereMaterialCellAtlas
    transfer: SurfaceAssociationTransfer
    policy: MeshAdaptationPolicy


def _prepare_sphere_metric_source(size: float) -> _SphereProducerFixture:
    """Retain actual native corner charts, not a manufactured latitude atlas."""
    domain, source, atlas = _accepted.__wrapped__(size)
    retained = source.surface_source
    if not isinstance(retained, SurfaceSourceCharts):
        raise TypeError(
            "The native sphere must retain its actual original source charts."
        )
    retained.require_root(source.mesh, source.geometry)
    charts, boundary = retained.root_parameters, retained.boundary_source
    transfer = SurfaceAssociationTransfer(
        PreparedSurfaceSourceSupport(
            domain,
            source,
            jnp.asarray(charts),
            boundary,
            maximum_support_queries=1_280_000,
        )
    )
    limits = MeshingLimits(
        maximum_vertices=20_000,
        maximum_edges=160_000,
        maximum_faces=160_000,
        maximum_cells=160_000,
        maximum_work_units=640_000,
        maximum_geometry_queries=1_280_000,
        maximum_scratch_bytes=81_920_000,
    )
    policy = MeshAdaptationPolicy(
        MeshAdaptationRoute.NATIVE_SURFACE_METRIC,
        association_transfer=transfer,
        maximum_passes=4,
        relocation=False,
        limits=limits,
        geometry_transition=CellGeometryTransitionPolicy(
            reconstruction="bounded_chart_deformation",
            reconstruction_tolerance=0.32,
            maximum_evaluations=640_000,
        ),
    )
    return _SphereProducerFixture(domain, source, atlas, transfer, policy)


@pytest.fixture(scope="module")
def sphere_metric_source() -> _SphereProducerFixture:
    return _prepare_sphere_metric_source(0.8)


@pytest.fixture(scope="module")
def fine_sphere_metric_source() -> _SphereProducerFixture:
    return _prepare_sphere_metric_source(0.4)


def _sphere_native_budget(limits: MeshingLimits) -> NativeExecutionBudget:
    return NativeExecutionBudget(
        max_work=limits.maximum_work_units,
        max_geometry_queries=limits.maximum_geometry_queries,
        max_cavity_cells=limits.maximum_cavity_cells,
        max_scratch_bytes=limits.maximum_scratch_bytes,
        max_wall_seconds=limits.maximum_wall_seconds,
    )


def test_sphere_cone_filter_resolves_zero_area_contact_without_losing_actual_cells(
    sphere_metric_source: _SphereProducerFixture,
) -> None:
    from phydrax.discretization._sphere_chart_deformation import _cone_candidates

    _, source, atlas, _, policy = sphere_metric_source
    cells = np.concatenate([np.asarray(block.vertices) for block in source.mesh.blocks])
    physical = np.asarray(atlas.physical_rows)
    neighbors = np.flatnonzero(
        np.sum(np.isin(cells[physical], cells[physical[0]]), axis=1) == 2
    )
    assert neighbors.size
    ledger = CoordinateEnclosureBudget(
        policy.limits.maximum_work_units, policy.limits.maximum_scratch_bytes
    )
    with _sphere_native_budget(policy.limits), ledger.activate():
        kept_source, kept_target = _cone_candidates(
            atlas,
            atlas,
            np.asarray((0, 0), dtype=np.int64),
            np.asarray((0, neighbors[0]), dtype=np.int64),
            policy.limits.maximum_scratch_bytes,
        )
    np.testing.assert_array_equal(kept_source, (0,))
    np.testing.assert_array_equal(kept_target, (0,))


def test_sphere_displacement_zero_requires_exact_coefficient_cancellation() -> None:
    from phydrax.discretization._sphere_chart_deformation import (
        _sphere_full_map_displacement,
    )

    vertices = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    zero = CoordinateEnclosureBudget(3, 0)
    assert _sphere_full_map_displacement(({}, {}, {}), vertices, zero) == 0.0
    nonzero = CoordinateEnclosureBudget(3, 0)
    with pytest.raises(CoordinateEnclosureResourceError):
        _sphere_full_map_displacement(
            ({(0, 0): Fraction(1, 2**100)}, {}, {}), vertices, nonzero
        )


def test_native_sphere_metric_changes_actual_cells_with_complete_radial_correspondence(
    sphere_metric_source: _SphereProducerFixture,
) -> None:
    domain, source, atlas, transfer, policy = sphere_metric_source
    before = np.asarray(source.geometry.coordinates).copy()
    ledger = CoordinateEnclosureBudget(
        policy.limits.maximum_work_units, policy.limits.maximum_scratch_bytes
    )
    metric = np.broadcast_to(
        np.eye(3, dtype=np.float64) / 0.8**2, (source.mesh.coordinates.shape[0], 3, 3)
    )
    native = _sphere_native_budget(policy.limits)
    with native:
        outcome = execute_sphere_metric_adaptation(
            source,
            metric,
            domain,
            atlas,
            transfer,
            policy=policy,
            coordinate_contract=source.coordinate_contract,
            maximum_fidelity=0.16,
            topology_operations=True,
            relocation=False,
            coordinate_budget=ledger,
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
    target, lineage, _ = assemble_topology_edit(
        source.mesh, outcome.edit, numeric_version=outcome.target_mesh.numeric_version
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
    # Native clipping and other owning FFI stages have separate work; only the
    # coordinate attribution counter must equal this original host ledger.
    assert (
        ledger.work_units
        <= native.evidence.externally_charged_work
        <= policy.limits.maximum_work_units
    )
    np.testing.assert_array_equal(np.asarray(source.geometry.coordinates), before)


@pytest.mark.parametrize(
    "source_dimension", [1, 2], ids=["source-curve", "source-surface"]
)
def test_sphere_relocation_certification_refusal_restores_trial_carrier(
    sphere_metric_source: _SphereProducerFixture,
    monkeypatch: pytest.MonkeyPatch,
    source_dimension: int,
) -> None:
    import phydrax.meshing._surface_metric as owner

    domain, source, atlas, transfer, policy = sphere_metric_source
    ledger = CoordinateEnclosureBudget(
        policy.limits.maximum_work_units, policy.limits.maximum_scratch_bytes
    )
    metric = np.broadcast_to(
        np.eye(3, dtype=np.float64), (source.mesh.coordinates.shape[0], 3, 3)
    )
    with _sphere_native_budget(policy.limits), ledger.activate():
        state, background = owner._sphere_initial_state(
            source, metric, atlas, transfer, policy, 0.16
        )
        vertex = next(
            int(value)
            for value in np.unique(state.cells)
            if int(value) not in state.fixed
            and state.dimensions[int(value)] == source_dimension
            and (
                source_dimension == 1
                and len([edge for edge in state.features if int(value) in edge]) == 2
                and all(
                    edge not in state.blocked and curve == state.indices[int(value)]
                    for edge, curve in state.features.items()
                    if int(value) in edge
                )
                or source_dimension == 2
                and not any(int(value) in edge for edge in state.features)
            )
            and np.unique(state.patches[np.any(state.cells == value, axis=1)]).size == 1
        )
        points, directions, tensors = (
            state.points.copy(),
            state.directions.copy(),
            state.metric.copy(),
        )
        parameters = tuple(value.copy() for value in state.parameters)
        stencils = tuple(state.stencils)
        ancestry = tuple(frozenset(value) for value in state.vertex_sources)
        kinds = dict(state.kinds)
        work_before = ledger.work_units

        def refuse(*args: Any, **kwargs: Any) -> bool:
            # The real proposal has already installed its numeric trial. A
            # certification refusal must not leave that trial in the carrier.
            raise ValueError("owning certification refusal")

        monkeypatch.setattr(owner, "_sphere_legal", refuse)
        with pytest.raises(ValueError, match="owning certification refusal"):
            owner._sphere_relocate(state, vertex, domain, background, ledger, transfer)
        np.testing.assert_array_equal(state.points, points)
        np.testing.assert_array_equal(state.directions, directions)
        np.testing.assert_array_equal(state.metric, tensors)
        for actual, original in zip(state.parameters, parameters, strict=True):
            np.testing.assert_array_equal(actual, original)
        assert tuple(state.stencils) == stencils
        assert tuple(frozenset(value) for value in state.vertex_sources) == ancestry
        assert state.kinds == kinds
        assert vertex not in state.relocated
        assert ledger.work_units > work_before


def test_sphere_carrier_operation_admits_its_actual_cavity_before_mutation(
    sphere_metric_source: _SphereProducerFixture,
) -> None:
    import phydrax.meshing._surface_metric as owner
    from phydrax._meshcore import MeshcoreError

    domain, source, atlas, transfer, policy = sphere_metric_source
    ledger = CoordinateEnclosureBudget(
        policy.limits.maximum_work_units, policy.limits.maximum_scratch_bytes
    )
    metric = np.broadcast_to(
        np.eye(3, dtype=np.float64), (source.mesh.coordinates.shape[0], 3, 3)
    )
    with _sphere_native_budget(policy.limits), ledger.activate():
        state, background = owner._sphere_initial_state(
            source, metric, atlas, transfer, policy, 0.16
        )
    vertex = int(state.cells[0, 0])
    before = state.points.copy()
    cells = state.cells.copy()
    stencils = tuple(state.stencils)
    work_before = ledger.work_units
    native = NativeExecutionBudget(
        max_work=policy.limits.maximum_work_units,
        max_geometry_queries=policy.limits.maximum_geometry_queries,
        max_cavity_cells=0,
        max_scratch_bytes=policy.limits.maximum_scratch_bytes,
        max_wall_seconds=policy.limits.maximum_wall_seconds,
    )
    with pytest.raises(MeshcoreError), native, ledger.activate():
        owner._sphere_apply_operation(
            state, "relocate", (vertex,), domain, background, transfer, ledger
        )
    np.testing.assert_array_equal(state.points, before)
    np.testing.assert_array_equal(state.cells, cells)
    assert tuple(state.stencils) == stencils
    assert ledger.work_units == work_before


def test_native_sphere_metric_zero_ledger_refuses_without_source_mutation(
    sphere_metric_source: _SphereProducerFixture,
) -> None:
    domain, source, atlas, transfer, policy = sphere_metric_source
    before = np.asarray(source.geometry.coordinates).copy()
    ledger = CoordinateEnclosureBudget(0, 0)
    with (
        _sphere_native_budget(policy.limits),
        pytest.raises(CoordinateEnclosureResourceError),
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
            coordinate_budget=ledger,
        )
    assert ledger.work_units == ledger.native_charged_work_units == 0
    np.testing.assert_array_equal(np.asarray(source.geometry.coordinates), before)


def test_native_sphere_metric_stale_source_revision_refuses_without_mutation(
    sphere_metric_source: _SphereProducerFixture,
) -> None:
    _, source, atlas, transfer, policy = sphere_metric_source
    before = np.asarray(source.geometry.coordinates).copy()
    ledger = CoordinateEnclosureBudget(
        policy.limits.maximum_work_units, policy.limits.maximum_scratch_bytes
    )
    with pytest.raises(ValueError, match="stale"):
        execute_sphere_metric_adaptation(
            source,
            np.broadcast_to(
                np.eye(3, dtype=np.float64), (source.mesh.coordinates.shape[0], 3, 3)
            ),
            sphere(revision="changed").domain,
            atlas,
            transfer,
            policy=policy,
            coordinate_contract=source.coordinate_contract,
            maximum_fidelity=0.16,
            topology_operations=True,
            relocation=False,
            coordinate_budget=ledger,
        )
    assert ledger.work_units == 0
    np.testing.assert_array_equal(np.asarray(source.geometry.coordinates), before)


def test_public_sphere_metric_resource_refusal_reaches_the_consumer_as_evidence(
    sphere_metric_source: _SphereProducerFixture,
) -> None:
    from phydrax.meshing import (
        execute_mesh_adaptation,
        MeshAdaptationStatus,
        MeshingEntityKind,
        MeshingScope,
        MeshMetricField,
        MetricMeshAdaptation,
        prepare_mesh_adaptation,
    )

    _, source, _, transfer, policy = sphere_metric_source
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
    result = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            source,
            MetricMeshAdaptation(metric, coordinate_contract=source.coordinate_contract),
            policy=starved,
        )
    )

    assert result.status is MeshAdaptationStatus.RESOURCE_LIMIT
    assert result.target.result_id == source.result_id
    evidence = result.evidence
    assert isinstance(evidence, MetricRemeshingEvidence)
    assert evidence.status is MetricRemeshingStatus.RESOURCE_LIMIT
    assert evidence.resource_message
    assert dict(evidence.resource_requested)["maximum_work_units"] == 1_000
    assert dict(evidence.resource_achieved)["coordinate_work_units"] <= 1_000
    assert evidence.execution_evidence is not None
    np.testing.assert_array_equal(np.asarray(source.geometry.coordinates), before)


def test_public_native_sphere_query_refusal_retains_ended_status_and_source(
    sphere_metric_source: _SphereProducerFixture,
) -> None:
    import phydrax as phx
    from phydrax._meshcore import MeshcoreStatus

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
    assert (
        int(result.evidence.execution_evidence.total_geometry_queries)
        == limits.maximum_geometry_queries
    )
    np.testing.assert_array_equal(source.geometry.coordinates, before)


def test_public_fine_sphere_metric_preserves_original_all_phase_capacity(
    fine_sphere_metric_source: _SphereProducerFixture,
) -> None:
    import phydrax as phx

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
        raise TypeError("The fine sphere route requires its actual metric evidence.")
    assert adaptation.status is not phx.meshing.MeshAdaptationStatus.RESOURCE_LIMIT
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
    ledger = CoordinateEnclosureBudget(10_000, 10_000)
    with pytest.raises(CoordinateEnclosureResourceError):
        for polynomial in polynomials:
            ledger.retain_basis((polynomial,))


def test_rational_derivative_batch_reuses_only_exact_boxes_under_original_budget() -> (
    None
):
    from phydrax.geometry import BSplineSurfacePatch

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
    prepared = CoordinateEnclosureBudget(640_000, 81_920_000)
    with prepared.activate():
        lower, upper = patch.derivative_bounds_batch(boxes, order=2)
    cold_work = 0
    for row, query in enumerate(boxes):
        cold = CoordinateEnclosureBudget(640_000, 81_920_000)
        with cold.activate():
            expected_lower, expected_upper = patch.derivative_bounds_batch(
                query[None], order=2
            )
        np.testing.assert_array_equal(lower[row], expected_lower[0])
        np.testing.assert_array_equal(upper[row], expected_upper[0])
        cold_work += cold.work_units
    assert 0 < prepared.work_units < cold_work < 640_000
    first_work = prepared.work_units
    first_retained = prepared.retained_basis_bytes
    with prepared.activate():
        repeated_lower, repeated_upper = patch.derivative_bounds_batch(boxes, order=2)
    np.testing.assert_array_equal(repeated_lower, lower)
    np.testing.assert_array_equal(repeated_upper, upper)
    assert 0 < prepared.work_units - first_work < first_work
    assert prepared.retained_basis_bytes == first_retained
    before_first_jet = prepared.work_units
    with prepared.activate():
        prepared_first_lower, prepared_first_upper = patch.derivative_bounds_batch(
            boxes, order=1
        )
    cold_first = CoordinateEnclosureBudget(640_000, 81_920_000)
    with cold_first.activate():
        cold_first_lower, cold_first_upper = patch.derivative_bounds_batch(boxes, order=1)
    np.testing.assert_array_equal(prepared_first_lower, cold_first_lower)
    np.testing.assert_array_equal(prepared_first_upper, cold_first_upper)
    assert 0 < prepared.work_units - before_first_jet < cold_first.work_units
    assert prepared.retained_basis_bytes == first_retained
    starved = CoordinateEnclosureBudget(first_work - 1, 81_920_000)
    with starved.activate(), pytest.raises(CoordinateEnclosureResourceError):
        patch.derivative_bounds_batch(boxes, order=2)
    altered_points = np.asarray(patch.control_points).copy()
    altered_points[1, 1, 2] += 0.1
    altered = BSplineSurfacePatch(
        altered_points, patch.weights, patch.u_knots, patch.v_knots, 1, 1
    )
    with prepared.activate():
        changed_lower, changed_upper = altered.derivative_bounds_batch(boxes, order=2)
    independent = CoordinateEnclosureBudget(640_000, 81_920_000)
    with independent.activate():
        independent_lower, independent_upper = altered.derivative_bounds_batch(
            boxes, order=2
        )
    np.testing.assert_array_equal(changed_lower, independent_lower)
    np.testing.assert_array_equal(changed_upper, independent_upper)
    assert not np.array_equal(changed_lower, lower)


class _UVProducerFixture(NamedTuple):
    domain: MeshingDomain
    source: CellMeshingResult
    transfer: SurfaceAssociationTransfer
    policy: MeshAdaptationPolicy


@pytest.fixture(scope="module")
def rational_trim_metric_source() -> _UVProducerFixture:
    """One actual authored rational patch/trim producer under the original cap."""
    import phydrax as phx

    geometry, meshing = phx.geometry, phx.meshing
    controls = np.asarray(
        (
            ((0.75, 0.5), (0.75, 0.75), (0.5, 0.75)),
            ((0.5, 0.75), (0.25, 0.75), (0.25, 0.5)),
            ((0.25, 0.5), (0.25, 0.25), (0.5, 0.25)),
            ((0.5, 0.25), (0.75, 0.25), (0.75, 0.5)),
        ),
        dtype=np.float64,
    )
    loop = tuple(
        geometry.PatchCurveUse(
            index,
            geometry.BSplineCurve.bezier(points, (1.0, 0.75, 1.0)),
            0.0,
            1.0,
        )
        for index, points in enumerate(controls)
    )
    patch = geometry.BSplineSurfacePatch(
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
    domain = geometry.MeshingDomain(
        (geometry.MeshingSurfacePatch(patch, (loop,)),),
        tuple(geometry.MeshingDomainCurve(index, (index + 1) % 4) for index in range(4)),
        4,
        source_id="native-rational-trim-metric",
        source_revision="authored",
    )
    scope = meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        meshing.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        np.asarray(domain.scope_indices(2), dtype=np.int64),
    )
    limits = MeshingLimits(
        maximum_vertices=20_000,
        maximum_edges=160_000,
        maximum_faces=160_000,
        maximum_cells=160_000,
        maximum_work_units=640_000,
        maximum_geometry_queries=1_280_000,
        maximum_scratch_bytes=81_920_000,
    )
    specification = meshing.SurfaceMeshingSpec(
        meshing.CellMeshingTarget(2, 3, meshing.CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(
            meshing.UniformSizeControl(
                scope, 0.16, maximum_size=0.24, strength=meshing.SizeControlStrength.HARD
            ),
        ),
        size_compliance=meshing.SizeCompliancePolicy(
            relative_tolerance=0.5, target_statistics=("p50",)
        ),
        protected_features=(
            meshing.ProtectedFeature(
                scope, meshing.FeatureKind.SURFACE, maximum_deviation=0.01
            ),
        ),
        limits=limits,
    )
    source = (
        meshing.NativeMeshingProvider(meshing.NativeMeshingOptions("parametric_surface"))
        .plan(
            meshing.NativeSurfaceSource(domain),
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )
    if not isinstance(source, CellMeshingResult):
        raise TypeError(
            "The authored rational surface plan must produce its actual cell result."
        )
    retained = source.surface_source
    if not isinstance(retained, SurfaceSourceCharts):
        raise TypeError(
            "The authored rational surface must retain its original source charts."
        )
    retained.require_root(source.mesh, source.geometry)
    boundary = retained.boundary_source
    atlas = prepare_surface_source_root_atlas(
        domain,
        boundary.patches,
        source.coordinate_contract,
        maximum_support_queries=limits.maximum_geometry_queries,
    )
    transfer = SurfaceAssociationTransfer(
        PreparedSurfaceSourceSupport(
            domain,
            atlas,
            atlas.root_parameters,
            boundary,
            maximum_support_queries=limits.maximum_geometry_queries,
        )
    )
    policy = MeshAdaptationPolicy(
        MeshAdaptationRoute.NATIVE_SURFACE_METRIC,
        association_transfer=transfer,
        maximum_passes=4,
        relocation=False,
        limits=limits,
        geometry_transition=CellGeometryTransitionPolicy(
            reconstruction="bounded_chart_deformation",
            reconstruction_tolerance=0.02,
            maximum_evaluations=640_000,
        ),
    )
    return _UVProducerFixture(domain, source, transfer, policy)


def test_rational_source_required_growth_preserves_hard_density_and_geometry(
    rational_trim_metric_source: _UVProducerFixture,
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
def test_actual_rational_curve_witness_refuses_changed_source_strata(
    rational_trim_metric_source: _UVProducerFixture,
    component: str,
) -> None:
    from dataclasses import replace

    import equinox as eqx

    from phydrax.meshing._surface_association_transfer import (
        prepare_surface_curve_witness,
    )

    domain, source, transfer, policy = rational_trim_metric_source
    with _sphere_native_budget(policy.limits):
        witness = prepare_surface_curve_witness(transfer.support, source)
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
        with pytest.raises(ValueError):
            changed.validate_restored()
        witness.validate_restored()


def test_actual_rational_curve_split_keeps_independent_exact_material_anchor(
    rational_trim_metric_source: _UVProducerFixture,
) -> None:
    import phydrax.meshing._surface_metric as owner
    from phydrax.meshing._surface_association_transfer import (
        prepare_surface_curve_witness,
    )
    from phydrax.meshing._topology_edit import entity_keys, key_rows

    domain, source, transfer, policy = rational_trim_metric_source
    before = np.asarray(source.mesh.coordinates).copy()
    ledger = CoordinateEnclosureBudget(
        policy.limits.maximum_work_units, policy.limits.maximum_scratch_bytes
    )
    with _sphere_native_budget(policy.limits), ledger.activate():
        witness = owner.prepare_surface_metric_source(source, transfer)
        curve_witness = prepare_surface_curve_witness(transfer.support, source)
        cells = np.concatenate(
            [np.asarray(block.vertices) for block in source.mesh.blocks]
        )
        cell_ids = np.concatenate(
            [np.asarray(block.global_ids) for block in source.mesh.blocks]
        )
        vertices = np.asarray(source.mesh.vertex_global_ids)
        rows = key_rows(np.asarray(witness.cell_global_ids)[:, None], cell_ids[:, None])
        charts, patches = (
            np.asarray(witness.charts)[rows],
            np.asarray(witness.patches)[rows],
        )
        classifications = transfer.classes(source)
        class_rows = key_rows(
            np.asarray(source.mesh.entity_set(2).entity_ids)[:, None], cell_ids[:, None]
        )
        metric = np.broadcast_to(np.eye(3) / 0.14**2, (vertices.size, 3, 3)).copy()
        state = owner._SurfaceState(
            before.copy(),
            metric,
            cells.copy(),
            patches,
            charts.copy(),
            classifications[2].codes[class_rows],
            cell_ids.copy(),
            vertices.copy(),
            [{int(value)} for value in cell_ids],
            [{int(value)} for value in vertices],
            int(np.max(cell_ids)) + 1,
            int(np.max(vertices)) + 1,
        )
        state.curve_witness = curve_witness
        state.vertex_strata = [
            (int(dimension), int(index), np.asarray(parameters).copy())
            for dimension, index, parameters in zip(
                curve_witness.source_dimensions,
                curve_witness.source_indices,
                curve_witness.source_parameters,
                strict=True,
            )
        ]
        state.material_charts = np.asarray(
            [Fraction(float(value)) for value in charts.flat], dtype=object
        ).reshape(charts.shape)
        edges, _, _ = owner._edge_table(state)
        edge_rows = key_rows(
            entity_keys(source.mesh, 1), np.sort(vertices[edges], axis=1)
        )
        edge = edges[np.flatnonzero(classifications[1].dimensions[edge_rows] == 1)[0]]
        tokens = [state.vertex_strata[int(vertex)] for vertex in edge]
        curve, left, right = curve_witness.edge_interval(
            np.asarray([token[0] for token in tokens]),
            np.asarray([token[1] for token in tokens]),
            np.asarray([token[2] for token in tokens]),
        )
        cavity = np.flatnonzero(np.sum(np.isin(cells, edge), axis=1) == 2)
        material_midpoint = np.mean(
            state.material_charts[cavity[0], np.isin(cells[cavity[0]], edge)], axis=0
        )
        affine_uv_midpoint = np.mean(
            charts[cavity[0], np.isin(cells[cavity[0]], edge)], axis=0
        )
        vertex = state.points.shape[0]
        assert owner._split(domain, state, edge, metric[edge[0]], source_curve=True)
        point, uses = curve_witness.evaluate_curve(curve, (left + right) / 2)
        np.testing.assert_allclose(
            state.points[vertex], point, rtol=0.0, atol=domain.tolerance * domain.scale
        )
        actual_uv = next(chart for patch, _, chart in uses if patch == patches[cavity[0]])
        assert np.linalg.norm(actual_uv - affine_uv_midpoint) > domain.tolerance
        assert state.vertex_strata[vertex][0:2] == (1, curve)
        assert state.vertex_sources[vertex] == {int(vertices[value]) for value in edge}
        for row in np.flatnonzero(np.any(state.cells == vertex, axis=1)):
            slot = np.flatnonzero(state.cells[row] == vertex)[0]
            np.testing.assert_array_equal(
                state.material_charts[row, slot], material_midpoint
            )
            np.testing.assert_allclose(
                state.charts[row, slot], actual_uv, rtol=0.0, atol=domain.tolerance
            )
    np.testing.assert_array_equal(
        np.asarray(source.mesh.coordinates).view(np.uint64), before.view(np.uint64)
    )


def test_rational_surface_reconstruction_preserves_original_complete_map(
    rational_trim_metric_source: _UVProducerFixture,
) -> None:
    from phydrax.discretization._cell_geometry import (
        PolynomialComposedCellGeometryElement,
        SplineCellGeometryElement,
    )
    from phydrax.discretization._cell_geometry_transfer import (
        reconstruct_parametric_surface_cell_geometry,
    )
    from phydrax.discretization._coordinate_enclosure import (
        add,
        axes,
        constant,
        coordinate_expressions,
        expression_add,
        expression_compose,
        expression_parts,
        expression_scale,
        scale,
    )
    from phydrax.geometry._surface_source_support import SurfaceSourceRootAtlas
    from phydrax.meshing._surface_metric import prepare_surface_metric_source

    domain, source, transfer, policy = rational_trim_metric_source
    atlas = transfer.support.original
    assert isinstance(atlas, SurfaceSourceRootAtlas)
    witness = prepare_surface_metric_source(source, transfer)
    original_controls = np.asarray(atlas.geometry.coordinates).copy()
    source_coordinates = np.asarray(source.geometry.coordinates).copy()
    originals, original_routes, _ = atlas.geometry.resolve(atlas.mesh)
    original = originals[0]
    assert isinstance(original, SplineCellGeometryElement)
    ledger = CoordinateEnclosureBudget(
        policy.limits.maximum_work_units, policy.limits.maximum_scratch_bytes
    )
    with _sphere_native_budget(policy.limits), ledger.activate():
        original_maps = coordinate_expressions(
            original, original_controls[np.asarray(original_routes[0])[0]]
        )
        assert original_maps is not None
        reconstruction = reconstruct_parametric_surface_cell_geometry(
            source.mesh,
            source.geometry,
            source.mesh,
            None,
            domain,
            domain_id=domain.domain_id,
            cell_ids=np.asarray(witness.cell_global_ids),
            cell_patches=np.asarray(witness.patches),
            cell_charts=np.asarray(witness.charts),
            cell_geometry_entity_ids=witness.geometry_entity_ids,
            cell_occurrence_paths=witness.occurrence_paths,
            maximum_fidelity=0.01,
            policy=policy.geometry_transition,
            source_atlas=atlas,
        )
        mesh = reconstruction.target_mesh
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
        elements, routes, _ = reconstruction.geometry.resolve(mesh)
        chart_rows = {
            int(identifier): row
            for row, identifier in enumerate(np.asarray(reconstruction.cell_global_ids))
        }
        variables = axes(2)
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
                exact = tuple(
                    tuple(Fraction(float(value)) for value in point) for point in charts
                )
                arguments = tuple(
                    add(
                        constant(exact[0][axis], 2),
                        add(
                            scale(variables[0], exact[1][axis] - exact[0][axis]),
                            scale(variables[1], exact[2][axis] - exact[0][axis]),
                        ),
                    )
                    for axis in range(2)
                )
                expected = tuple(
                    expression_compose(value, arguments) for value in original_maps
                )
                actual_maps = coordinate_expressions(element, original_controls[dofs])
                assert actual_maps is not None
                for actual, authored in zip(actual_maps, expected, strict=True):
                    numerator, _ = expression_parts(
                        expression_add(actual, expression_scale(authored, -1)), 2
                    )
                    assert not numerator
                # Independent bilinear rational map of the authored fixture,
                # at a nonnodal point. The exact identities above, not this
                # probe, establish whole-map preservation.
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
                    represented, physical, rtol=1.0e-13, atol=1.0e-14
                )
    np.testing.assert_array_equal(
        np.asarray(source.geometry.coordinates).view(np.uint64),
        source_coordinates.view(np.uint64),
    )
    np.testing.assert_array_equal(
        np.asarray(atlas.geometry.coordinates).view(np.uint64),
        original_controls.view(np.uint64),
    )


def _surface_reaction_load(points: Array, context: object) -> Array:
    del context
    return 1.0 + points[..., 0] ** 2


def _solve_surface_reaction(
    prepared: FiniteElementDiscretization,
    seed: Array | None = None,
) -> tuple[CompiledFiniteElementProblem, Array]:
    import phydrax as phx

    form = phx.equations.FiniteElementForm(
        "surface-reaction",
        "temperature",
        (
            phx.equations.MassAction("temperature", 1.0),
            phx.equations.SourceAction(
                "temperature",
                phx.equations.coefficient(
                    _surface_reaction_load, coefficient_id="ambient-temperature-reaction"
                ),
            ),
        ),
    )
    problem = phx.equations.compile_finite_element_problem(form, prepared)
    system, raw_rhs = problem.linear_system()
    space = prepared.field_spaces[0].vector_space
    if not isinstance(space, ArraySpace):
        raise TypeError(
            "The single temperature field requires its actual array coefficient space."
        )
    rhs = space.validate(raw_rhs)
    initial = space.validate(seed) if seed is not None else jnp.zeros_like(rhs)
    solved = phx.linalg.solve(system, rhs - space.validate(system.operator.mv(initial)))
    if not bool(solved.successful):
        raise ValueError("The physical surface reaction solve did not succeed.")
    value = initial + space.validate(solved.value)
    correction = phx.linalg.solve(system, rhs - space.validate(system.operator.mv(value)))
    if not bool(correction.successful):
        raise ValueError("The physical surface reaction refinement did not succeed.")
    value += space.validate(correction.value)
    np.testing.assert_allclose(
        np.asarray(space.validate(system.operator.mv(value))),
        np.asarray(rhs),
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    return problem, space.validate(problem.expand(value))


def test_native_rational_trim_metric_archive_reopens_and_continues_physical_fields(
    rational_trim_metric_source: _UVProducerFixture,
    tmp_path: Path,
) -> None:
    import phydrax as phx
    from phydrax._differentiation import BranchDifferentiationPolicy
    from phydrax._frozendict import frozendict
    from phydrax.discretization._views import FieldTracePolicy
    from phydrax.discretization.fem._precision import FiniteElementPrecisionPolicy
    from phydrax.discretization.fem._surface_chart_transfer import (
        prepare_surface_chart_field_transfer,
    )
    from phydrax.discretization.finite_volume import (
        prepare_unstructured_conservative_remap,
        UnstructuredFiniteVolumePlan,
    )
    from phydrax.lifecycle._meshing_field_records import (
        MeshingFieldDeclaration,
        MeshingFieldStateRole,
        prepare_meshing_field_index_binding,
        prepare_meshing_field_owner,
    )
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        validate_meshing_source_closure,
        write_meshing_source_closure,
    )
    from phydrax.meshing._assembly import MeshPart
    from phydrax.meshing._surface_association_transfer import SurfaceChartBoundarySource

    domain, source, _, policy = rational_trim_metric_source
    before = np.asarray(source.geometry.coordinates).copy()
    field = phx.discretization.FiniteElementFieldSpec(
        "temperature", phx.discretization.lagrange_element("triangle", 1)
    )
    source_space = phx.discretization.FiniteElementPlan(
        source.mesh, field, coordinate_spec=source.geometry
    ).prepare()
    source_problem, source_temperature = _solve_surface_reaction(source_space)
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
    adaptation = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source,
            phx.meshing.MetricMeshAdaptation(
                metric, coordinate_contract=source.coordinate_contract
            ),
            policy=policy,
        )
    )
    if not isinstance(adaptation.target, CellMeshingResult):
        raise TypeError("The surface metric route must return its actual cell carrier.")
    assert adaptation.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    assert adaptation.target.mesh.topology_id != source.mesh.topology_id
    assert adaptation.geometry_transition is not None
    chart = adaptation.geometry_transition.chart_deformation
    if not isinstance(chart, PreparedSurfaceChartDeformation):
        raise TypeError(
            "The rational trim route requires its actual UV deformation, not a sphere map."
        )
    assert chart.source_domain_coverage == chart.target_domain_coverage == "certified"
    validate_meshing_source_closure(chart)
    for partial in (chart.occurrences[0], chart.occurrences[0].pieces[0]):
        with pytest.raises(
            ValueError,
            match="lacks its original owning material deformation and source banks",
        ):
            validate_meshing_source_closure(partial)
    from dataclasses import replace

    from phydrax.discretization._surface_chart_deformation import SurfaceChartWitness

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
    with pytest.raises(ValueError):
        malformed_chart.require_bound(
            chart.source_mesh,
            chart.source_geometry,
            chart.target_mesh,
            chart.target_geometry,
        )
    assert (
        adaptation.target.certification is not None
        and adaptation.target.certification.passed
    )
    if not isinstance(adaptation.evidence, MetricRemeshingEvidence):
        raise TypeError(
            "The actual surface route must expose its owning metric evidence."
        )
    assert adaptation.evidence.execution_evidence is not None
    assert int(adaptation.evidence.execution_evidence.total_work_units) <= 640_000
    assert all(
        value.source_id == domain.source_id
        and value.source_revision == domain.source_revision
        for value in adaptation.target.associations
    )
    vertex_association = next(
        value
        for value in adaptation.target.associations
        if value.target_entity_set_id
        == adaptation.target.mesh.entity_set(0).entity_set_id
    )
    feature_rows = np.flatnonzero(np.asarray(vertex_association.source_dimensions) == 1)
    assert feature_rows.size > 0 and not vertex_association.exact
    actual_residuals = []
    vertex_rows = vertex_association.target_rows(
        np.asarray(adaptation.target.mesh.vertex_global_ids)
    )
    for row in feature_rows:
        curve = next(
            index
            for index in range(len(domain.curves))
            if domain.entity_id(1, index) == vertex_association.source_entity_ids[row]
        )
        patch, loop, position = domain.curve_owners[curve]
        use = domain.patches[patch].loops[loop][position]
        parameter = float(vertex_association.parameters[row, 0])
        fraction = (parameter - use.first) / (use.last - use.first)
        analytic = domain.curve_atlas.map(
            jnp.asarray((curve,), dtype=jnp.int32),
            jnp.asarray(((fraction,),), dtype=jnp.float64),
        )[0]
        target_row = int(np.flatnonzero(vertex_rows == row)[0])
        residual = np.linalg.norm(
            np.asarray(analytic)
            - np.asarray(adaptation.target.mesh.coordinates)[target_row]
        )
        np.testing.assert_allclose(
            np.asarray(vertex_association.residuals[row]),
            np.asarray(residual),
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        actual_residuals.append(residual)
    assert max(actual_residuals) <= domain.tolerance * domain.scale
    assert (
        source.certification is not None
        and source.certification.request.fidelity_tolerance is not None
    )
    assert max(actual_residuals) <= source.certification.request.fidelity_tolerance
    old_fv = UnstructuredFiniteVolumePlan.from_cell_mesh(
        source.mesh, component_names=("temperature",)
    ).prepare(cell_geometry=source.geometry)
    new_fv = UnstructuredFiniteVolumePlan.from_cell_mesh(
        adaptation.target.mesh, component_names=("temperature",)
    ).prepare(cell_geometry=adaptation.target.geometry)
    remap = prepare_unstructured_conservative_remap(
        old_fv,
        new_fv,
        provenance="native-rational-material-history",
        surface_chart_deformation=chart,
        adaptation=adaptation,
        policy=phx.geometry.CommonRefinementPolicy(
            maximum_exact_work=640_000, maximum_memory_bytes=81_920_000
        ),
    )
    if not isinstance(remap.evidence, MappedSurfaceChartRemapEvidence):
        raise TypeError(
            "The surface FV remap must expose its actual material correspondence "
            f"evidence, got {remap.evidence!r}."
        )
    assert remap.succeeded and remap.plan is not None and remap.evidence.passed
    remap_plan = remap.plan
    history = (1.0 + old_fv.cell_centers[:, 0] ** 2)[:, None]
    source_material = phx.equations.MaterialTransaction(
        (
            phx.equations.MaterialState(
                phx.equations.MaterialSiteId("surface-temperature-history"),
                "surface-temperature-history",
                history,
            ),
        )
    )
    carried_history = remap.plan.apply(
        source_material.state("surface-temperature-history").committed
    )
    defect = np.max(
        np.abs(np.asarray(remap.plan.conservation_defect(history, carried_history)))
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
    prepared = phx.discretization.FiniteElementPlan(
        adaptation.target.mesh,
        field,
        coordinate_spec=adaptation.target.geometry,
    ).prepare(numeric_version=adaptation.target.mesh.numeric_version)
    transfer = prepare_surface_chart_field_transfer(
        source_space,
        prepared,
        chart,
        field_name="temperature",
        source_geometry=source.geometry,
        target_geometry=adaptation.target.geometry,
        semantics="intensive",
        maximum_work=640_000,
    )
    assert transfer.evidence.passed
    carried_temperature = transfer.transfer.apply(source_temperature)
    from jax import jvp, vjp

    direction = jnp.cos(jnp.arange(source_temperature.size, dtype=jnp.float64)).reshape(
        source_temperature.shape
    )
    primal, tangent = jvp(transfer.transfer.apply, (source_temperature,), (direction,))
    np.testing.assert_allclose(primal, carried_temperature, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        tangent, transfer.transfer.apply(direction), rtol=1.0e-12, atol=1.0e-12
    )
    cotangent = jnp.sin(jnp.arange(primal.size, dtype=jnp.float64) + 0.5).reshape(
        primal.shape
    )
    _, pullback = vjp(transfer.transfer.apply, source_temperature)
    (source_cotangent,) = pullback(cotangent)
    np.testing.assert_allclose(
        jnp.vdot(tangent, cotangent),
        jnp.vdot(direction, source_cotangent),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    problem, temperature = _solve_surface_reaction(prepared, carried_temperature)
    material = phx.equations.MaterialTransaction(
        (
            phx.equations.MaterialState(
                phx.equations.MaterialSiteId("surface-temperature-history"),
                "surface-temperature-history",
                carried_history,
            ),
        )
    )
    from phydrax._meshcore import NativeHostStorageWorkspace
    from phydrax.solver import (
        FiniteElementAcceptedState,
        FiniteElementTopologyTransaction,
        MaterialTopologyTransferResult,
    )

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

    def move_material(
        value: phx.equations.MaterialTransaction,
        lineage: object,
        args: object,
        *,
        source_workspace: NativeHostStorageWorkspace,
    ) -> MaterialTopologyTransferResult:
        del lineage, args
        actual = prepare_unstructured_conservative_remap(
            old_fv,
            new_fv,
            provenance="native-rational-transaction-history",
            surface_chart_deformation=chart,
            adaptation=adaptation,
            policy=phx.geometry.CommonRefinementPolicy(
                maximum_exact_work=640_000, maximum_memory_bytes=81_920_000
            ),
            source_workspace=source_workspace,
        )
        if not actual.succeeded or actual.plan is None:
            raise ValueError(
                "The actual material history remap refused its original transaction allowance."
            )
        prior = value.state("surface-temperature-history")
        moved = actual.plan.apply(prior.committed)
        candidate = phx.equations.MaterialTransaction(
            (
                phx.equations.MaterialState(
                    prior.site_id,
                    prior.model_id,
                    moved,
                    state_version=prior.state_version + 1,
                ),
            )
        )
        return MaterialTopologyTransferResult(materials=candidate, remaps=(actual,))

    def certify_material(
        mesh: CellMesh,
        values: tuple[Array, ...],
        candidate: phx.equations.MaterialTransaction | None,
        lineage: object,
        args: object,
    ) -> bool:
        del lineage, args
        if candidate is None or mesh.mesh_id != adaptation.target.mesh.mesh_id:
            return False
        actual_history = candidate.state("surface-temperature-history").committed
        actual_defect = np.max(
            np.abs(np.asarray(remap_plan.conservation_defect(history, actual_history)))
        )
        return bool(
            adaptation.target.audit.passed
            and adaptation.target.certification is not None
            and adaptation.target.certification.passed
            and actual_defect <= bound
            and np.all(np.isfinite(np.asarray(values[0])))
            and np.allclose(
                np.asarray(values[0]),
                np.asarray(carried_temperature),
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )

    executor = FiniteElementTopologyTransaction(
        certify_material,
        fields=(field,),
        material_transfer=move_material,
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
        published.state.fields[0], carried_temperature, rtol=1.0e-12, atol=1.0e-12
    )
    np.testing.assert_allclose(
        published.state.materials.state("surface-temperature-history").committed,
        carried_history,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    _, continued_after_rebind = _solve_surface_reaction(
        prepared, published.state.fields[0]
    )
    np.testing.assert_allclose(
        continued_after_rebind, temperature, rtol=1.0e-12, atol=1.0e-12
    )
    assert accepted.state_version == 1 and published.state.state_version == 2
    assert published.state.schedule_cursor == accepted.schedule_cursor
    assert published.state.transition_id == published.receipt.receipt_id
    part = MeshPart(domain.source_id, adaptation.target)
    carrier = part.carrier
    if not isinstance(carrier, CellMeshingResult) or not isinstance(
        carrier.geometry, CellGeometrySpec
    ):
        raise TypeError(
            "The archived temperature owner requires its actual cell geometry carrier."
        )
    coefficient_binding = prepare_meshing_field_index_binding(
        carrier, "mesh/vertex_ids", field_space=prepared.field_spaces[0]
    )
    history_binding = prepare_meshing_field_index_binding(carrier, "mesh/cell_ids")
    roles = (
        MeshingFieldStateRole(
            "temperature",
            "temperature",
            "coefficients",
            temperature.shape,
            np.dtype(np.float64),
            1,
            index_binding=coefficient_binding,
        ),
        MeshingFieldStateRole(
            "material-temperature",
            "temperature",
            "material-history",
            carried_history.shape,
            np.dtype(np.float64),
            1,
            value_units=(phx.units.KELVIN,),
            index_binding=history_binding,
        ),
        MeshingFieldStateRole(
            "accepted-epoch", "temperature", "state-epoch", (), np.dtype(np.int64), 1
        ),
    )
    declaration = MeshingFieldDeclaration(
        part_name=part.name,
        owner="finite_element",
        topology_id=carrier.mesh.topology_id,
        geometry_layout_id=carrier.geometry.geometry_layout_id,
        field_space_ids=frozendict(
            {"temperature": prepared.field_spaces[0].field_space_id}
        ),
        value_units=frozendict({"temperature": (phx.units.KELVIN,)}),
        maximum_derivative_orders=frozendict({"temperature": 1}),
        branch_policy=BranchDifferentiationPolicy.SMOOTH,
        trace_policy=FieldTracePolicy("cell-sided"),
        state_roles=roles,
        history_policy="material-history",
        numeric_version=carrier.mesh.numeric_version,
        finite_element_fields=(field,),
        precision_policy=FiniteElementPrecisionPolicy(),
    )
    if carrier.certification is None:
        raise TypeError("The rational archive requires its accepted certification.")
    if adaptation.transition is None or adaptation.lineage is None:
        raise TypeError(
            "The rational archive requires its accepted source-to-target event."
        )
    from phydrax._fingerprint import canonical_fingerprint

    event_values = {
        "source_result_id": adaptation.source.result_id,
        "source_mesh_id": adaptation.source.mesh.mesh_id,
        "source_topology_id": adaptation.source.mesh.topology_id,
        "target_result_id": carrier.result_id,
        "target_mesh_id": carrier.mesh.mesh_id,
        "target_topology_id": carrier.mesh.topology_id,
        "transition_id": adaptation.transition.transition_id,
    }
    adaptation_event: dict[str, Any] = {
        **event_values,
        "lineage": adaptation.lineage,
        "accepted_event_id": canonical_fingerprint(
            {
                "kind": "accepted-serial-adaptation-event",
                **event_values,
                "lineage": adaptation.lineage.lineage_id,
            }
        ),
    }
    archive_records = {
        "accepted_target": carrier,
        "certification_inputs": carrier.certification.request,
        "report": carrier.certification,
        "associations": carrier.associations,
        "field_declarations": {part.name: declaration},
        "accepted_data": {
            "adaptation_event": adaptation_event,
            "fields": {
                "temperature": np.asarray(temperature),
                "material-temperature": np.asarray(carried_history),
                "accepted-epoch": np.asarray(1, dtype=np.int64),
            },
        },
    }
    from copy import copy

    malformed_carrier = copy(carrier)
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
    object.__setattr__(malformed_carrier, "provider", altered_provider)
    runtime = carrier.runtime
    altered_runtime = type(runtime)(
        altered_provider.provider_id,
        runtime.actual_version,
        runtime.execution_mode,
        deterministic=runtime.deterministic,
        enforced_limits=runtime.enforced_limits,
        unenforced_limits=runtime.unenforced_limits,
    )
    object.__setattr__(malformed_carrier, "runtime", altered_runtime)
    from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
    from phydrax.lifecycle._meshing_sources import _native_authority_equal
    from phydrax.meshing._assembly import MeshCarrierKind

    fresh_part = MeshPart(
        part.name,
        carrier,
        coordinate_contract=part.coordinate_contract,
    )
    assert _native_authority_equal(
        part,
        fresh_part,
        limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
        freshly_rebuilt=True,
    )
    name_fault = copy(part)
    object.__setattr__(name_fault, "name", f"{part.name}-authority-fault")
    kind_fault = copy(part)
    object.__setattr__(
        kind_fault,
        "carrier_kind",
        next(value for value in MeshCarrierKind if value is not part.carrier_kind),
    )
    contract_fault = copy(part)
    altered_contract = copy(part.coordinate_contract)
    object.__setattr__(altered_contract, "spatial_id", "0" * 64)
    object.__setattr__(contract_fault, "coordinate_contract", altered_contract)
    for fault in (name_fault, kind_fault, contract_fault):
        assert not _native_authority_equal(
            fault,
            fresh_part,
            limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
            freshly_rebuilt=True,
        )
    malformed_records = dict(archive_records)
    malformed_records["accepted_target"] = malformed_carrier
    with pytest.raises(
        ValueError,
        match="differs from its freshly certified native source authority",
    ):
        validate_meshing_source_closure(malformed_records)
    missing_certification = copy(carrier)
    object.__setattr__(missing_certification, "certification", None)
    missing_certification_records = dict(archive_records)
    missing_certification_records["accepted_target"] = missing_certification
    with pytest.raises(ValueError):
        validate_meshing_source_closure(missing_certification_records)
    missing_lineage_records = dict(archive_records)
    missing_lineage_event = dict(adaptation_event)
    missing_lineage_event["lineage"] = None
    missing_lineage_records["accepted_data"] = dict(
        archive_records["accepted_data"],
        adaptation_event=missing_lineage_event,
    )
    with pytest.raises((TypeError, ValueError)):
        validate_meshing_source_closure(missing_lineage_records)
    duplicate_owner_records = dict(archive_records)
    duplicate_owner_records["field_declarations"] = {
        part.name: declaration,
        f"{part.name}-duplicate": declaration,
    }
    with pytest.raises(ValueError):
        validate_meshing_source_closure(duplicate_owner_records)
    mislabeled_generation_records: dict[str, Any] = dict(archive_records)
    mislabeled_generation_records["generation_part"] = part
    with pytest.raises(ValueError, match="must not be mislabeled"):
        validate_meshing_source_closure(mislabeled_generation_records)
    receipt = write_meshing_source_closure(
        tmp_path / "native-rational-temperature.npz",
        archive_records,
    )
    reopened = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    cold = reopened["accepted_target"]
    assert isinstance(cold, CellMeshingResult) and cold.certification is not None
    boundary = cold.certification.request.source
    assert isinstance(boundary, SurfaceChartBoundarySource)
    cold_chart = boundary.deformation
    assert isinstance(cold_chart, PreparedSurfaceChartDeformation)
    validate_meshing_source_closure(cold_chart)
    cold_chart.require_bound(
        cold_chart.source_mesh, cold_chart.source_geometry, cold.mesh, cold.geometry
    )
    assert cold_chart.deformation_id == chart.deformation_id
    assert (
        cold_chart.material_root_witness.witness_id
        == chart.material_root_witness.witness_id
    )
    for target in (False, True):
        np.testing.assert_array_equal(
            cold_chart.material_charts(target=target),
            chart.material_charts(target=target),
        )
    for restored, original in zip(cold_chart.occurrences, chart.occurrences, strict=True):
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
    restored_space, _ = prepare_meshing_field_owner(
        reopened["field_declarations"][part.name], cold
    )
    if not isinstance(restored_space, FiniteElementDiscretization):
        raise TypeError(
            "The reopened temperature declaration must prepare its finite-element owner."
        )
    assert restored_space.prepared_id == prepared.prepared_id
    cold_problem, continued = _solve_surface_reaction(
        restored_space, jnp.asarray(reopened["accepted_data"]["fields"]["temperature"])
    )
    assert cold_problem.compilation_id == problem.compilation_id
    np.testing.assert_allclose(continued, temperature, rtol=1.0e-12, atol=1.0e-12)
    restored_material = phx.equations.MaterialTransaction(
        (
            phx.equations.MaterialState(
                phx.equations.MaterialSiteId("surface-temperature-history"),
                "surface-temperature-history",
                reopened["accepted_data"]["fields"]["material-temperature"],
            ),
        )
    )
    assert restored_material.transaction_id == material.transaction_id
    np.testing.assert_array_equal(
        source_material.state("surface-temperature-history").committed, history
    )
    np.testing.assert_array_equal(source.geometry.coordinates, before)


def test_public_rational_uv_resource_refusal_retains_the_actual_source(
    rational_trim_metric_source: _UVProducerFixture,
) -> None:
    import phydrax as phx

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


def test_rational_uv_capacity_refusal_rolls_back_an_actual_partial_edit(
    rational_trim_metric_source: _UVProducerFixture,
) -> None:
    import phydrax as phx

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
