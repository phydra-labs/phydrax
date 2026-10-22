#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import replace
from math import sqrt

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._meshcore import (
    MeshcoreError,
    meshcore_available,
    NativeExecutionScope,
    NativeHostStorageWorkspace,
)
from phydrax._strict import StrictModule
from phydrax.discretization import (
    CellGeometryTransitionPolicy,
    prepare_unstructured_conservative_remap,
    PreparedUnstructuredConservativeRemap,
    UnstructuredFiniteVolumePlan,
)
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.discretization.finite_volume import MappedSurfaceChartRemapEvidence
from phydrax.equations import MaterialTransaction
from phydrax.geometry import (
    BSplineSurfacePatch,
    CommonRefinementPolicy,
    LineCurve,
    MeshingDomain,
    MeshingDomainCurve,
    MeshingSurfacePatch,
    PatchCurveUse,
    PreparedCommonRefinement,
)
from phydrax.geometry._surface_source_support import (
    prepare_surface_source_root_atlas,
    PreparedSurfaceSourceSupport,
    SurfaceSourceCharts,
)
from phydrax.lifecycle import CompositionRebind
from phydrax.meshing import (
    MeshAdaptationPolicy,
    MeshAdaptationRoute,
    MeshingLimits,
    SizeControlStrength,
    SurfaceAssociationTransfer,
)
from phydrax.solver import (
    FiniteElementAcceptedState,
    FiniteElementTopologyTransaction,
    MaterialTopologyTransferResult,
)
from tests._support.cell_meshes import unit_square
from tests._support.mapped_cells import curved_surface_chart_fv
from tests._support.meshing_fixtures import surface_mesh, surface_specification
from tests._support.surface_metric import (
    prepare_rational_metric_fields,
    prepare_rational_metric_scenario,
    RationalMaterialHistory,
    RationalMetricFieldState,
    UVMetricSource,
)


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="phydrax-meshcore unavailable"
)


class _SuppliedMaterialRemap(StrictModule):
    """Supply a computed material candidate with a deliberately foreign remap."""

    materials: MaterialTransaction
    remap: PreparedUnstructuredConservativeRemap

    def __call__(
        self,
        accepted: MaterialTransaction,
        lineage: object,
        args: object,
        *,
        source_workspace: NativeHostStorageWorkspace,
        scope: NativeExecutionScope,
        budget: CoordinateEnclosureBudget,
    ) -> MaterialTopologyTransferResult:
        del accepted, lineage, args, source_workspace, scope, budget
        return MaterialTopologyTransferResult(
            materials=self.materials, remaps=(self.remap,)
        )


def _planar_metric_source() -> UVMetricSource:
    """Prepare four affine equilateral source cells under the unchanged limits."""
    # The authored source is the midpoint subdivision of the triangle with
    # corners (0,0,0), (1,1,0), (1,0,1). Each child edge has squared length 1/2.
    reference_points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.0, 1.0),
            (0.5, 0.5, 0.0),
            (1.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
        ),
        dtype=np.float64,
    )
    reference_cells = np.asarray(
        ((0, 3, 5), (3, 1, 4), (5, 4, 2), (3, 4, 5)), dtype=np.int32
    )
    corners = reference_points[reference_cells]
    edge_vectors = corners - np.roll(corners, -1, axis=1)
    np.testing.assert_array_equal(
        np.sum(edge_vectors * edge_vectors, axis=-1),
        np.full((4, 3), 0.5, dtype=np.float64),
    )
    source_size = sqrt(0.5)
    uv = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
    loop = tuple(
        PatchCurveUse(
            index,
            LineCurve(uv[index], uv[(index + 1) % 3] - uv[index]),
            0.0,
            1.0,
        )
        for index in range(3)
    )
    domain = MeshingDomain(
        (
            MeshingSurfacePatch(
                BSplineSurfacePatch(
                    np.asarray(
                        (
                            ((0.0, 0.0, 0.0), (1.0, 0.0, 1.0)),
                            ((1.0, 1.0, 0.0), (2.0, 1.0, 1.0)),
                        ),
                        dtype=np.float64,
                    ),
                    np.ones((2, 2), dtype=np.float64),
                    (0.0, 0.0, 1.0, 1.0),
                    (0.0, 0.0, 1.0, 1.0),
                    1,
                    1,
                ),
                (loop,),
            ),
        ),
        tuple(MeshingDomainCurve(index, (index + 1) % 3) for index in range(3)),
        3,
        source_id="material-remap-affine-triangle",
        source_revision="r1",
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
    source = surface_mesh(
        domain,
        surface_specification(
            domain, source_size, limits=limits, strength=SizeControlStrength.HARD
        ),
    )
    retained = source.surface_source
    assert isinstance(retained, SurfaceSourceCharts)
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
    return UVMetricSource(domain, source, transfer, policy)


@pytest.fixture(scope="module")
def chart_fields() -> RationalMetricFieldState:
    source = _planar_metric_source()
    scenario = prepare_rational_metric_scenario(source, metric_size=0.5)
    return prepare_rational_metric_fields(scenario)


def _assert_material_route_rejected(
    fields: RationalMetricFieldState,
    remap: PreparedUnstructuredConservativeRemap,
) -> None:
    """Assert route admission refuses before publishing or mutating accepted data."""
    scenario = fields.scenario
    source = scenario.source_fixture.source
    accepted = FiniteElementAcceptedState(
        (scenario.source_temperature,),
        jnp.asarray(0.0, dtype=jnp.float64),
        1,
        source.mesh.topology_id,
        scenario.source_space.prepared_id,
        scenario.source_problem.compilation_id,
        materials=fields.source_material,
        schedule_cursor=1,
        state_version=1,
    )
    accepted_id = accepted.accepted_id
    before_field = np.asarray(accepted.fields[0]).copy()
    before_history = np.asarray(
        fields.source_material.state("surface-temperature-history").committed
    ).copy()
    before_coordinates = np.asarray(source.geometry.coordinates).copy()
    material_history = RationalMaterialHistory(
        scenario.adaptation,
        fields.old_fv,
        fields.new_fv,
        scenario.chart,
        fields.remap_plan,
        fields.history,
        fields.carried_temperature,
        fields.bound,
        scenario.target,
    )
    supplied = _SuppliedMaterialRemap(fields.material, remap)
    publication_attempts: list[str] = []

    def observe_publication(request: CompositionRebind) -> CompositionRebind:
        publication_attempts.append("publication")
        return request

    transaction = FiniteElementTopologyTransaction(
        material_history.certify_material,
        fields=(scenario.field,),
        material_transfer=supplied,
        surface_chart_semantics={"temperature": "intensive"},
        composition_rebind=observe_publication,
        projection_policy=CommonRefinementPolicy(
            overlap_simplices=True,
            maximum_exact_work=640_000,
            maximum_memory_bytes=81_920_000,
        ),
        callback_owners=(
            supplied,
            material_history,
            fields.old_fv,
            fields.new_fv,
            scenario.source_space,
            fields.prepared,
        ),
    )
    with pytest.raises(
        ValueError,
        match=r"^(MappedSurfaceChartRemapEvidence\.deformation\.deformation_id|PreparedUnstructuredConservativeRemap\.route) must",
    ):
        transaction.execute(
            accepted,
            source.mesh,
            scenario.adaptation,
            compiled_layout_id=fields.problem.compilation_id,
        )
    assert publication_attempts == []
    assert accepted.accepted_id == accepted_id
    assert accepted.state_version == 1
    assert accepted.schedule_cursor == 1
    assert accepted.transition_id is None
    assert accepted.materials is fields.source_material
    np.testing.assert_array_equal(accepted.fields[0], before_field)
    np.testing.assert_array_equal(
        fields.source_material.state("surface-temperature-history").committed,
        before_history,
    )
    np.testing.assert_array_equal(source.geometry.coordinates, before_coordinates)


def test_chart_material_transaction_refuses_common_refinement_route(
    chart_fields: RationalMetricFieldState,
) -> None:
    geometry = UnstructuredFiniteVolumePlan.from_cell_mesh(unit_square()).prepare()
    remap = prepare_unstructured_conservative_remap(
        geometry, geometry, provenance="foreign-material-common-refinement"
    )
    assert remap.succeeded, remap.reason
    assert isinstance(remap.route, PreparedCommonRefinement)
    _assert_material_route_rejected(chart_fields, remap)


def test_chart_material_transaction_refuses_foreign_deformation_identity(
    chart_fields: RationalMetricFieldState,
) -> None:
    source, target, chart, transition = curved_surface_chart_fv(radius=2.0)
    remap = prepare_unstructured_conservative_remap(
        source,
        target,
        provenance="foreign-material-chart-correspondence",
        surface_chart_deformation=chart,
        source_geometry=source.cell_geometry,
        target_geometry=target.cell_geometry,
        geometry_transition=transition,
    )
    assert remap.succeeded, remap.reason
    assert isinstance(remap.route, MappedSurfaceChartRemapEvidence)
    expected = chart_fields.scenario.adaptation.geometry_transition
    assert expected is not None
    assert remap.route.deformation.deformation_id != expected.chart_correspondence_id
    _assert_material_route_rejected(chart_fields, remap)


def test_chart_transaction_uses_complete_active_request_work_allowance(
    chart_fields: RationalMetricFieldState,
) -> None:
    fields = chart_fields
    scenario = fields.scenario
    source = scenario.source_fixture.source
    accepted = FiniteElementAcceptedState(
        (scenario.source_temperature,),
        jnp.asarray(0.0, dtype=jnp.float64),
        1,
        source.mesh.topology_id,
        scenario.source_space.prepared_id,
        scenario.source_problem.compilation_id,
        materials=fields.source_material,
        state_version=1,
    )
    history = RationalMaterialHistory(
        scenario.adaptation,
        fields.old_fv,
        fields.new_fv,
        scenario.chart,
        fields.remap_plan,
        fields.history,
        fields.carried_temperature,
        fields.bound,
        scenario.target,
    )
    transaction = FiniteElementTopologyTransaction(
        history.certify_material,
        fields=(scenario.field,),
        material_transfer=history.move_material,
        surface_chart_semantics={"temperature": "intensive"},
        callback_owners=(history,),
    )
    limits = scenario.adaptation.policy.limits

    with pytest.raises(TypeError, match="budget"):
        transaction.execute(
            accepted,
            source.mesh,
            scenario.adaptation,
            budget="another-ledger",  # ty: ignore[invalid-argument-type]
        )

    def execute(cap: int) -> int:
        native_limits = replace(limits.native_limits(), maximum_work_units=cap)
        with NativeExecutionScope(native_limits) as scope:
            initial = scope.batch(0)
            initial.settle()
            ledger = CoordinateEnclosureBudget(
                initial,
                maximum_work_units=cap,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            with ledger.bind_ticket(scope.ticket_view()):
                # Preparation and transfer share this pre-consumed request.
                ledger.reserve(7)
                result = transaction.execute(
                    accepted,
                    source.mesh,
                    scenario.adaptation,
                    compiled_layout_id=fields.problem.compilation_id,
                    budget=ledger,
                )
                assert bool(result.committed), result.diagnostics
                assert result.execution_evidence is None
                assert len(result.material_remaps) == 1
                assert all(transfer.evidence.passed for transfer in result.transfers)
                assert result.state.materials is not None
                np.testing.assert_allclose(
                    result.state.materials.state("surface-temperature-history").committed,
                    fields.carried_history,
                    rtol=1e-12,
                    atol=1e-12,
                )
        assert scope.evidence is not None
        return scope.evidence.work_counters.construction_work

    required = execute(limits.maximum_work_units)
    assert required > 7
    assert execute(required) == required
    with pytest.raises((MeshcoreError, CoordinateEnclosureResourceError)):
        execute(required - 1)
    assert accepted.state_version == 1
    assert accepted.materials is fields.source_material
    np.testing.assert_array_equal(accepted.fields[0], scenario.source_temperature)
