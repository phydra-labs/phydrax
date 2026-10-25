#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Mark-only periodic closure retains live source proof and ended charges."""

import copy
from typing import NoReturn

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import NativeExecutionScope
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.meshing import _adaptation as adaptation_owner, _periodic as periodic_owner
from phydrax.meshing._native_preparation import NativePreparationLedger
from phydrax.meshing.providers import _native_periodic as provider_owner
from tests._support.periodic_meshing import periodic_seed_request


type PeriodicRefinement = tuple[
    adaptation_owner.MeshAdaptationResult, adaptation_owner.MeshAdaptationPolicy
]


@pytest.fixture(scope="module")
def periodic_refinement() -> PeriodicRefinement:
    M = phx.meshing
    source, specification = periodic_seed_request()
    result = (
        M.NativeMeshingProvider(M.NativeMeshingOptions("periodic_delaunay"))
        .plan(source, specification, coordinate_contract=phx.SpatialCoordinateContract.si())
        .execute()
    )
    policy = M.MeshAdaptationPolicy(
        M.MeshAdaptationRoute.NATIVE_BISECTION,
        limits=specification.limits,
        association_transfer=provider_owner.PeriodicAssociationTransfer(source, specification),
    )
    refined = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            result,
            M.MarkedMeshAdaptation(result.mesh.entity_set(2).entity_ids),
            policy=policy,
        )
    )
    assert refined.status is M.MeshAdaptationStatus.COMPLETE
    return refined, policy


def _prepare(
    periodic_refinement: PeriodicRefinement, *, incomplete: bool = False
) -> adaptation_owner.PreparedMeshAdaptation:
    refined, policy = periodic_refinement
    ids = np.asarray(refined.target.mesh.entity_set(2).entity_ids)
    return adaptation_owner.prepare_mesh_adaptation(
        refined.target,
        adaptation_owner.MarkedMeshAdaptation(
            (), ids[:1] if incomplete else ids, hierarchy=refined.hierarchy
        ),
        policy=policy,
    )


@pytest.mark.parametrize("incomplete", (False, True))
def test_closed_marks_equal_metered_query_without_source_repreparation(
    periodic_refinement: PeriodicRefinement, monkeypatch: pytest.MonkeyPatch, incomplete: bool
) -> None:
    prepared = _prepare(periodic_refinement, incomplete=incomplete)
    expected = periodic_owner.periodic_coarsening_cell_ids(prepared)

    def unexpected(
        source: phx.meshing.CellMeshingResult,
        request: adaptation_owner.MeshAdaptationRequest,
        policy: adaptation_owner.MeshAdaptationPolicy,
        /,
        *,
        source_preparation: adaptation_owner._RouteSourcePreparation,
        budget: CoordinateEnclosureBudget,
        scope: NativeExecutionScope,
    ) -> NoReturn:
        raise AssertionError("mark closure must not repeat source preparation")

    def unexpected_associations(
        self: provider_owner.PeriodicAssociationTransfer,
        source: phx.meshing.CellMeshingResult,
        /,
        *,
        scope: NativeExecutionScope,
        budget: CoordinateEnclosureBudget,
        _prepared_source: list[provider_owner._PeriodicSourceAssociationPreparation] | None = None,
    ) -> NoReturn:
        raise AssertionError("mark closure must not repeat source preparation")

    monkeypatch.setattr(adaptation_owner, "_check_route_request", unexpected)
    monkeypatch.setattr(adaptation_owner, "_resolve_constraints", unexpected)
    monkeypatch.setattr(
        provider_owner.PeriodicAssociationTransfer, "source_associations", unexpected_associations
    )
    closed = periodic_owner.prepare_periodic_coarsening(prepared)
    assert isinstance(closed.request, phx.meshing.MarkedMeshAdaptation)
    assert isinstance(prepared.request, phx.meshing.MarkedMeshAdaptation)
    assert closed.periodic_source_receipt is not None
    np.testing.assert_array_equal(closed.request.coarsen_cell_ids, expected)
    assert closed is not prepared
    assert closed.request is not prepared.request
    assert closed.source is prepared.source
    assert closed.policy is prepared.policy
    assert closed.constraints is prepared.constraints
    assert closed.request.hierarchy is prepared.request.hierarchy
    assert closed.periodic_source_proofs is prepared.periodic_source_proofs
    assert closed.periodic_source_proofs
    assert closed.periodic_source_receipt.proofs is prepared.periodic_source_proofs
    if incomplete:
        assert expected.size == 0
        assert closed.prepared_id != prepared.prepared_id


def test_receipt_imports_original_once_and_preserves_limits(
    periodic_refinement: PeriodicRefinement, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepare(periodic_refinement, incomplete=True)
    original = prepared.periodic_source_receipt
    imports: list[phx.meshing.NativeExecutionRecord] = []
    import_receipt = NativePreparationLedger.import_receipt

    def capture(
        self: NativePreparationLedger,
        scope: NativeExecutionScope,
        receipt: phx.meshing.NativeExecutionRecord,
        /,
        *,
        import_work: bool = True,
    ) -> None:
        imports.append(receipt)
        return import_receipt(self, scope, receipt, import_work=import_work)

    monkeypatch.setattr(NativePreparationLedger, "import_receipt", capture)
    closed = periodic_owner.prepare_periodic_coarsening(prepared)
    receipt = closed.periodic_source_receipt
    assert receipt is not None and original is not None
    record, prior = receipt.evidence, original.evidence
    assert len(imports) == 1 and imports[0] is prior
    assert record.imported_preparation_hex() == (prior.receipt_hex(),)
    assert record.owner_id == closed.prepared_id
    assert prior.owner_id == prepared.prepared_id
    assert record.owner_id != prior.owner_id
    # The closure chains the original preparation; raw counters are its own work.
    assert record.preparation_evidence is prior
    assert int(record.total_work_units) == int(prior.total_work_units) + int(
        record.work_counters.construction_work
    )
    assert int(record.total_work_units) > int(prior.total_work_units)
    assert int(record.total_geometry_queries) >= int(prior.total_geometry_queries)
    assert float(record.prior_elapsed_seconds) == 0.0
    assert float(record.total_elapsed_seconds) == pytest.approx(
        float(prior.total_elapsed_seconds) + float(record.elapsed_seconds)
    )
    assert int(record.host_storage_peak_bytes_upper) > 0
    assert receipt.expires_at == original.expires_at
    limits, accounted = adaptation_owner._preparation_execution_limits(closed)
    assert accounted is record
    assert limits.maximum_work_units == (
        prepared.policy.limits.maximum_work_units - int(record.total_work_units)
    )
    assert limits.maximum_geometry_queries == (
        prepared.policy.limits.maximum_geometry_queries - int(record.total_geometry_queries)
    )
    assert limits.maximum_seconds <= (
        prepared.policy.limits.maximum_wall_seconds - float(record.total_elapsed_seconds)
    )
    record.require_valid()
    prior.require_valid()


def test_closed_request_executes_same_target_and_reproves_target(
    periodic_refinement: PeriodicRefinement, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepare(periodic_refinement)
    selected = periodic_owner.periodic_coarsening_cell_ids(prepared)
    assert isinstance(prepared.request, adaptation_owner.MarkedMeshAdaptation)
    legacy = adaptation_owner.prepare_mesh_adaptation(
        prepared.source,
        adaptation_owner.MarkedMeshAdaptation((), selected, hierarchy=prepared.request.hierarchy),
        policy=prepared.policy,
    )
    closed = periodic_owner.prepare_periodic_coarsening(prepared)
    expected = adaptation_owner.execute_mesh_adaptation(legacy)
    target_proofs: list[CellMesh] = []
    prove = provider_owner.periodic_source_associations

    def capture(
        source: phx.meshing.NativePeriodicSource,
        specification: phx.meshing.SurfaceMeshingSpec | phx.meshing.VolumeMeshingSpec,
        target: CellMesh,
        /,
        *,
        scope: NativeExecutionScope,
        budget: CoordinateEnclosureBudget,
        geometry: CellGeometrySpec | None = None,
        _prepared_cover: provider_owner._PeriodicSourceAssociationPreparation | None = None,
    ) -> tuple[phx.meshing.GeometryAssociation, ...]:
        target_proofs.append(target)
        return prove(
            source, specification, target, scope=scope, budget=budget,
            geometry=geometry, _prepared_cover=_prepared_cover,
        )

    monkeypatch.setattr(provider_owner, "periodic_source_associations", capture)
    actual = adaptation_owner.execute_mesh_adaptation(closed)
    assert actual.status is expected.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    assert actual.target.mesh.topology_id == expected.target.mesh.topology_id
    np.testing.assert_array_equal(actual.target.mesh.coordinates, expected.target.mesh.coordinates)
    assert target_proofs
    actual.target.audit.require_passed()
    assert all(association.complete and association.exact for association in actual.target.associations)
    assert actual.target.execution_evidence is not None
    assert closed.periodic_source_receipt is not None
    assert actual.target.execution_evidence.preparation_evidence is closed.periodic_source_receipt.evidence


@pytest.mark.parametrize("kind", ("shallow", "stored", "reconstructed", "receipt"))
def test_equal_identity_copies_cannot_inherit_live_authority(
    periodic_refinement: PeriodicRefinement, kind: str
) -> None:
    prepared = _prepare(periodic_refinement)
    if kind == "shallow":
        detached = copy.copy(prepared)
    elif kind == "stored":
        detached = adaptation_owner.PreparedMeshAdaptation.from_stored_fields(prepared)
    elif kind == "receipt":
        assert prepared.periodic_source_receipt is not None
        restored = provider_owner._PeriodicSourceAssociationReceipt.from_stored_fields(
            prepared.periodic_source_receipt
        )
        detached = eqx.tree_at(lambda value: value.periodic_source_receipt, prepared, restored)
    else:
        detached = adaptation_owner.PreparedMeshAdaptation(
            prepared.source, prepared.request, prepared.policy, prepared.constraints,
            periodic_source_proofs=prepared.periodic_source_proofs,
            periodic_source_receipt=prepared.periodic_source_receipt,
        )
    assert detached.prepared_id == prepared.prepared_id
    with pytest.raises(ValueError, match="live unchanged"):
        periodic_owner.prepare_periodic_coarsening(detached)


@pytest.mark.parametrize("kind", ("source", "geometry", "association", "policy"))
def test_stale_scientific_binding_cannot_reuse_receipt(
    periodic_refinement: PeriodicRefinement, kind: str
) -> None:
    prepared = _prepare(periodic_refinement)
    if kind == "source":
        changed = eqx.tree_at(
            lambda value: value.source.mesh.coordinates,
            prepared, prepared.source.mesh.coordinates + 0.125,
        )
    elif kind == "geometry":
        changed = eqx.tree_at(
            lambda value: value.source.geometry,
            prepared, None, is_leaf=lambda value: value is None,
        )
    elif kind == "association":
        changed = eqx.tree_at(lambda value: value.source.associations, prepared, ())
    else:
        changed = eqx.tree_at(
            lambda value: value.policy, prepared,
            adaptation_owner.MeshAdaptationPolicy(adaptation_owner.MeshAdaptationRoute.NATIVE_BISECTION),
        )
    with pytest.raises(ValueError, match="live unchanged|preparation identity"):
        periodic_owner.prepare_periodic_coarsening(changed)


@pytest.mark.parametrize("name", ("cell_classes", "facet_classes", "protected_edge_mask", "fixed_vertex_mask"))
def test_in_place_constraint_mutation_refuses_live_reuse(
    periodic_refinement: PeriodicRefinement, name: str
) -> None:
    prepared = _prepare(periodic_refinement)
    values = getattr(prepared.constraints, name)
    before = values.copy()
    try:
        values[0] = not values[0] if values.dtype == np.bool_ else values[0] + 1
        with pytest.raises(ValueError, match="live unchanged"):
            periodic_owner.prepare_periodic_coarsening(prepared)
    finally:
        values[:] = before
