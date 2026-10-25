"""Retained adaptation preparation and accepted-transition reconstruction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NoReturn

import equinox as eqx
import numpy as np
import pytest

from phydrax._fingerprint import tree_content_equal
from phydrax._meshcore import NativeExecutionLimits, NativeExecutionScope
from phydrax.discretization import CoordinateEnclosureBudget
from phydrax.discretization._cell_geometry import owned_coordinate_preparation
from phydrax.lifecycle import MeshingAcceptedEpoch, MeshingFieldDeclaration
from phydrax.lifecycle._meshing_source_records import reconstruct_stored_fields
from phydrax.lifecycle._meshing_sources import register_meshing_source_artifacts
from phydrax.meshing import (
    BisectionCompatibility,
    BisectionEvidence,
    BisectionHierarchy,
    CellMeshingResult,
    CellMeshTransition,
    execute_mesh_adaptation,
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationResult,
    MeshAdaptationRoute,
    OversetRegistration,
    prepare_mesh_adaptation,
    PreparedMeshAdaptation,
)
from phydrax.meshing._adaptation import MeshAdaptationRequest
from phydrax.meshing.qualification._native_lifecycle import NativeConsumerResult
from tests._support.cell_meshes import proposal_source
from tests._support.recipes import restore_with_identity


@pytest.fixture(scope="module")
def prepared() -> PreparedMeshAdaptation:
    source = proposal_source()
    ids = np.concatenate([np.asarray(block.global_ids) for block in source.mesh.blocks])
    return prepare_mesh_adaptation(
        source,
        MarkedMeshAdaptation(ids),
        policy=MeshAdaptationPolicy(
            MeshAdaptationRoute.NATIVE_BISECTION,
            compatibility=BisectionCompatibility.UNIFORM_REFINEMENT,
        ),
    )


@pytest.fixture(scope="module")
def accepted(prepared: PreparedMeshAdaptation) -> MeshAdaptationResult:
    return execute_mesh_adaptation(prepared)


@pytest.fixture(scope="module")
def coarsened(accepted: MeshAdaptationResult) -> MeshAdaptationResult:
    ids = np.concatenate(
        [np.asarray(block.global_ids) for block in accepted.target.mesh.blocks]
    )
    return execute_mesh_adaptation(
        prepare_mesh_adaptation(
            accepted.target,
            MarkedMeshAdaptation((), ids, hierarchy=accepted.hierarchy),
            policy=accepted.policy,
        )
    )


@pytest.mark.parametrize("active", (False, True), ids=("settled-caller", "active-caller"))
def test_preparation_preserves_actual_caller_budget(active: bool) -> None:
    from contextlib import nullcontext

    from tests._support.coordinate_references import native_meshing_request

    source = proposal_source()
    policy = MeshAdaptationPolicy(
        MeshAdaptationRoute.NATIVE_BISECTION,
        compatibility=BisectionCompatibility.UNIFORM_REFINEMENT,
    )
    with native_meshing_request(policy.limits) as (scope, budget, _):
        with budget.bind_ticket(scope.ticket_view()) if active else nullcontext():
            prior_ticket = budget.ticket
            if active:
                budget.reserve(1)
            prior_work = budget.work_units
            prepared = prepare_mesh_adaptation(
                source, MarkedMeshAdaptation(), policy=policy, budget=budget
            )
            assert budget.work_units >= prior_work
            if active:
                assert budget.ticket is prior_ticket
                assert not prior_ticket.settled
                budget.reserve(1)
            else:
                assert budget.ticket.settled
            assert prepared.preparation_evidence is not None
            assert prepared.source is source
        assert scope.remaining().remaining_work_units <= policy.limits.maximum_work_units - budget.work_units


def test_preparation_reconstruction_preserves_binding(
    prepared: PreparedMeshAdaptation,
) -> None:
    restored = PreparedMeshAdaptation.from_stored_fields(prepared)
    assert tree_content_equal(restored, prepared)
    assert restored.prepared_id == prepared.prepared_id
    assert restored.source.result_id == prepared.source.result_id
    assert restored.request.request_id == prepared.request.request_id
    assert restored.policy.policy_id == prepared.policy.policy_id
    np.testing.assert_array_equal(
        restored.constraints.cell_classes, prepared.constraints.cell_classes
    )
    np.testing.assert_array_equal(
        restored.constraints.protected_edge_keys, prepared.constraints.protected_edge_keys
    )


def test_preparation_reconstruction_refuses_changed_request(
    prepared: PreparedMeshAdaptation,
) -> None:
    altered = eqx.tree_at(lambda value: value.request, prepared, MarkedMeshAdaptation())
    # The changed request also orphans the retained preparation receipt's owner.
    with pytest.raises(
        ValueError,
        match=r"PreparedMeshAdaptation\.(prepared_id|preparation_evidence\.owner_id)",
    ):
        PreparedMeshAdaptation.from_stored_fields(altered)


def test_preparation_restores_inside_an_exhausted_native_reservation(
    prepared: PreparedMeshAdaptation,
) -> None:
    with NativeExecutionScope(NativeExecutionLimits(1, 0, 0, 1 << 20, 120.0)) as scope:
        ticket = scope.batch(1)
        ticket.consume(1)
        with ticket.native_target():
            restored = PreparedMeshAdaptation.from_stored_fields(prepared)
        ticket.settle()
    assert tree_content_equal(restored, prepared)


@pytest.mark.parametrize(
    "field", ("protected_edge_mask", "cell_classes", "metric_values")
)
def test_preparation_restoration_refuses_invalid_constraint_bindings(
    prepared: PreparedMeshAdaptation, field: str
) -> None:
    invalid = prepared.constraints._replace(**{field: np.zeros((1,), dtype=np.float64)})
    altered = eqx.tree_at(lambda value: value.constraints, prepared, invalid)
    with pytest.raises(
        ValueError, match=rf"PreparedMeshAdaptation\.constraints\.{field}"
    ):
        PreparedMeshAdaptation.from_stored_fields(altered)


def test_result_reconstruction_preserves_accepted_transition(
    accepted: MeshAdaptationResult,
) -> None:
    restored = MeshAdaptationResult.from_stored_fields(accepted)
    assert restored.result_id == accepted.result_id
    assert restored.target.result_id == accepted.target.result_id
    assert restored.transition is not None
    assert accepted.transition is not None
    assert restored.transition.transition_id == accepted.transition.transition_id
    assert isinstance(restored.hierarchy, BisectionHierarchy)
    assert isinstance(accepted.hierarchy, BisectionHierarchy)
    assert restored.hierarchy.hierarchy_id == accepted.hierarchy.hierarchy_id
    assert isinstance(restored.evidence, BisectionEvidence)
    assert isinstance(accepted.evidence, BisectionEvidence)
    assert restored.evidence.evidence_id == accepted.evidence.evidence_id
    assert restored.elapsed_seconds == accepted.elapsed_seconds


def test_result_reconstruction_refuses_changed_request(
    accepted: MeshAdaptationResult,
) -> None:
    altered = eqx.tree_at(lambda value: value.request, accepted, MarkedMeshAdaptation())
    with pytest.raises(ValueError, match=r"MeshAdaptationResult\.prepared_id"):
        MeshAdaptationResult.from_stored_fields(altered)


def _bounded_replay(transition: CellMeshTransition) -> CellMeshTransition:
    work, storage, queries = transition.reconstruction_preparation_bounds()
    with owned_coordinate_preparation(
        work, storage, maximum_geometry_queries=queries
    ) as ledger:
        return reconstruct_stored_fields(transition, bindings={"budget": ledger})


def test_transition_reconstruction_preserves_parent_and_coarsening_banks(
    accepted: MeshAdaptationResult,
    coarsened: MeshAdaptationResult,
) -> None:
    register_meshing_source_artifacts()
    assert accepted.transition is not None
    assert accepted.transition.parent_cell_ids is not None
    assert accepted.transition.parent_reference_vertices is not None
    assert coarsened.transition is not None
    assert coarsened.transition.coarsened_cell_ids is not None
    assert coarsened.transition.coarsened_cell_ids.size > 0
    assert coarsened.transition.coarsened_into_ids is not None
    assert coarsened.transition.coarsened_reference_vertices is not None
    for transition in (accepted.transition, coarsened.transition):
        # Replay binds its rational-bank ledger explicitly; it cannot bypass it.
        with pytest.raises(TypeError, match=r"CellMeshTransition\.budget"):
            reconstruct_stored_fields(transition)
        rebuilt = _bounded_replay(transition)
        assert rebuilt is not transition
        assert rebuilt.target is transition.target
        assert rebuilt.lineage is transition.lineage
        assert rebuilt.geometry_transition is transition.geometry_transition
        assert rebuilt.transition_id == transition.transition_id
        assert tree_content_equal(rebuilt, transition)
        for name in (
            "parent_cell_ids",
            "parent_reference_vertices",
            "coarsened_cell_ids",
            "coarsened_into_ids",
            "coarsened_reference_vertices",
        ):
            assert tree_content_equal(getattr(rebuilt, name), getattr(transition, name))


def test_transition_reconstruction_refuses_forged_identity(
    accepted: MeshAdaptationResult,
) -> None:
    register_meshing_source_artifacts()
    assert accepted.transition is not None
    forged = restore_with_identity(accepted.transition, "transition_id", "forged")
    assert isinstance(forged, CellMeshTransition)
    with pytest.raises(
        ValueError, match=r"stored_fields\.CellMeshTransition.*transition_id"
    ):
        _bounded_replay(forged)


@pytest.fixture(scope="module")
def consumed_target(accepted: MeshAdaptationResult) -> CellMeshingResult:
    from phydrax.meshing.qualification._native_lifecycle import execute_native_consumer

    def fail(stage: str, message: str, data: dict[str, object]) -> ValueError:
        return ValueError(f"{stage}: {message}; {data}")

    def consume(budget: CoordinateEnclosureBudget) -> None:
        budget.reserve(1, 0)

    return execute_native_consumer(
        accepted.target, accepted.policy.limits, consume, failure_factory=fail
    ).result


def test_accepted_endpoint_rebinding_preserves_exact_science_and_nested_target(
    accepted: MeshAdaptationResult,
    consumed_target: CellMeshingResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.meshing._adaptation as adaptation_owner
    from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
    from phydrax.lifecycle._meshing_field_records import (
        rebind_meshing_adaptation_execution_endpoints,
    )
    from phydrax.lifecycle._meshing_source_records import content_equal

    def refuse_helper_request(
        source: CellMeshingResult,
        request: MeshAdaptationRequest,
        /,
        *,
        policy: MeshAdaptationPolicy,
        budget: CoordinateEnclosureBudget | None = None,
        observe_managed_admission: bool = False,
    ) -> NoReturn:
        raise AssertionError("receipt-only rebinding must not prepare another request")

    monkeypatch.setattr(adaptation_owner, "prepare_mesh_adaptation", refuse_helper_request)
    assert not content_equal(
        accepted.target, consumed_target, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS
    )
    rebound = rebind_meshing_adaptation_execution_endpoints(
        accepted, accepted.source, consumed_target
    )
    assert rebound.source is accepted.source
    assert rebound.target is consumed_target
    assert rebound.transition is not None and accepted.transition is not None
    assert rebound.transition.target is consumed_target
    assert rebound.transition.transition_id == accepted.transition.transition_id
    assert rebound.prepared_id == accepted.prepared_id
    assert rebound.result_id == accepted.result_id
    assert rebound.request is accepted.request and rebound.policy is accepted.policy
    assert rebound.lineage is accepted.lineage and rebound.transfer is accepted.transfer
    assert rebound.geometry_transition is accepted.geometry_transition


@pytest.mark.parametrize("endpoint", ("source", "target"))
def test_accepted_endpoint_rebinding_refuses_scientific_changes(
    accepted: MeshAdaptationResult, consumed_target: CellMeshingResult, endpoint: str
) -> None:
    from phydrax.lifecycle._meshing_field_records import (
        rebind_meshing_adaptation_execution_endpoints,
    )

    original = accepted.source if endpoint == "source" else consumed_target
    changed = eqx.tree_at(
        lambda value: value.geometry.coordinates,
        original,
        original.geometry.coordinates + 0.125,
    )
    with pytest.raises(ValueError, match="changed scientific content"):
        rebind_meshing_adaptation_execution_endpoints(
            accepted,
            changed if endpoint == "source" else accepted.source,
            changed if endpoint == "target" else consumed_target,
        )


@pytest.mark.parametrize("history_only", (False, True))
def test_accepted_endpoint_rebinding_refuses_unrelated_or_history_only_receipts(
    accepted: MeshAdaptationResult, history_only: bool
) -> None:
    from phydrax.lifecycle._meshing_field_records import (
        rebind_meshing_adaptation_execution_endpoints,
    )
    from phydrax.meshing._measurements import NativeExecutionRecord

    previous = accepted.target.execution_evidence
    assert previous is not None
    with NativeExecutionScope(accepted.policy.limits.native_limits()) as scope:
        pass
    unrelated = NativeExecutionRecord(
        scope.completed_evidence(),
        owner_id=previous.owner_id,
        history_evidence=(previous,) if history_only else (),
    )
    changed = accepted.target.with_execution_evidence(unrelated)
    with pytest.raises(ValueError, match="actual preceding receipt"):
        rebind_meshing_adaptation_execution_endpoints(
            accepted, accepted.source, changed
        )


@pytest.fixture(scope="module")
def admitted_epoch_request(
    accepted: MeshAdaptationResult,
) -> NativeConsumerResult[MeshingAcceptedEpoch]:
    import jax.numpy as jnp

    from phydrax.discretization.fem._generic import (
        FiniteElementFieldSpec,
        FiniteElementPlan,
    )
    from phydrax.discretization.fem._reference import lagrange_element
    from phydrax.meshing.qualification._envelope import prepare_envelope_accepted_epoch
    from phydrax.meshing.qualification._native_lifecycle import execute_native_consumer

    def admit(budget: CoordinateEnclosureBudget) -> MeshingAcceptedEpoch:
        scalar = FiniteElementPlan(
            accepted.target.mesh,
            (FiniteElementFieldSpec("u", lagrange_element("triangle", 1)),),
            coordinate_spec=accepted.target.geometry,
            budget=budget,
        ).prepare(budget=budget)
        count = scalar.dof_maps[0].global_dof_count
        return prepare_envelope_accepted_epoch(
            accepted.target, scalar, jnp.zeros(count, dtype=jnp.float64),
            (jnp.ones(count, dtype=jnp.float64), scalar.dof_maps[0].dof_coordinates[:, 0]),
            1, part_name="domain", transition=accepted, budget=budget,
        )

    def fail(stage: str, message: str, data: dict[str, object]) -> ValueError:
        return ValueError(f"{stage}: {message}; {data}")

    return execute_native_consumer(
        accepted.target, accepted.policy.limits, admit, failure_factory=fail,
    )


def test_adaptation_epoch_receipt_sealing_preserves_admitted_banks_without_reproof(
    admitted_epoch_request: NativeConsumerResult[MeshingAcceptedEpoch],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.lifecycle import _meshing_field_records as field_owner

    epoch, carrier = admitted_epoch_request.value, admitted_epoch_request.result
    transition = epoch.transition
    assert isinstance(transition, MeshAdaptationResult)
    continued = field_owner.rebind_meshing_adaptation_execution_endpoints(
        transition, transition.source, carrier
    )

    def reproof_forbidden(
        owners: tuple[tuple[MeshingFieldDeclaration, Any], ...],
        arrays: Mapping[str, Any],
        /,
        *,
        index_source: CellMeshingResult | None = None,
        registration: OversetRegistration | None = None,
        require_complete: bool = True,
        source_validation: Any | None = None,
    ) -> NoReturn:
        pytest.fail("receipt sealing must not reprepare geometry or field owners")

    monkeypatch.setattr(field_owner, "_validate_field_owner_banks", reproof_forbidden)
    sealed = field_owner.rebind_meshing_accepted_epoch_execution(
        epoch, carrier, transition=continued
    )
    assert sealed.carrier is carrier
    assert sealed.transition is continued
    assert sealed.declarations is epoch.declarations
    assert sealed.fields is epoch.fields
    assert sealed.epoch_id == epoch.epoch_id


@pytest.mark.parametrize("defect", ("missing", "changed-request"))
def test_adaptation_epoch_receipt_sealing_refuses_changed_history(
    admitted_epoch_request: NativeConsumerResult[MeshingAcceptedEpoch], defect: str,
) -> None:
    from phydrax.lifecycle import _meshing_field_records as field_owner

    epoch, carrier = admitted_epoch_request.value, admitted_epoch_request.result
    transition = epoch.transition
    assert isinstance(transition, MeshAdaptationResult)
    continued = field_owner.rebind_meshing_adaptation_execution_endpoints(
        transition, transition.source, carrier
    )
    changed = None if defect == "missing" else eqx.tree_at(
        lambda value: value.request, continued, MarkedMeshAdaptation()
    )
    with pytest.raises(ValueError, match="transition"):
        field_owner.rebind_meshing_accepted_epoch_execution(
            epoch, carrier, transition=changed
        )
