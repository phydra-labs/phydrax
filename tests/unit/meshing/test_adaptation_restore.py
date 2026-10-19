"""Retained adaptation preparation and accepted-transition reconstruction."""

from __future__ import annotations

import equinox as eqx
import numpy as np
import pytest

from phydrax._fingerprint import tree_content_equal
from phydrax._meshcore import NativeExecutionLimits, NativeExecutionScope
from phydrax.lifecycle._meshing_source_records import reconstruct_stored_fields
from phydrax.lifecycle._meshing_sources import register_meshing_source_artifacts
from phydrax.meshing import (
    BisectionCompatibility,
    BisectionEvidence,
    BisectionHierarchy,
    CellMeshTransition,
    execute_mesh_adaptation,
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationResult,
    MeshAdaptationRoute,
    prepare_mesh_adaptation,
    PreparedMeshAdaptation,
)
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
    with pytest.raises(ValueError, match=r"PreparedMeshAdaptation\.prepared_id"):
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
        rebuilt = reconstruct_stored_fields(transition)
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
        reconstruct_stored_fields(forged)
