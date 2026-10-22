# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Reprove real regenerated refinements without rewriting historical observations."""

import gc
import weakref
from copy import copy
from typing import TypedDict

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from phydrax._fingerprint import tree_content_equal
from phydrax._meshcore import meshcore_available, NativeParallelExecution
from phydrax.discretization import CellGeometrySpec, CellMesh
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry import _supermesh as supermesh_owner
from phydrax.geometry._supermesh import (
    CommonRefinementEvidence,
    Counts,
    Decomposition,
    Entries,
    PreparedCommonRefinement,
)
from phydrax.lifecycle._meshing_source_records import (
    reconstruct_stored_fields,
    register_meshing_source_types,
)
from phydrax.meshing import CellMeshingResult, MeshAdaptationResult, MeshingLimits
from tests._support.common_refinement import perturbed_triangles
from tests._support.coordinate_references import native_meshing_request
from tools._meshing_cases import prepare_polyhedral_case


pytestmark = [
    pytest.mark.meshcore,
    pytest.mark.skipif(not meshcore_available(), reason="native meshcore is unavailable"),
]


type _RegeneratedRefinement = tuple[
    CellMeshingResult, MeshAdaptationResult, MeshingLimits
]


class _RestoreBindings(TypedDict):
    source_mesh: CellMesh
    target_mesh: CellMesh
    source_geometry: CellGeometrySpec
    target_geometry: CellGeometrySpec
    budget: CoordinateEnclosureBudget | None


@pytest.fixture(scope="module")
def regenerated_refinement() -> _RegeneratedRefinement:
    source_authority, source_spec, source_options = prepare_polyhedral_case(
        "polyhedral-material-L",
        4,
        20000,
        120,
        source=True,
    )
    target_authority, target_spec, target_options = prepare_polyhedral_case(
        "polyhedral-material-L",
        4,
        20000,
        120,
        source=False,
    )
    with NativeParallelExecution(None, 120):
        source = (
            phx.meshing.NativeMeshingProvider(source_options)
            .plan(
                source_authority,
                source_spec,
                coordinate_contract=phx.SpatialCoordinateContract.si(),
            )
            .execute()
        )
        request = phx.meshing.PolyhedralMeshAdaptation(
            phx.meshing.PolyhedralAdaptationOperation.REGENERATE,
            complex_=target_authority.complex,
            sites=target_authority.sites,
            weights=target_authority.weights,
            schedule=target_options.polyhedral_schedule,
        )
        policy = phx.meshing.MeshAdaptationPolicy(
            phx.meshing.MeshAdaptationRoute.NATIVE_POLYHEDRAL,
            limits=target_spec.limits,
            audit_policy=phx.meshing.CellMeshAuditPolicy(
                require_complete_association=True,
                watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT,
            ),
        )
        result = phx.meshing.execute_mesh_adaptation(
            phx.meshing.prepare_mesh_adaptation(source, request, policy=policy)
        )
    assert result.common_refinement is not None
    assert result.common_refinement.succeeded
    assert result.common_refinement.source_cell_count == 2
    assert result.common_refinement.target_cell_count == 6
    register_meshing_source_types()
    return source, result, target_spec.limits


@pytest.fixture(scope="module")
def canonical_refinement() -> tuple[CellMesh, Decomposition, PreparedCommonRefinement]:
    mesh = perturbed_triangles(2, 0)
    original = supermesh_owner.prepare_common_refinement(mesh, mesh)
    assert original.succeeded
    decomposition = supermesh_owner._decompose(mesh, original.policy.predicate_mode)
    return mesh, decomposition, original


def _stored_entries(original: PreparedCommonRefinement) -> Entries:
    return Entries(
        target_cells=np.asarray(original.target_cells),
        source_cells=np.asarray(original.source_cells),
        volumes=np.asarray(original.volumes),
        first_moments=np.asarray(original.first_moments),
        second_moments=None,
        simplex_offsets=None,
        simplices=None,
    )


def _with_entries(
    decomposition: Decomposition,
    original: PreparedCommonRefinement,
    entries: Entries,
) -> PreparedCommonRefinement:
    return PreparedCommonRefinement(
        source=decomposition,
        target=decomposition,
        entries=entries,
        identities=(
            original.source_mesh_id,
            original.target_mesh_id,
            original.source_topology_id,
            original.target_topology_id,
        ),
        geometry_ids=(original.source_geometry_id, original.target_geometry_id),
        volume_error_bounds=(
            None
            if original.volume_error_bounds is None
            else np.asarray(original.volume_error_bounds)
        ),
        policy=original.policy,
        evidence=original.evidence,
    )


def test_constructor_identity_uses_canonical_stored_indices(
    canonical_refinement: tuple[CellMesh, Decomposition, PreparedCommonRefinement],
) -> None:
    mesh, decomposition, original = canonical_refinement
    entries = _stored_entries(original)
    canonical = _with_entries(decomposition, original, entries)
    wide = _with_entries(
        decomposition,
        original,
        entries._replace(
            source_cells=entries.source_cells.astype(np.int64),
            target_cells=entries.target_cells.astype(np.int64),
        ),
    )
    assert tree_content_equal(wide, canonical)
    assert tree_content_equal(original, canonical)
    assert wide.source_cells.dtype == wide.target_cells.dtype == np.dtype(np.int32)
    assert wide.target_offsets.dtype == np.dtype(np.int32)
    restored = PreparedCommonRefinement.from_stored_fields(
        wide, source_mesh=mesh, target_mesh=mesh
    )
    assert tree_content_equal(wide, restored)


@pytest.mark.parametrize("field", ("volumes", "first_moments"))
def test_constructor_identity_uses_canonical_stored_moments(
    canonical_refinement: tuple[CellMesh, Decomposition, PreparedCommonRefinement],
    field: str,
) -> None:
    _, decomposition, original = canonical_refinement
    entries = _stored_entries(original)
    narrow = getattr(entries, field).astype(np.float32)
    from_narrow = _with_entries(
        decomposition, original, entries._replace(**{field: narrow})
    )
    from_canonical = _with_entries(
        decomposition,
        original,
        entries._replace(**{field: narrow.astype(np.float64)}),
    )
    assert tree_content_equal(from_narrow, from_canonical)


@pytest.mark.parametrize("field", ("source_cells", "target_cells"))
@pytest.mark.parametrize(
    "invalid",
    (
        np.asarray([2**31 - 1], dtype=np.int64),
        np.asarray([2**31], dtype=np.int64),
        np.asarray([2**32], dtype=np.int64),
        np.asarray([2**64 - 1], dtype=np.uint64),
        np.asarray([-1], dtype=np.int64),
        np.asarray([0.0], dtype=np.float64),
        np.asarray([False], dtype=np.bool_),
    ),
)
def test_constructor_refuses_invalid_indices_before_narrowing(
    canonical_refinement: tuple[CellMesh, Decomposition, PreparedCommonRefinement],
    field: str,
    invalid: np.ndarray,
) -> None:
    _, decomposition, original = canonical_refinement
    entries = _stored_entries(original)
    invalid = np.full(getattr(entries, field).shape, invalid[0], dtype=invalid.dtype)
    with pytest.raises((TypeError, ValueError), match=field):
        _with_entries(decomposition, original, entries._replace(**{field: invalid}))


def test_constructor_refuses_overflowing_csr_offsets(
    canonical_refinement: tuple[CellMesh, Decomposition, PreparedCommonRefinement],
) -> None:
    _, decomposition, original = canonical_refinement
    entries = _stored_entries(original)._replace(
        volumes=np.broadcast_to(np.asarray(1.0), (2**31,))
    )
    with pytest.raises(ValueError, match="int32 CSR offsets"):
        _with_entries(decomposition, original, entries)


@pytest.mark.parametrize("field", ("source_cells", "target_cells", "target_offsets"))
@pytest.mark.parametrize("change_dtype", (False, True))
def test_restore_refuses_changed_index_values_or_dtype(
    canonical_refinement: tuple[CellMesh, Decomposition, PreparedCommonRefinement],
    field: str,
    change_dtype: bool,
) -> None:
    mesh, _, original = canonical_refinement
    values = getattr(original, field)
    changed = eqx.tree_at(
        lambda value: getattr(value, field),
        original,
        values.astype(np.int64) if change_dtype else values + np.int32(1),
    )
    with pytest.raises(ValueError, match=f"changed {field}"):
        PreparedCommonRefinement.from_stored_fields(
            changed, source_mesh=mesh, target_mesh=mesh
        )


def _bindings(
    source: CellMeshingResult,
    result: MeshAdaptationResult,
    budget: CoordinateEnclosureBudget | None,
) -> _RestoreBindings:
    return {
        "source_mesh": source.mesh,
        "target_mesh": result.target.mesh,
        "source_geometry": source.geometry,
        "target_geometry": result.target.geometry,
        "budget": budget,
    }


def test_real_regenerated_refinement_strictly_restores_historical_evidence(
    regenerated_refinement: _RegeneratedRefinement,
) -> None:
    source, result, limits = regenerated_refinement
    original = result.common_refinement
    assert original is not None
    with native_meshing_request(limits) as (_, budget, _):
        rebuilt = reconstruct_stored_fields(
            original, bindings=_bindings(source, result, budget)
        )
        assert tree_content_equal(original, rebuilt)
        assert rebuilt is not original
        assert rebuilt.evidence is not original.evidence
        assert rebuilt.refinement_id == original.refinement_id
        fresh_observations = [
            value
            for value in budget.retained_owner_objects.values()
            if type(value) is CommonRefinementEvidence and value is not rebuilt.evidence
        ]
        assert len(fresh_observations) == 1
        fresh = fresh_observations[0]
        assert fresh is not rebuilt.evidence
        # The diagnostic fixture has genuine adaptation predecessor storage;
        # standalone reproof has no such entry baseline. Neither is rewritten.
        assert fresh.working_bytes != original.evidence.working_bytes
        assert fresh.working_bytes <= budget.peak_bytes_upper
        assert fresh.retained_bytes == original.evidence.retained_bytes
        fresh_reference = weakref.ref(fresh)
        del fresh, fresh_observations
        gc.collect()
        assert fresh_reference() is not None
        assert budget.work_units > 0
        assert budget.work_units <= limits.maximum_work_units
        assert budget.peak_bytes_upper <= limits.maximum_scratch_bytes
        assert budget.temporary_bytes_upper == 0
        assert budget.retained_owner_objects[id(rebuilt)] is rebuilt


@pytest.mark.parametrize(
    "field", ("volumes", "first_moments", "source_measures", "volume_error_bounds")
)
def test_real_regenerated_refinement_refuses_changed_scientific_banks(
    regenerated_refinement: _RegeneratedRefinement,
    field: str,
) -> None:
    source, result, limits = regenerated_refinement
    original = result.common_refinement
    assert original is not None
    changed = eqx.tree_at(
        lambda value: getattr(value, field),
        original,
        getattr(original, field) + np.float64(0.125),
    )
    with native_meshing_request(limits) as (_, budget, _):
        with pytest.raises(ValueError, match="stored_fields.PreparedCommonRefinement"):
            reconstruct_stored_fields(changed, bindings=_bindings(source, result, budget))


@pytest.mark.parametrize(
    "field", ("source_coverage_defects", "target_coverage_tolerances")
)
def test_real_regenerated_refinement_refuses_changed_coverage_evidence(
    regenerated_refinement: _RegeneratedRefinement,
    field: str,
) -> None:
    source, result, limits = regenerated_refinement
    original = result.common_refinement
    assert original is not None
    changed = eqx.tree_at(
        lambda value: getattr(value.evidence, field),
        original,
        getattr(original.evidence, field) + np.float64(0.125),
    )
    with native_meshing_request(limits) as (_, budget, _):
        with pytest.raises(ValueError, match="CommonRefinementEvidence"):
            reconstruct_stored_fields(changed, bindings=_bindings(source, result, budget))


def test_real_regenerated_refinement_refuses_changed_execution_counts(
    regenerated_refinement: _RegeneratedRefinement,
) -> None:
    source, result, limits = regenerated_refinement
    original = result.common_refinement
    assert original is not None
    evidence = original.evidence
    altered = CommonRefinementEvidence(
        status=evidence.status,
        reason=evidence.reason,
        source_coverage_defects=np.asarray(evidence.source_coverage_defects),
        target_coverage_defects=np.asarray(evidence.target_coverage_defects),
        source_measures=np.asarray(original.source_measures),
        target_measures=np.asarray(original.target_measures),
        source_tolerances=np.asarray(evidence.source_coverage_tolerances),
        target_tolerances=np.asarray(evidence.target_coverage_tolerances),
        counts=Counts(
            evidence.candidate_pair_count + 1,
            evidence.accepted_pair_count,
            evidence.piece_pair_count,
            evidence.uncertain_predicate_count,
            evidence.invalid_cell_count,
            evidence.intersection_failure_count,
        ),
        retained_bytes=evidence.retained_bytes,
        working_bytes=evidence.working_bytes,
    )
    changed = eqx.tree_at(lambda value: value.evidence, original, altered)
    with native_meshing_request(limits) as (_, budget, _):
        with pytest.raises(
            ValueError, match="CommonRefinementEvidence.*candidate_pair_count"
        ):
            PreparedCommonRefinement.from_stored_fields(
                changed, **_bindings(source, result, budget)
            )


def test_real_regenerated_refinement_standalone_restore_keeps_optional_budget(
    regenerated_refinement: _RegeneratedRefinement,
) -> None:
    source, result, _ = regenerated_refinement
    assert result.common_refinement is not None
    restored = PreparedCommonRefinement.from_stored_fields(
        result.common_refinement, **_bindings(source, result, None)
    )
    assert restored is not result.common_refinement
    assert tree_content_equal(restored, result.common_refinement)


@pytest.mark.parametrize(
    "field", ("coverage_tolerance", "maximum_exact_work", "second_moments")
)
def test_real_regenerated_refinement_refuses_changed_policy(
    regenerated_refinement: _RegeneratedRefinement,
    field: str,
) -> None:
    source, result, limits = regenerated_refinement
    original = result.common_refinement
    assert original is not None
    changed = copy(original)
    policy = copy(original.policy)
    prior = getattr(policy, field)
    object.__setattr__(
        policy, field, not prior if isinstance(prior, bool) else prior + 0.125
    )
    object.__setattr__(changed, "policy", policy)
    with native_meshing_request(limits) as (_, budget, _):
        with pytest.raises((TypeError, ValueError), match="policy|maximum_exact_work"):
            PreparedCommonRefinement.from_stored_fields(
                changed, **_bindings(source, result, budget)
            )


def test_real_regenerated_refinement_refuses_changed_historical_observation_id(
    regenerated_refinement: _RegeneratedRefinement,
) -> None:
    source, result, limits = regenerated_refinement
    original = result.common_refinement
    assert original is not None
    changed = copy(original)
    evidence = copy(original.evidence)
    object.__setattr__(evidence, "working_bytes", evidence.working_bytes + 1)
    object.__setattr__(changed, "evidence", evidence)
    with native_meshing_request(limits) as (_, budget, _):
        with pytest.raises(ValueError, match="CommonRefinementEvidence"):
            PreparedCommonRefinement.from_stored_fields(
                changed, **_bindings(source, result, budget)
            )
