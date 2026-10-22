# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""White-box identity contracts and consumer replay for initial certification."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax.meshing import MeshingFailure, MeshingFailureCategory
from phydrax.meshing._distributed_generation import (
    DistributedSurfaceConstruction,
    execute_distributed_surface_initial,
    PreparedDistributedSurfaceGeneration,
)
from phydrax.meshing._initial_certification import (
    _CollectiveSourcePayload,
    _initial_closures,
    InitialCollectiveMeshEvidence,
    InitialSurfacePremise,
)
from tests._support.distributed_meshing.device_generation import _prepare


@pytest.fixture
def source_payload() -> _CollectiveSourcePayload:
    return _CollectiveSourcePayload(
        "source",
        "revision",
        (("compiled", "compiled"),),
        "specification",
        "limits",
        np.asarray((0.25, np.inf), dtype=np.float64),
        "content",
        "organization",
    )


def test_collective_source_fingerprint_preserves_identity_payload(
    source_payload: _CollectiveSourcePayload,
) -> None:
    expected = canonical_fingerprint(
        {
            "kind": "collective-initial-source-theorem",
            "source": "source",
            "revision": "revision",
            "compiled": "compiled",
            "specification": "specification",
            "limits": "limits",
            "physical_angles": array_tree_fingerprint(
                np.asarray((0.25, np.inf), dtype=np.float64)
            ),
            "content": "content",
            "organization": "organization",
        }
    )
    assert source_payload.fingerprint() == expected


@pytest.mark.parametrize(
    "field",
    (
        "source",
        "revision",
        "generation",
        "specification",
        "limits",
        "content",
        "organization",
    ),
    ids=(
        "source",
        "revision",
        "compiled",
        "specification",
        "limits",
        "content",
        "organization",
    ),
)
def test_collective_source_fingerprint_binds_each_contract_field(
    source_payload: _CollectiveSourcePayload,
    field: str,
) -> None:
    value = (("compiled", "different"),) if field == "generation" else "different"
    changed = replace(source_payload, **{field: value})
    assert changed.fingerprint() != source_payload.fingerprint()


def test_collective_source_fingerprint_binds_physical_angles(
    source_payload: _CollectiveSourcePayload,
) -> None:
    changed = replace(
        source_payload,
        physical_angles=np.asarray((0.125, np.inf), dtype=np.float64),
    )
    assert changed.fingerprint() != source_payload.fingerprint()


@pytest.fixture(scope="module")
def initial_inputs() -> tuple[
    PreparedDistributedSurfaceGeneration, DistributedSurfaceConstruction
]:
    prepared = _prepare((0, 0))
    generated = execute_distributed_surface_initial(prepared, minimum_angle=0.25)
    return prepared, generated


@pytest.fixture(scope="module")
def initial_evidence(
    initial_inputs: tuple[
        PreparedDistributedSurfaceGeneration, DistributedSurfaceConstruction
    ],
) -> InitialCollectiveMeshEvidence:
    prepared, generated = initial_inputs
    return InitialCollectiveMeshEvidence(prepared, generated, minimum_angle=0.25)



def test_initial_evidence_retains_constructed_rows_not_unused_layout_capacity(
    initial_inputs: tuple[
        PreparedDistributedSurfaceGeneration, DistributedSurfaceConstruction
    ],
    initial_evidence: InitialCollectiveMeshEvidence,
) -> None:
    prepared, generated = initial_inputs
    arrays = dict(initial_evidence.logical_arrays)
    assert arrays["coordinates"].shape[0] == generated.vertex_ids.size
    assert arrays["cell_global_ids"].shape[0] == generated.cell_ids.size
    assert arrays["closure/cell_ids"].shape[1] == min(
        prepared.layout.cell_capacity, generated.global_cell_count
    )
    assert arrays["coordinates"].shape[0] < prepared.layout.vertex_capacity
    assert arrays["cell_global_ids"].shape[0] < prepared.layout.cell_capacity
    premise = initial_evidence.premise
    assert isinstance(premise, InitialSurfacePremise)
    assert any(
        leaf is premise.patch_owners for leaf in jax.tree_util.tree_leaves(premise)
    )


@pytest.mark.parametrize("degree", (0, 1, 2), ids=("corner", "curve", "patch"))
def test_initial_source_binding_rejects_changed_stratum_coordinates(
    initial_inputs: tuple[
        PreparedDistributedSurfaceGeneration, DistributedSurfaceConstruction
    ],
    degree: int,
) -> None:
    prepared, generated = initial_inputs
    construction = generated.construction
    if construction is None:
        raise AssertionError("fixture.construction must contain generated rows; got None")
    rows = np.flatnonzero(construction.vertex_source_dimensions == degree)
    assert rows.size > 0
    coordinates = construction.vertices.copy()
    coordinates[rows[0], 0] += 0.125
    changed = replace(
        generated,
        construction=eqx.tree_at(lambda value: value.vertices, construction, coordinates),
    )
    with pytest.raises(MeshingFailure, match="initial.owner_checks") as rejected:
        InitialCollectiveMeshEvidence(prepared, changed, minimum_angle=0.25)
    assert rejected.value.category is MeshingFailureCategory.COMPLIANCE_FAILED
    findings = dict(rejected.value.evidence.logical_findings)
    assert "initial/failed_owner_category" in findings
    assert "initial/rejected_cell_ids" in findings


def test_initial_collective_evidence_replays_retained_content(
    initial_evidence: InitialCollectiveMeshEvidence,
) -> None:
    initial_evidence.require_current()
    assert bool(jnp.all(initial_evidence.global_checks))
    assert initial_evidence.global_entity_counts[0] > 0
    assert initial_evidence.global_entity_counts[2] > 0


def test_initial_cold_construction_preserves_nominal_boundary_occurrences(
    initial_inputs: tuple[
        PreparedDistributedSurfaceGeneration, DistributedSurfaceConstruction
    ],
    initial_evidence: InitialCollectiveMeshEvidence,
) -> None:
    from phydrax.meshing._initial_certification import restore_initial_construction

    original = initial_inputs[1]
    restored = restore_initial_construction(initial_evidence, 0)
    assert original.construction is not None and restored.construction is not None
    np.testing.assert_array_equal(restored.cell_ids, original.cell_ids)
    np.testing.assert_array_equal(
        restored.construction.boundary_occurrences,
        original.construction.boundary_occurrences,
    )
    assert restored.construction.boundary_occurrences.dtype == np.dtype(np.int32)
    banks = dict(initial_evidence.logical_arrays)
    assert "initial/cell_boundary_occurrences" in banks
    assert "initial/input/boundary_occurrences" in banks


def test_initial_collective_evidence_rejects_changed_numerical_content(
    initial_evidence: InitialCollectiveMeshEvidence,
) -> None:
    banks = dict(initial_evidence.logical_arrays)
    banks["coordinates"] = banks["coordinates"].at[0, 0].add(1.0)
    changed = eqx.tree_at(
        lambda evidence: evidence.logical_arrays,
        initial_evidence,
        tuple(sorted(banks.items())),
    )
    with pytest.raises(MeshingFailure, match="initial.content_id"):
        changed.require_current()


def test_initial_collective_evidence_rejects_changed_global_verdict(
    initial_evidence: InitialCollectiveMeshEvidence,
) -> None:
    changed = eqx.tree_at(
        lambda evidence: evidence.global_checks,
        initial_evidence,
        jnp.zeros(initial_evidence.global_checks.shape, dtype=jnp.bool_),
    )
    with pytest.raises(MeshingFailure, match="initial.global_checks"):
        changed.require_current()


def test_initial_closure_depth_is_earned_and_capacity_never_truncates_acceptance() -> None:
    vertices = np.arange(100, 160, 10, dtype=np.int64)
    cells = np.asarray(((0, 1, 2), (1, 3, 2), (2, 3, 4), (3, 5, 4)), dtype=np.int32)
    ids = np.asarray((41, 52, 63, 74), dtype=np.int64)
    arrays = {
        "vertex_global_ids": jnp.asarray(vertices),
        "coordinates": jnp.asarray(((0., 0.), (1., 0.), (0., 1.), (1., 1.), (0., 2.), (1., 2.))),
        "cell_global_ids": jnp.asarray(ids),
        "cell_vertices": jnp.asarray(vertices[cells]),
        "cell_owners": jnp.asarray((0, 1, 1, 1), dtype=jnp.int32),
    }
    first, first_fits = _initial_closures(
        arrays, vertex_count=6, cell_count=4, capacity=4, parts=2, neighborhood_depth=1
    )
    second, second_fits = _initial_closures(
        arrays, vertex_count=6, cell_count=4, capacity=4, parts=2, neighborhood_depth=2
    )
    assert bool(first_fits) and bool(second_fits)
    np.testing.assert_array_equal(
        np.asarray(first["closure/cell_ids"])[0][np.asarray(first["closure/cell_valid"])[0]],
        ids[:3],
    )
    np.testing.assert_array_equal(second["closure/cell_ids"][0], ids)
    saturated, saturated_fits = _initial_closures(
        arrays, vertex_count=6, cell_count=4, capacity=4, parts=2, neighborhood_depth=1_000_000
    )
    assert bool(saturated_fits)
    np.testing.assert_array_equal(saturated["closure/cell_ids"], second["closure/cell_ids"])
    _, truncated = _initial_closures(
        arrays, vertex_count=6, cell_count=4, capacity=3, parts=2, neighborhood_depth=2
    )
    assert not bool(truncated)
