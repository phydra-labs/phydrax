# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""White-box identity contracts and consumer replay for initial certification."""

from dataclasses import replace

import equinox as eqx
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
    InitialCollectiveMeshEvidence,
)
from tests._support.distributed_meshing.device_generation import _prepare


@pytest.fixture
def source_payload() -> _CollectiveSourcePayload:
    return _CollectiveSourcePayload(
        "source",
        "revision",
        "compiled",
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
        "compiled",
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
    changed = replace(source_payload, **{field: "different"})
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
