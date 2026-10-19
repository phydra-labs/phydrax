"""Collective category arbitration and process-local exception preservation."""

from __future__ import annotations

import numpy as np
import pytest

from phydrax.meshing import (
    _collective_failure as failure_owner,
    MeshingFailure,
    MeshingFailureCategory,
)


def test_collective_barrier_accepts_unanimous_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def gather(code: np.int32, *, tiled: bool) -> np.ndarray:
        assert code == 0
        assert not tiled
        return np.zeros(3, dtype=np.int32)

    monkeypatch.setattr(failure_owner.multihost_utils, "process_allgather", gather)
    failure_owner.collective_failure_barrier(
        None, message="remote refusal", stage="commit"
    )


def test_collective_barrier_preserves_matching_local_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    categories = tuple(MeshingFailureCategory)
    failure = MeshingFailure(categories[0], "local refusal", stage="prepare")
    cause = ValueError("fixture.source must succeed; got injected failure")
    failure.__cause__ = cause

    def gather(code: np.int32, *, tiled: bool) -> np.ndarray:
        assert code == 1
        return np.asarray([code, 0, len(categories)], dtype=np.int32)

    monkeypatch.setattr(failure_owner.multihost_utils, "process_allgather", gather)
    with pytest.raises(MeshingFailure) as refusal:
        failure_owner.collective_failure_barrier(
            failure, message="remote refusal", stage="commit"
        )
    assert refusal.value is failure
    assert refusal.value.__cause__ is cause
    assert refusal.value.stage == "prepare"


@pytest.mark.parametrize(
    "local_failure", [False, True], ids=["successful-local", "lower-priority-local"]
)
def test_collective_barrier_uses_remote_priority_and_caller_evidence(
    monkeypatch: pytest.MonkeyPatch, local_failure: bool
) -> None:
    categories = tuple(MeshingFailureCategory)
    failure = (
        MeshingFailure(categories[-1], "local refusal", stage="prepare")
        if local_failure
        else None
    )

    def gather(code: np.int32, *, tiled: bool) -> np.ndarray:
        return np.asarray([code, 1, 0], dtype=np.int32)

    monkeypatch.setattr(failure_owner.multihost_utils, "process_allgather", gather)
    with pytest.raises(MeshingFailure) as refusal:
        failure_owner.collective_failure_barrier(
            failure, message="remote refusal", stage="commit"
        )
    assert refusal.value.category is categories[0]
    assert str(refusal.value) == "remote refusal"
    assert refusal.value.stage == "commit"
    assert refusal.value is not failure


def test_collective_decision_retains_process_codes_for_failure_findings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    categories = tuple(MeshingFailureCategory)
    codes = np.asarray([0, len(categories), 1, 0], dtype=np.int32)

    def gather(code: np.int32, *, tiled: bool) -> np.ndarray:
        assert code == len(categories)
        return codes

    monkeypatch.setattr(failure_owner.multihost_utils, "process_allgather", gather)
    decision = failure_owner.collective_failure_decision(categories[-1])
    np.testing.assert_array_equal(decision.rejected, codes)
    assert decision.code == len(categories)
    assert decision.category is categories[0]
