from contextlib import nullcontext
from typing import cast

import numpy as np
import pytest

from phydrax._identity import SemanticProvenance
from phydrax._meshcore import (
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionScope,
    NativeHostStorageWorkspace,
)
from phydrax.meshing._contracts import MeshingLimits
from phydrax.meshing.providers._native_publication import (
    _primitive_provenance_storage,
    _publication_provenance_bytes,
    coordinate_budget,
)


def _limits() -> MeshingLimits:
    return MeshingLimits(maximum_scratch_bytes=81_920_000, maximum_wall_seconds=120.0)


def test_live_ticket_handover_preserves_union_and_single_close() -> None:
    limits = _limits()
    values = np.arange(16, dtype=np.float64)
    with NativeExecutionScope(limits.native_limits()) as scope:
        with NativeHostStorageWorkspace(scope) as workspace:
            budget = coordinate_budget(limits)
            with budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size)):
                with budget.bind_ticket(scope.ticket_view()):
                    budget.retain_owner(values)
                before = scope.remaining().remaining_managed_bytes
                with workspace.cover_retained_storage(budget.retained_owner_objects):
                    workspace.retain_owner(values)
                    assert scope.remaining().remaining_managed_bytes == before
                    budget.handover_retained_storage(workspace.adopt_retained_storage)
                    assert scope.remaining().remaining_managed_bytes == before
                assert workspace.retained_owner_objects[id(values)] is values
                with pytest.raises(RuntimeError, match="detached ledger"):
                    budget.reserve(0)
            assert scope.remaining().remaining_managed_bytes == before
        assert scope.remaining().remaining_managed_bytes > before
        workspace.close()


def test_failed_handover_keeps_original_live_reservation() -> None:
    limits = _limits()
    values = np.arange(8, dtype=np.float64)
    with NativeExecutionScope(limits.native_limits()) as scope:
        with NativeHostStorageWorkspace(scope) as workspace:
            budget = coordinate_budget(limits)
            with budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size)):
                with budget.bind_ticket(scope.ticket_view()):
                    budget.retain_owner(values)
                ticket = budget.storage_ticket
                before = scope.remaining().remaining_managed_bytes

                def refuse(*args: object) -> None:
                    raise RuntimeError("receiver rejected before ownership commit")

                with pytest.raises(RuntimeError, match="receiver rejected"):
                    budget.handover_retained_storage(refuse)
                assert budget.storage_ticket is ticket
                assert ticket is not None and not ticket.settled
                assert scope.remaining().remaining_managed_bytes == before
                with budget.bind_ticket(scope.ticket_view()):
                    budget.reserve(0)
                budget.handover_retained_storage(workspace.adopt_retained_storage)
                assert scope.remaining().remaining_managed_bytes == before


def test_phase_workspace_adoption_transfers_same_token_atomically() -> None:
    limits = _limits()
    with NativeExecutionScope(limits.native_limits()) as scope:
        parent = NativeHostStorageWorkspace(scope).open()
        child = NativeHostStorageWorkspace(scope).open()
        child.set_bound(4096)
        before = scope.remaining().remaining_managed_bytes
        with pytest.raises(RuntimeError, match="separate live token"):
            parent.adopt_workspace(parent)
        assert scope.remaining().remaining_managed_bytes == before
        parent.adopt_workspace(child)
        assert scope.remaining().remaining_managed_bytes == before
        with pytest.raises(RuntimeError, match="transferred workspace"):
            child.set_bound(0)
        with pytest.raises(RuntimeError, match="through its owner"):
            child.close()
        parent.close()
        assert child.bound == 0
        assert scope.remaining().remaining_managed_bytes > before
        parent.close()


@pytest.mark.parametrize("shortfall", (0, 1))
def test_primitive_provenance_exact_headroom_before_normalization(shortfall: int) -> None:
    payload = {
        "kind": "native-polyhedral-volume-mesh",
        "unicode": "\U0001f600\n\"",
        "rows": [[1, -2, 0.125, None], [2**240, False, "shared"]],
        "tuple": ("x", (1, 2)),
    }
    bound = _publication_provenance_bytes(payload)
    normalized = _primitive_provenance_storage(payload)
    assert bound is not None and normalized is not None
    entered = False
    expected = pytest.raises(MeshcoreError) if shortfall else nullcontext()
    with expected as error:
        with NativeExecutionScope(_limits().native_limits()) as scope:
            with NativeHostStorageWorkspace(scope) as storage:
                storage.reserve_empty()
                with NativeHostStorageWorkspace(scope) as ballast:
                    ballast.reserve_empty()
                    available = scope.remaining().remaining_managed_bytes
                    ballast.set_bound(available - bound + shortfall)
                    assert scope.remaining().remaining_managed_bytes == bound - shortfall
                    storage.set_bound(bound)
                    entered = True
                    provenance = SemanticProvenance(payload)
                    assert len(provenance.content_json) <= normalized[1]
    assert entered is (shortfall == 0)
    if shortfall:
        assert error.value.status is MeshcoreStatus.CAPACITY_EXCEEDED


def test_scope_storage_refresh_keeps_live_owner_and_exact_tokens() -> None:
    limits = _limits()
    values = np.arange(16, dtype=np.float64)
    with NativeExecutionScope(limits.native_limits()) as scope:
        with NativeHostStorageWorkspace(scope) as destination:
            budget = coordinate_budget(limits)
            with budget.retained_storage(lambda _size: scope.ticket_view()):
                with budget.bind_ticket(scope.ticket_view()):
                    budget.retain_owner(values)
                view = budget.storage_ticket
                contributor = budget._storage_owner
                tokens = tuple(
                    workspace._reservation.value for workspace in view._workspaces.values()
                )
                assert tokens
                for _ in range(2):
                    with budget.bind_ticket(scope.ticket_view()):
                        budget.reserve(0)
                    assert budget.storage_ticket is view
                    assert budget._storage_owner is contributor
                    assert tuple(
                        workspace._reservation.value for workspace in view._workspaces.values()
                    ) == tokens
                budget.handover_retained_storage(destination.adopt_retained_storage)
                assert view.settled and view not in scope._ticket_views
                assert tuple(
                    workspace._reservation.value for workspace in destination._adopted_workspaces
                ) == tokens
                before = scope.remaining().remaining_managed_bytes
                contributor.close()
                contributor.close()
                assert scope.remaining().remaining_managed_bytes == before
                with pytest.raises(RuntimeError, match="detached ledger"):
                    budget.reserve(0)
            assert all(
                workspace._adopted_by is destination
                for workspace in destination._adopted_workspaces
            )


@pytest.mark.parametrize("shortfall", (0, 1))
def test_scope_storage_handover_metadata_real_r_boundary(
    monkeypatch: pytest.MonkeyPatch, shortfall: int,
) -> None:
    limits = _limits()
    values = np.arange(16, dtype=np.float64)
    with (
        pytest.raises(MeshcoreError) if shortfall else nullcontext()
    ) as terminal, NativeExecutionScope(limits.native_limits()) as scope:
        with NativeHostStorageWorkspace(scope) as destination:
            destination.reserve_empty()
            budget = coordinate_budget(limits)
            with budget.retained_storage(lambda _size: scope.ticket_view()):
                with budget.bind_ticket(scope.ticket_view()):
                    budget.retain_owner(values)
                view = budget.storage_ticket
                original = destination.set_bound
                attempts = []
                with NativeHostStorageWorkspace(scope) as ballast:
                    ballast.reserve_empty()

                    def exact(total):
                        attempts.append(total)
                        available = scope.remaining().remaining_managed_bytes
                        delta = total - destination.bound
                        ballast.set_bound(available - delta + shortfall)
                        original(total)

                    with monkeypatch.context() as patch:
                        patch.setattr(destination, "set_bound", exact)
                        if shortfall:
                            with pytest.raises(MeshcoreError) as failure:
                                budget.handover_retained_storage(
                                    destination.adopt_retained_storage
                                )
                            assert failure.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
                            assert budget.storage_ticket is view and not view.settled
                            assert not destination._adopted_workspaces
                            assert all(
                                workspace._adopted_by is None
                                for workspace in view._workspaces.values()
                            )
                        else:
                            budget.handover_retained_storage(
                                destination.adopt_retained_storage
                            )
                            assert view.settled
                    assert len(attempts) == 1
    if shortfall:
        assert terminal.value.status is MeshcoreStatus.CAPACITY_EXCEEDED


@pytest.mark.parametrize("shortfall", (0, 1))
def test_adaptation_publication_rejects_unknown_owner_at_its_work_boundary(
    monkeypatch: pytest.MonkeyPatch, shortfall: int,
) -> None:
    from phydrax._meshcore import NativeExecutionLimits
    from phydrax.meshing import _adaptation as adaptation

    # The argument tuple, None, then this unsupported owner are fresh visits.
    # No publication schema or ended receipt is manufactured for the negative.
    unknown = cast(adaptation.MeshAdaptationResult, object())
    inspected: list[int] = []
    fanout = adaptation.retained_host_storage_fanout_upper

    def inspect(value: object) -> int:
        inspected.append(id(value))
        return fanout(value)

    monkeypatch.setattr(adaptation, "retained_host_storage_fanout_upper", inspect)
    limits = _limits().native_limits()
    bounded = NativeExecutionLimits(
        3 - shortfall, 0, limits.maximum_cavity_cells,
        limits.maximum_managed_bytes, limits.maximum_seconds,
    )
    expected = MeshcoreError if shortfall else TypeError
    with pytest.raises(expected), NativeExecutionScope(bounded) as scope:
        imports = adaptation.NativeExecutionImports(scope)
        storage = adaptation._AdaptationPublicationStorage(scope)
        try:
            storage.admit(unknown, None, None, imports, None)
        finally:
            storage.close()
        pytest.fail("An unmapped owner cannot earn publication storage.")
    assert len(inspected) == 3 - shortfall
    assert (id(unknown) in inspected) is (not shortfall)
    assert scope.completed_evidence().externally_charged_work == 3 - shortfall
    assert scope.completed_evidence().host_storage_live_bytes_upper == 0
