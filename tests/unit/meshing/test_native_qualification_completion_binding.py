"""Qualification resolves genuine publication targets before its paid root ends."""

from typing import Any

import pytest

import phydrax as phx
from phydrax._meshcore import current_native_execution_scope, NativeExecutionScope
from phydrax.meshing._native_preparation import native_preparation_ledger
from phydrax.meshing.qualification._native_lifecycle import (
    execute_native_consumer,
    execute_native_request,
)


pytest_plugins = [
    "tests.unit.meshing.test_live_compartment_material_registration_archiveclosure",
]


def _failure(stage: str, message: str, data: dict[str, object]) -> ValueError:
    return ValueError(f"{stage}: {message}; {data}")


@pytest.mark.parametrize("new_request", (False, True), ids=("consumer", "request"))
def test_actual_qualification_target_is_resolved_once_inside_its_paid_root(
    cold_material_publication: tuple[Any, Any, Any, phx.meshing.CellMeshingResult],
    new_request: bool,
) -> None:
    _, specification, _, publication = cold_material_publication
    callbacks = []
    targets = []

    def consume(
        budget: phx.discretization.CoordinateEnclosureBudget,
    ) -> phx.meshing.CellMeshingResult:
        scope = current_native_execution_scope()
        assert scope is not None
        assert not budget.ticket.settled
        native_preparation_ledger(scope)
        budget.reserve(1, 0)
        callbacks.append(scope)
        return publication

    def target(value: phx.meshing.CellMeshingResult) -> phx.meshing.CellMeshingResult:
        scope = current_native_execution_scope()
        assert scope is callbacks[0]
        targets.append(scope)
        assert value is publication
        return value

    if new_request:
        executed = execute_native_request(
            publication, specification.limits, consume,
            target=target, failure_factory=_failure,
        )
    else:
        executed = execute_native_consumer(
            publication, specification.limits, consume,
            target=target, failure_factory=_failure,
        )

    assert len(callbacks) == len(targets) == 1
    assert executed.value is publication
    assert executed.result.execution_evidence is not None
    assert executed.result.execution_evidence.owner_id is not None
    assert current_native_execution_scope() is None
    receipt = executed.result.execution_evidence
    assert receipt.consumer_evidence is publication.execution_evidence
    if new_request:
        assert receipt.history_evidence == (publication.execution_evidence,)
        assert receipt.preparation_evidence is None
    else:
        assert receipt.preparation_evidence is publication.execution_evidence
    # A later root authenticates the actual completed wrapper, not its PREFIX.
    with NativeExecutionScope(specification.limits.native_limits()) as later:
        native_preparation_ledger(later).record_history(
            later, receipt, completed_source=executed.result,
        )
