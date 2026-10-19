# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from typing import Any

import pytest

import phydrax as phx
from tests._support.periodic_operator_cases import (
    periodic_task,
    solution_binding,
    trained_operator,
)


def _assert_task(task: Any, revision: str) -> None:
    assert task.task_id == "periodic-map"
    assert task.revision == revision
    assert task.field_by_name["input"].source_name == "u"
    assert task.field_by_name["solution"].query_name == "solution-query"
    assert task.field_by_name["input"].scale == (2.0,)
    assert task.field_by_name["solution"].scale == (3.0,)


def _assert_binding(binding: dict[str, Any], task: Any) -> None:
    expected = task.field_by_name["solution"].value_port().port_id
    assert set(binding["output_ports"]) == {"output"}
    assert binding["output_ports"]["output"].port_id == expected
    assert binding["port_mapping"].outputs == ((expected, expected),)


def test_periodic_task_preserves_requested_revision_and_field_contract() -> None:
    _assert_task(periodic_task(revision="fixture-revision"), "fixture-revision")


def test_periodic_task_oracle_rejects_wrong_revision() -> None:
    with pytest.raises(AssertionError):
        _assert_task(periodic_task(revision="wrong"), "fixture-revision")


def test_solution_binding_preserves_exact_target_port_identity() -> None:
    task = periodic_task()
    _assert_binding(solution_binding(task), task)


def test_solution_binding_oracle_rejects_dropped_route() -> None:
    task = periodic_task()
    corrupted = solution_binding(task)
    corrupted["port_mapping"] = phx.PortMapping(outputs=())
    with pytest.raises(AssertionError):
        _assert_binding(corrupted, task)


def test_trained_operator_preserves_task_binding_and_provenance() -> None:
    trained = trained_operator(revision="fixture-revision")
    _assert_task(trained.task, "fixture-revision")
    _assert_binding(
        {"output_ports": trained.output_ports, "port_mapping": trained.port_binding},
        trained.task,
    )
    assert dict(trained.provenance) == {"dataset": "immutable-dataset"}


def test_trained_operator_oracle_rejects_wrong_task_revision() -> None:
    trained = trained_operator(revision="wrong")
    with pytest.raises(AssertionError):
        _assert_task(trained.task, "fixture-revision")
