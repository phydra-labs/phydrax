# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from fractions import Fraction

import pytest

from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
from phydrax.discretization._coordinate_enclosure import constant
from phydrax.geometry._embedding_state import MeshCertificateLimits
from phydrax.meshing import MeshingLimits
from tests._support.coordinate_references import (
    coordinate_reference_budget,
    native_meshing_request,
)


def test_reference_budget_retains_domain_caps_and_settles_exact_work() -> None:
    limits = MeshCertificateLimits(
        maximum_work_units=1234, maximum_scratch_bytes=16 * 1024
    )
    with coordinate_reference_budget(limits, queries_upper=17) as budget:
        ticket = budget.ticket
        assert ticket.work_upper == limits.maximum_work_units
        assert ticket.queries_upper == 17
        assert ticket.bytes_upper == limits.maximum_scratch_bytes
        assert ticket.cavity_upper == 0
        assert budget.maximum_work_units == limits.maximum_work_units
        assert budget.maximum_memory_bytes == limits.maximum_scratch_bytes
        assert not ticket.settled
        assert constant(Fraction(1, 3), 2, budget=budget) == {(0, 0): Fraction(1, 3)}
        assert budget.work_units > 0
    assert ticket.settled
    assert ticket.work_units == budget.work_units


def test_reference_budget_settles_before_propagating_scientific_refusal() -> None:
    with pytest.raises(ValueError, match="reference-refusal"):
        with coordinate_reference_budget() as budget:
            ticket = budget.ticket
            constant(Fraction(2, 7), 2, budget=budget)
            raise ValueError("reference-refusal")
    assert ticket.settled
    assert ticket.work_units == budget.work_units


def test_reference_budget_refuses_one_unit_beyond_the_declared_domain_cap() -> None:
    limits = MeshCertificateLimits(maximum_work_units=13)
    with coordinate_reference_budget(limits) as budget:
        with pytest.raises(CoordinateEnclosureResourceError) as refusal:
            budget.reserve(14)
        assert refusal.value.resource == "coefficient_work"
        assert refusal.value.limit == 13
        assert refusal.value.requested == 14
    assert budget.ticket.settled


def test_native_request_retains_caps_and_closes_cache_before_scope() -> None:
    limits = MeshingLimits(
        maximum_work_units=1234,
        maximum_scratch_bytes=16 * 1024,
    )
    with native_meshing_request(limits) as (scope, budget, workspace):
        assert scope.limits.maximum_work_units == limits.maximum_work_units
        assert scope.limits.maximum_managed_bytes == limits.maximum_scratch_bytes
        assert budget.maximum_work_units == limits.maximum_work_units
        assert budget.maximum_memory_bytes == limits.maximum_scratch_bytes
        assert budget.ticket.settled
        assert budget.storage_ticket is not None
        assert workspace.bound == 0
        # One degree-zero two-dimensional polynomial needs fewer than these
        # declared term/bit preparation bounds; they fit the request's limits.
        with budget.bind_ticket(scope.batch(128, bytes_upper=4096)):
            assert constant(Fraction(1, 3), 2, budget=budget) == {(0, 0): Fraction(1, 3)}
        storage_ticket = budget.storage_ticket
        assert storage_ticket is not None
        assert not storage_ticket.settled
    assert storage_ticket.settled
    assert scope.evidence is not None
    assert scope.evidence.work_counters.construction_work == budget.work_units
    assert scope.evidence.externally_charged_work == budget.work_units
    assert scope.evidence.host_storage_live_bytes_upper == 0
