#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Optional reference admission distinguishes algebraic support from resource refusal."""

import numpy as np
import pytest

from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._cell_geometry import (
    BarycentricCellGeometryElement,
    coordinate_lagrange_element,
)
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.discretization._coordinate_enclosure import (
    coordinate_partition_unity_reference_chain,
    try_partition_unity_reference_chain,
)


@pytest.mark.parametrize("unity", (True, False), ids=("unity", "nonunit"))
def test_optional_reference_admission_preserves_strict_charge_order(unity: bool) -> None:
    root = coordinate_lagrange_element("triangle", 1)
    weights = np.eye(3, dtype=np.float64)
    if not unity:
        weights[0, 0] = 2.0
    element = BarycentricCellGeometryElement(root, weights)
    records: list[tuple[int, int]] = []
    for optional in (False, True):
        ticket = HostBatchTicket(100_000, bytes_upper=10_000_000)
        ledger = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=100_000,
            maximum_memory_bytes=10_000_000,
        )
        with ticket, ledger.activate():
            if optional:
                result = try_partition_unity_reference_chain(element, budget=ledger)
                assert (result is not None) == unity
                if result is not None:
                    assert result[0].element_id == root.element_id
            elif unity:
                result = coordinate_partition_unity_reference_chain(
                    element, budget=ledger
                )
                assert result[0].element_id == root.element_id
            else:
                with pytest.raises(
                    ValueError, match=r"coordinate_enclosure\.reference_action"
                ):
                    coordinate_partition_unity_reference_chain(element, budget=ledger)
        records.append((ledger.work_units, ledger.peak_bytes_upper))
    assert records[0] == records[1]


def test_optional_reference_admission_propagates_coordinate_resource_refusal() -> None:
    root = coordinate_lagrange_element("triangle", 1)
    element = BarycentricCellGeometryElement(root, np.eye(3, dtype=np.float64))
    ticket = HostBatchTicket(0, bytes_upper=10_000_000)
    ledger = CoordinateEnclosureBudget(
        ticket,
        maximum_work_units=0,
        maximum_memory_bytes=10_000_000,
    )
    with ticket, ledger.activate():
        with pytest.raises(CoordinateEnclosureResourceError, match="coefficient_work"):
            try_partition_unity_reference_chain(element, budget=ledger)
