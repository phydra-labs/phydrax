# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import pytest

from benchmarks.monolithic_contact import prepare_benchmark_case
from phydrax.applications.contact import (
    prepare_monolithic_contact_step,
    solve_monolithic_contact_step,
)


@pytest.mark.parametrize("bodies", (1, 2), ids=("one-body", "two-bodies"))
def test_friction_benchmark_configuration_accepts_physical_step(bodies: int) -> None:
    plan, previous = prepare_benchmark_case(bodies, 1)
    result = solve_monolithic_contact_step(
        prepare_monolithic_contact_step(plan, previous, 0.001)
    )
    assert bool(result.accepted)
    assert (
        result.diagnostics.final_residual_norm
        <= plan.policy.termination.residual_threshold(
            result.diagnostics.initial_residual_norm
        )
    )
    assert result.diagnostics.lag_residual <= plan.policy.lag_tolerance
    assert bool(result.safety.successful)
    assert bool(result.safety.conservative)
    assert result.contact.minimum_gap > 0.0
    assert result.commit().state_version == previous.state_version + 1
