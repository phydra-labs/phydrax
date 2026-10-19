#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source derivative bounds and scratch refusal at native surface boundaries."""

from __future__ import annotations

import numpy as np
import pytest

from phydrax._meshcore import NativeExecutionScope
from phydrax.meshing import MeshingFailure, MeshingFailureCategory, MeshingLimits
from phydrax.meshing.providers._native_publication import (
    coordinate_budget,
    coordinate_phase,
)
from tests._support.meshing_fixtures import surface_mesh, surface_specification
from tests._support.surface_generation_sources import accounting_spline_domain


@pytest.mark.parametrize("order", (1, 2), ids=("first-derivatives", "second-derivatives"))
def test_spline_derivative_bounds_use_bounded_source_request(order: int) -> None:
    domain = accounting_spline_domain()
    patch = domain.patches[0].surface
    limits = MeshingLimits(maximum_scratch_bytes=2_000_000)
    budget = coordinate_budget(limits)
    # The rational bilinear source is N(u,v)/W(u,v). At the chart center
    # derive its quotient derivatives directly from the authored control net,
    # independently of patch.evaluate and its autodiff program.
    weights = np.asarray(((1.0, 0.9), (1.1, 1.0)), dtype=np.float64)
    controls = np.asarray(
        (
            ((0.0, 0.0, 0.0), (0.0, 1.0, 0.05)),
            ((1.0, 0.0, 0.1), (1.0, 1.0, 0.2)),
        ),
        dtype=np.float64,
    )
    numerator = controls * weights[..., None]
    denominator = np.mean(weights)
    value = np.mean(numerator, axis=(0, 1)) / denominator
    weight_u = np.mean(weights[1] - weights[0])
    weight_v = np.mean(weights[:, 1] - weights[:, 0])
    first_u = (
        np.mean(numerator[1] - numerator[0], axis=0) - value * weight_u
    ) / denominator
    first_v = (
        np.mean(numerator[:, 1] - numerator[:, 0], axis=0) - value * weight_v
    ) / denominator
    mixed = (
        numerator[1, 1]
        - numerator[1, 0]
        - numerator[0, 1]
        + numerator[0, 0]
        - first_u * weight_v
        - first_v * weight_u
        - value * (weights[1, 1] - weights[1, 0] - weights[0, 1] + weights[0, 0])
    ) / denominator
    reference = (
        np.stack((first_u, first_v), axis=-1)
        if order == 1
        else np.stack(
            (
                np.stack((-2 * first_u * weight_u / denominator, mixed), axis=-1),
                np.stack((mixed, -2 * first_v * weight_v / denominator), axis=-1),
            ),
            axis=-1,
        )
    )
    with NativeExecutionScope(limits.native_limits()) as scope:
        with budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size)):
            before = scope.remaining().remaining_work_units
            with budget.temporary_scope(), coordinate_phase(scope, limits, budget):
                lower, upper = patch.derivative_bounds(
                    np.asarray(((0.0, 0.0), (1.0, 1.0)), dtype=np.float64),
                    order=order,
                    budget=budget,
                )
            assert budget.maximum_memory_bytes <= 2_000_000
            assert budget.temporary_bytes_upper == 0
            assert budget.work_units > 0
            assert budget.native_charged_work_units == budget.work_units
            assert before - scope.remaining().remaining_work_units == budget.work_units
            assert scope.remaining().remaining_work_units == (
                limits.maximum_work_units - budget.work_units
            )
    values = reference
    assert lower.shape == upper.shape == values.shape
    assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
    assert np.all(lower <= values) and np.all(values <= upper)
    assert budget.ticket.settled and budget.storage_ticket is None


def test_spline_source_failure_releases_scratch_and_settles_original_work() -> None:
    patch = accounting_spline_domain().patches[0].surface
    limits = MeshingLimits(maximum_scratch_bytes=2_000_000)
    box = np.asarray(((0.0, 0.0), (1.0, 1.0)), dtype=np.float64)
    budget = coordinate_budget(limits)
    with (
        pytest.raises(RuntimeError, match=r"spline_source\.failure"),
        NativeExecutionScope(limits.native_limits()) as scope,
        budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size)),
    ):
        before = scope.remaining().remaining_work_units
        with budget.temporary_scope(), coordinate_phase(scope, limits, budget):
            patch.derivative_bounds(box, order=1, budget=budget)
            patch.derivative_bounds(box, order=2, budget=budget)
            assert budget.work_units > 0
            budget.reserve(0, 64)
            assert budget.temporary_bytes_upper >= 64
            raise RuntimeError(
                "spline_source.failure must terminate the measured source phase; got an injected failure"
            )
    assert budget.temporary_bytes_upper == 0
    assert budget.ticket.settled and budget.storage_ticket is None
    assert budget.native_charged_work_units == budget.work_units
    assert scope.completed_evidence().work_counters.construction_work == budget.work_units
    # A new request must not inherit the failed phase's owner or reservations.
    successor = coordinate_budget(limits)
    with NativeExecutionScope(limits.native_limits()) as successor_scope:
        with successor.retained_storage(
            lambda size: successor_scope.batch(0, bytes_upper=size)
        ):
            with (
                successor.temporary_scope(),
                coordinate_phase(successor_scope, limits, successor),
            ):
                lower, upper = patch.derivative_bounds(box, order=2, budget=successor)
            assert before - successor_scope.remaining().remaining_work_units == (
                successor.work_units
            )
    assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
    assert successor.temporary_bytes_upper == 0
    assert successor.ticket.settled and successor.storage_ticket is None


def test_native_spline_surface_refuses_scratch_overflow_with_limit_evidence() -> None:
    domain = accounting_spline_domain()
    specification = surface_specification(
        domain,
        0.4,
        limits=MeshingLimits(maximum_scratch_bytes=2_000_000),
    )
    with pytest.raises(MeshingFailure) as refused:
        surface_mesh(domain, specification)
    assert refused.value.category in (
        MeshingFailureCategory.RESOURCE_EXHAUSTED,
        MeshingFailureCategory.AUDIT_FAILED,
    )
    requested = dict(refused.value.evidence.requested)
    storage_limits = [
        value
        for name, value in requested.items()
        if name == "maximum_scratch_bytes"
        or name.endswith("resource:polynomial_storage:limit")
    ]
    assert storage_limits and max(storage_limits) <= 2_000_000
