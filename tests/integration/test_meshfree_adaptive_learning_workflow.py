# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Adaptive bulk/surface refinement transactions and learned metric corrections.

The workflow is `examples/meshfree_adaptive_learning.py`. Errors are measured
against the analytic boundary-layer and zonal sphere solutions, independent of
the indicators that drive the refinement.
"""

from __future__ import annotations

import numpy as np
import pytest

from examples.meshfree_adaptive_learning import (
    BulkReport,
    LearnedReport,
    run_bulk,
    run_learned,
    run_surface,
    SurfaceReport,
)


@pytest.fixture(scope="module")
def bulk() -> BulkReport:
    return run_bulk(start=12, levels=2, uniform=(12, 17, 24, 34))


@pytest.fixture(scope="module")
def surface() -> SurfaceReport:
    return run_surface(start=192, levels=2, uniform=(192, 384, 768))


def test_bulk_boundary_layer_adapts_better_than_uniform_at_equal_points(
    bulk: BulkReport,
) -> None:
    published = bulk["bulk_epochs_published"]
    assert published[0], bulk["bulk_epoch_refusals"]
    points = bulk["bulk_adaptive_points"]
    errors = bulk["bulk_adaptive_max_errors"]
    assert len(points) == 1 + sum(published)
    assert all(
        later > earlier for earlier, later in zip(points, points[1:], strict=False)
    )
    assert errors[-1] < errors[0]
    assert bulk["bulk_adaptive_gain_at_equal_points"] > 1.0
    assert all(
        outcome == "admitted"
        for outcome in bulk["bulk_epoch_stability"][: sum(published)]
    )


def test_rejected_and_capacity_refused_transactions_publish_nothing(
    bulk: BulkReport,
) -> None:
    assert bulk["bulk_rejected_published"] is False
    assert bulk["bulk_rejected_source_unchanged"] is True
    assert bulk["bulk_rejected_refusals"] != ""
    assert bulk["bulk_capacity_status"] == "CAPACITY_REFUSED"


def test_surface_peak_adapts_better_than_uniform_at_equal_points(
    surface: SurfaceReport,
) -> None:
    assert all(status == "ADMITTED" for status in surface["surface_proposal_statuses"])
    errors = np.asarray(surface["surface_adaptive_relative_l2"])
    assert errors[-1] < errors[0]
    assert surface["surface_adaptive_gain_at_equal_points"] > 1.0


@pytest.fixture(scope="module")
def learned() -> LearnedReport:
    return run_learned()


def test_learned_correction_keeps_moments_and_refuses_positivity_conflict(
    learned: LearnedReport,
) -> None:
    assert learned["learned_signed_status"] == "POSITIVITY_CONFLICT"
    assert learned["learned_signed_moments_exact"] is True
    assert learned["learned_signed_sign_margin"] < 0.0
    assert learned["learned_signed_weights_published"] is False
    assert learned["learned_constrained_status"] == "ADMITTED"
    assert learned["learned_constrained_provider"] == "conic"
    assert learned["learned_constrained_max_moment_residual"] < 1e-9
    assert learned["learned_constrained_min_weight"] >= 1e-2 - 1e-12
    assert learned["learned_constrained_coercive"] is True


def test_learned_correction_trains_through_the_implicit_conservation_adjoint(
    learned: LearnedReport,
) -> None:
    assert learned["learned_gradient_fd_relative_error"] < 1e-5
    assert learned["learned_training_accepted_updates"] >= 1
    assert (
        learned["learned_training_loss_after"] < learned["learned_training_loss_before"]
    )
    assert learned["learned_training_status"] == "ADMITTED"
    assert learned["learned_training_max_moment_residual"] < 1e-9
    assert learned["learned_training_primal_successful"] is True
    assert learned["learned_training_adjoint_successful"] is True
    assert learned["learned_failed_case_loss_finite"] is False
    assert learned["learned_failed_step_rejected"] is True
