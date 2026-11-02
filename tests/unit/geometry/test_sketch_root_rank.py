#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np
import pytest

from phydrax.geometry.design._sketch_certificate import RootDOFEvidence
from phydrax.linalg._policies import RankPolicy
from phydrax.linalg._rank import host_numerical_rank_data


@pytest.mark.parametrize(
    "jacobian",
    [
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        np.asarray([[1.0, 0.0]], dtype=np.float64),
    ],
)
def test_root_evidence_uses_shared_rank_policy(jacobian: np.ndarray) -> None:
    evidence = RootDOFEvidence(
        "test",
        np.zeros(jacobian.shape[1], dtype=np.float64),
        np.zeros(jacobian.shape[0], dtype=np.float64),
        jacobian,
        residual_tolerance=1e-9,
    )
    rank = host_numerical_rank_data(
        np.linalg.svd(jacobian, compute_uv=False),
        jacobian.shape[0],
        jacobian.shape[1],
        RankPolicy(),
    )
    assert evidence.jacobian_rank == int(rank.rank)
    assert evidence.rank_tolerance == float(rank.cutoff)
    assert evidence.degrees_of_freedom == jacobian.shape[1] - int(rank.rank)
    assert evidence.root_converged
    assert evidence.regular_root_indicated == (rank.rank == jacobian.shape[1])


def test_empty_jacobian_has_zero_rank() -> None:
    evidence = RootDOFEvidence(
        "empty",
        np.zeros(2, dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.empty((0, 2), dtype=np.float64),
        residual_tolerance=1e-9,
    )
    assert evidence.jacobian_rank == 0
    assert evidence.degrees_of_freedom == 2
    assert evidence.rank_tolerance == 0.0
    assert evidence.root_converged
    assert not evidence.regular_root_indicated


def test_nonfinite_jacobian_does_not_claim_regular_root() -> None:
    evidence = RootDOFEvidence(
        "nonfinite",
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        np.asarray([[np.nan]], dtype=np.float64),
        residual_tolerance=1e-9,
    )
    assert not evidence.numerically_finite
    assert not evidence.root_converged
    assert not evidence.regular_root_indicated
