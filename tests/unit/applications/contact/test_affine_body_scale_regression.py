# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.contact._affine_body import (
    AffineBodyHistory,
    AffineBodyParticipant,
    AffineBodyPlan,
    prepare_affine_body_participant,
)
from phydrax.discretization.contact._surface import CollisionSurfacePlan


def _scaled_body(scale: float) -> AffineBodyParticipant:
    reference = (
        np.concatenate(
            (np.zeros((1, 3), dtype=np.float64), np.eye(3, dtype=np.float64)),
            axis=0,
        )
        * scale
    )
    topology = CollisionSurfacePlan(
        np.arange(4, dtype=np.int64),
        ambient_dimension=3,
        faces=np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32),
        body_ids=0,
        participant_ids=0,
    )
    high = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
    low = (5.0 - np.sqrt(5.0)) / 20.0
    shapes = np.full((4, 4), low, dtype=np.float64)
    np.fill_diagonal(shapes, high)
    return prepare_affine_body_participant(
        AffineBodyPlan(
            topology,
            reference[None, :, :],
            reference,
            np.zeros(4, dtype=np.int32),
            shapes[None, :, :],
            np.full((1, 4), scale**3 / 24.0, dtype=np.float64),
            np.array([2.0], dtype=np.float64),
            np.array([30.0], dtype=np.float64),
            gravity=(0.0, 0.0, 0.0),
        )
    )


@pytest.mark.parametrize(
    "scale", [1.0e-5, 1.0, 1.0e5], ids=["ten-micrometers", "unit", "large"]
)
def test_undeformed_affine_body_is_admissible_at_every_scale(scale: float) -> None:
    participant = _scaled_body(scale)
    zero = jnp.zeros_like(participant.plan.reference_controls)
    time = jnp.zeros((), dtype=jnp.float64)
    history = AffineBodyHistory(zero, zero)
    evaluation = participant.evaluate(zero, history, zero, time, None, None)
    assert bool(evaluation.admissible)
    np.testing.assert_allclose(evaluation.residual, 0.0, atol=0.0)
    limit = eqx.filter_jit(participant.step_limit)(zero, zero, zero, time, None, None)
    assert bool(limit.successful)
    assert bool(limit.guarantee.successful)
    assert float(limit.maximum_step) == 1.0


@pytest.mark.parametrize(
    "matrix, crossing",
    [(np.diag([0.5, 0.5, 0.5]), False), (np.diag([-1.0, -1.0, 1.0]), True)],
    ids=["positive-contraction", "interior-collapse"],
)
def test_affine_inversion_step_is_scale_consistent(
    matrix: np.ndarray, crossing: bool
) -> None:
    bounds: list[float] = []
    for scale in (1.0e-5, 1.0, 1.0e5):
        participant = _scaled_body(scale)
        reference = participant.plan.reference_controls
        zero = jnp.zeros_like(reference)
        direction = reference @ jnp.asarray(matrix, dtype=jnp.float64).T - reference
        limit = eqx.filter_jit(participant.step_limit)(
            zero,
            direction,
            zero,
            jnp.zeros((), dtype=jnp.float64),
            None,
            None,
        )
        assert bool(limit.successful)
        assert bool(limit.conservative)
        assert bool(limit.guarantee.successful)
        bound = float(limit.maximum_step)
        if crossing:
            assert 0.0 < bound < 0.5
        else:
            assert bound == 1.0
        bounds.append(bound)
    np.testing.assert_allclose(bounds, bounds[1], rtol=0.0, atol=1.0e-9)
