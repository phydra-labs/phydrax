import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.geometry.multiregion_surface import (
    BoundedFieldReconstruction,
    ConservativeFieldTransfer,
)


SOURCE_ACTIVE = np.asarray((True, True, True, True, False, True))
TARGET_ACTIVE = np.asarray((True, True, False, True))
# Source 0 splits between targets 0 and 1, source 3 is split three ways, and the
# other active sources move whole; inactive source 4 and target 2 are unused.
ROUTES = (
    np.asarray((0, 0, 1, 2, 3, 3, 3, 5)),
    np.asarray((0, 1, 1, 3, 0, 1, 3, 3)),
    np.asarray((0.25, 0.75, 1.0, 1.0, 0.2, 0.3, 0.5, 1.0)),
)


def test_conservative_transfer_preserves_totals_positivity_and_support() -> None:
    transfer = ConservativeFieldTransfer(
        *ROUTES, source_active=SOURCE_ACTIVE, target_active=TARGET_ACTIVE
    )
    content = jnp.asarray(
        ((1.0, 0.5), (2.0, 0.0), (0.5, 3.0), (4.0, 1.0), (99.0, 99.0), (0.25, 2.0))
    )
    moved = transfer.apply(content)
    expected = np.zeros((4, 2))
    for source, target, weight in zip(*ROUTES, strict=True):
        expected[target] += weight * np.asarray(content[source])
    np.testing.assert_allclose(np.asarray(moved), expected, rtol=1e-15)
    assert np.all(np.asarray(moved[2]) == 0.0)
    evidence = transfer.evidence(content, moved)
    assert bool(evidence.successful)
    assert bool(evidence.conservative) and bool(evidence.positivity_preserved)
    active_total = np.sum(np.asarray(content)[SOURCE_ACTIVE], axis=0)
    np.testing.assert_allclose(np.asarray(evidence.target_total), active_total)
    assert float(evidence.relative_defect) < 1e-15

    tampered = moved.at[1, 0].add(1e-3)
    assert not bool(transfer.evidence(content, tampered).conservative)


def test_conservative_transfer_refuses_nonconservative_or_unsupported_weights() -> None:
    sources, targets, weights = ROUTES
    with pytest.raises(ValueError, match="sum to one"):
        ConservativeFieldTransfer(
            sources,
            targets,
            weights * 1.01,
            source_active=SOURCE_ACTIVE,
            target_active=TARGET_ACTIVE,
        )
    negative = weights.copy()
    negative[0], negative[1] = -0.25, 1.25
    with pytest.raises(ValueError, match="nonnegative"):
        ConservativeFieldTransfer(
            sources, targets, negative, source_active=SOURCE_ACTIVE, target_active=TARGET_ACTIVE
        )
    unsupported = TARGET_ACTIVE.copy()
    unsupported[2] = True
    with pytest.raises(ValueError, match="receive positive"):
        ConservativeFieldTransfer(
            sources, targets, weights, source_active=SOURCE_ACTIVE, target_active=unsupported
        )


def test_bounded_reconstruction_stays_within_supporting_values() -> None:
    reconstruction = BoundedFieldReconstruction(
        *ROUTES, source_active=SOURCE_ACTIVE, target_active=TARGET_ACTIVE
    )
    thickness = jnp.asarray((1.0, 3.0, 2.0, 5.0, -7.0, 4.0))
    values = reconstruction.apply(thickness)
    # Target 1 averages sources 0, 1, 3 with weights 0.75, 1.0, 0.3.
    assert float(values[1]) == pytest.approx((0.75 * 1.0 + 3.0 + 0.3 * 5.0) / 2.05)
    evidence = reconstruction.evidence(thickness, values)
    assert bool(evidence.successful) and bool(evidence.bounded)
    uniform = reconstruction.apply(jnp.full((6,), 2.5))
    np.testing.assert_allclose(np.asarray(uniform)[TARGET_ACTIVE], 2.5, rtol=1e-15)
    overshoot = values.at[0].set(9.0)
    assert not bool(reconstruction.evidence(thickness, overshoot).bounded)
