#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Rank decisions of the method-of-snapshots physical POD against constructed spectra."""

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


SAMPLES = 12
SIZE = 40
SOURCES = ("snapshots",)


def _space() -> phx.linalg.ArraySpace:
    return phx.linalg.ArraySpace((SIZE,), dtype=jnp.float64, space_id="pod-space")


def _snapshots(singular_values: np.ndarray, seed: int) -> np.ndarray:
    """Uncentered snapshots whose uniformly weighted singular values are given."""
    rng = np.random.default_rng(seed)
    rank = singular_values.size
    samples, _ = np.linalg.qr(rng.standard_normal((SAMPLES, rank)))
    modes, _ = np.linalg.qr(rng.standard_normal((SIZE, rank)))
    return np.sqrt(SAMPLES) * (samples * singular_values) @ modes.T


def _fit(
    snapshots: np.ndarray, **options: float
) -> phx.ml.decomposition.PhysicalPODResult:
    return phx.ml.decomposition.PhysicalPODPlan(SAMPLES, centered=False, **options).fit(
        _space(), jnp.asarray(snapshots), source_artifact_ids=SOURCES
    )


@pytest.mark.parametrize("seed", range(6), ids=lambda seed: f"seed-{seed}")
def test_exact_low_rank_data_is_recovered_at_its_rank_under_ulp_perturbation(
    seed: int,
) -> None:
    # Rank-3 snapshots: every further Gram eigenvalue is roundoff (sigma about
    # sqrt(eps) sigma_1) and must neither enter the basis nor let a 1-ulp change
    # of the snapshots move the rank.
    rng = np.random.default_rng(100 + seed)
    exact = rng.standard_normal((SAMPLES, 3)) @ rng.standard_normal((3, SIZE))
    for snapshots in (exact, np.nextafter(exact, np.inf), np.nextafter(exact, -np.inf)):
        result = _fit(snapshots)
        assert result.achieved_rank == 3
        assert result.target_met
        basis = np.asarray(result.subspace.basis)
        assert float(result.orthogonality_defect) < 1e-12
        np.testing.assert_allclose(basis.T @ basis, np.eye(3), rtol=0.0, atol=1e-12)
        residual = snapshots - (snapshots @ basis) @ basis.T
        assert np.linalg.norm(residual) < 1e-13 * np.linalg.norm(snapshots)
        assert float(result.tail_energy) < 1e-14


def test_resolved_small_singular_values_are_kept_with_their_values() -> None:
    # sqrt((N + sqrt(m)) eps) is about 6e-8, so 1e-6 is resolved by the method of
    # snapshots (to that absolute accuracy relative to sigma_1).
    singular = np.asarray([2.0, 3e-2, 1e-6])
    result = _fit(_snapshots(singular, seed=7))
    assert result.achieved_rank == 3
    assert result.target_met
    np.testing.assert_allclose(
        np.asarray(result.singular_values)[:3],
        singular,
        rtol=0.0,
        atol=1e-7 * singular[0],
    )


def test_caller_minimum_singular_value_truncates_below_the_energy_rank() -> None:
    singular = np.asarray([2.0, 3e-2, 1e-4])
    result = _fit(_snapshots(singular, seed=8), minimum_singular_value=1e-3)
    assert result.achieved_rank == 2
    assert not result.target_met
    energies = singular**2 / np.sum(singular**2)
    # Energy fractions are accurate to the Gram floor (N + sqrt(m)) eps, absolute.
    tolerance = (SAMPLES + np.sqrt(SIZE)) * np.finfo(np.float64).eps
    np.testing.assert_allclose(
        float(result.tail_energy), energies[2], rtol=0.0, atol=tolerance
    )


@pytest.mark.parametrize(
    ("retained", "rank"),
    [(0.45, 1), (0.75, 2), (0.85, 3), (1.0, 3)],
    ids=["first", "second", "third", "all"],
)
def test_energy_rank_is_the_smallest_rank_meeting_the_request(
    retained: float, rank: int
) -> None:
    # Energy fractions 0.5, 0.3, 0.2.
    singular = np.sqrt(np.asarray([0.5, 0.3, 0.2]))
    result = _fit(_snapshots(singular, seed=9), retained_energy=retained)
    assert result.achieved_rank == rank
    assert result.target_met
    assert float(result.retained_energy) >= retained - 1e-12
