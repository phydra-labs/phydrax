#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


PIC = phx.discretization.pic


def _plan(**limits: int) -> PIC.PICCellBinningPlan:
    return PIC.PICCellBinningPlan(
        (0.0, -1.0), (1.0, 1.0), (4, 5), (True, False), **limits
    )


def _population(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    position = np.column_stack(
        (generator.uniform(-1.0, 2.0, 30), generator.uniform(-1.2, 1.2, 30))
    )
    active = generator.uniform(size=30) > 0.2
    identity_hi = generator.integers(0, 3, 30).astype(np.uint32)
    identity_lo = generator.permutation(30).astype(np.uint32)
    return position, active, identity_hi, identity_lo


def _members(bins: PIC.PICCellBins, hi: np.ndarray, lo: np.ndarray) -> list:
    order = np.asarray(bins.order)
    groups = []
    for start, count, key in zip(
        np.asarray(bins.cell_starts)[np.asarray(bins.occupied)],
        np.asarray(bins.cell_counts)[np.asarray(bins.occupied)],
        np.asarray(bins.cell_keys)[np.asarray(bins.occupied)],
        strict=True,
    ):
        slots = order[start : start + count]
        groups.append((int(key), [(int(hi[s]), int(lo[s])) for s in slots]))
    return groups


def test_cell_binning_is_invariant_to_particle_slot_order() -> None:
    position, active, hi, lo = _population(0)
    plan = _plan()
    bins = plan.bin(position, active, identity=(hi, lo))
    permutation = np.random.default_rng(1).permutation(30)
    permuted = plan.bin(
        position[permutation],
        active[permutation],
        identity=(hi[permutation], lo[permutation]),
    )
    ordered = _members(bins, hi, lo)
    assert ordered == _members(permuted, hi[permutation], lo[permutation])
    # Members of each cell appear in ascending global identity.
    assert all(members == sorted(members) for _, members in ordered)


def test_cell_binning_counts_match_an_independent_histogram() -> None:
    position, active, hi, lo = _population(2)
    bins = _plan().bin(position, active, identity=(hi, lo))
    inside = (position[:, 1] >= -1.0) & (position[:, 1] <= 1.0)
    column = np.floor(np.mod(position[:, 0], 1.0) * 4).astype(int)
    row = np.minimum(np.floor((position[:, 1] + 1.0) * 2.5).astype(int), 4)
    expected = np.bincount((column * 5 + row)[active & inside], minlength=20)
    counts = np.zeros(20, dtype=int)
    occupied = np.asarray(bins.occupied)
    counts[np.asarray(bins.cell_keys)[occupied]] = np.asarray(bins.cell_counts)[occupied]
    np.testing.assert_array_equal(counts, expected)
    assert int(bins.outside) == int(np.sum(active & ~inside))
    np.testing.assert_array_equal(
        np.asarray(bins.cell), np.where(active & inside, column * 5 + row, -1)
    )
    assert bool(bins.successful)
    assert int(bins.identity_ties) == 0


def test_cell_binning_reports_per_cell_capacity_overflow() -> None:
    position = jnp.asarray([[0.1, 0.1]] * 4 + [[0.6, 0.5]])
    active = jnp.ones((5,), dtype=jnp.bool_)
    bins = _plan(maximum_particles_per_cell=3).bin(position, active)
    assert bool(bins.overflow)
    assert not bool(bins.successful)
    assert int(bins.maximum_cell_count) == 4
    assert bool(_plan(maximum_particles_per_cell=4).bin(position, active).successful)
