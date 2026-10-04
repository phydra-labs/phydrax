"""Fractional owner grids of oblique and partially periodic cells.

The alias oracle enumerates every (receiver, source, translation) pair on the
host and checks that each image within the radius of a receiver is selected for
that receiver's owner: missing an alias would silently drop an edge.
"""

from __future__ import annotations

import numpy as np
import pytest

from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization.particle._distributed import FractionalOwnerPartition


_CASES = (
    pytest.param(
        np.array([[2.0, 0.0, 0.0], [1.3, 1.7, 0.0], [0.6, -0.4, 2.1]]),
        (True, True, True),
        (2, 2, 1),
        id="triclinic-periodic",
    ),
    pytest.param(
        np.array([[2.4, 0.0, 0.0], [0.9, 2.2, 0.0], [0.0, 0.0, 3.0]]),
        (True, True, False),
        (1, 2, 2),
        id="slab-open-z",
    ),
)


@pytest.mark.parametrize(("vectors", "periodic", "counts"), _CASES)
def test_alias_mask_reaches_every_receiver_owner_within_radius(
    vectors: np.ndarray, periodic: tuple[bool, ...], counts: tuple[int, int, int]
) -> None:
    cell = PeriodicCell(vectors, periodic_axes=periodic)
    partition = FractionalOwnerPartition(cell, counts)
    radius = 2.7
    stencil = cell.image_stencil(radius, maximum_image_count=4096)
    rng = np.random.default_rng(1)
    fractional = rng.uniform(0.0, 1.0, (24, 3))
    open_axes = ~np.asarray(periodic)
    fractional[:, open_axes] = rng.uniform(-0.6, 1.6, (24, int(open_axes.sum())))
    wrapped = np.where(periodic, fractional - np.floor(fractional), fractional)
    owners = np.asarray(partition.owners(wrapped))
    assert owners.min() >= 0 and owners.max() < partition.owner_count
    inverse = np.linalg.pinv(vectors)
    reach = radius * np.sqrt(np.sum(inverse * inverse, axis=0))
    shifts = np.asarray(stencil.shifts)
    selected = np.asarray(partition.alias_mask(wrapped, shifts, reach))
    positions = wrapped @ vectors
    reached = 0
    for receiver in range(wrapped.shape[0]):
        for source in range(wrapped.shape[0]):
            images = positions[source] + shifts @ vectors
            near = np.linalg.norm(images - positions[receiver], axis=1) < radius
            reached += int(near.sum())
            assert np.all(selected[source, near, owners[receiver]])
    assert reached > 0
    # Owners far from an atom are not contacted for every translation.
    assert not np.all(selected)


def test_open_axis_owners_extend_to_infinity_and_refuse_bad_grids() -> None:
    cell = PeriodicCell(np.diag([2.0, 2.0, 3.0]), periodic_axes=(True, True, False))
    partition = FractionalOwnerPartition(cell, (1, 1, 3))
    owners = np.asarray(partition.owners(np.array([[0.5, 0.5, -4.0], [0.5, 0.5, 9.0]])))
    assert owners.tolist() == [0, 2]
    lower, upper = partition.region_bounds(np.float64)
    assert np.isneginf(np.asarray(lower)[0, 2]) and np.isposinf(np.asarray(upper)[2, 2])
    with pytest.raises(ValueError, match="one owner count per lattice axis"):
        FractionalOwnerPartition(cell, (2, 2))
    with pytest.raises(ValueError, match="positive"):
        FractionalOwnerPartition(cell, (0, 1, 1))
