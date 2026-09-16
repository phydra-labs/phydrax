#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np
import pytest

from phydrax.operators.quantum import FermionModeOrder
from phydrax.operators.quantum.lattice import (
    FixedBosonNumberBasis,
    FixedCardinalityFermionBasis,
    FixedSpinProjectionBasis,
    SectorBasisResourcePolicy,
)


def _resources(maximum_dimension=128):
    return SectorBasisResourcePolicy(
        maximum_dimension=maximum_dimension, maximum_table_bytes=64_000
    )


def _assert_rank_roundtrip(basis):
    coordinates = [
        np.asarray(basis.coordinate(index)) for index in range(basis.dimension)
    ]
    assert len({tuple(value) for value in coordinates}) == basis.dimension
    for index, coordinate in enumerate(coordinates):
        assert bool(basis.contains(coordinate))
        assert int(basis.rank(coordinate)) == index


def test_direct_fixed_charge_bases_rank_without_ambient_enumeration():
    fermions = FixedCardinalityFermionBasis(
        FermionModeOrder(("a", "b", "c", "d")), 2, resources=_resources()
    )
    spins = FixedSpinProjectionBasis(
        ("i", "j", "k"), (1, 2, 1), 0, resources=_resources()
    )
    bosons = FixedBosonNumberBasis(("x", "y", "z"), (3, 2, 4), 3, resources=_resources())
    assert fermions.dimension == 6
    assert spins.dimension == 4
    assert bosons.dimension == 6
    _assert_rank_roundtrip(fermions)
    _assert_rank_roundtrip(spins)
    _assert_rank_roundtrip(bosons)
    with pytest.raises(TypeError, match="integer dtype"):
        fermions.rank((0.0, 0.0, 1.0, 1.0))


def test_sector_construction_refuses_dimension_before_coordinate_storage():
    with pytest.raises(ValueError, match="maximum_dimension"):
        FixedCardinalityFermionBasis(
            FermionModeOrder(tuple(f"m{index}" for index in range(12))),
            6,
            resources=_resources(maximum_dimension=100),
        )
