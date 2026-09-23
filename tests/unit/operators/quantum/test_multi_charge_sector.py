import numpy as np

from phydrax.operators.quantum import AbelianGroup
from phydrax.operators.quantum.lattice import (
    FixedAbelianChargeBasis,
    SectorBasisResourcePolicy,
)


def test_fixed_abelian_charge_basis_ranks_particle_and_projection_sector():
    basis = FixedAbelianChargeBasis(
        ("m-3", "m-1", "m+1", "m+3"),
        (
            ((0, 0), (1, -3)),
            ((0, 0), (1, -1)),
            ((0, 0), (1, 1)),
            ((0, 0), (1, 3)),
        ),
        ("particle-number", "twice-projection"),
        (2, 0),
        AbelianGroup((None, None)),
        resources=SectorBasisResourcePolicy(
            maximum_dimension=32,
            maximum_table_bytes=1_000_000,
        ),
    )

    coordinates = np.stack(
        [np.asarray(basis.coordinate(index)) for index in range(basis.dimension)]
    )
    assert basis.dimension == 2
    np.testing.assert_array_equal(np.sum(coordinates, axis=1), 2)
    np.testing.assert_array_equal(coordinates @ np.asarray((-3, -1, 1, 3)), 0)
    for index, coordinate in enumerate(coordinates):
        assert int(basis.rank(coordinate)) == index
        assert bool(basis.contains(coordinate))
