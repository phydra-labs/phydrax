import numpy as np

from phydrax.atomistic import AtomisticUnitSystem
from phydrax.chemistry.periodic._lattice_force_constants import (
    IFCConstraintPolicy,
    normalize_second_order_force_constants,
    PrimitiveSupercellImageMap,
    second_order_force_constant_unit,
)
from phydrax.discretization import PeriodicCell
from phydrax.sparse import EdgeRelation


def test_lattice_force_constants_scenario_1() -> None:
    units = AtomisticUnitSystem.reduced()
    cell = PeriodicCell(np.eye(3))
    # ty: ignore[invalid-argument-type]
    relation = EdgeRelation([0, 0, 0], [0, 0, 0], source_size=1, target_size=1)
    blocks = np.asarray([-np.eye(3), 2.0 * np.eye(3), -np.eye(3)])
    raw = blocks.copy()
    raw[1, 0, 0] += 1.0e-3
    artifact = normalize_second_order_force_constants(
        relation,
        # ty: ignore[invalid-argument-type]
        [[-1, 0, 0], [0, 0, 0], [1, 0, 0]],
        raw,
        # ty: ignore[invalid-argument-type]
        [[0.0, 0.0, 0.0]],
        cell,
        second_order_force_constant_unit(
            units.scale.energy_unit, units.scale.length_unit
        ),
        system_id="monatomic-chain",
        source_kind="analytic-provider",
        source_id="analytic-provider-ifc2",
        constraint_policy=IFCConstraintPolicy(maximum_relative_correction=0.01),
    )

    assert float(artifact.constraints.raw_acoustic_residual) > 0.0
    assert float(artifact.constraints.corrected_acoustic_residual) < 1.0e-12
    assert float(artifact.constraints.corrected_pair_residual) < 1.0e-12
    assert float(artifact.constraints.corrected_rotational_residual) < 1.0e-12
    assert bool(artifact.constraints.successful)
    np.testing.assert_array_equal(artifact.reverse_indices, [2, 1, 0])
    cell = PeriodicCell(np.eye(3))
    image_map = PrimitiveSupercellImageMap(
        # ty: ignore[invalid-argument-type]
        [10],
        # ty: ignore[invalid-argument-type]
        [100, 101, 102],
        # ty: ignore[invalid-argument-type]
        [0, 0, 0],
        # ty: ignore[invalid-argument-type]
        [[0, 0, 0], [-1, 0, 0], [1, 0, 0]],
        # ty: ignore[invalid-argument-type]
        [[0.0, 0.0, 0.0]],
        cell,
        primitive_system_id="primitive-chain",
    )
    # ty: ignore[invalid-argument-type]
    relation = EdgeRelation([0, 0, 0], [0, 0, 0], source_size=1, target_size=1)
    source, target = image_map.route_supercell_indices(
        relation,
        # ty: ignore[invalid-argument-type]
        [[-1, 0, 0], [0, 0, 0], [1, 0, 0]],
    )
    np.testing.assert_array_equal(source, [1, 0, 2])
    np.testing.assert_array_equal(target, [0, 0, 0])
