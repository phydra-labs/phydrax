import numpy as np

from phydrax.atomistic import AtomisticUnitSystem
from phydrax.chemistry.periodic._lattice_dynamics import (
    HarmonicPhononPlan,
    NonanalyticPhononCorrection,
)
from phydrax.chemistry.periodic._lattice_force_constants import (
    IFCConstraintPolicy,
    normalize_second_order_force_constants,
    second_order_force_constant_unit,
)
from phydrax.discretization import PeriodicCell
from phydrax.sparse import EdgeRelation


def _ifc(relation, translations, values, positions, cell, units, *, rotation=True):
    return normalize_second_order_force_constants(
        relation,
        translations,
        values,
        positions,
        cell,
        second_order_force_constant_unit(
            units.scale.energy_unit, units.scale.length_unit
        ),
        system_id="analytic-crystal",
        source_kind="analytic-provider",
        source_id="analytic-ifc2",
        constraint_policy=IFCConstraintPolicy(
            enforce_rotational_sum_rule=rotation,
            maximum_relative_correction=1.0e-10,
        ),
    )


def test_monatomic_nearest_neighbor_dispersion_uses_canonical_family():
    units = AtomisticUnitSystem.reduced()
    cell = PeriodicCell(np.eye(3))
    relation = EdgeRelation([0, 0, 0], [0, 0, 0], source_size=1, target_size=1)
    values = np.asarray([-np.eye(3), 2.0 * np.eye(3), -np.eye(3)])
    ifc = _ifc(
        relation,
        [[-1, 0, 0], [0, 0, 0], [1, 0, 0]],
        values,
        [[0.0, 0.0, 0.0]],
        cell,
        units,
    )
    result = (
        HarmonicPhononPlan(ifc, [1.0], units)
        .prepare()
        .evaluate([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    )

    assert bool(result.successful)
    np.testing.assert_allclose(result.angular_frequencies[0], 0.0, atol=1.0e-7)
    np.testing.assert_allclose(result.angular_frequencies[1], np.sqrt(2.0), atol=1.0e-12)


def test_directional_3d_loto_splits_only_longitudinal_optical_mode():
    units = AtomisticUnitSystem.reduced()
    cell = PeriodicCell(2.0 * np.eye(3))
    relation = EdgeRelation([0, 0, 1, 1], [0, 1, 0, 1], source_size=2, target_size=2)
    identity = np.eye(3)
    values = np.asarray([identity, -identity, -identity, identity])
    ifc = _ifc(
        relation,
        np.zeros((4, 3), dtype="int64"),
        values,
        [[0, 0, 0], [0.5, 0, 0]],
        cell,
        units,
        rotation=False,
    )
    prepared = HarmonicPhononPlan(ifc, [1.0, 1.0], units).prepare()
    nonpolar = prepared.evaluate([[0.0, 0.0, 0.0]])
    polar = NonanalyticPhononCorrection(
        [np.eye(3), -np.eye(3)],
        2.0 * np.eye(3),
        units,
        cell_id=cell.cell_id,
    )
    corrected = prepared.evaluate(
        [[0.0, 0.0, 0.0]],
        nonanalytic=polar,
        gamma_directions=[[1.0, 0.0, 0.0]],
    )

    baseline = np.sort(np.asarray(nonpolar.angular_frequencies[0]))
    shifted = np.sort(np.asarray(corrected.angular_frequencies[0]))
    np.testing.assert_allclose(shifted[-3:-1], baseline[-3:-1], atol=1.0e-12)
    assert shifted[-1] > baseline[-1]
