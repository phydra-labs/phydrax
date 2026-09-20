import numpy as np

from phydrax.atomistic import AtomisticUnitSystem
from phydrax.chemistry.periodic._lattice_force_constants import (
    third_order_force_constant_unit,
    ThirdOrderForceConstants,
)
from phydrax.chemistry.periodic._lattice_transport import (
    IFC3ModeVertexPlan,
    ThreePhononRTAPlan,
)


def _ifc3(units, strength):
    triplets = np.asarray(
        [
            (first, second, third)
            for first in range(2)
            for second in range(2)
            for third in range(2)
        ],
        dtype="int64",
    )
    signs = np.asarray([1.0, -1.0])
    coefficients = (
        strength * signs[triplets[:, 0]] * signs[triplets[:, 1]] * signs[triplets[:, 2]]
    )
    values = coefficients[:, None, None, None] * np.ones((1, 3, 3, 3))
    translations = np.zeros((8, 2, 1), dtype="int64")
    return ThirdOrderForceConstants(
        triplets,
        translations,
        values,
        values,
        third_order_force_constant_unit(units.scale.energy_unit, units.scale.length_unit),
        atom_count=2,
        system_id="analytic-anharmonic-crystal",
        source_kind="provider-normalized-ifc3",
        source_id=f"analytic-provider-ifc3-{strength}",
    )


def _mode_vertices(ifc3, units):
    return IFC3ModeVertexPlan(
        ifc3,
        [[0]],
        (1,),
        [[0.0]],
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
        [np.eye(6)],
        [1.0, 1.0],
        units,
        phonon_result_id="analytic-phonons",
    ).evaluate()


def _rta(ifc3, units):
    return ThreePhononRTAPlan(
        ifc3,
        [[0]],
        (1,),
        [1.0],
        0.02,
        1.0,
        1.0,
        units,
        detailed_balance_tolerance=1.0e-10,
    )


def test_ifc3_to_three_phonon_rta_enforces_balance_and_ballistic_refusal():
    units = AtomisticUnitSystem.reduced()
    ifc3 = _ifc3(units, 0.01)
    vertices = _mode_vertices(ifc3, units)
    velocities = np.zeros((1, 6, 3))
    velocities[0, 0, 0] = 1.0
    heat = np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    result = _rta(ifc3, units).evaluate(vertices, velocities, heat)

    assert vertices.ifc3_id == ifc3.ifc_id
    assert bool(result.successful)
    assert float(result.scattering.momentum_residual) == 0.0
    assert float(result.scattering.detailed_balance_residual) < 1.0e-10
    assert np.isfinite(float(result.thermal_conductivity[0, 0]))

    zero_ifc3 = _ifc3(units, 0.0)
    ballistic = _rta(zero_ifc3, units).evaluate(
        _mode_vertices(zero_ifc3, units),
        velocities,
        np.ones((1, 6)),
    )
    assert not bool(ballistic.finite_conductivity)
    assert not bool(ballistic.successful)
    assert np.isnan(np.asarray(ballistic.thermal_conductivity)).all()
