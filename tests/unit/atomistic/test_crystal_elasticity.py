from typing import Any

import numpy as np

from phydrax.atomistic import (
    AtomisticCellListImageSearch,
    AtomisticGraphExecutionPlan,
    AtomisticPotentialProgram,
    AtomisticSystemPlan,
    AtomisticUnitSystem,
    EAMPotential,
)
from phydrax.atomistic._crystal_elasticity import (
    CrystalElasticityPlan,
    CrystalNVEEvidence,
)
from phydrax.discretization import PeriodicCell
from phydrax.units import derived_unit
from tests._support.atomistic_bundles import (
    prepare_program,
    role_capacity,
    search_capacity,
)


FRACTIONS = [[0.25, 0.5, 0.5], [0.5, 0.5, 0.5]]


def _resources(*, widths: tuple[int, ...] = (64,)) -> dict[str, Any]:
    """Explicit resources of the EAM pair and graph roles in a 4 Å cube."""
    return {
        "search": AtomisticCellListImageSearch(maximum_candidate_slots=4096),
        "skin": 0.4,
        "search_capacities": tuple(
            search_capacity(width, particles_per_cell=4, images=27) for width in widths
        ),
        "role_capacities": (role_capacity(64, 32),),
    }


def _elasticity(**resources: Any) -> Any:
    units = AtomisticUnitSystem.reduced()
    cell = PeriodicCell(4.0 * np.eye(3))
    system = AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        [0, 1],
        # ty: ignore[invalid-argument-type]
        [1, 1],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0],
        units,
        # ty: ignore[invalid-argument-type]
        atom_type_ids=[0, 0],
        cell=cell,
    ).prepare()
    potential = prepare_program(
        # ty: ignore[invalid-argument-type]
        AtomisticPotentialProgram([EAMPotential([2.0, 1.0, 1.0, 0.5, 2.0], 1.5)]),
        system,
        graph_execution=AtomisticGraphExecutionPlan(1, backend="particle"),
        **resources,
    )
    # ty: ignore[invalid-argument-type]
    return CrystalElasticityPlan(potential, FRACTIONS).evaluate()


def test_strained_cells_own_independently_valid_recovered_resources() -> None:
    reference = _elasticity(**_resources())
    # A first search entry too small for every strained cell is replayed on
    # the declared ladder inside each independent preparation.
    recovered = _elasticity(**_resources(widths=(1, 64)))
    assert bool(reference.successful) and bool(recovered.successful)
    np.testing.assert_allclose(recovered.energy, reference.energy, rtol=1e-12)
    np.testing.assert_allclose(
        recovered.voigt_stiffness, reference.voigt_stiffness, rtol=1e-9, atol=1e-12
    )
    exhausted = _elasticity(**_resources(widths=(1,)))
    assert not bool(exhausted.successful)


def test_crystal_elasticity_scenario_1() -> None:
    units = AtomisticUnitSystem.reduced()
    result = _elasticity(**_resources())

    assert bool(result.successful)
    np.testing.assert_allclose(result.stress, result.stress.T, atol=1.0e-12)
    np.testing.assert_allclose(
        result.elastic_tensor, result.elastic_tensor.transpose(2, 3, 0, 1), atol=1.0e-10
    )
    assert result.stress_unit.unit_id == units.pressure_unit.unit_id
    assert result.potential_kind_ids == ("eam",)
    units = AtomisticUnitSystem.reduced()
    momentum_unit = derived_unit(
        "reduced_mass*reduced_length/reduced_time",
        ((units.mass_unit, 1), (units.scale.length_unit, 1), (units.time_unit, -1)),
    )
    stable = CrystalNVEEvidence(
        # ty: ignore[invalid-argument-type]
        [0.0, 1.0, 2.0],
        # ty: ignore[invalid-argument-type]
        [2.0, 2.0 + 1.0e-8, 2.0],
        np.zeros((3, 3)),
        units.scale.energy_unit,
        momentum_unit,
        trajectory_id="nve-chain",
        maximum_relative_energy_drift=1.0e-6,
        maximum_momentum_drift=1.0e-8,
    )
    drifting = CrystalNVEEvidence(
        # ty: ignore[invalid-argument-type]
        [0.0, 1.0, 2.0],
        # ty: ignore[invalid-argument-type]
        [2.0, 2.1, 2.2],
        np.zeros((3, 3)),
        units.scale.energy_unit,
        momentum_unit,
        trajectory_id="drifting-chain",
        maximum_relative_energy_drift=1.0e-3,
        maximum_momentum_drift=1.0e-8,
    )
    assert bool(stable.successful)
    assert not bool(drifting.successful)
