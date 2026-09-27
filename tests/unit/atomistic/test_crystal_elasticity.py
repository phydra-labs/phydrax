import numpy as np

from phydrax.atomistic import (
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
from phydrax.discretization import DenseParticleNeighborhoodPlan, PeriodicCell
from phydrax.units import derived_unit


def test_crystal_elasticity_scenario_1() -> None:
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
    neighborhood = DenseParticleNeighborhoodPlan(1, box=cell).prepare(system.particles)
    potential = AtomisticPotentialProgram(
        # ty: ignore[invalid-argument-type]
        [EAMPotential([2.0, 1.0, 1.0, 0.5, 2.0], 1.5)]
    ).prepare(
        system,
        graph_execution=AtomisticGraphExecutionPlan(1, backend="particle"),
    )
    result = CrystalElasticityPlan(
        potential,
        neighborhood,
        # ty: ignore[invalid-argument-type]
        [[0.25, 0.5, 0.5], [0.5, 0.5, 0.5]],
    ).evaluate()

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
