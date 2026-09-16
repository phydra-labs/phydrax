#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _mechanism():
    schema = phx.equations.ChemicalSpeciesSchema.from_unique_species(
        ("A", "B"),
        (phx.equations.ChemicalPhaseKind.GAS,) * 2,
        jnp.asarray((0.01, 0.01)),
        ("E",),
        jnp.asarray(((1, 1),), dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
    )
    species = phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray((20.0, 20.0)),
        jnp.asarray((0.0, -1.0e4)),
        reference_temperature=300.0,
        minimum_temperature=200.0,
        maximum_temperature=3000.0,
    )
    mechanism = phx.equations.ChemicalMechanismIR(
        "A-to-B",
        schema,
        species,
        (
            phx.equations.ChemicalReactionSpec(
                "A->B",
                {"A": 1.0},
                {"B": 1.0},
                phx.equations.ArrheniusRatePlan(0.1),
            ),
        ),
    ).prepare()
    return species, mechanism


def test_energy_deposition_has_exact_spatial_power_and_cumulative_work():
    plan = phx.applications.reacting_flow.EnergyDepositionSourcePlan(
        jnp.asarray((1.0, 2.0)),
        jnp.asarray((2.0, 1.0)),
        total_energy=12.0,
        start_time=1.0,
        end_time=3.0,
        source_id="ignition",
    )
    midpoint = plan.evaluate(2.0)
    end = plan.evaluate(3.0)

    np.testing.assert_allclose(
        jnp.sum(plan.cell_volumes * midpoint.enthalpy_density_rate),
        midpoint.instantaneous_power,
    )
    np.testing.assert_allclose(midpoint.cumulative_work, 6.0)
    np.testing.assert_allclose(end.cumulative_work, 12.0)
    assert bool(midpoint.successful)


def test_fixed_connectivity_ale_remap_preserves_species_and_enthalpy_extensives():
    plan = phx.applications.reacting_flow.FixedConnectivityReactingALERemapPlan(
        jnp.asarray((1.0, 2.0)), jnp.asarray((2.0, 1.0))
    )
    result = plan.remap(
        jnp.asarray(((1.0, 2.0), (3.0, 4.0))),
        jnp.asarray((5.0, 6.0)),
    )

    assert bool(result.evidence.successful)
    np.testing.assert_allclose(result.evidence.species_extensive_defect, 0.0)
    np.testing.assert_allclose(result.evidence.enthalpy_extensive_defect, 0.0)


def test_measured_chemistry_work_schedule_is_atomic_and_deterministic():
    plan = phx.applications.reacting_flow.ChemistryWorkSchedulePlan(4, 2)
    accepted = plan.initialize()
    candidate = plan.propose(accepted, jnp.asarray((10.0, 1.0, 8.0, 1.0)))

    assert bool(candidate.successful)
    assert set(np.asarray(candidate.proposed_state.worker_assignment)) == {0, 1}
    rejected = plan.commit(accepted, candidate, False)
    np.testing.assert_array_equal(rejected.worker_assignment, accepted.worker_assignment)
    committed = plan.commit(accepted, candidate, True)
    assert int(committed.accepted_epoch) == 1


def test_amr_specialist_updates_active_cells_and_preserves_enthalpy():
    species, mechanism = _mechanism()
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray(((0.0,), (1.0,))))
    hierarchy_plan = phx.discretization.BlockHierarchyPlan(
        grid,
        (phx.discretization.BlockLevelPlan(0, (4,), 3),),
    )
    prepared = phx.discretization.FDAMRHierarchyPlan(hierarchy_plan).prepare()
    topology = prepared.initial_topology()
    amount = jnp.asarray((80.0, 20.0))
    initial_species = amount * mechanism.schema.molar_masses
    thermo = species.evaluate(800.0)
    enthalpy = jnp.sum(amount * thermo.molar_enthalpy)
    cell = jnp.concatenate((initial_species, jnp.asarray((enthalpy,))))
    values = jnp.full((3, 4, 3), jnp.nan)
    values = values.at[:2].set(jnp.broadcast_to(cell, (2, 4, 3)))
    level = phx.discretization.BlockLevelState(
        hierarchy_plan.levels[0], topology.levels[0], values
    )
    hierarchy = phx.discretization.BlockHierarchyState(topology, (level,))
    plan = phx.applications.reacting_flow.ReactingAMRSynchronizationPlan(
        mechanism,
        thermodynamic_pressure=1.0e5,
        correction_sweeps=3,
        tolerance=1.0e-6,
    )

    updated, accepted, evidence = plan.synchronize(0, hierarchy, 0.0, 0.01)

    assert bool(accepted)
    assert bool(evidence.successful)
    assert jnp.mean(updated.levels[0].values[:2, ..., 1]) > initial_species[1]
    np.testing.assert_array_equal(updated.levels[0].values[:2, ..., 2], enthalpy)
