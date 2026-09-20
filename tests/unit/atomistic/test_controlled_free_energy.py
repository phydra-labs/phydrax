import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.atomistic._alchemical import (
    AlchemicalControlKind,
    AlchemicalControlSchedulePlan,
    AlchemicalInteractionPartitionPlan,
    ControlledHamiltonianPlan,
)
from phydrax.atomistic._classical import HarmonicBondPotential, LennardJonesPotential
from phydrax.atomistic._dynamics import AtomisticDynamicsPlan, VelocityVerletPlan
from phydrax.atomistic._electrostatics import (
    DirectCoulombPotential,
    EwaldReferencePotential,
)
from phydrax.atomistic._force_field import (
    AtomisticForceFieldPlan,
    AtomisticForceFieldProvenance,
    AtomisticNonbondedPolicy,
)
from phydrax.atomistic._potential_program import AtomisticPotentialProgram
from phydrax.atomistic._stress import atomistic_cell_energy_and_stress
from phydrax.atomistic._system import AtomisticSystemPlan
from phydrax.atomistic._thermodynamic import (
    AtomisticPhaseSpaceMeasurePlan,
    AtomisticThermodynamicStatePlan,
    PreparedThermodynamicStateTable,
)
from phydrax.atomistic._topology import MolecularTopologyPlan
from phydrax.atomistic._units import AtomisticUnitSystem
from phydrax.atomistic.free_energy import (
    AbsoluteBindingPlan,
    AlchemicalSwitchingLineage,
    AlchemicalSwitchingPlan,
    FreeEnergyProtocolLegPlan,
    FreeEnergyStatePlan,
    MappedRelativeBindingPlan,
    MappedRelativeSolvationPlan,
    MappedRelativeTransformationPlan,
    NeutralAbsoluteSolvationPlan,
    RestraintCorrectionPlan,
    StandardStateCorrectionPlan,
    SymmetryCorrectionPlan,
)
from phydrax.atomistic.sampling import AtomisticCanonicalSamplingQualification
from phydrax.discretization import DenseParticleNeighborhoodPlan, PeriodicCell
from phydrax.uq import FreeEnergyResult, ReducedWorkDataset


def _force_field():
    units = AtomisticUnitSystem.reduced()
    system = AtomisticSystemPlan(
        [10, 20, 30],
        [1, 1, 1],
        [1.0, 1.0, 1.0],
        units,
        atom_type_ids=[0, 0, 1],
        charges=[1.0, -1.0, 0.5],
        region_ids=[1, 1, 0],
    )
    potential = AtomisticPotentialProgram(
        [
            LennardJonesPotential([0.4, 0.8], [1.0, 1.2], 3.0),
            DirectCoulombPotential(),
        ]
    )
    force_field = AtomisticForceFieldPlan(
        system,
        potential,
        AtomisticNonbondedPolicy(3.0, electrostatics="direct"),
        AtomisticForceFieldProvenance(
            "native", ("test-parameters",), "test", "controlled"
        ),
    ).prepare()
    neighborhood = DenseParticleNeighborhoodPlan(3).prepare(force_field.system.particles)
    return force_field, neighborhood


def _controlled():
    force_field, neighborhood = _force_field()
    schedule = AlchemicalControlSchedulePlan(
        ("coupled", "middle", "decoupled"),
        ("solute-sterics", "solute-electrostatics"),
        (AlchemicalControlKind.STERICS, AlchemicalControlKind.ELECTROSTATICS),
        jnp.asarray([[1.0, 1.0], [0.4, 0.6], [0.0, 0.0]]),
    )
    partition = AlchemicalInteractionPartitionPlan(
        schedule.control_ids,
        ([10, 20], [10, 20]),
        mapped_particle_ids=[[10, 10], [20, 20]],
    )
    return ControlledHamiltonianPlan(
        force_field, schedule, partition
    ).prepare(), neighborhood


def _runtime(controlled, neighborhood):
    dynamics = AtomisticDynamicsPlan(
        controlled.system,
        controlled,
        neighborhood,
        VelocityVerletPlan(1.0e-3),
    ).prepare()
    measure = AtomisticPhaseSpaceMeasurePlan(controlled.system)
    plans = tuple(
        AtomisticThermodynamicStatePlan(
            measure,
            ensemble="nvt",
            temperature=1.0,
            controls=controls,
            control_ids=controlled.control_ids,
            state_id=state_id,
        )
        for state_id, controls in zip(
            controlled.plan.schedule.state_ids,
            controlled.plan.schedule.controls,
            strict=True,
        )
    )
    return dynamics, PreparedThermodynamicStateTable(dynamics, plans)


def test_controlled_hamiltonian_has_exact_base_endpoint_and_control_derivatives():
    controlled, neighborhood = _controlled()
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [2.2, 0.2, 0.0]])
    relation = neighborhood.build(positions)
    base = controlled.potential.evaluate(positions, relation)
    coupled = controlled.evaluate(positions, relation, state_index=0)
    np.testing.assert_allclose(coupled.energy, base.energy, rtol=1.0e-12)
    np.testing.assert_allclose(coupled.forces, base.forces, rtol=1.0e-12)
    np.testing.assert_allclose(coupled.virial, base.virial, rtol=1.0e-12)
    assert bool(coupled.status.successful)

    controls = jnp.asarray([0.4, 0.6])
    center = controlled.evaluate(positions, relation, control_values=controls)
    step = 1.0e-5
    finite_difference = []
    for index in range(2):
        direction = jnp.zeros((2,)).at[index].set(step)
        plus = controlled.evaluate(
            positions, relation, control_values=controls + direction
        ).energy
        minus = controlled.evaluate(
            positions, relation, control_values=controls - direction
        ).energy
        finite_difference.append((plus - minus) / (2.0 * step))
    np.testing.assert_allclose(
        center.dU_dcontrols,
        jnp.stack(tuple(finite_difference)),
        rtol=5.0e-5,
        atol=5.0e-6,
    )


def test_soft_core_and_reduced_potential_evidence_fail_closed():
    controlled, neighborhood = _controlled()
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 0.0, 0.0]])
    relation = neighborhood.build(positions)
    decoupled = controlled.evaluate(positions, relation, state_index=2)
    assert bool(decoupled.successful)
    assert bool(jnp.all(jnp.isfinite(decoupled.forces)))
    invalid = controlled.evaluate(positions, relation, state_index=8)
    assert not bool(invalid.status.state_index_valid)
    assert bool(jnp.isnan(invalid.energy))

    first = positions.at[2, 0].set(2.0)
    second = positions.at[2, 0].set(2.2)
    _, thermodynamic = _runtime(controlled, neighborhood)
    cross = controlled.reduced_potentials(
        jnp.stack((first, second)),
        (neighborhood.build(first), neighborhood.build(second)),
        thermodynamic,
    )
    assert cross.values.shape == (3, 2)
    assert bool(jnp.all(cross.coverage))
    assert cross.state_ids == ("coupled", "middle", "decoupled")
    assert len(set(cross.potential_ids)) == 3
    assert cross.unit_system_id == controlled.system.plan.units.unit_system_id
    assert cross.thermodynamic_table_id == thermodynamic.table_id
    assert cross.control_ids == controlled.control_ids
    np.testing.assert_allclose(cross.controls, thermodynamic.controls)
    assert cross.bias_ids == thermodynamic.bias_ids
    negative_beta = eqx.tree_at(
        lambda value: value.beta,
        thermodynamic,
        thermodynamic.beta.at[0].set(-1.0),
    )
    with pytest.raises(ValueError, match="non-negative"):
        controlled.reduced_potentials(
            first[None],
            (neighborhood.build(first),),
            negative_beta,
            state_indices=[0],
        )


def test_preparation_rejects_charge_change_and_state_dependent_geometry():
    force_field, _ = _force_field()
    schedule = AlchemicalControlSchedulePlan(
        ("on", "off"),
        ("charged-region",),
        (AlchemicalControlKind.ELECTROSTATICS,),
        jnp.asarray([[1.0], [0.0]]),
    )
    charged = AlchemicalInteractionPartitionPlan(schedule.control_ids, ([10],))
    with pytest.raises(ValueError, match="zero net charge"):
        ControlledHamiltonianPlan(force_field, schedule, charged).prepare()
    changed_masses = AlchemicalInteractionPartitionPlan(
        schedule.control_ids, ([10, 20],), changes_masses=True
    )
    with pytest.raises(ValueError, match="State-dependent masses"):
        ControlledHamiltonianPlan(force_field, schedule, changed_masses).prepare()
    changed_constraints = AlchemicalInteractionPartitionPlan(
        schedule.control_ids, ([10, 20],), changes_constraints=True
    )
    with pytest.raises(ValueError, match="State-dependent constraints"):
        ControlledHamiltonianPlan(force_field, schedule, changed_constraints).prepare()
    changed_virtual_geometry = AlchemicalInteractionPartitionPlan(
        schedule.control_ids, ([10, 20],), changes_virtual_geometry=True
    )
    with pytest.raises(ValueError, match="State-dependent virtual-site geometry"):
        ControlledHamiltonianPlan(
            force_field, schedule, changed_virtual_geometry
        ).prepare()
    unknown_mapping = AlchemicalInteractionPartitionPlan(
        schedule.control_ids,
        ([10, 20],),
        mapped_particle_ids=[[10, 999]],
    )
    with pytest.raises(ValueError, match="mapping references"):
        ControlledHamiltonianPlan(force_field, schedule, unknown_mapping).prepare()


def test_disappearing_bond_has_defined_off_endpoint_at_collapsed_geometry():
    units = AtomisticUnitSystem.reduced()
    system = AtomisticSystemPlan(
        [10, 20],
        [1, 1],
        [1.0, 1.0],
        units,
        topology=MolecularTopologyPlan(bonds=[[10, 20]]),
    )
    force_field = AtomisticForceFieldPlan(
        system,
        AtomisticPotentialProgram([HarmonicBondPotential([3.0], [1.0])]),
        AtomisticNonbondedPolicy(3.0, electrostatics="direct"),
        AtomisticForceFieldProvenance(
            "native", ("bond-parameters",), "test", "controlled-bond"
        ),
    ).prepare()
    schedule = AlchemicalControlSchedulePlan(
        ("bonded", "unbonded"),
        ("bond-control",),
        (AlchemicalControlKind.BOND,),
        jnp.asarray([[1.0], [0.0]]),
    )
    controlled = ControlledHamiltonianPlan(
        force_field,
        schedule,
        AlchemicalInteractionPartitionPlan(schedule.control_ids, ([10],)),
    ).prepare()
    neighborhood = DenseParticleNeighborhoodPlan(1).prepare(force_field.system.particles)
    positions = jnp.zeros((2, 3))
    evaluation = controlled.evaluate(
        positions, neighborhood.build(positions), state_index=1
    )
    np.testing.assert_allclose(evaluation.energy, 0.0, atol=0.0)
    np.testing.assert_allclose(evaluation.forces, 0.0, atol=0.0)
    assert bool(jnp.all(jnp.isfinite(evaluation.dU_dcontrols)))
    assert bool(evaluation.successful)


def test_preparation_refuses_partial_reciprocal_electrostatic_control():
    units = AtomisticUnitSystem.reduced()
    cell = PeriodicCell(jnp.eye(3) * 10.0, periodic_axes=(True, True, True))
    system = AtomisticSystemPlan(
        [10, 20, 30],
        [1, 1, 1],
        [1.0, 1.0, 1.0],
        units,
        charges=[1.0, -1.0, 0.0],
        cell=cell,
    )
    force_field = AtomisticForceFieldPlan(
        system,
        AtomisticPotentialProgram([EwaldReferencePotential(0.3, 3.0, 1)]),
        AtomisticNonbondedPolicy(3.0, electrostatics="ewald"),
        AtomisticForceFieldProvenance(
            "native", ("ewald-parameters",), "test", "controlled-ewald"
        ),
    ).prepare()
    schedule = AlchemicalControlSchedulePlan(
        ("on", "off"),
        ("electrostatics",),
        (AlchemicalControlKind.ELECTROSTATICS,),
        jnp.asarray([[1.0], [0.0]]),
    )
    partition = AlchemicalInteractionPartitionPlan(schedule.control_ids, ([10, 20],))
    with pytest.raises(ValueError, match="cannot diverge"):
        ControlledHamiltonianPlan(force_field, schedule, partition).prepare()


def test_controlled_cell_stress_differentiates_the_same_scalar():
    units = AtomisticUnitSystem.reduced()
    cell = PeriodicCell(jnp.eye(3) * 10.0, periodic_axes=(True, True, True))
    system = AtomisticSystemPlan(
        [10, 20],
        [1, 1],
        [1.0, 1.0],
        units,
        atom_type_ids=[0, 0],
        cell=cell,
    )
    force_field = AtomisticForceFieldPlan(
        system,
        AtomisticPotentialProgram([LennardJonesPotential([0.5], [1.0], 3.0)]),
        AtomisticNonbondedPolicy(3.0, electrostatics="direct"),
        AtomisticForceFieldProvenance(
            "native", ("periodic-parameters",), "test", "controlled-cell"
        ),
    ).prepare()
    schedule = AlchemicalControlSchedulePlan(
        ("on", "off"),
        ("sterics",),
        (AlchemicalControlKind.STERICS,),
        jnp.asarray([[1.0], [0.0]]),
    )
    controlled = ControlledHamiltonianPlan(
        force_field,
        schedule,
        AlchemicalInteractionPartitionPlan(schedule.control_ids, ([10],)),
    ).prepare()
    fractional = jnp.asarray([[0.1, 0.1, 0.1], [0.25, 0.1, 0.1]])
    positions = fractional @ cell.vectors
    neighborhood = DenseParticleNeighborhoodPlan(1, box=cell).prepare(
        force_field.system.particles
    )
    result = atomistic_cell_energy_and_stress(
        controlled,
        fractional,
        neighborhood.build(positions),
        control_values=jnp.asarray([0.4]),
    )
    assert result.control_derivatives.shape == (1,)
    assert bool(jnp.all(jnp.isfinite(result.stress)))
    assert bool(result.successful)


def _leg(controlled, thermodynamic, environment):
    return FreeEnergyProtocolLegPlan(
        FreeEnergyStatePlan(controlled, thermodynamic, 0),
        FreeEnergyStatePlan(controlled, thermodynamic, 2),
        environment,
    )


def _analysis(leg, value, variance, identity, *, mapping_id=None):
    states = (leg.source, leg.destination)
    dataset = ReducedWorkDataset(
        jnp.zeros((2,)),
        jnp.ones((2,), dtype="bool"),
        jnp.ones((2,), dtype="bool"),
        jnp.asarray([0, 1]),
        jnp.asarray([1, 0]),
        jnp.asarray([0, 1]),
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.asarray([0, 1]),
        state_ids=tuple(state.state_id for state in states),
        potential_ids=tuple(state.potential_id for state in states),
        measure_ids=tuple(state.measure_id for state in states),
        producer_id="focused-protocol-test",
        run_id=f"run:{identity}",
        work_id=f"work:{identity}",
        work_kind="targeted-map" if mapping_id is not None else "equilibrium-difference",
        qualification_id=f"qualification:{identity}",
        sampling_exact=True,
        sampling_bias_bound=0.0,
        mapping_id=mapping_id,
        bias_ids=tuple(state.bias_id for state in states),
        unit_system_id=states[0].unit_system_id,
        unit_id="1",
    )
    covariance = jnp.asarray([[0.0, 0.0], [0.0, variance]])
    result = FreeEnergyResult(
        jnp.asarray([0.0, value]),
        covariance,
        jnp.ones((2, 2)),
        jnp.ones((2, 2), dtype="bool"),
        jnp.asarray([10.0, 10.0]),
        jnp.asarray([10.0, 10.0]),
        jnp.asarray([[0.0], [jnp.sqrt(variance)]]),
        1,
        0.0,
        1,
        0,
        0,
        state_ids=dataset.state_ids,
        gauge_state_id=dataset.state_ids[0],
        method="focused-authenticated-result",
        dataset_id=dataset.dataset_id,
        selection_id=f"selection:{identity}",
    )
    return result, dataset


def test_protocol_signs_and_covariance_are_explicit():
    controlled, neighborhood = _controlled()
    _, thermodynamic = _runtime(controlled, neighborhood)
    vacuum_plan = _leg(controlled, thermodynamic, "vacuum")
    solvent_plan = _leg(controlled, thermodynamic, "solvent")
    correction_plan = StandardStateCorrectionPlan(
        "one-molar", convention="add-to-destination-minus-source"
    )
    plan = NeutralAbsoluteSolvationPlan(
        vacuum_plan, solvent_plan, corrections=(correction_plan,)
    )
    vacuum_analysis = _analysis(vacuum_plan, 3.0, 0.04, "vacuum")
    solvent_analysis = _analysis(solvent_plan, 2.0, 0.09, "solvent")
    vacuum = vacuum_plan.project(*vacuum_analysis)
    solvent = solvent_plan.project(*solvent_analysis)
    correction = correction_plan.result(0.5, 0.01, evidence_id="analytic-volume")
    covariance = jnp.asarray([[0.04, 0.01, 0.0], [0.01, 0.09, 0.0], [0.0, 0.0, 0.01]])
    result = plan.evaluate(vacuum, solvent, (correction,), covariance)
    np.testing.assert_allclose(result.value, 1.5)
    np.testing.assert_allclose(result.variance, 0.12)
    np.testing.assert_array_equal(result.weights, [1.0, -1.0, 1.0])
    assert result.formula == "solvation = D_vac - D_solv + C"


def test_absolute_and_mapped_protocol_formulas_are_oriented():
    controlled, neighborhood = _controlled()
    _, thermodynamic = _runtime(controlled, neighborhood)
    solvent_plan = _leg(controlled, thermodynamic, "solvent")
    complex_plan = _leg(controlled, thermodynamic, "complex")
    restraint_plan = RestraintCorrectionPlan(
        "six-dof", convention="restrained-to-standard"
    )
    standard_plan = StandardStateCorrectionPlan(
        "one-molar", convention="restrained-to-standard"
    )
    symmetry_plan = SymmetryCorrectionPlan(2, convention="ligand-indistinguishability")
    binding_plan = AbsoluteBindingPlan(
        solvent_plan,
        complex_plan,
        (restraint_plan, standard_plan, symmetry_plan),
    )
    solvent = solvent_plan.project(*_analysis(solvent_plan, 5.0, 0.1, "binding-solvent"))
    complex_value = complex_plan.project(
        *_analysis(complex_plan, 8.0, 0.2, "binding-complex")
    )
    corrections = (
        restraint_plan.result(1.0, 0.01, evidence_id="restraint"),
        standard_plan.result(0.5, 0.0, evidence_id="standard"),
        symmetry_plan.analytic_result(evidence_id="symmetry"),
    )
    binding = binding_plan.evaluate(
        solvent,
        complex_value,
        corrections,
        jnp.diag(jnp.asarray([0.1, 0.2, 0.01, 0.0, 0.0])),
    )
    np.testing.assert_allclose(binding.value, -1.5 - np.log(2.0))

    vacuum_plan = _leg(controlled, thermodynamic, "vacuum")
    solvent_mapping = MappedRelativeTransformationPlan(solvent_plan, "mapping-A-to-B")
    vacuum_mapping = MappedRelativeTransformationPlan(vacuum_plan, "mapping-A-to-B")
    solvent_mapped = solvent_mapping.project(
        *_analysis(
            solvent_plan,
            2.5,
            0.04,
            "mapped-solvent",
            mapping_id="mapping-A-to-B",
        )
    )
    vacuum_mapped = vacuum_mapping.project(
        *_analysis(
            vacuum_plan,
            1.0,
            0.01,
            "mapped-vacuum",
            mapping_id="mapping-A-to-B",
        )
    )
    relative_solvation = MappedRelativeSolvationPlan(
        solvent_mapping, vacuum_mapping
    ).evaluate(
        solvent_mapped,
        vacuum_mapped,
        jnp.diag(jnp.asarray([0.04, 0.01])),
    )
    np.testing.assert_allclose(relative_solvation.value, 1.5)

    complex_mapping = MappedRelativeTransformationPlan(complex_plan, "mapping-A-to-B")
    complex_mapped = complex_mapping.project(
        *_analysis(
            complex_plan,
            4.0,
            0.09,
            "mapped-complex",
            mapping_id="mapping-A-to-B",
        )
    )
    relative_binding = MappedRelativeBindingPlan(
        complex_mapping, solvent_mapping
    ).evaluate(
        complex_mapped,
        solvent_mapped,
        jnp.diag(jnp.asarray([0.09, 0.04])),
    )
    np.testing.assert_allclose(relative_binding.value, 1.5)
    assert complex_mapped.mapping_plan_id == complex_mapping.plan_id


def test_switching_executes_native_dynamics_and_emits_lineage():
    controlled, neighborhood = _controlled()
    dynamics, thermodynamic = _runtime(controlled, neighborhood)
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [2.2, 0.2, 0.0]])
    zeros = jnp.zeros_like(positions)
    forward = tuple(
        dynamics.initialize_state(
            positions,
            thermodynamic,
            state_index=0,
            velocity=zeros,
            key=jax.random.key(index),
        )
        for index in range(2)
    )
    reverse = tuple(
        dynamics.initialize_state(
            positions,
            thermodynamic,
            state_index=2,
            velocity=zeros,
            key=jax.random.key(index + 2),
        )
        for index in range(2)
    )
    qualification = AtomisticCanonicalSamplingQualification(
        dynamics,
        thermodynamic,
        "equilibrated-endpoint-samples",
        sampling_exact=True,
        sampling_bias_bound=0.0,
    )
    switching = AlchemicalSwitchingPlan(
        dynamics,
        thermodynamic,
        qualification,
        0,
        2,
        2,
        2.0e-3,
        2,
    )
    forward_lineage = AlchemicalSwitchingLineage([0, 0], [10, 11], [0, 0], [0, 0], [7, 7])
    reverse_lineage = AlchemicalSwitchingLineage([2, 2], [12, 13], [0, 0], [0, 0], [7, 7])
    record = switching.execute(
        forward,
        reverse,
        forward_lineage,
        reverse_lineage,
        producer_id="switching-runtime",
        run_id="run-42",
    )
    assert record.source_state_id == "coupled"
    np.testing.assert_array_equal(record.forward_lineage.origin_ids, [0, 0])
    np.testing.assert_array_equal(record.reverse_lineage.origin_ids, [2, 2])
    assert record.destination_state_id == "decoupled"
    assert record.forward_orientation == "source-to-destination"
    assert record.reverse_orientation == "destination-to-source"
    assert record.forward_lineage.sample_count == 2
    assert len(record.forward_final_states) == 2
    assert record.qualification_id == qualification.qualification_id
    assert record.sampling_exact
    assert record.sampling_bias_bound == 0.0
    assert bool(record.successful)
