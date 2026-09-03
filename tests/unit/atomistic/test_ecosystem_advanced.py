import os
import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _runtime(*, units=None, topology=None, charges=None, cell=None):
    units = phx.atomistic.AtomisticUnitSystem.reduced() if units is None else units
    system_plan = phx.atomistic.AtomisticSystemPlan(
        [10, 20, 30],
        [1, 1, 1],
        [1.0, 1.0, 1.0],
        units,
        atom_type_ids=[0, 0, 0],
        charges=[0.0, 0.0, 0.0] if charges is None else charges,
        topology=topology,
        cell=cell,
    )
    system = system_plan.prepare()
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(3).prepare(
        system.particles
    )
    potential = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.LennardJonesPotential([0.2], [1.0], 2.5)]
    ).prepare(system)
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        potential,
        neighborhood,
        phx.atomistic.VelocityVerletPlan(1.0e-3),
    ).prepare()
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.2, 0.0]])
    state = dynamics.initialize_state(
        positions, velocity=jnp.zeros_like(positions), key=jax.random.key(0)
    )
    return system_plan, system, neighborhood, potential, dynamics, state


def test_force_field_mapping_roundtrip_preserves_energy():
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    topology = phx.atomistic.MolecularTopologyPlan(
        bonds=[[10, 20]],
        bond_type_ids=[0],
        pair_exceptions=[[10, 20]],
        lennard_jones_scales=[0.5],
        electrostatic_scales=[0.25],
    )
    system_plan = phx.atomistic.AtomisticSystemPlan(
        [10, 20, 30],
        [1, 1, 1],
        [1.0, 1.0, 1.0],
        units,
        atom_type_ids=[0, 0, 0],
        charges=[0.4, -0.2, -0.2],
        topology=topology,
        name="mapping-roundtrip",
    )
    potential_plan = phx.atomistic.AtomisticPotentialProgram(
        [
            phx.atomistic.HarmonicBondPotential([2.0], [1.0]),
            phx.atomistic.LennardJonesPotential([0.2], [1.0], 2.5),
            phx.atomistic.DirectCoulombPotential(),
        ],
        coefficients=[1.2, 0.8, 1.0],
    )
    force_field = phx.atomistic.AtomisticForceFieldPlan(
        system_plan,
        potential_plan,
        phx.atomistic.AtomisticNonbondedPolicy(2.5, electrostatics="direct"),
        phx.atomistic.AtomisticForceFieldProvenance(
            "native", ("qualification",), "custom", "roundtrip"
        ),
    )
    bundle = phx.atomistic.interchange.AtomisticInterchangeBundle(
        force_field,
        phx.atomistic.interchange.AtomisticInterchangeReport(
            "native", tuple(term.name for term in potential_plan.terms)
        ),
    )
    restored = phx.atomistic.interchange.force_field_from_mapping(
        phx.atomistic.interchange.force_field_to_mapping(bundle), units
    )
    first = force_field.prepare()
    second = restored.force_field.prepare()
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.3, 0.0]])
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(3).prepare(
        first.system.particles
    )
    relation = neighborhood.build(positions)
    first_evaluation = first.potential.evaluate(positions, relation)
    second_evaluation = second.potential.evaluate(positions, relation)
    np.testing.assert_allclose(
        second_evaluation.energy, first_evaluation.energy, rtol=1.0e-12, atol=1.0e-12
    )
    np.testing.assert_allclose(
        second_evaluation.forces, first_evaluation.forces, rtol=1.0e-12, atol=1.0e-12
    )


def test_force_field_term_families_and_settle():
    cell = phx.discretization.PeriodicCell(jnp.eye(3) * 6.0)
    topology = phx.atomistic.MolecularTopologyPlan(
        torsions=[[0, 1, 2, 3]],
        impropers=[[0, 1, 2, 3]],
    )
    system = phx.atomistic.AtomisticSystemPlan(
        [0, 1, 2, 3],
        [8, 1, 1, 1],
        [16.0, 1.0, 1.0, 1.0],
        phx.atomistic.AtomisticUnitSystem.reduced(),
        atom_type_ids=[0, 0, 0, 0],
        charges=[-0.3, 0.1, 0.1, 0.1],
        topology=topology,
        cell=cell,
    ).prepare()
    positions = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.2, 1.0, 1.0],
        ]
    )
    route = [[0, 1, 2, 3]]
    terms = (
        phx.atomistic.HarmonicImproperPotential([1.0], [0.2]),
        phx.atomistic.UreyBradleyPotential([1.0], [1.4], [[0, 1, 2]]),
        phx.atomistic.PeriodicTorsionSeriesPotential(
            [[0.2, 0.1]], [[1, 2]], [[0.0, 0.3]], [[1.0, 1.0]], route
        ),
        phx.atomistic.RyckaertBellemansPotential([[0.1, 0.2, 0.1, 0.0, 0.0, 0.0]], route),
        phx.atomistic.CMAPPotential(
            jnp.arange(64, dtype=float).reshape((8, 8)) * 1.0e-3,
            [[0, 1, 2, 3, 0, 1, 2, 3]],
        ),
        phx.atomistic.PairOverrideLennardJonesPotential(
            jnp.ones((4, 4)) * 0.1,
            jnp.ones((4, 4)),
            jnp.ones((4, 4)),
            2.5,
        ),
        phx.atomistic.TabulatedPairPotential([0.5, 1.5, 2.5], [[[0.2, 0.1, 0.0]]], 2.5),
        phx.atomistic.ReactionFieldPotential(78.5, 2.5),
        phx.atomistic.LennardJonesDispersionCorrection(0.1),
    )
    potential = phx.atomistic.AtomisticPotentialProgram(terms).prepare(system)
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(6).prepare(
        system.particles
    )
    evaluation = potential.evaluate(positions, neighborhood.build(positions))
    assert bool(evaluation.successful)
    assert bool(jnp.isfinite(evaluation.energy))
    assert bool(jnp.all(jnp.isfinite(evaluation.forces)))

    settle_system = phx.atomistic.AtomisticSystemPlan(
        [0, 1, 2],
        [8, 1, 1],
        [16.0, 1.0, 1.0],
        phx.atomistic.AtomisticUnitSystem.reduced(),
    ).prepare()
    settle = phx.atomistic.SETTLEPlan([[0, 1, 2]], 1.0, 1.6).prepare(settle_system)
    projection = settle.project(
        [[0.0, 0.0, 0.0], [1.1, 0.1, 0.0], [-0.2, 0.9, 0.1]],
        [[0.0, 0.0, 0.0], [0.1, 0.2, 0.0], [-0.1, 0.0, 0.2]],
    )
    assert bool(projection.successful)
    assert float(projection.position_residual) <= 1.0e-10
    assert float(projection.velocity_residual) <= 1.0e-10


def test_interaction_site_pair_terms_honor_topology_scales():
    topology = phx.atomistic.MolecularTopologyPlan(
        pair_exceptions=[[0, 1]],
        lennard_jones_scales=[0.0],
        electrostatic_scales=[0.0],
    )
    system = phx.atomistic.AtomisticSystemPlan(
        [0, 1],
        [1, 1],
        [1.0, 1.0],
        phx.atomistic.AtomisticUnitSystem.reduced(),
        atom_type_ids=[0, 0],
        charges=[1.0, -1.0],
        topology=topology,
    ).prepare()
    potential = phx.atomistic.AtomisticPotentialProgram(
        [
            phx.atomistic.MorsePotential([[0.2]], [[2.0]], [[1.0]], 2.5),
            phx.atomistic.ReactionFieldPotential(78.5, 2.5),
        ]
    ).prepare(system)
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(2).prepare(
        system.particles
    )
    result = potential.evaluate(positions, neighborhood.build(positions))
    assert bool(result.successful)
    np.testing.assert_allclose(result.energy, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(result.forces, 0.0, atol=1.0e-12)


def test_openmm_import_export_energy_force_parity():
    openmm = pytest.importorskip("openmm")
    source = openmm.System()
    source.addParticle(12.0 * openmm.unit.dalton)
    source.addParticle(16.0 * openmm.unit.dalton)
    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    nonbonded.addParticle(
        0.1 * openmm.unit.elementary_charge,
        1.1 * openmm.unit.angstrom,
        0.3 * openmm.unit.kilojoule_per_mole,
    )
    nonbonded.addParticle(
        -0.1 * openmm.unit.elementary_charge,
        1.3 * openmm.unit.angstrom,
        0.2 * openmm.unit.kilojoule_per_mole,
    )
    bond = openmm.HarmonicBondForce()
    bond.addBond(
        0,
        1,
        1.0 * openmm.unit.angstrom,
        100.0 * openmm.unit.kilojoule_per_mole / openmm.unit.angstrom**2,
    )
    nonbonded.addException(
        0,
        1,
        -0.01 * openmm.unit.elementary_charge**2,
        1.2 * openmm.unit.angstrom,
        np.sqrt(0.3 * 0.2) * openmm.unit.kilojoule_per_mole,
    )
    source.addForce(nonbonded)
    source.addForce(bond)
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    bundle = phx.atomistic.interchange.from_openmm_system(
        source, units, atomic_numbers=[6, 8], cutoff=10.0
    )
    prepared = bundle.force_field.prepare()
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(2).prepare(
        prepared.system.particles
    )
    native = prepared.potential.evaluate(positions, neighborhood.build(positions))

    def openmm_evaluation(system):
        integrator = openmm.VerletIntegrator(0.001)
        context = openmm.Context(
            system, integrator, openmm.Platform.getPlatformByName("Reference")
        )
        context.setPositions(np.asarray(positions) * openmm.unit.angstrom)
        state = context.getState(getEnergy=True, getForces=True)
        energy = (
            state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
            / 96.48533212331002
        )
        forces = (
            np.asarray(
                state.getForces(asNumpy=True).value_in_unit(
                    openmm.unit.kilojoule_per_mole / openmm.unit.angstrom
                )
            )
            / 96.48533212331002
        )
        del context, integrator
        return energy, forces

    reference_energy, reference_forces = openmm_evaluation(source)
    np.testing.assert_allclose(
        native.energy, reference_energy, rtol=1.0e-10, atol=1.0e-10
    )
    np.testing.assert_allclose(
        native.forces, reference_forces, rtol=1.0e-10, atol=1.0e-10
    )
    exported, exported_topology, report = phx.atomistic.interchange.to_openmm_system(
        bundle
    )
    assert not report.unsupported_terms
    exported_energy, exported_forces = openmm_evaluation(exported)
    np.testing.assert_allclose(
        exported_energy, reference_energy, rtol=1.0e-10, atol=1.0e-10
    )
    np.testing.assert_allclose(
        exported_forces, reference_forces, rtol=1.0e-10, atol=1.0e-10
    )
    parmed = pytest.importorskip("parmed")
    with pytest.warns(parmed.exceptions.OpenMMWarning, match="incomplete exceptions"):
        structure = parmed.openmm.load_topology(exported_topology, exported)
    parmed_bundle = phx.atomistic.interchange.from_parmed_structure(
        structure, units, cutoff=10.0
    ).force_field.prepare()
    parmed_evaluation = parmed_bundle.potential.evaluate(
        positions,
        phx.discretization.DenseParticleNeighborhoodPlan(2)
        .prepare(parmed_bundle.system.particles)
        .build(positions),
    )
    np.testing.assert_allclose(
        parmed_evaluation.energy, reference_energy, rtol=1.0e-10, atol=1.0e-10
    )
    np.testing.assert_allclose(
        parmed_evaluation.forces, reference_forces, rtol=1.0e-10, atol=1.0e-10
    )


def test_h5md_extended_xyz_and_rerun_reporting(tmp_path: Path):
    _, _, neighborhood, potential, dynamics, state = _runtime()
    frame = phx.atomistic.AtomisticFrame(
        state.time,
        state.step_index,
        state.kinematics.positions,
        dynamics.system.plan.particle_ids,
        velocities=dynamics.velocity(state),
        momenta=state.kinematics.momenta,
        forces=state.force.forces,
        energy=[state.energy.total_energy],
        auxiliary={"temperature": jnp.asarray(1.0)},
        system_id=dynamics.system.plan.system_id,
        topology_id=dynamics.system.topology.topology_id,
        unit_system_id=dynamics.system.plan.units.unit_system_id,
        source_id="advanced-frame",
    )
    h5md = phx.atomistic.interchange.H5MDTrajectoryPlan(tmp_path / "input.h5")
    with h5md.open(append=False) as writer:
        writer.write(frame)
    with h5md.open(append=True) as writer:
        writer.write(
            phx.atomistic.AtomisticFrame(
                frame.time + 0.1,
                frame.step + 1,
                frame.positions,
                frame.stable_ids,
                velocities=frame.velocities,
                momenta=frame.momenta,
                forces=frame.forces,
                energy=frame.energy,
                auxiliary=frame.auxiliary,
                system_id=frame.system_id,
                topology_id=frame.topology_id,
                unit_system_id=frame.unit_system_id,
                source_id="advanced-frame-1",
            )
        )
    with h5md.open() as reader:
        recovered = tuple(reader)
    assert len(recovered) == 2
    np.testing.assert_allclose(recovered[1].velocities, frame.velocities)
    np.testing.assert_allclose(recovered[1].auxiliary["temperature"], 1.0)
    assert recovered[0].source_id == frame.source_id

    xyz = phx.atomistic.interchange.ExtendedXYZTrajectoryPlan(tmp_path / "output.xyz")
    with xyz.open(append=False) as writer:
        writer.write(frame)
    with xyz.open() as reader:
        xyz_frame = tuple(reader)[0]
    np.testing.assert_allclose(xyz_frame.forces, frame.forces)
    np.testing.assert_allclose(xyz_frame.momenta, frame.momenta)
    assert xyz_frame.source_id == frame.source_id

    rerun_output = phx.atomistic.interchange.H5MDTrajectoryPlan(tmp_path / "rerun.h5")
    fields = (
        phx.atomistic.AtomisticFrameFields.POSITIONS
        | phx.atomistic.AtomisticFrameFields.FORCES
        | phx.atomistic.AtomisticFrameFields.ENERGY
        | phx.atomistic.AtomisticFrameFields.AUXILIARY
    )
    reporter = phx.atomistic.AtomisticReporterPlan(rerun_output, fields=fields)
    result = phx.atomistic.AtomisticRerunPlan(
        h5md,
        potential,
        neighborhood,
        force_groups=(0,),
        lambda_values=(0.0, 1.0),
        reporter=reporter,
    ).run()
    assert bool(result.successful)
    assert result.reduction.frame_count == 2
    assert result.reduction.mean_energies.shape == (2,)
    assert result.reduction.mean_force_group_energies.shape == (2, 1)
    with rerun_output.open() as reader:
        reported = tuple(reader)
    assert len(reported) == 2
    assert reported[0].auxiliary["rerun_lambda_energies"].shape == (2,)
    assert reported[0].source_id.startswith(frame.source_id)
    assert reported[0].auxiliary["rerun_force_group_energies"].shape == (2, 1)


def test_bias_checkpoint_replay_and_abf_update(tmp_path: Path):
    _, system, _, _, dynamics, state = _runtime()
    cv = phx.atomistic.sampling.CollectiveVariablePlan(
        phx.atomistic.sampling.CollectiveVariableKind.DISTANCE, [0, 1]
    ).prepare(system)
    variables = phx.atomistic.sampling.CollectiveVariableProgram((cv,))
    bias = phx.atomistic.sampling.PreparedAtomisticBias(
        phx.atomistic.sampling.AtomisticBiasPlan(
            phx.atomistic.sampling.BiasKind.HARMONIC,
            variables,
            center=[1.0],
            stiffness=[2.0],
        ),
        dynamics,
    )
    runtime = phx.atomistic.sampling.PreparedBiasedDynamics(dynamics, bias)
    biased_state = runtime.initialize(state)
    checkpoint_plan = phx.atomistic.sampling.BiasedDynamicsCheckpointPlan(runtime)
    path = tmp_path / "biased.npz"
    written = phx.atomistic.sampling.write_biased_dynamics_checkpoint(
        path, checkpoint_plan, biased_state
    )
    restored = phx.atomistic.sampling.read_biased_dynamics_checkpoint(
        path, checkpoint_plan, biased_state
    )
    assert restored.payload_id == written.payload_id
    replay = runtime.replay(restored.state, 1)
    assert replay.accepted.shape == (1,)

    abf = phx.atomistic.sampling.PreparedAtomisticBias(
        phx.atomistic.sampling.AtomisticBiasPlan(
            phx.atomistic.sampling.BiasKind.ABF,
            variables,
            grid_minimum=[0.5],
            grid_maximum=[2.0],
            grid_bins=16,
        ),
        dynamics,
    )
    abf_state = abf.plan.initialize(state.kinematics.positions.dtype)
    evaluation = abf.evaluate(state.kinematics.positions, abf_state, state.time)
    updated = abf.update(abf_state, evaluation, jnp.ones_like(state.force.forces))
    assert int(jnp.sum(updated.abf_counts)) == 1
    assert bool(updated.successful)
    with pytest.raises(ValueError):
        phx.atomistic.sampling.AtomisticBiasPlan(
            phx.atomistic.sampling.BiasKind.METADYNAMICS,
            variables,
            maximum_hills=0,
        )


def test_free_energy_estimators_on_identical_states():
    fep = phx.uq.free_energy_perturbation([0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(fep.free_energies, [0.0, 0.0], atol=1.0e-12)
    ti = phx.uq.thermodynamic_integration(
        [0.0, 0.5, 1.0], [2.0, 2.0, 2.0], [0.1, 0.1, 0.1]
    )
    np.testing.assert_allclose(ti.free_energies[-1], 2.0, atol=1.0e-12)
    samples = phx.uq.ReducedPotentialSamples(jnp.zeros((2, 4)), [2, 2], [0, 0, 1, 1])
    mbar = phx.uq.multistate_bennett_acceptance_ratio(samples)
    assert bool(mbar.converged)
    np.testing.assert_allclose(mbar.free_energies, [0.0, 0.0], atol=1.0e-12)


def test_collective_variable_families():
    cell = phx.discretization.PeriodicCell(jnp.eye(3) * 8.0)
    system = phx.atomistic.AtomisticSystemPlan(
        [0, 1, 2, 3],
        [1, 1, 1, 1],
        [1.0, 2.0, 1.0, 2.0],
        phx.atomistic.AtomisticUnitSystem.reduced(),
        cell=cell,
    ).prepare()
    positions = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )
    kinds = (
        (phx.atomistic.sampling.CollectiveVariableKind.DISTANCE, [0, 1], {}, None),
        (phx.atomistic.sampling.CollectiveVariableKind.ANGLE, [0, 1, 2], {}, None),
        (
            phx.atomistic.sampling.CollectiveVariableKind.TORSION,
            [0, 1, 2, 3],
            {},
            None,
        ),
        (
            phx.atomistic.sampling.CollectiveVariableKind.CENTER_OF_MASS_DISTANCE,
            [0, 1, 2, 3],
            {"parameters": [2]},
            None,
        ),
        (
            phx.atomistic.sampling.CollectiveVariableKind.RADIUS_OF_GYRATION,
            [0, 1, 2, 3],
            {},
            None,
        ),
        (
            phx.atomistic.sampling.CollectiveVariableKind.COORDINATION,
            [[0, 1], [2, 3]],
            {"parameters": [1.5, 6.0]},
            None,
        ),
        (
            phx.atomistic.sampling.CollectiveVariableKind.CONTACT_SIMILARITY,
            [[0, 1], [2, 3]],
            {"parameters": [0.5], "reference": [1.0, 1.0]},
            None,
        ),
        (
            phx.atomistic.sampling.CollectiveVariableKind.ALIGNED_RMSD,
            [0, 1, 2, 3],
            {"reference": np.asarray(positions)},
            None,
        ),
        (
            phx.atomistic.sampling.CollectiveVariableKind.CELL_VOLUME,
            np.asarray([], dtype=np.int32),
            {},
            {"cell_vectors": cell.vectors},
        ),
        (
            phx.atomistic.sampling.CollectiveVariableKind.DENSITY,
            np.asarray([], dtype=np.int32),
            {},
            {"cell_vectors": cell.vectors},
        ),
        (
            phx.atomistic.sampling.CollectiveVariableKind.PATH_PROGRESS,
            [0, 1, 2, 3],
            {
                "parameters": [1.0],
                "reference": np.stack(
                    (np.asarray(positions), np.asarray(positions) + 1.0)
                ),
            },
            None,
        ),
        (
            phx.atomistic.sampling.CollectiveVariableKind.PATH_DISTANCE,
            [0, 1, 2, 3],
            {
                "parameters": [1.0],
                "reference": np.stack(
                    (np.asarray(positions), np.asarray(positions) + 1.0)
                ),
            },
            None,
        ),
    )
    for kind, indices, arguments, evaluation_arguments in kinds:
        prepared = phx.atomistic.sampling.CollectiveVariablePlan(
            kind, indices, **arguments
        ).prepare(system)
        evaluation = prepared.evaluate(
            positions,
            cell=cell,
            **({} if evaluation_arguments is None else evaluation_arguments),
        )
        assert bool(evaluation.successful), kind
        assert bool(jnp.isfinite(evaluation.value)), kind


def test_mdanalysis_frame_metadata_and_selection_adapters():
    mda = pytest.importorskip("MDAnalysis")
    universe = mda.Universe.empty(
        2,
        n_residues=2,
        n_segments=1,
        atom_resindex=[0, 1],
        residue_segindex=[0, 0],
        trajectory=True,
    )
    universe.add_TopologyAttr("names", ["H1", "H2"])
    universe.add_TopologyAttr("elements", ["H", "H"])
    universe.add_TopologyAttr("resids", [1, 2])
    universe.add_TopologyAttr("resnames", ["MOL", "MOL"])
    universe.add_TopologyAttr("chainIDs", ["A", "A"])
    universe.add_TopologyAttr("segids", ["SYSTEM"])
    universe.atoms.positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    frame = phx.atomistic.interchange.atomistic_frame_from_mdanalysis(
        universe,
        system_id="mda-system",
        topology_id="mda-topology",
        unit_system_id="mda-units",
        source_id="mda-frame",
    )
    metadata = phx.atomistic.interchange.atomistic_metadata_from_mdanalysis(universe)
    selection = phx.atomistic.interchange.mdanalysis_selection(
        universe, "name H1", stable_ids=[10, 20]
    )
    recovered = phx.atomistic.interchange.mdanalysis_universe_from_frames((frame,))
    assert metadata.atom_names == ("H1", "H2")
    np.testing.assert_array_equal(selection.mask, [True, False])
    np.testing.assert_allclose(recovered.atoms.positions, frame.positions)


def test_committee_diversity_and_advanced_methods():
    _, system, neighborhood, _, _, state = _runtime()
    frame = phx.atomistic.AtomisticFrame(
        0.0,
        0,
        state.kinematics.positions,
        system.plan.particle_ids,
        system_id=system.plan.system_id,
        topology_id=system.topology.topology_id,
        unit_system_id=system.plan.units.unit_system_id,
        source_id="acquisition-frame",
    )
    frames = tuple(
        phx.atomistic.AtomisticFrame(
            frame.time,
            index,
            frame.positions + index,
            frame.stable_ids,
            system_id=frame.system_id,
            topology_id=frame.topology_id,
            unit_system_id=frame.unit_system_id,
            source_id=f"frame-{index}",
        )
        for index in range(3)
    )
    evidence = tuple(
        phx.atomistic.AtomisticUncertaintyEvidence(
            jnp.asarray(score),
            jnp.zeros((3, 3)),
            jnp.zeros((3,)),
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            jnp.asarray(False),
            jnp.asarray(True),
            "committee",
        )
        for score in (3.0, 2.0, 1.0)
    )
    records = phx.atomistic.AcquisitionPlan(
        2, phx.atomistic.CommitteeAcquisitionScorePolicy(1.0, 1.0, 1.0)
    ).select(frames, evidence, descriptors=[[0.0], [0.1], [10.0]])
    assert tuple(record.source_index for record in records) == (0, 2)
    fallback = phx.atomistic.SegmentFallbackPolicy("reference-provider").decide(
        evidence[0]
    )
    assert not fallback.use_fallback
    assert fallback.reason == "primary"

    masses = system.plan.masses
    mobile = system.active_mask
    momenta = jnp.ones((3, 3)) * 0.1
    nhc = phx.atomistic.NoseHooverChainPlan(1.0, chain_length=3)
    first = nhc.apply(
        momenta,
        masses,
        mobile,
        jax.random.key(2),
        0,
        0.001,
        system.plan.units,
    )
    second = nhc.apply(
        first.momenta,
        masses,
        mobile,
        jax.random.key(2),
        1,
        0.001,
        system.plan.units,
        auxiliary=first.auxiliary,
    )
    splitting = phx.atomistic.AtomisticSplittingPlan(
        (
            phx.atomistic.SplittingOperatorKind.DRIFT,
            phx.atomistic.SplittingOperatorKind.FORCE_KICK,
        ),
        (0.5, 0.5),
    )
    split_value = splitting.apply(
        jnp.asarray(0.0),
        2.0,
        {
            phx.atomistic.SplittingOperatorKind.DRIFT: lambda value, width: value + width,
            phx.atomistic.SplittingOperatorKind.FORCE_KICK: lambda value, width: (
                value + width
            ),
        },
    )
    np.testing.assert_allclose(split_value, 2.0)
    assert first.auxiliary.shape == (3,)
    assert bool(second.successful)

    gle = phx.atomistic.GeneralizedLangevinPlan(
        [[1.0, 0.1], [0.1, 1.5]], [[0.2, 0.0], [0.0, 0.2]], 1.0
    )
    gle_result = gle.apply(
        momenta,
        masses,
        mobile,
        jax.random.key(3),
        0,
        0.001,
        system.plan.units,
    )
    assert gle_result.auxiliary.shape == (3, 3, 1)

    normal_modes = phx.atomistic.RingPolymerNormalModePlan(4, 1.0)
    beads = jnp.stack(tuple(state.kinematics.positions + 0.01 * i for i in range(4)))
    np.testing.assert_allclose(
        normal_modes.inverse(normal_modes.forward(beads)), beads, atol=1.0e-6
    )
    staging = phx.atomistic.StagingCoordinatePlan(4)
    np.testing.assert_allclose(
        staging.inverse(staging.forward(beads)), beads, atol=1.0e-12
    )
    contracted = phx.atomistic.ring_polymer_contract(beads, 2)
    np.testing.assert_allclose(jnp.mean(contracted, axis=0), jnp.mean(beads, axis=0))
    trpmd = phx.atomistic.ThermostattedRPMDPlan(normal_modes, [0.0, 1.0, 1.0, 1.0])
    trpmd_result = trpmd.apply(
        jnp.zeros_like(beads),
        masses,
        1.0,
        0.001,
        jax.random.key(4),
        system.plan.units,
    )
    assert bool(trpmd_result.successful)
    piglet = phx.atomistic.PIGLETPlan((gle,) * 4, normal_modes)
    piglet_result = piglet.apply(
        jnp.zeros_like(beads),
        masses,
        mobile,
        jax.random.key(5),
        0,
        0.001,
        system.plan.units,
    )
    assert bool(piglet_result.successful)
    polymer = phx.atomistic.RingPolymerState(
        beads,
        jnp.zeros_like(beads),
        jnp.asarray(0),
        jax.random.key(6),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.asarray(True),
        "barostat-test",
    )
    estimators = phx.atomistic.quantum_estimators(
        polymer,
        masses,
        jnp.zeros_like(beads),
        1.0,
        system.plan.units.boltzmann_constant,
        1.0,
        isotope_masses=masses * 2.0,
    )
    assert bool(estimators.successful)
    momentum_distribution = phx.atomistic.open_path_momentum_distribution(
        beads, jnp.zeros_like(beads)
    )
    assert momentum_distribution.shape == (4,)
    correction, directional = phx.atomistic.suzuki_chin_correction(
        lambda value: -value, beads, 0.1
    )
    assert bool(jnp.isfinite(correction) & jnp.all(jnp.isfinite(directional)))
    barostat = phx.atomistic.ConstantPressureRingPolymerPlan(1.0, 10.0)
    barostat_state = barostat.initialize(polymer, jnp.eye(3) * 6.0)
    barostat_state = barostat.step(barostat_state, 1.1, 0.001)
    assert bool(barostat_state.successful)
    assert bool(jnp.all(jnp.isfinite(barostat_state.cell_vectors)))

    manifold = phx.atomistic.ManifoldConstraintPlan(
        phx.atomistic.WallKind.CYLINDER, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0]
    )
    pressure = phx.atomistic.AnisotropicPressurePlan(
        [1.0, 1.0, 1.0], [0.01, 0.01, 0.01], semi_isotropic=True
    )
    updated_cell = pressure.update_cell(jnp.eye(3) * 6.0, [1.1, 0.9, 1.2], 0.01)
    assert bool(jnp.all(jnp.isfinite(updated_cell)))
    brownian = phx.atomistic.BrownianDynamicsPlan(0.1, 1.0).step(
        state.kinematics.positions,
        state.force.forces,
        0.001,
        jax.random.key(8),
        system.plan.units,
    )
    assert bool(jnp.all(jnp.isfinite(brownian)))
    projection = manifold.project(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    )
    assert bool(projection.successful)
    np.testing.assert_allclose(
        jnp.sqrt(jnp.sum(projection.positions[:, :2] ** 2, axis=-1)), 2.0
    )
    active = phx.atomistic.ActiveForcePlan(1.0, 0.1).evaluate(
        jnp.ones((3, 3)), jax.random.key(7), 0.01
    )
    assert bool(active.successful & jnp.all(jnp.isfinite(active.forces)))
    np.testing.assert_allclose(
        jnp.sqrt(jnp.sum(active.orientations**2, axis=-1)), 1.0, atol=1.0e-12
    )
    dpd_force = phx.atomistic.DissipativeParticleDynamicsPlan(
        1.0, 0.5, 1.0, 2.5
    ).pair_force(
        jnp.asarray([[1.0, 0.0, 0.0]]),
        jnp.asarray([[0.1, 0.0, 0.0]]),
        jnp.asarray([0.2]),
        1.0,
        0.01,
    )
    assert bool(jnp.all(jnp.isfinite(dpd_force)))
    wall = phx.atomistic.AtomisticPotentialProgram(
        [
            phx.atomistic.ScalarWallPotential(
                phx.atomistic.WallKind.PLANE, [1.0, 0.0, 0.0, 0.5], 2.0
            )
        ]
    ).prepare(system)
    wall_evaluation = wall.evaluate(
        state.kinematics.positions, neighborhood.build(state.kinematics.positions)
    )
    assert bool(wall_evaluation.successful)

    eam = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.EAMPotential([2.0, 1.0, 1.0, 0.5, 2.0], 2.5)]
    ).prepare(system)
    eam_result = eam.evaluate(
        state.kinematics.positions, neighborhood.build(state.kinematics.positions)
    )
    assert bool(eam_result.successful)

    assert bool(jnp.isfinite(eam_result.energy))
    many_body_cases = (
        (
            phx.atomistic.StillingerWeberPotential,
            [1.0, 1.0, 7.049556277, 0.6022245584, 4.0, 0.0, 1.8, 21.0, 1.2, -1.0 / 3.0],
            1.8,
        ),
        (
            phx.atomistic.TersoffPotential,
            [1.0, 0.5, 2.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, -0.3, 2.0, 0.5, 3.0],
            2.5,
        ),
    )
    for factory, parameters, cutoff in many_body_cases:
        many_body = phx.atomistic.AtomisticPotentialProgram(
            [factory(parameters, cutoff)]
        ).prepare(system)
        evaluation = many_body.evaluate(
            state.kinematics.positions,
            neighborhood.build(state.kinematics.positions),
        )
        assert bool(evaluation.successful)
        assert bool(jnp.all(jnp.isfinite(evaluation.forces)))


def test_rigid_coordinate_map_and_rotational_step():
    particles = phx.discretization.ParticleSetPlan(
        [0, 1], [1.0, 1.0], ambient_dimension=3
    ).prepare()
    bodies = phx.discretization.RigidBodySetPlan(
        [0, 0], jnp.stack((jnp.eye(3), jnp.eye(3)))
    ).prepare(particles)
    orientation = jnp.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    kinematics = bodies.kinematics(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        jnp.zeros((2, 3)),
        orientation,
        jnp.zeros((2, 3)),
    )
    coordinate_map = phx.atomistic.RigidAtomisticCoordinateMap(
        bodies,
        [0, 1],
        [[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]],
        [10, 20],
    )
    sites = coordinate_map.realize(kinematics)
    np.testing.assert_allclose(sites.positions, [[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]])
    load = phx.discretization.RigidBodyLoad(
        jnp.zeros((2, 3)), jnp.asarray([[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]])
    )
    state = phx.atomistic.RotationalAtomisticState(
        kinematics, load.force, load.torque, jnp.asarray(True)
    )
    advanced = phx.atomistic.rotational_velocity_verlet(
        bodies,
        state,
        1.0e-3,
        lambda time, value, args: load,
    )
    assert bool(advanced.successful)
    np.testing.assert_allclose(
        jnp.sqrt(jnp.sum(advanced.kinematics.orientation**2, axis=-1)),
        1.0,
        atol=1.0e-12,
    )


def test_implicit_polarization_and_multipole_pme():
    cell = phx.discretization.PeriodicCell(jnp.eye(3) * 6.0)
    _, system, neighborhood, _, _, state = _runtime(cell=cell, charges=[0.4, -0.2, -0.2])
    multipoles = phx.atomistic.PermanentMultipoleSiteData(
        [0.4, -0.2, -0.2],
        jnp.zeros((3, 3)),
        jnp.zeros((3, 3, 3)),
        [0.1, 0.1, 0.1],
        [1.0, 1.0, 1.0],
    )
    plan = phx.atomistic.PolarizationPlan(maximum_iterations=200, tolerance=1.0e-6)
    primal, tangent = phx.atomistic.implicit_polarization_jvp(
        plan,
        state.kinematics.positions,
        jnp.ones_like(state.kinematics.positions) * 0.01,
        multipoles,
    )
    assert bool(jnp.all(jnp.isfinite(primal)))
    assert bool(jnp.all(jnp.isfinite(tangent)))
    site_state = system.coordinate_map.realize(
        state.kinematics.positions, cell=system.cell
    )
    pme = phx.atomistic.MultipolePMEPlan((8, 8, 8), 0.4)
    energy = pme.energy(site_state, multipoles, cell.vectors, 1.0)
    assert bool(jnp.isfinite(energy))
    dispersion = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.LennardJonesPMEPotential([[0.1]], 0.4, 2.5, (8, 8, 8))]
    ).prepare(system)
    dispersion_result = dispersion.evaluate(
        state.kinematics.positions, neighborhood.build(state.kinematics.positions)
    )
    assert bool(dispersion_result.successful)
    assert bool(jnp.all(jnp.isfinite(dispersion_result.forces)))
    np.testing.assert_allclose(
        jnp.sum(dispersion_result.forces, axis=0), 0.0, atol=1.0e-8
    )
    gb = phx.atomistic.ImplicitSolventPlan("gb").energy(
        state.kinematics.positions, [0.4, -0.2, -0.2], [1.2, 1.2, 1.2], 1.0
    )
    gk = phx.atomistic.ImplicitSolventPlan("gk").energy(
        state.kinematics.positions, [0.4, -0.2, -0.2], [1.2, 1.2, 1.2], 1.0
    )
    assert bool(jnp.isfinite(gb) & jnp.isfinite(gk))
    assert not bool(jnp.isclose(gb, gk))


def _ipi_roundtrip(plan, system, positions):
    listener = plan.listen()

    def evaluator(prepared, coordinate, cell_vectors):
        del prepared, cell_vectors
        return phx.atomistic.ExternalAtomisticEvaluation(
            jnp.sum(coordinate**2),
            -2.0 * coordinate,
            jnp.zeros((3, 3)),
            jnp.asarray(True),
            "local-provider",
        )

    provider = phx.atomistic.CallableBornOppenheimerProvider(evaluator, "local-provider")

    def serve():
        with listener.accept() as session:
            return phx.atomistic.interchange.serve_ipi_once(session, provider, system)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(serve)
        with plan.connect() as session:
            remote = phx.atomistic.interchange.TransportedExternalAtomisticProvider(
                session, "remote-provider"
            )
            result = remote.evaluate(system, positions, None)
        status = future.result(timeout=10.0)
    listener.close()
    assert status is phx.atomistic.interchange.IPITransportStatus.READY
    np.testing.assert_allclose(result.energy, jnp.sum(positions**2))
    np.testing.assert_allclose(result.forces, -2.0 * positions)


@pytest.mark.parametrize("mode", ["unix", "tcp"])
def test_ipi_unix_and_tcp_roundtrip(tmp_path: Path, mode: str):
    _, system, _, _, _, state = _runtime()
    if mode == "unix":
        plan = phx.atomistic.interchange.IPITransportPlan.unix(
            f"/tmp/phydrax-ipi-{os.getpid()}.sock", timeout=5.0
        )
    else:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        plan = phx.atomistic.interchange.IPITransportPlan.tcp(
            "127.0.0.1", port, timeout=5.0
        )
    _ipi_roundtrip(plan, system, state.kinematics.positions)


def test_packmol_subprocess_boundary_and_provenance(tmp_path: Path):
    executable = tmp_path / "fake_packmol.py"
    executable.write_text(
        """#!/usr/bin/env python3
from pathlib import Path
import sys
lines = sys.stdin.read().splitlines()
structure = next(line.split()[1] for line in lines if line.startswith('structure '))
number = int(next(line.split()[1] for line in lines if line.strip().startswith('number ')))
source = Path(structure).read_text().splitlines()
atoms = source[2:]
output = []
for molecule in range(number):
    for atom in atoms:
        fields = atom.split()
        output.append(f'{fields[0]} {float(fields[1]) + 4.0 * molecule} {fields[2]} {fields[3]}')
Path('output.xyz').write_text(f'{len(output)}\\nfake packmol\\n' + '\\n'.join(output) + '\\n')
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    template = phx.atomistic.AtomisticFrame(
        0.0,
        0,
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [10, 20],
        system_id="packmol-system",
        topology_id="packmol-topology",
        unit_system_id="packmol-units",
        source_id="packmol-template",
    )
    component = phx.atomistic.interchange.PackmolComponentPlan(
        template,
        2,
        constraints=(
            phx.atomistic.interchange.PackmolRegionConstraint(
                "inside-box", [0.0, 0.0, 0.0, 10.0, 10.0, 10.0]
            ),
        ),
    )
    result = phx.atomistic.interchange.PackmolAssemblyPlan(
        (component,), executable=str(executable)
    ).run()
    assert result.successful
    assert result.positions.shape == (4, 3)
    assert result.component_slices == ((0, 4),)
    assert result.input_digest
