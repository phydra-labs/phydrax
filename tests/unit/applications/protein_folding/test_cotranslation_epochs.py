import hashlib
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.protein_folding._construct import ProteinConstruct
from phydrax.applications.protein_folding.cotranslation import (
    CotranslationProtocol,
    CotranslationStage,
    NascentChainObservations,
    RibosomeBoundaryPotential,
)
from phydrax.atomistic import (
    AtomisticDynamicsPlan,
    AtomisticPotentialProgram,
    AtomisticSystemPlan,
    AtomisticUnitSystem,
    BAOABLangevinPlan,
    DistanceConstraintPlan,
    HarmonicBondPotential,
    LennardJonesPotential,
    MolecularTopologyPlan,
    VelocityVerletPlan,
)
from phydrax.atomistic._topology_epoch import (
    activate_topology_epoch,
    prepare_dormant_system,
)
from phydrax.discretization import DenseParticleNeighborhoodPlan
from phydrax.qualification import ReferenceArtifactManifest


def _source(label="analytical-fixture", *, commercial=True):
    content = label.encode()
    return ReferenceArtifactManifest(
        label,
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        license_id="CC0-1.0",
        commercial_use_permitted=commercial,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"reduced-length": 1.0},
        uncertainty=None,
        lineage_ids=("independently-specified-numerical-fixture",),
    )


def _fixture(*, constrained=False, work_bound=None, thermal=False):
    ids = (101, 205, 309)
    topology = MolecularTopologyPlan(
        bonds=[[101, 205], [205, 309]],
        pair_exceptions=[[101, 205], [205, 309]],
        lennard_jones_scales=[0.0, 0.0],
        electrostatic_scales=[0.0, 0.0],
        constraints=[[205, 309]] if constrained else None,
        constraint_distances=[1.0] if constrained else None,
    )
    material = AtomisticSystemPlan(
        ids,
        [0, 0, 0],
        [1.0, 2.0, 3.0],
        AtomisticUnitSystem.reduced(),
        element_mask=[False, False, False],
        atom_type_ids=[0, 0, 0],
        molecule_ids=[0, 0, 0],
        topology=topology,
    ).prepare()
    stages = []
    for count in (1, 2, 3, 3):
        released = len(stages) == 3
        system = prepare_dormant_system(material, ids[:count])
        boundary = RibosomeBoundaryPotential(
            tether_particle_id=None if released else ids[count - 1],
            anchor=[1.2 * (count - 1), 0.0, 0.0],
            tether_stiffness=0.0 if released else 0.1,
            sphere_centers=[[-2.0, 0.0, 0.0]],
            sphere_radii=[1.0],
            exclusion_stiffness=0.2,
        )
        potential = AtomisticPotentialProgram(
            [
                HarmonicBondPotential([2.0], [1.0]),
                LennardJonesPotential([0.02], [0.5], 3.0, switch_distance=2.5),
                boundary,
            ]
        ).prepare(system)
        runtime = AtomisticDynamicsPlan(
            system,
            potential,
            DenseParticleNeighborhoodPlan(3).prepare(system.particles),
            BAOABLangevinPlan(1e-3, 0.1, 0.2) if thermal else VelocityVerletPlan(1e-3),
            constraints=DistanceConstraintPlan(tolerance=1e-10).prepare(system)
            if constrained
            else None,
        ).prepare()
        inserted = count > 1 and not released
        stages.append(
            CotranslationStage(
                runtime,
                count,
                3,
                None if released else "GCU",
                f"synthetic-stage:{len(stages)}",
                ((1.2 * (count - 1), 0.0, 0.0),) if inserted else (),
                ((0.03, 0.0, 0.0),) if inserted else (),
                work_bound,
            )
        )
    source = _source()
    protocol = CotranslationProtocol(
        ProteinConstruct(("A",), ("AAA",)), ids, tuple(stages), source, source
    )
    return material, protocol


def _initial(protocol):
    return protocol.initialize(
        jnp.zeros((3, 3)), jnp.zeros((3, 3)), key=jax.random.key(8)
    )


def test_dormant_material_has_no_topology_or_preactivation_force():
    material, protocol = _fixture()
    runtime = protocol.stages[0].runtime
    state = _initial(protocol).state
    assert runtime.system.topology.bond_indices.shape[0] == 0
    assert runtime.system.topology.exception_indices.shape[0] == 0
    np.testing.assert_array_equal(state.force.forces[1:], 0.0)
    positions = state.kinematics.positions.at[1:].set(
        jnp.array([[0.01, 0, 0], [0.02, 0, 0]])
    )
    evaluation = runtime.potential.evaluate(
        positions, runtime.neighborhood.build(positions)
    )
    np.testing.assert_array_equal(evaluation.forces, state.force.forces)
    np.testing.assert_array_equal(evaluation.energy, state.energy.potential_energy)
    transition = protocol.transition(1)
    activation = activate_topology_epoch(transition, state, [[1.2, 0, 0]], [[0.03, 0, 0]])
    assert activation.successful
    assert activation.runtime.system.prepared_id != runtime.system.prepared_id
    assert (
        activation.runtime.neighborhood.particle_discretization_id
        != runtime.neighborhood.particle_discretization_id
    )
    np.testing.assert_array_equal(
        activation.runtime.system.topology.plan.bonds, [[101, 205]]
    )
    np.testing.assert_array_equal(
        activation.runtime.system.topology.lennard_jones_scales, [0.0]
    )
    assert float(jnp.max(jnp.abs(activation.state.force.forces))) > 0.1
    np.testing.assert_array_equal(activation.state.force.forces[2], 0.0)
    assert material.topology.bond_indices.shape[0] == 2


def test_insertion_sources_close_energy_mass_and_momentum_balance():
    _, protocol = _fixture()
    initial = _initial(protocol).state
    activation = activate_topology_epoch(
        protocol.transition(1), initial, [[1.2, 0, 0]], [[0.03, 0, 0]]
    )
    assert activation.successful
    ledger = activation.ledger
    np.testing.assert_allclose(ledger.mass_source, 2.0)
    np.testing.assert_allclose(ledger.momentum_source, [0.03, 0, 0], atol=1e-14)
    np.testing.assert_allclose(ledger.boundary_impulse, 0.0, atol=1e-14)
    np.testing.assert_allclose(ledger.carried_kinetic_energy, 0.5 * 0.03**2 / 2.0)
    np.testing.assert_allclose(
        activation.state.energy.total_energy - initial.energy.total_energy,
        ledger.external_work,
    )
    np.testing.assert_allclose(
        activation.state.energy.external_work, ledger.external_work
    )
    step = eqx.filter_jit(activation.runtime.step_detailed)(activation.state)
    assert bool(step.successful)
    energy = step.accepted_state.energy
    initial_total = energy.initial_kinetic_energy + energy.initial_potential_energy
    np.testing.assert_allclose(
        energy.total_energy
        - initial_total
        - energy.external_work
        - energy.constraint_work
        - energy.thermostat_heat,
        energy.cumulative_balance_residual,
        atol=1e-12,
    )


def test_activation_rebuilds_constraint_executor_and_projects_inserted_geometry():
    _, protocol = _fixture(constrained=True)
    first = protocol.run(_initial(protocol), stop_after_stage=1)
    assert first.successful
    activation = activate_topology_epoch(
        protocol.transition(2), first.cursor.state, [[2.4, 0, 0]], [[0.03, 0, 0]]
    )
    assert activation.successful
    assert activation.runtime.system.topology.constraint_count == 1
    positions = activation.state.kinematics.positions
    np.testing.assert_allclose(
        jnp.sqrt(jnp.sum((positions[2] - positions[1]) ** 2)), 1.0, atol=1e-10
    )
    np.testing.assert_allclose(
        activation.state.constraint_velocity_residual, 0.0, atol=1e-10
    )
    assert (
        activation.runtime.constraints.system.prepared_id
        == activation.runtime.system.prepared_id
    )


def test_failed_work_admission_and_singular_insertion_rollback_without_mutation():
    _, protocol = _fixture()
    initial = _initial(protocol).state
    transition = replace(protocol.transition(1), maximum_absolute_work=0.0)
    rejected = activate_topology_epoch(transition, initial, [[1.2, 0, 0]], [[0.03, 0, 0]])
    assert not rejected.successful
    assert rejected.state is initial and rejected.runtime is transition.before
    assert rejected.ledger is None
    singular = activate_topology_epoch(
        protocol.transition(1), initial, [[0, 0, 0]], [[0, 0, 0]]
    )
    assert not singular.successful and singular.state is initial
    good = activate_topology_epoch(
        protocol.transition(1), initial, [[1.2, 0, 0]], [[0.03, 0, 0]]
    )
    assert good.successful
    np.testing.assert_array_equal(good.state.random_key, initial.random_key)


def test_complete_protocol_checkpoint_replay_and_schedule_scope(tmp_path):
    _, protocol = _fixture(thermal=True)
    initial = _initial(protocol)
    before = protocol.run(initial, stop_after_stage=0)
    path = tmp_path / "nascent.chk"
    protocol.write_checkpoint(path, before.cursor)
    restored = protocol.read_checkpoint(path, initial)
    whole = protocol.run(initial)
    replay = protocol.run(restored)
    assert whole.successful and replay.successful
    assert whole.cursor.stage_index == 3 and whole.cursor.completed_steps == 3
    assert int(whole.cursor.state.step_index) == 12
    for a, b in zip(
        jax.tree_util.tree_leaves(whole.cursor.state),
        jax.tree_util.tree_leaves(replay.cursor.state),
        strict=True,
    ):
        if eqx.is_array(a):
            np.testing.assert_array_equal(a, b)
    assert len(whole.insertions) == 3
    assert len(whole.segments) == 4
    for left, right in zip(whole.segments[:-1], whole.segments[1:], strict=True):
        np.testing.assert_allclose(
            left.support.coordinates[-1], right.support.coordinates[0]
        )
        assert left.support.coordinate_id != right.support.coordinate_id
    changed = replace(protocol, schedule_source=_source("different-source-same-runtime"))
    with pytest.raises(ValueError):
        changed.read_checkpoint(path, _initial(changed))


def test_protocol_refuses_biological_and_identity_shortcuts():
    _, protocol = _fixture()
    with pytest.raises(ValueError):
        replace(protocol, residue_particle_ids=(0, 1, 2))
    with pytest.raises(ValueError):
        replace(protocol.stages[1], codon="GCT")
    with pytest.raises(ValueError):
        replace(
            protocol,
            stages=(
                protocol.stages[0],
                replace(protocol.stages[1], codon="UGG"),
                *protocol.stages[2:],
            ),
        )
    with pytest.raises(ValueError):
        replace(
            protocol,
            timing_calibration=_source(),
            timing_calibration_scope="uncalibrated reduced timing",
        )
    with pytest.raises(PermissionError):
        replace(protocol, parameter_source=_source(commercial=False), commercial_use=True)


def test_contact_observation_preserves_future_coverage():
    _, protocol = _fixture()
    observer = NascentChainObservations(
        protocol.stages[1].runtime.system,
        contact_particle_pairs=((101, 205), (205, 309)),
        reference_distances=(1.0, 1.0),
        contact_width=0.2,
    )
    observation = eqx.filter_jit(observer.evaluate)(
        jnp.array([[0.0, 0, 0], [1.0, 0, 0], [50.0, 0, 0]])
    )
    np.testing.assert_allclose(observation.contact_similarity, 1.0)
    assert int(observation.contact_count) == 1
    assert bool(observation.contact_available) and not bool(
        observation.entanglement_available
    )


def test_gauss_entanglement_orientation_rigid_invariance_and_crossing_refusal():
    ids = (11, 22, 33, 44)
    system = AtomisticSystemPlan(
        ids, [0] * 4, [1.0] * 4, AtomisticUnitSystem.reduced(), element_mask=[False] * 4
    ).prepare()
    kwargs = dict(
        contact_particle_pairs=(),
        reference_distances=(),
        contact_width=1.0,
        left_curve_ids=(11, 22),
        right_curve_ids=(33, 44),
        quadrature_order=8,
    )
    observer = NascentChainObservations(system, **kwargs)
    positions = jnp.array([[-1.0, 0, 0], [1.0, 0, 0], [0, -1.0, 1.0], [0, 1.0, 1.0]])
    observed = eqx.filter_jit(observer.evaluate)(positions)
    assert bool(observed.successful) and abs(float(observed.gauss_entanglement)) > 0.01
    rotation = jnp.array([[0.0, -1.0, 0], [1.0, 0, 0], [0, 0, 1.0]])
    moved = observer.evaluate(positions @ rotation.T + jnp.array([4.0, 2.0, -3.0]))
    np.testing.assert_allclose(
        moved.gauss_entanglement, observed.gauss_entanglement, atol=1e-12
    )
    reversed_observer = NascentChainObservations(
        system, **{**kwargs, "left_curve_ids": (22, 11)}
    )
    np.testing.assert_allclose(
        reversed_observer.evaluate(positions).gauss_entanglement,
        -observed.gauss_entanglement,
        atol=1e-12,
    )
    crossing = observer.evaluate(positions.at[2:, 2].set(0.0))
    assert not bool(crossing.successful)
    np.testing.assert_allclose(crossing.curve_separation, 0.0)


def test_ribosome_exclusion_and_tether_produce_conservative_nonzero_forces():
    _, protocol = _fixture()
    runtime = protocol.stages[0].runtime
    positions = jnp.array([[-1.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    neighborhood = runtime.neighborhood.build(positions)
    evaluated = eqx.filter_jit(runtime.potential.evaluate)(positions, neighborhood)
    assert bool(evaluated.successful)
    # The soft sphere contributes +0.1 and the tether contributes +0.15.
    np.testing.assert_allclose(evaluated.forces[0], [0.25, 0.0, 0.0], atol=1e-12)
    delta = jnp.zeros_like(positions).at[0, 0].set(1e-5)
    plus = runtime.potential.energy(positions + delta, neighborhood)[0]
    minus = runtime.potential.energy(positions - delta, neighborhood)[0]
    np.testing.assert_allclose(
        -(plus - minus) / (2e-5), evaluated.forces[0, 0], rtol=1e-9
    )
    np.testing.assert_array_equal(evaluated.forces[1:], 0.0)
