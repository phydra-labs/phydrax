import jax.numpy as jnp
import jax.random as jr
import numpy as np

from phydrax.atomistic import (
    AtomicStructure,
    AtomisticBatch,
    AtomisticGraphExecutionPlan,
    AtomisticSystemPlan,
    AtomisticUnitSystem,
    realize_atomistic_graph,
)
from phydrax.atomistic._spin import (
    ClassicalSpinHamiltonianPlan,
    evaluate_classical_spin_hamiltonian,
    prepare_classical_spin_hamiltonian,
)
from phydrax.atomistic._spin_checkpoint import (
    AtomisticSpinCheckpointPlan,
    read_atomistic_spin_checkpoint,
    write_atomistic_spin_checkpoint,
)
from phydrax.atomistic._spin_dynamics import (
    initial_spin_dynamics_state,
    LandauLifshitzGilbertPlan,
    prepare_llg_dynamics,
    solve_llg_dynamics,
)
from phydrax.stochastic import WienerRealization


def _dimer():
    units = AtomisticUnitSystem.reduced()
    positions = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    system = AtomisticSystemPlan([20, 10], [1, 1], [1.0, 1.0], units).prepare()
    structure = AtomicStructure(
        [1, 1], positions, [1.0, 1.0], units.scale, particle_ids=[20, 10]
    )
    graph = realize_atomistic_graph(
        AtomisticBatch.from_structure(structure),
        AtomisticGraphExecutionPlan(1, maximum_dense_atoms=2),
        cutoff=2.0,
    )
    return units, ClassicalSpinHamiltonianPlan(
        system,
        graph,
        energy_unit="reduced-energy",
        field_unit="reduced-field",
        moment_unit="reduced-energy/reduced-field",
    )


def test_once_per_bond_exchange_dmi_and_effective_field_signs():
    _, plan = _dimer()
    prepared = prepare_classical_spin_hamiltonian(
        plan,
        exchange=[2.0],
        dmi=[[0.0, 0.0, 3.0]],
        moments=[2.0, 4.0],
    )
    # Canonical orientation follows stable IDs: site 1 (ID 10) -> site 0 (ID 20).
    directions = jnp.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    result = evaluate_classical_spin_hamiltonian(prepared, directions)
    assert plan.bond_senders.tolist() == [1]
    assert plan.bond_receivers.tolist() == [0]
    assert jnp.allclose(result.exchange_energy, 0.0)
    assert jnp.allclose(result.dmi_energy, -3.0)
    assert jnp.allclose(result.total_energy, -3.0)
    assert jnp.allclose(
        result.total_energy,
        result.exchange_energy
        + result.symmetric_exchange_energy
        + result.dmi_energy
        + result.anisotropy_energy
        + result.zeeman_energy,
    )
    # H_eff=-(1/mu)dE/dm: compare a tangent rotation on S².
    epsilon = 1.0e-6
    shifted_direction = jnp.asarray([1.0, epsilon, 0.0])
    shifted_direction = shifted_direction / jnp.linalg.norm(shifted_direction)
    shifted = directions.at[1].set(shifted_direction)
    numerical = (
        evaluate_classical_spin_hamiltonian(prepared, shifted).total_energy
        - result.total_energy
    ) / epsilon
    assert jnp.allclose(
        numerical,
        -prepared.moments[1] * result.effective_field[1, 1],
        atol=1.0e-5,
    )


def test_anisotropy_zeeman_sign_and_rkmk_norm_preservation():
    _, hamiltonian_plan = _dimer()
    hamiltonian = prepare_classical_spin_hamiltonian(
        hamiltonian_plan,
        anisotropy=[2.0, 2.0],
        anisotropy_axes=[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        moments=[1.0, 1.0],
        magnetic_field=[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
    )
    aligned = evaluate_classical_spin_hamiltonian(
        hamiltonian, [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
    )
    transverse = evaluate_classical_spin_hamiltonian(
        hamiltonian, [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    assert jnp.allclose(aligned.total_energy, -6.0)
    assert aligned.total_energy < transverse.total_energy

    dynamics = prepare_llg_dynamics(
        LandauLifshitzGilbertPlan(
            hamiltonian,
            [1.0, 1.0],
            [0.1, 0.1],
            step_size=1.0e-3,
            maximum_steps=20,
        )
    )
    state = initial_spin_dynamics_state(dynamics, [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    trajectory = solve_llg_dynamics(dynamics, state, jnp.asarray([0.0, 0.01]))
    assert trajectory.evidence.interpretation == "deterministic"
    assert (
        jnp.max(jnp.abs(jnp.linalg.norm(trajectory.directions, axis=-1) - 1.0)) < 1.0e-8
    )
    assert trajectory.directions[-1, 0, 2] > trajectory.directions[0, 0, 2]


def test_thermal_llg_uses_stratonovich_srkmk_and_fdt_amplitude():
    _, hamiltonian_plan = _dimer()
    hamiltonian = prepare_classical_spin_hamiltonian(
        hamiltonian_plan,
        moments=[1.0, 2.0],
        magnetic_field=[[0.0, 0.0, 0.2], [0.0, 0.0, 0.2]],
    )
    dynamics = prepare_llg_dynamics(
        LandauLifshitzGilbertPlan(
            hamiltonian,
            [1.0, 1.5],
            [0.1, 0.2],
            temperature=[0.5, 0.75],
            step_size=1.0e-3,
            maximum_steps=4,
        )
    )
    state = initial_spin_dynamics_state(dynamics, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    realization = WienerRealization(
        jr.key(7),
        (6,),
        support=(0.0, 0.002),
        tolerance=1.0e-5,
        noise_id=f"llg-field-noise:{dynamics.prepared_id}",
    )
    trajectory = solve_llg_dynamics(
        dynamics, state, jnp.asarray([0.0, 0.001, 0.002]), realization=realization
    )
    assert trajectory.evidence.interpretation == "stratonovich"
    assert trajectory.evidence.solver_id.startswith("solver:srkmk:")
    assert trajectory.evidence.wiener_realization_id == realization.realization_id
    assert trajectory.evidence.coordinate_map_id is not None
    assert trajectory.evidence.fluctuation_dissipation_residual < 1.0e-10
    assert trajectory.evidence.maximum_norm_residual < 1.0e-8


def test_spin_checkpoint_roundtrip_preserves_runtime_identity(tmp_path):
    _, hamiltonian_plan = _dimer()
    hamiltonian = prepare_classical_spin_hamiltonian(
        hamiltonian_plan, exchange=[1.0], moments=[1.0, 1.0]
    )
    dynamics = prepare_llg_dynamics(
        LandauLifshitzGilbertPlan(
            hamiltonian,
            [1.0, 1.0],
            [0.0, 0.0],
            step_size=0.01,
            maximum_steps=2,
        )
    )
    state = initial_spin_dynamics_state(dynamics, [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    plan = AtomisticSpinCheckpointPlan(dynamics, scope_id="spin-restart")
    written = write_atomistic_spin_checkpoint(tmp_path / "spin.npz", plan, state)
    restored = read_atomistic_spin_checkpoint(tmp_path / "spin.npz", plan, state)
    assert restored.payload_id == written.payload_id
    assert restored.checkpoint_id == plan.checkpoint_id
    assert jnp.array_equal(restored.state.directions, state.directions)
    assert restored.state.prepared_dynamics_id == dynamics.prepared_id
