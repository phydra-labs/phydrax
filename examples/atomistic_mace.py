"""Construct and train a native MACE potential on energy, force, and stress labels.

A fixed "teacher" MACE labels a few strained periodic water-like cells through
the native batch E/F/S prediction route. The labels become one immutable
training source; a loader streams it and a differently initialized student is
fitted with one declared objective per label kind (structure-balanced per-atom
energy, component force and cell-volume stress MSE) through the shared training
kernel. Both models are random native initializations, so this demonstrates the
workflow, not accuracy.
"""

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


UNITS = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
BASE_CELL = np.array([[4.2, 0.0, 0.0], [0.9, 4.0, 0.0], [0.5, -0.6, 4.4]])
POSITIONS = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
MAGNITUDES = (-0.02, 0.0, 0.015, 0.03)


def build_model(seed: int) -> phx.nn.atomistic.MACEPotential:
    architecture = phx.nn.atomistic.MACEArchitecture(
        species=(1, 8),
        cutoff=3.0,
        radial_basis_count=4,
        cutoff_power=5,
        channel_count=4,
        hidden_degree=1,
        edge_degree=2,
        interactions=("real-agnostic", "real-agnostic-residual"),
        correlations=(2, 2),
        radial_widths=(8,),
        readout_width=4,
        average_neighbor_count=2.0,
    )
    return phx.nn.atomistic.MACEPotential(
        UNITS.scale,
        architecture,
        atomic_energies=np.asarray([[-1.0, -2.0]], dtype=np.float64),
        key=jax.random.key(seed),
    )


def strained_batch() -> phx.atomistic.AtomisticBatch:
    structures = []
    for index, magnitude in enumerate(MAGNITUDES):
        deformation = np.eye(3) + magnitude * np.array(
            [[1.0, 0.2, 0.0], [0.2, -0.5, 0.1], [0.0, 0.1, 0.7]]
        )
        structures.append(
            phx.atomistic.AtomicStructure(
                jnp.asarray([8, 1, 1]),
                jnp.asarray(POSITIONS @ deformation.T),
                jnp.asarray([15.999, 1.008, 1.008]),
                UNITS.scale,
                cell=jnp.asarray(BASE_CELL @ deformation.T),
                periodic_axes=jnp.asarray([True, True, True]),
                name=f"strained-water-{index}",
            )
        )
    return phx.atomistic.AtomisticBatch.from_structures(structures)


def main() -> None:
    # One reusable training layout: the declared search envelope (the base
    # cell plus a deformation margin) covers every strained cell.
    execution = phx.atomistic.AtomisticGraphExecutionPlan(
        32,
        backend="particle",
        streamed=phx.sparse.StreamedRelationPlan(receiver_tile=16, edge_tile=128),
        image_capacity=phx.discretization.ParticleImageCapacity(
            maximum_particles_per_cell=16,
            maximum_edges=1024,
            maximum_degree=64,
            maximum_images=343,
        ),
        search_cell=BASE_CELL,
        search_margin=0.1,
    )
    batch = strained_batch()
    labels = phx.atomistic.energy_and_forces(
        build_model(0), batch, execution, compute_stress=True
    )
    if not bool(jnp.all(labels.valid)):
        raise RuntimeError("Teacher labels failed.")
    source = phx.atomistic.atomistic_training_source(
        batch,
        mass_unit=UNITS.mass_unit,
        energy=labels.energy,
        forces=labels.forces,
        stress=labels.stress,
        record_ids=tuple(f"strained-water-{magnitude}" for magnitude in MAGNITUDES),
    )
    # batch_size=None is the full-batch policy: one global batch per update.
    problem = phx.atomistic.AtomisticTrainingProblem(
        phx.atomistic.atomistic_training_loader(source),
        execution,
        (
            phx.atomistic.AtomisticEnergyObjective(),
            phx.atomistic.AtomisticForceObjective(),
            phx.atomistic.AtomisticStressObjective(),
        ),
        scale=UNITS.scale,
        cutoff=3.0,
        fields=phx.atomistic.AtomisticTrainingFields(cell="cell", periodic="periodic"),
    )
    policy = phx.atomistic.AtomisticTrainingPolicy(maximum_steps=25, learning_rate=1.0e-2)
    student = build_model(1)
    result = phx.atomistic.fit_atomistic_potential(
        student, problem, policy, key=jax.random.key(7)
    )
    if not bool(result.successful):
        raise RuntimeError(f"Training failed: {result.termination}")
    print(
        "fitted scales:",
        dict(zip(result.normalization.names, result.normalization.scales)),
    )
    print("loss history:", np.asarray(result.training_values))
    print("stress loss history:", np.asarray(result.training_objectives)[:, 2])
    trained = phx.atomistic.energy_and_forces(
        result.potential, batch, execution, compute_stress=True
    )
    if trained.stress is None or labels.stress is None:
        raise RuntimeError("Requested stress is absent from prediction.")
    print("trained energy error [eV]:", np.asarray(trained.energy - labels.energy))
    print(
        "trained stress error [eV/Ang^3]:", np.asarray(trained.stress - labels.stress)[0]
    )


if __name__ == "__main__":
    main()
