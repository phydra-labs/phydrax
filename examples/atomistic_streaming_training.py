"""Stream minibatches of a source mixture into a native atomistic fit.

A fixed "teacher" PaiNN labels perturbed water and hydrogen molecules. Each
species group becomes one immutable training source; a seeded epoch mixture
draws the hydrogen source twice per epoch. Every global batch of four draws is
one accumulation window packed as two-case payloads, so numerators and
supports merge across payloads before the update. Energies are matched
relative to native least-squares per-species reference energies, the force
weight follows a cosine schedule over committed updates, the merged gradient is
clipped, and validation selects the EMA target view on a lineage-disjoint
loader. A checkpoint written after a few updates continues the run.
Random native initializations only: the workflow, not accuracy.
"""

import tempfile
from pathlib import Path

import jax
import numpy as np

import phydrax as phx


UNITS = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
CUTOFF = 2.5
WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
HYDROGEN = np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0], [0.0, 0.0, 0.0]])


def build_model(seed: int) -> phx.nn.atomistic.PaiNNPotential:
    return phx.nn.atomistic.PaiNNPotential(
        UNITS.scale,
        cutoff=CUTOFF,
        feature_count=8,
        interaction_count=2,
        radial_basis_count=6,
        key=jax.random.key(seed),
    )


def molecules(
    kind: str, count: int, *, seed: int, prefix: str
) -> phx.atomistic.AtomisticBatch:
    """Perturbed copies of one molecule, padded to three atom slots."""
    generator = np.random.default_rng(seed)
    if kind == "water":
        numbers = np.array([8, 1, 1])
        masses = np.array([15.999, 1.008, 1.008])
        base = WATER
    else:
        numbers = np.array([1, 1, 0])
        masses = np.array([1.008, 1.008, 1.0])
        base = HYDROGEN
    mask = numbers > 0
    positions = base[None] + 0.05 * generator.normal(size=(count, 3, 3))
    return phx.atomistic.AtomisticBatch(
        np.broadcast_to(numbers, (count, 3)),
        np.where(mask[None, :, None], positions, 0.0),
        np.broadcast_to(masses, (count, 3)),
        UNITS.scale,
        atom_mask=np.broadcast_to(mask, (count, 3)),
        structure_ids=tuple(f"{prefix}-{kind}-{index}" for index in range(count)),
    )


def labeled_source(
    teacher: phx.nn.atomistic.PaiNNPotential,
    execution: phx.atomistic.AtomisticGraphExecutionPlan,
    batch: phx.atomistic.AtomisticBatch,
) -> phx.atomistic.AtomisticStructureSource:
    labels = phx.atomistic.energy_and_forces(teacher, batch, execution)
    return phx.atomistic.atomistic_training_source(
        batch, mass_unit=UNITS.mass_unit, energy=labels.energy, forces=labels.forces
    )


def accumulating_loader(
    *sources: phx.atomistic.AbstractAtomisticStructureSource,
    batch_size: int,
    payload_cases: int,
    weights: tuple[int, ...],
) -> phx.atomistic.AtomisticStructureLoader:
    """Seeded epoch mixture whose global batches pack ``payload_cases`` per payload."""
    canonical = phx.atomistic.atomistic_training_loader(
        *sources, batch_size=batch_size, weights=weights, seed=7
    )
    atom_level = canonical.mixture.schema.atom_level
    rows = canonical.buckets.buckets[0].capacity(atom_level).rows
    bucket = phx.atomistic.AtomisticWorkloadBucket(
        "molecules",
        cases=payload_cases,
        levels=(
            phx.atomistic.AtomisticLevelCapacity(
                atom_level, layout="per-case", rows=rows
            ),
        ),
        max_bytes=1 << 24,
    )
    return phx.atomistic.AtomisticStructureLoader(
        canonical.mixture,
        canonical.bindings,
        phx.atomistic.AtomisticWorkloadBuckets((bucket,)),
        batch_size=batch_size,
        stream="training",
    )


def main() -> None:
    execution = phx.atomistic.AtomisticGraphExecutionPlan(4, maximum_dense_atoms=64)
    teacher = build_model(0)
    water = labeled_source(
        teacher, execution, molecules("water", 6, seed=1, prefix="train")
    )
    hydrogen = labeled_source(
        teacher, execution, molecules("hydrogen", 3, seed=2, prefix="train")
    )
    training = accumulating_loader(
        water, hydrogen, batch_size=4, payload_cases=2, weights=(1, 2)
    )
    validation = phx.atomistic.atomistic_training_loader(
        labeled_source(
            teacher, execution, molecules("water", 3, seed=3, prefix="held-out")
        ),
        stream="validation",
    )
    print(
        "windows per epoch:",
        training.batch_count,
        "payloads in the first window:",
        training.payload_count(0, 0),
    )

    reference = phx.atomistic.fit_atomistic_reference_energies(
        training, species_slot="species", energy_slot="energy", species=(1, 8)
    )
    print(
        "reference energies [eV]:",
        dict(zip(reference.species, np.asarray(reference.energies).tolist())),
        "rank:",
        reference.rank,
    )
    problem = phx.atomistic.AtomisticTrainingProblem(
        training,
        execution,
        (
            phx.atomistic.AtomisticEnergyObjective(reference=reference),
            phx.atomistic.AtomisticForceObjective(
                schedule=phx.atomistic.AtomisticObjectiveSchedule.cosine(
                    10.0, 1.0, steps=12
                )
            ),
        ),
        scale=UNITS.scale,
        cutoff=CUTOFF,
        validation=validation,
    )

    def policy(steps: int) -> phx.atomistic.AtomisticTrainingPolicy:
        return phx.atomistic.AtomisticTrainingPolicy(
            maximum_steps=steps,
            learning_rate=5.0e-3,
            clip_global_norm=10.0,
            target=phx.nn.operator.training.ExponentialMovingAverageTargetPolicy(
                decay=0.9
            ),
            evaluation="target",
            validation_every=4,
        )

    student = build_model(1)
    first = phx.atomistic.fit_atomistic_potential(
        student, problem, policy(8), key=jax.random.key(11)
    )
    if not bool(first.successful):
        raise RuntimeError(f"Training failed: {first.termination}")
    with tempfile.TemporaryDirectory() as directory:
        path = phx.atomistic.write_atomistic_training_checkpoint(
            Path(directory) / "streaming", first, problem, policy(8)
        )
        restored = phx.atomistic.read_atomistic_training_checkpoint(
            path, student, problem, policy(16)
        )
    result = phx.atomistic.fit_atomistic_potential(
        student, problem, policy(16), key=jax.random.key(11), continuation=restored
    )
    if not bool(result.successful):
        raise RuntimeError(f"Training failed: {result.termination}")
    print(
        "fitted scales:",
        dict(zip(result.normalization.names, result.normalization.scales)),
    )
    print("window objective per update:", np.asarray(result.training_values))
    print("validation steps:", np.asarray(result.validation_steps))
    print("validation (EMA view):", np.asarray(result.validation_values))
    print("best step:", result.progress.best_step, "best loss:", float(result.best_loss))
    print("loader cursor:", result.cursor.epoch, result.cursor.batch_index)


if __name__ == "__main__":
    main()
