# Train PaiNN, NequIP, and MACE potentials

The first recipe trains PaiNN or NequIP on a local rMD17 archive of finite molecules.
The second trains a native MACE potential on energy, force, and stress labels of
periodic cells. Neither downloads data or infers units from array shapes.

## Finite molecules from a local rMD17 archive

The archive must contain nuclear charges, coordinates, molecular energies, and
Cartesian forces under one of the field names accepted by `load_rmd17_npz`.

```text
from pathlib import Path

import jax.random as jr
import phydrax as phx
from phydrax.units import ANGSTROM, ELECTRONVOLT

scale = phx.atomistic.AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
dataset = phx.atomistic.load_rmd17_npz(
    Path("/absolute/path/to/rmd17_aspirin.npz"),
    scale=scale,
)
assert dataset.source_energy_unit.symbol == "kcal/mol"
assert dataset.scale.energy_semantics == "single-simulated-system"
split = phx.atomistic.split_rmd17(
    dataset,
    train_size=950,
    validation_size=50,
    test_size=1000,
    seed=0,
)
train_batch, train_energy, train_forces = dataset.take(split.train_indices)
validation_batch, validation_energy, validation_forces = dataset.take(
    split.validation_indices
)
test_batch, test_energy, test_forces = dataset.take(split.test_indices)
```

The loader treats rMD17 energies and forces as kcal/mol source data and performs
the explicit Avogadro conversion to the requested ordinary energy unit. The
dataset records the source length, energy, and dalton mass unit definitions plus
the `codata-2018` constant-set identity; the compiled model still receives raw
homogeneous arrays.

Inspect the molecule size before declaring dense resources. The guard is an
acceptance boundary, not a hint: a batch whose padded atom capacity exceeds it
is rejected. The neighbor limit is also fail-closed and is never implemented by
truncation.

```text
atom_capacity = train_batch.atom_capacity
execution = phx.atomistic.AtomisticGraphExecutionPlan(
    32,
    maximum_dense_atoms=atom_capacity,
)
potential = phx.nn.atomistic.PaiNNPotential(
    scale,
    cutoff=5.0,
    feature_count=128,
    interaction_count=3,
    radial_basis_count=20,
    key=jr.key(10),
)
```

To use degree-zero-through-two Cartesian NequIP without changing the graph,
prediction, training, or result path, replace only the model construction:

```text
potential = phx.nn.atomistic.NequIPPotential(
    scale,
    cutoff=5.0,
    feature_count=32,
    interaction_count=3,
    radial_basis_count=20,
    key=jr.key(10),
)
```

NequIP resolves and resource-checks its legal tensor-product instructions before
allocating layer coefficients and radial outputs. Its radial map has one output
for every multiplicity weight on every legal instruction. It stays a Cartesian
model of degree at most two. Higher degrees and symmetric contractions belong to
MACE (next recipe).

Store each split as an immutable training source and stream it through a
loader. The default fitted scales use only training energies and forces; the
lineage-disjoint validation loader is used for model selection, not
normalization.

```text
def loader(batch, energy, forces, *, stream, batch_size=None):
    source = phx.atomistic.atomistic_training_source(
        batch, mass_unit=dataset.source_mass_unit, energy=energy, forces=forces
    )
    return phx.atomistic.atomistic_training_loader(
        source, batch_size=batch_size, stream=stream
    )


problem = phx.atomistic.AtomisticTrainingProblem(
    loader(train_batch, train_energy, train_forces, stream="training", batch_size=32),
    execution,
    (
        phx.atomistic.AtomisticEnergyObjective(),
        phx.atomistic.AtomisticForceObjective(
            schedule=phx.atomistic.AtomisticObjectiveSchedule.constant(100.0)
        ),
    ),
    scale=scale,
    cutoff=5.0,
    validation=loader(
        validation_batch, validation_energy, validation_forces, stream="validation"
    ),
)
policy = phx.atomistic.AtomisticTrainingPolicy(
    maximum_steps=500,
    learning_rate=1e-3,
    validation_every=10,
    patience=20,
    min_delta=1e-6,
)
result = phx.atomistic.fit_atomistic_potential(
    potential,
    problem,
    policy,
    key=jr.key(11),
)
selected = result.best_potential
```

Each update draws 32 training structures as one accumulation window. For
energy-only or force-only training, store and declare only that target: omit
`forces=` (or `energy=`) from the sources and keep the matching objective.

Evaluate the test split through the conservative prediction surface:

```text
prediction = phx.atomistic.energy_and_forces(selected, test_batch, execution)
if not bool(prediction.valid.all()):
    raise RuntimeError("Test prediction failed neighborhood or finite checks")

atom_count = test_batch.atom_counts
energy_error_per_atom = (prediction.energy - test_energy) / atom_count
force_error = prediction.forces - test_forces
```

`prediction.forces` is the negative derivative of `prediction.energy`; it is not
a separate learned head. Keep `prediction.provenance`, `split.split_id`, the
fitted `result.normalization`, and the resource settings with reported metrics.
rMD17 molecules are finite, so this workflow has no stress labels, and it says
nothing about long-range accuracy or molecular-dynamics stability.

To continue the exact optimizer and selection state to a higher total step
ceiling:

```text
continued_policy = phx.atomistic.AtomisticTrainingPolicy(
    maximum_steps=1_000,
    learning_rate=1e-3,
    validation_every=10,
    patience=20,
    min_delta=1e-6,
)
continued = phx.atomistic.fit_atomistic_potential(
    potential,
    problem,
    continued_policy,
    continuation=result,
)
```

Only `maximum_steps` (and execution-only staging) may change in the policy. The
graph execution plan may be re-planned, for example with a grown capacity; the
loader buckets and placement are bound by the result's loader cursor. A changed
optimizer, objective, weighting, validation cadence, patience, delta, or
selection policy is rejected rather than silently starting a different run. The
result's `training_state` is the committed training-kernel state (optimizer
moments, root key, and cursors); the continuation resumes it and ignores a newly
supplied `key`.

To continue in a fresh process, persist the result without pickle and restore it
against the same rebuilt problem:

```text
phx.atomistic.write_atomistic_training_restart(
    "aspirin-restart", result, problem, policy
)
artifact, restored = phx.atomistic.read_atomistic_training_restart(
    "aspirin-restart", problem, continued_policy
)
continued = phx.atomistic.fit_atomistic_potential(
    artifact.model, problem, continued_policy, continuation=restored
)
```

## Periodic MACE with energy, force, and stress labels

This recipe is `examples/atomistic_mace.py`. A fixed teacher MACE labels four
strained triclinic water cells through the native E/F/S prediction route. A
differently initialized student is then fitted on all three label kinds. Both
models are random native initializations, so the recipe demonstrates the
workflow, not accuracy.

```python
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
    print("fitted scales:", dict(zip(result.normalization.names, result.normalization.scales)))
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


main()
```

The cells are smaller than twice the cutoff, so a minimum-image graph would not be
a valid assumption. Each topology instead holds every directed image route within
the cutoff. This 3.0 Å cutoff is shorter than every lattice translation, so these
cells have no self-image routes; a cutoff longer than a lattice translation would
add them. The plan's `search_cell` and `search_margin` declare one reusable
search envelope that covers every strained cell. Stress labels are the
tensile-positive `(1/V) dE/d strain` of each row cell deformed as `H @ F.T`, in
eV/Å³. To supervise stress on only the periodic cases of a mixed finite/periodic
batch, pass `stress_available` to `atomistic_training_source`; an available
stress label on a case without a periodic axis and a finite nonsingular cell is
refused. Each payload's graph layout and topology are prepared once on the host
and never rebuilt during an optimizer step, so a capacity overflow ends the run
with `NEIGHBOR_OVERFLOW` at the last committed update instead of silently
changing the graph.

Persist the trained model with
`phx.atomistic.write_atomistic_model_artifact(path, result.potential)`. The archive
reloads without pickle or any provider package. The
[native MACE execution guide](../guides_mace_execution.md) covers prepared and
tabulated inference, accelerated coupling, and the unreleased status of every MACE
route.

## Streamed minibatches, accumulation, and EMA selection

`examples/atomistic_streaming_training.py` mixes two immutable sources (the
hydrogen source drawn twice per epoch), packs every four-draw global batch as
two-case payloads, fits per-species reference energies, schedules the force
weight with a cosine over committed updates, clips the merged gradient, and
selects the EMA target view on a lineage-disjoint validation loader. A
checkpoint written after eight updates continues to sixteen. The teacher and
student are random PaiNN initializations, so the recipe demonstrates mechanics,
not accuracy.

```text
reference = phx.atomistic.fit_atomistic_reference_energies(
    training, species_slot="species", energy_slot="energy", species=(1, 8)
)
problem = phx.atomistic.AtomisticTrainingProblem(
    training,
    execution,
    (
        phx.atomistic.AtomisticEnergyObjective(reference=reference),
        phx.atomistic.AtomisticForceObjective(
            schedule=phx.atomistic.AtomisticObjectiveSchedule.cosine(10.0, 1.0, steps=12)
        ),
    ),
    scale=units.scale,
    cutoff=2.5,
    validation=validation,
)
policy = phx.atomistic.AtomisticTrainingPolicy(
    maximum_steps=16,
    learning_rate=5e-3,
    clip_global_norm=10.0,
    target=phx.nn.operator.training.ExponentialMovingAverageTargetPolicy(decay=0.9),
    evaluation="target",
    validation_every=4,
)
```

The accumulating `training` loader and its bucket are shown in the
[training-data guide](../guides_atomistic_training_data.md#mixtures-loaders-and-workload-buckets).
