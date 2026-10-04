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

Construct a typed joint problem. Each split freezes a candidate graph topology
prepared on the host from `cutoff`. The default fitted scales use only training
energy and forces. The validation values are used for model selection, not
normalization.

```text
problem = phx.atomistic.AtomisticTrainingProblem(
    train_batch,
    execution,
    cutoff=5.0,
    training_energy=train_energy,
    training_forces=train_forces,
    validation_batch=validation_batch,
    validation_energy=validation_energy,
    validation_forces=validation_forces,
)
policy = phx.atomistic.AtomisticTrainingPolicy(
    maximum_steps=500,
    learning_rate=1e-3,
    energy_weight=1.0,
    force_weight=100.0,
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

For energy-only training, omit the force targets and set `force_weight=0.0`.
For force-only training, omit molecular energies and set `energy_weight=0.0`.
At least one available target kind must have positive weight.

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
    energy_weight=1.0,
    force_weight=100.0,
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

Only `maximum_steps` may change across that continuation. A changed optimizer,
loss scale or weight, validation cadence, patience, delta, or selection policy
is rejected rather than silently starting a different run. The result's
`training_state` is the committed training-kernel state (Adam moments, root key,
and cursors); the continuation resumes it and ignores a newly supplied `key`.

To continue in a fresh process, persist the result without pickle and restore it
against the same rebuilt problem:

```text
phx.atomistic.write_atomistic_training_restart("aspirin-restart", result, policy)
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
        atomic_energies=jnp.asarray([[-1.0, -2.0]], dtype=jnp.float64),
        key=jax.random.key(seed),
    )


def strained_batch() -> phx.atomistic.AtomisticBatch:
    structures = []
    for index, magnitude in enumerate((-0.02, 0.0, 0.015, 0.03)):
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
)
batch = strained_batch()
labels = phx.atomistic.energy_and_forces(build_model(0), batch, execution, compute_stress=True)
if not bool(jnp.all(labels.valid)):
    raise RuntimeError("Teacher labels failed.")
problem = phx.atomistic.AtomisticTrainingProblem(
    batch,
    execution,
    cutoff=3.0,
    training_energy=labels.energy,
    training_forces=labels.forces,
    training_stress=labels.stress,
)
policy = phx.atomistic.AtomisticTrainingPolicy(maximum_steps=25, learning_rate=1.0e-2)
result = phx.atomistic.fit_atomistic_potential(
    build_model(1), problem, policy, key=jax.random.key(7)
)
if not bool(result.successful):
    raise RuntimeError(f"Training failed: {result.termination}")
trained = phx.atomistic.energy_and_forces(
    result.potential, batch, execution, compute_stress=True
)
print("stress loss history:", np.asarray(result.stress_loss_history))
print("trained stress error [eV/Ang^3]:", np.asarray(trained.stress - labels.stress)[0])
```

The cells are smaller than twice the cutoff, so a minimum-image graph would not be
a valid assumption. Each topology instead holds every directed image route within
the cutoff. This 3.0 Å cutoff is shorter than every lattice translation, so these
cells have no self-image routes; a cutoff longer than a lattice translation would
add them. Stress labels are the tensile-positive
`(1/V) dE/d strain` of each row cell deformed as `H @ F.T`, in eV/Å³. To
supervise stress on only the periodic cases of a mixed finite/periodic batch,
pass `training_stress_mask`. Each split's topology is prepared once on the host
and never rebuilt during an optimizer step, so a capacity overflow fails the
attempt with `NONFINITE` instead of silently changing the graph.

Persist the trained model with
`phx.atomistic.write_atomistic_model_artifact(path, result.potential)`. The archive
reloads without pickle or any provider package. The
[native MACE execution guide](../guides_mace_execution.md) covers prepared and
tabulated inference, accelerated coupling, and the unreleased status of every MACE
route.
