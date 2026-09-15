# Atomistic interoperability

## Resumable trajectory output and rerun

Write accepted frames with a reporter, reopen the H5MD file, and rescore it with the same
prepared potential.

```python
import tempfile
from pathlib import Path

import jax.numpy as jnp
import phydrax as phx

system = phx.atomistic.AtomisticSystemPlan(
    [10, 20],
    [1, 1],
    [1.0, 1.0],
    phx.atomistic.AtomisticUnitSystem.reduced(),
    atom_type_ids=[0, 0],
).prepare()
potential = phx.atomistic.AtomisticPotentialProgram(
    [phx.atomistic.LennardJonesPotential([0.2], [1.0], 2.5)]
).prepare(system)
neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(1).prepare(
    system.particles
)
frame = phx.atomistic.AtomisticFrame(
    0.0,
    0,
    jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]),
    system.plan.particle_ids,
    system_id=system.plan.system_id,
    topology_id=system.topology.topology_id,
    units=system.plan.units,
    source_id="interop-cookbook-frame",
)
with tempfile.TemporaryDirectory(prefix="phydrax-doc-") as directory:
    sink = phx.atomistic.interchange.H5MDTrajectoryPlan(
        Path(directory) / "trajectory.h5"
    )
    with sink.open(append=False) as writer:
        writer.write(frame)
    rerun = phx.atomistic.AtomisticRerunPlan(
        sink,
        potential,
        neighborhood,
    ).run()
    assert bool(rerun.successful)
```

`append=True` resumes at the committed frame boundary; it does not infer simulation state.
The artifact stores its complete unit descriptor once and rejects legacy ID-only
metadata. Resume dynamics from its atomistic checkpoint, then append frames whose
system, topology, and complete unit system match the existing stream.

For analysis selections, convert an MDAnalysis selection once into an
`AtomisticSelectionPlan` and store the stable selected IDs. Do not execute selection strings
inside compiled dynamics.

## Copy a structure from ASE without losing atom identity

Make particle identity explicit before atoms can be sliced or reordered. ASE carries the
ID array with each atom, while the adapter report carries source provenance. This
fragment requires the optional `ase` package and is not part of the core executable
documentation environment.

```text
import numpy as np
from ase import Atoms
import phydrax as phx
from phydrax.units import ANGSTROM, ELECTRONVOLT

scale = phx.atomistic.AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
source = Atoms(
    numbers=[14, 14],
    positions=[[0.0, 0.0, 0.0], [1.35, 1.35, 1.35]],
    masses=[28.085, 28.085],
    cell=[[2.7, 0.0, 0.0], [0.0, 2.7, 0.0], [0.0, 0.0, 2.7]],
    pbc=True,
    info={phx.atomistic.interchange.ASE_SOURCE_ID_INFO: "relaxed-silicon"},
)
source.new_array(
    phx.atomistic.interchange.ASE_PARTICLE_ID_ARRAY,
    np.asarray([1001, 1002], dtype=np.int64),
)

structure, report = phx.atomistic.interchange.from_ase_atoms(source, scale)
phx.interchange.require_lossless(report)

# ASE slicing reorders the reserved ID array with atomic data.
reordered, reordered_report = phx.atomistic.interchange.from_ase_atoms(
    source[[1, 0]], scale
)
phx.interchange.require_lossless(reordered_report)
assert reordered.particle_ids.tolist() == [1002, 1001]

detached, export_report = phx.atomistic.interchange.to_ase_atoms(structure)
phx.interchange.require_lossless(export_report)
assert detached.calc is None
```

If the source does not contain `ASE_PARTICLE_ID_ARRAY`, import deliberately assigns IDs
`0, 1, ...` in the current atom order and reports a synthesized semantic. Carry the
returned report and call `require_lossless` when that default is unacceptable. Velocity,
constraint, charge, calculator, and arbitrary array or info content is likewise never
silently attached to `AtomicStructure`: it is either listed in the report or rejected.
