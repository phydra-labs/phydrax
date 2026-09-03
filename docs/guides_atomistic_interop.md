# Atomistic interoperability

Interoperability is an explicit host boundary. PhydraX keeps compiled simulation state in
native arrays and converts only immutable plans, metadata, or accepted frames.

## Frames and reporting

`AtomisticFrame` carries positions plus optional velocities, momenta, forces, cell, image
flags, energy, and auxiliary fields. It also records system, topology, unit-system, and
source identities. `AtomisticReporterPlan` chooses the cadence and whether output uses the
physical degree-of-freedom domain or the derived interaction-site domain.

H5MD is the resumable binary path. A frame becomes visible only after its datasets are
written and the committed-frame counter advances. Extended XYZ is the portable text path.
Both are exposed as trajectory source/sink plans and can feed `AtomisticRerunPlan`.

## Rerun

Rerun builds a fresh neighborhood for every accepted input frame. It can rescore several
lambda states and force groups without mutating the trajectory. Use bounded chunks to keep
memory independent of trajectory length; reductions and reporters operate as host-side
consumers.

## ASE structures

`from_ase_atoms(atoms, scale, source_id=...)` copies an optional `ase.Atoms` value
into `AtomicStructure`; `to_ase_atoms(structure)` creates a new detached ASE value.
The scale is mandatory and must be the exact
`AtomisticScaleContract("angstrom", "electronvolt")` used by ASE. Atomic numbers,
ordered positions, dalton masses, triclinic cells, per-axis PBC, stable particle IDs,
and source identity are audited by the returned
`phydrax.interchange.AdapterReport`.

ASE's zero cell with all PBC flags false is its finite-cell absence representation;
it maps to `cell=None` and `periodic_axes=None` so native nonperiodic system
construction remains nonperiodic. Export reconstructs the zero-cell ASE representation.

Set the ASE array named by `ASE_PARTICLE_ID_ARRAY` to carry stable integer particle
IDs through slicing and reordering. Without it, import uses `AtomicStructure`'s
deterministic order-based IDs and declares that synthesis as an `AdapterLoss`. The
optional `source_id` argument, or the ASE info field named by `ASE_SOURCE_ID_INFO`,
provides source provenance; conflicting values are rejected. Export writes both
reserved fields so a subsequent ASE reorder retains material-atom identity.

ASE velocities, constraints, charges, calculator state, and unrecognized arrays or
info fields are never attached to the native structure. Each permitted omission is
enumerated as declared loss, and `phydrax.interchange.require_lossless(report)` rejects
it when a lossless boundary is required. Partial occupancy or disorder, topology,
spin state, competing unit metadata, dummy atoms, ambiguous particle IDs, inactive
native padding,
and malformed periodic cells are rejected rather than guessed. Calculator objects and
their cached results are neither inspected nor retained.

## MDAnalysis

The optional MDAnalysis bridge converts topology metadata, frames, selections, and
universes. Selection results are frozen into `AtomisticSelectionPlan`, which makes an
analysis selection auditable and replayable.

## i-PI

`IPITransportPlan.unix(...)` and `IPITransportPlan.tcp(...)` configure the i-PI request
state machine. The same transport can expose a PhydraX evaluator to an i-PI driver or wrap
a remote evaluator as `TransportedExternalAtomisticProvider`. Energy, forces, virial,
cell, and protocol status are validated at every transaction.

## PACKMOL

`PackmolAssemblyPlan` combines typed components and spatial regions. The returned assembly
includes component slices, input digest, executable identity, molecule identities, and
final coordinates. Validate minimum separation before promoting an assembly into a
production system.

Optional packages remain lazy imports. `pip install phydrax[atomistic-interop]` installs
ASE, OpenMM, ParmEd, and MDAnalysis. OpenFF Interchange is currently distributed through
its upstream channels and must be installed separately. h5py is a core trajectory
dependency; PACKMOL remains an external executable. None of these boundaries changes
core imports.
