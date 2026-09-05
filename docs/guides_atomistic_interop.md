# Atomistic interoperability

Interoperability is an explicit host boundary. PhydraX keeps compiled simulation state in
native arrays and converts only immutable plans, metadata, or accepted frames.

## Frames and reporting

`AtomisticFrame` carries positions plus optional velocities, momenta, forces, cell, image
flags, energy, and auxiliary fields. It carries the complete `AtomisticUnitSystem`
descriptor in addition to system, topology, and source identities.
`AtomisticReporterPlan` chooses the cadence and whether output uses the physical
degree-of-freedom domain or the derived interaction-site domain.

H5MD persists the complete unit descriptor once in the stream metadata. Extended
XYZ persists it once in the first PhydraX frame header; later frames carry its
verified content identity. Readers reject legacy ID-only streams. Appends and
reruns require the same complete unit system. Both formats are exposed as
trajectory source/sink plans.

## Rerun

Rerun builds a fresh neighborhood for every accepted input frame. It can rescore several
lambda states and force groups without mutating the trajectory. Use bounded chunks to keep
memory independent of trajectory length; reductions and reporters operate as host-side
consumers.

## ASE structures

`from_ase_atoms(atoms, scale, source_id=...)` copies an optional `ase.Atoms` value
into `AtomicStructure`; `to_ase_atoms(structure)` creates a new detached ASE value.
The scale is mandatory and must be
`AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)`, matching ASE's native units.
Atomic numbers, ordered positions, dalton masses, triclinic cells, per-axis PBC,
stable particle IDs,
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

## OpenMM molar energy boundary

OpenMM energy parameters are `KILOJOULE_PER_MOLE`. Import and export use an
explicit host-only `ENERGY / AMOUNT` to ordinary `ENERGY` conversion with the
unit system's recorded Avogadro constant-set identity. This exceptional semantic
boundary is fingerprinted in the report's complete unit descriptor; it does not
make molar and single-system energies ordinarily convertible.

Multiple OpenMM Fourier components for the same ordered atom quartet share one
topological torsion and are represented by `PeriodicTorsionSeriesPotential`.
Component amplitudes, periodicities, phases, and masks survive native serialization
and OpenMM export; dropping duplicate quartet rows would lose physical energy and
forces.

The neutral `read_pdb_atom_records`/`select_pdb_model` boundary preserves source
record identity for the protein and nucleic-acid applications. Biological chemistry,
alternate-conformer selection, missing-atom completion, and force-field admission
remain explicit application/caller responsibilities rather than parser guesses.

## MDAnalysis

The optional MDAnalysis bridge treats its documented base values as angstrom,
picosecond, angstrom/picosecond, and kJ/(mol·angstrom). Frame import converts
each populated value into the declared physical `AtomisticUnitSystem`, including
the explicit Avogadro force conversion; an uncalibrated reduced system is
rejected. Position export converts back to angstrom. Selection results are
frozen into `AtomisticSelectionPlan`, making an analysis selection auditable and
replayable.

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
