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
OpenMM, ParmEd, and MDAnalysis. OpenFF Interchange is currently distributed through its
upstream channels and must be installed separately. h5py is a core trajectory dependency;
PACKMOL remains an external executable. None of these boundaries changes core imports.
