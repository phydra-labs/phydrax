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

## Native model deployment status

The native MACE model, its conversion, persistence, and ASE/i-PI/IREE deployment
boundaries described below are unreleased candidates. Their capability profiles are
registered as candidates, not as released support, and no independent signed release
has been obtained. A catalog entry, a passing targeted test, or a successful example run
is not release qualification. External speed, capacity, or scaling numbers reported for
other MACE implementations are not Phydrax results.

## Native model artifacts

`phydrax.atomistic.write_atomistic_model_artifact(path, model, source=..., licenses=...)`
validates a native model through its registered domain validator and atomically writes
a pickle-free array artifact. `read_atomistic_model_artifact(path,
numeric_revision_id=...)` restores, validates, and identity-checks it; pinning
`numeric_revision_id` refuses any other parameter revision. Reading imports no optional
provider package, so a converted MACE model is evaluated, trained, served, or exported
without mace-torch, e3nn, or torch. `source` binds `MACESourceProvenance` from a
conversion (species, head, and cutoff must agree with it); `licenses` records the
model's rights identifiers, and redistribution permission is never inferred. The
artifact and training-restart APIs are documented with the atomistic model surface in
[Atomistic learning and dynamics](api/atomistic.md).

## MACE checkpoint conversion

`convert_mace_checkpoint(source, provider=..., head=...)` converts one admitted
standard mace-torch source into a native `phydrax.nn.atomistic.MACEPotential`.
Native evaluation and the caller's conversion process never import mace-torch,
e3nn, or torch; only the explicitly pinned provider subprocess imports them.
No checkpoint is downloaded implicitly. Conversion has three explicit inputs:

- **Source bytes.** The caller admits a local file through the canonical external-artifact
  policy (`phydrax.artifacts.ExternalArtifactPolicy`, `ArtifactManifest`, and
  `admit_external_artifact`): an exact out-of-band SHA-256, byte size and maximum byte
  bound, allowed license identifiers, and allowed suffixes. `MACESource(artifact,
  manifest, policy, kind, trust=..., architecture=...)` carries the admitted artifact and
  its loading contract.
- **Provider runtime.** `MACEProviderRuntime(interpreter, site_packages, torch_version,
  cuequivariance_version=None)` names a separately installed interpreter pinned with
  `phydrax.interchange.pin_executable` and its exact site-packages directory. mace-torch
  and e3nn must be the admitted `MACE_PROVIDER_RELEASES` (mace-torch 0.3.16, e3nn
  0.4.4); their installed files are verified against their installed RECORD digests on every
  run. The torch release is caller-declared, version-checked, and recorded with its
  RECORD digest; the campaign provider environment used torch 2.14.1.
  `cuequivariance_version` is declared exactly when the optional cuequivariance packages
  that construct reduced generalized-CG bases are installed, and reduced-CG sources are
  identified and rebuilt only with it.
- **Trust.** A `"torch-state-dict"` source is loaded with `torch.load(weights_only=True)`
  and requires the exact declared provider architecture (`architecture=`). A
  `"torch-full-model"` source is an executable pickle; it is deserialized only when the
  caller constructs `TrustedTorchPickleSource(sha256, statement)` for exactly the admitted
  digest with a nonempty statement of why it is trusted. A refused or failed safe load
  never falls back to an unsafe one.

The provider runs in a subprocess with a fixed isolated bootstrap and explicit
`MACEConversionLimits` for source/result bytes, per-tensor elements and bytes,
aggregate elements, archive members, manifest/log bytes, and timeout. Safe
state-dict storage, views, tensor metadata, declared irreps extents, provider
construction workspace, and model parameters are admitted before their owning
deserialization or allocation. Evaluation and gradient requests conservatively
admit edge/image topology and result tensors before provider neighborhood
discovery. Returned metadata is checked before payload conversion, and the
provider streams a bounded pickle-free array archive.

These checks bound the admitted standard provider path; the subprocess is still
**not a security or arbitrary-memory sandbox**. A trusted full-object pickle
executes with the caller's privileges, and its own deserialization may allocate
before the recovered standard model can be checked and rebuilt under the
construction limits.

The provider refuses non-standard module trees, normalizations, layouts, or inconsistent
buffers with `MACESourceRefusedError` before any native model exists, and proves that the
extracted architecture declaration rebuilds a provider model reproducing the source.
`head` selects the active readout head and is required for multihead sources. The
native model keeps the original source parameterization: e3nn weights, symmetric
contraction weights `W`, and the provider's fixed `U` basis as non-trainable leaves; it is
not a merged-polynomial reparameterization, so `MACECheckpointConversion.reconstruct`
maps source parameter directions into native parameter space. Source energies are eV and
lengths are angstrom.

`MACECheckpointConversion.provenance` (`MACESourceProvenance`) records the source digest,
size, license, manifest and admission identities, kind, whether a trusted full object was
executed, source dtype, provider release record, interpreter digest and version,
declaration, head, `U`-basis residual, layout normalization, and conversion identity.
It is serialized into the native artifact as evidence of origin, not as a native schema
generation. `layout_normalization` records how a source saved by an earlier mace-torch
release is expressed in the admitted release's exact layout; every entry is false or
empty for safe state-dict sources already in that layout. After conversion, write the
model with `write_atomistic_model_artifact(..., source=conversion.provenance)`; later
inference, training, and fresh restarts read the native artifact and need no provider.

`evaluate_mace_source` evaluates the admitted source with the provider's own neighbor
list (`MACEProviderConfiguration` in, `MACEProviderCase` energy/forces/stress out, stress
as `(1/V) dE/d(symmetric strain)` in eV/Å³) and never touches native execution; it is the
independent source-side oracle for fidelity comparisons. `mace_source_gradients` returns
provider parameter gradients in the original source parameterization.
`create_mace_provider_fixture` builds a deterministic provider-initialized fixture under
a declared architecture and seed. A fixture is a lawful test oracle; it does not qualify
any named external checkpoint.

Normalization of legacy foundation checkpoints and their restart/source-consumer
coverage are still being completed. No foundation-model row is claimed to have passed a
fidelity campaign.

### Model rights

Model weights and provider code are licensed separately, and code licenses do not
relicense weights. The mace-torch and e3nn code is MIT-licensed; MACE-MP/MPA foundation
weights are MIT-licensed, while OMAT, MH, and MACE-OFF weights are distributed under the
Academic Software License (ASL). See the
[mace-foundations README](https://github.com/ACEsuit/mace-foundations) and the
[mace-off README](https://github.com/ACEsuit/mace-off). Phydrax bundles no checkpoint
weights. MIT foundation weights were downloaded locally, with matching SHA-256 digests,
only for conversion campaigns; ASL-licensed OMAT, OFF, and MH weights have not been
downloaded because their rights are not admitted. Record each converted model's rights
identifiers with `licenses=` when writing its artifact.

## Native ASE calculator

`NativeASECalculatorPlan(provider_plan, units, coordinate_dtype="float64")` binds one
`NativeAtomisticProviderPlan` (model, graph execution, finite neighborhood, skin, and
deformation margin) to an SI-convertible `AtomisticUnitSystem` with the model's scale.
`NativeASECalculator(plan, artifact_id=...)` is an ASE calculator over that recipe. Both
names resolve lazily from `phydrax.atomistic.interchange` and require ASE when accessed;
importing Phydrax does not.

ASE positions and cells in angstrom and masses in dalton are converted exactly into the
native units. Periodic ASE `pbc` axes become periodic cell axes; a structure with no
periodic axis is finite regardless of its ASE box. Results use ASE units:

| Property | Meaning |
|---|---|
| `energy`, `free_energy` | The same potential energy in eV; a classical surface has no electronic entropy. |
| `energies` | Per-atom energies in eV, summing to `energy`. |
| `forces` | Conservative negative energy gradient in eV/Å. |
| `stress` | Tensile stress `(1/V) dE/d(strain)` in eV/Å³ as an ASE Voigt vector `(xx, yy, zz, yz, xz, xy)`; positive under tension, matching ASE's sign convention. |

Stress is available only for fully periodic 3D cells whose program terms all own cell
derivatives. Requesting it otherwise raises ASE's `PropertyNotImplementedError`; no zero
is substituted.

ASE invalidates results on position, atomic-number, cell, PBC, charge, or magnetic-moment
changes. Atomic-number or PBC changes re-prepare the native system and provider; position
and cell changes advance the cached Verlet state through its certificate. A failed
evaluation is retried once from a fresh preparation at the current geometry before
`CalculationFailed` is raised with the neighborhood capacity-failure flag.
`update_plan(plan)` binds a new model revision or preparation and drops every cache. The
calculator's `provenance` records the plan, model revision, artifact, provider,
neighborhood epoch, and preparation count of the latest result.

`phydrax.chemistry.interchange.ASECalculatorProvider` is the opposite boundary: it
evaluates an external ASE calculator as an electronic-structure provider (see
[Chemistry interoperability](guides_chemistry_interop.md)).

## i-PI

`IPITransportPlan.unix(path, ...)` and `IPITransportPlan.tcp(host, port, ...)` configure
one i-PI link; `listen()` returns a listener whose `accept()` yields an `IPISession`, and
`connect()` yields the peer session. `timeout`, `maximum_atoms`, and
`maximum_extra_bytes` bound every transaction, and partial socket reads are assembled
exactly. `IPITransportStatus` reports `READY`, `HAVE_DATA`, `CLOSED`, `PROTOCOL_ERROR`, or
`PROVIDER_ERROR`.

The wire carries little-endian float64 payloads in Hartree atomic units: bohr, hartree,
and hartree/bohr. The system's unit scale must therefore be SI-convertible (for example
`AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()`); reduced units are
refused. Conversions are exact unit-registry factors.

Cells travel as `h`, whose **columns** are lattice vectors, in C order, followed by
`inv(h)`. Phydrax lattice vectors are rows, so `h = cell_vectors.T`. Only fully periodic
3D cells or the aperiodic all-zero cell and inverse are carried. `inverse_cell="verify"`
(default) requires the received inverse to equal `inv(h)`; `"ignore"` treats `h` as the
sole geometry for peers, such as ASE's socket server, that send a transposed legacy
inverse.

Native records hold only the tensile stress `σ = (1/V) dE/d(strain)` of
`ExternalAtomisticEvaluation`. The force reply carries the configurational virial
`W = -dE/d(strain) = -V σ` in hartree, transmitted as `W.T` in C order; a received
virial is converted back to tensile stress, never stored as stress. The virial exists
only on the wire.

- `virial="required"` (default) refuses a transaction whose virial is unavailable: a
  serving provider must return stress for a fully periodic cell, and a received virial
  must be finite. A finite system has no stress, so it cannot be served under this
  policy.
- `virial="optional"` transmits an all-NaN virial when stress is unavailable and maps a
  received all-NaN virial to `stress=None`.

No policy substitutes a zero virial, and a partially non-finite virial is always refused.

`serve_ipi_once(session, provider, system)` acts as the i-PI driver until one force reply
is sent (`READY`) or the peer exits (`CLOSED`). A periodic system evaluates the received
cell; an aperiodic system ignores the transmitted box. The provider may be any
`AbstractExternalAtomisticProvider`. `NativeAtomisticProviderPlan(...).prepare(system)`
returns a `NativeAtomisticProvider` that serves a learned native model: energy, forces,
and stress come from one evaluation of the same scalar energy, with no second energy or
force loop, and stress is available exactly when the system is a fully periodic 3D cell
and every program term owns cell derivatives. `CallableBornOppenheimerProvider` serves
an arbitrary host evaluator. In the other direction,
`TransportedExternalAtomisticProvider(session, provider_id)` wraps a remote i-PI driver
as a host, nondifferentiable provider.

## Frozen IREE deployment

`phydrax.export.save_atomistic_iree` compiles a native provider recipe at one reference
geometry into a fixed-capacity energy/force/stress executable. Neighbor discovery and
cache lifecycles stay on the host: the module freezes the candidate graph of one
lifecycle epoch, takes only positions and, for periodic systems, cell vectors and lattice
image counts, and returns energy, forces, per-atom energy, optional stress, a traced
status, and the frozen route count. The loaded executable is digest- and
contract-pinned, host-only, and nondifferentiable; training and neighbor lifecycles are
not exported. A float64 model needs
`IREEExportPolicy(executable_format="system-library")`, a host-specific executable.
See [Export](api/export.md#frozen-atomistic-energy-force-and-stress) for the ABI. Numerical
parity of exported float64 MACE executables is not claimed.

## PACKMOL

`PackmolAssemblyPlan` combines typed components and spatial regions. The returned assembly
includes component slices, input digest, executable identity, molecule identities, and
final coordinates. Validate minimum separation before promoting an assembly into a
production system.

Optional packages remain lazy imports. `pip install phydrax[atomistic-interop]` installs
ASE, OpenMM, ParmEd, and MDAnalysis. OpenFF Interchange is currently distributed through
its upstream channels and must be installed separately. h5py is a core trajectory
dependency; PACKMOL remains an external executable. The MACE source provider (mace-torch,
e3nn, torch) is never a Phydrax dependency; install it in a separate interpreter and pin
that interpreter explicitly. None of these boundaries changes core imports.
