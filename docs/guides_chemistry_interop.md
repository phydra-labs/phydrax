# Computational chemistry interoperability

External chemistry tools are optional host providers. Native structures, stable
IDs, electronic sectors, physical model plans, numerical plans, units, tasks,
results, checkpoints, lifecycle records, and qualification identities remain
authoritative. Selecting an installed package never changes the requested
physics.

## Governed basis artifacts

`import_basis_set_exchange` imports an explicitly named basis only after the
caller supplies a `ReferenceArtifactManifest` with checksum, size, license,
commercial-use, redistribution, training-use, export, and lineage declarations.
The resulting `GaussianBasisImport` binds the parsed record to the manifest and
can construct `GaussianBasisPlan` shells for stable particle IDs. The importer
never treats a package label as artifact provenance.

## QCSchema and QCEngine

`electronic_calculation_to_qcschema` converts the active geometry to bohr,
preserves stable particle IDs in extras, and maps the concrete method, basis,
sector, and typed task to one driver. `electronic_evaluation_from_qcschema`
converts atomic units to the calculation unit system and maps gradients to
negative forces. Missing IDs are a declared adapter loss; changed order or IDs
are rejected.

`QCEngineProvider` binds one explicit program and model-chemistry identity. It
does not search for an engine. Its finite molecular route supplies energy and
forces; Hessians use the native force-difference workflow unless an exact
analytic-Hessian provider is selected.

## PySCF molecular providers

`PySCFProvider` supports finite vacuum HF/KS ground-state energy, force, and
dipole tasks for its declared RHF/UHF/ROHF/RKS/UKS/ROKS coordinates. Spin
multiplicity maps to alpha minus beta electron count. An unconverged SCF result,
unsupported environment/correction, or undeclared task is rejected.

`PySCFCoupledClusterProvider` operates on a native restricted MO integral store.
It returns CCSD/CCSD(T) correlation and triples energies, right and Lambda
amplitudes, residuals, iterations, and provider/plan/store identities.
`CoupledClusterCheckpoint` can restart only that exact identity.

Analytic molecular CC gradients need derivative integrals and a real molecular
geometry, which an MO store does not contain.
`PySCFMolecularCoupledClusterGradientProvider` therefore binds a caller-defined,
content-identified molecule builder. It verifies that the builder returns the
requested bohr geometry and executes RHF/UHF/ROHF CCSD or CCSD(T) with the
matching Lambda-gradient route. The builder definition is part of the plan ID.

## Active-space and excited-state providers

`AbstractActiveSpaceSolver` is the exact boundary for selected-CI, DMRG, and
FCIQMC. Results must include energies, one- and two-particle densities,
residuals, discarded weights, solver kind, and bound plan/store identity.

`AbstractCorrelatedManifoldProvider` covers ADC(2), ADC(2)-x, ADC(3), EOM-CCSD
EE/IP/EA/spin-flip, and provider-computed CAS manifolds. Right/left amplitudes are
represented by `BiorthogonalStateRepresentation`; a provider must not place them
in an orthonormal TDA representation. Correlated gradients, transition
properties, and nonadiabatic couplings use explicit derivative provider
boundaries.

## Periodic providers

Periodic work uses the same `ElectronicCalculationPlan`, typed task plans, and
`CallableElectronicProvider` boundary as finite systems. The calculation binds
the exact method definition IDs, basis identity, electronic sector, numerical
plan, and periodic cell. Provider capabilities must admit the cell rank and each
requested observable. `make_electronic_evaluation` returns
`ElectronicPeriodicEvaluation`, checks that every requested periodic field is
present, binds provider/task/geometry identities, and retains governed provider
artifacts in `header.artifact_ids`. Diagonal GW and supplied-kernel BSE are
separate provenance-retaining postprocessors; neither is presented as a
first-principles provider implementation.

## ASE calculators

`ASECalculatorProvider` creates a fresh detached `ase.Atoms` for every
evaluation. An `ASEElectronicStateBinding` either writes declared charge/spin
fields or explicitly limits a calculator to state-invariant neutral singlets.
ASE units are eV, Angstrom, dalton, and elementary charge. Missing requested
properties fail capability admission.

## Executable isolation

Executable providers must use `PinnedExecutable` and `run_energy_command`: exact
binary digest, argv without a shell, private working directory, bounded I/O,
timeout, process-group cleanup, and detached artifacts. This operational
isolation is not a security sandbox; provider inputs remain trusted executable
content.
