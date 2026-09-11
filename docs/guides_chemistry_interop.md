# Computational chemistry interoperability

External chemistry tools are optional host providers. Native structures, state,
model chemistry, units, results, and workflow identity remain authoritative.
Optional modules are imported only after the caller selects their provider.

## QCSchema and QCEngine

`electronic_calculation_to_qcschema` produces a QCSchema AtomicInput mapping.
It converts active coordinates to bohr, masses to dalton, fixes center-of-mass
and orientation transformations, and stores native stable particle IDs in
extras. Method and basis come from model chemistry; the requested energy,
gradient, or Hessian driver comes from `ElectronicPropertyRequest`.

`electronic_evaluation_from_qcschema` converts atomic units to the calculation's
native unit system and maps gradients to negative forces. Missing stable IDs are
a declared adapter loss. Changed IDs/order are rejected.

`QCEngineProvider` requires an explicit program, one exact model-chemistry ID,
and caller-declared provider capabilities. It never selects an installed
program automatically. QCEngine and the selected engine remain separate
provenance identities.
The initial QCEngine provider is finite and nonperiodic and supplies energy and
force evaluations; molecular Hessians use the native force-difference workflow.

## PySCF

`PySCFProvider` supports finite nonperiodic RHF, UHF, ROHF, RKS, UKS, and ROKS
energy/force calculations where the installed PySCF method supplies gradients.
It maps multiplicity to the PySCF spin value `alpha electrons - beta electrons`,
uses explicit bohr/Hartree source units, and refuses an unconverged SCF result.

The initial adapter delegates Hessians to the native finite-difference force
workflow. It supports the vacuum environment only.

## ASE calculators

`ASECalculatorProvider` accepts a calculator factory. Each evaluation receives a
fresh detached `ase.Atoms`; no caller-owned atoms or calculator cache becomes
native state.
The provider also binds one exact `model_chemistry_id`; preparation rejects a
calculation that would otherwise reuse the calculator for different physics.

ASE has no universal electronic-state protocol. The caller must supply an
`ASEElectronicStateBinding` that either:

- writes total charge and spin to named `Atoms.info` fields with declared spin
  semantics; or
- explicitly declares a state-invariant neutral-singlet calculator. That
  binding rejects charged and open-shell calculations.

ASE values are interpreted in eV, Angstrom, dalton, and elementary charge. A
calculator that lacks a requested property is rejected through provider
capabilities before execution.

Foundation atomistic models, including fairchem calculators, should normally
enter through this boundary. Model task, checkpoint, charge/spin conditioning,
and use rights remain part of the declared model/provider identity.

## External executable security

Future executable providers such as QUICK must use
`phydrax.interchange.energy_runtime.PinnedExecutable` and
`run_energy_command`: exact executable digest, argv without a shell, private
working directory, bounded inputs/outputs, timeout, process-group cleanup, and
detached artifacts. This operational isolation is not a security sandbox; input
model files remain trusted executable content.
