# Molecular computational chemistry

`phydrax.chemistry` owns finite-molecule electronic-state, model-chemistry,
provider, property, stationary-point, thermochemistry, and spectroscopy
contracts. It reuses `phydrax.atomistic.AtomicStructure` and
`AtomisticSystemPlan`; it does not introduce another molecule or atom system.

The initial execution profile is finite and nonperiodic. Conventional electronic
structure is supplied by explicit host providers. PhydraX owns the scientific
plan, units, result validation, downstream workflows, lifecycle records, and
qualification evidence.

## State, model, provider, and task are separate

`MolecularElectronicStatePlan` carries total molecular charge, spin
multiplicity, and root identity. Preparation derives the electron count and
alpha/beta populations from active nuclear charges. Classical per-site charges
in `AtomisticSystemPlan` are never interpreted as electronic charge.

`ElectronicModelChemistryPlan` identifies the physical approximation: method,
reference, basis, corrections, environment, relativistic semantics, and any
governed model artifact. It deliberately excludes the program that implements
the approximation and the algorithm that consumes its results.

```text
state = phx.chemistry.MolecularElectronicStatePlan(0, 1)
method = phx.chemistry.ElectronicMethodPlan(
    phx.chemistry.ElectronicMethodFamily.HARTREE_FOCK,
    "hf",
    phx.chemistry.ElectronicReferenceKind.RESTRICTED,
)
model = phx.chemistry.ElectronicModelChemistryPlan(
    method,
    basis=phx.chemistry.BasisSetReference("sto-3g", "provider-library"),
)
request = phx.chemistry.ElectronicPropertyRequest.energy_and_forces()
calculation = phx.chemistry.ElectronicCalculationPlan(
    system, state, model, request
)
prepared = calculation.prepare(phx.chemistry.interchange.PySCFProvider())
```

A provider declares exact properties, reference kinds, periodicity, state,
execution, concurrency, differentiability, and conservation capabilities.
Preparation rejects a mismatch before invoking the provider. There is no
provider search or mid-run fallback.

## Units and signs

Electronic results use the `AtomisticUnitSystem` bound to the system. Energy is
ordinary single-system energy. Molar energy remains dimensionally distinct and
crosses an explicit Avogadro boundary only through
`single_system_energy_to_molar_factor` or
`molar_energy_to_single_system_factor`.

The canonical force convention is the negative coordinate gradient:

```text
force = -d energy / d position
```

QCSchema gradients are negated during import. Hessians have the exact derived
unit energy/length squared. Dipoles use charge times length. Spectroscopic
wavenumbers are reported with `INVERSE_CENTIMETER`.

## Potential-energy surfaces

Geometry-dependent workflows consume
`AbstractPreparedPotentialEnergySurface`, not a provider-specific object.
Adapters expose:

- a prepared electronic calculation;
- a prepared native atomistic potential and neighborhood;
- the existing external atomistic provider;
- an explicit weighted sum of compatible surfaces.

Every component in a composite surface must use the same system and unit
identity. A failed component fails the complete evaluation.

`SurfaceExternalAtomisticProvider` adapts a qualified chemistry surface into the
existing Born–Oppenheimer atomistic dynamics boundary. It preserves the
surface's conservative and differentiability declarations and does not invent
stress.

## Geometry optimization

`MolecularGeometryOptimizationPlan` optimizes mobile Cartesian atoms using
provider-supplied forces. The shared `MinimizationProblem` supports a fused
`explicit-host` value/gradient evaluator; `SciPyMinimize` consumes it without
asking JAX to differentiate through an external process. Native compiled
optimizers reject that derivative mode before tracing.

Success requires the optimizer, final provider evaluation, maximum-force and
RMS-force criteria, and evaluation budget to pass. The result contains a new
`AtomicStructure` with unchanged stable IDs, masses, masks, cell, and name.

The initial route does not provide molecular internal coordinates, periodic cell
optimization, or automatic symmetry constraints.

## Hessians and normal modes

`MolecularHessianPlan` consumes an analytic surface Hessian when declared;
otherwise it performs a central finite difference of forces. Displacements and
task IDs are deterministic. An explicit `HostTaskExecutor` may be supplied.
Failed displacement calculations invalidate the whole Hessian. Raw and
symmetrized Hessians are both retained with an antisymmetry residual.

`VibrationalAnalysisPlan`:

1. mass-weights the active Hessian;
2. constructs translations and infinitesimal rotations about the center of mass;
3. detects atomic, linear, or nonlinear external rank;
4. removes 3, 5, or 6 external modes;
5. solves the reduced self-adjoint eigenproblem with `phydrax.linalg`;
6. reports signed frequencies, cm^-1 wavenumbers, Cartesian modes, reduced
   masses, projector residuals, and stationary-point class.

A minimum has no significant imaginary mode. A first-order saddle has exactly
one. Raw negative curvatures are never deleted.

## Thermochemistry and infrared spectra

`HarmonicThermochemistryPlan` implements finite-molecule ideal-gas RRHO with
explicit temperature, pressure, rotational symmetry number, and electronic
degeneracy. It accepts only a qualified minimum or first-order saddle. For a
saddle, exactly one classified imaginary reaction mode is excluded. The result
separates electronic, translational, rotational, and vibrational contributions,
ZPE, internal energy, enthalpy, entropy, and Gibbs energy.

The native result is per system. `to_molar_thermochemistry` creates a separate
molar result in a caller-selected energy/amount unit.

`IRSpectrumPlan` obtains Cartesian dipole derivatives by central finite
differences of a prepared dipole-capable electronic calculation, projects them
onto qualified normal modes, and reports line strengths. Gaussian broadening is
a separate explicit grid transformation. The initial line strengths use the
native charge-squared/mass unit; they are not mislabeled as conventional
km/mol intensities.

## Failure semantics and current non-claims

A result is successful only when the provider converged, every requested field
is present and finite, units convert, and stable particle order is preserved.
Provider exceptions are not numerical penalties.

The initial package does not claim:

- periodic electronic structure;
- native AO integral, SCF, DFT, or post-HF execution;
- ECP or relativistic support;
- internal-coordinate optimization;
- constrained vibrational analysis;
- Raman or UV-visible spectra;
- excited-state manifolds or nonadiabatic couplings;
- reaction-path, transition-state search, IRC, or QM/MM;
- automatic molecular identity or reaction inference.
