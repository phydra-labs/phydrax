# Semiconductor devices

`phydrax.applications.semiconductor` provides a bounded family of classical,
electrothermal, quantum, inference, and qualification models on native Phydrax
substrates. Every constitutive law declares its physical domain and provenance.
Implemented or numerically verified behavior is not a calibrated foundry-process
claim; that status requires authorized measurements and independent evidence for
one exact process revision, structure population, and operating envelope.

## Physics, state, and sign conventions

The elementary-charge magnitude `q` is positive. Electrons carry `−q`, holes `+q`,
and `E = −grad(psi)`. A support edge runs from `tail` to `head`; positive
particle-number flux transfers particles in that direction. Terminal current is
positive **into the device** and includes displacement current when a state rate is
supplied.

`DevicePlan.layout` names every solver field. The default homojunction reduction
retains the familiar node-major shape:

- `potential = psi / Vt_ref`;
- `electron = EFn / (k Tref)`;
- `hole = EFp / (k Tref)`.

Electrothermal plans add logarithmic lattice-temperature and optional
electron/hole-temperature charts. Thermionic material interfaces add distinct
material-sided algebraic Fermi traces. Dynamic traps add logit charts, while their
actual storage is area or volume times trap density times occupancy. These charts
are not conserved quantities: carrier storage is particle count and thermal
storage is extensive internal energy. Reference scales remain fixed when
temperature changes.

`SemiconductorMaterial` has two deliberately distinct thermodynamic modes:

- `intrinsic_density=...` is the restricted nondegenerate, constant-band
  homojunction reduction with `n = ni exp(psi/Vt + EFn/kT)` and the corresponding
  hole relation;
- `thermodynamics=BandThermodynamics(...)` supplies independently measured or
  assumed `Nc`, `Nv`, aligned `Ec`, `Ev`, a named energy datum, Boltzmann or
  Fermi–Dirac statistics, compressibility, inverse statistics, carrier energy, and
  the generalized Einstein ratio from one closure.

For explicit bands, `Ec = Ec_material − q psi` and `Ev = Ev_material − q psi`.
Band edges, quasi-Fermi energies, trap levels, tunneling barriers, Hamiltonians,
and quantum reservoir chemical potentials must use the same named energy
reference. A simultaneous potential/electronic-energy gauge shift leaves
populations and observables invariant.

A dielectric has no mobile-carrier state. A material interface is an oriented
lower-dimensional object with an explicit edge location and side fraction.
`MaterialInterface` preserves separate material traces, fixed integrated sheet
charge, and an optional potential jump. Transparent interfaces use the common
electrochemical traces; `ThermionicInterface` introduces four algebraic carrier
traces and one reciprocal Maxwell–Boltzmann transmission spectrum. It transports
particle energy and enforces detailed balance rather than averaging a band offset
onto a bulk edge.

The bulk discretization uses `B(x) = x/expm1(x)` and generalized
Scharfetter–Gummel exponential fitting. The activity difference is factored with
`expm1`, so common-temperature/common-Fermi equilibrium produces exact zero flux.
Abrupt explicit bands therefore admit a physical carrier-density jump without a
spurious equilibrium current.

## Mesh and material ownership

`TransportSupport` lowers nonuniform intervals and orthogonal tensor grids to
oriented edges, physical storage volumes, and transmissibilities. Lower-dimensional
models require their physical cross-section or extrusion measure. Otherwise a 2D
current is not an ampere-valued three-dimensional device current.
`TransportSupport.from_meshing` also lowers full-dimensional affine interval,
triangle, and tetrahedral `CellMeshingResult` blocks using lumped P1 volumes.
Assembled two-point weights must be strictly positive: inadmissible metrics are
rejected, not clipped into artificial diffusion.

Native mesh organization remains in `phydrax.meshing`:

- `MeshZone` binds exclusive materials;
- `MeshPatch` or scoped selections bind terminals;
- `MeshAttribute` stores unit-bearing dopant fields;
- `MeshingScope` binds selections to source identity and revision.

Mesh revision mismatches, overlapping exclusive materials, uncovered nodes, invalid
units, invalid contacts, and inadmissible transport metrics are errors before the
nonlinear solve. Mesh quality certification does not imply two-point-flux
admissibility on arbitrary distorted, anisotropic, or curved elements.

### A complete PN-junction operating point

Enable double precision before constructing device arrays:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from phydrax.applications import semiconductor as sc

plan = sc.pn_junction(nodes=81)
device = sc.PreparedSemiconductorDevice(plan)
equilibrium = device.equilibrium()
if not bool(equilibrium.successful):
    raise RuntimeError("Equilibrium failed")

point = device.solve(jnp.array([0.05, 0.0]), initial=equilibrium.coordinates)
if not bool(point.successful):
    raise RuntimeError("Biased operating point failed")

n, p = device.densities(point.coordinates)
print(plan.terminal_names)
print(point.terminal_currents)  # amperes, into the device
print(point.terminal_charges)   # coulombs
```

`pn_junction`, `mos_capacitor`, and `bipolar_transistor` are physical example
constructors. The transistor has a separate top-surface base contact in a 2D lateral
NPN geometry. They use scoped material/contact/dopant bindings rather than a hidden
solver-specific device format. Their silicon constants are illustrative 300 K
parameters, not a claim of agreement with a fabricated device.

## Constitutive model levels

The restricted `SemiconductorMaterial.silicon()` constants preserve the original
300 K nondegenerate examples. They are illustrative, not silicon-process data.
Explicit `BandThermodynamics` is required for aligned heterojunctions,
electrothermal material energy, traps, and tunneling energy references. Its
Fermi–Dirac quadrature, inverse population map, compressibility, pressure, and
kinetic/internal energies use one statistics definition rather than unrelated
fits.

`IncompleteIonization` is a local-equilibrium shallow-level closure. Out of
equilibrium it binds donors to `EFn` and acceptors to `EFp`; common-Fermi
equilibrium reduces to the standard neutral root. Because it does not own a
finite-rate impurity population, AC, transient, and conservative-transfer routes
reject it. Use explicit `DynamicTrap` populations when bound-charge dynamics
matter.

Higher-field mechanisms are separate owners:

- `LocalVelocitySaturation` declares electric-field or quasi-Fermi-gradient drive,
  orientation, temperature range, and maximum field. It preserves the conservative
  face flux and recovers the low-field mobility.
- `LocalImpactIonization` is a bounded Chynoweth pair-generation model with paired
  charge and carrier/band energy ledgers. It is not nonlocal avalanche or a
  stability criterion near breakdown.
- `CarrierEnergyTransport` and `CarrierEnergyRelaxation` form a reduced
  carrier-energy moment model. They do not implement momentum balance or full
  hydrodynamics.
- `WKBBarrierPath` resolves actual piecewise-linear turning points for one scalar
  effective-mass barrier. `NonlocalTunnelingPath` scatters separated endpoint
  charge and energy incidence. It is not a multiband complex-band model.
- `DynamicTrap` owns bounded bulk or surface occupancy, capture/emission, stored
  charge, trap energy, and equal-and-opposite carrier/lattice exchanges. It does
  not predict defect creation, annealing, BTI, HCI, or TDDB without additional
  calibrated kinetics.

Every invalid constitutive evaluation remains invalid; field, temperature,
degeneracy, WKB, and resource limits are not silently extrapolated or clipped.

## Storage, roots, transients, and adaptation

The physical state, nonlinear chart, and conserved storage are distinct.
`physical_storage(u)` reconstructs extensive electron/hole counts, lattice and
carrier material internal energies, and trap populations. `storage(u)` applies
fixed numerical scales and zeros algebraic rows. The semidiscrete equation is:

`d storage(u)/dt + residual(u, voltages) = 0`.

Poisson, terminal constraints, and thermionic interface traces are algebraic.
Native BDF therefore differences actual inventories instead of temperatures,
logarithms, or temperature-dependent normalizations. Consistent-rate recovery
solves every differentiated algebraic row, including contacts and interface
traces; it does not fill their rates with zero.

Equilibrium uses a reduced common-Fermi Poisson solve when the selected layout has
only three bulk fields. Energy, trace, or trap states use the full coupled root.
Biased sweeps retain native continuation attempts and rejected candidates.
`SemiconductorEvidence` independently checks finite state, residuals, terminal
KCL, bulk plus sheet plus terminal charge, and electrothermal port-power closure.

```python
trajectory = sc.semiconductor_transient(
    device,
    equilibrium,
    jnp.linspace(0.0, 1e-8, 5),
    lambda time: jnp.array([1e6 * time, 0.0]),
)
print(trajectory.successful)
print(trajectory.conduction_currents + trajectory.displacement_currents)
```

Drives must be JAX-differentiable within a run; split discontinuities into
separate runs. Dynamic local-equilibrium ionization is rejected because its bound
population has no conserved state.

Meshing owns topology transitions. `semiconductor_reprepare` uses native positive
lineage weights to preserve material-routed carrier/dopant counts, differential
material energy, and explicit bulk/surface trap populations. It reconstructs
temperatures, Fermi energies, and logits from transferred inventories, then
atomically reinitializes algebraic constraints. Electrostatic field energy is
reclosed by Poisson rather than falsely conserved across changed geometry.
Unsupported material routes or inadequate target trap capacity reject the whole
candidate and retain the prior accepted state.

## Terminal and derivative evidence

Terminal conduction current and charge are computed from the same discrete transfers
and storage as the residual. This enables KCL and charge-balance checks without
postprocessing an independently interpolated electric field. In transients, include
charge storage and displacement current; conduction current alone need not sum to
zero.

Small-signal analysis uses the operating-point residual and storage derivatives:
`(J + i omega C) delta_u = B delta_V`. Only terminal-sized excitations and observations
are needed; a dense all-state circuit Jacobian is not the scalable device API.

Observable derivatives are implicit derivatives of the converged residual. They are
conditional on a regular operating-point Jacobian and fixed topology, contact laws,
and model branches. Mesh transitions and branch changes are not silently smooth.

```python
ac = sc.semiconductor_small_signal(
    device, point, 2 * jnp.pi * jnp.array([1e3, 1e6]),
)
print(ac.admittance)  # frequency, output terminal, input terminal; siemens
print(ac.evidence.successful)

sensitivity = sc.semiconductor_sensitivity(device, point)
print(sensitivity.derivatives)  # d terminal current / d terminal bias
print(sensitivity.evidence.successful)
```

Material or fixed-topology geometry derivatives accept an explicit
`parameterize(theta) -> (prepared_device, terminal_voltages)` function and optional
direction vectors. Use `plan.with_doping` to reclose doping-dependent mobility and
contact neutrality. Geometry parameterizations must update physical volumes and
transmissibilities together with positions. Numeric coefficients and normalization
are recomputed from the replaced plan; sparse derivative topology remains fixed.
Failed derivative columns are marked invalid and contain NaNs.

`SemiconductorCircuitLaw(device)` is a native implicit circuit element.
`semiconductor_circuit_operating_point` solves coupled MNA/device residuals through
native Newton–Krylov without forming all-state dense circuit Jacobians.
`semiconductor_circuit_transient` requires explicit initial state/rate and supports
a caller-specified consistency mask: PDE displacement current can produce a
descriptor structure that a diagonal differential-role guess does not resolve.
The circuit owns topology and voltage-source constraints; the device owns its
internal storage and terminal ledger.

Both coupled wrappers require `current_scale=...` in amperes: a positive scalar
or one value per non-ground circuit node. This scales KCL equations, not physical
terminal-current outputs. For example, declare `current_scale=1e-10` for a
nanoampere-scale circuit rather than accepting its unsolved KCL residual against
a dimensionless default tolerance. Device contact rates are tied to the circuit
voltage rates, including capacitive gate displacement current.

Choose `current_scale` and the native nonlinear termination tolerances together
to declare the required ampere-valued error. At very small currents, flux
cancellation can limit attainable accuracy; the wrapper retains failure rather
than silently relaxing the requested tolerance.

The circuit uses the same named extensive-storage chart. Convert physical device
coordinates with `law.initialize(u)` and physical coordinate rates with
`law.initialize_rate(u, du_dt)`; initialization is nonlinear, so applying
`initialize` to a rate is not a valid coordinate transformation.

DC Newton iterations use native right nonlinear preconditioning in quasi-Fermi
coordinates. Native reconstruction independently certifies the returned conserved
storage state and retains solver-coordinate transformation evidence.

## Quantum model hierarchy

`phydrax.applications.semiconductor.quantum` is a separate bounded backend. It
does not reinterpret density-gradient correction, AC drift–diffusion, or a
numerical imaginary regulator as quantum transport.

1. `DensityGradient1D` evaluates a declared von Weizsäcker/Bohm correction on
   the same conservative effective-mass chain used by confinement. It has no
   reservoir injection.
2. `EffectiveMass1D` plus `solve_schrodinger` provides orthonormal-cell
   confinement with explicit mass, area, transverse modes/degeneracy, selected
   native eigensolves, and a Sturm-certified omitted-population bound.
3. `solve_schrodinger_poisson` uses the same cell projector to pull potential
   into the Hamiltonian and push occupied charge into Poisson.
4. `CoherentDevice` binds a scalar nearest-neighbor Hamiltonian to two analytic
   semi-infinite leads. `integrate_coherent` solves only selected Green columns,
   isolates true bound poles with explicit preparation, integrates the complete
   finite lead band, and gates current, energy, spectral-sum, quadrature,
   resonance, and numerical-broadening refinement.
5. `solve_phonon_transport` is a specific local optical-phonon Fock SCBA on an
   integer-shift energy grid. Retarded, lesser, and greater self-energies are
   iterated together; discrete particle, energy, causality, KMS, grid, and
   window gates are retained. It is not an arbitrary imaginary potential.
6. `solve_quantum_transient` evolves the full one-body correlation of a finite,
   refinable lead dilation. It retains initial-state preparation, switching
   work, return-time, memory-kernel, unitarity, particle, energy, and lead-size
   evidence. This is bounded coherent memory, not general interacting
   time-dependent NEGF.
7. `coherent_low_frequency_noise` implements two-terminal coherent
   Landauer–Büttiker noise and checks equilibrium fluctuation–dissipation.
   `finite_frequency_quantum_response` uses connected-equilibrium Kubo response,
   explicit lead polarization, and capacitive Hartree screening with gauge,
   KCL, Ward, lead-size, adiabatic-rate, and recurrence gates. It does not claim
   an interacting vertex correction.

The high-level quantum basis is one connected scalar orthogonal 1D chain with a
finite declared transverse-mode list. Arbitrary atomistic, 3D, multiband, and
nonorthogonal charge reconstruction are not claimed. `scalar_embedding` exposes
the correct `E S − H` scalar cross-overlap primitive, but using it does not turn
the orthogonal high-level backend into a generalized-basis solver.

`QuantumClassicalInterface` joins one electron quantum reservoir to one ohmic
classical terminal through a common chemical energy and scalar voltage bracket.
It shifts lead band and chemical potential together, matches the
species-resolved electron current, rejects unresolved classical hole leakage,
and additionally matches electron heat for an energy-enabled classical region.
Distinct region identities are mandatory; one population cannot simultaneously
own DD mobility/SRH and quantum self-energies.

```python
from phydrax.applications.semiconductor import quantum as sq

lead = sq.SemiInfiniteLead(
    onsite, hopping, coupling, chemical_potential, temperature,
    energy_reference="declared datum",
)
device = sq.CoherentDevice(hamiltonian, lead, lead)
transport = sq.integrate_coherent(device)
if not bool(transport.successful):
    raise RuntimeError("Quantum transport did not pass its numerical gates")
```

## Calibration and evidence claims

`SemiconductorCalibrationDomain` fixes the process revision, population,
geometries, controls, and envelope. `SemiconductorMeasurementCase` is one joint
correlated event retaining lot/wafer/die/structure/condition grouping, terminal
definitions, observation units, complete control history, instrument and
de-embedding records, covariance, provenance, uncertainty, and data rights.
`SemiconductorCalibrationCampaign` enforces a locked, leakage-free split over
whole physical groups.

`prepare_semiconductor_calibration` binds a constrained native
`ParameterSpace`, measured-control uncertainty, and correlated Gaussian
likelihoods. A physically inadmissible parameter can have zero prior density; a
numerically unresolved forward solve instead raises with its retained evidence
and is never treated as impossible physics. `calibrate_semiconductor` returns a
native MAP, exact local Laplace approximation, local sensitivity rank, and
untouched held-out predictive checks. Local full rank is not global
identifiability.

`qualify_semiconductor_calibration` separates:

- numerical conservation/refinement/solver evidence for one model/build;
- independent global-identifiability or separating-metrology evidence;
- locked held-out experimental evidence;
- authorized experimental source rights and uncertainty;
- geometry, dopant, material, and contact metrology;
- coverage of every declared geometry and control bound.

Synthetic recovery qualifies inference mechanics only. This repository contains
no authorized named-foundry campaign, so it does not produce a
foundry-calibrated capability claim.

## Qualification and performance

Qualification is model-level and domain-level. Classical checks include
common-Fermi equilibrium, band and displacement jumps, carrier/trap conservation,
thermal/electrical port power, terminal KCL, and mesh/time refinement. Quantum
checks additionally include eigenmode truncation, basis charge normalization,
causality, spectral completeness, bound-pole population, current/energy collision
balance, lead/grid/window/broadening refinement, memory recurrence, gauge/Ward
identities, and fluctuation–dissipation. A small algebraic residual is necessary,
never sufficient.

Run matched classical and coherent-quantum benchmark routes from the repository
root:

```sh
JAX_ENABLE_X64=1 PYTHONPATH=.:benchmarks .venv/bin/python benchmarks/semiconductor.py \
  --case all --nodes 21 41 --warmup 1 --repeats 3
```

The benchmark records environment, physical acceptance gates, preparation,
complete classical continuation, coherent energy integration, lowering and
compilation, warm residual/JVP or selected-source execution, and physical
observables. Failed physics is never reported as fast successful performance. No
external-code speedup or empirical/foundry claim is inferred from timing.

## Scientific references

- [DEVSIM](https://github.com/devsim/devsim): conservative classical TCAD and terminal/circuit workflows.
- [DEVSIM BJT example](https://github.com/devsim/devsim_bjt_example): equilibrium, continuation, DC, and AC workflow.
- [SIMUDO](https://doi.org/10.1007/s10825-019-01414-3): mixed drift–diffusion and multiband optoelectronic modeling.
- [NIST DLMF 25.12](https://dlmf.nist.gov/25.12): normalized Fermi–Dirac integrals.
- [Kantner and Koprucki](https://arxiv.org/abs/2002.10133): generalized Einstein relations and nonisothermal semiconductor transport.
- [Lake et al.](https://doi.org/10.1063/1.366764): nonequilibrium Green-function device transport.
- [Blanter and Büttiker](https://doi.org/10.1016/S0370-1573(99)00123-4): coherent scattering noise.

The implementation uses Phydrax-native kernels and solver substrates; it does not
import or translate external simulator implementations.
