# Fermionic frontier field methods

These APIs are bounded **candidate** controls. They are not released production
profiles, do not establish a material phase diagram, do not imply long-time
nonequilibrium convergence, and do not provide a generic cure for a sign
problem. A unit test, smoke result, benchmark result, or declared candidate
profile is not release evidence.

## Two-particle Matsubara convention

`FermionicTwoParticleChannelConvention` fixes the operator order to

```text
(annihilation, creation, annihilation, creation)
```

for the connected correlator `⟨T c₀ c₁† c₂ c₃†⟩c`. Fermionic integer labels
mean `ωₙ = (2n + 1)π/β`; transfer labels mean `νₘ = 2mπ/β`. The three routed
channels parameterize the same conserving four external labels:

| Channel | External fermionic labels `(n₀,n₁,n₂,n₃)` |
| --- | --- |
| particle-hole direct | `(n + m, n, n′, n′ + m)` |
| particle-hole crossed | `(n′ + m, n, n - m, n′)` |
| particle-particle | `(n, n′, m - n - 1, m - n′ - 1)` |

`MatsubaraTwoParticleGreenFunction` stores axes as
`(transfer,left,right,mode₀,mode₁,mode₂,mode₃)`. Its route record retains both
unwrapped external labels and integer cyclic-bank wrap counts. Channel
conversion is an explicit cyclic permutation; it does not reorder operators or
silently complete missing data. `fermionic_crossing_evidence` checks both
single exchanges with a minus sign and the simultaneous exchange with a plus
sign wherever the finite bank has coverage.

The type does not infer disconnected subtraction, analytic continuation,
orbital gauge, or a one-particle-to-many-body bridge.

## Two-dimensional SU(2) patch fRG

`FermiSurfacePatchRGPlan` is restricted to a two-dimensional, single-band,
static spin-reduced vertex. Preparation proves that every routed fourth patch
closes modulo the supplied reciprocal vectors and retains the integer wraps.
`FermionicEnergyShellRegulator` is the additive linearized-dispersion shell

```text
RΛ(ξ) = sgn(ξ) max(Λ - |ξ|, 0).
```

The finite-temperature single-scale particle-particle and particle-hole loops
are evaluated on caller-declared radial and Matsubara nodes. The one-loop 1PI
flow is contracted in the full SU(2) spin tensor and projected back to the
direct spin-reduced amplitude. Evidence reports input and beta-function Pauli
crossing, reciprocal routing, regulator denominators, finiteness, and the
fixed U(1) routing Ward residual. This candidate does not reuse or relabel the
existing scalar/bosonic functional-RG methods.

## Lattice parquet and Schwinger–Dyson control

`LatticeParquetPlan` consumes a connected single-band two-particle Green
function in the direct particle-hole channel. Preparation constructs exact
bijections among direct particle-hole, crossed particle-hole, and
particle-particle cyclic banks. Iteration retains each reducible channel,
channel residual, the parquet identity residual, crossing residual, and routing
residual.

`lattice_schwinger_dyson_evidence` independently checks the declared local
single-band Matsubara contraction against a supplied scalar self-energy. The
one-particle Green function, self-energy, interaction, density, temperature,
and all Matsubara axes are explicit; a parquet fixed point is not treated as a
Schwinger–Dyson proof by itself.

## CT-INT and low-order diagram Monte Carlo

`SignFreeCTINTPlan` is only the repulsive, half-filled, bipartite
particle-hole-symmetric control. The caller supplies spin kernels and a stable
kernel identity. Every sampled configuration checks

```text
M↓ = -D conjugate(M↑) D,
```

where `D` contains the supplied sublattice signs. Under that condition,
`(-U)^k det(M↑) det(M↓) = U^k |det(M↑)|²`. The implementation deliberately
uses the native one-to-three-dimensional determinant substrate and refuses a
higher expansion order. Uniform insertion in site and imaginary time and
uniform removal give the exact proposal-density ratios

```text
q(remove)/q(insert) = number_of_sites × β / (k + 1),
q(insert)/q(remove) = k / (number_of_sites × β).
```

`LowOrderFermionDiagramMonteCarloPlan` instead samples a finite caller-supplied
fermionic diagram catalogue with a reversible proposal matrix. It retains the
raw configuration index, perturbative order, phase, proposal ratio, weight
ratio, acceptance probability, and decision at every step. Both controls
report detailed balance, order autocorrelation/ESS, phase covariance, average
sign, signed ESS, acceptance, and failed capacity proposals. Neither result is
a thermodynamic-limit or generic sign claim.

## Fermionic Keldysh and Kadanoff–Baym functions

`fermionic_keldysh_from_propagators` uses the conventions

```text
G<(t,t′) =  i ⟨c†(t′) c(t)⟩,
G>(t,t′) = -i ⟨c(t) c†(t′)⟩.
```

The fermionic CAR is therefore `i [G>(t,t) - G<(t,t)] = I`. Retarded,
advanced, and Keldysh components are distinct from the existing real-scalar
statistical/spectral functions. Identity evidence checks lesser/greater
anti-Hermiticity, equal-time CAR, retarded/advanced adjointness, and both
causal supports.

`FermionicSecondBornPlan` is a spin-degenerate, closed, local-Hubbard control.
It iterates the discrete weighted Dyson equation self-consistently with the
local second-Born lesser/greater self-energies. Its result contains the raw
self-energy, fixed-point and Schwinger–Dyson/Kadanoff–Baym residuals, particle
number, a discrete energy ledger, CAR, causality, convergence, and conservation
status. Failure of any required predicate leaves the candidate unsuccessful.
Finite memory, open reservoirs, initial correlations beyond the supplied free
functions, and long-time convergence are outside this profile.

## Controlled sign study and abstention

`ControlledSignStudyPlan` assumes its leading axes are phase-quenched Markov
chains. `evaluate_controlled_sign_study` retains all raw complex weights,
phases, and observables. It reports the complex average phase, its magnitude,
phase covariance, phase–observable covariance, integrated autocorrelation
time, effective sample size, and a delta-method reweighted uncertainty.
Nonfinite input, too few chains/draws, phase cancellation, or insufficient ESS
returns an explicit status and NaN reweighted estimates. Failed samples are not
dropped.

The smoke command is:

```bash
python tools/cm_frontier_field_smoke.py
```

The decision benchmark is:

```bash
python benchmarks/cm_frontier_field.py --repeats 2 --output frontier.json
```

Both are candidate engineering evidence only. Qualification requires
predeclared independent calibration and locked cases, retained failed attempts,
measured resource envelopes, current runtime/build attestations, and signed
release gates.
