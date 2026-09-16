# Production and frontier quantum field theory

Phydrax treats quantum field theory as a family of explicitly regulated
computational regimes. It does not define one universal QFT state or solver.
Each model lowers into the canonical numerical substrate matching its state,
measure, constraints, and approximation theory.

## Computational regimes

The supported architecture separates:

- positive-weight Euclidean lattice configurations;
- complex or signed measures;
- classical canonical real-time fields;
- finite Hilbert states and tensor networks;
- open-system density, Gaussian, and trajectory representations;
- imaginary-time Green functions and diagrammatic expansions;
- relativistic scattering amplitudes and event measures;
- variable-particle continuum Fock configurations;
- finite functional-renormalization truncations.

These regimes may share topology, linear algebra, integration, autodiff,
sampling, persistence, and qualification infrastructure. They do not share an
implicit measure, norm, physicality condition, or convergence claim.

## Production levels

A capability progresses independently through the following claim levels.

### R0: mathematical specification

The equations, regulator, reference measure, field/operator spaces,
symmetries, constraints, observables, approximation axes, and nonclaims are
explicit. No numerical correctness claim is made.

### R1: finite algebraic qualification

Construction identities, adjoints, gauge covariance, conservation laws,
resource bounds, JIT behavior, and exact small-system controls are satisfied.
The capability remains a finite reference implementation.

### R2: research-grade scientific capability

A reproducible workflow carries correlated uncertainty, independent
convergence axes, semantic randomness, checkpoint ancestry, domain evidence,
and an analytic or independent reference artifact.

### R3: production single-device capability

The public API, failure/status surface, precision policy, interruption-safe
restart, warmup/production separation, memory behavior, and steady-state
runtime envelope are fixed and qualified.

### R4: distributed production capability

Partition independence, global entity identity, halo/collective evidence,
rank-independent checkpointing, scaling, and distributed failure propagation
are qualified.

### R5: external cross-qualification

An independent implementation or admissible reference artifact agrees under
an explicit convention and uncertainty reconciliation. Promotion is governed
by the existing qualification repository and campaign-causality rules.

### X: frontier/research-only capability

Complex Langevin, holomorphic-flow, unrestricted functional truncations,
learned topology-changing proposals, 2PI/Keldysh closures, and semiclassical
gravity remain explicitly research-only. Reusing a production numerical
solver does not promote their scientific claim.

## Canonical lowering paths

A positive Euclidean action lowers to a Markov target relative to a named
Lebesgue, counting, angular, Haar, or pseudofermion reference measure. A
Hamiltonian model lowers to dense, matrix-free, local, MPO, PEPS, or quantum
program representations. A perturbative process lowers to a Lorentz-invariant
phase-space integrand. A semiclassical model lowers to coupled differential,
Maxwell, and ensemble problems. A functional truncation lowers to a finite
flow equation.

Every lowering records model, regulator, representation, topology, precision,
operator, algorithm, and approximation identities. Two lowerings are called
equivalent only when their comparison is qualified.

## Permanent separation boundaries

Phydrax never silently equates:

- Euclidean sampling with real-time evolution;
- positive measures with complex weights;
- group-valued links with algebra or complexified matrix fields;
- finite groups with compact Lie groups;
- exact gauge sectors with penalty sectors;
- gauge fixing with gauge-invariant ensemble generation;
- Hilbert vectors with Euclidean configurations;
- density evolution with stochastic trajectory evolution;
- DLR compression with analytic continuation;
- perturbative scattering with spatial field evolution;
- semiclassical electrodynamics with fully quantized QED;
- interchange archives with restart checkpoints;
- external kernels with canonical physics semantics;
- learned proposals with target distributions;
- warmup adaptation with frozen production;
- continuum-fit orchestration with an automatic continuum claim.

## Euclidean lattice production

Production lattice calculations compose field spaces, gauge representations,
ordered covariant transport, loop actions, staples, local updates, molecular
dynamics, pseudofermions, linear solves, measurements, and ensemble manifests.

Dynamical fermion actions remain separate concrete implementations: Wilson,
Wilson-clover, twisted mass, staggered/HISQ, domain wall/Mobius, and overlap.
Even-odd Schur complements, clover inverses, smearing transforms, rational
approximations, pseudofermion refresh, trajectory forces, and acceptance
energies all carry independent evidence.

RHMC checkpoints are published only after a completed accept/reject boundary.
Transient solver workspaces and candidate forces are never restart state.

## Distributed lattice production

Global site, link, face, spinor, and observable ownership lowers into existing
distributed fields and collectives. Operator execution distinguishes halo
packing, communication start, interior work, communication completion,
boundary work, and global reduction. Backend-specific memory layouts and
communication handles do not enter the public model API.

Native JAX implementations remain authoritative. Optional external providers
supply granular kernels only after cross-qualification.

## Continuum and thermodynamic studies

A continuum campaign binds bare theory points, regulator, volume, lattice
spacing, anisotropy, action improvement, scale setting, renormalization,
measurement covariance, autocorrelation, and systematic fit variations.

Continuum or thermodynamic-limit claims require declared resolution counts,
fit stability, volume control, correlated residuals, scale-setting evidence,
and absence of unresolved topology-freezing or overlap failure. Automation
organizes this evidence; it does not certify a limit by itself.

## Hamiltonian field theory

Finite-group, truncated U(1), and representation-truncated SU(2) link Hilbert
spaces own their local algebras and cutoff defects. Gauge constraints lower to
resource-bounded exact sectors, sparse projectors, tensor-network intertwiner
bases, or gauge-preserving quantum programs. Penalty terms are optional
algorithmic aids, never the physical-sector definition.

Open and periodic Schwinger models remain separate because periodic topology
retains a global electric-flux sector. Higher-dimensional theories retain
explicit links rather than reusing the one-dimensional prefix elimination.

## Thermal and diagrammatic field theory

DLR bases represent finite-temperature kernels under explicit spectral cutoff
and tolerance. Imaginary-time, Matsubara, and DLR Green functions carry
statistics, moments, and representation evidence. Analytic continuation is a
separate ill-posed inference problem with its own prior or regularization.

Typed diagrams record fields, propagators, vertices, routing, symmetry factors,
regulator, subtraction, renormalization scheme, and perturbative order before
lowering to the existing computational DAG. Diagrammatic Monte Carlo records
proposal balance, sign, order tails, and normalization references.

## Relativistic scattering

Relativistic processes keep momentum, frame, mass shell, spin, polarization,
color, and incoming/outgoing conventions explicit. Amplitudes remain separate
from spin/color sums and from phase-space integration. VEGAS adaptation is
finite and followed by a frozen production grid. Weighted, signed, and
unweighted events carry distinct measures and provenance.

## Open and semiclassical field theory

Gaussian partial transpose and logarithmic negativity extend the existing
Gaussian-state substrate. Angular-momentum compilers lower atomic manifolds,
drives, and radiative channels into existing Hamiltonian, Lindblad, CPTP, and
trajectory solvers.

Deterministic initial-condition ensembles sit above differential solvers and
remain distinct from stochastic dynamics. Semiclassical QED carries its gauge,
mode normalization, current renormalization, cutoff, backreaction, Ward,
energy, and response evidence. Classical Maxwell convergence is independent
from quantum or semiclassical modeling error.

## Frontier methods

Variable-sector neural states use a fixed compiled capacity and an exact
disjoint-union measure. Birth/death moves include their reverse density,
combinatorics, and Jacobian. Functional RG exposes finite truncations and
regulator dependence, not arbitrary functionals. Complex-weight methods
report overlap or correctness diagnostics and abstain when they fail.

Supersymmetric complexified links are never reunitarized. Anyonic fusion data
must satisfy pentagon and hexagon identities. Conformal-bootstrap results
retain their tensor-basis gauge, spin/derivative/pole/block/precision
truncations, exact PMP input, external process identity and independent audit.
Sampled positivity and external numerical convergence remain finite claims,
not continuum CFT exclusions. Curved-spacetime calculations retain the state,
subtraction, background, and stress-conservation evidence and make no
quantum-gravity claim.
