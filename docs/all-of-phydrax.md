# All of Phydrax

This page provides a high-level map of the library, how the parts fit together,
and where to look for specific functionality.

## Unifying formalism: minimizing functionals over domains

Phydrax is designed to make a single idea modular:

> Define fields on labeled domains and minimize scalar **functionals** built from operators and
> measures over domain components.

A training functional combines nonnegative penalty terms and model-level losses:

$$
\mathcal J[u] = \sum_i \ell_i[u] + \sum_k r_k(\theta).
$$

Each \(\ell_i\) pairs a residual, moment, or observation condition with an
explicit numerical integration source. Domain components and their induced
measures define the semantics of those sources.

`phydrax.variational.Functional` is the representation-independent physical
case: named field jets and signed region terms can be bound either to
`DomainFunction` integration terms or to a finite-element discretization. The
neural optimizer derivative is a parameter pullback; the FE residual is a
dual-valued discrete first variation. Neither is silently relabeled as a metric
gradient.

## The compositional contract

At a practical level, most workflows look like:

1) choose a **domain** \(\Omega\) and a **component** \(\Omega_{\text{comp}}\subseteq\Omega\),  
2) define one or more **fields** \(u_\theta:\Omega\to\mathbb{R}^m\) as `DomainFunction`s,  
3) build **residual operators** \(r=\mathcal{N}(u_\theta,\dots)\) using `phydrax.operators`,  
4) declare residual, moment, or observation **conditions**, then pair them with
   explicit integration sources and **penalty terms**,
5) sum terms and optional model losses into \(\mathcal J\) and optimize with
   `FunctionalSolver`.

Two design choices make this interoperable:

- **Labeled product domains**: every coordinate is a named factor (`"x"`, `"t"`, `"data"`, `"p"`, …).
- **Structured batches**: sampling preserves axis semantics (paired sampling and coord-separable grids).

## Key choice points (what makes workflows differ)

### Sampling: point batches vs axis-based grids

PhydraX exposes two typed plans and matching batch schemas:

- `PointSampling` → `PointBatch` for paired collocation and scattered data;
- `GridSampling` → `GridBatch` for basis/spectral operators and neural operators
  with explicit coordinate axes.

Sampling owns sites and measure metadata; interpolation owns deterministic
reconstruction of stored values at query sites. Low-level source-to-target
execution is shared through `phydrax.sparse`: arbitrary edge relations,
fixed-width case-local rows, robust masked routing, target reduction, and
weighted forward/transpose/adjoint actions. Interpolation stencils,
`QueryNeighborhood`, `GraphIR`, and cochain incidences retain their own
geometry, topology, support, and measure semantics above that substrate.
Fourier evaluation, sparse Smolyak approximation, sparse Gaussian processes,
and stochastic estimators remain specialized methods rather than sparse
storage types. See [API → Operators → Interpolation](api/operators/interpolation.md).

Adaptive residual policies remain source-owned. R3, RAR-D, and coreset policies
change point support; `ResidualAttentionCollocation` instead keeps support fixed and
updates a unit-mean local multiplier with explicit uniform and effective-sample-size
guards. Independent fixed evaluation terms remain the evidence surface.

Finite-dimensional algebra above those storage kernels is shared through
`phydrax.linalg`: paired array/PyTree/block spaces, composable explicit and
matrix-free operators, exact/least-squares/minimum-norm problem contracts,
deterministic capability-based planning, reusable factorizations, and portable
status, diagnostics, and provenance. Standard and generalized nonsymmetric
eigenproblems use the same spaces, operators, prepared transforms, cost models,
and failure semantics; dense QZ and native restarted-Arnoldi/Krylov--Schur paths
are explicit backend choices. Provider-neutral sparse derivative plans compile
known structural patterns natively or use ASDEX once for global pattern
detection and optimized coloring. Repeated Jacobian and Hessian evaluation is
native JAX and produces ordinary sparse coordinate operators. See
[API → Linear algebra runtime](api/linalg.md) and
[API → Sparse derivatives](api/sparse_derivatives.md).

Einstein-style array operations are shared through `phydrax.ein`: an exact
optimized contraction boundary plus named static JAX rearrangement, reduction,
and repetition. Patterns compile to reshape, transpose, reduction, and
broadcast primitives without runtime axis metadata. See
[Guide → Einstein operations](guides_ein.md) and
[API → Einstein operations](api/ein.md).

Certified positive-semidefinite actions may prepare a fixed-rank
`RandomizedNystromPreconditioner`. Its deterministic sketch, positive shift,
refresh mode, retained Ritz evidence, storage, and exact setup matvec count flow
through the same preconditioner plan/prepare/refresh provenance as deterministic
builders.

### Discretization: supports, field spaces, and formulations

`phydrax.discretization` binds labeled continuum semantics to finite topology,
geometry, DOF layouts, measures, prepared operators, transfers, and complete
approximation bundles. Tensor support is independent of finite-difference,
spectral, or collocation calculus. Global tensor spectral spaces separate mathematical
modes, physical grids, modal DOFs, dealiasing, constrained/Galerkin/tau formulations,
and periodic Leray projection. Periodic incompressible integration retains
full-complex live state; independent-real Hermitian coordinates are optional
backend/checkpoint encodings. Constant-power forcing uses volume-mean native-modal
normalization and fails closed below its declared forced-energy threshold. OU forcing
continues exactly in real solenoidal modal coordinates; its prepared ETDRK wrapper
uses the declared start/half/end stage values and commits coefficients atomically
with velocity. This exact stochastic transition does not make the fluid quadrature
exact. Full-complex shell statistics conserve native modal integrals, and
accepted-step statistical windows provide sample/time weighting and completed-block
uncertainty.

Fourier--Chebyshev--Fourier channels retain primitive public velocity/pressure fields
while the default internal Stokes route eliminates pressure into fixed-band
ultraspherical systems with fixed-rank corrections. The zero mode owns
pressure-gradient or bulk-flux control; nonzero modes use wall-normal
velocity/vorticity elimination. Restartable SBDF2 carries complete nonlinear,
velocity, pressure, and history state and requires its prepared step exactly.
`dense_reference` is an explicit channel oracle and carries no banded-route
production or qualification evidence.

Axis domains cover bounded, periodic, half-line, and real-line support; rational
Chebyshev bases, canonical modal transfers, physical modal-tail diagnostics, and
cross-resolution eigenspace evidence retain their mapping, trace, exactness, and
resource identities. Exact-sampling round-sphere spaces separately expose S2FFT mode
layouts, physical area measure, scalar Laplace--Beltrami actions, complete-degree noise
bases, and the prepared discretization consumed by SFNO.
Local stencil programs, structured compact and transform-line solves,
periodic/bounded SBP calculus, entropy-conservative SBP flux differencing, and
compatible finite-volume MAC flow compose without conflating quadrature sites, mesh
entities, and field DOFs. Constant-density separable all-Neumann MAC pressure may use
a three-dimensional hybrid route: transforms on two uniform transverse axes plus a
nonuniform physical Neumann line, with explicit compatibility and volume gauge.
Variable-density, mixed/open, distributed, and sharded-line cases are excluded from
that hybrid route, and MAC has no fixed-bulk-flux controller.

The broader MAC substrate includes dynamic wall/inflow/open closures,
symmetry-preserving momentum, named scalar/Boussinesq and conservative
variable-density dynamics, implicit diffusion, resolved face-marker coupling,
sharded pressure CG, mapped/ALE geometry, remesh epochs, adaptive replay, and
bounded sensitivity modes. Its raw plane/wall statistic route centers staggered
components to cells for exact-volume plane moments, retains native wall-normal
boundary faces, and reports separate signed one-sided shears relative to explicit
lower/upper wall velocities. WENO fluxes, fixed-capacity AMR, and distributed halo
plans remain available to the wider finite-volume family.
Route-specific periodic-spectral, channel, and structured-MAC production assemblers
carry accepted PyTrees through bounded compiled segments to an absolute end
time/capacity, with content-derived checkpoint IDs, exact-time scheduled output,
typed trigger actions, and windowed/block moments.
These capabilities do not constitute a universal DNS claim or distributed
spectral/hybrid-line support.

Qualification remains route and case specific. When generated, separate
periodic-spectral, spectral-channel, and MAC artifacts bind exact
support/input/reference/configuration identities. Assembly of passed artifacts
produces an unsigned `CapabilityProfile` candidate with `released=false`, not a
library-wide badge.


The S1 isogeometric path binds two clamped isotropic B-spline grids to a
positive, mean-one-gauge NURBS control net and an exactly isoparametric scalar
H1 field. It covers regular, untrimmed, full-dimensional two-dimensional
single-patch volume maps, explicit Gauss quadrature, homogeneous strong traces
or natural Neumann boundaries, and matrix-free sum-factorized finite-element
forms. Its regularity evidence is sampled at the declared quadrature sites:
even when a quarter-annulus control net represents its rational map exactly,
the sampled minimum denominator, orientation determinant, and reciprocal
condition margins are not a global injectivity proof. These volume maps are not
the repository's BRep surface geometry and do not imply trimming, CAD topology,
shells, multipatch coupling, or three-dimensional solids.
Explicit polygon H1 elements bind conformingly segmented, star-shaped planar
polygons to a transported witness fan. A native discrete-harmonic condensation
removes the private witness coordinate while retaining exact linear traces,
partition of unity, affine reproduction, actual piecewise gradients, direct
reconstruction, and dense matrix-free local functional execution. Qualification
retains fan validity, factorization, rank, spectrum, conditioning, and
reproduction evidence. Higher order, nonmatching T-junctions, sparse realization,
and three-dimensional cells remain outside this surface.
Enhanced conforming virtual elements bind arbitrary-arity polygonal cell blocks,
vertex/edge/moment functional coordinates, certified H1 and L2 polynomial
projections, explicit projector-kernel stabilization, matrix-free or sparse
realization, trace constraints, projected reconstruction, heat/eigen reuse, and
fixed-topology differentiable geometry. They remain distinct from reference
finite elements and do not fabricate virtual interior basis values.
Material particles retain stable entity IDs and a physical mass measure while
current positions remain temporal state. Fixed-h conservative barotropic SPH
uses canonical unordered pairs, normalized compact kernels, energy-derived
pressure forces, dense or fail-closed cell-list execution, exact `GraphIR`
views, and the native separable-Hamiltonian solver path. First-order WCSPH adds
explicit summation/continuity density semantics, Morris physical viscosity,
energy/dissipation ledgers, and native SSPRK integration. Structured
particle-grid splatting adds measure-aware extensive deposition, intensive
reconstruction, adjoint gather, explicit boundary loss, multilinear and
degree-one through degree-three B-spline assignments, mixed entity layouts,
route moments, and fast/deterministic/compensated reductions. Compatible
particle-in-cell methods attach extensive macrocharge to the same stable
particle support, deposit endpoint charge on degree-zero cochains, gather
physical E/B from oriented cochain layouts, solve compatible electrostatics, and
advance periodic 3-D Maxwell fields with a trajectory current that certifies
discrete continuity. Fixed-population free-surface FLIP separately binds cell and
staggered-face splats to a runtime atmospheric MAC projection and an explicit
PIC/FLIP grid-delta update; it neither reuses MPM constitutive state nor claims
SPH/VOF interface geometry.
Advanced particle-grid methods preserve those authorities while adding runtime
population/incarnation state, integer PIC charge and microphysics events,
one/two-dimensional mixed Maxwell blocks, open-boundary and moving-window
ledgers, affine simplicial ownership and Whitney current, and a matrix-free
semi-implicit response. Advanced FLIP derives one particle interface geometry
for ghost pressure, capillarity, free-surface viscosity and multiphase material
reconstruction, with moving-solid cut measures and fixed-pool reseeding treated
as explicit accepted-step transactions.
Vortex methods bind scalar 2-D circulation or vector 3-D integrated vorticity
to typed source states without reinterpreting material mass. Capability-driven
direct, Ewald, free-space FFT, corrected P3M, hierarchical FMM, periodic VIC,
PSE/core-spreading/redistribution, classic/rVPM/LES/baroclinic formulations,
transactional populations, shared ring/sheet wakes, multi-surface lifting,
native 2-D/3-D panels, no-slip and immersed wall coupling, native rigid/flexible
FSI, rotors, control, acoustics, stochastic ensembles, learned reconstruction,
assimilation, checkpoints, replay, export, and sharding retain explicit validity
and derivative evidence. See [Guide → Vortex architecture](guides_vortex_architecture.md),
[Guide → Vortex field backends](guides_vortex_backends.md),
[Guide → Vortex diffusion and topology](guides_vortex_diffusion_topology.md),
[Guide → Vortex FSI and control](guides_vortex_fsi_control.md), and
[Guide → Stochastic and learned vortex methods](guides_vortex_stochastic_learning.md).
Cosmological applications reuse those particle, grid, solver, operator, artifact, and
likelihood identities rather than introducing a separate framework. Canonical physical
states project only declared dependencies into content-addressed products; pinned
precision processes, one-loop SPT, calibrated 200m halo ingredients, release-window
likelihoods, primordial microphysics, CMB sky/TOD/mapmaking, periodic Ewald evidence,
snapshots, and distributed-PM feasibility extend the curved/CPL background, named
transfer/power, flat LPT/PM, and gas--DM foundations. Native relic, BBN, recombination,
nonlinear/halo, lensing, light-cone, survey-selection, and baryonic-feedback products
share those identities. Bounded maximal profiles add fixed-layout scalar transfer/LOS,
global S3 geometry and particles, typed multi-release surveys, deterministic FoF and
merger products, stochastic star populations, two-level AMR, sparse occupied Morton
point hierarchies, isolated Barnes--Hut, sparse occupied-level Cartesian FMM, and
BH-short-range TreePM. Core discretization additionally provides fixed-resolution
sparse voxel fields; covering, face-balanced dyadic cell topology with conservative
adaptation and explicit-face finite-volume lowering; and oriented finite-footprint
surfels with physical surface measure, primitive bounds, ray queries, local voxel
projection, atlas/mesh materialization, and branchwise differentiation. Every profile
states unsupported
species, topology, approximation, capacity, distribution, and communication branches;
precision parity beyond qualified profiles, multilevel distributed AMR, distributed
trees, production feedback, and full release coverage remain separate qualification
claims rather than hidden flags.
Prepared periodic Fourier shells now provide continuum-normalized isotropic auto/cross
power, phase-sensitive residuals, Hermitian mode accounting, and explicit spectral
validity. One-epoch measured power products stack into canonical tables. Inverse
particle-field realization composes the existing measure-aware splat, shared covariance
algebra, periodic position parameters, and optimization/sensitivity machinery without
introducing a second assignment, PM, inference, or FFT substrate.
Cross-domain reconciliation keeps those domain states distinct while moving shared
mechanics to core owners: dimensional scales, artifacts and derivative capabilities,
labelled observation/covariance/likelihood algebra, direct and hierarchical particle
gravity, coefficient-driven KDK, and ratio-two AMR transfer/reflux. Cosmology supplies
comoving/canonical/scale-factor adapters; astrodynamics supplies physical velocity,
epoch, frame, encounter, and mission adapters; astrophysics supplies concrete
instrument/sky response. Application-local nominal FMM/TreePM and duplicate response
or covariance implementations are removed rather than maintained beside the core.
Astrodynamics applications reuse the existing differential, geometric, particle,
rigid-body, hybrid-event, nonlinear, control, likelihood, and inference substrates.
Exact astronomical time routes, IERS Earth orientation, compiled frame graphs,
pinned ephemerides/interchange, high-fidelity forces, universal/analytical/DSST/IAS15
propagation, bounded events and maneuvers, encounter/hierarchical gravity, coupled
vehicles, tracking, variational dynamics, OD, access, targeting, and conjunction
products remain dense fixed-capacity plans. Astrophysical observation applications
add WCS/calibrated imaging, surveys, scalar/polarized transfer, waveform/QNM networks,
oblate occultation, and finite-source microlensing without introducing a second
observation or inference runtime. External provider calls and file access never enter
traced execution.
Material point dynamics compose that transfer with PIC/FLIP/APIC families,
USF/USL-minus/classical/affine/post-advection MUSL, adaptive realization and replay,
isotropic or general plane stress, J2 and pressure-dependent geomechanics, porothermal
field operators, simultaneous K-way and rigid contact, moving-domain and compact
implicit actions, diffuse or sharp fracture, active/compact block storage, distributed
ownership, particle lifecycle and ratio-two AMR. Commercial claim tuples, intended use,
durable checkpoints/output, host supervision, event/topology journals, derivative
taxonomy, standards traceability, and G0--G7 release evidence close each exact supported
configuration without implying universal validation or certification.
DEM adds stable compositional
normal/cohesion/tangential/rotational history, accepted-step
work/energy ledgers, cached and fused neighborhoods, DMT/capillary/lubrication,
elastic rolling–torsion, plasticity, multicontact correction, SO(2)/SO(3)
bodies, clumps, triangle/convex/implicit/superquadric geometry, wall traction
and wear, bonds/topology events, and certified sensitivity modes. Rigid mechanics
adds globally coupled planar/spatial fixed, ball, hinge, prismatic, and distance
joints; compliant/dissipative laws and motors; unilateral limits; hard restitution
and exact Coulomb-cone contact; breakage; fixed-capacity topology transactions; and
physical residual/rank/energy/branch certificates. Radial particle
conversion adds typed thermochemistry, reactions, evaporation,
shrinking-core conversion, morphology, conservative continuum/contact/radiative
exchange, fixed-pool process events, and atomic reactive CFD–DEM scheduling. See
[Guide → Particle methods](guides_particle_methods.md),
[Guide → Particle-grid splatting](guides_particle_splatting.md),
[Guide → Material point method](guides_material_point_method.md),
[Guide → MPM schedules](guides_mpm_schedules.md),
[Guide → MPM constitutive extensions](guides_mpm_constitutive_extensions.md),
[Guide → MPM contact and fields](guides_mpm_contact_fields.md),
[Guide → MPM particle domains](guides_mpm_particle_domains.md),
[Guide → MPM adaptive and implicit](guides_mpm_adaptive_implicit.md),
[Guide → MPM fracture and sparse storage](guides_mpm_fracture_sparse.md),
[Guide → Discrete element method](guides_discrete_element_method.md),
[Guide → Wet granular contact](guides_wet_granular_contact.md),
[Guide → Superquadric DEM](guides_superquadric_dem.md),
[Guide → Particle internal transport](guides_particle_internal_transport.md),
[Guide → Particle thermochemistry](guides_particle_thermochemistry.md),
[Guide → Reactive CFD–DEM](guides_reactive_cfd_dem.md),
[Guide → DEM rigid bodies](guides_dem_rigid_bodies.md),
[Guide → Constrained rigid-body dynamics](guides_constrained_rigid_bodies.md),
[Guide → Extended constrained and deformable mechanics](guides_extended_mechanics.md),
[Guide → Differentiable DEM](guides_differentiable_dem.md), and
[Guide → CFD-DEM coupling](guides_cfd_dem.md),
[Guide → Smoothed particle hydrodynamics](guides_sph.md),
[Guide → Weakly compressible SPH](guides_wcsph.md),
[Guide → SPH boundaries](guides_sph_boundaries.md), and
[Guide → Multiphase and incompressible SPH](guides_multiphase_incompressible_sph.md).
PINNs participate through trial/residual records rather than a fabricated mesh. See
[Guide → Discretization](guides_discretization.md),
[Guide → Isogeometric analysis](guides_isogeometric_analysis.md),
[Guide → Explicit polygon H1](guides_explicit_polygon_h1.md),
[Guide → Virtual elements](guides_virtual_elements.md),
[Guide → Global spectral methods](guides_spectral_methods.md), and
[Guide → Solver substrates](guides_solver_substrates.md).

Native partitioned multiphysics coupling compiles exact participant ports and
forward/paired-adjoint `FieldTransfer` exchanges into deterministic graph stages.
Explicit Jacobi/Gauss–Seidel sweeps remain distinct from physically certified
implicit interface roots. Candidate/accepted rollback, fixed-window replay,
fixed-grid waveform exchange, subcycling, resource evidence, and implicit root
sensitivities reuse the existing solver, discretization, and nonlinear substrates.
No communication, mesh, or fallback-solver stack is introduced. See
[Guide → Partitioned multiphysics coupling](guides_partitioned_coupling.md) and
[API → Solver → Partitioned coupling](api/solver/coupling.md).

### Atomistic learning and conservative dynamics

`phydrax.atomistic` specializes the existing material-particle and `GraphIR`
substrates for scale-identified finite molecules. `AtomicStructure` and
`AtomisticBatch` retain stable atomic IDs, masses, padding masks, and explicit
length/energy identity. Case-isolated dense candidate graphs expose
displacement, distance, direction, masks, and neighbor work under mandatory
atom-count and neighbor-capacity guards; overflow invalidates the result without
edge truncation. `phydrax.nn.atomistic.PaiNNPotential` provides invariant
scalar/equivariant vector interactions. The drop-in `NequIPPotential` adds
species-conditioned self connections, parity-safe gates, degree-zero-through-two
edge features, and independently derived Cartesian O(3) tensor products whose
legal instructions, radial parameter count, work, resource limits, and identity
are planned before allocation. Both models mask padded nodes and edges, use a
smooth cutoff, and sum invariant per-atom energy. Forces are only the negative
position derivative of that scalar energy with frozen candidate topology.
Prediction evidence includes validity/status, scale and precision identity,
graph provenance, and net-force/net-torque defects.

Domain-specific training accepts energy-only, force-only, or joint supervision,
fits loss normalization from the training split only, and reuses the shared key,
callback, selection, patience, and deterministic-continuation lifecycle. The
offline rMD17 utility parses only local NPZ data and fingerprints disjoint split
indices; the campaign tool compares matched PaiNN and NequIP runs across seeds.
The existing learned models retain their finite nonperiodic training and prediction
scope unless explicitly wrapped with periodic graph execution; execution capability
does not certify rollout stability. The atomistic runtime additionally provides complete
unit systems, interaction-site coordinate maps and differentiable virtual sites, stable-ID
topology, native force-field bundles and adapters, dense/cell/Verlet and distributed
execution, constrained NVE/NVT/NPT and rigid dynamics, polarization, implicit solvent,
quantum-nuclear propagation, many-body and soft-matter models, H5MD/XYZ reporting and
rerun, MDAnalysis, i-PI and PACKMOL boundaries, collective variables, adaptive biases,
replica exchange, free-energy estimators, and committee uncertainty/acquisition. State,
labels, bias history, transport resources, and analysis frames remain separate typed
contracts. Every capacity, convergence, protocol, or physical failure is typed and
fail-closed. See [Guide → Atomistic learning](guides_atomistic.md),
[Guide → Atomistic dynamics](guides_atomistic_dynamics.md),
[Guide → Atomistic force fields](guides_atomistic_force_fields.md),
[Guide → Atomistic interoperability](guides_atomistic_interop.md),
[Guide → Enhanced atomistic sampling](guides_atomistic_sampling.md), and
[API → Atomistic learning and dynamics](api/atomistic.md).

### Advanced biophysics

Advanced biophysics is a composition of canonical numerical owners rather than a
parallel simulation engine. `phydrax.stochastic.path_sampling` adds fixed-capacity
path-space ensembles, exact proposal accounting, TPS/TIS/RETIS workflows, reactive
rates, committors, uncertainty, and identity-bound restart. It accepts only dynamics
with declared path-density or qualified reversible-map semantics and deterministically
rejects failed propagation without hidden retries.

`phydrax.applications.electrophysiology` compiles stable-ID cable morphologies,
membrane mechanisms, stimuli, recordings, ion pools, synapses, plasticity, and
stochastic channel transitions into residual-checked implicit tree solves.
`phydrax.applications.cellular_mechanics` provides energy-derived Helfrich membranes,
surface chemistry, fluid-transfer composition, topology-epoch remeshing, polygonal and
polyhedral vertex tissues, dynamic particle relations, chromatin loop extrusion, actin
networks, motors, and focal adhesions. Discrete binding and topology changes remain
candidate/evaluation/commit transactions; derivatives are conditional on the accepted
fixed event program.

The atomistic owner additionally supplies residual-gated matrix-free polarization,
prepared alchemical endpoint mappings, elastic networks, differentiable external
fields, and fixed-capacity distributed decomposition. Systems-biology plans compile
compartmental stoichiometry, gene-expression kinetics, source evidence, and atomic
multirate process coupling. Observation plans keep latent physics separate from
fluorescence, correlation, lifetime, FRET, channel, and current-voltage measurement
models. See the Advanced biophysics guides in the navigation for the exact support and
qualification boundaries.

### Experimental velocimetry from images to trajectories

`phydrax.velocimetry` keeps particle image velocimetry, dense image displacement,
particle tracking velocimetry, and residual-image Lagrangian refinement
scientifically distinct. Classical PIV plans prepare mask-aware FFT correlation,
extended search, deterministic peak evidence, multipass image deformation,
validation, optional non-mutating replacement, and calibrated physical
conversion. Camera rigs expose pinhole/distorted/refractive projection and rays;
robust calibration, conflict-free multi-view association, triangulation, temporal
tracking, and smoothing retain frames, identities, covariance, failures, and
capacity evidence.

Radiometric particle-image formation supports deterministic synthetic
qualification and continuous Shake refinement without reusing conservative
particle-grid deposition semantics. An optional native learned backend shares
only the neutral dense image-displacement contract. Canonical archives and
explicit-loss external adapters preserve zero versus invalid data, raw versus
filled values, coordinate transforms, and provenance. PIV fields can adapt to
compatible tensor grids and state-space observations; reconstructed PTV tracks
can adapt to `TrajectoryData` with explicit gap resets. See
[Guide → Velocimetry](guides_velocimetry.md) and
[API → Velocimetry](api/velocimetry/index.md).

### Computational topology: exact invariants and filtered fields

`phydrax.topology` consumes the canonical oriented cell complexes above without
introducing another mesh or graph representation. Compact active layouts, algebraic
subcomplexes and relative pairs, exact prime-field homology, exact rational Betti
dimensions, validated lower/upper-star filtrations, and ordinary or induced-relative
persistent homology retain topology, entity, coefficient, resource, and reduction
evidence. Natural host diagrams pack explicitly for JAX; frozen pairings expose local
endpoint derivatives only while the complete filtration order remains valid.

Exact rational dimensions bridge to metric harmonic cochains through independently
verified rank, orthonormality, residual, and spectral-gap evidence. Solvers still own
whether a harmonic class is a gauge, compatibility obstruction, or physical
circulation/flux mode. See
[Guide → Computational topology](guides_computational_topology.md).

### Precision is an execution contract

Precision is attached to executable stages, not inferred from one global dtype.
Finite differences separate coefficient storage, field storage, accumulation,
certification, communication, checkpoint, and output placement. Integration
separates integrand evaluation, reduction accumulation, adaptive/statistical
decisions, and output. Neural operators retain master parameters while creating
transient compute views, preserve geometry arrays, and record contraction and FFT
behavior. Spatial noise, predictive summaries, particle filters, and linear
solvers expose their own stage vocabulary.

Every supported policy resolves to content-addressed precision evidence. Parent
computations can retain child evidence—for example, an SPDE combines its spatial
discretization and noise-basis envelopes—and persistence compatibility includes
the effective precision contract. Static resource estimates use separate dtype
item-size assumptions; they are not presented as evidence that execution occurred.
Unsupported dtype placements fail during planning or preparation rather than
silently widening, narrowing, or changing real/complex kind.

### Differentiation: AD / jets / FD / basis

Differential operators support multiple backends (`backend="ad"|"jet"|"fd"|"basis"`) and autodiff modes
(`mode="reverse"|"forward"`). For deeper math, see [Appendix → Differentiation modes](appendix/differentiation_modes.md).

### Conditions: soft penalties vs enforcement by construction

Boundary/initial conditions can be handled in two ways:

- **Soft**: declare a boundary/initial condition, give it an integration source,
  and add its penalty term to `terms`.
- **Enforced**: build an ansatz \(\tilde u=\mathcal{H}(u)\) satisfying conditions
  exactly, then train on the remaining terms.

The enforced route is staged as boundary → initial → interior data. See:

- [API reference](api/phydrax.md)
- [API → Solver](api/solver/index.md)
- [Appendix → Physics-Constrained Interpolation](appendix/physics_constrained_interpolation.md)

### Exact PDE trial spaces

Finite Trefftz fields satisfy the homogeneous Laplace, polyharmonic, or
constant-wavenumber Helmholtz equation by construction and fit only boundary
conditions. Harmonic bases use a canonical exact-rational polynomial nullspace;
Almansi and plane-wave bases extend the same certificate/audit contract. Generic
hard boundary enforcement is rejected because it need not preserve the exact PDE
space. See [Exact PDE trial spaces](guides_exact_trial_spaces.md).

### Models: fields vs operators

- **Field learning**: learn \(u_\theta(x,t,\dots)\) directly (MLPs, separable models, etc.).
- **Operator learning**: learn \(G_\theta\) mapping inputs to fields, using a dataset factor \(\Omega_{\text{data}}\) so
  the domain becomes \(\Omega_{\text{data}}\times\Omega_x\times\cdots\). See [API → Domain → Composition](api/domain/composition.md)
  and [API → NN → Architectures](api/nn/architectures.md).

Trainable coordinates and their physical geometry are separate. Reusable
positivity, interval, symmetry, packed skew symmetry, semidefinite/definite
factorization, stability, and orthogonality maps live in
`phydrax.nn.parameters`; raw arrays remain optimizer leaves and physical values
are constructed on demand. The same package owns explicit model-PyTree
selection through `ParameterSubspace`.

Finite-width empirical neural tangent kernels reuse the same selected parameter
PyTrees, prepared linearizations, vector spaces, matrix-free operators, and
spectral diagnostics as the wider linear-algebra runtime. Functional residual
kernels additionally retain exact integration coefficients, masks, enforcement,
complex realification, and named residual blocks. The functional training
runtime keeps authored physical terms separate from pseudo-transient, causal,
and dynamically balanced optimizer surrogates; independent fixed evaluation
terms own model selection. Exact accepted-update checkpoints, named sample-axis
sharding, physical time windows, and defect correction preserve those
identities rather than introducing a parallel PINN framework. Native SOAP
supplies bounded per-axis Shampoo covariances and Adam moments in adaptive
orthogonal bases through the same Optax-compatible solver boundary; its complete
basis and moment state participates in exact resume. See
[Functional training runtime](guides_functional_training.md),
[Neural tangent kernels](api/nn/neural_tangent.md), and
[API → Optimization → SOAP](api/optim.md#stateful-orthogonal-adaptive-preconditioning-soap).

`NeuralGalerkinProblem` evolves a selected model subspace as a Diffrax parameter
ODE. Fixed physical integration realizations define the field metric; rectangular
least squares or a damped empirical Gram solve supplies the tangent rate. The result
retains ordinary Diffrax evidence plus independent saved-node projection audits and
reconstructs valid parameter states as named fields. Backward characteristic maps
reuse the same Diffrax substrate before optional macro-step field projection.

Exact native `Linear` paths can instead carry factorized low-rank updates over
a shared dense base. The factor-only `ParameterSubspace` is the complete
gradient and optimizer-state boundary for `fit_operator` and Optax
`FunctionalSolver` runs; merging returns an ordinary dense deployment model,
while adapter-only archives verify the complete base content before loading.
Standard and rank-stabilized scaling are explicit per site. RWF layers adapt
their unscaled `V` coordinate before the frozen row scale and remain RWF after
deployment merging.

The checked transfer campaign improves its frozen baseline with 53 selected
parameters versus 197 for full fine-tuning, and the resource campaign reduces
Adam state from 4,194,308 to 131,076 bytes with merged/factorized disagreement
below \(7\times10^{-15}\).

### Native ML: fitted array models, not mutable estimators

`phydrax.ml` covers preprocessing and composition; linear, generalized-linear,
robust, sparse, discriminant, Bayes, and calibrated supervision; decomposition;
kernel, neighbor, covariance, mixture, clustering, manifold, outlier, tree, and
ensemble methods; selection, metrics, inspection, artifacts, and audited
conversion.

An immutable recipe plus `MLBatch` produces a `FitResult` containing a
solver-frozen `AbstractArrayModel`, fit diagnostics, validity/status, the resolved
method, and a per-input `GradientContract`. Exact discrete algorithms and smooth
relaxations have separate types. Dense-only recipes reject sparse storage rather
than allocating silently. The resulting model uses the same `ModelBinding` as
neural models, so it can remain a fixed domain closure or be explicitly unwrapped
as a trainable warm start. See [Native machine learning](guides/ml.md), the
[scientific ML workflow](cookbook/native_ml.md), and the
[complete ML API](api/ml/index.md).

### Irregular sequences: invariant affine recurrence

`phydrax.nn.operator.architectures.DiagonalStateSpaceMixer` is the
input-independent diagonal continuous-time baseline.
`SelectiveStateSpaceMixer` adds input-dependent positive step scaling, injection,
and readout while preserving an affine latent recurrence. Both use exact
zero-order-hold or linear interval integration on physical schedules and share
serial/associative execution semantics. The selective model additionally accepts
declared packed-segment resets and reports extrapolation diagnostics. Capability
metadata records both implementations as research status.

### Uncertainty: stochastic functions, processes, inputs, and observations

`phydrax.uq` keeps epistemic, uncertain-input, observation, stochastic-process,
and numerical axes explicit in named `PredictiveField` results. NUTS/HMC, Laplace
approximation, deep ensembles, and Gaussian-process discrepancy models produce
coherent epistemic draws. Scalar exact/FITC and computation-aware inference,
correlated heterotopic outputs, and linear-functional value/PDE observations share
the covariance-safe `phydrax.kernels` PyTree algebra. Computation-aware scalar
factors use native dense or sparse linear actions, bounded kernel-row workspaces,
small projected Cholesky solves, and explicit resource/conditioning evidence while
retaining unresolved action directions in covariance. Exact scalar GP inference
automatically selects weight space for a lower-rank finite-feature kernel; learned
feature maps and kernel hyperparameters remain differentiable leaves.

Finite stationary rational temporal kernels—including Matérn-3/2,
Matérn-5/2, SHO, stable CARMA, finite sums, repeated/derivative observation
rows, and prepared finite separable spatial designs—compile to exact
finite-dimensional continuous linear Gaussian models. One stable sorted
schedule shares train/query overlaps and repeated queries, while real
observation masks represent query-only and missing training positions.
Sequential square-root Kalman filtering and reverse-scan RTS smoothing return
linear-storage query marginals and the exact active-observation log marginal
likelihood. Hybrid short-gap/stationary long-gap process covariance stays
bounded on wide irregular schedules, and results retain query-scoped
status/masks, prepared/evaluated kernel and external schedule identity, method
provenance, evaluated parameters, and precision evidence. Repeated training
times, unsupported kernel algebra, mixed compute dtypes, approximate Laplace
sites outside their declared evidence, and parallel square-root execution are
rejected rather than repaired or silently approximated.

Matrix-free JVP/VJP propagation
transports diagonal, dense, low-rank, or operator-valued covariance through
scientific maps; normalized
errors-in-variables likelihoods account jointly for uncertain predictors and
observations. Probability domains, static random fields, and joint QMC propagate
full uncertain-input distributions. Global Wiener, Poisson-clock, composite, and
coefficient-process realizations provide replayable process paths.
Independent scalar Uniform and Normal inputs also support labeled nonintrusive
polynomial chaos. Deterministic guarded total-degree multiindices, normalized
Legendre/Hermite tensor bases, existing product-integration projection, and
diagnosed native exact/least-squares regression produce immutable PyTree- and
Field-preserving expansions. Mean, variance, and first/total Sobol effects follow
from orthonormal coefficient energy. Rank deficiency and nonfinite data fail
without silent pseudoinverse repair; this surface does not claim intrusive
stochastic Galerkin semantics.


Regular Bernoulli, Poisson, exponential-rate, and Normal families expose typed natural
and mean coordinates, normalized laws, weighted sufficient-statistic projection, and
exact matrix-free Fisher pullbacks. Boundary maximum-likelihood estimates and invalid
support or weight inputs remain explicit statuses rather than clipped parameters.

State-space inference binds each physical case and schedule step to one canonical
`StateSpaceStepContext`. `SampledStateSpaceInput` and `BSplineStateSpaceInput`
provide case-indexed exogenous signals with explicit support, breakpoint masks,
and stable input provenance rather than untyped callback payloads.
Euler--Maruyama transition kernels reuse canonical `ContinuousSystem` and
`WienerTerm` contracts, retain singular covariance support exactly, and expose
masked irregular-trajectory quasi-likelihoods without claiming exact SDE
likelihoods. Isothermal Port-Hamiltonian dynamics add the full
state-dependent Itô fluctuation--dissipation correction and a normalized
stationary Fokker--Planck diagnostic.
Complete-field Gaussian or conditional-flow operators define transition
marginals; typed Wiener/jump operator adapters define pathwise or composite
process transitions without pretending that independent marginal draws share
a path. Process diagnostics, calibration reports, shift matrices, and
retention gates keep raw results, statistical uncertainty, and provenance
explicit. See
[Guides → Uncertainty quantification](guides_uncertainty.md).

Gaussian inference uses `GaussianFactor` rather than silently converting every
covariance to a dense matrix. Rank, factor method, regularization, validity, and
status remain explicit through conditioning, nonlinear moment transforms,
expectation-only quadrature, and continuous-discrete filtering and smoothing.
First-order, scaled-unscented, spherical-radial, Gauss--Hermite, and keyed Monte
Carlo expectations are declared approximations; they do not make nonlinear
inference exact. Dense-only paths enforce dimension guards, and covariance inputs
are never silently repaired.

Integration-native Bayesian quadrature binds an explicit kernel-mean object to
its represented target. Exact embeddings cover supported interval kernels,
finite positive measures, finite real feature kernels, and the normalized
Gaussian squared-exponential route. Fixed and bounded sequential designs apply
prepared `phydrax.linalg` conditioning weights to scalar, array, field, or
PyTree integrands through ordinary `materialize`/`reduce` calls. Observation
noise and solve regularization remain distinct; target mismatch, non-finite
outputs, failed solves, resource overrun, and materially invalid posterior
variance fail closed. The reported Bayesian posterior standard deviation is
model/RKHS uncertainty, **not a deterministic or frequentist error bound**;
exactness depends on the declared embedding, and WSABI, unnormalized evidence,
and arbitrary undeclared kernels remain outside this surface.

The completed state-space surface also includes SING natural-gradient
variational smoothing for additive-noise latent SDEs; square-root sequential
Kalman filtering/smoothing; exact finite-state backward smoothing, Viterbi paths
and expected statistics; particle backward/full smoothing; ensemble smoothing;
Rao--Blackwellized filtering; and structural model compilation. SING retains its
Euler-discretized ELBO, expectation rule, Gaussian-chain execution method,
accepted steps, natural residuals, coherent path samples, and explicit
unsupported-model boundary. Physical cases, schedule masks, state/process
ancestry, stable IDs, validity/status, and input/method/backend provenance remain
present in results. Square-root Kalman execution does not support the parallel
method. Discrete particle ancestry and resampling choices are nondifferentiable.


Certified multidimensional cubature extends the same measure contract: private
polynomial data owns positive simplex, spherical, periodic, and radial reference
rules, while integration plans and geometry-owned cubature atlases map those
rules to physical targets. Reference exactness, physical Jacobians, content
identity, resource limits, and paired adaptive-triangle error evidence remain
distinct rather than being collapsed into a generic quadrature flag.

### Moment calibration and target-aware finite measures

`phydrax.weighting` computes the minimum-relative-entropy reweighting of a finite
prior subject to exact feature expectations, or reconciles uncertain and
unreachable expectations with a diagonal quadratic discrepancy. Dense,
`SparseLinearMap`, and compatible matrix-free moment actions share one moment-space
geometry path with explicit affine-rank, target-residual, regularity, optimizer, KL,
effective-sample-size, support, and provenance evidence. Exact success means a
finite regular relative-interior solution; boundary and inconsistent targets remain
typed failures.

`phydrax.integration.calibrate` applies that contract to a reusable realized
measure while preserving physical mass, axes, masks, ancestry, support validity,
and execution key. Calibration and coreset compression compose as an ordered
transformation chain, allowing correction before support reduction without
conflating feature preservation with a general integration-error bound. See
[API → Moment weighting](api/weighting.md) and
[Guides → Integrals and measures](guides_integrals.md#calibrate-a-reusable-finite-realization).

### Optimal transport: geometry between finite measures

`phydrax.transport` lowers integration-native finite targets and explicit realizations
while retaining physical mass, active support, event encoding, ground cost, and
provenance. Stabilized dense and blockwise balanced and unbalanced Sinkhorn solvers
return potentials, objective components, residuals, status, and matrix-free plan
actions. The unbalanced family declares independent source and target marginal KL
penalties and represents transported-mass collapse explicitly instead of silently
normalizing unequal mass. Debiased divergences, exact and sliced Wasserstein distances,
soft order operations, prepared references, spatial/intensity UQ metrics, scalar
terms, distributional semigroup losses, and deterministic particle transforms reuse
the native substrate. See [Guides → Optimal transport](guides_transport.md).

### Native combinatorial decisions and learning

`phydrax.combinatorial` separates logical finite decisions from the real feature
PyTrees dual to linear objective costs. Explicit catalogs, stable fixed-cardinality
selection, primal-dual Hungarian assignment, and signed-cost DAG shortest paths
share native JAX batching, deterministic ties, content-sensitive topology,
portable statuses, and independent feasibility/objective/optimality certificates.
Hard solves stop gradients by default. `BlackboxInterpolation` adds an explicit
loss-dependent one-extra-solve surrogate pullback without presenting it as a
classical solver Jacobian. See
[API → Native combinatorial optimization](api/combinatorial.md) and the
[combinatorial learning cookbook](cookbook/combinatorial_learning.md).

### Dynamical systems, identification, nonlinear analysis, and chaos

`phydrax.dynamics` separates local system laws, pathwise numerical evolution,
masked trajectory data, identification, and nonlinear analysis. `StateLayout`
retains physical shape, labels, and state geometry; `ContinuousSystem` and
`DiscreteSystem` retain optional typed inputs; `TimeGrid` and `IterationGrid`
keep physical-time and map-iteration normalization distinct. Solver, control,
stochastic, memory/delay, rough, and canonical evolution outputs enter the same
`TrajectoryData` contract through explicit adapters without losing masks, reset
boundaries, case/realization axes, or provenance.
`DifferentialAlgebraicSystem` adds a state-shaped implicit residual with declared
differential/algebraic component roles and independent state, rate, and residual
scales. Its prepared BDF solver supports fixed and adaptive accepted grids,
consistent segmented continuation, frozen-grid implicit replay derivatives,
checkpointed reverse passes, guarded numerical reuse, and local regularity evidence
without introducing a second identification representation.
Array models bind into `ContinuousSystem` as explicit trainable PyTree children.
Pointwise array models also bind as deterministic fixed-step `DiscreteSystem`
maps with an explicit coordinate-step contract. Neural identification reads
`TrajectoryData` through lazy validity/reset/control-aware windows and combines
supervised, deterministic reference-branch, and residual rollout objectives.
One authored recurrent step underlies full, prefix, chunked, rematerialized,
and resumed execution; Phydrax does not infer carry state from a JAXPR trace.
Structured Port-Hamiltonian fields provide state-dependent energy,
interconnection, dissipation, control, and forcing components while preserving
exact skew and semidefinite geometry; solver-owned isothermal dynamics add the
matching thermal diffusion without creating a second dynamics hierarchy.

Identification includes mask-safe DMD/DMDc and EDMD; strong, discrete, integral,
and weak SINDy; polynomial, Fourier, tensor-product, transformed, symmetry, and
custom feature libraries; STLSQ, SR3, temporally embargoed selection, and
ensembles; exact coefficient groups and equalities; implicit SINDy; and
structured-grid PDE-FIND. Coefficients are returned in named physical feature
coordinates. Ambient map conversion rejects non-Euclidean states unless a
geometry-aware identification method is declared.

Nonlinear analysis includes section crossings and return maps, multiple-shooting
periodic orbits, dense or matrix-free monodromy/Floquet analysis, resumable
finite-time Lyapunov spectra, covariant or adjoint directions, finite-size growth,
RQA, the modified 0--1 test, correlation dimension, surrogate significance,
explicit uncertainty-source aggregation, and a matrix-free shadowing-candidate
boundary. Bifurcation flags and statistical diagnostics are finite-resolution
evidence, not automatic certificates.

See [Nonlinear-dynamics cookbook](cookbook/nonlinear_dynamics.md) and
[API → Dynamical systems, identification, and chaos](api/dynamics.md).

### Controlled dynamics, estimation, and optimization

Differentiable driving-path classes and `solve_diffrax_cde` cover controlled
differential equations; `NeuralCDEVectorField` and `train_neural_cde` provide the
corresponding learned vector-field workflow. Path interpolation is explicit, so
causal, offline, piecewise-linear, and B-spline approximations are not conflated.
`solve_probabilistic_ode` returns calibrated Gaussian numerical uncertainty with
declared factorization, update, status, validity, and method provenance; it is a
probabilistic numerical ODE solver, not posterior uncertainty about an unknown
physical model.

`phydrax.control` composes typed time grids, control parameterizations, dynamics,
costs, and sampled constraints into trajectories with stable control,
discretization, approximation, method, and backend IDs. It includes
linearization and frequency response, Lyapunov/Riccati equations, Gramians,
finite- and infinite-horizon LQR, iLQR, dense multiple shooting, implicit
direct collocation, dense or structural-sparse prepared linear-control QPs,
explicit MPC warm-start shifting, and affine stage/terminal SOCP constraints.
`phydrax.control.games` adds finite-horizon affine linear-quadratic
full-state feedback Nash policies with explicit player control ownership,
per-player values, nonsymmetric dense-LU solves, diagnostic-only rank SVDs,
and independent curvature, stationarity, Bellman, conditioning, and causal
failure evidence.
Direct collocation accepts explicit systems or controlled state-shaped DAEs,
shared parameter coordinates, fixed or variable duration, exact sparse
derivatives, and explicitly selected dense-native, sparse-native, or sparse
Ipopt optimization. Its structured template supports numeric refresh, portable
primal/dual warm starts, and explicit completion pools for independent initial
decisions. Typed Ipopt evidence retains callback work, topology, status, and
warm starts; per-interval sampled defects drive nested h-refinement with
primal-only transfer; and controlled-DAE replay binds held controls into
consistency and every implicit stage. Fingerprinted native/Ipopt campaigns
derive graduation claims. Multiple shooting additionally lowers to the same
structured nonlinear IR without changing the original dense SQP contract.
Sampled nonlinear path constraints and off-grid audits are not continuous-time
certificates, and replay does not rewrite collocation success. Coefficient
search is bounded initialization, not a globally optimal solver. Dense
algorithms enforce dimension guards; no failed solve is hidden by fallback,
projection, covariance repair, or undeclared regularization.

Canonical LPs, QPs, and zero/nonnegative/SOC/rotated-SOC/PSD/exponential/power
product-cone programs live in `phydrax.optim`. PSD uses scaled upper-column symmetric
coordinates; EXP and POW use safeguarded JAX-native projectors and Moreau duals. Native
bounds remain separate from user constraint axes; typed methods, reusable
plan/prepare/bind/refresh lifecycles, strict warm starts, independent KKT/ray audits,
status, provenance, and regular projection-KKT sensitivities share one contract. Native
dense, QPax 0.1.4, optional MPAX 0.2.4, and optional Clarabel 0.11.1 methods remain
explicit choices with no automatic fallback or universal differentiability claim.

General nonlinear optimization lives in `phydrax.optim`. Scalar, block-residual,
proximal-composite, constrained, state/design, stochastic, manifold, factor
graph, and structured sparse programs share typed termination, status,
diagnostics, provenance, and certificate contracts.
`StructuredNonlinearProgram` separates fixed bound roles and sparse derivative
topology from refreshable numeric data. Native `PrimalDualInteriorPoint`
explicitly selects dense filter, matrix-free, or sparse augmented KKT
execution; structured methods return portable primal/dual warm starts and may
run input-ordered completion pools. Sparse LDLT through optional
Spineax/cuDSS remains an explicit provider with reported inertia and resource
release. Native methods also include Newton--Krylov, Steihaug--Toint,
dense/subspace dogleg and dogbox, robust-loss GN/LM, trust-reflective bounds,
variable projection, POUNDERS, projected quasi-Newton, augmented Lagrangian,
and filter/SOC SQP with BFGS/SR1/exact Hessians. BOBYQA, COBYQA,
deterministic multistart, and explicitly recertified SciPy/NLopt/Ipopt/Ceres
boundaries cover black-box and specialist routes. Residual graphs retain block
sparsity, Schur ordering, manifold retractions, and incremental factor versions.
Method of Moving Asymptotes adds a finite-box, feasible-start route for very
large designs with few inequalities. Its reduced state/design form reuses exact
adjoints and supports fixed-mesh SIMP compliance optimization with sparse
physical-radius filtering and mandatory independent reanalysis evidence.

Pin-jointed structural form-finding lives in
`phydrax.applications.solid_mechanics`. `ForceDensityStructure` compiles graph,
surface, external-ID, and coordinate or orthonormal affine-restraint topology.
Sign-definite tension and compression expose certified positive-definite systems;
mixed signs retain only self-adjoint evidence. Fixed nodal, reference line,
self-weight, current/reference traction, follower pressure, and volume-coupled
pneumatic loads share one component ledger. Linear and nonlinear plans preserve
input-tree, derivative, preconditioner, precision, and numeric-refresh identity;
same-topology cases vmap while disjoint graphs retain per-graph evidence.
Pure geometry/force observables compose in reduced or structured state/design
optimization. Rigidity spectra distinguish mechanisms and self stress, supplied
axial rigidities enable constitutive tangent stability, and scalar parameter
paths connect directly to continuation. Results retain member forces, reactions,
physical residuals, nested solver evidence, and stable plan identities; they do
not infer material behavior, buckling, or stability without the required
constitutive evidence.

Constitutive member-network verification consumes force-density geometry without
changing that boundary. Stress-free lengths, materials, physical section
families, translation/rotation DOFs, exact tension-only active sets,
corotational frames, discrete rods, and surface hinges define prepared elastic
equilibrium. Local/generalized buckling and continuation retain their assumptions;
prestress inversion includes fabrication, actuator, stability, and sequence
evidence; construction stages transfer immutable external IDs and reference
states; and continuous or finite-catalog sizing reports governing members and
cases. Required evidence aggregates to certified, failed, or incomplete—absence
never becomes structural safety.

Advanced evidence adds named generalized channels, sourced section frames,
connection/support mechanics, extensible catenaries, contact/friction,
nonuniform warping, fiber plasticity transactions, thin-walled GBT/finite-strip
modes, shell escalation, physical collapse events, dynamic stepping, exact
precedence search, standards clauses, reliability, calibration, evidence
acquisition, and immutable structural-twin ancestry. Each layer retains model
fidelity, applicability, generalized derivative, optimality-gap, uncertainty,
and data provenance rather than collapsing them into one safety Boolean.

Nonlinear algebraic systems live in `phydrax.nonlinear`. Certified scalar
bracketing, Newton/trust methods, chord and limited-memory Broyden, DF-SANE,
pseudo-transient continuation, vector Halley, Type-I/II Anderson, Steffensen,
and deterministic robust attempt graphs all terminate on the original physical
residual. Finite nonlinear updates carry traced nested budgets and fixed
component evidence; failed multiplicative components truly short-circuit.
Semismooth complementarity offers infeasible and projected feasible searches.
Scaling, model/direction/certificate precision, batched small kernels, explicit
sharding, and first/second-order solution maps remain declared rather than
inferred.

Generic parameterized residual curves and local bifurcation workflows live in
`phydrax.continuation`. Natural and pseudo-arclength continuation compose complete
nonlinear correctors, exact coordinate targets, explicit state/residual geometry,
canonical real-coordinate maps, adaptive curvature rejection, full-augmented event
localization, and dense or Krylov stability analyzers. Fold, Hopf, branch-point, and
pitchfork workflows separate numerical convergence from nullspace, transversality,
spectral, symmetry, and normal-form certificates; branch switching requires certified
geometry and a validated seed.

Lyapunov spectra for flows and maps, control-theoretic Gramian actions, implicit
Lyapunov/Riccati sensitivities, state-space score/Fisher actions, empirical
controllability/observability directions, and stationary linear-Gaussian spectra
share diagnosed validity and method provenance. Stationary spectra require a
stable nonsingular resolvent and positive-semidefinite supplied spectra; inputs
are rejected rather than clipped or repaired.

### Geometry: Euclidean coordinates vs metric-aware calculus

`phydrax.metrix` supplies charts and differentiable maps; tensors and compressed
differential forms; positive and signed metrics; affine connections and curvature;
finite real algebras; Lie groups; symplectic, Poisson, horizontal, complex, and G2
structures; metric-aware stochastic kernels; and immutable, measure-orthonormal
Laplacian spectra. Exact rational multiplication tables keep commutators, Jordan
products, associators, and left/right regular actions explicit. The canonical octonion
bridge derives the seven-dimensional cross product and G2 three-form from that same
table, while derivation preparation exposes numerical infinitesimal-symmetry evidence.
Graph and cochain constructors bind spectra to explicit topology, metric, boundary,
and entity provenance. Positive norms, Lorentzian wave operators, Poisson brackets,
sub-Laplacians, and nonassociative bracket trees remain distinct named operations
rather than overloads with hidden defaults. Bounds, seams, sampling, and admissibility
remain domain concerns. See [API → Metrix](api/metrix/index.md).

For trainable arrays on spheres, hyperbolic spaces, probability simplices, matrix
manifolds, SO(n), or SPD(n), `ParameterGeometry` binds exact PyTree leaf paths to
declared metrics. Weighted product metrics, Riemannian SGD and momentum, conjugate
gradient, and L-BFGS update those leaves through tangent conversion, retraction, and
transport while ordinary leaves remain Euclidean.
See [API → Optimization](api/optim.md#riemannian-optimization).

### Bounded capability closures

Finite-volume closure FVS-01–FVS-06 includes mapped-periodic viscous seams,
coupled multiblock positivity, strict MAC restart, bounded polyhedral geometry,
staged epoch transfer, moving-WLSQ and fixed-route remap derivatives,
generalized entropy, mapped/ALE hydrostatic balance, and
WENO/open/geostrophic/multilayer/Exner/shoreline/LPP mixed-precision routes.
These are single-device, fixed-topology or fixed-combinatorics derivative
envelopes rather than unrestricted distributed or topology-changing claims.

The bounded stochastic closure includes multiplicative and affine-Hausdorff
SING with explicit surrogate/audit semantics; finite coupled SPDE and
particle/sparse-grid/separated Fokker–Planck approximations; represented-positive
normalized densities and replayable stochastic boundaries; intrinsic
Stratonovich and fixed-route rough preparation; finite-degree Wiener-signature
certification; finite GW/assignment/Gaussian-component/learned transport;
prepared finite diffusion bridges; and measure-explicit
Riemannian/injective/conditional/eventful/hybrid/trajectory/finite-field flow
laws. It makes no claim of infinite-dimensional execution, generic
high-dimensional density solution, global GW/Monge optimality, exact mixture
W2, continuum bridge exactness, path-space density, or densities for
surjective/noninvertible routes.

LPP-01–LPP-09 adds public-JAX precision rewrite and finite-workload selection
evidence, portable sub-float32 formats and block-scaled contraction, local
optimizer-state compression, complete complex training interchange, batched
dense/sparse actions, and threshold-defined numerical inertia. Finite candidate
selection is not universal hardware optimality; numerical inertia is not a
symbolic proof; distributed compression, communication collectives, and
provider qualification are not implied.

UQI adds calibrated held-out MC-dropout intervals, frozen residual-noise
mappings, proper/improper complex Gaussian laws, SWAG/SVGP state, overlap-gated
Flow-NUTS, structured kinetic actions, scheduled SG-MCMC, audited minibatches,
bounded nested plans, dense-exact causal mass, and buffered particle evidence.
MC-dropout remains nonposterior uncertainty; the residual-weight mapping is
exact only for frozen positive quadratic coefficients; bridge evidence needs
overlap gates; stochastic geometry is an expected-log objective; and finite
steps, missed modes, and finite buffers remain explicit limitations. The clean
API uses
`kinetic=MCMCMassAdaptationPlan.diagonal()/blocks(...)/diagonal_low_rank(...)`
for `sample_hmc`, `sample_nuts`, and `sample_flow_nuts`, with no mass boolean.
Causal NUTS uses `sample_nuts(..., trajectory="causal",
causal_config=CausalNUTSConfig(...))`, fixed capacity
`2**max_num_doublings`, dense-exact recurrence residual gates, and ordinary
multinomial/U-turn selection over certified states only.

GTA-01–GTA-06 and GTA-08–GTA-13 provide bounded atlas, rank-strata, topology,
algebra, geometry-network, continuation, and certified-tail kernel products.
GTA-07 remains qualification-only: no K3/quintic checkpoint, downloader,
registry, format, or schema ships.

MPC-01–MPC-05 closes bounded particle/mechanics routes for mesh splats and
epochs, wet DEM reservoirs and stress evidence, LBVH/nonmatching hydroelastic
and Reynolds-film contact, wall-vortex injection and load recovery,
solver-owned hybrid-event replay, compressible augmentation, and atomic
capacity-limited SPH emission. Obsolete
`extract_hydroelastic_pressure_patch` and local replay/saltation names are not
part of the public surface.

The DAE/control/delay closure covers declared-incidence structural reduction,
bounded DAE resets and manifold stages, generalized-pencil/Hopf continuation,
case-axis iLQR, audited Radau/multiphase/complementarity/stochastic/manifold
transcription, represented-interpolant path certificates, finite coefficient-box
optimality certificates, one-Wiener-path stochastic adaptation, archived-primal
delay backsolve, and finite-realization or checked-tail memory. Structural index
claims are conditional on finite declared incidence, generalized spectra require
a square projected pencil, and whole-solve JIT requires static
segment/step/event maxima with fail-closed overflow.

QPV-01–QPV-08 covers finite positive-regulator path integrals, canonical
adaptive/source/geometry evidence, periodic/U(1)/exchange measures, root HMC and
incremental caches, adaptive/symmetric and finite-subspace Cayley TDVP,
resource-admitted electronic routes, PR #236 canonical `QuantumProgram`
measurement/control/tensor execution, finite CPTP maps, and finite
Fock/HEOM/compression/steady-state/identifiability certificates. It makes no
regulator-zero, unrestricted scaling/QED, curved overdamped reflection, or
unbounded-convergence claim.

Optimization/search/calibration provides sparse native and Clarabel conic
routes, bounded CVXPY/MPAX representations, finite reducers and mixed-integer
search, guarded differential evolution, typed calibration, ordering
surrogates, and prepared CSG continuation. It does not claim arbitrary CVXPY
atoms, MPAX callback cones, MINLP, or global nonsmooth derivatives.
The additional sparse route lowers `SparseStorage` directly to BCOO for MPAX
zero/nonnegative cones. Matrix-free conic JVP/VJP uses
`JacobianLinearOperator` with matching verified `StabilityLowerBound` and
selected-projection derivative evidence; conic calibration has canonical
exact/interval/group relative-entropy contracts; and KFAC exposes logical
block-axis, kind, complex-Cartesian, and sharing metadata with structured layout
lowering. The legacy private finite reducer is removed rather than aliased.


CID-01–CID-12 adds typed proof-carrying hard enforcement, immutable adaptive
signed populations, bounded discovery/cubature/Smolyak epochs, one validated
probability reference-transport surface, GTA-owned geometry-Jacobian cubature,
matrix-free scattered and mixed spectral reconstruction, lazy ragged execution,
and trainable/certified KAN knot transitions. Selection and topology changes
remain explicit nondifferentiable preparation boundaries.

The bounded GP/BQ/coreset closure adds the finite stationary rational and
separable state-space scopes above, finite-array-coordinate path functionals,
finite-candidate Monte Carlo qEI/feasibility with Monte Carlo standard errors,
explicit kernel-mean BQ, and moment/MMD or trajectory-block coresets. Lazy
fixed-size saved-state blocks never cross flattened path/case/realization
boundaries; selection retains positive weights, masks, and
`StochasticDriverSegmentReference` provenance, and canonical lowering preserves
driver/case/realization/coupling metadata in operator datasets. Path
observations are not Fréchet or topology derivatives, Bayesian optimization
carries no global-optimum claim, BQ uncertainty is model/RKHS uncertainty, and
trajectory blocks are dependent views rather than fabricated independent
paths. Kernel documentation covers `SHOKernel`, `CARMAKernel`, and
`SignaturePDEKernel` regularity inheritance.

Fixed-capacity per-collocation Diffrax quadrature participates in the canonical
`IntegrationPlan`/`materialize`/`reduce` lifecycle with solver identity and
failure evidence.

SNM-01–SNM-17 includes arbitrary-query wavelets and directional scattering,
point O(d) CNO, multi-source frames and coefficient flows, checksummed
first-party FNO/DeepONet weights, complete recurrence/rollout, boundary-aware
masked CNO/UNO, replayable Galerkin/characteristics, attention
replacement/anchors, modal discovery/recovery, generalized residual layouts,
transformed complex alias-aware low rank, constrained polyconvex/Onsager
wrappers, CID collocation with typed integral rewrite and target/causal
workflows, and soft/learned-cutpoint ordinal classification. Tier promotion is
excluded.

## A first real PDE example: Poisson on a square

This example trains a neural field \(u_\theta(x,y)\) to satisfy

$$
\Delta u = 4 \quad \text{in }\Omega=[-1,1]^2,\qquad
u = g \quad \text{on }\partial\Omega,
$$

with the analytic choice \(g(x,y)=x^2+y^2\) (so the exact solution is \(u^\star(x,y)=x^2+y^2\)).

*The configurations are kept small for demonstration purposes.*

!!! example
    ```python
    import jax.numpy as jnp
    import jax.random as jr
    import optax
    import phydrax as phx

    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )  # [-1,1]^2, label "x"


    # Exact solution / boundary target g(x,y) = x^2 + y^2
    @geom.Function("x")
    def g(x):
        return x[0] ** 2 + x[1] ** 2


    # Trainable field u_theta(x)
    model = phx.nn.models.MLP(
        in_size=2,
        out_size="scalar",
        width_size=16,
        depth=2,
        key=jr.key(0),
    )
    u = geom.Model("x")(model)

    layout = phx.domain.SampleLayout((("x",),))
    interior = geom.component()

    # Interior PDE residual: Δu - 4 = 0
    pde_condition = phx.conditions.Residual(
        "u", interior, lambda u: phx.operators.laplacian(u, var="x") - 4.0
    )
    pde_source = phx.integration.per_step(
        phx.integration.mean_over(pde_condition.on),
        phx.domain.PointSampling(64, layout=layout),
    )
    pde_term = phx.terms.ResidualPenalty(pde_condition, pde_source)

    # Soft Dirichlet boundary: u - g = 0 on ∂Ω
    boundary = geom.component({"x": phx.domain.Boundary()})
    boundary_condition = phx.conditions.Residual("u", boundary, lambda u: u - g)
    boundary_source = phx.integration.per_step(
        phx.integration.mean_over(boundary_condition.on),
        phx.domain.PointSampling(32, layout=layout),
    )
    boundary_term = phx.terms.ResidualPenalty(boundary_condition, boundary_source, scale=10.0)

    solver = phx.solver.FunctionalSolver(functions={"u": u}, terms=[pde_term, boundary_term])
    solver = solver.solve(num_iter=20, optim=optax.adam(1e-3), seed=0)
    ```

### Enforced boundary conditions (replace penalties with an ansatz)

Instead of penalizing boundary violations, you can enforce \(u=g\) **by construction** and train only on the interior
PDE term. This is often numerically cleaner: terms are separate from enforcement,
which maps \(u\mapsto\tilde u\).

!!! example
    ```python
    import jax.random as jr
    import optax
    import phydrax as phx

    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )


    @geom.Function("x")
    def g(x):
        return x[0] ** 2 + x[1] ** 2


    model = phx.nn.models.MLP(
        in_size=2, out_size="scalar", width_size=16, depth=2, key=jr.key(0)
    )
    u = geom.Model("x")(model)
    functions = {"u": u}

    layout = phx.domain.SampleLayout((("x",),))
    interior = geom.component()
    pde_condition = phx.conditions.Residual(
        "u", interior, lambda u: phx.operators.laplacian(u, var="x") - 4.0
    )
    pde_source = phx.integration.per_step(
        phx.integration.mean_over(pde_condition.on),
        phx.domain.PointSampling(64, layout=layout),
    )
    pde_term = phx.terms.ResidualPenalty(pde_condition, pde_source)

    boundary = geom.component({"x": phx.domain.Boundary()})
    boundary_condition = phx.conditions.Dirichlet("u", boundary, target=g)
    program = phx.enforcement.compile(
        functions,
        [phx.enforcement.EnforcementSpec(boundary_condition)],
        options=phx.enforcement.EnforcementOptions(num_reference=128),
        key=jr.key(1),
    )

    solver = phx.solver.FunctionalSolver(
        functions=functions, terms=[pde_term], enforcement=program
    )
    solver = solver.solve(num_iter=20, optim=optax.adam(1e-3), seed=0)
    ```

### Adding data (anchors / sensors) is “just another term”

Phydrax treats data fit the same way as PDE residuals: an observation condition
paired with an explicit finite integration source. For scattered anchor data
\(\{(x_i,y_i)\}\), construct the point batch directly:

```python
import jax.numpy as jnp
import phydrax as phx

# Continuing from the Poisson example above:
# - geom is the geometry domain
# - u is your trainable field

anchors = jnp.array([[0.0, 0.0], [0.5, -0.5], [-0.25, 0.75]])
interior = geom.component()
batch = interior.points({"x": anchors})
data_condition = phx.conditions.Observation("u", interior, g)
data_source = phx.integration.fixed(
    phx.integration.from_samples(phx.integration.mean_over(data_condition.on), batch)
)
data_term = phx.terms.ObservationPenalty(data_condition, data_source)
```

### Operator learning (dataset × coordinates)

To model operators \(G: f \mapsto u(\cdot)\), represent the domain as a product
\(\Omega=\Omega_{\text{data}}\times\Omega_x\times\cdots\) using `DatasetDomain`, and use a structured model like
DeepONet/FNO. See [API → Domain → Composition](api/domain/composition.md) and
[API → NN → Architectures](api/nn/architectures.md).

For row-indexed trajectories with a shared time step but different sequence
lengths, use `TrajectoryDatasetDomain` and `TrajectoryCaseDataTerm`. This keeps
each sampled time tied to the dataset row that owns it while still allowing time
residuals and other `DomainFunction` operators.

When a row has static covariates and observed ragged signals, keep those semantics
separate: put the static covariates in the `TrajectoryDatasetDomain` input row,
expose measured signals with `TrajectorySignal`, and supervise row-level targets
with `TrajectoryCaseDataTerm`. Observed trajectory signals and domain arrays are
JAX-traceable fixed state, not solver parameters.

If trajectory data must be exact, use the corresponding helper in
`phx.enforcement` to build a hard ansatz and train only the remaining physics
terms. Linear interpolation covers first-order time residuals; cubic-Hermite
interpolation covers second-order time residuals and optional selected output
components.

## Notation

We use $x$ for spatial variables, $t$ for time, $q$ for configuration, $v$ for
velocity, and $p$ for canonical momentum. $\mathcal J$ denotes the full optimized
functional, $L(q,v,t)$ a Lagrangian density, \(\mathcal S\) an action, and
$H(q,p,t)$ a Hamiltonian.

## By task: “what do I compose?”

Below are the common SciML regimes expressed in Phydrax’s primitives.

- **Forward PDE solve (PINN-style)**: interior residual + boundary/initial terms (soft or enforced).
  Start at [Getting started](index.md) and continue with the conditions-and-terms guide.
- **Neural eigenproblems**: use `VariationalEigenspace` to select the lowest
  self-adjoint trial subspace, `InvariantSubspaceResidual` to refine the strong
  equation `A U = B U H`, learned `FunctionSamples` trial spaces for amortized
  warm starts, or multi-state VMC for discrete quantum amplitudes. Eigenvalues
  come from the reduced Ritz problem; linear eigen-PINNs do not require a
  separately trainable scalar eigenvalue.
- **Integral and nonlocal field learning**: compose deterministic causal,
  spatial, or fractional operators inside ordinary residuals. Use
  `RandomizedMomentPenalty` when a squared moment is resampled rather than
  silently squaring one stochastic estimate. See the
  [integral-physics cookbook](cookbook/integral_physics.md).
- **Enforced BC/IC**: declare `EnforcementSpec` values with `phx.enforcement`,
  compile them into an `EnforcementProgram`, and pass that program to the solver.
  See [API reference](api/phydrax.md).
- **Data assimilation / hybrid physics-data**: pair continuous `Observation`
  conditions with finite sources and `ObservationPenalty`; use likelihood-backed
  binary, multiclass, multilabel, or ordinal classification terms for discrete
  observations. Soft-target and focal terms remain explicit non-posterior scores.
  Dense-site, ragged-trajectory, and graph classification preserve their native
  masks/measures; Dice/Jaccard/Tversky terms aggregate one support realization before
  forming an overlap ratio. Transform one raw logit `DomainFunction` to probabilities
  or a physical order parameter inside physics operators rather than duplicating the
  trainable model in `FunctionalSolver.functions`.
  See [API reference](api/phydrax.md).
- **Inverse problems (unknown coefficients/parameters)**: represent unknowns as additional fields or domain parameters, and couple them in residual operators.
  See [API → Domain → Functions](api/domain/functions.md) and [API reference](api/phydrax.md).
- **Operator learning**: use `DatasetDomain` and structured models on \(\Omega_{\text{data}}\times\Omega_x\). The canonical `OperatorBatch` path supports independent source/query discretizations across DeepONet, graph, geometry-informed, transformer, and spectral families; validate architecture choices with the audited benchmark protocol.
  See [Operator-learning cookbook](cookbook/operator_learning.md) and [API → NN → Architectures](api/nn/architectures.md).
- **Conditional-affine chemical operators**: certify a directional mass-action
  factorization with `ChemicalConditionalAffinePlan`, predict only auxiliary
  midpoint drivers, and reconstruct the complete physical species state from
  inverse-free phi-function reaction extents. One positive correction per base
  reaction is shared by forward and reverse channels. The research-tier
  `ChemicalConditionalAffineOperator` supports staged driver/correction fitting
  and explicit `DiscreteSystem` deployment without clipping or hidden fallback.
  See [Chemical kinetics](guides_chemical_kinetics.md) and
  [Operator-learning cookbook](cookbook/operator_learning.md).
- **Autoregressive operator learning**: bind one coincident physical state
  source and prediction with `OperatorRolloutRoute`. Training and deployment use
  the same authored step: raw output is physicalized, constrained, restored to
  the physical source, and reprocessed through source normalization before the
  next call. Named future targets and recurrent residuals support traced
  fixed-capacity horizons, while `final_batch` and an absolute step offset make
  chunk continuation and semantic keys explicit. Dynamic controls, independent
  queries, multiple recurrent states, and inferred carry are not accepted by
  this route.
- **Irregular-time sequence mixing**: use `DiagonalStateSpaceMixer` for an
  input-independent stable diagonal continuous-time baseline, or
  `SelectiveStateSpaceMixer` when input-dependent step, injection, and readout
  maps are justified. Both preserve exact zero-order-hold or linear interval
  integration and serial/associative parity; the selective route also exposes
  reset-aware packed segments and time-step extrapolation diagnostics.
  See [API → NN → Architectures](api/nn/architectures.md).
- **Stochastic neural operators**: declare state, duration, optional source-time,
  typed drivers, query, and output roles with `OperatorTransitionSpec`. Adapt a
  process-valued probabilistic operator with `OperatorMarginalTransition`, an
  additive Wiener operator with `OperatorPathwiseTransition`, a jump-conditioned
  operator with `OperatorJumpTransition`, or a mixed-driver operator with
  `OperatorProcessTransition`. Their rollouts produce canonical
  `StochasticTrajectory` and `PredictiveField` results with physical cases,
  process realizations, time, geometry, and provenance kept separate. Train or
  diagnose adjacent likelihood, direct-horizon likelihood, semigroup, cocycle,
  weak-generator, and nonlocal jump-generator contracts independently.
  See [API → UQ → Neural-operator uncertainty](api/uq/operator.md#process-consistent-operator-transitions).
- **Integral / conservation laws**: declare a moment condition, choose
  `mean_over(component)` or `over(component)`, and attach its source to a
  `MomentPenalty` term.
  See [Guides → Integrals and measures](guides_integrals.md).
- **ODEs, SDEs, Lévy/rough/memory equations, interacting particles, and
  semidiscrete SPDEs**: either learn a trajectory by enforcing
  \(\dot u-f(u,t)=0\), or integrate an explicit finite-dimensional problem.
  Brownian, Poisson, stable Lévy, and fractional Gaussian realizations own
  replayable global randomness rather than acting as local seed wrappers.
  Native solvers cover Itô/Stratonovich SDEs, finite-activity jump and hybrid
  systems, truncated or Gaussian-closed stable Lévy equations, step-two rough
  equations, stochastic Volterra and delay equations, and empirical
  McKean--Vlasov particles with idiosyncratic and common noise. Spatial systems
  combine a tensor or spectral discretization with finite-rank noise.
  Semilinear splits support exact compatible modal convolution, exponential
  Euler, and commutative exponential Milstein; general systems retain the
  Diffrax backend. Stochastic collocation provides a separate deterministic
  quadrature path for finite-dimensional random inputs.
  See [API → Solver → Differential equations](api/solver/differential.md).
- **Time integration and differential-algebraic equations**: preserve explicit,
  additive IMEX, implicit residual, second-order, partitioned, stochastic, and
  geometric equation forms. Capability-checked methods include Diffrax ERK/ESDIRK/ARK,
  native SSPRK, endpoint theta, BDF1--BDF5, matrix-free Rosenbrock-W,
  generalized-alpha, partitioned RK, Gauss--Legendre IRK, and geometric/exponential
  families. Native residual solves retain consistent initialization, implicit
  derivatives, replay, continuation, local regularity, and complete attempt evidence.
  See [API → Solver → Time integrators](api/solver/time_integrators.md) and
  [API → Solver → Differential-algebraic equations](api/solver/differential_algebraic.md).
- **System identification and equation discovery**: normalize canonical
  evolution, differential/delay/memory/rough, controlled, or stochastic output
  as `TrajectoryData`; preserve sample/transition masks and reset boundaries;
  then choose DMD/EDMD, a strong/discrete/integral/weak SINDy formulation,
  structured or implicit sparse regression, or structured-grid PDE-FIND.
  Results retain design rank, conditioning, residuals, convergence history,
  physical coefficient names, source IDs, and all rejected selection or
  ensemble evidence.
  See [Nonlinear-dynamics cookbook](cookbook/nonlinear_dynamics.md) and
  [API → Dynamical systems, identification, and chaos](api/dynamics.md).
- **Nonlinear dynamics and chaos**: evolve one declared flow or map; extract
  sections and return maps; solve periodic orbits and Floquet spectra; continue
  equilibrium or orbit residuals through folds; or evaluate Lyapunov spectra,
  covariant directions, finite-size growth, recurrence statistics, the 0--1
  test, correlation dimension, and surrogate significance. Every path records
  its grid, estimator assumptions, masks, fit/Theiler/burn-in windows, RNG, and
  numerical convergence evidence. Aggregate initial-condition, parameter,
  path-noise, and numerical cases only through explicitly named uncertainty
  axes.
  See [Nonlinear-dynamics cookbook](cookbook/nonlinear_dynamics.md) and
  [API → Dynamical systems, identification, and chaos](api/dynamics.md).
- **Controlled differential equations and Neural CDEs**: select an explicit
  differentiable driving path, integrate with `solve_diffrax_cde`, or train a
  `NeuralCDEVectorField` with `train_neural_cde`. Offline cubic interpolation is
  noncausal; causal backward-Hermite, piecewise-linear, fixed B-spline, and
  callable paths declare distinct interpolation and derivative contracts.
  See [Controlled-dynamics cookbook](cookbook/controlled_dynamics.md) and
  [API → Solver → Differential equations](api/solver/differential.md).
- **Probabilistic numerical ODEs**: use `solve_probabilistic_ode` when numerical
  integration uncertainty is part of the result contract. Gaussian factors,
  calibration, step status, masks, and method/factorization provenance stay
  explicit. This numerical uncertainty is not a physical-model posterior.
  See [API → Solver → Differential equations](api/solver/differential.md).
- **Coupled estimation and rare events**: declare refinement axes and
  coarse/fine transfers in a `StochasticCouplingPlan`, run paired levels with one
  realization, and allocate multilevel Monte Carlo work from measured
  correction variance and cost. Estimator state, checkpoints, and result
  archives preserve hierarchy and sampler identities. Canonical path events
  drive stopping diagnostics and adaptive multilevel splitting; Girsanov and
  jump compensator changes expose explicit path weights. A Smolyak surrogate can
  enter the same hierarchy as a paired control level.
  See [API → Integration](api/integration.md) and
  [API → Stochastic processes](api/stochastic/index.md).
- **Martingale and stopping-time validation**: declare observables and generator
  actions with `MartingaleProblem`, then evaluate interval or stopped
  martingale increments, predictable brackets, quadratic variation, and
  finite-activity jump compensators. Statistical reports use realization
  independence clusters rather than treating coupled paths as independent.
  See [API → Stochastic → Martingales](api/stochastic/martingales.md).
- **Optimal control, QPs, and MPC**: compose `ControlProblem` from a typed grid,
  dynamics, parameterization, costs, and sampled constraints; use LQR,
  case-axis iLQR, compiled linear-control QPs, bounded coefficient search, dense
  multiple shooting, direct collocation, or receding-horizon MPC according to
  the problem structure. `TrajectoryOptimizationProblem` adds controlled
  implicit DAEs, bound-form global constraints, shared optimized parameters,
  and variable duration for direct collocation. Per-interval audit evidence
  supports explicit nested refinement; represented interpolants can carry
  separate continuous path certificates; controlled DAEs can be causally
  replayed through a held input policy; and structured Ipopt results retain
  typed work/KKT/warm-start evidence. Results retain case/control axes,
  validity and backend status, plus control, discretization, approximation,
  method, and backend IDs. A certificate covers only its represented
  interpolant or finite coefficient box; replay remains independent evidence;
  multiple shooting remains single-case; and bounded search is not globally
  optimal. Dense paths enforce guards and never hide failure behind repair or
  fallback.
  See [Control cookbook](cookbook/control.md), [API → Control](api/control.md),
  and [API → Optimization](api/optim.md).
- **Linear systems, sensitivities, and spectra**: linearize dynamics; solve
  Lyapunov and Riccati equations; compute LQR policies, Gramian actions, frequency
  responses, flow/map Lyapunov spectra, state-space score/Fisher actions, and
  stationary linear-Gaussian spectra. Each path reports its stability,
  singularity, validity/status, regularization, and method/backend provenance.
  Dense dimension guards and explicit stability/positive-semidefinite
  requirements apply; no hidden clipping or repair is performed.
  See [API → Control](api/control.md), [API → UQ → Global sensitivity](api/uq/sensitivity.md),
  and [API → Solver → Differential equations](api/solver/differential.md).
- **Filtering and smoothing**: compose a state prior, transition kernel,
  observation model, masked schedule, and optional typed exogenous signal in
  `StateSpaceProblem`. Every transition and observation receives one
  context-last `StateSpaceStepContext`; sampled and B-spline inputs preserve
  endpoint values, breakpoint masks, internal-time evaluation, support, and
  `input_id`.

  Linear-Gaussian paths include sequential or parallel covariance-form Kalman
  filtering, sequential square-root filtering, and matching RTS smoothing.
  Square-root execution does not support the parallel method. Exact finite-state
  inference includes backward smoothing, Viterbi paths, transition counts, and
  expected sufficient statistics. Particle, ensemble, Rao--Blackwellized, and
  conditional-SMC paths include fixed-lag/full/backward smoothers and posterior
  simulation or MCMC where declared. Particle ancestry and resampling remain
  discrete and nondifferentiable.

  `GaussianFactor`, conditional moments, declared nonlinear Gaussian transforms,
  and continuous-discrete Gaussian filtering/smoothing preserve rank,
  approximation, regularization, validity/status, physical cases, schedule
  masks, stable IDs, and solver/backend provenance. Dense guards apply.

  SING adds a Gaussian information-form variational smoother for
  Euler--Maruyama latent SDEs with full-rank additive diffusion. It supports
  differentiable non-Gaussian observations, irregular masked schedules, per-case
  natural-gradient backtracking, sequential or associative Gaussian-chain
  conversion, fixed-posterior model gradients, coherent path sampling, and
  portable result export. Its objective is an ELBO, not a relabeled marginal
  likelihood; multiplicative or singular diffusion has no silent fallback.
  Nonlinear moment propagation and sampled continuous-discrete observations are
  approximations, not exact inference, and no invalid covariance is silently
  repaired. Structural local-level, trend, seasonal, autoregressive, regression,
  deterministic-transition, and process-noise components compile into the same
  state-space contract.
  See [Filtering cookbook](cookbook/filtering.md),
  [API → Stochastic → State-space models](api/stochastic/state_space.md),
  [API → UQ → Filtering](api/uq/filtering.md), and
  [API → UQ → Inference and ensembles](api/uq/inference.md).
- **Finite-discrete probabilistic graphical models**: define stable named variable
  groups and dense, enumerated, Ising, Potts, logical, or cardinality factors over
  arbitrary hypergraph topology. Explicitly capped enumeration returns exact
  normalizers, marginals, and MAP states. Sum/max-product belief propagation is exact
  on forests and carries fixed-point-only evidence on loops; loopy sum-product labels
  its normalizer as Bethe. Validated chromatic Gibbs preserves exact hard support,
  clamping, persistent chain identity, and correlated-sample semantics. Parameter
  refresh never silently changes topology. See
  [Probabilistic graphical models](guides_probabilistic_graphical_models.md) and
  [API → Probabilistic graphical models](api/pgm.md).
- **Backward stochastic equations and semilinear high-dimensional PDEs**:
  evaluate terminal, local, and global BSDE residuals with explicit or
  autodifferentiated controls; fit one time-conditioned field from trajectory-node
  or query-conditioned Feynman--Kac labels; or alternate frozen labels and global
  optimization with Deep Picard iteration. Label batches retain conditional Monte
  Carlo errors and path-dependence clusters. Masked regularized least-squares,
  finite-activity compensated jumps, reflected path-dependent obstacles, empirical
  mean-field Hamiltonian control, and structured matrix-free nonlinear Picard
  sources have distinct declared contracts and diagnostics.
  `tools/high_dimensional_pde_benchmarks.py --suite methods` exercises the public
  query-conditioned label path and, with `--include-training`, the global Deep Picard
  training path. Its common result schema separates value/control, global-field,
  terminal, and estimator errors instead of treating unlike targets as one metric.
  See [BSDE cookbook](cookbook/bsde.md) and
  [API → Stochastic → BSDE](api/stochastic/bsde.md).
- **Static random fields and stochastic coefficient processes**: synthesize
  replayable Gaussian fields from a `SpatialNoiseBasis`, attach an explicit
  input role, and use stable mode IDs for deliberate cross-resolution coupling.
  `LatentGaussianCoefficientProcess` supplies reusable pathwise realizations;
  `LatentFlowJAXCoefficientProcess` supplies learned marginal transition laws.
  See [Guides → Uncertainty quantification](guides_uncertainty.md).
- **Curvilinear or manifold PDE/PINN**: define a `CoordinateChart` and
  `RiemannianMetric`, then use `riemannian_grad`, `riemannian_div`,
  `covariant_hessian`, or the metric overload of `laplace_beltrami`. Attach
  `sqrt(det(g))` to component integration with `with_riemannian_measure`.
  See [API → Metrix](api/metrix/index.md).
- **Continuous learned transport and flow matching**: sample independent or native
  balanced endpoint couplings, construct explicit endpoint interpolants, train a
  state-shaped conditional velocity with `FlowMatchingTerm`, and advance source-law
  samples through `ContinuousTransport` and an existing `DiffraxEvolution`.
  `ContinuousFlowLaw` adds exact finite-dimensional Euclidean density evaluation;
  keyed Hutchinson log-density estimates remain separate uncertainty-bearing
  diagnostics. Fixed-query field objectives can retain masks, physical quadrature,
  and channel geometry through `OperatorFlowMatchingMetric`.
  See [Guides → Optimal transport](guides_transport.md) and
  [API → Continuous learned transport](api/transport/continuous.md).
- **Score-based diffusion transport**: prescribe an exact VP or VE Gaussian
  perturbation, train a state-shaped marginal score with
  `DenoisingScoreMatchingTerm`, and reuse that field in either replayable
  `ReverseDiffusion` or `probability_flow_system`. Reverse samples retain distinct
  terminal states, global Wiener paths, solver status, and terminal-reference
  semantics. Probability flow composes with `ContinuousFlowLaw` instead of creating a
  second density implementation. Structured extensions add matrix and
  state-dependent Itô reversal, exactness-labeled conditioning, discrete Gaussian and
  categorical chains, Hausdorff subspace laws, coefficient-space field/path
  diffusion, intrinsic manifold and complex-coordinate semantics, and
  latent/graph/atomistic composition without erasing their distinct measures.
  See [API → Gaussian score diffusions](api/stochastic/diffusion.md),
  [API → Score diffusion transport](api/transport/diffusion.md), and
  [API → Advanced generative transport](api/transport/generative_expansion.md).
- **Stochastic PINNs, randomized residuals, and density equations**: use
  `phx.conditions.stochastic.Kolmogorov` for stationary or backward equations
  and `phx.conditions.stochastic.FokkerPlanck` for stationary or forward density
  equations, each paired with an explicit residual-penalty source. Exact
  factor-HVP contractions avoid dense Hessians. When exact coordinate sums are
  still too expensive, raw Hutchinson probes or unbiased coordinate sampling
  expose estimator uncertainty to signed U-statistic, independent-product, or
  biased plug-in residual estimators. PDE-IR compilation statically rejects
  nonlinear combinations that would bias randomized intermediates.

  For high-dimensional density evolution with simulable particles,
  `trajectory_state_time_samples` plus `ScoreMatchingTerm` learns
  \(\nabla_x\log p_t(x)\) without representing or normalizing \(p_t\). This produces
  a score field, not a reconstructed density. Probability-flux boundaries, strong,
  weak, and mild SPDE solution concepts remain separate explicit contracts.
  See [Stochastic-dynamics cookbook](cookbook/stochastic_dynamics.md),
  [API reference](api/phydrax.md), and
  [API → Operators → Differential](api/operators/differential.md).
- **Uncertainty quantification**: use NUTS/HMC or Laplace for explicit posterior
  problems, ensembles for neural-model epistemic variation, scalar or correlated
  Gaussian processes for model discrepancy, linear-functional GPs for operator
  observations, joint QMC for uncertain inputs, fixed-design Bayesian quadrature
  for a kernel-conditioned normalized Gaussian expectation, proper
  likelihoods/scores for observations, and conformal calibration for coverage.
  Bayesian quadrature posterior SD is not a deterministic/frequentist error
  bound. Use FITC only after dense scaling is measured.
  For repeated low-dimensional propagation under independent scalar Uniform or
  Normal laws, use nonintrusive polynomial-chaos projection or diagnosed regression;
  retain its coefficient moments and Sobol effects as finite-span evidence, not a
  truncation-error certificate.
  See [Guides → Uncertainty quantification](guides_uncertainty.md),
  [Guides → Integrals and measures](guides_integrals.md#fixed-design-bayesian-quadrature),
  [API → Positive-definite kernels](api/kernels.md), and
  [API → Uncertainty quantification](api/uq/index.md).
- **Transient and deformable mechanics**: solve transactional implicit Newmark
  volumetric FEM with material-state/admissibility ledgers; couple rigid bodies
  through exact interpolation/transpose attachment KKT blocks and mixed pressure
  gauges; evolve objective two-/three-dimensional Cosserat rods and triangular
  membrane/bending shells; construct exact-map collision surfaces, deterministic
  candidate epochs, conservative and certified trajectory/simplex bounds,
  smooth barrier/adhesive/friction closure, hard cone impact, mortar/Nitsche
  and mesh tying, hydroelastic/rough patches, multiphysics transport,
  route-state transfer, distributed ownership, and qualified derivatives; and
  retain explicit rigid–MPM coupling routes. See
  [Guide → Extended constrained and deformable mechanics](guides_extended_mechanics.md),
  [Guide → Deformable contact](guides_deformable_contact.md), and
  [Guide → Contact formulations](guides_contact_formulations.md).
- **Force-density structural form-finding**: build sparse or affine-restraint
  tension, compression, or mixed-sign networks; compose self-weight, traction,
  pressure, or pneumatic loads; optimize forces, supports, loads, gridshell
  planarity, or target geometry; and inspect mechanisms, self stress, constitutive
  tangent stability, per-case/per-graph evidence, and continuation branches. See
  [Guides → Force-density form-finding](guides_force_density.md) and
  [API → Force-density structural design](api/force_density.md).
- **Member-network structural verification**: supply stress-free references,
  materials, physical sections, cable unilateral laws, frame/rod/hinge bending,
  local/global buckling assumptions, construction stages, and sizing candidates;
  then aggregate equilibrium, prestress, sequence, and capacity evidence. See
  [Guides → Member-network structural verification](guides_member_network_structures.md)
  and [API → Member-network structural verification](api/member_network.md).
- **Advanced structural evidence**: analyze catenary/contact regimes, sourced
  section orientation, joint and support mechanics, warping and fiber plasticity,
  local/distortional buckling, collapse events, construction-order optimality,
  standards clauses, reliability, calibration, and evidence acquisition. See
  [Guides → Advanced structural evidence](guides_advanced_structural_evidence.md)
  and [API → Advanced structural evidence](api/advanced_structural.md).
- **Classical circuit networks and periodic analysis**: compose typed scattering
  networks, grounded MNA circuits, implicit device DAEs, operating points,
  descriptors, macromodels, noise, calibration, and field coupling. Harmonic
  balance plans fixed Fourier-collocation resources, prepares the native
  matrix-free nonlinear solve, and refreshes frequency and circuit coefficients
  without changing device equations. See
  [Guides → Circuit networks](guides_circuit_networks.md) and
  [Guides → Circuit periodic analysis](guides_circuit_periodic.md).
- **Lagrangian/Hamiltonian mechanics**: build Euler–Lagrange, canonical Hamiltonian,
  Poisson-bracket, or Hamilton–Jacobi operators on labeled state spaces.
  See [Guides → Lagrangian and Hamiltonian mechanics](guides_mechanics.md).
- **Quantum systems and dynamics**: construct composite states, explicit
  mixed-dimensional Hilbert layouts, local unitary/Kraus programs, reduced
  densities, information measures, matrix commutators, and closed- or
  open-system residuals. Dense local programs plan exact target routes and
  resource envelopes, refresh only numerical matrices under fixed structure,
  avoid global operator or superoperator promotion, and return unitarity,
  trace-preservation, and state physicality evidence. Connected discrete VMC
  and resource-admitted nonrelativistic finite electronic VMC share persistent
  MCMC, matrix-free score/Gram SR, statuses, diagnostics, and checkpoints.
  Periodic, no-pair, and stochastic-trace routes expose separate finite
  admission and truncation evidence rather than a global electron ceiling.
  Complex residual penalties remain real and nonnegative. See
  [Guides → Quantum operators and dynamics](guides_quantum.md),
  [Guides → Dense local quantum programs](guides_quantum_programs.md),
  [Cookbook → Variational Monte Carlo](cookbook/quantum_vmc.md), and
  [Cookbook → Open-system amplitude damping](cookbook/quantum_open_system.md).
- **Ritz/energy minimization**: use an explicit integral source with the
  appropriate term, with essential boundary conditions enforced in the ansatz.
  See [Cookbook → Mechanics and Deep Ritz](cookbook/mechanics.md).
- **Stochastic path expectation**: use Euclidean bridge kernels for imaginary-time
  propagation or Feynman–Kac diffusion paths for terminal PDE and reliability quantities.
  See [Euclidean path integrals and Feynman–Kac expectations](guides_path_integrals.md).
- **Cookbook recipes**: end-to-end patterns for field and operator learning,
  stochastic dynamics, filtering and smoothing, controlled differential
  equations, probabilistic inference, optimal control, QPs/MPC, mechanics, and
  quantum dynamics.
  Start at [Cookbook → Overview](cookbook/index.md).

## Where to go next

- [Cookbook](cookbook/index.md)
- [Advanced solver workflows](cookbook/advanced_solvers.md)
- [External solver backends](api/backends.md)
- [Continuation and bifurcation](api/continuation.md)
- [Domains and sampling](guides_domain.md)
- [Discretization](guides_discretization.md)
- [Solver substrates](guides_solver_substrates.md)
- [Differential operators](guides_differential.md)
- [Linear algebra runtime](api/linalg.md)
- [Einstein operations](guides_ein.md)
- [Metrix: differentiable geometry](api/metrix/index.md)
- [Positive-definite kernels](api/kernels.md)
- [Integrals and measures](guides_integrals.md)
- [Special functions and named integrals](guides_special_functions.md)
- [Euclidean path integrals and Feynman–Kac expectations](guides_path_integrals.md)
- [Lagrangian and Hamiltonian mechanics](guides_mechanics.md)
- [Quantum operators and dynamics](guides_quantum.md)
- Conditions and terms
- [Uncertainty quantification](guides_uncertainty.md)
- [State-space models and transition adapters](api/stochastic/state_space.md)
- [Dynamical systems, identification, and chaos](api/dynamics.md)
- [Nonlinear dynamics and chaos cookbook](cookbook/nonlinear_dynamics.md)
- [Controlled dynamics](cookbook/controlled_dynamics.md)
- [Control workflows](cookbook/control.md)
- [Control API](api/control.md)
- [Solvers and training](guides_solver.md)
- [API reference](api/phydrax.md)
- `phydrax.domain` for geometry, time, and sampling.
- `phydrax.sampling` for typed reference designs and capability inspection.
- `phydrax.sparse` for JAX-native relations, routing kernels, and sparse linear actions.
- `phydrax.linalg` for paired vector spaces, composable operators, linear
  problems, solve policies, reusable plans and factorizations, general eigensolvers,
  diagnostics, and backend provenance.
- `phydrax.backends` for explicit lazy PETSc, SLEPc, PyAMGCL, and NVIDIA AmgX
  lifecycle bridges with availability, transfer, convergence, and provenance evidence.
- `phydrax.metrix` for charts, tensors, metrics, curvature, and stochastic geometry.
- `phydrax.data_utils` for CSV loading, array scaling, and case-index splits.
- `phydrax.conditions` for residual, moment, observation, and physical conditions.
- `phydrax.terms` for penalty and specialized numerical/data terms.
- `phydrax.integration` for targets, sources, and reductions.
- `phydrax.weighting` for exact and quadratically reconciled moment calibration.
- `phydrax.special` for JAX-native named special functions and integral primitives.
- `phydrax.enforcement` for exact condition transforms.
- `phydrax.operators` for PDE operators.
- `phydrax.nn` for models, wrappers, and the generic diagonal state-space mixer.
- `phydrax.dynamics` for typed flow/map laws, pathwise evolution, trajectory
  data, DMD/EDMD, SINDy/PDE-FIND, periodic-orbit and chaos analysis, uncertainty
  aggregation, and the shadowing solver boundary.
- `phydrax.stochastic` for process paths, trajectories, typed state-space
  problems and inputs, transition kernels, exact signature and log-signature
  features, and structural model compilation.
- `phydrax.kernels` for covariance-safe stationary, algebraic, transformed,
  finite-feature, structured-input, signature-PDE, graph/Hodge spectral, compact,
  combinatorial, and fixed-noise noncompact kernels shared by GP and coreset methods.
- `phydrax.uq` for Gaussian factors and transforms, filtering/smoothing,
  state-space estimation, sensitivities, and stochastic spectra.
- `phydrax.combinatorial` for exact native finite, cardinality, assignment, and
  DAG path oracles, independent certificates, and explicit blackbox surrogate
  pullbacks.
- `phydrax.optim` for typed scalar, least-squares, proximal-composite, constrained,
  state/design, and stochastic optimization, differentiable solution maps,
  canonical QPs, and the explicit QPax backend.
- `phydrax.nonlinear` for nonlinear algebraic systems, fixed points,
  preconditioning, multigrid, variational inequalities, and implicit roots.
- `phydrax.continuation` for generic parameterized residual curves, stability,
  event localization, branch switching, and fold/Hopf/pitchfork workflows.
- `phydrax.control` for finite-horizon control, linear systems, LQR/iLQR,
  multiple shooting, compiled QPs, and MPC.
- `phydrax.solver` for training, differential, delay/memory, rough, stochastic,
  controlled, probabilistic, and geometry-preserving equation solvers.
