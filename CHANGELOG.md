# Changelog

## Unreleased

### Added
- Added JAX-native direct collocation for explicit controlled systems and
  input-aware state-shaped DAEs, with fixed/nonuniform or optimized-duration
  meshes, backward-Euler and midpoint transcription, interval controls, shared
  optimized parameter spaces, bound-form path and trajectory constraints,
  physical scaling, exact sparse Jacobians and optional Lagrangian Hessians,
  explicitly selected native-dense or low-level sparse-Ipopt execution,
  independent KKT recertification, typed decisions/layouts/results, and
  non-certifying off-grid defect audits.
- Added a first-class material-particle discretization with stable physical IDs,
  static active selections, physical mass measures, explicit precision/execution
  policies, periodic pair geometry, budgeted canonical dense neighborhoods,
  fail-closed fixed-capacity cell-list neighborhoods, exact solver-relation
  `GraphIR` views, and equal/opposite pair accumulation. Added normalized
  Wendland C2 and cubic-spline SPH kernels, a Tait barotropic energy closure,
  conservative fixed-h summation-density SPH, complete conservation and step
  diagnostics, and native separable-Hamiltonian compilation through
  `StormerVerlet`.
- Added first-order weakly compressible SPH with explicit summation- and
  continuity-density state layouts, pair-once continuity and conservative
  pressure operators, symmetric Morris physical viscosity, external
  acceleration power accounting, SSPRK33/SSPRK54 lowering, complete
  energy/dissipation diagnostics, periodic shear qualification, and dense versus
  cell-list scaling evidence.
- Added particle assemblies and bipartite relations, fixed-step programs and
  accepted-step transforms, geometry-derived wall particles, free-surface
  detection and atmospheric pressure policies, first-order kernel correction,
  delta-SPH diffusion, Monaghan artificial viscosity, Shepard density
  renormalization, transport velocity, adaptive smoothing length and grad-h,
  reciprocal multiphase WCSPH, and fail-closed IISPH/DFSPH projection steps with
  explicit residual and work evidence.
- Added explicit experimental/qualified/production/certified particle-method
  maturity, evidence-backed claims, dimensionless original density/divergence
  residuals, pressure and boundary constraint diagnostics, and a production gate
  that separates finite execution from numerical qualification.
- Added native multi-population cell execution, specialized batched 1D--3D
  local solves, safeguarded adaptive-h roots, production boundary and interface
  geometry evidence, stateful stabilization controls, assembled projection
  oracles and complementarity diagnostics, reference halo/migration semantics,
  benchmark registries, qualification artifacts, support matrices, and replay
  packets for commercial particle-method hardening.
- Made the finite-element local-action/workset program authoritative for scalar
  and product-space residuals; replaced split mixed subproblems with one compiled
  problem; added executable SIPG/Nitsche/nullspace handling, conservative upwind
  facets, Darcy, Maxwell, generated lowest-order primal HDG, and Taylor-Hood
  forms; connected high-order cell-local simplex/tensor and hexahedron Q1
  execution; and bound smoothed-elasticity actions to the same form program.
  Added accepted field/material/topology transactions, deterministic local T3
  marking/refinement/coarsening and transfer data, local DWR indicators, and
  executable phase-field, CPFEM, persistent-pair contact, fracture, and fixed-
  crack XFEM application workflows. Distributed execution remains out of scope.
- Added first-class exact-sampling round-sphere spectral discretizations with
  explicit S2FFT mode layouts, physical area measures, matrix-free
  Laplace--Beltrami actions, complete-degree real eigenbases and spatial noise,
  radius-aware addition-theorem kernels, resource/precision provenance, and one
  shared prepared-space contract for SFNO.
- Added native computation-aware scalar Gaussian processes with fixed, normalized
  block-sparse, and pseudo-input action policies; bounded sparse kernel-action
  contraction; reusable projected factors and low-storage conditioners; diagonal
  prediction; a full-data ELBO; numerical/resource diagnostics; and exact,
  conservative-covariance, differentiation, and scaling qualification gates.
- Added JAX-native static nested slice sampling over `PosteriorProblem` with
  weighted posterior quadrature, stochastic evidence-shrinkage uncertainty,
  insertion-rank and constrained-kernel diagnostics, semantic replay keys,
  portable checkpoints/results, and predictive integration.
- Added exact mathematical complex parameter interchange for dense and low-rank
  holomorphic layers, HMLPs, polynomial and constrained frame coefficients,
  meromorphic coefficients, and trainable pole locations while retaining real
  Cartesian trainable leaves, destination dtype/sharding, and affine membership
  evidence.
- Generalized representation-preserving holomorphic constraints around reusable
  real-coordinate frames, target-independent functional operators, batched
  minimum-norm lifts, coupled outputs, and target-specific affine maps. Added
  exact nonlinear cardinal projection plus named Goursat and plane-elasticity
  boundary functionals with explicit gauge and nullspace evidence.
- Added query-holomorphic DeepONet trunks with fixed or source-dependent hard
  targets, analytic conditional jets, and real harmonic operator adapters.
- Added exact finite Fourier circle traces, explicit contour/period functionals,
  fixed-pole meromorphic frames with domain-clearance evidence, and reduced
  variable-projection fitting for trainable pole locations.
- Added several-complex-variable multi-indices, analytic multijets for
  polynomial frames, holomorphic MLPs, and product potentials, pluriharmonic
  real-field wrappers, and metric-invariant holomorphic Kähler gauges.
- Replaced the scalar triangular P1 vertical slice with a shared computational
  cell mesh and generic fitted finite-element substrate: triangle P1/P2,
  quadrilateral Q1, tetrahedron P1, global DOF maps, fixed-topology geometry
  differentiation, dual-space weak residuals, affine Dirichlet constraints,
  compensated functionals, sparse affine lowering, and native linear,
  nonlinear, and DAE adapters.
- Completed the fitted finite-element runtime contracts with dynamic geometry
  and coefficient refresh, component and mixed `BlockSpace` fields, selected
  cell/exterior/interior domains with native rules, user residual/energy/
  bilinear/facet kernels, execution and accumulation policies, sparse lifecycle,
  lagged and adjoint operator factories, nullspace-aware linear solves, dynamic
  DAE/second-order/eigen adapters, curved coordinate elements, P0/RT0/Nedelec0,
  local/HDG condensation, material transactions and checkpoints, hierarchy/error
  evidence, embedded/enriched bases, partitioned DOFs, halo semantics, and FE IO.
- Added shared twofold compensated conservation accounting across structured,
  triangle, unstructured, spectral, and SBP conservation diagnostics, plus
  finite-volume ledgers, remap, overset/sliding, multiblock, small-cell, and
  diffusion certificates. Spectral diagnostics now honor their declared reduction
  dtype before accumulation.
- Reworked generic continuation around exact terminal coordinates, complete
  `phydrax.nonlinear` correctors, prepared Newton/trust-region reuse, explicit
  state/residual geometry, canonical real-coordinate maps for complex, algebra, and
  constrained spectral states, tangent predictors, full bordered tangents, curvature
  rejection, full-augmented event localization, execution-coordinate stability, and
  geometry-aware bifurcation evidence and normal forms.
- Added independently selectable tangent and adjoint linear policies for exact
  implicit root derivatives, plus prepared lagged-linear nonlinear updates that
  refresh structure-preserving operators, retain complete failure evidence, and
  certify every accepted root against the original physical residual.
- Added coefficient-resident global Fourier, sine, cosine, Chebyshev, Legendre,
  constrained, and mixed tensor spectral spaces; explicit padding/filter dealiasing;
  modal PDE and periodic conservation lowering; entropy diagnostics;
  conjugacy-preserving modal spatial noise; internal-linalg Galerkin, boundary-lift,
  and generalized tau formulations; and diagonal ETDRK2/4 integration with shared
  stable phi-three matrix actions.
- Added dealiased periodic incompressible Fourier dynamics, Hermitian real analysis
  coordinates, tensor spectral symmetry actions, primitive Fourier--Chebyshev--Fourier
  channel Stokes solves with pressure-gradient or bulk-flux control, fixed-step
  channel SBDF2 integration, shared-runtime periodic/Floquet analysis, relative
  invariant residuals, bounded evolution observation, portable spectral state
  artifacts, recurrence seeding, and finite-horizon edge tracking.
- Hardened incompressible spectral workflows around one shared flow problem,
  semidiscrete energy-balance diagnostics, constraint-valid channel initialization,
  fail-closed SBDF2 histories, and reproducible qualification artifacts. Added
  structured periodic compact first/second derivatives and staggered interpolation
  without dense line solves; periodic diagonal-norm SBP flux differencing with a
  reusable symmetric entropy-conservative Euler volume flux; and a geometry/solver
  split for structured MAC pressure projection with exact transform and refreshed
  variable-coefficient linalg routes.
- Added explicit real/imaginary Diffrax state packing for complex ODE, split, CDE,
  and stochastic paths. Public states remain complex; temporal evidence records the
  doubled real backend shape, dtype, tolerance geometry, and policy, while native and
  reject strategies remain explicit.
- Added exact finite real algebra specifications for real, complex, quaternion,
  octonion, Cayley--Dickson, and multicomplex families; three-valued law evidence,
  resource-bounded sparse/dense products, shared real-coordinate maps, algebra-valued
  spaces and operators, Diffrax algebra state policies, and unit complex/quaternion
  state geometries.
- Added exact and numerical commutator, Jordan-product, and associator operations;
  explicit left/right regular-action operators; resource-bounded algebra derivation
  spaces; and an octonion-derived G2 bridge with local metric, torsion, Ricci, and
  infinitesimal invariance diagnostics.
- Added canonical-complex holomorphic construction dependencies, spectrally
  initialized low-rank complex-affine layers, per-layer factorized
  `HolomorphicMLP` plans, certified independent branch bundles, same-coordinate
  holomorphic product potentials with exact Taylor-convolution jets, multiplicative
  gauge diagnostics, and a deterministic separability benchmark.
- Expanded compatible electromagnetics around conservative electric-displacement and
  magnetic-flux cochains: canonical structured/unstructured calculus, diagonal and
  metric-Hermitian constitutive maps, conductivity, Lorentz/Drude ADEs, Kerr/Pockels,
  gyrotropy, active/saturable gain, PEC/PMC/impedance/interface policies, periodic and
  Bloch calculus, electromagnetic CPML, probes/DFT/energy/Poynting observers, modal and
  near-to-far outputs, time/frequency/reversible adjoints, tetrahedral Whitney Hodge
  assembly, distribution metadata, and complete energy/constraint/CFL evidence.
- Added certified point-cloud differential functionals and dissipative Poisson
  execution, sixth/eighth-order smooth-exact TENO, explicit stabilization filters,
  cochain multirate scheduling, a backend-neutral lowered operator program with JAX
  and NumPy parity, neutral data interchange schemas, and runtime-integration
  guardrails for external research packages.
- Added public callable adaptive interval and triangle engines that reuse
  `AdaptiveQuadraturePlan`, `AdaptiveTrianglePlan`, `IntegrationPrecisionPolicy`,
  `IntegrationEstimate`, bounded partitions, statuses, and error-kind diagnostics.
  Specialized evaluators can now keep singularity classification and correction
  orchestration separate without duplicating the adaptive refinement subsystem.
- Expanded boundary layers with an explicit representation/discretization/evaluator
  split, global-error-aware adaptive near/self panel evaluation, declared corner
  topology and Kress/dyadic partitions, outgoing 2D Helmholtz kernels and explicit
  Brakhage--Werner CFIE assembly reports, target-associated 2D local expansions,
  triangular 3D surface panels, and a corrected near/far direct backend contract.
- Added target-centered Duffy self integration for 3D surface layers, coefficient-
  quadrature 3D QBX with continuous signed-distance clearance, outgoing 3D
  Helmholtz fields, and explicit Duffy-based 3D CFIE assembly reports. The
  reference near/far backend remains explicitly direct; no FMM claim is attached.
- Split the analytic sphere boundary atlas into two trimmed reference triangles,
  preserving full-sphere measure under triangular 3D panel quadrature and sampling.
- Added a fixed-topology Laplace multipole treecode reference with truncation
  estimates and direct/multipole work accounting; the production FMM path and
  global QBX coupling are recorded separately below.
- Added genuine 2D Laplace M2M/M2L/L2L translations and global QBX/FMM coupling
  with prepared target associations, continuous expansion clearance, separate FMM
  truncation, coefficient-quadrature, and local expansion error evidence.
- Added a metric-dependent Clifford algebra substrate with canonical blade layouts,
  sparse/dense prepared geometric, exterior, and contraction products, involutions,
  resource evidence, differential-form bridges, finite and standalone metric
  isometries, outermorphism actions, and exhaustive algebra-automorphism audits.
- Added flat constant-metric Dirac operators and exact-rational monogenic polynomial
  Trefftz fields with analytic partial derivatives, algebraic trial certificates,
  boundary-only linear fitting, and independent Dirac residual audits.
- Added complete-grade Clifford neural representations, grade-wise equivariant
  linear and geometric-product layers, Euclidean invariant gating, operator field
  schemas, sampled equivariance evidence, and non-promoted differential-context
  benchmark scenarios for incompressible flow, entropy-aware Euler, and Maxwell
  fields.
- Added likelihood-backed binary and multiclass empirical classification terms
  with encoded target schemas, case masks, positive statistical sample weights,
  posterior-compatible raw log probabilities, classification diagnostics, and a
  gathered categorical hard-label kernel that avoids one-hot target allocation.
- Expanded classification with posterior-compatible independent multilabel and
  fixed-threshold ordinal likelihoods; soft-target and focal objectives; explicit
  target-event masks; differentiable sigmoid/softmax/expectation field transforms;
  Dice, Jaccard, and Tversky overlap scores; and dense-grid, regular/irregular
  trajectory, graph-entity, and neural-operator classification. Structured terms
  retain geometry masks and physical measures, operator schemas use canonical
  JSON-safe ordered class names, and zero-weight objectives bypass evaluation.

- Added explicit convex entropy pairs with Euler mathematical-entropy factories,
  entropy-variable and flux compatibility validation, relative-entropy diagnostics,
  and volume-weighted structured/mapped finite-volume entropy evidence. Compiler
  integration rejects viscous, triangle, and modern unstructured pair diagnostics
  until those contributions have separate certified contracts.
- Promoted `h5py>=3.16.0` to a core dependency so finite-volume checkpoint and
  restart persistence is available in the default installation.
- Added domain-aware Legendre geometry with explicit primal/dual supports,
  conjugate and Fenchel--Young operations, representative validation, and direct
  dual translations. Added fixed-step mirror descent over mixed trainable
  PyTrees, including FunctionalSolver diagnostics and documented
  exponential-family KL and simplex exponentiated-gradient identities.
- Added a sequential Gaussian-process MAP initializer for expensive bounded posterior
  objectives. The UQ-owned search normalizes unconstrained positions, treats surrogate
  noise in raw negative-log-density units, records complete evaluated-point and
  fallback evidence, composes with state-space global/local MAP and Laplace workflows,
  and preserves the existing differential-evolution result contract.

- Added regular, first-order conic primal JVP/VJP operators over audited
  `ConicProgram` executions, with cached dense projection-KKT Jacobians,
  native-bound cotangents, projection/linear regularity evidence, and exact
  numeric binding. Added real scaled-triangle PSD, exponential, and standard
  power cones with JAX-native primal/dual projections and direct Clarabel
  mappings. Fixed versus interval bounds now participate in conic structure
  identity.
- Added certified finite Trefftz trial spaces for nD harmonic,
  polyharmonic-Almansi, and homogeneous Helmholtz fields, with deterministic
  exact-rational harmonic bases, resource preflight, sampled PDE audits,
  enforcement safety, provenance, and direct fixed-boundary least-squares
  fitting.

- Added real-parameter complex affine layers, polynomial and exponential
  holomorphic potentials, certified 2D harmonic, biharmonic, and plane-elastic
  representations, plus prepared 2D Laplace single/double boundary layers and
  an interior Dirichlet boundary-integral solve. Holomorphic coverage and parameter
  linearity propagate into physical certificates, distinguishing linear finite
  subspaces from nonlinear finite parametric families. Layer fields retain algebraic
  PDE exactness off singular support; continuous-boundary target admissibility is
  validated before residual audits, while target-clearance and panel/trace/BC
  approximation evidence remain separate.
- Added a single-device unstructured finite-volume stack over canonical cell complexes:
  triangle, quadrilateral, mixed polygonal, and affine tetrahedral geometry; stable
  topology/geometry/global identities; normal Rusanov-HLL-HLLC fluxes; general
  cell-polynomial and CWENO/WENO-Z reconstruction; explicit viscous triangle closure;
  shared SSPRK positivity/retry; matrix-free backward Euler; momentum-weighted
  Rhie--Chow pressure correction; schema-versioned mesh, case, checkpoint, HDF5/XDMF,
  and VTK persistence; fixed-connectivity GCL diagnostics and conservative remap;
  polygonal embedded-boundary clipping and PLIC/VOF transport; fixed-capacity two-level
  AMR; conservative overset interpolation; and accepted-step periodic sliding overlap.
  Tetrahedral reconstruction/dynamics qualification is degree-one affine only; degree-two
  k-exact and WENO qualification remains limited to the tested 2-D geometries.

- Added explicit stage flux-rate and accepted content-integral ledgers, conservative
  content state, epoch/event transactions, automatic certified AABB/polygon/tetra
  remap artifacts, embedded small-cell stabilization, two-material EOS/system
  foundations, capillarity/contact-angle evidence, and fail-closed rejection of
  unintegrated two-material PLIC runtime coupling.

- Added native-precision open-system campaign records, integrity-checked
  artifacts, semantic-variate replay evidence, fail-closed promotion policies,
  frozen campaign-matrix tooling, and cross-campaign graduation with permanent
  unsupported-claim provenance.

- Replaced Boolean archive qualification with exact campaign
  deserialization and reproduction, added derived physicality/capacity gates,
  adaptive preconditioned HEOM, eventful multi-event MPS jumps, analytic Padé
  residues, direct-memory Choi certification, active-memory refit, and separate
  neural projection-audit semantics.
  Campaign orchestration and graduation now live under developer tooling rather
  than the public solver API.

- Added connected VMC neural trajectory execution, seeded disjoint process
  tomography designs, count-aware held-out refit evidence, and pre-fit/post-fit
  recovery gates for sequential-process and causal-memory campaigns.

- Added fail-closed open-system evidence with quantified approximation
  thresholds, exact pseudomode initial states, process initial-state
  physicality, semantic trajectory keys, fixed-step probability guards,
  Gaussian convention checks, versioned artifact tooling, generic jump-solver
  adapters, certified local Kraus preparation, BDF HEOM diagnostics, and causal
  process simulation.

- Hardened open-system workflows with shared scalar-root event refinement,
  semantic trajectory checkpoints, environment MPS contractions,
  nonnormalizing TEBD, LPDO Strang evolution, scaled/implicit HEOM, matched
  spin-boson evidence, causal process tomography, and neural no-jump TDVP.

- Added event-driven quantum jumps, MPS canonicalization and TEBD, MPS jump
  trajectories, locally purified Kraus evolution, HEOM continuation,
  non-Markovian cross-representation diagnostics, adaptive Fock continuation,
  fermionic Gaussian dynamics, process-comb causality, and neural jump
  projection.

- General nonlinear root families with certified scalar bracketing,
  safeguarded Newton/Halley, chord and limited-memory Broyden, DF-SANE,
  pseudo-transient continuation, vector Halley, capability-selected and robust
  attempt graphs, Type-I/II Anderson, Steffensen acceleration, exact nested work
  budgets, scaling, mixed precision, batched small-system kernels, explicit
  sharding semantics, and first/second-order solution maps.
- Block residual/factor graphs with robust losses, manifold parameter blocks,
  route and Schur planning, traditional/subspace dogleg, dogbox,
  trust-reflective bounds, variable projection, POUNDERS, and incremental
  add/remove/relinearization evidence; plus BOBYQA, COBYQA, deterministic
  multistart, and independently recertified SciPy, NLopt, Ipopt, and Ceres
  boundaries.
- Scaled constrained models, SQP BFGS/SR1/exact Hessian choices, a native filter
  interior-point method with restoration, KKT inertia/null/range planning, fixed
  active-set and barrier sensitivities, frozen peer/corpus manifests,
  cross-family performance-profile campaigns, and solver graduation/regression
  gates.
- Authoritative physical optimization certificates now demote false-success
  POUNDERS and interior-point exits; condensed interior-point KKT systems reuse
  one factorization across predictor/corrector right-hand sides. Root
  polyalgorithms reuse residual and prepared Newton evidence across attempts.
  Residual-graph plans now execute dense, LSMR, or Schur routes with block-local
  robust curvature and explicit clipping evidence. Nonlinear comparisons keep
  backend claims separate from mathematics, enforce frozen runner identity and
  initial fingerprints, record cold/warm/steady phases in flat JSON, and form
  family-compatible performance profiles.
- Added Gaussian bosonic Lindblad dynamics, quantum-jump ensembles, adaptive
  bosonic Fock spaces, pseudomode/reaction-coordinate embeddings, HEOM,
  memory-kernel and TCL evolution, tensor-network states and truncation
  evidence, and process-tensor MPO contracts.
- Unified nonlinear and optimization model/direction/certificate precision,
  routed dense root, interpolation, Schur, KKT, and sensitivity systems through
  `phydrax.linalg`, and retained nested execution evidence. Added temporal,
  integration, geometry, and Hermitian precision to Gaussian, trajectory, HEOM,
  memory-kernel, Fock, and process-tensor paths, plus an explicit
  `TensorNetworkPrecisionPolicy` for storage, contraction, factorization,
  accumulation, certification, and output roles.

- Added projective-line Calabi–Yau campaign preparation, residue and induced
  hypersurface geometry, positivity-globalized Kähler-potential solving,
  Hermitian spectral/Sylvester infrastructure, faithful Bures density geometry,
  SLD quantum Fisher actions, mixed-state tomography, fixed-rank/Uhlmann
  primitives, and finite-dimensional Lindblad channel evolution.

- Estimator-aware `RandomizedMomentPenalty` with U-statistic,
  independent-product, and explicit plug-in modes; deterministic causal
  convolution and Caputo field-operator provenance; integral/nonlocal physics
  guidance; and an accuracy, bias, and performance benchmark campaign.
- Added end-to-end weighted geometric diffusion semantics, right-trivialized
  unitary propagation, abelian metric-DEC gauge fields, matrix-free Fisher and
  Hessian operators, geodesic manifold flow matching, explicit atlas covers and
  patch integration, CP^n Fubini–Study references, Dolbeault/Chern/Berry
  calculus, projective hypersurfaces, Kähler-potential Monge–Ampère operators,
  and Ricci-flat Kähler optimization composition.
- Expanded `phydrax.metrix` with immersion validation and Riemannian map
  geometry; correct tensor-density covariant derivatives; weighted metric
  measures and intrinsic hypersurface normals; exact and numerical endpoint
  geodesics with Fréchet statistics and transport/flow-matching adapters;
  complex-projective, unitary, special-unitary, and Hermitian-positive-definite
  manifolds; real-coordinate almost-complex, Hermitian, Kähler, atlas, and local
  SU(n) diagnostics; Hessian and exponential-family information geometry;
  vector-bundle gauge curvature; metric cochain Hodge assembly; anisotropic
  horizontal cometrics; and fixed-step Störmer–Verlet integration.

- Capability-checked temporal integration with complete Diffrax configuration
  provenance; additive KenCarp/Sil3 IMEX; native SSPRK3/SSPRK54, endpoint theta,
  variable-step BDF1--BDF5, matrix-free RA34PW2 Rosenbrock-W, generalized-alpha,
  fixed-ratio partitioned RK2/RK3, and one- through three-stage Gauss--Legendre
  implicit RK with collocation dense output.
- `phydrax.transport.continuous` endpoint couplings, linear probability
  interpolants, status-preserving continuous sampling, exact Euclidean continuous-flow
  densities, and uncertainty-bearing Hutchinson density estimates; plus
  `FlowMatchingTerm` and fixed-query quadrature-aware operator velocity metrics.

- Prepared finite nonlinear updates with typed application status, hard work
  controls, refreshable plans, additive/multiplicative/residual-optimal
  composition, Armijo Richardson and typed NGMRES outer methods, FAS/Picard/Newton
  updates, nonlinear Schwarz/Gauss--Seidel decomposition, and ASPIN with
  independently certified physical roots.
- Strict box-preserving semismooth variational inequalities with prepared
  topology-preserving refresh; matrix-free Steihaug--Toint quadratic trust
  regions; large-scale unconstrained and bounded Newton trust-region methods;
  and bound-aware Gauss--Newton and Levenberg--Marquardt residual optimization.
- Positive certified Xiao--Gimbutas, Lebedev, periodic, radial, and Duffy
  cubature with content identity and bounded storage; measure-matched fixed
  Gauss--Hermite expectations; geometry-owned native disk, circle, ball,
  sphere, and triangle-surface maps; mixed product plans; and static-capacity
  differentiable adaptive triangle refinement with explicit paired-rule error,
  partition, evaluation-budget, and terminal-status evidence.
- Positive total-degree standard-normal cubature through degree five, including
  grouped probability-product lowering; static-capacity Markov cubature for weak
  Itô and Stratonovich law propagation; positive polynomial recombination with
  frozen-support continuous derivatives; signature-certified piecewise-linear
  Wiener controls; weighted-measure result interoperability; explicit resource,
  moment, rank, positivity, and terminal-status diagnostics; and a compiled
  accuracy/performance benchmark harness.
- Cross-domain executable precision contracts with strict content-addressed
  request, resolution, nested evidence, and resource-assumption records.
  Finite differences separate coefficient, field, accumulation, certification,
  communication, checkpoint, AMR, distributed-halo, multigrid, and adjoint
  placement. Structured finite volume separates state, reconstruction, flux,
  conservative reduction/decision, output, and checkpoint precision across
  dynamics, SSPRK runtime, AMR, rollouts, HDF5, and restart. Integration
  separates evaluation, accumulation, decision, and output precision across
  fixed, mapped, adaptive, stochastic, product, weighted, MLMC, atlas,
  Riemannian, and projective execution. Neural operators retain master
  parameters, transient compute views, scoped matmul/FFT precision, dynamic
  loss-scale state, and persisted effective evidence. Spatial noise, SPDE
  composition, predictive summaries, bootstrap particles, native
  GMRES/FGMRES basis storage, Jacobi preconditioning, nonlinear Newton and
  globalization decisions, native/SSP temporal integration, flow-matching and
  manifold reductions, Hermitian spectra/Sylvester solves, quantum tomography,
  Calabi--Yau campaigns, randomized estimator objectives, and experimental
  standard-Optax `FunctionalSolver` contractions expose matching policies and
  evidence. Information geometry composes geometry precision with the linear
  runtime instead of raw dense solves. Precision-sensitive persistence formats
  reject incompatible contracts and retain effective evidence. The precision
  benchmark reports accuracy, storage, runtime, and evidence across FD, finite
  volume, integration, linear, nonlinear, temporal, geometry, Hermitian,
  operator, and UQ domains.
- `phydrax.discretization`: canonical entity/topology/support, finite-measure,
  DOF/field-space, plan/preparation, transfer, bundle, hierarchy, temporal, tensor,
  spectral, metric-cochain, conforming P1 finite-element, and conservative
  first-order finite-volume contracts. Strong, variational, and conservation
  compilers now retain complete discretization provenance; adaptive DAE results add
  their realized accepted-step mesh.
- Shared normalized proposals and fixed-kernel persistent Metropolis--Hastings chains
  with semantic key addressing, exact asymmetric Hastings correction, chain-preserving
  transition evidence, target refresh after parameter changes, and direct lowering to
  correlated `WeightedSampleTarget` measures that never claim IID uncertainty.
- Pairing-aware `EmpiricalGramLinearOperator` geometry with normalized nonnegative
  weights, sample centering, masking through zero weights, damping, rank/ESS evidence,
  complex adjoint/transpose actions, existing linear-runtime interoperability, and a
  single numerical implementation reused by UQ empirical Fisher actions.
- Discrete variational Monte Carlo with stable log-magnitude/unit-phase amplitudes,
  explicit real/holomorphic/nonholomorphic parameter modes, fixed-capacity connected
  operators, matrix-free local energies, persistent walkers, centered score geometry,
  damped SR updates, frozen-model final evaluation, complete status histories,
  documentation, tests, and compile/steady-state/storage benchmarks. JaQMC, jQMC, and
  Quantax are acknowledged as design references; this implementation is independent
  and Phydrax-native.
- Frozen-chain VMC diagnostics now report rank-normalized R-hat and bulk/tail ESS;
  pickle-free checkpoints preserve exact model/walker/key continuation; finite signed
  permutation groups provide validated symmetry-sector amplitude projection; and
  fixed-step real/imaginary-time TDVP reuses the same persistent sampling and
  pairing-aware score geometry. The scale benchmark now profiles sampler, connected
  local-energy, geometry, solve, storage, and end-to-end costs for periodic 8/12/16-site
  transverse-field Ising chains. Masked nonfinite feature and connection payloads are
  selected to safe values before Gram or local-energy multiplication.

- Industrial structured finite differences: exact point/interval entity layouts;
  masked variable-width Fornberg banks with consistency, adjoint, conservation, and
  stability evidence; manufactured convergence studies; stage-cached dynamic
  Dirichlet/Neumann/Robin and conforming interface programs; conservative scalar,
  diagonal, and tensor diffusion/advection lowering; diagonal-norm SBP orders
  2/4/6/8 with compatible second derivatives and SAT coupling; discrete-curl mapped
  geometry; oriented conforming and 2:1 multiblock mortars; geometric multigrid with
  Jacobi, red-black, and line smoothers; compact interior kernels, fused CSE pipelines,
  and multidimensional collective halos; WENO-Z/TENO/MP5, characteristic and
  multispecies Euler, ideal MHD, unsplit multidimensional fluxes, positivity, and
  entropy policies; entity-aware AMR halo/transfer/subcycling/reflux/regrid/migration;
  portable checkpoints, exact discrete adjoints, resource/precision preflight,
  structured cochains, and compatible Maxwell, MHD induction, elasticity,
  variable-density projection, poroelasticity, and thermoelasticity. Certified
  FFT/DCT/DST direct solves and directional split-field acoustic PML remain integrated
  with the same provenance.
- Structured finite volume now binds cell-average and directional face spaces directly
  to tensor support; supports uniform/nonuniform Cartesian and stationary mapped
  geometry, typed physical boundaries, piecewise-constant/MUSCL/WENO-Z/TENO/MP5 and
  characteristic reconstruction, Rusanov/HLL/HLLC/Roe and entropy fluxes, normal and
  transverse wave propagation, shallow-water f-wave balancing, multidimensional
  split/unsplit execution, Euler/multispecies/MHD systems, positivity and
  differentiability policies, conservative diffusion and compressible viscous fluxes,
  MAC pressure projection, matrix-free linearization, conforming/nested multiblock
  fluxes, and fixed-capacity AMR synchronization with integrated reflux.
- Structured finite-volume runtime hardening adds immutable ideal/stiffened-gas
  materials and constant/Sutherland/Prandtl transport closures; material-owned viscous
  and mapped-viscous fluxes; slip, no-slip thermal, supersonic, characteristic, and
  far-field boundaries; one prepared halo authority; Einfeldt-HLL fallback blending;
  bounded SSPRK retry/status runtime; versioned case, precision, checksum checkpoint,
  optional HDF5/XDMF output, differentiable scan/rematerialization rollout, quantitative
  verification contracts and CLI, and NamedSharding decomposition with scaling
  benchmarks.



- `phydrax.weighting` exact and quadratically reconciled relative-entropy moment
  calibration for dense, sparse, and matrix-free feature actions, with affine-rank
  reduction, audited convergence/regularity evidence, warm starts, and implicit
  target/prior derivatives. The mathematical formulation follows Barratt, Angeris,
  and Boyd's *Optimal Representative Sample Weighting*; the public `cvxgrp/rsw`
  and Apache-2.0 `andytimm/rswjax` packages are acknowledged as design
  inspiration, while this implementation is independent and Phydrax-native.
- Finite-measure calibration in `phydrax.integration`, shared calibration/coreset
  lowering, and ordered transformation diagnostics that preserve physical mass,
  masks, named axes, ancestry, support validity, execution keys, and provenance
  while invalidating inapplicable inherited integration-error bounds.
- Dense/sparse exact/soft calibration benchmarks with separate setup, first
  compilation, steady-state, cold-nearby, and warm-nearby timing and numerical
  evidence.
- Newton--Krylov now passes its adaptive forcing tolerance into each inner linear
  solve, so forcing policy changes actual Krylov work under eager and compiled
  execution.

- Learned function frames with masked, quadrature- and channel-metric-aware
  projection, explicit rank and residual evidence, reusable source encodings,
  arbitrary-query reconstruction, frozen inference, portable artifacts, and a
  research-tier operator benchmark composition.
- Scalar Tikhonov damping for dense SVD least-squares solves and a direct
  weighted least-squares path that reuses one factorization for coefficients,
  rank diagnostics, residuals, and differentiation.
- Native SING natural-gradient variational smoothing for additive-noise latent
  SDEs, with Gaussian information-chain algebra, deterministic and fixed-sample
  Gaussian expectations, irregular masked schedules, per-case backtracking,
  coherent posterior paths, fixed-posterior ELBO gradients, portable archives,
  diagnostics, tests, documentation, benchmarks, and explicit numerical status.
- Certified causal nonlinear recurrence with associative temporal linear solves,
  exact implicit adjoints, dense and quasi-Newton linearizations, and ELK-style
  Levenberg--Marquardt damping; plus opt-in recurrent-layer and fixed-trajectory HMC
  consumers with explicit convergence and fallback diagnostics.
- Normalized mean-field and FlowJAX reverse-KL variational inference with deterministic
  checkpoint replay, full-path Gaussian Markov state-space VI, reusable amortized
  encoders, and inverse-inclusion-weighted buffered target windows.
- Normalized latent-path and parameterized state-space density contracts that preserve
  the existing prior, transition, observation, schedule, mask, physical-time, and
  exogenous-input model hierarchy.
- JAX-compiled bootstrap particle filtering, retained initial genealogy, complete-model
  `O(TN)` genealogical scores, replaceable SG-MCMC gradient estimators, and
  complete-sequence particle-driven SGLD/SGNHT.
- `phydrax.nonlinear` contracts for algebraic systems, Newton line-search and
  trust-region roots, nonlinear GMRES and preconditioning, fixed-point acceleration,
  full-approximation multigrid cycles, variational inequalities, semismooth
  complementarity solves, and implicit root derivatives with explicit failure status.
- Native regular index-one differential-algebraic systems with explicit structural
  roles and scales, consistent initialization contracts, prepared fixed/adaptive
  BDF1--BDF5 integration, guarded cross-step numerical reuse, segmented continuation,
  local regularity evidence, frozen accepted-grid JVP/VJP replay with bounded
  checkpoint memory, status-rich trajectory evidence, semidiscrete implicit PDE
  compilation, and canonical identification adapters.
- Reusable nonlinear Newton preparation/refresh/solve lifecycles, adaptive
  Eisenstat--Walker forcing, explicit Jacobian refresh policies, hard aggregate
  inner-linear work budgets, and physical-root certification for transformed
  nonlinear preconditioners.
- Dynamic per-invocation controls for prepared native Krylov solves, plus
  capability-checked mixed-precision dense LU with pre-factorization condition
  screening, high-precision residual certification, iterative refinement, and
  requested/effective precision evidence.
- Matrix-free low-rank ADI lifecycles for factored continuous Lyapunov equations,
  including fixed-capacity factors, rank/truncation evidence, per-shift convergence,
  exact low-rank residual certification, numeric refresh, and factor-versus-dense
  storage accounting.
- Typed nonlinear optimization with matrix-free Newton--Krylov and trust regions,
  nonlinear conjugate gradients and strong-Wolfe search, Gauss--Newton,
  Levenberg--Marquardt, deterministic finite-difference least squares, proximal
  gradient/Newton methods and built-in functionals, filter/SOC SQP, primal--dual
  predictor--corrector KKT solves, state/design adjoints, stochastic risks and
  decomposition, explicit Optimistix interoperation, and `FunctionalSolver`
  integration.
- Canonical linear and quadratic programs with native variable bounds, typed solve and
  differentiation policies, reusable plan/prepare/bind/refresh lifecycles, explicit
  warm starts, independently audited KKT and infeasibility/recession certificates,
  portable status, and complete numeric provenance.
- Public zero, nonnegative, second-order, rotated second-order, and product-cone
  programs, with optional MPAX 0.2.4 LP/QP and Clarabel 0.11.1 conic execution behind
  lazy provider lifecycles and original-coordinate audits.
- Dense/structural-sparse linear-control compilation, reusable numeric refresh,
  explicit receding-horizon warm-start shifting, and affine stage/terminal SOCP
  constraints.
- Reproducible LP/QP/SOCP advanced-solver cases and independent certificates, plus
  control-horizon campaigns for sparse storage and cold-versus-warm MPC evidence.
- `phydrax.continuation` contracts for generic parameterized residual curves,
  arbitrary physical-parameter PyTrees, natural and pseudo-arclength
  predictor/corrector methods, reusable bordered solves, adaptive rejection,
  event localization, dense and Krylov stability evidence, explicit branch switching,
  fold/Hopf/pitchfork extended systems and certificates, normal forms, homotopies, and
  metric-aware root deflation.
- Standard and generalized nonsymmetric eigenproblem lifecycles with dense Schur/QZ
  and native restarted-Arnoldi/Krylov--Schur methods, standard, shift-invert, and
  Cayley transforms, homogeneous finite/infinite classification, paired left/right
  eigenvectors, residual and conditioning evidence, resource accounting, and
  isolated-eigenvalue derivatives.
- Lazy optional PETSc KSP/SNES, SLEPc EPS, PyAMGCL, and NVIDIA AmgX backends with
  dependency-free package import, explicit capability probes and lifecycle plans,
  sparse or matrix-free execution contracts, independently verified residual/status
  evidence, numeric refresh, transfer accounting, and explicit GPU/collective release.
- Canonical sparse triangular analysis and numeric solve, provider-neutral sparse
  factorization plans, incomplete Cholesky/ILU/ILUT preconditioner builders, and
  explicit sparse-provider availability/capability inspection.
- Array-backed finite axes and lazy Cartesian products, exact streaming exhaustive reduction, deterministic finite MAP screening, exact finite control-catalog search, portable MAP candidate archives, and a dense-oracle benchmark harness; independently implemented with [Brutax](https://github.com/michael-0brien/brutax) acknowledged as design inspiration.
- Typed preconditioner builders, prepared actions, planning costs, refresh provenance, and materialization-aware resource rejection.
- Additive and multiplicative subspace correction, Chebyshev smoothing, block factorization, and immutable multigrid hierarchy preparation.
- Native fixed-capacity BlockGMRES and BlockCG with explicit right-hand-side layouts, shared-subspace diagnostics, and block-aware differentiation.
- Immutable GCRO-DR recycling state with explicit reuse and rebuild policy.
- Standard and generalized Hermitian eigensolve plans with LOBPCG and thick-restarted Lanczos, residual/status diagnostics, refresh, and isolated-eigenvalue differentiation.
- Exact Galerkin and smoothed-aggregation hierarchy builders, deterministic diagnostics, transfer reuse, and optional PyAMG conversion.
- Exact diagonal and uniform local-block assembly, local block operators and factorizations, block-Jacobi preparation, and native Kronecker-sum direct solves.
- Policy-bounded canonical sparse assembly plans with reusable prepare/refresh recipes for sparse algebraic operator graphs.
- Deduplicated resident operator-state and per-right-hand-side action-workspace estimates, propagated into solve candidate costs.
- Symbolic `LinearSolveTemplate` planning with separate numeric binding, scoped kernel and spectral-interval certificates, and quotient-space `ProjectedPCG`.
- Bounded dense standard/generalized Hermitian eigensolves and pairing-aware singular-value decompositions with plan/prepare/refresh lifecycles and restricted scalar differentiation.
- Arbitrary-base `BasePlusLowRankLinearOperator` Woodbury solves with reusable base state, correction conditioning, resources, status, and provenance.
- Two-sided equilibration, iterative refinement, and resilient solve lifecycles that verify residuals in the original coordinates.
- Explicit structure compilation for diagonal, permutation, tridiagonal, triangular, banded, DCT-diagonal, and FFT-diagonal operators, including refresh-safe transform-diagonal operators.
- Numerically bound reusable Arnoldi/Lanczos projections, shared-basis shifted solve families, and partial-fraction rational matrix-function actions.
- Adaptive fixed-capacity stochastic trace and log-determinant estimation with separate statistical and projection-error evidence.
- General dense real/complex Schur eigensolves, nonnormal spectral observables, Riesz spectral subspaces, and first-order projector derivatives.
- Generalized, Sylvester, and continuous/discrete Lyapunov matrix-equation lifecycles built on the shared linear runtime.
- An advanced JSON benchmark harness covering Krylov reuse, shifted/rational actions, matrix equations, spectral projectors, low-rank updates, resilience, and adaptive spectral estimation.
- A schema-validated advanced-solver benchmark package with deterministic problem
  generators, independent original and refreshed certificates, explicit setup/
  compilation/preparation/solve/differentiation/refresh/verification and transfer
  accounting, reproducible JSON comparison, and lazy Phydrax, JAX, Lineax, Optimistix,
  SciPy, PyAMG, AMGCL, PETSc, and SLEPc adapters.
- Reusable full self-adjoint spectra with exact cluster-safe projector,
  density-kernel, and Loewner spectral-function derivatives for standard and
  generalized real or complex problems.
- Native batched dense Hermitian eigensolves and batched self-adjoint spectral
  calculus with per-member diagnostics, status, provenance, and batch-scaled
  resource accounting.

### Changed
- Compatible Maxwell state now stores electric displacement `D`, magnetic flux `B`,
  charge, material/boundary auxiliary state, and observer state. `E` and `H` are
  constitutive outputs. Construction uses `CompatibleMaxwellPlan(...).prepare()` and
  `PreparedCompatibleMaxwell`; the scalar-only direct `CompatibleMaxwellDynamics`
  path and E/B-primary state were removed.
- `MomentPenalty` now rejects resampled stochastic integration rather than
  silently optimizing a variance-biased squared estimate. `time_convolution`
  now accepts a deterministic `IntervalRule`; randomized QMC and importance
  modes were removed from the field-valued operator.
  Caputo field operators now use direct deterministic Gauss--Jacobi or
  Gauss--Legendre evaluation for both supported order intervals; stochastic
  sampler and endpoint-regularization arguments were removed.
- Finite-volume ownership is now structured and face-first. The triangular generic
  `FiniteVolumePlan`, system-specific reconstruction dynamics, and the
  `phydrax.discretization.reconstruction` owner were removed. Physical conservation
  systems now live in `phydrax.equations`, conservative face operators live in
  `phydrax.discretization.finite_volume`, time advancement lives in `phydrax.solver`,
  and `FDAMRSubcyclingPlan` is now `ConservativeAMRSubcyclingPlan`.

- Orthogonal-polynomial evaluation and Gaussian rule construction now pass through
  one private convention boundary. Hermite and Laguerre KAN identity/default
  initialization now represent the intended affine map, standard-normal Hermite
  rules own their probability normalization, invalid Legendre rule kinds fail
  explicitly, and functional collocation reuses canonical Chebyshev--Lobatto data.
- Numerical axis specifications, tensor/spectral methods, temporal path slicing,
  spatial-noise bases, cochain field semantics, and Laplacian spectral bases now have
  one canonical owner. Old `phydrax.domain`, `phydrax.solver`, `phydrax.operators`,
  `phydrax.graph`, and `phydrax.metrix` aliases were removed rather than deprecated:
  use `phydrax.discretization`, and use `phydrax.stochastic.SpatialNoiseBasis` for
  spatial stochastic forcing. `StochasticCouplingPlan` now owns a generic
  `DiscretizationHierarchy`.

- Spectral representations now split reusable `ModalTransform` objects from
  operator-specific `OperatorSpectrum` values. `SpectralDecomposition` pairs them
  where one API needs both, while `TransformDiagonalRepresentation` supports
  finite-difference, pseudospectral, graph, manifold, and covariance modal symbols.
- DeepONet now accepts generalized coordinate-evaluated basis trunks, exposes
  explicit output-bias control, and preserves existing pointwise and POD
  behavior while supporting projection branches and frozen nested models.
- Associative Gaussian-chain primitives now live in the lower-level linear-algebra
  implementation while their existing `phydrax.uq` public names remain unchanged.
- Nonlinear algebraic systems now have one public owner, `phydrax.nonlinear`, and
  generic continuation/bifurcation workflows have one public owner,
  `phydrax.continuation`; obsolete optimization and dynamics-analysis continuation
  exports were removed rather than retained as aliases.

- Linear solve planning now treats preconditioning as an explicit prepared subsystem with compatibility, memory, workspace, and setup-action budgets.
- Nonlinear optimizer runtimes now keep callable refresh structure static, preserve
  float32 and accepted-state carries under JIT, distinguish all-nonfinite globalization
  failures, reject unenforceable adapter evaluation budgets, and aggregate nested
  refresh and derivative diagnostics without negative sentinel arithmetic.
- Operator property evidence is propagated conservatively through preconditioners, block solvers, eigensolvers, and multigrid construction.
- RHS-width, GCRO-DR extraction, and multigrid setup/reuse resources are now rejected and reported before numerical work, including transformed prepared solves and dependency-invalidated hierarchy refreshes.
- Galerkin and smoothed-aggregation hierarchies now retain sparse assembly recipes and refresh coarse coefficients without rebuilding symbolic routes or silently densifying downstream levels.
- The linear benchmark harnesses now cover block-Jacobi preparation, sparse assembly planning/refresh/action, structured Kronecker-sum solves, and the advanced reusable and higher-operator lifecycles against dense, exact, finite-difference, or invariant references.
- The advanced-solver benchmark now provides algorithmically matched explicit-dense
  Newton-LU and matrix-free Newton-GMRES root cases for Phydrax and Optimistix, plus a
  semilinear sparse-PDE case covering Phydrax sparse-Jacobian preparation, symbolic
  reuse, numeric refresh, and native Jacobi-preconditioned PCG. It separately measures
  compiled root solves, implicit-root derivative compilation/execution, numeric
  refresh, refreshed solves, and refreshed verification; campaigns preserve float64
  inputs and canonical problem identities across adapters.
- Tensor contractions across the package, tests, benchmarks, and tools now consistently use `opt_einsum.contract` instead of direct `jax.numpy.einsum` calls.
- Closure-converted matrix-free SVD, eigensolve, spectral-projector, density-kernel, and spectral-function derivatives now support filtered JVP, reverse mode, and JIT.
- Named truncated-normal initializers now produce their conventional target variance, while rectangular orthogonal initialization avoids max-dimension square samples.
- `phydrax.nn.layers.inference_mode` now switches every inference-aware Equinox or Phydrax leaf in mixed model trees.

### Fixed
- Causal interval clustering now uses the Jacobian of the original reference
  coordinate, and zero-duration convolution and Caputo evaluations return exact
  zeros without evaluating singular kernels.
- Spectral, cochain, point-cloud, Maxwell, finite-volume, Clifford, and entropy
  contracts now preserve numeric identity and dtype, validate selectors and evidence,
  use exact Hodge pairings, apply directional CPML forcing, and fail closed when an
  advertised decomposition or eigenproblem is unsupported.
- Open-system solvers now enforce complete positivity and physical process evidence,
  preserve tensor precision and canonical gauges, process every event up to explicit
  capacity, align comparison time grids, propagate truncation failures, and mint
  verified campaigns only through provenance-bound artifact reproduction.
- Conic projections and sensitivities now classify canonical bounds, respect
  materialization budgets, avoid overflow in symmetric/fixed-bound arithmetic, and
  robustly handle low-precision PSD, exponential, and power-cone edge cases through
  the Phydrax linear-algebra substrate.
- Classification now preserves scalar and batch axes, validates vocabularies, labels,
  focal policies, masks, and operator semantics, excludes zero-support observations,
  and uses stable ordinal tails and positive-class binary overlap.
- Layer-potential QBX/FMM paths now use target normals and declared source reference
  triangles, share polynomial density reconstruction, propagate quadrature failures,
  report bounded omitted tails, bind source provenance, and route dense solves through
  `phydrax.linalg`. Bounded GP MAP maps avoid overflow, use disjoint design streams,
  and report source-bound, correctly normalized benchmark timings.
- Incompressible spectral workflows now bind every callable identity, preflight
  Hermitian-coordinate, recurrence, and channel-factor resources, compose reflected
  translations correctly, preserve linear-algebra precision, certify autonomous-flow
  neutral multipliers, latch bounded-observer failures, and verify spectral artifact
  content fingerprints on read.

- Masked BSDE and deep-splitting losses now sanitize inactive residuals before
  nonlinear reductions, Flower sanitizes masked source and normalization state,
  and ragged-series pooling selects inactive latent values before reduction.
