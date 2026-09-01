# Changelog

## Unreleased

### Added
- Added prepared periodic Fourier-shell statistics with continuum FFT normalization,
  Hermitian mode accounting, DC/Nyquist/final-edge policies, measured one-epoch matter
  power, auto/cross spectra, phase-sensitive spectral discrepancy, Parseval evidence,
  and inverse particle-field realization through existing splat, covariance,
  optimization, and sensitivity substrates.
- Added fixed-topology one-phase free-surface ALE hydrodynamics with graph
  geometry, extensive mapped momentum/scalars, conservative kinematic GCL,
  nonorthogonal mapped Hodge, mixed pressure projection, strongly coupled
  second-order stepping, accepted work ledgers, strict restart/output,
  qualification, examples, and benchmarks.
- Added hydrostatic primitive-equation ocean modeling with prognostic free surface,
  extensive layer transports and tracer inventories, implicit and split-explicit
  external modes, z-star and partial-cell geometry, freshwater volume sources,
  Flather/radiation boundaries, conservative wetting/drying, beta-plane and bounded
  latitude-longitude metrics, checked vertical implicit mixing, nonlinear seawater
  thermodynamics, Ri/KPP-like/TKE/Redi-GM closures, accepted ledgers, restart/output,
  qualification scenarios, examples, and benchmarks.
- Extended marker-flow coupling from the fixed uniform baseline to a shared
  stage-inverse KKT contract with explicit route state, physical-boundary correction,
  rank/condition gates, multiple regularized kernels, deterministic/compensated
  transpose reduction, variable-density and SPD variable-viscosity stages, nonuniform
  and mapped transfer, accepted-time rigid backward-Euler/midpoint, monolithic FE
  Newmark, native joint/contact adapters, resolved-subtraction lubrication, composite
  AMR impulse reflux, distributed single-owner transfer, conservative marker topology
  epochs, divergence-free projected transfer, sharp cut-cell and immersed-interface
  families, moving sharp epochs, fluctuating inertial and overdamped FIB methods,
  complete checkpoint/replay/output/runtime records, canonical examples, qualification,
  and scaling benchmarks. Advanced mapped, AMR, distributed, contact, sharp, and
  stochastic families retain explicit case-specific qualification gates rather than a
  blanket production claim.
- Added an authoritative surface and boundary-integral platform with checked
  linear solves, scalar Calderón formulations, periodic scalar kernels,
  FEM–BEM coupling, finite-depth potential-flow hydrodynamics, adaptive and
  block execution, portable archives, static elasticity and Stokes kernels,
  convolution quadrature, RWG Maxwell support, and fail-closed commercial
  qualification evidence.
- Reconciled cosmology, astrodynamics, and astrophysical-observation foundations:
  dimensional scales, artifacts and derivative capabilities, labelled observation/
  covariance/likelihood algebra, direct and hierarchical particle gravity, KDK
  transactions, ratio-two AMR mechanics, and event replay now have core owners.
  Domain applications retain comoving/canonical/scale-factor, physical epoch/frame/
  encounter, and instrument-specific semantics. Removed the astrodynamics nominal FMM
  and TreePM names that did not implement those algorithms.
- Added bounded maximal native cosmology profiles: fixed-layout thermodynamics/scalar
  transfer/line-of-sight algebra, global S3 manifold/KDK/harmonic Poisson/particle
  transfer, typed multi-release survey composition, deterministic FoF/unbinding/M200m/
  substructure/merger products, dynamic replayable stochastic stellar feedback,
  two-level ratio-two AMR, a shared Morton particle octree, isolated Barnes--Hut,
  uniform Cartesian FMM, and BH-short-range single-device TreePM. Each profile records
  explicit unsupported physics, topology, approximation, capacity, distribution, and
  communication boundaries.
- Added experimental granular micro--macro completion: fitted finite-volume
  capillary bridges with analytic energy and fit margins, radius-derived contact
  envelopes, conserved film/bridge inventory and exposed-area evaporation,
  balance-audited particle and interaction-segment continuum fields, sparse
  multilevel polydisperse neighborhoods, and dense-authority deforming periodic
  DEM cells with mixed stress/strain control and cell-work rollback.
- Added commercial Material Point Method closure contracts: exact claim tuples and
  executable support decisions, intended-use and G0--G7 release evidence, durable
  atomic checkpoint generations, HDF5/XDMF/VTK output, host-side supervision and
  observability, PIC/FLIP/blended/APIC transfer and independent advection, affine and
  post-advection MUSL, vector-root anisotropic plane stress, non-associated
  Drucker--Prager and Mohr--Coulomb plus Modified Cam-Clay, typed porothermal fields,
  simultaneous K-way contact with essential constraints and shared rigid reactions,
  topology-aware moving-domain and compact implicit actions, deterministic execution
  and capacity certificates, distributed ownership/global transactions, conservative
  particle lifecycle and ratio-two AMR, and evidence-tagged branchwise, event-aware,
  generalized, surrogate, stochastic, or nondifferentiable derivative products.
- Added full nonlinear solid-mechanics closure: canonical finite-strain laws,
  safeguarded plane stress, mixed incompressibility, conservative and follower
  loads, transactional continuation, physical bifurcation/selection, current-
  geometry contact, sharp and diffuse fracture, state-certified topology
  optimization, and parameter-measure-aware amortized operator learning.
- Added explicit state/adjoint acceptance evidence, prepared neural-field
  stationarity and virtual-work roots, separate physical static/dynamic
  stability contracts, and accepted-state continuation checkpoints/replay.
- Added three-dimensional Cartesian rigid-lid Boussinesq ocean process modeling
  with linear temperature-salinity reference physics, weighted-skew f-plane
  Coriolis, directional scalar diffusion, conservative surface scalar fluxes,
  impermeable surface stress, coupled fail-closed SSPRK3, accepted budgets,
  strict restart/output archives, qualification scenarios, examples, and
  benchmarks. Hardened coupled MAC scalar CFL, boundary-stage propagation,
  buoyancy exchange evidence, and rotation/stratification step restrictions.
- Added native fixed-topology partitioned multiphysics coupling with exact typed
  participant ports, direct and paired field transfers, deterministic SCC plans,
  explicit Jacobi/Gauss–Seidel sweeps, physically certified implicit interface
  roots, atomic rollback, fixed-window replay, fixed-grid waveform/subcycling
  contracts, resource and work evidence, and explicit algorithmic or implicit
  differentiation semantics.
- Added the fail-closed `iga.tensor` R1 isogeometric foundation for regular
  untrimmed full-dimensional 1D/2D/3D polynomial and NURBS maps; anisotropic and
  independent geometry/field grids; direct-tensor and extracted-Bernstein
  realizations; explicit common integration overlays; self-periodic traces;
  h-, p-, and k-refinement transfer; scalar, vector, and mixed H1 fields; linear
  elasticity, thermoelasticity, and generalized eigenspaces; fixed-topology
  differentiable numeric refresh; and immutable native restart lineage. Added
  exact allow-listed support tuples, deterministic per-case qualification
  manifests, an unreleased capability-profile producer, public examples,
  S1 migration fixtures, and record-only performance producers. Sampled map
  evidence remains neither a global injectivity certificate nor BRep/CAD
  support, and no capability is released without separately signed gate evidence.
- Added the representation-independent `phydrax.variational` functional substrate,
  DomainFunction bindings, and prepared-local value/first-variation/Hessian
  execution for coupled finite-element and isogeometric potentials.
- Unified `IntegralFunctional`, `VariationalEigenspace`, and
  `InvariantSubspaceResidual` on typed integration sources; fixed randomized
  objectives now require explicit realizations.
- Added `LocalFunctionalAction`, `finite_element_form_from_functional`, and
  `compile_finite_element_functional` alongside representation-bound
  `CellEnergyAction` and `FiniteElementFunctional` adapters.
- Routed the portable Neo-Hookean functional, DomainFunction operators,
  finite elements, and material points through the canonical finite-strain law.
- Closed remaining advanced-cosmology boundaries with projected canonical physical
  state, content-addressed products/artifacts, dependency-aware derivatives, shared
  observation/covariance likelihood algebra, concrete pinned precision-process
  wrappers, one-loop SPT, calibrated 200m halo/galaxy foundations, release-locked
  survey likelihood contracts, primordial H/He microphysics, local-curvature evidence,
  low-resolution CMB sky/TOD/mapmaking, periodic Ewald qualification, snapshots, and
  distributed-PM feasibility. Native full Boltzmann/CMB parity, global curved N-body,
  generic surveys, stochastic feedback, and production tree gravity remain explicit
  non-goals rather than fallbacks.
- Added a common chemical species and phase schema, NASA and polynomial species
  thermodynamics, prepared deterministic and stochastic mechanisms, native stiff
  reactors, extended rate laws, YAML interchange, and calibration coordinates;
  added compatible Poisson--Nernst--Planck transport, reactive electrodes,
  electrohydrodynamic and multiphase electrolyte coupling; and added compact
  Q-tensor Landau--de Gennes, Beris--Edwards, anchoring, active, chiral,
  electrostatic, and electrolytic liquid-crystal dynamics.
- Added one fixed-capacity runtime particle-population authority with activity,
  mass, incarnation-safe slot reuse, deterministic allocation/deactivation, DEM
  lifecycle migration, and runtime particle-splat masks.
- Added advanced PIC capabilities: integer charge states, conservative binary and
  background collisions, impact/field ionization, 1D3V/2D3V compatible Maxwell,
  reduced PIC current projection, open particle ledgers, CPML compatibility,
  integer moving windows, affine simplicial electrostatic/Whitney-current PIC,
  conductor KKT coupling, unstructured electromagnetic PIC, and matrix-free
  semi-implicit particle response with bounded Gauss correction.
- Added advanced FLIP capabilities: deterministic fixed-pool reseeding, particle
  level-set and ghost-fluid geometry, sharp capillary pressure jumps, moving-solid
  cut-cell and particle collision ledgers, free-surface viscous measures,
  variational symmetric-strain viscosity, and two-phase one-velocity FLIP.
- Added fixed-geometry 3D Laplace DP0 surface Galerkin capacitance solves with
  explicit weak/strong maps, bounded singular and near-pair quadrature,
  nonmaterializable blocked actions, immutable conductor selections, physical
  charge integration, and reuse of the existing direct and QBX layer evaluators.
- Added fixed-capacity two- and three-dimensional vortex methods with
  Gaussian free-space direct fields, periodic vortex-in-cell inversion,
  conservative particle-strength exchange, classic stretching, regularized
  filaments, steady and unsteady lifting surfaces, rigid polygonal vortex
  panels, boundary-sheet transfer, conservative remeshing, explicit rVPM and
  relaxation operators, nonlinear polar closure, fixed-tree acceleration,
  actuator/rigid/stochastic/learned workflows, qualification evidence, and
  fixed-topology differentiation contracts.
- Closed the native vortex capability boundaries with typed source/target and
  capability contracts, dynamic-core formulations, periodic Ewald and
  free-space FFT authorities, corrected P3M, hierarchical 2-D/3-D FMM,
  transactional populations and epoch replay, shared ring/sheet wakes,
  multi-surface lifting and complete loads, native 2-D/3-D panels, no-slip and
  immersed wall coupling, rigid/flexible FSI, rotor/actuator/control/acoustic
  workflows, stochastic ensembles, constrained learned reconstruction and
  assimilation, portable checkpoints/exports, and explicit sharding evidence.
- Added fixed-topology material-measure immersed-boundary coupling on uniform
  unit-density MAC grids: local cubic B-spline marker routes, force/torque/work
  certificates, exact prescribed pressure-plus-marker projection, IMEX-Euler and
  SBDF2 execution, explicitly separate penalty CFD–DEM, generic free rigid-body
  coupling, fixed FE marker H/H* maps, synchronized deformable coupling, and
  fixed-routing implicit sensitivities. Variable density, mapped/AMR/distributed
  markers, remeshing, contact extensions, fluctuating hydrodynamics,
  divergence-free interpolation, and sharp-interface changes remain unsupported.
- Added field-valued logarithmic compressible Neo-Hookean reference energy,
  line-search-safe nonfinite integral propagation, and an experimental matched
  neural-variational/finite-element hyperelastic qualification.
- Added advanced cosmology contracts: curved/CPL FLRW geometry and distances,
  realization-safe semantic transfer/power products, process-isolated linear-theory
  interoperation, neutrino component algebra, model-card power corrections,
  adiabatic gas--particle shared gravity, analytic halo foundations, Limber/RSD
  predictions, canonical CMB spectra, and bounded periodic force qualification.
  Periodic LPT/PM remains flat-only; calibrated external spectra and production
  distributed gravity remain explicit provider/qualification boundaries.
- Closed the declared astronomy capability boundaries with exact astronomical time
  instants and routes, IERS Earth orientation, compiled frame graphs, pinned
  artifacts and Chebyshev ephemerides, CCSDS/TLE products, high-fidelity force and
  light-time models, adaptive Gauss--Radau IAS15, analytical/DSST propagation,
  bounded event and maneuver schedules, encounter and hierarchical gravity,
  coupled variable-mass vehicles, tracking/variational/orbit-determination/mission
  products, calibrated WCS imaging, surveys, radiative transfer, waveform and
  exoplanet operators, native early-universe/Boltzmann/nonlinear cosmology, and
  compact-object EOS/TOV models. Provider discovery, network access, external data
  redistribution, and smooth-gradient claims across discrete topology remain
  intentionally excluded.
- Extended Material Point Method with explicit USF/USL-minus/MUSL schedules,
  fixed-capacity adaptive realization and scheduled replay, constitutive capability
  and algorithmic-tangent contracts, isotropic plane stress, multiplicative
  finite-strain J2 plasticity, uGIMP/cpGIMP/CPDI/CPDI2 particle domains, rigid and
  two-field Coulomb contact, material/velocity-field identity state, active-block
  semantics, compact block storage, dense matrix-free implicit roots, AT2 diffuse
  fracture, and separate field-partition/CPIC sharp-fracture paths. Each family
  carries transactional rollback, branch/topology evidence, qualification artifacts,
  compatibility limits, and explicit differentiation semantics.
- Added one- and two-dimensional Cartesian wet/dry shallow-water finite volumes with
  exact dry-state semantics, prepared static bathymetry, Chen--Noelle hydrostatic HLL
  face contributions, equilibrium-aware MUSCL reconstruction, SSPRK-stage conservative
  positivity, accepted one-sided bed-integral evidence, f/beta-plane Coriolis forcing,
  renderer-neutral observables, output support, qualification cases, and benchmarks.
  Removed the unqualified one-dimensional shallow-water f-wave path.
- Added native experimental velocimetry with mask-aware multipass and ensemble
  PIV, explicit peak/validation/replacement evidence, calibrated physical
  conversion, pinhole/distorted/refractive camera rigs, robust calibration and
  triangulation, conflict-free multi-view particle reconstruction, streaming and
  globally refined PTV tracks, frozen-association smoothing, radiometric
  particle-image formation, residual-image Lagrangian refinement, deterministic
  synthetic qualification, optional learned dense displacement, canonical
  archives, and explicit-loss ecosystem adapters.
- Added fixed-population compatible particle-in-cell dynamics over stable charged
  particle supports: measure-aware endpoint charge, physical cochain E/B gather,
  matrix-free compatible electrostatics, relativistic Boris stepping, periodic
  cubical-Whitney trajectory current with discrete-continuity evidence, and
  transactional coupling to the existing compatible Maxwell runtime.
- Added constant-density fixed-population free-surface FLIP over prepared
  particle splats and MAC grids: cell and staggered-face mass/momentum transfer,
  runtime atmospheric pressure projection, bounded velocity extrapolation,
  explicit PIC/FLIP grid-delta blending, problem compilation, fixed-step
  rollback, and complete transfer/projection/energy evidence.
- Added `phydrax.circuit`: block-valued typed wave ports, dense and matrix-free
  hierarchical scattering, grounded dense/sparse MNA, causal implicit element laws,
  native DAE/DC/continuation/descriptor analysis, rational macromodels, periodic
  analysis, correlated noise, metrology/de-embedding, field/electrothermal coupling,
  SPICE and restricted behavioral interchange, certified learned dissipative laws,
  Touchstone I/O, and thin native optimization/UQ adapters.
- Generalized compatible time-domain Maxwell to explicit full-3D, TEz, and TMz
  cochain roles; added resource-preflighted final-state runs, sparse magnetic
  constraint projection with proved elision, boundary-packed CPML, prepared paired
  electric/magnetic sources and mode ports, harmonic-defect evidence, scalar
  geometry material assembly, and independent case batching.
- Added sparse metric-aware conic density filtering with explicit fixed-region
  semantics, finite-beta tanh projection, differentiable composed transforms, and
  separate forward-only hard thresholding.
- Added explicit cosmological length/mass/time scales, parameter-differentiable flat
  FLRW backgrounds, native first/second Lagrangian growth, immutable expansion/growth
  and linear-power products, state-ready 1LPT/2LPT, and transactional periodic
  scale-factor particle-mesh rollout. The cosmological path reuses the existing
  particle discretization, splat, self-gravity, and typed PM force evaluation;
  synchronized baryon/particle orchestration remains distinct from physical coupling.
- Added `phydrax.applications.astrodynamics`: explicit scale, two-part epoch, and
  frame contexts; Cartesian and modified-equinoctial states; bounded universal
  Kepler propagation with implicit JVP; fixed-capacity multi-revolution Lambert
  branches; pure force composition; adaptive and symplectic propagation; hybrid
  orbital events; provenance-bearing time/frame/ephemeris products; third-body and
  J2--J4 gravity; direct and nearly-Keplerian N-body dynamics; CR3BP; rigid
  spacecraft, finite-burn, reaction-wheel, and orbit-measurement contracts; and
  host-only coordinate, SPICE, and SGP4 adapters. No provider discovery, data
  download, close-encounter regularization, DSST, or adaptive IAS15 is implied.
- Added native astrophysical observation operators for observer projection,
  polynomial limb-darkened circular occultation, photon-counting bandpasses,
  transit count likelihood composition, binned and image responses,
  frequency-domain detector likelihoods, ordered ray transfer, and static complex
  field sequences. Contacts, event/branch selection, provider loading, and capacity
  changes remain explicit non-smooth boundaries.
- Added provenance-bearing CMB angular-power tables with explicit `Cl`/`Dl`
  conversion and fixed response-window Gaussian likelihood composition. Spectrum
  generation and experiment data remain external.
- Consolidated kinetic multiphysics around one thermodynamic closure for energy,
  variational derivative, symmetric stress, and explicit force representation.
  Added auditable kinetic field/stage manifests, exact portable checkpoints,
  production prepared sharding, signed-distance geometry-to-link compilation,
  parabolic and Womersley targets, collision-aware ratio-two AMR transfer with
  half-time interface data, and graduated scientific qualification evidence.
- Added enhanced conforming scalar virtual elements of qualified degree one
  through three on arbitrary-arity polygonal cell blocks, including certified
  H1/L2 projectors, explicit stabilization, functional trace constraints,
  matrix-free and sparse execution, fixed-topology geometry differentiation,
  projected reconstruction, mass-matrix DAEs, and generalized eigenproblems.
- Added static three-dimensional fixed, ball, and hinge rigid-body graphs with
  globally coupled mass-metric SO(3) pose projection, full velocity KKT projection,
  implicit root derivatives, physical position/velocity residual certification,
  multiplier warm starts, and fail-closed candidate/accepted transitions. Contact,
  friction, compliance, motors, dynamic topology, two-dimensional joints, and PBD
  compatibility remain outside this contract.
- Extended constrained mechanics with native planar fixed/ball joints, dimension-aware
  prismatic and distance joints, canonical stable row/coordinate layouts, physical
  compliant/dissipative laws, bounded effort motors and servos, unilateral joint
  limits, hard velocity restitution, exact planar/spatial Coulomb-cone impulses,
  irreversible joint breakage, and fixed-capacity topology transactions. Added
  transactional implicit Newmark volumetric FEM, mixed pressure gauges,
  rigid--deformable attachment KKT operators, objective two-/three-dimensional
  Cosserat rods and triangular membrane/bending shells. Replaced the partial
  particle-local deformable-contact routes and 2-D penalty workflow with
  exact-map collision surfaces, dense/sweep-and-prune candidate epochs,
  area-weighted physical barrier contact, conservative inclusion CCD, T3/T4
  inversion limits, static and transactional Newmark solves, lagged Coulomb
  friction, fixed-route sensitivities, and direct rod/shell collision-surface
  adapters. Explicit rigid--MPM weld/penalty/impulse coupling retains separate
  branch, rank, energy, route, and rollback certificates.
- Extended the contact substrate with an ordered guarantee lattice,
  roundoff-directed certified swept-AABB CCD, cached and fully compiled
  fixed-shape candidate filters, per-vertex separation, nonlinear/independent
  participant kinematics, rigid/articulated/point/MPM adapters, high-order
  proxy error inflation, implicit geometry, cubic and rigid sweep trajectories,
  closed-surface geometric-contact filters, deterministic triangle-overlap
  mortar quadrature, equal-pressure tetrahedral patch extraction, distributed
  route ownership/halo exchange, and remeshing state transfer. Added composable
  material-pair closure with barrier/geometric/compliant/adhesive normal laws,
  static/dynamic, anisotropic, and rate-state friction, irreversible
  wear/cohesive evolution, smooth force assembly, hard Coulomb-cone impact,
  projected, SAP, semismooth, and primal-dual cone solvers, rolling/spinning
  resistance, mortar, one-sided/unbiased Nitsche, mesh tying,
  cross-discretization coupling, hydroelastic patches, periodic/homogenized
  rough contact, thermal/electrical/mass flux, lubrication, contact-graph
  preconditioning, and fixed-branch closure/cone/mortar derivatives.
- Added epochal particle-capacity growth with stable structured interaction
  identities, transactional state migration, fixed-pool insertion and fragmentation
  retries, segmented replay, and transition pullbacks. Added multidimensional
  body-frame particle interiors, conservative unstructured transport, local
  coarse/fine AMR, boundary-face exchange, and native sparse implicit conversion.
  Added feature-certified superquadric triangle-wall contact with canonical shared
  feature ownership, wall histories, reactions, wear observables, and explicit
  feature curvature. Added matrix-free monolithic fluid-particle Newton coupling
  with momentum, heat, species, reaction, contact/radiative sources, route
  certificates, block preconditioning, atomic rollback, and implicit sensitivity.
- Added native atomistic dynamics with complete unit identities,
  position-independent prepared systems, stable-ID molecular topology and pair
  exceptions, composable classical/learned scalar-energy programs, dense and
  triclinic cell/Verlet execution, momentum-form NVE and BAOAB NVT,
  SHAKE/RATTLE constraints, stress, direct Ewald and B-spline PME, isotropic
  NPT moves, bounded replayable trajectories, exact checkpoints, hybrid and
  RESPA composition, Born–Oppenheimer provider boundaries, ring polymers with
  PILE, and variance-constrained semi-grand transitions. Dense graph resources
  are now explicit execution-plan identity rather than learned architecture identity.
- Extended the atomistic runtime with interaction-site coordinate maps and virtual-site
  force pullback; native force-field bundles, terms, policies, SETTLE, and OpenMM/OpenFF/
  ParmEd adapters; typed frames, H5MD/XYZ reporting, rerun, MDAnalysis, i-PI, and PACKMOL
  boundaries; collective variables, static/adaptive biases, replica exchange, FEP/TI/BAR/
  MBAR; committee uncertainty and deterministic acquisition; advanced thermostats,
  anisotropic pressure control, rigid and Brownian dynamics; polarization, multipoles,
  implicit solvent, advanced quantum-nuclear estimators; walls, manifold constraints,
  active/DPD and EAM/SW/Tersoff models; and distributed atomistic execution.
- Added fixed-capacity explicit Material Point Method dynamics for plane-strain and
  three-dimensional Neo-Hookean solids: nodal quadratic B-splines, matched APIC
  transfer, first-Piola reference-volume forces, transactional USL updates,
  support-halo and prescribed-velocity boundaries, acoustic/advective/force step
  evidence, full/step/block replay, final/checkpoint/trajectory retention, and
  piecewise-versus-frozen gradient reports. Corrected logarithmic Neo-Hookean
  parameter naming so its volumetric coefficient is Lamé lambda, with an explicit
  physical shear/bulk constructor.
- Added bounded and periodic unit-density MAC incompressible dynamics with static
  no-slip wall closure, face-dual velocity coordinates, symmetry-preserving momentum
  transport, conservative explicit viscosity, transform-or-iterative stage
  projection, fixed-step SSPRK composition, short-horizon differentiation, step
  restrictions, and complete constraint and kinetic-energy diagnostics. Hardened
  singular transform solves so masked pressure nullspaces retain finite reverse-mode
  derivatives.
- Extended the MAC flow substrate with dynamic no-slip/free-slip/inflow/pressure/open
  boundary closures, named conservative scalars and Boussinesq exchange, iterative,
  transform, hybrid-line and IMEX/SBDF2 viscous solves, conservative variable-density
  face momentum, dual-measure resolved IB–DEM coupling, transactional adaptivity and
  replay, short-horizon and least-squares-shadowing sensitivities, explicit sharded
  pressure CG, compatible mapped/ALE geometry, and conservative nondifferentiable
  remesh epochs. Every path exposes its mass, momentum, energy, residual, topology,
  differentiation, resource, and fail-closed acceptance evidence.
- Added exact fixed-temporal finite-volume replay with full, step, or block
  rematerialization; transactional balance-law source composition and persistence;
  periodic Newtonian and particle-mesh gravity; replayable Hermitian spectral
  Ornstein--Uhlenbeck forcing; implicitly differentiated tabulated cooling;
  conservative trainable face closures; and periodic Cartesian constrained MHD with
  integrated cochain face fluxes, edge circulations, coupled stage positivity, and
  HLLD-to-HLL fallback evidence.
- Added bounded adaptive balance-law realization with process-aware step limits,
  transactional retry rollback, fixed-capacity decision journals, and exact scheduled
  replay of accepted temporal meshes. Added global OU realizations whose innovations
  obey the OU semigroup under interval subdivision, including antithetic coupling.
- Unified ordinary finite-volume and constrained-MHD source composition behind one
  prepared balance-law transport contract. Gravity, cooling, and OU forcing now compose
  with face-flux MHD under the same adaptive realization, scheduled replay, rollback,
  component-ownership checks, and portable checkpoint semantics.
- Added dimension-generic constrained-MHD layouts, primitive PLM/WENO/TENO/MP5
  reconstruction, HLL-UCT, accepted face/edge integral ledgers, physical boundary
  policies, dual-energy and CTU support, non-ideal and AMR cochain operators, bounded
  gravity, exact cooling coordinates, modal forcing, thermochemistry, radiation
  moments, cosmological workflows, field inference, and structure-preserving closures.
- Added fixed-rank randomized Nyström preconditioning with auditable sketch and
  refresh evidence; Diffrax-backed neural Galerkin evolution over fixed physical
  field metrics with rectangular or Gram tangent solves and saved-node audits;
  backward Diffrax characteristic tracing with macro-step neural projection; and
  mass-preserving fixed-support residual-attention collocation with explicit ESS,
  KFAC, and controlled-policy contracts.
- Added native athermal lattice-Boltzmann flow on uniform isotropic cell grids:
  certified D2Q9/D3Q19 velocity sets, BGK/TRT collision with collision-coupled
  Guo forcing, periodic and frozen halfway-wall link routing, fixed tangential
  moving walls, explicit physical/lattice scaling and precision evidence,
  fail-closed fixed-step integration, differentiable runtime controls, and a
  memory-bounded generic fixed-step rollout with final/checkpoint/trajectory
  retention.
- Expanded kinetic methods with D3Q27, prepared moment bases and advanced collision
  families, staged open/curved/moving-wall ownership, explicit local implicit forcing,
  geometry epochs and conservative transfers, multiblock and ratio-2 refinement
  contracts, colour-gradient/free-energy/thermal/species/reactive distributions,
  certified D2V17 and off-lattice D2V37 smooth-compressible methods, fixed FV/kinetic
  interfaces, sharded and AA/fused execution, block reverse replay, and a forward-only
  stable-tuple IREE export contract. Advanced paths report capability, conservation,
  realizability, equivalence, and qualification evidence without extending the
  qualified low-Mach baseline by implication.
- Added reciprocal-lattice harmonic discretization with true one-dimensional and
  oblique two-dimensional periodicity, selected FFT analysis/synthesis,
  pairwise-difference material convolution, translation covariance, resource
  preflight, and Gamma-containing Brillouin-zone rules.
- Added `phydrax.solver.maxwell.fourier_modal`: full-tensor periodic finite layers,
  homogeneous ports, differentiable boundary-field cascade propagation, direct,
  inverse, and local-frame Fourier factorization, a nondifferentiable modal reference
  backend, stable scattering composition, named electric/magnetic current planes,
  multi-RHS and Brillouin source semantics, interior field reconstruction,
  diffraction-order far fields, explicit refresh, convergence, resource, diagnostic,
  status, and provenance contracts.
- Extended native low-rank adaptation with rank-stabilized scaling, exact
  adapter-artifact reconstruction, and composition with frozen random-weight
  factorization coordinates.
- Added field-certificate-aware geometry-to-material rasterization for
  Fourier-modal Maxwell, with sharp and differentiable compact-Heaviside paths,
  fixed subpixel sampling, fill-fraction evidence, and material identities.
- Added certified finite-box Method of Moving Asymptotes, constrained
  reduced-adjoint state/design optimization, sparse physical-radius density
  filtering, SIMP compliance topology optimization, and independent
  reference-discretization reanalysis.
- Added native force-density structural design for tension, compression, and
  mixed-sign pin-jointed systems with sparse coordinate or orthonormal affine
  restraints, reciprocal GraphIR conversion, stable external IDs, prepared
  linear/nonlinear refresh, weighted-Laplacian Newton preconditioning, fixed,
  line, self-weight, traction, follower-pressure, and pneumatic load laws with a
  component ledger, mathematical solution derivatives, reduced and structured
  force/support/load design, pure geometry/force observables, same-topology
  batches, per-graph evidence, mechanism/self-stress spectra, supplied-rigidity
  tangent stability, and continuation bridges.
- Added member-network constitutive verification over force-density topology:
  stress-free reference states, exact tension-only cable active sets,
  corotational frame and discrete-rod bending, surface hinges, local and global
  buckling, nonlinear continuation bridges, prestress fabrication/actuation
  evidence, staged construction replay, continuous and catalog sizing, and
  explicit certified/failed/incomplete structural verdicts.
- Added advanced structural evidence: generalized coordinate channels, explicit
  section-orientation fields, semirigid connections and nonlinear supports,
  extensible catenaries, cable/saddle contact, warping beam and bracing energy,
  fiber-section plasticity transactions, imperfections, collapse and dynamics,
  thin-walled GBT/finite-strip/shell-submodel evidence, exact precedence
  branch-and-bound, standards clauses, reliability, calibration, evidence
  acquisition, and immutable structural-twin snapshots.
- Added pickle-free StableHLO/IREE inference export with matched optional
  compiler/runtime versions, in-process compilation and loading, exact
  shape/dtype ABI checks, checksummed manifests, and native parity evidence.
- Added full-rank Euclidean VP/VE score diffusion with structured diagonal Gaussian
  laws, exact perturbation marginals, weighted denoising score matching, replayable
  reverse-time SDE sampling, probability-flow composition, per-realization Diffrax
  initial states, and memory-linear diagonal Wiener coefficients. Replaced the
  flow-specific `FlowMatchingPolicy` with shared `UniformTimeSamplingPolicy`.
- Extended generative transport with stable array/PyTree/complex event coordinates,
  block-operator Wiener noise, full and Hausdorff Gaussian factor laws,
  matrix/state-dependent Itô reversal, exactness-labeled guidance, discrete Gaussian
  and categorical diffusion, coefficient-space field/path diffusion, intrinsic
  manifold and complex diffusion, latent/graph/atomistic compositions, persistent
  energy training, normalized autoregressive laws, and sample-only adversarial
  objectives. Every family retains explicit measure, geometry, approximation, and
  density capabilities rather than sharing a universal model facade.
- Generalized matrix-free quantum local actions through
  `AbstractLocalQuantumOperator` and evidence-rich `LocalOperatorEstimate`, with
  a clean migration of discrete VMC/TDVP while preserving the connected-action
  algorithm. Added finite nonperiodic Born--Oppenheimer
  `ElectronicCoulombHamiltonian`, validated Bohr/Hartree reference conversion,
  exact and chunked-exact coordinate kinetic traces, singularity statuses without
  distance clipping, replayable electronic walkers, an exactly corrected
  state-dependent proposal, and a full-generalized-determinant
  `phydrax.nn.quantum.FermiNet` with same-spin antisymmetry, sparsity-aware
  scaled log envelopes, higher-order-correct zero/subnormal signed products,
  polynomial singular-term determinant derivatives, coefficient-aware
  nonzero-product mixture shifts with coefficient- and singularity-reactivation
  fallbacks, a positive physical decay floor, and determinant mixtures
  differentiable at zero coefficients, under an explicit four-electron ceiling.
  Electronic VMC
  folds local statuses into validity and reuses persistent chains, matrix-free
  score/Gram stochastic reconfiguration, training lifecycle, linear solves,
  diagnostics, statuses, and checkpoints. Added H/He/H₂ tests,
  documentation, and a fixed multi-seed benchmark campaign with predeclared
  statistical/chemical gates and provenance; periodic, relativistic, and
  stochastic-trace electrons remain unsupported.
- Added exact scalar temporal Matérn-3/2 and Matérn-5/2 Gaussian processes
  through content-addressed continuous state-space compilation, origin-shifted
  stable irregular train/query schedules, exact missing/query masks, bounded
  stationary long-gap discretization, sequential square-root filtering and
  reverse-scan RTS smoothing,
  dense-parity parameter gradients, active-observation marginal likelihoods,
  linear-storage predictive marginals, explicit compute precision,
  prepared/evaluated identity and failure provenance, portable result export,
  and complete retained-storage scaling benchmarks.
- Added integration-native fixed-design Bayesian quadrature for normalized scalar
  Gaussian targets with analytic squared-exponential kernel means, optional
  kernel scaling, content-bound Gaussian targets, separate observation noise and
  solve regularization, true evaluation-stage dtype placement, scale-normalized
  prepared `phydrax.linalg` conditioning, reusable PyTree/field reductions,
  overflow-stable analytic means, posterior-SD diagnostics, dtype-aware variance
  validity, explicit target/contraction/solve/resource failure boundaries, and
  an analytic Gaussian benchmark against IID and
  randomized QMC. The posterior SD is explicitly model uncertainty, not a
  deterministic or frequentist error bound.
- Added `phydrax.atomistic` and `phydrax.nn.atomistic.PaiNNPotential` for finite
  nonperiodic molecular research: scale-identified atomic structures and padded
  batches reuse material-particle identities and `GraphIR`; resource-guarded
  case-isolated dense neighborhoods fail closed without truncation; invariant
  energies yield conservative forces with typed status, diagnostics, precision,
  and provenance; energy-only, force-only, and joint training retain fitted
  training-only normalization, selection, restart, and complete histories; and
  local-NPZ rMD17 parsing/splitting plus a fingerprinted multi-seed benchmark
  tool require explicit data provenance. Periodic execution, stress, long-range
  electrostatics, direct-force heads, ASE integration, and molecular-dynamics
  stability claims remain outside this capability.
- Added labeled nonintrusive polynomial chaos for independent scalar Uniform and
  Normal inputs: preflight-guarded graded total-degree multiindices and
  sample-by-feature projection storage, stable normalized Legendre/Hermite tensor
  bases, content-addressed measure-honoring product-integration projection, diagnosed
  exact/least-squares regression with complete solver-policy identity, immutable
  array/Field/PyTree expansions, coefficient moments and first/total Sobol effects,
  portable fit evidence, and a matched-model-call benchmark campaign.
- Added resource-planned `O3TensorProductPlan`/`O3TensorProduct` layers and a
  drop-in `phydrax.nn.atomistic.NequIPPotential`. Independently derived
  Cartesian Clebsch–Gordan maps cover legal scalar/pseudoscalar,
  vector/pseudovector, and symmetric-traceless tensor/pseudotensor paths through
  degree two with per-instruction radial weights, masked finite-molecule
  aggregation, species-conditioned self connections, parity-safe gates, and the
  existing conservative prediction/training contracts. The rMD17 campaign now
  records matched PaiNN-versus-NequIP seeds, errors, equivariance defects,
  timing, memory, parameters, neighborhood work, gates, summaries, and
  provenance. High-degree irreps, MACE/symmetric contraction, periodic systems,
  stress, long range, and molecular-dynamics claims remain out of scope.
- Added experimental two- and three-dimensional soft-sphere DEM with rigid
  translational/angular state, collision-free stable pair keys, persistent
  Cundall--Strack contact history, linear spring--dashpot and Hertz--Mindlin
  contact families, exact-signed-distance barriers, dense/cell-list execution,
  structured fail-closed fixed stepping, contact qualification, an executable
  settling example, and dense/cell performance evidence.
- Added source-resolved accepted-step DEM energy/work ledgers, explicit rejection
  reasons, qualification artifacts, certified Verlet caching, fused/reference
  pair reductions, radius-class filtering, replay/checkpointed VJPs, sharp and
  smooth sensitivity contracts, inverse/UQ qualification, and transverse
  hybrid-event saltation.
- Added compositional normal/cohesion/tangential/rotational DEM contact history,
  elastic rolling–torsional resistance, finite-range DMT cohesion, conservative
  capillary bridge lifecycle, near-contact lubrication, bilinear elasto-plastic
  normal response, elastic half-space multicontact correction, and conservative
  contact heat exchange.
- Added prescribed force/torque servo barriers, certified analytic contact
  curvature, curved Hertz walls, facet traction/work/heat observables, Finnie
  wear accumulation and geometry commits, SO(2)/SO(3) rigid bodies, immutable
  sphere-clump templates, triangle walls, elastic/damageable bond graphs,
  fixed-pool topology events, convex SAT contact, certified sphere-to-implicit
  contact, and support-map superquadric contact and dynamics.
- Added conservative slab/cylindrical/spherical internal shell meshes, typed
  species/phase/element thermochemistry, polynomial heat-capacity inversion,
  heat/species transport, stoichiometric Arrhenius networks, evaporation,
  shrinking-core conversion, reference Rosenbrock and structured tridiagonal
  solvers, morphology, fragmentation, radiation, and process operations.
- Added conservative particle-grid transfer, unresolved Stokes CFD–DEM,
  work-adjoint resolved immersed-boundary coupling, reactive continuum heat and
  species exchange, atomic Strang/iterated reactive CFD–DEM windows, generic
  hybrid-event sensitivity, replay/checkpointed VJPs, UQ, compositional support
  claims, executable examples, qualification campaigns, and performance evidence.
- Generalized fixed-step problems and solutions to mixed-dtype array PyTrees
  while preserving the existing array-valued SSPRK contract.
- Added all-coordinate tensor spectral PDE residual compilation with explicit
  full-closure versus retained-projection semantics, polynomial closure
  dealiasing, exactness and resource evidence, physical quadrature norms,
  external hard-condition contracts, and targetless operator fitting through
  `SpectralPDEResidualLoss`.
- Added native high-order quadrilateral/hexahedral finite elements with explicit
  nodal representations, GLL reference actions, dense and sum-factorized
  workset kernels, mapped cell/facet metrics, high-order CG/DG routing,
  mass-policy-aware rates, tensor SBP and periodic mapped DGSEM conservation,
  p-multigrid, Schwarz/FDM and auxiliary preconditioning, two-sided mortars,
  fixed-capacity hp transactions, and backend-neutral distributed ownership.
- Added operational nonconforming tensor-hp epochs with stable refinement forests,
  isotropic quad/hex h-refinement and coarsening, 2:1 closure, anisotropic p
  buckets, curved parent-map inheritance, H1 master-trace constraints, asymmetric
  DG mortar worksets, role-correct h/p transfers, atomic solver transactions,
  adaptive indicators and budgets, hp condensation/multigrid, inherited
  partition ownership, and certified entropy-compatible DGSEM mortars.
- Completed the single-host spectral-hp stack with native epoch compilation,
  anisotropic h and geometry-order adaptation, robust viscous/shock/ALE policies,
  tensor de Rham complexes, simplex/prism/pyramid references, nonlinear hp
  solvers, CAD/unfitted geometry, frozen-schedule adjoints, semantic caches,
  high-order output/import adapters, and complete public examples and guidance.
- Added implicit tensor-modal neural fields with Hermitian real-field projection,
  explicit modal input scaling and resource bounds, optional positive exponential
  decay and prepared-basis modulation, masked modal observations, and direct
  residual training against compiled coefficient-resident spectral dynamics.
- Added native low-rank adaptation for exact real `Linear` weight paths,
  factor-only `ParameterSubspace` training through `fit_operator` and Optax
  `FunctionalSolver`, safe scan fallback, explicit KFAC rejection, pure dense
  deployment merging, and checksum-validated adapter artifacts bound to the
  complete base model content and structure.
- Added deterministic fixed-step learned discrete systems with lazy
  mask/reset/control-aware trajectory windows, supervised, reference-branch,
  and residual rollout objectives, evidence-weighted gradient accumulation,
  exact update-boundary resume, and full/prefix/chunk causal equivalence.
- Added task-bound recurrent neural-operator training with one pipeline-safe
  physical state route, named future targets, supervised and residual rollout
  losses, route-aware deployed continuation, and instance-authoritative
  pointwise/finite/global/unknown dependency-support evidence.
- Added advanced computational topology: exact cellular and filtered maps,
  induced maps and cone audits, extended and temporal field topology, diagram
  features and certified matching, rational and integral class algebra,
  harmonic-period constraints, exact Morse cancellation, structured cubical
  analysis, local homology, certified implicit covers, and Conley homology
  index-pair workflows.
- Added `phydrax.pgm`: immutable finite-discrete factor graphs over native
  bipartite `GraphIR` topology; dense, sparse-enumerated, structured, and open
  capability-declared kernels; explicit precision/resource evidence; directed linear
  forest propagation; synchronous, Gauss--Seidel, accelerated, and qualified implicit
  loopy BP; same-topology and heterogeneous graph batches; bounded variable
  elimination, junction trees, normalized laws, smooth dual MAP bounds, and
  perturb-and-MAP estimates; systematic/random/block/tempered/qualified-cluster
  sampling with online reducers; persistent CD/SML, pseudolikelihood, Bethe, and exact
  EM objectives; and pickle-free graph/BP/Gibbs checkpoints. Added general PyTree
  conditional update programs under `phydrax.sampling.conditional` and factor-graph
  reverse denoising, adaptive mixing control, and hybrid embeddings under
  `phydrax.transport.discrete`.
- Added `phydrax.topology`: compact active subcomplexes and relative pairs,
  exact prime-field homology with cycle/cocycle representatives, exact rational
  Betti dimensions, explicit cell-vertex supports, lower/upper-star filtrations,
  ordinary and induced-relative persistent homology, natural and fixed-capacity
  diagrams, frozen-order endpoint derivatives, fail-closed resource evidence,
  and exact-nullity validation of metric cochain harmonic kernels.
- Added explicit bounded, periodic, half-line, and real-line spectral domains;
  endpoint-correct tensor measures; canonical modal transfers; rational
  Chebyshev line and half-line bases; linear trace constraints; exact periodic
  Hilbert transforms; physical modal-tail diagnostics; homogeneous
  cross-resolution eigen and eigenspace evidence; pairing-aware resolvent scans;
  and original-residual-certified polynomial eigenproblems.
- Added a native linear-combinatorial substrate with separate logical decisions
  and objective features, content-addressed plans, deterministic ties, portable
  statuses, and independent certificates. Added exact streamed finite,
  fixed-cardinality, primal-dual Hungarian assignment, and signed-cost DAG path
  oracles, plus explicit one-extra-solve blackbox surrogate pullbacks.
- Added a method-neutral structured nonlinear optimization spine with
  topology/numeric prepare-refresh lifecycles, exact sparse Jacobian and
  Lagrangian-Hessian reuse, portable primal/dual warm starts, independently
  certified native and Ipopt results, a unified dense/matrix-free/sparse
  `PrimalDualInteriorPoint`, truthful provider-backed KKT preparation, fixed-width
  root and structured-solve pools, generic PyTree/state-design/multiple-shooting
  compilers, fixed-active sensitivities and continuation, and optional
  Spineax/cuDSS sparse LDLT with numerical refactorization, reported inertia,
  and explicit resource release.
- Added an end-to-end free-boundary SciML substrate: differentiable compact
  Heaviside/delta and coarea calculus; level-set phase, normal, curvature,
  velocity, and Eikonal operators; discontinuity-aware coordinate lifts;
  implicit phase/interface functionals; Stefan, jump, kinematic,
  Gibbs--Thomson, Young--Laplace, and traction conditions; causal time slabs
  and narrow-band adaptive collocation.
- Added explicit-front, implicit-level-set, reference-map, and relaxed
  probabilistic Stefan workflows with common collocation, optimization, and
  representation comparison. Added free-boundary operator contracts,
  Jacobian/pullback/GCL evidence, VOF/PLIC and SPH adapters, and
  residual-controlled hybrid rollouts.
- Added interface predictive uncertainty, residual/diversity acquisition,
  bounded test-time context adaptation, phase geometry and masked interface
  distance evidence, plus exact Stefan, Mullins--Sekerka, topology-event,
  Hysing bubble, Turek--Hron FSI, obstacle, fracture, and
  trajectory-disjoint OOD benchmark contracts.
- Added certified variational eigenspaces across continuous, learned-operator,
  factorized high-dimensional, and discrete quantum workflows: basis-invariant
  block Rayleigh objectives, native reduced Ritz extraction and full-space
  residuals, learned trial-space warm starts, product-factor bilinear assembly
  without global tensor materialization, and mixture-sampled multi-state VMC
  with overlap/Hamiltonian evidence, span conditioning, Ritz modes, stochastic
  reconfiguration, and explicit failure statuses.
  Added a self-adjoint strong-form `InvariantSubspaceResidual` for neural trial
  fields, with projected reduced operators, basis-invariant residual Grams,
  generalized positive metrics, complex/vector pairings, continuous residual
  modes, absolute/relative residual evidence, and explicit rejection of
  collapsed, indefinite, or non-Hermitian trial systems.
- Added JAX-native direct collocation for explicit controlled systems and
  input-aware state-shaped DAEs, with fixed/nonuniform or optimized-duration
  meshes, backward-Euler and midpoint transcription, interval controls, shared
  optimized parameter spaces, bound-form path and trajectory constraints,
  physical scaling, exact sparse Jacobians and optional Lagrangian Hessians,
  explicitly selected native-dense or low-level sparse-Ipopt execution,
  independent KKT recertification, typed decisions/layouts/results, and
  non-certifying off-grid defect audits.
- Hardened direct collocation with canonical typed sparse-Ipopt evidence,
  callback/conversion counts, exact status mapping, topology-valid warm starts,
  optional `cyipopt` packaging, exact/limited-memory qualification artifacts,
  per-interval off-grid evidence, explicit nested h-refinement and primal
  transfer, controlled-DAE input policies and causal replay, and a fingerprinted
  eight-family graduation/regression campaign.
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
- Added experimental measure-aware particle-grid splatting with multilinear
  and degree-one through degree-three tensor B-spline assignments over nodal,
  cell, face, and edge layouts; extensive content and density outputs; weighted
  intensive reconstruction; adjoint grid gather; route gradients and moments;
  periodic and explicit reject/drop boundaries; piecewise or frozen geometry
  differentiation; static resource budgets; precision evidence; and
  fast/deterministic/compensated accumulation with independent balance,
  partition, reproduction, and gradient diagnostics.
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
- Added accepted linear-solve histories, exact FEM diagonal data, pairing-aware
  p-transfer roles and p-level planning, collocated tensor-product actions,
  staged DG traces, explicit quadrature evidence, one-ring patch and low-order
  auxiliary preconditioners, an incompressible pressure-correction workflow, and
  conservative multirate DG traces. The execution and preconditioning design is
  informed by libParanumal and its published high-order solver algorithms while
  remaining Phydrax-native.
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
  transverse wave propagation, hydrostatic wet/dry shallow water, multidimensional
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
- Replaced the legacy isotropic plane-stress MPM, mixed volumetric-constraint,
  contact workflow, sharp-fracture workflow, and compliance-only topology APIs
  with their explicit clean-cutover contracts. No deprecated aliases remain.
- Neo-Hookean field stress operators now name Lamé's first parameter `lambda_`
  instead of incorrectly describing the same coefficient as bulk modulus
  `kappa`; the old keyword is removed in one clean cutover.
- Neural-operator autoregression now requires a task-bound physical state route
  and the deployed normalization/constraint pipeline. The raw callable/advance
  rollout, standalone autoregressive loss, and teacher-forcing schedule were
  removed in one clean cutover.
- CNO and UNO now have periodic-Fourier semantics: circular measure-aware local
  convolution, endpoint-exclusive sine/cosine coordinate features, periodic
  uniform axes, and new semantic architecture identities. Nonperiodic and
  legacy artifact routes are rejected rather than reinterpreted.
- Benchmark tooling now shares one synchronized PyTree timing runtime, normalized
  software/hardware fingerprints, raw duration distributions, official XLA
  cost/memory evidence, atomic artifact writes, and environment-checked bootstrap
  comparisons. Operator reports separate lowering, compilation, first execution,
  and steady samples and no longer relabel process allocator high-water state as
  operation-local peak memory.
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
- Neo-Hookean finite-element forms now derive their residual from cell energy
  and support explicit two-dimensional plane strain as well as three-dimensional
  kinematics.
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
