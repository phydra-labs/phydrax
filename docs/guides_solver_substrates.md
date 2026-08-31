# Solver substrates

Phydrax separates finite support, numerical method, operator representation, and
solution algorithm. A tensor grid does not select finite differences or spectral
calculus, and an FFT direct solve does not change an FD operator into a spectral
method.

```python
import jax.numpy as jnp
import phydrax as phx
```

## Nonlinear update graphs

`phydrax.nonlinear` separates finite nonlinear work from complete root
certification. A prepared `AbstractNonlinearUpdate` graph may contain Newton or
Picard corrections, FAS cycles, additive/multiplicative Schwarz, or static
composition. An outer `NonlinearRichardson` or `NonlinearGMRES` method owns
termination and globalization. The returned `NonlinearResult` always
re-evaluates the original physical problem.

The lifecycle is:

```text
physical problem
  -> plan and prepare a fixed update graph
  -> apply bounded work and retain component evidence
  -> outer method accepts or rejects the proposal
  -> certify the original physical residual
```

`LaggedLinearSolveUpdate` is the state-dependent linear-model path:

```text
physical residual at state
  -> refresh a structure-preserving lagged operator
  -> solve one canonical prepared linear system
  -> propose a finite physical state
  -> globalize and certify the original residual
```

This targets quasilinear problems whose useful frozen-coefficient structure is
not a time-independent semilinear split. Eligible semilinear problems should
continue to use `SemilinearDrift`, ETDRK, SBDF2, or the corresponding split
method. The lagged operator may accelerate the primal solve or precondition an
exact derivative solve; it never replaces the exact root Jacobian in an implicit
JVP or adjoint.

Linear subspace correction and nonlinear Schwarz reuse explicit restriction and
prolongation ideas, but not one result type: nonlinear local work owns a local
problem, update status, domain validity, and physical reconstruction.

## Structured nonlinear optimization

Smooth fixed-topology constrained problems lower through one interoperable
lifecycle:

```text
physical domain problem
  -> compile one StructuredNonlinearProgram
  -> freeze argument, bound-role, and sparse derivative topology
  -> bind or refresh one numeric instance
  -> select a structured nonlinear method and KKT strategy
  -> execute scalar, full-batch, pooled, or continuation work
  -> construct a structured KKT certificate and portable warm start
  -> decode and independently audit the physical domain result
```

The domain compiler owns physical layouts, units, decoding, and audits.
`phydrax.sparse` owns derivative topology and coefficient execution.
`phydrax.optim` owns primal-dual, globalization, KKT, and certification
semantics. `phydrax.linalg` owns factorization providers and residual
certification. Execution pools own task placement only; lane identity never
becomes scientific identity.

`PrimalDualInteriorPoint` uses explicit dense-filter, matrix-free, or sparse
augmented modes. `IpoptMinimize` consumes the same structured program through
an external host boundary. Neither method is selected automatically and neither
is a fallback for the other.

## Solver graduation

New nonlinear methods remain internal until they have a derivation, capability
validation, accepted-point invariant, independent certificate, exact work,
failure taxonomy, JIT/batch behavior, focused tests, benchmark artifact, and
selection documentation.

`SolverGraduationEvidence` and `evaluate_solver_graduation` enforce zero false
successes, certified coverage, peer-profile, derivative, execution, and product
gates. `evaluate_solver_regression` rejects new false successes, excessive
coverage/profile loss, derivative degradation, hidden dense materialization,
numeric-refresh recompilation, or loss of complete work accounting. Wall-time
thresholds remain in controlled benchmark campaigns rather than ordinary unit
tests.

## Structured support

`TensorGridPlan.prepare(bounds)` returns `PreparedTensorGrid`: axes, topology,
embedding, measure, points, and exact grid locations without derivative methods.

```python
support = phx.discretization.TensorGridPlan(
    (phx.discretization.UniformAxisSpec(128),),
    axis_names=("x",),
).prepare(jnp.asarray([[0.0], [1.0]]))
```

`GridLocation` stores exact rational offsets and resolves to a concrete tensor entity
layout. Cell, vertex, and axis-normal face fields have independent shapes, coordinates,
measures, and boundary masks; bounded cell and face counts are intentionally different.

## Linear finite differences

A derivative request separates derivative order from accuracy order and declares
source/target locations and boundary realization:

```python
request = phx.discretization.DerivativeRequest(
    "dx",
    support,
    "x",
    derivative_order=1,
    accuracy_order=4,
)
fd = phx.discretization.FiniteDifferencePlan(support, (request,)).prepare()
u = fd.operator("dx").source.zeros()
du = fd.operator("dx").mv(u)
```

Coefficients use Fornberg recursion and carry moment residuals, conditioning,
footprint, interior order, and closure order. Bounded axes use explicit one-sided
closures; periodic wrap is never inferred on a bounded axis. Direct second
derivatives have independently prepared stencils.

`PreparedStencilOperator` exposes coordinate transpose and pairing adjoint. The
aggregate `HaloPlan` distinguishes physical, same-level, coarse/fine, and distributed
neighbor data.

## Generic patch kernels

`PatchKernelPlan` generalizes local nonlinear kernels without Kernex-style full index
matrices. `kernel_indices` can dispatch different prepared kernels by output region.
`OrderedPatchKernelPlan` exposes causal `lax.scan` semantics for dependence-ordered
sweeps.

## Stencil programs

`StencilProgramPlan` binds named fields and assignments. Preparation records reads,
writes, derivative reuse, footprint, and halo identity. `compile_stencil_dynamics`
adds stable field packing and complete discretization provenance.

## Fast transforms and spectra

`AbstractLinearTransform` declares physical/modal spaces and invertible analysis and
synthesis actions. `FFTLinearTransform`, `RealTrigonometricTransform`, and
`TensorLinearTransform` execute without dense transform storage; DCT/DST types I–IV
are JAX-native. `SimilarityScaledLinearTransform` handles the endpoint weighting
required by nodal Neumann closures.

`ModalTransform` remains the weighted dense basis contract for small or irregular
eigenbases. `OperatorSpectrum` separately owns one operator's modal values, nullspace,
groups, and approximation provenance. `SpectralDecomposition` pairs those dense
objects for convenience.

`TransformDiagonalRepresentation` accepts either a dense basis or an
`AbstractLinearTransform`. `TransformDiagonalSolvePlan` requires explicit
compatibility and gauge policy for singular systems; projecting an incompatible RHS
is never implicit.

`TensorSpectralPlan` composes these execution transforms with mode layouts, physical
measures, modal field spaces, precision, and resource evidence. Nonlinear
`PseudospectralMethodPlan` execution owns explicit mode-aware padding or filtering.
Chebyshev and Legendre construction reuses the internal polynomial rules, Vandermonde
evaluation, derivative recurrences, and canonical dense linalg factorization. Common
endpoint constraints, Galerkin solves, boundary lifts, and generalized tau block
systems also route through `phydrax.linalg`; no parallel polynomial or solve substrate
is introduced. Eligible diagonal semilinear systems integrate with `ETDRKMethod`.

`AxisDomain` makes bounded, periodic, half-line, and real-line support part of
the discretization identity. Rational Chebyshev transforms use endpoint-free
Fejér nodes and mapped physical weights; their derivative actions report
overresolved closure residuals rather than claiming finite-mode closure.
`SpectralModalTransferPlan` is the sole resolution-change authority used by
padding and eigen verification.

General eigensolve convergence is local numerical evidence, not a resolution
certificate. `compare_general_eigen_resolutions` matches homogeneous finite and
infinite modes one-to-one. The spectral wrapper transfers modes into a common
field space and compares physical subspaces. `ResolventScanProblem` reuses one
pairing-canonical Schur form across declared shifts. `PolynomialEigenproblem`
uses block companion pencils but accepts modes only against the original
homogeneous operator-polynomial residual.

Periodic one-dimensional FD stencils expose a certified FFT representation via
`PreparedFiniteDifferenceDiscretization.transform_diagonalization`.
`diagonalize_fd_laplacian` additionally certifies tensor FD2 operators on uniform point
or cell axes for every Dirichlet/Neumann side pairing. It selects DCT/DST-I–IV,
eliminates nodal Dirichlet values, preserves Neumann nullspaces, and exposes affine
boundary forcing. `FDLaplacianSolvePlan` reuses the transform for scaled and shifted
direct solves.

## Boundaries and nullspaces

`PreparedFDBoundaryProgram` evaluates time-, parameter-, coordinate-, and state-dependent
targets once per stage. `CellGhostBoundary` supports arbitrary halo depth for periodic,
Dirichlet, coordinate-oriented Neumann, and Robin laws. `NodalBoundaryRuntime`
separates strong state constraints, coordinate derivatives, and time derivatives.
`CornerPolicy` distinguishes axis-separable stencils from explicitly requested
tensor-product corner fills. Conforming interface bindings preserve field and outward
flux jumps without inventing topology.

Singular direct and multigrid solves distinguish compatibility (`error` or
`project_rhs`) from gauge (`zero_mean` or Euclidean `minimum_norm`).

## Compatible electromagnetics

`CompatibleMaxwellPlan` evolves electric displacement `D`, magnetic flux `B`, and
charge on one exact cochain complex. Prepared constitutive maps derive `E` and `H`;
the primary state therefore remains valid for diagonal, anisotropic, conductive,
dispersive, nonlinear, gyrotropic, and active materials. Every prepared material
declares loss, passivity/activity, reversibility, frequency-domain support, and
auxiliary state. Unsupported combinations fail during preparation.

PEC/PMC traces, periodic quotient topology, unitary Bloch twists, passive impedance
boundaries, conforming jumps, and norm-compatible mortars remain explicit prepared
objects. Structured electromagnetic CPML owns one convolutional memory per derivative
axis, separately from material state and observers; both electric and magnetic forces
consume those directional memories. The physical magnetic flux is projected through
the prepared minimum-norm `phydrax.linalg` solve so the topological magnetic constraint
remains exact under anisotropic damping. The runtime reports electric and magnetic
constraints, source, boundary, material, and PML power, energy, and CFL evidence.

Specialized materials, boundaries, observers, modal analysis, and adjoints live under
`phydrax.solver.maxwell`; the solver root exposes only the four core Maxwell lifecycle
types. Observers stream native/weighted probes, synchronized energy, Poynting flux,
and DFT state without retaining full histories. Frequency and transverse mode solves
use the certified generalized self-adjoint eigen engine; isolated-cluster sensitivities
use basis-invariant spectral-projector derivatives. Checkpointed PyTree, two-run
DFT-field, frequency-domain, and reversible adjoints have distinct eligibility
contracts; reversible execution rejects PML, dispersion, conductivity, active media,
and other noninvertible state.

`phydrax.solver.maxwell.fourier_modal` is a separate frequency-domain substrate for
one- or two-dimensionally periodic layer stacks. It uses reciprocal-lattice harmonic
convolution, full-tensor finite-layer operators, eigendecomposition-free boundary
cascades, homogeneous ports, diffraction orders, named current planes, and
Brillouin-zone source integration. It reuses `phydrax.linalg` solves and spectral
precision but does not reinterpret cochain material arrays or replace compatible
Maxwell. See [Fourier-modal Maxwell](guides_fourier_modal_maxwell.md).

The same degree-safe calculus applies full tetrahedral Whitney Hodge matrices and their
inverse actions directly in codifferentials, energy pairings, frequency solves, and
unstructured time evolution; diagonal metadata is not substituted for those operators.
`PointCloudPlan` is separate: it prepares a fixed-capacity,
rank/condition-certified nodal polynomial calculus over `PointTopology`. Its
`DissipativePointDiffusion` supplies a negative-semidefinite factorization; no generic
conservation claim is inferred from local polynomial reproduction.

## Staggered acoustics

`StaggeredAcousticPlan` stores pressure on cells and each velocity component on its
normal faces. `StaggeredAcousticState` retains one directional pressure split per
axis; their sum is the observable pressure.

`SplitFieldPMLPlan` constructs independent polynomial cell/face damping profiles for
each axis. The prepared equations damp pressure split `i` and velocity `i` with the
same directional profile, which is the split-field acoustic PML system rather than an
isotropic sponge. Leapfrog updates integrate the damping factors analytically, and the
reference solver reports a multidimensional CFL bound, energy, sources, sensors, and
complete discretization provenance.

## Conservative and high-resolution methods

Conservative face operators are finite-volume owned. `FaceCoefficientPlan` declares
arithmetic, harmonic, upwind, or callable interpolation.
`ConservativeDiffusionPlan` realizes `div(A grad(u))` as cell-to-face flux followed by
face-to-cell divergence, including discontinuous scalar, diagonal, and full tensor
coefficients. `ConservativeAdvectionPlan` keeps advective, conservative, skew, and
energy-split forms distinct. Finite-difference PDE lowering reuses these prepared
operators when it recognizes conservative expression structure.

The structured finite-volume compiler combines cell-average geometry, physical systems,
boundary policies, reconstruction, and one conservative interface method.
`HighResolutionReconstructionPlan` provides WENO-Z, TENO, and MP5;
`CharacteristicReconstructionPlan` uses equation-owned eigensystems. Euler,
multispecies Euler, shallow water, and ideal MHD live under `phydrax.equations`.
Ordinary numerical fluxes, wave propagation, and positivity are compatibility-checked.
Bathymetric shallow water instead uses `ShallowWaterHydrostaticHLLPlan`, which keeps
shared transport, one-sided bed corrections, and SSPRK-stage positivity indivisible.
See [Structured finite volume](guides_finite_volume.md) and
[Shallow water](guides_shallow_water.md).

## SBP-SAT, mapped grids, and multiblock coupling

`SBPDerivativePlan` prepares periodic or bounded diagonal-norm orders 2, 4, 6,
and 8 with compatible second derivatives and algebraically checked
`H D + Dᵀ H = B`. Periodic operators have `B = 0` and a skew norm derivative.
`SATBoundaryPlan` and `SATInterfacePlan` separate bounded boundary residuals from
penalties and carry explicit energy evidence.

`CompactDerivativePlan` and `CompactInterpolationPlan` prepare periodic implicit
line actions `A q = B u` without dense matrices. The cyclic target operator is a
tridiagonal base plus rank-two correction and reuses `phydrax.linalg` Woodbury
planning for multiple right-hand sides. Fourth/sixth-order first/second
derivatives and staggered interpolation retain exact source/target locations,
transpose and pairing-adjoint actions, modified-symbol evidence, and resource
counts. The acted-on line axis is nonlocal and must remain unsharded.

`TensorSBPPlan` combines periodic tensor axes with
`SBPFluxDifferencingMethodPlan`. The method evaluates one symmetric two-point
flux per nonzero SBP pair and scatters equal/opposite contributions; it does not
allocate a dense all-pairs state tensor. Initial compilation supports periodic
inviscid Euler. Bounded SAT fluxes and viscous entropy production remain separate
future contracts.

The compact, SBP conservation, and MAC projection qualification campaign is:

```bash
python tools/structured_flow_benchmarks.py \
  --output benchmarks/structured_flow.json
```


`MappedTensorGridPlan` prepares one-, two-, or three-dimensional stationary mappings.
Discrete curl metrics enforce metric identities and free-stream preservation;
physical gradient, divergence, diffusion, quadrature, normals, and dual-face measures
share one Jacobian/cofactor state. `evaluate_mapped_metrics` is the pure
differentiable metric kernel for fixed topology.

`MultiblockGridPlan` validates physical trace coincidence, tangential permutations,
reflections, conforming interfaces, and nested 2:1 traces.
`NormCompatibleInterpolationPlan` builds local polynomial prolongation and norm-adjoint
restriction. `MultiblockSATCoupling` uses these transfers for energy-conserving central
or dissipative upwind scalar-advection coupling.

## Polygonal virtual elements

The virtual-element substrate separates polygon topology, functional field
coordinates, computable polynomial projections, form consistency,
projector-kernel stabilization, and the downstream algebraic solver. Enhanced
conforming H1 spaces of qualified degree one through three expose full L2 and
energy projectors without presenting virtual interior basis values.

Matrix-free execution retains factorized projection actions; sparse execution
materializes the same local tensors into canonical coordinate storage.
Constraints, nullspaces, DAEs, eigenproblems, precision, geometry refresh, and
provenance reuse their ordinary Phydrax contracts. See
[Virtual elements](guides_virtual_elements.md).

## AMR and multigrid

The AMR substrate remains fixed-capacity and masks inactive payload before arithmetic.
`FDAMRHaloPlan` fills multidimensional same-level and parent-derived fine halos.
`AMREntityTransferPlan` declares point/interval axes, covering cells, nodes, faces, and
edges. `ConservativeAMRSubcyclingPlan` accumulates time-integrated fine/coarse fluxes before
reflux. `FDRegridPlan` records deterministic child activation/population and explicit
overflow; `AMRMigrationPlan` moves active slots without exposing inactive NaN/Inf.

`StructuredMultigridPlan` rediscretizes conservative diffusion on coarser tensor grids,
uses conservative cell or nested nodal transfers, selectable damped Jacobi,
red-black Gauss-Seidel, or bounded-axis line smoothing, and a nullspace-aware dense
coarse pseudoinverse. V/W/F/full cycle semantics come from the generic
`phydrax.linalg` hierarchy. Transfer storage and fields follow the bound
`FDExecutionPrecisionPolicy`; compatibility, gauge, and residual-norm decisions
remain in its certification precision.

## Temporal backend state representation

`DifferentialProblem` owns the public state dtype and shape independently of the
temporal backend representation. Standard Diffrax solves preserve real states
natively. Complex states default to `DiffraxComplexStatePolicy(\"real_imag\")`, which
prepares a real backend state with shape `(2,) + state_shape`, componentwise real
adaptive tolerance geometry, and explicit packing evidence.

Vector fields, stochastic coefficients, complex argument leaves, events, dense
interpolation, and saved trajectories cross this boundary through one prepared
adapter. Diffusion control axes remain trailing, so both real components contract
against the same declared Wiener controls. Nontrivial state geometry and the separate
delay/jump Diffrax backends are not assigned an inferred packing contract.

## Learned field manifolds and characteristic flows

`NeuralGalerkinProblem` lowers selected model leaves to one array-valued parameter
ODE. Each vector-field call rebuilds the enforced named fields, evaluates the
physical rate on fixed integration realizations, and solves a weighted tangent
projection with the ordinary linear runtime. Diffrax remains the sole temporal
backend. Saved-node audits report the tangent status and physical projection defect
without presenting hidden-stage work as measured evidence.

`trace_characteristics` converts backward physical time to increasing pseudo-time
and reuses `DifferentialProblem` plus `solve_diffrax`. The optional
`solve_characteristic_projection` layer is deliberately macro-step orchestration:
Diffrax computes feet, while `FunctionalSolver` fits fixed pulled-back targets.

## Execution, distribution, and production lifecycle

`StencilExecutionPlan` lowers regular interiors to one offset/weight kernel and retains
only irregular closure rows. `PreparedStencilProgram` fuses these kernels under JAX
and performs derivative common-subexpression elimination. The canonical masked bank
remains the scientific representation and transpose/adjoint authority.

`DistributedHaloSchedule` owns multi-axis `Mesh`, `NamedSharding`, collective
`ppermute` routes, face/edge/corner descriptors, explicit physical slots, and
deterministic reference exchange. It does not expose `pmap` as a generic distribution
model.

`FDExecutionPreflightPlan` accounts for state, halo, compact-stencil metadata,
temporaries, fixed-capacity AMR, and checkpoint copies under an explicit
`FDExecutionPrecisionPolicy` and memory budget. Its resource-assumption identity
and the executable policy identity are both retained in the estimate.

`FDCheckpointPlan` writes checksum-validated pickle-free fields and auxiliary PML,
AMR, partition, and integrator state. `FDActionAdjointPlan` transposes fixed-topology
boundary, halo, transfer, and stencil actions; `CheckpointedFDAdjointPlan` differentiates
the complete time-discrete scan.

`StructuredCochainBridge` assembles tensor entities into one oriented cubical complex.
Its incidence matrices satisfy boundary-of-boundary exactly and drive compatible
Maxwell, ideal-MHD induction, elasticity, variable-density incompressible projection,
poroelastic, and thermoelastic references.

## Collocation and constrained direct solves

`ChebyshevCollocation` is a separate global collocation method with a strict dense
dimension budget. `LowRankBoundaryCorrectionPlan` prepares a Green/capacitance Schur
correction with an explicit construction-byte limit. Neither is represented as local
finite difference.

Control direct collocation compiles local explicit/DAE defects into both a guarded
dense-native nonlinear program and an exact structured sparse program. The sparse Ipopt
boundary canonicalizes Jacobian coordinates, supplies one lower-triangular Hessian
representative, records complete callback work, and independently reconstructs KKT
evidence. Per-interval off-grid residuals drive nested h-refinement with explicit primal
transfer and no topology-changing dual transfer. Controlled-DAE replay binds a
`HeldInputPolicy` into initialization, every implicit stage, residual certification, and
continuation identity.

The deterministic qualification campaigns are:

```bash
python tools/run_direct_collocation_qualification.py
python tools/direct_collocation_ipopt_qualification.py --intervals 16 64
```

They write fingerprinted artifacts under `benchmarks/` without promoting sampled
off-grid evidence to a continuous-time certificate.
