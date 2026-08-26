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

Linear subspace correction and nonlinear Schwarz reuse explicit restriction and
prolongation ideas, but not one result type: nonlinear local work owns a local
problem, update status, domain validity, and physical reconstruction.

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
Rusanov, HLL, HLLC, Roe, entropy fluxes, wave propagation, positivity, and
multidimensional shared-face divergence remain independently selectable and
compatibility-checked. See [Structured finite volume](guides_finite_volume.md).

## SBP-SAT, mapped grids, and multiblock coupling

`SBPDerivativePlan` prepares diagonal-norm orders 2, 4, 6, and 8 with compatible
second derivatives and algebraically checked `H D + Dᵀ H = B`. `SATBoundaryPlan` and
`SATInterfacePlan` separate boundary residuals from penalties and carry explicit
energy evidence. `CompactFirstDerivative` remains the budgeted fourth-order periodic
cyclic line solve.

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
